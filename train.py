import argparse
import copy
import os
import random
import time
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch.utils.benchmark import Timer
import yaml

from src.nn.loss import (
    BinaryDiceLoss,
    CrossEntropyLoss,
    TrainEgoPathRegressionLoss,
)
from src.nn.model import ClassificationNet, RegressionNet, SegmentationNet
from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset
from src.utils.dataset import PathsDataset, get_labeled_image_names
from src.utils.evaluate import IoUEvaluator, evaluate_regression_model_iou
from src.utils.evaluate import collect_regression_eval_records
from src.utils.evaluate import grid_search_regression_postprocess
from src.utils.evaluate import regression_prediction_to_mask
from src.utils.postprocessing import average_regression_flip_predictions
from src.utils.trainer import train

try:
    import wandb
except ImportError:
    wandb = None


BACKBONES = (
    [f"resnet{x}" for x in [18, 34, 50]]
    + [f"efficientnet-b{x}" for x in [0, 1, 2, 3]]
    + ["dinov3-vits16", "dinov3-vits16plus", "dinov3-vitb16"]
    + [f"rtdetrv4-{x}" for x in ["s", "m", "l", "x"]]
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ego-Path Detection Training Script")
    parser.add_argument(
        "method",
        type=str,
        choices=["regression", "classification", "segmentation"],
        help="Method to use for the prediction head.",
    )
    parser.add_argument(
        "backbone",
        type=str,
        choices=BACKBONES,
        help="Backbone to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"]
        + [f"cuda:{x}" for x in range(torch.cuda.device_count())],
        help="Device to use.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--pool-channels", type=int, default=None)
    parser.add_argument("--fc-hidden-size", type=int, default=None)
    parser.add_argument("--dinov3-adapter-channels", type=int, default=None)
    parser.add_argument("--dinov3-adapter-depth", type=int, default=None)
    parser.add_argument("--rtdetrv4-adapter-channels", type=int, default=None)
    parser.add_argument("--rtdetrv4-adapter-depth", type=int, default=None)
    parser.add_argument("--rtdetrv4-feature-level", type=int, default=None)
    parser.add_argument("--rtdetrv4-no-encoder", action="store_true")
    parser.add_argument("--weights-subdir", type=str, default=None)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", type=str, choices=["one_cycle", "none"], default=None)
    parser.add_argument("--val-iou-interval", type=int, default=None)
    parser.add_argument("--val-iou-iterations", type=int, default=None)
    parser.add_argument("--test-iterations", type=int, default=None)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--calibrate-postprocess", action="store_true")
    parser.add_argument("--calibration-iterations", type=int, default=None)
    parser.add_argument("--inference-tta-flip", action="store_true")
    parser.add_argument("--benchmark-latency", action="store_true")
    parser.add_argument("--benchmark-tensorrt", action="store_true")
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument("--trt-force-rebuild", action="store_true")
    parser.add_argument(
        "--trt-precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp32",
    )
    parser.add_argument(
        "--dinov3-layer-set",
        type=str,
        choices=["last", "four_last", "four_even"],
        default=None,
    )
    parser.add_argument("--dinov3-use-cls-token", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def override_config(config, args):
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "workers": args.workers,
        "pool_channels": args.pool_channels,
        "fc_hidden_size": args.fc_hidden_size,
        "dinov3_adapter_channels": args.dinov3_adapter_channels,
        "dinov3_adapter_depth": args.dinov3_adapter_depth,
        "rtdetrv4_adapter_channels": args.rtdetrv4_adapter_channels,
        "rtdetrv4_adapter_depth": args.rtdetrv4_adapter_depth,
        "rtdetrv4_feature_level": args.rtdetrv4_feature_level,
        "weights_subdir": args.weights_subdir,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "val_iou_interval": args.val_iou_interval,
        "val_iou_iterations": args.val_iou_iterations,
        "test_iterations": args.test_iterations,
        "limit_train_samples": args.limit_train_samples,
        "limit_val_samples": args.limit_val_samples,
        "limit_test_samples": args.limit_test_samples,
        "calibration_iterations": args.calibration_iterations,
        "dinov3_layer_set": args.dinov3_layer_set,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.input_size is not None:
        config["input_shape"] = [3, args.input_size, args.input_size]
    if args.no_compile:
        config["compile"] = False
    if args.resume_from is not None:
        config["resume_from"] = args.resume_from
    if args.dinov3_use_cls_token:
        config["dinov3_use_cls_token"] = True
    if args.rtdetrv4_no_encoder:
        config["rtdetrv4_use_encoder"] = False
    if args.inference_tta_flip:
        config["inference_tta_flip"] = True
    config["wandb"] = args.wandb or config.get("wandb", False)
    return config


def make_dataloader(dataset, config, device, shuffle):
    workers = config["workers"]
    kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": shuffle and config.get("drop_last_train", False),
        "worker_init_fn": set_worker_seeds,
        "generator": torch.Generator().manual_seed(config["seed"]),
    }
    if workers > 0:
        kwargs["persistent_workers"] = config.get("persistent_workers", True)
        kwargs["prefetch_factor"] = config.get("prefetch_factor", 4)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def build_model(config):
    method = config["method"]
    backbone = config["backbone"]
    if method == "regression":
        return RegressionNet(
            backbone=backbone,
            input_shape=tuple(config["input_shape"]),
            anchors=config["anchors"],
            pool_channels=config["pool_channels"],
            fc_hidden_size=config["fc_hidden_size"],
            pretrained=config["pretrained"],
            dinov3_repo_dir=config.get("dinov3_repo_dir", "external/dinov3"),
            dinov3_weights_dir=config.get("dinov3_weights_dir", "dinov3_models"),
            dinov3_intermediate_layers=config.get("dinov3_intermediate_layers", 4),
            dinov3_adapter_channels=config.get("dinov3_adapter_channels", 256),
            dinov3_adapter_depth=config.get("dinov3_adapter_depth", 1),
            dinov3_layer_set=config.get("dinov3_layer_set", "four_last"),
            dinov3_use_cls_token=config.get("dinov3_use_cls_token", False),
            rtdetrv4_repo_dir=config.get(
                "rtdetrv4_repo_dir", "external/RT-DETRv4"
            ),
            rtdetrv4_weights_dir=config.get(
                "rtdetrv4_weights_dir", "rtdetrv4_models"
            ),
            rtdetrv4_use_encoder=config.get("rtdetrv4_use_encoder", True),
            rtdetrv4_feature_level=config.get("rtdetrv4_feature_level", 1),
            rtdetrv4_adapter_channels=config.get(
                "rtdetrv4_adapter_channels", 256
            ),
            rtdetrv4_adapter_depth=config.get("rtdetrv4_adapter_depth", 1),
        )
    if method == "classification":
        return ClassificationNet(
            backbone=backbone,
            input_shape=tuple(config["input_shape"]),
            anchors=config["anchors"],
            classes=config["classes"],
            pool_channels=config["pool_channels"],
            fc_hidden_size=config["fc_hidden_size"],
            pretrained=config["pretrained"],
        )
    if method == "segmentation":
        return SegmentationNet(
            backbone=backbone,
            decoder_channels=tuple(config["decoder_channels"]),
            pretrained=config["pretrained"],
        )
    raise ValueError


def build_criterion(config, train_dataset, logger):
    method = config["method"]
    if method == "regression":
        return TrainEgoPathRegressionLoss(
            ylimit_loss_weight=config["ylimit_loss_weight"],
            perspective_weight_limit=train_dataset.get_perspective_weight_limit(
                percentile=config["perspective_weight_limit_percentile"],
                logger=logger,
            )
            if config["perspective_weight_limit_percentile"] is not None
            else None,
        )
    if method == "classification":
        return CrossEntropyLoss()
    if method == "segmentation":
        return BinaryDiceLoss()
    raise ValueError


def get_amp_dtype(config):
    if not config.get("use_amp", False):
        return None
    dtype_name = config.get("amp_dtype", "bfloat16")
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {dtype_name}")


def load_resume_weights(model, resume_from, device, logger):
    checkpoint_path = resume_from
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded resume checkpoint: {checkpoint_path}")


def maybe_compile_model(model, config, device, logger):
    if not config.get("compile", False):
        return model
    if device.type != "cuda" or not hasattr(torch, "compile"):
        logger.info("torch.compile disabled for this device/runtime.")
        return model
    try:
        import importlib

        dynamo = importlib.import_module("torch._dynamo")
        dynamo.config.suppress_errors = True
        logger.info(
            f"Compiling model with torch.compile mode={config['compile_mode']}..."
        )
        return torch.compile(
            model,
            mode=config["compile_mode"],
            fullgraph=False,
        )
    except Exception as exc:
        logger.warning(f"torch.compile unavailable, continuing eager: {exc}")
        return model


def run_postprocess_calibration(
    model,
    config,
    val_indices,
    test_indices,
    save_path,
    device,
    logger,
    amp_dtype,
):
    if config["method"] != "regression":
        raise ValueError("Postprocess calibration is only supported for regression.")
    if len(val_indices) == 0:
        raise ValueError("Postprocess calibration requires a validation split.")

    val_dataset = PathsDataset(
        imgs_path=config["images_path"],
        annotations_path=config["annotations_path"],
        indices=val_indices,
        config=config,
        method="segmentation",
    )
    iterations = config.get("calibration_iterations", 3)
    logger.info(
        f"Collecting validation predictions for postprocess calibration "
        f"({iterations} iteration(s))..."
    )
    records = collect_regression_eval_records(
        model=model,
        dataset=val_dataset,
        config=config,
        device=device,
        amp_dtype=amp_dtype,
        iterations=iterations,
    )
    best = grid_search_regression_postprocess(records, config, logger=logger)
    config["postprocess_ylimit_offset"] = best["ylimit_offset"]
    config["postprocess_rail_width_scale"] = best["rail_width_scale"]
    config["postprocess_center_offset"] = best["center_offset"]
    config["calibration_val_iou"] = best["iou"]

    state_model = getattr(model, "_orig_mod", model)
    torch.save(state_model.state_dict(), os.path.join(save_path, "best.pt"))
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    test_iou = None
    if len(test_indices) > 0:
        logger.info("\nEvaluating calibrated model on test set...")
        test_dataset = PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=test_indices,
            config=config,
            method="segmentation",
        )
        test_iou = evaluate_regression_model_iou(
            model=model,
            dataset=test_dataset,
            config=config,
            device=device,
            amp_dtype=amp_dtype,
            iterations=config.get("test_iterations", 10),
        )
        logger.info(f"Calibrated test IoU: {test_iou:.5f}")

    metrics = {
        "best_val_iou": best["iou"],
        "test_iou": test_iou,
        "run_name": config["run_name"],
        "save_path": save_path,
        "postprocess_ylimit_offset": best["ylimit_offset"],
        "postprocess_rail_width_scale": best["rail_width_scale"],
        "postprocess_center_offset": best["center_offset"],
    }
    with open(os.path.join(save_path, "metrics.yaml"), "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    logger.info(f"Saved calibrated run to {save_path}")


def summarize_latency(times):
    values = sorted(t * 1000 for t in times)
    count = len(values)

    def percentile(q):
        index = min(round((count - 1) * q), count - 1)
        return values[index]

    mean_ms = sum(values) / count
    return {
        "runs": count,
        "mean_ms": mean_ms,
        "median_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "fps": 1000.0 / mean_ms,
    }


def benchmark_latency(
    model,
    config,
    image_names,
    test_indices,
    save_path,
    device,
    logger,
    amp_dtype,
    warmup,
    runs,
    benchmark_tensorrt=False,
    trt_force_rebuild=False,
    trt_precision="fp32",
):
    from PIL import Image, ImageOps
    from src.utils.common import image_to_model_tensor

    model.eval()
    autocast_enabled = amp_dtype is not None and device.type == "cuda"

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def time_function(fn):
        with torch.inference_mode():
            for _ in range(warmup):
                fn()
            synchronize()
            times = []
            for _ in range(runs):
                synchronize()
                start = time.perf_counter()
                fn()
                synchronize()
                times.append(time.perf_counter() - start)
        return summarize_latency(times)

    dummy_input = torch.randn((1, *config["input_shape"]), device=device)

    repo_style_timer = Timer(
        stmt="model(dummy_input)",
        globals={"model": model, "dummy_input": dummy_input},
        num_threads=torch.get_num_threads(),
    )
    repo_style_result = repo_style_timer.timeit(runs)
    repo_style_latency = {
        "runtime": "pytorch",
        "precision": "fp32",
        "runs": runs,
        "mean_ms": repo_style_result.mean * 1000,
        "fps": 1.0 / repo_style_result.mean,
    }
    trt_latency = None
    if benchmark_tensorrt:
        trt_latency = benchmark_tensorrt_forward(
            model=model,
            config=config,
            save_path=save_path,
            device=device,
            logger=logger,
            runs=runs,
            warmup=warmup,
            force_rebuild=trt_force_rebuild,
            precision=trt_precision,
        )

    def raw_forward():
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            model(dummy_input)

    image_index = test_indices[0] if len(test_indices) > 0 else 0
    image_name = image_names[image_index]
    image = Image.open(os.path.join(config["images_path"], image_name)).convert("RGB")
    flipped_image = ImageOps.mirror(image)

    def detector_once():
        tensor = image_to_model_tensor(image, config).unsqueeze(0).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            pred = model(tensor)
            if config.get("inference_tta_flip", False):
                flipped_tensor = (
                    image_to_model_tensor(flipped_image, config).unsqueeze(0).to(device)
                )
                flipped_pred = model(flipped_tensor)
                pred = average_regression_flip_predictions(
                    pred.float().cpu().numpy(),
                    flipped_pred.float().cpu().numpy(),
                    config["anchors"],
                )
        if torch.is_tensor(pred):
            pred = pred.float().cpu().numpy()
        regression_prediction_to_mask(pred, image.size, config)

    latency = {
        "device": str(device),
        "input_shape": config["input_shape"],
        "image_name": image_name,
        "image_size": list(image.size),
        "amp_dtype": config.get("amp_dtype") if autocast_enabled else None,
        "inference_tta_flip": config.get("inference_tta_flip", False),
        "repo_style_pytorch_forward": repo_style_latency,
        "repo_style_tensorrt_forward": trt_latency,
        "raw_forward": time_function(raw_forward),
        "detector_end_to_end": time_function(detector_once),
    }
    latency_path = os.path.join(save_path, "latency.yaml")
    with open(latency_path, "w") as f:
        yaml.safe_dump(latency, f, sort_keys=False)
    logger.info(
        "Repo-style PyTorch latency: %.2f ms",
        latency["repo_style_pytorch_forward"]["mean_ms"],
    )
    if trt_latency is not None:
        logger.info(
            "Repo-style TensorRT latency: %.2f ms",
            latency["repo_style_tensorrt_forward"]["mean_ms"],
        )
    logger.info(
        "Latency raw forward: %.3f ms mean, %.3f ms p95",
        latency["raw_forward"]["mean_ms"],
        latency["raw_forward"]["p95_ms"],
    )
    logger.info(
        "Latency detector end-to-end: %.3f ms mean, %.3f ms p95",
        latency["detector_end_to_end"]["mean_ms"],
        latency["detector_end_to_end"]["p95_ms"],
    )
    logger.info(f"Saved latency benchmark to {latency_path}")


def benchmark_tensorrt_forward(
    model,
    config,
    save_path,
    device,
    logger,
    runs,
    warmup,
    force_rebuild=False,
    precision="fp32",
):
    if device.type != "cuda":
        raise ValueError("TensorRT benchmark requires a CUDA device.")
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
    except ImportError as exc:
        raise ImportError(
            "TensorRT benchmark requires tensorrt and pycuda packages."
        ) from exc

    suffix = "" if precision == "fp32" else f".{precision}"
    onnx_path = os.path.join(save_path, f"best{suffix}.onnx")
    trt_path = os.path.join(save_path, f"best{suffix}.trt")
    dummy_input = torch.randn((1, *config["input_shape"]), device=device)

    if force_rebuild or not os.path.exists(onnx_path):
        logger.info(f"Exporting ONNX model to {onnx_path}...")
        export_model = model
        export_input = dummy_input
        if precision == "fp16":
            export_model = copy.deepcopy(model).half().to(device).eval()
            export_input = dummy_input.half()
        else:
            export_model.eval()
        with torch.inference_mode():
            torch.onnx.export(
                export_model,
                export_input,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )
        if export_model is not model:
            del export_model
            torch.cuda.empty_cache()

    trt_logger = trt.Logger(trt.Logger.WARNING)
    if force_rebuild or not os.path.exists(trt_path):
        logger.info(f"Building TensorRT engine to {trt_path}...")
        builder = trt.Builder(trt_logger)
        flag = (
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
            else 0
        )
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, trt_logger)
        with open(onnx_path, "rb") as f:
            parsed = parser.parse(f.read())
        if not parsed:
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")

        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            4 << 30,
        )
        serialized_engine = builder.build_serialized_network(network, builder_config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build failed.")
        with open(trt_path, "wb") as f:
            f.write(serialized_engine)

    runtime = trt.Runtime(trt_logger)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT engine: {trt_path}")
    context = engine.create_execution_context()

    cuda.init()
    device_index = device.index if device.index is not None else 0
    cuda_context = cuda.Device(device_index).retain_primary_context()
    cuda_context.push()
    try:
        input_shape = tuple(dummy_input.shape)
        input_name = None
        output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                output_name = name
        if input_name is None or output_name is None:
            raise RuntimeError("TensorRT engine does not have one input and one output.")

        context.set_input_shape(input_name, input_shape)
        input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
        input_array = np.random.random(input_shape).astype(input_dtype)
        input_device = cuda.mem_alloc(input_array.nbytes)
        output_shape = tuple(context.get_tensor_shape(output_name))
        output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
        output_device = cuda.mem_alloc(
            int(np.prod(output_shape)) * np.dtype(output_dtype).itemsize
        )
        context.set_tensor_address(input_name, int(input_device))
        context.set_tensor_address(output_name, int(output_device))

        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_device, input_array, stream)
        stream.synchronize()
        for _ in range(warmup):
            context.execute_async_v3(stream.handle)
        stream.synchronize()

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            context.execute_async_v3(stream.handle)
            stream.synchronize()
            times.append(time.perf_counter() - start)
    finally:
        cuda_context.pop()

    stats = summarize_latency(times)
    return {
        "runtime": "tensorrt",
        "precision": precision,
        **stats,
    }


def init_wandb(config, base_path):
    if not config.get("wandb", False) or wandb is None:
        return None
    return wandb.init(
        project="train-ego-path-detection",
        config=config,
        dir=base_path,
    )


def main(args):
    method = args.method
    device = torch.device(args.device)
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)

    with open(os.path.join(base_path, "configs", "global.yaml")) as f:
        global_config = yaml.safe_load(f)
    with open(os.path.join(base_path, "configs", f"{method}.yaml")) as f:
        method_config = yaml.safe_load(f)
    config = override_config(
        {
            **global_config,
            **method_config,
            "method": method,
            "backbone": args.backbone,
        },
        args,
    )
    if args.backbone.startswith("rtdetrv4"):
        if args.input_size is None:
            config["input_shape"] = [3, 640, 640]
        config["normalize_images"] = False
        config.setdefault("weights_subdir", "rt_detrv4_weights")

    if config.get("deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(config.get("matmul_precision", "high"))
        torch.backends.cuda.matmul.allow_tf32 = config.get("allow_tf32", True)
        torch.backends.cudnn.allow_tf32 = config.get("allow_tf32", True)
        torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", True)

    set_seeds(config["seed"])
    image_names = get_labeled_image_names(
        config["images_path"], config["annotations_path"]
    )
    indices = list(range(len(image_names)))
    random.shuffle(indices)
    proportions = (config["train_prop"], config["val_prop"], config["test_prop"])
    train_indices, val_indices, test_indices = split_dataset(indices, proportions)
    if config.get("limit_train_samples") is not None:
        train_indices = train_indices[: config["limit_train_samples"]]
    if config.get("limit_val_samples") is not None:
        val_indices = val_indices[: config["limit_val_samples"]]
    if config.get("limit_test_samples") is not None:
        test_indices = test_indices[: config["limit_test_samples"]]
    config["labeled_images"] = len(image_names)
    config["train_count"] = len(train_indices)
    config["val_count"] = len(val_indices)
    config["test_count"] = len(test_indices)
    set_seeds(config["seed"])

    train_dataset = PathsDataset(
        imgs_path=config["images_path"],
        annotations_path=config["annotations_path"],
        indices=train_indices,
        config=config,
        method=method,
        img_aug=True,
        to_tensor=True,
    )
    val_dataset = (
        PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=val_indices,
            config=config,
            method=method,
            img_aug=False,
            to_tensor=True,
        )
        if len(val_indices) > 0
        else None
    )
    train_loader = make_dataloader(train_dataset, config, device, shuffle=True)
    val_loader = (
        make_dataloader(val_dataset, config, device, shuffle=False)
        if val_dataset is not None
        else None
    )

    model = build_model(config).to(device)
    if args.resume_from is not None:
        load_resume_weights(model, args.resume_from, device, logger)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    config["trainable_params"] = trainable_params
    config["total_params"] = total_params
    logger.info(
        f"Model params: {trainable_params:,} trainable / {total_params:,} total"
    )

    run = init_wandb(config, base_path)
    run_name = args.save_run_name or args.run_name or (
        run.name
        if run is not None
        else f"{method}-{args.backbone}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    config["run_name"] = run_name
    weights_root = os.path.join(base_path, "weights")
    if config.get("weights_subdir"):
        weights_root = os.path.join(weights_root, config["weights_subdir"])
    save_path = os.path.join(weights_root, run_name)
    os.makedirs(save_path, exist_ok=True)
    if not args.benchmark_latency:
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    amp_dtype = get_amp_dtype(config)
    if args.benchmark_latency:
        benchmark_latency(
            model=model,
            config=config,
            image_names=image_names,
            test_indices=test_indices,
            save_path=save_path,
            device=device,
            logger=logger,
            amp_dtype=amp_dtype,
            warmup=args.latency_warmup,
            runs=args.latency_runs,
            benchmark_tensorrt=args.benchmark_tensorrt,
            trt_force_rebuild=args.trt_force_rebuild,
            trt_precision=args.trt_precision,
        )
        return
    if args.calibrate_postprocess:
        run_postprocess_calibration(
            model=model,
            config=config,
            val_indices=val_indices,
            test_indices=test_indices,
            save_path=save_path,
            device=device,
            logger=logger,
            amp_dtype=amp_dtype,
        )
        return

    criterion = build_criterion(config, train_dataset, logger)
    if method == "regression" and config["perspective_weight_limit_percentile"] is not None:
        set_seeds(config["seed"])

    optimizer_name = config.get("optimizer", "adamw")
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["learning_rate"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    total_steps = config["epochs"] * len(train_loader)
    scheduler = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config["learning_rate"],
            total_steps=total_steps,
            pct_start=config.get("warmup_pct", 0.03),
            anneal_strategy="cos",
            final_div_factor=config.get("final_div_factor", 1e4),
        )
        if config.get("scheduler") == "one_cycle"
        else None
    )

    train_model = maybe_compile_model(model, config, device, logger)

    val_iou_dataset = (
        PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=val_indices,
            config=config,
            method="segmentation",
        )
        if method == "regression" and len(val_indices) > 0
        else None
    )

    def val_iou_fn(current_model):
        if val_iou_dataset is None:
            return None
        return evaluate_regression_model_iou(
            model=current_model,
            dataset=val_iou_dataset,
            config=config,
            device=device,
            amp_dtype=amp_dtype,
            iterations=config.get("val_iou_iterations", 1),
        )

    logger.info(
        f"\nTraining {method} model for {config['epochs']} epochs "
        f"({total_steps} optimizer steps)..."
    )
    train_stats = train(
        epochs=config["epochs"],
        dataloaders=(train_loader, val_loader),
        model=train_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=save_path,
        device=device,
        logger=logger,
        val_iterations=config["val_iterations"],
        val_iou_fn=val_iou_fn if method == "regression" else None,
        val_iou_interval=config.get("val_iou_interval", 1),
        amp_dtype=amp_dtype,
        gradient_clip=config.get("gradient_clip"),
    )

    test_iou = None
    if len(test_indices) > 0:
        logger.info("\nEvaluating on test set...")
        test_dataset = PathsDataset(
            imgs_path=config["images_path"],
            annotations_path=config["annotations_path"],
            indices=test_indices,
            config=config,
            method="segmentation",
        )
        iou_evaluator = IoUEvaluator(
            dataset=test_dataset,
            model_path=save_path,
            runtime="pytorch",
            device=device,
        )
        test_iou = iou_evaluator.evaluate()
        logger.info(f"Test IoU: {test_iou:.5f}")
        if wandb is not None and wandb.run is not None:
            wandb.log({"test_iou": test_iou})

    metrics = {
        **train_stats,
        "test_iou": test_iou,
        "run_name": run_name,
        "save_path": save_path,
    }
    with open(os.path.join(save_path, "metrics.yaml"), "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    if wandb is not None and wandb.run is not None:
        wandb.finish()
    logger.info(f"Saved run to {save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
