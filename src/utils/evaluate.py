import timeit

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.benchmark import Timer

from src.utils.common import image_to_model_tensor, set_seeds
from src.utils.interface import Detector
from src.utils.postprocessing import (
    average_regression_flip_predictions,
    regression_to_rails,
    rails_to_mask,
    scale_rails,
)


def compute_iou(input, target):
    """Computes the Intersection over Union (IoU) between two binary masks.

    Args:
        input (numpy.ndarray or PIL.Image.Image): Input mask. Can be a numpy array (True/False, 0/1, 0/255) or a PIL image.
        target (numpy.ndarray or PIL.Image.Image): Ground truth mask. Can be a numpy array (True/False, 0/1, 0/255) or a PIL image.

    Returns:
        float: The IoU score.
    """
    input = np.array(input) if isinstance(input, Image.Image) else input
    target = np.array(target) if isinstance(target, Image.Image) else target
    input = input.astype(bool)
    target = target.astype(bool)
    if np.sum(target) == 0:  # if target is empty, we compute iou on negated masks
        input = np.logical_not(input)
        target = np.logical_not(target)
    intersection = np.logical_and(input, target)
    union = np.logical_or(input, target)
    return (np.sum(intersection) / np.sum(union)).item()


def get_regression_postprocess_kwargs(config):
    return {
        "ylimit_offset": config.get("postprocess_ylimit_offset", 0.0),
        "rail_width_scale": config.get("postprocess_rail_width_scale", 1.0),
        "center_offset": config.get("postprocess_center_offset", 0.0),
    }


def regression_prediction_to_mask(pred, img_size, config, postprocess_kwargs=None):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    traj = pred[:-1].reshape(2, config["anchors"])
    ylim = 1 / (1 + np.exp(-pred[-1].item()))
    kwargs = (
        get_regression_postprocess_kwargs(config)
        if postprocess_kwargs is None
        else postprocess_kwargs
    )
    rails = regression_to_rails(traj, ylim, **kwargs)
    rails = scale_rails(rails, None, img_size)
    rails = np.round(rails).astype(int).tolist()
    return rails_to_mask(rails, img_size)


def regression_record_to_iou(record, config, postprocess_kwargs):
    traj, ylim, target, img_size = record
    rails = regression_to_rails(traj, ylim, **postprocess_kwargs)
    rails = scale_rails(rails, None, img_size)
    rails = np.round(rails).astype(int).tolist()
    pred_mask = rails_to_mask(rails, img_size)
    return compute_iou(pred_mask, target)


def collect_regression_eval_records(
    model,
    dataset,
    config,
    device,
    amp_dtype=None,
    iterations=1,
):
    model.eval()
    records = []
    set_seeds(config["seed"])
    autocast_enabled = amp_dtype is not None and torch.device(device).type == "cuda"
    for _ in range(iterations):
        for i in range(len(dataset)):
            img, target = dataset[i]
            tensor = image_to_model_tensor(img, config).unsqueeze(0).to(device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=torch.device(device).type,
                    dtype=amp_dtype,
                    enabled=autocast_enabled,
                ):
                    pred = model(tensor)
                    if config.get("inference_tta_flip", False):
                        flipped = ImageOps.mirror(img)
                        flipped_tensor = (
                            image_to_model_tensor(flipped, config)
                            .unsqueeze(0)
                            .to(device)
                        )
                        flipped_pred = model(flipped_tensor)
                        pred = average_regression_flip_predictions(
                            pred.float().cpu().numpy(),
                            flipped_pred.float().cpu().numpy(),
                            config["anchors"],
                        )
            pred = pred.float().cpu().numpy().reshape(-1) if torch.is_tensor(pred) else pred.reshape(-1)
            traj = pred[:-1].reshape(2, config["anchors"])
            ylim = 1 / (1 + np.exp(-pred[-1].item()))
            records.append((traj.copy(), float(ylim), np.array(target).astype(bool), img.size))
    return records


def score_regression_records(records, config, postprocess_kwargs):
    ious = [
        regression_record_to_iou(record, config, postprocess_kwargs)
        for record in records
    ]
    return np.mean(ious).item()


def grid_search_regression_postprocess(records, config, logger=None):
    base_kwargs = get_regression_postprocess_kwargs(config)
    best = {
        "iou": score_regression_records(records, config, base_kwargs),
        **base_kwargs,
    }
    phases = [
        (
            np.arange(-0.05, 0.0501, 0.01),
            np.arange(0.94, 1.0601, 0.01),
            np.array([0.0]),
        ),
        (
            np.arange(-0.01, 0.0101, 0.0025),
            np.arange(-0.015, 0.0151, 0.005),
            np.array([0.0]),
        ),
    ]
    for phase_idx, (ylimit_offsets, width_offsets, center_offsets) in enumerate(phases):
        phase_best = best.copy()
        for ylimit_offset in ylimit_offsets:
            for width_offset in width_offsets:
                for center_offset in center_offsets:
                    kwargs = {
                        "ylimit_offset": float(
                            best["ylimit_offset"] + ylimit_offset
                            if phase_idx > 0
                            else ylimit_offset
                        ),
                        "rail_width_scale": float(
                            best["rail_width_scale"] + width_offset
                            if phase_idx > 0
                            else width_offset
                        ),
                        "center_offset": float(best["center_offset"] + center_offset),
                    }
                    iou = score_regression_records(records, config, kwargs)
                    if iou > phase_best["iou"]:
                        phase_best = {"iou": iou, **kwargs}
        best = phase_best
        if logger is not None:
            logger.info(
                "Calibration phase %d best: val_iou=%.5f, ylimit_offset=%.4f, "
                "rail_width_scale=%.4f, center_offset=%.4f",
                phase_idx + 1,
                best["iou"],
                best["ylimit_offset"],
                best["rail_width_scale"],
                best["center_offset"],
            )
    return best


class IoUEvaluator:
    def __init__(self, dataset, model_path, runtime, device):
        """Creates an IoU evaluator for the train ego-path detection model.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to evaluate on.
            model_path (str): Path to the trained model directory (containing config.yaml and best.pt).
            runtime (str): Runtime to use for model inference ("pytorch" or "tensorrt").
            device (str): Device to use for model inference ("cpu", "cuda", "cuda:x" or "mps").
        """
        self.dataset = dataset
        self.runtime = runtime
        if runtime == "pytorch":
            self.detector = Detector(model_path, None, runtime, device)
        elif runtime == "tensorrt":
            self.detector = Detector(model_path, None, runtime, device)
        else:
            raise ValueError

    def evaluate(self):
        set_seeds(self.detector.config["seed"])
        ious = []
        # each test epoch is unique due to data augmentation, so we average multiple runs to get a stable result
        for _ in range(self.detector.config["test_iterations"]):
            for i in range(len(self.dataset)):
                img, target = self.dataset[i]
                pred = self.detector.detect(img)
                if self.detector.config["method"] in ["classification", "regression"]:
                    pred = rails_to_mask(pred, img.size)
                ious.append(compute_iou(pred, target))
        return np.mean(ious).item()


def evaluate_regression_model_iou(
    model,
    dataset,
    config,
    device,
    amp_dtype=None,
    iterations=1,
):
    model.eval()
    ious = []
    set_seeds(config["seed"])
    autocast_enabled = amp_dtype is not None and torch.device(device).type == "cuda"
    for _ in range(iterations):
        for i in range(len(dataset)):
            img, target = dataset[i]
            tensor = image_to_model_tensor(img, config).unsqueeze(0).to(device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=torch.device(device).type,
                    dtype=amp_dtype,
                    enabled=autocast_enabled,
                ):
                    pred = model(tensor)
                    if config.get("inference_tta_flip", False):
                        flipped = ImageOps.mirror(img)
                        flipped_tensor = (
                            image_to_model_tensor(flipped, config)
                            .unsqueeze(0)
                            .to(device)
                        )
                        flipped_pred = model(flipped_tensor)
                        pred = average_regression_flip_predictions(
                            pred.float().cpu().numpy(),
                            flipped_pred.float().cpu().numpy(),
                            config["anchors"],
                        )
            pred = pred.float().cpu().numpy() if torch.is_tensor(pred) else pred
            pred_mask = regression_prediction_to_mask(pred, img.size, config)
            ious.append(compute_iou(pred_mask, target))
    return np.mean(ious).item()


class LatencyEvaluator:
    def __init__(self, model_path, runtime, device):
        """Creates a latency evaluator for the train ego-path detection model.

        Args:
            model_path (str): Path to the trained model directory (containing config.yaml and best.pt).
            runtime (str): Runtime environment to use for model inference ("pytorch" or "tensorrt").
            device (str): Device to use for model inference ("cpu", "cuda", "cuda:x" or "mps").
        """        
        self.runtime = runtime
        if runtime == "pytorch":
            self.detector = Detector(model_path, None, runtime, device)
        elif runtime == "tensorrt":
            self.detector = Detector(model_path, None, runtime, device)
        else:
            raise ValueError
        self.device = torch.device(device)

    def evaluate_pytorch(self, runs):
        dummy_input = torch.rand(
            (1, *self.detector.config["input_shape"]), device=self.device
        )
        for _ in range(runs // 10):  # warmup
            self.detector.model(dummy_input)
        timer = Timer(
            stmt="self.detector.model(dummy_input)",
            globals={"torch": torch, "self": self, "dummy_input": dummy_input},
            num_threads=torch.get_num_threads(),
        )
        result = timer.timeit(runs)
        return result.mean  # in seconds

    def evaluate_tensorrt(self, runs):
        for _ in range(runs // 10):  # warmup
            self.detector.exectx.execute_async_v3(
                self.detector.trt_stream.handle
            )
        self.detector.trt_stream.synchronize()
        timer = timeit.Timer(
            stmt=(
                "self.detector.exectx.execute_async_v3("
                "self.detector.trt_stream.handle); "
                "self.detector.trt_stream.synchronize()"
            ),
            globals={"self": self},
        )
        return timer.timeit(runs) / runs  # in seconds

    def evaluate(self, runs=1000):
        if self.runtime == "pytorch":
            return self.evaluate_pytorch(runs)
        elif self.runtime == "tensorrt":
            return self.evaluate_tensorrt(runs)
