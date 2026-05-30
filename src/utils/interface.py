import importlib
import os

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from ..nn.model import ClassificationNet, RegressionNet, SegmentationNet
from .autocrop import Autocropper
from .common import image_to_model_tensor
from .postprocessing import (
    average_regression_flip_predictions,
    classifications_to_rails,
    regression_to_rails,
    scale_mask,
    scale_rails,
)


class Detector:
    def __init__(self, model_path, crop_coords, runtime, device):
        """Interface to infer the train ego-path detection model using PyTorch or TensorRT.

        Args:
            model_path (str): Path to the trained model directory (containing config.yaml and best.pt)
            crop_coords (tuple or str or None): Coordinates to use for cropping the input image before inference:
                - If tuple, should be the inclusive absolute coordinates (xleft, ytop, xright, ybottom) of the fixed region.
                - If str, should be "auto" to use automatic cropping.
                - If None, no cropping is performed.
            runtime (str): Runtime to use for model inference ("pytorch" or "tensorrt").
            device (str): Device to use for model inference ("cpu", "cuda", "cuda:x" or "mps").
        """
        self.model_path = model_path
        self.runtime = runtime
        self.device = torch.device(device)
        with open(os.path.join(self.model_path, "config.yaml")) as f:
            self.config = yaml.safe_load(f)
        if isinstance(crop_coords, tuple) and len(crop_coords) == 4:
            self.crop_coords = crop_coords
        elif crop_coords == "auto":
            self.crop_coords = Autocropper(self.config)
        else:
            self.crop_coords = None

        if self.runtime == "pytorch":
            self.model = self.init_model_pytorch()
        elif self.runtime == "tensorrt":
            # lazy imports
            self.trt = importlib.import_module("tensorrt")
            self.cuda = importlib.import_module("pycuda.driver")
            os.environ["CUDA_MODULE_LOADING"] = "LAZY"
            # convert model to tensorrt if not already done
            fp16_engine_path = os.path.join(self.model_path, "best.fp16.trt")
            default_engine_path = os.path.join(self.model_path, "best.trt")
            self.trt_engine_path = (
                fp16_engine_path
                if os.path.exists(fp16_engine_path)
                else default_engine_path
            )
            if not os.path.exists(self.trt_engine_path):
                self.convert_to_tensorrt()
                self.trt_engine_path = default_engine_path
            # init cuda context on device
            self.cuda.init()
            device = 0 if device == "cuda" else int(device.split(":")[-1])
            self.ctx = self.cuda.Device(device).retain_primary_context()
            self.ctx.push()
            self.exectx, self.bindings, self.shapes = self.init_model_tensorrt()
        else:
            raise ValueError

    def __del__(self):
        if self.runtime == "tensorrt":
            self.ctx.pop()

    def get_crop_coords(self):
        return (
            self.crop_coords()
            if isinstance(self.crop_coords, Autocropper)
            else self.crop_coords
        )

    def init_model_pytorch(self):
        if self.config["method"] == "classification":
            model = ClassificationNet(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                classes=self.config["classes"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
            )
        elif self.config["method"] == "regression":
            model = RegressionNet(
                backbone=self.config["backbone"],
                input_shape=tuple(self.config["input_shape"]),
                anchors=self.config["anchors"],
                pool_channels=self.config["pool_channels"],
                fc_hidden_size=self.config["fc_hidden_size"],
                dinov3_repo_dir=self.config.get("dinov3_repo_dir", "external/dinov3"),
                dinov3_weights_dir=self.config.get(
                    "dinov3_weights_dir", "dinov3_models"
                ),
                dinov3_intermediate_layers=self.config.get(
                    "dinov3_intermediate_layers", 4
                ),
                dinov3_adapter_channels=self.config.get(
                    "dinov3_adapter_channels", 256
                ),
                dinov3_adapter_depth=self.config.get("dinov3_adapter_depth", 1),
                dinov3_layer_set=self.config.get("dinov3_layer_set", "four_last"),
                dinov3_use_cls_token=self.config.get(
                    "dinov3_use_cls_token", False
                ),
            )
        elif self.config["method"] == "segmentation":
            model = SegmentationNet(
                backbone=self.config["backbone"],
                decoder_channels=tuple(self.config["decoder_channels"]),
            )
        model.to(self.device).eval()
        model.load_state_dict(
            torch.load(
                os.path.join(self.model_path, "best.pt"), map_location=self.device
            )
        )
        return model

    def init_model_tensorrt(self):
        runtime = self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        with open(self.trt_engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        exectx = engine.create_execution_context()
        self.trt_engine = engine
        self.trt_stream = self.cuda.Stream()
        self.trt_input_name = None
        self.trt_output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self.trt_input_name = name
            elif mode == self.trt.TensorIOMode.OUTPUT:
                self.trt_output_name = name
        if self.trt_input_name is None or self.trt_output_name is None:
            raise RuntimeError("TensorRT engine must have one input and one output.")

        input_shape = (1, *self.config["input_shape"])
        exectx.set_input_shape(self.trt_input_name, input_shape)
        output_shape = tuple(exectx.get_tensor_shape(self.trt_output_name))
        input_dtype = self.trt.nptype(engine.get_tensor_dtype(self.trt_input_name))
        output_dtype = self.trt.nptype(engine.get_tensor_dtype(self.trt_output_name))
        bindings = {
            self.trt_input_name: self.cuda.mem_alloc(
                int(np.prod(input_shape)) * np.dtype(input_dtype).itemsize
            ),
            self.trt_output_name: self.cuda.mem_alloc(
                int(np.prod(output_shape)) * np.dtype(output_dtype).itemsize
            ),
        }
        exectx.set_tensor_address(self.trt_input_name, int(bindings[self.trt_input_name]))
        exectx.set_tensor_address(
            self.trt_output_name,
            int(bindings[self.trt_output_name]),
        )
        self.trt_output_shape = output_shape
        self.trt_output_dtype = output_dtype
        self.trt_input_dtype = input_dtype
        self.trt_input_shape = input_shape
        dummy_input = np.random.random(input_shape).astype(input_dtype)
        self.cuda.memcpy_htod_async(
            bindings[self.trt_input_name],
            dummy_input,
            self.trt_stream,
        )
        self.trt_stream.synchronize()
        shapes = (input_shape, output_shape)
        return exectx, bindings, shapes

    def convert_to_tensorrt(self, precision="fp16"):
        pytorch_model = self.init_model_pytorch()
        dummy_input = torch.rand((1, *self.config["input_shape"])).to(self.device)
        onnx_path = os.path.join(self.model_path, "best.onnx")
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
        trt_logger = self.trt.Logger(self.trt.Logger.ERROR)
        builder = self.trt.Builder(trt_logger)
        flag = (
            1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            if hasattr(self.trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
            else 0
        )
        network = builder.create_network(flag)
        config = builder.create_builder_config()
        parser = self.trt.OnnxParser(network, trt_logger)
        with open(onnx_path, "rb") as model:
            parsed = parser.parse(model.read())
        if not parsed:
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
        config.set_memory_pool_limit(self.trt.MemoryPoolType.WORKSPACE, 4 << 30)
        if precision == "fp16" and hasattr(self.trt.BuilderFlag, "FP16"):
            config.set_flag(self.trt.BuilderFlag.FP16)
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed.")
        with open(os.path.join(self.model_path, "best.trt"), "wb") as f:
            f.write(engine)

    def infer_model_pytorch(self, img):
        tensor = image_to_model_tensor(img, self.config).unsqueeze(0).to(self.device)
        amp_dtype = (
            getattr(torch, self.config.get("amp_dtype", "bfloat16"))
            if self.config.get("use_amp", False) and self.device.type == "cuda"
            else None
        )
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                pred = self.model(tensor)
        return pred.float().cpu().numpy()

    def infer_model_tensorrt(self, img):
        tensor = image_to_model_tensor(img, self.config).unsqueeze(0).contiguous()
        tensor = tensor.numpy().astype(self.trt_input_dtype)
        pred = np.empty(self.trt_output_shape, dtype=self.trt_output_dtype)
        self.cuda.memcpy_htod_async(
            self.bindings[self.trt_input_name],
            tensor,
            self.trt_stream,
        )
        self.exectx.execute_async_v3(self.trt_stream.handle)
        self.cuda.memcpy_dtoh_async(
            pred,
            self.bindings[self.trt_output_name],
            self.trt_stream,
        )
        self.trt_stream.synchronize()
        return pred

    def detect(self, img):
        """Detects the train ego-path on an image using the model.

        Args:
            img (PIL.Image.Image): Input image on which detection is to be performed.

        Returns:
            list or PIL.Image.Image: Train ego-path detection result, whose type depends on the method used:
                - Classification/Regression: List containing the left and right rails lists of rails point coordinates (x, y).
                - Segmentation: PIL.Image.Image representing the binary mask of detected region.
        """     
        original_shape = img.size
        crop_coords = self.get_crop_coords()
        if crop_coords is not None:
            xleft, ytop, xright, ybottom = crop_coords
            img = img.crop((xleft, ytop, xright + 1, ybottom + 1))

        if self.runtime == "pytorch":
            pred = self.infer_model_pytorch(img)
        elif self.runtime == "tensorrt":
            pred = self.infer_model_tensorrt(img)
        if (
            self.config["method"] == "regression"
            and self.config.get("inference_tta_flip", False)
        ):
            flipped_img = ImageOps.mirror(img)
            if self.runtime == "pytorch":
                flipped_pred = self.infer_model_pytorch(flipped_img)
            elif self.runtime == "tensorrt":
                flipped_pred = self.infer_model_tensorrt(flipped_img)
            pred = average_regression_flip_predictions(
                pred,
                flipped_pred,
                self.config["anchors"],
            )

        if self.config["method"] == "classification":
            clf = pred.reshape(2, self.config["anchors"], self.config["classes"] + 1)
            clf = np.argmax(clf, axis=2)
            rails = classifications_to_rails(clf, self.config["classes"])
            rails = scale_rails(rails, crop_coords, original_shape)
            rails = np.round(rails).astype(int)
            res = rails.tolist()
        elif self.config["method"] == "regression":
            traj = pred[:, :-1].reshape(2, self.config["anchors"])
            ylim = 1 / (1 + np.exp(-pred[:, -1].item()))  # sigmoid
            rails = regression_to_rails(
                traj,
                ylim,
                ylimit_offset=self.config.get("postprocess_ylimit_offset", 0.0),
                rail_width_scale=self.config.get("postprocess_rail_width_scale", 1.0),
                center_offset=self.config.get("postprocess_center_offset", 0.0),
            )
            rails = scale_rails(rails, crop_coords, original_shape)
            rails = np.round(rails).astype(int)
            res = rails.tolist()
        elif self.config["method"] == "segmentation":
            mask = pred.squeeze(0).squeeze(0)
            mask = (mask > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask)
            res = scale_mask(mask, crop_coords, original_shape)

        if isinstance(self.crop_coords, Autocropper):
            self.crop_coords.update(original_shape, res)

        return res
