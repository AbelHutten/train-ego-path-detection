import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, version, out_levels=(5,), pretrained=False):
        """Initializes the ResNet backbone.

        Args:
            version (str): Version of the ResNet backbone.
            out_levels (tuple): Which stage outputs to return. Defaults to (5,) (i.e. the last stage).
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        super(ResNetBackbone, self).__init__()
        model_versions = {
            "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }
        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]
        model = model_fn(weights=weights if pretrained else None)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(model.conv1, model.bn1, model.relu),
                nn.Sequential(model.maxpool, model.layer1),
                model.layer2,
                model.layer3,
                model.layer4,
            ]
        )
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features


class EfficientNetBackbone(nn.Module):
    def __init__(self, version, out_levels=(8,), pretrained=False):
        """Initializes the EfficientNet backbone.

        Args:
            version (str): Version of the EfficientNet backbone.
            out_levels (tuple): Which stage outputs to return. Defaults to (8,) (i.e. the last stage).
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        super(EfficientNetBackbone, self).__init__()
        model_versions = {
            "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        }
        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]
        # last block is discarded because it would be redundant with the pooling layer
        model = model_fn(weights=weights if pretrained else None).features[:-1]
        self.stages = nn.ModuleList([model[i] for i in range(len(model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        name,
        repo_dir="external/dinov3",
        weights_dir="dinov3_models",
        intermediate_layers=4,
        layer_set="four_last",
        use_cls_token=False,
    ):
        """Frozen DINOv3 ViT feature extractor.

        DINOv3 is loaded from the official local repository and checkpoint files.
        The module always stays frozen/eval, including when the parent model is
        switched to training mode.
        """
        super().__init__()
        specs = {
            "dinov3-vits16": (
                "dinov3_vits16",
                "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                384,
            ),
            "dinov3-vits16plus": (
                "dinov3_vits16plus",
                "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                384,
            ),
            "dinov3-vitb16": (
                "dinov3_vitb16",
                "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                768,
            ),
        }
        if name not in specs:
            raise NotImplementedError

        project_root = Path(__file__).resolve().parents[2]
        repo_path = Path(repo_dir)
        weights_path = Path(weights_dir)
        if not repo_path.is_absolute():
            repo_path = project_root / repo_path
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path

        hub_name, weights_file, embed_dim = specs[name]
        checkpoint_path = weights_path / weights_file
        if not repo_path.exists():
            raise FileNotFoundError(
                f"DINOv3 repository not found at {repo_path}. "
                "Clone facebookresearch/dinov3 into external/dinov3."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint_path}")

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        self.model = torch.hub.load(
            str(repo_path),
            hub_name,
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.use_cls_token = use_cls_token
        n_blocks = getattr(self.model, "n_blocks")
        if layer_set == "last":
            self.layer_indices = [n_blocks - 1]
        elif layer_set == "four_last":
            self.layer_indices = list(range(n_blocks - intermediate_layers, n_blocks))
        elif layer_set == "four_even":
            self.layer_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
        else:
            raise ValueError(f"Unsupported DINOv3 layer_set: {layer_set}")
        channel_multiplier = 2 if use_cls_token else 1
        self.out_channels = (embed_dim * len(self.layer_indices) * channel_multiplier,)
        self.reduction_factor = 16

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.model.get_intermediate_layers(
                x,
                n=self.layer_indices,
                reshape=True,
                return_class_token=self.use_cls_token,
                norm=True,
            )
        if self.use_cls_token:
            features = [
                torch.cat((patch, cls[:, :, None, None].expand_as(patch)), dim=1)
                for patch, cls in features
            ]
        return [torch.cat(features, dim=1)]


class RTDetrV4Backbone(nn.Module):
    def __init__(
        self,
        name,
        repo_dir="external/RT-DETRv4",
        weights_dir="rtdetrv4_models",
        use_encoder=True,
        feature_level=1,
        checkpoint_key="ema",
        deploy=True,
    ):
        """Frozen RT-DETRv4 HGNet/HybridEncoder feature extractor.

        The local RT-DETRv4 repo is used only to instantiate the official model
        graph. We load the local detector checkpoint, keep the detector feature
        stack frozen/eval, and return a single concatenated feature map for the
        ego-path regression head.
        """
        super().__init__()
        specs = {
            "rtdetrv4-s": (
                "configs/rtv4/rtv4_hgnetv2_s_coco.yml",
                "RTv4-S-hgnet.pth",
                True,
            ),
            "rtdetrv4-m": (
                "configs/rtv4/rtv4_hgnetv2_m_coco.yml",
                "RTv4-M-hgnet.pth",
                True,
            ),
            "rtdetrv4-l": (
                "configs/rtv4/rtv4_hgnetv2_l_coco.yml",
                "RTv4-L-hgnet.pth",
                False,
            ),
            "rtdetrv4-x": (
                "configs/rtv4/rtv4_hgnetv2_x_coco.yml",
                "RTv4-X-hgnet.pth",
                False,
            ),
        }
        if name not in specs:
            raise NotImplementedError

        project_root = Path(__file__).resolve().parents[2]
        repo_path = Path(repo_dir)
        weights_path = Path(weights_dir)
        if not repo_path.is_absolute():
            repo_path = project_root / repo_path
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        if not repo_path.exists():
            raise FileNotFoundError(
                f"RT-DETRv4 repository not found at {repo_path}. "
                "Clone RT-DETRs/RT-DETRv4 into external/RT-DETRv4."
            )

        config_file, weights_file, use_lab = specs[name]
        config_path = repo_path / config_file
        checkpoint_path = weights_path / weights_file
        if not config_path.exists():
            raise FileNotFoundError(f"RT-DETRv4 config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"RT-DETRv4 checkpoint not found: {checkpoint_path}"
            )

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from engine.core import YAMLConfig, yaml_utils

        yaml_utils.load_config.__defaults__ = ({},)
        cfg = YAMLConfig(str(config_path))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        cfg.yaml_cfg["HGNetv2"]["use_lab"] = use_lab
        detector = cfg.model
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                checkpoint_key,
                checkpoint.get("model", checkpoint),
            )
            if isinstance(state_dict, dict) and "module" in state_dict:
                state_dict = state_dict["module"]
        detector.load_state_dict(state_dict, strict=True)
        detector.eval()
        if deploy:
            detector.deploy()

        self.detector_backbone = detector.backbone
        self.detector_encoder = detector.encoder if use_encoder else None
        if self.detector_encoder is not None:
            # RT-DETRv4 stores eval positional embeddings as plain CPU tensor
            # attributes, not buffers. Rebuilding them on the feature device keeps
            # the frozen encoder portable across CPU/CUDA moves and input sizes.
            self.detector_encoder.eval_spatial_size = None
        self.use_encoder = use_encoder
        if use_encoder:
            feature_channels = list(detector.encoder.out_channels)
            feature_strides = list(detector.encoder.feat_strides)
        else:
            return_idx = list(detector.backbone.return_idx)
            feature_channels = [detector.backbone._out_channels[i] for i in return_idx]
            feature_strides = [detector.backbone._out_strides[i] for i in return_idx]
        if not 0 <= feature_level < len(feature_channels):
            raise ValueError(
                f"feature_level must be in [0, {len(feature_channels) - 1}], "
                f"got {feature_level}"
            )
        self.feature_level = feature_level
        self.out_channels = (sum(feature_channels),)
        self.reduction_factor = feature_strides[feature_level]

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode=True):
        super().train(False)
        self.detector_backbone.eval()
        if self.detector_encoder is not None:
            self.detector_encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.detector_backbone(x)
            if self.detector_encoder is not None:
                features = self.detector_encoder(features)
                if isinstance(features, tuple):
                    features = features[0]
            target_size = features[self.feature_level].shape[-2:]
            resized = [
                feature
                if feature.shape[-2:] == target_size
                else F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for feature in features
            ]
        return [torch.cat(resized, dim=1)]
