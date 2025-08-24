import torch
import torch.nn as nn
import torchvision.models as models
from dinov3.hub.backbones import (
    dinov3_convnext_base,
    dinov3_convnext_large,
    dinov3_convnext_small,
    dinov3_convnext_tiny,
    Weights,
)


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
    def __init__(self, version, out_levels: tuple[int, ...] | None = None, pretrained=False):
        """Initializes the EfficientNet backbone.

        Args:
            version (str): Version of the EfficientNet backbone.
            out_levels (tuple): Which stage outputs to return. Defaults to (8,) (i.e. the last stage).
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        if out_levels is None:
            if version in ["v2-s", "v2-m", "v2-l"]:
                out_levels = (7,)
            else:
                out_levels = (8,)
        super(EfficientNetBackbone, self).__init__()
        model_versions = {
            "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            "v2-s": (
                models.efficientnet_v2_s,
                models.EfficientNet_V2_S_Weights.DEFAULT,
            ),
            "v2-m": (
                models.efficientnet_v2_m,
                models.EfficientNet_V2_M_Weights.DEFAULT,
            ),
            "v2-l": (
                models.efficientnet_v2_l,
                models.EfficientNet_V2_L_Weights.DEFAULT,
            ),
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


class ConvNeXtBackbone(nn.Module):
    def __init__(
        self,
        version: str,
        out_levels: tuple[int, ...] = (3,),
        pretrained: bool = True,
        weights: str | None = None,
        **kwargs,
    ):
        """Initializes the ConvNeXt backbone.

        Args:
            version (str): ConvNeXt size. Accepts "convnext_tiny", "convnext_small", "convnext_base", "convnext_large".
            out_levels (tuple): Which stage outputs to return. 1..4 correspond to the
                four ConvNeXt stages (after each downsample+stage). 0 includes the input.
                Defaults to (4,) (i.e., last stage only).
            pretrained (bool): Kept for API parity; not used by this custom ConvNeXt.
            **kwargs: Passed to ConvNeXt constructor (e.g., drop_path_rate, layer_scale_init_value, patch_size).
        """
        super().__init__()
        convnext_size = version.split("_")[1]
        if weights is None:
            weights = Weights.LVD1689M
        if convnext_size == "tiny":
            weights = "/home/abel/Documents/tepnet_fork/models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
            pretrained = True
            model = dinov3_convnext_tiny(pretrained=pretrained, weights=weights)
        elif convnext_size == "small":
            model = dinov3_convnext_small(pretrained=pretrained, weights=weights)
        elif convnext_size == "base":
            model = dinov3_convnext_base(pretrained=pretrained, weights=weights)
        elif convnext_size == "large":
            weights = "/home/abel/Documents/tepnet_fork/models/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"
            pretrained = True
            model = dinov3_convnext_large(pretrained=pretrained, weights=weights)
        self.stages = nn.ModuleList([nn.Sequential(model.downsample_layers[i], model.stages[i]) for i in range(4)])
        self.out_levels = out_levels
        self.out_channels = [3] if (len(self.out_levels) > 0 and self.out_levels[0] == 0) else []
        for i in self.out_levels:
            if i == 0:
                continue
            if not (1 <= i <= 4):
                raise ValueError(f"out_levels must be in {{0,1,2,3,4}}; got {i}")
            stage = self.stages[i - 1]
            convs = [m for m in stage.modules() if isinstance(m, nn.Conv2d)]
            if len(convs) == 0:
                raise RuntimeError(f"No Conv2d modules found in ConvNeXt stage {i}.")
            self.out_channels.append(convs[-1].out_channels)
        self.out_channels = tuple(self.out_channels)
        self.reduction_factor = 4 * 2 ** (out_levels[-1] - 1)

    def forward(self, x: torch.Tensor):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if i in self.out_levels:
                features.append(x)
        return features
