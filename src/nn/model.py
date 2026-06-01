import math

import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

from .backbone import (
    DinoV3Backbone,
    EfficientNetBackbone,
    ResNetBackbone,
    RTDetrV4Backbone,
)
from .decoder import UNetDecoder


def make_conv_adapter(in_channels, adapter_channels, adapter_depth, pool_channels):
    if adapter_depth < 1:
        raise ValueError("adapter_depth must be >= 1")
    adapter_layers = [
        Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=adapter_channels,
            kernel_size=3,
            padding=1,
            activation_layer=nn.ReLU,
        )
    ]
    for _ in range(adapter_depth - 1):
        adapter_layers.append(
            Conv2dNormActivation(
                in_channels=adapter_channels,
                out_channels=adapter_channels,
                kernel_size=3,
                padding=1,
                activation_layer=nn.ReLU,
            )
        )
    adapter_layers.append(
        nn.Conv2d(
            in_channels=adapter_channels,
            out_channels=pool_channels,
            kernel_size=1,
        )
    )
    return nn.Sequential(*adapter_layers)


class ClassificationNet(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        classes,
        pool_channels,
        fc_hidden_size,
        pretrained=False,
    ):
        """Initializes the train ego-path detection model for the classification method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            input_shape (tuple): Input shape (C, H, W).
            anchors (int): Number of horizontal anchors in the input image where the path is classified.
            classes (int): Number of classes (grid cells) for each anchor. Background class is not included.
            pool_channels (int): Number of output channels of the pooling layer.
            fc_hidden_size (int): Number of units in the hidden layer of the fully connected part.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(ClassificationNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(
                version=backbone[13:], pretrained=pretrained
            )
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        else:
            raise NotImplementedError
        self.pool = nn.Conv2d(
            in_channels=self.backbone.out_channels[-1],
            out_channels=pool_channels,
            kernel_size=1,
        )  # stride=1, padding=0
        self.fc = nn.Sequential(
            nn.Linear(
                pool_channels
                * math.ceil(input_shape[1] / self.backbone.reduction_factor)
                * math.ceil(input_shape[2] / self.backbone.reduction_factor),
                fc_hidden_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, anchors * (classes + 1) * 2),
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        fea = self.pool(x).flatten(start_dim=1)
        clf = self.fc(fea)
        return clf


class RegressionNet(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        pool_channels,
        fc_hidden_size,
        pretrained=False,
        dinov3_repo_dir="external/dinov3",
        dinov3_weights_dir="dinov3_models",
        dinov3_intermediate_layers=4,
        dinov3_adapter_channels=256,
        dinov3_adapter_depth=1,
        dinov3_layer_set="four_last",
        dinov3_use_cls_token=False,
        rtdetrv4_repo_dir="external/RT-DETRv4",
        rtdetrv4_weights_dir="rtdetrv4_models",
        rtdetrv4_use_encoder=True,
        rtdetrv4_feature_level=1,
        rtdetrv4_adapter_channels=256,
        rtdetrv4_adapter_depth=1,
    ):
        """Initializes the train ego-path detection model for the regression method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            input_shape (tuple): Input shape (C, H, W).
            anchors (int): Number of horizontal anchors in the input image where the path is regressed.
            pool_channels (int): Number of output channels of the pooling layer.
            fc_hidden_size (int): Number of units in the hidden layer of the fully connected part.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(RegressionNet, self).__init__()
        if backbone.startswith("dinov3"):
            self.backbone = DinoV3Backbone(
                name=backbone,
                repo_dir=dinov3_repo_dir,
                weights_dir=dinov3_weights_dir,
                intermediate_layers=dinov3_intermediate_layers,
                layer_set=dinov3_layer_set,
                use_cls_token=dinov3_use_cls_token,
            )
            self.pool = make_conv_adapter(
                in_channels=self.backbone.out_channels[-1],
                adapter_channels=dinov3_adapter_channels,
                adapter_depth=dinov3_adapter_depth,
                pool_channels=pool_channels,
            )
        elif backbone.startswith("rtdetrv4"):
            self.backbone = RTDetrV4Backbone(
                name=backbone,
                repo_dir=rtdetrv4_repo_dir,
                weights_dir=rtdetrv4_weights_dir,
                use_encoder=rtdetrv4_use_encoder,
                feature_level=rtdetrv4_feature_level,
            )
            self.pool = make_conv_adapter(
                in_channels=self.backbone.out_channels[-1],
                adapter_channels=rtdetrv4_adapter_channels,
                adapter_depth=rtdetrv4_adapter_depth,
                pool_channels=pool_channels,
            )
        elif backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(
                version=backbone[13:], pretrained=pretrained
            )
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
            self.pool = nn.Conv2d(
                in_channels=self.backbone.out_channels[-1],
                out_channels=pool_channels,
                kernel_size=1,
            )  # stride=1, padding=0
        else:
            raise NotImplementedError
        self.fc = nn.Sequential(
            nn.Linear(
                pool_channels
                * math.ceil(input_shape[1] / self.backbone.reduction_factor)
                * math.ceil(input_shape[2] / self.backbone.reduction_factor),
                fc_hidden_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, anchors * 2 + 1),
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        fea = self.pool(x).flatten(start_dim=1)
        reg = self.fc(fea)
        return reg


class SegmentationNet(nn.Module):
    def __init__(
        self,
        backbone,
        decoder_channels,
        pretrained=False,
    ):
        """Initializes the train ego-path detection model for the segmentation method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            decoder_channels (tuple): Number of output channels of each decoder block.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(SegmentationNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.encoder = EfficientNetBackbone(
                version=backbone[13:],
                out_levels=(1, 3, 4, 6, 8),
                pretrained=pretrained,
            )
        elif backbone.startswith("resnet"):
            self.encoder = ResNetBackbone(
                version=backbone[6:],
                out_levels=(1, 2, 3, 4, 5),
                pretrained=pretrained,
            )
        else:
            raise NotImplementedError
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )
        self.segmentation_head = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=1,  # binary segmentation
            kernel_size=3,
            padding=1,
        )  # stride=1

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks
