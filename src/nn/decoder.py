import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dNormActivation(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
        )
        self.conv2 = Conv2dNormActivation(
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super(UNetDecoder, self).__init__()
        # encoder_channels is like (3, 96, 192, 384, 768) for ConvNeXt-Tiny with out_levels=(0..4)
        encoder_channels = encoder_channels[::-1]  # -> (768, 384, 192, 96, 3)

        head_channels = encoder_channels[0]  # 768
        in_channels = tuple([head_channels] + list(decoder_channels[:-1]))  # [768, 256, 128, 64, 32]

        # ---- IMPORTANT: make skip_channels match the desired skip routing ----
        # We want block skips: [level3, level2, level1, None, level0]
        # In channel terms (with the reversed list above): [384, 192, 96, 0, 3]
        if len(decoder_channels) == 5 and len(encoder_channels) == 5:
            skip_channels = (
                encoder_channels[1],  # 384
                encoder_channels[2],  # 192
                encoder_channels[3],  # 96
                0,  # None at 256x
                encoder_channels[4],
            )  # 3 (input image)
        else:
            # default fallback for other encoder/decoder depths
            skip_channels = tuple(list(encoder_channels[1:]) + [0])

        self.blocks = nn.ModuleList(
            [DecoderBlock(in_ch, skip_ch, out_ch) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, decoder_channels)]
        )

    def forward(self, features):
        # features = [level0, level1, level2, level3, level4]; reverse to start from deepest:
        features = features[::-1]  # [level4, level3, level2, level1, level0]

        x = features[0]  # level4
        # Route skips per block: [level3, level2, level1, None, level0]
        if len(self.blocks) == 5 and len(features) == 5:
            skips = [features[1], features[2], features[3], None, features[4]]
        else:
            skips = features[1:] + [None]

        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
        return x
