"""
MobileViT backbone for joint RGB + LiDAR feature extraction.

The image (C_img channels) and the LiDAR depth map (C_lid channels) are
concatenated channel-wise before being passed to MobileViT (early fusion, a
single shared backbone).  The spatial output is optionally refined by a small
convolutional head and then globally pooled to a single feature vector.

The backbone can be initialised from ImageNet-pretrained MobileViT weights
(e.g. ``apple/mobilevit-small``).  Because the fused input has more than 3
channels, the pretrained 3-channel stem convolution is "inflated" to the
required channel count: the RGB filters are copied verbatim and each extra
(LiDAR) channel is initialised from the mean of the pretrained RGB filters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from transformers import MobileViTConfig, MobileViTModel

from unical.data.dataset import Batch


class MobileViTBackbone(nn.Module):
    """
    MobileViT feature extractor for UniCal.

    Args:
        img_channels:          Number of image channels (1 for grayscale, 3 for RGB).
        lidar_channels:        Number of lidar map channels (1 depth, 2 depth+intensity).
        image_size:            Square image side length fed to MobileViT.
        hidden_sizes:          Channel widths of the three MobileViT stages.
        num_attention_heads:   Self-attention heads per MobileViT block.
        patch_size:            MobileViT patch size.
        conv_kernel_size:      Convolution kernel size in MobileViT.
        pretrained:            HuggingFace model id to load ImageNet weights from
                               (e.g. "apple/mobilevit-small"), or ``None`` to train
                               from scratch. When set, ``hidden_sizes`` /
                               ``num_attention_heads`` / ``patch_size`` /
                               ``conv_kernel_size`` are taken from the checkpoint.
        spatial_head:          If True, refine the spatial feature map with a small
                               conv block before global pooling (preserves spatial
                               misalignment cues that matter for calibration).
    """

    def __init__(
        self,
        img_channels:        int = 3,
        lidar_channels:      int = 1,
        image_size:          int = 512,
        hidden_sizes:        list[int] | None = None,
        num_attention_heads: int = 4,
        patch_size:          int = 2,
        conv_kernel_size:    int = 3,
        pretrained:          str | None = "apple/mobilevit-small",
        spatial_head:        bool = True,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes) if hidden_sizes is not None else [144, 192, 240]
        in_channels  = img_channels + lidar_channels

        if pretrained:
            self.model = MobileViTModel.from_pretrained(pretrained)
            self._inflate_stem(in_channels, img_channels)
        else:
            cfg = MobileViTConfig(
                num_channels        = in_channels,
                image_size          = image_size,
                hidden_sizes        = hidden_sizes,
                num_attention_heads = num_attention_heads,
                patch_size          = patch_size,
                conv_kernel_size    = conv_kernel_size,
            )
            self.model = MobileViTModel(cfg)

        feat_dim = self.model.config.neck_hidden_sizes[-1]
        if spatial_head:
            self.spatial_head: nn.Module = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(feat_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.spatial_head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _inflate_stem(self, in_channels: int, img_channels: int) -> None:
        """Expand the pretrained 3-channel stem conv to ``in_channels`` channels."""
        old = self.model.conv_stem.convolution
        if old.in_channels == in_channels:
            return
        new = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size = old.kernel_size,
            stride      = old.stride,
            padding     = old.padding,
            bias        = old.bias is not None,
        )
        with torch.no_grad():
            new.weight[:, :img_channels] = old.weight
            if in_channels > img_channels:
                # initialise each extra (LiDAR) channel from the mean RGB filter
                mean_w = old.weight.mean(dim=1, keepdim=True)
                new.weight[:, img_channels:] = mean_w.repeat(1, in_channels - img_channels, 1, 1)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.model.conv_stem.convolution = new

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch: Batch NamedTuple; uses .img and .lidar_map

        Returns:
            (B, D) feature vector
        """
        x = torch.cat([batch.img, batch.lidar_map], dim=1)  # (B, C_img+C_lid, H, W)
        out = self.model(x, return_dict=True)["last_hidden_state"]  # (B, D, H', W')
        out = self.spatial_head(out)                                 # (B, D, H', W')
        out = self.pool(out)                                         # (B, D, 1, 1)
        return rearrange(out, "b c 1 1 -> b c")                     # (B, D)
