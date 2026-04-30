"""
MobileViT backbone for joint RGB + LiDAR feature extraction.

The image (C_img channels) and the LiDAR depth map (C_lid channels) are
concatenated channel-wise before being passed to MobileViT.  The spatial
output is then globally pooled to a single feature vector.
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
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes) if hidden_sizes is not None else [144, 192, 240]
        in_channels  = img_channels + lidar_channels

        cfg = MobileViTConfig(
            num_channels        = in_channels,
            image_size          = image_size,
            hidden_sizes        = hidden_sizes,
            num_attention_heads = num_attention_heads,
            patch_size          = patch_size,
            conv_kernel_size    = conv_kernel_size,
        )
        self.model = MobileViTModel(cfg)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch: Batch NamedTuple; uses .img and .lidar_map

        Returns:
            (B, D) feature vector
        """
        x = torch.cat([batch.img, batch.lidar_map], dim=1)  # (B, C_img+C_lid, H, W)
        out = self.model(x, return_dict=True)["last_hidden_state"]  # (B, D, H', W')
        out = self.pool(out)                                         # (B, D, 1, 1)
        return rearrange(out, "b c 1 1 -> b c")                     # (B, D)
