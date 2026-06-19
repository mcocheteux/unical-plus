"""
Regression head: shared trunk → split translation / rotation branches.

The split design lets the network learn specialised representations for
the two geometrically different quantities while still sharing early
feature processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(dims: list[int], activate_last: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        if i > 0 or activate_last:
            layers.append(nn.LeakyReLU(0.1, inplace=False))
        layers.append(nn.Linear(dims[i], dims[i + 1]))
    return nn.Sequential(*layers)


class SplitRegressionHead(nn.Module):
    """
    Two-branch MLP head.

    Architecture:
        feature_vector
            → common layers
            → split →  translation branch  → tx, ty, tz  (3 values)
                    →  rotation branch     → 6-D continuous rotation (Zhou et al.)

    Args:
        in_features:        Size of the input feature vector from the backbone.
        common_hidden:      Hidden sizes of the shared trunk (can be empty list).
        trans_hidden:       Hidden sizes of the translation branch.
        rot_hidden:         Hidden sizes of the rotation branch.
        rot_dim:            Size of the rotation output. Defaults to 6 for the
                            continuous 6-D representation; the model converts this
                            to a rotation matrix downstream.
    """

    def __init__(
        self,
        in_features: int,
        common_hidden: list[int],
        trans_hidden: list[int],
        rot_hidden: list[int],
        rot_dim: int = 6,
    ) -> None:
        super().__init__()

        # Shared trunk
        trunk_dims = [in_features] + list(common_hidden)
        if len(trunk_dims) > 1:
            self.trunk = _mlp(trunk_dims, activate_last=False)
            trunk_out = trunk_dims[-1]
        else:
            self.trunk = nn.Identity()
            trunk_out = in_features

        # Translation branch: LeakyReLU before each linear layer except first
        self.trans_head = _mlp([trunk_out] + list(trans_hidden) + [3], activate_last=True)
        # Rotation branch (continuous 6-D representation by default)
        self.rot_head = _mlp([trunk_out] + list(rot_hidden) + [rot_dim], activate_last=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, in_features)

        Returns:
            (trans, rot): trans (B, 3) in metres; rot (B, rot_dim) raw rotation
            representation (6-D by default, converted to a matrix downstream).
        """
        h = self.trunk(x)
        return self.trans_head(h), self.rot_head(h)
