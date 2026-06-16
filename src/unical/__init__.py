"""
UniCal: camera-LiDAR extrinsic calibration via MobileViT.

Programmatic usage::

    from unical import UniCal, MobileViTBackbone, SplitRegressionHead
    from unical import CombinedLoss, Transform
"""
from unical.losses.combined import CombinedLoss
from unical.losses.regression import RegressionLoss
from unical.losses.spatial import SpatialLoss
from unical.models.backbone import MobileViTBackbone
from unical.models.head import SplitRegressionHead
from unical.models.module import UniCal
from unical.utils.transform import Transform

__version__ = "0.1.0"

__all__ = [
    "UniCal",
    "MobileViTBackbone",
    "SplitRegressionHead",
    "CombinedLoss",
    "RegressionLoss",
    "SpatialLoss",
    "Transform",
    "__version__",
]
