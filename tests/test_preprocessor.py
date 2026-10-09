"""Verify image values supplied to the pretrained backbone without changing geometry."""

import numpy as np
import pytest

from unical.data.preprocessor import DataPreprocessor, PreprocessorConfig
from unical.utils.geometry import imagenet_normalize


@pytest.mark.parametrize("mode", ["imagenet_rgb", "mobilevit_bgr"])
def test_image_normalization_modes(mode: str) -> None:
    image = np.broadcast_to(np.array([10, 20, 30], dtype=np.uint8), (4, 4, 3)).copy()
    cloud = np.array([[0.0, 0.0, 10.0, 1.0]], dtype=np.float32)
    K = np.array([[1, 0, 2], [0, 1, 2], [0, 0, 1]], dtype=np.float64)
    process = DataPreprocessor(PreprocessorConfig(width=4, height=4, image_normalization=mode))
    result, depth = process(image, cloud, np.eye(4), K)
    expected = (
        imagenet_normalize(image)
        if mode == "imagenet_rgb"
        else np.broadcast_to(np.array([30, 20, 10]) / 255.0, image.shape)
    )
    np.testing.assert_allclose(result, expected, atol=1e-7)
    assert result.dtype == np.float32
    legacy_depth = DataPreprocessor(PreprocessorConfig(width=4, height=4))(
        image, cloud, np.eye(4), K
    )[1]
    np.testing.assert_array_equal(depth, legacy_depth)


def test_unknown_image_normalization_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown image normalization"):
        DataPreprocessor(PreprocessorConfig(image_normalization="unknown"))
