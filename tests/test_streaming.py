"""Check streaming inference against the existing full-window forward pass."""

from copy import deepcopy

import pytest
import torch

from unical.data.dataset import Batch
from unical.losses.combined import CombinedLoss
from unical.losses.regression import RegressionLoss
from unical.models.head import SplitRegressionHead
from unical.models.module import UniCal
from unical.models.streaming import StreamingCalibrator
from unical.models.temporal import TemporalFusion


class CountingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 1), torch.nn.BatchNorm2d(8))
        self.encoded_frames = 0

    def forward(self, batch):
        self.encoded_frames += batch.img.shape[0]
        return self.encoder(torch.cat((batch.img, batch.lidar_map), dim=1)).mean((2, 3))


def model(fusion="none"):
    torch.manual_seed(42)
    return UniCal(
        CountingBackbone(),
        SplitRegressionHead(8, [8], [4], [4]),
        CombinedLoss(RegressionLoss()),
        temporal=TemporalFusion(
            feature_dim=8,
            fusion_type=fusion,
            gru_hidden=8,
            transformer_layers=1,
            transformer_heads=2,
            transformer_ff_dim=16,
            max_seq_len=5,
        ),
    ).eval()


def frame(index, batch_size=2):
    generator = torch.Generator().manual_seed(index)
    return Batch(
        img=torch.randn(batch_size, 1, 3, 8, 8, generator=generator),
        lidar_map=torch.randn(batch_size, 1, 1, 8, 8, generator=generator),
        target_reg=(torch.zeros(batch_size, 3), torch.eye(3).expand(batch_size, 3, 3)),
        pcl=[torch.zeros(batch_size, 1, 4)],
        metadata=[[{} for _ in range(batch_size)]],
    )


def window(frames):
    return frames[-1]._replace(
        img=torch.cat([f.img for f in frames], dim=1),
        lidar_map=torch.cat([f.lidar_map for f in frames], dim=1),
        pcl=[f.pcl[0] for f in frames],
        metadata=[f.metadata[0] for f in frames],
    )


@pytest.mark.parametrize("fusion", ["none", "gru", "transformer"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("length", [1, 3])
def test_stream_matches_naive_windows_and_encodes_each_frame_once(fusion, batch_size, length):
    cached_model = model(fusion)
    reference = deepcopy(cached_model)
    stream = StreamingCalibrator(cached_model, sequence_length=length)
    frames = [frame(i, batch_size) for i in range(6)]
    before = {k: v.clone() for k, v in cached_model.state_dict().items()}
    for i, batch in enumerate(frames):
        batch.img.requires_grad_(True)
        actual = stream.step(batch, context_id="drive/calibration-v1", frame_index=i)
        assert stream.history_length == min(i + 1, length)
        if i < length - 1:
            assert actual is None
        else:
            with torch.no_grad():
                expected = reference(window(frames[i - length + 1 : i + 1]))
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                assert not a.requires_grad
    assert cached_model.backbone.encoded_frames == len(frames) * batch_size
    assert reference.backbone.encoded_frames == (len(frames) - length + 1) * length * batch_size
    for key, value in cached_model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_projection_context_change_and_manual_reset_release_history():
    net = model()
    stream = StreamingCalibrator(net)
    for i in range(3):
        stream.step(frame(i), context_id=("drive", "projection-v1"), frame_index=i)
    assert stream.step(frame(3), context_id=("drive", "projection-v2"), frame_index=3) is None
    assert stream.history_length == 1
    stream.reset()
    assert stream.history_length == 0


@pytest.mark.parametrize("next_index", [0, 2, 4])
def test_gap_duplicate_or_reversed_frame_resets_history(next_index):
    stream = StreamingCalibrator(model())
    for i in range(3):
        stream.step(frame(i), context_id="drive", frame_index=i)
    assert stream.step(frame(next_index), context_id="drive", frame_index=next_index) is None
    assert stream.history_length == 1


@pytest.mark.parametrize("state", ["parameter", "buffer"])
def test_model_state_change_resets_history(state):
    net = model()
    stream = StreamingCalibrator(net)
    for i in range(3):
        stream.step(frame(i), context_id="drive", frame_index=i)
    with torch.no_grad():
        tensor = (
            net.backbone.encoder[0].weight
            if state == "parameter"
            else net.backbone.encoder[1].running_mean
        )
        tensor.add_(0.25)
    assert stream.step(frame(3), context_id="drive", frame_index=3) is None
    for i in [4, 5]:
        actual = stream.step(frame(i), context_id="drive", frame_index=i)
    with torch.no_grad():
        expected = net(window([frame(i) for i in [3, 4, 5]]))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("child_only", [False, True])
def test_training_mode_is_rejected_and_clears_cache(child_only):
    net = model()
    stream = StreamingCalibrator(net)
    for i in range(3):
        stream.step(frame(i), context_id="drive", frame_index=i)
    (net.backbone if child_only else net).train()
    with pytest.raises(ValueError, match="eval mode"):
        stream.step(frame(3), context_id="drive", frame_index=3)
    assert stream.history_length == 0


def test_frame_stride_preserves_causal_observations():
    net = model()
    stream = StreamingCalibrator(net, frame_stride=2)
    for i in [0, 2, 4]:
        actual = stream.step(frame(i), context_id="drive", frame_index=i)
    with torch.no_grad():
        expected = net(window([frame(i) for i in [0, 2, 4]]))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


def test_precision_change_starts_a_fresh_history():
    net = model()
    stream = StreamingCalibrator(net)
    for i in range(2):
        stream.step(frame(i), context_id="drive", frame_index=i)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert stream.step(frame(2), context_id="drive", frame_index=2) is None
        assert stream.step(frame(3), context_id="drive", frame_index=3) is None
        actual = stream.step(frame(4), context_id="drive", frame_index=4)
        with torch.no_grad():
            expected = net(window([frame(i) for i in [2, 3, 4]]))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


def test_matrix_precision_change_resets_history():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        stream = StreamingCalibrator(model())
        for i in range(2):
            stream.step(frame(i), context_id="drive", frame_index=i)
        torch.set_float32_matmul_precision("high")
        assert stream.step(frame(2), context_id="drive", frame_index=2) is None
        assert stream.history_length == 1
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires local CUDA")
@pytest.mark.parametrize("fusion", ["none", "gru", "transformer"])
def test_cuda_stream_matches_bfloat16_naive_windows(fusion):
    net = model(fusion).cuda()
    stream = StreamingCalibrator(net)
    frames = [frame(i) for i in range(4)]
    frames = [f._replace(img=f.img.cuda(), lidar_map=f.lidar_map.cuda()) for f in frames]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, batch in enumerate(frames):
            actual = stream.step(batch, context_id="drive", frame_index=i)
            if i >= 2:
                with torch.no_grad():
                    expected = net(window(frames[i - 2 : i + 1]))
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)


def test_stream_rejects_multiframe_input_and_invalid_lengths():
    net = model()
    with pytest.raises(ValueError, match="positive"):
        StreamingCalibrator(net, sequence_length=0)
    with pytest.raises(ValueError, match="positive"):
        StreamingCalibrator(net, frame_stride=0)
    with pytest.raises(ValueError, match="positional capacity"):
        StreamingCalibrator(model("transformer"), sequence_length=6)
    with pytest.raises(ValueError, match="exactly one frame"):
        StreamingCalibrator(net).step(
            window([frame(0), frame(1)]), context_id="drive", frame_index=1
        )
