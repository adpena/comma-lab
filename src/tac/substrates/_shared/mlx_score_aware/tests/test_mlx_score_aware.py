# SPDX-License-Identifier: MIT
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tac.substrates._shared.mlx_score_aware import (
    MlxScoreAwareHarnessError,
    RendererBundle,
    assert_numpy_portable_inflate,
    decode_frames_nhwc01,
    is_mlx_available,
    require_mlx_for_harness,
    score_aware_loss,
)

mx = pytest.importorskip("mlx.core")


class ReconstructPairModel:
    def __init__(self, target_rgb_0, target_rgb_1) -> None:
        self.target_rgb_0 = target_rgb_0
        self.target_rgb_1 = target_rgb_1

    def parameters(self):
        return {}

    def reconstruct_pair(self, idx):
        rgb_0 = mx.transpose(self.target_rgb_0[idx], (0, 3, 1, 2))
        rgb_1 = mx.transpose(self.target_rgb_1[idx], (0, 3, 1, 2))
        return rgb_0, rgb_1


class CallPairModel:
    def __init__(self, target_rgb_0, target_rgb_1) -> None:
        pair = mx.stack(
            [
                mx.transpose(target_rgb_0, (0, 3, 1, 2)),
                mx.transpose(target_rgb_1, (0, 3, 1, 2)),
            ],
            axis=1,
        )
        self.pair_255 = pair * 255.0

    def parameters(self):
        return {}

    def __call__(self, idx):
        return self.pair_255[idx]


def _targets():
    target_0 = mx.array(
        np.linspace(0.0, 1.0, num=2 * 4 * 4 * 3, dtype=np.float32).reshape(
            2, 4, 4, 3
        )
    )
    target_1 = 1.0 - target_0
    return target_0, target_1


def _scalar(value) -> float:
    return float(np.array(value))


def test_device_gate_reports_mlx_available_on_local_host() -> None:
    assert is_mlx_available()
    assert require_mlx_for_harness().__name__ == "mlx.core"


def test_renderer_bundle_validation_fail_closed() -> None:
    target_0, target_1 = _targets()
    with pytest.raises(MlxScoreAwareHarnessError, match="forward_convention"):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            forward_convention="unknown",
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="num_pairs"):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=0,
        )
    with pytest.raises(MlxScoreAwareHarnessError, match="distillation_weight"):
        RendererBundle(
            model=object(),
            target_rgb_0=target_0,
            target_rgb_1=target_1,
            num_pairs=2,
            distillation_weight=-1.0,
        )


def test_decode_frames_supports_reconstruct_pair_nchw01() -> None:
    target_0, target_1 = _targets()
    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
    )
    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, mx.array([0, 1]))
    np.testing.assert_allclose(np.array(rgb_0), np.array(target_0), atol=1e-7)
    np.testing.assert_allclose(np.array(rgb_1), np.array(target_1), atol=1e-7)


def test_decode_frames_supports_call_b2chw_255() -> None:
    target_0, target_1 = _targets()
    bundle = RendererBundle(
        model=CallPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="call_b2chw_255",
    )
    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, mx.array([0, 1]))
    np.testing.assert_allclose(np.array(rgb_0), np.array(target_0), atol=1e-6)
    np.testing.assert_allclose(np.array(rgb_1), np.array(target_1), atol=1e-6)


def test_score_aware_loss_recon_distill_and_extra_terms_are_composed() -> None:
    target_0, target_1 = _targets()

    def extra_loss(_model, _idx):
        return {"regularizer": mx.array(2.0)}

    bundle = RendererBundle(
        model=ReconstructPairModel(target_0, target_1),
        target_rgb_0=target_0,
        target_rgb_1=target_1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        extra_loss_terms=extra_loss,
        extra_loss_weights={"regularizer": 0.25},
        distillation_weight=1.0,
    )
    total, parts = score_aware_loss(bundle, mx.array([0, 1]))
    assert _scalar(parts["recon"]) < 1e-10
    assert _scalar(parts["distill"]) < 1e-8
    assert _scalar(parts["regularizer"]) == pytest.approx(2.0)
    assert _scalar(total) == pytest.approx(0.5, abs=1e-6)
    assert _scalar(parts["total"]) == pytest.approx(_scalar(total), abs=1e-7)


def test_numpy_portable_inflate_gate_uses_fail_closed_error_type() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good = root / "inflate_good.py"
        bad = root / "inflate_bad.py"
        good.write_text("import numpy as np\nx = np.array([1])\n", encoding="utf-8")
        bad.write_text("import torch\n", encoding="utf-8")
        result = assert_numpy_portable_inflate(good)
        assert result["numpy_portable"] is True
        assert "numpy" in result["import_roots"]
        with pytest.raises(MlxScoreAwareHarnessError, match="forbidden"):
            assert_numpy_portable_inflate(bad)
