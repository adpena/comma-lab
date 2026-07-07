# SPDX-License-Identifier: MIT
"""Tests for tac.witness_control.perclass_verdict (2026-07-07 telemetry enhancement)."""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.perclass_verdict import (
    N_CLASSES,
    memory_telemetry_fields,
    per_class_dseg_fields,
    per_class_flip_stats,
)


def _mk(gt_val: int, realized_val: int, h: int = 4, w: int = 5):
    gt = np.full((h, w), gt_val, dtype=np.int64)
    realized = np.full((h, w), realized_val, dtype=np.int64)
    return realized, gt


def test_all_agree_zero_flips():
    realized, gt = _mk(0, 0)
    flips, pixels = per_class_flip_stats([realized], [gt])
    assert flips.sum() == 0
    assert pixels[0] == 20 and pixels[1:].sum() == 0


def test_flip_attributed_to_gt_class_not_realized():
    # GT says Lane(1) everywhere; realized says Road(0) => all 20 flips belong to class 1.
    realized, gt = _mk(1, 0)
    flips, pixels = per_class_flip_stats([realized], [gt])
    assert flips[1] == 20 and flips[0] == 0
    assert pixels[1] == 20


def test_sum_identity_matches_total_dseg():
    # Mixed map: half GT Road correct, half GT Lane flipped => total d_seg == 0.5 and the
    # per-class accumulation reproduces it exactly (the NO-FAKE self-check).
    gt = np.zeros((2, 10), dtype=np.int64)
    gt[1, :] = 1
    realized = gt.copy()
    realized[1, :] = 3  # flip every Lane pixel
    flips, pixels = per_class_flip_stats([realized], [gt])
    total_dseg = float((realized != gt).mean())
    assert flips.sum() / pixels.sum() == pytest.approx(total_dseg)
    assert flips[1] == 10 and flips[0] == 0


def test_multi_pair_accumulation():
    r1, g1 = _mk(2, 2)      # agree on Undrivable
    r2, g2 = _mk(4, 0)      # MyCar flipped to Road
    flips, pixels = per_class_flip_stats([r1, r2], [g1, g2])
    assert pixels[2] == 20 and pixels[4] == 20
    assert flips[2] == 0 and flips[4] == 20


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        per_class_flip_stats([np.zeros((2, 2), np.int64)], [np.zeros((3, 3), np.int64)])


def test_fields_rates_and_shares():
    flips = np.array([0, 10, 0, 0, 0], dtype=np.int64)
    pixels = np.array([100, 20, 0, 50, 0], dtype=np.int64)
    f = per_class_dseg_fields(flips, pixels)
    assert f["d_seg_by_class"][1] == pytest.approx(0.5)
    assert f["d_seg_by_class"][2] == 0.0  # zero-pixel class => 0.0, never NaN
    assert f["flip_share_by_class"][1] == pytest.approx(1.0)
    assert len(f["d_seg_by_class"]) == N_CLASSES
    # JSON-serializable plain floats
    import json
    json.dumps(f)


def test_fields_zero_flips_zero_share():
    f = per_class_dseg_fields(np.zeros(5, np.int64), np.full(5, 10, np.int64))
    assert all(x == 0.0 for x in f["flip_share_by_class"])
    assert all(x == 0.0 for x in f["d_seg_by_class"])


def test_memory_fields_fail_open_and_json_safe():
    out = memory_telemetry_fields()
    # On this host psutil exists => rss present; but the contract is only "never raises,
    # values are floats, json-safe".
    import json
    json.dumps(out)
    for v in out.values():
        assert isinstance(v, float) and v >= 0.0


def test_torch_tensor_inputs_accepted():
    torch = pytest.importorskip("torch")
    gt = torch.zeros((3, 3), dtype=torch.int64)
    realized = torch.ones((3, 3), dtype=torch.int64)
    flips, pixels = per_class_flip_stats([realized], [gt])
    assert flips[0] == 9 and pixels[0] == 9
