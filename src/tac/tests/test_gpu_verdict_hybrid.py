# SPDX-License-Identifier: MIT
"""Unit tests for the GPU (MLX) verdict device + CPU-torch positive-control ANCHOR (HYBRID).

Operator 2026-07-08: run the ADVISORY verdict on GPU (deterministic w/ fused-R, faster) as a
fast trajectory monitor, keeping the CPU-torch verdict as the slow-cadence positive-control
sentinel (the instrument must not fold into what it measures; prior baselines are CPU numbers).

These lock the DEVICE-FREE contract ($0, no torch/MLX): anchor cadence, flip-disagreement
counter, paired drift-row schema + NON-PROMOTABLE axis tags, the fail-closed conflict guard,
AND the default-OFF byte-identity invariants (trainer arg default + DSL default = cpu).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.gpu_verdict import (
    AXIS_TAG_CPU,
    AXIS_TAG_GPU,
    build_paired_drift_row,
    flip_disagreement_count,
    gpu_verdict_conflicts,
    should_anchor,
)

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"


# ── anchor cadence ─────────────────────────────────────────────────────────────────────
def test_should_anchor_disabled_by_default():
    # anchor_every 0 => never fire (gpu-only monitoring, the default).
    assert should_anchor(1, 0) is False
    assert should_anchor(1000, 0) is False


def test_should_anchor_negative_never_fires():
    assert should_anchor(5, -1) is False


def test_should_anchor_every_one():
    assert all(should_anchor(c, 1) for c in range(1, 6))


def test_should_anchor_every_n_boundaries():
    every = 25
    fired = [c for c in range(1, 101) if should_anchor(c, every)]
    assert fired == [25, 50, 75, 100]


# ── flip-disagreement counter ──────────────────────────────────────────────────────────
def test_flip_disagreement_identical_is_zero():
    m = np.array([[0, 1], [2, 3]], dtype=np.int64)
    assert flip_disagreement_count([m, m], [m.copy(), m.copy()]) == 0


def test_flip_disagreement_counts_pixels():
    a = np.array([[0, 1], [2, 3]], dtype=np.int64)
    b = np.array([[0, 9], [2, 9]], dtype=np.int64)  # 2 pixels differ
    assert flip_disagreement_count([a], [b]) == 2


def test_flip_disagreement_stacked_equals_list():
    rng = np.random.default_rng(0)
    cpu = rng.integers(0, 5, size=(4, 6, 7)).astype(np.int64)
    gpu = cpu.copy()
    gpu[1, 0, 0] += 1
    gpu[3, 2, 3] += 1
    stacked = flip_disagreement_count(cpu, gpu)
    listed = flip_disagreement_count([cpu[i] for i in range(4)], [gpu[i] for i in range(4)])
    assert stacked == listed == 2


def test_flip_disagreement_paircount_mismatch_raises():
    m = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="pair-count mismatch"):
        flip_disagreement_count([m, m], [m])


def test_flip_disagreement_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        flip_disagreement_count([np.zeros((2, 2), np.int64)], [np.zeros((2, 3), np.int64)])


# ── paired drift-row schema ─────────────────────────────────────────────────────────────
def _paired(**over):
    base = {
        "d_seg_cpu": 0.0050, "d_seg_gpu": 0.0051,
        "d_pose_cpu": 3.4e-5, "d_pose_gpu": 3.6e-5,
        "argmax_flip_disagreement_count": 42,
        "max_abs_dpose_delta": 1.2e-6,
    }
    base.update(over)
    return base


def test_paired_row_has_all_fields_and_axis_tags():
    row = build_paired_drift_row(_paired(), epoch=225, verdict_batch=32)
    for k in ("stage", "epoch", "verdict_batch", "d_seg_gpu", "d_seg_cpu", "d_pose_gpu",
              "d_pose_cpu", "d_seg_delta", "d_pose_delta", "argmax_flip_disagreement_count",
              "max_abs_dpose_delta", "axis_gpu", "axis_cpu", "promotable"):
        assert k in row, f"missing {k}"
    assert row["stage"] == "verdict_anchor"
    assert row["epoch"] == 225
    assert row["verdict_batch"] == 32
    assert row["axis_gpu"] == AXIS_TAG_GPU
    assert row["axis_cpu"] == AXIS_TAG_CPU


def test_paired_row_is_non_promotable():
    # CLAUDE.md NON-NEGOTIABLE: MLX/MPS is never a score; both flavours advisory.
    row = build_paired_drift_row(_paired(), epoch=1, verdict_batch=32)
    assert row["promotable"] is False


def test_paired_row_deltas_are_gpu_minus_cpu():
    row = build_paired_drift_row(
        _paired(d_seg_cpu=0.0050, d_seg_gpu=0.0060, d_pose_cpu=1e-5, d_pose_gpu=3e-5),
        epoch=10, verdict_batch=16)
    assert row["d_seg_delta"] == pytest.approx(0.001, abs=1e-9)
    assert row["d_pose_delta"] == pytest.approx(2e-5, abs=1e-12)


def test_paired_row_passes_through_instrument_scalars():
    row = build_paired_drift_row(
        _paired(argmax_flip_disagreement_count=777, max_abs_dpose_delta=9.9e-4),
        epoch=5, verdict_batch=32)
    assert row["argmax_flip_disagreement_count"] == 777
    assert row["max_abs_dpose_delta"] == pytest.approx(9.9e-4)


def test_axis_tags_are_advisory_labels():
    assert "advisory" in AXIS_TAG_GPU and "MLX" in AXIS_TAG_GPU
    assert "advisory" in AXIS_TAG_CPU and "CPU" in AXIS_TAG_CPU


# ── fail-closed conflict guard ──────────────────────────────────────────────────────────
def test_gpu_conflicts_clean_is_empty():
    assert gpu_verdict_conflicts(
        async_verdict=False, curriculum_nucleus_guard=False, ladder_island_homotopy=False) == []


def test_gpu_conflicts_flags_async():
    c = gpu_verdict_conflicts(
        async_verdict=True, curriculum_nucleus_guard=False, ladder_island_homotopy=False)
    assert len(c) == 1 and "async" in c[0]


def test_gpu_conflicts_flags_training_feeding_controllers():
    c = gpu_verdict_conflicts(
        async_verdict=False, curriculum_nucleus_guard=True, ladder_island_homotopy=True)
    assert len(c) == 2
    assert any("nucleus" in x for x in c) and any("ladder" in x for x in c)


# ── default-OFF byte-identity invariants ────────────────────────────────────────────────
def test_trainer_arg_default_is_cpu_byte_identity():
    """The trainer default MUST be cpu (today's byte-identical CPU-torch authority). A flip to
    gpu-by-default would silently change the live #205 trajectory device — forbidden."""
    src = _TRAINER.read_text()
    m = re.search(r'--verdict-device".*?default="(\w+)"', src, re.DOTALL)
    assert m is not None and m.group(1) == "cpu"
    m2 = re.search(r'--verdict-anchor-every".*?default=(\d+)', src, re.DOTALL)
    assert m2 is not None and int(m2.group(1)) == 0


def test_dsl_verdict_cadence_default_is_cpu():
    from tac.witness_dsl.curriculum_dsl import VerdictCadence
    vc = VerdictCadence()
    assert vc.verdict_device == "cpu"
    assert vc.verdict_anchor_every == 0
    assert vc.validate() == []
    assert vc.flags()["--verdict-device"] == "cpu"


def test_dsl_verdict_cadence_gpu_async_conflict_flagged():
    from tac.witness_dsl.curriculum_dsl import VerdictCadence
    probs = VerdictCadence(verdict_device="gpu", async_verdict=True).validate()
    assert probs and "async" in probs[0]


def test_dsl_verdict_device_lever_overrides():
    from tac.witness_dsl.curriculum_dsl import VerdictDevice
    lv = VerdictDevice(50)
    assert lv.overrides["--verdict-device"] == "gpu"
    assert lv.overrides["--verdict-anchor-every"] == 50


def test_dsl_verdict_flags_covered_by_registry():
    """The two flags must be HELD by the DSL (registry coverage), not orphaned trainer flags."""
    from tac.witness_dsl import lever_registry as lr
    comp = lr.completeness()
    assert "--verdict-device" not in comp.unmapped
    assert "--verdict-anchor-every" not in comp.unmapped


# ── typed_config (requirement-V layer) ──────────────────────────────────────────────────
def _typed(**over):
    from tac.witness_dsl.typed_config import (
        ProvenanceClass,
        Provenanced,
        TypedAnneal,
        TypedWitnessConfig,
    )
    kw = dict(
        name="t", out_dir="experiments/results/x", gt_cache="g.npz", num_pairs=600, epochs=10,
        wall_clock_budget_days=Provenanced(
            value=0.5, provenance=ProvenanceClass.DERIVED_AT_CONFIG, unit="days"),
        temp=TypedAnneal(
            start=Provenanced(value=1.0, provenance=ProvenanceClass.MEASURED_ANCHOR, unit="tau"),
            end=Provenanced(value=0.31, provenance=ProvenanceClass.MEASURED_ANCHOR, unit="tau")),
    )
    kw.update(over)
    return TypedWitnessConfig(**kw)


def test_typed_config_default_is_cpu():
    fd = _typed().to_program().flag_dict()
    assert fd["--verdict-device"] == "cpu"
    assert fd["--verdict-anchor-every"] == 0


def test_typed_config_gpu_compiles():
    fd = _typed(verdict_device="gpu", verdict_anchor_every=25).to_program().flag_dict()
    assert fd["--verdict-device"] == "gpu"
    assert fd["--verdict-anchor-every"] == 25


def test_typed_config_rejects_bad_device_and_double_set():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _typed(verdict_device="mps")
    with pytest.raises(ValidationError):
        _typed(base={"--verdict-device": "gpu"})
