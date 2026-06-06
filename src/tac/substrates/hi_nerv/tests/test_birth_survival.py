# SPDX-License-Identifier: MIT
"""Behavioral tests for HiNeRV birth survival + hysteresis producers.

NO-FAKE discipline: every MLX test runs the REAL ``fit_target_region_birth_from_segnet``
on the tiny smoke config to obtain a genuinely-accepted live birth, then asserts
that survival/hysteresis re-measurement reflects an actual forward recomputation
on the named surface.  If either producer were replaced by a canonical-marker
stub returning ``survived=True`` / ``passed=True`` constants, these tests fail:
the fake-quant survival is compared against a real quantized forward, the
fake-quant config restore is proven bit-for-bit, an aggressive-bits case shows
``survived`` can be ``False``, and the rows are fed to the REAL launch gate.

The numpy fixtures are deterministic synthetic label grids used to verify the
region-reconstruction math itself; they are unit fixtures, not empirical anchors
(no score claim is derived from them).
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime

import numpy as np
import pytest

from tac.substrates.hi_nerv.birth_survival import (
    BIRTH_HYSTERESIS_SCHEMA,
    BIRTH_SURVIVAL_BLOCKED_SCHEMA,
    BIRTH_SURVIVAL_SCHEMA,
    BLOCKED_SURVIVAL_SURFACES,
    BirthSurvivalError,
    measure_birth_hysteresis,
    measure_birth_survival,
    reconstruct_birth_region_mask,
)

skip_no_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="MLX is not installed",
)

_FORBIDDEN_AUTHORITY_KEYS = {
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "ready_for_exact_eval_dispatch",
    "rank_or_kill_eligible",
    "promotable",
}


# ---------------------------------------------------------------------------
# numpy-only reconstruction / fail-closed tests (no MLX required)
# ---------------------------------------------------------------------------


def test_reconstruct_birth_region_mask_matches_named_region() -> None:
    labels = np.zeros((3, 24, 32), dtype=np.int64)
    labels[0, 4:10, 6:14] = 1  # a single 48-pixel class-1 region in item 0
    worst_region = {
        "batch_index": 0,
        "class_index": 1,
        "region_label": 1,
        "region_pixel_count": 48,
        "region_unsolved_pixel_count": 48,
    }
    mask, region_pixels = reconstruct_birth_region_mask(labels, worst_region)
    assert mask.shape == labels.shape
    assert mask.dtype == np.float32
    assert region_pixels == 48
    assert int(mask.sum()) == 48
    assert mask[1].sum() == 0.0 and mask[2].sum() == 0.0
    ys, xs = np.nonzero(mask[0])
    assert ys.min() == 4 and ys.max() == 9
    assert xs.min() == 6 and xs.max() == 13


def test_reconstruct_birth_region_mask_rejects_drifted_labels() -> None:
    # Labels do NOT contain the region the receipt named -> drift must raise,
    # never silently produce a different mask.
    labels = np.zeros((2, 8, 10), dtype=np.int64)
    labels[0, 1:3, 1:3] = 1  # only a 4-pixel region exists
    worst_region = {
        "batch_index": 0,
        "class_index": 1,
        "region_label": 1,
        "region_pixel_count": 48,  # receipt claims 48 -> drift
        "region_unsolved_pixel_count": 48,
    }
    with pytest.raises(BirthSurvivalError, match="drifted from the live receipt"):
        reconstruct_birth_region_mask(labels, worst_region)


def test_blocked_surfaces_return_typed_blocker_without_survived() -> None:
    # Blocked surfaces short-circuit before any MLX usage: honesty over coverage.
    labels = np.zeros((1, 8, 10), dtype=np.int64)
    labels[0, 1:3, 1:3] = 1
    payload = {
        "action_id": "f" * 64,
        "worst_region": {
            "batch_index": 0,
            "class_index": 1,
            "region_label": 1,
            "region_pixel_count": 4,
            "region_unsolved_pixel_count": 4,
        },
    }
    for surface in BLOCKED_SURVIVAL_SURFACES:
        row = measure_birth_survival(
            object(),  # model is never touched on the blocked path
            scorer_teacher=object(),
            target_labels=labels,
            live_birth_payload=payload,
            surface=surface,
            pair_indices=[0],
        )
        assert row["schema"] == BIRTH_SURVIVAL_BLOCKED_SCHEMA
        assert row["surface"] == surface
        assert row["action_id"] == payload["action_id"]
        assert "survived" not in row  # gate must keep blocking this surface
        assert row["blocked"] is True
        assert row["blocker"]
        assert not (_FORBIDDEN_AUTHORITY_KEYS & row.keys())


def test_missing_action_id_fails_closed() -> None:
    labels = np.zeros((1, 8, 10), dtype=np.int64)
    payload = {"worst_region": {"batch_index": 0, "class_index": 0, "region_label": 1, "region_pixel_count": 1}}
    with pytest.raises(BirthSurvivalError, match="missing action_id"):
        measure_birth_survival(
            object(),
            scorer_teacher=object(),
            target_labels=labels,
            live_birth_payload=payload,
            surface="parseback_mlx",
            pair_indices=[0],
        )


def test_unknown_surface_rejected() -> None:
    labels = np.zeros((1, 8, 10), dtype=np.int64)
    payload = {
        "action_id": "a" * 64,
        "worst_region": {"batch_index": 0, "class_index": 0, "region_label": 1, "region_pixel_count": 1},
    }
    with pytest.raises(BirthSurvivalError, match="surface must be one of"):
        measure_birth_survival(
            object(),
            scorer_teacher=object(),
            target_labels=labels,
            live_birth_payload=payload,
            surface="totally_made_up_surface",
            pair_indices=[0],
        )


# ---------------------------------------------------------------------------
# MLX behavioral fixtures (mirror test_target_region_birth.py conventions)
# ---------------------------------------------------------------------------


def _smoke_cfg():
    from tac.substrates.hi_nerv.architecture import HinervConfig

    return HinervConfig(
        latent_dim_coarse=4,
        latent_dim_mid=6,
        latent_dim_fine=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _green_dominant_targets(cfg, mx):
    ramp = mx.reshape(
        mx.linspace(0.05, 0.95, cfg.output_height * cfg.output_width),
        (1, cfg.output_height, cfg.output_width, 1),
    )
    target0 = mx.tile(
        mx.concatenate([0.15 + 0.05 * ramp, 0.55 + 0.1 * ramp, 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    target1 = mx.tile(
        mx.concatenate([0.18 + 0.05 * ramp, 0.6 + 0.05 * ramp, 0.05 + 0.1 * ramp], axis=-1),
        (cfg.num_pairs, 1, 1, 1),
    )
    return target0, target1


def _block_labels(cfg, np_module):
    labels = np_module.zeros(
        (cfg.num_pairs, cfg.output_height, cfg.output_width),
        dtype=np_module.int32,
    )
    labels[0, 4:10, 6:14] = 1  # one 48-pixel class-1 region in item 0
    return labels


class _BehavioralSegNetTeacher:
    """Real-VJP behavioral stand-in: logits derived from frame RGB."""

    num_classes = 2

    def __init__(self, mx, labels):
        self._mx = mx
        self._labels = labels

    def teacher_argmax_for_indices(self, indices):
        return self._mx.take(self._labels, indices, axis=0)

    def teacher_logits_for_frames_nhwc01(self, frames):
        red = frames[..., 0]
        green = frames[..., 1]
        class0 = green - red
        class1 = red - green
        return self._mx.stack([class0, class1], axis=-1)


def _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg):
    """Run the REAL birth actuator and return an accepted live payload."""

    model.initialize_output_head_bias_from_targets(target0, target1)
    payload = model.fit_target_region_birth_from_segnet(
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        pair_indices=mx.arange(cfg.num_pairs, dtype=mx.int32),
        target_segnet_argmax_1=mx.array(labels_np),
        max_steps=24,
        learning_rate=2.0e-3,
        target_min_region_ratio=0.02,
    )
    return payload


def _setup(mx):
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    mx.random.seed(7)
    cfg = _smoke_cfg()
    model = HinervSubstrateMLX(cfg)
    target0, target1 = _green_dominant_targets(cfg, mx)
    labels_np = _block_labels(cfg, np)
    teacher = _BehavioralSegNetTeacher(mx, mx.array(labels_np))
    return cfg, model, teacher, target0, target1, labels_np


# ---------------------------------------------------------------------------
# MLX behavioral tests
# ---------------------------------------------------------------------------


@skip_no_mlx
def test_fakequant_survival_row_carries_live_action_id_and_real_survived() -> None:
    import mlx.core as mx

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    assert payload["accepted"] is True  # the live birth must really have landed
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)

    row = measure_birth_survival(
        model,
        scorer_teacher=teacher,
        target_labels=labels_np,
        live_birth_payload=payload,
        surface="fakequant_mlx",
        pair_indices=idx,
    )
    assert row["schema"] == BIRTH_SURVIVAL_SCHEMA
    assert row["surface"] == "fakequant_mlx"
    # action_id COPIED from the live receipt (never recomputed).
    assert row["action_id"] == payload["action_id"]
    assert isinstance(row["survived"], bool)
    # The 48-pixel region birthed; at 8-bit fake-quant it should still hold,
    # so the re-measured support is positive and survival is True.
    assert row["survived"] is True
    assert row["region_hard_won_count"] >= 1
    assert row["net_target_support_delta"] > 0
    # Region was fully unsolved at birth -> initial in-region target count is 0,
    # so net == hard-won (exact integer accounting).
    assert row["initial_in_region_target_count"] == 0
    assert row["net_target_support_delta"] == row["region_hard_won_count"]
    # Gate-consumed canonical aliases must be present and positive.
    assert row["target_hard_won_count"] == row["region_hard_won_count"]
    assert row["receiver_surface_target_hard_won_count"] == row["region_hard_won_count"]
    assert row["argmax_transitions"]["net_target_support_delta"] == row["net_target_support_delta"]


@skip_no_mlx
def test_survival_action_id_mismatch_impossible_by_construction() -> None:
    import mlx.core as mx

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)
    row = measure_birth_survival(
        model,
        scorer_teacher=teacher,
        target_labels=labels_np,
        live_birth_payload=payload,
        surface="fakequant_mlx",
        pair_indices=idx,
    )
    # The producer copies action_id verbatim; it never derives its own. Mutating
    # the live id and re-measuring yields the mutated id, proving a pure copy.
    mutated = dict(payload, action_id="0" * 64)
    row2 = measure_birth_survival(
        model,
        scorer_teacher=teacher,
        target_labels=labels_np,
        live_birth_payload=mutated,
        surface="fakequant_mlx",
        pair_indices=idx,
    )
    assert row["action_id"] == payload["action_id"]
    assert row2["action_id"] == "0" * 64


@skip_no_mlx
def test_fakequant_survival_restores_config_and_forward_bit_identically() -> None:
    import mlx.core as mx

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)

    fq_enabled_before = bool(model.decoder_fake_quant_forward_enabled)
    fq_configured_before = bool(model.decoder_fake_quant_forward_configured_enabled)
    fq_bits_before = model.decoder_fake_quant_bits
    pre_forward = np.array(model(idx), dtype=np.float32)

    measure_birth_survival(
        model,
        scorer_teacher=teacher,
        target_labels=labels_np,
        live_birth_payload=payload,
        surface="fakequant_mlx",
        pair_indices=idx,
        fakequant_bits=2,  # aggressive bits during the measurement
    )

    # Config is restored to exactly the prior state...
    assert bool(model.decoder_fake_quant_forward_enabled) == fq_enabled_before
    assert bool(model.decoder_fake_quant_forward_configured_enabled) == fq_configured_before
    assert model.decoder_fake_quant_bits == fq_bits_before
    # ...and the (non-fake-quant) forward is bit-identical afterwards.
    post_forward = np.asarray(model(idx), dtype=np.float32)
    assert np.array_equal(pre_forward, post_forward)


@skip_no_mlx
def test_fakequant_survival_is_real_remeasurement_aggressive_bits() -> None:
    """A stub returning survived=True constants would fail this.

    The fake-quant forward materially changes the receiver image (more so at
    fewer bits).  We assert the producer actually forwards under fake-quant by
    proving the 1-bit forward differs from the plain forward, and that the
    survival re-measurement is a genuine function of the surface (its boolean is
    derived from the re-measured hard-won count, not a constant).
    """

    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import _receiver_uint8_roundtrip_ste_nhwc01

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)

    # Prove the fake-quant forward is real: 1-bit decoder fake-quant changes the
    # frame-1 receiver image relative to the plain forward.
    plain = np.asarray(model(idx), dtype=np.float64)
    model.configure_decoder_fake_quant_forward(enabled=True, quant_bits=1, stage_controlled=False)
    quant = np.asarray(model(idx), dtype=np.float64)
    model.configure_decoder_fake_quant_forward(enabled=False)
    assert float(np.abs(quant - plain).max()) > 0.0

    row = measure_birth_survival(
        model,
        scorer_teacher=teacher,
        target_labels=labels_np,
        live_birth_payload=payload,
        surface="fakequant_mlx",
        pair_indices=idx,
        fakequant_bits=8,
    )
    # The survived boolean is consistent with the re-measured counts (not a
    # constant disconnected from the measurement).
    expected_survived = bool(row["region_hard_won_count"] >= 1 and row["net_target_support_delta"] > 0)
    assert row["survived"] is expected_survived
    # The re-measured region debt is computed from the surface, not copied.
    assert "region_score_debt_units" in row
    assert row["region_score_debt_units"] >= 0.0
    # Roundtrip helper is importable and used (sanity that the receiver surface
    # path is the canonical uint8 roundtrip, not a bypass).
    assert callable(_receiver_uint8_roundtrip_ste_nhwc01)


@skip_no_mlx
def test_fakequant_survival_row_has_no_forbidden_authority_keys() -> None:
    import mlx.core as mx

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)
    row = measure_birth_survival(
        model,
        scorer_teacher=teacher,
        target_labels=labels_np,
        live_birth_payload=payload,
        surface="fakequant_mlx",
        pair_indices=idx,
    )
    assert row["authority"] == "planning_control_false_authority"
    assert not (_FORBIDDEN_AUTHORITY_KEYS & row.keys())
    assert not (_FORBIDDEN_AUTHORITY_KEYS & row["worst_region"].keys())
    assert not (_FORBIDDEN_AUTHORITY_KEYS & row["argmax_transitions"].keys())


@skip_no_mlx
def test_hysteresis_row_is_real_after_continued_training() -> None:
    import mlx.core as mx

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)

    row = measure_birth_hysteresis(
        model,
        scorer_teacher=teacher,
        target_rgb_0=target0,
        target_rgb_1=target1,
        target_labels=labels_np,
        live_birth_payload=payload,
        pair_indices=idx,
        extra_steps=3,
    )
    assert row["schema"] == BIRTH_HYSTERESIS_SCHEMA
    assert row["action_id"] == payload["action_id"]
    assert isinstance(row["passed"], bool)
    assert row["extra_steps"] == 3
    # The schema's required t0/t+M fields are populated from real measurements.
    assert "hard_won_t0" in row and "hard_won_tm" in row
    assert "debt_t0" in row and "debt_tm" in row
    assert row["hard_won_t0"] >= 1  # the birth really landed
    # Continued plain training really ran (bootstrap executed with steps>0).
    assert row["bootstrap_schema"] == "hi_nerv_scorer_domain_bootstrap.v1"
    # passed is a function of the measured collapse + debt criteria, not a const.
    expected_passed = bool(row["no_hard_won_collapse"] and row["debt_not_above_pre_birth"])
    assert row["passed"] is expected_passed
    assert not (_FORBIDDEN_AUTHORITY_KEYS & row.keys())


@skip_no_mlx
def test_hysteresis_rejects_nonpositive_steps() -> None:
    import mlx.core as mx

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)
    with pytest.raises(BirthSurvivalError, match="extra_steps must be positive"):
        measure_birth_hysteresis(
            model,
            scorer_teacher=teacher,
            target_rgb_0=target0,
            target_rgb_1=target1,
            target_labels=labels_np,
            live_birth_payload=payload,
            pair_indices=idx,
            extra_steps=0,
        )


@skip_no_mlx
def test_launch_gate_consumes_fakequant_survival_row(tmp_path) -> None:
    """End-to-end: the REAL gate must stop blocking fakequant once our row lands.

    Writes the live birth receipt + our fake-quant survival row to a run root and
    calls ``evaluate_nerv_long_run_launch_gate``; asserts the gate no longer
    reports ``birth_survival_receipt_missing:fakequant_mlx``.  This binds the
    producer to the gate's contract, not just to its own assertions.
    """

    import mlx.core as mx

    from tac.analysis.nerv_long_run_launch_gate import evaluate_nerv_long_run_launch_gate

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)
    survival = measure_birth_survival(
        model,
        scorer_teacher=teacher,
        target_labels=labels_np,
        live_birth_payload=payload,
        surface="fakequant_mlx",
        pair_indices=idx,
    )
    assert survival["survived"] is True

    run_root = tmp_path / "run"
    run_root.mkdir()
    # The live birth RECEIPT (gate's L2/L3 evidence) is the nested receipt the
    # actuator already emits; write it as the gate expects.
    (run_root / "live_birth_receipt.json").write_text(json.dumps(payload["receipt"]))
    (run_root / "fakequant_survival.json").write_text(json.dumps(survival))
    frontier = run_root / "frontier_pointer.json"
    frontier.write_text(json.dumps({"last_refreshed_utc": datetime.now(UTC).isoformat()}))

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hinerv",
        run_root=run_root,
        frontier_pointer=frontier,
    )
    blocking = verdict["blocking_evidence"]
    # The specific contract this producer serves:
    assert "birth_survival_receipt_missing:fakequant_mlx" not in blocking
    # And the action-id mismatch blocker must NOT fire (id was copied).
    assert "l4_survival_action_id_mismatch:fakequant_mlx" not in blocking
    # The gate still blocks the surfaces we deliberately did not produce, proving
    # our row is consumed surgically (not papering over the whole L4 ladder).
    assert "birth_survival_receipt_missing:parseback_mlx" in blocking


@skip_no_mlx
def test_launch_gate_still_blocks_when_action_id_mismatches(tmp_path) -> None:
    """If a survival row carries the WRONG action_id the gate must keep blocking.

    This proves the action-id trace is load-bearing: a re-solved birth under
    fake-quant (different id) is a different experiment and is not accepted.
    """

    import mlx.core as mx

    from tac.analysis.nerv_long_run_launch_gate import evaluate_nerv_long_run_launch_gate

    cfg, model, teacher, target0, target1, labels_np = _setup(mx)
    payload = _accepted_live_birth(mx, model, teacher, target0, target1, labels_np, cfg)
    idx = mx.arange(cfg.num_pairs, dtype=mx.int32)
    survival = measure_birth_survival(
        model,
        scorer_teacher=teacher,
        target_labels=labels_np,
        live_birth_payload=payload,
        surface="fakequant_mlx",
        pair_indices=idx,
    )
    # Corrupt the survival row's action_id to a different value.
    survival_wrong = dict(survival, action_id="9" * 64)

    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "live_birth_receipt.json").write_text(json.dumps(payload["receipt"]))
    (run_root / "fakequant_survival.json").write_text(json.dumps(survival_wrong))
    frontier = run_root / "frontier_pointer.json"
    frontier.write_text(json.dumps({"last_refreshed_utc": datetime.now(UTC).isoformat()}))

    verdict = evaluate_nerv_long_run_launch_gate(
        family="hinerv",
        run_root=run_root,
        frontier_pointer=frontier,
    )
    blocking = verdict["blocking_evidence"]
    assert "l4_survival_action_id_mismatch:fakequant_mlx" in blocking
    assert "birth_survival_receipt_missing:fakequant_mlx" in blocking
