# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_g1_worldsheet import (
    encode_g1_movable_worldsheet,
    encode_lifted_g1_movable_worldsheet,
    lift_g1_movable_worldsheet,
)
from tac.optimization.direct_description_joint_descent import (
    EXPECTED_PROGRAM_SHA256,
    AdamStateV1,
    DirectDescriptionJointDescentTypedConfigV1,
    classify_memory_preflight,
    clipped_adam_step,
    initial_adam_state,
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError

REPO = Path(__file__).resolve().parents[4]
TICKET = REPO / ".omx/research/configs/ddm_j1_366_joint_descent_witness_program_20260723.json"


def test_g1_lift_preserves_exact_stream_and_explicit_lifecycle() -> None:
    labels = np.zeros((5, 384, 512), dtype=np.int64)
    labels[1, 40:48, 60:72] = 3
    labels[2, 41:49, 62:74] = 3
    labels[3, 42:50, 64:76] = 3
    payload, _ = encode_g1_movable_worldsheet(labels)

    lift = lift_g1_movable_worldsheet(payload)

    assert encode_lifted_g1_movable_worldsheet(lift) == payload
    assert len(lift.tracks) == 1
    assert (lift.tracks[0].birth_pair, lift.tracks[0].death_pair_exclusive) == (1, 4)
    assert len(lift.tracks[0].knot_indices) == 3
    assert all(knot.template_ref for knot in lift.knots)
    assert all(np.isfinite(knot.aspect_log) and np.isfinite(knot.rotation_radians) for knot in lift.knots)


def test_hash_sealed_ticket_compiles_to_typed_config() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(TICKET)
    assert config.dsl_compile_hash == EXPECTED_PROGRAM_SHA256
    assert config.num_pairs == 600
    assert config.seed == 0
    assert config.custom_grouped_backward_required is True
    assert config.fused_r_required is True
    assert config.score_claim is False
    assert config.research_only is True


def test_typed_config_refuses_semantic_ticket_mutation(tmp_path: Path) -> None:
    ticket = json.loads(TICKET.read_bytes())
    ticket["semantic_program"]["seed"] = 1
    mutated = tmp_path / "mutated_ticket.json"
    mutated.write_text(json.dumps(ticket), encoding="utf-8")
    with pytest.raises(DirectDescriptionError, match="DSL hash mismatch"):
        DirectDescriptionJointDescentTypedConfigV1.from_ticket(mutated)


@pytest.mark.parametrize(
    ("peak", "admit", "reason"),
    [
        (115.999, True, "SAFE_PROJECTED_PEAK_WITHIN_116_GIB_CEILING"),
        (116.001, False, "REFUSE_PROJECTED_PEAK_EXCEEDS_116_GIB_CEILING"),
        (float("nan"), False, "REFUSE_INVALID_MEASURED_PEAK"),
    ],
)
def test_memory_preflight_is_fail_closed(peak: float, admit: bool, reason: str) -> None:
    assert classify_memory_preflight(peak) == (admit, reason)


def test_adam_checkpoint_is_atomic_preserved_and_bit_exact(tmp_path: Path) -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(TICKET)
    initial = initial_adam_state(7)
    gradient = np.linspace(-0.2, 0.2, 7, dtype=np.float32)
    stepped = clipped_adam_step(
        initial,
        gradient,
        learning_rate=0.05,
        grad_clip=config.grad_clip,
        ema_decay=config.ema_decay,
    )
    path = tmp_path / "stage00_step000001.npz"
    checkpoint_sha = save_stage_checkpoint(
        path,
        stepped,
        stage_id="00_receiver_replay_and_adapter",
        config=config,
        telemetry=({"event": "unit_resume_boundary", "score_claim": False},),
    )

    loaded, metadata = load_stage_checkpoint(path, config=config)

    assert len(checkpoint_sha) == 64
    with np.load(path, allow_pickle=False) as archive:
        assert "__resume_registry_manifest" in archive.files
        assert "__ddmjd_optimizer_state_sha256" in archive.files
    assert metadata["ema_shadow_saved"] is True
    assert metadata["rng"] == {"kind": "deterministic_no_sampling", "state": 0}
    assert metadata["canonical_resume_registry"]["controller"] == "ddm_joint_descent_optimizer"
    for field in ("theta", "ema", "first_moment", "second_moment"):
        assert np.array_equal(getattr(stepped, field), getattr(loaded, field))
    assert loaded.step == stepped.step
    with pytest.raises(DirectDescriptionError, match="already exists"):
        save_stage_checkpoint(
            path,
            stepped,
            stage_id="00_receiver_replay_and_adapter",
            config=config,
            telemetry=(),
        )

    corrupt_path = tmp_path / "stage00_step000001_corrupt.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["theta"][0] += np.float32(1.0)
    np.savez(corrupt_path, **arrays)
    with pytest.raises(DirectDescriptionError, match="optimizer state hash differs"):
        load_stage_checkpoint(corrupt_path, config=config)


def test_adam_resume_continuation_matches_uninterrupted_bits() -> None:
    state = initial_adam_state(5)
    gradient = np.asarray((0.2, -0.1, 0.05, -0.02, 0.3), dtype=np.float32)
    first = clipped_adam_step(state, gradient, learning_rate=0.01, grad_clip=0.5, ema_decay=0.997)
    resumed = AdamStateV1(
        step=first.step,
        theta=first.theta.copy(),
        ema=first.ema.copy(),
        first_moment=first.first_moment.copy(),
        second_moment=first.second_moment.copy(),
    )
    uninterrupted = clipped_adam_step(first, gradient, learning_rate=0.01, grad_clip=0.5, ema_decay=0.997)
    continued = clipped_adam_step(resumed, gradient, learning_rate=0.01, grad_clip=0.5, ema_decay=0.997)
    for field in ("theta", "ema", "first_moment", "second_moment"):
        assert np.array_equal(getattr(uninterrupted, field), getattr(continued, field))
