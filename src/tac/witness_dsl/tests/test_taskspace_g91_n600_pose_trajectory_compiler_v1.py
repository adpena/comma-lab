# SPDX-License-Identifier: MIT
"""Exact fresh-trajectory, coder, and measured-selector proofs for G91."""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.xi_pose_coder import parse_xi_payload
from tac.witness_dsl import taskspace_g91_n600_pose_trajectory_compiler_v1 as g91

_G85_MEMBER = Path("/Volumes/VertigoDataTier/pact/g85_pvsa_public_receiver_20260727_r1/archive/0.bin")
_G85_MEMBER_SHA256 = "d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31"
_SEMANTIC_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _poses() -> np.ndarray:
    pair = np.arange(600, dtype=np.float64)
    result = np.zeros((600, 6), dtype=np.float64)
    result[:, 0] = 25.0 + 0.01 * pair + 0.4 * np.sin(pair / 13.0)
    result[:, 1] = 0.03 * np.cos(pair / 17.0)
    result[:, 2] = 0.02 * np.sin(pair / 19.0)
    result[:, 3] = 0.001 * np.sin(pair / 23.0)
    result[:, 4] = -0.01 + 0.002 * np.cos(pair / 29.0)
    result[:, 5] = 0.001 * np.cos(pair / 31.0)
    return result


def _treatment(rank: int = 2) -> g91.G91TrajectoryTreatmentV1:
    return g91.G91TrajectoryTreatmentV1(
        treatment_id=f"unit_rank{rank}",
        s_t=0.044,
        s_r=0.0,
        pitch=0.0,
        centered_rank=rank,
        q_levels=4096,
    )


def _member() -> bytes:
    if not _G85_MEMBER.is_file():
        pytest.skip("retained exact G85 base member is absent")
    payload = _G85_MEMBER.read_bytes()
    assert len(payload) == 133_363
    assert _sha(payload) == _G85_MEMBER_SHA256
    return payload


def test_fresh_trajectory_is_deterministic_ranked_and_quantized() -> None:
    poses = _poses()
    first = g91.derive_fresh_trajectory(poses, _treatment(rank=2))
    second = g91.derive_fresh_trajectory(poses.copy(), _treatment(rank=2))

    assert first.source_target_sha256 == second.source_target_sha256
    assert np.array_equal(first.calibrated_xi, second.calibrated_xi)
    assert np.array_equal(first.factorized_xi, second.factorized_xi)
    assert np.array_equal(first.q_codes, second.q_codes)
    assert np.array_equal(first.scales, second.scales)
    assert first.q_codes.shape == (600, 6)
    assert first.q_codes.dtype == np.int16
    assert first.scales.dtype == np.float32
    assert first.factorability["selected_centered_rank"] == 2
    assert len(first.factorability["rank_curve"]) == 6
    assert first.factorability["rank_curve"][1]["relative_frobenius_error"] < 0.01
    assert first.factorability["int16_expanded_bytes"] == 7_224
    assert first.factorability["quantization_rmse"] >= 0.0


def test_mode_selection_dataclass_fields_are_unique_and_exact() -> None:
    names = tuple(field.name for field in fields(g91.G91ModeSelectionRowV1))
    assert names == (
        "default_mode",
        "xip2_pair_ids",
        "pass_pair_ids",
        "coder_row",
        "mean_d_pose",
        "pose_score_term",
        "rate_score_term",
        "selector_objective",
        "control_rows",
    )
    assert names.count("pass_pair_ids") == 1
    assert g91.INVERSE_CONTROL_SOLVED is False
    assert g91.DIRECT_POSE_TARGET_AS_WARP_CONTROL_ADMISSIBLE is False
    assert g91.ONLY_ADMISSIBLE_PROMOTION_PATH.startswith("G95_POSENET_IN_LOOP_INVERSE_SOLVE")


def test_coder_race_is_exact_eof_bound_to_g88_base_and_same_q() -> None:
    trajectory = g91.derive_fresh_trajectory(_poses(), _treatment(rank=2))
    member = _member()
    race = g91.compile_coder_race(
        trajectory=trajectory,
        base_pvsa_member_bytes=member,
        semantic_p_sha256=_SEMANTIC_SHA256,
        default_mode="XIP2_SE3_FRAME0_WARP",
        xip2_pair_ids=tuple(range(600)),
        coders=("none", "delta_res"),
    )

    assert len(race) == 2
    assert race[0].selected_outer_bytes <= race[1].selected_outer_bytes
    for row in race:
        q, scales = parse_xi_payload(row.xip2_payload)
        assert np.array_equal(q, trajectory.q_codes)
        assert np.array_equal(scales, trajectory.scales)
        assert row.parsed_operand.transport is not None
        assert row.parsed_operand.transport.predictor_program_sha256 == _G85_MEMBER_SHA256
        assert row.archive_build.stored == row.archive_build.deflated
        assert row.archive_build.selected.conditional_operand.to_bytes() == (row.parsed_operand.to_bytes())
    forged_receipt_fields = {
        "xip2_sha256": "0" * 64,
        "operand_bytes": race[0].operand_bytes + 1,
        "operand_sha256": "0" * 64,
        "successor_member_bytes": race[0].successor_member_bytes + 1,
        "successor_member_sha256": "0" * 64,
        "stored_outer_bytes": race[0].stored_outer_bytes + 1,
        "stored_outer_sha256": "0" * 64,
        "deflated_outer_bytes": race[0].deflated_outer_bytes + 1,
        "deflated_outer_sha256": "0" * 64,
        "selected_outer_bytes": race[0].selected_outer_bytes + 1,
        "selected_outer_sha256": "0" * 64,
        "selected_outer_encoding": "zip_stored"
        if race[0].selected_outer_encoding == "zip_deflated"
        else "zip_deflated",
    }
    for label, forged in forged_receipt_fields.items():
        with pytest.raises(g91.G91PoseTrajectoryError):
            replace(race[0], **{label: forged})


def test_measured_sparse_selector_uses_actual_pair_losses_and_exact_outer_bytes() -> None:
    trajectory = g91.derive_fresh_trajectory(_poses(), _treatment(rank=2))
    pair = np.arange(600)
    pass_loss = (2.0 + pair / 1000.0).astype(np.float64)
    xip2_loss = pass_loss.copy()
    xip2_loss[:300] -= np.linspace(0.001, 0.3, 300)
    xip2_loss[300:] += np.linspace(0.001, 0.2, 300)
    rows = g91.compile_measured_mode_selection(
        trajectory=trajectory,
        base_pvsa_member_bytes=_member(),
        semantic_p_sha256=_SEMANTIC_SHA256,
        pass_d_pose=pass_loss,
        xip2_d_pose=xip2_loss,
        coders=("delta_res",),
    )

    assert rows
    best = rows[0]
    assert len(best.xip2_pair_ids) + len(best.pass_pair_ids) == 600
    assert not (set(best.xip2_pair_ids) & set(best.pass_pair_ids))
    chosen = pass_loss.copy()
    chosen[np.asarray(best.xip2_pair_ids)] = xip2_loss[np.asarray(best.xip2_pair_ids)]
    assert best.mean_d_pose == pytest.approx(float(chosen.mean()), abs=0.0)
    assert best.pose_score_term == pytest.approx(np.sqrt(10.0 * chosen.mean()))
    assert best.rate_score_term == pytest.approx(
        25.0 * best.coder_row.selected_outer_bytes / g91.RATE_DENOMINATOR_BYTES
    )
    assert best.selector_objective == pytest.approx(best.pose_score_term + best.rate_score_term)
    assert any(row.default_mode == "PASS_P0" for row in rows)
    assert any(row.default_mode == "XIP2_SE3_FRAME0_WARP" for row in rows)
    assert len(rows) == 1_200
    assert {len(row.pass_pair_ids) for row in rows if row.default_mode == "XIP2_SE3_FRAME0_WARP"} == set(range(600))
    assert {len(row.xip2_pair_ids) for row in rows if row.default_mode == "PASS_P0"} == set(range(1, 601))


def test_full_prefix_objective_can_cross_zero_gain_boundary_when_exact_outer_bytes_win() -> None:
    """A negative local-gain row may enter the optimum when exact ZIP bytes fall."""

    ordinary_exact_bytes = 132_000 + len(zlib.compress(bytes(range(256)) * 64, level=9))
    crossing_exact_bytes = 132_000 + len(zlib.compress(b"\x00" * 16_384, level=9))
    assert crossing_exact_bytes < ordinary_exact_bytes

    pass_loss = np.ones(600, dtype=np.float64)
    xip2_gain = np.concatenate(
        (
            np.asarray([1.0e-6, 5.0e-7], dtype=np.float64),
            -np.arange(1, 599, dtype=np.float64) * 1.0e-12,
        )
    )
    xip2_loss = pass_loss - xip2_gain
    prefix_objectives: list[float] = []
    for count in range(1, 601):
        chosen = pass_loss.copy()
        chosen[:count] = xip2_loss[:count]
        exact_bytes = crossing_exact_bytes if count == 3 else ordinary_exact_bytes
        prefix_objectives.append(
            g91.exact_prefix_objective(
                mean_d_pose=float(chosen.mean()),
                exact_outer_bytes=exact_bytes,
            )[2]
        )

    assert np.count_nonzero(xip2_gain > 0.0) == 2
    assert int(np.argmin(prefix_objectives)) + 1 == 3


def test_all_worse_executable_rows_preserve_exact_base_without_empty_xip2() -> None:
    trajectory = g91.derive_fresh_trajectory(_poses(), _treatment(rank=2))
    coder = g91.compile_coder_race(
        trajectory=trajectory,
        base_pvsa_member_bytes=_member(),
        semantic_p_sha256=_SEMANTIC_SHA256,
        default_mode="PASS_P0",
        xip2_pair_ids=tuple(range(600)),
        coders=("delta_res",),
    )[0]
    xip2_loss = np.full(600, 2.0, dtype=np.float64)
    pose_term, rate_term, objective = g91.exact_prefix_objective(
        mean_d_pose=float(xip2_loss.mean()),
        exact_outer_bytes=coder.selected_outer_bytes,
    )
    all_worse = g91.G91ModeSelectionRowV1(
        default_mode="PASS_P0",
        xip2_pair_ids=tuple(range(600)),
        pass_pair_ids=(),
        coder_row=coder,
        mean_d_pose=float(xip2_loss.mean()),
        pose_score_term=pose_term,
        rate_score_term=rate_term,
        selector_objective=objective,
        control_rows=len(coder.parsed_operand.controls),
    )
    decision = g91.select_measured_mode_or_base(
        rows=(all_worse,),
        pass_d_pose=np.ones(600, dtype=np.float64),
        base_outer_bytes=129_392,
    )

    assert decision is None
    assert all_worse.coder_row.xip2_payload.startswith(b"XIP2")


def test_refuses_wrong_population_shape_and_empty_xip2_allocation() -> None:
    with pytest.raises(g91.G91PoseTrajectoryError, match=r"float64 \[600,6\]"):
        g91.derive_fresh_trajectory(np.zeros((599, 6), dtype=np.float64), _treatment())
    trajectory = g91.derive_fresh_trajectory(_poses(), _treatment())
    with pytest.raises(g91.G91PoseTrajectoryError, match="at least one pair"):
        g91.compile_coder_race(
            trajectory=trajectory,
            base_pvsa_member_bytes=_member(),
            semantic_p_sha256=_SEMANTIC_SHA256,
            default_mode="PASS_P0",
            xip2_pair_ids=(),
            coders=("none",),
        )
