from __future__ import annotations

import json

import numpy as np
import pytest

from tac.optimization.direct_description_measurement_ladder import (
    _ANCHOR_RECORD,
    _GRADIENT_RECORD,
    _POSE_RECORD,
    _RESIDUAL_RECORD,
    STREAM_ORDER,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
    compile_chart_archive,
    receive_chart_archive,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.direct_description_receiver_priced_member import (
    TOLERANCE_LADDER,
    DirectDescriptionReceiverPricedMemberCheckpointV1,
    DirectDescriptionReceiverPricedMemberConfigV1,
    build_safe_zero_residual_proposal,
    exact_rate_probe_rows,
    run_receiver_priced_member_stages,
)


def _synthetic_z(n_pairs: int = 64) -> DirectDescriptionChartZV1:
    bodies = {name: bytearray() for name in STREAM_ORDER}
    for pair_id in range(n_pairs):
        for plane_id in range(2):
            bodies["global_chart_anchors"].extend(_ANCHOR_RECORD.pack(pair_id, plane_id, 96, 112, 128))
            bodies["axial_chart_gradients"].extend(_GRADIENT_RECORD.pack(pair_id, plane_id, 0, 0, 0, 0, 0, 0))
            for stratum_index, stream_name in enumerate(STREAM_ORDER[2:5]):
                for chart_id in range(stratum_index * 64, (stratum_index + 1) * 64):
                    bodies[stream_name].extend(_RESIDUAL_RECORD.pack(pair_id, plane_id, chart_id, 1, -1, 2))
        bodies["pose6_pair_codes"].extend(_POSE_RECORD.pack(pair_id, 0, 1, 2, 3, 4, 5))
    return DirectDescriptionChartZV1(
        n_pairs=n_pairs,
        **{name: CountedChartStreamV1(payload=bytes(bodies[name])) for name in STREAM_ORDER},
    )


def _membership(receiver: object) -> dict[str, object]:
    n_pairs = receiver.z.n_pairs  # type: ignore[attr-defined]
    rows = [{"pair_id": pair_id} for pair_id in range(n_pairs)]
    counts = {
        "sites": n_pairs * 384 * 512,
        "argmax_cell_escape_fraction": "0.500000000000",
        "same_c1_argmax_cell_fraction": "0.500000000000",
    }
    return {
        "same_c1_argmax_cell_fraction": "0.500000000000",
        "argmax_cell_escape_fraction": "0.500000000000",
        "strata": {"overall": {"all": counts}},
        "per_pair": rows,
        "per_pair_rows_sha256": "0" * 64,
    }


def _config() -> DirectDescriptionReceiverPricedMemberConfigV1:
    return DirectDescriptionReceiverPricedMemberConfigV1(
        scorer_threads=4,
        target_receipt_path="target.json",
        target_receipt_sha256="1" * 64,
        upstream_root="/abs/upstream",
    )


def _pose_codes() -> np.ndarray:
    values = np.zeros((600, 6), dtype=np.uint8)
    values[:, :] = np.arange(6, dtype=np.uint8)
    return values


def test_safe_zero_proposals_are_real_receiver_encodes_but_byte_flat() -> None:
    z = _synthetic_z()
    baseline = compile_chart_archive(z).archive
    proposal = build_safe_zero_residual_proposal(z, "low_variation_chart_residuals")
    encoded = compile_chart_archive(proposal.z).archive
    assert proposal.changed_scalars > 0
    assert encoded != baseline
    assert len(encoded) == len(baseline)
    assert receive_chart_archive(encoded).archive == encoded
    rows = exact_rate_probe_rows(z)
    assert len(rows) == 3
    assert {row["delta_archive_bytes"] for row in rows} == {0}
    assert {row["accepted"] for row in rows} == {False}
    assert all(row["receiver_consumed"] for row in rows)


def test_tolerance_ladder_runs_inside_selection_and_resumes(tmp_path) -> None:
    config = _config()
    z = _synthetic_z()
    argv = ("python3", "tools/run_direct_description_receiver_priced_member.py")
    partial = run_receiver_priced_member_stages(
        config,
        baseline_z=z,
        target_pose_codes=_pose_codes(),
        membership_measure=_membership,
        semantic_argv=argv,
        checkpoint_directory=tmp_path,
        stop_after_rung_index=1,
    )
    assert partial.complete is False
    assert len(partial.curve) == 2
    resumed = run_receiver_priced_member_stages(
        config,
        baseline_z=z,
        target_pose_codes=_pose_codes(),
        membership_measure=_membership,
        semantic_argv=argv,
        checkpoint_directory=tmp_path,
        resume_from=partial.checkpoint_paths[-1],
    )
    assert resumed.complete is True
    assert resumed.resumed is True
    assert len(resumed.curve) == len(TOLERANCE_LADDER)
    assert {row["archive_bytes"] for row in resumed.curve} == {len(partial.final_archive)}
    assert {row["membership_fraction"] for row in resumed.curve} == {"0.500000000000"}
    assert {row["pose_completeness"] for row in resumed.curve} == {"1.000000000000"}
    assert all(row["rung_feasible"] is False for row in resumed.curve)
    assert all(row["source_raw_reference_used"] is False for row in resumed.curve)


def test_checkpoint_is_canonical_and_tamper_evident(tmp_path) -> None:
    partial = run_receiver_priced_member_stages(
        _config(),
        baseline_z=_synthetic_z(),
        target_pose_codes=_pose_codes(),
        membership_measure=_membership,
        semantic_argv=("python3", "tool.py"),
        checkpoint_directory=tmp_path,
        stop_after_rung_index=0,
    )
    payload = partial.checkpoint_paths[-1].read_bytes()
    parsed = DirectDescriptionReceiverPricedMemberCheckpointV1.from_bytes(payload)
    assert parsed.completed_rung_index == 0
    envelope = json.loads(payload)
    envelope["body"]["selected_archive_bytes"] += 1
    tampered = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises((DirectDescriptionError, ValueError)):
        DirectDescriptionReceiverPricedMemberCheckpointV1.from_bytes(tampered)


def test_config_refuses_tolerance_or_relative_scorer_drift() -> None:
    with pytest.raises(ValueError):
        DirectDescriptionReceiverPricedMemberConfigV1(
            scorer_threads=4,
            target_receipt_path="target.json",
            target_receipt_sha256="1" * 64,
            upstream_root="relative/upstream",
        )
    with pytest.raises(ValueError):
        DirectDescriptionReceiverPricedMemberConfigV1(
            scorer_threads=4,
            target_receipt_path="target.json",
            target_receipt_sha256="1" * 64,
            upstream_root="/abs/upstream",
            tolerance_ladder=("0.000000", "0.1"),
        )
