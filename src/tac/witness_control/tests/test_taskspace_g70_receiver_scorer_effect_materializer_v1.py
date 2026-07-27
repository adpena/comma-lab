# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control import taskspace_g70_receiver_scorer_effect_materializer_v1 as g70
from tac.witness_control.taskspace_g70_receiver_scorer_effect_materializer_v1 import (
    DERIVATIVE_JOIN_BLOCKER,
    G70EffectMaterializationError,
    audit_vjp_campaign_for_transition,
    materialize_g70_finite_effect_transition,
    require_g70_actionable_costate_input,
    write_g70_finite_effect_bundle,
)
from tac.witness_dsl.taskspace_g17_forward_observation import (
    G17CandidateForwardObservationV1,
    G17TargetForwardObservationV1,
)


def _zip(member: bytes, *, name: str = "payload.bin") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, member)
    return output.getvalue()


def _zip_with_duplicate_content(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        for name in ("first.bin", "second.bin"):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, member)
    return output.getvalue()


def _target() -> G17TargetForwardObservationV1:
    pair_ids = tuple(range(600))
    camera = np.zeros((600, 2, 1, 1, 3), dtype=np.uint8)
    numerators = np.zeros((600, 2, 1, 1, 3), dtype=np.float32)
    denominators = np.ones((600, 2, 1, 1, 3), dtype=np.float32)
    projected = np.zeros((600, 2, 1, 1, 3), dtype=np.uint8)
    labels = np.zeros((600, 1, 1), dtype=np.int8)
    pose = np.zeros((600, 6), dtype=np.float32)
    return G17TargetForwardObservationV1(
        source_pair_ids=pair_ids,
        target_artifact_bytes=b"synthetic-test-target-not-authority",
        target_member_bytes=b"synthetic-test-target-member",
        camera_frames=camera,
        exact_r_numerators=numerators,
        exact_r_denominators=denominators,
        exact_r_projected_rgb=projected,
        seg_labels=labels,
        pose6=pose,
        second_exact_r_numerators=numerators,
        second_exact_r_denominators=denominators,
        second_exact_r_projected_rgb=projected,
        second_seg_labels=labels,
        second_pose6=pose,
        frozen_scorer_bytes=b"synthetic-test-frozen-scorer",
        scorer_runtime_environment_bytes=b"synthetic-test-scorer-runtime",
    )


def _candidate(
    target: G17TargetForwardObservationV1,
    *,
    member: bytes,
    flipped_pairs: int,
    pose_value: float,
) -> G17CandidateForwardObservationV1:
    archive = _zip(member)
    camera = np.zeros((600, 2, 1, 1, 3), dtype=np.uint8)
    numerators = np.zeros((600, 2, 1, 1, 3), dtype=np.float32)
    denominators = np.ones((600, 2, 1, 1, 3), dtype=np.float32)
    projected = camera.copy()
    labels = np.zeros((600, 1, 1), dtype=np.int8)
    labels[:flipped_pairs] = 1
    pose = np.full((600, 6), pose_value, dtype=np.float32)
    return G17CandidateForwardObservationV1(
        target=target,
        archive_bytes=archive,
        member_bytes=member,
        receiver_receipt_bytes=b"synthetic-test-receiver:" + member,
        decoded_output_bytes=b"synthetic-test-decoded:" + member,
        camera_y1=camera,
        exact_r_numerators=numerators,
        exact_r_denominators=denominators,
        exact_r_projected_rgb=projected,
        realized_seg_labels=labels,
        realized_pose6=pose,
        second_exact_r_numerators=numerators,
        second_exact_r_denominators=denominators,
        second_exact_r_projected_rgb=projected,
        second_realized_seg_labels=labels,
        second_realized_pose6=pose,
        frozen_scorer_bytes=b"synthetic-test-frozen-scorer",
        scorer_runtime_environment_bytes=b"synthetic-test-scorer-runtime",
    )


def _complete_campaign(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "vjp_custody_n600_extension.v1",
                "status": "COMPLETE_N600",
                "final_completed_count": 600,
                "final_completed_pair_ids": list(range(600)),
                "refused_pair_ids": [],
                "still_missing_pair_ids": [],
            }
        )
    )
    return path


def test_exact_n600_endpoint_effects_emit_deterministically_but_are_not_actionable(
    tmp_path: Path,
) -> None:
    target = _target()
    baseline = _candidate(target, member=b"baseline", flipped_pairs=0, pose_value=0.0)
    candidate = _candidate(target, member=b"candidate-longer", flipped_pairs=60, pose_value=0.1)
    transition = materialize_g70_finite_effect_transition(
        baseline=baseline,
        candidate=candidate,
    )

    assert transition.pair_ids.tolist() == list(range(600))
    assert transition.scorer_term_effect_vectors.shape == (600, 2)
    assert transition.scorer_term_effect_vectors.dtype == np.float64
    assert np.isclose(
        transition.scorer_term_effect_vectors[:, 0].sum(dtype=np.float64),
        transition.aggregate_effects["seg_score_term_delta"],
        atol=1e-6,
    )
    assert np.isclose(
        transition.scorer_term_effect_vectors[:, 1].sum(dtype=np.float64),
        transition.aggregate_effects["pose_score_term_delta"],
        atol=1e-6,
    )
    assert transition.derivative_audit is None
    assert DERIVATIVE_JOIN_BLOCKER in transition.blockers
    assert transition.actionable_costate_input is False

    first = write_g70_finite_effect_bundle(transition, output_dir=tmp_path / "first")
    second = write_g70_finite_effect_bundle(transition, output_dir=tmp_path / "second")
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.receipt_sha256 == second.receipt_sha256
    receipt = json.loads(first.receipt_path.read_text())
    assert receipt["actionable_costate_input"] is False
    assert receipt["actionable_consumers"] == []
    assert receipt["effect_bundle"]["sha256"] == hashlib.sha256(first.bundle_path.read_bytes()).hexdigest()
    with np.load(first.bundle_path, allow_pickle=False) as bundle:
        assert bundle.files == [
            "pair_ids",
            "scorer_term_effect_vectors",
            "baseline_per_pair_d_seg",
            "candidate_per_pair_d_seg",
            "baseline_per_pair_d_pose",
            "candidate_per_pair_d_pose",
        ]
        assert np.array_equal(bundle["scorer_term_effect_vectors"], transition.scorer_term_effect_vectors)
    with pytest.raises(G70EffectMaterializationError, match=DERIVATIVE_JOIN_BLOCKER):
        require_g70_actionable_costate_input(transition)


def test_arbitrary_complete_campaign_cannot_mint_actual_vjp_authority(tmp_path: Path) -> None:
    with pytest.raises(G70EffectMaterializationError, match="descriptor-stable canonical live path"):
        audit_vjp_campaign_for_transition(_complete_campaign(tmp_path / "campaign.json"))


def test_symlink_campaign_receipt_is_rejected_before_resolution(tmp_path: Path) -> None:
    target = _complete_campaign(tmp_path / "target.json")
    link = tmp_path / "campaign-link.json"
    link.symlink_to(target)
    with pytest.raises(G70EffectMaterializationError, match="non-symlink"):
        audit_vjp_campaign_for_transition(link)


def test_materializer_rejects_different_target_objects() -> None:
    baseline = _candidate(_target(), member=b"baseline", flipped_pairs=0, pose_value=0.0)
    candidate = _candidate(_target(), member=b"candidate", flipped_pairs=1, pose_value=0.0)
    with pytest.raises(G70EffectMaterializationError, match="same exact target"):
        materialize_g70_finite_effect_transition(baseline=baseline, candidate=candidate)


def test_materializer_rejects_non_zip_archive() -> None:
    target = _target()
    baseline = _candidate(target, member=b"baseline", flipped_pairs=0, pose_value=0.0)
    candidate = _candidate(target, member=b"candidate", flipped_pairs=1, pose_value=0.0)
    object.__setattr__(candidate, "archive_bytes", b"not-a-zip")
    with pytest.raises(Exception, match="archive"):
        materialize_g70_finite_effect_transition(baseline=baseline, candidate=candidate)


def test_materializer_rejects_ambiguous_duplicate_member_content() -> None:
    target = _target()
    baseline = _candidate(target, member=b"baseline", flipped_pairs=0, pose_value=0.0)
    candidate = _candidate(target, member=b"candidate", flipped_pairs=1, pose_value=0.0)
    object.__setattr__(candidate, "archive_bytes", _zip_with_duplicate_content(candidate.member_bytes))
    object.__setattr__(
        candidate.receipt,
        "archive_sha256",
        hashlib.sha256(candidate.archive_bytes).hexdigest(),
    )
    with pytest.raises(G70EffectMaterializationError, match="exactly one member"):
        materialize_g70_finite_effect_transition(baseline=baseline, candidate=candidate)


def test_designated_campaign_must_be_terminal_complete_n600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _complete_campaign(tmp_path / "campaign.json")
    value = json.loads(path.read_text())
    value["still_missing_pair_ids"] = [599]
    value["final_completed_pair_ids"] = list(range(599))
    value["final_completed_count"] = 599
    path.write_text(json.dumps(value))
    payload = path.read_bytes()
    monkeypatch.setattr(g70, "CANONICAL_VJP_CAMPAIGN_PATH", path.resolve())
    monkeypatch.setattr(g70, "CANONICAL_VJP_CAMPAIGN_BYTES", len(payload))
    monkeypatch.setattr(g70, "CANONICAL_VJP_CAMPAIGN_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(G70EffectMaterializationError, match="terminal complete n600"):
        audit_vjp_campaign_for_transition(path)
