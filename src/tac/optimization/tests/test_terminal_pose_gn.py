from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pytest

from tac.optimization.terminal_pose_gn import (
    FULL_N600_POSE_AUTHORITY_MARKER,
    STALE_POSE_REHEARSAL_AUTHORITY_MARKER,
    CandidateArtifactScope,
    ContestAxis,
    PoseAuthorityMode,
    PoseJointEvaluation,
    ProductionPoseCustodyV1,
    TerminalPoseCandidateArtifact,
    TerminalPoseError,
    TerminalPoseGNConfig,
    TerminalPosePacketV1,
    normalize_terminal_pose_basis,
    parse_terminal_pose_packet,
    realize_terminal_pose_pair,
    serialize_terminal_pose_packet,
    solve_terminal_pose_gn,
)


def _parent() -> np.ndarray:
    pair = np.full((2, 16, 20, 3), 128, dtype=np.uint8)
    pair[1] = np.arange(16 * 20 * 3, dtype=np.uint16).reshape(16, 20, 3) % 256
    return pair


def _basis(_seed: int, _selector: str, shape: tuple[int, int, int]) -> np.ndarray:
    height, width, channels = shape
    assert channels == 3
    x = np.cos(2.0 * np.pi * (np.arange(width) + 0.5) / width)
    y = np.cos(2.0 * np.pi * (np.arange(height) + 0.5) / height)
    fields = np.zeros((6, height, width, channels), dtype=np.float64)
    for channel in range(3):
        fields[channel, :, :, channel] = x[None, :]
        fields[channel + 3, :, :, channel] = y[:, None]
    return fields


def _packet(codes: np.ndarray) -> bytes:
    return serialize_terminal_pose_packet(
        TerminalPosePacketV1(
            seed=19,
            basis_selector="test_low_frequency_v1",
            amplitude_q8=512,
            coefficients=np.asarray(codes, dtype=np.int16)[None, :],
        )
    )


def _artifact(codes: np.ndarray, *, production: bool = False) -> TerminalPoseCandidateArtifact:
    packet = _packet(codes)
    return TerminalPoseCandidateArtifact(
        outer_archive=(b"full-outer-archive:" + packet if production else packet),
        terminal_packet=packet,
        scope=(
            CandidateArtifactScope.FULL_OUTER_ARCHIVE if production else CandidateArtifactScope.TERMINAL_SECTION_ONLY
        ),
    )


def _custody() -> ProductionPoseCustodyV1:
    def digest(label: str) -> str:
        return sha256(label.encode("ascii")).hexdigest()

    return ProductionPoseCustodyV1(
        parent_archive_sha256=digest("parent"),
        compiler_sha256=digest("compiler"),
        receiver_sha256=digest("receiver"),
        evaluator_sha256=digest("evaluator"),
        upstream_sha256=digest("upstream"),
        contest_axis=ContestAxis.CONTEST_CPU,
        command=("uv", "run", "upstream/evaluate.py"),
        hardware="synthetic-test-cpu",
    )


def _score_callback(
    parent: np.ndarray,
    rendered_basis: np.ndarray,
    *,
    production: bool,
    custody: ProductionPoseCustodyV1 | None = None,
    frozen_observations: list[bool] | None = None,
    calls: list[str] | None = None,
):
    normalized = normalize_terminal_pose_basis(rendered_basis)
    denominators = np.sum(normalized * normalized, axis=(1, 2, 3))
    offset = np.full(6, 6.0, dtype=np.float64)

    def score(pair: np.ndarray, artifact: TerminalPoseCandidateArtifact) -> PoseJointEvaluation:
        frame1_frozen = np.array_equal(pair[1], parent[1])
        if frozen_observations is not None:
            frozen_observations.append(frame1_frozen)
        if calls is not None:
            calls.append(artifact.binding_sha256)
        assert frame1_frozen
        delta = pair[0].astype(np.float64) - parent[0].astype(np.float64)
        response = np.array([float(np.sum(delta * normalized[index]) / denominators[index]) for index in range(6)])
        pose = offset + response
        pose_mse = float(np.mean(pose * pose))
        return PoseJointEvaluation(
            pose6=pose,
            d_seg=0.0,
            d_pose=pose_mse,
            archive_bytes=artifact.archive_bytes,
            archive_sha256=artifact.archive_sha256,
            sample_count=600 if production else 1,
            authority_marker=(FULL_N600_POSE_AUTHORITY_MARKER if production else STALE_POSE_REHEARSAL_AUTHORITY_MARKER),
            custody_digest=None if custody is None else custody.digest,
            realized=True,
        )

    return score


def test_packet_roundtrip_is_canonical_and_corruption_fails() -> None:
    packet = TerminalPosePacketV1(
        seed=123,
        basis_selector="generic_low_frequency_12",
        amplitude_q8=384,
        coefficients=np.arange(24, dtype=np.int16).reshape(2, 12),
    )
    payload = serialize_terminal_pose_packet(packet)
    parsed = parse_terminal_pose_packet(payload)
    assert parsed.seed == packet.seed
    assert parsed.basis_selector == packet.basis_selector
    assert parsed.amplitude_q8 == packet.amplitude_q8
    assert np.array_equal(parsed.coefficients, packet.coefficients)
    assert serialize_terminal_pose_packet(parsed) == payload

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(TerminalPoseError, match="checksum"):
        parse_terminal_pose_packet(bytes(corrupted))
    corrupted_header = bytearray(payload)
    corrupted_header[20] ^= 1
    with pytest.raises(TerminalPoseError, match="checksum"):
        parse_terminal_pose_packet(bytes(corrupted_header))
    with pytest.raises(TerminalPoseError, match="length/trailing"):
        parse_terminal_pose_packet(payload + b"x")


def test_receiver_normalizes_quantizes_and_freezes_frame1() -> None:
    parent = _parent()
    rendered = _basis(19, "test_low_frequency_v1", tuple(parent.shape[1:]))
    result = realize_terminal_pose_pair(
        parent,
        rendered,
        np.array([2, -2, 1, -1, 3, -3], dtype=np.int16),
        amplitude_q8=512,
    )
    assert result.dtype == np.uint8
    assert np.array_equal(result[1], parent[1])
    assert not np.array_equal(result[0], parent[0])
    normalized = normalize_terminal_pose_basis(rendered)
    assert np.allclose(np.mean(normalized, axis=(1, 2, 3)), 0.0, atol=1e-12)
    assert np.allclose(
        np.sqrt(np.mean(normalized * normalized, axis=(1, 2, 3))),
        1.0,
        atol=1e-12,
    )


def test_rehearsal_gn_uses_realized_verdicts_but_never_promotes() -> None:
    parent = _parent()
    rendered = _basis(19, "test_low_frequency_v1", tuple(parent.shape[1:]))
    frozen: list[bool] = []
    result = solve_terminal_pose_gn(
        parent,
        np.zeros(6, dtype=np.float64),
        _basis,
        _artifact,
        _score_callback(parent, rendered, production=False, frozen_observations=frozen),
        seed=19,
        basis_selector="test_low_frequency_v1",
        config=TerminalPoseGNConfig(
            relinearizations=2,
            damping=1.0e-3,
            amplitude_q8=512,
            authority_mode=PoseAuthorityMode.STALE_REHEARSAL,
        ),
    )
    assert result.strict_realized_improvement
    assert result.pose_mse_final < result.pose_mse_initial
    assert result.final_evaluation.joint_action < result.initial_evaluation.joint_action
    assert any(step.admitted for step in result.steps)
    assert not result.governed_handoff_eligible
    assert result.to_payload()["promotion_allowed"] is False
    assert frozen and all(frozen)


def test_production_yields_external_review_never_handoff_or_promotion(
    tmp_path,
) -> None:
    parent = _parent()
    rendered = _basis(19, "test_low_frequency_v1", tuple(parent.shape[1:]))
    custody = _custody()
    calls: list[str] = []
    resume_root = tmp_path / "resume"
    config = TerminalPoseGNConfig(
        relinearizations=2,
        damping=1.0e-3,
        amplitude_q8=512,
        authority_mode=PoseAuthorityMode.PRODUCTION_FULL_N600,
        production_custody=custody,
        resume_path=str(resume_root),
    )
    result = solve_terminal_pose_gn(
        parent,
        np.zeros(6, dtype=np.float64),
        _basis,
        lambda codes: _artifact(codes, production=True),
        _score_callback(parent, rendered, production=True, custody=custody, calls=calls),
        seed=19,
        basis_selector="test_low_frequency_v1",
        config=config,
    )
    assert result.strict_realized_improvement
    assert not result.governed_handoff_eligible
    assert result.external_governor_review_required
    assert result.final_evaluation.full_n600
    payload = result.to_payload()
    assert payload["governed_handoff_eligible"] is False
    assert payload["external_governor_review_required"] is True
    assert payload["production_accepted"] is False
    assert payload["promotion_allowed"] is False
    assert payload["score_claim"] is False
    assert payload["pointer_moved"] is False
    assert calls
    first_call_count = len(calls)
    assert (resume_root / "manifest.json").is_file()
    assert (resume_root / "completed.json").is_file()
    assert len(list((resume_root / "iterations").glob("*.json"))) == 2
    assert len(list((resume_root / "verdicts").glob("*.json"))) == first_call_count

    def must_not_replay(_pair: np.ndarray, _artifact_value: TerminalPoseCandidateArtifact) -> PoseJointEvaluation:
        raise AssertionError("completed production resume replayed a scorer verdict")

    # Simulate a crash after the last durable iteration but before the distinct
    # completed ledger. Restart must use verdict and iteration caches.
    (resume_root / "completed.json").unlink()
    resumed = solve_terminal_pose_gn(
        parent,
        np.zeros(6, dtype=np.float64),
        _basis,
        lambda codes: _artifact(codes, production=True),
        must_not_replay,
        seed=19,
        basis_selector="test_low_frequency_v1",
        config=config,
    )
    assert resumed.to_payload() == result.to_payload()
    assert (resume_root / "completed.json").is_file()
    assert len(calls) == first_call_count

    completed_resume = solve_terminal_pose_gn(
        parent,
        np.zeros(6, dtype=np.float64),
        _basis,
        lambda codes: _artifact(codes, production=True),
        must_not_replay,
        seed=19,
        basis_selector="test_low_frequency_v1",
        config=config,
    )
    assert completed_resume.to_payload() == result.to_payload()

    completed_path = resume_root / "completed.json"
    corrupted = json.loads(completed_path.read_text())
    corrupted["payload"]["completed"] = False
    completed_path.write_text(json.dumps(corrupted))
    with pytest.raises(TerminalPoseError, match="checksum"):
        solve_terminal_pose_gn(
            parent,
            np.zeros(6, dtype=np.float64),
            _basis,
            lambda codes: _artifact(codes, production=True),
            must_not_replay,
            seed=19,
            basis_selector="test_low_frequency_v1",
            config=config,
        )


def test_constant_realized_oracle_cannot_admit_soft_gn_move() -> None:
    parent = _parent()

    def constant_score(pair: np.ndarray, artifact: TerminalPoseCandidateArtifact) -> PoseJointEvaluation:
        assert np.array_equal(pair[1], parent[1])
        return PoseJointEvaluation(
            pose6=np.ones(6),
            d_seg=0.0,
            d_pose=0.1,
            archive_bytes=artifact.archive_bytes,
            archive_sha256=artifact.archive_sha256,
            sample_count=1,
            authority_marker=STALE_POSE_REHEARSAL_AUTHORITY_MARKER,
            custody_digest=None,
            realized=True,
        )

    result = solve_terminal_pose_gn(
        parent,
        np.zeros(6),
        _basis,
        _artifact,
        constant_score,
        seed=19,
        basis_selector="test_low_frequency_v1",
        config=TerminalPoseGNConfig(
            relinearizations=2,
            amplitude_q8=512,
            authority_mode=PoseAuthorityMode.STALE_REHEARSAL,
        ),
    )
    assert not result.strict_realized_improvement
    assert not result.governed_handoff_eligible
    assert not any(step.admitted for step in result.steps)
    assert np.array_equal(result.final_coefficients, np.zeros(6, dtype=np.int16))


def test_production_refuses_subset_or_unmarked_verdict(tmp_path) -> None:
    parent = _parent()
    rendered = _basis(19, "test_low_frequency_v1", tuple(parent.shape[1:]))
    custody = _custody()
    with pytest.raises(TerminalPoseError, match="matching full-n600 custody"):
        solve_terminal_pose_gn(
            parent,
            np.zeros(6),
            _basis,
            lambda codes: _artifact(codes, production=True),
            _score_callback(parent, rendered, production=False),
            seed=19,
            basis_selector="test_low_frequency_v1",
            config=TerminalPoseGNConfig(
                relinearizations=2,
                amplitude_q8=512,
                authority_mode=PoseAuthorityMode.PRODUCTION_FULL_N600,
                production_custody=custody,
                resume_path=str(tmp_path / "resume"),
            ),
        )


def test_scorer_must_repeat_exact_candidate_archive_binding() -> None:
    parent = _parent()

    def mismatched_archive(pair: np.ndarray, artifact: TerminalPoseCandidateArtifact) -> PoseJointEvaluation:
        assert np.array_equal(pair[1], parent[1])
        return PoseJointEvaluation(
            pose6=np.ones(6),
            d_seg=0.0,
            d_pose=0.1,
            archive_bytes=artifact.archive_bytes + 1,
            archive_sha256=artifact.archive_sha256,
            sample_count=1,
            authority_marker=STALE_POSE_REHEARSAL_AUTHORITY_MARKER,
            custody_digest=None,
            realized=True,
        )

    with pytest.raises(TerminalPoseError, match="bytes/SHA differ"):
        solve_terminal_pose_gn(
            parent,
            np.zeros(6),
            _basis,
            _artifact,
            mismatched_archive,
            seed=19,
            basis_selector="test_low_frequency_v1",
            config=TerminalPoseGNConfig(
                relinearizations=2,
                amplitude_q8=512,
                authority_mode=PoseAuthorityMode.STALE_REHEARSAL,
            ),
        )


def test_production_requires_custody_and_absolute_resume_path(tmp_path) -> None:
    with pytest.raises(TerminalPoseError, match="typed custody"):
        TerminalPoseGNConfig(
            authority_mode=PoseAuthorityMode.PRODUCTION_FULL_N600,
            resume_path=str(tmp_path / "resume"),
        )
    with pytest.raises(TerminalPoseError, match="atomic resume path"):
        TerminalPoseGNConfig(
            authority_mode=PoseAuthorityMode.PRODUCTION_FULL_N600,
            production_custody=_custody(),
        )
    with pytest.raises(TerminalPoseError, match="must be absolute"):
        TerminalPoseGNConfig(
            authority_mode=PoseAuthorityMode.PRODUCTION_FULL_N600,
            production_custody=_custody(),
            resume_path="relative/resume",
        )


def test_solver_refuses_out_of_band_packet_amplitude() -> None:
    parent = _parent()
    rendered = _basis(19, "test_low_frequency_v1", tuple(parent.shape[1:]))

    def wrong_amplitude_artifact(
        codes: np.ndarray,
    ) -> TerminalPoseCandidateArtifact:
        packet = serialize_terminal_pose_packet(
            TerminalPosePacketV1(
                seed=19,
                basis_selector="test_low_frequency_v1",
                amplitude_q8=511,
                coefficients=np.asarray(codes, dtype=np.int16)[None, :],
            )
        )
        return TerminalPoseCandidateArtifact(
            outer_archive=packet,
            terminal_packet=packet,
            scope=CandidateArtifactScope.TERMINAL_SECTION_ONLY,
        )

    with pytest.raises(TerminalPoseError, match="amplitude differs"):
        solve_terminal_pose_gn(
            parent,
            np.zeros(6),
            _basis,
            wrong_amplitude_artifact,
            _score_callback(parent, rendered, production=False),
            seed=19,
            basis_selector="test_low_frequency_v1",
            config=TerminalPoseGNConfig(
                relinearizations=2,
                amplitude_q8=512,
                authority_mode=PoseAuthorityMode.STALE_REHEARSAL,
            ),
        )
