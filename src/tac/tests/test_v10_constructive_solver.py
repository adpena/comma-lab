from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import tac.optimization.v10_constructive_solver as solver_module
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator, Factor2ExactVerification
from tac.optimization.v10_constructive_solver import (
    ACTIVE_ARRANGEMENT,
    EXPECTED_SOURCE_HASHES,
    HARD_ORACLE_SCHEMA,
    PAIR_SCHEMA,
    RECEIVER_ARITHMETIC,
    REPRESENTATION,
    ConstructiveSolveError,
    HardOracleDecision,
    load_vjp_custody_pair,
    project_pixelwise_seg_relaxation,
    project_rank6_pose_ellipsoid,
    project_weighted_box_halfspace,
    realize_factor2_and_require_hard_oracle,
    run_resumable_pair_chunk,
    solve_constructive_projection,
)


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _custody_npz(path: Path, *, pair_id: int = 7, extra: bool = False) -> Path:
    scorer_hw, camera_hw = (2, 3), (4, 5)
    seg_q = np.zeros((*scorer_hw, 3), dtype=np.float32)
    seg_q[..., 0] = 1.0
    lipschitz = np.arange(1, 7, dtype=np.float32).reshape(scorer_hw)
    arrays = {
        "winner": np.zeros(scorer_hw, dtype=np.int8),
        "rival": np.ones(scorer_hw, dtype=np.int8),
        "cached_margin": np.ones(scorer_hw, dtype=np.float32),
        "native_margin": np.ones(scorer_hw, dtype=np.float32),
        "head_pair_norms": np.ones(scorer_hw, dtype=np.float32),
        "seg_g_y": lipschitz[..., None] * seg_q,
        "seg_g_x": np.ones((*camera_hw, 3), dtype=np.float32),
        "seg_q": seg_q,
        "seg_local_lipschitz": lipschitz,
        "pose_j_y": np.ones((6, 2, *scorer_hw, 3), dtype=np.float32),
        "pose_j_x": np.ones((6, 2, *camera_hw, 3), dtype=np.float32),
    }
    metadata = {
        "schema": PAIR_SCHEMA,
        "pair_id": pair_id,
        "receiver_arithmetic": RECEIVER_ARITHMETIC,
        "active_arrangement": ACTIVE_ARRANGEMENT,
        "winner_source": "cached_lstars_verified_against_fresh_native_fp32_logits",
        "rival_source": "fresh_native_fp32_logits_highest_nonwinner_not_cached",
        "representation": REPRESENTATION,
        "source_hashes": EXPECTED_SOURCE_HASHES,
        "checks": {"fixture": True},
        "tensors": {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _array_sha(value),
            }
            for key, value in arrays.items()
        },
        "reconstruction": {"rebuildable": True},
    }
    payload: dict[str, np.ndarray] = {
        **arrays,
        "pair_id": np.asarray(pair_id, dtype=np.int64),
        "custody_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    }
    if extra:
        payload["unreviewed_extra"] = np.asarray(1, dtype=np.int8)
    np.savez_compressed(path, **payload)
    return path


def _rewrite_custody(path: Path, mutate: Callable[[dict[str, np.ndarray]], None]) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    metadata = json.loads(str(payload["custody_json"].reshape(())))
    mutate(payload)
    for key in metadata["tensors"]:
        value = payload[key]
        metadata["tensors"][key] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_sha(value),
        }
    payload["custody_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    np.savez_compressed(path, **payload)


def _hard_decision(
    frames: tuple[np.ndarray, np.ndarray],
    *,
    admitted: bool = True,
    frame_hashes: tuple[str, str] | None = None,
) -> HardOracleDecision:
    return HardOracleDecision(
        admitted=admitted,
        schema=HARD_ORACLE_SCHEMA,
        receiver_arithmetic=RECEIVER_ARITHMETIC,
        realized_frame_sha256s=frame_hashes or (_array_sha(frames[0]), _array_sha(frames[1])),
        d_seg=0.0,
        d_pose=0.0,
        source_hashes=EXPECTED_SOURCE_HASHES,
    )


def test_exact_vjp_pair_npz_validation_and_pair_id(tmp_path: Path) -> None:
    path = _custody_npz(tmp_path / "pair_0007.vjp.npz")
    pair = load_vjp_custody_pair(path, expected_pair_id=7, scorer_hw=(2, 3), camera_hw=(4, 5))
    assert pair.pair_id == 7
    assert pair.seg_q.dtype == np.float32
    assert pair.pose_j_y.shape == (6, 2, 2, 3, 3)
    assert len(pair.npz_sha256) == 64

    with pytest.raises(ConstructiveSolveError, match="pair id differs"):
        load_vjp_custody_pair(path, expected_pair_id=8, scorer_hw=(2, 3), camera_hw=(4, 5))


def test_vjp_pair_refuses_extra_field_and_tensor_hash_drift(tmp_path: Path) -> None:
    extra = _custody_npz(tmp_path / "extra.npz", extra=True)
    with pytest.raises(ConstructiveSolveError, match="field mismatch"):
        load_vjp_custody_pair(extra, expected_pair_id=7, scorer_hw=(2, 3), camera_hw=(4, 5))

    valid = _custody_npz(tmp_path / "valid.npz")
    with np.load(valid, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    payload["seg_q"][0, 0, 0] = 0.5
    np.savez_compressed(valid, **payload)
    with pytest.raises(ConstructiveSolveError, match="embedded hash"):
        load_vjp_custody_pair(valid, expected_pair_id=7, scorer_hw=(2, 3), camera_hw=(4, 5))


def test_vjp_pair_refuses_negative_lipschitz_and_nonzero_q_at_zero_scale(tmp_path: Path) -> None:
    negative = _custody_npz(tmp_path / "negative.npz")

    def make_negative(payload: dict[str, np.ndarray]) -> None:
        payload["seg_local_lipschitz"][0, 0] = -1.0
        payload["seg_q"][0, 0] = 0.0
        payload["seg_g_y"][0, 0] = 0.0

    _rewrite_custody(negative, make_negative)
    with pytest.raises(ConstructiveSolveError, match="valid domains"):
        load_vjp_custody_pair(negative, expected_pair_id=7, scorer_hw=(2, 3), camera_hw=(4, 5))

    zero_scale = _custody_npz(tmp_path / "zero-scale.npz")

    def leave_nonzero_q(payload: dict[str, np.ndarray]) -> None:
        payload["seg_local_lipschitz"][0, 0] = 0.0
        payload["seg_g_y"][0, 0] = 0.0

    _rewrite_custody(zero_scale, leave_nonzero_q)
    with pytest.raises(ConstructiveSolveError, match="unit/zero convention"):
        load_vjp_custody_pair(zero_scale, expected_pair_id=7, scorer_hw=(2, 3), camera_hw=(4, 5))


def test_vjp_pair_expected_id_requires_exact_non_bool_int(tmp_path: Path) -> None:
    path = _custody_npz(tmp_path / "pair.npz")
    for malformed in (True, 7.0, np.int64(7)):
        with pytest.raises(ConstructiveSolveError, match="exact non-bool"):
            load_vjp_custody_pair(
                path,
                expected_pair_id=cast("int", malformed),
                scorer_hw=(2, 3),
                camera_hw=(4, 5),
            )

    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    metadata = json.loads(str(payload["custody_json"].reshape(())))
    metadata["pair_id"] = True
    payload["custody_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    np.savez_compressed(path, **payload)
    with pytest.raises(ConstructiveSolveError, match="pair id differs"):
        load_vjp_custody_pair(path, expected_pair_id=7, scorer_hw=(2, 3), camera_hw=(4, 5))


def test_weighted_box_halfspace_inactive_returns_clipped_anchor() -> None:
    result = project_weighted_box_halfspace(
        np.array([-2.0, 0.25, 3.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 2.0, 4.0]),
        margin=3.0,
    )
    np.testing.assert_array_equal(result, [-1.0, 0.25, 1.0])
    assert result.dtype == np.float32


def test_weighted_box_halfspace_binding_matches_kkt_and_is_deterministic() -> None:
    args = (
        np.array([-2.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([-4.0, -4.0]),
        np.array([4.0, 4.0]),
        np.array([1.0, 3.0]),
    )
    first = project_weighted_box_halfspace(*args, margin=0.0)
    second = project_weighted_box_halfspace(*args, margin=0.0)
    # delta = anchor + lambda*q/w, q.delta=0 gives lambda=1.5.
    np.testing.assert_allclose(first, [-0.5, 0.5], atol=2e-6)
    assert first.dtype == np.float32
    assert first.tobytes() == second.tobytes()


def test_weighted_box_halfspace_clips_and_refuses_infeasible_or_malformed_q() -> None:
    result = project_weighted_box_halfspace(
        np.array([-10.0, -10.0]),
        np.array([1.0, 1.0]),
        np.array([-1.0, -1.0]),
        np.array([0.25, 2.0]),
        np.array([0.1, 10.0]),
        margin=0.0,
    )
    assert result[0] == pytest.approx(0.25)
    assert result.sum() == pytest.approx(0.0, abs=5e-11)

    with pytest.raises(ConstructiveSolveError, match="infeasible"):
        project_weighted_box_halfspace(np.zeros(2), np.ones(2), -np.ones(2), -np.full(2, 0.5), np.ones(2), margin=0)
    with pytest.raises(ConstructiveSolveError, match="finite"):
        project_weighted_box_halfspace(
            np.zeros(2), np.array([np.nan, 0.0]), -np.ones(2), np.ones(2), np.ones(2), margin=0.0
        )


def test_vector_pixelwise_breakpoint_solve_matches_scalar_kkt() -> None:
    rng = np.random.default_rng(541)
    anchor = rng.normal(size=(5, 7, 3)) * 2
    q = rng.normal(size=anchor.shape)
    lower = rng.uniform(-2.0, -0.25, size=anchor.shape)
    upper = rng.uniform(0.25, 2.0, size=anchor.shape)
    weights = rng.uniform(0.2, 3.0, size=anchor.shape)
    # margin=0 is feasible because every box spans zero.
    margins = np.zeros(anchor.shape[:-1])
    vector = project_pixelwise_seg_relaxation(anchor, q, lower, upper, weights, margins)
    scalar = np.empty_like(vector)
    for index in np.ndindex(margins.shape):
        scalar[index] = project_weighted_box_halfspace(
            anchor[index], q[index], lower[index], upper[index], weights[index], margin=0.0
        )
    np.testing.assert_allclose(vector, scalar, atol=3e-5, rtol=0)
    assert vector.dtype == np.float32


def test_rank6_pose_projection_uses_weighted_gram_root() -> None:
    value = np.array([1.0, 2.0, 3.0])
    jac = np.zeros((6, 3))
    jac[0, 0] = 1.0
    jac[1, 1] = 1.0
    jac[2, 2] = 1.0
    # Rows 3..5 may be rank-deficient here; the surface is rank at most six.
    projected = project_rank6_pose_ellipsoid(value, jac, np.array([1.0, 2.0, 4.0]), tau_pose=1.0 / 6.0)
    assert np.sum((jac @ projected) ** 2) == pytest.approx(1.0, abs=5e-5)
    assert np.linalg.norm(projected) < np.linalg.norm(value)
    assert projected.dtype == np.float32


def test_zero_band_returns_exact_target_displacement() -> None:
    target = np.arange(12, dtype=np.uint8).reshape(2, 1, 2, 3)
    predictor = np.flip(target, axis=0).copy()
    result = solve_constructive_projection(
        target_planes=target,
        predictor_planes=predictor,
        lower=np.zeros_like(target, dtype=np.float64),
        upper=np.zeros_like(target, dtype=np.float64),
        weights=np.ones_like(target, dtype=np.float64),
        seg_q=np.ones((1, 2, 3), dtype=np.float32),
        seg_margins=np.ones((1, 2), dtype=np.float32),
        pose_j_y=np.ones((6, 2, 1, 2, 3), dtype=np.float32),
        tau_pose=0.0,
    )
    np.testing.assert_array_equal(result.delta, 0.0)
    assert result.delta.dtype == np.float32
    assert result.zero_band_exact is True
    assert result.pose_mse == 0.0
    assert "proposal_only" in result.certificate_scope


@pytest.mark.parametrize("shape", [(2, 3, 3), (2, 1, 2, 3, 3)])
def test_projection_refuses_non_4d_two_plane_shapes(shape: tuple[int, ...]) -> None:
    target = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ConstructiveSolveError, match="same-shape two-plane RGB"):
        solve_constructive_projection(
            target_planes=target,
            predictor_planes=target,
            lower=np.zeros_like(target, dtype=np.float32),
            upper=np.zeros_like(target, dtype=np.float32),
            weights=np.ones_like(target, dtype=np.float32),
            seg_q=np.ones(target.shape[1:], dtype=np.float32),
            seg_margins=np.ones(target.shape[1:-1], dtype=np.float32),
        )


def test_zero_band_validates_pose_geometry_finiteness_and_tau() -> None:
    target = np.zeros((2, 1, 1, 3), dtype=np.uint8)
    kwargs: dict[str, Any] = {
        "target_planes": target,
        "predictor_planes": target,
        "lower": np.zeros_like(target, dtype=np.float32),
        "upper": np.zeros_like(target, dtype=np.float32),
        "weights": np.ones_like(target, dtype=np.float32),
        "seg_q": np.ones((1, 1, 3), dtype=np.float32),
        "seg_margins": np.ones((1, 1), dtype=np.float32),
    }
    with pytest.raises(ConstructiveSolveError, match="tau_pose is required"):
        solve_constructive_projection(**kwargs, pose_j_y=np.ones((6, 2, 1, 1, 3), dtype=np.float32))
    malformed_j = np.ones((6, 2, 1, 1, 3), dtype=np.float32)
    malformed_j[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ConstructiveSolveError, match="finite with shape"):
        solve_constructive_projection(**kwargs, pose_j_y=malformed_j, tau_pose=0.0)
    with pytest.raises(ConstructiveSolveError, match="finite and nonnegative"):
        solve_constructive_projection(**kwargs, pose_j_y=np.ones((6, 2, 1, 1, 3), dtype=np.float32), tau_pose=-1.0)


def test_dykstra_intersects_pixel_seg_box_and_rank6_pose() -> None:
    target = np.zeros((2, 1, 1, 3), dtype=np.float64)
    predictor = np.zeros_like(target)
    predictor[1, 0, 0, 0] = -2.0
    jac = np.zeros((6, *target.shape))
    jac[0, 1, 0, 0, 0] = 1.0
    result = solve_constructive_projection(
        target_planes=target,
        predictor_planes=predictor,
        lower=np.full_like(target, -4.0),
        upper=np.full_like(target, 4.0),
        weights=np.ones_like(target),
        seg_q=np.array([[[1.0, 0.0, 0.0]]]),
        seg_margins=np.array([[0.5]]),
        pose_j_y=jac,
        tau_pose=0.25 / 6.0,
        tolerance=2e-5,
    )
    # Seg requires x>=-0.5; pose requires |x|<=0.5. Their intersection meets at -0.5.
    assert result.delta[1, 0, 0, 0] == pytest.approx(-0.5, abs=5e-5)
    assert result.pose_mse == pytest.approx(0.25 / 6.0, abs=5e-5)
    assert result.converged is True
    assert result.delta.dtype == np.float32


def test_factor2_lattice_requires_and_obeys_hard_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    operator = DisjointResizeOperator.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)
    planes = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    verifier_calls = 0
    original_verifier = solver_module.verify_factor2_uint8_scorer_plane

    def counted_verifier(
        counted_operator: DisjointResizeOperator,
        frame: np.ndarray,
        target: np.ndarray,
    ) -> Factor2ExactVerification:
        nonlocal verifier_calls
        verifier_calls += 1
        return original_verifier(counted_operator, frame, target)

    monkeypatch.setattr(solver_module, "verify_factor2_uint8_scorer_plane", counted_verifier)

    def admit(frames: tuple[np.ndarray, np.ndarray]) -> HardOracleDecision:
        calls.append(frames)
        assert all(not frame.flags.writeable for frame in frames)
        return _hard_decision(frames)

    admitted = realize_factor2_and_require_hard_oracle(operator, planes, admit)
    assert len(calls) == 1
    assert verifier_calls == 4  # Both planes are independently checked before and after the callback.
    assert all(proof.certified_exact for proof in admitted.proofs)
    for frame, plane in zip(admitted.camera_frames, planes, strict=True):
        numerators, denominator = operator.apply_numerators(frame)
        np.testing.assert_array_equal(numerators, plane.astype(np.int64) * denominator)
    returned_before_hostile_mutation = tuple(frame.copy() for frame in admitted.camera_frames)
    for callback_frame in calls[0]:
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            callback_frame.flags.writeable = True
    for returned, expected in zip(admitted.camera_frames, returned_before_hostile_mutation, strict=True):
        assert not returned.flags.writeable
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            returned.setflags(write=True)
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            returned.flags.writeable = True
        np.testing.assert_array_equal(returned, expected)
    assert tuple(_array_sha(frame) for frame in admitted.camera_frames) == admitted.hard_oracle.realized_frame_sha256s
    assert all(
        not np.shares_memory(returned, callback_frame)
        for returned, callback_frame in zip(admitted.camera_frames, calls[0], strict=True)
    )

    with pytest.raises(ConstructiveSolveError, match="hard oracle refused"):
        realize_factor2_and_require_hard_oracle(operator, planes, lambda frames: _hard_decision(frames, admitted=False))
    with pytest.raises(ConstructiveSolveError, match="hard oracle is mandatory"):
        realize_factor2_and_require_hard_oracle(operator, planes, None)


def test_hard_oracle_cannot_reopen_or_mutate_callback_frames() -> None:
    operator = DisjointResizeOperator.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)
    planes = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)

    def hostile(frames: tuple[np.ndarray, np.ndarray]) -> HardOracleDecision:
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            frames[0].setflags(write=True)
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            frames[1].flags.writeable = True
        return _hard_decision(frames)

    admitted = realize_factor2_and_require_hard_oracle(operator, planes, hostile)
    assert tuple(_array_sha(frame) for frame in admitted.camera_frames) == admitted.hard_oracle.realized_frame_sha256s
    for frame, plane, proof in zip(admitted.camera_frames, planes, admitted.proofs, strict=True):
        assert proof.certified_exact and proof.numerator_exact
        numerators, denominator = operator.apply_numerators(frame)
        np.testing.assert_array_equal(numerators, plane.astype(np.int64) * denominator)


def test_hard_oracle_decision_is_closed_and_binds_realized_frames() -> None:
    operator = DisjointResizeOperator.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)
    planes = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    with pytest.raises(ConstructiveSolveError, match="exact bool"):
        HardOracleDecision(
            admitted=cast("bool", 1),
            schema=HARD_ORACLE_SCHEMA,
            receiver_arithmetic=RECEIVER_ARITHMETIC,
            realized_frame_sha256s=("0" * 64, "1" * 64),
            d_seg=0.0,
            d_pose=0.0,
            source_hashes=EXPECTED_SOURCE_HASHES,
        )
    for schema, receiver_arithmetic in (
        ("wrong-schema", RECEIVER_ARITHMETIC),
        (HARD_ORACLE_SCHEMA, "not-native-f32"),
    ):
        with pytest.raises(ConstructiveSolveError, match="schema/native-fp32 marker"):
            HardOracleDecision(
                admitted=True,
                schema=schema,
                receiver_arithmetic=receiver_arithmetic,
                realized_frame_sha256s=("0" * 64, "1" * 64),
                d_seg=0.0,
                d_pose=0.0,
                source_hashes=EXPECTED_SOURCE_HASHES,
            )
    with pytest.raises(ConstructiveSolveError, match="finite and nonnegative"):
        HardOracleDecision(
            admitted=True,
            schema=HARD_ORACLE_SCHEMA,
            receiver_arithmetic=RECEIVER_ARITHMETIC,
            realized_frame_sha256s=("0" * 64, "1" * 64),
            d_seg=float("nan"),
            d_pose=0.0,
            source_hashes=EXPECTED_SOURCE_HASHES,
        )
    with pytest.raises(ConstructiveSolveError, match="source-hash custody"):
        HardOracleDecision(
            admitted=True,
            schema=HARD_ORACLE_SCHEMA,
            receiver_arithmetic=RECEIVER_ARITHMETIC,
            realized_frame_sha256s=("0" * 64, "1" * 64),
            d_seg=0.0,
            d_pose=0.0,
            source_hashes={**EXPECTED_SOURCE_HASHES, "cache_sha256": "0" * 64},
        )
    with pytest.raises(ConstructiveSolveError, match="realized-frame SHA-256 custody"):
        realize_factor2_and_require_hard_oracle(
            operator,
            planes,
            lambda frames: _hard_decision(frames, frame_hashes=("0" * 64, "1" * 64)),
        )


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 4, 4),
        (2, 3, 3, 3),
        (1, 3, 4, 3),
    ],
)
def test_lattice_admission_requires_exact_operator_rgb_geometry(shape: tuple[int, ...]) -> None:
    operator = DisjointResizeOperator.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)
    with pytest.raises(ConstructiveSolveError, match="exact uint8 scorer geometry"):
        realize_factor2_and_require_hard_oracle(
            operator,
            np.zeros(shape, dtype=np.uint8),
            lambda frames: _hard_decision(frames),
        )


def test_chunk_resume_is_equivalent_and_refuses_input_drift(tmp_path: Path) -> None:
    hashes = {pair_id: hashlib.sha256(f"pair:{pair_id}".encode()).hexdigest() for pair_id in (1, 2, 3)}
    state = tmp_path / "resume.json"
    stages = tmp_path / "stages"
    calls: list[int] = []

    def interrupted(pair_id: int) -> dict[str, int]:
        calls.append(pair_id)
        if pair_id == 2:
            raise RuntimeError("synthetic interruption")
        return {"value": pair_id * 2}

    with pytest.raises(RuntimeError, match="interruption"):
        run_resumable_pair_chunk(
            pair_ids=[1, 2, 3],
            config={"band": 0.5},
            state_path=state,
            stage_dir=stages,
            derive_input_hash=hashes.__getitem__,
            solve_pair=interrupted,
        )
    assert calls == [1, 2]

    resumed_calls: list[int] = []
    resumed = run_resumable_pair_chunk(
        pair_ids=[1, 2, 3],
        config={"band": 0.5},
        state_path=state,
        stage_dir=stages,
        derive_input_hash=hashes.__getitem__,
        solve_pair=lambda pair_id: resumed_calls.append(pair_id) or {"value": pair_id * 2},
    )
    assert resumed_calls == [2, 3]
    assert resumed == [{"value": 2}, {"value": 4}, {"value": 6}]

    hashes[1] = "0" * 64
    with pytest.raises(ConstructiveSolveError, match="input/stage custody drift"):
        run_resumable_pair_chunk(
            pair_ids=[1, 2, 3],
            config={"band": 0.5},
            state_path=state,
            stage_dir=stages,
            derive_input_hash=hashes.__getitem__,
            solve_pair=lambda pair_id: {"value": pair_id},
        )


def test_chunk_output_is_repeated_byte_deterministic(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"pair9").hexdigest()
    kwargs = {
        "pair_ids": [9],
        "config": {"weights": [1, 2, 3], "mode": "zero-band"},
        "derive_input_hash": lambda pair_id: digest,
        "solve_pair": lambda pair_id: {"pair_id": pair_id, "status": "exact"},
    }
    run_resumable_pair_chunk(**kwargs, state_path=tmp_path / "a.json", stage_dir=tmp_path / "a")
    run_resumable_pair_chunk(**kwargs, state_path=tmp_path / "b.json", stage_dir=tmp_path / "b")
    stage_a = next((tmp_path / "a").glob("*.json")).read_bytes()
    stage_b = next((tmp_path / "b").glob("*.json")).read_bytes()
    assert stage_a == stage_b


@pytest.mark.parametrize("pair_id", [True, 1.0, np.int64(1)])
def test_chunk_pair_ids_require_exact_non_bool_int(tmp_path: Path, pair_id: object) -> None:
    with pytest.raises(ConstructiveSolveError, match="exact non-bool"):
        run_resumable_pair_chunk(
            pair_ids=cast("list[int]", [pair_id]),
            config={"band": 0.0},
            state_path=tmp_path / "resume.json",
            stage_dir=tmp_path / "stages",
            derive_input_hash=lambda value: hashlib.sha256(str(value).encode()).hexdigest(),
            solve_pair=lambda value: {"pair_id": value},
        )


def test_chunk_resume_refuses_boolean_pair_id_in_preserved_state(tmp_path: Path) -> None:
    state = tmp_path / "resume.json"
    digest = hashlib.sha256(b"pair:1").hexdigest()
    kwargs = {
        "pair_ids": [1],
        "config": {"band": 0.0},
        "state_path": state,
        "stage_dir": tmp_path / "stages",
        "derive_input_hash": lambda pair_id: digest,
        "solve_pair": lambda pair_id: {"pair_id": pair_id},
    }
    run_resumable_pair_chunk(**kwargs)
    preserved = json.loads(state.read_text())
    preserved["pair_ids"] = [True]
    state.write_text(json.dumps(preserved, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ConstructiveSolveError, match="resume config/pair drift"):
        run_resumable_pair_chunk(**kwargs)
