from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.vjp_custody import (
    ACTIVE_ARRANGEMENT,
    EXPECTED_HASHES,
    MANIFEST_SCHEMA,
    RECEIVER_ARITHMETIC,
    REPRESENTATION,
    VJPCustodyError,
    atomic_json,
    canonical_json,
    factor_vjp,
    largest_feasible_pose_step,
    linearized_pose_delta6,
    load_vjp_manifest,
    recover_pair_sidecar_row,
    write_pair_sidecar,
)
from tools import measure_joint_seg_pose_rate as measurement
from tools import produce_vjp_custody as producer


def test_factor_vjp_is_exact_and_zero_safe() -> None:
    gradient = np.array(
        [[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32
    )
    q, local_lipschitz = factor_vjp(gradient)
    np.testing.assert_array_equal(local_lipschitz, [[5.0, 0.0]])
    np.testing.assert_allclose(q[0, 0], [0.6, 0.8, 0.0])
    np.testing.assert_array_equal(q[0, 1], 0.0)
    np.testing.assert_allclose(local_lipschitz[..., None] * q, gradient)


def test_largest_pose_step_solves_quadratic_boundary() -> None:
    jacobian = np.zeros((6, 2, 1, 1, 3), dtype=np.float32)
    jacobian[:, 0, 0, 0, 0] = 1.0
    direction = np.array([[[1.0, 0.0, 0.0]]])
    result = largest_feasible_pose_step(
        jacobian, np.zeros_like(direction), direction, np.zeros(6), tau_pose=0.25
    )
    assert result["feasible"] is True
    assert result["selected_step"] == pytest.approx(0.5)
    assert result["planned_predictor_step_pose_mse"] == pytest.approx(0.25)


def test_largest_pose_step_includes_nonzero_source_pose_base_debt() -> None:
    jacobian = np.zeros((6, 2, 1, 1, 3), dtype=np.float32)
    jacobian[:, 0, 0, 0, 0] = 1.0
    direction = np.array([[[1.0, 0.0, 0.0]]])
    source_base_delta = np.full(6, 0.25)
    result = largest_feasible_pose_step(
        jacobian,
        np.zeros_like(direction),
        direction,
        source_base_delta,
        tau_pose=0.25,
    )
    assert result["feasible"] is True
    assert result["selected_step"] == pytest.approx(0.25)
    np.testing.assert_allclose(result["source_pose_base_delta6"], source_base_delta)
    np.testing.assert_allclose(result["fixed_pose_delta6"], source_base_delta)
    assert result["planned_predictor_step_pose_mse"] == pytest.approx(0.25)


def test_pose_step_refuses_when_fixed_frame1_debt_is_infeasible() -> None:
    jacobian = np.zeros((6, 2, 1, 1, 3), dtype=np.float32)
    jacobian[:, 1, 0, 0, 0] = 1.0
    delta = np.array([[[1.0, 0.0, 0.0]]])
    result = largest_feasible_pose_step(
        jacobian, delta, np.zeros_like(delta), np.zeros(6), tau_pose=0.5
    )
    assert result["feasible"] is False
    assert result["selected_step"] is None


def test_actual_lattice_pose_linearization_uses_both_solved_frames_and_base() -> None:
    jacobian = np.zeros((6, 2, 1, 1, 3), dtype=np.float32)
    jacobian[:, 0, 0, 0, 0] = 2.0
    jacobian[:, 1, 0, 0, 1] = 3.0
    d0 = np.array([[[0.5, 0.0, 0.0]]])
    d1 = np.array([[[0.0, 0.25, 0.0]]])
    got = linearized_pose_delta6(jacobian, d0, d1, np.full(6, 0.1))
    np.testing.assert_allclose(got, np.full(6, 1.85))


def _tiny_arrays(pair_id: int = 7) -> dict[str, object]:
    scorer_hw, camera_hw = (2, 3), (4, 5)
    gradient = np.arange(18, dtype=np.float32).reshape(*scorer_hw, 3)
    gradient[0, 0] = 0.0
    q, local_lipschitz = factor_vjp(gradient)
    return {
        "pair_id": pair_id,
        "winner": np.zeros(scorer_hw, dtype=np.int8),
        "rival": np.ones(scorer_hw, dtype=np.int8),
        "cached_margin": np.ones(scorer_hw, dtype=np.float32),
        "native_margin": np.ones(scorer_hw, dtype=np.float32),
        "head_pair_norms": np.ones(scorer_hw, dtype=np.float32),
        "seg_g_y": gradient,
        "seg_g_x": np.ones((*camera_hw, 3), dtype=np.float32),
        "seg_q": q,
        "seg_local_lipschitz": local_lipschitz,
        "pose_j_y": np.ones((6, 2, *scorer_hw, 3), dtype=np.float32),
        "pose_j_x": np.ones((6, 2, *camera_hw, 3), dtype=np.float32),
        "checks": {"synthetic_fixture": True},
    }


def _tiny_manifest(tmp_path: Path) -> Path:
    arrays = _tiny_arrays()
    row = write_pair_sidecar(tmp_path / "pair_0007.vjp.npz", arrays, EXPECTED_HASHES)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "pair_ids": [7],
        "receiver_arithmetic": RECEIVER_ARITHMETIC,
        "active_arrangement": ACTIVE_ARRANGEMENT,
        "representation": REPRESENTATION,
        "source_hashes": EXPECTED_HASHES,
        "sidecars": [row],
    }
    manifest["manifest_content_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    return path


def test_manifest_loads_per_pair_hashes_shapes_and_representation(tmp_path: Path) -> None:
    path = _tiny_manifest(tmp_path)
    rows = load_vjp_manifest(path, [7], scorer_hw=(2, 3), camera_hw=(4, 5))
    assert rows[7].pose_j_y.shape == (6, 2, 2, 3, 3)
    assert rows[7].metadata["representation"] == REPRESENTATION


def test_positive_band_telemetry_summarizes_oracle_attempts_and_vjp_fields(
    tmp_path: Path,
) -> None:
    attempts = [
        {"hard_oracle": None, "PASS": False},
        {"hard_oracle": {"d_seg": 0.1}, "PASS": False},
        {"hard_oracle": {"d_seg": 0.0}, "PASS": True},
    ]
    oracle = measurement._hard_oracle_band_proposal_summary(attempts)
    assert oracle == {
        "attempt_count": 3,
        "hard_oracle_evaluated_count": 2,
        "hard_oracle_admit_count": 1,
        "hard_oracle_reject_count": 1,
        "hard_oracle_not_evaluated_count": 1,
        "hard_oracle_admit_rate_over_attempts": pytest.approx(1 / 3),
        "hard_oracle_reject_rate_over_attempts": pytest.approx(1 / 3),
        "hard_oracle_not_evaluated_rate_over_attempts": pytest.approx(1 / 3),
        "hard_oracle_admit_rate_over_evaluations": pytest.approx(0.5),
    }

    rows = load_vjp_manifest(
        _tiny_manifest(tmp_path), [7], scorer_hw=(2, 3), camera_hw=(4, 5)
    )
    fields = measurement._vjp_field_summary(rows[7])
    assert fields["measured_lip_local"]["zero_count"] == 1
    assert fields["q_unit_norm_error_on_positive_lip"]["count"] == 5
    assert fields["q_unit_norm_error_on_positive_lip"]["max"] < 1e-6
    assert fields["q_norm_max_on_zero_lip"] == 0.0


def test_manifest_rejects_tampered_pair_hash(tmp_path: Path) -> None:
    path = _tiny_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["sidecars"][0]["sha256"] = "0" * 64
    manifest["manifest_content_sha256"] = hashlib.sha256(
        canonical_json({key: value for key, value in manifest.items() if key != "manifest_content_sha256"})
    ).hexdigest()
    atomic_json(path, manifest)
    with pytest.raises(VJPCustodyError, match="hash/path custody"):
        load_vjp_manifest(path, [7], scorer_hw=(2, 3), camera_hw=(4, 5))


def test_manifest_rejects_pair_order_mismatch(tmp_path: Path) -> None:
    path = _tiny_manifest(tmp_path)
    with pytest.raises(VJPCustodyError, match="pair ids/order"):
        load_vjp_manifest(path, [8], scorer_hw=(2, 3), camera_hw=(4, 5))


def test_recover_orphan_sidecar_validates_custody_without_rewrite(tmp_path: Path) -> None:
    manifest_path = _tiny_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["sidecars"] = []
    manifest.pop("manifest_content_sha256")
    sidecar = tmp_path / "pair_0007.vjp.npz"
    before = (sidecar.stat().st_mtime_ns, hashlib.sha256(sidecar.read_bytes()).hexdigest())
    row = recover_pair_sidecar_row(
        sidecar,
        7,
        manifest,
        scorer_hw=(2, 3),
        camera_hw=(4, 5),
    )
    after = (sidecar.stat().st_mtime_ns, hashlib.sha256(sidecar.read_bytes()).hexdigest())
    assert before == after
    assert row["sha256"] == before[1]

    bad_manifest = {**manifest, "source_hashes": {**EXPECTED_HASHES, "cache_sha256": "0" * 64}}
    with pytest.raises(VJPCustodyError, match="source_hashes mismatch"):
        recover_pair_sidecar_row(
            sidecar,
            7,
            bad_manifest,
            scorer_hw=(2, 3),
            camera_hw=(4, 5),
        )


def test_producer_resume_appends_valid_orphan_after_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Path("/synthetic/frozen-cache.npz")
    upstream = Path("/synthetic/upstream")
    manifest = producer._manifest_base(
        pair_ids=[7],
        cache=cache,
        upstream=upstream,
        hashes=EXPECTED_HASHES,
        output_dir=tmp_path,
    )
    producer.atomic_json(tmp_path / "manifest.json", manifest)
    sidecar = tmp_path / "pair_0007.vjp.npz"
    write_pair_sidecar(sidecar, _tiny_arrays(), EXPECTED_HASHES)
    before = (sidecar.stat().st_mtime_ns, hashlib.sha256(sidecar.read_bytes()).hexdigest())

    monkeypatch.setattr(producer, "_enforce_output_tier", lambda output, pairs_remaining: output)
    monkeypatch.setattr(producer, "source_hashes", lambda actual_cache, actual_upstream: EXPECTED_HASHES)
    monkeypatch.setattr(producer, "_stat_tree_snapshot", lambda path: {"unchanged": True})
    original_recover = recover_pair_sidecar_row

    def recover_tiny(path: Path, pair_id: int, active_manifest: dict[str, object]) -> dict[str, object]:
        return original_recover(
            path,
            pair_id,
            active_manifest,
            scorer_hw=(2, 3),
            camera_hw=(4, 5),
        )

    monkeypatch.setattr(producer, "recover_pair_sidecar_row", recover_tiny)
    monkeypatch.setattr(
        producer,
        "_load_cache",
        lambda path: pytest.fail("completed orphan recovery must not load the real cache"),
    )
    monkeypatch.setattr(
        producer,
        "_load_scorers",
        lambda path, threads: pytest.fail("completed orphan recovery must not load scorers"),
    )
    result = producer.produce(
        argparse.Namespace(
            pair_indices=[7],
            cache=cache,
            upstream=upstream,
            output_dir=tmp_path,
            resume=True,
            cpu_threads=1,
        )
    )
    after = (sidecar.stat().st_mtime_ns, hashlib.sha256(sidecar.read_bytes()).hexdigest())
    assert before == after
    assert [row["pair_id"] for row in result["sidecars"]] == [7]
    assert result["sidecars"][0]["sha256"] == before[1]


def test_producer_persists_scoped_immutable_refusal_and_keeps_manifest_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Path("/synthetic/frozen-cache.npz")
    upstream = Path("/synthetic/upstream")
    literal_error = (
        "pair 11 active arrangement incompatible: "
        "cached/native winner mismatch pixels=1"
    )
    fields = {
        "gt_f0": np.zeros((12, 1, 1, 3), dtype=np.uint8),
        "gt_f1": np.zeros((12, 1, 1, 3), dtype=np.uint8),
        "lstars": np.zeros((12, 1, 1), dtype=np.int8),
        "margins": np.zeros((12, 1, 1), dtype=np.float32),
    }
    monkeypatch.setattr(producer, "_enforce_output_tier", lambda output, pairs_remaining: output)
    monkeypatch.setattr(producer, "source_hashes", lambda actual_cache, actual_upstream: EXPECTED_HASHES)
    monkeypatch.setattr(producer, "_stat_tree_snapshot", lambda path: {"unchanged": True})
    monkeypatch.setattr(producer, "_load_cache", lambda path: fields)
    monkeypatch.setattr(producer, "_load_scorers", lambda path, threads: (object(), object(), object()))

    def refuse(**kwargs: object) -> dict[str, object]:
        raise VJPCustodyError(literal_error)

    monkeypatch.setattr(producer, "compute_pair_derivatives", refuse)
    args = argparse.Namespace(
        pair_indices=[11],
        cache=cache,
        upstream=upstream,
        output_dir=tmp_path,
        resume=False,
        cpu_threads=1,
    )
    with pytest.raises(VJPCustodyError, match="winner mismatch pixels=1"):
        producer.produce(args)

    refusal_path = tmp_path / "pair_0011.vjp_refusal.json"
    refusal = json.loads(refusal_path.read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    before = (refusal_path.stat().st_mtime_ns, hashlib.sha256(refusal_path.read_bytes()).hexdigest())
    assert refusal == {
        "schema": producer.REFUSAL_SCHEMA,
        "pair_id": 11,
        "error": literal_error,
        "verdict_scope": producer.REFUSAL_VERDICT_SCOPE,
        "source_hashes": EXPECTED_HASHES,
        "config": manifest["config"],
        "config_sha256": manifest["config_sha256"],
        "authority": {
            "score_claim": False,
            "pointer": "0.1910828242 [contest-CPU] UNMOVED",
            "pointer_moved": False,
        },
    }
    assert manifest["sidecars"] == []
    assert manifest["completed_at_utc"] is None
    assert "manifest_content_sha256" not in manifest
    assert manifest["refusals"] == [
        {
            "pair_id": 11,
            "path": str(refusal_path.resolve()),
            "bytes": refusal_path.stat().st_size,
            "sha256": before[1],
            "verdict_scope": producer.REFUSAL_VERDICT_SCOPE,
        }
    ]

    args.resume = True
    with pytest.raises(VJPCustodyError, match="winner mismatch pixels=1"):
        producer.produce(args)
    after = (refusal_path.stat().st_mtime_ns, hashlib.sha256(refusal_path.read_bytes()).hexdigest())
    assert after == before
    with pytest.raises(VJPCustodyError, match="non-byte-identical"):
        producer._write_immutable_json(
            refusal_path,
            {**refusal, "error": "different refusal must not replace custody"},
            resume=True,
        )
    assert hashlib.sha256(refusal_path.read_bytes()).hexdigest() == before[1]


def _producer_manifest(
    root: Path,
    pair_ids: list[int],
    completed_ids: list[int],
    *,
    complete: bool,
    config_overrides: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    manifest = producer._manifest_base(
        pair_ids=pair_ids,
        cache=Path("/synthetic/frozen-cache.npz"),
        upstream=Path("/synthetic/upstream"),
        hashes=EXPECTED_HASHES,
        output_dir=root,
    )
    if config_overrides:
        manifest["config"].update(config_overrides)
        manifest["config_sha256"] = hashlib.sha256(canonical_json(manifest["config"])).hexdigest()
    for pair_id in completed_ids:
        manifest["sidecars"].append(
            write_pair_sidecar(root / f"pair_{pair_id:04d}.vjp.npz", _tiny_arrays(pair_id), EXPECTED_HASHES)
        )
    if complete:
        manifest["completed_at_utc"] = "2026-07-19T00:00:00+00:00"
        manifest["manifest_content_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    path = root / "manifest.json"
    atomic_json(path, manifest)
    return path, manifest


def test_compose_partial_and_complete_manifests_is_zero_copy_and_consumer_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial_path, partial = _producer_manifest(
        tmp_path / "partial", [7, 8], [7], complete=False
    )
    refusal_error = VJPCustodyError("pair 8 synthetic active-arrangement refusal")
    producer._record_pair_refusal(
        output_dir=partial_path.parent,
        manifest_path=partial_path,
        manifest=partial,
        pair_id=8,
        error=refusal_error,
        resume=False,
    )
    complete_path, _ = _producer_manifest(
        tmp_path / "complete",
        [9],
        [9],
        complete=True,
        config_overrides={"producer_sha256": "f" * 64},
    )
    sidecars = [partial_path.parent / "pair_0007.vjp.npz", complete_path.parent / "pair_0009.vjp.npz"]
    before = {
        path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sidecars
    }
    monkeypatch.setattr(producer, "_enforce_output_tier", lambda output, pairs_remaining: output)
    monkeypatch.setattr(producer, "_stat_tree_snapshot", lambda path: {"unchanged": True})
    original_load = producer.load_vjp_pair_row

    def load_tiny(row: dict[str, object], manifest: dict[str, object]) -> object:
        return original_load(row, manifest, scorer_hw=(2, 3), camera_hw=(4, 5))

    monkeypatch.setattr(producer, "load_vjp_pair_row", load_tiny)
    output_dir = tmp_path / "composed"
    result = producer.compose_manifests(
        argparse.Namespace(
            pair_indices=[7, 9],
            compose_manifests=[partial_path, complete_path],
            output_dir=output_dir,
        )
    )
    after = {
        path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sidecars
    }
    assert after == before
    assert list(output_dir.iterdir()) == [output_dir / "manifest.json"]
    assert result["composition"] == {
        "zero_copy": True,
        "requested_selected_pair_ids": [7, 9],
        "omitted_pair_ids": [8],
        "refused_pair_ids": [8],
        "refusal_scope": producer.REFUSAL_VERDICT_SCOPE,
        "refusals_are_family_negatives": False,
    }
    assert result["source_manifests"] == [
        {"path": str(partial_path.resolve()), "sha256": hashlib.sha256(partial_path.read_bytes()).hexdigest()},
        {"path": str(complete_path.resolve()), "sha256": hashlib.sha256(complete_path.read_bytes()).hexdigest()},
    ]
    source_manifests = [json.loads(path.read_text()) for path in (partial_path, complete_path)]
    assert source_manifests[0]["config"]["producer_sha256"] != source_manifests[1]["config"]["producer_sha256"]
    for source, record in zip(source_manifests, result["source_manifests"], strict=True):
        assert source["config_sha256"] == hashlib.sha256(canonical_json(source["config"])).hexdigest()
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    assert "producer_sha256" not in result["config"]["source_common_config"]
    assert result["config"]["source_common_config"]["library_sha256"] == source_manifests[0]["config"][
        "library_sha256"
    ]
    assert [row["path"] for row in result["sidecars"]] == [str(path.resolve()) for path in sidecars]
    loaded = load_vjp_manifest(
        output_dir / "manifest.json", [7, 9], scorer_hw=(2, 3), camera_hw=(4, 5)
    )
    assert list(loaded) == [7, 9]
    with pytest.raises(VJPCustodyError, match="destination already exists"):
        producer.compose_manifests(
            argparse.Namespace(
                pair_indices=[7, 9],
                compose_manifests=[partial_path, complete_path],
                output_dir=output_dir,
            )
        )
    incompatible_path, _ = _producer_manifest(
        tmp_path / "incompatible",
        [10],
        [10],
        complete=True,
        config_overrides={
            "producer_sha256": "e" * 64,
            "library_sha256": "0" * 64,
        },
    )
    with pytest.raises(VJPCustodyError, match="incompatible producer configs"):
        producer.compose_manifests(
            argparse.Namespace(
                pair_indices=[7, 10],
                compose_manifests=[partial_path, incompatible_path],
                output_dir=tmp_path / "incompatible_composed",
            )
        )
