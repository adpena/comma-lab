# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import measure_c2_integer_plane_emitter as m


def _write_fake_custody(tmp_path: Path) -> tuple[Path, str, Path]:
    sidecar = tmp_path / "pair_0000.vjp.npz"
    sidecar.write_bytes(b"immutable-vjp-fixture")
    sidecar_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    manifest = {
        "schema": "vjp_custody_manifest.v1",
        "source_hashes": dict(m.EXPECTED_VJP_SOURCE_HASHES),
        "pair_ids": [0],
        "sidecars": [
            {
                "pair_id": 0,
                "path": str(sidecar),
                "bytes": sidecar.stat().st_size,
                "sha256": sidecar_hash,
                "tensor_hashes": {
                    name: hashlib.sha256(name.encode()).hexdigest()
                    for name in m.EXPECTED_VJP_TENSOR_HASH_KEYS
                },
            }
        ],
    }
    document = tmp_path / "manifest.json"
    document.write_text(json.dumps(manifest), encoding="utf-8")
    return document, hashlib.sha256(document.read_bytes()).hexdigest(), sidecar


def test_frozen_n24_is_unique_and_primary_n6_is_prefix() -> None:
    assert len(m.FROZEN_N24) == 24
    assert len(set(m.FROZEN_N24)) == 24
    assert m.PRIMARY_N6 == m.FROZEN_N24[:6] == (0, 1, 2, 3, 4, 5)


def test_select_n24_rows_enforces_hard_oracle_floor() -> None:
    with pytest.raises(m.C2MeasureError, match=r"6\.\.24"):
        m._select_n24_rows(5)
    with pytest.raises(m.C2MeasureError, match=r"6\.\.24"):
        m._select_n24_rows(25)
    assert m._select_n24_rows(6) == m.PRIMARY_N6


def test_timing_summary_reports_median_and_p95() -> None:
    result = m._timing_summary(
        [1.0, 2.0, 3.0],
        setup_seconds=0.5,
        total_seconds=7.0,
        setup_scope="load",
        iteration_scope="score one pair",
    )
    assert result["completed_iterations"] == 3
    assert result["median_seconds_per_pair_iteration"] == 2.0
    assert result["p95_seconds_per_pair_iteration"] == pytest.approx(2.9)
    assert result["setup_scope"] == "load"
    assert result["iteration_scope"] == "score one pair"


def test_timing_summary_refuses_negative_sample() -> None:
    with pytest.raises(m.C2MeasureError, match="finite and nonnegative"):
        m._timing_summary(
            [0.1, -0.2],
            setup_seconds=0.0,
            total_seconds=1.0,
            setup_scope="load",
            iteration_scope="score",
        )


def test_durable_output_refuses_tmp_and_sacred() -> None:
    with pytest.raises(m.C2MeasureError, match="repository research root"):
        m._durable_output(Path("/tmp/c2.json"))
    with pytest.raises(m.C2MeasureError, match="sacred"):
        m._durable_output(m.SACRED / "c2.json")


def test_write_once_json_is_atomic_and_non_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "RESEARCH_ROOT", tmp_path)
    output = tmp_path / "receipt.json"
    reference = m._write_once_json(output, {"schema": "fixture", "score_claim": False})
    assert reference["path"] == str(output.resolve())
    assert reference["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(m.C2MeasureError, match="already exists"):
        m._write_once_json(output, {"schema": "fixture", "score_claim": False})


def test_validate_vjp_custody_rechecks_manifest_sidecar_and_embedded_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_hash, sidecar = _write_fake_custody(tmp_path)
    calls: list[int] = []

    def canonical_loader(row: dict, _manifest: dict, **kwargs: object) -> SimpleNamespace:
        calls.append(int(row["pair_id"]))
        assert kwargs == {"scorer_hw": m.SCORER_HW, "camera_hw": m.CAMERA_HW}
        return SimpleNamespace(pair_id=int(row["pair_id"]))

    monkeypatch.setattr(m, "load_vjp_pair_row", canonical_loader)
    result = m.validate_vjp_custody(
        manifest,
        [0],
        expected_sha256=manifest_hash,
    )
    assert result["manifest_sha256"] == manifest_hash
    assert result["selected_sidecars"][0]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert result["decoder_serialization"] is False
    assert result["candidate_admission"] is False
    assert result["selected_sidecars"][0]["hash_rechecked"] is True
    assert result["selected_sidecars"][0]["embedded_pair_id_rechecked"] is True
    assert result["selected_sidecars"][0]["custody_json_rechecked"] is True
    assert result["selected_sidecars"][0]["tensor_bytes_rehashed"] is True
    assert calls == [0]


def test_validate_vjp_custody_propagates_canonical_tensor_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, manifest_hash, _ = _write_fake_custody(tmp_path)

    def reject(*_args: object, **_kwargs: object) -> None:
        raise m.VJPCustodyError("manifest tensor hash failed for winner")

    monkeypatch.setattr(m, "load_vjp_pair_row", reject)
    with pytest.raises(m.C2MeasureError, match="canonical tensor custody failed"):
        m.validate_vjp_custody(manifest, [0], expected_sha256=manifest_hash)


def test_two_manifest_contract_validates_secondary_route_even_for_primary_n6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, tuple[int, ...], str]] = []

    def validate(
        path: Path, selected: tuple[int, ...], *, expected_sha256: str
    ) -> dict[str, object]:
        calls.append((path, tuple(selected), expected_sha256))
        return {
            "manifest": str(path),
            "manifest_sha256": expected_sha256,
            "selected_sidecars": [{"pair_id": pair} for pair in selected],
            "decoder_serialization": False,
            "candidate_admission": False,
        }

    monkeypatch.setattr(m, "validate_vjp_custody", validate)
    result = m.validate_vjp_custody_set(Path("primary.json"), Path("secondary.json"), (0, 1))
    assert calls == [
        (Path("primary.json"), (0, 1), m.EXPECTED_VJP_MANIFEST_SHA256),
        (Path("secondary.json"), (), m.EXPECTED_VJP_MANIFEST_SECONDARY_SHA256),
    ]
    assert len(result["manifests"]) == 2
    assert [row["pair_id"] for row in result["selected_sidecars"]] == [0, 1]


def test_validate_vjp_custody_refuses_manifest_hash_mismatch(tmp_path: Path) -> None:
    manifest, _, _ = _write_fake_custody(tmp_path)
    with pytest.raises(m.C2MeasureError, match="manifest is missing or hash-mismatched"):
        m.validate_vjp_custody(manifest, [0], expected_sha256="0" * 64)


def test_validate_vjp_custody_refuses_uncovered_pair(tmp_path: Path) -> None:
    manifest, manifest_hash, _ = _write_fake_custody(tmp_path)
    with pytest.raises(m.C2MeasureError, match="not covered"):
        m.validate_vjp_custody(manifest, [1], expected_sha256=manifest_hash)


def test_validate_vjp_custody_refuses_mutated_sidecar(tmp_path: Path) -> None:
    manifest, manifest_hash, sidecar = _write_fake_custody(tmp_path)
    sidecar.write_bytes(b"mutated-but-same-path")
    with pytest.raises(m.C2MeasureError, match=r"byte custody mismatch|hash custody mismatch"):
        m.validate_vjp_custody(manifest, [0], expected_sha256=manifest_hash)


def test_validate_vjp_custody_refuses_source_hash_drift(tmp_path: Path) -> None:
    manifest, _, _ = _write_fake_custody(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["source_hashes"]["cache_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(m.C2MeasureError, match="source_hashes mismatch"):
        m.validate_vjp_custody(manifest, [0], expected_sha256=manifest_hash)


def test_validate_vjp_custody_refuses_incomplete_tensor_hashes(tmp_path: Path) -> None:
    manifest, _, _ = _write_fake_custody(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["sidecars"][0]["tensor_hashes"].pop("winner")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(m.C2MeasureError, match="tensor_hashes are incomplete"):
        m.validate_vjp_custody(manifest, [0], expected_sha256=manifest_hash)


def test_cache_loader_refuses_unpinned_bytes(tmp_path: Path) -> None:
    cache = tmp_path / "gt_n600.npz"
    cache.write_bytes(b"not-the-frozen-cache")
    with pytest.raises(m.C2MeasureError, match="cache SHA custody mismatch"):
        m._load_real_cache(cache)


def test_metal_status_is_explicit_and_never_infers_timing() -> None:
    status = m._metal_environment_status()
    assert status["status"] in {
        "READY_METAL",
        "BLOCKED_ENVIRONMENT_NO_METAL",
        "BLOCKED_ENVIRONMENT_MLX_NOT_IMPORTABLE",
        "BLOCKED_EXECUTION_MLX_PROBE",
    }
    if status["status"] != "READY_METAL":
        assert status["timing"]["completed_iterations"] == 0
        assert status["timing"]["total_seconds"] is None
        assert status["exact_numpy_parity"] == "NOT_MEASURED"


def test_mlx_probe_blockers_do_not_misclassify_api_failures_as_no_metal() -> None:
    missing = m._mlx_probe_blocker(
        ModuleNotFoundError("mlx"), device=None, setup_seconds=0.1
    )
    no_metal = m._mlx_probe_blocker(
        RuntimeError("metal::load_device: No Metal device available"),
        device=None,
        setup_seconds=0.1,
    )
    api_failure = m._mlx_probe_blocker(
        AttributeError("missing API"), device="gpu(0)", setup_seconds=0.1
    )
    assert missing["status"] == "BLOCKED_ENVIRONMENT_MLX_NOT_IMPORTABLE"
    assert no_metal["status"] == "BLOCKED_ENVIRONMENT_NO_METAL"
    assert api_failure["status"] == "BLOCKED_EXECUTION_MLX_PROBE"


def test_mlx_execution_blocker_is_distinct_from_environment_absence() -> None:
    status = m._mlx_execution_blocker(RuntimeError("kernel failed"), device="gpu(0)")
    assert status["status"] == "BLOCKED_EXECUTION_MLX_METAL"
    assert status["metal_device_ready"] is True
    assert status["fast_path_claim"] is False
    assert "kernel failed" in status["blocker"]


def test_mlx_cpu_distortion_diagnostic_binds_actual_pair_ids_without_verdict() -> None:
    cpu = [
        {"pair_id": 24, "hard_oracle": {"d_seg": 0.01, "d_pose": 0.002}},
        {"pair_id": 3, "hard_oracle": {"d_seg": 0.02, "d_pose": 0.003}},
    ]
    mlx = [
        {"pair_id": 24, "d_seg": 0.0101, "d_pose": 0.00201},
        {"pair_id": 3, "d_seg": 0.0201, "d_pose": 0.00301},
    ]
    bound = m._bind_mlx_cpu_distortion_diagnostic(mlx, cpu)
    assert [row["pair_id"] for row in bound] == [24, 3]
    assert all("diagnostic only" in row["comparison_authority"] for row in bound)
    assert all("within_explicit_tolerance" not in row for row in bound)

    with pytest.raises(m.C2MeasureError, match="no CPU-Torch authority row"):
        m._bind_mlx_cpu_distortion_diagnostic(
            [{"pair_id": 8, "d_seg": 0.0, "d_pose": 0.0}], cpu
        )


def test_mlx_cpu_distortion_diagnostic_never_mints_parity_pass() -> None:
    cpu = [{"pair_id": 5, "hard_oracle": {"d_seg": 0.0, "d_pose": 0.0}}]
    mlx = [{"pair_id": 5, "d_seg": 1.0, "d_pose": 1.0}]
    bound = m._bind_mlx_cpu_distortion_diagnostic(mlx, cpu)
    assert bound[0]["d_seg_abs_delta"] == 1.0
    assert bound[0]["d_pose_abs_delta"] == 1.0
    assert "passed" not in bound[0]


def test_exact_source_plane_constant_frame_is_exact_uint8() -> None:
    operator = m.DisjointResizeOperator.build(
        camera_h=m.CAMERA_HW[0],
        camera_w=m.CAMERA_HW[1],
        scorer_h=m.SCORER_HW[0],
        scorer_w=m.SCORER_HW[1],
    )
    frame = np.full((*m.CAMERA_HW, 3), 137, dtype=np.uint8)
    plane = m._exact_source_plane(operator, frame)
    assert plane.dtype == np.uint8
    assert plane.shape == m.SCORER_SHAPE
    assert np.all(plane == 137)


def test_parser_has_only_explicit_fixture_and_advisory_modes() -> None:
    fixture = m._parser().parse_args(["fixture"])
    advisory = m._parser().parse_args(["n24-advisory"])
    assert fixture.mode == "fixture"
    assert advisory.mode == "n24-advisory"
    assert advisory.completed_rows == 6
    assert advisory.cpu_threads == 1
    assert advisory.vjp_manifest_secondary == m.DEFAULT_VJP_MANIFEST_SECONDARY


def test_parser_has_no_vjp_rehash_bypass() -> None:
    with pytest.raises(SystemExit):
        m._parser().parse_args(["n24-advisory", "--skip-vjp-sidecar-rehash"])


def test_frozen_scorer_custody_pins_both_sources_and_weights() -> None:
    assert m.EXPECTED_CACHE_SHA256 == "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
    assert m.EXPECTED_VJP_MANIFEST_SHA256 == "3d1218a52ededc4b347ae94c5c2bf58d06d70dd8f530bec67bf9cab36ee00694"
    assert m.EXPECTED_VJP_MANIFEST_SECONDARY_SHA256 == "200e8cfa375cbdb8154777156441ae6adadf33e75668c86cc52b816f79488e94"
    assert m.EXPECTED_MODULES_SHA256 == "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa"
    assert m.EXPECTED_FRAME_UTILS_SHA256 == "d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90"
    assert m.EXPECTED_SEGNET_SHA256 == "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
    assert m.EXPECTED_POSENET_SHA256 == "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"


def test_code_custody_binds_git_and_every_mlx_measurement_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m,
        "_git_output",
        lambda *args: "1" * 40 if args == ("rev-parse", "HEAD") else " M tracked.py",
    )
    custody = m._code_custody(include_mlx_scorer=True)
    assert custody["git_head"] == "1" * 40
    assert custody["git_worktree_dirty"] is True
    assert set(custody["source_files"]) == {
        "measurement_tool",
        "integer_plane_emitter",
        "integer_lattice_solver",
        "npy_memmap_loader",
        "uint8_ste",
        "mlx_scorer_adapter",
        "mlx_scorer_profile",
        "mlx_roundtrip_primitives",
        "mlx_roundtrip_runtime",
        "mlx_roundtrip_contract",
        "canonical_kernels",
        "scorer_loader",
        "vjp_custody_validator",
        "mlx_torch_scorer_parity",
    }
    assert all(
        len(record["sha256"]) == 64 for record in custody["source_files"].values()
    )


def test_main_fixture_dispatches_without_score_surface(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        m,
        "run_fixture",
        lambda **_: {"schema": m.SCHEMA, "mode": "fixture", "score_claim": False},
    )
    assert m.main(["fixture", "--output", "unused.json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"mode": "fixture", "schema": m.SCHEMA, "score_claim": False}
