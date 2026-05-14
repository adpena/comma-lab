# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from tac.sensitivity_map import save_sensitivity_map


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "build_component_response_perturbation_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_component_response_perturbation_plan",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _baseline_members() -> dict[str, bytes]:
    renderer = bytearray(b"ASYM" + bytes(range(4, 80)))
    renderer[8] = 100
    return {
        "renderer.bin": bytes(renderer),
        "masks.mkv": b"mask-bytes",
        "optimized_poses.pt": b"pose-bytes",
    }


def _read_member(archive: Path, member: str) -> bytes:
    with zipfile.ZipFile(archive, "r") as zf:
        return zf.read(member)


def _prediction_artifact(
    module,
    path: Path,
    *,
    baseline: Path,
    atoms: list[object],
) -> Path:
    baseline_meta = module._file_meta(baseline)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "official_component_response_prediction_deltas_v1",
                "producer": "test",
                "prediction_source": {
                    "source_kind": "component_sensitivity_map_projection",
                    "baseline_archive": {
                        "bytes": baseline_meta["bytes"],
                        "sha256": baseline_meta["sha256"],
                    },
                    "perturbation_basis": {
                        "atom_set_sha256": module._atom_set_sha256(atoms),
                    },
                    "component_maps": {},
                },
                "prediction_model": {
                    "model": "unit_test_projection",
                    "prediction_delta_semantics": "signed_component_delta",
                    "prediction_error_mode": "signed_delta",
                    "uses_official_response_observations": False,
                },
                "epsilon_ladder": [-1.0, 0.0, 1.0],
                "points": [
                    {
                        "epsilon": -1.0,
                        "predicted_delta": {"posenet": -0.001, "segnet": -0.002},
                    },
                    {
                        "epsilon": 0.0,
                        "predicted_delta": {
                            "combined": 0.0,
                            "posenet": 0.0,
                            "segnet": 0.0,
                        },
                    },
                    {
                        "epsilon": 1.0,
                        "predicted_delta": {
                            "combined": 0.21,
                            "posenet": 0.001,
                            "segnet": 0.002,
                        },
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_builds_deterministic_official_plan_and_archive_variants(tmp_path: Path) -> None:
    module = _load_module()
    baseline = _write_archive(tmp_path / "baseline.zip", _baseline_members())
    atoms = [module._parse_patch_spec("renderer.bin:8:2")]
    prediction = _prediction_artifact(
        module,
        tmp_path / "predictions.json",
        baseline=baseline,
        atoms=atoms,
    )
    out_dir = tmp_path / "out"

    first = module.build_component_response_perturbation_plan(
        baseline_archive=baseline,
        output_dir=out_dir,
        atoms=atoms,
        epsilons=[-1.0, 0.0, 1.0],
        predicted_deltas_json=prediction,
        require_predicted_deltas=True,
    )
    first_plan_text = Path(first["plan"]["path"]).read_text()
    first_manifest_text = Path(first["archive_variants_manifest"]["path"]).read_text()
    first_basis_text = Path(first["basis"]["path"]).read_text()

    second = module.build_component_response_perturbation_plan(
        baseline_archive=baseline,
        output_dir=out_dir,
        atoms=atoms,
        epsilons=[-1.0, 0.0, 1.0],
        predicted_deltas_json=prediction,
        require_predicted_deltas=True,
    )

    assert Path(second["plan"]["path"]).read_text() == first_plan_text
    assert Path(second["archive_variants_manifest"]["path"]).read_text() == first_manifest_text
    assert Path(second["basis"]["path"]).read_text() == first_basis_text

    plan = json.loads(first_plan_text)
    assert plan["format"] == "official_component_response_plan_v1"
    assert plan["perturbation"]["format"] == "perturbation_basis_v1"
    assert plan["perturbation"]["auth_eval_required"] == "cuda"
    assert plan["perturbation"]["prediction_model"]["model"] == "unit_test_projection"
    assert plan["perturbation"]["prediction_model"]["prediction_error_mode"] == "signed_delta"
    assert [point["epsilon"] for point in plan["points"]] == [-1.0, 0.0, 1.0]

    by_eps = {point["epsilon"]: point for point in plan["points"]}
    neg_archive = (Path(first["plan"]["path"]).parent / by_eps[-1.0]["archive"]).resolve()
    pos_archive = (Path(first["plan"]["path"]).parent / by_eps[1.0]["archive"]).resolve()
    zero_archive = Path(by_eps[0.0]["archive"]).resolve()

    assert zero_archive == baseline.resolve()
    assert _read_member(neg_archive, "renderer.bin")[8] == 98
    assert _read_member(pos_archive, "renderer.bin")[8] == 102
    assert _read_member(pos_archive, "renderer.bin")[:4] == b"ASYM"
    assert by_eps[1.0]["predicted_delta"]["segnet"] == pytest.approx(0.002)
    assert by_eps[1.0]["changed_member_bytes"] == 1
    assert by_eps[1.0]["raw_l1_byte_delta"] == 2

    variants = json.loads(first_manifest_text)
    assert variants["format"] == "official_component_response_archive_variants_v1"
    assert variants["points"][0]["deterministic_rebuild"] is True
    assert variants["points"][0]["members"][0]["date_time"] == [1980, 1, 1, 0, 0, 0]


def test_baseline_eval_path_is_portable_when_repo_internal(tmp_path: Path) -> None:
    module = _load_module()
    module.REPO_ROOT = tmp_path
    (tmp_path / "experiments" / "results" / "lane").mkdir(parents=True)
    baseline = _write_archive(
        tmp_path / "experiments" / "results" / "lane" / "archive.zip",
        _baseline_members(),
    )
    baseline_eval = tmp_path / "experiments" / "results" / "lane" / "eval" / "contest_auth_eval.json"
    baseline_eval.parent.mkdir(parents=True)
    baseline_eval.write_text("{}\n")
    out_dir = tmp_path / "experiments" / "results" / "official_response"

    summary = module.build_component_response_perturbation_plan(
        baseline_archive=baseline,
        output_dir=out_dir,
        atoms=[module._parse_patch_spec("renderer.bin:8:2")],
        epsilons=[-1.0, 0.0, 1.0],
        baseline_contest_auth_eval_json=baseline_eval,
    )

    plan = json.loads(Path(summary["plan"]["path"]).read_text())
    assert plan["baseline_contest_auth_eval_json"] == "../lane/eval/contest_auth_eval.json"


def test_rejects_unsafe_duplicate_archive_members(tmp_path: Path) -> None:
    module = _load_module()
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", _baseline_members()["renderer.bin"])
        with pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr("renderer.bin", _baseline_members()["renderer.bin"])
        zf.writestr("masks.mkv", b"mask")
        zf.writestr("optimized_poses.pt", b"pose")

    with pytest.raises(module.ComponentResponsePerturbationError, match="duplicate archive member"):
        module.build_component_response_perturbation_plan(
            baseline_archive=archive,
            output_dir=tmp_path / "out",
            atoms=[module._parse_patch_spec("renderer.bin:8:1")],
            epsilons=[-1.0, 0.0, 1.0],
        )


def test_rejects_hidden_or_resource_fork_sidecars(tmp_path: Path) -> None:
    module = _load_module()
    archive = _write_archive(
        tmp_path / "hidden.zip",
        {
            **_baseline_members(),
            ".DS_Store": b"macos",
        },
    )

    with pytest.raises(module.ComponentResponsePerturbationError, match="hidden archive sidecar"):
        module.build_component_response_perturbation_plan(
            baseline_archive=archive,
            output_dir=tmp_path / "out",
            atoms=[module._parse_patch_spec("renderer.bin:8:1")],
            epsilons=[-1.0, 0.0, 1.0],
        )


def test_rejects_unbounded_byte_underflow(tmp_path: Path) -> None:
    module = _load_module()
    members = _baseline_members()
    renderer = bytearray(members["renderer.bin"])
    renderer[8] = 0
    members["renderer.bin"] = bytes(renderer)
    archive = _write_archive(tmp_path / "baseline.zip", members)

    with pytest.raises(module.ComponentResponsePerturbationError, match="outside byte range"):
        module.build_component_response_perturbation_plan(
            baseline_archive=archive,
            output_dir=tmp_path / "out",
            atoms=[module._parse_patch_spec("renderer.bin:8:1")],
            epsilons=[-1.0, 0.0, 1.0],
        )


def test_rejects_renderer_magic_mutation_by_default(tmp_path: Path) -> None:
    module = _load_module()
    archive = _write_archive(tmp_path / "baseline.zip", _baseline_members())

    with pytest.raises(module.ComponentResponsePerturbationError, match="renderer magic bytes"):
        module.build_component_response_perturbation_plan(
            baseline_archive=archive,
            output_dir=tmp_path / "out",
            atoms=[module._parse_patch_spec("renderer.bin:0:1")],
            epsilons=[-1.0, 0.0, 1.0],
        )


def test_rejects_non_string_basis_member(tmp_path: Path) -> None:
    module = _load_module()
    basis = tmp_path / "basis.json"
    basis.write_text(
        json.dumps(
            {
                "atoms": [
                    {
                        "member": 123,
                        "offset": 8,
                        "delta_per_epsilon": 1,
                    }
                ]
            }
        )
        + "\n"
    )

    with pytest.raises(module.ComponentResponsePerturbationError, match="member must be a string"):
        module._load_basis_atoms(basis)


def test_cli_help_imports_without_scorer_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--baseline-archive" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_require_predicted_deltas_rejects_legacy_prediction_json(tmp_path: Path) -> None:
    module = _load_module()
    baseline = _write_archive(tmp_path / "baseline.zip", _baseline_members())
    prediction = tmp_path / "legacy_predictions.json"
    prediction.write_text(json.dumps({"1": {"posenet": 0.001, "segnet": 0.002}}) + "\n")

    with pytest.raises(module.ComponentResponsePerturbationError, match="requires format"):
        module.build_component_response_perturbation_plan(
            baseline_archive=baseline,
            output_dir=tmp_path / "out",
            atoms=[module._parse_patch_spec("renderer.bin:8:1")],
            epsilons=[-1.0, 0.0, 1.0],
            predicted_deltas_json=prediction,
            require_predicted_deltas=True,
        )


def test_require_predicted_deltas_rejects_prediction_artifact_with_observed_fields(
    tmp_path: Path,
) -> None:
    module = _load_module()
    baseline = _write_archive(tmp_path / "baseline.zip", _baseline_members())
    atoms = [module._parse_patch_spec("renderer.bin:8:1")]
    prediction = _prediction_artifact(
        module,
        tmp_path / "predictions.json",
        baseline=baseline,
        atoms=atoms,
    )
    payload = json.loads(prediction.read_text())
    payload["points"][0]["observed_delta"] = 0.123
    prediction.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(module.ComponentResponsePerturbationError, match="observed-response"):
        module.build_component_response_perturbation_plan(
            baseline_archive=baseline,
            output_dir=tmp_path / "out",
            atoms=atoms,
            epsilons=[-1.0, 0.0, 1.0],
            predicted_deltas_json=prediction,
            require_predicted_deltas=True,
        )


def test_prediction_delta_builder_projects_component_maps(tmp_path: Path) -> None:
    script = REPO_ROOT / "experiments" / "build_component_response_prediction_deltas.py"
    spec = importlib.util.spec_from_file_location(
        "build_component_response_prediction_deltas",
        script,
    )
    assert spec is not None and spec.loader is not None
    pred_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pred_module
    spec.loader.exec_module(pred_module)

    baseline = _write_archive(tmp_path / "baseline.zip", _baseline_members())
    basis = tmp_path / "basis.json"
    basis.write_text(
        json.dumps(
            {
                "format": "perturbation_basis_v1",
                "atoms": [
                    {
                        "atom_id": "a0",
                        "member": "renderer.bin",
                        "offset": 8,
                        "delta_per_epsilon": 2,
                        "layer_name": "conv",
                        "channel_index": 1,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    maps: dict[str, Path] = {}
    for component, scale in {"posenet": 0.01, "segnet": 0.02, "combined": 0.03}.items():
        path = tmp_path / f"{component}.pt"
        save_sensitivity_map(
            path,
            {"conv.weight": torch.tensor([0.0, scale], dtype=torch.float32)},
            metadata={"component": component, "device": "cuda"},
        )
        maps[component] = path
    output = tmp_path / "prediction_deltas.json"

    summary = pred_module.build_component_response_prediction_deltas(
        baseline_archive=baseline,
        perturbation_basis_json=basis,
        output_json=output,
        component_maps=maps,
        epsilons=[-1.0, 0.0, 1.0],
    )

    payload = json.loads(output.read_text())
    by_eps = {point["epsilon"]: point for point in payload["points"]}
    assert payload["format"] == "official_component_response_prediction_deltas_v1"
    assert payload["prediction_model"]["uses_official_response_observations"] is False
    assert payload["prediction_model"]["prediction_error_mode"] == "absolute_magnitude"
    assert (
        payload["prediction_model"]["prediction_delta_semantics"]
        == "nonnegative_component_delta_magnitude"
    )
    assert by_eps[1.0]["predicted_delta"]["posenet"] == pytest.approx(0.04)
    assert by_eps[-1.0]["predicted_delta"]["segnet"] == pytest.approx(0.08)
    assert by_eps[0.0]["predicted_delta"]["combined"] == 0.0
    assert summary["point_count"] == 3


def test_prediction_delta_builder_rejects_uncalibrated_direct_fd_maps(tmp_path: Path) -> None:
    script = REPO_ROOT / "experiments" / "build_component_response_prediction_deltas.py"
    spec = importlib.util.spec_from_file_location(
        "build_component_response_prediction_deltas_direct_fd_guard",
        script,
    )
    assert spec is not None and spec.loader is not None
    pred_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pred_module
    spec.loader.exec_module(pred_module)

    baseline = _write_archive(tmp_path / "baseline.zip", _baseline_members())
    basis = tmp_path / "basis.json"
    basis.write_text(
        json.dumps(
            {
                "format": "perturbation_basis_v1",
                "atoms": [
                    {
                        "atom_id": "a0",
                        "member": "renderer.bin",
                        "offset": 8,
                        "delta_per_epsilon": 1,
                        "metadata": {"layer_name": "conv", "channel_index": 0},
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    maps: dict[str, Path] = {}
    for component in ("posenet", "segnet", "combined"):
        path = tmp_path / f"{component}.pt"
        save_sensitivity_map(
            path,
            {"conv.weight": torch.tensor([1.0], dtype=torch.float32)},
            metadata={
                "component": component,
                "device": "cuda",
                "sensitivity_source": "direct_renderer_cuda_finite_difference_component_response",
            },
        )
        maps[component] = path

    with pytest.raises(pred_module.ComponentResponsePredictionError, match="archive-byte"):
        pred_module.build_component_response_prediction_deltas(
            baseline_archive=baseline,
            perturbation_basis_json=basis,
            output_json=tmp_path / "prediction_deltas.json",
            component_maps=maps,
            epsilons=[-1.0, 0.0, 1.0],
        )


def test_plan_from_sensitivity_artifacts_builds_prediction_and_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = REPO_ROOT / "experiments" / "build_component_response_plan_from_sensitivity_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "build_component_response_plan_from_sensitivity_artifacts",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    baseline = _write_archive(tmp_path / "baseline.zip", _baseline_members())
    baseline_meta = module._file_meta(baseline)
    baseline_eval = tmp_path / "contest_auth_eval.json"
    baseline_eval.write_text('{"device":"cuda"}\n')
    artifact_dir = tmp_path / "sensitivity"
    artifact_dir.mkdir()
    (artifact_dir / "perturbation_basis_v1.json").write_text(
        json.dumps(
            {
                "format": "perturbation_basis_v1",
                "atoms": [
                    {
                        "atom_id": "a0",
                        "member": "renderer.bin",
                        "offset": 8,
                        "delta_per_epsilon": 1,
                        "layer_name": "conv",
                        "channel_index": 0,
                    }
                ],
                "source_archive": {
                    "bytes": baseline_meta["bytes"],
                    "sha256": baseline_meta["sha256"],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    for component, scale in {"posenet": 0.01, "segnet": 0.02, "combined": 0.03}.items():
        save_sensitivity_map(
            artifact_dir / f"{component}_sensitivity_map.pt",
            {"conv.weight": torch.tensor([scale], dtype=torch.float32)},
            metadata={"component": component, "device": "cuda"},
        )

    seen_validation_args: dict[str, object] = {}

    def _fake_validate(artifact_root, **kwargs):
        seen_validation_args["artifact_root"] = Path(artifact_root)
        seen_validation_args.update(kwargs)
        return {
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "baseline_archive_sha256": baseline_meta["sha256"],
            "baseline_archive_size_bytes": baseline_meta["bytes"],
            "promotion_eligible": False,
            "score_claim": False,
            "score_source": "none:diagnostic_component_sensitivity_non_promotable",
        }

    monkeypatch.setattr(module, "validate_local_component_sensitivity_artifact_dir", _fake_validate)

    summary = module.build_component_response_plan_from_sensitivity_artifacts(
        sensitivity_artifact_dir=artifact_dir,
        baseline_archive=baseline,
        baseline_contest_auth_eval_json=baseline_eval,
        output_dir=tmp_path / "out",
        epsilons=[-1.0, 0.0, 1.0],
    )

    assert seen_validation_args["artifact_root"] == artifact_dir.resolve()
    assert seen_validation_args["expected_baseline_archive_sha256"] == baseline_meta["sha256"]
    assert seen_validation_args["expected_baseline_archive_size_bytes"] == baseline_meta["bytes"]
    assert summary["format"] == "official_component_response_plan_from_sensitivity_artifacts_summary_v1"
    assert summary["promotion_eligible"] is False
    assert summary["score_claim"] is False
    assert summary["perturbation_basis"]["basis_source"] == "sensitivity_artifact_dir"
    prediction_path = Path(summary["prediction_deltas"]["path"])
    plan_path = Path(summary["official_response_plan"]["path"])
    assert prediction_path.is_file()
    assert plan_path.is_file()
    prediction = json.loads(prediction_path.read_text())
    plan = json.loads(plan_path.read_text())
    assert prediction["prediction_model"]["prediction_error_mode"] == "absolute_magnitude"
    assert plan["perturbation"]["prediction_model"]["prediction_error_mode"] == "absolute_magnitude"
    assert plan["points"][0]["predicted_delta"] is not None


def test_plan_from_sensitivity_artifacts_accepts_fresh_basis_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = REPO_ROOT / "experiments" / "build_component_response_plan_from_sensitivity_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "build_component_response_plan_from_sensitivity_artifacts_fresh_basis",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    baseline = _write_archive(tmp_path / "baseline.zip", _baseline_members())
    baseline_meta = module._file_meta(baseline)
    baseline_eval = tmp_path / "contest_auth_eval.json"
    baseline_eval.write_text('{"device":"cuda"}\n')
    artifact_dir = tmp_path / "sensitivity"
    artifact_dir.mkdir()
    stale_basis = artifact_dir / "perturbation_basis_v1.json"
    stale_basis.write_text(
        json.dumps(
            {
                "format": "perturbation_basis_v1",
                "atoms": [
                    {
                        "atom_id": "stale",
                        "member": "renderer.bin",
                        "offset": 9,
                        "delta_per_epsilon": 1,
                        "layer_name": "conv",
                        "channel_index": 0,
                    }
                ],
                "source_archive": {
                    "bytes": baseline_meta["bytes"],
                    "sha256": baseline_meta["sha256"],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    fresh_basis = tmp_path / "fresh_basis.json"
    fresh_basis.write_text(
        json.dumps(
            {
                "format": "perturbation_basis_v1",
                "atoms": [
                    {
                        "atom_id": "fresh",
                        "member": "renderer.bin",
                        "offset": 8,
                        "delta_per_epsilon": 1,
                        "layer_name": "conv",
                        "channel_index": 0,
                    }
                ],
                "source_archive": {
                    "bytes": baseline_meta["bytes"],
                    "sha256": baseline_meta["sha256"],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    for component in ("posenet", "segnet", "combined"):
        save_sensitivity_map(
            artifact_dir / f"{component}_sensitivity_map.pt",
            {"conv.weight": torch.tensor([0.01], dtype=torch.float32)},
            metadata={"component": component, "device": "cuda"},
        )

    def _fake_validate(*_args, **_kwargs):
        return {
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "baseline_archive_sha256": baseline_meta["sha256"],
            "baseline_archive_size_bytes": baseline_meta["bytes"],
            "promotion_eligible": False,
            "score_claim": False,
            "score_source": "none:diagnostic_component_sensitivity_non_promotable",
        }

    monkeypatch.setattr(module, "validate_local_component_sensitivity_artifact_dir", _fake_validate)

    summary = module.build_component_response_plan_from_sensitivity_artifacts(
        sensitivity_artifact_dir=artifact_dir,
        baseline_archive=baseline,
        baseline_contest_auth_eval_json=baseline_eval,
        perturbation_basis_json=fresh_basis,
        output_dir=tmp_path / "out",
        epsilons=[-1.0, 0.0, 1.0],
    )

    plan_path = Path(summary["official_response_plan"]["path"])
    plan = json.loads(plan_path.read_text())
    basis_out = json.loads((plan_path.parent / plan["perturbation"]["basis_path"]).read_text())
    assert summary["perturbation_basis"]["basis_source"] == "explicit_perturbation_basis_json"
    assert basis_out["atoms"][0]["atom_id"] == "fresh"


def test_plan_from_sensitivity_artifacts_accepts_complete_merged_shard_artifact(
    tmp_path: Path,
) -> None:
    script = REPO_ROOT / "experiments" / "build_component_response_plan_from_sensitivity_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "build_component_response_plan_from_sensitivity_artifacts_merged",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    baseline = _write_archive(tmp_path / "baseline.zip", _baseline_members())
    baseline_meta = module._file_meta(baseline)
    baseline_eval = tmp_path / "contest_auth_eval.json"
    baseline_eval.write_text('{"device":"cuda"}\n')
    basis = tmp_path / "basis.json"
    basis.write_text(
        json.dumps(
            {
                "format": "perturbation_basis_v1",
                "atoms": [
                    {
                        "atom_id": "merged_fresh",
                        "member": "renderer.bin",
                        "offset": 8,
                        "delta_per_epsilon": 1,
                        "layer_name": "conv",
                        "channel_index": 0,
                    }
                ],
                "source_archive": {
                    "bytes": baseline_meta["bytes"],
                    "sha256": baseline_meta["sha256"],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )

    source_dirs: list[Path] = []
    for shard_index in range(2):
        source_dir = tmp_path / f"source_shard_{shard_index}"
        source_dir.mkdir()
        source_dirs.append(source_dir)
        (source_dir / "diagnostic_component_sensitivity_inputs.json").write_text(
            json.dumps(
                {
                    "baseline_archive": {
                        "bytes": baseline_meta["bytes"],
                        "sha256": baseline_meta["sha256"],
                    },
                    "score_claim": False,
                    "promotion_eligible": False,
                },
                sort_keys=True,
            )
            + "\n"
        )
        (source_dir / "component_sensitivity_profile_summary.json").write_text(
            json.dumps(
                {
                    "device": "cuda",
                    "sensitivity_source": "direct_renderer_cuda_finite_difference_component_response",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "official_component_response": False,
                    "canonical_scorer_path": False,
                    "n_pairs_total": 600,
                    "finite_difference_shard": {
                        "is_shard": True,
                        "shard_index": shard_index,
                        "shard_count": 2,
                        "assigned_channel_count": 1,
                        "assigned_channel_sha256": f"sha{shard_index}",
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )

    artifact_dir = tmp_path / "merged_artifact"
    artifact_dir.mkdir()
    merged_shard = {
        "is_shard": False,
        "merge_required_for_certification_handoff": False,
        "merged_from_shards": True,
    }
    merge = {
        "schema": "component_sensitivity_direct_fd_merge_v1",
        "coverage": "exactly_once",
        "declared_shard_count": 2,
        "source_shard_dirs": [str(path) for path in source_dirs],
        "source_shard_count": 2,
        "source_shard_indices": [0, 1],
        "all_channel_count": 2,
        "covered_channel_count": 2,
        "missing_channel_count": 0,
        "missing_shard_indices": [],
        "certification_handoff_eligible": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    (artifact_dir / "component_sensitivity_profile_summary.json").write_text(
        json.dumps(
            {
                "merge_tool": "experiments/merge_component_sensitivity_shards.py",
                "device": "cuda",
                "sensitivity_source": "direct_renderer_cuda_finite_difference_component_response",
                "score_claim": False,
                "promotion_eligible": False,
                "official_component_response": False,
                "canonical_scorer_path": False,
                "n_pairs_total": 600,
                "finite_difference_shard": merged_shard,
                "finite_difference_merge": merge,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (artifact_dir / "component_sensitivity_shard_merge_validation.json").write_text(
        json.dumps(
            {
                "format": "component_sensitivity_shard_merge_validation_v1",
                "coverage": "exactly_once",
                "all_channel_count": 2,
                "covered_channel_count": 2,
                "missing_channel_count": 0,
                "missing_shard_indices": [],
                "certification_handoff_eligible": True,
                "score_claim": False,
                "promotion_eligible": False,
                "finite_difference_merge": merge,
                "source_shards": [
                    {
                        "path": str(source_dirs[0]),
                        "shard_index": 0,
                        "assigned_channel_count": 1,
                        "assigned_channel_sha256": "sha0",
                    },
                    {
                        "path": str(source_dirs[1]),
                        "shard_index": 1,
                        "assigned_channel_count": 1,
                        "assigned_channel_sha256": "sha1",
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    for component in ("posenet", "segnet", "combined"):
        metadata = {
            "component": component,
            "scorer_target": component,
            "device": "cuda",
            "score_claim": False,
            "promotion_eligible": False,
            "official_component_response": False,
            "canonical_scorer_path": False,
            "sensitivity_source": "direct_renderer_cuda_finite_difference_component_response",
            "finite_difference_shard": merged_shard,
        }
        save_sensitivity_map(
            artifact_dir / f"{component}_sensitivity_map.pt",
            {"conv.weight": torch.tensor([0.01], dtype=torch.float32)},
            metadata=metadata,
        )
        save_sensitivity_map(
            artifact_dir / f"{component}_holdout_sensitivity_map.pt",
            {"conv.weight": torch.tensor([0.01], dtype=torch.float32)},
            metadata={**metadata, "split": "holdout"},
        )

    summary = module.build_component_response_plan_from_sensitivity_artifacts(
        sensitivity_artifact_dir=artifact_dir,
        baseline_archive=baseline,
        baseline_contest_auth_eval_json=baseline_eval,
        perturbation_basis_json=basis,
        output_dir=tmp_path / "out",
        epsilons=[-1.0, 0.0, 1.0],
        allow_merged_shard_artifact=True,
        response_only_no_prediction_deltas=True,
    )

    assert summary["sensitivity_artifact"]["role"] == "diagnostic_component_sensitivity_merged_shards"
    assert summary["sensitivity_artifact"]["score_claim"] is False
    assert summary["response_only_no_prediction_deltas"] is True
    assert Path(summary["official_response_plan"]["path"]).is_file()


def test_plan_from_sensitivity_artifacts_cli_help_imports() -> None:
    script = REPO_ROOT / "experiments" / "build_component_response_plan_from_sensitivity_artifacts.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--sensitivity-artifact-dir" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
