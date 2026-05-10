from __future__ import annotations

import importlib.util
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tac.component_sensitivity_artifact import validate_component_sensitivity_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "build_component_sensitivity_manifest.py"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _load_module():
    spec = importlib.util.spec_from_file_location("build_component_sensitivity_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(root: Path) -> dict[str, Path]:
    upstream = root / "upstream"
    upstream.mkdir()
    (upstream / "README").write_text("synthetic upstream scorer tree\n")
    checkpoint = root / "renderer.bin"
    checkpoint.write_bytes(b"renderer")
    video = root / "0.mkv"
    video.write_bytes(b"video")
    archive = root / "archive.zip"
    archive.write_bytes(b"archive")
    contest = root / "contest_auth_eval.json"
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    contest.write_text(
        json.dumps(
            {
                "n_samples": 600,
                "score_recomputed_from_components": 1.0,
                "archive_size_bytes": archive.stat().st_size,
                "provenance": {"device": "cuda", "archive_sha256": archive_sha},
            }
        )
        + "\n"
    )
    stability = root / "stability.json"
    stability.write_text(
        json.dumps(
            {
                "cv": {"posenet": 0.01, "segnet": 0.02, "combined": 0.03},
                "rank": {"posenet": 0.97, "segnet": 0.96, "combined": 0.95},
                "top_k": {
                    "posenet": {"k": 8, "overlap": 0.9},
                    "segnet": {"k": 8, "overlap": 0.88},
                    "combined": {"k": 8, "overlap": 0.91},
                },
                "thresholds": {
                    "cv_max": 0.35,
                    "spearman_min": 0.3,
                    "top_decile_overlap_min": 0.5,
                },
                "passed": True,
            }
        )
        + "\n"
    )
    readouts = {
        "posenet": "official_pose_mse",
        "segnet": "official_argmax_disagreement",
        "combined": "official_component_formula",
    }
    for component in ("posenet", "segnet", "combined"):
        torch.save(
            {
                "format": "tac_score_sensitivity_map_v1",
                "sensitivities": {f"{component}.weight": torch.ones(3, dtype=torch.float32)},
                "metadata": _certified_map_metadata(component),
            },
            root / f"{component}_map.pt",
        )
        (root / f"{component}_curve.json").write_text(
            json.dumps(
                {
                    "official_component_response": True,
                    "passed": True,
                    "gate_results": {
                        "finite_values": True,
                        "coverage_passed": True,
                        "zero_repro": True,
                        "zero_repro_error": 0.0,
                        "signal_present": True,
                        "observed_delta_max": 0.02,
                        "prediction_error_passed": True,
                        "max_relative_prediction_error": 0.02,
                        "promotion_gate_passed": True,
                    },
                    "gate_spec": {
                        "zero_repro_tolerance": 1e-7,
                        "holdout_error_max": 0.05,
                        "spearman_min": 0.3,
                    },
                    "promotion_blockers": [],
                    "component_readout": readouts[component],
                    "response_kind": "symmetric",
                    "points": [
                        {"epsilon": -0.1, "holdout_error": 0.02},
                        {"epsilon": 0.0, "holdout_error": 0.0},
                        {"epsilon": 0.1, "holdout_error": 0.02},
                    ],
                }
            )
            + "\n"
        )
    return {
        "_root": root,
        "checkpoint": checkpoint,
        "video": video,
        "upstream": upstream,
        "archive": archive,
        "contest": contest,
        "stability": stability,
        "posenet_map": root / "posenet_map.pt",
        "segnet_map": root / "segnet_map.pt",
        "combined_map": root / "combined_map.pt",
        "posenet_curve": root / "posenet_curve.json",
        "segnet_curve": root / "segnet_curve.json",
        "combined_curve": root / "combined_curve.json",
    }


def _certified_map_metadata(component: str) -> dict[str, object]:
    return {
        "component": component,
        "device": "cuda",
        "promotion_eligible": True,
        "official_component_response": True,
        "canonical_scorer_path": True,
        "sensitivity_source": "certified_official_component_sensitivity",
        "certification": {
            "format": "component_sensitivity_map_certification_v1",
            "component": component,
            "device": "cuda",
            "official_component_response": True,
            "canonical_scorer_path": True,
            "promotion_eligible": True,
            "source_map_sha256": SHA_A,
            "official_response_curve_sha256": SHA_B,
            "stability_sha256": SHA_C,
            "sample_plan_sha256": SHA_D,
            "baseline_archive_sha256": SHA_A,
            "baseline_archive_bytes": 686635,
            "contest_auth_eval_json_sha256": SHA_B,
            "prediction_deltas_sha256": SHA_C,
            "perturbation_basis_sha256": SHA_D,
            "review_packet_sha256": SHA_A,
            "review_clean_passes": 3,
            "review_unresolved_blockers": [],
            "response_gate_results": {
                "finite_values": True,
                "coverage_passed": True,
                "zero_repro": True,
                "zero_repro_error": 0.0,
                "signal_present": True,
                "observed_delta_max": 0.02,
                "prediction_error_passed": True,
                "max_relative_prediction_error": 0.02,
                "promotion_gate_passed": True,
            },
            "stability_gate_results": {
                "passed": True,
                "cv_max": 0.01,
                "spearman_min": 0.96,
                "top_decile_overlap_min": 0.9,
            },
        },
    }


def _path_arg(paths: dict[str, Path], key: str, *, rooted: bool) -> str:
    path = paths[key]
    if not rooted:
        return str(path)
    return path.relative_to(paths["_root"]).as_posix()


def _base_argv(paths: dict[str, Path], output: Path, *, rooted: bool = True) -> list[str]:
    argv = [
        "--checkpoint",
        _path_arg(paths, "checkpoint", rooted=rooted),
        "--video",
        _path_arg(paths, "video", rooted=rooted),
        "--upstream",
        _path_arg(paths, "upstream", rooted=rooted),
        "--archive",
        _path_arg(paths, "archive", rooted=rooted),
        "--contest-auth-eval-json",
        _path_arg(paths, "contest", rooted=rooted),
        "--posenet-map",
        _path_arg(paths, "posenet_map", rooted=rooted),
        "--segnet-map",
        _path_arg(paths, "segnet_map", rooted=rooted),
        "--combined-map",
        _path_arg(paths, "combined_map", rooted=rooted),
        "--posenet-response-curve",
        _path_arg(paths, "posenet_curve", rooted=rooted),
        "--segnet-response-curve",
        _path_arg(paths, "segnet_curve", rooted=rooted),
        "--combined-response-curve",
        _path_arg(paths, "combined_curve", rooted=rooted),
        "--stability-json",
        _path_arg(paths, "stability", rooted=rooted),
        "--split-seed",
        "123",
        "--output",
        str(output),
    ]
    if rooted:
        argv.extend(["--root", str(paths["_root"])])
    return argv


def test_build_manifest_materializes_and_validates(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    output = tmp_path / "component_sensitivity_manifest.json"

    args = module.parse_args(_base_argv(paths, output))
    manifest = module.build_manifest(args)

    validate_component_sensitivity_manifest(manifest)
    assert manifest["format"] == "component_sensitivity_v1"
    assert manifest["device"] == "cuda"
    assert manifest["promotion_eligible"] is True
    assert manifest["component_maps"]["posenet"]["scorer_target"] == "posenet"
    assert manifest["component_maps"]["posenet"]["tensor"]["shape"] == [3]
    assert manifest["response_curves"]["segnet"]["count"] == 3
    assert manifest["response_curves"]["segnet"]["official_component_response"] is True
    assert manifest["response_curves"]["segnet"]["component_readout"] == "official_argmax_disagreement"
    assert manifest["response_curves"]["segnet"]["gate_results"]["promotion_gate_passed"] is True
    assert manifest["response_curves"]["segnet"]["promotion_blockers"] == []
    assert len(manifest["sample_plan"]["calibration_pairs"]) == 480
    assert len(manifest["sample_plan"]["holdout_pairs"]) == 120
    assert manifest["contest_eval"]["archive"]["bytes"] == len(b"archive")


def test_cli_writes_deterministic_manifest(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output = tmp_path / "manifest.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), *_base_argv(paths, output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["promotion_eligible"] is True
    first = output.read_text()
    subprocess.run(
        [sys.executable, str(SCRIPT), *_base_argv(paths, output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert output.read_text() == first


def test_absolute_tmp_evidence_paths_still_rejected(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)

    args = module.parse_args(
        _base_argv(paths, tmp_path / "manifest.json", rooted=False)
    )
    with pytest.raises(
        module.ComponentSensitivityArtifactError,
        match="refusing transient /tmp-style evidence path",
    ):
        module.build_manifest(args)


def test_rejects_non_cuda_contest_eval(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    paths["contest"].write_text(
        json.dumps({"n_samples": 600, "provenance": {"device": "cpu"}}) + "\n"
    )

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    try:
        module.build_manifest(args)
    except module.ComponentSensitivityManifestBuildError as exc:
        assert "provenance.device" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-CUDA contest eval rejection")


def test_rejects_contest_eval_archive_sha_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    paths["contest"].write_text(
        json.dumps(
            {
                "n_samples": 600,
                "archive_size_bytes": paths["archive"].stat().st_size,
                "provenance": {"device": "cuda", "archive_sha256": "a" * 64},
            }
        )
        + "\n"
    )

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    with pytest.raises(
        module.ComponentSensitivityManifestBuildError,
        match="archive_sha256",
    ):
        module.build_manifest(args)


def test_rejects_contest_eval_archive_size_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    archive_sha = hashlib.sha256(paths["archive"].read_bytes()).hexdigest()
    paths["contest"].write_text(
        json.dumps(
            {
                "n_samples": 600,
                "archive_size_bytes": paths["archive"].stat().st_size + 1,
                "provenance": {"device": "cuda", "archive_sha256": archive_sha},
            }
        )
        + "\n"
    )

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    with pytest.raises(
        module.ComponentSensitivityManifestBuildError,
        match="archive_size_bytes",
    ):
        module.build_manifest(args)


def test_rejects_nonfinite_sensitivity_tensor(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    bad = torch.ones(3, dtype=torch.float32)
    bad[1] = float("nan")
    torch.save(
        {
            "format": "tac_score_sensitivity_map_v1",
            "sensitivities": {"posenet.weight": bad},
            "metadata": _certified_map_metadata("posenet"),
        },
        paths["posenet_map"],
    )

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    with pytest.raises(
        module.ComponentSensitivityManifestBuildError,
        match="NaN/Inf",
    ):
        module.build_manifest(args)


def test_rejects_clean_but_uncertified_sensitivity_map(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    torch.save(
        {
            "format": "tac_score_sensitivity_map_v1",
            "sensitivities": {"posenet.weight": torch.ones(3, dtype=torch.float32)},
            "metadata": {
                "component": "posenet",
                "device": "cuda",
                "promotion_eligible": True,
                "official_component_response": True,
                "canonical_scorer_path": True,
            },
        },
        paths["posenet_map"],
    )

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    with pytest.raises(
        module.ComponentSensitivityManifestBuildError,
        match="certification",
    ):
        module.build_manifest(args)


def test_rejects_sample_plan_split_hash_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    sample_plan = tmp_path / "sample_plan.json"
    sample_plan.write_text(
        json.dumps(
            {
                "calibration_pairs": [{"video": 0, "pair_index": 1, "t": 2, "t1": 3}],
                "holdout_pairs": [{"video": 0, "pair_index": 2, "t": 4, "t1": 5}],
                "split_seed": 123,
                "split_hash": "a" * 64,
            }
        )
        + "\n"
    )
    argv = [
        *_base_argv(paths, tmp_path / "manifest.json"),
        "--sample-plan-json",
        str(sample_plan),
    ]

    args = module.parse_args(argv)
    with pytest.raises(
        module.ComponentSensitivityManifestBuildError,
        match="sample_plan.split_hash",
    ):
        module.build_manifest(args)


def test_rejects_diagnostic_profile_sensitivity_map(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    torch.save(
        {
            "format": "sensitivity_map_v1",
            "sensitivities": {"posenet.weight": torch.ones(3, dtype=torch.float32)},
            "metadata": {
                "tool": "experiments/profile_component_sensitivity.py",
                "promotion_eligible": False,
                "evidence_grade": "diagnostic_cuda_fisher_proxy",
                "official_component_response": False,
                "promotion_blockers": [
                    {
                        "code": "fisher_proxy_not_official_component_response",
                        "mathematical_explanation": "diagnostic Fisher proxy",
                    }
                ],
            },
        },
        paths["posenet_map"],
    )

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    with pytest.raises(
        module.ComponentSensitivityManifestBuildError,
        match="diagnostic/non-promotable",
    ):
        module.build_manifest(args)


def test_rejects_diagnostic_profile_response_curve(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    paths["segnet_curve"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "component": "segnet",
                "device": "cuda",
                "promotion_eligible": False,
                "evidence_grade": "diagnostic_cuda_fisher_proxy",
                "official_component_response": False,
                "promotion_blockers": [
                    {
                        "code": "fisher_proxy_not_official_component_response",
                        "mathematical_explanation": "diagnostic Fisher proxy",
                    }
                ],
                "points": [{"epsilon": 0.1, "holdout_error": 0.02}],
            }
        )
        + "\n"
    )

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    with pytest.raises(
        module.ComponentSensitivityManifestBuildError,
        match="diagnostic/non-promotable",
    ):
        module.build_manifest(args)


def test_response_curve_requires_holdout_error(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    paths["posenet_curve"].write_text(json.dumps({"points": [{"epsilon": 0.1}]}) + "\n")

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    try:
        module.build_manifest(args)
    except module.ComponentSensitivityManifestBuildError as exc:
        assert "holdout_error" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected response curve rejection")


def test_rejects_response_curve_without_official_gate(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_inputs(tmp_path)
    paths["segnet_curve"].write_text(
        json.dumps(
            {
                "points": [
                    {"epsilon": -0.1, "holdout_error": 0.02},
                    {"epsilon": 0.0, "holdout_error": 0.0},
                    {"epsilon": 0.1, "holdout_error": 0.02},
                ],
                "holdout_error": 0.02,
            }
        )
        + "\n"
    )

    args = module.parse_args(_base_argv(paths, tmp_path / "manifest.json"))
    try:
        module.build_manifest(args)
    except Exception as exc:
        assert "official_component_response" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing official response gate rejection")
