from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "certify_component_sensitivity_maps.py"
COMPONENTS = ("posenet", "segnet", "combined")


def _load_module():
    spec = importlib.util.spec_from_file_location("certify_component_sensitivity_maps", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _sample_plan() -> dict[str, object]:
    return {
        "calibration_pairs": [
            {"video": 0, "pair_index": i, "t": 2 * i, "t1": 2 * i + 1}
            for i in range(480)
        ],
        "holdout_pairs": [
            {"video": 0, "pair_index": i, "t": 2 * i, "t1": 2 * i + 1}
            for i in range(480, 600)
        ],
        "split_seed": 123,
        "split_hash": "a" * 64,
    }


def _response_curve(
    component: str,
    *,
    prediction_deltas_sha256: str,
    perturbation_basis_sha256: str,
    external_baseline_repro: bool | None = True,
) -> dict[str, object]:
    readouts = {
        "posenet": "official_pose_mse",
        "segnet": "official_argmax_disagreement",
        "combined": "official_component_formula",
    }
    return {
        "schema_version": 1,
        "format": "official_component_response_curves_v1",
        "tool": "experiments/profile_component_sensitivity_official.py",
        "component": component,
        "device": "cuda",
        "promotion_eligible": True,
        "evidence_grade": "A",
        "official_component_response": True,
        "canonical_scorer_path": True,
        "component_response_path": "archive_zip_inflate_sh_upstream_evaluate_py",
        "passed": True,
        "component_readout": readouts[component],
        "response_kind": "symmetric",
        "epsilon_ladder": [-0.001, 0.0, 0.001],
        "count": 3,
        "holdout_error": 0.02,
        "promotion_blockers": [],
        "gate_results": {
            "finite_values": True,
            "coverage_passed": True,
            "zero_repro": True,
            "zero_repro_error": 0.0,
            "external_baseline_repro": external_baseline_repro,
            "external_baseline_repro_error": 0.0 if external_baseline_repro else 1.0,
            "signal_present": True,
            "observed_delta_max": 0.01,
            "prediction_error_passed": True,
            "max_relative_prediction_error": 0.02,
            "promotion_gate_passed": True,
        },
        "perturbation": {
            "basis_sha256": perturbation_basis_sha256,
            "predicted_deltas_source": {
                "path": "../prediction_deltas.json",
                "bytes": 123,
                "sha256": prediction_deltas_sha256,
            },
        },
        "points": [
            {"epsilon": -0.001, "delta": -0.01},
            {"epsilon": 0.0, "delta": 0.0},
            {"epsilon": 0.001, "delta": 0.01},
        ],
    }


def _write_fixture(root: Path) -> dict[str, Path]:
    candidate = root / "candidate"
    response = root / "response"
    candidate.mkdir()
    response.mkdir()
    archive = root / "archive.zip"
    archive.write_bytes(b"archive")
    prediction = _write_json(
        root / "prediction_deltas.json",
        {
            "schema_version": 1,
            "format": "official_component_response_prediction_deltas_v1",
            "epsilon_ladder": [-0.001, 0.0, 0.001],
            "atom_set_sha256": "a" * 64,
            "points": [
                {
                    "epsilon": -0.001,
                    "atom_contributions": [
                        {
                            "atom_id": "atom0",
                            "integer_byte_delta": -1,
                            "component_abs_delta": {
                                "posenet": 0.01,
                                "segnet": 0.01,
                                "combined": 0.01,
                            },
                        }
                    ],
                }
            ],
        },
    )
    basis = _write_json(
        root / "perturbation_basis_v1.json",
        {
            "schema_version": 1,
            "format": "perturbation_basis_v1",
            "basis_kind": "archive_byte_additive",
            "basis_id": "basis0",
            "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
            "epsilon_ladder": [-0.001, 0.0, 0.001],
            "source_archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": _sha(archive),
            },
            "atoms": [
                {
                    "atom_id": "atom0",
                    "member": "renderer.bin",
                    "offset": 0,
                    "delta_per_epsilon": 1,
                    "metadata": {"layer_name": "renderer.test", "channel_index": 0},
                }
            ],
        },
    )
    _write_json(candidate / "sample_plan.json", _sample_plan())
    _write_json(
        candidate / "stability.json",
        {
            "passed": True,
            "thresholds": {
                "cv_max": 0.35,
                "spearman_min": 0.3,
                "top_decile_overlap_min": 0.5,
            },
            "cv": {"posenet": 0.01, "segnet": 0.02, "combined": 0.03},
            "rank": {"posenet": 0.91, "segnet": 0.92, "combined": 0.93},
            "top_k": {
                "posenet": {"overlap": 0.81},
                "segnet": {"overlap": 0.82},
                "combined": {"overlap": 0.83},
            },
        },
    )
    for component in COMPONENTS:
        torch.save(
            {
                "format": "tac_score_sensitivity_map_v1",
                "sensitivities": {f"{component}.weight": torch.ones(4)},
                "metadata": {
                    "component": component,
                    "scorer_target": component,
                    "device": "cuda",
                    "sensitivity_source": "direct_renderer_cuda_finite_difference_component_response",
                },
            },
            candidate / f"{component}_sensitivity_map.pt",
        )
        _write_json(
            response / f"{component}_official_response_curve.json",
            _response_curve(
                component,
                prediction_deltas_sha256=_sha(prediction),
                perturbation_basis_sha256=_sha(basis),
            ),
        )
    _write_json(
        response / "official_component_response_summary.json",
        {
            "format": "official_component_response_summary_v1",
            "promotion_eligible": True,
            "device": "cuda",
            "external_baseline_contest_auth_eval_json": {
                "path": str(root / "contest_auth_eval.json"),
                "bytes": 123,
                "sha256": "b" * 64,
            },
        },
    )
    pose = 0.01
    seg = 0.003
    score = 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive.stat().st_size / 37_545_489
    eval_json = root / "contest_auth_eval.json"
    _write_json(
        eval_json,
        {
            "n_samples": 600,
            "avg_posenet_dist": pose,
            "avg_segnet_dist": seg,
            "archive_size_bytes": archive.stat().st_size,
            "score_recomputed_from_components": score,
            "provenance": {
                "device": "cuda",
                "archive_sha256": _sha(archive),
            },
        },
    )
    review = _write_json(root / "review.json", {"clean_passes": 3, "unresolved_blockers": []})
    return {
        "candidate": candidate,
        "response": response,
        "archive": archive,
        "eval": eval_json,
        "review": review,
        "prediction": prediction,
        "basis": basis,
        "output": root / "certified",
    }


def _argv(paths: dict[str, Path]) -> list[str]:
    return [
        "--candidate-artifact-dir",
        str(paths["candidate"]),
        "--official-response-artifact-dir",
        str(paths["response"]),
        "--baseline-archive",
        str(paths["archive"]),
        "--baseline-contest-auth-eval-json",
        str(paths["eval"]),
        "--review-packet-json",
        str(paths["review"]),
        "--prediction-deltas-json",
        str(paths["prediction"]),
        "--perturbation-basis-json",
        str(paths["basis"]),
        "--output-dir",
        str(paths["output"]),
    ]


def test_certifier_writes_certified_maps_without_mutating_sources(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_fixture(tmp_path)
    before = {
        component: _sha(paths["candidate"] / f"{component}_sensitivity_map.pt")
        for component in COMPONENTS
    }

    summary = module.certify_maps(module.parse_args(_argv(paths)))

    assert summary["promotion_eligible"] is True
    assert summary["score_claim"] is False
    for component in COMPONENTS:
        assert _sha(paths["candidate"] / f"{component}_sensitivity_map.pt") == before[component]
        out = paths["output"] / f"{component}_certified_sensitivity_map.pt"
        payload = torch.load(out, map_location="cpu", weights_only=False)
        assert payload["metadata"]["promotion_eligible"] is True
        cert = payload["metadata"]["certification"]
        assert cert["format"] == "component_sensitivity_map_certification_v1"
        assert cert["component"] == component
        assert cert["review_clean_passes"] == 3
        assert cert["prediction_deltas_sha256"] == _sha(paths["prediction"])
        assert cert["perturbation_basis_sha256"] == _sha(paths["basis"])
    assert summary["prediction_deltas"]["sha256"] == _sha(paths["prediction"])
    assert summary["perturbation_basis"]["sha256"] == _sha(paths["basis"])


def test_certifier_rejects_fisher_proxy_source_maps(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_fixture(tmp_path)
    payload = torch.load(
        paths["candidate"] / "combined_sensitivity_map.pt",
        map_location="cpu",
        weights_only=False,
    )
    payload["metadata"]["sensitivity_source"] = "fisher_proxy"
    torch.save(payload, paths["candidate"] / "combined_sensitivity_map.pt")

    with pytest.raises(module.CertificationError, match="Fisher/proxy"):
        module.certify_maps(module.parse_args(_argv(paths)))


def test_certifier_rejects_external_baseline_repro_failure(tmp_path: Path) -> None:
    module = _load_module()
    paths = _write_fixture(tmp_path)
    curve_path = paths["response"] / "combined_official_response_curve.json"
    curve = json.loads(curve_path.read_text())
    curve["gate_results"]["external_baseline_repro"] = False
    curve["gate_results"]["external_baseline_repro_error"] = 1e-3
    curve_path.write_text(json.dumps(curve, indent=2, sort_keys=True) + "\n")

    with pytest.raises(module.CertificationError, match="external_baseline_repro"):
        module.certify_maps(module.parse_args(_argv(paths)))


def test_certifier_help_imports() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--candidate-artifact-dir" in result.stdout
