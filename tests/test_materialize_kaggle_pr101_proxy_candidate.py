from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "materialize_kaggle_pr101_proxy_candidate.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "materialize_kaggle_pr101_proxy_candidate_under_test",
        TOOL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate() -> dict[str, object]:
    return {
        "candidate_id": "proxy_cmaes_0037",
        "evidence_semantics": "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval",
        "optimizer": "cmaes",
        "optimizer_status": "cmaes_style_stdlib",
        "params": {
            "bias_b": -0.7928396476287223,
            "bias_g": -0.8821541213924035,
            "bias_r": -1.0058836754765783,
            "delta_scale": 0.009816389307286146,
            "latent_delta_scale": 0.009748367009466553,
            "smooth_weight": 0.019311518140752215,
        },
        "proxy_components": {
            "anchor_proximity": 0.08050894756328586,
            "bias_asymmetry": 0.007630426175465566,
            "deterministic_jitter": 2.2098532737274997e-06,
            "latent_delta_mismatch": 6.802229781959282e-05,
            "smooth_penalty": 0.0,
        },
        "proxy_objective": 0.19287550335547282,
        "proxy_only": True,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "score_claim_valid": False,
        "trial_index": 37,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materializes_proxy_candidate_handoff_and_manifest(tmp_path: Path) -> None:
    tool = load_tool()
    candidate_path = tmp_path / "run" / "best_proxy_candidate.json"
    sweep_manifest = tmp_path / "run" / "proxy_sweep_manifest.json"
    output_dir = tmp_path / "out"
    _write_json(candidate_path, _candidate())
    _write_json(sweep_manifest, {"schema": "pr101_kaggle_proxy_sweep_v1"})

    manifest = tool.materialize_candidate(
        candidate_path=candidate_path,
        output_dir=output_dir,
    )

    handoff = json.loads((output_dir / "archive_builder_handoff.json").read_text())
    manifest_on_disk = json.loads((output_dir / "materialization_manifest.json").read_text())

    assert manifest == manifest_on_disk
    assert manifest["schema"] == "pr101_kaggle_proxy_candidate_materialization_v1"
    assert manifest["candidate_id"] == "proxy_cmaes_0037"
    assert manifest["candidate_params"] == _candidate()["params"]
    assert manifest["score_claim"] is False
    assert manifest["score_claim_valid"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["archive_zip_emitted"] is False
    assert manifest["inflate_runtime_emitted"] is False
    assert "no_archive_zip_emitted" in manifest["dispatch_blockers"]
    assert len(manifest["input_files"]) == 2
    assert len(manifest["output_files"]) == 1
    assert manifest["output_files"][0]["path"].endswith("archive_builder_handoff.json")
    assert len(manifest["output_files"][0]["sha256"]) == 64
    assert len(manifest["manifest_sha256_excluding_self"]) == 64

    assert handoff["schema"] == "pr101_kaggle_proxy_candidate_archive_builder_handoff_v1"
    assert handoff["param_schema"] == "pr101_kaggle_proxy_candidate_params_v1"
    assert handoff["evidence_boundary"]["score_claim"] is False
    assert handoff["evidence_boundary"]["ready_for_exact_eval_dispatch"] is False
    contract = handoff["archive_builder_handoff_contract"]
    assert contract["status"] == "pending_real_archive_builder"
    assert contract["builder_must_consume"]["param_keys"] == list(tool.PARAM_KEYS)
    assert "byte_closed_archive_zip" in contract["builder_must_emit_before_dispatch"]
    assert "claim_score_from_this_proxy_artifact" in contract["builder_must_not"]


def test_materialization_is_deterministic_for_same_input(tmp_path: Path) -> None:
    tool = load_tool()
    candidate_path = tmp_path / "best_proxy_candidate.json"
    output_dir = tmp_path / "out"
    _write_json(candidate_path, _candidate())

    first = tool.materialize_candidate(candidate_path=candidate_path, output_dir=output_dir)
    first_handoff = (output_dir / "archive_builder_handoff.json").read_text()
    second = tool.materialize_candidate(
        candidate_path=candidate_path,
        output_dir=output_dir,
        force=True,
    )

    assert first["manifest_sha256_excluding_self"] == second["manifest_sha256_excluding_self"]
    assert first["output_files"][0]["sha256"] == second["output_files"][0]["sha256"]
    assert first_handoff == (output_dir / "archive_builder_handoff.json").read_text()


def test_refuses_candidate_that_claims_dispatch_readiness(tmp_path: Path) -> None:
    tool = load_tool()
    payload = _candidate()
    payload["ready_for_exact_eval_dispatch"] = True
    candidate_path = tmp_path / "best_proxy_candidate.json"
    _write_json(candidate_path, payload)

    with pytest.raises(tool.MaterializationError, match="ready_for_exact_eval_dispatch"):
        tool.materialize_candidate(candidate_path=candidate_path, output_dir=tmp_path / "out")


def test_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    tool = load_tool()
    candidate_path = tmp_path / "best_proxy_candidate.json"
    output_dir = tmp_path / "out"
    _write_json(candidate_path, _candidate())
    (output_dir / "existing.txt").parent.mkdir(parents=True)
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(tool.MaterializationError, match="not empty"):
        tool.materialize_candidate(candidate_path=candidate_path, output_dir=output_dir)


def test_force_refuses_unexpected_output_contents(tmp_path: Path) -> None:
    tool = load_tool()
    candidate_path = tmp_path / "best_proxy_candidate.json"
    output_dir = tmp_path / "out"
    _write_json(candidate_path, _candidate())
    (output_dir / "operator_note.txt").parent.mkdir(parents=True)
    (output_dir / "operator_note.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(tool.MaterializationError, match="unexpected files"):
        tool.materialize_candidate(
            candidate_path=candidate_path,
            output_dir=output_dir,
            force=True,
        )

    assert (output_dir / "operator_note.txt").read_text(encoding="utf-8") == "preserve"


def test_cli_writes_handoff_without_dispatching(tmp_path: Path) -> None:
    candidate_path = tmp_path / "best_proxy_candidate.json"
    output_dir = tmp_path / "out"
    _write_json(candidate_path, _candidate())

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--candidate",
            str(candidate_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(proc.stdout)
    assert stdout["candidate_id"] == "proxy_cmaes_0037"
    assert stdout["proxy_only"] is True
    assert stdout["score_claim"] is False
    assert stdout["ready_for_exact_eval_dispatch"] is False
    assert (output_dir / "archive_builder_handoff.json").is_file()
    assert (output_dir / "materialization_manifest.json").is_file()
