from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = REPO_ROOT / "tools" / "recover_modal_auth_eval.py"
    spec = importlib.util.spec_from_file_location("recover_modal_auth_eval_tool_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_auth_eval(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cuda",
                "device": "cuda",
                "canonical_score": 0.20664588545741508,
                "avg_segnet_dist": 0.00064260,
                "avg_posenet_dist": 0.00003236,
                "archive_size_bytes": 186822,
                "n_samples": 600,
                "evidence_grade": "contest-CUDA",
                "lane_tag": "[contest-CUDA]",
                "score_claim_valid": True,
                "promotion_eligible": True,
                "provenance": {
                    "archive_sha256": "a" * 64,
                    "archive_size_bytes": 186822,
                    "device": "cuda",
                    "hardware": "Modal Tesla T4 Linux x86_64",
                    "platform_system": "Linux",
                    "platform_machine": "x86_64",
                    "gpu_model": "Tesla T4",
                    "gpu_t4_match": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_terminal_status_uses_exact_readiness_cuda_prefix() -> None:
    tool = _load_tool()

    status = tool._terminal_status(
        {"status": "recovered", "passed": True, "score_claim": True},
        {"axis": "contest_cuda"},
    )

    assert status == "completed_contest_cuda_modal_auth_eval_recovered"


def test_terminal_notes_include_exact_custody_fields(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_auth_eval(tmp_path / "contest_auth_eval.json")

    notes = tool._terminal_notes(
        {
            "output_dir": str(tmp_path),
            "passed": True,
            "result_json": str(tmp_path / "modal_cuda_auth_eval_result.json"),
        },
        {"lane_id": "lane_pr106_r2"},
        "posterior_update=accepted",
    )

    assert "archive_sha=" + "a" * 64 in notes
    assert "archive_bytes=186822" in notes
    assert "score_recomputed=0.20664588545741508" in notes
    assert "hardware_substrate=linux_x86_64_t4" in notes
    assert "posterior_update=accepted" in notes


def test_auth_eval_artifact_path_accepts_adjudicated_fallback(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_auth_eval(tmp_path / "contest_auth_eval.adjudicated.json")

    artifact = tool._auth_eval_artifact_path(
        {
            "output_dir": str(tmp_path),
            "result_json": str(tmp_path / "modal_cuda_auth_eval_result.json"),
        }
    )

    assert artifact == tmp_path / "contest_auth_eval.adjudicated.json"


def test_maybe_update_posterior_routes_auth_eval_artifact(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    _write_auth_eval(tmp_path / "contest_auth_eval.json")
    calls: list[dict[str, object]] = []

    class Update:
        accepted = True
        posterior_n_anchors_after = 7
        refusal_reason = ""

    def fake_update(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return Update()

    monkeypatch.setattr(tool, "posterior_update_locked_from_auth_eval_json", fake_update)

    note = tool._maybe_update_posterior(
        {"output_dir": str(tmp_path), "passed": True},
        {"lane_id": "lane_pr106_r2"},
    )

    assert calls[0]["path"] == tmp_path / "contest_auth_eval.json"
    assert calls[0]["architecture_class"] == "lane_pr106_r2"
    assert "posterior_update=accepted" in note
