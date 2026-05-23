# SPDX-License-Identifier: MIT
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


def test_terminal_status_fails_missing_canonical_artifact() -> None:
    tool = _load_tool()

    status = tool._terminal_status(
        {
            "status": "recovered_missing_canonical_auth_eval_artifact",
            "passed": False,
            "score_claim": False,
        },
        {"axis": "contest_cuda"},
    )

    assert status == "failed_modal_auth_eval_missing_canonical_artifact"


def test_terminal_notes_include_exact_custody_fields(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_auth_eval(tmp_path / "contest_auth_eval.json")

    notes = tool._terminal_notes(
        {
            "output_dir": str(tmp_path),
            "passed": True,
            "result_json": str(tmp_path / "modal_cuda_auth_eval_result.json"),
        },
        {"lane_id": "lane_pr106_r2"},  # FAKE_LANE_OK:test-fixture lane_id
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


def test_auth_eval_artifact_path_rejects_modal_result_json_fallback(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    _write_auth_eval(tmp_path / "modal_cuda_auth_eval_result.json")

    artifact = tool._auth_eval_artifact_path(
        {
            "output_dir": str(tmp_path),
            "result_json": str(tmp_path / "modal_cuda_auth_eval_result.json"),
        }
    )

    assert artifact is None


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
        {"lane_id": "lane_pr106_r2"},  # FAKE_LANE_OK:test-fixture lane_id
    )

    assert calls[0]["path"] == tmp_path / "contest_auth_eval.json"
    assert calls[0]["architecture_class"] == "lane_pr106_r2"  # FAKE_LANE_OK:test-fixture lane_id
    assert "posterior_update=accepted" in note


def test_main_requires_auditable_no_close_reason(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    recover_calls: list[dict[str, object]] = []

    def fake_recover(**kwargs):
        recover_calls.append(kwargs)
        return {"status": "recovered", "passed": True}

    monkeypatch.setattr(tool, "recover_modal_auth_eval", fake_recover)

    try:
        tool.main(["--output-dir", str(tmp_path), "--no-close-claim"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected argparse failure")

    assert recover_calls == []


def test_main_fails_loud_when_terminal_metadata_lacks_claim_fields(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    tool = _load_tool()
    close_calls: list[dict[str, object]] = []

    monkeypatch.setattr(tool, "read_spawn_metadata", lambda _out_dir: {"axis": "contest_cuda"})
    monkeypatch.setattr(
        tool,
        "recover_modal_auth_eval",
        lambda **_kwargs: {"status": "recovered", "passed": True, "score_claim": True},
    )
    monkeypatch.setattr(tool, "terminal_modal_auth_eval_claim", lambda **kwargs: close_calls.append(kwargs))

    rc = tool.main(["--output-dir", str(tmp_path), "--no-posterior-update"])

    assert rc == tool.CLAIM_CLOSURE_ERROR_RC
    assert close_calls == []
    err = capsys.readouterr().err
    assert "cannot close the dispatch claim" in err
    assert "lane_id" in err
    assert "instance_job_id" in err


def test_main_allows_no_close_only_with_auditable_reason(monkeypatch, tmp_path: Path, capsys) -> None:
    tool = _load_tool()

    monkeypatch.setattr(tool, "read_spawn_metadata", lambda _out_dir: {"axis": "contest_cuda"})
    monkeypatch.setattr(
        tool,
        "recover_modal_auth_eval",
        lambda **_kwargs: {"status": "recovered", "passed": True, "score_claim": True},
    )

    def forbidden_close(**_kwargs):
        raise AssertionError("terminal claim should be suppressed by explicit operator reason")

    monkeypatch.setattr(tool, "terminal_modal_auth_eval_claim", forbidden_close)

    rc = tool.main(
        [
            "--output-dir",
            str(tmp_path),
            "--no-posterior-update",
            "--no-close-claim",
            "--skip-close-claim-reason",
            "duplicate terminal row already recorded by paired harvester",
        ]
    )

    assert rc == 0
    assert "duplicate terminal row already recorded" in capsys.readouterr().err


def test_main_skips_duplicate_terminal_recovery_without_posterior_touch(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_tool()
    claims_path = tmp_path / ".omx" / "state" / "active_lane_dispatch_claims.md"
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    result_json = str(tmp_path / "modal_cuda_auth_eval_result.json")
    claims_path.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                (
                    "| 2026-05-23T20:11:04Z | codex:modal_auth_eval | lane_cuda | modal | job_cuda | "
                    "2026-05-23T20:11:04Z | completed_contest_cuda_modal_auth_eval_recovered | "
                    f"Modal auth eval recovered; passed=True; result_json={result_json}; "
                    "posterior_update=accepted |"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        tool,
        "read_spawn_metadata",
        lambda _out_dir: {
            "axis": "contest_cuda",
            "lane_id": "lane_cuda",
            "instance_job_id": "job_cuda",
            "claim_agent": "codex:modal_auth_eval",
            "claim_platform": "modal",
        },
    )
    monkeypatch.setattr(
        tool,
        "recover_modal_auth_eval",
        lambda **_kwargs: {
            "status": "recovered",
            "passed": True,
            "score_claim": True,
            "result_json": result_json,
        },
    )
    monkeypatch.setattr(
        tool,
        "posterior_update_locked_from_auth_eval_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("posterior update must be skipped")
        ),
    )
    monkeypatch.setattr(
        tool,
        "terminal_modal_auth_eval_claim",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal claim must be skipped")
        ),
    )

    rc = tool.main(["--output-dir", str(tmp_path)])

    assert rc == 0
    assert "already closed" in capsys.readouterr().err


def test_main_returns_nonzero_for_missing_canonical_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    close_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        tool,
        "read_spawn_metadata",
        lambda _out_dir: {
            "axis": "contest_cuda",
            "lane_id": "lane_cuda",
            "instance_job_id": "job_cuda",
            "claim_agent": "codex:modal_auth_eval",
            "claim_platform": "modal",
        },
    )
    monkeypatch.setattr(
        tool,
        "recover_modal_auth_eval",
        lambda **_kwargs: {
            "status": "recovered_missing_canonical_auth_eval_artifact",
            "passed": False,
            "returncode": 97,
            "score_claim": False,
            "promotion_eligible": False,
            "diagnostic_blockers": ["missing_canonical_contest_auth_eval_json"],
        },
    )
    monkeypatch.setattr(
        tool,
        "terminal_modal_auth_eval_claim",
        lambda **kwargs: close_calls.append(kwargs),
    )

    rc = tool.main(["--output-dir", str(tmp_path), "--no-posterior-update"])

    assert rc == 97
    assert close_calls[0]["status"] == (
        "failed_modal_auth_eval_missing_canonical_artifact"
    )
