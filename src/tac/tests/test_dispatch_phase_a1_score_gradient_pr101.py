from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "dispatch_phase_a1_score_gradient_pr101.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "dispatch_phase_a1_score_gradient_pr101_under_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_success_without_session_id_closes_claim_and_manifest_is_terminal(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_tool()
    claims: list[dict] = []

    def fake_claim_lane(**kwargs):
        claims.append(dict(kwargs))
        return 0

    def fake_dispatch_lightning(**_kwargs):
        return None, True

    monkeypatch.setattr(tool, "claim_lane", fake_claim_lane)
    monkeypatch.setattr(tool, "dispatch_lightning", fake_dispatch_lightning)

    assert tool.main(
        [
            "--pr101-archive",
            "CLAUDE.md",
            "--video-path",
            "CLAUDE.md",
            "--pr101-source-dir",
            "src",
            "--provider",
            "lightning",
            "--gpu-tier",
            "T4",
            "--output-root",
            str(tmp_path),
        ]
    ) == 0

    statuses = [claim.get("status", "active_dispatching") for claim in claims]
    assert statuses == ["active_dispatching", "fired_no_session_id_verify_manually"]
    assert claims[1]["force"] is True

    manifests = list(tmp_path.glob("track1_phase_a1_score_gradient_*/build_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))

    assert manifest["session_id"] is None
    assert manifest["dispatch_status"] == "fired_no_session_id_verify_manually"
    assert manifest["dispatch_blockers"] == ["fired_no_session_id_verify_manually"]
    assert manifest["evidence_grade"] == "[advisory only — no dispatch]"
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "verify platform state manually" in manifest["harvest_command_hint"]

    captured = capsys.readouterr()
    assert "Lane claim closed as 'fired_no_session_id_verify_manually'" in captured.err
