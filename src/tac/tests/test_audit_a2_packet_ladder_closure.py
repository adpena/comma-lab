"""Tests for the A2 packet-ladder closure audit CLI."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    path = REPO_ROOT / "tools" / "audit_a2_packet_ladder_closure.py"
    spec = importlib.util.spec_from_file_location("_audit_a2_packet_ladder_closure", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_a2_packet_ladder_closure_accepts_repo_root(tmp_path: Path) -> None:
    """All-lanes preflight invokes guards with --repo-root; keep that contract."""
    mod = _load_module()
    out = tmp_path / "report.json"

    rc = mod.main(["--repo-root", str(tmp_path), "--strict", "--json-out", str(out)])

    assert rc == 0
    assert out.exists()
    assert "a2_packet_ladder_closure_audit_v1" in out.read_text(encoding="utf-8")


def test_audit_a2_packet_ladder_closure_scans_ignored_local_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    manifest = (
        tmp_path
        / "experiments"
        / "results"
        / "a2_local"
        / "candidate_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "a2_candidate_manifest.v1",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": ["no_exact_cuda_auth_eval"],
                "runtime_closure": {
                    "cleared_blockers": ["packet_local_inflate_parity_not_run"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = mod.audit(tmp_path)

    assert report["passed"] is False
    assert report["scanned_artifacts"] == [
        "experiments/results/a2_local/candidate_manifest.json"
    ]
    assert any("cleared without evidence" in item for item in report["violations"])
