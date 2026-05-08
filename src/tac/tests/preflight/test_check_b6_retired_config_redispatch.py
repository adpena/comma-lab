"""Tests for B6 — retired-config redispatch guard scanner.

Every retired evidence row must carry dispatch_blockers containing
``reactivation_required_before_new_dispatch`` AND non-empty
``reactivation_criteria`` so the parallel-dispatch actuator can block
re-dispatch of a known-bad config.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCANNER = REPO_ROOT / "tools" / "check_predispatch_retired_config_warning.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_b6_test", SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(repo: Path, rel: str, rows: list[dict]) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_b6_flags_retired_without_guards(tmp_path: Path) -> None:
    mod = _load_module()
    _write(
        tmp_path,
        "reports/cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "contest_dispatch_verdict": "measured_config_retired_exact_cuda_negative",
                "measured_config_status": "measured_config_retired",
            }
        ],
    )
    findings = mod.scan(tmp_path)
    assert len(findings) == 1


def test_b6_accepts_retired_with_full_guards(tmp_path: Path) -> None:
    mod = _load_module()
    _write(
        tmp_path,
        "reports/cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "contest_dispatch_verdict": "measured_config_retired_exact_cuda_negative",
                "dispatch_blockers": [
                    "measured_config_retired_exact_cuda_negative",
                    "reactivation_required_before_new_dispatch",
                ],
                "reactivation_criteria": [
                    "retrain renderer under scorer-aware loss",
                    "prove byte-closed runtime packet",
                ],
            }
        ],
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b6_rejects_retired_with_blocker_only(tmp_path: Path) -> None:
    mod = _load_module()
    _write(
        tmp_path,
        "reports/cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "measured_config_status": "measured_config_retired",
                "dispatch_blockers": ["reactivation_required_before_new_dispatch"],
                # missing reactivation_criteria
            }
        ],
    )
    findings = mod.scan(tmp_path)
    assert len(findings) == 1


def test_b6_skips_non_retired_rows(tmp_path: Path) -> None:
    mod = _load_module()
    _write(
        tmp_path,
        "reports/cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "evidence_grade": "[CPU-prep]",
                "score_claim": False,
            }
        ],
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b6_strict_exits_nonzero(tmp_path: Path) -> None:
    mod = _load_module()
    _write(
        tmp_path,
        "reports/cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "contest_dispatch_verdict": "measured_config_retired_exact_cuda_negative",
            }
        ],
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--strict"])
    assert rc == 1
