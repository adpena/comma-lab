# SPDX-License-Identifier: MIT
"""Tests for Catalog #192 -- macOS-CPU advisory must not be promoted.

Per CLAUDE.md operator routing 2026-05-13 + "Submission auth eval — BOTH
CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable: macOS-CPU
is wired as a free advisory proxy; the STRICT gate refuses any persisted
artifact that combines ``evidence_grade="macOS-CPU-advisory"`` (or
``evidence_tag`` carrying ``[macOS-CPU advisory``) with any of:
``score_claim=True``, ``promotion_eligible=True``,
``ready_for_exact_eval_dispatch=True``.

The check scans ``.omx/state/``, ``experiments/results/``, ``reports/``
under the repo root for ``*.json`` / ``*.jsonl`` files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_macos_cpu_advisory_not_promoted_without_linux_verification,
)


def _make_fake_repo(tmp_path: Path) -> Path:
    """Create the canonical scan-root directory layout under tmp_path."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / "experiments" / "results").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    return tmp_path


# ----------------------------------------------------------------------------
# 1) Live-repo regression guard (must always be 0)
# ----------------------------------------------------------------------------


def test_live_repo_clean_no_violations() -> None:
    """The live repo must remain at 0 advisory-promoted violations."""
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        strict=False, verbose=False
    )
    assert violations == [], (
        "Live repo MUST stay at 0 macOS-CPU-advisory authority violations. "
        "Found:\n  " + "\n  ".join(violations[:5])
    )


# ----------------------------------------------------------------------------
# 2) Positive: catches synthetic violations
# ----------------------------------------------------------------------------


def test_catches_promoted_score_claim_in_omx_state(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / ".omx" / "state" / "bad_advisory.json"
    bad.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "evidence_tag": "[macOS-CPU advisory only]",
        "score_claim": True,
        "archive_sha256": "ab" * 32,
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert ".omx/state/bad_advisory.json" in violations[0]
    assert "score_claim=True" in violations[0]


def test_catches_promoted_promotion_eligible_in_experiments_results(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / "experiments" / "results" / "bad.json"
    bad.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "promotion_eligible": True,
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "promotion_eligible=True" in violations[0]


def test_catches_promoted_ready_for_exact_eval_in_reports(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / "reports" / "frontier.json"
    bad.write_text(json.dumps({
        "evidence_tag": "[macOS-CPU advisory only]",
        "ready_for_exact_eval_dispatch": True,
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "ready_for_exact_eval_dispatch=True" in violations[0]


def test_catches_nested_rows_in_manifest(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / "experiments" / "results" / "manifest.json"
    bad.write_text(json.dumps({
        "schema": "macos_cpu_advisory_signal_manifest.v1",
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": False,  # top-level OK
        "rows": [
            {
                "family": "f1",
                "variant_id": "v1",
                "evidence_grade": "macOS-CPU-advisory",
                "score_claim": False,  # OK row
            },
            {
                "family": "f2",
                "variant_id": "v2",
                "evidence_grade": "macOS-CPU-advisory",
                "promotion_eligible": True,  # VIOLATION
            },
        ],
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    # 1 violation from the nested row.
    assert len(violations) == 1
    assert "promotion_eligible=True" in violations[0]


def test_catches_jsonl_promoted_row(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / ".omx" / "state" / "manifest.jsonl"
    rows = [
        {"evidence_grade": "macOS-CPU-advisory", "score_claim": False},
        {"evidence_grade": "macOS-CPU-advisory", "score_claim": True},  # VIOLATION
        {"evidence_grade": "macOS-CPU-advisory", "promotion_eligible": False},
    ]
    bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "score_claim=True" in violations[0]


def test_strict_raises_PreflightError(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / ".omx" / "state" / "v.json"
    bad.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": True,
    }))
    with pytest.raises(PreflightError) as exc_info:
        check_macos_cpu_advisory_not_promoted_without_linux_verification(
            repo_root=root, strict=True, verbose=False
        )
    msg = str(exc_info.value)
    assert "score_claim=True" in msg
    assert "1:1 CONTEST-COMPLIANT" in msg or "Catalog #192" in msg


def test_strict_silent_when_no_violations(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    good = root / ".omx" / "state" / "ok.json"
    good.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }))
    # Should not raise.
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=True, verbose=False
    )
    assert violations == []


# ----------------------------------------------------------------------------
# 3) Negative: out-of-scope rows / paths ignored
# ----------------------------------------------------------------------------


def test_ignores_rows_without_macos_cpu_tag(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    p = root / ".omx" / "state" / "not_advisory.json"
    p.write_text(json.dumps({
        "evidence_grade": "[contest-CUDA]",
        "score_claim": True,  # Not a #192 violation — different tag.
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_ignores_files_outside_scan_roots(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    elsewhere = root / "outside" / "dir"
    elsewhere.mkdir(parents=True)
    bad = elsewhere / "bad.json"
    bad.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": True,
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_ignores_non_json_extensions(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / ".omx" / "state" / "bad.txt"
    bad.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": True,
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


def test_ignores_malformed_json(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / ".omx" / "state" / "broken.json"
    bad.write_text("not valid json {{{ macOS-CPU-advisory")
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    # Malformed file is skipped, not flagged as a violation.
    assert violations == []


def test_ignores_files_without_advisory_token(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    p = root / "experiments" / "results" / "irrelevant.json"
    p.write_text(json.dumps({"family": "f", "score": 0.1, "score_claim": True}))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


# ----------------------------------------------------------------------------
# 4) File-level waiver
# ----------------------------------------------------------------------------


def test_waiver_token_exempts_explicitly_superseded_artifact(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    bad = root / ".omx" / "state" / "waived.json"
    # Embed the waiver token inside the JSON's free-form notes field. The
    # whole-file fast-pre-filter sees the token and exempts the file.
    bad.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": True,
        "notes": (
            "# MACOS_CPU_ADVISORY_PROMOTED_OK:paired_gha_linux_x86_64_anchor_"
            "landed_at_xyz_sha256_<...>"
        ),
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert violations == []


# ----------------------------------------------------------------------------
# 5) Multiple violations across multiple files
# ----------------------------------------------------------------------------


def test_aggregates_multiple_violations(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    p1 = root / ".omx" / "state" / "v1.json"
    p1.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": True,
    }))
    p2 = root / "experiments" / "results" / "v2.json"
    p2.write_text(json.dumps({
        "evidence_grade": "macOS-CPU-advisory",
        "promotion_eligible": True,
    }))
    p3 = root / "reports" / "v3.json"
    p3.write_text(json.dumps({
        "evidence_tag": "[macOS-CPU advisory only]",
        "ready_for_exact_eval_dispatch": True,
    }))
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=root, strict=False, verbose=False
    )
    assert len(violations) == 3
    assert any("score_claim=True" in v for v in violations)
    assert any("promotion_eligible=True" in v for v in violations)
    assert any("ready_for_exact_eval_dispatch=True" in v for v in violations)


# ----------------------------------------------------------------------------
# 6) Edge: empty repo / missing scan dirs
# ----------------------------------------------------------------------------


def test_empty_repo_returns_no_violations(tmp_path: Path) -> None:
    """When the scan roots don't exist, the check returns cleanly."""
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_partial_scan_root_layout_okay(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir(parents=True)
    # Only reports/ exists; .omx/state/ and experiments/results/ absent.
    violations = check_macos_cpu_advisory_not_promoted_without_linux_verification(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ----------------------------------------------------------------------------
# 7) Wired into preflight_all
# ----------------------------------------------------------------------------


def test_strict_callsite_wired_in_preflight_all() -> None:
    """Catalog #176 sister assertion: the strict callsite must be present in
    preflight.py inside the canonical preflight_all() body."""
    repo_root = Path(__file__).resolve().parents[3]
    preflight_text = (repo_root / "src" / "tac" / "preflight.py").read_text(encoding="utf-8")
    # Check both that the function is defined AND that it's invoked strict=True
    # inside preflight_all.
    assert (
        "def check_macos_cpu_advisory_not_promoted_without_linux_verification("
        in preflight_text
    )
    # The strict=True wire-in.
    assert (
        "check_macos_cpu_advisory_not_promoted_without_linux_verification(\n"
        "            strict=True"
    ) in preflight_text, "Catalog #192 must be wired strict=True in preflight_all()"
