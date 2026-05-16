# SPDX-License-Identifier: MIT
"""Tests for Catalog #307 ``check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification``.

Per FALSIFICATION-AUDIT-v2 Pattern D (commit ``c5e4953e6``).

The gate refuses memos containing kill / falsified / retired tokens WITHOUT
explicit paradigm-vs-implementation falsification classification.

Empirical anchor: NSCS06 v6 Strip-Everything was initially FALSIFIED at
105.15 substrate-class; FALSIFICATION-AUDIT-v2 Lens 7 promoted to Tier 1
paradigm-intact after v7 unwound 4-of-7 cargo-cults and achieved 58.89
(44% reduction) -- proof the paradigm was INTACT.

Sister of Catalog #290 (canonical-vs-unique decision per layer), #294
(9-dim checklist), #296 (predicted-band Dykstra), #301 (kill memos have
substrate compatibility evidence).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _research_dir(repo_root: Path) -> Path:
    d = repo_root / ".omx" / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(d: Path, name: str, body: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_307_live_repo_count_bounded() -> None:
    """Live count is bounded; warn-only at landing.

    At landing 4 violations exist (pre-Pattern-D resurrection memos);
    regression guard ceilings at 20 to allow some sister-subagent drift.
    """
    violations = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        strict=False, verbose=False,
    )
    assert len(violations) <= 20, (
        f"Live count {len(violations)} > 20; backfill regressed?"
    )


# ---------------------------------------------------------------------------
# Out-of-scope behavior
# ---------------------------------------------------------------------------


def test_307_no_research_dir_returns_empty(tmp_path: Path) -> None:
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_307_memo_without_kill_token_out_of_scope(tmp_path: Path) -> None:
    """A memo without kill/falsified tokens is out-of-scope."""
    _write(_research_dir(tmp_path), "foo_design_20260516.md", "# Foo\n\nNothing about kill.\n")
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_307_non_design_memo_not_scanned(tmp_path: Path) -> None:
    """Files not matching `*_design|falsification|audit|kill|retir_<YYYYMMDD>.md` are skipped."""
    _write(_research_dir(tmp_path), "random_file.md", "TECHNIQUE FALSIFIED\n")
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_307_pre_cutoff_exempt(tmp_path: Path) -> None:
    """Pre-2026-05-16 memos exempt by date filter."""
    _write(
        _research_dir(tmp_path),
        "foo_design_20260515.md",
        "TECHNIQUE FALSIFIED -- no classification.\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Positive: trigger without classification = flagged
# ---------------------------------------------------------------------------


def test_307_technique_falsified_without_classification_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X Design\n\nVERDICT: KILL -- TECHNIQUE FALSIFIED at substrate class.\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_307_substrate_falsified_without_classification_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "y_falsification_20260516.md",
        "# Y Falsification\n\nThe SUBSTRATE FALSIFIED at empirical anchor.\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_307_audit_memo_with_class_kill_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "z_audit_20260516.md",
        "# Z Audit\n\nClass-kill on the lane.\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_307_multiple_violations_aggregated(tmp_path: Path) -> None:
    research = _research_dir(tmp_path)
    _write(research, "a_design_20260516.md", "VERDICT: FALSIFIED\n")
    _write(research, "b_kill_20260516.md", "VERDICT: KILL\n")
    _write(research, "c_audit_20260516.md", "TECHNIQUE FALSIFIED\n")
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 3


# ---------------------------------------------------------------------------
# Negative: classification token accepts
# ---------------------------------------------------------------------------


def test_307_paradigm_level_falsification_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "a_design_20260516.md",
        "# A\n\nTECHNIQUE FALSIFIED\n\nThis is a PARADIGM-LEVEL FALSIFICATION; the underlying class is structurally infeasible.\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_307_implementation_level_falsification_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "a_design_20260516.md",
        "# A\n\nTECHNIQUE FALSIFIED\n\nThis is an IMPLEMENTATION-LEVEL FALSIFICATION; the paradigm is INTACT.\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_307_paradigm_intact_token_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "a_design_20260516.md",
        "# A\n\nTECHNIQUE FALSIFIED\n\nVerdict: paradigm-intact; iterative cargo-cult unwind queued.\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_307_implementation_cargo_cult_token_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "a_design_20260516.md",
        "# A\n\nTECHNIQUE FALSIFIED\n\nFalsification root cause: implementation-cargo-cult, NOT paradigm-class.\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_307_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "a_design_20260516.md",
        "# A\n\nTECHNIQUE FALSIFIED\n\n<!-- # PARADIGM_VS_IMPLEMENTATION_FALSIFICATION_OK:cited-in-sister-memo-x -->\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_307_placeholder_rationale_rejected(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "a_design_20260516.md",
        "# A\n\nTECHNIQUE FALSIFIED\n\n<!-- # PARADIGM_VS_IMPLEMENTATION_FALSIFICATION_OK:<rationale> -->\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_307_reason_placeholder_rejected(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "a_design_20260516.md",
        "# A\n\nTECHNIQUE FALSIFIED\n\n# PARADIGM_VS_IMPLEMENTATION_FALSIFICATION_OK:<reason>\n",
    )
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_307_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "a_design_20260516.md",
        "TECHNIQUE FALSIFIED -- no classification.\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "Catalog #307" in str(exc_info.value)
    assert "Pattern D" in str(exc_info.value)


def test_307_strict_mode_silent_on_clean_repo(tmp_path: Path) -> None:
    _research_dir(tmp_path)
    # No memos = no violations
    v = check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert v == []
