# SPDX-License-Identifier: MIT
"""Tests for Catalog #308 ``check_kill_verdict_enumerates_alternative_probe_methodologies``.

Per FALSIFICATION-AUDIT-v2 Pattern E (commit ``c5e4953e6``).

The gate refuses substrate-class kill memos WITHOUT enumeration of >=3
alternative probe methodologies that could produce different verdicts.

Empirical anchor: Wunderkind G1 v2 PIVOT kill-verdict based on a single
per-pair-dominant SegNet argmax probe; 4 alternative reducers (per-pair
HISTOGRAM / per-region HISTOGRAM / per-segment-class / per-temporal-window)
were UNPROBED.

Sister of Catalog #301 (kill memos have substrate compatibility evidence),
#292 (per-deliberation assumption surfacing), #307 (paradigm-vs-
implementation classification).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_kill_verdict_enumerates_alternative_probe_methodologies,
)


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


def test_308_live_repo_count_bounded() -> None:
    violations = check_kill_verdict_enumerates_alternative_probe_methodologies(
        strict=False, verbose=False,
    )
    assert len(violations) <= 20, (
        f"Live count {len(violations)} > 20; backfill regressed?"
    )


# ---------------------------------------------------------------------------
# Out-of-scope behavior
# ---------------------------------------------------------------------------


def test_308_no_research_dir_returns_empty(tmp_path: Path) -> None:
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_308_memo_without_kill_token_out_of_scope(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "foo_design_20260516.md", "no kill here.\n")
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_308_pre_cutoff_exempt(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "foo_design_20260515.md",
        "VERDICT: KILL\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_308_non_design_memo_not_scanned(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "log_20260516.md", "VERDICT: KILL\n")
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Positive: kill without alternative-probe enumeration = flagged
# ---------------------------------------------------------------------------


def test_308_kill_without_enumeration_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "VERDICT: KILL -- TECHNIQUE FALSIFIED. The single SegNet argmax probe failed.\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_308_falsification_memo_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "y_falsification_20260516.md",
        "# Y Falsification\n\nClass-kill on the lane; single probe.\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_308_multiple_violations_aggregated(tmp_path: Path) -> None:
    research = _research_dir(tmp_path)
    _write(research, "a_design_20260516.md", "VERDICT: FALSIFIED\n")
    _write(research, "b_kill_20260516.md", "VERDICT: KILL\n")
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 2


# ---------------------------------------------------------------------------
# Negative: enumeration accepts
# ---------------------------------------------------------------------------


def test_308_alternative_reducers_phrase_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "VERDICT: KILL. 4 alternative reducers were probed: per-pair HISTOGRAM, per-region HISTOGRAM, per-segment-class, per-temporal-window.\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_308_split_verdict_token_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "VERDICT: FALSIFIED. T2 council Q1 SPLIT-VERDICT.\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_308_probe_disambiguator_path_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "VERDICT: KILL. See tools/probe_g1_disambiguator.py for alternative probe.\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_308_ratify_methodology_phrase_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "VERDICT: KILL. RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY + REQUEST-REINVESTIGATION-OF-ALTERNATIVES.\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_308_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "VERDICT: KILL\n\n# ALTERNATIVE_PROBE_METHODOLOGIES_ENUMERATED_OK:cited-in-sister-memo-y\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_308_placeholder_rationale_rejected(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "VERDICT: KILL\n\n# ALTERNATIVE_PROBE_METHODOLOGIES_ENUMERATED_OK:<rationale>\n",
    )
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_308_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "VERDICT: KILL -- no alternative probes.\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_kill_verdict_enumerates_alternative_probe_methodologies(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "Catalog #308" in str(exc_info.value)
    assert "Pattern E" in str(exc_info.value)


def test_308_strict_mode_silent_on_clean_repo(tmp_path: Path) -> None:
    _research_dir(tmp_path)
    v = check_kill_verdict_enumerates_alternative_probe_methodologies(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert v == []
