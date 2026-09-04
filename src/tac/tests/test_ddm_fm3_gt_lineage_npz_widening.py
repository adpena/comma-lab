# SPDX-License-Identifier: MIT
"""Tests for the ``ddm_fm3`` GT-lineage ``.npz`` widening (2026-09-04).

The gap: ``_GT_LINEAGE_ARTIFACT_PATTERNS`` matched only ``.npy`` and ``.pt``, so
``gt_n600.npz`` -- the PyAV-lineage table the born trainer pins as authority --
sat in the gate's blind spot (ddm_bh1 finding 2).

These tests pin BEHAVIOUR, not markers. Each one would fail if the widening were
reverted, mis-scoped, or wired so that it changed the primary gate's contract:

* the vocabulary reaches the real ``gt_n*.npz`` caches and stops short of the
  synthetic fixtures;
* every exemption the primary scope honours (DALI declaration, registry route,
  same-line waiver, comment-only mention, tests) still holds under the widened
  vocabulary -- one shared scanner, so the exclusions cannot drift;
* the widening is REPORT-ONLY: it never enters the return value and never
  raises, so the standalone strict surface's contract is untouched;
* the derived prefilter covers every shipped pattern -- the anti-drift invariant
  that stops a half-added pattern from silently matching nothing (the
  VACUITY==PASS shape this scanner was bitten by once already).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tac.preflight import (
    _GT_LINEAGE_ARTIFACT_PATTERNS,
    _GT_LINEAGE_NPZ_PATTERNS,
    _GT_LINEAGE_NPZ_PREFILTER_STEMS,
    _GT_LINEAGE_PREFILTER_STEMS,
    _GT_LINEAGE_PRIMARY_PREFILTER_STEMS,
    PreflightError,
    _check_351_gt_lineage_objective_custody,
    _gt_artifact_hits_outside_comment,
    check_gt_lineage_objective_custody,
    gt_lineage_npz_widening_findings,
)

WIDENED = (*_GT_LINEAGE_ARTIFACT_PATTERNS, *_GT_LINEAGE_NPZ_PATTERNS)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# ── the vocabulary: what it reaches, and where it deliberately stops ──────────


@pytest.mark.parametrize(
    "artifact",
    [
        "gt_n600.npz",
        "gt_n96.npz",
        "gt_n24.npz",
        "gt_n6.npz",
        "gt_n2.npz",
        "gt_n8.npz",
        "gt_n600_lstars_slim.npz",
        "gt_n600_sR.npz",
        "gt_strided_n200.npz",  # count-bearing but not gt_n-prefixed
        "gt_heldout_n400.npz",
    ],
)
def test_real_gt_caches_are_in_the_widened_scope(tmp_path, artifact):
    _write(tmp_path, "tools/a.py", f'P = "caches/{artifact}"\n')
    assert len(_check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS)) == 1


@pytest.mark.parametrize(
    "artifact",
    [
        "gt_tiny.npz",
        "gt_synth.npz",
        "gt_bad.npz",
        "gt_bad_geometry.npz",
        "gt_exact.npz",
        "gt_nokey.npz",
        "gt_nomargin.npz",
        "gt_nN.npz",  # a template placeholder, not a materialised cache
    ],
)
def test_synthetic_fixtures_stay_out_of_scope(tmp_path, artifact):
    """A fixture with no decode lineage has nothing to declare; flagging it is noise."""
    _write(tmp_path, "tools/a.py", f'P = "caches/{artifact}"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


def test_gt_argmax_npy_stays_out_of_scope_under_the_widening(tmp_path):
    """The sp2 exclusion is unchanged -- one settled DALI lineage, 93-file flood."""
    _write(tmp_path, "tools/a.py", 'P = "x/gt_argmax_n600.npy"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


def test_npz_is_invisible_to_the_primary_vocabulary(tmp_path):
    """The blind spot itself, pinned: without the widening the gate sees nothing."""
    _write(tmp_path, "tools/a.py", 'P = "caches/gt_n600.npz"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []
    assert len(_check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS)) == 1


# ── every primary exemption still holds under the widened vocabulary ──────────


def test_dali_declaration_clears_an_npz_site(tmp_path):
    _write(tmp_path, "tools/a.py", 'P = "caches/gt_n600_dali.npz"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


def test_registry_route_clears_an_npz_file(tmp_path):
    _write(
        tmp_path,
        "tools/a.py",
        "from tac.gt_lineage import assert_gt_lineage\n"
        'P = "caches/gt_n600.npz"\nassert_gt_lineage(P, expected="PYAV")\n',
    )
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


def test_same_line_waiver_clears_an_npz_site(tmp_path):
    _write(
        tmp_path,
        "tools/a.py",
        'P = "caches/gt_n600.npz"  # GT_LINEAGE_OK: PyAV is the intended target here\n',
    )
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


def test_placeholder_waiver_does_not_clear_an_npz_site(tmp_path):
    _write(
        tmp_path,
        "tools/a.py",
        'P = "caches/gt_n600.npz"  # GT_LINEAGE_OK: <rationale>\n',
    )
    assert len(_check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS)) == 1


def test_comment_only_mention_of_an_npz_table_is_not_a_consumption(tmp_path):
    _write(
        tmp_path,
        "tools/a.py",
        'P = "caches/gt_cache_dali.pt"  # the sister table is gt_n600.npz\n',
    )
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


def test_tests_are_out_of_scope_under_the_widening(tmp_path):
    _write(tmp_path, "tools/test_a.py", 'P = "caches/gt_n600.npz"\n')
    _write(tmp_path, "src/pkg/tests/b.py", 'P = "caches/gt_n600.npz"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


def test_results_directory_is_out_of_scope_under_the_widening(tmp_path):
    _write(tmp_path, "experiments/results/run/a.py", 'P = "caches/gt_n600.npz"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


def test_a_docstring_mention_is_not_a_consumption(tmp_path):
    _write(
        tmp_path,
        "tools/a.py",
        '"""This module explains why gt_n600.npz is the PyAV lineage."""\n'
        'P = "caches/gt_cache_dali.pt"\n',
    )
    assert _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS) == []


# ── the parameter plumbing itself ────────────────────────────────────────────


def test_scanner_defaults_to_the_primary_vocabulary(tmp_path):
    """Omitting ``patterns`` must behave exactly as before the refactor."""
    _write(tmp_path, "tools/a.py", 'A = "caches/gt_n600.npz"\nB = "x/gt_cache_av.pt"\n')
    default = _check_351_gt_lineage_objective_custody(tmp_path)
    explicit = _check_351_gt_lineage_objective_custody(
        tmp_path, _GT_LINEAGE_ARTIFACT_PATTERNS
    )
    assert default == explicit
    assert len(default) == 1  # only the .pt, not the .npz


def test_line_helper_defaults_to_the_primary_vocabulary():
    line = 'P = "caches/gt_n600.npz"'
    assert _gt_artifact_hits_outside_comment(line) == []
    assert _gt_artifact_hits_outside_comment(line, WIDENED) == ["gt_n600.npz"]


def test_line_helper_still_ignores_trailing_comments_under_the_widening():
    line = 'P = "caches/gt_cache_dali.pt"  # sister of gt_n600.npz'
    assert _gt_artifact_hits_outside_comment(line, WIDENED) == ["gt_cache_dali.pt"]


# ── the anti-drift invariant ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("artifact", "patterns", "stems"),
    [
        ("gt_first6_n600.npy", _GT_LINEAGE_ARTIFACT_PATTERNS, _GT_LINEAGE_PRIMARY_PREFILTER_STEMS),
        ("gt_cache_av.pt", _GT_LINEAGE_ARTIFACT_PATTERNS, _GT_LINEAGE_PRIMARY_PREFILTER_STEMS),
        ("gt_n600.npz", _GT_LINEAGE_NPZ_PATTERNS, _GT_LINEAGE_NPZ_PREFILTER_STEMS),
        ("gt_n96.npz", _GT_LINEAGE_NPZ_PATTERNS, _GT_LINEAGE_NPZ_PREFILTER_STEMS),
        ("gt_strided_n200.npz", _GT_LINEAGE_NPZ_PATTERNS, _GT_LINEAGE_NPZ_PREFILTER_STEMS),
    ],
)
def test_each_scope_prefilter_reaches_its_own_patterns(artifact, patterns, stems):
    """A pattern its prefilter cannot reach matches nothing and reports a clean 0.

    That is the VACUITY==PASS failure this scanner already suffered once (a
    relative root made every candidate skip, so 11 real findings read as 0).
    Each scope carries its own stems -- a shared list made the PRIMARY scan pay
    the widening's cost (160 -> 676 candidates) -- so the invariant is checked
    per scope, not against the union.
    """
    assert any(p.search(artifact) for p in patterns), f"{artifact} unmatched"
    assert re.compile("|".join(stems)).search(artifact), (
        f"{artifact} matches a shipped pattern but NOT its own prefilter -- "
        "the scanner would skip its file and report a vacuous clean 0"
    )


def test_union_prefilter_is_the_two_scopes_and_nothing_else():
    """The union must stay derived; a hand-maintained third list would drift."""
    assert _GT_LINEAGE_PREFILTER_STEMS == (
        *_GT_LINEAGE_PRIMARY_PREFILTER_STEMS,
        *_GT_LINEAGE_NPZ_PREFILTER_STEMS,
    )


def test_primary_prefilter_does_not_pay_the_widening_cost():
    """The measured regression, pinned: primary stems must not match .npz names."""
    primary = re.compile("|".join(_GT_LINEAGE_PRIMARY_PREFILTER_STEMS))
    assert not primary.search("gt_n600.npz")
    assert not primary.search("gt_strided_n200.npz")


# ── report-only: the widening must not change the gate's contract ────────────


def test_widening_findings_exclude_the_primary_findings(tmp_path):
    """No double counting: a site the primary gate already reports is not repeated."""
    _write(tmp_path, "tools/a.py", 'A = "x/gt_cache_av.pt"\nB = "caches/gt_n600.npz"\n')
    primary = _check_351_gt_lineage_objective_custody(tmp_path)
    widened = gt_lineage_npz_widening_findings(tmp_path)
    assert len(primary) == 1
    assert len(widened) == 1
    assert not set(primary) & set(widened)
    assert "gt_n600.npz" in widened[0]


def test_widening_accepts_precomputed_primary_findings(tmp_path):
    """Reusing the caller's primary result must give the same answer as recomputing."""
    _write(tmp_path, "tools/a.py", 'A = "x/gt_cache_av.pt"\nB = "caches/gt_n600.npz"\n')
    primary = _check_351_gt_lineage_objective_custody(tmp_path)
    assert gt_lineage_npz_widening_findings(tmp_path, primary) == (
        gt_lineage_npz_widening_findings(tmp_path)
    )


def test_widening_never_enters_the_return_value(tmp_path):
    _write(tmp_path, "tools/a.py", 'B = "caches/gt_n600.npz"\n')
    returned = check_gt_lineage_objective_custody(
        tmp_path, strict=False, verbose=False, report_npz_widening=True
    )
    assert returned == [], "the .npz widening is report-only; it must not be returned"


def test_widening_never_raises_even_in_strict_mode(tmp_path):
    """An .npz-only tree must pass strict: 362 live findings would wedge the repo."""
    _write(tmp_path, "tools/a.py", 'B = "caches/gt_n600.npz"\n')
    assert check_gt_lineage_objective_custody(tmp_path, strict=True, verbose=False) == []


def test_strict_still_raises_on_a_primary_finding(tmp_path):
    """The positive control: the refusal path the widening must not have broken."""
    _write(tmp_path, "tools/a.py", 'A = "x/gt_cache_av.pt"\n')
    with pytest.raises(PreflightError):
        check_gt_lineage_objective_custody(tmp_path, strict=True, verbose=False)


def test_widening_report_is_on_by_default(capsys, tmp_path):
    """Score-neutral observability defaults ON -- a default-off gauge is orphaned."""
    _write(tmp_path, "tools/a.py", 'B = "caches/gt_n600.npz"\n')
    check_gt_lineage_objective_custody(tmp_path, strict=False, verbose=True)
    assert ".npz widening" in capsys.readouterr().out


def test_widening_report_can_be_silenced(capsys, tmp_path):
    _write(tmp_path, "tools/a.py", 'B = "caches/gt_n600.npz"\n')
    check_gt_lineage_objective_custody(
        tmp_path, strict=False, verbose=True, report_npz_widening=False
    )
    assert ".npz widening" not in capsys.readouterr().out


def test_scanner_survives_a_syntax_error_under_the_widening(tmp_path):
    _write(tmp_path, "tools/broken.py", 'def f(\nP = "caches/gt_n600.npz"\n')
    _check_351_gt_lineage_objective_custody(tmp_path, WIDENED, _GT_LINEAGE_PREFILTER_STEMS)  # must not raise
