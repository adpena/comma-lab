# SPDX-License-Identifier: MIT
"""Tests for Catalog #296 ``check_substrate_predicted_band_has_dykstra_feasibility_check``.

Per ``.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md``
+ ``.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md``
+ commit ``4292c8ce2`` (symposium) + commit ``b0a7ff474`` (META-assumption audit).

The gate refuses substrate design memos at ``.omx/research/*_design_<YYYYMMDD>.md``
containing the literal section header ``## Predicted ΔS band`` (or sister)
that lack ONE of: (a) Dykstra-feasibility token, (b) first-principles
citation, (c) sister probe-disambiguator path, (d) same-line waiver
``# PREDICTED_BAND_VIBES_OK:<rationale>``.

Empirical anchor: NSCS06 v6 dispatch landed 105.15 vs predicted [0.10, 0.20]
(553x OUTSIDE band) because 5-move composition was assumed additive under
contest polytope constraints WITHOUT a Dykstra-feasibility intersection check.

Sister of Catalog #290 (canonical-vs-unique decision per layer), Catalog #229
(premise verification), Catalog #292 (per-deliberation assumption surfacing),
Catalog #294 (9-dim checklist).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_predicted_band_has_dykstra_feasibility_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _research_dir(repo_root: Path) -> Path:
    d = repo_root / ".omx" / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_memo(repo_root: Path, name: str, body: str) -> Path:
    d = _research_dir(repo_root)
    path = d / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_296_live_repo_count_bounded() -> None:
    """Live count should be ≤ 5 per audit prediction."""
    violations = check_substrate_predicted_band_has_dykstra_feasibility_check(
        strict=False, verbose=False,
    )
    assert len(violations) <= 5, (
        f"Live-repo Catalog #296 violation count exceeded predicted ceiling "
        f"of 5. Found {len(violations)}: {violations[:3]}"
    )


# ---------------------------------------------------------------------------
# Out-of-scope behavior
# ---------------------------------------------------------------------------


def test_296_no_research_dir_returns_empty(tmp_path: Path) -> None:
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_empty_research_dir_returns_empty(tmp_path: Path) -> None:
    _research_dir(tmp_path)
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_memo_without_predicted_band_out_of_scope(tmp_path: Path) -> None:
    """A design memo without the trigger section header is out-of-scope."""
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo Design\n\nNo predicted band section in this memo.\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_non_design_memo_not_scanned(tmp_path: Path) -> None:
    """Files not matching ``*_design_<YYYYMMDD>.md`` pattern are not scanned."""
    _write_memo(
        tmp_path,
        "audit_20260516.md",
        "## Predicted ΔS band\n\n[0.10, 0.20]\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Positive: trigger present without acceptance token = flagged
# ---------------------------------------------------------------------------


def test_296_predicted_band_without_dykstra_flagged(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "nscs_test_design_20260516.md",
        "# NSCS Test Design\n\n## Predicted ΔS band\n\n[0.10, 0.20] derived from "
        "ad-hoc back-of-envelope math.\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1
    assert "Predicted ΔS band" in v[0] or "predicted" in v[0].lower()


def test_296_delta_s_band_variant_flagged(tmp_path: Path) -> None:
    """Sister variant ``## Predicted delta S band`` also triggers."""
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted delta S band\n\n[0.05, 0.10] from heuristic.\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_296_predicted_score_band_variant_flagged(tmp_path: Path) -> None:
    """Sister variant ``## Predicted score band`` also triggers."""
    _write_memo(
        tmp_path,
        "bar_design_20260516.md",
        "# Bar\n\n## Predicted score band\n\n[0.18, 0.22] from intuition.\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Negative: acceptance cascade satisfied
# ---------------------------------------------------------------------------


def test_296_dykstra_token_accepts(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band\n\n[0.10, 0.20] -- pending Dykstra "
        "feasibility intersection check at next probe.\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_convex_feasibility_token_accepts(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band\n\nIntersection of constraints "
        "computed via convex feasibility analysis: [0.05, 0.15].\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_shannon_citation_accepts(tmp_path: Path) -> None:
    """First-principles Shannon R(D) citation accepts the predicted band."""
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted score band\n\nDerived from Shannon R(D) "
        "bound applied to the contest's per-frame entropy.\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_mdl_citation_accepts(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band\n\nMDL (minimum description length) "
        "lower bound argument: [0.18, 0.22].\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_probe_disambiguator_path_accepts(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band\n\n[0.10, 0.20] -- empirical "
        "disambiguator probe at tools/probe_foo_disambiguator.py will "
        "compute the intersection.\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_296_same_line_waiver_accepts(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band <!-- PREDICTED_BAND_VIBES_OK:"
        "scaffold-only memo; Dykstra check deferred to v2 design per "
        "operator approval -->\n\n[0.10, 0.20]\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """``<rationale>`` literal MUST NOT self-waive."""
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band <!-- PREDICTED_BAND_VIBES_OK:"
        "<rationale> -->\n\n[0.10, 0.20]\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_296_placeholder_reason_rejected(tmp_path: Path) -> None:
    """``<reason>`` literal MUST NOT self-waive."""
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band <!-- PREDICTED_BAND_VIBES_OK:"
        "<reason> -->\n\n[0.10, 0.20]\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_296_waiver_without_rationale_rejected(tmp_path: Path) -> None:
    """Bare ``PREDICTED_BAND_VIBES_OK:`` without text MUST NOT self-waive."""
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band <!-- PREDICTED_BAND_VIBES_OK: -->\n\n"
        "[0.10, 0.20]\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_296_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band\n\n[0.10, 0.20]\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_substrate_predicted_band_has_dykstra_feasibility_check(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "Catalog #296" in str(exc_info.value)
    assert "Dykstra" in str(exc_info.value) or "feasibility" in str(exc_info.value)


def test_296_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "# Foo\n\n## Predicted ΔS band\n\nShannon R(D) bound: [0.10, 0.20]\n",
    )
    # Should not raise.
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Aggregation + edge cases
# ---------------------------------------------------------------------------


def test_296_aggregates_multiple_violations(tmp_path: Path) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "## Predicted ΔS band\n\n[0.10, 0.20]\n",
    )
    _write_memo(
        tmp_path,
        "bar_design_20260516.md",
        "## Predicted score band\n\n[0.05, 0.15]\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 2


def test_296_string_repo_root_accepted(tmp_path: Path) -> None:
    """``repo_root`` may be a string per the canonical helper convention."""
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "## Predicted ΔS band\n\nShannon R(D) bound: [0.10, 0.20]\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=str(tmp_path), strict=False, verbose=False,
    )
    assert v == []


def test_296_verbose_clean_output(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "## Predicted ΔS band\n\nShannon R(D): [0.10, 0.20]\n",
    )
    check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=True,
    )
    captured = capsys.readouterr()
    assert "check_substrate_predicted_band_has_dykstra_feasibility_check" in captured.out
    assert "OK" in captured.out


def test_296_verbose_dirty_output(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _write_memo(
        tmp_path,
        "foo_design_20260516.md",
        "## Predicted ΔS band\n\n[0.10, 0.20]\n",
    )
    check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=True,
    )
    captured = capsys.readouterr()
    assert "1 violation" in captured.out


def test_296_invalid_date_suffix_skipped(tmp_path: Path) -> None:
    """``foo_design_invalid.md`` does not match the YYYYMMDD pattern."""
    _write_memo(
        tmp_path,
        "foo_design_invalid.md",
        "## Predicted ΔS band\n\n[0.10, 0.20]\n",
    )
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_296_unicode_decode_error_skipped(tmp_path: Path) -> None:
    """Binary file with non-UTF-8 bytes is silently skipped."""
    d = _research_dir(tmp_path)
    (d / "bad_design_20260516.md").write_bytes(b"\xff\xfe\x00\x00not utf-8")
    v = check_substrate_predicted_band_has_dykstra_feasibility_check(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []
