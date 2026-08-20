# SPDX-License-Identifier: MIT
"""Tests for the ddm_dg1 cross-lineage DELTA guard and the #351 vacuity fix.

TWO defects are covered here, both MEASURED on 2026-08-20.

1.  **The delta guard** (:func:`tac.local_contest_instruments.assert_comparable_legs` /
    :func:`~tac.local_contest_instruments.receipt_delta`).  The 2026-08-19 ``jg4``
    refusal subtracted a PyAV-lineage advisory seg reading from a DALI-lineage T4
    base and called a working candidate net-negative.  Neither existing guard could
    see it: ``gt_lineage.assert_gt_lineage`` keys on a GT FILE and the T4 leg was a
    number from a contest report, and ``assert_single_lineage`` checks the span
    WITHIN one instrument, not ACROSS two separately-measured legs.

2.  **The #351 detector vacuity** (``preflight._check_351_gt_lineage_objective_custody``).
    The ripgrep prefilter returns ABSOLUTE paths; with a relative ``repo_root`` every
    ``relative_to`` raised ``ValueError`` into a bare ``continue``, so the gate
    reported 0 findings while 11 stood.  ``VACUITY==PASS``: a skip counted as a pass.

The jg4 numbers below are POSITIVE CONTROLS executed against the guard, not
re-assertions of a docstring: the same-lineage delta must reproduce jg4's published
-1.090e-4, so a guard that silently passed everything would still fail this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac import gt_lineage, preflight
from tac import local_contest_instruments as lci

# --- the jg4 incident, verbatim ---------------------------------------------
# `.omx/research` + memory `advisory_gate_cross_instrument_false_refusal_20260819`.
JG4_CANDIDATE_ADVISORY_SEG = 0.0003244  # PyAV lineage
JG4_BASE_T4_SEG = 0.00030309  # DALI lineage (the contest row)
JG4_BASE_ADVISORY_SEG = 0.00043336  # the SAME base, read on the candidate's lineage
JG4_TRUE_SAME_INSTRUMENT_DELTA = -1.0896e-04  # candidate IMPROVED


def _receipt(instrument, axis, lineage, *, pairs=600, d_seg=None, d_pose=None, archive_bytes=None):
    return lci.InstrumentReceipt(
        instrument=instrument,
        axis=axis,
        gt_lineage=lineage,
        pairs=pairs,
        sampling="full-field",
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=archive_bytes,
    )


@pytest.fixture
def jg4_candidate():
    return _receipt(
        "jg4-candidate",
        lci.AXIS_MACOS_CPU_ADVISORY,
        gt_lineage.PYAV_YUV420_TO_RGB,
        d_seg=JG4_CANDIDATE_ADVISORY_SEG,
    )


@pytest.fixture
def jg4_base_t4():
    return _receipt(
        "t4-base", lci.AXIS_CONTEST_CUDA, gt_lineage.DALI_NVDEC, d_seg=JG4_BASE_T4_SEG
    )


@pytest.fixture
def jg4_base_advisory():
    return _receipt(
        "base-advisory",
        lci.AXIS_MACOS_CPU_ADVISORY,
        gt_lineage.PYAV_YUV420_TO_RGB,
        d_seg=JG4_BASE_ADVISORY_SEG,
    )


# --- 1. the refusal ----------------------------------------------------------


def test_jg4_cross_lineage_subtraction_is_refused(jg4_candidate, jg4_base_t4):
    """The exact subtraction that nearly killed a candidate projecting S ~ 0.1467."""
    with pytest.raises(lci.CrossLineageDelta):
        lci.receipt_delta(jg4_candidate, jg4_base_t4, quantity="d_seg")


def test_refusal_names_both_lineages_and_the_fork_cost(jg4_candidate, jg4_base_t4):
    """A refusal that does not teach the mechanism just gets waived away."""
    with pytest.raises(lci.CrossLineageDelta) as excinfo:
        lci.receipt_delta(jg4_candidate, jg4_base_t4, quantity="d_seg")
    message = str(excinfo.value)
    assert gt_lineage.PYAV_YUV420_TO_RGB in message
    assert gt_lineage.DALI_NVDEC in message
    assert str(lci.ADVISORY_SEG_MULTIPLICATIVE_FACTOR) in message


def test_pose_refusal_quotes_the_additive_floor_not_a_ratio():
    """Pose forks ADDITIVELY; quoting a multiplier is the refuted form (ddm_na10)."""
    cand = _receipt(
        "c", lci.AXIS_MACOS_CPU_ADVISORY, gt_lineage.PYAV_YUV420_TO_RGB, d_pose=1.48e-04
    )
    base = _receipt("b", lci.AXIS_CONTEST_CUDA, gt_lineage.DALI_NVDEC, d_pose=7.77e-06)
    with pytest.raises(lci.CrossLineageDelta) as excinfo:
        lci.receipt_delta(cand, base, quantity="d_pose")
    assert "ADDITIVELY" in str(excinfo.value)


# --- 2. the positive control: the CORRECT comparison still works --------------


def test_same_lineage_delta_reproduces_the_published_jg4_value(
    jg4_candidate, jg4_base_advisory
):
    """EXECUTED positive control: same-instrument, the candidate IMPROVED.

    A guard that refused everything would be useless; this pins the number.
    """
    delta = lci.receipt_delta(jg4_candidate, jg4_base_advisory, quantity="d_seg")
    assert delta == pytest.approx(JG4_TRUE_SAME_INSTRUMENT_DELTA, abs=2e-7)
    assert delta < 0, "the candidate improved seg on its own instrument"


def test_base_advisory_over_t4_reproduces_the_measured_seg_fork():
    """The base's two readings differ by the seg fork alone -- 1.430x (jg4)."""
    assert pytest.approx(1.430, abs=5e-3) == JG4_BASE_ADVISORY_SEG / JG4_BASE_T4_SEG


# --- 3. the other comparability legs -----------------------------------------


def test_population_mismatch_is_refused(jg4_candidate):
    """600-vs-96 is a different population; prefix bias can invert the sign."""
    prefix_base = _receipt(
        "b96",
        lci.AXIS_MACOS_CPU_ADVISORY,
        gt_lineage.PYAV_YUV420_TO_RGB,
        pairs=96,
        d_seg=JG4_BASE_ADVISORY_SEG,
    )
    with pytest.raises(lci.CrossLineageDelta):
        lci.receipt_delta(jg4_candidate, prefix_base, quantity="d_seg")


def test_same_lineage_different_axis_is_refused():
    """contest-CPU and macOS-advisory share PyAV, but are not one instrument."""
    cand = _receipt(
        "c", lci.AXIS_MACOS_CPU_ADVISORY, gt_lineage.PYAV_YUV420_TO_RGB, d_seg=3e-4
    )
    base = _receipt("b", lci.AXIS_CONTEST_CPU, gt_lineage.PYAV_YUV420_TO_RGB, d_seg=4e-4)
    with pytest.raises(lci.CrossLineageDelta):
        lci.receipt_delta(cand, base, quantity="d_seg")


def test_missing_leg_is_not_a_zero_delta(jg4_candidate):
    """A None leg must refuse, never silently read as 0.0."""
    base = _receipt("b", lci.AXIS_MACOS_CPU_ADVISORY, gt_lineage.PYAV_YUV420_TO_RGB)
    with pytest.raises(lci.InstrumentRefusal):
        lci.receipt_delta(jg4_candidate, base, quantity="d_seg")


def test_unknown_quantity_is_refused(jg4_candidate, jg4_base_advisory):
    with pytest.raises(lci.InstrumentRefusal):
        lci.receipt_delta(jg4_candidate, jg4_base_advisory, quantity="d_everything")


# --- 4. the waiver: usable for real diagnostics, not for laziness ------------


def test_substantive_rationale_permits_a_deliberate_fork_measurement(
    jg4_candidate, jg4_base_t4
):
    """Measuring the fork ITSELF is legitimate -- and must be nameable."""
    delta = lci.receipt_delta(
        jg4_candidate,
        jg4_base_t4,
        quantity="d_seg",
        allow_cross_lineage_rationale=(
            "measuring the PyAV-vs-DALI seg fork itself, per ddm_pi2 section 0.3"
        ),
    )
    assert delta == pytest.approx(JG4_CANDIDATE_ADVISORY_SEG - JG4_BASE_T4_SEG, abs=1e-12)


@pytest.mark.parametrize("placeholder", ["ok", "tbd", "<reason>", "  ", "n/a", "todo"])
def test_placeholder_rationales_are_rejected(jg4_candidate, jg4_base_t4, placeholder):
    with pytest.raises(lci.InstrumentRefusal):
        lci.receipt_delta(
            jg4_candidate,
            jg4_base_t4,
            quantity="d_seg",
            allow_cross_lineage_rationale=placeholder,
        )


def test_cross_lineage_waiver_does_not_also_waive_population(jg4_candidate):
    """The waiver is scoped to LINEAGE only.

    One flag that switches off three unrelated refusals is the over-broad-waiver
    shape.  "I meant to cross lineages" is not a reason to also cross populations.
    """
    small_base = _receipt(
        "b96", lci.AXIS_CONTEST_CUDA, gt_lineage.DALI_NVDEC, pairs=96, d_seg=JG4_BASE_T4_SEG
    )
    with pytest.raises(lci.CrossLineageDelta) as excinfo:
        lci.receipt_delta(
            jg4_candidate,
            small_base,
            quantity="d_seg",
            allow_cross_lineage_rationale=(
                "measuring the PyAV-vs-DALI seg fork itself, per ddm_pi2 section 0.3"
            ),
        )
    assert "populations" in str(excinfo.value)


def test_unknown_axis_is_refused():
    with pytest.raises(lci.InstrumentRefusal):
        lci.assert_comparable_legs(
            candidate_axis="contest-TPU",
            candidate_lineage=gt_lineage.DALI_NVDEC,
            base_axis=lci.AXIS_CONTEST_CUDA,
            base_lineage=gt_lineage.DALI_NVDEC,
        )


def test_matched_legs_pass_cleanly():
    """The guard must be silent when the legs genuinely match."""
    lci.assert_comparable_legs(
        candidate_axis=lci.AXIS_CONTEST_CUDA,
        candidate_lineage=gt_lineage.DALI_NVDEC,
        base_axis=lci.AXIS_CONTEST_CUDA,
        base_lineage=gt_lineage.DALI_NVDEC,
        candidate_pairs=600,
        base_pairs=600,
    )


# --- 5. the #351 detector vacuity fix ----------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_detector_is_root_form_invariant():
    """A relative root must give the SAME answer as an absolute root.

    Before the fix: relative -> 0 findings, absolute -> 11.  The gate silently
    passed because ``relative_to`` raised into a bare ``continue``.
    """
    absolute = preflight._check_351_gt_lineage_objective_custody(REPO_ROOT)
    relative = preflight._check_351_gt_lineage_objective_custody(Path("."))
    assert relative == absolute


def test_detector_denominator_is_non_zero():
    """The detector must actually REACH its line scan -- the denominator check.

    A gate that returns ``[]`` because nothing ever entered its inner loop is
    indistinguishable, at the call site, from a clean repo.  That is exactly how
    this defect hid.  Asserting on the FINDING COUNT would be the wrong test: it
    would start failing the day the repo is genuinely cured, which is success.
    So assert the DENOMINATOR instead -- candidates exist AND are resolvable
    against the root, which is precisely the invariant that broke.
    """
    candidates = preflight._rg_python_files_matching_regex(
        REPO_ROOT, list(preflight._GT_LINEAGE_SCAN_DIRS), r"gt_first6|gt_cache_"
    )
    if candidates is None:  # ripgrep unavailable; the pure-Python walker is used
        candidates = tuple(
            preflight._iter_python_files(REPO_ROOT, list(preflight._GT_LINEAGE_SCAN_DIRS))
        )
    assert candidates, "prefilter matched no files; the detector cannot see anything"

    root = REPO_ROOT.resolve()
    resolvable = sum(
        1 for p in candidates if str(p.resolve()).startswith(str(root))
    )
    assert resolvable > 0, (
        "no prefilter candidate resolves under the repo root, so every "
        "relative_to() would raise into the bare `continue` -- the vacuity shape"
    )


def test_detector_findings_are_well_formed():
    """Whatever the detector reports must be actionable, not bare paths."""
    for finding in preflight._check_351_gt_lineage_objective_custody(REPO_ROOT):
        assert "UNDECLARED decode lineage" in finding
        assert ":" in finding, "a finding must carry file:line"
