# SPDX-License-Identifier: MIT
"""Tests for tac.verdicts.measurement_row — the canonical typed measurement row.

Covers schema validation (positive + negative) for MeasurementRow / Provenance /
AxisTag / ReviewStatus, with a focus on the design-philosophy invariants:
  * P2 — a non-None noise_floor MUST carry provenance (no silent-zero floor).
  * n600 — a subset (n_samples != 600) MUST state its reason.
  * axis authority — contest-CPU/CUDA are authority; others are not.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from tac.verdicts import (
    AxisTag,
    MeasurementRow,
    MeasurementRowError,
    Provenance,
    ReviewStatus,
)


def _prov() -> Provenance:
    return Provenance(git_sha="deadbeef", tool="tac.verdicts.tests", seed=0)


def _row(**over) -> MeasurementRow:
    kw = {
        "value": 0.0031,
        "units": "d_seg",
        "axis_tag": AxisTag.THROUGH_R,
        "provenance": _prov(),
        "n_samples": 600,
        "review_status": ReviewStatus.REVIEWED,
    }
    kw.update(over)
    return MeasurementRow(**kw)


# ------------------------------- AxisTag ---------------------------------------
def test_axis_tag_authority_only_contest():
    assert AxisTag.CONTEST_CPU.is_authority
    assert AxisTag.CONTEST_CUDA.is_authority
    assert not AxisTag.THROUGH_R.is_authority
    assert not AxisTag.MACOS_MLX_RESEARCH_SIGNAL.is_authority
    assert not AxisTag.PROXY.is_authority


def test_axis_tag_coerce_from_exact_bracketed_string():
    assert AxisTag.coerce("[contest-CPU]") is AxisTag.CONTEST_CPU
    assert AxisTag.coerce("[macOS-MLX research-signal]") is AxisTag.MACOS_MLX_RESEARCH_SIGNAL
    assert AxisTag.coerce(AxisTag.MASK_LEVEL) is AxisTag.MASK_LEVEL


def test_axis_tag_coerce_rejects_unknown():
    with pytest.raises(MeasurementRowError):
        AxisTag.coerce("[contest-mps]")
    with pytest.raises(MeasurementRowError):
        AxisTag.coerce("contest-CPU")  # missing brackets


def test_axis_tag_value_is_bracketed_canonical_string():
    # StrEnum: the .value must be the exact canonical axis label used repo-wide.
    assert AxisTag.CONTEST_CPU.value == "[contest-CPU]"
    assert AxisTag.THROUGH_R.value == "[through-R]"


# ------------------------------- ReviewStatus ----------------------------------
def test_review_status_load_bearing_only_reviewed():
    assert ReviewStatus.REVIEWED.is_load_bearing
    assert not ReviewStatus.UNREVIEWED_RECOVERY_WRITTEN.is_load_bearing
    assert not ReviewStatus.PROVISIONAL.is_load_bearing


def test_review_status_coerce_and_reject():
    assert ReviewStatus.coerce("provisional") is ReviewStatus.PROVISIONAL
    with pytest.raises(MeasurementRowError):
        ReviewStatus.coerce("rubber-stamped")


# ------------------------------- Provenance ------------------------------------
def test_provenance_requires_git_sha_and_tool():
    with pytest.raises(MeasurementRowError):
        Provenance(git_sha="", tool="t")
    with pytest.raises(MeasurementRowError):
        Provenance(git_sha="abc", tool="   ")


def test_provenance_seed_must_be_int_not_bool():
    with pytest.raises(MeasurementRowError):
        Provenance(git_sha="abc", tool="t", seed=True)
    # a real int seed is fine
    assert Provenance(git_sha="abc", tool="t", seed=7).seed == 7


def test_provenance_inputs_sha256_shape_validated():
    good = "a" * 64
    assert Provenance(git_sha="abc", tool="t", inputs_sha256=good).inputs_sha256 == good
    with pytest.raises(MeasurementRowError):
        Provenance(git_sha="abc", tool="t", inputs_sha256="a" * 63)
    with pytest.raises(MeasurementRowError):
        Provenance(git_sha="abc", tool="t", inputs_sha256="z" * 64)  # non-hex


# ------------------------------- MeasurementRow: value -------------------------
def test_row_value_must_be_real_finite_not_bool():
    with pytest.raises(MeasurementRowError):
        _row(value=True)
    with pytest.raises(MeasurementRowError):
        _row(value=math.inf)
    with pytest.raises(MeasurementRowError):
        _row(value=math.nan)


def test_row_units_must_be_nonempty():
    with pytest.raises(MeasurementRowError):
        _row(units="")


def test_row_accepts_string_axis_and_review_status_coerced_to_enum():
    r = _row(axis_tag="[contest-CUDA]", review_status="provisional")
    assert r.axis_tag is AxisTag.CONTEST_CUDA
    assert r.review_status is ReviewStatus.PROVISIONAL


def test_row_provenance_must_be_provenance_instance():
    with pytest.raises(MeasurementRowError):
        _row(provenance={"git_sha": "abc", "tool": "t"})


# ------------------------------- MeasurementRow: n_samples --------------------
def test_row_n600_needs_no_reason():
    r = _row(n_samples=600)
    assert r.is_n600
    assert r.n_samples_reason is None


def test_row_subset_requires_reason():
    with pytest.raises(MeasurementRowError):
        _row(n_samples=96)  # subset, no reason
    r = _row(n_samples=96, n_samples_reason="n96 gt cache micro-probe")
    assert not r.is_n600
    assert r.n_samples_reason


def test_row_n_samples_must_be_positive_int():
    with pytest.raises(MeasurementRowError):
        _row(n_samples=0)
    with pytest.raises(MeasurementRowError):
        _row(n_samples=True)


# ------------------------------- MeasurementRow: P2 noise floor ----------------
def test_row_floor_none_is_unknown_and_allowed():
    r = _row(noise_floor=None)
    assert not r.floor_is_known
    assert r.to_json_dict()["noise_floor"] is None


def test_row_nonzero_floor_requires_provenance():
    with pytest.raises(MeasurementRowError):
        _row(noise_floor=0.0004)  # no floor_provenance
    r = _row(noise_floor=0.0004, floor_provenance="label-noise floor #141")
    assert r.floor_is_known


def test_row_silent_zero_floor_is_impossible_P2():
    # The core P2 invariant: a 0.0 floor without provenance is REFUSED, so a
    # silent-zero floor cannot be constructed.
    with pytest.raises(MeasurementRowError):
        _row(noise_floor=0.0)
    # 0.0 WITH provenance is legal (an explicitly justified zero).
    r = _row(noise_floor=0.0, floor_provenance="bit-exact byte-close: exact match")
    assert r.floor_is_known


def test_row_negative_floor_rejected():
    with pytest.raises(MeasurementRowError):
        _row(noise_floor=-1e-4, floor_provenance="x")


# ------------------------------- to_json_dict ----------------------------------
def test_to_json_dict_stable_shape_and_derived_flags():
    r = _row(
        quantity="d_seg",
        axis_tag=AxisTag.CONTEST_CPU,
        noise_floor=0.0004,
        floor_provenance="label-noise floor",
    )
    d = r.to_json_dict()
    assert d["quantity"] == "d_seg"
    assert d["axis_tag"] == "[contest-CPU]"
    assert d["is_authority_axis"] is True
    assert d["floor_is_known"] is True
    assert d["is_n600"] is True
    assert d["is_load_bearing"] is True
    assert d["provenance"]["git_sha"] == "deadbeef"
    # JSON-serializable (no enum objects leak through).
    import json
    json.dumps(d)


def test_row_is_frozen():
    r = _row()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.value = 0.9  # type: ignore[misc]
