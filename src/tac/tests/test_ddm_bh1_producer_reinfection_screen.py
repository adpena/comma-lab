# SPDX-License-Identifier: MIT
"""Tests for the ddm_bh1 re-infection screen on the prefix-bias law.

The screen encodes a fact a constant census cannot see: curing every CONSUMER of a retired
prefix constant leaves the class re-infectable while the PRODUCER that measured it still
defaults to the prefix cohort.  These pin the predicate's logic, the anchor's arithmetic, and
the cure that landed in the producer -- so a future edit cannot quietly revert either half.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from tac.canonical_equations.annulus_restricted_prefix_bias_detector_20260904 import (
    DELTA_R_N96,
    DELTA_R_N600,
    POPULATION_N,
    PREFIX_N,
    build_annulus_restricted_prefix_bias_detector_v1,
    producer_default_reinfects_cured_constant,
)

REPO = Path(__file__).resolve().parents[3]
PRODUCER = REPO / "tools" / "measure_delta_R_noise_floor.py"


# --- the predicate -------------------------------------------------------------------------


def test_cured_consumers_with_prefix_producer_is_the_reinfection_state():
    assert producer_default_reinfects_cured_constant(
        consumers_cured=True, producer_default_cohort_is_prefix=True
    )


def test_cured_consumers_with_population_producer_is_clean():
    assert not producer_default_reinfects_cured_constant(
        consumers_cured=True, producer_default_cohort_is_prefix=False
    )


@pytest.mark.parametrize("producer_is_prefix", [True, False])
def test_uncured_consumers_are_not_a_reinfection_state(producer_is_prefix):
    """Uncured consumers are the ORIGINAL disease, not re-infection -- a distinct screen."""

    assert not producer_default_reinfects_cured_constant(
        consumers_cured=False, producer_default_cohort_is_prefix=producer_is_prefix
    )


# --- the anchor ----------------------------------------------------------------------------


def test_law_carries_the_bh1_anchor_beside_dr1():
    equation = build_annulus_restricted_prefix_bias_detector_v1()
    ids = [anchor.anchor_id for anchor in equation.empirical_anchors]
    assert "dr1_delta_r_n600_vs_n96_prefix_annulus_vs_global_20260904" in ids
    assert "bh1_producer_default_still_the_prefix_after_consumer_cure_20260904" in ids


def test_bh1_anchor_falsifier_fired():
    """Predicted 0 producers on the prefix, measured 1 -- residual 1.0, and it must say so."""

    equation = build_annulus_restricted_prefix_bias_detector_v1()
    anchor = next(
        a for a in equation.empirical_anchors if a.anchor_id.startswith("bh1_producer_default")
    )
    assert anchor.predicted_output["producers_still_defaulting_to_the_prefix"] == 0
    assert anchor.empirical_output["producers_still_defaulting_to_the_prefix"] == 1
    assert anchor.empirical_output["reinfection_open"] is True
    assert anchor.residual == 1.0
    assert equation.predicted_vs_empirical_residual[anchor.anchor_id] == 1.0


def test_the_two_delta_r_values_are_the_prefix_and_the_population():
    """The anchor is only meaningful if the retired value really is the biased one."""

    assert DELTA_R_N96 < DELTA_R_N600, "the prefix UNDERSTATED the floor (anti-conservative)"
    assert PREFIX_N == 96 and POPULATION_N == 600


# --- the cure actually landed in the producer -----------------------------------------------


def _producer_defaults() -> dict:
    spec = importlib.util.spec_from_file_location("_delta_r_producer", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _Captured(Exception):
        def __init__(self, parser):
            super().__init__("captured")
            self.parser = parser

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = lambda self, *a, **k: (_ for _ in ()).throw(_Captured(self))
    try:
        module.main([])
    except _Captured as exc:
        return {
            action.option_strings[0]: action.default
            for action in exc.parser._actions
            if action.option_strings
        }
    finally:
        argparse.ArgumentParser.parse_args = original
    raise AssertionError("the producer did not build a parser")


def test_producer_default_is_no_longer_the_prefix():
    defaults = _producer_defaults()
    assert defaults["--n"] == POPULATION_N
    assert defaults["--gt-npz"].endswith(f"gt_n{POPULATION_N}.npz")
    assert producer_default_reinfects_cured_constant(
        consumers_cured=True,
        producer_default_cohort_is_prefix=defaults["--n"] == PREFIX_N,
    ) is False
