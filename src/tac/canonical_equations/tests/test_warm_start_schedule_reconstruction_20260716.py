# SPDX-License-Identifier: MIT
"""Isolated tests for warm_start_schedule_reconstruction_20260716 (c2_surgical_warm DERIVED law)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.canonical_equations.warm_start_schedule_reconstruction_20260716 import (
    CONFIG_OF_RECORD,
    CONFIG_OF_RECORD_SHA256,
    EQUATION_ID,
    build_warm_start_schedule_reconstruction_v1,
    eval_warm_start_schedule_reconstruction,
)

REPO_ROOT = Path(__file__).resolve().parents[3].parent


def test_config_of_record_mode() -> None:
    assert eval_warm_start_schedule_reconstruction(
        {"mode": "config_of_record", "config_of_record_value": 300}) == 300
    assert eval_warm_start_schedule_reconstruction(
        {"mode": "config_of_record", "config_of_record_value": 726}) == 726


def test_run_length_exclusion_mode() -> None:
    # l7 never-runs: start == epochs (the trainer's documented exclusion pattern; A3).
    assert eval_warm_start_schedule_reconstruction(
        {"mode": "run_length_exclusion", "run_epochs": 1400}) == 1400


def test_resume_plus_window_mode() -> None:
    # the c2 surgical engage boundary: resume 651 + 49 re-anchor epochs = 700.
    assert eval_warm_start_schedule_reconstruction(
        {"mode": "resume_plus_window", "resume_epoch": 651, "re_anchor_window": 49}) == 700


def test_original_plant_end_mode() -> None:
    assert eval_warm_start_schedule_reconstruction(
        {"mode": "original_plant_end", "original_schedule_epochs": 1000}) == 1000


def test_unknown_mode_refuses() -> None:
    with pytest.raises(Exception):
        eval_warm_start_schedule_reconstruction({"mode": "vibes", "value": 1})


def test_missing_input_refuses() -> None:
    with pytest.raises(KeyError):
        eval_warm_start_schedule_reconstruction({"mode": "config_of_record"})


def test_extra_inputs_ignored() -> None:
    # LawRef gate passes the WHOLE inputs mapping; extras must not change the value.
    assert eval_warm_start_schedule_reconstruction(
        {"mode": "resume_plus_window", "resume_epoch": 651, "re_anchor_window": 49,
         "note": "extra", "source": "literal"}) == 700


def test_equation_builds_and_id() -> None:
    eq = build_warm_start_schedule_reconstruction_v1()
    assert eq.equation_id == EQUATION_ID == "warm_start_schedule_reconstruction_v1"
    assert eq.empirical_anchors  # carries the config-of-record source-inspection anchor
    assert eq.domain_of_validity.get("score_claim") is False


def test_config_of_record_sha_custody_matches_disk() -> None:
    """The SHA the constants rows cite must match the actual launch.sh bytes (real custody)."""
    p = REPO_ROOT / CONFIG_OF_RECORD
    if not p.is_file():
        pytest.skip("config-of-record launch.sh not present on this checkout")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == CONFIG_OF_RECORD_SHA256


def test_evaluator_registered_as_builtin() -> None:
    from tac.canonical_equations.evaluators import (
        has_evaluator,
        populate_lawref_evaluators,
        resolve_equation_value,
    )

    populate_lawref_evaluators()
    assert has_evaluator(EQUATION_ID)
    assert resolve_equation_value(
        EQUATION_ID, {"mode": "config_of_record", "config_of_record_value": 300}) == 300


def test_c2_spec_constants_rows_recompute() -> None:
    """Every c2 DERIVED constants row citing this law must recompute to its emitted value."""
    from tac.canonical_equations.evaluators import populate_lawref_evaluators, resolve_equation_value
    from tac.witness_dsl.spec_c2_surgical_20260716 import (
        compile_c2_surgical_warm_launch_config,
    )

    populate_lawref_evaluators()
    cfg = compile_c2_surgical_warm_launch_config()
    rows = {k: v for k, v in cfg.constants_manifest.items()
            if isinstance(v, dict) and v.get("equation_id") == EQUATION_ID}
    assert len(rows) == 6, sorted(rows)
    for key, row in rows.items():
        inputs = row["inputs"]
        if isinstance(inputs, list):  # SHA-custody list form -> {name: value}
            inputs = {str(r["name"]): r.get("value") for r in inputs}
        assert resolve_equation_value(EQUATION_ID, inputs) == row["value"], key
