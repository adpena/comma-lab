# SPDX-License-Identifier: MIT
"""DSL-leg tests for EmaDecayCalibrated (LawRef-resolved --ema-decay) + EmaDecayFinisher."""
from __future__ import annotations

import pytest

from tac.witness_dsl.curriculum_dsl import EmaDecayCalibrated, EmaDecayFinisher, Lever
from tac.witness_dsl.lever_registry import lever_factories


def test_calibrated_resolves_seed_fraction_mode():
    lv = EmaDecayCalibrated(updates_per_run=749, target_seed_fraction=0.01)
    assert isinstance(lv, Lever) and lv.name == "ema_decay_calibrated"
    assert lv.overrides["--ema-decay"] == pytest.approx(0.01 ** (1 / 749))


def test_calibrated_resolves_warmup_fraction_mode():
    lv = EmaDecayCalibrated(updates_per_run=1400, target_seed_fraction=None, warmup_fraction=0.5)
    assert lv.overrides["--ema-decay"] == pytest.approx(1.0 - 2.0 / (0.5 * 1400))


def test_calibrated_carries_lawref_and_manifest_custody():
    lv = EmaDecayCalibrated(updates_per_run=749, target_seed_fraction=0.01)
    assert "--ema-decay" in lv.lawrefs
    assert lv.lawrefs["--ema-decay"].equation_id == "ema_decay_run_geometry_v1"
    man = lv.constant_manifest["--ema-decay"]
    assert man["equation_id"] == "ema_decay_run_geometry_v1"
    assert man["ladder_class"] == "derived_at_config"
    # constant_refs is the provenance-gate alias of lawrefs — same object mapping.
    assert lv.constant_refs is lv.lawrefs


def test_calibrated_requires_exactly_one_pinned_quantity():
    with pytest.raises(ValueError):
        EmaDecayCalibrated(updates_per_run=100, target_seed_fraction=0.5, warmup_fraction=0.5)
    with pytest.raises(ValueError):
        EmaDecayCalibrated(updates_per_run=100, target_seed_fraction=None, warmup_fraction=None)


def test_calibrated_fail_closed_on_infeasible_geometry():
    # phi*U <= 2 -> d <= 0 -> the LawRef resolve raises (fail-closed, no silent fallback).
    with pytest.raises(Exception):
        EmaDecayCalibrated(updates_per_run=10, target_seed_fraction=None, warmup_fraction=0.2)


def test_finisher_defaults_and_start_epoch():
    lv = EmaDecayFinisher()
    assert lv.name == "ema_decay_finisher"
    assert lv.overrides == {"--ema-decay-finisher": 0.999}
    lv2 = EmaDecayFinisher(decay=0.9995, start_epoch=726)
    assert lv2.overrides["--ema-decay-finisher"] == 0.9995
    assert lv2.overrides["--ema-decay-finisher-start-epoch"] == 726


def test_finisher_notes_run_gated_honesty():
    assert "RUN-GATED" in EmaDecayFinisher().notes


def test_registry_discovers_both_factories():
    facs = lever_factories()
    assert "EmaDecayCalibrated" in facs
    assert "EmaDecayFinisher" in facs
    assert "--ema-decay" in facs["EmaDecayCalibrated"]
    assert facs["EmaDecayFinisher"] == frozenset(
        {"--ema-decay-finisher", "--ema-decay-finisher-start-epoch"})


def test_flags_exist_in_trainer_argparse():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[4]
           / "experiments" / "train_levelset_witness_realized_through_R_mlx.py").read_text()
    for flag in ("--ema-decay", "--ema-decay-finisher", "--ema-decay-finisher-start-epoch"):
        assert f'"{flag}"' in src
