"""#334: the first-class DSL Curriculum object is the schedule's VERIFIED source-of-record.

``verify_schedule_consistency`` binds the DSL ``Curriculum`` object to the autoconfig's flat schedule
flags: they MUST agree for the curriculum-owned subset. This closes the config-orphan confound where
the schedule could be set in two places that silently drift. These tests prove (a) the sealed config
passes, and (b) a deliberately-drifted config is CAUGHT."""
from __future__ import annotations

from dataclasses import replace

import pytest

from tac import witness_autoconfig as wac
from tac.witness_dsl.curriculum_dsl import (
    CURRICULUM_OWNED_FLAGS,
    verify_schedule_consistency,
)

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _sealed():
    return wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)


def test_sealed_config_schedule_is_consistent():
    """The sealed #205 config: the DSL Curriculum object's schedule flags EQUAL the autoconfig's."""
    assert verify_schedule_consistency(_sealed(), handoff="fixed") == []


class _FieldEmitDivergentCfg:
    """A stub whose schedule FIELDS disagree with its EMITTED flags — the config-orphan the gate
    exists to catch. The DSL Curriculum object reads the stage epoch from the FIELD
    (``tau_softplus_start_epoch``); the autoconfig-side flags come from ``to_trainer_flags``. Here the
    field says tau@300 but the emitted flag says tau@999 → the gate must flag the divergence."""

    def __init__(self, base):
        self._base = base
        self.tau_softplus_start_epoch = 300          # the FIELD the object reads for the tau stage
        self.l7_start_epoch = base.l7_start_epoch
        self.muon_start_epoch = base.muon_start_epoch

    def to_trainer_flags(self, out_dir):
        flags = list(self._base.to_trainer_flags(out_dir))
        # force the EMITTED tau-start to diverge from the FIELD (the orphan condition)
        return [("--tau-softplus-start-epoch", 999) if f == "--tau-softplus-start-epoch" else (f, v)
                for f, v in flags]


def test_field_emission_divergence_is_caught():
    """When the config's schedule FIELD disagrees with its EMITTED flag, the gate FAILS — proving the
    binding is real (catches the config-orphan where the schedule is set in two places that drift)."""
    problems = verify_schedule_consistency(_FieldEmitDivergentCfg(_sealed()), handoff="fixed")
    assert problems, "the gate must CATCH a field-vs-emission divergence"
    assert any("--tau-softplus-start-epoch" in p for p in problems)


def test_curriculum_owned_flags_are_the_schedule_subset():
    """The owned-flags set covers the master curriculum flag + the stage epochs + temp/hosc/tau +
    the stage-transition reheat — the schedule the object is the SoR for."""
    assert "--curriculum" in CURRICULUM_OWNED_FLAGS
    assert "--tau-softplus-start-epoch" in CURRICULUM_OWNED_FLAGS
    assert "--hosc-beta-anneal" in CURRICULUM_OWNED_FLAGS
    assert "--muon-start-epoch" in CURRICULUM_OWNED_FLAGS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
