# SPDX-License-Identifier: MIT
"""Tests for the AdamC corrected-weight-decay unit (2026-07-15).

Covers: the trainer's pure helper math + argparse flag (default OFF), the DSL
``CorrectedWeightDecay`` lever (flag exists on the REAL trainer parser — never-invent-flags),
the Muon adaptivization ticket, and the canonical equation module.
Memo: .omx/research/adamc_muonc_optimizer_research_20260715.md
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _trainer_source() -> str:
    return _TRAINER.read_text()


# ---------------------------------------------------------------------------
# Pure helper math (mirror of the trainer's _corrected_weight_decay; the trainer
# module is too heavy to import in unit tests, so assert the source + mirror the math)
# ---------------------------------------------------------------------------
def _mirror(base_wd: float, lr_current: float, lr_max: float) -> float:
    if base_wd <= 0.0 or lr_max <= 0.0:
        return float(base_wd)
    return float(base_wd) * (float(lr_current) / float(lr_max))


def test_trainer_has_helper_and_flag_and_apply_site() -> None:
    src = _trainer_source()
    assert "def _corrected_weight_decay(" in src
    assert '"--weight-decay-corrected"' in src
    # the apply site follows the trunk lr assignment and is gated on the flag
    assert "opt.weight_decay = _corrected_weight_decay(" in src
    assert 'getattr(args, "weight_decay_corrected", False)' in src
    # default OFF (byte-identical when unset)
    assert "action=argparse.BooleanOptionalAction, default=False" in src.split(
        '"--weight-decay-corrected"'
    )[1][:200]


def test_corrected_wd_identity_at_peak_lr() -> None:
    # at gamma_t == gamma_max the corrected wd equals the base wd (paper: gamma_max
    # keeps the scale consistent with the uncorrected case)
    assert _mirror(1e-4, 1e-3, 1e-3) == pytest.approx(1e-4)


def test_corrected_wd_scales_linearly_with_lr() -> None:
    # cosine floor lr-end=1e-4 at lr-max 1e-3 => lambda_hat = lambda/10
    assert _mirror(1e-4, 1e-4, 1e-3) == pytest.approx(1e-5)
    assert _mirror(1e-4, 5e-4, 1e-3) == pytest.approx(5e-5)


def test_corrected_wd_degenerate_inputs_pass_through() -> None:
    assert _mirror(0.0, 1e-4, 1e-3) == 0.0
    assert _mirror(-1.0, 1e-4, 1e-3) == -1.0
    assert _mirror(1e-4, 1e-4, 0.0) == pytest.approx(1e-4)


def test_trainer_helper_matches_mirror() -> None:
    # execute JUST the helper function body extracted from the trainer source so the
    # unit under test is the real code, not only the mirror
    src = _trainer_source()
    start = src.index("def _corrected_weight_decay(")
    end = src.index("\ndef ", start + 1)
    ns: dict = {}
    exec(compile(src[start:end], str(_TRAINER), "exec"), ns)  # noqa: S102 - own source
    fn = ns["_corrected_weight_decay"]
    for args in ((1e-4, 1e-3, 1e-3), (1e-4, 1e-4, 1e-3), (0.0, 1.0, 1.0), (5e-2, 2e-4, 1e-3)):
        assert fn(*args) == pytest.approx(_mirror(*args))


# ---------------------------------------------------------------------------
# DSL lever (never-invent-flags: the emitted flag must exist on the REAL parser)
# ---------------------------------------------------------------------------
def test_corrected_weight_decay_lever_emits_real_flag() -> None:
    from tac.witness_dsl.curriculum_dsl import CorrectedWeightDecay, real_trainer_flags

    lever = CorrectedWeightDecay()
    assert lever.name == "corrected_weight_decay_adamc"
    assert lever.overrides == {"--weight-decay-corrected": True}
    assert "--weight-decay-corrected" in real_trainer_flags()


def test_corrected_weight_decay_lever_notes_carry_null_prediction() -> None:
    from tac.witness_dsl.curriculum_dsl import CorrectedWeightDecay

    lever = CorrectedWeightDecay()
    assert "PREDICTED-NULL" in lever.notes
    assert "2506.02285" in lever.notes


# ---------------------------------------------------------------------------
# Muon ticket (the honest "MuonC": decoupled+corrected is OWED, never a scaled no-op)
# ---------------------------------------------------------------------------
def test_muon_weight_decay_ticket_registered() -> None:
    from tac.witness_dsl.adaptivization_tickets_20260715 import (
        ADAPTIVIZATION_TICKETS,
        AdaptivizationTicketQueue,
    )

    tickets = {t.constant: t for t in ADAPTIVIZATION_TICKETS}
    t = tickets["--muon-weight-decay"]
    assert "COUPLED" in t.poison_evidence
    assert "adamc_wd_lr_equilibrium_v1" in t.law_source
    assert "NOT wired" in t.built_implementation  # no fake Muon arming
    # queue invariants still hold with the new ticket (no duplicates, fields present)
    AdaptivizationTicketQueue()


# ---------------------------------------------------------------------------
# Canonical equation module
# ---------------------------------------------------------------------------
def test_equation_builds_and_math_is_exact() -> None:
    from tac.canonical_equations.adamc_wd_lr_equilibrium_20260715 import (
        EQUATION_ID,
        build_adamc_wd_lr_equilibrium_v1,
        corrected_weight_decay,
        mechanism_strength,
        steady_state_grad_to_weight_ratio,
    )

    eq = build_adamc_wd_lr_equilibrium_v1()
    assert eq.equation_id == EQUATION_ID == "adamc_wd_lr_equilibrium_v1"
    assert len(eq.empirical_anchors) == 2
    # Eq. 2: ratio doubles when gamma quarters
    r1 = steady_state_grad_to_weight_ratio(1e-4, 1e-3)
    r2 = steady_state_grad_to_weight_ratio(1e-4, 2.5e-4)
    assert r2 == pytest.approx(2.0 * r1)
    # corrected wd at the floor of our live schedule
    assert corrected_weight_decay(1e-4, 1e-4, 1e-3) == pytest.approx(1e-5)
    # the local predicted-null: n24 window strength ~1.8e-4 << 1
    assert mechanism_strength(1e-4, 1e-3 * 24 * 75) < 1e-3


def test_equation_ratio_rejects_degenerate() -> None:
    from tac.canonical_equations.adamc_wd_lr_equilibrium_20260715 import (
        steady_state_grad_to_weight_ratio,
    )

    with pytest.raises(ValueError):
        steady_state_grad_to_weight_ratio(0.0, 1e-3)
    with pytest.raises(ValueError):
        steady_state_grad_to_weight_ratio(1e-4, 0.0)


def test_trainer_source_parses() -> None:
    # cheap structural guard for the hot live-chain file: it must still compile
    import py_compile

    py_compile.compile(str(_TRAINER), doraise=True)
