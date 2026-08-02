"""ddm_tp2 (2026-08-02) — two LIVE BUGS on the TR1 trainer, each with its own regression.

ROW 1 (this module's first block): ``basin_entry_fires`` compared the DISPLAY LABEL
``row["stage"] == "seg_trunk_tau"`` by exact string. The label varies with the
``--seg-form-start`` launch flag, so a run started directly at ``tau_softplus`` ran the
IDENTICAL loss form under the label ``seg_trunk_tau_softplus`` and could NEVER satisfy the
predicate. The fix keys the predicate on the loss FORM (``state_form["form"]``), which is
"tau_softplus" in BOTH launch paths.

ROW 2 lands in a separate commit and extends this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[3]
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.train_tr1_partition_renderer_mlx import (  # noqa: E402
    BASIN_TERMINAL_SEG_FORM,
    SEG_TRUNK_CE_STAGE,
    basin_entry_fires,
    initial_stage_label,
)

# The four values ``--seg-form-start`` actually accepts (argparse ``choices``).
SEG_FORM_START_CHOICES = ("ce", "tau_softplus", "unify_tau", "margin_hinge")


def _window(*, form: str = BASIN_TERMINAL_SEG_FORM, stage: str = "seg_trunk_tau",
            basis: str = "ema_shadow") -> list[dict]:
    """A 3-gate window that satisfies EVERY numeric basin threshold, so the only thing
    under test is the form/stage/basis key."""
    smooth = [1.0, 0.9975, 0.9950]      # 0.50% over the window  (< 1% threshold)
    dseg = [0.0040, 0.003970, 0.003950]  # 1.25% over the window  (< 2% threshold)
    return [{"epoch": 800 + 10 * i, "basis": basis, "stage": stage, "form": form,
             "dseg": dseg[i], "smooth": smooth[i], "alarm": False,
             "lane_b0": 30, "lane_er": 4} for i in range(3)]


# --------------------------------------------------------------------------------------
# ROW 1 — the basin predicate must key on the loss FORM, never the display label.
# --------------------------------------------------------------------------------------

def test_baseline_window_fires():
    """Positive control: without it, every 'does not fire' assertion below is vacuous."""
    assert basin_entry_fires(_window()) is True


def test_predicate_fires_for_a_run_launched_directly_at_tau_softplus():
    """THE REGRESSION. This window carries the label a ``--seg-form-start tau_softplus``
    launch actually produces. The pre-fix predicate (``stage == "seg_trunk_tau"``) returns
    False here; the form-keyed predicate must return True."""
    label = initial_stage_label("tau_softplus")
    w = _window(stage=label)
    assert label != "seg_trunk_tau"          # the exact-string comparison could never match
    assert basin_entry_fires(w) is True      # ... yet the loss form is terminal, so it fires


def test_predicate_is_invariant_to_every_stage_label():
    """The label must have ZERO influence: same form, any label, same verdict."""
    verdicts = {lbl: basin_entry_fires(_window(stage=lbl))
                for lbl in [initial_stage_label(c) for c in SEG_FORM_START_CHOICES]
                + ["seg_trunk_tau", "basin_entry", "", "anything"]}
    assert set(verdicts.values()) == {True}, verdicts


def test_predicate_refuses_a_non_terminal_form_even_under_the_terminal_label():
    """The dual of the bug: a CE-form window mislabelled "seg_trunk_tau" must NOT fire.
    Pre-fix this window WOULD have fired (label matched) -- a false basin entry."""
    assert basin_entry_fires(_window(form="ce", stage="seg_trunk_tau")) is False


def test_predicate_refuses_when_the_form_key_is_absent():
    """Fail-closed: a row that never recorded its form cannot certify a basin."""
    w = _window()
    for row in w:
        del row["form"]
    assert basin_entry_fires(w) is False


@pytest.mark.parametrize("bad", ["live", "", "shadow"])
def test_non_shadow_basis_still_refuses(bad):
    """The sister key is untouched by this fix."""
    assert basin_entry_fires(_window(basis=bad)) is False


def test_initial_stage_label_only_ce_gets_the_rewritable_label():
    """Documents the mechanism: exactly ONE of the four launch choices produces the label
    the knee events rewrite; the other three are stranded under their own label forever."""
    labels = {c: initial_stage_label(c) for c in SEG_FORM_START_CHOICES}
    assert labels["ce"] == SEG_TRUNK_CE_STAGE
    rewritable = [c for c, lbl in labels.items() if lbl == SEG_TRUNK_CE_STAGE]
    assert rewritable == ["ce"]
    assert labels["tau_softplus"] == "seg_trunk_tau_softplus"
    assert len(set(labels.values())) == len(SEG_FORM_START_CHOICES)  # labels are distinct


def test_stage_label_derivation_has_a_single_owner():
    """Anti-drift: the f-string label convention must not be re-inlined anywhere in the
    trainer, or the predicate can silently diverge from the label again."""
    src = (WORKTREE / "experiments" / "train_tr1_partition_renderer_mlx.py").read_text()
    assert src.count('f"seg_trunk_{') == 1, (
        "the stage-label convention must live ONLY in initial_stage_label()")
