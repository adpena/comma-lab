"""ddm_tp2 (2026-08-02) — two LIVE BUGS on the TR1 trainer, each with its own regression.

ROW 1 (this module's first block): ``basin_entry_fires`` compared the DISPLAY LABEL
``row["stage"] == "seg_trunk_tau"`` by exact string. The label varies with the
``--seg-form-start`` launch flag, so a run started directly at ``tau_softplus`` ran the
IDENTICAL loss form under the label ``seg_trunk_tau_softplus`` and could NEVER satisfy the
predicate. The fix keys the predicate on the loss FORM (``state_form["form"]``), which is
"tau_softplus" in BOTH launch paths.

ROW 2 (second block): ``--margin-weighted-loss on`` is threaded into ``make_loss_fn`` for
every seg form, but the ``tau_softplus`` and ``l7_softplus`` branches never read
``apply_mw`` -- the flag is declared-ON and INERT for those forms. Guarded by a STRUCTURAL
test that parses the canonical loss source, so the honoring set cannot drift out of sync
with the branches that actually implement it.
"""

from __future__ import annotations

import ast
import inspect
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
    MARGIN_WEIGHTED_HONORING_SEG_FORMS,
    SEG_TRUNK_CE_STAGE,
    assert_margin_weighted_loss_is_honored,
    basin_entry_fires,
    initial_stage_label,
    reachable_seg_forms,
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


# --------------------------------------------------------------------------------------
# ROW 2 — --margin-weighted-loss must never be silently inert.
# --------------------------------------------------------------------------------------

def _apply_mw_branches_from_source() -> dict[str, bool]:
    """Parse the CANONICAL loss source and return {seg_form: reads `apply_mw`}.

    This is the anti-drift core: the honoring set in the trainer is a hand-written
    constant, and a constant that mirrors code silently rots. Reading the real if/elif
    chain means any future edit that adds or removes an ``apply_mw`` guard fails this
    test until the constant is updated to match.
    """
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn

    tree = ast.parse(inspect.getsource(make_loss_fn))
    inner = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "loss_fn")

    def _reads_apply_mw(body) -> bool:
        return any(isinstance(n, ast.Name) and n.id == "apply_mw"
                   for stmt in body for n in ast.walk(stmt))

    # Locate the `if form == "...": ... elif ... else: # ce` chain assigning seg_l.
    def _is_form_test(node) -> str | None:
        t = node.test
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "form" and isinstance(t.ops[0], ast.Eq)
                and isinstance(t.comparators[0], ast.Constant)):
            return t.comparators[0].value
        return None

    chain = next(n for n in ast.walk(inner)
                 if isinstance(n, ast.If) and _is_form_test(n) is not None)
    out: dict[str, bool] = {}
    node = chain
    while True:
        form = _is_form_test(node)
        assert form is not None, "unexpected non-`form ==` test in the seg-form chain"
        out[form] = _reads_apply_mw(node.body)
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
            continue
        out["ce"] = _reads_apply_mw(node.orelse)  # the terminal `else:  # "ce"` branch
        break
    return out


def test_honoring_set_matches_the_branches_that_actually_read_apply_mw():
    """THE ANTI-DRIFT TEST. The trainer's constant must equal the measured source truth."""
    branches = _apply_mw_branches_from_source()
    assert len(branches) == 5, f"expected 5 seg forms, parsed {sorted(branches)}"
    measured = {f for f, reads in branches.items() if reads}
    assert measured == set(MARGIN_WEIGHTED_HONORING_SEG_FORMS), (
        f"MARGIN_WEIGHTED_HONORING_SEG_FORMS={sorted(MARGIN_WEIGHTED_HONORING_SEG_FORMS)} but the "
        f"loss source honors {sorted(measured)}. Update the constant (and the refusal message).")


def test_the_two_inert_forms_are_exactly_tau_softplus_and_l7_softplus():
    """Documents the MEASURED bug: these two branches ignore the flag entirely."""
    branches = _apply_mw_branches_from_source()
    inert = sorted(f for f, reads in branches.items() if not reads)
    assert inert == ["l7_softplus", "tau_softplus"], inert


def test_reachable_forms_ce_reaches_tau_and_the_others_are_terminal():
    """The knee (with its unconditional F2 midpoint fallback) is the only transition."""
    assert reachable_seg_forms("ce") == frozenset({"ce", "tau_softplus"})
    for terminal in ("tau_softplus", "unify_tau", "margin_hinge"):
        assert reachable_seg_forms(terminal) == frozenset({terminal})


def test_off_never_refuses_for_any_form():
    """Positive control for the negatives below: the gate is silent when the flag is off."""
    for form in SEG_FORM_START_CHOICES:
        assert_margin_weighted_loss_is_honored(form, "off")  # must not raise


@pytest.mark.parametrize("form", ["ce", "tau_softplus"])
def test_on_refuses_for_forms_that_reach_an_inert_branch(form):
    """'ce' must ALSO refuse -- it is honored only until the knee, then dies silently.
    This is the exact configuration the entire b4s burn lineage launched with."""
    with pytest.raises(SystemExit) as exc:
        assert_margin_weighted_loss_is_honored(form, "on")
    msg = str(exc.value)
    assert "tau_softplus" in msg and "INERT" in msg
    assert "drop --margin-weighted-loss" in msg      # the message is actionable


@pytest.mark.parametrize("form", ["unify_tau", "margin_hinge"])
def test_on_is_allowed_for_forms_that_honor_it(form):
    assert_margin_weighted_loss_is_honored(form, "on")  # must not raise


def test_refusal_is_wired_into_the_trainer_before_the_loss_is_built():
    """A gate nobody calls is the very orphan class this row is fixing."""
    src = (WORKTREE / "experiments" / "train_tr1_partition_renderer_mlx.py").read_text()
    call = src.index("assert_margin_weighted_loss_is_honored(cfg.seg_form_start")
    assert call < src.index("loss_fn = make_loss_fn("), "refusal must precede loss construction"
