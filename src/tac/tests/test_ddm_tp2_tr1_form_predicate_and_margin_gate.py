"""ddm_tp2 (2026-08-02) — two LIVE BUGS on the TR1 trainer, each with its own regression.

ROW 1 (this module's first block): ``basin_entry_fires`` compared the DISPLAY LABEL
``row["stage"] == "seg_trunk_tau"`` by exact string. The label varies with the
``--seg-form-start`` launch flag, so a run started directly at ``tau_softplus`` ran the
IDENTICAL loss form under the label ``seg_trunk_tau_softplus`` and could NEVER satisfy the
predicate. The fix keys the predicate on the loss FORM (``state_form["form"]``), which is
"tau_softplus" in BOTH launch paths.

ROW 3 (third block): the #274 spike/coherent PRODUCER, ported from the levelset trainer onto
the live vehicle. Cross-validated at n600 against two independent implementations.

ROW 2 (second block): ``--margin-weighted-loss on`` must be consumed by every reachable
seg form that claims to honor it. EN1 wires the previously missing ``tau_softplus`` consumer;
``l7_softplus`` remains intentionally outside this lever because it carries its own hard-pixel
weight. Guarded by a STRUCTURAL test that parses the canonical loss source, so the honoring set
cannot drift out of sync with the branches that actually implement it.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
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
    SEG_COHERENT_MH_LIFT_N600,
    SEG_COHERENT_UPWEIGHT_RACE_START,
    SEG_SPIKE_DOWNWEIGHT_RACE_START,
    SEG_SPIKE_MH_LIFT_N600,
    SPIKE_CODE_COHERENT,
    SPIKE_CODE_SPIKE,
    SPIKE_CODE_STABLE,
    assert_margin_weighted_loss_is_honored,
    assert_spike_scalars_have_their_gate,
    basin_entry_fires,
    build_argparser,
    build_spike_coherent_codes,
    initial_stage_label,
    reachable_seg_forms,
    spike_weight_lut,
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


def test_the_only_inert_form_is_l7_softplus():
    """EN1 regression: tau_softplus now honors the flag; only l7_softplus ignores it."""
    branches = _apply_mw_branches_from_source()
    inert = sorted(f for f, reads in branches.items() if not reads)
    assert inert == ["l7_softplus"], inert


def test_reachable_forms_ce_reaches_tau_and_the_others_are_terminal():
    """The knee (with its unconditional F2 midpoint fallback) is the only transition."""
    assert reachable_seg_forms("ce") == frozenset({"ce", "tau_softplus"})
    for terminal in ("tau_softplus", "unify_tau", "margin_hinge"):
        assert reachable_seg_forms(terminal) == frozenset({terminal})


def test_off_never_refuses_for_any_form():
    """Positive control for the negatives below: the gate is silent when the flag is off."""
    for form in SEG_FORM_START_CHOICES:
        assert_margin_weighted_loss_is_honored(form, "off")  # must not raise


@pytest.mark.parametrize("form", SEG_FORM_START_CHOICES)
def test_on_is_allowed_for_forms_that_honor_it(form):
    assert_margin_weighted_loss_is_honored(form, "on")  # must not raise


def test_on_refuses_for_l7_softplus_only():
    """l7_softplus is not an argparse start choice, but the guard must still fail closed if a
    future transition or direct call reaches it under this lever."""
    with pytest.raises(SystemExit) as exc:
        assert_margin_weighted_loss_is_honored("l7_softplus", "on")
    msg = str(exc.value)
    assert "l7_softplus" in msg and "INERT" in msg
    assert "drop --margin-weighted-loss" in msg      # the message is actionable


def test_tau_softplus_margin_weighted_loss_calls_the_live_margin_weight_consumer():
    """EN1 source proof: the tau branch's apply_mw block must call the same live-margin
    weighting function used by CE/unify/hinge, not merely mention the flag."""
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn

    tree = ast.parse(inspect.getsource(make_loss_fn))
    inner = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "loss_fn")

    def _is_tau_test(node) -> bool:
        t = node.test
        return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "form" and isinstance(t.ops[0], ast.Eq)
                and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value == "tau_softplus")

    tau_branch = next(n for n in ast.walk(inner) if isinstance(n, ast.If) and _is_tau_test(n))
    apply_blocks = [n for n in ast.walk(ast.Module(body=tau_branch.body, type_ignores=[]))
                    if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                    and n.test.id == "apply_mw"]
    assert len(apply_blocks) == 1
    calls = [n.func.id for n in ast.walk(apply_blocks[0])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_live_margin_weight" in calls


def test_refusal_is_wired_into_the_trainer_before_the_loss_is_built():
    """A gate nobody calls is the very orphan class this row is fixing."""
    src = (WORKTREE / "experiments" / "train_tr1_partition_renderer_mlx.py").read_text()
    call = src.index("assert_margin_weighted_loss_is_honored(cfg.seg_form_start")
    assert call < src.index("loss_fn = make_loss_fn("), "refusal must precede loss construction"


# --------------------------------------------------------------------------------------
# ROW 3 — the ported #274 spike/coherent producer.
# --------------------------------------------------------------------------------------

def _toy_lstars() -> np.ndarray:
    """4 pairs x 1 x 4 px. Column semantics at the INTERIOR pairs (1, 2):
      col0 stable everywhere · col1 SPIKE at pair 1 (differs from both neighbours)
      col2 COHERENT at pair 1 (matches prev, differs from next)
      col3 COHERENT at pair 1 (differs from prev, matches next)
    """
    return np.array([[[0, 0, 0, 1]], [[0, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 1, 0]]],
                    dtype=np.int64)


def test_producer_classifies_spike_coherent_and_stable():
    codes = build_spike_coherent_codes(_toy_lstars(), 4)
    assert codes[1].tolist() == [[SPIKE_CODE_STABLE, SPIKE_CODE_SPIKE,
                                  SPIKE_CODE_COHERENT, SPIKE_CODE_COHERENT]]


def test_producer_leaves_endpoints_neutral():
    """Endpoints have ONE neighbour, so they carry no verdict -- this is what makes the
    scope 598 interior pairs at n600, matching ddm_ti1 exactly."""
    codes = build_spike_coherent_codes(_toy_lstars(), 4)
    assert codes[0].max() == 0 and codes[-1].max() == 0


def test_producer_is_deterministic_and_theta_independent():
    a = build_spike_coherent_codes(_toy_lstars(), 4)
    b = build_spike_coherent_codes(_toy_lstars(), 4)
    assert np.array_equal(a, b) and a.dtype == np.uint8


def test_lut_is_all_ones_when_both_scalars_are_inert():
    """The byte-identity guarantee: 1.0/1.0 => the fold multiplies by exactly 1.0."""
    lut = spike_weight_lut(1.0, 1.0)
    assert lut.tolist() == [1.0, 1.0, 1.0]
    codes = build_spike_coherent_codes(_toy_lstars(), 4)
    assert np.array_equal(lut[codes], np.ones_like(codes, dtype=np.float32))


def test_lut_applies_each_scalar_to_its_own_class():
    lut = spike_weight_lut(0.25, 1.3)
    w = lut[build_spike_coherent_codes(_toy_lstars(), 4)]
    assert w[1].tolist() == [[1.0, pytest.approx(0.25), pytest.approx(1.3), pytest.approx(1.3)]]


def test_scalars_without_the_gate_are_refused():
    """Same declared-but-inert genus as row 2: a magnitude with no gate is a silent no-op."""
    for dn, up in ((0.25, 1.0), (1.0, 1.3), (0.25, 1.3)):
        with pytest.raises(SystemExit) as exc:
            assert_spike_scalars_have_their_gate(False, dn, up)
        assert "without --seg-spike-reweight" in str(exc.value)
    assert_spike_scalars_have_their_gate(False, 1.0, 1.0)   # inert+off is fine
    assert_spike_scalars_have_their_gate(True, 0.25, 1.3)   # gated is fine


def test_the_two_knobs_are_priced_asymmetrically():
    """ddm_ti1 §3a. COHERENT is RISK-PROPORTIONAL (its race start IS its measured lift);
    SPIKE is CONCESSION-priced and must NOT equal its own (higher) lift."""
    assert SEG_SPIKE_MH_LIFT_N600 > SEG_COHERENT_MH_LIFT_N600 > 1.0
    assert SEG_COHERENT_UPWEIGHT_RACE_START == SEG_COHERENT_MH_LIFT_N600
    assert SEG_SPIKE_DOWNWEIGHT_RACE_START != SEG_SPIKE_MH_LIFT_N600
    assert SEG_SPIKE_DOWNWEIGHT_RACE_START < 1.0 < SEG_COHERENT_UPWEIGHT_RACE_START


def test_defaults_are_inert_so_the_port_is_byte_identical():
    from experiments.train_tr1_partition_renderer_mlx import TR1Config
    ns = build_argparser().parse_args(["--variant", "plain", "--out-dir", "/dev/null"])
    assert ns.seg_spike_reweight is False
    assert ns.seg_spike_downweight == 1.0 and ns.seg_coherent_upweight == 1.0
    c = TR1Config.__dataclass_fields__
    assert c["seg_spike_reweight"].default is False
    assert c["seg_spike_downweight"].default == 1.0
    assert c["seg_coherent_upweight"].default == 1.0


def test_producer_is_wired_into_pair_loss_multiplicatively():
    """A ported producer whose output nothing consumes is the orphan class this row closes."""
    src = (WORKTREE / "experiments" / "train_tr1_partition_renderer_mlx.py").read_text()
    assert "spike_codes = build_spike_coherent_codes(lstars, cfg.num_pairs)" in src
    assert "w_np = _sp_w if w_np is None else (w_np * _sp_w)" in src, "must MULTIPLY, not replace"
    # ... and it must compose AFTER the additive lane-guard accumulation, so it scales it.
    assert src.index("_lg_add") < src.index("_sp_w = spike_lut[")


_GT_N600 = Path(
    "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")


@pytest.mark.skipif(not _GT_N600.exists(), reason="n600 GT cache not present on this host")
def test_n600_cross_validation_against_two_independent_implementations():
    """THE PORT'S POSITIVE CONTROL, at n600 on the real frozen-authority GT (no toy, no
    scorer forward). ddm_ti1 and ddm_fl1 measured 625,297 SPIKE px with two independent
    implementations; this port is a third and must agree EXACTLY. A port that silently
    computes a different field is the failure mode this catches."""
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    lstars = open_stored_npy_memmap(str(_GT_N600), "lstars")
    assert lstars.shape == (600, 384, 512)
    codes = build_spike_coherent_codes(lstars, 600)
    interior_px = 598 * 384 * 512
    assert interior_px == 117_571_584                       # ti1's stated scope
    assert int((codes == SPIKE_CODE_SPIKE).sum()) == 625_297  # ti1 + fl1, exact
    union = int((codes != SPIKE_CODE_STABLE).sum())
    assert union / interior_px == pytest.approx(0.0196, abs=5e-4)  # ti1's "1.96% of pixels"


def test_off_path_leaves_the_composed_weight_bit_untouched():
    """Byte-identity of the OFF path, proven STRUCTURALLY (see the memo: the trainer is not
    run-to-run bit-deterministic on this host, so a checkpoint diff cannot prove it).
    The guard is a None sentinel, so with the lever off the fold below never executes; and
    even if it did with inert scalars, the LUT is all-ones. Both legs asserted."""
    rng = np.random.default_rng(0)
    base = rng.random((1, 4), dtype=np.float32)          # a composed class/lane-guard weight

    def fold(w_np, spike_codes, spike_lut, idx):          # verbatim consumer expression
        if spike_codes is not None:
            _sp_w = spike_lut[spike_codes[idx]]
            w_np = _sp_w if w_np is None else (w_np * _sp_w)
        return w_np

    # leg 1: lever OFF -> sentinel None -> the expression is the identity on w_np
    assert fold(base, None, None, 1) is base
    assert fold(None, None, None, 1) is None
    # leg 2: lever ON with inert scalars -> multiplies by exactly 1.0, bit-for-bit
    codes = build_spike_coherent_codes(_toy_lstars(), 4)
    out = fold(base, codes, spike_weight_lut(1.0, 1.0), 1)
    assert np.array_equal(out, base) and out.dtype == base.dtype
