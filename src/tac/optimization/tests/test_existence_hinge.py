# SPDX-License-Identifier: MIT
"""ddm_p4x (#920) — tests for the lane existence primitive + per-class birth matrix.

Two families, and the split is deliberate:

  * BEHAVIOUR tests assert the primitive actually computes the existence semantics
    (logsumexp_beta -> witness pixel, exactness vs brute force, connectivity effects).
    They would FAIL if the body were replaced by a constant, which is the property
    the "tests-verify-constants-not-behavior" fake class lacks.
  * WIRING tests assert the trainer really CONSUMES the flags -- the #417
    counted-but-inert class. A flag that parses but never reaches the loss is a
    fake implementation regardless of how many unit tests pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tac.optimization import existence_hinge as eh

_REPO = Path(__file__).resolve().parents[4]
_TR1 = _REPO / "experiments" / "train_tr1_partition_renderer_mlx.py"
_BASE = _REPO / "experiments" / "train_witness_realized_through_R_mlx.py"


# --------------------------------------------------------------------------- #
# BEHAVIOUR
# --------------------------------------------------------------------------- #
def test_derived_constants_cross_check_the_ledger():
    """S_PER_FLIP must reproduce the cg1r ledger's Lane magnitude, not be asserted."""
    assert eh.S_PER_FLIP == pytest.approx(100.0 / (600 * 384 * 512))
    # tr1.lane.annihilate magnitude_s = 0.1575 from 185,801 Lane flips.
    assert 185801 * eh.S_PER_FLIP == pytest.approx(0.1575, abs=1e-5)


def test_beta_is_derived_not_a_literal():
    """beta comes from the component-size law + a declared tolerance."""
    lane = eh.BIRTH_MATRIX[eh.LANE]
    expected = np.log(eh.mean_component_area(eh.LANE)) / eh.BETA_TOLERANCE_MARGIN_UNITS
    assert lane.beta == pytest.approx(expected)
    # Larger mean area demands a sharper softmax to stay inside the same tolerance.
    assert eh.derive_beta(100.0, 0.5) > eh.derive_beta(10.0, 0.5)
    # Tighter tolerance demands a sharper softmax.
    assert eh.derive_beta(40.0, 0.1) > eh.derive_beta(40.0, 0.5)


def test_logsumexp_beta_converges_to_the_witness_pixel():
    """Existence semantics: s(c) -> max_p margin(p), NOT the mean or the sum."""
    lstar = np.zeros((8, 8), dtype=np.int64)
    lstar[2, 2:5] = eh.LANE          # a 3-px word
    idx = eh.build_component_index(lstar, (eh.LANE,))
    assert idx.n_comp == 1
    margin = np.full((8, 8), -5.0, dtype=np.float32)
    margin[2, 2], margin[2, 3], margin[2, 4] = -3.0, 2.0, -1.0
    prev = None
    for beta in (1.0, 4.0, 16.0, 64.0):
        s = float(eh.existence_scores_np(margin, idx, np.full(1, beta, np.float32))[0])
        assert s >= 2.0 - 1e-5           # never below the true max
        if prev is not None:
            assert s <= prev + 1e-6      # monotone decreasing toward the max
        prev = s
    assert float(eh.existence_scores_np(
        margin, idx, np.full(1, 64.0, np.float32))[0]) == pytest.approx(2.0, abs=1e-3)
    # The MEAN of this component is negative; a mean-based term would report the word
    # as failing while its witness pixel is comfortably alive. That distinction is the
    # entire reason this primitive exists.
    assert margin[2, 2:5].mean() < 0.0


def test_scores_match_brute_force_logsumexp():
    rng = np.random.default_rng(7)
    lstar = rng.integers(0, 5, size=(48, 48))
    idx = eh.build_component_index(lstar, eh.DEFAULT_PROTECTED_CLASSES)
    assert idx.n_comp > 5
    margin = rng.normal(0.0, 3.0, size=(48, 48)).astype(np.float32)
    betas = np.full(idx.n_comp, 6.0, dtype=np.float32)
    got = eh.existence_scores_np(margin, idx, betas)
    flat = margin.reshape(-1)
    for c in range(idx.n_comp):
        vals = flat[idx.pixel_flat[idx.comp_of_px == c]].astype(np.float64)
        want = np.log(np.exp(6.0 * vals).sum()) / 6.0
        assert got[c] == pytest.approx(want, abs=1e-4)


def test_scores_are_stable_for_large_margins():
    """No overflow: exp(beta*m) would overflow fp32 well before m=400."""
    lstar = np.zeros((4, 4), dtype=np.int64)
    lstar[0, 0:2] = eh.LANE
    idx = eh.build_component_index(lstar, (eh.LANE,))
    margin = np.zeros((4, 4), dtype=np.float32)
    margin[0, 0], margin[0, 1] = 400.0, 399.0
    s = eh.existence_scores_np(margin, idx, np.full(1, 8.0, np.float32))
    assert np.isfinite(s).all()
    assert s[0] == pytest.approx(400.0, abs=1e-2)


def test_connectivity_changes_the_word_grammar():
    """A diagonal pair is ONE 8-connected word and TWO 4-connected words."""
    lstar = np.zeros((6, 6), dtype=np.int64)
    lstar[1, 1] = eh.LANE
    lstar[2, 2] = eh.LANE
    assert eh.build_component_index(
        lstar, (eh.LANE,), connectivity=eh.CONNECTIVITY_8).n_comp == 1
    assert eh.build_component_index(
        lstar, (eh.LANE,), connectivity=eh.CONNECTIVITY_4).n_comp == 2
    with pytest.raises(ValueError):
        eh.build_component_index(lstar, (eh.LANE,), connectivity=6)


def test_gt2_grammar_is_recorded_as_4_connected():
    """The measured mismatch must stay in the module, not in a memo only.

    gt2's published per-word rates are 4-connected; the primitive defends 8-connected
    words. The S ceilings differ and the module must carry both.
    """
    assert eh.GT2_VERB_MEASUREMENTS[eh.LANE]["gt_components"] == 16581.0
    assert eh.GT2_VERB_MEASUREMENTS_8CONN[eh.LANE]["gt_components"] == 14323.0
    four = eh.annihilate_ceiling_s(eh.LANE, eh.CONNECTIVITY_4)
    eight = eh.annihilate_ceiling_s(eh.LANE, eh.CONNECTIVITY_8)
    assert four == pytest.approx(0.040034, abs=1e-6)
    assert eight == pytest.approx(0.037276, abs=1e-6)
    assert eight < four  # merging diagonal fragments lets some merged words survive
    assert eh.protected_ceiling_s(connectivity=eh.CONNECTIVITY_8) == pytest.approx(
        0.044175, abs=1e-5)


def test_birth_matrix_is_per_class_geometry():
    """ONE mechanism, instantiated per class geometry -- not one global setting."""
    lane, mov = eh.BIRTH_MATRIX[eh.LANE], eh.BIRTH_MATRIX[eh.MOVABLE]
    assert lane.weight_policy == "uniform"      # Lane has no interior
    assert mov.weight_policy == "sqrt_area"     # Movable does (GOUGE 16,718 px)
    assert lane.interior_bearing is False
    assert mov.interior_bearing is True
    assert lane.beta != mov.beta                # derived from each class's own area law
    assert eh.MYCAR not in eh.BIRTH_MATRIX      # 0 annihilations in 600 frames
    assert eh.DEFAULT_PROTECTED_CLASSES == (eh.LANE, eh.MOVABLE)


def test_config_refuses_unmeasured_class_and_bad_policy():
    eh.ExistenceHingeConfig(weight=1.0).validate()
    with pytest.raises(ValueError):
        eh.ExistenceHingeConfig(weight=1.0, protected_classes=(eh.MYCAR,)).validate()
    with pytest.raises(ValueError):
        eh.ExistenceHingeConfig(weight=1.0, weight_policy_override="bogus").validate()
    with pytest.raises(ValueError):
        eh.ExistenceHingeConfig(weight=-1.0).validate()
    assert not eh.ExistenceHingeConfig().enabled()
    assert eh.ExistenceHingeConfig(weight=0.5).enabled()


def test_uniform_weights_are_area_blind():
    """The volumetric law applied literally: a 1-px word counts like a 40-px word."""
    lstar = np.zeros((10, 10), dtype=np.int64)
    lstar[0, 0] = eh.LANE                # 1 px
    lstar[5, 0:6] = eh.LANE              # 6 px
    idx = eh.build_component_index(lstar, (eh.LANE,))
    cfg = eh.ExistenceHingeConfig(weight=1.0)
    w = eh.component_weights(idx, cfg)
    assert idx.n_comp == 2
    assert w[0] == pytest.approx(w[1])   # area-blind
    w_area = eh.component_weights(
        idx, eh.ExistenceHingeConfig(weight=1.0, weight_policy_override="area"))
    assert w_area.max() > w_area.min()   # area policy is NOT area-blind


def test_hinge_only_penalizes_components_below_target():
    lstar = np.zeros((6, 6), dtype=np.int64)
    lstar[0, 0] = eh.LANE
    lstar[3, 3] = eh.LANE
    idx = eh.build_component_index(lstar, (eh.LANE,))
    cfg = eh.ExistenceHingeConfig(weight=1.0)
    margin = np.zeros((6, 6), dtype=np.float32)
    margin[0, 0] = 5.0      # alive
    margin[3, 3] = -5.0     # dying
    loss, tel = eh.existence_hinge_np(margin, idx, cfg)
    assert loss > 0.0
    assert tel["at_risk"] == 1
    assert tel["n_comp"] == 2
    # All words alive => zero loss (the term never penalizes a healthy decode).
    margin[3, 3] = 5.0
    loss2, tel2 = eh.existence_hinge_np(margin, idx, cfg)
    assert loss2 == pytest.approx(0.0)
    assert tel2["at_risk"] == 0


def test_empty_component_set_is_zero_not_nan():
    lstar = np.zeros((8, 8), dtype=np.int64)   # no Lane, no Movable
    idx = eh.build_component_index(lstar, eh.DEFAULT_PROTECTED_CLASSES)
    assert idx.n_comp == 0
    loss, tel = eh.existence_hinge_np(
        np.zeros((8, 8), np.float32), idx, eh.ExistenceHingeConfig(weight=1.0))
    assert loss == 0.0 and tel["n_comp"] == 0


def test_component_budget_refuses_rather_than_truncates():
    rng = np.random.default_rng(3)
    lstar = np.where(rng.random((64, 64)) < 0.35, eh.LANE, eh.ROAD).astype(np.int64)
    with pytest.raises(ValueError, match="max_components"):
        eh.build_component_index(lstar, (eh.LANE,), max_components=4)


def test_membership_mask_is_exclusive_and_finite():
    lstar = np.zeros((6, 6), dtype=np.int64)
    lstar[0, 0:2] = eh.LANE
    lstar[4, 4] = eh.LANE
    idx = eh.build_component_index(lstar, (eh.LANE,))
    mask = eh.membership_mask_np(idx)
    assert mask.shape == (idx.n_px, idx.n_comp)
    assert np.isfinite(mask).all()               # -1e9, never -inf (no inf-inf NaN)
    assert (mask == 0.0).sum(axis=1).max() == 1  # each pixel in exactly one component
    assert (mask == 0.0).sum() == idx.n_px


# --------------------------------------------------------------------------- #
# MLX parity (skipped where MLX is unavailable; numpy stays authoritative)
# --------------------------------------------------------------------------- #
def test_mlx_matches_numpy_and_gradient_reaches_the_witness():
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(11)
    lstar = rng.integers(0, 5, size=(32, 32))
    idx = eh.build_component_index(lstar, eh.DEFAULT_PROTECTED_CLASSES)
    assert idx.n_comp > 0
    cfg = eh.ExistenceHingeConfig(weight=1.0)
    logits = rng.normal(0.0, 2.0, size=(1, 32, 32, 5)).astype(np.float32)
    oh = (lstar[..., None] == np.arange(5)).astype(np.float32)[None]
    live = ((logits * oh).sum(-1) - (logits + oh * (-1e9)).max(-1))[0]
    betas, targets = eh.component_betas_targets(idx, cfg)
    w = eh.component_weights(idx, cfg)
    ref_s = eh.existence_scores_np(live, idx, betas)
    ref = float((w * np.maximum(targets - ref_s, 0.0)).sum() / idx.n_comp)

    args = (mx.array(idx.pixel_flat.astype(np.int32)),
            mx.array(eh.membership_mask_np(idx)),
            mx.array(betas), mx.array(targets), mx.array(w))
    out = eh.existence_hinge_mlx(mx.array(logits), mx.array(oh), *args, mx)
    mx.eval(out)
    assert float(out) == pytest.approx(ref, abs=1e-5)

    grad = mx.grad(lambda lg: eh.existence_hinge_mlx(lg, mx.array(oh), *args, mx))(
        mx.array(logits))
    mx.eval(grad)
    g = np.asarray(grad)
    assert np.isfinite(g).all()
    assert np.abs(g).sum() > 0.0            # the term is NOT gradient-dead


# --------------------------------------------------------------------------- #
# WIRING -- the #417 counted-but-inert guard
# --------------------------------------------------------------------------- #
def _argparse_flags(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                out.add(first.value)
    return out


EXPECTED_FLAGS = {
    "--existence-hinge-weight", "--existence-hinge-classes", "--existence-hinge-beta",
    "--existence-hinge-target", "--existence-hinge-weight-policy",
    "--existence-hinge-connectivity",
}


def test_tr1_trainer_declares_every_existence_flag():
    missing = EXPECTED_FLAGS - _argparse_flags(_TR1)
    assert not missing, f"TR1 argparse is missing {sorted(missing)}"


def test_loss_fn_accepts_the_existence_parameters():
    """The per-pair loss_fn must take existence_pack/existence_weight.

    NOTE the disambiguation: the base trainer defines ``loss_fn`` TWICE -- once inside
    ``make_loss_fn`` (per-pair, the path TR1 uses) and once inside ``make_loss_fn_batch``
    (batched, which TR1 does NOT use). An earlier draft of this test collected them into
    a dict and silently kept the batch one, which is why the assertion is written against
    the set of ALL definitions: a lookalike signature must not be able to satisfy it.
    """
    tree = ast.parse(_BASE.read_text())
    sigs = [{a.arg for a in n.args.args} | {a.arg for a in n.args.kwonlyargs}
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "loss_fn"]
    assert sigs, "loss_fn not found in the base trainer"
    assert any({"existence_pack", "existence_weight"} <= s for s in sigs), \
        "no loss_fn definition accepts the existence parameters"


def test_trainer_batch_path_routes_through_pair_loss():
    """Anti-bypass guard: the term is wired into pair_loss, so the optimized path
    must actually go through pair_loss. If the trainer ever switches to a batched
    loss that skips it, the flag would parse and train nothing -- inert by routing
    rather than inert by declaration, which no flag-presence test would catch.
    """
    tree = ast.parse(_TR1.read_text())
    batch = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "batch_loss"]
    assert batch, "batch_loss not found in the TR1 trainer"
    called = {n.func.id for n in ast.walk(batch[0])
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "pair_loss" in called, \
        "batch_loss no longer calls pair_loss -- the existence term would be BYPASSED"


def test_trainer_actually_passes_the_pack_to_loss_fn():
    """A declared flag that never reaches loss_fn is the counted-but-inert fake."""
    src = _TR1.read_text()
    assert "existence_pack=existence_pack" in src
    assert "existence_weight=cfg.existence_hinge_weight" in src
    # and the config must carry every flag from args
    for field in ("existence_hinge_weight", "existence_hinge_classes", "existence_hinge_beta",
                  "existence_hinge_target", "existence_hinge_weight_policy",
                  "existence_hinge_connectivity"):
        assert f"{field}=args.{field}" in src, f"{field} never threaded from args into cfg"


def test_base_loss_adds_the_term_and_reports_it():
    src = _BASE.read_text()
    assert "existence_hinge_mlx" in src, "the base loss never calls the primitive"
    assert 'terms_out["existence"]' in src, "the term is not observable in telemetry"
    assert "total = total + existence_weight * existence_term" in src


def test_off_is_structurally_byte_identical():
    """OFF must skip construction entirely -- not multiply by zero."""
    src = _BASE.read_text()
    assert "existence_on = existence_pack is not None and existence_weight != 0.0" in src
    tr1 = _TR1.read_text()
    assert "if cfg.existence_hinge_weight > 0.0:" in tr1, \
        "the TR1 setup block must be gated so OFF imports nothing and builds no state"
