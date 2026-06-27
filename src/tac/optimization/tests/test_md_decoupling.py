# SPDX-License-Identifier: MIT
"""Tests for the MD-Decoupling optimizer (arXiv:2606.25971), MLX, CPU-only.

CPU-pinned (``mx.set_default_device(mx.cpu)``) so these never contend with a GPU
witness run for the unified-memory slot. Tiny toy nets only — $0, CPU-light.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

mx.set_default_device(mx.cpu)

from tac.optimization.md_decoupling import (  # noqa: E402
    MDDecoupledOptimizer,
    default_md_eligible,
    newton_schulz5,
)


def _toy_mlp(seed: int = 0, layers=(8, 16, 16, 1)):
    mx.random.seed(seed)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

        def __call__(self, x):
            for i, lyr in enumerate(self.fc):
                x = lyr(x)
                if i < len(self.fc) - 1:
                    x = nn.relu(x)
            return x

    return MLP()


def _regression_data(n=128, din=8, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, din)).astype(np.float32)
    w = rng.standard_normal((din, 1)).astype(np.float32)
    y = (X @ w + 0.1 * rng.standard_normal((n, 1))).astype(np.float32)
    return mx.array(X), mx.array(y)


def _train(model, opt, X, y, steps=200):
    def loss_fn(m):
        return mx.mean(mx.square(m(X) - y))

    vg = nn.value_and_grad(model, loss_fn)
    last = None
    for _ in range(steps):
        loss, grads = vg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        last = float(loss)
    return last


# --- 1. construct ------------------------------------------------------------
def test_construct_and_invalid_base():
    opt = MDDecoupledOptimizer(learning_rate=1e-3)
    assert opt.base == "adam"
    assert float(opt.learning_rate) == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        MDDecoupledOptimizer(learning_rate=1e-3, base="rmsprop")


# --- 2. eligibility predicate ------------------------------------------------
def test_default_md_eligible_predicate():
    W = mx.zeros((4, 5))
    b = mx.zeros((4,))
    code = mx.zeros((10, 3))
    Bfix = mx.zeros((2, 24))
    assert default_md_eligible("fc.0.weight", W) is True
    assert default_md_eligible("fc.0.bias", b) is False  # 1D
    assert default_md_eligible("code", code) is False  # not *.weight
    assert default_md_eligible("_B", Bfix) is False  # not *.weight
    assert default_md_eligible("out.weight", b) is False  # *.weight but 1D


# --- 3. Newton-Schulz orthogonalization --------------------------------------
def test_newton_schulz_orthogonalizes():
    rng = np.random.default_rng(0)
    X = mx.array(rng.standard_normal((6, 10)).astype(np.float32))
    O = newton_schulz5(X, steps=5)
    mx.eval(O)
    # NS5 (Muon) is an APPROXIMATE orthogonalization: it bands the singular
    # values toward 1 (empirically ~[0.69, 1.13] at 5 steps), NOT exact identity.
    # The faithful invariant is that ALL singular values are pulled into a tight
    # band around 1 (a well-conditioned, near-orthogonal map).
    sv = np.linalg.svd(np.asarray(O), compute_uv=False)
    assert np.all(sv > 0.5) and np.all(sv < 1.5), f"NS5 singular values out of band: {sv}"
    # Raw input is far from orthogonal (wide SV spread) — NS5 must tighten it.
    sv_raw = np.linalg.svd(rng.standard_normal((6, 10)), compute_uv=False)
    assert (sv.max() - sv.min()) < (sv_raw.max() - sv_raw.min())


# --- 4. step-0 identity: gains=1, What=W -> reconstruction == W ---------------
def test_md_step_identity_at_init_with_zero_grad():
    opt = MDDecoupledOptimizer(learning_rate=1e-3)
    W = mx.array(np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32))
    G = mx.zeros((4, 5))
    out = opt._md_step("fc.weight", W, G, mx.array(1e-3), mx.array(3e-4))
    mx.eval(out)
    # Zero grad => gains stay 1, direction unchanged after projection => W reconstructed.
    assert np.allclose(np.asarray(out), np.asarray(W), atol=1e-5)
    st = opt._pstate["fc.weight"]
    assert np.allclose(np.asarray(st["gr"]), 1.0)
    assert np.allclose(np.asarray(st["gc"]), 1.0)


# --- 5. direction stays on the hypersphere (constant Frobenius norm) ----------
def test_direction_stays_on_hypersphere():
    model = _toy_mlp(seed=2)
    X, y = _regression_data()
    opt = MDDecoupledOptimizer(learning_rate=3e-3)
    # capture init Frobenius norms of the *.weight leaves
    from mlx.utils import tree_flatten

    _train(model, opt, X, y, steps=120)
    for name, st in opt._pstate.items():
        if "What" in st:  # an MD leaf
            sphere = float(st["sphere"])
            norm = float(mx.sqrt(mx.sum(st["What"] * st["What"])))
            assert norm == pytest.approx(sphere, rel=1e-3), f"{name} drifted off sphere"
    assert tree_flatten(model.parameters())  # sanity


# --- 6. gains actually learn (move off 1.0) ----------------------------------
def test_gains_learn():
    model = _toy_mlp(seed=3)
    X, y = _regression_data()
    opt = MDDecoupledOptimizer(learning_rate=3e-3, gain_lr_scale=1.0)
    _train(model, opt, X, y, steps=150)
    moved = False
    for st in opt._pstate.values():
        if "gr" in st:
            if not np.allclose(np.asarray(st["gr"]), 1.0, atol=1e-3):
                moved = True
            if not np.allclose(np.asarray(st["gc"]), 1.0, atol=1e-3):
                moved = True
    assert moved, "per-row/col gains never moved off 1.0"


# --- 7. learning-rate schedule awareness -------------------------------------
def test_learning_rate_schedule_advances():
    import mlx.optimizers as optim

    sched = optim.cosine_decay(1e-2, 50, 1e-4)
    opt = MDDecoupledOptimizer(learning_rate=sched)
    lr0 = float(opt.learning_rate)
    model = _toy_mlp(seed=4)
    X, y = _regression_data()
    _train(model, opt, X, y, steps=60)
    lr1 = float(opt.learning_rate)
    assert lr1 < lr0  # cosine decayed


# --- 8. state is a tree of mx arrays (mx.eval works) -------------------------
def test_state_is_evaluable_tree():
    model = _toy_mlp(seed=5)
    X, y = _regression_data()
    opt = MDDecoupledOptimizer(learning_rate=1e-3)
    _train(model, opt, X, y, steps=5)
    mx.eval(opt.state)  # must not raise
    assert "params" in opt.state and "step" in opt.state
    assert int(opt.state["step"]) == 5


# --- 9. non-MD leaves (bias) get a plain Adam step (they change) -------------
def test_non_md_leaf_updates_via_plain_adam():
    model = _toy_mlp(seed=6)
    X, y = _regression_data()
    from mlx.utils import tree_flatten

    before = {k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters())}
    opt = MDDecoupledOptimizer(learning_rate=3e-3)
    _train(model, opt, X, y, steps=40)
    after = dict(tree_flatten(model.parameters()))
    bias_keys = [k for k in before if k.endswith(".bias")]
    assert bias_keys
    for k in bias_keys:
        assert not np.allclose(before[k], np.asarray(after[k]), atol=1e-6), f"{k} bias never moved"


# --- 10. muon direction base runs + stays on sphere --------------------------
def test_muon_base_runs_and_stays_on_sphere():
    model = _toy_mlp(seed=7)
    X, y = _regression_data()
    opt = MDDecoupledOptimizer(learning_rate=2e-2, base="muon")
    final = _train(model, opt, X, y, steps=120)
    assert np.isfinite(final)
    for st in opt._pstate.values():
        if "What" in st:
            sphere = float(st["sphere"])
            norm = float(mx.sqrt(mx.sum(st["What"] * st["What"])))
            assert norm == pytest.approx(sphere, rel=1e-3)


# --- 11. model param tree structure unchanged (non-invasive) -----------------
def test_param_tree_keys_unchanged():
    model = _toy_mlp(seed=8)
    X, y = _regression_data()
    from mlx.utils import tree_flatten

    keys_before = sorted(k for k, _ in tree_flatten(model.parameters()))
    opt = MDDecoupledOptimizer(learning_rate=1e-3)
    _train(model, opt, X, y, steps=10)
    keys_after = sorted(k for k, _ in tree_flatten(model.parameters()))
    assert keys_before == keys_after  # MD lives in opt state, model tree intact


# --- 12b. chain-rule: _md_step descends the reconstructed W to a target -------
def test_md_step_chain_rule_descends_to_target_weight():
    """End-to-end regression guard on the ACTUAL ``_md_step`` chain-rule.

    Minimizes ``L = 0.5*||W_recon - W*||^2`` whose W-space gradient is exactly
    ``G = W_recon - W*``. We feed that G into the real ``_md_step`` each step and
    assert the reconstructed ``W = gr (x) What (x) gc`` converges to ``W*``. If
    ANY of the three hand-derived gradients (dL/dWhat, dL/dgr, dL/dgc) in
    ``_md_step`` were wrong, this descent would stall or diverge — so this is a
    behavioral lock on the load-bearing math, not a constant check.
    """
    rng = np.random.default_rng(123)
    out, inn = 4, 5
    W0 = mx.array(rng.standard_normal((out, inn)).astype(np.float32))
    Wstar = mx.array(rng.standard_normal((out, inn)).astype(np.float32))
    # gain_lr_scale=1.0 so magnitude moves at the same rate as direction.
    opt = MDDecoupledOptimizer(learning_rate=5e-2, gain_lr_scale=1.0)
    W = W0
    L0 = float(0.5 * mx.sum((W - Wstar) ** 2))
    for _ in range(400):
        G = W - Wstar  # dL/dW for L = 0.5||W - W*||^2
        W = opt._md_step("fc.weight", W, G, mx.array(5e-2), mx.array(5e-2))
        mx.eval(W)
    Lf = float(0.5 * mx.sum((W - Wstar) ** 2))
    assert np.isfinite(Lf)
    assert Lf < 1e-3 * L0, f"_md_step did not descend to target: L0={L0} Lf={Lf}"
    # Sphere invariant still holds at the end (projection active every step).
    st = opt._pstate["fc.weight"]
    norm = float(mx.sqrt(mx.sum(st["What"] * st["What"])))
    assert norm == pytest.approx(float(st["sphere"]), rel=1e-3)


# --- 12. PARITY: MD (no warmup) trains a trivial regression as well as Adam ---
def test_md_matches_adam_without_warmup_on_toy_regression():
    import mlx.optimizers as optim

    X, y = _regression_data()
    # Adam baseline at its reasonable LR (constant, NO warmup).
    m_adam = _toy_mlp(seed=11)
    adam_final = _train(m_adam, optim.Adam(learning_rate=3e-3), X, y, steps=400)
    # MD-Decoupling at its reasonable LR (constant, NO warmup). Each optimizer at
    # a fair LR — MD's optimum differs from Adam's by parametrization (that LR
    # then transfers across width, the paper's headline). Empirically MD reaches
    # ~0.0015, BEATING Adam's ~0.0024 with no warmup.
    m_md = _toy_mlp(seed=11)
    md_final = _train(m_md, MDDecoupledOptimizer(learning_rate=1e-2, gain_lr_scale=1.0), X, y, steps=400)
    assert np.isfinite(md_final)
    assert md_final < 0.01, f"MD did not converge: {md_final}"
    # As-well-as / better-than Adam without warmup (small margin for seed/platform).
    assert md_final <= 1.1 * adam_final, f"MD {md_final} not <= Adam {adam_final}"
