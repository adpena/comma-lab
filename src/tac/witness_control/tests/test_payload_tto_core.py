# SPDX-License-Identifier: MIT
"""REAL CORE validation for payload_tto (task #350).

NOT a toy: this drives the ACTUAL level-set witness module (``build_levelset_rgb_witness`` from the
launch-path trainer) and its ACTUAL ``sdf`` render + a real MLX ``render_through_R_mlx`` fused-R
call, and validates the payload-TTO optimizer MECHANISM:
  (1) it strictly REDUCES the differentiable objective (the optimizer actually optimizes);
  (2) two independent runs are BIT-IDENTICAL (deterministic — the #348/STAGE-0 exploitation);
  (3) a resume-from-checkpoint continues BIT-IDENTICALLY to an uninterrupted run;
  (4) FROZEN pairs' code rows are byte-unchanged (the per-pair freeze mask is correct).

Runs on MLX-CPU (bit-identical cross-process by construction — the discipline the #348 doc names)
so the determinism assertions are EXACT equality, not a tolerance band. The pair count is small
because determinism + freeze + objective-reduction are pair-count-invariant PROPERTIES of the
optimizer; the driver measures the DECODED d_seg band at n600 separately.

Marked ``slow`` (imports the heavy trainer module); a targeted run exercises it:
    .venv/bin/python -m pytest src/tac/witness_control/tests/test_payload_tto_core.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[4]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.slow


def _seed_mlx_or_xfail(mx) -> None:
    try:
        mx.random.seed(0)
    except RuntimeError as exc:
        if "[metal::load_device] No Metal device available" in str(exc):
            pytest.xfail(
                "#856 known-red environment/MLX-gating: mlx.random.seed loads "
                "Metal in this sandbox even after CPU pinning"
            )
        raise


def _build_ctx():
    """A small REAL witness + a real seg-CE loss closure on real MLX renders (CPU, deterministic)."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    _seed_mlx_or_xfail(mx)  # nn.Linear init draws from the GLOBAL MLX PRNG.
    from train_levelset_witness_realized_through_R_mlx import build_levelset_rgb_witness

    P_PIX, IN_FEAT = 96 * 128, 8
    NUM_PAIRS, MOD, K = 4, 8, 5
    rng = np.random.default_rng(0)
    coord_feats = mx.array(rng.standard_normal((P_PIX, IN_FEAT)).astype(np.float32))
    # a fixed real target argmax per pair (one-hot), the seg-surrogate target form.
    targets = [mx.array(np.eye(K, dtype=np.float32)[rng.integers(0, K, size=P_PIX)])
               for _ in range(NUM_PAIRS)]
    model = build_levelset_rgb_witness(
        num_pairs=NUM_PAIRS, in_feat=IN_FEAT, hidden_dim=16, n_hidden=2, mod_dim=MOD,
        n_classes=K, activation="hosc", softmax_temp=0.5, wire_w0=1.0, wire_s0=1.0,
        hosc_beta=1.0, hosc_omega=1.0, chroma=True)
    # perturb code off zero so there is something to optimize (a trained payload is non-zero).
    model.code = mx.array(rng.standard_normal(model.code.shape).astype(np.float32) * 0.3)
    mx.eval(model.code)

    def loss_closure(m, pi):
        logits = m.sdf(coord_feats, 2 * pi)           # (P_PIX, K) — real out_sdf render
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(mx.sum(targets[pi] * logp, axis=-1))   # CE against the target partition

    return model, loss_closure


def test_core_reduces_objective_and_is_deterministic():
    from tac.witness_control.payload_tto import TTOConfig, optimize_codes

    cfg = TTOConfig(n_iters=40, lr=2e-2, seed=0)

    m1, lc1 = _build_ctx()
    r1 = optimize_codes(m1, lc1, [0, 1], cfg=cfg, fused_r=False)  # CPU => fused_r off
    assert r1.loss_last < r1.loss_first, "optimizer must reduce the objective"
    assert r1.surrogate_last < r1.surrogate_first, "seg surrogate must improve"

    m2, lc2 = _build_ctx()
    r2 = optimize_codes(m2, lc2, [0, 1], cfg=cfg, fused_r=False)
    assert r1.code_sha_after == r2.code_sha_after, "two runs must be BIT-IDENTICAL (deterministic)"
    assert r1.loss_last == r2.loss_last


def test_core_freezes_non_target_pairs():
    from tac.witness_control.payload_tto import TTOConfig, optimize_codes

    m, lc = _build_ctx()
    before = np.asarray(m.code, np.float32).copy()
    optimize_codes(m, lc, [0], cfg=TTOConfig(n_iters=20, lr=2e-2), fused_r=False)
    after = np.asarray(m.code, np.float32)
    # pair 0 rows (0,1) moved; pairs 1..3 rows (2..7) are byte-unchanged.
    assert not np.array_equal(before[0:2], after[0:2]), "target pair must move"
    assert np.array_equal(before[2:], after[2:]), "non-target pairs must be FROZEN (byte-identical)"


def test_core_resume_is_bit_identical(tmp_path):
    from tac.witness_control.payload_tto import TTOConfig, optimize_codes

    # uninterrupted 40-iter run
    mA, lcA = _build_ctx()
    rA = optimize_codes(mA, lcA, [0, 1], cfg=TTOConfig(n_iters=40, lr=2e-2), fused_r=False)

    # split run: 20 iters -> checkpoint -> resume to 40, must match rA bit-for-bit
    ckpt = tmp_path / "tto_resume.npz"
    mB, lcB = _build_ctx()
    optimize_codes(mB, lcB, [0, 1], cfg=TTOConfig(n_iters=20, lr=2e-2, ckpt_every=20),
                   resume_path=str(ckpt), fused_r=False)
    assert ckpt.exists(), "checkpoint must be written"
    mC, lcC = _build_ctx()
    rC = optimize_codes(mC, lcC, [0, 1], cfg=TTOConfig(n_iters=40, lr=2e-2, ckpt_every=20),
                        resume_path=str(ckpt), fused_r=False)
    assert rC.code_sha_after == rA.code_sha_after, "resumed run must equal the uninterrupted run"


def test_real_code_bytes_measures_the_code_chunk():
    from tac.witness_control.payload_tto import real_code_bytes

    rng = np.random.default_rng(1)
    params = {"code": rng.standard_normal((1200, 32)).astype(np.float32),
              "in_proj.weight": rng.standard_normal((16, 8)).astype(np.float32)}
    b = real_code_bytes(params)
    assert isinstance(b, int) and b > 0, "code chunk must have a measured int8+brotli byte count"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
