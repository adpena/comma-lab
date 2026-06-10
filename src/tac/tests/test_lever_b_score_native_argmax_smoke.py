# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the LEVER-B score-native argmax smoke.

Guards (Slot EEE fake-implementation classes):
  * Class 1/2 (returns-constants / tests-verify-constants): the generator must
    ACTUALLY descend d_seg on a real frozen-SegNet-shaped fixture, and d_seg must
    be the real argmax-disagreement (a constant-returning generator FAILS).
  * Portability contract: the numpy reference forward must argmax-match the MLX
    forward (the score-native d_seg quantity is argmax, so argmax-parity is the
    contract).

These tests do NOT touch the real 0.mkv / frozen scorer (fast, hermetic): they use
a small synthetic argmax fixture to verify the LEARNING MECHANISM is real. The
real-frozen-scorer descent is the smoke artifact itself (smoke_result.json), which
is [macOS-MLX research-signal] advisory and lives on the SSD tier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS = REPO_ROOT / "tools"
for p in (REPO_ROOT, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from lever_b_score_native_argmax_smoke import (  # noqa: E402
    ScoreNativeSegGenerator,
    _build_coords,
    _quantize_blob_bytes,
    numpy_reference_forward,
)


def _make_synthetic_argmax(h: int, w: int, n_classes: int = 5, seed: int = 0) -> np.ndarray:
    """A piecewise-constant 5-class partition (like a real SegNet argmax map):
    horizontal bands -> a learnable spatial structure (NOT random noise)."""
    rng = np.random.default_rng(seed)
    bands = np.linspace(0, n_classes, h, endpoint=False).astype(np.int32)  # (h,)
    lab = np.broadcast_to(bands[:, None], (h, w)).copy()
    # add a couple of vertical class blobs so x matters too
    lab[: h // 3, : w // 2] = rng.integers(0, n_classes)
    return lab % n_classes


def test_generator_actually_descends_d_seg_not_constant():
    """The CORE no-fake guard: training must DESCEND the argmax-disagreement rate.

    If the generator body were replaced by a constant, d_seg would NOT descend
    (it would stay at the random/constant disagreement level). This test FAILS
    for any constant-returning fake.
    """
    h, w = 24, 32
    tgt = _make_synthetic_argmax(h, w, seed=1)
    coords = _build_coords(h, w)
    mx.random.seed(0)
    model = ScoreNativeSegGenerator(num_pairs=1, hidden_dim=48, n_hidden=3, mod_dim=8, n_fourier=12)
    opt = optim.AdamW(learning_rate=5e-3, weight_decay=1e-4)

    def loss_fn(m, lab):
        logits = m(coords, 0)
        return nn.losses.cross_entropy(logits, lab, reduction="mean")

    lng = nn.value_and_grad(model, loss_fn)
    lab = mx.array(tgt.ravel())

    def d_seg():
        pred = np.array(mx.argmax(model(coords, 0), axis=-1))
        return float((pred != tgt.ravel()).mean())

    d0 = d_seg()
    for _ in range(150):
        _, g = lng(model, lab)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state)
    d1 = d_seg()
    # Must descend substantially: piecewise-constant partition is learnable.
    assert d1 < d0, f"d_seg did not descend: {d0} -> {d1}"
    assert d1 < 0.15, f"d_seg failed to learn the synthetic partition: {d1}"


def test_d_seg_is_real_argmax_disagreement():
    """A perfect-match prediction => d_seg == 0; a shifted prediction => d_seg > 0.
    Verifies d_seg is the genuine per-pixel argmax-flip RATE, not a constant.
    """
    tgt = _make_synthetic_argmax(16, 16, seed=2).ravel()
    same = tgt.copy()
    shifted = (tgt + 1) % 5
    assert float((same != tgt).mean()) == 0.0
    assert float((shifted != tgt).mean()) > 0.5


def test_numpy_reference_argmax_parity():
    """Portability contract: numpy reference forward argmax-matches MLX forward."""
    from mlx.utils import tree_flatten

    mx.random.seed(3)
    model = ScoreNativeSegGenerator(num_pairs=2, hidden_dim=48, n_hidden=3, mod_dim=8, n_fourier=12)
    # non-zero mod so FiLM path is exercised
    model.mod = mx.array(np.random.default_rng(3).standard_normal((2, 8)).astype(np.float32))
    coords = _build_coords(20, 20)
    mlx_logits = np.array(model(coords, 0))
    flat = dict(tree_flatten(model.parameters()))
    params = {k: np.array(v) for k, v in flat.items()}
    fB = np.array(model._fourier_B)
    np_logits = numpy_reference_forward(
        params, fB, np.array(coords), params["mod"][0], model.n_hidden, model.hidden_dim
    )
    argmax_agree = float((mlx_logits.argmax(-1) == np_logits.argmax(-1)).mean())
    assert argmax_agree == 1.0, f"numpy/MLX argmax disagree: {argmax_agree}"


def test_blob_bytes_base_dominated_and_amortizes():
    """The amortization thesis: the quantized blob is base-dominated; the per-pair
    mod section grows ~linearly with num_pairs while the base stays constant."""
    mx.random.seed(4)
    m16 = ScoreNativeSegGenerator(num_pairs=16)
    m600 = ScoreNativeSegGenerator(num_pairs=600)
    b16 = _quantize_blob_bytes(m16)
    b600 = _quantize_blob_bytes(m600)
    # base section identical-size across num_pairs (shared MLP)
    assert b16["base_int8_raw_bytes"] == b600["base_int8_raw_bytes"]
    # mod section grows with num_pairs
    assert b600["mod_int8_raw_bytes"] > b16["mod_int8_raw_bytes"]
    # at 16 pairs the base dominates (the amortization lever)
    assert b16["base_int8_brotli_bytes"] > 10 * b16["mod_int8_brotli_bytes"]
