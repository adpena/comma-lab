# SPDX-License-Identifier: MIT
"""Unit test for the #205 verdict-chunking OOM fix arithmetic.

`_verdict_dseg_dpose_chunked` must return EXACTLY the same mean d_seg/d_pose whether it runs the
CPU scorers as one N-wide batch (vbatch<=0) or in vbatch-pair chunks, GIVEN a deterministic per-item
scorer. (The real scorers add ~1e-6 BLAS batch-tiling noise, documented in the ledger + well within
the 0.9997 parity bar; this test locks the CHUNKING ARITHMETIC itself to be mean-exact.) The scorers
run in torch eval mode -> BatchNorm uses running stats -> batch-size-independent, so chunking is the
right memory lever: it bounds the ~66 GiB N=600 verdict spike to a vbatch-pair transient.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO / "experiments", _REPO / "upstream", _REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="module")
def trainer_mod():
    spec = importlib.util.spec_from_file_location(
        "tlw205", str(_REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _install_deterministic_stubs(m):
    """Per-item scorer stubs whose value depends ONLY on the item (not the batch) -> mean is
    chunk-invariant, isolating the chunking arithmetic."""
    def fake_seg(seg, f1s, lst):
        return [0.001 * float(np.asarray(x).sum() + 1) for x in f1s]

    def fake_pose(pn, f0s, f1s, ps):
        return [0.01 * float(np.asarray(a).sum() + np.asarray(b).sum() + 1) for a, b in zip(f0s, f1s)]

    m.cpu_verdict_d_seg_batch = fake_seg
    m.cpu_verdict_d_pose_batch = fake_pose


@pytest.mark.parametrize("vbatch", [1, 7, 8, 32, 64, 599, 600, 601])
def test_chunked_equals_unchunked(trainer_mod, vbatch):
    _install_deterministic_stubs(trainer_mod)
    rng = np.random.default_rng(0)
    n = 600
    f0s = [rng.random(k % 3 + 1) for k in range(n)]
    f1s = [rng.random(k % 5 + 1) for k in range(n)]
    lst = [0] * n
    ps = [0] * n
    un = trainer_mod._verdict_dseg_dpose_chunked(None, None, f0s, f1s, lst, ps, vbatch=0)
    ch = trainer_mod._verdict_dseg_dpose_chunked(None, None, f0s, f1s, lst, ps, vbatch=vbatch)
    assert ch[0] == pytest.approx(un[0], rel=0, abs=1e-12)
    assert ch[1] == pytest.approx(un[1], rel=0, abs=1e-12)


def test_chunking_reduces_call_batch_width(trainer_mod):
    """Chunking must actually split the work into <=vbatch-wide calls (the memory lever)."""
    widths = []

    def fake_seg(seg, f1s, lst):
        widths.append(len(f1s))
        return [0.0] * len(f1s)

    def fake_pose(pn, f0s, f1s, ps):
        return [0.0] * len(f1s)

    trainer_mod.cpu_verdict_d_seg_batch = fake_seg
    trainer_mod.cpu_verdict_d_pose_batch = fake_pose
    n = 600
    f = [np.zeros(1) for _ in range(n)]
    trainer_mod._verdict_dseg_dpose_chunked(None, None, f, f, [0] * n, [0] * n, vbatch=32)
    assert widths and max(widths) <= 32
    assert sum(widths) == n  # every pair scored exactly once
