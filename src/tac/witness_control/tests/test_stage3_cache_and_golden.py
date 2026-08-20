# SPDX-License-Identifier: MIT
"""STAGE 3 (task #350): decode-cache unit tests + payload-TTO GOLDEN-TRAJECTORY regression.

GOLDEN TRAJECTORY: a fixed-seed, fixed-config payload-TTO run must reproduce a COMMITTED code-sha
bit-for-bit. This locks the CORE's exact numeric trajectory against silent regression (an MLX bump,
an optimizer refactor, a determinism break). Regenerate the golden intentionally with
``TAC_REGEN_GOLDEN=1`` (then commit the updated json).

DECODE CACHE: content-addressed memoization round-trips (get/put/memoized_decode) + last-write-wins.

Marked ``slow`` (the golden test imports the heavy trainer module for the real witness).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[4]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_GOLDEN = Path(__file__).resolve().parent / "golden_tto_trajectory.json"


# ── decode cache (fast; no trainer import) ──────────────────────────────────
def test_decode_cache_roundtrip(tmp_path):
    from tac.witness_control import decode_cache as dc

    store = tmp_path / "cache.jsonl"
    code = np.arange(64, dtype=np.float32).reshape(8, 8)
    cfg = {"n_pairs": 600, "render_hw": [384, 512], "self_orient": True}
    p_sha, c_sha = dc.payload_sha256(code), dc.config_sha256(cfg)
    assert dc.get(p_sha, c_sha, store=store) is None
    dc.put(p_sha, c_sha, {"d_seg": 0.0033, "d_pose": 12.3, "axis": "[macOS-CPU advisory]"}, store=store)
    hit = dc.get(p_sha, c_sha, store=store)
    assert hit is not None and hit["verdict"]["d_seg"] == 0.0033
    # last-write-wins
    dc.put(p_sha, c_sha, {"d_seg": 0.0031}, store=store)
    assert dc.get(p_sha, c_sha, store=store)["verdict"]["d_seg"] == 0.0031


def test_memoized_decode_computes_once(tmp_path):
    from tac.witness_control import decode_cache as dc

    store = tmp_path / "cache.jsonl"
    code = np.ones((4, 4), np.float32)
    cfg = {"k": 1}
    calls = {"n": 0}

    def _decode():
        calls["n"] += 1
        return {"d_seg": 0.5}

    v1, cached1 = dc.memoized_decode(code, cfg, _decode, store=store)
    v2, cached2 = dc.memoized_decode(code, cfg, _decode, store=store)
    assert v1 == v2 == {"d_seg": 0.5}
    assert cached1 is False and cached2 is True
    assert calls["n"] == 1, "decode_fn must run ONCE (second call served from cache)"


def test_content_address_distinguishes_payload_and_config():
    from tac.witness_control import decode_cache as dc

    a = dc.cache_key(dc.payload_sha256(np.zeros((2, 2), np.float32)), dc.config_sha256({"x": 1}))
    b = dc.cache_key(dc.payload_sha256(np.ones((2, 2), np.float32)), dc.config_sha256({"x": 1}))
    c = dc.cache_key(dc.payload_sha256(np.zeros((2, 2), np.float32)), dc.config_sha256({"x": 2}))
    assert a != b and a != c and b != c


# ── golden trajectory (slow; real witness) ──────────────────────────────────
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


@pytest.mark.slow
def test_payload_tto_golden_trajectory():
    import mlx.core as mx

    from tac.witness_control.payload_tto import TTOConfig, optimize_codes

    mx.set_default_device(mx.cpu)
    _seed_mlx_or_xfail(mx)
    from train_levelset_witness_realized_through_R_mlx import build_levelset_rgb_witness

    P_PIX, IN_FEAT, NUM_PAIRS, MOD, K = 96 * 128, 8, 4, 8, 5
    rng = np.random.default_rng(0)
    coord_feats = mx.array(rng.standard_normal((P_PIX, IN_FEAT)).astype(np.float32))
    targets = [mx.array(np.eye(K, dtype=np.float32)[rng.integers(0, K, size=P_PIX)])
               for _ in range(NUM_PAIRS)]
    model = build_levelset_rgb_witness(
        num_pairs=NUM_PAIRS, in_feat=IN_FEAT, hidden_dim=16, n_hidden=2, mod_dim=MOD,
        n_classes=K, activation="hosc", softmax_temp=0.5, wire_w0=1.0, wire_s0=1.0,
        hosc_beta=1.0, hosc_omega=1.0, chroma=True)
    model.code = mx.array(rng.standard_normal(model.code.shape).astype(np.float32) * 0.3)
    mx.eval(model.code)

    def loss_closure(m, pi):
        logits = m.sdf(coord_feats, 2 * pi)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(mx.sum(targets[pi] * logp, axis=-1))

    r = optimize_codes(model, loss_closure, [0, 1], cfg=TTOConfig(n_iters=30, lr=2e-2), fused_r=False)
    got = {"code_sha_after": r.code_sha_after, "n_iters": 30, "lr": 2e-2}

    if os.environ.get("TAC_REGEN_GOLDEN") or not _GOLDEN.exists():
        _GOLDEN.write_text(json.dumps(got, indent=1))
        pytest.skip(f"golden (re)generated at {_GOLDEN} — commit it")
    golden = json.loads(_GOLDEN.read_text())
    assert r.code_sha_after == golden["code_sha_after"], (
        f"payload-TTO trajectory REGRESSED: got {r.code_sha_after} vs golden "
        f"{golden['code_sha_after']} (regen intentionally with TAC_REGEN_GOLDEN=1)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
