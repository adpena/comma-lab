#!/usr/bin/env python3
"""WAVE-1E — Gumbel-vs-argmax eval-render gap audit (Z8 + DreamerV3).

THE QUESTION
============
Sister Wave-1C2 (commit 16df2f660) proved Z8's trained EMA-shadow checkpoint
COLLAPSES on the deterministic-argmax inflate path (recon_mean 155.3 vs GT 21.5;
real DistortionNet ``d_seg`` ≈ chance 0.505). DreamerV3 (sisters C/C′, 153ece040)
collapsed the same way. CRUCIAL: these categorical renderers TRAIN via stochastic
Gumbel-Softmax sampling (``forward_training``) but RENDER at inflate via
deterministic-argmax / MAP (``forward_eval_from_indices``). The argmax of a
categorical posterior can be DEGENERATE even when samples are faithful — a known
failure mode (Jang 2016 §3.2; Maddison 2016). So the collapse may live in the
EVAL DECODE, not the trained weights.

WHAT THIS RUNNER DOES (read-only render comparison, $0, MLX-LOCAL)
=================================================================
For a given TRAINED EMA-shadow checkpoint, render the same real
``upstream/videos/0.mkv`` frames the trainer saw via FOUR decodes feeding the
SAME ``cat_to_continuous`` per-level Linear (the only thing that changes is
*which categorical representation* is fed in):

  (a) GUMBEL          — the training-time render: stochastic Gumbel-Softmax STE
                        one-hot at τ=1.0 (the training τ). Averaged over a few
                        seeds so the stochastic ensemble is captured. NOT
                        contest-valid (random) — diagnostic only.
  (b) ARGMAX          — the eval/inflate render: hard one-hot at
                        argmax(logits). DETERMINISTIC + contest-valid. The path
                        Wave-1C2 found collapses.
  (c) EXPECTED-VALUE  — the "soft MAP": softmax(logits) probability simplex fed
                        directly into cat_to_continuous (NO one-hot). The
                        posterior MEAN. DETERMINISTIC + numpy-portable +
                        contest-valid (no RNG). Candidate faithful decode.
  (d) LOW-TEMP-GUMBEL — Gumbel at τ→0.1 (sharper toward argmax). FIXED-SEED so
                        deterministic. As τ→0 collapses toward argmax;
                        diagnostic of whether sharpening helps or hurts.

For BOTH (each decode) measure:
  * recon_mean / recon_std vs GT (GT mean ≈21.5, std ≈21.1).
  * real DistortionNet ``d_seg`` = ``(SegNet(GT).argmax != SegNet(recon).argmax)``
    per-pixel argmax-flip rate (upstream/modules.py:112) — chance ≈ 0.505.
  * real DistortionNet ``d_pose`` (PoseNet first-half MSE).
  * render-faithfulness verdict (FAITHFUL vs COLLAPSED).

DECISION
========
* If a faithful decode exists (recon in GT range AND d_seg meaningfully < chance)
  while argmax collapses → **EVAL-PATH BUG CONFIRMED**. The fix is to use that
  deterministic faithful decode (EXPECTED-VALUE preferred — it is numpy-portable
  and RNG-free) in the inflate/eval render path. This could make the categorical
  PC stack contest-faithful — a genuine unlock.
* If ALL decodes collapse → **CAPACITY ISSUE**. The trained weights never learned
  a faithful render at this sizing. Per Catalog #307 this is IMPLEMENTATION-LEVEL
  (scaffold sizing), not a paradigm kill.

NO FAKE (CLAUDE.md non-negotiable)
==================================
* Loads REAL trained EMA-shadow checkpoints; renders each decode on REAL frames;
  measures REAL contest DistortionNet ``d_seg``. No surrogate, no MLX training-
  loss proxy.
* Identity guard: ``compute_distortion(GT, GT) == 0.0``.
* Each decode's recon stats + d_seg are reported as REAL numbers. A "fix" is
  only claimed if a deterministic faithful decode actually moves the real
  DistortionNet ``d_seg`` below chance while landing recon in the GT range.

CUSTODY (CLAUDE.md "MPS auth eval is NOISE" / Catalog #192/#341/#127/#323)
=========================================================================
ALL results ``[macOS-MLX research-signal]`` / ``[macOS-CPU advisory]``,
non-promotable. The contest DistortionNet runs on CPU. The MLX render is a
local research signal only. Canonical Provenance via
``tac.provenance.build_provenance_for_macos_mlx_research_signal``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
UPSTREAM = REPO_ROOT / "upstream"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Z8 build / load (sizing from Wave-1C2 _full_main config)
# ---------------------------------------------------------------------------

Z8_FULL_MAIN_CONFIG = {
    "num_levels": 3,
    "num_groups_per_level": (4, 3, 2),
    "num_categories_per_level": (16, 8, 4),
    "base_channels": 8,
    "decoder_latent_dim": 12,
    "deterministic_state_dim": 8,
}


def _find_ema_shadow_npsd(out_dir: Path) -> Path:
    ckpt = out_dir / "checkpoints"
    cands = sorted(ckpt.glob("final_*.ema_shadow.state.npsd"))
    if not cands:
        cands = sorted(ckpt.glob("*.ema_shadow.state.npsd"))
    if not cands:
        raise FileNotFoundError(f"no EMA-shadow .npsd checkpoint found under {ckpt}")
    return cands[-1]


def _build_z8_model(num_pairs: int):
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
        Z8HierarchicalPredictiveCoderMLX,
    )

    cfg = Z8HierarchicalConfig(
        num_levels=int(Z8_FULL_MAIN_CONFIG["num_levels"]),
        num_groups_per_level=tuple(Z8_FULL_MAIN_CONFIG["num_groups_per_level"]),
        num_categories_per_level=tuple(
            Z8_FULL_MAIN_CONFIG["num_categories_per_level"]
        ),
        base_channels=int(Z8_FULL_MAIN_CONFIG["base_channels"]),
        decoder_latent_dim=int(Z8_FULL_MAIN_CONFIG["decoder_latent_dim"]),
        num_pairs=int(num_pairs),
        deterministic_state_dim=int(Z8_FULL_MAIN_CONFIG["deterministic_state_dim"]),
        gumbel_temperature=1.0,
        use_straight_through=True,
    )
    return Z8HierarchicalPredictiveCoderMLX(cfg), cfg


def _load_trained_z8(out_dir: Path, *, num_pairs: int):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )

    model, cfg = _build_z8_model(num_pairs)
    npsd = _find_ema_shadow_npsd(out_dir)
    sd = unpack_state_dict_numpy(npsd.read_bytes())
    items = [(k, mx.array(v)) for k, v in sd.items()]
    model.update(tree_unflatten(items))
    mx.eval(model.parameters())
    return model, cfg, npsd


# ---------------------------------------------------------------------------
# DreamerV3 cross-check (single-level categorical posterior; G=24, K=256)
# ---------------------------------------------------------------------------

DREAMER_CONFIG = {
    "num_groups": 24,
    "num_categories": 256,
    "base_channels": 24,
}


def _build_dreamer_model(num_pairs: int):
    from tac.substrates.dreamer_v3_rssm.module import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(
        num_groups=int(DREAMER_CONFIG["num_groups"]),
        num_categories=int(DREAMER_CONFIG["num_categories"]),
        base_channels=int(DREAMER_CONFIG["base_channels"]),
        num_pairs=int(num_pairs),
        gumbel_temperature=1.0,
        use_straight_through=True,
    )
    return DreamerV3RSSMSubstrateMLX(cfg), cfg


def _load_trained_dreamer(out_dir: Path, *, num_pairs: int):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )

    model, cfg = _build_dreamer_model(num_pairs)
    npsd = _find_ema_shadow_npsd(out_dir)
    sd = unpack_state_dict_numpy(npsd.read_bytes())
    items = [(k, mx.array(v)) for k, v in sd.items()]
    model.update(tree_unflatten(items))
    mx.eval(model.parameters())
    return model, cfg, npsd


def _dreamer_decode_from_simplex(model, cfg, simplex):
    """Run DreamerV3 decoder on a (B, G, K) simplex (one-hot OR soft).

    Replicates module.forward_training/forward_eval downstream math exactly
    (module.py:572-574, 593-598) but accepts an arbitrary simplex so the
    expected-value (soft) decode works.
    """
    import mlx.core as mx

    B = int(simplex.shape[0])
    G = int(cfg.num_groups)
    K = int(cfg.num_categories)
    flat = mx.reshape(simplex, (B, G * K))
    embedding = model.cat_to_continuous(flat)
    return model._decoder_forward(embedding)  # (B, 2, 3, H, W)


def _render_all_pairs_dreamer(model, cfg, num_pairs: int, *, decode: str,
                              gumbel_seeds: int = 4, low_temp: float = 0.1):
    """Render all pairs via the named decode for DreamerV3. (P, 2, H, W, 3)."""
    import mlx.core as mx
    import numpy as np

    from tac.substrates.dreamer_v3_rssm import gumbel_softmax_sample

    K = int(cfg.num_categories)
    recon = []
    for p in range(num_pairs):
        logits = model.logits[p : p + 1]  # (1, G, K)

        if decode == "argmax":
            idx = mx.argmax(logits, axis=-1)  # (1, G)
            G = int(logits.shape[1])
            eye = mx.eye(K)
            oh = mx.reshape(mx.take(eye, mx.reshape(idx, (1 * G,)), axis=0), (1, G, K))
            rgb = _dreamer_decode_from_simplex(model, cfg, oh)
            mx.eval(rgb)
            arr = np.asarray(rgb)[0]

        elif decode == "expected_value":
            sm = mx.softmax(logits, axis=-1)  # (1, G, K)
            rgb = _dreamer_decode_from_simplex(model, cfg, sm)
            mx.eval(rgb)
            arr = np.asarray(rgb)[0]

        elif decode == "low_temp_gumbel":
            key = mx.random.key(1234 + p)
            soft, _ = gumbel_softmax_sample(
                logits, temperature=float(low_temp), use_straight_through=True, key=key
            )
            rgb = _dreamer_decode_from_simplex(model, cfg, soft)
            mx.eval(rgb)
            arr = np.asarray(rgb)[0]

        elif decode == "gumbel":
            acc = None
            for s in range(int(gumbel_seeds)):
                key = mx.random.key(7000 + p * 97 + s)
                soft, _ = gumbel_softmax_sample(
                    logits, temperature=1.0, use_straight_through=True, key=key
                )
                rgb = _dreamer_decode_from_simplex(model, cfg, soft)
                mx.eval(rgb)
                a = np.asarray(rgb)[0]
                acc = a if acc is None else (acc + a)
            arr = acc / float(max(1, int(gumbel_seeds)))

        else:
            raise ValueError(f"unknown decode {decode!r}")

        arr = np.transpose(arr, (0, 2, 3, 1))  # (2,H,W,3)
        recon.append(arr)
    return np.stack(recon, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# The four decode variants — all feed the SAME cat_to_continuous Linear.
#
# The ONLY architectural difference between training-render and eval-render is
# WHICH categorical representation hits cat_to_continuous_per_level[level]:
#   forward_training (Gumbel):  STE one-hot at Gumbel-perturbed argmax.
#   forward_eval (argmax):      one-hot at argmax(logits).
# This runner injects all four. The fusion + decoder topology downstream is
# IDENTICAL across decodes (verified against mlx_renderer.py lines 561-698).
# ---------------------------------------------------------------------------


def _decode_from_per_level_simplex(model, cfg, per_level_simplex: list):
    """Run Z8 fusion + decoder on a list of per-level simplex tensors.

    per_level_simplex[level] : (B, G_l, K_l) — any categorical representation
    (one-hot OR soft probability simplex). Replicates forward_eval_from_indices'
    downstream math exactly (mlx_renderer.py:672-698) but accepts an arbitrary
    simplex rather than hard indices, so the expected-value (soft) decode works.
    """
    import mlx.core as mx

    L = int(cfg.num_levels)
    B = int(per_level_simplex[0].shape[0])

    per_level_embeddings = []
    for level_idx in range(L):
        G_l = int(cfg.num_groups_per_level[level_idx])
        K_l = int(cfg.num_categories_per_level[level_idx])
        simplex = per_level_simplex[level_idx]  # (B, G_l, K_l)
        flat = mx.reshape(simplex, (B, G_l * K_l))
        embedding = model.cat_to_continuous_per_level[level_idx](flat)
        per_level_embeddings.append(embedding)

    fused_latent = per_level_embeddings[0]
    for level_idx in range(1, L):
        fused_latent = fused_latent + per_level_embeddings[level_idx]

    ego_motion = mx.zeros((B, int(cfg.ego_motion_dim)))
    gru_input = mx.concatenate([fused_latent, ego_motion], axis=-1)
    det_state = mx.tanh(model.deterministic_gate(gru_input))

    decoder_input = mx.concatenate([fused_latent, det_state], axis=-1)
    decoder_embedding = model.level_fusion(decoder_input)
    return model._decoder_forward(decoder_embedding)  # (B, 2, 3, H, W)


def _per_level_logits_for_pair(model, cfg, p: int):
    """Return list of (1, G_l, K_l) logits for pair p, one per level."""
    L = int(cfg.num_levels)
    out = []
    for level_idx in range(L):
        out.append(model.logits_per_level[level_idx][p : p + 1])  # (1, G_l, K_l)
    return out


def _onehot_from_logits(logits, K_l):
    """Hard one-hot at argmax(logits). logits (1, G_l, K_l) -> (1, G_l, K_l)."""
    import mlx.core as mx

    G_l = int(logits.shape[1])
    idx = mx.argmax(logits, axis=-1)  # (1, G_l)
    eye = mx.eye(K_l)
    flat = mx.take(eye, mx.reshape(idx, (1 * G_l,)), axis=0)  # (G_l, K_l)
    return mx.reshape(flat, (1, G_l, K_l))


def _softmax_simplex(logits):
    """Expected-value (soft MAP): softmax(logits) probability simplex."""
    import mlx.core as mx

    return mx.softmax(logits, axis=-1)


def _gumbel_simplex(logits, K_l, *, temperature: float, key):
    """Gumbel-Softmax STE one-hot at the given temperature (canonical helper)."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        gumbel_softmax_sample,
    )

    soft, _idx = gumbel_softmax_sample(
        logits,
        temperature=float(temperature),
        use_straight_through=True,
        key=key,
    )
    return soft  # (1, G_l, K_l) STE one-hot in forward


def _render_all_pairs_decode(model, cfg, num_pairs: int, *, decode: str,
                             gumbel_seeds: int = 4, low_temp: float = 0.1):
    """Render all pairs via the named decode. Returns numpy (P, 2, H, W, 3).

    decode ∈ {"argmax", "expected_value", "gumbel", "low_temp_gumbel"}.
    For "gumbel" we average ``gumbel_seeds`` stochastic renders (the training-
    render ensemble). For "low_temp_gumbel" we use a single FIXED seed (so it is
    deterministic) at ``low_temp``.
    """
    import mlx.core as mx
    import numpy as np

    L = int(cfg.num_levels)
    Ks = [int(cfg.num_categories_per_level[lv]) for lv in range(L)]
    recon = []
    for p in range(num_pairs):
        logits = _per_level_logits_for_pair(model, cfg, p)  # list of (1,G_l,K_l)

        if decode == "argmax":
            simplex = [_onehot_from_logits(logits[lv], Ks[lv]) for lv in range(L)]
            rgb = _decode_from_per_level_simplex(model, cfg, simplex)
            mx.eval(rgb)
            arr = np.asarray(rgb)[0]  # (2,3,H,W)

        elif decode == "expected_value":
            simplex = [_softmax_simplex(logits[lv]) for lv in range(L)]
            rgb = _decode_from_per_level_simplex(model, cfg, simplex)
            mx.eval(rgb)
            arr = np.asarray(rgb)[0]

        elif decode == "low_temp_gumbel":
            # FIXED seed per pair => deterministic. Sharper toward argmax.
            key = mx.random.key(1234 + p)
            simplex = [
                _gumbel_simplex(logits[lv], Ks[lv], temperature=low_temp, key=key)
                for lv in range(L)
            ]
            rgb = _decode_from_per_level_simplex(model, cfg, simplex)
            mx.eval(rgb)
            arr = np.asarray(rgb)[0]

        elif decode == "gumbel":
            # Stochastic training-render ensemble: average gumbel_seeds renders
            # at τ=1.0 (the training τ).
            acc = None
            for s in range(int(gumbel_seeds)):
                key = mx.random.key(7000 + p * 97 + s)
                simplex = [
                    _gumbel_simplex(logits[lv], Ks[lv], temperature=1.0, key=key)
                    for lv in range(L)
                ]
                rgb = _decode_from_per_level_simplex(model, cfg, simplex)
                mx.eval(rgb)
                a = np.asarray(rgb)[0]
                acc = a if acc is None else (acc + a)
            arr = acc / float(max(1, int(gumbel_seeds)))

        else:
            raise ValueError(f"unknown decode {decode!r}")

        arr = np.transpose(arr, (0, 2, 3, 1))  # (2,H,W,3)
        recon.append(arr)
    return np.stack(recon, axis=0).astype(np.float32)  # (P,2,H,W,3)


# ---------------------------------------------------------------------------
# GT decode + real DistortionNet (reused contract from Wave-1C2)
# ---------------------------------------------------------------------------


def _decode_gt_pairs(video_path: str, num_pairs: int):
    import numpy as np

    from tac.data import decode_video

    frames = decode_video(
        video_path, target_h=384, target_w=512, max_frames=2 * num_pairs
    )
    if len(frames) < 2 * num_pairs:
        raise RuntimeError(
            f"decoded {len(frames)} frames; need {2 * num_pairs}"
        )
    gt = np.stack([f.numpy() for f in frames[: 2 * num_pairs]], axis=0)
    return gt.reshape(num_pairs, 2, 384, 512, 3).astype(np.float32)


def _real_distortion_net():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from modules import DistortionNet  # type: ignore[import-not-found]

    dn = DistortionNet().eval()
    dn.load_state_dicts(
        str(UPSTREAM / "models" / "posenet.safetensors"),
        str(UPSTREAM / "models" / "segnet.safetensors"),
        "cpu",
    )
    return dn


def _measure_real(dn, gt_pairs, recon_pairs, *, batch: int = 32) -> dict:
    import numpy as np
    import torch

    P = gt_pairs.shape[0]
    d_seg_all: list[float] = []
    d_pose_all: list[float] = []
    gt_t = torch.from_numpy(gt_pairs)
    rec_t = torch.from_numpy(recon_pairs)
    for s in range(0, P, batch):
        e = min(s + batch, P)
        with torch.inference_mode():
            d_pose, d_seg = dn.compute_distortion(gt_t[s:e], rec_t[s:e])
        d_pose_all.extend([float(x) for x in d_pose.tolist()])
        d_seg_all.extend([float(x) for x in d_seg.tolist()])
    d_seg_arr = np.asarray(d_seg_all, dtype=np.float64)
    d_pose_arr = np.asarray(d_pose_all, dtype=np.float64)
    return {
        "mean_d_seg": float(d_seg_arr.mean()),
        "mean_d_pose": float(d_pose_arr.mean()),
        "max_d_seg": float(d_seg_arr.max()),
        "min_d_seg": float(d_seg_arr.min()),
        "n_pairs": int(P),
    }


def _render_faithfulness(recon_pairs, gt_pairs) -> dict:
    """COLLAPSED if std < 10% GT std OR mean > 3x / < 1/3 GT mean."""
    import numpy as np

    gt_mean = float(np.mean(gt_pairs))
    gt_std = float(np.std(gt_pairs))
    recon_mean = float(np.mean(recon_pairs))
    recon_std = float(np.std(recon_pairs))
    collapsed_const = recon_std < 0.10 * gt_std
    collapsed_sat = (recon_mean > 3.0 * gt_mean) or (recon_mean < gt_mean / 3.0)
    collapsed = bool(collapsed_const or collapsed_sat)
    return {
        "gt_mean": gt_mean,
        "gt_std": gt_std,
        "recon_mean": recon_mean,
        "recon_std": recon_std,
        "collapsed_near_constant": bool(collapsed_const),
        "collapsed_saturated": bool(collapsed_sat),
        "verdict": "COLLAPSED" if collapsed else "FAITHFUL",
    }


# ---------------------------------------------------------------------------
# DreamerV3 cross-check (optional; reuses its in-substrate renderer)
# ---------------------------------------------------------------------------


def _dreamer_logits_decode_available() -> bool:
    try:
        from tac.substrates.dreamer_v3_rssm.module import (  # noqa: F401
            DreamerV3RSSMConfig,  # type: ignore
        )

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _provenance_for(result_path: Path) -> dict:
    import hashlib

    from tac.provenance import (
        build_provenance_for_macos_mlx_research_signal,
        provenance_to_dict,
    )

    sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    prov = build_provenance_for_macos_mlx_research_signal(
        artifact_sha256=sha,
        source_path=str(result_path.relative_to(REPO_ROOT)),
    )
    return provenance_to_dict(prov)


def run_audit(*, mode: str, arm_dir: Path, num_pairs: int, gumbel_seeds: int,
              low_temp: float, out_dir: Path, substrate_label: str) -> dict:
    if mode == "z8":
        print(f"[gap-audit] loading trained Z8 from {arm_dir}", flush=True)
        model, cfg, npsd = _load_trained_z8(arm_dir, num_pairs=num_pairs)
        render_fn = _render_all_pairs_decode
        substrate_config = dict(Z8_FULL_MAIN_CONFIG)
    elif mode == "dreamer":
        print(f"[gap-audit] loading trained DreamerV3 from {arm_dir}", flush=True)
        model, cfg, npsd = _load_trained_dreamer(arm_dir, num_pairs=num_pairs)
        render_fn = _render_all_pairs_dreamer
        substrate_config = dict(DREAMER_CONFIG)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    print(f"[gap-audit] EMA-shadow checkpoint = {npsd.name}", flush=True)

    print("[gap-audit] decoding GT frames", flush=True)
    gt_pairs = _decode_gt_pairs(str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
                                num_pairs)

    print("[gap-audit] loading real contest DistortionNet (CPU)", flush=True)
    dn = _real_distortion_net()

    # NO-FAKE identity guard: compute_distortion(GT, GT) == 0.
    import torch

    with torch.inference_mode():
        gt_t = torch.from_numpy(gt_pairs[:1])
        idp, ids = dn.compute_distortion(gt_t, gt_t)
    identity_d_seg = float(ids.mean())
    identity_d_pose = float(idp.mean())
    print(
        f"[gap-audit] NO-FAKE identity guard: d_seg(GT,GT)={identity_d_seg:.6f} "
        f"d_pose(GT,GT)={identity_d_pose:.6f}",
        flush=True,
    )

    decodes = ["argmax", "expected_value", "gumbel", "low_temp_gumbel"]
    per_decode: dict[str, dict] = {}
    for decode in decodes:
        print(f"[gap-audit] rendering decode={decode}", flush=True)
        recon = render_fn(
            model, cfg, num_pairs, decode=decode,
            gumbel_seeds=gumbel_seeds, low_temp=low_temp,
        )
        faith = _render_faithfulness(recon, gt_pairs)
        metrics = _measure_real(dn, gt_pairs, recon)
        per_decode[decode] = {
            "deterministic": decode in ("argmax", "expected_value", "low_temp_gumbel"),
            "contest_valid": decode in ("argmax", "expected_value", "low_temp_gumbel"),
            "render_faithfulness": faith,
            "real_distortion_net": metrics,
        }
        print(
            f"[gap-audit]   {decode}: recon_mean={faith['recon_mean']:.2f} "
            f"recon_std={faith['recon_std']:.2f} verdict={faith['verdict']} "
            f"d_seg={metrics['mean_d_seg']:.6f} d_pose={metrics['mean_d_pose']:.2f}",
            flush=True,
        )

    # Verdict logic.
    chance = 0.505
    faithful_decodes = [
        d for d in decodes
        if per_decode[d]["render_faithfulness"]["verdict"] == "FAITHFUL"
        and per_decode[d]["real_distortion_net"]["mean_d_seg"] < chance - 0.005
    ]
    deterministic_faithful = [
        d for d in faithful_decodes if per_decode[d]["deterministic"]
    ]
    argmax_collapsed = (
        per_decode["argmax"]["render_faithfulness"]["verdict"] == "COLLAPSED"
        or per_decode["argmax"]["real_distortion_net"]["mean_d_seg"] >= chance - 0.005
    )

    if deterministic_faithful and argmax_collapsed:
        verdict = "EVAL_PATH_BUG"
        verdict_detail = (
            f"deterministic faithful decode(s) {deterministic_faithful} render in "
            f"GT range with d_seg < chance while argmax collapses; the inflate "
            f"render path should switch to a deterministic faithful decode "
            f"(prefer 'expected_value' — numpy-portable, RNG-free)."
        )
    elif not faithful_decodes:
        verdict = "CAPACITY_ISSUE"
        verdict_detail = (
            "ALL decodes collapse (recon out of GT range and/or d_seg >= chance). "
            "The trained weights never learned a faithful render at this sizing. "
            "Per Catalog #307 this is IMPLEMENTATION-LEVEL (scaffold sizing), not a "
            "paradigm kill."
        )
    elif faithful_decodes and not deterministic_faithful:
        verdict = "STOCHASTIC_ONLY_FAITHFUL"
        verdict_detail = (
            f"only stochastic decode(s) {faithful_decodes} are faithful; no "
            f"deterministic decode qualifies. A random-Gumbel inflate is NOT "
            f"contest-valid (Catalog #295). Faithful render exists but is not "
            f"contest-deployable as-is."
        )
    else:
        # faithful exists, argmax also faithful (collapse not reproduced)
        verdict = "NO_COLLAPSE_REPRODUCED"
        verdict_detail = (
            "argmax did not collapse on this checkpoint; no eval-path gap to fix."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "gumbel_vs_argmax_eval_render_gap_audit_v1",
        "evidence_grade": "[macOS-MLX research-signal]",
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "captured_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "substrate": substrate_label,
        "mode": mode,
        "arm_dir": str(arm_dir.relative_to(REPO_ROOT)),
        "ema_shadow_npsd": str(npsd.relative_to(REPO_ROOT)),
        "substrate_config": substrate_config,
        "num_pairs": int(num_pairs),
        "gumbel_seeds": int(gumbel_seeds),
        "low_temp": float(low_temp),
        "chance_d_seg": chance,
        "no_fake_identity_guard": {
            "d_seg_gt_gt": identity_d_seg,
            "d_pose_gt_gt": identity_d_pose,
        },
        "real_teacher": (
            "contest SegNet (smp.Unet EfficientNet-B2) + PoseNet (fastvit_t12) "
            "on upstream/videos/0.mkv; real upstream.modules.DistortionNet (CPU)"
        ),
        "measurement_functional": (
            "d_seg = (SegNet(GT).argmax != SegNet(recon).argmax).mean() "
            "[upstream/modules.py:112]; recon = per-decode render of trained "
            "EMA-shadow Z8 feeding cat_to_continuous Linear; the four decodes "
            "differ ONLY in the categorical representation fed in (argmax one-hot "
            "/ softmax simplex / Gumbel STE one-hot / low-temp Gumbel)."
        ),
        "per_decode": per_decode,
        "faithful_decodes": faithful_decodes,
        "deterministic_faithful_decodes": deterministic_faithful,
        "argmax_collapsed": bool(argmax_collapsed),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }

    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    # Attach canonical Provenance (hash the file written above).
    result["provenance"] = _provenance_for(result_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[gap-audit] VERDICT={verdict}", flush=True)
    print(f"[gap-audit] wrote {result_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("z8", "dreamer"), default="z8")
    ap.add_argument(
        "--arm-dir",
        default="experiments/results/z8_seg_lever_faithful_render_confirm/kl_t2",
        help="trained arm dir containing checkpoints/final_*.ema_shadow.state.npsd",
    )
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--gumbel-seeds", type=int, default=4)
    ap.add_argument("--low-temp", type=float, default=0.1)
    ap.add_argument(
        "--out-dir",
        default="experiments/results/gumbel_vs_argmax_eval_render_gap_audit/z8_kl_t2",
    )
    ap.add_argument("--substrate-label", default="z8_hierarchical_predictive_coding")
    args = ap.parse_args(argv)

    arm_dir = (REPO_ROOT / args.arm_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    run_audit(
        mode=str(args.mode),
        arm_dir=arm_dir,
        num_pairs=int(args.num_pairs),
        gumbel_seeds=int(args.gumbel_seeds),
        low_temp=float(args.low_temp),
        out_dir=out_dir,
        substrate_label=str(args.substrate_label),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
