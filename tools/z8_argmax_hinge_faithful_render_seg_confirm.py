# SPDX-License-Identifier: MIT
"""Confirm the boundary_argmax_hinge seg lever on Z8's FAITHFUL render.

WAVE-1C2 2026-05-31 (sister of subagent C's DreamerV3 in-substrate confirm at
``tools/dreamer_v3_argmax_hinge_in_substrate_seg_confirm.py`` commit
``153ece040`` + subagent A's per-pixel-logit-field A/B at
``tools/ab_boundary_tckd_vs_kl_t2.py`` commit ``030367a9c``).

## The gap this closes

Sister A's standalone A/B proved the seg-distill OBJECTIVE mechanism on an
UNBOUNDED per-pixel logit field (``boundary_argmax_hinge`` drove a free student
proxy's ``d_seg -> 0.0`` robustly; soft ``boundary_tckd`` was fragile /
falsified at the 8% bar). Sister C/C' tried to confirm it IN-SUBSTRATE on the
capacity-limited DreamerV3 renderer but the test was INCONCLUSIVE: the
DreamerV3 ``_full_main`` render COLLAPSED to a near-constant white frame
(``recon_mean`` 254.5 vs GT 26.1; all arms incl. the ``kl_t2`` control hit
``d_seg`` 0.505906 = chance). The collapse is the base render config, NOT the
seg objective — so the lever could not be tested.

Z8's ``_full_main`` produces a FAITHFUL render: its 3-level Rao-Ballard +
DreamerV3 categorical posterior + HNeRV decoder solved pose to ~0.067 and its
seg loss was still descending (a real reconstruction, NOT chance). So Z8 is the
right testbed for the seg lever.

## What this runner measures (NO FAKE — the EXACT contest functionals)

For each of two arms (``kl_t2`` control + ``boundary_argmax_hinge`` candidate):

1. Run the canonical Z8 ``_full_main``
   (``experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py``)
   via subprocess at MATCHED ``--epochs`` / ``--num-pairs`` / ``--seed`` /
   ``--full-lr`` + the Wave N+11 stabilizer (grad-clip + warmup + weight-decay +
   AdamW) with the REAL SegNet (KL T=2.0) + REAL PoseNet (pose-MSE) Hinton-
   distilled teachers (``--allow-mock-scorer-teacher`` is NEVER passed, so the
   pose axis is real per the sister Z7-Mamba-2 / real-Hinton pattern). Both arms
   differ ONLY in ``--seg-distill-objective`` (+ ``--seg-hinge-margin`` /
   ``--seg-tau-boundary`` for the hinge arm).
2. Load the arm's EMA-shadow trained weights
   (``checkpoints/final_*.ema_shadow.state.npsd``) into a fresh
   ``Z8HierarchicalPredictiveCoderMLX`` (the SAME ``_full_main`` config sizing)
   and render the DETERMINISTIC argmax eval path (per-level
   ``argmax(logits_per_level[l][pair]) -> forward_eval_from_indices``) for every
   pair -> reconstruction ``(P, 2, 3, 384, 512)`` in ``[0, 255]``.
3. Decode the SAME real GT frames the trainer saw (``decode_video`` at 384×512,
   adjacent-frame pairs) and run the REAL ``upstream.modules.DistortionNet`` on
   ``(gt, recon)``:
     * ``d_seg`` = ``(SegNet(GT).argmax != SegNet(recon).argmax).float().mean()``
       — the EXACT ``upstream/modules.py:112`` SegNet.compute_distortion.
     * ``d_pose`` = PoseNet first-half pose-MSE — the EXACT
       ``upstream/modules.py:84`` PoseNet.compute_distortion.

The measurement is genuine by construction: ``compute_distortion(gt, gt) == 0``
(verified), so a constant / unchanged renderer output yields exactly 0 and a
real rendered frame yields the real argmax-flip rate. A measurement that would
be identical with the renderer output replaced by a constant is FORBIDDEN per
CLAUDE.md "NO FAKE IMPLEMENTATIONS"; this is not that — the d_seg responds to
the actual rendered pixels. We ALSO assert render-faithfulness (recon_mean /
recon_std vs GT range; not collapsed) — if Z8 ALSO collapses, the runner
reports INCONCLUSIVE-IF-COLLAPSED honestly (the lever still can't be tested).

## Falsifiable claim (Catalog #307 paradigm-vs-implementation)

On Z8's faithful render, ``boundary_argmax_hinge`` reduces REAL post-training
SegNet ``d_seg`` BELOW the ``kl_t2`` baseline at matched epochs WITHOUT pose
regression OR render collapse.
  * YES   -> PARADIGM-VALIDATED: the seg lever genuinely works on a faithful
    render; advances sister A's proposed canonical equation
    ``d_seg_faithful_seg_distill_argmax_hinge_dominates_soft_kd_v1``.
  * NO    -> IMPLEMENTATION-LEVEL-FALSIFIED: the hinge objective did not transfer
    through the categorical-posterior -> HNeRV decoder to lower the real SegNet
    argmax-flip rate. The objective mechanism (sister A) is NOT killed.
  * COLLAPSE -> INCONCLUSIVE-IF-COLLAPSED: the Z8 render also collapsed under
    this run config; the lever cannot be tested. Reported honestly.

## Custody (NON-NEGOTIABLE per Catalog #192/#341/#127/#323)

MLX-LOCAL $0 (M5 Max); NO Modal/Vast/Lightning/HF Jobs dispatch. Every persisted
score row carries ``evidence_grade="[macOS-MLX research-signal]"`` +
``score_claim=false`` + ``promotion_eligible=false`` +
``ready_for_exact_eval_dispatch=false`` + canonical Provenance via
``tac.provenance.build_provenance_for_macos_mlx_research_signal``. The d_seg
numbers here are a TRAINING-DYNAMICS research signal on the contest SegNet
functional, NOT a contest-score claim. Paired Linux x86_64 [contest-CPU] +
[contest-CUDA] replay on a byte-closed archive is the canonical promotion path
and is DEFERRED (operator-funded ratification op-routable, NOT fired here).

[verified-against: upstream/modules.py:112 SegNet.compute_distortion argmax-flip rate]
[verified-against: upstream/modules.py:84 PoseNet.compute_distortion first-half MSE]
[verified-against: tools/dreamer_v3_argmax_hinge_in_substrate_seg_confirm.py sister C measurement methodology]
[verified-against: experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py _full_main seg-distill flag + config sizing]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
UPSTREAM = REPO_ROOT / "upstream"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CANONICAL_NON_PROMOTABLE = {
    "evidence_grade": "[macOS-MLX research-signal]",
    "axis_tag": "[macOS-MLX research-signal]",
    "score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
}

# Canonical Z8 _full_main config sizing (mirrors
# experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py:606-616).
# Eval-render reconstruction depends on these EXACTLY matching the trained model.
Z8_FULL_MAIN_CONFIG = {
    "num_levels": 3,
    "num_groups_per_level": (4, 3, 2),
    "num_categories_per_level": (16, 8, 4),
    "base_channels": 8,
    "decoder_latent_dim": 12,
    "deterministic_state_dim": 8,
}


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat()


# ---------------------------------------------------------------------------
# Arm training (subprocess, fully isolated per arm)
# ---------------------------------------------------------------------------


def _train_arm(
    *,
    objective: str,
    out_dir: Path,
    epochs: int,
    num_pairs: int,
    seed: int,
    full_lr: float,
    hinge_margin: float,
    seg_tau_boundary: float,
    video_path: str,
    grad_clip_max_norm: float,
    warmup_epochs: int,
    weight_decay: float,
    extra_args: list[str] | None = None,
) -> dict:
    """Run Z8 _full_main for one arm via subprocess; return run metadata.

    Z8 ``main()`` dispatches ``--smoke -> _smoke_main``; ``--canonical-quadruple-
    binding -> quadruple``; else ``_full_main`` (there is NO ``--full`` flag).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = (
        REPO_ROOT
        / "experiments"
        / "train_substrate_z8_hierarchical_predictive_coding_mlx.py"
    )
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(trainer),
        "--output-dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--num-pairs",
        str(num_pairs),
        "--seed",
        str(seed),
        "--video-path",
        video_path,
        "--full-lr",
        str(full_lr),
        "--seg-distill-objective",
        objective,
        "--seg-hinge-margin",
        str(hinge_margin),
        "--seg-tau-boundary",
        str(seg_tau_boundary),
        # REAL teacher: --allow-mock-scorer-teacher is NEVER passed (default off
        # builds the real SegNet + PoseNet teachers per the trainer contract).
        # distillation_weight > 0 so the seg-distill term is live; pose weight
        # 1.0 keeps the dominant pose axis bound (no pose-drift failure).
        "--distillation-weight",
        "1.0",
        "--pose-distillation-weight",
        "1.0",
        # Wave N+11 stabilizer (matched across both arms; the only difference is
        # --seg-distill-objective).
        "--grad-clip-max-norm",
        str(grad_clip_max_norm),
        "--warmup-epochs",
        str(warmup_epochs),
        "--weight-decay",
        str(weight_decay),
        "--optimizer-kind",
        "adamw",
    ]
    if extra_args:
        cmd.extend(extra_args)
    t0 = time.perf_counter()
    import os

    full_env = dict(os.environ)
    full_env.update({"PYTHONPATH": f"{SRC}:{UPSTREAM}"})
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=full_env,
    )
    wall = time.perf_counter() - t0
    tail = (
        (proc.stdout or "")[-2000:]
        + "\n--- stderr ---\n"
        + (proc.stderr or "")[-2000:]
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"arm '{objective}' training rc={proc.returncode}\n{tail}"
        )
    return {
        "objective": objective,
        "returncode": proc.returncode,
        "wall_clock_s": round(wall, 1),
        "stdout_tail": (proc.stdout or "")[-800:],
        "output_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# Load trained EMA-shadow renderer + deterministic argmax render
# ---------------------------------------------------------------------------


def _find_ema_shadow_npsd(out_dir: Path) -> Path:
    ckpt = out_dir / "checkpoints"
    cands = sorted(ckpt.glob("final_*.ema_shadow.state.npsd"))
    if not cands:
        # fall back to latest non-final ema shadow
        cands = sorted(ckpt.glob("*.ema_shadow.state.npsd"))
    if not cands:
        raise FileNotFoundError(
            f"no EMA-shadow .npsd checkpoint found under {ckpt}"
        )
    return cands[-1]


def _build_z8_model(num_pairs: int):
    """Build a fresh Z8HierarchicalPredictiveCoderMLX matching _full_main sizing."""
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
        deterministic_state_dim=int(
            Z8_FULL_MAIN_CONFIG["deterministic_state_dim"]
        ),
        gumbel_temperature=1.0,
        use_straight_through=True,
    )
    return Z8HierarchicalPredictiveCoderMLX(cfg), cfg


def _load_trained_model(out_dir: Path, *, num_pairs: int):
    """Load the EMA-shadow trained Z8 model from its .npsd checkpoint."""
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


def _render_all_pairs(model, cfg, num_pairs: int):
    """Deterministic per-level argmax eval render for all pairs.

    Returns numpy (P, 2, H, W, 3) [0,255]. Mirrors the Z8 ``forward_eval_from_
    indices`` contract: per level take ``argmax(logits_per_level[l][pair])`` ->
    ``(1, G_l)`` int32; the list of per-level indices decodes WITHOUT Gumbel.
    """
    import mlx.core as mx
    import numpy as np

    L = int(cfg.num_levels)
    recon = []
    for p in range(num_pairs):
        per_level_indices = []
        for level_idx in range(L):
            # logits_per_level[level_idx] is (num_pairs, G_l, K_l); take pair p.
            level_logits = model.logits_per_level[level_idx][p : p + 1]  # (1, G_l, K_l)
            idx = mx.argmax(level_logits, axis=-1)  # (1, G_l)
            per_level_indices.append(idx)
        rgb = model.forward_eval_from_indices(per_level_indices)  # (1, 2, 3, H, W) [0,255]
        mx.eval(rgb)
        arr = np.asarray(rgb)[0]  # (2, 3, H, W)
        arr = np.transpose(arr, (0, 2, 3, 1))  # (2, H, W, 3)
        recon.append(arr)
    return np.stack(recon, axis=0).astype(np.float32)  # (P, 2, H, W, 3)


# ---------------------------------------------------------------------------
# Real contest DistortionNet on rendered vs GT
# ---------------------------------------------------------------------------


def _decode_gt_pairs(video_path: str, num_pairs: int):
    """Decode the SAME GT frames the trainer saw: (P, 2, 384, 512, 3) float [0,255]."""
    import numpy as np

    from tac.data import decode_video

    frames = decode_video(
        video_path, target_h=384, target_w=512, max_frames=2 * num_pairs
    )
    if len(frames) < 2 * num_pairs:
        raise RuntimeError(
            f"decoded {len(frames)} frames; need {2 * num_pairs} for "
            f"{num_pairs} pairs"
        )
    gt = np.stack([f.numpy() for f in frames[: 2 * num_pairs]], axis=0)
    return gt.reshape(num_pairs, 2, 384, 512, 3).astype(np.float32)


def _real_distortion_net():
    """Load the REAL contest DistortionNet (SegNet + PoseNet) on CPU."""
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


def _measure_real_d_seg_d_pose(dn, gt_pairs, recon_pairs, *, batch: int = 32) -> dict:
    """Run the REAL DistortionNet -> per-pair (d_pose, d_seg); return aggregates.

    gt_pairs / recon_pairs: numpy (P, 2, H, W, 3) float in [0, 255].
    DistortionNet.preprocess_input expects (B, seq_len=2, H, W, C).
    """
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
    # Contest score components (evaluate.py:92): 100*d_seg + sqrt(10*d_pose).
    # Rate term omitted (no byte-closed archive here; research signal only).
    mean_d_seg = float(d_seg_arr.mean())
    mean_d_pose = float(d_pose_arr.mean())
    return {
        "mean_d_seg": mean_d_seg,
        "mean_d_pose": mean_d_pose,
        "max_d_seg": float(d_seg_arr.max()),
        "min_d_seg": float(d_seg_arr.min()),
        "max_d_pose": float(d_pose_arr.max()),
        "n_pairs": int(P),
        "score_seg_pose_partial_no_rate": float(
            100.0 * mean_d_seg + (10.0 * mean_d_pose) ** 0.5
        ),
    }


# ---------------------------------------------------------------------------
# Render-faithfulness check (NOT collapsed)
# ---------------------------------------------------------------------------


def _render_faithfulness(recon_pairs, gt_pairs) -> dict:
    """Assess whether the render is FAITHFUL (in GT distribution range) or COLLAPSED.

    Collapse heuristic (mirrors sister C's renderer-collapse confound: DreamerV3
    collapsed to recon_mean=254.5 / recon_std=1.5 vs GT mean=26.1 / std=21.3).
    A render is judged COLLAPSED if EITHER (a) its std is < 10% of the GT std
    (near-constant frame) OR (b) its mean is > 3x or < 1/3 the GT mean
    (saturated toward white/black). Faithful otherwise.
    """
    import numpy as np

    gt_mean = float(np.mean(gt_pairs))
    gt_std = float(np.std(gt_pairs))
    rec_mean = float(np.mean(recon_pairs))
    rec_std = float(np.std(recon_pairs))
    std_ratio = rec_std / gt_std if gt_std > 0 else 0.0
    mean_ratio = rec_mean / gt_mean if gt_mean > 0 else float("inf")
    near_constant = std_ratio < 0.10
    saturated = mean_ratio > 3.0 or mean_ratio < (1.0 / 3.0)
    collapsed = bool(near_constant or saturated)
    return {
        "gt_mean": gt_mean,
        "gt_std": gt_std,
        "recon_mean": rec_mean,
        "recon_std": rec_std,
        "std_ratio_recon_over_gt": std_ratio,
        "mean_ratio_recon_over_gt": mean_ratio,
        "near_constant": bool(near_constant),
        "saturated": bool(saturated),
        "collapsed": collapsed,
        "faithful": (not collapsed),
    }


# ---------------------------------------------------------------------------
# Arm measurement (load + render + real distortion + faithfulness)
# ---------------------------------------------------------------------------


def _measure_arm(out_dir: Path, *, num_pairs: int, dn, gt_pairs) -> dict:
    model, cfg, npsd = _load_trained_model(out_dir, num_pairs=num_pairs)
    recon = _render_all_pairs(model, cfg, num_pairs)
    metrics = _measure_real_d_seg_d_pose(dn, gt_pairs, recon)
    metrics["ema_shadow_npsd"] = str(npsd.relative_to(REPO_ROOT))
    # NO-FAKE invariant: recon mean/std (proves the render is not a constant).
    import numpy as np

    metrics["recon_mean"] = float(recon.mean())
    metrics["recon_std"] = float(recon.std())
    metrics["recon_is_nonconstant"] = bool(np.std(recon) > 1e-3)
    metrics["render_faithfulness"] = _render_faithfulness(recon, gt_pairs)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Confirm the boundary_argmax_hinge seg-distill lever on Z8's "
            "FAITHFUL render (REAL post-training SegNet d_seg vs kl_t2)."
        )
    )
    ap.add_argument(
        "--epochs", type=int, default=2000,
        help="Matched training epochs per arm (Z8 canonical curriculum = 2000).",
    )
    ap.add_argument(
        "--num-pairs", type=int, default=600,
        help="Real GT pairs (full contest=600).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-lr", type=float, default=0.001)
    ap.add_argument(
        "--hinge-margins", type=str, default="1.0",
        help="Comma-separated hinge margins to sweep for the "
        "boundary_argmax_hinge arm (e.g. '0.5,1.0').",
    )
    ap.add_argument(
        "--seg-tau-boundary", type=float, default=2.0,
        help="Boundary-band temperature (A/B found 2.0 robust).",
    )
    ap.add_argument("--grad-clip-max-norm", type=float, default=1.0)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--video", default="upstream/videos/0.mkv")
    ap.add_argument(
        "--out-root", type=Path,
        default=Path("experiments/results/z8_seg_lever_faithful_render_confirm"),
    )
    ap.add_argument(
        "--reuse-existing", action="store_true",
        help="Skip training arms whose EMA-shadow checkpoint already exists "
        "(measure-only).",
    )
    args = ap.parse_args(argv)

    margins = [float(x) for x in str(args.hinge_margins).split(",") if x.strip()]
    started = _utc_now()
    out_root = (
        (REPO_ROOT / args.out_root)
        if not args.out_root.is_absolute()
        else args.out_root
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Train all arms (control kl_t2 + one hinge arm per margin) ---
    arm_specs: list[tuple[str, str, float]] = [("kl_t2", "kl_t2", 1.0)]
    for m in margins:
        arm_specs.append((f"argmax_hinge_m{m:g}", "boundary_argmax_hinge", m))

    train_meta: dict[str, dict] = {}
    for arm_name, objective, margin in arm_specs:
        arm_dir = out_root / arm_name
        ckpt_dir = arm_dir / "checkpoints"
        npsd_exists = (
            bool(list(ckpt_dir.glob("final_*.ema_shadow.state.npsd")))
            if ckpt_dir.exists()
            else False
        )
        if args.reuse_existing and npsd_exists:
            train_meta[arm_name] = {
                "objective": objective, "reused": True,
                "output_dir": str(arm_dir),
            }
            print(f"[{arm_name}] reuse-existing: skipping training")
            continue
        print(
            f"[{arm_name}] training objective={objective} margin={margin} "
            f"epochs={args.epochs} pairs={args.num_pairs} ..."
        )
        meta = _train_arm(
            objective=objective,
            out_dir=arm_dir,
            epochs=int(args.epochs),
            num_pairs=int(args.num_pairs),
            seed=int(args.seed),
            full_lr=float(args.full_lr),
            hinge_margin=float(margin),
            seg_tau_boundary=float(args.seg_tau_boundary),
            video_path=str(args.video),
            grad_clip_max_norm=float(args.grad_clip_max_norm),
            warmup_epochs=int(args.warmup_epochs),
            weight_decay=float(args.weight_decay),
        )
        train_meta[arm_name] = meta
        print(f"[{arm_name}] trained in {meta['wall_clock_s']}s")

    # --- Measure REAL post-training d_seg / d_pose for every arm ---
    print("[measure] loading real DistortionNet + GT frames ...")
    dn = _real_distortion_net()
    gt_pairs = _decode_gt_pairs(str(args.video), int(args.num_pairs))

    # NO-FAKE identity guard: compute_distortion(gt, gt) MUST be exactly 0.
    identity = _measure_real_d_seg_d_pose(dn, gt_pairs, gt_pairs)
    if identity["mean_d_seg"] != 0.0 or identity["mean_d_pose"] != 0.0:
        raise RuntimeError(
            "NO-FAKE identity guard FAILED: compute_distortion(gt, gt) must be "
            f"0.0 but got d_seg={identity['mean_d_seg']} "
            f"d_pose={identity['mean_d_pose']}. The measurement is not the "
            "contest functional."
        )
    print(
        "[measure] NO-FAKE identity guard PASS: compute_distortion(gt, gt) == 0.0"
    )

    arm_metrics: dict[str, dict] = {}
    for arm_name, objective, margin in arm_specs:
        arm_dir = out_root / arm_name
        print(f"[measure] {arm_name} ...")
        metrics = _measure_arm(
            arm_dir, num_pairs=int(args.num_pairs), dn=dn, gt_pairs=gt_pairs
        )
        metrics["objective"] = objective
        metrics["hinge_margin"] = margin
        arm_metrics[arm_name] = metrics
        ff = metrics["render_faithfulness"]
        print(
            f"[measure] {arm_name}: d_seg={metrics['mean_d_seg']:.6f} "
            f"d_pose={metrics['mean_d_pose']:.6e} "
            f"recon_mean={ff['recon_mean']:.1f} recon_std={ff['recon_std']:.1f} "
            f"faithful={ff['faithful']}"
        )

    # --- Verdict ---
    control = arm_metrics["kl_t2"]
    control_d_seg = control["mean_d_seg"]
    control_d_pose = control["mean_d_pose"]
    hinge_arms = {k: v for k, v in arm_metrics.items() if k != "kl_t2"}
    best_hinge_name = min(hinge_arms, key=lambda k: hinge_arms[k]["mean_d_seg"])
    best_hinge = hinge_arms[best_hinge_name]
    best_d_seg = best_hinge["mean_d_seg"]
    best_d_pose = best_hinge["mean_d_pose"]

    # Render-faithfulness gate: if EITHER arm collapsed, the lever can't be
    # tested -> INCONCLUSIVE-IF-COLLAPSED (sister C/C' confound).
    control_faithful = bool(control["render_faithfulness"]["faithful"])
    best_faithful = bool(best_hinge["render_faithfulness"]["faithful"])
    any_collapsed = (not control_faithful) or (not best_faithful)

    rel_d_seg_reduction = (
        (control_d_seg - best_d_seg) / control_d_seg if control_d_seg > 0 else 0.0
    )
    pose_rel_change = (
        (best_d_pose - control_d_pose) / control_d_pose
        if control_d_pose > 0
        else 0.0
    )
    pose_regressed = pose_rel_change > 0.05
    seg_improved = best_d_seg < control_d_seg

    if any_collapsed:
        verdict = "INCONCLUSIVE-IF-COLLAPSED"
        collapsed_arms = [
            k for k, v in arm_metrics.items()
            if not v["render_faithfulness"]["faithful"]
        ]
        verdict_detail = (
            "the Z8 render COLLAPSED under this run config so the seg lever "
            f"cannot be tested honestly (collapsed arms: {collapsed_arms}). "
            "Same confound class as sister C/C' DreamerV3 (renderer collapse, "
            "NOT seg-objective signal). recon_mean/std are out of GT range. "
            "Reactivation = larger epoch budget / different render config that "
            "produces a faithful reconstruction; the lever mechanism (sister A) "
            "is NOT killed."
        )
    elif seg_improved and not pose_regressed:
        verdict = "PARADIGM-VALIDATED"
        verdict_detail = (
            f"on Z8's FAITHFUL render, boundary_argmax_hinge ({best_hinge_name}) "
            f"lowered REAL post-training SegNet d_seg {control_d_seg:.6f} -> "
            f"{best_d_seg:.6f} ({rel_d_seg_reduction:+.1%}) WITHOUT pose "
            f"regression (d_pose {control_d_pose:.4e} -> {best_d_pose:.4e}, "
            f"{pose_rel_change:+.1%}) and WITHOUT render collapse "
            f"(recon_mean={best_hinge['render_faithfulness']['recon_mean']:.1f} "
            f"vs GT {control['render_faithfulness']['gt_mean']:.1f}). The seg "
            f"lever genuinely transfers from sister A's unbounded per-pixel proxy "
            f"to the capacity-limited Z8 categorical-posterior + HNeRV renderer."
        )
    else:
        verdict = "IMPLEMENTATION-LEVEL-FALSIFIED"
        why = []
        if not seg_improved:
            why.append(
                f"d_seg did NOT improve ({control_d_seg:.6f} -> "
                f"{best_d_seg:.6f}); the hinge gradient (sparse, non-smooth) did "
                "not transfer through the distilled student-head -> categorical "
                "posterior -> HNeRV decoder to lower the real SegNet argmax-flip "
                "rate on Z8's faithful render"
            )
        if pose_regressed:
            why.append(
                f"pose REGRESSED {pose_rel_change:+.1%} "
                f"(d_pose {control_d_pose:.4e} -> {best_d_pose:.4e})"
            )
        verdict_detail = (
            "boundary_argmax_hinge did not dominate kl_t2 on Z8's faithful "
            "render: " + "; ".join(why) + ". The objective MECHANISM (sister A's "
            "proxy A/B) is NOT killed — only this specific substrate-render "
            "coupling is IMPLEMENTATION-LEVEL falsified per Catalog #307; "
            "reactivation = larger epoch budget / full curriculum / joint "
            "hinge+kl warmup."
        )

    result = {
        "schema": "z8_argmax_hinge_faithful_render_seg_confirm_v1",
        **CANONICAL_NON_PROMOTABLE,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "substrate": "z8_hierarchical_predictive_coding",
        "real_teacher": (
            "contest SegNet (smp.Unet EfficientNet-B2) + PoseNet "
            "(fastvit_t12) on upstream/videos/0.mkv"
        ),
        "real_distortion_net": "upstream.modules.DistortionNet (CPU)",
        "measurement_functional": (
            "d_seg = (SegNet(GT).argmax != SegNet(recon).argmax).mean() "
            "[upstream/modules.py:112]; d_pose = PoseNet first-half MSE "
            "[upstream/modules.py:84]; recon = DETERMINISTIC per-level argmax "
            "eval render of EMA-shadow trained Z8 "
            "(argmax(logits_per_level) -> forward_eval_from_indices)"
        ),
        "no_fake_identity_guard": "compute_distortion(gt, gt) == 0.0 (verified)",
        "z8_full_main_config": Z8_FULL_MAIN_CONFIG,
        "epochs": int(args.epochs),
        "num_pairs": int(args.num_pairs),
        "seed": int(args.seed),
        "full_lr": float(args.full_lr),
        "seg_tau_boundary": float(args.seg_tau_boundary),
        "hinge_margins_swept": margins,
        "wave_n11_stabilizer": {
            "grad_clip_max_norm": float(args.grad_clip_max_norm),
            "warmup_epochs": int(args.warmup_epochs),
            "weight_decay": float(args.weight_decay),
            "optimizer": "adamw",
            "ema_decay": 0.997,
        },
        "train_meta": train_meta,
        "arm_metrics": arm_metrics,
        "control_arm": "kl_t2",
        "control_d_seg": control_d_seg,
        "control_d_pose": control_d_pose,
        "control_faithful": control_faithful,
        "best_hinge_arm": best_hinge_name,
        "best_hinge_d_seg": best_d_seg,
        "best_hinge_d_pose": best_d_pose,
        "best_hinge_faithful": best_faithful,
        "any_arm_collapsed": any_collapsed,
        "relative_d_seg_reduction_hinge_vs_kl_t2": rel_d_seg_reduction,
        "pose_relative_change_hinge_vs_kl_t2": pose_rel_change,
        "pose_regressed": pose_regressed,
        "seg_improved": seg_improved,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "canonical_equation_status": (
            "ADVANCES d_seg_faithful_seg_distill_argmax_hinge_dominates_soft_kd_v1"
            if verdict == "PARADIGM-VALIDATED"
            else "FALSIFIES d_seg_faithful_seg_distill_argmax_hinge_dominates_soft_kd_v1 "
            "at the Z8 in-domain context (IMPLEMENTATION-LEVEL)"
            if verdict == "IMPLEMENTATION-LEVEL-FALSIFIED"
            else "PENDING: Z8 render collapsed; lever untestable this run "
            "(d_seg_faithful_seg_distill_argmax_hinge_dominates_soft_kd_v1 "
            "remains FORMALIZATION_PENDING)"
        ),
        "paired_cuda_ratification_op_routable": (
            "OPERATOR-FUNDED (<=$20, DO NOT FIRE): if PARADIGM-VALIDATED, build a "
            "byte-closed Z8 archive from the boundary_argmax_hinge EMA shadow, run "
            "paired [contest-CPU] (GHA Linux x86_64) + [contest-CUDA] (T4) auth-"
            "eval per Catalog #246 to ratify the real-d_seg reduction translates "
            "to a real contest score delta. NOTE: Z8's contest archive is a "
            "classical Mallat wavelet codec that does NOT consume the trained "
            "categorical-posterior renderer weights (per the Z8 export-bridge "
            "memo); the seg lever is a TRAINING-DYNAMICS signal until the "
            "renderer-weight-consuming inflate path lands."
        ),
    }

    # Canonical Provenance: write the result first, hash it, then patch the
    # provenance field referencing the artifact's own sha256 (non-promotable
    # macOS-MLX research-signal grade per Catalog #192/#341/#323).
    import hashlib

    from tac.provenance import build_provenance_for_macos_mlx_research_signal

    out_json = out_root / "result.json"
    body_no_prov = json.dumps(result, indent=2, sort_keys=False) + "\n"
    artifact_sha = hashlib.sha256(body_no_prov.encode("utf-8")).hexdigest()
    try:
        rel_source = str(out_json.relative_to(REPO_ROOT))
    except ValueError:
        rel_source = str(out_json)
    prov = build_provenance_for_macos_mlx_research_signal(
        artifact_sha256=artifact_sha,
        source_path=rel_source,
        captured_at_utc=_utc_now(),
    )
    result["provenance"] = (
        prov.to_dict() if hasattr(prov, "to_dict") else str(prov)
    )
    result["result_body_sha256_pre_provenance"] = artifact_sha
    out_json.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")

    # human-readable verdict.
    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    cf = control["render_faithfulness"]
    bf = best_hinge["render_faithfulness"]
    print(
        f"  kl_t2 (control)         d_seg={control_d_seg:.6f} "
        f"d_pose={control_d_pose:.4e} "
        f"recon_mean={cf['recon_mean']:.1f}/std={cf['recon_std']:.1f} "
        f"faithful={cf['faithful']}"
    )
    print(
        f"  {best_hinge_name} (best hinge) d_seg={best_d_seg:.6f} "
        f"d_pose={best_d_pose:.4e} "
        f"recon_mean={bf['recon_mean']:.1f}/std={bf['recon_std']:.1f} "
        f"faithful={bf['faithful']}"
    )
    print(
        f"  GT distribution: mean={cf['gt_mean']:.1f}/std={cf['gt_std']:.1f}"
    )
    print(
        f"  relative d_seg reduction (hinge vs kl_t2): {rel_d_seg_reduction:+.1%}"
    )
    print(
        f"  pose relative change (hinge vs kl_t2):     {pose_rel_change:+.1%} "
        f"(regressed={pose_regressed})"
    )
    print(f"  detail: {verdict_detail}")
    print(f"\nwrote {out_json.relative_to(REPO_ROOT)}")
    print("[macOS-MLX research-signal] — NON-PROMOTABLE per Catalog #192/#341")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
