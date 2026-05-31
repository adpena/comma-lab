# SPDX-License-Identifier: MIT
"""In-substrate contest-faithful confirmation of the boundary_argmax_hinge seg lever.

WAVE-1 SUBAGENT C 2026-05-31 (sister of subagent A's per-pixel-logit-field
STUDENT-PROXY A/B at ``tools/ab_boundary_tckd_vs_kl_t2.py`` commit ``030367a9c``).

## The gap this closes

Sister A's A/B proved the seg-distill OBJECTIVE mechanism by optimizing a
STANDALONE learnable per-pixel logit field initialized near the teacher
(``init = teacher + noise``). That field has UNBOUNDED per-pixel degrees of
freedom, so ``boundary_argmax_hinge`` could drive its ``d_seg -> 0.0`` trivially.
The contest-faithful question A could NOT answer with a proxy: **does a real
capacity-limited DreamerV3 renderer** (per-pair 24×256 categorical posterior +
HNeRV decoder; the seg objective reaches it ONLY through the distilled
student-head surrogate gradient) trained with ``boundary_argmax_hinge`` actually
LOWER the REAL post-training SegNet ``d_seg`` vs ``kl_t2`` at matched epochs,
WITHOUT pose regression?

## What this runner measures (NO FAKE — the EXACT contest functionals)

For each of two arms (``kl_t2`` control + ``boundary_argmax_hinge`` candidate):

1. Run the canonical DreamerV3 ``_full_main`` (``experiments/train_substrate_dreamer_v3_rssm.py``)
   via subprocess at MATCHED ``--epochs`` / ``--num-pairs`` / ``--seed`` with the
   REAL SegNet (KL T=2.0) + REAL PoseNet (pose-MSE) Hinton-distilled teachers
   (the ``--allow-mock-scorer-teacher`` flag is NEVER passed, so pose-axis is
   real per the sister Z7-Mamba-2 / real-Hinton pattern). Both arms differ ONLY
   in ``--seg-distill-objective``.
2. Load the arm's EMA-shadow trained weights (``checkpoints/final_*.ema_shadow.state.npsd``)
   into a fresh ``DreamerV3RSSMSubstrateMLX`` and render the DETERMINISTIC
   argmax eval path (``argmax(logits) -> forward_eval_from_indices``) for every
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
the actual rendered pixels.

## Falsifiable claim (Catalog #307 paradigm-vs-implementation)

``boundary_argmax_hinge`` reduces REAL post-training SegNet ``d_seg`` vs
``kl_t2`` at matched epochs WITHOUT pose regression.
  * YES  -> PARADIGM-VALIDATED: the objective transfers from the unbounded
    proxy to the capacity-limited renderer; advances sister A's proposed
    canonical equation ``d_seg_faithful_seg_distill_argmax_hinge_dominates_soft_kd_v1``.
  * NO   -> IMPLEMENTATION-LEVEL-FALSIFIED: the proxy did not transfer to the
    capacity-limited renderer (e.g. hinge gradient too sparse for the categorical
    posterior to exploit; non-smooth optimum unreachable through the HNeRV
    decoder). The objective mechanism (sister A) is NOT killed — only this
    specific substrate-render coupling is falsified.

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
[verified-against: tools/ab_boundary_tckd_vs_kl_t2.py sister A canonical d_seg formula]
[verified-against: experiments/train_substrate_dreamer_v3_rssm.py _full_main seg-distill flag]
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
    hinge_margin: float,
    seg_tau_boundary: float,
    video_path: str,
    extra_args: list[str] | None = None,
) -> dict:
    """Run DreamerV3 _full_main for one arm via subprocess; return run metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = REPO_ROOT / "experiments" / "train_substrate_dreamer_v3_rssm.py"
    # NOTE: the trainer selects _full_main by the ABSENCE of --smoke (there is
    # NO --full flag; main() dispatches `--smoke -> _smoke_main else _full_main`).
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
        "--seg-distill-objective",
        objective,
        "--seg-hinge-margin",
        str(hinge_margin),
        "--seg-tau-boundary",
        str(seg_tau_boundary),
        # REAL teacher: --allow-mock-scorer-teacher is NEVER passed (default off
        # builds the real SegNet + PoseNet teachers per the trainer contract).
        # distillation_weight > 0 so the seg-distill term is live.
        "--distillation-weight",
        "1.0",
        "--pose-distillation-weight",
        "1.0",
    ]
    if extra_args:
        cmd.extend(extra_args)
    t0 = time.perf_counter()
    env = {"PYTHONPATH": f"{SRC}:{UPSTREAM}"}
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=full_env,
    )
    wall = time.perf_counter() - t0
    tail = (proc.stdout or "")[-2000:] + "\n--- stderr ---\n" + (proc.stderr or "")[-2000:]
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


def _load_trained_model(out_dir: Path, *, num_pairs: int, num_groups: int,
                        num_categories: int, base_channels: int):
    """Load the EMA-shadow trained DreamerV3 model from its .npsd checkpoint."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )
    from tac.substrates.dreamer_v3_rssm.module import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(
        num_groups=num_groups,
        num_categories=num_categories,
        base_channels=base_channels,
        num_pairs=num_pairs,
        gumbel_temperature=1.0,
        use_straight_through=True,
    )
    model = DreamerV3RSSMSubstrateMLX(cfg)
    npsd = _find_ema_shadow_npsd(out_dir)
    sd = unpack_state_dict_numpy(npsd.read_bytes())
    items = [(k, mx.array(v)) for k, v in sd.items()]
    model.update(tree_unflatten(items))
    mx.eval(model.parameters())
    return model, npsd


def _render_all_pairs(model, num_pairs: int):
    """Deterministic argmax eval render for all pairs -> numpy (P, 2, H, W, 3) [0,255]."""
    import mlx.core as mx
    import numpy as np

    recon = []
    for p in range(num_pairs):
        idx = mx.argmax(model.logits[p : p + 1], axis=-1)  # (1, G)
        rgb = model.forward_eval_from_indices(idx)  # (1, 2, 3, H, W) [0,255]
        mx.eval(rgb)
        # (1, 2, 3, H, W) -> (2, H, W, 3)
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


def _measure_real_d_seg_d_pose(
    dn, gt_pairs, recon_pairs, *, batch: int = 32
) -> dict:
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
# Arm measurement (load + render + real distortion)
# ---------------------------------------------------------------------------


def _measure_arm(
    out_dir: Path,
    *,
    num_pairs: int,
    num_groups: int,
    num_categories: int,
    base_channels: int,
    dn,
    gt_pairs,
) -> dict:
    model, npsd = _load_trained_model(
        out_dir,
        num_pairs=num_pairs,
        num_groups=num_groups,
        num_categories=num_categories,
        base_channels=base_channels,
    )
    recon = _render_all_pairs(model, num_pairs)
    metrics = _measure_real_d_seg_d_pose(dn, gt_pairs, recon)
    metrics["ema_shadow_npsd"] = str(npsd.relative_to(REPO_ROOT))
    # NO-FAKE invariant: recon mean/std (proves the render is not a constant).
    import numpy as np

    metrics["recon_mean"] = float(recon.mean())
    metrics["recon_std"] = float(recon.std())
    metrics["recon_is_nonconstant"] = bool(np.std(recon) > 1e-3)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "In-substrate contest-faithful confirmation of the "
            "boundary_argmax_hinge seg-distill lever on the capacity-limited "
            "DreamerV3 renderer (REAL post-training SegNet d_seg vs kl_t2)."
        )
    )
    ap.add_argument("--epochs", type=int, default=120,
                    help="Matched training epochs per arm.")
    ap.add_argument("--num-pairs", type=int, default=64,
                    help="Real GT pairs (full contest=600; default 64 for a "
                    "fast in-substrate confirmation).")
    ap.add_argument("--num-groups", type=int, default=24)
    ap.add_argument("--num-categories", type=int, default=256)
    ap.add_argument("--base-channels", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hinge-margins", type=str, default="1.0",
                    help="Comma-separated hinge margins to sweep for the "
                    "boundary_argmax_hinge arm (e.g. '0.5,1.0').")
    ap.add_argument("--seg-tau-boundary", type=float, default=2.0,
                    help="Boundary-band temperature (A/B found 2.0 robust).")
    ap.add_argument("--video", default="upstream/videos/0.mkv")
    ap.add_argument("--out-root", type=Path,
                    default=Path("experiments/results/"
                                 "dreamer_v3_argmax_hinge_in_substrate_confirm"))
    ap.add_argument("--reuse-existing", action="store_true",
                    help="Skip training arms whose EMA-shadow checkpoint "
                    "already exists (measure-only).")
    args = ap.parse_args(argv)

    margins = [float(x) for x in str(args.hinge_margins).split(",") if x.strip()]
    started = _utc_now()
    out_root = (REPO_ROOT / args.out_root) if not args.out_root.is_absolute() else args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Train all arms (control kl_t2 + one hinge arm per margin) ---
    arm_specs: list[tuple[str, str, float]] = [("kl_t2", "kl_t2", 1.0)]
    for m in margins:
        arm_specs.append((f"argmax_hinge_m{m:g}", "boundary_argmax_hinge", m))

    train_meta: dict[str, dict] = {}
    for arm_name, objective, margin in arm_specs:
        arm_dir = out_root / arm_name
        npsd_exists = bool(list((arm_dir / "checkpoints").glob(
            "final_*.ema_shadow.state.npsd"))) if (arm_dir / "checkpoints").exists() else False
        if args.reuse_existing and npsd_exists:
            train_meta[arm_name] = {
                "objective": objective, "reused": True,
                "output_dir": str(arm_dir),
            }
            print(f"[{arm_name}] reuse-existing: skipping training")
            continue
        print(f"[{arm_name}] training objective={objective} margin={margin} "
              f"epochs={args.epochs} pairs={args.num_pairs} ...")
        meta = _train_arm(
            objective=objective,
            out_dir=arm_dir,
            epochs=int(args.epochs),
            num_pairs=int(args.num_pairs),
            seed=int(args.seed),
            hinge_margin=float(margin),
            seg_tau_boundary=float(args.seg_tau_boundary),
            video_path=str(args.video),
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
    print("[measure] NO-FAKE identity guard PASS: "
          "compute_distortion(gt, gt) == 0.0")

    arm_metrics: dict[str, dict] = {}
    for arm_name, objective, margin in arm_specs:
        arm_dir = out_root / arm_name
        print(f"[measure] {arm_name} ...")
        metrics = _measure_arm(
            arm_dir,
            num_pairs=int(args.num_pairs),
            num_groups=int(args.num_groups),
            num_categories=int(args.num_categories),
            base_channels=int(args.base_channels),
            dn=dn,
            gt_pairs=gt_pairs,
        )
        metrics["objective"] = objective
        metrics["hinge_margin"] = margin
        arm_metrics[arm_name] = metrics
        print(
            f"[measure] {arm_name}: d_seg={metrics['mean_d_seg']:.6f} "
            f"d_pose={metrics['mean_d_pose']:.6e} "
            f"recon_nonconstant={metrics['recon_is_nonconstant']}"
        )

    # --- Verdict ---
    control = arm_metrics["kl_t2"]
    control_d_seg = control["mean_d_seg"]
    control_d_pose = control["mean_d_pose"]
    hinge_arms = {k: v for k, v in arm_metrics.items() if k != "kl_t2"}
    # best hinge arm = lowest d_seg.
    best_hinge_name = min(hinge_arms, key=lambda k: hinge_arms[k]["mean_d_seg"])
    best_hinge = hinge_arms[best_hinge_name]
    best_d_seg = best_hinge["mean_d_seg"]
    best_d_pose = best_hinge["mean_d_pose"]

    rel_d_seg_reduction = (
        (control_d_seg - best_d_seg) / control_d_seg
        if control_d_seg > 0
        else 0.0
    )
    # pose regression check: d_pose increase > 5% relative is a regression.
    pose_rel_change = (
        (best_d_pose - control_d_pose) / control_d_pose
        if control_d_pose > 0
        else 0.0
    )
    pose_regressed = pose_rel_change > 0.05
    seg_improved = best_d_seg < control_d_seg

    if seg_improved and not pose_regressed:
        verdict = "PARADIGM-VALIDATED"
        verdict_detail = (
            f"boundary_argmax_hinge ({best_hinge_name}) lowered REAL "
            f"post-training SegNet d_seg {control_d_seg:.6f} -> {best_d_seg:.6f} "
            f"({rel_d_seg_reduction:+.1%}) WITHOUT pose regression "
            f"(d_pose {control_d_pose:.4e} -> {best_d_pose:.4e}, "
            f"{pose_rel_change:+.1%}). The objective transfers from sister A's "
            f"unbounded per-pixel proxy to the capacity-limited DreamerV3 "
            f"categorical-posterior renderer."
        )
    else:
        verdict = "IMPLEMENTATION-LEVEL-FALSIFIED"
        why = []
        if not seg_improved:
            why.append(
                f"d_seg did NOT improve ({control_d_seg:.6f} -> {best_d_seg:.6f}); "
                "the hinge gradient (sparse, non-smooth) did not transfer "
                "through the distilled student-head -> categorical posterior -> "
                "HNeRV decoder to lower the real SegNet argmax-flip rate"
            )
        if pose_regressed:
            why.append(
                f"pose REGRESSED {pose_rel_change:+.1%} "
                f"(d_pose {control_d_pose:.4e} -> {best_d_pose:.4e})"
            )
        verdict_detail = (
            "boundary_argmax_hinge did not dominate kl_t2 on the "
            "capacity-limited renderer: " + "; ".join(why) + ". The objective "
            "MECHANISM (sister A's proxy A/B) is NOT killed — only this specific "
            "substrate-render coupling is IMPLEMENTATION-LEVEL falsified per "
            "Catalog #307; reactivation = larger epoch budget / full 600 pairs / "
            "joint hinge+kl warmup."
        )

    result = {
        "schema": "dreamer_v3_argmax_hinge_in_substrate_seg_confirm_v1",
        **CANONICAL_NON_PROMOTABLE,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "real_teacher": (
            "contest SegNet (smp.Unet EfficientNet-B2) + PoseNet "
            "(fastvit_t12) on upstream/videos/0.mkv"
        ),
        "real_distortion_net": "upstream.modules.DistortionNet (CPU)",
        "measurement_functional": (
            "d_seg = (SegNet(GT).argmax != SegNet(recon).argmax).mean() "
            "[upstream/modules.py:112]; d_pose = PoseNet first-half MSE "
            "[upstream/modules.py:84]; recon = DETERMINISTIC argmax eval render "
            "of EMA-shadow trained DreamerV3 (forward_eval_from_indices)"
        ),
        "no_fake_identity_guard": "compute_distortion(gt, gt) == 0.0 (verified)",
        "epochs": int(args.epochs),
        "num_pairs": int(args.num_pairs),
        "num_groups": int(args.num_groups),
        "num_categories": int(args.num_categories),
        "base_channels": int(args.base_channels),
        "seed": int(args.seed),
        "seg_tau_boundary": float(args.seg_tau_boundary),
        "hinge_margins_swept": margins,
        "train_meta": train_meta,
        "arm_metrics": arm_metrics,
        "control_arm": "kl_t2",
        "control_d_seg": control_d_seg,
        "control_d_pose": control_d_pose,
        "best_hinge_arm": best_hinge_name,
        "best_hinge_d_seg": best_d_seg,
        "best_hinge_d_pose": best_d_pose,
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
            "at the capacity-limited-renderer in-domain context (IMPLEMENTATION-LEVEL)"
        ),
        "paired_cuda_ratification_op_routable": (
            "OPERATOR-FUNDED (<=$20, DO NOT FIRE): if PARADIGM-VALIDATED, build "
            "a byte-closed DreamerV3 archive from the boundary_argmax_hinge EMA "
            "shadow, run paired [contest-CPU] (GHA Linux x86_64) + [contest-CUDA] "
            "(T4) auth-eval per Catalog #246 to ratify the real-d_seg reduction "
            "translates to a real contest score delta."
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
        rel_source = str((out_json).relative_to(REPO_ROOT))
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
    print(f"  kl_t2 (control)         d_seg={control_d_seg:.6f} "
          f"d_pose={control_d_pose:.4e}")
    print(f"  {best_hinge_name} (best hinge) d_seg={best_d_seg:.6f} "
          f"d_pose={best_d_pose:.4e}")
    print(f"  relative d_seg reduction (hinge vs kl_t2): "
          f"{rel_d_seg_reduction:+.1%}")
    print(f"  pose relative change (hinge vs kl_t2):     "
          f"{pose_rel_change:+.1%} (regressed={pose_regressed})")
    print(f"  detail: {verdict_detail}")
    print(f"\nwrote {out_json.relative_to(REPO_ROOT)}")
    print("[macOS-MLX research-signal] — NON-PROMOTABLE per Catalog #192/#341")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
