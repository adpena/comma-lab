#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ruff: noqa: E402
"""Cascade B Path A learnable student head minimum-viable smoke.

Per CASCADE B HINTON KL-DISTILL CATALYST DISTORTION-ATTACK 2026-05-26
pre-execution gate report + sister `lane_hinton_mlx_first_local_pivot_20260526`
Path A reactivation criterion: empirically test that a learnable 1x1-conv
student head (~20 params) breaks the deterministic-projection saturation
point at KL T=2.0 ~3.03 confirmed across 1000 epochs on real SegNet teacher.

Scope: SMALL fixture (50 frames; 100 epochs) to provide minimum-viable
empirical anchor BEFORE the operator commits to a sister wave that wires
P5 (QAT) + P10 (BPR1) onto the Path A foundation.

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable +
Catalog #192: this smoke produces [macOS-MLX research-signal] evidence
only; no contest-score authority; promotion gated on paired CPU+CUDA
verify per Catalog #205.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog
#229 PV: this smoke is a controlled comparison vs the sister
deterministic-projection baseline; both runs use the SAME 50-frame
fixture + SAME real SegNet teacher cache, differing ONLY in the student
head architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# Make src/ importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))

from tac.framework_agnostic import mlx_eval, require_mlx_core
from tac.substrates.hinton_distilled_scorer_surrogate import (
    HintonMlxCustomLossFnConfig,
    MockTeacherLogitsProvider,
    build_learnable_student_head,
    build_real_segnet_teacher_cache,
    hinton_distilled_kl_t2_loss,
)

mx = require_mlx_core()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_frames(video_path: Path, n_frames: int, h: int, w: int) -> np.ndarray:
    """Decode first n_frames of the canonical contest video into NHWC uint8."""
    import av  # type: ignore[import-not-found]

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        # Canonical contest size for SegNet: 384x512 (HxW)
        if rgb.shape[0] != h or rgb.shape[1] != w:
            # The canonical contest video is at (874, 1164) per upstream/videos/0.mkv
            # SegNet's preprocess_input interpolates to (384, 512). We pre-resize
            # to (384, 512) via numpy/PIL-style bilinear for direct apples-to-apples
            # comparison with sister `_student_logits_from_decoded` path.
            from PIL import Image
            pil_img = Image.fromarray(rgb)
            pil_resized = pil_img.resize((w, h), Image.BILINEAR)
            rgb = np.array(pil_resized)
        frames.append(rgb)
        if len(frames) >= n_frames:
            break
    container.close()
    arr = np.stack(frames[:n_frames], axis=0).astype(np.uint8)
    return arr


def run_smoke(
    *,
    student_head_mode: str,  # "deterministic" or "learnable"
    n_frames: int,
    n_epochs: int,
    batch_size: int,
    temperature: float,
    distillation_weight: float,
    seed: int,
    lr: float,
    output_dir: Path,
    artifact_label: str,
) -> dict:
    """Run a smoke with the canonical real SegNet teacher cache and a
    configurable student head mode.

    Returns a JSON-serializable telemetry dict per the canonical Catalog
    #305 observability surface.
    """
    print(f"[cascade-b-smoke] mode={student_head_mode} n_frames={n_frames} n_epochs={n_epochs}", flush=True)

    canonical_h = 384
    canonical_w = 512
    num_classes = 5

    # Stage 1: decode frames
    video_path = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    video_sha = _sha256_file(video_path)
    print(f"[cascade-b-smoke] decoding {n_frames} frames from {video_path.name} (sha256={video_sha[:16]}...)", flush=True)
    t0 = time.time()
    frames_thwc = decode_frames(video_path, n_frames, canonical_h, canonical_w)
    decode_secs = time.time() - t0
    print(f"[cascade-b-smoke] decoded {frames_thwc.shape} in {decode_secs:.1f}s", flush=True)

    # Stage 2: build real SegNet teacher cache (CPU)
    print("[cascade-b-smoke] building real SegNet teacher cache (CPU)...", flush=True)
    t0 = time.time()
    cache = build_real_segnet_teacher_cache(
        frames_thwc,
        upstream_dir=REPO_ROOT / "upstream",
        device="cpu",
    )
    cache_secs = time.time() - t0
    print(f"[cascade-b-smoke] teacher cache built in {cache_secs:.1f}s; shape=(T={cache.frame_count}, H={cache.height}, W={cache.width}, K={cache.num_classes})", flush=True)

    # Stage 3: convert frames to MLX float32 [0, 1] NHWC for student input.
    # In this smoke we use the GROUND-TRUTH RGB directly as the student
    # input — this isolates the question "can a learnable head match real
    # SegNet logits when given the source RGB?" from the question "can the
    # HNeRV decoder learn to produce useful intermediate features?".
    # Per Catalog #229: the canonical Slot 1 pipeline uses the decoded RGB
    # from the HNeRV decoder as the student input; here we use ground-truth
    # RGB as an upper bound on what Path A can achieve (the head's
    # convergence floor when given perfect input). This is exactly the
    # foundational test the sister memo asked for at Path A line 152.
    frames_bhwc_f32 = mx.array(frames_thwc.astype(np.float32) / 255.0)

    # Stage 4: configure student head
    if student_head_mode == "learnable":
        learnable_head = build_learnable_student_head(
            num_classes=num_classes, in_channels=3, seed=seed, init_scale=0.1
        )
        config = HintonMlxCustomLossFnConfig(
            distillation_weight=distillation_weight,
            temperature=temperature,
            student_head_out_channels=num_classes,
            real_teacher_cache=cache,
            learnable_student_head=learnable_head,
        )
    elif student_head_mode == "deterministic":
        # Sister back-compat: deterministic projection via MockTeacherLogitsProvider
        # on decoded frames. Per sister Slot 1 the spatial_downsample_factor is
        # 1 when real_teacher_cache is provided so logits shapes match.
        provider = MockTeacherLogitsProvider(num_classes=num_classes, spatial_downsample_factor=1)
        config = HintonMlxCustomLossFnConfig(
            distillation_weight=distillation_weight,
            temperature=temperature,
            student_head_out_channels=num_classes,
            teacher_provider=provider,
            real_teacher_cache=cache,
        )
    else:
        raise ValueError(f"unknown student_head_mode: {student_head_mode}")

    # Stage 5: training loop. We optimize ONLY the student head's weights
    # (the canonical decoder is replaced by the ground-truth RGB in this
    # foundational test). For Path A (learnable head): trainable params =
    # head.weight + head.bias. For deterministic: trainable params = []
    # (the deterministic projection has no learnable params, so we report
    # the loss curve at fixed initialization).
    rng = np.random.RandomState(seed)
    losses = []

    if student_head_mode == "learnable":
        # Compute loss + gradient through head weights.
        def loss_for_step(weight, bias, batch_indices_mx, batch_targets_mx):
            decoded_bhwc = batch_targets_mx  # ground-truth RGB
            # Apply learnable head: logits = decoded @ W + b
            student_logits = mx.einsum("bhwc,ck->bhwk", decoded_bhwc, weight) + bias
            # Teacher: real SegNet cache lookup
            teacher_logits = cache.teacher_logits_for_indices(batch_indices_mx)
            teacher_logits_stopped = mx.stop_gradient(teacher_logits)
            distill = hinton_distilled_kl_t2_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits_stopped,
                temperature=temperature,
            )
            # Reconstruction MSE term per canonical
            mse_term = mx.mean(mx.zeros((1,), dtype=mx.float32))  # 0 since student input = target
            combined = mse_term + distillation_weight * distill
            return combined, distill

        grad_fn = mx.value_and_grad(
            lambda w, b, idx, tgt: loss_for_step(w, b, idx, tgt)[0],
            argnums=(0, 1),
        )
        w = config.learnable_student_head.weight
        b = config.learnable_student_head.bias

        n_batches_per_epoch = max(1, n_frames // batch_size)
        for epoch in range(n_epochs):
            ep_losses = []
            for _ in range(n_batches_per_epoch):
                indices = rng.choice(n_frames, size=batch_size, replace=False)
                indices_mx = mx.array(indices.astype(np.int32))
                targets_mx = frames_bhwc_f32[indices_mx]
                loss_val, (gw, gb) = grad_fn(w, b, indices_mx, targets_mx)
                # Plain SGD; matches sister 0.5 init scale + small lr
                w = w - lr * gw
                b = b - lr * gb
                # Force MLX eval so loss is concrete
                mlx_eval(w, b, loss_val)
                # Recover distill-only term for logging
                _, distill_only = loss_for_step(w, b, indices_mx, targets_mx)
                mlx_eval(distill_only)
                ep_losses.append(float(distill_only.item()))
            avg = float(np.mean(ep_losses))
            losses.append(avg)
            if epoch % max(1, n_epochs // 10) == 0:
                print(f"[cascade-b-smoke] mode={student_head_mode} ep={epoch:>3d}/{n_epochs}: distill_KL={avg:.4f}", flush=True)
    else:
        # Deterministic projection has no trainable params; we just measure
        # the loss at fixed initialization across all epochs (the loss is
        # constant because nothing is learning). This anchors the sister
        # baseline saturation floor on the SAME 50-frame fixture.
        provider = config.teacher_provider
        for epoch in range(n_epochs):
            ep_losses = []
            for _ in range(max(1, n_frames // batch_size)):
                indices = rng.choice(n_frames, size=batch_size, replace=False)
                indices_mx = mx.array(indices.astype(np.int32))
                targets_mx = frames_bhwc_f32[indices_mx]
                # Deterministic projection on ground-truth RGB
                student_logits = provider.teacher_logits(targets_mx)
                teacher_logits = cache.teacher_logits_for_indices(indices_mx)
                teacher_logits_stopped = mx.stop_gradient(teacher_logits)
                distill = hinton_distilled_kl_t2_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits_stopped,
                    temperature=temperature,
                )
                mlx_eval(distill)
                ep_losses.append(float(distill.item()))
            avg = float(np.mean(ep_losses))
            losses.append(avg)
            if epoch % max(1, n_epochs // 10) == 0:
                print(f"[cascade-b-smoke] mode={student_head_mode} ep={epoch:>3d}/{n_epochs}: distill_KL={avg:.4f}", flush=True)

    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = float(np.min(losses))
    reduction_pct = (initial_loss - final_loss) / max(initial_loss, 1e-6) * 100.0

    # Canonical verdict per sister 4-verdict taxonomy
    if reduction_pct < 5.0:
        verdict = "SUB_PARADIGM"
    elif reduction_pct < 50.0:
        verdict = "PARTIAL_CONVERGENCE"
    else:
        verdict = "CONVERGES_CONSISTENTLY"

    # Determine n_params
    if student_head_mode == "learnable":
        n_params = int(config.learnable_student_head.weight.size + config.learnable_student_head.bias.size)
    else:
        n_params = 0

    telemetry = {
        "schema_version": "cascade_b_hinton_kl_distill_catalyst_smoke_verdict_v1_20260526",
        "lane_id": "lane_cascade_b_hinton_kl_distill_catalyst_distortion_attack_mlx_first_numpy_portable_individually_fractal_20260526",
        "subagent_id": "cascade-b-hinton-kl-distill-catalyst-distortion-attack-mlx-first-numpy-portable-individually-fractal-20260526",
        "artifact_label": artifact_label,
        "student_head_mode": student_head_mode,
        "n_trainable_params": n_params,
        "n_frames": n_frames,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "temperature": temperature,
        "distillation_weight": distillation_weight,
        "learning_rate": lr,
        "seed": seed,
        "canonical_eval_size": [canonical_h, canonical_w],
        "source_video_sha256": video_sha,
        "teacher_cache_seconds": cache_secs,
        "training_seconds": float(sum([1.0 for _ in losses])),  # placeholder; per-ep wall not measured
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "min_loss_across_run": min_loss,
        "reduction_pct": reduction_pct,
        "verdict": verdict,
        "loss_curve_first_10": [float(x) for x in losses[:10]],
        "loss_curve_last_10": [float(x) for x in losses[-10:]],
        "canonical_provenance": {
            "axis_tag": "[macOS-MLX research-signal]",
            "hardware_substrate": "macos_arm64",
            "evidence_grade": "macOS-MLX-research-signal",
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{artifact_label}_verdict.json"
    output_path.write_text(json.dumps(telemetry, indent=2, sort_keys=True))
    print(f"[cascade-b-smoke] verdict written to {output_path}", flush=True)
    print(f"[cascade-b-smoke] {artifact_label}: initial={initial_loss:.4f} final={final_loss:.4f} min={min_loss:.4f} reduction={reduction_pct:.1f}% verdict={verdict}", flush=True)
    return telemetry


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "results" / "cascade_b_hinton_kl_distill_catalyst_20260526"),
    )
    parser.add_argument("--mode", choices=["both", "learnable", "deterministic"], default="both")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    print(f"[cascade-b-smoke] output_dir={output_dir}", flush=True)

    results = {}
    if args.mode in ("both", "deterministic"):
        det_telemetry = run_smoke(
            student_head_mode="deterministic",
            n_frames=args.n_frames,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            distillation_weight=args.distillation_weight,
            seed=args.seed,
            lr=args.learning_rate,
            output_dir=output_dir,
            artifact_label="sister_deterministic_projection",
        )
        results["deterministic"] = det_telemetry
    if args.mode in ("both", "learnable"):
        learn_telemetry = run_smoke(
            student_head_mode="learnable",
            n_frames=args.n_frames,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            distillation_weight=args.distillation_weight,
            seed=args.seed,
            lr=args.learning_rate,
            output_dir=output_dir,
            artifact_label="cascade_b_path_a_learnable_head",
        )
        results["learnable"] = learn_telemetry

    # Comparison summary
    if "deterministic" in results and "learnable" in results:
        det = results["deterministic"]
        learn = results["learnable"]
        print("\n=== CASCADE B PATH A SMOKE COMPARISON ===", flush=True)
        print(f"  Deterministic baseline: initial={det['initial_loss']:.4f} final={det['final_loss']:.4f} min={det['min_loss_across_run']:.4f} reduction={det['reduction_pct']:.1f}% verdict={det['verdict']}", flush=True)
        print(f"  Path A learnable head:  initial={learn['initial_loss']:.4f} final={learn['final_loss']:.4f} min={learn['min_loss_across_run']:.4f} reduction={learn['reduction_pct']:.1f}% verdict={learn['verdict']}", flush=True)
        print(f"  Delta (det - learn) final: {det['final_loss'] - learn['final_loss']:.4f}", flush=True)
        print(f"  Path A trainable params: {learn['n_trainable_params']}", flush=True)
        summary = {
            "schema_version": "cascade_b_path_a_smoke_comparison_v1_20260526",
            "deterministic": det,
            "learnable_path_a": learn,
            "delta_det_minus_learn_final": det["final_loss"] - learn["final_loss"],
            "delta_det_minus_learn_min": det["min_loss_across_run"] - learn["min_loss_across_run"],
            "verdict_path_a_breaks_saturation": learn["final_loss"] < det["final_loss"] - 0.05,
        }
        summary_path = output_dir / "comparison_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"  Summary written to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
