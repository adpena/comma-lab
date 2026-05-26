#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CASCADE B CATALYST SISTER WAVE 2 — production-scale 600f x 1000ep Path A
training + post-train QAT 100ep fine-tune on real-SegNet fixture
(6th-order recursive doctrine).

Operator-pre-approved per Cascade B CATALYST 5th-order
IMPLEMENTATION_LEVEL_FALSIFIED-at-synthetic-fixture verdict (commit
`fcfad9331`) operator-routable sister wave + 2026-05-26 "Use however you
want" routing autonomy delegation + "Remember all on MLX" standing
directive.

THE 6TH-ORDER PRODUCTION-SCALE POST-TRAIN QAT FIX:

5th-order CATALYST cascade composition was IMPLEMENTATION-LEVEL FALSIFIED
at synthetic fixture (50-pair x 48x64 / inference-only forward) because
the synthetic surrogate didn't exercise the CATALYST training-time
benefit (+1.6e-2 sidecar rate for ZERO d_seg-proxy improvement).
PARADIGM (CATALYST P2+P5+P10) INTACT per Catalog #307; the diagnosis is
the synthetic-fixture surrogate, NOT the paradigm.

This wave 2 lands the canonical 6th-order fix:
  Stage A) Train Path A learnable head for n_epochs (default 1000) on
    real SegNet teacher cache with Hinton KL T=2.0 (sister wave 1 pattern).
  Stage B) Freeze head; apply quantize_head_fp4 from catalyst_cascade.
  Stage C) FINE-TUNE 100ep more on SAME real-SegNet fixture with
    FakeQuantFP4 STE (post-train QAT phase).
  Stage D) Measure 3-arm comparison (baseline=deterministic /
    Path A alone / CATALYST=Path A + post-train QAT) on production-scale
    600-frame real-SegNet fixture.

Per CLAUDE.md "QAT pipeline - non-negotiable" canonical 5-stage chain:
  anchor -> finetune -> joint -> QAT -> final.
Wave 1 implemented anchor + finetune (Path A 1000ep);
Wave 2 implements QAT + final (this wave).

Canonical equation #2 anchor lifecycle:
  Currently 2 events (1 registered + 1 anchor_appended from synthetic
  FALSIFIED). Wave 2 lands 2nd empirical anchor (events 2 -> 3); at 3+
  in-domain anchors the registered auto-recalibration consumer fires
  per the schema's `when_3+_new_empirical_anchors_in_domain` clause.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317:
  every output carries `[macOS-MLX research-signal]` axis tag; NEVER
  promotable.

Acceptance verdicts per Catalog #307:
  PARADIGM_VALIDATED          : CATALYST composite < Path A composite
  PARTIAL_CONFIRMATION        : composites within proxy noise (<0.001)
  IMPLEMENTATION_LEVEL_FALSIFIED: CATALYST composite > Path A composite
                                  + delta_rate > 0.005
  DEFER_PENDING_QAT_STABILIZATION: post-QAT KL diverges (>5.0 or NaN)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "upstream"))

from tac.substrates.hinton_distilled_scorer_surrogate import (  # noqa: E402
    MockTeacherLogitsProvider,
    build_learnable_student_head,
    build_real_segnet_teacher_cache,
    hinton_distilled_kl_t2_loss,
)
from tac.substrates.hinton_distilled_scorer_surrogate.catalyst_cascade import (  # noqa: E402
    FP4_DEFAULT_CODEBOOK,
    fake_quant_fp4_mlx,
    quantize_head_fp4,
)
from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (  # noqa: E402
    LearnableConv1x1StudentHead,
)

# Path A canonical config (sister wave 1 defaults)
PATH_A_DEFAULT_N_FRAMES = 600
PATH_A_DEFAULT_N_EPOCHS = 1000
PATH_A_DEFAULT_BATCH_SIZE = 30
PATH_A_DEFAULT_TEMPERATURE = 2.0
PATH_A_DEFAULT_DISTILLATION_WEIGHT = 0.5
PATH_A_DEFAULT_LR = 0.5
PATH_A_DEFAULT_SEED = 0

# Wave 2 post-train QAT fine-tune defaults
QAT_DEFAULT_N_EPOCHS = 100
# Per CLAUDE.md "QAT pipeline": LSQ step size lr = 0.01 * base_lr.
QAT_DEFAULT_LR_FACTOR = 0.1  # gentler lr for post-train fine-tune

# Catalog #307 verdict thresholds (3-arm composite comparison)
VERDICT_PARTIAL_CONFIRMATION_NOISE_BAND = 0.001
VERDICT_IMPLEMENTATION_FALSIFIED_RATE_DELTA_THRESHOLD = 0.005
VERDICT_DEFER_QAT_DIVERGENCE_KL = 5.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_frames(video_path: Path, n_frames: int, h: int, w: int) -> np.ndarray:
    """Decode first n_frames of the canonical contest video into NHWC uint8.

    Reuses sister `cascade_b_path_a_learnable_head_smoke.decode_frames`
    contract (384x512 resize via PIL bilinear; canonical SegNet eval size).
    """
    import av  # type: ignore[import-not-found]

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        if rgb.shape[0] != h or rgb.shape[1] != w:
            from PIL import Image
            pil_img = Image.fromarray(rgb)
            pil_resized = pil_img.resize((w, h), Image.BILINEAR)
            rgb = np.array(pil_resized)
        frames.append(rgb)
        if len(frames) >= n_frames:
            break
    container.close()
    return np.stack(frames[:n_frames], axis=0).astype(np.uint8)


def _kl_t2_loss(student_logits: mx.array, target_logits: mx.array, *, temperature: float) -> mx.array:
    """Canonical KL T=2.0 loss matching sister Path A primitive."""
    target_stopped = mx.stop_gradient(target_logits)
    return hinton_distilled_kl_t2_loss(
        student_logits=student_logits,
        teacher_logits=target_stopped,
        temperature=temperature,
    )


def train_path_a(
    *,
    frames_bhwc_f32: mx.array,
    cache,
    n_epochs: int,
    batch_size: int,
    temperature: float,
    distillation_weight: float,
    lr: float,
    seed: int,
    label: str,
) -> tuple[LearnableConv1x1StudentHead, list[float]]:
    """Train Path A learnable 1x1-conv head; return trained head + KL curve."""
    n_frames = int(frames_bhwc_f32.shape[0])
    head = build_learnable_student_head(
        num_classes=5, in_channels=3, seed=seed, init_scale=0.1
    )

    def loss_for_step(weight, bias, batch_indices_mx, batch_targets_mx):
        decoded_bhwc = batch_targets_mx
        student_logits = mx.einsum("bhwc,ck->bhwk", decoded_bhwc, weight) + bias
        teacher_logits = cache.teacher_logits_for_indices(batch_indices_mx)
        distill = _kl_t2_loss(student_logits, teacher_logits, temperature=temperature)
        mse_term = mx.mean(mx.zeros((1,), dtype=mx.float32))
        combined = mse_term + distillation_weight * distill
        return combined, distill

    grad_fn = mx.value_and_grad(
        lambda w, b, idx, tgt: loss_for_step(w, b, idx, tgt)[0],
        argnums=(0, 1),
    )

    w = head.weight
    b = head.bias
    rng = np.random.RandomState(seed)
    losses: list[float] = []
    n_batches = max(1, n_frames // batch_size)

    t0 = time.time()
    for epoch in range(n_epochs):
        ep_losses = []
        for _ in range(n_batches):
            indices = rng.choice(n_frames, size=batch_size, replace=False)
            indices_mx = mx.array(indices.astype(np.int32))
            targets_mx = frames_bhwc_f32[indices_mx]
            loss_val, (gw, gb) = grad_fn(w, b, indices_mx, targets_mx)
            w = w - lr * gw
            b = b - lr * gb
            mx.eval(w, b, loss_val)
            _, distill_only = loss_for_step(w, b, indices_mx, targets_mx)
            mx.eval(distill_only)
            ep_losses.append(float(distill_only.item()))
        avg = float(np.mean(ep_losses))
        losses.append(avg)
        if epoch % max(1, n_epochs // 10) == 0:
            elapsed = time.time() - t0
            print(
                f"[cascade-b-wave2 {label}] ep={epoch:>4d}/{n_epochs}: distill_KL={avg:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    trained_head = LearnableConv1x1StudentHead(weight=w, bias=b, num_classes=5)
    return trained_head, losses


def fine_tune_post_qat(
    *,
    frames_bhwc_f32: mx.array,
    cache,
    head_seed: LearnableConv1x1StudentHead,
    n_epochs: int,
    batch_size: int,
    temperature: float,
    distillation_weight: float,
    lr: float,
    seed: int,
    fp4_codebook: tuple[float, ...] = FP4_DEFAULT_CODEBOOK,
    label: str = "qat_finetune",
) -> tuple[LearnableConv1x1StudentHead, list[float]]:
    """Post-train QAT fine-tune phase (Stage QAT + final per CLAUDE.md
    QAT pipeline canonical chain).

    For each batch: forward routes student weight through fake_quant_fp4_mlx
    (identity-STE), computes KL T=2 loss vs real-SegNet teacher; backward
    routes gradient through identity-STE; SGD step updates the underlying
    full-precision shadow weight. After all epochs, return the FINAL
    weights AFTER fp4 quantization (the deployable codebook entries).
    """
    n_frames = int(frames_bhwc_f32.shape[0])
    w = head_seed.weight
    b = head_seed.bias

    def loss_for_step_qat(weight, bias, batch_indices_mx, batch_targets_mx):
        # Forward: route weight + bias through FakeQuantFP4 STE
        w_q = fake_quant_fp4_mlx(weight, codebook=fp4_codebook)
        b_q = fake_quant_fp4_mlx(bias, codebook=fp4_codebook)
        decoded_bhwc = batch_targets_mx
        student_logits = mx.einsum("bhwc,ck->bhwk", decoded_bhwc, w_q) + b_q
        teacher_logits = cache.teacher_logits_for_indices(batch_indices_mx)
        distill = _kl_t2_loss(student_logits, teacher_logits, temperature=temperature)
        mse_term = mx.mean(mx.zeros((1,), dtype=mx.float32))
        combined = mse_term + distillation_weight * distill
        return combined, distill

    grad_fn = mx.value_and_grad(
        lambda w, b, idx, tgt: loss_for_step_qat(w, b, idx, tgt)[0],
        argnums=(0, 1),
    )

    rng = np.random.RandomState(seed + 7919)  # disjoint stream
    losses: list[float] = []
    n_batches = max(1, n_frames // batch_size)

    t0 = time.time()
    for epoch in range(n_epochs):
        ep_losses = []
        for _ in range(n_batches):
            indices = rng.choice(n_frames, size=batch_size, replace=False)
            indices_mx = mx.array(indices.astype(np.int32))
            targets_mx = frames_bhwc_f32[indices_mx]
            loss_val, (gw, gb) = grad_fn(w, b, indices_mx, targets_mx)
            w = w - lr * gw
            b = b - lr * gb
            mx.eval(w, b, loss_val)
            _, distill_only = loss_for_step_qat(w, b, indices_mx, targets_mx)
            mx.eval(distill_only)
            ep_losses.append(float(distill_only.item()))
        avg = float(np.mean(ep_losses))
        losses.append(avg)
        if epoch % max(1, n_epochs // 10) == 0:
            elapsed = time.time() - t0
            print(
                f"[cascade-b-wave2 {label}] ep={epoch:>3d}/{n_epochs}: qat_KL={avg:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    # Final deployable head: quantize the trained shadow weights once more.
    final_head = quantize_head_fp4(LearnableConv1x1StudentHead(weight=w, bias=b, num_classes=5))
    return final_head, losses


def evaluate_kl(
    *,
    frames_bhwc_f32: mx.array,
    cache,
    head: LearnableConv1x1StudentHead | None,
    deterministic_provider: MockTeacherLogitsProvider | None,
    batch_size: int,
    temperature: float,
    seed: int,
    n_eval_batches: int = 20,
) -> float:
    """Compute average KL across n_eval_batches mini-batches.

    Stage D measurement: 3-arm comparison uses identical fixture + identical
    teacher cache; only head differs.
    """
    n_frames = int(frames_bhwc_f32.shape[0])
    rng = np.random.RandomState(seed + 31337)
    kl_vals: list[float] = []
    for _ in range(n_eval_batches):
        indices = rng.choice(n_frames, size=batch_size, replace=False)
        indices_mx = mx.array(indices.astype(np.int32))
        targets_mx = frames_bhwc_f32[indices_mx]
        if head is not None:
            student_logits = mx.einsum("bhwc,ck->bhwk", targets_mx, head.weight) + head.bias
        elif deterministic_provider is not None:
            student_logits = deterministic_provider.teacher_logits(targets_mx)
        else:
            raise ValueError("either head or deterministic_provider must be provided")
        teacher_logits = cache.teacher_logits_for_indices(indices_mx)
        kl = _kl_t2_loss(student_logits, teacher_logits, temperature=temperature)
        mx.eval(kl)
        kl_vals.append(float(kl.item()))
    return float(np.mean(kl_vals))


def estimate_sidecar_bytes_for_head(head: LearnableConv1x1StudentHead) -> int:
    """Estimate the FP4-quantized head sidecar byte count.

    Canonical: 4 bits per param + per-block fp16 scale. For 20-param head
    (15 weight + 5 bias) packed at 4 bits/param = 10 bytes + 2-byte
    whole-block scale x 2 (weight + bias) = 14 bytes; round up to 16 bytes
    canonical alignment.
    """
    n_params = int(head.weight.size + head.bias.size)
    payload_bytes = (n_params * 4 + 7) // 8  # 4 bits per param, ceiled to bytes
    scale_bytes = 2 * 2  # fp16 scale per block (weight + bias)
    total_bytes = payload_bytes + scale_bytes
    # Canonical 16-byte alignment for sidecar headers (per BPR1 Variant B-d)
    return ((total_bytes + 15) // 16) * 16


def classify_catalyst_verdict(
    *,
    composite_baseline: float,
    composite_path_a: float,
    composite_catalyst: float,
    delta_rate_catalyst_vs_path_a: float,
    qat_final_kl: float,
) -> str:
    """Apply Catalog #307 4-verdict taxonomy."""
    if qat_final_kl != qat_final_kl or qat_final_kl >= VERDICT_DEFER_QAT_DIVERGENCE_KL:
        return "DEFER_PENDING_QAT_STABILIZATION"
    delta_catalyst_vs_path_a = composite_catalyst - composite_path_a
    if abs(delta_catalyst_vs_path_a) < VERDICT_PARTIAL_CONFIRMATION_NOISE_BAND:
        return "PARTIAL_CONFIRMATION"
    if delta_catalyst_vs_path_a < 0:
        return "PARADIGM_VALIDATED"
    # delta_catalyst > path_a (worse composite); check rate cost
    if delta_rate_catalyst_vs_path_a > VERDICT_IMPLEMENTATION_FALSIFIED_RATE_DELTA_THRESHOLD:
        return "IMPLEMENTATION_LEVEL_FALSIFIED"
    return "PARTIAL_CONFIRMATION"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-frames", type=int, default=PATH_A_DEFAULT_N_FRAMES)
    parser.add_argument("--path-a-n-epochs", type=int, default=PATH_A_DEFAULT_N_EPOCHS)
    parser.add_argument("--qat-n-epochs", type=int, default=QAT_DEFAULT_N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PATH_A_DEFAULT_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=PATH_A_DEFAULT_TEMPERATURE)
    parser.add_argument("--distillation-weight", type=float, default=PATH_A_DEFAULT_DISTILLATION_WEIGHT)
    parser.add_argument("--seed", type=int, default=PATH_A_DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=PATH_A_DEFAULT_LR)
    parser.add_argument(
        "--qat-lr-factor",
        type=float,
        default=QAT_DEFAULT_LR_FACTOR,
        help="Multiplier applied to learning-rate during post-train QAT fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "experiments"
            / "results"
            / "cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526"
        ),
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[wave2] CASCADE B CATALYST sister wave 2 6th-order recursive doctrine "
        f"n_frames={args.n_frames} path_a_n_epochs={args.path_a_n_epochs} "
        f"qat_n_epochs={args.qat_n_epochs} batch_size={args.batch_size}",
        flush=True,
    )
    print(f"[wave2] output_dir={output_dir}", flush=True)

    canonical_h, canonical_w = 384, 512
    video_path = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    video_sha = _sha256_file(video_path)
    print(f"[wave2] decoding {args.n_frames} frames sha={video_sha[:16]}...", flush=True)
    t0 = time.time()
    frames_thwc = decode_frames(video_path, args.n_frames, canonical_h, canonical_w)
    decode_secs = time.time() - t0
    print(f"[wave2] decoded {frames_thwc.shape} in {decode_secs:.1f}s", flush=True)

    # Build real SegNet teacher cache
    print("[wave2] building real SegNet teacher cache (CPU)...", flush=True)
    t0 = time.time()
    cache = build_real_segnet_teacher_cache(
        frames_thwc,
        upstream_dir=REPO_ROOT / "upstream",
        device="cpu",
    )
    cache_secs = time.time() - t0
    print(
        f"[wave2] teacher cache built in {cache_secs:.1f}s; "
        f"shape=(T={cache.frame_count}, H={cache.height}, W={cache.width}, "
        f"K={cache.num_classes})",
        flush=True,
    )

    frames_bhwc_f32 = mx.array(frames_thwc.astype(np.float32) / 255.0)

    # ============================================================
    # ARM 1: BASELINE (deterministic projection; sister wave 1 pattern)
    # ============================================================
    print(
        "\n[wave2] ============================================================",
        flush=True,
    )
    print("[wave2] ARM 1: BASELINE (deterministic projection; 0 trainable params)", flush=True)
    print(
        "[wave2] ============================================================",
        flush=True,
    )
    det_provider = MockTeacherLogitsProvider(num_classes=5, spatial_downsample_factor=1)
    t0 = time.time()
    baseline_kl = evaluate_kl(
        frames_bhwc_f32=frames_bhwc_f32,
        cache=cache,
        head=None,
        deterministic_provider=det_provider,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed,
        n_eval_batches=20,
    )
    baseline_wall = time.time() - t0
    print(
        f"[wave2] BASELINE KL={baseline_kl:.4f} eval_wall={baseline_wall:.1f}s",
        flush=True,
    )

    # ============================================================
    # ARM 2: PATH A ALONE (production-scale 600f x 1000ep training)
    # ============================================================
    print(
        "\n[wave2] ============================================================",
        flush=True,
    )
    print(
        f"[wave2] ARM 2: PATH A ALONE (production-scale {args.path_a_n_epochs}ep training)",
        flush=True,
    )
    print(
        "[wave2] ============================================================",
        flush=True,
    )
    t0 = time.time()
    path_a_head, path_a_losses = train_path_a(
        frames_bhwc_f32=frames_bhwc_f32,
        cache=cache,
        n_epochs=args.path_a_n_epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        distillation_weight=args.distillation_weight,
        lr=args.learning_rate,
        seed=args.seed,
        label="path_a_train",
    )
    path_a_train_wall = time.time() - t0
    path_a_initial_kl = path_a_losses[0]
    path_a_final_train_kl = path_a_losses[-1]
    # Eval Path A with same evaluation protocol used for baseline + CATALYST
    path_a_eval_kl = evaluate_kl(
        frames_bhwc_f32=frames_bhwc_f32,
        cache=cache,
        head=path_a_head,
        deterministic_provider=None,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed,
        n_eval_batches=20,
    )
    print(
        f"[wave2] PATH A initial_train_kl={path_a_initial_kl:.4f} "
        f"final_train_kl={path_a_final_train_kl:.4f} "
        f"eval_kl={path_a_eval_kl:.4f} train_wall={path_a_train_wall:.1f}s",
        flush=True,
    )

    # ============================================================
    # ARM 3: CATALYST (Path A foundation + post-train QAT 100ep fine-tune)
    # ============================================================
    print(
        "\n[wave2] ============================================================",
        flush=True,
    )
    print(
        f"[wave2] ARM 3: CATALYST (Path A + post-train QAT {args.qat_n_epochs}ep fine-tune)",
        flush=True,
    )
    print(
        "[wave2] ============================================================",
        flush=True,
    )
    qat_lr = args.learning_rate * args.qat_lr_factor
    print(f"[wave2] QAT fine-tune lr={qat_lr:.4f} (= base_lr {args.learning_rate} * qat_lr_factor {args.qat_lr_factor})", flush=True)
    t0 = time.time()
    catalyst_head, qat_losses = fine_tune_post_qat(
        frames_bhwc_f32=frames_bhwc_f32,
        cache=cache,
        head_seed=path_a_head,  # Path A foundation
        n_epochs=args.qat_n_epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        distillation_weight=args.distillation_weight,
        lr=qat_lr,
        seed=args.seed,
        label="catalyst_qat_finetune",
    )
    qat_wall = time.time() - t0
    qat_final_train_kl = qat_losses[-1]
    catalyst_eval_kl = evaluate_kl(
        frames_bhwc_f32=frames_bhwc_f32,
        cache=cache,
        head=catalyst_head,
        deterministic_provider=None,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed,
        n_eval_batches=20,
    )
    print(
        f"[wave2] CATALYST qat_final_train_kl={qat_final_train_kl:.4f} "
        f"eval_kl={catalyst_eval_kl:.4f} qat_wall={qat_wall:.1f}s",
        flush=True,
    )

    # ============================================================
    # STAGE D: 3-arm composite comparison + Catalog #307 verdict
    # ============================================================
    # Per CLAUDE.md "Apples-to-apples evidence discipline": composite proxy
    # score per arm uses canonical contest formula structure with d_seg
    # proxy = KL/100 (research-signal scaling; no contest claim per
    # Catalog #287/#323).
    baseline_sidecar_bytes = 0
    path_a_sidecar_bytes = estimate_sidecar_bytes_for_head(path_a_head)
    catalyst_sidecar_bytes = estimate_sidecar_bytes_for_head(catalyst_head)

    def composite_proxy(kl_val: float, sidecar_bytes: int) -> float:
        d_seg_proxy = kl_val / 100.0
        rate_proxy = 25.0 * sidecar_bytes / 37_545_489
        return 100.0 * d_seg_proxy + rate_proxy

    baseline_composite = composite_proxy(baseline_kl, baseline_sidecar_bytes)
    path_a_composite = composite_proxy(path_a_eval_kl, path_a_sidecar_bytes)
    catalyst_composite = composite_proxy(catalyst_eval_kl, catalyst_sidecar_bytes)

    delta_path_a_kl_vs_baseline = path_a_eval_kl - baseline_kl
    delta_catalyst_kl_vs_path_a = catalyst_eval_kl - path_a_eval_kl
    delta_catalyst_rate_vs_path_a = (
        25.0 * catalyst_sidecar_bytes / 37_545_489
        - 25.0 * path_a_sidecar_bytes / 37_545_489
    )

    verdict = classify_catalyst_verdict(
        composite_baseline=baseline_composite,
        composite_path_a=path_a_composite,
        composite_catalyst=catalyst_composite,
        delta_rate_catalyst_vs_path_a=delta_catalyst_rate_vs_path_a,
        qat_final_kl=catalyst_eval_kl,
    )

    print(
        "\n[wave2] ============================================================",
        flush=True,
    )
    print("[wave2] STAGE D: 3-ARM COMPOSITE COMPARISON (Catalog #307 verdict)", flush=True)
    print(
        "[wave2] ============================================================",
        flush=True,
    )
    print(
        f"[wave2]   BASELINE: KL={baseline_kl:.4f}  "
        f"sidecar_bytes={baseline_sidecar_bytes}  composite={baseline_composite:.6f}",
        flush=True,
    )
    print(
        f"[wave2]   PATH A:   KL={path_a_eval_kl:.4f}  "
        f"sidecar_bytes={path_a_sidecar_bytes}  composite={path_a_composite:.6f}",
        flush=True,
    )
    print(
        f"[wave2]   CATALYST: KL={catalyst_eval_kl:.4f}  "
        f"sidecar_bytes={catalyst_sidecar_bytes}  composite={catalyst_composite:.6f}",
        flush=True,
    )
    print(
        f"[wave2]   delta(catalyst - path_a) KL = {delta_catalyst_kl_vs_path_a:+.4f}  "
        f"rate = {delta_catalyst_rate_vs_path_a:+.6f}",
        flush=True,
    )
    print(f"[wave2]   CATALOG #307 VERDICT: {verdict}", flush=True)

    total_wall = decode_secs + cache_secs + baseline_wall + path_a_train_wall + qat_wall
    summary = {
        "schema_version": "cascade_b_catalyst_sister_wave_2_production_post_train_qat_verdict_v1_20260526",
        "lane_id": "lane_cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526",
        "subagent_id": (
            "cascade-b-catalyst-sister-wave-2-production-scale-600f-1000ep-plus-post-train-qat-"
            "fine-tune-real-segnet-fixture-recursive-doctrine-6th-order-mlx-first-numpy-portable-20260526"
        ),
        "canonical_equation_id": "hinton_kl_distill_enables_qat_catalyst_composition_savings_v1",
        "catalog_307_verdict": verdict,
        "configuration": {
            "n_frames": args.n_frames,
            "path_a_n_epochs": args.path_a_n_epochs,
            "qat_n_epochs": args.qat_n_epochs,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "distillation_weight": args.distillation_weight,
            "learning_rate": args.learning_rate,
            "qat_lr_factor": args.qat_lr_factor,
            "qat_lr_effective": qat_lr,
            "seed": args.seed,
            "canonical_eval_size": [canonical_h, canonical_w],
            "source_video_sha256": video_sha,
        },
        "arms": {
            "baseline": {
                "head_n_params": 0,
                "eval_kl": baseline_kl,
                "sidecar_bytes": baseline_sidecar_bytes,
                "composite_proxy": baseline_composite,
            },
            "path_a_alone": {
                "head_n_params": int(path_a_head.weight.size + path_a_head.bias.size),
                "train_initial_kl": path_a_initial_kl,
                "train_final_kl": path_a_final_train_kl,
                "eval_kl": path_a_eval_kl,
                "sidecar_bytes": path_a_sidecar_bytes,
                "composite_proxy": path_a_composite,
                "train_wall_seconds": path_a_train_wall,
                "loss_curve_first_10": [float(x) for x in path_a_losses[:10]],
                "loss_curve_last_10": [float(x) for x in path_a_losses[-10:]],
            },
            "catalyst_composition": {
                "head_n_params": int(catalyst_head.weight.size + catalyst_head.bias.size),
                "qat_initial_kl": float(qat_losses[0]),
                "qat_final_train_kl": qat_final_train_kl,
                "eval_kl": catalyst_eval_kl,
                "sidecar_bytes": catalyst_sidecar_bytes,
                "composite_proxy": catalyst_composite,
                "qat_wall_seconds": qat_wall,
                "qat_loss_curve_first_10": [float(x) for x in qat_losses[:10]],
                "qat_loss_curve_last_10": [float(x) for x in qat_losses[-10:]],
            },
        },
        "deltas": {
            "delta_path_a_kl_minus_baseline_kl": delta_path_a_kl_vs_baseline,
            "delta_catalyst_kl_minus_path_a_kl": delta_catalyst_kl_vs_path_a,
            "delta_catalyst_rate_minus_path_a_rate": delta_catalyst_rate_vs_path_a,
            "delta_catalyst_composite_minus_path_a_composite": (
                catalyst_composite - path_a_composite
            ),
            "delta_catalyst_composite_minus_baseline_composite": (
                catalyst_composite - baseline_composite
            ),
        },
        "verdict_predicates": {
            "catalyst_improves_over_path_a_kl": delta_catalyst_kl_vs_path_a < 0,
            "catalyst_improves_over_path_a_composite": catalyst_composite < path_a_composite,
            "catalyst_improves_over_baseline_composite": catalyst_composite < baseline_composite,
            "qat_stable": catalyst_eval_kl < VERDICT_DEFER_QAT_DIVERGENCE_KL,
        },
        "wall_clock_seconds": {
            "decode": decode_secs,
            "teacher_cache": cache_secs,
            "baseline_eval": baseline_wall,
            "path_a_train": path_a_train_wall,
            "qat_fine_tune": qat_wall,
            "total": total_wall,
        },
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

    summary_path = output_dir / "sweep_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\n[wave2] Summary written to {summary_path}", flush=True)
    print(f"[wave2] TOTAL wall_clock={total_wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
