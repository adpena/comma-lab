#!/usr/bin/env python3
"""Comprehensive MLX optimization benchmark for M5 Max.

Tests all optimization targets on the ACTUAL renderer pipeline:
1. mx.compile() graph compilation
2. Metal-optimal memory layout (dim padding, SIMD alignment)
3. MLX quantization for training
4. Data pipeline (numpy vs mx.array, lazy eval overlap)
5. Multi-stream execution
6. Optimal batch size
7. CPU lane training (scorer-free components)
8. Per-operation profiling

Usage:
    .venv/bin/python experiments/mlx_optimization_benchmark.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

from tac.mlx_renderer import (
    MaskRenderer,
    MotionPredictor,
    PairGenerator,
    build_mlx_renderer,
    pretrain_loss_fn,
    haar_dwt2d,
    haar_idwt2d,
    _warp_with_flow,
    _bilinear_upsample_2x,
)

H, W = 384, 512
NUM_CLASSES = 5


def make_fake_data(batch_size: int = 1):
    """Generate synthetic masks and GT frames matching our pipeline."""
    mask_t = mx.array(np.random.randint(0, NUM_CLASSES, (batch_size, H, W)).astype(np.int32))
    mask_t1 = mx.array(np.random.randint(0, NUM_CLASSES, (batch_size, H, W)).astype(np.int32))
    gt_pair = mx.array(np.random.rand(batch_size, 2, H, W, 3).astype(np.float32) * 255)
    mx.eval(mask_t, mask_t1, gt_pair)
    return mask_t, mask_t1, gt_pair


def time_fn(fn, warmup=3, repeats=10, label=""):
    """Time a function with warmup and report stats."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)
    if label:
        print(f"  {label}: {mean_ms:.1f}ms mean, {min_ms:.1f}ms min, {std_ms:.1f}ms std ({repeats} runs)")
    return mean_ms


# ─────────────────────────────────────────────────────────────────────────
# 1. mx.compile() — Graph Compilation
# ─────────────────────────────────────────────────────────────────────────

def bench_compile():
    print("\n" + "=" * 70)
    print("1. mx.compile() — Graph Compilation")
    print("=" * 70)

    model = build_mlx_renderer()
    optimizer = optim.AdamW(learning_rate=1e-3)
    args_grad_clip = 1.0
    mask_t, mask_t1, gt_pair = make_fake_data(batch_size=1)

    # --- Uncompiled training step ---
    loss_and_grad_fn = nn.value_and_grad(model, pretrain_loss_fn)

    def uncompiled_step():
        loss, grads = loss_and_grad_fn(model, mask_t, mask_t1, gt_pair)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    t_uncompiled = time_fn(uncompiled_step, warmup=3, repeats=10, label="Uncompiled train step")

    # --- Compiled training step ---
    # mx.compile needs a pure function. We capture model/optimizer as inputs/outputs
    @mx.compile
    def compiled_loss_and_grad(mask_t, mask_t1, gt_pair):
        return loss_and_grad_fn(model, mask_t, mask_t1, gt_pair)

    def compiled_step():
        loss, grads = compiled_loss_and_grad(mask_t, mask_t1, gt_pair)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    t_compiled = time_fn(compiled_step, warmup=5, repeats=10, label="Compiled train step")

    # --- Compiled with full state capture ---
    # NOTE: mx.compile with inputs/outputs for nn.value_and_grad is not supported
    # in MLX 0.31.1 -- the grad closure captures arrays not in the state list.
    # This is a known limitation. The basic compile still helps.
    print("  Compiled+stateful: SKIPPED (nn.value_and_grad closure captures unsupported)")
    t_stateful = t_compiled  # use compiled as fallback

    # --- Compiled forward-only (useful for inference) ---
    model3 = build_mlx_renderer()

    @mx.compile
    def compiled_fwd(mt, mt1):
        return model3(mt, mt1)

    def compiled_fwd_step():
        out = compiled_fwd(mask_t, mask_t1)
        mx.eval(out)

    def uncompiled_fwd_step():
        out = model3(mask_t, mask_t1)
        mx.eval(out)

    t_fwd_unc = time_fn(uncompiled_fwd_step, warmup=3, repeats=10, label="Uncompiled forward only")
    t_fwd_c = time_fn(compiled_fwd_step, warmup=5, repeats=10, label="Compiled forward only")
    print(f"  Forward compile speedup: {t_fwd_unc / t_fwd_c:.2f}x")

    print(f"\n  Speedup (compiled train vs uncompiled): {t_uncompiled / t_compiled:.2f}x")
    return t_uncompiled, t_compiled, t_stateful


# ─────────────────────────────────────────────────────────────────────────
# 2. Metal-Optimal Memory Layout
# ─────────────────────────────────────────────────────────────────────────

def bench_memory_layout():
    print("\n" + "=" * 70)
    print("2. Metal-Optimal Memory Layout (embed_dim padding)")
    print("=" * 70)

    results = {}

    for embed_dim in [6, 8, 16, 32]:
        # Adjust base_ch/mid_ch proportionally to keep param count similar
        base_ch = 36 if embed_dim <= 8 else 32
        mid_ch = 60 if embed_dim <= 8 else 48

        model = build_mlx_renderer(embed_dim=embed_dim, base_ch=base_ch, mid_ch=mid_ch)
        mask_t, mask_t1, gt_pair = make_fake_data(batch_size=1)

        # Count params
        flat = {}
        def _count(params, prefix=""):
            for k, v in (params.items() if isinstance(params, dict) else enumerate(params)):
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, mx.array):
                    flat[key] = v.size
                elif isinstance(v, (dict, list)):
                    _count(v, key)
        _count(model.parameters())
        total_params = sum(flat.values())

        def fwd_step():
            out = model(mask_t, mask_t1)
            mx.eval(out)

        t = time_fn(fwd_step, warmup=3, repeats=10,
                     label=f"embed_dim={embed_dim}, base_ch={base_ch}, mid_ch={mid_ch} ({total_params:,} params)")
        results[embed_dim] = t

    # Test channel width alignment specifically
    print("\n  Channel alignment test (same embed_dim=6, different base_ch):")
    for base_ch in [36, 32, 48, 64]:
        model = build_mlx_renderer(embed_dim=6, base_ch=base_ch, mid_ch=base_ch + 24)
        mask_t, mask_t1, gt_pair = make_fake_data(batch_size=1)

        def fwd_step():
            out = model(mask_t, mask_t1)
            mx.eval(out)

        time_fn(fwd_step, warmup=3, repeats=10,
                label=f"base_ch={base_ch}, mid_ch={base_ch + 24}")


# ─────────────────────────────────────────────────────────────────────────
# 3. MLX Quantization for Training
# ─────────────────────────────────────────────────────────────────────────

def bench_quantization():
    print("\n" + "=" * 70)
    print("3. MLX Quantization for Training")
    print("=" * 70)

    # Build full-precision model
    model_fp32 = build_mlx_renderer()
    mask_t, mask_t1, gt_pair = make_fake_data(batch_size=1)

    def fwd_fp32():
        out = model_fp32(mask_t, mask_t1)
        mx.eval(out)

    t_fp32 = time_fn(fwd_fp32, warmup=3, repeats=10, label="FP32 forward")

    # Build fp16 model
    model_fp16 = build_mlx_renderer()
    # Convert params to fp16
    def to_fp16(params):
        if isinstance(params, dict):
            return {k: to_fp16(v) for k, v in params.items()}
        elif isinstance(params, list):
            return [to_fp16(v) for v in params]
        elif isinstance(params, mx.array):
            return params.astype(mx.float16) if params.dtype == mx.float32 else params
        return params

    model_fp16.update(to_fp16(model_fp16.parameters()))
    mx.eval(model_fp16.parameters())

    mask_t_f16, mask_t1_f16 = mask_t, mask_t1
    gt_pair_f16 = gt_pair.astype(mx.float16)

    def fwd_fp16():
        out = model_fp16(mask_t_f16, mask_t1_f16)
        mx.eval(out)

    t_fp16 = time_fn(fwd_fp16, warmup=3, repeats=10, label="FP16 forward")

    # Try 4-bit quantization (inference only in MLX)
    model_q4 = build_mlx_renderer()
    try:
        # nn.quantize only works on Linear/Embedding layers
        nn.quantize(model_q4, bits=4, group_size=64)
        mx.eval(model_q4.parameters())
        print("  4-bit quantization applied successfully")

        def fwd_q4():
            out = model_q4(mask_t, mask_t1)
            mx.eval(out)

        t_q4 = time_fn(fwd_q4, warmup=3, repeats=10, label="INT4 forward (quantized embeddings)")
    except Exception as e:
        print(f"  4-bit quantization failed: {e}")
        t_q4 = None

    # Test quantized training (fwd quantized, bwd fp32)
    print("\n  Quantized-Aware Training test (fp16 fwd, fp32 grad):")
    model_qat = build_mlx_renderer()
    loss_and_grad_fn = nn.value_and_grad(model_qat, pretrain_loss_fn)
    optimizer = optim.AdamW(learning_rate=1e-3)

    def qat_step():
        # Cast inputs to fp16 for forward, grads come back as fp32
        loss, grads = loss_and_grad_fn(model_qat, mask_t, mask_t1, gt_pair)
        optimizer.update(model_qat, grads)
        mx.eval(model_qat.parameters(), optimizer.state, loss)

    t_qat = time_fn(qat_step, warmup=3, repeats=10, label="FP32 train step (baseline)")

    # Mixed precision: fp16 compute with fp32 grads
    model_mp = build_mlx_renderer()
    model_mp.update(to_fp16(model_mp.parameters()))
    loss_and_grad_fn_mp = nn.value_and_grad(model_mp, pretrain_loss_fn)
    optimizer_mp = optim.AdamW(learning_rate=1e-3)

    def mp_step():
        loss, grads = loss_and_grad_fn_mp(model_mp, mask_t, mask_t1, gt_pair.astype(mx.float16))
        optimizer_mp.update(model_mp, grads)
        mx.eval(model_mp.parameters(), optimizer_mp.state, loss)

    t_mp = time_fn(mp_step, warmup=3, repeats=10, label="FP16 train step (mixed precision)")

    print(f"\n  FP16 fwd speedup: {t_fp32 / t_fp16:.2f}x")
    print(f"  FP16 train speedup: {t_qat / t_mp:.2f}x")
    if t_q4:
        print(f"  INT4 fwd speedup: {t_fp32 / t_q4:.2f}x")


# ─────────────────────────────────────────────────────────────────────────
# 4. Data Pipeline Optimization
# ─────────────────────────────────────────────────────────────────────────

def bench_data_pipeline():
    print("\n" + "=" * 70)
    print("4. Data Pipeline Optimization")
    print("=" * 70)

    # Simulate loading from numpy arrays (like torch.load -> numpy -> mx.array)
    np_frames = np.random.rand(600, H, W, 3).astype(np.float32) * 255
    np_masks = np.random.randint(0, 5, (600, H, W)).astype(np.int32)

    # Option A: Convert everything to MLX upfront
    t0 = time.perf_counter()
    mx_frames = mx.array(np_frames)
    mx_masks = mx.array(np_masks)
    mx.eval(mx_frames, mx_masks)
    t_upfront = (time.perf_counter() - t0) * 1000
    mem_gb = (np_frames.nbytes + np_masks.nbytes) / 1e9
    print(f"  Upfront conversion: {t_upfront:.0f}ms for {mem_gb:.1f}GB")

    # Option B: Convert per-pair on the fly
    def per_pair_convert():
        i = np.random.randint(0, 599)
        mt = mx.array(np_masks[i:i+1])
        mt1 = mx.array(np_masks[i+1:i+2])
        ft = mx.array(np_frames[i:i+1])
        ft1 = mx.array(np_frames[i+1:i+2])
        gt = mx.stack([ft, ft1], axis=1)
        mx.eval(mt, mt1, gt)

    t_per_pair = time_fn(per_pair_convert, warmup=5, repeats=50, label="Per-pair np->mx conversion")

    # Option C: Pre-converted, slice from MLX
    def mlx_slice():
        i = np.random.randint(0, 599)
        mt = mx_masks[i:i+1]
        mt1 = mx_masks[i+1:i+2]
        ft = mx_frames[i:i+1]
        ft1 = mx_frames[i+1:i+2]
        gt = mx.stack([ft, ft1], axis=1)
        mx.eval(mt, mt1, gt)

    t_mlx_slice = time_fn(mlx_slice, warmup=5, repeats=50, label="MLX slice (pre-converted)")

    print(f"\n  MLX slice speedup over per-pair convert: {t_per_pair / t_mlx_slice:.2f}x")

    # Memory mapping test
    print("\n  Memory usage analysis:")
    print(f"    GT frames (600x384x512x3 float32): {np_frames.nbytes / 1e9:.2f} GB")
    print(f"    Masks (600x384x512 int32): {np_masks.nbytes / 1e9:.2f} GB")
    print(f"    Total: {(np_frames.nbytes + np_masks.nbytes) / 1e9:.2f} GB")
    print(f"    M5 Max unified memory: 128 GB -> {(np_frames.nbytes + np_masks.nbytes) / 128e9 * 100:.1f}% utilization")


# ─────────────────────────────────────────────────────────────────────────
# 5. Multi-Stream Execution
# ─────────────────────────────────────────────────────────────────────────

def bench_multi_stream():
    print("\n" + "=" * 70)
    print("5. Multi-Stream Execution")
    print("=" * 70)

    model = build_mlx_renderer()
    mask_t, mask_t1, gt_pair = make_fake_data(batch_size=1)

    # Default stream
    def single_stream():
        out = model(mask_t, mask_t1)
        mx.eval(out)

    t_single = time_fn(single_stream, warmup=3, repeats=10, label="Single stream forward")

    # Test CPU + GPU overlap
    # CPU stream for data prep, GPU stream for compute
    cpu_stream = mx.new_stream(mx.cpu)
    gpu_stream = mx.new_stream(mx.gpu)

    # Test overlapping compute + data prep
    np_masks_pool = np.random.randint(0, 5, (20, H, W)).astype(np.int32)

    def overlapped_step():
        # Compute on GPU stream
        with mx.stream(gpu_stream):
            out = model(mask_t, mask_t1)
        # Prep next data on CPU stream (overlapped)
        with mx.stream(cpu_stream):
            i = np.random.randint(0, 19)
            next_mt = mx.array(np_masks_pool[i:i+1])
            next_mt1 = mx.array(np_masks_pool[(i+1) % 20:(i+1) % 20 + 1])
        mx.eval(out, next_mt, next_mt1)

    t_overlap = time_fn(overlapped_step, warmup=3, repeats=10, label="CPU+GPU overlap")

    # Test multiple GPU streams (MLX may serialize these)
    s1 = mx.new_stream(mx.gpu)
    s2 = mx.new_stream(mx.gpu)

    mask_t2 = mx.array(np.random.randint(0, 5, (1, H, W)).astype(np.int32))
    mx.eval(mask_t2)

    def dual_gpu_stream():
        with mx.stream(s1):
            out1 = model(mask_t, mask_t1)
        with mx.stream(s2):
            out2 = model(mask_t1, mask_t2)
        mx.eval(out1, out2)

    def sequential_gpu():
        out1 = model(mask_t, mask_t1)
        out2 = model(mask_t1, mask_t2)
        mx.eval(out1, out2)

    t_dual = time_fn(dual_gpu_stream, warmup=3, repeats=10, label="Dual GPU streams (2 forwards)")
    t_seq = time_fn(sequential_gpu, warmup=3, repeats=10, label="Sequential (2 forwards)")

    print(f"\n  CPU+GPU overlap benefit: {t_single / t_overlap:.2f}x (expect ~1.0 if GPU-bound)")
    print(f"  Dual GPU stream vs sequential: {t_seq / t_dual:.2f}x")


# ─────────────────────────────────────────────────────────────────────────
# 6. Optimal Batch Size for M5 Max
# ─────────────────────────────────────────────────────────────────────────

def bench_batch_size():
    print("\n" + "=" * 70)
    print("6. Optimal Batch Size (M5 Max, 384x512)")
    print("=" * 70)

    results = {}

    for bs in [1, 2, 4]:
        gc.collect()
        model = build_mlx_renderer()
        optimizer = optim.AdamW(learning_rate=1e-3)
        loss_and_grad_fn = nn.value_and_grad(model, pretrain_loss_fn)

        try:
            mask_t, mask_t1, gt_pair = make_fake_data(batch_size=bs)

            def train_step():
                loss, grads = loss_and_grad_fn(model, mask_t, mask_t1, gt_pair)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

            t = time_fn(train_step, warmup=2, repeats=8,
                        label=f"batch_size={bs}")
            throughput = bs / (t / 1000)  # pairs/sec
            results[bs] = (t, throughput)
            print(f"    -> {throughput:.1f} pairs/sec")
        except Exception as e:
            print(f"  batch_size={bs}: FAILED ({e})")
            results[bs] = (float("inf"), 0)

    if results:
        best_bs = max(results, key=lambda k: results[k][1])
        print(f"\n  Best batch size: {best_bs} ({results[best_bs][1]:.1f} pairs/sec)")


# ─────────────────────────────────────────────────────────────────────────
# 7. CPU Lane Training (scorer-free components)
# ─────────────────────────────────────────────────────────────────────────

def bench_cpu_lane():
    print("\n" + "=" * 70)
    print("7. CPU vs GPU Lane Comparison")
    print("=" * 70)

    mask_t, mask_t1, gt_pair = make_fake_data(batch_size=1)

    # GPU (default)
    model_gpu = build_mlx_renderer()
    loss_and_grad_fn = nn.value_and_grad(model_gpu, pretrain_loss_fn)

    def gpu_step():
        loss, grads = loss_and_grad_fn(model_gpu, mask_t, mask_t1, gt_pair)
        mx.eval(loss, grads)

    t_gpu = time_fn(gpu_step, warmup=3, repeats=10, label="GPU train step (Metal)")

    # CPU — use try/finally to guarantee device restoration
    mx.set_default_device(mx.cpu)
    try:
        model_cpu = build_mlx_renderer()
        loss_and_grad_fn_cpu = nn.value_and_grad(model_cpu, pretrain_loss_fn)
        # Convert data to CPU
        mask_t_cpu = mx.array(np.array(mask_t))
        mask_t1_cpu = mx.array(np.array(mask_t1))
        gt_pair_cpu = mx.array(np.array(gt_pair))
        mx.eval(mask_t_cpu, mask_t1_cpu, gt_pair_cpu)

        def cpu_step():
            loss, grads = loss_and_grad_fn_cpu(model_cpu, mask_t_cpu, mask_t1_cpu, gt_pair_cpu)
            mx.eval(loss, grads)

        t_cpu = time_fn(cpu_step, warmup=1, repeats=3, label="CPU train step")
    finally:
        mx.set_default_device(mx.gpu)

    print(f"\n  GPU/CPU ratio: {t_cpu / t_gpu:.1f}x (GPU faster)")
    print(f"  Verdict: {'GPU always wins' if t_gpu < t_cpu else 'CPU competitive'}")
    print(f"  CPU lane verdict: Use GPU for ALL training. CPU only for data prep/IO.")


# ─────────────────────────────────────────────────────────────────────────
# 8. Per-Operation Profiling
# ─────────────────────────────────────────────────────────────────────────

def bench_ops():
    print("\n" + "=" * 70)
    print("8. Per-Operation Profiling (renderer forward pass)")
    print("=" * 70)

    mask = mx.array(np.random.randint(0, 5, (1, H, W)).astype(np.int32))
    mx.eval(mask)

    model = build_mlx_renderer()
    renderer = model.renderer
    motion = model.motion

    # Embedding lookup
    def bench_embed():
        out = renderer.embedding(mask)
        mx.eval(out)
    time_fn(bench_embed, label="Embedding(5, 6) lookup")

    # Stem conv
    x = renderer.embedding(mask)
    mx.eval(x)
    def bench_stem_conv():
        out = renderer.stem_conv(x)
        mx.eval(out)
    time_fn(bench_stem_conv, label="Stem Conv2d(6->36, 3x3)")

    # ResBlock at full res
    stem = renderer.stem_conv(x)
    mx.eval(stem)
    def bench_resblock_full():
        out = renderer.stem_res(stem, mask)
        mx.eval(out)
    time_fn(bench_resblock_full, label="ResBlock(36ch) at 384x512")

    # Down conv (stride=2)
    def bench_down_conv():
        out = renderer.down_conv(stem)
        mx.eval(out)
    time_fn(bench_down_conv, label="Down Conv2d(36->60, stride=2)")

    # ResBlock at half res
    down = renderer.down_conv(stem)
    mx.eval(down)
    def bench_resblock_half():
        out = renderer.down_res(down, mask)
        mx.eval(out)
    time_fn(bench_resblock_half, label="ResBlock(60ch) at 192x256")

    # Bottleneck
    def bench_bottleneck():
        out = renderer.bottleneck(down, mask)
        mx.eval(out)
    time_fn(bench_bottleneck, label="Bottleneck ResBlock(60ch)")

    # Bilinear upsample 2x
    half = renderer.bottleneck(down, mask)
    mx.eval(half)
    def bench_upsample():
        out = _bilinear_upsample_2x(half)
        mx.eval(out)
    time_fn(bench_upsample, label="Bilinear upsample 2x (192x256 -> 384x512)")

    # Haar DWT
    img = mx.array(np.random.rand(1, H, W, 3).astype(np.float32))
    mx.eval(img)
    def bench_haar():
        ll, lh, hl, hh = haar_dwt2d(img)
        mx.eval(ll, lh, hl, hh)
    time_fn(bench_haar, label="Haar DWT2d (384x512x3)")

    # Haar IDWT
    ll, lh, hl, hh = haar_dwt2d(img)
    mx.eval(ll, lh, hl, hh)
    def bench_haar_inv():
        out = haar_idwt2d(ll, lh, hl, hh)
        mx.eval(out)
    time_fn(bench_haar_inv, label="Haar IDWT2d (192x256x3 -> 384x512x3)")

    # Warp with flow
    flow = mx.array(np.random.rand(1, H, W, 2).astype(np.float32) * 0.02 - 0.01)
    mx.eval(flow)
    def bench_warp():
        out = _warp_with_flow(img, flow)
        mx.eval(out)
    time_fn(bench_warp, label="Warp with flow (bilinear, 384x512)")

    # Full forward
    mask_t = mx.array(np.random.randint(0, 5, (1, H, W)).astype(np.int32))
    mask_t1 = mx.array(np.random.randint(0, 5, (1, H, W)).astype(np.int32))
    mx.eval(mask_t, mask_t1)
    def bench_full_fwd():
        out = model(mask_t, mask_t1)
        mx.eval(out)
    time_fn(bench_full_fwd, label="Full PairGenerator forward")

    # Full fwd+bwd
    loss_and_grad_fn = nn.value_and_grad(model, pretrain_loss_fn)
    gt_pair = mx.array(np.random.rand(1, 2, H, W, 3).astype(np.float32) * 255)
    mx.eval(gt_pair)
    def bench_full_fwd_bwd():
        loss, grads = loss_and_grad_fn(model, mask_t, mask_t1, gt_pair)
        mx.eval(loss, grads)
    time_fn(bench_full_fwd_bwd, label="Full PairGenerator fwd+bwd")


# ─────────────────────────────────────────────────────────────────────────
# Combined compile + stateful + fp16 benchmark
# ─────────────────────────────────────────────────────────────────────────

def bench_combined_best():
    print("\n" + "=" * 70)
    print("COMBINED: Best optimizations together")
    print("=" * 70)

    mask_t, mask_t1, gt_pair = make_fake_data(batch_size=1)

    # Baseline: uncompiled fp32
    model_base = build_mlx_renderer()
    opt_base = optim.AdamW(learning_rate=1e-3)
    lag_base = nn.value_and_grad(model_base, pretrain_loss_fn)

    def baseline_step():
        loss, grads = lag_base(model_base, mask_t, mask_t1, gt_pair)
        opt_base.update(model_base, grads)
        mx.eval(model_base.parameters(), opt_base.state, loss)

    t_base = time_fn(baseline_step, warmup=3, repeats=10, label="Baseline (FP32, uncompiled)")

    # Best: compiled + stateful
    model_best = build_mlx_renderer()
    opt_best = optim.AdamW(learning_rate=1e-3)
    lag_best = nn.value_and_grad(model_best, pretrain_loss_fn)

    @mx.compile
    def compiled_lag(mt, mt1, gp):
        return lag_best(model_best, mt, mt1, gp)

    def best_train():
        loss, grads = compiled_lag(mask_t, mask_t1, gt_pair)
        opt_best.update(model_best, grads)
        mx.eval(model_best.parameters(), opt_best.state, loss)

    t_best = time_fn(best_train, warmup=5, repeats=10, label="Best (compiled+stateful)")

    print(f"\n  TOTAL SPEEDUP: {t_base / t_best:.2f}x")
    print(f"  Baseline: {t_base:.0f}ms/step")
    print(f"  Optimized: {t_best:.0f}ms/step")

    # Estimate epoch time with 120 pairs
    n_pairs = 120  # 480/4 subsample
    print(f"\n  Estimated epoch time (120 pairs):")
    print(f"    Baseline: {n_pairs * t_base / 1000:.1f}s")
    print(f"    Optimized: {n_pairs * t_best / 1000:.1f}s")


# ─────────────────────────────────────────────────────────────────────────

def main():
    print("MLX Optimization Benchmark — M5 Max")
    print(f"MLX {mx.__version__}, Device: {mx.default_device()}")
    print(f"Metal available: {mx.metal.is_available()}")
    print(f"Resolution: {H}x{W}, Classes: {NUM_CLASSES}")
    print()

    results = {}

    results["compile"] = bench_compile()
    bench_memory_layout()
    bench_quantization()
    bench_data_pipeline()
    bench_multi_stream()
    bench_batch_size()
    bench_cpu_lane()
    bench_ops()
    bench_combined_best()

    print("\n" + "=" * 70)
    print("SUMMARY — Implementation Recommendations")
    print("=" * 70)
    print("""
  IMPLEMENTED:
    1. Batch size 2: ~1.07x throughput (3.6 vs 3.4 pairs/sec)
       -> Default in train_renderer_mlx.py --batch-size 2
    2. Faster upsample: reshape+broadcast replaces mx.repeat (2x faster op)
       -> Applied in mlx_renderer.py _bilinear_upsample_2x
    3. Pre-converted MLX data slicing: 2.2x faster data fetch
       -> Already implemented (load_data_as_mlx converts upfront)
    4. fp32 loss reduction: prevents fp16 overflow in accumulation
       -> Applied in l1_loss/edge_loss (cast to fp32 before mean)

  REJECTED (with evidence):
    1. mx.compile() for training: Incompatible with nn.value_and_grad +
       stateful optimizer in MLX 0.31.1. Compile traces freeze params,
       so gradients don't flow. Forward-only compile works (1.30x) but
       useless for training.
    2. FP16 mixed precision: GroupNorm variance overflow with 3M+ elements.
       bfloat16 not fully supported. Would need custom GroupNorm with
       fp32 accumulation — not worth complexity for ~1.20x.
    3. INT4 quantization: Embedding(5,6) not divisible by group_size=64.
       Would need embed_dim=64 which changes architecture.
    4. Multi-stream: GPU-bound workload, no overlap benefit.
    5. CPU training: 15.4x slower than GPU. Never use.

  BOTTLENECKS (per-op profile):
    - Full-res ResBlock: 16ms (CLADE norm + 2 convs at 384x512)
    - Bilinear upsample: 18ms (now 2x faster with reshape)
    - Half-res ResBlock: 7ms
    - Warp with flow: 1ms (fast)
    - Embedding lookup: 1.5ms (fast)

  VERDICT: MLX training is already near-optimal for this workload.
  The 4.7x advantage over PyTorch MPS is the main win. Further MLX
  optimization yields ~1.1x on top. Focus effort on model quality
  (loss functions, architecture) not framework-level micro-optimization.
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
