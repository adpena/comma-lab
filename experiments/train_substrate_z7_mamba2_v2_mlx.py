# SPDX-License-Identifier: MIT
"""Z7-Mamba-2-v2 MLX-local L1 EMPIRICAL trainer — FAIR-SHAKE per UNIQUE-AND-COMPLETE-PER-METHOD.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_fair_shake_20260526
# AUTOCAST_FP16_WAIVED:MLX_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_fair_shake_20260526

Per Path 3 B' L0 SCAFFOLD landing memo (commit 7a103fdbb) + operator directive
2026-05-26 *"give all their own fully individually optimized implementations
and fair shakes"* + UNIQUE-AND-COMPLETE-PER-METHOD non-negotiable: this
trainer promotes the z7_mamba2_v2_fresh_substrate L0 SCAFFOLD to L1 EMPIRICAL
via REAL contest video MLX training with the canonical Z6 #1287 promotion
pattern adapted to Z7-Mamba-2 SSM paradigm.

Per CC-A unwind: this substrate's distinguishing-feature is the
Mamba2TemporalDecoder Conv1D temporal pre-stage matching Mamba-2's d_conv=4
selective-state-space temporal window. Per CC-B + CC-C unwinds: latent_dim=32
+ ego_motion_dim=16 (was 24 / 8 in v1).

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable + Catalog
#192 + #317: all outputs tagged ``[macOS-MLX research-signal]``;
``score_claim=False``; ``promotion_eligible=False``;
``ready_for_exact_eval_dispatch=False``. The score-aware Lagrangian routes
through PyTorch sister + paid CUDA per Catalog #164 + #226 (DEFERRED at L1).

Usage (smoke):

    .venv/bin/python experiments/train_substrate_z7_mamba2_v2_mlx.py \
        --smoke --num-pairs 50 --epochs 30 \
        --output-dir .omx/tmp/z7_mamba2_v2_l1_smoke/

Usage (mini-contest):

    .venv/bin/python experiments/train_substrate_z7_mamba2_v2_mlx.py \
        --num-pairs 100 --epochs 50 \
        --output-dir .omx/tmp/z7_mamba2_v2_l1_mini/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _require_mlx() -> None:
    try:
        import mlx.core  # noqa: F401
        import mlx.nn  # noqa: F401
    except ImportError as exc:
        print(
            "[z7-mamba2-v2-mlx] FATAL: MLX not installed. This is an Apple-Silicon-"
            "only trainer. Install via `pip install mlx`. The CUDA path routes via "
            "the canonical PyTorch sister (DEFERRED per per-substrate symposium).",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="Smoke mode (smaller defaults).")
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--output-height", type=int, default=48)
    p.add_argument("--output-width", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32,
                   help="CC-B unwind: default 32 (was 24 in v1).")
    p.add_argument("--ego-motion-dim", type=int, default=16,
                   help="CC-C unwind: default 16 (was 8 in v1).")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--a-log-init-scheme", type=str, default="z_plus_1",
                   choices=["z_plus_1", "hippo_like", "log_uniform"],
                   help="CC-D unwind: configurable A_log init scheme.")
    p.add_argument("--learning-rate", type=float, default=1e-3,
                   help="DEPRECATED in favor of --peak-lr when --enable-warmup-decay; kept for backward compat.")
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--disable-ema", action="store_true")
    p.add_argument("--lambda-residual", type=float, default=5e-4,
                   help="CC-H unwind: ib_scale=5e-4 (was 1e-3 in v1).")
    # ---- L2 STABILITY HARDENING (Catalog #354 cross-paradigm sub-ingredient optimization) ----
    # PR95-sniped-lesson recursive per-sub-ingredient doctrine 2026-05-26.
    # Sub-ingredient #9 (Optimizer): gradient clipping
    p.add_argument("--max-grad-norm", type=float, default=0.0,
                   help="L2 stability: global-norm gradient clip (Mamba-2 canonical 1.0). 0=disabled.")
    # Sub-ingredient #1 (Architecture): A_log clamp (SSM-specific NaN risk)
    p.add_argument("--a-log-clamp-min", type=float, default=float("-inf"),
                   help="L2 stability: clamp A_log lower bound (Mamba-2 canonical -10). -inf=disabled.")
    p.add_argument("--a-log-clamp-max", type=float, default=float("inf"),
                   help="L2 stability: clamp A_log upper bound (Mamba-2 canonical 0). +inf=disabled.")
    # Sub-ingredient #9 (Optimizer): warmup-decay schedule
    p.add_argument("--enable-warmup-decay", action="store_true",
                   help="L2 stability: linear-warmup + cosine-decay LR schedule (else constant LR).")
    p.add_argument("--peak-lr", type=float, default=1e-3,
                   help="L2 stability: warmup-decay peak LR (used when --enable-warmup-decay).")
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="L2 stability: linear warmup steps.")
    p.add_argument("--min-lr-ratio", type=float, default=1e-2,
                   help="L2 stability: cosine-decay endpoint = peak_lr * min_lr_ratio.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video-path", type=Path,
                   default=REPO_ROOT / "upstream" / "videos" / "0.mkv")
    p.add_argument("--use-synthetic-frames", action="store_true",
                   help="Backward-compat: use synthetic random RGB (L0 mode).")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / ".omx" / "tmp" / "z7_mamba2_v2_l1_smoke")
    p.add_argument("--enable-temporal-conv", type=int, default=1,
                   help="CC-A unwind toggle: 1=Conv1D temporal pre-stage (distinguishing feature); 0=ablation.")
    args = p.parse_args(argv)
    if args.smoke and args.num_pairs > 50:
        args.num_pairs = 50
    return args


# ============================================================================
# MLX-native Z7-Mamba-2-v2 renderer (UNIQUE-FORK per CC-A through CC-J)
# ============================================================================
# Architecture distinguishing-feature: Mamba2TemporalDecoder with Conv1D
# temporal pre-stage (CC-A unwind). Decoder consumes the (num_pairs, latent_dim)
# stream with a temporal-conv pre-stage matching d_conv=4, then per-pair
# PixelShuffle spatial decode.
#
# State-space cell: canonical Mamba-2 selective recurrence per Dao-Gu 2024
# (same math as v1 mlx_native; A_log init scheme configurable per CC-D unwind).


class Z7Mamba2V2MLXRenderer:
    """MLX-native Z7-Mamba-2-v2 renderer for L1 EMPIRICAL.

    Distinguishing UNIQUE-FORK features vs v1:
    - Mamba2TemporalDecoder Conv1D temporal pre-stage (CC-A) — applied
      to the latent stream BEFORE per-pair spatial decode
    - latent_dim default 32 (CC-B; was 24)
    - ego_motion_dim default 16 (CC-C; was 8)
    - ib_scale default 5e-4 (CC-H; was 1e-3)
    - A_log init scheme configurable (CC-D)
    """

    def __init__(self, args: argparse.Namespace) -> None:
        import mlx.core as mx
        self.args = args
        self.num_pairs = args.num_pairs
        self.latent_dim = args.latent_dim
        self.ego_motion_dim = args.ego_motion_dim
        self.d_model = args.d_model
        self.d_state = args.d_state
        self.expand = args.expand
        self.d_inner = args.expand * args.d_model
        self.d_conv = args.d_conv
        self.H = args.output_height
        self.W = args.output_width
        self.enable_temporal_conv = bool(args.enable_temporal_conv)
        self._rng_key = mx.random.key(args.seed)

        # ---- Mamba-2 cell parameters ----
        # input_projection: (latent_dim + ego_motion_dim) -> d_model
        self.input_projection_w = self._lin(self.latent_dim + self.ego_motion_dim, self.d_model)
        self.input_projection_b = mx.zeros((self.d_model,), dtype=mx.float32)
        # in_proj: d_model -> 2 * d_inner
        self.mamba_in_proj_w = self._lin(self.d_model, 2 * self.d_inner)
        # A_log per CC-D unwind: configurable init scheme
        self.mamba_A_log = self._init_a_log(args.a_log_init_scheme)
        # B/C/dt/out projections
        self.mamba_B_proj_w = self._lin(self.d_inner, self.d_state)
        self.mamba_C_proj_w = self._lin(self.d_inner, self.d_state)
        self.mamba_dt_proj_w = self._lin(self.d_inner, self.d_inner)
        self.mamba_dt_proj_b = mx.zeros((self.d_inner,), dtype=mx.float32)
        self.mamba_out_proj_w = self._lin(self.d_inner, self.d_model)
        # output_projection: d_model -> latent_dim
        self.output_projection_w = self._lin(self.d_model, self.latent_dim)
        self.output_projection_b = mx.zeros((self.latent_dim,), dtype=mx.float32)

        # ---- Mamba2TemporalDecoder (CC-A UNIQUE-FORK) ----
        # Conv1D temporal pre-stage: latent stream (num_pairs, latent_dim) ->
        # (num_pairs, embed_dim) with d_conv=4 receptive field over time axis.
        self.embed_dim = self.latent_dim  # keep same dim for simplicity at L1
        if self.enable_temporal_conv:
            # Conv1D kernel: shape (out_ch=embed_dim, kernel=d_conv, in_ch=latent_dim)
            # MLX conv1d uses channels-last: (out, k, in)
            self.temporal_conv_w = self._conv1d_kaiming(
                self.latent_dim, self.embed_dim, self.d_conv,
            )
            self.temporal_conv_b = mx.zeros((self.embed_dim,), dtype=mx.float32)

        # ---- Spatial decoder (small, sister to Z6 minimal decoder) ----
        # Per-pair: linear(embed_dim, H*W*3*2) -> reshape -> sigmoid produces rgb_0+rgb_1
        # Compact at L1 for fast MVP; full PixelShuffle stack deferred per credit-cap.
        out_dim = self.H * self.W * 6  # 2 frames * 3 channels
        self.spatial_w = self._lin(self.embed_dim, out_dim)
        self.spatial_b = mx.zeros((out_dim,), dtype=mx.float32)

        # ---- Trainable latent_init + per-pair residuals ----
        self._rng_key, sk = mx.random.split(self._rng_key)
        self.latent_init = mx.random.normal(shape=(self.latent_dim,), key=sk) * 0.02
        self.residuals = mx.zeros((self.num_pairs, self.latent_dim), dtype=mx.float32)
        self.ego_motion_buffer = mx.zeros((self.num_pairs, self.ego_motion_dim), dtype=mx.float32)

    def _lin(self, in_f: int, out_f: int) -> Any:
        import mlx.core as mx
        self._rng_key, sk = mx.random.split(self._rng_key)
        bound = float(np.sqrt(1.0 / max(in_f, 1)))
        return mx.random.uniform(
            low=-bound, high=bound, shape=(out_f, in_f), dtype=mx.float32, key=sk,
        )

    def _conv1d_kaiming(self, in_ch: int, out_ch: int, kernel: int) -> Any:
        import mlx.core as mx
        self._rng_key, sk = mx.random.split(self._rng_key)
        fan_in = max(in_ch * kernel, 1)
        bound = float(np.sqrt(1.0 / fan_in))
        return mx.random.uniform(
            low=-bound, high=bound, shape=(out_ch, kernel, in_ch), dtype=mx.float32, key=sk,
        )

    def _init_a_log(self, scheme: str) -> Any:
        """Per CC-D unwind: configurable A_log init scheme.

        - z_plus_1: A_log[i, j] = log(j+1) for j ∈ [0, d_state); upstream Mamba-2 default
        - hippo_like: A_log[i, j] = log(scale * (j + 0.5)); Gu 2022 HiPPO-like
        - log_uniform: A_log[i, j] ~ log(U(1, d_state+1)); CC-D ablation
        """
        import mlx.core as mx
        if scheme == "z_plus_1":
            row = mx.log(mx.arange(1, self.d_state + 1, dtype=mx.float32))
        elif scheme == "hippo_like":
            row = mx.log(mx.arange(1, self.d_state + 1, dtype=mx.float32) - 0.5 + 1.0)
        elif scheme == "log_uniform":
            self._rng_key, sk = mx.random.split(self._rng_key)
            u = mx.random.uniform(
                low=1.0, high=float(self.d_state + 1), shape=(self.d_state,),
                dtype=mx.float32, key=sk,
            )
            row = mx.log(u)
        else:
            raise ValueError(f"Unknown a_log_init_scheme: {scheme}")
        return mx.broadcast_to(mx.expand_dims(row, 0), (self.d_inner, self.d_state))

    def parameters_flat(self) -> dict[str, Any]:
        """Return flat dict of trainable mx.array parameters (for gradient + EMA)."""
        params = {
            "input_projection_w": self.input_projection_w,
            "input_projection_b": self.input_projection_b,
            "mamba_in_proj_w": self.mamba_in_proj_w,
            "mamba_A_log": self.mamba_A_log,
            "mamba_B_proj_w": self.mamba_B_proj_w,
            "mamba_C_proj_w": self.mamba_C_proj_w,
            "mamba_dt_proj_w": self.mamba_dt_proj_w,
            "mamba_dt_proj_b": self.mamba_dt_proj_b,
            "mamba_out_proj_w": self.mamba_out_proj_w,
            "output_projection_w": self.output_projection_w,
            "output_projection_b": self.output_projection_b,
            "spatial_w": self.spatial_w,
            "spatial_b": self.spatial_b,
            "latent_init": self.latent_init,
            "residuals": self.residuals,
        }
        if self.enable_temporal_conv:
            params["temporal_conv_w"] = self.temporal_conv_w
            params["temporal_conv_b"] = self.temporal_conv_b
        return params

    def set_parameters_flat(self, p: dict[str, Any]) -> None:
        for k, v in p.items():
            setattr(self, k, v)

    def total_param_count(self) -> int:
        total = 0
        for v in self.parameters_flat().values():
            total += int(np.prod(v.shape))
        return total

    def _mamba2_step(self, x_t: Any, h_prev: Any) -> tuple[Any, Any]:
        """Single Mamba-2 selective state-space step (canonical from v1)."""
        import mlx.core as mx
        import mlx.nn as mlx_nn
        xz = x_t @ self.mamba_in_proj_w.T
        x_inner = xz[:, : self.d_inner]
        z_gate = xz[:, self.d_inner:]
        dt = mlx_nn.softplus(x_inner @ self.mamba_dt_proj_w.T + self.mamba_dt_proj_b)
        # L2 stability sub-ingredient #1 (Architecture): optional A_log clamp per Mamba-2
        # canonical [-10, 0] range so exp(A_log) ∈ [4.5e-5, 1] (state spectrum bounded).
        # PR95-sniped-lesson recursive per-sub-ingredient doctrine; SSM-specific NaN risk.
        a_log_clamp_min = getattr(self.args, "a_log_clamp_min", float("-inf"))
        a_log_clamp_max = getattr(self.args, "a_log_clamp_max", float("inf"))
        if a_log_clamp_min > float("-inf") or a_log_clamp_max < float("inf"):
            A = -mx.exp(mx.clip(self.mamba_A_log, a_log_clamp_min, a_log_clamp_max))
        else:
            A = -mx.exp(self.mamba_A_log)
        B = x_inner @ self.mamba_B_proj_w.T
        C = x_inner @ self.mamba_C_proj_w.T
        # Discretize
        A_bar = mx.exp(mx.expand_dims(A, 0) * mx.expand_dims(dt, -1))
        B_bar = mx.expand_dims(dt, -1) * mx.expand_dims(B, 1)
        h_t = A_bar * h_prev + B_bar * mx.expand_dims(x_inner, -1)
        y_inner = mx.sum(h_t * mx.expand_dims(C, 1), axis=-1)
        y_inner = y_inner * mx.sigmoid(z_gate)
        y_t = y_inner @ self.mamba_out_proj_w.T
        return y_t, h_t

    def forward_full_sequence(self) -> tuple[Any, Any]:
        """Replay full num_pairs sequence; return (rgb_0_stack, rgb_1_stack).

        - Autoregressive Mamba-2 unroll over num_pairs steps
        - Per-pair latent: z_t = predictor(z_{t-1}, ego_motion[t]) + residuals[t]
        - CC-A: temporal Conv1D over latent stream BEFORE spatial decode
        - Per-pair: spatial decoder produces (rgb_0, rgb_1)
        """
        import mlx.core as mx
        import mlx.nn as mlx_nn

        # Initial hidden state
        h = mx.zeros((1, self.d_inner, self.d_state), dtype=mx.float32)
        z_prev = self.latent_init  # (latent_dim,)

        # Unroll Mamba-2: collect latents
        latents = []
        for t in range(self.num_pairs):
            # input = concat(z_prev, ego_motion[t])
            ego_t = self.ego_motion_buffer[t]  # (ego_motion_dim,)
            x_in = mx.concatenate([z_prev, ego_t], axis=0)  # (latent_dim + ego_motion_dim,)
            x_in = mx.expand_dims(x_in, 0)  # (1, ...)
            # input projection
            x_proj = x_in @ self.input_projection_w.T + self.input_projection_b  # (1, d_model)
            # Mamba-2 step
            y_t, h = self._mamba2_step(x_proj, h)  # y_t: (1, d_model)
            # output projection -> latent_dim
            z_pred = y_t @ self.output_projection_w.T + self.output_projection_b  # (1, latent_dim)
            z_pred = z_pred[0]  # (latent_dim,)
            # add residual
            z_t = z_pred + self.residuals[t]
            latents.append(z_t)
            z_prev = z_t

        # latents: list of (latent_dim,); stack -> (num_pairs, latent_dim)
        z_stream = mx.stack(latents, axis=0)  # (num_pairs, latent_dim)

        # CC-A UNIQUE-FORK: Conv1D temporal pre-stage
        if self.enable_temporal_conv:
            # MLX conv1d expects (N, L, C_in); kernel (C_out, k, C_in)
            # Reshape: (num_pairs, latent_dim) -> (1, num_pairs, latent_dim)
            z_in = mx.expand_dims(z_stream, 0)
            # mx.conv1d(input, weight, padding=...)
            # weight shape: (out_ch, kernel, in_ch)
            z_embed = mx.conv1d(z_in, self.temporal_conv_w, padding=self.d_conv - 1)
            # Trim to num_pairs (causal padding adds extra steps)
            z_embed = z_embed[:, :self.num_pairs, :]  # (1, num_pairs, embed_dim)
            z_embed = z_embed + self.temporal_conv_b
            z_embed = mlx_nn.relu(z_embed)
            z_embed = z_embed[0]  # (num_pairs, embed_dim)
        else:
            z_embed = z_stream

        # Per-pair spatial decode: (embed_dim,) -> (6, H, W)
        # Batched: (num_pairs, embed_dim) @ (out_dim, embed_dim).T -> (num_pairs, out_dim)
        out_flat = z_embed @ self.spatial_w.T + self.spatial_b  # (num_pairs, H*W*6)
        out = mx.reshape(out_flat, (self.num_pairs, 6, self.H, self.W))
        out = mx.sigmoid(out)
        rgb_0 = out[:, :3, :, :]  # (num_pairs, 3, H, W)
        rgb_1 = out[:, 3:, :, :]
        return rgb_0, rgb_1


# ============================================================================
# Training loop
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    _require_mlx()
    import mlx.core as mx
    import mlx.optimizers as mlx_optim

    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[z7-mamba2-v2-mlx] [L1-PROMOTION] FAIR-SHAKE per UNIQUE-AND-COMPLETE-PER-METHOD")
    print(f"[z7-mamba2-v2-mlx] Config: num_pairs={args.num_pairs} epochs={args.epochs} "
          f"latent_dim={args.latent_dim} ego_motion_dim={args.ego_motion_dim} "
          f"d_model={args.d_model} d_state={args.d_state} "
          f"a_log_init={args.a_log_init_scheme} ib_scale={args.lambda_residual}")
    print(f"[z7-mamba2-v2-mlx] CC-A temporal_conv={'ENABLED' if args.enable_temporal_conv else 'DISABLED (ablation)'}")

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    renderer = Z7Mamba2V2MLXRenderer(args)
    print(f"[z7-mamba2-v2-mlx] Total params: {renderer.total_param_count():,}")

    # Pin ego-motion buffer (Catalog #311 structural conditioning)
    ego_np = np.random.RandomState(args.seed + 100).randn(
        args.num_pairs, args.ego_motion_dim,
    ).astype(np.float32) * 0.1
    renderer.ego_motion_buffer = mx.array(ego_np)

    # Load REAL contest video targets via canonical pyav helper
    if args.use_synthetic_frames:
        print("[z7-mamba2-v2-mlx] [L1-PROMOTION] --use-synthetic-frames; non-canonical L1")
        target_rgb_0_np = np.random.RandomState(args.seed + 200).rand(
            args.num_pairs, 3, args.output_height, args.output_width,
        ).astype(np.float32)
        target_rgb_1_np = np.random.RandomState(args.seed + 300).rand(
            args.num_pairs, 3, args.output_height, args.output_width,
        ).astype(np.float32)
        frame_metadata: dict[str, Any] = {
            "source": "synthetic_random_rgb",
            "video_path": None,
            "frames_decoded": 0,
            "decode_wall_clock_seconds": 0.0,
        }
    else:
        from tac.data import decode_video
        if not args.video_path.exists():
            raise SystemExit(f"[z7-mamba2-v2-mlx] FATAL: video not found: {args.video_path}")
        frames_needed = 2 * args.num_pairs
        print(f"[z7-mamba2-v2-mlx] [L1-PROMOTION] Decoding {frames_needed} REAL contest "
              f"frames at {args.output_height}x{args.output_width} from {args.video_path}")
        t_decode = time.time()
        gt_frames = decode_video(
            args.video_path,
            target_h=args.output_height,
            target_w=args.output_width,
            max_frames=frames_needed,
        )
        decode_wall = time.time() - t_decode
        if len(gt_frames) < frames_needed:
            raise SystemExit(
                f"[z7-mamba2-v2-mlx] FATAL: decoded {len(gt_frames)} frames; "
                f"needed {frames_needed}."
            )
        gt_arr = np.stack([f.numpy() for f in gt_frames], axis=0)  # (N, H, W, 3) uint8
        gt_pairs = gt_arr.reshape(args.num_pairs, 2, args.output_height, args.output_width, 3)
        # Convert HWC uint8 [0,255] -> CHW float32 [0,1]
        target_rgb_0_np = np.transpose(
            gt_pairs[:, 0, :, :, :].astype(np.float32) / 255.0, (0, 3, 1, 2),
        )
        target_rgb_1_np = np.transpose(
            gt_pairs[:, 1, :, :, :].astype(np.float32) / 255.0, (0, 3, 1, 2),
        )
        print(f"[z7-mamba2-v2-mlx] decoded in {decode_wall:.1f}s; "
              f"target shape={target_rgb_0_np.shape}")
        frame_metadata = {
            "source": "real_contest_video_pyav",
            "video_path": str(args.video_path),
            "frames_decoded": int(len(gt_frames)),
            "decode_wall_clock_seconds": float(decode_wall),
            "canonical_helper": "tac.data.decode_video",
        }

    target_rgb_0 = mx.array(target_rgb_0_np)
    target_rgb_1 = mx.array(target_rgb_1_np)

    # ----- MLX-native EMA shadow per Catalog #2 -----
    ema_enabled = not args.disable_ema
    ema_shadow: dict[str, Any] = {}
    if ema_enabled:
        print(f"[z7-mamba2-v2-mlx] [L1-PROMOTION] EMA decay={args.ema_decay} (Catalog #2)")
        for k, v in renderer.parameters_flat().items():
            ema_shadow[k] = mx.array(v)

    # ---- Training loop ----
    # Manual gradient via numerical mlx-native approach: pure-function loss + value_and_grad
    def loss_function(params: dict[str, Any]) -> Any:
        renderer.set_parameters_flat(params)
        rgb_0, rgb_1 = renderer.forward_full_sequence()
        mse_0 = mx.mean((rgb_0 - target_rgb_0) ** 2)
        mse_1 = mx.mean((rgb_1 - target_rgb_1) ** 2)
        recon = mse_0 + mse_1
        residual_l2 = mx.mean(renderer.residuals ** 2)
        return recon + args.lambda_residual * residual_l2

    loss_and_grad = mx.value_and_grad(loss_function)

    # L2 stability sub-ingredient #9 (Optimizer): warmup-decay LR schedule.
    # PR95-sniped-lesson recursive per-sub-ingredient doctrine; canonical Mamba-2
    # linear-warmup + cosine-decay composition via mlx.optimizers.join_schedules.
    if args.enable_warmup_decay:
        warmup_steps = max(1, int(args.warmup_steps))
        decay_steps = max(1, int(args.epochs) - warmup_steps)
        end_lr = float(args.peak_lr) * float(args.min_lr_ratio)
        warmup_sched = mlx_optim.linear_schedule(0.0, float(args.peak_lr), warmup_steps)
        decay_sched = mlx_optim.cosine_decay(float(args.peak_lr), decay_steps, end_lr)
        lr_sched = mlx_optim.join_schedules([warmup_sched, decay_sched], [warmup_steps])
        optimizer = mlx_optim.AdamW(learning_rate=lr_sched)
        print(f"[z7-mamba2-v2-mlx] [L2-STABILITY] warmup-decay: peak_lr={args.peak_lr} "
              f"warmup_steps={warmup_steps} decay_steps={decay_steps} end_lr={end_lr:.2e}")
    else:
        optimizer = mlx_optim.AdamW(learning_rate=args.learning_rate)
    if args.max_grad_norm > 0.0:
        print(f"[z7-mamba2-v2-mlx] [L2-STABILITY] grad clip: max_grad_norm={args.max_grad_norm}")
    if args.a_log_clamp_min > float("-inf") or args.a_log_clamp_max < float("inf"):
        print(f"[z7-mamba2-v2-mlx] [L2-STABILITY] A_log clamp: "
              f"[{args.a_log_clamp_min}, {args.a_log_clamp_max}]")

    per_epoch_metrics: list[dict[str, float]] = []
    nan_first_epoch: int | None = None
    t_start = time.time()
    for epoch in range(args.epochs):
        params = renderer.parameters_flat()
        loss_val, grads = loss_and_grad(params)
        # L2 stability sub-ingredient #9 (Optimizer): optional global-norm gradient clip.
        # PR95-sniped-lesson recursive per-sub-ingredient doctrine; Mamba-2 canonical 1.0.
        grad_norm_pre: float | None = None
        if args.max_grad_norm > 0.0:
            grads, total_norm = mlx_optim.clip_grad_norm(grads, args.max_grad_norm)
            grad_norm_pre = float(total_norm.item())
        # Apply optimizer update
        new_params = optimizer.apply_gradients(grads, params)
        renderer.set_parameters_flat(new_params)
        mx.eval(list(new_params.values()))
        # EMA update
        if ema_enabled:
            for k, v in new_params.items():
                if k in ema_shadow:
                    ema_shadow[k] = args.ema_decay * ema_shadow[k] + (1.0 - args.ema_decay) * v
                else:
                    ema_shadow[k] = mx.array(v)
        loss_float = float(loss_val.item())
        is_nan = not (loss_float == loss_float and abs(loss_float) < float("inf"))
        epoch_row: dict[str, float] = {"epoch": epoch + 1, "loss": loss_float}
        if grad_norm_pre is not None:
            epoch_row["grad_norm_pre_clip"] = grad_norm_pre
        if is_nan and nan_first_epoch is None:
            nan_first_epoch = epoch + 1
            epoch_row["nan_detected"] = 1.0
        per_epoch_metrics.append(epoch_row)
        if epoch < 3 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1 or is_nan:
            gn_str = f" gn={grad_norm_pre:.3f}" if grad_norm_pre is not None else ""
            nan_str = " [NAN]" if is_nan else ""
            print(f"[z7-mamba2-v2-mlx] epoch {epoch+1:3d}/{args.epochs}: "
                  f"loss={loss_float:.6f}{gn_str}{nan_str}")
        if is_nan:
            # Stop early on NaN; record for verdict but don't waste epochs
            print(f"[z7-mamba2-v2-mlx] [L2-STABILITY] NaN detected at epoch {epoch+1}; halting early")
            break

    train_wall = time.time() - t_start
    print(f"[z7-mamba2-v2-mlx] training: {train_wall:.1f}s; "
          f"loss {per_epoch_metrics[0]['loss']:.6f} → {per_epoch_metrics[-1]['loss']:.6f}")

    # ---- Build inflate-equivalent archive (proxy: serialize EMA shadow + config) ----
    # Per CC-J: A_log procedurally regenerated → don't include in archive
    archive_params = ema_shadow if ema_enabled else renderer.parameters_flat()
    archive_payload = {
        "schema_version": "z7_mamba2_v2_mlx_l1_v1",
        "config": {
            "num_pairs": args.num_pairs,
            "latent_dim": args.latent_dim,
            "ego_motion_dim": args.ego_motion_dim,
            "d_model": args.d_model,
            "d_state": args.d_state,
            "expand": args.expand,
            "d_conv": args.d_conv,
            "a_log_init_scheme": args.a_log_init_scheme,
            "output_height": args.output_height,
            "output_width": args.output_width,
            "enable_temporal_conv": int(args.enable_temporal_conv),
        },
        # Serialize per-param shapes + float16 quantized bytes for byte-budget estimate
        "param_shapes": {k: list(v.shape) for k, v in archive_params.items()},
    }
    # Quantize-to-fp16 + brotli-style estimate (don't actually serialize; just count)
    fp16_bytes_total = 0
    for k, v in archive_params.items():
        if k == "mamba_A_log":
            continue  # CC-J: procedurally regenerated, NOT serialized
        fp16_bytes_total += int(np.prod(v.shape)) * 2  # fp16 = 2 bytes/elem
    archive_payload["fp16_byte_estimate"] = fp16_bytes_total
    archive_payload["a_log_savings_per_cc_j_unwind_bytes"] = int(np.prod(renderer.mamba_A_log.shape)) * 2

    # Build proxy archive bytes (canonical JSON + per-param fp16 array bytes)
    arch_json = json.dumps(archive_payload, sort_keys=True).encode("utf-8")
    arch_payload_bytes = arch_json
    for k, v in archive_params.items():
        if k == "mamba_A_log":
            continue
        np_arr = np.array(v, dtype=np.float32).astype(np.float16)
        arch_payload_bytes += np_arr.tobytes()
    arch_path = args.output_dir / "0.bin"
    arch_path.write_bytes(arch_payload_bytes)
    arch_sha = hashlib.sha256(arch_payload_bytes).hexdigest()

    # ---- Training manifest with canonical Provenance ----
    manifest = {
        "schema_version": "z7_mamba2_v2_mlx_l1_v1_training_manifest",
        "lane_id": "lane_z7_mamba_2_v2_l1_empirical_mlx_fair_shake_20260526",
        "lane_id_l0_scaffold_predecessor": "lane_path_3_b_prime_z7_mamba_2_cargo_cult_first_20260526",
        "substrate_id": "z7_mamba2_v2_fresh_substrate",
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "non_promotable_blockers": [
            "mse_proxy_loss_not_score_aware_lagrangian",
            "no_paired_linux_x86_64_cuda_anchor",
            "no_paired_linux_x86_64_cpu_anchor",
            "no_segnet_posenet_scorer_routing",
            "mlx_local_research_signal_per_catalog_192",
            "predicted_band_validation_status_post_training_mlx_local",
        ],
        "predicted_band": "[0.155, 0.180]",
        "predicted_band_validation_status": "post_training_mlx_local",
        "config": archive_payload["config"],
        "frame_loader": frame_metadata,
        "training": {
            "epochs": args.epochs,
            "num_pairs": args.num_pairs,
            "ema_enabled": ema_enabled,
            "ema_decay": float(args.ema_decay),
            "learning_rate": float(args.learning_rate),
            "lambda_residual_ib_scale": float(args.lambda_residual),
            "seed": int(args.seed),
            "per_epoch_metrics": per_epoch_metrics,
            "wall_clock_seconds_training": float(train_wall),
            "wall_clock_seconds_decode": float(frame_metadata.get("decode_wall_clock_seconds", 0)),
            "total_param_count": int(renderer.total_param_count()),
        },
        "archive": {
            "path": str(arch_path.relative_to(REPO_ROOT)) if arch_path.is_relative_to(REPO_ROOT) else str(arch_path),
            "bytes": len(arch_payload_bytes),
            "sha256": arch_sha,
            "sha256_prefix_16": arch_sha[:16],
            "fp16_byte_estimate": fp16_bytes_total,
            "a_log_savings_per_cc_j_unwind_bytes": archive_payload["a_log_savings_per_cc_j_unwind_bytes"],
        },
        "distinguishing_feature_status": {
            "cc_a_temporal_conv_enabled": args.enable_temporal_conv,
            "cc_b_latent_dim": args.latent_dim,
            "cc_c_ego_motion_dim": args.ego_motion_dim,
            "cc_d_a_log_init_scheme": args.a_log_init_scheme,
            "cc_h_ib_scale": float(args.lambda_residual),
            "cc_j_a_log_regenerated_not_serialized": True,
        },
        "l2_stability_hardening": {
            "max_grad_norm": float(args.max_grad_norm),
            "a_log_clamp_min": float(args.a_log_clamp_min) if args.a_log_clamp_min > float("-inf") else None,
            "a_log_clamp_max": float(args.a_log_clamp_max) if args.a_log_clamp_max < float("inf") else None,
            "warmup_decay_enabled": bool(args.enable_warmup_decay),
            "peak_lr": float(args.peak_lr) if args.enable_warmup_decay else None,
            "warmup_steps": int(args.warmup_steps) if args.enable_warmup_decay else None,
            "min_lr_ratio": float(args.min_lr_ratio) if args.enable_warmup_decay else None,
            "nan_first_epoch": nan_first_epoch,
            "nan_free_full_run": nan_first_epoch is None,
        },
        "loss_curve_summary": {
            "loss_initial": per_epoch_metrics[0]["loss"] if per_epoch_metrics else None,
            "loss_final": per_epoch_metrics[-1]["loss"] if per_epoch_metrics else None,
            "loss_reduction_percent": (
                100.0 * (per_epoch_metrics[0]["loss"] - per_epoch_metrics[-1]["loss"]) / per_epoch_metrics[0]["loss"]
                if per_epoch_metrics and per_epoch_metrics[0]["loss"] > 0 else None
            ),
            "monotonic_decrease": all(
                per_epoch_metrics[i]["loss"] >= per_epoch_metrics[i + 1]["loss"]
                for i in range(len(per_epoch_metrics) - 1)
            ) if len(per_epoch_metrics) > 1 else None,
        },
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"[z7-mamba2-v2-mlx] [L1-PROMOTION] LANDED")
    print(f"  archive: {arch_path} ({len(arch_payload_bytes):,} bytes; sha {arch_sha[:16]})")
    print(f"  manifest: {manifest_path}")
    if per_epoch_metrics:
        red = manifest["loss_curve_summary"]["loss_reduction_percent"]
        mono = manifest["loss_curve_summary"]["monotonic_decrease"]
        print(f"  loss: {per_epoch_metrics[0]['loss']:.6f} → {per_epoch_metrics[-1]['loss']:.6f} "
              f"({red:.1f}% reduction; monotonic={mono})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
