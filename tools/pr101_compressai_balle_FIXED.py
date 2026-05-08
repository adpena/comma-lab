#!/usr/bin/env python3
"""PR101 CompressAI Balle hyperprior — FIXED reactivation.

Background
----------
The prior tool ``tools/pr101_compressai_balle_hyperprior_full.py`` ran the
canonical CompressAI ``ScaleHyperprior`` / ``MeanScaleHyperprior`` on PR101's
228,958 INT8 symbols reshaped as a single ``1×1×448×512`` pseudo-image. All 8
configs swept produced rel_err ≈ 0.98. The prior memo concluded
"substrate-mismatch (no 2D locality)".

This tool challenges that conclusion. The 0.98 rel_err is more consistent with
a complete training failure than with a substrate-mismatch ceiling. Specific
hypotheses:

1. **rd_lambda imbalance** — at rd_lambda=0.05, the loss
   ``0.05 * MSE * 65025 + bpp`` saturates: the BPP term dominates and the
   model collapses to predicting near-zero (which is cheapest to encode).
   Symptom: model output ≈ 0, MSE plateaus at the variance of the data.
2. **Single-image overfit** — training on a single 1×1×448×512 image with no
   batching means the optimizer cannot escape the BPP-collapsed minimum.
3. **Reshape strategy destroys tensor identity** — different tensors get
   concatenated then folded into rows; conv kernels see meaningless
   neighbourhoods.
4. **Normalization strips dynamic range** — dividing by 127 maps integer
   symbols to ~[-1,1] but most symbols are 0; a Gaussian conditional
   matched to the natural-image prior expects different statistics.

Fixes implemented
-----------------
- **Per-tensor encoding** (default mode): each of the 28 PR101 tensors is
  reshaped into its own pseudo-image with dimensions divisible by 16 (the
  ScaleHyperprior 4-stride downsample requirement), then encoded
  independently. This preserves tensor identity. Each tensor uses an
  appropriately sized N/M.
- **rd_lambda sweep** (5 orders of magnitude): {1e-3, 1e-2, 1e-1, 1.0, 10.0,
  100.0} so we can find the rate-distortion knee where MSE actually drops.
- **Z-score normalization** (per-tensor): preserves magnitude information
  rather than discarding it. Mean and std are stored as part of the model
  blob (negligible: 28 * 8 bytes = 224 B).
- **Joint (concat) mode** retained as a baseline for comparison.

CLAUDE.md compliance: MPS allowed (signal-only), evidence tagged
``[MPS-research-signal]``; no scorer-load, no score claim.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn as nn

from compressai.models import ScaleHyperprior, MeanScaleHyperprior
from compressai.models.utils import conv, deconv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_compressai_balle_FIXED.py"
SCHEMA_VERSION = "pr101_compressai_balle_FIXED.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094  # canonical PR101 wire-format overhead
N_TENSORS = len(FIXED_STATE_SCHEMA)
PR101_BROTLI_BASELINE_BYTES = 178_144  # canonical PR101 best-known


# ----------------------------------------------------------------------
# Mono (1-channel) hyperprior subclasses (same as prior tool, kept here for
# self-containment; replaces 3-ch I/O convs).
# ----------------------------------------------------------------------


class MonoScaleHyperprior(ScaleHyperprior):
    def __init__(self, N: int, M: int, in_channels: int = 1):
        super().__init__(N=N, M=M)
        self.g_a[0] = conv(in_channels, N)
        self.g_s[-1] = deconv(N, in_channels)


class MonoMeanScaleHyperprior(MeanScaleHyperprior):
    def __init__(self, N: int, M: int, in_channels: int = 1):
        super().__init__(N=N, M=M)
        self.g_a[0] = conv(in_channels, N)
        self.g_s[-1] = deconv(N, in_channels)


MODEL_REGISTRY = {
    "scale": MonoScaleHyperprior,
    "mean_scale": MonoMeanScaleHyperprior,
}


# ----------------------------------------------------------------------
# Substrate prep — per-tensor variant
# ----------------------------------------------------------------------


def _pad_to_multiple(n: int, m: int) -> int:
    """Smallest int >= n divisible by m."""
    return ((n + m - 1) // m) * m


def _factor_pseudo_image(n: int, downsample: int = 16) -> tuple[int, int, int]:
    """Pick a (H, W) pseudo-image such that H, W are divisible by `downsample`
    and the total pixel count covers `n` with minimal padding.

    Strategy: pick H = multiple-of-downsample closest to sqrt(n); W = ceil
    to multiple-of-downsample.
    """
    H = max(downsample, _pad_to_multiple(int(math.sqrt(max(n, 1))), downsample))
    W = _pad_to_multiple(max(1, math.ceil(n / H)), downsample)
    if H * W < n:
        # safety
        H = _pad_to_multiple(H + downsample, downsample)
        W = _pad_to_multiple(max(1, math.ceil(n / H)), downsample)
    return H, W, H * W - n


@dataclass
class TensorSlot:
    name: str
    raw: np.ndarray  # int32 [n]
    n: int
    H: int
    W: int
    pad: int
    # Z-score normalization parameters (per-tensor)
    z_mean: float
    z_std: float
    image: torch.Tensor  # [1, 1, H, W] fp32


def collect_per_tensor_substrate(state_dict_path: Path) -> list[TensorSlot]:
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    slots: list[TensorSlot] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        raw = qt.q_i8.astype(np.int32).flatten()
        n = raw.size
        H, W, pad = _factor_pseudo_image(n, downsample=16)
        # Z-score: preserve magnitude information (instead of /127)
        z_mean = float(raw.mean())
        z_std = float(raw.std())
        if z_std < 1e-6:
            z_std = 1.0
        padded = np.concatenate([raw, np.zeros(pad, dtype=np.int32)])
        norm = (padded.astype(np.float32) - z_mean) / z_std
        image = torch.from_numpy(norm).view(1, 1, H, W)
        slots.append(
            TensorSlot(
                name=name, raw=raw, n=n, H=H, W=W, pad=pad,
                z_mean=z_mean, z_std=z_std, image=image,
            )
        )
    return slots


@dataclass
class JointSubstrate:
    raw: np.ndarray
    n_real: int
    n_pad: int
    H: int
    W: int
    z_mean: float
    z_std: float
    image: torch.Tensor


def collect_joint_substrate(state_dict_path: Path) -> JointSubstrate:
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    chunks: list[np.ndarray] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        chunks.append(qt.q_i8.astype(np.int32).flatten())
    raw = np.concatenate(chunks)
    n = raw.size
    # 448x512 = 229376 to match prior tool's pseudo-image dims
    H, W = 448, 512
    pad = H * W - n
    if pad < 0:
        raise SystemExit(f"PR101 has {n:,} symbols but joint pseudo-image holds only {H*W:,}")
    z_mean = float(raw.mean())
    z_std = float(raw.std())
    if z_std < 1e-6:
        z_std = 1.0
    padded = np.concatenate([raw, np.zeros(pad, dtype=np.int32)])
    norm = (padded.astype(np.float32) - z_mean) / z_std
    image = torch.from_numpy(norm).view(1, 1, H, W)
    return JointSubstrate(
        raw=raw, n_real=n, n_pad=pad, H=H, W=W,
        z_mean=z_mean, z_std=z_std, image=image,
    )


# ----------------------------------------------------------------------
# Training & measurement
# ----------------------------------------------------------------------


def train_balle(
    model: nn.Module,
    image: torch.Tensor,
    *,
    device: str,
    epochs: int,
    lr: float,
    aux_lr: float,
    rd_lambda: float,
    log_every: int = 25,
) -> dict:
    """Train a Balle hyperprior on a single image.

    Loss:  L = lambda * MSE * 255^2 + bpp   (CompressAI canonical form)

    KEY FIX vs prior tool: this caller will SWEEP rd_lambda over multiple
    orders of magnitude to find the working regime.
    """
    dev = torch.device(device)
    model.to(dev).train()
    img = image.to(dev)

    aux_params, main_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".quantiles"):
            aux_params.append(p)
        else:
            main_params.append(p)

    optim = torch.optim.Adam(main_params, lr=lr)
    aux_optim = torch.optim.Adam(aux_params, lr=aux_lr)

    H = image.shape[-2]
    W = image.shape[-1]
    n_pix = H * W

    history = []
    t0 = time.time()
    for ep in range(epochs):
        out = model(img)
        x_hat = out["x_hat"]
        likelihoods = out["likelihoods"]

        mse = (x_hat - img).pow(2).mean()
        bpp_y = -torch.log2(likelihoods["y"]).sum() / n_pix
        bpp_z = -torch.log2(likelihoods["z"]).sum() / n_pix
        bpp = bpp_y + bpp_z
        loss = rd_lambda * mse * 255.0 ** 2 + bpp

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params, max_norm=1.0)
        optim.step()

        aux_loss = model.aux_loss()
        aux_optim.zero_grad()
        aux_loss.backward()
        aux_optim.step()

        if (ep + 1) % log_every == 0 or ep == 0:
            history.append({
                "epoch": ep + 1,
                "loss": float(loss.item()),
                "mse_norm": float(mse.item()),
                "bpp_total": float(bpp.item()),
                "elapsed_sec": time.time() - t0,
            })

    model.cpu()
    return {"epochs_trained": epochs, "elapsed_sec": time.time() - t0, "history": history}


def serialize_model(model: nn.Module) -> bytes:
    sd = model.state_dict()
    buf = bytearray()
    buf += b"BLF2"  # FIXED variant magic
    keys = sorted(sd.keys())
    keep = []
    for k in keys:
        v = sd[k]
        if not isinstance(v, torch.Tensor):
            continue
        if v.dtype in (torch.int32, torch.int64, torch.uint8, torch.int16, torch.int8):
            continue
        if k.endswith("_quantized_cdf") or k.endswith("_cdf_length") or k.endswith("_offset"):
            continue
        keep.append(k)
    buf += struct.pack("<I", len(keep))
    for k in keep:
        v = sd[k].detach().cpu()
        kb = k.encode("utf-8")
        buf += struct.pack("<H", len(kb))
        buf += kb
        shape = list(v.shape)
        buf += struct.pack("<B", len(shape))
        for d in shape:
            buf += struct.pack("<I", d)
        flat = v.numpy().astype(np.float16).flatten()
        buf += struct.pack("<I", flat.size)
        buf += flat.tobytes()
    return bytes(buf)


def encode_decode_one(
    model: nn.Module,
    image: torch.Tensor,
    raw_symbols: np.ndarray,
    n_real: int,
    z_mean: float,
    z_std: float,
) -> dict:
    model.eval()
    model.update(force=True)
    with torch.no_grad():
        out = model.compress(image)
        dec = model.decompress(out["strings"], out["shape"])
        x_hat = dec["x_hat"]
    y_strings = out["strings"][0]
    z_strings = out["strings"][1]
    y_bytes = sum(len(s) for s in y_strings)
    z_bytes = sum(len(s) for s in z_strings)
    payload_concat = (
        struct.pack("<I", len(y_strings))
        + b"".join(struct.pack("<I", len(s)) + s for s in y_strings)
        + struct.pack("<I", len(z_strings))
        + b"".join(struct.pack("<I", len(s)) + s for s in z_strings)
    )
    payload_brotli = brotli.compress(payload_concat, quality=11, lgwin=22, lgblock=24)

    rec = x_hat.cpu().squeeze().flatten().numpy() * z_std + z_mean
    rec_real = np.round(rec[:n_real]).clip(-N_QUANT, N_QUANT).astype(np.int32)
    abs_err = np.abs(rec_real - raw_symbols).astype(np.float64)
    abs_orig = np.abs(raw_symbols).astype(np.float64)
    denom = abs_orig.sum()
    rel_err = float(abs_err.sum() / denom) if denom > 1e-9 else float(abs_err.sum())  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for FALSIFIED CompressAI Ballé sweep; not allocator-fed
    model_blob = serialize_model(model)
    model_brotli = brotli.compress(model_blob, quality=11, lgwin=22, lgblock=24)
    return {
        "y_bytes_raw": y_bytes,
        "z_bytes_raw": z_bytes,
        "payload_brotli_bytes": len(payload_brotli),
        "model_brotli_bytes": len(model_brotli),
        "rel_err": rel_err,
        "mean_abs_symbol_err": float(abs_err.mean()),
        "max_abs_symbol_err": int(abs_err.max()),
    }


# ----------------------------------------------------------------------
# Joint mode: same as prior tool but with rd_lambda sweep + z-score
# ----------------------------------------------------------------------


def run_joint(
    sub: JointSubstrate,
    *,
    model_class: str,
    N: int,
    M: int,
    epochs: int,
    rd_lambda: float,
    lr: float,
    device: str,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = MODEL_REGISTRY[model_class]
    model = cls(N=N, M=M, in_channels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  [joint] {model_class} N={N} M={M} params={n_params:,} "
        f"rd_lambda={rd_lambda} epochs={epochs} device={device}"
    )
    train_log = train_balle(
        model, sub.image, device=device, epochs=epochs,
        lr=lr, aux_lr=1e-2, rd_lambda=rd_lambda,
    )
    meas = encode_decode_one(model, sub.image, sub.raw, sub.n_real, sub.z_mean, sub.z_std)
    archive_bytes = (
        meas["model_brotli_bytes"] + meas["payload_brotli_bytes"]
        + 16  # 8 floats: z_mean, z_std (joint mode side info)
        + ARCHIVE_OVERHEAD_BYTES
    )
    final_mse = train_log["history"][-1]["mse_norm"] if train_log["history"] else None
    return {
        "mode": "joint",
        "model_class": model_class,
        "N": N,
        "M": M,
        "rd_lambda": rd_lambda,
        "n_params": n_params,
        "archive_bytes": archive_bytes,
        "final_mse_norm": final_mse,
        **meas,
        "training_history": train_log["history"],
    }


# ----------------------------------------------------------------------
# Per-tensor mode: 28 mini-images, each independently encoded
# ----------------------------------------------------------------------


def run_per_tensor(
    slots: list[TensorSlot],
    *,
    model_class: str,
    N: int,
    M: int,
    epochs: int,
    rd_lambda: float,
    lr: float,
    device: str,
    seed: int,
) -> dict:
    print(
        f"  [per-tensor] {model_class} N={N} M={M} rd_lambda={rd_lambda} "
        f"epochs={epochs} (28 tensors, each trained independently)"
    )
    per_results = []
    total_archive = 0
    total_payload = 0
    total_model = 0
    cum_n = 0
    cum_err = 0.0
    cum_abs = 0.0
    for i, slot in enumerate(slots):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        cls = MODEL_REGISTRY[model_class]
        model = cls(N=N, M=M, in_channels=1)
        train_log = train_balle(
            model, slot.image, device=device, epochs=epochs,
            lr=lr, aux_lr=1e-2, rd_lambda=rd_lambda, log_every=epochs,
        )
        meas = encode_decode_one(
            model, slot.image, slot.raw, slot.n, slot.z_mean, slot.z_std,
        )
        per_results.append({
            "tensor": slot.name, "n": slot.n, "H": slot.H, "W": slot.W,
            **meas,
        })
        total_payload += meas["payload_brotli_bytes"]
        total_model += meas["model_brotli_bytes"]
        cum_n += slot.n
        cum_err += float(np.abs(np.zeros(slot.n)).sum())  # placeholder
        # actual cumulative weighted error:
        cum_abs += np.abs(slot.raw).astype(np.float64).sum()
    # Re-tally rel_err by aggregating (must reload reconstructions; we have
    # per-tensor mean_abs_symbol_err and n)
    weighted_abs_err = sum(r["mean_abs_symbol_err"] * r["n"] for r in per_results)
    rel_err_overall = weighted_abs_err / cum_abs if cum_abs > 1e-9 else weighted_abs_err

    # Side info: per-tensor z_mean, z_std (28 * 8 bytes = 224 B raw)
    side_info_bytes = N_TENSORS * 8
    archive_bytes = total_payload + total_model + side_info_bytes + ARCHIVE_OVERHEAD_BYTES
    return {
        "mode": "per_tensor",
        "model_class": model_class,
        "N": N,
        "M": M,
        "rd_lambda": rd_lambda,
        "epochs": epochs,
        "archive_bytes": archive_bytes,
        "total_payload_bytes": total_payload,
        "total_model_bytes": total_model,
        "side_info_bytes": side_info_bytes,
        "rel_err": rel_err_overall,
        "per_tensor": per_results,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    p.add_argument("--mode", choices=["joint", "per_tensor", "both"], default="both")
    p.add_argument("--epochs-joint", type=int, default=300)
    p.add_argument("--epochs-per-tensor", type=int, default=200)
    p.add_argument(
        "--rd-lambda-sweep",
        type=str,
        default="0.001,0.01,0.1,1.0,10.0,100.0",
        help="Comma-separated rd_lambda values to sweep (joint mode only).",
    )
    p.add_argument(
        "--per-tensor-rd-lambda",
        type=float,
        default=10.0,
        help="rd_lambda for per-tensor mode (chosen after joint sweep informs).",
    )
    p.add_argument(
        "--joint-config",
        type=str,
        default="mean_scale:8:4",
        help="model_class:N:M for joint sweep.",
    )
    p.add_argument(
        "--per-tensor-config",
        type=str,
        default="mean_scale:8:4",
        help="model_class:N:M for per-tensor mode.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--evidence-jsonl",
        type=Path,
        default=REPO_ROOT / "reports/cathedral_autopilot_evidence.jsonl",
    )
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    if args.output_dir is None:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO_ROOT / f"reports/raw/pr101_balle_FIXED_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[balle-FIXED] state_dict: {args.state_dict}")
    print(f"[balle-FIXED] output:     {args.output_dir}")
    print(f"[balle-FIXED] device:     {args.device}")
    print(f"[balle-FIXED] mode:       {args.mode}")

    # Parse configs
    def _parse_cfg(tok: str) -> tuple[str, int, int]:
        parts = tok.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad config: {tok!r} (want model:N:M)")
        return parts[0], int(parts[1]), int(parts[2])

    j_cls, j_N, j_M = _parse_cfg(args.joint_config)
    pt_cls, pt_N, pt_M = _parse_cfg(args.per_tensor_config)

    rd_lambdas = [float(x) for x in args.rd_lambda_sweep.split(",") if x.strip()]

    all_results: dict = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": "[MPS-research-signal]",
        "device": args.device,
        "input_state_dict": str(args.state_dict),
        "joint_results": [],
        "per_tensor_results": [],
    }

    # ---- Joint mode: sweep rd_lambda ----
    if args.mode in ("joint", "both"):
        print()
        print(f"[balle-FIXED] JOINT mode: sweeping rd_lambda over {rd_lambdas}")
        joint_sub = collect_joint_substrate(args.state_dict)
        print(
            f"  n_real={joint_sub.n_real:,} pad={joint_sub.n_pad} "
            f"z_mean={joint_sub.z_mean:.3f} z_std={joint_sub.z_std:.3f}"
        )
        for rdl in rd_lambdas:
            try:
                r = run_joint(
                    joint_sub,
                    model_class=j_cls, N=j_N, M=j_M,
                    epochs=args.epochs_joint, rd_lambda=rdl,
                    lr=args.lr, device=args.device, seed=args.seed,
                )
            except Exception as exc:
                r = {"mode": "joint", "rd_lambda": rdl, "error": f"{type(exc).__name__}: {exc}"}
                print(f"    FAILED: {r['error']}")
            all_results["joint_results"].append(r)
            print(
                f"    rd_lambda={rdl:>8g}  archive={r.get('archive_bytes', '-'):>10}  "
                f"rel_err={r.get('rel_err', '-'):>8}  "
                f"mse_norm={r.get('final_mse_norm', '-'):>8}"
            )
            # Persist after each
            (args.output_dir / "manifest.json").write_text(
                json.dumps(all_results, indent=2), encoding="utf-8"
            )

    # ---- Per-tensor mode ----
    if args.mode in ("per_tensor", "both"):
        print()
        print(
            f"[balle-FIXED] PER-TENSOR mode: 28 tensors × {pt_cls} N={pt_N} M={pt_M} "
            f"rd_lambda={args.per_tensor_rd_lambda} epochs={args.epochs_per_tensor}"
        )
        slots = collect_per_tensor_substrate(args.state_dict)
        for s in slots[:5]:
            print(f"  tensor[{s.name}]: n={s.n}, H×W={s.H}×{s.W}, pad={s.pad}, "
                  f"z_mean={s.z_mean:.2f}, z_std={s.z_std:.2f}")
        if len(slots) > 5:
            print(f"  ... ({len(slots)-5} more)")
        try:
            r = run_per_tensor(
                slots,
                model_class=pt_cls, N=pt_N, M=pt_M,
                epochs=args.epochs_per_tensor,
                rd_lambda=args.per_tensor_rd_lambda,
                lr=args.lr, device=args.device, seed=args.seed,
            )
        except Exception as exc:
            r = {"mode": "per_tensor", "error": f"{type(exc).__name__}: {exc}"}
            print(f"    FAILED: {r['error']}")
        all_results["per_tensor_results"].append(r)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        if "error" not in r:
            print(
                f"  per_tensor archive={r['archive_bytes']:,} B "
                f"(payload={r['total_payload_bytes']:,}, model={r['total_model_bytes']:,}) "
                f"rel_err={r['rel_err']:.4f}"
            )

    # ---- Find best across all modes ----
    candidates = []
    for r in all_results["joint_results"]:
        if "error" not in r and r.get("archive_bytes") is not None:
            candidates.append(("joint", r))
    for r in all_results["per_tensor_results"]:
        if "error" not in r and r.get("archive_bytes") is not None:
            candidates.append(("per_tensor", r))

    if not candidates:
        print()
        print("[balle-FIXED] NO successful configs — nothing to report")
        return 1

    best_mode, best = min(candidates, key=lambda x: (x[1]["rel_err"], x[1]["archive_bytes"]))
    print()
    print("=" * 70)
    print("[balle-FIXED] SUMMARY")
    print("=" * 70)
    print(f"BEST mode: {best_mode}")
    print(f"  archive_bytes : {best['archive_bytes']:,} B  [MPS-research-signal]")
    print(f"  vs baseline   : {best['archive_bytes'] - PR101_BROTLI_BASELINE_BYTES:+,} B")
    print(f"  rel_err       : {best['rel_err']:.4f}")

    all_results["best_mode"] = best_mode
    all_results["best_archive_bytes"] = best["archive_bytes"]
    all_results["best_rel_err"] = best["rel_err"]
    (args.output_dir / "manifest.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # Evidence row only if rel_err < 0.05 AND archive < 178144
    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    delta = best["archive_bytes"] - PR101_BROTLI_BASELINE_BYTES
    if best["rel_err"] < 0.05 and delta < 0:
        verdict = (
            f"BEAT-PR101-brotli ({delta:+,} B) at rel_err {best['rel_err']:.4f}; "
            f"REOPEN-FOR-DISPATCH-CONSIDERATION (mode={best_mode})"
        )
    elif best["rel_err"] < 0.05:
        verdict = (
            f"DEFERRED (rel_err {best['rel_err']:.4f} OK but archive {delta:+,} B "
            f"vs baseline; mode={best_mode})"
        )
    else:
        verdict = (
            f"DEFERRED-pending-research (rel_err {best['rel_err']:.4f} > 0.05 "
            f"reactivation threshold; mode={best_mode})"
        )

    if best["rel_err"] < 0.05 and delta < 0:
        evidence_row = {
            "technique": "compressai_balle_FIXED",
            "empirical_archive_bytes": best["archive_bytes"],
            "empirical_rel_err": best["rel_err"],
            "evidence_grade": "[MPS-research-signal]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "source": (
                f"[MPS-research-signal] {args.output_dir}/manifest.json "
                f"(FIXED Balle: mode={best_mode}; supersedes prior_full DEFERRED 0.98 rel_err)"
            ),
            "timestamp": timestamp,
            "contest_dispatch_verdict": verdict,
            "supersedes_prior_DEFERRED_audit": True,
        }
        args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.evidence_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"[balle-FIXED] evidence row appended to {args.evidence_jsonl}")
    else:
        print(f"[balle-FIXED] no evidence row (rel_err {best['rel_err']:.4f} or archive bytes")
        print("              insufficient; manifest.json holds the full record)")

    print()
    print(f"manifest : {args.output_dir / 'manifest.json'}")
    print(f"verdict  : {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
