#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR101 FULL CompressAI ScaleHyperprior reactivation — anchors the
``compressai_balle_hyperprior`` row with the architecture the adversarial
audit (2026-05-07, ``feedback_adversarial_audit_4_falsifications_DEFERRED_not_killed_20260507.md``)
requested.

Why this tool exists
--------------------
The prior tool ``tools/pr101_compressai_balle_hyperprior.py`` shipped a 1.1KB
custom MLP — well below the predicted 5-10KB capacity threshold the adversarial
audit named as the explicit reactivation criterion. Per CLAUDE.md
"KILL is LAST RESORT" rule, a single-config failure with insufficient capacity
cannot kill the technique class.

This tool uses the actual ``compressai.models.ScaleHyperprior`` and
``compressai.models.MeanScaleHyperprior`` classes (with a 1-channel
``Mono*`` subclass — PR101 weights are scalar symbols, not RGB), trains FULL
analysis/synthesis transforms + hyperprior + EntropyBottleneck side-info on
MPS for 100-200 epochs, and measures total bytes via the proper
``compress()``/``decompress()`` end-to-end pipeline.

Substrate adaptation choices (documented per CLAUDE.md substrate-rule)
---------------------------------------------------------------------
- PR101 INT8 quantized symbols across 28 tensors (228,958 total) are
  reshaped to a single ``1×1×448×512`` pseudo-image (418-symbol pad,
  ~0.18% overhead). 448 and 512 are both divisible by 16 (the
  ScaleHyperprior 4× stride-2 downsample requires it).
- Symbols [-127, 127] are scaled to fp32 in roughly [-1, 1] (divide by
  127). The reconstruction is rounded back to integer and clipped.
- ``Mono*Hyperprior`` subclasses replace the first ``g_a`` conv (3→N
  becomes 1→N) and the last ``g_s`` deconv (N→3 becomes N→1). All other
  blocks (GDN, hyperprior h_a/h_s, EntropyBottleneck, GaussianConditional)
  are unmodified CompressAI canonical components.
- The FULL ``compress()`` pipeline emits TWO byte strings:
  * ``y_strings`` — the latent compressed under the GaussianConditional
    using the predicted scales from h_s (the "main" payload).
  * ``z_strings`` — the hyperprior side-info compressed under the
    EntropyBottleneck (CompressAI's standard pipeline).
  Both are charged.
- Decoder weights are serialized to fp16 + brotli (so the model itself is
  charged). For multi-config sweeps the manifest reports ALL variants;
  the row anchored to ``cathedral_autopilot_evidence.jsonl`` is the
  best-bytes variant.

Output evidence is tagged ``[MPS-research-signal]`` per CLAUDE.md MPS rules:
no scorer-load, no score claim, byte/roundtrip-rel_err only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn as nn
from compressai.models import MeanScaleHyperprior, ScaleHyperprior
from compressai.models.utils import conv, deconv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_compressai_balle_hyperprior_full.py"
SCHEMA_VERSION = "pr101_compressai_balle_hyperprior_full.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094  # PR101 wire-format overhead (header, sidecar tables, etc.)
N_TENSORS = len(FIXED_STATE_SCHEMA)
EVIDENCE_GRADE = "[MPS-research-signal]"
EVIDENCE_SEMANTICS = "mps_proxy_full_scale_hyperprior_no_score"
DISPATCH_BLOCKERS = (
    "mps_proxy_signal_not_score_evidence",
    "training_device_not_exact_auth_eval",
    "no_exact_archive_adjudication",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
)


def proxy_evidence_contract() -> dict[str, object]:
    return {
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


# Pseudo-image dimensions for ScaleHyperprior input.
# Chosen to be divisible by 16 (4 stride-2 downsamples) with minimal pad.
# 448*512 = 229,376 ; pad 418 over 228,958 raw symbols (~0.18%).
PSEUDO_H = 448
PSEUDO_W = 512
PSEUDO_PIXELS = PSEUDO_H * PSEUDO_W

# Symbol normalization: [-127, 127] -> [-1, 1] (approximately) using /127.
# Reconstruction inverts: round(clip(x_hat * 127, -127, 127)).
SYMBOL_NORM = float(N_QUANT)  # 127


# ----------------------------------------------------------------------
# Mono (1-channel) subclasses of CompressAI's canonical hyperpriors
# ----------------------------------------------------------------------


class MonoScaleHyperprior(ScaleHyperprior):
    """1-channel ScaleHyperprior. Replaces 3-ch I/O convs only."""

    def __init__(self, N: int, M: int, in_channels: int = 1):
        super().__init__(N=N, M=M)
        # First conv in g_a: 3 -> N becomes in_channels -> N
        self.g_a[0] = conv(in_channels, N)
        # Last deconv in g_s: N -> 3 becomes N -> in_channels
        self.g_s[-1] = deconv(N, in_channels)


class MonoMeanScaleHyperprior(MeanScaleHyperprior):
    """1-channel MeanScaleHyperprior (predicts both mean AND scale, vs Scale only)."""

    def __init__(self, N: int, M: int, in_channels: int = 1):
        super().__init__(N=N, M=M)
        self.g_a[0] = conv(in_channels, N)
        self.g_s[-1] = deconv(N, in_channels)


MODEL_REGISTRY = {
    "scale": MonoScaleHyperprior,
    "mean_scale": MonoMeanScaleHyperprior,
}


# ----------------------------------------------------------------------
# Substrate prep
# ----------------------------------------------------------------------


@dataclass
class Substrate:
    """PR101 substrate after concat + reshape into pseudo-image."""

    raw_symbols: np.ndarray  # int32 [N_total] zero-centered
    n_real: int
    n_padded: int
    image: torch.Tensor  # [1, 1, H, W] fp32 normalized to ~[-1, 1]
    per_tensor_offsets: list[tuple[str, int, int]]  # (name, start, n)
    per_tensor_scales: list[float]


def collect_substrate(state_dict_path: Path) -> Substrate:
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    flat_chunks: list[np.ndarray] = []
    offsets: list[tuple[str, int, int]] = []
    scales: list[float] = []
    cursor = 0
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        chunk = qt.q_i8.astype(np.int32).flatten()
        flat_chunks.append(chunk)
        offsets.append((name, cursor, chunk.size))
        cursor += chunk.size
        scales.append(float(qt.scale))

    raw = np.concatenate(flat_chunks)
    n_real = raw.size
    n_padded = PSEUDO_PIXELS - n_real
    if n_padded < 0:
        raise SystemExit(
            f"PR101 has {n_real:,} symbols but pseudo-image holds only {PSEUDO_PIXELS:,}"
        )
    padded = np.concatenate([raw, np.zeros(n_padded, dtype=np.int32)])
    image_np = padded.astype(np.float32) / SYMBOL_NORM
    image = torch.from_numpy(image_np).view(1, 1, PSEUDO_H, PSEUDO_W)
    return Substrate(
        raw_symbols=raw,
        n_real=n_real,
        n_padded=n_padded,
        image=image,
        per_tensor_offsets=offsets,
        per_tensor_scales=scales,
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train_hyperprior(
    model: nn.Module,
    image: torch.Tensor,
    *,
    device: str,
    epochs: int,
    lr: float = 1e-3,
    aux_lr: float = 1e-2,
    rd_lambda: float = 0.05,
    log_every: int = 10,
) -> dict:
    """Train hyperprior end-to-end on the single PR101 pseudo-image.

    The standard CompressAI loss is  L = lambda * MSE + bpp(y) + bpp(z).
    Here MSE is the symbol roundtrip distortion; bpp drives bytes down.

    Aux loss (entropy_bottleneck.aux_loss()) is trained on a separate optimizer
    per CompressAI's canonical recipe — it tunes the EntropyBottleneck CDF
    parameters.
    """
    dev = torch.device(device)
    model.to(dev).train()
    img = image.to(dev)

    # Split params: aux params (entropy_bottleneck CDF parameters) get their own optimizer
    aux_params = []
    main_params = []
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
        # bits per pixel
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
            with torch.no_grad():
                rec = (x_hat * SYMBOL_NORM).round().clamp(-N_QUANT, N_QUANT)
                ref = (img * SYMBOL_NORM).round().clamp(-N_QUANT, N_QUANT)
                px_err = float((rec - ref).abs().mean().item())
            elapsed = time.time() - t0
            entry = {
                "epoch": ep + 1,
                "loss": float(loss.item()),
                "mse_norm": float(mse.item()),
                "bpp_y": float(bpp_y.item()),
                "bpp_z": float(bpp_z.item()),
                "bpp_total": float(bpp.item()),
                "aux_loss": float(aux_loss.item()),
                "mean_abs_symbol_err_proxy": px_err,
                "elapsed_sec": elapsed,
            }
            history.append(entry)
            print(
                f"  ep {ep+1:4d}/{epochs}  loss={loss.item():.4f}  "
                f"bpp={bpp.item():.4f} (y={bpp_y.item():.4f}, z={bpp_z.item():.4f})  "
                f"mse={mse.item():.5f}  aux={aux_loss.item():.4f}  "
                f"sym_err={px_err:.3f}  t={elapsed:.0f}s"
            )

    model.cpu()
    return {
        "epochs_trained": epochs,
        "elapsed_sec": time.time() - t0,
        "history": history,
        "rd_lambda": rd_lambda,
        "lr": lr,
        "aux_lr": aux_lr,
    }


# ----------------------------------------------------------------------
# Encode / decode + measure
# ----------------------------------------------------------------------


def serialize_model(model: nn.Module) -> bytes:
    """Pack model state_dict to fp16 (excluding entropy_bottleneck buffers).

    CompressAI's EntropyBottleneck has internal buffers (CDF tables) that are
    rebuilt from the trainable ``quantiles`` parameter via ``update()`` at
    decode time, so we only persist trainable params + the EntropyBottleneck
    quantiles. This matches CompressAI's deployment story.
    """
    sd = model.state_dict()
    buf = bytearray()
    buf += b"BLF1"  # magic: BalLe Full v1
    keys = sorted(sd.keys())
    # Filter: keep all trainable params (which include 'quantiles')
    keep = []
    for k in keys:
        v = sd[k]
        if not isinstance(v, torch.Tensor):
            continue
        # Skip int buffers (CDF tables auto-regenerated by update())
        if v.dtype in (torch.int32, torch.int64, torch.uint8, torch.int16, torch.int8):
            continue
        # Skip _quantized_cdf / _cdf_length / _offset (regenerated)
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


def encode_decode_measure(
    model: nn.Module,
    substrate: Substrate,
) -> dict:
    """Run model.compress() / model.decompress() and measure bytes + rel_err."""
    model.eval()
    # Required: rebuild CDF tables from trained quantiles
    model.update(force=True)

    img = substrate.image
    with torch.no_grad():
        compressed = model.compress(img)
        decompressed = model.decompress(compressed["strings"], compressed["shape"])
        x_hat = decompressed["x_hat"]

    # Measure bytes
    strings = compressed["strings"]
    # CompressAI returns strings as List[List[bytes]] per stream
    # ScaleHyperprior: strings[0] is y_strings, strings[1] is z_strings
    y_strings = strings[0]
    z_strings = strings[1]
    y_bytes = sum(len(s) for s in y_strings)
    z_bytes = sum(len(s) for s in z_strings)
    payload_bytes = y_bytes + z_bytes

    # Brotli-compress the raw payloads (small additional savings sometimes)
    y_concat = b"".join(struct.pack("<I", len(s)) + s for s in y_strings)
    z_concat = b"".join(struct.pack("<I", len(s)) + s for s in z_strings)
    payload_concat = struct.pack("<I", len(y_concat)) + y_concat + struct.pack("<I", len(z_concat)) + z_concat
    payload_brotli = brotli.compress(payload_concat, quality=11, lgwin=22, lgblock=24)

    # Roundtrip: convert reconstructed pseudo-image back to symbols
    rec_norm = x_hat.cpu().squeeze().flatten().numpy()
    rec_symbols = np.round(rec_norm * SYMBOL_NORM).clip(-N_QUANT, N_QUANT).astype(np.int32)
    rec_real = rec_symbols[: substrate.n_real]
    abs_err = np.abs(rec_real - substrate.raw_symbols).astype(np.float64)
    abs_orig = np.abs(substrate.raw_symbols).astype(np.float64)
    # Relative error: mean(|err|) / mean(|orig|), clamped at 1.0 on zero-magnitude
    denom = abs_orig.sum()
    rel_err = float(abs_err.sum()) if denom < 1e-9 else float(abs_err.sum() / denom)  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for FALSIFIED CompressAI Ballé hyperprior full sweep; not allocator-fed
    mean_abs_err = float(abs_err.mean())
    max_abs_err = int(abs_err.max())
    nonzero_diff_frac = float((abs_err > 0).mean())

    # Serialize model
    model_blob = serialize_model(model)
    model_brotli = brotli.compress(model_blob, quality=11, lgwin=22, lgblock=24)

    decoder_blob_bytes = len(model_brotli) + len(payload_brotli)
    archive_bytes = decoder_blob_bytes + ARCHIVE_OVERHEAD_BYTES

    return {
        "y_bytes_raw": y_bytes,
        "z_bytes_raw": z_bytes,
        "payload_bytes_raw": payload_bytes,
        "payload_brotli_bytes": len(payload_brotli),
        "model_blob_raw_bytes": len(model_blob),
        "model_blob_brotli_bytes": len(model_brotli),
        "decoder_blob_bytes": decoder_blob_bytes,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "rel_err": rel_err,
        "mean_abs_symbol_err": mean_abs_err,
        "max_abs_symbol_err": max_abs_err,
        "nonzero_diff_fraction": nonzero_diff_frac,
        "y_string_count": len(y_strings),
        "z_string_count": len(z_strings),
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_one_config(
    model_class_name: str,
    N: int,
    M: int,
    substrate: Substrate,
    *,
    device: str,
    epochs: int,
    rd_lambda: float,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = MODEL_REGISTRY[model_class_name]
    model = cls(N=N, M=M, in_channels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[balle-full] config: {model_class_name} N={N} M={M} "
        f"params={n_params:,} (~{n_params*2/1024:.1f} KB fp16)  "
        f"epochs={epochs} rd_lambda={rd_lambda} device={device}"
    )

    train_log = train_hyperprior(
        model, substrate.image, device=device, epochs=epochs, rd_lambda=rd_lambda, lr=lr
    )
    measurements = encode_decode_measure(model, substrate)

    return {
        "model_class": model_class_name,
        "N": N,
        "M": M,
        **proxy_evidence_contract(),
        "n_params": n_params,
        "model_kb_fp16": n_params * 2 / 1024,
        **measurements,
        "training": train_log,
    }


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
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--rd-lambda", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--configs",
        type=str,
        default="scale:8:4,scale:12:6,scale:16:8,mean_scale:8:4,mean_scale:12:6",
        help="Comma-separated model_class:N:M tuples to sweep.",
    )
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
        args.output_dir = REPO_ROOT / f"reports/raw/pr101_compressai_balle_full_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[balle-full] loading PR101 substrate from {args.state_dict}")
    substrate = collect_substrate(args.state_dict)
    print(
        f"[balle-full] {substrate.n_real:,} real symbols + {substrate.n_padded} pad "
        f"-> {PSEUDO_H}x{PSEUDO_W} pseudo-image"
    )

    configs: list[tuple[str, int, int]] = []
    for tok in args.configs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad config token: {tok!r} (want model:N:M)")
        cls_name, N, M = parts[0], int(parts[1]), int(parts[2])
        if cls_name not in MODEL_REGISTRY:
            raise SystemExit(f"unknown model class {cls_name!r}; choices: {list(MODEL_REGISTRY)}")
        configs.append((cls_name, N, M))

    results: list[dict] = []
    for cls_name, N, M in configs:
        try:
            r = run_one_config(
                cls_name, N, M, substrate,
                device=args.device, epochs=args.epochs,
                rd_lambda=args.rd_lambda, lr=args.lr, seed=args.seed,
            )
        except Exception as exc:  # surface and continue
            r = {
                "model_class": cls_name, "N": N, "M": M,
                **proxy_evidence_contract(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  [balle-full] CONFIG FAILED: {r['error']}")
        results.append(r)
        # Persist after each config in case later ones crash
        partial = {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            **proxy_evidence_contract(),
            "device": args.device,
            "epochs": args.epochs,
            "rd_lambda": args.rd_lambda,
            "n_real_symbols": substrate.n_real,
            "n_padded": substrate.n_padded,
            "pseudo_image_h": PSEUDO_H,
            "pseudo_image_w": PSEUDO_W,
            "input_state_dict": str(args.state_dict),
            "configs_swept": [f"{c[0]}:N{c[1]}:M{c[2]}" for c in configs],
            "results": results,
            "completed_configs": len([r for r in results if "error" not in r]),
        }
        manifest_path = args.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    # Pick the best config (smallest archive_bytes among non-erroring configs)
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("\n[balle-full] ALL CONFIGS FAILED — no evidence row written")
        return 1
    best = min(ok, key=lambda r: r["archive_bytes"])

    print()
    print("=" * 70)
    print("[balle-full] SUMMARY")
    print("=" * 70)
    print(f"{'config':<28} {'archive_B':>12} {'model_B':>10} {'payload_B':>10} {'rel_err':>10}")
    for r in ok:
        cfg = f"{r['model_class']}:N{r['N']}:M{r['M']}"
        print(
            f"{cfg:<28} {r['archive_bytes']:>12,} {r['model_blob_brotli_bytes']:>10,} "
            f"{r['payload_brotli_bytes']:>10,} {r['rel_err']:>10.4f}"
        )
    print()
    print(f"BEST: {best['model_class']}:N{best['N']}:M{best['M']}")
    print(f"  archive_bytes    : {best['archive_bytes']:,} B  [MPS-research-signal]")
    print(f"  vs PR101 brotli  : {best['archive_bytes'] - 178144:+,} B (target 178,144 B)")
    print(f"  rel_err          : {best['rel_err']:.4f}")
    print(f"  payload (brotli) : {best['payload_brotli_bytes']:,} B (y+z streams)")
    print(f"  decoder model    : {best['model_blob_brotli_bytes']:,} B fp16+brotli")

    # Final manifest
    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        **proxy_evidence_contract(),
        "device": args.device,
        "epochs": args.epochs,
        "rd_lambda": args.rd_lambda,
        "lr": args.lr,
        "seed": args.seed,
        "n_real_symbols": substrate.n_real,
        "n_padded": substrate.n_padded,
        "pseudo_image_h": PSEUDO_H,
        "pseudo_image_w": PSEUDO_W,
        "symbol_norm": SYMBOL_NORM,
        "input_state_dict": str(args.state_dict),
        "configs_swept": [f"{c[0]}:N{c[1]}:M{c[2]}" for c in configs],
        "best_config": f"{best['model_class']}:N{best['N']}:M{best['M']}",
        "best_archive_bytes": best["archive_bytes"],
        "best_rel_err": best["rel_err"],
        "results": results,
        "substrate_adaptation_choice": (
            "PR101 INT8 symbols (228,958) reshaped to 1x1x448x512 pseudo-image "
            "(418-symbol pad ~0.18%); Mono*Hyperprior subclass replaces 3-ch I/O "
            "with 1-ch; symbols [-127,127] / 127 -> fp32 [-1,1]; reconstruction "
            "rounded back to int. Total bytes = brotli(model_fp16) + "
            "brotli(y_strings + z_strings) + 16,094 archive overhead."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Append evidence row
    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    pr101_brotli_baseline = 178144
    delta = best["archive_bytes"] - pr101_brotli_baseline
    if best["rel_err"] > 0.02:
        verdict = (
            f"DEFERRED-pending-research (rel_err {best['rel_err']:.4f} > 0.02 threshold; "
            f"FULL ScaleHyperprior architecture exhausted at {len(configs)} N/M configs "
            f"on {args.epochs} epochs)"
        )
    elif delta < -1000:
        verdict = (
            f"BEAT-PR101-brotli ({delta:+,} B) at rel_err {best['rel_err']:.4f}; "
            f"REOPEN-FOR-DISPATCH-CONSIDERATION"
        )
    else:
        verdict = (
            f"DEFERRED-pending-research (FULL arch lands {delta:+,} vs PR101 brotli; "
            f"reactivation needs cross-tensor or position-aware variant per audit)"
        )
    evidence_row = {
        "technique": "compressai_balle_hyperprior",
        "empirical_archive_bytes": best["archive_bytes"],
        "empirical_rel_err": best["rel_err"],
        **proxy_evidence_contract(),
        "source": (
            f"[MPS-research-signal] {manifest_path} (FULL ScaleHyperprior "
            f"reactivation; best={best['model_class']}:N{best['N']}:M{best['M']}, "
            f"{args.epochs} epochs)"
        ),
        "timestamp": timestamp,
        "contest_dispatch_verdict": verdict,
        "supersedes_prior_DEFERRED_audit": True,
    }
    args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evidence_row) + "\n")

    print()
    print(f"manifest        : {manifest_path}")
    print(f"evidence row    : {args.evidence_jsonl} (appended)")
    print(f"verdict         : {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
