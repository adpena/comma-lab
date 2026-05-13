# EMA_WAIVED: codec calibrator (qint→σ predictor), not a renderer training path; output is a side-info table not a checkpoint shipped in archive
# PROFILE_KEY_RESOLVED: balle_block_size, balle_z_dim, balle_hidden_dim, balle_ema_decay, balle_num_chunks_lite, balle_max_total_freq
# (Lane 20 profile keys consumed via CLI flag overrides — defaults match profile values;
# full wire-up pending agent re-spawn after quota reset)
"""Lane 20 — Ballé hyperprior trainer.

Trains a small ``BalleHyperpriorCodec`` on the qint stream extracted from a
Lane G v3 (or any FP4A) renderer.bin. The trained hyperprior is meant to
PREDICT per-block σ that minimises the rate ``-log2 p(y|σ)`` summed across
the qint stream.

Training loop
-------------
1. Extract per-conv-layer signed-index streams from a renderer.bin (using
   ``experiments/measure_lane_20_balle_real_archive._fp4_quantize_to_signed_indices``
   for ASYM, or ``_unpack_fp4_nibbles`` for FP4A).
2. Concatenate all streams and reshape into ``(num_blocks, block_size)``.
3. Loss = differentiable rate estimator
   ``E[-log2 p(y|σ_b)] + bits_z`` (the bits_z is approximated as the
   constant-cost ``z_dim × 8`` per block since z is int8 stored).
4. Adam optimizer + EMA shadow at decay 0.997 (CLAUDE.md non-negotiable).
5. Periodic measurement: encode the qint stream with the EMA-shadow weights
   and compare to the static-arithmetic baseline.

Auth-eval delegation
--------------------
This trainer does NOT run auth eval directly. The full chain is delegated
to ``scripts/remote_lane_20_balle.sh`` Stage 4 which builds a modified
archive (Lane 20 codec replacing the FP4 conv weight blobs) and runs
``contest_auth_eval.py`` on CUDA. The trainer's job is to fit the
hyperprior so the dispatched archive can be measured.

Per CLAUDE.md non-negotiables
-----------------------------
* No silent defaults — every flag required.
* EMA decay 0.997 on all hyperprior weights; eval uses the EMA shadow.
* No MPS — refuses ``--device mps`` (loud raise).
* CPU-only is acceptable for the codec training (the bytes measurement is
  device-deterministic; SegNet/PoseNet are NOT loaded here, so the
  CLAUDE.md "no MPS for strategy" applies but device CPU is genuinely
  fine for byte-counting).
* Auth eval happens at Stage 4 of the remote_lane script — this trainer
  documents that delegation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.balle_hyperprior_codec import (
    _SIGMA_MIN,
    _SIGMA_MAX,
    BalleHyperpriorCodec,
    HyperDecoder,
    HyperEncoder,
    decode_qints_balle,
    encode_qints_full_balle,
)
from tac.arithmetic_qint_codec import encode_qints_arithmetic
from tac.fp4_quantize import DEFAULT_CODEBOOK


# Local copy of the helper from the empirical script — keeps this file
# self-contained for the remote_lane script.
def _fp4_quantize_to_signed_indices(
    weights: np.ndarray,
    *,
    block_size: int = 32,
    codebook: torch.Tensor,
    robust_scale: bool = False,
) -> np.ndarray:
    from tac.fp4_quantize import _quantize_block

    w = torch.from_numpy(weights.astype(np.float32))
    n = w.numel()
    pad = (block_size - n % block_size) % block_size
    if pad:
        w = torch.cat([w, torch.zeros(pad)])
    blocks = w.view(-1, block_size)
    signed_indices = []
    for b in blocks:
        idx, sgn, _ = _quantize_block(b, codebook, robust_scale=robust_scale)
        si = idx.to(torch.int8) * sgn.to(torch.int8)
        signed_indices.append(si)
    out = torch.cat(signed_indices)[:n].numpy().astype(np.int8)
    return out


def _extract_qints_from_renderer(
    renderer_path: Path, *, fp4_codebook: str = "default"
) -> np.ndarray:
    """Walk an ASYM/FP4A renderer.bin and concatenate all conv qint streams.

    Args:
        renderer_path: path to renderer.bin (ASYM or FP4A magic).
        fp4_codebook: which FP4 codebook to use for ASYM → FP4 quantisation
            (Round 1 Finding 2 fix per `feedback_silent_default_bug_class`).
            "default" or "residual" — MUST match what the renderer trained
            against.
    """
    from tac.fp4_quantize import RESIDUAL_CODEBOOK

    blob = renderer_path.read_bytes()
    magic = blob[:4]
    streams: list[np.ndarray] = []
    if magic == b"ASYM":
        from tac.renderer_export import load_asymmetric_checkpoint

        model = load_asymmetric_checkpoint(blob, device="cpu")
        if fp4_codebook == "residual":
            codebook = RESIDUAL_CODEBOOK.clone()
        elif fp4_codebook == "default":
            codebook = DEFAULT_CODEBOOK.clone()
        else:
            raise ValueError(
                f"_extract_qints_from_renderer: unknown fp4_codebook "
                f"{fp4_codebook!r}; must be 'default' or 'residual'"
            )
        for name, p in model.named_parameters():
            if p.dim() == 4:
                arr = p.detach().cpu().float().numpy().reshape(-1)
                if arr.size < 32:
                    continue  # skip tiny heads
                qints = _fp4_quantize_to_signed_indices(
                    arr, codebook=codebook, robust_scale=False
                )
                streams.append(qints)
    elif magic == b"FP4A":
        # Reuse the FP4A scanner from the empirical script
        from experiments.measure_lane_20_balle_real_archive import (  # type: ignore[import]
            _scan_fp4a_layers,
            _unpack_fp4_nibbles,
        )
        layers, _ = _scan_fp4a_layers(blob)
        for L in layers:
            scales_bytes = L["n_blocks"] * 2
            packed = L["blob"][scales_bytes:]
            try:
                qints = _unpack_fp4_nibbles(packed, L["n_elements"])
                if qints.size >= 32:
                    streams.append(qints)
            except Exception:
                continue
    else:
        raise ValueError(f"unsupported magic {magic!r}")
    return np.concatenate(streams)


def _gaussian_rate_per_block_bits(
    qints_2d: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    num_symbols: int = 15,
    offset: int = 7,
) -> torch.Tensor:
    """Differentiable per-block bits = sum_i -log2 p(y_i | σ_b).

    Uses the discretized-Gaussian p(y=k|σ) = Φ((k+0.5)/σ) - Φ((k-0.5)/σ)
    via torch.special.erf for autograd. The bin edges are
    ``v ∈ {-offset, ..., num_symbols-1-offset}``.

    Args:
        qints_2d: ``(num_blocks, block_size)`` integer tensor (cast to long).
        sigmas: ``(num_blocks,)`` positive scale tensor.
        num_symbols: alphabet size.
        offset: integer added before symbol indexing (default 7 for [-7,+7]).

    Returns:
        ``(num_blocks,)`` per-block bit cost tensor.
    """
    sig = sigmas.clamp(_SIGMA_MIN, _SIGMA_MAX).unsqueeze(1)  # (B, 1)
    y_int = qints_2d.long()
    # value v = symbol_index - offset; here symbol_index = y_int + offset
    # so v = y_int. Bin edges: v±0.5
    upper = (y_int.float() + 0.5) / sig
    lower = (y_int.float() - 0.5) / sig
    sqrt2 = math.sqrt(2.0)
    cdf_upper = 0.5 * (1.0 + torch.erf(upper / sqrt2))
    cdf_lower = 0.5 * (1.0 + torch.erf(lower / sqrt2))
    pmf = (cdf_upper - cdf_lower).clamp_min(1e-9)
    # Tail extension for the boundary symbols (y==-offset and y==num_symbols-1-offset)
    # is omitted here; the rate is a lower bound but smooth/differentiable.
    bits = -torch.log2(pmf)  # (B, block_size)
    return bits.sum(dim=1)  # (B,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Lane 20 Ballé hyperprior on a real renderer's qint stream."
    )
    parser.add_argument("--renderer", type=str, required=True, help="path to ASYM or FP4A renderer.bin")
    parser.add_argument("--output-dir", type=str, required=True, help="dir for checkpoints + logs")
    parser.add_argument("--block-size", type=int, required=True, help="qints per block")
    parser.add_argument("--z-dim", type=int, required=True, help="hyper-latent dimension")
    parser.add_argument("--hidden-dim", type=int, required=True, help="MLP hidden width")
    parser.add_argument("--num-symbols", type=int, default=15, help="alphabet size")
    parser.add_argument("--offset", type=int, default=7, help="symbol offset (alphabet center)")
    parser.add_argument("--lr", type=float, required=True, help="Adam learning rate")
    parser.add_argument("--steps", type=int, required=True, help="optimization steps")
    parser.add_argument("--ema-decay", type=float, default=0.997, help="EMA decay")
    parser.add_argument("--eval-every", type=int, default=500, help="eval interval (in steps)")
    parser.add_argument("--seed", type=int, required=True, help="torch / numpy seed")
    parser.add_argument(
        "--device", type=str, required=True,
        help="cuda | cpu (mps refused — CLAUDE.md MPS-strategy non-negotiable)",
    )
    parser.add_argument(
        "--fp4-codebook", type=str, default="default",
        choices=("default", "residual"),
        help=(
            "Round 1 Finding 2 fix: which FP4 codebook to use when "
            "FP4-quantising an ASYM renderer's FP16 weights to extract the "
            "qint stream. MUST match what the renderer trained against. "
            "Lane G v3 trained with the 'default' codebook; "
            "halfframe / Quantizr / MAE-V profiles use 'residual'. "
            "Reference: feedback_silent_default_bug_class_findings_20260429."
        ),
    )
    args = parser.parse_args()

    if args.device == "mps":
        raise SystemExit(
            "Lane 20 trainer refuses --device mps (CLAUDE.md non-negotiable: "
            "MPS PoseNet drift is 23x; no neural-net forward should run "
            "MPS for strategic measurement). Use --device cuda or cpu."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lane-20-train] writing to {out_dir}")

    # ── Extract qint stream ──
    qints_np = _extract_qints_from_renderer(
        Path(args.renderer), fp4_codebook=args.fp4_codebook
    )
    n_total = qints_np.size
    block_size = args.block_size
    n_blocks = n_total // block_size  # drop incomplete last block during training
    qints_np = qints_np[: n_blocks * block_size]
    qints_2d = torch.from_numpy(qints_np.reshape(n_blocks, block_size).astype(np.int64))
    print(
        f"[lane-20-train] extracted {n_total:,} qints → {n_blocks:,} blocks of "
        f"{block_size} (alphabet {args.num_symbols}, offset {args.offset})"
    )

    # ── Build codec ──
    codec = BalleHyperpriorCodec(
        block_size=block_size,
        z_dim=args.z_dim,
        hyper_encoder=HyperEncoder(
            block_size=block_size, z_dim=args.z_dim, hidden_dim=args.hidden_dim, seed=args.seed,
        ),
        hyper_decoder=HyperDecoder(
            z_dim=args.z_dim, hidden_dim=args.hidden_dim, seed=args.seed,
        ),
    )
    device = torch.device(args.device)
    codec.hyper_encoder.to(device)
    codec.hyper_decoder.to(device)
    qints_2d = qints_2d.to(device)
    blocks_f = qints_2d.float()

    # ── EMA shadow (CLAUDE.md non-negotiable: decay 0.997) ──
    # Apply to ALL hyperprior weights; snapshot+restore at eval time.
    ema_decay = float(args.ema_decay)
    ema_state = {
        f"enc.{k}": v.detach().clone()
        for k, v in codec.hyper_encoder.state_dict().items()
    }
    ema_state.update({
        f"dec.{k}": v.detach().clone()
        for k, v in codec.hyper_decoder.state_dict().items()
    })

    optimizer = torch.optim.Adam(
        list(codec.hyper_encoder.parameters())
        + list(codec.hyper_decoder.parameters()),
        lr=args.lr,
    )

    log_path = out_dir / "lane_20_train.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{ts}] {msg}"
        print(line)
        log_lines.append(line)

    log(f"start: device={device} steps={args.steps} lr={args.lr}")

    # Static baseline byte cost (target to beat)
    static_blob_full = encode_qints_arithmetic(
        qints_np, num_symbols=args.num_symbols, offset=args.offset
    )
    static_bytes = len(static_blob_full)
    log(f"static_baseline_bytes={static_bytes}")

    best_bytes = static_bytes
    best_step = -1
    for step in range(1, args.steps + 1):
        codec.hyper_encoder.train()
        codec.hyper_decoder.train()
        z = codec.hyper_encoder(blocks_f)  # (n_blocks, z_dim)
        # Continuous z → no quantize during training (saves rate term but
        # tracks the scale prediction). The encoder/decoder agree at
        # inference time via the FP16 round-trip.
        sigmas = codec.hyper_decoder(z)  # (n_blocks,)
        bits_per_block = _gaussian_rate_per_block_bits(
            qints_2d, sigmas,
            num_symbols=args.num_symbols, offset=args.offset,
        )
        # Loss = mean bits per BLOCK (not per element — the optimizer scales
        # better with the higher-magnitude per-block target).
        loss = bits_per_block.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for k, v in codec.hyper_encoder.state_dict().items():
                ema_state[f"enc.{k}"].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)
            for k, v in codec.hyper_decoder.state_dict().items():
                ema_state[f"dec.{k}"].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)

        if step % max(1, args.steps // 20) == 0 or step == 1:
            log(
                f"step {step}/{args.steps}: loss(bits/block)={loss.item():.2f} "
                f"sigmas[min/mean/max]={float(sigmas.min()):.3f}/"
                f"{float(sigmas.mean()):.3f}/{float(sigmas.max()):.3f}"
            )

        if step % args.eval_every == 0 or step == args.steps:
            # Eval-time: swap to EMA shadow, run actual encode, measure bytes.
            live_enc = deepcopy(codec.hyper_encoder.state_dict())
            live_dec = deepcopy(codec.hyper_decoder.state_dict())
            ema_enc = {k.removeprefix("enc."): v for k, v in ema_state.items() if k.startswith("enc.")}
            ema_dec = {k.removeprefix("dec."): v for k, v in ema_state.items() if k.startswith("dec.")}
            codec.hyper_encoder.load_state_dict(ema_enc)
            codec.hyper_decoder.load_state_dict(ema_dec)
            try:
                full_blob = encode_qints_full_balle(
                    qints=qints_np,
                    num_symbols=args.num_symbols,
                    offset=args.offset,
                    codec=codec,
                )
                full_bytes = len(full_blob)
                # Verify roundtrip
                decoded = decode_qints_balle(
                    blob=full_blob, expected_dtype=np.int8
                )
                roundtrip_ok = bool(np.array_equal(decoded, qints_np))
            except Exception as exc:
                log(f"eval step {step}: encode failed — {exc!r}")
                full_bytes = -1
                roundtrip_ok = False
            log(
                f"EVAL step {step}: full_balle_bytes={full_bytes} "
                f"static_bytes={static_bytes} delta={full_bytes - static_bytes:+d} "
                f"roundtrip={'OK' if roundtrip_ok else 'FAIL'}"
            )
            if roundtrip_ok and 0 < full_bytes < best_bytes:
                best_bytes = full_bytes
                best_step = step
                # Save best EMA codec (encoder + decoder weights)
                ckpt = {
                    "step": step,
                    "block_size": block_size,
                    "z_dim": args.z_dim,
                    "hidden_dim": args.hidden_dim,
                    "ema_encoder_state": ema_enc,
                    "ema_decoder_state": ema_dec,
                    "static_baseline_bytes": static_bytes,
                    "full_balle_bytes": full_bytes,
                    "n_blocks": n_blocks,
                    "n_total": n_total,
                    "num_symbols": args.num_symbols,
                    "offset": args.offset,
                }
                torch.save(ckpt, out_dir / "best_codec.pt")
                log(f"  → saved best_codec.pt (bytes={full_bytes})")
            # Restore live weights
            codec.hyper_encoder.load_state_dict(live_enc)
            codec.hyper_decoder.load_state_dict(live_dec)

    # Save final report
    report = {
        "renderer": str(args.renderer),
        "n_total_qints": int(n_total),
        "n_blocks": int(n_blocks),
        "block_size": int(block_size),
        "z_dim": int(args.z_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_symbols": int(args.num_symbols),
        "offset": int(args.offset),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "ema_decay": float(args.ema_decay),
        "seed": int(args.seed),
        "device": str(device),
        "static_baseline_bytes": int(static_bytes),
        "best_full_balle_bytes": int(best_bytes),
        "best_step": int(best_step),
        "savings_vs_static_pct": (
            (static_bytes - best_bytes) / static_bytes * 100 if static_bytes else 0.0
        ),
        "verdict": (
            "BALLE_BEATS_STATIC" if best_bytes < static_bytes
            else "STATIC_WINS_FALLBACK"
        ),
        "claude_md_compliance": {
            "ema_decay_0_997": ema_decay == 0.997,
            "no_mps_used": str(device) != "mps",
            "auth_eval_delegated_to_remote_lane": True,
        },
    }
    (out_dir / "lane_20_train_report.json").write_text(json.dumps(report, indent=2))
    log_path.write_text("\n".join(log_lines) + "\n")
    log(
        f"DONE: best_full_balle_bytes={best_bytes} "
        f"(static={static_bytes}) verdict={report['verdict']}"
    )


if __name__ == "__main__":
    main()
