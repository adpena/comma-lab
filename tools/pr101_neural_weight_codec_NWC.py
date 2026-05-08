#!/usr/bin/env python3
"""PR101 Neural Weight Codec (NWC) — 1D-specific learned hyperprior with
shared encoder/decoder + per-tensor conditioning.

Background
----------
Both prior CompressAI experiments
(``tools/pr101_compressai_balle_hyperprior_full.py`` and
``tools/pr101_compressai_balle_FIXED.py``) inherit Balle's natural-image
inductive biases: GDN nonlinearities tuned for pixel-intensity statistics,
2D conv kernels assuming local spatial correlation, BPP-balanced losses
calibrated for ImageNet-scale natural images. PR101's INT8 weight stream
violates all three priors.

This tool builds a **fully custom** 1D neural weight codec with critical
design fixes vs the prior 28-independent-models approach:

1. **Shared encoder/decoder weights** across all 28 tensors — a 28× per-tensor
   architecture would cost 28 × 9 KB ≈ 250 KB just for the model overhead,
   exceeding the 178 KB PR101 brotli baseline before any data is encoded.
   The shared backbone with per-tensor conditioning (FiLM / tensor embedding)
   pays the model cost once.
2. **Per-tensor FiLM conditioning** — a learned (gamma, beta) pair per tensor
   modulates the shared encoder/decoder hidden activations. Each tensor's
   conditioning vector is small (2 × hidden ≈ 32 × 2 = 64 fp16 = 128 B).
3. **rd_lambda annealing** — starts very high (forces reconstruction to
   dominate) and decays to a low value (rate term gradually engages). This
   avoids the BPP collapse the prior tool hit.
4. **Custom 1D entropy bottleneck** via CompressAI's ``EntropyBottleneck``
   over the latent (no GDN, no 2D structure assumption).

CLAUDE.md compliance: MPS allowed (signal-only), evidence tagged
``[MPS-research-signal]``; no scorer-load, no score claim.
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

from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_neural_weight_codec_NWC.py"
SCHEMA_VERSION = "pr101_neural_weight_codec_NWC.v2"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_TENSORS = len(FIXED_STATE_SCHEMA)
PR101_BROTLI_BASELINE_BYTES = 178_144


# ----------------------------------------------------------------------
# Architecture: shared encoder/decoder + per-tensor FiLM
# ----------------------------------------------------------------------


class SharedEncoder(nn.Module):
    """Maps [B, chunk_T] -> [B, latent_dim] with FiLM-conditioning.

    FiLM(h, gamma, beta) = gamma * h + beta, applied after the GELU.
    """

    def __init__(self, chunk_T: int, hidden: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(chunk_T, hidden)
        self.fc2 = nn.Linear(hidden, latent_dim)
        self.act = nn.GELU()

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, chunk_T] ; gamma, beta: [hidden]
        h = self.act(self.fc1(x))
        h = h * (1.0 + gamma.view(1, -1)) + beta.view(1, -1)
        return self.fc2(h)


class SharedDecoder(nn.Module):
    """Maps [B, latent_dim] -> [B, chunk_T] with FiLM-conditioning."""

    def __init__(self, chunk_T: int, hidden: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, chunk_T)
        self.act = nn.GELU()

    def forward(
        self, z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    ) -> torch.Tensor:
        h = self.act(self.fc1(z))
        h = h * (1.0 + gamma.view(1, -1)) + beta.view(1, -1)
        return self.fc2(h)


class SharedNWC(CompressionModel):
    """Shared 1D NWC: one encoder/decoder + per-tensor FiLM (gamma, beta) +
    a single shared EntropyBottleneck."""

    def __init__(
        self, n_tensors: int, chunk_T: int, hidden: int, latent_dim: int,
    ):
        super().__init__()
        self.n_tensors = n_tensors
        self.chunk_T = chunk_T
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.encoder = SharedEncoder(chunk_T, hidden, latent_dim)
        self.decoder = SharedDecoder(chunk_T, hidden, latent_dim)
        self.entropy_bottleneck = EntropyBottleneck(latent_dim)
        # FiLM parameters per tensor: 4 x [n_tensors, hidden]
        # (encoder gamma/beta, decoder gamma/beta), init small
        self.enc_gamma = nn.Parameter(torch.zeros(n_tensors, hidden))
        self.enc_beta = nn.Parameter(torch.zeros(n_tensors, hidden))
        self.dec_gamma = nn.Parameter(torch.zeros(n_tensors, hidden))
        self.dec_beta = nn.Parameter(torch.zeros(n_tensors, hidden))

    def forward(
        self, chunks: torch.Tensor, tensor_idx: int,
    ) -> dict:
        """chunks: [B, chunk_T] ; tensor_idx: int in [0, n_tensors)"""
        eg, eb = self.enc_gamma[tensor_idx], self.enc_beta[tensor_idx]
        dg, db = self.dec_gamma[tensor_idx], self.dec_beta[tensor_idx]
        z = self.encoder(chunks, eg, eb)  # [B, latent_dim]
        # EB expects [1, latent_dim, B]
        z_eb = z.transpose(0, 1).unsqueeze(0)
        z_hat, z_likelihoods = self.entropy_bottleneck(z_eb)
        z_hat = z_hat.squeeze(0).transpose(0, 1)  # [B, latent_dim]
        x_hat = self.decoder(z_hat, dg, db)
        return {"x_hat": x_hat, "likelihoods": {"z": z_likelihoods}}

    def compress(self, chunks: torch.Tensor, tensor_idx: int) -> dict:
        eg = self.enc_gamma[tensor_idx]
        eb = self.enc_beta[tensor_idx]
        z = self.encoder(chunks, eg, eb)
        z_eb = z.transpose(0, 1).unsqueeze(0)
        strings = self.entropy_bottleneck.compress(z_eb)
        return {"strings": strings, "shape": z_eb.size()[-1:]}

    def decompress(self, strings, shape, tensor_idx: int) -> dict:
        dg = self.dec_gamma[tensor_idx]
        db = self.dec_beta[tensor_idx]
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        z_hat = z_hat.squeeze(0).transpose(0, 1)
        x_hat = self.decoder(z_hat, dg, db)
        return {"x_hat": x_hat}


# ----------------------------------------------------------------------
# Substrate prep — per-tensor chunked
# ----------------------------------------------------------------------


@dataclass
class TensorChunked:
    name: str
    raw: np.ndarray  # int32 [n]
    n: int
    n_pad: int
    n_chunks: int
    chunk_T: int
    z_mean: float
    z_std: float
    chunks: torch.Tensor  # [n_chunks, chunk_T] fp32


def collect_per_tensor_chunked(
    state_dict_path: Path, chunk_T: int,
) -> list[TensorChunked]:
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    out: list[TensorChunked] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        raw = qt.q_i8.astype(np.int32).flatten()
        n = raw.size
        n_chunks = (n + chunk_T - 1) // chunk_T
        n_pad = n_chunks * chunk_T - n
        z_mean = float(raw.mean())
        z_std = float(raw.std())
        if z_std < 1e-6:
            z_std = 1.0
        padded = np.concatenate([raw, np.zeros(n_pad, dtype=np.int32)])
        norm = (padded.astype(np.float32) - z_mean) / z_std
        chunks = torch.from_numpy(norm.reshape(n_chunks, chunk_T))
        out.append(TensorChunked(
            name=name, raw=raw, n=n, n_pad=n_pad, n_chunks=n_chunks,
            chunk_T=chunk_T, z_mean=z_mean, z_std=z_std, chunks=chunks,
        ))
    return out


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train_shared_nwc(
    model: SharedNWC,
    tensors: list[TensorChunked],
    *,
    device: str,
    epochs: int,
    lr: float,
    aux_lr: float,
    rd_lambda_init: float,
    rd_lambda_final: float,
    log_every: int = 25,
) -> dict:
    """Train shared NWC on all tensors with rd_lambda annealing.

    Each epoch processes ALL 28 tensors (one forward pass per tensor),
    accumulating gradients. The tensor with the most chunks dominates the
    loss naturally (good — it's the most byte-impactful).
    """
    dev = torch.device(device)
    model.to(dev).train()
    chunks_dev = [t.chunks.to(dev) for t in tensors]

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

    history = []
    t0 = time.time()
    for ep in range(epochs):
        frac = ep / max(epochs - 1, 1)
        rd_lambda = rd_lambda_init * ((rd_lambda_final / rd_lambda_init) ** frac)

        # Process all tensors, accumulate loss
        optim.zero_grad()
        aux_optim.zero_grad()
        total_loss = torch.zeros(1, device=dev)
        total_mse = 0.0
        total_bits = 0.0
        total_n_elements = 0

        for i, tc in enumerate(tensors):
            cdev = chunks_dev[i]
            n_elements = tc.n_chunks * tc.chunk_T

            out = model(cdev, tensor_idx=i)
            x_hat = out["x_hat"]
            z_likelihoods = out["likelihoods"]["z"]

            mse = (x_hat - cdev).pow(2).mean()
            bits = -torch.log2(z_likelihoods).sum()
            bpe = bits / n_elements

            # Weight per-tensor loss by tensor size (so big tensors matter more)
            weight = n_elements
            total_loss = total_loss + weight * (rd_lambda * mse * 255.0 ** 2 + bpe)
            total_mse += float(mse.item()) * weight
            total_bits += float(bits.item())
            total_n_elements += weight

        total_loss = total_loss / total_n_elements
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params, max_norm=1.0)
        optim.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optim.step()

        if (ep + 1) % log_every == 0 or ep == 0:
            avg_mse = total_mse / total_n_elements
            avg_bpe = total_bits / total_n_elements
            history.append({
                "epoch": ep + 1,
                "rd_lambda": rd_lambda,
                "loss": float(total_loss.item()),
                "mse_norm": avg_mse,
                "bpe": avg_bpe,
                "elapsed_sec": time.time() - t0,
            })
            print(
                f"  ep {ep+1:4d}/{epochs}  rd_lam={rd_lambda:>7.3f}  "
                f"loss={total_loss.item():.4f}  mse={avg_mse:.4f}  bpe={avg_bpe:.4f}  "
                f"t={time.time()-t0:.0f}s"
            )

    model.cpu()
    return {"epochs_trained": epochs, "elapsed_sec": time.time() - t0, "history": history}


def serialize_shared_model(model: SharedNWC) -> bytes:
    sd = model.state_dict()
    buf = bytearray()
    buf += b"NWC2"  # shared variant magic
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


def encode_decode_all(
    model: SharedNWC,
    tensors: list[TensorChunked],
) -> dict:
    model.eval()
    model.update(force=True)

    per_tensor = []
    total_payload_raw = 0
    cum_abs_err = 0.0
    cum_abs_orig = 0.0

    for i, tc in enumerate(tensors):
        with torch.no_grad():
            out = model.compress(tc.chunks, tensor_idx=i)
            dec = model.decompress(out["strings"], out["shape"], tensor_idx=i)
            x_hat = dec["x_hat"]
        strings = out["strings"]
        bytes_raw = sum(len(s) for s in strings)
        total_payload_raw += bytes_raw

        rec = x_hat.cpu().numpy().flatten() * tc.z_std + tc.z_mean
        rec_real = np.round(rec[: tc.n]).clip(-N_QUANT, N_QUANT).astype(np.int32)
        abs_err = np.abs(rec_real - tc.raw).astype(np.float64)
        cum_abs_err += abs_err.sum()
        cum_abs_orig += np.abs(tc.raw).astype(np.float64).sum()

        per_tensor.append({
            "tensor": tc.name,
            "n": tc.n,
            "n_chunks": tc.n_chunks,
            "bytes_raw": bytes_raw,
            "mean_abs_symbol_err": float(abs_err.mean()),
            "max_abs_symbol_err": int(abs_err.max()),
            "abs_err_sum": float(abs_err.sum()),
            "n_strings": len(strings),
        })

    # Concatenate ALL tensor payloads into one blob, then brotli it once
    blob = bytearray()
    blob += struct.pack("<I", len(tensors))
    for r in per_tensor:
        # We don't actually persist the per-tensor strings here; for byte
        # accounting we collect them in the same loop:
        pass
    # Re-collect bytes for blob
    blob = bytearray()
    blob += struct.pack("<I", len(tensors))
    for i, tc in enumerate(tensors):
        with torch.no_grad():
            out = model.compress(tc.chunks, tensor_idx=i)
        ss = out["strings"]
        blob += struct.pack("<I", len(ss))
        for s in ss:
            blob += struct.pack("<I", len(s)) + s

    payload_brotli = brotli.compress(bytes(blob), quality=11, lgwin=22, lgblock=24)
    model_blob = serialize_shared_model(model)
    model_brotli = brotli.compress(model_blob, quality=11, lgwin=22, lgblock=24)

    rel_err = cum_abs_err / cum_abs_orig if cum_abs_orig > 1e-9 else cum_abs_err
    side_info_bytes = N_TENSORS * 8  # z_mean, z_std per tensor
    archive_bytes = (
        len(payload_brotli) + len(model_brotli) + side_info_bytes
        + ARCHIVE_OVERHEAD_BYTES
    )

    return {
        "payload_bytes_raw": total_payload_raw,
        "payload_brotli_bytes": len(payload_brotli),
        "model_blob_raw_bytes": len(model_blob),
        "model_brotli_bytes": len(model_brotli),
        "side_info_bytes": side_info_bytes,
        "archive_bytes": archive_bytes,
        "rel_err": rel_err,
        "per_tensor": per_tensor,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_one(
    tensors: list[TensorChunked],
    *,
    chunk_T: int,
    hidden: int,
    latent_dim: int,
    epochs: int,
    rd_lambda_init: float,
    rd_lambda_final: float,
    lr: float,
    device: str,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SharedNWC(
        n_tensors=len(tensors), chunk_T=chunk_T,
        hidden=hidden, latent_dim=latent_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[nwc-shared] chunk_T={chunk_T} hidden={hidden} latent_dim={latent_dim} "
        f"params={n_params:,} (~{n_params*2/1024:.1f} KB fp16) "
        f"rd_lambda={rd_lambda_init}->{rd_lambda_final} epochs={epochs} device={device}"
    )

    train_log = train_shared_nwc(
        model, tensors,
        device=device, epochs=epochs,
        lr=lr, aux_lr=1e-2,
        rd_lambda_init=rd_lambda_init,
        rd_lambda_final=rd_lambda_final,
    )
    meas = encode_decode_all(model, tensors)

    return {
        "chunk_T": chunk_T,
        "hidden": hidden,
        "latent_dim": latent_dim,
        "rd_lambda_init": rd_lambda_init,
        "rd_lambda_final": rd_lambda_final,
        "epochs": epochs,
        "n_params": n_params,
        **meas,
        "training_history": train_log["history"],
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
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument(
        "--chunk-T", type=int, default=64,
        help="Symbol chunk size; latent_dim per chunk encodes chunk_T symbols.",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="32:8,32:4,16:4,64:8",
        help="Comma-separated hidden:latent_dim configs to sweep.",
    )
    p.add_argument("--rd-lambda-init", type=float, default=100.0)
    p.add_argument("--rd-lambda-final", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-3)
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
        args.output_dir = REPO_ROOT / f"reports/raw/pr101_nwc_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[nwc-shared] state_dict: {args.state_dict}")
    print(f"[nwc-shared] output:     {args.output_dir}")
    print(f"[nwc-shared] device:     {args.device}")

    tensors = collect_per_tensor_chunked(args.state_dict, chunk_T=args.chunk_T)
    print(f"[nwc-shared] {len(tensors)} tensors loaded; chunk_T={args.chunk_T}")
    print(f"      smallest tensor: {min(t.n for t in tensors)} symbols")
    print(f"      largest tensor:  {max(t.n for t in tensors)} symbols")
    print(f"      total real symbols: {sum(t.n for t in tensors):,}")

    configs: list[tuple[int, int]] = []
    for tok in args.configs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 2:
            raise SystemExit(f"bad config: {tok!r} (want hidden:latent)")
        configs.append((int(parts[0]), int(parts[1])))

    all_results = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": "[MPS-research-signal]",
        "device": args.device,
        "input_state_dict": str(args.state_dict),
        "chunk_T": args.chunk_T,
        "results": [],
    }

    for (hidden, latent_dim) in configs:
        try:
            r = run_one(
                tensors,
                chunk_T=args.chunk_T,
                hidden=hidden, latent_dim=latent_dim,
                epochs=args.epochs,
                rd_lambda_init=args.rd_lambda_init,
                rd_lambda_final=args.rd_lambda_final,
                lr=args.lr, device=args.device, seed=args.seed,
            )
        except Exception as exc:
            r = {
                "hidden": hidden, "latent_dim": latent_dim,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  CONFIG FAILED: {r['error']}")
        all_results["results"].append(r)
        print(
            f"  -> hidden={hidden} latent_dim={latent_dim}  "
            f"archive={r.get('archive_bytes', '-'):>10}  rel_err={r.get('rel_err', '-'):>8}"
        )
        (args.output_dir / "manifest.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )

    ok = [r for r in all_results["results"] if "error" not in r and r.get("archive_bytes") is not None]
    if not ok:
        print("\n[nwc-shared] NO successful configs")
        return 1
    best = min(ok, key=lambda r: (r["rel_err"], r["archive_bytes"]))
    print()
    print("=" * 70)
    print("[nwc-shared] SUMMARY")
    print("=" * 70)
    print(f"BEST: hidden={best['hidden']} latent_dim={best['latent_dim']}")
    print(f"  archive_bytes : {best['archive_bytes']:,} B  [MPS-research-signal]")
    print(f"  vs baseline   : {best['archive_bytes'] - PR101_BROTLI_BASELINE_BYTES:+,} B")
    print(f"  rel_err       : {best['rel_err']:.4f}")

    all_results["best_archive_bytes"] = best["archive_bytes"]
    all_results["best_rel_err"] = best["rel_err"]
    all_results["best_config"] = f"hidden{best['hidden']}_latent{best['latent_dim']}"
    (args.output_dir / "manifest.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    delta = best["archive_bytes"] - PR101_BROTLI_BASELINE_BYTES
    if best["rel_err"] < 0.05 and delta < 0:
        verdict = (
            f"BEAT-PR101-brotli ({delta:+,} B) at rel_err {best['rel_err']:.4f}; "
            f"REOPEN-FOR-DISPATCH-CONSIDERATION (1D NWC shared backbone)"
        )
        evidence_row = {
            "technique": "neural_weight_codec_NWC",
            "empirical_archive_bytes": best["archive_bytes"],
            "empirical_rel_err": best["rel_err"],
            "evidence_grade": "[MPS-research-signal]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "source": (
                f"[MPS-research-signal] {args.output_dir}/manifest.json "
                f"(custom 1D Neural Weight Codec, shared backbone + per-tensor FiLM)"
            ),
            "timestamp": timestamp,
            "contest_dispatch_verdict": verdict,
        }
        args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.evidence_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"[nwc-shared] evidence row appended to {args.evidence_jsonl}")
        verdict_line = verdict
    elif best["rel_err"] < 0.05:
        verdict_line = f"DEFERRED (rel_err OK but archive {delta:+,} B vs baseline)"
    else:
        verdict_line = f"DEFERRED-pending-research (rel_err {best['rel_err']:.4f} > 0.05)"
    print(f"verdict  : {verdict_line}")
    print(f"manifest : {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
