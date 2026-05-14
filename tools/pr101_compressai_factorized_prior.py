#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR101 CompressAI FactorizedPrior — alternative learned hyperprior class.

Background
----------
The prior ``tools/pr101_compressai_balle_hyperprior_full.py`` ran the
canonical 2D ``ScaleHyperprior`` and concluded substrate-mismatch (image-class
hyperprior cannot exploit PR101's near-iid INT8 weight-symbols). The reasoning
is sound for that architecture, but the wider Balle family includes the
simpler ``FactorizedPrior`` (the 2018 ICLR paper's bottleneck-only baseline
without a hyperprior) and we have not tried it on PR101.

This tool builds a **1D-conv FactorizedPrior** on PR101's flat 228,958-element
INT8 weight stream:

- Drop the 2D conv stack (which assumes natural-image spatial structure).
- Replace with 1D conv (or Linear if N=M=channels) transforms.
- Keep the canonical CompressAI ``EntropyBottleneck`` (factorized prior over
  the latent ``y``).
- No hyperprior side-info: the factorized prior on ``y`` is fixed (after
  training); only ``y_strings`` are charged.

Why it might work
-----------------
- The FactorizedPrior is correct for **near-iid** sources whose latent
  becomes near-iid through the analysis transform — exactly PR101's regime.
- 1D conv with stride-2 is the natural analog for sequence data; it preserves
  causal locality without assuming 2D.
- Compared to the 2D ScaleHyperprior, this drops the entire hyperprior chain
  (h_a, h_s, scale-conditioned Gaussian, z_strings), saving ~30-50% of
  trainable parameters.

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

TOOL_NAME = "tools/pr101_compressai_factorized_prior.py"
SCHEMA_VERSION = "pr101_compressai_factorized_prior.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_TENSORS = len(FIXED_STATE_SCHEMA)
PR101_BROTLI_BASELINE_BYTES = 178_144
EVIDENCE_GRADE = "[MPS-research-signal]"
EVIDENCE_SEMANTICS = "mps_proxy_factorized_prior_no_score"
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


# ----------------------------------------------------------------------
# 1D FactorizedPrior — Balle's bmshj2018-factorized adapted to 1D weight streams
# ----------------------------------------------------------------------


def _conv1d(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.Conv1d:
    """1D analog of CompressAI's `conv` helper."""
    return nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)


def _deconv1d(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.ConvTranspose1d:
    """1D analog of CompressAI's `deconv` helper."""
    return nn.ConvTranspose1d(
        in_ch, out_ch, kernel_size, stride=stride,
        padding=kernel_size // 2, output_padding=stride - 1,
    )


class GDN1d(nn.Module):
    """Generalized Divisive Normalization for 1D channel maps.

    GDN(x_i) = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)

    Same form as Balle's 2D GDN, just with channel-wise normalization across
    the (B, C, T) tensor.
    """

    def __init__(self, n_channels: int, inverse: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.inverse = inverse
        # Identity-init beta (small positive) and gamma (small diagonal)
        self.beta = nn.Parameter(torch.ones(n_channels) * 1e-1)
        self.gamma = nn.Parameter(torch.eye(n_channels) * 1e-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        # gamma: [C_out, C_in] -> applied as 1x1 conv on x^2
        gamma = self.gamma.abs() + 1e-6  # ensure positive
        beta = self.beta.abs() + 1e-6
        x2 = x.pow(2)
        # norm[b, c, t] = beta[c] + sum_j gamma[c, j] * x2[b, j, t]
        norm = torch.einsum("ij,bjt->bit", gamma, x2) + beta.view(1, -1, 1)
        if self.inverse:
            return x * norm.sqrt()
        return x * torch.rsqrt(norm)


class FactorizedPrior1D(CompressionModel):
    """1D FactorizedPrior for PR101 weight-symbol streams.

    Architecture (mirrors Balle's 2D FactorizedPrior):
      g_a: 4 × (Conv1d s=2 + GDN1d)  --> M-channel latent at T/16
      g_s: 4 × (DeconvTranspose1d s=2 + GDN1d_inv)  --> 1-channel reconstruction
      EntropyBottleneck on the M-channel latent
    """

    def __init__(self, N: int, M: int, in_channels: int = 1):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.g_a = nn.Sequential(
            _conv1d(in_channels, N),
            GDN1d(N),
            _conv1d(N, N),
            GDN1d(N),
            _conv1d(N, N),
            GDN1d(N),
            _conv1d(N, M),
        )
        self.g_s = nn.Sequential(
            _deconv1d(M, N),
            GDN1d(N, inverse=True),
            _deconv1d(N, N),
            GDN1d(N, inverse=True),
            _deconv1d(N, N),
            GDN1d(N, inverse=True),
            _deconv1d(N, in_channels),
        )
        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        # x: [B, in_channels, T] (T must be multiple of 16)
        y = self.g_a(x)
        # EntropyBottleneck expects [B, C, *spatial_dims]; for 1D the convention
        # is [B, C, T] which CompressAI handles natively (it sums log-probs
        # across all non-channel dims).
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        # Trim or pad to original input length
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = x_hat[..., : x.shape[-1]]
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-1:], "input_T": x.shape[-1]}

    def decompress(self, strings, shape, input_T: int | None = None):
        assert isinstance(strings, list) and len(strings) == 1
        # EntropyBottleneck.decompress expects (B,C,T) shape via decompress(string, size)
        # where size=(T,) for 1D
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat)
        if input_T is not None and x_hat.shape[-1] != input_T:
            x_hat = x_hat[..., :input_T]
        return {"x_hat": x_hat}


# ----------------------------------------------------------------------
# Substrate prep: flat 1D
# ----------------------------------------------------------------------


@dataclass
class FlatSubstrate:
    raw: np.ndarray  # int32 [n_real]
    n_real: int
    n_pad: int
    T: int  # padded sequence length, divisible by 16
    z_mean: float
    z_std: float
    sequence: torch.Tensor  # [1, 1, T] fp32


def collect_flat_substrate(state_dict_path: Path) -> FlatSubstrate:
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    chunks: list[np.ndarray] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        chunks.append(qt.q_i8.astype(np.int32).flatten())
    raw = np.concatenate(chunks)
    n = raw.size
    # Pad T up to multiple of 16
    T = ((n + 15) // 16) * 16
    pad = T - n
    z_mean = float(raw.mean())
    z_std = float(raw.std())
    if z_std < 1e-6:
        z_std = 1.0
    padded = np.concatenate([raw, np.zeros(pad, dtype=np.int32)])
    norm = (padded.astype(np.float32) - z_mean) / z_std
    sequence = torch.from_numpy(norm).view(1, 1, T)
    return FlatSubstrate(
        raw=raw, n_real=n, n_pad=pad, T=T,
        z_mean=z_mean, z_std=z_std, sequence=sequence,
    )


# ----------------------------------------------------------------------
# Training, encoding, measurement
# ----------------------------------------------------------------------


def train_factorized(
    model: FactorizedPrior1D,
    sequence: torch.Tensor,
    *,
    device: str,
    epochs: int,
    lr: float,
    aux_lr: float,
    rd_lambda: float,
    log_every: int = 25,
    chunk_T: int | None = None,
) -> dict:
    """Train the 1D FactorizedPrior.

    If `chunk_T` is set, the sequence is split into batched chunks of that
    length (multiple of 16), enabling proper minibatch SGD instead of single-
    image overfit. Default is no chunking.
    """
    dev = torch.device(device)
    model.to(dev).train()
    seq = sequence.to(dev)

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

    full_T = seq.shape[-1]

    history = []
    t0 = time.time()
    for ep in range(epochs):
        if chunk_T is not None and chunk_T < full_T:
            # Random crop of chunk_T (multiple of 16) for batched training
            n_chunks = full_T // chunk_T
            idx = torch.randint(0, n_chunks, (1,)).item()
            x = seq[..., idx * chunk_T : (idx + 1) * chunk_T]
        else:
            x = seq
        n_pix = x.shape[-1]

        out = model(x)
        x_hat = out["x_hat"]
        y_likelihoods = out["likelihoods"]["y"]

        mse = (x_hat - x).pow(2).mean()
        bpp_y = -torch.log2(y_likelihoods).sum() / n_pix
        loss = rd_lambda * mse * 255.0 ** 2 + bpp_y

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
                "bpp_y": float(bpp_y.item()),
                "aux_loss": float(aux_loss.item()),
                "elapsed_sec": time.time() - t0,
            })
            print(
                f"  ep {ep+1:4d}/{epochs}  loss={loss.item():.4f}  "
                f"bpp_y={bpp_y.item():.4f}  mse={mse.item():.5f}  "
                f"t={time.time()-t0:.0f}s"
            )

    model.cpu()
    return {"epochs_trained": epochs, "elapsed_sec": time.time() - t0, "history": history}


def serialize_model(model: nn.Module) -> bytes:
    sd = model.state_dict()
    buf = bytearray()
    buf += b"FP1D"  # FactorizedPrior 1D magic
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


def encode_decode_measure(
    model: FactorizedPrior1D,
    sub: FlatSubstrate,
) -> dict:
    model.eval()
    model.update(force=True)
    seq = sub.sequence
    with torch.no_grad():
        out = model.compress(seq)
        # decompress wants the latent shape returned by compress
        dec = model.decompress(
            out["strings"], out["shape"], input_T=seq.shape[-1],
        )
        x_hat = dec["x_hat"]
    y_strings = out["strings"][0]
    y_bytes = sum(len(s) for s in y_strings)

    payload_concat = (
        struct.pack("<I", len(y_strings))
        + b"".join(struct.pack("<I", len(s)) + s for s in y_strings)
    )
    payload_brotli = brotli.compress(payload_concat, quality=11, lgwin=22, lgblock=24)

    rec = x_hat.cpu().squeeze().flatten().numpy() * sub.z_std + sub.z_mean
    rec_real = np.round(rec[: sub.n_real]).clip(-N_QUANT, N_QUANT).astype(np.int32)
    abs_err = np.abs(rec_real - sub.raw).astype(np.float64)
    abs_orig = np.abs(sub.raw).astype(np.float64)
    denom = abs_orig.sum()
    rel_err = float(abs_err.sum() / denom) if denom > 1e-9 else float(abs_err.sum())  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for FALSIFIED CompressAI factorized-prior sweep; not allocator-fed

    model_blob = serialize_model(model)
    model_brotli = brotli.compress(model_blob, quality=11, lgwin=22, lgblock=24)

    archive_bytes = (
        len(payload_brotli) + len(model_brotli) + 16  # 8 floats z_mean,z_std side info
        + ARCHIVE_OVERHEAD_BYTES
    )

    return {
        "y_bytes_raw": y_bytes,
        "payload_brotli_bytes": len(payload_brotli),
        "model_brotli_bytes": len(model_brotli),
        "rel_err": rel_err,
        "mean_abs_symbol_err": float(abs_err.mean()),
        "max_abs_symbol_err": int(abs_err.max()),
        "archive_bytes": archive_bytes,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_one(
    sub: FlatSubstrate,
    *,
    N: int,
    M: int,
    epochs: int,
    rd_lambda: float,
    lr: float,
    chunk_T: int | None,
    device: str,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FactorizedPrior1D(N=N, M=M, in_channels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[fp1d] N={N} M={M} params={n_params:,} (~{n_params*2/1024:.1f} KB fp16) "
        f"rd_lambda={rd_lambda} epochs={epochs} chunk_T={chunk_T} device={device}"
    )
    train_log = train_factorized(
        model, sub.sequence,
        device=device, epochs=epochs, lr=lr, aux_lr=1e-2,
        rd_lambda=rd_lambda, chunk_T=chunk_T,
    )
    meas = encode_decode_measure(model, sub)
    return {
        "N": N, "M": M, "rd_lambda": rd_lambda,
        "epochs": epochs, "chunk_T": chunk_T, "n_params": n_params,
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
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument(
        "--configs",
        type=str,
        default="8:4,12:6,16:8,8:8",
        help="Comma-separated N:M tuples (no model-class prefix; only FactorizedPrior).",
    )
    p.add_argument(
        "--rd-lambda-sweep",
        type=str,
        default="0.01,0.1,1.0,10.0,100.0",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--chunk-T",
        type=int,
        default=2048,
        help="Train on random chunks of this length (must be multiple of 16). "
             "Set to 0 to disable chunking (single-image overfit).",
    )
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
        args.output_dir = REPO_ROOT / f"reports/raw/pr101_factorized_prior_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunk_T = args.chunk_T if args.chunk_T > 0 else None
    if chunk_T is not None and chunk_T % 16 != 0:
        raise SystemExit(f"--chunk-T must be multiple of 16, got {chunk_T}")

    print(f"[fp1d] state_dict: {args.state_dict}")
    print(f"[fp1d] output:     {args.output_dir}")
    print(f"[fp1d] device:     {args.device}")

    sub = collect_flat_substrate(args.state_dict)
    print(
        f"[fp1d] substrate: n_real={sub.n_real:,} pad={sub.n_pad} T={sub.T} "
        f"z_mean={sub.z_mean:.3f} z_std={sub.z_std:.3f}"
    )

    configs: list[tuple[int, int]] = []
    for tok in args.configs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 2:
            raise SystemExit(f"bad config: {tok!r} (want N:M)")
        configs.append((int(parts[0]), int(parts[1])))
    rd_lambdas = [float(x) for x in args.rd_lambda_sweep.split(",") if x.strip()]

    all_results = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        **proxy_evidence_contract(),
        "device": args.device,
        "input_state_dict": str(args.state_dict),
        "results": [],
    }

    for (N, M) in configs:
        for rdl in rd_lambdas:
            try:
                r = run_one(
                    sub, N=N, M=M, epochs=args.epochs, rd_lambda=rdl,
                    lr=args.lr, chunk_T=chunk_T, device=args.device, seed=args.seed,
                )
            except Exception as exc:
                r = {"N": N, "M": M, "rd_lambda": rdl, "error": f"{type(exc).__name__}: {exc}"}
                print(f"  CONFIG FAILED: {r['error']}")
            all_results["results"].append(r)
            print(
                f"  -> N={N} M={M} rd_lambda={rdl}  archive={r.get('archive_bytes', '-'):>10}  "
                f"rel_err={r.get('rel_err', '-'):>8}"
            )
            (args.output_dir / "manifest.json").write_text(
                json.dumps(all_results, indent=2), encoding="utf-8"
            )

    ok = [r for r in all_results["results"] if "error" not in r and r.get("archive_bytes") is not None]
    if not ok:
        print("\n[fp1d] NO successful configs")
        return 1
    best = min(ok, key=lambda r: (r["rel_err"], r["archive_bytes"]))
    print()
    print("=" * 70)
    print("[fp1d] SUMMARY")
    print("=" * 70)
    print(f"BEST: N={best['N']} M={best['M']} rd_lambda={best['rd_lambda']}")
    print(f"  archive_bytes : {best['archive_bytes']:,} B  [MPS-research-signal]")
    print(f"  vs baseline   : {best['archive_bytes'] - PR101_BROTLI_BASELINE_BYTES:+,} B")
    print(f"  rel_err       : {best['rel_err']:.4f}")

    all_results["best_archive_bytes"] = best["archive_bytes"]
    all_results["best_rel_err"] = best["rel_err"]
    all_results["best_config"] = f"N{best['N']}_M{best['M']}_rdl{best['rd_lambda']}"
    (args.output_dir / "manifest.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    delta = best["archive_bytes"] - PR101_BROTLI_BASELINE_BYTES
    if best["rel_err"] < 0.05 and delta < 0:
        verdict = (
            f"BEAT-PR101-brotli ({delta:+,} B) at rel_err {best['rel_err']:.4f}; "
            f"REOPEN-FOR-DISPATCH-CONSIDERATION (1D FactorizedPrior)"
        )
        evidence_row = {
            "technique": "compressai_factorized_prior_1d",
            "empirical_archive_bytes": best["archive_bytes"],
            "empirical_rel_err": best["rel_err"],
            **proxy_evidence_contract(),
            "source": (
                f"[MPS-research-signal] {args.output_dir}/manifest.json "
                f"(1D FactorizedPrior, no spatial assumption)"
            ),
            "timestamp": timestamp,
            "contest_dispatch_verdict": verdict,
        }
        args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.evidence_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"[fp1d] evidence row appended to {args.evidence_jsonl}")
    elif best["rel_err"] < 0.05:
        verdict = f"DEFERRED (rel_err OK but archive {delta:+,} B vs baseline)"
    else:
        verdict = f"DEFERRED-pending-research (rel_err {best['rel_err']:.4f} > 0.05)"
    print(f"verdict  : {verdict}")
    print(f"manifest : {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
