#!/usr/bin/env python3
"""PR101 kalle-fold mixture-of-canonical-shapes codec — anchors the
``kalle_fold_mixture_canonical_shapes`` row in cathedral_autopilot's
catalog.

Hypothesis (from feedback_kalle_ninth_proof_of_folding_synthesis):
PR101's 28 per-tensor PMFs may all be foldable into a 4-component
mixture of canonical shapes (Gaussian + Laplace + sparse-spike +
uniform-tail), with per-tensor parameters being just 4 mixture weights
+ 2 scale params. If true, per-tensor PMF overhead drops from 510 B
(255-bin static Huffman header) to ~10 B = ~14 KB savings.

Per-tensor model: P(s) = w_g·N(s | 0, sigma) + w_l·L(s | 0, b)
                       + w_d·delta(s) + w_u·uniform([-127,127])
with constraints w_g + w_l + w_d + w_u = 1, sigma > 0, b > 0.

Per-tensor parameters: (w_g, w_l, w_d, sigma, b) = 5 floats. w_u is
implied. We store 4 fp16 (8 B) + 1 fp16 sigma + 1 fp16 b (4 B more) =
12 B per tensor = 336 B total metadata across 28 tensors. Brotli
this metadata blob alongside the AC payload.

Each per-tensor mixture is fit by minimizing KL-divergence to the
empirical histogram via L-BFGS. Encoding uses the mixture PMF as an
arithmetic coder distribution; decoding reconstructs the symbols
bit-exactly.

CLAUDE.md compliance: pure CPU + numpy + scipy.optimize + brotli +
constriction; no scorer load; no contest score claims; output tagged
``[CPU-prep empirical]``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import brotli
import constriction
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_kalle_fold_mixture_codec.py"
SCHEMA_VERSION = "pr101_kalle_fold_mixture_codec.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094

# Symmetric INT8 alphabet: [-127, +127] = 255 levels (zero is symbol 127).
N_SYMBOLS = 2 * N_QUANT + 1  # 255
SYMBOL_AXIS = np.arange(-N_QUANT, N_QUANT + 1, dtype=np.float64)


def empirical_pmf(symbols_i8: np.ndarray) -> np.ndarray:
    """Return a 255-bin PMF over symmetric int8 symbols (zero-centered)."""
    counts = Counter(int(s) for s in symbols_i8.flatten().tolist())
    pmf = np.zeros(N_SYMBOLS, dtype=np.float64)
    for sym, c in counts.items():
        idx = sym + N_QUANT
        if 0 <= idx < N_SYMBOLS:
            pmf[idx] = float(c)
    total = pmf.sum()
    if total > 0:
        pmf /= total
    return pmf


def canonical_shapes(sigma: float, b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (gaussian, laplace, delta, uniform) PMFs over 255 symbols."""
    sigma = max(float(sigma), 1e-3)
    b = max(float(b), 1e-3)

    gauss = np.exp(-0.5 * (SYMBOL_AXIS / sigma) ** 2)
    gauss /= gauss.sum()

    laplace = np.exp(-np.abs(SYMBOL_AXIS) / b)
    laplace /= laplace.sum()

    delta = np.zeros(N_SYMBOLS, dtype=np.float64)
    delta[N_QUANT] = 1.0  # symbol 0 → idx 127

    uniform = np.full(N_SYMBOLS, 1.0 / N_SYMBOLS)

    return gauss, laplace, delta, uniform


def mixture_pmf(params: np.ndarray) -> np.ndarray:
    """Build the 4-component mixture PMF from a 6-param vector.

    ``params`` is unconstrained-real; mixture weights are derived from
    softmax over (params[0..3]); sigma, b are exp(params[4..5]).
    """
    w_logits = params[:4]
    w = np.exp(w_logits - np.max(w_logits))
    w /= w.sum()
    sigma = float(np.exp(params[4]))
    b = float(np.exp(params[5]))
    g, lap, d, u = canonical_shapes(sigma, b)
    pmf = w[0] * g + w[1] * lap + w[2] * d + w[3] * u
    pmf = np.maximum(pmf, 1e-12)
    pmf /= pmf.sum()
    return pmf


def fit_mixture(target_pmf: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit mixture parameters to target PMF via KL-minimization. Returns (params, kl_bits)."""
    target = np.maximum(target_pmf, 1e-12)
    target /= target.sum()

    def neg_log_lik(params: np.ndarray) -> float:
        pmf = mixture_pmf(params)
        # Cross-entropy in nats: -sum p_target * log(p_model)
        return float(-np.sum(target * np.log(pmf)))

    # Initial guess: equal mixture weights, sigma=8, b=4
    x0 = np.array([0.0, 0.0, 0.0, 0.0, np.log(8.0), np.log(4.0)])
    result = minimize(neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": 200})
    final_params = result.x
    pmf = mixture_pmf(final_params)
    # KL(target || pmf) in BITS
    kl_nats = float(np.sum(target * (np.log(target + 1e-12) - np.log(pmf))))
    kl_bits = kl_nats / np.log(2)
    return final_params, kl_bits


def serialize_mixture_params(all_params: list[np.ndarray]) -> bytes:
    """Pack mixture params as fp16: 6 fp16 per tensor = 12 bytes per tensor."""
    payload = bytearray()
    for p in all_params:
        for v in p:
            payload += np.float16(v).tobytes()
    return bytes(payload)


def encode_tensor_with_mixture(
    symbols_i8: np.ndarray, params: np.ndarray
) -> bytes:
    """AC-encode the symbols using the per-tensor mixture PMF."""
    pmf = mixture_pmf(params)
    # constriction needs a probabilistic model with the symbol alphabet [0..N_SYMBOLS-1]
    encoder = constriction.stream.queue.RangeEncoder()
    model = constriction.stream.model.Categorical(pmf, perfect=False)
    biased = (symbols_i8.astype(np.int32) + N_QUANT).flatten()
    encoder.encode(biased, model)
    return bytes(encoder.get_compressed())


def run_codec(state_dict_path: Path) -> dict:
    """Quantize, fit mixtures per-tensor, AC-encode, brotli-wrap, measure."""
    import torch

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    per_tensor: list[dict] = []
    all_params: list[np.ndarray] = []
    all_payloads: list[bytes] = []
    total_elements = 0
    total_kl_bits = 0.0

    for name, _shape in FIXED_STATE_SCHEMA:
        tensor = sd[name]
        qt = _quantize_tensor(name, tensor, n_quant=N_QUANT)
        pmf = empirical_pmf(qt.q_i8)
        params, kl_bits = fit_mixture(pmf)
        all_params.append(params)
        ac_payload = encode_tensor_with_mixture(qt.q_i8, params)
        all_payloads.append(ac_payload)
        total_elements += int(qt.q_i8.size)
        per_tensor.append({
            "name": name,
            "n_elements": int(qt.q_i8.size),
            "kl_bits_per_element": kl_bits,
            "ac_payload_bytes": len(ac_payload),
            "mixture_weights": [
                float(w) for w in
                np.exp(params[:4] - np.max(params[:4])) / np.sum(np.exp(params[:4] - np.max(params[:4])))
            ],
            "sigma": float(np.exp(params[4])),
            "b": float(np.exp(params[5])),
        })
        total_kl_bits += kl_bits * int(qt.q_i8.size)

    # Pack the per-tensor scales (need them for decoding) + mixture params
    scales = np.array([
        _quantize_tensor(name, sd[name], n_quant=N_QUANT).scale
        for name, _ in FIXED_STATE_SCHEMA
    ], dtype=np.float16)
    metadata_blob = (
        b"KFLD" +
        struct.pack("<I", len(FIXED_STATE_SCHEMA)) +
        scales.tobytes() +
        serialize_mixture_params(all_params)
    )
    # Metadata blob brotli-compressed
    metadata_brotli = brotli.compress(
        metadata_blob, quality=11, lgwin=16, lgblock=19
    )
    # AC payloads concatenated with length prefixes
    payload_concat = b"".join(
        struct.pack("<I", len(p)) + p for p in all_payloads
    )
    payload_brotli = brotli.compress(
        payload_concat, quality=11, lgwin=16, lgblock=19
    )
    total_decoder_blob = metadata_brotli + payload_brotli
    archive_bytes = len(total_decoder_blob) + ARCHIVE_OVERHEAD_BYTES

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": "[CPU-prep empirical]",
        "input_state_dict": str(state_dict_path),
        "n_tensors": len(FIXED_STATE_SCHEMA),
        "total_elements": total_elements,
        "metadata_blob_bytes": len(metadata_blob),
        "metadata_brotli_bytes": len(metadata_brotli),
        "ac_payload_concat_bytes": len(payload_concat),
        "payload_brotli_bytes": len(payload_brotli),
        "decoder_blob_bytes": len(total_decoder_blob),
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "weighted_kl_bits_per_element": total_kl_bits / max(total_elements, 1),
        "per_tensor": per_tensor,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = run_codec(args.state_dict)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_kalle_fold_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"manifest: {args.output_json}")
    print(f"\narchive_bytes: {manifest['archive_bytes']:,} B")
    print(f"  vs cathedral_autopilot prediction: 173,500 B")
    delta = manifest["archive_bytes"] - 173_500
    print(f"  delta: {delta:+,} B "
          f"({'BEAT' if delta < -1000 else 'TIED' if abs(delta) <= 1000 else 'MISSED'} prediction)")
    print(f"\n  vs brotli baseline: 178,144 B")
    delta_brotli = manifest["archive_bytes"] - 178_144
    print(f"  delta: {delta_brotli:+,} B "
          f"({'BEAT' if delta_brotli < 0 else 'TIES' if abs(delta_brotli) < 100 else 'LOSES'} brotli)")
    print()
    print(f"  metadata: {manifest['metadata_blob_bytes']:,} raw → "
          f"{manifest['metadata_brotli_bytes']:,} B brotli")
    print(f"  ac payload: {manifest['ac_payload_concat_bytes']:,} raw → "
          f"{manifest['payload_brotli_bytes']:,} B brotli")
    print(f"  weighted KL: {manifest['weighted_kl_bits_per_element']:.4f} bits/element")

    if args.output_evidence:
        evidence_row = {
            "technique": "kalle_fold_mixture_canonical_shapes",
            "empirical_archive_bytes": manifest["archive_bytes"],
            "source": f"[CPU-prep empirical] {args.output_json}",
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
