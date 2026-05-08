#!/usr/bin/env python3
"""PR101 kalle-fold 8-component HIERARCHICAL mixture codec — reactivation
of the kalle_fold_mixture_canonical_shapes catalog row per the
2026-05-07 adversarial audit (memo
``feedback_adversarial_audit_4_falsifications_DEFERRED_not_killed_20260507.md``).

The first kalle-fold codec (4-component) lost to brotli by +27,819 B
because the mixture had insufficient capacity to fit PR101's 28 distinct
PMFs. This reactivation tests the audit's first reactivation criterion:
8-component hierarchical mixture.

Architecture:
  - 8 canonical shapes per tensor:
    Gaussian(sigma_1), Gaussian(sigma_2), Laplace(b_1), Laplace(b_2),
    DeltaSpike, Uniform, Cauchy(gamma_1), Cauchy(gamma_2)
  - Per-tensor: 8 mixture weights + 6 scale params (2 each for the 3
    parametric shapes). Total: 14 fp16 = 28 B per tensor.
  - 28 tensors * 28 B = 784 B mixture metadata
  - Per-tensor scale fp16 = 56 B more
  - Total metadata: ~840 B (vs 4-component's 400 B)

If the empirical entropy gap (KL divergence to true PMF) drops enough,
the higher-capacity model wins. If not, kalle_fold lane stays DEFERRED
with one more criterion exhausted.

CLAUDE.md compliance: pure CPU + numpy + scipy + brotli + constriction;
no scorer load; no contest score claims; output tagged
[CPU-prep empirical reactivation].
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
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

TOOL_NAME = "tools/pr101_kalle_fold_8comp_hierarchical_codec.py"
SCHEMA_VERSION = "pr101_kalle_fold_8comp_hierarchical_codec.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_SYMBOLS = 2 * N_QUANT + 1  # 255
SYMBOL_AXIS = np.arange(-N_QUANT, N_QUANT + 1, dtype=np.float64)
EVIDENCE_GRADE = "[CPU-prep empirical reactivation]"
EVIDENCE_SEMANTICS = "cpu_kalle_fold_8comp_byte_anchor_no_decoder_no_score"
DISPATCH_BLOCKERS = (
    "hierarchical_mixture_decoder_not_wired_into_runtime_packet",
    "no_archive_substitution_performed",
    "no_decode_roundtrip_fixture_for_full_packet",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
)
N_COMPONENTS = 8


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


def empirical_pmf(symbols_i8: np.ndarray) -> np.ndarray:
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


def canonical_shapes_8(s1: float, s2: float, b1: float, b2: float, c1: float, c2: float) -> list[np.ndarray]:
    """Return 8 canonical PMFs over 255 symbols.

    Components:
      0: Gaussian(0, sigma=s1)  — narrow Gaussian
      1: Gaussian(0, sigma=s2)  — wide Gaussian
      2: Laplace(0, b=b1)        — narrow Laplace
      3: Laplace(0, b=b2)        — wide Laplace
      4: DeltaSpike at 0
      5: Uniform [-127, 127]
      6: Cauchy(0, gamma=c1)     — narrow Cauchy (heavier tails than Gaussian)
      7: Cauchy(0, gamma=c2)     — wide Cauchy
    """
    s1, s2 = max(s1, 1e-3), max(s2, 1e-3)
    b1, b2 = max(b1, 1e-3), max(b2, 1e-3)
    c1, c2 = max(c1, 1e-3), max(c2, 1e-3)

    g1 = np.exp(-0.5 * (SYMBOL_AXIS / s1) ** 2)
    g1 /= g1.sum()
    g2 = np.exp(-0.5 * (SYMBOL_AXIS / s2) ** 2)
    g2 /= g2.sum()
    l1 = np.exp(-np.abs(SYMBOL_AXIS) / b1)
    l1 /= l1.sum()
    l2 = np.exp(-np.abs(SYMBOL_AXIS) / b2)
    l2 /= l2.sum()
    d = np.zeros(N_SYMBOLS, dtype=np.float64)
    d[N_QUANT] = 1.0
    u = np.full(N_SYMBOLS, 1.0 / N_SYMBOLS)
    cy1 = 1.0 / (np.pi * c1 * (1.0 + (SYMBOL_AXIS / c1) ** 2))
    cy1 /= cy1.sum()
    cy2 = 1.0 / (np.pi * c2 * (1.0 + (SYMBOL_AXIS / c2) ** 2))
    cy2 /= cy2.sum()
    return [g1, g2, l1, l2, d, u, cy1, cy2]


def mixture_pmf_8(params: np.ndarray) -> np.ndarray:
    """Build 8-component mixture from a 14-param vector.

    params[0:8]   = mixture weight logits (softmax)
    params[8:14]  = log-scales for the 6 parametric components (s1,s2,b1,b2,c1,c2)
    """
    w_logits = params[:8]
    w = np.exp(w_logits - np.max(w_logits))
    w /= w.sum()
    s1 = float(np.exp(params[8]))
    s2 = float(np.exp(params[9]))
    b1 = float(np.exp(params[10]))
    b2 = float(np.exp(params[11]))
    c1 = float(np.exp(params[12]))
    c2 = float(np.exp(params[13]))
    shapes = canonical_shapes_8(s1, s2, b1, b2, c1, c2)
    pmf = sum(w[i] * shapes[i] for i in range(8))
    pmf = np.maximum(pmf, 1e-12)
    pmf /= pmf.sum()
    return pmf


def fit_mixture_8(target_pmf: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit 8-comp mixture via L-BFGS-B with multiple inits to escape local optima."""
    target = np.maximum(target_pmf, 1e-12)
    target /= target.sum()

    def neg_log_lik(params: np.ndarray) -> float:
        pmf = mixture_pmf_8(params)
        return float(-np.sum(target * np.log(pmf)))

    # Multi-start: try 3 inits with different scale priors, keep the best
    inits = [
        # narrow / wide Gaussian + narrow / wide Laplace + narrow/wide Cauchy
        np.array([0.0]*8 + [np.log(2.0), np.log(16.0), np.log(1.0), np.log(8.0), np.log(1.5), np.log(12.0)]),
        np.array([0.0]*8 + [np.log(4.0), np.log(32.0), np.log(2.0), np.log(16.0), np.log(3.0), np.log(24.0)]),
        np.array([0.0]*8 + [np.log(1.0), np.log(8.0), np.log(0.5), np.log(4.0), np.log(0.8), np.log(6.0)]),
    ]
    best_params = inits[0]
    best_loss = float("inf")
    for x0 in inits:
        result = minimize(neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": 300})
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_params = result.x

    pmf = mixture_pmf_8(best_params)
    kl_nats = float(np.sum(target * (np.log(target + 1e-12) - np.log(pmf))))
    kl_bits = kl_nats / np.log(2)
    return best_params, kl_bits


def serialize_mixture_params(all_params: list[np.ndarray]) -> bytes:
    payload = bytearray()
    for p in all_params:
        for v in p:
            payload += np.float16(v).tobytes()
    return bytes(payload)


def encode_tensor_with_mixture(symbols_i8: np.ndarray, params: np.ndarray) -> bytes:
    pmf = mixture_pmf_8(params)
    encoder = constriction.stream.queue.RangeEncoder()
    model = constriction.stream.model.Categorical(pmf, perfect=False)
    biased = (symbols_i8.astype(np.int32) + N_QUANT).flatten()
    encoder.encode(biased, model)
    return bytes(encoder.get_compressed())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_codec(state_dict_path: Path) -> dict:
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    per_tensor: list[dict] = []
    all_params: list[np.ndarray] = []
    all_payloads: list[bytes] = []
    total_elements = 0
    total_kl_bits = 0.0
    scales: list[float] = []

    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        scales.append(float(qt.scale))
        pmf = empirical_pmf(qt.q_i8)
        params, kl_bits = fit_mixture_8(pmf)
        all_params.append(params)
        ac_payload = encode_tensor_with_mixture(qt.q_i8, params)
        all_payloads.append(ac_payload)
        n = int(qt.q_i8.size)
        total_elements += n
        total_kl_bits += kl_bits * n
        # Resolved mixture weights for forensic
        w_logits = params[:8]
        w = np.exp(w_logits - np.max(w_logits))
        w /= w.sum()
        per_tensor.append({
            "name": name,
            "n_elements": n,
            "kl_bits_per_element": kl_bits,
            "ac_payload_bytes": len(ac_payload),
            "mixture_weights": [float(x) for x in w.tolist()],
            "sigmas": [float(np.exp(params[8])), float(np.exp(params[9]))],
            "bs": [float(np.exp(params[10])), float(np.exp(params[11]))],
            "cs": [float(np.exp(params[12])), float(np.exp(params[13]))],
        })

    scales_arr = np.array(scales, dtype=np.float16)
    metadata_blob = (
        b"KFL8" +
        struct.pack("<I", len(FIXED_STATE_SCHEMA)) +
        scales_arr.tobytes() +
        serialize_mixture_params(all_params)
    )
    metadata_brotli = brotli.compress(metadata_blob, quality=11, lgwin=16, lgblock=19)
    payload_concat = b"".join(struct.pack("<I", len(p)) + p for p in all_payloads)
    payload_brotli = brotli.compress(payload_concat, quality=11, lgwin=16, lgblock=19)
    decoder_blob_bytes = len(metadata_brotli) + len(payload_brotli)
    archive_bytes = decoder_blob_bytes + ARCHIVE_OVERHEAD_BYTES

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        **proxy_evidence_contract(),
        "input_state_dict": str(state_dict_path),
        "input_state_dict_sha256": sha256_file(state_dict_path),
        "n_components": N_COMPONENTS,
        "n_tensors": len(FIXED_STATE_SCHEMA),
        "total_elements": total_elements,
        "metadata_blob_bytes": len(metadata_blob),
        "metadata_brotli_bytes": len(metadata_brotli),
        "ac_payload_concat_bytes": len(payload_concat),
        "payload_brotli_bytes": len(payload_brotli),
        "decoder_blob_bytes": decoder_blob_bytes,
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
        out_dir = REPO_ROOT / f"reports/raw/pr101_kalle_fold_8comp_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"manifest: {args.output_json}\n")
    print(f"archive_bytes: {manifest['archive_bytes']:,} B")
    print("  vs prior 4-component result: 205,963 B")
    delta_4c = manifest["archive_bytes"] - 205_963
    print(f"  delta vs 4-comp: {delta_4c:+,} B")
    print("  vs brotli baseline: 178,144 B")
    delta_brotli = manifest["archive_bytes"] - 178_144
    print(f"  delta vs brotli: {delta_brotli:+,} B "
          f"({'BEAT' if delta_brotli < 0 else 'TIES' if abs(delta_brotli) < 100 else 'LOSES'} brotli)")
    print(f"\n  metadata: {manifest['metadata_blob_bytes']:,} raw -> {manifest['metadata_brotli_bytes']:,} B brotli")
    print(f"  ac payload: {manifest['ac_payload_concat_bytes']:,} -> {manifest['payload_brotli_bytes']:,} B brotli")
    print(f"  weighted KL: {manifest['weighted_kl_bits_per_element']:.4f} bits/element")

    if args.output_evidence:
        verdict_str = "BEAT-brotli" if delta_brotli < 0 else "still-DEFERRED-pending-research"
        evidence_row = {
            "technique": "kalle_fold_mixture_canonical_shapes",
            "empirical_archive_bytes": manifest["archive_bytes"],
            **proxy_evidence_contract(),
            "source": (
                f"[CPU-prep empirical reactivation] {args.output_json} "
                f"8-component hierarchical mixture (Gaussian x2 + Laplace x2 + delta + uniform + Cauchy x2)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "contest_dispatch_verdict": verdict_str,
            "reactivation_criteria_tested": ["8-component-hierarchical-mixture"],
            "reactivation_criteria_remaining": [
                "12-component-mixture",
                "tensor-class-conditioned-mixture",
                "sparse-dictionary-codebook",
            ],
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
