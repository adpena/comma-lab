#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ATW V2-1 Faiss-IVF-PQ V1/V2/V3 disambiguator.

This is the missing $0 CPU gate named by the ATW V2-1 design memo. It measures
whether the proposed Faiss-PQ side-info channel carries enough information
about the A1 latent stream to justify any paid ATW V2-1 dispatch.

The probe is diagnostic only. It writes byte-closed packets, MI upper-bound
metrics, and explicit false-authority blockers. It never dispatches a provider
job and never claims a contest score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import struct
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tools", REPO_ROOT / "upstream"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from tools.probe_latent_conditional_entropy_h_latent_given_scorer_class import (  # noqa: E402
    INDEPENDENCE_TOLERANCE_BITS,
    _count_joint,
    _count_symbols,
    _plug_in_entropy_bits,
)
from tac.optimization.faiss_ivf_pq_atw_channel import (  # noqa: E402
    CONTEST_RATE_NORMALIZER_BYTES,
    build_pq_codebook,
    decode_per_region_histogram,
    encode_per_region_histogram,
    estimate_pq_encoding_budget,
    serialize_codebook,
)

SEGNET_TARGET_H = 384
SEGNET_TARGET_W = 512
DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS = 0.5
HIGH_CARDINALITY_UNIQUE_FRACTION = 0.25

DEFAULT_STATE_JSON = (
    REPO_ROOT / ".omx" / "state" / "atw_v2_1_faiss_pq_disambiguator_probe.json"
)
DEFAULT_RESEARCH_JSON = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.json"
)
DEFAULT_RESEARCH_MD = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.md"
)


@dataclass(frozen=True)
class PqVariantSpec:
    variant_id: str
    n_regions: int
    grid_side: int
    nlist: int
    m_subq: int
    nbits: int
    top_k_regions: int | None


@dataclass(frozen=True)
class PqMiVerdict:
    verdict: str
    h_latent_unconditional_bits_per_symbol: float
    h_latent_given_side_info_bits_per_symbol: float
    mutual_information_bits: float
    wyner_ziv_gain_ceiling_fraction: float
    num_latent_symbols: int
    num_side_info_symbols: int
    num_unique_side_info_symbols: int
    meaningful_mi_threshold_bits: float
    independence_tolerance_bits: float


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "experiments" / "results" / f"atw_v2_1_faiss_pq_probe_{stamp}"


def replay_command() -> str:
    return " ".join([".venv/bin/python", repo_rel(Path(__file__)), *sys.argv[1:]])


def canonical_variant_specs() -> tuple[PqVariantSpec, ...]:
    """Return the three design-memo variants in dispatch-test order."""
    return (
        PqVariantSpec(
            variant_id="v3_pool_shared",
            n_regions=16,
            grid_side=4,
            nlist=64,
            m_subq=2,
            nbits=6,
            top_k_regions=1,
        ),
        PqVariantSpec(
            variant_id="v2_sparse_top_k",
            n_regions=16,
            grid_side=4,
            nlist=64,
            m_subq=2,
            nbits=6,
            top_k_regions=8,
        ),
        PqVariantSpec(
            variant_id="v1_dense",
            n_regions=256,
            grid_side=16,
            nlist=256,
            m_subq=4,
            nbits=8,
            top_k_regions=None,
        ),
    )


def _softmax_region_means(prob_chw: Any, *, grid_side: int) -> np.ndarray:
    """Average SegNet class probabilities in a grid of spatial regions."""
    if hasattr(prob_chw, "detach"):
        arr = prob_chw.detach().to("cpu").numpy()
    else:
        arr = np.asarray(prob_chw)
    if arr.ndim != 3:
        raise ValueError(f"prob_chw must have shape (C,H,W), got {tuple(arr.shape)}")
    c, h, w = [int(v) for v in arr.shape]
    if h % grid_side or w % grid_side:
        raise ValueError(f"shape {(h, w)} not divisible by grid_side={grid_side}")
    region_h = h // grid_side
    region_w = w // grid_side
    regions: list[np.ndarray] = []
    for gy in range(grid_side):
        for gx in range(grid_side):
            tile = arr[
                :,
                gy * region_h : (gy + 1) * region_h,
                gx * region_w : (gx + 1) * region_w,
            ]
            regions.append(tile.mean(axis=(1, 2)))
    out = np.stack(regions, axis=0).astype(np.float32)
    if out.shape != (grid_side * grid_side, c):
        raise ValueError(f"unexpected region output shape {out.shape}")
    return np.ascontiguousarray(out.astype(np.float32))


def _load_a1_decoder_and_latents() -> tuple[Any, Any, dict[str, Any]]:
    """Load A1 HNeRV decoder and latents from the byte-closed archive."""
    import torch

    a1_archive = REPO_ROOT / "submissions" / "a1" / "archive.zip"
    a1_src = REPO_ROOT / "submissions" / "a1" / "src"
    if not a1_archive.is_file():
        raise FileNotFoundError(f"A1 archive missing: {a1_archive}")
    if not a1_src.is_dir():
        raise FileNotFoundError(f"A1 source dir missing: {a1_src}")
    if str(a1_src) not in sys.path:
        sys.path.insert(0, str(a1_src))

    with zipfile.ZipFile(a1_archive, "r") as zf:
        archive_bytes = zf.read("x")
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    decoder_blob = archive_bytes[4:section_total]
    latent_blob_len = 15_387
    latent_blob = archive_bytes[section_total : section_total + latent_blob_len]
    sidecar_blob = archive_bytes[section_total + latent_blob_len :]

    from codec import apply_latent_sidecar, decode_decoder_compact, decode_latents_compact
    from model import HNeRVDecoder

    decoder_sd = decode_decoder_compact(decoder_blob)
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    decoder = HNeRVDecoder(
        latent_dim=28,
        base_channels=36,
        eval_size=(SEGNET_TARGET_H, SEGNET_TARGET_W),
    )
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    provenance = {
        "a1_archive_path": repo_rel(a1_archive),
        "a1_inner_member": "x",
        "a1_inner_member_sha256": sha256_bytes(archive_bytes),
        "decoder_blob_sha256": sha256_bytes(decoder_blob),
        "latent_blob_sha256": sha256_bytes(latent_blob),
        "sidecar_blob_sha256": sha256_bytes(sidecar_blob),
        "latent_shape": [int(v) for v in latents.shape],
    }
    return decoder, latents, provenance


def load_a1_latent_bytes_for_probe() -> tuple[bytes, dict[str, Any]]:
    """Load A1 latents and quantize to uint8 without importing helper modules."""
    import torch

    _decoder, latents, provenance = _load_a1_decoder_and_latents()
    values = latents.detach().to("cpu", dtype=torch.float32)
    lo = float(values.min().item())
    hi = float(values.max().item())
    value_range = max(hi - lo, 1e-9)
    quantized = ((values - lo) / value_range * 255.0).round().clamp(0, 255)
    latent_bytes = bytes(quantized.to(torch.uint8).flatten().tolist())
    return latent_bytes, {
        **provenance,
        "latent_quantizer": {"min": lo, "max": hi, "range": value_range},
        "latent_u8_sha256": sha256_bytes(latent_bytes),
        "latent_u8_bytes": len(latent_bytes),
    }


def collect_a1_region_softmaxes(
    *,
    max_pairs: int = 600,
    chunk_size: int = 8,
    grid_sides: tuple[int, ...] = (4, 16),
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Render A1 pairs, run canonical SegNet on frame_1, and region-average softmax."""
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    from tac.scorer import load_default_scorers
    import torch

    decoder, latents, a1_provenance = _load_a1_decoder_and_latents()
    device = torch.device("cpu")
    decoder = decoder.to(device)
    _posenet, segnet = load_default_scorers(REPO_ROOT / "upstream", device=device)
    segnet.eval()

    n_pairs = min(max_pairs, int(latents.shape[0]))
    latents = latents[:n_pairs].to(device)
    by_grid: dict[int, list[np.ndarray]] = {side: [] for side in grid_sides}
    t0 = time.monotonic()
    with torch.no_grad():
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            rendered = decoder(latents[start:end]).clamp(0, 255)
            for i in range(rendered.shape[0]):
                pair = rendered[i].unsqueeze(0)
                preprocessed = segnet.preprocess_input(pair)
                logits = segnet(preprocessed)
                probs = torch.softmax(logits.squeeze(0), dim=0)
                for side in grid_sides:
                    by_grid[side].append(_softmax_region_means(probs, grid_side=side))
            elapsed = time.monotonic() - t0
            rate = end / elapsed if elapsed > 0 else 0.0
            eta = (n_pairs - end) / rate if rate > 0 else 0.0
            print(
                f"[atw-pq-probe] softmax pair {end}/{n_pairs} "
                f"rate={rate:.2f} pair/s eta={eta:.0f}s",
                flush=True,
            )

    arrays = {
        side * side: np.stack(values, axis=0).astype(np.float32)
        for side, values in by_grid.items()
    }
    provenance = {
        **a1_provenance,
        "num_pairs": n_pairs,
        "grid_sides": list(grid_sides),
        "elapsed_seconds_softmax_collection": float(time.monotonic() - t0),
        "segnet_source": "tac.scorer.load_default_scorers(upstream, cpu)",
        "softmax_scope": "A1 HNeRV-rendered frame_1 via SegNet canonical preprocess_input",
    }
    return arrays, provenance


def _entropy_deficit(region_softmax: np.ndarray) -> np.ndarray:
    clipped = np.clip(region_softmax.astype(np.float64), 1e-12, 1.0)
    entropy = -(clipped * np.log2(clipped)).sum(axis=1)
    return math.log2(region_softmax.shape[1]) - entropy


def select_region_indices(region_softmax: np.ndarray, *, top_k: int | None) -> list[int]:
    """Pick the most non-uniform regions, sorted for deterministic packet layout."""
    n_regions = int(region_softmax.shape[0])
    if top_k is None:
        return list(range(n_regions))
    if top_k <= 0 or top_k > n_regions:
        raise ValueError(f"top_k={top_k} invalid for n_regions={n_regions}")
    scores = _entropy_deficit(region_softmax)
    ranked = sorted(range(n_regions), key=lambda idx: (-float(scores[idx]), idx))
    return sorted(ranked[:top_k])


def _stable_symbol(raw: bytes) -> int:
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little", signed=False)


def encode_variant_packets(
    softmax_by_pair: np.ndarray,
    *,
    spec: PqVariantSpec,
) -> dict[str, Any]:
    """Train codebook and encode each pair into deterministic byte packets."""
    if softmax_by_pair.ndim != 3 or softmax_by_pair.shape[1:] != (spec.n_regions, 5):
        raise ValueError(
            f"softmax_by_pair must have shape (N,{spec.n_regions},5), "
            f"got {softmax_by_pair.shape}"
        )
    training_vectors = np.ascontiguousarray(
        softmax_by_pair.reshape(-1, softmax_by_pair.shape[-1]).astype(np.float32)
    )
    codebook = build_pq_codebook(
        training_vectors,
        nlist=spec.nlist,
        m_subq=spec.m_subq,
        nbits=spec.nbits,
        seed=42,
    )
    codebook_blob = serialize_codebook(codebook)
    code_size = int(codebook.sa_code_size())

    stream = bytearray()
    symbols: list[int] = []
    region_indices_by_pair: list[list[int]] = []
    first_decode_shape: list[int] | None = None
    for pair_softmax in softmax_by_pair:
        selected = select_region_indices(pair_softmax, top_k=spec.top_k_regions)
        selected_softmax = np.ascontiguousarray(pair_softmax[selected].astype(np.float32))
        encoded = encode_per_region_histogram(selected_softmax, codebook)
        decoded = decode_per_region_histogram(
            encoded,
            codebook,
            n_regions=len(selected),
            softmax_dim=5,
        )
        if first_decode_shape is None:
            first_decode_shape = [int(v) for v in decoded.shape]
        packet = bytearray()
        if spec.top_k_regions is not None:
            for offset, region_idx in enumerate(selected):
                code_start = offset * code_size
                code_end = code_start + code_size
                packet.append(int(region_idx))
                packet.extend(encoded[code_start:code_end])
        else:
            packet.extend(encoded)
        raw_packet = bytes(packet)
        stream.extend(raw_packet)
        symbols.append(_stable_symbol(raw_packet))
        region_indices_by_pair.append([int(idx) for idx in selected])

    stream_bytes = bytes(stream)
    codebook_brotli = brotli.compress(codebook_blob, quality=11)
    stream_brotli = brotli.compress(stream_bytes, quality=11)
    budget = estimate_pq_encoding_budget(
        variant_id=spec.variant_id,
        n_regions=spec.n_regions,
        nlist=spec.nlist,
        m_subq=spec.m_subq,
        nbits=spec.nbits,
        top_k_regions=spec.top_k_regions,
        total_pairs=int(softmax_by_pair.shape[0]),
    )
    actual_total = len(codebook_brotli) + len(stream_brotli)
    return {
        "spec": asdict(spec),
        "budget_estimate": budget.as_dict(),
        "_codebook_blob_raw": codebook_blob,
        "_codeword_stream_raw": stream_bytes,
        "_codebook_brotli_raw": codebook_brotli,
        "_codeword_stream_brotli_raw": stream_brotli,
        "actual_codebook_bytes": len(codebook_blob),
        "actual_codebook_sha256": sha256_bytes(codebook_blob),
        "actual_codeword_stream_bytes": len(stream_bytes),
        "actual_codeword_stream_sha256": sha256_bytes(stream_bytes),
        "brotli_codebook_bytes": len(codebook_brotli),
        "brotli_codebook_sha256": sha256_bytes(codebook_brotli),
        "brotli_codeword_stream_bytes": len(stream_brotli),
        "brotli_codeword_stream_sha256": sha256_bytes(stream_brotli),
        "actual_total_archive_contribution_bytes": actual_total,
        "actual_rate_cost": 25.0 * actual_total / CONTEST_RATE_NORMALIZER_BYTES,
        "code_size_bytes_per_selected_region": code_size,
        "first_decode_shape": first_decode_shape,
        "per_pair_symbols": symbols,
        "per_pair_packet_unique_count": len(set(symbols)),
        "region_indices_by_pair_sha256": sha256_bytes(
            json.dumps(region_indices_by_pair, sort_keys=True).encode("utf-8")
        ),
    }


def compute_pq_mi_verdict(
    *,
    latent_stream: bytes,
    per_pair_symbols: list[int],
    symbols_per_pair: int,
    threshold: float = DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS,
) -> PqMiVerdict:
    if symbols_per_pair <= 0:
        raise ValueError("symbols_per_pair must be positive")
    if not per_pair_symbols:
        raise ValueError("per_pair_symbols must be non-empty")
    expanded: list[int] = []
    for symbol in per_pair_symbols:
        if int(symbol) < 0:
            raise ValueError("per_pair_symbols must be non-negative")
        expanded.extend([int(symbol)] * symbols_per_pair)
    if len(expanded) != len(latent_stream):
        raise ValueError(
            f"expanded side-info length {len(expanded)} != latent length {len(latent_stream)}"
        )
    latent_counts = _count_symbols(latent_stream)
    side_counts = _count_symbols(expanded)
    joint = _count_joint(latent_stream, expanded)
    h_latent = _plug_in_entropy_bits(latent_counts)
    total = sum(side_counts.values())
    h_cond = 0.0
    for symbol, per_symbol in joint.items():
        h_cond += (side_counts[symbol] / total) * _plug_in_entropy_bits(per_symbol)
    mi = max(h_latent - h_cond, 0.0)
    if mi <= INDEPENDENCE_TOLERANCE_BITS:
        verdict = "INDEPENDENT"
    elif mi >= threshold:
        verdict = "MEANINGFUL_CONDITIONING"
    else:
        verdict = "WEAK_CONDITIONING"
    return PqMiVerdict(
        verdict=verdict,
        h_latent_unconditional_bits_per_symbol=float(h_latent),
        h_latent_given_side_info_bits_per_symbol=float(h_cond),
        mutual_information_bits=float(mi),
        wyner_ziv_gain_ceiling_fraction=float(mi / h_latent if h_latent > 0 else 0.0),
        num_latent_symbols=len(latent_stream),
        num_side_info_symbols=len(expanded),
        num_unique_side_info_symbols=len(set(per_pair_symbols)),
        meaningful_mi_threshold_bits=threshold,
        independence_tolerance_bits=INDEPENDENCE_TOLERANCE_BITS,
    )


def build_probe_payload(
    *,
    latent_stream: bytes,
    softmax_by_region_count: dict[int, np.ndarray],
    output_dir: Path,
    softmax_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run V1/V2/V3 and return the fail-closed disambiguator payload."""
    if not latent_stream:
        raise ValueError("latent_stream must be non-empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_pairs = next(iter(softmax_by_region_count.values())).shape[0]
    if len(latent_stream) % n_pairs != 0:
        raise ValueError(f"latent length {len(latent_stream)} not divisible by n_pairs {n_pairs}")
    symbols_per_pair = len(latent_stream) // n_pairs

    variants: list[dict[str, Any]] = []
    for spec in canonical_variant_specs():
        softmax = softmax_by_region_count.get(spec.n_regions)
        if softmax is None:
            raise ValueError(f"missing softmax array for n_regions={spec.n_regions}")
        encoded = encode_variant_packets(softmax, spec=spec)
        verdict = compute_pq_mi_verdict(
            latent_stream=latent_stream,
            per_pair_symbols=list(encoded["per_pair_symbols"]),
            symbols_per_pair=symbols_per_pair,
        )
        unique_fraction = verdict.num_unique_side_info_symbols / n_pairs
        high_cardinality = unique_fraction > HIGH_CARDINALITY_UNIQUE_FRACTION
        blockers: list[str] = []
        if high_cardinality:
            blockers.append("pq_side_info_high_cardinality_plugin_mi_upper_bound_only")
        if encoded["actual_total_archive_contribution_bytes"] > 5_000:
            blockers.append("actual_pq_payload_exceeds_v3_shippable_5kb_target")
        if verdict.verdict != "MEANINGFUL_CONDITIONING":
            blockers.append("pq_variant_did_not_reach_meaningful_conditioning_threshold")

        stream_path = output_dir / f"atw_v2_1_{spec.variant_id}_pq_stream.bin"
        codebook_path = output_dir / f"atw_v2_1_{spec.variant_id}_faiss_codebook.bin"
        stream_brotli_path = output_dir / f"atw_v2_1_{spec.variant_id}_pq_stream.bin.br"
        codebook_brotli_path = output_dir / f"atw_v2_1_{spec.variant_id}_faiss_codebook.bin.br"
        stream_path.write_bytes(encoded.pop("_codeword_stream_raw"))
        codebook_path.write_bytes(encoded.pop("_codebook_blob_raw"))
        stream_brotli_path.write_bytes(encoded.pop("_codeword_stream_brotli_raw"))
        codebook_brotli_path.write_bytes(encoded.pop("_codebook_brotli_raw"))
        # The stream and codebook are rebuildable; durable JSON records hashes,
        # and the experiment directory keeps local forensic bytes for review.

        variants.append(
            {
                "variant_id": spec.variant_id,
                "verdict": asdict(verdict),
                "unique_fraction": unique_fraction,
                "high_cardinality_bias_guard_triggered": high_cardinality,
                "actual_total_archive_contribution_bytes": encoded[
                    "actual_total_archive_contribution_bytes"
                ],
                "actual_rate_cost": encoded["actual_rate_cost"],
                "budget_estimate": encoded["budget_estimate"],
                "actual_codebook_bytes": encoded["actual_codebook_bytes"],
                "actual_codebook_sha256": encoded["actual_codebook_sha256"],
                "actual_codeword_stream_bytes": encoded["actual_codeword_stream_bytes"],
                "actual_codeword_stream_sha256": encoded["actual_codeword_stream_sha256"],
                "brotli_codebook_bytes": encoded["brotli_codebook_bytes"],
                "brotli_codebook_sha256": encoded["brotli_codebook_sha256"],
                "brotli_codeword_stream_bytes": encoded["brotli_codeword_stream_bytes"],
                "brotli_codeword_stream_sha256": encoded["brotli_codeword_stream_sha256"],
                "codebook_path": repo_rel(codebook_path),
                "codeword_stream_path": repo_rel(stream_path),
                "brotli_codebook_path": repo_rel(codebook_brotli_path),
                "brotli_codeword_stream_path": repo_rel(stream_brotli_path),
                "code_size_bytes_per_selected_region": encoded[
                    "code_size_bytes_per_selected_region"
                ],
                "first_decode_shape": encoded["first_decode_shape"],
                "region_indices_by_pair_sha256": encoded["region_indices_by_pair_sha256"],
                "dispatch_blockers": blockers,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_paid_dispatch": False,
            }
        )

    meaningful_without_bias = [
        row
        for row in variants
        if row["verdict"]["verdict"] == "MEANINGFUL_CONDITIONING"
        and not row["high_cardinality_bias_guard_triggered"]
        and row["actual_total_archive_contribution_bytes"] <= 5_000
    ]
    weak_or_biased = [
        row
        for row in variants
        if row["verdict"]["verdict"] in {"MEANINGFUL_CONDITIONING", "WEAK_CONDITIONING"}
    ]
    if meaningful_without_bias:
        phase2_status = "pq_variant_meaningful_requires_new_d4_and_wave_n_plus_1_council"
        recommended_next_gate = "run_new_d4_probe_on_selected_pq_variant_before_dispatch"
    elif weak_or_biased:
        phase2_status = "pq_variants_not_dispatch_authority_upper_bound_or_weak"
        recommended_next_gate = "pivot_to_scorer_logit_compression_or_trained_atw_residual_probe"
    else:
        phase2_status = "pq_variants_independent_or_over_budget"
        recommended_next_gate = "defer_atw_v2_1_faiss_pq_and_pivot_channel_family"

    best = max(
        variants,
        key=lambda row: (
            float(row["verdict"]["mutual_information_bits"]),
            -int(row["actual_total_archive_contribution_bytes"]),
        ),
    )
    return {
        "schema": "atw_v2_1_faiss_pq_disambiguator_v1",
        "observed_at_utc": utc_now(),
        "command": replay_command(),
        "output_dir": repo_rel(output_dir),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "dispatch_attempted": False,
        "provider_spend_attempted": False,
        "evidence_grade": "diagnostic_cpu",
        "axis_label": "[diagnostic-CPU; ATW V2-1 Faiss-PQ side-info MI probe]",
        "num_pairs": n_pairs,
        "symbols_per_pair": symbols_per_pair,
        "meaningful_mi_threshold_bits": DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS,
        "high_cardinality_unique_fraction_limit": HIGH_CARDINALITY_UNIQUE_FRACTION,
        "variants": variants,
        "best_variant": {
            "variant_id": best["variant_id"],
            "verdict": best["verdict"]["verdict"],
            "mutual_information_bits": best["verdict"]["mutual_information_bits"],
            "actual_total_archive_contribution_bytes": best[
                "actual_total_archive_contribution_bytes"
            ],
            "high_cardinality_bias_guard_triggered": best[
                "high_cardinality_bias_guard_triggered"
            ],
        },
        "phase2_status": phase2_status,
        "recommended_next_gate": recommended_next_gate,
        "result_review_blockers": [
            "diagnostic_probe_not_score_claim",
            "faiss_pq_mi_is_upper_bound_until_trained_wz_head_and_new_d4_probe",
            "requires_paired_contest_cuda_cpu_harvest_before_promotion",
        ],
        "softmax_provenance": softmax_provenance or {},
    }


def missing_faiss_payload(*, output_dir: Path, error: BaseException) -> dict[str, Any]:
    return {
        "schema": "atw_v2_1_faiss_pq_disambiguator_v1",
        "observed_at_utc": utc_now(),
        "command": replay_command(),
        "output_dir": repo_rel(output_dir),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "dispatch_attempted": False,
        "provider_spend_attempted": False,
        "evidence_grade": "dependency_blocked",
        "axis_label": "[diagnostic-CPU; ATW V2-1 Faiss-PQ dependency probe]",
        "phase2_status": "dependency_blocked_faiss_cpu_missing",
        "recommended_next_gate": "uv_pip_install_faiss_cpu_then_rerun_disambiguator",
        "result_review_blockers": [
            "faiss_cpu_dependency_missing",
            "v1_v2_v3_disambiguator_not_executed",
            "no_paid_dispatch_authority",
        ],
        "error_type": type(error).__name__,
        "error": str(error),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _parse_softmax_npy_args(items: list[str] | None) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--softmax-npy item must be REGIONS=PATH, got {item!r}")
        key, value = item.split("=", 1)
        out[int(key)] = Path(value)
    return out


def _run_faiss_worker(
    *,
    latent_bytes_path: Path,
    softmax_npy_by_region: dict[int, Path],
    softmax_provenance_json: Path,
    output_dir: Path,
    json_out: Path,
    research_json: Path,
    research_md: Path,
) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--faiss-worker",
        "--latent-bytes",
        str(latent_bytes_path),
        "--softmax-provenance-json",
        str(softmax_provenance_json),
        "--output-dir",
        str(output_dir),
        "--json-out",
        str(json_out),
        "--research-json",
        str(research_json),
        "--research-md",
        str(research_md),
    ]
    for n_regions, path in sorted(softmax_npy_by_region.items()):
        cmd.extend(["--softmax-npy", f"{n_regions}={path}"])
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ATW V2-1 Faiss-PQ Disambiguator",
        "",
        f"- observed_at_utc: `{payload['observed_at_utc']}`",
        f"- axis_label: `{payload['axis_label']}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- dispatch_attempted: `false`",
        "- provider_spend_attempted: `false`",
        f"- phase2_status: `{payload['phase2_status']}`",
        f"- recommended_next_gate: `{payload['recommended_next_gate']}`",
        "",
    ]
    variants = payload.get("variants") or []
    if variants:
        lines.extend(
            [
                "## Variant Results",
                "",
                "| Variant | Archive bytes (Brotli) | Rate cost | Unique frac | MI bits/symbol | Verdict | Bias guard | Blockers |",
                "|---|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for row in variants:
            verdict = row["verdict"]
            lines.append(
                "| {variant} | {bytes} | {rate:.6f} | {unique:.3f} | {mi:.12f} | {verdict} | {bias} | {blockers} |".format(
                    variant=row["variant_id"],
                    bytes=row["actual_total_archive_contribution_bytes"],
                    rate=row["actual_rate_cost"],
                    unique=row["unique_fraction"],
                    mi=verdict["mutual_information_bits"],
                    verdict=verdict["verdict"],
                    bias=str(row["high_cardinality_bias_guard_triggered"]).lower(),
                    blockers=", ".join(row["dispatch_blockers"]) or "none",
                )
            )
        best = payload["best_variant"]
        lines.extend(
            [
                "",
                "## Best Variant",
                "",
                (
                    f"`{best['variant_id']}` produced MI "
                    f"`{best['mutual_information_bits']:.12f}` bits/symbol with "
                    f"`{best['actual_total_archive_contribution_bytes']}` bytes. "
                    f"Bias guard triggered: `{str(best['high_cardinality_bias_guard_triggered']).lower()}`."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "## Dependency Blocker",
                "",
                f"- error_type: `{payload.get('error_type')}`",
                f"- error: `{payload.get('error')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## False-Authority Guard",
            "",
            "This diagnostic artifact is not a score claim and is not dispatch authority.",
            "A paid ATW V2-1 run still requires a selected channel, a new D4 probe,",
            "sextet council ratification, and paired contest CPU/CUDA harvest.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--research-json", type=Path, default=DEFAULT_RESEARCH_JSON)
    parser.add_argument("--research-md", type=Path, default=DEFAULT_RESEARCH_MD)
    parser.add_argument("--faiss-worker", action="store_true")
    parser.add_argument("--latent-bytes", type=Path, default=None)
    parser.add_argument("--softmax-npy", action="append", default=[])
    parser.add_argument("--softmax-provenance-json", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or default_output_dir()
    if args.faiss_worker:
        try:
            if args.latent_bytes is None:
                raise ValueError("--faiss-worker requires --latent-bytes")
            softmax_paths = _parse_softmax_npy_args(args.softmax_npy)
            if not softmax_paths:
                raise ValueError("--faiss-worker requires at least one --softmax-npy")
            latent_stream = args.latent_bytes.read_bytes()
            softmax_by_region_count = {
                n_regions: np.load(path) for n_regions, path in softmax_paths.items()
            }
            provenance = (
                json.loads(args.softmax_provenance_json.read_text(encoding="utf-8"))
                if args.softmax_provenance_json is not None
                and args.softmax_provenance_json.is_file()
                else {}
            )
            payload = build_probe_payload(
                latent_stream=latent_stream,
                softmax_by_region_count=softmax_by_region_count,
                output_dir=output_dir,
                softmax_provenance=provenance,
            )
            rc = 0
        except ImportError as exc:
            payload = missing_faiss_payload(output_dir=output_dir, error=exc)
            rc = 12
        write_json(args.json_out, payload)
        write_json(args.research_json, payload)
        args.research_md.parent.mkdir(parents=True, exist_ok=True)
        args.research_md.write_text(render_markdown(payload), encoding="utf-8")
        print(f"[atw-pq-probe] wrote {args.json_out}")
        print(f"[atw-pq-probe] phase2_status={payload['phase2_status']}")
        return rc

    try:
        latent_stream, latent_provenance = load_a1_latent_bytes_for_probe()
        softmax_by_region_count, softmax_provenance = collect_a1_region_softmaxes(
            max_pairs=args.max_pairs,
            chunk_size=args.chunk_size,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        n_pairs = next(iter(softmax_by_region_count.values())).shape[0]
        symbols_per_pair = len(latent_stream) // 600
        latent_path = output_dir / "a1_latents_u8.bin"
        latent_path.write_bytes(latent_stream[: n_pairs * symbols_per_pair])
        softmax_paths: dict[int, Path] = {}
        for n_regions, array in softmax_by_region_count.items():
            path = output_dir / f"segnet_region_softmax_{n_regions}.npy"
            np.save(path, array)
            softmax_paths[n_regions] = path
        provenance_path = output_dir / "softmax_collection_provenance.json"
        write_json(
            provenance_path,
            {
                **softmax_provenance,
                "latent_provenance": latent_provenance,
                "worker_boundary": "faiss_runs_in_subprocess_to_avoid_torch_faiss_openmp_conflict",
            },
        )
        return _run_faiss_worker(
            latent_bytes_path=latent_path,
            softmax_npy_by_region=softmax_paths,
            softmax_provenance_json=provenance_path,
            output_dir=output_dir,
            json_out=args.json_out,
            research_json=args.research_json,
            research_md=args.research_md,
        )
    except ImportError as exc:
        payload = missing_faiss_payload(output_dir=output_dir, error=exc)
        rc = 12

    write_json(args.json_out, payload)
    write_json(args.research_json, payload)
    args.research_md.parent.mkdir(parents=True, exist_ok=True)
    args.research_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[atw-pq-probe] wrote {args.json_out}")
    print(f"[atw-pq-probe] phase2_status={payload['phase2_status']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
