#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ATW V2-1 Faiss-IVF-PQ V4 hand-rolled probe per Symposium #1 op-routable #4.

V4 fills the Dykstra-feasibility transition zone between V2 (k_topk=8, MI=2.46)
and V3 (k_topk=1, MI=0.12) with the canonical mid-point (M=2, ksub=128,
top-k=3) targeting predicted ~3KB byte cost at bias-corrected MI ~1.0-1.8
bits/symbol per Shannon's grand-council analysis.

[verified-against: .omx/research/council_per_substrate_symposium_v1_dense_faiss_ivf_pq_reactivation_20260518.md]
[verified-against: Catalog #313 probe-outcomes ledger; Catalog #319 deliverability proof builder]
[verified-against: Catalog #287 evidence tag discipline; Catalog #323 canonical Provenance]

Per CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden empirical-
claim-without-evidence-tag": V4 emits axis-labelled `[diagnostic-CPU]` results
ONLY; not a contest score; no promotion eligibility.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
this probe is research-only; it never dispatches paid provider work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tools", REPO_ROOT / "upstream"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

# Reuse all canonical primitives from the V1/V2/V3 disambiguator
from tools.probe_atw_v2_1_faiss_pq_disambiguator import (  # noqa: E402
    PqVariantSpec,
    collect_a1_region_softmaxes,
    compute_pq_mi_verdict,
    encode_variant_packets,
    load_a1_latent_bytes_for_probe,
    repo_rel,
    sha256_bytes,
    utc_now,
)


def v4_hand_rolled_spec() -> PqVariantSpec:
    """V4 hand-rolled probe spec per Symposium #1 op-routable #4.

    M=2, ksub=128, top-k=3 — the Dykstra-feasibility transition zone between
    V3 (k_topk=1, MI=0.12) and V2 (k_topk=8, MI=2.46). Predicted byte cost:
    600 * 3 * (log2(128) + log2(64))/8 + codebook ~ ~3KB.
    """
    return PqVariantSpec(
        variant_id="v4_hand_rolled_m2_ksub128_topk3",
        n_regions=16,
        grid_side=4,
        nlist=64,  # IVF coarse quantizer (same as V2/V3)
        m_subq=2,  # M=2 sub-quantizers per Symposium #1 spec
        nbits=7,  # ksub=128 → 7 bits per sub-quantizer
        top_k_regions=3,  # top-k=3 per Symposium #1 spec
    )


@dataclass(frozen=True)
class V4ProbeOutcome:
    """Structured V4 outcome for the probe-outcomes ledger + downstream consumers."""

    variant_id: str
    mutual_information_bits: float
    h_latent_given_side_info_bits_per_symbol: float
    h_latent_unconditional_bits_per_symbol: float
    actual_total_archive_contribution_bytes: int
    actual_rate_cost: float
    verdict: str
    meaningful_mi_threshold_bits: float
    num_unique_side_info_symbols: int
    num_side_info_symbols: int
    advancement_recommendation: str
    blockers: list[str]


def _advancement_recommendation(
    mi_bits: float,
    bytes_total: int,
    meaningful_threshold: float = 0.5,
) -> str:
    """Advancement decision per Symposium #1 verdict tree."""
    if mi_bits >= meaningful_threshold and bytes_total <= 5_000:
        return (
            "ADVANCE_TO_PHASE_2_SYMPOSIUM: V4 hand-rolled satisfies "
            "MEANINGFUL_CONDITIONING + budget; proceed to codec-loop probe."
        )
    if mi_bits < meaningful_threshold and bytes_total <= 5_000:
        return (
            "FALSIFY_PER_REGION_HISTOGRAM_FAMILY: V4 hand-rolled fails MI "
            "threshold within budget; PIVOT to Atick channel #1 (per-pixel "
            "softmax logits) or Channel #3 (pose-bin discretization) per "
            "Symposium #1 verdict tree (cargo-cult-PARTIAL falsification of "
            "the per-region histogram family at the <5KB budget)."
        )
    if mi_bits >= meaningful_threshold and bytes_total > 5_000:
        return (
            "DEFER_PENDING_BUDGET_REEXAMINATION: V4 hand-rolled clears MI "
            "but exceeds budget; per Yousfi/Fridrich grand-council lens, "
            "re-examine <2KB constraint."
        )
    return (
        "DEFER_PENDING_REVIEW: V4 hand-rolled fails BOTH MI AND budget; "
        "no immediate advancement path."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--latent-bytes", type=Path, default=None)
    parser.add_argument("--softmax-npy", action="append", default=[])
    parser.add_argument(
        "--softmax-provenance-json",
        type=Path,
        default=None,
        help="Optional JSON file documenting softmax provenance (per Catalog #287)",
    )
    args = parser.parse_args(argv)

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "results" / f"atw_v2_1_faiss_pq_v4_hand_rolled_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.json_out or (
        REPO_ROOT
        / ".omx"
        / "state"
        / f"atw_v2_1_faiss_ivf_pq_v4_hand_rolled_probe_{stamp}.json"
    )

    # Extract A1 latent stream + SegNet softmax arrays
    print(f"[v4-probe] extracting A1 latent stream (max_pairs={args.max_pairs})")
    latent_bytes, latent_provenance = load_a1_latent_bytes_for_probe()
    # Trim to max_pairs * symbols_per_pair if needed (load returns full A1)
    full_latent_pairs = latent_provenance.get("num_pairs", 600)
    if args.max_pairs < full_latent_pairs:
        symbols_per_pair_full = len(latent_bytes) // full_latent_pairs
        latent_stream = latent_bytes[: args.max_pairs * symbols_per_pair_full]
    else:
        latent_stream = latent_bytes

    print(f"[v4-probe] collecting SegNet softmax (max_pairs={args.max_pairs})")
    softmax_by_region_count, softmax_provenance = collect_a1_region_softmaxes(
        max_pairs=args.max_pairs,
        chunk_size=args.chunk_size,
    )

    spec = v4_hand_rolled_spec()
    softmax = softmax_by_region_count.get(spec.n_regions)
    if softmax is None:
        raise RuntimeError(f"V4 probe requires softmax for n_regions={spec.n_regions}")

    n_pairs = softmax.shape[0]
    symbols_per_pair = len(latent_stream) // n_pairs
    print(
        f"[v4-probe] V4 spec: variant={spec.variant_id}, n_regions={spec.n_regions}, "
        f"nlist={spec.nlist}, m_subq={spec.m_subq}, nbits={spec.nbits}, "
        f"top_k={spec.top_k_regions}"
    )
    print(f"[v4-probe] n_pairs={n_pairs}, symbols_per_pair={symbols_per_pair}")

    # Encode V4 packet
    print(f"[v4-probe] encoding V4 packet via Faiss-IVF-PQ canonical helper")
    encoded = encode_variant_packets(softmax, spec=spec)

    # Compute MI verdict
    print(f"[v4-probe] computing per-pair MI verdict (plugin estimator)")
    verdict = compute_pq_mi_verdict(
        latent_stream=latent_stream,
        per_pair_symbols=list(encoded["per_pair_symbols"]),
        symbols_per_pair=symbols_per_pair,
    )

    bytes_total = encoded["actual_total_archive_contribution_bytes"]
    mi_bits = verdict.mutual_information_bits

    blockers: list[str] = []
    if verdict.num_unique_side_info_symbols / n_pairs > 0.95:
        blockers.append("pq_side_info_high_cardinality_plugin_mi_upper_bound_only")
    if bytes_total > 5_000:
        blockers.append("v4_actual_pq_payload_exceeds_5kb_shippable_target")
    if verdict.verdict != "MEANINGFUL_CONDITIONING":
        blockers.append("v4_did_not_reach_meaningful_conditioning_threshold")

    outcome = V4ProbeOutcome(
        variant_id=spec.variant_id,
        mutual_information_bits=mi_bits,
        h_latent_given_side_info_bits_per_symbol=verdict.h_latent_given_side_info_bits_per_symbol,
        h_latent_unconditional_bits_per_symbol=verdict.h_latent_unconditional_bits_per_symbol,
        actual_total_archive_contribution_bytes=bytes_total,
        actual_rate_cost=encoded["actual_rate_cost"],
        verdict=verdict.verdict,
        meaningful_mi_threshold_bits=verdict.meaningful_mi_threshold_bits,
        num_unique_side_info_symbols=verdict.num_unique_side_info_symbols,
        num_side_info_symbols=verdict.num_side_info_symbols,
        advancement_recommendation=_advancement_recommendation(mi_bits, bytes_total),
        blockers=blockers,
    )

    # Save stream + codebook + decode shape for forensic record
    stream_path = output_dir / f"{spec.variant_id}_pq_stream.bin"
    codebook_path = output_dir / f"{spec.variant_id}_faiss_codebook.bin"
    stream_path.write_bytes(encoded.pop("_codeword_stream_raw"))
    codebook_path.write_bytes(encoded.pop("_codebook_blob_raw"))
    encoded.pop("_codeword_stream_brotli_raw", None)
    encoded.pop("_codebook_brotli_raw", None)

    payload = {
        "axis_label": "[diagnostic-CPU; ATW V2-1 V4 hand-rolled Faiss-IVF-PQ MI probe]",
        "v4_spec": {
            "variant_id": spec.variant_id,
            "n_regions": spec.n_regions,
            "grid_side": spec.grid_side,
            "nlist": spec.nlist,
            "m_subq": spec.m_subq,
            "nbits": spec.nbits,
            "ksub": 2 ** spec.nbits,
            "top_k_regions": spec.top_k_regions,
        },
        "outcome": asdict(outcome),
        "encoded_stats": {
            "actual_total_archive_contribution_bytes": encoded[
                "actual_total_archive_contribution_bytes"
            ],
            "actual_rate_cost": encoded["actual_rate_cost"],
            "actual_codebook_bytes": encoded["actual_codebook_bytes"],
            "actual_codeword_stream_bytes": encoded["actual_codeword_stream_bytes"],
            "brotli_codebook_bytes": encoded["brotli_codebook_bytes"],
            "brotli_codeword_stream_bytes": encoded["brotli_codeword_stream_bytes"],
            "code_size_bytes_per_selected_region": encoded[
                "code_size_bytes_per_selected_region"
            ],
        },
        "verdict_full": asdict(verdict),
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_paid_dispatch": False,
        "score_claim_valid": False,
        "evidence_grade": "diagnostic_cpu",
        "captured_at_utc": utc_now(),
        "v4_stream_path": repo_rel(stream_path),
        "v4_codebook_path": repo_rel(codebook_path),
        "symposium_anchor": ".omx/research/council_per_substrate_symposium_v1_dense_faiss_ivf_pq_reactivation_20260518.md",
        "predecessor_probe": ".omx/research/atw_v2_1_byte_closed_side_info_probe_20260518_codex.json",
        "n_pairs_probed": n_pairs,
        "symbols_per_pair": symbols_per_pair,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print()
    print("=" * 72)
    print(f"V4 HAND-ROLLED FAISS-IVF-PQ PROBE OUTCOME")
    print("=" * 72)
    print(f"  variant_id:       {spec.variant_id}")
    print(f"  MI (bits/symbol): {mi_bits:.4f} (threshold: {verdict.meaningful_mi_threshold_bits})")
    print(f"  H(X|Y) bits:      {verdict.h_latent_given_side_info_bits_per_symbol:.4f}")
    print(f"  H(X) bits:        {verdict.h_latent_unconditional_bits_per_symbol:.4f}")
    print(f"  bytes_total:      {bytes_total}")
    print(f"  rate_cost:        {encoded['actual_rate_cost']:.6f}")
    print(f"  verdict:          {verdict.verdict}")
    print(f"  unique fraction:  {verdict.num_unique_side_info_symbols / n_pairs:.4f}")
    print()
    print(f"  ADVANCEMENT: {outcome.advancement_recommendation}")
    print()
    if blockers:
        print(f"  BLOCKERS:")
        for b in blockers:
            print(f"    - {b}")
    print()
    print(f"  JSON: {repo_rel(json_out)}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
