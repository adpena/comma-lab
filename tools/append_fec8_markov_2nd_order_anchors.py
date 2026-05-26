#!/usr/bin/env python3
"""Append missing FEC8 second-order empirical anchors idempotently.

# SPDX-License-Identifier: MIT
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.canonical_equations import (  # noqa: E402
    EmpiricalAnchor,
    update_equation_with_empirical_anchor,
)
from tac.provenance import build_provenance_for_macos_cpu_advisory  # noqa: E402

EQUATION_ID = "markov_context_selector_stream_compression_savings_v1"
REGISTRY_PATH = REPO_ROOT / ".omx/state/canonical_equations_registry.jsonl"
MEASUREMENT_UTC = "2026-05-26T19:32:03Z"
STAMP = "20260526T193203Z"
SUBAGENT = (
    "fec8-markov-2nd-order-p19-posenet-null-bucket-extension-pr111-candidate-"
    "20260526"
)
ARCHIVE_SHA = "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
EMPIRICAL_JSON = (
    ".omx/research/fec8_markov_2nd_order_p19_artifacts_20260526/"
    "fec8_markov_2nd_order_p19_bucket_extension_empirical.json"
)
PROMPTED_PREFIX = "fec8_markov_2nd_order_p19_bucket_prompted_"
TRUE_SECOND_ORDER_PREFIX = "fec8_markov_2nd_order_true_alternative_a_"
IMPLEMENTED_SECOND_ORDER_PREFIX = (
    "fec8_markov_static_second_order_arithmetic_implemented_"
)


def _provenance():
    return build_provenance_for_macos_cpu_advisory(
        archive_sha256=ARCHIVE_SHA,
        source_path="tools/measure_fec8_markov_2nd_order_p19_bucket_extension.py",
        captured_at_utc=MEASUREMENT_UTC,
    )


def _anchor_prefixes_present() -> set[str]:
    found: set[str] = set()
    if not REGISTRY_PATH.exists():
        return found
    for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("equation_id") != EQUATION_ID:
            continue
        payload = row.get("equation_payload")
        if not isinstance(payload, dict):
            continue
        for anchor in payload.get("empirical_anchors") or []:
            if not isinstance(anchor, dict):
                continue
            anchor_id = str(anchor.get("anchor_id") or "")
            for prefix in (
                PROMPTED_PREFIX,
                TRUE_SECOND_ORDER_PREFIX,
                IMPLEMENTED_SECOND_ORDER_PREFIX,
            ):
                if anchor_id.startswith(prefix):
                    found.add(prefix)
    return found


def _prompted_anchor() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=f"{PROMPTED_PREFIX}{STAMP}",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "in_domain_context": (
                "pr110_fec6_k16_selector_1st_order_markov_plus_"
                "deterministic_from_symbol_bucket_extension_FALSIFIED"
            ),
            "h_marginal_bits_per_pair": 3.2116,
            "h_first_order_markov_bits_per_pair": 2.9402,
            "h_first_order_plus_bucket_bits_per_pair": 1.9603,
            "k_palette": 16,
            "k_contexts_with_bucket_partition": 31,
            "n_pairs": 600,
            "n_pairs_in_bucket": 298,
            "bucket_membership_mode_ids": [0, 1, 2],
            "bucket_membership_mode_names": [
                "none",
                "frame0_blue_chroma_amp_1",
                "frame0_blue_chroma_amp_3",
            ],
            "live_stream_archive_sha256": ARCHIVE_SHA,
            "transition_table_kind": (
                "static_per_context_huffman_plus_1_bit_per_pair_bucket_flag"
            ),
            "structural_finding": (
                "Bucket is deterministic-from-symbol. The operational encoder "
                "must transmit bucket membership before symbol decode; that "
                "flag wire cost exceeds per-context Huffman savings on this "
                "stream by +4 to +8 bytes versus FEC8 1st-order Huffman."
            ),
        },
        predicted_output={
            "savings_bytes_wire_vs_fec8_1st_order_huffman": 5.0,
        },
        empirical_output={
            "savings_bytes_wire_vs_fec8_1st_order_huffman_raw_flag": -4.0,
            "savings_bytes_wire_vs_fec8_1st_order_huffman_brotli_flag": -8.0,
            "savings_bytes_wire_vs_fec6_baseline_raw_flag": 13.0,
            "savings_bytes_wire_vs_fec6_baseline_brotli_flag": 9.0,
            "verdict_per_catalog_307": "IMPLEMENTATION_LEVEL_FALSIFICATION",
        },
        residual=12.0,
        source_artifact=EMPIRICAL_JSON,
        measurement_method=(
            "fec8_per_context_huffman_with_p19_bucket_flag_stream_apples_to_"
            "apples_huffman_comparison_against_fec8_1st_order_huffman"
        ),
        provenance=_provenance(),
    )


def _true_second_order_anchor() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=f"{TRUE_SECOND_ORDER_PREFIX}{STAMP}",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "in_domain_context": (
                "pr110_fec6_k16_selector_true_2nd_order_markov_huffman_"
                "shared_prior_table_in_source_DIRECTIONAL_WIN"
            ),
            "h_marginal_bits_per_pair": 3.2116,
            "h_first_order_markov_bits_per_pair": 2.9402,
            "h_second_order_true_markov_bits_per_pair": 1.9788,
            "k_palette": 16,
            "k_contexts_observed_in_600_pair_stream": 130,
            "k_contexts_theoretical_max": 256,
            "n_pairs": 600,
            "live_stream_archive_sha256": ARCHIVE_SHA,
            "transition_table_kind": (
                "static_per_context_huffman_seeded_from_observed_2nd_order_"
                "transition_counts_zero_wire_overhead_under_wyner_ziv_pattern"
            ),
            "caveat_codebook_overhead": (
                "130 contexts x 16 cells is about 2KB source-text table "
                "embedded in source code, zero wire bytes under the shared-prior "
                "pattern."
            ),
            "caveat_generalization": (
                "table is fit to one 600-pair stream; new selector streams need "
                "re-measurement before promotion."
            ),
        },
        predicted_output={
            "savings_bytes_wire_vs_fec8_1st_order_huffman": 2.0,
        },
        empirical_output={
            "savings_bytes_wire_vs_fec8_1st_order_huffman": 66.0,
            "savings_bytes_wire_vs_fec6_baseline": 83.0,
            "verdict_per_catalog_307": "PARADIGM_VALIDATED_STACKING_EXTENSION",
        },
        residual=64.0,
        source_artifact=EMPIRICAL_JSON,
        measurement_method=(
            "fec8_per_context_huffman_with_true_2nd_order_markov_context_"
            "apples_to_apples_huffman_comparison_against_fec8_1st_order_huffman_"
            "excluding_codebook_source_text_overhead_per_wyner_ziv_shared_prior_pattern"
        ),
        provenance=_provenance(),
    )


def _implemented_second_order_anchor() -> EmpiricalAnchor:
    return EmpiricalAnchor(
        anchor_id=f"{IMPLEMENTED_SECOND_ORDER_PREFIX}{STAMP}",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "in_domain_context": (
                "pr110_fec6_k16_selector_static_2nd_order_markov_arithmetic_"
                "receiver_decode_only_IMPLEMENTED"
            ),
            "archive_sha256": ARCHIVE_SHA,
            "fec6_fixed_huffman_bytes": 249,
            "fec8_static_1st_order_arithmetic_bytes": 245,
            "fec8_static_2nd_order_arithmetic_bytes": 239,
            "n_pairs": 600,
            "k_palette": 16,
            "k_second_order_contexts_theoretical_max": 256,
            "wire_format": "FEC8 variant 0x0003",
            "receiver_roundtrip_verified": True,
            "source_prior_kind": "sparse_16x16x16_triple_counts_baked_in_source",
            "correction_of_prior_unpriced_estimator_anchor": (
                f"{TRUE_SECOND_ORDER_PREFIX}{STAMP}"
            ),
        },
        predicted_output={
            "savings_bytes_wire_vs_fec8_static_1st_order": 2.0,
        },
        empirical_output={
            "savings_bytes_wire_vs_fec8_static_1st_order": 6.0,
            "savings_bytes_wire_vs_fec6_fixed_huffman": 10.0,
            "absolute_wire_bytes": 239,
            "verdict_per_catalog_307": "IMPLEMENTED_DIRECTIONAL_WIN",
            "score_claim": False,
            "promotion_eligible": False,
            "modal_auth_eval_ready": False,
            "modal_auth_eval_blocker": (
                "needs byte-closed archive swap-in plus same-runtime inflate parity "
                "before exact CPU/CUDA auth eval dispatch"
            ),
        },
        residual=4.0,
        source_artifact=EMPIRICAL_JSON,
        measurement_method=(
            "implemented_fec8_variant_0003_static_second_order_arithmetic_roundtrip_"
            "and_live_selector_wire_byte_measurement"
        ),
        provenance=_provenance(),
    )


def _append_anchor(label: str, anchor: EmpiricalAnchor, notes: str) -> None:
    update_equation_with_empirical_anchor(
        equation_id=EQUATION_ID,
        anchor=anchor,
        agent="codex",
        subagent_id=SUBAGENT,
        notes=notes,
    )
    print(f"appended {label}: anchor_id={anchor.anchor_id}")


def main() -> int:
    present = _anchor_prefixes_present()
    if PROMPTED_PREFIX not in present:
        _append_anchor(
            "prompted P19 bucket falsification",
            _prompted_anchor(),
            "FEC8 P19 bucket extension negative anchor; prevents re-spending on "
            "deterministic-from-symbol bucket flags.",
        )
    if TRUE_SECOND_ORDER_PREFIX not in present:
        _append_anchor(
            "true second-order Markov win",
            _true_second_order_anchor(),
            "FEC8 true second-order selector-context positive anchor; routes "
            "follow-up to executable P14 static-prior coder.",
        )
    if IMPLEMENTED_SECOND_ORDER_PREFIX not in present:
        _append_anchor(
            "implemented static second-order arithmetic win",
            _implemented_second_order_anchor(),
            "Implemented FEC8 variant 0x0003 receiver roundtrip and live selector "
            "wire bytes; corrects earlier unpriced Huffman estimator into "
            "promotion-gated arithmetic-codec evidence.",
        )
    after = _anchor_prefixes_present()
    missing = {
        PROMPTED_PREFIX,
        TRUE_SECOND_ORDER_PREFIX,
        IMPLEMENTED_SECOND_ORDER_PREFIX,
    } - after
    if missing:
        print(f"missing expected anchor prefixes: {sorted(missing)}", file=sys.stderr)
        return 1
    if present == after:
        print("FEC8 second-order anchors already present; no registry append needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
