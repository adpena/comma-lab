#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute the archive surface recode queue planner.

Consumes the newest ``tools/plan_archive_surface_recode_queue.py`` output and
classifies every queue entry by canonical equation #26
(``procedural_codebook_from_seed_compression_savings_v1``) IN-DOMAIN vs
EXCLUDED vs UNCLASSIFIED contexts per OVERNIGHT-G TRIAGE Pick 7 spec.

The execution layer is observability-only per CLAUDE.md "Forbidden
empirical-claim-without-evidence-tag" + Catalog #287 + Catalog #323 canonical
Provenance: every queue entry is tagged with its closed-form predicted
archive-bytes-saved + canonical equation #26 classification + ready-to-paste
operator command. No score claims are made; no canonical equations registry
mutations happen here — RATIFY is reserved for empirical anchors.

Per Catalog #110 / #113 HISTORICAL_PROVENANCE: this is a NEW analysis output
under ``.omx/state/`` keyed by UTC timestamp; no existing artifacts are
mutated. Per Catalog #131: write is single-process append-style (no fcntl
lock needed because the output path is per-invocation unique).

The recode queue's ``estimated_recoverable_zip_bytes`` field uses Shannon
entropy floor on the existing compressed bytes — that is a DIFFERENT model
from canonical equation #26 (REPLACEMENT savings via procedural-codebook-
from-seed). The execution surfaces both numbers per-row so the operator can
compare apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Canonical contest rate-term constants per CLAUDE.md "Submission auth eval"
CANONICAL_RATE_DENOM_BYTES = 37_545_489
CANONICAL_RATE_MULTIPLIER = 25.0


@dataclass(frozen=True)
class ClassificationVerdict:
    queue_rank: int
    candidate_class: str
    archive_zip_path: str
    archive_zip_bytes: int
    archive_zip_sha256: str
    member_names: list[str]
    submission_shape_hint: str
    # Recode-queue-source (Shannon entropy floor on current compressed bytes)
    inventory_recoverable_zip_bytes: int
    inventory_rate_delta_if_floor_reached: float
    # Canonical equation #26 IN-DOMAIN/EXCLUDED/UNCLASSIFIED classification
    canonical_equation_26_classification: str  # IN_DOMAIN | EXCLUDED | UNCLASSIFIED
    canonical_equation_26_in_domain_context: str | None
    canonical_equation_26_excluded_context: str | None
    canonical_equation_26_unclassified_reason: str | None
    # Predicted bytes-saved per canonical equation #26 IF in-domain
    # (REPLACEMENT savings via procedural-codebook-from-seed)
    eq26_predicted_codebook_size_bytes_lower: int | None
    eq26_predicted_codebook_size_bytes_upper: int | None
    eq26_predicted_seed_size_bytes: int | None
    eq26_predicted_bytes_saved_lower: int | None
    eq26_predicted_bytes_saved_upper: int | None
    eq26_predicted_delta_s_lower: float | None
    eq26_predicted_delta_s_upper: float | None
    # Reactivation criteria for EXCLUDED contexts
    excluded_reactivation_criteria: str | None
    # Operator routing
    ready_to_paste_command: str
    blockers: list[str]
    # Canonical non-promotable markers per Catalog #341 + #323
    score_claim: bool = False
    promotable: bool = False
    axis_tag: str = "[predicted]"
    evidence_grade: str = "predicted"


@dataclass(frozen=True)
class ExecutedQueue:
    schema: str
    generated_at_utc: str
    lane_id: str
    source_queue_json: str
    source_inventory_json: str
    canonical_equation_id: str
    canonical_equation_in_domain_contexts: list[str]
    canonical_equation_excluded_contexts: list[str]
    classifications: list[ClassificationVerdict]
    aggregate_in_domain_predicted_delta_s_lower: float
    aggregate_in_domain_predicted_delta_s_upper: float
    aggregate_excluded_count: int
    aggregate_unclassified_count: int
    aggregate_in_domain_count: int
    top_5_in_domain_candidates: list[int]  # queue_rank indices
    sister_binding: list[str]
    score_claim: bool = False
    promotable: bool = False
    axis_tag: str = "[predicted]"
    evidence_grade: str = "predicted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# -----------------------------------------------------------------------------
# Canonical equation #26 IN-DOMAIN vs EXCLUDED classification per candidate
# class. Derived from canonical equation #26 module:
#   src/tac/canonical_equations/procedural_codebook_savings.py
#
# Reference IN-DOMAIN tuple (11 contexts):
#   intermediate_transform_quantizer / intermediate_transform_dequantizer /
#   procedural_codebook_as_lookup_table / comma2k19_ood_derived_basis_replacement /
#   chroma_lut_replacement / class_anchor_replacement / nscs06_v8_chroma_lut /
#   atw_v2_codec_quantizer_lut / tt5l_transformer_tokens / dp1_codebook_bytes /
#   deterministic_constants_codebook_replacement
#
# Reference EXCLUDED tuple (6 contexts, post-RATIFY-4):
#   direct_dwt_detail_subband_byte_substitution /
#   direct_byte_substitution_on_wavelet_decomposition_coefficients /
#   master_gradient_null_byte_removal_with_constant_reconstruction /
#   master_gradient_null_byte_replacement_with_arbitrary_constant /
#   direct_byte_substitution_on_parser_safe_but_score_affecting_raw_sections /
#   direct_byte_substitution_on_decode_opaque_raw_sections  (NEW RATIFY-4)
# -----------------------------------------------------------------------------

# Per-candidate-class canonical equation #26 mapping:
#   key = candidate_class from plan_archive_surface_recode_queue.py
#   value = (classification, in_domain_context or None, excluded_context or None,
#            unclassified_reason or None, eq26_seed_bytes, eq26_codebook_lower,
#            eq26_codebook_upper, eq26_reactivation_criteria or None)
_CLASS_TO_EQ26_VERDICT: dict[str, dict[str, Any]] = {
    # pr101_null_byte_smoke: master-gradient null-byte removal smoke; the
    # variant suffixes V_BASELINE / V_HALF / V_ZERO / V_RANDOM are literal byte
    # replacements at gradient-null positions. Per canonical equation #26
    # EXCLUDED context `master_gradient_null_byte_removal_with_constant_reconstruction`
    # + `master_gradient_null_byte_replacement_with_arbitrary_constant`. Bug
    # class anchor: master-gradient correctly reports zero gradient-leverage,
    # but the bytes are BIT-ESSENTIAL for inflate parsing.
    "pr101_null_byte_smoke": {
        "classification": "EXCLUDED",
        "in_domain_context": None,
        "excluded_context": "master_gradient_null_byte_replacement_with_arbitrary_constant",
        "unclassified_reason": None,
        "eq26_seed_bytes": None,
        "eq26_codebook_lower": None,
        "eq26_codebook_upper": None,
        "reactivation_criteria": (
            "if a future smoke proves the bytes are NOT bit-essential for "
            "inflate parsing AND the replacement preserves decode integrity, "
            "re-classify per the canonical byte-mutation smoke (Catalog #272 "
            "distinguishing-feature integration contract)"
        ),
    },
    # hfv1_pr101_adapter: HFV1 sidecar adapter; `foveation_params.bin` is a
    # parser-visible adapter payload (canonical PR101 grammar extension). Not
    # a procedural-codebook replacement candidate per canonical equation #26
    # (the adapter is BUILT-AT-COMPRESS-TIME-FROM-MEASURED-FOVEATION-DATA, not
    # derived from a small deterministic seed). Per CLAUDE.md "Forbidden
    # empirical-claim-without-evidence-tag" + Catalog #344 canonical-equation
    # evolution discipline: UNCLASSIFIED until a future sister canonical
    # equation handles compress-time-measured-payload sidecars.
    "hfv1_pr101_adapter": {
        "classification": "UNCLASSIFIED",
        "in_domain_context": None,
        "excluded_context": None,
        "unclassified_reason": (
            "HFV1 sidecar adapter is a compress-time-measured-foveation-data "
            "payload, NOT a procedural-codebook-from-seed candidate. Canonical "
            "equation #26 predicts REPLACEMENT savings via deterministic seed "
            "derivation; HFV1's payload is empirically measured. Awaits future "
            "sister canonical equation for compress-time-measured-payload "
            "sidecars per Catalog #344 evolution discipline."
        ),
        "eq26_seed_bytes": None,
        "eq26_codebook_lower": None,
        "eq26_codebook_upper": None,
        "reactivation_criteria": None,
    },
    # lfv1_lapose_foveation: LFV1 sidecar `lapose_foveation_tuples.lfv1` +
    # `foveation_params.bin`. Same UNCLASSIFIED reasoning as HFV1: the tuples
    # are compress-time-measured-from-PoseNet-output, not procedurally-
    # derived. The high inventory-recoverable bytes (40-60KB) measure
    # entropy-floor headroom under arithmetic coding, NOT codebook-replacement
    # savings.
    "lfv1_lapose_foveation": {
        "classification": "UNCLASSIFIED",
        "in_domain_context": None,
        "excluded_context": None,
        "unclassified_reason": (
            "LFV1 sidecar carries compress-time-measured-foveation tuples + "
            "params, NOT procedural-codebook-from-seed bytes. Inventory's "
            "entropy-floor recoverable estimate uses Shannon arithmetic-coding "
            "headroom, NOT canonical equation #26's REPLACEMENT-savings model. "
            "Awaits future sister equation for tuple-based foveation sidecars."
        ),
        "eq26_seed_bytes": None,
        "eq26_codebook_lower": None,
        "eq26_codebook_upper": None,
        "reactivation_criteria": None,
    },
    # openpilot_prior_candidate: `class_codebook.json` IS a canonical
    # equation #26 IN-DOMAIN candidate — the class-codebook is a
    # deterministic-constants-codebook-replacement context. The
    # `categorical_payload.bin` is parser-essential codes pointing into the
    # codebook (not procedural-replacement candidate itself).
    "openpilot_prior_candidate": {
        "classification": "IN_DOMAIN",
        "in_domain_context": "deterministic_constants_codebook_replacement",
        "excluded_context": None,
        "unclassified_reason": None,
        "eq26_seed_bytes": 32,  # canonical seed size per memo §4
        "eq26_codebook_lower": 2048,  # 2 KB lower bound per memo §4
        "eq26_codebook_upper": 6144,  # 6 KB upper bound per memo §4
        "reactivation_criteria": None,
    },
    # z7_world_model_candidate: Z7 mamba2 static_capacity_control `0.bin`
    # member is the canonical single-blob substrate. Per canonical equation
    # #26 the candidate IS a procedural-codebook replacement target IF the
    # static-capacity-control codebook is byte-level identifiable. Per
    # Catalog #325 per-substrate symposium discipline + Catalog #324 post-
    # training Tier-C validation: classification IS IN-DOMAIN
    # (`tt5l_transformer_tokens` is a sister canonical context; mamba2 is
    # structurally similar — sequential mixer + linear-attention with
    # tokenized inputs) BUT empirical anchor pending Z7 sister symposium per
    # operator-routable.
    "z7_world_model_candidate": {
        "classification": "IN_DOMAIN",
        "in_domain_context": "tt5l_transformer_tokens",
        "excluded_context": None,
        "unclassified_reason": None,
        "eq26_seed_bytes": 32,
        "eq26_codebook_lower": 2048,
        "eq26_codebook_upper": 6144,
        "reactivation_criteria": None,
    },
    # generic_entropy_headroom: catch-all for archives without a frontier-
    # relevant binding. Per Catalog #344 canonical-equation evolution:
    # UNCLASSIFIED until the operator binds the archive to a specific
    # substrate class.
    "generic_entropy_headroom": {
        "classification": "UNCLASSIFIED",
        "in_domain_context": None,
        "excluded_context": None,
        "unclassified_reason": (
            "Generic entropy-headroom archive lacks frontier-relevant binding; "
            "canonical equation #26 requires substrate-class identification "
            "before predicted-bytes-saved can be computed. Operator-routable: "
            "bind archive to a canonical substrate class first."
        ),
        "eq26_seed_bytes": None,
        "eq26_codebook_lower": None,
        "eq26_codebook_upper": None,
        "reactivation_criteria": None,
    },
}


def _compute_eq26_prediction(
    seed_bytes: int | None,
    codebook_lower: int | None,
    codebook_upper: int | None,
) -> tuple[int | None, int | None, float | None, float | None]:
    """Return ``(bytes_saved_lower, bytes_saved_upper, delta_s_lower, delta_s_upper)``."""
    if seed_bytes is None or codebook_lower is None or codebook_upper is None:
        return (None, None, None, None)
    bytes_saved_lower = codebook_lower - seed_bytes
    bytes_saved_upper = codebook_upper - seed_bytes
    delta_s_lower = (
        -CANONICAL_RATE_MULTIPLIER * bytes_saved_lower / CANONICAL_RATE_DENOM_BYTES
    )
    delta_s_upper = (
        -CANONICAL_RATE_MULTIPLIER * bytes_saved_upper / CANONICAL_RATE_DENOM_BYTES
    )
    return (bytes_saved_lower, bytes_saved_upper, delta_s_lower, delta_s_upper)


def _ready_to_paste_command(
    *, queue_rank: int, candidate_class: str, classification: str
) -> str:
    """Per-classification ready-to-paste operator command."""
    if classification == "IN_DOMAIN":
        return (
            "# in-domain: queue this candidate for per-substrate symposium "
            "(Catalog #325) + post-training Tier-C validation (Catalog #324) "
            "+ canonical byte-mutation smoke (Catalog #272 distinguishing-"
            "feature integration contract) before any paid dispatch. NO "
            "operator-authorize invocation until empirical anchor lands."
        )
    if classification == "EXCLUDED":
        return (
            "# excluded: DO NOT dispatch — canonical equation #26 EXCLUDED "
            "context. The reactivation criteria field documents the empirical "
            "smoke that would unblock this candidate."
        )
    return (
        "# unclassified: queue for canonical-equation-26 evolution per "
        "Catalog #344 — a future sister equation may handle this candidate "
        "class. NO dispatch until classification resolves."
    )


def execute_queue(
    queue_json_path: Path,
    *,
    lane_id: str,
) -> ExecutedQueue:
    payload = json.loads(queue_json_path.read_text(encoding="utf-8"))
    queue_rows = payload.get("queue", [])
    if not isinstance(queue_rows, list):
        raise ValueError(f"queue JSON {queue_json_path} has no list-valued 'queue'")

    inventory_json = payload.get("inventory_json", "")

    # Canonical equation #26 reference (frozen at module load time).
    from tac.canonical_equations.procedural_codebook_savings import (
        _INCLUDED_CONTEXTS,
        _EXCLUDED_CONTEXTS,
    )
    in_domain_contexts = list(_INCLUDED_CONTEXTS)
    excluded_contexts = list(_EXCLUDED_CONTEXTS)

    classifications: list[ClassificationVerdict] = []
    aggregate_lower = 0.0
    aggregate_upper = 0.0
    in_domain_count = 0
    excluded_count = 0
    unclassified_count = 0
    in_domain_ranks: list[tuple[int, float]] = []

    for row in queue_rows:
        candidate_class = str(row.get("candidate_class", ""))
        verdict_template = _CLASS_TO_EQ26_VERDICT.get(
            candidate_class,
            _CLASS_TO_EQ26_VERDICT["generic_entropy_headroom"],
        )
        (
            bs_lower,
            bs_upper,
            ds_lower,
            ds_upper,
        ) = _compute_eq26_prediction(
            seed_bytes=verdict_template["eq26_seed_bytes"],
            codebook_lower=verdict_template["eq26_codebook_lower"],
            codebook_upper=verdict_template["eq26_codebook_upper"],
        )
        classification = verdict_template["classification"]
        if classification == "IN_DOMAIN":
            in_domain_count += 1
            if ds_lower is not None and ds_upper is not None:
                aggregate_lower += ds_lower
                aggregate_upper += ds_upper
            in_domain_ranks.append((int(row["rank"]), ds_lower or 0.0))
        elif classification == "EXCLUDED":
            excluded_count += 1
        else:
            unclassified_count += 1
        ready_cmd = _ready_to_paste_command(
            queue_rank=int(row["rank"]),
            candidate_class=candidate_class,
            classification=classification,
        )
        classifications.append(
            ClassificationVerdict(
                queue_rank=int(row["rank"]),
                candidate_class=candidate_class,
                archive_zip_path=str(row["archive_zip_path"]),
                archive_zip_bytes=int(row["archive_zip_bytes"]),
                archive_zip_sha256=str(row["archive_zip_sha256"]),
                member_names=list(row.get("member_names", [])),
                submission_shape_hint=str(row.get("submission_shape_hint", "")),
                inventory_recoverable_zip_bytes=int(
                    row.get("estimated_recoverable_zip_bytes", 0)
                ),
                inventory_rate_delta_if_floor_reached=float(
                    row.get("estimated_rate_delta_if_floor_reached", 0.0)
                ),
                canonical_equation_26_classification=classification,
                canonical_equation_26_in_domain_context=verdict_template[
                    "in_domain_context"
                ],
                canonical_equation_26_excluded_context=verdict_template[
                    "excluded_context"
                ],
                canonical_equation_26_unclassified_reason=verdict_template[
                    "unclassified_reason"
                ],
                eq26_predicted_codebook_size_bytes_lower=verdict_template[
                    "eq26_codebook_lower"
                ],
                eq26_predicted_codebook_size_bytes_upper=verdict_template[
                    "eq26_codebook_upper"
                ],
                eq26_predicted_seed_size_bytes=verdict_template["eq26_seed_bytes"],
                eq26_predicted_bytes_saved_lower=bs_lower,
                eq26_predicted_bytes_saved_upper=bs_upper,
                eq26_predicted_delta_s_lower=ds_lower,
                eq26_predicted_delta_s_upper=ds_upper,
                excluded_reactivation_criteria=verdict_template[
                    "reactivation_criteria"
                ],
                ready_to_paste_command=ready_cmd,
                blockers=list(row.get("promotion_blockers", [])),
            )
        )

    # Top-5 IN_DOMAIN by abs(ds_lower) magnitude (most negative = best)
    in_domain_ranks.sort(key=lambda t: t[1])  # most negative first
    top_5 = [r for r, _ in in_domain_ranks[:5]]

    return ExecutedQueue(
        schema="archive_surface_recode_queue_executed_v1",
        generated_at_utc=_utc_iso(),
        lane_id=lane_id,
        source_queue_json=str(queue_json_path),
        source_inventory_json=inventory_json,
        canonical_equation_id="procedural_codebook_from_seed_compression_savings_v1",
        canonical_equation_in_domain_contexts=in_domain_contexts,
        canonical_equation_excluded_contexts=excluded_contexts,
        classifications=classifications,
        aggregate_in_domain_predicted_delta_s_lower=aggregate_lower,
        aggregate_in_domain_predicted_delta_s_upper=aggregate_upper,
        aggregate_excluded_count=excluded_count,
        aggregate_unclassified_count=unclassified_count,
        aggregate_in_domain_count=in_domain_count,
        top_5_in_domain_candidates=top_5,
        sister_binding=[
            "NSCS06 v8 chroma_lut canonical pattern (lane lane_overnight_a_nscs06_v8_phase_2_design_20260521)",
            "DP1 codebook_bytes canonical pattern (lane lane_overnight_b_dp1_paired_smoke_harvest_verdict_20260521)",
            "VQ-VAE PROCEDURAL pattern (lane lane_overnight_e_procedural_codebook_generator_20260521)",
            "ATW V2 cdf_table_blob RATIFY-4 EXCLUDED context (commit 057130de4)",
        ],
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-json",
        type=Path,
        help="Path to archive_surface_recode_queue.json; defaults to newest.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/results"),
        help="Root used to find newest queue if --queue-json omitted.",
    )
    parser.add_argument(
        "--lane-id",
        type=str,
        default="lane_overnight_g_archive_surface_recode_queue_planner_execution_20260521",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(".omx/state") / f"archive_surface_recode_queue_executed_{_utc_stamp()}.json",
    )
    return parser.parse_args(argv)


def _newest_queue_json(results_root: Path) -> Path:
    candidates = sorted(
        results_root.glob(
            "archive_surface_recode_queue_*/archive_surface_recode_queue.json"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no archive_surface_recode_queue_*/archive_surface_recode_queue.json under {results_root}"
        )
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    queue_json = args.queue_json or _newest_queue_json(args.results_root)
    executed = execute_queue(queue_json, lane_id=args.lane_id)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(executed.to_dict(), indent=2, sort_keys=True) + "\n"
    args.output_path.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
