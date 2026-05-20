# SPDX-License-Identifier: MIT
"""Cathedral consumer for master-gradient null-space candidate bytes.

Sister of ``tac.cathedral_consumers.per_frame_sensitivity_consumer`` at
the *byte-axis* surface (per_frame is the *frame-axis* surface). Where
per-frame ranks frames by aggregated sensitivity, this consumer flags
*indices* where the scorer is structurally insensitive — candidates for
procedural-codebook replacement per
``.omx/research/procedural_codebook_generator_null_exploit_design_20260520.md``
+ the canonical helper ``tac.master_gradient`` (Catalog #318).

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #341 Tier
A: this consumer is **observability-only** (``predicted_delta_adjustment
= 0.0``; ``promotable = False``; ``axis_tag = "[predicted]"``). It does
NOT propose mutations. Any mutation MUST route through
``CandidateModificationSpec`` per Catalog #318 raw-byte-authority guard.

The candidate payload on each ``consume_candidate`` call MUST surface a
null-byte probe payload — either inline as a mapping under one of the
``_PAYLOAD_KEYS``, or as a JSON path under one of the ``_PATH_KEYS``
(produced by ``tools/probe_null_byte_master_gradient.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "null_byte_codebook_candidate_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)

# Auto-trigger pattern (commit a129c8857) — declares this consumer
# IS interested in master-gradient anchor updates so the cathedral
# autopilot's continual-learning posterior loop wakes it on every new
# anchor (per Catalog #335 + #354 sister discipline).
CONSUMES_MASTER_GRADIENT_ANCHORS = True

_PAYLOAD_KEYS = (
    "null_byte_probe",
    "null_byte_master_gradient_probe",
    "master_gradient_null_byte_probe",
)
_PATH_KEYS = (
    "null_byte_probe_json",
    "null_byte_master_gradient_probe_json",
    "master_gradient_null_byte_probe_json",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5.

    The null-byte probe is produced by a separate canonical tool
    (`tools/probe_null_byte_master_gradient.py`) reading a master-
    gradient anchor + archive.zip. There is no consumer-local cache to
    mutate when a fresh anchor lands; the next `consume_candidate` call
    will re-derive against the canonical probe payload supplied on the
    candidate row.
    """
    _ = anchor


def _load_payload(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in _PAYLOAD_KEYS:
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return value
    for key in _PATH_KEYS:
        value = candidate.get(key)
        if not isinstance(value, str):
            continue
        path = Path(value)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, Mapping):
            return loaded
    return None


def _section_top_k(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Rank sections by absolute null-byte count (highest first).

    Sections with 100% intra-section null density are most actionable
    candidates for procedural-codebook replacement (their entire byte
    span has zero score-axis sensitivity, modulo the rate-axis
    underflow caveat documented in the probe tool).
    """
    sections = payload.get("section_breakdown")
    if not isinstance(sections, Mapping):
        return []
    rows: list[Mapping[str, Any]] = []
    for name, info in sections.items():
        if not isinstance(info, Mapping) or name.startswith("_"):
            continue
        n_null = info.get("n_null")
        if not isinstance(n_null, int):
            continue
        rows.append(
            {
                "section": name,
                "n_null": n_null,
                "length_bytes": info.get("length_bytes"),
                "null_fraction_within_section": info.get("null_fraction_within_section"),
                "range": info.get("range"),
            }
        )
    rows.sort(key=lambda row: int(row["n_null"]), reverse=True)
    return rows


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4: surface null-byte candidates as observability metadata."""
    if not isinstance(candidate, Mapping):
        return _no_signal("candidate is not a mapping")
    payload = _load_payload(candidate)
    if payload is None:
        return _no_signal("no null-byte master-gradient probe payload on candidate")

    n_total = payload.get("n_total_bytes")
    n_null = payload.get("n_null_bytes")
    null_fraction = payload.get("null_fraction")
    epsilon = payload.get("epsilon")
    if not isinstance(n_total, int) or not isinstance(n_null, int):
        return _no_signal("null-byte probe payload missing n_total_bytes / n_null_bytes")
    if not isinstance(null_fraction, (int, float)):
        null_fraction = (n_null / n_total) if n_total else 0.0

    section_top_k = _section_top_k(payload)
    grammar = payload.get("grammar_detected")
    grammar_label = (
        f"{grammar.get('selector_magic', 'unknown')}-grammar"
        if isinstance(grammar, Mapping)
        else "no-grammar"
    )
    rationale = (
        f"null-byte master-gradient probe available; n_total={n_total}; "
        f"n_null={n_null} ({null_fraction*100:.2f}%); epsilon={epsilon}; "
        f"grammar={grammar_label}; sections_ranked_by_null_count="
        f"{[row['section'] for row in section_top_k[:3]]}"
    )

    return {
        # Catalog #341 Tier A canonical-routing markers (observability-only)
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "consumer_signal_kind": "null_byte_codebook_candidate_routing",
        # Operator-facing summary surfaces
        "n_total_bytes": n_total,
        "n_null_bytes": n_null,
        "null_fraction": null_fraction,
        "epsilon": epsilon,
        "grammar_detected": grammar,
        "section_top_k": section_top_k,
        # Cite-chain (Catalog #305 observability surface)
        "candidate_kind_for_procedural_codebook_replacement": "100_percent_null_sections_first",
        "source_anchor_sha256": payload.get("anchor_sha256"),
        "source_archive_zip_path": payload.get("archive_zip_path"),
    }


def _no_signal(reason: str) -> Mapping[str, Any]:
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": f"null-byte codebook candidate consumer: {reason} [predicted]",
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "consumer_signal_kind": "null_byte_codebook_candidate_absent",
    }
