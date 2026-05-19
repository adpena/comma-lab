# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.procedural_codebook_generator``.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.procedural_codebook_generator`` per wiring + integration audit
2026-05-19 (commit 3821cfb6b).

``tac.procedural_codebook_generator`` is the canonical helper for
procedurally-generated codebook expansion from archive seeds (per Catalog
#329 contest-compliance + per the inflate_py extreme compression symposium
2026-05-18 ProvenanceKind extension). Exposes ``classify_procedural_seed_authority``
/ ``derive_codebook_from_archive_bytes`` / ``expand_seed_to_codebook`` /
``verify_no_new_bytes_added`` / ``verify_generator_seed_mutation_smoke``.
Consumer surfaces availability + a clear ``[predicted]`` discipline marker
per Catalog #287; per-candidate codebook derivation must route through
explicit archive-seed inclusion checks.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "procedural_codebook_generator_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. Procedural codebook derivation is
    a deterministic function of the input archive seed bytes; no
    anchor-driven posterior update.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation. Per-candidate procedural
    codebook contribution requires explicit archive-seed inclusion
    verification per Catalog #329 contest-compliance payload-kinds.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.procedural_codebook_generator canonical procedural codebook "
            "expansion helpers available (classify_procedural_seed_authority "
            "/ derive_codebook_from_archive_bytes / expand_seed_to_codebook / "
            "verify_no_new_bytes_added / verify_generator_seed_mutation_smoke) "
            "[predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
