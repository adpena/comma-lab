# SPDX-License-Identifier: MIT
"""tac.wyner_ziv_deliverability — canonical helper for per-substrate Wyner-Ziv
deliverability proof construction.

Sister of:
  * ``tac.master_gradient_consumers.wyner_ziv_side_info_covariance`` (PRODUCER —
    classifies bytes by cross-pair gradient correlation into candidate-shared /
    pair-specific / mixed).
  * ``tac.sensitivity_map.wyner_ziv_reweight`` (CONSUMER — applies WZ
    classification to sensitivity-map axis-level reweight; sister wire-in #2).
  * ``tac.side_information`` (decorator namespace for canonical bakers; this
    module's per-tier classification draws on the same Wyner-Ziv 1976 +
    Catalog #213 Comma2k19LocalCache + HNeRV parity L4/L9 contracts).

Per the T3 Grand Council symposium 2026-05-17 verdict
``.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md``:
this module is the **Q1 canonical helper** in the 5-subagent dispatch chain.
Q2 (Catalog #319 preflight gate), Q3 (autopilot reweight v2), Q4 (FEC6
Comma2k19 palette smoke), and Q5 (lane registry integration) all CONSUME the
``DeliverabilityProof`` dataclass produced here.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L4 +
L9 and "Forbidden /tmp paths in any persisted artifact" — the proof's
``canonical_helper_invocation`` field anchors every Tier 2 byte to a
deterministic public-source helper (e.g. ``Comma2k19LocalCache.fetch_chunk``)
so the contest decoder can reconstruct the side-info channel without
loading scorer state OR fetching from the network at inflate time.

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience"):

  Dataclass + enum:
    - ``DeliverabilityProof`` — frozen dataclass with all canonical fields
    - ``DeliverabilityTier`` — str Enum (TIER_1_ZERO_COST ... TIER_4_FORBIDDEN)

  Builder:
    - ``build_deliverability_proof_from_wyner_ziv_classification(...)``

  Reader:
    - ``load_deliverability_proof_for_archive(archive_sha256)``

  Verifier:
    - ``verify_deliverability_proof_contest_compliance(proof)``

  Path constants:
    - ``WYNER_ZIV_DELIVERABILITY_PROOFS_DIR``

  Schema version:
    - ``DELIVERABILITY_PROOF_SCHEMA_VERSION``

Cross-references:
  * Symposium verdict: ``.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md``
  * Implementation queue: ``.omx/research/wyner_ziv_optimal_implementation_queue_20260517.md``
  * Premise verifier: ``.omx/tmp/q1_deliverability_proof_builder_premise_verifier.txt``
  * Catalog #319 STRICT preflight gate (Q2 successor; re-claimed after
    #318 collision with sister-landed ``check_master_gradient_raw_byte_authority_not_landed``
    at commit ``84c8f5d5b``)
"""

from __future__ import annotations

from tac.wyner_ziv_deliverability.proof_builder import (
    DELIVERABILITY_PROOF_SCHEMA_VERSION,
    WYNER_ZIV_DELIVERABILITY_PROOFS_DIR,
    DeliverabilityProof,
    DeliverabilityTier,
    build_deliverability_proof_from_wyner_ziv_classification,
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

__all__ = [
    # Dataclass + enum
    "DeliverabilityProof",
    "DeliverabilityTier",
    # Builder + reader + verifier
    "build_deliverability_proof_from_wyner_ziv_classification",
    "load_deliverability_proof_for_archive",
    "verify_deliverability_proof_contest_compliance",
    # Path + schema constants
    "WYNER_ZIV_DELIVERABILITY_PROOFS_DIR",
    "DELIVERABILITY_PROOF_SCHEMA_VERSION",
]
