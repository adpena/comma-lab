"""S2SBS — Stride-2-Stem Byte-Stuffing substrate.

Substrate-engineering L0/L1 scaffold exploiting the empirically measured
stride-2-stem HF spatial-frequency blindspot of the upstream SegNet /
PoseNet scorers (audit memo: ``.omx/research/s2sbs_blindspot_audit_20260513.md``).

The HF blindspot lets the substrate encode arbitrary uint8 payload bytes
into the inflated raw RGB frames at high spatial frequencies without
moving the scorer-component distortion terms. Per the codex math
correction (`feedback_codex_math_correction_pr95_lora_dora_landed_20260513.md`)
the byte-stuffing INCREASES archive size (rate slope 6.66e-7 score/byte);
the stuffed bytes must therefore carry information that REDUCES the
seg+pose distortion components by at least the rate cost they impose.

The substrate is intentionally L0/L1 scaffold:

* Score-aware loss exists but is not yet exact-eval validated.
* Hermitian-FFT byte channel is empirically measured (PRBS-31 BER 0.42 at
  the largest joint-safe delta) but no ECC is bolted on yet.
* No archive bytes have entered the contest packet; ``score_claim`` and
  ``ready_for_exact_eval_dispatch`` are permanently False.

Catalog #124 archive-grammar fields:
    archive_grammar:            monolithic single-file 0.bin S2SBS_AR/S2S1
    parser_section_manifest:    parse_archive() -> S2sbsArchive
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:        torch only; no scorer imports in inflate
    export_format:              base_payload (LF) + hf_payload (Hermitian-FFT)
    score_aware_loss:           alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:         substrate_engineering L0/L1 scaffold
    no_op_detector_planned:     byte mutation changes parsed state and smoke RGB

Authority flags:
    research_only=true
    score_claim=false
    score_claim_valid=false
    promotion_eligible=false
    ready_for_exact_eval_dispatch=false

Wire-in hooks (per CLAUDE.md "Subagent coherence-by-default"):

1. Sensitivity-map contribution: per-pixel HF blindspot mask is exported
   via ``S2sbsConfig.hf_blindspot_radius`` and consumed by future bit-
   allocator hooks.
2. Pareto constraint: ``bytes_per_frame_joint_safe`` enters the rate-axis
   feasible region. Currently advisory.
3. Bit-allocator hook: payload-byte budget per frame is dictated by HF
   blindspot capacity, NOT per-tensor importance.
4. Cathedral autopilot dispatch hook: N/A — research-only.
5. Continual-learning posterior update: N/A — no empirical anchor yet.
6. Probe-disambiguator: N/A — single defensible interpretation
   (architectural HF blindspot empirically measured in φ3 audit).
"""

from __future__ import annotations

from .architecture import (
    CAMERA_HW,
    CONTEST_NUM_PAIRS,
    SCORER_HW,
    HfBlindspotMask,
    HfFftByteCodec,
    PayloadChannel,
    S2sbsConfig,
    S2sbsRenderer,
)
from .archive import (
    S2S1_GRAMMAR,
    S2S1_HEADER_STRUCT,
    S2S1_SCHEMA_VERSION,
    S2SBS_AR_MAGIC,
    S2sbsArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    S2sbsLossWeights,
    S2sbsScoreAwareLoss,
)

S2SBS_METADATA = {
    "research_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "audit_memo": ".omx/research/s2sbs_blindspot_audit_20260513.md",
    "audit_capacity_bytes_per_frame_joint_safe": 97460,
    "audit_prbs_ber_at_joint_safe_delta": 0.4231,
    "audit_evidence_grade": "macOS-CPU advisory",
}

__all__ = [
    "CAMERA_HW",
    "CONTEST_NUM_PAIRS",
    "S2S1_GRAMMAR",
    "S2S1_HEADER_STRUCT",
    "S2S1_SCHEMA_VERSION",
    "S2SBS_AR_MAGIC",
    "S2SBS_METADATA",
    "SCORER_HW",
    "HfBlindspotMask",
    "HfFftByteCodec",
    "PayloadChannel",
    "S2sbsArchive",
    "S2sbsConfig",
    "S2sbsLossWeights",
    "S2sbsRenderer",
    "S2sbsScoreAwareLoss",
    "pack_archive",
    "parse_archive",
]
