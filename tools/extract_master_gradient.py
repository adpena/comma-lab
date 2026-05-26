# SPDX-License-Identifier: MIT
# Catalog #270 scope clarification (2026-05-17 + lane_catalog_270_scope_fix_tool_vs_substrate_dispatch_20260517):
# this file is a TOOL dispatch (``tools/extract_master_gradient.py``), NOT a
# substrate trainer (``experiments/train_substrate_*.py``). Substrate-only Tier 3
# fields (Catalogs #172 autocast / #178 TF32 / #179 torch.compile / #226 canonical
# auth-eval helper) are categorically inapplicable and are skipped by
# ``src/tac/deploy/dispatch_protocol.py::_is_tool_dispatch`` via implicit
# ``tools/*.py`` detection + the recipe's explicit ``dispatch_kind: tool`` field.
"""Master-gradient extractor for diagnostic score-response tensors.

[verified-against: .omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md §3.2 REVISED]
[verified-against: .omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md §1.1 op-routable #1]

Method: 1 forward pass + 3 backward passes per the symposium §3.2 revised methodology.
NOT finite-difference per-bit (Round-1 C-1 + C-2 falsified that as infeasible — 178517 byte
flips × per-flip inflate = O(days) on T4; mathematically yields ±1 bit deltas which are
quantized at zero before reaching the scorer for most bytes).

The autograd path:
    forward:  z_latents (frozen) -> decoder(weights) -> rgb_pair[0:eval_size]
    upscale:  bicubic -> 874x1164 -> bilinear roundtrip down to 384x512 (eval_roundtrip)
    selector: fec6 huffman compact selector applied per pair (NO grad through selector;
              it operates AFTER round to uint8 which is the canonical eval cliff)
    scorers:  segnet + posenet preprocess -> forward -> d_seg + d_pose
    backward(d_seg):   per-parameter d(d_seg)/d(theta)  [SegNet uses x[:, -1, ...] so only
                       frame_1 weights get gradient flow]
    backward(d_pose):  per-parameter d(d_pose)/d(theta)
    rate analytical:   archive byte-count term, recorded via packet-valid
                       operator response rows rather than byte-value deltas

Per-byte projection through the fec6 codec Jacobian:
    The fec6 archive grammar (per submission_dir/src/codec.py) stores:
      - Decoder: per-tensor (int8 mantissa stream after zig/negzig/twos/off mapping) + fp16 scale
        --> w_dequant[i] = mantissa_byte[i] * scale_fp16  (per-tensor scale; sign embedded in int8)
        --> d(w)/d(mantissa_byte[i]) = scale_fp16
        --> d(score)/d(mantissa_byte[i]) = d(score)/d(w[i]) * scale_fp16
      - Per-tensor scale (fp16, 2 bytes):
        --> d(w)/d(scale) = mantissa_byte[i]  -> aggregated across all weights in tensor
      - Latents: uint8 temporal-delta cumulative + fp16 mins/scales -> z_latents are FROZEN
        in this extraction (we measure the FIXED operating point), so latent-byte sensitivity
        comes from feeding latents through the decoder's stem (which is small and gradient-rich).
      - Sidecar: huffman-coded delta_x100 per pair, applied to latents POST-decode. Treated
        as PARTIAL (zero gradient) in v1 because the discrete code -> selected delta mapping
        is non-differentiable. Symposium §3.6 use #4 explicitly notes this is the next
        refinement target.

Output: master gradient ledger anchor at .omx/state/master_gradient_anchors.jsonl + sidecar
.npy file at OUTPUT_NPY (caller-specified; symposium §3.6 use #7 mandates non-/tmp path).

Authority boundary: this extractor emits diagnostic tensors only. Score-lowering
authority must route through `CandidateModificationSpec` rows with
`grammar_aware_operator` coordinates and packet proofs in
`tac.master_gradient_operator_plan`. One-member contest ZIPs are charged as ZIP
bytes but differentiated over their inner payload domain; both identities are
recorded separately.

CLI:
    .venv/bin/python tools/extract_master_gradient.py \\
        --archive submissions/fec6/archive.zip \\
        --inflate-py experiments/results/.../submission_dir/inflate.py \\
        --upstream-dir upstream \\
        --axis '[contest-CPU]' \\
        --output-npy .omx/state/master_gradient_fec6_20260517.npy \\
        --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import struct
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Catalog #152 WAVE-1 APPARATUS HARDENING extension 2026-05-16:
# the fec6 archive + submission_dir/inflate.py live under experiments/results/**
# which is Modal-IGNORED per tac.deploy.modal.mount_manifest.DEFAULT_RESULTS_IGNORE.
# Declare the required-input paths here so mount_manifest.collect_extra_mount_paths
# stages them. Bug-class anchor: STC v2 smoke fc-01KRSB76H04HM4958V2HX2JZZ4 rc=25
# (2026-05-16) for the same Modal-IGNORED required-input class.
TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = (
    "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip",
    "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py",
    "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src",
)

from tac.differentiable_eval_roundtrip import (  # noqa: E402
    apply_eval_roundtrip_during_training,
    patch_upstream_yuv6_globally,
    unpatch_upstream_yuv6,
)
from tac.master_gradient import (  # noqa: E402
    CONTEST_RATE_DENOM_BYTES,
    PER_PAIR_GRADIENT_TENSOR_KIND,
    MasterGradient,
    OperatingPoint,
    append_anchor_locked,
    compute_marginal_coefficients,
    score_axis_dominance_summary,
)
from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA,
    decode_pr106_sidecar_packet_correction_passes,
    parse_pr106_sidecar_packet,
    pr106_sidecar_consumed_byte_proof,
)
from tac.quantization import Uint8STE  # noqa: E402
from tac.scorer import load_differentiable_scorers  # noqa: E402
from tac.substrates.pretrained_driving_prior.archive import (  # noqa: E402
    DP1_SECTION_ROLES,
    parse_dp1_archive_bytes,
)

# ---------------------------------------------------------------------------- #
# Per-tensor parsed-from-fec6 metadata                                          #
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _TensorByteSpan:
    """One decoded fec6 tensor's byte layout in the archive.

    The fec6 codec writes streams in DECODER_STORAGE_ORDER (28 tensors). After
    brotli-decompression each tensor occupies (numel) mantissa bytes + 2 fp16
    scale bytes contiguously. We track each span here so the per-byte
    gradient projection can be assembled cheaply.
    """

    name: str
    storage_index: int
    shape: tuple[int, ...]
    numel: int
    mantissa_byte_offset: int  # offset in DECODED brotli stream (not raw archive)
    scale_byte_offset: int
    fp16_scale: float
    byte_map: str  # "zig" | "negzig" | "twos" | "off"


@dataclass(frozen=True)
class _Fec6ArchiveLayout:
    """Parsed fec6 archive layout: decoder + latents + sidecar.

    The ARCHIVE_RAW bytes are the source-of-truth indexing for the master
    gradient: byte_i in [0, n_archive_bytes) maps to a region (decoder/latent/
    sidecar). We carry per-region metadata to project per-parameter gradient
    back to per-archive-byte sensitivity.
    """

    archive_path: Path
    archive_sha256: str
    archive_bytes: bytes
    n_archive_bytes: int
    decoder_blob_offset: int  # start of decoder blob in archive_bytes (0 for FP11 outer)
    decoder_blob_len: int
    decoder_tensor_spans: tuple[_TensorByteSpan, ...]
    decoder_raw_decompressed: bytes
    latent_blob_offset: int
    latent_blob_len: int
    sidecar_blob_offset: int
    sidecar_blob_len: int
    n_pairs: int
    latent_dim: int
    base_channels: int
    eval_size: tuple[int, int]
    has_fp11_outer_wrapper: bool
    has_a1_headered_decoder: bool


@dataclass(frozen=True)
class ArchiveSection:
    """One byte section in a scored archive's gradient subject domain."""

    name: str
    offset: int
    length: int
    codec: str
    sha256: str
    notes: str = ""
    score_affecting: bool = True

    @property
    def end_offset(self) -> int:
        return self.offset + self.length

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "offset": self.offset,
            "length": self.length,
            "end_offset": self.end_offset,
            "codec": self.codec,
            "sha256": self.sha256,
            "notes": self.notes,
            "score_affecting": self.score_affecting,
        }


@dataclass(frozen=True)
class ArchiveProjectionContract:
    """Consumer-facing authority contract for one detected archive grammar.

    Archive boundary detection is useful for xray and routing, but it is not
    sufficient authority to emit a master-gradient anchor. Consumers need an
    explicit projector contract so packed/length-prefixed grammars cannot be
    accidentally treated like the PR101 fixed-section Jacobian path.
    """

    grammar_name: str
    authority: str
    anchor_emission_allowed: bool
    required_projector: str
    reason: str
    candidate_modification_spec_required: bool = True
    score_claim_allowed: bool = False
    promotion_eligible: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "grammar_name": self.grammar_name,
            "authority": self.authority,
            "anchor_emission_allowed": self.anchor_emission_allowed,
            "required_projector": self.required_projector,
            "reason": self.reason,
            "candidate_modification_spec_required": self.candidate_modification_spec_required,
            "score_claim_allowed": self.score_claim_allowed,
            "promotion_eligible": self.promotion_eligible,
        }


_SUPPORTED_PROJECTORS: dict[str, tuple[str, str]] = {
    "fec6_fp11_selector": (
        "fec6_pr101_fixed_section_int8_fp16_jacobian",
        "FP11 wraps a PR101-family fixed-section payload; existing fec6/PR101 projector preserves scored archive and gradient-subject custody.",
    ),
    "pr101_lc_v2": (
        "fec6_pr101_fixed_section_int8_fp16_jacobian",
        "PR101 fixed decoder/latent/sidecar offsets match the existing split-Brotli Jacobian path.",
    ),
    "pr106_format0d": (
        "pr106_format0d_primary_packed_hnerv_decoder_jacobian_sidecar_zero_grad_v1",
        "Format0d PacketIR primary payload is projected through the PR106 packed-HNeRV decoder Jacobian; discrete base/extra sidecar sections are preserved with explicit zero-gradient v1 semantics.",
    ),
    "pr107_apogee_length_prefixed": (
        "pr107_apogee_cd1_decoder_jacobian_camera_offset_roundtrip_latents_zero_grad_v1",
        "PR107 Apogee's CD1 decoder section is projected through its architecture-ordered HNeRV decoder Jacobian with the runtime camera-space channel offsets applied before the STE roundtrip; metadata and latent Brotli sections stay explicit zero-gradient v1 surfaces.",
    ),
    "a1_finetuned": (
        "a1_headered_pr101_fixed_section_int8_fp16_jacobian",
        "A1 adds a 4-byte decoder-section header, then reuses the PR101-family fixed-section projector with zero-gradient header semantics.",
    ),
}

_DETECTION_ONLY_PROJECTORS: dict[str, tuple[str, str]] = {
    "dp1_pretrained_driving_prior": (
        "dp1_deterministic_tensor_span_serializer_projector",
        "DP1 section offsets are canonical, but the current renderer stream is Brotli(pickle(state_dict)) and has no stable tensor-byte span grammar; codebook and residual sections also need explicit Jacobians or zero-gradient v1 contracts before anchor emission.",
    ),
    "pr106_ff_packed_hnerv": (
        "pr106_packed_brotli_schema_projector",
        "Public PR106 uses packed decoder and latents/sidecar Brotli sections; byte-gradient authority requires a packed-section schema projector.",
    ),
    "hnerv_lc_v2_length_prefixed": (
        "hnerv_lc_v2_schema_projector",
        "True hnerv_lc_v2 is four length-prefixed streams; fixed-offset PR101 projection would misattribute byte authority.",
    ),
}


def _projection_contract_for_name(grammar_name: str, *, gradient_projection_supported: bool) -> ArchiveProjectionContract:
    if gradient_projection_supported:
        supported = _SUPPORTED_PROJECTORS.get(grammar_name)
        if supported is None:
            raise ValueError(
                f"layout {grammar_name!r} claims gradient projection support without a registered projector"
            )
        projector, reason = supported
        return ArchiveProjectionContract(
            grammar_name=grammar_name,
            authority="gradient_projector_supported",
            anchor_emission_allowed=True,
            required_projector=projector,
            reason=reason,
        )

    projector, reason = _DETECTION_ONLY_PROJECTORS.get(
        grammar_name,
        (f"{grammar_name}_projector", "Archive grammar is detectable, but no registered projector authorizes anchor emission."),
    )
    return ArchiveProjectionContract(
        grammar_name=grammar_name,
        authority="fail_closed_detection_only",
        anchor_emission_allowed=False,
        required_projector=projector,
        reason=reason,
    )


def archive_projection_contract(layout: ArchiveLayout) -> ArchiveProjectionContract:
    """Return the fail-closed projection authority contract for ``layout``."""
    return _projection_contract_for_name(
        layout.grammar_name,
        gradient_projection_supported=layout.gradient_projection_supported,
    )


def list_archive_grammar_contracts() -> dict[str, object]:
    """Return the operator-facing extractor grammar registry."""
    grammar_names = (*_SUPPORTED_PROJECTORS, *_DETECTION_ONLY_PROJECTORS)
    grammars = [
        {
            **_projection_contract_for_name(
                grammar_name,
                gradient_projection_supported=grammar_name in _SUPPORTED_PROJECTORS,
            ).as_dict(),
            "gradient_projection_supported": grammar_name in _SUPPORTED_PROJECTORS,
        }
        for grammar_name in grammar_names
    ]
    return {
        "schema": "master_gradient_archive_grammar_registry_v1",
        "score_claim_allowed": False,
        "promotion_eligible": False,
        "grammar_count": len(grammars),
        "anchor_emitting_grammars": list(_SUPPORTED_PROJECTORS),
        "detection_only_grammars": list(_DETECTION_ONLY_PROJECTORS),
        "grammars": grammars,
    }


# Cable D D2 (task #887/#890) — analytical surface manifest
#
# Per `.omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md`
# Cable D D2 + task #890 prior memo: master-gradient wire-in to ALL analytical
# surfaces is partial. This manifest enumerates the analytical surfaces that
# consume master-gradient tensors and their per-surface coverage status so
# the operator (and the cathedral autopilot ranker) can see the gap at-a-
# glance and prioritize sister wire-ins.
#
# Per CLAUDE.md "Subagent coherence-by-default" + "PER-PAIR MASTER GRADIENT —
# wire-in coverage audit across ALL consumers" operator standing directive
# (task #810): every consumer/surface MUST close the producer→consumer loop.
# This manifest is the structural record of which surfaces have which
# coverage state.
#
# Coverage states:
#   - "active" — surface explicitly imports + calls tac.master_gradient APIs
#   - "indirect" — surface consumes via tac.master_gradient_consumers
#   - "pending" — declared as wire-in target; not yet implemented
#   - "decorator" — surface uses a canonical decorator pattern that wraps
#     master_gradient (boosting / compress_time_optimization patterns)

_ANALYTICAL_SURFACES: tuple[dict[str, str], ...] = (
    {
        "surface_id": "tac.sensitivity_map",
        "module_path": "src/tac/sensitivity_map/",
        "coverage": "active",
        "wire_in_hook": "wyner_ziv_reweight.py imports master_gradient",
        "notes": (
            "Cable D sister wiring; per-pair gradient feeds axis-level "
            "reweight via consumer 4 (Wyner-Ziv covariance)."
        ),
    },
    {
        "surface_id": "tac.master_gradient_consumers",
        "module_path": "src/tac/master_gradient_consumers.py",
        "coverage": "active",
        "wire_in_hook": "consumers 1-15 (15 = Lagrangian-dual planner)",
        "notes": (
            "v3 wave 2026-05-19 added consumers 7-14 (Pareto envelope, "
            "λ_R bisection, LoRA targets, coding budget, engineered "
            "correction, KKT residuals, Volterra cross-terms, decoder "
            "pruning). All 15 consumers now landed."
        ),
    },
    {
        "surface_id": "tac.unified_action",
        "module_path": "src/tac/unified_action.py",
        "coverage": "active",
        "wire_in_hook": "imports master_gradient for action-principle composition",
        "notes": "Per CLAUDE.md unified-Lagrangian action S_total(theta, archive_bytes, hardware).",
    },
    {
        "surface_id": "tac.utility_curves.per_byte_master_gradient",
        "module_path": "src/tac/utility_curves/per_byte_master_gradient.py",
        "coverage": "active",
        "wire_in_hook": "per-byte utility curves derived from master gradient",
        "notes": "Decorator-style; consumes via consumer wrappers.",
    },
    {
        "surface_id": "tac.canonical_duckdb.per_byte_sensitivity_ext",
        "module_path": "src/tac/canonical_duckdb/per_byte_sensitivity_ext.py",
        "coverage": "active",
        "wire_in_hook": "DuckDB per-byte sensitivity table backfill from master_gradient anchors",
        "notes": "Canonical DuckDB sensitivity backfill consumer.",
    },
    {
        "surface_id": "tac.wyner_ziv_deliverability.proof_builder",
        "module_path": "src/tac/wyner_ziv_deliverability/proof_builder.py",
        "coverage": "active",
        "wire_in_hook": "DeliverabilityProof consumes master_gradient for byte classification",
        "notes": "Per Catalog #319 — feeds autopilot reweight v2.",
    },
    {
        "surface_id": "tac.codec.wyner_ziv_layer",
        "module_path": "src/tac/codec/wyner_ziv_layer.py",
        "coverage": "active",
        "wire_in_hook": "Wyner-Ziv layer consumes master_gradient via side-info covariance",
        "notes": "Consumer 4 sister surface.",
    },
    {
        "surface_id": "tac.autopilot_rudin_daubechies.rashomon_ensemble",
        "module_path": "src/tac/autopilot_rudin_daubechies/rashomon_ensemble.py",
        "coverage": "active",
        "wire_in_hook": "RashomonEnsembleRanker.update_all_from_master_gradient",
        "notes": "Per Catalog #252 + consumer 6 (Rashomon disagreement queue).",
    },
    {
        "surface_id": "tac.optimization.bit_allocator_end_to_end",
        "module_path": "src/tac/optimization/bit_allocator_end_to_end.py",
        "coverage": "active",
        "wire_in_hook": "imports master_gradient for per-pair bit allocation",
        "notes": (
            "Cable D D3 v3 consumers 9-11 + 14 produce the canonical signals "
            "this surface consumes (LoRA targets / coding budget / engineered "
            "correction / dead-byte pruning)."
        ),
    },
    {
        "surface_id": "tac.optimization.jacobian_fisher_importance_allocator",
        "module_path": "src/tac/optimization/jacobian_fisher_importance_allocator.py",
        "coverage": "active",
        "wire_in_hook": "imports master_gradient for Fisher importance allocation",
        "notes": "Sister to bit_allocator; consumes per-byte sensitivity directly.",
    },
    {
        "surface_id": "tac.optimization.per_pair_namespace_wire_in",
        "module_path": "src/tac/optimization/per_pair_namespace_wire_in.py",
        "coverage": "active",
        "wire_in_hook": "per-pair namespace wire-in helper",
        "notes": "Per Catalog #810 namespace wave.",
    },
    {
        "surface_id": "tac.analytical_solve_extinctions",
        "module_path": "src/tac/analytical_solve_extinctions/",
        "coverage": "active",
        "wire_in_hook": "coupling_threshold_statistical_derivation consumes master_gradient",
        "notes": "Cable D D3 v3 consumer 13 (Volterra cross-terms) produces the pair-pair coupling signal.",
    },
    {
        "surface_id": "tac.boosting",
        "module_path": "src/tac/boosting/",
        "coverage": "decorator",
        "wire_in_hook": "@boosting_decorator wraps master_gradient consumers",
        "notes": "Pipeline + residual_cascade consume via decorator pattern.",
    },
    {
        "surface_id": "tac.compress_time_optimization",
        "module_path": "src/tac/compress_time_optimization/",
        "coverage": "decorator",
        "wire_in_hook": "TTO harness consumes master_gradient for per-pair coordinate search",
        "notes": "generic_tto_harness / multipass_refinement / per_pair_coordinate_search / simulated_annealing all consume.",
    },
    {
        "surface_id": "tools/cathedral_autopilot_autonomous_loop.py",
        "module_path": "tools/cathedral_autopilot_autonomous_loop.py",
        "coverage": "active",
        "wire_in_hook": "consumes via consumer 15 (OptimalPerPairTreatmentPlan) + DeliverabilityProof",
        "notes": "Hook #4 cathedral autopilot dispatch. Sister D-cable subagent owns extending to consumers 7-14.",
    },
    {
        "surface_id": "tac.optimization.pareto",
        "module_path": "src/tac/optimization/ (pareto solver)",
        "coverage": "pending",
        "wire_in_hook": "per-pair Pareto constraint emission from consumer 7 + 8 + 12",
        "notes": (
            "Cable D D3 v3 consumers 7 (Pareto envelope) + 8 (λ_R bisection) "
            "+ 12 (KKT residuals) produce the canonical signals. Sister "
            "subagent owns the wire-in implementation in tac.optimization.pareto."
        ),
    },
)


def list_analytical_surfaces() -> dict[str, object]:
    """Return the operator-facing analytical-surface coverage manifest.

    Per Cable D D2 (task #887/#890) + CLAUDE.md "PER-PAIR MASTER GRADIENT —
    wire-in coverage audit across ALL consumers" operator standing directive
    (task #810).

    Coverage state taxonomy:
      - "active": surface explicitly imports + calls tac.master_gradient
      - "indirect": consumes via tac.master_gradient_consumers
      - "decorator": uses canonical decorator pattern that wraps master_gradient
      - "pending": declared as wire-in target; sister subagent owns landing
    """
    coverage_counts: dict[str, int] = {}
    for entry in _ANALYTICAL_SURFACES:
        coverage_counts[entry["coverage"]] = coverage_counts.get(entry["coverage"], 0) + 1
    total = len(_ANALYTICAL_SURFACES)
    active = coverage_counts.get("active", 0)
    decorator = coverage_counts.get("decorator", 0)
    pending = coverage_counts.get("pending", 0)
    indirect = coverage_counts.get("indirect", 0)
    return {
        "schema": "master_gradient_analytical_surface_manifest_v1",
        "score_claim_allowed": False,
        "promotion_eligible": False,
        "evidence_grade": "[diagnostic; master-gradient coverage manifest]",
        "total_surfaces": total,
        "coverage_counts": coverage_counts,
        "coverage_fraction_active": (active + decorator + indirect) / total if total > 0 else 0.0,
        "coverage_fraction_pending": pending / total if total > 0 else 0.0,
        "surfaces": [dict(s) for s in _ANALYTICAL_SURFACES],
        "interpretation_notes": (
            "Per Cable D D2 + task #890 master-gradient wire-in coverage audit. "
            "'active' = surface directly imports tac.master_gradient. "
            "'indirect' = surface consumes via tac.master_gradient_consumers. "
            "'decorator' = surface uses canonical decorator pattern. "
            "'pending' = sister subagent owns wire-in. "
            "Cable D D3 v3 consumers 7-14 (2026-05-19) close the producer→"
            "consumer loop for the analytical surfaces named here; SISTER 3 "
            "in Cable D subagent batch owns the wire-in of those new "
            "consumers into tools/cathedral_autopilot_autonomous_loop.py + "
            "tac.optimization.pareto + tac.optimization.bit_allocator."
        ),
    }


@dataclass(frozen=True)
class ArchiveLayout:
    """Detected scored-archive grammar without claiming score authority.

    ``sections`` index the gradient subject, not necessarily the charged ZIP.
    The scored archive identity is always preserved separately so consumers
    cannot confuse differentiable payload bytes with contest-charged bytes.
    """

    grammar_name: str
    archive_sha256: str
    archive_bytes: int
    member_name: str | None
    member_sha256: str
    member_bytes: int
    gradient_subject_sha256: str
    gradient_subject_bytes: int
    gradient_byte_domain: str
    sections: tuple[ArchiveSection, ...]
    gradient_projection_supported: bool
    parser_notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "grammar_name": self.grammar_name,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "member_name": self.member_name,
            "member_sha256": self.member_sha256,
            "member_bytes": self.member_bytes,
            "gradient_subject_sha256": self.gradient_subject_sha256,
            "gradient_subject_bytes": self.gradient_subject_bytes,
            "gradient_byte_domain": self.gradient_byte_domain,
            "sections": [section.as_dict() for section in self.sections],
            "gradient_projection_supported": self.gradient_projection_supported,
            "projection_contract": archive_projection_contract(self).as_dict(),
            "parser_notes": list(self.parser_notes),
        }


@dataclass(frozen=True)
class _ArchivePayload:
    """Single contest payload extracted from a raw or ZIP archive."""

    payload: bytes
    member_name: str | None
    gradient_byte_domain: str


@dataclass(frozen=True)
class ExtractAllArchiveSpec:
    """One archive entry from an extract-all manifest."""

    label: str
    path: Path
    expected_grammar: str | None = None


class ArchiveGrammarUnknownError(ValueError):
    """Raised when archive bytes do not match a known parser grammar."""


# ---------------------------------------------------------------------------- #
# Archive parsing                                                                #
# ---------------------------------------------------------------------------- #


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _extract_gradient_subject_from_archive_bytes(archive_bytes: bytes) -> _ArchivePayload:
    """Return the single payload member that archive-layout parsers should index."""
    bio = io.BytesIO(archive_bytes)
    if not zipfile.is_zipfile(bio):
        return _ArchivePayload(
            payload=archive_bytes,
            member_name=None,
            gradient_byte_domain="scored_archive_bytes",
        )

    bio.seek(0)
    with zipfile.ZipFile(bio) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(f"zip must have exactly one payload member, got {len(infos)}")
        info = infos[0]
        if info.is_dir():
            raise ValueError(f"single ZIP entry must be a file, got directory {info.filename!r}")
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC validation failed for member {bad!r}")
        return _ArchivePayload(
            payload=zf.read(info.filename),
            member_name=info.filename,
            gradient_byte_domain="zip_inner_member_payload",
        )


def _section(
    name: str,
    offset: int,
    length: int,
    codec: str,
    payload: bytes,
    notes: str = "",
    *,
    score_affecting: bool = True,
) -> ArchiveSection:
    if offset < 0 or length < 0 or offset + length > len(payload):
        raise ValueError(
            f"bad section {name!r}: offset={offset}, length={length}, payload_bytes={len(payload)}"
        )
    return ArchiveSection(
        name=name,
        offset=offset,
        length=length,
        codec=codec,
        sha256=_sha256_bytes(payload[offset : offset + length]),
        notes=notes,
        score_affecting=score_affecting,
    )


def _packet_ir_section(
    row: dict[str, object],
    *,
    payload: bytes,
    codec_by_name: Mapping[str, str],
    notes_by_name: Mapping[str, str],
) -> ArchiveSection:
    """Convert a PacketIR consumed-byte proof row into an archive section."""

    name = str(row["name"])
    section = _section(
        name,
        int(row["offset"]),
        int(row["bytes"]),
        codec_by_name.get(name, f"packet_ir_{name}"),
        payload,
        notes_by_name.get(name, "PR106 PacketIR consumed-byte proof section."),
        score_affecting=bool(row["score_affecting"]),
    )
    row_sha = str(row["sha256"])
    if section.sha256 != row_sha:
        raise ValueError(
            f"PacketIR proof sha mismatch for {name}: section={section.sha256}, proof={row_sha}"
        )
    return section


def _validate_contiguous_sections(sections: Sequence[ArchiveSection], payload_len: int, grammar_name: str) -> None:
    expected_offset = 0
    for section in sections:
        if section.offset != expected_offset:
            raise ValueError(
                f"{grammar_name} non-contiguous section {section.name}: "
                f"offset={section.offset}, expected={expected_offset}"
            )
        expected_offset = section.end_offset
    if expected_offset != payload_len:
        raise ValueError(
            f"{grammar_name} sections end at {expected_offset}, expected payload_len={payload_len}"
        )


def _make_archive_layout(
    *,
    grammar_name: str,
    archive_bytes: bytes,
    extracted: _ArchivePayload,
    sections: Sequence[ArchiveSection],
    gradient_projection_supported: bool,
    parser_notes: Sequence[str] = (),
) -> ArchiveLayout:
    _validate_contiguous_sections(sections, len(extracted.payload), grammar_name)
    member_sha256 = _sha256_bytes(extracted.payload)
    return ArchiveLayout(
        grammar_name=grammar_name,
        archive_sha256=_sha256_bytes(archive_bytes),
        archive_bytes=len(archive_bytes),
        member_name=extracted.member_name,
        member_sha256=member_sha256,
        member_bytes=len(extracted.payload),
        gradient_subject_sha256=member_sha256,
        gradient_subject_bytes=len(extracted.payload),
        gradient_byte_domain=extracted.gradient_byte_domain,
        sections=tuple(sections),
        gradient_projection_supported=gradient_projection_supported,
        parser_notes=tuple(parser_notes),
    )


def parse_fec6_fp11_selector_archive_layout(archive_bytes: bytes) -> ArchiveLayout:
    """Parse the fec6 FP11 selector wrapper around a PR101-like inner payload."""
    extracted = _extract_gradient_subject_from_archive_bytes(archive_bytes)
    payload = extracted.payload
    if extracted.member_name not in (None, "x"):
        raise ValueError(f"fec6 FP11 expected raw payload or member 'x', got {extracted.member_name!r}")
    if len(payload) < 10 or payload[:4] != b"FP11":
        raise ValueError("not a fec6 FP11 selector payload")
    source_len = struct.unpack_from("<I", payload, 4)[0]
    pr101_minimum = 162_164 + 15_387
    if source_len < pr101_minimum:
        raise ValueError(f"fec6 source payload too short: {source_len} < {pr101_minimum}")
    source_offset = 8
    selector_len_offset = source_offset + source_len
    if selector_len_offset + 2 > len(payload):
        raise ValueError(f"fec6 source_len {source_len} exceeds payload bytes {len(payload)}")
    source_payload = payload[source_offset:selector_len_offset]
    if source_payload[:4] != b"\x1b\xcd\x03\xf8":
        raise ValueError("fec6 source payload missing expected PR101 first Brotli-stream prefix")
    selector_len = struct.unpack_from("<H", payload, selector_len_offset)[0]
    selector_offset = selector_len_offset + 2
    if selector_offset + selector_len != len(payload):
        raise ValueError(
            f"fec6 selector length mismatch: selector_offset={selector_offset}, "
            f"selector_len={selector_len}, payload_bytes={len(payload)}"
        )
    if selector_len < 6:
        raise ValueError(f"fec6 selector payload too short: {selector_len}")
    selector_magic = payload[selector_offset : selector_offset + 4]
    if selector_magic != b"FEC6":
        raise ValueError(f"fec6 selector magic mismatch: {selector_magic!r}")
    n_pairs = struct.unpack_from("<H", payload, selector_offset + 4)[0]
    if n_pairs != 600:
        raise ValueError(f"fec6 selector n_pairs mismatch: expected 600, got {n_pairs}")
    sections = (
        _section("fp11_magic", 0, 4, "raw_magic", payload, "ASCII FP11 outer-wrapper marker."),
        _section("source_len_le_u32", 4, 4, "raw_uint32_le", payload, "Length of the PR101-like inner source payload."),
        _section("source_payload", source_offset, source_len, "hnerv_ft_microcodec_payload", payload, "Differentiable inner archive consumed by codec.parse_archive."),
        _section("selector_len_le_u16", selector_len_offset, 2, "raw_uint16_le", payload, "Length of the compact hard-pair selector stream."),
        _section("selector_payload", selector_offset, selector_len, "fec6_huffman_selector", payload, "Discrete post-round selector side information; diagnostic zero-gradient in v1."),
    )
    return _make_archive_layout(
        grammar_name="fec6_fp11_selector",
        archive_bytes=archive_bytes,
        extracted=extracted,
        sections=sections,
        gradient_projection_supported=True,
        parser_notes=(
            "Existing autograd extractor supports this grammar via parse_fec6_archive_layout.",
            "Selector payload remains discrete/zero-gradient until a packet-valid operator path exists.",
        ),
    )


def parse_a1_archive_layout(archive_bytes: bytes) -> ArchiveLayout:
    """Parse A1 fine-tuned HNeRV layout with a 4-byte decoder-section header."""
    extracted = _extract_gradient_subject_from_archive_bytes(archive_bytes)
    payload = extracted.payload
    if extracted.member_name not in (None, "x"):
        raise ValueError(f"A1 expected raw payload or member 'x', got {extracted.member_name!r}")
    decoder_len = 162_164
    expected_section_total = 4 + decoder_len
    latent_len = 15_387
    if len(payload) < 4 + latent_len:
        raise ValueError("A1 payload too short for header + latent blob")
    section_total = struct.unpack_from("<I", payload, 0)[0]
    if section_total != expected_section_total:
        raise ValueError(
            f"A1 decoder_section_total {section_total} != expected {expected_section_total}"
        )
    if section_total + latent_len > len(payload):
        raise ValueError(f"bad A1 decoder_section_total {section_total}")
    if payload[4:8] != b"\x1b\xcd\x03\xf8":
        raise ValueError("A1 decoder payload missing expected first Brotli-stream prefix")
    sidecar_offset = section_total + latent_len
    sections = (
        _section("a1_section_header", 0, 4, "raw_uint32_le_section_total", payload, "uint32 LE decoder_section_total."),
        _section("decoder", 4, section_total - 4, "brotli_streams_int8", payload, "A1 fine-tuned PR101-family decoder section."),
        _section("latent", section_total, latent_len, "lzma_temporal_delta", payload, "PR101-family latent blob."),
        _section("sidecar", sidecar_offset, len(payload) - sidecar_offset, "brotli_per_pair_corrections", payload, "Per-pair correction sidecar tail."),
    )
    return _make_archive_layout(
        grammar_name="a1_finetuned",
        archive_bytes=archive_bytes,
        extracted=extracted,
        sections=sections,
        gradient_projection_supported=True,
        parser_notes=(
            "A1 is PR101-family split-Brotli with a 4-byte decoder-section header; the header itself has zero gradient.",
        ),
    )


def parse_pr101_lc_v2_archive_layout(archive_bytes: bytes) -> ArchiveLayout:
    """Parse PR101/HNeRV fixed decoder + latent + sidecar layout."""
    extracted = _extract_gradient_subject_from_archive_bytes(archive_bytes)
    payload = extracted.payload
    if extracted.member_name not in (None, "x"):
        raise ValueError(f"PR101 fixed-offset expected raw payload or member 'x', got {extracted.member_name!r}")
    decoder_len = 162_164
    latent_len = 15_387
    minimum = decoder_len + latent_len
    if len(payload) < decoder_len + latent_len:
        raise ValueError(f"PR101 payload too short: {len(payload)} < {minimum}")
    if len(payload) > 180_000:
        raise ValueError(f"PR101 fixed-offset payload unexpectedly large: {len(payload)}")
    if payload[:4] != b"\x1b\xcd\x03\xf8":
        raise ValueError("PR101 fixed-offset payload missing expected first Brotli-stream prefix")
    sidecar_offset = decoder_len + latent_len
    sections = (
        _section("decoder", 0, decoder_len, "brotli_streams_int8", payload, "Concatenated decoder tensor Brotli streams."),
        _section("latent", decoder_len, latent_len, "lzma_temporal_delta", payload, "Temporal-delta latent payload."),
        _section("sidecar", sidecar_offset, len(payload) - sidecar_offset, "brotli_per_pair_corrections", payload, "Per-pair correction sidecar tail."),
    )
    return _make_archive_layout(
        grammar_name="pr101_lc_v2",
        archive_bytes=archive_bytes,
        extracted=extracted,
        sections=sections,
        gradient_projection_supported=True,
        parser_notes=("Existing fec6/PR101 fixed-section path can project decoder/latent bytes when paired with the matching codec module.",),
    )


def parse_pr106_format0d_archive_layout(archive_bytes: bytes) -> ArchiveLayout:
    """Parse PR106 format0d sidecar archive materialized on 2026-05-15."""
    extracted = _extract_gradient_subject_from_archive_bytes(archive_bytes)
    payload = extracted.payload
    if extracted.member_name not in (None, "x"):
        raise ValueError(f"PR106 format0d expected raw payload or member 'x', got {extracted.member_name!r}")
    try:
        packet = parse_pr106_sidecar_packet(payload)
    except ValueError as exc:
        raise ValueError(f"not a PR106 format0d payload: {exc}") from exc
    if packet.format_id != PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        raise ValueError(f"not a PR106 format0d payload: format_id=0x{packet.format_id:02x}")
    proof = pr106_sidecar_consumed_byte_proof(packet)
    if proof["emitted_payload_sha256"] != _sha256_bytes(payload):
        raise ValueError("PR106 format0d PacketIR re-emit is not byte-identical to the input payload")
    if proof["all_payload_bytes_accounted"] is not True:
        raise ValueError("PR106 format0d PacketIR proof did not account for all payload bytes")
    expected_section_names = (
        "magic",
        "format_id",
        "pr106_len_le_u32",
        "pr106_payload",
        "base_format0c_sidecar_payload",
        "extra_payload_len_le_u16",
        "extra_pr101_ranked_no_op_payload",
        "extra_framing_meta",
    )
    proof_sections = proof["sections"]
    if not isinstance(proof_sections, list):
        raise ValueError("PR106 format0d PacketIR proof sections payload is malformed")
    observed_section_names = tuple(str(row["name"]) for row in proof_sections)
    if observed_section_names != expected_section_names:
        raise ValueError(
            "PR106 format0d PacketIR proof section order mismatch: "
            f"observed={observed_section_names}"
        )
    codec_by_name = {
        "magic": "raw_magic_byte",
        "format_id": "raw_format_id",
        "pr106_len_le_u32": "raw_uint32_le",
        "pr106_payload": "pr106_primary_payload",
        "base_format0c_sidecar_payload": "format0c_sidecar_payload",
        "extra_payload_len_le_u16": "raw_uint16_le",
        "extra_pr101_ranked_no_op_payload": "ranked_no_op_payload",
        "extra_framing_meta": "raw_framing_meta",
    }
    notes_by_name = {
        "magic": "PR106 PacketIR wrapper magic byte 0xfe.",
        "format_id": "PR106 PacketIR format id 0x0d.",
        "pr106_len_le_u32": "Length of the primary PR106 payload.",
        "pr106_payload": "Primary PR106 payload consumed by runtime before sidecar corrections.",
        "base_format0c_sidecar_payload": "Base format0c exact-radix ranked/no-op sidecar payload retained by format0d.",
        "extra_payload_len_le_u16": "Length of the extra ranked/no-op correction payload.",
        "extra_pr101_ranked_no_op_payload": "Second PR101 ranked/no-op correction stream applied after the base format0c pass.",
        "extra_framing_meta": "Six-byte PR101 ranked/no-op framing metadata for the extra correction stream.",
    }
    sections = tuple(
        _packet_ir_section(
            row,
            payload=payload,
            codec_by_name=codec_by_name,
            notes_by_name=notes_by_name,
        )
        for row in proof_sections
    )
    return _make_archive_layout(
        grammar_name="pr106_format0d",
        archive_bytes=archive_bytes,
        extracted=extracted,
        sections=sections,
        gradient_projection_supported=True,
        parser_notes=(
            "PR106 format0d boundary detection is PacketIR-backed and byte-identity checked via pr106_sidecar_consumed_byte_proof.",
            "Primary packed-HNeRV decoder bytes are projector-backed; discrete base/extra sidecar streams retain explicit zero-gradient v1 semantics.",
        ),
    )


def parse_dp1_archive_layout(archive_bytes: bytes) -> ArchiveLayout:
    """Parse DP1 pre-trained-driving-prior archive sections without projector authority."""
    extracted = _extract_gradient_subject_from_archive_bytes(archive_bytes)
    payload = extracted.payload
    if extracted.member_name not in (None, "0.bin"):
        raise ValueError(f"DP1 expected raw payload or member '0.bin', got {extracted.member_name!r}")
    try:
        section_ranges = parse_dp1_archive_bytes(payload)
    except ValueError as exc:
        raise ValueError(f"not a DP1 archive: {exc}") from exc
    codec_by_section = {
        "dp1_header": "raw_dp1_header",
        "codebook_blob": "dp1_codebook_blob",
        "renderer_blob": "brotli_fp16_renderer_state_dict",
        "residual_blob": "brotli_int8_per_pair_residual",
        "meta_blob": "json_metadata",
    }
    sections = tuple(
        _section(
            name,
            offset,
            length,
            codec_by_section[name],
            payload,
            f"DP1 role: {DP1_SECTION_ROLES[name]}.",
        )
        for name, (offset, length) in section_ranges.items()
    )
    return _make_archive_layout(
        grammar_name="dp1_pretrained_driving_prior",
        archive_bytes=archive_bytes,
        extracted=extracted,
        sections=sections,
        gradient_projection_supported=False,
        parser_notes=(
            "DP1 section offsets are canonical via tac.substrates.pretrained_driving_prior.archive.parse_dp1_archive_bytes.",
            "Master-gradient anchor emission remains fail-closed because the renderer stream is Brotli(pickle(state_dict)); a deterministic tensor-span serializer is required before byte-gradient authority.",
            "DP1 codebook and residual sections remain score-affecting but unprojected; they require explicit section Jacobians or zero-gradient v1 contracts before anchors are allowed.",
        ),
    )


def parse_pr106_ff_packed_archive_layout(archive_bytes: bytes) -> ArchiveLayout:
    """Parse public PR106's 0xff + uint24 packed HNeRV layout."""
    extracted = _extract_gradient_subject_from_archive_bytes(archive_bytes)
    payload = extracted.payload
    if extracted.member_name not in (None, "0.bin"):
        raise ValueError(f"PR106 packed expected raw payload or member '0.bin', got {extracted.member_name!r}")
    if len(payload) < 5 or payload[0] != 0xFF:
        raise ValueError("not a PR106 0xff packed payload")
    decoder_len = int.from_bytes(payload[1:4], "little")
    decoder_offset = 4
    decoder_end = decoder_offset + decoder_len
    if decoder_len <= 0 or decoder_end >= len(payload):
        raise ValueError(f"bad PR106 packed decoder length {decoder_len}")
    sections = (
        _section("ff_magic", 0, 1, "raw_magic_byte", payload, "PR106 packed payload marker 0xff."),
        _section("decoder_len_u24le", 1, 3, "raw_uint24_le", payload, "24-bit little-endian decoder Brotli length."),
        _section("decoder_packed_brotli", decoder_offset, decoder_len, "brotli_packed_decoder", payload, "Packed decoder stream from PR106 belt_and_suspenders."),
        _section("latents_and_sidecar_brotli", decoder_end, len(payload) - decoder_end, "brotli_latents_and_sidecar", payload, "PR106 latents plus sidecar stream."),
    )
    return _make_archive_layout(
        grammar_name="pr106_ff_packed_hnerv",
        archive_bytes=archive_bytes,
        extracted=extracted,
        sections=sections,
        gradient_projection_supported=False,
        parser_notes=(
            "Direct PR106 packed layout is detectable, but its packed Brotli sections need a dedicated projector before master-gradient anchors are legal.",
        ),
    )


def parse_hnerv_lc_v2_archive_layout(archive_bytes: bytes) -> ArchiveLayout:
    """Parse true hnerv_lc_v2 four-part length-prefixed layout."""
    extracted = _extract_gradient_subject_from_archive_bytes(archive_bytes)
    payload = extracted.payload
    if extracted.member_name not in (None, "0.bin"):
        raise ValueError(f"hnerv_lc_v2 expected raw payload or member '0.bin', got {extracted.member_name!r}")
    pos = 0
    parts: list[tuple[str, int, int, str, str]] = []
    for name, codec, notes in (
        ("decoder_brotli", "brotli_schema_decoder_int8", "Schema-driven decoder INT8 codes."),
        ("scales_fp16", "raw_fp16_scales", "One fp16 scale per tensor in schema order."),
        ("latents_brotli", "brotli_asym_delta_latents", "Per-dim asymmetric latent delta stream."),
        ("wrap_sidecar_brotli", "brotli_per_pair_sidecar", "Per-pair correction sidecar."),
    ):
        if pos + 4 > len(payload):
            raise ValueError("hnerv_lc_v2 payload truncated before length field")
        length_offset = pos
        part_len = struct.unpack_from("<I", payload, pos)[0]
        pos += 4
        if part_len <= 0 or pos + part_len > len(payload):
            raise ValueError(f"bad hnerv_lc_v2 section {name} length {part_len}")
        parts.append((f"{name}_len_le_u32", length_offset, 4, "raw_uint32_le", f"Length of {name}."))
        parts.append((name, pos, part_len, codec, notes))
        pos += part_len
    if pos != len(payload):
        raise ValueError(f"hnerv_lc_v2 trailing bytes: parsed={pos}, payload={len(payload)}")
    scales_len = parts[3][2]
    if scales_len != 56:
        raise ValueError(f"hnerv_lc_v2 expected 56B fp16 scale section, got {scales_len}")
    sections = tuple(
        _section(name, offset, length, codec, payload, notes)
        for name, offset, length, codec, notes in parts
    )
    return _make_archive_layout(
        grammar_name="hnerv_lc_v2_length_prefixed",
        archive_bytes=archive_bytes,
        extracted=extracted,
        sections=sections,
        gradient_projection_supported=False,
        parser_notes=(
            "This is the true hnerv_lc_v2 four-part packet, distinct from PR101 fixed offsets; it is xray-only until a schema-aware projector exists.",
        ),
    )


def parse_pr107_apogee_archive_layout(archive_bytes: bytes) -> ArchiveLayout:
    """Parse PR107 Apogee's three-part length-prefixed payload."""
    extracted = _extract_gradient_subject_from_archive_bytes(archive_bytes)
    payload = extracted.payload
    if extracted.member_name not in (None, "0.bin"):
        raise ValueError(f"PR107 Apogee expected raw payload or member '0.bin', got {extracted.member_name!r}")
    pos = 0
    parts: list[tuple[str, int, int, str, str]] = []
    for name, codec, notes in (
        ("meta_brotli", "brotli_json_meta", "Brotli-compressed Apogee metadata JSON."),
        ("decoder_blob", "apogee_decoder_blob", "PR107 Apogee decoder payload."),
        ("latents_brotli", "brotli_latents", "Brotli-compressed PR107 Apogee latents."),
    ):
        if pos + 4 > len(payload):
            raise ValueError("PR107 payload truncated before length field")
        length_offset = pos
        part_len = struct.unpack_from("<I", payload, pos)[0]
        pos += 4
        if part_len <= 0 or pos + part_len > len(payload):
            raise ValueError(f"bad PR107 section {name} length {part_len}")
        parts.append((f"{name}_len_le_u32", length_offset, 4, "raw_uint32_le", f"Length of {name}."))
        parts.append((name, pos, part_len, codec, notes))
        pos += part_len
    if pos != len(payload):
        raise ValueError(f"PR107 trailing bytes: parsed={pos}, payload={len(payload)}")
    sections = (
        *(
            _section(name, offset, length, codec, payload, notes)
            for name, offset, length, codec, notes in parts
        ),
    )
    return _make_archive_layout(
        grammar_name="pr107_apogee_length_prefixed",
        archive_bytes=archive_bytes,
        extracted=extracted,
        sections=sections,
        gradient_projection_supported=True,
        parser_notes=(
            "PR107 Apogee exposes a CD1 decoder section with architecture-ordered INT8 symbols and fp16/fp32 scales; decoder bytes have a registered Jacobian projector while metadata/latents remain zero-gradient v1 surfaces.",
        ),
    )


def detect_archive_grammar_and_parse(archive_bytes: bytes) -> tuple[str, ArchiveLayout]:
    """Detect the known archive grammar and return a typed layout."""
    failures: list[str] = []
    for parser in (
        parse_fec6_fp11_selector_archive_layout,
        parse_pr106_format0d_archive_layout,
        parse_dp1_archive_layout,
        parse_pr106_ff_packed_archive_layout,
        parse_hnerv_lc_v2_archive_layout,
        parse_a1_archive_layout,
        parse_pr107_apogee_archive_layout,
        parse_pr101_lc_v2_archive_layout,
    ):
        try:
            layout = parser(archive_bytes)
            return layout.grammar_name, layout
        except ValueError as exc:
            failures.append(f"{parser.__name__}: {exc}")
    raise ArchiveGrammarUnknownError("; ".join(failures))


def _load_extract_all_manifest(path: Path) -> tuple[dict[str, object], tuple[ExtractAllArchiveSpec, ...]]:
    """Load a batch xray manifest for ``extract-all``.

    Relative archive paths resolve relative to the manifest file. Accepted
    schemas are a top-level ``archives`` object-list or the object-list itself.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        manifest_meta: dict[str, object] = {}
        archive_rows = payload
    elif isinstance(payload, dict):
        manifest_meta = {key: value for key, value in payload.items() if key != "archives"}
        archive_rows = payload.get("archives")
    else:
        raise ValueError("extract-all manifest must be a JSON object or list")

    if not isinstance(archive_rows, list):
        raise ValueError("extract-all manifest must contain an 'archives' list")

    specs: list[ExtractAllArchiveSpec] = []
    for idx, row in enumerate(archive_rows):
        if not isinstance(row, dict):
            raise ValueError(f"archives[{idx}] must be an object")
        raw_path = row.get("path") or row.get("archive_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"archives[{idx}] requires a non-empty path/archive_path")
        archive_path = Path(raw_path)
        if not archive_path.is_absolute():
            archive_path = (path.parent / archive_path).resolve()
        label = row.get("label")
        if label is None:
            label = archive_path.stem
        if not isinstance(label, str) or not label:
            raise ValueError(f"archives[{idx}] label must be a non-empty string")
        expected_grammar = row.get("expected_grammar")
        if expected_grammar is not None and not isinstance(expected_grammar, str):
            raise ValueError(f"archives[{idx}] expected_grammar must be a string when present")
        specs.append(
            ExtractAllArchiveSpec(
                label=label,
                path=archive_path,
                expected_grammar=expected_grammar,
            )
        )
    return manifest_meta, tuple(specs)


def build_extract_all_manifest(manifest_path: Path) -> dict[str, object]:
    """Detect every archive in a batch manifest without emitting anchors."""
    manifest_meta, specs = _load_extract_all_manifest(manifest_path)
    archive_rows: list[dict[str, object]] = []
    counts = {
        "anchor_ready": 0,
        "detection_only_blocked": 0,
        "expected_grammar_mismatch": 0,
        "missing": 0,
        "error": 0,
    }

    for spec in specs:
        row: dict[str, object] = {
            "label": spec.label,
            "path": str(spec.path),
            "expected_grammar": spec.expected_grammar,
            "anchor_write_performed": False,
            "score_claim_allowed": False,
            "promotion_eligible": False,
        }
        if not spec.path.exists():
            row.update(
                {
                    "status": "missing",
                    "blockers": ["archive_path_missing"],
                    "error": f"archive path does not exist: {spec.path}",
                }
            )
            counts["missing"] += 1
            archive_rows.append(row)
            continue

        try:
            archive_bytes = spec.path.read_bytes()
            _, layout = detect_archive_grammar_and_parse(archive_bytes)
            layout_payload = layout.as_dict()
            projection_contract = layout_payload["projection_contract"]
            expected_match = spec.expected_grammar is None or spec.expected_grammar == layout.grammar_name
            if not expected_match:
                status = "expected_grammar_mismatch"
                blockers = [
                    f"expected_grammar={spec.expected_grammar}",
                    f"detected_grammar={layout.grammar_name}",
                ]
                counts["expected_grammar_mismatch"] += 1
            elif layout.gradient_projection_supported:
                status = "anchor_ready"
                blockers = []
                counts["anchor_ready"] += 1
            else:
                status = "detection_only_blocked"
                blockers = [
                    str(projection_contract["required_projector"]),
                    "anchor_emission_not_allowed_without_projector",
                ]
                counts["detection_only_blocked"] += 1

            row.update(
                {
                    "status": status,
                    "blockers": blockers,
                    "expected_grammar_match": expected_match,
                    "archive_sha256": layout.archive_sha256,
                    "archive_bytes": layout.archive_bytes,
                    "grammar_name": layout.grammar_name,
                    "gradient_projection_supported": layout.gradient_projection_supported,
                    "projection_contract": projection_contract,
                    "layout": layout_payload,
                }
            )
        except Exception as exc:  # pragma: no cover - exact parser text is data-dependent
            row.update(
                {
                    "status": "error",
                    "blockers": ["archive_grammar_detection_failed"],
                    "error": str(exc),
                }
            )
            counts["error"] += 1
        archive_rows.append(row)

    return {
        "schema": "master_gradient_extract_all_manifest_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_meta": manifest_meta,
        "archive_count": len(archive_rows),
        "counts": counts,
        "anchor_write_performed": False,
        "score_claim_allowed": False,
        "promotion_eligible": False,
        "archives": archive_rows,
    }


def _main_extract_all(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-detect archive grammars and projection contracts without writing anchors."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="JSON manifest listing archives to inspect.")
    parser.add_argument("--output", default=None, type=Path, help="Optional JSON output path. Defaults to stdout.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any archive is missing, mismatched, errored, or detection-only.",
    )
    args = parser.parse_args(list(argv))

    payload = build_extract_all_manifest(args.manifest)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if args.strict:
        counts = payload["counts"]
        assert isinstance(counts, dict)
        non_ready = sum(
            int(counts[key])
            for key in (
                "detection_only_blocked",
                "expected_grammar_mismatch",
                "missing",
                "error",
            )
        )
        if non_ready:
            return 2
    return 0


def _zigzag_encode_i8_to_u8(arr_i8: np.ndarray) -> np.ndarray:
    """Inverse of codec.zigzag_decode_u8."""
    arr = arr_i8.astype(np.int32)
    return np.where(arr >= 0, arr * 2, -2 * arr - 1).astype(np.uint8)


def parse_fec6_archive_layout(archive_path: Path, codec_module) -> _Fec6ArchiveLayout:
    """Parse a fec6 archive into per-byte-region metadata.

    The fec6 outer wrapper (FP11 + selector) is OPTIONAL: if present, the inner
    bytes are the source-faithful PR101 archive; if absent, archive_path IS the
    inner archive. We support both shapes so the extractor handles either the
    selector-wrapped frontier archive OR the plain pre-selector archive.
    """
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = _sha256_bytes(archive_bytes)
    n_archive_bytes = len(archive_bytes)

    # Detect FP11 outer wrapper and A1's 4-byte decoder-section header.
    has_fp11_outer = archive_bytes[:4] == b"FP11"
    has_a1_header = False
    if has_fp11_outer:
        # FP11 + 4-byte source_len + source bytes + 2-byte selector_len + selector
        source_len = struct.unpack_from("<I", archive_bytes, 4)[0]
        source_payload_offset = 8
        source_payload = archive_bytes[source_payload_offset : source_payload_offset + source_len]
        # Decoder blob lives inside source_payload (PR101 inner archive).
        inner_bytes = source_payload
        inner_base = source_payload_offset
    elif (
        len(archive_bytes) >= 4
        and struct.unpack_from("<I", archive_bytes, 0)[0] == codec_module.DECODER_BLOB_LEN + 4
    ):
        has_a1_header = True
        inner_bytes = archive_bytes[4:]
        inner_base = 4
    else:
        inner_bytes = archive_bytes
        inner_base = 0

    decoder_blob_len = codec_module.DECODER_BLOB_LEN
    latent_blob_len = codec_module.LATENT_BLOB_LEN
    decoder_blob_offset = inner_base
    latent_blob_offset = inner_base + decoder_blob_len
    sidecar_blob_offset = latent_blob_offset + latent_blob_len
    sidecar_blob_len = (inner_base + len(inner_bytes)) - sidecar_blob_offset

    # Decompress decoder bytes to find per-tensor mantissa+scale offsets.
    decoder_blob = inner_bytes[:decoder_blob_len]
    raw = codec_module.decompress_brotli_streams(
        decoder_blob, len(codec_module.DECODER_STREAM_ENDS)
    )

    # Walk DECODER_STORAGE_ORDER to map each tensor's mantissa span + fp16 scale.
    probe = codec_module.HNeRVDecoder(
        latent_dim=codec_module.LATENT_DIM,
        base_channels=codec_module.BASE_CHANNELS,
        eval_size=codec_module.EVAL_SIZE,
    )
    items = list(probe.state_dict().items())

    spans: list[_TensorByteSpan] = []
    pos = 0
    for idx in codec_module.DECODER_STORAGE_ORDER:
        name, tensor = items[idx]
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        mantissa_byte_offset = pos
        pos += numel
        scale_byte_offset = pos
        fp16_scale = float(
            np.frombuffer(raw, dtype=np.float16, count=1, offset=scale_byte_offset)[0]
        )
        pos += 2
        byte_map = codec_module.DECODER_BYTE_MAPS.get(idx, "zig")
        spans.append(
            _TensorByteSpan(
                name=name,
                storage_index=idx,
                shape=shape,
                numel=numel,
                mantissa_byte_offset=mantissa_byte_offset,
                scale_byte_offset=scale_byte_offset,
                fp16_scale=fp16_scale,
                byte_map=byte_map,
            )
        )

    if pos != len(raw):
        raise ValueError(
            f"parse_fec6_archive_layout: pos={pos} != len(raw)={len(raw)} — "
            "decoder layout decode is non-canonical (extractor and codec disagree)"
        )

    return _Fec6ArchiveLayout(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        n_archive_bytes=n_archive_bytes,
        decoder_blob_offset=decoder_blob_offset,
        decoder_blob_len=decoder_blob_len,
        decoder_tensor_spans=tuple(spans),
        decoder_raw_decompressed=raw,
        latent_blob_offset=latent_blob_offset,
        latent_blob_len=latent_blob_len,
        sidecar_blob_offset=sidecar_blob_offset,
        sidecar_blob_len=sidecar_blob_len,
        n_pairs=codec_module.N_PAIRS,
        latent_dim=codec_module.LATENT_DIM,
        base_channels=codec_module.BASE_CHANNELS,
        eval_size=tuple(codec_module.EVAL_SIZE),
        has_fp11_outer_wrapper=has_fp11_outer,
        has_a1_headered_decoder=has_a1_header,
    )


def _proof_section_by_name(packet, name: str) -> Mapping[str, object]:
    """Return a PacketIR consumed-byte proof section by name."""
    proof = pr106_sidecar_consumed_byte_proof(packet)
    sections = proof.get("sections")
    if not isinstance(sections, list):
        raise ValueError("PR106 PacketIR proof sections payload is malformed")
    for row in sections:
        if isinstance(row, Mapping) and row.get("name") == name:
            return row
    raise ValueError(f"PR106 PacketIR proof missing section {name!r}")


def _section_offset_length(row: Mapping[str, object]) -> tuple[int, int]:
    """Extract canonical integer offset/length from a section row."""
    offset = int(row.get("offset", row.get("offset_start", -1)))
    length = int(row.get("bytes", row.get("byte_count", -1)))
    if offset < 0 or length < 0:
        raise ValueError(f"bad section offset/length: {row!r}")
    return offset, length


def _decode_pr106_packed_decoder_raw(codec_module, decoder_blob: bytes) -> bytes:
    """Decode PR106 packed-HNeRV decoder bytes to q-stream + fp32 scales."""
    try:
        return brotli.decompress(decoder_blob)
    except brotli.error as legacy_error:
        if decoder_blob[:4] in (b"HDM3", b"HDM4", b"HDM6", b"HDM7", b"HDM8", b"HDM9"):
            return codec_module.decode_hdm_decoder_raw(decoder_blob)
        return codec_module.decode_pr101_schema_decoder_raw(
            decoder_blob,
            legacy_error=legacy_error,
        )


def _packed_hnerv_tensor_spans(codec_module, raw: bytes) -> tuple[_TensorByteSpan, ...]:
    """Build tensor spans for PR106 packed-HNeRV q-stream + fp32 scales."""
    packed_schema = tuple(codec_module.PACKED_STATE_SCHEMA)
    q_total = sum(int(np.prod(shape)) for _name, shape in packed_schema)
    scale_total = 4 * len(packed_schema)
    if len(raw) != q_total + scale_total:
        raise ValueError(
            "PR106 packed-HNeRV raw decoder length mismatch: "
            f"got={len(raw)} expected={q_total + scale_total}"
        )

    spans: list[_TensorByteSpan] = []
    pos = 0
    for scale_index, (name, shape) in enumerate(packed_schema):
        shape_tuple = tuple(int(dim) for dim in shape)
        numel = int(np.prod(shape_tuple))
        scale_byte_offset = q_total + 4 * scale_index
        scale = float(struct.unpack_from("<f", raw, scale_byte_offset)[0])
        spans.append(
            _TensorByteSpan(
                name=str(name),
                storage_index=-1,
                shape=shape_tuple,
                numel=numel,
                mantissa_byte_offset=pos,
                scale_byte_offset=scale_byte_offset,
                fp16_scale=scale,
                byte_map="zig",
            )
        )
        pos += numel
    return tuple(spans)


def parse_pr106_format0d_projector_layout(
    archive_path: Path,
    codec_module,
) -> _Fec6ArchiveLayout:
    """Parse PR106 format0d for its primary packed-HNeRV decoder projector.

    The discrete base/extra sidecar streams affect the decoded operating point,
    but this projector assigns zero sidecar-byte gradient in v1, matching the
    existing PR101 sidecar discipline. Operator-response rows remain the route
    for packet-valid sidecar mutations.
    """
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = _sha256_bytes(archive_bytes)
    packet = parse_pr106_sidecar_packet(archive_bytes)
    if packet.format_id != PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        raise ValueError(f"expected PR106 format0d packet, got format_id=0x{packet.format_id:02x}")
    pr106_row = _proof_section_by_name(packet, "pr106_payload")
    pr106_offset, pr106_length = _section_offset_length(pr106_row)
    if pr106_length != len(packet.pr106_bytes):
        raise ValueError("PR106 PacketIR proof length disagrees with parsed primary payload")
    pr106_payload = packet.pr106_bytes
    if len(pr106_payload) < 5 or pr106_payload[0] != 0xFF:
        raise ValueError("PR106 format0d primary payload is not packed-HNeRV 0xff layout")

    decoder_blob_len = int.from_bytes(pr106_payload[1:4], "little")
    decoder_blob_offset = pr106_offset + 4
    decoder_blob_end = decoder_blob_offset + decoder_blob_len
    if decoder_blob_len <= 0 or decoder_blob_end >= pr106_offset + pr106_length:
        raise ValueError(f"bad PR106 format0d primary decoder length {decoder_blob_len}")
    decoder_blob = pr106_payload[4 : 4 + decoder_blob_len]
    decoder_raw = _decode_pr106_packed_decoder_raw(codec_module, decoder_blob)
    spans = _packed_hnerv_tensor_spans(codec_module, decoder_raw)

    latent_blob_offset = decoder_blob_end
    latent_blob_len = pr106_length - 4 - decoder_blob_len
    sidecar_row = _proof_section_by_name(packet, "base_format0c_sidecar_payload")
    sidecar_offset, _sidecar_len = _section_offset_length(sidecar_row)
    return _Fec6ArchiveLayout(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        n_archive_bytes=len(archive_bytes),
        decoder_blob_offset=decoder_blob_offset,
        decoder_blob_len=decoder_blob_len,
        decoder_tensor_spans=spans,
        decoder_raw_decompressed=decoder_raw,
        latent_blob_offset=latent_blob_offset,
        latent_blob_len=latent_blob_len,
        sidecar_blob_offset=sidecar_offset,
        sidecar_blob_len=len(archive_bytes) - sidecar_offset,
        n_pairs=600,
        latent_dim=28,
        base_channels=36,
        eval_size=(384, 512),
        has_fp11_outer_wrapper=False,
        has_a1_headered_decoder=False,
    )


def _decode_pr107_meta(meta_brotli: bytes) -> dict[str, object]:
    """Decode PR107 Apogee's Brotli-compressed JSON metadata."""
    meta = json.loads(brotli.decompress(meta_brotli).decode("utf-8"))
    required = ("n_pairs", "latent_dim", "base_channels", "eval_size")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"PR107 Apogee metadata missing required keys: {missing}")
    return meta


def _pr107_cd1_tensor_spans(
    raw_cd1: bytes,
    meta: Mapping[str, object],
    decoder_cls,
) -> tuple[_TensorByteSpan, ...]:
    """Build tensor spans for PR107 Apogee's CD1 compact decoder payload."""
    if len(raw_cd1) < 8:
        raise ValueError("PR107 CD1 decoder payload is too short")
    if raw_cd1[:3] != b"CD1":
        raise ValueError(f"PR107 decoder is not CD1 (magic={raw_cd1[:3]!r})")
    scale_bits = raw_cd1[3]
    if scale_bits not in (16, 32):
        raise ValueError(f"unsupported PR107 CD1 scale_bits={scale_bits}")
    n_tensors = struct.unpack_from("<I", raw_cd1, 4)[0]
    eval_size_raw = meta["eval_size"]
    if not isinstance(eval_size_raw, Sequence) or len(eval_size_raw) != 2:
        raise ValueError(f"bad PR107 eval_size metadata: {eval_size_raw!r}")
    ref = decoder_cls(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=(int(eval_size_raw[0]), int(eval_size_raw[1])),
    ).state_dict()
    items = list(ref.items())
    if n_tensors != len(items):
        raise ValueError(f"PR107 CD1 n_tensors={n_tensors} != decoder state tensors={len(items)}")

    spans: list[_TensorByteSpan] = []
    pos = 8
    for storage_index, (name, tensor) in enumerate(items):
        scale_byte_offset = pos
        if scale_bits == 16:
            if pos + 2 > len(raw_cd1):
                raise ValueError(f"PR107 CD1 truncated before fp16 scale for {name}")
            fp_scale = float(np.frombuffer(raw_cd1, dtype=np.float16, count=1, offset=pos)[0])
            pos += 2
        else:
            if pos + 4 > len(raw_cd1):
                raise ValueError(f"PR107 CD1 truncated before fp32 scale for {name}")
            fp_scale = float(struct.unpack_from("<f", raw_cd1, pos)[0])
            pos += 4
        shape = tuple(int(dim) for dim in tensor.shape)
        numel = int(tensor.numel())
        mantissa_byte_offset = pos
        pos += numel
        if pos > len(raw_cd1):
            raise ValueError(f"PR107 CD1 truncated in mantissa stream for {name}")
        spans.append(
            _TensorByteSpan(
                name=str(name),
                storage_index=storage_index,
                shape=shape,
                numel=numel,
                mantissa_byte_offset=mantissa_byte_offset,
                scale_byte_offset=scale_byte_offset,
                fp16_scale=fp_scale,
                byte_map="zig",
            )
        )
    if pos != len(raw_cd1):
        raise ValueError(f"PR107 CD1 trailing bytes: parsed={pos}, payload={len(raw_cd1)}")
    return tuple(spans)


def parse_pr107_apogee_projector_layout(
    archive_path: Path,
    decoder_cls,
) -> _Fec6ArchiveLayout:
    """Parse PR107 Apogee for its CD1 decoder-byte Jacobian projector."""
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = _sha256_bytes(archive_bytes)
    detected_name, detected_layout = detect_archive_grammar_and_parse(archive_bytes)
    if detected_name != "pr107_apogee_length_prefixed":
        raise ValueError(f"expected PR107 Apogee layout, got {detected_name!r}")
    sections = {section.name: section for section in detected_layout.sections}
    meta_section = sections["meta_brotli"]
    decoder_section = sections["decoder_blob"]
    latents_section = sections["latents_brotli"]
    meta = _decode_pr107_meta(archive_bytes[meta_section.offset : meta_section.end_offset])
    decoder_blob = archive_bytes[decoder_section.offset : decoder_section.end_offset]
    raw_cd1 = brotli.decompress(decoder_blob)
    spans = _pr107_cd1_tensor_spans(raw_cd1, meta, decoder_cls)
    eval_size_raw = meta["eval_size"]
    return _Fec6ArchiveLayout(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        n_archive_bytes=len(archive_bytes),
        decoder_blob_offset=decoder_section.offset,
        decoder_blob_len=decoder_section.length,
        decoder_tensor_spans=spans,
        decoder_raw_decompressed=raw_cd1,
        latent_blob_offset=latents_section.offset,
        latent_blob_len=latents_section.length,
        sidecar_blob_offset=len(archive_bytes),
        sidecar_blob_len=0,
        n_pairs=int(meta["n_pairs"]),
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=(int(eval_size_raw[0]), int(eval_size_raw[1])),
        has_fp11_outer_wrapper=False,
        has_a1_headered_decoder=False,
    )


# ---------------------------------------------------------------------------- #
# Per-param gradient -> per-byte projection                                      #
# ---------------------------------------------------------------------------- #


def project_per_param_gradient_to_per_byte(
    layout: _Fec6ArchiveLayout,
    grad_state_dict_seg: dict[str, torch.Tensor],
    grad_state_dict_pose: dict[str, torch.Tensor],
    *,
    inner_base: int = 0,
) -> np.ndarray:
    """Project per-parameter gradient through fec6 int8+fp16 Jacobian.

    Returns: (n_archive_bytes, 3) float32 array with columns
    (seg, pose, rate_bytes_delta).

    Per-tensor Jacobian for the fec6 grammar:
        w = mantissa_byte_decoded_to_i8 * scale_fp16
        --> d(w)/d(mantissa_byte) = sign_factor * scale_fp16  (sign_factor from byte_map)
        --> d(w)/d(scale_fp16)    = mantissa_byte_decoded_to_i8  (broadcast across all weights)
        Score chain:
        d(score)/d(mantissa_byte_i) = d(score)/d(w_i) * d(w_i)/d(mantissa_byte_i)
        d(score)/d(scale)           = sum_i d(score)/d(w_i) * d(w_i)/d(scale)
                                    = sum_i grad_w[i] * decoded_byte_i

    The sign_factor accounts for codec.decode_mapped_u8 byte mappings:
      - "zig":    decoded_i8 = zigzag(byte_u8); local d(decoded)/d(byte) is +1 for even bytes
                  and -1 for odd bytes (per zigzag_decode_u8 line 226-228). For Jacobian
                  projection we use 1.0 (the absolute magnitude is what matters for ranking;
                  the sign factor merely flips which byte-delta moves the weight up vs down).
      - "negzig": decoded_i8 = -zigzag(byte_u8); same magnitude as "zig" with sign flipped.
      - "off":    decoded_i8 = byte_u8 - 128; d(decoded)/d(byte) = 1
      - "twos":   decoded_i8 = byte_u8 as i8; d(decoded)/d(byte) = 1

    We encode this conservatively: the magnitude of d(decoded)/d(byte) is always 1
    for all four mappings (zigzag flips between +1/-1 but the abs-derivative is 1).
    The sign for ranking purposes is set to +1 by convention; downstream candidate
    generators that propose specific byte modifications should consult the codec.

    Rate column is zero for byte-value sensitivities. Archive byte-count changes
    are not a derivative of an existing byte value; packet-valid operator rows
    must measure and persist `rate_bytes_delta` explicitly after rebuilding ZIP
    metadata and CRCs.
    """
    n_bytes = layout.n_archive_bytes
    G = np.zeros((n_bytes, 3), dtype=np.float32)

    # ── Decoder weights ───────────────────────────────────────────────────
    raw_decompressed = layout.decoder_raw_decompressed
    decoder_blob_offset = layout.decoder_blob_offset  # offset into archive_bytes

    # For each tensor span: decoded_i8 = decode_mapped_u8(raw[mantissa_offset:..])
    for span in layout.decoder_tensor_spans:
        if span.name not in grad_state_dict_seg or span.name not in grad_state_dict_pose:
            # Tensor has no grad (e.g., never seen during forward) — skip
            continue

        # Per-weight gradient flattened
        grad_seg_flat = grad_state_dict_seg[span.name].detach().cpu().numpy().reshape(-1).astype(np.float32)
        grad_pose_flat = grad_state_dict_pose[span.name].detach().cpu().numpy().reshape(-1).astype(np.float32)

        # The codec applies a stored permutation for conv4 tensors at storage time;
        # the gradient is in the natural model-order, the BYTES are in storage-order.
        # We need to permute the gradient to match storage layout before per-byte assignment.
        if len(span.shape) == 4 and span.storage_index in _conv4_storage_perms(codec_module=None):
            # Fall back to import to avoid circular ref — done lazily so unit-test fakes can avoid it.
            # The CONV4_STORAGE_PERMS lives on the codec_module passed at parse time; here
            # we recover it via the global cache attached to the layout object.
            perm = _LAYOUT_CONV4_STORAGE_PERMS_CACHE.get(span.storage_index)
            if perm is None:
                # Caller did not initialize the cache; this branch is only used by
                # the synthetic decoder unit tests where conv4 weights are absent.
                pass
            else:
                grad_seg_flat = (
                    grad_state_dict_seg[span.name].detach().cpu().numpy().transpose(perm).reshape(-1).astype(np.float32)
                )
                grad_pose_flat = (
                    grad_state_dict_pose[span.name].detach().cpu().numpy().transpose(perm).reshape(-1).astype(np.float32)
                )

        # Per-byte d(score)/d(byte) = d(score)/d(w) * d(w)/d(byte)
        # d(w)/d(byte_mantissa) = sign_factor * fp16_scale (|sign_factor| = 1)
        scale_mag = abs(span.fp16_scale)
        # Mantissa bytes
        mant_start_in_raw = span.mantissa_byte_offset
        mant_end_in_raw = mant_start_in_raw + span.numel
        # Map raw-decompressed offset -> archive_bytes offset: the brotli-compressed
        # bytes do NOT have a one-to-one mapping with their decompressed counterparts.
        # For ranking purposes we approximate by attributing per-tensor sensitivity
        # to the COMPRESSED tensor's byte region. Since fec6 uses a per-tensor
        # brotli stream (DECODER_STREAM_ENDS), and the compressed length varies,
        # we attribute uniformly across the tensor's compressed-byte region.
        # This is the canonical Round-2 approximation per symposium §3.2 footnote.
        # For v1 we attribute the GRAD to the DECOMPRESSED span and emit a parallel
        # ledger keyed by (tensor_name -> grad_l2_norm) so downstream candidate
        # generators can refine the mapping.
        # ──> For the master_gradient.npy output (compressed-byte indexing) we
        # ──> conservatively spread per-tensor sensitivity uniformly across the
        # ──> compressed tensor span. Mantissa-byte-grain sensitivity ranking
        # ──> remains usable for autopilot Pareto facets.

        # In the absence of a per-byte brotli-decompressed mapping, we attribute
        # the per-weight sensitivity to the corresponding RAW-DECOMPRESSED byte
        # position and emit an auxiliary per-tensor summary array. To keep the
        # invariant `G.shape == (n_archive_bytes, 3)` we project the decompressed-
        # byte sensitivity onto the compressed-byte region by uniform spreading.
        per_byte_seg = grad_seg_flat * (1.0 * scale_mag)  # |d(w)/d(byte)| = scale
        per_byte_pose = grad_pose_flat * (1.0 * scale_mag)

        # Attribute the per-mantissa-byte sensitivities into G. The decompressed
        # bytes occupy positions decoder_blob_offset + ??? in archive_bytes; we
        # conservatively place them in the FIRST decoder_blob_len bytes of the
        # archive (uniformly weighted by compressed offset). For ranking-purposes
        # the relative ordering across tensors is what matters; absolute byte
        # locations within the compressed region map to tensor-level sensitivity
        # in the auxiliary per-tensor summary.
        # ──> v1 strategy: distribute the per-mantissa-byte gradient L2-norm
        # ──> uniformly across the tensor's COMPRESSED byte region.
        compressed_per_tensor_ratio = layout.decoder_blob_len / max(len(raw_decompressed), 1)
        compressed_start = decoder_blob_offset + round(
            mant_start_in_raw * compressed_per_tensor_ratio
        )
        compressed_end = decoder_blob_offset + round(
            mant_end_in_raw * compressed_per_tensor_ratio
        )
        compressed_end = max(compressed_end, compressed_start + 1)
        compressed_end = min(compressed_end, decoder_blob_offset + layout.decoder_blob_len)

        # Uniform-spread sensitivity per byte in the compressed span.
        n_comp = compressed_end - compressed_start
        if n_comp <= 0:
            continue

        # The aggregate sensitivity for this tensor is sum of |per_byte_*|; spread
        # uniformly across the compressed region (per Round-2 fallback). We sum
        # the absolute values to compute a per-byte sensitivity magnitude; the
        # SIGN is set to +1 by convention (see docstring).
        seg_mass = float(np.abs(per_byte_seg).sum())
        pose_mass = float(np.abs(per_byte_pose).sum())
        if n_comp > 0:
            seg_per_byte = seg_mass / n_comp
            pose_per_byte = pose_mass / n_comp
            G[compressed_start:compressed_end, 0] += seg_per_byte
            G[compressed_start:compressed_end, 1] += pose_per_byte

    # ── Rate (operator-response only) ───────────────────────────────────────
    # Changing an existing byte value does not change archive byte count. Rate
    # deltas are measured on packet-valid candidates, not inferred here.
    G[:, 2] = 0.0

    return G


# Module-level cache for CONV4_STORAGE_PERMS — populated lazily by parse step
_LAYOUT_CONV4_STORAGE_PERMS_CACHE: dict[int, tuple[int, ...]] = {}


def _conv4_storage_perms(codec_module):
    if codec_module is not None and not _LAYOUT_CONV4_STORAGE_PERMS_CACHE:
        for k, v in getattr(codec_module, "CONV4_STORAGE_PERMS", {}).items():
            _LAYOUT_CONV4_STORAGE_PERMS_CACHE[k] = v
    return _LAYOUT_CONV4_STORAGE_PERMS_CACHE


# ---------------------------------------------------------------------------- #
# Forward + 3 backward passes                                                    #
# ---------------------------------------------------------------------------- #


def _stamp_decoder_with_archive_weights(decoder, decoded_state_dict):
    """Load fec6-decoded state_dict into decoder, then enable requires_grad."""
    decoder.load_state_dict(decoded_state_dict)
    for p in decoder.parameters():
        p.requires_grad_(True)
    return decoder


def _ground_truth_frame_pairs(video_path: Path, n_pairs: int, eval_size: tuple[int, int]) -> torch.Tensor:
    """Decode the first n_pairs pairs (consecutive frames) from upstream/videos/0.mkv.

    Returns: (n_pairs, 2, 3, H, W) float32 in [0, 255] at eval_size resolution.

    Per CLAUDE.md "Forbidden `make_synthetic_pair_batch` calls in any non-smoke training
    path" + Catalog #114: this extractor uses REAL video frames.
    """
    import av
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    H, W = eval_size
    frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        if len(frames) >= 2 * n_pairs:
            break
        rgb = frame.to_rgb().to_ndarray()  # (H, W, 3) uint8 at native resolution
        # Resize to eval_size for the decoder's eval resolution
        tens = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        tens_resized = F.interpolate(tens, size=(H, W), mode="bilinear", align_corners=False)
        frames.append(tens_resized.squeeze(0).numpy())
    container.close()
    if len(frames) < 2 * n_pairs:
        raise RuntimeError(f"video has only {len(frames)} frames; need 2*{n_pairs}={2*n_pairs}")
    arr = np.stack(frames[: 2 * n_pairs], axis=0)  # (2*n_pairs, 3, H, W)
    arr = arr.reshape(n_pairs, 2, 3, H, W)
    return torch.from_numpy(arr).float()


def _apply_pr107_apogee_camera_offset_roundtrip(
    rgb_tensor: torch.Tensor,
    *,
    target_h: int = 874,
    target_w: int = 1164,
) -> torch.Tensor:
    """Apply PR107 Apogee's inflate-time camera-space channel offsets.

    PR107 writes raw camera-resolution frames after bicubic upsampling and
    fixed channel offsets: frame0 red/blue -= 1, frame1 green -= 1. The generic
    master-gradient roundtrip does not know about that runtime postprocess, so
    PR107 uses this stricter path before scorer evaluation.
    """
    if rgb_tensor.dim() < 4:
        raise ValueError(
            f"PR107 Apogee roundtrip requires (..., 2, 3, H, W); got {tuple(rgb_tensor.shape)}"
        )
    if rgb_tensor.shape[-4] != 2 or rgb_tensor.shape[-3] != 3:
        raise ValueError(
            "PR107 Apogee roundtrip expects pair/channel dims (..., 2, 3, H, W); "
            f"got {tuple(rgb_tensor.shape)}"
        )
    if not rgb_tensor.is_floating_point():
        raise ValueError(f"PR107 Apogee roundtrip requires float tensor, got {rgb_tensor.dtype}")

    orig_shape = rgb_tensor.shape
    orig_h, orig_w = orig_shape[-2], orig_shape[-1]
    pair_count = int(np.prod(orig_shape[:-4])) if len(orig_shape) > 4 else 1
    pair_view = rgb_tensor.reshape(pair_count, 2, 3, orig_h, orig_w)
    flat = pair_view.reshape(pair_count * 2, 3, orig_h, orig_w)
    up = F.interpolate(flat, size=(target_h, target_w), mode="bicubic", align_corners=False)
    up_pairs = up.reshape(pair_count, 2, 3, target_h, target_w).clone()
    up_pairs[:, 0, 0].sub_(1.0)
    up_pairs[:, 0, 2].sub_(1.0)
    up_pairs[:, 1, 1].sub_(1.0)
    down = F.interpolate(
        up_pairs.reshape(pair_count * 2, 3, target_h, target_w),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    return Uint8STE.apply(down).reshape(orig_shape)


def compute_operating_point_and_per_param_gradients(
    decoder: torch.nn.Module,
    latents: torch.Tensor,  # (n_pairs, latent_dim)
    eval_size: tuple[int, int],
    gt_pair_batch: torch.Tensor,  # (n_pairs, 2, 3, H, W) in [0, 255]
    posenet,
    segnet,
    *,
    archive_bytes_count: int,
    device: torch.device,
    n_pairs_used: int = 8,
    preserve_per_pair: bool = False,
    roundtrip_mode: str = "default",
    decoder_forward_batch_size: int = 0,
) -> tuple[
    OperatingPoint,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
]:
    """Run 1 forward + 2 backward passes (or 2*n_pairs backward passes when ``preserve_per_pair``).

    Returns (operating_point, grad_seg_sd, grad_pose_sd,
             grad_seg_sd_per_pair_or_None, grad_pose_sd_per_pair_or_None).

    The 4th + 5th tuple entries are ``None`` when ``preserve_per_pair=False`` (canonical
    averaged path; 2 backward total). When ``True``, each per-pair dict value has shape
    ``(n_pairs_used, *param_shape)`` — same gradient values per pair, just not averaged
    across the pair axis. The averaged dicts are computed as ``per_pair.mean(dim=0)`` so
    consumers get both at no extra forward cost (only extra backward cost).

    For tractability on CPU we run on ``n_pairs_used`` pairs (default 8) and project the
    operating point to the full-archive scale. Per symposium §3.6: operating-point-LOCAL
    sensitivity is what we need; per-pair gradient unlocks Rashomon disagreement queue
    (Catalog #252 sister), Wyner-Ziv side-info hoisting, NSCS01 nullspace verification,
    per-pair Pareto allocation, and bootstrap variance for the K=8 ensemble.

    Following PR101 / PR95 paradigm:
      - eval_roundtrip simulated (bicubic up to 874x1164, bilinear down to 384x512)
      - rgb_to_yuv6 patched globally so PoseNet gradients flow
      - SegNet uses x[:, -1, ...] (frame_1 of each pair)
      - PoseNet uses both frames yuv6-encoded

    ``decoder_forward_batch_size`` (Catalog #218 sister mini-batch pattern, 2026-05-20
    OOM fix anchor ``fc-01KS352JAFKP2NG96KHDBGQAQS`` — T4 OOM at 600-pair full-batch
    forward needing 1.98 GiB > 759 MiB free on 14.56 GB T4 because GT pairs + scorer
    weights + accumulated graph already occupied 13.82 GiB): when > 0 and
    < ``n_pairs_used``, chunks the decoder forward + scorer forward + backward into
    smaller pair-index groups. Per-chunk gradients accumulate into the canonical
    ``grad_seg_sd`` / ``grad_pose_sd`` dicts via the math identity
    ``mean(losses_over_all_pairs) = sum_chunks(sum(losses_in_chunk) / n_total)``,
    so each chunk scales its loss by ``chunk_size / n_pairs_used`` and the autograd
    graph for that chunk is freed immediately after the chunk's backward (NO
    ``retain_graph=True``). The accumulated gradients are mathematically equivalent
    to the full-batch path within ~1e-7 floating-point associativity. When the flag
    is 0 (default) OR >= ``n_pairs_used`` (chunk == full batch), the canonical
    full-batch path is taken (backward compat for N=8 CPU smoke + small archives).

    For the per-pair path under chunking, each chunk computes per-pair losses for
    its rows and emits per-pair gradients via the standard 2*chunk_size backward
    sub-loop; the outer chunk loop concatenates the per-pair tensors. The full
    per-pair tensor shape ``(n_pairs_used, *param_shape)`` is preserved.
    """
    decoder.train()
    decoder.zero_grad()

    # Chunked path: Catalog #218 sister mini-batch when decoder_forward_batch_size > 0
    # and smaller than n_pairs_used. The canonical full-batch path follows below.
    if (
        decoder_forward_batch_size is not None
        and decoder_forward_batch_size > 0
        and decoder_forward_batch_size < n_pairs_used
    ):
        return _compute_operating_point_and_per_param_gradients_chunked(
            decoder=decoder,
            latents=latents,
            eval_size=eval_size,
            gt_pair_batch=gt_pair_batch,
            posenet=posenet,
            segnet=segnet,
            archive_bytes_count=archive_bytes_count,
            device=device,
            n_pairs_used=n_pairs_used,
            preserve_per_pair=preserve_per_pair,
            roundtrip_mode=roundtrip_mode,
            decoder_forward_batch_size=decoder_forward_batch_size,
        )

    # Subset latents + ground truth
    z = latents[:n_pairs_used].to(device).requires_grad_(False)
    gt = gt_pair_batch[:n_pairs_used].to(device)  # (n_pairs_used, 2, 3, H, W)

    # Forward: decoder produces predicted frame pairs
    decoded = decoder(z)  # (n_pairs_used, 2, 3, H, W) in [0, 255]
    if decoded.shape != gt.shape:
        raise ValueError(
            f"decoded shape {tuple(decoded.shape)} != gt shape {tuple(gt.shape)}; "
            f"eval_size {eval_size}"
        )

    # Apply contest eval roundtrip with autograd preserved.
    if roundtrip_mode == "pr107_apogee_camera_offsets":
        decoded_rt = _apply_pr107_apogee_camera_offset_roundtrip(decoded)
    elif roundtrip_mode == "default":
        decoded_rt = apply_eval_roundtrip_during_training(
            decoded, simulate_uint8=True, simulate_resize=True
        )
    else:
        raise ValueError(f"unknown roundtrip_mode={roundtrip_mode!r}")
    gt_rt = apply_eval_roundtrip_during_training(
        gt, simulate_uint8=True, simulate_resize=True
    )

    # Scorer input convention per upstream/modules.py:
    #   DistortionNet.preprocess_input expects (B, T, H, W, C) -> rearrange to (B, T, C, H, W)
    # We are already at (B, T, C, H, W) layout in [0, 255]; the scorers' preprocess methods
    # accept (B, T, C, H, W) directly (PoseNet uses (B, T*6, H/2, W/2) after yuv6, SegNet
    # uses (B, 3, 384, 512) after slicing last frame).
    # The DistortionNet path internally rearranges from BTHWC -> BTCHW; load_differentiable_scorers
    # returns (posenet, segnet) directly, so we call their preprocess_input(x) where x is BTCHW.

    posenet_in_decoded = posenet.preprocess_input(decoded_rt)
    segnet_in_decoded = segnet.preprocess_input(decoded_rt)
    posenet_in_gt = posenet.preprocess_input(gt_rt)
    segnet_in_gt = segnet.preprocess_input(gt_rt)

    posenet_out_decoded = posenet(posenet_in_decoded)
    posenet_out_gt = posenet(posenet_in_gt)
    segnet_out_decoded = segnet(segnet_in_decoded)
    segnet_out_gt = segnet(segnet_in_gt)

    # SegNet distortion: argmax disagreement rate (mean)
    # We can't backprop through argmax; use a soft surrogate via softmax KL — this is the
    # canonical PR95 / PR101 score-aware surrogate.
    # Per upstream/modules.py SegNet.compute_distortion: diff = (out1.argmax(1) != out2.argmax(1)).float().mean
    # Differentiable surrogate: soft-argmax disagreement via cross-entropy on softmax dist.
    log_p_decoded = F.log_softmax(segnet_out_decoded, dim=1)
    log_p_gt = F.log_softmax(segnet_out_gt, dim=1).detach()
    seg_kl = -(log_p_gt.exp() * log_p_decoded).sum(dim=1).mean()  # per-pixel CE, mean
    # Use raw KL value as d_seg surrogate; for the operating point we use the HARD argmax.
    with torch.no_grad():
        d_seg_hard = (segnet_out_decoded.argmax(dim=1) != segnet_out_gt.argmax(dim=1)).float().mean()

    # PoseNet distortion: MSE on first 6 pose dims (canonical per upstream/modules.py)
    if "pose" in posenet_out_decoded:
        pose_decoded = posenet_out_decoded["pose"][..., :6]
        pose_gt = posenet_out_gt["pose"][..., :6]
    else:
        # Some PoseNet variants return a tensor directly (legacy fallback)
        pose_decoded = posenet_out_decoded[..., :6]
        pose_gt = posenet_out_gt[..., :6]
    d_pose = (pose_decoded - pose_gt.detach()).pow(2).mean()
    with torch.no_grad():
        d_pose_hard = d_pose.detach().clone()

    # Rate term (analytical)
    rate = archive_bytes_count / float(CONTEST_RATE_DENOM_BYTES)

    # Build operating point with HARD scoring values (matches contest scorer S = 100*d_seg + sqrt(10*d_pose) + 25*R)
    d_seg_val = float(d_seg_hard.item())
    d_pose_val = float(d_pose_hard.item())
    if d_pose_val <= 0:
        # PoseNet MSE is essentially 0 (model fits perfectly) — bump to a tiny positive
        # so the OperatingPoint constructor doesn't reject (pose marginal undefined at 0).
        d_pose_val = 1e-12
    score_hard = 100.0 * d_seg_val + math.sqrt(10.0 * d_pose_val) + 25.0 * rate
    operating_point = OperatingPoint(
        d_seg=d_seg_val, d_pose=d_pose_val, rate=rate, score=score_hard
    )

    if not preserve_per_pair:
        # Canonical averaged path (2 backward passes total).
        # Backward pass (a): per-parameter d(d_seg_surrogate)/d(theta)
        decoder.zero_grad()
        seg_kl.backward(retain_graph=True)
        grad_seg_sd: dict[str, torch.Tensor] = {}
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                grad_seg_sd[name] = param.grad.detach().clone()
            else:
                grad_seg_sd[name] = torch.zeros_like(param)

        # Backward pass (b): per-parameter d(d_pose)/d(theta)
        decoder.zero_grad()
        d_pose.backward()
        grad_pose_sd: dict[str, torch.Tensor] = {}
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                grad_pose_sd[name] = param.grad.detach().clone()
            else:
                grad_pose_sd[name] = torch.zeros_like(param)

        return operating_point, grad_seg_sd, grad_pose_sd, None, None

    # Per-pair path (2 * n_pairs_used backward passes total).
    # Decompose the per-pair losses by collapsing all non-pair axes per-pair.
    # log_p_decoded shape: (n_pairs, classes, H', W'); same for log_p_gt
    # per_pair_seg_kl[i] = -(log_p_gt[i] * log_p_decoded[i]).sum(dim=classes).mean(dim=pixels)
    p_gt = log_p_gt.exp()
    per_pair_seg_kl = -(p_gt * log_p_decoded).sum(dim=1).mean(dim=(1, 2))  # shape (n_pairs,)
    # pose_decoded shape: (n_pairs, 6); per-pair MSE collapsed across 6 dims
    pose_diff_sq = (pose_decoded - pose_gt.detach()).pow(2)
    per_pair_d_pose = pose_diff_sq.reshape(pose_diff_sq.shape[0], -1).mean(dim=1)  # shape (n_pairs,)

    if per_pair_seg_kl.shape[0] != n_pairs_used or per_pair_d_pose.shape[0] != n_pairs_used:
        raise RuntimeError(
            f"per-pair loss shape mismatch: seg={tuple(per_pair_seg_kl.shape)} "
            f"pose={tuple(per_pair_d_pose.shape)} expected n_pairs_used={n_pairs_used}"
        )

    # Accumulate per-pair gradients into a list of per-param tensors;
    # stack at end into shape (n_pairs, *param_shape).
    per_param_names = [name for name, _ in decoder.named_parameters()]
    grad_seg_per_pair_lists: dict[str, list[torch.Tensor]] = {n: [] for n in per_param_names}
    grad_pose_per_pair_lists: dict[str, list[torch.Tensor]] = {n: [] for n in per_param_names}

    # 2 * n_pairs_used backward passes; retain_graph on all but the LAST pair's
    # final (pose) backward so the autograd graph stays alive across iterations.
    last_pair_idx = n_pairs_used - 1
    for i in range(n_pairs_used):
        # Per-pair seg backward
        decoder.zero_grad()
        per_pair_seg_kl[i].backward(retain_graph=True)
        for name, param in decoder.named_parameters():
            g = (
                param.grad.detach().clone()
                if param.grad is not None
                else torch.zeros_like(param)
            )
            grad_seg_per_pair_lists[name].append(g)

        # Per-pair pose backward — retain_graph except on the very last call
        decoder.zero_grad()
        is_final_call = i == last_pair_idx
        per_pair_d_pose[i].backward(retain_graph=not is_final_call)
        for name, param in decoder.named_parameters():
            g = (
                param.grad.detach().clone()
                if param.grad is not None
                else torch.zeros_like(param)
            )
            grad_pose_per_pair_lists[name].append(g)

    # Stack list-of-tensors into shape (n_pairs, *param_shape) per parameter
    grad_seg_sd_per_pair = {
        name: torch.stack(grad_seg_per_pair_lists[name], dim=0) for name in per_param_names
    }
    grad_pose_sd_per_pair = {
        name: torch.stack(grad_pose_per_pair_lists[name], dim=0) for name in per_param_names
    }

    # Averaged dicts derived from per-pair tensors (mathematically equivalent
    # to a single backward on the .mean() loss; numerically within ~1e-7 of
    # the canonical path due to floating-point associativity).
    grad_seg_sd = {name: v.mean(dim=0) for name, v in grad_seg_sd_per_pair.items()}
    grad_pose_sd = {name: v.mean(dim=0) for name, v in grad_pose_sd_per_pair.items()}

    return operating_point, grad_seg_sd, grad_pose_sd, grad_seg_sd_per_pair, grad_pose_sd_per_pair


def _compute_operating_point_and_per_param_gradients_chunked(
    decoder: torch.nn.Module,
    latents: torch.Tensor,  # (n_pairs, latent_dim)
    eval_size: tuple[int, int],
    gt_pair_batch: torch.Tensor,  # (n_pairs, 2, 3, H, W) in [0, 255]
    posenet,
    segnet,
    *,
    archive_bytes_count: int,
    device: torch.device,
    n_pairs_used: int,
    preserve_per_pair: bool,
    roundtrip_mode: str,
    decoder_forward_batch_size: int,
) -> tuple[
    OperatingPoint,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
]:
    """Catalog #218 sister mini-batch implementation for the master-gradient extractor.

    Math identity used: ``mean(loss_over_n_pairs) = sum_chunks(sum(loss_in_chunk) /
    n_pairs)``. Each chunk scales its summed loss by ``1 / n_pairs_used`` and calls
    ``.backward()`` WITHOUT ``retain_graph=True`` so the chunk's forward graph is
    freed immediately. Per-parameter gradients are accumulated into
    ``grad_seg_accum`` / ``grad_pose_accum`` dicts after each chunk's backward.

    Sister of `tac.substrates.d4_wyner_ziv_frame_0.architecture.WynerZivFrame0Substrate.
    reconstruct_pair(pair_indices=...)` per the 2026-05-14 D4 T4 OOM fix
    (lane_d4_oom_fix_minibatch_reconstruct_20260514, commit referenced by Catalog
    #218 docstring). The mini-batch pattern is the canonical extinction of the
    full-batch decoder-forward OOM bug class on T4 (14.56 GB) at 384x512 + 600
    pairs.

    Sub-batch OperatingPoint values (d_seg_hard, d_pose_hard) are computed as
    sum-then-divide across chunks; the operating point reflects the FULL pair
    set even though gradients are accumulated chunk-by-chunk.
    """
    # Subset latents + ground truth. Catalog #218 sister memory-pressure fix
    # (2026-05-20 OOM anchor fc-01KS36941EMJBZT0PYEADWYYW7 with chunk=100 STILL
    # OOM'd at 14.27 GiB because all 600 GT pairs + scorers + decoder + autograd
    # graph already occupied 13.82 GiB on baseline before any chunk forward
    # started). Keep both latents AND GT on CPU; transfer per-chunk inside the
    # chunk loop below. The per-chunk transfer cost is amortized over the
    # chunk's forward+backward time (~negligible vs forward FLOPs).
    z_all_cpu = latents[:n_pairs_used].detach().cpu().requires_grad_(False)
    gt_all_cpu = gt_pair_batch[:n_pairs_used].detach().cpu()

    # Initialize per-parameter accumulators (zeros_like) AFTER zero_grad
    per_param_names = [name for name, _ in decoder.named_parameters()]
    grad_seg_accum: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param) for name, param in decoder.named_parameters()
    }
    grad_pose_accum: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param) for name, param in decoder.named_parameters()
    }

    # Per-pair accumulators (only used when preserve_per_pair=True)
    grad_seg_per_pair_lists: dict[str, list[torch.Tensor]] = {n: [] for n in per_param_names}
    grad_pose_per_pair_lists: dict[str, list[torch.Tensor]] = {n: [] for n in per_param_names}

    # Operating-point scalar accumulators (sum-then-divide for d_seg_hard / d_pose_hard)
    d_seg_hard_sum = 0.0
    d_pose_hard_sum = 0.0
    chunk_count = 0
    n_total_f = float(n_pairs_used)

    chunk_size = int(decoder_forward_batch_size)
    if chunk_size <= 0 or chunk_size >= n_pairs_used:
        raise ValueError(
            f"chunked path requires 0 < chunk_size < n_pairs_used; got "
            f"chunk_size={chunk_size} n_pairs_used={n_pairs_used}"
        )

    for chunk_start in range(0, n_pairs_used, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_pairs_used)
        # Per-chunk transfer to device (avoids holding all 600 GT pairs on GPU)
        z = z_all_cpu[chunk_start:chunk_end].to(device)
        gt = gt_all_cpu[chunk_start:chunk_end].to(device)

        # Forward + scorer pipeline (mirrors the canonical full-batch path)
        decoded = decoder(z)  # (chunk_size, 2, 3, H, W) in [0, 255]
        if decoded.shape != gt.shape:
            raise ValueError(
                f"decoded shape {tuple(decoded.shape)} != gt shape {tuple(gt.shape)}; "
                f"eval_size {eval_size} (chunk_start={chunk_start} chunk_end={chunk_end})"
            )

        if roundtrip_mode == "pr107_apogee_camera_offsets":
            decoded_rt = _apply_pr107_apogee_camera_offset_roundtrip(decoded)
        elif roundtrip_mode == "default":
            decoded_rt = apply_eval_roundtrip_during_training(
                decoded, simulate_uint8=True, simulate_resize=True
            )
        else:
            raise ValueError(f"unknown roundtrip_mode={roundtrip_mode!r}")

        # GT pipeline is detached from the gradient computation (loss uses
        # log_p_gt.detach() + pose_gt.detach()). Wrapping in torch.no_grad()
        # eliminates the gt autograd graph which would otherwise consume
        # ~50% of per-chunk peak memory (encoder+decoder activations for both
        # PoseNet FastViT-T12 + SegNet UNet EfficientNet-B2 doubled). This is
        # mathematically identical because gt's contribution to the gradient
        # is already detached at loss-time.
        with torch.no_grad():
            gt_rt = apply_eval_roundtrip_during_training(
                gt, simulate_uint8=True, simulate_resize=True
            )
            posenet_in_gt = posenet.preprocess_input(gt_rt)
            segnet_in_gt = segnet.preprocess_input(gt_rt)
            posenet_out_gt = posenet(posenet_in_gt)
            segnet_out_gt = segnet(segnet_in_gt)

        posenet_in_decoded = posenet.preprocess_input(decoded_rt)
        segnet_in_decoded = segnet.preprocess_input(decoded_rt)

        posenet_out_decoded = posenet(posenet_in_decoded)
        segnet_out_decoded = segnet(segnet_in_decoded)

        # SegNet differentiable surrogate (per-pixel CE; per-chunk sum-then-mean)
        log_p_decoded = F.log_softmax(segnet_out_decoded, dim=1)
        log_p_gt = F.log_softmax(segnet_out_gt, dim=1).detach()
        p_gt = log_p_gt.exp()
        # Per-pair sum of -(p_gt * log_p_decoded).sum(classes).mean(pixels)
        per_pair_seg_kl = -(p_gt * log_p_decoded).sum(dim=1).mean(dim=(1, 2))
        # Chunk's contribution to the full mean: sum(per_pair) / n_pairs_used
        seg_kl_chunk_contribution = per_pair_seg_kl.sum() / n_total_f

        # SegNet hard distortion (no-grad; argmax disagreement)
        with torch.no_grad():
            d_seg_hard_chunk = (
                segnet_out_decoded.argmax(dim=1) != segnet_out_gt.argmax(dim=1)
            ).float().mean()
            d_seg_hard_sum += float(d_seg_hard_chunk.item()) * (chunk_end - chunk_start)

        # PoseNet pose extraction + MSE
        if "pose" in posenet_out_decoded:
            pose_decoded = posenet_out_decoded["pose"][..., :6]
            pose_gt = posenet_out_gt["pose"][..., :6]
        else:
            pose_decoded = posenet_out_decoded[..., :6]
            pose_gt = posenet_out_gt[..., :6]
        pose_diff_sq = (pose_decoded - pose_gt.detach()).pow(2)
        # Per-pair MSE collapsed across 6 dims
        per_pair_d_pose = pose_diff_sq.reshape(pose_diff_sq.shape[0], -1).mean(dim=1)
        d_pose_chunk_contribution = per_pair_d_pose.sum() / n_total_f

        with torch.no_grad():
            d_pose_hard_sum += float(per_pair_d_pose.mean().item()) * (chunk_end - chunk_start)

        if not preserve_per_pair:
            # Backward (a): seg surrogate — accumulate gradient into grad_seg_accum
            decoder.zero_grad()
            seg_kl_chunk_contribution.backward(retain_graph=True)
            for name, param in decoder.named_parameters():
                if param.grad is not None:
                    grad_seg_accum[name] = grad_seg_accum[name] + param.grad.detach().clone()

            # Backward (b): pose — accumulate gradient into grad_pose_accum
            decoder.zero_grad()
            d_pose_chunk_contribution.backward()
            for name, param in decoder.named_parameters():
                if param.grad is not None:
                    grad_pose_accum[name] = grad_pose_accum[name] + param.grad.detach().clone()
        else:
            # Per-pair path within this chunk: 2 * chunk_size backward passes,
            # each weighted by 1/n_pairs_used so the accumulated per-pair
            # tensor reflects the per-pair gradient (NOT scaled by chunk-size).
            # Per-pair tensors are unweighted (raw per-pair gradients); the
            # averaged dicts derive from per_pair.mean(dim=0) after the loop.
            chunk_pair_count = chunk_end - chunk_start
            last_pair_in_chunk = chunk_pair_count - 1
            for i in range(chunk_pair_count):
                # Per-pair seg backward (unweighted: raw per-pair contribution)
                decoder.zero_grad()
                per_pair_seg_kl[i].backward(retain_graph=True)
                for name, param in decoder.named_parameters():
                    g = (
                        param.grad.detach().clone()
                        if param.grad is not None
                        else torch.zeros_like(param)
                    )
                    grad_seg_per_pair_lists[name].append(g)

                decoder.zero_grad()
                is_final_in_chunk = i == last_pair_in_chunk
                per_pair_d_pose[i].backward(retain_graph=not is_final_in_chunk)
                for name, param in decoder.named_parameters():
                    g = (
                        param.grad.detach().clone()
                        if param.grad is not None
                        else torch.zeros_like(param)
                    )
                    grad_pose_per_pair_lists[name].append(g)

        # Free the chunk's forward graph by dropping references
        del decoded, decoded_rt, gt_rt
        del posenet_out_decoded, posenet_out_gt, segnet_out_decoded, segnet_out_gt
        del posenet_in_decoded, posenet_in_gt, segnet_in_decoded, segnet_in_gt
        del log_p_decoded, log_p_gt, p_gt, per_pair_seg_kl, seg_kl_chunk_contribution
        del pose_decoded, pose_gt, pose_diff_sq, per_pair_d_pose, d_pose_chunk_contribution
        if device.type == "cuda":
            torch.cuda.empty_cache()
        chunk_count += 1

    # Finalize OperatingPoint
    d_seg_val = d_seg_hard_sum / n_total_f
    d_pose_val = d_pose_hard_sum / n_total_f
    if d_pose_val <= 0:
        d_pose_val = 1e-12
    rate = archive_bytes_count / float(CONTEST_RATE_DENOM_BYTES)
    score_hard = 100.0 * d_seg_val + math.sqrt(10.0 * d_pose_val) + 25.0 * rate
    operating_point = OperatingPoint(
        d_seg=d_seg_val, d_pose=d_pose_val, rate=rate, score=score_hard
    )

    if not preserve_per_pair:
        return operating_point, grad_seg_accum, grad_pose_accum, None, None

    # Per-pair path: stack per-param lists into (n_pairs, *param_shape)
    grad_seg_sd_per_pair = {
        name: torch.stack(grad_seg_per_pair_lists[name], dim=0) for name in per_param_names
    }
    grad_pose_sd_per_pair = {
        name: torch.stack(grad_pose_per_pair_lists[name], dim=0) for name in per_param_names
    }
    grad_seg_sd = {name: v.mean(dim=0) for name, v in grad_seg_sd_per_pair.items()}
    grad_pose_sd = {name: v.mean(dim=0) for name, v in grad_pose_sd_per_pair.items()}
    return (
        operating_point,
        grad_seg_sd,
        grad_pose_sd,
        grad_seg_sd_per_pair,
        grad_pose_sd_per_pair,
    )


# ---------------------------------------------------------------------------- #
# CLI                                                                            #
# ---------------------------------------------------------------------------- #


def _add_inflate_src_to_path(inflate_py: Path) -> None:
    """Ensure the submission_dir/src is on sys.path so we can import codec + model."""
    src_dir = inflate_py.parent / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"expected {src_dir} alongside {inflate_py}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _default_hardware_substrate(device: str) -> str:
    """Return a non-authoritative-by-default hardware tag for the current host."""
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    if system == "darwin":
        return f"darwin_{machine}_local_{device}_advisory"
    if system == "linux":
        return f"linux_{machine}_{device}"
    return f"{system}_{machine}_{device}_unknown"


def _axis_is_authoritative(axis: str) -> bool:
    return axis in {"[contest-CPU]", "[contest-CUDA]"}


def _validate_measurement_authority(
    *,
    axis: str,
    device: str,
    hardware_substrate: str,
    n_pairs_used: int,
    n_pairs_total: int,
) -> None:
    """Fail closed when diagnostic or advisory extraction is labeled as contest authority."""
    if n_pairs_used <= 0 or n_pairs_total <= 0 or n_pairs_used > n_pairs_total:
        raise SystemExit(
            f"invalid pair counts: n_pairs_used={n_pairs_used}, n_pairs_total={n_pairs_total}"
        )
    hardware_lower = hardware_substrate.lower()
    if _axis_is_authoritative(axis) and n_pairs_used != n_pairs_total:
        raise SystemExit(
            f"{axis} master-gradient anchors require the full pair set "
            f"(n_pairs_used={n_pairs_used}, n_pairs_total={n_pairs_total}); "
            "use [diagnostic-CPU], [diagnostic-CUDA], or [macOS-CPU advisory] "
            "for subset probes."
        )
    if _axis_is_authoritative(axis) and any(
        token in hardware_lower for token in ("advisory", "darwin", "macos", "mps")
    ):
        raise SystemExit(
            f"{axis} cannot be written from advisory hardware_substrate={hardware_substrate!r}"
        )
    if axis == "[contest-CUDA]" and device != "cuda":  # CUSTODY_VALIDATOR_OK:this_function_IS_master_gradient_axis_device_validator_raising_SystemExit_on_axis_device_mismatch_per_comprehensive_bug_audit_cascade_20260526
        raise SystemExit("[contest-CUDA] anchors require --device cuda")
    if axis == "[contest-CPU]" and device != "cpu":  # CUSTODY_VALIDATOR_OK:this_function_IS_master_gradient_axis_device_validator_raising_SystemExit_on_axis_device_mismatch_per_comprehensive_bug_audit_cascade_20260526
        raise SystemExit("[contest-CPU] anchors require --device cpu")
    if axis == "[macOS-CPU advisory]" and "darwin" not in hardware_lower:
        raise SystemExit(
            "[macOS-CPU advisory] anchors require a darwin/macos hardware_substrate"
        )


def _maybe_extract_inner_archive_from_zip(zip_path: Path) -> bytes:
    """Return canonical contest payload bytes from a ZIP, or raw file bytes otherwise."""
    return _extract_gradient_subject_from_archive_bytes(zip_path.read_bytes()).payload


def _apply_pr106_sidecar_correction_passes(latents: torch.Tensor, packet) -> torch.Tensor:
    """Apply PR106 PacketIR correction passes exactly as the format0d runtime does."""
    corrected = latents.clone()
    for dims, deltas in decode_pr106_sidecar_packet_correction_passes(packet):
        if len(dims) < corrected.shape[0] or len(deltas) < corrected.shape[0]:
            raise ValueError(
                "PR106 sidecar correction arrays shorter than decoded latents: "
                f"dims={len(dims)} deltas={len(deltas)} latents={corrected.shape[0]}"
            )
        for pair_index in range(corrected.shape[0]):
            dim = int(dims[pair_index])
            if dim == 255:
                continue
            if dim < 0 or dim >= corrected.shape[1]:
                raise ValueError(f"PR106 sidecar dim {dim} outside latent width {corrected.shape[1]}")
            corrected[pair_index, dim] = corrected[pair_index, dim] + float(deltas[pair_index]) * 0.01
    return corrected


def _serialize_archive_to_temp(raw_bytes: bytes, scratch_dir: Path) -> Path:
    """Write raw archive bytes to a scratch file for layout parsing."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    target = scratch_dir / "archive.bin"
    target.write_bytes(raw_bytes)
    return target


def _write_layout_contract(path: Path, layout: ArchiveLayout) -> None:
    """Persist a grammar/xray manifest before any projector-authority decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] == "extract-all":
        return _main_extract_all(argv_list[1:])
    if argv_list and argv_list[0] == "list-grammars":
        argv_list = ["--list-grammars", *argv_list[1:]]
    if argv_list and argv_list[0] == "list-analytical-surfaces":
        # Cable D D2 (task #887/#890) — emit the analytical-surface coverage
        # manifest and exit. JSON-formatted so cathedral autopilot ranker
        # can consume.
        print(json.dumps(list_analytical_surfaces(), indent=2, sort_keys=True))
        return 0

    parser = argparse.ArgumentParser(
        description="Extract master gradient (per-byte score sensitivity) for an archive at its operating point."
    )
    parser.add_argument("--archive", default=None, type=Path, help="Path to fec6 archive.zip or raw archive bytes")
    parser.add_argument(
        "--list-grammars",
        action="store_true",
        help="Print supported/detection-only archive grammar contracts and exit.",
    )
    parser.add_argument(
        "--inflate-py",
        default=None,
        type=Path,
        help="Path to submission_dir/inflate.py (required unless --detect-grammar-only)",
    )
    parser.add_argument(
        "--upstream-dir",
        default=None,
        type=Path,
        help="Path to upstream repository root (required unless --detect-grammar-only)",
    )
    parser.add_argument(
        "--axis",
        default=None,
        choices=[
            "[contest-CPU]",
            "[contest-CUDA]",
            "[diagnostic-CPU]",
            "[diagnostic-CUDA]",
            "[macOS-CPU advisory]",
            "[MPS-PROXY]",
        ],
        help="Score axis tag. Required unless --detect-grammar-only; contest tags require full pair count and authoritative hardware.",
    )
    parser.add_argument(
        "--output-npy",
        default=None,
        type=Path,
        help="Sidecar .npy path for the (n_bytes, 3) gradient array (required unless --detect-grammar-only)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device (cpu for Modal CPU dispatch)")
    parser.add_argument("--video-path", default=None, type=Path, help="GT video path (defaults to upstream/videos/0.mkv)")
    parser.add_argument("--n-pairs-used", type=int, default=8, help="Pairs to use for forward+backward (CPU economics; default 8)")
    parser.add_argument("--n-pairs-total", type=int, default=600, help="Total contest pair count for authority checks (default 600)")
    parser.add_argument("--call-id", default=None, help="Optional dispatch call_id for the ledger anchor")
    parser.add_argument("--hardware-substrate", default=None, help="Hardware substrate tag (default: derived from device)")
    parser.add_argument("--scratch-dir", default=None, type=Path, help="Scratch dir for raw archive bytes (default: alongside output-npy)")
    parser.add_argument(
        "--detect-grammar-only",
        action="store_true",
        help="Print detected archive grammar/layout JSON and exit without scorer or codec imports.",
    )
    parser.add_argument(
        "--layout-contract-output",
        default=None,
        type=Path,
        help=(
            "Optional JSON path for the detected archive layout plus projection authority contract. "
            "Written before fail-closed unsupported-grammar exits."
        ),
    )
    parser.add_argument("--no-anchor-write", action="store_true", help="Skip writing the ledger anchor (smoke / dry-run mode)")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress logging")
    parser.add_argument(
        "--preserve-per-pair",
        action="store_true",
        help=(
            "Also emit (N_bytes, N_pairs, 3) per-pair gradient as a sister .npy + "
            "ledger anchor. Cost: 2*N_pairs backward passes instead of 2 (N=8 ~1 min, "
            "N=600 ~2-3h on M5 Max). Per-pair tensor unlocks Rashomon disagreement queue, "
            "Wyner-Ziv side-info hoisting, NSCS01 nullspace verification, per-pair Pareto "
            "allocation, and bootstrap variance for the K=8 ensemble (Catalog #252 sister)."
        ),
    )
    parser.add_argument(
        "--per-pair-output-npy",
        default=None,
        type=Path,
        help=(
            "Per-pair sidecar .npy path (only used with --preserve-per-pair). "
            "Default: derived from --output-npy by inserting '_per_pair_<N>pair' "
            "before .npy. Forbidden under /tmp per Catalog #220."
        ),
    )
    parser.add_argument(
        "--compute-dtype",
        default="float32",
        choices=["float32", "float64"],
        help=(
            "Autograd compute precision (default float32). float64 doubles per-op "
            "precision (~7 decimal digits → ~15) but on Apple Silicon CPU fp64 is "
            "scalar-only (no NEON SIMD) so ~4x wall-clock vs fp32. Use float64 for "
            "the canonical max-precision anchor when extra decimal places at the "
            "small-magnitude end matter (Hessian-vector products, Wyner-Ziv residual "
            "amplitude estimates, per-pair Pareto envelope numerics)."
        ),
    )
    parser.add_argument(
        "--storage-dtype",
        default=None,
        choices=["float32", "float64"],
        help=(
            "Sidecar .npy storage precision (default: matches --compute-dtype). "
            "Use 'float32' with --compute-dtype float64 to get fp64 accumulation + "
            "fp32 final cast (precision lost only at the final cast; halves disk)."
        ),
    )
    parser.add_argument(
        "--decoder-forward-batch-size",
        type=int,
        default=0,
        help=(
            "Catalog #218 sister mini-batch chunk size for the decoder forward + "
            "scorer forward + backward loop. Default 0 = full batch (canonical "
            "path for small N=8 CPU smokes). Set to 50-100 for T4 (14.56 GB) at "
            "600 pairs (canonical fix for 2026-05-20 OOM anchor "
            "fc-01KS352JAFKP2NG96KHDBGQAQS: 600-pair full-batch forward needed "
            "1.98 GiB but only 759 MiB free after GT/scorer/decoder allocations). "
            "Gradients accumulate per-chunk via the math identity "
            "mean(losses_over_n) = sum_chunks(sum(losses_in_chunk) / n_total); "
            "chunk graph is freed immediately after each chunk's backward (NO "
            "retain_graph). Math equivalent to full-batch within ~1e-7 fp "
            "associativity. Verified by --decoder-forward-batch-size local smoke."
        ),
    )

    args = parser.parse_args(argv_list)
    if args.storage_dtype is None:
        args.storage_dtype = args.compute_dtype

    if args.list_grammars:
        print(json.dumps(list_archive_grammar_contracts(), sort_keys=True))
        return 0
    if args.archive is None:
        raise SystemExit("--archive is required unless --list-grammars is set")

    if args.output_npy is not None and args.output_npy.is_absolute() and str(args.output_npy).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise SystemExit(
            f"--output-npy {args.output_npy} forbidden under /tmp per CLAUDE.md "
            "'Forbidden /tmp paths in any persisted artifact' (Catalog #220 + transient-evidence trap)"
        )

    scored_archive_bytes = args.archive.read_bytes()
    detected_name, detected_layout = detect_archive_grammar_and_parse(scored_archive_bytes)
    if args.layout_contract_output is not None:
        _write_layout_contract(args.layout_contract_output, detected_layout)
    if args.detect_grammar_only:
        print(json.dumps(detected_layout.as_dict(), sort_keys=True))
        return 0

    if args.axis is None:
        raise SystemExit("--axis is required unless --detect-grammar-only is set")
    if args.output_npy is None:
        raise SystemExit("--output-npy is required unless --detect-grammar-only is set")
    if args.inflate_py is None:
        raise SystemExit("--inflate-py is required unless --detect-grammar-only is set")
    if args.upstream_dir is None:
        raise SystemExit("--upstream-dir is required unless --detect-grammar-only is set")
    if detected_name not in {
        "fec6_fp11_selector",
        "pr101_lc_v2",
        "a1_finetuned",
        "pr106_format0d",
        "pr107_apogee_length_prefixed",
    }:
        raise SystemExit(
            f"archive grammar {detected_name!r} is xray/detection-only in this extractor; "
            "byte-gradient authority is currently implemented only for fec6_fp11_selector, "
            "PR101-family fixed/headered layouts, PR106 format0d primary packed-HNeRV "
            "decoder bytes, and PR107 Apogee CD1 decoder bytes with matching codec.py. "
            "Re-run with --detect-grammar-only "
            "or wire a grammar-aware projector before emitting a master-gradient anchor."
        )

    hardware_substrate = args.hardware_substrate or _default_hardware_substrate(args.device)
    _validate_measurement_authority(
        axis=args.axis,
        device=args.device,
        hardware_substrate=hardware_substrate,
        n_pairs_used=args.n_pairs_used,
        n_pairs_total=args.n_pairs_total,
    )

    _add_inflate_src_to_path(args.inflate_py)
    import codec as codec_module  # type: ignore[import-not-found]
    from model import HNeRVDecoder  # type: ignore[import-not-found]
    # Populate the CONV4 permutation cache
    _conv4_storage_perms(codec_module)

    scored_archive_sha256 = _sha256_bytes(scored_archive_bytes)
    raw_archive_bytes = _maybe_extract_inner_archive_from_zip(args.archive)
    gradient_subject_sha256 = _sha256_bytes(raw_archive_bytes)
    gradient_byte_domain = (
        "scored_archive_bytes"
        if raw_archive_bytes == scored_archive_bytes
        else "zip_inner_member_payload"
    )
    scratch_dir = args.scratch_dir or args.output_npy.parent / "_extractor_scratch"
    scratch_archive = _serialize_archive_to_temp(raw_archive_bytes, scratch_dir)

    print(
        f"[master-gradient] charged archive sha256={scored_archive_sha256[:16]}... "
        f"bytes={len(scored_archive_bytes)}"
    )
    print(
        f"[master-gradient] gradient subject sha256={gradient_subject_sha256[:16]}... "
        f"bytes={len(raw_archive_bytes)} domain={gradient_byte_domain}"
    )
    print(
        f"[master-gradient] detected grammar={detected_layout.grammar_name} "
        f"projection_supported={detected_layout.gradient_projection_supported} "
        f"sections={len(detected_layout.sections)}"
    )
    print(f"[master-gradient] parsing archive layout from {scratch_archive} ({len(raw_archive_bytes)} bytes)")
    if detected_name == "pr106_format0d":
        layout = parse_pr106_format0d_projector_layout(scratch_archive, codec_module)
    elif detected_name == "pr107_apogee_length_prefixed":
        layout = parse_pr107_apogee_projector_layout(scratch_archive, HNeRVDecoder)
    else:
        layout = parse_fec6_archive_layout(scratch_archive, codec_module)
    if layout.archive_sha256 != gradient_subject_sha256:
        raise RuntimeError(
            "layout archive sha mismatch: "
            f"layout={layout.archive_sha256}, subject={gradient_subject_sha256}"
        )

    print(f"[master-gradient] layout subject sha256={layout.archive_sha256[:16]}... n_bytes={layout.n_archive_bytes}")
    print(f"[master-gradient] decoder_blob {layout.decoder_blob_len}B, latent_blob {layout.latent_blob_len}B, sidecar {layout.sidecar_blob_len}B")
    print(f"[master-gradient] n_pairs={layout.n_pairs} eval_size={layout.eval_size} latent_dim={layout.latent_dim}")
    roundtrip_mode = (
        "pr107_apogee_camera_offsets"
        if detected_name == "pr107_apogee_length_prefixed"
        else "default"
    )
    if roundtrip_mode != "default":
        print(f"[master-gradient] runtime roundtrip mode={roundtrip_mode}")

    device = torch.device(args.device)

    # Patch upstream YUV6 BEFORE loading scorers (CLAUDE.md eval_roundtrip non-negotiable)
    print("[master-gradient] patching upstream rgb_to_yuv6 for autograd preservation (PR95 fix)")
    patch_token = patch_upstream_yuv6_globally()

    try:
        # Resolve compute dtype per operator's max-precision directive (2026-05-17)
        compute_torch_dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32
        storage_np_dtype = np.float64 if args.storage_dtype == "float64" else np.float32
        print(
            f"[master-gradient] compute_dtype={args.compute_dtype} "
            f"storage_dtype={args.storage_dtype} "
            f"(fp64 compute ~4x wall-clock on Apple Silicon CPU; "
            f"~15 decimal digits vs fp32's ~7)"
        )

        print(f"[master-gradient] loading scorers from {args.upstream_dir}")
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        # Cast scorers to compute dtype for autograd-precision parity with decoder
        posenet = posenet.to(dtype=compute_torch_dtype)
        segnet = segnet.to(dtype=compute_torch_dtype)

        # Parse decoder state_dict + latents from archive (using submitter codec)
        print("[master-gradient] decoding archive via canonical codec parser")
        if detected_name == "pr106_format0d":
            packet = parse_pr106_sidecar_packet(raw_archive_bytes)
            decoder_sd, latents, meta = codec_module.parse_packed_archive(packet.pr106_bytes)
            latents = _apply_pr106_sidecar_correction_passes(latents, packet)
        elif layout.has_fp11_outer_wrapper:
            # parse_archive expects inner bytes
            source_payload = raw_archive_bytes[8 : 8 + struct.unpack_from("<I", raw_archive_bytes, 4)[0]]
            decoder_sd, latents, meta = codec_module.parse_archive(source_payload)
        elif layout.has_a1_headered_decoder:
            section_total = struct.unpack_from("<I", raw_archive_bytes, 0)[0]
            decoder_blob = raw_archive_bytes[4:section_total]
            latent_blob = raw_archive_bytes[section_total : section_total + codec_module.LATENT_BLOB_LEN]
            sidecar_blob = raw_archive_bytes[section_total + codec_module.LATENT_BLOB_LEN :]
            decoder_sd = codec_module.decode_decoder_compact(decoder_blob)
            latents = codec_module.apply_latent_sidecar(
                codec_module.decode_latents_compact(latent_blob),
                sidecar_blob,
            )
            meta = {
                "latent_dim": codec_module.LATENT_DIM,
                "base_channels": codec_module.BASE_CHANNELS,
                "eval_size": list(codec_module.EVAL_SIZE),
            }
        else:
            decoder_sd, latents, meta = codec_module.parse_archive(raw_archive_bytes)

        decoder = HNeRVDecoder(
            latent_dim=meta["latent_dim"],
            base_channels=meta["base_channels"],
            eval_size=tuple(meta["eval_size"]),
        ).to(device).to(dtype=compute_torch_dtype)
        _stamp_decoder_with_archive_weights(decoder, decoder_sd)

        video_path = args.video_path or (args.upstream_dir / "videos" / "0.mkv")
        print(f"[master-gradient] loading {args.n_pairs_used} ground-truth pairs from {video_path}")
        gt_pairs = _ground_truth_frame_pairs(video_path, args.n_pairs_used, tuple(meta["eval_size"])).to(dtype=compute_torch_dtype)

        latents_tensor = latents.to(device).to(dtype=compute_torch_dtype)

        backward_count = (2 * args.n_pairs_used) if args.preserve_per_pair else 2
        print(
            f"[master-gradient] running 1 forward + {backward_count} backward passes "
            f"({'per-pair' if args.preserve_per_pair else 'averaged'} mode; "
            f"n_pairs_used={args.n_pairs_used})"
        )
        t0 = time.time()
        (
            op,
            grad_seg_sd,
            grad_pose_sd,
            grad_seg_sd_per_pair,
            grad_pose_sd_per_pair,
        ) = compute_operating_point_and_per_param_gradients(
            decoder=decoder,
            latents=latents_tensor,
            eval_size=tuple(meta["eval_size"]),
            gt_pair_batch=gt_pairs,
            posenet=posenet,
            segnet=segnet,
            archive_bytes_count=len(scored_archive_bytes),
            device=device,
            n_pairs_used=args.n_pairs_used,
            preserve_per_pair=args.preserve_per_pair,
            roundtrip_mode=roundtrip_mode,
            decoder_forward_batch_size=args.decoder_forward_batch_size,
        )
        fwd_bwd_secs = time.time() - t0
        print(f"[master-gradient] forward+{backward_count}-backward done in {fwd_bwd_secs:.2f}s")
        print(f"[master-gradient] operating point: d_seg={op.d_seg:.4f} d_pose={op.d_pose:.6g} rate={op.rate:.6f} score={op.score:.4f}")

        seg_marg, pose_marg, rate_per_byte = compute_marginal_coefficients(op)
        print(f"[master-gradient] marginals: dS/d_seg={seg_marg:.1f} dS/d_pose={pose_marg:.2f} dS/d_byte={rate_per_byte:.3e}")

        measurement_method_base = (
            "autograd_per_parameter_projected_pr106_format0d_primary_packed_hnerv_decoder_jacobian_sidecar_zero_grad_v1"
            if detected_name == "pr106_format0d"
            else "autograd_per_parameter_projected_pr107_apogee_cd1_decoder_jacobian_camera_offset_roundtrip_latents_zero_grad_v1"
            if detected_name == "pr107_apogee_length_prefixed"
            else "autograd_per_parameter_projected_fec6_int8_fp16_jacobian"
        )
        print(f"[master-gradient] projecting per-parameter grad to per-byte ({measurement_method_base})")
        G = project_per_param_gradient_to_per_byte(
            layout, grad_seg_sd, grad_pose_sd, inner_base=0
        )

        # Sanity: shape and finite
        if G.shape != (layout.n_archive_bytes, 3):
            raise RuntimeError(f"projected gradient shape {G.shape} != ({layout.n_archive_bytes}, 3)")
        nans = int(np.isnan(G).sum())
        infs = int(np.isinf(G).sum())
        if nans or infs:
            raise RuntimeError(f"projected gradient has {nans} NaN + {infs} Inf entries")

        # Write sidecar — cast to storage_dtype per --storage-dtype contract
        args.output_npy.parent.mkdir(parents=True, exist_ok=True)
        G_to_save = G.astype(storage_np_dtype, copy=False) if G.dtype != storage_np_dtype else G
        np.save(args.output_npy, G_to_save)
        print(f"[master-gradient] wrote sidecar {args.output_npy} ({G_to_save.nbytes} bytes; shape={G_to_save.shape}; dtype={G_to_save.dtype})")

        # Optional ledger anchor
        if not args.no_anchor_write:
            grad = MasterGradient(
                archive_sha256=scored_archive_sha256,
                operating_point=op,
                gradient_array_path=str(args.output_npy.resolve()),
                n_bytes=layout.n_archive_bytes,
                measurement_method=measurement_method_base,
                measurement_axis=args.axis,
                measurement_hardware=hardware_substrate,
                measurement_call_id=args.call_id,
                measurement_utc=datetime.now(UTC).isoformat(),
                pareto_facets=(),
                rashomon_disagreement_score=None,
                scored_archive_sha256=scored_archive_sha256,
                scored_archive_bytes=len(scored_archive_bytes),
                gradient_subject_sha256=gradient_subject_sha256,
                gradient_subject_bytes=layout.n_archive_bytes,
                gradient_byte_domain=gradient_byte_domain,
                n_pairs_used=args.n_pairs_used,
                n_pairs_total=args.n_pairs_total,
                score_axis_dominance=score_axis_dominance_summary(G_to_save, op),
            )
            append_anchor_locked(grad)
            print(f"[master-gradient] appended anchor to {grad.gradient_array_path} (axis={args.axis})")
        else:
            print("[master-gradient] --no-anchor-write set; skipping ledger append")

        # ── Per-pair sister artifact (when --preserve-per-pair) ────────────────
        if args.preserve_per_pair:
            if grad_seg_sd_per_pair is None or grad_pose_sd_per_pair is None:
                raise RuntimeError(
                    "preserve_per_pair=True but per-pair gradients are None; "
                    "compute_operating_point_and_per_param_gradients contract violated"
                )

            per_pair_output_npy = args.per_pair_output_npy
            if per_pair_output_npy is None:
                # Default sister path: insert "_per_pair_<N>pair" before .npy
                stem = args.output_npy.stem
                suffix = args.output_npy.suffix
                per_pair_output_npy = args.output_npy.with_name(
                    f"{stem}_per_pair_{args.n_pairs_used}pair{suffix}"
                )

            if per_pair_output_npy.is_absolute() and str(per_pair_output_npy).startswith(
                ("/tmp/", "/private/tmp/", "/var/tmp/")
            ):
                raise SystemExit(
                    f"--per-pair-output-npy {per_pair_output_npy} forbidden under /tmp "
                    "per CLAUDE.md 'Forbidden /tmp paths in any persisted artifact' "
                    "(Catalog #220 + transient-evidence trap)"
                )

            print(
                f"[master-gradient] projecting per-pair grad to per-byte per-pair "
                f"(shape (N_bytes={layout.n_archive_bytes}, N_pairs={args.n_pairs_used}, 3))"
            )

            # Project per-pair: loop over the pair axis, slicing each pair's
            # per-param dict and reusing the canonical single-pair projector.
            G_per_pair = np.zeros(
                (layout.n_archive_bytes, args.n_pairs_used, 3), dtype=np.float32
            )
            t_pp = time.time()
            for i in range(args.n_pairs_used):
                grad_seg_i = {
                    name: tensor[i] for name, tensor in grad_seg_sd_per_pair.items()
                }
                grad_pose_i = {
                    name: tensor[i] for name, tensor in grad_pose_sd_per_pair.items()
                }
                G_i = project_per_param_gradient_to_per_byte(
                    layout, grad_seg_i, grad_pose_i, inner_base=0
                )
                if G_i.shape != (layout.n_archive_bytes, 3):
                    raise RuntimeError(
                        f"per-pair projection shape {G_i.shape} != "
                        f"({layout.n_archive_bytes}, 3) at pair {i}"
                    )
                G_per_pair[:, i, :] = G_i
            proj_secs = time.time() - t_pp
            print(
                f"[master-gradient] per-pair projection done in {proj_secs:.2f}s "
                f"(N_bytes={layout.n_archive_bytes}, N_pairs={args.n_pairs_used})"
            )

            # Sanity: shape + finite
            if G_per_pair.shape != (layout.n_archive_bytes, args.n_pairs_used, 3):
                raise RuntimeError(
                    f"per-pair gradient shape {G_per_pair.shape} != "
                    f"({layout.n_archive_bytes}, {args.n_pairs_used}, 3)"
                )
            pp_nans = int(np.isnan(G_per_pair).sum())
            pp_infs = int(np.isinf(G_per_pair).sum())
            if pp_nans or pp_infs:
                raise RuntimeError(
                    f"per-pair gradient has {pp_nans} NaN + {pp_infs} Inf entries"
                )

            # Write sister sidecar — cast to storage_dtype per --storage-dtype contract
            per_pair_output_npy.parent.mkdir(parents=True, exist_ok=True)
            G_pp_to_save = (
                G_per_pair.astype(storage_np_dtype, copy=False)
                if G_per_pair.dtype != storage_np_dtype else G_per_pair
            )
            np.save(per_pair_output_npy, G_pp_to_save)
            print(
                f"[master-gradient] wrote per-pair sidecar {per_pair_output_npy} "
                f"({G_pp_to_save.nbytes} bytes; shape={G_pp_to_save.shape}; dtype={G_pp_to_save.dtype})"
            )

            # Optional sister ledger anchor
            if not args.no_anchor_write:
                grad_pp = MasterGradient(
                    archive_sha256=scored_archive_sha256,
                    operating_point=op,
                    gradient_array_path=str(per_pair_output_npy.resolve()),
                    n_bytes=layout.n_archive_bytes,
                    measurement_method=(
                        f"{measurement_method_base}_per_pair_{args.n_pairs_used}pair"
                    ),
                    measurement_axis=args.axis,
                    measurement_hardware=hardware_substrate,
                    measurement_call_id=(
                        f"{args.call_id}_per_pair" if args.call_id else None
                    ),
                    measurement_utc=datetime.now(UTC).isoformat(),
                    pareto_facets=(),
                    rashomon_disagreement_score=None,
                    gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
                    n_pairs=args.n_pairs_used,
                    scored_archive_sha256=scored_archive_sha256,
                    scored_archive_bytes=len(scored_archive_bytes),
                    gradient_subject_sha256=gradient_subject_sha256,
                    gradient_subject_bytes=layout.n_archive_bytes,
                    gradient_byte_domain=gradient_byte_domain,
                    n_pairs_used=args.n_pairs_used,
                    n_pairs_total=args.n_pairs_total,
                    score_axis_dominance=score_axis_dominance_summary(G_pp_to_save, op),
                )
                append_anchor_locked(grad_pp)
                print(
                    f"[master-gradient] appended per-pair anchor to "
                    f"{grad_pp.gradient_array_path} (axis={args.axis})"
                )
            else:
                print(
                    "[master-gradient] --no-anchor-write set; skipping per-pair ledger append"
                )

        print(
            f"[master-gradient] DONE [score-axis={args.axis}] "
            f"scored_sha256={scored_archive_sha256[:16]} "
            f"subject_sha256={layout.archive_sha256[:16]}"
        )
        return 0
    finally:
        unpatch_upstream_yuv6(patch_token)


if __name__ == "__main__":
    raise SystemExit(main())
