# SPDX-License-Identifier: MIT
"""PR101/FEC6 PacketIR candidate queue with byte accounting.

The queue is a producer-to-consumer integration surface for PR101/FEC6.  It
does not materialize new archive bytes and it never claims score authority.
Every row is either the current identity archive or a blocked operator/probe
candidate that still needs archive materialization, runtime byte-consumption
proof, and paired exact eval before dispatch.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.packet_compiler.pr101_fec6_packetir import (
    FEC6_FIXED_K16_MODE_IDS,
    PR101_FEC6_DEFAULT_MEMBER_NAME,
    PacketIRSection,
    canonical_expected_sha256,
    parse_pr101_fec6_packetir_member,
    read_single_stored_fec6_member_archive,
    sha256_hex,
)
from tac.packet_compiler.pr101_fec6_source_anatomy import (
    PR101_DECODER_BLOB_LEN,
    PR101_LATENT_BLOB_LEN,
)
from tac.repo_io import read_json

PR101_FEC6_CANDIDATE_QUEUE_SCHEMA = "pr101_fec6_packetir_candidate_queue_v2"
PR101_FEC6_BYTE_ACCOUNTING_SCHEMA = "pr101_fec6_packetir_byte_accounting_v2"

# Per-section parser-domain surfaces.  These are the inflate.py / codec.py /
# codec_sidecar.py read sites we BELIEVE consume the named bytes at runtime.
# Per codex adversarial review 2026-05-19 (Catalog #105 + #139 + #220 sister
# discipline): parser-domain knowledge of "section X is named primary" does NOT
# imply runtime consumption.  The byte_accounting v2 contract requires an
# external runtime-consumption proof (Catalog #105 no-op detector + Catalog
# #139 packet-compiler runtime-consumes-payload-bytes evidence) before
# `runtime_consumption_proven` flips to True.  These surfaces are therefore
# `PARSER_RUNTIME_CANDIDATE_SURFACES` (candidate read sites), NOT proof.
PARSER_RUNTIME_CANDIDATE_SURFACES: tuple[str, ...] = (
    "submission_dir/inflate.py::parse_pr101_frame_selector_archive",
    "submission_dir/inflate.py::unpack_compact_selector_codes",
    "submission_dir/inflate.py::unpack_fec6_fixed_huffman_codes",
    "submission_dir/inflate.py::apply_pr101_selector_to_frames",
    "submission_dir/src/codec.py::parse_archive",
    "submission_dir/src/codec_sidecar.py::apply_latent_sidecar",
)
# v1 alias preserved for any external consumer that still references the
# original name (Catalog #110/#113 HISTORICAL_PROVENANCE: do not break
# downstream re-imports).  New code MUST use the renamed canonical above.
RUNTIME_CONSUMER_SURFACES: tuple[str, ...] = PARSER_RUNTIME_CANDIDATE_SURFACES

QUEUE_CONSUMER_SURFACES: tuple[str, ...] = (
    "tac.packet_compiler.pr101_frontier_packetir_matrix._candidate_queue_row",
    "tools/build_pr101_frontier_packetir_matrix.py",
    "tac.cathedral_consumers.packetir_candidate_queue_consumer.consume_candidate",
)

PRIMARY_SECTION_NAMES = frozenset(
    {
        "fp11_magic",
        "source_len_u32le",
        "source_pr101_payload",
        "selector_len_u16le",
        "selector_fec6_payload",
    }
)
SCORE_AFFECTING_SECTION_NAMES = frozenset(
    {"source_pr101_payload", "selector_fec6_payload"}
)

DEFAULT_OPERATOR_BLOCKERS: tuple[str, ...] = (
    "packet_candidate_not_materialized",
    "runtime_byte_consumption_noop_detector_missing",
    "paired_contest_cpu_cuda_exact_eval_missing",
)


class PR101FEC6CandidateQueueError(ValueError):
    """Raised when PR101/FEC6 queue inputs are malformed."""


def _load_operator_manifest(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise PR101FEC6CandidateQueueError(
            f"operator_space_manifest not found: {path}"
        )
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise PR101FEC6CandidateQueueError(
            f"operator_space_manifest is not a JSON object: {path}"
        )
    return payload


def _section_accounting_row(section: PacketIRSection) -> dict[str, Any]:
    row = section.to_manifest()
    primary = section.name in PRIMARY_SECTION_NAMES
    score_affecting = section.name in SCORE_AFFECTING_SECTION_NAMES
    # Per codex adversarial review 2026-05-19 + Catalog #105/#139/#220 sister
    # discipline: `runtime_consumed=True` was CARGO-CULTED from `primary` (parser
    # knowledge); it conflated parser custody with runtime authority.  v2 keeps
    # `primary_payload_section` (parser-domain truth) but does NOT assert
    # `runtime_consumed`.  Caller must supply a runtime-consumption proof
    # (Catalog #105 no-op detector evidence) to flip runtime_consumed True.
    row.update(
        {
            "primary_payload_section": primary,
            "parser_section_runtime_candidate": primary,
            "runtime_consumption_proven": False,
            "score_affecting": score_affecting,
            "parser_runtime_candidate_surfaces": (
                list(PARSER_RUNTIME_CANDIDATE_SURFACES) if primary else []
            ),
            # v1 alias preserved for backward-compat reads.  Defaults to
            # False until runtime_consumption_proof attaches per Catalog #105.
            "runtime_consumed": False,
        }
    )
    return row


def _runtime_consumption_proof_summary(
    proof: Mapping[str, Any] | None,
    *,
    member_payload_bytes: int,
) -> dict[str, Any]:
    """Validate an optional runtime-consumption proof (Catalog #105/#139).

    Returns a structured summary + per-blocker list.  When `proof is None` the
    summary is `runtime_consumption_proven=False` + blocker
    ``runtime_byte_consumption_noop_detector_missing`` so the queue-level
    blockers correctly surface the missing evidence.
    """

    if proof is None:
        return {
            "runtime_consumption_proven": False,
            "no_op_detector_passed": None,
            "runtime_consumption_proof_path": None,
            "runtime_consumption_proof_source": None,
            "runtime_consumption_proof_bytes_consumed": None,
            "blockers": ["runtime_byte_consumption_noop_detector_missing"],
        }
    blockers: list[str] = []
    no_op_detector_passed = proof.get("no_op_detector_passed")
    if no_op_detector_passed is not True:
        blockers.append("runtime_consumption_proof_no_op_detector_not_passed")
    consumed_bytes = proof.get("runtime_bytes_consumed")
    consumed_int = (
        int(consumed_bytes) if isinstance(consumed_bytes, (int, float)) else None
    )
    if consumed_int is None or consumed_int != member_payload_bytes:
        blockers.append("runtime_consumption_proof_bytes_consumed_mismatch")
    schema_version = proof.get("schema_version")
    if schema_version not in ("deterministic_no_op_proof.v1",):
        blockers.append("runtime_consumption_proof_schema_unrecognized")
    consumed_section_names = _string_list(proof.get("consumed_section_names"))
    consumed_byte_ranges = _mapping_list(proof.get("consumed_byte_ranges"))
    return {
        "runtime_consumption_proven": not blockers,
        "no_op_detector_passed": (
            no_op_detector_passed if isinstance(no_op_detector_passed, bool) else None
        ),
        "consumed_section_names": consumed_section_names,
        "consumed_byte_ranges": consumed_byte_ranges,
        "runtime_consumption_proof_path": proof.get("runtime_consumption_proof_path"),
        "runtime_consumption_proof_source": proof.get(
            "runtime_consumption_proof_source"
        ),
        "runtime_consumption_proof_bytes_consumed": consumed_int,
        "runtime_consumption_proof_sha256": proof.get(
            "runtime_consumption_proof_sha256"
        ),
        "blockers": blockers,
    }


def packetir_byte_accounting(
    sections: tuple[PacketIRSection, ...],
    *,
    member_payload_bytes: int,
    runtime_consumption_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic byte accounting for PR101/FEC6 PacketIR sections.

    Per codex adversarial review 2026-05-19: the v1 contract conflated parser
    byte accounting (we can sum primary-section lengths) with runtime
    consumption proof (inflate.py actually reads + consumes the bytes).  v2
    separates them:

    * ``parser_byte_accounting_passed`` — strict subset of v1 ``all_payload_bytes_accounted``;
      True iff the primary-section lengths sum equals the member payload bytes.
    * ``runtime_consumption_proven`` — defaults False; flips True only when a
      valid ``runtime_consumption_proof`` (Catalog #105 no-op-detector evidence
      with ``no_op_detector_passed=True`` and matching consumed-bytes count)
      is supplied by the caller.

    Backward-compat: the legacy ``runtime_consumed_byte_accounting_passed``
    field is preserved but rebound to ``runtime_consumption_proven`` so any
    downstream reader cannot accidentally consume the v1 cargo-culted value.
    """

    rows = [_section_accounting_row(section) for section in sections]
    accounted = sum(row["length"] for row in rows if row["primary_payload_section"])
    parser_passed = accounted == member_payload_bytes
    proof_summary = _runtime_consumption_proof_summary(
        runtime_consumption_proof,
        member_payload_bytes=member_payload_bytes,
    )
    runtime_proven = bool(parser_passed and proof_summary["runtime_consumption_proven"])
    consumed_section_names = set(proof_summary.get("consumed_section_names") or [])
    if runtime_proven and consumed_section_names:
        for row in rows:
            if row["name"] in consumed_section_names:
                row["runtime_consumption_proven"] = True
                row["runtime_consumed"] = True
    accounting_blockers: list[str] = []
    if not parser_passed:
        accounting_blockers.append("parser_byte_accounting_failed")
    accounting_blockers.extend(proof_summary["blockers"])
    return {
        "schema": PR101_FEC6_BYTE_ACCOUNTING_SCHEMA,
        "member_payload_bytes": member_payload_bytes,
        "accounted_primary_payload_bytes": accounted,
        # Parser-domain truth (strict subset of legacy v1 all_payload_bytes_accounted).
        "all_payload_bytes_accounted": parser_passed,
        "parser_byte_accounting_passed": parser_passed,
        # Runtime authority — independent of parser sum; default False per
        # Catalog #105 + #139 + #220 sister discipline; flips True only via
        # supplied runtime_consumption_proof.
        "runtime_consumption_proven": runtime_proven,
        "runtime_consumed_section_names": sorted(consumed_section_names),
        "runtime_consumed_byte_ranges": proof_summary.get("consumed_byte_ranges") or [],
        "runtime_consumption_proof_summary": proof_summary,
        # Legacy v1 alias: rebound to runtime_consumption_proven so downstream
        # readers cannot consume the old length-summation lie.
        "runtime_consumed_byte_accounting_passed": runtime_proven,
        "score_affecting_section_names": sorted(SCORE_AFFECTING_SECTION_NAMES),
        # Renamed: surfaces are parser-domain CANDIDATES, not proof.
        "parser_runtime_candidate_surfaces": list(PARSER_RUNTIME_CANDIDATE_SURFACES),
        # v1 alias preserved for backward-compat reads.
        "runtime_consumer_surfaces": list(PARSER_RUNTIME_CANDIDATE_SURFACES),
        "queue_consumer_surfaces": list(QUEUE_CONSUMER_SURFACES),
        "sections": rows,
        "byte_accounting_blockers": accounting_blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _selector_entropy_floor_bytes(codes: tuple[int, ...]) -> int:
    if not codes:
        return 0
    counts = Counter(codes)
    total = float(len(codes))
    entropy_bits = -sum(
        count * math.log2(count / total) for count in counts.values()
    )
    return math.ceil(entropy_bits / 8.0)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _false_authority_flags() -> dict[str, bool]:
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_operator_probe": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
    }


def _candidate_from_operator_row(
    row: Mapping[str, Any],
    *,
    parsed_codes: tuple[int, ...],
) -> dict[str, Any]:
    pair = row.get("pair")
    blockers = list(row.get("blockers") or [])
    for blocker in DEFAULT_OPERATOR_BLOCKERS:
        if blocker not in blockers:
            blockers.append(blocker)
    candidate: dict[str, Any] = {
        "candidate_id": str(row.get("operator_id") or f"operator_pair_{pair}"),
        "candidate_kind": "grammar_aware_selector_symbol_substitution",
        "status": "blocked_until_materialized_runtime_proven_and_paired_exact_eval",
        "source": "operator_space_manifest",
        "pair": pair,
        "current_code": row.get("current_code"),
        "candidate_code": row.get("candidate_code"),
        "current_mode_id": row.get("current_mode_id"),
        "candidate_mode_id": row.get("candidate_mode_id"),
        "selector_code_bit_delta": row.get("selector_code_bit_delta"),
        "selector_index_byte_delta_if_single_mutation": row.get(
            "selector_index_byte_delta_if_single_mutation"
        ),
        "local_proxy_delta_with_rate": row.get("local_proxy_delta_with_rate"),
        "evidence_axis": row.get("evidence_axis", "proxy/advisory only"),
        "blockers": blockers,
        "consumer_surfaces": list(RUNTIME_CONSUMER_SURFACES)
        + list(QUEUE_CONSUMER_SURFACES),
        **_false_authority_flags(),
    }
    mismatch = False
    if isinstance(pair, int) and 0 <= pair < len(parsed_codes):
        mismatch = row.get("current_code") != parsed_codes[pair]
        candidate["parsed_current_code"] = parsed_codes[pair]
    else:
        mismatch = True
        candidate["parsed_current_code"] = None
    candidate["current_code_matches_parsed_selector"] = not mismatch
    if mismatch and "operator_row_current_code_mismatch" not in blockers:
        candidate["blockers"] = [*blockers, "operator_row_current_code_mismatch"]
    return candidate


def _operator_candidates(
    operator_manifest: Mapping[str, Any],
    *,
    parsed_codes: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("proxy_and_nonpositive_bit_rows", "top_bit_saving_rows", "top_proxy_improving_rows"):
        value = operator_manifest.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, Mapping))
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate = _candidate_from_operator_row(row, parsed_codes=parsed_codes)
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidates.append(candidate)
    return candidates


def _load_runtime_consumption_proof(
    path: Path | None,
) -> Mapping[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise PR101FEC6CandidateQueueError(
            f"runtime_consumption_proof not found: {path}"
        )
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise PR101FEC6CandidateQueueError(
            f"runtime_consumption_proof is not a JSON object: {path}"
        )
    return payload


def build_pr101_fec6_packetir_candidate_queue(
    *,
    archive_path: str | Path,
    operator_space_manifest_path: str | Path | None = None,
    expected_member_name: str = PR101_FEC6_DEFAULT_MEMBER_NAME,
    expected_archive_sha256: str | None = None,
    runtime_consumption_proof_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the non-dispatching PR101/FEC6 PacketIR candidate queue.

    Per codex adversarial review 2026-05-19 + Catalog #105/#139/#220 sister
    discipline: callers SHOULD supply ``runtime_consumption_proof_path``
    pointing at a Catalog #105 no-op-detector proof JSON.  When omitted (the
    common case during $0 queue authoring) the top-level ``blockers`` list
    will surface ``runtime_byte_consumption_noop_detector_missing`` to
    explicitly propagate the missing runtime authority to downstream readers.
    """

    archive = Path(archive_path)
    archive_bytes = archive.read_bytes()
    archive_sha = sha256_hex(archive_bytes)
    expected_sha, expected_sha_well_formed = canonical_expected_sha256(
        expected_archive_sha256
    )
    member = read_single_stored_fec6_member_archive(
        archive_bytes,
        expected_member_name=expected_member_name,
    )
    packet = parse_pr101_fec6_packetir_member(member.payload)
    runtime_proof_path = (
        Path(runtime_consumption_proof_path)
        if runtime_consumption_proof_path is not None
        else None
    )
    runtime_consumption_proof = _load_runtime_consumption_proof(runtime_proof_path)
    byte_accounting = packetir_byte_accounting(
        packet.sections,
        member_payload_bytes=packet.payload_bytes,
        runtime_consumption_proof=runtime_consumption_proof,
    )
    operator_manifest_path = (
        Path(operator_space_manifest_path)
        if operator_space_manifest_path is not None
        else None
    )
    operator_manifest = _load_operator_manifest(operator_manifest_path)
    manifest_codes = (
        operator_manifest.get("source_archive", {}).get("codes")
        if isinstance(operator_manifest.get("source_archive"), Mapping)
        else None
    )
    operator_codes_match = (
        list(packet.selector_codes) == manifest_codes
        if isinstance(manifest_codes, list)
        else None
    )

    blockers: list[str] = []
    if expected_sha_well_formed is False:
        blockers.append("expected_archive_sha256_malformed")
    if expected_sha_well_formed is True and expected_sha != archive_sha:
        blockers.append("expected_archive_sha256_mismatch")
    if byte_accounting["parser_byte_accounting_passed"] is not True:
        blockers.append("parser_byte_accounting_failed")
    # Propagate any byte-accounting runtime-consumption blockers to the
    # top-level queue blockers per codex adversarial review 2026-05-19 F1:
    # the v1 top-level `blockers=[]` while 32-of-33 candidates carried
    # `runtime_byte_consumption_noop_detector_missing` was the
    # cargo-culted-authority bug.  v2 surfaces it at the queue level.
    accounting_blockers = byte_accounting.get("byte_accounting_blockers") or []
    for accounting_blocker in accounting_blockers:
        if accounting_blocker not in blockers:
            blockers.append(accounting_blocker)
    if operator_codes_match is False:
        blockers.append("operator_manifest_source_codes_mismatch")

    selector_histogram = {
        FEC6_FIXED_K16_MODE_IDS[code]: int(count)
        for code, count in sorted(Counter(packet.selector_codes).items())
    }
    identity_candidate = {
        "candidate_id": "fec6_identity_current_archive",
        "candidate_kind": "identity_reference",
        "status": "materialized_existing_archive_only",
        "archive_sha256": archive_sha,
        "archive_size_bytes": len(archive_bytes),
        "member_sha256": sha256_hex(member.payload),
        "member_payload_bytes": len(member.payload),
        "changed_sections": [],
        "blockers": [],
        "consumer_surfaces": list(RUNTIME_CONSUMER_SURFACES)
        + list(QUEUE_CONSUMER_SURFACES),
        "non_promotion_rationale": (
            "identity reference for PacketIR/compiler custody; not a new "
            "candidate and not score authority"
        ),
        **_false_authority_flags(),
    }
    model_candidates = [
        {
            "candidate_id": "fec6_selector_entropy_recode_probe",
            "candidate_kind": "selector_entropy_recode_probe",
            "status": "queue_only_not_materialized",
            "byte_region": "selector_fec6_fixed_huffman_bitstream",
            "current_selector_index_bytes": len(packet.selector_bitstream),
            "current_selector_code_bits_total": packet.selector_code_bits_total,
            "empirical_entropy_floor_bytes": _selector_entropy_floor_bytes(
                packet.selector_codes
            ),
            "histogram": selector_histogram,
            "blockers": list(DEFAULT_OPERATOR_BLOCKERS),
            "consumer_surfaces": list(RUNTIME_CONSUMER_SURFACES)
            + list(QUEUE_CONSUMER_SURFACES),
            **_false_authority_flags(),
        },
        {
            "candidate_id": "fec6_wrapper_length_field_elision_probe",
            "candidate_kind": "wrapper_metadata_recode_probe",
            "status": "queue_only_not_materialized",
            "byte_region": "fp11_magic/source_len_u32le/selector_len_u16le",
            "current_wrapper_overhead_bytes": 10,
            "blockers": list(DEFAULT_OPERATOR_BLOCKERS),
            "consumer_surfaces": list(RUNTIME_CONSUMER_SURFACES)
            + list(QUEUE_CONSUMER_SURFACES),
            **_false_authority_flags(),
        },
        {
            "candidate_id": "fec6_source_payload_packetir_recode_probe",
            "candidate_kind": "source_payload_recode_probe",
            "status": "queue_only_not_materialized",
            "byte_region": "source_pr101_payload",
            "current_source_payload_bytes": len(packet.source_pr101_payload),
            "blockers": list(DEFAULT_OPERATOR_BLOCKERS),
            "consumer_surfaces": list(RUNTIME_CONSUMER_SURFACES)
            + list(QUEUE_CONSUMER_SURFACES),
            **_false_authority_flags(),
        },
        {
            "candidate_id": "pr101_sidecar_only_runtime_probe",
            "candidate_kind": "section_runtime_visibility_probe",
            "status": "queue_only_not_materialized",
            "byte_region": "pr101_sidecar_blob",
            "current_sidecar_bytes": max(
                0,
                len(packet.source_pr101_payload)
                - PR101_DECODER_BLOB_LEN
                - PR101_LATENT_BLOB_LEN,
            ),
            "required_runtime_consumed_section": "pr101_sidecar_blob",
            "blockers": list(DEFAULT_OPERATOR_BLOCKERS)
            + ["sidecar_only_materializer_missing"],
            "consumer_surfaces": list(RUNTIME_CONSUMER_SURFACES)
            + list(QUEUE_CONSUMER_SURFACES),
            **_false_authority_flags(),
        },
        {
            "candidate_id": "pr101_latent_plus_sidecar_runtime_adapter_probe",
            "candidate_kind": "section_runtime_adapter_probe",
            "status": "queue_only_not_materialized",
            "byte_region": "pr101_latent_blob+pr101_sidecar_blob",
            "current_latent_bytes": PR101_LATENT_BLOB_LEN,
            "current_sidecar_bytes": max(
                0,
                len(packet.source_pr101_payload)
                - PR101_DECODER_BLOB_LEN
                - PR101_LATENT_BLOB_LEN,
            ),
            "required_runtime_consumed_sections": [
                "pr101_latent_blob",
                "pr101_sidecar_blob",
            ],
            "blockers": list(DEFAULT_OPERATOR_BLOCKERS)
            + ["latent_sidecar_runtime_adapter_missing"],
            "consumer_surfaces": list(RUNTIME_CONSUMER_SURFACES)
            + list(QUEUE_CONSUMER_SURFACES),
            **_false_authority_flags(),
        },
    ]
    operator_candidates = _operator_candidates(
        operator_manifest,
        parsed_codes=packet.selector_codes,
    )
    all_candidates = [identity_candidate, *model_candidates, *operator_candidates]
    return {
        "schema": PR101_FEC6_CANDIDATE_QUEUE_SCHEMA,
        "proof_scope": (
            "packetir_candidate_queue_byte_accounting_and_consumer_wiring_no_dispatch"
        ),
        "archive_path": archive.as_posix(),
        "archive_sha256": archive_sha,
        "archive_size_bytes": len(archive_bytes),
        "expected_archive_sha256": expected_sha,
        "expected_archive_sha256_well_formed": expected_sha_well_formed,
        "expected_archive_sha256_matches": (
            None
            if expected_sha is None or expected_sha_well_formed is False
            else archive_sha == expected_sha
        ),
        "operator_space_manifest_path": (
            operator_manifest_path.as_posix()
            if operator_manifest_path is not None
            else None
        ),
        "operator_manifest_source_codes_match": operator_codes_match,
        "member_name": member.name,
        "member_sha256": sha256_hex(member.payload),
        "member_payload_bytes": len(member.payload),
        "selector": {
            "n_pairs": packet.n_pairs,
            "selector_index_bytes": len(packet.selector_bitstream),
            "selector_code_bits_total": packet.selector_code_bits_total,
            "histogram": selector_histogram,
        },
        "byte_accounting": byte_accounting,
        "candidate_count": len(all_candidates),
        "operator_candidate_count": len(operator_candidates),
        "materialized_new_archive_count": 0,
        "candidates": all_candidates,
        "producer": "tac.packet_compiler.pr101_fec6_candidate_queue",
        "consumer_surfaces": list(QUEUE_CONSUMER_SURFACES),
        "blockers": blockers,
        **_false_authority_flags(),
    }


def render_pr101_fec6_packetir_candidate_queue_markdown(
    queue: Mapping[str, Any],
) -> str:
    """Render a compact markdown summary for the queue."""

    lines = [
        "# PR101/FEC6 PacketIR Candidate Queue",
        "",
        f"- Schema: `{queue.get('schema')}`",
        f"- Archive: `{queue.get('archive_path')}`",
        f"- Archive SHA-256: `{queue.get('archive_sha256')}`",
        f"- Candidate count: `{queue.get('candidate_count')}`",
        f"- Operator candidate count: `{queue.get('operator_candidate_count')}`",
        f"- Materialized new archives: `{queue.get('materialized_new_archive_count')}`",
        f"- Score claim: `{queue.get('score_claim')}`",
        f"- Promotion eligible: `{queue.get('promotion_eligible')}`",
        f"- Ready for exact eval dispatch: `{queue.get('ready_for_exact_eval_dispatch')}`",
        "",
        "## Consumer Surfaces",
        "",
    ]
    for surface in queue.get("consumer_surfaces", []):
        lines.append(f"- `{surface}`")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| id | kind | status | dispatch |",
            "|---|---|---|---|",
        ]
    )
    for candidate in queue.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        lines.append(
            "| `{id}` | `{kind}` | `{status}` | `{dispatch}` |".format(
                id=candidate.get("candidate_id", ""),
                kind=candidate.get("candidate_kind", ""),
                status=candidate.get("status", ""),
                dispatch=candidate.get("ready_for_exact_eval_dispatch"),
            )
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_OPERATOR_BLOCKERS",
    "PR101_FEC6_BYTE_ACCOUNTING_SCHEMA",
    "PR101_FEC6_CANDIDATE_QUEUE_SCHEMA",
    "QUEUE_CONSUMER_SURFACES",
    "RUNTIME_CONSUMER_SURFACES",
    "PR101FEC6CandidateQueueError",
    "build_pr101_fec6_packetir_candidate_queue",
    "packetir_byte_accounting",
    "render_pr101_fec6_packetir_candidate_queue_markdown",
]
