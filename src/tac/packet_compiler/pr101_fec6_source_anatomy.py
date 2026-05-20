# SPDX-License-Identifier: MIT
"""PR101/FEC6 source-payload anatomy and magic-codec grounding.

[verified-against: experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py]
[verified-against: src/tac/packet_compiler/pr101_fec6_packetir.py]

This module decomposes the PR101 source payload inside the local FEC6 wrapper:

``FP11 | source_len | decoder_blob | latent_blob | sidecar_blob | selector_len | selector``

It is a read-only profiling surface. It does not rewrite archives, run scorers,
or claim score movement. Magic-codec probes are byte-accounting diagnostics only.
"""

from __future__ import annotations

import hashlib
import lzma
import math
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np

from tac.authority_contract import apply_false_authority_contract
from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES
from tac.packet_compiler.magic_codec import MagicCodecError, StreamHint, encode_magic_codec
from tac.packet_compiler.pr101_fec6_packetir import (
    PR101_FEC6_DEFAULT_MEMBER_NAME,
    parse_pr101_fec6_packetir_member,
    read_single_stored_fec6_member_archive,
    sha256_hex,
)


SCHEMA = "pr101_fec6_source_payload_anatomy_profile_v1"
PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_N_PAIRS = 600
PR101_LATENT_DIM = 28
LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES


@dataclass(frozen=True)
class SourceAnatomySection:
    """One semantic byte section in the FEC6 member payload."""

    name: str
    role: str
    member_start: int
    member_end: int
    payload: bytes
    incumbent_codec: str
    rewrite_surface: str

    @property
    def length(self) -> int:
        return self.member_end - self.member_start

    def as_profile_row(
        self,
        *,
        null_indices: np.ndarray | None,
        run_preview_limit: int = 8,
        include_magic: bool = True,
    ) -> dict[str, Any]:
        null = _null_coverage(
            start=self.member_start,
            end=self.member_end,
            null_indices=null_indices,
            run_preview_limit=run_preview_limit,
        )
        row = {
            "name": self.name,
            "role": self.role,
            "range": [self.member_start, self.member_end],
            "length_bytes": self.length,
            "sha256": sha256_hex(self.payload),
            "incumbent_codec": self.incumbent_codec,
            "rewrite_surface": self.rewrite_surface,
            "null_coverage": null,
            "entropy": _entropy_profile(self.payload),
            "generic_recompression": _generic_recompression_profile(self.payload),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        if include_magic:
            row["magic_codec_byte_stream_probe"] = _magic_byte_stream_probe(
                self.payload,
                section_name=self.name,
            )
        return row


def read_archive_packet(archive_path: Path | str) -> tuple[str, bytes, Any]:
    """Read and parse the single stored FEC6 archive member."""

    archive_bytes = Path(archive_path).read_bytes()
    member = read_single_stored_fec6_member_archive(
        archive_bytes,
        expected_member_name=PR101_FEC6_DEFAULT_MEMBER_NAME,
    )
    packet = parse_pr101_fec6_packetir_member(member.payload)
    return member.name, archive_bytes, packet


def build_source_anatomy_sections(packet: Any) -> tuple[SourceAnatomySection, ...]:
    """Return semantic sections with absolute member-payload offsets."""

    source = packet.source_pr101_payload
    if len(source) < PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN:
        raise ValueError("source_pr101_payload is shorter than PR101 decoder+latent constants")
    sidecar_len = len(source) - PR101_DECODER_BLOB_LEN - PR101_LATENT_BLOB_LEN
    if sidecar_len < 0:
        raise ValueError("negative PR101 sidecar length")

    source_start = 8
    decoder_start = source_start
    decoder_end = decoder_start + PR101_DECODER_BLOB_LEN
    latent_end = decoder_end + PR101_LATENT_BLOB_LEN
    sidecar_end = source_start + len(source)
    selector_len_end = sidecar_end + 2
    selector_end = selector_len_end + len(packet.selector_fec6_payload)
    member = (
        b"FP11"
        + packet.source_len_u32le
        + packet.source_pr101_payload
        + packet.selector_len_u16le
        + packet.selector_fec6_payload
    )
    return (
        SourceAnatomySection(
            "fp11_magic",
            "fec6_wrapper_header",
            0,
            4,
            member[0:4],
            "literal",
            "header_not_material_rate_target",
        ),
        SourceAnatomySection(
            "source_len_u32le",
            "fec6_wrapper_header",
            4,
            8,
            packet.source_len_u32le,
            "literal_u32le",
            "header_not_material_rate_target",
        ),
        SourceAnatomySection(
            "pr101_decoder_blob",
            "pr101_hnerv_decoder_weights",
            decoder_start,
            decoder_end,
            source[:PR101_DECODER_BLOB_LEN],
            "concatenated_brotli_quantized_tensor_streams",
            "magic_codec_rewrite_possible_but_current_bytes_are_entropy_saturated",
        ),
        SourceAnatomySection(
            "pr101_latent_blob",
            "pr101_pair_latents",
            decoder_end,
            latent_end,
            source[PR101_DECODER_BLOB_LEN : PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN],
            "raw_lzma1_centered_delta_uint8_latents",
            "typed_runtime_adapter_candidate",
        ),
        SourceAnatomySection(
            "pr101_sidecar_blob",
            "pr101_latent_sidecar",
            latent_end,
            sidecar_end,
            source[PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN :],
            "compact_ranked_or_huffman_latent_sidecar",
            "typed_runtime_adapter_candidate",
        ),
        SourceAnatomySection(
            "selector_len_u16le",
            "fec6_wrapper_header",
            sidecar_end,
            selector_len_end,
            packet.selector_len_u16le,
            "literal_u16le",
            "header_not_material_rate_target",
        ),
        SourceAnatomySection(
            "selector_fec6_payload",
            "fec6_selector_payload",
            selector_len_end,
            selector_end,
            packet.selector_fec6_payload,
            "fixed_huffman_k16_selector",
            "selector_seed_adapter_already_empirically_falsified_on_current_payload",
        ),
    )


def profile_pr101_fec6_source_payload_anatomy(
    *,
    archive_path: Path | str,
    null_indices: np.ndarray | Sequence[int] | None = None,
    include_magic: bool = True,
) -> dict[str, Any]:
    """Profile source-payload sections and null-run decomposition."""

    member_name, archive_bytes, packet = read_archive_packet(archive_path)
    null_arr = None if null_indices is None else _normalise_null_indices(null_indices)
    sections = build_source_anatomy_sections(packet)
    _validate_sections_cover_member_payload(sections, packet.payload_bytes)
    if null_arr is not None:
        _validate_null_indices_coordinate_space(
            null_arr,
            member_payload_bytes=packet.payload_bytes,
            sections=sections,
        )
    section_rows = [
        section.as_profile_row(null_indices=null_arr, include_magic=include_magic)
        for section in sections
    ]
    latent_profile = _latent_typed_profile(packet.source_pr101_payload)
    null_run = _rank1_null_run_decomposition(
        section_rows=section_rows,
        null_indices=null_arr,
    )
    target_rows = _rank_next_targets(section_rows, latent_profile=latent_profile)
    profile = {
        "schema": SCHEMA,
        "archive_path": str(archive_path),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": sha256_hex(archive_bytes),
        "member_name": member_name,
        "member_payload_bytes": packet.payload_bytes,
        "member_payload_sha256": sha256_hex(
            b"FP11"
            + packet.source_len_u32le
            + packet.source_pr101_payload
            + packet.selector_len_u16le
            + packet.selector_fec6_payload
        ),
        "source_payload_bytes": len(packet.source_pr101_payload),
        "selector_payload_bytes": len(packet.selector_fec6_payload),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": "[predicted]",
        "null_indices": _null_indices_metadata(
            null_arr,
            member_payload_bytes=packet.payload_bytes,
            sections=sections,
        ),
        "sections": section_rows,
        "rank1_null_run_decomposition": null_run,
        "latent_typed_runtime_adapter_probe": latent_profile,
        "ranked_next_targets": target_rows,
        "cascade_relevance": _cascade_relevance(latent_profile),
        "interpretation": _interpretation(target_rows, latent_profile),
    }
    return apply_false_authority_contract(
        profile,
        preserve_dispatch_ready=False,
        reason="pr101_fec6_source_anatomy_is_read_only_byte_profile_not_score_evidence",
    )


def render_source_anatomy_markdown(profile: Mapping[str, Any]) -> str:
    """Render a compact operator-facing profile."""

    lines = [
        "# PR101/FEC6 Source Payload Anatomy",
        "",
        f"- Schema: `{profile.get('schema')}`",
        f"- Score claim: `{str(profile.get('score_claim')).lower()}`",
        f"- Score claim valid: `{str(profile.get('score_claim_valid')).lower()}`",
        f"- Promotion eligible: `{str(profile.get('promotion_eligible')).lower()}`",
        f"- Rank/kill eligible: `{str(profile.get('rank_or_kill_eligible')).lower()}`",
        f"- Ready for exact eval dispatch: `{str(profile.get('ready_for_exact_eval_dispatch')).lower()}`",
        f"- Axis tag: `{profile.get('axis_tag')}`",
        f"- Authority contract: `{profile.get('authority_contract')}`",
        f"- Archive bytes: `{profile.get('archive_bytes')}`",
        f"- Member payload bytes: `{profile.get('member_payload_bytes')}`",
        f"- Source payload bytes: `{profile.get('source_payload_bytes')}`",
        f"- Selector payload bytes: `{profile.get('selector_payload_bytes')}`",
        "",
        "## Semantic Sections",
        "",
        "| section | range | bytes | null bytes | null % | entropy floor | brotli q11 | lzma9e | magic best | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in profile.get("sections", []):
        if not isinstance(row, Mapping):
            continue
        null = _mapping(row.get("null_coverage"))
        entropy = _mapping(row.get("entropy"))
        codecs = _mapping(row.get("generic_recompression"))
        magic = _mapping(row.get("magic_codec_byte_stream_probe"))
        start, end = row.get("range", [None, None])
        verdict = str(row.get("rewrite_surface"))
        lines.append(
            "| `{name}` | `{start}-{end}` | {length} | {n_null} | {frac:.2%} | "
            "{entropy_floor} | {brotli_size} | {lzma_size} | {magic_size} | {verdict} |".format(
                name=row.get("name"),
                start=start,
                end=end,
                length=int(row.get("length_bytes", 0)),
                n_null=int(null.get("n_null_bytes", 0)),
                frac=float(null.get("null_fraction_within_section", 0.0)),
                entropy_floor=int(entropy.get("shannon_floor_bytes", 0)),
                brotli_size=int(codecs.get("brotli_q11_bytes", 0)),
                lzma_size=int(codecs.get("lzma9_extreme_bytes", 0)),
                magic_size=magic.get("best_payload_bytes", "n/a"),
                verdict=verdict,
            )
        )

    latent = _mapping(profile.get("latent_typed_runtime_adapter_probe"))
    lines.extend(
        [
            "",
            "## Rank-1 Null Run",
            "",
        ]
    )
    null_run = _mapping(profile.get("rank1_null_run_decomposition"))
    lines.append(f"- Range: `{null_run.get('range')}`")
    lines.append(f"- Bytes: `{null_run.get('length_bytes')}`")
    lines.append(
        "- Components: `"
        + ", ".join(str(item.get("name")) for item in null_run.get("components", []) if isinstance(item, Mapping))
        + "`"
    )
    lines.extend(
        [
            "",
            "## Typed Latent Probe",
            "",
            f"- Current latent blob bytes: `{latent.get('current_latent_blob_bytes')}`",
            f"- Latent raw bytes: `{latent.get('latent_raw_bytes')}`",
            f"- Best typed alternative bytes: `{latent.get('best_typed_candidate_bytes')}`",
            f"- Best typed alternative delta vs current latent blob: `{latent.get('best_typed_delta_vs_current_latent_blob')}`",
            "",
            "## Ranked Next Targets",
            "",
            "| rank | target | bytes | null % | materialization risk | next action |",
            "| ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in profile.get("ranked_next_targets", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {rank} | `{target}` | {bytes_} | {null_frac:.2%} | `{risk}` | {action} |".format(
                rank=int(row.get("rank", 0)),
                target=row.get("target_id"),
                bytes_=int(row.get("bytes_at_stake", 0)),
                null_frac=float(row.get("null_fraction", 0.0)),
                risk=row.get("materialization_risk"),
                action=row.get("recommended_next_action"),
            )
        )
    relevance = _mapping(profile.get("cascade_relevance"))
    lines.extend(
        [
            "",
            "## Cascade Relevance",
            "",
            str(relevance.get("summary", "")),
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(profile.get("interpretation", "")),
        ]
    )
    return "\n".join(lines) + "\n"


def _cascade_relevance(latent_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Describe how this byte profile should route magic-codec follow-up work."""

    best_delta = latent_profile.get("best_typed_delta_vs_current_latent_blob")
    current_byte_stream_supported = not (isinstance(best_delta, int) and best_delta >= 0)
    return {
        "direct_current_pr101_fec6_payload_magic_codec_supported": current_byte_stream_supported,
        "magic_codec_dense_streams_residual_path_status": "candidate_on_newly_produced_residual_streams_only",
        "summary": (
            "This profile is a negative control for post-hoc magic-codec rewrites of "
            "the current PR101/FEC6 archive member bytes. It does not refute magic_codec "
            "or magic_codec_dense_streams on newly produced streams; the plausible stack "
            "is procedural/DWT/world-model generation first, then dense-stream coding of "
            "the residual between generated and empirical bytes."
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _latent_typed_profile(source_payload: bytes) -> dict[str, Any]:
    decoder_end = PR101_DECODER_BLOB_LEN
    latent_end = decoder_end + PR101_LATENT_BLOB_LEN
    latent_blob = source_payload[decoder_end:latent_end]
    try:
        latent_raw = lzma.decompress(
            latent_blob,
            format=lzma.FORMAT_RAW,
            filters=LATENT_LZMA_FILTERS,
        )
    except lzma.LZMAError as exc:
        return {
            "status": "latent_lzma_decode_failed",
            "error": str(exc),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    expected = PR101_LATENT_DIM * 2 * 2 + PR101_N_PAIRS * PR101_LATENT_DIM
    if len(latent_raw) != expected:
        return {
            "status": "latent_raw_length_mismatch",
            "latent_raw_bytes": len(latent_raw),
            "expected_latent_raw_bytes": expected,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    mins = latent_raw[: PR101_LATENT_DIM * 2]
    scales = latent_raw[PR101_LATENT_DIM * 2 : PR101_LATENT_DIM * 4]
    codes = latent_raw[PR101_LATENT_DIM * 4 :]
    code_matrix = np.frombuffer(codes, dtype=np.uint8).reshape(PR101_LATENT_DIM, PR101_N_PAIRS).T.copy()
    typed_candidates = [
        {
            "candidate_id": "latent_raw_lzma_xz9_extreme",
            "payload_bytes": len(
                lzma.compress(latent_raw, preset=9 | lzma.PRESET_EXTREME)
            ),
            "runtime_adapter_required": True,
            "description": "generic xz/lzma over decoded PR101 latent raw stream",
        },
        {
            "candidate_id": "latent_raw_brotli_q11",
            "payload_bytes": len(brotli.compress(latent_raw, quality=11)),
            "runtime_adapter_required": True,
            "description": "brotli over decoded PR101 latent raw stream",
        },
        {
            "candidate_id": "latent_codes_lzma_xz9_extreme_plus_min_scale",
            "payload_bytes": len(mins) + len(scales) + len(
                lzma.compress(codes, preset=9 | lzma.PRESET_EXTREME)
            ),
            "runtime_adapter_required": True,
            "description": "xz/lzma over latent code bytes plus original fp16 min/scale vectors",
        },
        _typed_magic_candidate(
            "latent_delta_matrix_magic_codec",
            code_matrix,
            current_payload_bytes=len(latent_blob),
        ),
    ]
    typed_candidates = [
        _nested_false_authority_contract(row, reason="latent_typed_probe_diagnostic_only")
        for row in typed_candidates
        if isinstance(row, dict)
    ]
    best = min(typed_candidates, key=lambda row: int(row["payload_bytes"]))
    return {
        "status": "ok",
        "current_latent_blob_bytes": len(latent_blob),
        "latent_raw_bytes": len(latent_raw),
        "latent_min_fp16_bytes": len(mins),
        "latent_scale_fp16_bytes": len(scales),
        "latent_delta_code_bytes": len(codes),
        "typed_candidates": typed_candidates,
        "best_typed_candidate_id": best["candidate_id"],
        "best_typed_candidate_bytes": best["payload_bytes"],
        "best_typed_delta_vs_current_latent_blob": int(best["payload_bytes"]) - len(latent_blob),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _typed_magic_candidate(
    candidate_id: str,
    values: np.ndarray,
    *,
    current_payload_bytes: int,
) -> dict[str, Any]:
    try:
        result = encode_magic_codec(
            values,
            hint=StreamHint("latent_sidecar"),
            selection_strategy="smallest_byte_count",
        )
    except MagicCodecError as exc:
        return {
            "candidate_id": candidate_id,
            "status": "refused",
            "payload_bytes": 10**12,
            "error": str(exc),
            "runtime_adapter_required": True,
            "score_claim": False,
        }
    return {
        "candidate_id": candidate_id,
        "status": "ok",
        "payload_bytes": len(result.payload),
        "delta_vs_current_latent_blob": len(result.payload) - current_payload_bytes,
        "selected_primitive": result.selected_primitive,
        "byte_count_comparison_role": "diagnostic_only_not_lossless_alternative",
        "lossless_roundtrip_verified": False,
        "lossless_roundtrip_required_for_promotion": True,
        "runtime_adapter_required": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _nested_false_authority_contract(
    row: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    row["score_claim"] = False
    row["score_claim_valid"] = False
    row["promotion_eligible"] = False
    row["rank_or_kill_eligible"] = False
    row["ready_for_exact_eval_dispatch"] = False
    row.setdefault("authority_contract", reason)
    return row


def _rank1_null_run_decomposition(
    *,
    section_rows: Sequence[Mapping[str, Any]],
    null_indices: np.ndarray | None,
) -> dict[str, Any]:
    if null_indices is None or null_indices.size == 0:
        return {
            "status": "no_null_indices",
            "components": [],
            "score_claim": False,
        }
    runs = _contiguous_runs(null_indices)
    run = max(runs, key=lambda item: item[1] - item[0])
    start, end = run
    components = []
    for row in section_rows:
        r0, r1 = row.get("range", [None, None])
        if not isinstance(r0, int) or not isinstance(r1, int):
            continue
        overlap_start = max(start, r0)
        overlap_end = min(end, r1)
        if overlap_start < overlap_end:
            components.append(
                {
                    "name": row.get("name"),
                    "range": [overlap_start, overlap_end],
                    "bytes": overlap_end - overlap_start,
                    "role": row.get("role"),
                }
            )
    return {
        "status": "ok",
        "range": [start, end],
        "length_bytes": end - start,
        "components": components,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _rank_next_targets(
    section_rows: Sequence[Mapping[str, Any]],
    *,
    latent_profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_name = {str(row.get("name")): row for row in section_rows}
    targets = []
    for target_id, names, risk, action in (
        (
            "latent_blob_plus_sidecar_semantic_null_span",
            ("pr101_latent_blob", "pr101_sidecar_blob"),
            "high_runtime_adapter_and_exact_eval_required",
            "Build a runtime-adapter candidate that replaces latent+sidecar encoding, then run byte-mutation/no-op and exact CPU/CUDA eval.",
        ),
        (
            "pr101_sidecar_only_null_span",
            ("pr101_sidecar_blob",),
            "medium_exact_eval_required",
            "Probe sidecar-only seed or compact grammar mutations before touching the full latent blob.",
        ),
        (
            "selector_fec6_payload",
            ("selector_fec6_payload",),
            "low_but_already_falsified_by_seeded_selector_probe",
            "Do not materialize simple selector-seed adapter unless a new predictor beats 249 charged bytes.",
        ),
        (
            "wrapper_headers",
            ("fp11_magic", "source_len_u32le", "selector_len_u16le"),
            "header_no_op_and_too_small",
            "Ignore for score movement; preserve for parser custody.",
        ),
    ):
        rows = [by_name[name] for name in names if name in by_name]
        bytes_at_stake = sum(int(row.get("length_bytes", 0)) for row in rows)
        null_bytes = sum(
            int(_mapping(row.get("null_coverage")).get("n_null_bytes", 0))
            for row in rows
        )
        current_best_delta = None
        if target_id == "latent_blob_plus_sidecar_semantic_null_span":
            best_delta = latent_profile.get("best_typed_delta_vs_current_latent_blob")
            if isinstance(best_delta, int):
                current_best_delta = best_delta
        targets.append(
            {
                "target_id": target_id,
                "bytes_at_stake": bytes_at_stake,
                "null_bytes": null_bytes,
                "null_fraction": null_bytes / bytes_at_stake if bytes_at_stake else 0.0,
                "rate_delta_upper_bound_if_seed16": -RATE_SCORE_PER_BYTE
                * max(0, bytes_at_stake - 16),
                "current_typed_or_magic_delta_bytes": current_best_delta,
                "materialization_risk": risk,
                "recommended_next_action": action,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    targets.sort(
        key=lambda row: (
            -float(row["null_fraction"]),
            row["materialization_risk"].startswith("header"),
            -int(row["bytes_at_stake"]),
        )
    )
    for rank, row in enumerate(targets, start=1):
        row["rank"] = rank
    return targets


def _interpretation(
    target_rows: Sequence[Mapping[str, Any]],
    latent_profile: Mapping[str, Any],
) -> str:
    best_delta = latent_profile.get("best_typed_delta_vs_current_latent_blob")
    if isinstance(best_delta, int) and best_delta >= 0:
        magic_note = (
            "Direct magic-codec/generic rewrites of the current PR101 latent bytes "
            f"do not beat the incumbent latent blob in this profile (best delta {best_delta:+d} bytes)."
        )
    else:
        magic_note = "Typed latent rewrite may have byte headroom; inspect profile candidates."
    return (
        "The largest null run is not one semantic object: it is the final byte of "
        "the PR101 decoder blob, the entire PR101 latent blob, the entire latent "
        "sidecar, and the FEC6 selector length/payload. "
        f"{magic_note} The next artifact path should therefore be a "
        "runtime-adapter/no-op proof for the latent+sidecar semantic null span or "
        "a sidecar-only exact-eval probe, not another post-hoc recompression of "
        "the already-compressed archive member bytes."
    )


def _entropy_profile(payload: bytes) -> dict[str, Any]:
    if not payload:
        return {
            "shannon_bits_per_byte": 0.0,
            "shannon_floor_bytes": 0,
            "unique_byte_values": 0,
        }
    arr = np.frombuffer(payload, dtype=np.uint8)
    _vals, counts = np.unique(arr, return_counts=True)
    probs = counts.astype(np.float64) / float(arr.size)
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {
        "shannon_bits_per_byte": entropy,
        "shannon_floor_bytes": int(math.ceil(entropy * arr.size / 8.0)),
        "unique_byte_values": int(counts.size),
    }


def _generic_recompression_profile(payload: bytes) -> dict[str, int]:
    return {
        "source_bytes": len(payload),
        "brotli_q11_bytes": len(brotli.compress(payload, quality=11)),
        "lzma9_extreme_bytes": len(lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)),
        "zlib9_bytes": len(zlib.compress(payload, 9)),
    }


def _magic_byte_stream_probe(payload: bytes, *, section_name: str) -> dict[str, Any]:
    rows = []
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    for stream_type in ("categorical", "weight_tensor", "latent_sidecar"):
        try:
            result = encode_magic_codec(
                arr,
                hint=StreamHint(stream_type),
                selection_strategy="smallest_byte_count",
            )
        except MagicCodecError as exc:
            rows.append(
                {
                    "stream_type": stream_type,
                    "status": "refused",
                    "error": str(exc),
                    "payload_bytes": None,
                }
            )
            continue
        rows.append(
            {
                "stream_type": stream_type,
                "status": "ok",
                "payload_bytes": len(result.payload),
                "delta_vs_section_bytes": len(result.payload) - len(payload),
                "selected_primitive": result.selected_primitive,
                "selection_log": [
                    {
                        "primitive_name": item.primitive_name,
                        "encoded_bytes": len(item.encoded_bytes),
                        "refused": item.refused,
                        "refusal_reason": item.refusal_reason,
                    }
                    for item in result.selection_log
                ],
            }
        )
    ok = [row for row in rows if row["status"] == "ok"]
    best = min(ok, key=lambda row: int(row["payload_bytes"])) if ok else None
    return {
        "section_name": section_name,
        "best_payload_bytes": None if best is None else best["payload_bytes"],
        "best_delta_vs_section_bytes": None if best is None else best["delta_vs_section_bytes"],
        "best_stream_type": None if best is None else best["stream_type"],
        "best_selected_primitive": None if best is None else best["selected_primitive"],
        "best_direct_byte_stream_magic_candidate_beats_current": (
            False if best is None else int(best["payload_bytes"]) < len(payload)
        ),
        "candidates": rows,
        "score_claim": False,
    }


def _null_coverage(
    *,
    start: int,
    end: int,
    null_indices: np.ndarray | None,
    run_preview_limit: int,
) -> dict[str, Any]:
    length = end - start
    if null_indices is None:
        return {
            "n_null_bytes": 0,
            "null_fraction_within_section": 0.0,
            "contiguous_runs": [],
            "null_indices_provided": False,
        }
    mask = (null_indices >= start) & (null_indices < end)
    local = null_indices[mask]
    runs = _contiguous_runs(local)
    return {
        "n_null_bytes": int(local.size),
        "null_fraction_within_section": float(local.size / length) if length else 0.0,
        "contiguous_run_count": len(runs),
        "contiguous_runs": [[a, b] for a, b in runs[:run_preview_limit]],
        "contiguous_runs_truncated": len(runs) > run_preview_limit,
        "longest_contiguous_run_bytes": max((b - a for a, b in runs), default=0),
        "null_indices_provided": True,
    }


def _contiguous_runs(indices: Iterable[int] | np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices, dtype=np.int64)
    if arr.size == 0:
        return []
    arr = np.unique(arr)
    breaks = np.nonzero(np.diff(arr) != 1)[0] + 1
    parts = np.split(arr, breaks)
    return [(int(part[0]), int(part[-1]) + 1) for part in parts if part.size]


def _normalise_null_indices(values: np.ndarray | Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("null_indices must be one-dimensional")
    if np.any(arr < 0):
        raise ValueError("null_indices must be non-negative")
    return np.unique(arr)


def _validate_sections_cover_member_payload(
    sections: Sequence[SourceAnatomySection],
    member_payload_bytes: int,
) -> None:
    cursor = 0
    for section in sorted(sections, key=lambda item: item.member_start):
        if section.member_start != cursor:
            raise ValueError(
                "source anatomy sections do not cover member payload contiguously "
                f"at byte {cursor}"
            )
        cursor = section.member_end
    if cursor != member_payload_bytes:
        raise ValueError(
            "source anatomy sections do not cover full member payload: "
            f"covered={cursor} member_payload_bytes={member_payload_bytes}"
        )


def _validate_null_indices_coordinate_space(
    null_indices: np.ndarray,
    *,
    member_payload_bytes: int,
    sections: Sequence[SourceAnatomySection],
) -> None:
    if null_indices.size == 0:
        return
    max_index = int(null_indices[-1])
    if max_index >= member_payload_bytes:
        raise ValueError(
            "null_indices must be zero-based FEC6 member-payload byte offsets; "
            f"max_index={max_index} outside member_payload_bytes={member_payload_bytes}"
        )
    covered = _count_null_indices_covered_by_sections(null_indices, sections)
    if covered != int(null_indices.size):
        raise ValueError(
            "null_indices coordinate-space validation failed: "
            f"covered_by_sections={covered} count={int(null_indices.size)}"
        )


def _null_indices_metadata(
    null_indices: np.ndarray | None,
    *,
    member_payload_bytes: int,
    sections: Sequence[SourceAnatomySection],
) -> dict[str, Any] | None:
    if null_indices is None:
        return None
    covered = _count_null_indices_covered_by_sections(null_indices, sections)
    return {
        "count": int(null_indices.size),
        "sha256": sha256_hex(null_indices.astype("<i8", copy=False).tobytes()),
        "coordinate_space": "fec6_member_payload_byte_offsets_zero_based_half_open",
        "member_payload_bytes": int(member_payload_bytes),
        "bounds_validated_against_member_payload_bytes": True,
        "covered_by_semantic_sections_count": covered,
        "all_indices_covered_by_semantic_sections": covered == int(null_indices.size),
    }


def _count_null_indices_covered_by_sections(
    null_indices: np.ndarray,
    sections: Sequence[SourceAnatomySection],
) -> int:
    covered = 0
    for section in sections:
        mask = (null_indices >= section.member_start) & (null_indices < section.member_end)
        covered += int(mask.sum())
    return covered


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
