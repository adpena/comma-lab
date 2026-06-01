# SPDX-License-Identifier: MIT
"""Rate-collapse transcodes for HPRC compact receiver packets.

The first HPRC rate blocker is not scorer replay; it is payload mass. This
module keeps the receiver contract intact while changing section storage:
legacy compact receiver sections can be losslessly wrapped in a charged,
archive-contained entropy wrapper that the shipped numpy receiver decodes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.archive_byte_profile import contest_rate_term
from tac.repo_io import sha256_file
from tac.substrates.hprc.archive import HprcSectionKind, pack_hprc_packet, parse_hprc_packet
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.hprc.learned_receiver import (
    decode_compact_receiver_packet,
    is_entropy_wrapped_compact_section,
    pack_compact_residual_quantized,
    pack_entropy_wrapped_compact_section,
    unwrap_entropy_wrapped_compact_section,
)

HPRC_RATE_COLLAPSE_VARIANT_SCHEMA = "hprc_rate_collapse_variant.v1"
HPRC_RATE_COLLAPSE_REPORT_SCHEMA = "hprc_rate_collapse_report.v1"
HPRC_RATE_COLLAPSE_EXACT_EXECUTION_SCHEMA = "hprc_rate_collapse_exact_execution_report.v1"
DEFAULT_RATE_COLLAPSE_SECTIONS: tuple[HprcSectionKind, ...] = (
    HprcSectionKind.DECODER_QW,
    HprcSectionKind.LATENTS_RC,
    HprcSectionKind.SELECTORS_RC,
    HprcSectionKind.RESIDUAL_RC,
    HprcSectionKind.RECEIVER_STATE,
)
DEFAULT_RESIDUAL_TOKEN_COLLAPSE_SPECS: tuple[str, ...] = (
    "dz4_qd1",
    "dz4_qd2",
    "dz0_qd6",
    "dz4_qd4",
    "dz6_qd2",
    "dz8_qd1",
    "dz0_qd10",
    "dz16_qd1",
    "dz32_qd1",
    "dz64_qd1",
    "dz96_qd16",
    "dz126_qd1",
)

_SECTION_ALIASES: dict[str, HprcSectionKind] = {
    "decoder": HprcSectionKind.DECODER_QW,
    "decoder_qw": HprcSectionKind.DECODER_QW,
    "latents": HprcSectionKind.LATENTS_RC,
    "latents_rc": HprcSectionKind.LATENTS_RC,
    "selectors": HprcSectionKind.SELECTORS_RC,
    "selectors_rc": HprcSectionKind.SELECTORS_RC,
    "residual": HprcSectionKind.RESIDUAL_RC,
    "residual_rc": HprcSectionKind.RESIDUAL_RC,
    "state": HprcSectionKind.RECEIVER_STATE,
    "receiver_state": HprcSectionKind.RECEIVER_STATE,
}


@dataclass(frozen=True)
class ResidualTokenCollapseSpec:
    """Deterministic lossy rate gate for compact residual tokens."""

    deadzone: int
    quant_divisor: int

    @property
    def variant_id(self) -> str:
        return f"residual_tokens_dz{self.deadzone}_qd{self.quant_divisor}"


def parse_rate_collapse_sections(raw: str | Sequence[str] | None) -> tuple[HprcSectionKind, ...]:
    """Parse CLI/config section tokens into canonical section kinds."""

    if raw is None:
        return DEFAULT_RATE_COLLAPSE_SECTIONS
    tokens: list[str] = []
    if isinstance(raw, str):
        tokens.extend(raw.replace(",", " ").split())
    else:
        for item in raw:
            tokens.extend(str(item).replace(",", " ").split())
    if not tokens:
        return DEFAULT_RATE_COLLAPSE_SECTIONS
    out: list[HprcSectionKind] = []
    for token in tokens:
        key = token.strip().lower()
        if not key:
            continue
        try:
            kind = _SECTION_ALIASES[key]
        except KeyError as exc:
            raise ValueError(f"unknown HPRC rate-collapse section token: {token!r}") from exc
        if kind not in out:
            out.append(kind)
    return tuple(out)


def parse_residual_token_collapse_specs(
    raw: str | Sequence[str] | None,
) -> tuple[ResidualTokenCollapseSpec, ...]:
    """Parse ``dz4_qd2`` or ``4:2`` residual token-collapse schedules."""

    tokens: list[str] = []
    if raw is None:
        tokens.extend(DEFAULT_RESIDUAL_TOKEN_COLLAPSE_SPECS)
    elif isinstance(raw, str):
        tokens.extend(raw.replace(",", " ").split())
    else:
        for item in raw:
            tokens.extend(str(item).replace(",", " ").split())
    out: list[ResidualTokenCollapseSpec] = []
    for token in tokens:
        key = token.strip().lower()
        if not key:
            continue
        if key.startswith("dz") and "_qd" in key:
            left, right = key[2:].split("_qd", 1)
        elif ":" in key:
            left, right = key.split(":", 1)
        else:
            raise ValueError(f"invalid residual collapse token: {token!r}")
        spec = ResidualTokenCollapseSpec(deadzone=int(left), quant_divisor=int(right))
        if spec.deadzone < 0:
            raise ValueError("residual collapse deadzone must be nonnegative")
        if spec.quant_divisor < 1:
            raise ValueError("residual collapse quant_divisor must be positive")
        if spec not in out:
            out.append(spec)
    return tuple(out)


def rate_collapse_variant_groups(
    sections: Sequence[HprcSectionKind],
) -> tuple[tuple[str, tuple[HprcSectionKind, ...]], ...]:
    """Return deterministic individual and cumulative section groups."""

    ordered = tuple(sorted(dict.fromkeys(sections), key=int))
    groups: list[tuple[str, tuple[HprcSectionKind, ...]]] = []
    for kind in ordered:
        groups.append((f"{kind.name.lower()}_only", (kind,)))
    if len(ordered) > 1:
        groups.append(("selected_sections", ordered))
    semantic = (
        HprcSectionKind.DECODER_QW,
        HprcSectionKind.LATENTS_RC,
        HprcSectionKind.SELECTORS_RC,
        HprcSectionKind.RESIDUAL_RC,
        HprcSectionKind.RECEIVER_STATE,
    )
    semantic_group = tuple(kind for kind in semantic if kind in ordered)
    if len(semantic_group) > 1 and semantic_group != ordered:
        groups.append(("semantic_selected_sections", semantic_group))
    return tuple(groups)


def transcode_compact_receiver_sections(
    packet_bytes: bytes | bytearray | memoryview,
    *,
    sections: Iterable[HprcSectionKind],
    brotli_quality: int = 11,
    force: bool = False,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Losslessly entropy-wrap selected compact-receiver sections.

    Sections are replaced only when the section-local wrapped payload is smaller
    unless ``force`` is true. The returned packet is parsed and semantically
    decoded before returning, so callers never receive a syntactically valid but
    receiver-broken byte stream.
    """

    packet = parse_hprc_packet(packet_bytes)
    decode_compact_receiver_packet(packet)
    section_map = packet.section_map()
    selected = tuple(sorted(dict.fromkeys(sections), key=int))
    rows: list[dict[str, Any]] = []
    for kind in selected:
        if kind not in section_map:
            rows.append(
                {
                    "schema": HPRC_RATE_COLLAPSE_VARIANT_SCHEMA,
                    "section": kind.name.lower(),
                    "status": "missing",
                    "legacy_payload_bytes": 0,
                    "wrapped_payload_bytes": 0,
                    "accepted": False,
                }
            )
            continue
        current = section_map[kind]
        legacy = unwrap_entropy_wrapped_compact_section(current, expected_kind=kind)
        wrapped = pack_entropy_wrapped_compact_section(
            kind,
            legacy,
            brotli_quality=brotli_quality,
        )
        accepted = bool(force or len(wrapped) < len(current))
        if accepted:
            section_map[kind] = wrapped
        rows.append(
            {
                "schema": HPRC_RATE_COLLAPSE_VARIANT_SCHEMA,
                "section": kind.name.lower(),
                "status": "wrapped" if accepted else "not_smaller_section_local",
                "already_wrapped": is_entropy_wrapped_compact_section(current),
                "legacy_payload_bytes": len(current),
                "raw_semantic_payload_bytes": len(legacy),
                "wrapped_payload_bytes": len(wrapped),
                "section_payload_delta_bytes": len(wrapped) - len(current),
                "legacy_payload_sha256": hashlib.sha256(current).hexdigest(),
                "raw_semantic_payload_sha256": hashlib.sha256(legacy).hexdigest(),
                "wrapped_payload_sha256": hashlib.sha256(wrapped).hexdigest(),
                "accepted": accepted,
            }
        )
    out = pack_hprc_packet(section_map, config=packet.config)
    decode_compact_receiver_packet(parse_hprc_packet(out))
    return out, rows


def transcode_compact_receiver_residual_tokens(
    packet_bytes: bytes | bytearray | memoryview,
    *,
    spec: ResidualTokenCollapseSpec,
    sections: Iterable[HprcSectionKind] = DEFAULT_RATE_COLLAPSE_SECTIONS,
    brotli_quality: int = 11,
) -> tuple[bytes, list[dict[str, Any]], dict[str, Any]]:
    """Coarsen residual tokens, then entropy-wrap the compact receiver packet.

    The receiver grammar remains unchanged: the lossy step only mutates the
    archive-contained int8 residual token field before normal section wrapping.
    This keeps replay and receiver-proof machinery reusable while giving the
    queue a cheap rate-first actuator.
    """

    packet = parse_hprc_packet(packet_bytes)
    compact = decode_compact_receiver_packet(packet)
    section_map = packet.section_map()
    original_q = np.asarray(compact.residual.q, dtype=np.int16)
    collapsed_q = collapse_residual_tokens(original_q, spec=spec)
    section_map[HprcSectionKind.RESIDUAL_RC] = pack_compact_residual_quantized(
        collapsed_q,
        scale=compact.residual.scale,
    )
    collapsed_packet = pack_hprc_packet(section_map, config=packet.config)
    out, rows = transcode_compact_receiver_sections(
        collapsed_packet,
        sections=sections,
        brotli_quality=brotli_quality,
        force=False,
    )
    delta = collapsed_q.astype(np.int16) - original_q.astype(np.int16)
    changed = delta != 0
    metrics = {
        "schema": "hprc_residual_token_collapse_metrics.v1",
        "variant_id": spec.variant_id,
        "deadzone": int(spec.deadzone),
        "quant_divisor": int(spec.quant_divisor),
        "token_count": int(original_q.size),
        "tokens_changed": int(changed.sum()),
        "tokens_changed_fraction": float(changed.sum() / max(original_q.size, 1)),
        "original_nonzero_tokens": int((original_q != 0).sum()),
        "collapsed_nonzero_tokens": int((collapsed_q != 0).sum()),
        "residual_q_l1": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "residual_q_mse": float(np.mean(delta.astype(np.float32) ** 2)) if delta.size else 0.0,
        "residual_scale": float(compact.residual.scale),
    }
    decode_compact_receiver_packet(parse_hprc_packet(out))
    return out, rows, metrics


def collapse_residual_tokens(
    q: np.ndarray,
    *,
    spec: ResidualTokenCollapseSpec,
) -> np.ndarray:
    """Apply a deterministic residual dead-zone plus lattice projection."""

    values = np.asarray(q, dtype=np.int16).copy()
    if spec.deadzone:
        values[np.abs(values) <= int(spec.deadzone)] = 0
    if spec.quant_divisor > 1:
        divisor = int(spec.quant_divisor)
        values = np.rint(values.astype(np.float32) / float(divisor)).astype(np.int16) * divisor
    return values.clip(-127, 127).astype(np.int16)


def build_rate_collapse_exact_execution_report(
    *,
    rate_collapse_report_path: str | Path,
    source_exact_bridge_path: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Convert a receiver-proven rate-collapse report into exact-gate input."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    report_path = _resolve(rate_collapse_report_path, root)
    bridge_path = _resolve(source_exact_bridge_path, root)
    report = _load_json_object(report_path)
    bridge = _load_json_object(bridge_path)
    artifact = report.get("artifact") if isinstance(report.get("artifact"), dict) else {}
    archive_path = _resolve_required(artifact.get("archive_path"), root, "archive_path")
    hprc_0bin_path = _resolve_required(artifact.get("hprc_0bin_path"), root, "hprc_0bin_path")
    receiver_proof_path = _resolve_required(
        artifact.get("receiver_proof_path"),
        root,
        "receiver_proof_path",
    )
    archive_sha = str(artifact.get("archive_sha256") or "")
    hprc_sha = str(artifact.get("hprc_0bin_sha256") or "")
    if sha256_file(archive_path) != archive_sha:
        raise ValueError("rate-collapse artifact archive sha256 mismatch")
    if sha256_file(hprc_0bin_path) != hprc_sha:
        raise ValueError("rate-collapse artifact HPRC 0.bin sha256 mismatch")
    receiver_proof = _load_json_object(receiver_proof_path)
    if receiver_proof.get("archive_sha256") != archive_sha:
        raise ValueError("rate-collapse receiver proof archive sha256 mismatch")
    source_archive = (
        bridge.get("archive") if isinstance(bridge.get("archive"), dict) else {}
    )
    source_mlx = (
        bridge.get("mlx_advisory_summary")
        if isinstance(bridge.get("mlx_advisory_summary"), dict)
        else {}
    )
    best_variant_id = report.get("best_variant_id")
    best_variant = next(
        (
            row
            for row in report.get("variants", [])
            if isinstance(row, dict) and row.get("variant_id") == best_variant_id
        ),
        {},
    )
    best_is_lossy = bool(best_variant.get("lossy_residual_token_collapse"))
    saved = int(report.get("rate_collapse_archive_bytes_saved") or 0)
    return {
        "schema": HPRC_RATE_COLLAPSE_EXACT_EXECUTION_SCHEMA,
        "repo_root": root.as_posix(),
        "source_exact_bridge_path": bridge_path.as_posix(),
        "source_exact_bridge_sha256": sha256_file(bridge_path),
        "rate_collapse_report_path": report_path.as_posix(),
        "rate_collapse_report_sha256": sha256_file(report_path),
        "candidate_id": bridge.get("candidate_id"),
        "candidate_variant_id": (
            f"{bridge.get('candidate_variant_id')}_rate_collapse_"
            f"{best_variant_id}"
        ),
        "residual_transform": (
            f"hprc_lossy_residual_token_rate_collapse_{best_variant_id}"
            if best_is_lossy
            else "lossless_hprc_section_entropy_rate_collapse"
        ),
        "archive": {
            "path": archive_path.as_posix(),
            "bytes": int(artifact.get("archive_bytes")),
            "sha256": archive_sha,
            "hprc_0bin_path": hprc_0bin_path.as_posix(),
            "hprc_0bin_sha256": hprc_sha,
            "source_archive_path": source_archive.get("path"),
            "source_archive_sha256": source_archive.get("sha256"),
            "source_archive_bytes": source_archive.get("bytes"),
            "rate_collapse_archive_bytes_saved": saved,
            "rate_collapse_rate_term_saved": contest_rate_term(saved),
        },
        "receiver_proof_binding": {
            "status": "linked_by_rate_collapse_receiver_proof",
            "proof_path": receiver_proof_path.as_posix(),
            "archive_sha256": archive_sha,
            "receiver_contract_satisfied": (
                receiver_proof.get("receiver_contract_satisfied") is True
            ),
            "runtime_consumption_proof_ready": (
                receiver_proof.get("runtime_consumption_proof_ready") is True
            ),
            "receiver_output_sha256": receiver_proof.get("receiver_output_sha256"),
            "receiver_output_bytes": receiver_proof.get("receiver_output_bytes"),
            "blockers": list(receiver_proof.get("blockers") or []),
        },
        "cleanup": {
            "status": "planned",
            "plan_path": None,
            "blocked_bytes_retained": 0,
            "reclaimable_bytes": 0,
            "blockers": [],
        },
        "incremental_summary": {
            **source_mlx,
            "archive_bytes_removed_vs_baseline": (
                int(source_archive.get("bytes") or 0)
                - int(artifact.get("archive_bytes") or 0)
            ),
            "rate_collapse_archive_bytes_saved": saved,
            "rate_collapse_rate_term_saved": contest_rate_term(saved),
            "rate_collapse_lossless_semantic_transcode": not best_is_lossy,
            "rate_collapse_best_variant_id": best_variant_id,
            "rate_collapse_lossy_residual_token_collapse": best_is_lossy,
        },
        "wall_clock_profile": bridge.get("wall_clock_profile"),
        "exact_axis_gate": {
            "ready_for_exact_eval_dispatch": False,
            "blockers": [
                "contest_cpu_cuda_exact_eval_not_executed",
                "mlx_local_response_is_advisory_not_score_authority",
            ],
        },
        **FALSE_AUTHORITY,
    }


__all__ = [
    "DEFAULT_RATE_COLLAPSE_SECTIONS",
    "DEFAULT_RESIDUAL_TOKEN_COLLAPSE_SPECS",
    "HPRC_RATE_COLLAPSE_EXACT_EXECUTION_SCHEMA",
    "HPRC_RATE_COLLAPSE_REPORT_SCHEMA",
    "HPRC_RATE_COLLAPSE_VARIANT_SCHEMA",
    "ResidualTokenCollapseSpec",
    "build_rate_collapse_exact_execution_report",
    "collapse_residual_tokens",
    "parse_rate_collapse_sections",
    "parse_residual_token_collapse_specs",
    "rate_collapse_variant_groups",
    "transcode_compact_receiver_residual_tokens",
    "transcode_compact_receiver_sections",
]


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else root / p


def _resolve_required(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"rate-collapse report missing {label}")
    path = _resolve(value, root)
    if not path.is_file():
        raise FileNotFoundError(f"rate-collapse {label} missing: {path}")
    return path


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload
