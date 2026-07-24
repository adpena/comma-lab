# SPDX-License-Identifier: MIT
"""Typed DM4 scorer-recursive proposal-source adapter for the J5 consumer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

SCHEMA: Final = "ddm_dm4_j5_proposal_adapter.v1"
PROPOSAL_SCHEMA: Final = "ddm_dm4_j5_proposal.v1"
DM4_SCHEMA: Final = "ddm_dm4_targeted_realization_cures.v1"


class DM4J5AdapterError(ValueError):
    """Fail-closed malformed DM4 receipt or ambiguous proposal provenance."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class DM4J5ProposalV1:
    proposal_id: str
    aimed_cell: Mapping[str, Any]
    corrected_j_row: Mapping[str, Any]
    support_footprint: Mapping[str, Any]
    proposal_type: Literal["seg-only", "pose-only(frame_0)", "joint"]
    candidate: Mapping[str, Any]
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise DM4J5AdapterError("DM4/J5 proposal ID is empty")
        if self.proposal_type not in {"seg-only", "pose-only(frame_0)", "joint"}:
            raise DM4J5AdapterError("DM4/J5 proposal visibility type differs")
        if not {"pair_id", "bucket_id", "row_index"} <= set(self.aimed_cell):
            raise DM4J5AdapterError("DM4/J5 proposal aimed-cell provenance is incomplete")
        if self.corrected_j_row.get("metric") != (
            "rank4 target-vs-runner SegNet head margin on categorical Fisher base"
        ):
            raise DM4J5AdapterError("DM4/J5 proposal lacks the corrected-J Fisher row")
        if self.corrected_j_row.get("projected_input_adjoint") != (
            "exact sum over the canonical disjoint factor2 preimage taps"
        ):
            raise DM4J5AdapterError("DM4/J5 proposal lacks exact resize-adjoint custody")
        if self.support_footprint.get("support_rule") != (
            "scorer-recursive; no disks, global writes, or history"
        ):
            raise DM4J5AdapterError("DM4/J5 proposal support is not scorer-recursive")
        if self.support_footprint.get("stem_stride") != 2:
            raise DM4J5AdapterError("DM4/J5 proposal support lacks stride-2 stem custody")
        if len(self.source_receipt_sha256) != 64:
            raise DM4J5AdapterError("DM4/J5 proposal source receipt SHA differs")
        candidate_id = str(self.candidate.get("candidate_id", ""))
        if not candidate_id.startswith("scorer_recursive_"):
            raise DM4J5AdapterError("DM4/J5 proposal candidate is not scorer-recursive")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_SCHEMA,
            "proposal_id": self.proposal_id,
            "aimed_cell": dict(self.aimed_cell),
            "corrected_J_row": dict(self.corrected_j_row),
            "support_footprint": dict(self.support_footprint),
            "type": self.proposal_type,
            "candidate": dict(self.candidate),
            "source_receipt_sha256": self.source_receipt_sha256,
        }


def _read_receipt(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise DM4J5AdapterError(f"DM4 receipt is unavailable: {path}")
    raw = path.read_bytes()
    actual = _sha256(raw)
    if actual != expected_sha256:
        raise DM4J5AdapterError(f"DM4 receipt SHA differs: {actual} != {expected_sha256}")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != DM4_SCHEMA:
        raise DM4J5AdapterError("DM4 receipt schema differs")
    if payload.get("score_claim") is not False or payload.get("pointer_moved") is not False:
        raise DM4J5AdapterError("DM4 receipt authority boundary differs")
    return payload, actual


def _support_for_candidate(search: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    if "erf0.5" in candidate_id:
        target_fraction = 0.5
    elif "erf0.9" in candidate_id:
        target_fraction = 0.9
    else:
        raise DM4J5AdapterError(
            f"scorer-recursive candidate {candidate_id} does not bind an ERF energy fraction"
        )
    matches = [
        row
        for row in search.get("scorer_recursive_write_supports", ())
        if float(row.get("energy_fraction", -1.0)) == target_fraction
    ]
    if len(matches) != 1:
        raise DM4J5AdapterError(
            f"candidate {candidate_id} has {len(matches)} matching support footprints"
        )
    return matches[0]


def adapt_dm4_proposals(
    *,
    receipt_path: Path,
    receipt_sha256: str,
    base_archive: bytes,
    enabled: bool,
) -> tuple[bytes, tuple[DM4J5ProposalV1, ...], dict[str, Any]]:
    """Expose DM4 proposals; disabled mode preserves archive bytes exactly."""

    payload, actual_sha = _read_receipt(receipt_path, receipt_sha256)
    if not enabled:
        receipt = {
            "schema": SCHEMA,
            "enabled": False,
            "proposal_count": 0,
            "base_archive_sha256": _sha256(base_archive),
            "output_archive_sha256": _sha256(base_archive),
            "byte_identical_disabled": True,
            "source_receipt_sha256": actual_sha,
        }
        return base_archive, (), receipt

    proposals: list[DM4J5ProposalV1] = []
    for row in payload.get("rows", ()):
        search = row.get("search")
        if not isinstance(search, Mapping) or not isinstance(search.get("pullback"), Mapping):
            continue
        candidate_rows = (
            *search.get("probes", ()),
            *search.get("pose_parent_probes", ()),
        )
        for candidate in candidate_rows:
            if not isinstance(candidate, Mapping):
                raise DM4J5AdapterError("DM4 probe row must be a mapping")
            candidate_id = str(candidate.get("candidate_id", ""))
            if not candidate_id.startswith("scorer_recursive_"):
                continue
            if candidate.get("semantic_record_exact") is not True:
                continue
            support = _support_for_candidate(search, candidate_id)
            proposal_id = (
                f"dm4.row{int(row['row_index']):02d}.pair{int(row['pair_id']):03d}."
                f"{candidate_id}"
            )
            proposals.append(
                DM4J5ProposalV1(
                    proposal_id=proposal_id,
                    aimed_cell={
                        "row_index": int(row["row_index"]),
                        "pair_id": int(row["pair_id"]),
                        "bucket_id": str(row["bucket_id"]),
                        "support_seed_blocks": int(support["support_seed_blocks"]),
                    },
                    corrected_j_row=dict(search["pullback"]),
                    support_footprint={
                        "schema": support["schema"],
                        "stem_stride": int(support["stem_stride"]),
                        "erf_r50_pixels": float(support["erf_r50_pixels"]),
                        "energy_fraction": float(support["energy_fraction"]),
                        "stem_block_indices": list(support["stem_block_indices"]),
                        "stem_block_indices_sha256_uint32le": support[
                            "stem_block_indices_sha256_uint32le"
                        ],
                        "support_rule": support["support_rule"],
                    },
                    # DM4 candidates can affect both frames/channels.  No
                    # candidate is silently promoted to a narrower visibility
                    # class from pose non-harm alone.
                    proposal_type="joint",
                    candidate=dict(candidate),
                    source_receipt_sha256=actual_sha,
                )
            )
    if not proposals:
        raise DM4J5AdapterError("enabled DM4 adapter produced no scorer-recursive exact proposals")
    ordered = tuple(sorted(proposals, key=lambda proposal: proposal.proposal_id))
    if len({proposal.proposal_id for proposal in ordered}) != len(ordered):
        raise DM4J5AdapterError("DM4/J5 proposal IDs are not unique")
    receipt = {
        "schema": SCHEMA,
        "enabled": True,
        "proposal_count": len(ordered),
        "base_archive_sha256": _sha256(base_archive),
        "output_archive_sha256": _sha256(base_archive),
        "adapter_mutates_archive": False,
        "source_receipt_sha256": actual_sha,
        "proposal_manifest_sha256": _sha256(
            _canonical_bytes([proposal.to_payload() for proposal in ordered])
        ),
    }
    return base_archive, ordered, receipt


__all__ = [
    "PROPOSAL_SCHEMA",
    "SCHEMA",
    "DM4J5AdapterError",
    "DM4J5ProposalV1",
    "adapt_dm4_proposals",
]
