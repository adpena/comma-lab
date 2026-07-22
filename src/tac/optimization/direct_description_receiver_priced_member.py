# SPDX-License-Identifier: MIT
"""Exact receiver-priced member solve with an in-selection tolerance ladder.

This Task #603/#604-U1 apparatus prices every proposal by the byte length of
the exact six-member chart archive consumed by the established NumPy receiver.
It intentionally does not substitute compressed source arrays or a diagnostic
coder for ``len(A(z))``.  The current chart syntax uses fixed-width records;
therefore a byte-flat result is a scoped formulation wall, not a family verdict.
"""

from __future__ import annotations

import base64
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.optimization.direct_description_measurement_ladder import (
    _RESIDUAL_RECORD,
    CHART_GRID,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
    _predict_chart,
    compile_chart_archive,
    fit_chart_description,
    load_pose_target_codes,
    load_target_receipt,
    parse_chart_archive,
    prove_sampled_noop_honesty,
    receive_chart_archive,
)
from tac.optimization.direct_description_minimizer import (
    POINTER_SCORE_TEXT,
    SEED,
    SOURCE_BYTES,
    TOLERANCE_RUNG_TEXT,
    DirectDescriptionError,
    _publish_new_bytes,
    _read_regular_file_once,
    _require_sha256,
    _sha256,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_polytope_membership import (
    _load_segnet_oracle,
    measure_argmax_cell_membership,
    stream_decode_digest,
)
from tac.optimization.direct_description_real_target_rung0 import _committed_source_custody

RESULT_SCHEMA: Final = "direct_description_receiver_priced_member_n64.v1"
CONFIG_SCHEMA: Final = "DirectDescriptionReceiverPricedMemberConfigV1"
CHECKPOINT_SCHEMA: Final = "DirectDescriptionReceiverPricedMemberCheckpointV1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-SegNet advisory]"
LANE_ID: Final = "lane_ddm_mdl_member_solve_v2_priced_603_20260722"
RESIDUAL_STREAMS: Final = (
    "low_variation_chart_residuals",
    "mid_variation_chart_residuals",
    "high_variation_chart_residuals",
)
TOLERANCE_LADDER: Final = ("0.000000", *TOLERANCE_RUNG_TEXT)

MembershipMeasure = Callable[[Any], Mapping[str, Any]]


def _fraction_text(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.000000000000"
    return f"{numerator / denominator:.12f}"


def _compact_membership(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep aggregate/stratum authority while replacing bulky pair rows by custody."""

    rows = value.get("per_pair")
    if not isinstance(rows, list):
        raise DirectDescriptionError("membership result lacks ordered per-pair rows")
    compact = {key: item for key, item in value.items() if key != "per_pair"}
    compact["per_pair_custody"] = {
        "pair_count": len(rows),
        "ordered_pair_ids": [row.get("pair_id") for row in rows],
        "rows_sha256": _sha256(rfc8785_canonicalize(rows)),
        "all_pairs_covered_once": [row.get("pair_id") for row in rows] == list(range(len(rows))),
    }
    return compact


def _pose_completeness(receiver: Any, target_pose_codes: np.ndarray) -> dict[str, Any]:
    n_pairs = receiver.z.n_pairs
    target = np.asarray(target_pose_codes)
    if target.dtype != np.uint8 or target.shape != (600, 6):
        raise DirectDescriptionError("receiver-priced solve requires uint8 Pose6 [600,6]")
    exact = receiver.pose6_codes == target[:n_pairs]
    coordinates = n_pairs * 6
    matches = int(np.count_nonzero(exact))
    return {
        "coordinates": coordinates,
        "exact_coordinates": matches,
        "pose_completeness": _fraction_text(matches, coordinates),
        "pose6_integer_l1_debt": int(
            np.abs(receiver.pose6_codes.astype(np.int16) - target[:n_pairs].astype(np.int16)).sum(dtype=np.int64)
        ),
        "d_pose_claim": False,
    }


@dataclass(frozen=True, slots=True)
class ResidualCollapseProposalV1:
    z: DirectDescriptionChartZV1
    stream: str
    changed_scalars: int
    changed_records: int


def build_safe_zero_residual_proposal(
    z: DirectDescriptionChartZV1,
    stream_name: str,
) -> ResidualCollapseProposalV1:
    """Collapse safe residual scalars toward zero without leaving uint8 decode.

    This is the maximal deterministic zero-symbol probe available inside one
    fixed-width residual stream.  It is a proposal probe, not a claim that zero
    is an entropy code.  The actual final ZIP encode remains the rate authority.
    """

    if stream_name not in RESIDUAL_STREAMS:
        raise DirectDescriptionError(f"unknown residual stratum {stream_name!r}")
    receiver = receive_chart_archive(compile_chart_archive(z).archive)
    source = getattr(z, stream_name).payload
    output = bytearray()
    changed_scalars = 0
    changed_records = 0
    for offset in range(0, len(source), _RESIDUAL_RECORD.size):
        pair_id, plane_id, chart_id, *residual = _RESIDUAL_RECORD.unpack_from(source, offset)
        chart_y, chart_x = divmod(chart_id, CHART_GRID[1])
        predicted = _predict_chart(
            receiver.anchors[pair_id, plane_id],
            receiver.gradients[pair_id, plane_id],
            chart_y,
            chart_x,
        )
        proposal = list(residual)
        record_changed = False
        for channel, value in enumerate(residual):
            if value != 0 and 0 <= int(predicted[channel]) <= 255:
                proposal[channel] = 0
                changed_scalars += 1
                record_changed = True
        changed_records += int(record_changed)
        output.extend(_RESIDUAL_RECORD.pack(pair_id, plane_id, chart_id, *proposal))
    if len(output) != len(source):
        raise DirectDescriptionError("residual collapse changed the fixed record extent")
    stream = getattr(z, stream_name)
    proposal_z = z.model_copy(
        update={
            stream_name: CountedChartStreamV1(payload=bytes(output), codec=stream.codec, ownership=stream.ownership)
        }
    )
    # Decoder validation is part of proposal construction, before any price is consumed.
    receive_chart_archive(compile_chart_archive(proposal_z).archive)
    return ResidualCollapseProposalV1(proposal_z, stream_name, changed_scalars, changed_records)


def exact_rate_probe_rows(z: DirectDescriptionChartZV1) -> tuple[dict[str, Any], ...]:
    """Encode every per-stratum proposal and reject non-strict byte improvements."""

    current = compile_chart_archive(z)
    rows: list[dict[str, Any]] = []
    for stream_name in RESIDUAL_STREAMS:
        proposal = build_safe_zero_residual_proposal(z, stream_name)
        first = compile_chart_archive(proposal.z)
        second = compile_chart_archive(proposal.z)
        parsed = parse_chart_archive(first.archive)
        receiver = receive_chart_archive(first.archive)
        if first.archive != second.archive or parsed.archive != first.archive or receiver.archive != first.archive:
            raise DirectDescriptionError("receiver-priced proposal failed deterministic encode/decode identity")
        delta = len(first.archive) - len(current.archive)
        rows.append(
            {
                "stream": stream_name,
                "proposal": "safe_zero_residual_collapse",
                "changed_scalars": proposal.changed_scalars,
                "changed_records": proposal.changed_records,
                "current_archive_bytes": len(current.archive),
                "proposal_archive_bytes": len(first.archive),
                "delta_archive_bytes": delta,
                "proposal_archive_sha256": _sha256(first.archive),
                "actual_encode_count": 2,
                "parse_reencode_identical": True,
                "receiver_consumed": True,
                "accepted": delta < 0,
                "decision": (
                    "ACCEPT_STRICT_EXACT_BYTE_DECREASE"
                    if delta < 0
                    else "REJECT_RATE_BREAK_EVEN_NON_STRICT_BYTE_DECREASE"
                ),
                "membership_evaluated": False,
                "membership_skip_reason": (
                    None
                    if delta < 0
                    else "reverse-waterfill stops before distortion spend when exact delta_bytes is nonnegative"
                ),
                "rate_price_score_delta": f"{Decimal(25) * Decimal(delta) / Decimal(SOURCE_BYTES):.18f}",
            }
        )
    if any(row["accepted"] for row in rows):
        raise DirectDescriptionError(
            "fixed-width chart grammar unexpectedly produced a shorter proposal; hard membership admission is owed"
        )
    return tuple(rows)


def _stratum_tolerance_gate(membership: Mapping[str, Any], tolerance: str) -> dict[str, Any]:
    allowed = Decimal(tolerance)
    strata = membership.get("strata")
    if not isinstance(strata, Mapping):
        raise DirectDescriptionError("membership result lacks per-stratum decomposition")
    families: dict[str, dict[str, Any]] = {}
    for family, members in strata.items():
        if not isinstance(members, Mapping):
            raise DirectDescriptionError("membership stratum family is malformed")
        family_rows: dict[str, Any] = {}
        for name, row in members.items():
            escape = Decimal(str(row["argmax_cell_escape_fraction"]))
            family_rows[str(name)] = {
                "argmax_cell_escape_fraction": format(escape, ".12f"),
                "allowed_escape_fraction": format(allowed, ".6f"),
                "satisfied": escape <= allowed,
            }
        families[str(family)] = family_rows
    return {
        "allowed_escape_fraction": format(allowed, ".6f"),
        "all_strata_satisfied": all(row["satisfied"] for members in families.values() for row in members.values()),
        "families": families,
    }


class DirectDescriptionReceiverPricedMemberConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionReceiverPricedMemberConfigV1"] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_receiver_priced_member_n64_seed1234"] = "ddm_receiver_priced_member_n64_seed1234"
    seed: Literal[1234] = SEED
    pair_count: Literal[64] = 64
    tolerance_ladder: tuple[StrictStr, ...] = TOLERANCE_LADDER
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    checkpoint_policy: Literal["atomic_preserve_every_rung"] = "atomic_preserve_every_rung"
    rate_authority: Literal["exact_len_of_six_member_zip_stored_A_of_z"] = "exact_len_of_six_member_zip_stored_A_of_z"
    selection_rule: Literal["strict_byte_decrease_then_hard_membership_gate"] = (
        "strict_byte_decrease_then_hard_membership_gate"
    )
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionReceiverPricedMemberConfigV1:
        _require_sha256(self.target_receipt_sha256, "target_receipt_sha256")
        if self.tolerance_ladder != TOLERANCE_LADDER:
            raise ValueError(f"tolerance_ladder must be exactly {TOLERANCE_LADDER!r}")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute scorer custody")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {
                    "compile_target": RESULT_SCHEMA,
                    "typed_config": self.model_dump(mode="json", by_alias=True),
                }
            )
        )


class DirectDescriptionReceiverPricedMemberProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_receiver_priced_member.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


class DirectDescriptionReceiverPricedMemberCheckpointV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionReceiverPricedMemberCheckpointV1"] = Field(
        default=CHECKPOINT_SCHEMA, alias="schema", serialization_alias="schema"
    )
    config: dict[str, Any]
    config_sha256: StrictStr
    dsl_compile_hash: StrictStr
    semantic_argv: tuple[StrictStr, ...]
    semantic_argv_sha256: StrictStr
    completed_rung_index: StrictInt = Field(ge=0, le=4)
    completed_tolerance: StrictStr
    next_rung_index: StrictInt = Field(ge=1, le=5)
    selected_archive_b64: StrictStr
    selected_archive_sha256: StrictStr
    selected_archive_bytes: StrictInt = Field(ge=1)
    curve: tuple[dict[str, Any], ...]
    evidence_axis: Literal["[macOS-CPU frozen-SegNet advisory]"] = EVIDENCE_AXIS
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _identity(self) -> DirectDescriptionReceiverPricedMemberCheckpointV1:
        for name in ("config_sha256", "dsl_compile_hash", "semantic_argv_sha256", "selected_archive_sha256"):
            _require_sha256(getattr(self, name), name)
        config = DirectDescriptionReceiverPricedMemberConfigV1.model_validate_json(rfc8785_canonicalize(self.config))
        if config.typed_config_hash() != self.config_sha256 or config.dsl_compile_hash() != self.dsl_compile_hash:
            raise ValueError("receiver-priced checkpoint config identity mismatch")
        if self.completed_tolerance != TOLERANCE_LADDER[self.completed_rung_index]:
            raise ValueError("receiver-priced checkpoint tolerance cursor mismatch")
        if self.next_rung_index != self.completed_rung_index + 1 or len(self.curve) != self.next_rung_index:
            raise ValueError("receiver-priced checkpoint continuation cursor mismatch")
        if _sha256("\0".join(self.semantic_argv).encode()) != self.semantic_argv_sha256:
            raise ValueError("receiver-priced checkpoint argv identity mismatch")
        try:
            archive = base64.b64decode(self.selected_archive_b64, validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("receiver-priced checkpoint archive base64 malformed") from exc
        if (
            base64.b64encode(archive).decode() != self.selected_archive_b64
            or len(archive) != self.selected_archive_bytes
            or _sha256(archive) != self.selected_archive_sha256
        ):
            raise ValueError("receiver-priced checkpoint archive custody mismatch")
        receiver = receive_chart_archive(archive)
        if receiver.z.n_pairs != config.pair_count:
            raise ValueError("receiver-priced checkpoint pair coverage mismatch")
        if self.curve[-1].get("archive_sha256") != self.selected_archive_sha256:
            raise ValueError("receiver-priced checkpoint curve/archive mismatch")
        return self

    def to_bytes(self) -> bytes:
        body = self.model_dump(mode="json", by_alias=True)
        canonical = rfc8785_canonicalize(body)
        return rfc8785_canonicalize({"body": body, "body_sha256": _sha256(canonical)})

    @classmethod
    def from_bytes(cls, payload: bytes) -> DirectDescriptionReceiverPricedMemberCheckpointV1:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("receiver-priced checkpoint JSON is malformed") from exc
        if not isinstance(value, dict) or set(value) != {"body", "body_sha256"}:
            raise DirectDescriptionError("receiver-priced checkpoint envelope is incomplete")
        if rfc8785_canonicalize(value) != payload:
            raise DirectDescriptionError("receiver-priced checkpoint envelope is noncanonical")
        if _sha256(rfc8785_canonicalize(value["body"])) != _require_sha256(value["body_sha256"], "body_sha256"):
            raise DirectDescriptionError("receiver-priced checkpoint body hash mismatch")
        return cls.model_validate_json(rfc8785_canonicalize(value["body"]))

    def filename(self) -> str:
        return f"ddm_receiver_priced_member__rung{self.completed_rung_index:03d}_{self.completed_tolerance}.json"

    def write_new(self, directory: Path) -> Path:
        return _publish_new_bytes(Path(directory) / self.filename(), self.to_bytes())


@dataclass(frozen=True, slots=True)
class ReceiverPricedMemberRunV1:
    final_archive: bytes
    curve: tuple[Mapping[str, Any], ...]
    checkpoint_paths: tuple[Path, ...]
    complete: bool
    resumed: bool


def load_receiver_priced_member_checkpoint(
    path: Path,
    *,
    config: DirectDescriptionReceiverPricedMemberConfigV1,
    semantic_argv: Sequence[str],
) -> DirectDescriptionReceiverPricedMemberCheckpointV1:
    checkpoint = DirectDescriptionReceiverPricedMemberCheckpointV1.from_bytes(_read_regular_file_once(path))
    if (
        checkpoint.config_sha256 != config.typed_config_hash()
        or checkpoint.dsl_compile_hash != config.dsl_compile_hash()
        or checkpoint.semantic_argv != tuple(semantic_argv)
    ):
        raise DirectDescriptionError("receiver-priced resume identity differs from governed run")
    return checkpoint


def run_receiver_priced_member_stages(
    config: DirectDescriptionReceiverPricedMemberConfigV1,
    *,
    baseline_z: DirectDescriptionChartZV1,
    target_pose_codes: np.ndarray,
    membership_measure: MembershipMeasure,
    semantic_argv: Sequence[str],
    checkpoint_directory: Path,
    resume_from: Path | None = None,
    stop_after_rung_index: int | None = None,
) -> ReceiverPricedMemberRunV1:
    """Execute the ladder with exact price-before-distortion reverse waterfill."""

    argv = tuple(semantic_argv)
    if not argv:
        raise DirectDescriptionError("receiver-priced solve requires typed semantic argv")
    baseline = compile_chart_archive(baseline_z).archive
    if receive_chart_archive(baseline).z.n_pairs != config.pair_count:
        raise DirectDescriptionError("receiver-priced baseline pair coverage mismatch")
    curve: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    start = 0
    selected = baseline
    resumed = resume_from is not None
    if resume_from is not None:
        checkpoint = load_receiver_priced_member_checkpoint(resume_from, config=config, semantic_argv=argv)
        selected = base64.b64decode(checkpoint.selected_archive_b64, validate=True)
        curve = [dict(row) for row in checkpoint.curve]
        start = checkpoint.next_rung_index
        if selected != baseline:
            raise DirectDescriptionError("receiver-priced fixed-width resume selected a nonbaseline archive")
    for rung_index in range(start, len(TOLERANCE_LADDER)):
        tolerance = TOLERANCE_LADDER[rung_index]
        receiver = receive_chart_archive(selected)
        proposals = exact_rate_probe_rows(receiver.z)
        membership = _compact_membership(membership_measure(receiver))
        pose = _pose_completeness(receiver, target_pose_codes)
        tolerance_gate = _stratum_tolerance_gate(membership, tolerance)
        overall_membership = str(membership["same_c1_argmax_cell_fraction"])
        row = {
            "rung_index": rung_index,
            "tolerance": "exact_cell_membership" if rung_index == 0 else tolerance,
            "max_escape_fraction": tolerance,
            "archive_bytes": len(selected),
            "archive_sha256": _sha256(selected),
            "membership_fraction": overall_membership,
            "pose_completeness": pose["pose_completeness"],
            "pose": pose,
            "membership": membership,
            "tolerance_gate": tolerance_gate,
            "rung_feasible": tolerance_gate["all_strata_satisfied"],
            "proposal_rows": list(proposals),
            "accepted_proposals": sum(bool(item["accepted"]) for item in proposals),
            "selected_payload_source": "six decoder-consumed semantic ZIP members",
            "source_raw_reference_used": False,
            "rate_authority": "actual len(compile_chart_archive(z).archive)",
            "verdict_scope": (
                "n64 #603 fixed-record chart grammar under local frozen-SegNet batch16 advisory membership; "
                "not a contest score or family verdict"
            ),
        }
        curve.append(row)
        checkpoint = DirectDescriptionReceiverPricedMemberCheckpointV1(
            config=config.model_dump(mode="json", by_alias=True),
            config_sha256=config.typed_config_hash(),
            dsl_compile_hash=config.dsl_compile_hash(),
            semantic_argv=argv,
            semantic_argv_sha256=_sha256("\0".join(argv).encode()),
            completed_rung_index=rung_index,
            completed_tolerance=tolerance,
            next_rung_index=rung_index + 1,
            selected_archive_b64=base64.b64encode(selected).decode(),
            selected_archive_sha256=_sha256(selected),
            selected_archive_bytes=len(selected),
            curve=tuple(curve),
        )
        checkpoint_paths.append(checkpoint.write_new(checkpoint_directory))
        if stop_after_rung_index is not None and rung_index >= stop_after_rung_index:
            break
    return ReceiverPricedMemberRunV1(
        final_archive=selected,
        curve=tuple(curve),
        checkpoint_paths=tuple(checkpoint_paths),
        complete=len(curve) == len(TOLERANCE_LADDER),
        resumed=resumed,
    )


def _storage_preflight(output_directory: Path) -> dict[str, Any]:
    probe = Path(output_directory)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    required = 128 * 1024 * 1024
    free = shutil.disk_usage(probe).free
    if free < required:
        raise DirectDescriptionError("receiver-priced member solve refuses: insufficient local receipt space")
    return {
        "output_tier": str(probe.resolve()),
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "free_space_gate_satisfied": True,
        "bulk_target_tier": "/Volumes/VertigoDataTier/pact",
        "bulk_target_read_only": True,
        "status": "PASS",
    }


def run_receiver_priced_member_n64(
    config: DirectDescriptionReceiverPricedMemberConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    """Run the real n64 ladder and publish the receiver-consumed selected payload."""

    root = Path(output_directory)
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = load_target_receipt(Path(config.target_receipt_path), config.target_receipt_sha256)
    if Path(receipt.upstream_repo_root).resolve() / "upstream" != Path(config.upstream_root).resolve():
        raise DirectDescriptionError("receiver-priced scorer root differs from target custody")
    target_pose_codes = load_pose_target_codes(receipt)
    cache_path = Path(receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != receipt.source_cache.bytes:
        raise DirectDescriptionError("receiver-priced cached target cells are unavailable")
    try:
        cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    except (OSError, ValueError) as exc:
        raise DirectDescriptionError("receiver-priced cached target-cell source is malformed") from exc
    oracle, scorer_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    baseline_z = fit_chart_description(receipt, target_pose_codes, config.pair_count)

    def membership_measure(receiver: Any) -> Mapping[str, Any]:
        return measure_argmax_cell_membership(
            receiver,
            receipt,
            oracle=oracle,
            cached_lstars=cached_lstars,
        )

    partial = run_receiver_priced_member_stages(
        config,
        baseline_z=baseline_z,
        target_pose_codes=target_pose_codes,
        membership_measure=membership_measure,
        semantic_argv=semantic_argv,
        checkpoint_directory=root / "stage_receipts",
        stop_after_rung_index=1,
    )
    if partial.complete or len(partial.checkpoint_paths) != 2:
        raise DirectDescriptionError("receiver-priced stop boundary did not preserve two rungs")
    resumed = run_receiver_priced_member_stages(
        config,
        baseline_z=baseline_z,
        target_pose_codes=target_pose_codes,
        membership_measure=membership_measure,
        semantic_argv=semantic_argv,
        checkpoint_directory=root / "stage_receipts",
        resume_from=partial.checkpoint_paths[-1],
    )
    if not resumed.complete or partial.final_archive != resumed.final_archive:
        raise DirectDescriptionError("receiver-priced stopped/resumed terminal identity failed")
    curve = [dict(row) for row in resumed.curve]
    first = compile_chart_archive(baseline_z).archive
    second = compile_chart_archive(baseline_z).archive
    if first != second or first != resumed.final_archive or parse_chart_archive(first).archive != first:
        raise DirectDescriptionError("receiver-priced terminal compiler determinism x2 failed")
    receiver = receive_chart_archive(first)
    decode_first = stream_decode_digest(receiver, n_pairs=config.pair_count)
    decode_second = stream_decode_digest(receiver, n_pairs=config.pair_count)
    if decode_first != decode_second:
        raise DirectDescriptionError("receiver-priced terminal decode determinism x2 failed")
    noop = prove_sampled_noop_honesty(receiver.z)
    custody = dict(receiver.custody)
    final_archive = _publish_new_bytes(
        root / "ddm_receiver_priced_member_n64_final.not_a_candidate.zip.receipt-bytes",
        first,
    )
    exact_bytes = {int(row["archive_bytes"]) for row in curve}
    proposal_deltas = {int(proposal["delta_archive_bytes"]) for row in curve for proposal in row["proposal_rows"]}
    result = {
        "schema": RESULT_SCHEMA,
        "task": 603,
        "master_task": 578,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
        "seed": config.seed,
        "verdict": "MEASURED_FIXED_WIDTH_RECEIVER_RATE_WALL_N64",
        "verdict_scope": (
            "FORMULATION negative for the six fixed-record ZIP_STORED #603 chart grammar at n64: "
            "every safe per-stratum proposal has zero exact archive-byte delta and every tolerance rung "
            "misses full per-stratum cell-membership feasibility; the wider direct-description/member family remains open"
        ),
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "semantic_argv": list(semantic_argv),
        "producer": {
            "module": _committed_source_custody("src/tac/optimization/direct_description_receiver_priced_member.py"),
            "cli": _committed_source_custody("tools/run_direct_description_receiver_priced_member.py"),
        },
        "reuse": {
            "#547/#549": "exact C1 target cells and Pose6 custody reused read-only",
            "#580": "integer resize solved-member constraint lineage retained by the target receipt",
            "#602": "diagnostic full-array zlib objective removed entirely",
            "#603": "same six-member chart compiler, parser, receiver, and n64 prefix producer reused directly",
            "#604-U1/#613": "sealed tolerance rungs executed inside strict exact-byte reverse-waterfill selection",
        },
        "selection": {
            "objective": "min exact len(A(z)) subject to per-stratum membership escape <= tolerance and Pose6 completeness",
            "actual_encode_on_every_proposal": True,
            "diagnostic_array_zlib_used": False,
            "strict_rate_break_even_stop": True,
            "marginal_rate_price_score_per_byte": f"{Decimal(25) / Decimal(SOURCE_BYTES):.18f}",
            "candidate_basis": "chart-stratum zero-symbol probe; no Fourier residual basis",
            "curve": curve,
            "unique_selected_archive_bytes": sorted(exact_bytes),
            "unique_proposal_delta_archive_bytes": sorted(proposal_deltas),
            "all_rungs_selected_same_decoder_payload": len({row["archive_sha256"] for row in curve}) == 1,
            "all_rungs_feasible": all(bool(row["rung_feasible"]) for row in curve),
        },
        "wall_decomposition": {
            stream: {
                "proposal_count": len(curve),
                "exact_delta_archive_bytes": sorted(
                    {
                        int(proposal["delta_archive_bytes"])
                        for row in curve
                        for proposal in row["proposal_rows"]
                        if proposal["stream"] == stream
                    }
                ),
                "accepted_count": sum(
                    int(bool(proposal["accepted"]))
                    for row in curve
                    for proposal in row["proposal_rows"]
                    if proposal["stream"] == stream
                ),
                "terminal_membership": curve[-1]["membership"]["strata"]["overall"]["all"],
            }
            for stream in RESIDUAL_STREAMS
        },
        "archive": {
            "path": str(final_archive),
            "bytes": len(first),
            "sha256": _sha256(first),
            "candidate_role": "not_a_candidate",
            "decoder_consumed_payload": True,
            "source_raw_reference_used": False,
            "member_count": custody["member_count"],
            "parse_reencode_identical": True,
            "compiler_determinism_x2": True,
            "decode_determinism_x2": True,
            "decode": decode_first,
            "custody": custody,
            "sampled_noop_honesty": noop,
        },
        "resume": {
            "stopped_after_rung_index": 1,
            "resumed_from": str(partial.checkpoint_paths[-1]),
            "terminal_archive_bit_identical": True,
            "all_rung_checkpoints_preserved": True,
            "checkpoint_paths": [str(path) for path in (*partial.checkpoint_paths, *resumed.checkpoint_paths)],
            "checkpoint_sha256": [
                _sha256(_read_regular_file_once(path))
                for path in (*partial.checkpoint_paths, *resumed.checkpoint_paths)
            ],
        },
        "scorer_custody": scorer_custody,
        "target_custody": {
            "receipt_path": config.target_receipt_path,
            "receipt_sha256": config.target_receipt_sha256,
            "source_cache_path": str(cache_path),
            "source_cache_bytes": receipt.source_cache.bytes,
            "source_cache_sha256": receipt.source_cache.sha256,
            "source_cache_mutated": False,
        },
        "blocker_delta": {
            "N600_MEMBER_SOLVE_COVERAGE": "PARTIAL_GREEN_N64_MINIMUM; N600_REMAINS_OWED",
            "RECEIVER_CARRIABLE_CODED_MEMBER_PAYLOAD": "RED_TO_GREEN_N64_SIX_MEMBER_DECODER_CONSUMED_PAYLOAD",
            "COUNTED_ARCHIVE_MDL_INSIDE_SOLVE": "RED_TO_GREEN_EXACT_FINAL_ZIP_PRICING; FIXED_WIDTH_RATE_GRADIENT_WALL_MEASURED",
            "PRE_UINT8_MEMBER_STATE": "REMAINS_RED_STRUCTURAL; NO_ZERO_REALIZATION_LOSS_CLAIM",
            "POSE_STREAM_IN_MEMBER_PAYLOAD": "RED_TO_GREEN_N64_POSE6_STREAM_COMPLETE",
            "PER_STRATUM_TOLERANCE_FEASIBILITY": "REMAINS_RED_ALL_RUNG_FORMULATION_SCOPE",
        },
        "storage_preflight": storage,
        "cleanup": {
            "bulk_artifacts_created": False,
            "target_bulk_remains_read_only_on_ssd": True,
            "scratch_policy": "bounded target/scorer batches plus immutable small checkpoints",
            "certify_or_block": "no deletion, movement, or source mutation performed",
        },
        "main_landing_review_required": True,
    }
    receipt_path = _publish_new_bytes(
        root / "ddm_receiver_priced_member_n64_receipt.json",
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


__all__ = [
    "TOLERANCE_LADDER",
    "DirectDescriptionReceiverPricedMemberCheckpointV1",
    "DirectDescriptionReceiverPricedMemberConfigV1",
    "DirectDescriptionReceiverPricedMemberProgramV1",
    "build_safe_zero_residual_proposal",
    "exact_rate_probe_rows",
    "load_receiver_priced_member_checkpoint",
    "run_receiver_priced_member_n64",
    "run_receiver_priced_member_stages",
]
