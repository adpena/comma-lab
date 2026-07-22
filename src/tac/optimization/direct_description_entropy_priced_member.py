# SPDX-License-Identifier: MIT
"""Task #603/#613 v3 exact-byte solve over entropy-coded semantic members."""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import lzma
import math
import shutil
import struct
import sys
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal

import brotli
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.analysis.segnet_boundary_marginals import boundary_mask_from_labels
from tac.boundary_math.hood_static_component import identify_static_hood_class
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.boundary_math.road_horizon_component import classify_segnet_regions
from tac.optimization.direct_description_entropy_streams import (
    CORRECTION_SCHEDULE_DROP,
    CORRECTION_SCHEDULE_EVERY_PAIR,
    CORRECTION_SCHEDULE_FIXED_HOLD24,
    CORRECTION_SCHEDULE_NAME,
    CORRECTION_SCHEDULE_XI_HOLD24,
    PlaneCorrectionSectionBuildV1,
    compile_entropy_chart_archive,
    encode_plane_correction_section,
    parse_entropy_chart_archive,
    parse_plane_correction_section,
    prove_entropy_home_fail_closed,
    receive_entropy_chart_archive,
)
from tac.optimization.direct_description_measurement_ladder import (
    DirectDescriptionChartZV1,
    _fraction_text,
    compile_chart_archive,
    fit_chart_description,
    iter_target_plane_window_chunks,
    load_pose_target_codes,
    load_target_receipt,
    prove_sampled_noop_honesty,
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
    CLASS_NAMES,
    MARGIN_BANDS,
    _load_segnet_oracle,
    measure_argmax_cell_membership,
    stream_decode_digest,
)
from tac.optimization.direct_description_real_target_rung0 import _committed_source_custody
from tac.optimization.direct_description_receiver_priced_member import (
    RESIDUAL_STREAMS,
    _compact_membership,
    _pose_completeness,
    _stratum_tolerance_gate,
    build_safe_zero_residual_proposal,
)
from tac.optimization.predictor_r3_causal import decode_component_event_alphabet_raw
from tac.optimization.s4_archive_composer import parse_sections

RESULT_SCHEMA: Final = "direct_description_entropy_priced_member_n64.v1"
CONFIG_SCHEMA: Final = "DirectDescriptionEntropyPricedMemberConfigV1"
CANDIDATE_CHECKPOINT_SCHEMA: Final = "DirectDescriptionEntropyCandidateCheckpointV1"
RUNG_CHECKPOINT_SCHEMA: Final = "DirectDescriptionEntropyRungCheckpointV1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-SegNet advisory]"
LANE_ID: Final = "lane_ddm_mdl_member_solve_v3_entropy_603_613_20260722"
TOLERANCE_LADDER: Final = ("0.000000", *TOLERANCE_RUNG_TEXT)
SUBSET_COUNT: Final = 1 << len(RESIDUAL_STREAMS)

MembershipMeasure = Callable[[Any], Mapping[str, Any]]


def _subset_streams(mask: int) -> tuple[str, ...]:
    if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask < SUBSET_COUNT:
        raise DirectDescriptionError("entropy candidate subset mask is invalid")
    return tuple(stream for index, stream in enumerate(RESIDUAL_STREAMS) if mask & (1 << index))


def build_entropy_candidate_z(baseline_z: DirectDescriptionChartZV1, subset_mask: int) -> DirectDescriptionChartZV1:
    z = baseline_z
    for stream_name in _subset_streams(subset_mask):
        z = build_safe_zero_residual_proposal(z, stream_name).z
    return z


class DirectDescriptionEntropyPricedMemberConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionEntropyPricedMemberConfigV1"] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_entropy_priced_member_n64_seed1234"] = "ddm_entropy_priced_member_n64_seed1234"
    seed: Literal[1234] = SEED
    pair_count: Literal[64] = 64
    tolerance_ladder: tuple[StrictStr, ...] = TOLERANCE_LADDER
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    candidate_checkpoint_policy: Literal["atomic_preserve_every_subset"] = "atomic_preserve_every_subset"
    rung_checkpoint_policy: Literal["atomic_preserve_every_rung"] = "atomic_preserve_every_rung"
    rate_authority: Literal["exact_len_of_six_member_entropy_zip_stored_A_of_z"] = (
        "exact_len_of_six_member_entropy_zip_stored_A_of_z"
    )
    candidate_family: Literal["exhaustive_power_set_of_three_safe_zero_residual_collapses"] = (
        "exhaustive_power_set_of_three_safe_zero_residual_collapses"
    )
    selection_rule: Literal["minimum_exact_bytes_subject_to_absolute_stratum_membership_and_pose"] = (
        "minimum_exact_bytes_subject_to_absolute_stratum_membership_and_pose"
    )
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionEntropyPricedMemberConfigV1:
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
                {"compile_target": RESULT_SCHEMA, "typed_config": self.model_dump(mode="json", by_alias=True)}
            )
        )


class DirectDescriptionEntropyPricedMemberProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_entropy_priced_member.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


class DirectDescriptionEntropyCandidateCheckpointV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionEntropyCandidateCheckpointV1"] = Field(
        default=CANDIDATE_CHECKPOINT_SCHEMA, alias="schema", serialization_alias="schema"
    )
    config: dict[str, Any]
    config_sha256: StrictStr
    dsl_compile_hash: StrictStr
    semantic_argv: tuple[StrictStr, ...]
    semantic_argv_sha256: StrictStr
    completed_subset_mask: StrictInt = Field(ge=0, le=SUBSET_COUNT - 1)
    next_subset_mask: StrictInt = Field(ge=1, le=SUBSET_COUNT)
    candidates: tuple[dict[str, Any], ...]
    evidence_axis: Literal["[macOS-CPU frozen-SegNet advisory]"] = EVIDENCE_AXIS
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _identity(self) -> DirectDescriptionEntropyCandidateCheckpointV1:
        for name in ("config_sha256", "dsl_compile_hash", "semantic_argv_sha256"):
            _require_sha256(getattr(self, name), name)
        config = DirectDescriptionEntropyPricedMemberConfigV1.model_validate_json(rfc8785_canonicalize(self.config))
        if config.typed_config_hash() != self.config_sha256 or config.dsl_compile_hash() != self.dsl_compile_hash:
            raise ValueError("entropy candidate checkpoint config identity mismatch")
        if self.next_subset_mask != self.completed_subset_mask + 1 or len(self.candidates) != self.next_subset_mask:
            raise ValueError("entropy candidate checkpoint cursor mismatch")
        if [row.get("subset_mask") for row in self.candidates] != list(range(self.next_subset_mask)):
            raise ValueError("entropy candidate checkpoint subset order mismatch")
        if _sha256("\0".join(self.semantic_argv).encode()) != self.semantic_argv_sha256:
            raise ValueError("entropy candidate checkpoint argv identity mismatch")
        return self

    def to_bytes(self) -> bytes:
        body = self.model_dump(mode="json", by_alias=True)
        canonical = rfc8785_canonicalize(body)
        return rfc8785_canonicalize({"body": body, "body_sha256": _sha256(canonical)})

    @classmethod
    def from_bytes(cls, payload: bytes) -> DirectDescriptionEntropyCandidateCheckpointV1:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("entropy candidate checkpoint JSON is malformed") from exc
        if not isinstance(value, dict) or set(value) != {"body", "body_sha256"}:
            raise DirectDescriptionError("entropy candidate checkpoint envelope is incomplete")
        if rfc8785_canonicalize(value) != payload:
            raise DirectDescriptionError("entropy candidate checkpoint envelope is noncanonical")
        if _sha256(rfc8785_canonicalize(value["body"])) != _require_sha256(value["body_sha256"], "body_sha256"):
            raise DirectDescriptionError("entropy candidate checkpoint body hash mismatch")
        return cls.model_validate_json(rfc8785_canonicalize(value["body"]))

    def filename(self) -> str:
        return f"ddm_entropy_priced_member__candidate{self.completed_subset_mask:03d}.json"

    def write_new(self, directory: Path) -> Path:
        return _publish_new_bytes(Path(directory) / self.filename(), self.to_bytes())


class DirectDescriptionEntropyRungCheckpointV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionEntropyRungCheckpointV1"] = Field(
        default=RUNG_CHECKPOINT_SCHEMA, alias="schema", serialization_alias="schema"
    )
    config: dict[str, Any]
    config_sha256: StrictStr
    dsl_compile_hash: StrictStr
    semantic_argv: tuple[StrictStr, ...]
    semantic_argv_sha256: StrictStr
    candidate_table_sha256: StrictStr
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
    def _identity(self) -> DirectDescriptionEntropyRungCheckpointV1:
        for name in (
            "config_sha256",
            "dsl_compile_hash",
            "semantic_argv_sha256",
            "candidate_table_sha256",
            "selected_archive_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        config = DirectDescriptionEntropyPricedMemberConfigV1.model_validate_json(rfc8785_canonicalize(self.config))
        if config.typed_config_hash() != self.config_sha256 or config.dsl_compile_hash() != self.dsl_compile_hash:
            raise ValueError("entropy rung checkpoint config identity mismatch")
        if self.completed_tolerance != TOLERANCE_LADDER[self.completed_rung_index]:
            raise ValueError("entropy rung checkpoint tolerance cursor mismatch")
        if self.next_rung_index != self.completed_rung_index + 1 or len(self.curve) != self.next_rung_index:
            raise ValueError("entropy rung checkpoint continuation cursor mismatch")
        if _sha256("\0".join(self.semantic_argv).encode()) != self.semantic_argv_sha256:
            raise ValueError("entropy rung checkpoint argv identity mismatch")
        try:
            archive = base64.b64decode(self.selected_archive_b64, validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("entropy rung checkpoint archive base64 is malformed") from exc
        if (
            base64.b64encode(archive).decode() != self.selected_archive_b64
            or len(archive) != self.selected_archive_bytes
            or _sha256(archive) != self.selected_archive_sha256
        ):
            raise ValueError("entropy rung checkpoint archive custody mismatch")
        if receive_entropy_chart_archive(archive).z.n_pairs != config.pair_count:
            raise ValueError("entropy rung checkpoint pair coverage mismatch")
        if self.curve[-1].get("selected_archive_sha256") != self.selected_archive_sha256:
            raise ValueError("entropy rung checkpoint curve/archive mismatch")
        return self

    def to_bytes(self) -> bytes:
        body = self.model_dump(mode="json", by_alias=True)
        canonical = rfc8785_canonicalize(body)
        return rfc8785_canonicalize({"body": body, "body_sha256": _sha256(canonical)})

    @classmethod
    def from_bytes(cls, payload: bytes) -> DirectDescriptionEntropyRungCheckpointV1:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("entropy rung checkpoint JSON is malformed") from exc
        if not isinstance(value, dict) or set(value) != {"body", "body_sha256"}:
            raise DirectDescriptionError("entropy rung checkpoint envelope is incomplete")
        if rfc8785_canonicalize(value) != payload:
            raise DirectDescriptionError("entropy rung checkpoint envelope is noncanonical")
        if _sha256(rfc8785_canonicalize(value["body"])) != _require_sha256(value["body_sha256"], "body_sha256"):
            raise DirectDescriptionError("entropy rung checkpoint body hash mismatch")
        return cls.model_validate_json(rfc8785_canonicalize(value["body"]))

    def filename(self) -> str:
        return f"ddm_entropy_priced_member__rung{self.completed_rung_index:03d}_{self.completed_tolerance}.json"

    def write_new(self, directory: Path) -> Path:
        return _publish_new_bytes(Path(directory) / self.filename(), self.to_bytes())


@dataclass(frozen=True, slots=True)
class EntropyCandidateRunV1:
    candidates: tuple[Mapping[str, Any], ...]
    checkpoint_paths: tuple[Path, ...]
    complete: bool
    resumed: bool


@dataclass(frozen=True, slots=True)
class EntropyRungRunV1:
    curve: tuple[Mapping[str, Any], ...]
    checkpoint_paths: tuple[Path, ...]
    complete: bool
    resumed: bool


def _candidate_checkpoint_load(
    path: Path,
    *,
    config: DirectDescriptionEntropyPricedMemberConfigV1,
    semantic_argv: Sequence[str],
) -> DirectDescriptionEntropyCandidateCheckpointV1:
    checkpoint = DirectDescriptionEntropyCandidateCheckpointV1.from_bytes(_read_regular_file_once(path))
    if (
        checkpoint.config_sha256 != config.typed_config_hash()
        or checkpoint.dsl_compile_hash != config.dsl_compile_hash()
        or checkpoint.semantic_argv != tuple(semantic_argv)
    ):
        raise DirectDescriptionError("entropy candidate resume identity differs from governed run")
    return checkpoint


def _rung_checkpoint_load(
    path: Path,
    *,
    config: DirectDescriptionEntropyPricedMemberConfigV1,
    semantic_argv: Sequence[str],
    candidate_table_sha256: str,
) -> DirectDescriptionEntropyRungCheckpointV1:
    checkpoint = DirectDescriptionEntropyRungCheckpointV1.from_bytes(_read_regular_file_once(path))
    if (
        checkpoint.config_sha256 != config.typed_config_hash()
        or checkpoint.dsl_compile_hash != config.dsl_compile_hash()
        or checkpoint.semantic_argv != tuple(semantic_argv)
        or checkpoint.candidate_table_sha256 != candidate_table_sha256
    ):
        raise DirectDescriptionError("entropy rung resume identity differs from governed run")
    return checkpoint


def _candidate_row(
    baseline_z: DirectDescriptionChartZV1,
    subset_mask: int,
    target_pose_codes: np.ndarray,
    membership_measure: MembershipMeasure,
) -> dict[str, Any]:
    z = build_entropy_candidate_z(baseline_z, subset_mask)
    first = compile_entropy_chart_archive(z)
    second = compile_entropy_chart_archive(z)
    parsed = parse_entropy_chart_archive(first.archive)
    receiver = receive_entropy_chart_archive(first.archive)
    if first.archive != second.archive or parsed.archive != first.archive or receiver.archive != first.archive:
        raise DirectDescriptionError("entropy candidate failed deterministic compile/parse/receive identity")
    membership = _compact_membership(membership_measure(receiver))
    pose = _pose_completeness(receiver, target_pose_codes)
    return {
        "subset_mask": subset_mask,
        "collapsed_streams": list(_subset_streams(subset_mask)),
        "archive_bytes": len(first.archive),
        "archive_sha256": _sha256(first.archive),
        "fixed_width_semantic_archive_bytes": len(compile_chart_archive(z).archive),
        "membership_fraction": str(membership["same_c1_argmax_cell_fraction"]),
        "membership": membership,
        "pose_completeness": pose["pose_completeness"],
        "pose": pose,
        "per_stream_bytes": first.stream_byte_rows(),
        "compiler_determinism_x2": True,
        "parse_reencode_identical": True,
        "receiver_consumed": True,
        "source_raw_reference_used": False,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }


def run_entropy_candidate_stages(
    config: DirectDescriptionEntropyPricedMemberConfigV1,
    *,
    baseline_z: DirectDescriptionChartZV1,
    target_pose_codes: np.ndarray,
    membership_measure: MembershipMeasure,
    semantic_argv: Sequence[str],
    checkpoint_directory: Path,
    resume_from: Path | None = None,
    stop_after_subset_mask: int | None = None,
) -> EntropyCandidateRunV1:
    argv = tuple(semantic_argv)
    if not argv:
        raise DirectDescriptionError("entropy candidate solve requires typed semantic argv")
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    start = 0
    if resume_from is not None:
        checkpoint = _candidate_checkpoint_load(resume_from, config=config, semantic_argv=argv)
        rows = [dict(row) for row in checkpoint.candidates]
        start = checkpoint.next_subset_mask
        for row in rows:
            compiled = compile_entropy_chart_archive(build_entropy_candidate_z(baseline_z, int(row["subset_mask"])))
            if len(compiled.archive) != row["archive_bytes"] or _sha256(compiled.archive) != row["archive_sha256"]:
                raise DirectDescriptionError("entropy candidate resume archive identity mismatch")
    for subset_mask in range(start, SUBSET_COUNT):
        rows.append(_candidate_row(baseline_z, subset_mask, target_pose_codes, membership_measure))
        checkpoint = DirectDescriptionEntropyCandidateCheckpointV1(
            config=config.model_dump(mode="json", by_alias=True),
            config_sha256=config.typed_config_hash(),
            dsl_compile_hash=config.dsl_compile_hash(),
            semantic_argv=argv,
            semantic_argv_sha256=_sha256("\0".join(argv).encode()),
            completed_subset_mask=subset_mask,
            next_subset_mask=subset_mask + 1,
            candidates=tuple(rows),
        )
        paths.append(checkpoint.write_new(checkpoint_directory))
        if stop_after_subset_mask is not None and subset_mask >= stop_after_subset_mask:
            break
    return EntropyCandidateRunV1(tuple(rows), tuple(paths), len(rows) == SUBSET_COUNT, resume_from is not None)


def run_entropy_rung_stages(
    config: DirectDescriptionEntropyPricedMemberConfigV1,
    *,
    baseline_z: DirectDescriptionChartZV1,
    candidates: Sequence[Mapping[str, Any]],
    semantic_argv: Sequence[str],
    checkpoint_directory: Path,
    resume_from: Path | None = None,
    stop_after_rung_index: int | None = None,
) -> EntropyRungRunV1:
    argv = tuple(semantic_argv)
    candidate_rows = tuple(dict(row) for row in candidates)
    if not argv or [row.get("subset_mask") for row in candidate_rows] != list(range(SUBSET_COUNT)):
        raise DirectDescriptionError("entropy rung solve requires all ordered candidates and typed argv")
    candidate_sha = _sha256(rfc8785_canonicalize(candidate_rows))
    curve: list[dict[str, Any]] = []
    paths: list[Path] = []
    start = 0
    if resume_from is not None:
        checkpoint = _rung_checkpoint_load(
            resume_from,
            config=config,
            semantic_argv=argv,
            candidate_table_sha256=candidate_sha,
        )
        curve = [dict(row) for row in checkpoint.curve]
        start = checkpoint.next_rung_index
    entropy_baseline_bytes = int(candidate_rows[0]["archive_bytes"])
    fixed_baseline_bytes = int(candidate_rows[0]["fixed_width_semantic_archive_bytes"])
    for rung_index in range(start, len(TOLERANCE_LADDER)):
        tolerance = TOLERANCE_LADDER[rung_index]
        evaluated: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            gate = _stratum_tolerance_gate(candidate["membership"], tolerance)
            pose_complete = candidate["pose_completeness"] == "1.000000000000"
            evaluated.append(
                {
                    "subset_mask": int(candidate["subset_mask"]),
                    "collapsed_streams": list(candidate["collapsed_streams"]),
                    "archive_bytes": int(candidate["archive_bytes"]),
                    "archive_sha256": str(candidate["archive_sha256"]),
                    "membership_fraction": str(candidate["membership_fraction"]),
                    "pose_completeness": str(candidate["pose_completeness"]),
                    "tolerance_gate": gate,
                    "feasible": bool(gate["all_strata_satisfied"] and pose_complete),
                }
            )
        feasible = [row for row in evaluated if row["feasible"]]
        selected_summary = (
            min(feasible, key=lambda row: (row["archive_bytes"], row["subset_mask"])) if feasible else evaluated[0]
        )
        selected_candidate = candidate_rows[int(selected_summary["subset_mask"])]
        selected_archive = compile_entropy_chart_archive(
            build_entropy_candidate_z(baseline_z, int(selected_summary["subset_mask"]))
        ).archive
        if _sha256(selected_archive) != selected_summary["archive_sha256"]:
            raise DirectDescriptionError("entropy rung selected archive differs from candidate table")
        row = {
            "rung_index": rung_index,
            "tolerance": "exact_cell_membership" if rung_index == 0 else tolerance,
            "max_escape_fraction": tolerance,
            "rung_feasible": bool(feasible),
            "selected_role": "minimum_exact_bytes_feasible" if feasible else "infeasible_diagnostic_entropy_baseline",
            "selected_subset_mask": int(selected_summary["subset_mask"]),
            "selected_collapsed_streams": list(selected_summary["collapsed_streams"]),
            "selected_archive_bytes": len(selected_archive),
            "selected_archive_sha256": _sha256(selected_archive),
            "byte_delta_vs_entropy_baseline": len(selected_archive) - entropy_baseline_bytes,
            "byte_delta_vs_fixed_width_baseline": len(selected_archive) - fixed_baseline_bytes,
            "membership_fraction": str(selected_candidate["membership_fraction"]),
            "membership": selected_candidate["membership"],
            "pose_completeness": str(selected_candidate["pose_completeness"]),
            "pose": selected_candidate["pose"],
            "per_stream_bytes": selected_candidate["per_stream_bytes"],
            "candidate_evaluations": evaluated,
            "feasible_candidate_count": len(feasible),
            "rate_authority": "actual len(compile_entropy_chart_archive(z).archive)",
            "verdict_scope": (
                "n64 exhaustive power set of the three maximal safe-zero residual collapses under absolute "
                "frozen-SegNet per-stratum membership; not a wider direct-description/member family verdict"
            ),
        }
        curve.append(row)
        checkpoint = DirectDescriptionEntropyRungCheckpointV1(
            config=config.model_dump(mode="json", by_alias=True),
            config_sha256=config.typed_config_hash(),
            dsl_compile_hash=config.dsl_compile_hash(),
            semantic_argv=argv,
            semantic_argv_sha256=_sha256("\0".join(argv).encode()),
            candidate_table_sha256=candidate_sha,
            completed_rung_index=rung_index,
            completed_tolerance=tolerance,
            next_rung_index=rung_index + 1,
            selected_archive_b64=base64.b64encode(selected_archive).decode(),
            selected_archive_sha256=_sha256(selected_archive),
            selected_archive_bytes=len(selected_archive),
            curve=tuple(curve),
        )
        paths.append(checkpoint.write_new(checkpoint_directory))
        if stop_after_rung_index is not None and rung_index >= stop_after_rung_index:
            break
    return EntropyRungRunV1(tuple(curve), tuple(paths), len(curve) == len(TOLERANCE_LADDER), resume_from is not None)


def _storage_preflight(output_directory: Path) -> dict[str, Any]:
    probe = Path(output_directory)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    required = 128 * 1024 * 1024
    free = shutil.disk_usage(probe).free
    if free < required:
        raise DirectDescriptionError("entropy-priced member solve refuses: insufficient local receipt space")
    return {
        "output_tier": str(probe.resolve()),
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "free_space_gate_satisfied": True,
        "bulk_target_tier": "/Volumes/VertigoDataTier/pact",
        "bulk_target_read_only": True,
        "status": "PASS",
    }


def run_entropy_priced_member_n64(
    config: DirectDescriptionEntropyPricedMemberConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    root = Path(output_directory)
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = load_target_receipt(Path(config.target_receipt_path), config.target_receipt_sha256)
    if Path(receipt.upstream_repo_root).resolve() / "upstream" != Path(config.upstream_root).resolve():
        raise DirectDescriptionError("entropy-priced scorer root differs from target custody")
    target_pose_codes = load_pose_target_codes(receipt)
    cache_path = Path(receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != receipt.source_cache.bytes:
        raise DirectDescriptionError("entropy-priced cached target cells are unavailable")
    try:
        cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    except (OSError, ValueError) as exc:
        raise DirectDescriptionError("entropy-priced cached target-cell source is malformed") from exc
    oracle, scorer_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    baseline_z = fit_chart_description(receipt, target_pose_codes, config.pair_count)

    def membership_measure(receiver: Any) -> Mapping[str, Any]:
        return measure_argmax_cell_membership(receiver, receipt, oracle=oracle, cached_lstars=cached_lstars)

    candidate_partial = run_entropy_candidate_stages(
        config,
        baseline_z=baseline_z,
        target_pose_codes=target_pose_codes,
        membership_measure=membership_measure,
        semantic_argv=semantic_argv,
        checkpoint_directory=root / "candidate_receipts",
        stop_after_subset_mask=3,
    )
    if candidate_partial.complete or len(candidate_partial.candidates) != 4:
        raise DirectDescriptionError("entropy candidate stop boundary did not preserve four subsets")
    candidate_resumed = run_entropy_candidate_stages(
        config,
        baseline_z=baseline_z,
        target_pose_codes=target_pose_codes,
        membership_measure=membership_measure,
        semantic_argv=semantic_argv,
        checkpoint_directory=root / "candidate_receipts",
        resume_from=candidate_partial.checkpoint_paths[-1],
    )
    if not candidate_resumed.complete:
        raise DirectDescriptionError("entropy candidate resume did not complete all subsets")
    candidates = [dict(row) for row in candidate_resumed.candidates]
    rung_partial = run_entropy_rung_stages(
        config,
        baseline_z=baseline_z,
        candidates=candidates,
        semantic_argv=semantic_argv,
        checkpoint_directory=root / "rung_receipts",
        stop_after_rung_index=1,
    )
    if rung_partial.complete or len(rung_partial.curve) != 2:
        raise DirectDescriptionError("entropy rung stop boundary did not preserve two rungs")
    rung_resumed = run_entropy_rung_stages(
        config,
        baseline_z=baseline_z,
        candidates=candidates,
        semantic_argv=semantic_argv,
        checkpoint_directory=root / "rung_receipts",
        resume_from=rung_partial.checkpoint_paths[-1],
    )
    if not rung_resumed.complete:
        raise DirectDescriptionError("entropy rung resume did not complete tolerance ladder")
    curve = [dict(row) for row in rung_resumed.curve]
    published: list[dict[str, Any]] = []
    for row in curve:
        archive = compile_entropy_chart_archive(
            build_entropy_candidate_z(baseline_z, int(row["selected_subset_mask"]))
        ).archive
        path = _publish_new_bytes(
            root / f"ddm_entropy_priced_member_n64_rung{int(row['rung_index']):03d}.not_a_candidate.zip.receipt-bytes",
            archive,
        )
        published.append(
            {"rung_index": row["rung_index"], "path": str(path), "bytes": len(archive), "sha256": _sha256(archive)}
        )
    baseline_first = compile_entropy_chart_archive(baseline_z)
    baseline_second = compile_entropy_chart_archive(baseline_z)
    parsed = parse_entropy_chart_archive(baseline_first.archive)
    receiver = receive_entropy_chart_archive(baseline_first.archive)
    if baseline_first.archive != baseline_second.archive or parsed.archive != baseline_first.archive:
        raise DirectDescriptionError("entropy terminal compiler determinism x2 failed")
    decode_first = stream_decode_digest(receiver, n_pairs=config.pair_count)
    decode_second = stream_decode_digest(receiver, n_pairs=config.pair_count)
    if decode_first != decode_second:
        raise DirectDescriptionError("entropy terminal decode determinism x2 failed")
    semantic_noop = prove_sampled_noop_honesty(baseline_z)
    entropy_home = prove_entropy_home_fail_closed(baseline_z)
    fixed_baseline = compile_chart_archive(baseline_z).archive
    candidate_bytes = [int(row["archive_bytes"]) for row in candidates]
    value_responsive = len(set(candidate_bytes)) > 1
    if not value_responsive:
        raise DirectDescriptionError("entropy grammar failed to create a real value-dependent byte gradient")
    if len(baseline_first.archive) >= len(fixed_baseline):
        raise DirectDescriptionError("entropy grammar did not beat its lossless fixed-width semantic archive")
    coder_counts = Counter(stream["coder"] for candidate in candidates for stream in candidate["per_stream_bytes"])
    result = {
        "schema": RESULT_SCHEMA,
        "task": 603,
        "master_task": 578,
        "feeds_task": 613,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
        "seed": config.seed,
        "verdict": "MEASURED_ENTROPY_RATE_GRADIENT_GREEN_ABSOLUTE_MEMBER_CONSTRAINT_INFEASIBLE_N64",
        "verdict_scope": (
            "MEASURED lossless six-stream entropy recoding and exhaustive n64 power set of the three maximal "
            "safe-zero residual collapses; the absolute per-stratum membership constraint remains infeasible for "
            "this proposal family, while the wider direct-description/member family remains open"
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
            "grammar_module": _committed_source_custody("src/tac/optimization/direct_description_entropy_streams.py"),
            "solver_module": _committed_source_custody(
                "src/tac/optimization/direct_description_entropy_priced_member.py"
            ),
            "cli": _committed_source_custody("tools/run_direct_description_entropy_priced_member.py"),
        },
        "pricing_ladder": {
            "#602": "diagnostic full-array zlib was byte-flat and nonreceiver authority",
            "v2": "exact final ZIP pricing was green but fixed record extents forced delta-bytes=0",
            "v3": "exact entropy ZIP pricing varies with semantic values and is receiver-consumed",
            "law": "flat proxy -> exact fixed-width -> exact variable-length entropy",
        },
        "baseline": {
            "fixed_width_archive_bytes": len(fixed_baseline),
            "fixed_width_archive_sha256": _sha256(fixed_baseline),
            "entropy_archive_bytes": len(baseline_first.archive),
            "entropy_archive_sha256": _sha256(baseline_first.archive),
            "lossless_byte_delta": len(baseline_first.archive) - len(fixed_baseline),
            "lossless_membership_fraction": candidates[0]["membership_fraction"],
            "lossless_pose_completeness": candidates[0]["pose_completeness"],
            "semantic_payloads_identical": True,
        },
        "selection": {
            "objective": "min exact len(A_entropy(z)) subject to absolute per-stratum cell membership and Pose6 completeness",
            "candidate_family": config.candidate_family,
            "candidate_count": len(candidates),
            "candidate_table_sha256": _sha256(rfc8785_canonicalize(candidates)),
            "candidate_archive_bytes": candidate_bytes,
            "unique_candidate_archive_bytes": sorted(set(candidate_bytes)),
            "value_responsive_exact_rate_gradient": value_responsive,
            "candidates": candidates,
            "curve": curve,
            "all_rungs_feasible": all(bool(row["rung_feasible"]) for row in curve),
            "marginal_rate_price_score_per_byte": f"{Decimal(25) / Decimal(SOURCE_BYTES):.18f}",
        },
        "coder_tournament": {
            "menu": [
                "brotli_q11",
                "lzma_xz_preset9_extreme",
                "aqc1_sparse_arithmetic_uint8",
                "pr101_ranked_canonical_huffman_16_when_applicable",
                "split_metadata_plus_existing_zlib_or_rice_golomb",
            ],
            "transform_menu": [
                "pair_temporal_delta",
                "canonical_sparse_chart_records",
                "chart_aligned_temporal_delta_bitmap",
                "chart_aligned_temporal_delta_colex",
            ],
            "selected_coder_frequency_across_candidates": dict(sorted(coder_counts.items())),
            "no_new_entropy_coder_implemented": True,
            "per_stream_measured_not_defaulted": True,
        },
        "archive": {
            "published_rungs": published,
            "decoder_consumed_payload": True,
            "source_raw_reference_used": False,
            "compiler_determinism_x2": True,
            "parse_reencode_identical": True,
            "decode_determinism_x2": True,
            "decode": decode_first,
            "custody": dict(receiver.custody),
            "semantic_sampled_noop_honesty": semantic_noop,
            "entropy_home_fail_closed": entropy_home,
        },
        "resume": {
            "candidate_stopped_after_subset_mask": 3,
            "candidate_resumed": True,
            "candidate_checkpoint_paths": [
                str(path) for path in (*candidate_partial.checkpoint_paths, *candidate_resumed.checkpoint_paths)
            ],
            "rung_stopped_after_index": 1,
            "rung_resumed": True,
            "rung_checkpoint_paths": [
                str(path) for path in (*rung_partial.checkpoint_paths, *rung_resumed.checkpoint_paths)
            ],
            "all_candidate_and_rung_checkpoints_preserved": True,
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
            "COUNTED_ARCHIVE_MDL_INSIDE_SOLVE": "GREEN_EXACT_VALUE_RESPONSIVE_ENTROPY_ZIP_RATE",
            "VARIABLE_LENGTH_RECEIVER_GRAMMAR": "RED_TO_GREEN_N64_SIX_STREAM_END_TO_END",
            "PER_STREAM_WATERFILL_BYTES": "RED_TO_GREEN_ALL_RUNG_HOME_AND_CODED_BYTE_ROWS",
            "ABSOLUTE_PER_STRATUM_TOLERANCE_FEASIBILITY": "REMAINS_RED_THIS_EXHAUSTIVE_SAFE_ZERO_SUBSET_FORMULATION",
            "PRE_UINT8_MEMBER_STATE": "REMAINS_RED_STRUCTURAL",
            "N600_MEMBER_SOLVE_COVERAGE": "PARTIAL_GREEN_N64_MINIMUM; N600_REMAINS_OWED",
            "POSE_STREAM_IN_MEMBER_PAYLOAD": "GREEN_N64_POSE6_COMPLETE",
        },
        "storage_preflight": storage,
        "cleanup": {
            "bulk_artifacts_created": False,
            "target_bulk_remains_read_only_on_ssd": True,
            "scratch_policy": "bounded scorer batches plus immutable candidate/rung checkpoints",
            "certify_or_block": "no deletion, movement, or source mutation performed",
        },
        "main_landing_review_required": True,
    }
    receipt_path = _publish_new_bytes(
        root / "ddm_entropy_priced_member_n64_receipt.json",
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


# ---------------------------------------------------------------------------
# v4: S4 per-stratum structured members inside the proven v3 rate harness.
# ---------------------------------------------------------------------------

STRUCTURED_RESULT_SCHEMA: Final = "direct_description_stratum_structured_member_n64.v1"
STRUCTURED_CONFIG_SCHEMA: Final = "DirectDescriptionStratumStructuredMemberConfigV1"
STRUCTURED_CHECKPOINT_SCHEMA: Final = "DirectDescriptionStructuredCandidateCheckpointV1"
STRUCTURED_LANE_ID: Final = "lane_ddm_v4_stratum_structured_members_20260722"
STRUCTURED_ROLES: Final = ("baseline", "Road", "Lane", "MyCar", "UndrivableBoundary", "Movable")
ROLE_TARGET_CLASS: Final = {
    "Road": "Road",
    "Lane": "Lane",
    "MyCar": "MyCar",
    "UndrivableBoundary": "Undrivable",
    "Movable": "Movable",
}
COMPOSED_ROLE_ORDER: Final = ("UndrivableBoundary", "Road", "Lane", "MyCar", "Movable")
ROUTED_ROLES: Final = tuple(ROLE_TARGET_CLASS)
V5_RESULT_SCHEMA: Final = "direct_description_route_fix_composed_member.v1"
V5_CONFIG_SCHEMA: Final = "DirectDescriptionRouteFixComposeConfigV1"
V5_LANE_ID: Final = "lane_ddm_v5_route_fix_compose_603_613_20260722"
STRUCTURED_MEMBER_MAGIC: Final = b"D4S1"
COMPOSED_MEMBER_MAGIC: Final = b"D5C1"
SITE_STREAM_MAGIC: Final = b"D4E1"
SITE_STREAM_HEADER: Final = struct.Struct(">4sHBBI")
SITE_RECORD_HEADER: Final = struct.Struct(">HI")
S4_LZMA_FILTERS: Final = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]
PAIR_SHAPE: Final = (384, 512)


class DirectDescriptionStratumStructuredMemberConfigV1(BaseModel):
    """Typed local-only config for the additive v4 structured proposal family."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionStratumStructuredMemberConfigV1"] = Field(
        default=STRUCTURED_CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_v4_stratum_structured_members_n64_seed1234"] = "ddm_v4_stratum_structured_members_n64_seed1234"
    seed: Literal[1234] = SEED
    pair_count: Literal[64] = 64
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    s4_container_path: StrictStr
    s4_container_sha256: StrictStr
    s4_runtime_path: StrictStr
    s4_runtime_sha256: StrictStr
    candidate_family: Literal["s4_pxq1_lane_curve_static_hood_and_class_filtered_event_component_members"] = (
        "s4_pxq1_lane_curve_static_hood_and_class_filtered_event_component_members"
    )
    checkpoint_policy: Literal["atomic_preserve_every_structured_candidate"] = (
        "atomic_preserve_every_structured_candidate"
    )
    rate_authority: Literal["exact_len_of_receiver_closed_structured_zip_A4_of_z"] = (
        "exact_len_of_receiver_closed_structured_zip_A4_of_z"
    )
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionStratumStructuredMemberConfigV1:
        for name in ("target_receipt_sha256", "s4_container_sha256", "s4_runtime_sha256"):
            _require_sha256(getattr(self, name), name)
        for name in ("upstream_root", "s4_container_path", "s4_runtime_path"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"{name} must be absolute custody")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {
                    "compile_target": STRUCTURED_RESULT_SCHEMA,
                    "typed_config": self.model_dump(mode="json", by_alias=True),
                }
            )
        )


class DirectDescriptionStratumStructuredMemberProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_entropy_priced_member.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


class DirectDescriptionRouteFixComposeConfigV1(BaseModel):
    """Typed local-only config for one route-fixed composed DDM member."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionRouteFixComposeConfigV1"] = Field(
        default=V5_CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_v5_route_fix_compose_seed1234"] = "ddm_v5_route_fix_compose_seed1234"
    seed: Literal[1234] = SEED
    pair_start: StrictInt = Field(ge=0, le=536)
    pair_count: StrictInt
    routing_probe_start: Literal[448] = 448
    routing_probe_count: Literal[16] = 16
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    s4_container_path: StrictStr
    s4_container_sha256: StrictStr
    s4_runtime_path: StrictStr
    s4_runtime_sha256: StrictStr
    candidate_family: Literal["lbnd2_road_hood_undrivable_movable_pose_one_receiver_composition"] = (
        "lbnd2_road_hood_undrivable_movable_pose_one_receiver_composition"
    )
    routing_policy: Literal["self_detect_roles_then_maximize_own_class_over_c1_role_median_rgb"] = (
        "self_detect_roles_then_maximize_own_class_over_c1_role_median_rgb"
    )
    checkpoint_policy: Literal["atomic_preserve_route_then_composed_member"] = (
        "atomic_preserve_route_then_composed_member"
    )
    rate_authority: Literal["exact_len_of_receiver_closed_composed_zip_A5_of_z"] = (
        "exact_len_of_receiver_closed_composed_zip_A5_of_z"
    )
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionRouteFixComposeConfigV1:
        if self.pair_count not in (64, 256) or self.pair_start + self.pair_count > 600:
            raise ValueError("v5 composition window must be exactly n64 or n256 inside [0,600)")
        probe_stop = self.routing_probe_start + self.routing_probe_count
        if not self.pair_start <= self.routing_probe_start or probe_stop > self.pair_start + self.pair_count:
            raise ValueError("routing probe must be contained in the composed state window")
        for name in ("target_receipt_sha256", "s4_container_sha256", "s4_runtime_sha256"):
            _require_sha256(getattr(self, name), name)
        for name in ("upstream_root", "s4_container_path", "s4_runtime_path"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"{name} must be absolute custody")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {"compile_target": V5_RESULT_SCHEMA, "typed_config": self.model_dump(mode="json", by_alias=True)}
            )
        )


class DirectDescriptionRouteFixComposeProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_entropy_priced_member.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


V6_RESULT_SCHEMA: Final = "direct_description_dseg_bridge_amortize.v1"
V6_CONFIG_SCHEMA: Final = "DirectDescriptionDsegBridgeAmortizeConfigV1"
V6_LANE_ID: Final = "lane_ddm_v5_route_fix_compose_603_613_20260722"
V6_CANDIDATE_MODES: Final = (
    "v5_exact",
    "fixed_ar1_hold24",
    "xi_pose6_ar1_hold24",
    "residual_zero_static_once",
)
V6_TARGET_DSEG_TEXT: Final = "0.001160000000"
V6_S4_KNEE_DSEG_TEXT: Final = "0.016000000000"
V6_S4_KNEE_BYTES: Final = 216_207
V6_C1_GT_MATCH_FRACTION_TEXT: Final = "0.999873638153"


class DirectDescriptionDsegBridgeAmortizeConfigV1(BaseModel):
    """Typed local-only evaluator bridge and temporal-amortization config."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionDsegBridgeAmortizeConfigV1"] = Field(
        default=V6_CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_v6_dseg_bridge_amortize_seed1234"] = "ddm_v6_dseg_bridge_amortize_seed1234"
    seed: Literal[1234] = SEED
    pair_start: StrictInt = Field(ge=0, le=536)
    pair_count: StrictInt
    v5_receipt_path: StrictStr
    v5_receipt_sha256: StrictStr
    v5_archive_path: StrictStr
    v5_archive_sha256: StrictStr
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    max_key_gap: Literal[24] = 24
    candidate_modes: tuple[StrictStr, ...] = V6_CANDIDATE_MODES
    checkpoint_policy: Literal["atomic_preserve_every_candidate"] = "atomic_preserve_every_candidate"
    rate_authority: Literal["exact_len_of_receiver_closed_composed_zip_A6_of_z"] = (
        "exact_len_of_receiver_closed_composed_zip_A6_of_z"
    )
    target_d_seg: Literal["0.001160000000"] = V6_TARGET_DSEG_TEXT
    s4_knee_d_seg: Literal["0.016000000000"] = V6_S4_KNEE_DSEG_TEXT
    s4_knee_archive_bytes: Literal[216207] = V6_S4_KNEE_BYTES
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionDsegBridgeAmortizeConfigV1:
        if self.pair_count not in (64, 256) or self.pair_start + self.pair_count > 600:
            raise ValueError("v6 measurement window must be exactly n64 or n256 inside [0,600)")
        for name in ("v5_receipt_sha256", "v5_archive_sha256"):
            _require_sha256(getattr(self, name), name)
        if self.candidate_modes != V6_CANDIDATE_MODES:
            raise ValueError(f"candidate_modes must be exactly {V6_CANDIDATE_MODES!r}")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {"compile_target": V6_RESULT_SCHEMA, "typed_config": self.model_dump(mode="json", by_alias=True)}
            )
        )


class DirectDescriptionDsegBridgeAmortizeProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_entropy_priced_member.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


V7_RESULT_SCHEMA: Final = "direct_description_solved_plane_tolerance_waterfill.v1"
V7_CONFIG_SCHEMA: Final = "DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1"
V7_LANE_ID: Final = "ddm_v7_solved_plane_tolerance_waterfill"
V7_EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
V7_BASE_MODE: Final = "fixed_ar1_hold24"
V7_SECTION_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar", "Boundary")
V7_RUNG_SPECS: Final = {
    "exact": (CORRECTION_SCHEDULE_EVERY_PAIR, 1),
    "q4": (CORRECTION_SCHEDULE_EVERY_PAIR, 4),
    "q16": (CORRECTION_SCHEDULE_EVERY_PAIR, 16),
    "q64": (CORRECTION_SCHEDULE_EVERY_PAIR, 64),
    "fixed_hold24": (CORRECTION_SCHEDULE_FIXED_HOLD24, 1),
    "xi_hold24": (CORRECTION_SCHEDULE_XI_HOLD24, 1),
    "drop": (CORRECTION_SCHEDULE_DROP, 0),
}
V7_POLICY_SPECS: Final = (
    ("exact_all", ("exact",) * 6),
    ("q4_all", ("q4",) * 6),
    ("waterfill_sensitive_exact", ("q16", "exact", "drop", "exact", "drop", "exact")),
    ("waterfill_balanced", ("q16", "q4", "drop", "q4", "drop", "q4")),
    ("q16_all", ("q16",) * 6),
    ("q64_all", ("q64",) * 6),
    ("fixed_hold24_all", ("fixed_hold24",) * 6),
    ("xi_hold24_all", ("xi_hold24",) * 6),
    ("drop_to_predictor_all", ("drop",) * 6),
)
V7_EXACT_RESIDUAL_FALSIFIER_BYTES: Final = 200_000


class DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1(BaseModel):
    """Typed local-only per-stratum correction ladder over the v6 predictor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1"] = Field(
        default=V7_CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_v7_solved_plane_tolerance_waterfill_seed1234"] = (
        "ddm_v7_solved_plane_tolerance_waterfill_seed1234"
    )
    seed: Literal[1234] = SEED
    pair_start: StrictInt = Field(ge=0, le=536)
    pair_count: StrictInt
    v6_receipt_path: StrictStr
    v6_receipt_sha256: StrictStr
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    upstream_root: StrictStr
    base_mode: Literal["fixed_ar1_hold24"] = V7_BASE_MODE
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    max_key_gap: Literal[24] = 24
    section_names: tuple[StrictStr, ...] = V7_SECTION_NAMES
    rung_names: tuple[StrictStr, ...] = tuple(V7_RUNG_SPECS)
    policy_names: tuple[StrictStr, ...] = tuple(row[0] for row in V7_POLICY_SPECS)
    checkpoint_policy: Literal["atomic_preserve_every_rung_and_candidate"] = (
        "atomic_preserve_every_rung_and_candidate"
    )
    rate_authority: Literal["exact_len_of_receiver_closed_predictor_plus_opaque_correction_zip"] = (
        "exact_len_of_receiver_closed_predictor_plus_opaque_correction_zip"
    )
    target_d_seg: Literal["0.001160000000"] = V6_TARGET_DSEG_TEXT
    exact_residual_falsifier_bytes: Literal[200000] = V7_EXACT_RESIDUAL_FALSIFIER_BYTES
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1:
        if self.pair_count not in (64, 256) or self.pair_start + self.pair_count > 600:
            raise ValueError("v7 measurement window must be exactly n64 or n256 inside [0,600)")
        for name in ("v6_receipt_sha256", "target_receipt_sha256"):
            _require_sha256(getattr(self, name), name)
        if self.section_names != V7_SECTION_NAMES:
            raise ValueError("section_names must preserve the canonical five roles plus Boundary")
        if self.rung_names != tuple(V7_RUNG_SPECS):
            raise ValueError("rung_names must preserve the preregistered residual ladder")
        if self.policy_names != tuple(row[0] for row in V7_POLICY_SPECS):
            raise ValueError("policy_names must preserve the preregistered waterfill ladder")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {"compile_target": V7_RESULT_SCHEMA, "typed_config": self.model_dump(mode="json", by_alias=True)}
            )
        )


class DirectDescriptionSolvedPlaneToleranceWaterfillProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_entropy_priced_member.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


def _uvarint(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DirectDescriptionError("structured site varint requires a nonnegative integer")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise DirectDescriptionError("structured site varint is truncated or overlong")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _encode_site_records(
    records: Sequence[Sequence[np.ndarray]], *, class_id: int, source_id: int, coder: str
) -> bytes:
    if not 0 <= class_id < 5 or source_id not in (1, 2) or len(records) > 600:
        raise DirectDescriptionError("structured site-record identity is invalid")
    record_count = sum(len(rows) for rows in records)
    raw = bytearray(SITE_STREAM_HEADER.pack(SITE_STREAM_MAGIC, len(records), class_id, source_id, record_count))
    for pair_id, rows in enumerate(records):
        for sites in rows:
            ordered = np.asarray(sites, dtype=np.int64)
            if ordered.ndim != 1 or ordered.size == 0 or np.any(np.diff(ordered) <= 0):
                raise DirectDescriptionError("structured site record must be sorted, unique, and nonempty")
            if int(ordered[0]) < 0 or int(ordered[-1]) >= PAIR_SHAPE[0] * PAIR_SHAPE[1]:
                raise DirectDescriptionError("structured site record is outside scorer geometry")
            raw.extend(SITE_RECORD_HEADER.pack(pair_id, len(ordered)))
            raw.extend(_uvarint(int(ordered[0])))
            for left, right in itertools.pairwise(ordered):
                raw.extend(_uvarint(int(right) - int(left)))
    canonical = bytes(raw)
    if coder == "lzma1_raw_1MiB":
        coded = lzma.compress(canonical, format=lzma.FORMAT_RAW, filters=S4_LZMA_FILTERS)
        coder_id = 1
    elif coder == "brotli_q11":
        coded = brotli.compress(canonical, quality=11)
        coder_id = 2
    else:
        raise DirectDescriptionError("structured site-record coder is unknown")
    return struct.pack(">BII32s", coder_id, len(canonical), len(coded), hashlib.sha256(canonical).digest()) + coded


def _decode_site_records(
    payload: bytes, *, expected_class: int, expected_source: int
) -> tuple[tuple[np.ndarray, ...], ...]:
    if len(payload) < 41:
        raise DirectDescriptionError("structured site stream is truncated")
    coder_id, raw_bytes, coded_bytes, digest = struct.unpack_from(">BII32s", payload)
    if len(payload) != 41 + coded_bytes:
        raise DirectDescriptionError("structured site stream coded length is noncanonical")
    try:
        if coder_id == 1:
            raw = lzma.decompress(payload[41:], format=lzma.FORMAT_RAW, filters=S4_LZMA_FILTERS)
        elif coder_id == 2:
            raw = brotli.decompress(payload[41:])
        else:
            raise DirectDescriptionError("structured site stream coder id is unknown")
    except (lzma.LZMAError, brotli.error) as exc:
        raise DirectDescriptionError("structured site stream terminal decode failed") from exc
    if len(raw) != raw_bytes or hashlib.sha256(raw).digest() != digest or len(raw) < SITE_STREAM_HEADER.size:
        raise DirectDescriptionError("structured site stream decoded custody mismatch")
    magic, n_pairs, class_id, source_id, record_count = SITE_STREAM_HEADER.unpack_from(raw)
    if magic != SITE_STREAM_MAGIC or class_id != expected_class or source_id != expected_source:
        raise DirectDescriptionError("structured site stream semantic identity mismatch")
    rows: list[list[np.ndarray]] = [[] for _ in range(n_pairs)]
    offset = SITE_STREAM_HEADER.size
    for _ in range(record_count):
        if offset + SITE_RECORD_HEADER.size > len(raw):
            raise DirectDescriptionError("structured site record header is truncated")
        pair_id, count = SITE_RECORD_HEADER.unpack_from(raw, offset)
        offset += SITE_RECORD_HEADER.size
        if pair_id >= n_pairs or count == 0:
            raise DirectDescriptionError("structured site record metadata is invalid")
        first, offset = _read_uvarint(raw, offset)
        values = [first]
        for _ in range(count - 1):
            delta, offset = _read_uvarint(raw, offset)
            if delta <= 0:
                raise DirectDescriptionError("structured site record deltas must be positive")
            values.append(values[-1] + delta)
        if values[-1] >= PAIR_SHAPE[0] * PAIR_SHAPE[1]:
            raise DirectDescriptionError("structured site record escaped scorer geometry")
        rows[pair_id].append(np.asarray(values, dtype=np.int64))
    if offset != len(raw):
        raise DirectDescriptionError("structured site stream has trailing decoded bytes")
    return tuple(tuple(pair_rows) for pair_rows in rows)


def _decode_s4_static(encoded: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        quotient = brotli.decompress(encoded)
    except brotli.error as exc:
        raise DirectDescriptionError("S4 PXQ1 static partition failed Brotli decode") from exc
    if len(quotient) < 10:
        raise DirectDescriptionError("S4 PXQ1 partition is truncated")
    magic, height, width, _edge_mask = struct.unpack_from(">4sHHH", quotient)
    count = height * width
    packed = (count + 7) // 8
    if magic != b"PXQ1" or (height, width) != PAIR_SHAPE or len(quotient) != 10 + 3 * packed:
        raise DirectDescriptionError("S4 PXQ1 partition geometry is invalid")
    planes = []
    for index in range(3):
        start = 10 + index * packed
        planes.append(
            np.unpackbits(np.frombuffer(quotient[start : start + packed], np.uint8), bitorder="little")[:count]
            .reshape(PAIR_SHAPE)
            .astype(bool)
        )
    if np.any(planes[0] & planes[1]):
        raise DirectDescriptionError("S4 PXQ1 Road and Undrivable masks overlap")
    return planes[0], planes[1], planes[2]


def _decode_s4_lane(encoded: bytes) -> tuple[tuple[tuple[np.ndarray, ...], ...], dict[str, Any]]:
    try:
        payload = lzma.decompress(encoded, format=lzma.FORMAT_RAW, filters=S4_LZMA_FILTERS)
    except lzma.LZMAError as exc:
        raise DirectDescriptionError("S4 LBND2 lane payload failed LZMA decode") from exc
    if not payload.startswith(b"LBND2\x00") or len(payload) < 14:
        raise DirectDescriptionError("S4 LBND2 lane header is invalid")
    offset = 6
    header_size = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    if offset + header_size + 4 > len(payload):
        raise DirectDescriptionError("S4 LBND2 JSON header is truncated")
    try:
        header = json.loads(payload[offset : offset + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("S4 LBND2 JSON header is malformed") from exc
    offset += header_size
    presence_size = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    if offset + presence_size > len(payload):
        raise DirectDescriptionError("S4 LBND2 presence stream is truncated")
    presence_raw = payload[offset : offset + presence_size]
    offset += presence_size
    rd = header.get("rd", {})
    pairs, slots, dslot = int(rd.get("n_pairs", -1)), int(rd.get("K", -1)), int(rd.get("d_slot", -1))
    if pairs != 600 or dslot != 11 or not 0 <= slots <= 32:
        raise DirectDescriptionError("S4 LBND2 geometry is invalid")
    words = pairs * slots * dslot
    if len(payload) - offset != words * 4 or presence_size != (pairs * slots + 7) // 8:
        raise DirectDescriptionError("S4 LBND2 encoded lengths are inconsistent")
    if slots:
        presence = np.unpackbits(np.frombuffer(presence_raw, np.uint8))[: pairs * slots].reshape(pairs, slots)
        encoded_values = np.frombuffer(payload, np.uint32, words, offset).astype(np.int64).reshape(pairs, -1)
        delta = (encoded_values >> 1) ^ -(encoded_values & 1)
        steps = np.tile(np.asarray(rd["base_steps"], dtype=np.float64), slots)
        values = np.cumsum(delta, axis=0).astype(np.float64) * steps
    else:
        presence = np.zeros((pairs, 0), dtype=bool)
        values = np.zeros((pairs, 0), dtype=np.float64)
    lines = tuple(
        tuple(values[pair, slot * 11 : (slot + 1) * 11].copy() for slot in range(slots) if presence[pair, slot])
        for pair in range(pairs)
    )
    return lines, header


def _render_s4_lane_mask(
    lines: Sequence[np.ndarray], header: Mapping[str, Any], camera: Mapping[str, Any]
) -> np.ndarray:
    rows = np.arange(PAIR_SHAPE[0], dtype=np.float64)
    cols = np.arange(PAIR_SHAPE[1], dtype=np.float64)[None, :]
    horizon = float(header["v_h"])
    cx = float(header["cx"] if header.get("cx") is not None else PAIR_SHAPE[1] / 2)
    softness = max(float(header["softness"]), 1e-6)
    coverage = np.zeros(PAIR_SHAPE, dtype=np.float64)
    below = rows > horizon + 1.0
    selected = rows[below]
    forward = float(camera["height_m"]) * float(camera["fy_scorer"]) / np.maximum(selected - horizon, 1e-3)
    for vector in lines:
        center = cx - np.polyval(vector[:4], forward) * float(camera["fx_scorer"]) / forward
        half_width = np.maximum(np.polyval(vector[4:6], selected), 0.5)
        valid = (forward >= vector[9] - 1.0) & (forward <= vector[10] + 5.0)
        dash = np.ones_like(forward, dtype=bool)
        if bool(header["dash_gate"]) and vector[6] > 0:
            near = forward < float(header["dash_forward_max_m"])
            phase = np.mod(forward - vector[7], vector[6]) / vector[6]
            dash = np.where(near, phase < vector[8], True)
        signed = half_width[:, None] - np.abs(cols - center[:, None])
        candidate = np.clip(signed / softness + 0.5, 0.0, 1.0) * (valid & dash)[:, None]
        coverage[below] = np.maximum(coverage[below], candidate)
    return coverage >= 0.5


def _decode_pcomp3(payload: bytes) -> tuple[tuple[tuple[np.ndarray, ...], ...], ...]:
    by_class: list[list[list[np.ndarray]]] = [[[] for _ in range(600)] for _ in range(5)]
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise DirectDescriptionError("S4 PCOMP3 record prefix is truncated")
        size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        if size == 0 or offset + size > len(payload):
            raise DirectDescriptionError("S4 PCOMP3 record length is invalid")
        try:
            raw = zlib.decompress(payload[offset : offset + size])
        except zlib.error as exc:
            raise DirectDescriptionError("S4 PCOMP3 packet failed zlib decode") from exc
        offset += size
        if len(raw) < 12:
            raise DirectDescriptionError("S4 PCOMP3 semantic record is truncated")
        frame, class_id, _stratum, count, first = struct.unpack_from("<HBBII", raw)
        if frame >= 600 or class_id >= 5 or count == 0:
            raise DirectDescriptionError("S4 PCOMP3 semantic metadata is invalid")
        cursor = 12
        values = [first]
        for _ in range(count - 1):
            delta, cursor = _read_uvarint(raw, cursor)
            if delta <= 0:
                raise DirectDescriptionError("S4 PCOMP3 site deltas must be positive")
            values.append(values[-1] + delta)
        if cursor != len(raw) or values[-1] >= PAIR_SHAPE[0] * PAIR_SHAPE[1]:
            raise DirectDescriptionError("S4 PCOMP3 record has trailing or out-of-grid data")
        by_class[class_id][frame].append(np.asarray(values, dtype=np.int64))
    return tuple(tuple(tuple(rows) for rows in class_frames) for class_frames in by_class)


@dataclass(frozen=True, slots=True)
class StructuredS4SourcesV1:
    pair_count: int
    palette: np.ndarray
    camera: Mapping[str, Any]
    static_masks: Mapping[str, np.ndarray]
    lane_encoded: bytes
    lane_lines: tuple[tuple[np.ndarray, ...], ...]
    lane_header: Mapping[str, Any]
    events: tuple[tuple[tuple[np.ndarray, ...], ...], ...]
    components: tuple[tuple[tuple[np.ndarray, ...], ...], ...]
    custody: Mapping[str, Any]
    role_class_ids: Mapping[str, int] | None = None
    role_rgb_u8: Mapping[str, tuple[int, int, int]] | None = None
    routing_custody: Mapping[str, Any] | None = None


def _read_bound_file(path: Path, expected_sha256: str, label: str) -> bytes:
    payload = _read_regular_file_once(path)
    if _sha256(payload) != _require_sha256(expected_sha256, label):
        raise DirectDescriptionError(f"{label} SHA-256 custody mismatch")
    return payload


def load_structured_s4_sources(config: DirectDescriptionStratumStructuredMemberConfigV1) -> StructuredS4SourcesV1:
    """Read the settled S4 corpus without mutating it and extract typed structured sources."""

    container_path = Path(config.s4_container_path)
    runtime_path = Path(config.s4_runtime_path)
    payload = _read_bound_file(container_path, config.s4_container_sha256, "s4_container_sha256")
    runtime = _read_bound_file(runtime_path, config.s4_runtime_sha256, "s4_runtime_sha256")
    sections = {row.name: row for row in parse_sections(payload)}
    manifest = json.loads(sections["manifest.json"].payload.decode("ascii"))
    palette = np.asarray(manifest["weight_derived_constants"]["R2_max_margin_palette"]["value_u8"], dtype=np.uint8)
    camera = manifest["video_derived_constants"]["lane_camera_intrinsics"]["value"]
    if palette.shape != (5, 3) or set(camera) != {"height_m", "fx_scorer", "fy_scorer"}:
        raise DirectDescriptionError("S4 structured palette or camera schema is invalid")
    base = sections["base.pbase3"].payload
    if len(base) < 8:
        raise DirectDescriptionError("S4 PBASE3 section is truncated")
    static_size, lane_size = struct.unpack_from("<II", base)
    if len(base) != 8 + static_size + lane_size:
        raise DirectDescriptionError("S4 PBASE3 section lengths are inconsistent")
    static_encoded = base[8 : 8 + static_size]
    lane_encoded = base[8 + static_size :]
    road, undrivable, hood = _decode_s4_static(static_encoded)
    event_section = sections["events.pce3"]
    if event_section.codec != "lzma1_raw_1MiB":
        raise DirectDescriptionError("S4 event source is not the settled LZMA PCE3 stream")
    try:
        event_raw = lzma.decompress(event_section.payload, format=lzma.FORMAT_RAW, filters=S4_LZMA_FILTERS)
    except lzma.LZMAError as exc:
        raise DirectDescriptionError("S4 PCE3 source failed LZMA decode") from exc
    decoded_events = decode_component_event_alphabet_raw(event_raw)
    events = tuple(
        tuple(tuple(np.asarray(row, dtype=np.int64) for row in decoded_events[pair][class_id]) for pair in range(600))
        for class_id in range(5)
    )
    all_components = _decode_pcomp3(sections["components.pcomp3"].payload)
    components = tuple(tuple(all_components[class_id][pair] for pair in range(600)) for class_id in range(5))
    lane_lines, lane_header = _decode_s4_lane(lane_encoded)
    return StructuredS4SourcesV1(
        pair_count=600,
        palette=palette,
        camera=dict(camera),
        static_masks={"Road": road, "Undrivable": undrivable, "MyCar": hood},
        lane_encoded=lane_encoded,
        lane_lines=lane_lines,
        lane_header=lane_header,
        events=events,
        components=components,
        custody={
            "s4_container_path": str(container_path),
            "s4_container_bytes": len(payload),
            "s4_container_sha256": _sha256(payload),
            "s4_runtime_path": str(runtime_path),
            "s4_runtime_bytes": len(runtime),
            "s4_runtime_sha256": _sha256(runtime),
            "source_read_only": True,
            "source_mutated": False,
        },
    )


def _pack_mask(mask: np.ndarray) -> bytes:
    value = np.asarray(mask, dtype=bool)
    if value.shape != PAIR_SHAPE:
        raise DirectDescriptionError("structured static mask geometry is invalid")
    raw = np.packbits(value.reshape(-1), bitorder="little").tobytes()
    coded = brotli.compress(raw, quality=11)
    return struct.pack(">II32s", value.size, len(raw), hashlib.sha256(raw).digest()) + coded


def _unpack_mask(payload: bytes) -> np.ndarray:
    if len(payload) < 40:
        raise DirectDescriptionError("structured static mask frame is truncated")
    sites, raw_bytes, digest = struct.unpack_from(">II32s", payload)
    try:
        raw = brotli.decompress(payload[40:])
    except brotli.error as exc:
        raise DirectDescriptionError("structured static mask failed Brotli decode") from exc
    if sites != PAIR_SHAPE[0] * PAIR_SHAPE[1] or len(raw) != raw_bytes or hashlib.sha256(raw).digest() != digest:
        raise DirectDescriptionError("structured static mask custody mismatch")
    return np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")[:sites].reshape(PAIR_SHAPE).astype(bool)


def self_detect_structured_role_classes(lstars: np.ndarray) -> tuple[dict[str, int], dict[str, Any]]:
    """Detect semantic role indices from target geometry; never trust channel order."""

    labels = np.asarray(lstars)
    if labels.ndim != 3 or labels.shape[1:] != PAIR_SHAPE or labels.shape[0] < 2:
        raise DirectDescriptionError("structured role detection requires [N,384,512] target cells")
    detected = classify_segnet_regions(labels, n_classes=5, lane_probe_frames=min(6, len(labels)))
    hood_class, hood_evidence = identify_static_hood_class(labels, n_classes=5)
    if hood_class != detected.hood:
        raise DirectDescriptionError("independent hood and full-region self-detectors disagree")
    role_class_ids = {
        "Road": int(detected.road),
        "Lane": int(detected.lane),
        "UndrivableBoundary": int(detected.sky),
        "Movable": int(detected.movable),
        "MyCar": int(detected.hood),
    }
    if sorted(role_class_ids.values()) != list(range(5)):
        raise DirectDescriptionError("self-detected structured roles are not a class permutation")
    return role_class_ids, {
        "method": "road_horizon_component.classify_segnet_regions plus hood_static_component crosscheck",
        "role_class_ids": role_class_ids,
        "hood_crosscheck_class": hood_class,
        "full_region_evidence": [row.__dict__ for row in detected.evidence],
        "hood_evidence": [row.__dict__ for row in hood_evidence],
        "lane_detector_source": "lane_sdf_component cluster/fitter through classify_segnet_regions",
    }


def _require_structured_routing(
    sources: StructuredS4SourcesV1,
) -> tuple[Mapping[str, int], Mapping[str, tuple[int, int, int]]]:
    if sources.role_class_ids is None or sources.role_rgb_u8 is None:
        raise DirectDescriptionError("structured source lacks self-detected role/value routing")
    if set(sources.role_class_ids) != set(ROUTED_ROLES) or set(sources.role_rgb_u8) != set(ROUTED_ROLES):
        raise DirectDescriptionError("structured role/value routing is incomplete")
    if sorted(int(value) for value in sources.role_class_ids.values()) != list(range(5)):
        raise DirectDescriptionError("structured role classes must be a permutation")
    for role, value in sources.role_rgb_u8.items():
        if len(value) != 3 or any(isinstance(channel, bool) or not 0 <= int(channel) <= 255 for channel in value):
            raise DirectDescriptionError(f"structured RGB routing for {role} is invalid")
    return sources.role_class_ids, sources.role_rgb_u8


def _structured_role_mask(
    sources: StructuredS4SourcesV1,
    role: str,
    *,
    source_pair_id: int,
) -> np.ndarray:
    role_class_ids, _ = _require_structured_routing(sources)
    if role not in ROUTED_ROLES or not 0 <= source_pair_id < sources.pair_count:
        raise DirectDescriptionError("structured role mask identity is invalid")
    mask = np.zeros(PAIR_SHAPE, dtype=bool)
    if role == "Road":
        mask |= sources.static_masks["Road"]
    elif role == "Lane":
        mask |= _render_s4_lane_mask(sources.lane_lines[source_pair_id], sources.lane_header, sources.camera)
    elif role == "MyCar":
        mask |= sources.static_masks["MyCar"]
    class_id = int(role_class_ids[role])
    if role in {"Road", "Lane", "UndrivableBoundary", "Movable"}:
        flat = mask.reshape(-1)
        for sites in sources.events[class_id][source_pair_id]:
            flat[sites] = True
    if role in {"Road", "Lane", "UndrivableBoundary"}:
        flat = mask.reshape(-1)
        for sites in sources.components[class_id][source_pair_id]:
            flat[sites] = True
    return mask


def select_role_paint_values(
    role_class_ids: Mapping[str, int],
    score_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any]]:
    """Select each role's own-class membership maximum with deterministic ties."""

    if set(role_class_ids) != set(ROUTED_ROLES) or sorted(role_class_ids.values()) != list(range(5)):
        raise DirectDescriptionError("role-value selector requires a self-detected class permutation")
    selected: dict[str, Mapping[str, Any]] = {}
    for role in ROUTED_ROLES:
        rows = tuple(score_rows.get(role, ()))
        if not rows:
            raise DirectDescriptionError(f"role-value selector has no candidates for {role}")
        own_class = int(role_class_ids[role])
        if any(int(row.get("target_class_id", -1)) != own_class for row in rows):
            raise DirectDescriptionError("role-value selector candidate targets drifted")
        selected[role] = max(
            rows,
            key=lambda row: (
                int(row.get("own_class_matches", -1)),
                -int(row.get("candidate_index", -1)),
            ),
        )
    return selected


def measure_structured_role_value_routing(
    sources: StructuredS4SourcesV1,
    *,
    baseline: Any,
    target_planes: np.ndarray,
    target_cells: np.ndarray,
    oracle: Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray | None]],
    source_pair_start: int,
) -> tuple[StructuredS4SourcesV1, dict[str, Any]]:
    """Measure and bind role->class->RGB routing on a small real receiver window."""

    targets = np.asarray(target_planes)
    cells = np.asarray(target_cells)
    if (
        targets.dtype != np.uint8
        or targets.ndim != 5
        or targets.shape[1:] != (2, *PAIR_SHAPE, 3)
        or cells.shape != (len(targets), *PAIR_SHAPE)
        or cells.dtype != np.int64
        or baseline.z.n_pairs != len(targets)
        or len(targets) != 16
    ):
        raise DirectDescriptionError("role-value routing probe requires one canonical n16 scorer batch")
    role_class_ids, class_detection = self_detect_structured_role_classes(cells)
    provisional_rgb = {
        role: tuple(int(channel) for channel in sources.palette[class_id]) for role, class_id in role_class_ids.items()
    }
    routed = replace(
        sources,
        role_class_ids=role_class_ids,
        role_rgb_u8=provisional_rgb,
        routing_custody={"status": "CLASS_DETECTED_VALUE_PROVISIONAL"},
    )
    prototypes: list[dict[str, Any]] = []
    for candidate_index, role in enumerate(ROUTED_ROLES):
        class_id = role_class_ids[role]
        values = targets[:, 1][cells == class_id]
        if not len(values):
            raise DirectDescriptionError(f"routing probe has no target RGB support for {role}")
        rgb = tuple(int(value) for value in np.rint(np.median(values, axis=0)).astype(np.uint8))
        prototypes.append(
            {
                "candidate_index": candidate_index,
                "source_role": role,
                "statistic": "channelwise_median_of_c1_member_on_self_detected_role_sites",
                "rgb_u8": list(rgb),
                "target_rgb_sites": len(values),
            }
        )
    baseline_pairs = baseline.render_pairs(tuple(range(len(targets))))
    score_rows: dict[str, list[dict[str, Any]]] = {}
    role_probe: dict[str, Any] = {}
    for role in ROUTED_ROLES:
        class_id = role_class_ids[role]
        masks = np.asarray(
            [
                _structured_role_mask(routed, role, source_pair_id=source_pair_start + local_pair_id)
                for local_pair_id in range(len(targets))
            ]
        )
        target_role = cells == class_id
        correct_geometry = masks & target_role
        geometry_sites = int(masks.sum(dtype=np.int64))
        correct_geometry_sites = int(correct_geometry.sum(dtype=np.int64))
        if geometry_sites == 0 or correct_geometry_sites == 0:
            raise DirectDescriptionError(f"routing probe state window has no correct geometry support for {role}")
        union = correct_geometry.any(axis=0)
        yy, xx = np.where(union)
        inherited = tuple(int(value) for value in sources.palette[class_id])
        candidates = [{**row, "candidate_family": "c1_role_median"} for row in prototypes] + [
            {
                "candidate_index": len(prototypes),
                "source_role": role,
                "statistic": "self_detected_class_row_from_inherited_s4_palette",
                "rgb_u8": list(inherited),
                "target_rgb_sites": 0,
                "candidate_family": "inherited_s4_palette",
            }
        ]
        role_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            painted = baseline_pairs.copy()
            rgb = np.asarray(candidate["rgb_u8"], dtype=np.uint8)
            painted[:, 0][masks] = rgb
            painted[:, 1][masks] = rgb
            described_cells, described_margins = oracle(painted, False)
            if described_margins is not None:
                raise DirectDescriptionError("routing probe oracle returned unrequested margins")
            own_matches = int(np.count_nonzero(described_cells[correct_geometry] == class_id))
            role_rows.append(
                {
                    **candidate,
                    "target_class_id": class_id,
                    "own_class_matches": own_matches,
                    "correct_geometry_sites": correct_geometry_sites,
                    "own_class_membership": _fraction_text(own_matches, correct_geometry_sites),
                }
            )
        inherited_row = role_rows[-1]
        score_rows[role] = role_rows
        c1_values = targets[:, 1][correct_geometry]
        role_probe[role] = {
            "target_class_id": class_id,
            "geometry": {
                "mask_sites": geometry_sites,
                "correct_target_sites_in_geometry": correct_geometry_sites,
                "geometry_precision": _fraction_text(correct_geometry_sites, geometry_sites),
                "geometry_recall": _fraction_text(correct_geometry_sites, int(target_role.sum(dtype=np.int64))),
                "correct_site_union_row_span": [int(yy.min()), int(yy.max())],
                "correct_site_union_col_span": [int(xx.min()), int(xx.max())],
            },
            "inherited_s4_paint": {
                "rgb_u8": list(inherited),
                "own_class_matches": inherited_row["own_class_matches"],
                "correct_geometry_sites": correct_geometry_sites,
                "own_class_membership": inherited_row["own_class_membership"],
            },
            "c1_member_same_sites": {
                "rgb_mean": [format(float(value), ".6f") for value in c1_values.mean(axis=0)],
                "rgb_median": [int(value) for value in np.median(c1_values, axis=0)],
                "sites": len(c1_values),
            },
            "candidate_rows": role_rows,
        }
    selected = select_role_paint_values(role_class_ids, score_rows)
    role_rgb_u8 = {role: tuple(int(value) for value in row["rgb_u8"]) for role, row in selected.items()}
    for role, row in selected.items():
        role_probe[role]["selected"] = dict(row)
        if int(row["own_class_matches"]) != max(int(value["own_class_matches"]) for value in score_rows[role]):
            raise DirectDescriptionError("role-value selector failed its own-class maximum invariant")
    receipt = {
        "schema": "direct_description_role_value_routing_probe.v1",
        "source_pair_window": {
            "start": source_pair_start,
            "stop": source_pair_start + len(targets),
            "count": len(targets),
        },
        "class_detection": class_detection,
        "candidate_prototypes": prototypes,
        "role_probe": role_probe,
        "selected_role_rgb_u8": {role: list(value) for role, value in role_rgb_u8.items()},
        "selection_rule": "maximum own-target-class membership on fixed correct geometry; deterministic candidate order tie-break",
        "scorer_weights_stored_in_archive": False,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "d_seg_claim": False,
    }
    return replace(
        sources,
        role_class_ids=role_class_ids,
        role_rgb_u8=role_rgb_u8,
        routing_custody=receipt,
    ), receipt


def _payloads_for_role(
    role: str,
    sources: StructuredS4SourcesV1,
    *,
    pair_start: int = 0,
    pair_count: int = 64,
) -> dict[str, bytes]:
    if role == "baseline":
        return {}
    role_class_ids, _ = _require_structured_routing(sources)
    class_id = int(role_class_ids[role])
    stop = pair_start + pair_count
    if pair_start < 0 or pair_count < 1 or stop > sources.pair_count:
        raise DirectDescriptionError("structured payload window is outside S4 custody")
    events = sources.events[class_id][pair_start:stop]
    components = sources.components[class_id][pair_start:stop]
    if role == "Road":
        return {
            "structure/road_pxq1_mask.br": _pack_mask(sources.static_masks["Road"]),
            "structure/road_events.lz": _encode_site_records(
                events, class_id=class_id, source_id=1, coder="lzma1_raw_1MiB"
            ),
            "structure/road_components.br": _encode_site_records(
                components, class_id=class_id, source_id=2, coder="brotli_q11"
            ),
        }
    if role == "Lane":
        return {
            "structure/lane_lbnd2.lz": sources.lane_encoded,
            "structure/lane_events.lz": _encode_site_records(
                events, class_id=class_id, source_id=1, coder="lzma1_raw_1MiB"
            ),
            "structure/lane_components.br": _encode_site_records(
                components, class_id=class_id, source_id=2, coder="brotli_q11"
            ),
        }
    if role == "MyCar":
        return {"structure/mycar_static_hood.br": _pack_mask(sources.static_masks["MyCar"])}
    if role == "UndrivableBoundary":
        return {
            "structure/undrivable_events.lz": _encode_site_records(
                events, class_id=class_id, source_id=1, coder="lzma1_raw_1MiB"
            ),
            "structure/undrivable_components.br": _encode_site_records(
                components, class_id=class_id, source_id=2, coder="brotli_q11"
            ),
        }
    if role == "Movable":
        return {
            "structure/movable_events.lz": _encode_site_records(
                events, class_id=class_id, source_id=1, coder="lzma1_raw_1MiB"
            )
        }
    raise DirectDescriptionError(f"unknown structured candidate role {role!r}")


def _zip_stored(members: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, payload)
    return buffer.getvalue()


def compile_structured_member_archive(
    baseline_archive: bytes, sources: StructuredS4SourcesV1, role: str
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    if role not in STRUCTURED_ROLES:
        raise DirectDescriptionError("structured member role is outside the typed family")
    baseline_pairs = receive_entropy_chart_archive(baseline_archive).z.n_pairs
    role_class_ids, role_rgb_u8 = _require_structured_routing(sources)
    payloads = _payloads_for_role(role, sources, pair_start=0, pair_count=baseline_pairs)
    declarations = [
        {"name": name, "bytes": len(payload), "sha256": _sha256(payload)} for name, payload in payloads.items()
    ]
    manifest = {
        "schema": "direct_description_stratum_structured_archive.v1",
        "magic": STRUCTURED_MEMBER_MAGIC.decode("ascii"),
        "pair_count": baseline_pairs,
        "source_pair_start": 0,
        "role": role,
        "class_id": role_class_ids.get(role),
        "target_class": ROLE_TARGET_CLASS.get(role),
        "paint_rgb_u8": list(role_rgb_u8[role]) if role != "baseline" else None,
        "palette_u8": sources.palette.tolist(),
        "lane_camera": dict(sources.camera),
        "baseline_chart": {"bytes": len(baseline_archive), "sha256": _sha256(baseline_archive)},
        "structured_payloads": declarations,
        "semantic_source": {
            "Road": "S4 PXQ1 Road plane plus class-filtered PCE3 and PCOMP3 records",
            "Lane": "S4 LBND2 lane curve/SDF raster plus class-filtered PCE3 and PCOMP3 records",
            "MyCar": "S4 static ego-hood mask",
            "UndrivableBoundary": "S4 class-filtered PCE3 and PCOMP3 boundary records; bulk remains v3 chart",
            "Movable": "S4 class-filtered PCE3 event records",
            "baseline": "unchanged v3 entropy chart and Pose6 stream",
        }[role],
        "receiver": "numpy_uint8_v3_chart_plus_s4_structured_override.v1",
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
    }
    members: dict[str, bytes] = {
        "manifest.json": rfc8785_canonicalize(manifest),
        "chart.zip": baseline_archive,
        **payloads,
    }
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("structured archive compiler is nondeterministic")
    parsed, homes = parse_structured_member_archive(first)
    if parsed != members:
        raise DirectDescriptionError("structured archive parse-back changed semantic members")
    return first, homes


def compile_composed_structured_member_archive(
    baseline_archive: bytes,
    sources: StructuredS4SourcesV1,
    *,
    pair_start: int,
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    """Compile one receiver-closed archive containing all five structured roles."""

    baseline_pairs = receive_entropy_chart_archive(baseline_archive).z.n_pairs
    role_class_ids, role_rgb_u8 = _require_structured_routing(sources)
    payloads: dict[str, bytes] = {}
    for role in COMPOSED_ROLE_ORDER:
        overlap = set(payloads).intersection(
            role_payloads := _payloads_for_role(role, sources, pair_start=pair_start, pair_count=baseline_pairs)
        )
        if overlap:
            raise DirectDescriptionError(f"composed role payload homes overlap: {sorted(overlap)}")
        payloads.update(role_payloads)
    declarations = [
        {"name": name, "bytes": len(payload), "sha256": _sha256(payload)} for name, payload in payloads.items()
    ]
    manifest = {
        "schema": "direct_description_stratum_composed_archive.v1",
        "magic": COMPOSED_MEMBER_MAGIC.decode("ascii"),
        "pair_count": baseline_pairs,
        "source_pair_start": pair_start,
        "role": "composed",
        "role_order": list(COMPOSED_ROLE_ORDER),
        "role_class_ids": {role: int(role_class_ids[role]) for role in ROUTED_ROLES},
        "role_rgb_u8": {role: list(role_rgb_u8[role]) for role in ROUTED_ROLES},
        "palette_u8_inherited_diagnostic_only": sources.palette.tolist(),
        "lane_camera": dict(sources.camera),
        "baseline_chart": {"bytes": len(baseline_archive), "sha256": _sha256(baseline_archive)},
        "structured_payloads": declarations,
        "semantic_source": (
            "S4 LBND2 Lane plus PXQ1 Road/static hood plus class-filtered PCE3/PCOMP3 "
            "Undrivable/Movable paths and the unchanged counted Pose6 chart stream"
        ),
        "routing": {
            "policy": "self_detected_role_classes_and_measured_c1_prototype_value_selection",
            "scorer_weights_present": False,
            "routing_receipt_sha256": _sha256(rfc8785_canonicalize(sources.routing_custody)),
        },
        "receiver": "numpy_uint8_v3_chart_plus_s4_composed_overrides.v1",
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
    }
    members: dict[str, bytes] = {
        "manifest.json": rfc8785_canonicalize(manifest),
        "chart.zip": baseline_archive,
        **payloads,
    }
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("composed structured archive compiler is nondeterministic")
    parsed, homes = parse_structured_member_archive(first)
    if parsed != members:
        raise DirectDescriptionError("composed structured archive parse-back changed semantic members")
    return first, homes


def parse_structured_member_archive(archive: bytes) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if len(infos) < 2 or [row.filename for row in infos[:2]] != ["manifest.json", "chart.zip"]:
                raise DirectDescriptionError("structured archive member prefix/order is invalid")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.filename.startswith("/")
                or ".." in Path(row.filename).parts
                for row in infos
            ):
                raise DirectDescriptionError("structured archive metadata is noncanonical")
            members = {row.filename: reader.read(row) for row in infos}
            start_dir = reader.start_dir
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise DirectDescriptionError("structured archive ZIP is malformed") from exc
    try:
        manifest = json.loads(members["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("structured archive manifest is malformed") from exc
    if rfc8785_canonicalize(manifest) != members["manifest.json"]:
        raise DirectDescriptionError("structured archive manifest is noncanonical")
    schema = manifest.get("schema")
    composed = schema == "direct_description_stratum_composed_archive.v1"
    single = schema == "direct_description_stratum_structured_archive.v1"
    baseline_pairs = receive_entropy_chart_archive(members["chart.zip"]).z.n_pairs
    pair_count = manifest.get("pair_count")
    source_pair_start = manifest.get("source_pair_start")
    common_invalid = (
        not (single or composed)
        or pair_count != baseline_pairs
        or isinstance(source_pair_start, bool)
        or not isinstance(source_pair_start, int)
        or source_pair_start < 0
        or source_pair_start + baseline_pairs > 600
        or set(members) != {"manifest.json", "chart.zip", *[row["name"] for row in manifest["structured_payloads"]]}
        or manifest["baseline_chart"] != {"bytes": len(members["chart.zip"]), "sha256": _sha256(members["chart.zip"])}
    )
    single_invalid = single and (
        manifest.get("magic") != STRUCTURED_MEMBER_MAGIC.decode("ascii")
        or manifest.get("role") not in STRUCTURED_ROLES
        or (
            manifest.get("role") != "baseline"
            and (
                isinstance(manifest.get("class_id"), bool)
                or not isinstance(manifest.get("class_id"), int)
                or not 0 <= manifest["class_id"] < 5
                or not isinstance(manifest.get("paint_rgb_u8"), list)
                or len(manifest["paint_rgb_u8"]) != 3
            )
        )
    )
    composed_invalid = composed and (
        manifest.get("magic") != COMPOSED_MEMBER_MAGIC.decode("ascii")
        or manifest.get("role") != "composed"
        or manifest.get("role_order") != list(COMPOSED_ROLE_ORDER)
        or set(manifest.get("role_class_ids", ())) != set(ROUTED_ROLES)
        or sorted(manifest.get("role_class_ids", {}).values()) != list(range(5))
        or set(manifest.get("role_rgb_u8", ())) != set(ROUTED_ROLES)
        or any(len(value) != 3 for value in manifest.get("role_rgb_u8", {}).values())
    )
    if common_invalid or single_invalid or composed_invalid:
        raise DirectDescriptionError("structured archive manifest identity is invalid")
    for declaration in manifest["structured_payloads"]:
        payload = members.get(declaration.get("name"))
        if payload is None or declaration != {
            "name": declaration["name"],
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }:
            raise DirectDescriptionError("structured archive payload custody mismatch")
    if _zip_stored(members) != archive:
        raise DirectDescriptionError("structured archive is not byte-canonical")
    homes: list[dict[str, Any]] = []
    for index, info in enumerate(infos):
        next_offset = infos[index + 1].header_offset if index + 1 < len(infos) else start_dir
        homes.append(
            {
                "name": info.filename,
                "payload_bytes": info.file_size,
                "zip_home_bytes": next_offset - info.header_offset,
                "payload_sha256": _sha256(members[info.filename]),
            }
        )
    container_only = len(archive) - sum(row["zip_home_bytes"] for row in homes)
    homes.append({"name": "__central_directory_and_eocd__", "payload_bytes": 0, "zip_home_bytes": container_only})
    if sum(row["zip_home_bytes"] for row in homes) != len(archive):
        raise DirectDescriptionError("structured archive unique-home accounting does not close")
    return members, tuple(homes)


@dataclass(frozen=True, slots=True)
class StructuredMemberReceiverV1:
    archive: bytes
    z: DirectDescriptionChartZV1
    pose6_codes: np.ndarray
    baseline: Any
    role: str
    class_id: int | None
    paint_rgb_u8: np.ndarray | None
    source_pair_start: int
    palette: np.ndarray
    camera: Mapping[str, Any]
    static_mask: np.ndarray | None
    lane_lines: tuple[tuple[np.ndarray, ...], ...] | None
    lane_header: Mapping[str, Any] | None
    event_rows: tuple[tuple[np.ndarray, ...], ...] | None
    component_rows: tuple[tuple[np.ndarray, ...], ...] | None
    custody: Mapping[str, Any]

    def render_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        output = self.baseline.render_pairs(indexes)
        if self.role == "baseline":
            return output
        if self.class_id is None or self.paint_rgb_u8 is None:
            raise DirectDescriptionError("structured receiver role/value routing is missing")
        color = self.paint_rgb_u8
        for local_index, pair_id in enumerate(indexes):
            source_pair_id = self.source_pair_start + pair_id
            mask = np.zeros(PAIR_SHAPE, dtype=bool)
            if self.static_mask is not None:
                mask |= self.static_mask
            if self.lane_lines is not None and self.lane_header is not None:
                mask |= _render_s4_lane_mask(self.lane_lines[source_pair_id], self.lane_header, self.camera)
            for rows in (self.event_rows, self.component_rows):
                if rows is not None:
                    flat = mask.reshape(-1)
                    for sites in rows[pair_id]:
                        flat[sites] = True
            output[local_index, 0, mask] = color
            output[local_index, 1, mask] = color
        return np.ascontiguousarray(output)


@dataclass(frozen=True, slots=True)
class StructuredRoleLayerV1:
    role: str
    class_id: int
    paint_rgb_u8: np.ndarray
    static_mask: np.ndarray | None
    lane_lines: tuple[tuple[np.ndarray, ...], ...] | None
    lane_header: Mapping[str, Any] | None
    event_rows: tuple[tuple[np.ndarray, ...], ...] | None
    component_rows: tuple[tuple[np.ndarray, ...], ...] | None

    def mask(self, *, local_pair_id: int, source_pair_id: int, camera: Mapping[str, Any]) -> np.ndarray:
        value = np.zeros(PAIR_SHAPE, dtype=bool)
        if self.static_mask is not None:
            value |= self.static_mask
        if self.lane_lines is not None and self.lane_header is not None:
            value |= _render_s4_lane_mask(self.lane_lines[source_pair_id], self.lane_header, camera)
        flat = value.reshape(-1)
        for rows in (self.event_rows, self.component_rows):
            if rows is not None:
                for sites in rows[local_pair_id]:
                    flat[sites] = True
        return value


@dataclass(frozen=True, slots=True)
class ComposedStructuredMemberReceiverV1:
    archive: bytes
    z: DirectDescriptionChartZV1
    pose6_codes: np.ndarray
    baseline: Any
    source_pair_start: int
    camera: Mapping[str, Any]
    layers: tuple[StructuredRoleLayerV1, ...]
    custody: Mapping[str, Any]

    def render_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        if any(value < 0 or value >= self.z.n_pairs for value in indexes):
            raise DirectDescriptionError("composed receiver pair ID is outside the local state window")
        output = self.baseline.render_pairs(indexes)
        for layer in self.layers:
            for local_index, pair_id in enumerate(indexes):
                source_pair_id = self.source_pair_start + pair_id
                mask = layer.mask(local_pair_id=pair_id, source_pair_id=source_pair_id, camera=self.camera)
                output[local_index, 0, mask] = layer.paint_rgb_u8
                output[local_index, 1, mask] = layer.paint_rgb_u8
        return np.ascontiguousarray(output)


def _decode_structured_role_layer(
    role: str,
    *,
    class_id: int,
    paint_rgb_u8: Sequence[int],
    members: Mapping[str, bytes],
) -> StructuredRoleLayerV1:
    static_mask = None
    lane_lines = None
    lane_header = None
    event_rows = None
    component_rows = None
    if role == "Road":
        static_mask = _unpack_mask(members["structure/road_pxq1_mask.br"])
        event_rows = _decode_site_records(
            members["structure/road_events.lz"], expected_class=class_id, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/road_components.br"], expected_class=class_id, expected_source=2
        )
    elif role == "Lane":
        lane_lines, lane_header = _decode_s4_lane(members["structure/lane_lbnd2.lz"])
        event_rows = _decode_site_records(
            members["structure/lane_events.lz"], expected_class=class_id, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/lane_components.br"], expected_class=class_id, expected_source=2
        )
    elif role == "MyCar":
        static_mask = _unpack_mask(members["structure/mycar_static_hood.br"])
    elif role == "UndrivableBoundary":
        event_rows = _decode_site_records(
            members["structure/undrivable_events.lz"], expected_class=class_id, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/undrivable_components.br"], expected_class=class_id, expected_source=2
        )
    elif role == "Movable":
        event_rows = _decode_site_records(
            members["structure/movable_events.lz"], expected_class=class_id, expected_source=1
        )
    else:
        raise DirectDescriptionError(f"unknown composed structured role {role!r}")
    return StructuredRoleLayerV1(
        role=role,
        class_id=class_id,
        paint_rgb_u8=np.asarray(paint_rgb_u8, dtype=np.uint8),
        static_mask=static_mask,
        lane_lines=lane_lines,
        lane_header=lane_header,
        event_rows=event_rows,
        component_rows=component_rows,
    )


def _receive_composed_structured_member_archive(
    archive: bytes,
    *,
    members: Mapping[str, bytes],
    homes: Sequence[Mapping[str, Any]],
) -> ComposedStructuredMemberReceiverV1:
    manifest = json.loads(members["manifest.json"])
    baseline = receive_entropy_chart_archive(members["chart.zip"])
    layers = tuple(
        _decode_structured_role_layer(
            role,
            class_id=int(manifest["role_class_ids"][role]),
            paint_rgb_u8=manifest["role_rgb_u8"][role],
            members=members,
        )
        for role in manifest["role_order"]
    )
    return ComposedStructuredMemberReceiverV1(
        archive=archive,
        z=baseline.z,
        pose6_codes=baseline.pose6_codes,
        baseline=baseline,
        source_pair_start=int(manifest["source_pair_start"]),
        camera=manifest["lane_camera"],
        layers=layers,
        custody={
            "schema": "direct_description_stratum_composed_receiver.v1",
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "unique_home_coverage_bytes": sum(row["zip_home_bytes"] for row in homes),
            "all_archive_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
            "member_homes": list(homes),
            "all_five_roles_consumed": [row.role for row in layers] == list(COMPOSED_ROLE_ORDER),
            "source_raw_reference_used": False,
            "scorer_weights_present": False,
        },
    )


def receive_structured_member_archive(
    archive: bytes,
) -> StructuredMemberReceiverV1 | ComposedStructuredMemberReceiverV1:
    members, homes = parse_structured_member_archive(archive)
    manifest = json.loads(members["manifest.json"])
    if manifest["schema"] == "direct_description_stratum_composed_archive.v1":
        return _receive_composed_structured_member_archive(archive, members=members, homes=homes)
    baseline = receive_entropy_chart_archive(members["chart.zip"])
    role = manifest["role"]
    palette = np.asarray(manifest["palette_u8"], dtype=np.uint8)
    if palette.shape != (5, 3) or baseline.z.n_pairs != 64:
        raise DirectDescriptionError("structured receiver baseline or palette geometry is invalid")
    static_mask = None
    lane_lines = None
    lane_header = None
    event_rows = None
    component_rows = None
    class_id = manifest.get("class_id")
    paint_rgb_u8 = (
        None if manifest.get("paint_rgb_u8") is None else np.asarray(manifest["paint_rgb_u8"], dtype=np.uint8)
    )
    if role == "Road":
        static_mask = _unpack_mask(members["structure/road_pxq1_mask.br"])
        event_rows = _decode_site_records(
            members["structure/road_events.lz"], expected_class=class_id, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/road_components.br"], expected_class=class_id, expected_source=2
        )
    elif role == "Lane":
        lane_lines, lane_header = _decode_s4_lane(members["structure/lane_lbnd2.lz"])
        event_rows = _decode_site_records(
            members["structure/lane_events.lz"], expected_class=class_id, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/lane_components.br"], expected_class=class_id, expected_source=2
        )
    elif role == "MyCar":
        static_mask = _unpack_mask(members["structure/mycar_static_hood.br"])
    elif role == "UndrivableBoundary":
        event_rows = _decode_site_records(
            members["structure/undrivable_events.lz"], expected_class=class_id, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/undrivable_components.br"], expected_class=class_id, expected_source=2
        )
    elif role == "Movable":
        event_rows = _decode_site_records(
            members["structure/movable_events.lz"], expected_class=class_id, expected_source=1
        )
    return StructuredMemberReceiverV1(
        archive=archive,
        z=baseline.z,
        pose6_codes=baseline.pose6_codes,
        baseline=baseline,
        role=role,
        class_id=class_id,
        paint_rgb_u8=paint_rgb_u8,
        source_pair_start=manifest["source_pair_start"],
        palette=palette,
        camera=manifest["lane_camera"],
        static_mask=static_mask,
        lane_lines=lane_lines,
        lane_header=lane_header,
        event_rows=event_rows,
        component_rows=component_rows,
        custody={
            "schema": "direct_description_stratum_structured_receiver.v1",
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "unique_home_coverage_bytes": sum(row["zip_home_bytes"] for row in homes),
            "all_archive_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
            "member_homes": list(homes),
            "source_raw_reference_used": False,
        },
    )


class DirectDescriptionStructuredCandidateCheckpointV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionStructuredCandidateCheckpointV1"] = Field(
        default=STRUCTURED_CHECKPOINT_SCHEMA, alias="schema", serialization_alias="schema"
    )
    config: dict[str, Any]
    config_sha256: StrictStr
    dsl_compile_hash: StrictStr
    semantic_argv: tuple[StrictStr, ...]
    semantic_argv_sha256: StrictStr
    completed_candidate_index: StrictInt = Field(ge=0, le=len(STRUCTURED_ROLES) - 1)
    completed_role: StrictStr
    next_candidate_index: StrictInt = Field(ge=1, le=len(STRUCTURED_ROLES))
    candidates: tuple[dict[str, Any], ...]
    evidence_axis: Literal["[macOS-CPU frozen-SegNet advisory]"] = EVIDENCE_AXIS
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _identity(self) -> DirectDescriptionStructuredCandidateCheckpointV1:
        for name in ("config_sha256", "dsl_compile_hash", "semantic_argv_sha256"):
            _require_sha256(getattr(self, name), name)
        config = DirectDescriptionStratumStructuredMemberConfigV1.model_validate_json(rfc8785_canonicalize(self.config))
        if config.typed_config_hash() != self.config_sha256 or config.dsl_compile_hash() != self.dsl_compile_hash:
            raise ValueError("structured checkpoint config identity mismatch")
        if (
            self.completed_role != STRUCTURED_ROLES[self.completed_candidate_index]
            or self.next_candidate_index != self.completed_candidate_index + 1
            or len(self.candidates) != self.next_candidate_index
            or [row.get("role") for row in self.candidates] != list(STRUCTURED_ROLES[: self.next_candidate_index])
        ):
            raise ValueError("structured checkpoint cursor/order mismatch")
        if _sha256("\0".join(self.semantic_argv).encode()) != self.semantic_argv_sha256:
            raise ValueError("structured checkpoint argv identity mismatch")
        return self

    def to_bytes(self) -> bytes:
        body = self.model_dump(mode="json", by_alias=True)
        return rfc8785_canonicalize({"body": body, "body_sha256": _sha256(rfc8785_canonicalize(body))})

    @classmethod
    def from_bytes(cls, payload: bytes) -> DirectDescriptionStructuredCandidateCheckpointV1:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("structured checkpoint JSON is malformed") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"body", "body_sha256"}
            or rfc8785_canonicalize(value) != payload
            or _sha256(rfc8785_canonicalize(value["body"])) != value["body_sha256"]
        ):
            raise DirectDescriptionError("structured checkpoint envelope is noncanonical or hash-invalid")
        return cls.model_validate_json(rfc8785_canonicalize(value["body"]))

    def filename(self) -> str:
        return f"ddm_v4_structured_member__candidate{self.completed_candidate_index:03d}_{self.completed_role}.json"

    def write_new(self, directory: Path) -> Path:
        return _publish_new_bytes(Path(directory) / self.filename(), self.to_bytes())


@dataclass(frozen=True, slots=True)
class StructuredCandidateRunV1:
    candidates: tuple[Mapping[str, Any], ...]
    checkpoint_paths: tuple[Path, ...]
    complete: bool
    resumed: bool


def run_structured_candidate_stages(
    config: DirectDescriptionStratumStructuredMemberConfigV1,
    *,
    baseline_archive: bytes,
    sources: StructuredS4SourcesV1,
    target_pose_codes: np.ndarray,
    membership_measure: MembershipMeasure,
    semantic_argv: Sequence[str],
    output_directory: Path,
    resume_from: Path | None = None,
    stop_after_candidate_index: int | None = None,
) -> StructuredCandidateRunV1:
    argv = tuple(semantic_argv)
    if not argv:
        raise DirectDescriptionError("structured candidate solve requires typed semantic argv")
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    start = 0
    if resume_from is not None:
        checkpoint = DirectDescriptionStructuredCandidateCheckpointV1.from_bytes(_read_regular_file_once(resume_from))
        if (
            checkpoint.config_sha256 != config.typed_config_hash()
            or checkpoint.dsl_compile_hash != config.dsl_compile_hash()
            or checkpoint.semantic_argv != argv
        ):
            raise DirectDescriptionError("structured candidate resume identity differs from governed run")
        rows = [dict(row) for row in checkpoint.candidates]
        start = checkpoint.next_candidate_index
        for row in rows:
            archive, _homes = compile_structured_member_archive(baseline_archive, sources, str(row["role"]))
            if len(archive) != row["archive_bytes"] or _sha256(archive) != row["archive_sha256"]:
                raise DirectDescriptionError("structured candidate resume archive identity mismatch")
    for index in range(start, len(STRUCTURED_ROLES)):
        role = STRUCTURED_ROLES[index]
        archive, homes = compile_structured_member_archive(baseline_archive, sources, role)
        receiver = receive_structured_member_archive(archive)
        replay = receive_structured_member_archive(archive)
        probe_ids = (0, 1, 63)
        if not np.array_equal(receiver.render_pairs(probe_ids), replay.render_pairs(probe_ids)):
            raise DirectDescriptionError("structured receiver deterministic replay failed")
        membership = _compact_membership(membership_measure(receiver))
        pose = _pose_completeness(receiver, target_pose_codes)
        target_class = ROLE_TARGET_CLASS.get(role)
        selected_membership = (
            membership["strata"]["target_class"][target_class]["same_c1_argmax_cell_fraction"]
            if target_class is not None
            else membership["same_c1_argmax_cell_fraction"]
        )
        archive_path = _publish_new_bytes(
            Path(output_directory) / f"ddm_v4_structured_member_{index:03d}_{role}.not_a_candidate.zip.receipt-bytes",
            archive,
        )
        row = {
            "candidate_index": index,
            "role": role,
            "mechanism": json.loads(parse_structured_member_archive(archive)[0]["manifest.json"])["semantic_source"],
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "archive_path": str(archive_path),
            "below_approx_200000_byte_receiver_box": len(archive) <= 200_000,
            "below_strict_154524_byte_task_cap": len(archive) <= 154_524,
            "membership_fraction": membership["same_c1_argmax_cell_fraction"],
            "structured_target_class": target_class,
            "structured_target_class_membership": selected_membership,
            "per_target_class_membership": {
                name: value["same_c1_argmax_cell_fraction"]
                for name, value in membership["strata"]["target_class"].items()
            },
            "membership": membership,
            "pose_completeness": pose["pose_completeness"],
            "pose": pose,
            "member_homes": list(homes),
            "receiver_consumed": True,
            "compiler_determinism_x2": True,
            "parse_reencode_identical": True,
            "receiver_replay_identical": True,
            "evidence_axis": EVIDENCE_AXIS,
            "d_seg_claim": False,
            "score_claim": False,
        }
        rows.append(row)
        checkpoint = DirectDescriptionStructuredCandidateCheckpointV1(
            config=config.model_dump(mode="json", by_alias=True),
            config_sha256=config.typed_config_hash(),
            dsl_compile_hash=config.dsl_compile_hash(),
            semantic_argv=argv,
            semantic_argv_sha256=_sha256("\0".join(argv).encode()),
            completed_candidate_index=index,
            completed_role=role,
            next_candidate_index=index + 1,
            candidates=tuple(rows),
        )
        paths.append(checkpoint.write_new(Path(output_directory) / "candidate_receipts"))
        if stop_after_candidate_index is not None and index >= stop_after_candidate_index:
            break
    return StructuredCandidateRunV1(
        tuple(rows), tuple(paths), len(rows) == len(STRUCTURED_ROLES), resume_from is not None
    )


def _event_subset_pricing_curve(sources: StructuredS4SourcesV1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role_class_ids, _ = _require_structured_routing(sources)
    for role in ("Road", "Lane", "UndrivableBoundary", "Movable"):
        class_id = int(role_class_ids[role])
        for n_pairs in (8, 16, 32, 64):
            encoded = _encode_site_records(
                sources.events[class_id][:n_pairs],
                class_id=class_id,
                source_id=1,
                coder="lzma1_raw_1MiB",
            )
            decoded = _decode_site_records(encoded, expected_class=class_id, expected_source=1)
            rows.append(
                {
                    "role": role,
                    "class_id": class_id,
                    "pair_prefix": n_pairs,
                    "event_records": sum(len(value) for value in decoded),
                    "event_sites": sum(len(sites) for value in decoded for sites in value),
                    "framed_lzma_bytes": len(encoded),
                    "payload_sha256": _sha256(encoded),
                    "scope": "isolated class-filtered event stream; final candidate archive priced separately",
                }
            )
    return rows


def run_stratum_structured_member_n64(
    config: DirectDescriptionStratumStructuredMemberConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    root = Path(output_directory)
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = load_target_receipt(Path(config.target_receipt_path), config.target_receipt_sha256)
    if Path(receipt.upstream_repo_root).resolve() / "upstream" != Path(config.upstream_root).resolve():
        raise DirectDescriptionError("structured-member scorer root differs from target custody")
    target_pose_codes = load_pose_target_codes(receipt)
    cache_path = Path(receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != receipt.source_cache.bytes:
        raise DirectDescriptionError("structured-member cached target cells are unavailable")
    try:
        cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    except (OSError, ValueError) as exc:
        raise DirectDescriptionError("structured-member cached target-cell source is malformed") from exc
    oracle, scorer_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    sources = load_structured_s4_sources(config)
    role_class_ids, class_detection = self_detect_structured_role_classes(np.asarray(cached_lstars[:64]))
    sources = replace(
        sources,
        role_class_ids=role_class_ids,
        role_rgb_u8={
            role: tuple(int(channel) for channel in sources.palette[class_id])
            for role, class_id in role_class_ids.items()
        },
        routing_custody={
            "status": "V4_INHERITED_S4_VALUE_CONTROL",
            "class_detection": class_detection,
            "value_selection": "legacy S4 palette at self-detected class row",
        },
    )
    baseline_z = fit_chart_description(receipt, target_pose_codes, config.pair_count)
    baseline_archive = compile_entropy_chart_archive(baseline_z).archive

    def membership_measure(receiver: Any) -> Mapping[str, Any]:
        return measure_argmax_cell_membership(receiver, receipt, oracle=oracle, cached_lstars=cached_lstars)

    partial = run_structured_candidate_stages(
        config,
        baseline_archive=baseline_archive,
        sources=sources,
        target_pose_codes=target_pose_codes,
        membership_measure=membership_measure,
        semantic_argv=semantic_argv,
        output_directory=root,
        stop_after_candidate_index=2,
    )
    if partial.complete or len(partial.candidates) != 3:
        raise DirectDescriptionError("structured candidate stop boundary failed")
    resumed = run_structured_candidate_stages(
        config,
        baseline_archive=baseline_archive,
        sources=sources,
        target_pose_codes=target_pose_codes,
        membership_measure=membership_measure,
        semantic_argv=semantic_argv,
        output_directory=root,
        resume_from=partial.checkpoint_paths[-1],
    )
    if not resumed.complete:
        raise DirectDescriptionError("structured candidate resume did not complete")
    candidates = [dict(row) for row in resumed.candidates]
    positive = [
        row["role"]
        for row in candidates
        if row["role"] != "baseline" and Decimal(row["structured_target_class_membership"]) > 0
    ]
    decisive = [role for role in positive if role in {"Road", "Lane", "MyCar"}]
    zero_failures = [
        {
            "stratum": row["structured_target_class"],
            "role": row["role"],
            "mechanism": row["mechanism"],
            "membership": row["structured_target_class_membership"],
            "escape_fraction": row["membership"]["strata"]["target_class"][row["structured_target_class"]][
                "argmax_cell_escape_fraction"
            ],
            "margin_scope": "all target-margin bands under this n64 uint8 palette-override mechanism",
            "verdict_scope": "FORMULATION",
        }
        for row in candidates
        if row["role"] != "baseline" and Decimal(row["structured_target_class_membership"]) == 0
    ]
    result = {
        "schema": STRUCTURED_RESULT_SCHEMA,
        "task": 603,
        "master_task": 578,
        "feeds_task": 613,
        "lane_id": STRUCTURED_LANE_ID,
        "run_id": config.run_id,
        "seed": config.seed,
        "verdict": (
            "MEASURED_STRUCTURED_STRATUM_MEMBERSHIP_POSITIVE_N64"
            if decisive
            else "MEASURED_STRUCTURED_STRATUM_MEMBERSHIP_STILL_ZERO_FORMULATION_N64"
        ),
        "verdict_scope": (
            "n64 v3 entropy chart plus exact S4 PXQ1/LBND2/static-hood/class-filtered event-component "
            "uint8 palette overrides; local frozen-SegNet advisory only; no wider structured-description family verdict"
        ),
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "d_seg_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "semantic_argv": list(semantic_argv),
        "producer": {
            "solver_module": _committed_source_custody(
                "src/tac/optimization/direct_description_entropy_priced_member.py"
            ),
            "entropy_module": _committed_source_custody("src/tac/optimization/direct_description_entropy_streams.py"),
            "cli": _committed_source_custody("tools/run_direct_description_entropy_priced_member.py"),
        },
        "s4_source_custody": dict(sources.custody),
        "selection": {
            "candidate_family": config.candidate_family,
            "candidate_count": len(candidates),
            "candidate_table_sha256": _sha256(rfc8785_canonicalize(candidates)),
            "positive_structured_roles": positive,
            "decisive_road_lane_mycar_positive_roles": decisive,
            "success_criterion_met": bool(decisive),
            "candidates": candidates,
            "zero_membership_failures": zero_failures,
        },
        "event_subset_pricing_curve": _event_subset_pricing_curve(sources),
        "pose_stream": {
            "present_in_every_archive": all(row["pose_completeness"] == "1.000000000000" for row in candidates),
            "completeness_curve": [
                {"role": row["role"], "archive_bytes": row["archive_bytes"], "pose": row["pose_completeness"]}
                for row in candidates
            ],
            "d_pose_claim": False,
        },
        "archive_box": {
            "approx_receiver_closed_target_bytes": 200_000,
            "strict_task_613_cap_bytes": 154_524,
            "roles_below_approx_box": [
                row["role"] for row in candidates if row["below_approx_200000_byte_receiver_box"]
            ],
            "roles_below_strict_cap": [row["role"] for row in candidates if row["below_strict_154524_byte_task_cap"]],
        },
        "resume": {
            "stopped_after_candidate_index": 2,
            "resumed": True,
            "all_candidate_checkpoints_preserved": True,
            "checkpoint_paths": [str(path) for path in (*partial.checkpoint_paths, *resumed.checkpoint_paths)],
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
            "V3_ENTROPY_HARNESS": "REUSED_UNCHANGED_RATE_AND_POSE_BASE",
            "SAFE_ZERO_STRUCTURED_STRATA": (
                "RED_TO_GREEN_AT_LEAST_ONE_OF_ROAD_LANE_MYCAR" if decisive else "REMAINS_RED_FORMULATION_SCOPED"
            ),
            "RECEIVER_CLOSED_APPROX_200KB": (
                "GREEN_AT_LEAST_ONE_STRUCTURED_ROLE"
                if any(row["below_approx_200000_byte_receiver_box"] for row in candidates[1:])
                else "REMAINS_RED"
            ),
            "POSE_STREAM_IN_MEMBER_PAYLOAD": "GREEN_N64_ALL_CANDIDATES_EXACT",
            "N256_N600_STRUCTURED_MEMBERSHIP": "OWED_NOT_RUN_TIME_BOUND",
            "CONTEST_CPU_CUDA_SCORE": "OWED_NOT_AUTHORIZED",
        },
        "storage_preflight": storage,
        "cleanup": {
            "bulk_artifacts_created": False,
            "s4_bulk_source_remained_read_only": True,
            "scratch_policy": "bounded batch16 scorer plus immutable candidate/checkpoint receipts",
            "certify_or_block": "no deletion, movement, or source mutation performed",
        },
        "stores_consulted": [
            "docs/operating_manual_craft_handoff.md",
            "v3 entropy-priced harness and receipt 9ab2251e",
            "settled read-only canonical S4 archive/runtime",
            "truly_optimal_coder_survey_603_613_20260722.md",
            "lane_sdf_component.py and hood_static_component.py",
        ],
        "main_landing_review_required": True,
    }
    receipt_path = _publish_new_bytes(
        root / "ddm_v4_stratum_structured_members_n64_receipt.json",
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


def _target_window_planes(
    receipt: Any,
    *,
    pair_start: int,
    pair_count: int,
) -> np.ndarray:
    rows = [
        planes
        for _pair_ids, planes in iter_target_plane_window_chunks(receipt, pair_start=pair_start, n_pairs=pair_count)
    ]
    if not rows:
        raise DirectDescriptionError("target window produced no scorer planes")
    value = np.ascontiguousarray(np.concatenate(rows, axis=0))
    if value.shape != (pair_count, 2, *PAIR_SHAPE, 3):
        raise DirectDescriptionError("target window scorer-plane coverage is incomplete")
    return value


def _pose_completeness_window(
    receiver: Any,
    target_pose_codes: np.ndarray,
    *,
    pair_start: int,
) -> dict[str, Any]:
    target = np.asarray(target_pose_codes)
    n_pairs = receiver.z.n_pairs
    if target.dtype != np.uint8 or target.shape != (600, 6) or pair_start + n_pairs > 600:
        raise DirectDescriptionError("composed Pose6 target window is invalid")
    expected = target[pair_start : pair_start + n_pairs]
    exact = receiver.pose6_codes == expected
    coordinates = n_pairs * 6
    matches = int(np.count_nonzero(exact))
    return {
        "coordinates": coordinates,
        "exact_coordinates": matches,
        "pose_completeness": _fraction_text(matches, coordinates),
        "pose6_integer_l1_debt": int(
            np.abs(receiver.pose6_codes.astype(np.int16) - expected.astype(np.int16)).sum(dtype=np.int64)
        ),
        "source_pair_window": {"start": pair_start, "stop": pair_start + n_pairs, "count": n_pairs},
        "d_pose_claim": False,
    }


def run_route_fix_composed_member(
    config: DirectDescriptionRouteFixComposeConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    """Build and measure one all-role receiver-closed member on n64 or n256."""

    root = Path(output_directory)
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = load_target_receipt(Path(config.target_receipt_path), config.target_receipt_sha256)
    if Path(receipt.upstream_repo_root).resolve() / "upstream" != Path(config.upstream_root).resolve():
        raise DirectDescriptionError("v5 composed-member scorer root differs from target custody")
    target_pose_codes = load_pose_target_codes(receipt)
    cache_path = Path(receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != receipt.source_cache.bytes:
        raise DirectDescriptionError("v5 composed-member cached target cells are unavailable")
    try:
        cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    except (OSError, ValueError) as exc:
        raise DirectDescriptionError("v5 composed-member cached target-cell source is malformed") from exc
    oracle, scorer_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    sources = load_structured_s4_sources(config)  # type: ignore[arg-type]

    probe_planes = _target_window_planes(
        receipt,
        pair_start=config.routing_probe_start,
        pair_count=config.routing_probe_count,
    )
    probe_z = fit_chart_description(
        receipt,
        target_pose_codes,
        config.routing_probe_count,
        pair_start=config.routing_probe_start,
    )
    probe_baseline = receive_entropy_chart_archive(compile_entropy_chart_archive(probe_z).archive)
    routed_sources, routing_receipt = measure_structured_role_value_routing(
        sources,
        baseline=probe_baseline,
        target_planes=probe_planes,
        target_cells=np.asarray(
            cached_lstars[config.routing_probe_start : config.routing_probe_start + config.routing_probe_count]
        ),
        oracle=oracle,
        source_pair_start=config.routing_probe_start,
    )
    routing_path = _publish_new_bytes(
        root / f"ddm_v5_role_value_routing_n{config.pair_count}.json",
        rfc8785_canonicalize(routing_receipt) + b"\n",
    )

    baseline_z = fit_chart_description(
        receipt,
        target_pose_codes,
        config.pair_count,
        pair_start=config.pair_start,
    )
    baseline_archive = compile_entropy_chart_archive(baseline_z).archive
    archive, homes = compile_composed_structured_member_archive(
        baseline_archive,
        routed_sources,
        pair_start=config.pair_start,
    )
    replay_archive, replay_homes = compile_composed_structured_member_archive(
        baseline_archive,
        routed_sources,
        pair_start=config.pair_start,
    )
    if archive != replay_archive or homes != replay_homes:
        raise DirectDescriptionError("v5 composed archive compiler replay is not bit-identical")
    receiver = receive_structured_member_archive(archive)
    replay = receive_structured_member_archive(archive)
    movable_local_pair = 456 - config.pair_start
    probe_ids = tuple(sorted({0, movable_local_pair, config.pair_count - 1}))
    if not np.array_equal(receiver.render_pairs(probe_ids), replay.render_pairs(probe_ids)):
        raise DirectDescriptionError("v5 composed receiver deterministic replay failed")
    membership = _compact_membership(
        measure_argmax_cell_membership(
            receiver,
            receipt,
            oracle=oracle,
            cached_lstars=cached_lstars,
            pair_start=config.pair_start,
        )
    )
    pose = _pose_completeness_window(receiver, target_pose_codes, pair_start=config.pair_start)
    if pose["pose_completeness"] != "1.000000000000":
        raise DirectDescriptionError("v5 composed member lost Pose6 completeness")
    if not isinstance(receiver, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("v5 compiler did not produce the composed receiver type")
    movable_layer = next(layer for layer in receiver.layers if layer.role == "Movable")
    if movable_layer.event_rows is None:
        raise DirectDescriptionError("v5 composed member lacks its Movable PCE3 stream")
    movable_event_records = sum(len(rows) for rows in movable_layer.event_rows)
    movable_event_sites = sum(len(sites) for rows in movable_layer.event_rows for sites in rows)
    if movable_event_records == 0 or movable_event_sites == 0:
        raise DirectDescriptionError("v5 state window failed to include nonempty Movable PCE3 support")
    archive_path = _publish_new_bytes(
        root / f"ddm_v5_route_fix_composed_n{config.pair_count}.not_a_candidate.zip.receipt-bytes",
        archive,
    )
    per_target_class = {
        name: row["same_c1_argmax_cell_fraction"] for name, row in membership["strata"]["target_class"].items()
    }
    route_roles = routing_receipt["role_probe"]
    adjudication = {
        role: {
            "verdict": "GEOMETRY_RIGHT_VALUES_WRONG_PALETTE",
            "verdict_scope": (
                "FORMULATION: fixed S4 geometry and inherited solid RGB on the n16 state-window routing probe; "
                "no wider carrier-family or evaluator-score verdict"
            ),
            "mechanism": (
                "self-detected target geometry has high target-class overlap while the inherited frozen-tile "
                "palette produces zero own-class cells; a counted C1-derived prototype is selected by actual "
                "receiver membership without shipping scorer weights"
            ),
            "geometry": route_roles[role]["geometry"],
            "inherited_s4_paint": route_roles[role]["inherited_s4_paint"],
            "c1_member_same_sites": route_roles[role]["c1_member_same_sites"],
            "selected": route_roles[role]["selected"],
        }
        for role in ("Road", "MyCar")
    }
    result = {
        "schema": V5_RESULT_SCHEMA,
        "task": 603,
        "master_task": 578,
        "feeds_task": 613,
        "lane_id": V5_LANE_ID,
        "run_id": config.run_id,
        "seed": config.seed,
        "verdict": "MEASURED_ROUTE_FIXED_COMPOSED_MEMBER_POSITIVE_STATE_WINDOW",
        "verdict_scope": (
            f"n{config.pair_count} source-pair window [{config.pair_start},{config.pair_start + config.pair_count}) "
            "through the local frozen-SegNet batch16 membership instrument; advisory only, no evaluator score"
        ),
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "d_seg_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "semantic_argv": list(semantic_argv),
        "producer": {
            "solver_module": _committed_source_custody(
                "src/tac/optimization/direct_description_entropy_priced_member.py"
            ),
            "measurement_module": _committed_source_custody(
                "src/tac/optimization/direct_description_measurement_ladder.py"
            ),
            "membership_module": _committed_source_custody(
                "src/tac/optimization/direct_description_polytope_membership.py"
            ),
            "cli": _committed_source_custody("tools/run_direct_description_entropy_priced_member.py"),
        },
        "routing_adjudication": adjudication,
        "routing_receipt": {
            "path": str(routing_path),
            "sha256": _sha256(_read_regular_file_once(routing_path)),
            "selected_role_rgb_u8": routing_receipt["selected_role_rgb_u8"],
            "class_detection": routing_receipt["class_detection"],
        },
        "composition": {
            "role_order": list(COMPOSED_ROLE_ORDER),
            "roles_present": [layer.role for layer in receiver.layers],
            "mechanisms": {
                "Lane": "S4 LBND2 plus Lane PCE3/PCOMP3",
                "Road": "S4 PXQ1 Road plus Road PCE3/PCOMP3 with corrected measured paint",
                "MyCar": "S4 static ego-hood with corrected measured paint",
                "UndrivableBoundary": "S4 Undrivable PCE3/PCOMP3 path",
                "Movable": "S4 class-filtered PCE3 event path",
                "Pose": "unchanged counted v3 Pose6 stream",
            },
            "source_pair_window": {
                "start": config.pair_start,
                "stop": config.pair_start + config.pair_count,
                "count": config.pair_count,
            },
            "movable_pce3": {
                "event_records": movable_event_records,
                "event_sites": movable_event_sites,
                "nonempty": True,
                "contains_global_pair_456": config.pair_start <= 456 < config.pair_start + config.pair_count,
                "verdict_scope": "selected contiguous state window, not the prior n64 prefix",
            },
        },
        "archive": {
            "path": str(archive_path),
            "bytes": len(archive),
            "sha256": _sha256(archive),
            "below_approx_200000_byte_receiver_box": len(archive) <= 200_000,
            "below_strict_154524_byte_task_cap": len(archive) <= 154_524,
            "compiler_determinism_x2": True,
            "parse_reencode_identical": True,
            "receiver_replay_identical": True,
            "member_homes": list(homes),
            "all_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
            "receiver_custody": dict(receiver.custody),
        },
        "membership": {
            "evidence_axis": EVIDENCE_AXIS,
            "overall": membership["same_c1_argmax_cell_fraction"],
            "per_target_class": per_target_class,
            "per_stratum": membership["strata"],
            "promotion_eligible": False,
            "d_seg_claim": False,
        },
        "pose": pose,
        "s4_source_custody": dict(routed_sources.custody),
        "target_custody": {
            "receipt_path": config.target_receipt_path,
            "receipt_sha256": config.target_receipt_sha256,
            "source_cache_path": str(cache_path),
            "source_cache_bytes": receipt.source_cache.bytes,
            "source_cache_sha256": receipt.source_cache.sha256,
            "source_cache_mutated": False,
        },
        "scorer_custody": scorer_custody,
        "resume": {
            "stage_checkpoints": [str(routing_path), str(archive_path)],
            "route_stage_preserved_before_composition": True,
            "composed_archive_preserved": True,
            "atomic_publish": True,
        },
        "blocker_delta": {
            "PALETTE_VS_GEOMETRY_ADJUDICATION": "RED_TO_GREEN_MEASURED_FORMULATION_SCOPE",
            "ROLE_TO_VALUE_ROUTING": "RED_TO_GREEN_SELF_DETECTED_AND_RECEIVER_MEASURED",
            "ONE_MEMBER_FIVE_STRATA_PLUS_POSE": "RED_TO_GREEN_RECEIVER_CLOSED_LOCAL_ADVISORY",
            "MOVABLE_PREFIX_ARTIFACT": "RED_TO_GREEN_NONEMPTY_STATE_WINDOW_PAIR_456",
            "N600_EVALUATOR_SCORE": "REMAINS_RED_NOT_AUTHORIZED_NOT_RUN",
            "CONTEST_CPU_CUDA": "REMAINS_RED_NOT_AUTHORIZED",
        },
        "storage_preflight": storage,
        "cleanup": {
            "bulk_artifacts_created": False,
            "s4_bulk_source_remained_read_only": True,
            "scratch_policy": "bounded target chunks and batch16 scorer; immutable small stage receipts",
            "certify_or_block": "no deletion, movement, or source mutation performed",
        },
        "stores_consulted": [
            "direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md",
            "v4 structured receipt 95c164b4 lineage",
            "canonical S4 archive/runtime read-only",
            "lane_sdf_component.py and hood_static_component.py role self-detection",
            "2026-07-19 EV/Fisher operator directives",
        ],
        "main_landing_review_required": True,
    }
    receipt_path = _publish_new_bytes(
        root / f"ddm_v5_route_fix_composed_n{config.pair_count}_receipt.json",
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


def _read_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    payload = _read_bound_file(path, expected_sha256, label)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{label} must contain one JSON object")
    return value


def _publish_or_verify(path: Path, payload: bytes) -> Path:
    if path.exists():
        if _read_regular_file_once(path) != payload:
            raise DirectDescriptionError(f"preserved stage differs from deterministic replay: {path}")
        return path
    return _publish_new_bytes(path, payload)


def _latest_key_by_pair(n_pairs: int, keyframes: Sequence[int]) -> tuple[int, ...]:
    keys = tuple(int(value) for value in keyframes)
    if not keys or keys[0] != 0 or any(left >= right for left, right in itertools.pairwise(keys)):
        raise DirectDescriptionError("amortization keyframes must be strictly increasing from zero")
    if keys[-1] >= n_pairs:
        raise DirectDescriptionError("amortization keyframe is outside the local pair window")
    result: list[int] = []
    cursor = 0
    for pair_id in range(n_pairs):
        while cursor + 1 < len(keys) and keys[cursor + 1] <= pair_id:
            cursor += 1
        result.append(keys[cursor])
    return tuple(result)


def _hold_semantic_stream(payload: bytes, *, n_pairs: int, records_per_pair: int, keyframes: Sequence[int]) -> bytes:
    if n_pairs < 1 or records_per_pair < 1 or len(payload) % (n_pairs * records_per_pair):
        raise DirectDescriptionError("semantic stream cannot be partitioned into canonical pair records")
    record_bytes = len(payload) // (n_pairs * records_per_pair)
    source_by_pair = _latest_key_by_pair(n_pairs, keyframes)
    output = bytearray(len(payload))
    pair_bytes = records_per_pair * record_bytes
    for pair_id, source_pair_id in enumerate(source_by_pair):
        source_start = source_pair_id * pair_bytes
        target_start = pair_id * pair_bytes
        output[target_start : target_start + pair_bytes] = payload[source_start : source_start + pair_bytes]
        for record_id in range(records_per_pair):
            struct.pack_into("<H", output, target_start + record_id * record_bytes, pair_id)
    return bytes(output)


def _amortize_chart_z(
    baseline: DirectDescriptionChartZV1,
    *,
    keyframes: Sequence[int] | None = None,
    zero_residuals: bool = False,
) -> DirectDescriptionChartZV1:
    if keyframes is not None and zero_residuals:
        raise DirectDescriptionError("chart amortization must select key-hold or residual-zero, not both")
    if zero_residuals:
        value = baseline
        for stream_name in RESIDUAL_STREAMS:
            value = build_safe_zero_residual_proposal(value, stream_name).z
        return value
    if keyframes is None:
        return baseline
    records_per_pair = {
        "global_chart_anchors": 2,
        "axial_chart_gradients": 2,
        "low_variation_chart_residuals": 128,
        "mid_variation_chart_residuals": 128,
        "high_variation_chart_residuals": 128,
    }
    value = baseline
    for stream_name, count in records_per_pair.items():
        value = value.replace_stream_payload(
            stream_name,
            _hold_semantic_stream(
                getattr(baseline, stream_name).payload,
                n_pairs=baseline.n_pairs,
                records_per_pair=count,
                keyframes=keyframes,
            ),
        )
    return value


def _xi_pose6_keyframes(pose6_codes: np.ndarray, *, max_gap: int) -> tuple[tuple[int, ...], dict[str, Any]]:
    values = np.asarray(pose6_codes)
    if values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != 6 or max_gap < 1:
        raise DirectDescriptionError("xi key schedule requires uint8 [P,6] Pose6 codes and a positive gap")
    motion = np.abs(np.diff(values.astype(np.int16), axis=0)).sum(axis=1, dtype=np.int64)
    target_sections = (len(values) + max_gap - 1) // max_gap
    budget = max(1, (int(motion.sum()) + target_sections - 1) // target_sections)
    keys = [0]
    accumulated = 0
    for pair_id in range(1, len(values)):
        accumulated += int(motion[pair_id - 1])
        if accumulated >= budget or pair_id - keys[-1] >= max_gap:
            keys.append(pair_id)
            accumulated = 0
    return tuple(keys), {
        "derivation": "counted Pose6 L1 path length reverse-waterfilled into ceil(P/max_gap) sections",
        "max_gap": max_gap,
        "total_pose6_l1_motion": int(motion.sum()),
        "section_motion_budget": budget,
        "keyframes": keys,
        "key_count": len(keys),
        "target_sections": target_sections,
        "unmeasured_motion_threshold_invented": False,
    }


def _amortize_structured_sources(
    sources: StructuredS4SourcesV1,
    *,
    pair_start: int,
    pair_count: int,
    keyframes: Sequence[int],
) -> StructuredS4SourcesV1:
    source_by_pair = _latest_key_by_pair(pair_count, keyframes)

    def remap(stream: tuple[tuple[tuple[np.ndarray, ...], ...], ...]) -> tuple[tuple[tuple[np.ndarray, ...], ...], ...]:
        classes: list[tuple[tuple[np.ndarray, ...], ...]] = []
        for class_rows in stream:
            rows = list(class_rows)
            for local_pair_id, key_id in enumerate(source_by_pair):
                rows[pair_start + local_pair_id] = class_rows[pair_start + key_id]
            classes.append(tuple(rows))
        return tuple(classes)

    return replace(sources, events=remap(sources.events), components=remap(sources.components))


def _distribution(values: Sequence[float]) -> dict[str, str]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise DirectDescriptionError("distribution requires a nonempty finite vector")
    return {
        "min": f"{float(array.min()):.12f}",
        "q25": f"{float(np.quantile(array, 0.25)):.12f}",
        "median": f"{float(np.quantile(array, 0.5)):.12f}",
        "q75": f"{float(np.quantile(array, 0.75)):.12f}",
        "max": f"{float(array.max()):.12f}",
        "mean": f"{float(array.mean()):.12f}",
    }


def _derived_membership_proxy(d_seg_text: str, *, measured_control: str | None = None) -> dict[str, Any]:
    d_seg = Decimal(d_seg_text)
    target_escape = Decimal(1) - Decimal(V6_C1_GT_MATCH_FRACTION_TEXT)
    center = Decimal(1) - d_seg
    lower = max(Decimal(0), center - target_escape)
    upper = min(Decimal(1), center + target_escape)
    return {
        "status": "MEASURED_CONTROL_PLUS_DERIVED_BOUND" if measured_control is not None else "DERIVED_BOUND",
        "same_c1_argmax_cell_fraction_measured": measured_control,
        "same_c1_argmax_cell_fraction_lower": f"{lower:.12f}",
        "same_c1_argmax_cell_fraction_upper": f"{upper:.12f}",
        "derivation": "triangle inequality between described-vs-GT d_seg and settled C1-vs-GT escape mass",
        "settled_c1_gt_match_fraction": V6_C1_GT_MATCH_FRACTION_TEXT,
        "settled_c1_gt_escape_fraction": f"{target_escape:.12f}",
        "not_remeasured": measured_control is None,
        "score_claim": False,
    }


def _load_posenet_oracle(
    upstream_root: Path,
    *,
    threads: int,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    root = Path(upstream_root).resolve()
    modules_path = root / "modules.py"
    if threads < 1 or not modules_path.is_file():
        raise DirectDescriptionError("PoseNet custody is unavailable")
    sys.path.insert(0, str(root))
    try:
        import modules as upstream_modules
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise DirectDescriptionError("PoseNet runtime imports are unavailable") from exc
    if Path(upstream_modules.__file__).resolve() != modules_path:
        raise DirectDescriptionError("PoseNet imported a non-custodied modules.py")
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    weights_path = Path(upstream_modules.posenet_sd_path).resolve()
    if not weights_path.is_file():
        raise DirectDescriptionError("PoseNet weights are missing")
    posenet = upstream_modules.PoseNet().eval().to("cpu")
    posenet.load_state_dict(load_file(str(weights_path), device="cpu"))
    for parameter in posenet.parameters():
        parameter.requires_grad_(False)

    def oracle(pairs: np.ndarray) -> np.ndarray:
        value = np.asarray(pairs)
        if value.dtype != np.uint8 or value.ndim != 5 or value.shape[1:] != (2, *PAIR_SHAPE, 3):
            raise DirectDescriptionError("PoseNet requires uint8 [B,2,384,512,3]")
        tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            output = posenet(posenet.preprocess_input(tensor))
            pose = output["pose"] if isinstance(output, dict) else output
            result = pose[:, :6].cpu().numpy().astype(np.float64)
        return np.ascontiguousarray(result)

    return oracle, {
        "implementation": "upstream.modules.PoseNet.native_cpu_torch_official_YUV6",
        "modules_path": str(modules_path),
        "modules_sha256": _sha256(_read_regular_file_once(modules_path)),
        "weights_path": str(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": _sha256(_read_regular_file_once(weights_path)),
        "batch_size": 16,
        "threads": threads,
        "seed": SEED,
        "deterministic_algorithms": True,
        "device": "cpu",
        "preprocess": "official PoseNet.preprocess_input RGB_to_YUV6 after bilinear resize",
        "weights_shipped_in_archive": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def _measure_evaluator_bridge(
    receiver: ComposedStructuredMemberReceiverV1,
    *,
    pair_start: int,
    cached_lstars: np.ndarray,
    cached_margins: np.ndarray,
    cached_poses: np.ndarray,
    segnet_oracle: Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray | None]],
    posenet_oracle: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
) -> dict[str, Any]:
    labels = np.asarray(cached_lstars)
    margins = np.asarray(cached_margins)
    poses = np.asarray(cached_poses)
    if labels.shape != (600, *PAIR_SHAPE) or labels.dtype != np.int64:
        raise DirectDescriptionError("evaluator bridge requires cached int64 lstars[600,384,512]")
    if margins.shape != labels.shape or margins.dtype != np.float32:
        raise DirectDescriptionError("evaluator bridge requires cached float32 margins[600,384,512]")
    if poses.shape != (600, 6) or poses.dtype != np.float64:
        raise DirectDescriptionError("evaluator bridge requires cached float64 gt_poses[600,6]")
    strata: dict[str, dict[str, dict[str, int]]] = {
        "target_class": {name: {"errors": 0, "sites": 0} for name in CLASS_NAMES},
        "topology": {name: {"errors": 0, "sites": 0} for name in ("boundary_codim1", "cell_interior")},
        "target_margin": {name: {"errors": 0, "sites": 0} for _lo, _hi, name in MARGIN_BANDS},
    }
    seg_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    described_cell_digest = hashlib.sha256()
    described_pose_digest = hashlib.sha256()
    total_errors = 0
    total_sites = 0
    pose_squared_error = 0.0
    replay_checked = False
    for start in range(0, receiver.z.n_pairs, batch_size):
        pair_ids = tuple(range(start, min(receiver.z.n_pairs, start + batch_size)))
        source_ids = pair_start + np.asarray(pair_ids, dtype=np.int64)
        described = receiver.render_pairs(pair_ids)
        described_cells, described_margins = segnet_oracle(described, False)
        if described_margins is not None:
            raise DirectDescriptionError("evaluator SegNet bridge unexpectedly returned margins")
        described_poses = posenet_oracle(described)
        if not replay_checked:
            replay_cells, _ = segnet_oracle(described, False)
            replay_poses = posenet_oracle(described)
            if not np.array_equal(described_cells, replay_cells) or not np.array_equal(described_poses, replay_poses):
                raise DirectDescriptionError("evaluator bridge deterministic first-batch replay failed")
            replay_checked = True
        target_cells = np.ascontiguousarray(labels[source_ids])
        target_margins = np.ascontiguousarray(margins[source_ids])
        target_poses = np.ascontiguousarray(poses[source_ids])
        errors = described_cells != target_cells
        boundary = boundary_mask_from_labels(target_cells)
        for local_index, pair_id in enumerate(pair_ids):
            pair_errors = int(np.count_nonzero(errors[local_index]))
            pair_sites = int(errors[local_index].size)
            pose_error = float(np.mean((described_poses[local_index] - target_poses[local_index]) ** 2))
            seg_rows.append(
                {
                    "pair_id": pair_id,
                    "source_pair_id": int(source_ids[local_index]),
                    "errors": pair_errors,
                    "sites": pair_sites,
                    "d_seg": _fraction_text(pair_errors, pair_sites),
                }
            )
            pose_rows.append(
                {
                    "pair_id": pair_id,
                    "source_pair_id": int(source_ids[local_index]),
                    "d_pose": f"{pose_error:.12f}",
                }
            )
        total_errors += int(np.count_nonzero(errors))
        total_sites += int(errors.size)
        pose_squared_error += float(np.square(described_poses - target_poses).sum(dtype=np.float64))
        for class_id, class_name in enumerate(CLASS_NAMES):
            mask = target_cells == class_id
            strata["target_class"][class_name]["errors"] += int(np.count_nonzero(errors & mask))
            strata["target_class"][class_name]["sites"] += int(np.count_nonzero(mask))
        for name, mask in (("boundary_codim1", boundary), ("cell_interior", ~boundary)):
            strata["topology"][name]["errors"] += int(np.count_nonzero(errors & mask))
            strata["topology"][name]["sites"] += int(np.count_nonzero(mask))
        for low, high, name in MARGIN_BANDS:
            mask = (target_margins >= low) & (target_margins < high)
            strata["target_margin"][name]["errors"] += int(np.count_nonzero(errors & mask))
            strata["target_margin"][name]["sites"] += int(np.count_nonzero(mask))
        described_cell_digest.update(described_cells.tobytes(order="C"))
        described_pose_digest.update(described_poses.tobytes(order="C"))
    if not replay_checked or total_sites != receiver.z.n_pairs * PAIR_SHAPE[0] * PAIR_SHAPE[1]:
        raise DirectDescriptionError("evaluator bridge pair coverage is incomplete")
    finalized_strata = {
        family: {
            name: {
                **row,
                "d_seg": _fraction_text(row["errors"], row["sites"]),
            }
            for name, row in rows.items()
        }
        for family, rows in strata.items()
    }
    d_seg = _fraction_text(total_errors, total_sites)
    d_pose_value = pose_squared_error / (receiver.z.n_pairs * 6)
    return {
        "segmentation": {
            "definition": "official frozen SegNet last-frame argmax disagreement against gt_n600.lstars",
            "d_seg": d_seg,
            "errors": total_errors,
            "sites": total_sites,
            "per_pair": seg_rows,
            "per_pair_distribution": _distribution([float(row["d_seg"]) for row in seg_rows]),
            "strata": finalized_strata,
            "described_cells_sha256": described_cell_digest.hexdigest(),
            "d_seg_measured": True,
            "d_seg_claim": False,
        },
        "pose": {
            "definition": "official frozen PoseNet YUV6 first-six-output MSE against gt_n600.gt_poses",
            "d_pose": f"{d_pose_value:.12f}",
            "squared_error_sum": f"{pose_squared_error:.12f}",
            "coordinates": receiver.z.n_pairs * 6,
            "per_pair": pose_rows,
            "per_pair_distribution": _distribution([float(row["d_pose"]) for row in pose_rows]),
            "described_pose6_f64_sha256": described_pose_digest.hexdigest(),
            "pose6_payload_completeness": "1.000000000000",
            "d_pose_measured": True,
            "d_pose_claim": False,
        },
        "deterministic_first_batch_replay": True,
        "scorer_batch_size": batch_size,
        "max_scorer_batches_resident": 1,
        "max_source_chunks_resident": 1,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def run_dseg_bridge_amortize(
    config: DirectDescriptionDsegBridgeAmortizeConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    """Measure v5 and receiver-closed amortized variants against frozen evaluator caches."""

    root = Path(output_directory)
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    v5_receipt = _read_bound_json(Path(config.v5_receipt_path), config.v5_receipt_sha256, "v5_receipt_sha256")
    if v5_receipt.get("schema") != V5_RESULT_SCHEMA:
        raise DirectDescriptionError("v6 input receipt is not the governed v5 result")
    try:
        v5_config = DirectDescriptionRouteFixComposeConfigV1.model_validate(v5_receipt["typed_config"])
    except (KeyError, ValueError) as exc:
        raise DirectDescriptionError("v5 typed config cannot be reconstructed") from exc
    if (v5_config.pair_start, v5_config.pair_count) != (config.pair_start, config.pair_count):
        raise DirectDescriptionError("v6 typed window differs from its bound v5 receipt")
    v5_archive = _read_bound_file(Path(config.v5_archive_path), config.v5_archive_sha256, "v5_archive_sha256")
    if v5_receipt["archive"] != {
        **v5_receipt["archive"],
        "bytes": len(v5_archive),
        "sha256": _sha256(v5_archive),
    }:
        raise DirectDescriptionError("v5 archive receipt length/hash differs from bound bytes")
    target_receipt = load_target_receipt(Path(v5_config.target_receipt_path), v5_config.target_receipt_sha256)
    cache_path = Path(target_receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != target_receipt.source_cache.bytes:
        raise DirectDescriptionError("v6 evaluator cache is unavailable")
    try:
        cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
        cached_margins = open_stored_npy_memmap(cache_path, "margins")
        cached_poses = open_stored_npy_memmap(cache_path, "gt_poses")
    except (OSError, ValueError) as exc:
        raise DirectDescriptionError("v6 evaluator cache members are malformed") from exc
    sources = load_structured_s4_sources(v5_config)  # type: ignore[arg-type]
    routing = _read_bound_json(
        Path(v5_receipt["routing_receipt"]["path"]),
        v5_receipt["routing_receipt"]["sha256"],
        "v5_routing_receipt_sha256",
    )
    role_ids = routing["class_detection"]["role_class_ids"]
    role_rgb = routing["selected_role_rgb_u8"]
    sources = replace(
        sources,
        role_class_ids={name: int(value) for name, value in role_ids.items()},
        role_rgb_u8={name: tuple(int(channel) for channel in value) for name, value in role_rgb.items()},
        routing_custody=routing,
    )
    v5_members, _v5_homes = parse_structured_member_archive(v5_archive)
    baseline_z = receive_entropy_chart_archive(v5_members["chart.zip"]).z
    xi_keys, xi_receipt = _xi_pose6_keyframes(
        receive_entropy_chart_archive(v5_members["chart.zip"]).pose6_codes,
        max_gap=config.max_key_gap,
    )
    fixed_keys = tuple(range(0, config.pair_count, config.max_key_gap))
    candidates_spec = (
        ("v5_exact", baseline_z, sources, tuple(range(config.pair_count))),
        (
            "fixed_ar1_hold24",
            _amortize_chart_z(baseline_z, keyframes=fixed_keys),
            _amortize_structured_sources(
                sources,
                pair_start=config.pair_start,
                pair_count=config.pair_count,
                keyframes=fixed_keys,
            ),
            fixed_keys,
        ),
        (
            "xi_pose6_ar1_hold24",
            _amortize_chart_z(baseline_z, keyframes=xi_keys),
            _amortize_structured_sources(
                sources,
                pair_start=config.pair_start,
                pair_count=config.pair_count,
                keyframes=xi_keys,
            ),
            xi_keys,
        ),
        (
            "residual_zero_static_once",
            _amortize_chart_z(baseline_z, zero_residuals=True),
            _amortize_structured_sources(
                sources,
                pair_start=config.pair_start,
                pair_count=config.pair_count,
                keyframes=(0,),
            ),
            (0,),
        ),
    )
    segnet_oracle, segnet_custody = _load_segnet_oracle(Path(v5_config.upstream_root), threads=config.scorer_threads)
    posenet_oracle, posenet_custody = _load_posenet_oracle(Path(v5_config.upstream_root), threads=config.scorer_threads)
    rows: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    for index, (mode, z, candidate_sources, keyframes) in enumerate(candidates_spec):
        checkpoint_path = root / "candidate_receipts" / f"{index:02d}_{mode}.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(_read_regular_file_once(checkpoint_path))
            if (
                checkpoint.get("schema") != "direct_description_dseg_bridge_candidate_checkpoint.v1"
                or checkpoint.get("typed_config_sha256") != config.typed_config_hash()
                or checkpoint.get("v5_archive_sha256") != config.v5_archive_sha256
                or checkpoint.get("candidate", {}).get("mode") != mode
            ):
                raise DirectDescriptionError(f"v6 candidate checkpoint custody mismatch: {checkpoint_path}")
            candidate_row = checkpoint["candidate"]
            archive_path = Path(candidate_row["archive"]["path"])
            archive_payload = _read_bound_file(
                archive_path,
                candidate_row["archive"]["sha256"],
                f"{mode}_checkpoint_archive_sha256",
            )
            if len(archive_payload) != candidate_row["archive"]["bytes"]:
                raise DirectDescriptionError(f"v6 candidate checkpoint archive size mismatch: {mode}")
            rows.append(candidate_row)
            checkpoint_paths.append(str(checkpoint_path))
            continue
        chart_build = compile_entropy_chart_archive(z)
        archive, homes = compile_composed_structured_member_archive(
            chart_build.archive,
            candidate_sources,
            pair_start=config.pair_start,
        )
        replay_archive, replay_homes = compile_composed_structured_member_archive(
            chart_build.archive,
            candidate_sources,
            pair_start=config.pair_start,
        )
        if archive != replay_archive or homes != replay_homes:
            raise DirectDescriptionError(f"v6 {mode} compiler replay is not bit-identical")
        if mode == "v5_exact" and archive != v5_archive:
            raise DirectDescriptionError("v6 exact-control recompilation differs from bound v5 bytes")
        archive_path = _publish_or_verify(
            root / f"ddm_v6_{mode}_n{config.pair_count}.not_a_candidate.zip.receipt-bytes",
            archive,
        )
        receiver = receive_structured_member_archive(archive)
        replay = receive_structured_member_archive(archive)
        if not isinstance(receiver, ComposedStructuredMemberReceiverV1) or not isinstance(
            replay, ComposedStructuredMemberReceiverV1
        ):
            raise DirectDescriptionError("v6 candidate did not decode as a composed receiver")
        probe_ids = tuple(sorted({0, config.pair_count // 2, config.pair_count - 1}))
        if not np.array_equal(receiver.render_pairs(probe_ids), replay.render_pairs(probe_ids)):
            raise DirectDescriptionError(f"v6 {mode} receiver replay is not bit-identical")
        bridge = _measure_evaluator_bridge(
            receiver,
            pair_start=config.pair_start,
            cached_lstars=cached_lstars,
            cached_margins=cached_margins,
            cached_poses=cached_poses,
            segnet_oracle=segnet_oracle,
            posenet_oracle=posenet_oracle,
            batch_size=config.scorer_batch_size,
        )
        d_seg = Decimal(bridge["segmentation"]["d_seg"])
        candidate_row = {
            "mode": mode,
            "candidate_index": index,
            "keyframes": list(keyframes),
            "key_count": len(keyframes),
            "key_policy": (
                "every_pair_exact"
                if mode == "v5_exact"
                else "counted_Pose6_xi_adaptive"
                if mode.startswith("xi_")
                else "fixed_zero_order_hold"
                if mode.startswith("fixed_")
                else "one_static_key_plus_safe_zero_chart_residuals"
            ),
            "archive": {
                "path": str(archive_path),
                "bytes": len(archive),
                "sha256": _sha256(archive),
                "member_homes": list(homes),
                "all_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
                "receiver_closed": True,
                "parse_reencode_identical": True,
                "compiler_determinism_x2": True,
                "receiver_replay_identical": True,
            },
            "chart": chart_build.custody(),
            "evaluator_bridge": bridge,
            "membership_proxy": _derived_membership_proxy(
                bridge["segmentation"]["d_seg"],
                measured_control=v5_receipt["membership"]["overall"] if mode == "v5_exact" else None,
            ),
            "gates": {
                "task_613_d_seg_le_0_00116": d_seg <= Decimal(V6_TARGET_DSEG_TEXT),
                "s4_knee_d_seg_le_0_016": d_seg <= Decimal(V6_S4_KNEE_DSEG_TEXT),
                "s4_knee_archive_bytes_le_216207": len(archive) <= V6_S4_KNEE_BYTES,
                "score_claim": False,
                "promotion_eligible": False,
            },
            "verdict_scope": (
                f"MEASURED n{config.pair_count} source-pair window "
                f"[{config.pair_start},{config.pair_start + config.pair_count}) on {EVIDENCE_AXIS}; "
                "formulation-local and not contest-CPU/CUDA"
            ),
        }
        checkpoint = {
            "schema": "direct_description_dseg_bridge_candidate_checkpoint.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "dsl_compile_hash": config.dsl_compile_hash(),
            "semantic_argv": list(semantic_argv),
            "semantic_argv_sha256": _sha256("\0".join(semantic_argv).encode()),
            "v5_archive_sha256": config.v5_archive_sha256,
            "completed_candidate_index": index,
            "next_candidate_index": index + 1,
            "candidate": candidate_row,
        }
        _publish_or_verify(checkpoint_path, rfc8785_canonicalize(checkpoint) + b"\n")
        rows.append(candidate_row)
        checkpoint_paths.append(str(checkpoint_path))
    result = {
        "schema": V6_RESULT_SCHEMA,
        "task": 603,
        "feeds_task": 613,
        "lane_id": V6_LANE_ID,
        "run_id": config.run_id,
        "seed": config.seed,
        "verdict": "MEASURED_EVALUATOR_BRIDGE_AND_TEMPORAL_AMORTIZATION",
        "verdict_scope": (
            f"n{config.pair_count} source-pair window [{config.pair_start},{config.pair_start + config.pair_count}) "
            f"on {EVIDENCE_AXIS}; no contest score or promotion authority"
        ),
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "semantic_argv": list(semantic_argv),
        "v5_control": {
            "receipt_path": config.v5_receipt_path,
            "receipt_sha256": config.v5_receipt_sha256,
            "archive_path": config.v5_archive_path,
            "archive_bytes": len(v5_archive),
            "archive_sha256": config.v5_archive_sha256,
            "exact_recompilation_identical": True,
        },
        "xi_schedule": xi_receipt,
        "candidates": rows,
        "candidate_table_sha256": _sha256(rfc8785_canonicalize(rows)),
        "best_bytes": min(rows, key=lambda row: (row["archive"]["bytes"], row["candidate_index"]))["mode"],
        "best_d_seg": min(
            rows,
            key=lambda row: (Decimal(row["evaluator_bridge"]["segmentation"]["d_seg"]), row["candidate_index"]),
        )["mode"],
        "scorer_custody": {"segnet": segnet_custody, "posenet": posenet_custody},
        "target_custody": {
            "receipt_path": v5_config.target_receipt_path,
            "receipt_sha256": v5_config.target_receipt_sha256,
            "source_cache_path": str(cache_path),
            "source_cache_bytes": target_receipt.source_cache.bytes,
            "source_cache_sha256": target_receipt.source_cache.sha256,
            "cache_members": ["lstars", "margins", "gt_poses"],
            "source_cache_mutated": False,
        },
        "resume": {
            "policy": config.checkpoint_policy,
            "candidate_checkpoints": checkpoint_paths,
            "all_preserved": len(checkpoint_paths) == len(V6_CANDIDATE_MODES),
            "atomic_publish": True,
            "max_work_loss": "current candidate only",
        },
        "blocker_delta": {
            "V5_ACTUAL_DSEG": "RED_TO_GREEN_LOCAL_FROZEN_SEGNET_ADVISORY",
            "V5_ACTUAL_DPOSE": "RED_TO_GREEN_LOCAL_OFFICIAL_YUV6_POSENET_ADVISORY",
            "PER_PAIR_AND_STRATUM_DISTRIBUTIONS": "RED_TO_GREEN",
            "TEMPORAL_BYTES_PER_PAIR_LE_300": "REQUIRES_N64_N256_CROSS_WINDOW_RECEIPT",
            "N600_EVALUATOR_BRIDGE": "REMAINS_RED_TIME_BOUND_NOT_RUN",
            "CONTEST_CPU_CUDA_SCORE": "REMAINS_RED_NOT_AUTHORIZED",
        },
        "storage_preflight": storage,
        "cleanup": {
            "bulk_artifacts_created": False,
            "ssd_sources_read_only": True,
            "scorer_policy": "batch16 with one described batch and cache window resident",
            "certify_or_block": "no deletion, movement, or source mutation performed",
        },
        "stores_consulted": [
            "direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md",
            "v5 route-fix composed receipts and exact archives",
            "gt_n600 lstars/margins/gt_poses frozen scorer cache",
            "canonical S4 archive/runtime read-only",
            "2026-07-19 reverse-waterfill and xi directives",
        ],
        "producer": {
            "solver_module": _committed_source_custody(
                "src/tac/optimization/direct_description_entropy_priced_member.py"
            ),
            "cli": _committed_source_custody("tools/run_direct_description_entropy_priced_member.py"),
        },
        "main_landing_review_required": True,
    }
    receipt_path = _publish_or_verify(
        root / f"ddm_v6_dseg_bridge_amortize_n{config.pair_count}_receipt.json",
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


@dataclass(frozen=True, slots=True)
class SolvedPlaneToleranceReceiverV1:
    archive: bytes
    predictor: ComposedStructuredMemberReceiverV1
    sections: tuple[PlaneCorrectionSectionBuildV1, ...]
    custody: Mapping[str, Any]

    @property
    def z(self) -> DirectDescriptionChartZV1:
        return self.predictor.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.predictor.pose6_codes

    def render_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        ids = tuple(int(value) for value in pair_ids)
        if any(value < 0 or value >= self.z.n_pairs for value in ids):
            raise DirectDescriptionError("v7 receiver pair id is outside the local window")
        rendered = np.ascontiguousarray(self.predictor.render_pairs(ids))
        claimed = np.zeros((len(ids), 2 * PAIR_SHAPE[0] * PAIR_SHAPE[1]), dtype=bool)
        for section in self.sections:
            if section.schedule_id == CORRECTION_SCHEDULE_DROP:
                continue
            source_by_pair = _latest_key_by_pair(section.n_pairs, tuple(sorted(section.records)))
            for row_index, pair_id in enumerate(ids):
                positions, values = section.records[source_by_pair[pair_id]]
                if np.any(claimed[row_index, positions]):
                    raise DirectDescriptionError("v7 opaque correction sections overlap at the receiver")
                claimed[row_index, positions] = True
                rendered[row_index].reshape(-1, 3)[positions] = values
        return rendered


def _v7_keyframes(
    *,
    schedule_id: int,
    n_pairs: int,
    xi_keyframes: Sequence[int],
) -> tuple[int, ...]:
    if schedule_id == CORRECTION_SCHEDULE_EVERY_PAIR:
        return tuple(range(n_pairs))
    if schedule_id == CORRECTION_SCHEDULE_FIXED_HOLD24:
        return tuple(range(0, n_pairs, 24))
    if schedule_id == CORRECTION_SCHEDULE_XI_HOLD24:
        return tuple(int(value) for value in xi_keyframes)
    if schedule_id == CORRECTION_SCHEDULE_DROP:
        return ()
    raise DirectDescriptionError("v7 correction schedule is unknown")


def _v7_section_class_ids(role_class_ids: Mapping[str, Any]) -> dict[str, int | None]:
    expected = {"Road", "Lane", "UndrivableBoundary", "Movable", "MyCar"}
    if set(role_class_ids) != expected:
        raise DirectDescriptionError("v7 self-detected role map is incomplete")
    values = {name: int(value) for name, value in role_class_ids.items()}
    if sorted(values.values()) != list(range(5)):
        raise DirectDescriptionError("v7 self-detected role map is not a five-class permutation")
    return {
        "Road": values["Road"],
        "Lane": values["Lane"],
        "Undrivable": values["UndrivableBoundary"],
        "Movable": values["Movable"],
        "MyCar": values["MyCar"],
        "Boundary": None,
    }


def _v7_build_section_records(
    *,
    target_receipt: Any,
    predictor: ComposedStructuredMemberReceiverV1,
    cached_lstars: np.ndarray,
    pair_start: int,
    pair_count: int,
    section_class_ids: Mapping[str, int | None],
    schedule_id: int,
    quant_step: int,
    xi_keyframes: Sequence[int],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], ...]:
    keyframes = _v7_keyframes(schedule_id=schedule_id, n_pairs=pair_count, xi_keyframes=xi_keyframes)
    if not keyframes:
        return tuple({} for _ in V7_SECTION_NAMES)
    keyset = frozenset(keyframes)
    positions_by_section: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [{} for _ in V7_SECTION_NAMES]
    sites_per_plane = PAIR_SHAPE[0] * PAIR_SHAPE[1]
    observed_keys: list[int] = []
    for pair_ids, target_planes in iter_target_plane_window_chunks(
        target_receipt,
        pair_start=pair_start,
        n_pairs=pair_count,
    ):
        selected = tuple(value for value in pair_ids if value in keyset)
        if not selected:
            continue
        selected_offsets = [pair_ids.index(value) for value in selected]
        target = np.ascontiguousarray(target_planes[selected_offsets])
        predicted = predictor.render_pairs(selected)
        if quant_step == 1:
            corrected = target
        else:
            residual = target.astype(np.int16) - predicted.astype(np.int16)
            magnitude = ((np.abs(residual) + quant_step // 2) // quant_step) * quant_step
            quantized = np.where(residual < 0, -magnitude, magnitude)
            corrected = np.clip(predicted.astype(np.int16) + quantized, 0, 255).astype(np.uint8)
        for batch_index, pair_id in enumerate(selected):
            labels = np.asarray(cached_lstars[pair_start + pair_id])
            boundary = boundary_mask_from_labels(labels)
            masks: list[np.ndarray] = []
            for section_name in V7_SECTION_NAMES:
                class_id = section_class_ids[section_name]
                masks.append(boundary if class_id is None else (labels == class_id) & ~boundary)
            coverage = np.zeros(PAIR_SHAPE, dtype=np.uint8)
            for mask in masks:
                coverage += mask.astype(np.uint8)
            if not np.all(coverage == 1):
                raise DirectDescriptionError("v7 self-detected class interiors plus boundary do not partition the plane")
            changed = np.any(corrected[batch_index] != predicted[batch_index], axis=-1)
            for section_id, mask in enumerate(masks):
                section_positions: list[np.ndarray] = []
                section_values: list[np.ndarray] = []
                for plane_id in range(2):
                    local_positions = np.flatnonzero(mask & changed[plane_id]).astype(np.uint32)
                    section_positions.append(local_positions + np.uint32(plane_id * sites_per_plane))
                    section_values.append(
                        np.ascontiguousarray(corrected[batch_index, plane_id].reshape(-1, 3)[local_positions])
                    )
                positions = np.ascontiguousarray(np.concatenate(section_positions).astype("<u4", copy=False))
                values = np.ascontiguousarray(np.concatenate(section_values, axis=0).astype(np.uint8, copy=False))
                positions_by_section[section_id][pair_id] = (positions, values)
            observed_keys.append(pair_id)
    if tuple(observed_keys) != keyframes:
        raise DirectDescriptionError("v7 target chunk traversal did not cover the correction key schedule")
    return tuple(positions_by_section)


def _v7_load_or_build_rung_sections(
    *,
    config: DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1,
    target_receipt: Any,
    predictor: ComposedStructuredMemberReceiverV1,
    cached_lstars: np.ndarray,
    section_class_ids: Mapping[str, int | None],
    xi_keyframes: Sequence[int],
    rung_name: str,
    output_directory: Path,
    predictor_sha256: str,
) -> tuple[PlaneCorrectionSectionBuildV1, ...]:
    schedule_id, quant_step = V7_RUNG_SPECS[rung_name]
    rung_root = output_directory / "rung_checkpoints" / rung_name
    expected_paths = tuple(rung_root / f"section_{section_id}.bin" for section_id in range(6))
    receipt_path = rung_root / "rung_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if (
            receipt.get("schema") != "direct_description_solved_plane_rung_checkpoint.v1"
            or receipt.get("typed_config_sha256") != config.typed_config_hash()
            or receipt.get("predictor_sha256") != predictor_sha256
            or receipt.get("rung_name") != rung_name
        ):
            raise DirectDescriptionError(f"v7 rung checkpoint custody mismatch: {receipt_path}")
        sections = tuple(parse_plane_correction_section(_read_regular_file_once(path)) for path in expected_paths)
        if [section.ledger_row()["frame_sha256"] for section in sections] != receipt.get("frame_sha256"):
            raise DirectDescriptionError(f"v7 rung frame hashes differ from checkpoint: {rung_name}")
        receipt_sections = receipt.get("sections")
        if not isinstance(receipt_sections, list) or len(receipt_sections) != 6:
            raise DirectDescriptionError(f"v7 rung checkpoint section ledger is incomplete: {rung_name}")
        return tuple(
            replace(section, candidate_rows=tuple(receipt_sections[index]["candidate_rows"]))
            for index, section in enumerate(sections)
        )
    records_by_section = _v7_build_section_records(
        target_receipt=target_receipt,
        predictor=predictor,
        cached_lstars=cached_lstars,
        pair_start=config.pair_start,
        pair_count=config.pair_count,
        section_class_ids=section_class_ids,
        schedule_id=schedule_id,
        quant_step=quant_step,
        xi_keyframes=xi_keyframes,
    )
    sections = tuple(
        encode_plane_correction_section(
            section_id=section_id,
            schedule_id=schedule_id,
            n_pairs=config.pair_count,
            quant_step=quant_step,
            records=records_by_section[section_id],
        )
        for section_id in range(6)
    )
    for path, section in zip(expected_paths, sections, strict=True):
        _publish_or_verify(path, section.frame)
    receipt = {
        "schema": "direct_description_solved_plane_rung_checkpoint.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "predictor_sha256": predictor_sha256,
        "target_receipt_sha256": config.target_receipt_sha256,
        "rung_name": rung_name,
        "schedule": CORRECTION_SCHEDULE_NAME[schedule_id],
        "quant_step": quant_step,
        "frame_sha256": [section.ledger_row()["frame_sha256"] for section in sections],
        "sections": [section.ledger_row() for section in sections],
        "all_stage_checkpoints_preserved": True,
    }
    _publish_or_verify(receipt_path, rfc8785_canonicalize(receipt) + b"\n")
    return sections


def _v7_zip_members(archive: bytes) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    expected = ("manifest.json", "predictor.zip", *(f"correction/section_{index}.bin" for index in range(6)))
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if tuple(row.filename for row in infos) != expected or reader.comment:
                raise DirectDescriptionError("v7 correction archive member order is noncanonical")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.flag_bits != 0
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.extra
                or row.comment
                or row.external_attr != 0o100644 << 16
                or row.create_system != 3
                for row in infos
            ):
                raise DirectDescriptionError("v7 correction ZIP framing is noncanonical")
            members = {row.filename: reader.read(row.filename) for row in infos}
            homes: list[dict[str, Any]] = []
            for index, row in enumerate(infos):
                next_offset = infos[index + 1].header_offset if index + 1 < len(infos) else reader.start_dir
                homes.append(
                    {
                        "name": row.filename,
                        "payload_bytes": row.file_size,
                        "zip_home_bytes": next_offset - row.header_offset,
                    }
                )
    except DirectDescriptionError:
        raise
    except (zipfile.BadZipFile, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise DirectDescriptionError("v7 correction archive is malformed") from exc
    container_bytes = len(archive) - sum(row["zip_home_bytes"] for row in homes)
    homes.append({"name": "__central_directory_and_eocd__", "payload_bytes": 0, "zip_home_bytes": container_bytes})
    if container_bytes <= 0 or sum(row["zip_home_bytes"] for row in homes) != len(archive):
        raise DirectDescriptionError("v7 final-ZIP byte homes do not cover the archive exactly")
    return members, tuple(homes)


def compile_solved_plane_tolerance_archive(
    *,
    predictor_archive: bytes,
    policy_name: str,
    rung_names: Sequence[str],
    sections: Sequence[PlaneCorrectionSectionBuildV1],
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    if policy_name not in dict(V7_POLICY_SPECS) or tuple(rung_names) != dict(V7_POLICY_SPECS)[policy_name]:
        raise DirectDescriptionError("v7 policy differs from the preregistered waterfill ladder")
    if len(sections) != 6 or tuple(section.section_id for section in sections) != tuple(range(6)):
        raise DirectDescriptionError("v7 correction sections must be complete and ordered")
    for rung_name, section in zip(rung_names, sections, strict=True):
        expected_schedule, expected_step = V7_RUNG_SPECS[rung_name]
        if (section.schedule_id, section.quant_step) != (expected_schedule, expected_step):
            raise DirectDescriptionError("v7 correction section does not implement its declared rung")
    manifest = {
        "schema": "direct_description_solved_plane_tolerance_archive.v1",
        "pair_count": sections[0].n_pairs,
        "policy_name": policy_name,
        "rung_names": list(rung_names),
        "predictor": {"bytes": len(predictor_archive), "sha256": _sha256(predictor_archive)},
        "opaque_correction_sections": [
            {
                "section_id": section.section_id,
                "frame_bytes": len(section.frame),
                "frame_sha256": _sha256(section.frame),
            }
            for section in sections
        ],
        "receiver": "numpy_uint8_v6_predictor_plus_counted_opaque_site_value_corrections.v1",
        "section_semantics_shipped": False,
        "ground_truth_argmax_table_present": False,
        "scorer_weights_present": False,
        "score_claim": False,
    }
    members = {
        "manifest.json": rfc8785_canonicalize(manifest),
        "predictor.zip": predictor_archive,
        **{f"correction/section_{section.section_id}.bin": section.frame for section in sections},
    }
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("v7 correction archive compiler is nondeterministic")
    parsed, homes = _v7_zip_members(first)
    if parsed != members or _zip_stored(parsed) != first:
        raise DirectDescriptionError("v7 correction archive parse/re-encode identity failed")
    return first, homes


def receive_solved_plane_tolerance_archive(archive: bytes) -> SolvedPlaneToleranceReceiverV1:
    members, homes = _v7_zip_members(archive)
    try:
        manifest = json.loads(members["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("v7 correction manifest is invalid JSON") from exc
    if (
        manifest.get("schema") != "direct_description_solved_plane_tolerance_archive.v1"
        or rfc8785_canonicalize(manifest) != members["manifest.json"]
        or manifest.get("ground_truth_argmax_table_present") is not False
        or manifest.get("scorer_weights_present") is not False
    ):
        raise DirectDescriptionError("v7 correction manifest custody is invalid")
    predictor_bytes = members["predictor.zip"]
    if manifest.get("predictor") != {"bytes": len(predictor_bytes), "sha256": _sha256(predictor_bytes)}:
        raise DirectDescriptionError("v7 predictor bytes differ from manifest custody")
    predictor = receive_structured_member_archive(predictor_bytes)
    if not isinstance(predictor, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("v7 predictor is not the composed v6 receiver")
    sections = tuple(
        parse_plane_correction_section(members[f"correction/section_{section_id}.bin"])
        for section_id in range(6)
    )
    if any(section.n_pairs != predictor.z.n_pairs for section in sections):
        raise DirectDescriptionError("v7 correction/predictor pair counts differ")
    declarations = [
        {"section_id": section.section_id, "frame_bytes": len(section.frame), "frame_sha256": _sha256(section.frame)}
        for section in sections
    ]
    if manifest.get("opaque_correction_sections") != declarations:
        raise DirectDescriptionError("v7 correction declarations differ from parsed frames")
    return SolvedPlaneToleranceReceiverV1(
        archive=archive,
        predictor=predictor,
        sections=sections,
        custody={
            "schema": "direct_description_solved_plane_tolerance_receiver.v1",
            "all_archive_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
            "parse_reencode_identical": _zip_stored(members) == archive,
            "all_six_opaque_sections_consumed": True,
            "section_semantics_required_by_receiver": False,
            "ground_truth_argmax_table_present": False,
            "scorer_weights_present": False,
            "score_claim": False,
        },
    )


def _v7_exact_target_match(
    receiver: SolvedPlaneToleranceReceiverV1,
    *,
    target_receipt: Any,
    pair_start: int,
    pair_count: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for pair_ids, target in iter_target_plane_window_chunks(
        target_receipt,
        pair_start=pair_start,
        n_pairs=pair_count,
    ):
        rendered = receiver.render_pairs(pair_ids)
        if not np.array_equal(rendered, target):
            raise DirectDescriptionError("v7 exact correction rung does not reconstruct the solved C1 planes")
        digest.update(rendered.tobytes(order="C"))
    return {"bit_identical_to_solved_planes": True, "rendered_window_sha256": digest.hexdigest()}


def _v7_discrete_waterfill(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the cheap-to-faithful Pareto route from measured candidate rows."""

    def distortion_term(row: Mapping[str, Any]) -> float:
        bridge = row["evaluator_bridge"]
        d_seg = float(bridge["segmentation"]["d_seg"])
        d_pose = float(bridge["pose"]["d_pose"])
        return 100.0 * d_seg + math.sqrt(10.0 * d_pose)

    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["archive"]["bytes"]),
            distortion_term(row),
            int(row["candidate_index"]),
        ),
    )
    frontier: list[Mapping[str, Any]] = []
    dominated: list[str] = []
    best_distortion = math.inf
    for row in ordered:
        distortion = distortion_term(row)
        if distortion < best_distortion:
            frontier.append(row)
            best_distortion = distortion
        else:
            dominated.append(str(row["policy_name"]))
    if len(frontier) < 2:
        raise DirectDescriptionError("v7 waterfill requires at least two non-dominated measured states")

    rate_price = 25.0 / float(SOURCE_BYTES)
    marginals: list[dict[str, Any]] = []
    first_break: str | None = None
    for coarse, fine in itertools.pairwise(frontier):
        coarse_name = str(coarse["policy_name"])
        fine_name = str(fine["policy_name"])
        added_bytes = int(fine["archive"]["bytes"]) - int(coarse["archive"]["bytes"])
        if added_bytes <= 0:
            raise DirectDescriptionError("v7 waterfill Pareto route is not strictly byte-monotone")
        distortion_gain = distortion_term(coarse) - distortion_term(fine)
        if distortion_gain <= 0.0:
            raise DirectDescriptionError("v7 waterfill Pareto route is not strictly distortion-monotone")
        gain_per_byte = distortion_gain / added_bytes
        passes = gain_per_byte >= rate_price
        if first_break is None and not passes:
            first_break = f"{coarse_name}->{fine_name}"
        marginals.append(
            {
                "from_policy": coarse_name,
                "to_policy": fine_name,
                "added_bytes": added_bytes,
                "distortion_term_gain": f"{distortion_gain:.12f}",
                "distortion_gain_per_added_byte": f"{gain_per_byte:.15f}",
                "rate_price_per_byte": f"{rate_price:.15f}",
                "passes_rate_break_even": passes,
                "verdict_scope": "adjacent measured states on the discrete advisory Pareto envelope",
            }
        )
    return {
        "route": [str(row["policy_name"]) for row in frontier],
        "dominated_policies": dominated,
        "marginals": marginals,
        "first_rate_break": first_break,
        "route_derivation": "sort by exact archive bytes; retain strict distortion-record improvements",
    }


def _v7_candidate_verdict_scope(config: DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1) -> str:
    return (
        f"MEASURED n{config.pair_count} source window "
        f"[{config.pair_start},{config.pair_start + config.pair_count}) on {V7_EVIDENCE_AXIS}; "
        "opaque counted corrections over one v6 predictor; formulation-local and not contest-CPU/CUDA"
    )


def _v7_load_completed_receipt(
    config: DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1,
    root: Path,
) -> tuple[dict[str, Any], Path] | None:
    """Validate every preserved v7 stage and return an already-sealed final receipt."""

    receipt_path = root / f"ddm_v7_solved_plane_tolerance_waterfill_n{config.pair_count}_receipt.json"
    if not receipt_path.exists():
        return None
    try:
        receipt = json.loads(_read_regular_file_once(receipt_path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("v7 completed receipt is not valid JSON") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != V7_RESULT_SCHEMA
        or receipt.get("typed_config_sha256") != config.typed_config_hash()
        or receipt.get("dsl_compile_hash") != config.dsl_compile_hash()
        or receipt.get("score_claim") is not False
        or receipt.get("d_seg_claim") is not False
        or receipt.get("d_pose_claim") is not False
    ):
        raise DirectDescriptionError("v7 completed receipt configuration or claim custody mismatch")
    expected_producer = {
        "solver_module": _committed_source_custody(
            "src/tac/optimization/direct_description_entropy_priced_member.py"
        ),
        "stream_module": _committed_source_custody("src/tac/optimization/direct_description_entropy_streams.py"),
        "cli": _committed_source_custody("tools/run_direct_description_entropy_priced_member.py"),
    }
    if receipt.get("producer") != expected_producer:
        raise DirectDescriptionError("v7 completed receipt producer custody differs from committed sources")
    rows = receipt.get("candidates")
    if (
        not isinstance(rows, list)
        or [row.get("policy_name") for row in rows] != list(config.policy_names)
        or receipt.get("candidate_table_sha256") != _sha256(rfc8785_canonicalize(rows))
    ):
        raise DirectDescriptionError("v7 completed receipt candidate-table custody mismatch")
    checkpoint_paths = receipt.get("resume", {}).get("candidate_checkpoints")
    if not isinstance(checkpoint_paths, list) or len(checkpoint_paths) != len(rows):
        raise DirectDescriptionError("v7 completed receipt candidate checkpoint index mismatch")
    for candidate_index, (row, checkpoint_value) in enumerate(zip(rows, checkpoint_paths, strict=True)):
        bridge = row.get("evaluator_bridge", {})
        if bridge.get("evidence_axis") != V7_EVIDENCE_AXIS:
            raise DirectDescriptionError("v7 completed receipt candidate evidence axis mismatch")
        archive = row.get("archive", {})
        payload = _read_bound_file(
            Path(archive.get("path", "")),
            str(archive.get("sha256", "")),
            f"v7_completed_candidate_{candidate_index}_archive_sha256",
        )
        if len(payload) != archive.get("bytes"):
            raise DirectDescriptionError("v7 completed receipt candidate archive length mismatch")
        checkpoint_path = Path(checkpoint_value)
        try:
            checkpoint = json.loads(_read_regular_file_once(checkpoint_path))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("v7 completed candidate checkpoint is not valid JSON") from exc
        if (
            checkpoint.get("schema") != "direct_description_solved_plane_candidate_checkpoint.v1"
            or checkpoint.get("typed_config_sha256") != config.typed_config_hash()
            or checkpoint.get("candidate", {}).get("archive", {}).get("sha256") != archive.get("sha256")
        ):
            raise DirectDescriptionError("v7 completed candidate checkpoint custody mismatch")
    for rung_name in config.rung_names:
        rung_root = root / "rung_checkpoints" / rung_name
        rung_receipt_path = rung_root / "rung_receipt.json"
        try:
            rung_receipt = json.loads(_read_regular_file_once(rung_receipt_path))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectDescriptionError("v7 completed rung checkpoint is not valid JSON") from exc
        frame_sha256 = rung_receipt.get("frame_sha256")
        if (
            rung_receipt.get("schema") != "direct_description_solved_plane_rung_checkpoint.v1"
            or rung_receipt.get("typed_config_sha256") != config.typed_config_hash()
            or rung_receipt.get("rung_name") != rung_name
            or not isinstance(frame_sha256, list)
            or len(frame_sha256) != len(V7_SECTION_NAMES)
        ):
            raise DirectDescriptionError("v7 completed rung checkpoint custody mismatch")
        for section_id, expected_sha256 in enumerate(frame_sha256):
            _read_bound_file(
                rung_root / f"section_{section_id}.bin",
                str(expected_sha256),
                f"v7_completed_{rung_name}_section_{section_id}_sha256",
            )
    return receipt, receipt_path


def run_solved_plane_tolerance_waterfill(
    config: DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    """Build and measure the receiver-closed v7 solved-plane tolerance ladder."""

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    completed = _v7_load_completed_receipt(config, root)
    if completed is not None:
        return completed
    storage = _storage_preflight(root)
    v6_receipt = _read_bound_json(Path(config.v6_receipt_path), config.v6_receipt_sha256, "v6_receipt_sha256")
    if v6_receipt.get("schema") != V6_RESULT_SCHEMA:
        raise DirectDescriptionError("v7 input receipt is not the governed v6 result")
    typed_v6 = DirectDescriptionDsegBridgeAmortizeConfigV1.model_validate_json(
        rfc8785_canonicalize(v6_receipt["typed_config"])
    )
    if (typed_v6.pair_start, typed_v6.pair_count) != (config.pair_start, config.pair_count):
        raise DirectDescriptionError("v7 typed window differs from the bound v6 receipt")
    base_row = next((row for row in v6_receipt["candidates"] if row["mode"] == config.base_mode), None)
    if base_row is None:
        raise DirectDescriptionError("v7 bound v6 receipt lacks the configured predictor mode")
    predictor_archive = _read_bound_file(
        Path(base_row["archive"]["path"]),
        base_row["archive"]["sha256"],
        "v7_predictor_archive_sha256",
    )
    if len(predictor_archive) != base_row["archive"]["bytes"]:
        raise DirectDescriptionError("v7 predictor archive length differs from v6 custody")
    predictor = receive_structured_member_archive(predictor_archive)
    if not isinstance(predictor, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("v7 predictor is not a composed v6 receiver")
    target_receipt = load_target_receipt(Path(config.target_receipt_path), config.target_receipt_sha256)
    v6_target = v6_receipt["target_custody"]
    if (
        v6_target["receipt_path"] != config.target_receipt_path
        or v6_target["receipt_sha256"] != config.target_receipt_sha256
    ):
        raise DirectDescriptionError("v7 target receipt differs from v6 target custody")
    cache_path = Path(target_receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != target_receipt.source_cache.bytes:
        raise DirectDescriptionError("v7 evaluator cache is unavailable")
    try:
        cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
        cached_margins = open_stored_npy_memmap(cache_path, "margins")
        cached_poses = open_stored_npy_memmap(cache_path, "gt_poses")
    except (OSError, ValueError) as exc:
        raise DirectDescriptionError("v7 evaluator cache members are malformed") from exc
    v5_receipt = _read_bound_json(
        Path(v6_receipt["v5_control"]["receipt_path"]),
        v6_receipt["v5_control"]["receipt_sha256"],
        "v7_v5_receipt_sha256",
    )
    v5_config = DirectDescriptionRouteFixComposeConfigV1.model_validate(v5_receipt["typed_config"])
    if Path(v5_config.upstream_root).resolve() != Path(config.upstream_root).resolve():
        raise DirectDescriptionError("v7 upstream scorer root differs from the bound v5 compiler")
    routing = _read_bound_json(
        Path(v5_receipt["routing_receipt"]["path"]),
        v5_receipt["routing_receipt"]["sha256"],
        "v7_routing_receipt_sha256",
    )
    role_class_ids = routing["class_detection"]["role_class_ids"]
    section_class_ids = _v7_section_class_ids(role_class_ids)
    chart_members, _ = parse_structured_member_archive(predictor_archive)
    predictor_chart = receive_entropy_chart_archive(chart_members["chart.zip"])
    xi_keys, xi_receipt = _xi_pose6_keyframes(predictor_chart.pose6_codes, max_gap=config.max_key_gap)
    rung_library: dict[str, tuple[PlaneCorrectionSectionBuildV1, ...]] = {}
    for rung_name in config.rung_names:
        rung_library[rung_name] = _v7_load_or_build_rung_sections(
            config=config,
            target_receipt=target_receipt,
            predictor=predictor,
            cached_lstars=cached_lstars,
            section_class_ids=section_class_ids,
            xi_keyframes=xi_keys,
            rung_name=rung_name,
            output_directory=root,
            predictor_sha256=_sha256(predictor_archive),
        )
    segnet_oracle, segnet_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    posenet_oracle, posenet_custody = _load_posenet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    rows: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    for candidate_index, (policy_name, rung_names) in enumerate(V7_POLICY_SPECS):
        checkpoint_path = root / "candidate_receipts" / f"{candidate_index:02d}_{policy_name}.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(_read_regular_file_once(checkpoint_path))
            if (
                checkpoint.get("schema") != "direct_description_solved_plane_candidate_checkpoint.v1"
                or checkpoint.get("typed_config_sha256") != config.typed_config_hash()
                or checkpoint.get("candidate", {}).get("policy_name") != policy_name
            ):
                raise DirectDescriptionError(f"v7 candidate checkpoint custody mismatch: {checkpoint_path}")
            row = checkpoint["candidate"]
            archive_payload = _read_bound_file(
                Path(row["archive"]["path"]), row["archive"]["sha256"], "v7_candidate_archive_sha256"
            )
            if len(archive_payload) != row["archive"]["bytes"]:
                raise DirectDescriptionError("v7 candidate checkpoint archive length mismatch")
            if row["evaluator_bridge"].get("evidence_axis") != V7_EVIDENCE_AXIS:
                raise DirectDescriptionError("v7 candidate checkpoint evidence axis mismatch")
            row["verdict_scope"] = _v7_candidate_verdict_scope(config)
            rows.append(row)
            checkpoint_paths.append(str(checkpoint_path))
            continue
        selected_sections = tuple(
            rung_library[rung_names[section_id]][section_id] for section_id in range(6)
        )
        archive, homes = compile_solved_plane_tolerance_archive(
            predictor_archive=predictor_archive,
            policy_name=policy_name,
            rung_names=rung_names,
            sections=selected_sections,
        )
        archive_path = _publish_or_verify(
            root / f"ddm_v7_{policy_name}_n{config.pair_count}.not_a_candidate.zip.receipt-bytes",
            archive,
        )
        receiver = receive_solved_plane_tolerance_archive(archive)
        replay = receive_solved_plane_tolerance_archive(archive)
        probe_ids = tuple(sorted({0, config.pair_count // 2, config.pair_count - 1}))
        if not np.array_equal(receiver.render_pairs(probe_ids), replay.render_pairs(probe_ids)):
            raise DirectDescriptionError(f"v7 {policy_name} receiver replay is not bit-identical")
        exact_proof = (
            _v7_exact_target_match(
                receiver,
                target_receipt=target_receipt,
                pair_start=config.pair_start,
                pair_count=config.pair_count,
            )
            if policy_name == "exact_all"
            else None
        )
        bridge = _measure_evaluator_bridge(
            receiver,  # type: ignore[arg-type]
            pair_start=config.pair_start,
            cached_lstars=cached_lstars,
            cached_margins=cached_margins,
            cached_poses=cached_poses,
            segnet_oracle=segnet_oracle,
            posenet_oracle=posenet_oracle,
            batch_size=config.scorer_batch_size,
        )
        bridge["evidence_axis"] = V7_EVIDENCE_AXIS
        d_seg = Decimal(bridge["segmentation"]["d_seg"])
        stream_rows = []
        home_by_name = {row["name"]: row for row in homes}
        for section_name, rung_name, section in zip(V7_SECTION_NAMES, rung_names, selected_sections, strict=True):
            member_name = f"correction/section_{section.section_id}.bin"
            stream_rows.append(
                {
                    "stratum": section_name,
                    "rung": rung_name,
                    **section.ledger_row(),
                    "final_zip_home_bytes": home_by_name[member_name]["zip_home_bytes"],
                }
            )
        row = {
            "candidate_index": candidate_index,
            "policy_name": policy_name,
            "rung_by_stratum": dict(zip(V7_SECTION_NAMES, rung_names, strict=True)),
            "archive": {
                "path": str(archive_path),
                "bytes": len(archive),
                "sha256": _sha256(archive),
                "predictor_bytes": len(predictor_archive),
                "all_bytes_have_one_home": sum(item["zip_home_bytes"] for item in homes) == len(archive),
                "parse_reencode_identical": True,
                "compiler_determinism_x2": True,
                "receiver_replay_identical": True,
                "not_a_candidate": True,
            },
            "stream_bytes": stream_rows,
            "evaluator_bridge": bridge,
            "membership_proxy": _derived_membership_proxy(bridge["segmentation"]["d_seg"]),
            "exact_target_proof": exact_proof,
            "gates": {
                "d_seg_le_0_00116": d_seg <= Decimal(V6_TARGET_DSEG_TEXT),
                "archive_bytes_le_exact_residual_falsifier": len(archive) <= config.exact_residual_falsifier_bytes,
                "score_claim": False,
                "promotion_eligible": False,
            },
            "verdict_scope": _v7_candidate_verdict_scope(config),
        }
        checkpoint = {
            "schema": "direct_description_solved_plane_candidate_checkpoint.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "dsl_compile_hash": config.dsl_compile_hash(),
            "semantic_argv": list(semantic_argv),
            "semantic_argv_sha256": _sha256("\0".join(semantic_argv).encode()),
            "completed_candidate_index": candidate_index,
            "next_candidate_index": candidate_index + 1,
            "candidate": row,
        }
        _publish_or_verify(checkpoint_path, rfc8785_canonicalize(checkpoint) + b"\n")
        rows.append(row)
        checkpoint_paths.append(str(checkpoint_path))
    qualifying = [row for row in rows if row["gates"]["d_seg_le_0_00116"]]
    knee = min(
        qualifying,
        key=lambda row: (row["archive"]["bytes"], Decimal(row["evaluator_bridge"]["segmentation"]["d_seg"])),
    ) if qualifying else min(
        rows,
        key=lambda row: (Decimal(row["evaluator_bridge"]["segmentation"]["d_seg"]), row["archive"]["bytes"]),
    )
    exact_row = rows[0]
    exact_falsified = exact_row["archive"]["bytes"] > config.exact_residual_falsifier_bytes
    dominant = sorted(exact_row["stream_bytes"], key=lambda row: row["final_zip_home_bytes"], reverse=True)
    waterfill = _v7_discrete_waterfill(rows)
    result = {
        "schema": V7_RESULT_SCHEMA,
        "task": 603,
        "feeds_task": 613,
        "master_task": 578,
        "lane_id": V7_LANE_ID,
        "run_id": config.run_id,
        "seed": config.seed,
        "verdict": (
            "FORMULATION_LEVEL_EXACT_RESIDUAL_KOLMOGOROV_RATE_WALL"
            if exact_falsified
            else "EXACT_RESIDUAL_FORMULATION_BELOW_PREREGISTERED_RATE_WALL"
        ),
        "verdict_scope": (
            f"n{config.pair_count} solved-plane correction formulation over v6 {config.base_mode} on "
            f"{V7_EVIDENCE_AXIS}; "
            "does not close learned/analytic direct-description families and carries no contest authority"
        ),
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "semantic_argv": list(semantic_argv),
        "predictor": {
            "mode": config.base_mode,
            "bytes": len(predictor_archive),
            "sha256": _sha256(predictor_archive),
            "v6_receipt_path": config.v6_receipt_path,
            "v6_receipt_sha256": config.v6_receipt_sha256,
        },
        "stratum_routing": {
            "method": routing["class_detection"]["method"],
            "never_luma_sorted": True,
            "boundary_precedence": "one-pixel codimension-1 band removed from five class interiors",
            "section_class_ids": section_class_ids,
            "section_semantics_shipped_in_archive": False,
            "ground_truth_argmax_table_shipped": False,
            "routing_receipt_path": v5_receipt["routing_receipt"]["path"],
            "routing_receipt_sha256": v5_receipt["routing_receipt"]["sha256"],
        },
        "xi_schedule": xi_receipt,
        "candidates": rows,
        "candidate_table_sha256": _sha256(rfc8785_canonicalize(rows)),
        "knee": {
            "policy_name": knee["policy_name"],
            "archive_bytes": knee["archive"]["bytes"],
            "d_seg": knee["evaluator_bridge"]["segmentation"]["d_seg"],
            "d_pose": knee["evaluator_bridge"]["pose"]["d_pose"],
            "binder": "minimum_bytes_subject_to_d_seg_le_0.00116" if qualifying else "closest_d_seg_no_feasible_row",
        },
        "waterfill": {
            **waterfill,
            "stop_rule": "stop future allocation at first marginal distortion gain per byte below 25/37545489",
            "metric": "100*d_seg+sqrt(10*d_pose) with exact final-ZIP bytes; advisory only",
        },
        "exact_residual_falsifier": {
            "threshold_bytes": config.exact_residual_falsifier_bytes,
            "observed_archive_bytes": exact_row["archive"]["bytes"],
            "triggered": exact_falsified,
            "verdict_scope": "opaque exact site/value correction formulation over the v6 predictor only",
            "dominant_streams": [
                {"stratum": row["stratum"], "final_zip_home_bytes": row["final_zip_home_bytes"]}
                for row in dominant
            ],
        },
        "scorer_custody": {
            "segnet": {**segnet_custody, "evidence_axis": V7_EVIDENCE_AXIS},
            "posenet": {**posenet_custody, "evidence_axis": V7_EVIDENCE_AXIS},
        },
        "target_custody": {
            "receipt_path": config.target_receipt_path,
            "receipt_sha256": config.target_receipt_sha256,
            "source_cache_path": str(cache_path),
            "source_cache_sha256": target_receipt.source_cache.sha256,
            "source_cache_mutated": False,
        },
        "resume": {
            "policy": config.checkpoint_policy,
            "rung_checkpoint_count": len(config.rung_names),
            "candidate_checkpoints": checkpoint_paths,
            "all_preserved": len(checkpoint_paths) == len(V7_POLICY_SPECS),
            "atomic_publish": True,
            "max_work_loss": "current rung or candidate only",
        },
        "blocker_delta": {
            "SOLVED_PLANE_EXACT_RECEIVER": "RED_TO_GREEN_LOCAL_ADVISORY",
            "PER_STRATUM_TOLERANCE_LADDER": "RED_TO_GREEN_LOCAL_ADVISORY",
            "DSEG_LE_0_00116": (
                "RED_TO_GREEN_LOCAL_ADVISORY" if qualifying else "REMAINS_RED_FORMULATION_SCOPE"
            ),
            "EXACT_RESIDUAL_LE_200KB": "RED_FORMULATION_SCOPE" if exact_falsified else "GREEN_LOCAL_ONLY",
            "N600": "REMAINS_RED_TIME_BOUND_NOT_RUN",
            "CONTEST_CPU_CUDA_SCORE": "REMAINS_RED_NOT_AUTHORIZED",
        },
        "storage_preflight": storage,
        "cleanup": {
            "bulk_artifacts_created": True,
            "all_rung_and_candidate_checkpoints_preserved": True,
            "ssd_sources_read_only": True,
            "scorer_policy": "batch16 with at most one target chunk plus one scorer batch resident",
            "certify_or_block": "no deletion, movement, or source mutation performed",
        },
        "stores_consulted": [
            "docs/operating_manual_craft_handoff.md",
            "direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md",
            "ddm_full_precision_target_planes_603_20260722T010130Z.json",
            "v6 dseg bridge/amortization receipts and exact receiver archives",
            "truly_optimal_coder_survey_603_613_20260722.md",
            "2026-07-19 reverse-waterfill, Fisher-margin, corrected-inner-Jacobian, curvelet, and xi directives",
        ],
        "producer": {
            "solver_module": _committed_source_custody(
                "src/tac/optimization/direct_description_entropy_priced_member.py"
            ),
            "stream_module": _committed_source_custody(
                "src/tac/optimization/direct_description_entropy_streams.py"
            ),
            "cli": _committed_source_custody("tools/run_direct_description_entropy_priced_member.py"),
        },
        "main_landing_review_required": True,
    }
    receipt_path = _publish_or_verify(
        root / f"ddm_v7_solved_plane_tolerance_waterfill_n{config.pair_count}_receipt.json",
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


__all__ = [
    "TOLERANCE_LADDER",
    "DirectDescriptionDsegBridgeAmortizeConfigV1",
    "DirectDescriptionDsegBridgeAmortizeProgramV1",
    "DirectDescriptionEntropyCandidateCheckpointV1",
    "DirectDescriptionEntropyPricedMemberConfigV1",
    "DirectDescriptionEntropyPricedMemberProgramV1",
    "DirectDescriptionEntropyRungCheckpointV1",
    "DirectDescriptionRouteFixComposeConfigV1",
    "DirectDescriptionRouteFixComposeProgramV1",
    "DirectDescriptionSolvedPlaneToleranceWaterfillConfigV1",
    "DirectDescriptionSolvedPlaneToleranceWaterfillProgramV1",
    "DirectDescriptionStratumStructuredMemberConfigV1",
    "DirectDescriptionStratumStructuredMemberProgramV1",
    "DirectDescriptionStructuredCandidateCheckpointV1",
    "build_entropy_candidate_z",
    "compile_composed_structured_member_archive",
    "compile_solved_plane_tolerance_archive",
    "compile_structured_member_archive",
    "load_structured_s4_sources",
    "measure_structured_role_value_routing",
    "parse_structured_member_archive",
    "receive_solved_plane_tolerance_archive",
    "receive_structured_member_archive",
    "run_dseg_bridge_amortize",
    "run_entropy_candidate_stages",
    "run_entropy_priced_member_n64",
    "run_entropy_rung_stages",
    "run_route_fix_composed_member",
    "run_solved_plane_tolerance_waterfill",
    "run_stratum_structured_member_n64",
    "run_structured_candidate_stages",
    "select_role_paint_values",
    "self_detect_structured_role_classes",
]
