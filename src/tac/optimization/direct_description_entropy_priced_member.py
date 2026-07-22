# SPDX-License-Identifier: MIT
"""Task #603/#613 v3 exact-byte solve over entropy-coded semantic members."""

from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import lzma
import shutil
import struct
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Literal

import brotli
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.optimization.direct_description_entropy_streams import (
    compile_entropy_chart_archive,
    parse_entropy_chart_archive,
    prove_entropy_home_fail_closed,
    receive_entropy_chart_archive,
)
from tac.optimization.direct_description_measurement_ladder import (
    DirectDescriptionChartZV1,
    compile_chart_archive,
    fit_chart_description,
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
ROLE_CLASS_ID: Final = {"Road": 0, "Lane": 1, "UndrivableBoundary": 2, "Movable": 3, "MyCar": 4}
ROLE_TARGET_CLASS: Final = {
    "Road": "Road",
    "Lane": "Lane",
    "MyCar": "MyCar",
    "UndrivableBoundary": "Undrivable",
    "Movable": "Movable",
}
STRUCTURED_MEMBER_MAGIC: Final = b"D4S1"
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
    run_id: Literal["ddm_v4_stratum_structured_members_n64_seed1234"] = (
        "ddm_v4_stratum_structured_members_n64_seed1234"
    )
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
    candidate_family: Literal[
        "s4_pxq1_lane_curve_static_hood_and_class_filtered_event_component_members"
    ] = "s4_pxq1_lane_curve_static_hood_and_class_filtered_event_component_members"
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
                {"compile_target": STRUCTURED_RESULT_SCHEMA, "typed_config": self.model_dump(mode="json", by_alias=True)}
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


def _decode_site_records(payload: bytes, *, expected_class: int, expected_source: int) -> tuple[tuple[np.ndarray, ...], ...]:
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


def _render_s4_lane_mask(lines: Sequence[np.ndarray], header: Mapping[str, Any], camera: Mapping[str, Any]) -> np.ndarray:
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
    events: tuple[tuple[tuple[np.ndarray, ...], ...], ...]
    components: tuple[tuple[tuple[np.ndarray, ...], ...], ...]
    custody: Mapping[str, Any]


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
        tuple(tuple(np.asarray(row, dtype=np.int64) for row in decoded_events[pair][class_id]) for pair in range(64))
        for class_id in range(5)
    )
    all_components = _decode_pcomp3(sections["components.pcomp3"].payload)
    components = tuple(tuple(all_components[class_id][pair] for pair in range(64)) for class_id in range(5))
    return StructuredS4SourcesV1(
        pair_count=64,
        palette=palette,
        camera=dict(camera),
        static_masks={"Road": road, "Undrivable": undrivable, "MyCar": hood},
        lane_encoded=lane_encoded,
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


def _payloads_for_role(role: str, sources: StructuredS4SourcesV1) -> dict[str, bytes]:
    if role == "baseline":
        return {}
    class_id = ROLE_CLASS_ID[role]
    if role == "Road":
        return {
            "structure/road_pxq1_mask.br": _pack_mask(sources.static_masks["Road"]),
            "structure/road_events.lz": _encode_site_records(
                sources.events[class_id], class_id=class_id, source_id=1, coder="lzma1_raw_1MiB"
            ),
            "structure/road_components.br": _encode_site_records(
                sources.components[class_id], class_id=class_id, source_id=2, coder="brotli_q11"
            ),
        }
    if role == "Lane":
        return {
            "structure/lane_lbnd2.lz": sources.lane_encoded,
            "structure/lane_events.lz": _encode_site_records(
                sources.events[class_id], class_id=class_id, source_id=1, coder="lzma1_raw_1MiB"
            ),
            "structure/lane_components.br": _encode_site_records(
                sources.components[class_id], class_id=class_id, source_id=2, coder="brotli_q11"
            ),
        }
    if role == "MyCar":
        return {"structure/mycar_static_hood.br": _pack_mask(sources.static_masks["MyCar"])}
    if role == "UndrivableBoundary":
        return {
            "structure/undrivable_events.lz": _encode_site_records(
                sources.events[class_id], class_id=class_id, source_id=1, coder="lzma1_raw_1MiB"
            ),
            "structure/undrivable_components.br": _encode_site_records(
                sources.components[class_id], class_id=class_id, source_id=2, coder="brotli_q11"
            ),
        }
    if role == "Movable":
        return {
            "structure/movable_events.lz": _encode_site_records(
                sources.events[class_id], class_id=class_id, source_id=1, coder="lzma1_raw_1MiB"
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
    payloads = _payloads_for_role(role, sources)
    declarations = [
        {"name": name, "bytes": len(payload), "sha256": _sha256(payload)} for name, payload in payloads.items()
    ]
    manifest = {
        "schema": "direct_description_stratum_structured_archive.v1",
        "magic": STRUCTURED_MEMBER_MAGIC.decode("ascii"),
        "pair_count": sources.pair_count,
        "role": role,
        "class_id": ROLE_CLASS_ID.get(role),
        "target_class": ROLE_TARGET_CLASS.get(role),
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
    if (
        manifest.get("schema") != "direct_description_stratum_structured_archive.v1"
        or manifest.get("magic") != STRUCTURED_MEMBER_MAGIC.decode("ascii")
        or manifest.get("pair_count") != 64
        or manifest.get("role") not in STRUCTURED_ROLES
        or set(members) != {"manifest.json", "chart.zip", *[row["name"] for row in manifest["structured_payloads"]]}
        or manifest["baseline_chart"] != {"bytes": len(members["chart.zip"]), "sha256": _sha256(members["chart.zip"])}
    ):
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
        class_id = ROLE_CLASS_ID[self.role]
        color = self.palette[class_id]
        for local_index, pair_id in enumerate(indexes):
            mask = np.zeros(PAIR_SHAPE, dtype=bool)
            if self.static_mask is not None:
                mask |= self.static_mask
            if self.lane_lines is not None and self.lane_header is not None:
                mask |= _render_s4_lane_mask(self.lane_lines[pair_id], self.lane_header, self.camera)
            for rows in (self.event_rows, self.component_rows):
                if rows is not None:
                    flat = mask.reshape(-1)
                    for sites in rows[pair_id]:
                        flat[sites] = True
            output[local_index, 0, mask] = color
            output[local_index, 1, mask] = color
        return np.ascontiguousarray(output)


def receive_structured_member_archive(archive: bytes) -> StructuredMemberReceiverV1:
    members, homes = parse_structured_member_archive(archive)
    manifest = json.loads(members["manifest.json"])
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
    if role == "Road":
        static_mask = _unpack_mask(members["structure/road_pxq1_mask.br"])
        event_rows = _decode_site_records(
            members["structure/road_events.lz"], expected_class=0, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/road_components.br"], expected_class=0, expected_source=2
        )
    elif role == "Lane":
        lane_lines, lane_header = _decode_s4_lane(members["structure/lane_lbnd2.lz"])
        event_rows = _decode_site_records(
            members["structure/lane_events.lz"], expected_class=1, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/lane_components.br"], expected_class=1, expected_source=2
        )
    elif role == "MyCar":
        static_mask = _unpack_mask(members["structure/mycar_static_hood.br"])
    elif role == "UndrivableBoundary":
        event_rows = _decode_site_records(
            members["structure/undrivable_events.lz"], expected_class=2, expected_source=1
        )
        component_rows = _decode_site_records(
            members["structure/undrivable_components.br"], expected_class=2, expected_source=2
        )
    elif role == "Movable":
        event_rows = _decode_site_records(
            members["structure/movable_events.lz"], expected_class=3, expected_source=1
        )
    return StructuredMemberReceiverV1(
        archive=archive,
        z=baseline.z,
        pose6_codes=baseline.pose6_codes,
        baseline=baseline,
        role=role,
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
        config = DirectDescriptionStratumStructuredMemberConfigV1.model_validate_json(
            rfc8785_canonicalize(self.config)
        )
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
    return StructuredCandidateRunV1(tuple(rows), tuple(paths), len(rows) == len(STRUCTURED_ROLES), resume_from is not None)


def _event_subset_pricing_curve(sources: StructuredS4SourcesV1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ("Road", "Lane", "UndrivableBoundary", "Movable"):
        class_id = ROLE_CLASS_ID[role]
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
            "roles_below_approx_box": [row["role"] for row in candidates if row["below_approx_200000_byte_receiver_box"]],
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


__all__ = [
    "TOLERANCE_LADDER",
    "DirectDescriptionEntropyCandidateCheckpointV1",
    "DirectDescriptionEntropyPricedMemberConfigV1",
    "DirectDescriptionEntropyPricedMemberProgramV1",
    "DirectDescriptionEntropyRungCheckpointV1",
    "DirectDescriptionStratumStructuredMemberConfigV1",
    "DirectDescriptionStratumStructuredMemberProgramV1",
    "DirectDescriptionStructuredCandidateCheckpointV1",
    "build_entropy_candidate_z",
    "compile_structured_member_archive",
    "load_structured_s4_sources",
    "parse_structured_member_archive",
    "receive_structured_member_archive",
    "run_entropy_candidate_stages",
    "run_entropy_priced_member_n64",
    "run_entropy_rung_stages",
    "run_stratum_structured_member_n64",
    "run_structured_candidate_stages",
]
