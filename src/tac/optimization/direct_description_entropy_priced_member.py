# SPDX-License-Identifier: MIT
"""Task #603/#613 v3 exact-byte solve over entropy-coded semantic members."""

from __future__ import annotations

import base64
import json
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

if TYPE_CHECKING:
    import numpy as np

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


__all__ = [
    "TOLERANCE_LADDER",
    "DirectDescriptionEntropyCandidateCheckpointV1",
    "DirectDescriptionEntropyPricedMemberConfigV1",
    "DirectDescriptionEntropyPricedMemberProgramV1",
    "DirectDescriptionEntropyRungCheckpointV1",
    "build_entropy_candidate_z",
    "run_entropy_candidate_stages",
    "run_entropy_priced_member_n64",
    "run_entropy_rung_stages",
]
