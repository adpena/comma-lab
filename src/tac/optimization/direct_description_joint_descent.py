# SPDX-License-Identifier: MIT
"""Executable low-dimensional joint-descent consumer for DDM #366.

The receiver archive remains the source of truth.  This module adds three
strictly separated surfaces:

* a lossless G1/lane/template parameter lift and exact stage-00 recompile;
* a local realized-secant MLX module whose only differentiable leaves are the
  counted description coordinates;
* atomic, stage-preserving optimizer checkpoints with EMA and identity custody.

MLX/Metal results are training signal only.  Exact CPU/CUDA evaluation of the
emitted receiver archive remains the contest authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    LANE_PROGRAM_MEMBER,
    REALIZATION_STATIC_RULE_MEMBER,
    WORLDSHEET_G1_MEMBER,
    CarrierComposeReceiverV1,
    LanePeriodicProgramV1,
    RowBandScorerTemplateV1,
    compile_carrier_compose_archive,
    parse_carrier_compose_archive,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_entropy_priced_member import rfc8785_canonicalize
from tac.optimization.direct_description_g1_worldsheet import (
    G1WorldsheetParameterLiftV1,
    encode_lifted_g1_movable_worldsheet,
    lift_g1_movable_worldsheet,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.witness_control.resume_registry import (
    RESUME_REGISTRY_MANIFEST_KEY,
    ResumeRegistry,
)

TYPED_SCHEMA: Final = "DirectDescriptionJointDescentTypedConfigV1"
TICKET_SCHEMA: Final = "ddm_joint_descent_witness_program_ticket.v1"
MEMORY_RECEIPT_SCHEMA: Final = "ddm_joint_descent_memory_preflight.v1"
CHECKPOINT_SCHEMA: Final = "ddm_joint_descent_stage_checkpoint.v1"
LEGACY_PROGRAM_SHA256: Final = "68a8aa97b25a6be2f8f08e36fcf4957fe032233e43b1050b75ad13c9d7dad89c"
EXPECTED_PROGRAM_SHA256: Final = "df8db01f60d582b0a716ae62af3422997fcc12c014364939ab2935a2c403b824"
SUPPORTED_PROGRAM_SHA256: Final = frozenset({LEGACY_PROGRAM_SHA256, EXPECTED_PROGRAM_SHA256})
EXPECTED_ARCHIVE_SHA256: Final = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
EXPECTED_ARCHIVE_BYTES: Final = 133_941
POINTER: Final = "0.1910828242 [contest-CPU]"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
BASELINE_DSEG: Final = 0.027470296224


def classify_realized_stage_verdict(
    *,
    reference_d_seg: float,
    reference_d_pose: float,
    candidate_d_seg: float,
    candidate_d_pose: float,
    target_d_seg: float,
    target_d_pose: float | None,
) -> str:
    """Classify only an exact realized-through-receiver stage measurement.

    A stage may continue only after strict total d_seg descent without any pose
    regression.  This deliberately refuses proxy-loss, STE, or first-order
    predictions as campaign decisions.
    """

    values = (
        reference_d_seg,
        reference_d_pose,
        candidate_d_seg,
        candidate_d_pose,
        target_d_seg,
    )
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
        return "REFUSE_REALIZED_STAGE_VERDICT_NONFINITE_OR_NEGATIVE"
    if target_d_pose is not None and (
        not math.isfinite(float(target_d_pose)) or float(target_d_pose) < 0.0
    ):
        return "REFUSE_REALIZED_STAGE_TARGET_NONFINITE_OR_NEGATIVE"
    if candidate_d_seg > reference_d_seg:
        return "BLOCKED_REALIZED_DSEG_REGRESSION"
    if candidate_d_pose > reference_d_pose:
        return "BLOCKED_REALIZED_DPOSE_REGRESSION"
    if candidate_d_seg >= reference_d_seg:
        return "REALIZED_STAGE_NO_TOTAL_DSEG_DESCENT"
    target_met = candidate_d_seg <= target_d_seg and (
        target_d_pose is None or candidate_d_pose <= target_d_pose
    )
    return "REALIZED_STAGE_TARGET_MET" if target_met else "REALIZED_STAGE_DESCENT_CONTINUE"


@dataclass(frozen=True, slots=True)
class FullRunStageV1:
    stage_id: str
    active_groups: tuple[str, ...]
    maximum_steps: int
    verdict_interval_steps: int
    target_d_seg: float
    target_d_pose: float | None


@dataclass(frozen=True, slots=True)
class FullRunScheduleV1:
    """Hash-sealed schedule consumed only by the real n600 full-run path."""

    train_batch: int
    learning_rate_quantum_fraction: float
    checkpoint_interval_steps: int
    plateau_verdicts: int
    warm_start_pair: int
    warm_start_steps: int
    measured_seconds_per_step: float
    measured_seconds_per_step_low: float
    measured_seconds_per_step_high: float
    stages: tuple[FullRunStageV1, ...]

    @classmethod
    def from_semantic_program(cls, semantic: Mapping[str, Any]) -> FullRunScheduleV1 | None:
        payload = semantic.get("full_run_schedule")
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise DirectDescriptionError("full-run schedule must be a mapping")
        stages_raw = payload.get("stages")
        if not isinstance(stages_raw, list) or not stages_raw:
            raise DirectDescriptionError("full-run schedule requires nonempty stages")
        stages = tuple(
            FullRunStageV1(
                stage_id=str(row["stage_id"]),
                active_groups=tuple(str(value) for value in row["active_groups"]),
                maximum_steps=int(row["maximum_steps"]),
                verdict_interval_steps=int(row["verdict_interval_steps"]),
                target_d_seg=float(row["target_d_seg"]),
                target_d_pose=None if row.get("target_d_pose") is None else float(row["target_d_pose"]),
            )
            for row in stages_raw
        )
        result = cls(
            train_batch=int(payload["train_batch"]),
            learning_rate_quantum_fraction=float(payload["learning_rate_quantum_fraction"]),
            checkpoint_interval_steps=int(payload["checkpoint_interval_steps"]),
            plateau_verdicts=int(payload["plateau_verdicts"]),
            warm_start_pair=int(payload["warm_start_pair"]),
            warm_start_steps=int(payload["warm_start_steps"]),
            measured_seconds_per_step=float(payload["measured_seconds_per_step"]),
            measured_seconds_per_step_low=float(payload["measured_seconds_per_step_low"]),
            measured_seconds_per_step_high=float(payload["measured_seconds_per_step_high"]),
            stages=stages,
        )
        if not 1 <= result.train_batch <= 600:
            raise DirectDescriptionError("full-run train batch is outside n600")
        if not 0.0 < result.learning_rate_quantum_fraction <= 0.25:
            raise DirectDescriptionError("full-run learning rate exceeds the realized uint8 quarter-quantum bound")
        if result.checkpoint_interval_steps <= 0 or result.plateau_verdicts <= 0:
            raise DirectDescriptionError("full-run checkpoint/plateau schedule is invalid")
        if not 0 <= result.warm_start_pair < 600:
            raise DirectDescriptionError("full-run warm-start pair is outside n600")
        if result.warm_start_steps <= 0:
            raise DirectDescriptionError("full-run warm-start steps are invalid")
        timings = (
            result.measured_seconds_per_step_low,
            result.measured_seconds_per_step,
            result.measured_seconds_per_step_high,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in timings) or not (
            timings[0] <= timings[1] <= timings[2]
        ):
            raise DirectDescriptionError("full-run measured timing band is invalid")
        allowed_groups = {"island_worldsheet", "lane_program", "shared_template_dof"}
        if any(
            stage.maximum_steps <= 0
            or stage.verdict_interval_steps <= 0
            or stage.verdict_interval_steps > stage.maximum_steps
            or not set(stage.active_groups) <= allowed_groups
            or not 0.0 <= stage.target_d_seg <= BASELINE_DSEG
            or (stage.target_d_pose is not None and stage.target_d_pose < 0.0)
            for stage in result.stages
        ):
            raise DirectDescriptionError("full-run stage schedule is invalid")
        return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bound_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError(f"bound regular file is unavailable: {path}")
    payload = path.read_bytes()
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise DirectDescriptionError(
            f"bound file byte count differs for {path}: {len(payload)} != {expected_bytes}"
        )
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise DirectDescriptionError(f"bound file sha256 differs for {path}: {actual} != {expected_sha256}")
    return payload


@dataclass(frozen=True, slots=True)
class DirectDescriptionJointDescentTypedConfigV1:
    """Executable typed projection of the hash-sealed J1 semantic program."""

    ticket_path: str
    semantic_program: Mapping[str, Any]
    dsl_compile_hash: str
    source_archive_path: str
    source_archive_sha256: str
    source_archive_bytes: int
    target_cache_path: str
    target_cache_sha256: str
    target_cache_bytes: int
    upstream_root: str
    num_pairs: int
    seed: int
    verdict_batch: int
    ema_decay: float
    grad_clip: float
    memory_ceiling_gib: float
    custom_grouped_backward_required: bool
    fused_r_required: bool
    full_run_schedule: FullRunScheduleV1 | None
    score_claim: bool = False
    research_only: bool = True

    @classmethod
    def from_ticket(cls, ticket_path: Path) -> DirectDescriptionJointDescentTypedConfigV1:
        ticket_payload = ticket_path.read_bytes()
        ticket = json.loads(ticket_payload)
        if ticket.get("schema") != TICKET_SCHEMA:
            raise DirectDescriptionError("joint-descent ticket schema is not canonical")
        semantic = ticket.get("semantic_program")
        if not isinstance(semantic, dict):
            raise DirectDescriptionError("joint-descent ticket lacks a semantic program")
        semantic_hash = _sha256(rfc8785_canonicalize(semantic))
        sealed = ticket.get("compile_custody", {}).get("semantic_program_sha256")
        if semantic_hash != sealed or semantic_hash not in SUPPORTED_PROGRAM_SHA256:
            raise DirectDescriptionError(
                f"joint-descent DSL hash mismatch: computed={semantic_hash} sealed={sealed}"
            )
        warm = semantic["warm_start"]
        cache = semantic["target_cache"]
        stability = semantic["joint_objective"]["stability"]
        compute = semantic["compute_contract"]["baseline"]
        if int(semantic["num_pairs"]) != 600 or int(semantic["seed"]) != 0:
            raise DirectDescriptionError("joint-descent ticket must remain n600/seed0")
        if warm["sha256"] != EXPECTED_ARCHIVE_SHA256 or int(warm["bytes"]) != EXPECTED_ARCHIVE_BYTES:
            raise DirectDescriptionError("joint-descent warm-start identity drifted")
        env = compute.get("environment", {})
        required_kernels = " ".join(str(value) for value in compute.get("required_kernels", ())).lower()
        return cls(
            ticket_path=str(ticket_path),
            semantic_program=semantic,
            dsl_compile_hash=semantic_hash,
            source_archive_path=str(warm["path"]),
            source_archive_sha256=str(warm["sha256"]),
            source_archive_bytes=int(warm["bytes"]),
            target_cache_path=str(cache["path"]),
            target_cache_sha256=str(cache["sha256"]),
            target_cache_bytes=int(cache["bytes"]),
            upstream_root=str(Path(ticket["authority"]["delegation_prompt_path"]).parents[3] / "upstream"),
            num_pairs=600,
            seed=0,
            verdict_batch=16,
            ema_decay=float(semantic["joint_objective"]["ema_decay"]),
            grad_clip=float(stability["grad_clip"]),
            memory_ceiling_gib=116.0,
            custom_grouped_backward_required=env.get("TAC_MLX_CUSTOM_GROUPED_BACKWARD") == "1",
            fused_r_required="fused differentiable-r" in required_kernels,
            full_run_schedule=FullRunScheduleV1.from_semantic_program(semantic),
        )

    def identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": TYPED_SCHEMA,
            "dsl_compile_hash": self.dsl_compile_hash,
            "source_archive_sha256": self.source_archive_sha256,
            "target_cache_sha256": self.target_cache_sha256,
            "num_pairs": self.num_pairs,
            "seed": self.seed,
            "verdict_batch": self.verdict_batch,
            "ema_decay": self.ema_decay,
            "grad_clip": self.grad_clip,
            "memory_ceiling_gib": self.memory_ceiling_gib,
            "custom_grouped_backward_required": self.custom_grouped_backward_required,
            "fused_r_required": self.fused_r_required,
            "score_claim": False,
            "research_only": True,
        }
        if self.full_run_schedule is not None:
            payload["full_run_schedule"] = {
                "train_batch": self.full_run_schedule.train_batch,
                "learning_rate_quantum_fraction": self.full_run_schedule.learning_rate_quantum_fraction,
                "checkpoint_interval_steps": self.full_run_schedule.checkpoint_interval_steps,
                "plateau_verdicts": self.full_run_schedule.plateau_verdicts,
                "warm_start_pair": self.full_run_schedule.warm_start_pair,
                "warm_start_steps": self.full_run_schedule.warm_start_steps,
                "measured_seconds_per_step": self.full_run_schedule.measured_seconds_per_step,
                "measured_seconds_per_step_low": self.full_run_schedule.measured_seconds_per_step_low,
                "measured_seconds_per_step_high": self.full_run_schedule.measured_seconds_per_step_high,
                "stages": [
                    {
                        "stage_id": stage.stage_id,
                        "active_groups": list(stage.active_groups),
                        "maximum_steps": stage.maximum_steps,
                        "verdict_interval_steps": stage.verdict_interval_steps,
                        "target_d_seg": stage.target_d_seg,
                        "target_d_pose": stage.target_d_pose,
                    }
                    for stage in self.full_run_schedule.stages
                ],
            }
        return payload

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.identity_payload()))


@dataclass(frozen=True, slots=True)
class LaneProgramSeedV1:
    """Counted-on-activation Lane seed recovered from an inherited coherent slot.

    Stage 00 keeps these encode-side records inactive so the V15 archive stays
    byte-identical.  Before a Lane coordinate becomes trainable, the complete
    record is emitted through :class:`LanePeriodicProgramV1`, making its range
    gate, BEV polynomial, width, and dash-comb phase counted and receiver-used.
    """

    line_index: int
    birth_pair: int
    death_pair_exclusive: int
    bev_coefficients: tuple[float, float, float, float]
    width_bias: float
    width_slope: float
    dash_phase_origin: float
    dash_phase_xi_gain: float
    range_gate_forward_max_m: float
    activation_rule: str = "emit_complete_record_before_first_lane_gradient"

    def counted_record(self) -> LanePeriodicProgramV1:
        return LanePeriodicProgramV1(
            line_index=self.line_index,
            birth_pair=self.birth_pair,
            death_pair_exclusive=self.death_pair_exclusive,
            dash_phase_origin_delta_q8=0,
            dash_phase_xi_gain_q8=int(np.clip(np.rint(self.dash_phase_xi_gain * 256.0), -32768, 32767)),
            width_bias_q8=0,
            width_slope_q12=0,
        )


def derive_lane_program_seeds(receiver: CarrierComposeReceiverV1) -> tuple[LaneProgramSeedV1, ...]:
    lane = next((row for row in receiver.layers if row.role == "Lane"), None)
    if lane is None or lane.lane_lines is None:
        raise DirectDescriptionError("joint-descent lift lacks inherited coherent Lane slots")
    start = receiver.predictor.source_pair_start
    stop = start + receiver.z.n_pairs
    maximum = max(len(lane.lane_lines[pair]) for pair in range(start, stop))
    rows: list[LaneProgramSeedV1] = []
    range_max = float((lane.lane_header or {}).get("dash_forward_max_m", 50.0))
    for line_index in range(maximum):
        present = [pair for pair in range(start, stop) if line_index < len(lane.lane_lines[pair])]
        if not present:
            continue
        birth, death = present[0], present[-1] + 1
        vectors = np.stack([lane.lane_lines[pair][line_index] for pair in present]).astype(np.float64)
        representative = np.median(vectors, axis=0)
        local = np.asarray(present, dtype=np.int64) - start
        xi = receiver.pose6_codes[local, 0].astype(np.int16).astype(np.float64)
        xi -= xi[0]
        design = np.stack((np.ones_like(xi), xi), axis=1)
        intercept, gain = np.linalg.lstsq(design, vectors[:, 7], rcond=None)[0]
        rows.append(
            LaneProgramSeedV1(
                line_index=line_index,
                birth_pair=birth,
                death_pair_exclusive=death,
                bev_coefficients=(
                    float(representative[0]),
                    float(representative[1]),
                    float(representative[2]),
                    float(representative[3]),
                ),
                width_bias=float(representative[4]),
                width_slope=float(representative[5]),
                dash_phase_origin=float(intercept),
                dash_phase_xi_gain=float(gain),
                range_gate_forward_max_m=range_max,
            )
        )
    if not rows:
        raise DirectDescriptionError("joint-descent lift derived zero Lane seed records")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class JointDescriptionParameterLiftV1:
    source_archive: bytes
    source_archive_sha256: str
    g1: G1WorldsheetParameterLiftV1
    lane_seeds: tuple[LaneProgramSeedV1, ...]
    template_rows: tuple[RowBandScorerTemplateV1, ...]
    parameter_names: tuple[str, ...]
    template_parameter_start: int

    def exact_reemit(self) -> bytes:
        members, _ = parse_carrier_compose_archive(self.source_archive)
        receiver = receive_carrier_compose_archive(self.source_archive)
        payload = encode_lifted_g1_movable_worldsheet(self.g1)
        archive, _ = compile_carrier_compose_archive(
            members["predictor.zip"],
            worldsheet_g1_payload=payload,
            realization_profile=receiver.realization_profile,
            realization_static_rule_payload=members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
            realization_static_rule_id=receiver.realization_static_rule_id,
            scorer_solved_templates=receiver.scorer_solved_templates,
        )
        if archive != self.source_archive:
            raise DirectDescriptionError("joint-descent stage-00 archive recompile is not byte-identical")
        return archive

    def lane_seed_archive(self) -> bytes:
        """Emit every Lane seed atomically before any Lane coordinate trains."""

        members, _ = parse_carrier_compose_archive(self.source_archive)
        receiver = receive_carrier_compose_archive(self.source_archive)
        records = tuple(row.counted_record() for row in self.lane_seeds)
        archive, _ = compile_carrier_compose_archive(
            members["predictor.zip"],
            worldsheet_g1_payload=encode_lifted_g1_movable_worldsheet(self.g1),
            lane_programs=records,
            realization_profile=receiver.realization_profile,
            realization_static_rule_payload=members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
            realization_static_rule_id=receiver.realization_static_rule_id,
            scorer_solved_templates=receiver.scorer_solved_templates,
        )
        parsed, _ = parse_carrier_compose_archive(archive)
        if LANE_PROGRAM_MEMBER not in parsed:
            raise DirectDescriptionError("counted Lane seed lacks a receiver-consumed archive home")
        return archive

    def inventory(self) -> dict[str, Any]:
        lane_archive = self.lane_seed_archive()
        return {
            "schema": "ddm_joint_descent_parameter_lift.v1",
            "source_archive_bytes": len(self.source_archive),
            "source_archive_sha256": self.source_archive_sha256,
            "stage00_reemit_byte_identical": self.exact_reemit() == self.source_archive,
            "g1_payload_bytes": self.g1.source_payload_bytes,
            "g1_payload_sha256": self.g1.source_payload_sha256,
            "worldsheet_track_count": len(self.g1.tracks),
            "worldsheet_knot_count": len(self.g1.knots),
            "worldsheet_template_count": len(self.g1.templates),
            "lane_program_seed_count": len(self.lane_seeds),
            "lane_seed_archive_bytes": len(lane_archive),
            "lane_seed_archive_sha256": _sha256(lane_archive),
            "lane_seed_counted_byte_delta": len(lane_archive) - len(self.source_archive),
            "scorer_solved_template_count": len(self.template_rows),
            "low_dim_parameter_count": len(self.parameter_names),
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
            "pointer": POINTER,
            "pointer_moved": False,
        }


def lift_v15_archive(archive: bytes) -> JointDescriptionParameterLiftV1:
    if len(archive) != EXPECTED_ARCHIVE_BYTES or _sha256(archive) != EXPECTED_ARCHIVE_SHA256:
        raise DirectDescriptionError("joint-descent parameter lift requires the sealed V15 archive")
    members, _ = parse_carrier_compose_archive(archive)
    receiver = receive_carrier_compose_archive(archive)
    g1 = lift_g1_movable_worldsheet(members[WORLDSHEET_G1_MEMBER])
    if receiver.scorer_solved_templates is None:
        raise DirectDescriptionError("joint-descent V15 warm start lacks its counted template bank")
    lanes = derive_lane_program_seeds(receiver)
    names: list[str] = []
    # Only coordinates that survive the current receiver encoder belong in the
    # optimizer surface.  ``aspect_log``/``rotation_radians`` are lift metadata
    # but are not encoded by G1S1; the inherited BEV/range seed values likewise
    # have no LanePeriodicProgramV1 wire fields.  Counting those was the J2
    # 706-name overstatement.  The executable surface is therefore 368 DOFs:
    # 2*163 track translations + 4*6 counted Lane fields + 3*6 template bytes.
    for track in g1.tracks:
        names.extend(
            f"island.track{track.object_id}.{field}"
            for field in ("center_x", "center_y")
        )
    for lane in lanes:
        names.extend(
            f"lane.line{lane.line_index}.{field}"
            for field in ("dash_phase_origin_q8", "dash_phase_xi_gain_q8", "width_bias_q8", "width_slope_q12")
        )
    template_start = len(names)
    template_rows = receiver.scorer_solved_templates.templates
    for index, _ in enumerate(template_rows):
        names.extend(f"template.row{index}.rgb_{channel}" for channel in ("r", "g", "b"))
    result = JointDescriptionParameterLiftV1(
        source_archive=archive,
        source_archive_sha256=_sha256(archive),
        g1=g1,
        lane_seeds=lanes,
        template_rows=template_rows,
        parameter_names=tuple(names),
        template_parameter_start=template_start,
    )
    result.exact_reemit()
    return result


def _compile_lift_variant(
    lift: JointDescriptionParameterLiftV1,
    *,
    g1: G1WorldsheetParameterLiftV1 | None = None,
    lane_programs: Sequence[LanePeriodicProgramV1] = (),
    template_rows: Sequence[RowBandScorerTemplateV1] | None = None,
    verify_member_effects: bool = True,
) -> bytes:
    """Compile one receiver-consumed parameter mutation from the sealed base."""

    members, _ = parse_carrier_compose_archive(lift.source_archive)
    receiver = receive_carrier_compose_archive(lift.source_archive)
    bank = receiver.scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("joint-descent variant lacks the inherited template bank")
    if template_rows is not None:
        bank = replace(bank, templates=tuple(template_rows))
    archive, _ = compile_carrier_compose_archive(
        members["predictor.zip"],
        worldsheet_g1_payload=encode_lifted_g1_movable_worldsheet(g1 or lift.g1),
        lane_programs=tuple(lane_programs),
        realization_profile=receiver.realization_profile,
        realization_static_rule_payload=members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
        realization_static_rule_id=receiver.realization_static_rule_id,
        scorer_solved_templates=bank,
    )
    receive_carrier_compose_archive(archive, verify_member_effects=verify_member_effects)
    return archive


def parameter_group_indices(lift: JointDescriptionParameterLiftV1) -> dict[str, tuple[int, ...]]:
    """Return the three receiver-effective parameter groups by exact name."""

    groups = {
        "island_worldsheet": tuple(
            index for index, name in enumerate(lift.parameter_names) if name.startswith("island.")
        ),
        "lane_program": tuple(
            index for index, name in enumerate(lift.parameter_names) if name.startswith("lane.")
        ),
        "shared_template_dof": tuple(
            index for index, name in enumerate(lift.parameter_names) if name.startswith("template.")
        ),
    }
    if tuple(len(groups[name]) for name in groups) != (2 * len(lift.g1.tracks), 4 * len(lift.lane_seeds), 3 * len(lift.template_rows)):
        raise DirectDescriptionError("joint-descent receiver-effective parameter grouping differs")
    return groups


def realize_parameter_theta(lift: JointDescriptionParameterLiftV1, theta: np.ndarray) -> np.ndarray:
    """Quantize optimizer coordinates into exact receiver wire quanta."""

    value = np.asarray(theta, dtype=np.float32)
    if value.shape != (len(lift.parameter_names),) or not np.all(np.isfinite(value)):
        raise DirectDescriptionError("joint-descent parameter theta is invalid")
    realized = np.rint(value.astype(np.float64))
    groups = parameter_group_indices(lift)
    island = np.asarray(groups["island_worldsheet"], dtype=np.int64)
    lane = np.asarray(groups["lane_program"], dtype=np.int64)
    if np.any(np.abs(realized[island]) > 4096):
        raise DirectDescriptionError("joint-descent island translation exceeds the guarded scorer grid")
    realized[lane] = np.clip(realized[lane], -32768, 32767)
    return np.ascontiguousarray(realized, dtype=np.float32)


def compile_parameterized_archive(
    lift: JointDescriptionParameterLiftV1,
    theta: np.ndarray,
    *,
    include_lane_programs: bool,
) -> tuple[bytes, np.ndarray]:
    """Compile quantized low-dimensional state into one receiver-closed archive."""

    realized = realize_parameter_theta(lift, theta)
    cursor = 0
    knots = list(lift.g1.knots)
    for track in lift.g1.tracks:
        dx, dy = (int(realized[cursor]), int(realized[cursor + 1]))
        cursor += 2
        for knot_index in track.knot_indices:
            knot = knots[knot_index]
            knots[knot_index] = replace(knot, center_x=knot.center_x + dx, center_y=knot.center_y + dy)
    g1 = replace(lift.g1, knots=tuple(knots))

    lanes: list[LanePeriodicProgramV1] = []
    for seed in lift.lane_seeds:
        base = seed.counted_record()
        deltas = tuple(int(value) for value in realized[cursor : cursor + 4])
        cursor += 4
        lanes.append(
            replace(
                base,
                dash_phase_origin_delta_q8=int(np.clip(base.dash_phase_origin_delta_q8 + deltas[0], -32768, 32767)),
                dash_phase_xi_gain_q8=int(np.clip(base.dash_phase_xi_gain_q8 + deltas[1], -32768, 32767)),
                width_bias_q8=int(np.clip(base.width_bias_q8 + deltas[2], -32768, 32767)),
                width_slope_q12=int(np.clip(base.width_slope_q12 + deltas[3], -32768, 32767)),
            )
        )

    templates: list[RowBandScorerTemplateV1] = []
    for row in lift.template_rows:
        channel_delta = np.asarray(realized[cursor : cursor + 3], dtype=np.int16)
        cursor += 3
        rgb = np.frombuffer(row.rgb_u8, dtype=np.uint8).reshape(-1, 3).astype(np.int16)
        rgb = np.clip(rgb + channel_delta[None, :], 0, 255).astype(np.uint8)
        templates.append(replace(row, rgb_u8=rgb.tobytes()))
    if cursor != len(realized):
        raise DirectDescriptionError("joint-descent parameter compiler left coordinates unconsumed")
    archive = _compile_lift_variant(
        lift,
        g1=g1,
        lane_programs=lanes if include_lane_programs else (),
        template_rows=templates,
        verify_member_effects=False,
    )
    return archive, realized


def realized_training_state(
    lift: JointDescriptionParameterLiftV1,
    theta: np.ndarray,
    *,
    pair_ids: Sequence[int],
    active_groups: Sequence[str],
    include_lane_programs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], np.ndarray, bytes]:
    """Build sparse exact +1-quantum secants around current parse-back state.

    The returned basis contains only receiver-effective island/Lane coordinates
    that can affect this pair window.  Shared-template coordinates use their
    exact grammar masks in :class:`DirectDescriptionJointDescentMLXModule`.
    """

    indexes = tuple(int(value) for value in pair_ids)
    archive, realized = compile_parameterized_archive(
        lift, theta, include_lane_programs=include_lane_programs
    )
    receiver = receive_carrier_compose_archive(archive, verify_member_effects=False)
    camera = receiver.render_camera_pairs(indexes).astype(np.float32)
    template_rows = receiver.scorer_solved_templates
    if template_rows is None:
        raise DirectDescriptionError("parameterized archive lost its template bank")
    masks = np.stack(
        [receiver.template_camera_masks(indexes, row).astype(np.float32) for row in template_rows.templates],
        axis=0,
    )
    if "shared_template_dof" not in active_groups:
        masks.fill(0.0)
    if np.any(masks.sum(axis=0) > 1.0):
        raise DirectDescriptionError("parameterized template masks overlap")

    groups = parameter_group_indices(lift)
    selected: list[int] = []
    if "island_worldsheet" in active_groups:
        pair_set = set(indexes)
        for track_index, track in enumerate(lift.g1.tracks):
            if any(lift.g1.knots[knot_index].pair_index in pair_set for knot_index in track.knot_indices):
                selected.extend((2 * track_index, 2 * track_index + 1))
    if "lane_program" in active_groups:
        selected.extend(groups["lane_program"])
    secants: list[np.ndarray] = []
    for parameter_index in selected:
        secant: np.ndarray | None = None
        errors: list[str] = []
        for direction in (1.0, -1.0):
            probe = realized.copy()
            probe[parameter_index] += np.float32(direction)
            try:
                probe_archive, _ = compile_parameterized_archive(
                    lift, probe, include_lane_programs=include_lane_programs
                )
                probe_camera = receive_carrier_compose_archive(
                    probe_archive, verify_member_effects=False
                ).render_camera_pairs(indexes)
            except DirectDescriptionError as exc:
                errors.append(str(exc))
                continue
            secant = (probe_camera.astype(np.float32) - camera) / np.float32(direction)
            break
        if secant is None:
            raise DirectDescriptionError(
                "joint-descent coordinate has no feasible one-quantum secant: "
                f"{lift.parameter_names[parameter_index]}: {'; '.join(errors)}"
            )
        secants.append(secant)
    basis = np.stack(secants, axis=0) if secants else np.empty((0, *camera.shape), dtype=np.float32)
    local_theta = np.asarray(theta, dtype=np.float32) - realized
    return camera, masks, basis, tuple(selected), local_theta, archive


def verify_trainable_group_ownership(lift: JointDescriptionParameterLiftV1) -> dict[str, Any]:
    """Prove each trainable group owns counted bytes and receiver-visible output.

    This is deliberately a bounded one-coordinate proof, not efficacy evidence.
    Every mutation is encoded into a receiver-consumed archive member before its
    camera output is compared with stage 00.
    """

    base_receiver = receive_carrier_compose_archive(lift.source_archive)
    rows: dict[str, Any] = {}

    # Island: translate one explicit lifecycle track by one scorer-grid pixel.
    track = next((row for row in lift.g1.tracks if row.knot_indices), None)
    if track is None:
        raise DirectDescriptionError("island ownership probe lacks a nonempty G1 track")
    selected = set(track.knot_indices)
    island_g1 = replace(
        lift.g1,
        knots=tuple(
            replace(knot, center_x=knot.center_x + 1) if index in selected else knot
            for index, knot in enumerate(lift.g1.knots)
        ),
    )
    island_archive = _compile_lift_variant(lift, g1=island_g1)
    island_pair = lift.g1.knots[track.knot_indices[0]].pair_index
    island_delta = int(
        np.count_nonzero(
            base_receiver.render_camera_pairs((island_pair,))
            != receive_carrier_compose_archive(island_archive).render_camera_pairs((island_pair,))
        )
    )
    rows["island_worldsheet"] = {
        "coordinate": f"track{track.object_id}.center_x_plus_1",
        "pair_id": island_pair,
        "archive_bytes": len(island_archive),
        "archive_sha256": _sha256(island_archive),
        "archive_changed": island_archive != lift.source_archive,
        "receiver_camera_changed_values": island_delta,
    }

    # Lane: materialize the complete counted seed, then change its phase.
    lane = lift.lane_seeds[0]
    lane_records = [row.counted_record() for row in lift.lane_seeds]
    lane_records[0] = replace(lane_records[0], dash_phase_origin_delta_q8=256)
    lane_archive = _compile_lift_variant(lift, lane_programs=lane_records)
    candidate_pairs = tuple(
        sorted(
            {
                lane.birth_pair,
                (lane.birth_pair + lane.death_pair_exclusive - 1) // 2,
                lane.death_pair_exclusive - 1,
            }
        )
    )
    lane_delta = int(
        np.count_nonzero(
            base_receiver.render_camera_pairs(candidate_pairs)
            != receive_carrier_compose_archive(lane_archive).render_camera_pairs(candidate_pairs)
        )
    )
    rows["lane_program"] = {
        "coordinate": f"line{lane.line_index}.dash_phase_origin_plus_1",
        "pair_ids": list(candidate_pairs),
        "archive_bytes": len(lane_archive),
        "archive_sha256": _sha256(lane_archive),
        "archive_changed": lane_archive != lift.source_archive,
        "receiver_camera_changed_values": lane_delta,
        "base_bev_coefficients": list(lane.bev_coefficients),
        "base_range_gate_forward_max_m": lane.range_gate_forward_max_m,
        "seed_is_counted_before_gradient": True,
    }

    # Template: perturb one counted RGB byte while retaining its typed record.
    active_template: tuple[int, RowBandScorerTemplateV1, int] | None = None
    for template_index, candidate in enumerate(lift.template_rows):
        candidate_pair = next(
            (
                pair_id
                for pair_id in range(lift.g1.pair_count)
                if np.any(base_receiver.template_camera_masks((pair_id,), candidate))
            ),
            None,
        )
        if candidate_pair is not None:
            active_template = (template_index, candidate, candidate_pair)
            break
    if active_template is None:
        raise DirectDescriptionError("counted template ownership probe found no receiver-visible site")
    template_index, template, template_pair = active_template
    rgb = bytearray(value + 1 if value < 255 else value - 1 for value in template.rgb_u8)
    templates = list(lift.template_rows)
    templates[template_index] = replace(template, rgb_u8=bytes(rgb))
    template_archive = _compile_lift_variant(lift, template_rows=templates)
    template_delta = int(
        np.count_nonzero(
            base_receiver.render_camera_pairs((template_pair,))
            != receive_carrier_compose_archive(template_archive).render_camera_pairs((template_pair,))
        )
    )
    rows["shared_template"] = {
        "coordinate": f"template{template_index}.all_rgb_plus_or_minus_1",
        "pair_id": template_pair,
        "archive_bytes": len(template_archive),
        "archive_sha256": _sha256(template_archive),
        "archive_changed": template_archive != lift.source_archive,
        "receiver_camera_changed_values": template_delta,
    }

    inert = [
        name
        for name, row in rows.items()
        if not row["archive_changed"] or row["receiver_camera_changed_values"] <= 0
    ]
    if inert:
        raise DirectDescriptionError(
            "joint-descent trainable groups are counted but receiver-inert: " + ",".join(inert)
        )
    return {
        "schema": "ddm_joint_descent_trainable_group_ownership.v1",
        "groups": rows,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
    }


def template_camera_state(
    lift: JointDescriptionParameterLiftV1,
    pair_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact V15 camera pairs and disjoint template masks for an MLX step."""

    receiver = receive_carrier_compose_archive(lift.source_archive)
    indexes = tuple(int(value) for value in pair_ids)
    camera = receiver.render_camera_pairs(indexes).astype(np.float32)
    masks = np.stack(
        [receiver.template_camera_masks(indexes, row).astype(np.float32) for row in lift.template_rows],
        axis=0,
    )
    # Same solved-template paint is applied to both frames.  Overlapping rows
    # would make a linear color lift order-dependent, so refuse rather than
    # silently double-own a pixel.
    if np.any(masks.sum(axis=0) > 1.0):
        raise DirectDescriptionError("V15 solved-template masks overlap in the trainable lift")
    return np.ascontiguousarray(camera), np.ascontiguousarray(masks)


class DirectDescriptionJointDescentMLXModule:
    """MLX params -> exact grammar paint -> uint8 STE -> fused R -> frozen scorers.

    Island/Lane coordinates enter through caller-supplied *realized secant*
    fields produced by exact archive parse-back.  Template coordinates use the
    receiver's exact camera masks directly.  No pixel tensor is trainable.
    """

    def __init__(
        self,
        *,
        lift: JointDescriptionParameterLiftV1,
        scorer_adapter: Any,
        seg_targets: np.ndarray,
        pose_targets: np.ndarray,
        margin_targets: np.ndarray | None = None,
        margin_hinge_weight: float = 0.05,
        margin_floor: float = 0.1,
    ) -> None:
        import mlx.core as mx

        self.mx = mx
        self.lift = lift
        self.scorer = scorer_adapter
        self.seg_targets = mx.array(np.asarray(seg_targets, dtype=np.int32))
        self.pose_targets = mx.array(np.asarray(pose_targets, dtype=np.float32))
        self.margin_targets = None if margin_targets is None else mx.array(
            np.asarray(margin_targets, dtype=np.float32)
        )
        self.margin_hinge_weight = float(margin_hinge_weight)
        self.margin_floor = float(margin_floor)
        self.parameter_count = len(lift.parameter_names)

    def _validate_step_arrays(
        self,
        theta: np.ndarray,
        pair_ids: Sequence[int],
        base_camera: np.ndarray,
        template_masks: np.ndarray,
        realized_secant_basis: np.ndarray | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> None:
        count = len(tuple(pair_ids))
        if np.asarray(theta).shape != (self.parameter_count,):
            raise DirectDescriptionError("joint-descent theta geometry differs")
        if np.asarray(base_camera).shape != (count, 2, 874, 1164, 3):
            raise DirectDescriptionError("joint-descent camera batch geometry differs")
        if np.asarray(template_masks).shape != (len(self.lift.template_rows), count, 874, 1164):
            raise DirectDescriptionError("joint-descent template-mask geometry differs")
        if realized_secant_basis is not None and np.asarray(realized_secant_basis).shape != (
            len(tuple(realized_secant_indices or ())),
            count,
            2,
            874,
            1164,
            3,
        ):
            raise DirectDescriptionError("joint-descent realized-secant basis geometry differs")
        if realized_secant_basis is not None and any(
            index < 0 or index >= self.parameter_count for index in tuple(realized_secant_indices or ())
        ):
            raise DirectDescriptionError("joint-descent realized-secant index is outside theta")

    def _render_camera(
        self,
        theta: Any,
        base_camera: Any,
        template_masks: Any,
        realized_secant_basis: Any | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> Any:
        mx = self.mx
        template_count = len(self.lift.template_rows)
        start = self.lift.template_parameter_start
        colour_delta = mx.reshape(theta[start : start + template_count * 3], (template_count, 3))
        # masks K,B,H,W; delta -> B,H,W,3 and is shared across the two frames.
        paint_delta = mx.einsum("kbhw,kc->bhwc", template_masks, colour_delta)
        camera = base_camera + paint_delta[:, None, :, :, :]
        if realized_secant_basis is not None:
            # Basis is K,B,2,H,W,3 and is derived from exact archive finite
            # secants.  It is immutable receiver geometry, never a trainable
            # frame table; theta is the sole differentiable state.
            selected = theta[mx.array(np.asarray(realized_secant_indices, dtype=np.int32))]
            camera = camera + mx.tensordot(selected, realized_secant_basis, axes=[[0], [0]])
        clipped = mx.clip(camera, 0.0, 255.0)
        return clipped + mx.stop_gradient(mx.round(clipped) - clipped)

    def _loss(
        self,
        theta: Any,
        *,
        pair_ids: Any,
        base_camera: Any,
        template_masks: Any,
        realized_secant_basis: Any | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> Any:
        mx = self.mx
        seg, pose_mse, _ = self._components(
            theta,
            pair_ids=pair_ids,
            base_camera=base_camera,
            template_masks=template_masks,
            realized_secant_basis=realized_secant_basis,
            realized_secant_indices=realized_secant_indices,
        )
        # The sqrt term is the exact contest action; epsilon only defines its
        # derivative at zero and is far below the observed warm-start value.
        return 100.0 * seg + mx.sqrt(10.0 * pose_mse + 1.0e-12)

    def _components(
        self,
        theta: Any,
        *,
        pair_ids: Any,
        base_camera: Any,
        template_masks: Any,
        realized_secant_basis: Any | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> tuple[Any, Any, Any]:
        mx = self.mx
        from tac.local_acceleration.metal_fused_r_operator import fused_r_roundtrip
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx
        from tac.mlx_pr95_port.mlx_losses import (
            ce_seg_loss_mlx,
            margin_floor_hinge_mlx,
            pose_loss_mlx,
        )

        camera = self._render_camera(
            theta,
            base_camera,
            template_masks,
            realized_secant_basis,
            realized_secant_indices,
        )
        flat = mx.reshape(camera, (-1, 874, 1164, 3))
        scorer_rgb = fused_r_roundtrip(
            flat,
            camera_hw=(874, 1164),
            output_hw=(384, 512),
            ste_round=True,
        )
        pairs = mx.reshape(scorer_rgb, (-1, 2, 384, 512, 3))
        seg_logits = self.scorer.segnet(pairs[:, 1])
        seg_logits_nchw = mx.transpose(seg_logits, (0, 3, 1, 2))
        targets = self.seg_targets[pair_ids]
        seg = ce_seg_loss_mlx(seg_logits_nchw, targets)
        if self.margin_hinge_weight > 0.0:
            seg = seg + self.margin_hinge_weight * margin_floor_hinge_mlx(
                seg_logits_nchw, targets, margin_floor=self.margin_floor
            )
        yuv6 = rgb_to_yuv6_mlx(pairs)
        pose_input = mx.reshape(mx.transpose(yuv6, (0, 2, 3, 1, 4)), (-1, 192, 256, 12))
        pose = self.scorer.posenet(pose_input)["pose"][..., :6]
        pose_mse = pose_loss_mlx(pose, self.pose_targets[pair_ids])
        d_seg = mx.mean(mx.not_equal(mx.argmax(seg_logits, axis=-1), targets).astype(mx.float32))
        return seg, pose_mse, d_seg

    def loss_and_grad(
        self,
        theta: np.ndarray,
        *,
        pair_ids: Sequence[int],
        base_camera: np.ndarray,
        template_masks: np.ndarray,
        realized_secant_basis: np.ndarray | None = None,
        realized_secant_indices: Sequence[int] | None = None,
    ) -> tuple[float, np.ndarray]:
        self._validate_step_arrays(
            theta,
            pair_ids,
            base_camera,
            template_masks,
            realized_secant_basis,
            realized_secant_indices,
        )
        mx = self.mx
        pair_mx = mx.array(np.asarray(pair_ids, dtype=np.int32))
        base_mx = mx.array(np.asarray(base_camera, dtype=np.float32))
        masks_mx = mx.array(np.asarray(template_masks, dtype=np.float32))
        basis_mx = None if realized_secant_basis is None else mx.array(
            np.asarray(realized_secant_basis, dtype=np.float32)
        )

        def closure(value: Any) -> Any:
            return self._loss(
                value,
                pair_ids=pair_mx,
                base_camera=base_mx,
                template_masks=masks_mx,
                realized_secant_basis=basis_mx,
                realized_secant_indices=realized_secant_indices,
            )

        value, gradient = mx.value_and_grad(closure)(mx.array(np.asarray(theta, dtype=np.float32)))
        mx.eval(value, gradient)
        return float(np.asarray(value)), np.asarray(gradient, dtype=np.float32)

    def measure_components(
        self,
        theta: np.ndarray,
        *,
        pair_ids: Sequence[int],
        base_camera: np.ndarray,
        template_masks: np.ndarray,
        realized_secant_basis: np.ndarray | None = None,
        realized_secant_indices: Sequence[int] | None = None,
    ) -> dict[str, float]:
        """Measure the same MLX research-signal components without updating state."""

        self._validate_step_arrays(
            theta,
            pair_ids,
            base_camera,
            template_masks,
            realized_secant_basis,
            realized_secant_indices,
        )
        mx = self.mx
        pair_mx = mx.array(np.asarray(pair_ids, dtype=np.int32))
        basis = None if realized_secant_basis is None else mx.array(
            np.asarray(realized_secant_basis, dtype=np.float32)
        )
        seg, pose, d_seg = self._components(
            mx.array(np.asarray(theta, dtype=np.float32)),
            pair_ids=pair_mx,
            base_camera=mx.array(np.asarray(base_camera, dtype=np.float32)),
            template_masks=mx.array(np.asarray(template_masks, dtype=np.float32)),
            realized_secant_basis=basis,
            realized_secant_indices=realized_secant_indices,
        )
        objective = 100.0 * seg + mx.sqrt(10.0 * pose + 1.0e-12)
        mx.eval(seg, pose, d_seg, objective)
        return {
            "seg_ce_margin": float(np.asarray(seg)),
            "d_seg": float(np.asarray(d_seg)),
            "d_pose": float(np.asarray(pose)),
            "joint_objective_no_rate": float(np.asarray(objective)),
        }


@dataclass(frozen=True, slots=True)
class AdamStateV1:
    step: int
    theta: np.ndarray
    ema: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray


def _optimizer_state_sha256(state: AdamStateV1) -> str:
    digest = hashlib.sha256(int(state.step).to_bytes(8, "little", signed=False))
    for value in (state.theta, state.ema, state.first_moment, state.second_moment):
        array = np.ascontiguousarray(value, dtype="<f4")
        digest.update(len(array).to_bytes(8, "little", signed=False))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class JointDescentResumeControllerV1:
    """Canonical resume-registry integrity controller for optimizer state."""

    state: AdamStateV1
    typed_config_hash: str
    event_mode: bool = True

    def state_arrays(self, prefix: str) -> dict[str, Any]:
        return {
            prefix + "step": np.asarray(self.state.step, dtype=np.int64),
            prefix + "optimizer_state_sha256": np.asarray(_optimizer_state_sha256(self.state)),
            prefix + "typed_config_hash": np.asarray(self.typed_config_hash),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict[str, Any]) -> bool:
        required = (prefix + "step", prefix + "optimizer_state_sha256", prefix + "typed_config_hash")
        if any(key not in cfg for key in required):
            return False
        if int(cfg[required[0]]) != self.state.step:
            raise DirectDescriptionError("resume-registry optimizer step differs")
        if str(cfg[required[1]]) != _optimizer_state_sha256(self.state):
            raise DirectDescriptionError("resume-registry optimizer state hash differs")
        if str(cfg[required[2]]) != self.typed_config_hash:
            raise DirectDescriptionError("resume-registry typed config hash differs")
        return True


def _optimizer_resume_registry(
    state: AdamStateV1,
    config: DirectDescriptionJointDescentTypedConfigV1,
) -> ResumeRegistry:
    registry = ResumeRegistry()
    registry.register(
        "ddm_joint_descent_optimizer",
        "__ddmjd_",
        JointDescentResumeControllerV1(state=state, typed_config_hash=config.typed_config_hash()),
    )
    return registry


def initial_adam_state(parameter_count: int) -> AdamStateV1:
    zeros = np.zeros(int(parameter_count), dtype=np.float32)
    return AdamStateV1(0, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())


def clipped_adam_step(
    state: AdamStateV1,
    gradient: np.ndarray,
    *,
    learning_rate: float,
    grad_clip: float,
    ema_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> AdamStateV1:
    grad = np.asarray(gradient, dtype=np.float32)
    if grad.shape != state.theta.shape or any(
        value.shape != state.theta.shape
        for value in (state.ema, state.first_moment, state.second_moment)
    ):
        raise DirectDescriptionError("joint-descent Adam state/gradient geometry differs")
    if learning_rate <= 0.0 or grad_clip <= 0.0 or not 0.0 <= ema_decay < 1.0:
        raise DirectDescriptionError("joint-descent Adam hyperparameters are invalid")
    norm = float(np.linalg.norm(grad.astype(np.float64)))
    if not math.isfinite(norm):
        raise DirectDescriptionError("joint-descent gradient is nonfinite")
    if norm > grad_clip:
        grad = grad * np.float32(grad_clip / norm)
    step = state.step + 1
    first = beta1 * state.first_moment + (1.0 - beta1) * grad
    second = beta2 * state.second_moment + (1.0 - beta2) * np.square(grad)
    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)
    theta = state.theta - learning_rate * first_hat / (np.sqrt(second_hat) + 1.0e-8)
    ema = ema_decay * state.ema + (1.0 - ema_decay) * theta
    return AdamStateV1(
        step=step,
        theta=np.asarray(theta, dtype=np.float32),
        ema=np.asarray(ema, dtype=np.float32),
        first_moment=np.asarray(first, dtype=np.float32),
        second_moment=np.asarray(second, dtype=np.float32),
    )


def save_stage_checkpoint(
    path: Path,
    state: AdamStateV1,
    *,
    stage_id: str,
    config: DirectDescriptionJointDescentTypedConfigV1,
    telemetry: Sequence[Mapping[str, Any]],
    run_cursor: Mapping[str, Any] | None = None,
    realized_archive: Mapping[str, Any] | None = None,
) -> str:
    """Atomically preserve a distinct stage/step checkpoint; never overwrite."""

    if path.exists():
        raise DirectDescriptionError(f"preserved joint-descent checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "stage_id": stage_id,
        "step": state.step,
        "typed_config_hash": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash,
        "source_archive_sha256": config.source_archive_sha256,
        "target_cache_sha256": config.target_cache_sha256,
        "seed": config.seed,
        "rng": {"kind": "deterministic_no_sampling", "state": config.seed},
        "ema_shadow_saved": True,
        "live_weights_saved_for_resume_only": True,
        "optimizer": "adam_fp32",
        "canonical_resume_registry": {
            "helper": "tac.witness_control.resume_registry.ResumeRegistry",
            "controller": "ddm_joint_descent_optimizer",
            "prefix": "__ddmjd_",
            "manifest_key": RESUME_REGISTRY_MANIFEST_KEY,
        },
        "telemetry": list(telemetry),
        "run_cursor": dict(run_cursor or {}),
        "realized_archive": dict(realized_archive or {}),
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
    }
    registry_arrays = _optimizer_resume_registry(state, config).state_arrays()
    if RESUME_REGISTRY_MANIFEST_KEY not in registry_arrays:
        raise DirectDescriptionError("joint-descent checkpoint lacks canonical resume manifest")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            theta=state.theta,
            ema=state.ema,
            first_moment=state.first_moment,
            second_moment=state.second_moment,
            metadata=np.frombuffer(rfc8785_canonicalize(metadata), dtype=np.uint8),
            **registry_arrays,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return _sha256(path.read_bytes())


def load_stage_checkpoint(
    path: Path,
    *,
    config: DirectDescriptionJointDescentTypedConfigV1,
) -> tuple[AdamStateV1, dict[str, Any]]:
    payload = path.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(np.asarray(archive["metadata"], dtype=np.uint8).tobytes())
        if metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise DirectDescriptionError("joint-descent checkpoint schema differs")
        if metadata.get("typed_config_hash") != config.typed_config_hash():
            raise DirectDescriptionError("joint-descent checkpoint typed config differs")
        state = AdamStateV1(
            step=int(metadata["step"]),
            theta=np.ascontiguousarray(archive["theta"], dtype=np.float32),
            ema=np.ascontiguousarray(archive["ema"], dtype=np.float32),
            first_moment=np.ascontiguousarray(archive["first_moment"], dtype=np.float32),
            second_moment=np.ascontiguousarray(archive["second_moment"], dtype=np.float32),
        )
        shapes = {
            state.theta.shape,
            state.ema.shape,
            state.first_moment.shape,
            state.second_moment.shape,
        }
        if len(shapes) != 1 or len(state.theta.shape) != 1 or not all(
            np.all(np.isfinite(value))
            for value in (state.theta, state.ema, state.first_moment, state.second_moment)
        ):
            raise DirectDescriptionError("joint-descent checkpoint optimizer arrays are invalid")
        cfg = {
            key: np.asarray(archive[key]).item()
            for key in archive.files
            if key.startswith("__")
        }
        report = _optimizer_resume_registry(state, config).restore(cfg)
        if not report.manifest_present or report.restored != {"ddm_joint_descent_optimizer": True}:
            raise DirectDescriptionError("joint-descent canonical resume-registry restore is incomplete")
    return state, metadata


def classify_memory_preflight(projected_peak_gib: float, *, ceiling_gib: float = 116.0) -> tuple[bool, str]:
    peak = float(projected_peak_gib)
    ceiling = float(ceiling_gib)
    if not math.isfinite(peak) or peak <= 0.0:
        return False, "REFUSE_INVALID_MEASURED_PEAK"
    if peak > ceiling:
        return False, "REFUSE_PROJECTED_PEAK_EXCEEDS_116_GIB_CEILING"
    return True, "SAFE_PROJECTED_PEAK_WITHIN_116_GIB_CEILING"


__all__ = [
    "AdamStateV1",
    "DirectDescriptionJointDescentMLXModule",
    "DirectDescriptionJointDescentTypedConfigV1",
    "FullRunScheduleV1",
    "FullRunStageV1",
    "JointDescentResumeControllerV1",
    "JointDescriptionParameterLiftV1",
    "LaneProgramSeedV1",
    "classify_memory_preflight",
    "classify_realized_stage_verdict",
    "clipped_adam_step",
    "compile_parameterized_archive",
    "derive_lane_program_seeds",
    "initial_adam_state",
    "lift_v15_archive",
    "load_stage_checkpoint",
    "parameter_group_indices",
    "realize_parameter_theta",
    "realized_training_state",
    "save_stage_checkpoint",
    "template_camera_state",
    "verify_trainable_group_ownership",
]
