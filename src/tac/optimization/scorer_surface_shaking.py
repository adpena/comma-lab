"""Scorer-surface shaking plans for exact-eval score lowering.

This module is a deterministic planner for aggressive local exploration near a
known packet frontier.  It names pixel/frame/scorer perturbation families,
computes their local score economics, and records the PacketIR/materialization
gates that must pass before any row can become score evidence.

It does not load SegNet/PoseNet, read videos, build archives, dispatch jobs, or
claim scores.  It is the optimizer-facing bridge between "shake every surface"
and the byte-closed packet compiler discipline.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES

SCHEMA = "scorer_surface_shaking_plan.v1"
TOOL = "tac.optimization.scorer_surface_shaking"
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES


class ScorerSurfacePlanError(ValueError):
    """Raised when a scorer-surface plan input is malformed."""


@dataclass(frozen=True)
class OperatingPoint:
    """Measured packet state used for local rate-distortion economics."""

    label: str
    device_axis: str
    score: float
    archive_bytes: int
    seg_dist: float
    pose_dist: float
    archive_sha256: str | None = None
    evidence_grade: str = "[contest-CUDA]"

    def __post_init__(self) -> None:
        if not self.label:
            raise ScorerSurfacePlanError("operating point label is required")
        if self.device_axis not in {"contest_cuda", "contest_cpu", "diagnostic_cuda", "diagnostic_cpu"}:
            raise ScorerSurfacePlanError(f"unsupported device_axis: {self.device_axis}")
        if not math.isfinite(self.score) or self.score <= 0:
            raise ScorerSurfacePlanError("score must be a positive finite number")
        if self.archive_bytes <= 0:
            raise ScorerSurfacePlanError("archive_bytes must be positive")
        if self.seg_dist < 0 or self.pose_dist < 0:
            raise ScorerSurfacePlanError("distortions must be non-negative")

    @property
    def byte_slope(self) -> float:
        return RATE_SCORE_PER_BYTE

    @property
    def seg_slope(self) -> float:
        return 100.0

    # Boundary-regularization epsilon for pose_dist near zero. The score
    # function `sqrt(10*pose_dist)` has derivative `5/sqrt(10*pose_dist)`
    # which diverges as pose_dist → 0 (physically impossible — would imply
    # perfect pose recovery). Clamp at 1e-12 to produce a very-large-but-
    # finite slope (~1.6e6) instead of float("inf"). This makes the slope
    # JSON-serializable (RFC 8259 — no Infinity literal), preserves the
    # derivative's correct sign + order-of-magnitude, and lets consumers
    # tag the operating point as `pose_dist_at_singularity_boundary` if
    # they care to distinguish the regularized case.
    _POSE_DIST_REGULARIZATION_EPS: float = 1e-12

    @property
    def pose_slope(self) -> float:
        pose_dist_reg = max(self.pose_dist, self._POSE_DIST_REGULARIZATION_EPS)
        return math.sqrt(10.0) / (2.0 * math.sqrt(pose_dist_reg))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "device_axis": self.device_axis,
            "score": self.score,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "evidence_grade": self.evidence_grade,
            "seg_dist": self.seg_dist,
            "pose_dist": self.pose_dist,
            "score_slopes": {
                "d_score_d_byte": self.byte_slope,
                "d_score_d_seg": self.seg_slope,
                "d_score_d_pose": self.pose_slope,
                "pose_over_seg_marginal_value": self.pose_slope / self.seg_slope,
                "one_byte_break_even_seg_delta": self.byte_slope / self.seg_slope,
                "one_byte_break_even_pose_delta": (
                    self.byte_slope / self.pose_slope
                    if math.isfinite(self.pose_slope) and self.pose_slope > 0
                    else 0.0
                ),
            },
        }


@dataclass(frozen=True)
class SurfaceAtomFamily:
    """One local perturbation family before archive materialization."""

    atom_id: str
    surface: str
    perturbation_space: str
    expected_added_bytes: int
    expected_seg_delta: float
    expected_pose_delta: float
    packetir_stream: str
    materialization_path: str
    primary_solver: str
    local_parallelism_shape: str
    pass_sequence: tuple[str, ...]
    guardrails: tuple[str, ...]
    stackability: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.atom_id:
            raise ScorerSurfacePlanError("atom_id is required")
        if self.expected_added_bytes < 0:
            raise ScorerSurfacePlanError(f"{self.atom_id}: expected_added_bytes must be >= 0")
        if not self.pass_sequence:
            raise ScorerSurfacePlanError(f"{self.atom_id}: pass_sequence is required")


def default_operating_point() -> OperatingPoint:
    """Return the current PR106 CUDA frontier used by the handoff."""

    return OperatingPoint(
        label="pr106_latent_sidecar_r2_pr101_grammar",
        device_axis="contest_cuda",
        score=0.2066181354574151,
        archive_bytes=186_780,
        archive_sha256="c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383",
        seg_dist=0.00064260,
        pose_dist=0.00003236,
    )


def default_surface_atoms() -> tuple[SurfaceAtomFamily, ...]:
    """Return default score-surface atom families ranked by local economics."""

    return (
        SurfaceAtomFamily(
            atom_id="pixel_lsb_threshold_probe",
            surface="decoded_rgb_pixels",
            perturbation_space="single-pixel/channel +/-1 uint8 atoms near SegNet/PoseNet threshold cliffs",
            expected_added_bytes=768,
            expected_seg_delta=-2.0e-6,
            expected_pose_delta=-1.1e-6,
            packetir_stream="residual_program_or_score_table_sidecar",
            materialization_path="run-length / coordinate-coded residual program consumed by inflate",
            primary_solver="beam search over scorer-response table, then branch-and-bound by byte cost",
            local_parallelism_shape="frame shards x channel shards on local macOS CPU; exact axes rerun on Linux/T4",
            pass_sequence=(
                "retain raw decoded frames and scorer outputs",
                "controlled +/-1 pixel/channel probes on hard frames",
                "rank atoms by measured delta score per encoded byte",
                "compress selected runs through PacketIR residual stream",
                "no-op proof and paired CPU/CUDA exact eval",
            ),
            guardrails=(
                "proxy rows are not score evidence",
                "must not alter upstream scorer",
                "must record raw-output aggregate SHA before and after",
                "materialized bytes must be charged and consumed by inflate",
            ),
            stackability="high after PR106 R2 because it targets scorer cliffs rather than rate-only recoding",
            notes="This is the most direct version of engineering against pixels and auth eval.",
        ),
        SurfaceAtomFamily(
            atom_id="frame_channel_affine_shake",
            surface="decoded_rgb_frames",
            perturbation_space="per-frame/per-channel bias, gain, and clipped affine micro-updates",
            expected_added_bytes=384,
            expected_seg_delta=-8.0e-7,
            expected_pose_delta=-8.0e-7,
            packetir_stream="small frame-channel control table",
            materialization_path="fixed-width per-frame channel table with runtime clamp",
            primary_solver="coordinate descent -> CMA-ES/Optuna only after byte-closed controls exist",
            local_parallelism_shape="candidate vectors batched on local CPU, MPS only for advisory curve finding",
            pass_sequence=(
                "start at zero-control identity",
                "shake frame/channel offsets under exact raw-frame parity checks",
                "fit low-dimensional trust region from accepted probes",
                "materialize compact control table",
                "exact CPU/CUDA four-cell matrix",
            ),
            guardrails=(
                "do not hardcode arbitrary constants without candidate table provenance",
                "CPU and CUDA axes must remain separate",
                "identity control must be byte-consumed but output-equivalent",
            ),
            stackability="medium-high; composes with residual pixels if runtime order is explicit",
        ),
        SurfaceAtomFamily(
            atom_id="segnet_boundary_run_repair",
            surface="SegNet boundary pixels",
            perturbation_space="class-boundary run atoms and low-margin connected components",
            expected_added_bytes=1_536,
            expected_seg_delta=-1.8e-5,
            expected_pose_delta=2.0e-7,
            packetir_stream="boundary residual run stream",
            materialization_path="class/run-length residual sidecar decoded after base renderer",
            primary_solver="water-fill by boundary margin score benefit per charged byte",
            local_parallelism_shape="connected components can be generated frame-parallel on CPU",
            pass_sequence=(
                "derive boundary and low-margin maps",
                "shake run-length atoms instead of individual pixels",
                "block atoms that regress pose beyond local budget",
                "encode selected runs with categorical/range coding",
                "exact eval only after runtime consumes the run stream",
            ),
            guardrails=(
                "mask-only wins need pose regeneration/diagnostics",
                "component collapse risk must be tracked before dispatch",
                "no scorer-model import inside inflate",
            ),
            stackability="high with categorical mask and wavelet residual lanes",
        ),
        SurfaceAtomFamily(
            atom_id="pose_sensitive_luma_microfield",
            surface="PoseNet-sensitive luma/edge field",
            perturbation_space="low-rank luma fields, tiny y-shifts, ego-motion aligned edge corrections",
            expected_added_bytes=2_048,
            expected_seg_delta=2.0e-7,
            expected_pose_delta=-4.8e-6,
            packetir_stream="low_rank_luma_or_motion_residual_stream",
            materialization_path="basis coefficients plus deterministic runtime renderer",
            primary_solver="active-subspace search with pose-first Lagrangian guard",
            local_parallelism_shape="basis projection CPU batches; final exact CUDA required",
            pass_sequence=(
                "compute pose-sensitive frame clusters",
                "shake low-rank luma and y-shift coefficients",
                "fit compact basis under byte budget",
                "compile coefficients into PacketIR stream",
                "exact CPU/CUDA matrix to expose device-axis drift",
            ),
            guardrails=(
                "MPS is advisory only",
                "must retain loader/scorer device metadata",
                "runtime must be deterministic and scorer-free",
            ),
            stackability="medium; interacts with yshift and LRL1 sidechannels",
        ),
        SurfaceAtomFamily(
            atom_id="wavelet_foveated_residual_stack",
            surface="multi-resolution frame residual",
            perturbation_space="sparse wavelet coefficients weighted by foveation and scorer margin",
            expected_added_bytes=3_072,
            expected_seg_delta=-9.0e-6,
            expected_pose_delta=-2.0e-6,
            packetir_stream="wavelet_coeff_residual_stream",
            materialization_path="deterministic inverse-DWT residual runtime with conformance vectors",
            primary_solver="group lasso / ADMM water-fill under exact byte slope",
            local_parallelism_shape="coefficient blocks are embarrassingly parallel on local CPU",
            pass_sequence=(
                "build scorer-margin weighted coefficient candidates",
                "shake coefficients by group with no-op controls",
                "solve sparse allocation by byte-normalized gain",
                "encode coefficients with range/ANS or split Brotli",
                "exact eval after conformance and packet closure",
            ),
            guardrails=(
                "native inverse-DWT ports need byte-for-byte vectors",
                "do not promote advisory macOS CPU curves",
                "record transform ordering relative to other residual streams",
            ),
            stackability="high as a non-HNeRV residual program once runtime is closed",
        ),
        SurfaceAtomFamily(
            atom_id="renderer_training_score_surface_loop",
            surface="training loop over rendered RGB frames",
            perturbation_space="score-aware renderer training with eval-roundtrip, YUV6 gradient reachability, and packet export in loop",
            expected_added_bytes=0,
            expected_seg_delta=-1.2e-4,
            expected_pose_delta=-8.0e-6,
            packetir_stream="trained_renderer_weights_and_latents",
            materialization_path="export-first PacketIR compiler in the training inner loop",
            primary_solver="IB-Lagrangian training with T8/T20 losses and entropy pressure",
            local_parallelism_shape="local CPU smoke/gradcheck; GPU provider only after packet compiler gate",
            pass_sequence=(
                "prove gradients reach RGB->YUV6->SegNet/PoseNet surrogate",
                "simulate eval roundtrip during training",
                "emit archive candidate every validation interval",
                "select by exact packet score proxy with no research-only auth eval",
                "exact CUDA/CPU adjudication of exported packet",
            ),
            guardrails=(
                "no representation lane promotion without archive grammar",
                "no auth eval on non-exportable variants",
                "training signal must be score-domain, not raw rel_err only",
            ),
            stackability="highest; this is the route out of the local PR106 basin",
        ),
    )


def build_scorer_surface_shaking_plan(
    *,
    operating_point: OperatingPoint | None = None,
    atoms: tuple[SurfaceAtomFamily, ...] | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic no-score scorer-surface shaking plan."""

    point = operating_point or default_operating_point()
    families = atoms or default_surface_atoms()
    rows = [_atom_row(point, atom) for atom in families]
    rows.sort(
        key=lambda row: (
            row["predicted_score_delta"],  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
            -row["benefit_per_added_byte"],
            row["atom_id"],
        )
    )
    if max_rows is not None:
        if max_rows <= 0:
            raise ScorerSurfacePlanError("max_rows must be positive")
        rows = rows[:max_rows]
    cpu_count = os.cpu_count() or 1
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_no_score",
        "operating_point": point.as_dict(),
        "local_execution": {
            "machine_role": "local_macos_cpu_advisory_search",
            "cpu_count": cpu_count,
            "recommended_worker_count": max(1, min(cpu_count, 12)),
            "mps_policy": "advisory_curve_finding_only_never_auth_eval",
            "parallelism_model": "parallelize across frames, channels, atoms, and candidate batches; keep exact eval on explicit CPU/CUDA axes",
        },
        "ranked_atoms": rows,
        "top_atom_ids": [row["atom_id"] for row in rows[: min(5, len(rows))]],
        "default_pass_pipeline": [
            "identity_packet_and_raw_output_custody",
            "scorer_surface_feature_extraction",
            "local_parallel_atom_shaking",
            "byte_normalized_solver_selection",
            "PacketIR_materialization",
            "no_op_and_consumption_proof",
            "paired_CPU_CUDA_exact_eval_after_lane_claim",
        ],
        "integration_hooks": {
            "roadmap_status": "tools/build_frontier_roadmap_status.py includes this plan by default",
            "packet_compiler": "tac.packet_section_transform / tac.analysis.hnerv_packet_sections",
            "candidate_selection": "field-meta selector may ingest materialized packet manifests only",
            "continual_learning": "exact eval results, not proxy rows, update posterior state",
        },
        "dispatch_blockers": [
            "planning_only_no_archive_emitted",
            "requires_PacketIR_materialized_archive",
            "requires_runtime_consumption_proof",
            "requires_no_op_control",
            "requires_level2_lane_dispatch_claim",
            "requires_exact_cuda_and_paired_cpu_review",
        ],
    }


def render_markdown(plan: dict[str, Any]) -> str:
    """Render a compact operator-facing markdown summary."""

    point = plan["operating_point"]
    slopes = point["score_slopes"]
    lines = [
        "# Scorer Surface Shaking Plan",
        "",
        "Planning-only local search plan. It does not claim scores or dispatch work.",
        "",
        f"- operating_point: `{point['label']}`",
        f"- device_axis: `{point['device_axis']}`",
        f"- score: `{point['score']}`",
        f"- archive_bytes: `{point['archive_bytes']}`",
        f"- d_score_d_byte: `{slopes['d_score_d_byte']:.12g}`",
        f"- d_score_d_seg: `{slopes['d_score_d_seg']:.12g}`",
        f"- d_score_d_pose: `{slopes['d_score_d_pose']:.12g}`",
        f"- recommended_worker_count: `{plan['local_execution']['recommended_worker_count']}`",
        "",
        "| atom | predicted score delta | added bytes | benefit/byte | stream | solver |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in plan["ranked_atoms"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_md(row['atom_id'])}`",
                    f"`{row['predicted_score_delta']:.9f}`",  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
                    f"`{row['expected_added_bytes']}`",
                    f"`{row['benefit_per_added_byte']:.12g}`",
                    _md(row["packetir_stream"]),
                    _md(row["primary_solver"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Default Pass Pipeline",
            "",
            *[f"- `{step}`" for step in plan["default_pass_pipeline"]],
            "",
            "## Dispatch Blockers",
            "",
            *[f"- `{blocker}`" for blocker in plan["dispatch_blockers"]],
            "",
        ]
    )
    return "\n".join(lines)


def _atom_row(point: OperatingPoint, atom: SurfaceAtomFamily) -> dict[str, Any]:
    rate_delta = atom.expected_added_bytes * point.byte_slope
    seg_delta = 100.0 * atom.expected_seg_delta
    next_pose = max(point.pose_dist + atom.expected_pose_delta, 0.0)
    pose_delta = math.sqrt(10.0 * next_pose) - math.sqrt(10.0 * point.pose_dist)
    total_delta = rate_delta + seg_delta + pose_delta
    benefit = max(0.0, -total_delta)
    added = max(1, atom.expected_added_bytes)
    return {  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
        "atom_id": atom.atom_id,
        "surface": atom.surface,
        "perturbation_space": atom.perturbation_space,
        "expected_added_bytes": atom.expected_added_bytes,
        "expected_seg_delta": atom.expected_seg_delta,
        "expected_pose_delta": atom.expected_pose_delta,
        "score_delta_components": {
            "rate": rate_delta,
            "seg": seg_delta,
            "pose": pose_delta,
        },
        "predicted_score_delta": total_delta,
        "rate_positive_if_components_hold": total_delta < 0.0,
        "benefit_per_added_byte": benefit / added,
        "packetir_stream": atom.packetir_stream,
        "materialization_path": atom.materialization_path,
        "primary_solver": atom.primary_solver,
        "local_parallelism_shape": atom.local_parallelism_shape,
        "pass_sequence": list(atom.pass_sequence),
        "guardrails": list(atom.guardrails),
        "stackability": atom.stackability,
        "notes": atom.notes,
        "promotion_gates": [
            "PacketIR section or runtime stream declared",
            "archive bytes and SHA-256 recorded",
            "inflate consumes the charged bytes",
            "identity/no-op control passes",
            "candidate-specific pre-submission compliance passes",
            "paired CPU/CUDA exact eval reviewed before status change",
        ],
    }


def _md(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


__all__ = [
    "OperatingPoint",
    "RATE_SCORE_PER_BYTE",
    "SCHEMA",
    "ScorerSurfacePlanError",
    "SurfaceAtomFamily",
    "build_scorer_surface_shaking_plan",
    "default_operating_point",
    "default_surface_atoms",
    "render_markdown",
]
