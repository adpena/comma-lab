#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score-geometry dispatch advisor: per-candidate axis-priority recommendations.

Closes the gap from the May 4 race postmortem where we had every primitive
needed but lacked an analytical advisor that translates a candidate's
3-axis position (d_seg, d_pose, B) into "which axis to attack next" given
the fixed score geometry.

The contest objective S = 100*d_seg + sqrt(10*d_pose) + 25*B/N is convex
on each individual axis but **non-convex jointly** because the pose term
is concave (sqrt). The marginal value-per-unit on the pose axis depends
on the operating point: at d_pose=0.18 the SegNet axis dominates 77x,
but at PR106's d_pose=3.4e-5 the pose axis dominates 2.71x. Importance
flips at d_pose=2.5e-4 (closed-form from
``tac.score_geometry.importance_flip_threshold``).

This advisor:

  1. Reads a 3-axis Pareto frontier JSON (from
     ``tools/contest_score_pareto_3axis.py``) OR a single
     (d_seg, d_pose, B) tuple
  2. For each candidate, computes:
       - operating regime (seg-dominated / pose-dominated)
       - per-axis marginal (score-points per unit of axis improvement)
       - cost-effectiveness in score-points-per-byte for each axis
  3. Emits a per-candidate axis priority recommendation +
     score-target reverse-curve (e.g., "to hit 0.190 you need
     d_pose < X given current d_seg + B")

Pure CPU + math + ``tac.score_geometry``. No torch, no scorer load. The
advice is **planning evidence**, not contest-score claims.

Usage::

    .venv/bin/python tools/dispatch_advisor.py inspect \\
        --d-seg 6.7e-4 --d-pose 3.4e-5 --archive-bytes 178258 \\
        --target-score 0.190

    .venv/bin/python tools/dispatch_advisor.py from-pareto \\
        --pareto-json reports/pareto_3axis.json \\
        --output reports/dispatch_advice.json

CLAUDE.md compliance: pure-CPU planning tool. Outputs explicitly tagged
``[CPU-prep planning-only]``. Never emits contest scores; only relative
analytics.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.score_geometry import (  # noqa: E402
    contest_score,
    equal_score_curve_archive_bytes,
    equal_score_curve_d_pose,
    importance_flip_threshold,
    information_floor,
    operating_regime,
    predict_cpu_axis_marginals,
    score_decomposition,
    score_gradient,
)
from tac.optimization.cuda_cpu_axis_calibration import (  # noqa: E402
    CudaCpuCalibration,
)
from tac.score_geometry_floor_explorer import (  # noqa: E402
    TechniqueResult,
    rank_technique_results,
)

TOOL_NAME = "tools/dispatch_advisor.py"
SCHEMA_VERSION = "dispatch_advisor.v1"
EVIDENCE_GRADE = "[CPU-prep planning-only]"


@dataclass(frozen=True)
class AxisPriority:
    """Recommended axis to attack next, with quantitative justification."""

    name: str  # "seg" | "pose" | "bytes"
    marginal_value: float  # dS/d(axis) at the operating point
    rationale: str
    estimated_effort_class: str  # "tight" | "moderate" | "wide_open"


@dataclass
class CandidateAdvice:
    """Per-candidate dispatch advice."""

    label: str
    d_seg: float
    d_pose: float
    archive_bytes: int
    score: float
    score_decomposition: dict[str, float]
    operating_regime_summary: dict[str, object]
    axis_priorities: list[dict[str, object]]
    target_score_curves: dict[str, object]
    floor_technique_priorities: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Dual-axis (CUDA + CPU) advice — added 2026-05-08 per CLAUDE.md
    # "Submission auth eval — BOTH CPU AND CUDA". Default empty for
    # backward compatibility.
    cpu_axis_marginals: dict[str, object] = field(default_factory=dict)
    cpu_axis_priorities: list[dict[str, object]] = field(default_factory=list)
    predicted_cuda_score: float | None = None
    predicted_cpu_score: float | None = None
    predicted_cpu_score_lo: float | None = None
    predicted_cpu_score_hi: float | None = None
    predicted_cpu_score_calibration: str = ""


def load_floor_technique_results(path: Path) -> tuple[int | None, list[TechniqueResult]]:
    """Load planning-only floor technique rows from a JSON manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("techniques", payload.get("results", []))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected techniques/results list")
    baseline_raw = payload.get("baseline_floor_bytes")
    baseline_floor_bytes = int(baseline_raw) if baseline_raw is not None else None
    results: list[TechniqueResult] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: technique row {index} is not an object")
        try:
            name = str(row["name"])
            bytes_floor = int(row["bytes_floor"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: technique row {index} missing name/bytes_floor") from exc
        if bytes_floor <= 0:
            raise ValueError(f"{path}: technique row {index} bytes_floor must be positive")
        score_at_floor = row.get("score_at_floor_zero_distortion")
        results.append(TechniqueResult(
            name=name,
            description=str(row.get("description", name)),
            bytes_floor=bytes_floor,
            bytes_savings_vs_baseline=int(row.get("bytes_savings_vs_baseline", 0)),
            score_at_floor_zero_distortion=(
                float(score_at_floor)
                if score_at_floor is not None
                else contest_score(0.0, 0.0, bytes_floor)
            ),
            distortion_risk=str(row.get("distortion_risk", "unknown")),
            notes=[str(note) for note in row.get("notes", [])],
        ))
    return baseline_floor_bytes, results


def rank_floor_techniques_for_candidate(
    *,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    target_score: float | None,
    baseline_floor_bytes: int,
    technique_results: list[TechniqueResult],
) -> list[dict[str, object]]:
    """Attach closed-form floor techniques to the dispatch advisor safely."""
    ranked = rank_technique_results(
        baseline_floor_bytes=baseline_floor_bytes,
        results=technique_results,
    )
    rows: list[dict[str, object]] = []
    for result in ranked:
        score_holding_distortion = contest_score(d_seg, d_pose, result.bytes_floor)
        rows.append({
            "name": result.name,
            "description": result.description,
            "bytes_floor": result.bytes_floor,
            "bytes_savings_vs_floor_baseline": result.bytes_savings_vs_baseline,
            "archive_byte_delta_if_rate_only": result.bytes_floor - archive_bytes,
            "score_at_floor_zero_distortion": result.score_at_floor_zero_distortion,
            "score_holding_current_distortion": score_holding_distortion,
            "score_gap_to_target_holding_current_distortion": (
                score_holding_distortion - target_score
                if target_score is not None
                else None
            ),
            "distortion_risk": result.distortion_risk,
            "notes": result.notes,
            "score_claim": False,
            "evidence_grade": EVIDENCE_GRADE,
            "dispatch_blockers": [
                "closed_form_floor_only",
                "no_archive_candidate",
                "no_distortion_validation",
                "missing_exact_cuda_auth_eval",
            ],
        })
    return rows


def _axis_effort_class(
    axis: str,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
) -> str:
    """Return a coarse "how much room is left" tag for the axis.

    These are heuristic anchors aligned with leaderboard reality:
      - seg: PR106 frontier sits at d_seg ~ 6.7e-4. The renderer's
        SegNet ceiling is ~ 1e-4 (saturation observed). Below 5e-4 we
        call "tight" (architectural work needed); above 1e-3 "wide_open".
      - pose: PR106 frontier sits at d_pose ~ 3.4e-5. There's no
        observed hardware floor; below 1e-5 we call "tight"; above
        1e-3 "wide_open".
      - bytes: information floor is fixed. Below 150 KB we call "tight";
        above 250 KB "wide_open".
    """
    if axis == "seg":
        if d_seg < 5e-4:
            return "tight"
        if d_seg > 1e-3:
            return "wide_open"
        return "moderate"
    if axis == "pose":
        if d_pose < 1e-5:
            return "tight"
        if d_pose > 1e-3:
            return "wide_open"
        return "moderate"
    if axis == "bytes":
        if archive_bytes < 150_000:
            return "tight"
        if archive_bytes > 250_000:
            return "wide_open"
        return "moderate"
    return "unknown"


def advise_candidate(
    *,
    label: str,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    target_score: float | None = None,
    floor_technique_results: list[TechniqueResult] | None = None,
    floor_baseline_bytes: int | None = None,
    architecture_class: str = "hnerv",
    score_axis: str = "cuda",
) -> CandidateAdvice:
    """Build a CandidateAdvice for a single (d_seg, d_pose, B) point.

    Mathematical rigor:
      - score and decomposition come from the canonical contest formula
      - per-axis marginals come from analytic partial derivatives
      - regime classification comes from comparison vs the closed-form flip
      - reverse curves use the inverse of the contest formula
    """
    score = contest_score(d_seg, d_pose, archive_bytes)
    decomp = score_decomposition(d_seg, d_pose, archive_bytes)
    regime = operating_regime(d_pose)
    grad = score_gradient(d_seg, d_pose)

    # Build axis priorities sorted by marginal score impact per unit
    # axis improvement. The "byte" axis is normalized to per-1000-byte
    # so the numbers are comparable across axes.
    seg_priority = AxisPriority(
        name="seg",
        marginal_value=grad.d_seg,
        rationale=(
            f"dS/d(d_seg) = {grad.d_seg:.2f} (constant). Improving d_seg by "
            f"1e-4 reduces score by {grad.d_seg * 1e-4:.5f}."
        ),
        estimated_effort_class=_axis_effort_class("seg", d_seg, d_pose, archive_bytes),
    )
    pose_priority = AxisPriority(
        name="pose",
        marginal_value=grad.d_pose if not (grad.d_pose != grad.d_pose or grad.d_pose == float("inf")) else float("inf"),
        rationale=(
            f"dS/d(d_pose) = {grad.d_pose:.2f} at this operating point. "
            f"Improving d_pose by 1e-5 reduces score by ~{grad.d_pose * 1e-5:.5f} "
            f"(linear approximation; pose term is sqrt so the actual "
            f"reduction is slightly larger for the same delta)."
        ),
        estimated_effort_class=_axis_effort_class("pose", d_seg, d_pose, archive_bytes),
    )
    bytes_priority = AxisPriority(
        name="bytes",
        marginal_value=grad.d_bytes * 1000,  # per 1000 bytes for readability
        rationale=(
            f"dS/dB = {grad.d_bytes:.3e} per byte (constant {grad.d_bytes * 1000:.5f} per "
            f"1000 bytes). Saving 1 KB reduces score by {grad.d_bytes * 1000:.5f}."
        ),
        estimated_effort_class=_axis_effort_class("bytes", d_seg, d_pose, archive_bytes),
    )
    priorities = [seg_priority, pose_priority, bytes_priority]
    # Sort by absolute marginal value (highest first); but note pose can
    # be inf at d_pose=0
    priorities.sort(
        key=lambda p: (-1 if p.marginal_value == float("inf") else -p.marginal_value),
    )

    # Target-score reverse curves: if a target was supplied, compute
    # what each axis would need to be (holding the others fixed) to hit it
    target_curves: dict[str, object] = {}
    if target_score is not None:
        required_pose = equal_score_curve_d_pose(target_score, d_seg, archive_bytes)
        required_bytes = equal_score_curve_archive_bytes(target_score, d_seg, d_pose)
        target_curves = {
            "target_score": target_score,
            "current_score": score,
            "score_gap": score - target_score,
            "to_reach_target_holding_seg_and_bytes": (
                {
                    "required_d_pose": required_pose,
                    "current_d_pose": d_pose,
                    "improvement_factor": (
                        d_pose / required_pose if required_pose and required_pose > 0 else None
                    ),
                    "feasible": required_pose is not None,
                }
            ),
            "to_reach_target_holding_seg_and_pose": (
                {
                    "required_archive_bytes": required_bytes,
                    "current_archive_bytes": archive_bytes,
                    "byte_savings_required": (
                        archive_bytes - required_bytes if required_bytes is not None else None
                    ),
                    "feasible": required_bytes is not None,
                }
            ),
        }

    notes = [
        regime.advice,
        f"information floor at this byte budget = {information_floor(archive_bytes):.5f} "
        f"(strict lower bound; achievable only at d_seg=d_pose=0).",
    ]
    if score < information_floor(archive_bytes):
        # Sanity check; should never happen.
        notes.append(
            f"WARNING: score {score:.5f} is below information floor "
            f"{information_floor(archive_bytes):.5f}; check inputs."
        )

    floor_priorities: list[dict[str, object]] = []
    if floor_technique_results:
        baseline_bytes = floor_baseline_bytes or archive_bytes
        floor_priorities = rank_floor_techniques_for_candidate(
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=archive_bytes,
            target_score=target_score,
            baseline_floor_bytes=baseline_bytes,
            technique_results=floor_technique_results,
        )
        notes.append(
            "Floor technique priorities are closed-form planning signals only; "
            "they do not imply a score-affecting archive exists."
        )

    # Dual-axis (CUDA + CPU) extension — added 2026-05-08.
    # Always compute the CPU-axis marginals + predicted CPU score band so
    # the operator can see both axes regardless of how `score_axis` is set.
    cpu_marginals = predict_cpu_axis_marginals(
        d_seg_cuda=d_seg,
        d_pose_cuda=d_pose,
        archive_class=architecture_class,
    )
    # Build axis priorities from the CPU marginals.
    cpu_seg_priority = AxisPriority(
        name="seg",
        marginal_value=cpu_marginals["seg_marginal"],
        rationale=(
            f"dS_cpu/d(d_seg_cpu) = {cpu_marginals['seg_marginal']:.2f} "
            f"(constant). CPU d_seg is {cpu_marginals['cpu_d_seg']:.4e} "
            f"(rebased from CUDA via R_seg ≈ 1.17)."
        ),
        estimated_effort_class=_axis_effort_class("seg", d_seg, d_pose, archive_bytes),
    )
    cpu_pose_priority = AxisPriority(
        name="pose",
        marginal_value=(
            cpu_marginals["pose_marginal"]
            if cpu_marginals["pose_marginal"] != float("inf")
            else float("inf")
        ),
        rationale=(
            f"dS_cpu/d(d_pose_cpu) = {cpu_marginals['pose_marginal']:.2f} "
            f"at the rebased CPU operating point (d_pose_cpu="
            f"{cpu_marginals['cpu_d_pose']:.4e}). The pose floor saturation "
            f"means d_pose_cuda below ~1.4e-4 maps to d_pose_cpu near zero."
        ),
        estimated_effort_class=_axis_effort_class("pose", d_seg, d_pose, archive_bytes),
    )
    cpu_bytes_priority = AxisPriority(
        name="bytes",
        marginal_value=cpu_marginals["bytes_marginal"] * 1000,
        rationale=(
            f"dS/dB = {cpu_marginals['bytes_marginal']:.3e} per byte "
            "(identical CUDA and CPU)."
        ),
        estimated_effort_class=_axis_effort_class("bytes", d_seg, d_pose, archive_bytes),
    )
    cpu_priorities = [cpu_seg_priority, cpu_pose_priority, cpu_bytes_priority]
    cpu_priorities.sort(
        key=lambda p: (-1 if p.marginal_value == float("inf") else -p.marginal_value),
    )
    # Predicted CPU score band (from the calibration helper).
    cal = CudaCpuCalibration(architecture_class=architecture_class)
    cpu_band = cal.predict_cpu_from_cuda(
        score,
        d_pose_cuda=d_pose,
        d_seg_cuda=d_seg,
        archive_bytes=archive_bytes,
    )
    return CandidateAdvice(
        label=label,
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=archive_bytes,
        score=score,
        score_decomposition={
            "seg_term": decomp.seg_term,
            "pose_term": decomp.pose_term,
            "rate_term": decomp.rate_term,
            "total": decomp.total,
            "fractions": list(decomp.fractions),
        },
        operating_regime_summary={
            "d_pose": regime.d_pose,
            "flip_threshold": regime.flip_threshold,
            "seg_dominates": regime.seg_dominates,
            "pose_dominates": regime.pose_dominates,
            "crossover_distance_log10": regime.crossover_distance_log10,
            "marginal_ratio_seg_over_pose": regime.marginal_ratio_seg_over_pose,
            "advice": regime.advice,
        },
        axis_priorities=[asdict(p) for p in priorities],
        target_score_curves=target_curves,
        floor_technique_priorities=floor_priorities,
        notes=notes,
        cpu_axis_marginals=cpu_marginals,
        cpu_axis_priorities=[asdict(p) for p in cpu_priorities],
        predicted_cuda_score=score,
        predicted_cpu_score=cpu_band.score_point,
        predicted_cpu_score_lo=cpu_band.score_lo,
        predicted_cpu_score_hi=cpu_band.score_hi,
        predicted_cpu_score_calibration=cpu_band.calibration_quality,
    )


def advise_pareto_json(
    *,
    pareto_json_path: Path,
    target_score: float | None = None,
    floor_technique_results: list[TechniqueResult] | None = None,
    floor_baseline_bytes: int | None = None,
) -> list[CandidateAdvice]:
    """Run the advisor over every candidate in a 3-axis Pareto JSON dump."""
    payload = json.loads(pareto_json_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(
            f"{pareto_json_path}: expected 'candidates' to be a list, got {type(candidates).__name__}"
        )
    advice_rows: list[CandidateAdvice] = []
    for candidate in candidates:
        # Required fields per Candidate dataclass
        label = str(candidate.get("label", "<unknown>"))
        try:
            d_seg = float(candidate["d_seg"])
            d_pose = float(candidate["d_pose"])
            archive_bytes = int(candidate["archive_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{pareto_json_path}: candidate {label} missing required fields: {exc}"
            ) from exc
        advice = advise_candidate(
            label=label,
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=archive_bytes,
            target_score=target_score,
            floor_technique_results=floor_technique_results,
            floor_baseline_bytes=floor_baseline_bytes,
        )
        advice_rows.append(advice)
    return advice_rows


def render_advice_summary(advice: CandidateAdvice) -> str:
    """Human-readable one-page summary for the operator."""
    lines: list[str] = []
    lines.append(f"=== Dispatch Advice: {advice.label} ===")
    lines.append(f"Operating point: d_seg={advice.d_seg:.4e}, d_pose={advice.d_pose:.4e}, "
                 f"B={advice.archive_bytes:,} bytes")
    lines.append(f"Contest score: {advice.score:.5f}  "
                 f"(seg={advice.score_decomposition['seg_term']:.5f} "
                 f"+ pose={advice.score_decomposition['pose_term']:.5f} "
                 f"+ rate={advice.score_decomposition['rate_term']:.5f})")
    lines.append("")
    lines.append("Operating regime:")
    lines.append(f"  {advice.operating_regime_summary['advice']}")
    lines.append(f"  Importance-flip threshold: d_pose = {importance_flip_threshold():.2e}")
    lines.append("")
    lines.append("Axis priorities (highest marginal first):")
    for i, priority in enumerate(advice.axis_priorities, 1):
        lines.append(f"  {i}. {priority['name'].upper()} "
                     f"(effort: {priority['estimated_effort_class']})")
        lines.append(f"     {priority['rationale']}")
    if advice.target_score_curves:
        tsc = advice.target_score_curves
        lines.append("")
        lines.append(f"To reach target score {tsc['target_score']:.5f} (gap {tsc['score_gap']:.5f}):")
        pose_curve = tsc["to_reach_target_holding_seg_and_bytes"]
        bytes_curve = tsc["to_reach_target_holding_seg_and_pose"]
        if pose_curve["feasible"]:
            lines.append(f"  Pose-only path: improve d_pose to "
                         f"{pose_curve['required_d_pose']:.4e} "
                         f"(currently {pose_curve['current_d_pose']:.4e}; "
                         f"improvement factor {pose_curve['improvement_factor']:.2f}x)")
        else:
            lines.append("  Pose-only path: INFEASIBLE (seg+rate already exceed target)")
        if bytes_curve["feasible"]:
            lines.append(f"  Bytes-only path: shrink archive to "
                         f"{bytes_curve['required_archive_bytes']:,} bytes "
                         f"(currently {bytes_curve['current_archive_bytes']:,}; "
                         f"saving {bytes_curve['byte_savings_required']:,} bytes)")
        else:
            lines.append("  Bytes-only path: INFEASIBLE (seg+pose already exceed target)")
    if advice.floor_technique_priorities:
        lines.append("")
        lines.append("Closed-form floor technique priorities (planning-only):")
        for i, row in enumerate(advice.floor_technique_priorities[:5], 1):
            lines.append(
                f"  {i}. {row['name']}: floor {row['bytes_floor']:,} bytes, "
                f"savings {row['bytes_savings_vs_floor_baseline']:,} vs floor baseline, "
                f"risk {row['distortion_risk']}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Advise on a single candidate point")
    p_inspect.add_argument("--label", default="adhoc", help="Optional label for the candidate")
    p_inspect.add_argument("--d-seg", type=float, required=True)
    p_inspect.add_argument("--d-pose", type=float, required=True)
    p_inspect.add_argument("--archive-bytes", type=int, required=True)
    p_inspect.add_argument("--target-score", type=float, default=None)
    p_inspect.add_argument("--output", type=Path, default=None,
                           help="Optional JSON output path")
    p_inspect.add_argument("--floor-techniques-json", type=Path, default=None,
                           help="Optional floor-technique manifest from score_geometry_floor_explorer")
    p_inspect.add_argument("--summary-text", action="store_true",
                           help="Print human-readable summary instead of JSON")

    p_pareto = sub.add_parser("from-pareto",
                              help="Advise every candidate in a 3-axis Pareto JSON")
    p_pareto.add_argument("--pareto-json", type=Path, required=True)
    p_pareto.add_argument("--target-score", type=float, default=None)
    p_pareto.add_argument("--floor-techniques-json", type=Path, default=None,
                          help="Optional floor-technique manifest from score_geometry_floor_explorer")
    p_pareto.add_argument("--output", type=Path, required=True,
                          help="JSON output path with all advice rows")

    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        floor_baseline_bytes = None
        floor_results = None
        if args.floor_techniques_json:
            floor_baseline_bytes, floor_results = load_floor_technique_results(
                args.floor_techniques_json
            )
        advice = advise_candidate(
            label=args.label,
            d_seg=args.d_seg,
            d_pose=args.d_pose,
            archive_bytes=args.archive_bytes,
            target_score=args.target_score,
            floor_technique_results=floor_results,
            floor_baseline_bytes=floor_baseline_bytes,
        )
        if args.summary_text:
            print(render_advice_summary(advice))
        else:
            payload = {
                "schema": SCHEMA_VERSION,
                "tool": TOOL_NAME,
                "evidence_grade": EVIDENCE_GRADE,
                "advice": asdict(advice),
            }
            output = json.dumps(payload, indent=2, sort_keys=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output, encoding="utf-8")
            print(output)
        return 0

    if args.cmd == "from-pareto":
        floor_baseline_bytes = None
        floor_results = None
        if args.floor_techniques_json:
            floor_baseline_bytes, floor_results = load_floor_technique_results(
                args.floor_techniques_json
            )
        advice_rows = advise_pareto_json(
            pareto_json_path=args.pareto_json,
            target_score=args.target_score,
            floor_technique_results=floor_results,
            floor_baseline_bytes=floor_baseline_bytes,
        )
        payload = {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "evidence_grade": EVIDENCE_GRADE,
            "input_pareto_json": str(args.pareto_json),
            "n_candidates": len(advice_rows),
            "target_score": args.target_score,
            "floor_techniques_json": (
                str(args.floor_techniques_json)
                if args.floor_techniques_json
                else None
            ),
            "advice_rows": [asdict(a) for a in advice_rows],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote {len(advice_rows)} advice rows to {args.output}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
