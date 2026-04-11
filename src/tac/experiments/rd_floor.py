"""Empirical rate/distortion floor analysis.

Core logic extracted from experiments/rate_distortion_floor.py for use via ``tac rd-floor``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT / "reports" / "raw"


@dataclass(frozen=True)
class RunPoint:
    label: str
    summary_path: str
    score: float
    archive_bytes: int
    pose: float
    seg: float
    rate: float


def score_from_terms(*, pose: float, seg: float, rate: float) -> float:
    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * rate


def pose_leverage_at(pose: float) -> float:
    pose = max(pose, 1e-12)
    return 5.0 / math.sqrt(10.0 * pose)


def required_pose_for_target(*, target: float, seg: float, rate: float) -> float:
    remaining = target - 100.0 * seg - 25.0 * rate
    if remaining <= 0.0:
        return 0.0
    return (remaining ** 2) / 10.0


def required_seg_for_target(*, target: float, pose: float, rate: float) -> float:
    remaining = target - math.sqrt(10.0 * pose) - 25.0 * rate
    return remaining / 100.0


def load_summary_points(root: Path) -> list[RunPoint]:
    points: list[RunPoint] = []
    for path in sorted(root.glob("**/*summary.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        required = {
            "track", "current_workflow_score", "current_workflow_archive_bytes",
            "pose_distortion", "seg_distortion", "current_workflow_rate",
        }
        if not required.issubset(payload.keys()):
            continue
        if payload.get("track") != "robust_current":
            continue
        label = path.stem
        copied = payload.get("copied_report_path")
        if isinstance(copied, str):
            label = Path(copied).stem.replace("-current_workflow-cpu-report", "")
        points.append(
            RunPoint(
                label=label,
                summary_path=str(path),
                score=float(payload["current_workflow_score"]),
                archive_bytes=int(payload["current_workflow_archive_bytes"]),
                pose=float(payload["pose_distortion"]),
                seg=float(payload["seg_distortion"]),
                rate=float(payload["current_workflow_rate"]),
            )
        )
    return points


def pareto_frontier(points: Iterable[RunPoint]) -> list[RunPoint]:
    ordered = sorted(points, key=lambda p: (p.archive_bytes, p.score))
    frontier: list[RunPoint] = []
    best_score = math.inf
    for point in ordered:
        if point.score < best_score:
            frontier.append(point)
            best_score = point.score
    return frontier


def select_floor(points: Iterable[RunPoint]) -> RunPoint:
    return min(points, key=lambda p: (p.score, p.archive_bytes))


def build_report(points: list[RunPoint], targets: list[float]) -> dict[str, object]:
    if not points:
        raise ValueError("No scorer-backed summary points found under reports/raw")

    floor = select_floor(points)
    frontier = pareto_frontier(points)
    best_pose = min(points, key=lambda p: p.pose)
    best_seg = min(points, key=lambda p: p.seg)
    best_rate = min(points, key=lambda p: p.rate)

    counterfactual_floor = score_from_terms(
        pose=best_pose.pose, seg=best_seg.seg, rate=best_rate.rate,
    )

    local_leverage = {
        "seg": 100.0,
        "pose": pose_leverage_at(floor.pose),
        "rate": 25.0,
        "seg_vs_pose": 100.0 / pose_leverage_at(floor.pose),
        "seg_vs_rate": 4.0,
    }

    target_requirements: list[dict[str, object]] = []
    for target in targets:
        target_requirements.append({
            "target_score": target,
            "required_pose_if_seg_rate_fixed": required_pose_for_target(
                target=target, seg=floor.seg, rate=floor.rate,
            ),
            "required_seg_if_pose_rate_fixed": required_seg_for_target(
                target=target, pose=floor.pose, rate=floor.rate,
            ),
        })

    return {
        "n_points": len(points),
        "current_floor": {
            "label": floor.label,
            "score": floor.score,
            "archive_bytes": floor.archive_bytes,
            "pose": floor.pose,
            "seg": floor.seg,
            "rate": floor.rate,
        },
        "best_observed_components": {
            "pose": {"label": best_pose.label, "value": best_pose.pose},
            "seg": {"label": best_seg.label, "value": best_seg.seg},
            "rate": {"label": best_rate.label, "value": best_rate.rate},
        },
        "optimistic_non_joint_floor": counterfactual_floor,
        "local_leverage": local_leverage,
        "pareto_frontier": [
            {"label": point.label, "score": point.score, "archive_bytes": point.archive_bytes}
            for point in frontier
        ],
        "target_requirements": target_requirements,
    }


def print_human_report(report: dict[str, object]) -> None:
    floor = report["current_floor"]
    leverage = report["local_leverage"]
    print("Empirical rate / distortion report")
    print("=" * 72)
    print(
        f"Current floor: {floor['label']} | score={floor['score']:.5f} | "
        f"bytes={floor['archive_bytes']:,} | pose={floor['pose']:.8f} | "
        f"seg={floor['seg']:.8f} | rate={floor['rate']:.8f}"
    )
    print()
    print("Local leverage at the floor")
    print(
        f"  SegNet: {leverage['seg']:.3f} points per unit distortion\n"
        f"  PoseNet: {leverage['pose']:.3f} points per unit distortion\n"
        f"  Rate: {leverage['rate']:.3f} points per unit rate\n"
        f"  SegNet vs PoseNet: {leverage['seg_vs_pose']:.2f}x\n"
        f"  SegNet vs Rate: {leverage['seg_vs_rate']:.2f}x"
    )
    print()
    components = report["best_observed_components"]
    print("Best observed components (not jointly attainable)")
    print(
        f"  PoseNet: {components['pose']['value']:.8f} from {components['pose']['label']}\n"
        f"  SegNet: {components['seg']['value']:.8f} from {components['seg']['label']}\n"
        f"  Rate: {components['rate']['value']:.8f} from {components['rate']['label']}"
    )
    print(f"  Optimistic non-joint floor: {report['optimistic_non_joint_floor']:.5f}")
    print()
    print("Target requirements if everything else stayed fixed at the floor")
    for target in report["target_requirements"]:
        print(
            f"  target={target['target_score']:.3f} | "
            f"pose_needed<={target['required_pose_if_seg_rate_fixed']:.8f} | "
            f"seg_needed<={target['required_seg_if_pose_rate_fixed']:.8f}"
        )
    print()
    print("Pareto frontier")
    for point in report["pareto_frontier"]:
        print(f"  {point['label']}: score={point['score']:.5f}, bytes={point['archive_bytes']:,}")
