#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit per-pair PoseNet/SegNet and pixel-error diagnostics for inflated video.

This is a diagnostic XRay tool. It does not build archives, dispatch jobs, or
claim scores. It mirrors the evaluator's pair contract and records enough
per-pair signal to drive selector, film-grain, foveation, and repair work by
component tail instead of aggregate score alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pair_component_error_xray_v1"
NON_PROMOTABLE = {
    "score_claim": False,
    "dispatch_attempted": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


@dataclass(frozen=True)
class PairRow:
    pair_idx: int
    pose_dist: float
    seg_dist: float
    pose_score_contribution: float
    seg_score_contribution: float
    component_score_no_rate: float
    frame0_l1: float
    frame1_l1: float
    frame0_changed_fraction: float
    frame1_changed_fraction: float

    def to_json(self) -> dict[str, Any]:
        return {
            "pair_idx": self.pair_idx,
            "pose_dist": self.pose_dist,
            "seg_dist": self.seg_dist,
            "pose_score_contribution": self.pose_score_contribution,
            "seg_score_contribution": self.seg_score_contribution,
            "component_score_no_rate": self.component_score_no_rate,
            "frame0_l1": self.frame0_l1,
            "frame1_l1": self.frame1_l1,
            "frame0_changed_fraction": self.frame0_changed_fraction,
            "frame1_changed_fraction": self.frame1_changed_fraction,
        }


def _ensure_import_paths(upstream_dir: Path) -> None:
    for path in (REPO_ROOT / "src", upstream_dir, REPO_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _finite_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _score_terms(pose_dist: float, seg_dist: float) -> tuple[float, float, float]:
    pose_term = math.sqrt(max(0.0, 10.0 * pose_dist))
    seg_term = 100.0 * seg_dist
    return pose_term, seg_term, pose_term + seg_term


def _top_rows(rows: list[PairRow], field: str, top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    return [
        row.to_json()
        for row in sorted(rows, key=lambda row: getattr(row, field), reverse=True)[:top_k]
    ]


def _load_scorer(upstream_dir: Path, device: torch.device) -> torch.nn.Module:
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore

    scorer = DistortionNet().eval().to(device=device)
    posenet_path = upstream_dir / "models" / Path(posenet_sd_path).name
    segnet_path = upstream_dir / "models" / Path(segnet_sd_path).name
    scorer.load_state_dicts(posenet_path, segnet_path, device)
    return scorer


def _iter_pairs(
    *,
    inflated_dir: Path,
    upstream_dir: Path,
    video_names_file: Path,
    batch_size: int,
    device: torch.device,
    num_threads: int,
    prefetch_queue_depth: int,
) -> Iterator[tuple[Any, Any]]:
    from frame_utils import AVVideoDataset, DaliVideoDataset, TensorVideoDataset  # type: ignore

    test_video_names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    gt_dataset_cls = DaliVideoDataset if device.type == "cuda" else AVVideoDataset
    ds_gt = gt_dataset_cls(
        test_video_names,
        data_dir=upstream_dir / "videos",
        batch_size=batch_size,
        device=device,
        num_threads=num_threads,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    ds_comp = TensorVideoDataset(
        test_video_names,
        data_dir=inflated_dir,
        batch_size=batch_size,
        device=device,
        num_threads=num_threads,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    ds_gt.prepare_data()
    ds_comp.prepare_data()
    yield from zip(
        torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0),
        torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0),
        strict=True,
    )


def compute_pair_rows(
    *,
    inflated_dir: Path,
    upstream_dir: Path,
    video_names_file: Path,
    device: str,
    batch_size: int,
    max_pairs: int | None,
    num_threads: int,
    prefetch_queue_depth: int,
) -> list[PairRow]:
    _ensure_import_paths(upstream_dir)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    scorer = _load_scorer(upstream_dir, torch_device)

    rows: list[PairRow] = []
    pair_base = 0
    with torch.inference_mode():
        for (_, _, batch_gt), (_, _, batch_comp) in _iter_pairs(
            inflated_dir=inflated_dir,
            upstream_dir=upstream_dir,
            video_names_file=video_names_file,
            batch_size=batch_size,
            device=torch_device,
            num_threads=num_threads,
            prefetch_queue_depth=prefetch_queue_depth,
        ):
            batch_gt = batch_gt.to(torch_device)
            batch_comp = batch_comp.to(torch_device)
            if max_pairs is not None:
                remaining = max_pairs - len(rows)
                if remaining <= 0:
                    break
                batch_gt = batch_gt[:remaining]
                batch_comp = batch_comp[:remaining]
            pose_dist, seg_dist = scorer.compute_distortion(batch_gt, batch_comp)
            abs_delta = (batch_comp.float() - batch_gt.float()).abs()
            changed = abs_delta > 0
            frame0_l1 = abs_delta[:, 0].mean(dim=(1, 2, 3))
            frame1_l1 = abs_delta[:, 1].mean(dim=(1, 2, 3))
            frame0_changed = changed[:, 0].float().mean(dim=(1, 2, 3))
            frame1_changed = changed[:, 1].float().mean(dim=(1, 2, 3))

            for i in range(batch_gt.shape[0]):
                pose = float(pose_dist[i].item())
                seg = float(seg_dist[i].item())
                pose_term, seg_term, component_score = _score_terms(pose, seg)
                rows.append(
                    PairRow(
                        pair_idx=pair_base + i,
                        pose_dist=pose,
                        seg_dist=seg,
                        pose_score_contribution=pose_term,
                        seg_score_contribution=seg_term,
                        component_score_no_rate=component_score,
                        frame0_l1=float(frame0_l1[i].item()),
                        frame1_l1=float(frame1_l1[i].item()),
                        frame0_changed_fraction=float(frame0_changed[i].item()),
                        frame1_changed_fraction=float(frame1_changed[i].item()),
                    )
                )
            pair_base += batch_gt.shape[0]
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def build_report(
    *,
    rows: list[PairRow],
    inflated_dir: Path,
    upstream_dir: Path,
    video_names_file: Path,
    device: str,
    label: str,
    top_k: int,
    archive: Path | None,
) -> dict[str, Any]:
    pose_mean = _finite_mean([row.pose_dist for row in rows])
    seg_mean = _finite_mean([row.seg_dist for row in rows])
    pose_term, seg_term, component_score = _score_terms(pose_mean, seg_mean)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "label": label,
        "evidence_grade": f"diagnostic_pair_component_xray_{device}",
        "device": device,
        "inflated_dir": str(inflated_dir),
        "upstream_dir": str(upstream_dir),
        "video_names_file": str(video_names_file),
        "n_pairs": len(rows),
        "component_summary": {
            "avg_posenet_dist": pose_mean,
            "avg_segnet_dist": seg_mean,
            "pose_score_contribution": pose_term,
            "seg_score_contribution": seg_term,
            "component_score_no_rate": component_score,
        },
        "pixel_summary": {
            "avg_frame0_l1": _finite_mean([row.frame0_l1 for row in rows]),
            "avg_frame1_l1": _finite_mean([row.frame1_l1 for row in rows]),
            "avg_frame0_changed_fraction": _finite_mean([row.frame0_changed_fraction for row in rows]),
            "avg_frame1_changed_fraction": _finite_mean([row.frame1_changed_fraction for row in rows]),
        },
        "top_pairs": {
            "combined": _top_rows(rows, "component_score_no_rate", top_k),
            "pose": _top_rows(rows, "pose_score_contribution", top_k),
            "segnet": _top_rows(rows, "seg_score_contribution", top_k),
            "frame0_l1": _top_rows(rows, "frame0_l1", top_k),
            "frame1_l1": _top_rows(rows, "frame1_l1", top_k),
        },
        "rows": [row.to_json() for row in rows],
        **NON_PROMOTABLE,
    }
    if archive is not None:
        report["archive"] = {
            "path": str(archive),
            "bytes": archive.stat().st_size if archive.exists() else None,
        }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["component_summary"]
    pixel = report["pixel_summary"]
    lines = [
        "# Pair Component Error XRay",
        "",
        f"- schema: `{report['schema']}`",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- label: `{report['label']}`",
        f"- evidence_grade: `{report['evidence_grade']}`",
        f"- device: `{report['device']}`",
        f"- n_pairs: `{report['n_pairs']}`",
        "- score_claim: `false`",
        "- dispatch_attempted: `false`",
        "- promotion_eligible: `false`",
        "",
        "## Component Summary",
        "",
        f"- avg_posenet_dist: `{summary['avg_posenet_dist']}`",
        f"- avg_segnet_dist: `{summary['avg_segnet_dist']}`",
        f"- pose_score_contribution: `{summary['pose_score_contribution']}`",
        f"- seg_score_contribution: `{summary['seg_score_contribution']}`",
        f"- component_score_no_rate: `{summary['component_score_no_rate']}`",
        "",
        "## Pixel Summary",
        "",
        f"- avg_frame0_l1: `{pixel['avg_frame0_l1']}`",
        f"- avg_frame1_l1: `{pixel['avg_frame1_l1']}`",
        f"- avg_frame0_changed_fraction: `{pixel['avg_frame0_changed_fraction']}`",
        f"- avg_frame1_changed_fraction: `{pixel['avg_frame1_changed_fraction']}`",
        "",
    ]
    for section, rows in report["top_pairs"].items():
        lines.extend([f"## Top {section}", "", "| pair | component | pose | seg | f0_l1 | f1_l1 |", "|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            lines.append(
                f"| {row['pair_idx']} | {row['component_score_no_rate']:.9f} | "
                f"{row['pose_score_contribution']:.9f} | {row['seg_score_contribution']:.9f} | "
                f"{row['frame0_l1']:.6f} | {row['frame1_l1']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inflated-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--video-names-file", type=Path, default=Path("upstream/public_test_video_names.txt"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--label", default="pair_component_xray")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = compute_pair_rows(
        inflated_dir=args.inflated_dir.resolve(),
        upstream_dir=args.upstream_dir.resolve(),
        video_names_file=args.video_names_file.resolve(),
        device=args.device,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        num_threads=args.num_threads,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    report = build_report(
        rows=rows,
        inflated_dir=args.inflated_dir.resolve(),
        upstream_dir=args.upstream_dir.resolve(),
        video_names_file=args.video_names_file.resolve(),
        device=args.device,
        label=args.label,
        top_k=args.top_k,
        archive=args.archive.resolve() if args.archive else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pair_component_xray.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "pair_component_xray.md").write_text(render_markdown(report), encoding="utf-8")
    (args.output_dir / "rebuild_command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
