#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for granular MPS-vs-CUDA drift decomposition.

Wraps ``tac.mps_diagnostic.granular_drift.build_granular_drift_report`` so the
operator can decompose any (MPS-forward-outputs, CUDA-forward-outputs) pair
into the canonical 6-granularity report (per-frame / per-pixel / per-boundary
/ per-byte / per-pair / per-pair x master-gradient).

Usage:
    .venv/bin/python tools/analyze_mps_drift_granular.py \\
        --mps-forward-outputs <path>.pt \\
        --cuda-forward-outputs <path>.pt \\
        [--master-gradient-anchor-archive <archive_sha256>] \\
        [--archive-sha256 <archive_sha256>] \\
        [--checkpoint-mps <path>.pt --checkpoint-cuda <path>.pt] \\
        [--report-out .omx/state/mps_drift_granular_<utc>.json] \\
        [--summary]

If no ``--report-out`` is provided the report is written to
``.omx/state/mps_drift_granular_<utc_iso>.json``.

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #1/#192/#317,
every JSON the CLI writes carries:
  * evidence_grade = "MPS-research-signal"
  * axis_tag = "[macOS-MPS-PyTorch-vs-CUDA-diagnostic]"
  * score_claim = False
  * promotion_eligible = False
  * ready_for_exact_eval_dispatch = False

Exit codes:
  0 success
  1 input file missing / shape mismatch
  2 CLI arg error
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="analyze_mps_drift_granular",
        description=(
            "Decompose MPS-vs-CUDA forward drift across per-frame / per-pixel "
            "/ per-boundary / per-byte / per-pair / per-pair x master-gradient. "
            "All output is non-promotable per CLAUDE.md 'MPS auth eval is NOISE'."
        ),
    )
    p.add_argument(
        "--mps-forward-outputs",
        required=True,
        help="Path to MPS forward-outputs .pt (tensor or dict with 'rgb' key); shape (N_pairs, 2, 3, H, W).",
    )
    p.add_argument(
        "--cuda-forward-outputs",
        required=True,
        help="Path to CUDA forward-outputs .pt; same shape as MPS.",
    )
    p.add_argument(
        "--master-gradient-anchor-archive",
        default="",
        help=(
            "Optional 64-char archive sha to look up master-gradient anchor "
            "via tac.master_gradient.latest_anchor_for_archive. If absent or "
            "anchor not found, the cosine distribution falls back to "
            "NO_MASTER_GRADIENT_ANCHOR verdict."
        ),
    )
    p.add_argument(
        "--archive-sha256",
        default="",
        help="Full 64-char archive sha for provenance tagging.",
    )
    p.add_argument(
        "--checkpoint-mps",
        default="",
        help=(
            "Optional MPS-EMA checkpoint .pt; if provided alongside "
            "--checkpoint-cuda, the weight_delta is computed as the per-param "
            "concat of (mps_param - cuda_param) and threaded into the per-pair "
            "x master-gradient inner-product computation."
        ),
    )
    p.add_argument(
        "--checkpoint-cuda",
        default="",
        help="Optional CUDA-shadow EMA checkpoint .pt for weight-delta computation.",
    )
    p.add_argument(
        "--report-out",
        default="",
        help=(
            "Destination JSON path. If empty, writes to "
            ".omx/state/mps_drift_granular_<utc>.json (canonical posterior dir)."
        ),
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print human-readable summary on stdout (in addition to JSON file).",
    )
    p.add_argument(
        "--boundary-band-px",
        type=int,
        default=3,
        help="Width (px) of the boundary band around CUDA-baseline argmax class boundaries.",
    )
    return p.parse_args(argv)


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_tensor(path: Path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("rgb", "recon", "output", "outputs", "frames", "tensor"):
            if key in obj and hasattr(obj[key], "shape"):
                return obj[key]
        raise ValueError(
            f"{path}: dict without canonical key ('rgb'/'recon'/'output'/...); "
            f"keys present: {list(obj.keys())[:8]}"
        )
    raise TypeError(f"{path}: unsupported type {type(obj).__name__}")


def _load_master_gradient_anchor(archive_sha: str):
    """Load the latest master-gradient anchor for the given archive sha.

    Uses canonical helper `tac.master_gradient.latest_anchor_for_archive`. The
    helper returns a row dict; we reconstruct a MasterGradient instance from
    the row so the per-pair gradient tensor can be loaded via the dataclass
    method.
    """
    if not archive_sha:
        return None
    try:
        from tac.master_gradient import (  # type: ignore[import-not-found]
            MasterGradient,
            OperatingPoint,
            latest_anchor_for_archive,
        )
    except ImportError as exc:
        print(f"[granular-drift] master_gradient import failed: {exc}", file=sys.stderr)
        return None
    row = latest_anchor_for_archive(archive_sha)
    if row is None:
        return None
    try:
        op = OperatingPoint(
            d_seg=float(row["operating_point"]["d_seg"]),
            d_pose=float(row["operating_point"]["d_pose"]),
            rate=float(row["operating_point"]["rate"]),
            score=float(row["operating_point"]["score"]),
        )
        return MasterGradient(
            archive_sha256=row["archive_sha256"],
            operating_point=op,
            gradient_array_path=row["gradient_array_path"],
            n_bytes=int(row["n_bytes"]),
            measurement_method=row["measurement_method"],
            measurement_axis=row["measurement_axis"],
            measurement_hardware=row["measurement_hardware"],
            measurement_call_id=row.get("measurement_call_id"),
            measurement_utc=row["measurement_utc"],
            gradient_tensor_kind=row.get("gradient_tensor_kind", "aggregate_per_byte_v1"),
            n_pairs=row.get("n_pairs"),
            scored_archive_sha256=row.get("scored_archive_sha256"),
            scored_archive_bytes=row.get("scored_archive_bytes"),
            gradient_subject_sha256=row.get("gradient_subject_sha256"),
            gradient_subject_bytes=row.get("gradient_subject_bytes"),
            gradient_byte_domain=row.get("gradient_byte_domain", "scored_archive_bytes"),
            n_pairs_used=row.get("n_pairs_used"),
            n_pairs_total=row.get("n_pairs_total"),
            score_axis_dominance=row.get("score_axis_dominance"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[granular-drift] anchor reconstruction failed: {exc}", file=sys.stderr)
        return None


def _compute_weight_delta(mps_ckpt_path: Path, cuda_ckpt_path: Path):
    """Concatenate per-param (mps - cuda) deltas in a stable canonical order.

    Returns a 1-D numpy array. The canonical order is the sorted-by-key
    iteration over the intersection of the two state dicts.
    """
    import numpy as np
    import torch
    mps_ckpt = torch.load(mps_ckpt_path, map_location="cpu", weights_only=False)
    cuda_ckpt = torch.load(cuda_ckpt_path, map_location="cpu", weights_only=False)
    mps_sd = mps_ckpt.get("state_dict", mps_ckpt) if isinstance(mps_ckpt, dict) else mps_ckpt
    cuda_sd = cuda_ckpt.get("state_dict", cuda_ckpt) if isinstance(cuda_ckpt, dict) else cuda_ckpt
    if not isinstance(mps_sd, dict) or not isinstance(cuda_sd, dict):
        raise TypeError("checkpoints must be dict[str, Tensor] or {'state_dict': ...}")
    common_keys = sorted(set(mps_sd.keys()) & set(cuda_sd.keys()))
    parts = []
    for k in common_keys:
        mps_t = mps_sd[k]
        cuda_t = cuda_sd[k]
        if not (hasattr(mps_t, "shape") and hasattr(cuda_t, "shape")):
            continue
        if mps_t.shape != cuda_t.shape:
            continue
        diff = (mps_t.detach().to("cpu").to(torch.float64) - cuda_t.detach().to("cpu").to(torch.float64))
        parts.append(diff.flatten().numpy())
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def _print_summary(report) -> None:
    cs = report.cosine_distribution_summary
    print("=" * 70)
    print(f"GRANULAR MPS-vs-CUDA DRIFT REPORT  schema={report.schema_version}")
    print(f"  axis_tag={report.axis_tag}  evidence_grade={report.evidence_grade}")
    print(f"  score_claim={report.score_claim}  promotion_eligible={report.promotion_eligible}")
    print(f"  mps_artifact: {report.mps_artifact_path}")
    print(f"  cuda_artifact: {report.cuda_artifact_path}")
    print(f"  n_pairs: {report.n_pairs}")
    print("-" * 70)
    print("PER-FRAME:")
    for r in report.per_frame[:6]:
        print(
            f"  pair={r.pair_index} frame={r.frame_index}  "
            f"pixel_l1={r.pixel_l1:.3e}  seg={r.segnet_logit_l_inf:.3e}  "
            f"pose={r.posenet_pose_l2:.3e}  agg={r.aggregate:.3e}"
        )
    if len(report.per_frame) > 6:
        print(f"  ... ({len(report.per_frame) - 6} more)")
    print("PER-PIXEL:")
    for r in report.per_pixel:
        print(
            f"  layer={r.layer_name}  l_inf={r.l_inf:.3e}  l_2={r.l_2:.3e}  "
            f"mean_abs={r.mean_abs:.3e}  fraction>1e-3={r.fraction_above_1e_3:.3%}"
        )
    print(f"PER-BOUNDARY: {len(report.per_boundary)} pair records")
    if report.per_boundary:
        ratios = [b.flip_rate_in_band / max(b.flip_rate_overall, 1e-9) for b in report.per_boundary]
        avg_ratio = sum(ratios) / len(ratios)
        print(f"  avg in-band/overall flip-rate ratio = {avg_ratio:.2f}")
    print(f"PER-BYTE: {len(report.per_byte)} mutation records (0 = byte-mutation probe not run)")
    print(f"PER-PAIR: {len(report.per_pair)} pair records")
    if report.per_pair:
        aggs = [r.aggregate_drift for r in report.per_pair]
        print(
            f"  mean_agg={sum(aggs)/len(aggs):.3e}  "
            f"max_agg={max(aggs):.3e}  min_agg={min(aggs):.3e}"
        )
    print(f"PER-PAIR x MASTER-GRADIENT: {len(report.per_pair_master_gradient)} records")
    print(
        f"  cosine summary: verdict={cs.verdict}  mean={cs.mean:.4f}  "
        f"median={cs.median:.4f}  std={cs.std:.4f}  abs_mean={cs.abs_mean:.4f}"
    )
    print(
        f"  outliers |cos|>0.5: {cs.n_outliers_abs_above_0_5}  "
        f"|cos|>0.8: {cs.n_outliers_abs_above_0_8}  max_abs={cs.max_abs:.4f}"
    )
    print("-" * 70)
    print("CORRECTIVE ENGINEERING RECOMMENDATIONS (predicted, non-promotable):")
    for i, rec in enumerate(report.summary_corrective_engineering_recommendations, start=1):
        marker = "[TRIGGERED]" if rec.get("triggered") else "[silent]"
        print(
            f"  {i}. {marker} {rec['name']}  "
            f"predicted_dS=[{rec['predicted_delta_s_floor']:+.4f}, {rec['predicted_delta_s_ceiling']:+.4f}]  "
            f"cost=[${rec['cost_usd_estimate_floor']:.2f}, ${rec['cost_usd_estimate_ceiling']:.2f}]  "
            f"hooks={list(rec['hook_numbers'])}"
        )
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from tac.mps_diagnostic.granular_drift import (  # noqa: E402  (sys.path is patched above)
        build_granular_drift_report,
        write_granular_drift_report,
    )
    mps_path = Path(args.mps_forward_outputs)
    cuda_path = Path(args.cuda_forward_outputs)
    if not mps_path.exists():
        print(f"[granular-drift] MPS forward outputs not found: {mps_path}", file=sys.stderr)
        return 1
    if not cuda_path.exists():
        print(f"[granular-drift] CUDA forward outputs not found: {cuda_path}", file=sys.stderr)
        return 1
    try:
        mps_recon = _load_tensor(mps_path)
        cuda_recon = _load_tensor(cuda_path)
    except (TypeError, ValueError) as exc:
        print(f"[granular-drift] tensor load error: {exc}", file=sys.stderr)
        return 1
    if mps_recon.shape != cuda_recon.shape:
        print(
            f"[granular-drift] shape mismatch: mps {tuple(mps_recon.shape)} vs cuda {tuple(cuda_recon.shape)}",
            file=sys.stderr,
        )
        return 1
    anchor = _load_master_gradient_anchor(args.master_gradient_anchor_archive)
    weight_delta = None
    if args.checkpoint_mps and args.checkpoint_cuda:
        mps_ckpt = Path(args.checkpoint_mps)
        cuda_ckpt = Path(args.checkpoint_cuda)
        if not mps_ckpt.exists() or not cuda_ckpt.exists():
            print(f"[granular-drift] checkpoint(s) not found", file=sys.stderr)
            return 1
        try:
            weight_delta = _compute_weight_delta(mps_ckpt, cuda_ckpt)
        except (TypeError, ValueError) as exc:
            print(f"[granular-drift] weight_delta computation failed: {exc}", file=sys.stderr)
            weight_delta = None
    notes_lines = [
        "Granular MPS-vs-CUDA drift report produced by tools/analyze_mps_drift_granular.py.",
        "Non-promotable per CLAUDE.md 'MPS auth eval is NOISE' + Catalog #1/#192/#317.",
        "Score-claim tags forbidden until paired Linux x86_64 anchor lands.",
    ]
    if anchor is None and args.master_gradient_anchor_archive:
        notes_lines.append(
            f"NOTE: no master-gradient anchor found for archive_sha={args.master_gradient_anchor_archive!r}; "
            "cosine summary verdict = NO_MASTER_GRADIENT_ANCHOR."
        )
    if weight_delta is None and args.checkpoint_mps and args.checkpoint_cuda:
        notes_lines.append("NOTE: weight_delta computation failed; per-pair x master-gradient is empty.")
    report = build_granular_drift_report(
        mps_artifact_path=str(mps_path),
        cuda_artifact_path=str(cuda_path),
        mps_recon=mps_recon,
        cuda_recon=cuda_recon,
        weight_delta=weight_delta,
        master_gradient_anchor=anchor,
        archive_sha256=args.archive_sha256 or "0" * 64,
        boundary_band_px=args.boundary_band_px,
        notes="\n".join(notes_lines),
    )
    if args.report_out:
        out_path = Path(args.report_out)
    else:
        out_path = REPO_ROOT / ".omx" / "state" / f"mps_drift_granular_{_utc_iso()}.json"
    written = write_granular_drift_report(report, out_path)
    print(f"[granular-drift] wrote report: {written}")
    if args.summary:
        _print_summary(report)
    # Final non-promotability banner.
    print(
        "[granular-drift] NON-AUTHORITATIVE: this report is "
        "evidence_grade='MPS-research-signal' / score_claim=False / "
        "promotion_eligible=False per CLAUDE.md 'MPS auth eval is NOISE' + "
        "Catalog #1/#192/#317."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
