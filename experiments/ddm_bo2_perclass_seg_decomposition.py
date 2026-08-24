#!/usr/bin/env python3
"""ddm_bo2 -- per-class decomposition of the frozen SegNet argmax disagreement.

WHAT THIS MEASURES
------------------
``upstream/evaluate.py`` reports ONE composite number, ``avg_segnet_dist``.  Its
definition (``modules.py:111-113``) is the per-pair mean of
``L_gt.argmax != L_comp.argmax`` over the 512x384 SegNet output grid, where
``L_gt`` is SegNet's OWN argmax on the ground-truth frames -- not an external
label map.  So the "true class" of a position is a MEASURED quantity that has to
be produced by the same forward pass, on the same GT lineage, as the row itself.

This script reproduces that composite exactly and decomposes it into the 5x5
confusion matrix ``(gt_class, comp_class)`` plus per-class flip rates.  A
composite verdict would destroy the one signal most likely to route a successor:
``ddm_cb1`` measured a MyCar carrier ADMITTING (-1.05e-5 d_seg) while a Lane
carrier REJECTED (+0.0366 d_seg) on the same vehicle.

THE POSITIVE CONTROL IS FREE AND BUILT IN
-----------------------------------------
The per-pair flip rates this script accumulates are the SAME quantity
``evaluate.py`` averages.  Their mean must reproduce the row's reported
``avg_segnet_dist``.  If it does not, this script is a DIFFERENT instrument from
the row and no per-class number it produces may be attached to that row (the
``#1034`` cross-instrument genus).  The check is asserted, not printed, so a
mismatch fails closed.

Agreement is stated in ARGMAX PIXELS, not float ulps -- see the note at the
control itself.  One pixel is 8.478e-9 of ``d_seg``, which is LARGER than the
5e-9 half-width of the report's own 8-dp rounding, so a float tolerance below
one pixel cannot be satisfied except by luck.  Pass ``--batch-size`` equal to
the row's (``evaluate.py`` defaults to 16 and ``contest_auth_eval`` pins
nothing) or CPU BLAS will block differently and near-tie logits will argmax the
other way.

WHAT IT CANNOT RULE OUT
-----------------------
Reproducing ``avg_segnet_dist`` proves this forward pass agrees with the row's
forward pass on the SEG axis.  It says nothing about the POSE axis, which is
computed from a different head on a different (912x1368) preprocess and is NOT
re-derived here -- pose is read from the row's own report only.

CLASS INDEX ORDER
-----------------
``0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar`` -- the canonical comma10k
order, MEASURED (CLAUDE.md) and NEVER re-derived by luma-sorting the comma10k
class values, which yields a different and wrong order.  This script does not
assume it: it reports per-class areas so the caller can check them against the
canonical n600 shares (23.233 / 0.586 / 49.518 / 1.238 / 25.425 %).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

NUM_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
# CLAUDE.md, MEASURED on gt_n600; used only as a REPORTED cross-check, never as
# an assumption this script depends on.
CANONICAL_N600_AREA_SHARE = (0.23233, 0.005858, 0.495175, 0.012380, 0.254255)


class PerClassError(RuntimeError):
    """Fail-closed error for the per-class decomposition."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream(upstream_dir: Path):
    """Import the pinned upstream scorer surface WITHOUT mutating it."""
    sys.path.insert(0, str(upstream_dir))
    import frame_utils
    import modules

    return frame_utils, modules


def build_distortion_net(modules_mod, device: torch.device):
    net = modules_mod.DistortionNet().eval().to(device=device)
    net.load_state_dicts(modules_mod.posenet_sd_path, modules_mod.segnet_sd_path, device)
    return net


def segnet_argmax(net, batch: torch.Tensor) -> torch.Tensor:
    """SegNet argmax for one (B, seq_len, H, W, C) uint8 batch.

    Routed through the upstream ``preprocess_input`` chain verbatim so this is
    the same arithmetic evaluate.py performs, not a re-implementation.
    """
    _, segnet_in = net.preprocess_input(batch)
    return net.segnet(segnet_in).argmax(dim=1).to(torch.uint8)


def run(
    *,
    upstream_dir: Path,
    videos_dir: Path,
    comp_raw: Path,
    batch_size: int,
    device_str: str,
    gt_argmax_cache: Path | None,
    comp_argmax_out: Path | None,
    per_pair_out: Path | None,
    expected_avg_segnet_dist: float | None,
) -> dict[str, Any]:
    if device_str == "mps":
        raise PerClassError("MPS is never a scorer authority; refuse (CLAUDE.md)")
    device = torch.device(device_str)
    frame_utils, modules_mod = load_upstream(upstream_dir)
    net = build_distortion_net(modules_mod, device)

    names_file = upstream_dir / "public_test_video_names.txt"
    test_video_names = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]

    # The comp side is one raw file; stage it under the layout TensorVideoDataset expects.
    comp_dir = comp_raw.parent
    ds_gt = frame_utils.AVVideoDataset(
        test_video_names, data_dir=videos_dir, batch_size=batch_size, device=device
    )
    ds_gt.prepare_data()
    ds_comp = frame_utils.TensorVideoDataset(
        test_video_names, data_dir=comp_dir, batch_size=batch_size, device=device
    )
    ds_comp.prepare_data()

    reuse_gt = (
        gt_argmax_cache is not None
        and gt_argmax_cache.is_file()
        and gt_argmax_cache.stat().st_size > 0
    )
    gt_cached = (
        np.memmap(gt_argmax_cache, dtype=np.uint8, mode="r") if reuse_gt else None
    )

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_pair: list[float] = []
    gt_chunks: list[np.ndarray] = []
    comp_chunks: list[np.ndarray] = []
    cursor = 0

    with torch.inference_mode():
        # strict=True: a short comp file would otherwise silently truncate the
        # run to a PREFIX, and a prefix of this population is a different
        # population -- axis-dependent by 2.5-4.2x on pose ([[m88]]/[[m96]]).
        for (_, _, batch_gt), (_, _, batch_comp) in zip(ds_gt, ds_comp, strict=True):
            batch_gt = batch_gt.to(device)
            batch_comp = batch_comp.to(device)
            if batch_gt.shape != batch_comp.shape:
                raise PerClassError(
                    f"gt/comp batch shape mismatch: {batch_gt.shape} vs {batch_comp.shape}"
                )
            comp_arg = segnet_argmax(net, batch_comp).cpu().numpy()
            if gt_cached is not None:
                span = comp_arg.size
                gt_arg = np.asarray(
                    gt_cached[cursor : cursor + span], dtype=np.uint8
                ).reshape(comp_arg.shape)
                cursor += span
            else:
                gt_arg = segnet_argmax(net, batch_gt).cpu().numpy()
                gt_chunks.append(gt_arg.copy())
            comp_chunks.append(comp_arg.copy())

            flat_gt = gt_arg.reshape(gt_arg.shape[0], -1)
            flat_comp = comp_arg.reshape(comp_arg.shape[0], -1)
            per_pair.extend((flat_gt != flat_comp).mean(axis=1).tolist())
            np.add.at(
                confusion,
                (flat_gt.reshape(-1).astype(np.int64), flat_comp.reshape(-1).astype(np.int64)),
                1,
            )

    pairs = len(per_pair)
    if pairs == 0:
        raise PerClassError("no pairs were scored; the datasets produced nothing")
    avg_segnet_dist = float(np.mean(per_pair))

    # POSITIVE CONTROL: this must be the row's own number, or this is a
    # different instrument and none of the per-class output may attach to it.
    #
    # THE UNIT IS ARGMAX PIXELS, NOT THE 8-DP FLOAT.  d_seg is the mean of
    # ``pairs`` rationals with denominator ``EVAL_H*EVAL_W``, so its physical
    # quantum is ONE flipped argmax pixel = 1/positions = 8.478e-9 -- which is
    # LARGER than the 5e-9 half-width of the report's own 8-dp rounding.  A
    # tolerance stated in float ulps below that quantum is unsatisfiable by
    # construction; it does not measure agreement, it measures luck.  The band
    # is therefore (report rounding) + (one pixel of fp32 reduction-order
    # slack), and the realized pixel delta is REPORTED so a reader sees the
    # actual disagreement rather than a pass/fail bit.
    #
    # The reduction-order slack is real and has a named cause: CPU BLAS blocks
    # differently at different batch sizes, so a logit pair within 1 ulp at a
    # near-tie can argmax either way.  Run with the SAME --batch-size the row
    # used (evaluate.py's default is 16 and contest_auth_eval pins nothing) and
    # this term is normally 0.
    # PERSIST BEFORE JUDGING.  The control below can refuse, and an earlier
    # revision refused with every argmax field still only in memory -- twenty
    # minutes of measurement discarded because a LABEL was wrong.  Retention is
    # a precondition for running, never a step after the verdict.
    if gt_chunks and gt_argmax_cache is not None:
        gt_argmax_cache.parent.mkdir(parents=True, exist_ok=True)
        np.concatenate([c.reshape(-1) for c in gt_chunks]).tofile(gt_argmax_cache)
    if comp_argmax_out is not None:
        comp_argmax_out.parent.mkdir(parents=True, exist_ok=True)
        np.concatenate([c.reshape(-1) for c in comp_chunks]).tofile(comp_argmax_out)
    if per_pair_out is not None:
        per_pair_out.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(per_pair, dtype="<f8").tofile(per_pair_out)

    positions = int(confusion.sum())
    control: dict[str, Any] = {"expected": expected_avg_segnet_dist}
    if expected_avg_segnet_dist is not None:
        pixel_quantum = 1.0 / positions
        report_half_width = 5e-9
        delta = abs(avg_segnet_dist - expected_avg_segnet_dist)
        beyond_rounding = max(0.0, delta - report_half_width)
        control.update(
            {
                "recomputed": avg_segnet_dist,
                "abs_delta": delta,
                "report_8dp_half_width": report_half_width,
                "one_argmax_pixel": pixel_quantum,
                "pixel_delta_beyond_report_rounding": beyond_rounding / pixel_quantum,
                "tolerance": report_half_width + pixel_quantum,
                "unit": "argmax pixels; see the module note on why float ulps are the wrong unit",
            }
        )
        control["passed"] = bool(delta <= report_half_width + pixel_quantum)
        control["refusal"] = (
            None
            if control["passed"]
            else (
                f"recomputed avg_segnet_dist {avg_segnet_dist!r} vs row "
                f"{expected_avg_segnet_dist!r}: {beyond_rounding / pixel_quantum:.2f} argmax "
                f"pixels beyond the report's own rounding. This is a DIFFERENT instrument from "
                f"the row; its per-class output is NOT attachable to that row."
            )
        )

    gt_area = confusion.sum(axis=1)
    flips_by_gt_class = gt_area - np.diag(confusion)
    total_flips = int(flips_by_gt_class.sum())

    per_class = []
    for c in range(NUM_CLASSES):
        area = int(gt_area[c])
        flips = int(flips_by_gt_class[c])
        row = confusion[c].copy()
        row[c] = 0
        dominant = int(row.argmax()) if row.sum() else None
        per_class.append(
            {
                "class_index": c,
                "class_name": CLASS_NAMES[c],
                "gt_positions": area,
                "gt_area_share": area / positions if positions else 0.0,
                "canonical_n600_area_share": CANONICAL_N600_AREA_SHARE[c],
                "flips": flips,
                "within_class_flip_rate": flips / area if area else 0.0,
                "share_of_all_flips": flips / total_flips if total_flips else 0.0,
                # d_seg is a mean over the WHOLE grid, so each class's additive
                # contribution to the composite is flips / positions.
                "contribution_to_d_seg": flips / positions if positions else 0.0,
                "dominant_confusion_to": None if dominant is None else CLASS_NAMES[dominant],
                "dominant_confusion_count": None if dominant is None else int(row[dominant]),
            }
        )

    return {
        "schema": "ddm_bo2_perclass_seg_decomposition.v1",
        "comp_raw": {
            "path": str(comp_raw),
            "bytes": comp_raw.stat().st_size,
            "sha256": sha256_of(comp_raw),
        },
        "device": device_str,
        "gt_lineage": "PYAV_YUV420_TO_RGB via AVVideoDataset (authority=False)",
        "gt_argmax_reused_from_cache": bool(reuse_gt),
        "pairs_scored": pairs,
        "positions": positions,
        "avg_segnet_dist_recomputed": avg_segnet_dist,
        "positive_control_vs_row_report": control,
        "total_flips": total_flips,
        "confusion_gt_by_comp": confusion.tolist(),
        "per_class": per_class,
        "artifacts": {
            "gt_argmax": None if gt_argmax_cache is None else str(gt_argmax_cache),
            "comp_argmax": None if comp_argmax_out is None else str(comp_argmax_out),
            "per_pair_seg_dist": None if per_pair_out is None else str(per_pair_out),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--comp-raw", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gt-argmax-cache", type=Path, default=None)
    parser.add_argument("--comp-argmax-out", type=Path, default=None)
    parser.add_argument("--per-pair-out", type=Path, default=None)
    parser.add_argument(
        "--expected-avg-segnet-dist",
        type=float,
        default=None,
        help="the row's own reported avg_segnet_dist; enables the positive control",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = run(
        upstream_dir=args.upstream_dir.resolve(),
        videos_dir=args.videos_dir.resolve(),
        comp_raw=args.comp_raw.resolve(),
        batch_size=args.batch_size,
        device_str=args.device,
        gt_argmax_cache=args.gt_argmax_cache,
        comp_argmax_out=args.comp_argmax_out,
        per_pair_out=args.per_pair_out,
        expected_avg_segnet_dist=args.expected_avg_segnet_dist,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    # Fail closed on a control refusal -- but only AFTER the payload and the
    # receipt are on disk, so the refusal is itself a retained measurement
    # rather than a reason to have measured nothing.
    control = report["positive_control_vs_row_report"]
    if control.get("passed") is False:
        print(f"REFUSED: {control['refusal']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
