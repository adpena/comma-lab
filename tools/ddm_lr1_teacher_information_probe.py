#!/usr/bin/env python
"""ddm_lr1 — TEACHER-INFORMATION PROBE: does a candidate distillation teacher carry
information beyond the free GT scorer field?

Motivation (ddm_lr1, 2026-08-02). A distillation teacher can only help a student if its
supervisory content differs from what the student already gets for free. In this repo the
free reference is the frozen-scorer GT cache (``gt_n600.npz``: ``lstars`` = SegNet argmax on
the scored frame, ``margins`` = top1-top2 on the same frame, same scorer path as
``tools/ddm_b2b_segnet_field_pass.py``). Any teacher whose argmax is a subset of GT's and
whose margin field is a noisy copy of GT's is INFORMATIONALLY DOMINATED: it is a strictly
worse version of a field the trainer already memmaps.

This probe measures that domination directly and cheaply (ZERO scorer forwards -- it consumes
a precomputed teacher logit cache), and it is DEFENSIVE: every comparison is paired with a
shuffled-pair POSITIVE CONTROL so an "everything matches" readout cannot be confused with a
dead instrument. Empty scope is reported as VACUOUS, never as a pass.

Reusable as the admission test for any future teacher (the dw1 reformulation-queue rows
(3)/(5) alternative-teacher slots, and the gc12 staleness-gated field re-check).

Pointer honesty: advisory [macOS-CPU]; ``score_claim=False``; this moves no pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

#: Shuffle stride for the positive control. Coprime with typical pair counts so the control
#: pairing is a derangement (no pair compared against itself).
CONTROL_STRIDE: int = 137

#: LIVENESS FLOOR (not a physics constant, and deliberately not on the value-provenance ladder
#: for scientific values): the minimum correlation drop from same-pair to shuffled-pair below
#: which we refuse to trust our own readout. Its only job is to separate "the teacher really
#: does track GT" from "the comparison is degenerate and would report agreement against
#: anything". Any value well inside the measured separation serves; it is a guard, not a
#: measurement, and no result is derived from it.
CONTROL_LIVENESS_MIN_CORR_DROP: float = 0.05


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def top2_margin(logits: np.ndarray) -> np.ndarray:
    """top1-top2 margin over the class axis, clamped at 0.

    Mirrors ``tools/ddm_b2b_segnet_field_pass.py::real_segnet_field_fn`` exactly so teacher
    and GT margins are the same quantity.
    """
    if logits.ndim != 3:
        raise ValueError(f"expected (C,H,W) logits, got {logits.shape}")
    part = np.partition(logits.astype(np.float32), -2, axis=0)
    return np.maximum(part[-1] - part[-2], 0.0)


class _Accum:
    """Streaming mean/variance/covariance so n600 never materialises in RAM."""

    def __init__(self) -> None:
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = self.sse = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x = x.astype(np.float64).ravel()
        y = y.astype(np.float64).ravel()
        self.n += x.size
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())
        self.sxy += float((x * y).sum())
        self.sse += float(((x - y) ** 2).sum())

    def result(self) -> dict[str, float | int]:
        if self.n == 0:
            raise ValueError("VACUOUS accumulator: zero pixels compared")
        n = self.n
        mx, my = self.sx / n, self.sy / n
        vx, vy = self.sxx / n - mx * mx, self.syy / n - my * my
        cov = self.sxy / n - mx * my
        denom = float(np.sqrt(max(vx, 0.0) * max(vy, 0.0)))
        return {
            "n_pixels": n,
            "corr": float(cov / denom) if denom > 0.0 else float("nan"),
            "rmse": float(np.sqrt(self.sse / n)),
            "teacher_sigma": float(np.sqrt(max(vx, 0.0))),
            "reference_sigma": float(np.sqrt(max(vy, 0.0))),
        }


def probe(
    teacher_logits: np.ndarray,
    lstars: np.ndarray,
    margins: np.ndarray,
    *,
    control_stride: int = CONTROL_STRIDE,
) -> dict[str, Any]:
    """Compare a teacher field against the free GT field, with a shuffled-pair control.

    ``teacher_logits`` is (P,C,H,W) (memmap ok); ``lstars`` (P,H,W); ``margins`` (P,H,W).
    """
    n_pairs = int(teacher_logits.shape[0])
    if n_pairs == 0:
        raise ValueError("VACUOUS scope: zero pairs -- this is not a pass")
    if lstars.shape[0] != n_pairs or margins.shape[0] != n_pairs:
        raise ValueError(
            f"pair-count mismatch: teacher {n_pairs}, lstars {lstars.shape[0]}, margins {margins.shape[0]}"
        )
    if control_stride % n_pairs == 0:
        raise ValueError(
            f"control_stride {control_stride} is degenerate for {n_pairs} pairs "
            "(control would compare each pair against itself)"
        )

    same, ctrl = _Accum(), _Accum()
    per_pair_dseg = np.zeros(n_pairs, dtype=np.float64)
    n_classes = int(teacher_logits.shape[1])
    cls_err = np.zeros(n_classes, dtype=np.float64)
    cls_tot = np.zeros(n_classes, dtype=np.float64)

    for i in range(n_pairs):
        logits = np.asarray(teacher_logits[i], dtype=np.float32)
        t_argmax = logits.argmax(0)
        t_margin = top2_margin(logits)
        gt_label = lstars[i]
        wrong = t_argmax != gt_label
        per_pair_dseg[i] = float(wrong.mean())
        for c in range(n_classes):
            in_c = gt_label == c
            cls_tot[c] += float(in_c.sum())
            cls_err[c] += float((wrong & in_c).sum())
        same.add(t_margin, margins[i])
        ctrl.add(t_margin, margins[(i + control_stride) % n_pairs])

    same_r, ctrl_r = same.result(), ctrl.result()
    sigma = same_r["reference_sigma"]
    return {
        "schema": "ddm_lr1_teacher_information_probe.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "research_only": True,
        "n_pairs": n_pairs,
        "argmax": {
            "teacher_realized_d_seg_vs_gt": float(per_pair_dseg.mean()),
            "per_pair_max": float(per_pair_dseg.max()),
            "per_pair_argmax_pair_id": int(per_pair_dseg.argmax()),
            "per_class_error_rate": {
                str(c): float(cls_err[c] / cls_tot[c]) if cls_tot[c] else None for c in range(n_classes)
            },
        },
        "margin_same_pair": same_r,
        "margin_control_pair": ctrl_r,
        "control_stride": control_stride,
        "normalised": {
            "rmse_same_over_reference_sigma": (same_r["rmse"] / sigma if sigma > 0.0 else None),
            "rmse_control_over_reference_sigma": (ctrl_r["rmse"] / sigma if sigma > 0.0 else None),
        },
        "control_liveness_min_corr_drop": CONTROL_LIVENESS_MIN_CORR_DROP,
        "control_fired": bool(ctrl_r["corr"] < same_r["corr"] - CONTROL_LIVENESS_MIN_CORR_DROP),
    }


def _load_gt(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache) as data:
        missing = {"lstars", "margins"} - set(data.files)
        if missing:
            raise ValueError(f"GT cache {cache} missing {sorted(missing)}")
        return data["lstars"], data["margins"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--teacher-logits",
        required=True,
        type=Path,
        help="(P,C,H,W) .npy teacher SegNet logit cache (memmapped; no scorer forward run)",
    )
    ap.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
        help="frozen-authority GT cache carrying lstars + margins",
    )
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--control-stride", type=int, default=CONTROL_STRIDE)
    ap.add_argument(
        "--limit-pairs",
        type=int,
        default=0,
        help="0 = all pairs. Any non-zero value is a SUBSET and is labelled as such.",
    )
    args = ap.parse_args(argv)

    logits = np.load(args.teacher_logits, mmap_mode="r")
    lstars, margins = _load_gt(args.gt_cache)
    if args.limit_pairs:
        k = min(args.limit_pairs, logits.shape[0])
        logits, lstars, margins = logits[:k], lstars[:k], margins[:k]

    report = probe(logits, lstars, margins, control_stride=args.control_stride)
    report["teacher_logits_path"] = str(args.teacher_logits)
    report["teacher_logits_sha256"] = _sha256(args.teacher_logits)
    report["gt_cache_path"] = str(args.gt_cache)
    report["scope"] = "FULL" if not args.limit_pairs else f"SUBSET_{args.limit_pairs}"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    same = report["margin_same_pair"]
    ctrl = report["margin_control_pair"]
    print(f"scope={report['scope']} pairs={report['n_pairs']} px={same['n_pixels']}")
    print(f"teacher realized d_seg vs GT = {report['argmax']['teacher_realized_d_seg_vs_gt']:.10f}")
    print(f"margin SAME    corr {same['corr']:.6f}  rmse {same['rmse']:.6f}")
    print(f"margin CONTROL corr {ctrl['corr']:.6f}  rmse {ctrl['rmse']:.6f}")
    if not report["control_fired"]:
        print("WARNING: positive control did NOT fire -- instrument is UNTRUSTED")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
