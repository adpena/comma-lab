"""ddm_fo2h -- re-score the pose-null seg edit's pose leg on the DALI GT lineage.

THE QUESTION THIS SETTLES.  The eta gate (`ddm_rt1_eta_gate_pose_constrained.py`) measures
`d_pose_before/after` against GT frames decoded by **PyAV**.  `ddm_up1` measured that local PyAV
GT puts d_pose ~19x above the contest-CUDA value, and that a **DALI** GT cache reproduces the T4
row at 0.9999x.  Only the GT side changes with the device (`upstream/evaluate.py:31,39`), so the
edit's effect on PoseNet output is lineage-independent -- but the *error it is measured against*
is not:

    d_pose_before = |p_b - g|^2
    d_pose_after  = |p_b - g + D|^2          D = PoseNet(edited) - PoseNet(base)
    excess        = 2 (p_b - g) . D + |D|^2

The cross term is taken against `g`, and `g_pyav != g_dali`.  When |D| is small next to the error
-- fo2h measured excess/before = 37%, so it is -- that cross term dominates, and it can change
SIGN when the lineage changes.  fo2h's PyAV measurement (aggregate ratio 1.3725, pose WORSENS)
therefore does NOT establish the sign on the shipping axis.  This module measures it directly.

WHAT IT DOES.  For each pair with a retained edited camera frame:

    d_pose_*_pyav = compute_distortion(PoseNet(gt_pyav_pair),  PoseNet(cmp_pair))   [control]
    d_pose_*_dali = mean((gt_dali_pose[t] - PoseNet(cmp_pair)['pose'][..., :6])^2)  [shipping]

with `cmp_pair` = `[dec0, dec1]` for BEFORE and `[dec0, cam_edit]` for AFTER -- the exact geometry
of `ddm_rt1_eta_gate_pose_constrained.py:262-263`, so the PyAV column is a RECEIPT: it must
reproduce the eta gate's retained rows to within float noise, or this instrument is not measuring
the same object and no DALI number it prints may be believed.

MATCHED BY CONSTRUCTION.  The pairs are whichever ones have retained frames, and both lineages are
scored on the SAME pairs with the SAME frames in the same process.  The PyAV-vs-DALI difference is
therefore the lineage and nothing else -- never a fresh sample confounded with the axis change.

Axis `[macOS-CPU advisory]` -- NEVER a score.  `score_claim=false`.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

# MUST match ddm_rt1_eta_gate_pose_constrained.DEFAULT_RAW -- a different decode of the same
# pointer would silently change `dec0`/`dec1` and break the eta-gate reproduction control.
DEFAULT_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/"
    "base_optimized_n600_r3/output/0.raw"
)
DEFAULT_GT_DALI = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt"
)
DEFAULT_GT_MKV = REPO / "upstream" / "videos" / "0.mkv"
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening/null_retain12")

H, W, C = 874, 1164, 3
N_PAIRS = 600
FRAME_B = H * W * C


class Fo2hLineageError(RuntimeError):
    """Fail-closed error."""


def progress(work: Path, milestone: str, detail: dict) -> None:
    row = {"arm": "ddm_fo2h_lineage", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "milestone": milestone, "detail": detail, "host": socket.gethostname()}
    work.mkdir(parents=True, exist_ok=True)
    with (work / "PROGRESS.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[fo2h-lineage] {milestone}: {json.dumps(detail, sort_keys=True)}", flush=True)


def retained_pairs(frames_dir: Path) -> list[int]:
    """Pair ids that have a retained edited camera frame on disk, ascending."""
    out = []
    for p in sorted(frames_dir.glob("cam_edit_pair*.npy")):
        m = re.fullmatch(r"cam_edit_pair(\d+)\.npy", p.name)
        if m:
            out.append(int(m.group(1)))
    return out


def open_raw(path: Path) -> np.memmap:
    """The decoded pointer frames, as (1200, H, W, C) uint8."""
    if not path.is_file():
        raise Fo2hLineageError(f"raw does not exist: {path}")
    want = 2 * N_PAIRS * FRAME_B
    got = path.stat().st_size
    if got != want:
        raise Fo2hLineageError(f"raw is {got} B, expected {want} B -- wrong decode or wrong file")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(2 * N_PAIRS, H, W, C))


def score_rows(pairs: list[int], frames_dir: Path, raw: np.memmap,
               gt_dali: np.ndarray, gt_mkv: Path, threads: int) -> list[dict]:
    """Both lineages, same pairs, same frames, one process."""
    from ddm_rt1_eta_gate_pose_constrained import Scorer, decode_gt_frames

    sc = Scorer(threads)
    wanted: set[int] = set()
    for t in pairs:
        wanted.update({2 * t, 2 * t + 1})
    gt_frames = decode_gt_frames(gt_mkv, wanted)

    rows = []
    for t in pairs:
        dec0 = np.asarray(raw[2 * t])
        dec1 = np.asarray(raw[2 * t + 1])
        cam_edit = np.load(frames_dir / f"cam_edit_pair{t:04d}.npy")
        if cam_edit.shape != (H, W, C):
            raise Fo2hLineageError(f"pair {t}: retained frame has shape {cam_edit.shape}")

        out_before = sc.pose_out(np.stack([dec0, dec1]))
        out_after = sc.pose_out(np.stack([dec0, cam_edit]))

        # --- PyAV column: the eta gate's own geometry, kept as the receipt -------------------
        gt_out = sc.pose_out(np.stack([gt_frames[2 * t], gt_frames[2 * t + 1]]))
        pyav_before = sc.d_pose(gt_out, out_before)
        pyav_after = sc.d_pose(gt_out, out_after)

        # --- DALI column: the shipping-axis target, straight from the cache -----------------
        g = gt_dali[t]
        p_before = np.asarray(out_before["pose"].detach().cpu(), dtype=np.float64)[0, :6]
        p_after = np.asarray(out_after["pose"].detach().cpu(), dtype=np.float64)[0, :6]
        dali_before = float(((g - p_before) ** 2).mean())
        dali_after = float(((g - p_after) ** 2).mean())

        rows.append({
            "pair": t,
            "pyav_d_pose_before": pyav_before, "pyav_d_pose_after": pyav_after,
            "pyav_ratio": pyav_after / pyav_before if pyav_before else None,
            "dali_d_pose_before": dali_before, "dali_d_pose_after": dali_after,
            "dali_ratio": dali_after / dali_before if dali_before else None,
            "pyav_excess": pyav_after - pyav_before,
            "dali_excess": dali_after - dali_before,
            "pose_delta_norm2": float(((p_after - p_before) ** 2).sum()),
        })
        r = rows[-1]
        print(f"  pair {t}: pyav x{r['pyav_ratio']:.3f}  dali x{r['dali_ratio']:.3f}", flush=True)
    return rows


def before_side_lineage_factors(pairs: list[int], raw: np.memmap, gt_dali: np.ndarray,
                                gate_rows: dict[int, dict], threads: int) -> list[dict]:
    """Per-pair PyAV/DALI factor on the BEFORE side only -- no edited frames, no GT decode.

    The PyAV before-side value is already on disk in the eta gate's rows, and the DALI one needs
    only `PoseNet([dec0, dec1])` against the cached target.  So the factor that up1 reported as a
    population 19.09x can be resolved PER PAIR for free.  It is not a constant, and that is the
    whole reason a PyAV-measured ratio cannot be assumed to transfer.
    """
    from ddm_rt1_eta_gate_pose_constrained import Scorer

    sc = Scorer(threads)
    out = []
    for t in pairs:
        g = gt_dali[t]
        o = sc.pose_out(np.stack([np.asarray(raw[2 * t]), np.asarray(raw[2 * t + 1])]))
        p = np.asarray(o["pose"].detach().cpu(), dtype=np.float64)[0, :6]
        dali_before = float(((g - p) ** 2).mean())
        pyav_before = float(gate_rows[t]["d_pose_before"])
        out.append({"pair": t, "pyav_before": pyav_before, "dali_before": dali_before,
                    "lineage_factor": pyav_before / dali_before if dali_before else None})
    return out


def aggregate(rows: list[dict], key: str) -> float:
    """mean(after)/mean(before) -- the evaluate.py aggregation, never a mean of ratios."""
    b = np.array([r[f"{key}_d_pose_before"] for r in rows], dtype=np.float64)
    a = np.array([r[f"{key}_d_pose_after"] for r in rows], dtype=np.float64)
    return float(a.mean() / b.mean()) if b.mean() else float("nan")


def lineage_verdict(pyav_agg: float, dali_agg: float, *, tol: float = 1e-12) -> str:
    """Does the pose leg point the same way on both lineages?

    A product-of-signs test alone calls the DEGENERATE case (an edit that moves pose not at all,
    so both aggregates are exactly 1.0) a sign FLIP, which is the opposite of what it is.  The
    identity control -- `cam_edit = dec1` -- is exactly that case and is how this was caught, so
    it gets its own branch rather than being folded into a strict inequality.
    """
    import math
    if math.isnan(pyav_agg) or math.isnan(dali_agg):
        return "UNDETERMINED-NAN-AGGREGATE"
    dp, dd = pyav_agg - 1.0, dali_agg - 1.0
    if abs(dp) <= tol or abs(dd) <= tol:
        return "DEGENERATE-NO-POSE-CHANGE"
    if dp > 0 and dd > 0:
        return "SIGN AGREES ACROSS LINEAGES -- both WORSEN"
    if dp < 0 and dd < 0:
        return "SIGN AGREES ACROSS LINEAGES -- both IMPROVE"
    return "SIGN FLIPS ACROSS LINEAGES"


def control_vs_eta_gate(rows: list[dict], gate_rows_path: Path) -> dict:
    """The PyAV column MUST reproduce the eta gate's retained rows, or nothing here is believable."""
    if not gate_rows_path.is_file():
        return {"checked": False, "reason": f"no gate rows at {gate_rows_path}"}
    gate = {json.loads(x)["pair"]: json.loads(x)
            for x in gate_rows_path.read_text().splitlines() if x.strip()}
    worst_b = worst_a = 0.0
    n = 0
    for r in rows:
        g = gate.get(r["pair"])
        if not g:
            continue
        n += 1
        worst_b = max(worst_b, abs(r["pyav_d_pose_before"] - g["d_pose_before"])
                      / max(abs(g["d_pose_before"]), 1e-30))
        worst_a = max(worst_a, abs(r["pyav_d_pose_after"] - g["d_pose_after"])
                      / max(abs(g["d_pose_after"]), 1e-30))
    return {"checked": True, "n_compared": n,
            "worst_rel_err_before": worst_b, "worst_rel_err_after": worst_a,
            "reproduces_eta_gate": bool(n > 0 and worst_b < 1e-9 and worst_a < 1e-9)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--gt-dali", type=Path, default=DEFAULT_GT_DALI)
    ap.add_argument("--gt-mkv", type=Path, default=DEFAULT_GT_MKV)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--before-side-from-rows", type=Path, default=None,
                    help="eta-gate ETA_GATE_ROWS.jsonl; measure the per-pair BEFORE-side "
                         "PyAV/DALI factor for every pair in it and exit (no edited frames "
                         "needed, no GT decode)")
    args = ap.parse_args(argv)

    from ddm_up1_decode_axis_photometric_probe import load_gt_poses

    out_dir = args.out or args.frames_dir
    gt_dali, lineage = load_gt_poses(args.gt_dali)
    if lineage != "dali":
        raise Fo2hLineageError(
            f"gt cache at {args.gt_dali} reports lineage {lineage!r}, not 'dali' -- refusing: "
            "the whole point of this instrument is which GT it holds")

    if args.before_side_from_rows is not None:
        gate_rows = {json.loads(x)["pair"]: json.loads(x)
                     for x in args.before_side_from_rows.read_text().splitlines() if x.strip()}
        pairs = sorted(gate_rows)
        rows = before_side_lineage_factors(pairs, open_raw(args.raw), gt_dali, gate_rows,
                                           args.threads)
        f = np.array([r["lineage_factor"] for r in rows], dtype=np.float64)
        summary = {
            "schema": "ddm_fo2h_before_side_lineage_factor.v1",
            "axis": "[macOS-CPU advisory] frozen CPU-torch PoseNet -- NEVER a score",
            "score_claim": False, "promotable": False, "pointer_moved": False,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_pairs": len(rows), "gt_lineage": lineage,
            "factor_min": float(f.min()), "factor_p25": float(np.percentile(f, 25)),
            "factor_median": float(np.median(f)), "factor_p75": float(np.percentile(f, 75)),
            "factor_max": float(f.max()),
            "population_factor_pooled": float(
                np.array([r["pyav_before"] for r in rows]).mean()
                / np.array([r["dali_before"] for r in rows]).mean()),
            "up1_population_reference": 19.09,
            "note": "the per-pair factor is NOT a constant; a population figure may never be "
                    "applied per pair (the m88 genus one level down)",
            "rows": rows,
        }
        p = (args.out or args.frames_dir) / "FO2H_BEFORE_SIDE_LINEAGE_FACTOR.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=1, sort_keys=True))
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                         indent=1, sort_keys=True))
        return 0

    pairs = retained_pairs(args.frames_dir)
    if not pairs:
        raise Fo2hLineageError(f"no retained cam_edit_pair*.npy under {args.frames_dir}")
    progress(out_dir, "start", {"n_pairs": len(pairs), "pairs": pairs, "lineage": lineage})

    raw = open_raw(args.raw)
    rows = score_rows(pairs, args.frames_dir, raw, gt_dali, args.gt_mkv, args.threads)

    pyav_agg, dali_agg = aggregate(rows, "pyav"), aggregate(rows, "dali")
    ctrl = control_vs_eta_gate(rows, args.frames_dir / "ETA_GATE_ROWS.jsonl")
    verdict = lineage_verdict(pyav_agg, dali_agg)

    summary = {
        "schema": "ddm_fo2h_pose_lineage_rescore.v1",
        "axis": "[macOS-CPU advisory] frozen CPU-torch PoseNet -- NEVER a score",
        "score_claim": False, "promotable": False, "pointer_moved": False,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pairs": len(rows), "pairs": [r["pair"] for r in rows],
        "gt_dali_cache": str(args.gt_dali), "gt_lineage": lineage,
        "aggregate_ratio_pyav": pyav_agg,
        "aggregate_ratio_dali": dali_agg,
        "pairs_worsening_pyav": int(sum(1 for r in rows if r["pyav_ratio"] > 1.0)),
        "pairs_worsening_dali": int(sum(1 for r in rows if r["dali_ratio"] > 1.0)),
        "control_vs_eta_gate": ctrl,
        "verdict": verdict,
        "verdict_scope": "INSTANCE: hv1 ep0634 base, ring-0 described set, r=1 pose-null "
                         "realization, the retained pairs only; a matched same-pair lineage "
                         "contrast, NOT a population estimate of either ratio",
        "rows": rows,
    }
    out_path = out_dir / "FO2H_POSE_LINEAGE_RESCORE.json"
    out_path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    progress(out_dir, "done", {"pyav": pyav_agg, "dali": dali_agg, "verdict": verdict,
                               "control_ok": ctrl.get("reproduces_eta_gate")})
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
