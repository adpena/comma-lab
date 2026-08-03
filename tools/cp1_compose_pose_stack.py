#!/usr/bin/env python3
"""ddm_cp1 -- do the measured pose-line wins COMPOSE, or are they one win counted twice?

MEASURED STRUCTURAL FACT that motivates this tool (from the archives, not inferred):
``state/tokens.dr7t`` is BYTE-IDENTICAL (sha ``305a2be9...``, 346,478 B) across
``pw1``, ``ms8``, ``dc1_fold`` and ``pj2``.  Those four archives differ ONLY in
``manifest.json`` and ``state/pose_warp.stp``.  So they are not four stack layers --
they are four successive solutions of ONE member, and a naive sum of their ΔS
would double-count.

What DOES compose inside that member, because they act on different coordinates:

* ``pj2``      chooses the pose VALUES (8 coords/pair: 6 pose + a,b), solved.
* ``dc1_fold`` chooses the ``s_t`` REPRESENTATION (which grid + which index), by
  folding the effective scale into ``p[0:3]`` -- legal because
  ``pfs1_warp_receiver.pose_to_homography`` uses the translation only through the
  product ``s_t * [p2,p1,p0]``, an exact multiplicative degeneracy
  (``ddm_mq1`` 5.98e-16, ``ddm_dc1`` 4.539e-16, ``ddm_pj2`` 7.38e-16).

``pj2`` shipped on ms8's FITTED ``st_grid`` and therefore inherited ms8's +51 B.
This tool re-expresses pj2's solved geometry at the INCUMBENT (vendored) grid.
The homography is invariant BY ALGEBRA; the only thing that can move is float16
STORAGE rounding of the rescaled ``p[0:3]`` and the re-derived dim0 offset -- so
the realized ``d_pose`` is MEASURED per pair through the real receiver + the
frozen PoseNet, never assumed.

Modes
-----
``--mode fold``    emit the folded builder-ready final JSONL + the $0 algebraic
                   homography-identity leg + the $0 byte prediction.
``--mode score``   realized n600 ``d_pose`` of the QUANTIZED folded pose decoded
                   at the INCUMBENT ``s_t`` (shardable, resumable).
``--mode report``  n600 census + composed-S arithmetic against the live base.

Axis: ``[macOS-CPU frozen-PoseNet advisory]`` NON-PROMOTABLE.  ``score_claim=false``.
No training, no paid dispatch, NO exact gate fired, pointer untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_cp1_compose_pose_stack.v1"

N_PAIRS = 600
ARCHIVE_DENOM = 37_545_489.0

#: PR130's demonstrated floor, and the live own-vehicle base this unit composes
#: onto.  Both are quoted so every ΔS below carries its denominator
#: (``tac.canonical_equations.gap_decomposition_against_floor_20260802``).
PR130_BAR_S = 0.172141
DC1_FOLD_BYTES = 360_309
DC1_FOLD_SEG_TERM = 0.431179
DC1_FOLD_DPOSE = 0.00516574

#: ``ddm_ms8``'s MEASURED instrument floor on this exact vehicle.  A canary above
#: this means the harness is not reproducing the shipped decode and NO verdict
#: from it is admissible.
CANARY_MAX_ABS_ERR = 1.2e-05

MS8_OVERRIDE = Path("/Volumes/VertigoDataTier/pact/ddm_ms8_20260802/"
                    "ms8_st_override.json")
PJ2_FINAL = Path("/Volumes/VertigoDataTier/pact/ddm_pj2_20260802/final_pj2.jsonl")
PJ2_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/"
                   "v4d_composed_pj2_archive.zip")
DC1_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/"
                   "v4d_composed_dc1_fold_archive.zip")
OUT_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_cp1_20260802")


def _utc() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[int(r["pair"])] = r
    return out


def contribution(d_pose_mean: float) -> float:
    """The score's own pose term.  Non-linear, which is why the tail IS the term."""
    return math.sqrt(10.0 * float(d_pose_mean))


def rate_term(nbytes: int) -> float:
    return 25.0 * float(nbytes) / ARCHIVE_DENOM


def composed_S(seg_term: float, d_pose_mean: float, nbytes: int) -> dict:
    """Recompute S from COMPONENTS.  The rounded headline field lies."""
    pose = contribution(d_pose_mean)
    rate = rate_term(nbytes)
    total = float(seg_term) + pose + rate
    return {"seg": float(seg_term), "pose": pose, "rate": rate, "S": total,
            "gap_to_bar": total - PR130_BAR_S}


def incumbent_s_t() -> dict[int, float]:
    """The per-pair ``s_t`` the INCUMBENT (vendored) ladder actually ships.

    Read through ``ddm_v4c_resolve`` -- the same accessor ``ddm_dc1`` used -- so
    the fold this tool computes is the same object dc1 measured, not a second
    implementation that could agree for the wrong reason.
    """
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import ddm_v4c_resolve as v4c

    return {i: float(v4c._d2_row(i)["s_t"]) for i in range(N_PAIRS)}


def fold_to_incumbent(final: dict[int, dict], ms8_doc: dict,
                      s_ship: dict[int, float]):
    """Re-express a solution stored at ms8's FITTED grid on the INCUMBENT grid.

    Delegates the arithmetic to ``tools.dc1_menu_sweep.build_rescaled_poses`` so
    there is exactly ONE implementation of the fold in the repo.  That helper
    also re-derives the dim0 offset exactly as the builder does under
    ``--dim0-offset auto``, which is the one place the degeneracy is NOT exact
    (``p[0]`` ships as an f16 residual off a manifest offset, so its ABSOLUTE
    quantum depends on the residual magnitude).
    """
    sys.path.insert(0, str(REPO / "tools"))
    from dc1_menu_sweep import build_rescaled_poses

    return build_rescaled_poses(final, ms8_doc, s_ship)


# --------------------------------------------------------------------------- #
# mode: fold -- $0.  Emit the folded final JSONL + algebraic + byte legs.
# --------------------------------------------------------------------------- #
def run_fold(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = _load_jsonl(args.final_jsonl)
    ms8_doc = json.loads(MS8_OVERRIDE.read_text())
    grid = [float(x) for x in ms8_doc["st_grid"]]
    midx = [int(x) for x in ms8_doc["st_idx"]]

    # CONTROL: the final JSONL's own recorded s_t must agree with the override
    # it was solved against.  A silent mismatch here would fold the WRONG scale.
    mism = [i for i in range(N_PAIRS)
            if "s_t_ms8" in final[i]
            and abs(float(final[i]["s_t_ms8"]) - grid[midx[i]]) > 0.0]
    if mism:
        raise SystemExit(
            f"{len(mism)} rows record an s_t that differs from the ms8 override "
            f"they were solved against (first {mism[:5]}); folding would apply "
            "the wrong scale")

    s_ship = incumbent_s_t()
    resc, qpose, kfac, offset = fold_to_incumbent(final, ms8_doc, s_ship)

    # ALGEBRAIC LEG -- free, no scorer.  Identical homographies => bit-identical
    # rendered f0 => identical d_pose BY CONSTRUCTION.  Uses the SAME receiver
    # object the scorer path uses.
    sys.path.insert(0, str(REPO / "experiments"))
    import ddm_v4c_resolve as v4c

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)
    p2h = comp.recv.pose_to_homography
    K, Kinv = oracle.K, oracle.Kinv
    rel = []
    for i in range(N_PAIRS):
        p = np.asarray(final[i]["p"], np.float64)
        h_fit = p2h(p, K, Kinv, float(grid[midx[i]]), 1.0, 0.0)
        h_fold = p2h(resc[i], K, Kinv, s_ship[i], 1.0, 0.0)
        rel.append(float(np.abs(h_fit - h_fold).max()
                         / max(np.abs(h_fit).max(), 1e-300)))
    rel_max = float(max(rel))
    print(f"[cp1 fold] ALGEBRAIC leg: max relative homography difference over "
          f"{N_PAIRS} pairs = {rel_max:.3e}", flush=True)
    if rel_max > 1e-12:
        raise SystemExit(
            f"the fold is NOT the identity at {rel_max:.3e}; refusing to emit")

    # BYTE LEG -- free.  The real shipped encoder on the real member.
    sys.path.insert(0, str(REPO / "tools"))
    from dc1_menu_sweep import pose_member_bytes

    pj2_manifest = json.loads(zipfile.ZipFile(PJ2_ARCHIVE).read("manifest.json"))
    poses_fit = np.asarray([final[i]["p"] for i in range(N_PAIRS)], np.float64)
    tp_fit = pose_member_bytes(poses_fit, pj2_manifest.get("pose_dim0_offset"))
    tp_fold = pose_member_bytes(resc, offset)
    print(f"[cp1 fold] BYTE leg: tp_member {tp_fit} -> {tp_fold} B "
          f"({tp_fold - tp_fit:+d}); dim0_offset {pj2_manifest.get('pose_dim0_offset')}"
          f" -> {offset}", flush=True)

    rows = []
    for i in range(N_PAIRS):
        r = dict(final[i])
        r["p"] = [float(x) for x in resc[i]]
        r["s_t_incumbent"] = s_ship[i]
        r["k_fold"] = float(kfac[i])
        r["source"] = "cp1_pj2_folded_to_incumbent"
        rows.append(r)
    args.emit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.emit_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))

    receipt = args.out_dir / "cp1_fold_receipt.json"
    receipt.write_text(json.dumps({
        "schema": SCHEMA, "mode": "fold", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False, "pointer_moved": False,
        "source_final_jsonl": str(args.final_jsonl),
        "algebraic_max_rel_homography_err": rel_max,
        "dim0_offset_fitted_grid": pj2_manifest.get("pose_dim0_offset"),
        "dim0_offset_incumbent_refolded": offset,
        "tp_member_bytes_fitted_grid": tp_fit,
        "tp_member_bytes_folded": tp_fold,
        "k_min": float(kfac.min()), "k_max": float(kfac.max()),
        "pairs_with_k_ne_1": int((kfac != 1.0).sum()),
        "emit_jsonl": str(args.emit_jsonl),
    }, indent=1) + "\n")
    print(f"[cp1 fold] rows={len(rows)} -> {args.emit_jsonl}\n"
          f"[cp1 fold] receipt -> {receipt}", flush=True)


# --------------------------------------------------------------------------- #
# mode: score -- realized n600 d_pose of the QUANTIZED folded pose.
# --------------------------------------------------------------------------- #
def run_score(args: argparse.Namespace) -> None:
    for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_tv] = "1"
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    torch.set_num_threads(1)
    import ddm_v4c_resolve as v4c

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = _load_jsonl(args.final_jsonl)
    ms8_doc = json.loads(MS8_OVERRIDE.read_text())
    s_ship = incumbent_s_t()
    _resc, qpose, kfac, offset = fold_to_incumbent(final, ms8_doc, s_ship)

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)
    grid = [float(x) for x in ms8_doc["st_grid"]]
    midx = [int(x) for x in ms8_doc["st_idx"]]

    def score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g):
        """The receiver's own f0, branch for branch (ddm_pj2 §2)."""
        beta = g * (1.0 if pose[5] >= 0.0 else -1.0)
        wg_t, wf_t = comp.warps(f1_f, pose, s_t, 1.0 - beta / 2.0)
        wg_b, wf_b = comp.warps(f1_f, pose, s_t, 1.0 + beta / 2.0)
        f0_t = np.where(comp.far[..., None], wf_t, wg_t) if sel else wg_t
        f0_b = np.where(comp.far[..., None], wf_b, wg_b) if sel else wg_b
        f0 = (1.0 - comp.alpha_row) * f0_t + comp.alpha_row * f0_b
        if a != 1.0 or b != 0.0:
            f0 = a * f0 + b
        p6 = comp.o.p3v2.pose6_u8(comp.o.posenet, comp.recv._to_uint8(f0), f1_u8)
        return float(np.mean((p6 - tp) ** 2))

    seq = [p for p in range(min(args.pairs, N_PAIRS))
           if p % args.nshards == args.shard]
    jl = args.out_dir / f"cp1_score_shard{args.shard}.jsonl"
    cache = {int(json.loads(ln)["pair"]) for ln in
             (jl.read_text().splitlines() if jl.exists() else []) if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[cp1 score] shard {args.shard}/{args.nshards} pairs={len(seq)} "
          f"cached={len(cache)} dim0_offset={offset}", flush=True)

    done = 0
    for pidx in seq:
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[cp1 score] wall cap at pair {pidx}; rerun to resume",
                  flush=True)
            break
        sh = final[pidx]
        a, b = float(sh["a"]), float(sh["b"])
        sel = int(sh["selector"])
        g = float(sh["beta_mag"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        # CANARY: the UNFOLDED solution re-scored here must reproduce the value
        # the source arm reported.  An instrument that cannot reproduce a known
        # row may not publish a verdict from a new one (ddm_ms8 canary law).
        canary = None
        if args.canary:
            p_fit = np.asarray(sh["p"], np.float64)
            canary = abs(score(p_fit, float(grid[midx[pidx]]), sel, f1_u8, f1_f,
                               tp, a, b, g) - float(sh["d_final"]))
        d_fold_q = score(qpose[pidx], s_ship[pidx], sel, f1_u8, f1_f, tp, a, b, g)
        fj.write(json.dumps({
            "pair": int(pidx), "k": float(kfac[pidx]),
            "s_incumbent": s_ship[pidx], "s_fitted": float(grid[midx[pidx]]),
            "d_source_reported": float(sh["d_final"]),
            "d_folded_quantized_at_incumbent_st": d_fold_q,
            "canary_abs_err": canary,
        }) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        done += 1
        if done % 25 == 0 or done <= 2:
            el = time.time() - t0
            print(f"[cp1 score s{args.shard} {done}/{len(seq)}] pair {pidx} "
                  f"k {kfac[pidx]:.4f} d {float(sh['d_final']):.6f} -> "
                  f"{d_fold_q:.6f} ({el:.0f}s, {el / max(done, 1):.2f}s/pair)",
                  flush=True)
    fj.close()
    print(f"[cp1 score] shard {args.shard} done={done} {time.time() - t0:.0f}s",
          flush=True)


# --------------------------------------------------------------------------- #
# mode: report -- n600 census + composed S.
# --------------------------------------------------------------------------- #
def run_report(args: argparse.Namespace) -> None:
    rows: dict[int, dict] = {}
    for path in sorted(args.out_dir.glob("cp1_score_shard*.jsonl")):
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[int(r["pair"])] = r
    missing = [i for i in range(N_PAIRS) if i not in rows]
    if missing and not args.allow_partial:
        raise SystemExit(
            f"n600 IS the evidence bar: {len(rows)}/{N_PAIRS} scored, first "
            f"missing {missing[:5]}.  Pass --allow-partial only for a progress "
            "read, never for a verdict.")

    pairs = sorted(rows)
    can = [rows[p]["canary_abs_err"] for p in pairs
           if rows[p].get("canary_abs_err") is not None]
    canary_max = float(max(can)) if can else None
    if canary_max is not None and canary_max > CANARY_MAX_ABS_ERR:
        raise SystemExit(
            f"CANARY {canary_max:.3e} exceeds the measured instrument floor "
            f"{CANARY_MAX_ABS_ERR:.3e}: this harness is not reproducing the "
            "shipped decode, so no verdict from it is admissible")

    d_src = np.asarray([rows[p]["d_source_reported"] for p in pairs])
    d_fold = np.asarray([rows[p]["d_folded_quantized_at_incumbent_st"]
                         for p in pairs])
    delta = d_fold - d_src

    nbytes = int(args.archive_bytes) if args.archive_bytes else None
    base = composed_S(DC1_FOLD_SEG_TERM, DC1_FOLD_DPOSE, DC1_FOLD_BYTES)
    src = composed_S(DC1_FOLD_SEG_TERM, float(d_src.mean()), args.source_bytes)
    out = {
        "schema": SCHEMA, "mode": "report", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False, "pointer_moved": False,
        "n_scored": len(pairs),
        "canary_max_abs_err": canary_max,
        "canary_floor": CANARY_MAX_ABS_ERR,
        "d_pose_mean_source": float(d_src.mean()),
        "d_pose_mean_folded": float(d_fold.mean()),
        "d_pose_mean_delta": float(delta.mean()),
        "pairs_worse_after_fold": int((delta > 0).sum()),
        "pairs_better_after_fold": int((delta < 0).sum()),
        "pairs_identical_after_fold": int((delta == 0).sum()),
        "max_abs_delta": float(np.abs(delta).max()),
        "base_dc1_fold": base,
        "source_at_fitted_grid": src,
    }
    if nbytes is not None:
        out["composed_folded"] = composed_S(DC1_FOLD_SEG_TERM,
                                            float(d_fold.mean()), nbytes)
        out["composed_folded"]["archive_bytes"] = nbytes
        out["delta_S_vs_base"] = out["composed_folded"]["S"] - base["S"]
        out["pct_of_gap_vs_base"] = (
            100.0 * (base["S"] - out["composed_folded"]["S"]) / base["gap_to_bar"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"[cp1 report] -> {args.out}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=("fold", "score", "report"))
    ap.add_argument("--base", default="celldrop50")
    ap.add_argument("--final-jsonl", type=Path, default=PJ2_FINAL)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--emit-jsonl", type=Path,
                    default=OUT_DIR / "final_cp1_pj2_fold.jsonl")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "cp1_report.json")
    # n600 IS the evidence bar (CLAUDE.md "allergic to non-n600-scale"); a
    # subset must be asked for explicitly and can never carry a verdict.
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=600.0)
    ap.add_argument("--canary", action="store_true",
                    help="re-score the UNFOLDED solution too and record the "
                         "absolute error against the source arm's value")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--archive-bytes", type=int, default=0)
    ap.add_argument("--source-bytes", type=int, default=360_406)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    {"fold": run_fold, "score": run_score, "report": run_report}[args.mode](args)


if __name__ == "__main__":
    main()
