#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ft1 — THE CLIP-FRACTION FALSIFIER for photometric<->beta coordinate staleness.

WHAT IS UNDER TEST.  ddm_fs1 MEASURED that 244 of 600 shipped pairs carry an
auto-exposure ``(a,b)`` fitted at ``beta=0`` together with a NON-ZERO shipped
rolling-shutter ``beta`` (59 of them outside the ``{0,0.5,1.0}`` fitted hull).
The pre-registered PREDICTION was: the (a,b)<->beta coupling lives in the uint8
CLAMP, so its magnitude should scale with the SATURATED-PIXEL FRACTION per pair,
and pairs with ~0 clipping need no correction at all.

WHAT THIS TOOL MEASURES (scorer-free, $0, n600).  Per pair, through the REAL
receiver primitives (``inflate_runner_v4d.Decoder._warp_pair`` +
``pfs1_warp_receiver._to_uint8`` -- imported, never re-implemented), it forms
the two pre-photometric composites the fit and the ship correspond to::

    X_0    = _warp_pair(f1, pose, s_t, sel, rot=1.0)            # beta = 0 (fit)
    X_beta = (1-alpha)*_warp_pair(..., 1-beta/2)
             +   alpha*_warp_pair(..., 1+beta/2)                # shipped beta
    Y_. = a*X_. + b                                             # shipped (a,b)

and decomposes the total change ``uint8(Y_beta) != uint8(Y_0)`` into the three
channels that can carry coupling:

  CLAMP channel   ``clip_sym_diff`` -- pixels whose SATURATION STATUS flips
      between the two geometries.  This is the pre-registered predictor: it is
      the only place where the intensity map ``a*(.)+b`` and the warp fail to
      commute as OPERATORS, so it bounds the clamp-mediated coupling exactly.
  ROUND channel   ``round_only_frac`` -- pixels the geometry moved by LESS than
      half a quantization step that nonetheless changed uint8 value.  ``_to_uint8``
      is ``clip(round(.))``: ROUND is non-invertible too, and it bites on every
      pixel, not only saturated ones.  The pre-registered derivation named only
      the clamp; this channel is measured because the receiver contains it.
  GEOMETRY channel ``pre_rms`` / ``u8_diff_frac`` -- how much the image actually
      moved.  Even with NO quantization at all, ``d2L/d(a,b)dbeta != 0`` because
      PoseNet is NONLINEAR in the frame and ``X_beta != X_0``; that coupling is
      O(||dX/dbeta||) and is invisible to any clip-fraction argument.

CONTROL (this is the derivation's own load-bearing algebraic claim, so it is
executed, not asserted): ``warp_rgb`` is linear and constant-preserving, hence
``W(a*I + b) == a*W(I) + b`` exactly.  ``--control`` measures that residual.  If
it is at float round-off the OPERATORS commute and uint8 is the sole commutator
source; if it is not, the derivation is wrong at its first step.

Axis: [macOS-CPU scorer-free advisory] NON-PROMOTABLE.  score_claim=false.  No
PoseNet forward is executed anywhere in this tool -- the n600 scorer slot stays
free for the live job.

Usage::

    .venv/bin/python tools/ddm_ft1_clip_fraction_falsifier.py \
        --archive-dir <extracted v4d archive> \
        --final-jsonl .../mq1_emit/final_mq1.jsonl \
        --out-dir <ssd run dir>/ft1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "2")  # the n600 scorer job owns the machine

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "src", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.ddm_ft1_photometric_beta_commutator_20260802 import (  # noqa: E402
    FIT_MENU_MAGNITUDES,
    commutator_channels,
)

V4D_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731")
RECV_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/submission")


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_final(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows


def _row_beta_mag(row: dict, table: tuple[float, ...]) -> float:
    """Signed magnitude a final-JSONL row selected (mirrors the builder)."""
    if "beta_mag" in row:
        return float(row["beta_mag"])
    idx = int(row["beta_idx"])
    if not 0 <= idx < len(table):
        raise SystemExit(f"row has beta_idx={idx} and no beta_mag; "
                         "the magnitude it chose is not recoverable")
    return float(table[idx])


def _import_receiver(v4d_dir: Path, recv_dir: Path):
    for p in (str(v4d_dir), str(recv_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import inflate_runner_v4d as runner
    return runner


def run_control(dec, runner, n: int = 3) -> dict:
    """Execute the derivation's algebraic premise: does W commute with a*I+b?

    Returns the worst absolute and relative residual of
    ``W(a*I + b) - (a*W(I) + b)`` over ``n`` pairs at the SHIPPED (a,b).  A
    residual at float round-off CONFIRMS the operator-commutation premise; a
    large one FALSIFIES the derivation at step one.
    """
    worst_abs, worst_rel = 0.0, 0.0
    for i in range(min(n, dec.n_pairs)):
        f1 = dec.f1(i).astype(np.float64)
        a, b = float(dec.ab[i][0]), float(dec.ab[i][1])
        pose, s_t, sel = dec.p_best[i], float(dec.st_vals[dec.st_idx[i]]), int(dec.sel[i])
        lhs = dec._warp_pair(a * f1 + b, pose, s_t, sel, 1.0)   # W(aI+b)
        rhs = a * dec._warp_pair(f1, pose, s_t, sel, 1.0) + b   # aW(I)+b
        d = np.abs(lhs - rhs)
        worst_abs = max(worst_abs, float(d.max()))
        scale = max(float(np.abs(rhs).max()), 1e-12)
        worst_rel = max(worst_rel, float(d.max()) / scale)
    return {"pairs": int(min(n, dec.n_pairs)),
            "max_abs_residual": worst_abs, "max_rel_residual": worst_rel,
            "commutes": bool(worst_rel < 1e-12)}


def run_receiver_parity(dec, runner, n: int = 3) -> dict:
    """Positive control that this tool's compose IS the receiver's compose."""
    worst = 0
    for i in range(min(n, dec.n_pairs)):
        f1 = dec.f1(i)
        ours = _compose_u8(dec, runner, f1.astype(np.float64), i,
                           dec.p_best[i], float(dec.st_vals[dec.st_idx[i]]),
                           int(dec.sel[i]), float(dec.ab[i][0]),
                           float(dec.ab[i][1]),
                           float(dec.beta_mags[int(dec.beta_idx[i])]))
        theirs = dec.f0(i, f1)
        worst = max(worst, int(np.abs(ours.astype(np.int32)
                                      - theirs.astype(np.int32)).max()))
    return {"pairs": int(min(n, dec.n_pairs)), "max_abs_u8_diff": worst,
            "bit_identical": bool(worst == 0)}


def _pre_photometric(dec, f1_f, pose, s_t, sel, g):
    """The receiver's pre-photometric composite at signed magnitude ``g``."""
    if g == 0.0:
        return dec._warp_pair(f1_f, pose, s_t, sel, 1.0)
    beta = g * (1.0 if pose[5] >= 0.0 else -1.0)
    return ((1.0 - dec._alpha) * dec._warp_pair(f1_f, pose, s_t, sel,
                                                1.0 - beta / 2.0)
            + dec._alpha * dec._warp_pair(f1_f, pose, s_t, sel,
                                          1.0 + beta / 2.0))


def _compose_u8(dec, runner, f1_f, _i, pose, s_t, sel, a, b, g):
    f0f = _pre_photometric(dec, f1_f, pose, s_t, sel, g)
    if a != 1.0 or b != 0.0:
        f0f = a * f0f + b
    return runner._to_uint8(f0f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", required=True,
                    help="extracted v4d archive (manifest.json + state/)")
    ap.add_argument("--final-jsonl", required=True,
                    help="shipped per-pair table under test")
    ap.add_argument("--fit-jsonl", default=None,
                    help="table the (a,b) was actually FITTED against.  Default "
                         "(unset) isolates the beta axis alone: the fit geometry "
                         "is the shipped pose at beta=0.  Set it to the "
                         "pre-refinement table to measure the FULL fit->ship "
                         "displacement, which also carries any pose dims a later "
                         "stage moved (MEASURED: mq1 moved p1/p2 on 128 rows).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--v4d-dir", default=str(V4D_DIR))
    ap.add_argument("--recv-dir", default=str(RECV_DIR))
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = all (n600)")
    ap.add_argument("--control-pairs", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jl_path = out_dir / "ft1_channels.jsonl"

    runner = _import_receiver(Path(args.v4d_dir), Path(args.recv_dir))
    dec = runner.Decoder(Path(args.archive_dir))
    final = _load_final(Path(args.final_jsonl))
    fit = _load_final(Path(args.fit_jsonl)) if args.fit_jsonl else None
    n = args.n_pairs or dec.n_pairs
    missing = [i for i in range(n) if i not in final
               or (fit is not None and i not in fit)]
    if missing:
        raise SystemExit(f"final/fit jsonl missing {len(missing)} of {n} pairs "
                         f"(first {missing[:5]}) -- refusing a partial verdict")
    if fit is not None:
        moved_ab = [i for i in range(n)
                    if float(fit[i]["a"]) != float(final[i]["a"])
                    or float(fit[i]["b"]) != float(final[i]["b"])]
        if moved_ab:
            raise SystemExit(
                f"{len(moved_ab)} pairs have a DIFFERENT (a,b) between the fit "
                "table and the shipped table, so the shipped (a,b) was not "
                "fitted against the fit table -- the fit context is wrong and "
                "the displacement would be meaningless")

    control = run_control(dec, runner, args.control_pairs)
    parity = run_receiver_parity(dec, runner, args.control_pairs)
    if not parity["bit_identical"]:
        raise SystemExit(f"receiver parity control FAILED: {parity}")

    # Resume is a STALENESS hazard in this tool's own output: rows computed
    # against one displacement axis are indistinguishable from rows computed
    # against another once they are in the file.  Pin the config and fail
    # closed rather than silently interleaving two experiments.
    cfg = {"final_jsonl": str(Path(args.final_jsonl).resolve()),
           "fit_jsonl": str(Path(args.fit_jsonl).resolve()) if args.fit_jsonl
                        else None,
           "archive_dir": str(Path(args.archive_dir).resolve())}
    cfg_path = out_dir / "ft1_config.json"
    if cfg_path.exists():
        prior = json.loads(cfg_path.read_text())
        if prior != cfg:
            raise SystemExit(
                f"out-dir {out_dir} already holds rows measured under a "
                f"DIFFERENT config:\n  prior {prior}\n  now   {cfg}\n"
                "resuming would interleave two experiments -- use a fresh "
                "--out-dir")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")

    done: set[int] = set()
    if jl_path.exists():
        for ln in jl_path.read_text().splitlines():
            if ln.strip():
                done.add(int(json.loads(ln)["pair"]))

    sink = jl_path.open("a")
    for i in range(n):
        if i in done:
            continue
        row = final[i]
        pose = np.asarray(row["p"], np.float64)
        a, b = float(row["a"]), float(row["b"])
        sel = int(row["selector"])
        g = _row_beta_mag(row, dec.beta_mags)
        s_t = float(dec.st_vals[dec.st_idx[i]])

        # The (a,b) Gauss-Newton always ran at rot_scale=1.0, i.e. beta=0, on
        # the pose CURRENT AT THAT MOMENT -- which is the fit table's pose when
        # one is supplied and the shipped pose otherwise.
        pose_fit = (np.asarray(fit[i]["p"], np.float64) if fit is not None
                    else pose)
        drifted = ([k for k in range(6)
                    if float(pose_fit[k]) != float(pose[k])]
                   if fit is not None else [])

        f1_f = dec.f1(i).astype(np.float64)
        x0 = _pre_photometric(dec, f1_f, pose_fit, s_t, sel, 0.0)
        same_geometry = g == 0.0 and not drifted
        xb = x0 if same_geometry else _pre_photometric(dec, f1_f, pose, s_t,
                                                       sel, g)
        y0 = a * x0 + b if (a != 1.0 or b != 0.0) else x0
        yb = a * xb + b if (a != 1.0 or b != 0.0) else xb

        ch = commutator_channels(y_fit=y0, y_ship=yb)
        ch.update({"pair": i, "a": a, "b": b, "selector": sel, "beta_mag": g,
                   "abs_delta_beta": abs(g), "s_t": s_t,
                   "yaw_sign": 1.0 if pose[5] >= 0.0 else -1.0,
                   "ab_identity": bool(a == 1.0 and b == 0.0),
                   "outside_fit_hull": bool(abs(g) > max(FIT_MENU_MAGNITUDES)),
                   "opposes_yaw": bool(g < 0.0),
                   "pose_dims_drifted": drifted,
                   "stale": bool((a != 1.0 or b != 0.0) and (g != 0.0 or drifted)),
                   "source": row.get("source", "?")})
        sink.write(json.dumps(ch) + "\n")
        sink.flush()
        if i % 25 == 0:
            print(f"[{_utc()}] pair {i}/{n} beta={g:+.4f} "
                  f"clip_sym={ch['clip_sym_diff']:.3e} "
                  f"u8diff={ch['u8_diff_frac']:.3e}", flush=True)
    sink.close()

    rows = [json.loads(ln) for ln in jl_path.read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if r["pair"] < n]
    if len(rows) != n:
        # An empty or short population must never emit a clean-looking receipt.
        raise SystemExit(f"expected {n} rows, have {len(rows)} -- refusing to "
                         "emit a receipt over an incomplete population")
    receipt = {
        "schema": "ddm_ft1_clip_fraction_falsifier.v1",
        "axis": "[macOS-CPU scorer-free advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False, "research_only": True,
        "generated_at_utc": _utc(),
        "generated_by": "tools/ddm_ft1_clip_fraction_falsifier.py",
        "archive_dir": str(args.archive_dir),
        "final_jsonl": str(args.final_jsonl),
        "fit_jsonl": str(args.fit_jsonl) if args.fit_jsonl else None,
        "displacement_axis": ("full_fit_to_ship" if args.fit_jsonl
                              else "beta_axis_only"),
        "n_pairs": len(rows),
        "operator_commutation_control": control,
        "receiver_parity_control": parity,
        "fit_menu_magnitudes": list(FIT_MENU_MAGNITUDES),
        "note": "no PoseNet forward executed; d_pose coupling itself is NOT "
                "measured here -- this tool measures the three IMAGE-domain "
                "channels through which such coupling can travel.",
    }
    (out_dir / "ft1_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
