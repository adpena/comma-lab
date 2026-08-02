#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_dc1 — the LABEL (which-class) half of the correction-stream price.

MEANS. pointer 0.1910828242 [contest-CPU] UNMOVED. Authority: [macOS-CPU advisory]
NON-PROMOTABLE — real coder bytes over the cached n600 GT partition; NOT a byte-closed
`evaluate.py` row. score_claim=false.

WHY THIS EXISTS.  The registered band law
``ddm_pp1_correction_stream_position_band_v1`` prices a correction stream's POSITION
cost and its ``domain_of_validity["excluded"]`` states verbatim: *"the LABEL
(which-class) cost of a correction stream (this law is POSITION only)."*  The producer
``experiments/ddm_pp1_band_lemma_curve.py:79`` zeroes the class maps
(``cmaps = [np.zeros(...)]  # positions only; class stream trivial``) so the #307
contour coder's ``cls`` stream is excluded by construction.  The label half has
therefore never been measured, while every downstream verdict compares a
position-only price against the 1.2731 B/flip water level as if it were the whole cost.

WHAT THIS MEASURES.  The SAME supports (``margins < tau``), the SAME coders, the SAME
densities as the pp1 R2 receipt — but with the REAL GT class map instead of zeros, so
``stream_bytes`` splits into POSITION (counts+anchor+chain) and LABEL (cls).  Two label
coder families, mirroring pp1's two position families:

  * ``cls`` (#307 adaptive range coder, ctx = class of the walk's current pixel) — the
    boundary-coherent / neighbour-conditioned family.  This IS the direct test of the
    hypothesis "the label stream is largely predictable from neighbouring labels".
  * ``lzma_labels`` (labels at support, raster order, packed 1 byte/flip -> LZMA1-x9e) —
    the GENERIC CONTROL.  Per the generic-triple discipline a generic coder is a control,
    never an answer; the coherent family must BEAT it or the coherence claim is unearned.

CONTROLS (P4: no meter without a canary).  Three, all enforced in code:
  * POSITIVE CONTROL A (within-run) — re-encode with zeroed and with i.i.d. random class
    maps; the position streams must be byte-identical under all three class maps.
  * POSITIVE CONTROL B (cross-instrument) — load the pp1 R2 receipt and require this
    tool's measured contour position B/err to reproduce pp1's ``b_per_err_best`` at every
    tau where pp1's best coder WAS the contour walk.  (At tau >= 0.4 pp1's best is LZMA,
    which this tool does not compute, so those taus are compared as ">= pp1" — this tool
    over-prices position there.)  If position does not reproduce, the instrument is not
    the pp1 instrument and no label reading counts.
  * NEGATIVE CONTROL — re-encode with i.i.d. uniform random labels on the same support.
    A label coder that codes RANDOM labels much below log2(5)/8 = 0.2902410 B/flip is
    cheating (leaking position information into the class stream).

ROUND-TRIP SCOPE (stated because it is narrower than the sweep): bit-exact decode of BOTH
flip maps and class maps is verified on a small LEADING SUBSET only, not on the n600 rows.
The n600 byte counts come from the same encoder whose streams that subset proves decodable.

HONEST BOUND.  The class map coded here is the GT class alone.  A real receiver ALSO
knows the base partition's (wrong) class at each corrected site and can exclude it, so
every label price reported here is a conservative UPPER BOUND on the deployable label
cost.  If position+label stays under water at this upper bound, the conclusion is
robust to that slack.

Usage:
  PYTHONPATH=src:tools .venv/bin/python experiments/ddm_dc1_label_stream_price.py \\
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \\
      --taus 0.1,0.2 --n 600 --out <path>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import lzma
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WATER_B_PER_FLIP = 1.2731  # registered region_merge water level
N_CLASSES = 5
BLIND_LABEL_BOUND = math.log2(N_CLASSES) / 8.0  # 0.2902410 B/flip, i.i.d. uniform 5-ary
PP1_RECEIPT = _REPO / ".omx/research/ddm_pp1_band_lemma_receipt_20260728.json"
#: taus where pp1's ``b_per_err_best`` was the CONTOUR coder (>=0.4 pp1's best is LZMA,
#: which this tool does not compute, so those are a one-sided ">= pp1" check only).
PP1_CONTOUR_BEST_TAUS = (0.008, 0.02, 0.05, 0.1, 0.2)


def _lzma_raw(data: bytes) -> bytes:
    return lzma.compress(
        data,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
    )


def _load_m307():
    spec = importlib.util.spec_from_file_location(
        "m307c", str(_REPO / "tools" / "measure_contour_string_flip_coding.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["m307c"] = m
    spec.loader.exec_module(m)
    return m


def pp1_cross_check(rows: list[dict], tol: float = 1e-3) -> dict:
    """POSITIVE CONTROL B — reproduce the registered pp1 R2 position curve.

    At the taus where pp1's ``b_per_err_best`` was the contour coder the two must agree to
    ``tol``; at the taus where pp1's best was LZMA (which this tool does not compute) the
    only admissible claim is that this tool is NOT CHEAPER, i.e. it over-prices position.
    A FAIL here means the instrument is not the pp1 instrument and no label reading counts.
    """
    if not PP1_RECEIPT.exists():
        return {"available": False, "reason": f"{PP1_RECEIPT} not found"}
    pp1 = json.loads(PP1_RECEIPT.read_text())
    by_tau = {float(p["tau"]): p for p in pp1["coherent_curve"]}
    checks = []
    for r in rows:
        ref = by_tau.get(float(r["tau"]))
        if ref is None or not r.get("n_err"):
            continue
        mine = r["b_per_err_position_contour"]
        theirs = float(ref["b_per_err_best"])
        contour_was_best = any(abs(float(r["tau"]) - t) < 1e-9 for t in PP1_CONTOUR_BEST_TAUS)
        ok = abs(mine - theirs) <= tol if contour_was_best else (mine >= theirs - tol)
        checks.append({
            "tau": r["tau"], "dc1_position": mine, "pp1_b_per_err_best": theirs,
            "abs_diff": abs(mine - theirs),
            "pp1_best_was_contour": contour_was_best,
            "mode": "equality" if contour_was_best else "one_sided_not_cheaper",
            "pass": bool(ok),
        })
    return {
        "available": True,
        "tolerance": tol,
        "all_pass": bool(checks) and all(c["pass"] for c in checks),
        "n_equality_checks": sum(1 for c in checks if c["pp1_best_was_contour"]),
        "checks": checks,
    }


def _split_streams(enc: dict) -> tuple[int, int]:
    """(position_bytes, label_bytes) from a #307 encode result.

    POSITION = counts + anchor + chain (exactly pp1's ``contour_pos_bytes``).
    LABEL    = cls.
    """
    sb = enc["stream_bytes"]
    pos = int(sb["counts"] + sb["anchor"] + sb["chain"])
    return pos, int(sb["cls"])


def price_support(
    support: np.ndarray,
    lstars: np.ndarray,
    m307,
    *,
    rng: np.random.Generator,
    verify_roundtrip: bool,
) -> dict:
    """Position + label price for one support, with both controls.

    support: (N,H,W) bool.  lstars: (N,H,W) int (GT argmax classes).
    """
    n_flips = int(support.sum())
    if n_flips == 0:
        return {"n_err": 0}
    n_frames, h, w = support.shape
    fmaps = [np.ascontiguousarray(support[f]) for f in range(n_frames)]

    # --- REAL: GT class map -------------------------------------------------
    cmaps_real = [np.ascontiguousarray(lstars[f].astype(np.int64)) for f in range(n_frames)]
    t0 = time.time()
    enc_real = m307.contour_encode_frames(fmaps, cmaps_real)
    pos_b, lab_b = _split_streams(enc_real)
    enc_s = time.time() - t0

    # --- POSITIVE CONTROL: zeroed class map (pp1's exact call) --------------
    cmaps_zero = [np.zeros((h, w), np.int64) for _ in range(n_frames)]
    enc_zero = m307.contour_encode_frames(fmaps, cmaps_zero)
    pos_b_zero, lab_b_zero = _split_streams(enc_zero)

    # --- NEGATIVE CONTROL: i.i.d. uniform random labels on the support ------
    cmaps_rand = []
    for f in range(n_frames):
        cm = np.zeros((h, w), np.int64)
        idx = np.flatnonzero(fmaps[f].reshape(-1))
        if idx.size:
            cm.reshape(-1)[idx] = rng.integers(0, N_CLASSES, size=idx.size)
        cmaps_rand.append(cm)
    enc_rand = m307.contour_encode_frames(fmaps, cmaps_rand)
    pos_b_rand, lab_b_rand = _split_streams(enc_rand)

    # --- GENERIC CONTROL label coder: labels at support, raster order -------
    lab_bytes_raw = lstars[support].astype(np.uint8).tobytes()
    lzma_lab = len(_lzma_raw(lab_bytes_raw))

    out = {
        "n_err": n_flips,
        "n_components": int(enc_real["n_components"]),
        "singleton_flip_frac": float(enc_real["singleton_flip_frac"]),
        "coherent_ge4_flip_frac": float(enc_real["coherent_ge4_flip_frac"]),
        # position (must equal pp1)
        "contour_pos_bytes": pos_b,
        "b_per_err_position_contour": pos_b / n_flips,
        # label, coherent family
        "contour_label_bytes": lab_b,
        "b_per_err_label_contour": lab_b / n_flips,
        # label, generic control
        "lzma_label_bytes": lzma_lab,
        "b_per_err_label_lzma": lzma_lab / n_flips,
        # controls
        "ctrl_pos_zero_class_bytes": pos_b_zero,
        "ctrl_pos_zero_class_b_per_err": pos_b_zero / n_flips,
        "ctrl_zero_class_label_bytes": lab_b_zero,
        "ctrl_pos_random_class_bytes": pos_b_rand,
        "ctrl_random_label_bytes": lab_b_rand,
        "ctrl_random_label_b_per_err": lab_b_rand / n_flips,
        "ctrl_blind_label_bound_b_per_err": BLIND_LABEL_BOUND,
        "encode_s": round(enc_s, 2),
    }
    # positive control verdict: position must be invariant to the class map content
    out["positive_control_position_invariant"] = bool(pos_b == pos_b_zero == pos_b_rand)
    # negative control verdict: random labels must cost ~the blind bound (not far below)
    out["negative_control_random_near_blind"] = bool(
        out["ctrl_random_label_b_per_err"] >= 0.90 * BLIND_LABEL_BOUND
    )
    # best label coder (coherent must beat generic to earn the coherence claim)
    out["b_per_err_label_best"] = min(
        out["b_per_err_label_contour"], out["b_per_err_label_lzma"]
    )
    out["label_coherent_beats_generic"] = bool(
        out["b_per_err_label_contour"] < out["b_per_err_label_lzma"]
    )
    out["b_per_err_total_best"] = out["b_per_err_position_contour"] + out["b_per_err_label_best"]
    out["total_vs_water"] = out["b_per_err_total_best"] / WATER_B_PER_FLIP

    if verify_roundtrip:
        t1 = time.time()
        dec_f, dec_c = m307.contour_decode_frames(enc_real["streams"], n_frames, h, w)
        ok_f = all(np.array_equal(dec_f[i], fmaps[i]) for i in range(n_frames))
        ok_c = all(
            np.array_equal(dec_c[i][fmaps[i]], cmaps_real[i][fmaps[i]])
            for i in range(n_frames)
        )
        out["roundtrip_flips_exact"] = bool(ok_f)
        out["roundtrip_labels_exact"] = bool(ok_c)
        out["roundtrip_s"] = round(time.time() - t1, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--taus", default="0.1,0.2")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--roundtrip-frames", type=int, default=8,
                    help="verify bit-exact decode on this many leading frames (0 = skip)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    t0 = time.time()
    z = np.load(args.gt_cache, mmap_mode="r")
    margins = np.asarray(z["margins"][: args.n], dtype=np.float32)
    lstars = np.asarray(z["lstars"][: args.n], dtype=np.int64)
    n_frames, h, w = margins.shape
    m307 = _load_m307()
    rng = np.random.default_rng(0)
    print(f"[dc1] margins {margins.shape} lstars {lstars.shape} ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    for tau in taus:
        sup = margins < tau
        dens = float(sup.mean())
        pr = price_support(sup, lstars, m307, rng=rng, verify_roundtrip=False)
        pr["tau"] = tau
        pr["density"] = dens
        pr["uniform_position_bound_b_per_err"] = float(np.log2(1.0 / dens) / 8.0)
        rows.append(pr)
        print(
            f"   tau<{tau:5.3f} rho={dens:.6f} n={pr['n_err']:>9d} "
            f"POS={pr['b_per_err_position_contour']:.4f} "
            f"LAB_coh={pr['b_per_err_label_contour']:.4f} "
            f"LAB_lzma={pr['b_per_err_label_lzma']:.4f} "
            f"TOT={pr['b_per_err_total_best']:.4f} "
            f"({pr['total_vs_water']:.4f}x water) "
            f"posctrl={pr['positive_control_position_invariant']} "
            f"negctrl={pr['negative_control_random_near_blind']} "
            f"[{pr['encode_s']}s]",
            flush=True,
        )

    # bit-exact round-trip receipt on a small leading subset (NO-FAKE byte accounting)
    rt = None
    if args.roundtrip_frames > 0 and taus:
        k = min(args.roundtrip_frames, n_frames)
        sup_rt = margins[:k] < taus[0]
        rt = price_support(sup_rt, lstars[:k], m307, rng=rng, verify_roundtrip=True)
        rt["tau"] = taus[0]
        rt["n_frames"] = k
        print(
            f"   [roundtrip n={k}] flips_exact={rt.get('roundtrip_flips_exact')} "
            f"labels_exact={rt.get('roundtrip_labels_exact')}",
            flush=True,
        )

    xcheck = pp1_cross_check(rows)
    print(f"   [pp1 cross-check] all_pass={xcheck.get('all_pass')} "
          f"equality_checks={xcheck.get('n_equality_checks')}", flush=True)

    receipt = {
        "schema": "ddm_dc1_label_stream_price.v1",
        "evidence_axis": (
            "[macOS-CPU advisory] NON-PROMOTABLE — real coder bytes over the cached n600 GT "
            "partition, from an encoder whose streams are proven bit-exactly decodable on a "
            "LEADING SUBSET (see roundtrip_receipt); the n600 rows themselves are byte counts, "
            "not per-row round-trip proofs. NOT a byte-closed evaluate.py row. "
            "pointer 0.1910828242 UNMOVED."
        ),
        "pp1_cross_check": xcheck,
        "score_claim": False,
        "promotion_eligible": False,
        "generated_by": "experiments/ddm_dc1_label_stream_price.py",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "gt_cache": str(args.gt_cache),
        "n_frames": n_frames,
        "water_B_per_flip": WATER_B_PER_FLIP,
        "blind_label_bound_b_per_err": BLIND_LABEL_BOUND,
        "label_price_is_upper_bound_note": (
            "class map coded is the GT class alone; a real receiver also knows the base "
            "partition's wrong class at each site and can exclude it -> deployable label cost "
            "is <= these numbers"
        ),
        "rows": rows,
        "roundtrip_receipt": rt,
        "wall_seconds": round(time.time() - t0, 2),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(f"[dc1] wrote {outp} ({receipt['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
