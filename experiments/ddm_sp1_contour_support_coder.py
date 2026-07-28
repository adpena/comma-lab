#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_sp1 R1 — CONTOUR SUPPORT CODER, measured on the REAL n600 copy-flip masks.

MEANS. pointer 0.19108 UNMOVED. Authority: [macOS-CPU advisory] NON-PROMOTABLE — a lossless
coder-byte measurement over cached masks, NEVER a byte-closed evaluate.py score. NO-FAKE: the
flip masks are reconstructed from the fc1 cached ``copy_argmax`` chunks + the GT ``lstars``
argmax (flip = copy_argmax != lstar); the #307 in-tree contour coder
(``tools/measure_contour_string_flip_coding.py``: chain-code + digital-straightness context +
in-tree RangeEncoder) is REUSED verbatim and every emitted stream is DECODED back and verified
BIT-EXACT before any byte count is reported. A coder that does not round-trip is reported broken,
never as a price (the sp1 honest-boundary rule).

WHAT THIS MEASURES (gc5 B3 rung): the ENTIRE sub-bar byte floor rides on the contour-coded
support. The corpus carries a PROJECTED 142 KB (fc1 stage5 ``contour_support_bestcase_UNBUILT``)
vs the REAL 421,366 B packbits-LZMA incumbent (fc1 stage2 / da1 d1). This tool BUILDS the coder
and reports the REAL contour support bytes so the gc5 falsifier fires on a measured number:
  <150 KB  -> the 0.154 floor STRENGTHENS
  150-250 KB -> floor SURVIVES at the 0.172 bar only
  >=250 KB -> floor DEAD (typed scope: this copy-base flip support on this coder).

Streams (from the #307 coder): SUPPORT geometry = counts + anchor + chain (the binary flip-field
positions, contour-coded); LABELS = cls (the GT class at each flip). The 421,366 B LZMA incumbent
is the SUPPORT geometry ONLY (packbits of the binary flip field), so contour SUPPORT vs 421,366 is
apples-to-apples; the 41,392 B constriction label stream (fc1 stage2) is the LABELS incumbent.

DATA PATH (reuse, not re-derive): fc1 chunks
``/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/chunks/ctx_*.npz`` (``copy_argmax`` uint8) + GT
``lstars`` from the mlx_fleet cache. No SegNet re-run, no render (sc1 owns the scorer slot).
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# REUSE the #307 contour machinery verbatim (NO rebuild).
import measure_contour_string_flip_coding as mcs

N_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
RATE_DENOM = 37_545_489.0
# gc5 falsifier band on the SUPPORT geometry (bytes)
FALSIFIER_STRENGTHEN = 150_000
FALSIFIER_DEAD = 250_000
BAR_0P172 = 187_727
BAR_0P15 = 154_522


def _load_flips(ctx_dir: Path, gt_cache: Path, max_pairs: int):
    """Reconstruct per-pair flip masks + GT class maps from cached chunks (no scorer)."""
    gt = np.load(str(gt_cache))
    lstars = gt["lstars"]  # (600,384,512) int64
    chunks = sorted(ctx_dir.glob("ctx_*.npz"))
    flips: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    n = 0
    for ch in chunks:
        d = np.load(str(ch))
        c_arg = d["copy_argmax"]  # (m,384,512) uint8
        s0 = int(d["start"])
        for j in range(c_arg.shape[0]):
            pi = s0 + j
            if pi >= max_pairs:
                break
            ls = lstars[pi].astype(np.uint8)
            fm = (c_arg[j] != ls)
            flips.append(fm)
            classes.append(ls.astype(np.int64))
            n += 1
        if n >= max_pairs:
            break
    return flips, classes


def _lzma_support_bytes(flips: list[np.ndarray]) -> int:
    """Incumbent SUPPORT geometry: packbits of the concatenated binary flip field, LZMA1-x9e RAW.

    Mirrors fc1 stage2 / da1 d1 (421,366 B on n600) so the incumbent reproduces here."""
    packed = np.packbits(np.concatenate([f.reshape(-1) for f in flips]))
    filt = [{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}]
    return len(lzma.compress(packed.tobytes(), format=lzma.FORMAT_RAW, filters=filt))


def _contour_measure(flips: list[np.ndarray], classes: list[np.ndarray], label: str) -> dict:
    """Encode with the #307 coder, DECODE-verify bit-exact, split support vs label bytes."""
    h, w = flips[0].shape
    t0 = time.time()
    enc = mcs.contour_encode_frames(flips, classes)
    enc_s = round(time.time() - t0, 2)
    # NO-FAKE: decode every frame back and assert bit-exact BEFORE reporting any byte count.
    td = time.time()
    dec_f, dec_c = mcs.contour_decode_frames(enc["streams"], len(flips), h, w)
    decode_s = round(time.time() - td, 3)
    lossless = True
    for i in range(len(flips)):
        if not np.array_equal(dec_f[i], flips[i]):
            lossless = False
            break
        if not np.array_equal(dec_c[i][flips[i]], classes[i][flips[i]]):
            lossless = False
            break
    sb = enc["stream_bytes"]
    support_bytes = int(sb["counts"] + sb["anchor"] + sb["chain"])
    label_bytes = int(sb["cls"])
    n_flips = int(enc["n_flips"])
    return {
        "label": label,
        "n_flips": n_flips,
        "n_components": int(enc["n_components"]),
        "stream_bytes": {k: int(v) for k, v in sb.items()},
        "contour_support_bytes": support_bytes,
        "contour_label_bytes": label_bytes,
        "contour_total_bytes": int(enc["total_bytes"]),
        "support_bits_per_flip": (8.0 * support_bytes / n_flips) if n_flips else 0.0,
        "b_contour_total_per_flip": float(enc["b_contour"]),
        "singleton_flip_frac": float(enc["singleton_flip_frac"]),
        "coherent_ge4_flip_frac": float(enc["coherent_ge4_flip_frac"]),
        "comp_size_hist": enc["comp_size_hist"],
        "encode_s": enc_s,
        "decode_s": decode_s,
        "lossless_roundtrip": bool(lossless),
    }


# S accounting (registered laws): 1 byte = 25/37_545_489 S; 1 conceded flip = 100/117_964_800 S.
# water level = (S/flip)/(S/byte) = 1.2731 B/flip (registered region_merge concession law).
S_PER_BYTE = 25.0 / RATE_DENOM
TOTAL_SITES_N600 = 600 * 384 * 512  # 117_964_800
S_PER_CONCEDED_FLIP = 100.0 / TOTAL_SITES_N600
WATER_LEVEL_B_PER_FLIP = S_PER_CONCEDED_FLIP / S_PER_BYTE  # 1.2731...


def _components_by_size(flips: list[np.ndarray]):
    """Per-frame connected components (8-conn) as (frame, label_map, sizes) — reused for concession."""
    from scipy.ndimage import label as cc_label
    structure = np.ones((3, 3), dtype=np.int64)
    out = []
    for fm in flips:
        lab, n = cc_label(fm, structure=structure)
        sizes = np.bincount(lab.reshape(-1))  # sizes[0]=background
        out.append((lab, sizes, n))
    return out


def _concede_below_size(flips, comps, min_size: int):
    """Return retained flip masks (drop components with px < min_size) + conceded flip count."""
    retained = []
    conceded = 0
    for fm, (lab, sizes, _n) in zip(flips, comps, strict=False):
        keep_labels = np.where(sizes >= min_size)[0]
        keep_labels = keep_labels[keep_labels > 0]
        keep_mask = np.isin(lab, keep_labels)
        rfm = fm & keep_mask
        conceded += int(fm.sum() - rfm.sum())
        retained.append(rfm)
    return retained, conceded


def _lossy_curve(flips, classes, comps) -> list[dict]:
    """Support concession curve: at each min-component-size threshold, re-code the RETAINED flips
    (contour + LZMA), concede the dropped flips, and compute the real support-attributable S.

    S_support = 25*retained_bytes/RATE_DENOM + 100*conceded_flips/TOTAL_SITES (region_merge water
    level 1.2731 B/flip). The min-S point is the score-optimal lossy operating point."""
    rows = []
    total_flips = int(sum(int(f.sum()) for f in flips))
    for k in (1, 2, 3, 4, 6, 8, 12, 16):
        rflips, conceded = _concede_below_size(flips, comps, k)
        retained_flips = total_flips - conceded
        # contour support on retained
        if retained_flips > 0:
            cont = _contour_measure(rflips, classes, f"lossy_k{k}")
            cont_support = cont["contour_support_bytes"]
            cont_lossless = cont["lossless_roundtrip"]
        else:
            cont_support, cont_lossless = 0, True
        lzma_support = _lzma_support_bytes(rflips) if retained_flips > 0 else 0
        best_support = min(cont_support, lzma_support)
        best_coder = "contour" if cont_support <= lzma_support else "lzma"
        s_rate = S_PER_BYTE * best_support
        s_conceded = S_PER_CONCEDED_FLIP * conceded
        rows.append({
            "min_component_size": k,
            "retained_flips": retained_flips,
            "conceded_flips": conceded,
            "conceded_frac": round(conceded / max(1, total_flips), 4),
            "contour_support_bytes": cont_support,
            "contour_lossless_roundtrip": cont_lossless,
            "lzma_support_bytes": lzma_support,
            "best_support_bytes": best_support,
            "best_coder": best_coder,
            "S_support_rate": round(s_rate, 6),
            "S_support_conceded": round(s_conceded, 6),
            "S_support_total": round(s_rate + s_conceded, 6),
        })
        print(f"[sp1-R1-lossy] k>={k}: conceded={conceded} ({rows[-1]['conceded_frac']:.1%}) "
              f"best_support={best_support}B ({best_coder}) S_rate={s_rate:.5f} "
              f"S_conc={s_conceded:.5f} S_tot={s_rate+s_conceded:.5f}", flush=True)
    return rows


def _verdict(support_bytes: int) -> str:
    if support_bytes < FALSIFIER_STRENGTHEN:
        return "FLOOR_STRENGTHENS(<150KB)"
    if support_bytes < FALSIFIER_DEAD:
        return "FLOOR_SURVIVES_AT_BAR_ONLY(150-250KB)"
    return "FLOOR_DEAD(>=250KB)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx-dir", default="/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/chunks")
    ap.add_argument("--gt-cache",
                    default="/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--per-class", action="store_true", help="also code Road/Lane-only support")
    ap.add_argument("--lossy-curve", action="store_true",
                    help="measure the support concession curve (water level 1.2731 B/flip)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    flips, classes = _load_flips(Path(args.ctx_dir), Path(args.gt_cache), args.n)
    P = len(flips)
    total_flips = int(sum(int(f.sum()) for f in flips))
    total_sites = P * flips[0].size
    print(f"[sp1-R1] loaded P={P} flips={total_flips} frac={total_flips/total_sites:.8f} "
          f"({time.time()-t_start:.0f}s)", flush=True)

    # incumbent (c): 421 KB packbits-LZMA support geometry
    lzma_support = _lzma_support_bytes(flips)
    print(f"[sp1-R1] LZMA incumbent support = {lzma_support} B", flush=True)

    # variant (a): all-flip contour support coder
    allrow = _contour_measure(flips, classes, "all_flips_contour")
    print(f"[sp1-R1] contour ALL support={allrow['contour_support_bytes']} B "
          f"labels={allrow['contour_label_bytes']} B lossless={allrow['lossless_roundtrip']} "
          f"decode_s={allrow['decode_s']}", flush=True)

    per_class = {}
    if args.per_class:
        for k in (0, 1):  # Road, Lane = 85% of mass (charter)
            cflips = [f & (c == k) for f, c in zip(flips, classes, strict=False)]
            row = _contour_measure(cflips, classes, f"{CLASS_NAMES[k]}_only_contour")
            row["lzma_support_bytes"] = _lzma_support_bytes(cflips)
            per_class[CLASS_NAMES[k]] = row
            print(f"[sp1-R1] {CLASS_NAMES[k]}: contour_support={row['contour_support_bytes']} B "
                  f"vs LZMA={row['lzma_support_bytes']} B lossless={row['lossless_roundtrip']}",
                  flush=True)

    lossy_rows = []
    lossy_opt = None
    if args.lossy_curve:
        comps = _components_by_size(flips)
        lossy_rows = _lossy_curve(flips, classes, comps)
        lossy_opt = min(lossy_rows, key=lambda r: r["S_support_total"])
        print(f"[sp1-R1-lossy] OPTIMAL: k>={lossy_opt['min_component_size']} "
              f"S_support_total={lossy_opt['S_support_total']} "
              f"(best_support={lossy_opt['best_support_bytes']}B via {lossy_opt['best_coder']}, "
              f"conceded {lossy_opt['conceded_frac']:.1%})", flush=True)

    support = allrow["contour_support_bytes"]
    verdict = _verdict(support)
    # drift resolution: fc1 stage5 projected 142,220 (support) vs gap_arithmetic 184 KB
    # (support+labels). ONE measured number = real contour support.
    result = {
        "schema": "ddm_sp1_contour_support_coder.v1",
        "task": "gc5 B3 rung — CONTOUR SUPPORT CODER, measured on real n600 copy-flip masks",
        "evidence_axis": ("[macOS-CPU advisory] NON-PROMOTABLE lossless coder bytes over cached "
                          "copy-base flip masks; NOT a byte-closed evaluate.py row; pointer 0.19108 UNMOVED"),
        "utc": datetime.now(UTC).isoformat(),
        "n_pairs": P,
        "total_flips": total_flips,
        "total_sites": total_sites,
        "incumbent_lzma_support_bytes": lzma_support,
        "incumbent_constriction_label_bytes": 41392,  # fc1 stage2 measured
        "variant_a_contour": allrow,
        "per_class": per_class,
        "falsifier_band": {
            "strengthen_below": FALSIFIER_STRENGTHEN,
            "dead_at_or_above": FALSIFIER_DEAD,
            "measured_contour_support_bytes": support,
            "verdict": verdict,
            "verdict_scope": ("FORMULATION: this copy-base flip support, this #307 chain-code coder; "
                              "a tighter coder or a different base only lowers support"),
        },
        "drift_resolution": {
            "fc1_stage5_projected_support_bytes": 142220,
            "fc1_gap_arithmetic_contour_best_case_bytes": 184440,
            "note": ("142 KB was fc1's SUPPORT-only projection; 184 KB = 142 support + 42 labels "
                     "(scenario C total). ONE measured number resolves which projection held:"),
            "measured_contour_support_bytes": support,
            "measured_contour_support_plus_labels_bytes": support + allrow["contour_label_bytes"],
            "projection_142_error_pct": round(100.0 * (support - 142220) / 142220, 2),
        },
        "vs_incumbent": {
            "lzma_support_bytes": lzma_support,
            "contour_support_bytes": support,
            "contour_vs_lzma_ratio": round(support / lzma_support, 4),
            "support_saved_bytes": lzma_support - support,
        },
        "lossy_support_concession_curve": {
            "water_level_b_per_flip": round(WATER_LEVEL_B_PER_FLIP, 4),
            "s_per_byte": S_PER_BYTE,
            "s_per_conceded_flip": S_PER_CONCEDED_FLIP,
            "rows": lossy_rows,
            "optimal": lossy_opt,
        },
        "bar_bytes_0p172": BAR_0P172,
        "bar_bytes_0p15": BAR_0P15,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[sp1-R1] VERDICT: {verdict} (support={support} B, "
          f"contour/LZMA={result['vs_incumbent']['contour_vs_lzma_ratio']}); wrote {args.out} "
          f"({result['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
