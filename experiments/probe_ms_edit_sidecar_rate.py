#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""$0 PROBE: Morse-Smale / DMTz error-bounded EDIT-SIDECAR rate for restoring the SegNet separatrix.

Design memo: ``.omx/research/dmtz_taskaware_rate_lever_design_20260709.md``.

THESIS (both papers -> one lever). Our d_seg lives on the SegNet argmax SEPARATRIX (the class
boundary; ~97% of d_seg sits in a ~4.7%-area annulus of small-margin pixels). The witness is a
Morse-Smale complex generated FREE (rule-118) from the counted INR weights+code. DMTz (arXiv
2409.17346) stores error-bounded EDITS that restore a scalar field's Morse-Smale complex; mapped
here, that is a COUNTED sidecar coding ONLY the argmax-restoring DIFF between the free witness
partition and GT -- not the whole partition (which ``tac.boundary_math.dense_raster_lzma_baseline`` already
codes). The task-aware principle (arXiv 2404.04848) says: spend bits only on the flip-prone
boundary; the GoP principle says: amortize the diff temporally across pairs.

WHAT THIS PROBE MEASURES (the load-bearing quantity is BYTES PER CORRECTED FLIP):
  The whole lever is a rate-distortion trade. Storing a flip-restoring edit removes
  ``100/N_total`` score points of d_seg and adds ``bytes*25/37_545_489`` score points of rate.
  Break-even is bytes/flip == 1.2731 (DERIVED in the memo; == the published ~1.27 B/flip Lever-D
  floor -- not a coincidence, it IS the RD break-even). So the edit-sidecar is a GO iff it codes
  the argmax-restoring diff at < ~1.27 bytes per corrected flip; otherwise it is DOMINATED.

The codability of the diff is a property of the SegNet-argmax BOUNDARY GEOMETRY (thin jitter
annulus vs thick 2-D blobs), which is fully available in the GT cache (``lstars`` = GT argmax,
``margins`` = GT margin field). This probe therefore DERIVES a principled flip set from the
small-|margin| annulus (our established flip locus), calibrated to a target operating d_seg, and
measures bytes/flip for four honest coders:
  (F) sparse-residual FLOOR   -- log2 C(N,k) position + k*log2(K-1) class (no structure exploited).
  (E) edit-diff LZMA          -- DMTz-style raster edit map (0=no-edit, else target class), LZMA.
                                 Exploits spatial clustering == what dense-raster LZMA does, but on the
                                 DIFF not the whole partition.
  (T) edit-diff LZMA + temporal-delta (GoP proxy) -- consecutive-frame diff of the edit maps.
  (P) full-partition reference -- dense_raster_lzma_baseline.partition_description_bytes(GT argmax) (the cost of
                                 coding the WHOLE boundary; the edit sidecar must beat THIS too or
                                 it is not buying anything the free witness base doesn't already give).

AUTHORITY: ``[macOS-CPU advisory / DERIVED-from-GT-margin] NON-PROMOTABLE``. This is a $0 STRUCTURE
probe, NOT a witness measurement and NOT a score. It measures whether the boundary geometry admits
sub-1.27-B/flip coding IN PRINCIPLE. The real edit set of a SPECIFIC witness (witness_argmax vs GT
through the R operator) is the governed n600 follow-up named in the memo -- it requires the full
byte-close decode and is NOT run here (pointer 0.19110 UNMOVED).

Usage:
    .venv/bin/python experiments/probe_ms_edit_sidecar_rate.py \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
        --max-frames 8 --target-dseg 0.005 [--sweep] [--out reports/...json]
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
from math import lgamma, log2
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BREAK_EVEN_BYTES_PER_FLIP = 1.2731  # DERIVED == canonical #157 KKT water level λ* (not novel).
# The REAL admit bar: only sigma_eff~0.51 of stored edits survive R (uint8/resize/parse-back) to
# change the SCORED argmax (#280 Lever-D economics), so a coder must beat sigma_eff*lambda*:
RECOVERY_ADJUSTED_BAR = 0.65        # ~= 0.51 * 1.2731. The bar a real edit coder must clear.
EXISTING_CONTOUR_STRING_BPF = 0.820  # MEASURED n600 #307 contour-string (the coder to beat; NO-GO).
N_SEG_CLASSES = 5


# ---------------------------------------------------------------------------
# core geometry (pure numpy) -- the load-bearing measurements.
# ---------------------------------------------------------------------------
def separatrix_edge_length(argmax_hw: np.ndarray) -> int:
    """Number of 4-neighbour edges that cross a class boundary in an ``(H,W)`` argmax map.

    This is the discrete length of the Morse-Smale SEPARATRIX of the partition (the codim-1
    class boundary). It is the object contour/edit codecs pay for (O(boundary), not O(area))."""
    a = np.asarray(argmax_hw)
    h = int((a[:, 1:] != a[:, :-1]).sum())
    v = int((a[1:, :] != a[:-1, :]).sum())
    return h + v


def flip_mask_at_density(margin_hw: np.ndarray, density: float) -> np.ndarray:
    """DERIVED flip set = the ``density`` fraction of pixels with the SMALLEST |margin|.

    Our established finding: the flip-prone pixels are the small-margin annulus (~97% of d_seg in
    a ~4.7%-area band). Selecting the k = round(density*N) smallest-|margin| pixels is the
    principled proxy for the set a witness of that operating d_seg fails on. Labeled DERIVED (not a
    specific witness's exact flips) -- it measures the STRUCTURE that decides codability."""
    m = np.abs(np.asarray(margin_hw, dtype=np.float64)).ravel()
    N = m.size
    k = round(float(density) * N)
    k = max(0, min(N, k))
    mask = np.zeros(N, dtype=bool)
    if k > 0:
        # k smallest |margin| via argpartition (O(N), memory-cheap).
        idx = np.argpartition(m, k - 1)[:k]
        mask[idx] = True
    return mask.reshape(margin_hw.shape)


def sparse_residual_floor_bits(k: int, N: int, n_classes: int = N_SEG_CLASSES) -> float:
    """Info floor for a POSITION-AGNOSTIC flip residual: choose which k of N pixels flip
    (log2 C(N,k)) + name each flip's target class (k*log2(K-1)). No spatial structure exploited --
    the baseline any structured coder must beat. (== the sparse ``k`` colex-rank + class stream.)"""
    if k <= 0:
        return 0.0
    logC = (lgamma(N + 1) - lgamma(k + 1) - lgamma(N - k + 1)) / np.log(2.0)
    return float(logC + k * log2(max(1, n_classes - 1)))


def edit_diff_map(gt_argmax_hw: np.ndarray, flip_mask_hw: np.ndarray) -> np.ndarray:
    """DMTz-style raster EDIT map: 0 = no edit (free witness base is right), else target class+1.

    This is the counted sidecar the decoder would OVERLAY on the free witness argmax to restore the
    separatrix. Interior/agreeing pixels are the 0 sentinel -> LZMA codes only the (clustered) edit
    boundary structure. Exactly-reversible by construction (overlay where != 0)."""
    diff = np.zeros(gt_argmax_hw.shape, dtype=np.uint8)
    sel = np.asarray(flip_mask_hw, dtype=bool)
    diff[sel] = (np.asarray(gt_argmax_hw, dtype=np.uint8)[sel] + 1)
    return diff


def _lzma_bytes(arr: np.ndarray) -> int:
    """Reversible LZMA length of a uint8 array (the honest, decodeable byte cost -- matches the
    Dense-raster LZMA discipline: LZMA over a label/edit map exploits constant runs.
    entropy). preset 9|EXTREME to match the byte-optimal deploy path."""
    return len(lzma.compress(np.ascontiguousarray(arr, dtype=np.uint8).tobytes(),
                             preset=9 | lzma.PRESET_EXTREME))


# ---------------------------------------------------------------------------
# per-frame + temporal measurement.
# ---------------------------------------------------------------------------
def measure_frame(gt_argmax_hw: np.ndarray, margin_hw: np.ndarray, density: float) -> dict[str, Any]:
    """All per-frame coders at one operating density. Returns bytes + bytes/flip for F/E and the
    geometry (flip count, edit-map boundary length, area/perimeter ratio)."""
    N = int(gt_argmax_hw.size)
    fmask = flip_mask_at_density(margin_hw, density)
    k = int(fmask.sum())
    diff = edit_diff_map(gt_argmax_hw, fmask)
    # boundary (perimeter) of the EDIT set: 4-neighbour edges of the binary flip mask.
    bnd = separatrix_edge_length(fmask.astype(np.uint8))
    floor_bits = sparse_residual_floor_bits(k, N)
    edit_bytes = _lzma_bytes(diff)
    return {
        "N": N,
        "k_flips": k,
        "dseg_of_frame": float(k) / float(N),
        "edit_boundary_len": bnd,
        "area_per_perimeter": (float(k) / float(bnd)) if bnd > 0 else float("inf"),
        "floor_bits": floor_bits,
        "floor_bytes_per_flip": (floor_bits / 8.0 / k) if k > 0 else float("nan"),
        "edit_lzma_bytes": edit_bytes,
        "edit_bytes_per_flip": (edit_bytes / k) if k > 0 else float("nan"),
        "_diff": diff,  # kept for temporal-delta aggregation; stripped before JSON.
    }


def temporal_delta_bytes(diff_maps: list[np.ndarray]) -> int:
    """GoP proxy: code frame 0 raw, then consecutive int16 deltas (mod 256 uint8) of the edit maps.
    If the boundary is temporally coherent (adjacent pairs share the flip locus), the deltas are
    sparser than the raw maps -> fewer bytes. Reversible (prefix-sum decode)."""
    if not diff_maps:
        return 0
    total = _lzma_bytes(diff_maps[0])
    prev = diff_maps[0].astype(np.int16)
    for d in diff_maps[1:]:
        cur = d.astype(np.int16)
        delta = ((cur - prev) % 256).astype(np.uint8)
        total += _lzma_bytes(delta)
        prev = cur
    return total


def full_partition_reference_bytes(gt_argmax_stack: np.ndarray) -> int | None:
    """Reference: dense-raster LZMA cost of coding the WHOLE partition for the same frames.
    The edit sidecar must beat this (else the free witness base bought nothing). Fail-open (None)
    if the codec import is unavailable."""
    try:
        from tac.boundary_math.dense_raster_lzma_baseline import partition_description_bytes
    except Exception:
        return None
    total = 0
    for a in gt_argmax_stack:
        try:
            total += int(partition_description_bytes(np.asarray(a, dtype=np.int64)))
        except Exception:
            return None
    return total


# ---------------------------------------------------------------------------
# driver.
# ---------------------------------------------------------------------------
def run_probe(gt_cache: Path, max_frames: int, target_dseg: float,
              densities: list[float]) -> dict[str, Any]:
    z = np.load(gt_cache, allow_pickle=False)
    for req in ("lstars", "margins"):
        if req not in z.files:
            raise ValueError(f"gt cache {gt_cache} lacks {req!r} (need GT argmax + margin field).")
    lstars = np.asarray(z["lstars"])      # (P, H, W) GT SegNet argmax (frame1)
    margins = np.asarray(z["margins"], dtype=np.float64)  # (P, H, W) GT margin field
    P = int(min(max_frames, lstars.shape[0]))
    lstars = lstars[:P]
    margins = margins[:P]
    H, W = lstars.shape[1], lstars.shape[2]

    results: dict[str, Any] = {
        "gt_cache": str(gt_cache),
        "n_frames": P,
        "render_hw": [int(H), int(W)],
        "break_even_bytes_per_flip": BREAK_EVEN_BYTES_PER_FLIP,
        "recovery_adjusted_bar_bytes_per_flip": RECOVERY_ADJUSTED_BAR,
        "existing_contour_string_bytes_per_flip": EXISTING_CONTOUR_STRING_BPF,
        "target_dseg": target_dseg,
        "authority": "[macOS-CPU advisory / DERIVED-from-GT-margin] NON-PROMOTABLE",
        "per_density": [],
    }
    dens_list = sorted({*densities, target_dseg})
    for dens in dens_list:
        frames = [measure_frame(lstars[i], margins[i], dens) for i in range(P)]
        diff_maps = [f.pop("_diff") for f in frames]
        tot_k = sum(f["k_flips"] for f in frames)
        tot_floor_bits = sum(f["floor_bits"] for f in frames)
        tot_edit_bytes = sum(f["edit_lzma_bytes"] for f in frames)
        tot_temporal = temporal_delta_bytes(diff_maps)
        mean_apr = float(np.mean([f["area_per_perimeter"] for f in frames]))
        row = {
            "density": dens,
            "total_flips": tot_k,
            "floor_bytes_per_flip": (tot_floor_bits / 8.0 / tot_k) if tot_k else float("nan"),
            "edit_lzma_bytes_per_flip": (tot_edit_bytes / tot_k) if tot_k else float("nan"),
            "edit_temporal_bytes_per_flip": (tot_temporal / tot_k) if tot_k else float("nan"),
            "mean_area_per_perimeter": mean_apr,
            "verdict_vs_breakeven": None,  # filled below
        }
        best = min(v for v in (row["floor_bytes_per_flip"], row["edit_lzma_bytes_per_flip"],
                               row["edit_temporal_bytes_per_flip"]) if v == v)
        row["best_bytes_per_flip"] = best
        # Verdict against the REAL recovery-adjusted admit bar (0.65), not the rate-only break-even.
        # Must also beat the existing #307 contour-string (0.820) to be worth anything.
        row["verdict_vs_breakeven"] = (
            "GO" if (best < RECOVERY_ADJUSTED_BAR and best < EXISTING_CONTOUR_STRING_BPF)
            else "DOMINATED")
        results["per_density"].append(row)

    # full-partition reference at target density frames (geometry-only; density-independent).
    results["full_partition_ref_bytes"] = full_partition_reference_bytes(lstars)
    if results["full_partition_ref_bytes"] is not None:
        # per-frame full-boundary cost, for the "diff must beat whole partition" comparison.
        results["full_partition_ref_bytes_per_frame"] = (
            results["full_partition_ref_bytes"] / P)
    return results


def _print_report(r: dict[str, Any]) -> None:
    print(f"\n=== MS/DMTz edit-sidecar rate probe  [{r['authority']}] ===")
    print(f"gt_cache={r['gt_cache']}  n_frames={r['n_frames']}  render={r['render_hw']}")
    print(f"RD break-even (rate-only, == #157 KKT lambda*) = {r['break_even_bytes_per_flip']:.4f} B/flip")
    print(f"REAL admit bar (recovery-adjusted) = {r['recovery_adjusted_bar_bytes_per_flip']:.3f} B/flip"
          f"; must also beat existing #307 = {r['existing_contour_string_bytes_per_flip']:.3f} B/flip")
    fp = r.get("full_partition_ref_bytes_per_frame")
    if fp is not None:
        print(f"full-partition (dense-raster LZMA) reference = {fp:.0f} bytes/frame "
              f"(the whole-boundary cost the free witness base already avoids)")
    print(f"{'density':>9} {'flips':>8} {'floor B/flip':>13} {'edit B/flip':>12} "
          f"{'edit+GoP B/flip':>16} {'area/perim':>11} {'verdict':>10}")
    for row in r["per_density"]:
        print(f"{row['density']:>9.4f} {row['total_flips']:>8d} "
              f"{row['floor_bytes_per_flip']:>13.3f} {row['edit_lzma_bytes_per_flip']:>12.3f} "
              f"{row['edit_temporal_bytes_per_flip']:>16.3f} "
              f"{row['mean_area_per_perimeter']:>11.3f} {row['verdict_vs_breakeven']:>10}")
    tgt = next((x for x in r["per_density"] if abs(x["density"] - r["target_dseg"]) < 1e-9), None)
    if tgt:
        print(f"\nAT TARGET d_seg={r['target_dseg']}: best coder = {tgt['best_bytes_per_flip']:.3f} "
              f"B/flip -> {tgt['verdict_vs_breakeven']} "
              f"(break-even {r['break_even_bytes_per_flip']:.3f})")


def _selftest() -> None:
    """Cheap correctness checks on the core (no GT needed): reversibility of the edit overlay, the
    sparse floor monotonicity, and the separatrix-length of a known partition."""
    rng = np.random.default_rng(0)
    gt = rng.integers(0, N_SEG_CLASSES, size=(32, 40)).astype(np.int64)
    margin = rng.standard_normal((32, 40))
    mask = flip_mask_at_density(margin, 0.1)
    assert abs(int(mask.sum()) - round(0.1 * gt.size)) <= 1, "density selection off"
    diff = edit_diff_map(gt, mask)
    # overlay reversibility: applying diff onto a WRONG base restores GT exactly on the flip set.
    base = np.zeros_like(gt)
    restored = np.where(diff != 0, (diff.astype(np.int64) - 1), base)
    assert np.array_equal(restored[mask], gt[mask]), "edit overlay not reversible on flip set"
    # a 2-region split has separatrix == the split length.
    part = np.zeros((10, 10), dtype=np.int64)
    part[:, 5:] = 1
    assert separatrix_edge_length(part) == 10, "separatrix length wrong"
    # sparse floor grows with k.
    assert sparse_residual_floor_bits(10, 1000) < sparse_residual_floor_bits(50, 1000)
    print("[selftest] OK: overlay reversible, density calibrated, separatrix length, floor monotone")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", type=Path,
                    default=_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n24.npz",
                    help="GT npz with lstars (argmax) + margins (margin field).")
    ap.add_argument("--max-frames", type=int, default=8, help="SMOKE cap (n8 default; NOT n600).")
    ap.add_argument("--target-dseg", type=float, default=0.005,
                    help="operating d_seg to calibrate the derived flip set to.")
    ap.add_argument("--sweep", action="store_true",
                    help="also measure a density sweep (0.001..0.02) to show the trend.")
    ap.add_argument("--out", type=Path, default=None, help="write JSON report here.")
    ap.add_argument("--selftest", action="store_true", help="run core correctness checks + exit.")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0

    if not args.gt_cache.exists():
        print(f"[FATAL] gt cache not found: {args.gt_cache}", file=sys.stderr)
        return 2
    densities = [0.001, 0.0025, 0.005, 0.01, 0.02] if args.sweep else [args.target_dseg]
    r = run_probe(args.gt_cache, args.max_frames, args.target_dseg, densities)
    _print_report(r)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"\n[json] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
