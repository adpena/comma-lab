# SPDX-License-Identifier: MIT
"""ddm_ph1 step 2 --- price the F7 phase carrier with REAL coders (never i.i.d. bounds).

INPUT.  The per-cell offset symbol streams dumped by ``ph1_phase_mass_reach.py``: for each
block size K, an ``(n_pairs, n_cells, 2)`` int8 array of the (dy, dx) each grid cell must be
told in order to realise the measured flip reduction.  This is the ACTUAL payload a phase
receiver would consume, so coding it is a byte-closed price, not an entropy estimate.

WHY NOT AN ENTROPY ESTIMATE.  ``sx1`` quoted 253,341 B from an order-0 entropy estimate; the
real coder returned 410,584 B --- 1.62x worse.  Order-0 entropy is reported here ONLY as a
labelled reference line, never as a price.

THE ADDRESS TAX, MEASURED NOT ASSUMED (operator 2026-08-03).  ``gt2x`` measured that ~78% of
explicit-production bytes are WHERE, not WHAT.  A dense offset grid pays that tax at ZERO by
construction: the cell's position is implicit in raster order, so every coded byte is WHAT.
This module does not assert that --- it measures it, by racing the dense-implicit form
against a sparse-explicit form that must transmit (cell_index, dy, dx) for each non-zero
cell.  If dense wins, the address-solve for this carrier is "already solved by the form",
which is the operator's generative-addressing point made empirically.

XI-TRANSPORT (operator directive #1).  The phase field is ego-motion driven, so pair p's
offsets should be predictable from pair p-1's.  ``temporal`` codes the innovation against the
previous pair's field --- causal and receiver-computable (the receiver has already decoded
p-1).  For a PHASE carrier this is doubly native: the transport operator IS the mechanism.

COERS RACED.  zlib, lzma, brotli, and SMEVR (``experiments.ddm_r7_token_coder``, the shipped
coder that byte-closes the archive --- the matched-bytes currency).  SMEVR caps at 16 levels,
so the (dy, dx) pair is coded as two separate 11-level planes rather than one 121-symbol
alphabet.

AXIS.  ``[macOS-CPU advisory]`` NON-PROMOTABLE.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import lzma
import zlib
from pathlib import Path

import numpy as np

SCHEMA = "ddm_ph1_offset_coder_race.v1"
AXIS_TAG = "[macOS-CPU advisory]"
SCORED_PIXELS = 600 * 512 * 384


def _brotli(payload: bytes) -> int | None:
    try:
        import brotli
    except ImportError:
        return None
    return len(brotli.compress(payload, quality=11))


_SMEVR_FAILURES: list[str] = []


def _smevr(codes_u8: np.ndarray, levels: int) -> int | None:
    """SMEVR bytes via the SHIPPED r7 coder, or None with the reason RECORDED.

    A bare ``except: return None`` here is the silent-instrument bug class (m50): it made a
    *sys.path* import failure read as "SMEVR does not apply to this stream" when in fact
    SMEVR wins this race outright.  Every failure is now appended to ``_SMEVR_FAILURES`` and
    surfaced in the receipt, so a missing SMEVR number can never again be mistaken for a
    measured non-result.
    """
    import sys

    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from experiments.ddm_r7_token_coder import encode_token_codes
    except Exception as exc:
        _SMEVR_FAILURES.append(f"import: {type(exc).__name__}: {exc}")
        return None
    try:
        return len(
            encode_token_codes(
                np.ascontiguousarray(codes_u8, dtype=np.uint8), levels=int(levels), codec="smevr"
            )
        )
    except Exception as exc:
        _SMEVR_FAILURES.append(f"encode shape={codes_u8.shape}: {type(exc).__name__}: {exc}")
        return None


def _order0_bits(sym: np.ndarray) -> float:
    """Labelled REFERENCE ONLY -- an order-0 entropy estimate, never quoted as a price."""
    cnt = np.bincount(sym.ravel())
    cnt = cnt[cnt > 0].astype(np.float64)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum() * sym.size)


def _code_all(payload: bytes, field_phwc: np.ndarray, levels: int) -> dict[str, int | None]:
    """Race every coder on one representation.

    ``field_phwc`` is the SAME data shaped [P,H,W,C], which is the only layout the shipped
    r7 SMEVR coder accepts -- it is a 2-D field coder and flattening destroys the spatial
    context it models.
    """
    return {
        "zlib": len(zlib.compress(payload, 9)),
        "lzma": len(lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)),
        "brotli": _brotli(payload),
        "smevr": _smevr(field_phwc, levels),
    }


def race_block(off: np.ndarray, rmax: int, block: int) -> dict:
    """off: (n_pairs, n_cells, 2) int8 offsets. Returns coder bytes per representation."""
    n_pairs, n_cells, _ = off.shape
    levels = 2 * rmax + 1
    nrow = (384 + block - 1) // block
    ncol = (512 + block - 1) // block
    if nrow * ncol != n_cells:
        raise ValueError(f"block {block}: grid {nrow}x{ncol} != {n_cells} cells")
    dy = (off[..., 0].astype(np.int16) + rmax).astype(np.uint8)  # 0..2*rmax
    dx = (off[..., 1].astype(np.int16) + rmax).astype(np.uint8)
    grid = np.stack([dy, dx], axis=-1).reshape(n_pairs, nrow, ncol, 2)

    reps: dict[str, dict] = {}

    # --- dense implicit-address (raster order); every byte is WHAT ---------------------
    dense = np.concatenate([dy.ravel(), dx.ravel()])
    reps["dense_implicit"] = _code_all(dense.tobytes(), grid, levels)
    reps["dense_implicit"]["order0_bits_REFERENCE_ONLY"] = _order0_bits(dense)

    # --- xi-transport: code the innovation against the previous pair's field ----------
    # Causal: the receiver has already decoded pair p-1.  Residual is re-centred into the
    # same alphabet by modular wrap, which is lossless and keeps SMEVR's level cap valid.
    ty = np.empty_like(dy)
    tx = np.empty_like(dx)
    ty[0], tx[0] = dy[0], dx[0]
    ty[1:] = (dy[1:].astype(np.int16) - dy[:-1].astype(np.int16)) % levels
    tx[1:] = (dx[1:].astype(np.int16) - dx[:-1].astype(np.int16)) % levels
    tdense = np.concatenate([ty.ravel(), tx.ravel()])
    tgrid = np.stack([ty, tx], axis=-1).reshape(n_pairs, nrow, ncol, 2)
    reps["xi_transport_temporal"] = _code_all(tdense.tobytes(), tgrid, levels)
    reps["xi_transport_temporal"]["order0_bits_REFERENCE_ONLY"] = _order0_bits(tdense)

    # --- sparse explicit-address: (cell_index, dy, dx) for non-zero cells only ---------
    # This is the form that PAYS gt2x's where-tax; raced to measure the tax rather than
    # assume it.
    nz = (off != 0).any(axis=2)
    n_nz = int(nz.sum())
    idx_bytes = int(np.ceil(np.log2(max(n_cells, 2)) / 8))
    sparse_parts = []
    for p in range(n_pairs):
        cells = np.flatnonzero(nz[p]).astype(np.uint32)
        sparse_parts.append(cells.astype(f"<u{max(idx_bytes, 1)}" if idx_bytes <= 4 else "<u4").tobytes())
        sparse_parts.append(dy[p][nz[p]].tobytes())
        sparse_parts.append(dx[p][nz[p]].tobytes())
    sparse = b"".join(sparse_parts)
    reps["sparse_explicit_address"] = {
        "zlib": len(zlib.compress(sparse, 9)),
        "lzma": len(lzma.compress(sparse, preset=9 | lzma.PRESET_EXTREME)),
        "brotli": _brotli(sparse),
        "smevr": None,  # heterogeneous record stream; not a level-capped field
        "n_nonzero_cells": n_nz,
        "address_bytes_uncoded": n_nz * idx_bytes,
        "value_bytes_uncoded": n_nz * 2,
        "where_frac_uncoded": n_nz * idx_bytes / max(n_nz * (idx_bytes + 2), 1),
    }
    return {"n_pairs": n_pairs, "n_cells": n_cells, "levels": levels, "reps": reps}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offsets", type=Path, required=True)
    ap.add_argument("--reach-receipt", type=Path, required=True)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--rate-denominator-bytes", type=int, default=37_545_489)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    reach = json.loads(args.reach_receipt.read_text())
    rung0 = reach["rung0_total_flips"]
    removed_by = {r["name"]: r["flips_removed"] for r in reach["rungs"]}
    npz = np.load(args.offsets)

    results = []
    for key in sorted((k for k in npz.files if k.startswith("block")), key=lambda s: -int(s[5:])):
        off = npz[key]
        r = race_block(off, args.rmax, int(key[5:]))
        removed = removed_by[key]
        ds_seg = 100.0 * removed / SCORED_PIXELS
        best_name, best_bytes = None, None
        for rep, coders in r["reps"].items():
            for cname, b in coders.items():
                if not isinstance(b, int) or cname.startswith("n_") or cname.endswith("_uncoded"):
                    continue
                if best_bytes is None or b < best_bytes:
                    best_name, best_bytes = f"{rep}/{cname}", b
        ds_rate = 25.0 * best_bytes / args.rate_denominator_bytes
        results.append(
            {
                "block": key,
                "flips_removed": removed,
                "delta_s_seg_gross": ds_seg,
                "best_coder": best_name,
                "best_bytes": best_bytes,
                "delta_s_rate_cost": ds_rate,
                "net_delta_s": ds_seg - ds_rate,
                "coded_bytes_per_flip_removed": best_bytes / removed if removed else float("inf"),
                "detail": r,
            }
        )

    receipt = {
        "schema": SCHEMA,
        "evidence_axis": AXIS_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "rmax": args.rmax,
        "rung0_total_flips": rung0,
        "n_pairs": reach["n_pairs"],
        "selection_mode": reach["selection_mode"],
        "rate_denominator_bytes": args.rate_denominator_bytes,
        "smevr_failures": _SMEVR_FAILURES[:20],
        "smevr_failure_count": len(_SMEVR_FAILURES),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1))

    print(f"\n== ddm_ph1 offset coder race (n={reach['n_pairs']}, mode={reach['selection_mode']}) ==")
    hdr = f"{'block':>8s} {'removed':>9s} {'dS_seg':>8s} {'bestcoder':>28s} {'bytes':>9s} {'dS_rate':>8s} {'NET dS':>9s}"
    print(hdr)
    for r in results:
        print(f"{r['block']:>8s} {r['flips_removed']:9d} {r['delta_s_seg_gross']:8.5f} "
              f"{r['best_coder']:>28s} {r['best_bytes']:9d} {r['delta_s_rate_cost']:8.5f} "
              f"{r['net_delta_s']:+9.5f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
