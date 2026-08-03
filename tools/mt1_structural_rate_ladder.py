#!/usr/bin/env python3
"""ddm_mt1 — price the two STRUCTURAL token bounds byte-exactly, at $0.

Context.  ``state/tokens.dr7t`` is 346,478 B = **96.2%** of the 360,323 B live own-vehicle
archive, so the token tensor IS the rate axis.  Its size is set by three multiplicative factors
that are all discrete choice points in the ``ddm_mt1`` inventory:

    codes = num_pairs x grid_h x grid_w x code_width          (count)
    grid_h x grid_w   = (384/grid_downsample) x (512/grid_downsample)
    bits per code     ~ log2(token_quant_levels)              (depth)

Two of the three sit at a menu endpoint on the SHIPPED archive:

* ``grid_downsample`` argparse ``choices=[8, 16]``, shipped **16** = the coarse endpoint.  The
  RECEIVER admits any power of two that closes 384x512 (``ddm_tr1_runtime.py:283`` requires a
  power of two, ``:331`` requires ``grid*ds == (384,512)``); ds=32 gives (12,16), both integers.
  So 32 is receiver-legal and forbidden only by the trainer's menu -- pw1's exact mechanism
  ("the capability was built; only the solver's constants pinned it").
* ``token_quant_levels`` shipped **16** == ``_R7_SMEVR_MAX_LEVELS`` (``:83``), enforced at ``:342``
  and ``:533``.  The default IS the codec ceiling.

``code_width`` is interior (shipped 4 of {2,4,6}) and is priced here as the reference axis.

WHAT THIS MEASURES AND WHAT IT DOES NOT
---------------------------------------
MEASURED: the byte-exact coded size of the reduced token tensor through the REAL
``ddm_r7_token_coder`` SMEVR encoder, plus a lossless round-trip check on every arm.
NOT MEASURED: d_seg.  Post-hoc reduction of a tensor trained at (16, 4, 16) is a LOWER bound on
the quality a run trained at the reduced setting would reach, exactly as ``ddm_bs2`` §5.2 noted
for its levels ladder.  Each arm therefore carries a PRE-REGISTERED d_seg break-even instead of a
claimed win.  No scorer job is run here.

Axis: [macOS-CPU $0 byte-exact rate; distortion NOT measured] NON-PROMOTABLE. score_claim=false.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in ("src", "experiments"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

LIVE_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pw1_archive.zip"
)
# S-arithmetic constants, all MEASURED and cited, never guessed.
#
# Every dS below is anchored on the LIVE BEST, not on a superseded row: quoting a percentage of
# a stale gap is an unanchored number.  LIVE BEST = ddm_dc1's folded own-vehicle candidate.
RATE_DENOM_BYTES = 37_545_489  # upstream/evaluate.py:63 rate term denominator
RATE_COEF = 25.0
LIVE_ARCHIVE_BYTES = 360_309  # dc1_fold own-vehicle LIVE BEST (pw1's 360,323 is superseded)
LIVE_COMPOSED_S = 0.8983766  # dc1_fold composed S (a PREDICTION; gate not self-fired)
BAR = 0.172141  # PR130 measured [contest-CUDA]
GAP_TO_BAR = LIVE_COMPOSED_S - BAR  # 0.7262356
# d_seg is INHERITED UNCHANGED across pw1 -> ms8 -> dc1_fold: dc1 verified state/tokens.dr7t is
# byte-identical to pw1's, so every token-side saving below transfers to the live best exactly.
LIVE_D_SEG = 0.00431179  # v4c/v4d realized gate


def _load_shipped_codes(archive: Path) -> tuple[np.ndarray, dict]:
    """Return the shipped uint8 token codes and the shipped selector config.

    ``state/tokens.dr7t`` is the BARE R7 stream (magic ``DR7T``), not a TR1 outer frame -- the
    outer ``_unpack_frame`` magic check refuses it.  Decoding through the coder directly is the
    only path that closes, and the coder carries its own semantic digest, so the read is still
    fully validated.
    """
    import ddm_r7_token_coder as coder

    with zipfile.ZipFile(archive) as z:
        payload = z.read("state/tokens.dr7t")
        selector = json.loads(z.read("state/selector.sec"))
    codes = np.asarray(coder.decode_token_codes(payload))
    want = (
        int(selector["num_pairs"]),
        int(selector["grid_h"]),
        int(selector["grid_w"]),
        int(selector["code_width"]),
    )
    if codes.shape != want:
        raise SystemExit(f"decoded token shape {codes.shape} != selector-declared {want}")
    return codes, selector


def _encode_bytes(codes: np.ndarray, levels: int) -> tuple[int, bool]:
    """Encode ``codes`` with the real SMEVR coder; return (bytes, lossless)."""
    import ddm_r7_token_coder as coder

    frame = bytes(
        coder.encode_token_codes(np.ascontiguousarray(codes, dtype=np.uint8), levels=levels)
    )
    back = coder.decode_token_codes(frame)
    return len(frame), bool(np.array_equal(np.asarray(back), codes))


def _pool_grid(codes: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen the (lead, gh, gw, cw) code grid by ``factor`` via exact block median.

    Median (not mean) keeps the result ON the existing integer lattice, so the arm measures the
    rate of a COARSER GRID and never smuggles in a finer lattice as well.  Confounding those two
    is precisely the 2x2-on-the-diagonal defect.
    """
    lead, gh, gw, cw = codes.shape
    if gh % factor or gw % factor:
        raise ValueError(f"grid {gh}x{gw} not divisible by {factor}")
    blocks = codes.reshape(lead, gh // factor, factor, gw // factor, factor, cw)
    med = np.median(blocks, axis=(2, 4))
    return np.rint(med).astype(np.uint8)


def _requantize(codes: np.ndarray, src_levels: int, dst_levels: int) -> np.ndarray:
    """Map codes from an ``src_levels`` lattice onto a ``dst_levels`` lattice (nearest)."""
    x = codes.astype(np.float64) / max(src_levels - 1, 1)
    return np.rint(x * (dst_levels - 1)).astype(np.uint8)


def _row(name: str, axis: str, value, nbytes: int, lossless: bool, base_bytes: int) -> dict:
    saved = base_bytes - nbytes
    archive = LIVE_ARCHIVE_BYTES - saved
    ds_rate = -RATE_COEF * saved / RATE_DENOM_BYTES
    # the arm pays iff the d_seg it costs is smaller than this; 100*d_seg is the seg term
    breakeven = abs(ds_rate) / 100.0 if saved > 0 else float("nan")
    return {
        "arm": name,
        "axis": axis,
        "value": value,
        "token_bytes": nbytes,
        "bytes_saved_vs_shipped": saved,
        "archive_bytes_if_swapped": archive,
        "dS_rate": ds_rate,
        "pct_of_gap": 100.0 * (-ds_rate) / GAP_TO_BAR if saved > 0 else float("nan"),
        "pays_iff_d_seg_rises_less_than": breakeven,
        "breakeven_as_pct_of_live_d_seg": 100.0 * breakeven / LIVE_D_SEG if saved > 0 else float("nan"),
        "lossless_roundtrip": lossless,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", type=Path, default=LIVE_ARCHIVE)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    codes, selector = _load_shipped_codes(args.archive)
    levels = int(selector["token_quant_levels"])
    lead, gh, gw, cw = codes.shape
    base_bytes, base_lossless = _encode_bytes(codes, levels)

    print(
        f"shipped codes {codes.shape} levels={levels} -> {base_bytes} B "
        f"(lossless={base_lossless}); archive member is 346478 B"
    )
    rows: list[dict] = []

    # --- axis 1: grid_downsample (the AT_A_BOUND row; 16 is the coarse endpoint) -------------
    #
    # ROUND-1 SELF-REVIEW CONTROL.  Median-pooling SMOOTHS the code field, and a smoother field
    # compresses better, so a pooled arm could flatter the byte count relative to what a run
    # actually TRAINED at the coarse grid would emit.  Decimation (stride subsample) preserves
    # the original per-cell marginal distribution and the original local roughness, so it is the
    # adversarial twin.  Both are reported; if they disagree materially the pooled number is the
    # optimistic one and must not be quoted alone.
    for factor, ds in ((2, 32), (4, 64)):
        try:
            pooled = _pool_grid(codes, factor)
        except ValueError as exc:
            print(f"  ds={ds}: SKIP ({exc})")
            continue
        nb, ll = _encode_bytes(pooled, levels)
        rows.append(_row(f"grid_downsample={ds}", "grid_downsample", ds, nb, ll, base_bytes))
        dec = np.ascontiguousarray(codes[:, ::factor, ::factor, :])
        nb_d, ll_d = _encode_bytes(dec, levels)
        rows.append(
            _row(
                f"grid_downsample={ds} [CONTROL: decimated]",
                "grid_downsample_control",
                ds,
                nb_d,
                ll_d,
                base_bytes,
            )
        )

    # --- axis 2: token_quant_levels (shipped == the SMEVR ceiling) --------------------------
    for lv in (14, 12, 10, 8):
        red = _requantize(codes, levels, lv)
        nb, ll = _encode_bytes(red, lv)
        rows.append(_row(f"token_quant_levels={lv}", "token_quant_levels", lv, nb, ll, base_bytes))

    # --- axis 3: code_width (the INTERIOR reference axis) -----------------------------------
    for keep in (3, 2):
        red = np.ascontiguousarray(codes[..., :keep])
        nb, ll = _encode_bytes(red, levels)
        rows.append(_row(f"code_width={keep}", "code_width", keep, nb, ll, base_bytes))

    hdr = f"{'arm':26s} {'B':>9s} {'saved':>8s} {'dS_rate':>10s} {'%gap':>7s} {'pays iff dseg<':>15s} {'ll':>3s}"
    print(hdr)
    for r in rows:
        print(
            f"{r['arm']:26s} {r['token_bytes']:9d} {r['bytes_saved_vs_shipped']:8d} "
            f"{r['dS_rate']:10.6f} {r['pct_of_gap']:7.2f} {r['pays_iff_d_seg_rises_less_than']:15.3e} "
            f"{'Y' if r['lossless_roundtrip'] else 'N':>3s}"
        )

    out = {
        "schema": "ddm_mt1_structural_rate_ladder.v1",
        "axis": "[macOS-CPU $0 byte-exact rate; d_seg NOT measured] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "archive": str(args.archive),
        "shipped": {
            "code_shape": list(codes.shape),
            "token_quant_levels": levels,
            "grid_downsample": int(selector["grid_downsample"]),
            "code_width": cw,
            "reencoded_bytes": base_bytes,
            "reencode_is_lossless": base_lossless,
            "archive_member_bytes": 346478,
        },
        "constants": {
            "rate_denominator_bytes": RATE_DENOM_BYTES,
            "live_archive_bytes": LIVE_ARCHIVE_BYTES,
            "live_d_seg": LIVE_D_SEG,
            "gap_to_bar": GAP_TO_BAR,
        },
        "caveat": (
            "post-hoc reduction of a tensor TRAINED at (ds=16, cw=4, levels=16) is a LOWER bound "
            "on the quality a run trained at the reduced setting would reach; the d_seg column is "
            "a PRE-REGISTERED break-even, never a measured win"
        ),
        "rows": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
