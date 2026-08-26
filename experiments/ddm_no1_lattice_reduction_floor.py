"""ddm_no1 — the coded-support (lattice) reduction floor, measured scorer-free on the retained
dx2 token field.

WHAT THIS MEASURES AND WHY IT IS A FALSIFIER
--------------------------------------------
`ddm_tba1` section 6 records the invariant that closes this direction's status as UNKNOWN:

    "The invariant across every prior arm that produced a byte number: N was held at exactly
     117,964,800. Not one changed how many positions are coded."

tba1's D2a/D2b close *subset* reduction (drop CHEAP / drop EXPENSIVE positions) because a chosen
subset must be NAMED and the selector tax exceeds the prize. A REGULAR LATTICE is the one support
reduction that pays no selector tax at all: the kept set is a global, receiver-derivable rule whose
description length is O(1) and does not grow with |kept| (the `ddm_af1` section 3 criterion ->
address-FREE).

So the family's whole price is DISTORTION: the decoder must reconstruct the dropped positions from
the kept ones. This script measures that reconstruction error EXACTLY, with no scorer, no renderer,
no training, and no dispatch -- it is a property of the FIELD, not of any vehicle. It therefore
bounds the family from a direction that a retrain cannot move: no amount of training lets a decoder
recover information that was never transmitted.

Instruments, in increasing strength (all measured on the same pinned field):
  R1  nearest / replicate          -- 1 coarse neighbour, an ACHIEVED rule.
  R2  one-hot bilinear + argmax    -- 4 coarse neighbours, an ACHIEVED rule (this is what a smooth
                                      learned upsampler approximates).
  O4  in-sample Bayes-optimal predictor on the 2x2 coarse neighbourhood (5^4 = 625 contexts).
  O9  in-sample Bayes-optimal predictor on the 3x3 coarse neighbourhood (5^9 contexts), f=(2,2) only.

O4 and O9 are LOWER BOUNDS on the error of ANY deterministic rule that reads only that coarse
neighbourhood, and they are fitted and evaluated on the SAME data, so they are optimistic even for
that class -- the generous direction, which is what a kill argument requires.

BYTE CREDIT is taken as the EXACT static sum of the shipped coder's per-position -log2(p) cost over
the DROPPED positions. `ddm_ds1` measured static -log2(p) accounting on this exact field mispricing
by 14.59x in the FALSE-WIN direction, so this credit is an OVERSTATEMENT -- again the generous
direction. No re-encode is performed and no rate claim is made from it beyond an upper bound.

POSE is held at the dx2 value throughout. `ddm_gd3` measured that any token-side change invalidates
the fitted pose carrier (its probe: d_pose 0.00517 -> 76.19 on the then-live vehicle). Holding pose
fixed is therefore also generous.

AXIS: [macOS-CPU scorer-free advisory]. score_claim=false, promotion_eligible=false,
pointer_moved=false. No Modal, no Metal, no scorer, no training, no upstream/ write.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Pinned inputs (read-only) and the measured constants this arm CITES.
# ---------------------------------------------------------------------------

FIELD_PATH = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/"
    "retained/fields/decoded_tokens_instrumented.u8"
)
COST_PATH = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/"
    "retained/fields/position_rc64_frequency_cost_bits.f64le.bin"
)
# ddm_tba1 section 10 charter pin for the decoded categorical field.
FIELD_SHA256_PIN = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"

N_PAIRS = 600
H = 384
W = 512
N_POS = N_PAIRS * H * W  # 117,964,800
N_CLASSES = 5

# --- CITED measured constants (never re-derived here) ----------------------
# ddm_tx1 section 0 exchange rate, CITED per #1207.
LAMBDA_B = 6.658590e-07  # S per archive byte
# Live pointer: gb1, twentieth move.
POINTER_ARCHIVE_B = 180_215
POINTER_S = 0.14811799921260607
DX2_D_SEG = 0.00020139
DX2_D_POSE = 0.00000637
# ddm_ar1b exact zero-remainder census: the physical RC64 token stream.
TOKEN_STREAM_B = 113_777
TARGET_S = 0.12
# ddm_dg2 MEASURED conversion from a token-field edit to a realized d_seg flip.
FINAL_FLIPS_PER_EDIT = 0.9528

# Canonical comma10k class order (CLAUDE.md; NEVER luma-sorted). Areas are the
# n600 values reproduced by ddm_tba1 section 2 on this exact field.
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
TBA1_AREA_PCT = (23.2331, 0.5858, 49.5175, 1.2380, 25.4255)
TBA1_ROW_CENTROID = (239.3, 226.4, 95.0, 199.1, 334.6)

LATTICES: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2), (2, 2), (4, 4))


def seg_term(d_seg: float) -> float:
    return 100.0 * d_seg


def pose_term(d_pose: float) -> float:
    return math.sqrt(10.0 * d_pose)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class LatticeRow:
    fy: int
    fx: int
    kept_positions: int
    dropped_positions: int
    kept_fraction: float
    static_credit_bits: float
    static_credit_bytes: float
    err_nearest: int
    err_bilinear: int
    err_oracle4: int
    err_oracle9: int  # -1 when not computed


# ---------------------------------------------------------------------------
# Reconstruction rules
# ---------------------------------------------------------------------------


def coarse_of(field: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """Subsample: keep fine pixel (fy*i, fx*j). Shape (T, ceil(H/fy), ceil(W/fx))."""
    return field[:, ::fy, ::fx]


def recon_nearest(coarse: np.ndarray, fy: int, fx: int, h: int, w: int) -> np.ndarray:
    """R1: replicate each kept sample over its block, clipped to the fine shape."""
    up = np.repeat(np.repeat(coarse, fy, axis=1), fx, axis=2)
    return up[:, :h, :w]


def _bilinear_weights(n_fine: int, n_coarse: int, f: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine index -> (lower coarse index, upper coarse index, upper weight).

    Coarse sample k sits at fine coordinate k*f. Fine coordinate y therefore lies between
    coarse floor(y/f) and that +1, with fractional part (y % f) / f. Edge coordinates beyond
    the last coarse sample clamp to it.
    """
    y = np.arange(n_fine)
    lo = np.minimum(y // f, n_coarse - 1)
    hi = np.minimum(lo + 1, n_coarse - 1)
    frac = (y - lo * f) / float(f)
    frac = np.where(hi == lo, 0.0, frac)
    return lo.astype(np.int32), hi.astype(np.int32), frac.astype(np.float32)


def recon_bilinear_argmax(coarse: np.ndarray, fy: int, fx: int, h: int, w: int) -> np.ndarray:
    """R2: bilinear-interpolate the 5-channel one-hot of the coarse field, then argmax.

    Ties break to the lowest class index (numpy argmax), which is deterministic.
    """
    t, ch, cw = coarse.shape
    ry_lo, ry_hi, wy = _bilinear_weights(h, ch, fy)
    rx_lo, rx_hi, wx = _bilinear_weights(w, cw, fx)
    out = np.empty((t, h, w), dtype=np.uint8)
    for i in range(t):
        c = coarse[i]
        best = np.full((h, w), -1.0, dtype=np.float32)
        arg = np.zeros((h, w), dtype=np.uint8)
        for cls in range(N_CLASSES):
            oh = (c == cls).astype(np.float32)
            # rows first
            r = oh[ry_lo, :] * (1.0 - wy)[:, None] + oh[ry_hi, :] * wy[:, None]
            # then columns
            v = r[:, rx_lo] * (1.0 - wx)[None, :] + r[:, rx_hi] * wx[None, :]
            upd = v > best
            best = np.where(upd, v, best)
            arg = np.where(upd, np.uint8(cls), arg)
        out[i] = arg
    return out


def oracle_error(
    field: np.ndarray,
    coarse: np.ndarray,
    fy: int,
    fx: int,
    ctx_radius: int,
) -> int:
    """In-sample Bayes-optimal predictor of each DROPPED fine pixel from a coarse neighbourhood.

    ctx_radius=1 -> the 2x2 block corner neighbourhood (5^4 = 625 contexts).
    ctx_radius=2 -> the 3x3 coarse neighbourhood (5^9 contexts).

    Returns the total number of dropped fine pixels the optimal predictor still gets wrong.
    This is a LOWER BOUND on the error of any deterministic rule reading only that neighbourhood,
    and it is fitted and scored on the same data, so it is optimistic even within that class.
    """
    t, ch, cw = coarse.shape
    h, w = field.shape[1], field.shape[2]

    if ctx_radius == 1:
        offs = ((0, 0), (0, 1), (1, 0), (1, 1))
    elif ctx_radius == 2:
        offs = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported ctx_radius {ctx_radius}")
    n_ctx = N_CLASSES ** len(offs)

    # Coarse index of the block containing each fine pixel.
    fy_idx = np.minimum(np.arange(h) // fy, ch - 1).astype(np.int32)
    fx_idx = np.minimum(np.arange(w) // fx, cw - 1).astype(np.int32)

    # Context id per (frame, block-row, block-col), built once.
    ctx = np.zeros((t, ch, cw), dtype=np.int64)
    for dy, dx in offs:
        iy = np.clip(np.arange(ch) + dy, 0, ch - 1)
        ix = np.clip(np.arange(cw) + dx, 0, cw - 1)
        ctx *= N_CLASSES
        ctx += coarse[:, iy, :][:, :, ix].astype(np.int64)

    total_wrong = 0
    for sy in range(fy):
        for sx in range(fx):
            if sy == 0 and sx == 0:
                continue  # the kept sample is transmitted exactly
            rows = np.arange(sy, h, fy)
            cols = np.arange(sx, w, fx)
            if rows.size == 0 or cols.size == 0:
                continue
            labels = field[:, rows, :][:, :, cols].reshape(-1).astype(np.int64)
            sub_ctx = ctx[:, fy_idx[rows], :][:, :, fx_idx[cols]].reshape(-1)
            key = sub_ctx * N_CLASSES + labels
            counts = np.bincount(key, minlength=n_ctx * N_CLASSES).reshape(n_ctx, N_CLASSES)
            total_wrong += int(labels.size - counts.max(axis=1).sum())
            del labels, sub_ctx, key, counts
    return total_wrong


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def orientation_control(field: np.ndarray) -> dict:
    """Reproduce ddm_tba1 section 2's per-class area and row centroid.

    A transposed or mis-strided reshape moves the centroids by tens of rows, so this both
    validates the (600, 384, 512) layout and re-confirms the canonical class order without
    ever luma-sorting.
    """
    rows = np.arange(H, dtype=np.float64)
    areas = []
    centroids = []
    for cls in range(N_CLASSES):
        m = field == cls
        n = int(m.sum())
        per_row = m.sum(axis=(0, 2)).astype(np.float64)
        areas.append(100.0 * n / N_POS)
        centroids.append(float((per_row * rows).sum() / max(per_row.sum(), 1.0)))
    area_err = [abs(a - b) for a, b in zip(areas, TBA1_AREA_PCT, strict=True)]
    cent_err = [abs(a - b) for a, b in zip(centroids, TBA1_ROW_CENTROID, strict=True)]
    return {
        "class_order": list(CLASS_NAMES),
        "area_pct": areas,
        "row_centroid": centroids,
        "tba1_area_pct": list(TBA1_AREA_PCT),
        "tba1_row_centroid": list(TBA1_ROW_CENTROID),
        "max_abs_area_pct_error": max(area_err),
        "max_abs_row_centroid_error": max(cent_err),
        "orientation_ok": max(area_err) < 0.01 and max(cent_err) < 0.5,
    }


def breakeven_flips_per_edit(archive_bytes: float, field_errors: int) -> float:
    """The value of FINAL_FLIPS_PER_EDIT at which this rung would exactly reach S = TARGET_S.

    FINAL_FLIPS_PER_EDIT is `ddm_dg2`'s measurement on a DIFFERENT edit family, so using it here
    is a cross-regime constant transfer. This function says how wrong that transfer would have to
    be, and in which direction, for the verdict to flip -- so no verdict rests on the transfer.
    """
    if field_errors <= 0:
        return float("inf")
    head = TARGET_S - LAMBDA_B * archive_bytes - pose_term(DX2_D_POSE) - seg_term(DX2_D_SEG)
    return head * N_POS / (100.0 * field_errors)


def admission(archive_bytes: float, d_seg: float, d_pose: float) -> dict:
    s = LAMBDA_B * archive_bytes + seg_term(d_seg) + pose_term(d_pose)
    distortion_only = seg_term(d_seg) + pose_term(d_pose)
    b_max = (TARGET_S - distortion_only) / LAMBDA_B
    return {
        "archive_bytes": archive_bytes,
        "S": s,
        "distortion_only_S": distortion_only,
        "dead_at_any_archive": distortion_only >= TARGET_S,
        "B_max_bytes": b_max,
        "passes_sub012": s < TARGET_S,
    }


def main() -> int:
    t0 = time.time()
    out_dir = Path(os.environ.get("DDM_NO1_OUT", ".omx/tmp/ddm_no1_lattice_floor"))
    out_dir.mkdir(parents=True, exist_ok=True)

    verify_sha = os.environ.get("DDM_NO1_VERIFY_SHA", "1") == "1"
    # The 3x3-coarse-neighbourhood oracle is the strongest (most generous) lower bound this
    # instrument computes. It is run on the rungs whose 2x2-oracle bound lands closest to a
    # threshold, so that no verdict rests on the weaker bound.
    oracle9_set = {
        tuple(int(v) for v in tok.split("x"))
        for tok in os.environ.get("DDM_NO1_ORACLE9_LATTICES", "2x2,1x2").split(",")
        if tok.strip()
    }

    if not FIELD_PATH.exists() or not COST_PATH.exists():
        print(f"REFUSE: pinned input missing ({FIELD_PATH} / {COST_PATH})", file=sys.stderr)
        return 2

    field_sha = sha256_file(FIELD_PATH) if verify_sha else "SKIPPED"
    if verify_sha and field_sha != FIELD_SHA256_PIN:
        print(
            "REFUSE: decoded token field sha256 does not match the ddm_tba1 charter pin.\n"
            f"  measured {field_sha}\n  pinned   {FIELD_SHA256_PIN}",
            file=sys.stderr,
        )
        return 3

    raw = np.memmap(FIELD_PATH, dtype=np.uint8, mode="r")
    if raw.size != N_POS:
        print(f"REFUSE: field has {raw.size} bytes, expected {N_POS}", file=sys.stderr)
        return 4
    # 117,964,800 B copied into RAM once: every rule below indexes it repeatedly and a
    # memmap would re-read from the external SSD on each fancy-index.
    field = np.array(raw, dtype=np.uint8).reshape(N_PAIRS, H, W)
    del raw

    ctl = orientation_control(field)
    if not ctl["orientation_ok"]:
        print("REFUSE: orientation control failed against ddm_tba1 section 2", file=sys.stderr)
        print(json.dumps(ctl, indent=2), file=sys.stderr)
        return 5

    cost = np.memmap(COST_PATH, dtype="<f8", mode="r")
    if cost.size != N_POS:
        print(f"REFUSE: cost field has {cost.size} entries, expected {N_POS}", file=sys.stderr)
        return 6
    cost3 = np.asarray(cost).reshape(N_PAIRS, H, W)
    total_cost_bits = float(cost3.sum())

    rows: list[LatticeRow] = []
    masks: dict[str, np.ndarray] = {}
    for fy, fx in LATTICES:
        coarse = coarse_of(field, fy, fx)
        kept = int(coarse.shape[0] * coarse.shape[1] * coarse.shape[2])
        dropped = N_POS - kept

        # Exact static credit: the shipped coder's own cost on the dropped positions.
        kept_cost = float(cost3[:, ::fy, ::fx].sum())
        credit_bits = total_cost_bits - kept_cost

        r1 = recon_nearest(coarse, fy, fx, H, W)
        m1 = r1 != field
        e1 = int(m1.sum())
        masks[f"nearest_{fy}x{fx}"] = m1

        r2 = recon_bilinear_argmax(coarse, fy, fx, H, W)
        m2 = r2 != field
        e2 = int(m2.sum())
        masks[f"bilinear_{fy}x{fx}"] = m2
        del r1, r2

        e4 = oracle_error(field, coarse, fy, fx, ctx_radius=1) if (fy, fx) != (1, 1) else 0
        e9 = -1
        if (fy, fx) in oracle9_set:
            e9 = oracle_error(field, coarse, fy, fx, ctx_radius=2)

        rows.append(
            LatticeRow(
                fy=fy,
                fx=fx,
                kept_positions=kept,
                dropped_positions=dropped,
                kept_fraction=kept / N_POS,
                static_credit_bits=credit_bits,
                static_credit_bytes=credit_bits / 8.0,
                err_nearest=e1,
                err_bilinear=e2,
                err_oracle4=e4,
                err_oracle9=e9,
            )
        )
        print(
            f"[{time.time() - t0:7.1f}s] lattice {fy}x{fx}: kept {kept:,} "
            f"credit {credit_bits / 8.0:,.1f} B  nearest {e1:,}  bilinear {e2:,}  "
            f"oracle4 {e4:,}  oracle9 {e9:,}",
            flush=True,
        )
        del coarse

    # ---- admission arithmetic -------------------------------------------
    verdicts = []
    for r in rows:
        row_out = {"lattice": f"{r.fy}x{r.fx}", **asdict(r), "rules": {}}
        for rule, errs in (
            ("nearest", r.err_nearest),
            ("bilinear", r.err_bilinear),
            ("oracle4", r.err_oracle4),
            ("oracle9", r.err_oracle9),
        ):
            if errs < 0:
                continue
            # Field-level reconstruction errors -> realized d_seg flips at dg2's measured rate.
            added_flips = errs * FINAL_FLIPS_PER_EDIT
            d_seg_add = added_flips / N_POS
            d_seg_new = DX2_D_SEG + d_seg_add
            arch = POINTER_ARCHIVE_B - r.static_credit_bytes

            k_real = breakeven_flips_per_edit(arch, errs)
            k_zero = breakeven_flips_per_edit(0.0, errs)
            row_out["rules"][rule] = {
                "field_errors": errs,
                "field_errors_per_frame": errs / N_PAIRS,
                "d_seg_added": d_seg_add,
                "d_seg_new": d_seg_new,
                "flips_per_edit_breakeven_real_budget": k_real,
                "flips_per_edit_breakeven_zero_byte": k_zero,
                "over_real_budget_x": (
                    FINAL_FLIPS_PER_EDIT / k_real if k_real > 0 else float("inf")
                ),
                "over_zero_byte_x": (
                    FINAL_FLIPS_PER_EDIT / k_zero if k_zero > 0 else float("inf")
                ),
                **admission(arch, d_seg_new, DX2_D_POSE),
            }
        verdicts.append(row_out)

    result = {
        "arm": "ddm_no1",
        "instrument": "lattice_reduction_floor",
        "axis": "[macOS-CPU scorer-free advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "field_path": str(FIELD_PATH),
        "field_sha256": field_sha,
        "field_sha256_pin": FIELD_SHA256_PIN,
        "cost_path": str(COST_PATH),
        "total_cost_bits": total_cost_bits,
        "token_stream_bytes_cited": TOKEN_STREAM_B,
        "orientation_control": ctl,
        "cited_constants": {
            "lambda_B_S_per_byte": LAMBDA_B,
            "pointer_archive_bytes": POINTER_ARCHIVE_B,
            "pointer_S": POINTER_S,
            "dx2_d_seg": DX2_D_SEG,
            "dx2_d_pose": DX2_D_POSE,
            "final_flips_per_edit_dg2": FINAL_FLIPS_PER_EDIT,
            "target_S": TARGET_S,
        },
        "rows": verdicts,
        "elapsed_s": time.time() - t0,
    }

    # ---- payload retention (P0) -----------------------------------------
    # The reconstructed fields themselves are a pure deterministic function of the pinned
    # input field + (fy, fx) + rule, so they are certified rebuildable and not duplicated.
    # What is PERSISTED is the scientific payload: the exact disagreement masks (packbits)
    # plus the RESULT.
    retained = {}
    for name, m in masks.items():
        p = out_dir / f"disagree_{name}.n600.packbits"
        pb = np.packbits(m.reshape(-1))
        pb.tofile(p)
        retained[p.name] = {
            "bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
            "popcount": int(m.sum()),
            "bitorder": "big (numpy packbits default)",
        }
    result["retained_payload"] = retained
    result["payload_note"] = (
        "Reconstructed fields are certified rebuildable: recon_nearest / recon_bilinear_argmax "
        "are deterministic functions of the sha-pinned input field and (fy, fx). The measured "
        "disagreement masks are persisted in full."
    )

    rp = out_dir / "RESULT.json"
    rp.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
