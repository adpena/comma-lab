#!/usr/bin/env python
"""ddm_br1 — the DROP SURFACE, measured for the first time at any level.

Two drop axes, BOTH format-free and receiver-free (verified against
`ddm_tr1_runtime.decode_token_grid`: it computes `codes/(levels-1)*2-1` with
`token_quant_levels` from the selector, so using a SUBSET of the 16-symbol
alphabet, or making a unit temporally constant, needs no receiver change and no
container change at all):

  A  UNIT DROP   — replace one (cell,channel) unit's 600 temporal values with
                   its own per-cell mode.  Residual becomes all-zero there.
                   These are exactly the charter's 768 cells x 4 rungs = 3072
                   waterfill units.
  B  LEVEL DROP  — coarsen the alphabet to a sublattice of {0..15}.

The byte half is EXACT and scorer-free.  The flip half needs the scorer slot,
so for every level we report the FLIP BUDGET = bytes_saved / W: the number of
argmax flips below which that level is profitable.  W = 4*DEN/PX exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac.optimization import ddm_ix2_archive_container as C  # noqa: E402

LEVELS = 16
DEN = 37_545_489
PX = 196_608 * 600
W = 4.0 * DEN / PX  # bytes per flip, exact

TOK = np.load("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy")
P, R, Cc, K = TOK.shape
NUNITS = R * Cc * K

BASE_BYTES = len(C.encode_token_frame(TOK, levels=LEVELS))


def frame_bytes(codes: np.ndarray) -> int:
    return len(C.encode_token_frame(np.ascontiguousarray(codes, dtype=np.uint8), levels=LEVELS))


def unit_activity(v: np.ndarray) -> np.ndarray:
    """Per-unit residual activity = nonzero count of the mode residual."""
    base, delta = C._factor_mode_delta(v, LEVELS)
    return (delta != 0).sum(axis=0).reshape(-1)  # (R*Cc*K,)


def drop_units(v: np.ndarray, unit_idx: np.ndarray) -> np.ndarray:
    """Make the listed units temporally constant at their own mode."""
    base, _ = C._factor_mode_delta(v, LEVELS)
    out = v.copy()
    flat = out.reshape(P, -1)
    bflat = base.reshape(-1)
    flat[:, unit_idx] = bflat[unit_idx][None, :]
    return out


def coarsen(v: np.ndarray, keep: int) -> np.ndarray:
    """Map {0..15} onto `keep` equally spaced representatives INSIDE {0..15}.

    keep=16 is the identity.  The receiver is untouched: values stay in the
    same 16-symbol alphabet with the same dequantiser.
    """
    if keep >= LEVELS:
        return v.copy()
    reps = np.rint(np.linspace(0, LEVELS - 1, keep)).astype(np.int64)
    x = v.astype(np.int64)
    idx = np.abs(x[..., None] - reps[None, None, None, None, :]).argmin(axis=-1)
    return reps[idx].astype(np.uint8)


def row(label: str, axis: str, nbytes: int, extra: dict) -> dict:
    saved = BASE_BYTES - nbytes
    d = {
        "axis": axis,
        "level": label,
        "bytes": nbytes,
        "bytes_saved": saved,
        "pct_of_token_member": 100.0 * saved / BASE_BYTES,
        "flip_budget": saved / W,
        "dS_rate_only": -25.0 * saved / DEN,
    }
    d.update(extra)
    return d


def main():
    print(f"W = {W!r} B/flip")
    print(f"base token member = {BASE_BYTES} B  ({NUNITS} units x {P} pairs)")

    act = unit_activity(TOK)
    print(
        f"unit activity: dead(0 nonzero)={int((act==0).sum())}/{NUNITS}  "
        f"median={np.median(act):.0f}  max={act.max()}  mean={act.mean():.1f}"
    )

    rows = [row("none (control)", "control", BASE_BYTES, {"units_dropped": 0})]

    # ---- AXIS A: unit drop, byte-greedy (most active first) ----------------
    order_hi = np.argsort(-act, kind="stable")
    order_lo = np.argsort(act, kind="stable")
    fracs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
    for f in fracs:
        n = int(round(f * NUNITS))
        for name, order in (("most-active-first", order_hi), ("least-active-first", order_lo)):
            b = frame_bytes(drop_units(TOK, order[:n]))
            r = row(f"drop {f*100:g}% units ({name})", "A_unit_drop", b,
                    {"units_dropped": n, "ranking": name})
            rows.append(r)
            print(
                f"  A {f*100:5.1f}% {name:19s} n={n:4d}  {b:7d} B  "
                f"saved={r['bytes_saved']:6d}  flip_budget={r['flip_budget']:9.0f}"
            )

    # ---- AXIS B: level drop ------------------------------------------------
    for keep in (16, 12, 9, 8, 6, 5, 4, 3, 2):
        cv = coarsen(TOK, keep)
        b = frame_bytes(cv)
        changed = float((cv != TOK).mean())
        r = row(f"alphabet {keep}/16", "B_level_drop", b,
                {"keep_levels": keep, "frac_tokens_changed": changed})
        rows.append(r)
        print(
            f"  B keep={keep:2d}  {b:7d} B  saved={r['bytes_saved']:6d}  "
            f"changed={changed*100:5.1f}%  flip_budget={r['flip_budget']:9.0f}"
        )

    out = Path("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/br1_drop_surface.json")
    out.write_text(json.dumps({"W": W, "base_bytes": BASE_BYTES, "rows": rows}, indent=1))
    np.save("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/br1_unit_activity.npy", act)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
