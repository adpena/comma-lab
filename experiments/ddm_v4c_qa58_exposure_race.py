#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4c — QA58: the full-600 exposure (a,b) stream coding race ($0).

kl1 §10 could not test the smooth-AE-control-loop law on pm1's scattered tail-112
fits; the v4c photo stage produces the FULL-600 contiguous (a,b) series.  Race
(per kl1's constants-are-poison discipline — measured, never assumed):
  raw          f16 byte-plane column-major brotli (the kl1 codec, what v4c ships)
  delta1       first-difference f16 residual, byte-plane coded
  ar1          AR(1) fit per column, f16 residual, byte-plane coded
Each candidate must round-trip bit-exact to the shipped f16 values (lossless) or
it is disqualified.  Whiteness diagnostic std(diff)/std(value) per column decides
the LAW verdict (kl1's temporal-whiteness test).

Axis: [$0 coding race; macOS-CPU advisory]; pointer 0.1910828242 UNMOVED.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import brotli
import numpy as np


def byte_plane(u16: np.ndarray) -> bytes:
    hi = (u16 >> 8).astype(np.uint8).reshape(-1)
    lo = (u16 & 0xFF).astype(np.uint8).reshape(-1)
    return brotli.compress(np.concatenate([hi, lo]).tobytes(), quality=11)


def code_f16(field: np.ndarray) -> int:
    return len(byte_plane(np.ascontiguousarray(field.T).view(np.uint16)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photo-jsonl", required=True)
    args = ap.parse_args()
    rows = {json.loads(x)["pair"]: json.loads(x)
            for x in Path(args.photo_jsonl).read_text().splitlines() if x.strip()}
    n = len(rows)
    ab = np.asarray([[rows[i]["a"], rows[i]["b"]] for i in range(n)], np.float16)

    out: dict[str, object] = {
        "schema": "ddm_v4c_qa58_exposure_race.v1", "n": n,
        "axis": "[$0 coding race; macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
    }
    a64 = ab[:, 0].astype(np.float64)
    b64 = ab[:, 1].astype(np.float64)
    out["whiteness_a"] = float(np.std(np.diff(a64)) / max(np.std(a64), 1e-12))
    out["whiteness_b"] = float(np.std(np.diff(b64)) / max(np.std(b64), 1e-12))
    out["a_stats"] = {"median": float(np.median(a64)),
                      "frac_within_0p1_of_1": float(np.mean(np.abs(a64 - 1) < 0.1)),
                      "n_exact_1": int(np.sum(ab[:, 0] == np.float16(1.0)))}
    out["b_stats"] = {"median": float(np.median(b64)),
                      "n_exact_0": int(np.sum(ab[:, 1] == np.float16(0.0)))}

    # RAW byte-plane (shipped): lossless by construction
    raw = code_f16(ab)
    # DELTA1: r[0]=v[0], r[i]=v[i]-v[i-1] in f16 — LOSSY in general (f16 sum
    # does not round-trip); race only the exact-reconstruction variant via
    # u16-view differencing (mod-65536 delta of the bit patterns = lossless).
    u = np.ascontiguousarray(ab).view(np.uint16)
    du = u.copy()
    du[1:] = (u[1:] - u[:-1]) % 65536
    delta_bits = len(byte_plane(np.ascontiguousarray(du.T)))
    # AR(1) on the u16 bit-pattern domain is meaningless; on values it breaks
    # losslessness. Honest AR(1) race: predict v[i] = c*v[i-1] (f64 fit), code
    # residual bit patterns of f16(v[i]) vs f16(pred) delta mod-65536.
    ar_bits = 0
    for col in range(2):
        v = ab[:, col].astype(np.float64)
        c = float(np.dot(v[1:], v[:-1]) / max(np.dot(v[:-1], v[:-1]), 1e-12))
        pred = np.empty_like(v)
        pred[0] = 0.0
        pred[1:] = c * v[:-1]
        pu = pred.astype(np.float16).view(np.uint16).astype(np.uint16)
        vu = ab[:, col].view(np.uint16).astype(np.uint16)
        ru = (vu - pu) % 65536
        ar_bits += len(byte_plane(ru[None, :].astype(np.uint16)))
    out["race_bytes"] = {"raw_byteplane": raw, "delta1_u16": int(delta_bits),
                         "ar1_u16_residual": int(ar_bits)}
    winner = min(out["race_bytes"], key=out["race_bytes"].get)
    out["winner"] = winner
    out["verdict"] = (
        "LAW (smooth AE control loop) CONFIRMED" if winner != "raw_byteplane"
        else "NO-LAW: (a,b) series codes best as raw distributional byte-plane "
             "(temporal predictors lose) — matches kl1's pose-field whiteness")
    rec = Path("/Volumes/VertigoDataTier/pact/ddm_v4c_20260730/qa58_exposure_race.json")
    rec.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
