#!/usr/bin/env python3
"""RA1 correction: the rank-r keep set is EXHAUSTIVELY optimal, not energy-greedy.

ra1 (`ddm_ra1_carrier_rank_refit_preproof.py`) selects which r of the 12 carrier
atoms to keep by a GREEDY ENERGY heuristic

    order_energy = argsort(-(coeff**2).mean(axis=0) * diag(gram))

and then least-squares-refits the kept coefficients. The refit IS optimal given a
keep set -- but the keep set is not, and ra1's charter (P2) published the result as
"a LOWER BOUND on the reconstruction error of EVERY rank-r carrier ... a rank that
fails here fails under every refit heuristic". That is FALSE as published.

MEASURED by this tool (round-1 recursive adversarial review, 2026-08-16): the
greedy keep set is suboptimal at 10 of 11 ranks, by up to 2.01x in field MSE. The
whole search space is C(12,r), summing to 4,094 non-trivial subsets -- exhaustible
in seconds. There is no reason to use a heuristic at this size.

Consequence for the campaign: at rank 4 (the headline rung, 102% of the remaining
gap in rate credit) the exhaustive keep set is BOTH lower-error AND cheaper --
20.41% vs 30.60% carrier error at 7,471 vs 7,569 coded bytes. It strictly
dominates. Rank 6 halves the error (23.22% -> 11.55%) and also saves 23 B.

What this tool does NOT change: the byte accounting, the receiver geometry, the
custody pins, or premise P1 (the carrier renders frame_0 only, so a rank cut is
seg-invisible by construction). It replaces exactly one line of ra1 -- the keep-set
choice -- and re-prices every rung through the SAME shipped CPR1 codec and the SAME
shipped Brotli cell, so every byte number remains a real coded length.

Still NOT measured here, and still the binding question: d_pose. Reconstruction
error is a EUCLIDEAN quantity; the scored quantity is a PoseNet readout. The
exhaustive set minimises the former, which is not the same as minimising the
latter -- a pose-Jacobian-weighted (Fisher) selection is a further, strictly better
aimed candidate at identical bytes and remains unbuilt. See AMENDMENT 2 of
.omx/research/charters/ddm_ra2_carrier_rank_pose_calibration.md.

Axis: exact coded bytes [MEASURED]; carrier-field MSE [MEASURED, exact closed
form]; d_pose [NOT measured here -- ra2 owns it, on a same-instrument base].
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import brotli
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RA1_SOURCE = REPO / "experiments/ddm_ra1_carrier_rank_refit_preproof.py"
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_ra1b_exhaustive_keepset_20260816/retained")


def load_ra1():
    """Import ra1 wholesale: same custody pins, same codec, same geometry."""
    spec = importlib.util.spec_from_file_location("ra1_preproof", RA1_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ra1_preproof"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.time()

    out = args.output
    payloads = out / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)

    ra1 = load_ra1()
    custody = ra1.verify_custody()  # fail-closed on the four frontier pins
    codec, repack, encoder = ra1.load_codec()
    basis_scales, basis_codes, coeff_scales, encoded, coeff_codes = ra1.decode_reference(
        codec, repack
    )
    D = ra1.CARRIER_DIM

    raw_basis = (
        torch.from_numpy(basis_codes.astype(np.float32))
        .reshape(D, 3, ra1.CARRIER_H, ra1.CARRIER_W)
        * torch.from_numpy(np.asarray(basis_scales, dtype=np.float32))[:, None, None, None]
    )
    Bn = ra1.normalized_basis(raw_basis.to(torch.float64))
    flat = Bn.reshape(D, -1)
    gram = (flat @ flat.T).numpy() / flat.shape[1]
    coeff = coeff_codes.astype(np.float64) * np.asarray(coeff_scales, dtype=np.float64)
    pix_scale = (ra1.CARRIER_AMPLITUDE ** 2) / D

    def field_mse(delta_c: np.ndarray) -> float:
        return float(pix_scale * np.einsum("bi,ij,bj->b", delta_c, gram, delta_c).mean())

    signal_energy = field_mse(coeff)
    baseline_coded = len(brotli.compress(ra1.CANONICAL_CPR1, quality=ra1.BROTLI_QUALITY))
    greedy_order = np.argsort(-(coeff ** 2).mean(axis=0) * np.diag(gram))
    basis_codes_rows = basis_codes.reshape(D, -1)

    def refit(keep) -> tuple[np.ndarray, float, float]:
        """Least-squares refit onto `keep`; returns (c_refit, mse, cond(Grr))."""
        keep = np.sort(np.asarray(keep))
        Grr = gram[np.ix_(keep, keep)]
        Grc = gram[np.ix_(keep, np.arange(D))]
        c_refit = np.linalg.lstsq(Grr, Grc @ coeff.T, rcond=None)[0].T
        approx = np.zeros_like(coeff)
        approx[:, keep] = c_refit
        return c_refit, field_mse(coeff - approx), float(np.linalg.cond(Grr))

    def realise(keep) -> dict:
        """Requantise onto the shipped int12 lattice and encode through the real codec."""
        keep = np.sort(np.asarray(keep))
        c_refit, mse_refit, cond = refit(keep)
        sub_scale = np.maximum(np.abs(c_refit).max(axis=0) / 2047.0, 1e-12)
        q = np.clip(np.rint(c_refit / sub_scale), -2048, 2047).astype(np.int64)
        delta = np.diff(np.concatenate([np.zeros((1, len(keep)), dtype=np.int64), q]), axis=0)
        # Wrap into the signed 12-bit range BEFORE zigzag; the receiver's cumsum
        # is modular (inflate.py:278). See ra1's _assert_round_trip docstring.
        delta = ((delta + 2048) & 0xFFF) - 2048
        zigzag = ((delta << 1) ^ (delta >> 63)) & 0xFFF
        ra1._assert_round_trip(zigzag, q)
        blob = ra1.encoded_bytes(
            encoder,
            np.asarray(basis_scales)[keep],
            basis_codes_rows[keep].reshape(-1),
            sub_scale.astype("<f4"),
            zigzag,
        )
        approx_q = np.zeros_like(coeff)
        approx_q[:, keep] = q.astype(np.float64) * sub_scale
        return {
            "keep": [int(k) for k in keep],
            "coded_bytes": int(blob["coded_bytes"]),
            "mse_refit_lower_bound_pixel_units": mse_refit,
            "mse_realised_int12_pixel_units": field_mse(coeff - approx_q),
            "cond_Grr": cond,
            "payload": blob,
        }

    rows = []
    for r in range(1, D):
        greedy = realise(greedy_order[:r])
        best_keep, best_mse = None, None
        for subset in itertools.combinations(range(D), r):
            _, mse, _ = refit(subset)
            if best_mse is None or mse < best_mse:
                best_mse, best_keep = mse, subset
        exhaustive = realise(best_keep)

        # P0 ALWAYS KEEP THE PAYLOAD: both candidates materialised real coded
        # bytes; persist BOTH arms, not just the winner, with sha256 + length.
        for label, row in (("exhaustive", exhaustive), ("greedy", greedy)):
            blob = row.pop("payload")
            for key, ext in (("br", "br"), ("raw", "cpr1")):
                data = blob.get(key)
                if not isinstance(data, (bytes, bytearray)):
                    raise SystemExit(
                        f"PAYLOAD MISSING: encoded_bytes()['{key}'] absent for "
                        f"rank {r} {label} — refusing to emit a scalars-only artifact"
                    )
                path = payloads / f"rank{r:02d}_{label}.{ext}"
                path.write_bytes(bytes(data))
                row[f"{key}_payload_path"] = str(path)
                row[f"{key}_payload_sha256"] = hashlib.sha256(bytes(data)).hexdigest()
                row[f"{key}_payload_bytes"] = len(data)

        rows.append(
            {
                "rank": r,
                "greedy": greedy,
                "exhaustive": exhaustive,
                "greedy_keep_is_optimal": greedy["keep"] == exhaustive["keep"],
                "mse_ratio_greedy_over_exhaustive": (
                    greedy["mse_refit_lower_bound_pixel_units"]
                    / exhaustive["mse_refit_lower_bound_pixel_units"]
                ),
                "bytes_saved_greedy": baseline_coded - greedy["coded_bytes"],
                "bytes_saved_exhaustive": baseline_coded - exhaustive["coded_bytes"],
            }
        )
        print(
            f"rank {r:2d}  greedy {greedy['coded_bytes']:>6,} B / "
            f"{100 * greedy['mse_refit_lower_bound_pixel_units'] / signal_energy:5.2f}%   "
            f"exhaustive {exhaustive['coded_bytes']:>6,} B / "
            f"{100 * exhaustive['mse_refit_lower_bound_pixel_units'] / signal_energy:5.2f}%"
            f"   optimal={rows[-1]['greedy_keep_is_optimal']}",
            flush=True,
        )

    n_suboptimal = sum(1 for row in rows if not row["greedy_keep_is_optimal"])
    receipt = {
        "schema": "ra1b_exhaustive_keepset_refit.v1",
        "finding": (
            "ra1's energy-greedy keep set is suboptimal at "
            f"{n_suboptimal} of {len(rows)} ranks; max MSE ratio "
            f"{max(r['mse_ratio_greedy_over_exhaustive'] for r in rows):.3f}x. "
            "P2's 'lower bound on every rank-r carrier' is REFUTED as published: "
            "the refit is optimal GIVEN the keep set, the keep set was a heuristic, "
            "and the full C(12,r) space is exhaustible in seconds."
        ),
        "axis": "[exact coded bytes MEASURED; carrier-field MSE MEASURED; d_pose NOT measured]",
        "score_claim": False,
        "promotable": False,
        "custody": custody,
        "carrier": {
            "signal_energy_pixel_units": signal_energy,
            "baseline_coded_bytes": baseline_coded,
            "carrier_dim": D,
            "subsets_searched": sum(1 for r in range(1, D) for _ in itertools.combinations(range(D), r)),
        },
        "rows": rows,
        "elapsed_s": time.time() - started,
    }
    path = out / "RA1B_EXHAUSTIVE_KEEPSET_REFIT.json"
    path.write_text(json.dumps(receipt, indent=2))
    print(f"\n{n_suboptimal} of {len(rows)} greedy keep sets were SUBOPTIMAL")
    print(f"receipt -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
