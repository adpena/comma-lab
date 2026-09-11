"""ddm_pc3 rate leg -- the EXACT CAP1 carrier cost of a refined coefficient lattice.

The distortion leg of this arm's curve is measured by the renderer and the frozen
scorer.  This is the other leg: what a rung actually COSTS, in the coder that ships,
with a bit-exact positive control before any delta is quoted.

WHY THE COST IS NOT A GUESS
---------------------------
Move 44's carrier is CAP1 (18,915 B in the archive): a fixed 190-byte frame (header 14,
AR(1)+bias predictor metadata 36, twelve basis scales and twelve coefficient scales 96,
Huffman lengths 32, Rice parameters 12), a 12,277-byte basis payload the lattice never
touches, and a 6,448-byte Rice payload over the AR(1) residuals of the 600x12 codes, with
all twelve Rice parameters currently at ``k = 5``.  Halving a dimension's coefficient
scale doubles that dimension's codes, so the naive price is one Rice bit per symbol --
600 bits, 75 B, per dimension.  "Naive" because the AR(1) predictor absorbs part of the
doubling and the ``k`` search may or may not step; the only honest price is the encoder's
own output, which is what this module computes.

THE CONTROLS, IN ORDER
----------------------
1. ``forward_ar1`` on the shipped codes through the shipped model, Rice-encoded, must
   reproduce the shipped payload BYTE-FOR-BYTE and the shipped ``k`` vector.  If it does
   not, nothing below is a price.
2. The predictor REFIT on the shipped codes must recover the shipped
   ``(factors_q8, biases)``.  If it does, the refit is the shipped encoder's fit and a
   refined rung may be priced with its own refit.  If it does not, every rung is priced
   with the shipped model frozen instead, which can only OVERSTATE the rung's cost -- the
   conservative direction for a family this arm expects to refuse.

Axis: exact coder arithmetic, no scorer, no score claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_dx1_dxi_recode_race import forward_ar1, load_shipped

N_PAIRS = 600
CARRIER_DIM = 12
Q8_MIN, Q8_MAX = -512, 512
BIAS_MIN, BIAS_MAX = -16, 16
BYTE_TO_SCORE = 25.0 / 37_545_489.0


class Pc3RateError(RuntimeError):
    """A ddm_pc3 rate precondition failed. Fail closed, never approximate."""


def rice_bits(values: np.ndarray, k: int) -> int:
    quotient = int((np.asarray(values, dtype=np.uint64) >> k).sum())
    return quotient + int(values.size) * (k + 1)


def best_rice(values: np.ndarray) -> tuple[int, int]:
    return min((rice_bits(values, k), k) for k in range(12))


def residuals_for(codes_column: np.ndarray, factor: int, bias: int, predictor):
    """AR(1)+bias residuals for one dimension, in the receiver's exact integer domain."""
    values = np.asarray(codes_column, dtype=np.int32)
    prediction = predictor.signed_mod(
        predictor.round_q8(values[:-1], np.int64(factor)) + bias
    )
    return np.concatenate(
        [values[:1], predictor.signed_mod(values[1:] - prediction)]
    )


def fit_dimension(codes_column: np.ndarray, predictor) -> tuple[int, int, int, int]:
    """Exhaustive (factor, bias) fit for one dimension, minimising Rice bits.

    The schema domain is small and closed -- 1,025 Q8 factors x 33 biases -- so the fit
    is a search over the WHOLE legal set, not a heuristic that could differ from the
    encoder's by an unmeasured amount.
    """
    best = (1 << 60, 0, 0, 0)
    for factor in range(Q8_MIN, Q8_MAX + 1):
        base = predictor.round_q8(
            np.asarray(codes_column[:-1], dtype=np.int32), np.int64(factor)
        )
        for bias in range(BIAS_MIN, BIAS_MAX + 1):
            prediction = predictor.signed_mod(base + bias)
            residual = predictor.signed_mod(
                np.asarray(codes_column[1:], dtype=np.int32) - prediction
            )
            full = np.concatenate([np.asarray(codes_column[:1]), residual])
            zig = ((full.astype(np.int64) << 1) ^ (full.astype(np.int64) >> 63)) & 0xFFF
            bits, k = best_rice(zig)
            if bits < best[0]:
                best = (int(bits), int(factor), int(bias), int(k))
    return best


def price_codes(
    codes: np.ndarray,
    *,
    carrier_repack,
    predictor,
    model,
    refit: bool,
) -> dict[str, Any]:
    """Exact Rice payload bits for a code lattice, with or without a predictor refit."""
    codes = np.asarray(codes, dtype=np.int32)
    if codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Pc3RateError(f"codes have shape {codes.shape}, expected (600, 12)")
    if np.abs(codes).max() > 2047:
        raise Pc3RateError(
            f"code magnitude {int(np.abs(codes).max())} leaves the signed int12 "
            "container; this rung does not fit the shipped format"
        )
    if refit:
        total_bits = 0
        factors, biases, ks = [], [], []
        for dim in range(CARRIER_DIM):
            bits, factor, bias, k = fit_dimension(codes[:, dim], predictor)
            total_bits += bits
            factors.append(factor)
            biases.append(bias)
            ks.append(k)
        return {
            "rice_payload_bits": int(total_bits),
            "rice_payload_bytes": int((total_bits + 7) // 8),
            "factors_q8": factors,
            "biases": biases,
            "rice_ks": ks,
            "predictor": "refit",
        }
    residuals = forward_ar1(codes, model, predictor)
    zig = carrier_repack._zigzag(residuals)
    ks, _payload, bits = carrier_repack._rice_encode(zig, 1)
    return {
        "rice_payload_bits": int(bits),
        "rice_payload_bytes": int((int(bits) + 7) // 8),
        "factors_q8": np.asarray(model.factors_q8).tolist(),
        "biases": np.asarray(model.biases).tolist(),
        "rice_ks": ks.reshape(-1).tolist(),
        "predictor": "shipped_frozen",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--halvings", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    (carrier_repack, _cap1, predictor, carrier_blob, info, model, codes) = load_shipped(
        args.archive, args.runtime
    )

    # Control 1: the shipped codes through the shipped model must reproduce the shipped
    # payload byte-for-byte. A price from an encoder that cannot reproduce what shipped
    # is not a price.
    frozen = price_codes(
        codes,
        carrier_repack=carrier_repack,
        predictor=predictor,
        model=model,
        refit=False,
    )
    control_one = bool(
        frozen["rice_payload_bits"] == int(info["rice_payload_bits"])
        and frozen["rice_ks"] == list(info["rice_ks"])
    )
    if not control_one:
        raise Pc3RateError(
            "the shipped-model re-encode does not reproduce the shipped Rice payload: "
            f"{frozen['rice_payload_bits']} bits / ks {frozen['rice_ks']} vs shipped "
            f"{info['rice_payload_bits']} / {info['rice_ks']}"
        )

    # Control 2: does the exhaustive refit recover the shipped predictor?
    refit = price_codes(
        codes,
        carrier_repack=carrier_repack,
        predictor=predictor,
        model=model,
        refit=True,
    )
    control_two = bool(
        refit["factors_q8"] == list(info["factors_q8"])
        and refit["biases"] == list(info["biases"])
    )

    scales_are_free = (
        "the twelve coefficient scales are already twelve float32 words in the "
        "archive; changing their VALUES changes no length, so a lattice rung's entire "
        "cost is the Rice payload delta"
    )
    shipped_rice = int(info["rice_payload_bytes"])
    rungs = []
    menu: list[tuple[str, np.ndarray]] = []
    for halving in args.halvings:
        menu.append(
            (f"global_div{2 ** halving}", np.full(CARRIER_DIM, 2.0**halving))
        )
    for dim in range(CARRIER_DIM):
        multipliers = np.ones(CARRIER_DIM, dtype=np.float64)
        multipliers[dim] = 2.0
        menu.append((f"dim{dim}_div2", multipliers))

    for label, multipliers in menu:
        # Refining the lattice by 1/f multiplies every code by f EXACTLY: the rung's
        # codes are the shipped point re-expressed, so this is the rate of the rung's
        # CONTAINER, before any re-solve moves the codes. A re-solve moves them by a
        # fraction of a step, which cannot change the Rice parameter.
        scaled = np.rint(codes.astype(np.float64) * multipliers[None]).astype(np.int64)
        if np.abs(scaled).max() > 2047:
            rungs.append(
                {
                    "rung": label,
                    "refused": "code magnitude leaves the signed int12 container",
                    "max_abs_code": int(np.abs(scaled).max()),
                }
            )
            continue
        priced = price_codes(
            scaled.astype(np.int32),
            carrier_repack=carrier_repack,
            predictor=predictor,
            model=model,
            refit=control_two,
        )
        delta_bytes = priced["rice_payload_bytes"] - shipped_rice
        rungs.append(
            {
                "rung": label,
                "max_abs_code": int(np.abs(scaled).max()),
                "rice_payload_bytes": priced["rice_payload_bytes"],
                "delta_bytes": int(delta_bytes),
                "delta_score": float(delta_bytes) * BYTE_TO_SCORE,
                "rice_ks": priced["rice_ks"],
                "predictor": priced["predictor"],
            }
        )

    payload = {
        "schema": "ddm_pc3_carrier_rate.v1",
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "shipped_carrier": {
            key: info[key]
            for key in (
                "carrier_bytes",
                "basis_bytes",
                "rice_payload_bytes",
                "rice_payload_bits",
                "rice_ks",
                "factors_q8",
                "biases",
            )
        },
        "control_shipped_reencode_is_byte_exact": control_one,
        "control_refit_recovers_shipped_predictor": control_two,
        "predictor_policy": (
            "refit" if control_two else "shipped_model_frozen (conservative: a frozen "
            "model can only OVERSTATE a refined rung's cost)"
        ),
        "why_only_the_rice_payload_moves": scales_are_free,
        "rungs": rungs,
        "byte_to_score": BYTE_TO_SCORE,
        "axis": "[exact coder arithmetic, no scorer]",
        "score_claim": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
