#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""t1h -- exact byte price of moving a shipped carrier code, in the real CAP1 container.

The headroom sweep (``ddm_t1h_pose_coeff_headroom.py``) measures what a code move buys on
the pose axis.  It buys nothing unless the move is affordable, and "zero added bytes" is a
claim about the CONTAINER, not about the code lattice.  This tool supplies the other half:
the exact, measured byte cost of any candidate lattice inside the shipped CAP1 carrier.

WHY THIS IS EXACTLY COMPUTABLE
------------------------------
CAP1 (``runtime/entropy/coefficient_ar1_codec.py``) lays out a fixed header, fixed AR(1)
metadata, fixed fp64 scales, a fixed basis-length table, one Rice parameter per dimension,
the basis bitstream, and finally the Rice-coded coefficient residuals.  Every field except
the Rice residual payload is INDEPENDENT of the coefficient codes.  So

    carrier_bytes(codes) = fixed_prefix_bytes + ceil(rice_bits(codes) / 8)

and ``rice_bits`` is produced by the receiver's own ``_rice_encode``.  Nothing is estimated.

THE PREDICTOR IS AR(1), WHICH BOUNDS THE BLAST RADIUS
-----------------------------------------------------
``restore_ar1_bias`` predicts frame i from frame i-1 only.  Inverting it, residual i depends
on codes i and i-1, so changing one code perturbs exactly TWO Rice symbols (i and i+1).
A code move is therefore a local, cheap-to-price edit rather than a re-encode of the stream.

THE CONTROL.  This tool re-encodes the SHIPPED codes and requires that it reproduce the
shipped Rice bit count and the shipped per-dimension Rice parameters exactly.  If it cannot
reproduce the container it is pricing, every byte number it prints would be fiction, so a
mismatch aborts.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: The shipped receiver tree.  Pricing must use the code that actually decodes the archive.
RUNTIME = Path(
    "/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime"
)
ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/retained/archive.zip"
)
OUT = Path("/Volumes/APDataStore/pact/ddm_t1h")

N_PAIRS = 600
CARRIER_DIM = 12
UNCOMPRESSED_BYTES = 37_545_489
SEED = 1234


def load_runtime(runtime: Path):
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from runtime import carrier_repack
    from runtime.entropy import coefficient_ar1_codec as cap1
    from runtime.entropy import coefficient_predictor as predictor
    from runtime.residual_archive import read_residual_archive

    return carrier_repack, cap1, predictor, read_residual_archive


def forward_ar1(codes: np.ndarray, model, predictor) -> np.ndarray:
    """Invert ``restore_ar1_bias``: coefficient codes -> CAP1 predictor residuals.

    ``restore_ar1_bias`` computes
        out[0]   = res[0]
        out[i]   = signed_mod(signed_mod(round_q8(out[i-1], f) + b) + res[i])
    so the encoder is the same recurrence solved for ``res``.
    """
    values = np.asarray(codes, dtype=np.int32)
    residuals = np.empty_like(values)
    residuals[0] = values[0]
    factors = np.asarray(model.factors_q8, dtype=np.int16)
    biases = np.asarray(model.biases, dtype=np.int16)
    for frame in range(1, values.shape[0]):
        prediction = predictor.signed_mod(
            predictor.round_q8(values[frame - 1], factors) + biases
        )
        residuals[frame] = predictor.signed_mod(values[frame] - prediction)
    return residuals


def rice_bits(codes: np.ndarray, model, carrier_repack, predictor) -> tuple[np.ndarray, int]:
    """Exact CAP1 Rice parameters and bit count for a candidate code lattice."""
    residuals = forward_ar1(codes, model, predictor)
    ks, _payload, bits = carrier_repack._rice_encode(
        carrier_repack._zigzag(residuals), 1
    )
    return ks.reshape(-1), int(bits)


def load_shipped(archive: Path, runtime: Path):
    """Return the shipped carrier blob, its CAP1 fields, and the base code lattice."""
    carrier_repack, cap1, predictor, read_residual_archive = load_runtime(runtime)
    parts = read_residual_archive(archive)
    carrier_blob, selector_blob = carrier_repack.split_frame0_selector_carrier(
        parts.carrier_blob
    )
    if not carrier_blob.startswith(cap1.CAP1_MAGIC):
        raise SystemExit(
            "the shipped carrier is not CAP1; this pricer models the CAP1 container only"
        )
    info = cap1.inspect_cap1(carrier_blob, frames=N_PAIRS, dimensions=CARRIER_DIM)
    model = predictor.unpack_ar1_bias_metadata(
        carrier_blob[cap1._HEADER_BYTES : cap1._HEADER_BYTES + CARRIER_DIM * 3],
        CARRIER_DIM,
    )
    canonical = cap1.decode_cap1(carrier_blob, frames=N_PAIRS, dimensions=CARRIER_DIM)

    # Rebuild the signed-int12 base lattice exactly as f26_inflate:468-470 does.
    offset = 4 + 8 + 8 * CARRIER_DIM + 32
    basis_bits, residual_bits = struct.unpack_from("<II", canonical, 4)
    basis_bytes = (basis_bits + 7) // 8
    ks_canon = np.frombuffer(canonical[offset : offset + CARRIER_DIM], dtype=np.uint8)
    offset += CARRIER_DIM + basis_bytes
    encoded = carrier_repack._rice_decode(
        ks_canon.reshape(CARRIER_DIM, 1).astype(np.int64),
        canonical[offset:],
        residual_bits,
        N_PAIRS,
        CARRIER_DIM,
    )
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    base_codes = np.cumsum(delta, axis=0) & 0xFFF
    base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes)
    return (
        carrier_repack, cap1, predictor, carrier_blob, selector_blob,
        info, model, base_codes.astype(np.int32),
        parts.compensation_blob,
    )


def run(args) -> int:
    rng = np.random.default_rng(SEED)
    (carrier_repack, cap1, predictor, carrier_blob, selector_blob, info, model,
     base_codes, compensation_blob) = load_shipped(Path(args.archive), Path(args.runtime))

    shipped_bits = int(info["rice_payload_bits"])
    shipped_ks = np.asarray(info["rice_ks"], dtype=np.uint8)
    fixed_prefix = len(carrier_blob) - int(info["rice_payload_bytes"])

    ks, bits = rice_bits(base_codes, model, carrier_repack, predictor)
    control = {
        "rice_bits_reencoded": bits,
        "rice_bits_shipped": shipped_bits,
        "rice_bits_match": bits == shipped_bits,
        "rice_ks_match": bool(np.array_equal(ks.astype(np.uint8), shipped_ks)),
        "carrier_bytes_shipped": len(carrier_blob),
        "carrier_bytes_reencoded": fixed_prefix + (bits + 7) // 8,
    }
    if not (control["rice_bits_match"] and control["rice_ks_match"]):
        raise SystemExit(
            "CONTROL FAILED: the pricer cannot reproduce the shipped CAP1 Rice stream.\n"
            f"  {json.dumps(control, indent=2)}\n"
            "Every byte number this tool could print would be fiction.  Refusing."
        )

    # Marginal price of a single +/-1 code move, over a seeded random sample of positions.
    samples = []
    for _ in range(args.samples):
        pair = int(rng.integers(0, N_PAIRS))
        coord = int(rng.integers(0, CARRIER_DIM))
        step = int(rng.choice([-1, 1]))
        trial = base_codes.copy()
        value = int(trial[pair, coord]) + step
        if not -2048 <= value <= 2047:
            continue
        trial[pair, coord] = value
        _, moved_bits = rice_bits(trial, model, carrier_repack, predictor)
        samples.append({
            "pair": pair, "coord": coord, "step": step,
            "delta_bits": moved_bits - shipped_bits,
        })

    delta_bits = np.array([s["delta_bits"] for s in samples], dtype=np.float64)
    receipt = {
        "schema": "ddm_t1h_carrier_byte_pricer.v1",
        "axis": "[exact local byte arithmetic, no scorer]",
        "score_claim": False,
        "archive": str(args.archive),
        "carrier": {
            "carrier_blob_bytes": len(carrier_blob),
            "selector_bytes": 0 if selector_blob is None else len(selector_blob),
            "compensation_bytes": 0 if compensation_blob is None else len(compensation_blob),
            "fixed_prefix_bytes": fixed_prefix,
            "rice_payload_bytes": int(info["rice_payload_bytes"]),
            "rice_ks": info["rice_ks"],
        },
        "control": control,
        "single_move_price": {
            "samples": len(samples),
            "delta_bits_mean": float(delta_bits.mean()) if len(samples) else None,
            "delta_bits_median": float(np.median(delta_bits)) if len(samples) else None,
            "delta_bits_min": float(delta_bits.min()) if len(samples) else None,
            "delta_bits_max": float(delta_bits.max()) if len(samples) else None,
            "delta_bits_p90": float(np.percentile(delta_bits, 90)) if len(samples) else None,
            "fraction_free_or_negative": (
                float((delta_bits <= 0).mean()) if len(samples) else None
            ),
            "s_per_byte": 25.0 / UNCOMPRESSED_BYTES,
            "s_per_bit_mean_move": (
                float(delta_bits.mean() / 8.0 * 25.0 / UNCOMPRESSED_BYTES)
                if len(samples) else None
            ),
        },
        "rows": samples,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / (args.receipt or "T1H_BYTE_PRICE.json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    summary = {k: v for k, v in receipt.items() if k != "rows"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--receipt", type=str, default=None)
    parser.add_argument("--samples", type=int, default=240)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
