"""ddm_pc3 -- the CAP1 predictor refit: -164 B of carrier at ZERO distortion change.

WHAT THIS IS AND WHERE IT CAME FROM
-----------------------------------
This arm was chartered to price the pose carrier's rate/distortion CURVE.  Its rate leg
(``ddm_pc3_carrier_rate.py``) opens with a mandatory positive control: re-encode the
SHIPPED coefficient codes through the SHIPPED AR(1)+bias predictor and require the Rice
payload back byte-for-byte.  That control passed.  Its sister control -- refit the
predictor exhaustively over the whole legal schema and check it RECOVERS the shipped
parameters -- FAILED, and it failed in the informative direction: the refit is smaller.

    shipped   51,581 Rice bits = 6,448 B   factors [167,178,173,216,205,146,149,146,186,181,233,177]
    refit     50,270 Rice bits = 6,284 B   factors [142,132,125, 94,161,145,141,114,141,120,146,148]
                                     biases shipped [16,12,16,16,-16,-16,15,-15,-16,15,16,16]
                                     biases refit   [ 6, 3, 6,16, -8, -5, 2, -4,-11,15,16, 8]

-164 B of carrier body, and the CODES ARE BIT-IDENTICAL.  The predictor is a pure coding
choice: ``decode_cap1`` inverts it and hands the renderer the same canonical CPR1 bytes
either way (proved here: identical, sha 68e4784c4eef9db2 both sides).  So frame 0 is
bit-identical, d_pose and d_seg are the pointer's BY CONSTRUCTION rather than by
measurement, and the receiver is untouched -- the twelve Q8 factors and twelve biases are
data fields the shipped receiver already reads and range-validates.

This is a point on the arm's own curve, at the far LEFT of it: delta bytes negative,
delta distortion exactly zero.

WHY THE SEARCH IS EXHAUSTIVE AND NOT A HEURISTIC
-------------------------------------------------
The schema is closed and small: ``coefficient_predictor.py`` bounds the Q8 factor to
[-512, 512] and the bias to [-16, 16], so one dimension has 1,025 x 33 = 33,825 legal
models and twelve dimensions are independent under a per-dimension Rice parameter.  The
fit enumerates all of them and keeps the minimum-bit model.  There is no tolerance, no
initialisation and no local optimum: the answer is the schema's own minimum.

THE TWO CONSTRAINTS THE PACKED CONTAINER IMPOSES, CHECKED NOT ASSUMED
---------------------------------------------------------------------
The archive does not store the 36-byte unpacked CAP1 metadata; ``up3.pack_cap1_metadata``
packs the factors as 7-bit offsets from their own minimum and the Rice parameters as ONE
bit over a u8 base.  So a refit is only free if (a) ``max(factors) - min(factors) <= 127``
and (b) ``max(ks) - min(ks) <= 1``.  Measured: the refit spans 67 and 1.  Both hold, and
``build_archive`` fails closed on either.

Axis: exact coder arithmetic plus a real archive build.  ``score_claim=false`` until an
exact eval row exists; the projection below is arithmetic on a MEASURED byte delta and a
distortion pair that is unchanged by construction.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import ddm_pc3_carrier_rate as rate
import ddm_pc3_pose_carrier_curve as pc3

N_PAIRS = 600
CARRIER_DIM = 12


class Pc3RefitError(RuntimeError):
    """A ddm_pc3 refit precondition failed. Fail closed, never approximate."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fit_predictor(codes: np.ndarray, predictor) -> dict[str, Any]:
    """Exhaustive per-dimension (factor, bias) fit over the closed legal schema."""
    factors, biases, ks, bits = [], [], [], 0
    for dim in range(CARRIER_DIM):
        dim_bits, factor, bias, k = rate.fit_dimension(codes[:, dim], predictor)
        bits += dim_bits
        factors.append(int(factor))
        biases.append(int(bias))
        ks.append(int(k))
    return {
        "factors_q8": factors,
        "biases": biases,
        "rice_ks": ks,
        "rice_payload_bits": int(bits),
        "rice_payload_bytes": int((bits + 7) // 8),
        "factor_span": int(max(factors) - min(factors)),
        "k_span": int(max(ks) - min(ks)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=pc3.MOVE44_TREE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--expect-archive-sha256", default=pc3.MOVE44_ARCHIVE_SHA256
    )
    parser.add_argument("--twin", type=int, default=2)
    args = parser.parse_args()

    import ddm_up3_carrier_splice as splice

    runtime = Path(args.runtime)
    base_bytes = (runtime / "archive.zip").read_bytes()
    observed = _sha256(base_bytes)
    if observed != args.expect_archive_sha256:
        raise Pc3RefitError(
            f"archive sha {observed} != expected {args.expect_archive_sha256}"
        )

    (carrier_repack, cap1, predictor, carrier_blob, info, model, codes) = (
        rate.load_shipped(runtime / "archive.zip", runtime)
    )

    # CONTROL 1 -- the shipped model must reproduce the shipped Rice payload exactly.
    frozen = rate.price_codes(
        codes,
        carrier_repack=carrier_repack,
        predictor=predictor,
        model=model,
        refit=False,
    )
    if frozen["rice_payload_bits"] != int(info["rice_payload_bits"]):
        raise Pc3RefitError(
            "the shipped-model re-encode does not reproduce the shipped payload"
        )

    fitted = fit_predictor(codes, predictor)
    if fitted["factor_span"] > 127:
        raise Pc3RefitError(
            f"refit factor span {fitted['factor_span']} exceeds the packed 7-bit field"
        )
    if fitted["k_span"] > 1:
        raise Pc3RefitError(
            f"refit Rice k span {fitted['k_span']} exceeds the packed 1-bit field"
        )

    # CONTROL 2 -- the canonical CPR1 the renderer consumes must be UNCHANGED. This is
    # what makes d_seg and d_pose the pointer's by construction: same bytes in, same
    # frame 0 out, no scorer needed to know it.
    new_model = predictor.Ar1BiasModel(
        np.asarray(fitted["factors_q8"], dtype="<i2"),
        np.asarray(fitted["biases"], dtype="i1"),
    )
    residuals = rate.forward_ar1(codes, new_model, predictor)
    zigzag = carrier_repack._zigzag(residuals)
    ks_encoded, rice_payload, residual_bits = carrier_repack._rice_encode(zigzag, 1)
    header = (
        carrier_blob[:8]
        + int(info["basis_bits"]).to_bytes(3, "little")
        + int(residual_bits).to_bytes(3, "little")
    )
    metadata_bytes = CARRIER_DIM * 3
    fixed_offset = 14 + metadata_bytes
    fixed = carrier_blob[fixed_offset : fixed_offset + 8 * CARRIER_DIM + 32]
    basis_offset = fixed_offset + 8 * CARRIER_DIM + 32 + CARRIER_DIM
    basis = carrier_blob[basis_offset : basis_offset + int(info["basis_bytes"])]
    refit_blob = (
        header
        + np.asarray(fitted["factors_q8"], dtype="<i2").tobytes()
        + np.asarray(fitted["biases"], dtype="i1").tobytes()
        + fixed
        + bytes(np.asarray(ks_encoded, dtype=np.uint8).reshape(-1))
        + basis
        + rice_payload
    )
    canonical_shipped = cap1.decode_cap1(
        carrier_blob, frames=N_PAIRS, dimensions=CARRIER_DIM
    )
    canonical_refit = cap1.decode_cap1(
        refit_blob, frames=N_PAIRS, dimensions=CARRIER_DIM
    )
    if canonical_refit != canonical_shipped:
        raise Pc3RefitError(
            "the refit carrier does not decode to the shipped canonical CPR1; the "
            "renderer would see different bytes and the distortion legs would move"
        )

    # CONTROL 3 -- rebuilding the archive from the body's OWN predictor must reproduce
    # the shipped bytes, or a byte delta cannot be attributed to the refit.
    body = splice.parse_shipped_body(runtime, verify_sha=False)
    identity = splice.build_archive(
        body, np.asarray(body.codes, dtype=np.int32), runtime_dir=runtime,
        container_search=True, verify=True,
    )
    identity_ok = identity["archive_sha256"] == observed
    if not identity_ok:
        raise Pc3RefitError(
            "CONTROL FAILED: rebuilding the body from its own codes and predictor gives "
            f"{identity['archive_sha256']} ({identity['archive_size']} B), not "
            f"{observed} ({len(base_bytes)} B)"
        )

    # The candidate: the SAME codes, a different predictor. ``build_archive`` re-encodes
    # from ``body.factors``/``body.biases``, so replacing those two fields is the whole
    # change, and its own parse-back verify proves the codes survive.
    refit_body = dataclasses.replace(
        body,
        factors=np.asarray(fitted["factors_q8"], dtype=np.int16),
        biases=np.asarray(fitted["biases"], dtype=np.int8),
    )
    twins = []
    for index in range(max(1, args.twin)):
        built = splice.build_archive(
            refit_body,
            np.asarray(body.codes, dtype=np.int32),
            runtime_dir=runtime,
            container_search=True,
            verify=True,
        )
        twins.append(
            {
                "twin": index,
                "archive_size": int(built["archive_size"]),
                "archive_sha256": built["archive_sha256"],
            }
        )
    if len({twin["archive_sha256"] for twin in twins}) != 1:
        raise Pc3RefitError(f"twin encodes disagree: {twins}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.out_dir / "candidate_archive.zip"
    archive_path.write_bytes(built["archive_bytes"])
    (args.out_dir / "refit_cap1_carrier.bin").write_bytes(refit_blob)

    delta_bytes = int(built["archive_size"]) - len(base_bytes)
    delta_score = delta_bytes * pc3.BYTE_TO_SCORE
    payload = {
        "schema": "ddm_pc3_predictor_refit.v1",
        "base": {
            "runtime": str(runtime),
            "archive_sha256": observed,
            "archive_bytes": len(base_bytes),
            "carrier_bytes": int(info["carrier_bytes"]),
            "rice_payload_bytes": int(info["rice_payload_bytes"]),
            "factors_q8": list(info["factors_q8"]),
            "biases": list(info["biases"]),
            "rice_ks": list(info["rice_ks"]),
        },
        "refit": fitted,
        "controls": {
            "shipped_model_reencode_byte_exact": True,
            "canonical_cpr1_identical": True,
            "canonical_cpr1_sha256": _sha256(canonical_refit),
            "archive_identity_rebuild_reproduces_shipped_bytes": identity_ok,
            "twin_encodes_agree": True,
            "packed_factor_span_within_7_bits": fitted["factor_span"] <= 127,
            "packed_k_span_within_1_bit": fitted["k_span"] <= 1,
        },
        "candidate": {
            "archive_path": str(archive_path),
            "archive_sha256": built["archive_sha256"],
            "archive_bytes": int(built["archive_size"]),
            "twins": twins,
        },
        "delta": {
            "archive_delta_bytes": delta_bytes,
            "carrier_rice_delta_bytes": (
                fitted["rice_payload_bytes"] - int(info["rice_payload_bytes"])
            ),
            "delta_score": delta_score,
            "d_seg_delta": 0.0,
            "d_pose_delta": 0.0,
            "why_distortion_cannot_move": (
                "the coefficient codes, the coefficient scales, the basis scales and "
                "the basis payload are bit-identical, and decode_cap1 reconstructs the "
                "byte-identical canonical CPR1; frame 0 is therefore the same image and "
                "the odd-frame sections are untouched"
            ),
            "projected_score": pc3.composed_score(
                pc3.MOVE44_D_SEG_T4,
                pc3.MOVE44_D_POSE_T4,
                int(built["archive_size"]),
            ),
            "pointer_score": pc3.MOVE44_SCORE_T4,
            "net_vs_pointer": (
                pc3.composed_score(
                    pc3.MOVE44_D_SEG_T4,
                    pc3.MOVE44_D_POSE_T4,
                    int(built["archive_size"]),
                )
                - pc3.MOVE44_SCORE_T4
            ),
            "admit_bar": pc3.ADMIT_BAR,
        },
        "axis": "[exact coder arithmetic + real archive build; no exact eval row yet]",
        "score_claim": False,
        "promotable": False,
    }
    (args.out_dir / "PREDICTOR_REFIT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
