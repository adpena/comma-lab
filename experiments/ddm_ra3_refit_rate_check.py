#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-RA3 rate check: the carrier rank-r byte credit, MEASURED through the SHIPPED coder.

WHY THIS EXISTS.  Every carrier rung priced on this base -- ra2c's ladder, jc1's keep-sets,
ra2's subspaces, and ra3's trust-regioned re-fit -- divides its measured pose damage by a byte
credit of ``(12 - r) * 22161/12`` and calls the quotient a verdict.  That credit has never
been measured; it is a uniform-bytes-per-dimension ASSUMPTION.  It is also the DENOMINATOR of
every one of those verdicts, and it is exactly the quantity a re-fit can silently destroy: a
re-fit changes the stored coefficient VALUES, and the shipped coefficient stream is Rice-coded
over zigzagged frame-deltas, so higher-entropy values cost more bits and hand back fewer.

So this tool encodes each candidate through ``carrier_codec`` -- the SHIPPED encoder, the one
whose output the receiver parses -- and reports the real payload sizes.

THE SHIPPED FORMAT, READ AT SOURCE (``carrier_codec``, verified by round-trip below):

  basis         27,648 signed 5-bit codes = 12 atoms x 2,304 (3 x 24 x 32), one float32 scale
                per atom, static canonical Huffman over the 32-symbol alphabet.
  coefficients  600 x 12 signed 12-bit codes, DELTA along the frame axis per dimension, then
                zigzagged, then Rice with one k per dimension chosen by exhaustive minimisation
                (``_encode_rice``), walked dimension-major.

A rank-r carrier stores r atoms and r codes per frame in the same two containers, so both
halves are re-encodable and neither needs to be assumed.

CONTROLS (all three must pass or the tool refuses):
  1. ``encode_compact_carrier`` on the SHIPPED fields reproduces the canonical CPR1 blob
     byte-for-byte -- proves this is the receiver's own coder, not a re-implementation.
  2. The zigzag-delta round trip reproduces the shipped ``encoded`` array exactly -- proves
     the code-building path this tool applies to candidates is the shipped one.
  3. Re-quantizing the SHIPPED coefficients through the tool's own quantizer reproduces the
     shipped codes -- proves the quantizer is the shipped grid, not a nearby one.

NAMED LIMIT, and it is shared by every row in the family rather than introduced here.  The
receiver applies ``normalized_basis`` (bicubic resize, per-atom mean removal, per-atom RMS
normalisation) to the atoms BEFORE the einsum, so an 11-atom container renormalises its own
rotated atoms and does not render exactly what a rank-11-constrained 12-atom render produces.
Every exact ``d_pose`` in this family -- ra2c's, jc1's, ra2's and ra3's -- is measured on the
12-atom render, so the container change is unmeasured for all of them equally.  That is a
FIDELITY caveat; the RATE numbers below are unaffected, because they count codes.

Axis: [macOS-CPU advisory]. score_claim=false, promotable=false. Lossless, no scorer runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RA2B_SOURCE = REPO / "experiments/ddm_ra2b_carrier_chain_control.py"
RA3_SOURCE = REPO / "experiments/ddm_ra3_subspace_trust_region_refit.py"

DEFAULT_ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/archive.zip"
)
JC1_PAYLOAD = Path("/Volumes/APDataStore/pact/ddm_jc1/retained")
DEFAULT_OUT = Path("/Volumes/APDataStore/pact/ddm_ra3")

N_FRAMES = 600
CARRIER_DIM = 12
CARRIER_BYTES_SHIPPED = 22_161            # the Brotli'd body actually in archive.zip
S_PER_BYTE = 25.0 / 37_545_489.0
BASE_D_POSE = 0.00014747
POSE_TERM_T4 = 0.0082945765


class RateRefusal(RuntimeError):
    """Fail-closed refusal: a control failed, or a code left its declared range."""


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


# --------------------------------------------------------------------------- #
# the shipped code-building path, applied to arbitrary stored values
# --------------------------------------------------------------------------- #
def quantize_to_codes(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Signed 12-bit codes on the shipped grid: ``code = round(value / scale)``."""
    codes = np.round(values / scales[None]).astype(np.int64)
    if codes.min() < -0x800 or codes.max() > 0x7FF:
        raise RateRefusal(
            f"coefficient code outside the shipped 12-bit range [{codes.min()}, {codes.max()}]"
        )
    return codes


def codes_to_zigzag_delta(codes: np.ndarray) -> np.ndarray:
    """Frame-axis delta then zigzag, exactly inverting ra2b:141-147's reconstruction."""
    delta = np.diff(codes, axis=0, prepend=np.zeros((1, codes.shape[1]), dtype=codes.dtype))
    delta = ((delta + 0x800) & 0xFFF) - 0x800
    return ((delta << 1) ^ (delta >> 63)).astype(np.int64)


def rice_payload_bits(encoded: np.ndarray, codec) -> tuple[int, list[int]]:
    """Exact Rice payload bits, from the SHIPPED encoder -- not a re-implementation."""
    ks, _payload, bits = codec._encode_rice(np.asarray(encoded, dtype=np.int64))
    return int(bits), [int(k) for k in ks]


def huffman_payload_bits(codes: np.ndarray, codec) -> int:
    """Exact basis payload bits, from the SHIPPED encoder.

    The shipped path zigzags the signed 5-bit codes before Huffman
    (``encode_compact_carrier``: ``_zigzag_signed(basis_codes, BASIS_BITS)``), read at
    source rather than inferred -- an offset mapping would have produced a different and
    silently wrong symbol histogram.
    """
    unsigned = codec._zigzag_signed(np.asarray(codes, dtype=np.int64), codec.BASIS_BITS)
    _lengths, _payload, bits = codec._encode_huffman(unsigned)
    return int(bits)


def quantize_basis(atoms: np.ndarray, codec) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom signed 5-bit quantization, matching the shipped basis container."""
    levels = float(codec.ALPHABET_SIZE // 2 - 1)
    scales = np.abs(atoms).max(axis=1) / levels
    scales[scales <= 0] = 1.0
    codes = np.clip(np.round(atoms / scales[:, None]), -levels, levels).astype(np.int64)
    return codes, scales


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--jc1-payload", type=Path, default=JC1_PAYLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument(
        "--accepted",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_ra3/retained/ra3_r11_accepted.float64.npy"),
        help="the realised-accepted candidate; omitted rows are simply not reported",
    )
    args = parser.parse_args()

    ra2b = _load(RA2B_SOURCE, "ddm_ra2b_chain")
    ra3 = _load(RA3_SOURCE, "ddm_ra3_refit")
    chain = ra2b.load_chain()
    renderer, read_archive, split_selector, materialize = chain[0], chain[1], chain[2], chain[3]
    parts = read_archive(Path(args.archive))
    carrier_blob, _selector = split_selector(parts.carrier_blob)
    canonical = bytes(materialize(carrier_blob, renderer))

    import carrier_codec as codec

    basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
    basis_scales, basis_codes, coeff_scales, encoded = renderer.decode_compact_carrier(
        canonical, basis_count=basis_count, frames=renderer.N, dimensions=renderer.CARRIER_DIM
    )

    controls: dict[str, Any] = {}

    # CONTROL 1 -- this really is the receiver's coder.
    reencoded = codec.encode_compact_carrier(basis_scales, basis_codes, coeff_scales, encoded)
    controls["shipped_reencode_byte_identical"] = bool(reencoded == canonical)

    # CONTROL 2 -- the code-building path inverts the shipped reconstruction.
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    base_codes = np.cumsum(delta, axis=0) & 0xFFF
    base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes)
    controls["zigzag_delta_roundtrip_exact"] = bool(
        np.array_equal(codes_to_zigzag_delta(base_codes), encoded.astype(np.int64))
    )

    # CONTROL 3 -- the quantizer lands on the shipped grid.
    #
    # THE CONTROL EARNED ITS KEEP HERE.  Comparing against the Rice-decoded codes FAILED on
    # exactly 30 of 7,200 coordinates: the receiver applies a 36-byte COMPENSATION OVERLAY
    # to the decoded codes before scaling (ra2b:139-160, chain[4]), so the shipped
    # coefficients are the POST-overlay codes.  Pricing a re-fit against the pre-overlay
    # codes would have mis-stated the grid on 30 coordinates and gone unnoticed.
    apply_overlay = chain[4]
    reference_codes = base_codes.astype(np.int32)
    overlay_changed = 0
    if parts.compensation_blob is not None:
        reference_codes = apply_overlay(reference_codes, parts.compensation_blob)
        overlay_changed = int(np.count_nonzero(reference_codes != base_codes))
    shipped_coeff = np.load(Path(args.jc1_payload) / "coeff.float64.npy")
    controls["quantizer_reproduces_shipped_codes"] = bool(
        np.array_equal(quantize_to_codes(shipped_coeff, coeff_scales.astype(np.float64)),
                       reference_codes.astype(np.int64))
    )
    for name, passed in controls.items():
        if not passed:
            raise RateRefusal(f"control failed: {name}; refusing to report a byte credit")

    shipped_coeff_bits, shipped_ks = rice_payload_bits(encoded.astype(np.int64), codec)
    shipped_basis_bits = huffman_payload_bits(basis_codes, codec)

    # ---- the rank-r containers -----------------------------------------------
    jacobian = np.load(Path(args.jc1_payload) / "jacobian_pose6_x_coeff12.float64.npy")
    geometry = ra3.SubspaceGeometry(shipped_coeff, jacobian, args.rank)

    # The rank-r basis is 11 ROTATED atoms, not 11 of the shipped 12, so its code
    # statistics are measured rather than pro-rated.
    atoms = (basis_codes.reshape(CARRIER_DIM, -1).astype(np.float64)
             * basis_scales.astype(np.float64)[:, None])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rotated = (geometry.v_r.T @ geometry.inverse) @ atoms            # (r, 2304)
    if not np.isfinite(rotated).all():
        raise RateRefusal("rotated basis is non-finite")
    rotated_codes, rotated_scales = quantize_basis(rotated, codec)
    rank_basis_bits = huffman_payload_bits(rotated_codes.reshape(-1), codec)

    # CONTROL 4 -- RECORDED, not refusing, and it is why the credit is a BRACKET.
    # The shipped basis does NOT use the max-abs-to-15 grid this tool applies: three of the
    # twelve atoms cap at |code| = 7, and those three are exactly the lowest-RMS atoms, so
    # the producer allocated per-atom precision by rate-distortion.  That rule is not
    # reconstructible from the archive, so the rotated 11-atom basis is coded on a FINER
    # grid than the shipped one and its measured size is an UPPER bound on what a matched
    # container would cost.  Guessing the rule would be worse than bracketing it.
    shipped_atom_codes, _ = quantize_basis(atoms, codec)
    controls_recorded = {
        "basis_quantizer_reproduces_shipped_codes": bool(
            np.array_equal(shipped_atom_codes, basis_codes.reshape(CARRIER_DIM, -1).astype(np.int64))
        ),
        "shipped_max_abs_code_per_atom":
            np.abs(basis_codes.reshape(CARRIER_DIM, -1)).max(axis=1).astype(int).tolist(),
        "note": "max-abs grid is FINER than the shipped one on the low-RMS atoms, so the "
                "measured rotated-basis size is an upper bound; see credit_bracket",
    }

    stored = {"projection": geometry.z_proj}
    accepted_path = Path(args.accepted)
    if accepted_path.exists():
        accepted_coeff = np.load(accepted_path)
        # Recover the stored r-vector from the accepted 12-dim candidate.
        stored["accepted"] = (accepted_coeff @ geometry.root) @ geometry.v_r

    # A REAL rank-r CPR1 blob, built by the shipped encoder, is the byte credit -- not a
    # sum of estimated fields.  ALWAYS KEEP THE PAYLOAD: each blob is written to disk with
    # its sha256 before any credit derived from it is reported.
    import brotli

    shipped_brotli = min(
        len(brotli.compress(canonical, quality=q)) for q in (10, 11)
    )
    retained_blobs: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for label, z in stored.items():
        z_scales = np.abs(z).max(axis=0) / float(0x7FF)
        z_scales[z_scales <= 0] = 1.0
        codes = quantize_to_codes(z, z_scales)
        encoded_r = codes_to_zigzag_delta(codes)
        bits, ks = rice_payload_bits(encoded_r, codec)
        blob = codec.encode_compact_carrier(
            rotated_scales.astype("<f4"),
            rotated_codes.reshape(-1).astype(np.int64),
            z_scales.astype("<f4"),
            encoded_r,
        )
        blob_path = Path(args.output) / "retained" / f"ra3_cpr1_r{args.rank}_{label}.bin"
        _atomic_bytes(blob_path, bytes(blob))
        retained_blobs[label] = {
            "path": str(blob_path), "bytes": len(blob),
            "sha256": hashlib.sha256(bytes(blob)).hexdigest(),
        }
        blob_brotli = min(len(brotli.compress(bytes(blob), quality=q)) for q in (10, 11))
        raw_credit = len(canonical) - len(blob)
        credit = shipped_brotli - blob_brotli
        # The MOST FAVOURABLE credit the evidence permits: keep the coefficient half
        # MEASURED (that is the half a re-fit can spoil, and it is exactly re-encodable),
        # but grant the basis half ra2's own proportional assumption, which control 4 shows
        # is a lower bound on the rotated container's cost.  The verdict is computed at
        # this end so a refusal cannot be an artifact of a pessimistic container.
        favourable_credit = (
            (shipped_basis_bits + 7) // 8 * (CARRIER_DIM - args.rank) / CARRIER_DIM
            + ((shipped_coeff_bits + 7) // 8 - (bits + 7) // 8)
        )
        rows.append({
            "favourable_credit_bytes": favourable_credit,
            "favourable_advisory_bar": ra3.exact_pose_bar(
                favourable_credit, float(np.sqrt(10 * BASE_D_POSE))),
            "favourable_t4_bar_ratio_transfer": ra3.exact_pose_bar(
                favourable_credit, POSE_TERM_T4),
            "label": label,
            "rank": args.rank,
            "coeff_payload_bytes": (bits + 7) // 8,
            "coeff_rice_k": ks,
            "basis_payload_bytes": (rank_basis_bits + 7) // 8,
            "cpr1_blob_bytes": len(blob),
            "cpr1_blob_brotli_bytes": blob_brotli,
            "raw_credit_bytes": raw_credit,
            "brotli_credit_bytes": credit,
            "advisory_bar": ra3.exact_pose_bar(credit, float(np.sqrt(10 * BASE_D_POSE))),
            "t4_bar_ratio_transfer": ra3.exact_pose_bar(credit, POSE_TERM_T4),
        })

    assumed_credit = (CARRIER_DIM - args.rank) * CARRIER_BYTES_SHIPPED / CARRIER_DIM
    receipt = {
        "arm": "ddm_ra3",
        "schema": "ddm_ra3_refit_rate_check.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "measurement_status": "MEASURED_THROUGH_SHIPPED_CARRIER_CODEC",
        "controls": controls,
        "controls_recorded": controls_recorded,
        "shipped": {
            "canonical_brotli_bytes": shipped_brotli,
            "compensation_overlay_bytes":
                0 if parts.compensation_blob is None else len(parts.compensation_blob),
            "compensation_overlay_changed_codes": overlay_changed,
            "compensation_overlay_note":
                "a separate archive section, unchanged by rank reduction and excluded from "
                "both sides of the credit; MEASURED rate-neutral at the Rice layer "
                "(post-overlay codes re-encode to the same 79,020 bits)",
            "canonical_cpr1_bytes": len(canonical),
            "canonical_cpr1_sha256": hashlib.sha256(canonical).hexdigest(),
            "archive_carrier_bytes": CARRIER_BYTES_SHIPPED,
            "basis_payload_bytes": (shipped_basis_bits + 7) // 8,
            "coeff_payload_bytes": (shipped_coeff_bits + 7) // 8,
            "coeff_rice_k": shipped_ks,
        },
        "assumed_credit_bytes_ra2_convention": assumed_credit,
        "rows": rows,
    }
    _atomic_bytes(
        Path(args.output) / f"RA3_RATE_CHECK_r{args.rank}.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )

    print(f"controls: {controls}")
    print(f"basis quantizer matches shipped grid: "
          f"{controls_recorded['basis_quantizer_reproduces_shipped_codes']} "
          f"(shipped max|code| per atom {controls_recorded['shipped_max_abs_code_per_atom']})")
    print(f"shipped canonical {len(canonical)} B  basis {(shipped_basis_bits + 7) // 8} B  "
          f"coeff {(shipped_coeff_bits + 7) // 8} B")
    print(f"assumed credit (ra2 convention, uniform) {assumed_credit:.1f} B")
    for row in rows:
        print(f"  r={row['rank']} {row['label']:>10s}  basis {row['basis_payload_bytes']:6d} B  "
              f"coeff {row['coeff_payload_bytes']:6d} B  blob {row['cpr1_blob_bytes']:6d} B"
              f"  raw credit {row['raw_credit_bytes']:6d} B  brotli credit "
              f"{row['brotli_credit_bytes']:6d} B\n"
              f"{'':>14s}   credit BRACKET: measured-container {row['brotli_credit_bytes']} B "
              f"(bar {row['advisory_bar']:.4f}x)  ..  most-favourable "
              f"{row['favourable_credit_bytes']:.0f} B (bar {row['favourable_advisory_bar']:.4f}x)"
              f"  ..  ra2-assumed {assumed_credit:.0f} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
