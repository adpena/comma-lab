#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""t1h -- compose the measured per-pair moves into ONE candidate lattice and price it.

The headroom sweep evaluates, for every pair, each single-coordinate integer move against
the EXACT chain and the frozen CPU-torch PoseNet.  Because pair i's pose energy depends on
pair i's codes alone (the odd frame is renderer output and carries no carrier code), the
per-pair winners compose into a single candidate WITHOUT any further scorer work: the
candidate's ``d_pose`` is the mean of the per-pair energies already measured, exactly.

This tool therefore does not estimate.  It:

1. rebuilds the shipped signed-int12 base lattice from the archive;
2. folds each pair's measured winning move into that lattice;
3. prices the result through the receiver's own CAP1 encoder, exactly;
4. reports the advisory-axis ``dS``, split into its pose and rate halves.

WHERE THE MOVE IS APPLIED, AND WHY IT IS THE BASE LATTICE.  The receiver computes
``codes = base_codes + compensation_overlay`` (``f26_inflate.py:468-475``).  The sweep
measured moves relative to that SUM.  Folding the move into ``base_codes`` and leaving the
36 B overlay untouched reproduces exactly the lattice that was measured, and keeps the
overlay section byte-identical, so no counted section grows by construction.

THIS IS NOT A SCORE.  The pose half is CPU-torch advisory; only ``upstream/evaluate.py`` on
contest hardware produces a score.  The rate half IS exact -- bytes are bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PRICER_SOURCE = REPO / "experiments/ddm_t1h_carrier_byte_pricer.py"

OUT = Path("/Volumes/APDataStore/pact/ddm_t1h")
SWEEP = OUT / "T1H_HEADROOM_N600.json"

N_PAIRS = 600
CARRIER_DIM = 12
POSE_ROWS = 6
UNCOMPRESSED_BYTES = 37_545_489
T4_D_POSE = 6.88e-6
T4_D_SEG = 2.9613e-4
T4_ARCHIVE_BYTES = 181_161
T4_SCORE = 0.15853325034789678


def load_pricer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("t1h_pricer", PRICER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["t1h_pricer"] = module
    spec.loader.exec_module(module)
    return module


def counted_carrier_bytes(archive: Path, runtime: Path, new_rice_payload: bytes,
                          new_rice_bits: int, new_ks: np.ndarray) -> dict:
    """Price the candidate in the bytes the CONTEST actually counts.

    The rate term reads ``archive.zip``.  Inside it the carrier travels as
    ``brotli(packed_CAP1 + overlay)`` (``residual_archive.py`` + the ps1u writer), so raw
    CAP1 length is NOT the counted quantity -- the brotli-compressed stream is.  The packed
    section keeps the basis bitstream and the Rice payload verbatim in its tail (verified:
    ``canonical[182:] == packed[142:]``), so a candidate body is built by patching the
    canonical ``residual_bits`` header field, the packed Rice-parameter field, and the Rice
    payload, then re-compressing with the shipped brotli parameters.

    The packed Rice-parameter field stores ``k_base`` plus ONE bit per dimension, so the
    container can only express two adjacent k values.  A candidate whose optimal k escapes
    that pair cannot be packed without a format change, and this reports that rather than
    silently mis-pricing it.
    """
    import zipfile as zf

    import brotli

    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from runtime import residual_archive as ra

    member = zf.ZipFile(archive).read("p")
    header = ra.RX1_MODEL_HEADER
    magic, version, codec, table_mode, reserved, hpac_b, sem_b, car_b = header.unpack_from(
        member
    )
    offset = header.size + hpac_b + sem_b
    stream = member[offset : offset + car_b]
    body = brotli.decompress(stream)
    if brotli.compress(body, quality=11, lgwin=24) != stream:
        raise SystemExit(
            "brotli(quality=11, lgwin=24) does not reproduce the shipped carrier stream; "
            "the counted price cannot be proven under these parameters"
        )
    packed_len = ra.PACKED_CAP1_SECTION_BYTES
    packed, overlay = body[:packed_len], body[packed_len:]

    k_base = packed[139]
    offsets = np.asarray(new_ks, dtype=np.int64) - int(k_base)
    if np.any(offsets < 0) or np.any(offsets > 1):
        return {
            "packable": False,
            "blocker": (
                f"candidate Rice parameters {np.asarray(new_ks).tolist()} escape the packed "
                f"container's two-value window [{k_base}, {k_base + 1}]"
            ),
        }
    # ``_unpack_unsigned`` reads LSB-first within each byte and REFUSES nonzero padding, so
    # the field is packed little-bit-order and then round-tripped through the receiver's own
    # reader.  Constructing this by eye is exactly how the first attempt produced a section
    # the receiver rejected.
    bits = np.zeros(16, dtype=np.uint8)
    bits[: offsets.size] = offsets.astype(np.uint8)
    ks_field = np.packbits(bits, bitorder="little")[:2].tobytes()
    if not np.array_equal(
        ra._unpack_unsigned(ks_field, CARRIER_DIM, 1), offsets.astype(np.int16)
    ):
        raise SystemExit(
            "Rice-parameter field does not round-trip through the receiver's own reader"
        )

    # The stored section is the canonical CAP1 blob from offset 8 onward with its metadata
    # bit-packed (``_restore_cap1`` re-adds the 8-byte prefix), so ``packed[3:6]`` is the
    # Rice bit count and the Rice payload is the section's final field.  Patch exactly those
    # three places and nothing else -- an earlier version recomputed the basis length from a
    # header field it had mislocated and silently duplicated the payload.
    old_rice_bits = int.from_bytes(packed[3:6], "little")
    old_rice_bytes = (old_rice_bits + 7) // 8
    basis_bytes = (int.from_bytes(packed[0:3], "little") + 7) // 8
    rice_start = 142 + basis_bytes
    rice_end = rice_start + old_rice_bytes
    # The Rice payload is NOT the last field: the frame-0 selector's suffix trails it
    # (``residual_archive.py:206-208`` splits the selector off the restored carrier body).
    # Patching from the end of the section would overwrite that suffix and produce an
    # archive whose selector header no longer parses -- which is exactly what happened.
    selector_suffix_bytes = len(packed) - rice_end
    if selector_suffix_bytes < 0:
        raise SystemExit("packed CAP1 section is shorter than its own fields")

    def assemble(bits_value: int, ks_bytes: bytes, payload: bytes) -> bytes:
        buffer = bytearray(packed)
        buffer[3:6] = int(bits_value).to_bytes(3, "little")
        buffer[140:142] = ks_bytes
        buffer[rice_start:rice_end] = payload
        return bytes(buffer)

    # IDENTITY CONTROL: reassembling from the SHIPPED fields must reproduce the shipped
    # section byte for byte, or the price describes a body the receiver never sees.
    if assemble(old_rice_bits, packed[140:142], packed[rice_start:rice_end]) != packed:
        raise SystemExit(
            "PACKED REASSEMBLY CONTROL FAILED: rebuilding the shipped CAP1 section from its "
            "own fields does not reproduce it; the counted price would be fiction"
        )

    # The reader dispatches on an EXACT section length (``PACKED_CAP1_SECTION_BYTES``), so a
    # candidate whose Rice payload rounds to a different byte count cannot be carried at all.
    if len(new_rice_payload) != old_rice_bytes:
        return {
            "packable": False,
            "blocker": (
                f"candidate Rice payload is {len(new_rice_payload)} B but the packed section "
                f"is dispatched on an exact length requiring {old_rice_bytes} B"
            ),
        }

    new_packed = assemble(new_rice_bits, ks_field, new_rice_payload)
    new_body = new_packed + overlay
    new_stream = brotli.compress(new_body, quality=11, lgwin=24)
    return {
        "packable": True,
        "new_stream": new_stream,
        "member": member,
        "stream_offset": offset,
        "stream_bytes_before": car_b,
        "packed_bytes_before": packed_len,
        "packed_bytes_after": len(new_packed),
        "carrier_body_bytes_before": len(body),
        "carrier_body_bytes_after": len(new_body),
        "counted_stream_bytes_before": len(stream),
        "counted_stream_bytes_after": len(new_stream),
        "counted_delta_bytes": len(new_stream) - len(stream),
        "overlay_bytes_unchanged": len(overlay),
        "rice_k_base": int(k_base),
        "rice_ks_after": np.asarray(new_ks).tolist(),
        "rice_payload_offset_in_packed_section": rice_start,
        "rice_payload_bytes": old_rice_bytes,
        "basis_bytes_preserved": basis_bytes,
        "selector_suffix_bytes_preserved": selector_suffix_bytes,
    }


def packed_rice_bit_budget(archive: Path, runtime: Path) -> dict:
    """Read the HARD Rice bit budget the packed CAP1 section can carry.

    The receiver dispatches on an EXACT section length, so the Rice payload field inside it
    has a fixed byte width.  That width -- not a compression ratio, not a heuristic -- is the
    ceiling on the residual bitstream, and a composition that exceeds it cannot be carried at
    all.  Reading it from the shipped archive rather than hardcoding it keeps the budget
    bound to the container actually being modified.
    """
    import zipfile as zf

    import brotli

    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from runtime import residual_archive as ra

    member = zf.ZipFile(archive).read("p")
    fields = ra.RX1_MODEL_HEADER.unpack_from(member)
    start = ra.RX1_MODEL_HEADER.size + fields[5] + fields[6]
    body = brotli.decompress(member[start : start + fields[7]])
    packed = body[: ra.PACKED_CAP1_SECTION_BYTES]
    shipped_rice_bits = int.from_bytes(packed[3:6], "little")
    rice_payload_bytes = (shipped_rice_bits + 7) // 8
    return {
        "rice_payload_bytes": rice_payload_bytes,
        "max_rice_bits": rice_payload_bytes * 8,
        "shipped_rice_bits": shipped_rice_bits,
        "shipped_slack_bits": rice_payload_bytes * 8 - shipped_rice_bits,
    }


def fit_to_bit_budget(
    candidate: np.ndarray,
    choice: list[tuple[int, int, float]],
    rows: list[dict],
    base_energies: np.ndarray,
    encode_bits,
    max_bits: int,
) -> tuple[np.ndarray, list[tuple[int, int, float]], int, list[dict]]:
    """Trade the least measured pose gain per bit until the composition fits the container.

    Every alternative considered here is a move the sweep ALREADY evaluated exactly, so the
    resulting energy is measured, never modelled: ``per_coord[c]`` holds the exact energy of
    that pair's best move on coordinate ``c``, and pairs are independent, so substituting one
    pair's choice changes that pair's energy and nothing else.

    The search is exhaustive over SINGLE substitutions: every pair, every measured
    per-coordinate option, plus reverting to no move.  It picks the substitution minimising
    ``energy increase / bits saved`` -- a negative score (better pose AND fewer bits) is taken
    first.  Joint substitutions are not searched, so the result is a good feasible point, not
    a proven optimum, and the receipt says so.

    ONE INTERACTION WITH ``--min-gain-frac`` WORTH STATING.  The repair scans every pair,
    including pairs the threshold left unmoved, so it may re-admit a move the threshold
    excluded -- but only when that move both saves a bit and lowers the measured energy, i.e.
    only when it is better on BOTH axes.  Every such re-admission is listed in the receipt's
    ``substitutions`` with a negative ``energy_cost``, so it is visible rather than silent.
    """
    # Copy rather than mutate: this function also RETURNS the choice, so a caller that kept a
    # reference to the pre-repair selection would otherwise find it silently rewritten.
    choice = list(choice)
    bits, _payload, _ks = encode_bits(candidate)
    log: list[dict] = []
    while bits > max_bits:
        best = None
        for pair, (coord, delta, energy) in enumerate(choice):
            options = [(-1, 0, float(base_energies[pair]))]
            for entry in rows[pair]["per_coord"]:
                if entry["best_delta"] != 0:
                    options.append(
                        (int(entry["coord"]), int(entry["best_delta"]), float(entry["energy"]))
                    )
            for alt_coord, alt_delta, alt_energy in options:
                if alt_coord == coord and alt_delta == delta:
                    continue
                trial = candidate.copy()
                if coord >= 0:
                    trial[pair, coord] -= delta
                if alt_coord >= 0:
                    trial[pair, alt_coord] += alt_delta
                    if not -2048 <= int(trial[pair, alt_coord]) <= 2047:
                        continue
                trial_bits, _, _ = encode_bits(trial)
                saved = bits - trial_bits
                if saved <= 0:
                    continue
                score = (alt_energy - energy) / saved
                if best is None or score < best[0]:
                    best = (score, pair, alt_coord, alt_delta, alt_energy, trial, trial_bits,
                            saved, energy)
        if best is None:
            raise SystemExit(
                f"CONTAINER INFEASIBLE: the composition needs {bits} Rice bits but the packed "
                f"section carries at most {max_bits}, and no single measured substitution "
                "saves a bit.  A smaller move set or a container format change is required."
            )
        score, pair, alt_coord, alt_delta, alt_energy, trial, trial_bits, saved, was = best
        candidate = trial
        choice[pair] = (alt_coord, alt_delta, alt_energy)
        log.append({
            "pair": pair,
            "substituted_to_coord": alt_coord,
            "substituted_to_delta": alt_delta,
            "bits_saved": int(saved),
            "energy_before": was,
            "energy_after": alt_energy,
            "energy_cost": alt_energy - was,
            "energy_cost_per_bit_saved": score,
        })
        bits = trial_bits
    return candidate, choice, bits, log


def run(args) -> int:
    pricer = load_pricer()
    (carrier_repack, cap1, predictor, carrier_blob, selector_blob, info, model,
     base_codes, compensation_blob) = pricer.load_shipped(
        Path(args.archive), Path(args.runtime)
    )
    shipped_bits = int(info["rice_payload_bits"])
    fixed_prefix = len(carrier_blob) - int(info["rice_payload_bytes"])

    sweep = json.loads(Path(args.sweep).read_text())
    rows = sweep["rows"]
    if len(rows) != N_PAIRS:
        raise SystemExit(
            f"the sweep covers {len(rows)} pairs; composing a shippable candidate needs "
            f"all {N_PAIRS}"
        )
    if sweep["control"]["failures"]:
        raise SystemExit(
            "the sweep recorded byte-identity control failures; its energies describe a "
            "chain that is not the shipped chain.  Refusing to compose."
        )

    base_energies = np.array([r["base_energy"] for r in rows], dtype=np.float64)
    swept_best = np.array([r["best_energy"] for r in rows], dtype=np.float64)
    absolute_gain = base_energies - swept_best
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_gain = np.where(base_energies > 0, absolute_gain / base_energies, 0.0)

    # The winner's-curse guard.  Moves are chosen by argmin on the CPU-torch instrument and
    # will be scored on CUDA.  Taking the best of ~48 candidates per pair captures some of
    # whatever the two instruments disagree about, so the SMALLEST wins are the least likely
    # to survive the change of instrument.  ``--min-gain-frac`` keeps only pairs whose
    # measured relative gain clears a threshold, trading a little headroom for transfer
    # robustness; the concentration curve says how much that trade actually costs.
    order = np.argsort(-absolute_gain)
    total_gain = float(absolute_gain.sum())
    concentration = {
        f"top_{count}_moves_share_of_total_gain": (
            float(absolute_gain[order[:count]].sum() / total_gain)
            if total_gain > 0 else None
        )
        for count in (10, 25, 50, 100, 200, 300, 600)
        if count <= len(rows)
    }

    # A later pass folds its moves into the PREVIOUS pass's base lattice, not the shipped
    # one, so the passes accumulate instead of overwriting each other.
    if args.base_codes:
        start = np.load(Path(args.base_codes)).astype(np.int32)
        if start.shape != (N_PAIRS, CARRIER_DIM):
            raise SystemExit(f"base codes have shape {start.shape}, expected (600, 12)")
        base_codes = start

    # ``fit_to_bit_budget`` indexes ``rows`` by pair id, so prove that ordering rather than
    # trusting it: a shuffled sweep would silently substitute one pair's move onto another.
    if [row["pair"] for row in rows] != list(range(N_PAIRS)):
        raise SystemExit("sweep rows are not ordered by pair id; refusing to compose")

    # The per-pair choice, held explicitly as (coord, delta, measured energy).  ``coord < 0``
    # means no move.  Holding it this way lets the container-fit repair substitute a pair's
    # choice and keep the candidate's energy exact.
    choice = []
    for index, row in enumerate(rows):
        if (row["best_coord"] < 0 or row["best_delta"] == 0
                or relative_gain[index] < args.min_gain_frac):
            choice.append((-1, 0, float(base_energies[index])))
        else:
            choice.append(
                (int(row["best_coord"]), int(row["best_delta"]), float(swept_best[index]))
            )

    candidate = base_codes.copy()
    for pair, (coord, delta, _energy) in enumerate(choice):
        if coord < 0:
            continue
        value = int(candidate[pair, coord]) + delta
        if not -2048 <= value <= 2047:
            raise SystemExit(f"move on pair {pair} leaves signed-int12")
        candidate[pair, coord] = value

    def encode_bits(codes):
        residuals = pricer.forward_ar1(codes, model, predictor)
        ks, payload, bits = carrier_repack._rice_encode(carrier_repack._zigzag(residuals), 1)
        return int(bits), payload, ks.reshape(-1)

    budget = packed_rice_bit_budget(Path(args.archive), Path(args.runtime))
    bits_before_repair, _, _ = encode_bits(candidate)
    repair_log: list[dict] = []
    if args.fit_packed_container:
        candidate, choice, _bits, repair_log = fit_to_bit_budget(
            candidate, choice, rows, base_energies, encode_bits, budget["max_rice_bits"]
        )

    applied = sum(1 for coord, _delta, _energy in choice if coord >= 0)

    # The candidate's exact energy: the measured energy of whichever move each pair ended up
    # with.  Every term here was evaluated on the exact chain; none is modelled.
    realized = np.array([energy for _coord, _delta, energy in choice], dtype=np.float64)

    moved_bits, moved_payload, moved_ks = encode_bits(candidate)
    carrier_bytes_before = len(carrier_blob)
    carrier_bytes_after = fixed_prefix + (moved_bits + 7) // 8

    counted = counted_carrier_bytes(
        Path(args.archive), Path(args.runtime), moved_payload, int(moved_bits), moved_ks
    )
    if not counted["packable"]:
        raise SystemExit(
            "COUNTED PRICE UNAVAILABLE: " + counted["blocker"] + "\n"
            "Refusing to report a rate delta the shipped container cannot carry."
        )
    delta_bytes = counted["counted_delta_bytes"]

    d_pose_before = float(base_energies.mean() / POSE_ROWS)
    d_pose_after = float(realized.mean() / POSE_ROWS)
    d_pose_all_moves = float(swept_best.mean() / POSE_ROWS)
    ratio = d_pose_after / d_pose_before

    pose_term_advisory_before = float(np.sqrt(10.0 * d_pose_before))
    pose_term_advisory_after = float(np.sqrt(10.0 * d_pose_after))
    rate_delta = 25.0 * delta_bytes / UNCOMPRESSED_BYTES

    t4_pose_before = float(np.sqrt(10.0 * T4_D_POSE))
    t4_pose_after = float(np.sqrt(10.0 * T4_D_POSE * ratio))

    receipt = {
        "schema": "ddm_t1h_compose_pass1.v1",
        "axis": "[macOS-CPU advisory pose, EXACT local byte arithmetic]",
        "score_claim": False,
        "promotable": False,
        "moves_applied": applied,
        "moves_available": N_PAIRS,
        "pairs_unimproved": N_PAIRS - applied,
        "carrier_raw_cap1": {
            "bytes_before": carrier_bytes_before,
            "bytes_after": carrier_bytes_after,
            "rice_bits_before": shipped_bits,
            "rice_bits_after": int(moved_bits),
            "note": "raw CAP1 length; NOT the counted quantity",
        },
        "carrier_counted_in_archive": {
            key: value for key, value in counted.items()
            if not isinstance(value, (bytes, bytearray))
        },
        "selector_bytes_unchanged": 0 if selector_blob is None else len(selector_blob),
        "compensation_bytes_unchanged": (
            0 if compensation_blob is None else len(compensation_blob)
        ),
        "min_gain_frac": args.min_gain_frac,
        "packed_container_fit": {
            "enabled": bool(args.fit_packed_container),
            "max_rice_bits": budget["max_rice_bits"],
            "shipped_rice_bits": budget["shipped_rice_bits"],
            "shipped_slack_bits": budget["shipped_slack_bits"],
            "rice_bits_before_repair": bits_before_repair,
            "rice_bits_after_repair": int(moved_bits),
            "overflow_bits_before_repair": max(
                0, bits_before_repair - budget["max_rice_bits"]
            ),
            "substitutions": repair_log,
            "gain_retained_fraction_vs_unconstrained": (
                float((base_energies.sum() - realized.sum())
                      / (base_energies.sum() - swept_best.sum()))
                if base_energies.sum() > swept_best.sum() else None
            ),
            "gain_retained_note": (
                "fraction of THIS SWEEP's total measured pose gain that survives BOTH the "
                "--min-gain-frac filter AND the container repair, against the unconstrained "
                "argmin selection.  It is not the repair's cost alone."
            ),
            "search_scope": (
                "exhaustive over SINGLE substitutions (every pair x every measured "
                "per-coordinate option, plus reverting to no move); joint substitutions are "
                "not searched, so this is a good feasible point, not a proven optimum"
            ),
        },
        "gain_concentration": concentration,
        "pose_advisory": {
            "d_pose_before": d_pose_before,
            "d_pose_after": d_pose_after,
            "d_pose_if_all_moves_kept": d_pose_all_moves,
            "ratio": ratio,
            "pose_term_before": pose_term_advisory_before,
            "pose_term_after": pose_term_advisory_after,
            "delta_pose_term": pose_term_advisory_after - pose_term_advisory_before,
        },
        "delta_S_advisory_axis": (
            pose_term_advisory_after - pose_term_advisory_before + rate_delta
        ),
        "t4_ratio_transfer": {
            "assumption": (
                "the CPU-torch pose RATIO transfers to the contest-CUDA axis.  This is an "
                "assumption about how the two instruments relate under the same edit; it "
                "is not measured here and only a T4 row can settle it."
            ),
            "t4_pose_term_before": t4_pose_before,
            "t4_pose_term_after": t4_pose_after,
            "delta_S_pose": t4_pose_after - t4_pose_before,
            "delta_S_rate": rate_delta,
            "delta_S_total": t4_pose_after - t4_pose_before + rate_delta,
            "projected_S": T4_SCORE + (t4_pose_after - t4_pose_before) + rate_delta,
            "d_seg_unchanged": T4_D_SEG,
            "archive_bytes_after": T4_ARCHIVE_BYTES + delta_bytes,
        },
        "falsifiers_for_the_T4_row": [
            "d_seg MUST be unchanged to all printed digits: no move touches the odd frame, "
            "and SegNet reads only the last frame of the pair.",
            f"archive bytes MUST be {T4_ARCHIVE_BYTES + delta_bytes} exactly.",
            "d_pose MUST fall.  If it rises, the CPU-torch accept oracle does not select "
            "moves that survive the CUDA instrument, and the whole approach is refuted.",
        ],
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "candidate_base_codes.int32.npy", candidate)
    np.save(out / "shipped_base_codes.int32.npy", base_codes)
    # The sweep operates on the RECEIVER lattice (base + overlay), so a later pass must be
    # seeded with that sum, not with the base the carrier actually encodes.
    from runtime.compensation_overlay import apply_compensation_overlay

    receiver_codes = (
        apply_compensation_overlay(candidate, compensation_blob)
        if compensation_blob is not None else candidate
    )
    np.save(out / "candidate_receiver_codes.int32.npy", receiver_codes)
    (out / (args.receipt or "T1H_PASS1_COMPOSE.json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=pricer_default())
    parser.add_argument("--runtime", type=Path, default=runtime_default())
    parser.add_argument("--sweep", type=Path, default=SWEEP)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--receipt", type=str, default=None)
    parser.add_argument(
        "--base-codes", type=Path, default=None,
        help="(600, 12) int32 base lattice to fold moves into, for passes after the first",
    )
    parser.add_argument(
        "--min-gain-frac", type=float, default=0.0,
        help="keep only moves whose measured relative pose gain clears this fraction",
    )
    parser.add_argument(
        "--fit-packed-container", action="store_true",
        help=(
            "if the composition overflows the packed CAP1 section's exact Rice payload "
            "width, substitute the cheapest measured moves until it fits"
        ),
    )
    return run(parser.parse_args())


def pricer_default() -> Path:
    return Path(
        "/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/retained/archive.zip"
    )


def runtime_default() -> Path:
    return Path("/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime")


if __name__ == "__main__":
    raise SystemExit(main())
