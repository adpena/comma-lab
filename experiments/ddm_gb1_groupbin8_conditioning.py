#!/usr/bin/env python3
"""ddm_gb1 - realize ``ddm_mi1``'s ONE free positive: decode-scan conditioning.

WHAT MI1 FOUND, AND WHAT THIS ARM DOES WITH IT.  ``ddm_mi1`` measured that the
shipped DX2 probability model's confidence does not fully account for how much of
its own causal context has been decoded yet: it is over-cautious early in the
within-tile scan and over-confident late.  Eight bins of the within-tile decode
group index (``groupbin8``) recovered **+64.20 B held-out** as a per-cell log-odds
offset on the final coding row.  This arm (a) RE-DERIVES that number against its
own base rather than inheriting it, (b) prices it in the coder's OWN integer
frequency units instead of a float ``-log2 p`` ledger, and (c) realizes it as a
shipped mixer member so the number is a real stream, not a model ledger.

THE CAUSALITY ARGUMENT, VERIFIED AT SOURCE.  The shipped group plan is
``grid = columns + HPAC_DELTA * rows`` over a ``HPAC_PATCH = 64`` tile with
``HPAC_DELTA = 2``, enumerated ``for group in range((1 + HPAC_DELTA) * HPAC_PATCH
- HPAC_DELTA)`` = ``range(190)`` (``cpr1/inflate.py:33-34, 275-287``).  The decoder
walks those 190 masks in increasing ``group``, so

    g(x, y) = (x mod 64) + 2 * (y mod 64)          in 0 .. 189

is the INDEX OF THE DECODE STEP CURRENTLY BEING TAKEN.  It is not a property of
the symbol; it is a property of the position, and the decoder selects the position
before it decodes the symbol there.  So the feature is causally available BY
CONSTRUCTION -- there is no ordering hazard to check, only an identity to verify,
and ``stage_verify`` verifies it against the retained ``group_index.u8`` rather
than trusting this docstring.

WHY THE DISTORTION IS ZERO BY CONSTRUCTION, AND HOW IT IS PROVEN.  A probability
model feeds the range coder; the coder emits the transmitted symbol whatever the
model said.  Changing the model changes BITS, never SYMBOLS.  ``stage_fit`` never
touches a token.  ``stage_encode`` re-encodes the SAME retained token field and
its receipt carries ``tokens_changed`` and the reconstructed-field digest, so the
identity is PROVEN by digest and not asserted.

WHY THE MODEL COSTS ZERO STORED BYTES, AND WHY THAT IS NOT A LOOPHOLE.  The
mechanism is an online Krichevsky-Trofimov counter over a context both sides
derive from already-decoded state.  Nothing is transmitted and nothing learned is
shipped, so the archive carries no table (rule 118 clean).  Its real costs are
(i) the warm-up, which is charged automatically because the encoder and the
receiver run the SAME cold-start trajectory and the emitted stream pays for every
mis-estimate along it, and (ii) decode wall-clock.  Both are measured, not waived.

NO SCORER, NO MODAL, NO METAL, NO TRAINING.  Every input is a retained field from
a prior arm, consumed by sha256, and the run fails closed on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# --- custody ---------------------------------------------------------------

VERTIGO = Path("/Volumes/VertigoDataTier/pact")
APSTORE = Path("/Volumes/APDataStore/pact")

RUNTIME_ROOT = APSTORE / "ddm_dx2" / "r7" / "candidate_runtime_dx2"

TO2_TOKENS = (
    VERTIGO
    / "ddm_to2_token_ordering_race"
    / "measurement_v1"
    / "retained"
    / "input"
    / "dx2_tokens_decoded.u8"
)
DF1_FIELDS = APSTORE / "ddm_df1_dddb_field" / "measurement_v1" / "retained" / "fields"
DF1_ARGMAX = DF1_FIELDS / "position_coding_argmax.u8.bin"
DF1_PMAX = DF1_FIELDS / "position_coding_pmax.f32le.bin"
DF1_RC64 = DF1_FIELDS / "position_rc64_frequency_cost_bits.f64le.bin"
HM1_GROUP_INDEX = APSTORE / "ddm_hm1_20260816" / "retained" / "group_index.u8"

OUT_ROOT = APSTORE / "ddm_gb1_groupbin8_conditioning" / "measurement_v1"

# Digests every consumer must agree on.  A mismatch is a REFUSAL, never a warning:
# a field from a different body would silently reprice a different object.
EXPECTED = {
    "tokens_sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",  # gitleaks:allow -- public content digest
    "argmax_sha256": "db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e",  # gitleaks:allow -- public content digest
    "pmax_sha256": "f37e3d8a21d02647437bf950d7a8a75b751c2a9644c7b8ad48aca2833be4794b",  # gitleaks:allow -- public content digest
    "rc64_sha256": "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86",  # gitleaks:allow -- public content digest
}

#: Prior-arm constants this run must REPRODUCE, not inherit.  Each is re-derived
#: from the retained fields and compared; a disagreement fails the stage.
PRIOR = {
    "flips": 227_671,  # ddm_hc1, ddm_df1, ddm_mi1
    "saturated": 67_955_679,  # ddm_df1, ddm_mi1
    "indicator_bytes": 111_275.62229665744,  # ddm_hc1, ddm_mi1
    "wrong_branch_bytes": 76_601.5389368755,  # ddm_hc1, ddm_mi1
    "right_branch_bytes": 34_674.083359781944,  # ddm_hc1, ddm_mi1
    "dc1_ledger_bits": 910_209.4321425341,  # ddm_bl1 EXPECTED, ddm_fx5 encode receipt
    "stream_bytes": 113_777,  # ddm_bl1 EXPECTED, ddm_rr9, ddm_fx5 encode receipt
    "mi1_groupbin8_heldout_bytes": 64.20,  # ddm_mi1 sec.5 -- the number under test
}

N = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
NUM_CLASSES = 5

# Shipped group plan, transcribed from cpr1/inflate.py:33-34,275-287 and VERIFIED
# against the retained group_index.u8 in stage_verify.
HPAC_PATCH = 64
HPAC_DELTA = 2
GROUPS = (1 + HPAC_DELTA) * HPAC_PATCH - HPAC_DELTA  # 190
GROUP_BINS = 8

TOTAL_FREQUENCY = 1 << 31
"""RC64 denominator (``experiments/ddm_bl1_per_position_bit_allocation.py:79``)."""

PROB_EPS = 1e-12
NEWTON_STEPS = 40
NEWTON_CLIP = 4.0
CHUNK = 8_000_000


class Gb1Error(RuntimeError):
    """A custody, control, or internal-consistency refusal."""


# --- small helpers ---------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def check_custody() -> dict[str, str]:
    """Refuse unless every input field is the exact object the priors were taken on."""
    seen = {}
    for key, path in (
        ("tokens_sha256", TO2_TOKENS),
        ("argmax_sha256", DF1_ARGMAX),
        ("pmax_sha256", DF1_PMAX),
        ("rc64_sha256", DF1_RC64),
    ):
        if not path.is_file():
            raise Gb1Error(f"missing retained input {path}")
        digest = sha256_file(path)
        if digest != EXPECTED[key]:
            raise Gb1Error(f"{key} mismatch at {path}: {digest} != {EXPECTED[key]}")
        seen[key] = digest
    return seen


def group_in_tile_plane() -> np.ndarray:
    """``g(x, y) = (x mod 64) + 2 * (y mod 64)`` over one plane, raster order."""
    rows = np.arange(HEIGHT, dtype=np.int64).reshape(HEIGHT, 1)
    columns = np.arange(WIDTH, dtype=np.int64).reshape(1, WIDTH)
    grid = (columns % HPAC_PATCH) + HPAC_DELTA * (rows % HPAC_PATCH)
    return np.broadcast_to(grid, (HEIGHT, WIDTH)).reshape(-1).copy()


def groupbin8_plane() -> np.ndarray:
    """Eight equal-width bins of the 190-step decode index, mi1's definition."""
    return ((group_in_tile_plane() * GROUP_BINS) // GROUPS).astype(np.int8)


# --- stage: verify ---------------------------------------------------------


def stage_verify(out: Path) -> dict[str, Any]:
    """Re-derive every prior-arm constant, and verify the decode-order identity."""
    started = time.perf_counter()
    custody = check_custody()

    # (1) The causality identity, checked against the SHIPPED retained field
    # rather than against this module's transcription of the source.
    if not HM1_GROUP_INDEX.is_file():
        raise Gb1Error(f"missing group index {HM1_GROUP_INDEX}")
    shipped_group = np.fromfile(HM1_GROUP_INDEX, dtype=np.uint8).astype(np.int64)
    if shipped_group.size != PLANE:
        raise Gb1Error(f"group index must cover the plane, got {shipped_group.size}")
    derived_group = group_in_tile_plane()
    group_mismatches = int(np.count_nonzero(shipped_group != derived_group))
    if group_mismatches:
        raise Gb1Error(
            f"decode-order identity FAILED at {group_mismatches} of {PLANE} positions"
        )

    binning = groupbin8_plane()
    per_bin_plane = np.bincount(binning.astype(np.int64), minlength=GROUP_BINS).tolist()

    # (2) Re-derive the base, streaming so peak RSS stays small.
    tokens = np.memmap(TO2_TOKENS, dtype=np.uint8, mode="r")
    argmax = np.memmap(DF1_ARGMAX, dtype=np.uint8, mode="r")
    pmax = np.memmap(DF1_PMAX, dtype="<f4", mode="r")
    rc64 = np.memmap(DF1_RC64, dtype="<f8", mode="r")
    for name, array, expected in (
        ("tokens", tokens, POSITIONS),
        ("argmax", argmax, POSITIONS),
        ("pmax", pmax, POSITIONS),
        ("rc64", rc64, POSITIONS),
    ):
        if array.size != expected:
            raise Gb1Error(f"{name} has {array.size} entries, expected {expected}")

    flips = 0
    saturated = 0
    right_bits = 0.0
    wrong_bits = 0.0
    rc64_bits = 0.0
    live = 0
    for start in range(0, POSITIONS, CHUNK):
        stop = min(start + CHUNK, POSITIONS)
        p = np.asarray(pmax[start:stop], dtype=np.float64)
        flip = np.asarray(tokens[start:stop]) != np.asarray(argmax[start:stop])
        q = np.clip(1.0 - p, 0.0, 1.0)
        sat = q <= 0.0
        flips += int(np.count_nonzero(flip))
        saturated += int(np.count_nonzero(sat))
        live += int(np.count_nonzero(~sat))
        safe_q = np.maximum(q, 1e-300)
        wrong_bits += float(-np.log2(safe_q[flip]).sum())
        right_bits += float(-np.log2(np.maximum(p[~flip], 1e-300)).sum())
        rc64_bits += float(np.asarray(rc64[start:stop], dtype=np.float64).sum())

    indicator_bytes = (right_bits + wrong_bits) / 8.0
    controls = {
        "flips": {"measured": flips, "prior": PRIOR["flips"], "exact": flips == PRIOR["flips"]},
        "saturated": {
            "measured": saturated,
            "prior": PRIOR["saturated"],
            "exact": saturated == PRIOR["saturated"],
        },
        "indicator_bytes": {
            "measured": indicator_bytes,
            "prior": PRIOR["indicator_bytes"],
            "abs_delta": abs(indicator_bytes - PRIOR["indicator_bytes"]),
        },
        "wrong_branch_bytes": {
            "measured": wrong_bits / 8.0,
            "prior": PRIOR["wrong_branch_bytes"],
            "abs_delta": abs(wrong_bits / 8.0 - PRIOR["wrong_branch_bytes"]),
        },
        "right_branch_bytes": {
            "measured": right_bits / 8.0,
            "prior": PRIOR["right_branch_bytes"],
            "abs_delta": abs(right_bits / 8.0 - PRIOR["right_branch_bytes"]),
        },
    }
    for key in ("flips", "saturated"):
        if not controls[key]["exact"]:
            raise Gb1Error(f"control {key} does not reproduce the prior arms")
    for key in ("indicator_bytes", "wrong_branch_bytes", "right_branch_bytes"):
        if controls[key]["abs_delta"] > 1e-6:
            raise Gb1Error(f"control {key} drifted by {controls[key]['abs_delta']} B")

    receipt = {
        "stage": "verify",
        "axis": "[macOS-CPU advisory / scorer-free retained-field instrumentation]",
        "score_claim": False,
        "promotable": False,
        "custody": custody,
        "decode_order_identity": {
            "law": "g(x, y) = (x mod 64) + 2 * (y mod 64)",
            "source": "cpr1/inflate.py:33-34,275-287 (HPAC_PATCH=64, HPAC_DELTA=2)",
            "verified_against": str(HM1_GROUP_INDEX),
            "mismatches": group_mismatches,
            "groups": int(derived_group.max()) + 1,
            "positions_per_groupbin8_plane": per_bin_plane,
        },
        "controls": controls,
        "rc64_frequency_cost": {
            "bits": rc64_bits,
            "bytes": rc64_bits / 8.0,
            "float_ledger_bits_prior": PRIOR["dc1_ledger_bits"],
            "float_minus_rc64_bits": PRIOR["dc1_ledger_bits"] - rc64_bits,
            "physical_stream_bytes_prior": PRIOR["stream_bytes"],
            "rc64_minus_physical_bytes": rc64_bits / 8.0 - PRIOR["stream_bytes"],
        },
        "live_positions": live,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(out / "VERIFY.json", receipt)
    return receipt


# --- the offset fit --------------------------------------------------------


def _load_live() -> dict[str, np.ndarray]:
    """Live (non-saturated) positions with everything the fit and the pricing need."""
    tokens = np.memmap(TO2_TOKENS, dtype=np.uint8, mode="r")
    argmax = np.memmap(DF1_ARGMAX, dtype=np.uint8, mode="r")
    pmax = np.memmap(DF1_PMAX, dtype="<f4", mode="r")
    rc64 = np.memmap(DF1_RC64, dtype="<f8", mode="r")
    plane_bin = groupbin8_plane()

    q_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    f_parts: list[np.ndarray] = []
    for start in range(0, POSITIONS, CHUNK):
        stop = min(start + CHUNK, POSITIONS)
        p = np.asarray(pmax[start:stop], dtype=np.float64)
        q = 1.0 - p
        keep = q > 0.0
        if not keep.any():
            continue
        flip = (np.asarray(tokens[start:stop]) != np.asarray(argmax[start:stop]))[keep]
        cells = plane_bin[np.arange(start, stop) % PLANE][keep]
        q_parts.append(q[keep])
        y_parts.append(flip.astype(np.int8))
        c_parts.append(cells)
        f_parts.append(np.asarray(rc64[start:stop], dtype=np.float64)[keep])
    return {
        "q": np.concatenate(q_parts),
        "y": np.concatenate(y_parts),
        "cell": np.concatenate(c_parts),
        "rc64_bits": np.concatenate(f_parts),
    }


def _fit_offsets(
    q: np.ndarray, y: np.ndarray, cell: np.ndarray, n_cells: int
) -> np.ndarray:
    """Newton fit of one log-odds offset per cell.  ``beta = 0`` nests the base."""
    logit = np.log(np.clip(q, PROB_EPS, 1.0 - PROB_EPS))
    logit -= np.log1p(-np.clip(q, PROB_EPS, 1.0 - PROB_EPS))
    target = y.astype(np.float64)
    beta = np.zeros(n_cells, dtype=np.float64)
    for _ in range(NEWTON_STEPS):
        adjusted = 1.0 / (1.0 + np.exp(-(logit + beta[cell])))
        gradient = np.bincount(cell, weights=adjusted - target, minlength=n_cells)
        curvature = np.bincount(
            cell, weights=adjusted * (1.0 - adjusted), minlength=n_cells
        )
        step = np.where(curvature > 0.0, gradient / np.maximum(curvature, 1e-30), 0.0)
        np.clip(step, -NEWTON_CLIP, NEWTON_CLIP, out=step)
        beta -= step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta


def _ledger_bits(q: np.ndarray, y: np.ndarray) -> float:
    """Indicator code length in bits under the float ``-log2 p`` ledger."""
    safe = np.clip(q, PROB_EPS, 1.0 - PROB_EPS)
    return float(
        -(np.where(y.astype(bool), np.log2(safe), np.log2(1.0 - safe))).sum()
    )


def _rc64_bits(base_bits: np.ndarray, y: np.ndarray, ratio: np.ndarray) -> float:
    """Cost in the coder's OWN integer frequency units after a binary reweighting.

    The retained field is ``31 - log2(freq_selected)`` for the shipped law, so the
    shipped integer frequency is recovered EXACTLY as ``2 ** (31 - bits)`` (it is an
    integer below ``2**31``, well inside a float64 mantissa).  A binary reweighting
    of ``q -> q'`` scales the mass of every non-argmax class by ``ratio = q'/q`` and
    leaves the conditional distribution among them untouched, so:

    * on a FLIP the selected class is one of those, and its frequency scales;
    * on a HIT the selected class is the balanced winner, whose frequency is
      ``TOTAL - sum(others)``, and it is the OTHERS that scale.

    The encoder floors every class frequency at 1, so the four non-winner classes
    sum to at least ``NUM_CLASSES - 1``; that floor is applied here rather than
    ignored, which is what keeps the estimate conservative in the confident tail
    where most positions live.  ``ratio = 1`` reproduces the base to the last bit,
    and ``stage_fit`` asserts exactly that.
    """
    freq = np.exp2(31.0 - base_bits)
    hit = ~y.astype(bool)
    scaled = np.empty_like(freq)

    flip_freq = np.rint(freq[~hit] * ratio[~hit])
    scaled[~hit] = np.maximum(flip_freq, 1.0)

    others = float(TOTAL_FREQUENCY) - freq[hit]
    others_new = np.maximum(np.rint(others * ratio[hit]), float(NUM_CLASSES - 1))
    scaled[hit] = np.minimum(
        float(TOTAL_FREQUENCY) - others_new, float(TOTAL_FREQUENCY - 1)
    )

    return float((31.0 - np.log2(np.maximum(scaled, 1.0))).sum())


def _cross_fit(
    data: dict[str, np.ndarray], n_cells: int, seed: int
) -> dict[str, Any]:
    """Two-fold cross-fit over a SEEDED RANDOM split -- never a prefix.

    A prefix of this field is a different population (the prefix-bias law: the
    first frames are a distinct scene block), so a temporal split would measure
    scene change rather than mechanism.
    """
    q = data["q"]
    y = data["y"]
    cell = data["cell"].astype(np.int64)
    base_bits = data["rc64_bits"]

    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, size=q.size, dtype=np.int8)

    ledger_gain = 0.0
    rc64_gain = 0.0
    in_sample_gain = 0.0
    betas: list[list[float]] = []
    for held in (0, 1):
        train = fold != held
        test = ~train
        beta = _fit_offsets(q[train], y[train], cell[train], n_cells)
        betas.append([float(v) for v in beta])

        for mask, is_test in ((test, True), (train, False)):
            adjusted = 1.0 / (
                1.0
                + np.exp(
                    -(
                        np.log(np.clip(q[mask], PROB_EPS, 1.0 - PROB_EPS))
                        - np.log1p(-np.clip(q[mask], PROB_EPS, 1.0 - PROB_EPS))
                        + beta[cell[mask]]
                    )
                )
            )
            delta = _ledger_bits(q[mask], y[mask]) - _ledger_bits(adjusted, y[mask])
            if is_test:
                ledger_gain += delta
                ratio = adjusted / np.clip(q[mask], PROB_EPS, 1.0)
                base_rc64 = float(base_bits[mask].sum())
                rc64_gain += base_rc64 - _rc64_bits(base_bits[mask], y[mask], ratio)
            else:
                in_sample_gain += delta

    return {
        "seed": seed,
        "cells": n_cells,
        "heldout_ledger_bytes": ledger_gain / 8.0,
        "heldout_rc64_bytes": rc64_gain / 8.0,
        "in_sample_ledger_bytes": in_sample_gain / 8.0,
        "max_abs_beta": max(abs(v) for row in betas for v in row),
        "betas_per_fold": betas,
    }


def stage_fit(out: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    """Re-derive mi1's groupbin8 gain, in BOTH the float ledger and real coder bits."""
    started = time.perf_counter()
    custody = check_custody()
    data = _load_live()

    # Nesting control: ratio == 1 must reproduce the retained RC64 field exactly.
    ones = np.ones(1_000_000, dtype=np.float64)
    head = {k: v[:1_000_000] for k, v in data.items()}
    nested = _rc64_bits(head["rc64_bits"], head["y"], ones)
    nesting_delta = nested - float(head["rc64_bits"].sum())
    if abs(nesting_delta) > 1e-6:
        raise Gb1Error(f"RC64 nesting control failed by {nesting_delta} bits")

    # KEEP THE PAYLOAD.  The per-cell table IS the mechanism -- it is what says the
    # model is over-cautious early in the scan and over-confident late -- so it is
    # retained alongside the aggregate rather than reduced to one number.
    cell = data["cell"].astype(np.int64)
    populations = np.bincount(cell, minlength=GROUP_BINS)
    cell_flips = np.bincount(cell, weights=data["y"].astype(np.float64), minlength=GROUP_BINS)
    predicted_mass = np.bincount(cell, weights=data["q"], minlength=GROUP_BINS)
    cell_table = [
        {
            "groupbin8": int(index),
            "live_positions": int(populations[index]),
            "flips": int(cell_flips[index]),
            "observed_flip_rate": float(cell_flips[index] / max(populations[index], 1)),
            "model_predicted_flip_rate": float(
                predicted_mass[index] / max(populations[index], 1)
            ),
        }
        for index in range(GROUP_BINS)
    ]

    rows = [_cross_fit(data, GROUP_BINS, seed) for seed in seeds]
    ledger = [r["heldout_ledger_bytes"] for r in rows]
    rc64 = [r["heldout_rc64_bytes"] for r in rows]

    # NEGATIVE CONTROL.  Re-label every position with a cell drawn independently of
    # its position.  An 8-cell offset table fitted on 25M points and scored on the
    # other 25M can always find SOMETHING in-sample; if it also finds bytes
    # held-out on labels that carry no information, the 63 B above is split noise
    # rather than mechanism.  The null must return approximately zero.
    null_data = dict(data)
    null_rng = np.random.default_rng(0xC0FFEE)
    null_data["cell"] = null_rng.integers(
        0, GROUP_BINS, size=data["q"].size, dtype=np.int8
    )
    null_row = _cross_fit(null_data, GROUP_BINS, seeds[0])
    del null_data

    receipt = {
        "stage": "fit",
        "axis": "[macOS-CPU advisory / scorer-free retained-field instrumentation]",
        "score_claim": False,
        "promotable": False,
        "custody": custody,
        "context": "groupbin8 = (((x mod 64) + 2 * (y mod 64)) * 8) // 190",
        "live_positions": int(data["q"].size),
        "nesting_control_bits": nesting_delta,
        "rows": rows,
        "heldout_ledger_bytes": {
            "mean": float(np.mean(ledger)),
            "min": float(np.min(ledger)),
            "max": float(np.max(ledger)),
            "spread": float(np.max(ledger) - np.min(ledger)),
        },
        "heldout_rc64_bytes": {
            "mean": float(np.mean(rc64)),
            "min": float(np.min(rc64)),
            "max": float(np.max(rc64)),
            "spread": float(np.max(rc64) - np.min(rc64)),
        },
        "mi1_prior_heldout_bytes": PRIOR["mi1_groupbin8_heldout_bytes"],
        "null_control_shuffled_cells": null_row,
        "cell_table": cell_table,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(out / f"FIT_seeds_{'_'.join(str(s) for s in seeds)}.json", receipt)
    return receipt


# --- stage: patch a candidate runtime --------------------------------------

FEATURE_PATCH_ANCHOR = '            "homog": homog,\n'
FEATURE_PATCH = (
    '            "homog": homog,\n'
    "            # ddm_gb1: the index of the decode step being taken, binned.\n"
    "            # g(x, y) = (x mod 64) + 2 * (y mod 64) is the shipped group plan\n"
    "            # (cpr1/inflate.py:33-34,275-287); the decoder selects the position\n"
    "            # before it decodes the symbol there, so this is causal by\n"
    "            # construction and costs zero transmitted bytes.\n"
    '            "groupbin8": (\n'
    "                (((flat % WIDTH) % 64) + 2 * ((flat // WIDTH) % 64)) * 8\n"
    "            )\n"
    "            // 190,\n"
)

SPEC_PATCH_ANCHOR = "    specs.update(\n"
SPEC_PATCH = """    def groupbin8_only(f):
        return f["groupbin8"]

    def cls_groupbin8(f):
        return f["cls"] * GROUP_BINS + f["groupbin8"]

    def groupbin8_surprise(f):
        return (f["cls"] * GROUP_BINS + f["groupbin8"]) * U_BINS + f["ubin"]

    specs.update(
        {
            "groupbin8_only": (GROUP_BINS, groupbin8_only),
            "cls_groupbin8": (NUM_CLASSES * GROUP_BINS, cls_groupbin8),
            "groupbin8_surprise": (
                NUM_CLASSES * GROUP_BINS * U_BINS,
                groupbin8_surprise,
            ),
        }
    )

    specs.update(
"""

CONST_PATCH_ANCHOR = "SPATIAL4_LEVELS = 6\n"
CONST_PATCH = (
    "GROUP_BINS = 8\n"
    '"""ddm_gb1: bins of the 190-step within-tile decode index.  Eight, because\n'
    "``ddm_mi1`` measured the exact 190-level index OVERFITTING (51.25 B held-out\n"
    "against 104.37 in-sample) while eight bins held 64.20 / 65.77.\"\"\"\n\n"
    "SPATIAL4_LEVELS = 6\n"
)


def patch_runtime(destination: Path, member: str) -> dict[str, Any]:
    """Copy the shipped runtime and install the groupbin8 member.

    The native C corrector compiles the member list in as constants and REFUSES on
    config drift (``runtime/native_free_corrector.py``), so a patched Python config
    falls back to the Python corrector rather than desynchronising.  That refusal is
    the correct behaviour here and it is why shipping this member needs a C port,
    which this arm names as a blocker rather than hiding.
    """
    # A destructive rmtree needs a structural guard, not care.  The destination must
    # be one of THIS arm's own build directories or the copy is refused.
    if APSTORE not in destination.parents or not destination.name.startswith("runtime_"):
        raise Gb1Error(
            f"refusing to build into {destination}: the destination must live under "
            f"{APSTORE} and be named runtime_<member>"
        )
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(RUNTIME_ROOT, destination, symlinks=True)
    for cache in destination.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    target = destination / "runtime" / "fx2_model_axis_corrector.py"
    text = target.read_text()
    # The last entry of the frozen SHIPPED_CONFIG member tuple; extending it here
    # is what makes the new member LIVE rather than merely available in the pool.
    member_anchor = '        "spatial4_surprise_fast256",\n'
    for anchor, replacement in (
        (CONST_PATCH_ANCHOR, CONST_PATCH),
        (FEATURE_PATCH_ANCHOR, FEATURE_PATCH),
        (SPEC_PATCH_ANCHOR, SPEC_PATCH),
        (member_anchor, member_anchor + f'        "{member}",\n'),
    ):
        if text.count(anchor) != 1:
            raise Gb1Error(f"patch anchor is not unique in {target}: {anchor!r}")
        text = text.replace(anchor, replacement, 1)
    target.write_text(text)

    return {
        "runtime": str(destination),
        "member": member,
        "patched_file_sha256": sha256_file(target),
        "base_runtime": str(RUNTIME_ROOT),
    }


def stage_patch(out: Path, member: str, destination: Path) -> dict[str, Any]:
    started = time.perf_counter()
    info = patch_runtime(destination, member)

    # Import the patched module in isolation and prove the member is BOTH in the
    # pool and in the frozen live config -- "available" is not "wired".
    code = "\n".join(
        (
            "import json, sys",
            f"sys.path.insert(0, {str(destination)!r})",
            "from runtime.fx2_model_axis_corrector import fx2_family_specs",
            "from runtime.free_corrector import SHIPPED_CONFIG as LIVE",
            "specs = fx2_family_specs()",
            f"name = {member!r}",
            "print(json.dumps({",
            "    'member_in_pool': name in specs,",
            "    'member_in_config': name in LIVE['families'],",
            "    'member_cells': int(specs[name][0]) if name in specs else -1,",
            "    'family_count': len(LIVE['families']),",
            "}))",
        )
    )
    probe = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise Gb1Error(f"patched runtime failed to import:\n{probe.stderr}")
    wiring = json.loads(probe.stdout.strip().splitlines()[-1])
    if not (wiring["member_in_pool"] and wiring["member_in_config"]):
        raise Gb1Error(f"member {member} is not wired: {wiring}")

    receipt = {
        "stage": "patch",
        "axis": "[build artifact -- no measurement, no score claim]",
        "score_claim": False,
        "counted_bytes_added": 0,
        "rule_118": (
            "the member reads only already-decoded symbols and the position index; "
            "nothing is transmitted, learned-and-shipped, or video-derived"
        ),
        "wiring": wiring,
        **info,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(out / f"PATCH_{member}.json", receipt)
    return receipt


# --- stage: seal -----------------------------------------------------------

DX2_POINTER = {
    "archive_sha256": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",  # gitleaks:allow -- public content digest
    "archive_bytes": 180_368,
    "score": 0.14821987563243377,
    "axis": "contest_cuda",
    "inflate_seconds": 505.546,
    "decode_and_render_seconds": 500.837,
}
EXCHANGE_RATE_S_PER_BYTE = 6.658590e-07
"""``ddm_tx1`` sec.0 = 25 / 37,545,489.  CITED, not re-derived."""


def stage_seal(out: Path, store: Path) -> dict[str, Any]:
    """Assemble the candidate seal from the physical re-encode receipts.

    Refuses unless the CONTROL row proves the encoder inverts the shipping decoder;
    without it no byte delta from this encoder is trustworthy, so a seal built on
    one would be a rate claim resting on an unproven instrument.
    """
    started = time.perf_counter()
    retained = store / "retained"
    control_path = retained / "S1_control_600.json"
    if not control_path.is_file():
        raise Gb1Error(f"no control receipt at {control_path}; refusing to seal")
    control = json.loads(control_path.read_text())
    if not control.get("byte_identical"):
        raise Gb1Error(
            "CONTROL FAILED: the re-encoded stream is not byte-identical to the "
            "shipped token section. Every candidate byte delta is VOID."
        )

    candidates = []
    for receipt in sorted(retained.glob("S1_encode_gb1_*.json")):
        # ExFAT writes AppleDouble siblings next to every file; they are binary
        # metadata, not receipts, and json.loads on one is a confusing crash.
        if receipt.name.startswith("._"):
            continue
        row = json.loads(receipt.read_text())
        if row.get("tokens_changed", -1) != 0:
            raise Gb1Error(
                f"{receipt.name} changed {row.get('tokens_changed')} tokens; this "
                "family must move zero symbols and a nonzero count VOIDS the row"
            )
        delta = float(row["archive_delta_bytes"])
        candidates.append(
            {
                "receipt": str(receipt),
                "tag": row.get("tag"),
                "token_stream_bytes": row.get("token_stream_bytes_candidate"),
                "token_stream_delta_bytes": row.get("token_stream_delta_bytes"),
                "archive_bytes": row.get("archive_bytes_candidate"),
                "archive_delta_bytes": delta,
                "tokens_changed": row.get("tokens_changed"),
                "delta_S_rate": delta * EXCHANGE_RATE_S_PER_BYTE,
                "code_bytes_ideal": row.get("code_bytes_ideal"),
            }
        )

    best = min(candidates, key=lambda row: row["archive_delta_bytes"], default=None)
    seal = {
        "schema": "ddm_gb1_groupbin8_candidate_seal.v1",
        "candidate_id": "ddm_gb1_groupbin8_decode_scan_conditioning",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
        "pointer_at_seal": DX2_POINTER,
        "exchange_rate_s_per_byte": EXCHANGE_RATE_S_PER_BYTE,
        "control": {
            "emitted_bytes": control.get("emitted_bytes"),
            "shipped_token_stream_bytes": control.get("shipped_token_stream_bytes"),
            "byte_identical": control.get("byte_identical"),
            "emitted_sha256": control.get("emitted_sha256"),
        },
        "candidates": candidates,
        "best": best,
        "counted_bytes_added_by_the_member": 0,
        "distortion": {
            "d_seg_delta": 0.0,
            "d_pose_delta": 0.0,
            "why": (
                "the coder emits the transmitted symbol whatever the model said, so a "
                "probability-law change moves bits and never symbols; tokens_changed=0 "
                "on every row above is the digest-level proof that it was honoured"
            ),
        },
        "blockers": [
            "C PORT OWED: runtime/f26_corrector_native.c compiles the member list in as "
            "constants and native_free_corrector.py REFUSES on drift, so a Python-only "
            "member falls back to the Python corrector (ddm_rr8: 1,419.9 s inflate vs "
            "464.6 s native). Precedent: ddm_fx5 ported six members in 3 files.",
            "CANDIDATE-SIDE PARSE-BACK OWED: tokens_changed=0 is an encoder-side "
            "statement; an independent receiver run over a candidate stream is not done.",
            "CONTEST-CPU WALL: ddm_rc2's 13-member CPU row measured 2,850.78 s against "
            "the 1,800 s budget. That failure predates this member and is not caused by "
            "it, but it is the axis where the wall is actually live.",
        ],
        "falsifiers": [
            "INSTANCE: tokens_changed != 0 on any row VOIDS that row -- the arithmetic "
            "decoder desynchronised and the number is not a saving.",
            "INSTANCE: the control row must emit a stream byte-identical to the shipped "
            "113,777 B token section, or every candidate delta here is void.",
            "INSTANCE: a candidate whose archive_delta_bytes is >= -20 B fails ddm_mi1's "
            "own admission bar and is an honest negative about the realization gap.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(out / "CANDIDATE_SEAL_gb1_groupbin8.json", seal)
    return seal


# --- entry point -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", required=True, choices=("verify", "fit", "patch", "seal")
    )
    parser.add_argument("--out", default=str(OUT_ROOT))
    parser.add_argument("--seeds", default="20260824,777,31337")
    parser.add_argument("--member", default="cls_groupbin8")
    parser.add_argument("--destination", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.stage == "verify":
        receipt = stage_verify(out)
    elif args.stage == "fit":
        seeds = tuple(int(s) for s in args.seeds.split(",") if s)
        receipt = stage_fit(out, seeds)
    elif args.stage == "seal":
        receipt = stage_seal(out, out.parent)
    else:
        destination = Path(
            args.destination
            or (APSTORE / "ddm_gb1_groupbin8_conditioning" / f"runtime_{args.member}")
        )
        receipt = stage_patch(out, args.member, destination)

    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
