"""ddm_mi1 - is there RESIDUAL INFORMATION about the flip indicator that the
shipped DX2 stack does not already condition on?

THE OBJECT.  ``ddm_hc1`` measured that 97.80% of the DX2 token stream is one
binary question asked 117,964,800 times: "is the receiver's argmax right?".
``ddm_hc1`` then closed RECALIBRATION of that question (8.44 net bytes).  This
module attacks the other half: CONDITIONING.  A recalibration reprices what the
model already says; a conditioning gain needs a variable the model does not see.

WHY IT IS SCORER-FREE AND ZERO-DISTORTION.  Nothing here changes a decoded
symbol.  A probability model feeds the range coder; the coder emits the
transmitted symbol whatever the model said.  Changing the model changes BITS,
never SYMBOLS, so every SegNet cell and every PoseNet input is bit-identical and
``dD = 0`` is an identity, not a measurement (``ddm_hc1`` sec.7, same argument).

THE INSTRUMENT.  For a candidate context ``C`` we fit ONE log-odds offset per
cell of ``C`` on a training fold and score the held-out fold:

    q_i  = 1 - pmax_i                      the model's own flip probability
    l_i  = logit(q_i)
    q'_i = sigmoid(l_i + beta_c(i))        beta = 0 nests the shipped model
    cost = -log2(q') on a flip, -log2(1-q') otherwise

``beta = 0`` reproduces the shipped code length EXACTLY, so every reported delta
is the new mechanism and never the plumbing.  This is ``ddm_hc1``'s per-``d``
offset rung generalised to arbitrary contexts, which makes hc1's measured rows
usable as positive controls on my own instrument.

WHAT IS A CONTROL AND WHAT IS A TEST.  Read at source from the shipped runtime
(``runtime/rr4_free_corrector.py`` ``CONTEXT_SIZE`` and ``group_state``), the
shipped adaptive corrector's context is exactly

    5 (predicted class) x 64 (ubin) x 2 (agree1) x 2 (agree2) x 8 (run)
      x 5 (boundary bucket)  =  51,200 cells

so ``boundary``, ``agree2`` and ``run`` are ALREADY CONSUMED and are carried here
as NEGATIVE controls: my instrument must report them near zero.  The one
causally-legal, receiver-derivable axis that appears NOWHERE in the shipped stack
is ABSOLUTE POSITION IN THE FRAME.  The shipped network is built with
``patch=HPAC_PATCH=64`` (``cpr1/inflate.py:33,257``), so it tiles the frame into
6x8 = 48 tiles, gives EVERY tile the same tile-relative coordinate grid and the
same per-frame shift, and no corrector carries a position feature at all.  WHICH
tile a pixel is in is therefore structurally invisible to it.  ``tile48`` is that
TEST.

``subtile4`` (quadrant within the tile) was designed as its NEGATIVE control,
because the coordinate channels DO carry it -- and THE CONTROL FAILED, at +56.51
held-out bytes from four cells.  Chasing that produced the arm's mechanism:
``groupbin8``, eight bins of the within-tile decode-group index
(``cpr1/inflate.py:275-281``), returns +64.20 B, and the fitted offsets run
monotonically along the scan.  The shipped model is over-cautious where little of
its causal context has been decoded and over-confident where most of it has.
**A feature being an INPUT to a trained network does not mean the network
consumed it**; an adaptive counter converges on its own context by construction,
a trained network carries no such guarantee.  Read every ``shipped`` flag below
as "measurably consumed", never as "present as an input".

SATURATED POSITIONS ARE EXCLUDED, EXACTLY.  Where ``pmax == 1.0`` in float32 the
cost is 0 for any finite offset, so those positions are immovable by this family
and are dropped from the fit.  ``ddm_df1`` measured 67,955,679 such positions
carrying 0.0486 B in total.

NO SCORER, NO MODAL, NO METAL, NO TRAINING, NO ARCHIVE MUTATION.  Every input is
a retained field from a prior arm, consumed by sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

# --- custody ---------------------------------------------------------------

VERTIGO = Path("/Volumes/VertigoDataTier/pact")
APSTORE = Path("/Volumes/APDataStore/pact")

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
HC1_D = (
    APSTORE
    / "ddm_hc1_hpac_calibration"
    / "measurement_v1"
    / "retained"
    / "boundary_distance_d.u8.bin"
)

OUT_ROOT = APSTORE / "ddm_mi1_indicator_model_axis" / "measurement_v1"

N = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
NUM_CLASSES = 5

# Shipped rr4 constants, transcribed from runtime/rr4_free_corrector.py so the
# controls reproduce the shipped feature exactly rather than an approximation.
RR4_RUN_LEVELS = 8
RR4_RUN_CAP = 255
PATCH = 32  # sub-tile granularity used to split the model tile
HPAC_PATCH = 64  # cpr1/inflate.py:33 -- the SHIPPED IntegerHPAC.P and group tile

EXPECTED = {
    "tokens_sha256": (
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
    ),  # gitleaks:allow -- public content digest
    "tokens_bytes": 117_964_800,
    # ddm_hc1 sec.2 / ddm_df1 FIELD.json, both on this exact body.
    "hc1_flips": 227_671,
    "hc1_indicator_bytes": 111_275.62,
    "hc1_zero_branch_bytes": 34_674.08,
    "hc1_one_branch_bytes": 76_601.54,
    "df1_address_bound_bits": 872_907.9671775216,
    "df1_saturated_positions": 67_955_679,
}

# ddm_tx1_toolbox_crosswalk_20260819.md sec.0 -- cited, never re-derived.
S_PER_ARCHIVE_BYTE = 6.658590e-07
DEMAND_BYTES = 42_381.16120555642
# ddm_eu2 EU2_RECEIPT.md: 10K int8 counted weight packet.
EU2_MODEL_BYTES = 10_000
EU2_MODEL_S = 0.006658589531

LN2 = math.log(2.0)


class Mi1Error(RuntimeError):
    """Fail closed; never degrade to a partial measurement."""


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


# --- field access ----------------------------------------------------------


def open_fields() -> dict[str, np.memmap]:
    for path in (TO2_TOKENS, DF1_ARGMAX, DF1_PMAX, HC1_D):
        if not path.exists():
            raise Mi1Error(f"retained input absent: {path}")
    tokens = np.memmap(TO2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    argmax = np.memmap(DF1_ARGMAX, dtype=np.uint8, mode="r", shape=(N, PLANE))
    pmax = np.memmap(DF1_PMAX, dtype="<f4", mode="r", shape=(N, PLANE))
    boundary = np.memmap(HC1_D, dtype=np.uint8, mode="r", shape=(N, PLANE))
    return {"tokens": tokens, "argmax": argmax, "pmax": pmax, "boundary": boundary}


def _binary_entropy_bits(q: np.ndarray) -> np.ndarray:
    """H_b(q) in bits, exact at the endpoints."""
    out = np.zeros_like(q)
    interior = (q > 0.0) & (q < 1.0)
    qi = q[interior]
    out[interior] = -(qi * np.log(qi) + (1.0 - qi) * np.log1p(-qi)) / LN2
    return out


def _code_variance_bits2(q: np.ndarray) -> np.ndarray:
    """Var of the realised code length at one position, bits squared.

    The code length is ``-log2 q`` with probability ``q`` and ``-log2(1-q)``
    otherwise, so the variance is ``q(1-q)(log2(q/(1-q)))^2``.
    """
    out = np.zeros_like(q)
    interior = (q > 0.0) & (q < 1.0)
    qi = q[interior]
    logit = (np.log(qi) - np.log1p(-qi)) / LN2
    out[interior] = qi * (1.0 - qi) * logit * logit
    return out


# --- stage: verify ---------------------------------------------------------


def stage_verify(out_root: Path) -> dict[str, Any]:
    """Custody, then re-derive the hc1/df1 decomposition from first principles."""
    started = time.time()
    fields = open_fields()
    tokens, argmax, pmax = fields["tokens"], fields["argmax"], fields["pmax"]

    custody = {
        "to2_tokens": {
            "path": str(TO2_TOKENS),
            "bytes": TO2_TOKENS.stat().st_size,
            "sha256": sha256_file(TO2_TOKENS),
        },
        "df1_coding_argmax": {
            "path": str(DF1_ARGMAX),
            "bytes": DF1_ARGMAX.stat().st_size,
            "sha256": sha256_file(DF1_ARGMAX),
        },
        "df1_coding_pmax": {
            "path": str(DF1_PMAX),
            "bytes": DF1_PMAX.stat().st_size,
            "sha256": sha256_file(DF1_PMAX),
        },
        "hc1_boundary_d": {
            "path": str(HC1_D),
            "bytes": HC1_D.stat().st_size,
            "sha256": sha256_file(HC1_D),
        },
    }
    if custody["to2_tokens"]["sha256"] != EXPECTED["tokens_sha256"]:
        raise Mi1Error("TO2 decoded token field digest drifted from the shipped body")
    if custody["to2_tokens"]["bytes"] != EXPECTED["tokens_bytes"]:
        raise Mi1Error("TO2 decoded token field size drifted")

    flips = 0
    saturated = 0
    saturated_flips = 0
    zero_branch_bits = 0.0
    one_branch_bits = 0.0
    entropy_bits = 0.0
    variance_bits2 = 0.0

    for frame in range(N):
        token = np.asarray(tokens[frame], dtype=np.uint8).reshape(-1)
        arg = np.asarray(argmax[frame], dtype=np.uint8)
        flip = token != arg
        q = 1.0 - np.asarray(pmax[frame], dtype=np.float64)

        flips += int(flip.sum())
        sat = q <= 0.0
        saturated += int(sat.sum())
        saturated_flips += int((sat & flip).sum())

        live = ~sat
        ql = q[live]
        fl = flip[live]
        # -log2(1-q) on the "argmax is right" branch; -log2(q) on the other.
        zero_branch_bits += float(-np.log1p(-ql[~fl]).sum() / LN2)
        one_branch_bits += float(-np.log(ql[fl]).sum() / LN2)
        entropy_bits += float(_binary_entropy_bits(ql).sum())
        variance_bits2 += float(_code_variance_bits2(ql).sum())

    indicator_bits = zero_branch_bits + one_branch_bits
    excess_bits = indicator_bits - entropy_bits
    sigma_bits = math.sqrt(variance_bits2)

    result = {
        "axis": "[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]",
        "score_claim": False,
        "promotable": False,
        "custody": custody,
        "positions": POSITIONS,
        "flips": flips,
        "saturated_positions": saturated,
        "saturated_flips": saturated_flips,
        "indicator": {
            "bits": indicator_bits,
            "bytes": indicator_bits / 8.0,
            "zero_branch_bits": zero_branch_bits,
            "zero_branch_bytes": zero_branch_bits / 8.0,
            "one_branch_bits": one_branch_bits,
            "one_branch_bytes": one_branch_bits / 8.0,
            "bits_per_flip_one_branch": one_branch_bits / flips if flips else 0.0,
            "mean_model_flip_probability_over_flips": (
                2.0 ** (-one_branch_bits / flips) if flips else 0.0
            ),
        },
        "model_entropy": {
            "definition": "sum_i H_b(1 - pmax_i), df1 address_bound",
            "bits": entropy_bits,
            "bytes": entropy_bits / 8.0,
        },
        "realised_excess_over_entropy": {
            "bits": excess_bits,
            "bytes": excess_bits / 8.0,
            "null_sigma_bits": sigma_bits,
            "z": excess_bits / sigma_bits if sigma_bits > 0 else float("nan"),
            "note": (
                "under the null that the model is perfectly calibrated AT EVERY "
                "POSITION the realised code length has mean = the entropy and "
                "this sigma; z says whether the gap is sampling noise or "
                "systematic headroom"
            ),
        },
        "controls": {
            "hc1_flips": EXPECTED["hc1_flips"],
            "hc1_indicator_bytes": EXPECTED["hc1_indicator_bytes"],
            "hc1_zero_branch_bytes": EXPECTED["hc1_zero_branch_bytes"],
            "hc1_one_branch_bytes": EXPECTED["hc1_one_branch_bytes"],
            "df1_address_bound_bits": EXPECTED["df1_address_bound_bits"],
            "df1_saturated_positions": EXPECTED["df1_saturated_positions"],
        },
        "break_even": break_even_table(indicator_bits / 8.0),
        "elapsed_seconds": time.time() - started,
    }

    if flips != EXPECTED["hc1_flips"]:
        raise Mi1Error(
            f"flip count {flips} disagrees with hc1's {EXPECTED['hc1_flips']}"
        )
    if saturated != EXPECTED["df1_saturated_positions"]:
        raise Mi1Error(
            f"saturated count {saturated} disagrees with df1's "
            f"{EXPECTED['df1_saturated_positions']}"
        )
    if not math.isclose(
        entropy_bits, EXPECTED["df1_address_bound_bits"], rel_tol=1e-9, abs_tol=1e-3
    ):
        raise Mi1Error("re-derived model entropy disagrees with df1's address bound")
    if abs(indicator_bits / 8.0 - EXPECTED["hc1_indicator_bytes"]) > 0.05:
        raise Mi1Error("re-derived indicator code length disagrees with hc1")

    _atomic_write_json(out_root / "VERIFY.json", result)
    return result


def break_even_table(indicator_bytes: float) -> dict[str, Any]:
    """What a PAID probability model has to beat, in its own units."""
    gross_to_break_even = EU2_MODEL_BYTES
    gross_to_close_demand = DEMAND_BYTES + EU2_MODEL_BYTES
    return {
        "indicator_bytes": indicator_bytes,
        "eu2_model_bytes": EU2_MODEL_BYTES,
        "eu2_model_S": EU2_MODEL_S,
        "eu2_model_bytes_from_S": EU2_MODEL_S / S_PER_ARCHIVE_BYTE,
        "S_per_archive_byte": S_PER_ARCHIVE_BYTE,
        "demand_bytes_at_fixed_distortion": DEMAND_BYTES,
        "break_even_fraction_of_indicator": gross_to_break_even / indicator_bytes,
        "close_demand_fraction_of_indicator": gross_to_close_demand / indicator_bytes,
    }


# --- stage: ladder ---------------------------------------------------------


def _build_compact(fields: dict[str, np.memmap]) -> dict[str, np.ndarray]:
    """One streaming pass -> compact per-position arrays over live positions.

    ``run``, ``agree2`` and ``ubin`` replicate ``rr4_free_corrector`` exactly --
    its dtypes, its saturation, and its frame-0/frame-1 initialisation -- so the
    control rows measure the SHIPPED features rather than a lookalike.
    ``agree1`` is not built: ``agree2`` and ``run`` already cover the temporal
    axis this arm is testing, and neither is a candidate.
    """
    tokens, argmax, pmax, boundary = (
        fields["tokens"],
        fields["argmax"],
        fields["pmax"],
        fields["boundary"],
    )

    row_index = np.repeat(np.arange(HEIGHT, dtype=np.int32), WIDTH)
    col_index = np.tile(np.arange(WIDTH, dtype=np.int32), HEIGHT)
    patch_row = row_index // PATCH
    patch_col = col_index // PATCH
    patch192 = (patch_row * (WIDTH // PATCH) + patch_col).astype(np.int32)
    # The SHIPPED model tiles at HPAC_PATCH = 64 (cpr1/inflate.py:33,257), not at
    # IntegerHPAC's class default of 32, so its own tessellation is 6x8 = 48.
    tile48 = (
        (row_index // HPAC_PATCH) * (WIDTH // HPAC_PATCH) + col_index // HPAC_PATCH
    ).astype(np.int32)
    subtile4 = (
        ((row_index % HPAC_PATCH) // PATCH) * 2 + (col_index % HPAC_PATCH) // PATCH
    ).astype(np.int32)
    # Scan position WITHIN the tile.  cpr1/inflate.py:275-281 orders decoding by
    # ``columns + HPAC_DELTA * rows`` over the 64x64 tile, giving 190 groups, so
    # this is exactly "how much of my tile has already been decoded".
    group_in_tile = (
        (col_index % HPAC_PATCH) + 2 * (row_index % HPAC_PATCH)
    ).astype(np.int32)
    groupbin8 = (group_in_tile * 8 // 190).astype(np.int32)

    chunks: dict[str, list[np.ndarray]] = {
        "logit": [],
        "flip": [],
        "boundary": [],
        "agree2": [],
        "run": [],
        "patch192": [],
        "tile48": [],
        "subtile4": [],
        "groupbin8": [],
        "group190": [],
        "row": [],
        "ubin": [],
        "cls": [],
        "frame": [],
    }

    prev1 = np.zeros(PLANE, dtype=np.uint8)
    prev2 = np.zeros(PLANE, dtype=np.uint8)
    # int64, exactly as rr4_free_corrector.RR4FreeCorrector.__init__ declares it.
    # uint8 here would WRAP at RUN_CAP instead of saturating, and would mislabel
    # every position whose class has been stable for 256+ frames -- which on a
    # 600-frame dashcam is most of the static ego-hood and sky.
    run_state = np.zeros(PLANE, dtype=np.int64)
    have_prev = False

    for frame in range(N):
        token = np.asarray(tokens[frame], dtype=np.uint8).reshape(-1)
        arg = np.asarray(argmax[frame], dtype=np.uint8)
        q = 1.0 - np.asarray(pmax[frame], dtype=np.float64)
        live = q > 0.0

        agree2 = (
            (prev2 == arg).astype(np.uint8)
            if have_prev
            else np.zeros(PLANE, dtype=np.uint8)
        )
        run_level = np.minimum(run_state, RR4_RUN_LEVELS - 1).astype(np.uint8)

        ql = q[live]
        chunks["logit"].append((np.log(ql) - np.log1p(-ql)).astype(np.float64))
        chunks["flip"].append((token != arg)[live].astype(np.uint8))
        chunks["boundary"].append(np.asarray(boundary[frame], dtype=np.uint8)[live])
        chunks["agree2"].append(agree2[live])
        chunks["run"].append(run_level[live])
        chunks["patch192"].append(patch192[live])
        chunks["tile48"].append(tile48[live])
        chunks["subtile4"].append(subtile4[live])
        chunks["groupbin8"].append(groupbin8[live])
        chunks["group190"].append(group_in_tile[live])
        chunks["row"].append(row_index[live])
        chunks["cls"].append(arg[live])
        # ubin: rr4's surprise axis, floor(-log2(q) / 0.5) clipped to [0, 63].
        ubin = np.clip(np.floor(-np.log2(ql) / 0.5), 0, 63).astype(np.uint8)
        chunks["ubin"].append(ubin)
        chunks["frame"].append(np.full(int(live.sum()), frame, dtype=np.int32))

        if have_prev:
            run_state = np.where(
                token == prev1, np.minimum(run_state + 1, RR4_RUN_CAP), 0
            ).astype(np.int64)
            prev2 = prev1
        prev1 = token.copy()
        have_prev = True

    compact = {key: np.concatenate(value) for key, value in chunks.items()}
    return compact


def _fit_offsets(
    logit: np.ndarray,
    flip: np.ndarray,
    cell: np.ndarray,
    n_cells: int,
    train: np.ndarray,
    iterations: int = 24,
) -> np.ndarray:
    """Newton on one log-odds offset per cell, over the training fold only."""
    beta = np.zeros(n_cells, dtype=np.float64)
    cell_train = cell[train]
    logit_train = logit[train]
    flip_train = flip[train].astype(np.float64)
    for _ in range(iterations):
        u = logit_train + beta[cell_train]
        q = 1.0 / (1.0 + np.exp(-u))
        gradient = np.bincount(cell_train, weights=flip_train - q, minlength=n_cells)
        hessian = np.bincount(cell_train, weights=q * (1.0 - q), minlength=n_cells)
        step = np.divide(
            gradient, hessian, out=np.zeros_like(gradient), where=hessian > 0.0
        )
        step = np.clip(step, -4.0, 4.0)
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def _code_bits(logit: np.ndarray, flip: np.ndarray, offset: np.ndarray) -> float:
    """Total code length in bits at the given per-position log-odds offset."""
    u = logit + offset
    # -log2(q) on a flip, -log2(1-q) otherwise; both stable through logaddexp.
    bits = np.where(
        flip.astype(bool), np.logaddexp(0.0, -u), np.logaddexp(0.0, u)
    ) / LN2
    return float(bits.sum())


def context_definitions() -> dict[str, dict[str, Any]]:
    """Every context is a function of symbols the receiver has ALREADY decoded.

    ``shipped`` marks a feature verified present in the shipped runtime, so its
    row is a NEGATIVE control on this instrument rather than a candidate.
    """
    return {
        "none": {
            "shipped": True,
            "levels": 1,
            "why": "pure recalibration; hc1 closed this family at 8.44 net B",
            "build": lambda c: (np.zeros(c["flip"].size, dtype=np.int64), 1),
        },
        "boundary_d": {
            "shipped": True,
            "levels": 5,
            "why": "rr4 BOUNDARY_LEVELS; hc1 measured the per-d offset rung",
            "build": lambda c: (c["boundary"].astype(np.int64), 5),
        },
        "agree2": {
            "shipped": True,
            "levels": 2,
            "why": "rr4 agree2 -- t-2 agreement, tba1 D6 names it consumed",
            "build": lambda c: (c["agree2"].astype(np.int64), 2),
        },
        "run8": {
            "shipped": True,
            "levels": RR4_RUN_LEVELS,
            "why": "rr4 run -- saturated temporal run length, also in D6",
            "build": lambda c: (c["run"].astype(np.int64), RR4_RUN_LEVELS),
        },
        "cls_ubin": {
            "shipped": True,
            "levels": NUM_CLASSES * 64,
            "why": "the rr4 context head; must be near zero if rr4 converged",
            "build": lambda c: (
                c["cls"].astype(np.int64) * 64 + c["ubin"].astype(np.int64),
                NUM_CLASSES * 64,
            ),
        },
        "tile48": {
            "shipped": False,
            "levels": 48,
            "why": (
                "THE MODEL'S OWN 64px TILE INDEX -- cpr1/inflate.py:33,257 builds "
                "IntegerHPAC with patch=HPAC_PATCH=64, so the frame is 6x8 tiles "
                "and the coordinate channels are tile-RELATIVE; which tile it is "
                "in is structurally invisible to the network"
            ),
            "build": lambda c: (c["tile48"].astype(np.int64), 48),
        },
        "subtile4": {
            "shipped": True,
            "levels": 4,
            "why": (
                "quadrant WITHIN the model's 64px tile -- the model DOES see this "
                "through its coordinate channels, so it is a negative control"
            ),
            "build": lambda c: (c["subtile4"].astype(np.int64), 4),
        },
        "groupbin8": {
            "shipped": False,
            "why": (
                "SCAN POSITION within the tile, 8 bins of the 190 decode groups "
                "(cpr1/inflate.py:275-281) -- 'how much of my tile is already "
                "decoded'.  subtile4 is a crude 4-level proxy for this; if the "
                "mechanism is scan position, this must beat it"
            ),
            "levels": 8,
            "build": lambda c: (c["groupbin8"].astype(np.int64), 8),
        },
        "group190": {
            "shipped": False,
            "why": "the exact decode group index within the tile, unbinned",
            "levels": 190,
            "build": lambda c: (c["group190"].astype(np.int64), 190),
        },
        "patch192": {
            "shipped": False,
            "levels": 192,
            "why": "tile48 x subtile4 -- a 4x refinement of the model's own tile",
            "build": lambda c: (c["patch192"].astype(np.int64), 192),
        },
        "row384": {
            "shipped": False,
            "levels": HEIGHT,
            "why": "ABSOLUTE ROW -- the scene's strongest structural axis",
            "build": lambda c: (c["row"].astype(np.int64), HEIGHT),
        },
        "row384_x_ubin": {
            "shipped": False,
            "levels": HEIGHT * 64,
            "why": "absolute row crossed with the model's own confidence",
            "build": lambda c: (
                c["row"].astype(np.int64) * 64 + c["ubin"].astype(np.int64),
                HEIGHT * 64,
            ),
        },
        "patch192_x_ubin": {
            "shipped": False,
            "levels": 192 * 64,
            "why": "absolute patch crossed with the model's own confidence",
            "build": lambda c: (
                c["patch192"].astype(np.int64) * 64 + c["ubin"].astype(np.int64),
                192 * 64,
            ),
        },
        "frame_row": {
            "shipped": False,
            "levels": 24 * HEIGHT,
            "why": "absolute row crossed with a coarse frame band (drift check)",
            "build": lambda c: (
                (c["frame"].astype(np.int64) // 25) * HEIGHT
                + c["row"].astype(np.int64),
                24 * HEIGHT,
            ),
        },
    }


def stage_ladder(out_root: Path, seed: int) -> dict[str, Any]:
    started = time.time()
    fields = open_fields()
    compact = _build_compact(fields)
    logit = compact["logit"]
    flip = compact["flip"]
    live = logit.size

    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, size=live, endpoint=False).astype(np.uint8)
    fold_a = fold == 0
    fold_b = ~fold_a

    zero = np.zeros(live, dtype=np.float64)
    base_bits = _code_bits(logit, flip, zero)

    rows = []
    for name, spec in context_definitions().items():
        cell, n_cells = spec["build"](compact)
        beta_a = _fit_offsets(logit, flip, cell, n_cells, fold_a)
        beta_b = _fit_offsets(logit, flip, cell, n_cells, fold_b)

        offset_heldout = np.empty(live, dtype=np.float64)
        offset_heldout[fold_b] = beta_a[cell[fold_b]]
        offset_heldout[fold_a] = beta_b[cell[fold_a]]
        heldout_bits = _code_bits(logit, flip, offset_heldout)

        beta_all = _fit_offsets(logit, flip, cell, n_cells, np.ones(live, dtype=bool))
        in_sample_bits = _code_bits(logit, flip, beta_all[cell])

        rows.append(
            {
                "context": name,
                "shipped_already": bool(spec["shipped"]),
                "cells": int(n_cells),
                "why": spec["why"],
                "heldout_gain_bits": base_bits - heldout_bits,
                "heldout_gain_bytes": (base_bits - heldout_bits) / 8.0,
                "in_sample_gain_bits": base_bits - in_sample_bits,
                "in_sample_gain_bytes": (base_bits - in_sample_bits) / 8.0,
                "max_abs_offset": float(np.max(np.abs(beta_all))),
            }
        )

    best = max(rows, key=lambda row: row["heldout_gain_bytes"])
    base_bytes = base_bits / 8.0
    result = {
        "axis": "[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]",
        "score_claim": False,
        "promotable": False,
        "seed": seed,
        "live_positions": int(live),
        "excluded_saturated": POSITIONS - int(live),
        "base_indicator_bits": base_bits,
        "base_indicator_bytes": base_bytes,
        "rows": rows,
        "best_heldout_bytes": best["heldout_gain_bytes"],
        "best_context": best["context"],
        "break_even": break_even_table(base_bytes),
        "best_as_fraction_of_break_even": (
            best["heldout_gain_bytes"] / EU2_MODEL_BYTES
        ),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_write_json(out_root / f"LADDER_seed{seed}.json", result)
    return result


# --- stage: retain ---------------------------------------------------------

RETAIN_CONTEXTS = (
    "boundary_d",
    "agree2",
    "run8",
    "tile48",
    "subtile4",
    "groupbin8",
    "patch192",
    "row384",
)


def stage_retain(out_root: Path) -> dict[str, Any]:
    """Persist the FITTED OBJECT, not just the aggregate it produced.

    The ladder's scalar gains are summaries of a per-cell table -- live
    positions, observed flips, the model's own stated flip mass, and the fitted
    log-odds offset.  That table IS the mechanism a downstream arm would ship,
    and it is what tells them WHICH cells carry the gain.  Persisting only the
    total would be the measure-and-discard pattern.

    The per-position cost field is deliberately NOT retained: it is exactly
    reconstructible from this table plus the already-retained ``pmax`` field and
    this script, so it is certified rebuildable rather than discarded.
    """
    started = time.time()
    fields = open_fields()
    compact = _build_compact(fields)
    logit = compact["logit"]
    flip = compact["flip"]
    live = logit.size
    everything = np.ones(live, dtype=bool)
    definitions = context_definitions()

    tables = {}
    for name in RETAIN_CONTEXTS:
        cell, n_cells = definitions[name]["build"](compact)
        beta = _fit_offsets(logit, flip, cell, n_cells, everything)
        stated = 1.0 / (1.0 + np.exp(-logit))
        counts = np.bincount(cell, minlength=n_cells).astype(np.int64)
        flips_per_cell = np.bincount(
            cell, weights=flip.astype(np.float64), minlength=n_cells
        )
        stated_mass = np.bincount(cell, weights=stated, minlength=n_cells)
        base = np.bincount(
            cell,
            weights=np.where(
                flip.astype(bool), np.logaddexp(0.0, -logit), np.logaddexp(0.0, logit)
            )
            / LN2,
            minlength=n_cells,
        )
        u = logit + beta[cell]
        fitted = np.bincount(
            cell,
            weights=np.where(
                flip.astype(bool), np.logaddexp(0.0, -u), np.logaddexp(0.0, u)
            )
            / LN2,
            minlength=n_cells,
        )
        # Internal consistency: the per-cell base bits must re-sum to the whole
        # indicator, or the partition dropped or double-counted positions.
        total_base = float(base.sum())
        if not math.isclose(
            total_base, EXPECTED["hc1_indicator_bytes"] * 8.0, rel_tol=1e-6
        ):
            raise Mi1Error(
                f"{name}: per-cell base bits {total_base} do not re-sum to the "
                "indicator code length"
            )
        if int(counts.sum()) != live:
            raise Mi1Error(f"{name}: cell partition does not cover every position")

        tables[name] = {
            "cells": int(n_cells),
            "live_positions": counts.tolist(),
            "observed_flips": flips_per_cell.tolist(),
            "model_stated_flip_mass": stated_mass.tolist(),
            "fitted_log_odds_offset": beta.tolist(),
            "base_bits": base.tolist(),
            "fitted_bits": fitted.tolist(),
        }

    payload = {
        "axis": "[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]",
        "score_claim": False,
        "promotable": False,
        "live_positions": int(live),
        "note": (
            "in-sample fits over ALL live positions; the held-out numbers are in "
            "the LADDER receipts. base_bits minus fitted_bits per cell is the "
            "in-sample gain attributable to that cell."
        ),
        "rebuild": (
            "the per-position cost field is reconstructible from these offsets, "
            "the retained df1 pmax/argmax fields, and this script's --stage retain"
        ),
        "tables": tables,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_write_json(out_root / "RETAIN_cell_tables.json", payload)
    return {key: value for key, value in payload.items() if key != "tables"}


# --- stage: manifest -------------------------------------------------------


def stage_manifest(out_root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(out_root.rglob("*")):
        if path.is_file() and path.suffix != ".tmp":
            entries.append(
                {
                    "path": str(path.relative_to(out_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "root": str(out_root),
        "entries": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    _atomic_write_json(out_root / "MANIFEST.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("verify", "ladder", "retain", "manifest")
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args(argv)

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "verify":
        payload = stage_verify(args.out_root)
    elif args.stage == "ladder":
        payload = stage_ladder(args.out_root, args.seed)
    elif args.stage == "retain":
        payload = stage_retain(args.out_root)
    else:
        payload = stage_manifest(args.out_root)
    print(json.dumps(payload, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
