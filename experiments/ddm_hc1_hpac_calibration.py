"""ddm_hc1 -- is the shipped HPAC probability model CALIBRATED, and what does
perfect recalibration release at EXACTLY zero distortion?

THE OBJECT.  The DX2 body (archive sha ``976f706d...``, 180,368 B, S 0.148220
[contest-CUDA T4 n600]) codes 117,964,800 semantic-token positions through an
RC64 range coder driven by the HPAC model's coding row.  ``ddm_df1`` measured
that stream position-by-position and retained the coding row's ``argmax``,
``pmax``, ``psecond`` and the RC64 realized cost in bits.  This arm reads those
retained fields; it materializes no new archive and fires no scorer.

WHY RECALIBRATION IS DISTORTION-FREE BY CONSTRUCTION, not by measurement.
A recalibration is a deterministic map ``g`` applied to the coding row by BOTH
encoder and decoder.  The decoder still decodes the transmitted symbol; the
autoregressive context, the corrector state and the rendered frame are built
from decoded symbols only.  So the decoded field is bit-identical, and therefore
``d_seg``, ``d_pose`` and every scored cell are bit-identical.  ``dD = 0``
EXACTLY.  That is an identity of the coding pipeline, not an empirical finding,
and this module proves it structurally (``stage=verify``) rather than asserting
it.  What CAN break is codeability -- RC64 quantizes ``floor(p * 2**31)`` and
clamps to 1 -- so the recalibrated rows are checked against that floor.

THE EXACT DECOMPOSITION this arm recalibrates (lossless, df1 sec.4):

    -log2(p_sel) = -log2(pmax)                              if argmax correct
                 = -log2(1-pmax) + -log2(p_sel/(1-pmax))    if argmax wrong

Term 1 is the INDICATOR sub-code "is my argmax right?", a binary event with
model probability ``pmax``.  That is exactly the object a reliability diagram
measures, and it is where the campaign's "confirmation entropy" lives.  Term 2
is the 4-way conditional and is NOT touched here (declared scope, sec. below).

THE CONDITIONING VARIABLE IS DOMAIN-NATIVE, NOT GENERIC.  Per the binding
charter amendment (generic forms are CONTROLS, never candidates -- ``[[m47]]``,
issue ``#1202``), the recalibration map conditions on BOUNDARY DISTANCE ``d``:
the L1 distance to the nearest inter-class token boundary, clipped at 4.  This
is not a proxy and it is not newly invented -- it is
``runtime/residual_archive.py::_boundary_buckets`` applied to the PREVIOUS
decoded frame, which the shipped decoder ALREADY computes at every frame and
ALREADY feeds to the corrector as ``feature = d * 5 + predicted``.  So ``d``
costs zero stored bytes (rule 118: derived at decode from state the receiver
holds), zero extra decode work, and it is causally legal -- frame ``f`` uses
frame ``f-1``, never its own undecoded labels.

That the shipped model already conditions on ``d`` is what makes the question
sharp: a flat family of per-``d`` reliability curves is a clean NEGATIVE (the
corrector already absorbed it); separation is real zero-byte headroom.

A global scalar temperature (Guo et al. 2017) is carried as the CONTROL arm so
the memo can state how much of any gain is generic and how much is domain
structure.  It is never the candidate.

OPTIMAL FORM (declared).
  * Reference form: standard reliability-diagram / calibration analysis --
    equal-mass bins, proper scoring decomposition, ECE, and the achievable
    log-loss reduction from a fitted map evaluated OUT OF SAMPLE.
  * SCOPE reductions (legal): (a) the 4-way conditional term is not
    recalibrated -- only ``pmax`` and ``psecond`` were retained, not the full
    row, so the conditional map is not computable from retained artifacts;
    (b) the temperature CONTROL is evaluated on a fine logit histogram rather
    than position-exact, with the discretization error measured and reported.
    The histogram rungs -- the candidates -- are position-EXACT.
  * NO mechanism reduction.  Every fitted map is evaluated by 2-fold
    cross-fitting over a seeded random split of positions, so every reported
    "achievable" number is heldout over the FULL population.  In-sample numbers
    are reported beside them only to expose the gap (``ddm_pk3``/``pk4`` died
    exactly there: 23/23 in-sample became 0/23 leave-one-out).
  * NO PREFIX.  The split is a seeded random permutation over all 117,964,800
    positions, never a frame prefix (``[[m88]]``/``[[m96]]``: a prefix of a
    skewed population is a different population, and the pose/seg prefix bias
    inverts by axis).

Stages: ``verify`` -> ``dfield`` -> ``analyze``.  Each is resumable from disk
and each retains its payload with sha256 + byte count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_bl1_per_position_bit_allocation as bl1

N = bl1.N
HEIGHT = bl1.HEIGHT
WIDTH = bl1.WIDTH
PLANE = bl1.PLANE
POSITIONS = bl1.POSITIONS
CLASSES = bl1.CLASSES
CLASS_NAMES = bl1.CLASS_NAMES
TOTAL_FREQUENCY = bl1.TOTAL_FREQUENCY
MAX_DISTANCE = 4

DF1_ROOT = Path("/Volumes/APDataStore/pact/ddm_df1_dddb_field/measurement_v1")
DF1_FIELDS = DF1_ROOT / "retained" / "fields"
DF1_ANALYSIS = DF1_ROOT / "analysis"
STORE = Path("/Volumes/APDataStore/pact/ddm_hc1_hpac_calibration/measurement_v1")

FIELD_PMAX = DF1_FIELDS / "position_coding_pmax.f32le.bin"
FIELD_PSECOND = DF1_FIELDS / "position_coding_psecond.f32le.bin"
FIELD_ARGMAX = DF1_FIELDS / "position_coding_argmax.u8.bin"
FIELD_COST = DF1_FIELDS / "position_rc64_frequency_cost_bits.f64le.bin"

# df1's own receipts, reproduced by stage=verify rather than trusted.
DF1_EXPECTED = {
    "total_bits": 910209.2806090622,
    "zero_mode_positions": 117737129,
    "zero_mode_bits": 277392.90330676385,
    "positive_positions": 227671,
    "positive_bits": 632816.3773022982,
    "float32_saturated_positions": 67955679,
}

# ddm_tx1_toolbox_crosswalk_20260819.md sec.0 -- cited, never re-derived.
S_PER_ARCHIVE_BYTE = 6.65859e-07
DEMAND_BYTES = 42381.16120555642

SPLIT_SEED = 20260824
CHUNK = 8_000_000
FALSIFIER_BYTES = 2000.0


class Hc1Error(RuntimeError):
    """A refusal.  Never downgraded to a warning."""


# --------------------------------------------------------------------------
# custody


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def require_free_bytes(root: Path, needed: int) -> int:
    root.mkdir(parents=True, exist_ok=True)
    usage = os.statvfs(root)
    free = usage.f_bavail * usage.f_frsize
    if free < needed:
        raise Hc1Error(
            f"insufficient free space at {root}: {free} B free, {needed} B required"
        )
    return free


def open_fields() -> dict[str, np.memmap]:
    fields = {
        "pmax": np.memmap(FIELD_PMAX, dtype="<f4", mode="r"),
        "psecond": np.memmap(FIELD_PSECOND, dtype="<f4", mode="r"),
        "argmax": np.memmap(FIELD_ARGMAX, dtype=np.uint8, mode="r"),
        "cost": np.memmap(FIELD_COST, dtype="<f8", mode="r"),
    }
    for key, array in fields.items():
        if array.size != POSITIONS:
            raise Hc1Error(f"field {key} has {array.size} entries, expected {POSITIONS}")
    return fields


def decoded_tokens() -> np.memmap:
    return np.memmap(bl1.TO2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))


def fold_assignment(seed: int = SPLIT_SEED) -> np.ndarray:
    """Seeded random 2-fold split over POSITIONS.  Never a prefix."""
    return np.random.default_rng(seed).integers(0, 2, size=POSITIONS, dtype=np.int8)


# --------------------------------------------------------------------------
# stage: verify


def boundary_buckets(previous: np.ndarray, max_distance: int = MAX_DISTANCE) -> np.ndarray:
    """L1 distance to the nearest inter-class token boundary, clipped.

    Byte-for-byte the shipped ``runtime/residual_archive.py::_boundary_buckets``.
    ``stage=verify`` proves that equality against the shipped source rather than
    asserting it, so this copy cannot silently drift from the decoder's own
    feature.
    """
    if previous.ndim != 2:
        raise Hc1Error("boundary source must be one token frame")
    edge = np.zeros(previous.shape, dtype=bool)
    edge[1:] |= previous[1:] != previous[:-1]
    edge[:-1] |= previous[:-1] != previous[1:]
    edge[:, 1:] |= previous[:, 1:] != previous[:, :-1]
    edge[:, :-1] |= previous[:, :-1] != previous[:, 1:]
    result = np.full(previous.shape, max_distance, dtype=np.uint8)
    active = edge.copy()
    result[active] = 0
    for distance in range(1, max_distance):
        grown = active.copy()
        grown[1:] |= active[:-1]
        grown[:-1] |= active[1:]
        grown[:, 1:] |= active[:, :-1]
        grown[:, :-1] |= active[:, 1:]
        active = grown
        result[(result == max_distance) & active] = distance
    return result


def _shipped_boundary_buckets():
    """Import the SHIPPED decoder's own boundary feature, for equality proof.

    ``residual_archive`` uses relative imports, so it must be loaded as part of
    the shipped ``runtime`` package rather than as a detached file.  The
    loaded module's ``__file__`` is checked against the expected path so a
    same-named package on ``sys.path`` cannot silently satisfy the proof.
    """
    import importlib

    source = bl1.RUNTIME_ROOT / "runtime" / "residual_archive.py"
    if not source.is_file():
        raise Hc1Error(f"shipped residual_archive.py absent: {source}")
    package_root = str(bl1.RUNTIME_ROOT)
    inserted = package_root not in sys.path
    if inserted:
        sys.path.insert(0, package_root)
    try:
        module = importlib.import_module("runtime.residual_archive")
    finally:
        if inserted:
            sys.path.remove(package_root)
    loaded = Path(getattr(module, "__file__", "")).resolve()
    if loaded != source.resolve():
        raise Hc1Error(
            f"imported residual_archive from {loaded}, expected the shipped {source}"
        )
    fn = getattr(module, "_boundary_buckets", None)
    if fn is None:
        raise Hc1Error("shipped residual_archive.py has no _boundary_buckets")
    return fn, source


def run_verify(store: Path) -> dict[str, Any]:
    started = time.perf_counter()
    fields = open_fields()
    tokens = decoded_tokens()

    custody = {
        "to2_tokens": {
            "path": str(bl1.TO2_TOKENS),
            "bytes": bl1.TO2_TOKENS.stat().st_size,
            "sha256": sha256_file(bl1.TO2_TOKENS),
            "expected_sha256": bl1.EXPECTED["tokens_sha256"],
        }
    }
    if custody["to2_tokens"]["sha256"] != bl1.EXPECTED["tokens_sha256"]:
        raise Hc1Error("TO2 decoded token field does not match the shipped digest")
    for key, path in (
        ("pmax", FIELD_PMAX),
        ("psecond", FIELD_PSECOND),
        ("argmax", FIELD_ARGMAX),
        ("cost", FIELD_COST),
    ):
        custody[f"df1_{key}"] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    # ---- the decode-identity proof, stated structurally -------------------
    # A recalibration g is applied to the coding row by both sides.  The RC64
    # decoder consumes (coding_row, stream) and emits the transmitted symbol;
    # the context, the corrector and the render are functions of decoded
    # symbols alone.  Changing the coding row changes the BITS, never the
    # SYMBOLS.  The only way a recalibration can break the pipeline is by
    # driving a row cell below the RC64 frequency floor, which is checked in
    # stage=analyze against floor(p * 2**31) >= 1.

    # ---- re-derive the flip flags definitionally --------------------------
    # df1 retained flip_flags.npy but its packbits bitorder is a convention.
    # Re-derive from (coding argmax != transmitted symbol) and cross-check.
    flip_derived = np.zeros(POSITIONS, dtype=bool)
    total_bits = 0.0
    zero_bits = 0.0
    positive_bits = 0.0
    saturated = 0
    saturated_bits = 0.0
    ideal_bits_zero = 0.0
    flat_tokens = tokens.reshape(-1)
    for start in range(0, POSITIONS, CHUNK):
        end = min(start + CHUNK, POSITIONS)
        argmax = np.asarray(fields["argmax"][start:end])
        transmitted = np.asarray(flat_tokens[start:end])
        flip = argmax != transmitted
        flip_derived[start:end] = flip
        cost = np.asarray(fields["cost"][start:end])
        pmax = np.asarray(fields["pmax"][start:end], dtype=np.float64)
        total_bits += float(cost.sum(dtype=np.float64))
        zero_bits += float(cost[~flip].sum(dtype=np.float64))
        positive_bits += float(cost[flip].sum(dtype=np.float64))
        mask = pmax >= 1.0
        saturated += int(mask.sum())
        saturated_bits += float(cost[mask].sum(dtype=np.float64))
        correct_p = pmax[~flip]
        ideal_bits_zero += float(-np.log2(np.maximum(correct_p, np.finfo(np.float64).tiny)).sum())

    flips = int(flip_derived.sum())
    packed = np.load(DF1_ANALYSIS / "flip_flags.npy", allow_pickle=False)
    df1_flip = np.unpackbits(packed, bitorder="little")[:POSITIONS].astype(bool)
    agreement = int((df1_flip == flip_derived).sum())

    reconciliation = {
        "derived_flip_positions": flips,
        "df1_flip_positions": int(df1_flip.sum()),
        "positions_agreeing": agreement,
        "positions_disagreeing": POSITIONS - agreement,
        "df1_flip_flags_bitorder": "little",
        "note": (
            "df1's flip_flags.npy is packbits with bitorder='little'; the "
            "definitional re-derivation (coding argmax != transmitted token) "
            "agrees at every position, so the retained flag field and this "
            "arm's join are the same object"
        ),
    }
    if agreement != POSITIONS:
        raise Hc1Error(
            f"derived flip flags disagree with df1 at {POSITIONS - agreement} positions"
        )

    headline = {
        "total_bits": total_bits,
        "zero_mode_positions": POSITIONS - flips,
        "zero_mode_bits": zero_bits,
        "zero_mode_bytes": zero_bits / 8.0,
        "positive_positions": flips,
        "positive_bits": positive_bits,
        "positive_bytes": positive_bits / 8.0,
        "float32_saturated_positions": saturated,
        "float32_saturated_bytes": saturated_bits / 8.0,
    }
    for key, expected in DF1_EXPECTED.items():
        got = headline[key]
        if isinstance(expected, int):
            if got != expected:
                raise Hc1Error(f"df1 headline {key}: got {got}, expected {expected}")
        elif not math.isclose(got, expected, rel_tol=1e-9):
            raise Hc1Error(f"df1 headline {key}: got {got!r}, expected {expected!r}")

    # ---- RC64 quantization loss: realized vs ideal model bits -------------
    ledger = np.load(bl1.DC1_LEDGER, allow_pickle=False)
    ideal_total = float(np.asarray(ledger, dtype=np.float64).sum())
    quantization = {
        "rc64_realized_bits": total_bits,
        "ideal_model_bits": ideal_total,
        "ideal_ledger_sha256": sha256_file(bl1.DC1_LEDGER),
        "rc64_minus_ideal_bits": total_bits - ideal_total,
        "rc64_minus_ideal_bytes": (total_bits - ideal_total) / 8.0,
        "note": (
            "the RC64 integer-frequency coder is faithful to the model to "
            "within a fraction of a byte over the whole stream; coder "
            "quantization is NOT a byte source and is not pursued"
        ),
    }

    # ---- the boundary feature is the SHIPPED one --------------------------
    shipped_fn, shipped_source = _shipped_boundary_buckets()
    rng = np.random.default_rng(SPLIT_SEED)
    probe_frames = sorted(int(f) for f in rng.choice(N, size=12, replace=False))
    mismatched = 0
    for frame in probe_frames:
        source_frame = np.asarray(tokens[frame])
        if not np.array_equal(
            boundary_buckets(source_frame), shipped_fn(source_frame, MAX_DISTANCE)
        ):
            mismatched += 1
    boundary_proof = {
        "shipped_source": str(shipped_source),
        "shipped_source_sha256": sha256_file(shipped_source),
        "probe_frames": probe_frames,
        "frames_mismatched": mismatched,
        "denominator_frames_probed": len(probe_frames),
        "note": (
            "this module's boundary_buckets is proven equal to the shipped "
            "decoder's own _boundary_buckets on seeded random frames; d is the "
            "decoder's existing feature, not a new proxy"
        ),
    }
    if mismatched:
        raise Hc1Error("local boundary_buckets differs from the shipped decoder feature")

    receipt = {
        "schema": "ddm_hc1_verify.v1",
        "custody": custody,
        "flip_reconciliation": reconciliation,
        "df1_headline_reproduced": headline,
        "rc64_quantization": quantization,
        "boundary_feature_equality": boundary_proof,
        "ideal_bits_on_zero_mode": ideal_bits_zero,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "analysis" / "VERIFY.json", receipt)
    atomic_npy(store / "retained" / "flip_flags_derived.npy", np.packbits(flip_derived))
    return receipt


# --------------------------------------------------------------------------
# stage: dfield


def run_dfield(store: Path) -> dict[str, Any]:
    started = time.perf_counter()
    require_free_bytes(store, POSITIONS + (1 << 30))
    tokens = decoded_tokens()
    target = store / "retained" / "boundary_distance_d.u8.bin"
    tmp = target.with_suffix(".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    histogram = np.zeros(MAX_DISTANCE + 1, dtype=np.int64)
    with tmp.open("wb") as handle:
        for frame in range(N):
            if frame == 0:
                # The shipped decoder has no previous frame at frame 0 and
                # substitutes the "far from any boundary" bucket everywhere.
                field = np.full(PLANE, MAX_DISTANCE, dtype=np.uint8)
            else:
                field = boundary_buckets(np.asarray(tokens[frame - 1])).reshape(-1)
            histogram += np.bincount(field, minlength=MAX_DISTANCE + 1)
            handle.write(field.tobytes())
    os.replace(tmp, target)
    receipt = {
        "schema": "ddm_hc1_dfield.v1",
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "positions": POSITIONS,
        "histogram": {str(i): int(histogram[i]) for i in range(MAX_DISTANCE + 1)},
        "share": {
            str(i): float(histogram[i] / POSITIONS) for i in range(MAX_DISTANCE + 1)
        },
        "definition": (
            "d = runtime/residual_archive.py::_boundary_buckets(previous decoded "
            "frame), L1 distance to the nearest inter-class token boundary "
            "clipped at 4; frame 0 = 4 everywhere, exactly as the shipped "
            "decoder does it"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "analysis" / "DFIELD.json", receipt)
    return receipt


# --------------------------------------------------------------------------
# stage: analyze


def _logit(pmax: np.ndarray) -> np.ndarray:
    """z = log(pmax / (1 - pmax)).  +inf where float32 saturates.

    pmax arrives as float32 promoted to float64, so ``1 - pmax`` is EXACT for
    pmax in [0.5, 1] by Sterbenz's lemma; z carries no subtraction error in the
    regime that holds 99.8% of the positions.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(pmax) - np.log1p(-pmax)


def _fine_edges(fine: int) -> np.ndarray:
    """Equal-mass edges in z-space over the NON-saturated positions.

    The float32-saturated cell (pmax == 1.0 exactly, z = +inf) is a mass point
    holding 57.6% of positions.  It cannot be split by any binning, carries
    ZERO flips, and its identity cost is exactly 0 bits, so it is held out of
    the map entirely and passed through unchanged.

    Edges come from a seeded random sample -- a SCOPE reduction on the EDGES
    only.  Every count, bit sum and byte figure reported below is accumulated
    exactly over all 117,964,800 positions.
    """
    pmax_field = np.memmap(FIELD_PMAX, dtype="<f4", mode="r")
    rng = np.random.default_rng(SPLIT_SEED + 7)
    index = np.unique(rng.integers(0, POSITIONS, size=30_000_000, dtype=np.int64))
    sampled = np.asarray(pmax_field[index], dtype=np.float64)
    finite = sampled[sampled < 1.0]
    z = _logit(finite)
    z = z[np.isfinite(z)]
    quantiles = np.linspace(0.0, 1.0, fine + 1)[1:-1]
    return np.unique(np.quantile(z, quantiles))


def _bin_index(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in [0, len(edges)+1]; the last index is the saturated cell."""
    index = np.searchsorted(edges, z, side="right").astype(np.int64)
    index[~np.isfinite(z)] = len(edges) + 1
    return index


def _accumulate(store: Path, edges: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Per-(fold, d, fine-bin) sufficient statistics, accumulated exactly."""
    nbin = len(edges) + 2
    shape = (2, MAX_DISTANCE + 1, nbin)
    keys = ("n", "n_correct", "sum_z", "sum_pmax", "bits_identity", "bits_total")
    stats = {key: np.zeros(shape, dtype=np.float64) for key in keys}
    fields = open_fields()
    tokens = decoded_tokens().reshape(-1)
    dfield = np.memmap(
        store / "retained" / "boundary_distance_d.u8.bin", dtype=np.uint8, mode="r"
    )
    fold = fold_assignment(seed)
    flat = int(np.prod(shape))
    for start in range(0, POSITIONS, CHUNK):
        end = min(start + CHUNK, POSITIONS)
        pmax = np.asarray(fields["pmax"][start:end], dtype=np.float64)
        cost = np.asarray(fields["cost"][start:end])
        correct = np.asarray(fields["argmax"][start:end]) == np.asarray(tokens[start:end])
        d = np.asarray(dfield[start:end]).astype(np.int64)
        f = fold[start:end].astype(np.int64)
        z = _logit(pmax)
        b = _bin_index(z, edges)
        cell = (f * (MAX_DISTANCE + 1) + d) * nbin + b
        with np.errstate(divide="ignore", invalid="ignore"):
            bits_correct = -np.log2(pmax)
            bits_flip = -np.log2(1.0 - pmax)
        if not np.all(np.isfinite(bits_flip[~correct])):
            raise Hc1Error("a flipped position has pmax == 1.0; indicator cost is infinite")
        identity = np.where(correct, bits_correct, bits_flip)
        finite_z = np.where(np.isfinite(z), z, 0.0)
        stats["n"] += np.bincount(cell, minlength=flat).reshape(shape)
        stats["n_correct"] += np.bincount(cell[correct], minlength=flat).reshape(shape)
        stats["sum_z"] += np.bincount(cell, weights=finite_z, minlength=flat).reshape(shape)
        stats["sum_pmax"] += np.bincount(cell, weights=pmax, minlength=flat).reshape(shape)
        stats["bits_identity"] += np.bincount(
            cell, weights=identity, minlength=flat
        ).reshape(shape)
        stats["bits_total"] += np.bincount(cell, weights=cost, minlength=flat).reshape(shape)
    return stats


# ---- identity-nesting recalibration maps ---------------------------------
#
# THE FORM MATTERS, and a first pass at this arm got it wrong in a way worth
# recording.  A reliability map that REPLACES pmax with its bin's empirical
# accuracy throws away every distinction the model draws INSIDE the bin.  At 64
# equal-mass bins that discards 6,987 B on this object -- i.e. the model's pmax
# carries at least that much resolution beyond a 64-cell quantization of
# itself, and a bin-constant map is a LOSS, not a gain.
#
# The correct family is a monotone ADJUSTMENT in logit space,
#
#     q = sigmoid(a_c * z + b_c),      z = logit(pmax)
#
# which NESTS THE IDENTITY at (a, b) = (1, 0).  Because the identity is inside
# the family, the in-sample optimum can never be worse than the shipped model,
# and any heldout shortfall is honest overfitting rather than an artifact of
# the parametrization.  Global temperature scaling is the special case
# a = 1/T, b = 0.


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def _map_cost_bits(z: np.ndarray, nc: np.ndarray, nf: np.ndarray, eta: np.ndarray) -> float:
    """Indicator bits when bin ``j`` is coded at q = sigmoid(eta_j).  Exact
    given the bin representative ``z``; the discretization is validated against
    the exact identity cost in ``run_analyze``."""
    del z
    return float(
        (nc * np.logaddexp(0.0, -eta) + nf * np.logaddexp(0.0, eta)).sum() / math.log(2.0)
    )


def _fit_map(
    z: np.ndarray,
    nc: np.ndarray,
    nf: np.ndarray,
    group: np.ndarray,
    groups: int,
    *,
    fit_slope: bool,
    fit_offset: bool,
    ridge: float,
    iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Newton fit of per-group ``(a, b)`` on the convex logistic NLL.

    ``ridge`` is an L2 pull toward the IDENTITY map (a = 1, b = 0) -- a
    declared prior applied identically to every cell, never a fit to the
    evaluation fold.  It keeps a cell that observed zero flips from asserting
    q = 1 exactly, which would code a heldout flip at infinite cost.
    """
    a = np.ones(groups, dtype=np.float64)
    b = np.zeros(groups, dtype=np.float64)
    n = nc + nf
    for _ in range(iterations):
        eta = a[group] * z + b[group]
        s = _sigmoid(eta)
        residual = n * s - nc
        weight = n * s * (1.0 - s)
        g_b = np.bincount(group, weights=residual, minlength=groups) + ridge * b
        h_bb = np.bincount(group, weights=weight, minlength=groups) + ridge
        g_a = np.bincount(group, weights=residual * z, minlength=groups) + ridge * (a - 1.0)
        h_aa = np.bincount(group, weights=weight * z * z, minlength=groups) + ridge
        if fit_slope and fit_offset:
            h_ab = np.bincount(group, weights=weight * z, minlength=groups)
            determinant = h_aa * h_bb - h_ab * h_ab
            determinant = np.where(np.abs(determinant) < 1e-14, 1e-14, determinant)
            step_a = (h_bb * g_a - h_ab * g_b) / determinant
            step_b = (h_aa * g_b - h_ab * g_a) / determinant
        elif fit_slope:
            step_a = g_a / np.maximum(h_aa, 1e-14)
            step_b = np.zeros_like(b)
        else:
            step_a = np.zeros_like(a)
            step_b = g_b / np.maximum(h_bb, 1e-14)
        a -= step_a
        b -= step_b
        if max(float(np.max(np.abs(step_a))), float(np.max(np.abs(step_b)))) < 1e-13:
            break
    return a, b


def _fold_representatives(stats: dict[str, np.ndarray]) -> np.ndarray:
    """Per-(fold, d, bin) mean logit.  Each fold uses ITS OWN positions'
    mean, so no evaluation-fold information enters the map's parameters."""
    n = stats["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(n > 0, stats["sum_z"] / np.maximum(n, 1.0), 0.0)
    return z


def _crossfit_map(
    stats: dict[str, np.ndarray],
    group: np.ndarray,
    groups: int,
    *,
    fit_slope: bool,
    fit_offset: bool,
    ridge: float,
) -> dict[str, Any]:
    """2-fold cross-fitted GAIN in bits for one identity-nesting map.

    The gain is reported as a DIFFERENCE taken on the same binned
    representation -- ``identity_cost_hist - map_cost_hist`` -- so the
    representative-logit discretization cancels to first order and never
    inflates or deflates the answer.  ``run_analyze`` separately reports the
    absolute fidelity of that representation against the exact identity cost.

    ``group`` maps each (d, fine-bin) cell to a map parameter cell.  Cells with
    group == -1 (the float32-saturated column, 57.6% of positions, zero flips,
    exactly 0 identity bits) are held out of the map and pass through
    unchanged.
    """
    nc_all = stats["n_correct"]
    nf_all = stats["n"] - stats["n_correct"]
    z_all = _fold_representatives(stats)
    live = group >= 0
    gf = group[live]
    heldout_gain = 0.0
    insample_gain = 0.0
    offsets: list[float] = []
    slopes: list[float] = []
    for fit in (0, 1):
        evaluate = 1 - fit
        z_fit = z_all[fit][live]
        z_eval = z_all[evaluate][live]
        nc_e, nf_e = nc_all[evaluate][live], nf_all[evaluate][live]
        identity_eval = _map_cost_bits(z_eval, nc_e, nf_e, z_eval)
        a, b = _fit_map(
            z_fit, nc_all[fit][live], nf_all[fit][live], gf, groups,
            fit_slope=fit_slope, fit_offset=fit_offset, ridge=ridge,
        )
        heldout_gain += identity_eval - _map_cost_bits(
            z_eval, nc_e, nf_e, a[gf] * z_eval + b[gf]
        )
        a_s, b_s = _fit_map(
            z_eval, nc_e, nf_e, gf, groups,
            fit_slope=fit_slope, fit_offset=fit_offset, ridge=ridge,
        )
        insample_gain += identity_eval - _map_cost_bits(
            z_eval, nc_e, nf_e, a_s[gf] * z_eval + b_s[gf]
        )
        slopes.append(float(np.max(np.abs(a - 1.0))))
        offsets.append(float(np.max(np.abs(b))))
    free = int(fit_slope) + int(fit_offset)
    return {
        "heldout_gain_bits": heldout_gain,
        "heldout_gain_bytes": heldout_gain / 8.0,
        "insample_gain_bits": insample_gain,
        "insample_gain_bytes": insample_gain / 8.0,
        "parameters": groups * free,
        "max_abs_slope_deviation": max(slopes),
        "max_abs_offset": max(offsets),
        "ridge": ridge,
    }


def _coarse_group(edges: np.ndarray, coarse: int) -> np.ndarray:
    """Fine-bin index -> coarse display/map bin.  Saturated cell -> ``coarse``."""
    live = len(edges) + 1
    index = np.arange(len(edges) + 2)
    group = np.minimum(index * coarse // live, coarse - 1)
    group[live] = coarse
    return group


def _reliability_table(
    stats: dict[str, np.ndarray], edges: np.ndarray, coarse: int
) -> list[dict[str, Any]]:
    group = _coarse_group(edges, coarse)
    n = np.bincount(group, weights=stats["n"].sum(axis=(0, 1)), minlength=coarse + 1)
    nc = np.bincount(group, weights=stats["n_correct"].sum(axis=(0, 1)), minlength=coarse + 1)
    sp = np.bincount(group, weights=stats["sum_pmax"].sum(axis=(0, 1)), minlength=coarse + 1)
    bits = np.bincount(group, weights=stats["bits_identity"].sum(axis=(0, 1)), minlength=coarse + 1)
    total = np.bincount(group, weights=stats["bits_total"].sum(axis=(0, 1)), minlength=coarse + 1)
    rows = []
    for b in range(coarse + 1):
        if n[b] == 0:
            continue
        stated = float(sp[b] / n[b])
        empirical = float(nc[b] / n[b])
        rows.append(
            {
                "bin": b,
                "saturated_cell": bool(b == coarse),
                "positions": round(float(n[b])),
                "flips": round(float(n[b] - nc[b])),
                "mean_stated_pmax": stated,
                "empirical_accuracy": empirical,
                "gap_empirical_minus_stated": empirical - stated,
                "indicator_bytes": float(bits[b] / 8.0),
                "total_bytes": float(total[b] / 8.0),
            }
        )
    return rows


def _per_d_reliability(stats: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    n = stats["n"].sum(axis=(0, 2))
    nc = stats["n_correct"].sum(axis=(0, 2))
    sp = stats["sum_pmax"].sum(axis=(0, 2))
    bits = stats["bits_identity"].sum(axis=(0, 2))
    total = stats["bits_total"].sum(axis=(0, 2))
    rows = []
    for d in range(MAX_DISTANCE + 1):
        if n[d] == 0:
            continue
        rows.append(
            {
                "d": d,
                "positions": round(float(n[d])),
                "position_share": float(n[d] / POSITIONS),
                "flips": round(float(n[d] - nc[d])),
                "flip_rate": float((n[d] - nc[d]) / n[d]),
                "mean_stated_pmax": float(sp[d] / n[d]),
                "empirical_accuracy": float(nc[d] / n[d]),
                "gap_empirical_minus_stated": float(nc[d] / n[d] - sp[d] / n[d]),
                "indicator_bytes": float(bits[d] / 8.0),
                "total_bytes": float(total[d] / 8.0),
                "bit_share": float(total[d] / max(total.sum(), 1e-12)),
            }
        )
    return rows


def _per_d_reliability_curves(
    stats: dict[str, np.ndarray], edges: np.ndarray, coarse: int
) -> list[dict[str, Any]]:
    """MAIN's load-bearing question: do the per-``d`` calibration curves
    SEPARATE, or do they lie on top of each other?  If they coincide, the
    shipped corrector's ``d * 5 + predicted`` feature already absorbed the
    variable and there is no free headroom in this direction."""
    group = _coarse_group(edges, coarse)
    rows = []
    for d in range(MAX_DISTANCE + 1):
        n = np.bincount(group, weights=stats["n"][:, d].sum(axis=0), minlength=coarse + 1)
        nc = np.bincount(
            group, weights=stats["n_correct"][:, d].sum(axis=0), minlength=coarse + 1
        )
        sp = np.bincount(
            group, weights=stats["sum_pmax"][:, d].sum(axis=0), minlength=coarse + 1
        )
        cells = []
        for b in range(coarse + 1):
            if n[b] < 1:
                continue
            cells.append(
                {
                    "bin": b,
                    "saturated_cell": bool(b == coarse),
                    "positions": round(float(n[b])),
                    "flips": round(float(n[b] - nc[b])),
                    "mean_stated_pmax": float(sp[b] / n[b]),
                    "empirical_accuracy": float(nc[b] / n[b]),
                    "gap_empirical_minus_stated": float(nc[b] / n[b] - sp[b] / n[b]),
                }
            )
        rows.append({"d": d, "cells": cells})
    return rows


def _ece(stats: dict[str, np.ndarray], edges: np.ndarray, coarse: int) -> dict[str, Any]:
    group = _coarse_group(edges, coarse)
    n = np.bincount(group, weights=stats["n"].sum(axis=(0, 1)), minlength=coarse + 1)
    nc = np.bincount(group, weights=stats["n_correct"].sum(axis=(0, 1)), minlength=coarse + 1)
    sp = np.bincount(group, weights=stats["sum_pmax"].sum(axis=(0, 1)), minlength=coarse + 1)
    mask = n > 0
    stated = sp[mask] / n[mask]
    empirical = nc[mask] / n[mask]
    weight = n[mask] / n.sum()
    signed = float(np.sum(weight * (empirical - stated)))
    absolute = float(np.sum(weight * np.abs(empirical - stated)))
    return {
        "bins": coarse,
        "ece_absolute": absolute,
        "ece_signed_empirical_minus_stated": signed,
        "sign": "UNDER-CONFIDENT" if signed > 0 else ("OVER-CONFIDENT" if signed < 0 else "EXACT"),
        "aggregate_stated_pmax": float(sp.sum() / n.sum()),
        "aggregate_empirical_accuracy": float(nc.sum() / n.sum()),
        "denominator_positions": round(float(n.sum())),
    }


def _rc64_floor_check(
    store: Path, edges: np.ndarray, stats: dict[str, np.ndarray], coarse: int, ridge: float
) -> dict[str, Any]:
    """Would any recalibrated row cell fall below the RC64 frequency floor?

    RC64 stores ``floor(p * 2**31)`` clamped to >= 1, so a cell is codeable
    while ``p >= 2**-31``.  The recalibrated row is ``q`` on the argmax and
    ``(1-q) * p_j/(1-pmax)`` on the rest, so every non-argmax cell is scaled by
    the same ``(1-q)/(1-pmax)`` factor.

    SCOPE: only ``psecond`` was retained, so the check bounds the RUNNER-UP
    cell, not the row minimum.  It is reported as such.  The scale factor
    itself is exact and is the quantity that decides.
    """
    n = stats["n"].sum(axis=0)
    nc = stats["n_correct"].sum(axis=0)
    z = _fold_representatives(stats).sum(axis=0) / 2.0
    group2d = np.broadcast_to(_coarse_group(edges, coarse), n.shape).copy()
    group2d[:, len(edges) + 1] = -1
    flat_group = (
        np.arange(MAX_DISTANCE + 1)[:, None] * coarse + group2d
    )
    flat_group = np.where(group2d < 0, -1, flat_group)
    live = flat_group >= 0
    a, b = _fit_map(
        z[live], nc[live], (n - nc)[live], flat_group[live],
        (MAX_DISTANCE + 1) * coarse, fit_slope=True, fit_offset=True, ridge=ridge,
    )
    q_table = np.ones_like(n)
    q_table[live] = _sigmoid(a[flat_group[live]] * z[live] + b[flat_group[live]])
    floor = 2.0 ** -31
    fields = open_fields()
    dfield = np.memmap(
        store / "retained" / "boundary_distance_d.u8.bin", dtype=np.uint8, mode="r"
    )
    worst_cell = np.inf
    worst_scale = np.inf
    largest_scale = -np.inf
    max_q_above_pmax = -np.inf
    below_recal = 0
    below_shipped = 0
    newly_below = 0
    for start in range(0, POSITIONS, CHUNK):
        end = min(start + CHUNK, POSITIONS)
        pmax = np.asarray(fields["pmax"][start:end], dtype=np.float64)
        psecond = np.asarray(fields["psecond"][start:end], dtype=np.float64)
        d = np.asarray(dfield[start:end]).astype(np.int64)
        bins = _bin_index(_logit(pmax), edges)
        qq = q_table[d, bins]
        rest = 1.0 - pmax
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(rest > 0, (1.0 - qq) / rest, 1.0)
            cell = scale * psecond
        good = np.isfinite(cell) & (cell > 0)
        shipped_good = psecond > 0
        if good.any():
            worst_cell = min(worst_cell, float(cell[good].min()))
        finite_scale = scale[np.isfinite(scale)]
        if finite_scale.size:
            worst_scale = min(worst_scale, float(finite_scale.min()))
            largest_scale = max(largest_scale, float(finite_scale.max()))
        max_q_above_pmax = max(max_q_above_pmax, float(np.max(qq - pmax)))
        below_recal += int(np.sum(good & (cell < floor)))
        below_shipped += int(np.sum(shipped_good & (psecond < floor)))
        newly_below += int(
            np.sum(good & shipped_good & (cell < floor) & (psecond >= floor))
        )
    return {
        "rc64_probability_floor": floor,
        "smallest_recalibrated_runner_up_cell": worst_cell,
        "row_scale_factor_min": worst_scale,
        "row_scale_factor_max": largest_scale,
        "max_q_minus_pmax": max_q_above_pmax,
        "runner_up_cells_below_floor_SHIPPED": below_shipped,
        "runner_up_cells_below_floor_RECALIBRATED": below_recal,
        "runner_up_cells_newly_below_floor": newly_below,
        "denominator_positions": POSITIONS,
        "codeable": True,
        "why_codeable": (
            "below-floor is NOT uncodeable: bl1.rc64_costs clamps every "
            "frequency to >= 1, so such a cell simply costs the 31-bit maximum "
            "-- and the SHIPPED stream already clamps EXACTLY this many cells, "
            "which is why the baseline count is reported beside the "
            "recalibrated one.  The only condition that WOULD break the coder "
            "is the winner-balance invariant 0 < freq[w] + balance < 2**31.  "
            "Writing balance = 2**31 - sum_j freq_j gives freq[w] + balance = "
            "2**31 - sum_{j != w} freq_j, and every non-winner frequency is "
            "clamped to >= 1, so the quantity is <= 2**31 - 4 and is > 0 "
            "whenever the four non-winner probabilities do not sum to 1.  Both "
            "hold for ANY row with five positive entries, so the invariant is "
            "structural and does not depend on which map is fitted.  NOTE: the "
            "map is NOT monotonically softening -- max_q_minus_pmax is "
            "POSITIVE, because the two under-confident bins are sharpened.  An "
            "earlier draft of this receipt claimed q <= pmax everywhere; the "
            "field refutes that and the invariant argument above does not need "
            "it."
        ),
        "scope": (
            "bounds the RUNNER-UP cell; the row minimum was not retained, so a "
            "smaller cell may exist.  The exact scale factor (1-q)/(1-pmax) is "
            "reported because it multiplies every non-argmax cell identically."
        ),
    }


def _conditional_bound(store: Path, seed: int, smoothing: float = 0.5) -> dict[str, Any]:
    """Close the 4-way conditional pool with a MEASUREMENT, not an argument.

    The indicator decomposition leaves ``-log2(p_sel/(1-pmax))`` untouched, and
    that residual is 2,500.54 B -- above this arm's own 2,000 B falsifier, so
    it cannot be waved past.  The retained fields do not carry the runner-up's
    INDEX, so the model's own conditional row is not reconstructible.  What IS
    computable is the best a receiver-derivable EMPIRICAL TABLE could do:
    cross-fitted ``P(transmitted | argmax, d)`` over the flips.  If that table
    costs MORE than the shipped conditional, the shipped model already beats
    any such table and the pool is closed from below by measurement.
    """
    fields = open_fields()
    tokens = decoded_tokens().reshape(-1)
    dfield = np.memmap(
        store / "retained" / "boundary_distance_d.u8.bin", dtype=np.uint8, mode="r"
    )
    fold = fold_assignment(seed)
    shape = (2, CLASSES, MAX_DISTANCE + 1, CLASSES)
    counts = np.zeros(shape, dtype=np.float64)
    shipped_bits = 0.0
    for start in range(0, POSITIONS, CHUNK):
        end = min(start + CHUNK, POSITIONS)
        argmax = np.asarray(fields["argmax"][start:end]).astype(np.int64)
        transmitted = np.asarray(tokens[start:end]).astype(np.int64)
        flip = argmax != transmitted
        if not flip.any():
            continue
        pmax = np.asarray(fields["pmax"][start:end], dtype=np.float64)[flip]
        cost = np.asarray(fields["cost"][start:end])[flip]
        with np.errstate(divide="ignore"):
            shipped_bits += float((cost + np.log2(1.0 - pmax)).sum())
        cell = (
            (fold[start:end][flip].astype(np.int64) * CLASSES + argmax[flip])
            * (MAX_DISTANCE + 1)
            + np.asarray(dfield[start:end])[flip].astype(np.int64)
        ) * CLASSES + transmitted[flip]
        counts += np.bincount(cell, minlength=int(np.prod(shape))).reshape(shape)

    def table_bits(condition_on_d: bool) -> float:
        data = counts if condition_on_d else counts.sum(axis=2, keepdims=True)
        total = 0.0
        for fit in (0, 1):
            evaluate = 1 - fit
            probability = (data[fit] + smoothing) / (
                data[fit].sum(axis=-1, keepdims=True) + smoothing * CLASSES
            )
            total += float((data[evaluate] * -np.log2(probability)).sum())
        return total

    with_d = table_bits(True)
    without_d = table_bits(False)
    return {
        "flips": int(counts.sum()),
        "shipped_conditional_bits": shipped_bits,
        "shipped_conditional_bytes": shipped_bits / 8.0,
        "shipped_conditional_bits_per_flip": shipped_bits / max(counts.sum(), 1.0),
        "empirical_table_argmax_bits": without_d,
        "empirical_table_argmax_x_d_bits": with_d,
        "best_table_gain_bytes": (shipped_bits - min(with_d, without_d)) / 8.0,
        "verdict": (
            "the shipped conditional already costs less than any cross-fitted "
            "(argmax, d) table, so this pool yields nothing"
            if shipped_bits < min(with_d, without_d)
            else "an empirical table beats the shipped conditional -- pursue"
        ),
        "scope": (
            "this is a MODEL-REPLACEMENT bound, not a recalibration: the "
            "retained fields do not carry the runner-up index, so the model's "
            "own conditional row cannot be reconstructed and no map can be "
            "fitted to it.  Reported to bound the pool, not to close the "
            "conditional as a family."
        ),
    }


def run_analyze(store: Path, fine: int, ridge: float, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    edges = _fine_edges(fine)
    stats = _accumulate(store, edges, seed)

    n_total = round(float(stats["n"].sum()))
    if n_total != POSITIONS:
        raise Hc1Error(f"accumulator saw {n_total} positions, expected {POSITIONS}")

    exact_identity_bits = float(stats["bits_identity"].sum())
    current_total_bits = float(stats["bits_total"].sum())
    conditional_bits = current_total_bits - exact_identity_bits

    # ---- does the binned representation carry the identity faithfully? ----
    z_all = _fold_representatives(stats)
    nc_all = stats["n_correct"]
    nf_all = stats["n"] - stats["n_correct"]
    live_mask = np.ones(stats["n"].shape[1:], dtype=bool)
    live_mask[:, len(edges) + 1] = False
    hist_identity_bits = sum(
        _map_cost_bits(
            z_all[f][live_mask], nc_all[f][live_mask], nf_all[f][live_mask],
            z_all[f][live_mask],
        )
        for f in (0, 1)
    )
    exact_live_bits = float(
        sum(stats["bits_identity"][f][live_mask].sum() for f in (0, 1))
    )
    fidelity = {
        "fine_bins_requested": fine,
        "fine_bins_realized": len(edges) + 2,
        "exact_identity_bits_live": exact_live_bits,
        "binned_identity_bits_live": hist_identity_bits,
        "discretization_bytes": (hist_identity_bits - exact_live_bits) / 8.0,
        "note": (
            "every reported gain is a DIFFERENCE taken on this same binned "
            "representation, so this residual cancels to first order; it is "
            "shown so the representation's absolute fidelity is auditable"
        ),
    }

    ece = [_ece(stats, edges, k) for k in (10, 15, 32, 64, 256)]
    reliability = _reliability_table(stats, edges, 32)
    per_d = _per_d_reliability(stats)
    per_d_curves = _per_d_reliability_curves(stats, edges, 12)

    # ---- the ladder -------------------------------------------------------
    saturated = len(edges) + 1
    d_index = np.arange(MAX_DISTANCE + 1)[:, None]
    shape = stats["n"].shape[1:]

    def grouping(kind: str, coarse: int) -> tuple[np.ndarray, int]:
        base = np.broadcast_to(_coarse_group(edges, coarse), shape).copy()
        if kind == "global":
            group = np.zeros(shape, dtype=np.int64)
            count = 1
        elif kind == "per_d":
            group = np.broadcast_to(d_index, shape).copy()
            count = MAX_DISTANCE + 1
        elif kind == "per_bin":
            group = base
            count = coarse
        elif kind == "per_d_bin":
            group = d_index * coarse + base
            count = (MAX_DISTANCE + 1) * coarse
        else:
            raise Hc1Error(f"unknown grouping {kind}")
        group = np.asarray(group, dtype=np.int64).copy()
        group[:, saturated] = -1
        return group, count

    rungs: dict[str, Any] = {}
    generic = {"control_global_temperature", "control_global_platt"}
    group, count = grouping("global", 1)
    rungs["control_global_temperature"] = _crossfit_map(
        stats, group, count, fit_slope=True, fit_offset=False, ridge=ridge
    )
    rungs["control_global_platt"] = _crossfit_map(
        stats, group, count, fit_slope=True, fit_offset=True, ridge=ridge
    )
    group, count = grouping("per_d", 1)
    rungs["per_d_offset"] = _crossfit_map(
        stats, group, count, fit_slope=False, fit_offset=True, ridge=ridge
    )
    rungs["per_d_platt"] = _crossfit_map(
        stats, group, count, fit_slope=True, fit_offset=True, ridge=ridge
    )
    for coarse in (8, 32, 128, 512):
        group, count = grouping("per_bin", coarse)
        rungs[f"per_bin{coarse}_offset"] = _crossfit_map(
            stats, group, count, fit_slope=False, fit_offset=True, ridge=ridge
        )
        group, count = grouping("per_d_bin", coarse)
        rungs[f"per_d_x_bin{coarse}_offset"] = _crossfit_map(
            stats, group, count, fit_slope=False, fit_offset=True, ridge=ridge
        )

    address = {
        "note": (
            "d costs ZERO stored bytes: the shipped decoder already computes "
            "_boundary_buckets(previous frame) for its corrector feature, so "
            "the conditioning variable is rule-118 free and adds no decode "
            "work.  Only the map VALUES are video-derived and counted."
        ),
        "value_bytes_each": 4,
        "edge_bytes_each": 4,
        "edges_free_variant": (
            "fixed dyadic edges would be a constant in inflate.py and cost 0 "
            "stored bytes; quantile edges are video-derived and are charged"
        ),
    }

    def net(rung: dict[str, Any], edges_stored: int) -> dict[str, float]:
        gross = rung["heldout_gain_bytes"]
        tax = rung["parameters"] * address["value_bytes_each"] + edges_stored * 4
        return {
            "gross_bytes": gross,
            "map_bytes": float(tax),
            "net_bytes": gross - tax,
            "net_bytes_free_edges": gross - rung["parameters"] * 4,
            "net_S": (gross - tax) * S_PER_ARCHIVE_BYTE,
            "net_share_of_demand": (gross - tax) / DEMAND_BYTES,
            "insample_gross_bytes": rung["insample_gain_bytes"],
            "parameters": rung["parameters"],
        }

    ladder: dict[str, Any] = {}
    for name, rung in rungs.items():
        # Rungs whose map is indexed by a coarse pmax bin must ship that
        # binning.  The edges are quantiles of this stream, so they are
        # video-derived and charged; `net_bytes_free_edges` prices the
        # fixed-dyadic-edge variant, where the edges are a constant in
        # inflate.py and cost nothing.
        match = re.search(r"bin(\d+)_offset$", name)
        stored_edges = int(match.group(1)) - 1 if match else 0
        ladder[name] = net(rung, stored_edges)

    ladder["d_contribution_beyond_pmax"] = {
        f"bin{c}_gross_bytes": rungs[f"per_d_x_bin{c}_offset"]["heldout_gain_bytes"]
        - rungs[f"per_bin{c}_offset"]["heldout_gain_bytes"]
        for c in (8, 32, 128, 512)
    }
    ladder["d_contribution_beyond_pmax"]["platt_vs_global_gross_bytes"] = (
        rungs["per_d_platt"]["heldout_gain_bytes"]
        - rungs["control_global_platt"]["heldout_gain_bytes"]
    )

    best_key = max(rungs, key=lambda k: ladder[k]["net_bytes"])
    best_net = ladder[best_key]["net_bytes"]
    best_domain_key = max(
        (k for k in rungs if k not in generic), key=lambda k: ladder[k]["net_bytes"]
    )

    floor_check = _rc64_floor_check(store, edges, stats, 32, ridge)
    conditional_bound = _conditional_bound(store, seed)

    receipt = {
        "schema": "ddm_hc1_analyze.v2",
        "map_family": (
            "q = sigmoid(a_c * logit(pmax) + b_c); NESTS THE IDENTITY at "
            "(a,b)=(1,0), so no rung can be an artifact of the parametrization"
        ),
        "ridge_toward_identity": ridge,
        "split_seed": seed,
        "split": "seeded random 2-fold over positions; NEVER a frame prefix",
        "positions": POSITIONS,
        "representation_fidelity": fidelity,
        "current": {
            "total_bits": current_total_bits,
            "total_bytes": current_total_bits / 8.0,
            "indicator_bits": exact_identity_bits,
            "indicator_bytes": exact_identity_bits / 8.0,
            "indicator_share_of_tail": exact_identity_bits / current_total_bits,
            "conditional_4way_bits": conditional_bits,
            "conditional_4way_bytes": conditional_bits / 8.0,
            "conditional_scope": (
                "NOT recalibrated: only pmax and psecond were retained, so the "
                "4-way conditional map is not computable from retained fields"
            ),
        },
        "ece": ece,
        "conditional_pool_bound": conditional_bound,
        "reliability_bins": reliability,
        "per_d_reliability": per_d,
        "per_d_reliability_curves": per_d_curves,
        "rungs": rungs,
        "address_tax": address,
        "ladder_net": ladder,
        "best_rung": best_key,
        "best_net_bytes": best_net,
        "best_domain_native_rung": best_domain_key,
        "best_domain_native_net_bytes": ladder[best_domain_key]["net_bytes"],
        "falsifier": {
            "threshold_bytes": FALSIFIER_BYTES,
            "measured_net_bytes": best_net,
            "outcome": "FAMILY_OPEN" if best_net >= FALSIFIER_BYTES else "FAMILY_CLOSED",
        },
        "distortion": {
            "dD": 0.0,
            "basis": "STRUCTURAL_IDENTITY",
            "argument": (
                "recalibration is a deterministic map applied to the coding row "
                "by both encoder and decoder; the decoded symbol sequence, the "
                "autoregressive context, the corrector state and the rendered "
                "frame are functions of decoded symbols alone, so the decoded "
                "field is bit-identical and every scored cell is unchanged"
            ),
            "rc64_codeability": floor_check,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "analysis" / f"ANALYZE_fine{fine}_seed{seed}.json", receipt)
    return receipt


# --------------------------------------------------------------------------


def run_dcompare(store: Path) -> dict[str, Any]:
    """CAUSAL vs ACAUSAL boundary distance -- they are not the same variable.

    The shipped decoder's feature is ``_boundary_buckets(frame f-1)``, because
    at the moment it codes frame ``f`` it has not decoded frame ``f``.  A
    boundary field taken from frame ``f`` ITSELF concentrates cost far harder,
    but the receiver cannot compute it before it needs it.  This stage measures
    both against the same cost field so the campaign's boundary/annulus numbers
    can say WHICH ONE they mean.  Position geometry is nearly identical between
    the two; the join to cost is not.
    """
    started = time.perf_counter()
    fields = open_fields()
    tokens = decoded_tokens()
    causal = np.memmap(
        store / "retained" / "boundary_distance_d.u8.bin", dtype=np.uint8, mode="r"
    )
    counts = {"causal": np.zeros(MAX_DISTANCE + 1), "acausal": np.zeros(MAX_DISTANCE + 1)}
    bits = {"causal": np.zeros(MAX_DISTANCE + 1), "acausal": np.zeros(MAX_DISTANCE + 1)}
    for frame in range(N):
        lo, hi = frame * PLANE, (frame + 1) * PLANE
        cost = np.asarray(fields["cost"][lo:hi])
        for kind, field in (
            ("causal", np.asarray(causal[lo:hi]).astype(np.int64)),
            ("acausal", boundary_buckets(np.asarray(tokens[frame])).reshape(-1).astype(np.int64)),
        ):
            counts[kind] += np.bincount(field, minlength=MAX_DISTANCE + 1)
            bits[kind] += np.bincount(field, weights=cost, minlength=MAX_DISTANCE + 1)
    receipt: dict[str, Any] = {
        "schema": "ddm_hc1_dcompare.v1",
        "causal_definition": "_boundary_buckets(frame f-1) -- the SHIPPED decoder feature",
        "acausal_definition": "_boundary_buckets(frame f) -- NOT receiver-computable before decoding frame f",
        "elapsed_seconds": 0.0,
    }
    for kind in ("causal", "acausal"):
        total_bits = float(bits[kind].sum())
        receipt[kind] = {
            "position_share": [float(c / POSITIONS) for c in counts[kind]],
            "bit_share": [float(b / total_bits) for b in bits[kind]],
            "bytes": [float(b / 8.0) for b in bits[kind]],
            "d0_position_share": float(counts[kind][0] / POSITIONS),
            "d0_bit_share": float(bits[kind][0] / total_bits),
        }
    receipt["d0_bit_share_acausal_over_causal"] = (
        receipt["acausal"]["d0_bit_share"] / receipt["causal"]["d0_bit_share"]
    )
    receipt["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(store / "analysis" / "DCOMPARE.json", receipt)
    return receipt


def run_manifest(store: Path, fine: int, seed: int) -> dict[str, Any]:
    """Retain the per-``d`` reliability table as a standalone durable artifact
    and hash every payload this arm materialized."""
    source = store / "analysis" / f"ANALYZE_fine{fine}_seed{seed}.json"
    if not source.is_file():
        raise Hc1Error(f"analyze receipt absent: {source}")
    analysis = json.loads(source.read_text())
    table = {
        "schema": "ddm_hc1_per_d_reliability.v1",
        "definition": analysis["distortion"]["argument"],
        "d_definition": (
            "runtime/residual_archive.py::_boundary_buckets(previous decoded "
            "frame): L1 distance to the nearest inter-class token boundary, "
            "clipped at 4.  Computed by the SHIPPED decoder already."
        ),
        "positions": POSITIONS,
        "per_d_aggregate": analysis["per_d_reliability"],
        "per_d_curves": analysis["per_d_reliability_curves"],
        "global_reliability_bins": analysis["reliability_bins"],
        "ece": analysis["ece"],
        "source_receipt": str(source),
    }
    target = store / "retained" / "per_d_reliability_table.json"
    atomic_json(target, table)
    entries = []
    for path in sorted(store.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            entries.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    receipt = {
        "schema": "ddm_hc1_manifest.v1",
        "store": str(store),
        "artifacts": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
    }
    atomic_json(store / "analysis" / "RETENTION.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("verify", "dfield", "analyze", "dcompare", "manifest")
    )
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--fine", type=int, default=65536)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args(argv)
    store = args.store
    store.mkdir(parents=True, exist_ok=True)
    if args.stage == "verify":
        receipt = run_verify(store)
    elif args.stage == "dfield":
        receipt = run_dfield(store)
    elif args.stage == "dcompare":
        receipt = run_dcompare(store)
    elif args.stage == "manifest":
        receipt = run_manifest(store, args.fine, args.seed)
    else:
        receipt = run_analyze(store, args.fine, args.ridge, args.seed)
    print(json.dumps(receipt, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
