"""ddm_hm1 phase B -- the measured d(token bytes)/d(counted model bytes) ladder.

The shipped hv1 archive already contains one rung of a model-capacity ladder: an RCF1
additive-logit correction table indexed by ``(boundary distance bucket, argmax class)``,
25 states x 5 classes at 6 bits.  It is counted archive payload.  So the exchange rate
"one more counted model byte buys how many token bytes?" can be measured on the real
object with **no training at all** -- fit richer correction tables against the retained
raw HPAC logits, quantize each exactly the way RCF1 does, and price the packed table.

Every rung here is REALIZED, not an oracle: the reported token bytes are the true
cross-entropy of the receiver's own quantized-softmax pipeline under a table that is
itself 6-bit quantized and byte-priced.  There is no free-table optimism.

The estimator is exact per cell.  With a fixed base logit vector L_i, the family
``p_i = softmax(L_i + T[cell(i)])`` is a multinomial logistic model with per-cell
intercepts, separable across cells and convex in T, so damped Newton converges to the
global optimum.

Axis: ``[macOS-CPU advisory / scorer-free byte measurement]``.  ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RETAINED = Path("/Volumes/APDataStore/pact/ddm_hm1_20260816/retained")
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815"
    "/runs/base_optimized_n600_r3/output/.f26_decode_checkpoints"
    "/tokens_cpu_stage_complete.u8"
)

NUM_CLASSES = 5
LOGIT_PRECISION = 8
RCF1_TABLE_BITS = 6
RCF1_HEADER_BYTES = 6  # 4-byte magic + fp16 scale, exactly as residual_archive packs it
HEIGHT, WIDTH = 384, 512
PLANE = HEIGHT * WIDTH
SHIPPED_TOKEN_STREAM_BYTES = 112_110
SHIPPED_HPAC_MODEL_BYTES = 13_515
# ddm_dc1 retained/hpac_cross_entropy_n600.json -- the shipped rung's own cross-entropy.
SHIPPED_CROSS_ENTROPY_BYTES = 112_109.57757858819
MAX_NEWTON_STEP = 8.0
MAX_OFFSET_LOGITS = 16.0


class LadderError(RuntimeError):
    """Raised when a ladder input or invariant cannot be verified."""


@dataclass(frozen=True)
class Rung:
    name: str
    description: str
    cells: np.ndarray
    cell_count: int
    realizable: bool = True


def packed_table_bytes(cell_count: int, bits: int = RCF1_TABLE_BITS) -> int:
    """Exact RCF1 packing cost for a ``cell_count x 5`` table at ``bits`` per value."""
    values = cell_count * NUM_CLASSES
    return RCF1_HEADER_BYTES + -(-values * bits // 8)


def boundary_buckets(previous: np.ndarray, max_distance: int) -> np.ndarray:
    """``residual_archive._boundary_buckets`` generalized to any max distance."""
    if previous.ndim != 2:
        raise LadderError("boundary source must be one token frame")
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


def quantize_table(
    table: np.ndarray,
    weights: np.ndarray | None = None,
    bits: int = RCF1_TABLE_BITS,
) -> tuple[np.ndarray, float]:
    """Quantize to signed ``bits`` with one shared fp16 scale, exactly like RCF1.

    ``_decode_fixed_table`` reads a signed ``bits``-wide code per value and one fp16
    scale, then reconstructs ``codes * scale``.  Mirror that so the returned values are
    literally decodable by the shipped receiver.

    One shared scale means the peak entry sets the resolution for every other entry, so
    ``max|T| / limit`` is the wrong choice as soon as one cell is extreme.  Search the
    scale instead, weighting each cell by how many symbols actually land in it: a coarse
    step on a cell holding ten symbols is free, the same step on a dense cell is not.
    """
    if table.size == 0:
        return table.astype(np.float32), 1.0
    limit = (1 << (bits - 1)) - 1
    peak = float(np.abs(table).max())
    if peak == 0.0 or not np.isfinite(peak):
        return np.zeros_like(table, dtype=np.float32), 1.0
    if weights is None:
        weights = np.ones(table.shape[0], dtype=np.float64)
    column = np.asarray(weights, dtype=np.float64)[:, None]

    best_error = np.inf
    best: tuple[np.ndarray, float] | None = None
    for factor in np.geomspace(1.0, 64.0, 25):
        candidate = np.float16(peak / (limit * factor))
        value = float(candidate)
        if value <= 0.0 or not np.isfinite(value):
            continue
        codes = np.clip(np.rint(table / np.float32(value)), -limit - 1, limit)
        restored = (codes.astype(np.float32) * np.float32(value)).astype(np.float32)
        error = float((column * np.square(restored - table, dtype=np.float64)).sum())
        if error < best_error:
            best_error = error
            best = (restored, value)
    if best is None:  # pragma: no cover - fp16 always yields one usable candidate
        return np.zeros_like(table, dtype=np.float32), 1.0
    return best


def _chunked(symbols: int, chunk: int):
    start = 0
    while start < symbols:
        stop = min(start + chunk, symbols)
        yield start, stop
        start = stop


def evaluate_cost_bytes(
    logits: np.memmap,
    truth: np.ndarray,
    cells: np.ndarray,
    table: np.ndarray,
    chunk: int,
) -> tuple[float, float]:
    """Receiver-faithful cross-entropy in bytes under an additive correction table.

    Reproduces ``residual_archive._probability_table``: the corrected logits are
    quantized to int16 at precision 8 before the softmax, so the reported bytes are what
    the real decoder's coder would consume, not an idealized float cost.

    Returns ``(bytes, smallest probability assigned to an actual symbol)``.  The floor
    is not decoration: rc64 codes against a 2**31 frequency total, so a probability
    below 2**-31 is not representable and that rung is not realizable at all.
    """
    total_bits = 0.0
    floor = 1.0
    for start, stop in _chunked(truth.size, chunk):
        base = logits[start:stop].astype(np.float32) / LOGIT_PRECISION
        corrected = base + table[cells[start:stop]]
        quantized = np.clip(
            np.rint(corrected * LOGIT_PRECISION), -32768, 32767
        ).astype(np.int16)
        values = (quantized.astype(np.float32) / LOGIT_PRECISION).astype(np.float64)
        values -= values.max(axis=1, keepdims=True)
        probabilities = np.exp(values)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities.astype(np.float32)
        chosen = probabilities[
            np.arange(stop - start), truth[start:stop]
        ].astype(np.float64)
        floor = min(floor, float(chosen.min()))
        total_bits -= float(np.log2(np.maximum(chosen, np.float64(1e-300))).sum())
    return total_bits / 8.0, floor


def fit_table(
    logits: np.memmap,
    truth: np.ndarray,
    cells: np.ndarray,
    cell_count: int,
    iterations: int,
    chunk: int,
    ridge: float,
) -> np.ndarray:
    """Damped Newton on the per-cell intercepts.  Convex, so this is the global optimum."""
    table = np.zeros((cell_count, NUM_CLASSES), dtype=np.float32)
    # bincount is a tight C reduction; np.add.at is an unbuffered ufunc and is orders of
    # magnitude slower at 10^8 symbols.  Same arithmetic, same result.
    one_hot_counts = np.bincount(
        cells * NUM_CLASSES + truth, minlength=cell_count * NUM_CLASSES
    ).astype(np.float64).reshape(cell_count, NUM_CLASSES)
    cell_totals = one_hot_counts.sum(axis=1)

    for _ in range(iterations):
        expected = np.zeros((cell_count, NUM_CLASSES), dtype=np.float64)
        hessian = np.zeros((cell_count, NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        for start, stop in _chunked(truth.size, chunk):
            block_cells = cells[start:stop]
            base = logits[start:stop].astype(np.float32) / LOGIT_PRECISION
            values = (base + table[block_cells]).astype(np.float64)
            values -= values.max(axis=1, keepdims=True)
            probabilities = np.exp(values)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            for row in range(NUM_CLASSES):
                expected[:, row] += np.bincount(
                    block_cells, weights=probabilities[:, row], minlength=cell_count
                )
            for row in range(NUM_CLASSES):
                for column in range(row, NUM_CLASSES):
                    weights = -probabilities[:, row] * probabilities[:, column]
                    if row == column:
                        weights = weights + probabilities[:, row]
                    contribution = np.bincount(
                        block_cells, weights=weights, minlength=cell_count
                    )
                    hessian[:, row, column] += contribution
                    if row != column:
                        hessian[:, column, row] += contribution

        # L2 prior on the offsets themselves, NOT a relative step damper.  Without it
        # the optimum sends the offset of any class never observed inside a cell to
        # -infinity: the softmax underflows, the code length is infinite, and the fp16
        # scale of the shared RCF1 quantizer is destroyed by that one outlier.  An
        # absolute penalty shrinks sparse cells toward zero and leaves dense cells free,
        # which is the correct prior for a table that must be transmitted.
        gradient = expected - one_hot_counts + 2.0 * ridge * table
        hessian[:, np.arange(NUM_CLASSES), np.arange(NUM_CLASSES)] += 2.0 * ridge
        try:
            step = np.linalg.solve(hessian, gradient[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError as error:  # pragma: no cover - guarded by ridge
            raise LadderError(f"Newton step is singular: {error}") from error
        # The family is invariant to a per-cell constant; centre it so the quantizer
        # spends its dynamic range on real class contrast, not on a free offset.
        step -= step.mean(axis=1, keepdims=True)
        # An undamped Newton step diverges on a near-deterministic cell: the optimum
        # sits far out, the first step overshoots by orders of magnitude, the softmax
        # saturates, the Hessian collapses to the ridge alone, and the iterate
        # oscillates until centring leaves nothing but float32 noise.  Bounding both
        # the step and the iterate keeps the descent monotone.  The +/-16 logit bound
        # is a likelihood ratio of about 9e6, far inside what the coder can represent.
        np.clip(step, -MAX_NEWTON_STEP, MAX_NEWTON_STEP, out=step)
        table = np.clip(table - step, -MAX_OFFSET_LOGITS, MAX_OFFSET_LOGITS).astype(
            np.float32
        )
        table[cell_totals == 0] = 0.0
    return table


def oracle_bytes(cells: np.ndarray, truth: np.ndarray, cell_count: int) -> float:
    """Empirical conditional entropy in bytes: a FREE-TABLE lower bound on cost.

    This is the cost of a coder that already knows the exact within-cell class
    distribution and pays nothing to transmit it, so it lower-bounds any predictor whose
    information is a function of the cell index ALONE.

    READ THE SCOPE CAREFULLY -- ddm_hm1's own n600 run refuted the stronger reading these
    rows were built for.  A cell index derived from the model's output is a REPLACEMENT
    for that output, not a refinement of it: the oracle predicts from the cell and
    discards the base logits.  So these rows do NOT bound an additive correction table,
    which keeps the full logit vector and adds a per-cell offset.  Measured n600: even
    the finest cell index here costs +2,097 B MORE than the shipped model, which only
    shows the cell index is a lossy summary.  The realizable ``r*`` rungs are the valid
    instrument for the correction family; these rows measure how much of the model's
    output survives a hand-designed summary, and the answer is "not enough".
    """
    joint = np.bincount(
        cells * NUM_CLASSES + truth, minlength=cell_count * NUM_CLASSES
    ).astype(np.float64).reshape(cell_count, NUM_CLASSES)
    totals = joint.sum(axis=1)
    active = totals > 0
    if not np.any(active):
        return 0.0
    probabilities = joint[active] / totals[active][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, probabilities * np.log2(probabilities), 0.0)
    return float(-(terms.sum(axis=1) * totals[active]).sum()) / 8.0


def build_rungs(
    tokens: np.ndarray,
    logits: np.memmap,
    frames: int,
    chunk: int,
) -> list[Rung]:
    """Assemble the context ladder.  Every feature is causally available at decode time."""
    symbols = frames * PLANE

    print("  deriving context features ...", flush=True)
    argmax = np.empty(symbols, dtype=np.uint8)
    runner_up = np.empty(symbols, dtype=np.uint8)
    margin = np.empty(symbols, dtype=np.float32)
    for start, stop in _chunked(symbols, chunk):
        base = logits[start:stop].astype(np.float32)
        # MUST be argmax, not argsort.  residual_archive.decode_production_tokens keys
        # the shipped table on ``base_logits.argmax(axis=1)``, which resolves a tie to
        # the FIRST maximum; np.argsort is not stable and resolves it to the last.  On a
        # 5-class integer-logit field ties are common, and using the wrong one silently
        # de-aligns rung r1 from the table actually in the archive.
        top_index = base.argmax(axis=1)
        rows = np.arange(stop - start)
        argmax[start:stop] = top_index.astype(np.uint8)
        top = base[rows, top_index]
        masked = base.copy()
        masked[rows, top_index] = -np.inf
        second_index = masked.argmax(axis=1)
        runner_up[start:stop] = second_index.astype(np.uint8)
        margin[start:stop] = (top - masked[rows, second_index]) / LOGIT_PRECISION

    bucket4 = np.empty((frames, PLANE), dtype=np.uint8)
    bucket8 = np.empty((frames, PLANE), dtype=np.uint8)
    previous_class = np.empty((frames, PLANE), dtype=np.uint8)
    bucket4[0] = 4
    bucket8[0] = 8
    previous_class[0] = 0
    for frame in range(1, frames):
        previous = tokens[frame - 1]
        bucket4[frame] = boundary_buckets(previous, 4).reshape(-1)
        bucket8[frame] = boundary_buckets(previous, 8).reshape(-1)
        previous_class[frame] = previous.reshape(-1)
    bucket4 = bucket4.reshape(-1)
    bucket8 = bucket8.reshape(-1)
    previous_class = previous_class.reshape(-1)

    def margin_bins(count: int) -> np.ndarray:
        # Equal-mass bins over the model's own confidence: a pure recalibration axis
        # that costs the receiver only a fixed set of thresholds.
        sample = margin[:: max(1, symbols // 4_000_000)]
        edges = np.quantile(sample, np.linspace(0.0, 1.0, count + 1)[1:-1])
        edges = np.unique(edges.astype(np.float32))
        return np.searchsorted(edges, margin).astype(np.uint16)

    margin4 = margin_bins(4)
    margin8 = margin_bins(8)
    margin16 = margin_bins(16)
    margin32 = margin_bins(32)
    m4 = int(margin4.max()) + 1
    m8 = int(margin8.max()) + 1
    m16 = int(margin16.max()) + 1
    m32 = int(margin32.max()) + 1

    rungs: list[Rung] = []

    def add(
        name: str,
        description: str,
        cells: np.ndarray,
        count: int,
        realizable: bool = True,
    ) -> None:
        rungs.append(
            Rung(
                name=name,
                description=description,
                cells=cells.astype(np.int64, copy=False),
                cell_count=count,
                realizable=realizable,
            )
        )

    add("r0_no_table", "no correction table at all", np.zeros(symbols, np.int64), 1)
    add(
        "r1_shipped_context",
        "SHIPPED context: prev-frame boundary bucket (5) x argmax (5)",
        bucket4.astype(np.int64) * NUM_CLASSES + argmax,
        25,
    )
    add(
        "r2_margin4",
        "shipped context x model confidence margin (4 equal-mass bins)",
        (bucket4.astype(np.int64) * NUM_CLASSES + argmax) * m4 + margin4,
        25 * m4,
    )
    add(
        "r3_margin16",
        "shipped context x margin (16 bins)",
        (bucket4.astype(np.int64) * NUM_CLASSES + argmax) * m16 + margin16,
        25 * m16,
    )
    add(
        "r4_bucket8_margin16",
        "finer boundary distance (9) x argmax (5) x margin (16)",
        (bucket8.astype(np.int64) * NUM_CLASSES + argmax) * m16 + margin16,
        9 * NUM_CLASSES * m16,
    )
    add(
        "r5_bucket8_margin32",
        "finer boundary distance (9) x argmax (5) x margin (32)",
        (bucket8.astype(np.int64) * NUM_CLASSES + argmax) * m32 + margin32,
        9 * NUM_CLASSES * m32,
    )
    add(
        "r6_prevclass_margin16",
        "boundary (9) x argmax (5) x margin (16) x co-located previous class (5)",
        ((bucket8.astype(np.int64) * NUM_CLASSES + argmax) * m16 + margin16) * NUM_CLASSES
        + previous_class,
        9 * NUM_CLASSES * m16 * NUM_CLASSES,
    )
    add(
        "r7_prevclass_margin32",
        "boundary (9) x argmax (5) x margin (32) x previous class (5)",
        ((bucket8.astype(np.int64) * NUM_CLASSES + argmax) * m32 + margin32) * NUM_CLASSES
        + previous_class,
        9 * NUM_CLASSES * m32 * NUM_CLASSES,
    )
    add(
        "r8_second_choice",
        "boundary (9) x argmax (5) x margin (8) x prev class (5) x runner-up class (5)",
        (
            (
                ((bucket8.astype(np.int64) * NUM_CLASSES + argmax) * m8 + margin8)
                * NUM_CLASSES
                + previous_class
            )
            * NUM_CLASSES
        )
        + runner_up.astype(np.int64),
        9 * NUM_CLASSES * m8 * NUM_CLASSES * NUM_CLASSES,
    )

    # FREE-TABLE ORACLE ROWS -- not realizable, not rungs, and NOT a bound on the
    # correction family (see oracle_bytes).  They predict from the cell index alone, so
    # they REPLACE the model's output rather than refine it.  What they measure is how
    # much of that output survives a hand-designed summary; the n600 answer is that even
    # the finest summary here loses to the model itself, which is why keying a richer
    # correction table on more hand-designed features cannot be the route.
    margin64 = margin_bins(64)
    margin256 = margin_bins(256)
    m64 = int(margin64.max()) + 1
    m256 = int(margin256.max()) + 1
    runner = runner_up.astype(np.int64)
    add(
        "o1_oracle_model_output_coarse",
        "FREE-TABLE ORACLE: argmax (5) x margin (16)",
        argmax.astype(np.int64) * m16 + margin16,
        NUM_CLASSES * m16,
        realizable=False,
    )
    add(
        "o2_oracle_model_output_fine",
        "FREE-TABLE ORACLE: argmax (5) x margin (64) x runner-up (5)",
        (argmax.astype(np.int64) * m64 + margin64) * NUM_CLASSES + runner,
        NUM_CLASSES * m64 * NUM_CLASSES,
        realizable=False,
    )
    add(
        "o3_oracle_model_output_finest",
        "FREE-TABLE ORACLE: argmax (5) x margin (256) x runner-up (5) -- the finest "
        "hand-designed summary of the model output tested; REPLACES it, does not refine it",
        (argmax.astype(np.int64) * m256 + margin256) * NUM_CLASSES + runner,
        NUM_CLASSES * m256 * NUM_CLASSES,
        realizable=False,
    )
    add(
        "o4_oracle_output_plus_context",
        "FREE-TABLE ORACLE: o3 x boundary distance (9) x previous class (5) -- the same "
        "summary plus the two context features the shipped correction table can see",
        (
            ((argmax.astype(np.int64) * m256 + margin256) * NUM_CLASSES + runner) * 9
            + bucket8
        )
        * NUM_CLASSES
        + previous_class,
        NUM_CLASSES * m256 * NUM_CLASSES * 9 * NUM_CLASSES,
        realizable=False,
    )
    return rungs


def run(
    retained: Path,
    tokens_path: Path,
    frames: int,
    iterations: int,
    chunk: int,
    ridge: float,
    outdir: Path,
    logits_path: Path | None = None,
) -> dict[str, Any]:
    symbols = frames * PLANE
    if logits_path is None:
        logits_path = retained / f"base_logits_int16_n{frames}.i16"
    if not logits_path.is_file():
        raise LadderError(f"missing retained logits: {logits_path}")
    available = logits_path.stat().st_size // (2 * NUM_CLASSES)
    if available < symbols:
        raise LadderError(
            f"retained logits hold {available} symbols, {symbols} requested"
        )
    logits = np.memmap(logits_path, dtype=np.int16, mode="r").reshape(
        available, NUM_CLASSES
    )[:symbols]
    tokens = np.memmap(tokens_path, dtype=np.uint8, mode="r").reshape(-1, HEIGHT, WIDTH)[
        :frames
    ]
    truth = np.ascontiguousarray(tokens.reshape(-1)).astype(np.int64)

    started = time.perf_counter()
    rungs = build_rungs(tokens, logits, frames, chunk)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    baseline_cost: float | None = None
    for rung in rungs:
        rung_started = time.perf_counter()
        occupancy = np.bincount(rung.cells, minlength=rung.cell_count)
        used = int(np.count_nonzero(occupancy))
        if not rung.realizable:
            floor_bytes = oracle_bytes(rung.cells, truth, rung.cell_count)
            rows.append(
                {
                    "rung": rung.name,
                    "description": rung.description,
                    "free_table_oracle": True,
                    "cells_declared": rung.cell_count,
                    "cells_used": used,
                    "samples_per_used_cell": symbols / used,
                    "packed_model_bytes": None,
                    "oracle_token_bytes": floor_bytes,
                    "oracle_saving_vs_shipped_bytes": SHIPPED_CROSS_ENTROPY_BYTES
                    - floor_bytes
                    if frames == 600
                    else None,
                    "seconds": time.perf_counter() - rung_started,
                }
            )
            print(
                f"  {rung.name:32s} cells={used:>8,} samp/cell={symbols / used:>11,.0f} "
                f"ORACLE tokens={floor_bytes:>12,.1f}B",
                flush=True,
            )
            continue
        if rung.name == "r0_no_table":
            table = np.zeros((rung.cell_count, NUM_CLASSES), dtype=np.float32)
            quantized = table
            scale = 0.0
            model_bytes = 0
        else:
            table = fit_table(
                logits, truth, rung.cells, rung.cell_count, iterations, chunk, ridge
            )
            quantized, scale = quantize_table(table, occupancy)
            model_bytes = packed_table_bytes(rung.cell_count)
        cost, floor = evaluate_cost_bytes(logits, truth, rung.cells, quantized, chunk)
        if rung.name == "r0_no_table":
            continuous_cost, continuous_floor = cost, floor
        else:
            continuous_cost, continuous_floor = evaluate_cost_bytes(
                logits, truth, rung.cells, table, chunk
            )
        if baseline_cost is None:
            baseline_cost = cost
        payload = quantized.astype("<f4").tobytes()
        payload_path = outdir / f"table_{rung.name}.f32"
        payload_path.write_bytes(payload)
        row = {
            "rung": rung.name,
            "description": rung.description,
            "cells_declared": rung.cell_count,
            "cells_used": used,
            "samples_per_used_cell": symbols / used,
            "packed_model_bytes": model_bytes,
            "realized_token_bytes": cost,
            "continuous_optimum_token_bytes": continuous_cost,
            "quantization_loss_bytes": cost - continuous_cost,
            "joint_bytes": cost + model_bytes,
            "delta_tokens_vs_no_table": cost - baseline_cost,
            "table_scale": scale,
            "minimum_probability_realized": floor,
            "minimum_probability_continuous": continuous_floor,
            "coder_representable": bool(floor > 2.0**-31),
            "table_payload": {
                "path": str(payload_path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "seconds": time.perf_counter() - rung_started,
        }
        rows.append(row)
        print(
            f"  {rung.name:24s} cells={used:>7,} samp/cell={symbols / used:>12,.0f} "
            f"model={model_bytes:>7,}B tokens={cost:>12,.1f}B joint={cost + model_bytes:>12,.1f}B",
            flush=True,
        )

    shipped = _shipped_row(logits, truth, rungs, chunk, frames)
    rows.append(shipped)

    ordered = [
        row
        for row in rows
        if row["rung"] != "shipped_actual_table" and not row.get("free_table_oracle")
    ]
    for index in range(1, len(ordered)):
        previous_row, row = ordered[index - 1], ordered[index]
        model_delta = row["packed_model_bytes"] - previous_row["packed_model_bytes"]
        token_delta = row["realized_token_bytes"] - previous_row["realized_token_bytes"]
        row["adjacent_model_byte_delta"] = model_delta
        row["adjacent_token_byte_delta"] = token_delta
        row["adjacent_slope_tokens_per_model_byte"] = (
            token_delta / model_delta if model_delta else None
        )
        row["adjacent_pays"] = bool(model_delta and token_delta / model_delta < -1.0)

    best = min(ordered, key=lambda row: row["joint_bytes"])
    report = {
        "schema": "ddm_hm1_correction_capacity_ladder.v1",
        "axis": "[macOS-CPU advisory / scorer-free byte measurement]",
        "score_claim": False,
        "promotable": False,
        "frames": frames,
        "symbols": symbols,
        "is_full_field": frames == 600,
        "newton_iterations": iterations,
        "shipped_token_stream_bytes": SHIPPED_TOKEN_STREAM_BYTES,
        "shipped_hpac_model_bytes": SHIPPED_HPAC_MODEL_BYTES,
        "shipped_cross_entropy_bytes": SHIPPED_CROSS_ENTROPY_BYTES,
        "rows": rows,
        "best_joint": {
            "rung": best["rung"],
            "joint_bytes": best["joint_bytes"],
            "packed_model_bytes": best["packed_model_bytes"],
            "realized_token_bytes": best["realized_token_bytes"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def _shipped_row(
    logits: np.memmap,
    truth: np.ndarray,
    rungs: list[Rung],
    chunk: int,
    frames: int,
) -> dict[str, Any]:
    """Re-price the ACTUAL shipped table as a ladder row, straight from the archive."""
    from ddm_hm1_hpac_logit_replay import DEFAULT_PREPARED, _import_runtime

    residual_archive, _renderer, _code_dir = _import_runtime(DEFAULT_PREPARED)
    parts = residual_archive.read_residual_archive(DEFAULT_PREPARED / "archive.zip")
    shipped_table = np.asarray(parts.table.values, dtype=np.float32)
    context = next(rung for rung in rungs if rung.name == "r1_shipped_context")
    cost, floor = evaluate_cost_bytes(
        logits, truth, context.cells, shipped_table, chunk
    )
    return {
        "rung": "shipped_actual_table",
        "description": "the RCF1 table actually inside the hv1 archive",
        "cells_declared": 25,
        "cells_used": 25,
        "samples_per_used_cell": frames * PLANE / 25,
        "packed_model_bytes": packed_table_bytes(25),
        "realized_token_bytes": cost,
        "continuous_optimum_token_bytes": None,
        "quantization_loss_bytes": None,
        "joint_bytes": cost + packed_table_bytes(25),
        "table_scale": float(parts.table.scale),
        "minimum_probability_realized": floor,
        "coder_representable": bool(floor > 2.0**-31),
        "reproduces_dc1_cross_entropy": abs(cost - SHIPPED_CROSS_ENTROPY_BYTES) < 1e-3
        if frames == 600
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", type=Path, default=DEFAULT_RETAINED)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=8 * PLANE)
    parser.add_argument(
        "--ridge",
        type=float,
        default=1.0,
        help="absolute L2 prior on the offsets; shrinks sparse cells, frees dense ones",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--logits-path",
        type=Path,
        default=None,
        help="override the retained logit field; a prefix read is a SMOKE, never a verdict",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    outdir = args.outdir or (args.retained / f"ladder_n{args.frames}")
    report = run(
        args.retained,
        args.tokens,
        args.frames,
        args.iterations,
        args.chunk,
        args.ridge,
        outdir,
        args.logits_path,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    destination = args.report or (outdir / "ladder.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
