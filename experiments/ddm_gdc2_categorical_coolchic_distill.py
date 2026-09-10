#!/usr/bin/env python3
"""GDC2: distil the retained K=8 scanline teacher into a counted categorical decoder.

GDC1 measured the ordered scanline program FORMULATION-NO-GO and retained one
full-n600 K=8 teacher render of the move-43 token field.  This driver implements
GDC1's governed GDC2 launch spec verbatim: a Cool-Chic-form coordinate decoder
(two int8 latent grids, trilinear resampling, five width-24 depthwise-separable
blocks, a five-logit integer head) is trained against that teacher, and every
verdict is taken from the NumPy integer receiver in ``tac.gdc2_categorical_coolchic``
on the physically coded packet.

MLX training is research signal.  The receiver is the authority.  There is no
scorer, no Modal, no candidate archive and no score claim anywhere in this file.

The gate the run answers is GDC1's success inequality
``packet_bytes + real_exact_residual_bytes <= 94,010``,
where the residual carries the rendered field back to the move-43 field exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import mlx.core as mx
import mlx.optimizers as optim

from tac import gdc2_categorical_coolchic as rx

FIELD_PATH: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8"
)
FIELD_SHA256: Final = "78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6"
TEACHER_PATH: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/k08/receiver_render.u8"
)
TEACHER_SHA256: Final = "abd130921beec23255112e8b56148d55c49cf4fede03e99e0e99b55378b16ea2"
TEACHER_PACKET_BYTES: Final = 306_042
TEACHER_MISMATCHES: Final = 88_304
TEACHER_REAL_RESIDUAL_BYTES: Final = 60_520
REPLACEMENT_INTEGER_CAP: Final = 94_010
POINTER_ARCHIVE_BYTES: Final = 180_466
POINTER_S: Final = 0.1372848557085275

BLOCKS_PER_STEP: Final = 16
PAIRS_PER_BLOCK: Final = 8
TILE: Final = 64
HALO: Final = rx.BLOCKS * (rx.KERNEL // 2)
WINDOW: Final = TILE + 2 * HALO
CHECKPOINT_EVERY: Final = 250
STAGE_A_STEPS: Final = 10_000
STAGE_B_STEPS: Final = 8_000
STAGE_C_STEPS: Final = 4_000
LAMBDAS: Final = (1e-4, 3e-4, 1e-3)
EMA_DECAY: Final = 0.999
WEIGHT_DECAY: Final = 1e-4
# AdamW takes a normalised step of order the learning rate, so the trainable
# chart must be unit scale.  The counted parameters are integers on a +/-127
# lattice, so every trainable tensor is carried unit scale and multiplied by its
# fixed quantiser scale before rounding.  Without this the spec's 3e-3 learning
# rate moves an integer weight by 3e-3 per step and the run learns nothing
# (MEASURED: cross entropy 1.70 -> 1.26 over 200 steps in the integer chart).
LATENT_SCALE: Final = 127.0
WEIGHT_SCALE: Final = 127.0
BIAS_SCALE: Final = 1024.0
PARAM_SCALES: Final = {
    "z0": LATENT_SCALE, "z1": LATENT_SCALE,
    "stem_w": WEIGHT_SCALE, "stem_b": BIAS_SCALE,
    "dw_w": WEIGHT_SCALE, "dw_b": BIAS_SCALE,
    "pw_w": WEIGHT_SCALE, "pw_b": BIAS_SCALE,
    "head_w": WEIGHT_SCALE, "head_b": BIAS_SCALE,
}
INIT_LATENT_STD: Final = 8.0
# Per-layer initialisation scales derived from the fixed requant shifts so the
# activation scale is preserved through all five blocks: for a layer with n
# input taps and post-ReLU input scale s, std(acc) = sigma * sqrt(n / 2) * s,
# and unit gain needs sigma = 2**shift / sqrt(n / 2).  A single global scale
# decays the signal to zero by block five and kills the gradient (MEASURED).
INIT_STD_STEM: Final = 12.0
INIT_STD_DW: Final = 30.0
INIT_STD_PW: Final = 37.0
INIT_STD_HEAD: Final = 37.0
# The Stage B rate proxy is the measured zeroth-order symbol rate of the counted
# latent stream expressed in the gate's own unit: estimated kilobytes.  The
# spec fixes lambda in {1e-4, 3e-4, 1e-3} but not the rate unit, and the unit
# decides whether the sweep is informative.  In kilobytes the three lambdas put
# the rate term at roughly 0.02 / 0.06 / 0.20 against a cross entropy of order
# 0.05-0.30 nats, which spans weak / moderate / strong rate pressure.  In
# bits-per-symbol the same three lambdas would all be inert and the sweep would
# measure nothing (DERIVED from the measured Stage-A cross entropy and entropy).
LATENT_SYMBOLS: Final = int(np.prod(rx.Z0_SHAPE) + np.prod(rx.Z1_SHAPE))
RATE_KILOBYTE_FACTOR: Final = LATENT_SYMBOLS / 8000.0

RESIDUAL_ORDERS: Final = (
    "frame_raster", "class_frame_raster", "tile8_time", "tile16_time",
    "tile32_time", "tile64_time", "class_tile16_time", "pair_tile16",
)


class GDC2RunError(RuntimeError):
    """A GDC2 run, custody, or arithmetic invariant failed."""


# --------------------------------------------------------------------------
# custody helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return file_fact(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def peak_rss_gib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024**3)


def load_verified(path: Path, expected_sha: str) -> np.ndarray:
    if not path.is_file():
        raise GDC2RunError(f"missing input: {path}")
    if sha256_file(path) != expected_sha:
        raise GDC2RunError(f"sha mismatch for {path}")
    return np.memmap(path, dtype=np.uint8, mode="r").reshape(rx.N_PAIRS, rx.HEIGHT, rx.WIDTH)


# --------------------------------------------------------------------------
# MLX forward mirroring the integer receiver exactly
# --------------------------------------------------------------------------


def straight_through(soft: mx.array, hard: mx.array) -> mx.array:
    return soft + mx.stop_gradient(hard - soft)


def quantize(value: mx.array, low: float, high: float) -> mx.array:
    hard = mx.clip(mx.floor(value + 0.5), low, high)
    return straight_through(value, hard)


def requant(accumulated: mx.array, shift: int, relu: bool) -> mx.array:
    soft = (accumulated + float(1 << (shift - 1))) * (2.0 ** -shift)
    value = straight_through(soft, mx.floor(soft))
    return mx.clip(value, 0.0, float(rx.ACT_MAX)) if relu else value


def mlx_lattice(lattice: rx.Lattice) -> dict[str, mx.array]:
    return {
        "t0": mx.array(lattice.t0), "t1": mx.array(lattice.t1), "tw": mx.array(lattice.tw.astype(np.float32)),
        "y0": mx.array(lattice.y0), "y1": mx.array(lattice.y1), "yw": mx.array(lattice.yw.astype(np.float32)),
        "x0": mx.array(lattice.x0), "x1": mx.array(lattice.x1), "xw": mx.array(lattice.xw.astype(np.float32)),
    }


MLX_L0: Final = mlx_lattice(rx.LATTICE_Z0)
MLX_L1: Final = mlx_lattice(rx.LATTICE_Z1)


def resample_mlx(latent: mx.array, table: dict[str, mx.array], lattice: rx.Lattice, gain: int,
                 pairs: mx.array, rows: mx.array, columns: mx.array) -> mx.array:
    """Exact integer trilinear resample onto (len(pairs), len(rows), len(columns), C)."""
    weight = mx.take(table["tw"], pairs).reshape(-1, 1, 1, 1)
    first = mx.take(latent, mx.take(table["t0"], pairs), axis=0)
    second = mx.take(latent, mx.take(table["t1"], pairs), axis=0)
    stage = first * (float(lattice.td) - weight) + second * weight
    weight = mx.take(table["yw"], rows).reshape(1, -1, 1, 1)
    first = mx.take(stage, mx.take(table["y0"], rows), axis=1)
    second = mx.take(stage, mx.take(table["y1"], rows), axis=1)
    stage = first * (float(lattice.yd) - weight) + second * weight
    weight = mx.take(table["xw"], columns).reshape(1, 1, -1, 1)
    first = mx.take(stage, mx.take(table["x0"], columns), axis=2)
    second = mx.take(stage, mx.take(table["x1"], columns), axis=2)
    stage = first * (float(lattice.xd) - weight) + second * weight
    return stage * float(gain)


def feature_mlx(z0: mx.array, z1: mx.array, pairs: mx.array, rows: mx.array,
                columns: mx.array) -> mx.array:
    part0 = resample_mlx(z0, MLX_L0, rx.LATTICE_Z0, rx.Z0_GAIN, pairs, rows, columns)
    part1 = resample_mlx(z1, MLX_L1, rx.LATTICE_Z1, rx.Z1_GAIN, pairs, rows, columns)
    stacked = mx.concatenate([part0, part1], axis=3)
    soft = (stacked + float(1 << (rx.INPUT_SHIFT - 1))) * (2.0 ** -rx.INPUT_SHIFT)
    return straight_through(soft, mx.floor(soft))


def depthwise_valid(activation: mx.array, weight: mx.array) -> mx.array:
    """Valid 3x3 depthwise convolution as nine exact shifted multiply-adds."""
    height = activation.shape[1] - 2
    width = activation.shape[2] - 2
    total = None
    for row in range(rx.KERNEL):
        for column in range(rx.KERNEL):
            piece = activation[:, row:row + height, column:column + width, :]
            tap = weight[:, row, column].reshape(1, 1, 1, -1)
            total = piece * tap if total is None else total + piece * tap
    return total


def counted(params: dict[str, mx.array], name: str) -> mx.array:
    """The trainable unit-scale tensor lifted onto its counted integer lattice."""
    limit = rx.LATENT_MAX if name in ("z0", "z1") else (
        rx.WEIGHT_MAX if name.endswith("_w") else 2**30)
    return quantize(params[name] * PARAM_SCALES[name], -limit, limit)


def forward_window(params: dict[str, mx.array], pairs: mx.array, rows: mx.array,
                   columns: mx.array, mask: mx.array) -> mx.array:
    """Logits for a training window, masked exactly like the receiver's zero pad."""
    z0 = counted(params, "z0")
    z1 = counted(params, "z1")
    features = feature_mlx(z0, z1, pairs, rows, columns) * mask
    stem_w = counted(params, "stem_w")
    stem_b = counted(params, "stem_b")
    activation = requant(features @ stem_w.T + stem_b, rx.SHIFT_STEM, True) * mask
    dw_all = counted(params, "dw_w")
    dw_bias = counted(params, "dw_b")
    pw_all = counted(params, "pw_w")
    pw_bias = counted(params, "pw_b")
    for block in range(rx.BLOCKS):
        dw_w = dw_all[block]
        dw_b = dw_bias[block]
        pw_w = pw_all[block]
        pw_b = pw_bias[block]
        trimmed = mask[:, block + 1:mask.shape[1] - block - 1,
                       block + 1:mask.shape[2] - block - 1, :]
        activation = requant(depthwise_valid(activation, dw_w) + dw_b, rx.SHIFT_DW, True)
        activation = activation * trimmed
        activation = requant(activation @ pw_w.T + pw_b, rx.SHIFT_PW, True) * trimmed
    head_w = counted(params, "head_w")
    head_b = counted(params, "head_b")
    return activation @ head_w.T + head_b


def soft_symbol_entropy(params: dict[str, mx.array]) -> mx.array:
    """Measured zeroth-order symbol rate of the counted latent stream, in bits."""
    values = mx.concatenate([
        (params["z0"] * LATENT_SCALE).reshape(-1),
        (params["z1"] * LATENT_SCALE).reshape(-1),
    ])
    values = mx.clip(values, -float(rx.LATENT_MAX), float(rx.LATENT_MAX) - 1e-3)
    low = mx.stop_gradient(mx.floor(values))
    upper_weight = values - low
    index = mx.stop_gradient((low + float(rx.LATENT_MAX)).astype(mx.int32))
    bins = 2 * rx.LATENT_MAX + 2
    counts = mx.zeros((bins,))
    counts = counts.at[index].add(1.0 - upper_weight)
    counts = counts.at[index + 1].add(upper_weight)
    total = values.shape[0]
    probability = counts / float(total)
    safe = mx.maximum(probability, 1e-12)
    return -mx.sum(probability * mx.log(safe)) / math.log(2.0)


def initial_parameters(seed: int) -> dict[str, mx.array]:
    generator = np.random.default_rng(seed)

    def normal(shape: tuple[int, ...], scale: float, chart: float) -> mx.array:
        draw = generator.normal(0.0, scale / chart, size=shape)
        return mx.array(draw.astype(np.float32))

    return {
        "z0": normal(rx.Z0_SHAPE, INIT_LATENT_STD, LATENT_SCALE),
        "z1": normal(rx.Z1_SHAPE, INIT_LATENT_STD, LATENT_SCALE),
        "stem_w": normal((rx.WIDTH_CH, rx.FEATURES), INIT_STD_STEM, WEIGHT_SCALE),
        "stem_b": mx.zeros((rx.WIDTH_CH,)),
        "dw_w": normal((rx.BLOCKS, rx.WIDTH_CH, rx.KERNEL, rx.KERNEL), INIT_STD_DW, WEIGHT_SCALE),
        "dw_b": mx.zeros((rx.BLOCKS, rx.WIDTH_CH)),
        "pw_w": normal((rx.BLOCKS, rx.WIDTH_CH, rx.WIDTH_CH), INIT_STD_PW, WEIGHT_SCALE),
        "pw_b": mx.zeros((rx.BLOCKS, rx.WIDTH_CH)),
        "head_w": normal((rx.CLASSES, rx.WIDTH_CH), INIT_STD_HEAD, WEIGHT_SCALE),
        "head_b": mx.zeros((rx.CLASSES,)),
    }


def export_parameters(params: dict[str, mx.array]) -> tuple[np.ndarray, np.ndarray, rx.Parameters]:
    """Quantize the exported (EMA) parameters exactly as the trainer's fake-quant does."""

    def hard(name: str, low: float, high: float) -> np.ndarray:
        value = np.asarray(params[name], dtype=np.float32).astype(np.float64)
        value = value * PARAM_SCALES[name]
        return np.clip(np.floor(value + 0.5), low, high).astype(np.int64)

    z0 = hard("z0", -rx.LATENT_MAX, rx.LATENT_MAX)
    z1 = hard("z1", -rx.LATENT_MAX, rx.LATENT_MAX)
    parameters = rx.Parameters(
        stem_w=hard("stem_w", -rx.WEIGHT_MAX, rx.WEIGHT_MAX),
        stem_b=hard("stem_b", -(2**30), 2**30),
        dw_w=hard("dw_w", -rx.WEIGHT_MAX, rx.WEIGHT_MAX),
        dw_b=hard("dw_b", -(2**30), 2**30),
        pw_w=hard("pw_w", -rx.WEIGHT_MAX, rx.WEIGHT_MAX),
        pw_b=hard("pw_b", -(2**30), 2**30),
        head_w=hard("head_w", -rx.WEIGHT_MAX, rx.WEIGHT_MAX),
        head_b=hard("head_b", -(2**30), 2**30),
        shifts=rx.default_shifts(),
    )
    return z0, z1, parameters


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def window_indices(origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clipped gather indices plus the receiver-equivalent validity mask."""
    pair0, row0, column0 = (int(value) for value in origin)
    pairs = np.arange(pair0, pair0 + PAIRS_PER_BLOCK)
    rows = np.arange(row0 - HALO, row0 - HALO + WINDOW)
    columns = np.arange(column0 - HALO, column0 - HALO + WINDOW)
    inside = ((rows >= 0) & (rows < rx.HEIGHT)).astype(np.float32).reshape(-1, 1) * (
        (columns >= 0) & (columns < rx.WIDTH)
    ).astype(np.float32).reshape(1, -1)
    mask = np.broadcast_to(inside.reshape(1, WINDOW, WINDOW, 1),
                           (PAIRS_PER_BLOCK, WINDOW, WINDOW, 1)).copy()
    return pairs, np.clip(rows, 0, rx.HEIGHT - 1), np.clip(columns, 0, rx.WIDTH - 1), mask


def step_batch(schedule_step: np.ndarray, teacher: np.ndarray) -> dict[str, Any]:
    blocks = []
    labels = []
    for origin in schedule_step:
        pairs, rows, columns, mask = window_indices(origin)
        blocks.append((mx.array(pairs), mx.array(rows), mx.array(columns), mx.array(mask)))
        pair0, row0, column0 = (int(value) for value in origin)
        labels.append(np.asarray(
            teacher[pair0:pair0 + PAIRS_PER_BLOCK, row0:row0 + TILE, column0:column0 + TILE]
        ))
    return {"blocks": blocks, "labels": mx.array(np.concatenate(labels, axis=0).astype(np.int32))}


def batch_logits(params: dict[str, mx.array], blocks: list[Any]) -> mx.array:
    pieces = [forward_window(params, pairs, rows, columns, mask)
              for pairs, rows, columns, mask in blocks]
    return mx.concatenate(pieces, axis=0)


def loss_terms(params: dict[str, mx.array], blocks: list[Any], labels: mx.array,
               lambda_rate: float) -> tuple[mx.array, mx.array, mx.array]:
    logits = batch_logits(params, blocks) * (1.0 / rx.HEAD_LOSS_DIVISOR)
    flat = logits.reshape(-1, rx.CLASSES)
    target = labels.reshape(-1)
    cross_entropy = mx.mean(
        mx.logsumexp(flat, axis=1) - mx.take_along_axis(flat, target.reshape(-1, 1), axis=1).reshape(-1)
    )
    entropy_bits = soft_symbol_entropy(params)
    total = cross_entropy + lambda_rate * (entropy_bits * RATE_KILOBYTE_FACTOR)
    return total, cross_entropy, entropy_bits


def cosine_lr(step: int, total: int, high: float, low: float) -> float:
    fraction = min(max(step / max(total - 1, 1), 0.0), 1.0)
    return low + 0.5 * (high - low) * (1.0 + math.cos(math.pi * fraction))


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------


def save_checkpoint(path: Path, params: dict[str, mx.array], ema: dict[str, mx.array],
                    optimizer: Any, step: int, config: dict[str, Any]) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    payload: dict[str, np.ndarray] = {}
    for name, value in params.items():
        payload[f"param::{name}"] = np.asarray(value, dtype=np.float32)
    for name, value in ema.items():
        payload[f"ema::{name}"] = np.asarray(value, dtype=np.float32)
    for name, value in tree_flatten(optimizer.state):
        array = np.asarray(value)
        if array.dtype == np.float32 or array.dtype == np.int32 or array.dtype == np.uint32:
            payload[f"opt::{name}"] = array
    payload["meta::step"] = np.array([step], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)
    atomic_json(path.with_suffix(".config.json"), config | {"step": step})
    return file_fact(path)


def load_checkpoint(path: Path, params: dict[str, mx.array], ema: dict[str, mx.array],
                    optimizer: Any) -> int:
    from mlx.utils import tree_flatten, tree_unflatten

    with np.load(path) as archive:
        for name in list(params):
            params[name] = mx.array(archive[f"param::{name}"])
            ema[name] = mx.array(archive[f"ema::{name}"])
        restored = []
        for name, value in tree_flatten(optimizer.state):
            key = f"opt::{name}"
            restored.append((name, mx.array(archive[key]) if key in archive else value))
        optimizer.state = tree_unflatten(restored)
        return int(archive["meta::step"][0])


def latest_checkpoint(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("step_*.npz"))
    return candidates[-1] if candidates else None


# --------------------------------------------------------------------------
# stage-end evaluation on the authority receiver
# --------------------------------------------------------------------------


def render_and_count(z0: np.ndarray, z1: np.ndarray, parameters: rx.Parameters,
                     render_path: Path, teacher: np.ndarray, field: np.ndarray) -> dict[str, Any]:
    render_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = render_path.with_suffix(".u8.tmp")
    started = time.monotonic()
    teacher_mismatch = 0
    field_mismatch = 0
    field_by_class = np.zeros(rx.CLASSES, dtype=np.int64)
    with temporary.open("wb") as handle:
        for pair in range(rx.N_PAIRS):
            labels = rx.receiver_frame(z0, z1, parameters, pair)
            handle.write(labels.tobytes())
            teacher_mismatch += int((labels != np.asarray(teacher[pair])).sum())
            differ = labels != np.asarray(field[pair])
            field_mismatch += int(differ.sum())
            if differ.any():
                field_by_class += np.bincount(
                    np.asarray(field[pair])[differ].astype(np.int64), minlength=rx.CLASSES
                )
    decode_seconds = time.monotonic() - started
    os.replace(temporary, render_path)
    return {
        "render": file_fact(render_path),
        "decode_seconds": decode_seconds,
        "mismatches_vs_teacher": teacher_mismatch,
        "mismatches_vs_field": field_mismatch,
        "field_mismatches_by_class_0_to_4": [int(value) for value in field_by_class],
        "lane_share_of_field_mismatches": (
            float(field_by_class[1]) / field_mismatch if field_mismatch else 0.0
        ),
    }


def parity_check(z0: np.ndarray, z1: np.ndarray, parameters: rx.Parameters,
                 render_path: Path, samples: int, seed: int) -> dict[str, Any]:
    """MLX float forward vs the NumPy integer receiver: exact argmax identity."""
    integers = {
        "z0": z0, "z1": z1, "stem_w": parameters.stem_w, "stem_b": parameters.stem_b,
        "dw_w": parameters.dw_w, "dw_b": parameters.dw_b,
        "pw_w": parameters.pw_w, "pw_b": parameters.pw_b,
        "head_w": parameters.head_w, "head_b": parameters.head_b,
    }
    exported = {
        name: mx.array((value / PARAM_SCALES[name]).astype(np.float32))
        for name, value in integers.items()
    }
    rendered = np.memmap(render_path, dtype=np.uint8, mode="r").reshape(
        rx.N_PAIRS, rx.HEIGHT, rx.WIDTH)
    generator = np.random.default_rng(seed)
    agree = 0
    total = 0
    for _ in range(samples):
        origin = np.array([
            generator.integers(0, rx.N_PAIRS - PAIRS_PER_BLOCK + 1),
            generator.integers(0, rx.HEIGHT - TILE + 1),
            generator.integers(0, rx.WIDTH - TILE + 1),
        ])
        pairs, rows, columns, mask = window_indices(origin)
        logits = forward_window(exported, mx.array(pairs), mx.array(rows),
                                mx.array(columns), mx.array(mask))
        predicted = np.asarray(mx.argmax(logits, axis=3)).astype(np.uint8)
        pair0, row0, column0 = (int(value) for value in origin)
        reference = np.asarray(
            rendered[pair0:pair0 + PAIRS_PER_BLOCK, row0:row0 + TILE, column0:column0 + TILE])
        agree += int((predicted == reference).sum())
        total += int(reference.size)
    return {"sampled_sites": total, "agreeing_sites": agree,
            "exact_argmax_identity": agree == total,
            "agreement_fraction": agree / total if total else 0.0}


def evaluate_export(tag: str, root: Path, ema: dict[str, mx.array], teacher: np.ndarray,
                    field: np.ndarray, seed: int, full_render: bool) -> dict[str, Any]:
    """Export the EMA authority, physically code it, and measure it on the receiver."""
    started = time.monotonic()
    directory = root / "exports" / tag
    directory.mkdir(parents=True, exist_ok=True)
    z0, z1, parameters = export_parameters(ema)
    atomic_write(directory / "z0.raw.i8", z0.astype(np.int8).tobytes())
    atomic_write(directory / "z1.raw.i8", z1.astype(np.int8).tobytes())
    atomic_write(directory / "params.raw", parameters.to_bytes())
    packet, facts = rx.build_packet(z0, z1, parameters)
    packet_fact = atomic_write(directory / "decoder.packet", packet)
    repeat_packet, _ = rx.build_packet(z0, z1, parameters)
    repeat_fact = atomic_write(directory / "decoder.repeat.packet", repeat_packet)
    if repeat_packet != packet:
        raise GDC2RunError(f"{tag}: packet build is not deterministic")
    parsed_z0, parsed_z1, parsed_params = rx.parse_packet(packet)
    parse_back_exact = (
        np.array_equal(parsed_z0, z0) and np.array_equal(parsed_z1, z1)
        and parsed_params.to_bytes() == parameters.to_bytes()
    )
    if not parse_back_exact:
        raise GDC2RunError(f"{tag}: packet did not parse back exactly")
    row: dict[str, Any] = {
        "tag": tag,
        "packet": packet_fact,
        "packet_repeat": repeat_fact,
        "deterministic_repeat_equal": True,
        "receiver_parse_back_exact": True,
        "packet_bytes": facts["packet_bytes"],
        "packet_detail": facts,
        "latent_bits_per_symbol": (
            8.0 * facts["packet_bytes"] / (z0.size + z1.size)
        ),
        "nonzero_latent_fraction": float(
            (np.count_nonzero(z0) + np.count_nonzero(z1)) / (z0.size + z1.size)
        ),
    }
    if full_render:
        measured = render_and_count(parsed_z0, parsed_z1, parsed_params,
                                    directory / "render.u8", teacher, field)
        row.update(measured)
        row["parity"] = parity_check(parsed_z0, parsed_z1, parsed_params,
                                     directory / "render.u8", 8, seed)
        row["projected_total_at_teacher_rate"] = (
            row["packet_bytes"] + TEACHER_REAL_RESIDUAL_BYTES
            / TEACHER_MISMATCHES * row["mismatches_vs_field"]
        )
    row["evaluate_seconds"] = time.monotonic() - started
    atomic_json(directory / "EVAL.json", row)
    return row


def close_residual(render_path: Path, field: np.ndarray, root: Path) -> dict[str, Any]:
    """Code the REAL exact residual back to the move-43 field with the real coder."""
    from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1

    generated = np.memmap(render_path, dtype=np.uint8, mode="r").reshape(
        rx.N_PAIRS, rx.HEIGHT, rx.WIDTH)
    rows: list[dict[str, Any]] = []
    for order in RESIDUAL_ORDERS:
        raw_path = root / order / "residual.raw"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        fact = hg1.encode_residual(field, generated, raw_path, None, order)
        race = hg1.coder_race(f"gdc2_closure_{order}", raw_path, root)
        winner = str(race["winner"])
        coded = race["coders"][winner]["coded"]
        rows.append({"order": order, "raw": fact, "winner": winner,
                     "coded": coded, "coded_bytes": int(coded["bytes"])})
    best = min(rows, key=lambda item: (item["coded_bytes"], item["order"]))
    corrected = np.array(generated, copy=True)
    hg1.apply_residual(Path(best["raw"]["path"]).read_bytes(), corrected)
    exact = bool(np.array_equal(corrected, np.asarray(field)))
    del corrected
    if not exact:
        raise GDC2RunError("closure residual did not restore the move-43 field exactly")
    return {"rows": rows, "best_order": best["order"],
            "best_coded_bytes": best["coded_bytes"], "exact_field_closure": True}


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def run_stage(stage: str, steps: int, lr_high: float, lr_low: float, lambda_rate: float,
              params: dict[str, mx.array], ema: dict[str, mx.array], root: Path,
              schedule: np.ndarray, schedule_offset: int, teacher: np.ndarray,
              field: np.ndarray, config: dict[str, Any], evaluate_every: int,
              dominating: list[dict[str, Any]] | None) -> dict[str, Any]:
    directory = root / "stages" / stage
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = optim.AdamW(learning_rate=lr_high, weight_decay=WEIGHT_DECAY)
    optimizer.init(params)
    start = 0
    resumed = latest_checkpoint(directory / "ckpt")
    if resumed is not None:
        start = load_checkpoint(resumed, params, ema, optimizer)
    heartbeat = (directory / "heartbeat.jsonl").open("a", encoding="utf-8")
    evaluations: list[dict[str, Any]] = []
    eval_path = directory / "EVALUATIONS.json"
    if eval_path.is_file():
        evaluations = json.loads(eval_path.read_text())
    dominated_strikes = 0
    early_stop = None

    def objective(current: dict[str, mx.array], blocks: list[Any], labels: mx.array):
        return loss_terms(current, blocks, labels, lambda_rate)

    grad_fn = mx.value_and_grad(
        lambda current, blocks, labels: objective(current, blocks, labels)[0]
    )
    window_start = time.monotonic()
    for step in range(start, steps):
        batch = step_batch(schedule[schedule_offset + step], teacher)
        optimizer.learning_rate = cosine_lr(step, steps, lr_high, lr_low)
        loss, grads = grad_fn(params, batch["blocks"], batch["labels"])
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params, optimizer.state, loss)
        for name in ema:
            ema[name] = EMA_DECAY * ema[name] + (1.0 - EMA_DECAY) * params[name]
        mx.eval(ema)
        if (step + 1) % CHECKPOINT_EVERY == 0 or step + 1 == steps:
            _, cross_entropy, entropy_bits = objective(params, batch["blocks"], batch["labels"])
            mx.eval(cross_entropy, entropy_bits)
            elapsed = time.monotonic() - window_start
            window_start = time.monotonic()
            heartbeat.write(json.dumps({
                "stage": stage, "step": step + 1, "loss": float(loss),
                "cross_entropy_nats": float(cross_entropy),
                "latent_entropy_bits_per_symbol": float(entropy_bits),
                "learning_rate": float(optimizer.learning_rate),
                "seconds_per_checkpoint_window": elapsed,
                "peak_rss_gib": peak_rss_gib(),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "score_claim": False, "research_only": True,
            }) + "\n")
            heartbeat.flush()
            save_checkpoint(directory / "ckpt" / f"step_{step + 1:07d}.npz",
                            params, ema, optimizer, step + 1, config)
            if not math.isfinite(float(loss)):
                early_stop = f"numerical divergence at step {step + 1}"
                break
        due = evaluate_every and (step + 1) % evaluate_every == 0
        if due and step + 1 != steps:
            row = evaluate_export(f"{stage}_step{step + 1:07d}", root, ema, teacher,
                                  field, config["seed"], True)
            row["step"] = step + 1
            evaluations.append(row)
            atomic_json(eval_path, evaluations)
            if dominating:
                reference = next(
                    (item for item in dominating if item.get("step") == step + 1), None)
                if reference and (
                    reference["packet_bytes"] <= row["packet_bytes"]
                    and reference["mismatches_vs_field"] <= row["mismatches_vs_field"]
                ):
                    dominated_strikes += 1
                else:
                    dominated_strikes = 0
                if dominated_strikes >= 2:
                    early_stop = f"pareto-dominated at two consecutive evaluations (step {step + 1})"
                    break
    heartbeat.close()
    final = evaluate_export(f"{stage}_final", root, ema, teacher, field, config["seed"], True)
    final["step"] = steps if early_stop is None else step + 1
    evaluations.append(final)
    atomic_json(eval_path, evaluations)
    save_checkpoint(directory / f"stage_end_{stage}.npz", params, ema, optimizer,
                    final["step"], config)
    return {"stage": stage, "steps": steps, "completed_steps": final["step"],
            "lambda_rate": lambda_rate, "early_stop": early_stop,
            "evaluations": evaluations, "final": final,
            "params": params, "ema": ema}


def load_params_only(path: Path) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    params: dict[str, mx.array] = {}
    ema: dict[str, mx.array] = {}
    with np.load(path) as archive:
        for key in archive.files:
            if key.startswith("param::"):
                params[key[7:]] = mx.array(archive[key])
            elif key.startswith("ema::"):
                ema[key[5:]] = mx.array(archive[key])
    if not params or set(params) != set(ema):
        raise GDC2RunError(f"checkpoint {path} is not a complete parameter export")
    return params, ema


def free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=rx.SEED)
    parser.add_argument("--stage-a-steps", type=int, default=STAGE_A_STEPS)
    parser.add_argument("--stage-b-steps", type=int, default=STAGE_B_STEPS)
    parser.add_argument("--stage-c-steps", type=int, default=STAGE_C_STEPS)
    parser.add_argument("--evaluate-every", type=int, default=2_000,
                        help="full-n600 authority evaluation cadence inside stages B and C")
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    parser.add_argument("--minimum-free-gib", type=float, default=12.0,
                        help="refuse to start when the custody volume has less free space")
    parser.add_argument("--skip-closure", action="store_true",
                        help="smoke only: skip the real exact-residual closure race")
    parser.add_argument("--scope-note", type=str, default="",
                        help="declared SCOPE reduction for a parity smoke; produces no verdict")
    args = parser.parse_args(argv)

    started = time.monotonic()
    root: Path = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    available = free_bytes(root)
    if available < args.minimum_free_gib * 1024**3:
        raise GDC2RunError(
            f"custody volume has {available / 1024**3:.2f} GiB free, "
            f"below the declared {args.minimum_free_gib} GiB floor")

    mx.random.seed(args.seed)
    field = load_verified(FIELD_PATH, FIELD_SHA256)
    teacher = load_verified(TEACHER_PATH, TEACHER_SHA256)

    total_steps = args.stage_a_steps + args.stage_b_steps + args.stage_c_steps
    schedule_path = root / "tile_schedule.npy"
    if schedule_path.is_file():
        schedule = np.load(schedule_path)
        if schedule.shape[0] < total_steps:
            raise GDC2RunError("retained tile schedule is shorter than the requested run")
    else:
        schedule = rx.build_tile_schedule(total_steps, BLOCKS_PER_STEP, PAIRS_PER_BLOCK, TILE)
        np.save(schedule_path, schedule)

    config = {
        "seed": args.seed,
        "architecture": {
            "z0_shape": list(rx.Z0_SHAPE), "z1_shape": list(rx.Z1_SHAPE),
            "width": rx.WIDTH_CH, "blocks": rx.BLOCKS, "kernel": rx.KERNEL,
            "classes": rx.CLASSES,
            "shifts": {"stem": rx.SHIFT_STEM, "depthwise": rx.SHIFT_DW,
                       "pointwise": rx.SHIFT_PW, "input": rx.INPUT_SHIFT},
            "head_loss_divisor": rx.HEAD_LOSS_DIVISOR,
        },
        "schedule": {"blocks_per_step": BLOCKS_PER_STEP, "pairs_per_block": PAIRS_PER_BLOCK,
                     "tile": TILE, "window": WINDOW, "halo": HALO},
        "stage_steps": {"A": args.stage_a_steps, "B": args.stage_b_steps, "C": args.stage_c_steps},
        "lambdas": list(args.lambdas),
        "ema_decay": EMA_DECAY, "weight_decay": WEIGHT_DECAY,
        "scope_note": args.scope_note,
    }
    atomic_json(root / "CONFIG.json", config)

    params = initial_parameters(args.seed)
    ema = {name: mx.array(value) for name, value in params.items()}
    stage_a_end = root / "stages" / "stageA" / "stage_end_stageA.npz"
    if stage_a_end.is_file():
        stage_a = {"stage": "stageA", "resumed_from_retained_stage_end": True,
                   "evaluations": json.loads(
                       (root / "stages" / "stageA" / "EVALUATIONS.json").read_text())}
        params, ema = load_params_only(stage_a_end)
    else:
        stage_a = run_stage("stageA", args.stage_a_steps, 3e-3, 3e-4, 0.0, params, ema,
                            root, schedule, 0, teacher, field, config, 0, None)
        params, ema = stage_a.pop("params"), stage_a.pop("ema")
        params, ema = load_params_only(stage_a_end)

    branches: list[dict[str, Any]] = []
    completed_b: list[dict[str, Any]] = []
    for index, lambda_rate in enumerate(args.lambdas):
        label = f"lam{index}_{lambda_rate:g}"
        base_params, base_ema = load_params_only(stage_a_end)
        stage_b = run_stage(f"stageB_{label}", args.stage_b_steps, 1e-3, 1e-4, lambda_rate,
                            base_params, base_ema, root, schedule, args.stage_a_steps,
                            teacher, field, config | {"lambda_rate": lambda_rate},
                            args.evaluate_every,
                            completed_b[0]["evaluations"] if completed_b else None)
        stage_b.pop("params"), stage_b.pop("ema")
        completed_b.append(stage_b)
        b_end = root / "stages" / f"stageB_{label}" / f"stage_end_stageB_{label}.npz"
        c_params, c_ema = load_params_only(b_end)
        stage_c = run_stage(f"stageC_{label}", args.stage_c_steps, 1e-4, 1e-4, lambda_rate,
                            c_params, c_ema, root, schedule,
                            args.stage_a_steps + args.stage_b_steps, teacher, field,
                            config | {"lambda_rate": lambda_rate, "qat": True},
                            args.evaluate_every, None)
        stage_c.pop("params"), stage_c.pop("ema")
        branches.append({"label": label, "lambda_rate": lambda_rate,
                         "stage_b": stage_b, "stage_c": stage_c})

    closures: list[dict[str, Any]] = []
    for branch in [] if args.skip_closure else branches:
        final = branch["stage_c"]["final"]
        closure_root = root / "closure" / branch["label"]
        closure = close_residual(Path(final["render"]["path"]), field, closure_root)
        total = int(final["packet_bytes"]) + int(closure["best_coded_bytes"])
        closures.append({
            "label": branch["label"], "lambda_rate": branch["lambda_rate"],
            "packet_bytes": int(final["packet_bytes"]),
            "mismatches_vs_field": int(final["mismatches_vs_field"]),
            "mismatches_vs_teacher": int(final["mismatches_vs_teacher"]),
            "lane_share_of_field_mismatches": final["lane_share_of_field_mismatches"],
            "real_residual_bytes": int(closure["best_coded_bytes"]),
            "real_bytes_per_mismatch": (
                closure["best_coded_bytes"] / final["mismatches_vs_field"]
                if final["mismatches_vs_field"] else 0.0),
            "packet_plus_real_residual": total,
            "replacement_integer_cap": REPLACEMENT_INTEGER_CAP,
            "over_gate_bytes": total - REPLACEMENT_INTEGER_CAP,
            "gate_pass": total <= REPLACEMENT_INTEGER_CAP,
            "decode_seconds": final["decode_seconds"],
            "decode_under_900s": final["decode_seconds"] <= 900.0,
            "parity": final["parity"],
            "closure": closure,
        })
    best = min(closures, key=lambda row: row["packet_plus_real_residual"]) if closures else None

    result = {
        "schema": "ddm_gdc2_categorical_coolchic_distill.v1",
        "axis": "[macOS-MLX research-signal training; macOS-CPU scorer-free exact-field receiver measurement, n600]",
        "score_claim": False, "research_only": True, "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "config": config,
        "inputs": {"field": file_fact(FIELD_PATH), "teacher": file_fact(TEACHER_PATH),
                   "teacher_packet_bytes": TEACHER_PACKET_BYTES,
                   "teacher_mismatches": TEACHER_MISMATCHES,
                   "teacher_real_residual_bytes": TEACHER_REAL_RESIDUAL_BYTES},
        "bar": {
            "replacement_integer_cap": REPLACEMENT_INTEGER_CAP,
            "real_packet_cap_at_exact_teacher_identity":
                REPLACEMENT_INTEGER_CAP - TEACHER_REAL_RESIDUAL_BYTES,
            "pointer_archive_bytes": POINTER_ARCHIVE_BYTES, "pointer_s": POINTER_S,
        },
        "stage_a": stage_a,
        "branches": branches,
        "closures": closures,
        "best": best,
        "gate_pass": bool(best and best["gate_pass"]),
        "frontier_moved": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gib": peak_rss_gib(),
        "free_bytes_at_start": available,
        "host": os.uname().nodename,
    }
    atomic_json(root / "RESULT.json", result)
    print(json.dumps({"gate_pass": result["gate_pass"], "best": best and {
        k: best[k] for k in ("label", "packet_bytes", "mismatches_vs_field",
                             "real_residual_bytes", "packet_plus_real_residual",
                             "over_gate_bytes")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
