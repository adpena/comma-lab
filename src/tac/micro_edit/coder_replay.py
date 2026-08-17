# SPDX-License-Identifier: MIT
"""Exact code-length replay for the live HPAC token coder (charter stage 7).

This is the engine's DECODE-IDENTICAL measurement surface. It replays the shipped
decode order over the retained HPAC logits and the real decoded token field, and
sums ``-log2 P(actual)`` to get the exact code length in bits for ANY probability
law we care to test.

Why this surface matters more than it looks. Every other micro-edit family changes
the tokens, so its distortion must be re-measured through the render + scorer chain
-- and the local advisory instrument's d_pose sits 21x above the contest-CUDA value
(measured: 1.4747e-4 local vs 6.88e-6 T4), so no local run can admit one. A change
to the PROBABILITY LAW alone decodes to a bit-identical token field, so:

* distortion is unchanged BY CONSTRUCTION (provable by token-field sha, not assumed);
* bytes are exactly measurable here, locally, at zero cost;

which makes this the one axis the engine can decide on its own, with no scorer run
and no paid dispatch.

POSITIVE CONTROLS (both must hold or no verdict from this module is admissible):

1. ``uncorrected`` must reproduce the shipped HPAC cross-entropy
   112,109.57757858819 B (ddm_hm1 / ddm_dc1). MEASURED: exact to 0.000000 B.
2. ``live`` (the rr4 corrector) must reproduce the shipped corrected code length
   from ``bits_per_frame_corrected.npy``.

A replay that misses either control is a broken instrument and its numbers are
discarded -- the apparatus points its skepticism at itself first.

The probability construction is a verbatim transcription of
``runtime.residual_archive._probability_table`` + the table-correction step in
``experiments/ddm_hm1_hpac_logit_replay.py``:

    base      = int16_logit / HPAC_LOGIT_PRECISION      (exact, logits are k/8)
    feature   = boundary * 5 + argmax(base)
    corrected = base + table.values[feature]
    quantized = clip(rint(corrected * 8), -32768, 32767) -> int16
    values    = quantized / 8  -> float64
    p         = softmax(values) -> float32

STORES CONSULTED
----------------
* ``experiments/ddm_hm1_hpac_logit_replay.py`` -- the mapping + the expected
  cross-entropy constant.
* ``experiments/ddm_rr4_free_corrector_v2.py`` -- the LIVE probability law.
* ``/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/RESULT_build.json`` -- the
  shipped token-stream bytes the code length must be judged against.
"""
from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "EXPECTED_UNCORRECTED_CROSS_ENTROPY_BYTES",
    "HPAC_LOGIT_PRECISION",
    "NUM_CLASSES",
    "PLANE",
    "ReplayAssets",
    "ReplayResult",
    "load_corrector_module",
    "replay_code_length",
]

HEIGHT: int = 384
WIDTH: int = 512
PLANE: int = HEIGHT * WIDTH
NUM_FRAMES: int = 600
NUM_CLASSES: int = 5
HPAC_LOGIT_PRECISION: int = 8

EXPECTED_UNCORRECTED_CROSS_ENTROPY_BYTES: float = 112_109.57757858819
"""ddm_hm1 / ddm_dc1 shipped HPAC cross-entropy. Positive control #1."""


@dataclass(frozen=True)
class ReplayAssets:
    """Paths to the retained inputs the replay consumes (all read-only)."""

    logits_i16: Path
    tokens_u8: Path
    boundary_u8: Path
    group_index_u8: Path
    table_values_npy: Path

    def validate(self) -> None:
        for name in (
            "logits_i16",
            "tokens_u8",
            "boundary_u8",
            "group_index_u8",
            "table_values_npy",
        ):
            path = getattr(self, name)
            if not Path(path).exists():
                raise FileNotFoundError(f"replay asset {name} missing: {path}")


@dataclass(frozen=True)
class ReplayResult:
    """Exact code length for one probability law over the replayed field."""

    label: str
    code_bits: float
    frames: int
    positions: int
    bits_per_frame: np.ndarray

    @property
    def code_bytes(self) -> float:
        return self.code_bits / 8.0

    def delta_bytes_vs(self, other: ReplayResult) -> float:
        return self.code_bytes - other.code_bytes


def load_corrector_module(path: Path, name: str = "_me1_corrector") -> Any:
    """Import a corrector module from an explicit path.

    The engine drives the REAL shipped corrector rather than a reimplementation, so
    the live-law control cannot drift from what actually ships.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load corrector from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _group_positions(group_index: np.ndarray) -> list[np.ndarray]:
    """Flat plane indices per causal group, in decode order."""
    order = np.argsort(group_index, kind="stable")
    boundaries = np.searchsorted(group_index[order], np.arange(group_index.max() + 2))
    return [
        order[boundaries[g] : boundaries[g + 1]].astype(np.int64)
        for g in range(group_index.max() + 1)
    ]


def _frame_probabilities(
    base_logits: np.ndarray, boundary: np.ndarray, table_values: np.ndarray
) -> np.ndarray:
    """The shipped base->probability map, transcribed verbatim (see module docstring)."""
    predicted = base_logits.argmax(axis=1)
    feature = boundary.astype(np.int64) * NUM_CLASSES + predicted
    corrected = base_logits + table_values[feature]
    quantized = np.clip(
        np.rint(corrected.astype(np.float32) * HPAC_LOGIT_PRECISION), -32768, 32767
    ).astype(np.int16)
    values = (quantized.astype(np.float32) / HPAC_LOGIT_PRECISION).astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def replay_code_length(
    assets: ReplayAssets,
    *,
    label: str,
    corrector_factory: Callable[[int], Any] | None = None,
    frames: int = NUM_FRAMES,
    progress: Callable[[int, float], None] | None = None,
) -> ReplayResult:
    """Sum ``-log2 P(actual)`` over the replayed field.

    ``corrector_factory`` builds an online corrector exposing the shipped driving
    API (``begin_frame`` / ``group_state`` / ``coding_row`` / ``observe`` /
    ``end_frame``). Pass ``None`` to measure the uncorrected HPAC law, which is
    positive control #1.

    ``frames`` below 600 is a MECHANISM smoke only. A prefix of this field is a
    different population (the wallclock/prefix-bias law), so no byte verdict may be
    drawn from one; the caller is responsible for reporting the scale it used.
    """
    assets.validate()
    table_values = np.load(assets.table_values_npy).astype(np.float32)
    group_index = np.fromfile(assets.group_index_u8, dtype=np.uint8).astype(np.int64)
    if group_index.size != PLANE:
        raise ValueError(f"group index must cover the plane, got {group_index.size}")
    groups = _group_positions(group_index)

    logits = np.memmap(assets.logits_i16, dtype=np.int16, mode="r")
    tokens = np.memmap(assets.tokens_u8, dtype=np.uint8, mode="r")
    boundary_all = np.memmap(assets.boundary_u8, dtype=np.uint8, mode="r")

    corrector = corrector_factory(PLANE) if corrector_factory is not None else None
    per_frame = np.zeros(frames, dtype=np.float64)

    for frame in range(frames):
        base = (
            np.asarray(logits[frame * PLANE * NUM_CLASSES : (frame + 1) * PLANE * NUM_CLASSES])
            .reshape(PLANE, NUM_CLASSES)
            .astype(np.float32)
            / HPAC_LOGIT_PRECISION
        )
        boundary = np.asarray(boundary_all[frame * PLANE : (frame + 1) * PLANE])
        token = np.asarray(tokens[frame * PLANE : (frame + 1) * PLANE]).astype(np.int64)
        probability = _frame_probabilities(base, boundary, table_values)
        predicted = base.argmax(axis=1).astype(np.int64)

        if corrector is None:
            chosen = probability[np.arange(PLANE), token].astype(np.float64)
            per_frame[frame] = -float(np.log2(np.maximum(chosen, 1e-300)).sum())
        else:
            corrector.begin_frame(boundary)
            total = 0.0
            for flat in groups:
                state = corrector.group_state(probability[flat], predicted[flat], flat)
                row = corrector.coding_row(state)
                actual = token[flat]
                chosen = row[np.arange(actual.size), actual].astype(np.float64)
                total -= float(np.log2(np.maximum(chosen, 1e-300)).sum())
                corrector.observe(state, actual)
            corrector.end_frame(token.astype(np.uint8))
            per_frame[frame] = total

        if progress is not None and (frame + 1) % 50 == 0:
            progress(frame + 1, float(per_frame[: frame + 1].sum()))

    return ReplayResult(
        label=label,
        code_bits=float(per_frame.sum()),
        frames=frames,
        positions=frames * PLANE,
        bits_per_frame=per_frame,
    )
