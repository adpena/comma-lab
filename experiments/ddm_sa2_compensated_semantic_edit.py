"""ddm_sa2 — pose-COMPENSATED semantic-tensor edit on the rr4 base.

Executes the sa1 family's recorded reactivation criterion #1: re-solve the
frame-0 carrier (the qs1/qs5 Schur machinery) IN-COMPILE against the ACTUAL
S2-quantized semantic tensor, never carrying a compensation across objects
(the qs4 lesson).

Structural facts established by this arm's own preflight, not assumed:

* ``cpr1/inflate.py::render_video`` writes ``output[2p+1]`` (frame_1) from the
  semantic renderer and ``output[2p]`` (frame_0) from the carrier alone.  The
  S2 semantic edit therefore damages frame_1 ONLY; frame_0 is byte-identical
  between the base and S2 receiver outputs (asserted per sampled pair).
* frame_0 is a 12-dimensional signed-int12 actuator per pair (CPR1 carrier
  coefficients over a normalized 24x32 RGB basis), so the qs5 geometry
  applies unchanged.

Axis: ``[macOS-CPU advisory frozen CPU-torch PoseNet]`` — score_claim=false,
promotion_eligible=false.  No Modal, no GPU, no lane claim, no scorer-lane
fire.  The n600 advisory adjudication is MAIN's, against the bought base leg.

SCOPE REDUCTION (declared): the solve runs on a seeded RANDOM subset of pairs
(never a prefix, per the prefix-bias law).  The n600 projection is the subset
mean of the exact per-pair d_pose terms, which is an unbiased estimator of the
population mean because ``upstream/evaluate.py`` averages a per-pair MSE over
the 600 pairs.  Reported with its standard error.  This reduces SCOPE (n), not
MECHANISM: every rendered object, PoseNet call, and integer descent step is
the exact shipping receiver realization at full resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
UPSTREAM: Final = REPO / "upstream"
RR4: Final = Path("/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode")
RR4_RUNTIME: Final = RR4 / "candidate_runtime"
BASE_ARCHIVE: Final = RR4_RUNTIME / "archive.zip"
BASE_ARCHIVE_SHA256: Final = (
    "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956"
)
SA1: Final = Path("/Volumes/APDataStore/pact/ddm_sa1")
S2_ARCHIVE: Final = SA1 / "generations/S2_film23_q2_top3_q3/archive.zip"
S2_ARCHIVE_SHA256: Final = (
    "a36890b6541cf259b3f662996f8c3a935d0648aa977d02d30992aaa1e4feae29"
)
BASE_RAW: Final = (
    SA1 / "advisory_n600_cpu/rr4_base/attempt_0002/work/inflated/0.raw"
)
S2_RAW: Final = (
    SA1
    / "advisory_n600_cpu/S2_film23_q2_top3_q3/attempt_0001/work/inflated/0.raw"
)
OUTPUT: Final = SA1 / "retained/sa2"

PAIR_COUNT: Final = 600
DIMENSIONS: Final = 12
POSE_DIMENSIONS: Final = 6
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
BASIS_H: Final = 24
BASIS_W: Final = 32
EVAL_H: Final = 384
EVAL_W: Final = 512
CARRIER_AMPLITUDE: Final = 64.0
INT12_MIN: Final = -2048
INT12_MAX: Final = 2047

UNCOMPRESSED_BYTES: Final = 37_545_489
RATE_S_PER_BYTE: Final = 25.0 / UNCOMPRESSED_BYTES

# The bought same-instrument base leg (sa1 advisory adjudication, attempt_0002).
BASE_D_SEG: Final = 0.00042714
BASE_D_POSE: Final = 0.00014747
BASE_BYTES: Final = 181_161
S2_D_SEG: Final = 0.00042886
S2_D_POSE: Final = 0.00098653
S2_BYTES: Final = 179_828
ADMIT_BAR_S: Final = -3.5e-6

GN_DAMPING: Final = 0.01
POSE_BATCH: Final = 8
DESCENT_LADDER: Final = (64, 16, 4, 1)
JACOBIAN_STEPS: Final = (64, 8)

AXIS: Final = (
    "[macOS-CPU advisory frozen CPU-torch PoseNet; seeded-random pair subset] "
    "NON-PROMOTABLE"
)


class SA2Error(RuntimeError):
    """A sa2 precondition, instrument control, or geometry check failed."""


# --------------------------------------------------------------------------
# retention (ALWAYS KEEP THE PAYLOAD)
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return file_record(path)


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    return _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode())


def retain_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
    os.replace(temporary, path)
    return file_record(path)


# --------------------------------------------------------------------------
# the exact rr4 frame-0 actuator
# --------------------------------------------------------------------------


@dataclass
class RR4Frame0Surface:
    """The exact rr4 receiver's signed-int12 frame-0 actuator."""

    codes: np.ndarray  # (600, 12) effective codes, post shipped Q2C1 overlay
    raw_codes: np.ndarray  # (600, 12) pre-overlay lattice
    scales: np.ndarray  # (12,)
    normalized_basis: Any  # torch (12, 3, 384, 512)
    selector_modes: Sequence[Any]
    selector_indices: np.ndarray | None
    rice_parameters: np.ndarray
    pins: dict[str, Any]

    @classmethod
    def load(cls) -> RR4Frame0Surface:
        import torch
        import torch.nn.functional as functional

        sys.path.insert(0, str(RR4_RUNTIME))
        sys.path.insert(0, str(RR4_RUNTIME / "cpr1"))
        try:
            import carrier_codec as carrier_codec_module
            from runtime.carrier_repack import (
                materialize_cpr1,
                split_frame0_selector_carrier,
            )
            from runtime.compensation_overlay import apply_compensation_overlay
            from runtime.frame0_selector import decode_selector
            from runtime.residual_archive import read_residual_archive
        finally:
            sys.path.pop(0)
            sys.path.pop(0)

        parts = read_residual_archive(BASE_ARCHIVE)
        _, selector = split_frame0_selector_carrier(parts.carrier_blob)
        canonical = materialize_cpr1(
            parts.carrier_blob,
            SimpleNamespace(N=PAIR_COUNT, CARRIER_DIM=DIMENSIONS),
        )
        basis_scales, basis_codes, coefficient_scales, encoded = (
            carrier_codec_module.decode_compact_carrier(
                canonical,
                basis_count=DIMENSIONS * 3 * BASIS_H * BASIS_W,
                frames=PAIR_COUNT,
                dimensions=DIMENSIONS,
            )
        )
        delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
        raw_codes = np.cumsum(delta, axis=0) & 0xFFF
        raw_codes = np.where(
            raw_codes >= 0x800, raw_codes - 0x1000, raw_codes
        ).astype(np.int32)
        if parts.compensation_blob is None:
            codes = raw_codes.copy()
        else:
            codes = np.asarray(
                apply_compensation_overlay(raw_codes, parts.compensation_blob),
                dtype=np.int32,
            )

        header = carrier_codec_module.HEADER
        cursor = (
            header.size
            + 2 * DIMENSIONS * 4
            + carrier_codec_module.ALPHABET_SIZE
        )
        rice = np.frombuffer(
            canonical[cursor : cursor + DIMENSIONS], dtype=np.uint8
        ).copy()

        basis = torch.from_numpy(
            basis_codes.reshape(DIMENSIONS, 3, BASIS_H, BASIS_W).astype(np.float32)
        ) * torch.from_numpy(basis_scales)[:, None, None, None]
        basis = functional.interpolate(
            basis, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False
        )
        basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
        basis = basis / basis.square().mean(
            dim=(1, 2, 3), keepdim=True
        ).sqrt().clamp_min(1e-5)

        if selector is None:
            modes: Sequence[Any] = ()
            indices = None
        else:
            modes, indices = decode_selector(selector)

        pins = {
            "base_archive": file_record(BASE_ARCHIVE),
            "canonical_cpr1_bytes": len(canonical),
            "canonical_cpr1_sha256": hashlib.sha256(canonical).hexdigest(),
            "carrier_blob_bytes": len(parts.carrier_blob),
            "carrier_blob_sha256": hashlib.sha256(parts.carrier_blob).hexdigest(),
            "compensation_blob_bytes": (
                None
                if parts.compensation_blob is None
                else len(parts.compensation_blob)
            ),
            "shipped_overlay_changed_coordinates": int(
                np.count_nonzero(codes != raw_codes)
            ),
            "rice_parameters": rice.tolist(),
            "coefficient_scales": coefficient_scales.tolist(),
        }
        return cls(
            codes=codes,
            raw_codes=raw_codes,
            scales=coefficient_scales,
            normalized_basis=basis,
            selector_modes=modes,
            selector_indices=indices,
            rice_parameters=rice,
            pins=pins,
        )

    def render(self, code_rows: np.ndarray, pair: int) -> np.ndarray:
        """Exact receiver realization of frame_0 for a batch of code rows."""
        import torch
        import torch.nn.functional as functional

        values = np.asarray(code_rows, dtype=np.int32).reshape(-1, DIMENSIONS)
        if np.any(values < INT12_MIN) or np.any(values > INT12_MAX):
            raise SA2Error("carrier candidate exceeds signed-int12")
        coefficient = torch.from_numpy(
            np.ascontiguousarray(values.astype(np.float32) * self.scales[None])
        )
        with torch.inference_mode():
            carrier = torch.einsum(
                "bk,kchw->bchw", coefficient, self.normalized_basis
            )
            carrier = carrier / math.sqrt(DIMENSIONS)
            evaluated = (
                (127.5 + CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round()
            )
            slave = (
                functional.interpolate(
                    evaluated,
                    size=(CAMERA_H, CAMERA_W),
                    mode="bicubic",
                    align_corners=False,
                )
                .clamp(0.0, 255.0)
                .round()
            )
            result = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        if self.selector_indices is not None:
            sys.path.insert(0, str(RR4_RUNTIME))
            try:
                from runtime.frame0_selector import apply_pixel_mode
            finally:
                sys.path.pop(0)
            result = apply_pixel_mode(
                result, self.selector_modes[int(self.selector_indices[pair])]
            )
        return np.asarray(result, dtype=np.uint8)


def load_posenet() -> Any:
    import torch
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import PoseNet, posenet_sd_path
    finally:
        sys.path.pop(0)
    network = PoseNet().eval().cpu()
    network.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    torch.set_grad_enabled(False)
    return network


def pose_vectors(posenet: Any, pairs: np.ndarray) -> np.ndarray:
    """PoseNet first-six on (B, 2, H, W, 3) uint8, the exact evaluator path."""
    import torch

    value = np.asarray(pairs)
    if value.ndim != 5 or value.shape[1:] != (2, CAMERA_H, CAMERA_W, 3):
        raise SA2Error(f"PoseNet input geometry differs: {value.shape}")
    with torch.inference_mode():
        tensor = (
            torch.from_numpy(np.ascontiguousarray(value))
            .permute(0, 1, 4, 2, 3)
            .float()
        )
        output = posenet(posenet.preprocess_input(tensor))["pose"][
            ..., :POSE_DIMENSIONS
        ]
    return output.cpu().numpy().astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------


def decode_gt_pairs(pairs: Sequence[int]) -> np.ndarray:
    """Canonical GT decode (frame_utils.yuv420_to_rgb); never PyAV rgb24."""
    import av

    sys.path.insert(0, str(UPSTREAM))
    try:
        from frame_utils import yuv420_to_rgb
    finally:
        sys.path.pop(0)

    wanted: dict[int, int] = {}
    for slot, pair in enumerate(pairs):
        wanted[2 * int(pair)] = slot
        wanted[2 * int(pair) + 1] = slot
    result = np.zeros((len(pairs), 2, CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    seen = 0
    container = av.open(str(UPSTREAM / "videos/0.mkv"))
    try:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            slot = wanted.get(index)
            if slot is None:
                continue
            array = np.asarray(yuv420_to_rgb(frame), dtype=np.uint8)
            if array.shape != (CAMERA_H, CAMERA_W, 3):
                raise SA2Error(f"GT frame geometry differs: {array.shape}")
            result[slot, index % 2] = array
            seen += 1
            if seen == 2 * len(pairs):
                break
    finally:
        container.close()
    if seen != 2 * len(pairs):
        raise SA2Error("GT decode did not reach every requested frame")
    return result


# --------------------------------------------------------------------------
# the in-compile Schur solve, bound to the exact edited object
# --------------------------------------------------------------------------


def damped_least_squares(
    jacobian: np.ndarray, target: np.ndarray, damping: float
) -> tuple[np.ndarray, int, float]:
    """Ridge-regularized minimum-norm update solving J x ~= target."""
    matrix = np.asarray(jacobian, dtype=np.float64)
    rhs = np.asarray(target, dtype=np.float64)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular.max() * 1e-10)) if singular.size else 0
    condition = (
        float(singular.max() / singular.min())
        if singular.size and singular.min() > 0
        else float("inf")
    )
    ridge = damping * (singular.max() if singular.size else 1.0)
    factors = singular / (singular**2 + ridge**2)
    update = vt.T @ (factors * (u.T @ rhs))
    return update, rank, condition


class PairEvaluator:
    """Renders exact frame_0 candidates against a FIXED frame_1 and scores them."""

    def __init__(
        self,
        *,
        surface: RR4Frame0Surface,
        posenet: Any,
        master: np.ndarray,
        pair: int,
        target: np.ndarray | None = None,
    ) -> None:
        self.surface = surface
        self.posenet = posenet
        self.master = np.ascontiguousarray(master)
        self.pair = int(pair)
        self.target = target
        self.cache: dict[bytes, np.ndarray] = {}
        self.calls = 0
        self.renders = 0

    def vectors(self, codes: Sequence[np.ndarray]) -> np.ndarray:
        rows = [np.asarray(item, dtype=np.int32) for item in codes]
        pending = [
            row for row in rows if row.tobytes() not in self.cache
        ]
        unique: dict[bytes, np.ndarray] = {}
        for row in pending:
            unique.setdefault(row.tobytes(), row)
        todo = list(unique.values())
        for first in range(0, len(todo), POSE_BATCH):
            batch = np.stack(todo[first : first + POSE_BATCH])
            slaves = self.surface.render(batch, self.pair)
            masters = np.repeat(self.master[None], len(batch), axis=0)
            inputs = np.stack((slaves, masters), axis=1)
            vectors = pose_vectors(self.posenet, inputs)
            self.calls += 1
            self.renders += len(batch)
            for row, vector in zip(batch, vectors, strict=True):
                self.cache[row.tobytes()] = vector
        return np.stack([self.cache[row.tobytes()] for row in rows])

    def objectives(self, codes: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self.target is None:
            raise SA2Error("evaluator has no objective target")
        vectors = self.vectors(codes)
        values = np.mean(
            np.square(vectors.astype(np.float64) - self.target[None]), axis=1
        )
        return vectors, values


def jacobian_at(
    evaluator: PairEvaluator, codes: np.ndarray, step: int
) -> np.ndarray:
    """Exact receiver-realized central difference over the 12 int12 coordinates."""
    candidates: list[np.ndarray] = []
    for dimension in range(DIMENSIONS):
        for sign in (-1, 1):
            candidate = codes.copy()
            candidate[dimension] = int(
                np.clip(candidate[dimension] + sign * step, INT12_MIN, INT12_MAX)
            )
            candidates.append(candidate)
    vectors = evaluator.vectors(candidates)
    jacobian = np.empty((POSE_DIMENSIONS, DIMENSIONS), dtype=np.float64)
    for dimension in range(DIMENSIONS):
        minus = vectors[2 * dimension].astype(np.float64)
        plus = vectors[2 * dimension + 1].astype(np.float64)
        jacobian[:, dimension] = (plus - minus) / (2.0 * step)
    return jacobian


def multiscale_descent(
    evaluator: PairEvaluator,
    codes: np.ndarray,
    value: float,
    ladder: Sequence[int],
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Exact integer descent; each scale stops on a full non-improving pass."""
    current = np.asarray(codes, dtype=np.int32).copy()
    best = float(value)
    trace: list[dict[str, Any]] = []
    for step in ladder:
        passes = 0
        while True:
            candidates: list[np.ndarray] = [current.copy()]
            for dimension in range(DIMENSIONS):
                for sign in (-1, 1):
                    candidate = current.copy()
                    moved = int(candidate[dimension]) + sign * step
                    if not INT12_MIN <= moved <= INT12_MAX:
                        continue
                    candidate[dimension] = moved
                    candidates.append(candidate)
            _, values = evaluator.objectives(candidates)
            index = min(
                range(len(candidates)),
                key=lambda position: (float(values[position]), position),
            )
            passes += 1
            if not float(values[index]) < best:
                break
            current = candidates[index]
            best = float(values[index])
        trace.append({"step": int(step), "full_passes": passes, "objective": best})
    return current, best, trace


def solve_pair(
    *,
    surface: RR4Frame0Surface,
    posenet: Any,
    pair: int,
    master_base: np.ndarray,
    master_edited: np.ndarray,
    gt_pair: np.ndarray,
    root: Path,
) -> dict[str, Any]:
    result_path = root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    started = time.perf_counter()
    base_codes = surface.codes[pair].copy()

    baseline = PairEvaluator(
        surface=surface, posenet=posenet, master=master_base, pair=pair
    )
    base_vector = baseline.vectors([base_codes])[0]

    evaluator = PairEvaluator(
        surface=surface,
        posenet=posenet,
        master=master_edited,
        pair=pair,
        target=base_vector,
    )
    event_vector = evaluator.vectors([base_codes])[0]
    leak = event_vector.astype(np.float64) - base_vector.astype(np.float64)

    _, event_objective = evaluator.objectives([base_codes])
    current = base_codes.copy()
    best = float(event_objective[0])
    solves: list[dict[str, Any]] = []
    for step in JACOBIAN_STEPS:
        jacobian = jacobian_at(evaluator, current, step)
        residual_now = (
            evaluator.vectors([current])[0].astype(np.float64)
            - base_vector.astype(np.float64)
        )
        update, rank, condition = damped_least_squares(
            jacobian, -residual_now, GN_DAMPING
        )
        proposal = np.clip(
            np.rint(current.astype(np.float64) + update), INT12_MIN, INT12_MAX
        ).astype(np.int32)
        _, values = evaluator.objectives([proposal])
        solves.append(
            {
                "jacobian_step": int(step),
                "jacobian_rank": rank,
                "jacobian_condition": condition,
                "float_update": update.tolist(),
                "update_linf": float(np.max(np.abs(update))),
                "update_l2": float(np.linalg.norm(update)),
                "objective_before": best,
                "objective_at_proposal": float(values[0]),
                "accepted": bool(float(values[0]) < best),
            }
        )
        if float(values[0]) < best:
            current = proposal
            best = float(values[0])

    final_codes, final_objective, descent = multiscale_descent(
        evaluator, current, best, DESCENT_LADDER
    )
    final_vector = evaluator.vectors([final_codes])[0]
    residual = final_vector.astype(np.float64) - base_vector.astype(np.float64)

    gt_vector = pose_vectors(posenet, gt_pair[None])[0].astype(np.float64)
    def dpose(vector: np.ndarray) -> float:
        return float(
            np.mean(np.square(np.asarray(vector, dtype=np.float64) - gt_vector))
        )

    d_base = dpose(base_vector)
    d_event = dpose(event_vector)
    d_final = dpose(final_vector)

    leak_energy = float(leak @ leak)
    residual_energy = float(residual @ residual)
    cancellation = (
        1.0 - residual_energy / leak_energy if leak_energy > 0 else float("nan")
    )
    damage = d_event - d_base
    remaining = d_final - d_base
    dpose_cancellation = 1.0 - remaining / damage if damage != 0 else float("nan")

    retain_npy(root / "base_codes.int32.npy", base_codes)
    retain_npy(root / "final_codes.int32.npy", final_codes)
    retain_npy(root / "base_pose_vector.float32.npy", base_vector)
    retain_npy(root / "event_pose_vector.float32.npy", event_vector)
    retain_npy(root / "final_pose_vector.float32.npy", final_vector)
    retain_npy(root / "gt_pose_vector.float64.npy", gt_vector)
    codes_seen = np.stack(
        [np.frombuffer(key, dtype=np.int32) for key in evaluator.cache]
    )
    retain_npy(root / "all_evaluated_codes.int32.npy", codes_seen)
    retain_npy(
        root / "all_evaluated_pose_vectors.float32.npy",
        np.stack(list(evaluator.cache.values())),
    )
    retain_npy(root / "final_frame0.uint8.npy", surface.render(final_codes[None], pair)[0])

    result = {
        "schema": "ddm_sa2_pair_result.v1",
        "pair": int(pair),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "seconds": time.perf_counter() - started,
        "posenet_batches": evaluator.calls + baseline.calls,
        "frame0_realizations": evaluator.renders + baseline.renders,
        "codes": {
            "base": base_codes.tolist(),
            "final": final_codes.tolist(),
            "delta": (final_codes - base_codes).tolist(),
            "delta_linf": int(np.max(np.abs(final_codes - base_codes))),
            "delta_l1": int(np.sum(np.abs(final_codes - base_codes))),
            "changed_coordinates": int(np.count_nonzero(final_codes != base_codes)),
        },
        "solve": {
            "gauss_newton": solves,
            "descent": descent,
            "final_objective_mse_to_base_pose_vector": final_objective,
        },
        "pose": {
            "base_vector": base_vector.tolist(),
            "event_vector": event_vector.tolist(),
            "final_vector": final_vector.tolist(),
            "leak_l2": math.sqrt(leak_energy),
            "residual_l2": math.sqrt(residual_energy),
            "cancellation_energy_fraction": cancellation,
            "d_pose_pair_base": d_base,
            "d_pose_pair_event_uncompensated": d_event,
            "d_pose_pair_compensated": d_final,
            "d_pose_pair_damage_uncompensated": damage,
            "d_pose_pair_damage_remaining": remaining,
            "d_pose_damage_cancellation_fraction": dpose_cancellation,
        },
        "verdict_scope": (
            "INSTANCE: exact rr4 frame-0 int12 actuator x the S2_film23_q2_top3_q3 "
            "semantic quantization on this pair; local frozen CPU PoseNet advisory"
        ),
    }
    retain_json(result_path, result)
    return result


# --------------------------------------------------------------------------
# rate model for the re-solved carrier
# --------------------------------------------------------------------------


def rice_bits(encoded: np.ndarray, parameters: np.ndarray) -> int:
    total = 0
    values = np.asarray(encoded, dtype=np.int64)
    for dimension, parameter in enumerate(parameters):
        k = int(parameter)
        total += int((values[:, dimension] >> k).sum())
        total += values.shape[0] * (1 + k)
    return total


def zigzag(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64)
    return ((v << 1) ^ (v >> 63)).astype(np.int64)


def carrier_coefficient_bits(codes: np.ndarray, parameters: np.ndarray) -> int:
    """Rice cost of the shipped temporal-delta zigzag coefficient stream."""
    lattice = np.asarray(codes, dtype=np.int64)
    delta = np.diff(lattice, axis=0, prepend=np.zeros((1, DIMENSIONS), dtype=np.int64))
    wrapped = ((delta + 0x800) & 0xFFF) - 0x800
    encoded = zigzag(wrapped)
    if np.any(encoded < 0) or np.any(encoded >= (1 << 12)):
        raise SA2Error("re-encoded coefficient exceeds the 12-bit Rice domain")
    return rice_bits(encoded, parameters)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


CONTROL_PAIRS: Final = 24


def preflight(output: Path, pairs: np.ndarray, surface: RR4Frame0Surface) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_00_preflight.json"
    if checkpoint.is_file():
        return json.loads(checkpoint.read_text())
    # The instrument controls are a property of the receiver realization, not of
    # any one pair, so they run on a bounded seeded-RANDOM control sample.
    if len(pairs) > CONTROL_PAIRS:
        control_rng = np.random.default_rng(20260818_0)
        pairs = np.sort(
            control_rng.choice(np.asarray(pairs), size=CONTROL_PAIRS, replace=False)
        )
    for path, expected in ((BASE_ARCHIVE, BASE_ARCHIVE_SHA256), (S2_ARCHIVE, S2_ARCHIVE_SHA256)):
        digest = sha256_file(path)
        if digest != expected:
            raise SA2Error(f"archive sha differs: {path} {digest}")
    sys.path.insert(0, str(RR4_RUNTIME))
    try:
        from runtime.residual_archive import read_residual_archive
    finally:
        sys.path.pop(0)
    base_parts = read_residual_archive(BASE_ARCHIVE)
    edited_parts = read_residual_archive(S2_ARCHIVE)
    for name in ("hpac_blob", "carrier_blob", "token_stream", "residual_payload"):
        if getattr(base_parts, name) != getattr(edited_parts, name):
            raise SA2Error(f"S2 changed a non-semantic section: {name}")
    if base_parts.semantic_blob == edited_parts.semantic_blob:
        raise SA2Error("S2 semantic section is identical to base")

    base_raw = np.memmap(
        BASE_RAW, dtype=np.uint8, mode="r", shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3)
    )
    edited_raw = np.memmap(
        S2_RAW, dtype=np.uint8, mode="r", shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3)
    )
    controls = []
    for pair in pairs:
        pair = int(pair)
        rendered = surface.render(surface.codes[pair : pair + 1], pair)[0]
        mismatch = int(np.count_nonzero(rendered != np.asarray(base_raw[2 * pair])))
        if mismatch:
            raise SA2Error(
                f"frame-0 render is not the exact receiver raw: pair {pair}, "
                f"mismatched_values={mismatch}"
            )
        if not np.array_equal(
            np.asarray(base_raw[2 * pair]), np.asarray(edited_raw[2 * pair])
        ):
            raise SA2Error(f"S2 changed frame_0 on pair {pair}")
        difference = np.asarray(edited_raw[2 * pair + 1]).astype(np.int32) - np.asarray(
            base_raw[2 * pair + 1]
        ).astype(np.int32)
        controls.append(
            {
                "pair": pair,
                "frame0_render_mismatched_values": mismatch,
                "frame0_base_equals_edited": True,
                "frame1_rgb_rms": float(np.sqrt((difference.astype(np.float64) ** 2).mean())),
                "frame1_abs_max": int(np.abs(difference).max()),
            }
        )

    shipped_bits = carrier_coefficient_bits(surface.codes, surface.rice_parameters)
    record = {
        "schema": "ddm_sa2_preflight.v1",
        "axis": AXIS,
        "pairs": [int(value) for value in pairs],
        "pair_selection": "seeded RANDOM (never a prefix), rng 20260818",
        "surface_pins": surface.pins,
        "instrument_controls": controls,
        "shipped_coefficient_rice_bits": shipped_bits,
        "base_raw": {"path": str(BASE_RAW), "bytes": BASE_RAW.stat().st_size},
        "edited_raw": {"path": str(S2_RAW), "bytes": S2_RAW.stat().st_size},
    }
    retain_json(checkpoint, record)
    return record


def aggregate(output: Path, rows: list[dict[str, Any]], surface: RR4Frame0Surface) -> dict[str, Any]:
    damage = np.array(
        [row["pose"]["d_pose_pair_damage_uncompensated"] for row in rows], dtype=np.float64
    )
    remaining = np.array(
        [row["pose"]["d_pose_pair_damage_remaining"] for row in rows], dtype=np.float64
    )
    count = len(rows)
    mean_damage = float(damage.mean())
    mean_remaining = float(remaining.mean())
    se_remaining = float(remaining.std(ddof=1) / math.sqrt(count)) if count > 1 else float("nan")

    # Instrument-agreement control: my frozen CPU PoseNet vs the bought base leg.
    base_here = float(
        np.mean([row["pose"]["d_pose_pair_base"] for row in rows])
    )
    compensated_here = float(
        np.mean([row["pose"]["d_pose_pair_compensated"] for row in rows])
    )
    event_here = float(
        np.mean([row["pose"]["d_pose_pair_event_uncompensated"] for row in rows])
    )

    projected_d_pose = BASE_D_POSE + mean_remaining
    projected_high = BASE_D_POSE + mean_remaining + 1.96 * se_remaining
    base_pose_s = math.sqrt(10.0 * BASE_D_POSE)

    def pose_term(value: float) -> float:
        return math.sqrt(10.0 * max(value, 0.0)) - base_pose_s

    delta_bytes_semantic = S2_BYTES - BASE_BYTES
    seg_s = 100.0 * (S2_D_SEG - BASE_D_SEG)

    # exact ceiling on the d_pose the candidate may carry at zero extra bytes
    budget_pose_s = ADMIT_BAR_S - seg_s - delta_bytes_semantic * RATE_S_PER_BYTE
    ceiling_d_pose = ((base_pose_s + budget_pose_s) ** 2) / 10.0

    record = {
        "schema": "ddm_sa2_aggregate.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pairs_measured": count,
        "subset_mean_uncompensated_damage_d_pose": mean_damage,
        "subset_mean_remaining_damage_d_pose": mean_remaining,
        "subset_remaining_standard_error": se_remaining,
        "mean_cancellation_fraction_of_d_pose_damage": (
            1.0 - mean_remaining / mean_damage if mean_damage else float("nan")
        ),
        "n600_reference": {
            "base_d_pose": BASE_D_POSE,
            "uncompensated_d_pose": S2_D_POSE,
            "measured_n600_damage": S2_D_POSE - BASE_D_POSE,
            "subset_recovers_fraction_of_n600_damage": (
                mean_damage / (S2_D_POSE - BASE_D_POSE)
            ),
        },
        "instrument_agreement": {
            "note": (
                "this arm's frozen CPU PoseNet vs the bought advisory base leg; "
                "a ratio near 1.0 means the two instruments agree and the "
                "subset d_pose values are directly comparable to the base row"
            ),
            "base_d_pose_measured_here": base_here,
            "base_d_pose_bought_leg": BASE_D_POSE,
            "base_ratio_here_over_leg": base_here / BASE_D_POSE,
            "uncompensated_d_pose_measured_here": event_here,
            "uncompensated_d_pose_bought_leg": S2_D_POSE,
            "uncompensated_ratio_here_over_leg": event_here / S2_D_POSE,
            "compensated_d_pose_measured_here": compensated_here,
        },
        "admit_arithmetic": {
            "delta_bytes_before_compensation": delta_bytes_semantic,
            "rate_s": delta_bytes_semantic * RATE_S_PER_BYTE,
            "seg_s": seg_s,
            "admit_bar_s": ADMIT_BAR_S,
            "pose_budget_s_at_zero_extra_bytes": budget_pose_s,
            "d_pose_ceiling_at_zero_extra_bytes": ceiling_d_pose,
            "required_cancellation_of_d_pose_damage": (
                1.0 - (ceiling_d_pose - BASE_D_POSE) / (S2_D_POSE - BASE_D_POSE)
            ),
            "projected_compensated_d_pose": projected_d_pose,
            "projected_compensated_d_pose_upper_95": projected_high,
            "projected_pose_s": pose_term(projected_d_pose),
            "projected_net_delta_s": (
                delta_bytes_semantic * RATE_S_PER_BYTE + seg_s + pose_term(projected_d_pose)
            ),
            "projected_net_delta_s_upper_95": (
                delta_bytes_semantic * RATE_S_PER_BYTE + seg_s + pose_term(projected_high)
            ),
        },
        "code_moves": {
            "delta_linf_by_pair": [row["codes"]["delta_linf"] for row in rows],
            "delta_l1_by_pair": [row["codes"]["delta_l1"] for row in rows],
            "changed_coordinates_by_pair": [
                row["codes"]["changed_coordinates"] for row in rows
            ],
        },
        "rows": rows,
    }
    retain_json(output / "AGGREGATE.json", record)
    return record


def run(
    output: Path,
    pair_count: int,
    seed: int,
    shard: int = 0,
    shards: int = 1,
    aggregate_only: bool = False,
    reverse: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    surface = RR4Frame0Surface.load()
    rng = np.random.default_rng(seed)
    pairs = np.sort(rng.choice(PAIR_COUNT, size=pair_count, replace=False))
    if aggregate_only:
        rows = []
        for pair in pairs:
            path = output / f"pairs/pair_{int(pair):04d}/RESULT.json"
            if path.is_file():
                rows.append(json.loads(path.read_text()))
        if not rows:
            raise SA2Error("no retained pair results to aggregate")
        print(f"aggregating {len(rows)}/{len(pairs)} retained pair results")
        return aggregate(output, rows, surface)
    control = preflight(output, pairs, surface)
    pairs = pairs[int(shard) :: int(shards)]
    if reverse:
        # Idempotent helper: every solved pair short-circuits on its retained
        # RESULT.json, so a reverse worker simply meets the forward one.
        pairs = pairs[::-1]
    print(json.dumps(control["instrument_controls"], indent=2))

    posenet = load_posenet()
    gt = decode_gt_pairs([int(value) for value in pairs])
    # GT camera frames are a deterministic decode of a read-only upstream input,
    # so they are certified-rebuildable rather than dumped (3.66 GB per shard).
    # The derived measurement -- the GT PoseNet-6 vector -- is retained per pair.
    retain_json(
        output / f"checkpoints/gt_decode_receipt_shard_{shard:02d}.json",
        {
            "schema": "ddm_sa2_gt_decode_receipt.v1",
            "source": file_record(UPSTREAM / "videos/0.mkv"),
            "decoder": "upstream/frame_utils.py::yuv420_to_rgb (canonical; never PyAV rgb24)",
            "pairs": [int(value) for value in pairs],
            "frames_decoded": int(2 * len(pairs)),
            "rebuild": (
                "experiments/ddm_sa2_compensated_semantic_edit.py::decode_gt_pairs"
            ),
            "retained_derivative": "pairs/pair_<p>/gt_pose_vector.float64.npy",
        },
    )

    base_raw = np.memmap(
        BASE_RAW, dtype=np.uint8, mode="r", shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3)
    )
    edited_raw = np.memmap(
        S2_RAW, dtype=np.uint8, mode="r", shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3)
    )
    rows: list[dict[str, Any]] = []
    for slot, pair in enumerate(pairs):
        pair = int(pair)
        row = solve_pair(
            surface=surface,
            posenet=posenet,
            pair=pair,
            master_base=np.asarray(base_raw[2 * pair + 1]),
            master_edited=np.asarray(edited_raw[2 * pair + 1]),
            gt_pair=gt[slot],
            root=output / f"pairs/pair_{pair:04d}",
        )
        rows.append(row)
        pose = row["pose"]
        print(
            f"pair {pair:3d}  leak {pose['leak_l2']:.6f} -> residual {pose['residual_l2']:.6f}  "
            f"cancel(energy) {pose['cancellation_energy_fraction']*100:6.2f}%  "
            f"d_pose damage {pose['d_pose_pair_damage_uncompensated']:.3e} -> "
            f"{pose['d_pose_pair_damage_remaining']:.3e}  "
            f"|dc|inf {row['codes']['delta_linf']:5d}  {row['seconds']:.0f}s",
            flush=True,
        )
    if int(shards) != 1:
        return {"schema": "ddm_sa2_shard.v1", "shard": int(shard), "shards": int(shards),
                "pairs_solved": [int(value) for value in pairs]}
    return aggregate(output, rows, surface)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    torch.set_num_threads(int(args.threads))
    record = run(
        args.output,
        int(args.pairs),
        int(args.seed),
        shard=int(args.shard),
        shards=int(args.shards),
        aggregate_only=bool(args.aggregate_only),
        reverse=bool(args.reverse),
    )
    if "admit_arithmetic" in record:
        print(json.dumps(record["admit_arithmetic"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
