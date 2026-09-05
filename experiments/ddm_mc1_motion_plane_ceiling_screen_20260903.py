#!/usr/bin/env python3
"""MC1 closed-form ceiling for a decoder-derived motion-compensated prior.

This is a scorer-free SCREEN on the exact AFR1 categorical field.  It builds
and retains three full-n600 motion-compensated previous-field planes:

* ``global_translation``: one class-balanced, edge-weighted integer shift;
* ``row_tile_translation``: one integer shift per shipped 64-row HPAC band;
* ``affine_block_fit``: an exact-integer affine fit to 64x64 block shifts.

For pair ``t >= 2`` every motion estimate uses only fields ``t-2`` and
``t-1`` and is reused as a constant-velocity prediction from ``t-1`` to
``t``.  Pairs 0 and 1 fall back to the co-located previous plane.  The
categorical screen is the MI1/DDS1 family: a pair-level two-fold cross-fit of
one log-odds offset per decoder-visible context cell on top of the retained
HPAC coding-row pmax.  SCREEN bits are not physical coder bytes; division by
eight is used only as an optimistic refusal ceiling.

All materialized fields, motion parameters, fitted offset tables, stage
checkpoints, and the selected determinism repeat are retained under
``/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane``.
No training, real coder, archive mutation, scorer, Modal, or Metal path is
present in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
STORE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane"
)
TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/identity_v1/"
    "out/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
EXACT_NULL_RECEIPT: Final = Path(
    "/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/retained/exact/null/RESULT.json"
)
DF1_ROOT: Final = Path(
    "/Volumes/APDataStore/pact/ddm_df1_dddb_field/measurement_v1/retained/fields"
)
ARGMAX: Final = DF1_ROOT / "position_coding_argmax.u8.bin"
PMAX: Final = DF1_ROOT / "position_coding_pmax.f32le.bin"

N: Final = 600
H: Final = 384
W: Final = 512
PLANE: Final = H * W
CLASSES: Final = 5
PATCH: Final = 64
ROW_BANDS: Final = H // PATCH
TILE_ROWS: Final = H // PATCH
TILE_COLS: Final = W // PATCH
TILES: Final = TILE_ROWS * TILE_COLS
PAIR_FOLD_SEED: Final = 20_260_903
SEARCH_RADII: Final = (4, 8, 12)
AFFINE_Q: Final = 1 << 12
MAX_AFFINE_SHIFT: Final = SEARCH_RADII[-1] * 2
CHECKPOINT_EVERY: Final = 20
MINIMUM_AP_FREE_BYTES: Final = int(1.5 * (1 << 30))
CEILING_FIRE_BYTES: Final = 5_000.0
DEMAND_BYTES: Final = 42_016.0
AXIS: Final = "[macOS-CPU scorer-free conditional-codelength SCREEN, n600]"
MODELS: Final = (
    "global_translation",
    "row_tile_translation",
    "affine_block_fit",
)

EXPECTED: Final = {
    "tokens": (
        117_964_800,
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    ),
    "argmax": (
        117_964_800,
        "db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e",
    ),
    "pmax": (
        471_859_200,
        "f37e3d8a21d02647437bf950d7a8a75b751c2a9644c7b8ad48aca2833be4794b",
    ),
    "exact_archive": (
        180_002,
        "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25",
    ),
}

RECALL_QUERIES: Final = (
    "motion.compens|motion aligned|previous field|previous plane|warp context",
    "constant.velocity|row-dependent shift|affine|temporal alignment",
    "INTER-CAE|Wyner-Ziv|decoder side information|field_geometry_temporal",
    "previous decoded class|full previous frame|temporal IoU",
)

RECALL_SOURCES: Final = (
    ".omx/research/ddm_xi1_carried_xi_inter_race_20260729.md",
    ".omx/research/ddm_d3b_lossless_lane_factorization_20260826.md",
    ".omx/research/ddm_dds1_decoder_derivable_verdict_20260901.md",
    ".omx/research/ddm_dds1_ceiling_readjudication_20260901.md",
    ".omx/research/ddm_mi1_indicator_model_axis_20260824.md",
    ".omx/research/ddm_dc1_decode_budget_conditional_coding_20260816.md",
    ".omx/research/ddm_qbw2_temporal_bound_verdict_20260827.md",
    ".omx/research/ddm_dcc1_decoder_causal_conditioning_verdict_20260901.md",
    ".omx/research/ddm_dv3_divergent_weird_ideas_20260818.md",
    ".omx/research/generator_description_online_survey_20260719.md",
    ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md",
    "docs/operating_manual_craft_handoff.md",
    ".omx/state/main_hot_state.md",
)


class Mc1Error(RuntimeError):
    """A custody, causality, determinism, or ceiling invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require(path: Path, expected: tuple[int, str], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Mc1Error(f"missing {label}: {path}")
    fact = file_fact(path)
    if (fact["bytes"], fact["sha256"]) != expected:
        raise Mc1Error(f"{label} drifted: {fact} != {expected}")
    return fact


def require_fact(fact: dict[str, Any], label: str) -> dict[str, Any]:
    """Verify a receipt-carried path/size/hash fact against retained bytes."""
    required = {"path", "bytes", "sha256"}
    if set(fact) != required:
        raise Mc1Error(f"{label} fact keys changed: {sorted(fact)}")
    return require(
        Path(str(fact["path"])),
        (int(fact["bytes"]), str(fact["sha256"])),
        label,
    )


def _edge_mask(field: np.ndarray) -> np.ndarray:
    edge = np.zeros(field.shape, dtype=bool)
    edge[1:] |= field[1:] != field[:-1]
    edge[:-1] |= field[:-1] != field[1:]
    edge[:, 1:] |= field[:, 1:] != field[:, :-1]
    edge[:, :-1] |= field[:, :-1] != field[:, 1:]
    return edge


def _integer_match_weights(target: np.ndarray) -> np.ndarray:
    """Class-balance a region and emphasize its coder-relevant boundaries."""
    counts = np.bincount(target.reshape(-1), minlength=CLASSES)
    class_weight = np.zeros(CLASSES, dtype=np.int64)
    present = counts > 0
    class_weight[present] = target.size // counts[present]
    weight = class_weight[target]
    weight *= np.where(_edge_mask(target), 4, 1)
    return weight


def _translation_score(
    source: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    dy: int,
    dx: int,
    y0: int,
    y1: int,
) -> int:
    dst_y0 = max(y0, 0, dy)
    dst_y1 = min(y1, H, H + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)
    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        return 0
    source_view = source[
        dst_y0 - dy : dst_y1 - dy,
        dst_x0 - dx : dst_x1 - dx,
    ]
    target_view = target[dst_y0:dst_y1, dst_x0:dst_x1]
    weight_view = weight[dst_y0 - y0 : dst_y1 - y0, dst_x0:dst_x1]
    return int(weight_view[source_view == target_view].sum(dtype=np.int64))


def estimate_translation(
    source: np.ndarray,
    target: np.ndarray,
    *,
    y0: int = 0,
    y1: int = H,
    radii: tuple[int, ...] = SEARCH_RADII,
) -> tuple[int, int, int, int, int, bool]:
    """Return the exact deterministic shift maximizing integer agreement.

    Search expands only when the optimum reaches the current window boundary.
    Ties prefer smaller motion, then lexicographic ``(dy, dx)``.
    """
    if source.shape != (H, W) or target.shape != (H, W):
        raise Mc1Error("translation source geometry changed")
    if not (0 <= y0 < y1 <= H):
        raise Mc1Error("translation row band is invalid")
    weight = _integer_match_weights(target[y0:y1])
    zero_score = _translation_score(source, target, weight, 0, 0, y0, y1)
    best: tuple[int, int, int, int] | None = None
    used_radius = radii[0]
    for radius in radii:
        used_radius = radius
        best = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                score = _translation_score(source, target, weight, dy, dx, y0, y1)
                candidate = (-score, abs(dy) + abs(dx), dy, dx)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise Mc1Error("translation search was empty")
        if abs(best[2]) < radius and abs(best[3]) < radius:
            break
    assert best is not None
    boundary_hit = abs(best[2]) == used_radius or abs(best[3]) == used_radius
    return best[2], best[3], -best[0], zero_score, used_radius, boundary_hit


def warp_translation(source: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate with co-located fallback outside the valid overlap."""
    output = source.copy()
    dst_y0, dst_y1 = max(0, dy), min(H, H + dy)
    dst_x0, dst_x1 = max(0, dx), min(W, W + dx)
    if dst_y0 < dst_y1 and dst_x0 < dst_x1:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = source[
            dst_y0 - dy : dst_y1 - dy,
            dst_x0 - dx : dst_x1 - dx,
        ]
    return output


def warp_row_shifts(source: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Apply one destination-row-band translation with co-located fallback."""
    if shifts.shape != (ROW_BANDS, 2):
        raise Mc1Error("row-shift parameter shape changed")
    output = source.copy()
    for band, (dy_raw, dx_raw) in enumerate(shifts):
        dy, dx = int(dy_raw), int(dx_raw)
        y0, y1 = band * PATCH, (band + 1) * PATCH
        dst_y0, dst_y1 = max(y0, dy), min(y1, H + dy)
        dst_x0, dst_x1 = max(0, dx), min(W, W + dx)
        if dst_y0 < dst_y1 and dst_x0 < dst_x1:
            output[dst_y0:dst_y1, dst_x0:dst_x1] = source[
                dst_y0 - dy : dst_y1 - dy,
                dst_x0 - dx : dst_x1 - dx,
            ]
    return output


def _det3(matrix: list[list[int]]) -> int:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _round_fraction_away(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise Mc1Error("singular affine fit")
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    return sign * ((2 * magnitude + denominator) // (2 * denominator))


def _integer_median(values: np.ndarray) -> int:
    """Return an exact integer median with half-away rounding for even counts."""
    ordered = np.sort(np.asarray(values, dtype=np.int64).reshape(-1))
    if ordered.size == 0:
        raise Mc1Error("integer median received no values")
    middle = ordered.size // 2
    if ordered.size % 2:
        return int(ordered[middle])
    return _round_fraction_away(
        int(ordered[middle - 1]) + int(ordered[middle]),
        2,
    )


def fit_affine_integer(
    local_shifts: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Fit dy/dx affine fields with exact integer normal equations.

    Coordinates are twice-centered integers, coefficients use Q12 shifts, and
    Cramer's rule plus half-away rounding avoids a platform float solver.
    """
    if local_shifts.shape != (TILES, 2) or weights.shape != (TILES,):
        raise Mc1Error("affine local parameter geometry changed")
    rows: list[tuple[list[int], int]] = []
    for tile in range(TILES):
        tile_y, tile_x = divmod(tile, TILE_COLS)
        center_x = tile_x * PATCH + PATCH // 2
        center_y = tile_y * PATCH + PATCH // 2
        features = [1, 2 * center_x - (W - 1), 2 * center_y - (H - 1)]
        rows.append((features, max(1, int(weights[tile]))))
    normal = [[0 for _ in range(3)] for _ in range(3)]
    rhs = [[0 for _ in range(3)] for _ in range(2)]
    for tile, (features, weight) in enumerate(rows):
        for i in range(3):
            for j in range(3):
                normal[i][j] += weight * features[i] * features[j]
            for axis in range(2):
                rhs[axis][i] += (
                    weight * features[i] * int(local_shifts[tile, axis]) * AFFINE_Q
                )
    denominator = _det3(normal)
    if denominator == 0:
        median = [_integer_median(local_shifts[:, axis]) for axis in range(2)]
        return np.asarray(
            [[median[0] * AFFINE_Q, 0, 0], [median[1] * AFFINE_Q, 0, 0]],
            dtype=np.int64,
        )
    coefficients = np.zeros((2, 3), dtype=np.int64)
    for axis in range(2):
        for column in range(3):
            replaced = [row.copy() for row in normal]
            for row in range(3):
                replaced[row][column] = rhs[axis][row]
            coefficients[axis, column] = _round_fraction_away(
                _det3(replaced), denominator
            )
    return coefficients


def _round_q_array(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    magnitude = np.abs(values)
    rounded = (magnitude + AFFINE_Q // 2) // AFFINE_Q
    return np.where(positive, rounded, -rounded).astype(np.int64)


def warp_affine(source: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Nearest-neighbour integer affine warp with co-located fallback."""
    if coefficients.shape != (2, 3):
        raise Mc1Error("affine coefficient shape changed")
    yy, xx = np.indices((H, W), dtype=np.int64)
    x2 = 2 * xx - (W - 1)
    y2 = 2 * yy - (H - 1)
    dy_q = coefficients[0, 0] + coefficients[0, 1] * x2 + coefficients[0, 2] * y2
    dx_q = coefficients[1, 0] + coefficients[1, 1] * x2 + coefficients[1, 2] * y2
    dy = np.clip(_round_q_array(dy_q), -MAX_AFFINE_SHIFT, MAX_AFFINE_SHIFT)
    dx = np.clip(_round_q_array(dx_q), -MAX_AFFINE_SHIFT, MAX_AFFINE_SHIFT)
    source_y = yy - dy
    source_x = xx - dx
    valid = (source_y >= 0) & (source_y < H) & (source_x >= 0) & (source_x < W)
    output = source.copy()
    output[valid] = source[source_y[valid], source_x[valid]]
    return output


def _empty_parameters(model: str) -> dict[str, np.ndarray]:
    if model == "global_translation":
        return {
            "shifts": np.zeros((N, 2), dtype=np.int16),
            "used_radius": np.zeros(N, dtype=np.uint8),
            "boundary_hit": np.zeros(N, dtype=np.uint8),
            "score_gain": np.zeros(N, dtype=np.int64),
        }
    if model == "row_tile_translation":
        return {
            "shifts": np.zeros((N, ROW_BANDS, 2), dtype=np.int16),
            "used_radius": np.zeros((N, ROW_BANDS), dtype=np.uint8),
            "boundary_hit": np.zeros((N, ROW_BANDS), dtype=np.uint8),
            "score_gain": np.zeros((N, ROW_BANDS), dtype=np.int64),
        }
    if model == "affine_block_fit":
        return {
            "coefficients_q12": np.zeros((N, 2, 3), dtype=np.int64),
            "local_shifts": np.zeros((N, TILES, 2), dtype=np.int16),
            "local_weights": np.zeros((N, TILES), dtype=np.int64),
            "used_radius": np.zeros((N, TILES), dtype=np.uint8),
            "boundary_hit": np.zeros((N, TILES), dtype=np.uint8),
        }
    raise Mc1Error(f"unknown model: {model}")


def _estimate_and_warp(
    model: str,
    older: np.ndarray,
    previous: np.ndarray,
    parameters: dict[str, np.ndarray],
    pair: int,
) -> np.ndarray:
    if model == "global_translation":
        dy, dx, score, zero, radius, hit = estimate_translation(older, previous)
        parameters["shifts"][pair] = (dy, dx)
        parameters["used_radius"][pair] = radius
        parameters["boundary_hit"][pair] = hit
        parameters["score_gain"][pair] = score - zero
        return warp_translation(previous, dy, dx)
    if model == "row_tile_translation":
        shifts = np.zeros((ROW_BANDS, 2), dtype=np.int16)
        for band in range(ROW_BANDS):
            dy, dx, score, zero, radius, hit = estimate_translation(
                older,
                previous,
                y0=band * PATCH,
                y1=(band + 1) * PATCH,
            )
            shifts[band] = (dy, dx)
            parameters["used_radius"][pair, band] = radius
            parameters["boundary_hit"][pair, band] = hit
            parameters["score_gain"][pair, band] = score - zero
        parameters["shifts"][pair] = shifts
        return warp_row_shifts(previous, shifts)
    if model == "affine_block_fit":
        local = np.zeros((TILES, 2), dtype=np.int16)
        weights = np.ones(TILES, dtype=np.int64)
        for tile in range(TILES):
            tile_y, _tile_x = divmod(tile, TILE_COLS)
            # The search region is the complete 64-row band.  Distinct x tiles
            # still receive distinct estimates by masking the score below.
            y0, y1 = tile_y * PATCH, (tile_y + 1) * PATCH
            tile_x = tile % TILE_COLS
            x0, x1 = tile_x * PATCH, (tile_x + 1) * PATCH
            target_tile = previous[y0:y1, x0:x1]
            weight_tile = _integer_match_weights(target_tile)
            best: tuple[int, int, int, int] | None = None
            zero = 0
            used_radius = SEARCH_RADII[0]
            for radius in SEARCH_RADII:
                used_radius = radius
                best = None
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        dst_y0, dst_y1 = max(y0, dy), min(y1, H + dy)
                        dst_x0, dst_x1 = max(x0, dx), min(x1, W + dx)
                        if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
                            score = 0
                        else:
                            src = older[
                                dst_y0 - dy : dst_y1 - dy,
                                dst_x0 - dx : dst_x1 - dx,
                            ]
                            dst = previous[dst_y0:dst_y1, dst_x0:dst_x1]
                            wgt = weight_tile[
                                dst_y0 - y0 : dst_y1 - y0,
                                dst_x0 - x0 : dst_x1 - x0,
                            ]
                            score = int(wgt[src == dst].sum(dtype=np.int64))
                        if dy == 0 and dx == 0:
                            zero = score
                        candidate = (-score, abs(dy) + abs(dx), dy, dx)
                        if best is None or candidate < best:
                            best = candidate
                assert best is not None
                if abs(best[2]) < radius and abs(best[3]) < radius:
                    break
            assert best is not None
            local[tile] = (best[2], best[3])
            weights[tile] = max(1, -best[0] - zero + 1)
            parameters["used_radius"][pair, tile] = used_radius
            parameters["boundary_hit"][pair, tile] = (
                abs(best[2]) == used_radius or abs(best[3]) == used_radius
            )
        coefficients = fit_affine_integer(local, weights)
        parameters["coefficients_q12"][pair] = coefficients
        parameters["local_shifts"][pair] = local
        parameters["local_weights"][pair] = weights
        return warp_affine(previous, coefficients)
    raise Mc1Error(f"unknown model: {model}")


def _load_checkpoint(path: Path, model: str) -> tuple[int, dict[str, np.ndarray]]:
    if not path.is_file():
        return 0, _empty_parameters(model)
    with np.load(path, allow_pickle=False) as blob:
        if str(blob["model"].item()) != model:
            raise Mc1Error(f"checkpoint model mismatch: {path}")
        next_pair = int(blob["next_pair"].item())
        parameters = {
            key: np.asarray(blob[key]).copy()
            for key in blob.files
            if key not in {"model", "next_pair"}
        }
    expected = _empty_parameters(model)
    if parameters.keys() != expected.keys():
        raise Mc1Error(f"checkpoint parameter keys changed: {path}")
    for key, value in expected.items():
        if parameters[key].shape != value.shape or parameters[key].dtype != value.dtype:
            raise Mc1Error(f"checkpoint parameter {key} changed: {path}")
    return next_pair, parameters


def _save_checkpoint(
    path: Path,
    model: str,
    next_pair: int,
    parameters: dict[str, np.ndarray],
) -> None:
    atomic_npz(
        path,
        model=np.asarray(model),
        next_pair=np.asarray(next_pair, dtype=np.int16),
        **parameters,
    )


def materialize_colocated() -> dict[str, Any]:
    destination = STORE / "retained/fields/colocated_previous.u8"
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        field = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, H, W))
        output = np.memmap(temporary, dtype=np.uint8, mode="w+", shape=(N, H, W))
        output[0] = 0
        output[1:] = field[:-1]
        output.flush()
        del output, field
        os.replace(temporary, destination)
    if destination.stat().st_size != N * PLANE:
        raise Mc1Error("co-located previous plane size changed")
    return file_fact(destination)


def materialize_model(model: str, *, repeat: bool = False) -> dict[str, Any]:
    suffix = ".repeat" if repeat else ""
    completion = STORE / f"stage_01_{model}{suffix}_complete.json"
    if completion.is_file():
        payload = json.loads(completion.read_text())
        for key in ("field", "motion_parameters", "terminal_checkpoint"):
            if file_fact(Path(payload[key]["path"])) != payload[key]:
                raise Mc1Error(f"completed {model}{suffix} artifact drifted")
        return payload
    output_path = STORE / f"retained/fields/{model}{suffix}.u8"
    checkpoint = STORE / f"checkpoints/{model}{suffix}.progress.npz"
    next_pair, parameters = _load_checkpoint(checkpoint, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.is_file():
        output = np.memmap(output_path, dtype=np.uint8, mode="w+", shape=(N, H, W))
        output[:] = 0
        output.flush()
        del output
        next_pair = 0
        parameters = _empty_parameters(model)
    if output_path.stat().st_size != N * PLANE:
        raise Mc1Error(f"partial {model}{suffix} field size changed")
    source = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, H, W))
    output = np.memmap(output_path, dtype=np.uint8, mode="r+", shape=(N, H, W))
    if next_pair <= 0:
        output[0] = 0
        next_pair = 1
    if next_pair <= 1:
        output[1] = source[0]
        next_pair = 2
    started = time.time()
    for pair in range(next_pair, N):
        output[pair] = _estimate_and_warp(
            model,
            np.asarray(source[pair - 2]),
            np.asarray(source[pair - 1]),
            parameters,
            pair,
        )
        if (pair + 1) % CHECKPOINT_EVERY == 0:
            output.flush()
            _save_checkpoint(checkpoint, model, pair + 1, parameters)
            atomic_json(
                STORE / f"checkpoints/{model}{suffix}.progress.json",
                {
                    "model": model,
                    "repeat": repeat,
                    "next_pair": pair + 1,
                    "elapsed_seconds_this_resume": time.time() - started,
                    "checkpoint": file_fact(checkpoint),
                },
            )
    output.flush()
    del output, source
    _save_checkpoint(checkpoint, model, N, parameters)
    parameter_path = STORE / f"retained/motion/{model}{suffix}.npz"
    atomic_npz(parameter_path, **parameters)
    payload = {
        "schema": "ddm_mc1_motion_plane_materialization.v1",
        "axis": AXIS,
        "model": model,
        "repeat": repeat,
        "pairs": N,
        "causality": (
            "pair t>=2 estimates field[t-2] -> field[t-1] and extrapolates the same "
            "integer transform onto field[t-1]; t=0,1 use co-located fallback"
        ),
        "field": file_fact(output_path),
        "motion_parameters": file_fact(parameter_path),
        "terminal_checkpoint": file_fact(checkpoint),
        "boundary_hit_count": int(parameters["boundary_hit"].sum()),
        "elapsed_seconds_this_resume": time.time() - started,
    }
    atomic_json(completion, payload)
    return payload


def stage_preflight() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STORE.parent).free
    if free < MINIMUM_AP_FREE_BYTES:
        raise Mc1Error(
            f"AP free-space refusal: {free} < {MINIMUM_AP_FREE_BYTES} bytes"
        )
    exact = json.loads(EXACT_NULL_RECEIPT.read_text())
    checkpoint_fact = exact.get("terminal_checkpoint", {}).get("checkpoint", {})
    if checkpoint_fact.get("sha256") is None:
        raise Mc1Error("exact-null receipt lacks terminal checkpoint custody")
    if exact.get("archive", {}).get("sha256") != EXPECTED["exact_archive"][1]:
        raise Mc1Error("exact-null receipt does not bind the AFR1 archive")
    if exact.get("schema") != "ddm_rxc1_exact_run.v1" or exact.get("frames_encoded") != N:
        raise Mc1Error("exact-null receipt is not the complete RXC1 n600 run")
    if json.loads((EXACT_NULL_RECEIPT.parent / "RUN_SPEC.json").read_text()).get(
        "tokens_sha256"
    ) != EXPECTED["tokens"][1]:
        raise Mc1Error("exact-null run spec does not bind the decoded AFR1 field")
    sources = {
        "tokens": require(TOKENS, EXPECTED["tokens"], "AFR1 decoded field"),
        "argmax": require(ARGMAX, EXPECTED["argmax"], "DF1 coding argmax"),
        "pmax": require(PMAX, EXPECTED["pmax"], "DF1 coding pmax"),
        "exact_archive": require(
            Path(str(exact["archive"]["path"])),
            EXPECTED["exact_archive"],
            "AFR1 exact-null archive",
        ),
        "exact_terminal_checkpoint": require_fact(
            checkpoint_fact,
            "AFR1 exact-null terminal checkpoint",
        ),
        "exact_null_receipt": file_fact(EXACT_NULL_RECEIPT),
    }
    recall = []
    for relative in RECALL_SOURCES:
        path = REPO / relative
        if not path.is_file():
            raise Mc1Error(f"recall source disappeared: {relative}")
        recall.append(file_fact(path))
    payload = {
        "schema": "ddm_mc1_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "git_head": git_head(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "storage": {
            "root": str(STORE),
            "free_bytes": free,
            "required_free_bytes": MINIMUM_AP_FREE_BYTES,
            "status": "PASS",
        },
        "sources": sources,
        "recall": {
            "queries": list(RECALL_QUERIES),
            "sources": recall,
            "canonical_equations_command": (
                ".venv/bin/python tools/list_canonical_equations.py --json"
            ),
            "beyond_charter": [
                "MPEG-4 INTER-CAE uses motion-compensated previous alpha context",
                "ddm_dv3 left the full-resolution semantic-field dictionary/context leg unrun",
                "QBW2 excludes its decoded-carrier translation from the gate because it is not geometric Pose6",
            ],
        },
        "authority_boundaries": {
            "scorer_runs": 0,
            "modal_calls": 0,
            "metal_runs": 0,
            "training_runs": 0,
            "real_coder_runs": 0,
            "upstream_writes": 0,
        },
    }
    atomic_json(STORE / "PREFLIGHT.json", payload)
    return payload


def stage_materialize() -> dict[str, Any]:
    stage_preflight()
    colocated = materialize_colocated()
    rows = [materialize_model(model) for model in MODELS]
    payload = {
        "schema": "ddm_mc1_materialize_complete.v1",
        "axis": AXIS,
        "colocated_previous": colocated,
        "models": rows,
    }
    atomic_json(STORE / "stage_01_materialize_complete.json", payload)
    return payload


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _fit_fold(
    logit: np.ndarray,
    flip: np.ndarray,
    cells: np.ndarray,
    train: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, int, float]:
    beta = np.zeros(n_cells, dtype=np.float64)
    train_cells = cells[train]
    train_logit = logit[train]
    train_flip = flip[train].astype(np.float64)
    iterations = 0
    last_step = 0.0
    for iteration in range(24):
        q = _sigmoid(train_logit + beta[train_cells])
        gradient = np.bincount(
            train_cells, weights=train_flip - q, minlength=n_cells
        )
        hessian = np.bincount(
            train_cells, weights=q * (1.0 - q), minlength=n_cells
        )
        step = np.divide(
            gradient,
            hessian,
            out=np.zeros_like(gradient),
            where=hessian > 0.0,
        )
        step = np.clip(step, -4.0, 4.0)
        beta += step
        iterations = iteration + 1
        last_step = float(np.max(np.abs(step)))
        if last_step < 1e-10:
            break
    return beta, iterations, last_step


def _crossfit_bits_by_class(
    logit: np.ndarray,
    flip: np.ndarray,
    true_class: np.ndarray,
    cells: np.ndarray,
    folds: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    total = np.zeros(CLASSES, dtype=np.float64)
    tables: dict[str, np.ndarray] = {}
    diagnostics = []
    for test_fold in (0, 1):
        train = folds != test_fold
        test = ~train
        beta, iterations, last_step = _fit_fold(
            logit, flip, cells, train, n_cells
        )
        u = logit[test] + beta[cells[test]]
        bits = np.where(
            flip[test],
            np.logaddexp(0.0, -u),
            np.logaddexp(0.0, u),
        ) / math.log(2.0)
        total += np.bincount(
            true_class[test], weights=bits, minlength=CLASSES
        )
        tables[f"beta_train_fold_{1 - test_fold}"] = beta
        diagnostics.append(
            {
                "test_fold": test_fold,
                "train_sites": int(train.sum()),
                "test_sites": int(test.sum()),
                "iterations": iterations,
                "last_max_step": last_step,
                "active_train_cells": int(
                    np.count_nonzero(np.bincount(cells[train], minlength=n_cells))
                ),
                "beta_min": float(beta.min()),
                "beta_max": float(beta.max()),
            }
        )
    tables["train_counts_fold_0"] = np.bincount(
        cells[folds == 0], minlength=n_cells
    ).astype(np.int64)
    tables["train_counts_fold_1"] = np.bincount(
        cells[folds == 1], minlength=n_cells
    ).astype(np.int64)
    return total, tables, diagnostics


def _alignment(field: np.memmap, prediction: np.memmap) -> dict[str, Any]:
    intersection = np.zeros(CLASSES, dtype=np.int64)
    union = np.zeros(CLASSES, dtype=np.int64)
    correct = 0
    sites = 0
    for pair in range(2, N):
        current = np.asarray(field[pair])
        predicted = np.asarray(prediction[pair])
        correct += int(np.count_nonzero(current == predicted))
        sites += current.size
        for class_id in range(CLASSES):
            current_class = current == class_id
            predicted_class = predicted == class_id
            intersection[class_id] += int(np.count_nonzero(current_class & predicted_class))
            union[class_id] += int(np.count_nonzero(current_class | predicted_class))
    iou = np.divide(
        intersection,
        union,
        out=np.zeros(CLASSES, dtype=np.float64),
        where=union > 0,
    )
    return {
        "eligible_pairs": N - 2,
        "sites": sites,
        "label_agreement": correct / sites,
        "correct_sites": correct,
        "per_class_intersection": intersection.tolist(),
        "per_class_union": union.tolist(),
        "per_class_iou": iou.tolist(),
        "macro_iou": float(iou.mean()),
    }


def stage_analyze() -> dict[str, Any]:
    materialized = stage_materialize()
    field = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, H, W))
    argmax = np.memmap(ARGMAX, dtype=np.uint8, mode="r", shape=(N, H, W))
    pmax = np.memmap(PMAX, dtype="<f4", mode="r", shape=(N, H, W))
    colocated_path = Path(materialized["colocated_previous"]["path"])
    colocated = np.memmap(colocated_path, dtype=np.uint8, mode="r", shape=(N, H, W))

    pmax_flat = np.asarray(pmax).reshape(-1)
    live = pmax_flat < 1.0
    q = 1.0 - pmax_flat[live].astype(np.float64)
    if not np.all((q > 0.0) & (q < 1.0)):
        raise Mc1Error("live pmax values escaped the MI1 probability domain")
    logit = np.log(q) - np.log1p(-q)
    field_flat = np.asarray(field).reshape(-1)
    true_class = field_flat[live].astype(np.uint8, copy=True)
    predicted_class = np.asarray(argmax).reshape(-1)[live]
    flip = true_class != predicted_class

    rng = np.random.default_rng(PAIR_FOLD_SEED)
    pair_fold = np.zeros(N, dtype=np.uint8)
    pair_fold[rng.permutation(N)[N // 2 :]] = 1
    folds = np.repeat(pair_fold, PLANE)[live]
    colocated_cells = np.asarray(colocated).reshape(-1)[live].astype(np.int16)
    baseline_table_path = STORE / "retained/model/colocated_offsets.npz"
    baseline_receipt_path = STORE / "stage_02_colocated_screen_complete.json"
    if baseline_receipt_path.is_file():
        baseline_receipt = json.loads(baseline_receipt_path.read_text())
        if baseline_receipt.get("schema") != "ddm_mc1_baseline_screen.v1":
            raise Mc1Error("baseline screen receipt schema changed")
        if require_fact(
            baseline_receipt["prediction"], "co-located baseline prediction"
        ) != materialized["colocated_previous"]:
            raise Mc1Error("co-located baseline receipt no longer matches materialization")
        require_fact(baseline_receipt["fitted_offsets"], "baseline fitted offsets")
        baseline_bits = np.asarray(
            baseline_receipt["screen_bits_by_true_class"], dtype=np.float64
        )
        baseline_diagnostics = baseline_receipt["fit_diagnostics"]
        baseline_alignment = baseline_receipt["alignment"]
    else:
        (
            baseline_bits,
            baseline_tables,
            baseline_diagnostics,
        ) = _crossfit_bits_by_class(
            logit,
            flip,
            true_class,
            colocated_cells,
            folds,
            CLASSES,
        )
        atomic_npz(baseline_table_path, **baseline_tables)
        baseline_alignment = _alignment(field, colocated)
        baseline_receipt = {
            "schema": "ddm_mc1_baseline_screen.v1",
            "prediction": materialized["colocated_previous"],
            "screen_bits_by_true_class": baseline_bits.tolist(),
            "screen_bits_total": float(baseline_bits.sum()),
            "alignment": baseline_alignment,
            "fitted_offsets": file_fact(baseline_table_path),
            "fit_diagnostics": baseline_diagnostics,
        }
        atomic_json(baseline_receipt_path, baseline_receipt)

    rows = []
    for model in MODELS:
        model_path = STORE / f"retained/fields/{model}.u8"
        materialized_row = next(
            row for row in materialized["models"] if row["model"] == model
        )
        model_receipt_path = STORE / f"stage_02_{model}_screen_complete.json"
        if model_receipt_path.is_file():
            model_receipt = json.loads(model_receipt_path.read_text())
            if (
                model_receipt.get("schema") != "ddm_mc1_candidate_screen.v1"
                or model_receipt.get("model") != model
            ):
                raise Mc1Error(f"{model} screen receipt identity changed")
            if model_receipt.get("baseline_fitted_offsets") != file_fact(
                baseline_table_path
            ):
                raise Mc1Error(f"{model} screen baseline dependency drifted")
            row = model_receipt["row"]
            if require_fact(row["field"], f"{model} screen field") != materialized_row[
                "field"
            ]:
                raise Mc1Error(f"{model} screen field dependency drifted")
            if require_fact(
                row["motion_parameters"], f"{model} screen motion parameters"
            ) != materialized_row["motion_parameters"]:
                raise Mc1Error(f"{model} screen motion dependency drifted")
            require_fact(row["fitted_offsets"], f"{model} fitted offsets")
            rows.append(row)
            continue
        mc = np.memmap(model_path, dtype=np.uint8, mode="r", shape=(N, H, W))
        mc_cells = np.asarray(mc).reshape(-1)[live].astype(np.int16)
        joint_cells = colocated_cells * CLASSES + mc_cells
        model_bits, model_tables, model_diagnostics = _crossfit_bits_by_class(
            logit,
            flip,
            true_class,
            joint_cells,
            folds,
            CLASSES * CLASSES,
        )
        table_path = STORE / f"retained/model/{model}_offsets.npz"
        atomic_npz(table_path, **model_tables)
        gain = baseline_bits - model_bits
        row = {
            "model": model,
            "field": file_fact(model_path),
            "motion_parameters": materialized_row["motion_parameters"],
            "alignment": _alignment(field, mc),
            "baseline_colocated_alignment": baseline_alignment,
            "screen": {
                "baseline_bits_by_true_class": baseline_bits.tolist(),
                "model_bits_by_true_class": model_bits.tolist(),
                "gain_bits_by_true_class": gain.tolist(),
                "ideal_refusal_ceiling_bytes_by_true_class": (gain / 8.0).tolist(),
                "baseline_bits_total": float(baseline_bits.sum()),
                "model_bits_total": float(model_bits.sum()),
                "gain_bits_total": float(gain.sum()),
                "ideal_refusal_ceiling_bytes": float(gain.sum() / 8.0),
                "fraction_of_42016_byte_demand": float(
                    (gain.sum() / 8.0) / DEMAND_BYTES
                ),
                "physical_byte_claim": False,
            },
            "fitted_offsets": file_fact(table_path),
            "fit_diagnostics": model_diagnostics,
        }
        rows.append(row)
        atomic_json(
            model_receipt_path,
            {
                "schema": "ddm_mc1_candidate_screen.v1",
                "model": model,
                "baseline_fitted_offsets": file_fact(baseline_table_path),
                "row": row,
            },
        )
        del mc, mc_cells, joint_cells
    best = max(rows, key=lambda row: row["screen"]["ideal_refusal_ceiling_bytes"])
    ceiling_bytes = float(best["screen"]["ideal_refusal_ceiling_bytes"])
    typed_decision = "CEILING-PASSED__RETRAIN-OWED" if ceiling_bytes >= CEILING_FIRE_BYTES else "CEILING-REFUSED"
    result = {
        "schema": "ddm_mc1_motion_compensated_previous_plane_screen.v1",
        "axis": AXIS,
        "score_claim": False,
        "physical_byte_claim": False,
        "selection": {
            "pairs": N,
            "motion_eligible_pairs": N - 2,
            "pair_fold_seed": PAIR_FOLD_SEED,
            "pair_level_two_fold": True,
            "prefix": False,
            "live_sites": int(live.sum()),
            "excluded_pmax_one_sites": int(live.size - live.sum()),
            "wrong_sites": int(flip.sum()),
        },
        "family": {
            "name": "MI1/DDS1 cross-fitted log-odds offset",
            "base_probability": "q=1-position_coding_pmax",
            "baseline_context": "co-located previous decoded class (5 cells)",
            "candidate_context": "co-located previous class x MC previous class (25 cells)",
            "note": "SCREEN bits are categorical conditional codelength estimates; /8 is refusal-only and never a physical byte claim",
        },
        "baseline": {
            "alignment": baseline_alignment,
            "screen_bits_by_true_class": baseline_bits.tolist(),
            "screen_bits_total": float(baseline_bits.sum()),
            "fitted_offsets": file_fact(baseline_table_path),
            "fit_diagnostics": baseline_diagnostics,
        },
        "rows": rows,
        "best_model": best["model"],
        "best_ideal_refusal_ceiling_bytes": ceiling_bytes,
        "ceiling_fire_bytes": CEILING_FIRE_BYTES,
        "typed_decision": typed_decision,
        "decision_scope": (
            "FORMULATION: decoder-derived constant-velocity global, HPAC-row-band, and "
            "integer-affine previous-field inputs under the MI1/DDS1 screen family on AFR1 n600"
        ),
        "next_stage": (
            "warm-start 60-epoch integer HPAC retrain with a zero-initialized second past branch"
            if typed_decision != "CEILING-REFUSED"
            else "none; charter requires refusal before training"
        ),
        "authority_boundaries": {
            "training": "NOT RUN",
            "real_rc64": "NOT RUN",
            "receiver_copy": "NOT BUILT",
            "archive": "NOT BUILT",
            "decode_timing": "NOT MEASURED",
            "scorer": "NOT RUN",
            "modal": "NOT RUN",
            "metal": "NOT RUN",
        },
    }
    atomic_json(STORE / "RESULT.json", result)
    atomic_json(
        STORE / "stage_02_analyze_complete.json",
        {
            "schema": "ddm_mc1_analyze_complete.v1",
            "result": file_fact(STORE / "RESULT.json"),
            "typed_decision": typed_decision,
            "best_model": best["model"],
            "best_ideal_refusal_ceiling_bytes": ceiling_bytes,
        },
    )
    del field, argmax, pmax, colocated
    return result


def stage_repeat() -> dict[str, Any]:
    result = stage_analyze()
    best = str(result["best_model"])
    primary = next(row for row in result["rows"] if row["model"] == best)
    repeat = materialize_model(best, repeat=True)
    field_equal = primary["field"]["sha256"] == repeat["field"]["sha256"]
    parameter_equal = (
        primary["motion_parameters"]["sha256"]
        == repeat["motion_parameters"]["sha256"]
    )
    if not field_equal or not parameter_equal:
        raise Mc1Error("selected motion estimator determinism repeat differed")
    payload = {
        "schema": "ddm_mc1_determinism_repeat.v1",
        "model": best,
        "primary_field": primary["field"],
        "repeat_field": repeat["field"],
        "primary_motion_parameters": primary["motion_parameters"],
        "repeat_motion_parameters": repeat["motion_parameters"],
        "field_byte_identical": field_equal,
        "motion_parameters_byte_identical": parameter_equal,
    }
    atomic_json(STORE / "DETERMINISM.json", payload)
    return payload


def stage_manifest() -> dict[str, Any]:
    manifest_path = STORE / "MANIFEST.json"
    rows = [
        file_fact(path)
        for path in sorted(STORE.rglob("*"))
        if path.is_file()
        and path != manifest_path
        and not path.name.startswith("._")
        and ".partial" not in path.name
    ]
    payload = {
        "schema": "ddm_mc1_manifest.v1",
        "root": str(STORE),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "exclusion": "MANIFEST.json is self-referential and omitted",
    }
    atomic_json(manifest_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("preflight", "materialize", "analyze", "repeat", "manifest", "all"),
    )
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        payload = stage_preflight()
    elif args.stage == "materialize":
        payload = stage_materialize()
    elif args.stage == "analyze":
        payload = stage_analyze()
    elif args.stage == "repeat":
        payload = stage_repeat()
    elif args.stage == "manifest":
        payload = stage_manifest()
    else:
        stage_preflight()
        stage_materialize()
        payload = stage_analyze()
        stage_repeat()
        stage_manifest()
    print(json.dumps(payload, indent=2, sort_keys=True)[:12_000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
