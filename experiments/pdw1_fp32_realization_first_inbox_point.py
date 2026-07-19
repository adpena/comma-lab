#!/usr/bin/env python
"""PDW1 fp32 realization — the first measured (bytes, d_seg) point in the 100–300 KB box.

Unit charter (2026-07-19, lane ``pdw1_realization_20260719``): the #572 rate-crush
verdict left the 100–300 KB total-archive budget box with ZERO measured
(bytes, d_seg) points; the blocker was REALIZATION, not bytes.  This script:

  Phase A — sweeps the committed quotient prefix (frames 0..194) and measures,
            under the declared frozen-fp32 receiver contract
            (``pdw1-native-f32-power-first-max.v1``), contract-vs-L* and
            contract-vs-generic-float64 disagreements, with exact tie geometry
            for every instance (closes the #543 arithmetic-authority hole on
            the whole prefix, not just one pixel).
  Phase B — reproduces the #543 frame-195 blocker pixel first-hand from the
            sealed diagnostic receipt and shows it closes under the contract.
  Phase C — builds the minimal PDW1P payload (contract labels + per-class
            fills) for n24 pairs, realizes every plane through the PROVEN
            factor-2 lattice operator to exact uint8 camera frames, and
            measures d_seg through the hard oracle (frozen CPU-Torch SegNet
            argmax on the realized frames) plus d_pose (repeat-frame1 policy)
            through the frozen PoseNet.  Decomposes the residual by class
            pair and boundary distance.

Axis: [macOS-CPU advisory] research_only=true — NO score/promotion/pointer
authority.  Reads are read-only (GT cache, quotient cache, frozen scorers,
sealed receipts); bulk outputs go to the SSD evidence tier.

Usage:
    TAC_UPSTREAM_DIR=/Users/adpena/Projects/pact/upstream PYTHONPATH=src \
        .venv/bin/python experiments/pdw1_fp32_realization_first_inbox_point.py \
        --output .omx/research/pdw1_fp32_realization_receipt_20260719.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import platform
import resource
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path("/Users/adpena/Projects/pact")
QUOTIENT_CACHE = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "v10_power_diagram_byteclose_20260718/n600_rank4_features/quotient_features.f32.npy"
)
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DIAGNOSTIC = REPO / ".omx/research/v10_power_diagram_frame195_diagnostic_20260718.json"
DIAGNOSTIC_SHA256 = "65d97194c6298a5502d0fcc792ee2fe3bf05599c69f1130d64c270dec5ec36ee"
EVIDENCE_DIR = Path("/Volumes/VertigoDataTier/pact/evidence/pdw1_realization_20260719")

PREFIX_FRAMES = 195  # committed frames 0..194 in the preserved quotient cache
N24 = 24
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
SCORE_BYTES_NORMALIZER = 37_545_489
FRONTIER_S = 0.1910828242
# Budget box constants from the #572 memo (derived from the frozen score law).
BOX_TOTAL_AT_EXACT_REALIZATION = 286_682
BOX_PER_PAIR = 477.8
BOX_DSEG_AT_236KB = 3.39e-4
SETTLED_RECEIPT = REPO / ".omx/research/pdw1_fp32_realization_receipt_20260719.json"
SETTLED_RECEIPT_SHA256 = "603b8b0eb404051bbccd64ce953d30b68b065895323129f2a9c91aa5cde9d996"
PDW2_GAUGE_RECEIPT = REPO / ".omx/research/pdw2_gauge_packet_probe_20260719_receipt.json"
PDW2_GAUGE_RECEIPT_SHA256 = "eac796b86ee5081a6d5fb97441966c0d621a60b8dae193c35dfda603df12c5ad"
_FLIP_PATCH_MAGIC = b"PDW1FX1\0"
_FLIP_PATCH_HEADER = struct.Struct("<8sHHIII")
_FLIP_PATCH_RECORD = struct.Struct("<HIB")
_RATE_SCORE_PER_BYTE = 25.0 / SCORE_BYTES_NORMALIZER


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def order0_entropy_bytes(payload: bytes) -> int:
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    nz = counts[counts > 0].astype(np.float64)
    total = float(len(payload))
    bits = float(-np.sum((nz / total) * np.log2(nz / total))) * total
    return int(np.ceil(bits / 8.0))


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels whose 4-neighbourhood contains another class."""

    mask = np.zeros(labels.shape, dtype=bool)
    mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
    mask[1:, :] |= labels[1:, :] != labels[:-1, :]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return mask


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[:-1, :] |= out[1:, :]
        grown[1:, :] |= out[:-1, :]
        grown[:, :-1] |= out[:, 1:]
        grown[:, 1:] |= out[:, :-1]
        out = grown
    return out


def _peak_rss_bytes() -> int:
    """Normalize ``ru_maxrss`` to bytes on macOS and Linux."""

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def encode_flip_patch(base_frames: list[np.ndarray], candidate_frames: list[np.ndarray]) -> bytes:
    """Count only camera-value changes in a strict deterministic sidecar."""

    if len(base_frames) != len(candidate_frames) or not base_frames:
        raise ValueError("flip patch needs equal nonempty frame lists")
    raw = bytearray()
    records = 0
    for pair, (base, candidate) in enumerate(zip(base_frames, candidate_frames, strict=True)):
        b = np.asarray(base)
        c = np.asarray(candidate)
        if b.dtype != np.uint8 or c.dtype != np.uint8 or b.shape != c.shape:
            raise ValueError("flip patch frames must be same-shape uint8")
        changed = np.flatnonzero(b.reshape(-1) != c.reshape(-1))
        flat_candidate = c.reshape(-1)
        for index in changed:
            raw += _FLIP_PATCH_RECORD.pack(pair, int(index), int(flat_candidate[index]))
        records += int(changed.size)
    compressed = brotli.compress(bytes(raw), quality=11)
    return (
        _FLIP_PATCH_HEADER.pack(
            _FLIP_PATCH_MAGIC,
            1,
            len(base_frames),
            records,
            len(raw),
            len(compressed),
        )
        + compressed
    )


def apply_flip_patch(base_frames: list[np.ndarray], payload: bytes) -> list[np.ndarray]:
    """Strict parse-back for :func:`encode_flip_patch`."""

    data = bytes(payload)
    if len(data) < _FLIP_PATCH_HEADER.size:
        raise ValueError("flip patch shorter than header")
    magic, version, n_pairs, records, raw_len, compressed_len = _FLIP_PATCH_HEADER.unpack_from(data)
    if magic != _FLIP_PATCH_MAGIC or version != 1:
        raise ValueError("bad flip patch magic/version")
    if n_pairs != len(base_frames) or n_pairs < 1:
        raise ValueError("flip patch pair count mismatch")
    if compressed_len != len(data) - _FLIP_PATCH_HEADER.size:
        raise ValueError("flip patch length/trailer mismatch")
    try:
        raw = brotli.decompress(data[_FLIP_PATCH_HEADER.size :])
    except brotli.error as exc:
        raise ValueError("flip patch Brotli stream invalid") from exc
    if raw_len != len(raw) or raw_len != records * _FLIP_PATCH_RECORD.size:
        raise ValueError("flip patch raw record length mismatch")
    out = [np.asarray(frame).copy() for frame in base_frames]
    previous = (-1, -1)
    for offset in range(0, len(raw), _FLIP_PATCH_RECORD.size):
        pair, flat_index, value = _FLIP_PATCH_RECORD.unpack_from(raw, offset)
        if pair >= n_pairs or flat_index >= out[pair].size:
            raise ValueError("flip patch record out of bounds")
        key = (int(pair), int(flat_index))
        if key <= previous:
            raise ValueError("flip patch records must be strictly canonical")
        previous = key
        out[pair].reshape(-1)[flat_index] = value
    return out


@dataclass(frozen=True)
class Step2EVRank:
    """Canonical per-flip Fisher/rank-4/resize ordering for STEP-2."""

    order: np.ndarray
    eligible: np.ndarray
    fisher_trace: np.ndarray
    head_pair_norm: np.ndarray
    feature_flip_cost: np.ndarray
    resize_camera_cost: np.ndarray
    footprint_l2: np.ndarray


def build_step2_ev_rank(ledger, composite, *, annulus_unreachable_px: float = 8.0) -> Step2EVRank:
    """Rank realization repairs without inventing an aggregate proxy.

    The frozen-head law makes ``margin / ||w_c-w_c'||`` the exact feature-space
    flip distance.  Dividing by the exact local resize-footprint norm pulls that
    distance back to camera-LSB units.  Fisher trace is retained as an explicit
    monotone diagnostic, while the rank itself is the cheapest-first ordering
    required by the uniform-delta-S flip law.  The #149 interior set is refused,
    not silently assigned a finite resize cost.
    """

    from tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704 import (
        annulus_fisher_trace,
    )
    from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
        HEAD_PAIR_NORMS,
    )

    n = int(ledger.total_flips)
    arrays = (
        ledger.deficit,
        ledger.annulus_dist,
        ledger.c_wrong,
        ledger.c_gt,
        ledger.y,
        ledger.x,
    )
    if any(np.asarray(value).shape != (n,) for value in arrays):
        raise ValueError("flip-ledger arrays do not share total_flips length")
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return Step2EVRank(
            order=np.zeros(0, dtype=np.int64),
            eligible=np.zeros(0, dtype=bool),
            fisher_trace=empty,
            head_pair_norm=empty,
            feature_flip_cost=empty,
            resize_camera_cost=empty,
            footprint_l2=empty,
        )

    pair_norm = np.empty(n, dtype=np.float64)
    for i, (wrong, gt_class) in enumerate(zip(ledger.c_wrong.tolist(), ledger.c_gt.tolist(), strict=True)):
        lo, hi = sorted((int(wrong), int(gt_class)))
        if lo == hi or not (0 <= lo < hi < len(CLASS_NAMES)):
            raise ValueError(f"invalid flip class pair {(wrong, gt_class)}")
        pair_norm[i] = float(HEAD_PAIR_NORMS[f"{CLASS_NAMES[lo]}-{CLASS_NAMES[hi]}"])

    row_l2 = np.sqrt(np.sum(np.asarray(composite.down_col) ** 2, axis=1))
    col_l2 = np.sqrt(np.sum(np.asarray(composite.down_row) ** 2, axis=1))
    footprint_l2 = row_l2[np.asarray(ledger.y, dtype=np.int64)] * col_l2[np.asarray(ledger.x, dtype=np.int64)]
    if np.any(footprint_l2 <= 0.0) or np.any(pair_norm <= 0.0):
        raise ValueError("nonpositive rank-4 normal or resize footprint norm")

    deficit = np.asarray(ledger.deficit, dtype=np.float64)
    feature_cost = deficit / pair_norm
    resize_cost = feature_cost / footprint_l2
    fisher_trace = np.asarray([annulus_fisher_trace(float(value)) for value in deficit], dtype=np.float64)
    eligible = np.asarray(ledger.annulus_dist, dtype=np.float64) <= float(annulus_unreachable_px)
    eligible_idx = np.flatnonzero(eligible)
    order = eligible_idx[np.argsort(resize_cost[eligible_idx], kind="stable")].astype(np.int64)
    return Step2EVRank(
        order=order,
        eligible=eligible,
        fisher_trace=fisher_trace,
        head_pair_norm=pair_norm,
        feature_flip_cost=feature_cost,
        resize_camera_cost=resize_cost,
        footprint_l2=footprint_l2,
    )


def build_reference_probe_frames(
    base_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    ledger,
    order: np.ndarray,
    operator,
) -> tuple[list[np.ndarray], int]:
    """Apply full source-reference blocks on ranked, disjoint resize supports."""

    if len(base_frames) != len(reference_frames):
        raise ValueError("base/reference pair count mismatch")
    out = [np.asarray(frame).copy() for frame in base_frames]
    touched: list[set[int]] = [set() for _ in out]
    overlap = 0
    for ledger_index in np.asarray(order, dtype=np.int64).tolist():
        pair = int(ledger.pair_idx[ledger_index])
        row = int(ledger.y[ledger_index])
        col = int(ledger.x[ledger_index])
        row_support = operator.row_supports[row]
        col_support = operator.col_supports[col]
        for camera_row in row_support.indices:
            for camera_col in col_support.indices:
                flat = int(camera_row) * operator.camera_w + int(camera_col)
                overlap += int(flat in touched[pair])
                touched[pair].add(flat)
        index = np.ix_(row_support.indices, col_support.indices, range(out[pair].shape[2]))
        out[pair][index] = np.asarray(reference_frames[pair])[index]
    return out, overlap


def selected_signed_margins(
    segnet,
    frames: list[np.ndarray],
    ledger,
    order: np.ndarray,
    *,
    batch: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure ``z_target-z_base_wrong`` and hard winners at selected cells."""

    from tac.witness_control.factorized_features import segnet_logits_for_frames

    take = np.asarray(order, dtype=np.int64)
    if take.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int16)
    logits = segnet_logits_for_frames(segnet, frames, batch=batch)
    pair = np.asarray(ledger.pair_idx[take], dtype=np.int64)
    row = np.asarray(ledger.y[take], dtype=np.int64)
    col = np.asarray(ledger.x[take], dtype=np.int64)
    wrong = np.asarray(ledger.c_wrong[take], dtype=np.int64)
    target = np.asarray(ledger.c_gt[take], dtype=np.int64)
    signed = logits[pair, target, row, col].astype(np.float64) - logits[pair, wrong, row, col].astype(np.float64)
    winners = logits[pair, :, row, col].argmax(axis=1).astype(np.int16)
    return signed, winners


@dataclass(frozen=True)
class SecantRepairPlan:
    order: np.ndarray
    alpha: np.ndarray
    replacement_blocks: tuple[np.ndarray, ...]
    probe_hard_accept: np.ndarray
    quantization_dead: int
    outer_support_overlap: int


def build_secant_repair_plan(
    base_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    ledger,
    ranked_order: np.ndarray,
    operator,
    probe_signed_margin: np.ndarray,
    probe_winner: np.ndarray,
    *,
    safety_fraction: float,
    outer_support_overlap: int = 0,
) -> SecantRepairPlan:
    """Secant-correct the inner scorer response on disjoint integer supports.

    The full-reference probe supplies the measured second point of the secant.
    Only cells that hard-accept at that point are retained.  Because the exact
    factor-2 operator proves scorer-cell camera supports disjoint, the outer
    overlap QP is block-diagonal; any observed overlap is a custody failure.
    """

    take = np.asarray(ranked_order, dtype=np.int64)
    m1 = np.asarray(probe_signed_margin, dtype=np.float64)
    winner1 = np.asarray(probe_winner, dtype=np.int16)
    if m1.shape != take.shape or winner1.shape != take.shape:
        raise ValueError("probe margin/winner geometry mismatch")
    if not 0.0 <= float(safety_fraction) <= 1.0:
        raise ValueError("safety_fraction must be in [0,1]")
    if int(outer_support_overlap) != 0:
        raise AssertionError("factor-2 outer supports overlap; QP no longer decomposes")

    m0 = -np.asarray(ledger.deficit[take], dtype=np.float64)
    target = np.asarray(ledger.c_gt[take], dtype=np.int16)
    improves = m1 > m0
    hard_accept = improves & (winner1 == target)
    denominator = m1 - m0
    root = np.ones_like(m0)
    root[improves] = np.clip(-m0[improves] / denominator[improves], 0.0, 1.0)
    alpha_all = np.clip(root + float(safety_fraction) * (1.0 - root), 0.0, 1.0)

    admitted_order: list[int] = []
    admitted_alpha: list[float] = []
    blocks: list[np.ndarray] = []
    quantization_dead = 0
    for position, ledger_index in enumerate(take.tolist()):
        if not bool(hard_accept[position]):
            continue
        pair = int(ledger.pair_idx[ledger_index])
        row = int(ledger.y[ledger_index])
        col = int(ledger.x[ledger_index])
        row_support = operator.row_supports[row]
        col_support = operator.col_supports[col]
        index = np.ix_(
            row_support.indices,
            col_support.indices,
            range(np.asarray(base_frames[pair]).shape[2]),
        )
        base = np.asarray(base_frames[pair])[index].astype(np.float64)
        reference = np.asarray(reference_frames[pair])[index].astype(np.float64)
        block = np.clip(np.rint(base + alpha_all[position] * (reference - base)), 0.0, 255.0).astype(np.uint8)
        if np.array_equal(block, base.astype(np.uint8)):
            quantization_dead += 1
            continue
        admitted_order.append(ledger_index)
        admitted_alpha.append(float(alpha_all[position]))
        blocks.append(block)
    return SecantRepairPlan(
        order=np.asarray(admitted_order, dtype=np.int64),
        alpha=np.asarray(admitted_alpha, dtype=np.float64),
        replacement_blocks=tuple(blocks),
        probe_hard_accept=hard_accept,
        quantization_dead=quantization_dead,
        outer_support_overlap=int(outer_support_overlap),
    )


def apply_secant_repair_prefix(
    base_frames: list[np.ndarray], ledger, operator, plan: SecantRepairPlan, top_k: int
) -> list[np.ndarray]:
    """Materialize the first ``top_k`` EV-ranked integer repair cells."""

    k = max(0, min(int(top_k), int(plan.order.size)))
    out = [np.asarray(frame).copy() for frame in base_frames]
    for ledger_index, block in zip(plan.order[:k].tolist(), plan.replacement_blocks[:k], strict=True):
        pair = int(ledger.pair_idx[ledger_index])
        row = int(ledger.y[ledger_index])
        col = int(ledger.x[ledger_index])
        row_support = operator.row_supports[row]
        col_support = operator.col_supports[col]
        index = np.ix_(
            row_support.indices,
            col_support.indices,
            range(out[pair].shape[2]),
        )
        out[pair][index] = block
    return out


def prefix_schedule(n_cells: int) -> list[int]:
    """Deterministic coarse RD acquisition grid including the full admissible set."""

    n = int(n_cells)
    if n <= 0:
        return []
    grid = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    return sorted({min(n, value) for value in grid if min(n, value) > 0} | {n})


def reference_aimed_exact_target_preimage(
    operator,
    target_plane: np.ndarray,
    reference_frame: np.ndarray,
    selected_cells: np.ndarray,
    *,
    max_nodes_per_block: int = 4096,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose exact integer blocks nearest an unrounded camera reference.

    This is deliberately an *exact-target* diagnostic.  It composes the
    existing bounded continuous projection with the bounded Diophantine block
    solver only on selected scorer cells.  The output must preserve the exact
    rational target numerator everywhere; the frozen hard oracle is evaluated
    by the caller after the whole frame is assembled.

    ``reference_frame`` is encoder-side evidence and is not silently treated as
    receiver-available.  Callers must count a self-contained frame/sidecar or
    mark the source-reference construction non-byte-closed.
    """

    from tac.optimization.uint8_lattice_feasibility import (
        BlockSolveStatus,
        solve_bounded_integer_block,
    )

    y = np.asarray(target_plane)
    reference = np.asarray(reference_frame)
    selected = np.asarray(selected_cells)
    if y.dtype != np.uint8 or y.shape != (
        operator.scorer_h,
        operator.scorer_w,
        reference.shape[2] if reference.ndim == 3 else 0,
    ):
        raise ValueError("target_plane must be scorer-HxWxC uint8")
    if reference.dtype != np.uint8 or reference.shape != (
        operator.camera_h,
        operator.camera_w,
        y.shape[2],
    ):
        raise ValueError("reference_frame must be camera-HxWxC uint8")
    if selected.dtype.kind != "b" or selected.shape != y.shape[:2]:
        raise ValueError("selected_cells must be scorer-HxW bool")

    bounded = operator.bounded_continuous_preimage(y.astype(np.float64), reference=reference.astype(np.float64))
    frame = operator.realize_factor2_uint8(y)
    status_counts = {str(status): 0 for status in BlockSolveStatus}
    nodes_visited = 0
    for out_row, out_col in zip(*np.nonzero(selected), strict=True):
        row_support = operator.row_supports[int(out_row)]
        col_support = operator.col_supports[int(out_col)]
        coefficients = tuple(
            int(value) for value in np.outer(row_support.numerators, col_support.numerators).reshape(-1)
        )
        denominator = row_support.denominator * col_support.denominator
        for channel in range(y.shape[2]):
            index = np.ix_(row_support.indices, col_support.indices, (channel,))
            solved = solve_bounded_integer_block(
                coefficients,
                denominator,
                float(y[out_row, out_col, channel]),
                target_integer=int(y[out_row, out_col, channel]) * denominator,
                preferred=bounded[index].reshape(-1),
                max_nodes=max_nodes_per_block,
            )
            status_counts[str(solved.status)] += 1
            nodes_visited += int(solved.nodes_visited)
            if solved.status is not BlockSolveStatus.FEASIBLE_EXACT:
                # The canonical all-y block is a constructive affine witness.
                # Keep it rather than laundering a search-budget exit into an
                # infeasibility claim.
                continue
            frame[index] = np.asarray(solved.values, dtype=np.uint8).reshape(
                len(row_support.indices), len(col_support.indices), 1
            )

    numerators, denominator = operator.apply_numerators(frame)
    expected = y.astype(np.int64) * denominator
    certified_exact = bool(np.array_equal(numerators, expected))
    if not certified_exact:
        raise AssertionError("reference-aimed exact-target preimage lost numerator custody")
    canonical = operator.realize_factor2_uint8(y)
    return frame, {
        "selected_scorer_cells": int(np.count_nonzero(selected)),
        "selected_channel_blocks": int(np.count_nonzero(selected) * y.shape[2]),
        "status_counts": status_counts,
        "nodes_visited": nodes_visited,
        "diophantine_infeasible_channel_blocks": int(status_counts[str(BlockSolveStatus.INFEASIBLE_EXHAUSTIVE)]),
        "solver_budget_channel_blocks": int(status_counts[str(BlockSolveStatus.NOT_FOUND_BUDGET)]),
        "changed_camera_values_vs_canonical": int(np.count_nonzero(frame != canonical)),
        "exact_rational_target_numerators": certified_exact,
        "denominator": int(denominator),
    }


def load_frozen_target():
    from tac.boundary_math.power_diagram_witness import decode_pdw1

    receipt = json.loads(DIAGNOSTIC.read_text())
    target = decode_pdw1(bytes.fromhex(receipt["frozen_target"]["pdw1_hex"]))
    return target, receipt


def phase_a_prefix_closure(target, lstars_mm, quotient_mm) -> dict[str, Any]:
    """Contract-f32 vs L* and vs generic-f64 over the committed prefix."""

    from tac.boundary_math.pdw1_fp32_receiver_contract import contract_f32_assign
    from tac.boundary_math.power_diagram_witness import power_assign

    t0 = time.time()
    contract_vs_lstar: list[dict[str, Any]] = []
    contract_vs_f64: list[dict[str, Any]] = []
    n_lstar_mismatch = 0
    n_f64_mismatch = 0
    total = 0
    for frame in range(PREFIX_FRAMES):
        z = np.ascontiguousarray(quotient_mm[frame], dtype=np.float32).reshape(-1, 4)
        lst = np.asarray(lstars_mm[frame]).reshape(-1)
        ours = contract_f32_assign(z, target)
        generic = power_assign(z.astype(np.float64), target)
        total += lst.size
        bad = np.nonzero(ours != lst)[0]
        n_lstar_mismatch += bad.size
        for flat in bad[:64]:
            contract_vs_lstar.append(
                {
                    "frame": frame,
                    "y": int(flat // 512),
                    "x": int(flat % 512),
                    "lstar": int(lst[flat]),
                    "contract": int(ours[flat]),
                }
            )
        diff = np.nonzero(ours != generic)[0]
        n_f64_mismatch += diff.size
        for flat in diff[:64]:
            contract_vs_f64.append(
                {
                    "frame": frame,
                    "y": int(flat // 512),
                    "x": int(flat % 512),
                    "contract": int(ours[flat]),
                    "generic_f64": int(generic[flat]),
                    "lstar": int(lst[flat]),
                }
            )
    return {
        "label": "MEASURED_PREFIX_CLOSURE_FRAMES_0_194",
        "frames": PREFIX_FRAMES,
        "pixels": total,
        "contract_vs_lstar_mismatches": int(n_lstar_mismatch),
        "contract_vs_generic_f64_mismatches": int(n_f64_mismatch),
        "contract_vs_lstar_instances": contract_vs_lstar,
        "contract_vs_generic_f64_instances": contract_vs_f64,
        "wall_seconds": round(time.time() - t0, 2),
    }


def phase_b_frame195(target, receipt) -> dict[str, Any]:
    from tac.boundary_math.pdw1_fp32_receiver_contract import (
        CONTRACT_ID,
        contract_f32_assign,
        contract_f32_power_scores,
    )
    from tac.boundary_math.power_diagram_witness import power_assign, power_scores

    rep = receipt["reproduction"]
    z32 = np.asarray([rep["rank4_quotient"]], dtype=np.float32)
    scores32 = contract_f32_power_scores(z32, target)[0]
    label32 = int(contract_f32_assign(z32, target)[0])
    scores64 = np.asarray(power_scores(z32.astype(np.float64), target)[0])
    label64 = int(power_assign(z32.astype(np.float64), target)[0])
    tie_exact = bool(scores32[0] == scores32[1])
    return {
        "label": "MEASURED_FRAME195_CLOSURE_UNDER_CONTRACT",
        "contract_id": CONTRACT_ID,
        "pixel": {"frame": 195, "y": rep["pixel_y"], "x": rep["pixel_x"]},
        "cached_lstar": int(rep["cached_lstar"]),
        "cpu_torch_argmax": int(rep["cpu_torch"]["argmax"]),
        "cpu_torch_winner_margin": rep["cpu_torch"]["winner_margin"],
        "contract_f32": {
            "scores": [float(s) for s in scores32],
            "argmax": label32,
            "class0_class1_exact_tie": tie_exact,
        },
        "generic_f64": {
            "scores": [float(s) for s in scores64],
            "argmax": label64,
            "winner_margin": float(np.sort(scores64)[-1] - np.sort(scores64)[-2]),
        },
        "closes": bool(label32 == int(rep["cached_lstar"]) == int(rep["cpu_torch"]["argmax"])),
        "f64_disagreement_reproduced": bool(label64 != int(rep["cached_lstar"])),
        "tie_geometry": (
            "exact fp32 tie between classes 0/1 at score "
            f"{float(scores32[0])!r}; f64 real-arithmetic margin "
            f"{float(scores64[1] - scores64[0])!r} favouring class 1; frozen "
            "CPU-Torch fp32 logit margin 4.76837158203125e-07 favouring class 0; "
            "contract first-max resolves the fp32 tie to class 0 == L*"
        ),
    }


def build_planes_and_measure(target, args) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from tac.boundary_math.pdw1_fp32_receiver_contract import contract_f32_assign
    from tac.codec.pdw1_plane_codec import (
        Pdw1PlanePayload,
        decode_pdw1p,
        encode_pdw1p,
        expand_scorer_plane,
    )
    from tac.optimization.uint8_lattice_feasibility import (
        DisjointResizeOperator,
        realize_factor2_uint8_scorer_plane,
        verify_factor2_uint8_scorer_plane,
    )
    from tac.witness_control.factorized_features import load_frozen_segnet_cpu

    gt = np.load(GT_CACHE, mmap_mode="r")
    lstars = gt["lstars"]
    quotient = np.load(QUOTIENT_CACHE, mmap_mode="r")

    # --- labels under the contract + d_A -----------------------------------
    labels = np.empty((N24, 384, 512), dtype=np.uint8)
    d_a_mismatch = 0
    for pair in range(N24):
        z = np.ascontiguousarray(quotient[pair], dtype=np.float32).reshape(-1, 4)
        lab = contract_f32_assign(z, target).reshape(384, 512)
        labels[pair] = lab.astype(np.uint8)
        d_a_mismatch += int((lab != np.asarray(lstars[pair])).sum())

    # --- frozen scorers ------------------------------------------------------
    segnet = load_frozen_segnet_cpu()

    # GT scorer planes (for fills only; encoder-side) via the REAL preprocess.
    def camera_to_plane(frame_u8: np.ndarray) -> np.ndarray:
        xp = torch.from_numpy(frame_u8[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            plane = segnet.preprocess_input(xp)
        return plane[0].permute(1, 2, 0).numpy()

    fills = np.zeros((N24, 5, 3), dtype=np.uint8)
    gt_planes_mean_accum = np.zeros((5, 3), dtype=np.float64)
    gt_planes_area_accum = np.zeros(5, dtype=np.int64)
    for pair in range(N24):
        gt_plane = camera_to_plane(np.asarray(gt["gt_f1"][pair]))
        lab = labels[pair]
        for c in range(5):
            m = lab == c
            if m.any():
                mean = gt_plane[m].mean(axis=0)
                fills[pair, c] = np.clip(np.round(mean), 0, 255).astype(np.uint8)
                gt_planes_mean_accum[c] += gt_plane[m].sum(axis=0)
                gt_planes_area_accum[c] += int(m.sum())
    global_fills = np.clip(
        np.round(gt_planes_mean_accum / np.maximum(gt_planes_area_accum, 1)[:, None]),
        0,
        255,
    ).astype(np.uint8)

    # --- payload bytes -------------------------------------------------------
    payload = Pdw1PlanePayload(labels=labels, fills=fills)
    blob = encode_pdw1p(payload)
    decoded = decode_pdw1p(blob)
    assert encode_pdw1p(decoded) == blob, "PDW1P re-encode identity failed"

    label_streams = [brotli.compress(labels[p].tobytes(), quality=11) for p in range(N24)]
    label_bytes = [len(s) for s in label_streams]
    concat = b"".join(labels[p].tobytes() for p in range(N24))
    coder_rows = {
        "per_pair_brotli_q11_total": int(sum(label_bytes)),
        "joint_brotli_q11": len(brotli.compress(concat, quality=11)),
        "joint_zlib_9": len(zlib.compress(concat, 9)),
        "joint_lzma_preset9e": len(lzma.compress(concat, preset=9 | lzma.PRESET_EXTREME)),
        "order0_ideal_entropy_per_pair_total": int(sum(order0_entropy_bytes(labels[p].tobytes()) for p in range(N24))),
    }
    boundary_px = [int(boundary_mask(labels[p]).sum()) for p in range(N24)]

    # --- realize through the PROVEN factor-2 lattice + hard oracle ----------
    op = DisjointResizeOperator.build(camera_h=874, camera_w=1164, scorer_h=384, scorer_w=512)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    per_pair: list[dict[str, Any]] = []
    confusion = np.zeros((5, 5), dtype=np.int64)
    boundary_hist = {1: 0, 2: 0, 4: 0}
    mism_total = 0
    realized_frames = np.empty((N24, 874, 1164, 3), dtype=np.uint8)
    pred_all = np.empty((N24, 384, 512), dtype=np.uint8)
    t0 = time.time()
    for pair in range(N24):
        plane = expand_scorer_plane(decoded, pair)
        frame = realize_factor2_uint8_scorer_plane(op, plane)
        ver = verify_factor2_uint8_scorer_plane(op, frame, plane)
        if not ver.certified_exact:
            raise AssertionError(f"pair {pair}: factor-2 realization not certified exact")
        realized_frames[pair] = frame
        xr = torch.from_numpy(frame[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            pred = segnet(segnet.preprocess_input(xr))[0].argmax(dim=0).numpy()
        pred_all[pair] = pred.astype(np.uint8)
        lst = np.asarray(lstars[pair])
        mism = pred != lst
        mism_total += int(mism.sum())
        np.add.at(confusion, (lst[mism].ravel(), pred[mism].ravel()), 1)
        bmask = boundary_mask(lst)
        for radius in boundary_hist:
            boundary_hist[radius] += int((mism & dilate(bmask, radius)).sum())
        per_pair.append(
            {
                "pair": pair,
                "d_seg": float(mism.mean()),
                "mismatch_px": int(mism.sum()),
                "label_stream_bytes": label_bytes[pair],
                "boundary_px": boundary_px[pair],
                "realization_certified_exact": True,
            }
        )
    seg_wall = time.time() - t0

    d_seg_mean = float(np.mean([row["d_seg"] for row in per_pair]))
    d_b_mismatch = int(sum((pred_all[p] != labels[p]).sum() for p in range(N24)))

    # --- global-fills variant (RD slope row) ---------------------------------
    variant_mism = 0
    for pair in range(N24):
        plane = global_fills[labels[pair]]
        xr = torch.from_numpy(plane.astype(np.float32)).permute(2, 0, 1)[None]
        with torch.inference_mode():
            pred = segnet(xr)[0].argmax(dim=0).numpy()
        variant_mism += int((pred != np.asarray(lstars[pair])).sum())
    variant_d_seg = variant_mism / float(N24 * 384 * 512)
    # NOTE: variant runs SegNet on the plane directly — admissible because the
    # certified realization makes resize(realized frame) == plane exactly
    # (verified above; resized-vs-plane max abs < 2e-6 fp32).

    # --- d_pose under the repeat-frame1 policy -------------------------------
    sys.path.insert(0, str(Path(args.upstream)))
    from modules import PoseNet
    from safetensors.torch import load_file

    posenet = PoseNet()
    posenet.load_state_dict(load_file(str(Path(args.upstream) / "models/posenet.safetensors")), strict=True)
    posenet.eval()
    d_pose_rows: list[float] = []
    with torch.inference_mode():
        for pair in range(N24):
            dec_pair = np.stack([realized_frames[pair], realized_frames[pair]])
            gt_pair = np.stack([np.asarray(gt["gt_f0"][pair]), np.asarray(gt["gt_f1"][pair])])
            xd = torch.from_numpy(dec_pair[None]).permute(0, 1, 4, 2, 3).contiguous().float()
            xg = torch.from_numpy(gt_pair[None]).permute(0, 1, 4, 2, 3).contiguous().float()
            out_d = posenet(posenet.preprocess_input(xd))
            out_g = posenet(posenet.preprocess_input(xg))
            d_pose_rows.append(float(posenet.compute_distortion(out_d, out_g)[0]))
    d_pose_mean = float(np.mean(d_pose_rows))

    # --- bulk custody ---------------------------------------------------------
    payload_path = EVIDENCE_DIR / "pdw1p_n24_payload.bin"
    frames_path = EVIDENCE_DIR / "realized_frames_n24_u8.npy"
    pred_path = EVIDENCE_DIR / "hard_oracle_pred_n24_u8.npy"
    payload_path.write_bytes(blob)
    np.save(frames_path, realized_frames)
    np.save(pred_path, pred_all)

    # --- totals + box comparison ----------------------------------------------
    total_bytes = len(blob)
    per_pair_bytes = total_bytes / N24
    header_bytes = 17
    n600_total = header_bytes + round((total_bytes - header_bytes) / N24 * 600)
    implied_rate = 25.0 * n600_total / SCORE_BYTES_NORMALIZER
    implied_s_measured_pose = 100.0 * d_seg_mean + float(np.sqrt(10.0 * d_pose_mean)) + implied_rate
    implied_s_seg_rate_only = 100.0 * d_seg_mean + implied_rate

    confusion_rows = []
    for a in range(5):
        for b in range(5):
            if a != b and confusion[a, b] > 0:
                confusion_rows.append(
                    {
                        "lstar": CLASS_NAMES[a],
                        "pred": CLASS_NAMES[b],
                        "px": int(confusion[a, b]),
                        "share": round(float(confusion[a, b]) / mism_total, 4),
                    }
                )
    confusion_rows.sort(key=lambda r: -r["px"])

    return {
        "label": "MEASURED_N24_FIRST_INBOX_POINT",
        "n_pairs": N24,
        "d_A_contract_labels_vs_lstar_mismatch_px": d_a_mismatch,
        "d_A": d_a_mismatch / float(N24 * 384 * 512),
        "d_B_hard_oracle_vs_stored_labels_mismatch_px": d_b_mismatch,
        "d_B": d_b_mismatch / float(N24 * 384 * 512),
        "d_seg_hard_oracle_vs_lstar": d_seg_mean,
        "d_pose_repeat_frame1_policy": d_pose_mean,
        "payload": {
            "total_bytes_n24": total_bytes,
            "sha256": sha256_bytes(blob),
            "bytes_per_pair": round(per_pair_bytes, 2),
            "header_bytes": header_bytes,
            "fills_bytes_total": 15 * N24,
            "label_stream_bytes_total": int(sum(label_bytes)),
            "coder_comparison": coder_rows,
            "boundary_px_total": int(sum(boundary_px)),
            "bits_per_boundary_px": round(8.0 * sum(label_bytes) / max(sum(boundary_px), 1), 3),
        },
        "n600_extrapolation": {
            "label": "DERIVED_N24_TO_N600_LINEAR",
            "total_bytes": n600_total,
            "bytes_per_pair": round(n600_total / 600.0, 2),
            "rate_term": round(implied_rate, 6),
            "implied_S_with_measured_repeat_frame1_pose": round(implied_s_measured_pose, 4),
            "implied_S_seg_plus_rate_only_pose_external": round(implied_s_seg_rate_only, 4),
        },
        "box_comparison": {
            "box_total_bytes_at_exact_realization_distortion": BOX_TOTAL_AT_EXACT_REALIZATION,
            "box_bytes_per_pair": BOX_PER_PAIR,
            "box_d_seg_needed_at_236kb": BOX_DSEG_AT_236KB,
            "bytes_in_box": bool(n600_total <= BOX_TOTAL_AT_EXACT_REALIZATION),
            "d_seg_in_box": bool(d_seg_mean < BOX_DSEG_AT_236KB),
            "d_seg_over_need_factor": round(d_seg_mean / BOX_DSEG_AT_236KB, 1),
        },
        "decomposition": {
            "confusion_rows_lstar_to_pred": confusion_rows,
            "mismatch_within_chebyshev_r_of_lstar_boundary": {
                str(r): {"px": v, "share": round(v / mism_total, 4)} for r, v in boundary_hist.items()
            },
            "mismatch_total_px": mism_total,
        },
        "rd_slope_rows": {
            "global_fills_variant": {
                "d_seg": variant_d_seg,
                "delta_d_seg": variant_d_seg - d_seg_mean,
                "delta_bytes_n600": -15 * 599,
                "note": "one shared 15-byte fill table instead of per-pair fills",
            },
            "labels_joint_vs_per_pair_brotli_delta_bytes_n24": (
                coder_rows["joint_brotli_q11"] - coder_rows["per_pair_brotli_q11_total"]
            ),
        },
        "per_pair": per_pair,
        "seg_wall_seconds": round(seg_wall, 1),
        "bulk_custody": {
            "payload": {"path": str(payload_path), "bytes": total_bytes, "sha256": sha256_bytes(blob)},
            "realized_frames": {"path": str(frames_path), "sha256": sha256_file(frames_path)},
            "hard_oracle_pred": {"path": str(pred_path), "sha256": sha256_file(pred_path)},
        },
    }


def build_step2_preimage_attack(target, args, *, pair_count: int | None = None) -> dict[str, Any]:
    """Measure the EV-ranked, secant-corrected sparse realization frontier."""

    import torch

    torch.set_num_threads(1)
    from tac.boundary_math.pdw1_fp32_receiver_contract import contract_f32_assign
    from tac.codec.pdw1_plane_codec import (
        Pdw1PlanePayload,
        decode_pdw1p,
        encode_pdw1p,
        expand_scorer_plane,
    )
    from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
    from tac.through_r.flip_inverse import (
        ResizeComposite,
        build_flip_ledger,
        candidate_forward_fields,
        solve_flip_costs,
    )
    from tac.witness_control.factorized_features import load_frozen_segnet_cpu

    n_pairs = int(args.step2_pairs if pair_count is None else pair_count)
    if not 1 <= n_pairs <= 600:
        raise ValueError("STEP-2 pair count must be in [1,600]")
    gt = np.load(GT_CACHE, mmap_mode="r")
    lstars = np.asarray(gt["lstars"][:n_pairs])
    quotient = np.load(QUOTIENT_CACHE, mmap_mode="r")
    if n_pairs > min(len(gt["lstars"]), len(quotient)):
        raise ValueError("STEP-2 pair count exceeds frozen cache coverage")
    segnet = load_frozen_segnet_cpu()
    verdict_batch = max(1, int(args.verdict_batch))

    def camera_to_plane(frame_u8: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(frame_u8[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
        with torch.inference_mode():
            plane = segnet.preprocess_input(tensor)
        return plane[0].permute(1, 2, 0).numpy()

    labels = np.empty((n_pairs, 384, 512), dtype=np.uint8)
    fills = np.empty((n_pairs, 5, 3), dtype=np.uint8)
    reference_frames: list[np.ndarray] = []
    d_a_mismatch = 0
    for pair in range(n_pairs):
        quotient_row = np.ascontiguousarray(quotient[pair], dtype=np.float32).reshape(-1, 4)
        label = contract_f32_assign(quotient_row, target).reshape(384, 512)
        labels[pair] = label.astype(np.uint8)
        d_a_mismatch += int(np.count_nonzero(label != lstars[pair]))
        reference = np.asarray(gt["gt_f1"][pair]).copy()
        reference_frames.append(reference)
        unrounded_plane = camera_to_plane(reference)
        for class_id in range(5):
            class_mask = label == class_id
            if not class_mask.any():
                fills[pair, class_id] = 0
            else:
                fills[pair, class_id] = np.clip(np.rint(unrounded_plane[class_mask].mean(axis=0)), 0, 255).astype(
                    np.uint8
                )

    base_blob = encode_pdw1p(Pdw1PlanePayload(labels=labels, fills=fills))
    decoded = decode_pdw1p(base_blob)
    base_parseback_identical = bool(encode_pdw1p(decoded) == base_blob)
    operator = DisjointResizeOperator.build(camera_h=874, camera_w=1164, scorer_h=384, scorer_w=512)
    base_frames = [operator.realize_factor2_uint8(expand_scorer_plane(decoded, pair)) for pair in range(n_pairs)]
    started = time.monotonic()
    subset_reason = (
        f"STEP-2 EV-ranked acquisition n{n_pairs}; non-authority until exact n600 archive" if n_pairs != 600 else None
    )
    ledger = build_flip_ledger(
        base_frames,
        lstars=labels,
        candidate_class="PDW1P canonical integer-plane realization",
        segnet=segnet,
        verdict_batch=verdict_batch,
        allow_subset_reason=subset_reason,
    )
    composite = ResizeComposite.build()
    canonical_frontier = solve_flip_costs(ledger, composite)
    ev_rank = build_step2_ev_rank(ledger, composite)

    probe_frames, support_overlap = build_reference_probe_frames(
        base_frames, reference_frames, ledger, ev_rank.order, operator
    )
    probe_signed, probe_winner = selected_signed_margins(
        segnet,
        probe_frames,
        ledger,
        ev_rank.order,
        batch=verdict_batch,
    )
    plan = build_secant_repair_plan(
        base_frames,
        reference_frames,
        ledger,
        ev_rank.order,
        operator,
        probe_signed,
        probe_winner,
        safety_fraction=float(args.secant_safety_fraction),
        outer_support_overlap=support_overlap,
    )

    denominator = float(n_pairs * 384 * 512)
    base_d_b = float(ledger.total_flips) / denominator
    base_n600_bytes = 17 + round((len(base_blob) - 17) / n_pairs * 600)
    rd_rows: list[dict[str, Any]] = [
        {
            "ranked_cells": 0,
            "patch_records": 0,
            "patch_section_bytes_n_subset": 0,
            "patch_section_bytes_n600_projected": 0,
            "total_sections_bytes_n_subset": len(base_blob),
            "total_sections_bytes_n600_projected": base_n600_bytes,
            "hard_oracle_mismatch_px": int(ledger.total_flips),
            "d_B": base_d_b,
            "net_fixed_px": 0,
            "targeted_fixed_px": 0,
            "collateral_new_px_upper_bound": 0,
            "seg_plus_rate_no_pose": 100.0 * base_d_b + 25.0 * base_n600_bytes / SCORE_BYTES_NORMALIZER,
            "patch_sha256": None,
            "parseback_identical": True,
        }
    ]
    for top_k in prefix_schedule(int(plan.order.size)):
        candidate = apply_secant_repair_prefix(base_frames, ledger, operator, plan, top_k)
        patch_blob = encode_flip_patch(base_frames, candidate)
        parsed = apply_flip_patch(base_frames, patch_blob)
        parseback_identical = all(np.array_equal(a, b) for a, b in zip(candidate, parsed, strict=True))
        if not parseback_identical:
            raise AssertionError("STEP-2 sparse patch failed strict parse-back")
        prediction, _deficit = candidate_forward_fields(segnet, parsed, labels, verdict_batch=verdict_batch)
        mismatch = int(np.count_nonzero(prediction != labels))
        take = plan.order[:top_k]
        targeted_fixed = int(
            np.count_nonzero(
                prediction[
                    np.asarray(ledger.pair_idx[take], dtype=np.int64),
                    np.asarray(ledger.y[take], dtype=np.int64),
                    np.asarray(ledger.x[take], dtype=np.int64),
                ]
                == np.asarray(ledger.c_gt[take], dtype=np.uint8)
            )
        )
        _magic, _version, _pairs, records, _raw_len, _compressed_len = _FLIP_PATCH_HEADER.unpack_from(patch_blob)
        projected_patch = _FLIP_PATCH_HEADER.size + round((len(patch_blob) - _FLIP_PATCH_HEADER.size) / n_pairs * 600)
        projected_total = base_n600_bytes + projected_patch
        net_fixed = int(ledger.total_flips) - mismatch
        rd_rows.append(
            {
                "ranked_cells": int(top_k),
                "patch_records": int(records),
                "patch_section_bytes_n_subset": len(patch_blob),
                "patch_section_bytes_n600_projected": projected_patch,
                "total_sections_bytes_n_subset": len(base_blob) + len(patch_blob),
                "total_sections_bytes_n600_projected": projected_total,
                "hard_oracle_mismatch_px": mismatch,
                "d_B": mismatch / denominator,
                "net_fixed_px": net_fixed,
                "targeted_fixed_px": targeted_fixed,
                "collateral_new_px_upper_bound": max(0, targeted_fixed - net_fixed),
                "seg_plus_rate_no_pose": 100.0 * mismatch / denominator
                + 25.0 * projected_total / SCORE_BYTES_NORMALIZER,
                "patch_sha256": sha256_bytes(patch_blob),
                "parseback_identical": parseback_identical,
            }
        )
        previous, current = rd_rows[-2], rd_rows[-1]
        delta_bytes = int(current["patch_section_bytes_n600_projected"]) - int(
            previous["patch_section_bytes_n600_projected"]
        )
        delta_seg_score = 100.0 * (float(previous["d_B"]) - float(current["d_B"]))
        if delta_bytes > 0:
            marginal = delta_seg_score / delta_bytes
        elif delta_seg_score > 0:
            marginal = float("inf")
        else:
            marginal = float("-inf")
        current["adjacent_marginal_seg_score_per_projected_byte"] = marginal
        current["adjacent_segment_clears_rate_waterline"] = bool(marginal >= _RATE_SCORE_PER_BYTE)
        if not current["adjacent_segment_clears_rate_waterline"]:
            current["stop_after_this_row"] = True
            break

    chosen = min(
        rd_rows,
        key=lambda row: (
            float(row["seg_plus_rate_no_pose"]),
            int(row["patch_section_bytes_n600_projected"]),
            int(row["ranked_cells"]),
        ),
    )
    chosen_k = int(chosen["ranked_cells"])
    chosen_frames = apply_secant_repair_prefix(base_frames, ledger, operator, plan, chosen_k)
    chosen_patch = encode_flip_patch(base_frames, chosen_frames) if chosen_k else b""
    parsed_chosen = apply_flip_patch(base_frames, chosen_patch) if chosen_k else base_frames
    chosen_prediction, _chosen_deficit = candidate_forward_fields(
        segnet, parsed_chosen, labels, verdict_batch=verdict_batch
    )
    attack_mismatch = int(np.count_nonzero(chosen_prediction != labels))
    attack_d_b = attack_mismatch / denominator

    base_cell_lookup = {
        (int(pair), int(row), int(col)): index
        for index, (pair, row, col) in enumerate(zip(ledger.pair_idx, ledger.y, ledger.x, strict=True))
    }
    ev_position = {int(index): pos for pos, index in enumerate(ev_rank.order.tolist())}
    plan_position = {int(index): pos for pos, index in enumerate(plan.order.tolist())}
    residual_cells: list[dict[str, Any]] = []
    for pair, row, col in zip(*np.nonzero(chosen_prediction != labels), strict=True):
        key = (int(pair), int(row), int(col))
        base_index = base_cell_lookup.get(key)
        if base_index is None:
            reason = "COLLATERAL_NEW_HARD_FLIP"
        elif base_index not in ev_position:
            reason = "INTERIOR_OR_NONRESIZE_NECESSITY_ZERO_149_WALL"
        elif base_index not in plan_position:
            reason = "FULL_REFERENCE_PROBE_NOT_HARD_ACCEPT_OR_UINT8_DEAD"
        elif plan_position[base_index] >= chosen_k:
            reason = "STOPPED_AT_MEASURED_RATE_WATERLINE"
        else:
            reason = "SECANT_PREFIX_STILL_NOT_HARD_ACCEPT_AFTER_JOINT_REPLAY"
        residual_cells.append(
            {
                "pair": int(pair),
                "y": int(row),
                "x": int(col),
                "lstar": int(labels[pair, row, col]),
                "prediction": int(chosen_prediction[pair, row, col]),
                "reason": reason,
            }
        )

    per_pair = []
    for pair in range(n_pairs):
        base_pair = int(np.count_nonzero(ledger.pair_idx == pair))
        attack_pair = int(np.count_nonzero(chosen_prediction[pair] != labels[pair]))
        per_pair.append(
            {
                "pair": pair,
                "baseline_hard_oracle_mismatch_px": base_pair,
                "attack_hard_oracle_mismatch_px": attack_pair,
                "net_fixed_px": base_pair - attack_pair,
                "attacked_frame_sha256": sha256_bytes(parsed_chosen[pair].tobytes()),
            }
        )

    if sha256_file(PDW2_GAUGE_RECEIPT) != PDW2_GAUGE_RECEIPT_SHA256:
        raise AssertionError("PDW2 gauge receipt hash drift — refusing rate-base claim")
    pdw2_receipt = json.loads(PDW2_GAUGE_RECEIPT.read_text())
    pdw2_margin_bytes = int(pdw2_receipt["packets"]["margin_preserving"]["brotli_quality11_bytes"])
    pdw2_spatial_receiver = bool(pdw2_receipt["stop_rule"]["spatial_receiver_present"])
    selected_section_total = int(chosen["total_sections_bytes_n600_projected"])
    section_gate = bool(selected_section_total <= 264_320 and attack_d_b <= BOX_DSEG_AT_236KB)
    classification = (
        "EV_RANKED_REPAIR_REACHES_SUBSET_DSEG_GATE_BUT_FULL_ARCHIVE_AND_PDW2_SPATIAL_RECEIVER_CUSTODY_REMAIN_OPEN"
        if attack_d_b <= BOX_DSEG_AT_236KB
        else "NAMED_EV_RANKED_HARD_ORACLE_RESIDUAL_AFTER_RATE_WATERLINE"
    )
    return {
        "label": "MEASURED_STEP2_EV_RANKED_SECANT_INTEGER_REPAIR",
        "verdict_scope": "FORMULATION x SUBSET",
        "n_pairs": n_pairs,
        "d_A_contract_labels_vs_lstar_mismatch_px": d_a_mismatch,
        "d_A": d_a_mismatch / denominator,
        "baseline_d_B": base_d_b,
        "attack_d_B": attack_d_b,
        "delta_d_B": attack_d_b - base_d_b,
        "base_hard_oracle_flip_px": int(ledger.total_flips),
        "hard_oracle_residual_px": attack_mismatch,
        "ev_ranker": {
            "equations": [
                "realization_necessity_preimage_per_stratum_v1",
                "resize_exploit_flip_fix_frontier_v1",
                "segnet_head_rank4_linear_flipdist_v1",
                "frozen_scorer_fisher_curvature_margin_colocation_v1",
                "fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
                "flip_margin_step_law_v1",
                "witness_measured_reverse_waterfill_v1",
                "meta_lagrangian_dual_solver_per_axis_kkt_residual_v1",
            ],
            "policy": (
                "refuse annulus_dist>8 (#149); rank remaining flips by exact "
                "margin/(rank4_pair_norm*local_resize_footprint_l2), stable cheapest-first"
            ),
            "eligible_resize_edge_band_cells": int(np.count_nonzero(ev_rank.eligible)),
            "excluded_interior_cells": int(np.count_nonzero(~ev_rank.eligible)),
            "fisher_trace_min": (float(ev_rank.fisher_trace.min()) if ev_rank.fisher_trace.size else 0.0),
            "fisher_trace_max": (float(ev_rank.fisher_trace.max()) if ev_rank.fisher_trace.size else 0.0),
            "resize_camera_cost_quantiles": {
                str(q): float(np.quantile(ev_rank.resize_camera_cost, q))
                for q in (0.0, 0.1, 0.5, 0.9, 1.0)
                if ev_rank.resize_camera_cost.size
            },
            "canonical_flip_frontier": {
                "free": int(canonical_frontier.n_free),
                "cheap": int(canonical_frontier.n_cheap),
                "costed": int(canonical_frontier.n_costed),
                "unreachable": int(canonical_frontier.n_unreachable),
            },
        },
        "inner_jacobian_secant": {
            "full_reference_probe_cells": int(ev_rank.order.size),
            "full_reference_probe_hard_accept_cells": int(np.count_nonzero(plan.probe_hard_accept)),
            "admissible_integer_repair_cells": int(plan.order.size),
            "quantization_dead_cells": int(plan.quantization_dead),
            "safety_fraction": float(args.secant_safety_fraction),
            "outer_support_overlap": int(plan.outer_support_overlap),
            "outer_qp_status": (
                "EXACT_BLOCK_DIAGONAL_DISJOINT_SUPPORTS; joint hard-oracle replay "
                "retains inner-network collateral authority"
            ),
        },
        "reverse_waterfill": {
            "score_rate_waterline_per_byte": _RATE_SCORE_PER_BYTE,
            "selection_metric": "100*d_B + 25*n600_projected_section_bytes/37_545_489; pose unmeasured",
            "selected_ranked_cells": chosen_k,
            "rows": rd_rows,
            "caveat": (
                "coarse measured prefix curve; n600 patch bytes are a labeled linear "
                "projection until an exact n600 section is materialized"
            ),
        },
        "byte_custody": {
            "pdw1p_base_section_bytes_n_subset": len(base_blob),
            "pdw1p_base_sha256": sha256_bytes(base_blob),
            "pdw1p_parseback_identical": base_parseback_identical,
            "pdw1p_base_section_bytes_n600_projected": base_n600_bytes,
            "selected_patch_section_bytes_n_subset": len(chosen_patch),
            "selected_patch_sha256": sha256_bytes(chosen_patch) if chosen_patch else None,
            "selected_patch_parseback_identical": all(
                np.array_equal(a, b) for a, b in zip(chosen_frames, parsed_chosen, strict=True)
            ),
            "selected_total_sections_bytes_n600_projected": selected_section_total,
            "archive_total_not_claimed": True,
            "source_reference_needed_at_receiver": False,
            "receiver_order": "expand integer plane -> realize factor2 -> apply strict PDW1FX1 sparse patch",
        },
        "pdw2_consumability": {
            "gauge_receipt": str(PDW2_GAUGE_RECEIPT),
            "gauge_receipt_sha256": PDW2_GAUGE_RECEIPT_SHA256,
            "margin_packet_brotli_bytes": pdw2_margin_bytes,
            "repair_stage_codec_neutral_post_realization": True,
            "spatial_receiver_present": pdw2_spatial_receiver,
            "pdw2_packet_used_as_rate_base": False,
            "blocker": (
                "PDW2 receipt explicitly says spatial_receiver_present=false; counting its "
                "133-byte head packet as the spatial base would be a rate fake"
            ),
        },
        "residual": {
            "cells": residual_cells,
            "count": len(residual_cells),
            "diophantine_infeasible_cells": [],
            "diophantine_note": (
                "This target-changing repair writes valid uint8 camera blocks directly. "
                "The exact-pre-rounded-Y affine equation is not the hard-accept predicate; "
                "its separate one-pair canary found zero affine-infeasible blocks."
            ),
        },
        "carrier_basis": {
            "status": "NOT_ADMITTED",
            "diagnostic_patch_basis": "sparse camera-value records",
            "reason": (
                "the patch measures realization and rate only; any promoted residual/carrier "
                "must use the registered curvelet/shearlet basis and close Pose"
            ),
        },
        "gate": {
            "max_total_bytes": 264_320,
            "max_d_seg": BOX_DSEG_AT_236KB,
            "selected_sections_n600_projected_in_box": bool(selected_section_total <= 264_320),
            "hard_oracle_d_seg_in_box": bool(attack_d_b <= BOX_DSEG_AT_236KB),
            "subset_section_gate_pass": section_gate,
            "success": False,
            "success_blockers": [
                "no exact n600 archive total",
                "PDW2 spatial receiver absent",
                "Pose delta unmeasured",
                "contest CPU/CUDA authority absent",
            ],
        },
        "classification": classification,
        "per_pair": per_pair,
        "wall_seconds": round(time.monotonic() - started, 3),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=REPO / "upstream")
    parser.add_argument("--skip-prefix-scan", action="store_true")
    parser.add_argument("--step2-only", action="store_true")
    parser.add_argument("--step2-pairs", type=int, default=24)
    parser.add_argument("--step2-confirm-pairs", type=int, default=0)
    parser.add_argument("--max-nodes-per-block", type=int, default=4096)
    parser.add_argument("--verdict-batch", type=int, default=4)
    parser.add_argument("--secant-safety-fraction", type=float, default=0.125)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt {args.output}")
    if sha256_file(DIAGNOSTIC) != DIAGNOSTIC_SHA256:
        raise SystemExit("sealed frame-195 diagnostic receipt hash drift — refusing")

    target, receipt = load_frozen_target()
    gt = np.load(GT_CACHE, mmap_mode="r")
    quotient = np.load(QUOTIENT_CACHE, mmap_mode="r")

    from tac.boundary_math.pdw1_fp32_receiver_contract import CONTRACT_ID
    from tac.boundary_math.power_diagram_witness import encode_pdw1, encode_pdw2, pdw1_to_pdw2

    result: dict[str, Any] = {
        "schema": "pdw1_fp32_realization_first_inbox_point.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_id": CONTRACT_ID,
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": f"{FRONTIER_S} [contest-CPU Linux x86_64] UNMOVED",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch_threads": 1,
        },
        "inputs": {
            "gt_cache": {"path": str(GT_CACHE), "bytes": GT_CACHE.stat().st_size},
            "quotient_cache": {"path": str(QUOTIENT_CACHE), "bytes": QUOTIENT_CACHE.stat().st_size},
            "frame195_diagnostic": {"path": str(DIAGNOSTIC), "sha256": DIAGNOSTIC_SHA256},
        },
        "coefficient_certificate": {
            "note": (
                "encoder-side only; NOT counted in the payload because no "
                "receiver can expand coefficients into a spatial partition "
                "without a channel feature field (consumption discipline)"
            ),
            "pdw1_raw_bytes": len(encode_pdw1(target)),
            "pdw2_margin_raw_bytes": len(encode_pdw2(pdw1_to_pdw2(target))),
        },
    }
    if args.step2_only:
        if sha256_file(SETTLED_RECEIPT) != SETTLED_RECEIPT_SHA256:
            raise SystemExit("settled Phase A/B/C receipt hash drift — refusing")
        result["settled_state_not_remeasured"] = {
            "receipt": str(SETTLED_RECEIPT),
            "sha256": SETTLED_RECEIPT_SHA256,
            "phase_b_frame195": "CLOSED; not re-litigated",
            "phase_c_d_A": 0.0,
            "phase_c_d_B": 0.00806956821017795,
        }
    else:
        if not args.skip_prefix_scan:
            result["phase_a_prefix_closure"] = phase_a_prefix_closure(target, gt["lstars"], quotient)
        result["phase_b_frame195_closure"] = phase_b_frame195(target, receipt)
        result["phase_c_first_inbox_point"] = build_planes_and_measure(target, args)
    result["phase_c2_step2_preimage_attack"] = build_step2_preimage_attack(target, args)
    if args.step2_confirm_pairs:
        if not int(args.step2_pairs) < int(args.step2_confirm_pairs) <= 600:
            raise SystemExit("--step2-confirm-pairs must be > --step2-pairs and <=600")
        result["phase_c2_larger_subset_confirmation"] = build_step2_preimage_attack(
            target, args, pair_count=int(args.step2_confirm_pairs)
        )

    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"receipt -> {args.output}")
    point = result["phase_c2_step2_preimage_attack"]
    print(
        f"STEP2: base d_B={point['baseline_d_B']:.6f} · "
        f"attack d_B={point['attack_d_B']:.6f} · "
        f"residual={point['hard_oracle_residual_px']} px · "
        f"in-box={point['gate']['success']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
