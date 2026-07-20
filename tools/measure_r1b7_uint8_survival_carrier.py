#!/usr/bin/env python3
"""Autopsy R1b6 through uint8, exact resize, SegNet stem, and rank-4 head.

The tool consumes the already sealed R1b6 n16 archives.  It proves or falsifies
the requested R2b-fixed-magnitude distinction by reconstructing the replay,
then traces every exact-feasible site through the actual CPU Torch scorer.  A
bounded integer-lattice counterarm is attempted only after the fixed arm is
shown insufficient.  Nothing in this module is contest-score authority.

Authority: [macOS-CPU advisory], score_claim=false, pointer unmoved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
for candidate in (REPO, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tac.boundary_math.r1b4_section_receiver import (  # noqa: E402
    ReplayWrite,
    decode_r1b4_archive,
    encode_replay_payload,
    parse_r1b4_archive,
    sha256_file,
)
from tac.canonical_equations.day_consolidation_laws_20260720 import (  # noqa: E402
    breakeven_bytes,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    BlockSolveStatus,
    DisjointResizeOperator,
    solve_bounded_integer_block,
)
from tools.measure_r1b6_admissible_carrier import (  # noqa: E402
    BATCH_SIZE,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PAIR_COUNT,
    PLANE_COUNT,
    RGB_CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    SEED,
    _atomic_json,
    _custody,
    _decode_sealed_arm,
    _hard_measure,
    _load_fisher_rows,
    _load_model,
    _raw_memmap,
    _replay_for_rows,
)

POINTER: Final = "0.1910828242 [contest-CPU] UNMOVED"
R1B6_RECEIPT_SHA256: Final = "7bdcffeb838478de3f75dd5b3ad572383175d42b368b006a26e5351175fc1684"
BASELINE_ARCHIVE_SHA256: Final = "e81da354fb7a7d38dd0d58590d143e31ccd2b7d49f6c2d3c811ceaba02afe9b7"
FIXED_ARCHIVE_SHA256: Final = "193c4cb5dee2200b8a66a1f3c911c29e0261c345f82c6eda207d9acb5034a3c6"
TARGET_SHA256: Final = "a7192f9387856c849d406a322a08ff77080502751ac200cc63fe80a704989dd5"
FISHER_SHA256: Final = "765457d424eaf1de7e05ed8703853175ef415bd3f19fb00137a74a29de52ae00"

DEFAULT_R1B6_RECEIPT: Final = REPO / (".omx/research/r1b6_admissible_carrier_prefix_n16_20260720.json")
DEFAULT_SEALED_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b6_admissible_carrier_20260720/prefix_n16_512_v2"
)
DEFAULT_BASE_ARCHIVE: Final = DEFAULT_SEALED_ROOT / "baseline_sealed.zip"
DEFAULT_FIXED_ARCHIVE: Final = DEFAULT_SEALED_ROOT / "candidate_sealed.zip"
DEFAULT_DECODER: Final = Path(
    "/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py"
)
DEFAULT_TARGET: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/m2_live_target_selection_20260720T1528Z/inflated/0.raw"
)
DEFAULT_FISHER: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/fisher_ev/fisher_ev_ordering_38077.jsonl.br"
)
DEFAULT_UPSTREAM: Final = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
INTEGER_MULTIPLIERS: Final = (1, 2, 4, 8, 16, 32, 64, 128)
# The exact operator is rational authority. CPU Torch computes the same
# align_corners=False geometry in float32; on the sealed 498-site population,
# its worst exact-R delta error is 0.003428141276041685. One 1/256 output-unit
# envelope is the smallest power-of-two sub-byte bound above that measurement.
TORCH_RESIZE_PARITY_TOLERANCE: Final = 1.0 / 256.0
STAGE_BUCKETS: Final = (
    "killed_at_uint8",
    "killed_at_resize_dilution",
    "killed_at_stem",
    "killed_at_head_same_rival",
    "killed_at_head_wrong_rival",
    "survived_but_collateral",
    "survived_clean",
)


class R1B7MeasurementError(RuntimeError):
    """Custody, stage-instrumentation, or hard-admission failure."""


@dataclass(frozen=True)
class IntegerProposal:
    """One exact camera-lattice proposal for one scheduled scorer site."""

    site_offset: int
    multiplier: int
    block: np.ndarray
    projected_rgb_delta: tuple[float, float, float]
    camera_l1: int
    changed_camera_bytes: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verify_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise R1B7MeasurementError(f"{label} SHA-256 drifted: {actual} != {expected}")


def _exact_block_projection(
    operator: DisjointResizeOperator,
    frame: np.ndarray,
    row: int,
    col: int,
) -> tuple[np.ndarray, int]:
    row_support = operator.row_supports[row]
    col_support = operator.col_supports[col]
    coefficients = np.outer(row_support.numerators, col_support.numerators).astype(np.int64)
    denominator = int(row_support.denominator) * int(col_support.denominator)
    block = np.asarray(frame)[np.ix_(row_support.indices, col_support.indices, range(RGB_CHANNELS))].astype(np.int64)
    numerator = np.sum(block * coefficients[:, :, None], axis=(0, 1), dtype=np.int64)
    return numerator, denominator


def _site_writes(
    *,
    pair: int,
    operator: DisjointResizeOperator,
    row: int,
    col: int,
    block: np.ndarray,
    reference: np.ndarray | None = None,
) -> list[ReplayWrite]:
    row_support = operator.row_supports[row]
    col_support = operator.col_supports[col]
    writes: list[ReplayWrite] = []
    for local_y, camera_y in enumerate(row_support.indices):
        for local_x, camera_x in enumerate(col_support.indices):
            for channel in range(RGB_CHANNELS):
                value = int(block[local_y, local_x, channel])
                if reference is not None and value == int(reference[local_y, local_x, channel]):
                    continue
                writes.append(
                    ReplayWrite(
                        pair,
                        1,
                        int(camera_y),
                        int(camera_x),
                        channel,
                        value,
                    )
                )
    return writes


def _classify_stage(
    *,
    camera_changed: bool,
    resize_max_abs: float,
    stem_max_abs: float,
    survived: bool,
    rival_changed: bool,
    collateral_count: int,
) -> str:
    """Return exactly one ordered first-death bucket."""

    if not camera_changed:
        return "killed_at_uint8"
    if resize_max_abs <= 0.0:
        return "killed_at_resize_dilution"
    if stem_max_abs <= 0.0:
        return "killed_at_stem"
    if not survived:
        return "killed_at_head_wrong_rival" if rival_changed else "killed_at_head_same_rival"
    if collateral_count:
        return "survived_but_collateral"
    return "survived_clean"


def _validate_histogram(histogram: dict[str, int], expected: int) -> None:
    if set(histogram) != set(STAGE_BUCKETS):
        raise R1B7MeasurementError("stage histogram keys are not canonical")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in histogram.values()):
        raise R1B7MeasurementError("stage histogram counts must be nonnegative integers")
    if sum(histogram.values()) != expected:
        raise R1B7MeasurementError(f"stage histogram total {sum(histogram.values())} != {expected}")


def _non_target_rival(logits: np.ndarray, target_class: int) -> int:
    values = np.asarray(logits, dtype=np.float64).copy()
    values[target_class] = -np.inf
    return int(np.argmax(values))


def _is_new_hard_crossing(
    *,
    baseline_predicted_class: int,
    proposal_predicted_class: int,
    target_class: int,
    baseline_margin: float,
    proposal_margin: float,
    margin_gate: float,
) -> bool:
    """Require a proposal to change a previously wrong hard decision."""

    return (
        baseline_predicted_class != target_class
        and proposal_predicted_class == target_class
        and baseline_margin <= margin_gate
        and proposal_margin > margin_gate
    )


def _decode_existing_twice(
    *,
    label: str,
    archive: Path,
    decoder: Path,
    root: Path,
    workers: int,
) -> dict[str, Any]:
    outputs: list[Path] = []
    receipts: list[dict[str, Any]] = []
    for pass_index in (1, 2):
        output = root / f"{label}_decode_{pass_index}.raw"
        receipt_path = root / f"{label}_decode_{pass_index}.json"
        receipt = decode_r1b4_archive(
            archive=archive,
            base_decoder=decoder,
            scratch_root=root / f"{label}_scratch_{pass_index}",
            output_raw=output,
            receipt_path=receipt_path,
            workers=workers,
        )
        outputs.append(output)
        receipts.append(receipt)
    hashes = [sha256_file(path) for path in outputs]
    if hashes[0] != hashes[1]:
        raise R1B7MeasurementError(f"{label} sealed double decode drifted")
    return {
        "raw": outputs[0],
        "duplicate_raw": outputs[1],
        "decoded_sha256": hashes[0],
        "receipts": receipts,
    }


def _head_patch_margin(
    *,
    torch: Any,
    feature: Any,
    head: Any,
    batch_index: int,
    row: int,
    col: int,
    target_class: int,
    rival_class: int,
) -> float:
    import torch.nn.functional as torch_functional

    padded = torch_functional.pad(feature[batch_index : batch_index + 1], (1, 1, 1, 1))
    patch = padded[0, :, row : row + 3, col : col + 3]
    weight_delta = head.weight[target_class] - head.weight[rival_class]
    bias_delta = head.bias[target_class] - head.bias[rival_class]
    return float((patch * weight_delta).sum().item() + bias_delta.item())


def _stage_autopsy(
    *,
    baseline_raw: Path,
    fixed_raw: Path,
    target_raw: Path,
    rows: list[list[Any]],
    upstream: Path,
    pair_cap: int,
) -> dict[str, Any]:
    torch, model = _load_model(upstream)
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )
    baseline = _raw_memmap(baseline_raw, pairs=pair_cap)
    fixed = _raw_memmap(fixed_raw, pairs=pair_cap)
    target = _raw_memmap(target_raw, pairs=PAIR_COUNT)
    by_pair: dict[int, list[tuple[int, list[Any]]]] = {}
    for offset, row in enumerate(rows):
        by_pair.setdefault(int(row[0]), []).append((offset, row))

    captured: dict[str, Any] = {}

    def stem_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["stem"] = output.detach()

    def head_pre_hook(_module: Any, inputs: Any) -> None:
        captured["head_input"] = inputs[0].detach()

    stem_handle = model.segnet.encoder.model.conv_stem.register_forward_hook(stem_hook)
    head = model.segnet.segmentation_head[0]
    head_handle = head.register_forward_pre_hook(head_pre_hook)
    head_matrix = head.weight.detach().cpu().numpy().reshape(5, -1).astype(np.float64)
    centered = head_matrix - head_matrix.mean(axis=0, keepdims=True)
    head_singular_values = np.linalg.svd(centered, compute_uv=False)
    head_rank = int(np.linalg.matrix_rank(centered))
    records: list[dict[str, Any] | None] = [None] * len(rows)
    exact_torch_max_error = 0.0
    head_reconstruction_max_error = 0.0
    pose_squared_error: dict[str, list[np.ndarray]] = {
        "baseline": [],
        "fixed_magnitude": [],
    }
    global_flips = {"baseline": 0, "fixed_magnitude": 0}
    try:
        for start in range(0, pair_cap, BATCH_SIZE):
            stop = min(pair_cap, start + BATCH_SIZE)
            if stop - start != BATCH_SIZE:
                raise R1B7MeasurementError("hard oracle requires exact batch 16")

            arm_outputs: dict[str, dict[str, Any]] = {}
            for name, source in (
                ("target", target),
                ("baseline", baseline),
                ("fixed_magnitude", fixed),
            ):
                batch = torch.from_numpy(np.array(source[start:stop], copy=True))
                with torch.inference_mode():
                    pose, seg_input = (
                        model.posenet.preprocess_input(batch.permute(0, 1, 4, 2, 3).float()),
                        model.segnet.preprocess_input(batch.permute(0, 1, 4, 2, 3).float()),
                    )
                    pose_output = model.posenet(pose)
                    logits = model.segnet(seg_input)
                arm_outputs[name] = {
                    "seg_input": seg_input.detach(),
                    "pose": pose_output["pose"][:, :6].detach().cpu().numpy().astype(np.float64),
                    "logits": logits.detach(),
                    "stem": captured["stem"].clone(),
                    "head_input": captured["head_input"].clone(),
                }

            target_labels = arm_outputs["target"]["logits"].argmax(dim=1)
            for name in ("baseline", "fixed_magnitude"):
                prediction = arm_outputs[name]["logits"].argmax(dim=1)
                global_flips[name] += int(torch.count_nonzero(prediction != target_labels).item())
                pose_squared_error[name].append((arm_outputs["target"]["pose"] - arm_outputs[name]["pose"]) ** 2)

            base_prediction = arm_outputs["baseline"]["logits"].argmax(dim=1)
            fixed_prediction = arm_outputs["fixed_magnitude"]["logits"].argmax(dim=1)
            for pair in range(start, stop):
                for site_offset, fisher_row in by_pair.get(pair, ()):
                    _, row, col, linear_index, target_class, realized_class = map(int, fisher_row[:6])
                    batch_index = pair - start
                    base_frame = np.asarray(baseline[pair, 1])
                    fixed_frame = np.asarray(fixed[pair, 1])
                    base_num, denominator = _exact_block_projection(operator, base_frame, row, col)
                    fixed_num, fixed_denominator = _exact_block_projection(operator, fixed_frame, row, col)
                    if denominator != fixed_denominator:
                        raise R1B7MeasurementError("resize denominator drifted")
                    projected_delta = (fixed_num - base_num).astype(np.float64) / denominator
                    torch_base = (
                        arm_outputs["baseline"]["seg_input"][batch_index, :, row, col].cpu().numpy().astype(np.float64)
                    )
                    torch_fixed = (
                        arm_outputs["fixed_magnitude"]["seg_input"][batch_index, :, row, col]
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    exact_torch_error = float(np.max(np.abs((torch_fixed - torch_base) - projected_delta)))
                    exact_torch_max_error = max(exact_torch_max_error, exact_torch_error)

                    row_support = operator.row_supports[row]
                    col_support = operator.col_supports[col]
                    base_block = base_frame[np.ix_(row_support.indices, col_support.indices, range(RGB_CHANNELS))]
                    fixed_block = fixed_frame[np.ix_(row_support.indices, col_support.indices, range(RGB_CHANNELS))]
                    camera_changed = not np.array_equal(base_block, fixed_block)
                    stem_row = row // 2
                    stem_col = col // 2
                    base_stem = arm_outputs["baseline"]["stem"]
                    fixed_stem = arm_outputs["fixed_magnitude"]["stem"]
                    stem_delta = (
                        fixed_stem[
                            batch_index,
                            :,
                            max(0, stem_row - 1) : stem_row + 2,
                            max(0, stem_col - 1) : stem_col + 2,
                        ]
                        - base_stem[
                            batch_index,
                            :,
                            max(0, stem_row - 1) : stem_row + 2,
                            max(0, stem_col - 1) : stem_col + 2,
                        ]
                    )
                    stem_max_abs = float(stem_delta.abs().max().item())

                    base_logits = arm_outputs["baseline"]["logits"][batch_index, :, row, col].cpu().numpy()
                    fixed_logits = arm_outputs["fixed_magnitude"]["logits"][batch_index, :, row, col].cpu().numpy()
                    base_rival = _non_target_rival(base_logits, target_class)
                    fixed_rival = _non_target_rival(fixed_logits, target_class)
                    base_margin = float(base_logits[target_class] - base_logits[base_rival])
                    fixed_margin = float(fixed_logits[target_class] - fixed_logits[fixed_rival])
                    reconstructed = _head_patch_margin(
                        torch=torch,
                        feature=arm_outputs["fixed_magnitude"]["head_input"],
                        head=head,
                        batch_index=batch_index,
                        row=row,
                        col=col,
                        target_class=target_class,
                        rival_class=fixed_rival,
                    )
                    head_error = abs(reconstructed - fixed_margin)
                    head_reconstruction_max_error = max(head_reconstruction_max_error, head_error)
                    r0, r1 = max(0, row - 1), min(SCORER_HEIGHT, row + 2)
                    c0, c1 = max(0, col - 1), min(SCORER_WIDTH, col + 2)
                    target_patch = target_labels[batch_index, r0:r1, c0:c1]
                    base_patch = base_prediction[batch_index, r0:r1, c0:c1]
                    fixed_patch = fixed_prediction[batch_index, r0:r1, c0:c1]
                    new_collateral = int(
                        torch.count_nonzero((fixed_patch != target_patch) & (base_patch == target_patch)).item()
                    )
                    survived = int(fixed_prediction[batch_index, row, col]) == target_class
                    bucket = _classify_stage(
                        camera_changed=camera_changed,
                        resize_max_abs=float(np.max(np.abs(projected_delta))),
                        stem_max_abs=stem_max_abs,
                        survived=survived,
                        rival_changed=fixed_rival != base_rival,
                        collateral_count=new_collateral,
                    )
                    records[site_offset] = {
                        "site_offset": site_offset,
                        "linear_index": linear_index,
                        "pair": pair,
                        "row": row,
                        "col": col,
                        "target_class": target_class,
                        "sealed_fisher_realized_class": realized_class,
                        "baseline_rival_class": base_rival,
                        "fixed_rival_class": fixed_rival,
                        "rival_identity_changed": fixed_rival != base_rival,
                        "camera_changed_bytes": int(np.count_nonzero(base_block != fixed_block)),
                        "exact_resize_delta_rgb": projected_delta.tolist(),
                        "exact_resize_max_abs": float(np.max(np.abs(projected_delta))),
                        "torch_resize_delta_rgb": (torch_fixed - torch_base).tolist(),
                        "exact_vs_torch_resize_max_abs_error": exact_torch_error,
                        "stem_local_max_abs_delta": stem_max_abs,
                        "baseline_target_vs_rival_margin": base_margin,
                        "fixed_target_vs_rival_margin": fixed_margin,
                        "fixed_margin_delta": fixed_margin - base_margin,
                        "rank4_head_reconstruction_abs_error": head_error,
                        "scheduled_target_survived": survived,
                        "new_local_collateral_pixels": new_collateral,
                        "first_death_or_survival_stage": bucket,
                    }
    finally:
        stem_handle.remove()
        head_handle.remove()

    if any(record is None for record in records):
        raise R1B7MeasurementError("not every scheduled site received a stage record")
    concrete_records = [record for record in records if record is not None]
    counter = Counter(record["first_death_or_survival_stage"] for record in concrete_records)
    histogram = {name: int(counter.get(name, 0)) for name in STAGE_BUCKETS}
    _validate_histogram(histogram, len(rows))
    if exact_torch_max_error > TORCH_RESIZE_PARITY_TOLERANCE:
        raise R1B7MeasurementError(
            f"exact resize/Torch bilinear drift {exact_torch_max_error} exceeded {TORCH_RESIZE_PARITY_TOLERANCE}"
        )
    if head_reconstruction_max_error > 1e-4:
        raise R1B7MeasurementError(f"head reconstruction drift {head_reconstruction_max_error} exceeded 1e-4")

    pixels = pair_cap * SCORER_HEIGHT * SCORER_WIDTH
    hard: dict[str, Any] = {}
    for name in ("baseline", "fixed_magnitude"):
        d_seg = global_flips[name] / pixels
        d_pose = float(np.concatenate(pose_squared_error[name]).mean())
        hard[name] = {
            "flip_count": global_flips[name],
            "d_seg": d_seg,
            "d_pose": d_pose,
            "seg_component": 100.0 * d_seg,
            "pose_component": math.sqrt(10.0 * d_pose),
            "nonrate_score": 100.0 * d_seg + math.sqrt(10.0 * d_pose),
        }
    return {
        "stage_order": [
            "emitted_camera_uint8",
            "exact_four_tap_resize",
            "segnet_conv_stem",
            "rank4_segmentation_head_margin",
            "local_collateral_overlay",
        ],
        "histogram": histogram,
        "histogram_total": sum(histogram.values()),
        "wrong_rival_count": sum(bool(record["rival_identity_changed"]) for record in concrete_records),
        "new_local_collateral_pixels_total": sum(
            int(record["new_local_collateral_pixels"]) for record in concrete_records
        ),
        "scheduled_survival_count": sum(bool(record["scheduled_target_survived"]) for record in concrete_records),
        "exact_vs_torch_resize_max_abs_error": exact_torch_max_error,
        "exact_vs_torch_resize_bound": TORCH_RESIZE_PARITY_TOLERANCE,
        "rank4_head": {
            "centered_weight_matrix_shape": list(centered.shape),
            "measured_rank": head_rank,
            "singular_values": head_singular_values.tolist(),
            "reconstruction_max_abs_error": head_reconstruction_max_error,
        },
        "hard_oracle": hard,
        "site_records": concrete_records,
    }


def _integer_block_proposals(
    *,
    operator: DisjointResizeOperator,
    baseline_frame: np.ndarray,
    fisher_row: list[Any],
    site_offset: int,
    multipliers: tuple[int, ...] = INTEGER_MULTIPLIERS,
) -> list[IntegerProposal]:
    _, row, col = map(int, fisher_row[:3])
    q = np.asarray(fisher_row[14], dtype=np.float64)
    flip_distance = float(fisher_row[11])
    lipschitz = float(fisher_row[13])
    if q.shape != (RGB_CHANNELS,) or not np.isfinite(q).all() or lipschitz <= 0.0:
        raise R1B7MeasurementError("malformed Fisher integer proposal row")
    row_support = operator.row_supports[row]
    col_support = operator.col_supports[col]
    coefficients = np.outer(row_support.numerators, col_support.numerators).astype(np.int64).reshape(-1)
    denominator = int(row_support.denominator) * int(col_support.denominator)
    lattice_gcd = reduce(gcd, (int(value) for value in coefficients))
    baseline_block = np.asarray(baseline_frame)[
        np.ix_(row_support.indices, col_support.indices, range(RGB_CHANNELS))
    ].astype(np.uint8)
    baseline_numerators = np.asarray(
        [
            int(
                np.dot(
                    coefficients,
                    baseline_block[:, :, channel].reshape(-1).astype(np.int64),
                )
            )
            for channel in range(RGB_CHANNELS)
        ],
        dtype=np.int64,
    )
    base_step = max(flip_distance / lipschitz, lattice_gcd / denominator)
    proposals: list[IntegerProposal] = []
    seen_blocks: set[bytes] = set()
    for multiplier in multipliers:
        block = np.empty_like(baseline_block)
        target_numerators = np.empty(RGB_CHANNELS, dtype=np.int64)
        feasible = True
        for channel in range(RGB_CHANNELS):
            raw_units = multiplier * base_step * q[channel] * denominator / lattice_gcd
            units = int(np.rint(raw_units))
            if units == 0 and q[channel] != 0.0:
                units = 1 if q[channel] > 0.0 else -1
            target_integer = int(baseline_numerators[channel] + units * lattice_gcd)
            if not 0 <= target_integer <= 255 * denominator:
                feasible = False
                break
            solved = solve_bounded_integer_block(
                coefficients.tolist(),
                denominator,
                target_integer / denominator,
                target_integer=target_integer,
                preferred=baseline_block[:, :, channel].reshape(-1).astype(np.float64),
                max_nodes=16_384,
            )
            if solved.status != BlockSolveStatus.FEASIBLE_EXACT:
                feasible = False
                break
            block[:, :, channel] = np.asarray(solved.values, dtype=np.uint8).reshape(
                len(row_support.indices), len(col_support.indices)
            )
            target_numerators[channel] = target_integer
        encoded = block.tobytes()
        if not feasible or encoded in seen_blocks or np.array_equal(block, baseline_block):
            continue
        seen_blocks.add(encoded)
        proposals.append(
            IntegerProposal(
                site_offset=site_offset,
                multiplier=multiplier,
                block=block,
                projected_rgb_delta=tuple(
                    float(value) for value in (target_numerators - baseline_numerators) / denominator
                ),
                camera_l1=int(np.abs(block.astype(np.int16) - baseline_block.astype(np.int16)).sum()),
                changed_camera_bytes=int(np.count_nonzero(block != baseline_block)),
            )
        )
    return proposals


def _bounded_integer_search(
    *,
    baseline_raw: Path,
    rows: list[list[Any]],
    upstream: Path,
    pair_cap: int,
    site_cap: int,
    margin_gate: float,
) -> tuple[bytes, dict[str, Any]]:
    torch, model = _load_model(upstream)
    baseline = _raw_memmap(baseline_raw, pairs=pair_cap)
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )
    selected_rows = rows[:site_cap]
    proposals: list[IntegerProposal] = []
    for site_offset, fisher_row in enumerate(selected_rows):
        pair = int(fisher_row[0])
        proposals.extend(
            _integer_block_proposals(
                operator=operator,
                baseline_frame=np.asarray(baseline[pair, 1]),
                fisher_row=fisher_row,
                site_offset=site_offset,
            )
        )
    baseline_evaluations: dict[int, dict[str, Any]] = {}
    for start in range(0, len(selected_rows), BATCH_SIZE):
        chunk = selected_rows[start : start + BATCH_SIZE]
        frames = [np.array(baseline[int(fisher_row[0])], copy=True) for fisher_row in chunk]
        while len(frames) < BATCH_SIZE:
            frames.append(frames[-1].copy())
        batch = torch.from_numpy(np.stack(frames, axis=0))
        with torch.inference_mode():
            seg_input = model.segnet.preprocess_input(batch.permute(0, 1, 4, 2, 3).float())
            logits = model.segnet(seg_input)
        for batch_index, fisher_row in enumerate(chunk):
            _, row, col, _, target_class = map(int, fisher_row[:5])
            site_logits = logits[batch_index, :, row, col].cpu().numpy()
            rival = _non_target_rival(site_logits, target_class)
            baseline_evaluations[start + batch_index] = {
                "predicted_class": int(np.argmax(site_logits)),
                "rival_class": rival,
                "target_vs_rival_margin": float(site_logits[target_class] - site_logits[rival]),
            }
    evaluations: list[dict[str, Any]] = []
    for start in range(0, len(proposals), BATCH_SIZE):
        chunk = proposals[start : start + BATCH_SIZE]
        if not chunk:
            continue
        frames: list[np.ndarray] = []
        for proposal in chunk:
            fisher_row = selected_rows[proposal.site_offset]
            pair, row, col = map(int, fisher_row[:3])
            frame_pair = np.array(baseline[pair], copy=True)
            row_support = operator.row_supports[row]
            col_support = operator.col_supports[col]
            frame_pair[1][np.ix_(row_support.indices, col_support.indices, range(RGB_CHANNELS))] = proposal.block
            frames.append(frame_pair)
        while len(frames) < BATCH_SIZE:
            frames.append(frames[-1].copy())
        batch = torch.from_numpy(np.stack(frames, axis=0))
        with torch.inference_mode():
            seg_input = model.segnet.preprocess_input(batch.permute(0, 1, 4, 2, 3).float())
            logits = model.segnet(seg_input)
        for batch_index, proposal in enumerate(chunk):
            fisher_row = selected_rows[proposal.site_offset]
            _, row, col, linear_index, target_class = map(int, fisher_row[:5])
            site_logits = logits[batch_index, :, row, col].cpu().numpy()
            rival = _non_target_rival(site_logits, target_class)
            margin = float(site_logits[target_class] - site_logits[rival])
            proposal_predicted_class = int(np.argmax(site_logits))
            baseline_evaluation = baseline_evaluations[proposal.site_offset]
            new_crossing = _is_new_hard_crossing(
                baseline_predicted_class=int(baseline_evaluation["predicted_class"]),
                proposal_predicted_class=proposal_predicted_class,
                target_class=target_class,
                baseline_margin=float(baseline_evaluation["target_vs_rival_margin"]),
                proposal_margin=margin,
                margin_gate=margin_gate,
            )
            evaluations.append(
                {
                    "site_offset": proposal.site_offset,
                    "linear_index": linear_index,
                    "multiplier": proposal.multiplier,
                    "projected_rgb_delta": list(proposal.projected_rgb_delta),
                    "camera_l1": proposal.camera_l1,
                    "changed_camera_bytes": proposal.changed_camera_bytes,
                    "target_class": target_class,
                    "baseline_predicted_class": baseline_evaluation["predicted_class"],
                    "baseline_rival_class": baseline_evaluation["rival_class"],
                    "baseline_target_vs_rival_margin": baseline_evaluation["target_vs_rival_margin"],
                    "proposal_predicted_class": proposal_predicted_class,
                    "rival_class": rival,
                    "target_vs_rival_margin": margin,
                    "hard_crossed_with_margin": new_crossing,
                }
            )

    accepted: dict[int, tuple[IntegerProposal, dict[str, Any]]] = {}
    by_key = {(proposal.site_offset, proposal.multiplier): proposal for proposal in proposals}
    for evaluation in evaluations:
        if not evaluation["hard_crossed_with_margin"]:
            continue
        proposal = by_key[(evaluation["site_offset"], evaluation["multiplier"])]
        old = accepted.get(proposal.site_offset)
        key = (proposal.camera_l1, proposal.changed_camera_bytes, proposal.multiplier)
        if old is None or key < (
            old[0].camera_l1,
            old[0].changed_camera_bytes,
            old[0].multiplier,
        ):
            accepted[proposal.site_offset] = proposal, evaluation

    writes: list[ReplayWrite] = []
    accepted_rows: list[dict[str, Any]] = []
    for site_offset in sorted(accepted):
        proposal, evaluation = accepted[site_offset]
        fisher_row = selected_rows[site_offset]
        pair, row, col = map(int, fisher_row[:3])
        row_support = operator.row_supports[row]
        col_support = operator.col_supports[col]
        reference = np.asarray(baseline[pair, 1])[np.ix_(row_support.indices, col_support.indices, range(RGB_CHANNELS))]
        writes.extend(
            _site_writes(
                pair=pair,
                operator=operator,
                row=row,
                col=col,
                block=proposal.block,
                reference=reference,
            )
        )
        accepted_rows.append(evaluation)
    writes.sort()
    payload = encode_replay_payload(writes)
    if encode_replay_payload(tuple(writes)) != payload:
        raise R1B7MeasurementError("integer replay canonical round-trip failed")
    return payload, {
        "status": ("MEASURED_HARD_CROSSING_FOUND" if accepted_rows else "MEASURED_BOUNDED_PREFIX_NO_HARD_CROSSING"),
        "site_cap": site_cap,
        "multiplier_schedule": list(INTEGER_MULTIPLIERS),
        "proposal_count": len(proposals),
        "hard_evaluation_count": len(evaluations),
        "accepted_site_count": len(accepted_rows),
        "margin_gate_strictly_greater_than": margin_gate,
        "accepted_rows": accepted_rows,
        "replay_write_count": len(writes),
        "replay_payload_bytes": len(payload),
        "replay_sha256": _sha256_bytes(payload),
        "smallest_uint8_lattice_perturbation_policy": (
            "min_camera_l1_then_changed_bytes_then_multiplier_among_exact_R_"
            "proposals_with_hard_target_margin_above_gate"
        ),
        "infeasibility_claim": False,
    }


def _cleanup_rows(*, paths: list[Path], preserve_raw: bool, rebuild_commands: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rebuild_command": rebuild_commands[path.name],
                "reason": (
                    "success-only decoded raw scratch reproducible from the retained "
                    "sealed archive and recorded decoder"
                ),
                "delete_after_receipt_fsync": not preserve_raw,
            }
        )
    return rows


def execute(args: argparse.Namespace) -> int:
    started = time.monotonic()
    if args.pair_cap != 16 or args.batch_size != BATCH_SIZE or args.max_sites != 512:
        raise R1B7MeasurementError("sealed R1b7 authority requires pair-cap=16, max-sites=512, batch-size=16")
    if not 1 <= args.integer_site_cap <= 32:
        raise R1B7MeasurementError("integer-site-cap must stay in [1,32]")
    root = args.artifact_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise R1B7MeasurementError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    raw_bytes = args.pair_cap * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS
    free_bytes = shutil.disk_usage(root).free
    required_free = 7 * raw_bytes + 2 * 1024**3
    if free_bytes < required_free:
        raise R1B7MeasurementError(f"storage preflight refused: free={free_bytes} required={required_free}")

    receipt_path = args.r1b6_receipt.expanduser().resolve(strict=True)
    baseline_archive = args.baseline_archive.expanduser().resolve(strict=True)
    fixed_archive = args.fixed_archive.expanduser().resolve(strict=True)
    decoder = args.base_decoder.expanduser().resolve(strict=True)
    target = args.target_raw.expanduser().resolve(strict=True)
    fisher = args.fisher_ordering.expanduser().resolve(strict=True)
    upstream = args.upstream.expanduser().resolve(strict=True)
    _verify_sha(receipt_path, R1B6_RECEIPT_SHA256, "R1b6 receipt")
    _verify_sha(baseline_archive, BASELINE_ARCHIVE_SHA256, "R1b6 baseline archive")
    _verify_sha(fixed_archive, FIXED_ARCHIVE_SHA256, "R1b6 candidate archive")
    _verify_sha(target, TARGET_SHA256, "target raw")
    _verify_sha(fisher, FISHER_SHA256, "Fisher ordering")
    prior_receipt = json.loads(receipt_path.read_text())
    if prior_receipt.get("runtime", {}).get("pair_cap") != 16:
        raise R1B7MeasurementError("R1b6 receipt is not the sealed n16 run")

    parsed_baseline = parse_r1b4_archive(baseline_archive)
    parsed_fixed = parse_r1b4_archive(fixed_archive)
    if parsed_baseline.manifest["final_output_assertion"]["pair_cap"] != 16:
        raise R1B7MeasurementError("baseline final output assertion pair cap drifted")
    if parsed_fixed.manifest["final_output_assertion"]["pair_cap"] != 16:
        raise R1B7MeasurementError("fixed final output assertion pair cap drifted")
    rows = _load_fisher_rows(fisher, pair_cap=16, max_sites=512)

    baseline_decode = _decode_existing_twice(
        label="baseline",
        archive=baseline_archive,
        decoder=decoder,
        root=root,
        workers=args.decode_workers,
    )
    fixed_decode = _decode_existing_twice(
        label="fixed_magnitude",
        archive=fixed_archive,
        decoder=decoder,
        root=root,
        workers=args.decode_workers,
    )
    for parsed, decoded, label in (
        (parsed_baseline, baseline_decode, "baseline"),
        (parsed_fixed, fixed_decode, "fixed_magnitude"),
    ):
        expected = parsed.manifest["final_output_assertion"]["decoded_sha256"]
        if decoded["decoded_sha256"] != expected:
            raise R1B7MeasurementError(f"{label} decoded output assertion mismatch")

    reconstructed_replay, reconstruction = _replay_for_rows(
        baseline_raw=baseline_decode["raw"],
        target_raw=target,
        rows=rows,
        pair_cap=16,
    )
    fixed_equivalent = reconstructed_replay == parsed_fixed.replay_payload
    if not fixed_equivalent:
        raise R1B7MeasurementError(
            "requested R2b fixed-magnitude replay is distinct from sealed R1b6; "
            "a separate sealed arm is required before continuing"
        )
    reconstruction.update(
        {
            "reconstructed_replay_sha256": _sha256_bytes(reconstructed_replay),
            "sealed_replay_sha256": _sha256_bytes(parsed_fixed.replay_payload),
            "byte_identical_to_sealed_r1b6": fixed_equivalent,
            "formulation_result": ("MEASURED_EQUIVALENT_R2B_FIXED_MAGNITUDE_CONSTRUCTOR_AND_R1B6_SEALED_REPLAY"),
        }
    )

    # `_replay_for_rows` may reject exact-gamut endpoints.  The sealed replay
    # itself is the authority for the selected population; select by writes.
    feasible_rows: list[list[Any]] = []
    replay_coordinates = {(write.pair_index, write.y, write.x) for write in parsed_fixed.replay_writes}
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )
    for fisher_row in rows:
        pair, row, col = map(int, fisher_row[:3])
        rs = operator.row_supports[row]
        cs = operator.col_supports[col]
        if any((pair, int(y), int(x)) in replay_coordinates for y in rs.indices for x in cs.indices):
            feasible_rows.append(fisher_row)
    if len(feasible_rows) != reconstruction["selected_site_count"]:
        raise R1B7MeasurementError("sealed replay feasible-row reconstruction drifted")
    autopsy = _stage_autopsy(
        baseline_raw=baseline_decode["raw"],
        fixed_raw=fixed_decode["raw"],
        target_raw=target,
        rows=feasible_rows,
        upstream=upstream,
        pair_cap=16,
    )

    fixed_hard = autopsy["hard_oracle"]
    fixed_recovery = fixed_hard["baseline"]["nonrate_score"] - fixed_hard["fixed_magnitude"]["nonrate_score"]
    fixed_positive_with_margin = fixed_recovery > args.promotion_margin

    integer_payload = encode_replay_payload(())
    integer_search: dict[str, Any] = {
        "status": "NOT_RUN_FIXED_ARM_POSITIVE_WITH_MARGIN",
        "accepted_site_count": 0,
    }
    integer_arm: dict[str, Any] | None = None
    integer_hard: dict[str, Any] | None = None
    if not fixed_positive_with_margin:
        integer_payload, integer_search = _bounded_integer_search(
            baseline_raw=baseline_decode["raw"],
            rows=feasible_rows,
            upstream=upstream,
            pair_cap=16,
            site_cap=args.integer_site_cap,
            margin_gate=args.integer_margin_gate,
        )
        if integer_search["accepted_site_count"]:
            integer_arm = _decode_sealed_arm(
                label="integer_aware",
                base_archive=args.base_archive.expanduser().resolve(strict=True),
                base_decoder=decoder,
                boundary_payload=parsed_baseline.boundary_payload,
                replay_payload=integer_payload,
                xi0_payload=parsed_baseline.xi0_payload,
                source_hashes=parsed_baseline.manifest["source_manifest_hashes"],
                root=root,
                pair_cap=16,
                workers=args.decode_workers,
            )
            integer_hard = _hard_measure(
                target_raw=target,
                rows={
                    "baseline": baseline_decode["raw"],
                    "integer_aware": integer_arm["final_raw"],
                },
                pair_cap=16,
                upstream=upstream,
            )

    fixed_archive_delta = parsed_fixed.archive_bytes - parsed_baseline.archive_bytes
    fixed_bytes_per_site = fixed_archive_delta / len(feasible_rows)
    fixed_break_even = breakeven_bytes(max(0.0, fixed_recovery))
    integer_realized = None
    integer_archive_delta = None
    integer_break_even = None
    if integer_arm is not None and integer_hard is not None:
        integer_realized = integer_hard["baseline"]["nonrate_score"] - integer_hard["integer_aware"]["nonrate_score"]
        integer_archive_delta = integer_arm["sealed_archive"]["bytes"] - parsed_baseline.archive_bytes
        integer_break_even = breakeven_bytes(max(0.0, integer_realized))

    cleanup_paths = [
        baseline_decode["raw"],
        baseline_decode["duplicate_raw"],
        fixed_decode["raw"],
        fixed_decode["duplicate_raw"],
    ]
    rebuild_commands = {
        path.name: (
            f"{sys.executable} {Path(__file__).resolve()} --artifact-root <empty-ssd-dir> "
            f"--output <new-receipt> --preserve-raw"
        )
        for path in cleanup_paths
    }
    if integer_arm is not None:
        for key in ("discovery_raw", "final_raw"):
            path = integer_arm[key]
            cleanup_paths.append(path)
            rebuild_commands[path.name] = (
                f"{sys.executable} {Path(__file__).resolve()} --artifact-root <empty-ssd-dir> "
                f"--output <new-receipt> --preserve-raw"
            )
    cleanup = _cleanup_rows(
        paths=cleanup_paths,
        preserve_raw=args.preserve_raw,
        rebuild_commands=rebuild_commands,
    )

    if integer_realized is not None and integer_realized > args.promotion_margin:
        verdict = "MEASURED_N16_INTEGER_AWARE_POSITIVE_WITH_MARGIN"
    elif integer_search["accepted_site_count"] == 0:
        verdict = "MEASURED_N16_FIXED_NONPOSITIVE_INTEGER_PREFIX_NO_NEW_CROSSING"
    else:
        verdict = "MEASURED_N16_UINT8_SURVIVAL_COUNTERARMS_NONPOSITIVE"
    receipt = {
        "schema": "r1b7_uint8_survival_carrier_measurement.v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "verdict": verdict,
        "verdict_scope": (
            "exact sealed R1b4 n16 receiver; 498 exact-feasible Fisher-ordered "
            "Road-Lane sites for fixed magnitude; bounded highest-EV integer prefix with "
            "explicit wrong-to-target crossing admission; "
            "seed1234 batch16 hard macOS CPU Torch only. Not n600, not contest CPU/CUDA, "
            "not a marginal-prefix waterfill, and not curvelet/shearlet/boundary/full-kernel "
            "family authority."
        ),
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "n600_run": False,
        },
        "runtime": {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "pair_cap": 16,
            "elapsed_seconds": time.monotonic() - started,
        },
        "storage_preflight": {
            "artifact_root": str(root),
            "free_bytes": free_bytes,
            "required_free_bytes": required_free,
            "ok": True,
        },
        "inputs": {
            "r1b6_receipt": _custody(receipt_path),
            "baseline_sealed_archive": _custody(baseline_archive),
            "fixed_magnitude_sealed_archive": _custody(fixed_archive),
            "base_archive": _custody(args.base_archive.expanduser().resolve(strict=True)),
            "base_decoder": _custody(decoder),
            "target_raw": _custody(target),
            "fisher_ordering": _custody(fisher),
            "segnet_weights": _custody(upstream / "models/segnet.safetensors"),
            "posenet_weights": _custody(upstream / "models/posenet.safetensors"),
        },
        "fixed_magnitude_equivalence": reconstruction,
        "stage_autopsy": autopsy,
        "fixed_magnitude": {
            "realized_combined_nonrate_recovery_s": fixed_recovery,
            "positive_margin_required_s": args.promotion_margin,
            "positive_with_margin": fixed_positive_with_margin,
            "archive_delta_bytes_vs_sealed_baseline": fixed_archive_delta,
            "measured_bytes_per_site": fixed_bytes_per_site,
            "break_even_bytes": fixed_break_even,
            "pays_measured_archive_delta": fixed_break_even >= fixed_archive_delta,
        },
        "integer_aware": {
            "search": integer_search,
            "sealed_archive": (integer_arm["sealed_archive"] if integer_arm is not None else None),
            "hard_oracle": integer_hard,
            "realized_combined_nonrate_recovery_s": integer_realized,
            "archive_delta_bytes_vs_sealed_baseline": integer_archive_delta,
            "break_even_bytes": integer_break_even,
            "pays_measured_archive_delta": (
                integer_break_even is not None
                and integer_archive_delta is not None
                and integer_break_even >= integer_archive_delta
            ),
        },
        "waterfill": {
            "ordering": "sealed Fisher necessity/flip-distance order",
        "fixed_measured_bytes_per_site": fixed_bytes_per_site,
        "old_approx_bytes_per_site": 45.97,
        "status": "COMPOSED_SET_GATE_ONLY_PREFIX_WATERFILL_NOT_EXECUTED",
        "admission_scope": (
            "all-or-nothing full fixed set and all-or-nothing composed integer crossing set; "
            "no marginal per-site or every-prefix claim"
        ),
        "fixed_set_admitted_site_count": (
                len(feasible_rows) if fixed_positive_with_margin and fixed_break_even >= fixed_archive_delta else 0
            ),
            "integer_set_admitted_site_count": (
                integer_search["accepted_site_count"]
                if integer_realized is not None
                and integer_realized > args.promotion_margin
                and integer_break_even is not None
                and integer_archive_delta is not None
                and integer_break_even >= integer_archive_delta
                else 0
            ),
            "stop_rule": (
                "reject each measured composed set when its realized break-even bytes do not pay its exact archive delta; "
                "leave unevaluated prefixes open"
            ),
            "remaining_blocker": (
                "measure receiver-composed marginal prefixes, including collateral and pose debt, before claiming a "
                "site-level waterfill"
            ),
        },
        "equation": {
            "equation_id": "realization_breakeven_bytes_v1",
            "formula": "B=max(0,Delta_S_realized)*37545489/25",
            "fixed_break_even_bytes": fixed_break_even,
            "integer_break_even_bytes": integer_break_even,
            "domain_refinement_required": True,
        },
        "cleanup": {
            "schema": "certified_rebuildable_scratch_cleanup.v1",
            "preserve_raw": args.preserve_raw,
            "rows": cleanup,
        },
    }
    _atomic_json(args.output, receipt)
    if not args.preserve_raw:
        for row in cleanup:
            Path(row["path"]).unlink()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": verdict,
                "histogram": autopsy["histogram"],
                "fixed_recovery": fixed_recovery,
                "integer_recovery": integer_realized,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--r1b6-receipt", type=Path, default=DEFAULT_R1B6_RECEIPT)
    result.add_argument("--baseline-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    result.add_argument("--fixed-archive", type=Path, default=DEFAULT_FIXED_ARCHIVE)
    result.add_argument(
        "--base-archive",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/m1_c2_glue_rebuild_20260719/pre_archive.zip"),
    )
    result.add_argument("--base-decoder", type=Path, default=DEFAULT_DECODER)
    result.add_argument("--target-raw", type=Path, default=DEFAULT_TARGET)
    result.add_argument("--fisher-ordering", type=Path, default=DEFAULT_FISHER)
    result.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--pair-cap", type=int, default=16)
    result.add_argument("--max-sites", type=int, default=512)
    result.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    result.add_argument("--decode-workers", type=int, default=4)
    result.add_argument("--integer-site-cap", type=int, default=8)
    result.add_argument("--integer-margin-gate", type=float, default=0.0)
    result.add_argument("--promotion-margin", type=float, default=0.0)
    result.add_argument("--preserve-raw", action="store_true")
    return result


def main() -> None:
    raise SystemExit(execute(parser().parse_args()))


if __name__ == "__main__":
    main()
