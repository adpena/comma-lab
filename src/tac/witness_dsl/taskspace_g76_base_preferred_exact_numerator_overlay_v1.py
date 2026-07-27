# SPDX-License-Identifier: MIT
"""Base-preserving exact camera preimages for the live Torch scorer input.

The factor-2 camera-to-scorer operator owns four disjoint camera taps for each
scorer coordinate and is channel-separable.  A semantic actuator therefore
does not need to replace a whole camera frame, a whole RGB cell, or even all
four donor taps merely because one scorer support changed.

G76 treats the actuator-rendered camera pair as an exact scorer-input donor.
For every selected frame/scorer-coordinate/channel whose four camera taps
differ from the semantic base, it solves the independent bounded equation

    c dot x = donor_numerator,  x in {0, ..., 255}^4

with the semantic base taps as the deterministic preference.  The existing
audited GCD-pruned block solver supplies an exact feasible point.  A bounded
search miss falls back to the donor's four taps, which are feasible by
construction.

Integer numerator equality alone is not the final authority: different tap
values with the same rational numerator can round differently in the actual
float32 ``torch.nn.functional.interpolate`` used by both PoseNet and SegNet.
After the preferred solve, G76 runs that exact CPU Torch operation and falls
back to donor taps for every block that is not bit-identical to the donor at
the scorer input.  A final replay proves whole selected-frame Torch equality.
Every unowned camera value remains bit-identical to the semantic base.

This is generic receiver machinery.  It contains no video-specific factor,
target, selector table, scorer artifact, or learned payload.  The returned
receipt is proof of exact pre-scorer input transport, not a frozen-network
Pose, score, candidate, or global-nearest-preimage claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    BlockSolveStatus,
    DisjointResizeOperator,
    Uint8LatticeError,
    solve_bounded_integer_block,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

SCHEMA: Final = "tac.taskspace_g76_base_preferred_exact_numerator_overlay.v1"
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
FRAME_COUNT: Final = 2
CHANNELS: Final = 3


class G76ExactNumeratorOverlayError(ValueError):
    """The exact numerator overlay contract failed closed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json([int(item) for item in array.shape]))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise G76ExactNumeratorOverlayError(f"{label} must be one lowercase SHA-256")
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise G76ExactNumeratorOverlayError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


@lru_cache(maxsize=1)
def _operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )


def _selected_frames(
    selector: SelectedPreimageFrameSelectorV1,
) -> tuple[int, ...]:
    if type(selector) is not SelectedPreimageFrameSelectorV1:
        raise G76ExactNumeratorOverlayError("frame_selector must be an exact SelectedPreimageFrameSelectorV1")
    if selector is SelectedPreimageFrameSelectorV1.Y0:
        return (0,)
    if selector is SelectedPreimageFrameSelectorV1.Y1:
        return (1,)
    if selector is SelectedPreimageFrameSelectorV1.BOTH:
        return (0, 1)
    raise G76ExactNumeratorOverlayError("frame_selector is unsupported")


def _camera_pair(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    expected = (
        FRAME_COUNT,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        CHANNELS,
    )
    if array.dtype != np.uint8 or array.shape != expected:
        raise G76ExactNumeratorOverlayError(f"{label} must be uint8 with shape {expected}")
    return np.ascontiguousarray(array)


def _torch_bilinear_scorer_pair(camera_pair: np.ndarray) -> np.ndarray:
    """Mirror the shared PoseNet/SegNet float32 bilinear pre-resize."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - contest runtime owns torch
        raise G76ExactNumeratorOverlayError("live scorer-input parity requires torch") from exc
    source = torch.from_numpy(np.ascontiguousarray(camera_pair)).permute(
        0,
        3,
        1,
        2,
    )
    with torch.inference_mode():
        scorer = torch.nn.functional.interpolate(
            source.float(),
            size=(SCORER_HEIGHT, SCORER_WIDTH),
            mode="bilinear",
        )
    return np.ascontiguousarray(scorer.permute(0, 2, 3, 1).cpu().numpy())


def _torch_runtime_version() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - contest runtime owns torch
        raise G76ExactNumeratorOverlayError("Torch runtime identity requires torch") from exc
    version = str(torch.__version__)
    if not version or not version.isascii():
        raise G76ExactNumeratorOverlayError("Torch runtime version must be nonempty ASCII")
    return version


def _support_changed_mask(
    *,
    base: np.ndarray,
    donor: np.ndarray,
    selected_frames: tuple[int, ...],
    operator: DisjointResizeOperator,
) -> np.ndarray:
    """Return frame/scorer-coordinate/channel donor-tap ownership."""

    changed = np.zeros(
        (
            FRAME_COUNT,
            SCORER_HEIGHT,
            SCORER_WIDTH,
            CHANNELS,
        ),
        dtype=np.bool_,
    )
    row_indices = np.asarray(
        [support.indices for support in operator.row_supports],
        dtype=np.int64,
    )
    col_indices = np.asarray(
        [support.indices for support in operator.col_supports],
        dtype=np.int64,
    )
    if row_indices.shape != (SCORER_HEIGHT, 2) or col_indices.shape != (
        SCORER_WIDTH,
        2,
    ):
        raise G76ExactNumeratorOverlayError("live factor-2 operator lost its disjoint 2x2 support")
    for frame_index in selected_frames:
        for row_offset in range(2):
            for col_offset in range(2):
                changed[frame_index] |= (
                    base[
                        frame_index,
                        row_indices[:, row_offset, None],
                        col_indices[None, :, col_offset],
                        :,
                    ]
                    != donor[
                        frame_index,
                        row_indices[:, row_offset, None],
                        col_indices[None, :, col_offset],
                        :,
                    ]
                )
    return changed


@dataclass(frozen=True, slots=True)
class G76ExactNumeratorOverlayReceiptV1:
    """Typed proof summary for one pairwise Torch-exact projection."""

    frame_selector: str
    scorer_denominator: int
    owned_scorer_values: int
    owned_scorer_cells: int
    changed_numerator_values: int
    owned_camera_values: int
    actually_changed_camera_values: int
    donor_camera_values_changed_from_base: int
    base_preferred_torch_exact_blocks: int
    solver_budget_fallback_blocks: int
    torch_parity_fallback_blocks: int
    solver_nodes_visited: int
    solver_max_nodes_for_one_block: int
    camera_values_total: int
    scorer_values_total: int
    base_camera_sha256: str
    donor_camera_sha256: str
    output_camera_sha256: str
    owned_camera_mask_sha256: str
    base_numerators_sha256: str
    donor_numerators_sha256: str
    output_numerators_sha256: str
    donor_selected_torch_scorer_input_sha256: str
    output_selected_torch_scorer_input_sha256: str
    torch_version: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        try:
            selector = SelectedPreimageFrameSelectorV1(self.frame_selector)
        except ValueError as exc:
            raise G76ExactNumeratorOverlayError("receipt frame_selector is unsupported") from exc
        del selector
        _require_exact_int(
            self.scorer_denominator,
            label="scorer_denominator",
            minimum=1,
            maximum=1 << 62,
        )
        for label in (
            "owned_scorer_values",
            "owned_scorer_cells",
            "changed_numerator_values",
            "owned_camera_values",
            "actually_changed_camera_values",
            "donor_camera_values_changed_from_base",
            "base_preferred_torch_exact_blocks",
            "solver_budget_fallback_blocks",
            "torch_parity_fallback_blocks",
            "solver_nodes_visited",
            "solver_max_nodes_for_one_block",
            "camera_values_total",
            "scorer_values_total",
        ):
            _require_exact_int(
                getattr(self, label),
                label=label,
                minimum=0,
                maximum=1 << 62,
            )
        if self.owned_scorer_values < 1:
            raise G76ExactNumeratorOverlayError("receipt must describe at least one donor-changed support")
        if (
            self.base_preferred_torch_exact_blocks
            + self.solver_budget_fallback_blocks
            + self.torch_parity_fallback_blocks
            != self.owned_scorer_values
        ):
            raise G76ExactNumeratorOverlayError("receipt block disposition does not cover owned scorer values")
        if self.changed_numerator_values > self.owned_scorer_values:
            raise G76ExactNumeratorOverlayError("changed numerators exceed donor-changed supports")
        if self.owned_camera_values != 4 * self.owned_scorer_values:
            raise G76ExactNumeratorOverlayError("receipt ownership is not four channelwise taps per support")
        if self.actually_changed_camera_values > self.owned_camera_values:
            raise G76ExactNumeratorOverlayError("receipt changes exceed owned camera values")
        if self.camera_values_total != (
            FRAME_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS
        ) or self.scorer_values_total != (FRAME_COUNT * SCORER_HEIGHT * SCORER_WIDTH * CHANNELS):
            raise G76ExactNumeratorOverlayError("receipt geometry totals differ from the frozen contract")
        for label in (
            "base_camera_sha256",
            "donor_camera_sha256",
            "output_camera_sha256",
            "owned_camera_mask_sha256",
            "base_numerators_sha256",
            "donor_numerators_sha256",
            "output_numerators_sha256",
            "donor_selected_torch_scorer_input_sha256",
            "output_selected_torch_scorer_input_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if type(self.torch_version) is not str or not self.torch_version or not self.torch_version.isascii():
            raise G76ExactNumeratorOverlayError("receipt torch_version must be nonempty ASCII")
        if self.donor_selected_torch_scorer_input_sha256 != self.output_selected_torch_scorer_input_sha256:
            raise G76ExactNumeratorOverlayError("receipt selected Torch scorer-input hashes differ")
        if self.receipt_sha256 != _sha256(_canonical_json(self._body_dict())):
            raise G76ExactNumeratorOverlayError("receipt_sha256 differs from the canonical body")

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "frame_selector": self.frame_selector,
            "scorer_denominator": self.scorer_denominator,
            "owned_scorer_values": self.owned_scorer_values,
            "owned_scorer_cells": self.owned_scorer_cells,
            "changed_numerator_values": self.changed_numerator_values,
            "owned_camera_values": self.owned_camera_values,
            "actually_changed_camera_values": (self.actually_changed_camera_values),
            "donor_camera_values_changed_from_base": (self.donor_camera_values_changed_from_base),
            "base_preferred_torch_exact_blocks": (self.base_preferred_torch_exact_blocks),
            "solver_budget_fallback_blocks": (self.solver_budget_fallback_blocks),
            "torch_parity_fallback_blocks": (self.torch_parity_fallback_blocks),
            "solver_nodes_visited": self.solver_nodes_visited,
            "solver_max_nodes_for_one_block": (self.solver_max_nodes_for_one_block),
            "camera_values_total": self.camera_values_total,
            "scorer_values_total": self.scorer_values_total,
            "base_camera_sha256": self.base_camera_sha256,
            "donor_camera_sha256": self.donor_camera_sha256,
            "output_camera_sha256": self.output_camera_sha256,
            "owned_camera_mask_sha256": self.owned_camera_mask_sha256,
            "base_numerators_sha256": self.base_numerators_sha256,
            "donor_numerators_sha256": self.donor_numerators_sha256,
            "output_numerators_sha256": self.output_numerators_sha256,
            "donor_selected_torch_scorer_input_sha256": (self.donor_selected_torch_scorer_input_sha256),
            "output_selected_torch_scorer_input_sha256": (self.output_selected_torch_scorer_input_sha256),
            "torch_version": self.torch_version,
            "proof": {
                "selected_torch_scorer_input_bit_equal_donor": True,
                "unselected_frames_equal_base": True,
                "unowned_camera_values_equal_base": True,
                "ownership_is_frame_scorer_coordinate_channel_from_donor_taps": True,
                "supports_are_disjoint_two_by_two": True,
                "selected_output_integer_numerators_equal_donor": True,
            },
            "solver_policy": ("base_preferred_integer_exact_then_live_torch_bit_parity_else_exact_donor_fallback"),
            "global_norm_optimality_claim": False,
            "frozen_network_forward_claim": False,
            "cross_host_torch_parity_claim": False,
            "pose_claim": False,
            "score_claim": False,
            "candidate_claim": False,
            "research_only": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._body_dict(),
            "receipt_sha256": self.receipt_sha256,
        }

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class G76ExactNumeratorOverlayResultV1:
    """Exact projected camera pair plus channelwise ownership and receipt."""

    camera_pair: np.ndarray
    owned_camera_mask: np.ndarray
    receipt: G76ExactNumeratorOverlayReceiptV1

    def __post_init__(self) -> None:
        camera = _camera_pair(self.camera_pair, label="result camera_pair")
        owned = np.asarray(self.owned_camera_mask)
        if owned.dtype != np.bool_ or owned.shape != camera.shape:
            raise G76ExactNumeratorOverlayError("owned_camera_mask must be bool with the camera-pair shape")
        if type(self.receipt) is not G76ExactNumeratorOverlayReceiptV1:
            raise G76ExactNumeratorOverlayError("result receipt changed typed identity")
        if self.receipt.output_camera_sha256 != _sha256_array(camera):
            raise G76ExactNumeratorOverlayError("result camera hash differs from its receipt")
        if self.receipt.owned_camera_values != int(np.count_nonzero(owned)):
            raise G76ExactNumeratorOverlayError("result ownership count differs from its receipt")
        if self.receipt.owned_camera_mask_sha256 != _sha256_array(owned):
            raise G76ExactNumeratorOverlayError("result ownership mask hash differs from its receipt")
        camera = np.array(camera, copy=True, order="C")
        owned = np.array(owned, copy=True, order="C")
        camera.setflags(write=False)
        owned.setflags(write=False)
        object.__setattr__(self, "camera_pair", camera)
        object.__setattr__(self, "owned_camera_mask", owned)


def project_base_preferred_exact_numerator_overlay(
    *,
    base_camera_pair: np.ndarray,
    donor_camera_pair: np.ndarray,
    frame_selector: SelectedPreimageFrameSelectorV1,
    max_nodes_per_block: int = 4096,
) -> G76ExactNumeratorOverlayResultV1:
    """Project donor numerators into the base with channelwise exact support.

    The donor is an exact actuator render, not a target-byte plane.  Fractional
    scorer values therefore retain their native integer numerators.
    """

    base = _camera_pair(base_camera_pair, label="base_camera_pair")
    donor = _camera_pair(donor_camera_pair, label="donor_camera_pair")
    selected = _selected_frames(frame_selector)
    max_nodes = _require_exact_int(
        max_nodes_per_block,
        label="max_nodes_per_block",
        minimum=1,
        maximum=1 << 24,
    )
    operator = _operator()
    base_numerators: list[np.ndarray] = []
    donor_numerators: list[np.ndarray] = []
    denominator: int | None = None
    for frame_index in range(FRAME_COUNT):
        try:
            base_num, base_den = operator.apply_numerators(base[frame_index])
            donor_num, donor_den = operator.apply_numerators(donor[frame_index])
        except Uint8LatticeError as exc:
            raise G76ExactNumeratorOverlayError("exact camera-to-scorer numerator projection failed") from exc
        if base_den != donor_den or (denominator is not None and denominator != base_den):
            raise G76ExactNumeratorOverlayError("scorer denominator changed across frames or inputs")
        denominator = base_den
        base_numerators.append(np.ascontiguousarray(base_num))
        donor_numerators.append(np.ascontiguousarray(donor_num))
    if denominator is None:  # pragma: no cover - fixed two-frame loop
        raise G76ExactNumeratorOverlayError("camera pair produced no scorer denominator")

    base_num_pair = np.stack(base_numerators, axis=0)
    donor_num_pair = np.stack(donor_numerators, axis=0)
    owned_scorer = _support_changed_mask(
        base=base,
        donor=donor,
        selected_frames=selected,
        operator=operator,
    )
    owned_scorer_count = int(np.count_nonzero(owned_scorer))
    if owned_scorer_count < 1:
        raise G76ExactNumeratorOverlayError("selected donor changed no scorer-owned camera support")
    numerator_changed = np.zeros_like(base_num_pair, dtype=np.bool_)
    for frame_index in selected:
        numerator_changed[frame_index] = base_num_pair[frame_index] != donor_num_pair[frame_index]
    changed_numerator_count = int(np.count_nonzero(numerator_changed))

    output = np.array(base, copy=True, order="C")
    owned = np.zeros(output.shape, dtype=np.bool_)
    base_preferred = np.zeros_like(owned_scorer)
    budget_fallback = np.zeros_like(owned_scorer)
    nodes_visited = 0
    maximum_nodes_visited = 0

    for frame_index, scorer_row, scorer_col, channel in np.argwhere(owned_scorer):
        frame_id = int(frame_index)
        row_id = int(scorer_row)
        col_id = int(scorer_col)
        channel_id = int(channel)
        row_support = operator.row_supports[row_id]
        col_support = operator.col_supports[col_id]
        index = np.ix_(
            row_support.indices,
            col_support.indices,
            (channel_id,),
        )
        if bool(np.any(owned[frame_id][index])):
            raise G76ExactNumeratorOverlayError("changed channelwise scorer supports unexpectedly overlap")
        coefficients = tuple(
            int(value)
            for value in np.outer(
                row_support.numerators,
                col_support.numerators,
            ).reshape(-1)
        )
        preferred = base[frame_id][index].reshape(-1).astype(np.float64)
        target_integer = int(donor_num_pair[frame_id, row_id, col_id, channel_id])
        try:
            solved = solve_bounded_integer_block(
                coefficients,
                denominator,
                target_integer / denominator,
                target_integer=target_integer,
                preferred=preferred,
                max_nodes=max_nodes,
            )
        except Uint8LatticeError as exc:
            raise G76ExactNumeratorOverlayError("bounded exact block solve failed") from exc
        nodes_visited += solved.nodes_visited
        maximum_nodes_visited = max(
            maximum_nodes_visited,
            solved.nodes_visited,
        )
        if solved.status is BlockSolveStatus.FEASIBLE_EXACT:
            values = np.asarray(solved.values, dtype=np.uint8)
            base_preferred[frame_id, row_id, col_id, channel_id] = True
        elif solved.status is BlockSolveStatus.NOT_FOUND_BUDGET:
            values = donor[frame_id][index].reshape(-1)
            budget_fallback[
                frame_id,
                row_id,
                col_id,
                channel_id,
            ] = True
        else:
            raise G76ExactNumeratorOverlayError(
                "block solver claimed exhaustive infeasibility for a donor-proven feasible numerator"
            )
        output[frame_id][index] = values.reshape(
            len(row_support.indices),
            len(col_support.indices),
            1,
        )
        owned[frame_id][index] = True

    donor_torch = _torch_bilinear_scorer_pair(donor)
    output_torch = _torch_bilinear_scorer_pair(output)
    torch_mismatch = np.zeros_like(owned_scorer)
    for frame_index in selected:
        torch_mismatch[frame_index] = output_torch[frame_index] != donor_torch[frame_index]
    if bool(np.any(torch_mismatch & ~owned_scorer)):
        raise G76ExactNumeratorOverlayError("Torch scorer input differs outside donor-tap ownership")
    torch_fallback = np.zeros_like(owned_scorer)
    for frame_index, scorer_row, scorer_col, channel in np.argwhere(torch_mismatch):
        frame_id = int(frame_index)
        row_id = int(scorer_row)
        col_id = int(scorer_col)
        channel_id = int(channel)
        if budget_fallback[frame_id, row_id, col_id, channel_id]:
            raise G76ExactNumeratorOverlayError("donor budget fallback failed live Torch parity")
        row_support = operator.row_supports[row_id]
        col_support = operator.col_supports[col_id]
        index = np.ix_(
            row_support.indices,
            col_support.indices,
            (channel_id,),
        )
        output[frame_id][index] = donor[frame_id][index]
        base_preferred[frame_id, row_id, col_id, channel_id] = False
        torch_fallback[frame_id, row_id, col_id, channel_id] = True
    output_torch = _torch_bilinear_scorer_pair(output)
    for frame_index in selected:
        if not np.array_equal(
            output_torch[frame_index],
            donor_torch[frame_index],
        ):
            raise G76ExactNumeratorOverlayError(
                "selected output is not bit-identical to the donor at the live Torch scorer input"
            )

    output_numerators: list[np.ndarray] = []
    for frame_index in range(FRAME_COUNT):
        output_num, output_den = operator.apply_numerators(output[frame_index])
        if output_den != denominator:
            raise G76ExactNumeratorOverlayError("output scorer denominator differs")
        if frame_index in selected:
            if not np.array_equal(
                output_num,
                donor_num_pair[frame_index],
            ):
                raise G76ExactNumeratorOverlayError("selected output integer numerators do not equal the donor")
        elif not np.array_equal(output[frame_index], base[frame_index]):
            raise G76ExactNumeratorOverlayError("unselected frame differs from the base")
        output_numerators.append(np.ascontiguousarray(output_num))
    if not np.array_equal(output[~owned], base[~owned]):
        raise G76ExactNumeratorOverlayError("unowned camera values differ from the base")
    actual_changes = int(np.count_nonzero(output != base))
    if actual_changes < 1:
        raise G76ExactNumeratorOverlayError("exact overlay caused no realized camera-byte change")

    output_num_pair = np.stack(output_numerators, axis=0)
    selected_camera_mask = np.zeros(base.shape, dtype=np.bool_)
    selected_camera_mask[np.asarray(selected, dtype=np.int64)] = True
    body = {
        "schema": SCHEMA,
        "frame_selector": frame_selector.value,
        "scorer_denominator": int(denominator),
        "owned_scorer_values": owned_scorer_count,
        "owned_scorer_cells": int(np.count_nonzero(np.any(owned_scorer, axis=3))),
        "changed_numerator_values": changed_numerator_count,
        "owned_camera_values": int(np.count_nonzero(owned)),
        "actually_changed_camera_values": actual_changes,
        "donor_camera_values_changed_from_base": int(np.count_nonzero((donor != base) & selected_camera_mask)),
        "base_preferred_torch_exact_blocks": int(np.count_nonzero(base_preferred)),
        "solver_budget_fallback_blocks": int(np.count_nonzero(budget_fallback)),
        "torch_parity_fallback_blocks": int(np.count_nonzero(torch_fallback)),
        "solver_nodes_visited": nodes_visited,
        "solver_max_nodes_for_one_block": maximum_nodes_visited,
        "camera_values_total": int(base.size),
        "scorer_values_total": int(base_num_pair.size),
        "base_camera_sha256": _sha256_array(base),
        "donor_camera_sha256": _sha256_array(donor),
        "output_camera_sha256": _sha256_array(output),
        "owned_camera_mask_sha256": _sha256_array(owned),
        "base_numerators_sha256": _sha256_array(base_num_pair),
        "donor_numerators_sha256": _sha256_array(donor_num_pair),
        "output_numerators_sha256": _sha256_array(output_num_pair),
        "donor_selected_torch_scorer_input_sha256": _sha256_array(donor_torch[np.asarray(selected, dtype=np.int64)]),
        "output_selected_torch_scorer_input_sha256": _sha256_array(output_torch[np.asarray(selected, dtype=np.int64)]),
        "torch_version": _torch_runtime_version(),
        "proof": {
            "selected_torch_scorer_input_bit_equal_donor": True,
            "unselected_frames_equal_base": True,
            "unowned_camera_values_equal_base": True,
            "ownership_is_frame_scorer_coordinate_channel_from_donor_taps": True,
            "supports_are_disjoint_two_by_two": True,
            "selected_output_integer_numerators_equal_donor": True,
        },
        "solver_policy": ("base_preferred_integer_exact_then_live_torch_bit_parity_else_exact_donor_fallback"),
        "global_norm_optimality_claim": False,
        "frozen_network_forward_claim": False,
        "cross_host_torch_parity_claim": False,
        "pose_claim": False,
        "score_claim": False,
        "candidate_claim": False,
        "research_only": True,
    }
    receipt = G76ExactNumeratorOverlayReceiptV1(
        frame_selector=body["frame_selector"],
        scorer_denominator=body["scorer_denominator"],
        owned_scorer_values=body["owned_scorer_values"],
        owned_scorer_cells=body["owned_scorer_cells"],
        changed_numerator_values=body["changed_numerator_values"],
        owned_camera_values=body["owned_camera_values"],
        actually_changed_camera_values=body["actually_changed_camera_values"],
        donor_camera_values_changed_from_base=body["donor_camera_values_changed_from_base"],
        base_preferred_torch_exact_blocks=body["base_preferred_torch_exact_blocks"],
        solver_budget_fallback_blocks=body["solver_budget_fallback_blocks"],
        torch_parity_fallback_blocks=body["torch_parity_fallback_blocks"],
        solver_nodes_visited=body["solver_nodes_visited"],
        solver_max_nodes_for_one_block=body["solver_max_nodes_for_one_block"],
        camera_values_total=body["camera_values_total"],
        scorer_values_total=body["scorer_values_total"],
        base_camera_sha256=body["base_camera_sha256"],
        donor_camera_sha256=body["donor_camera_sha256"],
        output_camera_sha256=body["output_camera_sha256"],
        owned_camera_mask_sha256=body["owned_camera_mask_sha256"],
        base_numerators_sha256=body["base_numerators_sha256"],
        donor_numerators_sha256=body["donor_numerators_sha256"],
        output_numerators_sha256=body["output_numerators_sha256"],
        donor_selected_torch_scorer_input_sha256=body["donor_selected_torch_scorer_input_sha256"],
        output_selected_torch_scorer_input_sha256=body["output_selected_torch_scorer_input_sha256"],
        torch_version=body["torch_version"],
        receipt_sha256=_sha256(_canonical_json(body)),
    )
    return G76ExactNumeratorOverlayResultV1(
        camera_pair=output,
        owned_camera_mask=owned,
        receipt=receipt,
    )


def parse_g76_exact_numerator_overlay_receipt(
    payload: bytes,
) -> G76ExactNumeratorOverlayReceiptV1:
    """Strictly parse and canonically re-emit one G76 receipt."""

    if type(payload) is not bytes or not payload:
        raise G76ExactNumeratorOverlayError("receipt payload must be nonempty exact bytes")

    def unique_pairs(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G76ExactNumeratorOverlayError(f"receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_pairs,
        )
    except G76ExactNumeratorOverlayError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G76ExactNumeratorOverlayError("receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise G76ExactNumeratorOverlayError("receipt is not canonical JSON")
    expected = set(G76ExactNumeratorOverlayReceiptV1.__dataclass_fields__) | {
        "schema",
        "proof",
        "solver_policy",
        "global_norm_optimality_claim",
        "frozen_network_forward_claim",
        "cross_host_torch_parity_claim",
        "pose_claim",
        "score_claim",
        "candidate_claim",
        "research_only",
    }
    if set(value) != expected or value.get("schema") != SCHEMA:
        raise G76ExactNumeratorOverlayError("receipt fields or schema differ")
    if value.get("proof") != {
        "selected_torch_scorer_input_bit_equal_donor": True,
        "unselected_frames_equal_base": True,
        "unowned_camera_values_equal_base": True,
        "ownership_is_frame_scorer_coordinate_channel_from_donor_taps": True,
        "supports_are_disjoint_two_by_two": True,
        "selected_output_integer_numerators_equal_donor": True,
    }:
        raise G76ExactNumeratorOverlayError("receipt proof boundary differs")
    if (
        value.get("solver_policy")
        != ("base_preferred_integer_exact_then_live_torch_bit_parity_else_exact_donor_fallback")
        or value.get("global_norm_optimality_claim") is not False
        or value.get("frozen_network_forward_claim") is not False
        or value.get("cross_host_torch_parity_claim") is not False
        or value.get("pose_claim") is not False
        or value.get("score_claim") is not False
        or value.get("candidate_claim") is not False
        or value.get("research_only") is not True
    ):
        raise G76ExactNumeratorOverlayError("receipt authority boundary differs")
    constructor = {key: value[key] for key in G76ExactNumeratorOverlayReceiptV1.__dataclass_fields__}
    try:
        receipt = G76ExactNumeratorOverlayReceiptV1(**constructor)
    except TypeError as exc:
        raise G76ExactNumeratorOverlayError("receipt typed fields differ") from exc
    if receipt.to_receipt_bytes() != payload:
        raise G76ExactNumeratorOverlayError("receipt changed on typed parse-back")
    return receipt


__all__ = [
    "SCHEMA",
    "G76ExactNumeratorOverlayError",
    "G76ExactNumeratorOverlayReceiptV1",
    "G76ExactNumeratorOverlayResultV1",
    "parse_g76_exact_numerator_overlay_receipt",
    "project_base_preferred_exact_numerator_overlay",
]
