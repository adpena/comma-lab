# SPDX-License-Identifier: MIT
"""Population-score costates projected into real task-space actuator tangents.

This module deliberately never serializes a dense pixel costate.  Dense
candidate-space VJPs exist only for one scorer batch and are immediately paired
with finite, receiver-realized actuator displacements.  The durable coordinate
is a compact row per physical actuator group.

Segmentation and pose remain separate coordinates:

* pose is the VJP of the raw per-sample MSE, multiplied by the exact shared
  population chain-rule factor;
* segmentation is the target-vs-current logit-gap VJP on currently mismatched
  cells, used only for screening; and
* rate is absent until a caller supplies an actual composed-ZIP byte delta.

No weighted sum, local acceptance threshold, or candidate admission lives here.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import numpy as np

from tac.optimization.scorer_gradient_sparse_residual import (
    global_pose_score_costate_scale,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
    V15RoleAwareOverlayDecoderV1,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

if TYPE_CHECKING:
    from tac.optimization.direct_description_carrier_compose import (
        BoundaryShearletAtomV1,
    )
    from tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 import (
        G72BoundaryShearletProposalV1,
    )

PAIR_COUNT: Final = 600
MAX_BATCH_PAIRS: Final = 16
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
SOURCE_VIDEO_BYTES: Final = 37_545_489
G72_GROUP_LAW: Final = "ROLE_X_INFERRED_DIRECTION_X_AMPLITUDE_BATCH_GROUP_V1"
SEG_SCREEN_COORDINATE: Final = "SUM_TARGET_MINUS_CURRENT_LOGIT_GAP_ON_BASE_MISMATCH_CELLS"
POSE_COSTATE_COORDINATE: Final = "UPSTREAM_PAIR_MSE_TIMES_GLOBAL_POPULATION_CHAIN_RULE"
_G72_VARIANT = re.compile(r"_sh_d(?P<direction>[01])_a(?P<amplitude>0\.5|1)$")


class ProjectedPopulationCostateError(ValueError):
    """Custody, scorer geometry, actuator realization, or projection failed."""

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.context = {} if context is None else dict(context)


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _exact_camera_batch(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.uint8
        or array.ndim != 5
        or not 1 <= array.shape[0] <= MAX_BATCH_PAIRS
        or array.shape[1:] != (2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
    ):
        raise ProjectedPopulationCostateError(
            f"{label} must be exact uint8 [B,2,{CAMERA_HEIGHT},{CAMERA_WIDTH},3], B<=16"
        )
    return np.ascontiguousarray(array).copy()


def _exact_cells(value: np.ndarray, *, batch_pairs: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape != (
        batch_pairs,
        SCORER_HEIGHT,
        SCORER_WIDTH,
    ):
        raise ProjectedPopulationCostateError(f"{label} must be exact uint8 [B,{SCORER_HEIGHT},{SCORER_WIDTH}]")
    if bool(np.any(array >= 5)):
        raise ProjectedPopulationCostateError(f"{label} escaped the frozen five-class head")
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class PopulationScorePointV1:
    """Complete base operating point required by the evaluator chain rule."""

    global_mean_pose_dist: float
    sample_count: int
    archive_bytes: int
    archive_sha256: str

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.global_mean_pose_dist)) or float(self.global_mean_pose_dist) <= 0.0:
            raise ProjectedPopulationCostateError("global mean pose distortion must be finite and positive")
        if self.sample_count != PAIR_COUNT:
            raise ProjectedPopulationCostateError("population costate requires the exact n600 base row")
        if type(self.archive_bytes) is not int or self.archive_bytes <= 0:
            raise ProjectedPopulationCostateError("archive bytes must be one positive exact integer")
        if (
            type(self.archive_sha256) is not str
            or len(self.archive_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.archive_sha256)
        ):
            raise ProjectedPopulationCostateError("archive SHA-256 is not canonical")

    @property
    def pair_pose_mse_vjp_scale(self) -> float:
        return global_pose_score_costate_scale(
            global_mean_pose_dist=float(self.global_mean_pose_dist),
            sample_count=self.sample_count,
        ).pair_mse_vjp_scale


@dataclass(frozen=True, slots=True)
class BatchPopulationCostatesV1:
    """Ephemeral dense batch costates plus custody needed for exact replay."""

    pair_ids: tuple[int, ...]
    pose_costate_hwc: np.ndarray
    seg_gap_costate_hwc: np.ndarray
    base_pair_pose_mse: np.ndarray
    target_pose6: np.ndarray
    base_mismatch_count: int
    base_gap_sum: float
    score_point: PopulationScorePointV1
    candidate_sha256: str
    target_sha256: str
    target_cells_sha256: str
    described_cells_sha256: str

    def __post_init__(self) -> None:
        batch = len(self.pair_ids)
        if (
            not 1 <= batch <= MAX_BATCH_PAIRS
            or self.pair_ids != tuple(range(self.pair_ids[0], self.pair_ids[0] + batch))
            or any(not 0 <= pair_id < PAIR_COUNT for pair_id in self.pair_ids)
        ):
            raise ProjectedPopulationCostateError("pair IDs must be one contiguous exact n600 batch")
        expected = (batch, 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        for label, value in (
            ("pose costate", self.pose_costate_hwc),
            ("seg gap costate", self.seg_gap_costate_hwc),
        ):
            array = np.asarray(value)
            if array.dtype != np.float32 or array.shape != expected or not bool(np.isfinite(array).all()):
                raise ProjectedPopulationCostateError(f"{label} changed exact finite float32 batch ABI")
        if (
            np.asarray(self.base_pair_pose_mse).dtype != np.float32
            or np.asarray(self.base_pair_pose_mse).shape != (batch,)
            or np.asarray(self.target_pose6).dtype != np.float32
            or np.asarray(self.target_pose6).shape != (batch, 6)
        ):
            raise ProjectedPopulationCostateError("pose replay custody changed exact float32 ABI")
        if type(self.base_mismatch_count) is not int or self.base_mismatch_count < 0:
            raise ProjectedPopulationCostateError("base mismatch count must be nonnegative")


@dataclass(frozen=True, slots=True)
class G72BatchGroupV1:
    """One compact physical G72 batch-group basis element."""

    group_id: str
    role: str
    direction_rank: int
    amplitude_scale: str
    pair_ids: tuple[int, ...]
    proposals: tuple[G72BoundaryShearletProposalV1, ...]

    def __post_init__(self) -> None:
        if self.role not in {"Road", "UndrivableBoundary"}:
            raise ProjectedPopulationCostateError("G72 group escaped its two original roles")
        if self.direction_rank not in {0, 1} or self.amplitude_scale not in {"0.5", "1"}:
            raise ProjectedPopulationCostateError("G72 group escaped its four source variants")
        if not self.proposals:
            raise ProjectedPopulationCostateError("G72 batch group cannot be empty")
        if any(proposal.atom.role != self.role for proposal in self.proposals):
            raise ProjectedPopulationCostateError("G72 group mixed semantic roles")
        atom_keys = tuple(
            (row.atom.pair_index, row.atom.role, row.atom.center_y, row.atom.center_x) for row in self.proposals
        )
        if atom_keys != tuple(sorted(set(atom_keys))):
            raise ProjectedPopulationCostateError(
                "G72 batch group has colliding donor addresses and requires a new physical partition"
            )


@dataclass(frozen=True, slots=True)
class ProjectedOperandRowV1:
    """Separate-axis projection of one real finite actuator intervention."""

    operand_id: str
    family_id: str
    pair_ids: tuple[int, ...]
    operand_member_bytes: int
    operand_sha256: str
    atom_count: int
    changed_camera_values: int
    pose_linearized_score_delta: float
    seg_gap_directional_delta: float
    exact_zip_delta_bytes: int | None = None
    exact_seg_mismatch_delta: int | None = None
    exact_seg_score_delta: float | None = None
    exact_pose_mean_delta: float | None = None
    exact_pose_score_delta: float | None = None
    proposed_atoms_sha256: str | None = None
    incumbent_atoms_sha256: str | None = None

    def __post_init__(self) -> None:
        for value in (self.pose_linearized_score_delta, self.seg_gap_directional_delta):
            if not math.isfinite(float(value)):
                raise ProjectedPopulationCostateError("projected coordinates must be finite")
        if self.exact_zip_delta_bytes is not None and type(self.exact_zip_delta_bytes) is not int:
            raise ProjectedPopulationCostateError("ZIP delta must be one exact integer when measured")
        for label, value in (
            ("proposed atoms SHA-256", self.proposed_atoms_sha256),
            ("incumbent atoms SHA-256", self.incumbent_atoms_sha256),
        ):
            if value is not None and (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ProjectedPopulationCostateError(f"{label} is not canonical")

    @property
    def rate_status(self) -> str:
        return (
            "EXACT_COMPOSED_ZIP_DELTA"
            if self.exact_zip_delta_bytes is not None
            else "BLOCKED_MEMBER_BYTES_ARE_NOT_A_ZIP_DELTA"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "operand_id": self.operand_id,
            "family_id": self.family_id,
            "pair_ids": list(self.pair_ids),
            "operand_member_bytes": self.operand_member_bytes,
            "operand_sha256": self.operand_sha256,
            "atom_count": self.atom_count,
            "changed_camera_values": self.changed_camera_values,
            "pose_linearized_score_delta": self.pose_linearized_score_delta,
            "seg_gap_directional_delta": self.seg_gap_directional_delta,
            "exact_zip_delta_bytes": self.exact_zip_delta_bytes,
            "rate_status": self.rate_status,
            "exact_seg_mismatch_delta": self.exact_seg_mismatch_delta,
            "exact_seg_score_delta": self.exact_seg_score_delta,
            "exact_pose_mean_delta": self.exact_pose_mean_delta,
            "exact_pose_score_delta": self.exact_pose_score_delta,
            "proposed_atoms_sha256": self.proposed_atoms_sha256,
            "incumbent_atoms_sha256": self.incumbent_atoms_sha256,
        }


@runtime_checkable
class ActuatorTangentProviderV1(Protocol):
    """Plugin seam for G89 typed carriers and G88 conditional-pose tangents."""

    family_id: str

    def realize_batch(
        self,
        *,
        base_camera_pairs: np.ndarray,
        pair_ids: tuple[int, ...],
    ) -> tuple[str, bytes, np.ndarray]:
        """Return operand ID, serialized member bytes, and exact uint8 camera batch."""


def compute_batch_population_costates(
    *,
    candidate_pairs_hwc: np.ndarray,
    target_pairs_hwc: np.ndarray,
    target_cells: np.ndarray,
    described_cells: np.ndarray,
    pair_ids: tuple[int, ...],
    posenet: Any,
    segnet: Any,
    device: str,
    score_point: PopulationScorePointV1,
) -> BatchPopulationCostatesV1:
    """Differentiate the two scorer-native coordinates for one real batch."""

    import torch

    candidate_np = _exact_camera_batch(candidate_pairs_hwc, label="candidate pairs")
    target_np = _exact_camera_batch(target_pairs_hwc, label="target pairs")
    if candidate_np.shape != target_np.shape:
        raise ProjectedPopulationCostateError("candidate and target camera batches differ")
    batch = candidate_np.shape[0]
    target_cells_np = _exact_cells(target_cells, batch_pairs=batch, label="target cells")
    described_cells_np = _exact_cells(
        described_cells,
        batch_pairs=batch,
        label="described cells",
    )
    if len(pair_ids) != batch:
        raise ProjectedPopulationCostateError("pair IDs do not match scorer batch")

    torch_device = torch.device(device)
    candidate = (
        torch.from_numpy(candidate_np).to(torch_device).permute(0, 1, 4, 2, 3).float().contiguous().requires_grad_(True)
    )
    target = torch.from_numpy(target_np).to(torch_device).permute(0, 1, 4, 2, 3).float().contiguous()
    pose_pred = posenet(posenet.preprocess_input(candidate))
    seg_logits = segnet(segnet.preprocess_input(candidate))
    with torch.no_grad():
        target_pose = posenet(posenet.preprocess_input(target))
        target_logits = segnet(segnet.preprocess_input(target))
    if pose_pred["pose"].shape[-1] < 6 or target_pose["pose"].shape != pose_pred["pose"].shape:
        raise ProjectedPopulationCostateError("PoseNet output changed frozen pose-head ABI")

    target_cells_t = torch.from_numpy(target_cells_np.astype(np.int64)).to(torch_device)
    described_cells_t = torch.from_numpy(described_cells_np.astype(np.int64)).to(torch_device)
    actual_described = seg_logits.argmax(dim=1)
    actual_target = target_logits.argmax(dim=1)
    if not torch.equal(actual_described, described_cells_t):
        actual_described_np = actual_described.detach().cpu().numpy().astype(np.uint8)
        mismatch_np = actual_described_np != described_cells_np
        raise ProjectedPopulationCostateError(
            "current-base SegNet argmax differs from fresh G78 described cells",
            context={
                "failing_pair_range": [pair_ids[0], pair_ids[-1] + 1],
                "mismatch_cell_count": int(np.count_nonzero(mismatch_np)),
                "mismatch_pair_ids": [
                    pair_id
                    for pair_id, count in zip(
                        pair_ids,
                        mismatch_np.reshape(batch, -1).sum(axis=1),
                        strict=True,
                    )
                    if int(count) != 0
                ],
                "actual_cells_sha256": _sha256_array(actual_described_np),
                "expected_cells_sha256": _sha256_array(described_cells_np),
            },
        )
    if not torch.equal(actual_target, target_cells_t):
        raise ProjectedPopulationCostateError("fresh source SegNet argmax differs from owned G46 labels")

    pose_error = pose_pred["pose"][..., :6] - target_pose["pose"][..., :6]
    pair_pose_mse = pose_error.square().mean(dim=1)
    pose_objective = pair_pose_mse.sum() * score_point.pair_pose_mse_vjp_scale

    target_logit = seg_logits.gather(1, target_cells_t[:, None]).squeeze(1)
    current_logit = seg_logits.gather(1, described_cells_t[:, None]).squeeze(1)
    mismatch = target_cells_t != described_cells_t
    gap = target_logit - current_logit
    gap_objective = gap[mismatch].sum()

    pose_grad = torch.autograd.grad(
        pose_objective,
        candidate,
        retain_graph=True,
        create_graph=False,
    )[0]
    seg_grad = torch.autograd.grad(
        gap_objective,
        candidate,
        retain_graph=False,
        create_graph=False,
    )[0]
    pose_hwc = pose_grad.detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.float32)
    seg_hwc = seg_grad.detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.float32)
    pair_mse_np = pair_pose_mse.detach().cpu().numpy().astype(np.float32)
    target_pose_np = target_pose["pose"][..., :6].detach().cpu().numpy().astype(np.float32)
    return BatchPopulationCostatesV1(
        pair_ids=pair_ids,
        pose_costate_hwc=np.ascontiguousarray(pose_hwc),
        seg_gap_costate_hwc=np.ascontiguousarray(seg_hwc),
        base_pair_pose_mse=np.ascontiguousarray(pair_mse_np),
        target_pose6=np.ascontiguousarray(target_pose_np),
        base_mismatch_count=int(mismatch.sum().detach().cpu().item()),
        base_gap_sum=float(gap_objective.detach().cpu().item()),
        score_point=score_point,
        candidate_sha256=_sha256_array(candidate_np),
        target_sha256=_sha256_array(target_np),
        target_cells_sha256=_sha256_array(target_cells_np),
        described_cells_sha256=_sha256_array(described_cells_np),
    )


def group_g72_batch_proposals(
    proposals: tuple[G72BoundaryShearletProposalV1, ...],
    *,
    pair_ids: tuple[int, ...],
) -> tuple[G72BatchGroupV1, ...]:
    """Factor proposals into collision-free physical batch-group basis axes.

    There are eight semantic source axes before collision partitioning.  If two
    independently compiled components address the same physical donor, they
    cannot inhabit one G74 operand.  A deterministic first-fit partition keeps
    every source proposal while preserving the receiver's unique-address ABI.
    """

    pair_set = set(pair_ids)
    groups: dict[tuple[str, int, str], list[G72BoundaryShearletProposalV1]] = {}
    for proposal in proposals:
        if proposal.atom.pair_index not in pair_set:
            continue
        match = _G72_VARIANT.search(proposal.candidate_id)
        if match is None:
            raise ProjectedPopulationCostateError("G72 candidate ID lost source variant identity")
        direction = int(match.group("direction"))
        amplitude = match.group("amplitude")
        groups.setdefault((proposal.atom.role, direction, amplitude), []).append(proposal)
    result: list[G72BatchGroupV1] = []
    for (role, direction, amplitude), rows in sorted(groups.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                row.atom.pair_index,
                row.atom.role,
                row.atom.center_y,
                row.atom.center_x,
                row.candidate_id,
            ),
        )
        partitions: list[list[G72BoundaryShearletProposalV1]] = []
        partition_keys: list[set[tuple[int, str, int, int]]] = []
        for proposal in ordered:
            key = (
                proposal.atom.pair_index,
                proposal.atom.role,
                proposal.atom.center_y,
                proposal.atom.center_x,
            )
            for partition_index, keys in enumerate(partition_keys):
                if key not in keys:
                    partitions[partition_index].append(proposal)
                    keys.add(key)
                    break
            else:
                partitions.append([proposal])
                partition_keys.append({key})
        for partition_index, partition in enumerate(partitions):
            result.append(
                G72BatchGroupV1(
                    group_id=(
                        f"g72:{pair_ids[0]:04d}_{pair_ids[-1] + 1:04d}:"
                        f"{role}:d{direction}:a{amplitude}:p{partition_index}"
                    ),
                    role=role,
                    direction_rank=direction,
                    amplitude_scale=amplitude,
                    pair_ids=pair_ids,
                    proposals=tuple(partition),
                )
            )
    return tuple(result)


def realize_and_project_g72_group(
    *,
    decoder: V15RoleAwareOverlayDecoderV1,
    group: G72BatchGroupV1,
    base_camera_pairs: np.ndarray,
    costates: BatchPopulationCostatesV1,
    incumbent_atoms: tuple[BoundaryShearletAtomV1, ...] = (),
    incumbent_frame_selector: SelectedPreimageFrameSelectorV1 | None = None,
) -> tuple[ProjectedOperandRowV1, np.ndarray]:
    """Realize ``P + incumbent A + proposed dA`` and project its finite step."""

    base = _exact_camera_batch(base_camera_pairs, label="base camera pairs")
    if group.pair_ids != costates.pair_ids or base.shape[0] != len(group.pair_ids):
        raise ProjectedPopulationCostateError("G72 group/base/costate batch custody differs")
    relevant_incumbent = tuple(atom for atom in incumbent_atoms if atom.pair_index in set(group.pair_ids))
    if relevant_incumbent and type(incumbent_frame_selector) is not SelectedPreimageFrameSelectorV1:
        raise ProjectedPopulationCostateError("incumbent atoms require their exact original frame selector")
    proposed_atoms = tuple(row.atom for row in group.proposals)
    incumbent_keys = {(atom.pair_index, atom.role, atom.center_y, atom.center_x) for atom in relevant_incumbent}
    proposed_keys = {(atom.pair_index, atom.role, atom.center_y, atom.center_x) for atom in proposed_atoms}
    if incumbent_keys.intersection(proposed_keys):
        raise ProjectedPopulationCostateError(
            "G72 proposal collides with an incumbent current-base donor address; "
            "an exact replacement-coordinate law is required"
        )
    role_wire = {"UndrivableBoundary": 0, "Road": 1}
    incumbent_on_y1 = (
        relevant_incumbent
        if incumbent_frame_selector
        in {
            SelectedPreimageFrameSelectorV1.Y1,
            SelectedPreimageFrameSelectorV1.BOTH,
        }
        else ()
    )
    combined_atoms = tuple(
        sorted(
            (*incumbent_on_y1, *proposed_atoms),
            key=lambda atom: (
                atom.pair_index,
                role_wire[atom.role],
                atom.center_y,
                atom.center_x,
            ),
        )
    )
    operand = RoleAwareBoundaryShearletOperandV1(
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
        atoms=combined_atoms,
    )
    proposed_operand = RoleAwareBoundaryShearletOperandV1(
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
        atoms=proposed_atoms,
    )
    payload = operand.to_bytes()
    result = decoder.decode(
        payload,
        expected_operand_sha256=operand.sha256,
        maximum_operand_bytes=len(payload),
        local_pair_ids=group.pair_ids,
    )
    if relevant_incumbent:
        incumbent_operand = RoleAwareBoundaryShearletOperandV1(
            frame_selector=incumbent_frame_selector,
            atoms=relevant_incumbent,
        )
        incumbent_payload = incumbent_operand.to_bytes()
        incumbent_result = decoder.decode(
            incumbent_payload,
            expected_operand_sha256=incumbent_operand.sha256,
            maximum_operand_bytes=len(incumbent_payload),
            local_pair_ids=group.pair_ids,
        )
        realized_base = incumbent_result.camera_pairs
    else:
        realized_base = decoder.receiver.render_camera_pairs(group.pair_ids)
    if not np.array_equal(realized_base, base):
        raise ProjectedPopulationCostateError("G74 P+incumbent-A base differs from exact current G85 decoded base")
    candidate = base.copy()
    candidate[:, 1] = result.camera_pairs[:, 1]
    if not np.array_equal(candidate[:, 0], base[:, 0]):
        raise ProjectedPopulationCostateError("current-base Y1 tangent composition changed Y0")
    delta = candidate.astype(np.int16) - base.astype(np.int16)
    row = ProjectedOperandRowV1(
        operand_id=group.group_id,
        family_id="G72_CURRENT_BASE_COMPOSED_ROLE_AWARE_SHEARLET_BATCH_GROUP",
        pair_ids=group.pair_ids,
        operand_member_bytes=len(proposed_operand.to_bytes()),
        operand_sha256=proposed_operand.sha256,
        atom_count=len(group.proposals),
        changed_camera_values=int(np.count_nonzero(delta)),
        pose_linearized_score_delta=float(
            np.sum(costates.pose_costate_hwc.astype(np.float64) * delta.astype(np.float64))
        ),
        seg_gap_directional_delta=float(
            np.sum(costates.seg_gap_costate_hwc.astype(np.float64) * delta.astype(np.float64))
        ),
        proposed_atoms_sha256=proposed_operand.sha256,
        incumbent_atoms_sha256=(
            hashlib.sha256(
                RoleAwareBoundaryShearletOperandV1(
                    frame_selector=incumbent_frame_selector,
                    atoms=relevant_incumbent,
                ).to_bytes()
                if relevant_incumbent
                else b""
            ).hexdigest()
        ),
    )
    return row, np.ascontiguousarray(candidate)


def pareto_nondominated_projection_ids(
    rows: tuple[ProjectedOperandRowV1, ...],
) -> tuple[str, ...]:
    """Return the exact two-coordinate Pareto set without thresholds or weights."""

    output: list[str] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            pose_no_worse = other.pose_linearized_score_delta <= row.pose_linearized_score_delta
            seg_no_worse = other.seg_gap_directional_delta >= row.seg_gap_directional_delta
            one_strict = (
                other.pose_linearized_score_delta < row.pose_linearized_score_delta
                or other.seg_gap_directional_delta > row.seg_gap_directional_delta
            )
            if pose_no_worse and seg_no_worse and one_strict:
                dominated = True
                break
        if not dominated:
            output.append(row.operand_id)
    return tuple(output)


def exact_replay_projected_intervention(
    row: ProjectedOperandRowV1,
    *,
    candidate_pairs_hwc: np.ndarray,
    target_cells: np.ndarray,
    costates: BatchPopulationCostatesV1,
    posenet: Any,
    segnet: Any,
    device: str,
) -> ProjectedOperandRowV1:
    """Attach exact component deltas for a finite intervention affecting one batch."""

    import torch

    candidate_np = _exact_camera_batch(candidate_pairs_hwc, label="intervention pairs")
    target_cells_np = _exact_cells(
        target_cells,
        batch_pairs=candidate_np.shape[0],
        label="target cells",
    )
    torch_device = torch.device(device)
    candidate = torch.from_numpy(candidate_np).to(torch_device).permute(0, 1, 4, 2, 3).float().contiguous()
    with torch.inference_mode():
        pose = posenet(posenet.preprocess_input(candidate))["pose"][..., :6]
        logits = segnet(segnet.preprocess_input(candidate))
    target_pose = torch.from_numpy(costates.target_pose6).to(torch_device)
    after_pair_pose = (pose - target_pose).square().mean(dim=1)
    target_cells_t = torch.from_numpy(target_cells_np.astype(np.int64)).to(torch_device)
    after_mismatches = int((logits.argmax(dim=1) != target_cells_t).sum().detach().cpu().item())
    mismatch_delta = after_mismatches - costates.base_mismatch_count
    pose_sum_delta = float(
        after_pair_pose.sum().detach().cpu().item()
        - np.asarray(costates.base_pair_pose_mse, dtype=np.float32).sum(dtype=np.float32)
    )
    pose_mean_delta = pose_sum_delta / costates.score_point.sample_count
    new_pose_mean = costates.score_point.global_mean_pose_dist + pose_mean_delta
    if not math.isfinite(new_pose_mean) or new_pose_mean <= 0.0:
        raise ProjectedPopulationCostateError("intervention escaped positive global pose distortion")
    return ProjectedOperandRowV1(
        operand_id=row.operand_id,
        family_id=row.family_id,
        pair_ids=row.pair_ids,
        operand_member_bytes=row.operand_member_bytes,
        operand_sha256=row.operand_sha256,
        atom_count=row.atom_count,
        changed_camera_values=row.changed_camera_values,
        pose_linearized_score_delta=row.pose_linearized_score_delta,
        seg_gap_directional_delta=row.seg_gap_directional_delta,
        exact_zip_delta_bytes=row.exact_zip_delta_bytes,
        exact_seg_mismatch_delta=mismatch_delta,
        exact_seg_score_delta=(
            100.0 * mismatch_delta / (costates.score_point.sample_count * SCORER_HEIGHT * SCORER_WIDTH)
        ),
        exact_pose_mean_delta=pose_mean_delta,
        exact_pose_score_delta=(
            math.sqrt(10.0 * new_pose_mean) - math.sqrt(10.0 * costates.score_point.global_mean_pose_dist)
        ),
        proposed_atoms_sha256=row.proposed_atoms_sha256,
        incumbent_atoms_sha256=row.incumbent_atoms_sha256,
    )


__all__ = [
    "G72_GROUP_LAW",
    "POSE_COSTATE_COORDINATE",
    "SEG_SCREEN_COORDINATE",
    "ActuatorTangentProviderV1",
    "BatchPopulationCostatesV1",
    "G72BatchGroupV1",
    "PopulationScorePointV1",
    "ProjectedOperandRowV1",
    "ProjectedPopulationCostateError",
    "compute_batch_population_costates",
    "exact_replay_projected_intervention",
    "group_g72_batch_proposals",
    "pareto_nondominated_projection_ids",
    "realize_and_project_g72_group",
]
