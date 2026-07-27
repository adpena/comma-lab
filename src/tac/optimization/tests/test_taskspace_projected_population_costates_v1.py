from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.optimization.taskspace_projected_population_costates_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    PopulationScorePointV1,
    ProjectedOperandRowV1,
    compute_batch_population_costates,
    exact_replay_projected_intervention,
    group_g72_batch_proposals,
    pareto_nondominated_projection_ids,
    realize_and_project_g72_group,
)
from tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 import (
    G72BoundaryShearletProposalV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    parse_role_aware_boundary_shearlet_operand,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

H = "a" * 64


class _ToyPoseNet:
    def preprocess_input(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def __call__(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        summary = value.mean(dim=(1, 2, 3, 4), keepdim=False)[:, None]
        return {"pose": summary.repeat(1, 12)}


class _ToySegNet:
    def preprocess_input(self, value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            value[:, -1],
            size=(SCORER_HEIGHT, SCORER_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        intensity = value.mean(dim=1, keepdim=True)
        impossible = torch.full_like(intensity, -1000.0)
        return torch.cat(
            (255.0 - intensity, intensity, impossible, impossible, impossible),
            dim=1,
        )


class _FakeSemanticReceiver:
    def render_camera_pairs(self, pair_ids: tuple[int, ...]) -> np.ndarray:
        return np.zeros(
            (len(pair_ids), 2, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            dtype=np.uint8,
        )


class _FakeG74Decoder:
    receiver = _FakeSemanticReceiver()

    def decode(self, payload: bytes, **kwargs: object) -> SimpleNamespace:
        operand = parse_role_aware_boundary_shearlet_operand(payload)
        pair_ids = kwargs["local_pair_ids"]
        assert isinstance(pair_ids, tuple)
        result = self.receiver.render_camera_pairs(pair_ids)
        selected = (
            (0, 1)
            if operand.frame_selector is SelectedPreimageFrameSelectorV1.BOTH
            else ((0,) if operand.frame_selector is SelectedPreimageFrameSelectorV1.Y0 else (1,))
        )
        for frame_index in selected:
            result[:, frame_index, 0, 0, 0] = len(operand.atoms)
        return SimpleNamespace(camera_pairs=result)


def _score_point() -> PopulationScorePointV1:
    return PopulationScorePointV1(
        global_mean_pose_dist=163.06130981,
        sample_count=600,
        archive_bytes=129_392,
        archive_sha256=H,
    )


def _proposal(
    *,
    candidate_id: str,
    pair_index: int,
    center_x: int,
    role: str = "Road",
) -> G72BoundaryShearletProposalV1:
    return G72BoundaryShearletProposalV1(
        candidate_id=candidate_id,
        fisher_priority=1.0,
        atom=BoundaryShearletAtomV1(
            pair_index=pair_index,
            role=role,
            center_y=100,
            center_x=center_x,
            scale_y=8,
            scale_x=16,
            shear_q4=0,
            amplitude_q4=8,
        ),
    )


def _row(
    operand_id: str,
    *,
    pose: float,
    seg: float,
) -> ProjectedOperandRowV1:
    return ProjectedOperandRowV1(
        operand_id=operand_id,
        family_id="test",
        pair_ids=(0,),
        operand_member_bytes=7,
        operand_sha256=H,
        atom_count=1,
        changed_camera_values=1,
        pose_linearized_score_delta=pose,
        seg_gap_directional_delta=seg,
    )


def test_score_point_uses_exact_population_pose_chain_rule() -> None:
    point = _score_point()
    expected = 5.0 / (600.0 * math.sqrt(10.0 * 163.06130981))
    assert point.pair_pose_mse_vjp_scale == pytest.approx(expected, rel=0.0, abs=0.0)
    assert point.pair_pose_mse_vjp_scale == pytest.approx(
        0.00020636844449905425,
        rel=0.0,
        abs=1e-20,
    )


def test_g72_grouping_preserves_colliding_proposals_via_physical_partitions() -> None:
    proposals = (
        _proposal(candidate_id="c0_sh_d0_a1", pair_index=0, center_x=10),
        _proposal(candidate_id="c1_sh_d0_a1", pair_index=0, center_x=10),
        _proposal(candidate_id="c2_sh_d0_a1", pair_index=1, center_x=20),
        _proposal(
            candidate_id="c3_sh_d1_a0.5",
            pair_index=1,
            center_x=30,
            role="UndrivableBoundary",
        ),
    )
    groups = group_g72_batch_proposals(proposals, pair_ids=(0, 1))
    assert sum(len(group.proposals) for group in groups) == len(proposals)
    assert [group.group_id for group in groups] == [
        "g72:0000_0002:Road:d0:a1:p0",
        "g72:0000_0002:Road:d0:a1:p1",
        "g72:0000_0002:UndrivableBoundary:d1:a0.5:p0",
    ]


def test_projection_pareto_uses_separate_pose_and_seg_axes_without_rate_proxy() -> None:
    rows = (
        _row("a", pose=0.0, seg=1.0),
        _row("b", pose=1.0, seg=0.0),
        _row("c", pose=-1.0, seg=0.5),
        _row("d", pose=0.0, seg=2.0),
    )
    assert pareto_nondominated_projection_ids(rows) == ("c", "d")
    assert all(row.rate_status == "BLOCKED_MEMBER_BYTES_ARE_NOT_A_ZIP_DELTA" for row in rows)


def test_real_batch_costates_and_exact_replay_stay_scorer_native() -> None:
    candidate = np.zeros(
        (1, 2, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8,
    )
    target = np.full_like(candidate, 255)
    target_cells = np.ones((1, SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint8)
    described_cells = np.zeros_like(target_cells)
    costates = compute_batch_population_costates(
        candidate_pairs_hwc=candidate,
        target_pairs_hwc=target,
        target_cells=target_cells,
        described_cells=described_cells,
        pair_ids=(0,),
        posenet=_ToyPoseNet(),
        segnet=_ToySegNet(),
        device="cpu",
        score_point=_score_point(),
    )
    assert costates.base_mismatch_count == SCORER_HEIGHT * SCORER_WIDTH
    assert costates.base_pair_pose_mse.tolist() == pytest.approx([255.0**2], abs=0.01)
    assert np.count_nonzero(costates.pose_costate_hwc[:, 0]) > 0
    assert np.count_nonzero(costates.seg_gap_costate_hwc[:, 0]) == 0
    assert np.count_nonzero(costates.seg_gap_costate_hwc[:, 1]) > 0

    screened = _row("replace", pose=-1.0, seg=1.0)
    replayed = exact_replay_projected_intervention(
        screened,
        candidate_pairs_hwc=target,
        target_cells=target_cells,
        costates=costates,
        posenet=_ToyPoseNet(),
        segnet=_ToySegNet(),
        device="cpu",
    )
    assert replayed.exact_seg_mismatch_delta == -(SCORER_HEIGHT * SCORER_WIDTH)
    assert replayed.exact_seg_score_delta == pytest.approx(-100.0 / 600.0)
    assert replayed.exact_pose_mean_delta == pytest.approx(-float(costates.base_pair_pose_mse[0]) / 600.0)
    assert replayed.exact_pose_score_delta is not None
    assert replayed.exact_pose_score_delta < 0.0
    assert replayed.exact_zip_delta_bytes is None

    incumbent = BoundaryShearletAtomV1(
        pair_index=0,
        role="Road",
        center_y=10,
        center_x=10,
        scale_y=8,
        scale_x=16,
        shear_q4=0,
        amplitude_q4=8,
    )
    proposed = _proposal(
        candidate_id="current_sh_d0_a1",
        pair_index=0,
        center_x=20,
    )
    group = group_g72_batch_proposals((proposed,), pair_ids=(0,))[0]
    current_base = np.zeros_like(candidate)
    current_base[:, :, 0, 0, 0] = 1
    row, mixed = realize_and_project_g72_group(
        decoder=_FakeG74Decoder(),  # type: ignore[arg-type]
        group=group,
        base_camera_pairs=current_base,
        costates=costates,
        incumbent_atoms=(incumbent,),
        incumbent_frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
    )
    assert mixed[0, 0, 0, 0, 0] == 1
    assert mixed[0, 1, 0, 0, 0] == 2
    assert row.proposed_atoms_sha256 is not None
    assert row.incumbent_atoms_sha256 is not None


def test_current_cell_drift_carries_exact_retry_diagnostics() -> None:
    candidate = np.zeros(
        (1, 2, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8,
    )
    target = np.full_like(candidate, 255)
    target_cells = np.ones((1, SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint8)
    wrong_described_cells = np.ones_like(target_cells)
    with pytest.raises(
        ValueError,
        match="current-base SegNet argmax differs",
    ) as caught:
        compute_batch_population_costates(
            candidate_pairs_hwc=candidate,
            target_pairs_hwc=target,
            target_cells=target_cells,
            described_cells=wrong_described_cells,
            pair_ids=(17,),
            posenet=_ToyPoseNet(),
            segnet=_ToySegNet(),
            device="cpu",
            score_point=_score_point(),
        )
    context = caught.value.context
    assert context["failing_pair_range"] == [17, 18]
    assert context["mismatch_pair_ids"] == [17]
    assert context["mismatch_cell_count"] == SCORER_HEIGHT * SCORER_WIDTH
    assert context["actual_cells_sha256"] != context["expected_cells_sha256"]
