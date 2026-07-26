from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.optimization.v10_constructive_solver import (
    EXPECTED_SOURCE_HASHES,
    HARD_ORACLE_SCHEMA,
    RECEIVER_ARITHMETIC,
    HardOracleDecision,
)
from tac.witness_dsl.coupled_witness_state import ContentAddress, FrozenSpaceIdentity, canonical_sha256
from tac.witness_dsl.evaluator_obligation_ir import (
    CLASS_NAMES,
    FINAL_ADMISSION_POLICY,
    JOINT_OBJECTIVE_ID,
    CollateralOwner,
    ConditionalFrame0PoseFibreObligation,
    EvaluatorObligationIR,
    EvaluatorObligationIRError,
    ExplicitV10PreimageCompileResult,
    Frame1CellObligation,
    PairCompileRequest,
    PairHardOracleEvidence,
    compile_explicit_v10_preimages,
    frame1_pair_obligation_sha256,
)
from tac.witness_dsl.v10_production_receiver import (
    DESCRIPTION_FRAME0_POLICY_ID,
    RECEIVER_CONTRACT_ID,
)

PAIR_COUNT = 2
SCORER_H = 3
SCORER_W = 4
CAMERA_H = 8
CAMERA_W = 10


def _address(artifact_id: str) -> ContentAddress:
    return ContentAddress.from_payload(
        artifact_id=artifact_id,
        artifact_schema="test.fixture.v1",
        payload=f"payload:{artifact_id}".encode(),
    )


def _frozen_space() -> FrozenSpaceIdentity:
    evaluator_artifacts = tuple(
        sorted(
            (
                ContentAddress(
                    artifact_id=artifact_id,
                    artifact_schema="test.frozen-oracle-artifact.v1",
                    sha256=EXPECTED_SOURCE_HASHES[source_key],
                    byte_length=1,
                )
                for artifact_id, source_key in {
                    "upstream/frame_utils.py": "frame_utils_sha256",
                    "upstream/models/posenet.safetensors": "posenet_weights_sha256",
                    "upstream/models/segnet.safetensors": "segnet_weights_sha256",
                    "upstream/modules.py": "modules_sha256",
                }.items()
            ),
            key=lambda item: item.artifact_id,
        )
    )
    return FrozenSpaceIdentity(
        source_video=_address("source_video"),
        evaluator_artifacts=evaluator_artifacts,
        pair_count=PAIR_COUNT,
        pair_order_id="canonical-contiguous-pair-order.v1",
        pair_order_sha256=canonical_sha256(list(range(PAIR_COUNT))),
        scorer_height=SCORER_H,
        scorer_width=SCORER_W,
    )


def _cells() -> tuple[Frame1CellObligation, ...]:
    rows: list[Frame1CellObligation] = []
    owners = tuple(CollateralOwner)
    addresses = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0))
    for pair_id in range(PAIR_COUNT):
        for winner, (row, col) in enumerate(addresses):
            margins = [0.25] * len(CLASS_NAMES)
            margins[winner] = 0.0
            rows.append(
                Frame1CellObligation(
                    pair_id=pair_id,
                    row=row,
                    col=col,
                    winner_class_id=winner,
                    required_margin_by_class=tuple(margins),  # type: ignore[arg-type]
                    collateral_owner=owners[winner % len(owners)],
                )
            )
    return tuple(rows)


def _obligation_ir() -> EvaluatorObligationIR:
    cells = _cells()
    fibres = tuple(
        ConditionalFrame0PoseFibreObligation(
            pair_id=pair_id,
            target_pose6=tuple(float(pair_id + index) for index in range(6)),  # type: ignore[arg-type]
            conditioned_frame1_obligation_sha256=frame1_pair_obligation_sha256(cells, pair_id),
        )
        for pair_id in range(PAIR_COUNT)
    )
    return EvaluatorObligationIR(
        frozen_space=_frozen_space(),
        coupled_state_sha256="a" * 64,
        predictor_state_sha256="b" * 64,
        predictor_semantic_sha256="c" * 64,
        camera_height=CAMERA_H,
        camera_width=CAMERA_W,
        frame1_cells=cells,
        conditional_frame0_pose_fibres=fibres,
    )


def _planes() -> tuple[np.ndarray, np.ndarray]:
    y0 = np.arange(PAIR_COUNT * SCORER_H * SCORER_W * 3, dtype=np.uint8).reshape(PAIR_COUNT, SCORER_H, SCORER_W, 3)
    y1 = np.flip(y0, axis=2).copy()
    return y0, y1


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _oracle(
    mutate: Callable[[np.ndarray, np.ndarray, PairCompileRequest], None] | None = None,
) -> Callable[[PairCompileRequest, tuple[np.ndarray, np.ndarray]], PairHardOracleEvidence]:
    def evaluate(
        request: PairCompileRequest,
        frames: tuple[np.ndarray, np.ndarray],
    ) -> PairHardOracleEvidence:
        logits = np.zeros((len(request.frame1_cells), len(CLASS_NAMES)), dtype=np.float32)
        for index, cell in enumerate(request.frame1_cells):
            logits[index, cell.winner_class_id] = 2.0
        pose = np.asarray(request.conditional_frame0_pose_fibre.target_pose6, dtype=np.float32)
        if mutate is not None:
            mutate(logits, pose, request)
        residual = pose - np.asarray(request.conditional_frame0_pose_fibre.target_pose6, dtype=np.float32)
        d_pose = float(np.mean(residual * residual, dtype=np.float32))
        decision = HardOracleDecision(
            admitted=True,
            schema=HARD_ORACLE_SCHEMA,
            receiver_arithmetic=RECEIVER_ARITHMETIC,
            realized_frame_sha256s=(_array_sha(frames[0]), _array_sha(frames[1])),
            d_seg=0.0,
            d_pose=d_pose,
            source_hashes=EXPECTED_SOURCE_HASHES,
        )
        return PairHardOracleEvidence(
            pair_id=request.pair_id,
            evaluator_obligation_ir_sha256=request.evaluator_obligation_ir_sha256,
            pair_obligation_sha256=request.pair_obligation_sha256,
            decision=decision,
            frame1_cell_logits=logits,
            pose6=pose,
        )

    return evaluate


def test_obligation_ir_is_canonical_five_class_and_identity_bound() -> None:
    ir = _obligation_ir()
    decoded = EvaluatorObligationIR.from_bytes(ir.to_bytes())
    assert decoded == ir
    assert decoded.ir_sha256 == ir.ir_sha256
    assert {cell.winner_class_id for cell in ir.frame1_cells} == set(range(5))
    assert ir.as_dict()["class_names"] == list(CLASS_NAMES)
    assert ir.joint_objective_id == JOINT_OBJECTIVE_ID
    assert ir.final_admission_policy == FINAL_ADMISSION_POLICY
    assert ir.decoder_contains_scorer is False
    assert ir.decoder_contains_target_table is False
    assert ir.borrowed_candidate_bytes == 0


def test_obligation_ir_refuses_address_margin_order_and_pose_condition_drift() -> None:
    ir = _obligation_ir()
    first = ir.frame1_cells[0]
    with pytest.raises(EvaluatorObligationIRError, match="winner-to-self"):
        replace(first, required_margin_by_class=(1.0, 0.25, 0.25, 0.25, 0.25))
    with pytest.raises(EvaluatorObligationIRError, match="five-class universe"):
        replace(first, winner_class_id=5)
    with pytest.raises(EvaluatorObligationIRError, match="canonical pair/row/col order"):
        replace(ir, frame1_cells=tuple(reversed(ir.frame1_cells)))
    with pytest.raises(EvaluatorObligationIRError, match="exceeds frozen scorer geometry"):
        replace(
            ir,
            frame1_cells=(*ir.frame1_cells[:4], replace(ir.frame1_cells[4], row=SCORER_H), *ir.frame1_cells[5:]),
        )
    drifted_fibres = (replace(ir.conditional_frame0_pose_fibres[0], conditioned_frame1_obligation_sha256="d" * 64),)
    with pytest.raises(EvaluatorObligationIRError, match="exactly one row per pair"):
        replace(ir, conditional_frame0_pose_fibres=drifted_fibres)
    with pytest.raises(EvaluatorObligationIRError, match="not conditioned"):
        replace(
            ir,
            conditional_frame0_pose_fibres=(
                replace(ir.conditional_frame0_pose_fibres[0], conditioned_frame1_obligation_sha256="d" * 64),
                ir.conditional_frame0_pose_fibres[1],
            ),
        )


def test_explicit_preimages_run_real_solver_oracle_and_receiver_roundtrip() -> None:
    ir = _obligation_ir()
    y0, y1 = _planes()
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )
    result = compile_explicit_v10_preimages(
        ir,
        scorer_y0=y0,
        scorer_y1=y1,
        operator=operator,
        hard_oracle=_oracle(),
    )
    result.verify_preimages(y0, y1)
    decoded = ExplicitV10PreimageCompileResult.from_bytes(result.to_bytes())
    assert decoded == result
    assert decoded.result_sha256 == result.result_sha256
    assert decoded.receiver_contract_id == RECEIVER_CONTRACT_ID
    assert decoded.frame0_policy_id == DESCRIPTION_FRAME0_POLICY_ID
    assert decoded.receiver_packet_bytes > y0.size
    assert all(row.observed_pose_mse == 0.0 for row in decoded.pair_receipts)
    assert all(all(proof["certified_exact"] for proof in row.factor2_proofs) for row in decoded.pair_receipts)
    assert decoded.score_claim is False
    assert decoded.promotion_eligible is False
    assert decoded.archive_receipt_owed is True

    changed = y1.copy()
    changed[0, 0, 0, 0] ^= 1
    with pytest.raises(EvaluatorObligationIRError, match="Y1 bytes differ"):
        result.verify_preimages(y0, changed)


def test_compile_refuses_absent_or_non_evidentiary_oracle_and_non_uint8_preimages() -> None:
    ir = _obligation_ir()
    y0, y1 = _planes()
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )
    with pytest.raises(EvaluatorObligationIRError, match="real hard oracle"):
        compile_explicit_v10_preimages(ir, scorer_y0=y0, scorer_y1=y1, operator=operator, hard_oracle=None)
    with pytest.raises(EvaluatorObligationIRError, match="exact uint8 geometry"):
        compile_explicit_v10_preimages(
            ir,
            scorer_y0=y0.astype(np.int16),
            scorer_y1=y1,
            operator=operator,
            hard_oracle=_oracle(),
        )

    def no_evidence(
        request: PairCompileRequest,
        frames: tuple[np.ndarray, np.ndarray],
    ) -> PairHardOracleEvidence:
        del request, frames
        return cast("PairHardOracleEvidence", object())

    with pytest.raises(EvaluatorObligationIRError, match="no typed observation evidence"):
        compile_explicit_v10_preimages(
            ir,
            scorer_y0=y0,
            scorer_y1=y1,
            operator=operator,
            hard_oracle=no_evidence,
        )


def test_compile_refuses_hard_oracle_winner_margin_and_pose_drift() -> None:
    ir = _obligation_ir()
    y0, y1 = _planes()
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )

    def wrong_winner(logits: np.ndarray, pose: np.ndarray, request: PairCompileRequest) -> None:
        del pose, request
        logits[0] = 0.0
        logits[0, 4] = 3.0

    with pytest.raises(EvaluatorObligationIRError, match="winner cell"):
        compile_explicit_v10_preimages(
            ir,
            scorer_y0=y0,
            scorer_y1=y1,
            operator=operator,
            hard_oracle=_oracle(wrong_winner),
        )

    def weak_margin(logits: np.ndarray, pose: np.ndarray, request: PairCompileRequest) -> None:
        del pose, request
        logits[0, 1] = 1.9

    with pytest.raises(EvaluatorObligationIRError, match="class margin"):
        compile_explicit_v10_preimages(
            ir,
            scorer_y0=y0,
            scorer_y1=y1,
            operator=operator,
            hard_oracle=_oracle(weak_margin),
        )

    def pose_drift(logits: np.ndarray, pose: np.ndarray, request: PairCompileRequest) -> None:
        del logits, request
        pose[0] += np.float32(0.5)

    result = compile_explicit_v10_preimages(
        ir,
        scorer_y0=y0,
        scorer_y1=y1,
        operator=operator,
        hard_oracle=_oracle(pose_drift),
    )
    assert result.pair_receipts[0].observed_pose_mse > 0.0
    assert result.pair_receipts[0].oracle_d_pose == pytest.approx(result.pair_receipts[0].observed_pose_mse)


def test_compile_refuses_geometry_drift_and_authority_claim_mutation() -> None:
    ir = _obligation_ir()
    y0, y1 = _planes()
    wrong_operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H + 1,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )
    with pytest.raises(EvaluatorObligationIRError, match="geometry differs"):
        compile_explicit_v10_preimages(
            ir,
            scorer_y0=y0,
            scorer_y1=y1,
            operator=wrong_operator,
            hard_oracle=_oracle(),
        )

    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )
    result = compile_explicit_v10_preimages(
        ir,
        scorer_y0=y0,
        scorer_y1=y1,
        operator=operator,
        hard_oracle=_oracle(),
    )
    with pytest.raises(EvaluatorObligationIRError, match="forbidden authority"):
        replace(result, score_claim=True)
    with pytest.raises(EvaluatorObligationIRError, match="archive receipt debt"):
        replace(result, archive_receipt_owed=False)
