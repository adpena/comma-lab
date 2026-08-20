# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tac.codec.v10_predictor_residual import CODEC_ID as TWO_PLANE_CODEC_ID
from tac.optimization.direct_description_carrier_compose import TopologyEventV1
from tac.witness_dsl.coupled_witness_state import (
    ContentAddress,
    CoupledWitnessState,
    FrozenSpaceIdentity,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
)
from tac.witness_dsl.evaluator_obligation_ir import (
    V10_SOLVER_CONTRACT_ID,
    CollateralOwner,
    ConditionalFrame0PoseFibreObligation,
    EvaluatorObligationIR,
    ExplicitV10PreimageCompileResult,
    Frame1CellObligation,
    PairPreimageReceipt,
    frame1_pair_obligation_sha256,
)
from tac.witness_dsl.factorized_v9_predictor import (
    PREDICTOR_CONTRACT_ID,
    receive_factorized_v9_predictor,
)
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
    PredictorSemanticStateV1,
    apply_generative_taskspace_correction,
    compile_generative_taskspace_correction,
)
from tac.witness_dsl.pair_population_envelope import (
    CompactGeneratorBatch,
    CompactGeneratorDecode,
    CompactObligationGeneratorProgram,
    CompactProgramRole,
    CountedArtifactClass,
    CountedPayloadLineage,
    CountedPayloadProvenance,
    CountedProgramSection,
    CountedSectionRole,
    ExclusivePoseOwnership,
    IRCoveragePolicy,
    PairPopulation,
    PairPopulationEnvelope,
    PairPopulationEnvelopeError,
    ReopenedObjectJoin,
    RoleCounterfactualProgram,
    SparseDebtOwner,
    SparseObligationOwnership,
    _bind_receiver_source,
    _require_canonical_frozen_pair_order,
    derive_ir_coverage,
    reopen_pbr2_pair_reference,
    reopen_typed_pair_reference,
    validate_counted_program_sections,
    validate_reverse_causal_counted_program_sections,
)
from tac.witness_dsl.progressive_geometry_residual import build_progressive_geometry_residual
from tac.witness_dsl.tests.test_factorized_v9_predictor import _program
from tac.witness_dsl.v10_production_receiver import (
    DESCRIPTION_FRAME0_POLICY_ID,
    RECEIVER_CONTRACT_ID,
)

CELL_ROW = 100
CELL_COL = 100
COMPACT_SEED = 7
V19C_TYPED_PAIR_IDS = (447, 53, 416, 296, 547, 278, 501, 346)


def _address(artifact_id: str) -> ContentAddress:
    return ContentAddress.from_payload(
        artifact_id=artifact_id,
        artifact_schema="test.pair_population_envelope.v1",
        payload=artifact_id.encode("ascii"),
    )


def _frozen_space() -> FrozenSpaceIdentity:
    return FrozenSpaceIdentity(
        source_video=_address("source-video"),
        evaluator_artifacts=tuple(
            sorted(
                (
                    _address("upstream/frame_utils.py"),
                    _address("upstream/models/posenet.safetensors"),
                    _address("upstream/models/segnet.safetensors"),
                    _address("upstream/modules.py"),
                ),
                key=lambda item: item.artifact_id,
            )
        ),
        pair_count=1,
        pair_order_id="canonical-contiguous-pairs.v1",
        pair_order_sha256=canonical_sha256([0]),
        scorer_height=384,
        scorer_width=512,
    )


def _array_identity_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return canonical_sha256(
        {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "byte_length": int(contiguous.nbytes),
            "bytes_sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
        }
    )


def _generated_planes(
    seed: int,
    *,
    g_labels: np.ndarray,
    frame0_gain: int,
    terminal_bias: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(384 * 512 * 3, dtype=np.uint32).reshape(1, 384, 512, 3)
    y1 = ((values + seed + g_labels[..., None] + terminal_bias) % 256).astype(np.uint8)
    y0 = ((values * 3 + frame0_gain + 11) % 256).astype(np.uint8)
    return y0, y1


def _proof() -> dict[str, Any]:
    return {
        "scorer_values": 1,
        "owned_camera_values": 1,
        "unowned_camera_values": 0,
        "numerator_equal_values": 1,
        "canonical_equal_values": 1,
        "denominator": 1,
        "numerator_exact": True,
        "certified_exact": True,
    }


def _explicit_result(ir: EvaluatorObligationIR, y0: np.ndarray, y1: np.ndarray) -> ExplicitV10PreimageCompileResult:
    y0_identity = _array_identity_sha256(y0)
    y1_identity = _array_identity_sha256(y1)
    digest = hashlib.sha256(b"typed-proof").hexdigest()
    pair_receipt = PairPreimageReceipt(
        pair_id=0,
        pair_obligation_sha256=ir.pair_obligation_sha256(0),
        scorer_y0_identity_sha256=y0_identity,
        scorer_y1_identity_sha256=y1_identity,
        camera_frame0_sha256=digest,
        camera_frame1_sha256=digest,
        factor2_proofs=(_proof(), _proof()),
        hard_oracle_decision_sha256=digest,
        hard_oracle_receipt_sha256=digest,
        hard_oracle_receipt_bytes=1,
        observed_cell_logits_identity_sha256=digest,
        observed_pose6_identity_sha256=digest,
        observed_pose_mse=0.0,
        oracle_d_seg=0.0,
        oracle_d_pose=0.0,
    )
    return ExplicitV10PreimageCompileResult(
        evaluator_obligation_ir_sha256=ir.ir_sha256,
        frozen_space_identity_sha256=ir.frozen_space.identity_sha256,
        coupled_state_sha256=ir.coupled_state_sha256,
        predictor_state_sha256=ir.predictor_state_sha256,
        predictor_semantic_sha256=ir.predictor_semantic_sha256,
        pair_count=1,
        camera_height=768,
        camera_width=1024,
        scorer_height=384,
        scorer_width=512,
        scorer_y0_identity_sha256=y0_identity,
        scorer_y1_identity_sha256=y1_identity,
        pair_receipts=(pair_receipt,),
        solver_contract_id=V10_SOLVER_CONTRACT_ID,
        solver_source_sha256=digest,
        receiver_contract_id=RECEIVER_CONTRACT_ID,
        receiver_source_sha256=digest,
        receiver_packet_sha256=hashlib.sha256(b"dense-v10-packet").hexdigest(),
        receiver_packet_bytes=len(b"dense-v10-packet"),
        y_codec_id=TWO_PLANE_CODEC_ID,
        frame0_policy_id=DESCRIPTION_FRAME0_POLICY_ID,
    )


def _g_teacher(labels: np.ndarray) -> EncoderOnlyTeacherEvidenceV1:
    target = np.ascontiguousarray(labels, dtype=np.uint8)
    return EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=hashlib.sha256(memoryview(target).cast("B")).hexdigest(),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=0,
    )


class _Bundle:
    def __init__(self, *, target_differs: bool) -> None:
        self.program = _program()
        receiver = receive_factorized_v9_predictor(self.program)
        predictor = receiver.decode_all_semantics()
        predictor_state = PredictorSemanticStateV1(
            predictor_program_sha256=receiver.program_sha256,
            predictor_renderer_sha256=receiver.source_manifest_sha256,
            source_pair_ids=receiver.source_pair_ids,
            labels=predictor,
            pose6_codes=np.ascontiguousarray(receiver.receiver.pose6_codes),
        )
        event_sites = np.argwhere(predictor[0] != 4)

        def g_program_at(event_row: int, event_col: int) -> GenerativeCorrectionProgramV1:
            return GenerativeCorrectionProgramV1(
                topology_events=(
                    TopologyEventV1(
                        0,
                        "MyCar",
                        "birth",
                        "box",
                        1,
                        event_row,
                        event_col,
                        event_row + 1,
                        event_col + 1,
                    ),
                )
            )

        base_row, base_col = (int(value) for value in event_sites[0])
        alternate_row, alternate_col = (int(value) for value in event_sites[-1])
        compiled_g = compile_generative_taskspace_correction(
            predictor_state,
            g_program_at(base_row, base_col),
            teacher_evidence=_g_teacher(predictor),
        )
        alternate_g = compile_generative_taskspace_correction(
            predictor_state,
            g_program_at(alternate_row, alternate_col),
            teacher_evidence=_g_teacher(predictor),
        )
        assert compiled_g.packet != alternate_g.packet
        assert len(compiled_g.packet) == len(alternate_g.packet)
        self.g_program = compiled_g.packet
        self.g_program_alternative = alternate_g.packet
        self.g_program_options = (compiled_g.packet, alternate_g.packet)
        self.predictor_state = predictor_state
        self.g_labels = compiled_g.decoded.labels
        self.target = compiled_g.decoded.labels.copy()
        if target_differs:
            self.target[0, CELL_ROW, CELL_COL] = np.uint8((int(predictor[0, CELL_ROW, CELL_COL]) + 1) % 5)
        binding = receiver.semantic_binding(predictor)
        state = CoupledWitnessState.empty(
            _frozen_space(),
            generation_seed=0,
            generation_rng_id="numpy-pcg64-derived.v1",
        )
        winner = int(self.target[0, CELL_ROW, CELL_COL])
        margins = [0.25] * 5
        margins[winner] = 0.0
        cells = (
            Frame1CellObligation(
                pair_id=0,
                row=CELL_ROW,
                col=CELL_COL,
                winner_class_id=winner,
                required_margin_by_class=tuple(margins),  # type: ignore[arg-type]
                collateral_owner=CollateralOwner.CELL_VALUE_PREIMAGE,
            ),
        )
        fibres = (
            ConditionalFrame0PoseFibreObligation(
                pair_id=0,
                target_pose6=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                conditioned_frame1_obligation_sha256=frame1_pair_obligation_sha256(cells, 0),
            ),
        )
        ir = EvaluatorObligationIR(
            frozen_space=state.frozen_space,
            coupled_state_sha256=state.state_sha256,
            predictor_state_sha256=receiver.program_sha256,
            predictor_semantic_sha256=binding["predictor_semantic_sha256"],
            camera_height=768,
            camera_width=1024,
            frame1_cells=cells,
            conditional_frame0_pose_fibres=fibres,
        )
        y0, y1 = _generated_planes(
            COMPACT_SEED,
            g_labels=self.g_labels,
            frame0_gain=1,
        )
        result = _explicit_result(ir, y0, y1)
        self.population = PairPopulation.derive(
            source_pair_ids=(0,),
            v9_local_to_source_pair_ids=(0,),
            pbr_local_to_source_pair_ids=(0,),
            ir_local_to_source_pair_ids=(0,),
            v10_local_to_source_pair_ids=(0,),
        )
        self.join = ReopenedObjectJoin.reopen(
            state_bytes=state.to_bytes(),
            predictor_program_bytes=self.program,
            obligation_ir_bytes=ir.to_bytes(),
            explicit_preimage_result_bytes=result.to_bytes(),
            pair_population=self.population,
        )
        self.pbr2 = build_progressive_geometry_residual(
            predictor_program=self.program,
            predictor_contract_id=PREDICTOR_CONTRACT_ID,
            predictor_renderer_sha256=receiver.source_manifest_sha256,
            predictor_labels=predictor,
            target_labels=self.target,
            source_pair_ids=(0,),
            target_semantic_lineage="synthetic_fixture",
        )


@pytest.fixture(scope="module")
def complete_bundle() -> _Bundle:
    return _Bundle(target_differs=False)


@pytest.fixture(scope="module")
def sparse_bundle() -> _Bundle:
    return _Bundle(target_differs=True)


def _frame1_section(seed: int = COMPACT_SEED) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "test.frame1_preimage_parameters.v1",
            "seed": seed,
            "description": "original-own dense_y0 camera_preimage parameters are counted and typed",
        }
    )


def _frame0_section(gain: int = 1) -> bytes:
    return canonical_json_bytes({"schema": "test.frame0_pose_residual_parameters.v1", "gain": gain})


def _terminal_section(bias: int = 0) -> bytes:
    return canonical_json_bytes({"schema": "test.terminal_quotient_parameters.v1", "bias": bias})


def _compact_program_bytes(
    bundle: _Bundle,
    *,
    seed: int = COMPACT_SEED,
    include_terminal: bool = False,
) -> bytes:
    terminal = _terminal_section() if include_terminal else b""
    return bundle.g_program + _frame1_section(seed) + _frame0_section() + terminal


def _receiver_source_bytes() -> bytes:
    return Path(__file__).read_bytes()


def _section_provenance(
    role: CountedSectionRole,
    *,
    producer_source_sha256: str,
    derivation_input_sha256: str,
) -> CountedPayloadProvenance:
    artifact_classes = {
        CountedSectionRole.GENERATIVE_CORRECTION: CountedArtifactClass.GENERATOR_PARAMETERS,
        CountedSectionRole.FRAME1_PREIMAGE: CountedArtifactClass.FRAME_PREIMAGE_PARAMETERS,
        CountedSectionRole.FRAME0_FROM_EXACT_Y1: CountedArtifactClass.COUPLED_PREIMAGE_PARAMETERS,
        CountedSectionRole.FRAME0_POSE_RESIDUAL: CountedArtifactClass.POSE_RESIDUAL_PARAMETERS,
        CountedSectionRole.TERMINAL_QUOTIENT: CountedArtifactClass.TERMINAL_QUOTIENT,
    }
    return CountedPayloadProvenance(
        artifact_class=artifact_classes[role],
        lineage=CountedPayloadLineage.ORIGINAL_OWN,
        producer_id="test.algorithmic-y0-y1-generator",
        producer_source_sha256=producer_source_sha256,
        derivation_input_sha256=derivation_input_sha256,
        video_derived=True,
    )


def _compact_receiver(bundle: _Bundle):
    g_size = len(bundle.g_program)
    g_options = bundle.g_program_options
    predictor_state = bundle.predictor_state
    predictor_pose6_sha256 = bundle.join.predictor_pose6_sha256

    def receive(program: bytes) -> CompactGeneratorDecode:
        frame1_size = len(_frame1_section())
        frame0_size = len(_frame0_section())
        terminal_size = len(_terminal_section())
        base_size = g_size + frame1_size + frame0_size
        if len(program) not in (base_size, base_size + terminal_size):
            raise PairPopulationEnvelopeError("test compact generator byte geometry differs")
        g_stop = g_size
        frame1_stop = g_stop + frame1_size
        frame0_stop = frame1_stop + frame0_size
        g_payload = program[:g_stop]
        g_decoded = apply_generative_taskspace_correction(g_payload, predictor_state=predictor_state)
        frame1_payload = program[g_stop:frame1_stop]
        frame1_body = decode_canonical_json(frame1_payload)
        if (
            set(frame1_body) != {"schema", "seed", "description"}
            or frame1_body["schema"] != "test.frame1_preimage_parameters.v1"
            or frame1_body["description"] != "original-own dense_y0 camera_preimage parameters are counted and typed"
            or canonical_json_bytes(frame1_body) != frame1_payload
        ):
            raise PairPopulationEnvelopeError("test frame1 generator grammar differs")
        if type(frame1_body["seed"]) is not int:
            raise PairPopulationEnvelopeError("test compact generator seed differs")
        frame0_payload = program[frame1_stop:frame0_stop]
        frame0_body = decode_canonical_json(frame0_payload)
        if (
            set(frame0_body) != {"schema", "gain"}
            or frame0_body["schema"] != "test.frame0_pose_residual_parameters.v1"
            or type(frame0_body["gain"]) is not int
            or canonical_json_bytes(frame0_body) != frame0_payload
        ):
            raise PairPopulationEnvelopeError("test frame0 residual grammar differs")
        terminal_bias = 0
        section_payloads = [g_payload, frame1_payload, frame0_payload]
        roles = [
            CountedSectionRole.GENERATIVE_CORRECTION,
            CountedSectionRole.FRAME1_PREIMAGE,
            CountedSectionRole.FRAME0_POSE_RESIDUAL,
        ]
        if len(program) > frame0_stop:
            terminal_payload = program[frame0_stop:]
            terminal_body = decode_canonical_json(terminal_payload)
            if (
                set(terminal_body) != {"schema", "bias"}
                or terminal_body["schema"] != "test.terminal_quotient_parameters.v1"
                or type(terminal_body["bias"]) is not int
                or canonical_json_bytes(terminal_body) != terminal_payload
            ):
                raise PairPopulationEnvelopeError("test terminal quotient grammar differs")
            terminal_bias = terminal_body["bias"]
            section_payloads.append(terminal_payload)
            roles.append(CountedSectionRole.TERMINAL_QUOTIENT)
        y0, y1 = _generated_planes(
            frame1_body["seed"],
            g_labels=g_decoded.labels,
            frame0_gain=frame0_body["gain"],
            terminal_bias=terminal_bias,
        )
        producer_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        sections_list: list[CountedProgramSection] = []
        cursor = 0
        for role, payload in zip(roles, section_payloads, strict=True):
            sections_list.append(
                CountedProgramSection(
                    role,
                    cursor,
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                    _section_provenance(
                        role,
                        producer_source_sha256=producer_source_sha256,
                        derivation_input_sha256=predictor_state.binding_sha256,
                    ),
                )
            )
            cursor += len(payload)
        sections = tuple(sections_list)

        replacement_payloads = {
            CountedSectionRole.GENERATIVE_CORRECTION: g_options[0] if g_payload != g_options[0] else g_options[1],
            CountedSectionRole.FRAME1_PREIMAGE: _frame1_section(
                COMPACT_SEED + 1 if frame1_body["seed"] == COMPACT_SEED else COMPACT_SEED
            ),
            CountedSectionRole.FRAME0_POSE_RESIDUAL: _frame0_section(2 if frame0_body["gain"] == 1 else 1),
            CountedSectionRole.TERMINAL_QUOTIENT: _terminal_section(1 if terminal_bias == 0 else 0),
        }
        counterfactuals = tuple(
            RoleCounterfactualProgram(
                role,
                b"".join(
                    replacement_payloads[section_role] if section_role is role else payload
                    for section_role, payload in zip(roles, section_payloads, strict=True)
                ),
            )
            for role in roles
        )
        return CompactGeneratorDecode(
            decoder_contract_id="test.algorithmic-y0-y1-generator.v1",
            batches=(CompactGeneratorBatch((0,), y0, y1),),
            program_roles=tuple(CompactProgramRole),
            predictor_pose6_sha256=predictor_pose6_sha256,
            conditioned_frame1_identity_sha256=_array_identity_sha256(y1),
            consumed_program_sha256=hashlib.sha256(program).hexdigest(),
            section_manifest=sections,
            role_counterfactuals=counterfactuals,
        )

    return receive


def _reverse_program_and_sections(
    bundle: _Bundle,
) -> tuple[bytes, tuple[CountedProgramSection, ...]]:
    reverse_a = canonical_json_bytes({"schema": "test.reverse_causal_A.v1", "control": 1})
    payloads = (bundle.g_program, reverse_a)
    roles = (
        CountedSectionRole.GENERATIVE_CORRECTION,
        CountedSectionRole.FRAME0_FROM_EXACT_Y1,
    )
    producer_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sections: list[CountedProgramSection] = []
    cursor = 0
    for role, payload in zip(roles, payloads, strict=True):
        sections.append(
            CountedProgramSection(
                role=role,
                offset=cursor,
                byte_length=len(payload),
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                provenance=_section_provenance(
                    role,
                    producer_source_sha256=producer_source_sha256,
                    derivation_input_sha256=bundle.predictor_state.binding_sha256,
                ),
            )
        )
        cursor += len(payload)
    return b"".join(payloads), tuple(sections)


def test_legacy_and_reverse_causal_counted_grammars_are_disjoint(
    complete_bundle: _Bundle,
) -> None:
    legacy_program = _compact_program_bytes(complete_bundle)
    legacy_sections = _compact_receiver(complete_bundle)(legacy_program).section_manifest
    reverse_program, reverse_sections = _reverse_program_and_sections(complete_bundle)

    validate_counted_program_sections(legacy_program, legacy_sections)
    validate_reverse_causal_counted_program_sections(reverse_program, reverse_sections)

    with pytest.raises(PairPopulationEnvelopeError, match="legacy G/frame1/frame0-pose"):
        validate_counted_program_sections(reverse_program, reverse_sections)
    with pytest.raises(PairPopulationEnvelopeError, match="G/frame0-from-exact-Y1"):
        validate_reverse_causal_counted_program_sections(legacy_program, legacy_sections)


def test_legacy_compact_seal_rejects_reverse_causal_a_manifest(
    complete_bundle: _Bundle,
) -> None:
    reverse_program, reverse_sections = _reverse_program_and_sections(complete_bundle)
    legacy_program = _compact_program_bytes(complete_bundle)
    legacy_decoded = _compact_receiver(complete_bundle)(legacy_program)

    def reverse_role_receiver(program: bytes) -> CompactGeneratorDecode:
        if program != reverse_program:
            raise PairPopulationEnvelopeError("reverse-role receiver got foreign bytes")
        return replace(
            legacy_decoded,
            consumed_program_sha256=hashlib.sha256(program).hexdigest(),
            section_manifest=reverse_sections,
            role_counterfactuals=tuple(
                RoleCounterfactualProgram(section.role, program) for section in reverse_sections
            ),
        )

    with pytest.raises(PairPopulationEnvelopeError, match="legacy G/frame1/frame0-pose"):
        CompactObligationGeneratorProgram.seal(
            reverse_program,
            join=complete_bundle.join,
            receiver=reverse_role_receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )


class _TypedPairs:
    def __init__(self, body: dict[str, Any]) -> None:
        self.pair_ids = tuple(body["pair_ids"])
        self._body = body

    def model_dump(self, *, mode: str, by_alias: bool) -> dict[str, Any]:
        assert mode == "json" and by_alias is True
        return dict(self._body)


def test_pair_population_derives_arbitrary_windows_and_canonical_hash() -> None:
    population = PairPopulation.derive(
        source_pair_ids=(448, 472, 496, 511),
        v9_local_to_source_pair_ids=range(448, 512),
        pbr_local_to_source_pair_ids=range(448, 512),
        ir_local_to_source_pair_ids=range(600),
        v10_local_to_source_pair_ids=range(600),
    )
    rows = {row.source_pair_id: row for row in population.rows}
    assert (rows[448].v9_local_pair_id, rows[448].pbr_local_pair_id) == (0, 0)
    assert (rows[472].v9_local_pair_id, rows[472].ir_local_pair_id) == (24, 472)
    assert (rows[496].v9_local_pair_id, rows[496].v10_local_pair_id) == (48, 496)
    assert (rows[511].v9_local_pair_id, rows[511].ir_local_pair_id) == (63, 511)
    assert PairPopulation.from_bytes(population.to_bytes()) == population
    assert PairPopulation.from_bytes(population.to_bytes()).population_sha256 == population.population_sha256

    arbitrary = PairPopulation.derive(
        source_pair_ids=(5, 7),
        v9_local_to_source_pair_ids=(10, 5, 7),
        pbr_local_to_source_pair_ids=(7, 10, 5),
        ir_local_to_source_pair_ids=(7, 5, 10),
        v10_local_to_source_pair_ids=(10, 7, 5),
    )
    assert arbitrary.rows[0].v9_local_pair_id == 1
    assert arbitrary.rows[0].ir_local_pair_id == 1
    assert arbitrary.rows[1].pbr_local_pair_id == 0
    assert arbitrary.rows[1].v10_local_pair_id == 1


def test_pair_population_refuses_caller_attested_row_mutation() -> None:
    population = PairPopulation.derive(
        source_pair_ids=(448,),
        v9_local_to_source_pair_ids=range(448, 512),
        pbr_local_to_source_pair_ids=range(448, 512),
        ir_local_to_source_pair_ids=range(600),
        v10_local_to_source_pair_ids=range(600),
    )
    body = population.as_dict()
    body["rows"][0]["v9_local_pair_id"] = 448
    with pytest.raises(PairPopulationEnvelopeError, match="not derived"):
        PairPopulation.from_dict(body)
    with pytest.raises(PairPopulationEnvelopeError, match="outside"):
        PairPopulation.derive(
            source_pair_ids=(600,),
            v9_local_to_source_pair_ids=range(448, 512),
            pbr_local_to_source_pair_ids=range(448, 512),
            ir_local_to_source_pair_ids=range(600),
            v10_local_to_source_pair_ids=range(600),
        )

    with pytest.raises(PairPopulationEnvelopeError, match="frozen canonical contiguous pair order"):
        _require_canonical_frozen_pair_order((1, 0), (1, 0), pair_count=2)


def test_real_pbr2_window_and_v19c_typed_pair_ids_remain_encoder_only_foreign_keys() -> None:
    root = Path(__file__).resolve().parents[4]
    pbr2_path = (
        root / ".omx/research/original_taskspace_inverse_witness_codec_20260725/c0b_pbr2_progressive_geometry_n64.pbr2"
    )
    pbr2 = pbr2_path.read_bytes()
    pbr_reference = reopen_pbr2_pair_reference(pbr2)
    assert pbr_reference.source_pair_ids == tuple(range(448, 512))
    assert pbr_reference.artifact_sha256 == "3372eee1d989012fb3293c7abe08eac233c874bf485e5ea15c5bd26d7306f0a1"
    assert pbr_reference.candidate_payload_allowed is False

    config_path = root / ".omx/research/configs/ddm_v19_pure_priced_objective_20260723.json"
    config = config_path.read_bytes()

    def reopen_config(payload: bytes) -> _TypedPairs:
        body = json.loads(payload)
        return _TypedPairs(body)

    v19_reference = reopen_typed_pair_reference(
        config,
        role="v19c_typed_pair_ids",
        reopener=reopen_config,
    )
    assert v19_reference.source_pair_ids == V19C_TYPED_PAIR_IDS
    assert set(v19_reference.source_pair_ids).issubset(range(600))
    assert v19_reference.candidate_payload_allowed is False

    with pytest.raises(PairPopulationEnvelopeError, match="differs from the exact artifact body"):
        reopen_typed_pair_reference(
            config,
            role="v19c_typed_pair_ids",
            reopener=lambda payload: _TypedPairs(
                {**json.loads(payload), "pair_ids": list(reversed(V19C_TYPED_PAIR_IDS))}
            ),
        )


def test_reopened_join_derives_state_predictor_and_preimage_foreign_keys(complete_bundle: _Bundle) -> None:
    join = complete_bundle.join
    identities = join.identity_dict()
    assert identities["coupled_state_sha256"] == join.state.state_sha256
    assert identities["predictor_program_sha256"] == hashlib.sha256(complete_bundle.program).hexdigest()
    assert identities["predictor_semantic_sha256"] == join.obligation_ir.predictor_semantic_sha256
    assert identities["predictor_pose6_sha256"] == join.predictor_pose6_sha256
    assert identities["encoder_only_dense_v10_packet_bytes"] == len(b"dense-v10-packet")

    drifted_ir = replace(join.obligation_ir, coupled_state_sha256="f" * 64)
    with pytest.raises(PairPopulationEnvelopeError, match="coupled-state identity"):
        ReopenedObjectJoin.reopen(
            state_bytes=join.state.to_bytes(),
            predictor_program_bytes=complete_bundle.program,
            obligation_ir_bytes=drifted_ir.to_bytes(),
            explicit_preimage_result_bytes=join.explicit_preimage.to_bytes(),
            pair_population=complete_bundle.population,
        )


def test_ir_coverage_requires_complete_or_exact_sparse_ownership(
    complete_bundle: _Bundle,
    sparse_bundle: _Bundle,
) -> None:
    complete_compact = CompactObligationGeneratorProgram.seal(
        _compact_program_bytes(complete_bundle),
        join=complete_bundle.join,
        receiver=_compact_receiver(complete_bundle),
        receiver_source_bytes=_receiver_source_bytes(),
    )
    complete = derive_ir_coverage(
        complete_bundle.join,
        pbr2_packet=complete_bundle.pbr2,
        compact_program=complete_compact,
        candidate_program=complete_bundle.g_program,
        policy=IRCoveragePolicy.COMPLETE,
    )
    assert complete.matched_count == complete.obligation_count == 1
    assert complete.sparse_ownership == ()
    serialized = canonical_json_bytes(complete.as_dict(complete_bundle.population))
    assert b"winner_class_id" not in serialized
    assert b"required_margin_by_class" not in serialized
    assert b'teacher_truth_serialized":false' in serialized

    sparse_compact = CompactObligationGeneratorProgram.seal(
        _compact_program_bytes(sparse_bundle),
        join=sparse_bundle.join,
        receiver=_compact_receiver(sparse_bundle),
        receiver_source_bytes=_receiver_source_bytes(),
    )
    owner = SparseObligationOwnership(
        0,
        CELL_ROW,
        CELL_COL,
        SparseDebtOwner.FRAME1_PREIMAGE,
        sparse_compact.section(CountedSectionRole.FRAME1_PREIMAGE).payload_sha256,
    )
    sparse = derive_ir_coverage(
        sparse_bundle.join,
        pbr2_packet=sparse_bundle.pbr2,
        compact_program=sparse_compact,
        candidate_program=sparse_bundle.g_program,
        policy=IRCoveragePolicy.SPARSE_OWNED,
        sparse_ownership=(owner,),
    )
    assert sparse.matched_count == 0
    assert sparse.sparse_ownership == (owner,)
    with pytest.raises(ValueError):
        SparseDebtOwner("generative_correction")

    with pytest.raises(PairPopulationEnvelopeError, match="exactly cover"):
        derive_ir_coverage(
            sparse_bundle.join,
            pbr2_packet=sparse_bundle.pbr2,
            compact_program=sparse_compact,
            candidate_program=sparse_bundle.g_program,
            policy=IRCoveragePolicy.SPARSE_OWNED,
        )
    with pytest.raises(PairPopulationEnvelopeError, match="exactly cover"):
        derive_ir_coverage(
            sparse_bundle.join,
            pbr2_packet=sparse_bundle.pbr2,
            compact_program=sparse_compact,
            candidate_program=sparse_bundle.g_program,
            policy=IRCoveragePolicy.SPARSE_OWNED,
            sparse_ownership=(
                SparseObligationOwnership(
                    0,
                    CELL_ROW,
                    CELL_COL + 1,
                    SparseDebtOwner.FRAME1_PREIMAGE,
                    sparse_compact.section(CountedSectionRole.FRAME1_PREIMAGE).payload_sha256,
                ),
            ),
        )
    with pytest.raises(PairPopulationEnvelopeError, match="not bound"):
        derive_ir_coverage(
            sparse_bundle.join,
            pbr2_packet=sparse_bundle.pbr2,
            compact_program=sparse_compact,
            candidate_program=sparse_bundle.g_program,
            policy=IRCoveragePolicy.SPARSE_OWNED,
            sparse_ownership=(
                SparseObligationOwnership(
                    0,
                    CELL_ROW,
                    CELL_COL,
                    SparseDebtOwner.FRAME1_PREIMAGE,
                    "f" * 64,
                ),
            ),
        )
    with pytest.raises(PairPopulationEnvelopeError, match="terminal_quotient section is absent"):
        derive_ir_coverage(
            sparse_bundle.join,
            pbr2_packet=sparse_bundle.pbr2,
            compact_program=sparse_compact,
            candidate_program=sparse_bundle.g_program,
            policy=IRCoveragePolicy.SPARSE_OWNED,
            sparse_ownership=(
                SparseObligationOwnership(
                    0,
                    CELL_ROW,
                    CELL_COL,
                    SparseDebtOwner.TERMINAL_QUOTIENT,
                    "f" * 64,
                ),
            ),
        )
    with pytest.raises(PairPopulationEnvelopeError, match="PBR teacher bytes"):
        derive_ir_coverage(
            complete_bundle.join,
            pbr2_packet=complete_bundle.pbr2,
            compact_program=complete_compact,
            candidate_program=complete_bundle.pbr2,
            policy=IRCoveragePolicy.COMPLETE,
        )


def test_compact_program_retains_bytes_reopens_and_rejects_dense_hash_only_or_noop(
    complete_bundle: _Bundle,
) -> None:
    join = complete_bundle.join
    receiver = _compact_receiver(complete_bundle)
    program_bytes = _compact_program_bytes(complete_bundle)
    compact = CompactObligationGeneratorProgram.seal(
        program_bytes,
        join=join,
        receiver=receiver,
        receiver_source_bytes=_receiver_source_bytes(),
    )
    assert compact.program_bytes == program_bytes
    assert compact.program_sha256 == hashlib.sha256(program_bytes).hexdigest()
    assert compact.generated_y0_identity_sha256 == join.explicit_preimage.scorer_y0_identity_sha256
    assert compact.generated_y1_identity_sha256 == join.explicit_preimage.scorer_y1_identity_sha256
    assert compact.receiver_binding.python_reference_mode_only is True
    assert compact.receiver_binding.closure_defaults_globals_and_transitive_imports_bound is False
    assert compact.receiver_binding.standalone_archive_runtime_authority is False
    causality = {receipt.role: receipt for receipt in compact.section_causality}
    assert causality[CountedSectionRole.GENERATIVE_CORRECTION].y1_changed is True
    assert causality[CountedSectionRole.FRAME1_PREIMAGE].y1_changed is True
    assert causality[CountedSectionRole.FRAME0_POSE_RESIDUAL].y0_changed is True
    assert causality[CountedSectionRole.FRAME0_POSE_RESIDUAL].y1_changed is False
    assert all(receipt.target_section_only_changed for receipt in compact.section_causality)
    assert b"dense_y0 camera_preimage" in program_bytes
    decoded_once = receiver(program_bytes)

    def byte_insensitive_receiver(_program: bytes) -> CompactGeneratorDecode:
        return decoded_once

    with pytest.raises(PairPopulationEnvelopeError, match="counterfactual receiver"):
        CompactObligationGeneratorProgram.seal(
            program_bytes,
            join=join,
            receiver=byte_insensitive_receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )

    def exception_counterfactual_receiver(program: bytes) -> CompactGeneratorDecode:
        if program == program_bytes:
            return receiver(program)
        raise PairPopulationEnvelopeError("typed counterfactual parser refused")

    with pytest.raises(PairPopulationEnvelopeError, match="receiver raised"):
        CompactObligationGeneratorProgram.seal(
            program_bytes,
            join=join,
            receiver=exception_counterfactual_receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )

    def cross_role_counterfactual_receiver(program: bytes) -> CompactGeneratorDecode:
        decoded = receiver(program)
        if program != program_bytes:
            return decoded
        bad_g_counterfactual = (
            complete_bundle.g_program_alternative + _frame1_section(COMPACT_SEED + 1) + _frame0_section()
        )
        return replace(
            decoded,
            role_counterfactuals=(
                RoleCounterfactualProgram(CountedSectionRole.GENERATIVE_CORRECTION, bad_g_counterfactual),
                *decoded.role_counterfactuals[1:],
            ),
        )

    with pytest.raises(PairPopulationEnvelopeError, match="changed non-target frame1_preimage bytes"):
        CompactObligationGeneratorProgram.seal(
            program_bytes,
            join=join,
            receiver=cross_role_counterfactual_receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )

    with pytest.raises(PairPopulationEnvelopeError, match="not bound to the supplied source artifact"):
        CompactObligationGeneratorProgram.seal(
            program_bytes,
            join=join,
            receiver=receive_factorized_v9_predictor,  # type: ignore[arg-type]
            receiver_source_bytes=_receiver_source_bytes(),
        )

    calls = 0

    def nondeterministic_receiver(program: bytes) -> CompactGeneratorDecode:
        nonlocal calls
        calls += 1
        decoded = receiver(program)
        if calls == 2:
            batch = decoded.batches[0]
            changed_y0 = batch.scorer_y0.copy()
            changed_y0[0, 0, 0, 0] ^= 1
            return replace(decoded, batches=(replace(batch, scorer_y0=changed_y0),))
        return decoded

    with pytest.raises(PairPopulationEnvelopeError, match="nondeterministic"):
        CompactObligationGeneratorProgram.seal(
            program_bytes,
            join=join,
            receiver=nondeterministic_receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )
    assert (
        CompactObligationGeneratorProgram.reopen(
            compact.to_bytes(),
            join=join,
            receiver=receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )
        == compact
    )

    with pytest.raises(PairPopulationEnvelopeError, match="exact program/receiver bytes"):
        CompactObligationGeneratorProgram.seal(
            b"", join=join, receiver=receiver, receiver_source_bytes=_receiver_source_bytes()
        )
    with pytest.raises(PairPopulationEnvelopeError, match="outputs differ"):
        CompactObligationGeneratorProgram.seal(
            _compact_program_bytes(complete_bundle, seed=COMPACT_SEED + 1),
            join=join,
            receiver=receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )
    with pytest.raises(PairPopulationEnvelopeError, match="hash-only"):
        CompactObligationGeneratorProgram.seal(
            b"dense-v10-packet", join=join, receiver=receiver, receiver_source_bytes=_receiver_source_bytes()
        )
    body = decode_canonical_json(compact.to_bytes())
    body["body"].pop("program_base64")
    body["body_sha256"] = canonical_sha256(body["body"])
    missing_program = canonical_json_bytes(body)
    with pytest.raises(PairPopulationEnvelopeError, match="bytes are absent"):
        CompactObligationGeneratorProgram.reopen(
            missing_program,
            join=join,
            receiver=receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )


def test_terminal_counterfactual_is_valid_and_changes_a_realized_plane(complete_bundle: _Bundle) -> None:
    compact = CompactObligationGeneratorProgram.seal(
        _compact_program_bytes(complete_bundle, include_terminal=True),
        join=complete_bundle.join,
        receiver=_compact_receiver(complete_bundle),
        receiver_source_bytes=_receiver_source_bytes(),
    )
    terminal = compact.section_causality[-1]
    assert terminal.role is CountedSectionRole.TERMINAL_QUOTIENT
    assert terminal.y0_changed or terminal.y1_changed
    assert terminal.receiver_exception_free is True


@pytest.mark.parametrize(
    "artifact_class",
    (CountedArtifactClass.DENSE_REALIZED_Y, CountedArtifactClass.CAMERA_PREIMAGE),
)
def test_counted_own_lineage_dense_and_preimage_classes_are_structurally_legal(
    artifact_class: CountedArtifactClass,
) -> None:
    payload = f"counted-{artifact_class.value}".encode("ascii")
    provenance = CountedPayloadProvenance(
        artifact_class=artifact_class,
        lineage=CountedPayloadLineage.ORIGINAL_OWN,
        producer_id="test.own-lineage-producer",
        producer_source_sha256="a" * 64,
        derivation_input_sha256="b" * 64,
        video_derived=True,
    )
    section = CountedProgramSection(
        CountedSectionRole.FRAME1_PREIMAGE,
        0,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        provenance,
    )
    assert section.provenance.artifact_class is artifact_class
    assert section.provenance.originality_claimed is False


@pytest.mark.parametrize(
    "artifact_class",
    (
        CountedArtifactClass.SCORER_MODEL,
        CountedArtifactClass.GROUND_TRUTH,
        CountedArtifactClass.GT_ARGMAX_TABLE,
        CountedArtifactClass.ENCODER_ONLY_TEACHER,
        CountedArtifactClass.ENCODER_ONLY_OBLIGATION_IR,
        CountedArtifactClass.ENCODER_ONLY_ORACLE_EVIDENCE,
        CountedArtifactClass.ENCODER_ONLY_EXPLICIT_PREIMAGE,
    ),
)
def test_exact_encoder_scorer_and_gt_artifact_classes_fail_closed(
    artifact_class: CountedArtifactClass,
) -> None:
    with pytest.raises(PairPopulationEnvelopeError, match="forbidden in counted candidate bytes"):
        CountedPayloadProvenance(
            artifact_class=artifact_class,
            lineage=CountedPayloadLineage.ORIGINAL_OWN,
            producer_id="test.forbidden-producer",
            producer_source_sha256="a" * 64,
            derivation_input_sha256="b" * 64,
            video_derived=True,
        )

    with pytest.raises(PairPopulationEnvelopeError, match="cannot claim originality"):
        CountedPayloadProvenance(
            artifact_class=CountedArtifactClass.GENERATOR_PARAMETERS,
            lineage=CountedPayloadLineage.ORIGINAL_OWN,
            producer_id="test.fake-originality-claim",
            producer_source_sha256="a" * 64,
            derivation_input_sha256="b" * 64,
            video_derived=True,
            originality_claimed=True,
        )


def test_pose_ownership_and_full_envelope_are_exclusive_and_teacher_free(complete_bundle: _Bundle) -> None:
    join = complete_bundle.join
    compact = CompactObligationGeneratorProgram.seal(
        _compact_program_bytes(complete_bundle),
        join=join,
        receiver=_compact_receiver(complete_bundle),
        receiver_source_bytes=_receiver_source_bytes(),
    )
    pose = ExclusivePoseOwnership.bind(join, compact)
    pbr_reference = reopen_pbr2_pair_reference(complete_bundle.pbr2)
    typed_payload = canonical_json_bytes({"schema": "test.v19c_typed_pairs.v1", "pair_ids": [0]})
    typed_reference = reopen_typed_pair_reference(
        typed_payload,
        role="v19c_typed_pair_ids",
        reopener=lambda payload: _TypedPairs(dict(decode_canonical_json(payload))),
    )
    coverage = derive_ir_coverage(
        join,
        pbr2_packet=complete_bundle.pbr2,
        compact_program=compact,
        candidate_program=complete_bundle.g_program,
        policy=IRCoveragePolicy.COMPLETE,
    )
    envelope = PairPopulationEnvelope(
        reopened=join,
        pbr2_reference=pbr_reference,
        v19c_typed_pairs_reference=typed_reference,
        ir_coverage=coverage,
        compact_program=compact,
        pose_ownership=pose,
    )
    payload = envelope.to_bytes()
    envelope.validate_serialized(payload)
    with pytest.raises(PairPopulationEnvelopeError, match="not bound to the counted G section"):
        PairPopulationEnvelope(
            reopened=join,
            pbr2_reference=pbr_reference,
            v19c_typed_pairs_reference=typed_reference,
            ir_coverage=replace(coverage, compact_program_sha256="f" * 64),
            compact_program=compact,
            pose_ownership=pose,
        )
    parsed = decode_canonical_json(payload)["body"]
    assert (
        pose.frame0_residual_program_sha256 == compact.section(CountedSectionRole.FRAME0_POSE_RESIDUAL).payload_sha256
    )
    assert parsed["exclusive_pose_ownership"]["absolute_pose_owner_count"] == 1
    assert parsed["exclusive_pose_ownership"]["frame0_residual_owner_count"] == 1
    assert parsed["exclusive_pose_ownership"]["duplicate_pose_payload_allowed"] is False
    assert all(value is False for value in parsed["payload_firewall"].values())
    assert parsed["candidate_payload_eligible"] is False
    assert complete_bundle.pbr2 not in payload
    assert join.obligation_ir.to_bytes() not in payload
    assert b"required_margin_by_class" not in payload
    assert b"target_pose6" not in payload
    assert b"program_base64" in payload

    mutated = bytearray(payload)
    mutated[-2] ^= 1
    with pytest.raises(PairPopulationEnvelopeError, match="canonical JSON"):
        envelope.validate_serialized(bytes(mutated))


def test_mutable_closure_is_explicitly_outside_reference_binding_authority(
    complete_bundle: _Bundle,
) -> None:
    receiver = _compact_receiver(complete_bundle)
    before = _bind_receiver_source(receiver, _receiver_source_bytes())
    freevars = receiver.__code__.co_freevars
    assert receiver.__closure__ is not None
    g_size_cell = receiver.__closure__[freevars.index("g_size")]
    g_size_cell.cell_contents += 1
    after = _bind_receiver_source(receiver, _receiver_source_bytes())
    assert after == before
    assert before.binding_scope == "direct_python_function_source_and_code_nontransitive.v1"
    assert before.closure_defaults_globals_and_transitive_imports_bound is False
    assert before.standalone_archive_runtime_authority is False


def test_compact_decode_refuses_absolute_pose_or_forbidden_artifact_claim(complete_bundle: _Bundle) -> None:
    join = complete_bundle.join
    program_bytes = _compact_program_bytes(complete_bundle)
    baseline_receiver = _compact_receiver(complete_bundle)

    def forbidden_receiver(program: bytes) -> CompactGeneratorDecode:
        return replace(baseline_receiver(program), absolute_pose6_values_present=True)

    with pytest.raises(PairPopulationEnvelopeError, match="duplicated pose"):
        CompactObligationGeneratorProgram.seal(
            program_bytes,
            join=join,
            receiver=forbidden_receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )

    def scorer_receiver(program: bytes) -> CompactGeneratorDecode:
        return replace(baseline_receiver(program), scorer_artifacts_present=True)

    with pytest.raises(PairPopulationEnvelopeError, match="scorer"):
        CompactObligationGeneratorProgram.seal(
            program_bytes,
            join=join,
            receiver=scorer_receiver,
            receiver_source_bytes=_receiver_source_bytes(),
        )
