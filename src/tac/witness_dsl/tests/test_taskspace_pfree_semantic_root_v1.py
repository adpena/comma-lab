# SPDX-License-Identifier: MIT
"""G103 strict wire/receiver proofs for the P-free semantic root.

These are source-backed behavior tests, not empirical score evidence.  In
particular they preserve the settled FEED-ah negative: direct labels/contours
plus a palette are not sufficient after R and jointly fail Seg/Pose/rate.  The
fixture therefore exercises counted learned RGB tensors and the temporal
latent in the receiver, and never re-runs the dominated palette experiment.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import numpy as np
import pytest

import tac.witness_dsl.taskspace_pfree_semantic_root_v1 as semantic_root_module
from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    EXPLICIT_GAUGE_ARBITRATION_BLOCKER,
    PAIR_COUNT_N600,
    PALETTE_ONLY_DIRECT_LABEL_BLOCKER,
    PUBLIC_RECEIVER_BLOCKER,
    SOURCE_BACKED_COMPILER_BLOCKER,
    V9_PHASE_ADVECTION_ADAPTER_BLOCKER,
    GeneratorActivationV1,
    GeneratorArchitectureV1,
    GeneratorNumericContractV1,
    PairRGBGaugeV1,
    QuantizedGeneratorTensorV1,
    QuantizedSharedGeneratorV1,
    QuantizedTensorDTypeV1,
    QuantizedTensorRoleV1,
    RGBGaugeOwnershipV1,
    RGBQuotientAtomV1,
    SemanticRealizationProfileV1,
    SemanticRoleV1,
    SemanticRootSourceLineageV1,
    SemanticRootY1V1,
    SemanticRootY1V1Error,
    SemanticTopologyEventV1,
    SemanticTopologyTemplateV1,
    TemporalLatentStreamV1,
    TopologyShapeV1,
    bind_semantic_root_source_lineage_to_g17,
    bind_semantic_root_to_g17,
    encode_semantic_root_source_lineage_manifest,
    encode_semantic_root_y1_v1,
    final_semantic_root_y1_binding_sha256,
    iter_semantic_root_y1_batches,
    parse_semantic_root_source_lineage_manifest,
    parse_semantic_root_y1_v1,
    quantized_shared_generator_section_sha256,
    realize_semantic_root_y1_v10_factor2,
    render_semantic_root_y1_scorer,
    semantic_root_g17_logical_values,
    semantic_root_y1_population_sha256,
    temporal_y1_latents_from_interleaved_v9_codes,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17EncoderOnlyTeacherOracleEvidenceV1,
    G17LogicalOwnershipKindV1,
    G17LogicalOwnershipV1,
    G17PairPopulationV1,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _i8(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype=np.int8).tobytes(order="C")


def _i16(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype=">i2").tobytes(order="C")


def _tensor(
    tensor_id: int,
    role: QuantizedTensorRoleV1,
    values: np.ndarray,
    *,
    dtype: QuantizedTensorDTypeV1,
) -> QuantizedGeneratorTensorV1:
    array = np.asarray(values)
    return QuantizedGeneratorTensorV1(
        tensor_id=tensor_id,
        role=role,
        dtype=dtype,
        shape=tuple(int(value) for value in array.shape),
        scale_exponent=-7 if dtype is QuantizedTensorDTypeV1.INT8 else -12,
        zero_point=0,
        data=_i8(array) if dtype is QuantizedTensorDTypeV1.INT8 else _i16(array),
    )


def _model() -> QuantizedSharedGeneratorV1:
    hidden = 8
    input_weight = np.array(
        [
            [72, 16, 8, 32],
            [-24, 68, 12, 24],
            [40, -32, 56, 16],
            [16, 28, -44, 64],
            [-64, 20, 32, 36],
            [28, -56, 20, 48],
            [36, 44, -20, 28],
            [-32, -24, 48, 56],
        ],
        dtype=np.int8,
    )
    input_bias = np.arange(1, hidden + 1, dtype=np.int16) * 64
    hidden_weight = np.eye(hidden, dtype=np.int8) * 72
    hidden_weight += np.roll(np.eye(hidden, dtype=np.int8) * 24, 1, axis=1)
    hidden_bias = np.arange(hidden, dtype=np.int16) * 48 - 96
    film_weight = np.empty((2 * hidden, 2), dtype=np.int8)
    film_weight[:, 0] = np.where(np.arange(2 * hidden) % 2 == 0, 48, -40)
    film_weight[:, 1] = np.where(np.arange(2 * hidden) % 3 == 0, 32, -24)
    film_bias = np.concatenate(
        (
            np.full(hidden, 384, dtype=np.int16),
            np.arange(hidden, dtype=np.int16) * 96 - 256,
        )
    )
    output_weight = np.array(
        [
            [96, 64, 48, 32, 24, 16, 8, 4],
            [-80, 72, -56, 48, -32, 24, -16, 8],
            [32, 48, 64, 80, -24, -40, -56, -72],
        ],
        dtype=np.int8,
    )
    output_bias = np.array([1536, -1024, 2048], dtype=np.int16)
    tensors = (
        _tensor(0, QuantizedTensorRoleV1.INPUT_WEIGHT, input_weight, dtype=QuantizedTensorDTypeV1.INT8),
        _tensor(1, QuantizedTensorRoleV1.INPUT_BIAS, input_bias, dtype=QuantizedTensorDTypeV1.INT16_BE),
        _tensor(2, QuantizedTensorRoleV1.HIDDEN_WEIGHT, hidden_weight, dtype=QuantizedTensorDTypeV1.INT8),
        _tensor(3, QuantizedTensorRoleV1.HIDDEN_BIAS, hidden_bias, dtype=QuantizedTensorDTypeV1.INT16_BE),
        _tensor(4, QuantizedTensorRoleV1.FILM_WEIGHT, film_weight, dtype=QuantizedTensorDTypeV1.INT8),
        _tensor(5, QuantizedTensorRoleV1.FILM_BIAS, film_bias, dtype=QuantizedTensorDTypeV1.INT16_BE),
        _tensor(6, QuantizedTensorRoleV1.OUTPUT_WEIGHT, output_weight, dtype=QuantizedTensorDTypeV1.INT8),
        _tensor(7, QuantizedTensorRoleV1.OUTPUT_BIAS, output_bias, dtype=QuantizedTensorDTypeV1.INT16_BE),
    )
    return QuantizedSharedGeneratorV1(
        architecture=GeneratorArchitectureV1.ORIGINAL_COORDINR_FILM_MLP_V1,
        numeric_contract=GeneratorNumericContractV1.INT8_WEIGHT_INT16_STATE_INT32_ACCUM_Q12,
        activation=GeneratorActivationV1.HARD_TANH_Q12,
        input_dim=4,
        hidden_dim=hidden,
        hidden_layer_count=1,
        modulation_dim=2,
        tensors=tensors,
    )


def _latents(*, pair0_delta: int = 0) -> TemporalLatentStreamV1:
    pair_ids = np.arange(PAIR_COUNT_N600, dtype=np.int32)
    values = np.stack(
        (
            (pair_ids % 31) * 64 - 960,
            ((pair_ids * 3) % 29) * 64 - 896,
        ),
        axis=1,
    )
    values[0, 0] += pair0_delta
    return TemporalLatentStreamV1.from_array(values, rice_k=7)


def _root(
    *,
    model: QuantizedSharedGeneratorV1 | None = None,
    latents: TemporalLatentStreamV1 | None = None,
    ownership: RGBGaugeOwnershipV1 = RGBGaugeOwnershipV1.DERIVED_BY_SHARED_GENERATOR,
    gauges: tuple[PairRGBGaugeV1, ...] = (),
) -> SemanticRootY1V1:
    model = _model() if model is None else model
    latents = _latents() if latents is None else latents
    return SemanticRootY1V1(
        background_role=SemanticRoleV1.ROAD,
        profile=SemanticRealizationProfileV1(
            role_rgb=(
                (96, 92, 88),
                (180, 168, 72),
                (48, 56, 68),
                (140, 96, 72),
                (72, 92, 128),
            ),
            texture_gain_q4=16,
            edge_gain_q4=16,
            chroma_gain_q4=16,
            parallax_gain_q4=16,
            renderer_seed=0x10203,
        ),
        shared_generator=model,
        temporal_latents=latents,
        rgb_gauge_ownership=ownership,
        topology_templates=(
            SemanticTopologyTemplateV1(
                template_id=0,
                role=SemanticRoleV1.LANE,
                shape=TopologyShapeV1.QUADRATIC_STRIP,
                params_q=(0, 0, 0, 1, 48, 6_000),
            ),
        ),
        topology_events=(
            SemanticTopologyEventV1(
                event_id=0,
                template_id=0,
                pair_start=0,
                pair_stop=PAIR_COUNT_N600,
                z_order=1,
                anchor_x_q4=4_096,
                anchor_y_q4=3_072,
                velocity_x_q8=1,
                velocity_y_q8=0,
            ),
        ),
        rgb_basis=(),
        pair_rgb_gauges=gauges,
        irreducible_rgb_quotient=(
            RGBQuotientAtomV1(
                atom_id=0,
                pair_start=0,
                pair_stop=PAIR_COUNT_N600,
                role_mask=0b1_1111,
                edge_only=False,
                center_x_q4=4_096,
                center_y_q4=3_072,
                velocity_x_q8=0,
                velocity_y_q8=0,
                radius_x_q4=512,
                radius_y_q4=384,
                amplitude_rgb=(3, -2, 4),
            ),
        ),
    )


def _replace_tensor(root: SemanticRootY1V1, tensor_index: int) -> SemanticRootY1V1:
    tensor = root.shared_generator.tensors[tensor_index]
    array = tensor.array.copy()
    if tensor.dtype is QuantizedTensorDTypeV1.INT8:
        array = np.clip(-array + 7, -128, 127).astype(np.int8)
    else:
        array = np.clip(array + 4_096, -0x8000, 0x7FFF).astype(np.int16)
    changed = _tensor(
        tensor.tensor_id,
        tensor.role,
        array,
        dtype=tensor.dtype,
    )
    tensors = list(root.shared_generator.tensors)
    tensors[tensor_index] = changed
    model = replace(root.shared_generator, tensors=tuple(tensors))
    return replace(root, shared_generator=model)


def test_counted_packet_is_canonical_closed_and_reemittable() -> None:
    root = _root()
    packet = encode_semantic_root_y1_v1(root)
    parsed = parse_semantic_root_y1_v1(packet)
    assert parsed == root
    assert encode_semantic_root_y1_v1(parsed) == packet
    assert parsed.packet_sha256 == hashlib.sha256(packet).hexdigest()
    assert len(packet) < 100_000


def test_unknown_tampered_trailing_foreign_and_dense_payloads_fail_closed() -> None:
    packet = encode_semantic_root_y1_v1(_root())
    unknown = packet.replace(b"PROF", b"NOPE", 1)
    with pytest.raises(SemanticRootY1V1Error, match="section tags"):
        parse_semantic_root_y1_v1(unknown)
    tampered = bytearray(packet)
    tampered[-5] ^= 1
    with pytest.raises(SemanticRootY1V1Error, match="CRC32"):
        parse_semantic_root_y1_v1(bytes(tampered))
    with pytest.raises(SemanticRootY1V1Error, match="trailing bytes"):
        parse_semantic_root_y1_v1(packet + b"PK\x03\x04")
    with pytest.raises(SemanticRootY1V1Error, match="trailing bytes"):
        parse_semantic_root_y1_v1(packet + bytes(600 * 384 * 3))
    with pytest.raises(SemanticRootY1V1Error, match="dense scorer/raster"):
        QuantizedGeneratorTensorV1(
            tensor_id=0,
            role=QuantizedTensorRoleV1.INPUT_WEIGHT,
            dtype=QuantizedTensorDTypeV1.INT8,
            shape=(384, 512, 3),
            scale_exponent=-7,
            zero_point=0,
            data=b"x",
        )
    with pytest.raises(SemanticRootY1V1Error, match="foreign/raster"):
        QuantizedGeneratorTensorV1(
            tensor_id=0,
            role=QuantizedTensorRoleV1.INPUT_BIAS,
            dtype=QuantizedTensorDTypeV1.INT8,
            shape=(7,),
            scale_exponent=-7,
            zero_point=0,
            data=b"DDV15S1",
        )


def test_palette_only_and_dead_learned_operand_forms_are_unrepresentable() -> None:
    root = _root()
    with pytest.raises(SemanticRootY1V1Error, match="shared_generator"):
        replace(root, shared_generator=None)  # type: ignore[arg-type]
    zero_tensors = tuple(replace(tensor, data=bytes(len(tensor.data))) for tensor in root.shared_generator.tensors)
    with pytest.raises(SemanticRootY1V1Error, match="all-zero shared generator"):
        replace(root.shared_generator, tensors=zero_tensors)
    assert PALETTE_ONLY_DIRECT_LABEL_BLOCKER in semantic_root_module.PALETTE_ONLY_DIRECT_LABEL_BLOCKER
    field_names = {item.name.lower() for item in fields(SemanticRootY1V1)}
    assert not field_names & {
        "semantic_p",
        "pvsa",
        "dense_plane",
        "raster_bytes",
        "v15_payload",
        "g57_y1",
    }


def test_learned_model_every_tensor_and_temporal_latent_change_rgb_output() -> None:
    root = _root()
    baseline = render_semantic_root_y1_scorer(root, 0)
    assert baseline.shape == (384, 512, 3)
    assert baseline.dtype == np.uint8
    palette = np.asarray(root.profile.role_rgb, dtype=np.uint8)
    assert np.any(np.all(baseline[:, :, None, :] != palette[None, None, :, :], axis=3))
    for tensor_index in range(len(root.shared_generator.tensors)):
        changed = render_semantic_root_y1_scorer(
            _replace_tensor(root, tensor_index),
            0,
        )
        assert not np.array_equal(changed, baseline), tensor_index
    changed_latents = _latents(pair0_delta=2_048)
    changed_root = replace(root, temporal_latents=changed_latents)
    assert not np.array_equal(render_semantic_root_y1_scorer(changed_root, 0), baseline)


def test_topology_is_optional_factor_not_claimed_solution() -> None:
    root = replace(_root(), topology_templates=(), topology_events=())
    frame = render_semantic_root_y1_scorer(root, 9)
    assert frame.shape == (384, 512, 3)
    assert encode_semantic_root_y1_v1(root)


def test_temporal_stream_owns_default_gauge_and_explicit_overlap_blocks_g17() -> None:
    root = _root()
    assert root.rgb_gauge_ownership is RGBGaugeOwnershipV1.DERIVED_BY_SHARED_GENERATOR
    assert root.pair_rgb_gauges == ()
    explicit_gauges = tuple(
        PairRGBGaugeV1(
            pair_id=pair_id,
            phase_u16=0,
            parallax_x_q8=0,
            parallax_y_q8=0,
            luma_bias=0,
            chroma_u_bias=0,
            chroma_v_bias=0,
            texture_gain_q8=256,
        )
        for pair_id in range(PAIR_COUNT_N600)
    )
    explicit = _root(
        ownership=RGBGaugeOwnershipV1.EXPLICIT_NONOVERLAPPING_POST_GENERATOR,
        gauges=explicit_gauges,
    )
    values = semantic_root_g17_logical_values(explicit)
    population = _population()
    owners = _owners(values)
    with pytest.raises(SemanticRootY1V1Error, match=EXPLICIT_GAUGE_ARBITRATION_BLOCKER):
        bind_semantic_root_to_g17(
            explicit,
            population=population,
            topology_owner=owners[0],
            realization_owner=owners[1],
            generator_owner=owners[2],
            temporal_owner=owners[3],
            quotient_owner=owners[4],
        )


def _population() -> G17PairPopulationV1:
    coordinates = tuple(range(PAIR_COUNT_N600))
    return G17PairPopulationV1(
        global_pair_ids=coordinates,
        source_pair_ids=coordinates,
        v9_pair_coordinates=coordinates,
        pbr_pair_coordinates=coordinates,
        obligation_ir_coordinates=coordinates,
        v10_local_coordinates=coordinates,
    )


def _owners(
    values: tuple[object, ...],
) -> tuple[G17LogicalOwnershipV1, ...]:
    kinds = (
        G17LogicalOwnershipKindV1.SEMANTIC_TOPOLOGY,
        G17LogicalOwnershipKindV1.REALIZATION_GAUGE,
        G17LogicalOwnershipKindV1.LEARNED_RESIDUAL,
        G17LogicalOwnershipKindV1.POPULATION_SHARED,
        G17LogicalOwnershipKindV1.LEARNED_RESIDUAL,
    )
    return tuple(
        G17LogicalOwnershipV1(
            owner_id=f"g103-owner-{index}",
            ownership_kind=kind,
            value=value,  # type: ignore[arg-type]
        )
        for index, (kind, value) in enumerate(zip(kinds, values, strict=True))
    )


def test_g17_is_single_population_ownership_and_lifecycle_authority() -> None:
    root = _root()
    population = _population()
    values = semantic_root_g17_logical_values(root)
    owners = _owners(values)
    binding = bind_semantic_root_to_g17(
        root,
        population=population,
        topology_owner=owners[0],
        realization_owner=owners[1],
        generator_owner=owners[2],
        temporal_owner=owners[3],
        quotient_owner=owners[4],
    )
    assert len(binding) == 64
    with pytest.raises(SemanticRootY1V1Error, match="exactly retained"):
        bind_semantic_root_to_g17(
            root,
            population=population,
            topology_owner=owners[0],
            realization_owner=owners[0],
            generator_owner=owners[2],
            temporal_owner=owners[3],
            quotient_owner=owners[4],
        )


def test_source_lineage_is_external_packet_bound_g17_encoder_evidence() -> None:
    root = _root()
    lineage = SemanticRootSourceLineageV1(
        compiler_id="G103",
        root_packet_sha256=root.packet_sha256,
        source_video_sha256=_sha("fresh-source-video"),
        target_custody_sha256=_sha("fresh-target-custody"),
        compiler_source_sha256=_sha("fresh-compiler-source"),
        compile_config_sha256=_sha("fresh-compile-config"),
        originality_declaration_sha256=_sha("fresh-own-lineage"),
        model_section_sha256=quantized_shared_generator_section_sha256(root.shared_generator),
        latent_decoded_sha256=root.temporal_latents.decoded_sha256,
    )
    manifest = encode_semantic_root_source_lineage_manifest(root, lineage)
    assert manifest not in encode_semantic_root_y1_v1(root)
    assert parse_semantic_root_source_lineage_manifest(manifest) == lineage
    owner = G17LogicalOwnershipV1(
        owner_id="g103-external-lineage",
        ownership_kind=G17LogicalOwnershipKindV1.ENCODER_EVIDENCE,
        value=G17EncoderOnlyTeacherOracleEvidenceV1(manifest),
    )
    assert (
        len(
            bind_semantic_root_source_lineage_to_g17(
                root,
                lineage,
                lineage_owner=owner,
            )
        )
        == 64
    )
    with pytest.raises(SemanticRootY1V1Error, match="does not bind"):
        encode_semantic_root_source_lineage_manifest(
            root,
            replace(lineage, root_packet_sha256=_sha("another-packet")),
        )


def test_full_ordered_n600_batch_and_population_hash_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root()
    observed: list[int] = []

    def bounded_behavior_renderer(_root: SemanticRootY1V1, pair_id: int) -> np.ndarray:
        observed.append(pair_id)
        return np.array([[[pair_id >> 8, pair_id & 0xFF, 103]]], dtype=np.uint8)

    monkeypatch.setattr(
        semantic_root_module,
        "render_semantic_root_y1_scorer",
        bounded_behavior_renderer,
    )
    starts = []
    decoded_ids = []
    for start, batch in iter_semantic_root_y1_batches(root, batch_size=16):
        starts.append(start)
        decoded_ids.extend((int(frame[0, 0, 0]) << 8) | int(frame[0, 0, 1]) for frame in batch)
    assert starts == list(range(0, PAIR_COUNT_N600, 16))
    assert decoded_ids == list(range(PAIR_COUNT_N600))
    assert observed == list(range(PAIR_COUNT_N600))
    observed.clear()
    population_sha = semantic_root_y1_population_sha256(root, batch_size=16)
    assert len(population_sha) == 64
    assert observed == list(range(PAIR_COUNT_N600))


def test_v10_factor2_realization_and_final_y1_binding() -> None:
    root = _root()
    camera, proof = realize_semantic_root_y1_v10_factor2(root, 0)
    assert camera.shape == (874, 1164, 3)
    assert camera.dtype == np.uint8
    assert proof.certified_exact
    population = _population()
    binding = final_semantic_root_y1_binding_sha256(
        root_packet_sha256=root.packet_sha256,
        g17_population_binding_sha256=population.binding_sha256,
        scorer_y1_population_sha256=_sha("decoded-scorer-y1-population"),
    )
    changed = final_semantic_root_y1_binding_sha256(
        root_packet_sha256=_replace_tensor(root, 0).packet_sha256,
        g17_population_binding_sha256=population.binding_sha256,
        scorer_y1_population_sha256=_sha("changed-decoded-scorer-y1-population"),
    )
    assert binding != changed


def test_open_product_blockers_are_explicit_and_v9_is_not_cross_cast() -> None:
    assert SOURCE_BACKED_COMPILER_BLOCKER in semantic_root_module.OPEN_PRODUCT_BLOCKERS
    assert PUBLIC_RECEIVER_BLOCKER in semantic_root_module.OPEN_PRODUCT_BLOCKERS
    assert V9_PHASE_ADVECTION_ADAPTER_BLOCKER in semantic_root_module.OPEN_PRODUCT_BLOCKERS
    assert "ORIGINAL" in GeneratorArchitectureV1.ORIGINAL_COORDINR_FILM_MLP_V1.name


def test_v9_interleaved_projection_counts_only_odd_y1_and_never_y0() -> None:
    codes = np.zeros((2 * PAIR_COUNT_N600, 2), dtype=np.int16)
    codes[1::2, 0] = np.arange(PAIR_COUNT_N600, dtype=np.int16)
    baseline = temporal_y1_latents_from_interleaved_v9_codes(codes, rice_k=4)
    even_changed = codes.copy()
    even_changed[0::2] = 777
    assert (
        temporal_y1_latents_from_interleaved_v9_codes(even_changed, rice_k=4).decoded_sha256 == baseline.decoded_sha256
    )
    odd_changed = codes.copy()
    odd_changed[1, 1] = 1
    assert (
        temporal_y1_latents_from_interleaved_v9_codes(odd_changed, rice_k=4).decoded_sha256 != baseline.decoded_sha256
    )
    assert (
        semantic_root_module.V9_Y1_ODD_ROW_PROJECTION_CONTRACT
        == "COUNT_ONLY_CODE_2P_PLUS_1_Y1_ROWS_DISCARD_EVEN_Y0_ROWS"
    )
