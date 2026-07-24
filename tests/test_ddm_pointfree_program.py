from __future__ import annotations

import hashlib
import zlib
from dataclasses import replace

import numpy as np
import pytest

from tac.optimization.ddm_dv2_sdwl1 import (
    FactInventory,
    SentenceLayout,
    SentenceOptions,
    TemporalMode,
    serialize_sentence,
)
from tac.optimization.ddm_pointfree_program import (
    Formulation,
    PointFreeProgramError,
    apply_template,
    channel_affine,
    code_compiled_flat,
    code_compiled_program,
    code_real_payload,
    compile_dv2_sentence,
    compile_g1_worldsheet,
    compile_v15_template_bank,
    compose_pointfree,
    decode_real_payload,
    decode_two_typed_payload,
    execute_program,
    learn_shared_literals,
    rate_row,
    stratum_mask,
    xi_advect,
)
from tac.optimization.direct_description_carrier_compose import (
    RowBandScorerTemplateV1,
    ScorerSolvedTemplateBankV1,
    encode_scorer_solved_template_bank,
)
from tac.optimization.direct_description_g1_worldsheet import encode_g1_movable_worldsheet


@pytest.fixture
def g1_payload() -> bytes:
    labels = np.zeros((3, 384, 512), dtype=np.uint8)
    labels[0, 20:60, 30:70] = 3
    labels[1, 30:70, 40:80] = 3
    labels[2, 40:80, 50:90] = 3
    payload, _metadata = encode_g1_movable_worldsheet(labels)
    return payload


@pytest.fixture
def template_payload() -> bytes:
    repeated_rgb = bytes((17, 29, 43))
    bank = ScorerSolvedTemplateBankV1(
        (
            RowBandScorerTemplateV1("Lane", "fill", 0, 128, 1, 1, repeated_rgb),
            RowBandScorerTemplateV1("Lane", "fill", 128, 256, 1, 1, repeated_rgb),
            RowBandScorerTemplateV1("Lane", "fill", 256, 384, 1, 1, repeated_rgb),
        )
    )
    return encode_scorer_solved_template_bank(bank)


@pytest.fixture
def dv2_payload() -> bytes:
    tensor = np.zeros((2, 11, 8), dtype="<i8")
    tensor[0, 0, :5] = (1, 2, 3, 4, 5)
    tensor[1, 0, :5] = (2, 3, 4, 5, 6)
    tensor[:, 10, :6] = np.asarray((1, 2, 3, 4, 5, 6), dtype="<i8")
    inventory = FactInventory(
        tensor=tensor,
        source_height=64,
        source_width=96,
        semantic_sha256=hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),
    )
    inner = serialize_sentence(
        inventory,
        SentenceOptions(
            layout=SentenceLayout.TYPED_SECTION,
            temporal_mode=TemporalMode.CAUSAL_DELTA,
        ),
    )
    return zlib.compress(inner, level=9)


@pytest.mark.parametrize("formulation", list(Formulation))
def test_g1_program_replays_exact_source(
    g1_payload: bytes,
    formulation: Formulation,
) -> None:
    compiled = compile_g1_worldsheet(g1_payload, formulation)
    assert execute_program(compiled.program) == g1_payload
    assert rate_row(compiled, g1_payload)["semantic_parseback_exact"] is True


@pytest.mark.parametrize("formulation", list(Formulation))
def test_template_program_replays_exact_source(
    template_payload: bytes,
    formulation: Formulation,
) -> None:
    compiled = compile_v15_template_bank(template_payload, formulation)
    assert execute_program(compiled.program) == template_payload
    if formulation is Formulation.STRUCTURAL:
        assert compiled.video_derived_library_bytes == 3


@pytest.mark.parametrize("formulation", list(Formulation))
def test_dv2_program_replays_exact_outer_payload(
    dv2_payload: bytes,
    formulation: Formulation,
) -> None:
    compiled = compile_dv2_sentence(dv2_payload, formulation)
    assert execute_program(compiled.program) == dv2_payload


def test_measured_basis_is_rank_polymorphic() -> None:
    labels = np.asarray([[[0, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=np.uint8)
    mask = stratum_mask(labels, 1)
    patch = np.asarray([[[11, 12, 13]]], dtype=np.uint8)
    rendered = apply_template(mask, patch)
    assert rendered.shape == (2, 2, 2, 3)
    assert np.array_equal(rendered[mask], np.broadcast_to(patch[0, 0], (4, 3)))

    identity_batch = xi_advect(np.zeros((2, 3, 6), dtype=np.float64))
    assert identity_batch.shape == (2, 3, 4, 4)
    assert np.allclose(identity_batch, np.eye(4))

    source = np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    affine = channel_affine(
        source,
        np.asarray((2.0, 1.0, 0.5), dtype=np.float32),
        np.asarray((1.0, 0.0, 1.0), dtype=np.float32),
    )
    assert affine.tolist() == [[[[3, 2, 2], [9, 5, 4]]]][0]
    assert compose_pointfree(lambda value: value + 1, lambda value: value * 3)(4) == 15


def test_shared_literal_learning_is_bounded_and_deterministic() -> None:
    literals = (b"repeat", b"solo", b"repeat", b"repeat", b"also", b"also")
    first = learn_shared_literals(literals, max_entries=1)
    second = learn_shared_literals(literals, max_entries=1)
    assert first == second
    assert first.entries == (b"repeat",)
    assert first.references == (0, None, 0, 0, None, None)


def test_real_coder_is_strict_and_canonical() -> None:
    raw = b"program-description" * 20
    coded = code_real_payload(raw)
    assert decode_real_payload(coded.payload) == raw
    with pytest.raises(PointFreeProgramError):
        decode_real_payload(coded.payload + b"\x00")


def test_rate_row_rejects_a_source_other_than_compile_time_replay(g1_payload: bytes) -> None:
    compiled = compile_g1_worldsheet(g1_payload, Formulation.STRUCTURAL)
    with pytest.raises(PointFreeProgramError, match="compile-time exact replay"):
        rate_row(compiled, g1_payload + b"\x00")


def test_two_typed_counting_is_bound_to_executable_program(g1_payload: bytes) -> None:
    compiled = compile_g1_worldsheet(g1_payload, Formulation.STRUCTURAL)
    forged = replace(
        compiled,
        program_skeleton_wire=compiled.program_skeleton_wire + b"\x00",
    )
    with pytest.raises(
        PointFreeProgramError,
        match="differs from executable program",
    ):
        code_compiled_program(forged)


def test_structural_program_uses_exact_two_typed_skeleton_fiber_split(
    g1_payload: bytes,
    template_payload: bytes,
    dv2_payload: bytes,
) -> None:
    rows = (
        (compile_g1_worldsheet(g1_payload, Formulation.STRUCTURAL), g1_payload),
        (
            compile_v15_template_bank(template_payload, Formulation.STRUCTURAL),
            template_payload,
        ),
        (compile_dv2_sentence(dv2_payload, Formulation.STRUCTURAL), dv2_payload),
    )
    for compiled, source in rows:
        assert compiled.discrete_skeleton_scope_eligible is True
        assert execute_program(compiled.program) == source
        measured = rate_row(compiled, source)
        assert measured["two_typed_split_status"] == "MEASURED_EXACT"
        assert (
            measured["program_skeleton_counted_bytes"]
            + measured["program_fiber_counted_bytes"]
            == measured["program_counted_bytes"]
        )
        assert (
            measured["flat_skeleton_counted_bytes"]
            + measured["flat_fiber_counted_bytes"]
            == measured["flat_counted_bytes"]
        )
        program_coded = code_compiled_program(compiled)
        skeleton, slots = decode_two_typed_payload(program_coded.payload)
        assert skeleton == compiled.program_skeleton_wire
        assert slots == compiled.program_fiber_slots
        assert code_compiled_flat(compiled, source).framed_bytes == measured[
            "flat_counted_bytes"
        ]


def test_template_reuse_references_opaque_rgb_fiber_without_tokenizing(
    template_payload: bytes,
) -> None:
    compiled = compile_v15_template_bank(template_payload, Formulation.STRUCTURAL)
    row = rate_row(compiled, template_payload)
    assert row["program_fiber_slot_count"] == 1
    assert row["flat_fiber_slot_count"] == 3
    assert row["program_fiber_counted_bytes"] == 3
    assert row["flat_fiber_counted_bytes"] == 9
    assert row["delta_fiber_bytes"] == -6


@pytest.mark.parametrize("formulation", [Formulation.LITERAL, Formulation.SHARED_LIBRARY])
def test_opaque_controls_cannot_close_discrete_skeleton_scope(
    dv2_payload: bytes,
    formulation: Formulation,
) -> None:
    compiled = compile_dv2_sentence(dv2_payload, formulation)
    row = rate_row(compiled, dv2_payload)
    assert row["discrete_skeleton_scope_eligible"] is False
    assert row["two_typed_split_status"] == "OPAQUE_CONTROL_NOT_ROUTABLE"
    assert row["delta_skeleton_bytes"] is None
