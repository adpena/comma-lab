"""Behavioral certificate for the V10 counted compiler/receiver keystone."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable

import pytest

import tac.witness_dsl.v10_compiler_receiver as v10
from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser
from tac.witness_dsl.lawref import lawref_to_declaration
from tac.witness_dsl.lawref_builtins import LR_CONTROL_DENOMINATOR
from tac.witness_dsl.typed_config import (
    ProvenanceClass,
    Provenanced,
    TypedAnneal,
    TypedLever,
    TypedRegularizer,
    TypedStage,
    TypedWitnessConfig,
)
from tac.witness_dsl.v10_compiler_receiver import (
    CHECKPOINT_SCHEMA,
    FACTOR2_RECEIVER_CONTRACT_ID,
    FROZEN_FACTOR_IDS,
    FROZEN_HANDLER_REGISTRY,
    FROZEN_ROUTES,
    HANDLER_IMPLEMENTATION_SHA256S,
    HANDLER_REGISTRY_SHA256,
    HANDLER_SHARED_SEMANTICS_SHA256,
    IMPLEMENTED_FACTOR_IDS,
    MAGIC,
    MISSING_FACTOR_IDS,
    PREFIX,
    QUOTIENT_BASE_FACTOR_IDS,
    CompletenessRow,
    EvidenceArtifact,
    HandlerResult,
    ReceiverCheckpoint,
    Section,
    V10Refusal,
    build_payload_program,
    canonical_semantic_payload,
    compile_cold_v10,
    parse_payload_program,
    receive_payload_program,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _config(*, seed: int = 17) -> TypedWitnessConfig:
    lawref_flag = "--lr-anneal-epochs"
    return TypedWitnessConfig(
        name="v10-local-structural-fixture",
        out_dir="experiments/results/v10_local_structural_fixture",
        gt_cache="experiments/results/local_fixture.npz",
        num_pairs=600,
        epochs=1500,
        wall_clock_budget_days=Provenanced(
            value=3.7,
            provenance=ProvenanceClass.DERIVED_AT_CONFIG,
            unit="days",
            source="local structural fixture; no launch",
        ),
        seed=seed,
        temp=TypedAnneal(
            start=Provenanced(
                value=1.0, provenance=ProvenanceClass.MEASURED_ANCHOR, unit="tau"
            ),
            end=Provenanced(
                value=0.31, provenance=ProvenanceClass.MEASURED_ANCHOR, unit="tau"
            ),
        ),
        stages=(
            TypedStage(
                name="tau_softplus",
                start_epoch_flag="--tau-softplus-start-epoch",
                start_epoch=300,
            ),
        ),
        regularizers=(
            TypedRegularizer(
                flag="--eikonal-weight",
                weight=Provenanced(
                    value=0.01, provenance=ProvenanceClass.DERIVED_AT_CONFIG
                ),
            ),
        ),
        levers=(
            TypedLever(
                name="LawRef structural positive control",
                overrides={lawref_flag: 1000},
                lawrefs={lawref_flag: LR_CONTROL_DENOMINATOR},
                lawref_declarations={
                    lawref_flag: lawref_to_declaration(LR_CONTROL_DENOMINATOR)
                },
            ),
        ),
        base={
            "--w-seg": 100.0,
            "--curriculum": True,
            "--verdict-pairs": 0,
            "--seed": seed,
        },
    )


def _semantic_bodies(*, quotient_delta: int = 1, seed_bytes: list[int] | None = None):
    return {
        "counted_generator_v2": {
            "frame0_rgb": [10, 20, 30, 40, 50, 60],
            "frame1_rgb": [12, 22, 32, 42, 52, 62],
            "seed_bytes": list(seed_bytes or [1, 2, 3]),
        },
        "factor2_integer_scorer_plane_v1": {
            "y_uint8": list(range(3 * 4 * 3)),
            "camera_shape": [7, 9, 3],
            "scorer_shape": [3, 4, 3],
            "receiver_contract_id": FACTOR2_RECEIVER_CONTRACT_ID,
        },
        "frame0_pose_six_carrier_v1": {
            "frame0_delta": [1, 0, -1, 2, 0, -2],
            "pose_six": [1, 2, 3, 4, 5, 6],
        },
        "init_head_solve_v2": {"head_bias": [2, -2, 4, -4, 6, -6]},
        "shared_resize_preimage_v1": {"fanout": [0, 2, 4], "weights": [1, 2, 1]},
        "rgb_yuv6_projection_v1": {"rgb_bias": [1, -1, 2]},
        "blind_fill_rate_grammar_v1": {
            "blind_indices": [4],
            "fill_value": 77,
            "rate_tokens": [91, 3],
        },
        "quotient_residual_t_v2": {
            "updates": [
                {
                    "class_id": "road",
                    "cell_id": "cell_0",
                    "frame": "frame1",
                    "index": 1,
                    "delta": quotient_delta,
                }
            ]
        },
    }


def _sections(
    *,
    quotient_delta: int = 1,
    seed_bytes: list[int] | None = None,
    bodies: dict[str, dict[str, Any]] | None = None,
) -> tuple[Section, ...]:
    active_bodies = bodies or _semantic_bodies(
        quotient_delta=quotient_delta, seed_bytes=seed_bytes
    )
    sections: list[Section] = []
    for index, route in enumerate(FROZEN_ROUTES):
        prior = FROZEN_ROUTES[:index]
        is_t = index == len(FROZEN_ROUTES) - 1
        sections.append(
            Section(
                section_id=route.section_id,
                factor_ids=route.factor_ids,
                producer_id=route.producer_id,
                consumer_id=route.consumer_id,
                encoding=route.encoding,
                video_derived=True,
                payload=canonical_semantic_payload(active_bodies[route.encoding]),
                apply_order=index,
                owned_parameter_groups=(route.owned_parameter_group,),
                frozen_parameter_groups=tuple(item.owned_parameter_group for item in prior),
                class_ids=("road",) if is_t else (),
                cell_ids=("cell_0",) if is_t else (),
                depends_on=tuple(item.section_id for item in prior),
                quotient_base_factor_ids=QUOTIENT_BASE_FACTOR_IDS if is_t else (),
            )
        )
    return tuple(sections)


def _program_inputs(
    config: TypedWitnessConfig, sections: tuple[Section, ...]
) -> tuple[str, str, bytes]:
    argv, _manifest = config.to_program().compile_trainer_argv_with_constants(
        {"sigma_probe": "island_dilation_knee_0630"}
    )
    argv_hash = _sha("\x00".join(argv).encode("utf-8"))
    config_hash = config.typed_config_hash()
    program = build_payload_program(
        sections, typed_config_hash=config_hash, argv_sha256=argv_hash
    )
    return config_hash, argv_hash, program


def _rows(config_hash: str, parsed, receiver) -> tuple[CompletenessRow, ...]:
    section_by_factor = {
        factor_id: section
        for section in parsed.sections
        for factor_id in section.metadata["factor_ids"]
    }
    receipt_by_section = {receipt["section_id"]: receipt for receipt in receiver.receipts}
    rows: list[CompletenessRow] = []
    for factor_id in FROZEN_FACTOR_IDS:
        if factor_id in MISSING_FACTOR_IDS:
            rows.append(
                CompletenessRow(
                    factor_id=factor_id,
                    term_id=f"factor_{factor_id}_missing",
                    owner_task="matrix",
                    disposition="MISSING",
                    derivation_ref=(
                        ".omx/research/inverse_solve_completeness_matrix_20260718.md"
                    ),
                    build_sha="UNCOMMITTED_LOCAL",
                    compiled_config_hash=config_hash,
                    consumer_id="BLOCKED",
                    resume_schema_and_replay_ref=(
                        f"{CHECKPOINT_SCHEMA}:local-bit-identical-prefix-replay"
                    ),
                    measurement_receipt_sha256=None,
                    authority_axis="local-CPU structural/non-score",
                    interaction_receipts=(),
                    adoption_or_scoped_exclusion=(
                        "missing exact consumer; no launch, no score, and no promotion"
                    ),
                    strict_certificate="MISSING",
                )
            )
            continue
        route = next(route for route in FROZEN_ROUTES if factor_id in route.factor_ids)
        section = section_by_factor[factor_id]
        runtime_receipt = receipt_by_section[section.section_id]
        disposition = "FOLDED" if factor_id in {"1", "5"} else "HAVE"
        artifacts: tuple[EvidenceArtifact, ...] = ()
        if disposition == "FOLDED":
            artifacts = (
                EvidenceArtifact.create(
                    {
                        "interaction": "payload reopened by frozen semantic handler",
                        "score_claim": False,
                    },
                    factor_id=factor_id,
                    producer_id=route.producer_id,
                    consumer_id=route.consumer_id,
                    compiled_config_hash=config_hash,
                    program_sha256=parsed.program_sha256,
                    covered_section_id=section.section_id,
                    covered_section_sha256=section.metadata["sha256"],
                    receiver_receipt=runtime_receipt,
                ),
            )
        rows.append(
            CompletenessRow(
                factor_id=factor_id,
                term_id=f"factor_{factor_id}_local_structure",
                owner_task="#529" if factor_id == "1" else "#531" if factor_id == "5" else "matrix",
                disposition=disposition,
                derivation_ref=(
                    ".omx/research/inverse_solve_completeness_matrix_20260718.md"
                ),
                build_sha="UNCOMMITTED_LOCAL",
                compiled_config_hash=config_hash,
                consumer_id=route.consumer_id,
                resume_schema_and_replay_ref=(
                    f"{CHECKPOINT_SCHEMA}:local-bit-identical-prefix-replay"
                ),
                measurement_receipt_sha256=artifacts[0].sha256 if artifacts else None,
                authority_axis="local-CPU structural/non-score",
                interaction_receipts=artifacts,
                adoption_or_scoped_exclusion=(
                    "local structural evidence only; no launch, no score, and no promotion"
                ),
                strict_certificate="PARTIAL",
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class Fixture:
    config: TypedWitnessConfig
    sections: tuple[Section, ...]
    program: bytes
    parsed: Any
    receiver: Any
    rows: tuple[CompletenessRow, ...]


def _fixture(*, seed: int = 17, quotient_delta: int = 1) -> Fixture:
    config = _config(seed=seed)
    sections = _sections(quotient_delta=quotient_delta)
    config_hash, _argv_hash, program = _program_inputs(config, sections)
    parsed = parse_payload_program(program)
    receiver = receive_payload_program(program)
    return Fixture(
        config=config,
        sections=sections,
        program=program,
        parsed=parsed,
        receiver=receiver,
        rows=_rows(config_hash, parsed, receiver),
    )


def _compile(fixture: Fixture):
    return compile_cold_v10(
        fixture.config,
        fixture.sections,
        fixture.rows,
        target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
    )


def _rewrite_header(program: bytes, mutate: Callable[[dict[str, Any]], None]) -> bytes:
    _magic, _version, header_length = PREFIX.unpack_from(program)
    header_end = PREFIX.size + header_length
    header = json.loads(program[PREFIX.size:header_end])
    mutate(header)
    header_bytes = _canonical(header)
    return PREFIX.pack(MAGIC, v10.VERSION, len(header_bytes)) + header_bytes + program[header_end:]


def _rewrite_checkpoint(
    payload: bytes, mutate: Callable[[dict[str, Any]], None]
) -> bytes:
    doc = json.loads(payload)
    mutate(doc)
    return _canonical(doc)


def test_exact_paid_instruction_order_and_frozen_routes_are_pinned() -> None:
    assert [
        (route.kind.value, route.factor_ids, route.encoding, route.producer_id, route.consumer_id)
        for route in FROZEN_ROUTES
    ] == [
        (
            "CountedGenerator",
            ("1",),
            "counted_generator_v2",
            "typed_v10_compiler",
            "receiver.counted_generator",
        ),
        (
            "Factor2IntegerScorerPlane",
            ("2",),
            "factor2_integer_scorer_plane_v1",
            "production_archive_builder",
            "receiver.factor2_integer_scorer_plane",
        ),
        (
            "Frame0PoseSixCarrier",
            ("7", "8"),
            "frame0_pose_six_carrier_v1",
            "frame0_pose_six_compiler",
            "receiver.frame0_pose_six_carrier",
        ),
        (
            "InitHeadSolve",
            ("6",),
            "init_head_solve_v2",
            "init_head_solver",
            "receiver.init_head_solve",
        ),
        (
            "SharedResizePreimage",
            ("3a", "3b"),
            "shared_resize_preimage_v1",
            "shared_resize_preimage_solver",
            "receiver.shared_resize_preimage",
        ),
        (
            "RgbYuv6Projection",
            ("4",),
            "rgb_yuv6_projection_v1",
            "rgb_yuv6_projector",
            "receiver.rgb_yuv6_projection",
        ),
        (
            "BlindFillRateGrammar",
            ("9",),
            "blind_fill_rate_grammar_v1",
            "blind_fill_rate_compiler",
            "receiver.blind_fill_rate_grammar",
        ),
        (
            "QuotientResidualT",
            ("5",),
            "quotient_residual_t_v2",
            "quotient_residual_trainer",
            "receiver.quotient_residual_T",
        ),
    ]
    assert IMPLEMENTED_FACTOR_IDS == ("1", "2", "3a", "3b", "4", "5", "6", "7", "8", "9")
    assert MISSING_FACTOR_IDS == ("10",)
    assert set(HANDLER_IMPLEMENTATION_SHA256S) == {
        route.encoding for route in FROZEN_ROUTES
    }
    assert len(HANDLER_SHARED_SEMANTICS_SHA256) == 64
    assert all(len(digest) == 64 for digest in HANDLER_IMPLEMENTATION_SHA256S.values())


def test_valid_fixture_compiles_and_exposes_canonical_332_audit_without_authorizing() -> None:
    fixture = _fixture()
    result = _compile(fixture)
    assert result.typed_config_hash == fixture.config.typed_config_hash()
    assert result.argv_sha256 == _sha("\x00".join(result.trainer_argv).encode("utf-8"))
    assert result.resolved_lawref_manifest["--lr-anneal-epochs"]["value"] == 1000
    assert result.dsl_compile_provenance["dsl_compile_hash"] == result.dsl_compile_hash
    assert result.dsl_bijection_complete is (not result.dsl_bijection_violations)
    assert type(result.dsl_bijection_complete) is bool
    build_real_trainer_parser().parse_args(list(result.trainer_argv[2:]))
    assert result.resume_replay_equal is True
    factor2_receipt = next(
        receipt
        for receipt in result.receiver_receipts
        if receipt["section_id"] == "factor2_integer_scorer_plane"
    )
    assert factor2_receipt["consumption_count"] == 1
    assert factor2_receipt["authoritative_handler"] is True
    assert "2" in result.implemented_factor_ids
    assert "2" not in result.missing_factor_ids
    assert result.missing_factor_ids == ("10",)
    assert result.launch_ready is False
    assert result.score_claim is False and result.promotion_eligible is False


def test_factor2_route_refuses_oversized_geometry_as_v10_refusal() -> None:
    bodies = _semantic_bodies()
    bodies["factor2_integer_scorer_plane_v1"]["camera_shape"] = [4097, 9, 3]
    with pytest.raises(V10Refusal, match="exceeds production bounds"):
        receive_payload_program(
            build_payload_program(
                _sections(bodies=bodies),
                typed_config_hash="a" * 64,
                argv_sha256="b" * 64,
            )
        )


def test_program_and_receiver_are_deterministic() -> None:
    fixture = _fixture()
    first = _compile(fixture)
    second = _compile(fixture)
    assert first.payload_program_bytes == second.payload_program_bytes
    assert first.receiver_output_bytes == second.receiver_output_bytes
    assert first.dsl_compile_hash == second.dsl_compile_hash
    assert first.dsl_compile_provenance == second.dsl_compile_provenance
    assert first.dsl_bijection_violations == second.dsl_bijection_violations


def test_typed_seed_changes_the_counted_program_identity() -> None:
    first = _fixture(seed=17)
    second = _fixture(seed=18)
    assert first.config.typed_config_hash() != second.config.typed_config_hash()
    assert first.program != second.program
    assert first.parsed.program_sha256 != second.parsed.program_sha256


def test_completeness_generator_is_frozen_once_and_keeps_exact_eleven_rows() -> None:
    fixture = _fixture()
    result = compile_cold_v10(
        fixture.config,
        fixture.sections,
        (row for row in fixture.rows),
        target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
    )
    assert tuple(row["factor_id"] for row in result.completeness_rows) == FROZEN_FACTOR_IDS


def test_missing_factors_have_blocked_consumers_no_sections_and_no_receipts() -> None:
    fixture = _fixture()
    result = _compile(fixture)
    owned = {
        factor_id
        for section in fixture.parsed.sections
        for factor_id in section.metadata["factor_ids"]
    }
    by_factor = {row["factor_id"]: row for row in result.completeness_rows}
    assert not (owned & set(MISSING_FACTOR_IDS))
    for factor_id in MISSING_FACTOR_IDS:
        assert by_factor[factor_id]["disposition"] == "MISSING"
        assert by_factor[factor_id]["consumer_id"] == "BLOCKED"
        assert by_factor[factor_id]["interaction_receipts"] == []
        assert by_factor[factor_id]["measurement_receipt_sha256"] is None


def test_false_factor_map_and_factorless_or_extra_sections_refuse() -> None:
    sections = list(_sections())
    with pytest.raises(V10Refusal, match="route metadata drift"):
        build_payload_program(
            [replace(sections[0], factor_ids=("2",)), *sections[1:]],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )
    with pytest.raises(V10Refusal, match="factor_ids cannot be empty"):
        replace(sections[0], factor_ids=())
    with pytest.raises(V10Refusal, match="frozen instruction"):
        build_payload_program(
            [*sections, replace(sections[-1], apply_order=len(sections))],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )
    with pytest.raises(V10Refusal, match="duplicate|route metadata drift"):
        build_payload_program(
            [sections[0], replace(sections[1], factor_ids=("1",)), *sections[2:]],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )
    with pytest.raises(V10Refusal, match="unknown typed instruction"):
        build_payload_program(
            [replace(sections[0], encoding="unknown_v10_encoding"), *sections[1:]],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )


def test_video_false_and_coordinated_route_self_attestation_refuse() -> None:
    sections = list(_sections())
    with pytest.raises(V10Refusal, match="route metadata drift"):
        build_payload_program(
            [*sections[:4], replace(sections[4], video_derived=False), *sections[5:]],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )
    lied = replace(
        sections[1],
        producer_id="attacker.claimed_producer",
        consumer_id="attacker.claimed_consumer",
    )
    with pytest.raises(V10Refusal, match="route metadata drift"):
        build_payload_program(
            [sections[0], lied, *sections[2:]],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )


def test_quotient_t_is_factor_five_terminal_and_freezes_exact_base() -> None:
    sections = list(_sections())
    terminal = sections[-1]
    assert terminal.factor_ids == ("5",)
    assert terminal.quotient_base_factor_ids == QUOTIENT_BASE_FACTOR_IDS
    assert terminal.depends_on == tuple(section.section_id for section in sections[:-1])
    assert terminal.frozen_parameter_groups == tuple(
        section.owned_parameter_groups[0] for section in sections[:-1]
    )
    with pytest.raises(V10Refusal, match="depend on every predecessor"):
        build_payload_program(
            [*sections[:-1], replace(terminal, depends_on=terminal.depends_on[:-1])],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )
    with pytest.raises(V10Refusal, match="wrong exact quotient base"):
        build_payload_program(
            [
                *sections[:-1],
                replace(terminal, quotient_base_factor_ids=QUOTIENT_BASE_FACTOR_IDS[:-1]),
            ],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )
    with pytest.raises(V10Refusal, match="disjoint parameter group"):
        build_payload_program(
            [
                *sections[:-1],
                replace(terminal, owned_parameter_groups=sections[0].owned_parameter_groups),
            ],
            typed_config_hash="a" * 64,
            argv_sha256="b" * 64,
        )


def test_shared_resize_factors_share_one_range_and_full_wire_counts_it_once() -> None:
    fixture = _fixture()
    proof = fixture.parsed.parser_proof
    range_3a = proof["factor_ranges"]["3a"]
    range_3b = proof["factor_ranges"]["3b"]
    assert range_3a == range_3b
    assert range_3a["section_id"] == "shared_resize_preimage"
    assert proof["unique_payload_range_count"] == len(FROZEN_ROUTES)
    assert proof["payload_bytes"] == sum(len(section.payload) for section in fixture.sections)
    assert proof["counted_video_derived_payload_bytes"] == proof["payload_bytes"]


def test_all_program_bytes_are_contiguous_owned_and_consumed_once() -> None:
    fixture = _fixture()
    proof = fixture.parsed.parser_proof
    assert proof["partition_sum_bytes"] == len(fixture.program)
    assert proof["all_program_bytes_counted"] and proof["no_unowned_wire_bytes"]
    assert proof["contiguous"] and proof["no_gaps"] and proof["no_overlaps"]
    assert proof["no_trailing_bytes"]
    assert len(fixture.receiver.receipts) == len(FROZEN_ROUTES)
    assert all(receipt["consumption_count"] == 1 for receipt in fixture.receiver.receipts)
    assert all(receipt["authoritative_handler"] is True for receipt in fixture.receiver.receipts)
    assert all(
        receipt["handler_registry_sha256"] == HANDLER_REGISTRY_SHA256
        for receipt in fixture.receiver.receipts
    )
    assert all(
        receipt["handler_implementation_sha256"]
        == HANDLER_IMPLEMENTATION_SHA256S[receipt["encoding"]]
        and receipt["handler_shared_semantics_sha256"]
        == HANDLER_SHARED_SEMANTICS_SHA256
        for receipt in fixture.receiver.receipts
    )
    assert all(
        receipt["decoded_frames_before_sha256"]
        != receipt["decoded_frames_after_sha256"]
        for receipt in fixture.receiver.receipts
    )


def _mutated_bodies(index: int) -> dict[str, dict[str, Any]]:
    bodies = _semantic_bodies()
    encoding = FROZEN_ROUTES[index].encoding
    body = bodies[encoding]
    if index == 0:
        body["frame0_rgb"][0] += 1
    elif index == 1:
        body["y_uint8"][0] += 1
    elif index == 2:
        body["pose_six"][5] += 1
    elif index == 3:
        body["head_bias"][3] += 1
    elif index == 4:
        body["weights"][0] += 1
    elif index == 5:
        body["rgb_bias"][0] += 1
    elif index == 6:
        body["rate_tokens"][0] += 1
    else:
        body["updates"][0]["delta"] += 1
    return bodies


@pytest.mark.parametrize("section_index", range(len(FROZEN_ROUTES)))
def test_each_section_has_a_real_semantic_mutation(section_index: int) -> None:
    config = _config()
    before_sections = _sections()
    after_sections = _sections(bodies=_mutated_bodies(section_index))
    _config_hash, _argv_hash, before_program = _program_inputs(config, before_sections)
    _config_hash, _argv_hash, after_program = _program_inputs(config, after_sections)
    before = parse_payload_program(before_program)
    after = parse_payload_program(after_program)
    before_state = json.loads(receive_payload_program(before_program).output_bytes)
    after_state = json.loads(receive_payload_program(after_program).output_bytes)
    assert (
        before_state["frame0_rgb"],
        before_state["frame1_rgb"],
    ) != (
        after_state["frame0_rgb"],
        after_state["frame1_rgb"],
    )
    assert [section.metadata["sha256"] for section in before.sections if section.metadata["apply_order"] != section_index] == [
        section.metadata["sha256"] for section in after.sections if section.metadata["apply_order"] != section_index
    ]


def test_named_payload_mutation_positive_control_changes_decoded_frame() -> None:
    config = _config()
    _, _, before_program = _program_inputs(config, _sections(quotient_delta=1))
    _, _, after_program = _program_inputs(config, _sections(quotient_delta=2))
    before_state = json.loads(receive_payload_program(before_program).output_bytes)
    after_state = json.loads(receive_payload_program(after_program).output_bytes)
    assert before_state["frame1_rgb"] != after_state["frame1_rgb"]


def test_factor_one_atomically_counts_generator_and_seed_bytes() -> None:
    config = _config()
    before = _sections(seed_bytes=[1, 2, 3])
    after = _sections(seed_bytes=[1, 2, 4])
    _config_hash, _argv_hash, before_program = _program_inputs(config, before)
    _config_hash, _argv_hash, after_program = _program_inputs(config, after)
    before_parsed = parse_payload_program(before_program)
    parse_payload_program(after_program)
    assert before_parsed.sections[0].metadata["factor_ids"] == ["1"]
    assert all("seed" not in section.metadata["section_id"] for section in before_parsed.sections[1:])
    before_state = json.loads(receive_payload_program(before_program).output_bytes)
    after_state = json.loads(receive_payload_program(after_program).output_bytes)
    assert before_state["frame0_rgb"] != after_state["frame0_rgb"]
    assert before_parsed.sections[0].metadata["sha256"] != parse_payload_program(
        after_program
    ).sections[0].metadata["sha256"]


def test_generator_refuses_seed_bytes_that_never_reach_a_sample() -> None:
    bodies = _semantic_bodies(seed_bytes=list(range(8)))
    with pytest.raises(V10Refusal, match="each reach at least one"):
        receive_payload_program(
            _program_inputs(_config(), _sections(bodies=bodies))[2]
        )


def test_generator_accepts_exact_two_frame_seed_reachability_boundary() -> None:
    bodies = _semantic_bodies(seed_bytes=list(range(7)))
    result = receive_payload_program(
        _program_inputs(_config(), _sections(bodies=bodies))[2]
    )
    assert result.completed is True


def test_counted_but_semantically_inert_section_refuses() -> None:
    bodies = _semantic_bodies()
    bodies["frame0_pose_six_carrier_v1"] = {
        "frame0_delta": [0] * 6,
        "pose_six": [0] * 6,
    }
    with pytest.raises(V10Refusal, match="counted but made no semantic state change"):
        receive_payload_program(
            _program_inputs(_config(), _sections(bodies=bodies))[2]
        )


@pytest.mark.parametrize(
    "updates,pattern",
    [
        (
            [
                {
                    "class_id": "road",
                    "cell_id": "cell_0",
                    "frame": "frame1",
                    "index": 1,
                    "delta": 0,
                }
            ],
            "paid zero residual",
        ),
        (
            [
                {
                    "class_id": "road",
                    "cell_id": "cell_0",
                    "frame": "frame1",
                    "index": 1,
                    "delta": 1,
                },
                {
                    "class_id": "road",
                    "cell_id": "cell_0",
                    "frame": "frame1",
                    "index": 1,
                    "delta": -1,
                },
            ],
            "one residual owner",
        ),
    ],
)
def test_quotient_t_refuses_inert_or_double_owned_residuals(updates, pattern: str) -> None:
    bodies = _semantic_bodies()
    bodies["quotient_residual_t_v2"] = {"updates": updates}
    with pytest.raises(V10Refusal, match=pattern):
        receive_payload_program(
            _program_inputs(_config(), _sections(bodies=bodies))[2]
        )


def test_truncation_corruption_and_trailing_bytes_refuse() -> None:
    fixture = _fixture()
    boundaries = {1, PREFIX.size - 1, PREFIX.size}
    boundaries.update(section.end - 1 for section in fixture.parsed.sections)
    boundaries.update(section.end for section in fixture.parsed.sections[:-1])
    for boundary in sorted(boundaries):
        with pytest.raises(V10Refusal):
            parse_payload_program(fixture.program[:boundary])
    corrupted = bytearray(fixture.program)
    corrupted[fixture.parsed.sections[4].start] ^= 1
    with pytest.raises(V10Refusal, match="hash mismatch"):
        parse_payload_program(bytes(corrupted))
    with pytest.raises(V10Refusal, match="trailing"):
        parse_payload_program(fixture.program + b"\x00")


def test_public_receiver_always_reopens_bytes_and_rejects_forged_parsed_program() -> None:
    fixture = _fixture()
    forged = replace(fixture.parsed, sections=())
    with pytest.raises(V10Refusal, match="canonical program bytes"):
        receive_payload_program(forged)  # type: ignore[arg-type]


def test_exported_registry_rebinding_cannot_change_authoritative_receive(monkeypatch) -> None:
    fixture = _fixture()
    baseline = receive_payload_program(fixture.program)
    first = FROZEN_ROUTES[0]

    def substituted_handler(state, _metadata, payload):
        return HandlerResult(dict(state), first.semantic_fields, 1, _sha(payload))

    rebound = dict(FROZEN_HANDLER_REGISTRY)
    rebound[first.encoding] = substituted_handler
    monkeypatch.setattr(
        v10, "FROZEN_HANDLER_REGISTRY", MappingProxyType(rebound)
    )
    reopened = receive_payload_program(fixture.program)
    assert reopened.output_bytes == baseline.output_bytes
    assert all(receipt["authoritative_handler"] is True for receipt in reopened.receipts)


@pytest.mark.parametrize(
    "mutate,pattern",
    [
        (lambda header: header.__setitem__("version", True), "version"),
        (lambda header: header.__setitem__("section_count", 7.0), "section_count"),
        (
            lambda header: header.__setitem__("handler_registry_sha256", "0" * 64),
            "registry seal",
        ),
        (lambda header: header.__setitem__("implemented_factor_ids", ["1"]), "factor seal"),
    ],
)
def test_header_exact_types_and_registry_factor_seals_refuse(mutate, pattern: str) -> None:
    fixture = _fixture()
    with pytest.raises(V10Refusal, match=pattern):
        parse_payload_program(_rewrite_header(fixture.program, mutate))


def test_reopened_wire_refuses_video_or_factor_metadata_lies() -> None:
    fixture = _fixture()

    def video_lie(header):
        header["sections"][3]["video_derived"] = False

    def factor_lie(header):
        header["sections"][3]["factor_ids"] = ["2", "10"]

    for mutate in (video_lie, factor_lie):
        with pytest.raises(V10Refusal):
            parse_payload_program(_rewrite_header(fixture.program, mutate))


def test_reopened_wire_refuses_unknown_or_missing_section_schema_fields() -> None:
    fixture = _fixture()

    def extra_field(header):
        header["sections"][0]["attacker_field"] = True

    def missing_field(header):
        del header["sections"][0]["sha256"]

    for mutate in (extra_field, missing_field):
        with pytest.raises(V10Refusal, match="wrong schema"):
            parse_payload_program(_rewrite_header(fixture.program, mutate))


def test_custom_ignore_payload_handler_cannot_mint_authoritative_receipt() -> None:
    fixture = _fixture()
    first = FROZEN_ROUTES[0]

    def ignore_payload(state, _metadata, payload):
        return HandlerResult(
            dict(state), first.semantic_fields, 1, _sha(payload)
        )

    handlers = dict(FROZEN_HANDLER_REGISTRY)
    handlers[first.encoding] = ignore_payload
    with pytest.raises(V10Refusal, match="custom handler cannot issue"):
        v10._receive_payload_program_for_test(fixture.program, handlers=handlers)


def test_missing_handler_refuses_before_consumption() -> None:
    fixture = _fixture()
    handlers = dict(FROZEN_HANDLER_REGISTRY)
    del handlers[FROZEN_ROUTES[3].encoding]
    with pytest.raises(V10Refusal, match="missing handler"):
        v10._receive_payload_program_for_test(fixture.program, handlers=handlers)


def test_non_exact_or_non_one_handler_counts_refuse() -> None:
    with pytest.raises(V10Refusal, match="exact integer"):
        HandlerResult({}, ("x",), True, "a" * 64)
    with pytest.raises(V10Refusal, match="exact integer"):
        HandlerResult({}, ("x",), 1.0, "a" * 64)  # type: ignore[arg-type]
    fixture = _fixture()
    first = FROZEN_ROUTES[0]

    def zero_count(state, _metadata, payload):
        return HandlerResult(dict(state), first.semantic_fields, 0, _sha(payload))

    handlers = dict(FROZEN_HANDLER_REGISTRY)
    handlers[first.encoding] = zero_count
    with pytest.raises(V10Refusal, match="exactly once"):
        v10._receive_payload_program_for_test(fixture.program, handlers=handlers)


def test_resume_equals_uninterrupted_from_canonical_checkpoint_bytes() -> None:
    fixture = _fixture()
    full = receive_payload_program(fixture.program)
    stopped = receive_payload_program(fixture.program, stop_after=3)
    checkpoint_bytes = stopped.checkpoint.to_bytes()
    assert ReceiverCheckpoint.from_bytes(checkpoint_bytes) == stopped.checkpoint
    resumed = receive_payload_program(fixture.program, checkpoint=checkpoint_bytes)
    assert resumed.output_bytes == full.output_bytes
    assert len(resumed.receipts) == len(fixture.parsed.sections)
    assert tuple(receipt["section_id"] for receipt in resumed.receipts) == tuple(
        section.section_id for section in fixture.parsed.sections
    )
    with pytest.raises(V10Refusal, match="canonical reopened bytes"):
        receive_payload_program(
            fixture.program, checkpoint=stopped.checkpoint  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mutate,pattern",
    [
        (lambda doc: doc.__setitem__("extra", 1), "unknown or missing"),
        (lambda doc: doc.__setitem__("schema", "v10_receiver_checkpoint.v1"), "schema"),
        (lambda doc: doc.__setitem__("next_section_index", True), "exact integer"),
        (lambda doc: doc.__setitem__("next_section_index", 2.0), "exact integer"),
    ],
)
def test_checkpoint_exact_keyset_schema_and_types_refuse(mutate, pattern: str) -> None:
    fixture = _fixture()
    checkpoint = receive_payload_program(fixture.program, stop_after=2).checkpoint.to_bytes()
    with pytest.raises(V10Refusal, match=pattern):
        ReceiverCheckpoint.from_bytes(_rewrite_checkpoint(checkpoint, mutate))


def test_checkpoint_rejects_noncanonical_base64_pad_bits() -> None:
    fixture = _fixture()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    for stop_after in range(len(FROZEN_ROUTES)):
        payload = receive_payload_program(
            fixture.program, stop_after=stop_after
        ).checkpoint.to_bytes()
        doc = json.loads(payload)
        encoded = doc["state_bytes_b64"]
        padding = len(encoded) - len(encoded.rstrip("="))
        if padding:
            index = len(encoded) - padding - 1
            sextet = alphabet.index(encoded[index])
            unused_bits = 4 if padding == 2 else 2
            canonical_mask = ~((1 << unused_bits) - 1)
            replacement = (sextet & canonical_mask) | 1
            doc["state_bytes_b64"] = (
                encoded[:index] + alphabet[replacement] + encoded[index + 1 :]
            )
            assert base64.b64decode(doc["state_bytes_b64"]) == base64.b64decode(encoded)
            with pytest.raises(V10Refusal, match="noncanonical"):
                ReceiverCheckpoint.from_bytes(_canonical(doc))
            break
    else:
        pytest.fail("fixture produced no padded checkpoint base64 spelling")


def test_init_head_has_no_modulo_equivalent_payload_alias() -> None:
    positive = _semantic_bodies()
    negative = _semantic_bodies()
    positive["init_head_solve_v2"]["head_bias"] = [0, 0, 0, 0, 0, 1]
    negative["init_head_solve_v2"]["head_bias"] = [0, 0, 0, 0, 0, -255]
    _, _, positive_program = _program_inputs(
        _config(), _sections(bodies=positive)
    )
    _, _, negative_program = _program_inputs(
        _config(), _sections(bodies=negative)
    )
    positive_state = json.loads(receive_payload_program(positive_program).output_bytes)
    negative_state = json.loads(receive_payload_program(negative_program).output_bytes)
    assert positive_state["frame1_rgb"] != negative_state["frame1_rgb"]


def test_checkpoint_program_prefix_and_state_forgery_refuse() -> None:
    fixture = _fixture()
    checkpoint = receive_payload_program(fixture.program, stop_after=2).checkpoint.to_bytes()

    def program_drift(doc):
        doc["program_sha256"] = "0" * 64

    with pytest.raises(V10Refusal, match="program/config drift"):
        receive_payload_program(
            fixture.program, checkpoint=_rewrite_checkpoint(checkpoint, program_drift)
        )

    def prefix_drift(doc):
        doc["consumed_section_ids"] = list(reversed(doc["consumed_section_ids"]))

    with pytest.raises(V10Refusal, match="exact prefix"):
        receive_payload_program(
            fixture.program, checkpoint=_rewrite_checkpoint(checkpoint, prefix_drift)
        )

    forged_state = _canonical({"forged": True})

    def state_drift(doc):
        doc["state_bytes_b64"] = base64.b64encode(forged_state).decode("ascii")
        doc["state_sha256"] = _sha(forged_state)

    with pytest.raises(V10Refusal, match="deterministic consumed-prefix replay"):
        receive_payload_program(
            fixture.program, checkpoint=_rewrite_checkpoint(checkpoint, state_drift)
        )


def test_bool_stop_after_refuses() -> None:
    with pytest.raises(V10Refusal, match="exact integer"):
        receive_payload_program(_fixture().program, stop_after=True)


@pytest.mark.parametrize(
    "encoding",
    ("fork_head_solve_v1", "fork_ema_clearance_v1", "resume_lr_warmup_v1"),
)
def test_actual_fork_head_fork_ema_and_resume_lr_encodings_refuse(encoding: str) -> None:
    sections = list(_sections())
    sections[2] = replace(sections[2], encoding=encoding)
    with pytest.raises(V10Refusal, match="forbids typed instruction"):
        build_payload_program(
            sections, typed_config_hash="a" * 64, argv_sha256="b" * 64
        )


@pytest.mark.parametrize("token", ("fork_head", "fork_ema", "resume_lr"))
def test_cold_compile_refuses_fork_resume_state_tokens(token: str) -> None:
    fixture = _fixture()
    sections = list(fixture.sections)
    body = json.loads(sections[0].payload)
    body[token] = {"state": 1}
    sections[0] = replace(sections[0], payload=canonical_semantic_payload(body))
    with pytest.raises(V10Refusal, match="forbids fork/resume state token"):
        compile_cold_v10(
            fixture.config,
            sections,
            fixture.rows,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )


def test_cold_compile_refuses_non_null_resume_from(monkeypatch) -> None:
    fixture = _fixture()
    original = TypedWitnessConfig.to_program

    def resumed_program(self):
        return replace(original(self), resume_from="checkpoint.npz")

    monkeypatch.setattr(TypedWitnessConfig, "to_program", resumed_program)
    with pytest.raises(V10Refusal, match="non-null resume_from"):
        compile_cold_v10(
            fixture.config,
            fixture.sections,
            fixture.rows,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )


def test_folded_factor_cannot_clear_without_fresh_authoritative_evidence() -> None:
    fixture = _fixture()
    missing = list(fixture.rows)
    missing[0] = replace(
        missing[0], interaction_receipts=(), measurement_receipt_sha256=None
    )
    with pytest.raises(V10Refusal, match="FOLDED requires"):
        compile_cold_v10(
            fixture.config,
            fixture.sections,
            missing,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )
    stale_bytes = bytearray(fixture.rows[0].interaction_receipts[0].payload)
    stale_bytes[-1] ^= 1
    stale_artifact = replace(
        fixture.rows[0].interaction_receipts[0], payload=bytes(stale_bytes)
    )
    stale = list(fixture.rows)
    stale[0] = replace(stale[0], interaction_receipts=(stale_artifact,))
    with pytest.raises(V10Refusal, match="empty or stale"):
        compile_cold_v10(
            fixture.config,
            fixture.sections,
            stale,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )


@pytest.mark.parametrize(
    "kwargs,pattern",
    [
        ({"verdict": "FAIL"}, "verdict or authority axis"),
        ({"authority_axis": "invented-axis"}, "verdict or authority axis"),
    ],
)
def test_folded_factor_refuses_adverse_or_wrong_axis_evidence(kwargs, pattern: str) -> None:
    fixture = _fixture()
    source = fixture.rows[0].interaction_receipts[0]
    section = fixture.parsed.sections[0]
    adverse = EvidenceArtifact.create(
        {"interaction": "negative control", "score_claim": False},
        factor_id="1",
        producer_id=FROZEN_ROUTES[0].producer_id,
        consumer_id=FROZEN_ROUTES[0].consumer_id,
        compiled_config_hash=fixture.config.typed_config_hash(),
        program_sha256=fixture.parsed.program_sha256,
        covered_section_id=section.section_id,
        covered_section_sha256=section.metadata["sha256"],
        receiver_receipt=fixture.receiver.receipts[0],
        **kwargs,
    )
    rows = list(fixture.rows)
    rows[0] = replace(
        rows[0],
        interaction_receipts=(adverse,),
        measurement_receipt_sha256=adverse.sha256,
    )
    assert adverse.sha256 != source.sha256
    with pytest.raises(V10Refusal, match=pattern):
        compile_cold_v10(
            fixture.config,
            fixture.sections,
            rows,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )


def test_missing_factor_cannot_claim_consumer_or_receipt() -> None:
    fixture = _fixture()
    missing_factor_index = FROZEN_FACTOR_IDS.index("10")
    false_clear = list(fixture.rows)
    false_clear[missing_factor_index] = replace(
        false_clear[missing_factor_index],
        consumer_id=FROZEN_ROUTES[0].consumer_id,
        disposition="HAVE",
        strict_certificate="PARTIAL",
    )
    with pytest.raises(V10Refusal, match="MISSING requires BLOCKED"):
        compile_cold_v10(
            fixture.config,
            fixture.sections,
            false_clear,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )


def test_factor_receipt_forgery_cannot_claim_section_coverage() -> None:
    fixture = _fixture()
    artifact = fixture.rows[0].interaction_receipts[0]
    forged = replace(artifact, covered_section_sha256="0" * 64)
    rows = list(fixture.rows)
    rows[0] = replace(rows[0], interaction_receipts=(forged,))
    with pytest.raises(V10Refusal, match="disagrees|coverage drift"):
        compile_cold_v10(
            fixture.config,
            fixture.sections,
            rows,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )


def test_local_rows_cannot_self_declare_complete() -> None:
    fixture = _fixture()
    rows = tuple(replace(row, strict_certificate="COMPLETE") for row in fixture.rows)
    with pytest.raises(V10Refusal, match="cannot authorize COMPLETE"):
        compile_cold_v10(
            fixture.config,
            fixture.sections,
            rows,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )


def test_exact_cold_compiler_refuses_non_n600_or_partial_verdict() -> None:
    fixture = _fixture()
    with pytest.raises(V10Refusal, match="num_pairs == 600"):
        compile_cold_v10(
            fixture.config.model_copy(update={"num_pairs": 8}),
            fixture.sections,
            fixture.rows,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )
    partial = fixture.config.model_copy(
        update={"base": {"--w-seg": 100.0, "--curriculum": True, "--seed": 17}}
    )
    with pytest.raises(V10Refusal, match="verdict-pairs 0"):
        compile_cold_v10(
            partial,
            fixture.sections,
            fixture.rows,
            target_config_tags={"sigma_probe": "island_dilation_knee_0630"},
        )
