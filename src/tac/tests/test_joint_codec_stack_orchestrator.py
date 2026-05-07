"""Research-grade custody checks for the PARADIGM-gamma JCSP orchestrator."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from tac.balle_hyperprior_codec import (
    BalleHyperpriorCodec,
    HyperDecoder,
    HyperEncoder,
)
from tac.joint_codec_stack_orchestrator import (
    JCSP_ARCHIVE_MEMBER_CONTRACT_SCHEMA,
    JCSP_ARCHIVE_MEMBER_NAME,
    JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA,
    JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA,
    JCSP_STREAM_METADATA_SCHEMA,
    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_SCHEMA,
    JCSP_SUBMISSION_RUNTIME_OUTPUT_CONTRACT_SCHEMA,
    JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER,
    KIND_ARITHMETIC_STATIC,
    KIND_BALLE_HYPERPRIOR,
    StreamSource,
    build_jcsp_archive_member,
    build_jcsp_noop_archive_fixture,
    jcsp_stream_specs_manifest,
    load_jcsp_archive_member_for_runtime,
    model_to_jcsp_streams,
    run_joint_codec_stack,
    run_sequential_codec_stack,
    unpack_jcsp_container,
    validate_jcsp_container_runtime_parity,
)


def _balle_codec() -> BalleHyperpriorCodec:
    encoder = HyperEncoder(block_size=64, z_dim=4, hidden_dim=8, seed=2026)
    decoder = HyperDecoder(z_dim=4, hidden_dim=8, seed=2026)
    return BalleHyperpriorCodec(
        block_size=64,
        z_dim=4,
        hyper_encoder=encoder,
        hyper_decoder=decoder,
    )


def test_model_to_jcsp_streams_emits_deterministic_scoreless_stream_manifest() -> None:
    forward = {
        "z.bias": np.ones((2,), dtype=np.float32),
        "mask.index": np.arange(6, dtype=np.uint8),
        "a.weight": np.zeros((128, 128), dtype=np.float32),
    }
    reverse = {
        "a.weight": forward["a.weight"],
        "mask.index": forward["mask.index"],
        "z.bias": forward["z.bias"],
    }

    streams = model_to_jcsp_streams(forward)
    repeat_streams = model_to_jcsp_streams(reverse)
    manifest = jcsp_stream_specs_manifest(streams)
    repeat_manifest = jcsp_stream_specs_manifest(repeat_streams)

    assert streams == repeat_streams
    assert manifest == repeat_manifest
    assert manifest["schema"] == JCSP_STREAM_METADATA_SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["stream_count"] == 3
    assert [stream.name for stream in streams] == [
        "a.weight",
        "mask.index",
        "z.bias",
    ]
    weight = streams[0]
    assert weight.codec_kind == KIND_BALLE_HYPERPRIOR
    assert weight.tensor_shape == (128, 128)
    assert weight.tensor_dtype == "float32"
    assert weight.raw_bytes == 128 * 128 * 4
    assert weight.byte_estimate == 128 * 128 * 2
    assert weight.score_marginal_source == "missing_score_marginal"
    assert "score_marginal_artifact_missing" in weight.dispatch_blockers
    assert "refuse_dispatch_if_decode_validation_missing" in (
        weight.fail_closed_criteria
    )
    assert manifest["sidecar_policy"]["score_affecting_sidecars_allowed"] is False
    assert "exact_cuda_auth_eval_missing" in manifest["promotion_blockers"]


def test_model_to_jcsp_streams_fails_closed_on_stale_planning_annotations() -> None:
    state = {"a.weight": np.zeros((4, 4), dtype=np.float32)}

    with pytest.raises(ValueError, match=r"unknown streams: stale\.weight"):
        model_to_jcsp_streams(
            state,
            score_marginals={"stale.weight": 1.0e-6},
        )
    with pytest.raises(ValueError, match=r"unknown streams: stale\.weight"):
        model_to_jcsp_streams(
            state,
            codec_overrides={"stale.weight": KIND_ARITHMETIC_STATIC},
        )
    with pytest.raises(ValueError, match="explicit RAW_PASSTHROUGH"):
        model_to_jcsp_streams(state, wet_streams=("a.*",))


def test_balle_static_wins_records_actual_arithmetic_codec_kind() -> None:
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=128, dtype=np.int8)
    stream = StreamSource(
        name="renderer_qints",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_BALLE_HYPERPRIOR,
        balle_codec=_balle_codec(),
        score_per_byte_marginal=1e-6,
    )

    result = run_joint_codec_stack(
        streams=[stream],
        byte_budget=10_000,
        admm_max_iters=2,
    )
    parsed = unpack_jcsp_container(result.container_bytes)

    assert result.streams[0].codec_kind == KIND_ARITHMETIC_STATIC
    assert parsed["streams"][0]["codec_kind"] == KIND_ARITHMETIC_STATIC
    assert parsed["streams"][0]["payload"][:4] == b"AQv1"
    assert result.total_bytes == len(result.container_bytes)


def test_jcsp_budget_is_enforced_on_full_container_bytes() -> None:
    qints = np.zeros(64, dtype=np.int8)
    stream = StreamSource(
        name="pose_qints",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    baseline = run_joint_codec_stack(
        streams=[stream],
        byte_budget=10_000,
        admm_max_iters=2,
    )

    with pytest.raises(ValueError, match="JCSP container bytes"):
        run_joint_codec_stack(
            streams=[stream],
            byte_budget=baseline.total_bytes - 1,
            admm_max_iters=2,
        )


def test_unpack_jcsp_container_rejects_trailing_bytes() -> None:
    qints = np.array([0, 1, -1, 2], dtype=np.int8)
    stream = StreamSource(
        name="tiny",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    result = run_sequential_codec_stack(streams=[stream])

    with pytest.raises(ValueError, match="trailing bytes"):
        unpack_jcsp_container(result.container_bytes + b"x")


def test_jcsp_runtime_parity_reports_payload_magic() -> None:
    qints = np.array([0, 1, -1, 2], dtype=np.int8)
    stream = StreamSource(
        name="tiny",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    result = run_sequential_codec_stack(streams=[stream])

    parity = validate_jcsp_container_runtime_parity(result.container_bytes)

    assert parity["schema"] == "jcsp_runtime_loader_parity_v1"
    assert parity["score_claim"] is False
    assert parity["ready_for_runtime_loader"] is True
    assert parity["streams"] == [
        {
            "name": "tiny",
            "codec_kind": KIND_ARITHMETIC_STATIC,
            "payload_magic": "AQv1",
            "actual_bytes": result.streams[0].actual_bytes,
            "runtime_dispatch_checked": True,
        }
    ]


def test_unpack_jcsp_container_rejects_codec_kind_payload_magic_mismatch() -> None:
    qints = np.array([0, 1, -1, 2], dtype=np.int8)
    stream = StreamSource(
        name="tiny",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    result = run_sequential_codec_stack(streams=[stream])
    tampered = bytearray(result.container_bytes)
    name_len_offset = 4 + 2 + 1
    name_len = tampered[name_len_offset]
    codec_kind_offset = name_len_offset + 1 + name_len
    tampered[codec_kind_offset] = KIND_BALLE_HYPERPRIOR

    with pytest.raises(ValueError, match="payload magic"):
        unpack_jcsp_container(bytes(tampered))


def test_jcsp_noop_archive_fixture_is_deterministic_and_loader_consumes_member() -> None:
    archive_a = build_jcsp_noop_archive_fixture()
    archive_b = build_jcsp_noop_archive_fixture()

    assert archive_a == archive_b

    contract = load_jcsp_archive_member_for_runtime(
        archive_bytes=archive_a,
        require_single_member=True,
    )

    assert contract["schema"] == JCSP_ARCHIVE_MEMBER_CONTRACT_SCHEMA
    assert contract["score_claim"] is False
    assert contract["ready_for_runtime_loader"] is True
    assert contract["ready_for_submission_runtime_consumption"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["member_name"] == JCSP_ARCHIVE_MEMBER_NAME
    assert contract["archive_bytes"] == len(archive_a)
    assert contract["archive_members"] == [JCSP_ARCHIVE_MEMBER_NAME]
    assert contract["member_compress_type"] == zipfile.ZIP_STORED
    assert contract["noop_fixture"] is True
    assert contract["jcsp_runtime_parity"]["stream_count"] == 0
    assert contract["jcsp_runtime_parity"]["score_claim"] is False
    runtime_consumption = contract["runtime_consumption_contract"]
    assert runtime_consumption["schema"] == JCSP_SUBMISSION_RUNTIME_CONSUMPTION_SCHEMA
    assert runtime_consumption["required_submission_runtime"] == (
        "submissions/robust_current"
    )
    assert runtime_consumption["required_member_name"] == JCSP_ARCHIVE_MEMBER_NAME
    assert runtime_consumption["consumes_required_member"] is False
    assert runtime_consumption["dispatch_blocker"] == (
        JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER
    )
    assert JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER in (
        runtime_consumption["dispatch_blockers"]
    )
    output_contract = runtime_consumption["contest_output_contract"]
    assert output_contract["schema"] == JCSP_SUBMISSION_RUNTIME_OUTPUT_CONTRACT_SCHEMA
    parity_contract = output_contract["raw_output_parity_contract"]
    assert parity_contract["schema"] == JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA
    assert parity_contract["required_proof_schema"] == (
        JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA
    )
    assert parity_contract["expected_raw_outputs_known"] is False
    assert parity_contract["required_candidate_output_source"] == (
        "jcsp_runtime_bridge_emitted_rawvideo"
    )
    assert parity_contract["preexisting_raw_outputs_are_not_parity_proof"] is True
    assert output_contract["bridge_emits_contest_raw_outputs"] is False
    assert output_contract["output_parity_checked"] is False
    assert output_contract["ready_for_submission_runtime_consumption"] is False
    assert JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER in (
        contract["dispatch_blockers"]
    )
    assert JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER in (
        contract["dispatch_blockers"]
    )
    assert "exact_cuda_auth_eval_missing" in contract["dispatch_blockers"]


def test_jcsp_archive_member_contract_consumes_real_container_member() -> None:
    qints = np.array([0, 1, -1, 2], dtype=np.int8)
    stream = StreamSource(
        name="tiny",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    result = run_sequential_codec_stack(streams=[stream])

    archive = build_jcsp_archive_member(container_bytes=result.container_bytes)
    contract = load_jcsp_archive_member_for_runtime(
        archive_bytes=archive,
        require_single_member=True,
    )

    assert contract["noop_fixture"] is False
    assert contract["member_bytes"] == result.total_bytes
    assert contract["member_sha256"]
    assert contract["jcsp_runtime_parity"]["stream_count"] == 1
    assert contract["jcsp_runtime_parity"]["streams"] == [
        {
            "name": "tiny",
            "codec_kind": KIND_ARITHMETIC_STATIC,
            "payload_magic": "AQv1",
            "actual_bytes": result.streams[0].actual_bytes,
            "runtime_dispatch_checked": True,
        }
    ]


def test_jcsp_archive_member_loader_rejects_local_central_name_mismatch() -> None:
    archive = bytearray(build_jcsp_noop_archive_fixture())
    with zipfile.ZipFile(io.BytesIO(archive), "r") as zf:
        info = zf.getinfo(JCSP_ARCHIVE_MEMBER_NAME)
        name_start = int(info.header_offset) + 30
    archive[name_start] = ord("x")

    with pytest.raises(ValueError, match="central/local name mismatch"):
        load_jcsp_archive_member_for_runtime(
            archive_bytes=bytes(archive),
            require_single_member=True,
        )


def test_jcsp_archive_member_builder_rejects_non_jcsp_payload() -> None:
    with pytest.raises(ValueError, match="bad magic"):
        build_jcsp_archive_member(container_bytes=b"not-jcsp")
