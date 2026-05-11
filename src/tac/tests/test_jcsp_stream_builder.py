"""Tests for ``tac.jcsp_stream_builder``.

Adversarial-coverage angles:
- Quantizer edge cases: empty / zero / nan / non-odd num_levels.
- ARITHMETIC_STATIC happy path: roundtrip preserves shape and bounds.
- RAW_PASSTHROUGH: missing bytes raises; supplied bytes preserved.
- BALLE_HYPERPRIOR: missing codec raises.
- Marginal validation: nan / inf raises.
- model_to_stream_sources: missing tensor raises; RAW without payload raises;
  BALLE without codec raises; aligned (streams, specs) lengths.
- End-to-end: builder output flows cleanly into run_sequential_codec_stack
  and produces a valid JCSP container.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from tac.jcsp_stream_builder import (
    JCSP_RAWVIDEO_STREAM_SOURCE_SCHEMA,
    JCSP_STREAM_SOURCE_ARCHIVE_MEMBER_SCHEMA,
    JCSP_STREAM_SOURCE_DRY_RUN_SCHEMA,
    JCSP_STREAM_SOURCE_LOCAL_ARCHIVE_MEMBER_SCHEMA,
    jcsp_stream_source_archive_member,
    jcsp_stream_source_dry_run_metadata,
    jcsp_stream_source_local_archive_member,
    load_jcsp_score_marginals,
    model_to_stream_sources,
    quantize_tensor_symmetric,
    rawvideo_bytes_to_stream_source,
    tensor_to_stream_source,
)
from tac.joint_codec_stack_orchestrator import (
    JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
    JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA,
    JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA,
    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
    JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER,
    KIND_ARITHMETIC_STATIC,
    KIND_BALLE_HYPERPRIOR,
    KIND_RAW_PASSTHROUGH,
    run_sequential_codec_stack,
    unpack_jcsp_container,
)


def test_quantize_tensor_symmetric_roundtrip_preserves_bounds() -> None:
    rng = np.random.default_rng(123)
    tensor = rng.normal(size=128).astype(np.float32)
    qints, num_symbols, offset, scale = quantize_tensor_symmetric(
        tensor, num_levels=15
    )
    assert qints.dtype == np.int8
    assert qints.shape == (128,)
    assert num_symbols == 15
    assert offset == 7
    assert scale > 0.0
    assert int(qints.min()) >= -7
    assert int(qints.max()) <= 7
    # Reconstructed tensor sits in the original abs range
    reconstructed = qints.astype(np.float32) * scale
    assert np.max(np.abs(reconstructed)) <= np.max(np.abs(tensor)) + 1e-6


def test_quantize_tensor_symmetric_zero_tensor_yields_zero_qints() -> None:
    qints, num_symbols, offset, scale = quantize_tensor_symmetric(
        np.zeros(8, dtype=np.float32)
    )
    assert np.all(qints == 0)
    assert num_symbols == 15
    assert offset == 7
    assert scale == 1.0


def test_quantize_tensor_symmetric_empty_tensor_raises() -> None:
    with pytest.raises(ValueError, match="empty tensor"):
        quantize_tensor_symmetric(np.zeros(0, dtype=np.float32))


def test_quantize_tensor_symmetric_nan_tensor_raises() -> None:
    bad = np.array([1.0, float("nan"), 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="nan/inf"):
        quantize_tensor_symmetric(bad)


def test_quantize_tensor_symmetric_even_num_levels_raises() -> None:
    with pytest.raises(ValueError, match="odd"):
        quantize_tensor_symmetric(np.ones(4, dtype=np.float32), num_levels=8)


def test_quantize_tensor_symmetric_torch_tensor_supported() -> None:
    t = torch.linspace(-2.0, 2.0, 9, dtype=torch.float32)
    qints, num_symbols, offset, scale = quantize_tensor_symmetric(t)
    assert qints.dtype == np.int8
    assert qints.shape == (9,)
    assert num_symbols == 15
    assert offset == 7


def test_tensor_to_stream_source_arithmetic_static_happy() -> None:
    src = tensor_to_stream_source(
        torch.linspace(-1.0, 1.0, 16, dtype=torch.float32),
        name="renderer.weight",
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    assert src.name == "renderer.weight"
    assert src.codec_kind == KIND_ARITHMETIC_STATIC
    assert src.qints.dtype == np.int8
    assert src.qints.shape == (16,)
    assert src.num_symbols == 15
    assert src.offset == 7
    assert src.score_per_byte_marginal == pytest.approx(1e-6)


def test_tensor_to_stream_source_raw_passthrough_without_bytes_raises() -> None:
    with pytest.raises(ValueError, match="raw_passthrough_bytes"):
        tensor_to_stream_source(
            np.zeros(4, dtype=np.float32),
            name="wet",
            codec_kind=KIND_RAW_PASSTHROUGH,
            score_per_byte_marginal=0.0,
        )


def test_tensor_to_stream_source_raw_passthrough_preserves_payload() -> None:
    payload = b"AV1\x00\x01\x02"
    src = tensor_to_stream_source(
        np.zeros(0, dtype=np.float32),  # ignored for RAW
        name="masks.mkv",
        codec_kind=KIND_RAW_PASSTHROUGH,
        score_per_byte_marginal=0.0,
        raw_passthrough_bytes=payload,
    )
    assert src.codec_kind == KIND_RAW_PASSTHROUGH
    assert src.raw_passthrough_bytes == payload
    assert src.qints.size == 0


def test_rawvideo_bytes_to_stream_source_builds_production_aq_stream() -> None:
    payload = bytes(range(36))
    src = rawvideo_bytes_to_stream_source(
        payload,
        name="route/video.raw",
        score_per_byte_marginal=0.0,
    )
    assert JCSP_RAWVIDEO_STREAM_SOURCE_SCHEMA == (
        "jcsp_rawvideo_stream_source_contract_v1"
    )
    assert src.name == "route/video.raw"
    assert src.codec_kind == KIND_ARITHMETIC_STATIC
    assert src.num_symbols == 256
    assert src.offset == 0
    assert src.qints.dtype == np.uint8
    assert src.qints.tobytes() == payload

    result = run_sequential_codec_stack(streams=[src])
    parsed = unpack_jcsp_container(result.container_bytes)
    [stream] = parsed["streams"]
    assert stream["name"] == "route/video.raw"
    assert stream["codec_kind"] == KIND_ARITHMETIC_STATIC
    assert stream["payload_magic"] in (b"AQv1", b"AQc1")


def test_rawvideo_bytes_to_stream_source_rejects_fixture_shapes() -> None:
    with pytest.raises(ValueError, match=r"end with \.raw"):
        rawvideo_bytes_to_stream_source(b"\x00\x01\x02", name="route/video.hevc")
    with pytest.raises(ValueError, match="divisible by 3"):
        rawvideo_bytes_to_stream_source(b"\x00\x01", name="route/video.raw")
    with pytest.raises(ValueError, match="unsafe"):
        rawvideo_bytes_to_stream_source(b"\x00\x01\x02", name="../video.raw")


def test_tensor_to_stream_source_balle_without_codec_raises() -> None:
    with pytest.raises(ValueError, match="balle_codec"):
        tensor_to_stream_source(
            np.zeros(8, dtype=np.float32),
            name="big.weight",
            codec_kind=KIND_BALLE_HYPERPRIOR,
            score_per_byte_marginal=1e-6,
        )


def test_tensor_to_stream_source_invalid_codec_kind_raises() -> None:
    with pytest.raises(ValueError, match="invalid codec_kind"):
        tensor_to_stream_source(
            np.zeros(4, dtype=np.float32),
            name="weight",
            codec_kind=99,
            score_per_byte_marginal=0.0,
        )


def test_tensor_to_stream_source_nonfinite_marginal_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        tensor_to_stream_source(
            np.zeros(4, dtype=np.float32),
            name="weight",
            codec_kind=KIND_ARITHMETIC_STATIC,
            score_per_byte_marginal=float("nan"),
        )


def test_model_to_stream_sources_simple_state_dict_aligned_lengths() -> None:
    state_dict = {
        "a.weight": torch.linspace(-1.0, 1.0, 16, dtype=torch.float32),
        "b.weight": torch.zeros(8, dtype=torch.float32),
    }
    streams, specs = model_to_stream_sources(
        state_dict,
        score_marginals={"a.weight": 1e-6, "b.weight": 2e-6},
    )
    assert len(streams) == len(specs) == 2
    names = [s.name for s in streams]
    spec_names = [sp.name for sp in specs]
    assert names == spec_names
    assert all(s.codec_kind == KIND_ARITHMETIC_STATIC for s in streams)


def test_model_to_stream_sources_accepts_structured_marginal_rows() -> None:
    state_dict = {
        "a.weight": torch.linspace(-1.0, 1.0, 16, dtype=torch.float32),
    }
    streams, specs = model_to_stream_sources(
        state_dict,
        score_marginals={
            "a.weight": {
                "score_per_byte_marginal": 1e-6,
                "source": "reports/component_sensitivity.json",
                "evidence_grade": "empirical",
                "scorer_term_targeted": "seg",
            }
        },
    )
    assert len(streams) == len(specs) == 1
    assert streams[0].score_per_byte_marginal == pytest.approx(1e-6)
    assert specs[0].score_marginal_source == "reports/component_sensitivity.json"
    assert specs[0].scorer_term_targeted == "seg"


def test_model_to_stream_sources_raw_passthrough_without_payload_raises() -> None:
    state_dict = {
        "wet.mkv": torch.zeros(4, dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="RAW_PASSTHROUGH stream"):
        model_to_stream_sources(
            state_dict,
            score_marginals={"wet.mkv": 0.0},
            codec_overrides={"wet.mkv": KIND_RAW_PASSTHROUGH},
            wet_streams=["wet.mkv"],
        )


def test_model_to_stream_sources_raw_passthrough_with_payload_preserves() -> None:
    state_dict = {
        "wet.mkv": torch.zeros(4, dtype=torch.float32),
        "weight": torch.linspace(-1.0, 1.0, 8, dtype=torch.float32),
    }
    payload = b"AV1\x00\x12"
    streams, _specs = model_to_stream_sources(
        state_dict,
        score_marginals={"wet.mkv": 0.0, "weight": 1e-6},
        codec_overrides={"wet.mkv": KIND_RAW_PASSTHROUGH},
        wet_streams=["wet.mkv"],
        raw_passthrough_bytes_by_name={"wet.mkv": payload},
    )
    by_name = {s.name: s for s in streams}
    assert by_name["wet.mkv"].codec_kind == KIND_RAW_PASSTHROUGH
    assert by_name["wet.mkv"].raw_passthrough_bytes == payload
    assert by_name["weight"].codec_kind == KIND_ARITHMETIC_STATIC


def test_model_to_stream_sources_end_to_end_into_jcsp_container() -> None:
    state_dict = {
        "a.weight": torch.linspace(-1.0, 1.0, 32, dtype=torch.float32),
        "b.weight": torch.zeros(16, dtype=torch.float32),
    }
    streams, _specs = model_to_stream_sources(
        state_dict,
        score_marginals={"a.weight": 1e-6, "b.weight": 2e-6},
    )
    result = run_sequential_codec_stack(streams=streams)
    parsed = unpack_jcsp_container(result.container_bytes)
    assert [s["name"] for s in parsed["streams"]] == ["a.weight", "b.weight"]
    assert all(
        s["codec_kind"] == KIND_ARITHMETIC_STATIC for s in parsed["streams"]
    )
    assert result.total_bytes == len(result.container_bytes)


def test_jcsp_stream_source_archive_member_closes_real_jcsp_member(
    tmp_path,
) -> None:
    artifact = tmp_path / "marginals.json"
    artifact.write_text(
        json.dumps(
            {
                "score_marginals": {
                    "a.weight": 1e-6,
                    "b.weight": 2e-6,
                }
            },
            sort_keys=True,
        )
    )
    state_dict = {
        "b.weight": torch.zeros(8, dtype=torch.float32),
        "a.weight": torch.linspace(-1.0, 1.0, 16, dtype=torch.float32),
    }
    archive_path = tmp_path / "jcsp_archive_member.zip"

    archive_a, manifest_a = jcsp_stream_source_archive_member(
        state_dict,
        score_marginals_path=artifact,
        archive_path_for_manifest=archive_path,
    )
    archive_b, manifest_b = jcsp_stream_source_archive_member(
        state_dict,
        score_marginals_path=artifact,
        archive_path_for_manifest=archive_path,
    )

    assert archive_a == archive_b
    assert manifest_a == manifest_b
    assert manifest_a["schema"] == JCSP_STREAM_SOURCE_ARCHIVE_MEMBER_SCHEMA
    assert manifest_a["score_claim"] is False
    assert manifest_a["dispatch_attempted"] is False
    assert manifest_a["ready_for_runtime_loader"] is True
    assert manifest_a["ready_for_submission_runtime_consumption"] is False
    assert manifest_a["ready_for_exact_eval_dispatch"] is False
    assert manifest_a["archive_member_payload_kind"] == "jcsp_runtime_container"
    assert manifest_a["archive_path"] == str(archive_path)
    assert manifest_a["archive_bytes"] == len(archive_a)
    assert manifest_a["archive_sha256"] == hashlib.sha256(archive_a).hexdigest()
    assert manifest_a["candidate_archive_sha256"] == manifest_a["archive_sha256"]
    assert manifest_a["byte_closed_archive_member"] is True
    assert manifest_a["runtime_payloads_encoded"] is True
    assert manifest_a["runtime_loader_parity"]["schema"] == (
        "jcsp_runtime_loader_parity_v1"
    )
    assert [row["name"] for row in manifest_a["runtime_loader_parity"]["streams"]] == [
        "a.weight",
        "b.weight",
    ]
    assert all(
        row["payload_magic"] == "AQv1"
        for row in manifest_a["runtime_loader_parity"]["streams"]
    )
    reconciliation = manifest_a["stream_archive_byte_reconciliation"]
    assert reconciliation["stream_payload_bytes_reconciled"] is True
    assert all(
        row["bytes_charged_reconciled"] is True
        for row in reconciliation["per_stream"]
    )
    assert "stream_bytes_charged_reconciliation_missing" not in (
        manifest_a["dispatch_blockers"]
    )
    assert JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER in (
        manifest_a["dispatch_blockers"]
    )
    assert JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER in (
        manifest_a["dispatch_blockers"]
    )
    assert JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER in (
        manifest_a["runtime"]["blockers"]
    )
    output_contract = manifest_a["runtime_consumption_contract"][
        "contest_output_contract"
    ]
    parity_contract = output_contract["raw_output_parity_contract"]
    assert parity_contract["schema"] == JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA
    assert parity_contract["required_proof_schema"] == (
        JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA
    )
    assert parity_contract["preexisting_raw_outputs_are_not_parity_proof"] is True
    assert output_contract["bridge_emits_contest_raw_outputs"] is False
    assert output_contract["output_parity_checked"] is False


def test_load_jcsp_score_marginals_rejects_duplicate_json_keys(tmp_path) -> None:
    artifact = tmp_path / "marginals.json"
    artifact.write_text('{"score_marginals":{"a.weight":1e-6,"a.weight":2e-6}}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_jcsp_score_marginals(artifact)


def test_jcsp_stream_source_dry_run_metadata_is_deterministic(tmp_path) -> None:
    artifact = tmp_path / "marginals.json"
    artifact.write_text(
        json.dumps(
            {
                "score_marginals": {
                    "a.weight": {
                        "score_per_byte_marginal": 1e-6,
                        "source": "reports/component_sensitivity.json",
                        "evidence_grade": "empirical",
                        "scorer_term_targeted": "seg",
                    },
                    "b.weight": 2e-6,
                }
            },
            sort_keys=True,
        )
    )
    state_dict = {
        "b.weight": torch.zeros(8, dtype=torch.float32),
        "a.weight": torch.linspace(-1.0, 1.0, 16, dtype=torch.float32),
    }

    first = jcsp_stream_source_dry_run_metadata(
        state_dict,
        score_marginals_path=artifact,
    )
    second = jcsp_stream_source_dry_run_metadata(
        state_dict,
        score_marginals_path=artifact,
    )

    assert first == second
    assert first["schema"] == JCSP_STREAM_SOURCE_DRY_RUN_SCHEMA
    assert first["score_claim"] is False
    assert first["dispatch_attempted"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["archive_bytes_written"] is False
    assert first["container_bytes_built"] is False
    assert first["stream_source_objects_built"] is False
    assert first["stream_count"] == 2
    assert [row["name"] for row in first["streams"]] == [
        "a.weight",
        "b.weight",
    ]
    assert first["streams"][0]["qint_count"] == 16
    assert first["streams"][0]["score_per_byte_marginal"] == pytest.approx(1e-6)
    assert "dry_run_only_no_archive_bytes_written" in (
        first["streams"][0]["dispatch_blockers"]
    )


def test_jcsp_stream_source_local_archive_member_is_byte_closed_and_blocked(
    tmp_path,
) -> None:
    artifact = tmp_path / "marginals.json"
    artifact.write_text(
        json.dumps(
            {
                "score_marginals": {
                    "a.weight": 1e-6,
                    "b.weight": 2e-6,
                }
            },
            sort_keys=True,
        )
    )
    state_dict = {
        "b.weight": torch.zeros(2, dtype=torch.float32),
        "a.weight": torch.linspace(-1.0, 1.0, 6, dtype=torch.float32),
    }

    archive_a, contract_a = jcsp_stream_source_local_archive_member(
        state_dict,
        score_marginals_path=artifact,
        preview_bytes_per_stream=4,
    )
    archive_b, contract_b = jcsp_stream_source_local_archive_member(
        state_dict,
        score_marginals_path=artifact,
        preview_bytes_per_stream=4,
    )

    assert archive_a == archive_b
    assert contract_a == contract_b
    assert contract_a["score_claim"] is False
    assert contract_a["ready_for_local_skeleton_loader"] is True
    assert contract_a["ready_for_runtime_loader"] is False
    assert contract_a["ready_for_exact_eval_dispatch"] is False
    assert contract_a["byte_closed_archive_member"] is True
    assert contract_a["single_member_no_sidecars"] is True
    assert contract_a["archive_sha256"] == hashlib.sha256(archive_a).hexdigest()
    assert contract_a["member_name"] == "jcsp.bin"
    assert JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER in contract_a["dispatch_blockers"]

    manifest = contract_a["skeleton_manifest"]
    assert manifest["schema"] == JCSP_STREAM_SOURCE_LOCAL_ARCHIVE_MEMBER_SCHEMA
    assert manifest["container_magic"] == "JCSK"
    assert manifest["container_version"] == 1
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_bytes_written"] is True
    assert manifest["runtime_payloads_encoded"] is False
    assert manifest["byte_manifest"]["source_stream_bytes"] == 8
    assert manifest["byte_manifest"]["encoded_preview_bytes"] == 6
    assert [row["name"] for row in manifest["streams"]] == [
        "a.weight",
        "b.weight",
    ]
    first_preview = manifest["streams"][0]["preview"]
    assert first_preview["source_bytes_kind"] == (
        "quantized_qint_int8_prefix_not_codec_payload"
    )
    assert first_preview["preview_bytes"] == 4
    assert len(first_preview["preview_hex"]) == 8
    assert first_preview["preview_truncated"] is True
    assert "full_codec_payload_not_encoded" in (
        manifest["streams"][0]["dispatch_blockers"]
    )
