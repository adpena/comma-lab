from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from tac.joint_codec_stack_orchestrator import (
    JCSP_MODEL_STREAM_ARCHIVE_READINESS_SCHEMA,
    JCSP_STREAM_METADATA_SCHEMA,
    KIND_ARITHMETIC_STATIC,
    KIND_BALLE_HYPERPRIOR,
    KIND_RAW_PASSTHROUGH,
    StreamSource,
    build_jcsp_archive_member,
    jcsp_model_stream_archive_readiness,
    jcsp_stream_specs_manifest,
    model_to_jcsp_streams,
    run_sequential_codec_stack,
)


def test_model_to_jcsp_streams_records_tensor_contracts_without_score_claims() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.BatchNorm1d(3),
    )

    streams = model_to_jcsp_streams(model)
    by_name = {stream.name: stream for stream in streams}

    assert "0.weight" in by_name
    weight = by_name["0.weight"]
    assert len(weight.stream_id) == 64
    assert weight.decomposition_index == [stream.name for stream in streams].index(
        "0.weight"
    )
    assert weight.tensor_shape == (3, 4)
    assert weight.tensor_dtype == "torch.float32"
    assert weight.raw_bytes == 3 * 4 * 4
    assert weight.byte_estimate == 3 * 4 * 2
    assert weight.byte_estimate_source == "fp16_quantization_floor_estimate"
    assert weight.bytes_charged == weight.byte_estimate
    assert weight.bytes_charged_source == "planning_estimate_not_archive_closed"
    assert weight.score_per_byte_marginal == 0.0
    assert weight.score_marginal_source == "missing_score_marginal"
    assert "balle_hyperprior_2018" in weight.research_basis_ids
    assert "foveated_telepresence_2025" in weight.research_basis_ids
    assert "score_affecting_sidecars_forbidden" in weight.constraint_tags
    assert "score_marginal_artifact_missing" in weight.dispatch_blockers
    assert "byte_closed_archive_member_missing" in weight.dispatch_blockers
    assert "refuse_dispatch_if_decode_validation_missing" in weight.fail_closed_criteria
    assert "refuse_dispatch_if_sidecar_dependency_detected" in weight.fail_closed_criteria

    assert by_name["1.num_batches_tracked"].codec_kind == KIND_ARITHMETIC_STATIC


def test_model_to_jcsp_streams_accepts_marginals_overrides_and_wet_streams() -> None:
    state = {
        "large.weight": torch.zeros((4096, 4), dtype=torch.float32),
        "pose.bias": torch.ones((6,), dtype=torch.float32),
    }

    streams = model_to_jcsp_streams(
        state,
        score_marginals={
            "large.weight": {
                "score_per_byte_marginal": 1.25e-5,
                "source": "reports/component_sensitivity.json",
                "evidence_grade": "empirical",
                "scorer_term_targeted": "pose",
                "constraint_tags": ("additive_distortion_prior",),
            },
        },
        codec_overrides={"pose.bias": KIND_RAW_PASSTHROUGH},
        wet_streams=("pose.*",),
    )
    by_name = {stream.name: stream for stream in streams}

    large = by_name["large.weight"]
    assert large.codec_kind == KIND_BALLE_HYPERPRIOR
    assert large.score_per_byte_marginal == 1.25e-5
    assert large.score_marginal_source == "reports/component_sensitivity.json"
    assert large.score_marginal_evidence == "empirical"
    assert large.scorer_term_targeted == "pose"
    assert "balle_hyperprior_2018" in large.research_basis_ids
    assert "additive_distortion_prior" in large.constraint_tags
    assert "score_marginal_artifact_missing" not in large.dispatch_blockers

    wet = by_name["pose.bias"]
    assert wet.codec_kind == KIND_RAW_PASSTHROUGH
    assert wet.byte_estimate == wet.raw_bytes
    assert wet.byte_estimate_source == "raw_passthrough_tensor_bytes"
    assert wet.score_per_byte_marginal == 0.0
    assert wet.score_marginal_source == "wet_stream"
    assert "wet_stream_do_not_perturb" in wet.constraint_tags
    assert "wet_stream_raw_passthrough_required" in wet.constraint_tags
    assert "score_marginal_artifact_missing" not in wet.dispatch_blockers


def test_model_to_jcsp_streams_fails_closed_for_wet_stream_without_raw_override() -> None:
    state = {"pose.bias": torch.ones((6,), dtype=torch.float32)}

    with pytest.raises(ValueError, match="explicit RAW_PASSTHROUGH"):
        model_to_jcsp_streams(state, wet_streams=("pose.*",))

    with pytest.raises(ValueError, match="explicit RAW_PASSTHROUGH"):
        model_to_jcsp_streams(
            state,
            codec_overrides={"pose.bias": KIND_ARITHMETIC_STATIC},
            wet_streams=("pose.*",),
        )


def test_model_to_jcsp_streams_is_order_invariant_for_mapping_inputs() -> None:
    forward = {
        "z.bias": torch.ones((2,), dtype=torch.float32),
        "a.weight": torch.zeros((3, 2), dtype=torch.float32),
    }
    reverse = {
        "a.weight": torch.zeros((3, 2), dtype=torch.float32),
        "z.bias": torch.ones((2,), dtype=torch.float32),
    }

    forward_streams = model_to_jcsp_streams(forward)
    reverse_streams = model_to_jcsp_streams(reverse)

    assert [stream.name for stream in forward_streams] == ["a.weight", "z.bias"]
    assert forward_streams == reverse_streams
    assert [stream.decomposition_index for stream in forward_streams] == [0, 1]
    assert forward_streams[0].stream_id == reverse_streams[0].stream_id


def test_jcsp_stream_specs_manifest_is_deterministic_json_ready() -> None:
    state = {
        "z.bias": torch.ones((2,), dtype=torch.float32),
        "a.weight": torch.zeros((3, 2), dtype=torch.float32),
    }

    streams = model_to_jcsp_streams(state)
    manifest_a = jcsp_stream_specs_manifest(streams)
    manifest_b = jcsp_stream_specs_manifest(reversed(streams))

    assert manifest_a == manifest_b
    assert manifest_a["schema"] == JCSP_STREAM_METADATA_SCHEMA
    assert manifest_a["score_claim"] is False
    assert manifest_a["dispatch_attempted"] is False
    assert manifest_a["sidecar_policy"]["score_affecting_sidecars_allowed"] is False
    assert manifest_a["sidecar_policy"]["required_archive_members"] == ["jcsp.bin"]
    assert manifest_a["stream_count"] == 2
    assert [row["name"] for row in manifest_a["streams"]] == [
        "a.weight",
        "z.bias",
    ]
    assert len(manifest_a["manifest_sha256"]) == 64
    encoded_a = json.dumps(
        manifest_a,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    encoded_b = json.dumps(
        manifest_b,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert encoded_a == encoded_b


def test_jcsp_model_stream_archive_readiness_closes_single_member_bytes() -> None:
    specs = model_to_jcsp_streams(
        {
            "z.bias": torch.ones((2,), dtype=torch.float32),
            "a.weight": torch.zeros((2,), dtype=torch.float32),
        },
        score_marginals={
            "a.weight": 1.0e-6,
            "z.bias": 2.0e-6,
        },
    )
    result = run_sequential_codec_stack(
        streams=[
            StreamSource(
                name="a.weight",
                qints=np.array([0, 1], dtype=np.int8),
                num_symbols=3,
                offset=1,
                codec_kind=KIND_ARITHMETIC_STATIC,
                score_per_byte_marginal=1.0e-6,
            ),
            StreamSource(
                name="z.bias",
                qints=np.array([1, 0], dtype=np.int8),
                num_symbols=3,
                offset=1,
                codec_kind=KIND_ARITHMETIC_STATIC,
                score_per_byte_marginal=2.0e-6,
            ),
        ]
    )
    archive = build_jcsp_archive_member(container_bytes=result.container_bytes)

    readiness = jcsp_model_stream_archive_readiness(
        streams=specs,
        archive_bytes=archive,
    )

    assert readiness["schema"] == JCSP_MODEL_STREAM_ARCHIVE_READINESS_SCHEMA
    assert readiness["score_claim"] is False
    assert readiness["dispatch_attempted"] is False
    assert readiness["ready_for_runtime_loader"] is True
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["byte_closed_archive_member"] is True
    assert readiness["single_member_no_sidecars"] is True
    assert readiness["runtime_stream_order_matches_manifest"] is True
    assert [row["name"] for row in readiness["streams"]] == [
        "a.weight",
        "z.bias",
    ]
    assert readiness["streams"][0]["archive_actual_bytes"] == (
        result.streams[0].actual_bytes
    )
    assert readiness["streams"][0]["archive_actual_bytes_source"] == (
        "jcsp_container_member_payload"
    )
    assert "byte_closed_archive_member_missing" not in readiness["dispatch_blockers"]
    assert "qint_or_exact_wire_stream_missing" not in readiness["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in readiness["dispatch_blockers"]
    assert "stream_bytes_charged_reconciliation_missing" in (
        readiness["streams"][0]["dispatch_blockers"]
    )


def test_jcsp_model_stream_archive_readiness_rejects_nondeterministic_order() -> None:
    specs = model_to_jcsp_streams(
        {
            "z.bias": torch.ones((2,), dtype=torch.float32),
            "a.weight": torch.zeros((2,), dtype=torch.float32),
        }
    )
    result = run_sequential_codec_stack(
        streams=[
            StreamSource(
                name="z.bias",
                qints=np.array([1, 0], dtype=np.int8),
                num_symbols=3,
                offset=1,
                codec_kind=KIND_ARITHMETIC_STATIC,
                score_per_byte_marginal=2.0e-6,
            ),
            StreamSource(
                name="a.weight",
                qints=np.array([0, 1], dtype=np.int8),
                num_symbols=3,
                offset=1,
                codec_kind=KIND_ARITHMETIC_STATIC,
                score_per_byte_marginal=1.0e-6,
            ),
        ]
    )
    archive = build_jcsp_archive_member(container_bytes=result.container_bytes)

    with pytest.raises(ValueError, match="archive stream order"):
        jcsp_model_stream_archive_readiness(
            streams=specs,
            archive_bytes=archive,
        )


def test_jcsp_stream_specs_manifest_fails_closed_on_duplicate_ids() -> None:
    streams = model_to_jcsp_streams(
        {
            "a.weight": torch.zeros((3, 2), dtype=torch.float32),
            "z.bias": torch.ones((2,), dtype=torch.float32),
        }
    )

    corrupted = [
        streams[0],
        replace(streams[1], stream_id=streams[0].stream_id),
    ]
    with pytest.raises(ValueError, match="unique stream ids"):
        jcsp_stream_specs_manifest(corrupted)


def test_model_to_jcsp_streams_fails_closed_on_duplicate_stream_names() -> None:
    class DuplicateNames:
        def named_parameters(self):
            return [
                ("dup.weight", torch.nn.Parameter(torch.zeros((1,)))),
                ("dup.weight", torch.nn.Parameter(torch.ones((1,)))),
            ]

    with pytest.raises(ValueError, match=r"duplicates: dup\.weight"):
        model_to_jcsp_streams(DuplicateNames(), include_buffers=False)


def test_model_to_jcsp_streams_rejects_stale_overrides_and_nonfinite_marginals() -> None:
    state = {"a.weight": torch.zeros((2,), dtype=torch.float32)}

    with pytest.raises(ValueError, match=r"unknown streams: missing\.weight"):
        model_to_jcsp_streams(
            state,
            codec_overrides={"missing.weight": KIND_RAW_PASSTHROUGH},
        )
    with pytest.raises(ValueError, match="invalid codec override"):
        model_to_jcsp_streams(
            state,
            codec_overrides={"a.weight": "not-an-int"},
        )
    with pytest.raises(ValueError, match=r"unknown streams: stale\.bias"):
        model_to_jcsp_streams(
            state,
            score_marginals={"stale.bias": 1.0e-6},
        )
    with pytest.raises(ValueError, match="must be finite"):
        model_to_jcsp_streams(
            state,
            score_marginals={"a.weight": math.nan},
        )
