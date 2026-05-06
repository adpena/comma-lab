from __future__ import annotations

import math

import pytest
import torch

from tac.joint_codec_stack_orchestrator import (
    KIND_ARITHMETIC_STATIC,
    KIND_BALLE_HYPERPRIOR,
    KIND_RAW_PASSTHROUGH,
    model_to_jcsp_streams,
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
    assert weight.decomposition_index == [stream.name for stream in streams].index("0.weight")
    assert weight.tensor_shape == (3, 4)
    assert weight.tensor_dtype == "torch.float32"
    assert weight.raw_bytes == 3 * 4 * 4
    assert weight.byte_estimate == 3 * 4 * 2
    assert weight.bytes_charged == weight.byte_estimate
    assert weight.score_per_byte_marginal == 0.0
    assert weight.score_marginal_source == "missing_score_marginal"
    assert "balle_hyperprior_2018" in weight.research_basis_ids
    assert "foveated_telepresence_2025" in weight.research_basis_ids
    assert "score_marginal_artifact_missing" in weight.dispatch_blockers
    assert "refuse_dispatch_if_decode_validation_missing" in weight.fail_closed_criteria

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
    assert wet.score_per_byte_marginal == 0.0
    assert wet.score_marginal_source == "wet_stream"
    assert "wet_stream_do_not_perturb" in wet.constraint_tags
    assert "wet_stream_requires_explicit_override" in wet.dispatch_blockers


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
    with pytest.raises(ValueError, match="must be finite"):
        model_to_jcsp_streams(
            state,
            score_marginals={"a.weight": math.nan},
        )
