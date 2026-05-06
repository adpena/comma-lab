from __future__ import annotations

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
