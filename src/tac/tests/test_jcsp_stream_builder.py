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

import json

import numpy as np
import pytest
import torch

from tac.jcsp_stream_builder import (
    JCSP_STREAM_SOURCE_DRY_RUN_SCHEMA,
    jcsp_stream_source_dry_run_metadata,
    load_jcsp_score_marginals,
    model_to_stream_sources,
    quantize_tensor_symmetric,
    tensor_to_stream_source,
)
from tac.joint_codec_stack_orchestrator import (
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
