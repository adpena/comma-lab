# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.jcsp_stream_manifest import (
    JCSP_STREAM_MANIFEST_ROW_SCHEMA,
    build_jcsp_stream_manifest_row,
)
from tac.joint_codec_stack_orchestrator import (
    JCSP_ARCHIVE_MEMBER_NAME,
    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
    JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER,
    KIND_ARITHMETIC_STATIC,
    StreamSource,
    build_jcsp_archive_member,
    build_jcsp_noop_archive_fixture,
    run_sequential_codec_stack,
)


def _one_stream_archive_bytes() -> bytes:
    stream = StreamSource(
        name="renderer.weight",
        qints=np.array([0, 1, -1, 2, -2, 3, -3], dtype=np.int8),
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    result = run_sequential_codec_stack(streams=[stream])
    return build_jcsp_archive_member(container_bytes=result.container_bytes)


def test_jcsp_stream_manifest_row_is_deterministic_and_fail_closed() -> None:
    archive_bytes = _one_stream_archive_bytes()

    first = build_jcsp_stream_manifest_row(
        archive_bytes=archive_bytes,
        archive_path="experiments/results/jcsp_candidate/archive.zip",
        source_manifest_path="experiments/results/jcsp_candidate/manifest.json",
        source_manifest_sha256="0" * 64,
    )
    second = build_jcsp_stream_manifest_row(
        archive_bytes=archive_bytes,
        archive_path="experiments/results/jcsp_candidate/archive.zip",
        source_manifest_path="experiments/results/jcsp_candidate/manifest.json",
        source_manifest_sha256="0" * 64,
    )

    assert first == second
    assert first["schema"] == JCSP_STREAM_MANIFEST_ROW_SCHEMA
    assert first["score_claim"] is False
    assert first["dispatch_attempted"] is False
    assert first["ready_for_runtime_loader"] is True
    assert first["ready_for_submission_runtime_consumption"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["byte_closed_archive_member"] is True
    assert first["single_member_no_sidecars"] is True
    assert first["archive_bytes"] == len(archive_bytes)
    assert first["archive_sha256"] == hashlib.sha256(archive_bytes).hexdigest()
    assert first["member_name"] == JCSP_ARCHIVE_MEMBER_NAME
    assert first["member_bytes"] > first["stream_payload_bytes"]
    assert first["member_container_overhead_bytes"] > 0
    assert first["archive_wrapper_overhead_bytes"] > 0
    assert len(first["manifest_row_sha256"]) == 64

    [stream] = first["streams"]
    assert stream["name"] == "renderer.weight"
    assert stream["codec_kind"] == KIND_ARITHMETIC_STATIC
    assert stream["actual_bytes"] > 0
    assert len(stream["payload_sha256"]) == 64
    assert stream["payload_magic"] in {"AQv1", "AQc1"}
    assert stream["runtime_dispatch_checked"] is True

    runtime = first["runtime_consumer"]
    assert runtime["detects_required_member"] is True
    assert runtime["consumes_required_member"] is False
    assert runtime["ready_for_submission_runtime_consumption"] is False
    assert JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER in runtime["dispatch_blockers"]
    assert JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER in runtime["dispatch_blockers"]
    assert JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER in first["dispatch_blockers"]
    assert JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER in first["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in first["dispatch_blockers"]


def test_jcsp_stream_manifest_row_refuses_noop_fixture() -> None:
    with pytest.raises(ValueError, match="at least one stream payload"):
        build_jcsp_stream_manifest_row(
            archive_bytes=build_jcsp_noop_archive_fixture()
        )


def test_jcsp_stream_manifest_row_refuses_missing_byte_evidence() -> None:
    with pytest.raises(ValueError, match="archive_bytes must be non-empty"):
        build_jcsp_stream_manifest_row(archive_bytes=b"")
