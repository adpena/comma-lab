# SPDX-License-Identifier: MIT
"""Acceptance and fail-closed tests for the semantic pipeline stage contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.semantic_pipeline import (
    FullPipelineConfig,
    PipelineBlocked,
    SemanticPipeline,
    probe_clip,
    require_device,
)
from tac.semantic_pipeline.contracts import atomic_copy, run_payload_stage
from tac.semantic_pipeline.pipeline import DEFAULT_VIDEO, FINAL_BYTES, FINAL_SHA256

TEST_STORE = Path("/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/acceptance_v1")


def test_auto_config_probe_matches_real_zero_clip() -> None:
    clip = probe_clip(DEFAULT_VIDEO)
    assert clip.frame_count == 1200
    assert (clip.width, clip.height) == (1164, 874)
    assert clip.sequence_length == 2
    assert clip.pair_count == 600
    assert clip.trailing_frames == 0
    assert clip.first_frame_rgb_sha256 == "3afa2e8f58a65805a1d2daacefd0af7781206fd97f00d3a4a5de6a2e8a3e0bff"


def test_replay_mode_revalidates_exact_afr1_payload() -> None:
    result = SemanticPipeline(
        FullPipelineConfig(
            mode="replay",
            device="cpu",
            store=TEST_STORE,
            resume=True,
        )
    ).run()
    assert result["status"] == "PASS"
    assert result["archive"]["sha256"] == FINAL_SHA256
    assert result["archive"]["bytes"] == FINAL_BYTES
    assert result["fresh_rebuild_executed"] is False
    assert len(result["stages"]) == 6
    assert [row["stage"] for row in result["stages"]] == [
        "fx5",
        "dx2",
        "gb1_pointer",
        "gb1_joint",
        "lb1",
        "afr1",
    ]


def test_stage_boundary_resume_is_byte_identical() -> None:
    root = TEST_STORE / "resume_contract"
    source = root / "source.bin"
    destination = root / "retained" / "destination.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        source.write_bytes(bytes(range(256)) * 4)

    def action() -> None:
        atomic_copy(source, destination)

    first = run_payload_stage(
        store=root,
        ordinal=1,
        stage="copy_payload",
        device="cpu",
        seed=20260903,
        inputs=[source],
        outputs=[destination],
        config={"test": "real-byte resume boundary"},
        non_negotiables={"payload_retained": True},
        action=action,
        resume=True,
    )
    second = run_payload_stage(
        store=root,
        ordinal=1,
        stage="copy_payload",
        device="cpu",
        seed=20260903,
        inputs=[source],
        outputs=[destination],
        config={"test": "real-byte resume boundary"},
        non_negotiables={"payload_retained": True},
        action=action,
        resume=True,
    )
    assert first.outputs == second.outputs
    assert second.resumed is True


def test_requested_unavailable_device_is_refused() -> None:
    unavailable: list[str] = []
    for name in ("cuda", "mps"):
        try:
            require_device(name)
        except PipelineBlocked:
            unavailable.append(name)
    if not unavailable:
        pytest.skip("host exposes both optional torch devices")
    with pytest.raises(PipelineBlocked, match="requested torch device is unavailable"):
        require_device(unavailable[0])


def test_full_mode_refuses_before_a_fake_completion_claim() -> None:
    store = TEST_STORE / "full_blocker_contract"
    with pytest.raises(PipelineBlocked, match="full mode refused before training"):
        SemanticPipeline(
            FullPipelineConfig(
                mode="full",
                device="cpu",
                store=store,
                smoke_pairs=2,
                smoke_steps=2,
            )
        ).run()
    result_path = store / "full" / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "BLOCKED_PORT_REQUIRED"
    assert result["archive"] is None
    assert result["score_claim"] is False
    assert {row["code"] for row in result["blockers"]} >= {
        "QS5_INSTANCE_PINNED",
        "SHIPPED_RECEIVER_FRESH_ARCHIVE_REFUSAL",
        "PREFIX_RUNTIME_UNREACHABLE",
    }
