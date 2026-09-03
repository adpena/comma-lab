# SPDX-License-Identifier: MIT
"""Acceptance and fail-closed tests for the semantic pipeline stage contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.semantic_pipeline import (
    FullPipelineConfig,
    PipelineBlocked,
    SemanticPipeline,
    TargetLineage,
    probe_clip,
    require_device,
)
from tac.semantic_pipeline.contracts import atomic_copy, run_payload_stage
from tac.semantic_pipeline.pipeline import DEFAULT_VIDEO, FINAL_ARCHIVE, FINAL_BYTES, FINAL_SHA256
from tac.semantic_pipeline.receiver import ReceiverRequest, inflate

TEST_STORE = Path("/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/acceptance_v2")


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


def test_receiver_matches_shipped_afr1_prefix_bytes() -> None:
    fact = {"bytes": FINAL_ARCHIVE.stat().st_size, "sha256": FINAL_SHA256}
    root = TEST_STORE / "receiver_afr1_identity"
    report = inflate(
        ReceiverRequest(
            archive=FINAL_ARCHIVE,
            archive_sha256=fact["sha256"],
            archive_bytes=fact["bytes"],
            destination=root / "afr1_n2.raw",
            runtime_root=root / "runtime_copy",
            checkpoint_dir=root / "checkpoint",
            device="cpu",
            pair_count=2,
        )
    )
    assert report["raw_bytes"] == 12_208_032
    assert report["raw_sha256"] == "8ef6939b2041cee988a9484ed7c5c30be94ff7dec0fef8aed85c8d80c0999497"


def test_full_mode_real_n2_smoke_is_receiver_closed() -> None:
    result = SemanticPipeline(
        FullPipelineConfig(
            mode="full",
            device="cpu",
            store=TEST_STORE / "full_contract",
            smoke_pairs=2,
            smoke_steps=2,
            smoke=True,
            resume=True,
        )
    ).run()
    assert result["status"] == "PASS"
    assert result["fresh_archive"] is True
    assert result["archive"]["sha256"] != FINAL_SHA256
    assert result["receiver_identity"]["byte_identical"] is True
    assert result["receiver_identity"]["pair_count"] == 2
    assert result["receiver_identity"]["driver"]["path"].endswith(
        "direct_driver_render_n2.raw"
    )
    assert result["stages"][0]["outputs"][4]["path"].endswith(
        "semantic_quantized_state.pt"
    )
    assert result["advisory_score"]["axis"] == "[macOS-CPU advisory]"
    assert result["advisory_score"]["score_claim"] is False
    assert [row["stage"] for row in result["stages"]] == [
        "scorer_aware_train",
        "qs5_compensation_kernel_port",
        "direct_driver_render",
        "receiver_receiver_render",
        "upstream_evaluate_n2_advisory",
    ]


def test_target_lineage_refuses_silent_mixing() -> None:
    with pytest.raises(PipelineBlocked, match="silent target-lineage mixing refused"):
        TargetLineage(semantic="av", carrier="av", hpac="dali", token="dali")


def test_ported_cli_device_flag_subsets_parse() -> None:
    from experiments import ddm_fcd1_incompile_schur as fcd1
    from experiments import ddm_jg5_pose_resolve_on_edited_renders as jg5
    from experiments import ddm_qs5_resolve_compensation as qs5
    from experiments import ddm_up2_shipping_pose_solve as up2

    assert fcd1.build_parser().parse_args(
        ["decode", "--runtime", "r", "--output", "o", "--device", "cpu"]
    ).device == "cpu"
    assert jg5.build_parser().parse_args(
        ["retain", "--root", "r", "--out", "o", "--device", "cpu"]
    ).device == "cpu"
    assert qs5.parse_args(["--device", "cpu"]).device == "cpu"
    assert up2.build_parser().parse_args(["validate", "--device", "cpu"]).device == "cpu"
