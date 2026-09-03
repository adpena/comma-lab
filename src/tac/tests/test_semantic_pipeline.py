# SPDX-License-Identifier: MIT
"""Acceptance and fail-closed tests for the semantic pipeline stage contracts."""

from __future__ import annotations

import json
import time
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
from tac.semantic_pipeline.contracts import atomic_copy, atomic_json, file_fact, run_payload_stage
from tac.semantic_pipeline.pipeline import (
    DEFAULT_VIDEO,
    FINAL_ARCHIVE,
    FINAL_BYTES,
    FINAL_SHA256,
    governed_launch_ticket_payload,
    population_memory_preflight,
)
from tac.semantic_pipeline.receiver import ReceiverRequest, inflate
from tac.semantic_pipeline.stages.train import TrainRequest, build_chunk_schedule, run_train_stage
from tac.subset_selection import MODE_SEEDED_RANDOM

TEST_STORE = Path("/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/acceptance_v2")
FPC3_STORE = Path("/Volumes/VertigoDataTier/pact/ddm_fpc3_chunked_n600_trainer")


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


def test_chunk_order_is_seeded_random_full_permutation_not_prefix() -> None:
    first = build_chunk_schedule(6, 2, seed=20260903, mode=MODE_SEEDED_RANDOM)
    second = build_chunk_schedule(6, 2, seed=20260903, mode=MODE_SEEDED_RANDOM)
    flattened = tuple(pair for chunk in first for pair in chunk)
    assert first == second
    assert sorted(flattened) == list(range(6))
    assert flattened != tuple(range(6))
    assert all(len(chunk) == 2 for chunk in first)


@pytest.mark.timeout(600)
def test_chunked_resume_identity_three_real_cpu_chunks() -> None:
    trainer_source = Path(__file__).parents[1] / "semantic_pipeline" / "stages" / "train.py"
    source_key = file_fact(trainer_source)["sha256"][:16]
    root = FPC3_STORE / "resume_smoke" / source_key
    receipt_path = root / "RESUME_IDENTITY_RESULT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "PASS"
        assert all(receipt["equal"].values())
        return
    started = time.monotonic()
    common = {
        "video": DEFAULT_VIDEO,
        "source_archive": FINAL_ARCHIVE,
        "device": "cpu",
        "pair_count": 6,
        "steps": 3,
        "seed": 20260903,
        "lineage": TargetLineage(),
        "chunk_pairs": 2,
        "selection_mode": MODE_SEEDED_RANDOM,
        "verdict_batch": 2,
    }
    uninterrupted_root = root / "uninterrupted"
    uninterrupted_result = uninterrupted_root / "TRAIN_RESULT.json"
    if uninterrupted_result.is_file():
        uninterrupted = json.loads(uninterrupted_result.read_text(encoding="utf-8"))
    else:
        uninterrupted = run_train_stage(
            TrainRequest(
                output_dir=uninterrupted_root,
                resume=any((uninterrupted_root / "checkpoints").glob("stage_train_chunk_*.pt")),
                **common,
            )
        )
    resumed_root = root / "resumed"
    resumed_result = resumed_root / "TRAIN_RESULT.json"
    if resumed_result.is_file():
        resumed = json.loads(resumed_result.read_text(encoding="utf-8"))
    else:
        checkpoints = sorted((resumed_root / "checkpoints").glob("stage_train_chunk_*.pt"))
        if not checkpoints:
            interrupted = run_train_stage(
                TrainRequest(
                    output_dir=resumed_root,
                    resume=False,
                    stop_after_chunks=1,
                    **common,
                )
            )
            assert interrupted["status"] == "INTERRUPTED_AT_REQUESTED_BOUNDARY"
            checkpoints = [Path(interrupted["checkpoint"]["path"])]
        resumed = run_train_stage(
            TrainRequest(
                output_dir=resumed_root,
                resume=True,
                resume_from=checkpoints[-1],
                **common,
            )
        )
    equal = {
        "live_state": uninterrupted["live_state_sha256"] == resumed["live_state_sha256"],
        "ema_state": uninterrupted["ema_state_sha256"] == resumed["ema_state_sha256"],
        "archive": (
            uninterrupted["archive"]["archive"]["sha256"]
            == resumed["archive"]["archive"]["sha256"]
        ),
        "loss_rows": uninterrupted["loss_rows"] == resumed["loss_rows"],
    }
    receipt = {
        "schema": "ddm_fpc3_three_chunk_resume_identity.v1",
        "status": "PASS" if all(equal.values()) else "FAIL",
        "axis": "[macOS-CPU exact-scorer n6 mechanism smoke; not a verdict]",
        "score_claim": False,
        "pair_count": 6,
        "chunk_pairs": 2,
        "executed_updates": 6,
        "elapsed_seconds": time.monotonic() - started,
        "equal": equal,
        "uninterrupted_result": file_fact(uninterrupted_root / "TRAIN_RESULT.json"),
        "resumed_result": file_fact(resumed_root / "TRAIN_RESULT.json"),
        "all_payloads_retained": True,
    }
    atomic_json(receipt_path, receipt)
    assert receipt["status"] == "PASS"


def test_population_memory_preflight_refuses_over_70_percent_ram() -> None:
    gib = 1024**3
    result = population_memory_preflight(
        chunk_pairs=16,
        verdict_batch=32,
        total_ram_bytes=128 * gib,
        current_used_bytes=60 * gib,
    )
    assert result["status"] == "REFUSE"
    assert result["checks"]["projected_process_peak_within_limit"] is True
    assert result["checks"]["system_aware_used_plus_peak_within_limit"] is False
    assert result["host"]["safety_fraction"] == 0.70


def test_governed_ticket_has_resource_resume_lane_and_fire_contract() -> None:
    memory = population_memory_preflight(
        chunk_pairs=16,
        verdict_batch=32,
        total_ram_bytes=128 * 1024**3,
        current_used_bytes=0,
    )
    ticket = governed_launch_ticket_payload(
        store=FPC3_STORE,
        video=DEFAULT_VIDEO,
        seed=20260903,
        chunk_pairs=16,
        verdict_batch=32,
        memory_receipt=memory,
        storage_projection={"schema": "test", "status": "PASS"},
    )
    argv = ticket["argv"]
    assert ticket["schema"] == "ddm_fpc3_governed_n600_launch_ticket.v1"
    assert ticket["status"] == "QUEUED_WITH_FIRE_ORDER"
    assert ticket["owner"] == "MAIN"
    assert ticket["scorer_lane_claim_id_placeholder"]
    assert "Metal slot released by the QBR1 burn" in ticket["fire_trigger"]
    for flag in (
        "--derive-resource-budgets",
        "--measured-peak-rss-gib",
        "--measured-thread-need",
        "--walltime-cap-s",
        "--done-receipt",
        "--chunk-pairs",
        "--verdict-batch",
        "--resume-from",
    ):
        assert flag in argv
