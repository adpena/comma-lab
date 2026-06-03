from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "export_hinerv_checkpoint_archive.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "export_hinerv_checkpoint_archive", TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_state_path_selects_ema_and_live(tmp_path: Path) -> None:
    tool = _load_tool()
    ema_state = tmp_path / "epoch0001.ema_shadow.state.npsd"
    live_state = tmp_path / "epoch0001.live.state.npsd"
    ema_state.write_bytes(b"ema")
    live_state.write_bytes(b"live")
    meta = {
        "ema_shadow_state_path": ema_state.as_posix(),
        "live_state_path": live_state.as_posix(),
    }

    assert tool._checkpoint_state_path(meta, state_kind="ema") == ema_state.resolve(
        strict=False
    )
    assert tool._checkpoint_state_path(meta, state_kind="live") == live_state.resolve(
        strict=False
    )


def test_checkpoint_state_path_fails_closed_for_missing_state(tmp_path: Path) -> None:
    tool = _load_tool()
    meta = {"ema_shadow_state_path": (tmp_path / "missing.state.npsd").as_posix()}

    with pytest.raises(FileNotFoundError, match="checkpoint state not found"):
        tool._checkpoint_state_path(meta, state_kind="ema")


def test_checkpoint_state_path_fails_closed_for_missing_meta_key() -> None:
    tool = _load_tool()

    with pytest.raises(ValueError, match="checkpoint meta missing live_state_path"):
        tool._checkpoint_state_path({}, state_kind="live")


def test_blockers_price_archive_and_receiver_proof() -> None:
    tool = _load_tool()

    blockers = tool._blockers(
        archive_bytes=121_690,
        hard_byte_ceilings=[178_000, 216_000],
        receiver_proof={},
        receiver_proof_requested=False,
    )
    assert "archive_bytes_exceed_tightest_hard_ceiling" not in blockers
    assert "receiver_proof_not_requested" in blockers
    assert "contest_cpu_cuda_exact_eval_not_executed" in blockers

    blockers = tool._blockers(
        archive_bytes=214_654,
        hard_byte_ceilings=[178_000, 216_000],
        receiver_proof={},
        receiver_proof_requested=True,
    )
    assert "archive_bytes_exceed_tightest_hard_ceiling" in blockers
    assert "receiver_proof_not_ready" in blockers

    blockers = tool._blockers(
        archive_bytes=121_690,
        hard_byte_ceilings=[178_000],
        receiver_proof={"runtime_consumption_proof_ready": True},
        receiver_proof_requested=True,
    )
    assert "archive_bytes_exceed_tightest_hard_ceiling" not in blockers
    assert "receiver_proof_not_ready" not in blockers
    assert "receiver_proof_not_requested" not in blockers


def test_summary_keeps_false_authority_fields() -> None:
    tool = _load_tool()
    report = {
        "schema": "hinerv_checkpoint_archive_export.v1",
        "candidate_id": "hinerv_np600_demo",
        "checkpoint_epoch": 9499,
        "checkpoint_state_kind": "ema",
        "archive_path": "/ssd/archive.zip",
        "archive_bytes": 121_690,
        "rate_byte_profile": {"profile_ready": True},
        "receiver_proof_ready": False,
        "blockers": ["full_video_scorer_replay_not_executed"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    summary = tool._summary(report)

    assert summary["schema"] == "hinerv_checkpoint_archive_export.v1"
    assert summary["archive_bytes"] == 121_690
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["blockers"] == ["full_video_scorer_replay_not_executed"]
