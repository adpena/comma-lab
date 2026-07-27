from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.witness_control import (
    taskspace_g121_resumable_stage_harvest_v1 as g121,
)
from tools import run_taskspace_g121_live_stage_harvest as monitor


def _launch(producer: Path, payload: bytes) -> str:
    producer.mkdir()
    (producer / "launch_manifest.json").write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _preserved_stage(producer: Path) -> None:
    (producer / "levelset_ckpt_stageCE_ep25.npz").write_bytes(b"deploy")
    (producer / "levelset_resume_stageCE_ep25.npz").write_bytes(b"resume")


def _progress(tmp_path: Path) -> g121.G121StageHarvestProgressV1:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(b"{}\n")
    return g121.G121StageHarvestProgressV1(
        stage_ledger_path=ledger,
        stage_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        discovered_stage_count=1,
        accounted_stage_count=1,
        scorer_replay_count=1,
        reused_measurement_count=0,
    )


def _final(tmp_path: Path) -> g121.G121StageHarvestResultV1:
    retained = tmp_path / "retained.json"
    completion = tmp_path / "completion.json"
    ledger = tmp_path / "ledger.jsonl"
    for path in (retained, completion, ledger):
        path.write_bytes(path.name.encode())
    return g121.G121StageHarvestResultV1(
        retained_prepose_path=retained,
        retained_prepose_sha256=hashlib.sha256(retained.read_bytes()).hexdigest(),
        completion_receipt_path=completion,
        completion_receipt_sha256=hashlib.sha256(
            completion.read_bytes()
        ).hexdigest(),
        stage_ledger_path=ledger,
        stage_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        scheduling_hint_path=None,
        scheduling_hint_sha256=None,
        discovered_stage_count=1,
        accounted_stage_count=1,
        retained_stage_count=1,
        deferred_stage_count=0,
        pruned_stage_count=0,
        blocked_stage_count=0,
        scorer_replay_count=0,
        reused_measurement_count=1,
    )


def test_once_harvests_available_stage_without_exhaustive_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    launch_sha = _launch(producer, b"launch-a")
    _preserved_stage(producer)
    calls: list[str] = []
    monkeypatch.setattr(
        g121,
        "harvest_g111_available_stages_v1",
        lambda **_kwargs: (calls.append("incremental") or _progress(tmp_path)),
    )
    monkeypatch.setattr(
        g121,
        "harvest_g111_stages_v1",
        lambda **_kwargs: (calls.append("final") or _final(tmp_path)),
    )
    output = tmp_path / "out"
    progress = tmp_path / "progress"
    assert (
        monitor.run_monitor(
            producer_run_dir=producer,
            expected_launch_manifest_sha256=launch_sha,
            output_dir=output,
            progress_dir=progress,
            poll_seconds=0.01,
            once=True,
        )
        == 0
    )
    assert calls == ["incremental"]
    binding = json.loads(
        (output / monitor.MONITOR_BINDING_BASENAME).read_text()
    )
    assert binding["old_producer_payload_reuse"] is False
    assert binding["terminal_only_exhaustive_publication"] is True
    assert not (output / g121.COMPLETION_RECEIPT_BASENAME).exists()


def test_terminal_marker_routes_incremental_before_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    launch_sha = _launch(producer, b"launch-a")
    _preserved_stage(producer)
    (producer / "levelset_train_result.json").write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr(
        g121,
        "harvest_g111_available_stages_v1",
        lambda **_kwargs: (calls.append("incremental") or _progress(tmp_path)),
    )
    monkeypatch.setattr(
        g121,
        "harvest_g111_stages_v1",
        lambda **_kwargs: (calls.append("final") or _final(tmp_path)),
    )
    output = tmp_path / "out"
    assert (
        monitor.run_monitor(
            producer_run_dir=producer,
            expected_launch_manifest_sha256=launch_sha,
            output_dir=output,
            progress_dir=tmp_path / "progress",
            poll_seconds=0.01,
            once=False,
        )
        == 0
    )
    assert calls == ["incremental", "final"]
    status = json.loads(
        (output / monitor.MONITOR_STATUS_BASENAME).read_text()
    )
    assert status["status"] == "EXHAUSTIVE_STAGE_HARVEST_COMPLETE"


def test_output_binding_refuses_a_different_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer_a = tmp_path / "producer-a"
    producer_b = tmp_path / "producer-b"
    launch_a = _launch(producer_a, b"launch-a")
    launch_b = _launch(producer_b, b"launch-b")
    _preserved_stage(producer_a)
    _preserved_stage(producer_b)
    monkeypatch.setattr(
        g121,
        "harvest_g111_available_stages_v1",
        lambda **_kwargs: _progress(tmp_path),
    )
    output = tmp_path / "out"
    monitor.run_monitor(
        producer_run_dir=producer_a,
        expected_launch_manifest_sha256=launch_a,
        output_dir=output,
        progress_dir=tmp_path / "progress-a",
        poll_seconds=0.01,
        once=True,
    )
    with pytest.raises(
        g121.G121StageHarvestError,
        match="refusing to overwrite different",
    ):
        monitor.run_monitor(
            producer_run_dir=producer_b,
            expected_launch_manifest_sha256=launch_b,
            output_dir=output,
            progress_dir=tmp_path / "progress-b",
            poll_seconds=0.01,
            once=True,
        )


def test_same_producer_accepts_a_new_externally_bound_resume_launch_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    launch_a = _launch(producer, b"launch-a")
    _preserved_stage(producer)
    monkeypatch.setattr(
        g121,
        "harvest_g111_available_stages_v1",
        lambda **_kwargs: _progress(tmp_path),
    )
    output = tmp_path / "out"
    progress = tmp_path / "progress"
    assert (
        monitor.run_monitor(
            producer_run_dir=producer,
            expected_launch_manifest_sha256=launch_a,
            output_dir=output,
            progress_dir=progress,
            poll_seconds=0.01,
            once=True,
        )
        == 0
    )
    (producer / "launch_manifest.json").write_bytes(b"launch-resume")
    launch_resume = hashlib.sha256(b"launch-resume").hexdigest()
    assert (
        monitor.run_monitor(
            producer_run_dir=producer,
            expected_launch_manifest_sha256=launch_resume,
            output_dir=output,
            progress_dir=progress,
            poll_seconds=0.01,
            once=True,
        )
        == 0
    )
    epochs = (
        output / monitor.MONITOR_LAUNCH_EPOCHS_BASENAME
    ).read_text().splitlines()
    assert len(epochs) == 2
    assert json.loads(epochs[-1])["launch_manifest"]["sha256"] == launch_resume


def test_resume_launch_refuses_stale_terminal_reduction_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    launch_a = _launch(producer, b"launch-a")
    _preserved_stage(producer)
    monkeypatch.setattr(
        g121,
        "harvest_g111_available_stages_v1",
        lambda **_kwargs: _progress(tmp_path),
    )
    output = tmp_path / "out"
    progress = tmp_path / "progress"
    monitor.run_monitor(
        producer_run_dir=producer,
        expected_launch_manifest_sha256=launch_a,
        output_dir=output,
        progress_dir=progress,
        poll_seconds=0.01,
        once=True,
    )
    (output / g121.COMPLETION_RECEIPT_BASENAME).write_text("{}")
    (producer / "launch_manifest.json").write_bytes(b"launch-resume")
    with pytest.raises(
        g121.G121StageHarvestError,
        match="terminal G121 reductions",
    ):
        monitor.run_monitor(
            producer_run_dir=producer,
            expected_launch_manifest_sha256=hashlib.sha256(
                b"launch-resume"
            ).hexdigest(),
            output_dir=output,
            progress_dir=progress,
            poll_seconds=0.01,
            once=True,
        )
