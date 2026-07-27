from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.witness_control import (
    g120_governed_clean_dry_run_gate_v1 as subject,
)

REPO = Path(__file__).resolve().parents[4]
EXPECTED_SOURCE_BINDINGS = {
    "g105_semantic_adapter",
    "g111_stage_selector",
    "g112_checkpoint_partition",
    "g120_dry_run_cli",
    "g120_dry_run_gate",
    "g120_production_engine",
    "g120_production_wrapper",
    "g121_harvester",
    "g121_live_monitor",
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    ssd = tmp_path / "VertigoDataTier" / "pact"
    producer = ssd / "producer"
    producer.mkdir(parents=True)
    launch = producer / "launch_manifest.json"
    launch.write_bytes(b'{"governed":true}\n')
    launch_binding = {
        "path": str(launch.resolve()),
        "bytes": launch.stat().st_size,
        "sha256": _sha(launch.read_bytes()),
    }
    monkeypatch.setattr(subject, "_SSD_ROOTS", (ssd,))
    monkeypatch.setattr(
        subject.g121,
        "_open_governed_launch_manifest",
        lambda _producer, *, expected_sha256: (
            launch_binding,
            _sha(b"launch-dsl"),
        )
        if expected_sha256 == launch_binding["sha256"]
        else (_ for _ in ()).throw(
            AssertionError("wrong external launch SHA")
        ),
    )
    return {
        "repo_root": REPO,
        "producer_run_dir": producer,
        "expected_launch_manifest_sha256": launch_binding[
            "sha256"
        ],
        "monitor_output_dir": producer / "g121_harvest",
        "monitor_progress_dir": producer / "g121_progress",
        "measurement_cache_dir": (
            producer / "g121_measurement_cache"
        ),
        "gate_dir": producer / "g120_gate",
    }


def test_checkpoint_resume_and_reopen_are_distinct_process_no_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.os, "getpid", lambda: 1001)
    checkpoint = subject.run_g120_governed_clean_dry_run_v1(
        phase="checkpoint",
        **kwargs,
    )
    assert checkpoint.clean_dry_run_complete is False
    assert checkpoint.scorer_calls == 0

    monkeypatch.setattr(subject.os, "getpid", lambda: 1002)
    completed = subject.run_g120_governed_clean_dry_run_v1(
        phase="resume",
        **kwargs,
    )
    assert completed.clean_dry_run_complete is True
    assert completed.scorer_calls == 0
    receipt = subject.open_g120_governed_clean_dry_run_v1(
        completed.receipt_path,
        expected_sha256=completed.receipt_sha256,
        repo_root=kwargs["repo_root"],
        producer_run_dir=kwargs["producer_run_dir"],
        expected_launch_manifest_sha256=kwargs[
            "expected_launch_manifest_sha256"
        ],
        monitor_output_dir=kwargs["monitor_output_dir"],
        monitor_progress_dir=kwargs["monitor_progress_dir"],
        measurement_cache_dir=kwargs["measurement_cache_dir"],
    )
    assert receipt["checkpoint_pid"] == 1001
    assert receipt["resume_pid"] == 1002
    assert receipt["batch_resume_proof"]["scorer_calls"] == 0
    assert receipt["authority"]["heavy_scorer_run_launched"] is False
    assert set(receipt["binding"]["source_bindings"]) == (
        EXPECTED_SOURCE_BINDINGS
    )


def test_resume_in_checkpoint_process_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.os, "getpid", lambda: 2001)
    subject.run_g120_governed_clean_dry_run_v1(
        phase="checkpoint",
        **kwargs,
    )
    with pytest.raises(
        subject.G120GovernedDryRunError,
        match="identity/process binding",
    ):
        subject.run_g120_governed_clean_dry_run_v1(
            phase="resume",
            **kwargs,
        )


def test_resume_refuses_corrupt_physical_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.os, "getpid", lambda: 2101)
    subject.run_g120_governed_clean_dry_run_v1(
        phase="checkpoint",
        **kwargs,
    )
    prediction = (
        kwargs["gate_dir"]
        / subject.PROBE_DIRNAME
        / "batch_000_000_016.predicted_labels.npy"
    )
    payload = prediction.read_bytes()
    prediction.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))

    monkeypatch.setattr(subject.os, "getpid", lambda: 2102)
    with pytest.raises(
        subject.G120GovernedDryRunError,
        match="prediction batch 0 array physical identity differs",
    ):
        subject.run_g120_governed_clean_dry_run_v1(
            phase="resume",
            **kwargs,
        )


def test_non_ssd_path_and_insufficient_free_space_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / "local-output"
    kwargs["monitor_output_dir"] = outside
    with pytest.raises(
        subject.G120GovernedDryRunError,
        match="outside the configured SSD",
    ):
        subject.run_g120_governed_clean_dry_run_v1(
            phase="checkpoint",
            **kwargs,
        )

    kwargs = _fixture(tmp_path / "second", monkeypatch)
    monkeypatch.setattr(
        subject.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=subject.MINIMUM_FREE_BYTES,
            used=1,
            free=subject.MINIMUM_FREE_BYTES - 1,
        ),
    )
    with pytest.raises(
        subject.G120GovernedDryRunError,
        match="storage preflight",
    ):
        subject.run_g120_governed_clean_dry_run_v1(
            phase="checkpoint",
            **kwargs,
        )


@pytest.mark.parametrize(
    "changed_source",
    sorted(EXPECTED_SOURCE_BINDINGS),
)
def test_each_source_binding_change_invalidates_completed_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_source: str,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.os, "getpid", lambda: 3001)
    subject.run_g120_governed_clean_dry_run_v1(
        phase="checkpoint",
        **kwargs,
    )
    monkeypatch.setattr(subject.os, "getpid", lambda: 3002)
    completed = subject.run_g120_governed_clean_dry_run_v1(
        phase="resume",
        **kwargs,
    )
    original = subject._source_bindings

    def changed_sources(repo_root: Path):
        rows = original(repo_root)
        rows[changed_source] = {
            **rows[changed_source],
            "sha256": _sha(b"changed"),
        }
        return rows

    monkeypatch.setattr(
        subject,
        "_source_bindings",
        changed_sources,
    )
    with pytest.raises(
        subject.G120GovernedDryRunError,
        match="current production launch",
    ):
        subject.open_g120_governed_clean_dry_run_v1(
            completed.receipt_path,
            expected_sha256=completed.receipt_sha256,
            repo_root=kwargs["repo_root"],
            producer_run_dir=kwargs["producer_run_dir"],
            expected_launch_manifest_sha256=kwargs[
                "expected_launch_manifest_sha256"
            ],
            monitor_output_dir=kwargs["monitor_output_dir"],
            monitor_progress_dir=kwargs[
                "monitor_progress_dir"
            ],
            measurement_cache_dir=kwargs[
                "measurement_cache_dir"
            ],
        )


def test_resume_refuses_source_change_after_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.os, "getpid", lambda: 3101)
    subject.run_g120_governed_clean_dry_run_v1(
        phase="checkpoint",
        **kwargs,
    )
    original = subject._source_bindings

    def changed_sources(repo_root: Path):
        rows = original(repo_root)
        rows["g111_stage_selector"] = {
            **rows["g111_stage_selector"],
            "sha256": _sha(b"changed-between-processes"),
        }
        return rows

    monkeypatch.setattr(
        subject,
        "_source_bindings",
        changed_sources,
    )
    monkeypatch.setattr(subject.os, "getpid", lambda: 3102)
    with pytest.raises(
        subject.G120GovernedDryRunError,
        match="refusing to overwrite different output",
    ):
        subject.run_g120_governed_clean_dry_run_v1(
            phase="resume",
            **kwargs,
        )
