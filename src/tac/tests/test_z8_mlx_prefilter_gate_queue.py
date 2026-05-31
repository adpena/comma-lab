# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from comma_lab.scheduler.z8_mlx_prefilter_gate_queue import (
    Z8MlxPrefilterGateQueueError,
    build_z8_mlx_prefilter_gate_queue,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_z8_mlx_prefilter_gate_queue.py"


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    candidate = root / "candidate.bin"
    archive_zip = root / "archive.zip"
    reference = root / "reference_pairs.npy"
    candidate.write_bytes(b"z8-candidate")
    archive_zip.write_bytes(b"PK\x03\x04z8")
    np.save(reference, np.zeros((1, 2, 4, 4, 3), dtype=np.float32))
    return candidate, archive_zip, reference


def test_build_z8_mlx_prefilter_gate_queue_replay_then_gate(tmp_path: Path) -> None:
    candidate, archive_zip, reference = _write_inputs(tmp_path)

    queue = build_z8_mlx_prefilter_gate_queue(
        [
            {
                "candidate_id": "rd_waterfill",
                "candidate_archive_bin": candidate,
                "archive_zip": archive_zip,
            }
        ],
        queue_id="z8_prefilter_test",
        repo_root=tmp_path,
        reference_pairs_npy=reference,
        output_root=tmp_path / "out",
        auth_frontier_score=0.1919853363,
        pair_chunk_size=16,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 1
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["schema"] == "z8_mlx_prefilter_gate_queue_metadata.v1"
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
    replay, gate, learning, cleanup = experiment["steps"]
    assert replay["id"] == "run_z8_full_video_mlx_replay"
    assert replay["resources"]["kind"] == "local_mlx"
    assert "tools/replay_z8_full_video_mlx_candidate.py" in replay["command"]
    assert "--archive-zip" in replay["command"]
    assert gate["requires"] == ["run_z8_full_video_mlx_replay"]
    assert gate["resources"]["kind"] == "local_cpu"
    assert "--success-on-blocked" in gate["command"]
    assert any(
        post.get("type") == "json_false_authority"
        for post in gate["postconditions"]
    )
    assert learning["id"] == "record_mlx_prefilter_gate_learning"
    assert learning["requires"] == ["gate_mlx_prefilter_for_exact_auth"]
    assert "tools/record_local_exact_auth_gate_learning.py" in learning["command"]
    assert "--candidate-id" in learning["command"]
    assert "--family-id" in learning["command"]
    assert cleanup["id"] == "cleanup_rebuildable_raw_scratch"
    assert cleanup["requires"] == ["record_mlx_prefilter_gate_learning"]
    assert cleanup["resources"]["kind"] == "local_io_heavy"
    assert "tools/compact_experiment_artifacts.py" in cleanup["command"]
    assert "--execute" in cleanup["command"]


def test_build_z8_mlx_prefilter_gate_queue_rejects_missing_candidate(tmp_path: Path) -> None:
    _, _, reference = _write_inputs(tmp_path)

    with pytest.raises(Z8MlxPrefilterGateQueueError, match="candidate_archive_bin missing"):
        build_z8_mlx_prefilter_gate_queue(
            [{"candidate_id": "missing", "candidate_archive_bin": tmp_path / "missing.bin"}],
            queue_id="z8_prefilter_missing",
            repo_root=tmp_path,
            reference_pairs_npy=reference,
            output_root=tmp_path / "out",
            auth_frontier_score=0.1919853363,
        )


def test_build_z8_mlx_prefilter_gate_queue_cli_writes_queue(tmp_path: Path) -> None:
    candidate, archive_zip, reference = _write_inputs(tmp_path)
    out = tmp_path / "queue.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--queue-out",
            str(out),
            "--queue-id",
            "z8_prefilter_cli",
            "--candidate-id",
            "rd_waterfill",
            "--candidate-archive-bin",
            str(candidate),
            "--archive-zip",
            str(archive_zip),
            "--reference-pairs-npy",
            str(reference),
            "--output-root",
            str(tmp_path / "out"),
            "--auth-frontier-score",
            "0.1919853363",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    queue = json.loads(out.read_text(encoding="utf-8"))
    assert queue["queue_id"] == "z8_prefilter_cli"
    steps = queue["experiments"][0]["steps"]
    assert [step["id"] for step in steps] == [
        "run_z8_full_video_mlx_replay",
        "gate_mlx_prefilter_for_exact_auth",
        "record_mlx_prefilter_gate_learning",
        "cleanup_rebuildable_raw_scratch",
    ]
