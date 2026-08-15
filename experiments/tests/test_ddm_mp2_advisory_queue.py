# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "ddm_mp2_advisory_queue.py"
SPEC = importlib.util.spec_from_file_location("ddm_mp2_advisory_queue", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def _generation(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "complete": True,
        "receiver_closed": True,
        "archive": {"path": f"/tmp/{candidate_id}/archive.zip", "sha256": "a" * 64, "bytes": 1},
    }


def test_candidate_rows_excludes_control_and_requires_full_denominator() -> None:
    manifest = {"generations": [_generation("hv1_base_control")] + [_generation(f"c{i}") for i in range(7)]}
    assert [row["candidate_id"] for row in QUEUE._candidate_rows(manifest)] == [f"c{i}" for i in range(7)]


def test_candidate_rows_refuses_receiver_open() -> None:
    generations = [_generation("hv1_base_control")] + [_generation(f"c{i}") for i in range(7)]
    generations[-1]["receiver_closed"] = False
    with pytest.raises(QUEUE.QueueRefusal, match="receiver-open"):
        QUEUE._candidate_rows({"generations": generations})


def test_validate_result_binds_archive_and_n600(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "n_samples": 600,
                "archive_size_bytes": 123,
                "avg_segnet_dist": 0.1,
                "avg_posenet_dist": 0.2,
                "provenance": {"archive_sha256": "b" * 64, "archive_size_bytes": 123},
            }
        )
    )
    result = QUEUE._validate_result(result_path, archive_sha256="b" * 64, archive_bytes=123)
    assert result["n_samples"] == 600
    with pytest.raises(QUEUE.QueueRefusal, match="archive_sha256"):
        QUEUE._validate_result(result_path, archive_sha256="c" * 64, archive_bytes=123)


def test_clean_appledouble_uses_find_and_deletes_only_sidecars(tmp_path: Path) -> None:
    keep = tmp_path / "payload.bin"
    sidecar = tmp_path / "._payload.bin"
    keep.write_bytes(b"payload")
    sidecar.write_bytes(b"sidecar")
    removed = QUEUE._clean_appledouble([tmp_path])
    assert removed == [str(sidecar)]
    assert keep.read_bytes() == b"payload"
    assert not sidecar.exists()


def test_launch_argv_is_candidate_bound_and_retained(tmp_path: Path) -> None:
    generation = _generation("candidate")
    command, receipt, result = QUEUE._launch_argv(
        candidate_id="candidate",
        generation=generation,
        attempt_dir=tmp_path / "attempt_0000",
        attempt=0,
    )
    assert "tools/launch_detached_process.py" in " ".join(command)
    assert "--keep-work-dir" in command
    assert "PYTHONDONTWRITEBYTECODE=1" in command
    assert generation["archive"]["path"] in command
    assert receipt.name == "ddm_mp2_candidate_n600_attempt_0000.done"
    done_index = command.index("--done-receipt")
    assert command[done_index + 1] == "ddm_mp2_candidate_n600_attempt_0000"
    assert result == tmp_path / "attempt_0000" / "contest_auth_eval.json"
