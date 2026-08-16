# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_wc1_gate_is_optional_but_present_gate_fails_closed(tmp_path: Path) -> None:
    gate_path = tmp_path / "ADMISSION_GATE.json"
    assert QUEUE._validated_wc1_fast_path(gate_path) is None
    gate_path.write_text(json.dumps({"schema": "wrong"}))
    with pytest.raises(QUEUE.QueueRefusal, match="present but invalid"):
        QUEUE._validated_wc1_fast_path(gate_path)


def test_wc1_gate_threads_admitted_flags_through_staged_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_file = tmp_path / "code.py"
    code_file.write_text("# pinned\n")
    environment = {
        "F26_TOKEN_DECODER": "native-hpac",
        "F26_HPAC_NATIVE_LIBRARY": "/retained/native.so",
        "F26_ADVISORY_RENDER_WORKERS": "auto",
        "F26_ADVISORY_DECODE_CACHE_ROOT": "/retained/cache",
        "F26_ADVISORY_RENDER_RSS_BYTES": 123,
    }
    gate_path = tmp_path / "ADMISSION_GATE.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema": "ddm_wc1_advisory_fast_path_admission.v1",
                "complete": True,
                "identity_pass": True,
                "shipping_packet_touched": False,
                "consumer_environment": environment,
                "consumer_code": {"test": QUEUE._file_fact(code_file)},
            }
        )
    )
    fast_path = QUEUE._validated_wc1_fast_path(gate_path)
    assert fast_path is not None

    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    (generation_root / "archive.zip").write_bytes(b"archive")
    (generation_root / "inflate.sh").write_text("#!/bin/sh\n")

    def prepare(_source: Path, destination: Path) -> None:
        destination.mkdir(parents=True)
        (destination / "archive.zip").write_bytes(b"archive")
        (destination / "inflate.sh").write_text("#!/bin/sh\n")

    monkeypatch.setattr(
        QUEUE,
        "_load_wc1_builder",
        lambda: SimpleNamespace(prepare_advisory_runtime=prepare),
    )
    generation = _generation("candidate")
    generation["archive"]["path"] = str(generation_root / "archive.zip")
    attempt = tmp_path / "attempt"
    command, _, _ = QUEUE._launch_argv(
        candidate_id="candidate",
        generation=generation,
        attempt_dir=attempt,
        attempt=0,
        wc1_fast_path=fast_path,
    )
    joined = " ".join(command)
    assert str(attempt / "wc1_advisory_generation/archive.zip") in command
    for key, value in environment.items():
        assert f"{key}={value}" in command
    assert str(generation_root / "archive.zip") not in joined
