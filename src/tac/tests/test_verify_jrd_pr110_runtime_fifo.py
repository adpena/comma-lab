# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/verify_jrd_pr110_runtime_fifo.py"
spec = importlib.util.spec_from_file_location("verify_jrd_pr110_runtime_fifo_test", TOOL)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _fixture(tmp_path: Path, payload: bytes = b"receiver-bytes") -> tuple[Path, Path]:
    submission = tmp_path / "submission"
    for relative in tool.RUNTIME_RELATIVE_FILES:
        path = submission / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture runtime {relative}\n")
    (submission / "inflate.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())\n"
    )
    candidate = tmp_path / "candidate.zip"
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", payload)
    return candidate, submission


@pytest.fixture(autouse=True)
def _durable_results_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspace/experiments/results"
    root.mkdir(parents=True)
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "RESULTS_ROOT", root)


def _receipt(tmp_path: Path, name: str = "run_20260713T000000Z") -> Path:
    return tool.RESULTS_ROOT / name


def test_fifo_verifier_hashes_twice_without_raw_file(tmp_path: Path) -> None:
    payload = b"receiver-bytes"
    candidate, submission = _fixture(tmp_path, payload)
    receipt = _receipt(tmp_path)
    proof = tool.verify_runtime_fifo(
        candidate_path=candidate,
        submission_dir=submission,
        receipt_dir=receipt,
        expected_raw_sha256=tool.hashlib.sha256(payload).hexdigest(),
        expected_raw_bytes=len(payload),
    )
    assert proof["bit_exact"] is True
    assert proof["scratch_cleaned_on_success"] is True
    assert proof["streaming_fifo_no_bulk_raw_materialized"] is True
    assert proof["submission_runtime"]["tree_sha256"]
    assert [row["raw_bytes"] for row in proof["passes"]] == [len(payload), len(payload)]
    assert not list(receipt.rglob("*.raw"))
    assert not (receipt / "fifo_runtime_scratch").exists()


def test_fifo_verifier_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    candidate, submission = _fixture(tmp_path)
    receipt = _receipt(tmp_path)
    with pytest.raises(RuntimeError, match="raw SHA mismatch"):
        tool.verify_runtime_fifo(
            candidate_path=candidate,
            submission_dir=submission,
            receipt_dir=receipt,
            expected_raw_sha256="0" * 64,
            expected_raw_bytes=len(b"receiver-bytes"),
        )
    failed = list(receipt.glob("runtime_inflate_fifo_failed_*.json"))
    assert len(failed) == 1
    assert not (receipt / "runtime_inflate_proof.json").exists()
    payload = json.loads(failed[0].read_text())
    retained = payload["retained_failure_artifacts"]
    assert {Path(row["path"]).name for row in retained} == {"x"}
    for row in retained:
        path = Path(row["path"])
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == tool.sha256_file(path)


def test_fifo_verifier_records_nonzero_runtime_log_before_refusal(tmp_path: Path) -> None:
    candidate, submission = _fixture(tmp_path)
    (submission / "inflate.py").write_text(
        "print('runtime failed intentionally')\n"
        "raise SystemExit(7)\n"
    )
    receipt = _receipt(tmp_path)
    with pytest.raises(RuntimeError, match="failed rc=7"):
        tool.verify_runtime_fifo(
            candidate_path=candidate,
            submission_dir=submission,
            receipt_dir=receipt,
            expected_raw_sha256="0" * 64,
            expected_raw_bytes=len(b"receiver-bytes"),
        )
    failed_path = next(receipt.glob("runtime_inflate_fifo_failed_*.json"))
    failed = tool.json.loads(failed_path.read_text())
    assert failed["passes"][0]["returncode"] == 7
    log_path = Path(failed["passes"][0]["log_path"])
    assert tool.sha256_file(log_path) == failed["passes"][0]["log_sha256"]
    assert b"runtime failed intentionally" in log_path.read_bytes()


def test_fifo_verifier_refuses_transient_receipt_directory(tmp_path: Path) -> None:
    candidate, submission = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must be a child"):
        tool.verify_runtime_fifo(
            candidate_path=candidate,
            submission_dir=submission,
            receipt_dir=tmp_path / "transient_receipt",
            expected_raw_sha256=tool.hashlib.sha256(b"receiver-bytes").hexdigest(),
            expected_raw_bytes=len(b"receiver-bytes"),
        )


def test_fifo_verifier_records_invalid_zip_failure_and_retains_attempt(
    tmp_path: Path,
) -> None:
    candidate, submission = _fixture(tmp_path)
    candidate.write_bytes(b"not-a-zip")
    receipt = _receipt(tmp_path)
    with pytest.raises(zipfile.BadZipFile):
        tool.verify_runtime_fifo(
            candidate_path=candidate,
            submission_dir=submission,
            receipt_dir=receipt,
            expected_raw_sha256="0" * 64,
            expected_raw_bytes=1,
        )
    failed_path = next(receipt.glob("runtime_inflate_fifo_failed_*.json"))
    failed = json.loads(failed_path.read_text())
    assert failed["error"].startswith("BadZipFile:")
    attempt = receipt / "fifo_runtime_scratch" / f"attempt_{failed['attempt_id']}"
    assert attempt.is_dir()
    assert failed["retained_failure_artifacts"] == []


def test_fifo_verifier_records_missing_runtime_failure(tmp_path: Path) -> None:
    candidate, submission = _fixture(tmp_path)
    (submission / "inflate.py").unlink()
    receipt = _receipt(tmp_path)
    with pytest.raises(FileNotFoundError):
        tool.verify_runtime_fifo(
            candidate_path=candidate,
            submission_dir=submission,
            receipt_dir=receipt,
            expected_raw_sha256="0" * 64,
            expected_raw_bytes=1,
        )
    failed_path = next(receipt.glob("runtime_inflate_fifo_failed_*.json"))
    failed = json.loads(failed_path.read_text())
    assert failed["error"].startswith("FileNotFoundError:")
    assert failed["runtime_inflate_py"] == {"path": str(submission / "inflate.py")}
    attempt = receipt / "fifo_runtime_scratch" / f"attempt_{failed['attempt_id']}"
    assert attempt.is_dir()


def test_fifo_reader_is_daemon_and_live_descendant_is_reported(tmp_path: Path) -> None:
    candidate, submission = _fixture(tmp_path)
    (submission / "inflate.py").write_text(
        "import os\n"
        "import sys\n"
        "import time\n"
        "fd = os.open(sys.argv[2], os.O_WRONLY)\n"
        "if os.fork() == 0:\n"
        "    os.close(1)\n"
        "    os.close(2)\n"
        "    time.sleep(0.3)\n"
        "    os.close(fd)\n"
        "    os._exit(0)\n"
        "os.close(fd)\n"
    )
    receipt = _receipt(tmp_path)
    receipt.mkdir(parents=True)
    scratch = receipt / "scratch"
    scratch.mkdir()
    member = scratch / "x"
    member.write_bytes(b"unused")
    started = time.monotonic()
    row = tool.stream_inflate_pass(
        pass_index=1,
        member_path=member,
        submission_dir=submission,
        scratch_dir=scratch,
        receipt_dir=receipt,
        attempt_id="daemon_regression",
        reader_join_timeout_seconds=0.01,
    )
    assert time.monotonic() - started < 0.25
    assert row["returncode"] == 0
    assert row["reader_daemon"] is True
    assert row["reader_alive_after_join"] is True
    assert not (scratch / "pass_1.fifo").exists()
