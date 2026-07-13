#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify PR110 candidate inflation twice without materializing bulky raw files.

The frozen submission ``inflate.py`` writes to a named pipe.  A local reader
hashes and counts the exact byte stream, so the 3.66 GB receiver output never
lands on local disk when the required SSD tier is read-only in the sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

EXPECTED_RAW_BYTES = 1200 * 874 * 1164 * 3
REPO = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO / "experiments/results"
sys.path.insert(0, str(REPO / "src"))

from tac.packet_compiler import jrd_pr110_runtime_custody as runtime_custody_module  # noqa: E402

RUNTIME_RELATIVE_FILES = runtime_custody_module.RUNTIME_RELATIVE_FILES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)


def _hash_fifo(path: Path, result: dict[str, Any]) -> None:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb", buffering=0) as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
                total += len(chunk)
        result.update({"raw_sha256": digest.hexdigest(), "raw_bytes": total})
    except BaseException as exc:  # propagate reader failures to the parent gate
        result["error"] = f"{type(exc).__name__}: {exc}"


def stream_inflate_pass(
    *,
    pass_index: int,
    member_path: Path,
    submission_dir: Path,
    scratch_dir: Path,
    receipt_dir: Path,
    attempt_id: str,
    timeout_seconds: int = 3600,
    reader_join_timeout_seconds: float = 60,
) -> dict[str, Any]:
    fifo_path = scratch_dir / f"pass_{pass_index}.fifo"
    os.mkfifo(fifo_path)
    # A temporary read/write descriptor prevents an early EOF and also lets the
    # reader terminate if inflate exits before opening its write end.
    dummy_fd = os.open(fifo_path, os.O_RDWR | os.O_NONBLOCK)
    reader_result: dict[str, Any] = {}
    reader = threading.Thread(
        target=_hash_fifo,
        args=(fifo_path, reader_result),
        name=f"jrd-inflate-hash-{pass_index}",
        daemon=True,
    )
    reader.start()
    command = [
        sys.executable,
        str(submission_dir / "inflate.py"),
        str(member_path),
        str(fifo_path),
    ]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"})
    started = time.monotonic()
    completed: subprocess.CompletedProcess[str] | None = None
    execution_error: BaseException | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=submission_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except BaseException as exc:
        execution_error = exc
    finally:
        os.close(dummy_fd)
        reader.join(timeout=reader_join_timeout_seconds)

    log_path = receipt_dir / f"fifo_inflate_{attempt_id}_pass_{pass_index}.log"
    log_payload = (
        f"command={json.dumps(command)}\n"
        f"returncode={None if completed is None else completed.returncode}\n"
        f"execution_error={execution_error!r}\n"
        f"stdout:\n{'' if completed is None else completed.stdout}\n"
        f"stderr:\n{'' if completed is None else completed.stderr}\n"
    )
    atomic_bytes(log_path, log_payload.encode())
    row = {
        "pass": pass_index,
        "command": command,
        "returncode": None if completed is None else completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "execution_error": None if execution_error is None else repr(execution_error),
        "reader_daemon": reader.daemon,
        "reader_alive_after_join": reader.is_alive(),
        **reader_result,
    }
    fifo_path.unlink(missing_ok=True)
    return row


def verify_runtime_fifo(
    *,
    candidate_path: Path,
    submission_dir: Path,
    receipt_dir: Path,
    expected_raw_sha256: str,
    expected_raw_bytes: int = EXPECTED_RAW_BYTES,
) -> dict[str, Any]:
    candidate_path = candidate_path.resolve()
    submission_dir = submission_dir.resolve()
    receipt_dir = receipt_dir.resolve()
    results_root = RESULTS_ROOT.resolve()
    if results_root not in receipt_dir.parents:
        raise ValueError(f"receipt directory must be a child of {results_root}")
    if str(receipt_dir).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise ValueError("durable receipt directory must not be transient")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{time.time_ns()}_{os.getpid()}"
    scratch = receipt_dir / "fifo_runtime_scratch" / f"attempt_{attempt_id}"
    scratch.mkdir(parents=True, exist_ok=False)
    member_path = scratch / "x"
    proof: dict[str, Any] = {
        "schema": "jrd_pr110_runtime_inflate_fifo_proof.v1",
        "attempt_id": attempt_id,
        "candidate_path": str(candidate_path),
        "expected_in_process_raw_sha256": expected_raw_sha256,
        "expected_raw_bytes": expected_raw_bytes,
        "runtime_inflate_py": {"path": str(submission_dir / "inflate.py")},
        "streaming_fifo_no_bulk_raw_materialized": True,
        "passes": [],
        "bit_exact": False,
        "scratch_cleaned_on_success": False,
    }
    try:
        proof.update(
            {
                "candidate_sha256": sha256_file(candidate_path),
                "candidate_bytes": candidate_path.stat().st_size,
                "runtime_inflate_py": {
                    "path": str(submission_dir / "inflate.py"),
                    "bytes": (submission_dir / "inflate.py").stat().st_size,
                    "sha256": sha256_file(submission_dir / "inflate.py"),
                },
                "submission_runtime": runtime_custody_module.runtime_custody(
                    submission_dir, REPO
                ),
            }
        )
        with zipfile.ZipFile(candidate_path, "r") as archive_zip:
            if archive_zip.namelist() != ["x"]:
                raise RuntimeError("candidate must contain exactly member x")
            atomic_bytes(member_path, archive_zip.read("x"))
        for pass_index in (1, 2):
            row = stream_inflate_pass(
                pass_index=pass_index,
                member_path=member_path,
                submission_dir=submission_dir,
                scratch_dir=scratch,
                receipt_dir=receipt_dir,
                attempt_id=attempt_id,
            )
            proof["passes"].append(row)
            if row["execution_error"] is not None:
                raise RuntimeError(
                    f"inflate pass {pass_index} execution failed: {row['execution_error']}"
                )
            if row["returncode"] != 0:
                raise RuntimeError(
                    f"inflate pass {pass_index} failed rc={row['returncode']}"
                )
            if row["reader_alive_after_join"]:
                raise RuntimeError(
                    f"inflate pass {pass_index} FIFO reader did not terminate"
                )
            if "error" in row:
                raise RuntimeError(f"inflate pass {pass_index} FIFO reader failed")
            if row["raw_bytes"] != expected_raw_bytes:
                raise RuntimeError(f"inflate pass {pass_index} emitted wrong byte count")
            if row["raw_sha256"] != expected_raw_sha256:
                raise RuntimeError(f"inflate pass {pass_index} raw SHA mismatch")
        if proof["passes"][0]["raw_sha256"] != proof["passes"][1]["raw_sha256"]:
            raise RuntimeError("inflate repeated outputs are not bit-identical")
        proof["bit_exact"] = True
        member_path.unlink()
        scratch.rmdir()
        parent = scratch.parent
        if not any(parent.iterdir()):
            parent.rmdir()
        proof["scratch_cleaned_on_success"] = True
        atomic_json(receipt_dir / "runtime_inflate_proof.json", proof)
        return proof
    except BaseException as exc:
        proof["error"] = f"{type(exc).__name__}: {exc}"
        retained = []
        for path in sorted(item for item in scratch.rglob("*") if item.is_file()):
            retained.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        proof["retained_failure_artifacts"] = retained
        proof["failure_cleanup_disposition"] = "retained fail-closed"
        atomic_json(receipt_dir / f"runtime_inflate_fifo_failed_{attempt_id}.json", proof)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--expected-raw-bytes", type=int, default=EXPECTED_RAW_BYTES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proof = verify_runtime_fifo(
        candidate_path=args.candidate,
        submission_dir=args.submission_dir,
        receipt_dir=args.receipt_dir,
        expected_raw_sha256=args.expected_raw_sha256,
        expected_raw_bytes=args.expected_raw_bytes,
    )
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
