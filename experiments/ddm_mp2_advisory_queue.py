#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the DDM-MP2 Stage-1 n600 advisory queue, one retained row at a time.

The queue starts only after the separately-owned HV1 control receipt is complete.
Each MP2 candidate is then launched through ``tools/launch_detached_process.py``
with its own work directory, launch receipt, exact archive/runtime pair, and
retained output.  The state file is a crash-resume checkpoint; it never stands
in for a scorer result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / ".omx" / "tmp" / "codex_runs"
SHIM_BIN = Path(
    "/Volumes/APDataStore/pact/ddm_wd2_width_distillation/python_shim_bin"
)
MIRROR = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")
VIDEO_NAMES = MIRROR / "public_test_video_names.txt"
CONTROL_RESULT = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/"
    "contest_auth_eval.json"
)
CONTROL_DONE = RUNS_ROOT / "hv1_base_advisory_n600.done"
AXIS = "[macOS-CPU advisory]"
POLL_SECONDS = 15.0
ATTEMPT_TIMEOUT_SECONDS = 6 * 60 * 60
WC1_ADMISSION_GATE = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/"
    "receipts/ADMISSION_GATE.json"
)


class QueueRefusal(RuntimeError):
    """Fail-closed refusal for an invalid or unsafe queue transition."""


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueRefusal(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QueueRefusal(f"JSON object required at {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _validate_done_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path)
    if receipt.get("schema") != "detached_local_process_done.v2":
        raise QueueRefusal(f"unexpected detached receipt schema at {path}")
    if receipt.get("rc") != 0:
        raise QueueRefusal(f"detached process did not complete successfully: {path}")
    return receipt


def _validate_result(
    path: Path,
    *,
    archive_sha256: str,
    archive_bytes: int,
) -> dict[str, Any]:
    result = _read_json(path)
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise QueueRefusal(f"missing provenance in {path}")
    blockers: list[str] = []
    if result.get("schema_version") != 1:
        blockers.append("schema_version")
    if result.get("n_samples") != 600:
        blockers.append("n_samples")
    if result.get("archive_size_bytes") != archive_bytes:
        blockers.append("archive_size_bytes")
    if provenance.get("archive_sha256") != archive_sha256:
        blockers.append("archive_sha256")
    if provenance.get("archive_size_bytes") != archive_bytes:
        blockers.append("provenance_archive_size_bytes")
    for key in ("avg_segnet_dist", "avg_posenet_dist"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not float(value) >= 0.0:
            blockers.append(key)
    if blockers:
        raise QueueRefusal(f"invalid n600 result {path}: {blockers}")
    return result


def _wait_for_control() -> dict[str, Any]:
    while not CONTROL_DONE.exists() or not CONTROL_RESULT.exists():
        time.sleep(POLL_SECONDS)
    _validate_done_receipt(CONTROL_DONE)
    return _validate_result(
        CONTROL_RESULT,
        archive_sha256=(
            "80d9c8c6fdc72caaa3e180a8abb2a859"
            "e7f316a484b38f33fe90d5701420178e"
        ),
        archive_bytes=182_759,
    )


def _clean_appledouble(roots: list[Path]) -> list[str]:
    removed: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        completed = subprocess.run(
            [
                "/usr/bin/find",
                str(root),
                "-name",
                "._*",
                "-type",
                "f",
                "-print",
                "-delete",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise QueueRefusal(
                f"AppleDouble cleanup failed for {root}: {completed.stderr[-2000:]}"
            )
        removed.extend(line for line in completed.stdout.splitlines() if line)
    return removed


def _load_wc1_builder():
    path = REPO_ROOT / "experiments/ddm_wc1_advisory_decode_wallclock.py"
    spec = importlib.util.spec_from_file_location("_ddm_wc1_consumer_builder", path)
    if spec is None or spec.loader is None:
        raise QueueRefusal(f"cannot load WC1 advisory builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validated_wc1_fast_path(path: Path = WC1_ADMISSION_GATE) -> dict[str, Any] | None:
    """Enable WC1 only from its complete, byte-identity admission receipt."""

    if not path.is_file():
        return None
    gate = _read_json(path)
    blockers: list[str] = []
    if gate.get("schema") != "ddm_wc1_advisory_fast_path_admission.v1":
        blockers.append("schema")
    if gate.get("complete") is not True or gate.get("identity_pass") is not True:
        blockers.append("identity")
    if gate.get("shipping_packet_touched") is not False:
        blockers.append("shipping_containment")
    environment = gate.get("consumer_environment")
    required_environment = {
        "F26_TOKEN_DECODER",
        "F26_HPAC_NATIVE_LIBRARY",
        "F26_ADVISORY_RENDER_WORKERS",
        "F26_ADVISORY_DECODE_CACHE_ROOT",
        "F26_ADVISORY_RENDER_RSS_BYTES",
    }
    if not isinstance(environment, dict) or set(environment) != required_environment:
        blockers.append("consumer_environment")
    code = gate.get("consumer_code")
    if not isinstance(code, dict):
        blockers.append("consumer_code")
    else:
        for label, fact in code.items():
            if not isinstance(fact, dict):
                blockers.append(f"consumer_code:{label}")
                continue
            observed = Path(str(fact.get("path", "")))
            if not observed.is_file() or _file_fact(observed) != fact:
                blockers.append(f"consumer_code:{label}")
    if blockers:
        raise QueueRefusal(f"WC1 admission gate is present but invalid: {blockers}")
    return {"gate": _file_fact(path), "environment": environment}


def _launch_argv(
    *,
    candidate_id: str,
    generation: dict[str, Any],
    attempt_dir: Path,
    attempt: int,
    wc1_fast_path: dict[str, Any] | None = None,
) -> tuple[list[str], Path, Path]:
    archive = Path(generation["archive"]["path"])
    inflate_sh = archive.parent / "inflate.sh"
    if wc1_fast_path is not None:
        builder = _load_wc1_builder()
        stage = attempt_dir / "wc1_advisory_generation"
        builder.prepare_advisory_runtime(archive.parent, stage)
        archive = stage / "archive.zip"
        inflate_sh = stage / "inflate.sh"
    work_dir = attempt_dir / "work"
    result_path = attempt_dir / "contest_auth_eval.json"
    launcher_dir = attempt_dir / "launcher"
    receipt_name = f"ddm_mp2_{candidate_id}_n600_attempt_{attempt:04d}"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "launch_detached_process.py"),
        "--output-dir",
        str(launcher_dir),
        "--cwd",
        str(REPO_ROOT),
        "--purpose",
        f"DDM MP2 {candidate_id} n600 {AXIS}",
        "--authority",
        AXIS,
        "--env",
        f"PATH={SHIM_BIN}:{os.environ.get('PATH', '')}",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    if wc1_fast_path is not None:
        for key, value in sorted(wc1_fast_path["environment"].items()):
            command.extend(["--env", f"{key}={value}"])
    command.extend(
        [
            "--done-receipt",
            receipt_name,
            "--verify-alive-secs",
            "10",
            "--",
            str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / "experiments" / "contest_auth_eval.py"),
        "--archive",
        str(archive),
        "--inflate-sh",
        str(inflate_sh),
        "--upstream-dir",
        str(MIRROR),
        "--video-names-file",
        str(VIDEO_NAMES),
        "--device",
        "cpu",
        "--inflate-device",
        "cpu",
        "--work-dir",
        str(work_dir),
        "--keep-work-dir",
        "--json-out",
        str(result_path),
        "--inflate-timeout",
        "5400",
        "--evaluate-timeout",
        "14400",
        ]
    )
    return command, RUNS_ROOT / f"{receipt_name}.done", result_path


def _candidate_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    generations = manifest.get("generations")
    if not isinstance(generations, list) or len(generations) != 8:
        raise QueueRefusal("MP2 build manifest must contain exactly eight generations")
    rows = [row for row in generations if row.get("candidate_id") != "hv1_base_control"]
    if len(rows) != 7:
        raise QueueRefusal("MP2 build manifest must contain seven non-control candidates")
    for row in rows:
        if row.get("complete") is not True or row.get("receiver_closed") is not True:
            raise QueueRefusal(f"receiver-open generation: {row.get('candidate_id')}")
    return rows


def _new_state(manifest_path: Path) -> dict[str, Any]:
    return {
        "schema": "ddm_mp2_advisory_queue.v1",
        "axis": AXIS,
        "generated_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "control": {"status": "WAITING", "result_path": str(CONTROL_RESULT)},
        "candidates": {},
        "complete": False,
        "score_claim": False,
    }


def _wait_for_candidate_attempt(
    *,
    state: dict[str, Any],
    state_path: Path,
    row_state: dict[str, Any],
    attempt_state: dict[str, Any],
    archive_sha256: str,
    archive_bytes: int,
) -> dict[str, Any]:
    done_receipt = Path(attempt_state["done_receipt"])
    result_path = Path(attempt_state["result_path"])
    started = dt.datetime.fromisoformat(str(attempt_state["started_utc"]))
    deadline = started.timestamp() + ATTEMPT_TIMEOUT_SECONDS
    while not done_receipt.exists():
        if time.time() >= deadline:
            attempt_state["status"] = "TIMEOUT_BLOCKED"
            row_state["status"] = "BLOCKED"
            state["updated_utc"] = _utc_now()
            _atomic_json(state_path, state)
            raise QueueRefusal(f"timed out waiting for {done_receipt}")
        time.sleep(POLL_SECONDS)
    receipt = _validate_done_receipt(done_receipt)
    result = _validate_result(
        result_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
    )
    attempt_state.update(
        {
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "done_receipt_sha256": _sha256(done_receipt),
            "elapsed_s": receipt.get("elapsed_s"),
            "result_sha256": _sha256(result_path),
        }
    )
    row_state.update(
        {
            "status": "COMPLETE",
            "result_path": str(result_path),
            "result_sha256": _sha256(result_path),
            "avg_segnet_dist": result["avg_segnet_dist"],
            "avg_posenet_dist": result["avg_posenet_dist"],
            "archive_size_bytes": result["archive_size_bytes"],
        }
    )
    state["updated_utc"] = _utc_now()
    _atomic_json(state_path, state)
    return result


def run_queue(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    rows = _candidate_rows(manifest)
    state_path = output_root / "ADVISORY_QUEUE_STATE.json"
    state = _read_json(state_path) if state_path.exists() else _new_state(manifest_path)
    if state.get("manifest", {}).get("sha256") != _sha256(manifest_path):
        raise QueueRefusal("queue state belongs to a different build manifest")

    control = _wait_for_control()
    wc1_fast_path = _validated_wc1_fast_path()
    state["control"] = {
        "status": "COMPLETE",
        "result_path": str(CONTROL_RESULT),
        "result_sha256": _sha256(CONTROL_RESULT),
        "avg_segnet_dist": control["avg_segnet_dist"],
        "avg_posenet_dist": control["avg_posenet_dist"],
        "archive_size_bytes": control["archive_size_bytes"],
    }
    state["updated_utc"] = _utc_now()
    _atomic_json(state_path, state)

    for generation in rows:
        candidate_id = str(generation["candidate_id"])
        archive_sha = str(generation["archive"]["sha256"])
        archive_bytes = int(generation["archive"]["bytes"])
        candidate_root = output_root / "advisory_n600_cpu" / candidate_id
        row_state = state["candidates"].setdefault(
            candidate_id,
            {"status": "PENDING", "attempts": []},
        )
        completed_result = row_state.get("result_path")
        if row_state.get("status") == "COMPLETE" and completed_result:
            _validate_result(
                Path(completed_result),
                archive_sha256=archive_sha,
                archive_bytes=archive_bytes,
            )
            continue
        if row_state.get("status") == "BLOCKED":
            raise QueueRefusal(
                f"candidate {candidate_id} is blocked; refusing an implicit retry"
            )
        if row_state.get("status") in {"LAUNCHING", "RUNNING"}:
            if not row_state.get("attempts"):
                raise QueueRefusal(f"candidate {candidate_id} has no resumable attempt")
            attempt_state = row_state["attempts"][-1]
            manifest_path_value = (
                Path(attempt_state["attempt_dir"]) / "launcher" / "launch_manifest.json"
            )
            if not manifest_path_value.exists():
                raise QueueRefusal(
                    f"candidate {candidate_id} stopped before a launch manifest was retained"
                )
            _wait_for_candidate_attempt(
                state=state,
                state_path=state_path,
                row_state=row_state,
                attempt_state=attempt_state,
                archive_sha256=archive_sha,
                archive_bytes=archive_bytes,
            )
            continue

        attempt = len(row_state["attempts"])
        attempt_dir = candidate_root / f"attempt_{attempt:04d}"
        removed = _clean_appledouble(
            [Path(generation["archive"]["path"]).parent, attempt_dir, MIRROR]
        )
        command, done_receipt, result_path = _launch_argv(
            candidate_id=candidate_id,
            generation=generation,
            attempt_dir=attempt_dir,
            attempt=attempt,
            wc1_fast_path=wc1_fast_path,
        )
        attempt_state = {
            "attempt": attempt,
            "status": "LAUNCHING",
            "attempt_dir": str(attempt_dir),
            "done_receipt": str(done_receipt),
            "result_path": str(result_path),
            "token_checkpoint": {
                "status": "BUILT_BY_CANDIDATE_RUNTIME",
                "reason": (
                    "the control checkpoint is archive/runtime-tree bound and is not "
                    "transferred across candidate archives"
                ),
            },
            "appledouble_removed_count": len(removed),
            "appledouble_removed": removed,
            "wc1_fast_path": (
                {"status": "DISABLED_NO_ADMISSION_GATE"}
                if wc1_fast_path is None
                else {"status": "ADMITTED_DEFAULT", **wc1_fast_path}
            ),
            "launch_argv": command,
            "started_utc": _utc_now(),
        }
        row_state["attempts"].append(attempt_state)
        row_state["status"] = "LAUNCHING"
        state["updated_utc"] = _utc_now()
        _atomic_json(state_path, state)

        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        attempt_state["launcher_returncode"] = completed.returncode
        attempt_state["launcher_stdout"] = completed.stdout
        attempt_state["launcher_stderr"] = completed.stderr
        if completed.returncode != 0:
            attempt_state["status"] = "LAUNCH_REFUSED"
            row_state["status"] = "BLOCKED"
            _atomic_json(state_path, state)
            raise QueueRefusal(f"launcher refused {candidate_id}: {completed.stderr[-2000:]}")
        attempt_state["status"] = "RUNNING"
        row_state["status"] = "RUNNING"
        state["updated_utc"] = _utc_now()
        _atomic_json(state_path, state)

        _wait_for_candidate_attempt(
            state=state,
            state_path=state_path,
            row_state=row_state,
            attempt_state=attempt_state,
            archive_sha256=archive_sha,
            archive_bytes=archive_bytes,
        )

    state["complete"] = all(
        row.get("status") == "COMPLETE" for row in state["candidates"].values()
    ) and len(state["candidates"]) == 7
    state["completed_utc"] = _utc_now() if state["complete"] else None
    state["updated_utc"] = _utc_now()
    _atomic_json(state_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        state = run_queue(args.manifest.resolve(), args.output_root.resolve())
    except QueueRefusal as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "COMPLETE", "candidate_count": 7, "state": state}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
