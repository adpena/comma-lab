#!/usr/bin/env python3
"""Literal RLC1 public-entrypoint twins; retained full output and honest timing.

The two proof BLAS settings do not change the four-thread Torch renderer.
All stage state stays on SSD. Unknown host concurrency blocks timing admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from tac.candidate_seal import measure_runtime_digest
from tac.decode_wall_clock import LOCAL_SCHEMA, measure_receiver_digest

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure")
SOURCE_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/retained/source_parseback/0.raw"
)


def fact(path):
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha}


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return value


def load_snapshot():
    try:
        result = subprocess.run(["ps", "-axo", "pid,ppid,%cpu,command"], capture_output=True, text=True, timeout=10)
        code, output, error = result.returncode, result.stdout, result.stderr
    except OSError as exc:
        code, output, error = None, "", repr(exc)
    return {
        "load_average": list(os.getloadavg()),
        "inventory_returncode": code,
        "inventory": output,
        "inventory_error": error,
        "competing_process_count": None,
        "controlled_arm_worker_limit": 2,
        "count_status": "UNKNOWN: inventory unavailable or no independently audited workload classification",
    }


def compare_bytes(a, b):
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open("rb") as left, b.open("rb") as right:
        while chunk := left.read(8 * 1024**2):
            if chunk != right.read(len(chunk)):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 4), required=True)
    parser.add_argument("--stop-after", type=int, default=600)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve() or not 0 < args.stop_after <= 600:
        raise ValueError("wrong public proof root/boundary")
    runtime = ROOT / "candidate_runtime"
    work = ROOT / f"public_threads{args.threads}_g2"
    for name in ("data", "output", "scratch", "frame_checkpoints"):
        (work / name).mkdir(parents=True, exist_ok=True)
    archive = fact(runtime / "archive.zip")
    binding = {
        "runtime": measure_runtime_digest(runtime).sha256,
        "archive": archive,
        "producer": fact(Path(__file__)),
        "blas_threads": args.threads,
        "torch_threads": 4,
    }
    bound = work / "INPUTS.json"
    if bound.exists() and json.loads(bound.read_text()) != binding:
        raise ValueError("public proof resume code/config drift")
    record(bound, binding)
    done = work / "RESULT.json"
    if done.exists():
        result = json.loads(done.read_text())
        if result["candidate_raw"] != fact(work / "output/0.raw"):
            raise ValueError("completed raw custody drift")
        print(json.dumps(result), flush=True)
        return
    used = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    existing_raw = work / "output/0.raw"
    if existing_raw.exists():
        raise ValueError("raw without completed receipt: preserve; recover receipt without rerendering")
    required = 3662409600 + 192 * 1024**2
    if used + required > 8 * 1024**3 or shutil.disk_usage(ROOT).free < required + 16 * 1024**3:
        raise RuntimeError("STORAGE_BLOCK: preserve all outputs")
    with zipfile.ZipFile(runtime / "archive.zip") as z:
        payload = z.read("p")
    (work / "data/p").write_bytes(payload)
    (work / "file_list.txt").write_text("0.hevc\n")
    env = dict(
        os.environ,
        PATH=str(REPO / ".venv/bin") + os.pathsep + os.environ["PATH"],
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(work / "scratch"),
        RLC1_ADVISORY_CPU="1",
        RLC1_PROOF_BLAS_THREADS=str(args.threads),
        TC1_RECEIVER_CHECKPOINT_DIR=str(work / "frame_checkpoints"),
        TC1_RECEIVER_STOP_AFTER=str(args.stop_after),
        F26_TOKEN_DECODER="python",
    )
    for key in (
        "CPR1_RC64_LIBRARY",
        "F26_CORRECTOR_NATIVE_LIBRARY",
        "F26_HPAC_NATIVE_LIBRARY",
        "F26_ADVISORY_DECODE_CACHE_ROOT",
        "F26_ADVISORY_PAIR_LIMIT",
        "F26_ADVISORY_RENDER_WORKERS",
        "F26_ADVISORY_RENDER_RSS_BYTES",
    ):
        env.pop(key, None)
    command = [
        "bash",
        str(runtime / "inflate.sh"),
        str(work / "data"),
        str(work / "output"),
        str(work / "file_list.txt"),
    ]
    attempts = work / "attempts"
    attempts.mkdir(exist_ok=True)
    index = len(list(attempts.glob("attempt_*")))
    attempt = attempts / f"attempt_{index:04d}"
    attempt.mkdir()
    before = load_snapshot()
    record(attempt / "BEFORE.json", before)
    record(
        attempt / "COMMAND.json",
        {
            "argv": command,
            "env": {
                k: v
                for k, v in env.items()
                if k.startswith(("RLC1_", "TC1_", "F26_", "OMP_", "OPENBLAS_", "MKL_", "VECLIB_", "NUMEXPR_"))
            },
        },
    )
    resumed = (work / "frame_checkpoints/LATEST.json").exists() or (
        work / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
    ).exists()
    started = time.perf_counter()
    with (attempt / "run.log").open("wb") as log:
        process = subprocess.run(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    after = load_snapshot()
    record(
        attempt / "PROCESS.json",
        {
            "returncode": process.returncode,
            "wall_seconds": elapsed,
            "before": before,
            "after": after,
            "resumed": resumed,
        },
    )
    if process.returncode:
        if args.stop_after < 600 and (work / "frame_checkpoints/LATEST.json").exists():
            last = json.loads((work / "frame_checkpoints/LATEST.json").read_text())
            if last["frame"] == args.stop_after:
                print(
                    json.dumps({"partial_stage_complete": args.stop_after, "receipt": str(attempt / "PROCESS.json")}),
                    flush=True,
                )
                return
        raise RuntimeError("public entrypoint failed: " + str(attempt / "run.log"))
    reports = [
        json.loads(line)
        for line in (attempt / "run.log").read_text().splitlines()
        if line.startswith('{"archive_bytes"')
    ]
    if len(reports) != 1:
        raise ValueError("one full public report required")
    report = reports[0]
    output = work / "output/0.raw"
    actual, source = fact(output), fact(SOURCE_RAW)
    source_receipt = json.loads(SOURCE_RAW.with_name("PARSEBACK_RESULT.json").read_text())
    if any(source[k] != source_receipt["rendered_raw"][k] for k in ("bytes", "sha256")):
        raise ValueError("source retained raw custody mismatch")
    field = fact(work / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8")
    identity = compare_bytes(output, SOURCE_RAW)
    if not identity or field["sha256"] != "b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5":
        raise ValueError("public source identity failed; preserve all bytes")
    if (
        report["archive_sha256"] != archive["sha256"]
        or report["raw_sha256"] != actual["sha256"]
        or report["pair_count"] != 600
    ):
        raise ValueError("public report custody mismatch")
    timing = {
        "schema": LOCAL_SCHEMA,
        "axis": "macOS-CPU advisory",
        "measurement_kind": "public_entrypoint_decode",
        "completed": True,
        "cold_start": not resumed,
        "resumed": resumed,
        "checkpoint_resume": report["checkpoint_resume"],
        "checkpoint_resumed_from_frame": report["token_decoder"].get("checkpoint_resumed_from_frame", 0),
        "cpu_threads": 4,
        "proof_blas_threads": args.threads,
        "frames": list(range(600)),
        "host": platform.node(),
        "platform": platform.platform(),
        "hardware_fingerprint": platform.machine(),
        "command": command,
        "concurrency": before,
        "concurrency_after": after,
        "margin_time_basis": "unknown_host_concurrency",
        "runtime_dir": str(runtime),
        "archive_path": str(runtime / "archive.zip"),
        "runtime_sha256": binding["runtime"],
        "receiver_sha256": measure_receiver_digest(runtime),
        "archive_sha256": archive["sha256"],
        "wall_seconds": elapsed,
        "score_claim": False,
    }
    record(work / "TIMING.json", timing)
    result = {
        "binding": binding,
        "candidate_raw": actual,
        "source_raw": source,
        "decoded_field": field,
        "literal_public_output_byte_identity": identity,
        "all600": True,
        "report": report,
        "timing": fact(work / "TIMING.json"),
        "no_scorer_ran": True,
        "score_claim": False,
        "axis": "[macOS-CPU advisory]",
    }
    record(done, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
