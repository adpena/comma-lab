#!/usr/bin/env python3
"""Cold real move42 cure public parse-back, with retained resume state and builds.

No scorer, timing window, calibration, or dispatch. Public wall time is only
loaded macOS diagnostic data. Raw identity compares every output byte.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_rlc1_public as prior
from experiments import ddm_rlc4_rebase as rebase
from tac.candidate_seal import measure_runtime_digest
from tac.decode_wall_clock import measure_receiver_digest

ROOT = rebase.ROOT
SOURCE_RAW = Path("/Volumes/VertigoDataTier/pact/ddm_rp1_round2/parseback/0.raw")
SOURCE_RAW_SHA = "1db04341d5ae9972d6296831a9e822d50a5d0f25ef77af378eae75b69e917159"


def run(generation):
    rebase.check_base()
    runtime = ROOT / "candidate_runtime"
    work = ROOT / f"public_{generation}"
    for name in ("data", "output", "scratch", "frame_checkpoints", "retained_native"):
        (work / name).mkdir(parents=True, exist_ok=True)
    archive = prior.fact(runtime / "archive.zip")
    binding = {"runtime_sha256": measure_runtime_digest(runtime).sha256, "archive": archive,
               "receiver_sha256": measure_receiver_digest(runtime), "source_raw_sha256": SOURCE_RAW_SHA,
               "producer": prior.fact(Path(__file__)),
               "native_cache_source": prior.fact(REPO / "experiments/ddm_rlc4_native_cache.py")}
    path = work / "INPUTS.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("public proof source/config drift")
    prior.record(path, binding)
    done = work / "RESULT.json"
    if done.exists():
        result = json.loads(done.read_text())
        if result["candidate_raw"] != prior.fact(work / "output/0.raw"):
            raise ValueError("completed output custody drift")
        return result
    output = work / "output/0.raw"
    if output.exists():
        raise ValueError("unreceipted raw: preserve and recover the log, never rerender over it")
    seen, used = set(), 0
    for p in ROOT.rglob("*"):
        if p.is_file():
            st = p.stat()
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                seen.add(key)
                used += st.st_size
    need = 3662409600 + 256 * 1024**2
    if used + need > 8 * 1024**3 or shutil.disk_usage(ROOT).free < need + 16 * 1024**3:
        raise RuntimeError("STORAGE_BLOCK: keep all bytes")
    with zipfile.ZipFile(runtime / "archive.zip") as z:
        rebase.blob(work / "data/p", z.read("p"))
    rebase.blob(work / "file_list.txt", b"0.hevc\n")
    env = dict(os.environ, PATH=str(REPO / ".venv/bin") + os.pathsep + os.environ["PATH"],
               PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work / "scratch"), RLC1_ADVISORY_CPU="1",
               RLC1_PROOF_BLAS_THREADS="4", TC1_RECEIVER_CHECKPOINT_DIR=str(work / "frame_checkpoints"),
               TC1_RECEIVER_STOP_AFTER="600", F26_TOKEN_DECODER="python",
               CC=str(REPO / "experiments/ddm_rlc4_native_cache.py"),
               RLC4_NATIVE_CACHE=str(work / "retained_native"))
    for k in ("CPR1_RC64_LIBRARY", "F26_CORRECTOR_NATIVE_LIBRARY", "F26_HPAC_NATIVE_LIBRARY",
              "RLC1_GEOMETRY_LIBRARY", "F26_ADVISORY_DECODE_CACHE_ROOT", "F26_ADVISORY_PAIR_LIMIT",
              "F26_ADVISORY_RENDER_WORKERS", "F26_ADVISORY_RENDER_RSS_BYTES"):
        env.pop(k, None)
    command = ["bash", str(runtime / "inflate.sh"), str(work / "data"), str(work / "output"), str(work / "file_list.txt")]
    attempts = work / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f"attempt_{len(list(attempts.glob('attempt_*'))):04d}"
    attempt.mkdir()
    resumed = (work / "frame_checkpoints/LATEST.json").exists() or (work / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8").exists()
    prior.record(attempt / "COMMAND.json", {"argv": command, "env": {
        k: v for k, v in env.items() if k.startswith(("RLC", "TC1", "F26", "OMP", "MKL", "OPENBLAS", "VECLIB"))
        or k in ("CC", "TMPDIR", "PATH", "PYTHONDONTWRITEBYTECODE")}, "binding": binding,
        "resumed": resumed, "score_claim": False})
    started = time.perf_counter()
    with (attempt / "run.log").open("wb") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    prior.record(attempt / "PROCESS.json", {"returncode": result.returncode, "wall_seconds": wall,
        "resumed": resumed, "axis": "[macOS-CPU advisory diagnostic; no timing window]", "score_claim": False})
    if result.returncode:
        raise RuntimeError(f"public shell failed; retained log {attempt / 'run.log'}")
    reports = [json.loads(line) for line in (attempt / "run.log").read_text().splitlines()
               if line.startswith('{"archive_bytes"')]
    if len(reports) != 1:
        raise ValueError("exactly one public report required")
    report = reports[0]
    candidate_raw, pointer_raw = prior.fact(output), prior.fact(SOURCE_RAW)
    source_receipt = json.loads(SOURCE_RAW.with_name("PARSEBACK_RESULT.json").read_text())
    if pointer_raw != source_receipt["rendered_raw"] or pointer_raw["sha256"] != SOURCE_RAW_SHA:
        raise ValueError("move42 retained raw drift")
    field = prior.fact(work / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8")
    identity = prior.compare_bytes(output, SOURCE_RAW)
    if not identity or candidate_raw["bytes"] != 3662409600 or field["sha256"] != rebase.trace.FIELD_SHA:
        raise ValueError("full move42 raw/field identity failed; retain all bytes")
    if (report["archive_sha256"] != archive["sha256"] or report["raw_sha256"] != candidate_raw["sha256"]
            or report["pair_count"] != 600 or report["token_cache"]["status"] != "DISABLED"):
        raise ValueError("public report identity/cache mismatch")
    if binding["runtime_sha256"] != measure_runtime_digest(runtime).sha256:
        raise ValueError("runtime changed during proof")
    return prior.record(done, {"binding": binding, "candidate_raw": candidate_raw, "pointer_raw": pointer_raw,
        "decoded_field": field, "literal_public_output_byte_identity": True, "n_samples": 600,
        "cold_start": not resumed, "checkpoint_resume": report["checkpoint_resume"], "report": report,
        "command": command, "stdout": prior.fact(attempt / "run.log"), "wall_seconds": wall,
        "source_receipt": prior.fact(SOURCE_RAW.with_name("PARSEBACK_RESULT.json")),
        "no_scorer_ran": True, "score_claim": False, "axis": "[macOS-CPU advisory]"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--generation", default="rlc4")
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve() or not args.generation.isalnum():
        raise ValueError("invalid resume root or generation")
    print(json.dumps(run(args.generation)), flush=True)


if __name__ == "__main__":
    main()
