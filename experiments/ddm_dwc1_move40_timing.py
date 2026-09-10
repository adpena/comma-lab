"""Time the original move40 receiver backend through an explicit CPU proxy.

The public entrypoint refuses CPU. Its unmodified input verifier and inflation
backend are invoked directly instead. Native builds, imports, verification,
decode, checkpoint writes, and render are timed; public shell startup is absent.
No scorer runs. Call only after the preceding timing process has completed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from importlib import metadata, util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.dont_write_bytecode = True
from tac.candidate_seal import measure_runtime_digest

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock")
WORK = ROOT / "move40"
SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/candidate_runtime")
PROOF = SOURCE.parent.parent / "parseback/PARSEBACK_RESULT.json"
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
REPORT_PREFIX = "DWC1_MOVE40_REPORT "


def fact(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def storage(reserve=0):
    seen, logical, unique, allocated = set(), 0, 0, 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        info = path.stat()
        logical += info.st_size
        key = (info.st_dev, info.st_ino)
        if key not in seen:
            seen.add(key)
            unique += info.st_size
            allocated += getattr(info, "st_blocks", 0) * 512
    result = {"path_logical_bytes": logical, "unique_inode_bytes": unique,
              "allocated_bytes": allocated, "reserved_next_bytes": reserve,
              "budget_bytes": 8 * 1024**3, "free_bytes": shutil.disk_usage(ROOT).free,
              "policy": "retain every payload; count hardlinked bytes once by device and inode"}
    if unique + reserve > 8 * 1024**3 or result["free_bytes"] < 40 * 1024**3 + reserve:
        save(WORK / "STORAGE_BLOCKED.json", result)
        raise RuntimeError("storage reservation blocked; existing bytes retained")
    return result


def snapshot():
    value = {"load_average": list(os.getloadavg()), "quiesced": False,
             "normalization": None, "concurrent_process_count": None}
    try:
        result = subprocess.run(["ps", "-axo", "pid,ppid,%cpu,command"],
                                capture_output=True, text=True, timeout=5)
        value.update(process_inventory=result.stdout.splitlines(),
                     process_inventory_returncode=result.returncode,
                     process_inventory_error=result.stderr)
        if result.returncode == 0:
            value["concurrent_process_count"] = max(0, len(value["process_inventory"]) - 1)
    except (OSError, subprocess.TimeoutExpired) as exc:
        value["process_inventory_error"] = repr(exc)
    return value


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    storage(0 if (WORK / "TIMING.json").exists() else 4 * 1024**3)
    proof = json.loads(PROOF.read_text())
    archive_fact = fact(SOURCE / "archive.zip")
    if archive_fact != proof["archive"]:
        raise RuntimeError("move40 archive differs from named parseback proof")
    source_sha = measure_runtime_digest(SOURCE).sha256
    runtime = WORK / "runtime"
    if not runtime.exists():
        shutil.copytree(SOURCE, runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.dylib", "._*"))
    if measure_runtime_digest(runtime).sha256 != source_sha:
        raise RuntimeError("copied original receiver differs from source")
    for name in ("data", "output", "build", "frame_checkpoints", "scratch"):
        (WORK / name).mkdir(exist_ok=True)
    with zipfile.ZipFile(runtime / "archive.zip") as archive:
        if archive.namelist() != ["p"]:
            raise RuntimeError("unexpected public payload member layout")
        payload = archive.read("p")
    target = WORK / "data/p"
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError("retained payload drift")
    if not target.exists():
        target.write_bytes(payload)
    binding = {"schema": "ddm_dwc1.move40_binding.v1", "source_runtime": str(SOURCE),
               "runtime_sha256": source_sha, "copied_runtime": str(runtime),
               "archive": fact(runtime / "archive.zip"), "payload": fact(target),
               "source_proof": fact(PROOF), "producer": fact(Path(__file__)),
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               "seed": 20260910, "public_entrypoint": "original CUDA refusal unchanged",
               "expected_raw": proof["rendered_raw"],
               "expected_token_sha256": proof["inflate_report"]["token_decoder"]["decoded_token_sha256"]}
    path = WORK / "BINDING.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise RuntimeError("resume binding drift")
    save(path, binding)
    return binding


def worker():
    """Run in the supervised child; every native library and stage is retained."""
    runtime = WORK / "runtime"
    binding = json.loads((WORK / "BINDING.json").read_text())
    if fact(Path(__file__)) != binding["producer"] or measure_runtime_digest(runtime).sha256 != binding["runtime_sha256"]:
        raise RuntimeError("worker producer or copied receiver changed")
    if any(os.environ.get(key) != "4" for key in THREAD_KEYS):
        raise RuntimeError("worker must use exactly four CPU threads")
    import brotli
    if metadata.version("Brotli") != "1.2.0" or not callable(brotli.decompress):
        raise RuntimeError("public receiver requires Brotli==1.2.0")
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        raise RuntimeError("public receiver requires a C compiler")
    build = WORK / "build"
    libraries = {"CPR1_RC64_LIBRARY": build / "rc64_backend.so",
                 "F26_CORRECTOR_NATIVE_LIBRARY": build / "f26_corrector_native.so"}
    commands = [
        [compiler, "-O3", "-std=c11", "-shared", "-fPIC",
         str(runtime / "runtime/entropy/rc64_backend.c"), "-o", str(libraries["CPR1_RC64_LIBRARY"])],
        [compiler, "-O3", "-std=c11", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math",
         str(runtime / "runtime/f26_corrector_native.c"), "-lm", "-o", str(libraries["F26_CORRECTOR_NATIVE_LIBRARY"])],
    ]
    build_receipt = build / "BUILD.json"
    if build_receipt.exists():
        prior = json.loads(build_receipt.read_text())
        if prior["commands"] != commands or any(fact(path) != prior["libraries"][key] for key, path in libraries.items()):
            raise RuntimeError("retained native build drift")
    else:
        save(build / "BUILD_LAUNCH.json", {"commands": commands, "binding": binding,
             "compiler": fact(Path(compiler))})
        for command in commands:
            subprocess.run(command, check=True)
        save(build_receipt, {"commands": commands,
             "libraries": {key: fact(path) for key, path in libraries.items()}})
    os.environ.update({key: str(path) for key, path in libraries.items()})
    sys.path.insert(0, str(runtime))
    spec = util.spec_from_file_location("dwc1_move40_original_inflate", runtime / "inflate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import original input verifier")
    receiver = util.module_from_spec(spec)
    spec.loader.exec_module(receiver)
    receiver._verify_input(WORK / "data", runtime / "archive.zip")
    inflation = importlib.import_module("runtime.f26_inflate")
    if Path(inflation.__file__).resolve() != runtime / "runtime/f26_inflate.py":
        raise RuntimeError("inflation backend escaped the copied receiver")
    report = inflation.inflate_archive(runtime / "archive.zip", WORK / "output/0.raw",
        renderer_dir=runtime / "cpr1", device_name="cpu", num_threads=4,
        checkpoint_dir=WORK / "output/.f26_decode_checkpoints")
    print(REPORT_PREFIX + json.dumps(report, sort_keys=True), flush=True)


def run(timeout):
    binding = prepare()
    complete = WORK / "TIMING.json"
    if complete.exists():
        value = json.loads(complete.read_text())
        if value["binding"] != binding:
            raise RuntimeError("complete receipt binding drift")
        return value
    attempt = WORK / f"attempt_{len(list(WORK.glob('attempt_*'))):04d}"
    attempt.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("F26_", "TC1_RECEIVER_", "TC3_ADVISORY_", "CPR1_RC64_"))}
    env.update(dict.fromkeys(THREAD_KEYS, "4"))
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="20260910",
               TMPDIR=str(WORK / "scratch"), F26_TOKEN_DECODER="python",
               TC1_RECEIVER_CHECKPOINT_DIR=str(WORK / "frame_checkpoints"),
               TC1_RECEIVER_STOP_AFTER="600")
    command = [sys.executable, str(Path(__file__).resolve()), "--resume-from", str(WORK), "--worker"]
    resumed = ((WORK / "frame_checkpoints/LATEST.json").exists()
               or (WORK / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.json").exists()
               or (WORK / "build/BUILD.json").exists())
    native_build_reused = (WORK / "build/BUILD.json").exists()
    before = snapshot()
    save(attempt / "LAUNCH.json", {"command": command, "binding": binding,
         "environment": {key: env[key] for key in (*THREAD_KEYS, "TMPDIR", "PYTHONHASHSEED",
             "F26_TOKEN_DECODER", "TC1_RECEIVER_CHECKPOINT_DIR", "TC1_RECEIVER_STOP_AFTER")},
         "concurrency": before, "resumed": resumed, "native_build_reused": native_build_reused,
         "timeout_seconds": timeout,
         "retention": storage(4 * 1024**3)})
    started = time.monotonic()
    timed_out = False
    with (attempt / "stdout.log").open("wb") as stream:
        process = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    elapsed = time.monotonic() - started
    reports = [json.loads(line[len(REPORT_PREFIX):]) for line in (attempt / "stdout.log").read_text().splitlines()
               if line.startswith(REPORT_PREFIX)]
    value = {"schema": "ddm_dwc1.raw_timing.v1", "binding": binding, "command": command,
             "axis": "[macOS-CPU advisory]", "measurement_kind": "cuda_path_equivalent_proxy",
             "public_entrypoint_executed": False, "public_shell_startup_included": False,
             "timed_scope": "child process startup, native build/setup (retained build reused on resume), original input verifier, original CPU decode/backend and rendering; public shell startup excluded",
             "score_claim": False, "wall_seconds": elapsed, "cpu_threads": 4,
             "host": platform.node(), "platform": platform.platform(), "resumed": resumed,
             "native_build_reused": native_build_reused, "cold_start": not resumed,
             "returncode": code, "timed_out": timed_out, "concurrency_before": before,
             "concurrency_after": snapshot(), "log": fact(attempt / "stdout.log"),
             "report": reports[-1] if reports else None}
    latest = WORK / "frame_checkpoints/LATEST.json"
    value["last_token_checkpoint"] = json.loads(latest.read_text()) if latest.exists() else None
    save(attempt / "RESULT.json", value)
    value["retention"] = storage()
    if code == 0:
        if len(reports) != 1 or reports[0]["pair_count"] != 600:
            raise RuntimeError("worker exited without a complete n600 report")
        value["raw"] = fact(WORK / "output/0.raw")
        value["tokens"] = fact(WORK / "output/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8")
        if (value["raw"]["sha256"] != reports[0]["raw_sha256"]
                or value["raw"]["sha256"] != binding["expected_raw"]["sha256"]
                or value["tokens"]["sha256"] != binding["expected_token_sha256"]):
            raise RuntimeError("proxy output identity differs from source parseback proof")
        save(complete, value)
    save(attempt / "RESULT.json", value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2400)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    resume = args.resume_from.resolve()
    # A sibling work dir (e.g. ROOT/move40_quiesced) re-times the same copied receiver cold
    # under a measured concurrency sampler; the default ROOT/move40 run is retained as-is.
    if resume.parent != ROOT or not resume.name.startswith("move40") or not 0 < args.timeout_seconds <= 2400:
        raise ValueError("invalid resume root or wall budget")
    WORK = resume
    worker() if args.worker else run(args.timeout_seconds)
