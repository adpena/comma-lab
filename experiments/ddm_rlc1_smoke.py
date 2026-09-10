"""Structured setup probes and literal shell CUDA-gate controls; no score claim.

The setup observer is explicitly not a decode identity test. Full public twins
are retained by ddm_rlc1_public.py and are required independently.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_tc1_public_proof as proof
from tac.candidate_seal import _public_smoke_problems

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure")
SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/candidate_runtime")


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".new")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temp.replace(path)
    return value


def paths(role):
    return (ROOT / "candidate_runtime" if role == "candidate" else SOURCE), ROOT / "public_smoke" / role


def probe(role):
    runtime, work = paths(role)
    work.mkdir(parents=True, exist_ok=True)
    proof.build_libraries(runtime, work / "native")
    sys.path.insert(0, str(runtime))
    import torch
    from runtime import f26_inflate as public

    if Path(public.__file__).resolve() != runtime / "runtime/f26_inflate.py":
        raise ValueError("imported wrong receiver")
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    reached = False

    class ProbeReached(Exception):
        pass

    def observer(*args, **kwargs):
        nonlocal reached
        reached = True
        raise ProbeReached()

    public.decode_production_tokens = observer
    for key in (
        "F26_ADVISORY_DECODE_CACHE_ROOT",
        "F26_ADVISORY_PAIR_LIMIT",
        "TC1_RECEIVER_CHECKPOINT_DIR",
        "TC1_RECEIVER_STOP_AFTER",
    ):
        os.environ.pop(key, None)
    os.environ["F26_TOKEN_DECODER"] = "python"
    started = time.perf_counter()
    try:
        public.inflate_archive(
            runtime / "archive.zip",
            work / "not_rendered.raw",
            renderer_dir=runtime / "cpr1",
            device_name="cpu",
            num_threads=4,
            checkpoint_dir=work / "checkpoint",
        )
    except ProbeReached:
        pass
    if not reached:
        raise ValueError("public token setup not reached")
    return record(
        work / "DIRECT.json",
        {
            **proof.identity(runtime),
            "digest_definition": "tac.candidate_seal.measure_runtime_digest",
            "outcome": "REACHED_TOKEN_DECODE",
            "exception_class": None,
            "exception_message": "",
            "seconds": time.perf_counter() - started,
            "scope": "setup observer only; actual full public twins are separate",
        },
    )


def shell(role):
    runtime, work = paths(role)
    for name in ("data", "output", "scratch"):
        (work / name).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(runtime / "archive.zip") as z:
        (work / "data/p").write_bytes(z.read("p"))
    (work / "file_list.txt").write_text("0.hevc\n")
    env = dict(
        os.environ,
        PATH=str(REPO / ".venv/bin") + os.pathsep + os.environ["PATH"],
        TMPDIR=str(work / "scratch"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    for key in ("RLC1_ADVISORY_CPU", "TC3_ADVISORY_CPU", "CPR1_RC64_LIBRARY", "F26_CORRECTOR_NATIVE_LIBRARY"):
        env.pop(key, None)
    argv = ["bash", str(runtime / "inflate.sh"), str(work / "data"), str(work / "output"), str(work / "file_list.txt")]
    started = time.perf_counter()
    process = subprocess.run(argv, env=env, capture_output=True, timeout=180)
    (work / "shell.stdout").write_bytes(process.stdout)
    (work / "shell.stderr").write_bytes(process.stderr)
    message = "semantic_joint_ctxmix requires CUDA inflation on linux-nvidia-t4"
    if not process.returncode or message not in process.stderr.decode(errors="replace"):
        raise ValueError("shell did not reach declared CUDA gate")
    return record(
        work / "SHELL.json",
        {
            **proof.identity(runtime),
            "digest_definition": "tac.candidate_seal.measure_runtime_digest",
            "outcome": "REACHED_CUDA_GATE",
            "exception_class": "RuntimeError",
            "exception_message": message,
            "returncode": process.returncode,
            "seconds": time.perf_counter() - started,
            "argv": argv,
            "scope": "ordinary declared-axis gate; not full decode",
        },
    )


def collect():
    block = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "public_path_probe_seconds": 180,
        "public_path_probes": {},
        "inflate_sh_smokes": {},
    }
    for role in ("candidate", "frontier"):
        _, work = paths(role)
        block["public_path_probes"][role] = json.loads((work / "DIRECT.json").read_text())
        block["inflate_sh_smokes"][role] = json.loads((work / "SHELL.json").read_text())
    problems, _ = _public_smoke_problems(
        block,
        candidate_runtime_dir=ROOT / "candidate_runtime",
        candidate_archive_path=ROOT / "candidate_runtime/archive.zip",
        pointer_archive_sha256="986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857",
    )
    if problems:
        raise ValueError(problems)
    return record(ROOT / "PUBLIC_SMOKE.json", block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "shell", "collect"))
    parser.add_argument("--role", choices=("candidate", "frontier"), default="candidate")
    args = parser.parse_args()
    print(
        json.dumps(
            {"probe": lambda: probe(args.role), "shell": lambda: shell(args.role), "collect": collect}[args.stage]()
        )
    )
