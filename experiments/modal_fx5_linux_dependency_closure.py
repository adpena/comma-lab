"""FX5 — the Linux x86_64 half of FX1's dependency-closure receipt.

FX1 (`3e616f568a`) cured the `constriction` blocker and MEASURED it on macOS arm64,
then labelled its own platform statement a TOY-BRACKET: the install used a locally
retained wheel cache because that sandbox cannot resolve PyPI DNS, and the live-network
trial terminated rc=2 without publishing a partial target.

This dispatch closes exactly that gap and nothing else. It runs the CURED entrypoint on
Linux x86_64 CPython 3.11 from a container that provably lacks `constriction` at start,
WITH network fetch, and measures resolve/download/install/import.

COMPUTE-SPLIT COMPLIANCE (operator 2026-08-09): Modal is authorized here because a Linux
x86_64 wheel physically cannot execute on local Metal. CPU only, no GPU, seconds-to-minutes.

SCOPE (stated, not implied): dependency bootstrap + receiver import only. This is NOT a
full n600 decode and NOT a whole-job (`timeout-minutes: 30`, Catalog #835) total. The
rate denominator (`evaluate.py:64` rglob sum, Catalog #812) is deliberately NOT measured
here — it is platform-independent directory arithmetic, measurable locally without paying
for a mount of the video tree.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import modal

APP_NAME = "comma-fx5-linux-depclose"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCAL_TREE = REPO_ROOT / "src" / "tac" / "pr130_runtime" / "fx1_runtime_tree"
REMOTE_TREE = "/runtime/fx1_runtime_tree"

app = modal.App(APP_NAME, include_source=False)

# The image installs `uv` (bootstrap infrastructure, which inflate.sh requires on PATH)
# and MUST NOT pre-install `constriction` — the absence of constriction at container
# start is the entire premise of the measurement.
fx5_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv==0.9.2")
    .env({"PYTHONDONTWRITEBYTECODE": "1"})
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:the cured FX1 runtime tree under our custody
        str(LOCAL_TREE), remote_path=REMOTE_TREE
    )
    .add_local_python_source(  # MODAL_ENTRYPOINT_SELF_MOUNT_OK:include_source=False
        "modal_fx5_linux_dependency_closure"
    )
)


# SECOND IMAGE — contest-SHAPED. `upstream/uv.lock` pins torch 2.10.0+cpu and numpy as
# contest-runtime dependencies, so the real eval host HAS them; the bare image above does
# not, which is exactly why it exposed FX1's incomplete closure statement. This image
# mirrors the contest runtime for torch/numpy while STILL omitting constriction, so the
# bootstrap under test remains the only thing being bootstrapped.
fx5_contest_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv==0.9.2")
    .pip_install(
        "numpy",
        "torch==2.10.0+cpu",
        extra_options="--index-url https://download.pytorch.org/whl/cpu "
        "--extra-index-url https://pypi.org/simple",
    )
    .env({"PYTHONDONTWRITEBYTECODE": "1"})
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:the cured FX1 runtime tree under our custody
        str(LOCAL_TREE), remote_path=REMOTE_TREE
    )
    .add_local_python_source(  # MODAL_ENTRYPOINT_SELF_MOUNT_OK:include_source=False
        "modal_fx5_linux_dependency_closure"
    )
)


def _run_closure() -> dict:
    """Run the cured entrypoint from a provably-clean Linux x86_64 CPython 3.11 env."""
    import hashlib
    import importlib.util
    import os
    import platform

    result: dict = {
        "schema": "comma_lab.fx5_linux_closure.v1",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "libc": "-".join(x for x in platform.libc_ver() if x),
        },
    }

    # (1) PRECONDITION: constriction must be absent BEFORE the entrypoint runs.
    # This is the premise; if it is already importable the measurement is void.
    result["precondition_find_spec_is_none"] = (
        importlib.util.find_spec("constriction") is None
    )

    tree = pathlib.Path(REMOTE_TREE)
    # (2) CUSTODY: hash the mounted receiver modules and compare to the FX1 manifest,
    # so we know we measured OUR cured tree and not something else.
    manifest = json.loads((tree / "runtime-dependencies.json").read_text())
    copied = manifest["source"]["copied_files"]
    observed = {}
    for name in sorted(copied):
        observed[name] = hashlib.sha256((tree / name).read_bytes()).hexdigest()
    result["receiver_sha256_observed"] = observed
    result["receiver_sha256_matches_manifest"] = observed == copied

    # (3) NETWORK FETCH: fresh target dir, so the entrypoint MUST resolve+download+install.
    deps_dir = "/tmp/fx5-runtime-deps"
    env = dict(os.environ)
    env["PR130_RUNTIME_DEPS_DIR"] = deps_dir
    env["PR130_DEPENDENCY_SMOKE_ONLY"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    t0 = time.perf_counter()
    cold = subprocess.run(
        # GROUP_KILL_OK: runs INSIDE the fx5 Modal container, whose image mounts only
        # the fx1 runtime tree as data (`add_local_dir(LOCAL_TREE, REMOTE_TREE)`) plus this
        # entrypoint module (`add_local_python_source`), with `include_source=False` and no
        # tac wheel — `import tac` cannot resolve there. Mounting `src` to satisfy the gate
        # would also weaken the measurement, whose premise is a provably-minimal container.
        ["/bin/sh", str(tree / "inflate.sh")],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    cold_s = time.perf_counter() - t0

    result["cold"] = {
        "returncode": cold.returncode,
        "wall_clock_s": cold_s,
        "stdout": cold.stdout[-4000:],
        "stderr": cold.stderr[-4000:],
        "dependency_ready_line_present": "PR130_DEPENDENCY_READY" in cold.stdout,
    }

    # (4) WARM repeat against the published target — the steady-state contest cost.
    t1 = time.perf_counter()
    warm = subprocess.run(
        # GROUP_KILL_OK: runs INSIDE the fx5 Modal container, whose image mounts only
        # the fx1 runtime tree as data (`add_local_dir(LOCAL_TREE, REMOTE_TREE)`) plus this
        # entrypoint module (`add_local_python_source`), with `include_source=False` and no
        # tac wheel — `import tac` cannot resolve there. Mounting `src` to satisfy the gate
        # would also weaken the measurement, whose premise is a provably-minimal container.
        ["/bin/sh", str(tree / "inflate.sh")],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    warm_s = time.perf_counter() - t1
    result["warm"] = {
        "returncode": warm.returncode,
        "wall_clock_s": warm_s,
        "stdout": warm.stdout[-2000:],
        "dependency_ready_line_present": "PR130_DEPENDENCY_READY" in warm.stdout,
    }

    # (5) WHEEL CUSTODY: what actually landed, and was it a wheel (never a source build)?
    installed = []
    dp = pathlib.Path(deps_dir)
    if dp.is_dir():
        for p in sorted(dp.rglob("*")):
            if p.is_file():
                installed.append(
                    {"path": str(p.relative_to(dp)), "bytes": p.stat().st_size}
                )
    result["installed_target"] = {
        "dir": deps_dir,
        "exists": dp.is_dir(),
        "file_count": len(installed),
        "total_bytes": sum(f["bytes"] for f in installed),
        "files": installed[:40],
    }

    # (6) POSITIVE CONTROL: a pre-existing invalid target must REFUSE (rc=65), never
    # overwrite. FX1 measured this 1/1 on macOS; re-run it on Linux.
    bad = "/tmp/fx5-invalid-target"
    pathlib.Path(bad).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(bad) / "not-a-package.txt").write_text("invalid")
    ctl_env = dict(env)
    ctl_env["PR130_RUNTIME_DEPS_DIR"] = bad
    ctl = subprocess.run(
        # GROUP_KILL_OK: runs INSIDE the fx5 Modal container, whose image mounts only
        # the fx1 runtime tree as data (`add_local_dir(LOCAL_TREE, REMOTE_TREE)`) plus this
        # entrypoint module (`add_local_python_source`), with `include_source=False` and no
        # tac wheel — `import tac` cannot resolve there. Mounting `src` to satisfy the gate
        # would also weaken the measurement, whose premise is a provably-minimal container.
        ["/bin/sh", str(tree / "inflate.sh")],
        capture_output=True,
        text=True,
        env=ctl_env,
        timeout=300,
    )
    result["invalid_target_control"] = {
        "returncode": ctl.returncode,
        "expected_returncode": 65,
        "passed": ctl.returncode == 65,
        "stderr": ctl.stderr[-1000:],
    }

    return result


@app.function(image=fx5_image, timeout=900)
def linux_dependency_closure() -> dict:
    """BARE image: no numpy, no torch. Exposes the TRUE declared-closure statement."""
    r = _run_closure()
    r["image_shape"] = "bare_debian_slim_py311_uv_only"
    return r


@app.function(image=fx5_contest_image, timeout=900)
def linux_contest_shaped_closure() -> dict:
    """CONTEST-SHAPED image: numpy + torch 2.10.0+cpu present, constriction absent."""
    r = _run_closure()
    r["image_shape"] = "contest_shaped_numpy_torch2100cpu_no_constriction"
    return r


@app.local_entrypoint()
def main(out_json: str = "", contest_shaped: bool = False) -> None:
    fn = linux_contest_shaped_closure if contest_shaped else linux_dependency_closure
    receipt = fn.remote()
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if out_json:
        pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_json).write_text(text + "\n")
        print(f"wrote {out_json}", file=sys.stderr)
    print(text)
