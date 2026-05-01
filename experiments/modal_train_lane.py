"""Run a `scripts/remote_lane_*.sh` on Modal T4 / A10G — reliable training surface.

Why: 2026-04-29 night session showed Vast.ai 4090 NVDEC bad-host rate ≈ 85%.
~$5 burned across 5 dispatch rounds for 0 trained lanes. Modal is useful for
build/training work after repeated Vast.ai NVDEC host failures.
This wrapper forces Modal auth-eval to CPU to avoid NVDEC, so any score it
prints is advisory only and non-promotable until the exact archive is rerun
through CUDA auth eval.

Pattern mirrors `experiments/modal_auth_eval.py` (commit 11d56896).

USAGE — RECOMMENDED (`--detach` for unattended training):
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
        experiments/modal_train_lane.py \\
        --lane-script scripts/remote_lane_omega_hessian_qat.sh \\
        --label lane_omega_hessian --gpu T4 --timeout-hours 10

Without `--detach` the local CLI blocks for the full 8-14h training duration
— terminal disconnect = lost job. With `--detach`, Modal keeps the run
alive and you can poll with `experiments/modal_recover_lane.py`, or stream logs
via `modal app logs <app-id>`. Round 12 caught this.

For PARALLEL dispatch of multiple lanes, fire 6 separate `modal run --detach`
in background (each gets its own container — Modal handles concurrency
natively).

Output: artifacts saved to `experiments/results/lane_<label>_modal/`. Any
score extracted from `contest_auth_eval.json` is labelled advisory/non-promotable
by the recovery helper when the recorded device is not CUDA.

Cross-references:
  - feedback_vastai_nvdec_roulette_pivot_to_modal_20260429
  - project_modal_pipeline_trusted_lane_g_v3_1_04_20260429
  - feedback_canonical_lane_lifecycle_DECISION_TREE_20260428
"""
from __future__ import annotations

import modal

app = modal.App("comma-train-lane")

# Image with all deps. ffmpeg-master (with in_primaries support) is pulled
# at build time via the same BtbN nightly that setup_full.sh uses on Vast.ai.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "unzip", "wget", "curl", "build-essential",
        "libgl1", "libglib2.0-0",  # opencv runtime
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "av",
        "click",
        "nvidia-dali-cuda120",
        "tqdm",
        "timm",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "pydantic>=2.0",
    )
    # Install ffmpeg-master from BtbN nightly (has in_primaries + libsvtav1).
    # Mirrors `scripts/remote_setup_full.sh` Stage 6 EXACTLY — johnvansickle
    # builds did NOT have in_primaries (review of 2026-04-29 session showed
    # `ffmpeg-git-20240629` only had in_color_matrix, missing in_primaries).
    # The BtbN URL is canonical and includes a build-time verification that
    # the binary has both in_primaries (needed by inflate.sh:require_ffmpeg_parity)
    # AND libsvtav1 (needed by mask_codec).
    .run_commands(
        "curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o /tmp/ffmpeg-master.tar.xz",
        "cd /opt && tar xf /tmp/ffmpeg-master.tar.xz",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-master",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-new",
        # Build-time gate: fail image build if in_primaries missing.
        "/usr/local/bin/ffmpeg-master -hide_banner -h filter=scale 2>&1 | grep -q in_primaries || (echo FATAL: ffmpeg-master lacks in_primaries; exit 1)",
        "/usr/local/bin/ffmpeg-master -encoders 2>&1 | grep -qi svtav1 || (echo FATAL: ffmpeg-master lacks libsvtav1; exit 1)",
        "rm /tmp/ffmpeg-master.tar.xz",
    )
    # uv (used by inflate.sh)
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
)


# Mount the entire pact source tree at /workspace/pact (matches Vast.ai
# convention so lane scripts work without modification).
training_image = (
    image
    .add_local_dir("src", remote_path="/workspace/pact/src")
    .add_local_dir("scripts", remote_path="/workspace/pact/scripts")
    # Mount entire submissions/ — lanes anchor on different baselines
    # (UNIWARD uses submissions/baseline_dilated_h64_0_90/, etc.)
    .add_local_dir("submissions", remote_path="/workspace/pact/submissions")
    .add_local_dir("upstream", remote_path="/workspace/pact/upstream")
    .add_local_dir("experiments", remote_path="/workspace/pact/experiments")
    .add_local_dir("tools", remote_path="/workspace/pact/tools")
    .add_local_file("pyproject.toml", remote_path="/workspace/pact/pyproject.toml")
)


def _run_lane_inner(lane_script: str, label: str, env_overrides: dict,
                    max_seconds: int = 14 * 3600) -> dict:
    """Container-side execution. Imports MUST be local (Modal serialization)."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    image_workspace = Path("/workspace/pact")

    # COPY mounted source to a writable workspace. Modal's add_local_dir mounts
    # may be read-only at runtime (modal_auth_eval avoids this entirely by
    # using tempfile/copy). Lane scripts write env.sh + need scripts/ to be
    # writable for the NVDEC probe stub. Round 4 caught this.
    workspace = Path("/tmp/pact")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    print(f"[modal-train-lane] copying mounted source → {workspace}")
    for sub in ("src", "scripts", "submissions", "upstream", "experiments", "tools"):
        src_path = image_workspace / sub
        if src_path.exists():
            shutil.copytree(src_path, workspace / sub, symlinks=True)
    pp = image_workspace / "pyproject.toml"
    if pp.exists():
        shutil.copy2(pp, workspace / "pyproject.toml")

    os.chdir(workspace)

    # sys.path injection (matches modal_auth_eval.py pattern). Avoids
    # `pip install -e .` which would write src/tac.egg-info/ — risky.
    sys.path.insert(0, str(workspace / "src"))
    sys.path.insert(0, str(workspace / "upstream"))

    # Write env.sh that lane scripts source. Mirrors the one that
    # remote_setup_full.sh writes on Vast.ai.
    # FFmpeg-master path: BtbN tarball extracts to /opt/ffmpeg-master-latest-linux64-gpl/.
    # The /usr/local/bin/ffmpeg-master symlink covers binary lookup; LD_LIBRARY_PATH
    # must point to the actual lib dir (BtbN GPL builds are mostly static but
    # some shared libs ship in lib/).
    ffmpeg_root = "/opt/ffmpeg-master-latest-linux64-gpl"
    env_sh = workspace / "env.sh"
    # CRITICAL: WORKSPACE/TAC_UPSTREAM_DIR must point at the WRITABLE /tmp/pact
    # copy, NOT the read-only /workspace/pact mount. Lane scripts source
    # this file then use $WORKSPACE for all path lookups — wrong value here
    # silently re-anchors them at the read-only mount. Round 5 caught this.
    # bin_dir is computed below; refer forward via workspace/_modal_bin
    env_sh.write_text(
        "# auto-generated by modal_train_lane.py\n"
        "export FFMPEG_BIN=/usr/local/bin/ffmpeg-master\n"
        f"export PATH={workspace}/_modal_bin:{ffmpeg_root}/bin:/root/.local/bin:$PATH\n"
        f"export LD_LIBRARY_PATH={ffmpeg_root}/lib:${{LD_LIBRARY_PATH:-}}\n"
        f"export PYTHONPATH={workspace}/src:{workspace}/upstream:{workspace}\n"
        f"export TAC_UPSTREAM_DIR={workspace}/upstream\n"
        f"export PYBIN={sys.executable}\n"
        f"export WORKSPACE={workspace}\n"
    )

    # Stub probe_nvdec.sh so Stage 0 of every lane passes. Modal containers
    # don't reliably expose libnvcuvid.so. We work around this by:
    #   1. Stubbing the probe to always pass (training itself doesn't need NVDEC)
    #   2. Forcing auth_eval to --device cpu via AUTH_EVAL_DEVICE env (lane
    #      scripts honor this — see remote_lane_*.sh Stage 5). CPU device
    #      selects AVVideoDataset (PyAV) per upstream/evaluate.py:39-42 —
    #      the ONLY way to get a diagnostic score on Modal without NVDEC.
    # Round 4 caught the AVVideoDataset-auto-fallback claim was wrong;
    # the fallback ONLY happens when device.type != cuda. So we have to
    # explicitly tell auth_eval to use cpu.
    stub_probe = workspace / "scripts" / "probe_nvdec.sh"
    stub_probe.write_text(
        "#!/bin/bash\n"
        "# Modal-runtime stub. Lane training doesn't need NVDEC; auth_eval\n"
        "# uses --device cpu (AVVideoDataset/PyAV), so scores are advisory only.\n"
        "echo '[probe_nvdec] Modal runtime — NVDEC not required (auth_eval uses cpu device)'\n"
        "exit 0\n"
    )
    stub_probe.chmod(0o755)

    # Many lane scripts call bare `python3` (vs `$PYBIN`). On Modal debian_slim
    # the system python3 may not be the same interpreter as sys.executable
    # (with torch/tac installed). Symlink python3 → sys.executable in a
    # PATH-priority dir so bare `python3` resolves to the working interpreter.
    # Round 7 catch.
    bin_dir = workspace / "_modal_bin"
    bin_dir.mkdir(exist_ok=True)
    py3_link = bin_dir / "python3"
    if py3_link.exists() or py3_link.is_symlink():
        py3_link.unlink()
    py3_link.symlink_to(sys.executable)
    py_link = bin_dir / "python"
    if py_link.exists() or py_link.is_symlink():
        py_link.unlink()
    py_link.symlink_to(sys.executable)

    # NOTE: git is installed via apt_install (line 43). Lane scripts that
    # run `git rev-parse HEAD` get real git output. Round 13 caught that
    # the previous "_git_stub.sh" was written but never placed on PATH —
    # removed since lane scripts already have `|| echo no-git` fallbacks.

    # Sentinel that this is Modal (skip Vast.ai-specific paths)
    (workspace / ".MODAL_RUNTIME").write_text("1\n")

    # Heartbeat tracking
    log_dir = workspace / f"results/{label}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build env. PATH/LD_LIBRARY_PATH point to the actual extracted ffmpeg dir
    # (not a phantom /opt/ffmpeg-master). PYBIN is propagated to bash + child
    # python invocations so probe_nvdec.sh + lane scripts inherit it.
    # AUTH_EVAL_DEVICE=cpu forces lane Stage 5 contest_auth_eval to use
    # AVVideoDataset (PyAV) instead of DaliVideoDataset (which needs NVDEC).
    # Mark the path non-promotable in the child environment so harvested
    # metadata cannot be mistaken for exact CUDA evidence.
    env = {
        **os.environ,
        "WORKSPACE": str(workspace),
        "PYBIN": sys.executable,
        "PYTHONPATH": f"{workspace}/src:{workspace}/upstream:{workspace}",
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        # _modal_bin first so `python3` and `python` resolve to sys.executable
        # (the venv with torch/tac), not /usr/bin/python3 (system, no deps).
        "PATH": f"{bin_dir}:{ffmpeg_root}/bin:/root/.local/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{ffmpeg_root}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "TAC_UPSTREAM_DIR": str(workspace / "upstream"),
        "MODAL_RUNTIME": "1",
        "AUTH_EVAL_DEVICE": "cpu",
        "MODAL_AUTH_EVAL_ADVISORY_ONLY": "1",
        "SCORE_CLAIM": "false",
        "PROMOTION_ELIGIBLE": "false",
        # Lanes test for AUTO_DESTROY_VAST + VAST_INSTANCE_ID; on Modal these
        # are no-ops since Modal manages instance lifecycle.
        "AUTO_DESTROY_VAST": "0",
    }
    env.update(env_overrides)

    # Run the lane script
    lane_path = workspace / lane_script
    if not lane_path.exists():
        return {
            "returncode": 2,
            "error": f"lane script not found: {lane_script}",
            "artifacts": {},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    print(f"[modal-train-lane] starting lane: {lane_script} (label={label})")
    t0 = time.monotonic()

    log_path = workspace / f"modal_lane_{label}.log"
    timed_out = False
    with log_path.open("w") as logf:
        try:
            proc = subprocess.run(
                ["bash", str(lane_path)],
                env=env, cwd=workspace,
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=max_seconds,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            # Hit per-lane timeout (set via --timeout-hours). Round 13: this was
            # previously dead-code (Modal @app.function timeout was the only
            # cap and was hardcoded at 14h). Now the user-supplied timeout
            # actually triggers and we still collect partial artifacts.
            timed_out = True
            rc = 124
            print(f"[modal-train-lane] TIMEOUT after {max_seconds}s — collecting partial artifacts")
    elapsed = time.monotonic() - t0
    print(f"[modal-train-lane] finished in {elapsed:.1f}s rc={rc} timed_out={timed_out}")

    # Collect output artifacts. Lane scripts write to varied locations:
    #   - $WORKSPACE/results/<label>/  (canonical, used by some)
    #   - $WORKSPACE/<lane_X>_results/  (used by lane_omega, lane_mae_v, etc.)
    #   - $WORKSPACE/lane_*_results/  (catch-all for *_results/ siblings)
    #   - $WORKSPACE/submissions/robust_current/  (archive output sometimes)
    # Scan the WHOLE workspace for these extensions to avoid silent loss.
    artifacts: dict[str, bytes] = {}
    skipped_large: list[tuple[str, int]] = []
    extensions = (".bin", ".zip", ".pt", ".mkv", ".json", ".log", ".safetensors")
    # Top-level dirs to scan (avoid scanning src/ scripts/ etc.)
    scan_roots = [
        workspace / "results",
    ]
    # Also any */_results/ siblings of workspace root
    for child in workspace.iterdir():
        if child.is_dir() and child.name.endswith("_results"):
            scan_roots.append(child)
    # Plus archives written into submissions/robust_current
    scan_roots.append(workspace / "submissions" / "robust_current")
    # Plus the modal log we wrote at workspace root
    if log_path.exists():
        scan_roots.append(log_path)

    for root in scan_roots:
        if not root.exists():
            continue
        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob("*") if p.is_file()]
        for fp in files:
            if not fp.is_file() or not fp.suffix.lower() in extensions:
                continue
            try:
                rel = fp.relative_to(workspace)
            except ValueError:
                rel = Path(fp.name)
            rel_str = str(rel)
            if rel_str in artifacts:
                continue
            try:
                size = fp.stat().st_size
                # 500MB threshold — covers final .bin (~300KB) AND mid-training
                # .pt checkpoints (50-200MB) AND large .mkv masks. Anything
                # bigger is almost certainly intermediate state we don't need.
                if size > 500 * 1024 * 1024:
                    skipped_large.append((rel_str, size))
                    print(f"[modal-train-lane] SKIP large {rel_str} ({size/1e6:.1f}MB)")
                    continue
                artifacts[rel_str] = fp.read_bytes()
            except (FileNotFoundError, PermissionError) as e:
                print(f"[modal-train-lane] SKIP unreadable {fp}: {e}")
                continue

    # Tail log files for return value
    stdout_tail = ""
    if log_path.exists():
        try:
            stdout_tail = log_path.read_text(errors="ignore")[-4000:]
        except Exception:
            pass

    return {
        "returncode": rc,
        "timed_out": timed_out,
        "artifacts": artifacts,
        "stdout_tail": stdout_tail,
        "elapsed_seconds": elapsed,
        "skipped_large_artifacts": skipped_large,
    }


@app.function(
    image=training_image,
    gpu="T4",
    timeout=14 * 3600,  # 14h max — covers MAE-V (estimate)
)
def run_lane_training_t4(lane_script: str, label: str, env_overrides: dict,
                          max_seconds: int = 14 * 3600) -> dict:
    return _run_lane_inner(lane_script, label, env_overrides, max_seconds=max_seconds)


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=14 * 3600,
)
def run_lane_training_a10g(lane_script: str, label: str, env_overrides: dict,
                            max_seconds: int = 14 * 3600) -> dict:
    return _run_lane_inner(lane_script, label, env_overrides, max_seconds=max_seconds)


@app.local_entrypoint()
def main(
    lane_script: str,
    label: str,
    gpu: str = "T4",
    timeout_hours: float = 10.0,
    env_overrides: str = "",
):
    """Dispatch a lane training run on Modal.

    Args:
        lane_script: relative path like 'scripts/remote_lane_omega_hessian_qat.sh'
        label: short label used for output dir naming
        gpu: 'T4' (default, $0.59/hr) or 'A10G' ($1.10/hr, ~2x faster)
        timeout_hours: max runtime (Modal hard kills at this)
        env_overrides: 'KEY1=val1,KEY2=val2' optional env to pass to lane
    """
    import json
    import os
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    if not (repo_root / lane_script).exists():
        print(f"FATAL: lane script not found: {lane_script}", file=sys.stderr)
        sys.exit(2)

    overrides = {}
    if env_overrides:
        for kv in env_overrides.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                overrides[k.strip()] = v.strip()

    if gpu == "T4":
        fn = run_lane_training_t4
    elif gpu in ("A10G", "A10g"):
        fn = run_lane_training_a10g
    else:
        print(f"FATAL: unsupported gpu '{gpu}'. Use T4 or A10G.", file=sys.stderr)
        sys.exit(2)

    print(f"=== modal_train_lane: {lane_script} → {label} on {gpu} ===")
    max_seconds = int(timeout_hours * 3600)
    if max_seconds < 60:
        max_seconds = 60
    if max_seconds > 14 * 3600:
        max_seconds = 14 * 3600
    print(f"  per-lane timeout: {max_seconds}s ({timeout_hours:.1f}h)")

    # CRITICAL: use .spawn() not .remote() for detached runs.
    # `.remote()` is cancelled when the local CLI disconnects, even with
    # --detach (Modal's warning: ".remote() calls in detached apps may be
    # canceled when the local caller disconnects. Use .spawn() for detached
    # or background work."). Tonight's first 3 dispatches all got killed at
    # 4-6s by this exact issue.
    #
    # .spawn() returns a FunctionCall handle. We save it so the recovery script
    # can poll later through Modal's Python API. Modal 1.4 no longer exposes a
    # direct FunctionCall result CLI for this path.
    fn_call = fn.spawn(lane_script, label, overrides, max_seconds)
    call_id = fn_call.object_id

    print(f"\n✓ DISPATCHED via .spawn() — call_id={call_id}")
    print(
        "  Poll/recover:   "
        f".venv/bin/python experiments/modal_recover_lane.py --label {label}"
    )
    print("  Stream logs:    .venv/bin/modal app logs <app-id> (see modal app list)")
    print(
        "  Direct recover: "
        f".venv/bin/python experiments/modal_recover_lane.py --call-id {call_id}"
    )
    print(f"\n  Local entrypoint exiting; remote training continues for up to {timeout_hours:.0f}h.")

    # Save call_id to a sentinel file so a later script can recover artifacts.
    sentinel_dir = repo_root / "experiments" / "results" / f"lane_{label}_modal"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    (sentinel_dir / "modal_call_id.txt").write_text(call_id + "\n")
    (sentinel_dir / "modal_metadata.json").write_text(json.dumps({
        "lane_script": lane_script,
        "label": label,
        "gpu": gpu,
        "max_seconds": max_seconds,
        "call_id": call_id,
        "auth_eval_device": "cpu",
        "auth_eval_advisory_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatched_at": __import__("datetime").datetime.now().isoformat(),
    }, indent=2))
    print(f"  call_id saved:  {sentinel_dir}/modal_call_id.txt")
    print(f"\n  Use experiments/modal_recover_lane.py to fetch artifacts when complete.")
