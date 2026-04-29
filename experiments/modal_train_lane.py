"""Run a `scripts/remote_lane_*.sh` on Modal T4 / A10G — reliable training surface.

Why: 2026-04-29 night session showed Vast.ai 4090 NVDEC bad-host rate ≈ 85%.
~$5 burned across 5 dispatch rounds for 0 trained lanes. Modal pipeline
verified equivalent (Lane G v3 = 1.04 [Modal-T4-CUDA] vs 1.05 [Vast.ai-CUDA]
in commit fc26b800). Time to pivot training to Modal too.

Pattern mirrors `experiments/modal_auth_eval.py` (commit 11d56896).

Usage:
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run experiments/modal_train_lane.py \
        --lane-script scripts/remote_lane_omega_hessian_qat.sh \
        --label lane_omega_hessian \
        --gpu T4 \
        --timeout-hours 10

Output: artifacts saved to `experiments/results/lane_<label>_modal/`. Score
auto-extracted from `contest_auth_eval.json` if the lane completes auth eval.

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
    .add_local_dir("submissions/robust_current", remote_path="/workspace/pact/submissions/robust_current")
    .add_local_dir("upstream", remote_path="/workspace/pact/upstream")
    .add_local_dir("experiments", remote_path="/workspace/pact/experiments")
    .add_local_dir("tools", remote_path="/workspace/pact/tools")
    .add_local_file("pyproject.toml", remote_path="/workspace/pact/pyproject.toml")
)


def _run_lane_inner(lane_script: str, label: str, env_overrides: dict) -> dict:
    """Container-side execution. Imports MUST be local (Modal serialization)."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    workspace = Path("/workspace/pact")
    os.chdir(workspace)

    # Modal mounts add_local_dir as READ-ONLY. `pip install -e .` would fail
    # writing src/tac.egg-info/ → use sys.path injection (matches
    # modal_auth_eval.py's pattern). Lane scripts call `import tac` which
    # resolves via PYTHONPATH=/workspace/pact/src.
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
    env_sh.write_text(
        "# auto-generated by modal_train_lane.py\n"
        "export FFMPEG_BIN=/usr/local/bin/ffmpeg-master\n"
        f"export PATH={ffmpeg_root}/bin:/root/.local/bin:$PATH\n"
        f"export LD_LIBRARY_PATH={ffmpeg_root}/lib:${{LD_LIBRARY_PATH:-}}\n"
        f"export PYTHONPATH={workspace}/src:{workspace}/upstream:{workspace}\n"
        "export TAC_UPSTREAM_DIR=/workspace/pact/upstream\n"
        f"export PYBIN={sys.executable}\n"
        "export WORKSPACE=/workspace/pact\n"
        # Modal already provides CUDA — no probe needed but lane scripts
        # will run probe_nvdec.sh anyway. Stub it to always pass.
    )

    # Make a stub for git (some scripts run git rev-parse HEAD)
    git_stub = workspace / "_git_stub.sh"
    git_stub.write_text("#!/bin/bash\necho modal-no-git\n")
    git_stub.chmod(0o755)

    # Sentinel that this is Modal (skip Vast.ai-specific paths)
    (workspace / ".MODAL_RUNTIME").write_text("1\n")

    # Heartbeat tracking
    log_dir = workspace / f"results/{label}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build env. PATH/LD_LIBRARY_PATH point to the actual extracted ffmpeg dir
    # (not a phantom /opt/ffmpeg-master). PYBIN is propagated to bash + child
    # python invocations so probe_nvdec.sh + lane scripts inherit it.
    env = {
        **os.environ,
        "WORKSPACE": str(workspace),
        "PYBIN": sys.executable,
        "PYTHONPATH": f"{workspace}/src:{workspace}/upstream:{workspace}",
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        "PATH": f"{ffmpeg_root}/bin:/root/.local/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{ffmpeg_root}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "TAC_UPSTREAM_DIR": str(workspace / "upstream"),
        "MODAL_RUNTIME": "1",
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
    with log_path.open("w") as logf:
        proc = subprocess.run(
            ["bash", str(lane_path)],
            env=env, cwd=workspace,
            stdout=logf, stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - t0
    print(f"[modal-train-lane] finished in {elapsed:.1f}s rc={proc.returncode}")

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
        "returncode": proc.returncode,
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
def run_lane_training_t4(lane_script: str, label: str, env_overrides: dict) -> dict:
    return _run_lane_inner(lane_script, label, env_overrides)


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=14 * 3600,
)
def run_lane_training_a10g(lane_script: str, label: str, env_overrides: dict) -> dict:
    return _run_lane_inner(lane_script, label, env_overrides)


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
    result = fn.remote(lane_script, label, overrides)

    # Save artifacts locally
    out_dir = repo_root / "experiments" / "results" / f"lane_{label}_modal"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, data in result["artifacts"].items():
        full = out_dir / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
    print(f"Saved {len(result['artifacts'])} artifacts to {out_dir}")

    # Save metadata
    meta = {
        "lane_script": lane_script,
        "label": label,
        "gpu": gpu,
        "returncode": result["returncode"],
        "elapsed_seconds": result.get("elapsed_seconds"),
        "n_artifacts": len(result["artifacts"]),
        "stdout_tail": result.get("stdout_tail", "")[-2000:],
    }
    (out_dir / "modal_metadata.json").write_text(json.dumps(meta, indent=2))

    # Auto-extract score from any of the canonical sources:
    #   1. *.json files containing 'score' or 'final_score' field
    #   2. *.log files with `RESULT_JSON: {...}` line (contest_auth_eval.py emits this)
    score_found = False
    import re as _re
    for path, data_bytes in result["artifacts"].items():
        if score_found:
            break
        # Try JSON file
        if path.endswith(".json"):
            try:
                data = json.loads(data_bytes.decode())
                if isinstance(data, dict):
                    score = data.get("score") or data.get("final_score")
                    if score is not None:
                        print(f"\n=== AUTH SCORE: {score} [Modal-{gpu}-CUDA] ({label}) ===")
                        print(f"  source:  {path}")
                        print(f"  PoseNet: {data.get('pose') or data.get('pose_dist')}")
                        print(f"  SegNet:  {data.get('seg') or data.get('seg_dist')}")
                        print(f"  Rate:    {data.get('rate')}")
                        score_found = True
            except Exception:
                pass
        # Try RESULT_JSON: in log files
        elif path.endswith(".log"):
            try:
                text = data_bytes.decode(errors="ignore")
                m = _re.search(r"RESULT_JSON:\s*(\{[^\n]+\})", text)
                if m:
                    data = json.loads(m.group(1))
                    score = data.get("score") or data.get("final_score")
                    if score is not None:
                        print(f"\n=== AUTH SCORE: {score} [Modal-{gpu}-CUDA] ({label}) ===")
                        print(f"  source:  {path} (RESULT_JSON line)")
                        print(f"  PoseNet: {data.get('pose')}")
                        print(f"  SegNet:  {data.get('seg')}")
                        print(f"  Rate:    {data.get('rate')}")
                        score_found = True
            except Exception:
                pass
    if not score_found and result["returncode"] == 0:
        print(f"\n  WARNING: no auth score extracted from artifacts (lane succeeded but eval not detected).")
        print(f"  Inspect {out_dir}/ manually for the lane's auth_eval output.")

    if result["returncode"] != 0:
        print(f"\n✗ Lane FAILED with rc={result['returncode']}", file=sys.stderr)
        print(f"  stdout tail (last 2000B):", file=sys.stderr)
        print(result.get("stdout_tail", "")[-2000:], file=sys.stderr)
        sys.exit(result["returncode"])

    print(f"\n✓ Lane SUCCESS")
