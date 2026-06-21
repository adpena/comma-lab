# SPDX-License-Identifier: MIT
"""Modal CUDA resume path for the decisive PR95 8-stage curriculum run.

Reuses the canonical ``training_image`` from ``experiments/modal_train_lane.py``
(torch==2.5.1 cu121 GPU-safe pin + canonical structural mounts of
``src/`` / ``upstream/`` / ``experiments/`` (results ignored) / ``tools/`` /
``submissions/``) and invokes ``experiments/launch_split_by_head_basin.py``
DIRECTLY (not via a ``scripts/remote_lane_*.sh`` shell lane).

Why a dedicated wrapper instead of ``modal_train_lane.py``:
  - ``modal_train_lane.py`` is shell-lane oriented and HARDCODES
    ``AUTH_EVAL_DEVICE=cpu`` + advisory-only + REFUSES CUDA auth-eval. This run
    needs CUDA-AUTHORITY eval (a valid contest axis) via the launcher's own
    in-process ``async_eval`` byte-close path (pure torch + brotli; NO DALI /
    ffmpeg / inflate subprocess needed because the GT targets are pre-supplied
    from the n600 cache).
  - The decisive run is a plain python entry that RESUMES from a checkpoint in
    ``--out-dir``. The launcher's resume guards (n_pairs / base_channels /
    latent_dim / taper / muon / warmup) cover correctness; device is purely a
    runtime placement, so an MPS-saved checkpoint resumes on CUDA via
    ``load_checkpoint(map_location='cuda')`` (the checkpoint tensors are CPU
    snapshots of the EMA shadow → cross-device safe).

INPUTS shipped via the ``yousfi-r3-pr95-resume`` Modal Volume (mounted /vol):
  - ``cache/gt_targets_n600.pt``  (900MB GT targets cache; device-agnostic CPU blob)
  - ``out/torch_vehicle_checkpoint_state.pt``     (1.6MB resume checkpoint)
  - ``out/torch_vehicle_checkpoint_manifest.json``

The frozen scorers (``upstream/models/{segnet,posenet}.safetensors``) and the
video (``upstream/videos/0.mkv``) ride in the canonical image mount (upstream/
is mounted with NO ignore).

USAGE — FREE CPU smoke (proves wiring + cross-device checkpoint load, no GPU):
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run \\
        experiments/modal_yousfi_r3_pr95_resume.py::cpu_smoke

USAGE — GPU TIMING smoke (T4, resume + ~150 epochs, measures ep/h):
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run \\
        experiments/modal_yousfi_r3_pr95_resume.py::t4_smoke

USAGE — FULL run (held by the MAIN AGENT; see the printed command in the report):
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
        experiments/modal_yousfi_r3_pr95_resume.py::run_full --gpu <GPU> \\
        --total-epoch-budget 29650 --timeout-hours <H>

NON-PROMOTABLE NOTE: CUDA-authority eval rows from this run ARE a valid contest
axis, but a SCORE/FRONTIER claim still requires the byte-closed archive run
through ``upstream/evaluate.py`` per CLAUDE.md. This wrapper reports
component d_seg / d_pose / rate / score for the CONTINUITY check only.
"""
from __future__ import annotations

import modal

# Reuse the canonical GPU-safe image (torch==2.5.1 cu121 + structural mounts).
# Importing from modal_train_lane gives us the EXACT same image + mount manifest
# the rest of the fleet uses — no duplicated torch pin, no cu13 trap.
from experiments.modal_train_lane import training_image

app = modal.App("yousfi-r3-pr95-resume")
RESUME_VOL = "yousfi-r3-pr95-resume"
resume_vol = modal.Volume.from_name(RESUME_VOL, create_if_missing=True)

REMOTE_PYTHONPATH = "/workspace/pact/src:/workspace/pact/upstream:/workspace/pact"

# The local run's EXACT flags (device swapped per-function). Everything else is
# byte-identical to the live MPS basin.
BASE_LAUNCHER_ARGS = [
    "--no-split-by-head",
    "--base-channels", "20",
    "--latent-dim", "28",
    "--n-pairs", "600",
    "--async-eval",
    "--eval-every", "25",
    "--checkpoint-every-epochs", "25",
    "--no-muon-lr-floor-fix",
    "--seg-margin-hinge",
    "--stage-lr-warmup-frac", "0.03",
    "--taper-channels", "16,16,17,19,19,14,10",
]


def _prepare_workspace_and_inputs(device: str) -> tuple[str, str, str]:
    """Container-side: copy mounted source to a writable workspace, stage the
    cache + resume checkpoint from the volume, set up env. Returns
    (workspace, out_dir, cache_dir)."""
    import os
    import shutil
    import sys
    from pathlib import Path

    image_workspace = Path("/workspace/pact")
    workspace = Path("/tmp/pact")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    print(f"[yousfi-r3-resume] copying mounted source -> {workspace}", flush=True)
    for sub in ("src", "upstream", "experiments", "tools", "submissions"):
        src_path = image_workspace / sub
        if src_path.exists():
            shutil.copytree(src_path, workspace / sub, symlinks=True)
    pp = image_workspace / "pyproject.toml"
    if pp.exists():
        shutil.copy2(pp, workspace / "pyproject.toml")

    os.chdir(workspace)
    sys.path.insert(0, str(workspace / "src"))
    sys.path.insert(0, str(workspace / "upstream"))
    os.environ["PYTHONPATH"] = f"{workspace}/src:{workspace}/upstream:{workspace}"
    os.environ["TAC_UPSTREAM_DIR"] = str(workspace / "upstream")
    # Deterministic CUDA matmul workspace + DALI NVML guard (defensive; this run
    # does not use DALI, but the env is cheap and matches fleet discipline).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("DALI_DISABLE_NVML", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Stage the resume checkpoint + manifest into the writable out-dir.
    out_dir = workspace / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    vol_out = Path("/vol/out")
    for name in (
        "torch_vehicle_checkpoint_state.pt",
        "torch_vehicle_checkpoint_manifest.json",
    ):
        src = vol_out / name
        if not src.is_file():
            raise RuntimeError(
                f"resume checkpoint file missing on volume: {src} "
                "(upload via `modal volume put yousfi-r3-pr95-resume ...`)"
            )
        shutil.copy2(src, out_dir / name)
    print(f"[yousfi-r3-resume] staged resume checkpoint -> {out_dir}", flush=True)

    # Stage the vendored PR95 (hnerv_muon) src into the expected workspace path.
    # It lives under experiments/results/** locally (Modal-IGNORED subtree), so
    # it does NOT ride in the image mount — ship it via the volume. The driver's
    # import_vendored_bundle() + the launcher's import_vendored() resolve it at
    # ``experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/
    # source/submissions/hnerv_muon/src`` (see tac.torch_vehicle.vendored_imports).
    vendored_dst = (
        workspace
        / "experiments/results/public_pr_intake_full"
        / "public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src"
    )
    vendored_src = Path("/vol/vendored_hnerv_muon_src")
    if not vendored_src.is_dir():
        raise RuntimeError(
            f"vendored PR95 src missing on volume: {vendored_src} "
            "(upload via `modal volume put yousfi-r3-pr95-resume <staged_src> "
            "vendored_hnerv_muon_src`)"
        )
    vendored_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(vendored_src, vendored_dst, dirs_exist_ok=True)
    print(f"[yousfi-r3-resume] staged vendored PR95 src -> {vendored_dst}", flush=True)

    # Point --targets-cache at the volume cache dir directly (read-only is fine;
    # the launcher only reads gt_targets_n600.pt with map_location='cpu').
    cache_dir = Path("/vol/cache")
    cache_file = cache_dir / "gt_targets_n600.pt"
    if not cache_file.is_file():
        raise RuntimeError(
            f"GT targets cache missing on volume: {cache_file} "
            "(upload via `modal volume put yousfi-r3-pr95-resume "
            "experiments/results/capstone_gt_targets_cache/gt_targets_n600.pt "
            "cache/gt_targets_n600.pt`)"
        )
    print(
        f"[yousfi-r3-resume] cache present: {cache_file} "
        f"({cache_file.stat().st_size/1e6:.0f}MB)",
        flush=True,
    )
    return str(workspace), str(out_dir), str(cache_dir)


def _assert_cuda_or_die() -> str:
    """Fail LOUD if CUDA is unavailable (the cu13/cu124 silent-CPU-fallback
    money-burning bug class). Returns the GPU name."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FATAL: torch.cuda.is_available() == False on a GPU function. "
            "The torch wheel does not match the worker driver (cu13/cu124 "
            "silent-CPU-fallback trap). Refusing to run — every score would "
            "be advisory CPU, not the requested CUDA axis."
        )
    name = torch.cuda.get_device_name(0)
    print(f"[yousfi-r3-resume] CUDA OK: {name} (torch {torch.__version__})", flush=True)
    return name


def _run_launcher_and_collect(
    *,
    device: str,
    out_dir: str,
    cache_dir: str,
    total_epoch_budget: int | None,
    extra_args: list[str] | None = None,
) -> dict:
    """Invoke the launcher main() in-process, time it, and harvest the eval
    rows from the run's summary/trajectory."""
    import json
    import time
    from pathlib import Path

    from experiments.launch_split_by_head_basin import main as launcher_main

    argv = list(BASE_LAUNCHER_ARGS)
    argv += ["--device", device, "--train-device", device]
    argv += ["--targets-cache", cache_dir, "--out-dir", out_dir]
    # Point at the staged contest video explicitly so the launcher skips the
    # vendored get_default_video_path() (which needs a comma_video_compression_
    # challenge/ tree we do not ship). The GT cache is present so the video is
    # never actually decoded — only held as a path by RealScorerContext.
    # out_dir is ``<workspace>/out`` → workspace == out_dir.parent.
    workspace_root = Path(out_dir).parent
    argv += ["--video-path", str(workspace_root / "upstream/videos/0.mkv")]
    if total_epoch_budget is not None:
        argv += ["--total-epoch-budget", str(total_epoch_budget)]
    if extra_args:
        argv += extra_args

    print(f"[yousfi-r3-resume] launcher argv: {argv}", flush=True)
    t0 = time.monotonic()
    rc = launcher_main(argv)
    elapsed = time.monotonic() - t0

    out = Path(out_dir)
    summary = {}
    summary_path = out / "torch_vehicle_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())

    # Harvest the eval rows from the trajectory jsonl (each eval is a row with
    # d_seg/d_pose/rate/score + global_epoch).
    eval_rows: list[dict] = []
    traj_path = out / "torch_vehicle_trajectory.jsonl"
    if traj_path.is_file():
        for line in traj_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            # Only actual eval rows (evaluated=True => non-null d_seg/score).
            if (
                isinstance(row, dict)
                and row.get("evaluated") is True
                and row.get("d_seg") is not None
                and row.get("score") is not None
            ):
                eval_rows.append({
                    "global_epoch": row.get("global_epoch"),
                    "stage_index": row.get("stage_index"),
                    "d_seg": row.get("d_seg"),
                    "d_pose": row.get("d_pose"),
                    "rate": row.get("rate"),
                    "score": row.get("score"),
                    "archive_bytes": row.get("archive_bytes"),
                })

    manifest = {}
    man_path = out / "torch_vehicle_checkpoint_manifest.json"
    if man_path.is_file():
        manifest = json.loads(man_path.read_text())

    resume_vol.commit()
    return {
        "returncode": rc,
        "elapsed_seconds": elapsed,
        "device": device,
        "summary_last_eval": summary.get("last_eval"),
        "final_manifest": manifest,
        "eval_rows_tail": eval_rows[-12:],
        "n_eval_rows": len(eval_rows),
    }


@app.function(
    image=training_image,
    timeout=30 * 60,
    volumes={"/vol": resume_vol},
)
def cpu_smoke() -> dict:
    """FREE wiring + cross-device checkpoint-load proof: load scorers + the
    MPS-saved checkpoint (map_location cpu) + the n600 cache, run ~1 eval-worth
    of epochs on CPU. Validates the resume + eval path WITHOUT a GPU meter."""
    workspace, out_dir, cache_dir = _prepare_workspace_and_inputs("cpu")
    # total_epoch_budget=4901 -> resume at global ep 4875, run ~26 epochs so the
    # eval-every-25 fires at least one CPU-authority eval. Proves wiring cheaply.
    return _run_launcher_and_collect(
        device="cpu",
        out_dir=out_dir,
        cache_dir=cache_dir,
        total_epoch_budget=4901,
    )


def _gpu_smoke_body(device: str = "cuda") -> dict:
    gpu_name = _assert_cuda_or_die()
    workspace, out_dir, cache_dir = _prepare_workspace_and_inputs(device)
    # total_epoch_budget=5025 -> resume at global ep 4875, run ~150 epochs with
    # eval-every-25 => ~6 CUDA-authority evals. Measures ep/h + continuity.
    result = _run_launcher_and_collect(
        device=device,
        out_dir=out_dir,
        cache_dir=cache_dir,
        total_epoch_budget=5025,
    )
    result["gpu_name"] = gpu_name
    return result


@app.function(
    image=training_image,
    gpu="T4",
    timeout=30 * 60,
    volumes={"/vol": resume_vol},
)
def t4_smoke() -> dict:
    """GPU TIMING smoke on T4: resume + ~150 epochs, measure ep/h + continuity."""
    return _gpu_smoke_body("cuda")


def _full_body(total_epoch_budget: int, device: str = "cuda") -> dict:
    gpu_name = _assert_cuda_or_die()
    workspace, out_dir, cache_dir = _prepare_workspace_and_inputs(device)
    return {
        **_run_launcher_and_collect(
            device=device,
            out_dir=out_dir,
            cache_dir=cache_dir,
            total_epoch_budget=total_epoch_budget,
        ),
        "gpu_name": gpu_name,
    }


@app.function(
    image=training_image,
    gpu="T4",
    timeout=14 * 3600,
    volumes={"/vol": resume_vol},
)
def run_full_t4(total_epoch_budget: int = 29650) -> dict:
    """FULL run on T4 (held by the main agent). Resumes from the volume
    checkpoint and trains to ``total_epoch_budget`` global epochs."""
    return _full_body(total_epoch_budget, "cuda")


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=14 * 3600,
    volumes={"/vol": resume_vol},
)
def run_full_a10g(total_epoch_budget: int = 29650) -> dict:
    """FULL run on A10G (held by the main agent)."""
    return _full_body(total_epoch_budget, "cuda")


@app.function(
    image=training_image,
    gpu="A100",
    timeout=14 * 3600,
    volumes={"/vol": resume_vol},
)
def run_full_a100(total_epoch_budget: int = 29650) -> dict:
    """FULL run on A100 (held by the main agent)."""
    return _full_body(total_epoch_budget, "cuda")


@app.local_entrypoint()
def cpu_smoke_entry():
    import json

    print(json.dumps(cpu_smoke.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def t4_smoke_entry():
    import json

    print(json.dumps(t4_smoke.remote(), indent=2, sort_keys=True))
