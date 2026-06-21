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

    # Stage the vendored PR95 (hnerv_muon) src into the EXACT path
    # tac.torch_vehicle.vendored_imports resolves at runtime. That module pins
    # ``VENDORED_SRC = _REPO_ROOT / experiments/results/.../hnerv_muon/src`` where
    # ``_REPO_ROOT`` is fixed at import time to wherever ``tac`` actually loaded
    # from (the read-only /workspace/pact image mount, NOT this /tmp/pact copy,
    # because ``tac`` is imported at module top via the modal_train_lane import).
    # Local results/** is Modal-IGNORED so the clone does not ride in the mount —
    # ship it via the volume + copy it to the resolved path. If that path is
    # read-only (image mount), fall back to forcing tac onto the writable copy
    # and re-resolving.
    vendored_src = Path("/vol/vendored_hnerv_muon_src")
    if not vendored_src.is_dir():
        raise RuntimeError(
            f"vendored PR95 src missing on volume: {vendored_src} "
            "(upload via `modal volume put yousfi-r3-pr95-resume <staged_src> "
            "vendored_hnerv_muon_src`)"
        )
    import tac.torch_vehicle.vendored_imports as _vi

    vendored_dst = _vi.VENDORED_SRC
    try:
        vendored_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vendored_src, vendored_dst, dirs_exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"could not stage vendored PR95 src to resolved path {vendored_dst} "
            f"(_REPO_ROOT={_vi._REPO_ROOT}); the image-mount root may be "
            f"read-only. Original error: {exc}"
        ) from exc
    print(f"[yousfi-r3-resume] staged vendored PR95 src -> {vendored_dst}", flush=True)

    # The vendored data.py needs the challenge root (frame_utils.py + modules.py).
    # ``upstream/`` IS that root and rides in the image mount; set it explicitly
    # so resolve_challenge_root() does not depend on _REPO_ROOT placement.
    challenge_root = _vi._REPO_ROOT / "upstream"
    if (challenge_root / "frame_utils.py").exists():
        os.environ["COMMA_CHALLENGE_ROOT"] = str(challenge_root)
        print(f"[yousfi-r3-resume] COMMA_CHALLENGE_ROOT={challenge_root}", flush=True)

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


def _build_driver(device: str, out_dir: str, cache_dir: str, eval_every: int = 25):
    """Build the TorchVehicleDriver EXACTLY as launch_split_by_head_basin.main()
    does (same cfg + RealScorerContext + margin_hinge curriculum), but with
    ``total_epoch_budget=None`` (the UNSCALED faithful curriculum — identical to
    the live MPS run). ``eval_every`` overrides the eval cadence; pass a huge
    value to DISABLE evals for pure-training timing. Returns (driver, out_path)."""
    import dataclasses
    from pathlib import Path

    from tac.torch_vehicle.curriculum import build_curriculum
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig,
        TorchVehicleDriver,
        import_vendored_bundle,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext

    out_path = Path(out_dir)
    workspace_root = out_path.parent
    video_path = workspace_root / "upstream/videos/0.mkv"

    cfg = TorchVehicleConfig(
        base_channels=20,
        latent_dim=28,
        out_dir=out_path,
        checkpoint_every_epochs=25,
        total_epoch_budget=None,  # UNSCALED faithful curriculum (matches live run)
        ema_decay=0.999,
        eval_every=eval_every,
        device=device,
        train_device=device,
        split_by_head=False,       # --no-split-by-head
        async_eval=True,           # --async-eval
        muon_lr_floor_fix=False,   # --no-muon-lr-floor-fix
        stage_lr_warmup_frac=0.03,
        stage_lr_warmup_start_ratio=0.1,
        taper_channels=[16, 16, 17, 19, 19, 14, 10],
        seed=0,
    )
    scorer = RealScorerContext(
        video_path,
        device=device,
        train_device=device,
        split_by_head=False,
        max_pairs=600,
        targets_cache=cache_dir,
    )
    # --seg-margin-hinge: replace every stage's seg surrogate with margin_hinge.
    specs = build_curriculum(
        total_epoch_budget=None, ema_decay=0.999, eval_every=eval_every
    )
    curriculum = [dataclasses.replace(s, seg_surrogate="margin_hinge") for s in specs]
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    return driver, out_path


def _harvest(out_path, *, device: str, elapsed: float, rc: int) -> dict:
    """Read the run's summary + trajectory eval rows for the report."""
    import json

    summary = {}
    summary_path = out_path / "torch_vehicle_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())

    eval_rows: list[dict] = []
    traj_path = out_path / "torch_vehicle_trajectory.jsonl"
    if traj_path.is_file():
        for line in traj_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
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
    man_path = out_path / "torch_vehicle_checkpoint_manifest.json"
    if man_path.is_file():
        manifest = json.loads(man_path.read_text())

    resume_vol.commit()
    return {
        "returncode": rc,
        "elapsed_seconds": elapsed,
        "device": device,
        "summary_last_eval": summary.get("last_eval"),
        "final_manifest": manifest,
        "eval_rows_tail": eval_rows[-14:],
        "n_eval_rows": len(eval_rows),
    }


def _smoke_run_and_collect(
    *, device: str, out_dir: str, cache_dir: str, stop_after_global_epoch: int
) -> dict:
    """Resume the UNSCALED curriculum, run until ``stop_after_global_epoch``
    (raises _SimulatedDeath AFTER a checkpoint lands — the clean smoke stop),
    JOIN the in-flight async eval so the eval rows land, then harvest. This does
    NOT rescale the curriculum, so the cross-device resume mapping is faithful."""
    import time

    from tac.torch_vehicle.driver import _SimulatedDeath

    driver, out_path = _build_driver(device, out_dir, cache_dir)
    driver._stop_after_global_epoch = int(stop_after_global_epoch)

    print(
        f"[yousfi-r3-resume] SMOKE resume (unscaled) -> stop_after_global_epoch="
        f"{stop_after_global_epoch}",
        flush=True,
    )
    t0 = time.monotonic()
    rc = 0
    try:
        driver.run()
    except _SimulatedDeath as sd:
        print(f"[yousfi-r3-resume] smoke stop at global_epoch={sd}", flush=True)
        # run() raises BEFORE its own _join_async_eval; join here so the final
        # async eval row(s) land in the trajectory before we harvest.
        driver._join_async_eval()
    elapsed = time.monotonic() - t0
    return _harvest(out_path, device=device, elapsed=elapsed, rc=rc)


def _full_run_and_collect(*, device: str, out_dir: str, cache_dir: str) -> dict:
    """Resume + run the FULL unscaled curriculum to completion (held by the
    main agent). No stop hook → run() does its own async-eval join + DONE marker."""
    import time

    driver, out_path = _build_driver(device, out_dir, cache_dir)
    print("[yousfi-r3-resume] FULL resume (unscaled) -> run to curriculum end", flush=True)
    t0 = time.monotonic()
    summary = driver.run()
    elapsed = time.monotonic() - t0
    result = _harvest(out_path, device=device, elapsed=elapsed, rc=0)
    result["final_summary"] = summary
    return result


@app.function(
    image=training_image,
    timeout=30 * 60,
    volumes={"/vol": resume_vol},
)
def cpu_smoke() -> dict:
    """FREE wiring + cross-device checkpoint-load proof: load scorers + the
    MPS-saved checkpoint (map_location cpu) + the n600 cache, resume the UNSCALED
    curriculum, run ~26 epochs so eval-every-25 fires >=1 CPU-authority eval, then
    stop cleanly. Validates the resume + eval path WITHOUT a GPU meter."""
    _workspace, out_dir, cache_dir = _prepare_workspace_and_inputs("cpu")
    # Resume is at global ep 4875; stop at 4900 => ~25 epochs + the ep-4900 eval.
    return _smoke_run_and_collect(
        device="cpu",
        out_dir=out_dir,
        cache_dir=cache_dir,
        stop_after_global_epoch=4900,
    )


def _gpu_smoke_body(device: str = "cuda") -> dict:
    gpu_name = _assert_cuda_or_die()
    _workspace, out_dir, cache_dir = _prepare_workspace_and_inputs(device)
    # Resume at global ep 4875; stop at 5025 => ~150 epochs with eval-every-25
    # => ~6 CUDA-authority evals (ep 4900/4925/.../5025). Measures ep/h + continuity.
    result = _smoke_run_and_collect(
        device=device,
        out_dir=out_dir,
        cache_dir=cache_dir,
        stop_after_global_epoch=5025,
    )
    result["gpu_name"] = gpu_name
    return result


def _timing_micro_body(device: str = "cuda", stop_after_global_epoch: int = 5050) -> dict:
    """FAST pure-training timing. TRUE resume global epoch is 4950 (stage_index=1,
    epoch_in_stage=1950, stage0=3000). Builds with evals DISABLED (eval_every huge)
    so NO slow 600-pair eval fires, runs to ``stop_after_global_epoch`` (default
    5050 => 100 pure-training epochs), and reports clean training s/ep from the
    trajectory ``wall_clock_s`` deltas. Returns in low minutes."""
    import json
    import time
    from pathlib import Path

    from tac.torch_vehicle.driver import _SimulatedDeath

    gpu_name = _assert_cuda_or_die()
    _workspace, out_dir, cache_dir = _prepare_workspace_and_inputs(device)
    # eval_every huge => evals never fire => pure training timing.
    driver, out_path = _build_driver(device, out_dir, cache_dir, eval_every=10_000_000)
    driver._stop_after_global_epoch = int(stop_after_global_epoch)
    print(
        f"[yousfi-r3-resume] TIMING-MICRO (no eval) -> stop_after_global_epoch="
        f"{stop_after_global_epoch}",
        flush=True,
    )
    t0 = time.monotonic()
    try:
        driver.run()
    except _SimulatedDeath as sd:
        print(f"[yousfi-r3-resume] timing-micro stop at global_epoch={sd}", flush=True)
    elapsed = time.monotonic() - t0

    # Per-epoch wall deltas from the trajectory (wall_clock_s is cumulative from
    # the writer's _t0). The first delta includes resume overhead; later deltas
    # are clean per-epoch training time.
    rows = []
    traj_path = out_path / "torch_vehicle_trajectory.jsonl"
    if traj_path.is_file():
        for line in traj_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict) and r.get("wall_clock_s") is not None:
                rows.append((r.get("global_epoch"), float(r["wall_clock_s"])))
    rows.sort(key=lambda x: (x[1]))
    # Keep only rows AT/after this process's resume (wall_clock_s increases from 0
    # for THIS process — older rows from prior processes are not present because
    # the writer's _t0 is per-process and the file is append-only; filter to the
    # contiguous increasing tail belonging to this run).
    per_epoch_deltas = []
    for i in range(1, len(rows)):
        ep, w = rows[i]
        ep_prev, w_prev = rows[i - 1]
        d = w - w_prev
        if 0 < d < 3600:  # sane per-epoch bound
            per_epoch_deltas.append({"global_epoch": ep, "delta_s": round(d, 3)})
    # Robust central estimate: median of the last N clean deltas (skip the first
    # which carries resume/warmup overhead).
    clean = [d["delta_s"] for d in per_epoch_deltas[1:]] or [
        d["delta_s"] for d in per_epoch_deltas
    ]
    s_per_ep = round(sorted(clean)[len(clean) // 2], 3) if clean else None

    # Bulletproof fallback: pure-training s/ep from the trajectory wall-clock span
    # divided by the epoch count run (excludes resume/load which happen before the
    # first trajectory row's wall_clock_s baseline). resume global = 4950.
    span_s_per_ep = None
    epochs_run = None
    if len(rows) >= 2:
        first_ep, first_w = rows[0]
        last_ep, last_w = rows[-1]
        try:
            epochs_run = int(last_ep) - int(first_ep)
            if epochs_run > 0:
                span_s_per_ep = round((last_w - first_w) / epochs_run, 3)
        except (TypeError, ValueError):
            pass
    best_s_per_ep = s_per_ep if s_per_ep is not None else span_s_per_ep
    resume_vol.commit()
    return {
        "device": device,
        "gpu_name": gpu_name,
        "elapsed_seconds": round(elapsed, 2),
        "span_train_s_per_ep": span_s_per_ep,
        "epochs_run": epochs_run,
        "n_trajectory_rows": len(rows),
        "best_train_s_per_ep": best_s_per_ep,
        "best_train_ep_per_h": round(3600.0 / best_s_per_ep, 1) if best_s_per_ep else None,
        "stop_after_global_epoch": stop_after_global_epoch,
        "median_train_s_per_ep": s_per_ep,
        "train_ep_per_h": round(3600.0 / s_per_ep, 1) if s_per_ep else None,
        "per_epoch_deltas_tail": per_epoch_deltas[-20:],
        "n_epoch_deltas": len(per_epoch_deltas),
    }


@app.function(
    image=training_image,
    gpu="T4",
    timeout=30 * 60,
    volumes={"/vol": resume_vol},
)
def t4_smoke() -> dict:
    """GPU TIMING smoke on T4: resume + ~150 epochs, measure ep/h + continuity."""
    return _gpu_smoke_body("cuda")


@app.function(
    image=training_image,
    gpu="T4",
    timeout=20 * 60,
    volumes={"/vol": resume_vol},
)
def t4_timing_micro(stop_after_global_epoch: int = 5050) -> dict:
    """FAST T4 pure-training timing (no eval): clean median training s/ep."""
    return _timing_micro_body("cuda", stop_after_global_epoch)


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=20 * 60,
    volumes={"/vol": resume_vol},
)
def a10g_timing_micro(stop_after_global_epoch: int = 5050) -> dict:
    """FAST A10G pure-training timing (no eval): clean median training s/ep."""
    return _timing_micro_body("cuda", stop_after_global_epoch)


def _full_body(device: str = "cuda", stop_after_global_epoch: int = 0) -> dict:
    """Resume the UNSCALED curriculum. stop_after_global_epoch <= 0 => run to the
    full curriculum end (global ep 29,650). Set it to 19650 to stop at the
    decisive end-of-stage-5 (C1a-L7) verdict point."""
    gpu_name = _assert_cuda_or_die()
    _workspace, out_dir, cache_dir = _prepare_workspace_and_inputs(device)
    if stop_after_global_epoch and stop_after_global_epoch > 0:
        result = _smoke_run_and_collect(
            device=device,
            out_dir=out_dir,
            cache_dir=cache_dir,
            stop_after_global_epoch=stop_after_global_epoch,
        )
    else:
        result = _full_run_and_collect(
            device=device, out_dir=out_dir, cache_dir=cache_dir
        )
    result["gpu_name"] = gpu_name
    return result


@app.function(
    image=training_image,
    gpu="T4",
    timeout=14 * 3600,
    volumes={"/vol": resume_vol},
)
def run_full_t4(stop_after_global_epoch: int = 0) -> dict:
    """FULL run on T4 (held by the main agent). Resumes from the volume checkpoint
    and trains the UNSCALED curriculum to the end (29,650) — or to
    stop_after_global_epoch (e.g. 19650 = end of stage 5, the decisive point)."""
    return _full_body("cuda", stop_after_global_epoch)


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=14 * 3600,
    volumes={"/vol": resume_vol},
)
def run_full_a10g(stop_after_global_epoch: int = 0) -> dict:
    """FULL run on A10G (held by the main agent)."""
    return _full_body("cuda", stop_after_global_epoch)


@app.function(
    image=training_image,
    gpu="A100",
    timeout=14 * 3600,
    volumes={"/vol": resume_vol},
)
def run_full_a100(stop_after_global_epoch: int = 0) -> dict:
    """FULL run on A100 (held by the main agent)."""
    return _full_body("cuda", stop_after_global_epoch)


@app.local_entrypoint()
def cpu_smoke_entry():
    import json

    print(json.dumps(cpu_smoke.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def t4_smoke_entry():
    import json

    print(json.dumps(t4_smoke.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def t4_timing_micro_entry():
    import json

    print(json.dumps(t4_timing_micro.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def a10g_timing_micro_entry():
    import json

    print(json.dumps(a10g_timing_micro.remote(), indent=2, sort_keys=True))
