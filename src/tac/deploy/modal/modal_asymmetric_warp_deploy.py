"""Deploy asymmetric warp renderer training + auth eval to Modal T4.

Council-approved configuration for the Fridrich constrained renderer
with asymmetric warp architecture. T4 chosen for iteration budget over
speed (council decision).

Wall-clock budget: 5.5h training + 0.5h safety margin = 6h Modal timeout.
Resume support: auto-detects existing checkpoint on results volume.
Periodic commits: every 300s to survive client disconnects.

The training script writes checkpoints to experiments/results/fridrich_renderer/
(hardcoded RESULTS_DIR). We symlink that path to /results/<tag>/ on the
persistent volume so checkpoints survive across runs.

Usage (training):
    .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py
    .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py --tag my_run_v2
    .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py --extra-args '--smoke'

Usage (auth eval):
    .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py::app.auth_eval_entry --tag my_run
    .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py::app.auth_eval_entry --tag my_run --checkpoint renderer_best.pt
"""
from __future__ import annotations

from pathlib import Path

import modal

# Provider-agnostic training config — single source of truth for all platforms
from tac.deploy.deploy_config import (
    ALL_VARIANTS,
    EXPERIMENT_SCRIPT,
    VARIANT_FLAGS,
    build_flags,
)

APP_NAME = "tac-asymmetric-warp"
RESULTS_VOL = "tac-asymmetric-results"

# Where the training script hardcodes its output
SCRIPT_RESULTS_DIR = "/root/experiments/results/fridrich_renderer"

# Provider-specific script path inside the Modal container
_MODAL_SCRIPT_PATH = f"/root/{EXPERIMENT_SCRIPT}"

# REPO_ROOT only needed locally for add_local_dir -- guard for container env
_script = Path(__file__).resolve()
try:
    REPO_ROOT = _script.parents[4]  # src/tac/deploy/modal -> repo root
except IndexError:
    REPO_ROOT = Path("/root")  # inside Modal container

app = modal.App(APP_NAME)

# Build image: PyTorch + scorer deps + tac source + experiments/
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "ffmpeg")
    .pip_install(
        "torch==2.6.*",
        "torchvision",
        "av",
        "numpy",
        "pydantic>=2.0",
        "safetensors",
        "timm",
        "einops",
        "segmentation-models-pytorch",
        "click",
        "nvidia-dali-cuda120",
    )
    # Clone upstream scorer repo (PoseNet/SegNet model definitions + GT video)
    .run_commands(
        "git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git /root/upstream",
        "cd /root/upstream && git lfs pull",
    )
    .env({
        "PYTHONPATH": "/root/src:/root/upstream",
        "PYTHONUNBUFFERED": "1",
        "TAC_UPSTREAM_DIR": "/root/upstream",
        "TAC_MODELS_DIR": "/root/upstream/models",
        "DALI_DISABLE_NVML": "1",  # Modal containers lack NVML access; DALI works without it
    })
    # add_local_dir must be LAST -- Modal mounts these at startup, not during build
    .add_local_dir(str(REPO_ROOT / "src" / "tac"), "/root/src/tac")
    .add_local_dir(str(REPO_ROOT / "experiments"), "/root/experiments")
)

results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)

# Backward-compat aliases (kept so any existing call sites importing these names still work)
# These are derived from deploy_config — do NOT edit them here; edit deploy_config.py instead.
_BASE_CMD: list[str] = build_flags(variant="base", provider_script_path=_MODAL_SCRIPT_PATH)
TRAINING_CMD_SUPERVISED: list[str] = build_flags(variant="supervised", provider_script_path=_MODAL_SCRIPT_PATH)
TRAINING_CMD_RAFT_ONLY: list[str] = build_flags(variant="raft_only", provider_script_path=_MODAL_SCRIPT_PATH)
TRAINING_CMD_TEMPLATE: list[str] = _BASE_CMD


def _run_with_periodic_commits(cmd: list[str], env: dict, commit_interval: int = 300):
    """Run a subprocess with periodic Modal volume commits.

    Commits results_vol every commit_interval seconds during training,
    so partial results survive if the local CLI is killed or times out.
    """
    import subprocess
    import threading
    import time

    training_done = threading.Event()

    def _periodic_commit():
        while not training_done.is_set():
            training_done.wait(timeout=commit_interval)
            if not training_done.is_set():
                try:
                    results_vol.commit()
                    print(f"  [volume] Periodic commit at {time.strftime('%H:%M:%S')}")
                except Exception as e:
                    print(f"  [volume] Commit failed: {e}")

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    try:
        result = subprocess.run(cmd, env=env)
    finally:
        training_done.set()
        results_vol.commit()
        print("  [volume] Final commit done")

    return result


def _find_latest_checkpoint(vol_dir: str) -> str | None:
    """Find the latest checkpoint in the volume directory for resume.

    The Fridrich script saves checkpoints as:
      renderer_epoch00500.pt, renderer_epoch01000.pt, ...
      renderer_best.pt
      renderer_epoch*_timeout.pt
      renderer_epoch*_constraints_met.pt
    """
    import glob
    import os

    patterns = [
        os.path.join(vol_dir, "renderer_epoch*.pt"),
        os.path.join(vol_dir, "renderer_best.pt"),
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        return None

    # Return most recently modified checkpoint
    return max(candidates, key=os.path.getmtime)


def _resolve_eval_candidate(vol_dir: str, strategy: str) -> tuple[str, dict]:
    """Resolve which checkpoint to eval based on a selection strategy.

    Strategies:
      best_proxy             — epoch with lowest proxy score in history.json
                               (full_eval_score if present, else score_projection);
                               if no exact .pt exists, falls back to nearest
                               _constraints_met checkpoint within ±300 epochs.
      best_proxy_constraints_met — best proxy score in history.json restricted
                               to epochs that have a _constraints_met checkpoint.
      earliest_constraints_met — lowest epoch number among *_constraints_met.pt files.
      latest_constraints_met   — highest epoch number among *_constraints_met.pt files.

    Returns:
      (checkpoint_basename, metadata_dict)
        metadata_dict keys: selected_epoch, selection_method, proxy_score,
                            has_constraints_met, resume_floor_warning
    """
    import glob
    import json
    import os
    import re

    # ── Collect all constraints_met checkpoints ────────────────────────────────
    cm_files = glob.glob(os.path.join(vol_dir, "renderer_epoch*_constraints_met.pt"))
    cm_epochs: dict[int, str] = {}
    for f in cm_files:
        m = re.search(r"renderer_epoch(\d+)_constraints_met\.pt$", f)
        if m:
            ep = int(m.group(1))
            cm_epochs[ep] = os.path.basename(f)

    # ── Collect all periodic epoch checkpoints ────────────────────────────────
    periodic_files = glob.glob(os.path.join(vol_dir, "renderer_epoch*.pt"))
    periodic_epochs: dict[int, str] = {}
    for f in periodic_files:
        m = re.search(r"renderer_epoch(\d+)\.pt$", f)  # strict: no suffix after epoch
        if m:
            ep = int(m.group(1))
            periodic_epochs[ep] = os.path.basename(f)

    # ── Load history.json ─────────────────────────────────────────────────────
    history_path = os.path.join(vol_dir, "history.json")
    history: list[dict] = []
    resume_floor_warning = False
    if os.path.exists(history_path):
        with open(history_path) as hf:
            history = json.load(hf)
        # Detect resume floor: if best_score in early entries != inf, the run resumed
        if history and history[0].get("best_score") not in (None, float("inf")):
            resume_floor_warning = True

    # Filter history entries that have proxy scores.
    # full_eval_score: from the full SegNet+PoseNet eval pass (eval_every cadence).
    #   May be absent if the eval silently crashed (OOM loading all models simultaneously,
    #   wrapped in try/except that swallows exceptions in train_renderer_fridrich.py).
    # score_projection: batch-level proxy computed every log_every steps.
    #   Always present — used as fallback when full_eval_score is unavailable.
    def _proxy_score(h: dict) -> float | None:
        if h.get("full_eval_score") is not None:
            return float(h["full_eval_score"])
        if h.get("score_projection") is not None:
            return float(h["score_projection"])
        return None

    scored = [h for h in history if _proxy_score(h) is not None]
    # Attach a unified proxy_score key for downstream min()
    for h in scored:
        h["_proxy"] = _proxy_score(h)

    def _find_pt_near_epoch(target_ep: int, window: int = 300) -> str | None:
        """Find nearest checkpoint (.pt) within window epochs of target_ep."""
        # Prefer exact periodic checkpoint, then nearest constraints_met
        if target_ep in periodic_epochs:
            return periodic_epochs[target_ep]
        candidates_near = {ep: name for ep, name in {**periodic_epochs, **cm_epochs}.items()
                           if abs(ep - target_ep) <= window}
        if not candidates_near:
            return None
        return candidates_near[min(candidates_near, key=lambda ep: abs(ep - target_ep))]

    # ── Strategy dispatch ─────────────────────────────────────────────────────
    if strategy == "best_proxy":
        if not scored:
            raise ValueError(f"No proxy-scored entries in history.json at {history_path}")
        best_entry = min(scored, key=lambda h: h["_proxy"])
        target_ep = best_entry["epoch"]
        ckpt_name = _find_pt_near_epoch(target_ep)
        if not ckpt_name:
            raise FileNotFoundError(
                f"best_proxy: best epoch={target_ep} proxy={best_entry['_proxy']:.4f} "
                f"but no checkpoint within 300 epochs found in {vol_dir}"
            )
        return ckpt_name, {
            "selected_epoch": target_ep,
            "selection_method": "best_proxy",
            "proxy_score": best_entry["_proxy"],
            "has_constraints_met": target_ep in cm_epochs,
            "resume_floor_warning": resume_floor_warning,
        }

    elif strategy == "best_proxy_constraints_met":
        # Best proxy score among epochs that have a constraints_met checkpoint.
        # history.json eval cadence (eval_every) may not align with constraints_met
        # epochs; use a 300-epoch proximity window as the primary join, then fall
        # back to best_proxy over all scored entries if no match is found.
        cm_scored = [h for h in scored if h["epoch"] in cm_epochs]
        if not cm_scored and cm_epochs:
            # Proximity join: any scored epoch within 300 epochs of any cm checkpoint
            cm_scored = [
                h for h in scored
                if any(abs(h["epoch"] - ep) <= 300 for ep in cm_epochs)
            ]
        if not cm_scored:
            # Hard fallback: no history entries near constraints_met at all
            # (sparse eval_every, no constraints ever met in scored range, etc.)
            # Use best_proxy over all scored entries and select the nearest
            # constraints_met checkpoint if one exists.
            print(f"  [strategy] best_proxy_constraints_met: no scored entries near _constraints_met "
                  f"checkpoints; falling back to best_proxy")
            if scored:
                best_entry = min(scored, key=lambda h: h["_proxy"])
                target_ep = best_entry["epoch"]
                # Still prefer nearest cm checkpoint for the actual file
                if cm_epochs:
                    nearest_cm_ep = min(cm_epochs, key=lambda ep: abs(ep - target_ep))
                    ckpt_name = cm_epochs[nearest_cm_ep]
                else:
                    ckpt_name = _find_pt_near_epoch(target_ep)
                if not ckpt_name:
                    raise FileNotFoundError(
                        f"best_proxy fallback: best epoch={target_ep} but no checkpoint within 300 epochs"
                    )
                return ckpt_name, {
                    "selected_epoch": target_ep,
                    "selection_method": "best_proxy_fallback_from_bpcm",
                    "proxy_score": best_entry["_proxy"],
                    "has_constraints_met": bool(cm_epochs),
                    "resume_floor_warning": resume_floor_warning,
                }
            elif cm_epochs:
                # No history at all — use latest constraints_met as last resort
                ep = max(cm_epochs)
                print(f"  [strategy] No history.json scored entries at all; using latest_constraints_met ep={ep}")
                return cm_epochs[ep], {
                    "selected_epoch": ep,
                    "selection_method": "latest_constraints_met_fallback",
                    "proxy_score": None,
                    "has_constraints_met": True,
                    "resume_floor_warning": resume_floor_warning,
                }
            else:
                raise FileNotFoundError(
                    f"best_proxy_constraints_met: no history.json scored entries and no "
                    f"_constraints_met checkpoints in {vol_dir}. "
                    f"Run with --strategy single --checkpoint <explicit_name>."
                )
        best_entry = min(cm_scored, key=lambda h: h["_proxy"])
        target_ep = best_entry["epoch"]
        # Prefer exact constraints_met checkpoint nearest to target_ep
        if cm_epochs:
            nearest_cm_ep = min(cm_epochs, key=lambda ep: abs(ep - target_ep))
            ckpt_name = cm_epochs[nearest_cm_ep]
        else:
            ckpt_name = _find_pt_near_epoch(target_ep)
        if not ckpt_name:
            raise FileNotFoundError(f"best_proxy_constraints_met: no checkpoint found near ep={target_ep}")
        return ckpt_name, {
            "selected_epoch": target_ep,
            "selection_method": "best_proxy_constraints_met",
            "proxy_score": best_entry["_proxy"],
            "has_constraints_met": True,
            "resume_floor_warning": resume_floor_warning,
        }

    elif strategy == "earliest_constraints_met":
        if not cm_epochs:
            raise FileNotFoundError(f"No _constraints_met checkpoints found in {vol_dir}")
        ep = min(cm_epochs)
        return cm_epochs[ep], {
            "selected_epoch": ep,
            "selection_method": "earliest_constraints_met",
            "proxy_score": None,
            "has_constraints_met": True,
            "resume_floor_warning": resume_floor_warning,
        }

    elif strategy == "latest_constraints_met":
        if not cm_epochs:
            raise FileNotFoundError(f"No _constraints_met checkpoints found in {vol_dir}")
        ep = max(cm_epochs)
        return cm_epochs[ep], {
            "selected_epoch": ep,
            "selection_method": "latest_constraints_met",
            "proxy_score": None,
            "has_constraints_met": True,
            "resume_floor_warning": resume_floor_warning,
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. "
                         f"Valid: best_proxy, best_proxy_constraints_met, "
                         f"earliest_constraints_met, latest_constraints_met")


def _setup_results_symlink(vol_dir: str) -> None:
    """Symlink the script's hardcoded RESULTS_DIR to the volume.

    The training script writes to experiments/results/fridrich_renderer/.
    We point that to /results/<tag>/ on the persistent volume so all
    checkpoints, logs, and summaries are automatically persisted.
    """
    import os
    import shutil

    parent = os.path.dirname(SCRIPT_RESULTS_DIR)
    os.makedirs(parent, exist_ok=True)

    # Remove the dir if it exists (from add_local_dir mount)
    if os.path.exists(SCRIPT_RESULTS_DIR) and not os.path.islink(SCRIPT_RESULTS_DIR):
        shutil.rmtree(SCRIPT_RESULTS_DIR)
    elif os.path.islink(SCRIPT_RESULTS_DIR):
        os.unlink(SCRIPT_RESULTS_DIR)

    os.symlink(vol_dir, SCRIPT_RESULTS_DIR)
    print(f"  Symlink: {SCRIPT_RESULTS_DIR} -> {vol_dir}")


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 6,  # 6h hard timeout
    volumes={"/results": results_vol},
    memory=16384,
)
def train_asymmetric_warp(
    tag: str,
    extra_args: list[str] | None = None,
    variant: str = "base",
    resume_from: str | None = None,
):
    """Run asymmetric warp renderer training on T4.

    Args:
        tag: Output directory name on the results volume.
        extra_args: Additional CLI flags appended after the base command.
        variant: Experiment variant to run:
            "base"       — base Lagrangian training, no supervision layers (default)
            "supervised" — Path A: PoseNet supervision (Layer 1) + RAFT flow (Layer 2)
            "raft_only"  — Path B: RAFT flow supervision only (isolates Layer 2)
        resume_from: If set, resume from this checkpoint path (overrides auto-detect).
            Example: "/results/asym_v3_longer_tight/renderer_best.pt"
    """
    import os
    import time

    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from: {ALL_VARIANTS}")

    vol_dir = f"/results/{tag}"
    os.makedirs(vol_dir, exist_ok=True)

    # Symlink hardcoded RESULTS_DIR -> volume path
    _setup_results_symlink(vol_dir)

    # Resume detection: explicit override > auto-detect from volume
    if resume_from:
        checkpoint = resume_from
    else:
        checkpoint = _find_latest_checkpoint(vol_dir)

    print(f"=== tac asymmetric warp training | tag: {tag} | variant: {variant} ===")
    print(f"  GPU: T4 (council decision: iteration budget over speed)")
    print(f"  Wall-clock budget: 5.5h training / 6h timeout")
    print(f"  Resume: {'YES -> ' + checkpoint if checkpoint else 'NO (fresh start)'}")
    print(f"  Output: {vol_dir}")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Build command from centralized deploy_config (provider-agnostic flags + Modal path)
    cmd = build_flags(
        variant=variant,
        provider_script_path=_MODAL_SCRIPT_PATH,
        resume_from=checkpoint,
        extra=extra_args,
    )

    env = {**os.environ, "PYTHONPATH": "/root/src:/root/upstream"}

    print(f"  Command: {' '.join(cmd)}")
    print("  ---")

    # --- Deployment manifest: record everything needed to reproduce this run ---
    # Saved BEFORE training so it survives even if training crashes.
    # Complies with "DX must record and archive all necessary to replicate variants."
    import json
    import socket

    manifest = {
        "tag": tag,
        "variant": variant,
        "resume_from": checkpoint,
        "full_command": cmd,
        "extra_args": extra_args,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "gpu": "T4",
        "provider": "modal",
        "variant_extra_flags": VARIANT_FLAGS[variant],  # extra flags on top of BASE_FLAGS
    }
    manifest_path = os.path.join(vol_dir, "deployment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    results_vol.commit()
    print(f"  Manifest saved: {manifest_path}")

    result = _run_with_periodic_commits(cmd, env=env)

    print("  ---")
    print(f"  End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Exit code: {result.returncode}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Training subprocess failed with exit code {result.returncode}. "
            f"Check logs above for details."
        )

    # List artifacts saved
    artifacts = sorted(os.listdir(vol_dir))
    print(f"  Artifacts ({len(artifacts)}): {', '.join(artifacts[:10])}")
    if len(artifacts) > 10:
        print(f"    ... and {len(artifacts) - 10} more")

    return {"tag": tag, "exit_code": result.returncode, "artifacts": artifacts}


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,  # 1h — inflation + scoring is fast
    volumes={"/results": results_vol},
    memory=16384,
)
def auth_eval(tag: str, checkpoint: str = "renderer_best.pt", strategy: str = "single"):
    """Run authoritative evaluation of a checkpoint on the upstream scorer.

    Full pipeline:
        1. Resolve checkpoint (single explicit name, or via strategy)
        2. Load checkpoint from /results/<tag>/<checkpoint>
        3. Load upstream SegNet for mask extraction
        4. Decode GT video (upstream/videos/0.mkv via PyAV)
        5. Extract masks via SegNet
        6. Generate frames via renderer (asymmetric pair or independent)
        7. Upscale to 1164x874, write .raw
        8. Score via upstream DistortionNet (PoseNet + SegNet)
        9. Compute rate from checkpoint file size
       10. Final score: 100*seg + sqrt(10*pose) + 25*rate

    Args:
        tag:        experiment tag (volume subdirectory)
        checkpoint: explicit checkpoint filename (used when strategy="single")
        strategy:   checkpoint selection strategy when no explicit checkpoint is given.
                    Options: single, best_proxy, best_proxy_constraints_met,
                             earliest_constraints_met, latest_constraints_met.
                    When strategy != "single", checkpoint is resolved automatically
                    from history.json and available .pt files.

    Results are saved to /results/<tag>/auth_eval_<checkpoint>.json
    """
    import json
    import math
    import os
    import sys
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    t_start = time.monotonic()

    vol_dir = f"/results/{tag}"

    # ── Checkpoint resolution ─────────────────────────────────────────────────
    # strategy != "single": resolve checkpoint from history.json + .pt inventory.
    # strategy == "single": use the explicit checkpoint name; fall back to latest
    #   if the file doesn't exist (original behaviour, preserved for compat).
    _strategy_meta: dict = {}
    if strategy != "single":
        resolved, _strategy_meta = _resolve_eval_candidate(vol_dir, strategy)
        checkpoint = resolved
        print(f"  [strategy={strategy}] resolved checkpoint: {checkpoint}")
        if _strategy_meta.get("resume_floor_warning"):
            print(f"  WARNING: resume_floor_active — best_score inherited from prior run. "
                  f"Selection via '{strategy}' is independent of that artifact.")
        if _strategy_meta.get("proxy_score") is not None:
            print(f"  Proxy score at selection epoch: {_strategy_meta['proxy_score']:.4f}")

    ckpt_path = os.path.join(vol_dir, checkpoint)

    print(f"=== Auth Eval | tag: {tag} | checkpoint: {checkpoint} ===")
    print(f"  GPU: T4")
    print(f"  Strategy: {strategy}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(ckpt_path):
        if strategy != "single":
            raise FileNotFoundError(
                f"Resolved checkpoint not found: {ckpt_path}\n"
                f"Strategy '{strategy}' returned '{checkpoint}' but file is missing."
            )
        # single fallback: use latest available checkpoint (original behaviour)
        alt = _find_latest_checkpoint(vol_dir)
        if alt:
            print(f"  WARNING: {ckpt_path} not found, using {alt}")
            ckpt_path = alt
            checkpoint = os.path.basename(alt)
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Available files: {os.listdir(vol_dir) if os.path.isdir(vol_dir) else 'dir not found'}"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Auth eval loads scorers sequentially (not simultaneously like _full_eval),
    # so larger batches are safe. Keep conservative on CPU for memory.
    batch_size = 8 if device == "cuda" else 2
    print(f"  Device: {device}, eval batch_size: {batch_size}")

    # ── Constants ──
    OUT_W, OUT_H = 1164, 874
    SEG_W, SEG_H = 512, 384
    NUM_FRAMES = 1200
    UPSTREAM_ROOT = "/root/upstream"

    # ── 1. Load upstream SegNet for mask extraction ──
    print("\nStage 1: Loading SegNet ...")
    sys.path.insert(0, UPSTREAM_ROOT)
    from modules import SegNet, DistortionNet
    from modules import segnet_sd_path, posenet_sd_path
    from safetensors.torch import load_file

    segnet = SegNet()
    sd = load_file(str(segnet_sd_path), device=device)
    segnet.load_state_dict(sd)
    segnet.to(device).eval()
    for p in segnet.parameters():
        p.requires_grad = False
    print("  SegNet loaded.")

    # ── 2. Load renderer from checkpoint ──
    print("\nStage 2: Loading renderer ...")
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_size_bytes = os.path.getsize(ckpt_path)
    print(f"  Checkpoint size: {ckpt_size_bytes:,} bytes")

    # Determine checkpoint format: .bin export or .pt training checkpoint
    # Check for ASYM/DPSM magic at start of file
    is_bin = False
    with open(ckpt_path, "rb") as bf:
        magic = bf.read(4)
        is_bin = magic in (b"ASYM", b"DPSM")

    if is_bin:
        # Binary export — load via tac.renderer_export
        print(f"  Binary export detected (magic: {magic})")
        from tac.renderer_export import load_asymmetric_checkpoint
        model = load_asymmetric_checkpoint(ckpt_path, device=device)
        model.eval()
        archive_size = ckpt_size_bytes
        print(f"  Rate from .bin: {archive_size:,} bytes")
    elif isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
        # Training checkpoint — reconstruct model from config
        config = ckpt_data.get("config", {})
        print(f"  Training checkpoint, epoch={ckpt_data.get('epoch', '?')}")
        if ckpt_data.get("best_score"):
            print(f"  Proxy best_score={ckpt_data['best_score']:.4f}")

        from tac.renderer import AsymmetricPairGenerator
        model = AsymmetricPairGenerator(
            num_classes=config.get("num_classes", 5),
            embed_dim=config.get("embed_dim", 6),
            base_ch=config.get("base_ch", 36),
            mid_ch=config.get("mid_ch", 60),
            motion_hidden=config.get("motion_hidden", 32),
            depth=config.get("renderer_depth", config.get("depth", 1)),
            max_flow_px=config.get("max_flow_px", 20.0),
            max_residual=config.get("max_residual", 20.0),
            flow_only=config.get("flow_only", False),
        )
        model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
        model.to(device).eval()

        # For rate calculation, use the export size (quantized .bin).
        bin_candidates = [
            os.path.join(vol_dir, "renderer.bin"),
            os.path.join(vol_dir, "renderer_best.bin"),
        ]
        archive_size = None
        for bc in bin_candidates:
            if os.path.exists(bc):
                archive_size = os.path.getsize(bc)
                print(f"  Rate from .bin export: {bc} ({archive_size:,} bytes)")
                break

        if archive_size is None:
            print("  No .bin export found — exporting for rate calculation ...")
            try:
                from pathlib import Path as _ExportPath
                from tac.renderer_export import export_asymmetric_checkpoint
                bin_path = os.path.join(vol_dir, f"renderer_{checkpoint.replace('.pt', '')}.bin")
                archive_size = export_asymmetric_checkpoint(model, output_path=_ExportPath(bin_path), default_bits=4)
                print(f"  Exported: {bin_path} ({archive_size:,} bytes)")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot determine accurate archive size for rate calculation. "
                    f"No companion .bin found and export failed: {e}."
                )
    else:
        raise ValueError(
            f"Unsupported checkpoint format. Expected .pt with 'model_state_dict' "
            f"or .bin with ASYM/DPSM magic. Got: "
            f"{list(ckpt_data.keys()) if isinstance(ckpt_data, dict) else type(ckpt_data)}"
        )

    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Renderer loaded: {n_params:,} params")
    del ckpt_data

    # ── 3. Decode GT video ──
    print("\nStage 3: Decoding GT video ...")
    import av

    def _yuv420_to_rgb(frame):
        H, W = frame.height, frame.width
        y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
        u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
        v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]
        y_t = torch.from_numpy(y.copy()).float()
        u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
        v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)
        u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
        v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
        yf = (y_t - 16.0) * (255.0 / 219.0)
        uf = (u_up - 128.0) * (255.0 / 224.0)
        vf = (v_up - 128.0) * (255.0 / 224.0)
        r = (yf + 1.402 * vf).clamp(0, 255)
        g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
        b = (yf + 1.772 * uf).clamp(0, 255)
        return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)

    gt_video_path = os.path.join(UPSTREAM_ROOT, "videos", "0.mkv")
    container = av.open(gt_video_path)
    stream = container.streams.video[0]
    gt_frames = []
    for frame in container.decode(stream):
        gt_frames.append(_yuv420_to_rgb(frame).numpy())
    container.close()
    print(f"  Decoded {len(gt_frames)} GT frames")
    assert len(gt_frames) == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {len(gt_frames)}"

    # ── 4. Extract masks via SegNet ──
    print("\nStage 4: Extracting SegNet masks ...")
    t_mask = time.monotonic()
    masks_list = []
    with torch.inference_mode():
        for i in range(0, len(gt_frames), batch_size):
            end = min(i + batch_size, len(gt_frames))
            batch_np = np.stack(gt_frames[i:end], axis=0)
            batch_t = torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device)
            inp = batch_t.unsqueeze(1)  # (B, 1, 3, H, W) for preprocess_input
            seg_in = segnet.preprocess_input(inp)
            logits = segnet(seg_in)
            mask = logits.argmax(dim=1)
            masks_list.append(mask.to(torch.int8).cpu())
    masks = torch.cat(masks_list, dim=0)  # (N, 384, 512) int8
    print(f"  Extracted {masks.shape[0]} masks ({time.monotonic() - t_mask:.1f}s)")
    del gt_frames, segnet, masks_list  # free VRAM

    # ── 5. Generate frames via renderer ──
    print("\nStage 5: Generating frames ...")
    t_gen = time.monotonic()

    # Detect asymmetric mode
    is_asymmetric = (
        type(model).__name__ == "AsymmetricPairGenerator"
        or (hasattr(model, "renderer") and hasattr(model, "motion"))
    )

    raw_path = os.path.join(vol_dir, "auth_eval_inflated.raw")
    n_written = 0
    torch.manual_seed(42)

    with open(raw_path, "wb") as f:
        with torch.inference_mode():
            if is_asymmetric:
                print(f"  Mode: asymmetric pair generation ({len(masks)} masks)")
                N = masks.shape[0]
                pair_idx = 0
                while pair_idx < N - 1:
                    batch_t_list = []
                    batch_t1_list = []
                    batch_end = min(pair_idx + batch_size * 2, N - 1)
                    for j in range(pair_idx, batch_end, 2):
                        if j + 1 < N:
                            batch_t_list.append(masks[j])
                            batch_t1_list.append(masks[j + 1])
                    if not batch_t_list:
                        break
                    masks_t = torch.stack(batch_t_list).to(device=device, dtype=torch.long)
                    masks_t1 = torch.stack(batch_t1_list).to(device=device, dtype=torch.long)
                    pairs = model(masks_t, masks_t1)  # (B, 2, H, W, 3) HWC
                    B_pairs = pairs.shape[0]
                    for p_idx in range(B_pairs):
                        for frame_idx in range(2):
                            frame_hwc = pairs[p_idx, frame_idx]
                            frame_chw = frame_hwc.permute(2, 0, 1).unsqueeze(0)
                            frame_up = F.interpolate(
                                frame_chw, size=(OUT_H, OUT_W),
                                mode="bilinear", align_corners=False,
                            )
                            frame_uint8 = frame_up.round().clamp(0, 255).to(torch.uint8)
                            frame_out = frame_uint8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                            f.write(frame_out.tobytes())
                            n_written += 1
                    pair_idx += len(batch_t_list) * 2
                    if n_written % 200 == 0 or pair_idx >= N - 1:
                        print(f"    Generated: {n_written}/{N} frames")
                # Handle odd trailing mask
                if N % 2 != 0:
                    last_mask = masks[N - 1:N].to(device=device, dtype=torch.long)
                    frame = model.renderer(last_mask)
                    frame_up = F.interpolate(frame, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    frame_uint8 = frame_up.round().clamp(0, 255).to(torch.uint8)
                    frame_out = frame_uint8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                    f.write(frame_out.tobytes())
                    n_written += 1
            else:
                print(f"  Mode: independent frame generation ({len(masks)} masks)")
                for i in range(0, masks.shape[0], batch_size):
                    end = min(i + batch_size, masks.shape[0])
                    batch_masks = masks[i:end].to(device=device, dtype=torch.long)
                    frames = model(batch_masks)
                    frames_up = F.interpolate(frames, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    frames_uint8 = frames_up.round().clamp(0, 255).to(torch.uint8)
                    frames_hwc = frames_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    f.write(frames_hwc.tobytes())
                    n_written += batch_masks.shape[0]
                    if end % 200 == 0 or end == masks.shape[0]:
                        print(f"    Generated: {end}/{masks.shape[0]} frames")

    raw_size = os.path.getsize(raw_path)
    expected_size = OUT_W * OUT_H * 3 * n_written
    assert raw_size == expected_size, f"Raw size mismatch: {raw_size} vs {expected_size}"
    assert n_written == NUM_FRAMES, (
        f"Frame count mismatch: generated {n_written} but scorer expects {NUM_FRAMES}. "
        f"Check mask extraction — decode failure may have produced fewer frames."
    )
    print(f"  Generated {n_written} frames ({time.monotonic() - t_gen:.1f}s)")
    del model, masks  # free VRAM

    # ── 6. Score via upstream evaluate.py (DALI + CUDA — leaderboard-grade) ──
    # This is the ONLY correct scoring path. PyAV produces 29x PoseNet divergence.
    print("\nStage 6: Scoring via upstream evaluate.py (DALI) ...")
    t_score = time.monotonic()

    # Set up submission directory structure that evaluate.py expects
    submission_dir = os.path.join(vol_dir, "submission")
    os.makedirs(os.path.join(submission_dir, "inflated"), exist_ok=True)

    # Create archive.zip with just the .bin for rate calculation
    import zipfile
    archive_zip = os.path.join(submission_dir, "archive.zip")
    bin_path_for_archive = None
    for candidate in [os.path.join(vol_dir, "renderer_best.bin"),
                      os.path.join(vol_dir, f"renderer_{checkpoint.replace('.pt', '')}.bin")]:
        if os.path.exists(candidate):
            bin_path_for_archive = candidate
            break
    if bin_path_for_archive is None:
        raise FileNotFoundError("No .bin export found for archive — rate calculation impossible")

    with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_STORED) as zf:
        zf.write(bin_path_for_archive, "renderer.bin")
    archive_size = os.path.getsize(archive_zip)
    print(f"  Archive: {archive_zip} ({archive_size:,} bytes)")

    # Move .raw to submission/inflated/0.raw
    expected_raw = os.path.join(submission_dir, "inflated", "0.raw")
    if raw_path != expected_raw:
        if os.path.exists(expected_raw):
            os.remove(expected_raw)
        os.rename(raw_path, expected_raw)
        raw_path = expected_raw

    # Run upstream evaluate.py — uses DALI on CUDA for GT decode (leaderboard match)
    import subprocess
    report_path = os.path.join(submission_dir, "report.txt")
    # Use upstream evaluate.py with DALI on CUDA for leaderboard-grade scoring.
    # DALI_DISABLE_NVML=1 is set in the container env to work around Modal's
    # lack of NVML access. DALI video decode still uses GPU (NVDEC), it just
    # can't query GPU management info. Falls back to PyAV if DALI fails.
    eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_cmd = [
        sys.executable, os.path.join(UPSTREAM_ROOT, "evaluate.py"),
        "--submission-dir", submission_dir,
        "--uncompressed-dir", os.path.join(UPSTREAM_ROOT, "videos"),
        "--video-names-file", os.path.join(UPSTREAM_ROOT, "public_test_video_names.txt"),
        "--device", eval_device,
        "--report", report_path,
    ]
    print(f"  Command: {' '.join(eval_cmd)}")
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)
    if eval_result.returncode != 0:
        print(f"  STDERR: {eval_result.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError(f"evaluate.py failed with exit code {eval_result.returncode}")

    # Parse the report
    import re
    report_text = open(report_path).read()
    print(f"  Report:\n{report_text}")

    def _parse(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1).replace(",", "")) if m else None

    avg_posenet = _parse(r"Average PoseNet Distortion:\s*([0-9.]+)", report_text)
    avg_segnet = _parse(r"Average SegNet Distortion:\s*([0-9.]+)", report_text)
    rate = _parse(r"Compression Rate:\s*([0-9.]+)", report_text)
    final_score_parsed = _parse(r"Final score:.*=\s*([0-9.]+)", report_text)
    n_samples_raw = _parse(r"over (\d+) samples", report_text)
    n_samples = int(n_samples_raw) if n_samples_raw else 600

    gt_size = os.path.getsize(os.path.join(UPSTREAM_ROOT, "videos", "0.mkv"))
    if rate is None:
        rate = archive_size / gt_size

    # ── 8. Final score ──
    # Prefer the score parsed directly from evaluate.py (most authoritative)
    score = final_score_parsed if final_score_parsed else (
        100 * avg_segnet + math.sqrt(10 * avg_posenet) + 25 * rate
    )

    t_total = time.monotonic() - t_start

    print(f"\n{'=' * 60}")
    print(f"=== Authoritative Evaluation Results ({n_samples} samples) ===")
    print(f"{'=' * 60}")
    print(f"  Average PoseNet Distortion: {avg_posenet:.8f}")
    print(f"  Average SegNet Distortion:  {avg_segnet:.8f}")
    print(f"  Archive size:               {archive_size:,} bytes")
    print(f"  GT size:                    {gt_size:,} bytes")
    print(f"  Compression Rate:           {rate:.8f}")
    print(f"  Score breakdown:")
    print(f"    100*seg  = {100 * avg_segnet:.4f}")
    print(f"    sqrt(10*pose) = {math.sqrt(10 * avg_posenet):.4f}")
    print(f"    25*rate  = {25 * rate:.4f}")
    print(f"  FINAL SCORE: {score:.4f}")
    print(f"  Total time: {t_total:.1f}s")
    print(f"{'=' * 60}")

    # ── 9. Save results ──
    result = {
        "tag": tag,
        "checkpoint": checkpoint,
        "checkpoint_path": ckpt_path,
        "avg_posenet_dist": avg_posenet,
        "avg_segnet_dist": avg_segnet,
        "archive_size_bytes": archive_size,
        "gt_size_bytes": gt_size,
        "rate": rate,
        "score_seg": 100 * avg_segnet,
        "score_pose": math.sqrt(10 * avg_posenet),
        "score_rate": 25 * rate,
        "final_score": score,
        "n_samples": n_samples,
        "n_frames": n_written,
        "eval_method": f"upstream_evaluate_py_{eval_device}",
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": t_total,
        "selection_strategy": strategy,
        "selection_meta": _strategy_meta if _strategy_meta else None,
    }

    result_filename = f"auth_eval_{checkpoint.replace('.pt', '').replace('.bin', '')}.json"
    result_path = os.path.join(vol_dir, result_filename)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    # Clean up the .raw file (large, ~3.6GB)
    if os.path.exists(raw_path):
        os.remove(raw_path)
        print(f"  Cleaned up: {raw_path}")

    results_vol.commit()
    print("  Volume committed.")

    return result


@app.local_entrypoint()
def auth_eval_entry(
    tag: str = "asymmetric_warp_t4",
    checkpoint: str = "",
    strategy: str = "best_proxy_constraints_met",
):
    """Launch authoritative evaluation on Modal T4.

    Args:
        tag:        experiment tag (directory name on results volume)
        checkpoint: explicit checkpoint filename within /results/<tag>/.
                    If empty, strategy is used to resolve the best checkpoint.
        strategy:   checkpoint selection when checkpoint is not specified (default).
                    Options:
                      best_proxy_constraints_met  (DEFAULT) — best proxy score among
                          epochs with a _constraints_met checkpoint; immune to the
                          resume floor artifact where renderer_best.pt is frozen.
                      best_proxy                  — best proxy score overall (any .pt)
                      earliest_constraints_met    — first epoch that passed both constraints
                      latest_constraints_met      — last epoch that passed both constraints
                      single                      — use explicit checkpoint name only

    Usage:
        # Recommended — immune to resume floor:
        .venv/bin/modal run ...:app.auth_eval_entry --tag asym_v4_supervised

        # Explicit checkpoint (legacy):
        .venv/bin/modal run ...:app.auth_eval_entry --tag my_run --checkpoint renderer_best.pt --strategy single
    """
    from tac.cost_tracker import print_cost_estimate

    # If an explicit checkpoint is given with no strategy override, use "single"
    if checkpoint and strategy == "best_proxy_constraints_met":
        strategy = "single"

    print(f"\n=== Auth Eval -> Modal T4 ===")
    print(f"  Tag:      {tag}")
    print(f"  Strategy: {strategy}")
    if checkpoint:
        print(f"  Checkpoint (explicit): {checkpoint}")
    print(f"  Volume: {RESULTS_VOL}")

    print("\n--- Cost Estimate ---")
    print_cost_estimate(gpu="t4", estimated_hours=0.5, platform="modal")

    print("\nLaunching auth eval ...")
    result = auth_eval.remote(tag=tag, checkpoint=checkpoint or "renderer_best.pt", strategy=strategy)

    meta = result.get("selection_meta") or {}
    print(f"\n=== Auth Eval Complete ===")
    print(f"  Checkpoint evaluated: {result['checkpoint']}")
    if meta.get("selection_method"):
        print(f"  Selection method:     {meta['selection_method']}")
    if meta.get("proxy_score") is not None:
        print(f"  Proxy score:          {meta['proxy_score']:.4f}")
    if meta.get("resume_floor_warning"):
        print(f"  NOTE: resume_floor_active — renderer_best.pt was frozen at prior run's best. "
              f"This eval used strategy '{strategy}' which is immune to that artifact.")
    print(f"  Final Score: {result['final_score']:.4f}")
    print(f"    SegNet:  {result['score_seg']:.4f}")
    print(f"    PoseNet: {result['score_pose']:.4f}")
    print(f"    Rate:    {result['score_rate']:.4f}")
    print(f"  Time: {result['elapsed_seconds']:.0f}s")
    print(f"\nFull results: .venv/bin/modal volume get {RESULTS_VOL} {tag}/auth_eval_*.json ./")
    return result


@app.local_entrypoint()
def main(
    tag: str = "asymmetric_warp_t4",
    extra_args: str = "",
    variant: str = "base",
    resume_from: str = "",
):
    """Launch asymmetric warp training on Modal T4.

    Args:
        tag: experiment tag for output directory on results volume
        extra_args: space-separated extra CLI args (e.g., '--smoke')
        variant: Experiment variant:
            "base"       — base Lagrangian, no supervision (default)
            "supervised" — Path A: PoseNet supervision + RAFT flow (Layers 1+2)
            "raft_only"  — Path B: RAFT flow only (isolates Layer 2)
        resume_from: checkpoint to resume from (e.g., '/results/asym_v3_longer_tight/renderer_best.pt')
                     If empty, auto-detects from tag directory on volume.

    Examples:
        # Path A (PoseNet + RAFT, resume from v3 best):
        .venv/bin/modal run ... --tag asym_v4_supervised --variant supervised \\
            --resume-from /results/asym_v3_longer_tight/renderer_best.pt

        # Path B (RAFT only, fresh start):
        .venv/bin/modal run ... --tag asym_v4_raft_only --variant raft_only \\
            --resume-from /results/asym_v3_longer_tight/renderer_best.pt
    """
    from tac.cost_tracker import print_cost_estimate

    print(f"\n=== Asymmetric Warp Renderer -> Modal T4 ===")
    print(f"  Tag: {tag}")
    print(f"  Variant: {variant}")
    print(f"  Resume from: {resume_from or '(auto-detect)'}")
    print(f"  Volume: {RESULTS_VOL}")

    print("\n--- Cost Estimate ---")
    print_cost_estimate(gpu="t4", estimated_hours=5.5, platform="modal")

    parsed_extra = extra_args.split() if extra_args.strip() else None
    resume_arg = resume_from.strip() or None

    print("\nLaunching training...")
    result = train_asymmetric_warp.remote(
        tag=tag,
        extra_args=parsed_extra,
        variant=variant,
        resume_from=resume_arg,
    )

    status = "OK" if result["exit_code"] == 0 else f"FAILED (exit {result['exit_code']})"
    print(f"\n  Result: {status}")
    artifacts = result["artifacts"]
    print(f"  Artifacts ({len(artifacts)}): {', '.join(artifacts[:10])}")

    # ── Auto auth-eval (council binding decision) ─────────────────────────────
    # Evaluates up to 4 checkpoints unconditionally after a successful run.
    # Selection protocol (ordered, deduped):
    #   1. best_proxy_constraints_met — best proxy score among epochs that satisfy
    #      both Lagrangian constraints. Immune to the resume floor artifact where
    #      renderer_best.pt is frozen at a prior run's score. This is the PRIMARY
    #      eval candidate for comparison with raft_only and other runs.
    #   2. renderer_best.pt     — if it exists (may be stale/frozen on resume runs)
    #   3. renderer_best_ema.pt — EMA shadow best (independent of resume floor)
    #   4. Latest renderer_epoch*.pt — final periodic snapshot (always included so
    #      no run is silently dropped; explicitly labeled as a periodic checkpoint).
    # Each eval is non-destructive: failures are logged but never mask training result.
    epoch_ckpts = sorted(
        f for f in artifacts
        if f.startswith("renderer_epoch") and f.endswith(".pt")
        and not f.endswith("_constraints_met.pt")
        and not f.endswith("_timeout.pt")
    )
    # Use a list to preserve order; use a set to dedup without losing order
    checkpoints_to_eval: list[tuple[str, str]] = []  # (checkpoint_name, strategy)
    seen: set[str] = set()

    def _add_candidate(name: str, strat: str) -> None:
        if name not in seen:
            checkpoints_to_eval.append((name, strat))
            seen.add(name)

    # Primary candidate: best proxy among constraints_met (resume-floor immune)
    _add_candidate("__strategy__", "best_proxy_constraints_met")

    # Named best checkpoints (may be frozen on resume runs — eval anyway for transparency)
    for name in ("renderer_best.pt", "renderer_best_ema.pt"):
        if name in artifacts:
            _add_candidate(name, "single")

    # Latest periodic epoch — unconditional safety net
    if epoch_ckpts:
        _add_candidate(epoch_ckpts[-1], "single")

    if checkpoints_to_eval:
        print(f"\n--- Auto auth-eval ({len(checkpoints_to_eval)} candidates) ---")
        for ckpt_name, strat in checkpoints_to_eval:
            label = strat if ckpt_name == "__strategy__" else ckpt_name
            print(f"  Evaluating [{strat}] {label}...")
            try:
                if ckpt_name == "__strategy__":
                    eval_result = auth_eval.remote(tag=tag, checkpoint="", strategy=strat)
                else:
                    eval_result = auth_eval.remote(tag=tag, checkpoint=ckpt_name, strategy="single")
                resolved = eval_result.get("checkpoint", label)
                score = eval_result.get("final_score", "?")
                meta = eval_result.get("selection_meta") or {}
                proxy = meta.get("proxy_score")
                proxy_str = f" (proxy={proxy:.4f})" if proxy else ""
                print(f"  -> {resolved}{proxy_str}: auth={score}")
            except Exception as exc:
                print(f"  [{strat}] {label}: EVAL FAILED (non-fatal) — {exc}")
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\nResults: .venv/bin/modal volume ls {RESULTS_VOL}")
    print(f"Download: .venv/bin/modal volume get {RESULTS_VOL} {tag}/ ./results_{tag}/")


# ── TTO eval ─────────────────────────────────────────────────────────────────


def _run_tto_auth_eval(tag: str, tto_dir: str) -> dict | None:
    """Run authoritative evaluation on TTO-optimized frames.

    Loads tto_frames.pt, upsamples to camera resolution (874x1164),
    writes raw frames file, and runs upstream evaluate.py for scoring.

    Args:
        tag:     experiment tag (volume subdirectory)
        tto_dir: path to tto_results directory containing tto_frames.pt

    Returns:
        Result dict with scores, or None if tto_frames.pt not found.
    """
    import json
    import math
    import os
    import re
    import subprocess
    import sys
    import time

    import torch
    import torch.nn.functional as F

    t_start = time.monotonic()

    UPSTREAM_ROOT = "/root/upstream"
    OUT_H, OUT_W = 874, 1164
    NUM_FRAMES = 1200
    vol_dir = f"/results/{tag}"

    frames_path = os.path.join(tto_dir, "tto_frames.pt")
    if not os.path.exists(frames_path):
        print(f"  [tto_auth_eval] tto_frames.pt not found at {frames_path}, skipping auth eval")
        return None

    print(f"\n{'=' * 60}")
    print(f"=== TTO Auth Eval | tag: {tag} ===")
    print(f"{'=' * 60}")

    # ── 1. Load TTO frames ──
    print("\nStage 1: Loading TTO frames ...")
    tto_frames = torch.load(frames_path, map_location="cpu", weights_only=True)
    print(f"  Shape: {tto_frames.shape}, dtype: {tto_frames.dtype}")
    assert tto_frames.shape == (NUM_FRAMES, 384, 512, 3), (
        f"Expected (1200, 384, 512, 3), got {tto_frames.shape}"
    )
    assert tto_frames.dtype == torch.uint8, f"Expected uint8, got {tto_frames.dtype}"

    # ── 2. Upsample to camera resolution and write raw file ──
    print(f"\nStage 2: Upsampling {NUM_FRAMES} frames to {OUT_H}x{OUT_W} and writing raw ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    submission_dir = os.path.join(tto_dir, "submission")
    os.makedirs(os.path.join(submission_dir, "inflated"), exist_ok=True)
    raw_path = os.path.join(submission_dir, "inflated", "0.raw")

    batch_size = 16
    n_written = 0
    with open(raw_path, "wb") as f:
        for i in range(0, NUM_FRAMES, batch_size):
            end = min(i + batch_size, NUM_FRAMES)
            # (B, H, W, 3) uint8 -> (B, 3, H, W) float for interpolation
            batch = tto_frames[i:end].float().permute(0, 3, 1, 2).to(device)
            batch_up = F.interpolate(batch, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
            batch_uint8 = batch_up.round().clamp(0, 255).to(torch.uint8)
            # (B, 3, H, W) -> (B, H, W, 3) contiguous for raw write
            batch_hwc = batch_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            f.write(batch_hwc.tobytes())
            n_written += end - i
            if n_written % 200 == 0 or end == NUM_FRAMES:
                print(f"    Upsampled: {n_written}/{NUM_FRAMES}")

    raw_size = os.path.getsize(raw_path)
    expected_size = OUT_H * OUT_W * 3 * NUM_FRAMES
    assert raw_size == expected_size, f"Raw size mismatch: {raw_size} vs {expected_size}"
    print(f"  Written {n_written} frames, raw size: {raw_size:,} bytes")

    del tto_frames  # free memory

    # ── 3. Copy archive.zip for rate calculation ──
    print("\nStage 3: Setting up archive.zip ...")
    archive_zip = os.path.join(submission_dir, "archive.zip")

    # Look for existing archive.zip from a previous auth eval of this tag
    source_archive = os.path.join(vol_dir, "submission", "archive.zip")
    if not os.path.exists(source_archive):
        # Try renderer .bin files to create archive from scratch
        import zipfile
        bin_candidates = [
            os.path.join(vol_dir, "renderer_best.bin"),
            os.path.join(vol_dir, "renderer.bin"),
        ]
        bin_path = None
        for bc in bin_candidates:
            if os.path.exists(bc):
                bin_path = bc
                break
        if bin_path is None:
            print("  ERROR: No archive.zip or .bin found for rate calculation")
            print("  Available files:", os.listdir(vol_dir) if os.path.isdir(vol_dir) else "dir missing")
            return None
        with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_STORED) as zf:
            zf.write(bin_path, "renderer.bin")
        print(f"  Created archive.zip from {bin_path}")
    else:
        import shutil
        if source_archive != archive_zip:
            shutil.copy2(source_archive, archive_zip)
        print(f"  Copied archive.zip from {source_archive}")

    archive_size = os.path.getsize(archive_zip)
    print(f"  Archive size: {archive_size:,} bytes")

    # ── 4. Run upstream evaluate.py ──
    print("\nStage 4: Scoring via upstream evaluate.py (DALI) ...")
    report_path = os.path.join(submission_dir, "report.txt")
    eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_cmd = [
        sys.executable, os.path.join(UPSTREAM_ROOT, "evaluate.py"),
        "--submission-dir", submission_dir,
        "--uncompressed-dir", os.path.join(UPSTREAM_ROOT, "videos"),
        "--video-names-file", os.path.join(UPSTREAM_ROOT, "public_test_video_names.txt"),
        "--device", eval_device,
        "--report", report_path,
    ]
    print(f"  Command: {' '.join(eval_cmd)}")
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)

    if eval_result.returncode != 0:
        print(f"  evaluate.py FAILED (exit {eval_result.returncode})")
        print(f"  STDERR: {eval_result.stderr[-1000:]}")
        return None

    # ── 5. Parse report ──
    report_text = open(report_path).read()
    print(f"  Report:\n{report_text}")

    def _parse(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1).replace(",", "")) if m else None

    avg_posenet = _parse(r"Average PoseNet Distortion:\s*([0-9.]+)", report_text)
    avg_segnet = _parse(r"Average SegNet Distortion:\s*([0-9.]+)", report_text)
    rate = _parse(r"Compression Rate:\s*([0-9.]+)", report_text)
    final_score_parsed = _parse(r"Final score:.*=\s*([0-9.]+)", report_text)
    n_samples_raw = _parse(r"over (\d+) samples", report_text)
    n_samples = int(n_samples_raw) if n_samples_raw else 600

    gt_size = os.path.getsize(os.path.join(UPSTREAM_ROOT, "videos", "0.mkv"))
    if rate is None:
        rate = archive_size / gt_size

    score = final_score_parsed if final_score_parsed else (
        100 * avg_segnet + math.sqrt(10 * avg_posenet) + 25 * rate
    )

    t_total = time.monotonic() - t_start

    print(f"\n{'=' * 60}")
    print(f"=== TTO Authoritative Evaluation Results ({n_samples} samples) ===")
    print(f"{'=' * 60}")
    print(f"  Average PoseNet Distortion: {avg_posenet:.8f}")
    print(f"  Average SegNet Distortion:  {avg_segnet:.8f}")
    print(f"  Archive size:               {archive_size:,} bytes")
    print(f"  GT size:                    {gt_size:,} bytes")
    print(f"  Compression Rate:           {rate:.8f}")
    print(f"  Score breakdown:")
    print(f"    100*seg  = {100 * avg_segnet:.4f}")
    print(f"    sqrt(10*pose) = {math.sqrt(10 * avg_posenet):.4f}")
    print(f"    25*rate  = {25 * rate:.4f}")
    print(f"  FINAL SCORE: {score:.4f}")
    print(f"  Total time: {t_total:.1f}s")
    print(f"{'=' * 60}")

    # ── 6. Save results ──
    result = {
        "tag": tag,
        "source": "tto_frames",
        "avg_posenet_dist": avg_posenet,
        "avg_segnet_dist": avg_segnet,
        "archive_size_bytes": archive_size,
        "gt_size_bytes": gt_size,
        "rate": rate,
        "score_seg": 100 * avg_segnet,
        "score_pose": math.sqrt(10 * avg_posenet),
        "score_rate": 25 * rate,
        "final_score": score,
        "n_samples": n_samples,
        "n_frames": NUM_FRAMES,
        "eval_method": f"upstream_evaluate_py_{eval_device}",
        "device": eval_device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": t_total,
    }

    result_path = os.path.join(tto_dir, "tto_auth_eval.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    # Clean up the .raw file (large, ~3.6GB)
    if os.path.exists(raw_path):
        os.remove(raw_path)
        print(f"  Cleaned up: {raw_path}")

    results_vol.commit()
    print("  [volume] Committed TTO auth eval results")

    return result


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,  # 1h — renderer inference + TTO optimization
    volumes={"/results": results_vol},
    memory=16384,
)
def tto_eval(
    tag: str,
    checkpoint: str = "renderer_best.pt",
    tto_steps: int = 500,
    tto_lr: float = 0.005,
    batch_pairs: int = 50,
):
    """Run renderer+TTO on Modal T4.

    Executes experiments/renderer_tto.py as a subprocess (already mounted)
    to avoid duplicating logic. Results saved to /results/<tag>/tto_results/.

    Args:
        tag:         experiment tag (volume subdirectory containing checkpoint)
        checkpoint:  checkpoint filename within /results/<tag>/
        tto_steps:   number of TTO optimization steps per batch
        tto_lr:      TTO learning rate
        batch_pairs: number of frame pairs per TTO batch
    """
    import os
    import subprocess
    import time

    vol_dir = f"/results/{tag}"
    ckpt_path = f"{vol_dir}/{checkpoint}"
    output_dir = f"{vol_dir}/tto_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Renderer + TTO | tag: {tag} ===")
    print(f"  GPU: T4")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  TTO steps: {tto_steps}, lr: {tto_lr}, batch_pairs: {batch_pairs}")
    print(f"  Output: {output_dir}")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(ckpt_path):
        available = os.listdir(vol_dir) if os.path.isdir(vol_dir) else []
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available: {[f for f in available if f.endswith('.pt')]}"
        )

    cmd = [
        "python", "/root/experiments/renderer_tto.py",
        "--checkpoint", ckpt_path,
        "--device", "cuda",
        "--n-frames", "1200",
        "--tto-steps", str(tto_steps),
        "--tto-lr", str(tto_lr),
        "--batch-pairs", str(batch_pairs),
        "--upstream", "/root/upstream",
        "--output-dir", output_dir,
        "--simulate-resize",
    ]

    env = {**os.environ, "PYTHONPATH": "/root/src:/root/upstream"}
    print(f"  Command: {' '.join(cmd)}")
    print("  ---")

    result = subprocess.run(cmd, env=env, capture_output=False)

    print("  ---")
    print(f"  End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Exit code: {result.returncode}")

    # Save stdout log (subprocess prints to console, but record exit status)
    log_path = os.path.join(output_dir, "tto_run.log")
    with open(log_path, "w") as f:
        f.write(f"exit_code: {result.returncode}\n")
        f.write(f"tag: {tag}\n")
        f.write(f"checkpoint: {checkpoint}\n")
        f.write(f"tto_steps: {tto_steps}\n")
        f.write(f"tto_lr: {tto_lr}\n")
        f.write(f"batch_pairs: {batch_pairs}\n")
        f.write(f"finished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    # Commit volume so results survive
    results_vol.commit()
    print(f"  [volume] Final commit done")

    # List output artifacts
    if os.path.isdir(output_dir):
        artifacts = sorted(os.listdir(output_dir))
        print(f"  TTO artifacts ({len(artifacts)}): {', '.join(artifacts[:15])}")

    # ── Auth eval on TTO frames ──────────────────────────────────────────────
    auth_result = None
    if result.returncode == 0:
        print("\n--- Running auth eval on TTO frames ---")
        try:
            auth_result = _run_tto_auth_eval(tag, output_dir)
        except Exception as exc:
            print(f"  TTO auth eval FAILED (non-fatal): {exc}")
            import traceback
            traceback.print_exc()
    else:
        print("  Skipping TTO auth eval (subprocess failed)")

    return {
        "exit_code": result.returncode,
        "output_dir": output_dir,
        "auth_eval": auth_result,
    }


@app.local_entrypoint()
def tto_entry(
    tag: str = "asym_v5_lagrangian_fixed",
    checkpoint: str = "renderer_best.pt",
    tto_steps: int = 500,
    tto_lr: float = 0.005,
    batch_pairs: int = 50,
):
    """Launch renderer+TTO on Modal T4.

    Usage:
        .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py::app.tto_entry
        .venv/bin/modal run src/tac/deploy/modal/modal_asymmetric_warp_deploy.py::app.tto_entry --tag my_run --tto-steps 1000
    """
    from tac.cost_tracker import print_cost_estimate

    print(f"=== Renderer + TTO -> Modal T4 ===")
    print(f"  Tag: {tag}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  TTO steps: {tto_steps}, lr: {tto_lr}, batch_pairs: {batch_pairs}")

    print("\n--- Cost Estimate ---")
    print_cost_estimate(gpu="t4", estimated_hours=1.0, platform="modal")

    print("\nLaunching TTO eval...")
    result = tto_eval.remote(
        tag=tag,
        checkpoint=checkpoint,
        tto_steps=tto_steps,
        tto_lr=tto_lr,
        batch_pairs=batch_pairs,
    )

    status = "OK" if result["exit_code"] == 0 else f"FAILED (exit {result['exit_code']})"
    print(f"\n  Result: {status}")
    print(f"  Output: {result['output_dir']}")

    # Display auth eval results if available
    auth = result.get("auth_eval")
    if auth:
        print(f"\n=== TTO Auth Eval Results ===")
        print(f"  Final Score: {auth['final_score']:.4f}")
        print(f"    SegNet:  {auth['score_seg']:.4f}")
        print(f"    PoseNet: {auth['score_pose']:.4f}")
        print(f"    Rate:    {auth['score_rate']:.4f}")
        print(f"  Time: {auth['elapsed_seconds']:.0f}s")
    elif result["exit_code"] == 0:
        print(f"\n  Auth eval: not available (tto_frames.pt may not have been produced)")

    print(f"\nDownload: .venv/bin/modal volume get {RESULTS_VOL} {tag}/tto_results/ ./tto_results_{tag}/")
