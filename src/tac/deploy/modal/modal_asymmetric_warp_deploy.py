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

APP_NAME = "tac-asymmetric-warp"
RESULTS_VOL = "tac-asymmetric-results"

# Where the training script hardcodes its output
SCRIPT_RESULTS_DIR = "/root/experiments/results/fridrich_renderer"

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
    })
    # add_local_dir must be LAST -- Modal mounts these at startup, not during build
    .add_local_dir(str(REPO_ROOT / "src" / "tac"), "/root/src/tac")
    .add_local_dir(str(REPO_ROOT / "experiments"), "/root/experiments")
)

results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)

# --- Council v2 training flags (asymmetric warp, 2026-04-13) ---
# Key changes from v1: rho_growth 1.02→1.005, rho_max 1e4→1e3, lambda_cap 1e6→1e4,
# phase1_end 0.40→0.25, phase2_end 0.70→0.85, batch_size 4→16,
# flow_warmup_epochs=500, residual_ramp_epochs=500
TRAINING_CMD_TEMPLATE: list[str] = [
    "python", "/root/experiments/train_renderer_fridrich.py",
    "--pair-mode", "asymmetric",
    "--epochs", "10000",
    "--batch-size", "16",
    "--lr", "2e-4",
    "--embed-dim", "6",
    "--base-ch", "36",
    "--mid-ch", "60",
    "--motion-hidden", "32",
    "--max-flow-px", "20.0",
    "--max-residual", "20.0",
    "--seg-boundary", "0.005",
    "--pose-boundary", "0.02",
    "--rho-init", "10.0",
    "--rho-growth", "1.005",
    "--rho-max", "1000",
    "--lambda-cap", "10000",
    "--phase1-end", "0.25",
    "--phase2-end", "0.85",
    "--flow-warmup-epochs", "500",
    "--residual-ramp-epochs", "500",
    "--tv-weight", "0.1",
    "--flow-weight", "0.0",
    "--rate-weight", "0.01",
    "--target-bytes", "256000",
    "--gate-reg-weight", "0.1",
    "--even-pairs-only",
    "--device", "cuda",
    "--seed", "42",
    "--checkpoint-every", "500",
    "--eval-every", "200",
    "--log-every", "25",
    "--max-hours", "5.5",
    "--phase2-mse-weight", "0.1",
]


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
def train_asymmetric_warp(tag: str, extra_args: list[str] | None = None):
    """Run asymmetric warp renderer training on T4."""
    import os
    import time

    vol_dir = f"/results/{tag}"
    os.makedirs(vol_dir, exist_ok=True)

    # Symlink hardcoded RESULTS_DIR -> volume path
    _setup_results_symlink(vol_dir)

    # Resume detection: check for existing checkpoint on volume
    checkpoint = _find_latest_checkpoint(vol_dir)

    print(f"=== tac asymmetric warp training | tag: {tag} ===")
    print(f"  GPU: T4 (council decision: iteration budget over speed)")
    print(f"  Wall-clock budget: 5.5h training / 6h timeout")
    print(f"  Resume: {'YES -> ' + checkpoint if checkpoint else 'NO (fresh start)'}")
    print(f"  Output: {vol_dir}")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    cmd = list(TRAINING_CMD_TEMPLATE)

    if checkpoint:
        cmd.extend(["--resume", checkpoint])

    if extra_args:
        cmd.extend(extra_args)

    env = {**os.environ, "PYTHONPATH": "/root/src:/root/upstream"}

    print(f"  Command: {' '.join(cmd)}")
    print("  ---")

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
    artifacts = os.listdir(vol_dir)
    print(f"  Artifacts ({len(artifacts)}): {', '.join(sorted(artifacts)[:10])}")
    if len(artifacts) > 10:
        print(f"    ... and {len(artifacts) - 10} more")

    return {"tag": tag, "exit_code": result.returncode, "artifacts": len(artifacts)}


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,  # 1h — inflation + scoring is fast
    volumes={"/results": results_vol},
    memory=16384,
)
def auth_eval(tag: str, checkpoint: str = "renderer_best.pt"):
    """Run authoritative evaluation of a checkpoint on the upstream scorer.

    Full pipeline:
        1. Load checkpoint from /results/<tag>/<checkpoint>
        2. Load upstream SegNet for mask extraction
        3. Decode GT video (upstream/videos/0.mkv via PyAV)
        4. Extract masks via SegNet
        5. Generate frames via renderer (asymmetric pair or independent)
        6. Upscale to 1164x874, write .raw
        7. Score via upstream DistortionNet (PoseNet + SegNet)
        8. Compute rate from checkpoint file size
        9. Final score: 100*seg + sqrt(10*pose) + 25*rate

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
    ckpt_path = os.path.join(vol_dir, checkpoint)

    print(f"=== Auth Eval | tag: {tag} | checkpoint: {checkpoint} ===")
    print(f"  GPU: T4")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(ckpt_path):
        # Try finding best checkpoint if the specified one doesn't exist
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
    batch_size = 16 if device == "cuda" else 4
    print(f"  Device: {device}")

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

    # Determine if this is a .pt training checkpoint or a .bin export
    if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
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
            depth=config.get("depth", 1),
            max_flow_px=config.get("max_flow_px", 20.0),
            max_residual=config.get("max_residual", 20.0),
            flow_only=config.get("flow_only", False),
        )
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.to(device).eval()

        # For rate calculation, use the export size (quantized .bin).
        # If a .bin exists alongside, use its size. Otherwise estimate.
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
            # Estimate: export the model to get actual size
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
                    f"No companion .bin found and export failed: {e}. "
                    f"Using .pt file size would give 5-10x wrong rate. "
                    f"Export a .bin first or pass --archive-size-bytes explicitly."
                )
    else:
        raise ValueError(
            f"Unsupported checkpoint format. Expected .pt training checkpoint "
            f"with 'model_state_dict' key. Got keys: "
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
    print(f"  Generated {n_written} frames ({time.monotonic() - t_gen:.1f}s)")
    del model, masks  # free VRAM

    # ── 6. Score via upstream DistortionNet ──
    print("\nStage 6: Scoring via upstream DistortionNet ...")
    t_score = time.monotonic()

    distortion_net = DistortionNet().eval().to(device)
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    # Load generated frames via TensorVideoDataset pattern
    from frame_utils import TensorVideoDataset, AVVideoDataset, camera_size, seq_len

    video_names = ["0.mkv"]

    from pathlib import Path as _Path

    # GT dataset — use AVVideoDataset (PyAV decode).
    # NOTE: The official scorer uses DALI on CUDA which produces slightly
    # different pixel values than PyAV (BT.601 limited-range rounding).
    # This gives us a consistent eval (PyAV for both GT and generated frames)
    # but scores may differ from the official leaderboard by a small amount.
    # To match exactly, install nvidia-dali and use DaliVideoDataset.
    ds_gt = AVVideoDataset(
        video_names,
        data_dir=_Path(UPSTREAM_ROOT) / "videos",
        batch_size=batch_size,
        device=torch.device("cpu"),  # AVVideoDataset requires non-cuda
    )
    ds_gt.prepare_data()
    dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)

    # Compressed dataset — use TensorVideoDataset (reads .raw)
    ds_comp = TensorVideoDataset(
        video_names,
        data_dir=_Path(vol_dir),  # will look for 0.raw
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    # Rename our inflated file to match expected name
    expected_raw = os.path.join(vol_dir, "0.raw")
    if raw_path != expected_raw:
        if os.path.exists(expected_raw):
            os.remove(expected_raw)
        os.rename(raw_path, expected_raw)
        raw_path = expected_raw
    ds_comp.prepare_data()
    dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

    posenet_dists = torch.zeros([], device=device)
    segnet_dists = torch.zeros([], device=device)
    batch_sizes = torch.zeros([], device=device)

    with torch.inference_mode():
        for (_, _, batch_gt), (_, _, batch_comp) in zip(dl_gt, dl_comp):
            batch_gt = batch_gt.to(device)
            batch_comp = batch_comp.to(device)
            posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
            posenet_dists += posenet_dist.sum()
            segnet_dists += segnet_dist.sum()
            batch_sizes += batch_gt.shape[0]

    avg_posenet = (posenet_dists / batch_sizes).item()
    avg_segnet = (segnet_dists / batch_sizes).item()
    n_samples = int(batch_sizes.item())

    # ── 7. Compute rate ──
    gt_size = os.path.getsize(gt_video_path)
    rate = archive_size / gt_size

    # ── 8. Final score ──
    score = 100 * avg_segnet + math.sqrt(10 * avg_posenet) + 25 * rate

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
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": t_total,
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
    checkpoint: str = "renderer_best.pt",
):
    """Launch authoritative evaluation on Modal T4.

    Args:
        tag: experiment tag (directory name on results volume)
        checkpoint: checkpoint filename within /results/<tag>/
    """
    from tac.cost_tracker import print_cost_estimate

    print(f"\n=== Auth Eval -> Modal T4 ===")
    print(f"  Tag: {tag}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Volume: {RESULTS_VOL}")

    print("\n--- Cost Estimate ---")
    print_cost_estimate(gpu="t4", estimated_hours=0.5, platform="modal")

    print("\nLaunching auth eval ...")
    result = auth_eval.remote(tag=tag, checkpoint=checkpoint)

    print(f"\n=== Auth Eval Complete ===")
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
):
    """Launch asymmetric warp training on Modal T4.

    Args:
        tag: experiment tag for output directory on results volume
        extra_args: space-separated extra CLI args (e.g., '--smoke')
    """
    from tac.cost_tracker import print_cost_estimate

    print(f"\n=== Asymmetric Warp Renderer -> Modal T4 ===")
    print(f"  Tag: {tag}")
    print(f"  Volume: {RESULTS_VOL}")

    print("\n--- Cost Estimate ---")
    print_cost_estimate(gpu="t4", estimated_hours=5.5, platform="modal")

    parsed_extra = extra_args.split() if extra_args.strip() else None

    print("\nLaunching training...")
    result = train_asymmetric_warp.remote(tag=tag, extra_args=parsed_extra)

    status = "OK" if result["exit_code"] == 0 else f"FAILED (exit {result['exit_code']})"
    print(f"\n  Result: {status}")
    print(f"  Artifacts: {result['artifacts']}")
    print(f"\nResults: .venv/bin/modal volume ls {RESULTS_VOL}")
    print(f"Download: .venv/bin/modal volume get {RESULTS_VOL} {tag}/ ./results_{tag}/")
