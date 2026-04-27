#!/usr/bin/env python3
"""Lane C: optimize a sparse, UNIWARD-weighted, L∞-bounded δ to add to
renderer output at inflate-time. Detector-informed embedding (Yousfi 2022)
on a contest where the detector IS the scorer.

Algorithm
---------
1. Load FROZEN renderer + masks (from archive masks.mkv) + poses (optimized
   FiLM conditioning vectors).
2. Render all 600 pairs deterministically. Capture rendered frames at the
   renderer's native resolution (384x512).
3. Extract GT pose targets from upstream/videos/0.mkv via PoseNet (cached).
4. Compute per-frame UNIWARD cost map (textured = high "embedding capacity").
5. Initialise δ = 0 with shape matching renderer output.
6. Optimize δ (Adam) to minimize the COMBINED scorer loss:
       L = pose_weight * MSE(PoseNet(rendered + δ), pose_targets)
         + seg_weight * hinge(SegNet(rendered + δ), gt_masks)
   subject to per-pixel L∞ ≤ l_inf_budget * (1 - cost_norm) + ε.
   The simulate_eval_roundtrip wrapper (CRITICAL — see CLAUDE.md
   non-negotiable rule on eval_roundtrip) is applied to BOTH the SegNet
   and PoseNet paths every step.
7. Pack the dense δ to sparse UWD1 wire format with target_bytes ≤ ~5KB.
8. Write delta.bin + provenance JSON.

Determinism (CLAUDE.md non-negotiable)
--------------------------------------
- CUDA-required (MPS produces 23x PoseNet drift per CLAUDE.md).
- ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` set BEFORE any cuBLAS call.
- ``torch.manual_seed`` + ``cuda.manual_seed_all`` + ``np.random.seed``.
- ``torch.use_deterministic_algorithms(True, warn_only=True)`` — warn-only
  because some Conv2d backwards on EfficientNet-B2 lack a deterministic
  kernel; we accept the bit-noise floor in those layers (it is well
  below the int8 quantization step).
- Sorted batch indices, fixed batch size — no shuffling.
- All random ops (warm-start jitter) seeded from the CLI ``--seed``.
- The same (renderer, masks, poses, seed, GPU model, torch+CUDA versions)
  → same delta.bin bytes. The provenance JSON records all five so a
  future rebuild can detect drift.

Cost estimate (Vast.ai 4090, end-to-end)
----------------------------------------
- Rendering 600 pairs: ~10 s
- GT pose extraction (cached): ~30 s first run, 0 s thereafter
- δ optimization @ 200 steps × 10 batches of 60 pairs: ~6 min
- Pack + write: ~2 s
- Wall clock: ~7 min total. At $0.25/hr → ~$0.03 per run.
- Add ~3 min Vast bootstrap → ~$0.05 amortized per experiment.

Usage
-----
    python experiments/optimize_uniward_delta.py \
        --renderer submissions/baseline_dilated_h64_0_90/renderer.bin \
        --masks    submissions/baseline_dilated_h64_0_90/masks.mkv \
        --poses    submissions/baseline_dilated_h64_0_90/optimized_poses.pt \
        --upstream upstream \
        --output-dir experiments/results/uniward_delta_v1

The default ``--target-bytes 5000`` gives the council's predicted Lane C
operating point (Quantizr-killing zone when stacked with Lane A pose-TTO).
"""
from __future__ import annotations

# Line-buffer stdout for live progress (matches optimize_poses.py pattern).
import sys as _dx_sys
try:
    _dx_sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    _dx_sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path setup (must run before tac imports — mirrors optimize_poses.py).
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
_CANDIDATE_UPSTREAM = [
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
    REPO / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))
sys.path.insert(0, str(REPO / "src"))


SEGNET_INPUT_H, SEGNET_INPUT_W = 384, 512
NUM_FRAMES = 1200
NUM_PAIRS = NUM_FRAMES // 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lane C: UNIWARD δ-injection on rendered frames at compress-time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renderer", type=Path, required=True,
                   help="Path to renderer .bin (or .pt). Must match the renderer "
                        "shipped in archive.zip — δ depends on its pixel-level output.")
    p.add_argument("--masks", type=Path, required=True,
                   help="Path to masks.mkv (or .amrc/.pt). Must match archive masks "
                        "exactly — δ depends on the renderer's mask-driven output.")
    p.add_argument("--poses", type=Path, required=True,
                   help="Path to optimized_poses.pt (or .bin). Must match archive "
                        "poses exactly — δ depends on FiLM conditioning.")
    p.add_argument("--upstream", type=Path, default=None,
                   help="Path to upstream repo (auto-detected if None).")
    p.add_argument("--gt-video", type=Path, default=None,
                   help="GT video for pose-target extraction. Defaults to "
                        "upstream/videos/0.mkv.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory. delta.bin + provenance.json land here.")
    # ── Optimization knobs ────────────────────────────────────────────
    p.add_argument("--steps", type=int, default=200,
                   help="Adam steps per pair-batch.")
    p.add_argument("--lr", type=float, default=0.5,
                   help="Adam LR for δ. Higher than pose-TTO LR because δ is "
                        "bounded; the L∞ clip is the regulariser.")
    p.add_argument("--batch-pairs", type=int, default=20,
                   help="Pairs per optimization batch. 4090 24GB VRAM cap "
                        "(EfficientNet-B2 backward grad-graph dominates).")
    p.add_argument("--seg-weight", type=float, default=100.0,
                   help="SegNet hinge loss weight (mirrors scoring formula).")
    p.add_argument("--pose-weight", type=float, default=10.0,
                   help="PoseNet MSE loss weight (mirrors scoring formula).")
    p.add_argument("--hinge-margin", type=float, default=0.5,
                   help="Margin for SegNet hinge loss.")
    # ── δ knobs ──────────────────────────────────────────────────────
    p.add_argument("--l-inf-budget", type=float, default=4.0,
                   help="Per-pixel L∞ cap on |δ| in [0, 255] units. Council "
                        "default 4 ≈ 4/255 = sub-(256,192) blind to SegNet's "
                        "stride-2 stem (Fridrich principle 1).")
    p.add_argument("--target-bytes", type=int, default=5000,
                   help="Hard cap on the compressed delta.bin size. "
                        "Council Lane-C target ≤5KB (~+0.003 rate cost). "
                        "DO NOT raise without re-doing the rate-vs-distortion "
                        "tradeoff math — a 100KB delta wipes the score win. "
                        "M1 footgun guard: the underlying library defaults to "
                        "~1% sparsity (~100KB) when target_bytes is unset; the "
                        "CLI default of 5000 keeps us in the council range.")
    p.add_argument("--uniward-sigma", type=float, default=1e-4,
                   help="UNIWARD wavelet stabilizer (matches fridrich default).")
    # ── Determinism + provenance ─────────────────────────────────────
    p.add_argument("--seed", type=int, default=1234,
                   help="Master seed (matches upstream/evaluate.py default).")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="cuda is the only contest-faithful target. cpu is for "
                        "smoke tests only — δ bytes will NOT match a CUDA run.")
    # ── Roundtrip (CLAUDE.md non-negotiable) ─────────────────────────
    # M2 fix: the CLAUDE.md rule "EVERY training path MUST use eval_roundtrip"
    # is a hard non-negotiable. The previous --no-eval-roundtrip flag let any
    # caller silently violate it. The escape hatch is now an environment
    # variable (TAC_ALLOW_NO_ROUNDTRIP=1) so it cannot land in a tmux command
    # by accident or sit unnoticed in a profile dict; it leaves a clear
    # forensic trail in the run log via the loud banner.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate contest eval resolution roundtrip in loss. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")
    p.add_argument("--posetto-noise-std", type=float, default=0.5,
                   help="Hotz STE noise std applied DURING simulate_eval_roundtrip. "
                        "0.5 closes the proxy-CUDA gap on PoseNet (CLAUDE.md).")
    # ── Frame range (smoke testing) ──────────────────────────────────
    p.add_argument("--n-frames", type=int, default=NUM_FRAMES,
                   help="Number of frames to render and δ-optimize. Default "
                        "is the full 1200 (600 pairs).")
    # ── Compliance gate (Codex R5 HIGH + R5-2 #4) ────────────────────
    # Lane C δ.bin is scorer-derived; its contest compliance is unresolved
    # until the council rules on Yousfi PR #35. The optimizer MUST mark
    # every newly-built δ as PENDING_RULING so the archive builder gates
    # on it.
    #
    # CODEX R5-2 #4 fix (2026-04-27): the optimizer can NO LONGER issue
    # ``approved`` directly. The previous CLI accepted
    # ``--compliance-status approved`` and the archive builder trusted
    # that header value, which means an operator could bypass the gate
    # by self-asserting approval at δ-build time. The trust model was
    # operator-self-asserted rather than externally-attested.
    #
    # New behavior:
    #   - The optimizer accepts only ``pending_ruling`` (default, safe)
    #     and ``rejected`` (terminal, never bundle).
    #   - ``approved`` requires a SIGNED ATTESTATION FILE produced by
    #     ``tools/sign_lane_c_compliance.py``, written to the canonical
    #     path ``.omx/state/lane_c_compliance_attestations/<sha256>.json``
    #     after the actual delta.bin is built. The archive builder
    #     cross-checks the attestation's delta_sha256 against the bundled
    #     blob's sha256.
    p.add_argument("--compliance-status", type=str, default="pending_ruling",
                   choices=["pending_ruling", "rejected"],
                   help="compliance_status to write into the δ.bin header. "
                        "Default 'pending_ruling' is correct until the "
                        "council ruling on Yousfi PR #35 is recorded in "
                        ".omx/research/findings.md. 'rejected' is reserved "
                        "for forensic re-archival of failed experiments — "
                        "the archive builder refuses to bundle them. "
                        "'approved' is NOT accepted here; it requires an "
                        "external attestation produced by "
                        "tools/sign_lane_c_compliance.py against the built "
                        "delta.bin's sha256. See Codex R5-2 #4 fix.")
    return p.parse_args()


def _set_determinism(seed: int) -> None:
    """CLAUDE.md non-negotiable determinism setup. Must be called BEFORE any
    cuBLAS / cuDNN call. Matches build_baseline_archive.py and
    optimize_poses.py conventions.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only=True: EfficientNet-B2 has some Conv2d backward kernels that
    # lack deterministic CUDA implementations (cudnn.deterministic = False).
    # We accept this — the bit-noise floor in those backward passes is well
    # below our int8 quantization step (1 / 127 of the L∞ budget).
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_masks(masks_path: Path) -> torch.Tensor:
    """Load mask tensor from .pt OR .mkv (AV1 5-class). Mirrors the loader
    in experiments/optimize_poses.py — keep them in sync if you change one.
    Returns (N, H, W) long tensor. Upsamples to (384, 512) if smaller.
    """
    if masks_path.suffix == ".pt":
        gt_masks = torch.load(str(masks_path), weights_only=True).long()
    elif masks_path.suffix in (".mkv", ".mp4"):
        import subprocess
        cmd = ["ffmpeg", "-v", "quiet", "-i", str(masks_path),
               "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(masks_path)],
            capture_output=True, text=True, check=True,
        )
        w, h = map(int, probe.stdout.strip().split(","))
        pixels = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w)
        scale = 255 // 4  # 5-class palette: 0, 64, 128, 191, 255 → /63 ≈ /63
        masks_np = np.clip(np.round(pixels.astype(np.float32) / scale).astype(np.int64), 0, 4)
        gt_masks = torch.from_numpy(masks_np)
    else:
        raise ValueError(f"Unknown mask format: {masks_path.suffix}")
    if gt_masks.shape[1] < SEGNET_INPUT_H or gt_masks.shape[2] < SEGNET_INPUT_W:
        gt_masks = F.interpolate(
            gt_masks.float().unsqueeze(1),
            size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="nearest",
        ).squeeze(1).long()
    return gt_masks


def _segnet_hinge_loss(
    logits: torch.Tensor, gt_masks: torch.Tensor, *, margin: float = 0.5,
) -> torch.Tensor:
    """SegNet hinge loss — copy of optimize_poses.segnet_hinge_loss to keep
    this script self-contained. Penalises pixels at risk of argmax flip.
    """
    correct = logits.gather(1, gt_masks.unsqueeze(1)).squeeze(1)
    mask_inf = torch.zeros_like(logits)
    mask_inf.scatter_(1, gt_masks.unsqueeze(1), float("-inf"))
    runner_up = (logits + mask_inf).max(dim=1).values
    return F.relu(margin - (correct - runner_up)).mean()


def main() -> int:
    args = parse_args()

    # ─── M2 fix: enforce eval_roundtrip non-negotiable ───────────────
    # CLAUDE.md: EVERY training path MUST use eval_roundtrip. The only escape
    # hatch is the TAC_ALLOW_NO_ROUNDTRIP=1 environment variable, which
    # leaves a clear audit trail.
    if not args.eval_roundtrip:
        if os.environ.get("TAC_ALLOW_NO_ROUNDTRIP") != "1":
            raise SystemExit(
                "FATAL: eval_roundtrip is False but TAC_ALLOW_NO_ROUNDTRIP=1 "
                "is not set. The CLAUDE.md non-negotiable rule "
                "'EVERY training path MUST use eval_roundtrip' applies to "
                "Lane C. Without it, the proxy-auth gap is 2-11x on PoseNet "
                "and the run is a wasted Vast.ai dollar. Set the env var "
                "explicitly if doing a diagnostic ablation."
            )
        print(
            "\n" + "!" * 78 + "\n"
            "DANGER: eval_roundtrip is DISABLED via TAC_ALLOW_NO_ROUNDTRIP=1.\n"
            "  Proxy-auth gap will be 2-11x on PoseNet. This run is for\n"
            "  ablation/diagnostic ONLY — DO NOT report any score from it as\n"
            "  contest-compliant. Tag results [no-roundtrip-ablation].\n"
            + "!" * 78 + "\n",
            flush=True,
        )

    # ─── Determinism FIRST (before any torch/cuBLAS call) ────────────
    _set_determinism(args.seed)

    # ─── C4 fix: compliance warning banner ───────────────────────────
    # δ.bin is derived from PoseNet/SegNet gradients at compress time. Per
    # Yousfi PR #35 strict-scorer-rule, scorer-derived artefacts in the
    # archive may not be contest-compliant. Until the council issues a
    # binding ruling, every score from this pipeline must be tagged
    # [lane-c-pending-ruling] in the run log / report / commit.
    print(
        "\n" + "=" * 78 + "\n"
        "WARNING: Lane C δ contest-compliance is PENDING council ruling.\n"
        "  δ.bin is a SCORER-DERIVED artifact (compress-time PoseNet+SegNet\n"
        "  gradients). Yousfi PR #35 strict-scorer-rule may class this as\n"
        "  non-compliant. Tag any score downstream as [lane-c-pending-ruling]\n"
        "  until the council ruling is recorded in .omx/research/findings.md.\n"
        "  DO NOT submit a contest PR using this δ until the ruling is in.\n"
        + "=" * 78 + "\n",
        flush=True,
    )

    # ─── Resolve paths ───────────────────────────────────────────────
    upstream = args.upstream or UPSTREAM_ROOT
    if upstream is None or not (upstream / "modules.py").exists():
        raise SystemExit(
            "FATAL: --upstream not set and auto-detection failed. Pass "
            "--upstream <path-to-upstream-repo>."
        )
    upstream = Path(upstream)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    gt_video = args.gt_video or (upstream / "videos" / "0.mkv")
    for label, path in (("renderer", args.renderer), ("masks", args.masks),
                        ("poses", args.poses), ("gt-video", gt_video)):
        if not path.exists():
            raise SystemExit(f"--{label} does not exist: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Device check (CUDA-required per CLAUDE.md) ──────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda but CUDA not available. Lane C MUST run on "
            "CUDA — MPS produces 23x PoseNet drift (CLAUDE.md non-negotiable). "
            "Pass --device cpu only for code-correctness smoke tests; the "
            "resulting delta.bn bytes will NOT match a CUDA-built one."
        )
    device = torch.device(args.device)
    print(f"[config] device={device}, seed={args.seed}", flush=True)
    print(f"[config] renderer={args.renderer}", flush=True)
    print(f"[config] masks={args.masks}, poses={args.poses}", flush=True)
    print(f"[config] l_inf_budget={args.l_inf_budget}, target_bytes={args.target_bytes}", flush=True)
    # M1 banner: target_bytes is THE rate-cost knob. Surface it loudly so
    # operators don't accidentally ship a 100KB blob.
    if args.target_bytes > 10_000:
        print(
            f"  WARNING: target_bytes={args.target_bytes:,} exceeds the council "
            f"Lane-C ceiling of 10,000 B. Expect a meaningful rate-term hit.",
            flush=True,
        )
    elif args.target_bytes > 5_000:
        print(
            f"  NOTE: target_bytes={args.target_bytes:,} above council target "
            f"5,000 B. Re-run rate/distortion tradeoff before shipping.",
            flush=True,
        )
    print(f"[config] steps={args.steps}, lr={args.lr}, batch_pairs={args.batch_pairs}", flush=True)
    print(f"[config] eval_roundtrip={args.eval_roundtrip}, posetto_noise_std={args.posetto_noise_std}", flush=True)

    t_total = time.monotonic()

    # ─── Step 1: Load scorers (differentiable) ───────────────────────
    print("\n[1/7] Loading differentiable scorers...", flush=True)
    t0 = time.monotonic()
    from tac.scorer import load_differentiable_scorers, extract_gt_pose_targets, extract_gt_masks
    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    print(f"[1/7] Scorers loaded in {time.monotonic() - t0:.1f}s", flush=True)

    # ─── Step 2: Load renderer ───────────────────────────────────────
    print("\n[2/7] Loading renderer...", flush=True)
    t0 = time.monotonic()
    # Re-use the canonical renderer loader from optimize_poses (handles
    # ASYM, FP4A, .pt with config). This keeps Lane C consistent with
    # Lane A so we never have a second loader that diverges.
    sys.path.insert(0, str(REPO / "experiments"))
    from optimize_poses import load_renderer  # type: ignore[import-not-found]
    renderer = load_renderer(str(args.renderer), device).eval()
    for p_param in renderer.parameters():
        p_param.requires_grad = False
    renderer_pose_dim = getattr(renderer, "pose_dim", 0)
    print(f"[2/7] Renderer loaded in {time.monotonic() - t0:.1f}s "
          f"(pose_dim={renderer_pose_dim})", flush=True)

    # ─── Step 3: Load masks + poses ──────────────────────────────────
    print("\n[3/7] Loading archive masks + poses...", flush=True)
    t0 = time.monotonic()
    masks = _load_masks(args.masks)[: args.n_frames]
    n_frames_actual = masks.shape[0]
    n_pairs = n_frames_actual // 2
    print(f"  masks: {tuple(masks.shape)}", flush=True)

    from tac.submission_archive import load_optimized_poses
    poses = load_optimized_poses(args.poses, pose_dim=max(renderer_pose_dim, 1))
    poses = poses[:n_pairs]
    print(f"  poses: {tuple(poses.shape)}", flush=True)
    print(f"[3/7] Loaded archive inputs in {time.monotonic() - t0:.1f}s", flush=True)

    # ─── Step 4: GT video → pose targets (cached) ────────────────────
    print("\n[4/7] Decoding GT video + extracting pose targets...", flush=True)
    t0 = time.monotonic()
    from tac.data import load_gt_video
    gt_frames = load_gt_video(str(gt_video), n_frames=n_frames_actual)
    cache_path = args.output_dir / "gt_pose_targets.pt"
    if cache_path.exists():
        pose_targets = torch.load(cache_path, weights_only=True).float()
        print(f"  Loaded cached pose targets: {tuple(pose_targets.shape)}", flush=True)
    else:
        pose_targets = extract_gt_pose_targets(gt_frames, posenet, device)[:n_pairs]
        torch.save(pose_targets, cache_path)
        print(f"  Computed + cached pose targets: {tuple(pose_targets.shape)}", flush=True)
    gt_segnet_masks = extract_gt_masks(gt_frames, segnet, device)[:n_frames_actual]
    print(f"  GT segnet masks: {tuple(gt_segnet_masks.shape)}", flush=True)
    print(f"[4/7] Targets ready in {time.monotonic() - t0:.1f}s", flush=True)

    # ─── Step 5: Allocate δ. Stored on CPU; per-batch slice → device. ─
    # Shape (n_frames, 3, H, W) at the renderer's native resolution.
    H, W = SEGNET_INPUT_H, SEGNET_INPUT_W
    print(f"\n[5/7] Allocating δ ({n_frames_actual}, 3, {H}, {W}) on CPU...", flush=True)
    delta_full = torch.zeros(n_frames_actual, 3, H, W, dtype=torch.float32)
    cost_map_full = torch.zeros(n_frames_actual, H, W, dtype=torch.float32)
    print(f"  δ allocated: {delta_full.numel() * 4 / 2**20:.1f} MB", flush=True)

    # ─── Step 6: Optimize δ batch-by-batch ───────────────────────────
    print(f"\n[6/7] Optimizing δ in {math.ceil(n_pairs / args.batch_pairs)} batches "
          f"of {args.batch_pairs} pairs × {args.steps} steps...", flush=True)
    from tac.renderer import simulate_eval_roundtrip
    from tac.uniward_delta import compute_uniward_cost_map

    n_batches = math.ceil(n_pairs / args.batch_pairs)
    metrics = {"per_batch": []}
    t_opt = time.monotonic()

    for batch_idx in range(n_batches):
        ps = batch_idx * args.batch_pairs
        pe = min(ps + args.batch_pairs, n_pairs)
        fs, fe = 2 * ps, 2 * pe
        n_bp = pe - ps

        masks_t = masks[fs:fe:2].to(device, dtype=torch.long)
        masks_t1 = masks[fs + 1:fe + 1:2].to(device, dtype=torch.long)
        gt_pair_masks = gt_segnet_masks[fs:fe].to(device, dtype=torch.long)
        batch_pose_targets = pose_targets[ps:pe].to(device)
        batch_film = poses[ps:pe].to(device) if renderer_pose_dim > 0 else None

        # Render once; δ is added on top each step (renderer is FROZEN so
        # the rendered base does not move).
        fwd_kwargs: dict = {}
        if batch_film is not None:
            fwd_kwargs["pose"] = batch_film
        with torch.inference_mode():
            base_pairs = renderer(masks_t, masks_t1, **fwd_kwargs)  # (B, 2, H, W, 3)
        # Flatten to per-frame (2*B, H, W, 3) and convert to CHW
        base_hwc = torch.cat([base_pairs[:, 0], base_pairs[:, 1]], dim=0).contiguous()
        base_chw = base_hwc.permute(0, 3, 1, 2).contiguous().float()  # (2*B, 3, H, W)
        # We need .clone().requires_grad_(True) for δ; base is frozen.
        base_chw = base_chw.detach()

        # Compute UNIWARD cost on the rendered base (not GT) — this aligns
        # the embedding capacity with the actual frames we are perturbing.
        with torch.no_grad():
            cost = compute_uniward_cost_map(base_chw, sigma=args.uniward_sigma)
        # Stash for the pack step.
        # CRITICAL (C1 fix): base_chw is concatenated as
        #   [p0_t, p1_t, ..., p(N-1)_t, p0_t+1, p1_t+1, ..., p(N-1)_t+1]
        # but the inflate path expects the canonical pair-interleaved layout
        #   [p0_t, p0_t+1, p1_t, p1_t+1, ..., p(N-1)_t, p(N-1)_t+1].
        # Writing `cost_map_full[fs:fe] = cost` directly would put p1_t into
        # the global slot for p0_t+1 — silent data corruption that destroys
        # the score during inflate. Interleave the two halves.
        cost_cpu = cost.cpu()
        cost_map_full[fs:fe:2] = cost_cpu[:n_bp]      # t-frames → even slots
        cost_map_full[fs + 1:fe:2] = cost_cpu[n_bp:]  # t+1-frames → odd slots

        # ── Init δ for this batch (zeros) ────────────────────────────
        delta = torch.zeros_like(base_chw, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=args.lr)

        initial_loss = None
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)

            # Apply δ + L∞ clip via a soft tanh (differentiable). The hard
            # clip happens at pack time; here we use tanh to keep gradients
            # flowing near the boundary.
            perturbed = base_chw + args.l_inf_budget * torch.tanh(delta / args.l_inf_budget)

            # Eval roundtrip (CLAUDE.md non-negotiable).
            if args.eval_roundtrip:
                perturbed_round = simulate_eval_roundtrip(
                    perturbed, noise_std=args.posetto_noise_std,
                )
            else:
                perturbed_round = perturbed

            # SegNet path
            # CRITICAL (Codex R5 HIGH fix — SegNet wrong-frame-order):
            #   ``perturbed`` (and therefore ``perturbed_round``) is in BATCHED
            #   layout [p0_t, p1_t, ..., p(n-1)_t, p0_t+1, p1_t+1, ..., p(n-1)_t+1]
            #   because it inherits ``base_chw`` which was built via
            #   ``cat([base_pairs[:, 0], base_pairs[:, 1]], dim=0)``.
            #   But ``gt_pair_masks = gt_segnet_masks[fs:fe]`` slices the global
            #   per-frame mask tensor in PAIR-INTERLEAVED layout
            #   [p0_t, p0_t+1, p1_t, p1_t+1, ...]. For batch_pairs > 1 this
            #   means SegNet logits at row 1 (which is p1_t) get compared against
            #   gt_pair_masks[1] (which is p0_t+1) — the gradient is computed
            #   against the WRONG frame's labels, silently corrupting the SegNet
            #   loss for every batch-of-2-or-more.
            #
            #   Fix: interleave perturbed_round into pair order BEFORE the SegNet
            #   forward. Reshape (2*B, 3, H, W) [t-half, t+1-half] →
            #   (2, B, 3, H, W) → transpose to (B, 2, 3, H, W) → flatten to
            #   (2*B, 3, H, W) in [p0_t, p0_t+1, p1_t, p1_t+1, ...] order.
            #   Gradients flow back through the views into the original
            #   ``delta`` tensor unchanged — no impact on the C1 writeback,
            #   which still maps the BATCHED ``delta`` to global frames.
            #
            #   For batch_pairs == 1 the interleave is a no-op (the two layouts
            #   coincide), so existing single-pair behaviour is preserved bit-
            #   exactly. The new test_codex_r5_segnet_pair_order_with_batch2
            #   regression locks this against silent re-introduction.
            n_half = perturbed_round.shape[0] // 2  # == n_bp
            perturbed_round_pair_order = (
                perturbed_round
                .reshape(2, n_half, *perturbed_round.shape[1:])
                .transpose(0, 1)
                .reshape(perturbed_round.shape)
                .contiguous()
            )
            seg_in = segnet.preprocess_input(perturbed_round_pair_order.unsqueeze(1))
            seg_logits = segnet(seg_in)
            # CRITICAL (C2 fix): if the SegNet upstream resize differs from the
            # mask resolution we extracted, the hinge loss silently broadcasts
            # over wrong spatial dims and the optimizer gets nonsense
            # gradients. Hard-fail here so the bug surfaces immediately.
            assert seg_logits.shape[-2:] == gt_pair_masks.shape[-2:], (
                f"SegNet logits/GT-mask spatial mismatch: "
                f"logits={tuple(seg_logits.shape)} vs gt={tuple(gt_pair_masks.shape)}. "
                f"Both must be (..., {SEGNET_INPUT_H}, {SEGNET_INPUT_W})."
            )
            # CRITICAL (Codex R5 HIGH fix, continued): assert leading-dim parity
            # so any future renderer or roundtrip change that desynchronises
            # SegNet inputs vs labels surfaces here, not as a silent score
            # regression in CUDA auth eval.
            assert seg_logits.shape[0] == gt_pair_masks.shape[0], (
                f"SegNet logits/GT-mask batch mismatch: "
                f"logits.batch={seg_logits.shape[0]} vs gt.batch={gt_pair_masks.shape[0]}. "
                f"perturbed_round_pair_order must contain 2*batch_pairs frames "
                f"in pair-interleaved order matching gt_segnet_masks[fs:fe]."
            )
            seg_loss = _segnet_hinge_loss(
                seg_logits, gt_pair_masks, margin=args.hinge_margin,
            )

            # PoseNet path — needs (B, 2, 3, H, W) pair layout.
            # Re-pair the perturbed frames; first half is t, second half is t+1
            ft = perturbed[:n_bp]
            ft1 = perturbed[n_bp:]
            pairs_chw = torch.stack([ft, ft1], dim=1)  # (B, 2, 3, H, W)
            if args.eval_roundtrip:
                Bp, Tp, Cp, Hp, Wp = pairs_chw.shape
                flat = pairs_chw.reshape(Bp * Tp, Cp, Hp, Wp)
                flat = simulate_eval_roundtrip(flat, noise_std=args.posetto_noise_std)
                pairs_chw = flat.reshape(Bp, Tp, Cp, Hp, Wp)
            pose_in = posenet.preprocess_input(pairs_chw)
            pose_out = posenet(pose_in)["pose"][..., :6]
            pose_loss = F.mse_loss(pose_out, batch_pose_targets[:pose_out.shape[0]])

            total = args.seg_weight * seg_loss + args.pose_weight * pose_loss
            if initial_loss is None:
                initial_loss = float(total.item())
            total.backward()
            optimizer.step()

            if step == args.steps - 1 or (step + 1) % max(args.steps // 4, 1) == 0:
                print(f"    [batch {batch_idx + 1}/{n_batches} step {step + 1}/{args.steps}] "
                      f"total={total.item():.6f}, seg={seg_loss.item():.6f}, "
                      f"pose={pose_loss.item():.6f}", flush=True)

        # Materialize the final L∞-clipped δ.
        with torch.no_grad():
            delta_final = args.l_inf_budget * torch.tanh(delta / args.l_inf_budget)
            # CRITICAL (C1 fix): delta_final is in the [t-frames..., t+1-frames...]
            # layout (matching base_chw). The inflate path reads frames in
            # the pair-interleaved layout [p0_t, p0_t+1, p1_t, p1_t+1, ...].
            # Interleave when writing to global frame indices, otherwise δ
            # for pair 1's t-frame lands on pair 0's t+1-frame at inflate.
            delta_final_cpu = delta_final.detach().cpu()
            delta_full[fs:fe:2] = delta_final_cpu[:n_bp]      # t-frames
            delta_full[fs + 1:fe:2] = delta_final_cpu[n_bp:]  # t+1-frames

        metrics["per_batch"].append({
            "batch_idx": batch_idx,
            "pair_range": [ps, pe],
            "initial_loss": initial_loss,
            "final_loss": float(total.item()),
            "improvement_pct": (
                100.0 * (initial_loss - float(total.item())) / max(initial_loss, 1e-12)
                if initial_loss is not None else 0.0
            ),
            "delta_l_inf_max": float(delta_final.detach().abs().max().item()),
            "delta_nonzero_frac": float(
                (delta_final.detach().abs() > 1e-6).float().mean().item()
            ),
        })

        del base_pairs, base_hwc, base_chw, perturbed, perturbed_round
        del perturbed_round_pair_order
        del seg_in, seg_logits, pairs_chw, pose_in, pose_out, delta_final
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[6/7] δ optimization done in {time.monotonic() - t_opt:.1f}s "
          f"({(time.monotonic() - t_opt) / max(n_batches, 1):.1f}s per batch)",
          flush=True)

    # ─── Step 7: Pack + write δ ──────────────────────────────────────
    print(f"\n[7/7] Packing δ to UWD1 wire format (target ≤ {args.target_bytes} B)...",
          flush=True)
    t0 = time.monotonic()
    from tac.uniward_delta import (
        pack_sparse_delta, unpack_sparse_delta, fingerprint_inputs,
    )
    provenance_inputs = {
        "renderer_sha256": fingerprint_inputs(args.renderer),
        "masks_sha256": fingerprint_inputs(args.masks),
        "poses_sha256": fingerprint_inputs(args.poses),
        "n_pairs": int(n_pairs),
        "n_frames": int(n_frames_actual),
        "tool": "experiments/optimize_uniward_delta.py",
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_model": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    blob = pack_sparse_delta(
        delta_full, cost_map_full,
        l_inf_budget=args.l_inf_budget,
        target_bytes=args.target_bytes,
        seed=args.seed,
        extra_provenance=provenance_inputs,
        # Codex R5 HIGH: propagate compliance gate into wire format so
        # downstream archive build can fail closed without an explicit
        # operator override.
        compliance_status=args.compliance_status,
    )
    delta_path = args.output_dir / "delta.bin"
    delta_path.write_bytes(blob)
    print(f"  Wrote {delta_path}: {len(blob):,} bytes", flush=True)

    # Verify round-trip (sanity check before shipping).
    spec = unpack_sparse_delta(blob)
    print(f"  Round-trip OK: n_kept={spec.n_kept}, "
          f"any_delta={spec.any_delta}, scale={spec.scale:.4f}", flush=True)

    # Provenance JSON (separate from the wire-format header so external
    # tools can `jq` it without zlib-decompressing).
    prov = {
        "schema_version": 1,
        "tool": "experiments/optimize_uniward_delta.py",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_model": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "inputs": {
            "renderer": str(args.renderer),
            "masks": str(args.masks),
            "poses": str(args.poses),
            "gt_video": str(gt_video),
        },
        "input_sha256": provenance_inputs,
        "delta_bytes": len(blob),
        "delta_sha256": __import__("hashlib").sha256(blob).hexdigest(),
        "n_kept": spec.n_kept,
        "n_total_pixel_channels": int(n_frames_actual * H * W * 3),
        "sparsity_pct": (
            100.0 * spec.n_kept / max(n_frames_actual * H * W * 3, 1)
        ),
        "l_inf_budget": args.l_inf_budget,
        "target_bytes": args.target_bytes,
        "args": {
            "steps": args.steps,
            "lr": args.lr,
            "batch_pairs": args.batch_pairs,
            "seg_weight": args.seg_weight,
            "pose_weight": args.pose_weight,
            "hinge_margin": args.hinge_margin,
            "uniward_sigma": args.uniward_sigma,
            "eval_roundtrip": args.eval_roundtrip,
            "posetto_noise_std": args.posetto_noise_std,
            "n_frames": args.n_frames,
            "compliance_status": args.compliance_status,
        },
        "compliance_status": args.compliance_status,
        "metrics": metrics,
    }
    prov_path = args.output_dir / "delta.provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"  Wrote {prov_path}", flush=True)
    print(f"[7/7] Pack done in {time.monotonic() - t0:.1f}s", flush=True)

    print(f"\n=== Lane C complete: {time.monotonic() - t_total:.1f}s wall ===", flush=True)
    print(f"  delta.bin: {len(blob):,} bytes "
          f"(target ≤ {args.target_bytes:,})", flush=True)
    print(f"  sparsity:  {prov['sparsity_pct']:.4f}% "
          f"({spec.n_kept:,} / {prov['n_total_pixel_channels']:,} pixel-channels)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
