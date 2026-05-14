# SPDX-License-Identifier: MIT
"""Lane OS-A standalone: produce seed_poses.pt from openpilot supercombo.

Usage::

    python experiments/seed_poses_from_openpilot.py \\
        --supercombo-path /workspace/openpilot/models/supercombo.onnx \\
        --video upstream/videos/0.mkv \\
        --output seed_poses.pt \\
        --device cuda

The output is consumed by ``experiments/optimize_poses.py --seed-poses-path``.

V2 changes (2026-04-27):

* ``--scale-to-match-mode {none,linear,mlp}`` replaces ``--no-scale-to-match``
  (V1 flag is kept as a backward-compat alias for ``--scale-to-match-mode none``).
* ``--fallback-mode {baseline,lane_mark}`` chooses what to do when supercombo
  is unavailable. Default is ``baseline`` — load the canonical baseline poses
  (much higher correlation with PoseNet than the lane-mark fallback).
* ``--fov-crop / --no-fov-crop`` controls the road-centric crop before resize.
* ``--legacy-v1-yuv`` flag for byte-equivalent reproduction of V1 seed_poses.
* ``--no-features-buffer`` flag to disable the recurrent state propagation
  (zero-fill features_buffer every frame, V1 behavior).

Strict-scorer-rule: this tool runs at COMPRESS TIME only. The supercombo
model (~30 MB) is never bundled into the archive; only the resulting
``seed_poses.pt`` ((600, 6) fp16, ~7 KB) is consumed by the pipeline.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# Allow running as a script: src/ on path.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.openpilot_seeding import (  # noqa: E402
    OPENPILOT_SUPERCOMBO_DEFAULT_PATH,
    OPENPILOT_SUPERCOMBO_URL,
    SUPERCOMBO_VERSION_PIN,
    SupercomboUnavailable,
    fallback_seed_from_baseline,
    fallback_seed_from_masks,
    infer_pose_from_video,
    load_supercombo_model,
    seed_pose_tto,
)

DEFAULT_BASELINE_POSES = (
    "submissions/baseline_dilated_h64_0_90/optimized_poses.pt"
)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            f"Lane OS-A V2: produce seed_poses.pt from openpilot supercombo "
            f"(pinned: openpilot {SUPERCOMBO_VERSION_PIN})"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--supercombo-path", type=str,
        default=OPENPILOT_SUPERCOMBO_DEFAULT_PATH,
        help=(
            f"Path to supercombo.onnx (~30 MB). Download from "
            f"{OPENPILOT_SUPERCOMBO_URL}"
        ),
    )
    p.add_argument(
        "--video", type=str, default="upstream/videos/0.mkv",
        help="GT video path",
    )
    p.add_argument(
        "--output", type=str, default="seed_poses.pt",
        help="Output (N_pairs, 6) tensor path",
    )
    p.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help=(
            "Compute device. CUDA strongly preferred — CPU is only an "
            "escape hatch for unit tests (CLAUDE.md non-negotiable: MPS "
            "fallback forbidden, CPU acceptable only with the caveat that "
            "deterministic-bytes is not required for the seed file)."
        ),
    )
    p.add_argument(
        "--baseline-poses", type=str, default=DEFAULT_BASELINE_POSES,
        help=(
            "Path to baseline poses (.pt) used both for "
            "--scale-to-match-mode {linear,mlp} calibration and as the "
            "default fallback (--fallback-mode baseline). The "
            "submissions/baseline_dilated_h64_0_90/optimized_poses.pt "
            "anchor (2.29 contest-CUDA) is the strongest known prior."
        ),
    )
    p.add_argument(
        "--scale-to-match-mode", type=str, default="none",
        choices=["none", "linear", "mlp"],
        help=(
            "Calibration of supercombo's pose head to PoseNet's learned "
            "scale. V2 default 'none' lets pose TTO learn the scale via "
            "gradient descent (recommended). 'linear' = V1 per-dim affine. "
            "'mlp' = small 2-layer MLP fit on baseline poses."
        ),
    )
    # V1 backward-compat flag — alias for --scale-to-match-mode=none.
    p.add_argument(
        "--no-scale-to-match", action="store_true",
        help=(
            "DEPRECATED V1 alias for --scale-to-match-mode=none. Use "
            "--scale-to-match-mode instead."
        ),
    )
    p.add_argument(
        "--fov-crop", dest="fov_crop", action="store_true", default=True,
        help="Apply road-centric FOV crop before resize (V2 default).",
    )
    p.add_argument(
        "--no-fov-crop", dest="fov_crop", action="store_false",
        help="Disable FOV crop (resize the entire frame; V1 behavior).",
    )
    p.add_argument(
        "--legacy-v1-yuv", action="store_true",
        help=(
            "Use V1's incorrect Y-replicated YUV layout (4xY + U + V). "
            "Default is the correct YUV420 planar layout (4 quarter-Y + U + V)."
        ),
    )
    p.add_argument(
        "--no-features-buffer", action="store_true",
        help=(
            "Disable features_buffer recurrent state propagation (zero-fill "
            "every frame, V1 behavior). Default is propagate the buffer "
            "across the 600 pairs to preserve RNN-like temporal context."
        ),
    )
    p.add_argument(
        "--n-frames", type=int, default=1200,
        help="Number of frames to consume from the video",
    )
    p.add_argument(
        "--masks", type=str, default=None,
        help=(
            "Path to mask tensor (.pt or .mkv) for --fallback-mode lane_mark. "
            "Only required when --fallback-mode=lane_mark."
        ),
    )
    p.add_argument(
        "--fallback-mode", type=str, default="baseline",
        choices=["baseline", "lane_mark"],
        help=(
            "What to do when supercombo cannot be loaded. 'baseline' (V2 "
            "default) loads --baseline-poses directly — strongest prior. "
            "'lane_mark' uses compute_zero_cost_poses_from_masks (V1 "
            "behavior, ~0.017 correlation with PoseNet) — requires --masks."
        ),
    )
    p.add_argument(
        "--allow-fallback", action="store_true",
        help=(
            "If supercombo can't be loaded, use --fallback-mode. Without "
            "this flag, missing supercombo is a fatal error."
        ),
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("seed_poses")

    # Resolve V1 backward-compat flag.
    scale_mode = args.scale_to_match_mode
    if args.no_scale_to_match:
        if scale_mode != "none":
            log.warning(
                "--no-scale-to-match (V1) and --scale-to-match-mode=%s both "
                "set; honoring --no-scale-to-match (forcing 'none')",
                scale_mode,
            )
        scale_mode = "none"

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_poses: torch.Tensor | None = None
    used_fallback = False
    fallback_kind: str | None = None

    # ── Try supercombo first ────────────────────────────────────────────
    try:
        log.info("loading supercombo from %s on %s", args.supercombo_path, device)
        sess = load_supercombo_model(args.supercombo_path, device)
        log.info(
            "supercombo loaded — running inference on %s "
            "(fov_crop=%s legacy_v1_yuv=%s features_buffer=%s)",
            args.video, args.fov_crop, args.legacy_v1_yuv,
            not args.no_features_buffer,
        )
        t0 = time.monotonic()
        raw_poses = infer_pose_from_video(
            sess, args.video, n_frames=args.n_frames, device=device,
            fov_crop=args.fov_crop,
            legacy_v1_layout=args.legacy_v1_yuv,
            propagate_features_buffer=not args.no_features_buffer,
        )
        log.info(
            "supercombo inference: %d pairs in %.1fs",
            raw_poses.shape[0], time.monotonic() - t0,
        )

        baseline = None
        if args.baseline_poses:
            bp = Path(args.baseline_poses)
            if not bp.exists():
                log.warning("baseline poses not found at %s — skipping calibration", bp)
            else:
                baseline = torch.load(bp, map_location="cpu", weights_only=True).float()
                log.info("loaded baseline poses %s shape=%s", bp, tuple(baseline.shape))

        seed_poses = seed_pose_tto(
            raw_poses,
            baseline_poses=baseline,
            mode=scale_mode,
        )
    except SupercomboUnavailable as exc:
        log.warning("supercombo unavailable: %s", exc)
        if not args.allow_fallback:
            log.error(
                "supercombo unavailable and --allow-fallback not set. "
                "Either install supercombo or pass --allow-fallback "
                "(default --fallback-mode=baseline loads baseline poses)."
            )
            return 1

        if args.fallback_mode == "baseline":
            bp = Path(args.baseline_poses)
            log.info("fallback_mode=baseline — loading %s", bp)
            try:
                seed_poses = fallback_seed_from_baseline(
                    bp, n_pairs=args.n_frames // 2,
                )
            except SupercomboUnavailable as exc2:
                log.error(
                    "baseline fallback failed: %s. Try "
                    "--fallback-mode=lane_mark --masks <path>.",
                    exc2,
                )
                return 1
            used_fallback = True
            fallback_kind = "baseline"
        else:  # lane_mark
            if not args.masks:
                log.error(
                    "fallback_mode=lane_mark requires --masks <path/to/masks.pt> — "
                    "compute_zero_cost_poses_from_masks needs class-index masks."
                )
                return 1
            masks_path = Path(args.masks)
            if not masks_path.exists():
                log.error("masks file not found: %s", masks_path)
                return 1
            log.info("fallback_mode=lane_mark — loading masks from %s", masks_path)
            # Support both .mkv (decode) and .pt (torch.load).
            if masks_path.suffix.lower() in (".mkv", ".mp4", ".webm"):
                from tac.mask_codec import decode_masks
                masks = decode_masks(masks_path)
                if masks.dtype != torch.long:
                    masks = masks.long()
            else:
                masks = torch.load(
                    masks_path, map_location="cpu", weights_only=True
                )
                if masks.dtype != torch.long:
                    masks = masks.long()
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
            seed_poses = fallback_seed_from_masks(masks)
            used_fallback = True
            fallback_kind = "lane_mark"

    if seed_poses is None:
        log.error("no seed_poses produced — internal error")
        return 1

    # ── Persist ─────────────────────────────────────────────────────────
    # Save as fp16 for archive parity with optimized_poses.pt (which is also
    # fp16 for the same 7.2 KB cost).
    torch.save(seed_poses.half(), output_path)
    size_bytes = output_path.stat().st_size
    log.info(
        "wrote %s shape=%s used_fallback=%s fallback_kind=%s size=%d bytes",
        output_path, tuple(seed_poses.shape), used_fallback,
        fallback_kind, size_bytes,
    )
    # Note: keep the "fallback=True/False" token in stdout for tests that
    # grep for it; add fallback_kind for V2 telemetry.
    print(
        f"OK seed_poses={output_path} shape={tuple(seed_poses.shape)} "
        f"fallback={used_fallback} fallback_kind={fallback_kind} "
        f"size_bytes={size_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
