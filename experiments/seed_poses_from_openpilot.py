"""Lane OS-A standalone: produce seed_poses.pt from openpilot supercombo.

Usage::

    python experiments/seed_poses_from_openpilot.py \\
        --supercombo-path /workspace/openpilot/models/supercombo.onnx \\
        --video upstream/videos/0.mkv \\
        --output seed_poses.pt \\
        --device cuda

The output is consumed by ``experiments/optimize_poses.py --seed-poses-path``.

If supercombo is unavailable (model missing, ONNX runtime mismatch), the tool
falls back to the masks-only analytical path
(:func:`tac.openpilot_seeding.fallback_seed_from_masks`) provided ``--masks``
is supplied. Without ``--masks`` the fallback is impossible and the tool
exits with a non-zero status — fail loud per CLAUDE.md.

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
    SupercomboUnavailable,
    fallback_seed_from_masks,
    infer_pose_from_video,
    load_supercombo_model,
    seed_pose_tto,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Lane OS-A: produce seed_poses.pt from openpilot supercombo",
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
        "--baseline-poses", type=str, default=None,
        help=(
            "Path to baseline poses (.pt) for affine calibration. "
            "Recommended: submissions/baseline_dilated_h64_0_90/"
            "optimized_poses.pt — guarantees the seed lands inside "
            "PoseNet's training-time distribution."
        ),
    )
    p.add_argument(
        "--no-scale-to-match", action="store_true",
        help="Skip baseline calibration (use raw supercombo or "
             "lane_mark_pose constants)",
    )
    p.add_argument(
        "--n-frames", type=int, default=1200,
        help="Number of frames to consume from the video",
    )
    p.add_argument(
        "--masks", type=str, default=None,
        help=(
            "Path to mask tensor (.pt) for fallback if supercombo is "
            "unavailable. If supercombo loads successfully this is unused."
        ),
    )
    p.add_argument(
        "--allow-fallback", action="store_true",
        help=(
            "If supercombo can't be loaded, fall back to "
            "compute_zero_cost_poses_from_masks. Requires --masks. Without "
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

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_poses: torch.Tensor | None = None
    used_fallback = False

    # ── Try supercombo first ────────────────────────────────────────────
    try:
        log.info("loading supercombo from %s on %s", args.supercombo_path, device)
        sess = load_supercombo_model(args.supercombo_path, device)
        log.info("supercombo loaded — running inference on %s", args.video)
        t0 = time.monotonic()
        raw_poses = infer_pose_from_video(
            sess, args.video, n_frames=args.n_frames, device=device,
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
            scale_to_match=not args.no_scale_to_match,
        )
    except SupercomboUnavailable as exc:
        log.warning("supercombo unavailable: %s", exc)
        if not args.allow_fallback:
            log.error(
                "supercombo unavailable and --allow-fallback not set. "
                "Either install supercombo or pass --allow-fallback --masks <path>."
            )
            return 1
        if not args.masks:
            log.error(
                "fallback requires --masks <path/to/masks.pt> — "
                "compute_zero_cost_poses_from_masks needs class-index masks."
            )
            return 1
        masks_path = Path(args.masks)
        if not masks_path.exists():
            log.error("masks file not found: %s", masks_path)
            return 1
        log.info("loading masks from %s for fallback", masks_path)
        masks = torch.load(masks_path, map_location="cpu", weights_only=True)
        # Coerce to long if needed.
        if masks.dtype != torch.long:
            masks = masks.long()
        # If masks have a channel dim (N, 1, H, W) squeeze it.
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        seed_poses = fallback_seed_from_masks(masks)
        used_fallback = True

    if seed_poses is None:
        log.error("no seed_poses produced — internal error")
        return 1

    # ── Persist ─────────────────────────────────────────────────────────
    # Save as fp16 for archive parity with optimized_poses.pt (which is also
    # fp16 for the same 7.2 KB cost).
    torch.save(seed_poses.half(), output_path)
    size_bytes = output_path.stat().st_size
    log.info(
        "wrote %s shape=%s used_fallback=%s size=%d bytes",
        output_path, tuple(seed_poses.shape), used_fallback, size_bytes,
    )
    print(
        f"OK seed_poses={output_path} shape={tuple(seed_poses.shape)} "
        f"fallback={used_fallback} size_bytes={size_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
