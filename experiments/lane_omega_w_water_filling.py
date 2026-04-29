"""Lane Ω-W CLI driver — water-filling Lagrangian export.

Loads a SegMap inference checkpoint (anchor-trained), builds a calibration
batch from anchor masks + GT video, runs Hessian + σ² estimation with
eval_roundtrip=True, water-fills to a target bit budget, packs the
quantised state_dict into a tar.xz payload, and reports the realised
budget + roundtrip MSE.

Designed to be invoked from scripts/remote_lane_omega_w_water_filling.sh
as part of Stage 3. POST-COMPRESS only — no retraining.

Compliance:
  - eval_roundtrip=True default; --eval-roundtrip-off disabled (raises)
  - --device cuda required (default); --device cpu prints opt-in banner
  - No scorer load at inflate (this is COMPRESS-time)
  - Calls pack_payload_tar_xz WITHOUT exponents= kwarg (Round 1 lesson)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tac.mask_codec import decode_masks_auto  # noqa: E402
from tac.segmap_renderer import SegMap  # noqa: E402
from tac.water_filling_codec import (  # noqa: E402
    WaterFillError,
    export_with_water_filling,
)


def _build_calibration_batch(
    anchor_masks: Path,
    gt_video: Path | None,
    num_calib: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode anchor masks (one-hot) + GT frames into calibration tensors.

    Returns (inputs, targets, frame_idx). If gt_video is None, falls back
    to a deterministic mid-grey target (Hessian rank-correlation still
    holds; absolute scale will differ).
    """
    mask_classes = decode_masks_auto(anchor_masks).long()  # (N, H, W)
    n_total = int(mask_classes.shape[0])
    k = min(num_calib, n_total)
    # Evenly-spaced frames for diversity
    if k >= n_total:
        idx = torch.arange(n_total, dtype=torch.long)
    else:
        idx = torch.linspace(0, n_total - 1, steps=k).round().long()

    sel = mask_classes[idx]  # (K, H, W)
    one_hot = torch.nn.functional.one_hot(sel, num_classes=5).permute(0, 3, 1, 2).float()
    H, W = sel.shape[-2], sel.shape[-1]

    if gt_video is not None and gt_video.exists():
        try:
            from tac.video import decode_frames  # type: ignore[import-untyped]

            frames = decode_frames(str(gt_video))  # (N, H, W, 3) uint8
            if isinstance(frames, torch.Tensor):
                tgt = frames[idx].float().permute(0, 3, 1, 2)
            else:
                import numpy as np

                arr = np.asarray(frames)
                tgt = torch.from_numpy(arr[idx.numpy()]).float().permute(0, 3, 1, 2)
            # Resize to mask resolution if needed
            if tgt.shape[-2:] != (H, W):
                tgt = torch.nn.functional.interpolate(
                    tgt, size=(H, W), mode="bilinear", align_corners=False
                )
        except Exception as exc:  # pragma: no cover — exercised in lane script
            print(f"[CALIB] GT video decode failed ({exc}); falling back to mid-grey")
            tgt = torch.full((k, 3, H, W), 127.5)
    else:
        tgt = torch.full((k, 3, H, W), 127.5)

    return one_hot, tgt, idx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lane Ω-W water-filling Lagrangian export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="SegMap inference checkpoint (.pt)")
    parser.add_argument("--anchor-masks", type=Path, required=True,
                        help="anchor masks.mkv (decoded for calibration)")
    parser.add_argument("--gt-video", type=Path, default=None,
                        help="optional GT video (improves Hessian fidelity)")
    parser.add_argument("--total-bits", type=int, required=True,
                        help="signed-integer bit budget for conv weights")
    parser.add_argument("--output-payload", type=Path, required=True,
                        help="tar.xz payload destination")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                        help="cuda required in production; cpu opt-in only")
    parser.add_argument("--num-calib", type=int, default=64,
                        help="calibration batch size for Hessian estimation")
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--block-hidden", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--max-frame-index", type=int, default=1200)
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="optional JSON summary destination")
    parser.add_argument("--verify-tol", type=float, default=1e-1,
                        help="verify_roundtrip MSE tolerance")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda requested but torch.cuda.is_available() is False.",
              file=sys.stderr)
        print("Pass --device cpu explicitly to opt into CPU fallback.", file=sys.stderr)
        return 2

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SegMap(
        hidden=args.hidden,
        block_hidden=args.block_hidden,
        num_blocks=args.num_blocks,
        max_frame_index=args.max_frame_index,
    )
    model.load_state_dict(state, strict=True)

    inputs, targets, frame_idx = _build_calibration_batch(
        args.anchor_masks, args.gt_video, args.num_calib, args.device
    )
    print(f"[LANE-OMEGA-W] calibration: K={inputs.shape[0]} H={inputs.shape[-2]} "
          f"W={inputs.shape[-1]} target_bits={args.total_bits}")

    try:
        out = export_with_water_filling(
            model,
            inputs,
            targets,
            frame_idx,
            total_bits=args.total_bits,
            output_path=str(args.output_payload),
            device=args.device,
            eval_roundtrip=True,
            verify_tol=args.verify_tol,
        )
    except WaterFillError as exc:
        print(f"FATAL: WaterFillError: {exc}", file=sys.stderr)
        return 3

    out["target_bits"] = int(args.total_bits)
    out["payload_path"] = str(args.output_payload)
    out["checkpoint"] = str(args.checkpoint)
    print("[LANE-OMEGA-W] DONE:", json.dumps({
        "target_bits": out["target_bits"],
        "realised_bits": out["realised_bits"],
        "payload_bytes": out["payload_bytes"],
        "roundtrip_mse_max": out["roundtrip_mse_max"],
        "n_layers": len(out["allocations_per_layer"]),
    }, indent=2))

    if args.summary_json is not None:
        # Tensor → list for JSON
        out["qint_assignment"] = {
            k: list(v) for k, v in out["qint_assignment"].items()
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
