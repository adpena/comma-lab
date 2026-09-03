"""ddm_msr1 — the rendered-edge probe: is the one family left open (sharpen the
painted class boundary, rather than move it) discriminated by the render itself?

The directional interface bound closes every boundary-MOVING actuator, learned or
hand-written, because moving interface (a,b) trades one flow direction against the
other and the collateral population is 16.6x-646.5x the repairable one. It does not
close a boundary-SHARPENING actuator, which would reduce ambiguity in both
directions at once without moving the boundary.

This probe measures the two quantities that decide whether sharpening is even
indicated on this object:

1. INTERIOR CLASS PALETTE. The mean rendered native RGB of each class's interior
   (token-boundary distance saturated), and the pairwise separations. If two
   classes that exchange heavily are painted close together, contrast is a lever.

2. CROSS-BOUNDARY CONTRAST. Per pixel, the largest native-RGB distance to an
   8-neighbour carrying a different token class. Compared between the manufactured
   pixels and the currently-correct pixels on the same boundary shell. If the
   manufactured pixels sit at systematically softer rendered edges, sharpening is
   indicated and its size is measurable; if the two distributions coincide, the
   render's own edge contrast does not discriminate them and sharpening has no
   address either.

Authority: none. No archive changes, no scorer fires, no pointer moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

EVAL_H, EVAL_W, PAIRS = 384, 512, 600
NUM_PIXELS = PAIRS * EVAL_H * EVAL_W
NUM_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
SATURATING_RADIUS = 12


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def cross_boundary_contrast(rgb: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Largest native-RGB L2 distance to an 8-neighbour of a different token class.

    ``rgb`` is (frames, 3, H, W) float32 in 0..255; ``tokens`` is (frames, H, W).
    """
    padded_rgb = np.pad(rgb, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    padded_tokens = np.pad(tokens, ((0, 0), (1, 1), (1, 1)), mode="edge")
    best = np.zeros(tokens.shape, dtype=np.float32)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            neighbour_rgb = padded_rgb[:, :, dy : dy + EVAL_H, dx : dx + EVAL_W]
            neighbour_tokens = padded_tokens[:, dy : dy + EVAL_H, dx : dx + EVAL_W]
            delta = np.sqrt(((rgb - neighbour_rgb) ** 2).sum(axis=1))
            differs = neighbour_tokens != tokens
            best = np.maximum(best, np.where(differs, delta, 0.0).astype(np.float32))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retained",
        type=Path,
        default=Path(
            "/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/"
            "ddm_mst1_manufactured_stage_split/capture_r2_local/retained"
        ),
    )
    parser.add_argument(
        "--characterize",
        type=Path,
        default=Path(
            "/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/characterize_r1"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/edge_r1"),
    )
    args = parser.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    gt = np.load(args.retained / "inputs" / "gt_argmax_n600.npy").reshape(-1)
    tokens = np.fromfile(args.retained / "inputs" / "tokens_cpu_stage_complete.u8", dtype=np.uint8)
    terminal = np.load(args.retained / "inputs" / "cuda_terminal_argmax_n600.npy").reshape(-1)
    native = np.load(args.retained / "assembled" / "argmax_native_n600.npy").reshape(-1)
    stage_id = np.load(args.characterize / "earliest_stage_id.n600.npy")
    distance = np.load(args.characterize / "token_boundary_distance.n600.npy")
    manufactured = (terminal != gt) & (tokens == gt)
    native_manufactured = manufactured & (stage_id == 0)
    # Interior radius falls back per class: Lane markings are thin and never form a
    # wide uniform token block, so a single global radius would leave Lane's palette
    # undefined and silently NaN every Lane separation.
    interior_radius = {}
    for index in range(NUM_CLASSES):
        for radius in (SATURATING_RADIUS, 8, 4, 2, 1):
            if int(((distance >= radius) & (tokens == index)).sum()) >= 10_000:
                interior_radius[index] = radius
                break
        else:
            interior_radius[index] = 1
    interior = np.zeros(NUM_PIXELS, dtype=bool)
    for index, radius in interior_radius.items():
        interior |= (distance >= radius) & (tokens == index)
    shell = distance == 1
    shell_correct = shell & (terminal == gt) & (native == gt)
    shell_manufactured = shell & native_manufactured

    contrast = np.zeros(NUM_PIXELS, dtype=np.float32)
    palette_sum = np.zeros((NUM_CLASSES, 3), dtype=np.float64)
    palette_count = np.zeros(NUM_CLASSES, dtype=np.int64)
    chunk_dirs = sorted(p for p in (args.retained / "chunks").iterdir() if p.is_dir())
    cursor = 0
    for chunk_dir in chunk_dirs:
        rgb = np.load(chunk_dir / "native_rgb.float32.npy")
        frames = rgb.shape[0]
        count = frames * EVAL_H * EVAL_W
        span = slice(cursor, cursor + count)
        chunk_tokens = tokens[span].reshape(frames, EVAL_H, EVAL_W)
        contrast[span] = cross_boundary_contrast(rgb, chunk_tokens).reshape(-1)
        flat_rgb = rgb.transpose(0, 2, 3, 1).reshape(-1, 3)
        chunk_interior = interior[span]
        chunk_class = tokens[span]
        for index in range(NUM_CLASSES):
            selected = chunk_interior & (chunk_class == index)
            if selected.any():
                palette_sum[index] += flat_rgb[selected].sum(axis=0)
                palette_count[index] += int(selected.sum())
        cursor += count
        del rgb, flat_rgb
    if cursor != NUM_PIXELS:
        raise ValueError(f"chunks covered {cursor} pixels, expected {NUM_PIXELS}")
    print(f"[{time.time() - started:6.1f}s] contrast + palette computed")

    palette = palette_sum / palette_count[:, None]
    separations = []
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            separations.append(
                {
                    "interface": f"{CLASS_NAMES[a]}|{CLASS_NAMES[b]}",
                    "rgb_l2_separation": float(np.sqrt(((palette[a] - palette[b]) ** 2).sum())),
                }
            )
    separations.sort(key=lambda row: row["rgb_l2_separation"])

    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    result = {
        "arm": "ddm_msr1",
        "kind": "rendered_edge_probe",
        "authority": "none; no archive changed, no scorer fired, no pointer moved",
        "interior_palette": {
            "definition": "mean native rendered RGB over token-boundary-interior pixels; the "
            "chebyshev interior radius falls back per class so thin classes still resolve",
            "per_class": [
                {
                    "name": CLASS_NAMES[index],
                    "interior_radius_used": interior_radius[index],
                    "interior_pixels": int(palette_count[index]),
                    "mean_rgb": [float(v) for v in palette[index]],
                }
                for index in range(NUM_CLASSES)
            ],
            "pairwise_separation_sorted": separations,
        },
        "cross_boundary_contrast": {
            "definition": "largest native-RGB L2 distance to an 8-neighbour of a different "
            "token class",
            "shell_manufactured_pixels": int(shell_manufactured.sum()),
            "shell_correct_pixels": int(shell_correct.sum()),
            "quantiles_shell_manufactured": {
                str(q): float(v)
                for q, v in zip(
                    quantiles, np.quantile(contrast[shell_manufactured], quantiles), strict=True
                )
            },
            "quantiles_shell_correct": {
                str(q): float(v)
                for q, v in zip(
                    quantiles, np.quantile(contrast[shell_correct], quantiles), strict=True
                )
            },
            "mean_shell_manufactured": float(contrast[shell_manufactured].mean()),
            "mean_shell_correct": float(contrast[shell_correct].mean()),
        },
    }

    contrast_path = args.out / "cross_boundary_contrast.float32.n600.npy"
    np.save(contrast_path, contrast)  # PAYLOAD_WRITE_ORDER_OK:the result records this retained array's post-write size and digest
    result["payloads"] = {
        "cross_boundary_contrast": {
            "path": str(contrast_path),
            "bytes": contrast_path.stat().st_size,
            "sha256": sha256_of_file(contrast_path),
        }
    }
    result["elapsed_seconds"] = time.time() - started
    result_path = args.out / "MSR1_EDGE.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {result_path} sha256 {sha256_of_file(result_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
