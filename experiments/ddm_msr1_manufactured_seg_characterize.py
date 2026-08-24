"""ddm_msr1 — characterize the manufactured Seg error of the dx2 object.

Reproduces the mst1 stage decomposition from the retained PRIMARY fields (not from
mst1's derived masks and not from any quoted ratio), then characterizes the
manufactured set by class, stage, flip direction, token-boundary distance,
token-window purity, frozen-scorer margin deficit, pair, and image row.

Authority: none. This is a $0 derivation over already-measured retained fields.
It changes no archive, fires no scorer, and moves no pointer.

Class indices are SELF-DETECTED from the GT field's spatial/static signature and
asserted against the canonical comma10k order; they are never hardcoded blind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

EVAL_H, EVAL_W, PAIRS = 384, 512, 600
NUM_PIXELS = PAIRS * EVAL_H * EVAL_W
NUM_CLASSES = 5

# Rate-term derivative, CITED from ddm_tx1_toolbox_crosswalk_20260819.md sec 0.
# Never re-derived here (task #1207).
EXCHANGE_RATE_S_PER_BYTE = 6.658590e-07
# Two-currency demand, CITED from ddm_fb1_sub012_feasibility_bound_20260823.md.
DEMAND_BYTES = 42382

STAGE_NAMES = (
    "native_render",
    "preuint8_roundtrip",
    "uint8_roundtrip",
    "cpu_to_cuda_terminal",
)

# Canonical comma10k order. Each entry is the falsifiable spatial/static signature
# the class must satisfy in the GT field for the index assignment to be admitted.
# (name, area_pct_lo, area_pct_hi, centroid_row_lo, centroid_row_hi, iou_lo, iou_hi)
CLASS_SIGNATURE = (
    ("Road", 22.0, 24.5, 200.0, 270.0, 0.90, 0.99),
    ("Lane", 0.45, 0.75, 190.0, 265.0, 0.15, 0.40),
    ("Undrivable", 47.0, 52.0, 60.0, 130.0, 0.98, 1.00),
    ("Movable", 1.00, 1.50, 165.0, 235.0, 0.75, 0.95),
    ("MyCar", 24.0, 27.0, 300.0, 365.0, 0.98, 1.00),
)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def byte_ceiling(pixel_count: int) -> float:
    """Byte-equivalent of a pixel count, at the cited exchange rate."""
    return (100.0 * pixel_count / NUM_PIXELS) / EXCHANGE_RATE_S_PER_BYTE


def load_fields(retained: Path) -> dict[str, np.ndarray]:
    fields = {
        "L": np.fromfile(retained / "inputs" / "tokens_cpu_stage_complete.u8", dtype=np.uint8),
        "G": np.load(retained / "inputs" / "gt_argmax_n600.npy").reshape(-1),
        "A": np.load(retained / "inputs" / "cuda_terminal_argmax_n600.npy").reshape(-1),
        "B1": np.load(retained / "assembled" / "argmax_native_n600.npy").reshape(-1),
        "B2": np.load(retained / "assembled" / "argmax_preuint8_n600.npy").reshape(-1),
        "B3": np.load(retained / "assembled" / "argmax_uint8_n600.npy").reshape(-1),
    }
    for name, array in fields.items():
        if array.shape != (NUM_PIXELS,):
            raise ValueError(f"field {name} has shape {array.shape}, expected {NUM_PIXELS}")
        if array.dtype != np.uint8:
            raise ValueError(f"field {name} has dtype {array.dtype}, expected uint8")
        if int(array.max()) >= NUM_CLASSES:
            raise ValueError(f"field {name} carries a class index outside 0..{NUM_CLASSES - 1}")
    return fields


def self_detect_classes(gt: np.ndarray) -> list[dict]:
    """Derive each class index's identity from its own spatial/static signature.

    Raises if any measured signature falls outside the declared admissible band,
    so a re-ordered head can never be silently mislabelled.
    """
    volume = gt.reshape(PAIRS, EVAL_H, EVAL_W)
    rows = np.arange(EVAL_H, dtype=np.float64)
    detected = []
    for index in range(NUM_CLASSES):
        mask = volume == index
        area_pct = float(mask.mean() * 100.0)
        per_row = mask.sum(axis=(0, 2)).astype(np.float64)
        centroid = float((per_row * rows).sum() / per_row.sum())
        cumulative = np.cumsum(per_row) / per_row.sum()
        row_p05 = int(np.searchsorted(cumulative, 0.05))
        row_p95 = int(np.searchsorted(cumulative, 0.95))
        earlier, later = mask[:-1], mask[1:]
        iou = float((earlier & later).sum() / (earlier | later).sum())
        name, area_lo, area_hi, cen_lo, cen_hi, iou_lo, iou_hi = CLASS_SIGNATURE[index]
        inside = (
            area_lo <= area_pct <= area_hi
            and cen_lo <= centroid <= cen_hi
            and iou_lo <= iou <= iou_hi
        )
        if not inside:
            raise ValueError(
                f"class index {index} signature (area {area_pct:.4f}%, centroid {centroid:.1f}, "
                f"IoU {iou:.4f}) is outside the admissible band for '{name}'. "
                "Refusing to assign names; the head order may have changed."
            )
        detected.append(
            {
                "index": index,
                "name": name,
                "gt_pixels": int(mask.sum()),
                "area_pct": area_pct,
                "centroid_row": centroid,
                "row_p05": row_p05,
                "row_p95": row_p95,
                "temporal_iou": iou,
            }
        )
    return detected


def chebyshev_boundary_distance(label_volume: np.ndarray, max_radius: int) -> np.ndarray:
    """Chebyshev distance from each pixel to the nearest differently-labelled pixel.

    Value ``r`` in 1..max_radius-1 means the (2r+1)-square window around the pixel is
    the first non-uniform one, i.e. the nearest different label sits exactly r pixels
    away in the chessboard metric. The top value ``max_radius`` is saturating and means
    ">= max_radius": the (2*max_radius-1)-square window is a uniform block of the
    pixel's own token class. Exact for the chessboard metric because a pixel is within
    radius r of a different label exactly when its (2r+1)-square window is not uniform.
    """
    distance = np.full(label_volume.shape, max_radius, dtype=np.uint8)
    running_max = label_volume.copy()
    running_min = label_volume.copy()
    for radius in range(1, max_radius):
        running_max = _dilate3(running_max, np.maximum)
        running_min = _dilate3(running_min, np.minimum)
        mixed = running_max != running_min
        distance = np.where(mixed & (distance == max_radius), np.uint8(radius), distance)
    return distance


def _dilate3(volume: np.ndarray, op) -> np.ndarray:
    """One 3x3 morphological step per frame, replicate-padded."""
    padded = np.pad(volume, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = volume.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out = op(out, padded[:, dy : dy + volume.shape[1], dx : dx + volume.shape[2]])
    return out


def window_fidelity(label_volume: np.ndarray, gt_volume: np.ndarray, radius: int) -> np.ndarray:
    """Fraction of each pixel's (2r+1)-square window where the token field equals GT.

    This is local token-field FIDELITY (how lossy the transmitted field is around the
    pixel), which is a different quantity from the boundary distance above (how uniform
    the token field is around the pixel).
    """
    matches = (label_volume == gt_volume).astype(np.float32)
    size = 2 * radius + 1
    padded = np.pad(matches, ((0, 0), (radius, radius), (radius, radius)), mode="edge")
    cumulative = padded.cumsum(axis=1).cumsum(axis=2)
    cumulative = np.pad(cumulative, ((0, 0), (1, 0), (1, 0)), mode="constant")
    height, width = label_volume.shape[1], label_volume.shape[2]
    total = (
        cumulative[:, size:, size:]
        - cumulative[:, :height, size:]
        - cumulative[:, size:, :width]
        + cumulative[:, :height, :width]
    )
    return total / float(size * size)


def verify_chunk_order(retained: Path, assembled_native: np.ndarray) -> dict:
    """Positive control: lexical chunk order must reproduce the assembled n600 field.

    Every per-chunk quantity (logits, RGB) is indexed by this order. If it disagreed
    with the assembled argmax, every margin below would be silently misaligned with GT.
    """
    chunk_dirs = sorted(p for p in (retained / "chunks").iterdir() if p.is_dir())
    pieces = [np.load(d / "argmax_native.uint8.npy").reshape(-1) for d in chunk_dirs]
    rebuilt = np.concatenate(pieces)
    if rebuilt.shape != assembled_native.shape:
        raise ValueError(f"chunk concat shape {rebuilt.shape} != assembled {assembled_native.shape}")
    mismatches = int((rebuilt != assembled_native).sum())
    if mismatches:
        raise ValueError(
            f"lexical chunk order disagrees with the assembled native argmax at "
            f"{mismatches} pixels; per-chunk logits cannot be trusted"
        )
    return {
        "chunks": len(chunk_dirs),
        "first": chunk_dirs[0].name,
        "last": chunk_dirs[-1].name,
        "mismatch_pixels_vs_assembled_native_argmax": 0,
    }


def native_margin_deficit(retained: Path, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel frozen-SegNet native-stage logit deficit and top-1 margin.

    ``deficit = logit[argmax] - logit[gt_class]`` (0 when the native argmax is the GT
    class, positive when wrong).  ``margin = logit[top1] - logit[top2]``.
    Streamed chunk by chunk so the 2.3 GB of retained logits is never fully resident.
    """
    deficit = np.zeros(NUM_PIXELS, dtype=np.float32)
    margin = np.zeros(NUM_PIXELS, dtype=np.float32)
    chunk_dirs = sorted(p for p in (retained / "chunks").iterdir() if p.is_dir())
    cursor = 0
    for chunk_dir in chunk_dirs:
        logits = np.load(chunk_dir / "logits_native.float32.npy")
        if logits.ndim != 4 or logits.shape[-1] != NUM_CLASSES:
            logits = np.moveaxis(logits, 1, -1)
        flat = logits.reshape(-1, NUM_CLASSES)
        count = flat.shape[0]
        span = slice(cursor, cursor + count)
        ordered = np.sort(flat, axis=1)
        top1, top2 = ordered[:, -1], ordered[:, -2]
        gt_logit = flat[np.arange(count), gt[span]]
        deficit[span] = top1 - gt_logit
        margin[span] = top1 - top2
        cursor += count
        del logits, flat, ordered
    if cursor != NUM_PIXELS:
        raise ValueError(f"chunk logits covered {cursor} pixels, expected {NUM_PIXELS}")
    return deficit, margin


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
        "--out",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_msr1_manufactured_seg_reduction/characterize_r1"),
    )
    parser.add_argument("--boundary-radius", type=int, default=12)
    parser.add_argument("--purity-radius", type=int, default=8)
    parser.add_argument("--skip-margin", action="store_true")
    args = parser.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "arm": "ddm_msr1",
        "kind": "derivation_over_retained_measured_fields",
        "authority": "none; no archive changed, no scorer fired, no pointer moved",
        "exchange_rate_s_per_byte": EXCHANGE_RATE_S_PER_BYTE,
        "exchange_rate_source": "ddm_tx1_toolbox_crosswalk_20260819.md sec 0 (CITED, not re-derived)",
        "demand_bytes": DEMAND_BYTES,
        "num_pixels": NUM_PIXELS,
    }

    source_files = {
        "L_decoded_tokens": args.retained / "inputs" / "tokens_cpu_stage_complete.u8",
        "G_dali_gt_argmax": args.retained / "inputs" / "gt_argmax_n600.npy",
        "A_cuda_terminal_argmax": args.retained / "inputs" / "cuda_terminal_argmax_n600.npy",
        "B1_argmax_native": args.retained / "assembled" / "argmax_native_n600.npy",
        "B2_argmax_preuint8": args.retained / "assembled" / "argmax_preuint8_n600.npy",
        "B3_argmax_uint8": args.retained / "assembled" / "argmax_uint8_n600.npy",
        "renderer_source": args.retained / "provenance_sources" / "renderer_source.py",
        "dx2_archive": args.retained / "provenance_sources" / "archive.zip",
    }
    result["source_receipts"] = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_of_file(path)}
        for name, path in source_files.items()
    }
    print(f"[{time.time() - started:6.1f}s] hashed {len(source_files)} source receipts")

    fields = load_fields(args.retained)
    gt, tokens, terminal = fields["G"], fields["L"], fields["A"]

    result["classes"] = self_detect_classes(gt)
    print(f"[{time.time() - started:6.1f}s] self-detected class signatures")
    names = [entry["name"] for entry in result["classes"]]

    representation_error = tokens != gt
    final_error = terminal != gt
    manufactured = final_error & ~representation_error
    result["gates"] = {
        "transmitted_representation_errors_L_ne_G": int(representation_error.sum()),
        "final_dx2_errors_A_ne_G": int(final_error.sum()),
        "final_manufactured_A_ne_G_and_L_eq_G": int(manufactured.sum()),
        "representation_errors_surviving": int((final_error & representation_error).sum()),
        "representation_errors_repaired": int((~final_error & representation_error).sum()),
        "manufactured_fraction_of_final": float(manufactured.sum() / final_error.sum()),
        "d_seg_from_final_error": float(final_error.sum() / NUM_PIXELS),
    }
    print(f"[{time.time() - started:6.1f}s] gates reproduced: {result['gates']}")

    wrong_at = [
        fields["B1"] != gt,
        fields["B2"] != gt,
        fields["B3"] != gt,
        final_error,
    ]
    result["state_error_trajectory"] = {
        "L": int(representation_error.sum()),
        **{name: int(w.sum()) for name, w in zip(("B1_native", "B2_preuint8", "B3_uint8", "A_cuda"), wrong_at, strict=True)},
    }
    stage_id = np.full(NUM_PIXELS, 255, dtype=np.uint8)
    assigned = np.zeros(NUM_PIXELS, dtype=bool)
    stage_rows = []
    for index, (name, wrong) in enumerate(zip(STAGE_NAMES, wrong_at, strict=True)):
        selected = manufactured & wrong & ~assigned
        stage_id[selected] = index
        assigned |= selected
        count = int(selected.sum())
        stage_rows.append(
            {
                "stage": name,
                "earliest_final_manufactured": count,
                "share_of_manufactured": float(count / manufactured.sum()),
                "byte_ceiling": byte_ceiling(count),
                "pct_of_demand": 100.0 * byte_ceiling(count) / DEMAND_BYTES,
            }
        )
    result["stage_attribution"] = stage_rows
    result["all_manufactured_byte_ceiling"] = byte_ceiling(int(manufactured.sum()))
    print(f"[{time.time() - started:6.1f}s] stage attribution reproduced")

    # ---- per class x stage ------------------------------------------------
    per_class = []
    for index, entry in enumerate(result["classes"]):
        class_mask = gt == index
        class_manufactured = manufactured & class_mask
        row = {
            "index": index,
            "name": entry["name"],
            "gt_pixels": entry["gt_pixels"],
            "manufactured": int(class_manufactured.sum()),
            "manufactured_per_million_of_class": float(
                class_manufactured.sum() / entry["gt_pixels"] * 1e6
            ),
            "byte_ceiling": byte_ceiling(int(class_manufactured.sum())),
            "by_stage": {
                name: int((class_manufactured & (stage_id == stage)).sum())
                for stage, name in enumerate(STAGE_NAMES)
            },
        }
        per_class.append(row)
    result["per_class"] = per_class

    # ---- flip direction matrix (GT class -> terminal class), manufactured only
    manufactured_index = np.flatnonzero(manufactured)
    from_class = gt[manufactured_index]
    to_class = terminal[manufactured_index]
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (from_class, to_class), 1)
    result["flip_direction_matrix"] = {
        "row_is_gt_class": names,
        "column_is_terminal_class": names,
        "counts": matrix.tolist(),
    }

    # ---- token-field boundary distance -----------------------------------
    token_volume = tokens.reshape(PAIRS, EVAL_H, EVAL_W)
    gt_volume = gt.reshape(PAIRS, EVAL_H, EVAL_W)
    distance = chebyshev_boundary_distance(token_volume, args.boundary_radius).reshape(-1)
    print(f"[{time.time() - started:6.1f}s] token boundary distance computed")
    edges = list(range(args.boundary_radius + 1))
    result["boundary_distance"] = {
        "metric": "chebyshev distance in the decoded token field L to the nearest "
        "differently-labelled pixel; saturates at the top bucket",
        "saturating_radius": args.boundary_radius,
        "buckets": [
            {
                "distance": d,
                "manufactured": int((manufactured & (distance == d)).sum()),
                "all_pixels": int((distance == d).sum()),
                "native_stage_manufactured": int(
                    (manufactured & (stage_id == 0) & (distance == d)).sum()
                ),
            }
            for d in edges
        ],
    }

    # ---- token-window fidelity -------------------------------------------
    fidelity = window_fidelity(token_volume, gt_volume, args.purity_radius).reshape(-1)
    print(f"[{time.time() - started:6.1f}s] token window fidelity computed")
    fidelity_edges = [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    fidelity_rows = []
    for lo, hi in pairwise(fidelity_edges):
        band = (fidelity >= lo) & (fidelity < hi)
        fidelity_rows.append(
            {
                "lo": lo,
                "hi": hi,
                "manufactured": int((manufactured & band).sum()),
                "all_pixels": int(band.sum()),
            }
        )
    exact = fidelity >= 1.0
    fidelity_rows.append(
        {
            "lo": 1.0,
            "hi": 1.0,
            "label": f"window {2 * args.purity_radius + 1}^2 token field exactly equals GT",
            "manufactured": int((manufactured & exact).sum()),
            "all_pixels": int(exact.sum()),
            "native_stage_manufactured": int((manufactured & (stage_id == 0) & exact).sum()),
        }
    )
    result["token_window_fidelity"] = {
        "radius": args.purity_radius,
        "definition": "fraction of the (2r+1)^2 token window where L equals G",
        "bands": fidelity_rows,
    }

    # ---- per pair and per row --------------------------------------------
    per_pair = manufactured.reshape(PAIRS, -1).sum(axis=1)
    order = np.sort(per_pair)[::-1]
    result["per_pair"] = {
        "min": int(per_pair.min()),
        "median": float(np.median(per_pair)),
        "mean": float(per_pair.mean()),
        "max": int(per_pair.max()),
        "top10_share": float(order[:10].sum() / per_pair.sum()),
        "top60_share": float(order[:60].sum() / per_pair.sum()),
        "pairs_with_zero": int((per_pair == 0).sum()),
    }
    per_row = manufactured.reshape(PAIRS, EVAL_H, EVAL_W).sum(axis=(0, 2))
    result["per_row_manufactured"] = per_row.astype(int).tolist()

    # ---- frozen-scorer margin deficit ------------------------------------
    if not args.skip_margin:
        result["chunk_order_control"] = verify_chunk_order(args.retained, fields["B1"])
        print(f"[{time.time() - started:6.1f}s] chunk-order positive control passed")
        deficit, margin = native_margin_deficit(args.retained, gt)
        print(f"[{time.time() - started:6.1f}s] native logit deficit/margin computed")
        native_manufactured = manufactured & (stage_id == 0)
        correct_native = fields["B1"] == gt
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        result["native_margin"] = {
            "definition": "deficit = native logit[argmax] - native logit[GT class]; "
            "margin = native logit[top1] - logit[top2]",
            "deficit_quantiles_native_stage_manufactured": {
                str(q): float(v)
                for q, v in zip(quantiles, np.quantile(deficit[native_manufactured], quantiles), strict=True)
            },
            "top1_margin_quantiles_correct_pixels": {
                str(q): float(v)
                for q, v in zip(quantiles, np.quantile(margin[correct_native], quantiles), strict=True)
            },
            "correct_pixel_margin_mean": float(margin[correct_native].mean()),
            "native_stage_manufactured_deficit_mean": float(deficit[native_manufactured].mean()),
        }
        thresholds = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        result["native_margin"]["deficit_cdf_native_stage_manufactured"] = [
            {
                "deficit_below": t,
                "count": int((native_manufactured & (deficit < t)).sum()),
                "share": float((native_manufactured & (deficit < t)).sum() / native_manufactured.sum()),
                "byte_ceiling": byte_ceiling(int((native_manufactured & (deficit < t)).sum())),
            }
            for t in thresholds
        ]
        deficit_payload = args.out / "native_logit_deficit.float32.n600.npy"
        np.save(deficit_payload, deficit)
        result["payloads"] = result.get("payloads", {})
        result["payloads"]["native_logit_deficit"] = {
            "path": str(deficit_payload),
            "bytes": deficit_payload.stat().st_size,
            "sha256": sha256_of_file(deficit_payload),
        }

    # ---- persist derived payloads (never scalars alone) -------------------
    result.setdefault("payloads", {})
    for name, array in (
        ("earliest_stage_id", stage_id),
        ("manufactured_support_packbits", np.packbits(manufactured)),
        ("token_boundary_distance", distance.astype(np.uint8)),
    ):
        path = args.out / f"{name}.n600.npy"
        np.save(path, array)
        result["payloads"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_of_file(path),
        }
    fidelity_path = args.out / "token_window_fidelity.float32.n600.npy"
    np.save(fidelity_path, fidelity.astype(np.float32))
    result["payloads"]["token_window_fidelity"] = {
        "path": str(fidelity_path),
        "bytes": fidelity_path.stat().st_size,
        "sha256": sha256_of_file(fidelity_path),
    }

    result["elapsed_seconds"] = time.time() - started
    result_path = args.out / "MSR1_CHARACTERIZE.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[{time.time() - started:6.1f}s] wrote {result_path}")
    print(f"MSR1_CHARACTERIZE.json sha256 {sha256_of_file(result_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
