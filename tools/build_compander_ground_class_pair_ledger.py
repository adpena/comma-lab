#!/usr/bin/env python3
"""Build the read-only n600 class-pair row-density ledger for the S1 compander gate.

This consumes cached GT and cached witness argmax tensors only. It never imports or
invokes a scorer, renderer, evaluator, trainer, or provider client. The output is an
advisory mechanism ledger, not an A/B verdict or promotion-grade score receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GT = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_CHUNKS = (
    REPO / "experiments/results/residual_inr_adversarial_overturn_20260630T235910Z/n600"
)
DEFAULT_OUT = REPO / ".omx/research/compander_ground_class_pair_ledger_n600_20260713.json"
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
HORIZON_ROW = 174.0
SOFTENING_OFFSET_ROWS = 32.5257801441824
EXPECTED_SHAPE = (600, 384, 512)
SCHEMA = "compander_ground_class_pair_ledger.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    return values / total if total > 0.0 else np.zeros_like(values)


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = _normalize(left)
    right = _normalize(right)
    mid = 0.5 * (left + right)
    terms = np.zeros_like(left)
    mask = left > 0.0
    terms[mask] += 0.5 * left[mask] * np.log(left[mask] / mid[mask])
    mask = right > 0.0
    terms[mask] += 0.5 * right[mask] * np.log(right[mask] / mid[mask])
    return float(terms.sum())


def _candidate_metrics(row_counts: np.ndarray) -> dict[str, Any] | None:
    rows = np.arange(row_counts.size, dtype=np.float64)
    ground = rows > HORIZON_ROW
    target = np.asarray(row_counts, dtype=np.float64)[ground]
    if float(target.sum()) <= 0.0:
        return None
    distance = rows[ground] - HORIZON_ROW
    candidates = {
        "uniform": np.ones_like(distance),
        "log_depth_unshifted": 1.0 / distance,
        "softened_inverse_depth_s1_fixed": np.power(
            distance + SOFTENING_OFFSET_ROWS, -2.0
        ),
    }
    return {
        name: {
            "js_divergence_nats": _js_divergence(target, density),
            "cdf_l1_rows": float(
                np.abs(np.cumsum(_normalize(target)) - np.cumsum(_normalize(density))).sum()
            ),
        }
        for name, density in candidates.items()
    }


def accumulate_directed_row_counts(
    gt_lstars: np.ndarray,
    witness_chunks: list[tuple[np.ndarray, np.ndarray]],
    *,
    expected_shape: tuple[int, int, int] = EXPECTED_SHAPE,
) -> tuple[np.ndarray, list[int]]:
    """Return ``[source,target,row]`` flip counts after strict pair/shape validation."""

    n_pairs, height, width = expected_shape
    gt = np.asarray(gt_lstars)
    if gt.shape != expected_shape:
        raise ValueError(f"GT lstars shape {gt.shape} != required {expected_shape}")
    if gt.min() < 0 or gt.max() >= len(CLASS_NAMES):
        raise ValueError("GT lstars contain a class outside canonical range 0..4")

    seen: list[int] = []
    counts = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES), height), dtype=np.int64)
    row_index = np.broadcast_to(
        np.arange(height, dtype=np.int64)[:, None], (height, width)
    ).reshape(-1)
    for pairs_raw, witness_raw in witness_chunks:
        pairs = np.asarray(pairs_raw, dtype=np.int64).reshape(-1)
        witness = np.asarray(witness_raw)
        if witness.shape != (pairs.size, height, width):
            raise ValueError(
                f"witness chunk shape {witness.shape} != {(pairs.size, height, width)}"
            )
        if witness.min() < 0 or witness.max() >= len(CLASS_NAMES):
            raise ValueError("witness argmax contains a class outside canonical range 0..4")
        for local_index, pair_index in enumerate(pairs.tolist()):
            if not 0 <= pair_index < n_pairs:
                raise ValueError(f"pair index {pair_index} outside 0..{n_pairs - 1}")
            seen.append(pair_index)
            source = gt[pair_index].reshape(-1).astype(np.int64, copy=False)
            target = witness[local_index].reshape(-1).astype(np.int64, copy=False)
            code = source * len(CLASS_NAMES) + target
            flipped = source != target
            encoded = code[flipped] * height + row_index[flipped]
            counts += np.bincount(
                encoded,
                minlength=len(CLASS_NAMES) * len(CLASS_NAMES) * height,
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES), height)

    if len(seen) != n_pairs or sorted(seen) != list(range(n_pairs)):
        duplicates = sorted({item for item in seen if seen.count(item) > 1})
        missing = sorted(set(range(n_pairs)).difference(seen))
        raise ValueError(
            f"pair custody must be exactly 0..{n_pairs - 1}; "
            f"seen={len(seen)}, duplicates={duplicates[:20]}, missing={missing[:20]}"
        )
    return counts, sorted(seen)


def _pair_row(
    source: int,
    target: int,
    row_counts: np.ndarray,
    *,
    total_flips: int,
    total_pixels: int,
    strict_planar_ground: bool,
) -> dict[str, Any]:
    count = int(row_counts.sum())
    return {
        "source_class": source,
        "source_name": CLASS_NAMES[source],
        "target_class": target,
        "target_name": CLASS_NAMES[target],
        "strict_planar_ground_pair": strict_planar_ground,
        "flip_count": count,
        "share_of_all_flips": float(count / total_flips) if total_flips else 0.0,
        "share_of_all_pixels": float(count / total_pixels),
        "row_counts_0_through_383": row_counts.astype(int).tolist(),
        "row_density_0_through_383": _normalize(row_counts).tolist(),
        "ground_rows_v_gt_174_candidate_metrics": _candidate_metrics(row_counts),
    }


def ledger_from_counts(
    counts: np.ndarray,
    *,
    n_pairs: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    total_pixels = int(n_pairs * height * width)
    total_flips = int(counts.sum() - np.trace(counts.sum(axis=2)))
    directed: list[dict[str, Any]] = []
    undirected: list[dict[str, Any]] = []
    for source in range(len(CLASS_NAMES)):
        for target in range(len(CLASS_NAMES)):
            if source == target:
                continue
            directed.append(
                _pair_row(
                    source,
                    target,
                    counts[source, target],
                    total_flips=total_flips,
                    total_pixels=total_pixels,
                    strict_planar_ground={source, target} == {0, 1},
                )
            )
    for left in range(len(CLASS_NAMES)):
        for right in range(left + 1, len(CLASS_NAMES)):
            row_counts = counts[left, right] + counts[right, left]
            row = _pair_row(
                left,
                right,
                row_counts,
                total_flips=total_flips,
                total_pixels=total_pixels,
                strict_planar_ground=(left, right) == (0, 1),
            )
            row["pair_kind"] = "undirected_sum_of_both_directions"
            undirected.append(row)
    return {
        "n_pairs": n_pairs,
        "shape": [n_pairs, height, width],
        "canonical_classes": list(CLASS_NAMES),
        "total_pixels": total_pixels,
        "total_flip_count": total_flips,
        "cached_argmax_disagreement_rate": float(total_flips / total_pixels),
        "strict_planar_ground_definition": (
            "Road<->Lane only; Undrivable contains mixed sky/non-ground support and is not "
            "globally asserted planar"
        ),
        "directed_pairs": directed,
        "undirected_pairs": undirected,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def build_ledger(gt_path: Path, chunk_dir: Path, out_path: Path) -> dict[str, Any]:
    chunk_paths = sorted(chunk_dir.glob("chunk_*.npz"))
    if len(chunk_paths) != 6:
        raise ValueError(f"expected exactly six cached chunks, found {len(chunk_paths)}")
    sources = [gt_path, *chunk_paths]
    custody = [
        {
            "path": str(path.resolve().relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sources
    ]
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA,
                "sources": custody,
                "horizon_row": HORIZON_ROW,
                "softening_offset_rows": SOFTENING_OFFSET_ROWS,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if out_path.exists():
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        if prior.get("input_fingerprint_sha256") == fingerprint:
            return {**prior, "reused_existing_output": True}

    with np.load(gt_path, allow_pickle=False) as gt_npz:
        if int(gt_npz["n_pairs"]) != EXPECTED_SHAPE[0]:
            raise ValueError("GT cache n_pairs is not 600")
        gt_lstars = gt_npz["lstars"]
        chunks: list[tuple[np.ndarray, np.ndarray]] = []
        for path in chunk_paths:
            with np.load(path, allow_pickle=False) as chunk:
                chunks.append((chunk["pairs"], chunk["wit_am"]))
        counts, seen = accumulate_directed_row_counts(gt_lstars, chunks)

    payload = {
        "schema": SCHEMA,
        "input_fingerprint_sha256": fingerprint,
        "source_custody": custody,
        "pair_index_custody": {
            "count": len(seen),
            "minimum": min(seen),
            "maximum": max(seen),
            "unique": len(set(seen)),
            "gap_free_0_through_599": seen == list(range(600)),
        },
        "profile": {
            "horizon_row": HORIZON_ROW,
            "softening_offset_rows": SOFTENING_OFFSET_ROWS,
            "value_provenance": (
                ".omx/research/manifold_geometry_slots_probe_s1_s2_20260713.json"
            ),
            "fixed_not_refit_on_class_pairs": True,
        },
        "authority": {
            "axis": "[cached-argmax local-CPU analysis]",
            "score_authority": False,
            "promotion_authority": False,
            "receiver_close_authority": False,
            "false_authority_warning": (
                "cached source->witness argmax flips locate mechanism debt only; no scorer, "
                "archive parse-back, PoseNet, bytes, or chart-arm training A/B ran"
            ),
        },
        "command": (
            ".venv/bin/python tools/build_compander_ground_class_pair_ledger.py"
        ),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        **ledger_from_counts(
            counts,
            n_pairs=EXPECTED_SHAPE[0],
            height=EXPECTED_SHAPE[1],
            width=EXPECTED_SHAPE[2],
        ),
    }
    _atomic_json(out_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build_ledger(args.gt_cache.resolve(), args.chunk_dir.resolve(), args.out.resolve())
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_pairs": payload["n_pairs"],
                "total_flip_count": payload["total_flip_count"],
                "reused_existing_output": payload.get("reused_existing_output", False),
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
