#!/usr/bin/env python3
"""Recover all shipped-field coder misses and census retained DALI geometry.

Scorer-free, n600 only. Each frame is a resumable retained stage. These are token
cells, never connected components or the renderer's residual-error population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_bnd2_causal_trace as trace
from experiments import ddm_bnd2_segment_encode as segment
from experiments import ddm_jg1_seg_solve as jg1
from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_tc1_mixer_codec as tc1

ROOT = trace.ROOT / "generation2/census"
N, H, W = 600, 384, 512
EXPECTED_MISSES, EXPECTED_FILTERED = 235044, 213733
# Producer provenance: ddm_chroma_dali_av/result_summary.json (Tesla T4 n600),
# independently pinned in ddm_ft1_identity_gate_and_caches.py:75-85.
GT_SHA = "a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994"
LOCATION_KEYS = ("frame", "pos", "sym", "best", "gt", "is_new", "on_gt_edge", "on_shipped_edge")


def edge_mask(plane: np.ndarray) -> np.ndarray:
    """Both cells incident to an unequal horizontal/vertical neighbour pair."""
    edge = np.zeros((H, W), dtype=bool)
    vertical = plane[:-1] != plane[1:]
    horizontal = plane[:, :-1] != plane[:, 1:]
    edge[:-1] |= vertical
    edge[1:] |= vertical
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    return edge


def share(numerator: int, denominator: int) -> dict:
    return {
        "level": "cell",
        "numerator": int(numerator),
        "denominator": int(denominator),
        "percentage": None if denominator == 0 else 100.0 * int(numerator) / int(denominator),
    }


def describe(values: dict, selection: np.ndarray) -> dict:
    """Count token cells; class matrices are not adjacent-class edge matrices."""
    count = int(selection.sum())
    row = values["pos"][selection] // W
    a = values["sym"][selection].astype(np.int64)
    b = values["best"][selection].astype(np.int64)
    gt = values["gt"][selection].astype(np.int64)
    return {
        "cells": count,
        "share_of_all_token_cells": share(count, N * H * W),
        "on_gt_edge": share(values["on_gt_edge"][selection].sum(), count),
        "on_shipped_token_edge": share(values["on_shipped_edge"][selection].sum(), count),
        "stored_token_equals_gt": share((a == gt).sum(), count),
        "coder_best_equals_gt": share((b == gt).sum(), count),
        "stored_token_to_coder_best": np.bincount(a * 5 + b, minlength=25).reshape(5, 5).tolist(),
        "gt_label_to_coder_best": np.bincount(gt * 5 + b, minlength=25).reshape(5, 5).tolist(),
        "matrix_denominator_cells": count,
        "matrix_semantics": "row label to column coder argmax; not a boundary class-pair census",
        "row_bands": [
            {"rows_half_open": [lo, hi], **share(((row >= lo) & (row < hi)).sum(), count)}
            for lo, hi in ((0, 128), (128, 192), (192, 256), (256, 320), (320, 384))
        ],
    }


def prepare() -> tuple[dict, dict]:
    """Pin source custody and DALI provenance before loading any target labels."""
    source = segment.complete_trace()
    trace.storage(ROOT / "INPUTS.json", 64 * 1024**2)
    lineage = jg1.up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=jg1.up2.LINEAGE_DALI)
    gt_fact = jg2.file_fact(jg1.DEFAULT_GT_DALI)
    summary_path = jg1.DEFAULT_GT_DALI.parent / "result_summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        gt_fact["sha256"] != GT_SHA
        or summary["dali_cache_sha256"] != GT_SHA
        or summary["dali_cache_bytes"] != gt_fact["bytes"]
        or summary["dali_pairs"] != N
        or summary["env"]["device_name"] != "Tesla T4"
        or not summary["coverage"]["ok"]
    ):
        raise ValueError("retained DALI cache does not match its n600 producer receipt")
    filtered = trace.blob(
        ROOT / "sources/filtered_candidates.npz", (trace.ROOT / "retained/filtered_candidates.npz").read_bytes()
    )
    summary_fact = trace.blob(ROOT / "sources/gt_result_summary.json", summary_path.read_bytes())
    binding = {
        "source_trace": jg2.file_fact(trace.ROOT / "generation2/trace/RESULT.json"),
        "source_field_sha256": trace.FIELD_SHA,
        "source_trace_binding": source["binding"],
        "archive_sha256": trace.ARCHIVE_SHA,
        "gt_cache": gt_fact,
        "gt_producer_receipt": summary_fact,
        "gt_lineage_gate": lineage,
        "filtered": filtered,
        "helpers": {name: jg2.file_fact(Path(module.__file__)) for name, module in (
            ("segment", segment), ("trace", trace), ("jg1", jg1),
            ("up2", jg1.up2), ("jg2", jg2), ("tc1", tc1),
        )},
        "producer": jg2.file_fact(Path(__file__)),
    }
    path = ROOT / "INPUTS.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("census resume inputs changed; preserve existing stages")
    trace.record(path, binding)
    with np.load(filtered["path"], allow_pickle=False) as data:
        old = {key: data[key].astype(np.int64) for key in ("frame", "pos", "sym", "best")}
    if any(v.shape != (EXPECTED_FILTERED,) for v in old.values()):
        raise ValueError("filtered candidate population is not the expected 213733")
    if (
        np.any((old["frame"] < 0) | (old["frame"] >= N))
        or np.any((old["pos"] < 0) | (old["pos"] >= H * W))
        or np.any((old["sym"] < 0) | (old["sym"] >= 5) | (old["best"] < 0) | (old["best"] >= 5))
        or np.any(old["sym"] == old["best"])
        or len(np.unique(old["frame"] * (H * W) + old["pos"])) != EXPECTED_FILTERED
    ):
        raise ValueError("filtered locations/classes are invalid or duplicated")
    return binding, old


def census() -> dict:
    binding, old = prepare()
    gt = jg1.load_gt_seg_labels(jg1.up2.LINEAGE_DALI)
    if gt.shape != (N, H, W) or gt.dtype != np.uint8 or np.any(gt >= 5):
        raise ValueError("DALI target labels have invalid shape/type/classes")
    if jg2.file_fact(jg1.DEFAULT_GT_DALI) != binding["gt_cache"]:
        raise ValueError("DALI cache changed while loading")
    field_digest = hashlib.sha256()
    locations = {key: [] for key in LOCATION_KEYS}
    frames = []
    gt_edge_cells = shipped_edge_cells = 0
    for frame in range(N):
        source_path = trace.ROOT / "generation2/trace/frames" / f"frame_{frame:04d}.npz"
        source_receipt = json.loads(source_path.with_suffix(".json").read_text())
        if source_receipt["binding"] != binding["source_trace_binding"]:
            raise ValueError("source frame belongs to another trace trajectory")
        data = segment.load_frame(frame)
        plane = data["tokens"]
        if plane.shape != (H, W) or plane.dtype != np.uint8 or np.any(plane >= 5):
            raise ValueError("invalid shipped token plane")
        plane_sha = hashlib.sha256(plane.tobytes()).hexdigest()
        if plane_sha != source_receipt["plane_sha256"]:
            raise ValueError("source plane hash differs from its receipt")
        field_digest.update(plane.tobytes())
        stage_path = ROOT / "frames" / f"frame_{frame:04d}.npz"
        receipt_path = stage_path.with_suffix(".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["binding"] != binding or receipt["source_payload"] != source_receipt["payload"]
                or jg2.file_fact(stage_path) != receipt["payload"]
            ):
                raise ValueError("census stage custody changed")
            with np.load(stage_path, allow_pickle=False) as saved:
                values = {key: saved[key] for key in saved.files}
        else:
            best = tc1.frequencies(data["rows"]).argmax(axis=1).astype(np.uint8)
            pos = np.flatnonzero(best != plane.reshape(-1)).astype(np.int32)
            if not np.array_equal(best, data["best"]) or not np.array_equal(pos, data["miss_pos"]):
                raise ValueError("recomputed integer-frequency misses differ from the trace")
            chosen = old["frame"] == frame
            old_pos = old["pos"][chosen]
            if (
                not np.isin(old_pos, pos).all()
                or not np.array_equal(plane.reshape(-1)[old_pos], old["sym"][chosen])
                or not np.array_equal(best[old_pos], old["best"][chosen])
            ):
                raise ValueError("old filtered locations are not an exact shipped-miss subset")
            gt_edge, shipped_edge = edge_mask(gt[frame]), edge_mask(plane)
            values = {
                "frame": np.full(len(pos), frame, dtype=np.uint16), "pos": pos,
                "sym": plane.reshape(-1)[pos], "best": best[pos], "gt": gt[frame].reshape(-1)[pos],
                "is_new": ~np.isin(pos, old_pos), "on_gt_edge": gt_edge.reshape(-1)[pos],
                "on_shipped_edge": shipped_edge.reshape(-1)[pos],
                "gt_edge_packed": np.packbits(gt_edge.reshape(-1), bitorder="little"),
                "shipped_edge_packed": np.packbits(shipped_edge.reshape(-1), bitorder="little"),
            }
            fact = trace.arrays(stage_path, values)
            receipt = trace.record(receipt_path, {
                "binding": binding, "source_payload": source_receipt["payload"], "payload": fact,
                "frame": frame, "shipped_plane_sha256": plane_sha,
                "gt_plane_sha256": hashlib.sha256(gt[frame].tobytes()).hexdigest(),
                "integer_argmax_and_old_subset_verified": True,
            })
        for key in LOCATION_KEYS:
            locations[key].append(values[key])
        gt_edge_cells += int(np.unpackbits(values["gt_edge_packed"], bitorder="little").sum())
        shipped_edge_cells += int(np.unpackbits(values["shipped_edge_packed"], bitorder="little").sum())
        frames.append({"frame": frame, **describe(values, np.ones(len(values["pos"]), dtype=bool)),
                       "recovered_cells": int(values["is_new"].sum()), "stage_receipt": str(receipt_path)})
        if (frame + 1) % 20 == 0:
            trace.record(ROOT / "LATEST.json", {"binding": binding, "completed_frames": frame + 1,
                                                 "last_stage_receipt": str(receipt_path)})
            print(json.dumps({"stage": "census", "frame": frame + 1}), flush=True)
    if field_digest.hexdigest() != trace.FIELD_SHA:
        raise ValueError("complete census does not belong to the shipped field")
    full = {key: np.concatenate(value) for key, value in locations.items()}
    recovered = full["is_new"]
    if len(full["pos"]) != EXPECTED_MISSES or int((~recovered).sum()) != EXPECTED_FILTERED:
        raise ValueError("full/recovered population differs from the charter's exact counts")
    if int(recovered.sum()) != EXPECTED_MISSES - EXPECTED_FILTERED:
        raise ValueError("recovered population is not 21311")
    payloads = {
        "complete_locations": trace.arrays(ROOT / "complete_locations.npz", full),
        "previously_missing_locations": trace.arrays(
            ROOT / "previously_missing_locations.npz", {key: value[recovered] for key, value in full.items()}
        ),
    }
    result = {
        "schema": "ddm_bnd2_complete_miss_census.v1", "binding": binding,
        "axis": "[macOS-CPU scorer-free census; retained DALI target lineage]", "score_claim": False,
        "frames": N, "token_cells": N * H * W, "source_field_sha256": field_digest.hexdigest(),
        "selection": "all 600 frames; every chosen integer-frequency argmax mismatch; no threshold",
        "argmax_tie_rule": "lowest class index, matching TC1/RP1 numpy argmax",
        "edge_definition": "cell differs from an in-image four-neighbour; both sides counted; border alone is not an edge",
        "full_population": describe(full, np.ones(EXPECTED_MISSES, dtype=bool)),
        "filtered_population": describe(full, ~recovered), "recovered_population": describe(full, recovered),
        "gt_edge_population": share(gt_edge_cells, N * H * W),
        "shipped_edge_population": share(shipped_edge_cells, N * H * W),
        "class_order": ["Road", "Lane", "Undrivable", "Movable", "MyCar"],
        "per_frame": frames, "payloads": payloads,
        "rendered_residual_join": {"measured": False, "recalled_shipped_residual_cells": 12614,
                                  "reason": "No hash-bound rendered residual map is an input to this producer; no pass-5 join or new 12614-cell census."},
        "retention": "all frame masks, complete/recovered locations and stages retained; no deletion or movement",
    }
    if jg2.file_fact(trace.ROOT / "generation2/trace/RESULT.json") != binding["source_trace"]:
        raise ValueError("source trace receipt changed during census")
    return trace.record(ROOT / "RESULT.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != trace.ROOT.resolve():
        raise ValueError("resume root outside owned arm store")
    result = census()
    print(json.dumps({"result": str(ROOT / "RESULT.json"), "payloads": result["payloads"]}, sort_keys=True))


if __name__ == "__main__":
    main()
