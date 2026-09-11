# SPDX-License-Identifier: MIT
"""n600 residual census from retained shipped-public argmax, with no new scorer.

This independently checks the charter's premise while the scorer slot is
pending. It is cached CPU-advisory evidence, not the fresh-scoring deliverable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.ndimage import distance_transform_cdt, maximum_filter

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments.ddm_rbf1_boundary_probe import FIELD, GT, PINS, ROOT, atomic_json, fact, save_array
from experiments.ddm_rbf1_boundary_treatments import edges
from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage

ARGMAX = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/seg_final/argmax_n600.npy")
ARGMAX_SHA = "4784e33d7bedbc568aab0d6514acb54525d48a0581b97c89bd0002505612529a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", required=True, type=Path)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve():
        raise ValueError("wrong custody root")
    inputs = {"argmax": fact(ARGMAX), "tokens": fact(FIELD), "gt": fact(GT)}
    for name, expected in (("argmax", ARGMAX_SHA), ("tokens", PINS["field"]), ("gt", PINS["gt"])):
        if inputs[name]["sha256"] != expected:
            raise ValueError(f"input changed: {name}")
    assert_gt_lineage(GT, required=AUTHORITY_LINEAGE, instrument="ddm_rbf1_cached_residual")
    public = json.loads((REPO / ".omx/research/ddm_rlc5_20260910/PUBLIC_RESULT.json").read_text())
    if public["candidate_raw"]["sha256"] != PINS["raw"] or public["decoded_field"]["sha256"] != PINS["field"]:
        raise ValueError("cached residual no longer joins the move-44 public parse-back")
    predicted = np.load(ARGMAX, mmap_mode="r", allow_pickle=False)
    labels = torch.load(GT, map_location="cpu", weights_only=False)["seg"].numpy()
    tokens = np.memmap(FIELD, mode="r", dtype=np.uint8, shape=(600, 384, 512))
    if predicted.shape != labels.shape or labels.shape != tokens.shape:
        raise ValueError("not a full n600 field")
    rows = []
    confusion = np.zeros((5, 5), dtype=np.int64)
    class_population = np.zeros(5, dtype=np.int64)
    for pair in range(600):
        wrong = predicted[pair] != labels[pair]
        yy, xx = np.nonzero(wrong)
        boundary = edges(tokens[pair])
        distance = distance_transform_cdt(~boundary, metric="chessboard") if boundary.any() else np.full((384, 512), 512)
        neighbor = np.zeros(wrong.shape, dtype=bool)
        reverse_neighbor = np.zeros(wrong.shape, dtype=bool)
        for label in range(5):
            neighbor |= (labels[pair] == label) & maximum_filter(predicted[pair] == label, size=3, mode="nearest")
            reverse_neighbor |= (predicted[pair] == label) & maximum_filter(labels[pair] == label, size=3, mode="nearest")
        row = np.column_stack((np.full(len(yy), pair), yy, xx, labels[pair][wrong], predicted[pair][wrong], tokens[pair][wrong], tokens[pair][wrong] == labels[pair][wrong], distance[wrong], neighbor[wrong], reverse_neighbor[wrong])).astype(np.int32)
        rows.append(row)
        confusion += np.bincount(labels[pair][wrong] * 5 + predicted[pair][wrong], minlength=25).reshape(5, 5)
        class_population += np.bincount(labels[pair].ravel(), minlength=5)
        if (pair + 1) % 20 == 0:
            saved = save_array(ROOT / "cached_residual" / f"cells_{pair - 19:03d}_{pair + 1:03d}.npy", np.concatenate(rows[-20:]))
            atomic_json(ROOT / "cached_residual" / f"stage_{pair + 1:03d}.json", {"inputs": inputs, "artifact": saved, "completed_pairs": pair + 1})
            print(json.dumps({"stage": "cached_residual", "completed_pairs": pair + 1}), flush=True)
    cells = np.concatenate(rows)
    artifact = save_array(ROOT / "cached_residual/cells_n600.npy", cells)
    denominator = len(cells)
    error_by_class = np.bincount(cells[:, 3], minlength=5)
    result = {"axis": "[macOS-CPU advisory, cached shipped-public argmax; no new scorer]", "score_claim": False,
        "producer":fact(Path(__file__)), "argv":sys.argv,
        "git_sha":subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "packages":{"numpy":np.__version__, "scipy":scipy.__version__, "torch":torch.__version__},
        "randomness":"none; deterministic census of every pinned input position",
        "n": 600, "field_positions": int(tokens.size), "errors": denominator, "d_seg": denominator / tokens.size,
        "token_correct": int(cells[:, 6].sum()), "token_correct_fraction": float(cells[:, 6].mean()),
        "distance_to_token_edge_histogram": {str(int(d)):int(np.count_nonzero(cells[:, 7] == d)) for d in np.unique(cells[:, 7])},
        "within_one_pixel_of_token_edge": int((cells[:, 7] <= 1).sum()),
        "gt_class_in_predicted_3x3": int(cells[:, 8].sum()),
        "predicted_class_in_gt_3x3": int(cells[:, 9].sum()),
        "confusion_gt_rows_predicted_columns": confusion.tolist(), "class_population": class_population.tolist(),
        "error_by_gt_class": error_by_class.tolist(),
        "class_enrichment": ((error_by_class / denominator) / (class_population / tokens.size)).tolist(),
        "row_min":int(cells[:, 1].min()), "row_max":int(cells[:, 1].max()),
        "inputs":inputs, "artifact":artifact,
        "public_join_receipt":fact(REPO / ".omx/research/ddm_rlc5_20260910/PUBLIC_RESULT.json"),
        "cache_source_receipt":fact(ARGMAX.parent / "STEP0_RESULT.json"),
        "cache_limit":"Argmax producer provenance inherited from sj1; fresh frozen-CPU rescoring is still owed. All token correctness classifications recomputed against shipped subset6, not the STEP0 receipt's stale base token field."}
    atomic_json(ROOT / "CACHED_RESIDUAL.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"inputs", "artifact"}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
