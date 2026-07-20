#!/usr/bin/env python3
"""Emit the hash-bound R1b5 Fisher ordering and resize-coupling audit.

The large ranked field remains on the SSD evidence tier as Brotli-compressed
canonical JSONL.  A small canonical receipt and an augmented non-authorizing
candidate receipt are emitted beside it.  This is encoder-side audit work;
there is no scorer weight or per-video table in any receiver archive.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.power_diagram_witness import read_frozen_segmentation_head  # noqa: E402
from tac.optimization.r1b5_fisher_ev import (  # noqa: E402
    EXPECTED_MODERATE_R2B,
    EXPECTED_PDW1_CANDIDATES,
    R1B5FisherEVError,
    head_pair_norm_table,
    rank_pdw1_candidates,
    support_overlap_component_histogram,
)

SCHEMA = "r1b5_fisher_ev_and_resize_coupling_audit.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise R1B5FisherEVError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def load_sidecars(manifests: list[Path]) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]]]:
    selected: dict[int, dict[str, np.ndarray]] = {}
    custody: list[dict[str, Any]] = []
    for path in manifests:
        resolved = path.expanduser().resolve(strict=True)
        raw = resolved.read_bytes()
        manifest = json.loads(raw)
        if manifest.get("schema") != "vjp_custody_manifest.v1":
            raise R1B5FisherEVError(f"VJP manifest schema mismatch: {resolved}")
        used: list[int] = []
        for row in manifest.get("sidecars", []):
            pair = int(row["pair_id"])
            if pair >= 24:
                continue
            if pair in selected:
                raise R1B5FisherEVError(f"duplicate selected VJP pair {pair}")
            sidecar = Path(row["path"]).expanduser().resolve(strict=True)
            if sidecar.stat().st_size != row.get("bytes") or sha256_file(sidecar) != row.get("sha256"):
                raise R1B5FisherEVError(f"VJP sidecar byte custody mismatch for pair {pair}")
            with np.load(sidecar) as archive:
                selected[pair] = {
                    name: np.array(archive[name], copy=True)
                    for name in (
                        "winner",
                        "rival",
                        "cached_margin",
                        "seg_q",
                        "seg_local_lipschitz",
                    )
                }
            used.append(pair)
        custody.append(
            {
                "path": str(resolved),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "selected_pair_ids": used,
            }
        )
    if sorted(selected) != list(range(24)):
        raise R1B5FisherEVError(f"VJP n24 coverage mismatch: {sorted(selected)}")
    return selected, custody


def load_moderate_cells(stage_dir: Path) -> tuple[list[tuple[int, int, int]], list[dict[str, Any]]]:
    cells: list[tuple[int, int, int]] = []
    custody: list[dict[str, Any]] = []
    for path in sorted(stage_dir.expanduser().resolve(strict=True).glob("batch-*.json")):
        raw = path.read_bytes()
        payload = json.loads(raw)
        for pair, row, col, _target, _pred, margin in payload.get("flips", []):
            if 1e-3 <= abs(float(margin)) < 1.0:
                cells.append((int(pair), int(row), int(col)))
        custody.append(
            {
                "path": str(path),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if len(cells) != EXPECTED_MODERATE_R2B:
        raise R1B5FisherEVError(f"moderate R2b population {len(cells)} != {EXPECTED_MODERATE_R2B}")
    return cells, custody


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--gt-cache", type=Path, required=True)
    result.add_argument("--hard-pred", type=Path, required=True)
    result.add_argument("--pdw1-receipt", type=Path, required=True)
    result.add_argument("--vjp-manifest", type=Path, action="append", required=True)
    result.add_argument("--head-weights", type=Path, required=True)
    result.add_argument("--r2b-stage-dir", type=Path, required=True)
    result.add_argument("--candidate-receipt", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def execute(args: argparse.Namespace) -> int:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise R1B5FisherEVError(f"output directory already exists: {output}")
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise R1B5FisherEVError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    gt_path = args.gt_cache.expanduser().resolve(strict=True)
    pred_path = args.hard_pred.expanduser().resolve(strict=True)
    pdw1_path = args.pdw1_receipt.expanduser().resolve(strict=True)
    pdw1_raw = pdw1_path.read_bytes()
    pdw1 = json.loads(pdw1_raw)
    point = pdw1.get("phase_c_first_inbox_point", {})
    pred_custody = point.get("bulk_custody", {}).get("hard_oracle_pred", {})
    if (
        pdw1.get("schema") != "pdw1_fp32_realization_first_inbox_point.v1"
        or point.get("d_B_hard_oracle_vs_stored_labels_mismatch_px") != EXPECTED_PDW1_CANDIDATES
        or Path(str(pred_custody.get("path", ""))).expanduser().resolve(strict=True) != pred_path
        or pred_custody.get("sha256") != sha256_file(pred_path)
    ):
        raise R1B5FisherEVError("PDW1 candidate-population custody mismatch")
    candidate_path = args.candidate_receipt.expanduser().resolve(strict=True)
    candidate_raw = candidate_path.read_bytes()
    candidate = json.loads(candidate_raw)
    if candidate.get("schema") != "r2b_sparse_target_selection_receipt.v1":
        raise R1B5FisherEVError("candidate receipt schema mismatch")
    with np.load(gt_path, mmap_mode="r") as gt:
        labels = np.asarray(gt["lstars"][:24])
    hard_prediction = np.load(pred_path, mmap_mode="r")
    sidecars, vjp_custody = load_sidecars(args.vjp_manifest)
    head_path = args.head_weights.expanduser().resolve(strict=True)
    weight, _bias = read_frozen_segmentation_head(head_path)
    rows, ranking = rank_pdw1_candidates(
        labels=labels,
        hard_prediction=hard_prediction,
        sidecars=sidecars,
        head_pair_norm_table=head_pair_norm_table(weight),
    )
    if len(rows) != EXPECTED_PDW1_CANDIDATES:
        raise AssertionError("sealed candidate population changed after ranking")
    columns = ranking.pop("rank_columns")
    header = canonical_json(
        {
            "schema": "r1b5_fisher_ev_ordering_jsonl.v1",
            "columns": columns,
            "candidate_count": len(rows),
        }
    )
    ordering_raw = b"\n".join([header, *(canonical_json(row) for row in rows)]) + b"\n"
    ordering_path = output / "fisher_ev_ordering_38077.jsonl.br"
    staged_ordering = staging / ordering_path.name
    atomic_write(staged_ordering, brotli.compress(ordering_raw, quality=11))

    moderate_cells, r2b_custody = load_moderate_cells(args.r2b_stage_dir)
    coupling = support_overlap_component_histogram(moderate_cells)
    coupling.update(
        {
            "population": "R2b_baseline_moderate_margin_[1e-3,1)",
            "surrogate_grouping_used": False,
            "group_vs_singleton_covariance_audit": (
                "NOT_APPLICABLE_EXACT_SINGLETON_LOCAL_RESIZE_PARTITION"
                if coupling["non_singleton_component_count"] == 0
                else "REQUIRED_BEFORE_GROUP_SURROGATE_ADMISSION"
            ),
        }
    )
    ordering_custody = {
        "path": str(ordering_path),
        "bytes": staged_ordering.stat().st_size,
        "sha256": sha256_file(staged_ordering),
        "uncompressed_bytes": len(ordering_raw),
    }
    receipt = {
        "schema": SCHEMA,
        "authority": "[macOS-CPU advisory] ENCODER_SIDE_CUSTODY_NOT_SCORE_AUTHORITY",
        "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
        "score_claim": False,
        "promotion_eligible": False,
        "populations": {
            "fisher_ordering": "PDW1_n24_realization_mismatches_exact_38077",
            "coupling_partition": "R2b_n600_moderate_margin_exact_16319",
            "populations_are_not_interchangeable": True,
        },
        "ranking": ranking,
        "ordering_artifact": ordering_custody,
        "coupling_operator_audit": coupling,
        "lens_contract": {
            "hyperplanes": "public_frozen_rank4_head_target_realized_norms",
            "metric": "fisher_trace_0.5_sech2_margin_over_2",
            "pair_dependent_vjp": "per_pair_native_activation_pullback_not_reused",
            "resize": "exact_align_corners_false_half_pixel_supports",
            "intrinsic_structures": "38077_compact_rows_plus_16319_cell_union_find_no_ambient_mask",
            "kernel": "ker_A_receives_zero_fidelity_priority",
            "receiver_boundary": "no_scorer_weights_or_video_tables_added_to_decode",
        },
        "conditioning": {
            "seg_solve_precedes_pose_xi0": True,
            "joint_objective": "MAX_MIN_MARGIN_CHEBYSHEV_CENTER_REQUIRED_DOWNSTREAM",
        },
        "blockers": [
            "REALIZED_BACKBONE_COMPONENT_SECANTS_ABSENT",
            "PER_CANDIDATE_EXACT_PREFIX_BYTE_MARGINAL_ABSENT",
        ],
        "inputs": {
            "gt_cache": {"path": str(gt_path), "bytes": gt_path.stat().st_size, "sha256": sha256_file(gt_path)},
            "hard_prediction": {
                "path": str(pred_path),
                "bytes": pred_path.stat().st_size,
                "sha256": sha256_file(pred_path),
            },
            "pdw1_receipt": {
                "path": str(pdw1_path),
                "bytes": len(pdw1_raw),
                "sha256": hashlib.sha256(pdw1_raw).hexdigest(),
            },
            "head_weights": {
                "path": str(head_path),
                "bytes": head_path.stat().st_size,
                "sha256": sha256_file(head_path),
            },
            "vjp_manifests": vjp_custody,
            "r2b_batches": r2b_custody,
            "candidate_receipt": {
                "path": str(candidate_path),
                "bytes": len(candidate_raw),
                "sha256": hashlib.sha256(candidate_raw).hexdigest(),
            },
        },
    }
    receipt_path = output / "receipt.json"
    staged_receipt = staging / receipt_path.name
    atomic_write(staged_receipt, canonical_json(receipt) + b"\n")

    candidate["r1b5_fisher_ev_ordering_advisory"] = {
        "candidate_count": EXPECTED_PDW1_CANDIDATES,
        "metric": "fisher_top1_top2_margin",
        "policy": "measured_reverse_waterfill_highest_ev_first",
        "artifact": {"path": str(ordering_path), "sha256": ordering_custody["sha256"]},
        "audit_receipt": {"path": str(receipt_path), "sha256": sha256_file(staged_receipt)},
        "marginal_admission_blocked": True,
    }
    candidate["score_claim"] = False
    candidate["promotion_eligible"] = False
    augmented = output / "candidate_receipt_with_fisher_ev.json"
    atomic_write(staging / augmented.name, canonical_json(candidate) + b"\n")
    os.replace(staging, output)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "ordering": ordering_custody,
                "augmented_candidate_receipt": str(augmented),
                "component_histogram": coupling["component_size_histogram"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(execute(parser().parse_args()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"R1B5_FISHER_EV_REFUSED: {exc}") from exc


if __name__ == "__main__":
    main()
