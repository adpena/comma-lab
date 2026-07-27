#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Runner for the HOPE BN-capacity generator over the exact n600 measure (#725, arm hb1).

Produces, under one dated run directory:
  * ``hope_bn_capacity_table.json``       — per-unit/per-channel exact-kernel capacity table
  * ``hope_per_stratum_capacity_table.json`` — 16 pre-head channels x 37 pf2 buckets,
    composed with the exact rank-4 head into per-class-pair capacities
    (the FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK weighting)
  * ``hope_rg3_agreement_receipt.json``   — parity of the generator's fine-band selector
    against the 17 hand-derived RG3 Fisher-margin codebook rows (validation gate),
    plus capacity-refined proposals for the 9 blocked rows (advisory only)

Bulky rebuildable features go to the SSD tier per the disk-hygiene rule; all
JSON receipts are small and land in the repo research dir, SHA-pinned.

Everything here is ``[macOS-CPU frozen-scorer advisory]``: no score claims,
no rate columns (rate denominators must be measured coder bytes — none are
measured here), pointer untouched.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))

from tac.optimization.hope_bn_capacity import (  # noqa: E402
    EVIDENCE_AXIS,
    FISHER_FAMILY,
    MARGIN_F16_SHA256,
    MARGIN_SHAPE,
    RG3_ASSIGNMENT_SHA256,
    SCHEMA_AGREEMENT,
    V19C_BASE_SHA256,
    assemble_capacity_table,
    assemble_per_stratum_table,
    bucket_class_pair,
    fisher_fine_band,
    load_bucket_index,
    load_frozen_segnet,
    measure_exact_kernels,
    site_local_capacity_field,
)

MAIN_REPO = Path("/Users/adpena/Projects/pact")
SSD_TIER = Path("/Volumes/VertigoDataTier/pact")

RG3_DIR = ".omx/research/ddm_rg3_residual_family_productions_20260724T110418Z"
ASSIGNMENT = f"{RG3_DIR}/ddm_rg3_residual_family_assignment.json"
MS6_RECEIPT = f"{RG3_DIR}/ddm_ms6_receiver_support_measurement_receipt.json"
SUPPORT_SUMMARY = f"{RG3_DIR}/ddm_rg3_receiver_support_summary.json"
V19C = ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/ddm_v19c_final_n600.zip.receipt-bytes"
MARGIN_F16 = SSD_TIER / "lever_b_score_native_argmax_smoke_20260610/targets_n600/gt_segnet_margin.f16"
GT_N600 = MAIN_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
SEGNET = MAIN_REPO / "upstream/models/segnet.safetensors"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _pin(path: Path, expected: str | None, label: str) -> str:
    digest = _sha(path)
    if expected is not None and digest != expected:
        raise SystemExit(f"custody FAIL: {label} sha {digest} != expected {expected}")
    return digest


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return _sha(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    tag = args.run_tag or _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = WORKTREE / f".omx/research/ddm_hb1_hope_bn_capacity_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ssd_dir = SSD_TIER / f"ddm_hb1_hope_bn_capacity_{tag}"
    ssd_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---- custody pins ----------------------------------------------------
    assignment_path = WORKTREE / ASSIGNMENT
    assignment_sha = _pin(assignment_path, RG3_ASSIGNMENT_SHA256, "rg3 assignment")
    v19c_path = WORKTREE / V19C
    v19c_sha = _pin(v19c_path, V19C_BASE_SHA256, "v19c base")
    margin_sha = _pin(MARGIN_F16, MARGIN_F16_SHA256, "margin f16")
    ms6_path = WORKTREE / MS6_RECEIPT
    ms6_sha = _sha(ms6_path)
    summary_path = WORKTREE / SUPPORT_SUMMARY
    summary_sha = _sha(summary_path)
    print(f"[{time.time() - t0:7.1f}s] custody pinned", flush=True)

    ms6 = json.loads(ms6_path.read_text())
    pf2_receipt = ms6["input_hash_lineage"]["pf2_event_index_receipt"]
    bucket_name_map = pf2_receipt["bucket_arrays"]
    pf2_index_path = Path(pf2_receipt["index_path"])
    pf2_sha = pf2_receipt["index_sha256"]

    assignment = json.loads(assignment_path.read_text())
    fisher_rows = [r for r in assignment["rows"] if r["selected_coordinate_family"] == FISHER_FAMILY]
    if len(fisher_rows) != 17:
        raise SystemExit(f"expected 17 Fisher-margin assignment rows, found {len(fisher_rows)}")

    summary = json.loads(summary_path.read_text())
    missing = {(int(b["pair_id"]), b["bucket_id"]) for b in summary["g3_top24_coverage"]["missing_blocks"]}
    blocked_fisher = [(int(r["pair_id"]), r["bucket_id"]) for r in fisher_rows if (int(r["pair_id"]), r["bucket_id"]) in missing]
    print(f"[{time.time() - t0:7.1f}s] 17 fisher rows, {len(blocked_fisher)} blocked per sealed summary", flush=True)

    # ---- exact-kernel measurement over n600 ------------------------------
    import torch

    torch.manual_seed(0)
    segnet = load_frozen_segnet(SEGNET)
    bucket_index = load_bucket_index(pf2_index_path, bucket_name_map, expected_sha256=pf2_sha)
    print(f"[{time.time() - t0:7.1f}s] segnet + {len(bucket_index)} buckets loaded", flush=True)

    gt = np.load(GT_N600)
    frames = gt["gt_f1"]
    lstars = gt["lstars"]
    print(f"[{time.time() - t0:7.1f}s] gt cache loaded {frames.shape}", flush=True)

    target_pairs = sorted({int(r["pair_id"]) for r in fisher_rows})
    result = measure_exact_kernels(
        segnet,
        frames,
        lstars,
        bucket_index,
        target_pairs=target_pairs,
        batch_size=args.batch_size,
        progress=lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True),
    )
    del frames, lstars, gt
    print(f"[{time.time() - t0:7.1f}s] measurement done; argmax agreement {result.argmax_agreement:.8f}", flush=True)
    if result.argmax_agreement < 0.999:
        raise SystemExit("custody FAIL: forward argmax does not reproduce cached GT argmax — measure is not exact")

    # ---- tables -----------------------------------------------------------
    capacity_table = assemble_capacity_table(segnet, result)
    bucket_pairs = {b: bucket_class_pair(b) for b in bucket_index}
    stratum_table = assemble_per_stratum_table(segnet, result, bucket_class_pairs=bucket_pairs)
    for table in (capacity_table, stratum_table):
        table["input_custody"] = {
            "segnet_safetensors_sha256": _pin(SEGNET, None, "segnet"),
            "gt_n600_npz_path": str(GT_N600),
            "pf2_event_index_sha256": pf2_sha,
            "rg3_assignment_sha256": assignment_sha,
            "run_tag": tag,
        }
    cap_sha = _write_json(run_dir / "hope_bn_capacity_table.json", capacity_table)
    strat_sha = _write_json(run_dir / "hope_per_stratum_capacity_table.json", stratum_table)
    print(f"[{time.time() - t0:7.1f}s] tables written {cap_sha[:12]} {strat_sha[:12]}", flush=True)

    feat_path = ssd_dir / "target_pair_prehead_features.npz"
    np.savez_compressed(feat_path, **{f"pair_{p:04d}": f for p, f in result.target_pair_features.items()})
    feat_sha = _sha(feat_path)

    # ---- validation gate: parity + refinement on the 17 rows --------------
    from tac.optimization import direct_description_coupled_margin as coupled
    from tac.optimization import direct_description_preuint8_channel as preuint8
    from tac.optimization.ddm_rg1_receiver_grammar import (
        _base_masks_for_classes,
        derive_rg3_fisher_margin_band,
    )
    from tac.optimization.direct_description_carrier_compose import receive_carrier_compose_archive

    archive = v19c_path.read_bytes()
    pre_members, _ = preuint8.parse_preuint8_q8_archive(archive)
    coupled_members, _ = coupled.parse_coupled_margin_archive(pre_members[preuint8.BASE_MEMBER])
    base = receive_carrier_compose_archive(coupled_members[coupled.BASE_MEMBER], verify_member_effects=False)
    margin = np.memmap(MARGIN_F16, dtype=np.float16, mode="r", shape=MARGIN_SHAPE)
    print(f"[{time.time() - t0:7.1f}s] v19c receiver + margin field open", flush=True)

    stratum_by_bucket = {row["bucket_id"]: row for row in stratum_table["strata"]}
    agreement_rows = []
    n_match = 0
    n_ref_match = 0
    for row in sorted(fisher_rows, key=lambda r: (int(r["pair_id"]), r["bucket_id"])):
        pair = int(row["pair_id"])
        bucket = row["bucket_id"]
        class_a, class_b = (int(c) for c in row["typed_key"]["class_ids"])
        row_band = int(row["receiver_derived_row_band"])
        recorded = int(row["receiver_derived_fine_band"])
        margin_map = np.asarray(margin[pair], dtype=np.float32)
        mask_a, mask_b = _base_masks_for_classes(base, source_pair_id=pair, class_a=class_a, class_b=class_b)
        support = mask_a | mask_b

        mine = fisher_fine_band(margin_map, support, row_band=row_band)
        reference = derive_rg3_fisher_margin_band(
            base, pair_index=pair, class_a=class_a, class_b=class_b, row_band=row_band, margin_map=margin_map
        )
        cap_vec = np.asarray(stratum_by_bucket[bucket]["capacity_per_channel"], dtype=np.float64)
        weight = site_local_capacity_field(result.target_pair_features[pair], cap_vec)
        refined = fisher_fine_band(margin_map, support, row_band=row_band, site_weight=weight)

        match = mine == recorded and reference == recorded
        n_match += int(match)
        n_ref_match += int(refined == recorded)
        agreement_rows.append(
            {
                "pair_id": pair,
                "bucket_id": bucket,
                "class_pair": f"{class_a}-{class_b}",
                "row_band": row_band,
                "recorded_fine_band": recorded,
                "generator_parity_fine_band": mine,
                "in_repo_reference_fine_band": reference,
                "parity_match": bool(match),
                "capacity_refined_fine_band": refined,
                "refined_equals_recorded": bool(refined == recorded),
                "blocked_in_sealed_sweep": (pair, bucket) in missing,
                "refinement_status": (
                    "REFINEMENT_PROPOSAL_UNMEASURED" if refined != recorded else "REFINEMENT_CONFIRMS_PARITY"
                ),
            }
        )
        print(
            f"[{time.time() - t0:7.1f}s] pair {pair:3d} {bucket:45s} rec {recorded} mine {mine} ref {reference} refined {refined}",
            flush=True,
        )

    receipt = {
        "schema": SCHEMA_AGREEMENT,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "pointer_moved": False,
        "coordinate_family": "FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK",
        "validation_gate": {
            "required": "reproduce-or-refine the 17 hand-derived RG3 Fisher-margin codebook rows",
            "rows_total": len(agreement_rows),
            "parity_matches": n_match,
            "parity_verdict": "REPRODUCED_17_OF_17" if n_match == len(agreement_rows) else f"PARTIAL_{n_match}_OF_{len(agreement_rows)}",
            "capacity_refined_equal_to_recorded": n_ref_match,
            "blocked_rows_with_refined_proposals": [
                {k: r[k] for k in ("pair_id", "bucket_id", "recorded_fine_band", "capacity_refined_fine_band")}
                for r in agreement_rows
                if r["blocked_in_sealed_sweep"]
            ],
        },
        "rows": agreement_rows,
        "input_custody": {
            "rg3_assignment_sha256": assignment_sha,
            "rg3_support_summary_sha256": summary_sha,
            "ms6_receipt_sha256": ms6_sha,
            "v19c_base_sha256": v19c_sha,
            "margin_f16_sha256": margin_sha,
            "pf2_event_index_sha256": pf2_sha,
            "capacity_table_sha256": cap_sha,
            "per_stratum_table_sha256": strat_sha,
            "target_pair_features_npz": {"path": str(feat_path), "sha256": feat_sha},
        },
        "rate_denominator_policy": "no rate columns; score_units_per_byte_status=OWED_NOT_ADMITTED",
        "run_tag": tag,
        "wallclock_seconds": time.time() - t0,
    }
    receipt_sha = _write_json(run_dir / "hope_rg3_agreement_receipt.json", receipt)
    print(f"[{time.time() - t0:7.1f}s] agreement receipt {receipt_sha[:16]} parity {n_match}/{len(agreement_rows)}", flush=True)
    return 0 if n_match == len(agreement_rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
