#!/usr/bin/env python3
"""QS4 scorer-free collateral map, support trim, and byte-closed candidate.

This runner consumes the exact retained CP135, candidate, and GT argmax fields.
It never invokes SegNet or Modal.  It assigns every harmful change to the one
edited proposal and nearest edited token, compiles a binary strict-support
variant through the actual HP3/RC64 path, re-races the receiver-consumed carrier
overlay, and seals one unchanged-worker fire order for MAIN.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_cp135_rate_compose as cp135
from experiments import ddm_jo1_joint_probability_object as jo1
from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b
from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs2_compensation_overlay_runtime as overlay_codec
from experiments import ddm_qs2_compensation_rate_rung as qs2
from experiments import ddm_qs3_saturation_compose as qs3

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qs4_20260813")
BASE_FIELD: Final = qs3.QS1_BASE_FIELD
CANDIDATE_FIELD: Final = qs3.QS1_CANDIDATE_FIELD
GT_FIELD: Final = qs3.QS1_GT_FIELD
GT_RECEIPT: Final = qs3.GT_ATTRIBUTED_RECEIPT
QS1_STORE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qs1_20260813")
QS1_FIELD_ROOT: Final = qs3.QS1_FIELDS
JS6_INDEX: Final = qs3.JS6_INDEX
BASE_FIELD_SHA256: Final = qs3.EXPECTED_BASE_SHA256
CANDIDATE_FIELD_SHA256: Final = qs3.EXPECTED_CANDIDATE_SHA256
GT_FIELD_SHA256: Final = qs3.EXPECTED_GT_SHA256
PAIR_COUNT: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
DENOMINATOR_PIXELS: Final = PAIR_COUNT * HEIGHT * WIDTH
CP135_BYTES: Final = 186_252
RATE_S_PER_BYTE: Final = 25.0 / 37_545_489
BREAKEVEN_FLIPS_PER_BYTE: Final = 0.785
B_PRIOR: Final = 108.0 / 189.0
POSE_S_PER_EDIT_CELL: Final = 4.922749063924693e-7
DISTANCE_EDGES: Final = np.asarray(
    [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, np.inf],
    dtype=np.float64,
)
EXPECTED_SELECTED: Final = {
    105: "js6_0000_9fbf75d81c43",
    176: "js6_0072_f790b6493122",
    178: "js6_0006_92685b3e3e44",
    517: "js6_0004_06fc74e20d9e",
    523: "js6_0001_da319a6b65d0",
    532: "js6_0118_83f376603d6e",
}
AXIS: Final = "[macOS-CPU scorer-free retained-field + exact byte/container analysis]"


class QS4Error(RuntimeError):
    """A retained input, causal partition, coder, or sealed gate differed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    qs1.atomic_json(path, value)
    return qs1.file_record(path)


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    retained = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    expected = 32 * 1024**3
    reserve = 8 * 1024**3
    required = max(0, expected - retained) + reserve
    free = shutil.disk_usage(output).free
    result = {
        "schema": "ddm_qs4_storage_preflight.v1",
        "tier": str(output.resolve()),
        "already_retained_bytes": retained,
        "expected_total_bytes": expected,
        "reserve_bytes": reserve,
        "required_free_bytes": required,
        "free_bytes": free,
        "passed": free >= required,
        "cleanup_policy": "certify-or-block; no generated payload deleted or moved",
    }
    retain_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise QS4Error("SSD storage preflight failed")
    return result


def source_preflight(output: Path) -> dict[str, Any]:
    sources = {
        "base_field": qs1.require_file(BASE_FIELD, expected_sha256=BASE_FIELD_SHA256),
        "candidate_field": qs1.require_file(
            CANDIDATE_FIELD, expected_sha256=CANDIDATE_FIELD_SHA256
        ),
        "gt_field": qs1.require_file(GT_FIELD, expected_sha256=GT_FIELD_SHA256),
        "gt_receipt": qs1.require_file(GT_RECEIPT),
        "js6_index": qs1.require_file(JS6_INDEX),
        "cp135_archive": qs1.require_file(
            qs1.CP135_ARCHIVE,
            expected_bytes=CP135_BYTES,
            expected_sha256=qs1.CP135_ARCHIVE_SHA256,
        ),
        "qs1_engine": qs1.require_file(Path(qs1.__file__).resolve()),
        "qs2_engine": qs1.require_file(Path(qs2.__file__).resolve()),
        "qs3_engine": qs1.require_file(Path(qs3.__file__).resolve()),
        "qs4_engine": qs1.require_file(Path(__file__).resolve()),
        "overlay": qs1.require_file(Path(overlay_codec.__file__).resolve()),
        "dispatcher": qs1.require_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker": qs1.require_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker": qs1.require_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
    }
    result = {
        "schema": "ddm_qs4_source_preflight.v1",
        "sources": sources,
        "seed": 135,
        "axis": AXIS,
        "resume_from": str(output.resolve()),
        "scorer_slot_owned": False,
        "segnet_rerun": False,
        "modal_fired": False,
        "passed": True,
    }
    retain_json(output / "checkpoints/stage_00_source_preflight.json", result)
    return result


def selected_qs1_rows() -> list[dict[str, Any]]:
    rows = []
    for pair, proposal_id in sorted(EXPECTED_SELECTED.items()):
        path = QS1_STORE / "retained/proposals" / proposal_id / "RESULT.json"
        row = json.loads(path.read_text())
        if int(row["pair"]) != pair or row["proposal_id"] != proposal_id:
            raise QS4Error("QS1 selected-row identity differs")
        row["result_record"] = qs1.file_record(path)
        rows.append(row)
    return rows


def _nearest_assignment(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask.shape != (HEIGHT, WIDTH) or not np.any(mask):
        raise QS4Error("nearest-site assignment requires a nonempty 384x512 mask")
    distances, indices = distance_transform_edt(~mask, return_indices=True)
    nearest_flat = indices[0].astype(np.int64) * WIDTH + indices[1].astype(np.int64)
    return distances, nearest_flat


def _candidate_logits(pair: int) -> tuple[np.ndarray, dict[str, Any]]:
    roots = QS1_FIELD_ROOT / "retained/scorer/candidate/batches"
    for receipt_path in sorted(roots.glob("batch_*/BATCH_RESULT.json")):
        receipt = json.loads(receipt_path.read_text())
        first, last = int(receipt["pair_start"]), int(receipt["pair_end"])
        if first <= pair < last:
            path = receipt_path.parent / "logits.float32.npy"
            logits = np.load(path, mmap_mode="r", allow_pickle=False)
            if logits.shape != (last - first, 5, HEIGHT, WIDTH):
                raise QS4Error("candidate logits geometry differs")
            return logits[pair - first], qs1.file_record(path)
    raise QS4Error(f"retained candidate logits do not cover pair {pair}")


def classify_changes(
    base: np.ndarray, candidate: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    changed = base != candidate
    beneficial = changed & (base != gt) & (candidate == gt)
    harmful = changed & (base == gt) & (candidate != gt)
    wrong = changed & (base != gt) & (candidate != gt)
    if int(np.count_nonzero(beneficial | harmful | wrong)) != int(np.count_nonzero(changed)):
        raise QS4Error("B/H/W do not partition changed cells")
    return changed, beneficial, harmful, wrong


def collateral_map(output: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_10_collateral_map.json"
    if checkpoint.is_file():
        prior = json.loads(checkpoint.read_text())
        qs1.require_file(
            Path(prior["harmful_pixels"]["path"]),
            expected_bytes=int(prior["harmful_pixels"]["bytes"]),
            expected_sha256=str(prior["harmful_pixels"]["sha256"]),
        )
        for row in prior["per_proposal"]:
            for key in ("support", "candidate_tokens", "site_attribution"):
                record = row[key]
                qs1.require_file(
                    Path(record["path"]),
                    expected_bytes=int(record["bytes"]),
                    expected_sha256=str(record["sha256"]),
                )
        return prior
    base = np.load(BASE_FIELD, mmap_mode="r", allow_pickle=False)
    candidate = np.load(CANDIDATE_FIELD, mmap_mode="r", allow_pickle=False)
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    base_spatial = np.memmap(
        jo1.BASE_SPATIAL,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, HEIGHT, WIDTH),
    )
    harmful_rows: list[dict[str, Any]] = []
    per_proposal: list[dict[str, Any]] = []
    for row in rows:
        pair = int(row["pair"])
        proposal_id = str(row["proposal_id"])
        proposal_root = qs1.JS6_BANK / "proposals" / proposal_id
        proposed = np.load(
            proposal_root / "candidate_tokens.uint8.npy", allow_pickle=False
        )
        edit_mask = proposed != np.asarray(base_spatial[pair])
        edit_distances, nearest_edit = _nearest_assignment(edit_mask)
        changed, beneficial, harmful, wrong = classify_changes(
            np.asarray(base[pair]), np.asarray(candidate[pair]), np.asarray(gt[pair])
        )
        benefit_distances, nearest_benefit = _nearest_assignment(beneficial)
        target_component = np.load(
            proposal_root / "target_component.bool.npy", allow_pickle=False
        )
        logits, logits_record = _candidate_logits(pair)
        edit_sites = np.flatnonzero(edit_mask.reshape(-1))
        site_ordinal = {int(site): ordinal for ordinal, site in enumerate(edit_sites.tolist())}
        nearest_edit_flat = nearest_edit.reshape(-1)
        per_site = []
        keep_sites = []
        for site in edit_sites.tolist():
            assigned = nearest_edit_flat == site
            b_count = int(np.count_nonzero(assigned & beneficial.reshape(-1)))
            h_count = int(np.count_nonzero(assigned & harmful.reshape(-1)))
            w_count = int(np.count_nonzero(assigned & wrong.reshape(-1)))
            keep = b_count > h_count and pair != 532
            if keep:
                keep_sites.append(site)
            per_site.append(
                {
                    "site_flat": site,
                    "y": site // WIDTH,
                    "x": site % WIDTH,
                    "B": b_count,
                    "H": h_count,
                    "W": w_count,
                    "net": b_count - h_count,
                    "strict_support_keep": keep,
                }
            )
        keep_mask = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        keep_mask.reshape(-1)[np.asarray(keep_sites, dtype=np.int64)] = True
        strict_tokens = np.asarray(base_spatial[pair]).copy()
        strict_tokens[keep_mask] = proposed[keep_mask]
        support_root = output / "retained/supports" / proposal_id
        keep_record = qs1.retain_npy(support_root / "strict_support.bool.npy", keep_mask)
        token_record = qs1.retain_npy(
            support_root / "candidate_tokens.uint8.npy", strict_tokens
        )
        site_payload = b"".join(
            (json.dumps(item, sort_keys=True) + "\n").encode() for item in per_site
        )
        site_record = qs1.retain_bytes(support_root / "site_attribution.jsonl", site_payload)
        for y, x in np.argwhere(harmful):
            values = np.sort(np.asarray(logits[:, y, x], dtype=np.float64))
            assigned_site = int(nearest_edit[y, x])
            assigned_benefit = int(nearest_benefit[y, x])
            harmful_rows.append(
                {
                    "pair": pair,
                    "proposal_id": proposal_id,
                    "y": int(y),
                    "x": int(x),
                    "base_class": int(base[pair, y, x]),
                    "candidate_class": int(candidate[pair, y, x]),
                    "gt_class": int(gt[pair, y, x]),
                    "distance_to_nearest_intended_target_B": float(benefit_distances[y, x]),
                    "nearest_intended_target_y": assigned_benefit // WIDTH,
                    "nearest_intended_target_x": assigned_benefit % WIDTH,
                    "same_edited_cell": bool(edit_mask[y, x]),
                    "cell_relation": "same-cell" if edit_mask[y, x] else "neighbor-cell",
                    "nearest_edit_distance": float(edit_distances[y, x]),
                    "nearest_edit_site_flat": assigned_site,
                    "nearest_edit_site_ordinal": site_ordinal[assigned_site],
                    "inside_target_component": bool(target_component[y, x]),
                    "base_margin": None,
                    "base_margin_status": (
                        "UNAVAILABLE_NO_RETAINED_BASE_LOGITS; scorer rerun forbidden by lane contract"
                    ),
                    "candidate_margin_after_edit": float(values[-1] - values[-2]),
                    "candidate_logits_source": logits_record,
                    "causal_assignment": (
                        "one-proposal temporal ownership plus nearest edited-token Voronoi cell; "
                        "spatial attribution is a scorer-free model, not an intervention proof"
                    ),
                }
            )
        strict_b = sum(item["B"] for item in per_site if item["strict_support_keep"])
        strict_h = sum(item["H"] for item in per_site if item["strict_support_keep"])
        strict_w = sum(item["W"] for item in per_site if item["strict_support_keep"])
        per_proposal.append(
            {
                "pair": pair,
                "proposal_id": proposal_id,
                "changed": int(np.count_nonzero(changed)),
                "B": int(np.count_nonzero(beneficial)),
                "H": int(np.count_nonzero(harmful)),
                "W": int(np.count_nonzero(wrong)),
                "same_cell_H": int(np.count_nonzero(harmful & edit_mask)),
                "neighbor_cell_H": int(np.count_nonzero(harmful & ~edit_mask)),
                "original_edit_sites": int(np.count_nonzero(edit_mask)),
                "strict_keep_sites": len(keep_sites),
                "strict_model_B": strict_b,
                "strict_model_H": strict_h,
                "strict_model_W": strict_w,
                "strict_model_net": strict_b - strict_h,
                "dropped_pair_532": pair == 532,
                "support": keep_record,
                "candidate_tokens": token_record,
                "site_attribution": site_record,
            }
        )
    harmful_payload = b"".join(
        (json.dumps(item, sort_keys=True) + "\n").encode() for item in harmful_rows
    )
    harmful_record = qs1.retain_bytes(
        output / "retained/collateral/harmful_pixels.jsonl", harmful_payload
    )
    totals = {
        "changed": sum(item["changed"] for item in per_proposal),
        "B": sum(item["B"] for item in per_proposal),
        "H": sum(item["H"] for item in per_proposal),
        "W": sum(item["W"] for item in per_proposal),
        "same_cell_H": sum(item["same_cell_H"] for item in per_proposal),
        "neighbor_cell_H": sum(item["neighbor_cell_H"] for item in per_proposal),
        "strict_model_B": sum(item["strict_model_B"] for item in per_proposal),
        "strict_model_H": sum(item["strict_model_H"] for item in per_proposal),
        "strict_model_W": sum(item["strict_model_W"] for item in per_proposal),
        "strict_keep_sites": sum(item["strict_keep_sites"] for item in per_proposal),
    }
    if (totals["changed"], totals["B"], totals["H"], totals["W"]) != (189, 108, 76, 5):
        raise QS4Error(f"collateral attribution totals differ: {totals}")
    if len(harmful_rows) != 76:
        raise QS4Error("harmful-pixel row denominator differs")
    result = {
        "schema": "ddm_qs4_collateral_map.v1",
        "axis": "[contest-CUDA T4 retained argmax fields; macOS-CPU spatial attribution] COMPONENT-ONLY",
        "selection_mode": "all 189 changed pixels on all six selected pairs",
        "harmful_pixel_denominator": len(harmful_rows),
        "totals": totals,
        "per_proposal": per_proposal,
        "harmful_pixels": harmful_record,
        "base_margin_availability": (
            "0/76: base logits were not retained; candidate post-edit margins are supplemental"
        ),
        "support_rule": "keep an edited token iff its nearest-site Voronoi cell has B > H; drop pair 532",
        "support_model_status": "UNVERIFIED_UNTIL_UNCHANGED_WORKER",
        "all_payloads_retained": True,
        "score_claim": False,
    }
    retain_json(output / "COLLATERAL_MAP.json", result)
    retain_json(checkpoint, result)
    return result


def _distance_bin(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, DISTANCE_EDGES[1:-1], right=False)


def calibrated_hazard(
    map_result: dict[str, Any], base: np.ndarray, gt: np.ndarray
) -> dict[str, Any]:
    harmful_counts = np.zeros(len(DISTANCE_EDGES) - 1, dtype=np.int64)
    correct_counts = np.zeros_like(harmful_counts)
    base_spatial = np.memmap(
        jo1.BASE_SPATIAL,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, HEIGHT, WIDTH),
    )
    candidate = np.load(CANDIDATE_FIELD, mmap_mode="r", allow_pickle=False)
    for row in map_result["per_proposal"]:
        pair = int(row["pair"])
        proposal_root = qs1.JS6_BANK / "proposals" / row["proposal_id"]
        proposed = np.load(proposal_root / "candidate_tokens.uint8.npy", allow_pickle=False)
        edit_mask = proposed != np.asarray(base_spatial[pair])
        distances, _ = _nearest_assignment(edit_mask)
        bins = _distance_bin(distances)
        correct = np.asarray(base[pair]) == np.asarray(gt[pair])
        harmful = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        # The exact harmful coordinates are retained in the map; reconstruct from fields.
        harmful = (base[pair] == gt[pair]) & (candidate[pair] != gt[pair])
        for index in range(len(harmful_counts)):
            correct_counts[index] += int(np.count_nonzero(correct & (bins == index)))
            harmful_counts[index] += int(np.count_nonzero(harmful & (bins == index)))
    hazards = np.divide(
        harmful_counts,
        correct_counts,
        out=np.zeros_like(harmful_counts, dtype=np.float64),
        where=correct_counts != 0,
    )
    return {
        "edges": [
            float(value) if np.isfinite(value) else "inf"
            for value in DISTANCE_EDGES
        ],
        "correct_denominators": correct_counts.astype(int).tolist(),
        "harmful_numerators": harmful_counts.astype(int).tolist(),
        "harmful_hazard": hazards.tolist(),
    }


def admission_metrics(
    *, expected_net_flips: float, kept_sites: int, bytes_per_pair: float
) -> dict[str, Any]:
    rate_s = bytes_per_pair * RATE_S_PER_BYTE
    pose_s = kept_sites * POSE_S_PER_EDIT_CELL
    bar = BREAKEVEN_FLIPS_PER_BYTE * (1.0 + pose_s / rate_s)
    flips_per_byte = expected_net_flips / bytes_per_pair
    return {
        "bytes_per_pair": bytes_per_pair,
        "expected_net_flips": expected_net_flips,
        "expected_flips_per_byte": flips_per_byte,
        "projected_pose_s": pose_s,
        "rate_s": rate_s,
        "pose_adjusted_admission_bar_flips_per_byte": bar,
        "passes": flips_per_byte > bar,
    }


def full_bank_screen(output: Path, map_result: dict[str, Any]) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_20_full_bank_screen.json"
    if checkpoint.is_file():
        prior = json.loads(checkpoint.read_text())
        screen = prior["screen_rows"]
        qs1.require_file(
            Path(screen["path"]),
            expected_bytes=int(screen["bytes"]),
            expected_sha256=str(screen["sha256"]),
        )
        return prior
    base = np.load(BASE_FIELD, mmap_mode="r", allow_pickle=False)
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    base_spatial = np.memmap(
        jo1.BASE_SPATIAL,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, HEIGHT, WIDTH),
    )
    hazard = calibrated_hazard(map_result, base, gt)
    hazard_values = np.asarray(hazard["harmful_hazard"], dtype=np.float64)
    calibration = {row["proposal_id"]: row for row in map_result["per_proposal"]}
    rows = qs3.load_js6_rows()
    screen = []
    for row in rows:
        pair = int(row["pair"])
        proposal_id = str(row["proposal_id"])
        proposal_root = qs1.JS6_BANK / "proposals" / proposal_id
        proposed = np.load(proposal_root / "candidate_tokens.uint8.npy", allow_pickle=False)
        edit_mask = proposed != np.asarray(base_spatial[pair])
        distances, nearest = _nearest_assignment(edit_mask)
        edit_sites = np.flatnonzero(edit_mask.reshape(-1))
        ordinal = np.full(HEIGHT * WIDTH, -1, dtype=np.int32)
        ordinal[edit_sites] = np.arange(edit_sites.size, dtype=np.int32)
        assigned = ordinal[nearest.reshape(-1)]
        target = np.load(proposal_root / "target_component.bool.npy", allow_pickle=False)
        target_counts = np.bincount(
            assigned[target.reshape(-1)], minlength=edit_sites.size
        ).astype(np.float64)
        target_mass = int(row["receiver_surface"]["exact_field_target_edge_mass_on_support"])
        benefit_total = target_mass * B_PRIOR
        benefit = (
            benefit_total * target_counts / target_counts.sum()
            if target_counts.sum() > 0
            else np.zeros(edit_sites.size, dtype=np.float64)
        )
        correct = np.asarray(base[pair]) == np.asarray(gt[pair])
        bins = _distance_bin(distances)
        harm_weights = hazard_values[bins].reshape(-1)
        harm = np.bincount(
            assigned[correct.reshape(-1)],
            weights=harm_weights[correct.reshape(-1)],
            minlength=edit_sites.size,
        )
        keep = benefit > harm
        expected_b = float(benefit[keep].sum())
        expected_h = float(harm[keep].sum())
        model_source = "distance-hazard projection"
        if proposal_id in calibration:
            measured = calibration[proposal_id]
            expected_b = float(measured["strict_model_B"])
            expected_h = float(measured["strict_model_H"])
            keep_count = int(measured["strict_keep_sites"])
            model_source = "exact retained-field nearest-site calibration"
        else:
            keep_count = int(np.count_nonzero(keep))
        forced_drop = pair == 532 and proposal_id == EXPECTED_SELECTED[532]
        if forced_drop:
            expected_b = expected_h = 0.0
            keep_count = 0
        primary = admission_metrics(
            expected_net_flips=expected_b - expected_h,
            kept_sites=keep_count,
            bytes_per_pair=5.7,
        )
        step2 = admission_metrics(
            expected_net_flips=expected_b - expected_h,
            kept_sites=keep_count,
            bytes_per_pair=5.0,
        )
        fully_admitted = bool(
            primary["passes"]
            and proposal_id in calibration
            and not forced_drop
            and keep_count > 0
        )
        screen.append(
            {
                "proposal_id": proposal_id,
                "pair": pair,
                "directed_edge": row["directed_edge"],
                "model_source": model_source,
                "target_mass": target_mass,
                "original_edit_sites": int(edit_sites.size),
                "modeled_keep_sites": keep_count,
                "expected_B": expected_b,
                "expected_H": expected_h,
                "expected_net_flips": expected_b - expected_h,
                "primary_5_7": primary,
                "deadzone_step2_5_0_projection": step2,
                "forced_drop_pair_532": forced_drop,
                "fully_admitted_for_compile": fully_admitted,
                "uncalibrated_pass_disposition": (
                    "QUEUE_FOR_EXACT_SCHUR_AND_RETAINED_FIELD_INTERVENTION"
                    if primary["passes"] and proposal_id not in calibration
                    else None
                ),
                "verdict_scope": (
                    "INSTANCE measured-nearest-site model" if proposal_id in calibration else
                    "TOY-BRACKET full-bank spatial projection"
                ),
            }
        )
    screen.sort(key=lambda item: (-item["primary_5_7"]["expected_flips_per_byte"], item["proposal_id"]))
    payload = b"".join(
        (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode()
        for item in screen
    )
    screen_record = qs1.retain_bytes(output / "retained/full_bank_screen.jsonl", payload)
    admitted = [item for item in screen if item["fully_admitted_for_compile"]]
    result = {
        "schema": "ddm_qs4_full_bank_corrected_screen.v1",
        "axis": AXIS,
        "selection_mode": "complete sealed 200-row JS6 bank; no prefix or sample",
        "bank_denominator": len(screen),
        "beneficial_rate_prior": B_PRIOR,
        "spatial_collateral_model": hazard,
        "primary_coding_price_bytes_per_pair": 5.7,
        "deadzone_step2_projection_bytes_per_active_pair": 5.0,
        "admission_law": "expected net flips/B > 0.785*(1+pose_S/rate_S)",
        "projected_primary_pass_count": sum(item["primary_5_7"]["passes"] for item in screen),
        "projected_step2_pass_count": sum(
            item["deadzone_step2_5_0_projection"]["passes"] for item in screen
        ),
        "fully_admitted_count": len(admitted),
        "fully_admitted_proposal_ids": [item["proposal_id"] for item in admitted],
        "fully_admitted_pairs": [item["pair"] for item in admitted],
        "screen_rows": screen_record,
        "uncalibrated_projection_is_not_admission": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "FULL_BANK_SCREEN.json", result)
    retain_json(checkpoint, result)
    return result


def _rate_sources_from_archive(archive: Path) -> tuple[bytes, bytes, bytes, bytes]:
    streams, suffix = qs2._split_member(qs2._zip_member(archive))
    cp135_streams, _ = qs2._split_member(qs2._zip_member(qs1.CP135_ARCHIVE))
    base_carrier = qs2._brotli_decompress(cp135_streams[2])
    if len(base_carrier) != 22_183:
        raise QS4Error("CP135 packed carrier source length differs")
    return streams[0], streams[1], base_carrier, suffix


def compile_candidate(
    output: Path,
    map_result: dict[str, Any],
    bank: dict[str, Any],
) -> dict[str, Any]:
    map_by_id = {row["proposal_id"]: row for row in map_result["per_proposal"]}
    qs1_by_id = {row["proposal_id"]: row for row in selected_qs1_rows()}
    selected = []
    for proposal_id in bank["fully_admitted_proposal_ids"]:
        if proposal_id not in qs1_by_id:
            raise QS4Error("full admission lacks an exact QS1 Schur solve")
        support = map_by_id[proposal_id]
        row = dict(qs1_by_id[proposal_id])
        row["candidate_tokens_path"] = support["candidate_tokens"]["path"]
        row["token_site_count"] = support["strict_keep_sites"]
        selected.append(row)
    selected.sort(key=lambda item: int(item["pair"]))
    if not selected:
        raise QS4Error("corrected screen admitted no exact-solved strict-support row")
    result_path = output / "candidate/COMPILE_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        expected_ids = [row["proposal_id"] for row in selected]
        if prior.get("selected_proposal_ids") != expected_ids:
            raise QS4Error("resumed compiled candidate selection differs")
        for key in ("archive", "archive_repeat", "candidate_codes"):
            record = prior[key]
            qs1.require_file(
                Path(record["path"]),
                expected_bytes=int(record["bytes"]),
                expected_sha256=str(record["sha256"]),
            )
        runtime_root = Path(prior["runtime_root"])
        if cp135.tree_record(runtime_root) != prior["runtime_tree"]:
            raise QS4Error("resumed candidate runtime tree differs")
        return prior
    primary = qs1._compile_one(output=output, selected=selected, repeat=False)
    repeated = qs1._compile_one(output=output, selected=selected, repeat=True)
    if primary["archive"]["sha256"] != repeated["archive"]["sha256"]:
        raise QS4Error("strict-support HP3/RC64 archive repeat differs")
    pairs, exact = qs2.exact_deltas(selected)
    sources = _rate_sources_from_archive(Path(primary["archive"]["path"]))
    winners = {}
    candidates = []
    for step in (1, 2):
        label = f"strict_support_deadzone_step_{step}"
        deltas = qs2.deadzone_quantize(exact, step)
        rows = [
            qs2.build_rate_candidate(
                output=output,
                label=label,
                pair_indices=pairs,
                deltas=deltas,
                carrier_quality=quality,
                sources=sources,
            )
            for quality in range(12)
        ]
        candidates.extend(rows)
        winners[label] = min(
            rows, key=lambda item: (item["archive"]["bytes"], item["carrier_quality"])
        )
    winner = winners["strict_support_deadzone_step_1"]
    root = output / "candidate"
    archive_payload = Path(winner["archive"]["path"]).read_bytes()
    archive_record = qs1.retain_bytes(root / "archive.zip", archive_payload)
    repeat_payload = qs2.deterministic_zip(qs2._zip_member(Path(winner["archive"]["path"])))
    repeat_record = qs1.retain_bytes(root / "archive.repeat.zip", repeat_payload)
    if archive_record["sha256"] != repeat_record["sha256"]:
        raise QS4Error("final deterministic archive repeat differs")
    runtime_root = root / "adapted_runtime"
    runtime_copy = jo1.copy_runtime(runtime_root, archive_payload)
    runtime_patches = qs2.patch_runtime(runtime_root)
    parseback = qs2.runtime_parseback(
        runtime_root=runtime_root,
        archive=runtime_root / "archive.zip",
        expected_overlay=winner["overlay"],
    )
    base_codes = qs1._load_cp135_carrier_codes()
    overlay_payload = Path(winner["overlay"]["path"]).read_bytes()
    actual_codes = overlay_codec.apply_compensation_overlay(base_codes, overlay_payload)
    expected_codes = base_codes.copy()
    expected_codes[pairs.astype(np.int64)] += exact
    if not np.array_equal(actual_codes, expected_codes):
        raise QS4Error("final receiver overlay differs from the exact QS1 code lattice")
    codes_record = qs1.retain_npy(root / "candidate_codes.int32.npy", actual_codes)
    strict_b = sum(map_by_id[row["proposal_id"]]["strict_model_B"] for row in selected)
    strict_h = sum(map_by_id[row["proposal_id"]]["strict_model_H"] for row in selected)
    strict_sites = sum(map_by_id[row["proposal_id"]]["strict_keep_sites"] for row in selected)
    projected_net = strict_b - strict_h
    delta_bytes = archive_record["bytes"] - CP135_BYTES
    seg_delta_s = -projected_net * 100.0 / DENOMINATOR_PIXELS
    pose_delta_s = strict_sites * POSE_S_PER_EDIT_CELL
    rate_delta_s = delta_bytes * RATE_S_PER_BYTE
    projected_total = seg_delta_s + pose_delta_s + rate_delta_s
    result = {
        "schema": "ddm_qs4_compiled_candidate.v1",
        "axis": AXIS,
        "selected_proposal_ids": [row["proposal_id"] for row in selected],
        "selected_pairs": [int(row["pair"]) for row in selected],
        "strict_support_model": {
            "B": strict_b,
            "H": strict_h,
            "net_flips": projected_net,
            "kept_token_sites": strict_sites,
            "status": "SPATIAL_MODEL_UNVERIFIED_UNTIL_UNCHANGED_WORKER",
        },
        "hp3_rc64_primary": primary,
        "hp3_rc64_repeat": repeated["archive"],
        "hp3_rc64_archive_repeat_byte_identical": True,
        "rate_race_candidate_denominator": len(candidates),
        "rate_race_winners": winners,
        "deadzone_step2_reraced": True,
        "chosen_rung": "strict_support_deadzone_step_1",
        "archive": archive_record,
        "archive_repeat": repeat_record,
        "archive_repeat_byte_identical": True,
        "archive_delta_bytes_vs_cp135": delta_bytes,
        "overlay": winner["overlay"],
        "candidate_codes": codes_record,
        "exact_qs1_code_lattice_reproduced": True,
        "strict_semantic_token_object": primary["archive"],
        "runtime_root": str(runtime_root.resolve()),
        "runtime_copy": runtime_copy,
        "runtime_patches": runtime_patches,
        "runtime_parseback": parseback,
        "runtime_tree": cp135.tree_record(runtime_root),
        "preworker_projection": {
            "seg_delta_s": seg_delta_s,
            "pose_delta_s": pose_delta_s,
            "rate_delta_s": rate_delta_s,
            "complete_delta_s": projected_total,
            "target_abs_delta_s_at_least": 1e-5,
            "target_cleared": projected_total <= -1e-5,
            "authority": "TOY-BRACKET projection; worker field and Pose vectors are required",
        },
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(result_path, result)
    retain_json(output / "checkpoints/stage_40_candidate_compile.json", result)
    return result


def seal_fire_order(output: Path, compiled: dict[str, Any]) -> dict[str, Any]:
    run_id = "ddm_qs4_dual_axis_20260813_r1"
    fire_root = output / "fire_order"
    input_root = fire_root / "fire_inputs"
    archive_path = Path(compiled["archive"]["path"])
    runtime_root = Path(compiled["runtime_root"])
    runtime_bundle, runtime_manifest = js1b.build_runtime_bundle(
        runtime_root, label="ddm_qs4_strict_support"
    )
    screen_payload = (
        json.dumps(compiled["preworker_projection"], indent=2, sort_keys=True) + "\n"
    ).encode()
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
        "POSE_SCREEN_RESULT.json": screen_payload,
    }
    for name, payload in payloads.items():
        qs1.retain_bytes(input_root / name, payload)
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": run_id,
        "resume_from": run_id,
        "lane_id": "ddm_qs4_collateral_suppression_n600_20260813",
        "instance_job_id": f"modal:{run_id}",
        "claim_agent": "MAIN",
        "seed": 1234,
        "batch_size": 16,
        "retain_pose_vectors": True,
        "candidate_archive": qs1.file_record(archive_path),
        "candidate_runtime": compiled["runtime_tree"],
        "runtime_manifest": runtime_manifest,
        "inputs": {name: js1b.payload_record(payload) for name, payload in payloads.items()},
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": hashlib.sha256(git_status).hexdigest(),
        "dispatcher_source_sha256": sha256_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    request_record = retain_json(fire_root / "SEALED_REQUEST.json", request)
    # Exercise the unchanged dispatcher's local request/input validator without firing.
    from experiments import ddm_qs1_modal_t4_dual_axis as dispatcher

    dispatcher.load_sealed_inputs(
        Path(request_record["path"]), input_root, str(request_record["sha256"])
    )
    command = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_qs1_modal_t4_dual_axis.py::main",
        "--sealed-request",
        request_record["path"],
        "--fire-input-dir",
        str(input_root.resolve()),
        "--expected-request-sha256",
        request_record["sha256"],
        "--output-dir",
        str((output / "dispatch" / run_id).resolve()),
        "--detach",
        "--provider-detach-ack",
    ]
    order = {
        "schema": "ddm_qs4_sealed_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": (
            "MAIN confirms no active n600 scorer lane, claims lane "
            "ddm_qs4_collateral_suppression_n600_20260813, verifies every sealed SHA, "
            "then executes exact_command_argv"
        ),
        "base_worker_family": "po1/pz4r CP135 base: 34970 flips, 6.885642960696714e-6 d_pose",
        "fresh_run_id": run_id,
        "request": request_record,
        "fire_inputs": str(input_root.resolve()),
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "remote_scope": (
            "one candidate; unchanged worker self-measures n600 T4 Seg field and official "
            "Pose first-six vectors with deterministic repeat"
        ),
        "canonical_evaluate_follow_on": "NOT_NAMED_UNTIL_WORKER_SUPER_BAND_RESULT",
        "post_harvest_rule": (
            "recompute complete delta S from worker flips, Pose vectors, and exact bytes; "
            "only if |delta S| >= 1e-5 and delta S < 0 may MAIN name a canonical evaluate.py fire order"
        ),
        "dispatcher_validation_passed": True,
        "modal_fired": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "SEALED_FIRE_ORDER.json", order)
    retain_json(output / "checkpoints/stage_50_sealed_fire_order.json", order)
    return order


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if output.resolve() != OUTPUT.resolve():
        raise QS4Error(f"output must be the governed SSD store: {OUTPUT}")
    storage_preflight(output)
    preflight = source_preflight(output)
    rows = selected_qs1_rows()
    receipt = qs3.consume_gt_attributed_receipt(output, rows)
    mapping = collateral_map(output, rows)
    bank = full_bank_screen(output, mapping)
    compiled = compile_candidate(output, mapping, bank)
    fire_order = seal_fire_order(output, compiled)
    result = {
        "schema": "ddm_qs4_final_result.v1",
        "axis": AXIS,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "source_preflight": preflight,
        "gt_receipt_verification": receipt,
        "collateral_map": mapping,
        "full_bank_screen": bank,
        "compiled_candidate": compiled,
        "fire_order": fire_order,
        "segnet_rerun": False,
        "modal_fired": False,
        "all_materialized_payloads_retained": True,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "FINAL_RESULT.json", result)
    retain_json(output / "checkpoints/stage_90_final.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_from.resolve() != args.output.resolve():
        raise QS4Error("--resume-from must equal --output")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "RUN.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QS4Error("another QS4 process holds the governed run lock") from error
        result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
