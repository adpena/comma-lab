#!/usr/bin/env python3
"""Execute CB2's scorer-free inherited-state and step-2 mechanism gate.

This runner does not fit a new dictionary.  The CB2 charter requires an early
stop when the existing K=2,048 codebook-capacity allocation does not track
source class area.  The runner re-derives that allocation and per-class token
agreement from retained n600 tensors, verifies the retained G4 spatial-debt
field's coordinate join, and emits a typed stop receipt plus a sealed future
scorer order.  It never imports or runs a scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = Path(
    "/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5"
)
RC1_ROOT = Path("/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4")
RC1_RETAINED = RC1_ROOT / "retained"
RC1_CANDIDATE = RC1_RETAINED / "candidates/k2048_i3"
RC1_BASE256 = RC1_RETAINED / "candidates/k256_i3"
G4_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/"
    "ddm_g4_spatial_stationarity_n600_20260722T212138Z"
)
G4_ARRAYS = G4_ROOT / "stage_checkpoints/01_recurrence_arrays.npz"
RI1_RESULT = Path(
    "/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/"
    "advisory_r1/contest_auth_eval.json"
)

SHAPE = (600, 384, 512)
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
RC1_MEMO_SHA256 = "dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d"
RC1_MODULE_SHA256 = "6c2ea6f324ea32b21d8cc079bb327c6af97e283cc963ec610859f1f2b0cbfbc9"
RC1_RUNNER_SHA256 = "19a3f378cce0eebe47d4a68c029bf6975da0c0f74902975ccdcdac68c1717c54"
RC1_PAYLOAD_SHA256 = "eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164"
RC1_CODEBOOK_SHA256 = "d4e5f28b27bef4fca622108db92403942fe1d72470af40ebf11f8ceb308921bf"
RC1_ASSIGNMENTS_SHA256 = "34c4eaf615d8030a0afd877cc3c2f5896e1e4ed56e0460a8a7932d247f5f2053"
RC1_SHADOW_SHA256 = "6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a"
RC1_BASE256_CODEBOOK_SHA256 = "c8f1f7d7d2ba60a932e12af54b8d45a02d55c190dd04b09a9931bebe596c651c"
G4_ARRAYS_SHA256 = "dbc85e7a4f593ab9b7a7f4ed017dbb63a064cb681df806d0bb93277ae8f42451"
RI1_RESULT_SHA256 = "9d08795f9101a38c03f5b90e4081ced5fd112b15796af345a76789c168ed6425"

DX2_SCORE = 0.14821987563243377
DX2_BYTES = 180_368
DX2_D_SEG = 0.00020139
DX2_D_POSE = 0.00000637
SCORE_DENOMINATOR = 37_545_489
STRICT_TARGET = 0.12


class CB2Refusal(RuntimeError):
    """Raised when inherited custody or a declared measurement invariant drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CB2Refusal(f"required file is absent: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_file(path: Path, *, size: int | None = None, sha256: str | None = None) -> dict[str, Any]:
    record = file_record(path)
    if size is not None and record["bytes"] != size:
        raise CB2Refusal(f"byte drift for {path}: {record['bytes']} != {size}")
    if sha256 is not None and record["sha256"] != sha256:
        raise CB2Refusal(f"SHA-256 drift for {path}: {record['sha256']} != {sha256}")
    return record


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def matching_key_paths(value: Any, fragments: tuple[str, ...], prefix: tuple[str, ...] = ()) -> list[str]:
    """Return nested JSON key paths containing any requested lowercase fragment."""
    matches: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, str(key))
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in fragments):
                matches.append(".".join(path))
            matches.extend(matching_key_paths(child, fragments, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(matching_key_paths(child, fragments, (*prefix, str(index))))
    return matches


def checkpoint(root: Path, step: int, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = root / "checkpoints" / f"{step:02d}_{name}.json"
    row = {"schema": "ddm_cb2_stage_checkpoint.v1", "step": step, "stage": name, **payload}
    atomic_json(path, row)
    return file_record(path)


def storage_preflight(root: Path) -> dict[str, Any]:
    allowed = Path("/Volumes/APDataStore/pact").resolve()
    resolved = root.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise CB2Refusal("CB2 receipt root must be below /Volumes/APDataStore/pact")
    usage = shutil.disk_usage(allowed)
    required = 16 * 1024 * 1024
    reserve = 8 * 1024 * 1024 * 1024
    if usage.free < required + reserve:
        raise CB2Refusal(
            f"storage preflight refused: free={usage.free}, required={required}, reserve={reserve}"
        )
    receipt = {
        "schema": "ddm_cb2_storage_preflight.v1",
        "path": str(resolved),
        "free_bytes": usage.free,
        "required_bytes": required,
        "reserve_bytes": reserve,
        "passed": True,
        "payload_plan": "no refit or payload materialization unless the step-2 mechanism gate passes",
    }
    atomic_json(root / "STORAGE_PREFLIGHT.json", receipt)
    return receipt


def capacity_rows(codebook: np.ndarray, denominator: int) -> list[dict[str, Any]]:
    slot_counts = np.bincount(np.asarray(codebook).reshape(-1), minlength=5).astype(np.int64)
    containing = np.asarray(
        [np.count_nonzero(np.any(codebook == class_id, axis=1)) for class_id in range(5)],
        dtype=np.int64,
    )
    per_word_counts = np.stack(
        [np.count_nonzero(codebook == class_id, axis=1) for class_id in range(5)], axis=1
    )
    plurality = np.bincount(np.argmax(per_word_counts, axis=1), minlength=5).astype(np.int64)
    return [
        {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "codebook_slots": int(slot_counts[class_id]),
            "codebook_slot_denominator": int(denominator),
            "codebook_slot_share": float(slot_counts[class_id] / denominator),
            "codewords_containing_class": int(containing[class_id]),
            "codeword_denominator": int(codebook.shape[0]),
            "plurality_codewords_smallest_id_tie_break": int(plurality[class_id]),
        }
        for class_id in range(5)
    ]


def rederive_confusion(source_path: Path, decoded_path: Path) -> np.ndarray:
    source = np.memmap(source_path, dtype=np.uint8, mode="r", shape=SHAPE)
    decoded = np.memmap(decoded_path, dtype=np.uint8, mode="r", shape=SHAPE)
    confusion = np.zeros((5, 5), dtype=np.int64)
    for pair_index in range(SHAPE[0]):
        truth = np.asarray(source[pair_index], dtype=np.int64).reshape(-1)
        prediction = np.asarray(decoded[pair_index], dtype=np.int64).reshape(-1)
        if np.any(truth >= 5) or np.any(prediction >= 5):
            raise CB2Refusal("source or decoded token lies outside the five-class vocabulary")
        confusion += np.bincount(5 * truth + prediction, minlength=25).reshape(5, 5)
    return confusion


def per_class_agreement(confusion: np.ndarray) -> list[dict[str, Any]]:
    total_mismatches = int(confusion.sum() - np.trace(confusion))
    rows: list[dict[str, Any]] = []
    for class_id in range(5):
        true_count = int(confusion[class_id].sum())
        predicted_count = int(confusion[:, class_id].sum())
        correct = int(confusion[class_id, class_id])
        mismatches = true_count - correct
        union = true_count + predicted_count - correct
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "true_positions": true_count,
                "position_denominator": int(confusion.sum()),
                "area_share": true_count / int(confusion.sum()),
                "correct_positions": correct,
                "mismatched_positions": mismatches,
                "agreement_given_true_class": correct / true_count,
                "iou_proxy_not_score": correct / union,
                "share_of_all_token_mismatches": mismatches / total_mismatches,
                "predicted_positions": predicted_count,
            }
        )
    return rows


def score_arithmetic() -> dict[str, Any]:
    pose_term = math.sqrt(10.0 * DX2_D_POSE)
    distortion = 100.0 * DX2_D_SEG + pose_term
    recomputed = distortion + 25.0 * DX2_BYTES / SCORE_DENOMINATOR
    continuous_bytes = (STRICT_TARGET - distortion) * SCORE_DENOMINATOR / 25.0
    strict_integer_bytes = math.ceil(continuous_bytes) - 1
    rc1_rate = 25.0 * 113_006 / SCORE_DENOMINATOR
    dseg_ceiling = (STRICT_TARGET - rc1_rate - pose_term) / 100.0
    return {
        "dx2_score_recorded": DX2_SCORE,
        "dx2_score_recomputed": recomputed,
        "dx2_distortion_term": distortion,
        "dx2_pose_term": pose_term,
        "strict_sub012_continuous_byte_boundary": continuous_bytes,
        "strict_sub012_integer_archive_ceiling": strict_integer_bytes,
        "required_cut_from_dx2_bytes": DX2_BYTES - strict_integer_bytes,
        "rc1_shadow_archive_bytes": 113_006,
        "rc1_headroom_bytes": strict_integer_bytes - 113_006,
        "rc1_dseg_ceiling_at_fixed_dx2_pose": dseg_ceiling,
        "score_units_per_byte": 25.0 / SCORE_DENOMINATOR,
        "bytes_per_0p001_score": 0.001 * SCORE_DENOMINATOR / 25.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    storage = storage_preflight(args.out_dir)

    inherited = {
        "rc1_memo": require_file(
            REPO / ".omx/research/ddm_rc1_rate_crush_20260822.md", sha256=RC1_MEMO_SHA256
        ),
        "rc1_module": require_file(
            REPO / "src/tac/optimization/rc1_terminal_program_vq.py", sha256=RC1_MODULE_SHA256
        ),
        "rc1_runner": require_file(
            REPO / "experiments/ddm_rc1_rate_crush.py", sha256=RC1_RUNNER_SHA256
        ),
        "canonical_result": file_record(RC1_ROOT / "RESULT.json"),
        "candidate_result": file_record(RC1_CANDIDATE / "RESULT.json"),
        "base256_result": file_record(RC1_BASE256 / "RESULT.json"),
        "payload": require_file(
            RC1_CANDIDATE / "receiver/tokens.rc1v", size=59_884, sha256=RC1_PAYLOAD_SHA256
        ),
        "codebook": require_file(
            RC1_CANDIDATE / "model/codebook.u8", size=1_228_800, sha256=RC1_CODEBOOK_SHA256
        ),
        "base256_codebook": require_file(
            RC1_BASE256 / "model/codebook.u8",
            size=153_600,
            sha256=RC1_BASE256_CODEBOOK_SHA256,
        ),
        "assignments": require_file(
            RC1_CANDIDATE / "model/assignments.u16le",
            size=393_216,
            sha256=RC1_ASSIGNMENTS_SHA256,
        ),
        "shadow_archive": require_file(
            RC1_CANDIDATE / "shadow/archive.zip", size=113_006, sha256=RC1_SHADOW_SHA256
        ),
    }
    candidate_result = json.loads((RC1_CANDIDATE / "RESULT.json").read_text())
    section_bytes = {
        "codebook": candidate_result["codebook_winner"]["bytes"],
        "assignment_map": candidate_result["assignment_winner"]["bytes"],
        "header": candidate_result["rc1_payload"]["bytes"]
        - candidate_result["codebook_winner"]["bytes"]
        - candidate_result["assignment_winner"]["bytes"],
        "rc1_payload": candidate_result["rc1_payload"]["bytes"],
        "shadow_archive": candidate_result["shadow_archive"]["bytes"],
    }
    expected_sections = {
        "codebook": 48_920,
        "assignment_map": 10_900,
        "header": 64,
        "rc1_payload": 59_884,
        "shadow_archive": 113_006,
    }
    if section_bytes != expected_sections:
        raise CB2Refusal(f"RC1 section anatomy drifted: {section_bytes} != {expected_sections}")
    source_manifest = json.loads((RC1_RETAINED / "source_index/MANIFEST.json").read_text())
    for record in source_manifest["files"]:
        if file_record(Path(record["path"])) != record:
            raise CB2Refusal(f"RC1 source-index file drifted: {record['path']}")
    source_tokens = Path(source_manifest["source"]["path"])
    if file_record(source_tokens) != source_manifest["source"]:
        raise CB2Refusal("RC1 source token custody drifted")
    producer_copy = args.out_dir / "retained/producer_source" / Path(__file__).name
    source_bytes = Path(__file__).read_bytes()
    if producer_copy.exists() and producer_copy.read_bytes() != source_bytes:
        raise CB2Refusal("CB2 producer source changed inside an existing output directory")
    atomic_bytes(producer_copy, source_bytes)
    inherited_checkpoint = checkpoint(
        args.out_dir,
        1,
        "inherited_custody_complete",
        {
            "status": "complete",
            "inherited": inherited,
            "source_index_manifest": file_record(RC1_RETAINED / "source_index/MANIFEST.json"),
            "source_tokens": file_record(source_tokens),
            "section_bytes": section_bytes,
            "producer_source": file_record(producer_copy),
        },
    )

    unique_programs = int(source_manifest["unique_programs"])
    site_unique_ids = np.memmap(
        RC1_RETAINED / "source_index/site_unique_ids.u32",
        dtype="<u4",
        mode="r",
        shape=(SHAPE[1] * SHAPE[2],),
    )
    codebook = np.memmap(
        RC1_CANDIDATE / "model/codebook.u8", dtype=np.uint8, mode="r", shape=(2048, 600)
    )
    base256 = np.memmap(
        RC1_BASE256 / "model/codebook.u8", dtype=np.uint8, mode="r", shape=(256, 600)
    )
    base_keys = {bytes(row) for row in base256}
    base_mask = np.asarray([bytes(row) in base_keys for row in codebook], dtype=bool)
    if int(base_mask.sum()) != 256:
        raise CB2Refusal(f"K=2,048 no longer contains all 256 base codewords: {base_mask.sum()}")

    decoded_path = RC1_CANDIDATE / "receiver/decoded_tokens.u8"
    require_file(decoded_path, size=int(np.prod(SHAPE)))
    confusion = rederive_confusion(source_tokens, decoded_path)
    retained_confusion = np.asarray(
        candidate_result["diagnostics"]["confusion_true_rows_predicted_columns"], dtype=np.int64
    )
    if not np.array_equal(confusion, retained_confusion):
        raise CB2Refusal("independently re-derived confusion differs from RC1's retained result")
    base256_result = json.loads((RC1_BASE256 / "RESULT.json").read_text())
    base_decoded_path = RC1_BASE256 / "receiver/decoded_tokens.u8"
    require_file(base_decoded_path, size=int(np.prod(SHAPE)))
    base_confusion = rederive_confusion(source_tokens, base_decoded_path)
    retained_base_confusion = np.asarray(
        base256_result["diagnostics"]["confusion_true_rows_predicted_columns"], dtype=np.int64
    )
    if not np.array_equal(base_confusion, retained_base_confusion):
        raise CB2Refusal("independently re-derived K=256 confusion differs from its retained result")
    agreement_rows = per_class_agreement(confusion)
    total_capacity = capacity_rows(codebook, codebook.size)
    base_capacity = capacity_rows(np.asarray(codebook)[base_mask], int(base_mask.sum()) * SHAPE[0])
    added_capacity = capacity_rows(np.asarray(codebook)[~base_mask], int((~base_mask).sum()) * SHAPE[0])
    for total_row, agreement_row in zip(total_capacity, agreement_rows, strict=True):
        total_row["source_area_share"] = agreement_row["area_share"]
        total_row["capacity_share_over_area_share"] = (
            total_row["codebook_slot_share"] / agreement_row["area_share"]
        )
    base_mismatches = base_confusion.sum(axis=1) - np.diag(base_confusion)
    final_mismatches = confusion.sum(axis=1) - np.diag(confusion)
    removed_mismatches = base_mismatches - final_mismatches
    if np.any(removed_mismatches < 0) or int(removed_mismatches.sum()) <= 0:
        raise CB2Refusal("K=256 to K=2,048 mismatch reduction no longer closes")
    incremental_effect = [
        {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "k256_mismatches": int(base_mismatches[class_id]),
            "k2048_mismatches": int(final_mismatches[class_id]),
            "mismatches_removed": int(removed_mismatches[class_id]),
            "removed_mismatch_denominator": int(removed_mismatches.sum()),
            "share_of_all_removed_mismatches": float(
                removed_mismatches[class_id] / removed_mismatches.sum()
            ),
            "fraction_of_class_k256_mismatches_removed": float(
                removed_mismatches[class_id] / base_mismatches[class_id]
            ),
        }
        for class_id in range(5)
    ]
    area = np.asarray([row["area_share"] for row in agreement_rows], dtype=np.float64)
    capacity = np.asarray([row["codebook_slot_share"] for row in total_capacity], dtype=np.float64)
    total_variation = 0.5 * float(np.abs(area - capacity).sum())
    pearson = float(np.corrcoef(area, capacity)[0, 1])
    mechanism = {
        "prior_law": "existing K=2048 codebook capacity tracks source class area",
        "verdict": "REFUTED_AT_STEP_2_NO_REFIT",
        "verdict_scope": (
            "FORMULATION: raw class-valued time-slot allocation in RC1's retained K=2048 "
            "codebook versus the same retained DX2 token tensor's class-area distribution"
        ),
        "capacity_definition": (
            "one class-valued time slot in the counted 2048x600 raw codebook; compressed bytes "
            "remain jointly coded and are not falsely partitioned by class"
        ),
        "capacity_slot_denominator": int(codebook.size),
        "source_position_denominator": int(confusion.sum()),
        "distribution_total_variation": total_variation,
        "distribution_pearson": pearson,
        "class1_area_share": agreement_rows[1]["area_share"],
        "class1_capacity_share": total_capacity[1]["codebook_slot_share"],
        "class1_capacity_over_area": total_capacity[1]["capacity_share_over_area_share"],
        "class1_added_capacity_share": added_capacity[1]["codebook_slot_share"],
        "class1_share_of_k256_to_k2048_removed_mismatches": incremental_effect[1][
            "share_of_all_removed_mismatches"
        ],
        "stop_rule": (
            "The charter requires a terminal stop when capacity does not track area; no weighting "
            "design or refit is admissible after this gate fires."
        ),
    }
    decomposition_checkpoint = checkpoint(
        args.out_dir,
        2,
        "existing_fit_decomposition_complete",
        {
            "status": "complete",
            "axis": "[macOS-CPU scorer-free retained-token n600]",
            "confusion_true_rows_predicted_columns": confusion.tolist(),
            "base_k256_confusion_true_rows_predicted_columns": base_confusion.tolist(),
            "per_class_agreement_proxy_not_score": agreement_rows,
            "capacity_all_k2048": total_capacity,
            "capacity_base_k256": base_capacity,
            "capacity_increment_k256_to_k2048": added_capacity,
            "incremental_mismatch_reduction_k256_to_k2048": incremental_effect,
            "base_codewords_in_k2048": int(base_mask.sum()),
            "incremental_codewords": int((~base_mask).sum()),
            "mechanism_test": mechanism,
        },
    )

    g4_record = require_file(G4_ARRAYS, size=396_209, sha256=G4_ARRAYS_SHA256)
    with np.load(G4_ARRAYS) as arrays:
        flip_frequency = np.asarray(arrays["flip_frequency"], dtype=np.int64)
        transition_counts = np.asarray(arrays["transition_counts"], dtype=np.int64)
    if flip_frequency.shape != SHAPE[1:] or transition_counts.shape != (25, *SHAPE[1:]):
        raise CB2Refusal("G4 retained field geometry drifted")
    if site_unique_ids.size != flip_frequency.size or int(site_unique_ids.max()) >= unique_programs:
        raise CB2Refusal("G4-to-RC1 row-major site join is not total")
    program_flip_mass = np.bincount(
        np.asarray(site_unique_ids, dtype=np.int64),
        weights=flip_frequency.reshape(-1),
        minlength=unique_programs,
    )
    transitions = transition_counts.reshape(5, 5, *SHAPE[1:]).copy()
    for class_id in range(5):
        transitions[class_id, class_id] = 0
    target_class_flip_mass = transitions.sum(axis=(0, 2, 3))
    weighting_field = {
        "status": "RETAINED_FIELD_JOIN_EXISTS_BUT_NOT_CONSUMED_AFTER_STEP2_STOP",
        "field": g4_record,
        "field_axis": "[macOS-CPU frozen-scorer advisory, v12 vehicle]",
        "join_key": "row-major scorer-grid coordinate (y,x), shared 384x512 geometry",
        "joined_program_sites": int(site_unique_ids.size),
        "program_site_denominator": int(flip_frequency.size),
        "unique_programs_with_nonzero_g4_flip_mass": int(np.count_nonzero(program_flip_mass)),
        "unique_program_denominator": unique_programs,
        "g4_total_flip_mass": int(flip_frequency.sum()),
        "g4_target_class_flip_mass": target_class_flip_mass.astype(np.int64).tolist(),
        "g4_target_class_flip_share": (target_class_flip_mass / target_class_flip_mass.sum()).tolist(),
        "consumed_for_refit": False,
        "why_not_consumed": "the charter's step-2 stop rule fired before weighting design",
        "missing_current_calibration": (
            "G4 is a v12 predicted-vs-target field, not a current DX2/RI1 per-class scorer breakout; "
            "the spatial join is exact but vehicle transfer is unmeasured"
        ),
    }
    ri1_record = require_file(RI1_RESULT, sha256=RI1_RESULT_SHA256)
    ri1_result = json.loads(RI1_RESULT.read_text())
    breakout_key_paths = matching_key_paths(
        ri1_result,
        ("per_class", "confusion", "class_breakout", "breakout_by_class"),
    )
    ri1_calibration = {
        "status": "TERMINAL_AGGREGATE_ONLY_NO_PER_CLASS_BREAKOUT",
        "terminal_receipt": ri1_record,
        "terminal_receipt_consumed": True,
        "archive_sha256": ri1_result["provenance"]["archive_sha256"],
        "archive_bytes": ri1_result["archive_size_bytes"],
        "n_samples": ri1_result["n_samples"],
        "score_axis": ri1_result["score_axis"],
        "evidence_grade": ri1_result["evidence_grade"],
        "aggregate_d_seg": ri1_result["avg_segnet_dist"],
        "aggregate_d_pose": ri1_result["avg_posenet_dist"],
        "rank_or_kill_eligible": ri1_result["rank_or_kill_eligible"],
        "per_pair_distortion_retention": ri1_result["per_pair_distortion_retention"],
        "per_class_breakout_key_paths": breakout_key_paths,
        "per_class_breakout_present": bool(breakout_key_paths),
        "per_class_breakout_consumed": False,
        "why_no_per_class_calibration": (
            "RI1's terminal receipt contains only aggregate Seg/Pose distortion; its per-pair "
            "retention failed and no per-class or confusion key exists. The charter-required "
            "per-class calibration therefore does not exist to consume."
        ),
        "consumed_for_refit": False,
        "why_not_consumed_for_refit": "the mandatory step-2 stop fired before any weighting design",
    }
    weighting_checkpoint = checkpoint(
        args.out_dir,
        3,
        "weighting_field_join_checked",
        {
            "status": "complete",
            "weighting_field_provenance": weighting_field,
            "ri1_terminal_calibration": ri1_calibration,
        },
    )

    fire_order = {
        "schema": "ddm_cb2_sealed_scorer_fire_order.v1",
        "disposition": "QUEUED_WITH_FIRE_ORDER_BLOCKED_BY_STEP2_FALSIFIER",
        "owner": "MAIN scorer-lane owner",
        "consumer_store": str(args.out_dir / "main_fire"),
        "selected_candidate": None,
        "dispatch_argv": None,
        "axis_to_run": "[macOS-CPU advisory n600] followed by [contest-CUDA T4 n600] only on pass",
        "fire_trigger": (
            "A new arm, explicitly not justified by the refuted area-tracking premise, has a retained "
            "current-DX2 per-class scorer breakout, produced a fixed-K=2048 payload whose complete "
            "receiver archive is <=113006 B, retained every candidate and repeat, and MAIN owns the "
            "non-duplicated n600 scorer lane."
        ),
        "required_deciding_run": (
            "Compare baseline RC1 and the new fixed-K candidate through the same shipping RI1 full-RGB "
            "receiver on all 600 pairs; report per-class Seg confusion/d_seg, d_pose, exact archive bytes, "
            "receiver parse-back, repeat noise, and recomputed S. Agreement alone cannot admit it."
        ),
        "blocked_by": [
            "CB2 produced no refitted candidate because its mandatory step-2 premise was refuted",
            "RI1 is terminal but contains only aggregate distortion and no per-class breakout",
            "the only joined retained flip field is ancestor-v12 advisory rather than current-DX2 calibration",
            "the scorer lane is owned by MAIN and contended by RI1/NI1",
        ],
        "cb2_must_not_fire_now": True,
    }
    fire_order_path = args.out_dir / "SEALED_FIRE_ORDER.json"
    atomic_json(fire_order_path, fire_order)

    result = {
        "schema": "ddm_cb2_class_balanced_dictionary_result.v1",
        "axis": "[macOS-CPU scorer-free retained-token n600]",
        "score_claim": False,
        "frontier_moved": False,
        "verdict": "PRIOR_LAW_REFUTED_AT_STEP_2_NO_REFIT",
        "verdict_scope": mechanism["verdict_scope"],
        "inherited_custody": inherited,
        "section_bytes": section_bytes,
        "per_class_agreement_proxy_not_score": agreement_rows,
        "capacity_all_k2048": total_capacity,
        "capacity_base_k256": base_capacity,
        "capacity_increment_k256_to_k2048": added_capacity,
        "incremental_mismatch_reduction_k256_to_k2048": incremental_effect,
        "mechanism_test": mechanism,
        "weighting_field_provenance": weighting_field,
        "ri1_terminal_calibration": ri1_calibration,
        "refit": {
            "attempted": False,
            "reason": "mandatory charter stop at step 2",
            "materialized_payloads": [],
            "losing_variants": [],
            "archive_bytes": None,
        },
        "score_arithmetic": score_arithmetic(),
        "sealed_fire_order": file_record(fire_order_path),
        "storage_preflight": storage,
        "stage_checkpoints": [
            inherited_checkpoint,
            decomposition_checkpoint,
            weighting_checkpoint,
        ],
        "producer_source": file_record(producer_copy),
        "boundaries": [
            "No scorer, RGB render, Metal, MPS, CUDA, Modal, or upstream evaluator ran.",
            "Per-class token agreement and IoU are diagnostics, not d_seg or score evidence.",
            "Raw codebook slots measure representational capacity; jointly compressed codebook bytes are not partitioned by class.",
            "No refitted codebook, assignment map, payload, or archive was materialized after the step-2 stop.",
            "The G4 spatial join is total, but its v12 scorer field is not current-DX2 calibration.",
            "RI1 is terminal, but its retained result has no per-class breakout and is env-mismatch advisory.",
        ],
        "own_vehicle_frontier": {
            "score": DX2_SCORE,
            "archive_bytes": DX2_BYTES,
            "axis": "[contest-CUDA T4, n600]",
            "moved_by_cb2": False,
        },
    }
    result_path = args.out_dir / "RESULT.json"
    atomic_json(result_path, result)
    checkpoint(
        args.out_dir,
        4,
        "terminal_stop_sealed",
        {
            "status": "complete",
            "result": file_record(result_path),
            "sealed_fire_order": file_record(fire_order_path),
            "verdict": result["verdict"],
        },
    )
    print(json.dumps({"result": file_record(result_path), "verdict": result["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
