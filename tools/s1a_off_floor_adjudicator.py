#!/usr/bin/env python3
"""Adjudicate the two-seed S1A both-OFF floor from retained real artifacts.

This instrument is deliberately non-authoritative.  It reads the fixed,
evenly-strided n60 frozen-scorer advisory rows and the real stage-controller
allocation receipt, then applies the pre-registered GB1 renderer-corner
arithmetic.  It never emits a contest score or a population verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "ddm_s1e_off_floor_adjudicator.v1"
ROW_SCHEMA = "ddm_s1e_off_floor_checkpoint_row.v1"
EXPECTED_SEEDS = (20260815, 20260816)
EXPECTED_ENDPOINT_EPOCH = 65
EXPECTED_AXIS = "[Darwin-mps frozen-scorer advisory]"
EXPECTED_PAIR_IDS = tuple(range(0, 600, 10))
EXPECTED_N_PAIRS = len(EXPECTED_PAIR_IDS)

GB1_RENDERER_BYTES = 30_856
GB1_HARD_D_SEG = 0.00020139
GB1_D_POSE = 6.37e-6
RATE_NUMERATOR = 25.0
RATE_DENOMINATOR = 37_545_489.0
RATE_PER_BYTE = RATE_NUMERATOR / RATE_DENOMINATOR

CORNER_CROSSED = "CORNER_CROSSED_AT_LEAST_ONE_POINT"
ENTERED_AND_REFUSED = "ENTERED_AND_REFUSED_ALL_POINTS"
INCOMPLETE_DATA = "INCOMPLETE_DATA"
VERDICTS = frozenset({CORNER_CROSSED, ENTERED_AND_REFUSED, INCOMPLETE_DATA})

_EVALUATION_RE = re.compile(r"epoch_(\d{4})_n60\.json\Z")
_CHECKPOINT_RE = re.compile(r"wd3_epoch_(\d{4})\.pt\Z")


class AdjudicationError(RuntimeError):
    """The retained artifacts cannot support an unambiguous adjudication."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise AdjudicationError(f"artifact is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if size < 1:
        raise AdjudicationError(f"artifact is empty: {resolved}")
    return {"path": str(resolved), "bytes": size, "sha256": _sha256(resolved)}


def _sha256_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AdjudicationError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AdjudicationError(f"{field} must be hexadecimal") from exc
    return value.lower()


def _retained_record(
    value: Any,
    *,
    field: str,
    allowed_root: Path,
    verify_content_sha256: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdjudicationError(f"{field} must be a retained artifact record")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise AdjudicationError(f"{field}.path must be a nonempty string")
    try:
        path = Path(path_value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise AdjudicationError(f"{field} retained path is unavailable: {path_value}") from exc
    resolved_root = allowed_root.resolve(strict=True)
    if path != resolved_root and resolved_root not in path.parents:
        raise AdjudicationError(f"{field} retained path escapes seed store: {path}")
    if not path.is_file():
        raise AdjudicationError(f"{field} retained path is not a regular file: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes < 1:
        raise AdjudicationError(f"{field} retained artifact is empty: {path}")
    declared_bytes = _integer(value.get("bytes"), field=f"{field}.bytes", minimum=1)
    declared_sha = _sha256_string(value.get("sha256"), field=f"{field}.sha256")
    if observed_bytes != declared_bytes:
        raise AdjudicationError(
            f"{field} retained bytes mismatch: observed {observed_bytes} != declared {declared_bytes}"
        )
    if verify_content_sha256 and _sha256(path) != declared_sha:
        raise AdjudicationError(f"{field} retained SHA-256 mismatch at {path}")
    return {
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": declared_sha,
        "sha256_verification": (
            "live_content_verified"
            if verify_content_sha256
            else "producer_receipt_in_hashed_evaluation_plus_live_size_verified"
        ),
    }


def _load_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    try:
        raw = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"malformed JSON artifact: {record['path']}") from exc
    if not isinstance(raw, dict):
        raise AdjudicationError(f"JSON artifact must contain an object: {record['path']}")
    return raw, record


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdjudicationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise AdjudicationError(f"{field} must be >= {minimum}")
    return value


def _finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdjudicationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AdjudicationError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise AdjudicationError(f"{field} must be >= {minimum}")
    return result


def _controller_path(seed_root: Path) -> Path:
    controller_paths = sorted(
        path
        for path in (seed_root / "stage_controllers").glob("stage_*_from_epoch_*/STAGE_CONTROLLER_RESULT.json")
        if path.is_file() and not path.name.startswith("._")
    )
    if len(controller_paths) != 1:
        raise AdjudicationError(
            f"seed store must contain exactly one stage-controller result; "
            f"found {len(controller_paths)} under {seed_root}"
        )
    return controller_paths[0]


def _allocation_summary(controller: Mapping[str, Any], *, source: Path) -> dict[str, Any]:
    if controller.get("schema") != "ddm_wd3_stage_controller.v1":
        raise AdjudicationError(f"unexpected stage-controller schema at {source}")
    if controller.get("complete") is not True:
        raise AdjudicationError(f"stage controller is not complete at {source}")
    if controller.get("all_payloads_retained") is not True:
        raise AdjudicationError(f"stage controller did not retain all payloads at {source}")

    chosen = controller.get("chosen_allocation")
    ladder = controller.get("cheap_to_shrink_ladder")
    if not isinstance(chosen, dict) or not isinstance(ladder, dict):
        raise AdjudicationError(f"stage controller lacks chosen allocation/ladder at {source}")
    if chosen.get("schema") != "ddm_wd3_adaptive_quant_allocation.v1":
        raise AdjudicationError(f"unexpected chosen-allocation schema at {source}")
    policy = chosen.get("policy")
    selection_sha256 = chosen.get("selection_sha256")
    chosen_sha256 = controller.get("chosen_allocation_sha256")
    if not isinstance(policy, str) or not policy:
        raise AdjudicationError(f"chosen allocation has no policy at {source}")
    selection_sha256 = _sha256_string(
        selection_sha256,
        field=f"chosen_allocation.selection_sha256 at {source}",
    )
    chosen_sha256 = _sha256_string(
        chosen_sha256,
        field=f"chosen_allocation_sha256 at {source}",
    )

    bits = chosen.get("bits")
    if not isinstance(bits, dict) or not bits:
        raise AdjudicationError(f"chosen allocation has no tensor-group bits at {source}")
    flat_bits: list[int] = []
    for tensor, tensor_bits in bits.items():
        if not isinstance(tensor, str) or not tensor:
            raise AdjudicationError(f"chosen allocation has invalid tensor name at {source}")
        if not isinstance(tensor_bits, list) or not tensor_bits:
            raise AdjudicationError(f"chosen allocation tensor {tensor!r} has no group bits at {source}")
        flat_bits.extend(_integer(bit, field=f"chosen_allocation.bits[{tensor}]", minimum=2) for bit in tensor_bits)
    if any(bit > 8 for bit in flat_bits):
        raise AdjudicationError(f"chosen allocation bit depth exceeds 8 at {source}")

    base_bytes = _integer(ladder.get("base_bytes"), field="cheap_to_shrink_ladder.base_bytes", minimum=1)
    if ladder.get("byte_cost_checked") is not True:
        raise AdjudicationError(f"controller ladder byte cost was not checked at {source}")
    allocation_family = ladder.get("allocation_family")
    if not isinstance(allocation_family, str) or not allocation_family:
        raise AdjudicationError(f"controller ladder lacks allocation family at {source}")

    distinct_bits = sorted(set(flat_bits))
    expected_race_id = f"uniform{distinct_bits[0]}" if len(distinct_bits) == 1 else "adaptive"
    race = controller.get("quantization_race")
    if not isinstance(race, list) or not race:
        raise AdjudicationError(f"controller lacks quantization race at {source}")
    if any(not isinstance(row, dict) for row in race):
        raise AdjudicationError(f"quantization race contains a non-object row at {source}")
    selected_rows = [row for row in race if row.get("allocation_id") == expected_race_id]
    if len(selected_rows) != 1:
        raise AdjudicationError(f"quantization race has {len(selected_rows)} rows for {expected_race_id} at {source}")
    selected_race = selected_rows[0]
    if (
        selected_race.get("measured") is not True
        or selected_race.get("parse_back_exact") is not True
        or selected_race.get("retained_payload") is not True
        or selected_race.get("hard_cell_gate_pass") is not True
        or selected_race.get("pose_gate_pass") is not True
        or selected_race.get("road_lane_gate_pass") is not True
    ):
        raise AdjudicationError(
            f"selected race row is not measured, retained, parse-back exact, and gate-passing at {source}"
        )
    race_bytes = _integer(selected_race.get("packet_bytes"), field="quantization_race.packet_bytes", minimum=1)
    if race_bytes != base_bytes:
        raise AdjudicationError(f"selected race bytes {race_bytes} != ladder base bytes {base_bytes} at {source}")

    histogram = [{"bits": bit, "groups": sum(value == bit for value in flat_bits)} for bit in distinct_bits]
    return {
        "policy": policy,
        "allocation_family": allocation_family,
        "distinct_bit_depths": distinct_bits,
        "bit_depth_group_histogram": histogram,
        "tensor_count": len(bits),
        "group_count": len(flat_bits),
        "selection_sha256": selection_sha256,
        "chosen_allocation_sha256": chosen_sha256,
        "selected_race_id": expected_race_id,
        "selected_packet_bytes": base_bytes,
        "ladder_active": ladder.get("active"),
        "rung_bytes": ladder.get("rung_bytes"),
        "quantization_race": race,
    }


def _epoch_files(directory: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    if not directory.is_dir():
        return {}
    rows: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.name.startswith("._"):
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch in rows:
            raise AdjudicationError(f"duplicate epoch {epoch} under {directory}")
        rows[epoch] = path.resolve(strict=True)
    return rows


def _checkpoint_row(
    *,
    seed: int,
    epoch: int,
    evaluation_path: Path,
    checkpoint_path: Path,
    allocation: Mapping[str, Any],
    controller_record: Mapping[str, Any],
    seed_root: Path,
) -> dict[str, Any]:
    evaluation, evaluation_record = _load_json(evaluation_path)
    if evaluation.get("schema") != "ddm_wd3_retained_subset_evaluation.v1":
        raise AdjudicationError(f"unexpected evaluation schema at {evaluation_path}")
    if evaluation.get("axis") != EXPECTED_AXIS:
        raise AdjudicationError(f"evaluation axis is not the sealed advisory axis at {evaluation_path}")
    if evaluation.get("score_claim") is not False:
        raise AdjudicationError(f"evaluation must carry score_claim=false at {evaluation_path}")
    if evaluation.get("all_payloads_retained") is not True:
        raise AdjudicationError(f"evaluation did not retain all payloads at {evaluation_path}")
    if evaluation.get("n_pairs") != EXPECTED_N_PAIRS:
        raise AdjudicationError(f"evaluation is not n60 at {evaluation_path}")
    if evaluation.get("pair_ids") != list(EXPECTED_PAIR_IDS):
        raise AdjudicationError(f"evaluation is not the sealed evenly-strided n60 subset at {evaluation_path}")

    binding = evaluation.get("evaluation_binding")
    packet_archive = evaluation.get("packet_archive")
    if not isinstance(binding, dict) or not isinstance(packet_archive, dict):
        raise AdjudicationError(f"evaluation lacks binding/packet archive at {evaluation_path}")
    packet_allocation = packet_archive.get("allocation")
    if not isinstance(packet_allocation, dict):
        raise AdjudicationError(f"evaluation lacks packet allocation at {evaluation_path}")
    if packet_archive.get("schema") != "ddm_wd3_retained_packet_archive.v1":
        raise AdjudicationError(f"unexpected retained packet-archive schema at {evaluation_path}")
    if binding.get("allocation_sha256") != allocation["chosen_allocation_sha256"]:
        raise AdjudicationError(f"evaluation allocation SHA does not bind controller at {evaluation_path}")
    if packet_allocation.get("selection_sha256") != allocation["selection_sha256"]:
        raise AdjudicationError(f"evaluation selection SHA does not bind controller at {evaluation_path}")
    if packet_allocation.get("policy") != allocation["policy"]:
        raise AdjudicationError(f"evaluation allocation policy does not bind controller at {evaluation_path}")
    packet_bytes = _integer(
        packet_allocation.get("packet_bytes"),
        field="packet_archive.allocation.packet_bytes",
        minimum=1,
    )
    if packet_bytes != allocation["selected_packet_bytes"]:
        raise AdjudicationError(
            f"evaluation packet bytes {packet_bytes} != selected controller bytes "
            f"{allocation['selected_packet_bytes']} at {evaluation_path}"
        )

    parseback = packet_archive.get("parseback")
    parseback_report = parseback.get("report") if isinstance(parseback, dict) else None
    if (
        not isinstance(parseback_report, dict)
        or parseback.get("status") != "PASS"
        or parseback_report.get("packet_exact") is not True
        or parseback_report.get("repack_exact") is not True
        or packet_archive.get("receiver_parse_back_exact") is not True
        or packet_archive.get("untouched_sections_byte_identical") is not True
        or packet_archive.get("archive_repeat_byte_identical") is not True
    ):
        raise AdjudicationError(f"packet archive lacks exact parse-back/repeat/section proof at {evaluation_path}")

    payloads = packet_archive.get("payloads")
    if not isinstance(payloads, dict):
        raise AdjudicationError(f"packet archive lacks retained payload records at {evaluation_path}")
    required_payloads = (
        "archive",
        "archive_repeat",
        "member",
        "semantic_ck2_brotli_q11",
        "student_packet",
    )
    retained_payloads = {
        name: _retained_record(
            payloads.get(name),
            field=f"packet_archive.payloads.{name}",
            allowed_root=seed_root,
        )
        for name in required_payloads
    }
    archive_binding = _retained_record(
        packet_archive.get("archive_binding"),
        field="packet_archive.archive_binding",
        allowed_root=seed_root,
    )
    receiver_pairs = _retained_record(
        evaluation.get("receiver_pairs"),
        field="receiver_pairs",
        allowed_root=seed_root,
        verify_content_sha256=False,
    )
    scorer_bundle = _retained_record(
        evaluation.get("scorer_bundle"),
        field="scorer_bundle",
        allowed_root=seed_root,
        verify_content_sha256=False,
    )
    parseback_transcript = _retained_record(
        parseback.get("transcript"),
        field="packet_archive.parseback.transcript",
        allowed_root=seed_root,
    )
    section_receipt = _retained_record(
        packet_archive.get("section_preservation_receipt"),
        field="packet_archive.section_preservation_receipt",
        allowed_root=seed_root,
    )
    archive_bytes = _integer(
        packet_archive.get("archive_bytes"),
        field="packet_archive.archive_bytes",
        minimum=1,
    )
    archive = retained_payloads["archive"]
    if (
        archive_bytes != archive["bytes"]
        or archive_binding["bytes"] != archive["bytes"]
        or archive_binding["sha256"] != archive["sha256"]
    ):
        raise AdjudicationError(f"candidate archive receipt does not bind retained archive at {evaluation_path}")
    archive_repeat = retained_payloads["archive_repeat"]
    if archive_repeat["bytes"] != archive["bytes"] or archive_repeat["sha256"] != archive["sha256"]:
        raise AdjudicationError(f"candidate archive repeat is not byte-identical at {evaluation_path}")
    student_packet = retained_payloads["student_packet"]
    if student_packet["bytes"] != packet_bytes:
        raise AdjudicationError(f"retained student packet bytes do not bind controller at {evaluation_path}")
    if binding.get("student_packet_sha256") != student_packet["sha256"]:
        raise AdjudicationError(f"retained student packet SHA does not bind evaluation at {evaluation_path}")

    hard_d_seg = _finite_float(evaluation.get("hard_d_seg"), field="hard_d_seg", minimum=0.0)
    d_pose = _finite_float(evaluation.get("d_pose"), field="d_pose", minimum=0.0)
    seg_contribution = _finite_float(
        evaluation.get("seg_contribution"),
        field="seg_contribution",
        minimum=0.0,
    )
    pose_contribution = _finite_float(
        evaluation.get("pose_contribution"),
        field="pose_contribution",
        minimum=0.0,
    )
    if not math.isclose(seg_contribution, 100.0 * hard_d_seg, rel_tol=1e-12, abs_tol=1e-15):
        raise AdjudicationError(f"seg contribution does not bind hard_d_seg at {evaluation_path}")
    if not math.isclose(pose_contribution, math.sqrt(10.0 * d_pose), rel_tol=1e-12, abs_tol=1e-15):
        raise AdjudicationError(f"pose contribution does not bind d_pose at {evaluation_path}")
    bytes_shed = GB1_RENDERER_BYTES - allocation["selected_packet_bytes"]
    seg_damage = 100.0 * (hard_d_seg - GB1_HARD_D_SEG)
    pose_damage = math.sqrt(10.0 * d_pose) - math.sqrt(10.0 * GB1_D_POSE)
    damage = seg_damage + pose_damage
    rate_credit = bytes_shed * RATE_PER_BYTE
    composed_delta = damage - rate_credit

    return {
        "schema": ROW_SCHEMA,
        "seed": seed,
        "epoch": epoch,
        "axis": EXPECTED_AXIS,
        "pair_selection": "fixed_evenly_strided_n60_over_n600",
        "n_pairs": EXPECTED_N_PAIRS,
        "score_claim": False,
        "promotion_eligible": False,
        "submission_candidate": False,
        "bytes_shed_vs_gb1_renderer_30856B": bytes_shed,
        "selected_controller_packet_bytes": allocation["selected_packet_bytes"],
        "hard_d_seg": hard_d_seg,
        "d_pose": d_pose,
        "seg_damage_S_vs_gb1_reference": seg_damage,
        "pose_damage_S_vs_gb1_reference": pose_damage,
        "pose_plus_seg_damage_S_vs_gb1_reference": damage,
        "renderer_rate_credit_S_at_6_658e_7_per_B": rate_credit,
        "composed_delta_S_vs_break_even": composed_delta,
        "point_crosses_renderer_corner": composed_delta < 0.0,
        "seg_term_exceeds_pose_term": seg_damage > pose_damage,
        "falsifier_verdict": None,
        "controller_allocation": dict(allocation),
        "candidate_archive_observation": {
            "archive_bytes": archive_bytes,
            "archive": retained_payloads["archive"],
            "boundary": (
                "observed for provenance only; the chartered byte numerator comes from "
                "the controller-selected serialized packet, not ZIP-size fluctuation"
            ),
        },
        "retained_evaluation_artifacts": {
            "payloads": retained_payloads,
            "receiver_pairs": receiver_pairs,
            "scorer_bundle": scorer_bundle,
            "parseback_transcript": parseback_transcript,
            "section_preservation_receipt": section_receipt,
        },
        "checkpoint": _file_record(checkpoint_path),
        "evaluation": evaluation_record,
        "stage_controller": dict(controller_record),
    }


def _read_seed(training_root: Path, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_root = training_root / f"off_seed_{seed}" / "W96_flattened"
    if not seed_root.is_dir():
        return (
            {
                "seed": seed,
                "status": "MISSING_SEED_STORE",
                "seed_root": str(seed_root),
                "endpoint_epoch": EXPECTED_ENDPOINT_EPOCH,
                "endpoint_complete": False,
                "joined_checkpoint_count": 0,
            },
            [],
        )

    controller_results = sorted(
        path
        for path in (seed_root / "stage_controllers").glob("stage_*_from_epoch_*/STAGE_CONTROLLER_RESULT.json")
        if path.is_file() and not path.name.startswith("._")
    )
    if not controller_results:
        return (
            {
                "seed": seed,
                "status": "MISSING_STAGE_CONTROLLER_RESULT",
                "seed_root": str(seed_root.resolve(strict=True)),
                "endpoint_epoch": EXPECTED_ENDPOINT_EPOCH,
                "endpoint_complete": False,
                "joined_checkpoint_count": 0,
            },
            [],
        )
    controller_path = _controller_path(seed_root)
    controller, controller_record = _load_json(controller_path)
    allocation = _allocation_summary(controller, source=controller_path)
    evaluations = _epoch_files(seed_root / "evaluations", _EVALUATION_RE)
    checkpoints = _epoch_files(seed_root / "checkpoints", _CHECKPOINT_RE)
    joined_epochs = sorted(set(evaluations) & set(checkpoints))
    unpaired_evaluations = sorted(set(evaluations) - set(checkpoints))
    unpaired_checkpoints = sorted(set(checkpoints) - set(evaluations))
    if not joined_epochs:
        return (
            {
                "seed": seed,
                "status": "NO_JOINED_CHECKPOINT_EVALUATIONS",
                "seed_root": str(seed_root.resolve(strict=True)),
                "endpoint_epoch": EXPECTED_ENDPOINT_EPOCH,
                "endpoint_complete": False,
                "joined_checkpoint_count": 0,
                "unpaired_evaluation_epochs": unpaired_evaluations,
                "unpaired_checkpoint_epochs": unpaired_checkpoints,
                "stage_controller": controller_record,
                "selected_controller_allocation": allocation,
            },
            [],
        )

    rows = [
        _checkpoint_row(
            seed=seed,
            epoch=epoch,
            evaluation_path=evaluations[epoch],
            checkpoint_path=checkpoints[epoch],
            allocation=allocation,
            controller_record=controller_record,
            seed_root=seed_root,
        )
        for epoch in joined_epochs
    ]
    endpoint_complete = (
        EXPECTED_ENDPOINT_EPOCH in joined_epochs and not unpaired_evaluations and not unpaired_checkpoints
    )
    status = "ENDPOINT_COMPLETE" if endpoint_complete else "ENDPOINT_INCOMPLETE"
    return (
        {
            "seed": seed,
            "status": status,
            "seed_root": str(seed_root.resolve(strict=True)),
            "endpoint_epoch": EXPECTED_ENDPOINT_EPOCH,
            "endpoint_complete": endpoint_complete,
            "joined_checkpoint_count": len(joined_epochs),
            "joined_epochs": joined_epochs,
            "unpaired_evaluation_epochs": unpaired_evaluations,
            "unpaired_checkpoint_epochs": unpaired_checkpoints,
            "stage_controller": controller_record,
            "selected_controller_allocation": allocation,
        },
        rows,
    )


def adjudicate(training_root: Path, *, seeds: Sequence[int] = EXPECTED_SEEDS) -> dict[str, Any]:
    resolved_root = training_root.expanduser().resolve(strict=True)
    if not resolved_root.is_dir():
        raise AdjudicationError(f"training root is not a directory: {resolved_root}")
    expected_seeds = tuple(_integer(seed, field="seed", minimum=1) for seed in seeds)
    if expected_seeds != EXPECTED_SEEDS:
        raise AdjudicationError(f"both-OFF adjudication requires seeds {EXPECTED_SEEDS}, got {expected_seeds}")

    seed_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed in expected_seeds:
        summary, seed_rows = _read_seed(resolved_root, seed)
        seed_summaries.append(summary)
        rows.extend(seed_rows)

    all_endpoints_complete = all(row["endpoint_complete"] for row in seed_summaries)
    crossing_rows = [row for row in rows if row["point_crosses_renderer_corner"]]
    if crossing_rows:
        verdict = CORNER_CROSSED
    elif all_endpoints_complete and rows:
        verdict = ENTERED_AND_REFUSED
    else:
        verdict = INCOMPLETE_DATA
    if verdict not in VERDICTS:
        raise AssertionError(f"unregistered verdict: {verdict}")
    for row in rows:
        row["falsifier_verdict"] = verdict

    seg_dominant = [row for row in rows if row["seg_term_exceeds_pose_term"]]
    return {
        "schema": SCHEMA,
        "authority": EXPECTED_AXIS,
        "axis": EXPECTED_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "submission_candidate": False,
        "verdict_scope": "S1A_TWO_SEED_OFF_FLOOR_ADVISORY_CHECKPOINT_SET",
        "falsifier_verdict": verdict,
        "training_root": str(resolved_root),
        "expected_seeds": list(expected_seeds),
        "expected_endpoint_epoch": EXPECTED_ENDPOINT_EPOCH,
        "all_seed_endpoints_complete": all_endpoints_complete,
        "reference": {
            "gb1_renderer_bytes": GB1_RENDERER_BYTES,
            "gb1_hard_d_seg": GB1_HARD_D_SEG,
            "gb1_d_pose": GB1_D_POSE,
            "rate_per_byte": RATE_PER_BYTE,
            "damage_expression": ("100*(hard_d_seg-0.00020139) + sqrt(10*d_pose) - sqrt(10*6.37e-6)"),
            "composed_delta_expression": ("damage_S - (30856-controller_selected_packet_bytes)*(25/37545489)"),
            "boundary": (
                "GB1 references are contest-CUDA n600 anchors applied only as the "
                "pre-registered advisory falsifier baseline; rows remain n60/MPS, "
                "score_claim=false, and cannot promote or bank a population negative"
            ),
        },
        "subset_boundary": {
            "selection": "fixed evenly-strided n60: pair ids 0,10,...,590",
            "prefix": False,
            "population_pairs": 600,
            "use": "endpoint screening and falsifier arithmetic only",
        },
        "seed_summaries": seed_summaries,
        "rows": rows,
        "row_count": len(rows),
        "corner_crossing_point_count": len(crossing_rows),
        "prior_law_prediction": {
            "prediction": ("pose damage term exceeds seg damage term at every checkpoint"),
            "tested_checkpoint_count": len(rows),
            "seg_term_exceeds_pose_term_count": len(seg_dominant),
            "falsified_at_least_one_checkpoint": bool(seg_dominant),
            "falsifying_points": [{"seed": row["seed"], "epoch": row["epoch"]} for row in seg_dominant],
        },
        "consumer": ("MAIN both-OFF endpoint review before any authorization of ON seed 20260815"),
        "boundary": (
            "Stage A is an input to Stage B and never a submission candidate; even "
            "renderer-to-zero at fixed GB1 distortion remains above sub-0.12"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/training"),
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output_path = args.out.expanduser().resolve(strict=False)
        training_root = args.training_root.expanduser().resolve(strict=True)
        if training_root == output_path or training_root in output_path.parents:
            raise AdjudicationError("output receipt must be outside the read-only training tree")
        result = adjudicate(training_root)
        _atomic_json(output_path, result)
    except (AdjudicationError, OSError) as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc)}), file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "falsifier_verdict": result["falsifier_verdict"],
                "row_count": result["row_count"],
                "all_seed_endpoints_complete": result["all_seed_endpoints_complete"],
                "out": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
