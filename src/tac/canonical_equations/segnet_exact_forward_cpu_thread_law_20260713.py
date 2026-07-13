# SPDX-License-Identifier: MIT
"""Measured exact-forward CPU thread control law for task #456.

No external theorem or method is imported.  The law is a direct paired
measurement over receiver-realized witness frames.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "segnet_exact_forward_cpu_thread_control_v1"
_UTC = "2026-07-13T02:07:21Z"
_MEMO = ".omx/research/codex_findings_frozen_segnet_exact_forward_20260713_codex.md"
_RECEIPT = "experiments/results/segnet_exact_forward_20260713T020000Z/receipt.json"
_AXIS = "[macOS-CPU advisory; torch-fp32; autograd-enabled forward; no MPS/CUDA]"

N_REAL_PAIRS = 64
TOTAL_ARGMAX_PIXELS = 12_582_912
BASELINE_THREADS = 6
SELECTED_THREADS = 1
BASELINE_MEDIAN_MS = 936.3120624999999
CHEAP_MEDIAN_MS = 312.677146
MATCHED_SPEEDUP_X = BASELINE_MEDIAN_MS / CHEAP_MEDIAN_MS
MATCHED_SPEED_GAP_MS = BASELINE_MEDIAN_MS - CHEAP_MEDIAN_MS
COMPOSED_TIMING_NOISE_FLOOR_MS = 610.6709248999996


# v1 above is a historical n64 in-process alternating-thread anchor.  It is
# intentionally retained unchanged.  v2 has no import-time measurement claim:
# it can only be constructed from both terminal n600 static-process receipts.
STATIC_PROCESS_V2_EQUATION_ID = "segnet_exact_forward_cpu_thread_static_process_v2"
STATIC_PROCESS_V2_RECEIPTS = {
    "torch_2_12_1": Path(
        "experiments/results/segnet_exact_forward_static_transfer_torch2121_n600_20260713/receipt.json"
    ),
    "torch_2_12_0": Path(
        "experiments/results/segnet_exact_forward_static_transfer_torch2120_n600_20260713/receipt.json"
    ),
}
_STATIC_PROCESS_METHOD = "fresh_child_process_static_threads"
_STATIC_STAGE_ORDER = ("baseline_rep0", "selected_rep0", "selected_rep1", "baseline_rep1")
_STATIC_AXIS = "[macOS-CPU advisory; process-static torch-fp32 training-forward; no MPS/CUDA]"
_STATIC_MEMO = ".omx/research/cheaper_exact_forward_transfer_95kill_20260713.md"
_STATIC_VERDICT_SCOPE = (
    "fresh-child process-static ABBA formulation over first 600 receiver-realized pairs on the "
    "fingerprinted local macOS CPU/Torch build only; n<600 diagnostic; no transfer to another "
    "host/build/model/input set, backward, full training, contest-CPU/CUDA, evaluator, d_seg, "
    "d_pose, archive, score, or promotion"
)
_STATIC_LABELS = {
    "canary_count": "ASSUMED_HEURISTIC_SCREEN_ONLY",
    "checkpoint_interval": "ASSUMED_RECOVERY_ENVELOPE",
    "timing_and_sha": "MEASURED",
    "zero_flip_from_sha": "DERIVED",
}


class StaticProcessReceiptError(ValueError):
    """A terminal static-process receipt lacks the v2 no-false-authority contract."""


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StaticProcessReceiptError(f"{field} must be a mapping")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise StaticProcessReceiptError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StaticProcessReceiptError(f"{field} must be a positive integer")
    return value


def _finite_positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StaticProcessReceiptError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise StaticProcessReceiptError(f"{field} must be a finite positive number")
    return result


def _require_true(value: object, field: str) -> None:
    if value is not True:
        raise StaticProcessReceiptError(f"{field} must be true")


def _receipt_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_file():
        raise StaticProcessReceiptError(f"required static-process receipt is missing: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact_ref(value: object, field: str) -> Path:
    ref = _mapping(value, field)
    path_value = ref.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise StaticProcessReceiptError(f"{field}.path must be a non-empty string")
    expected_sha = _sha256(ref.get("sha256"), f"{field}.sha256")
    path = Path(path_value)
    if not path.is_file():
        raise StaticProcessReceiptError(f"{field} is missing: {path}")
    if _sha256_file(path) != expected_sha:
        raise StaticProcessReceiptError(f"{field} byte SHA does not match receipt custody")
    return path


def _timing_samples(value: object, field: str) -> list[float]:
    """Return receipt timing samples after re-deriving their stated median."""

    summary = _mapping(value, field)
    samples = summary.get("samples_ms")
    if not isinstance(samples, list) or len(samples) != 600:
        raise StaticProcessReceiptError(f"{field}.samples_ms must contain exactly 600 values")
    parsed = [_finite_positive(sample, f"{field}.samples_ms") for sample in samples]
    if summary.get("count") != 600:
        raise StaticProcessReceiptError(f"{field}.count must be 600")
    stated = _finite_positive(summary.get("median_ms"), f"{field}.median_ms")
    derived = float(statistics.median(parsed))
    if not math.isclose(stated, derived, rel_tol=1e-12, abs_tol=1e-12):
        raise StaticProcessReceiptError(f"{field}.median_ms does not match its receipt samples")
    return parsed


def _derived_one_sided_sign_pvalue(wins: int, losses: int) -> float:
    """Exact binomial tail for selected-faster wins, matching the probe contract."""

    trials = wins + losses
    if trials <= 0:
        raise StaticProcessReceiptError("matched sign test needs at least one non-tie pair")
    return float(sum(math.comb(trials, k) for k in range(wins, trials + 1)) / (2**trials))


def _validate_sign_test(
    value: object, *, baseline_samples: list[float], selected_samples: list[float], alpha: float
) -> float:
    sign_test = _mapping(value, "measurement.matched_sign_test")
    pairs = tuple(zip(baseline_samples, selected_samples, strict=True))
    wins = sum(selected < baseline for baseline, selected in pairs)
    losses = sum(selected > baseline for baseline, selected in pairs)
    ties = len(baseline_samples) - wins - losses
    for key, expected in (("wins", wins), ("losses", losses), ("ties", ties)):
        if sign_test.get(key) != expected:
            raise StaticProcessReceiptError(f"matched sign test {key} does not match receipt timings")
    if sign_test.get("alpha") != alpha or sign_test.get("label") != "DERIVED_FROM_MATCHED_MEASUREMENTS":
        raise StaticProcessReceiptError("matched sign test authority fields are not canonical")
    stated = _finite_positive(sign_test.get("one_sided_exact_binomial_pvalue"), "matched sign pvalue")
    derived = _derived_one_sided_sign_pvalue(wins, losses)
    if not math.isclose(stated, derived, rel_tol=1e-12, abs_tol=1e-300):
        raise StaticProcessReceiptError("matched sign pvalue does not match receipt timings")
    return stated


def load_and_validate_static_process_receipt(
    receipt_path: str | Path, *, expected_torch_build: str
) -> dict[str, Any]:
    """Parse one terminal n600 fresh-child static-ABBA receipt fail-closed.

    Terminal ``NO-GO`` is evidence, not malformed input: the validator derives
    the admission predicate again from receipt-custodied child artifacts and
    requires the emitted verdict to agree.  It never converts a cross-arm SHA
    mismatch into a zero-flip claim.
    """

    path = _receipt_path(receipt_path)
    try:
        receipt = _mapping(json.loads(path.read_text()), "receipt")
    except json.JSONDecodeError as exc:
        raise StaticProcessReceiptError(f"invalid JSON receipt: {path}") from exc

    schema = receipt.get("schema")
    if not isinstance(schema, str) or not schema.startswith("frozen_segnet_exact_forward_static_transfer_probe"):
        raise StaticProcessReceiptError("receipt schema is not a static-transfer probe schema")
    verdict = receipt.get("verdict")
    if verdict not in {"GO", "NO-GO"}:
        raise StaticProcessReceiptError("terminal static receipt verdict must be GO or NO-GO")
    _require_true(receipt.get("research_only"), "research_only")
    if receipt.get("verdict_scope") != _STATIC_VERDICT_SCOPE:
        raise StaticProcessReceiptError("terminal verdict_scope is missing or broader than the producer contract")
    if receipt.get("axis") != _STATIC_AXIS:
        raise StaticProcessReceiptError("axis must remain the exact local process-static advisory axis")
    if receipt.get("labels") != _STATIC_LABELS:
        raise StaticProcessReceiptError("evidence labels must preserve measured/derived/assumed authority")
    validation = _mapping(receipt.get("validation"), "validation")
    if validation.get("status") != "self-validated-from-terminal-child-bytes-before-and-after-write":
        raise StaticProcessReceiptError("receipt must be self-validated from terminal child bytes")
    authority = _mapping(receipt.get("authority"), "authority")
    if authority != {"score_claim": False, "pointer_moved": False, "promotion_eligible": False}:
        raise StaticProcessReceiptError("authority must preserve advisory-only, pointer-unchanged custody")

    runtime = _mapping(receipt.get("runtime"), "runtime")
    if runtime.get("torch") != expected_torch_build:
        raise StaticProcessReceiptError(
            f"Torch build mismatch: expected {expected_torch_build!r}, got {runtime.get('torch')!r}"
        )
    if any(runtime.get(key) is not False for key in ("mps_used", "cuda_used", "contest_cpu_timing_measured")):
        raise StaticProcessReceiptError("static-process receipt must be CPU-only")

    measurement = _mapping(receipt.get("measurement"), "measurement")
    if _positive_int(measurement.get("n_real_pairs"), "measurement.n_real_pairs") != 600:
        raise StaticProcessReceiptError("static-process v2 requires exactly n_pairs=600")
    if tuple(measurement.get("stage_order", ())) != _STATIC_STAGE_ORDER or tuple(measurement.get("child_passes", ())) != _STATIC_STAGE_ORDER:
        raise StaticProcessReceiptError("receipt must preserve the four-pass ABBA stage order")
    total_argmax_pixels = _positive_int(measurement.get("total_argmax_pixels"), "measurement.total_argmax_pixels")
    if measurement.get("label") != "MEASURED" or measurement.get("method") != _STATIC_PROCESS_METHOD:
        raise StaticProcessReceiptError("receipt does not establish measured fresh child process-static threads")
    if measurement.get("thread_binding") != (
        "fresh process per measurement and replay; intra/inter-op immutable after pre-model binding"
    ):
        raise StaticProcessReceiptError("thread-binding custody is not the process-static contract")
    _require_true(measurement.get("input_gradient_graph_preserved"), "input_gradient_graph_preserved")

    sequence_shas = _mapping(measurement.get("sequence_sha256"), "measurement.sequence_sha256")
    replay_sequence_shas = _mapping(measurement.get("replay_sequence_sha256"), "measurement.replay_sequence_sha256")
    if set(sequence_shas) != set(_STATIC_STAGE_ORDER) or set(replay_sequence_shas) != set(_STATIC_STAGE_ORDER):
        raise StaticProcessReceiptError("sequence SHA stages do not match ABBA stages")
    measured_shas = {stage: _sha256(sequence_shas[stage], f"measurement.sequence_sha256.{stage}") for stage in _STATIC_STAGE_ORDER}
    replay_shas = {stage: _sha256(replay_sequence_shas[stage], f"measurement.replay_sequence_sha256.{stage}") for stage in _STATIC_STAGE_ORDER}

    selected_arm = _mapping(receipt.get("selected_arm"), "selected_arm")
    baseline_threads = _positive_int(selected_arm.get("baseline_threads"), "selected_arm.baseline_threads")
    selected_threads = _positive_int(selected_arm.get("threads"), "selected_arm.threads")
    selected_strategy = selected_arm.get("strategy")
    if not isinstance(selected_strategy, str) or not selected_strategy:
        raise StaticProcessReceiptError("selected_arm.strategy must be a non-empty string")

    pass_receipts = _mapping(measurement.get("pass_receipts"), "measurement.pass_receipts")
    if set(pass_receipts) != set(_STATIC_STAGE_ORDER):
        raise StaticProcessReceiptError("pass receipts must cover each ABBA stage exactly once")
    child_ids: set[str] = set()
    pids: set[int] = set()
    for stage in _STATIC_STAGE_ORDER:
        child_row = _mapping(pass_receipts[stage], f"measurement.pass_receipts.{stage}")
        child_id = child_row.get("measurement_child_id")
        if not isinstance(child_id, str) or not child_id or child_id in child_ids:
            raise StaticProcessReceiptError("each ABBA pass must have a distinct measurement child_id")
        child_ids.add(child_id)
        child_pid = _positive_int(child_row.get("measurement_pid"), f"child {stage} measurement_pid")
        if child_pid in pids:
            raise StaticProcessReceiptError("each ABBA pass must have a distinct measurement PID")
        pids.add(child_pid)
        replay_ids = child_row.get("replay_child_ids")
        replay_pids = child_row.get("replay_pids")
        if not isinstance(replay_ids, list) or len(replay_ids) != 1 or not isinstance(replay_pids, list) or len(replay_pids) != 1:
            raise StaticProcessReceiptError(f"child {stage} needs exactly one replay child id and PID")
        replay_id = replay_ids[0]
        if not isinstance(replay_id, str) or not replay_id or replay_id in child_ids:
            raise StaticProcessReceiptError("every measurement and replay must use a distinct child ID")
        child_ids.add(replay_id)
        replay_pid = _positive_int(replay_pids[0], f"child {stage} replay PID")
        if replay_pid in pids:
            raise StaticProcessReceiptError("every measurement and replay must use a distinct PID")
        pids.add(replay_pid)
        intraop_threads = _positive_int(child_row.get("intraop_threads"), f"child {stage} intraop_threads")
        interop_threads = _positive_int(child_row.get("interop_threads"), f"child {stage} interop_threads")
        expected_threads = baseline_threads if stage.startswith("baseline") else selected_threads
        expected_strategy = "eager_nchw_autograd" if stage.startswith("baseline") else selected_strategy
        if intraop_threads != expected_threads or child_row.get("strategy") != expected_strategy:
            raise StaticProcessReceiptError(f"child {stage} arm disagrees with receipt-selected static arm")
        for binding_name in ("binding_before", "binding_after", "replay_binding_before", "replay_binding_after"):
            binding = _mapping(child_row.get(binding_name), f"child {stage} {binding_name}")
            if binding.get("intraop_threads") != intraop_threads or binding.get("interop_threads") != interop_threads:
                raise StaticProcessReceiptError(f"child {stage} {binding_name} disagrees with static custody")
        _require_true(child_row.get("measurement_complete"), f"child {stage} measurement_complete")
        _require_true(child_row.get("replay_complete"), f"child {stage} replay_complete")
        measurement_segments = child_row.get("measurement_process_segments")
        replay_segments = child_row.get("replay_process_segments")
        if not isinstance(measurement_segments, list) or len(measurement_segments) != 1 or not isinstance(replay_segments, list) or len(replay_segments) != 1:
            raise StaticProcessReceiptError(f"child {stage} must have one measurement and one replay process segment")
        expected_binding = {"intraop_threads": intraop_threads, "interop_threads": interop_threads}
        for segment_name, segment_value, expected_id, expected_pid in (
            ("measurement", measurement_segments[0], child_id, child_pid),
            ("replay", replay_segments[0], replay_id, replay_pid),
        ):
            segment = _mapping(segment_value, f"child {stage} {segment_name} process segment")
            if (
                segment.get("child_id") != expected_id
                or segment.get("pid") != expected_pid
                or segment.get("binding") != expected_binding
                or segment.get("started_from_completed_pairs") != 0
                or segment.get("completed_pairs") != 600
                or not isinstance(segment.get("started_at_utc"), str)
                or not isinstance(segment.get("completed_at_utc"), str)
            ):
                raise StaticProcessReceiptError(
                    f"child {stage} {segment_name} segment does not bind its terminal process"
                )
        _validate_artifact_ref(child_row.get("terminal_stage_file"), f"child {stage} terminal_stage_file")
        _validate_artifact_ref(child_row.get("terminal_replay_file"), f"child {stage} terminal_replay_file")
        measured_sha = _sha256(child_row.get("measurement_sequence_sha256"), f"child {stage} measurement SHA")
        replay_sha = _sha256(child_row.get("replay_sequence_sha256"), f"child {stage} replay SHA")
        if measured_sha != replay_sha or measured_sha != measured_shas[stage] or replay_sha != replay_shas[stage]:
            raise StaticProcessReceiptError(f"child {stage} measurement/replay sequence SHA mismatch")

    if len(child_ids) != 8 or len(pids) != 8:
        raise StaticProcessReceiptError("terminal receipt must preserve eight unique child IDs and PIDs")
    segment_counts = _mapping(
        measurement.get("process_segments_per_pass"), "measurement.process_segments_per_pass"
    )
    if set(segment_counts) != set(_STATIC_STAGE_ORDER) or any(
        segment_counts[stage] != 1 for stage in _STATIC_STAGE_ORDER
    ):
        raise StaticProcessReceiptError("each ABBA pass must preserve exactly one process segment")
    full_replays = _mapping(measurement.get("independent_full_replays"), "measurement.independent_full_replays")
    all_eight_sequences_equal = len({*measured_shas.values(), *replay_shas.values()}) == 1
    if (
        full_replays.get("count") != 4
        or full_replays.get("per_arm_count") != 2
        or full_replays.get("complete") is not True
        or full_replays.get("independent_processes") is not True
        or full_replays.get("unique_child_id_count") != 8
        or full_replays.get("unique_pid_count") != 8
        or full_replays.get("sha_equal") is not all_eight_sequences_equal
        or measurement.get("all_sequence_shas_equal") is not all_eight_sequences_equal
    ):
        raise StaticProcessReceiptError("full replay evidence does not match derived eight-way child evidence")

    pair_evidence = _mapping(measurement.get("pair_sha_evidence"), "measurement.pair_sha_evidence")
    pair_equal = pair_evidence.get("all_pair_sha256_equal")
    if not isinstance(pair_equal, bool) or pair_equal is not all_eight_sequences_equal:
        raise StaticProcessReceiptError("pair-level equality must agree with terminal eight-way SHA evidence")
    mismatch_pair_count = pair_evidence.get("mismatch_pair_count")
    if not isinstance(mismatch_pair_count, int) or isinstance(mismatch_pair_count, bool) or mismatch_pair_count < 0:
        raise StaticProcessReceiptError("pair mismatch count must be a non-negative integer")
    expected_flip_count: int | None = 0 if pair_equal else None
    if pair_equal:
        if mismatch_pair_count != 0 or pair_evidence.get("first_mismatch") is not None:
            raise StaticProcessReceiptError("equal pair SHA evidence must have zero mismatch rows")
        expected_flip_derivation = "DERIVED_ZERO_FROM_EIGHT_WAY_EXACT_PER_PAIR_SHA_EQUALITY"
    elif mismatch_pair_count <= 0 or not isinstance(pair_evidence.get("first_mismatch"), Mapping):
        raise StaticProcessReceiptError("SHA-mismatch NO-GO must preserve a positive mismatch witness")
    else:
        expected_flip_derivation = "UNAVAILABLE_SHA_MISMATCH_FAIL_CLOSED_NO_RAW_PRIOR_TENSOR"
    if pair_evidence.get("flip_count_derivation") != expected_flip_derivation:
        raise StaticProcessReceiptError("pair SHA flip-count derivation authority is inconsistent")
    for field in ("argmax_flip_count", "derived_argmax_flip_count"):
        if measurement.get(field) != expected_flip_count:
            raise StaticProcessReceiptError(f"{field} must preserve the receipt-derived flip authority")
    if pair_evidence.get("derived_argmax_flip_count") != expected_flip_count:
        raise StaticProcessReceiptError("pair SHA evidence derived flip authority was coerced")
    expected_flip_rate = 0.0 if pair_equal else None
    if measurement.get("argmax_flip_rate") != expected_flip_rate:
        raise StaticProcessReceiptError("argmax_flip_rate was coerced beyond SHA authority")
    for stage in _STATIC_STAGE_ORDER:
        expected_child_flips = 0 if pair_equal else None
        if _mapping(pass_receipts[stage], f"measurement.pass_receipts.{stage}").get("derived_argmax_flip_count") != expected_child_flips:
            raise StaticProcessReceiptError(f"child {stage} derived flip authority was coerced")

    pair78 = _mapping(measurement.get("pair78"), "measurement.pair78")
    if pair78.get("index") != 78:
        raise StaticProcessReceiptError("pair78 receipt must retain the known confound index")
    pair78_pass = _mapping(pair78.get("per_pass_sha256"), "measurement.pair78.per_pass_sha256")
    pair78_replay = _mapping(pair78.get("per_replay_sha256"), "measurement.pair78.per_replay_sha256")
    if set(pair78_pass) != set(_STATIC_STAGE_ORDER) or set(pair78_replay) != set(_STATIC_STAGE_ORDER):
        raise StaticProcessReceiptError("pair78 SHA stages must cover every measurement and replay")
    pair78_values = {
        *(_sha256(pair78_pass[stage], f"pair78 measurement SHA {stage}") for stage in _STATIC_STAGE_ORDER),
        *(_sha256(pair78_replay[stage], f"pair78 replay SHA {stage}") for stage in _STATIC_STAGE_ORDER),
    }
    pair78_stable = len(pair78_values) == 1
    pair78_resolved = pair78_stable and all_eight_sequences_equal and pair_equal
    if pair78.get("stable") is not pair78_stable or pair78.get("resolved") is not pair78_resolved:
        raise StaticProcessReceiptError("pair78 stable/resolved fields disagree with eight-way evidence")

    baseline_samples = _timing_samples(measurement.get("baseline_per_pair_replica_median"), "measurement.baseline_per_pair_replica_median")
    selected_samples = _timing_samples(measurement.get("selected_per_pair_replica_median"), "measurement.selected_per_pair_replica_median")
    baseline_median_ms = float(statistics.median(baseline_samples))
    selected_median_ms = float(statistics.median(selected_samples))
    speedup = _finite_positive(measurement.get("static_paired_speedup_x"), "static_paired_speedup_x")
    if not math.isclose(speedup, baseline_median_ms / selected_median_ms, rel_tol=1e-12, abs_tol=1e-12):
        raise StaticProcessReceiptError("receipt static speedup is inconsistent with terminal timings")
    alpha = _finite_positive(measurement.get("matched_sign_alpha"), "matched_sign_alpha")
    if measurement.get("matched_sign_alpha_provenance") != "OPERATOR_SEALED_TRANSFER_V4_FALSE_POSITIVE_BUDGET":
        raise StaticProcessReceiptError("matched sign alpha lacks its value-provenance ladder")
    sign_pvalue = _validate_sign_test(measurement.get("matched_sign_test"), baseline_samples=baseline_samples, selected_samples=selected_samples, alpha=alpha)
    distinct_arm = not (selected_strategy == "eager_nchw_autograd" and selected_threads == baseline_threads)
    admitted = (
        all_eight_sequences_equal
        and expected_flip_count == 0
        and distinct_arm
        and selected_median_ms < baseline_median_ms
        and sign_pvalue <= alpha
    )
    expected_verdict = "GO" if admitted else "NO-GO"
    if verdict != expected_verdict:
        raise StaticProcessReceiptError("terminal verdict does not match independently re-derived admission")

    return {
        "receipt_path": str(path),
        "receipt_sha256": _sha256_file(path),
        "torch_build": expected_torch_build,
        "completed_at_utc": receipt.get("completed_at_utc"),
        "n_real_pairs": 600,
        "total_argmax_pixels": total_argmax_pixels,
        "baseline_median_ms": baseline_median_ms,
        "selected_median_ms": selected_median_ms,
        "static_paired_speedup_x": speedup,
        "verdict": verdict,
        "admitted": admitted,
        "argmax_flip_count": expected_flip_count,
        "mismatch_pair_count": mismatch_pair_count,
        "pair78_stable": pair78_stable,
        "pair78_resolved": pair78_resolved,
        "sequence_sha256": dict(measured_shas),
        "replay_sequence_sha256": dict(replay_shas),
        "selected_arm": dict(selected_arm),
    }


def build_segnet_exact_forward_cpu_thread_static_process_v2(
    *, receipt_paths: Mapping[str, str | Path] | None = None
) -> CanonicalEquation:
    """Build v2 only after both real static-process terminal receipts validate."""

    paths = receipt_paths or STATIC_PROCESS_V2_RECEIPTS
    if set(paths) != set(STATIC_PROCESS_V2_RECEIPTS):
        raise StaticProcessReceiptError("v2 requires exactly Torch 2.12.1 and 2.12.0 terminal receipt paths")
    anchors_data = [
        load_and_validate_static_process_receipt(paths["torch_2_12_1"], expected_torch_build="2.12.1"),
        load_and_validate_static_process_receipt(paths["torch_2_12_0"], expected_torch_build="2.12.0"),
    ]
    calibration_utc = max(str(row["completed_at_utc"]) for row in anchors_data)
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_STATIC_MEMO,
        reactivation_criteria=(
            "re-run both build-specific n600 fresh-child static ABBA receipts after any host, Torch, "
            "model, corpus, static-thread custody, or process-lifecycle change"
        ),
        measurement_axis=_STATIC_AXIS,
        hardware_substrate="apple_macos_arm64_cpu_torch_fp32",
        captured_at_utc=calibration_utc,
    )
    anchors = tuple(
        EmpiricalAnchor(
            anchor_id=f"segnet_exact_forward_static_abba_n600_{row['torch_build'].replace('.', '_')}",
            measurement_utc=str(row["completed_at_utc"]),
            inputs={
                "n_real_pairs": row["n_real_pairs"],
                "total_argmax_pixels": row["total_argmax_pixels"],
                "thread_process_method": _STATIC_PROCESS_METHOD,
                "four_independent_child_passes": True,
                "dual_independent_full_replays": True,
                "pair78_stable": row["pair78_stable"],
                "pair78_resolved": row["pair78_resolved"],
                "torch_build": row["torch_build"],
            },
            predicted_output={
                "admission_gate": "all eight terminal child-pass/replay sequence SHA values match, derived flips are zero, the arm is distinct and faster, and the matched sign test passes"
            },
            empirical_output={
                "baseline_median_ms": row["baseline_median_ms"],
                "selected_median_ms": row["selected_median_ms"],
                "static_paired_speedup_x": row["static_paired_speedup_x"],
                "receipt_sha256": row["receipt_sha256"],
                "sequence_sha256": row["sequence_sha256"],
                "replay_sequence_sha256": row["replay_sequence_sha256"],
                "argmax_flip_count": row["argmax_flip_count"],
                "mismatch_pair_count": row["mismatch_pair_count"],
                "pair78_resolved": row["pair78_resolved"],
                "admitted": row["admitted"],
                "verdict": row["verdict"],
                "authority": "advisory_only; score_claim=false; promotion_eligible=false; pointer_unchanged; contest_cpu_unmeasured",
            },
            residual=0.0,
            source_artifact=row["receipt_path"],
            measurement_method=(
                "n600 process-static fresh-child ABBA passes with receipt-custodied intra/inter-op threads, "
                "dual independent full replays, and fail-closed re-derivation of exactness and timing admission"
            ),
            provenance=provenance,
            empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
            noise_floor=None,
            noise_floor_provenance="timing uncertainty is receipt-derived; score and contest-CPU authority remain unmeasured",
        )
        for row in anchors_data
    )
    return CanonicalEquation(
        equation_id=STATIC_PROCESS_V2_EQUATION_ID,
        name="Frozen-SegNet exact-forward static-process CPU thread law",
        one_line_summary=(
            "Advisory-only n600 speed evidence records either a rankable GO or a scoped NO-GO only after four "
            "fresh-child static-thread ABBA passes and their independent full replays re-derive the terminal gate."
        ),
        latex_form=(
            r"\mathrm{admit}_{\rm advisory}=[n=600]\,[F=0]\,[H_{B_0}=H_{S_0}=H_{S_1}=H_{B_1}]\,"
            r"[R_s=H_s\ \forall s]\,[t_S<t_B]\,[p_{\rm sign}\leq\alpha]"
        ),
        python_callable_module_path="tools.probe_segnet_exact_forward_static_transfer:main",
        domain_of_validity={
            "research_only": True,
            "included": [
                "macOS arm64 CPU", "Torch 2.12.1 and 2.12.0 only", "n600 receiver-realized pairs",
                "four fresh child-process static-thread ABBA passes", "dual independent full replays",
            ],
            "excluded": [
                "historical n64 v1 in-process alternating-thread anchor", "contest-CPU and contest-CUDA timing",
                "score, archive, d_seg, d_pose, promotion, or pointer movement", "backward or training throughput",
            ],
            "authority": "MEASURED local advisory only; score_claim=false; promotion_eligible=false; pointer unchanged; contest CPU unmeasured",
        },
        units_in={"forward_time": "milliseconds", "argmax_pixels": "receipt-derived count"},
        units_out={"static_paired_speedup": "dimensionless ratio", "advisory_admission": "boolean"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"admission_predicate": 0.0},
        last_calibration_utc=calibration_utc,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tools.probe_segnet_exact_forward_static_transfer",),
        canonical_producers=("tools.probe_segnet_exact_forward_static_transfer",),
        provenance=provenance,
    )


def build_segnet_exact_forward_cpu_thread_control_v1() -> CanonicalEquation:
    """Build the measured, fail-closed thread-selection control law."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "re-run the finite thread tournament and full paired argmax gate after any host, "
            "Torch build, model-weight, receiver corpus, or thread-baseline change"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="apple_macos_arm64_cpu_torch_fp32",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="segnet_exact_forward_first64_thread_control_20260713",
        measurement_utc=_UTC,
        inputs={
            "n_real_pairs": N_REAL_PAIRS,
            "total_argmax_pixels": TOTAL_ARGMAX_PIXELS,
            "baseline_threads": BASELINE_THREADS,
            "candidate_threads": [1, 2, 3, 4, 5, 6],
            "selected_threads": SELECTED_THREADS,
            "seed": 0,
        },
        predicted_output={
            "admission_gate": (
                "selected_threads differs from baseline AND paired argmax_flip_count == 0 "
                "AND matched_speed_gap_ms > composed_timing_noise_floor_ms AND controls pass"
            )
        },
        empirical_output={
            "baseline_median_ms": BASELINE_MEDIAN_MS,
            "cheap_median_ms": CHEAP_MEDIAN_MS,
            "matched_speedup_x": MATCHED_SPEEDUP_X,
            "matched_speed_gap_ms": MATCHED_SPEED_GAP_MS,
            "composed_timing_noise_floor_ms": COMPOSED_TIMING_NOISE_FLOOR_MS,
            "argmax_flip_count": 0,
            "argmax_flip_rate": 0.0,
            "argmax_bit_identical": True,
            "verdict": "GO",
            "review_status": "fresh-eyes-reviewed(1)-CLEAN",
        },
        residual=0.0,
        source_artifact=_RECEIPT,
        measurement_method=(
            "finite 1..6 CPU-thread canary tournament followed by alternating-order paired "
            "fp32 forwards on the first 64 receiver-realized witness pairs; exact argmax and "
            "logit comparison; positive rerun and class-axis-rotation negative controls"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=COMPOSED_TIMING_NOISE_FLOOR_MS,
        noise_floor_provenance=(
            "baseline p95-p05 timing width plus candidate p95-p05 timing width in the same n64 run"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Frozen-SegNet exact-forward CPU thread admission law",
        one_line_summary=(
            "Choose the fastest finite CPU-thread arm only when a full paired receiver-realized "
            "argmax gate is exact and its timing gain exceeds the composed run-local floor."
        ),
        latex_form=(
            r"k^*=\arg\min_{k\in\{1,\ldots,k_0\}}\widetilde t_k,\quad "
            r"\mathrm{admit}(k^*)=[k^*\ne k_0]\,[F_{\rm argmax}=0]"
            r"\,[\widetilde t_{k_0}-\widetilde t_{k^*}>\epsilon_t]\,[C_+\land C_-]"
        ),
        python_callable_module_path="tools.probe_segnet_exact_forward:main",
        domain_of_validity={
            "scope_level": "registered formulation/substrate",
            "research_only": True,
            "review_status": "fresh-eyes-reviewed(1)-CLEAN",
            "included": [
                "macOS arm64 CPU",
                "Torch 2.12.1 fp32 frozen SegNet",
                "autograd-enabled forward with input gradients enabled",
                "first 64 receiver-realized witness pairs",
                "six-thread baseline and finite one-through-six candidate set",
            ],
            "excluded": [
                "MLX, Metal, MPS, CUDA, contest-CPU, another host, or another Torch build",
                "unseen receiver pairs or across-seed safety",
                "backward, optimizer-step, or full-training speed",
                "d_seg, d_pose, archive score, promotion, or pointer movement",
                "quantized forward and activation banking family verdicts",
            ],
            "authority": _AXIS,
        },
        units_in={"forward_time": "milliseconds", "argmax_flip_count": "pixels"},
        units_out={"matched_speedup": "dimensionless ratio", "admission": "boolean"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"argmax_flip_count": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tools.probe_segnet_exact_forward",),
        canonical_producers=("tools.probe_segnet_exact_forward",),
        provenance=provenance,
    )


def populate_segnet_exact_forward_cpu_thread_control_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Append the measured equation through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_segnet_exact_forward_cpu_thread_control_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="task456; research_only; score_claim=false; pointer unmoved",
    )
    return equation


__all__ = [
    "BASELINE_MEDIAN_MS",
    "BASELINE_THREADS",
    "CHEAP_MEDIAN_MS",
    "COMPOSED_TIMING_NOISE_FLOOR_MS",
    "EQUATION_ID",
    "MATCHED_SPEEDUP_X",
    "MATCHED_SPEED_GAP_MS",
    "N_REAL_PAIRS",
    "SELECTED_THREADS",
    "STATIC_PROCESS_V2_EQUATION_ID",
    "STATIC_PROCESS_V2_RECEIPTS",
    "TOTAL_ARGMAX_PIXELS",
    "StaticProcessReceiptError",
    "build_segnet_exact_forward_cpu_thread_control_v1",
    "build_segnet_exact_forward_cpu_thread_static_process_v2",
    "load_and_validate_static_process_receipt",
    "populate_segnet_exact_forward_cpu_thread_control_v1",
]
