#!/usr/bin/env python3
"""Build an evidence-bound fixed-d_seg comparison from two witness run dirs.

This is a slice-training receipt, not a contest score.  Wall time is derived
only from timestamped ``stage=verdict`` rows plus each arm's measured one-time
initialization cost; a caller-supplied final-run duration is intentionally not
accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

from tac.witness_init.fixed_quality import compare_fixed_quality

MEASUREMENT_SCHEMA = "tac.witness_init.fixed_quality_receipt.v1"
BLOCKER_SCHEMA = "tac.witness_init.fixed_quality_blocker.v1"
SELECTION_SCHEMA = "tac.witness_init.fresh_runtime.v1"
SELECTION_SCOPE = "init_time_spectral_selection_not_contest_score"
COMMITTED_SCHEMA = "tac.witness_init.fresh_committed_state.v1"
COMMITTED_SCOPE = "post_init_spectral_telemetry_not_contest_score"

_CUSTODY_KEYS = (
    "git_sha",
    "git_dirty",
    "upstream_snapshot_sha256",
    "seed",
    "axis",
    "mlx_device",
    "selection_surface",
    "target_authority_sha256",
)
_DYNAMIC_RESULT_KEYS = {
    "utc",
    "history",
    "checkpoint",
    "stage_checkpoints",
    "best",
    "fresh_init",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain one JSON object: {path}")
    return payload


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an exact non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be an exact non-negative integer")
    return result


def _exact_positive_int(value: object, *, name: str) -> int:
    result = _exact_nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be an exact positive integer")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be 64 hexadecimal characters")
    return value.lower()


def _canonical_payload_sha256(payload: object, *, name: str) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _parse_timestamp(value: object, *, line_number: int) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"verdict line {line_number} is missing an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"verdict line {line_number} has invalid timestamp {value!r}"
        ) from exc
    if parsed.utcoffset() is None:
        raise ValueError(
            f"verdict line {line_number} timestamp must include a UTC offset"
        )
    return parsed


def _parse_epoch(value: object, *, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"verdict line {line_number} has invalid epoch {value!r}")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"verdict line {line_number} has invalid epoch {value!r}")
    return numeric


def parse_verdict_history(run_log: str | Path) -> list[dict[str, float | int]]:
    """Parse unique realized verdict rows and derive elapsed time from epoch zero."""

    source = Path(run_log)
    if not source.is_file():
        raise FileNotFoundError(f"run log is missing: {source}")
    by_epoch: dict[int, tuple[float, datetime]] = {}
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("stage") != "verdict":
            continue
        raw_epoch = row.get("epoch")
        raw_d_seg = row.get("d_seg")
        epoch = _parse_epoch(raw_epoch, line_number=line_number)
        if isinstance(raw_d_seg, bool) or not isinstance(raw_d_seg, (int, float)):
            raise ValueError(f"verdict line {line_number} has invalid d_seg {raw_d_seg!r}")
        d_seg = float(raw_d_seg)
        if not math.isfinite(d_seg) or not 0.0 <= d_seg <= 1.0:
            raise ValueError(f"verdict line {line_number} has invalid d_seg {raw_d_seg!r}")
        timestamp = _parse_timestamp(row.get("ts"), line_number=line_number)
        prior = by_epoch.get(epoch)
        if prior is not None and prior != (d_seg, timestamp):
            raise ValueError(
                f"run log contains conflicting verdict rows for epoch {epoch}"
            )
        by_epoch[epoch] = (d_seg, timestamp)
    if not by_epoch:
        raise ValueError(f"run log has no timestamped verdict rows: {source}")
    ordered = sorted(by_epoch.items())
    if ordered[0][0] != 0:
        raise ValueError(
            f"run log verdict history must start at epoch 0; first emitted epoch is {ordered[0][0]}"
        )
    first_timestamp = ordered[0][1][1]
    history: list[dict[str, float | int]] = []
    for epoch, (d_seg, timestamp) in ordered:
        elapsed = (timestamp - first_timestamp).total_seconds()
        if elapsed < 0.0:
            raise ValueError("verdict timestamps are not monotone with epoch")
        history.append(
            {"epoch": epoch, "d_seg": d_seg, "elapsed_seconds": elapsed}
        )
    return history


def read_init_accounting(
    run_dir: str | Path,
    *,
    expected_mode: Literal["control", "select"],
) -> dict[str, Any]:
    """Read and cross-link canonical selection and committed-state receipts."""

    receipt = Path(run_dir) / "fresh_init_receipt.json"
    payload = _read_json_object(receipt, description="FreSh selection receipt")
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"FreSh selection receipt has non-canonical schema: {receipt}")
    if payload.get("claim_scope") != SELECTION_SCOPE:
        raise ValueError(f"FreSh selection receipt has non-canonical claim_scope: {receipt}")
    result = payload.get("result")
    provenance = payload.get("provenance")
    if not isinstance(result, dict) or not isinstance(provenance, dict):
        raise ValueError(f"malformed FreSh init receipt: {receipt}")
    if provenance.get("mode") != expected_mode:
        raise ValueError(
            f"FreSh selection receipt mode must be {expected_mode!r}: {receipt}"
        )
    selection_forwards = _exact_nonnegative_int(
        result.get("init_scorer_forward_calls"),
        name="selection init_scorer_forward_calls",
    )
    selection_pairs = _exact_nonnegative_int(
        result.get("init_scorer_pair_equivalents"),
        name="selection init_scorer_pair_equivalents",
    )
    total_target_pairs = _exact_positive_int(
        result.get("total_target_pairs"), name="selection total_target_pairs"
    )
    requested_samples = _exact_positive_int(
        result.get("requested_sample_count"),
        name="selection requested_sample_count",
    )
    initialization_draws = _exact_positive_int(
        result.get("initialization_draws"), name="selection initialization_draws"
    )
    if initialization_draws != 1:
        raise ValueError("selection initialization_draws must be exactly one")
    sampled_raw = result.get("sampled_pair_indices")
    if not isinstance(sampled_raw, list):
        raise ValueError("selection sampled_pair_indices must be a JSON array")
    sampled_pair_indices = tuple(
        _exact_nonnegative_int(value, name=f"sampled_pair_indices[{index}]")
        for index, value in enumerate(sampled_raw)
    )
    if len(sampled_pair_indices) != min(requested_samples, total_target_pairs):
        raise ValueError("selection sampled_pair_indices has the wrong exact count")
    if len(set(sampled_pair_indices)) != len(sampled_pair_indices):
        raise ValueError("selection sampled_pair_indices must be unique")
    if any(index >= total_target_pairs for index in sampled_pair_indices):
        raise ValueError("selection sampled_pair_indices contains an out-of-range pair")
    targets = result.get("targets")
    if not isinstance(targets, list) or len(targets) != len(sampled_pair_indices):
        raise ValueError("selection targets must match sampled_pair_indices")
    target_indices: list[int] = []
    target_custody: list[tuple[int, tuple[int, int], str, str, str]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValueError(f"selection targets[{index}] must be an object")
        pair_index = _exact_nonnegative_int(
            target.get("pair_index"), name=f"targets[{index}].pair_index"
        )
        target_indices.append(pair_index)
        raw_shape = target.get("shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 2:
            raise ValueError(f"selection targets[{index}].shape must have two dimensions")
        shape = tuple(
            _exact_positive_int(value, name=f"targets[{index}].shape[{axis}]")
            for axis, value in enumerate(raw_shape)
        )
        target_custody.append(
            (
                pair_index,
                (shape[0], shape[1]),
                _require_sha256(
                    target.get("label_sha256"), name=f"targets[{index}].label_sha256"
                ),
                _require_sha256(
                    target.get("boundary_sha256"),
                    name=f"targets[{index}].boundary_sha256",
                ),
                _require_sha256(
                    target.get("spectral_weight_sha256"),
                    name=f"targets[{index}].spectral_weight_sha256",
                ),
            )
        )
    if tuple(target_indices) != sampled_pair_indices:
        raise ValueError("selection target pair indices do not match sampled_pair_indices")
    candidate_count = _exact_positive_int(
        provenance.get("candidate_count"), name="selection candidate_count"
    )
    candidates = result.get("ordered_candidates")
    if not isinstance(candidates, list) or len(candidates) != candidate_count:
        raise ValueError("selection candidate_count must match ordered_candidates")
    if selection_forwards != candidate_count or selection_pairs != candidate_count:
        raise ValueError("selection scorer accounting must match the exact candidate count")
    if expected_mode == "control" and candidate_count != 1:
        raise ValueError("control selection receipt must contain exactly one candidate")
    if expected_mode == "select" and candidate_count <= 1:
        raise ValueError("select selection receipt must contain more than one candidate")

    matched_config = provenance.get("matched_config")
    if not isinstance(matched_config, dict):
        raise ValueError("selection provenance lacks matched_config")
    matched_config_sha = _require_sha256(
        provenance.get("matched_config_sha256"),
        name="selection matched_config_sha256",
    )
    if _canonical_payload_sha256(matched_config, name="matched_config") != matched_config_sha:
        raise ValueError("selection matched_config_sha256 does not match matched_config bytes")
    target_authority_sha = _require_sha256(
        provenance.get("target_authority_sha256"),
        name="selection target_authority_sha256",
    )

    selection_sha = _sha256(receipt)
    committed_receipt = Path(run_dir) / "fresh_init_post_structured_receipt.json"
    committed = _read_json_object(
        committed_receipt, description="FreSh committed-state receipt"
    )
    if committed.get("schema") != COMMITTED_SCHEMA:
        raise ValueError(
            f"FreSh committed-state receipt has non-canonical schema: {committed_receipt}"
        )
    if committed.get("claim_scope") != COMMITTED_SCOPE:
        raise ValueError(
            f"FreSh committed-state receipt has non-canonical claim_scope: {committed_receipt}"
        )
    linked_sha = _require_sha256(
        committed.get("selection_receipt_sha256"),
        name="committed-state selection_receipt_sha256",
    )
    if linked_sha != selection_sha:
        raise ValueError("committed-state receipt does not link the selected receipt bytes")
    committed_provenance = committed.get("provenance")
    committed_result = committed.get("result")
    if not isinstance(committed_provenance, dict) or not isinstance(committed_result, dict):
        raise ValueError(f"malformed FreSh committed-state receipt: {committed_receipt}")
    if committed_provenance.get("mode") != expected_mode:
        raise ValueError("committed-state receipt mode does not match arm mode")
    for key in ("git_sha", "git_dirty", "upstream_snapshot_sha256", "seed"):
        if committed_provenance.get(key) != provenance.get(key):
            raise ValueError(f"selection/committed custody mismatch for {key}")
    if committed_provenance.get("matched_config") != matched_config:
        raise ValueError("selection/committed matched_config payload mismatch")
    committed_config_sha = _require_sha256(
        committed_provenance.get("matched_config_sha256"),
        name="committed matched_config_sha256",
    )
    if committed_config_sha != matched_config_sha:
        raise ValueError("selection/committed matched_config_sha256 mismatch")
    if committed_provenance.get("target_authority_sha256") != target_authority_sha:
        raise ValueError("selection/committed target_authority_sha256 mismatch")

    total_forwards = _exact_nonnegative_int(
        committed_result.get("total_init_scorer_forward_calls"),
        name="committed total_init_scorer_forward_calls",
    )
    total_pairs = _exact_nonnegative_int(
        committed_result.get("total_init_scorer_pair_equivalents"),
        name="committed total_init_scorer_pair_equivalents",
    )
    total_seconds = _finite_nonnegative(
        committed_provenance.get("total_init_seconds_to_epoch0"),
        name="committed total_init_seconds_to_epoch0",
    )
    if total_forwards != selection_forwards + 1 or total_pairs != selection_pairs + 1:
        raise ValueError(
            "committed total init accounting must equal the selection sweep plus "
            "the one mandatory epoch-zero committed-state scorer forward"
        )
    return {
        "init_scorer_forward_calls": total_forwards,
        "init_scorer_pair_equivalents": total_pairs,
        "init_seconds": total_seconds,
        "total_target_pairs": total_target_pairs,
        "sampled_pair_indices": sampled_pair_indices,
        "target_custody": tuple(target_custody),
        "mode": expected_mode,
        "custody": {key: provenance.get(key) for key in _CUSTODY_KEYS},
        "fresh_config": provenance.get("config"),
        "matched_config": matched_config,
        "matched_config_sha256": matched_config_sha,
        "receipt_path": str(receipt),
        "receipt_sha256": selection_sha,
        "committed_receipt_path": str(committed_receipt),
        "committed_receipt_sha256": _sha256(committed_receipt),
    }


def read_run_authority(run_dir: str | Path) -> dict[str, Any]:
    """Read completed-run authority for pair count, epoch budget, and config."""

    result_path = Path(run_dir) / "result.json"
    payload = _read_json_object(result_path, description="witness result authority")
    n_pairs = _exact_positive_int(payload.get("n_pairs"), name="result n_pairs")
    epochs = _exact_nonnegative_int(payload.get("epochs"), name="result epochs")
    final_epoch = _exact_nonnegative_int(
        payload.get("final_epoch"), name="result final_epoch"
    )
    if final_epoch != epochs:
        raise ValueError("result authority is incomplete: final_epoch must equal epochs")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"result authority lacks provenance: {result_path}")
    result_config_fingerprint = {
        key: value for key, value in payload.items() if key not in _DYNAMIC_RESULT_KEYS
    }
    return {
        "n_pairs": n_pairs,
        "epochs": epochs,
        "provenance": provenance,
        "result_config_fingerprint": result_config_fingerprint,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
    }


def _validate_matched_arms(
    baseline_init: dict[str, Any],
    treatment_init: dict[str, Any],
    baseline_authority: dict[str, Any],
    treatment_authority: dict[str, Any],
) -> None:
    if baseline_init["custody"] != treatment_init["custody"]:
        raise ValueError("control and select arms have mismatched custody/provenance")
    if baseline_init["sampled_pair_indices"] != treatment_init["sampled_pair_indices"]:
        raise ValueError("control and select arms sampled different target pairs")
    if baseline_init["target_custody"] != treatment_init["target_custody"]:
        raise ValueError("control and select arms have mismatched target-pair custody")
    if baseline_init["matched_config_sha256"] != treatment_init["matched_config_sha256"]:
        raise ValueError("control and select arms have mismatched matched_config_sha256")
    if baseline_init["matched_config"] != treatment_init["matched_config"]:
        raise ValueError("control and select arms have mismatched matched_config payloads")
    baseline_config = baseline_init.get("fresh_config")
    treatment_config = treatment_init.get("fresh_config")
    if not isinstance(baseline_config, dict) or not isinstance(treatment_config, dict):
        raise ValueError("selection receipts must contain canonical config objects")
    baseline_common = {key: value for key, value in baseline_config.items() if key != "bias_grid"}
    treatment_common = {key: value for key, value in treatment_config.items() if key != "bias_grid"}
    if baseline_common != treatment_common:
        raise ValueError("control and select arms differ outside allowed FreSh grid deltas")
    if baseline_authority["provenance"] != treatment_authority["provenance"]:
        raise ValueError("control and select result authorities have mismatched provenance")
    if (
        baseline_authority["result_config_fingerprint"]
        != treatment_authority["result_config_fingerprint"]
    ):
        raise ValueError("control and select result authorities have mismatched run config")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_durable_output(path: Path) -> Path:
    destination = path.expanduser()
    resolved = destination.resolve(strict=False)
    temporary_root = Path("/tmp").resolve(strict=False)
    if resolved == temporary_root or temporary_root in resolved.parents:
        raise ValueError("measurement output must be durable and must not be below /tmp")
    if destination.exists() and destination.is_dir():
        raise ValueError(f"measurement output is a directory: {destination}")
    return destination


def build_measurement(
    baseline_run_dir: str | Path,
    treatment_run_dir: str | Path,
    *,
    fixed_epoch_budget: int,
    scorer_pairs_per_epoch: int,
    threshold_factor: float = 0.90,
) -> dict[str, Any]:
    """Return a complete receipt; raise when any authority input is missing."""

    baseline_dir = Path(baseline_run_dir)
    treatment_dir = Path(treatment_run_dir)
    baseline_log = baseline_dir / "run.log"
    treatment_log = treatment_dir / "run.log"
    baseline_history = parse_verdict_history(baseline_log)
    treatment_history = parse_verdict_history(treatment_log)
    baseline_init = read_init_accounting(baseline_dir, expected_mode="control")
    treatment_init = read_init_accounting(treatment_dir, expected_mode="select")
    baseline_authority = read_run_authority(baseline_dir)
    treatment_authority = read_run_authority(treatment_dir)
    _validate_matched_arms(
        baseline_init, treatment_init, baseline_authority, treatment_authority
    )
    authority_pairs = baseline_authority["n_pairs"]
    authority_budget = baseline_authority["epochs"]
    if treatment_authority["n_pairs"] != authority_pairs:
        raise ValueError("control and select result authorities disagree on n_pairs")
    if treatment_authority["epochs"] != authority_budget:
        raise ValueError("control and select result authorities disagree on epochs")
    if baseline_init["total_target_pairs"] != authority_pairs:
        raise ValueError("control selection total_target_pairs disagrees with result n_pairs")
    if treatment_init["total_target_pairs"] != authority_pairs:
        raise ValueError("select selection total_target_pairs disagrees with result n_pairs")
    for arm, init, authority in (
        ("control", baseline_init, baseline_authority),
        ("select", treatment_init, treatment_authority),
    ):
        for key in ("git_sha", "git_dirty", "upstream_snapshot_sha256", "seed"):
            if init["custody"].get(key) != authority["provenance"].get(key):
                raise ValueError(f"{arm} selection/result custody mismatch for {key}")
    requested_budget = _exact_nonnegative_int(
        fixed_epoch_budget, name="fixed_epoch_budget"
    )
    requested_pairs = _exact_positive_int(
        scorer_pairs_per_epoch, name="scorer_pairs_per_epoch"
    )
    if requested_budget != authority_budget:
        raise ValueError("fixed_epoch_budget disagrees with result authority epochs")
    if requested_pairs != authority_pairs:
        raise ValueError("scorer_pairs_per_epoch disagrees with result authority n_pairs")
    baseline_final_elapsed = float(baseline_history[-1]["elapsed_seconds"])
    treatment_final_elapsed = float(treatment_history[-1]["elapsed_seconds"])
    comparison = compare_fixed_quality(
        baseline_history,
        treatment_history,
        fixed_epoch_budget=authority_budget,
        scorer_pairs_per_epoch=authority_pairs,
        baseline_one_time_init_seconds=float(baseline_init["init_seconds"]),
        treatment_one_time_init_seconds=float(treatment_init["init_seconds"]),
        baseline_measured_total_wall_seconds=(
            float(baseline_init["init_seconds"]) + baseline_final_elapsed
        ),
        treatment_measured_total_wall_seconds=(
            float(treatment_init["init_seconds"]) + treatment_final_elapsed
        ),
        baseline_init_scorer_forward_calls=int(
            baseline_init["init_scorer_forward_calls"]
        ),
        treatment_init_scorer_forward_calls=int(
            treatment_init["init_scorer_forward_calls"]
        ),
        baseline_init_scorer_pair_equivalents=int(
            baseline_init["init_scorer_pair_equivalents"]
        ),
        treatment_init_scorer_pair_equivalents=int(
            treatment_init["init_scorer_pair_equivalents"]
        ),
        threshold_factor=threshold_factor,
        baseline_arm="fresh_init_control",
        treatment_arm="fresh_frequency_shift_init",
    )
    return {
        "schema": MEASUREMENT_SCHEMA,
        "claim_scope": (
            "faithful_slice_training_trajectory_advisory_not_contest_score"
        ),
        "measurement_config": {
            "fixed_epoch_budget": authority_budget,
            "scorer_pairs_per_epoch": authority_pairs,
            "count_authority": "per_arm_result_json_cross_checked_with_receipts_and_cli",
            "threshold_factor": float(threshold_factor),
            "wall_clock_source": (
                "verdict_row_ts_delta_from_epoch_zero_plus_total_init_seconds_to_epoch0"
            ),
        },
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": _sha256(Path(__file__).resolve()),
            "input_authority": (
                "timestamped_run_log_selection_and_committed_receipts_and_result_json"
            ),
            "score_authority": "none_upstream_evaluate_py_not_run",
        },
        "baseline": {
            "run_dir": str(baseline_dir),
            "run_log_sha256": _sha256(baseline_log),
            **baseline_authority,
            **baseline_init,
        },
        "treatment": {
            "run_dir": str(treatment_dir),
            "run_log_sha256": _sha256(treatment_log),
            **treatment_authority,
            **treatment_init,
        },
        "comparison": comparison.to_dict(),
        "n600_validation": "OWED_GOVERNED_LAUNCH_NOT_RUN",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--treatment-run-dir", required=True)
    parser.add_argument("--fixed-epoch-budget", type=int, required=True)
    parser.add_argument("--scorer-pairs-per-epoch", type=int, required=True)
    parser.add_argument("--threshold-factor", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = _validate_durable_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        payload = build_measurement(
            args.baseline_run_dir,
            args.treatment_run_dir,
            fixed_epoch_budget=args.fixed_epoch_budget,
            scorer_pairs_per_epoch=args.scorer_pairs_per_epoch,
            threshold_factor=args.threshold_factor,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "schema": BLOCKER_SCHEMA,
            "claim_scope": "measurement_blocker_no_epochs_reduction_claim",
            "baseline_run_dir": args.baseline_run_dir,
            "treatment_run_dir": args.treatment_run_dir,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "n600_validation": "OWED_GOVERNED_LAUNCH_NOT_RUN",
        }
        _atomic_write_json(output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 2
    _atomic_write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
