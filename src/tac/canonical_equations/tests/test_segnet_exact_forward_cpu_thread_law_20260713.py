from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.canonical_equations.segnet_exact_forward_cpu_thread_law_20260713 import (
    BASELINE_MEDIAN_MS,
    BASELINE_THREADS,
    CHEAP_MEDIAN_MS,
    COMPOSED_TIMING_NOISE_FLOOR_MS,
    EQUATION_ID,
    MATCHED_SPEED_GAP_MS,
    MATCHED_SPEEDUP_X,
    N_REAL_PAIRS,
    SELECTED_THREADS,
    STATIC_PROCESS_V2_EQUATION_ID,
    TOTAL_ARGMAX_PIXELS,
    StaticProcessReceiptError,
    build_segnet_exact_forward_cpu_thread_control_v1,
    build_segnet_exact_forward_cpu_thread_static_process_v2,
    load_and_validate_static_process_receipt,
    populate_segnet_exact_forward_cpu_thread_control_v1,
)


def test_equation_preserves_exact_argmax_and_noise_floor_gate() -> None:
    equation = build_segnet_exact_forward_cpu_thread_control_v1()
    anchor = equation.empirical_anchors[0]

    assert equation.equation_id == EQUATION_ID
    assert N_REAL_PAIRS == 64
    assert TOTAL_ARGMAX_PIXELS == 64 * 192 * 256 * 4
    assert BASELINE_THREADS == 6
    assert SELECTED_THREADS == 1
    assert anchor.empirical_output["argmax_flip_count"] == 0
    assert anchor.empirical_output["argmax_bit_identical"] is True
    assert pytest.approx(BASELINE_MEDIAN_MS / CHEAP_MEDIAN_MS) == MATCHED_SPEEDUP_X
    assert MATCHED_SPEED_GAP_MS > COMPOSED_TIMING_NOISE_FLOOR_MS
    assert anchor.noise_floor == COMPOSED_TIMING_NOISE_FLOOR_MS


def test_equation_constants_match_primary_receipt() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    receipt = json.loads(
        (
            repo_root
            / "experiments/results/segnet_exact_forward_20260713T020000Z/receipt.json"
        ).read_text()
    )
    measurement = receipt["measurement"]
    equation = build_segnet_exact_forward_cpu_thread_control_v1()
    empirical = equation.empirical_anchors[0].empirical_output

    assert receipt["verdict"] == empirical["verdict"] == "GO"
    assert measurement["baseline_forward"]["median_ms"] == BASELINE_MEDIAN_MS
    assert measurement["cheap_forward"]["median_ms"] == CHEAP_MEDIAN_MS
    assert measurement["matched_speed_gap_ms"] == MATCHED_SPEED_GAP_MS
    assert (
        measurement["composed_timing_noise_floor_ms"]
        == COMPOSED_TIMING_NOISE_FLOOR_MS
        == empirical["composed_timing_noise_floor_ms"]
    )
    assert receipt["economics"]["matched_forward_speedup_x"] == MATCHED_SPEEDUP_X
    assert measurement["argmax_flip_count"] == empirical["argmax_flip_count"] == 0


def test_equation_refuses_cross_substrate_inference() -> None:
    equation = build_segnet_exact_forward_cpu_thread_control_v1()
    excluded = equation.domain_of_validity["excluded"]

    assert any("CUDA" in item and "contest-CPU" in item for item in excluded)
    assert any("unseen receiver pairs" in item for item in excluded)
    assert any("full-training speed" in item for item in excluded)
    assert any("quantized forward" in item for item in excluded)


def test_populate_uses_append_only_registry(tmp_path) -> None:
    path = tmp_path / "equations.jsonl"
    lock_path = tmp_path / "equations.lock"

    populate_segnet_exact_forward_cpu_thread_control_v1(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="task456-test",
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["equation_id"] == EQUATION_ID
    assert rows[0]["event_type"] == "registered"


def _artifact_ref(root: Path, name: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps({"artifact": name}, sort_keys=True))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _static_receipt(torch_build: str, artifact_root: Path) -> dict[str, object]:
    """Injectable terminal contract; no fixture is evidence of a real n600 run."""

    stages = ("baseline_rep0", "selected_rep0", "selected_rep1", "baseline_rep1")
    sequence_sha = "a" * 64
    pair78_sha = "b" * 64
    baseline_samples = [900.0] * 600
    selected_samples = [300.0] * 600
    pass_receipts = {
        stage: {
            "measurement_child_id": f"measure-{index}",
            "measurement_pid": 1000 + index,
            "replay_child_ids": [f"replay-{index}"],
            "replay_pids": [2000 + index],
            "strategy": "eager_nchw_autograd",
            "intraop_threads": 6 if stage.startswith("baseline") else 1,
            "interop_threads": 1,
            "measurement_complete": True,
            "replay_complete": True,
            "measurement_process_segments": [
                {
                    "child_id": f"measure-{index}",
                    "pid": 1000 + index,
                    "binding": {
                        "intraop_threads": 6 if stage.startswith("baseline") else 1,
                        "interop_threads": 1,
                    },
                    "started_from_completed_pairs": 0,
                    "completed_pairs": 600,
                    "started_at_utc": "2026-07-13T04:00:00Z",
                    "completed_at_utc": "2026-07-13T04:10:00Z",
                }
            ],
            "replay_process_segments": [
                {
                    "child_id": f"replay-{index}",
                    "pid": 2000 + index,
                    "binding": {
                        "intraop_threads": 6 if stage.startswith("baseline") else 1,
                        "interop_threads": 1,
                    },
                    "started_from_completed_pairs": 0,
                    "completed_pairs": 600,
                    "started_at_utc": "2026-07-13T04:10:00Z",
                    "completed_at_utc": "2026-07-13T04:20:00Z",
                }
            ],
            "measurement_sequence_sha256": sequence_sha,
            "replay_sequence_sha256": sequence_sha,
            "binding_before": {
                "intraop_threads": 6 if stage.startswith("baseline") else 1,
                "interop_threads": 1,
            },
            "binding_after": {
                "intraop_threads": 6 if stage.startswith("baseline") else 1,
                "interop_threads": 1,
            },
            "replay_binding_before": {
                "intraop_threads": 6 if stage.startswith("baseline") else 1,
                "interop_threads": 1,
            },
            "replay_binding_after": {
                "intraop_threads": 6 if stage.startswith("baseline") else 1,
                "interop_threads": 1,
            },
            "terminal_stage_file": _artifact_ref(artifact_root, f"{stage}-measurement.json"),
            "terminal_replay_file": _artifact_ref(artifact_root, f"{stage}-replay.json"),
            "derived_argmax_flip_count": 0,
        }
        for index, stage in enumerate(stages)
    }
    return {
        "schema": "frozen_segnet_exact_forward_static_transfer_probe_v2",
        "completed_at_utc": "2026-07-13T05:00:00Z",
        "verdict": "GO",
        "verdict_scope": (
            "fresh-child process-static ABBA formulation over first 600 receiver-realized pairs on the "
            "fingerprinted local macOS CPU/Torch build only; n<600 diagnostic; no transfer to another "
            "host/build/model/input set, backward, full training, contest-CPU/CUDA, evaluator, d_seg, "
            "d_pose, archive, score, or promotion"
        ),
        "research_only": True,
        "axis": "[macOS-CPU advisory; process-static torch-fp32 training-forward; no MPS/CUDA]",
        "labels": {
            "canary_count": "ASSUMED_HEURISTIC_SCREEN_ONLY",
            "checkpoint_interval": "ASSUMED_RECOVERY_ENVELOPE",
            "timing_and_sha": "MEASURED",
            "zero_flip_from_sha": "DERIVED",
        },
        "validation": {"status": "self-validated-from-terminal-child-bytes-before-and-after-write"},
        "authority": {"score_claim": False, "pointer_moved": False, "promotion_eligible": False},
        "runtime": {
            "torch": torch_build,
            "mps_used": False,
            "cuda_used": False,
            "contest_cpu_timing_measured": False,
        },
        "selected_arm": {"strategy": "eager_nchw_autograd", "threads": 1, "baseline_threads": 6},
        "measurement": {
            "method": "fresh_child_process_static_threads",
            "thread_binding": (
                "fresh process per measurement and replay; intra/inter-op immutable after pre-model binding"
            ),
            "n_real_pairs": 600,
            "total_argmax_pixels": 600 * 192 * 256,
            "stage_order": list(stages),
            "sequence_sha256": dict.fromkeys(stages, sequence_sha),
            "replay_sequence_sha256": dict.fromkeys(stages, sequence_sha),
            "all_sequence_shas_equal": True,
            "argmax_flip_count": 0,
            "derived_argmax_flip_count": 0,
            "argmax_flip_rate": 0.0,
            "label": "MEASURED",
            "input_gradient_graph_preserved": True,
            "pair_sha_evidence": {
                "all_pair_sha256_equal": True,
                "derived_argmax_flip_count": 0,
                "mismatch_pair_count": 0,
                "first_mismatch": None,
                "flip_count_derivation": "DERIVED_ZERO_FROM_EIGHT_WAY_EXACT_PER_PAIR_SHA_EQUALITY",
            },
            "baseline_per_pair_replica_median": {"count": 600, "median_ms": 900.0, "samples_ms": baseline_samples},
            "selected_per_pair_replica_median": {"count": 600, "median_ms": 300.0, "samples_ms": selected_samples},
            "static_paired_speedup_x": 3.0,
            "matched_sign_alpha": 0.01,
            "matched_sign_alpha_provenance": "OPERATOR_SEALED_TRANSFER_V4_FALSE_POSITIVE_BUDGET",
            "matched_sign_test": {
                "alpha": 0.01,
                "label": "DERIVED_FROM_MATCHED_MEASUREMENTS",
                "wins": 600,
                "losses": 0,
                "ties": 0,
                "one_sided_exact_binomial_pvalue": float(1 / (2**600)),
            },
            "child_passes": list(stages),
            "pass_receipts": pass_receipts,
            "process_segments_per_pass": dict.fromkeys(stages, 1),
            "independent_full_replays": {
                "count": 4,
                "per_arm_count": 2,
                "complete": True,
                "independent_processes": True,
                "unique_child_id_count": 8,
                "unique_pid_count": 8,
                "sha_equal": True,
            },
            "pair78": {
                "index": 78,
                "stable": True,
                "resolved": True,
                "per_pass_sha256": dict.fromkeys(stages, pair78_sha),
                "per_replay_sha256": dict.fromkeys(stages, pair78_sha),
            },
        },
    }


def _write_static_receipt(path: Path, receipt: dict[str, object]) -> Path:
    path.write_text(json.dumps(receipt, sort_keys=True))
    return path


def _static_no_go_receipt(torch_build: str, artifact_root: Path) -> dict[str, object]:
    """Legitimate scoped negative: each arm replays exactly, arms disagree."""

    receipt = _static_receipt(torch_build, artifact_root)
    measurement = receipt["measurement"]
    assert isinstance(measurement, dict)
    selected_sha = "d" * 64
    selected_pair78_sha = "e" * 64
    for stage in ("selected_rep0", "selected_rep1"):
        sequence = measurement["sequence_sha256"]
        replays = measurement["replay_sequence_sha256"]
        passes = measurement["pass_receipts"]
        assert isinstance(sequence, dict) and isinstance(replays, dict) and isinstance(passes, dict)
        sequence[stage] = selected_sha
        replays[stage] = selected_sha
        row = passes[stage]
        assert isinstance(row, dict)
        row["measurement_sequence_sha256"] = selected_sha
        row["replay_sequence_sha256"] = selected_sha
    passes = measurement["pass_receipts"]
    assert isinstance(passes, dict)
    for stage in ("baseline_rep0", "selected_rep0", "selected_rep1", "baseline_rep1"):
        row = passes[stage]
        assert isinstance(row, dict)
        row["derived_argmax_flip_count"] = None
    measurement["all_sequence_shas_equal"] = False
    measurement["argmax_flip_count"] = None
    measurement["derived_argmax_flip_count"] = None
    measurement["argmax_flip_rate"] = None
    pair_evidence = measurement["pair_sha_evidence"]
    assert isinstance(pair_evidence, dict)
    pair_evidence.update(
        {
            "all_pair_sha256_equal": False,
            "derived_argmax_flip_count": None,
            "mismatch_pair_count": 15,
            "first_mismatch": {"pair_index": 35},
            "flip_count_derivation": "UNAVAILABLE_SHA_MISMATCH_FAIL_CLOSED_NO_RAW_PRIOR_TENSOR",
        }
    )
    full_replays = measurement["independent_full_replays"]
    assert isinstance(full_replays, dict)
    full_replays["sha_equal"] = False
    pair78 = measurement["pair78"]
    assert isinstance(pair78, dict)
    for key in ("per_pass_sha256", "per_replay_sha256"):
        mapping = pair78[key]
        assert isinstance(mapping, dict)
        for stage in ("selected_rep0", "selected_rep1"):
            mapping[stage] = selected_pair78_sha
    pair78["stable"] = False
    pair78["resolved"] = False
    receipt["verdict"] = "NO-GO"
    return receipt


def test_static_process_v2_loads_both_fixture_receipts_and_keeps_advisory_scope(tmp_path) -> None:
    paths = {
        "torch_2_12_1": _write_static_receipt(
            tmp_path / "torch2121.json", _static_receipt("2.12.1", tmp_path / "a")
        ),
        "torch_2_12_0": _write_static_receipt(
            tmp_path / "torch2120.json", _static_receipt("2.12.0", tmp_path / "b")
        ),
    }

    equation = build_segnet_exact_forward_cpu_thread_static_process_v2(receipt_paths=paths)

    assert equation.equation_id == STATIC_PROCESS_V2_EQUATION_ID
    assert len(equation.empirical_anchors) == 2
    assert all(anchor.inputs["n_real_pairs"] == 600 for anchor in equation.empirical_anchors)
    assert all(anchor.inputs["total_argmax_pixels"] == 600 * 192 * 256 for anchor in equation.empirical_anchors)
    assert all(anchor.empirical_output["static_paired_speedup_x"] == 3.0 for anchor in equation.empirical_anchors)
    assert equation.last_calibration_utc == "2026-07-13T05:00:00Z"
    assert "contest-CPU and contest-CUDA timing" in equation.domain_of_validity["excluded"]
    assert "score_claim=false" in equation.domain_of_validity["authority"]


def test_static_process_v2_refuses_missing_receipt(tmp_path) -> None:
    paths = {
        "torch_2_12_1": tmp_path / "missing.json",
        "torch_2_12_0": _write_static_receipt(
            tmp_path / "torch2120.json", _static_receipt("2.12.0", tmp_path / "b")
        ),
    }

    with pytest.raises(StaticProcessReceiptError, match="missing"):
        build_segnet_exact_forward_cpu_thread_static_process_v2(receipt_paths=paths)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda receipt: receipt["measurement"].pop("method"), "fresh child process-static"),
        (lambda receipt: receipt["runtime"].update({"torch": "2.12.9"}), "Torch build mismatch"),
        (lambda receipt: receipt["measurement"]["independent_full_replays"].update({"complete": False}), "full replay"),
        (lambda receipt: receipt["measurement"].update({"static_paired_speedup_x": 99.0}), "speedup"),
    ],
)
def test_static_process_v2_refuses_malformed_build_replay_and_tampered_speedup(
    tmp_path, mutation, error
) -> None:
    receipt = _static_receipt("2.12.1", tmp_path / "artifacts")
    mutation(receipt)
    path = _write_static_receipt(tmp_path / "receipt.json", receipt)

    with pytest.raises(StaticProcessReceiptError, match=error):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")


def test_static_process_v2_loads_legitimate_no_go_without_zero_flip_coercion(tmp_path) -> None:
    receipt = _static_no_go_receipt("2.12.1", tmp_path / "artifacts")
    path = _write_static_receipt(tmp_path / "receipt.json", receipt)

    result = load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")

    assert result["verdict"] == "NO-GO"
    assert result["admitted"] is False
    assert result["argmax_flip_count"] is None
    assert result["mismatch_pair_count"] == 15
    assert result["pair78_resolved"] is False
    assert set(result["sequence_sha256"]) == {"baseline_rep0", "selected_rep0", "selected_rep1", "baseline_rep1"}


def test_static_process_v2_refuses_tampered_no_go_verdict_or_false_zero_flip(tmp_path) -> None:
    receipt = _static_no_go_receipt("2.12.1", tmp_path / "artifacts")
    receipt["verdict"] = "GO"
    path = _write_static_receipt(tmp_path / "verdict.json", receipt)
    with pytest.raises(StaticProcessReceiptError, match="verdict"):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")

    receipt = _static_no_go_receipt("2.12.1", tmp_path / "zero-artifacts")
    measurement = receipt["measurement"]
    assert isinstance(measurement, dict)
    measurement["argmax_flip_count"] = 0
    measurement["derived_argmax_flip_count"] = 0
    pair_evidence = measurement["pair_sha_evidence"]
    assert isinstance(pair_evidence, dict)
    pair_evidence["derived_argmax_flip_count"] = 0
    path = _write_static_receipt(tmp_path / "zero.json", receipt)
    with pytest.raises(StaticProcessReceiptError, match="flip authority"):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")


def test_static_process_v2_refuses_stage_replay_mismatch_and_pair78_coercion(tmp_path) -> None:
    receipt = _static_no_go_receipt("2.12.1", tmp_path / "artifacts")
    measurement = receipt["measurement"]
    assert isinstance(measurement, dict)
    passes = measurement["pass_receipts"]
    assert isinstance(passes, dict)
    selected = passes["selected_rep0"]
    assert isinstance(selected, dict)
    selected["replay_sequence_sha256"] = "f" * 64
    path = _write_static_receipt(tmp_path / "replay.json", receipt)
    with pytest.raises(StaticProcessReceiptError, match="measurement/replay sequence SHA mismatch"):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")

    receipt = _static_no_go_receipt("2.12.1", tmp_path / "pair78-artifacts")
    measurement = receipt["measurement"]
    assert isinstance(measurement, dict)
    pair78 = measurement["pair78"]
    assert isinstance(pair78, dict)
    pair78["stable"] = True
    pair78["resolved"] = True
    path = _write_static_receipt(tmp_path / "pair78.json", receipt)
    with pytest.raises(StaticProcessReceiptError, match="pair78 stable/resolved"):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")


def test_static_process_v2_refuses_reused_replay_pid(tmp_path) -> None:
    receipt = _static_receipt("2.12.1", tmp_path / "artifacts")
    measurement = receipt["measurement"]
    assert isinstance(measurement, dict)
    pass_receipts = measurement["pass_receipts"]
    assert isinstance(pass_receipts, dict)
    selected = pass_receipts["selected_rep0"]
    assert isinstance(selected, dict)
    selected["replay_pids"] = [2000]
    path = _write_static_receipt(tmp_path / "receipt.json", receipt)

    with pytest.raises(StaticProcessReceiptError, match="distinct PID"):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")


def test_static_process_v2_refuses_tampered_terminal_artifact_bytes(tmp_path) -> None:
    receipt = _static_receipt("2.12.1", tmp_path / "artifacts")
    measurement = receipt["measurement"]
    assert isinstance(measurement, dict)
    pass_receipts = measurement["pass_receipts"]
    assert isinstance(pass_receipts, dict)
    baseline = pass_receipts["baseline_rep0"]
    assert isinstance(baseline, dict)
    terminal_replay = baseline["terminal_replay_file"]
    assert isinstance(terminal_replay, dict)
    Path(str(terminal_replay["path"])).write_text("tampered")
    path = _write_static_receipt(tmp_path / "receipt.json", receipt)

    with pytest.raises(StaticProcessReceiptError, match="byte SHA"):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")


def test_static_process_v2_refuses_broadened_scope_or_unbound_process_segment(tmp_path) -> None:
    receipt = _static_no_go_receipt("2.12.1", tmp_path / "scope-artifacts")
    receipt["verdict_scope"] = "all hosts and builds"
    path = _write_static_receipt(tmp_path / "scope.json", receipt)
    with pytest.raises(StaticProcessReceiptError, match="verdict_scope"):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")

    receipt = _static_no_go_receipt("2.12.1", tmp_path / "segment-artifacts")
    measurement = receipt["measurement"]
    assert isinstance(measurement, dict)
    pass_receipts = measurement["pass_receipts"]
    assert isinstance(pass_receipts, dict)
    selected = pass_receipts["selected_rep0"]
    assert isinstance(selected, dict)
    segments = selected["replay_process_segments"]
    assert isinstance(segments, list) and isinstance(segments[0], dict)
    segments[0]["child_id"] = "forged-replay-child"
    path = _write_static_receipt(tmp_path / "segment.json", receipt)
    with pytest.raises(StaticProcessReceiptError, match="segment does not bind"):
        load_and_validate_static_process_receipt(path, expected_torch_build="2.12.1")
