# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import platform
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools/probe_segnet_exact_forward_static_transfer.py"
SPEC = importlib.util.spec_from_file_location("probe_segnet_exact_forward_static_transfer", TOOL)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _selected() -> dict[str, object]:
    return {"strategy": "eager_nchw_autograd", "threads": 2, "baseline_threads": 6}


def test_static_admission_requires_n600_eight_way_sha_replay_and_uninterrupted_passes() -> None:
    sha = "a" * 64
    valid = {
        "n_pairs": 600,
        "selected": _selected(),
        "sequence_sha256": dict.fromkeys(probe.STAGE_ORDER, sha),
        "replay_sequence_sha256": dict.fromkeys(probe.STAGE_ORDER, sha),
        "total_flip_count": 0,
        "baseline_median_ms": 900.0,
        "selected_median_ms": 300.0,
        "matched_sign_pvalue": 0.001,
        "replay_complete": True,
        "process_segments_per_pass": dict.fromkeys(probe.STAGE_ORDER, 1),
        "independent_processes": True,
    }
    assert probe.static_admission(**valid)
    mutations = (
        {"n_pairs": 599},
        {"total_flip_count": None},
        {"selected_median_ms": 900.0},
        {"matched_sign_pvalue": 0.02},
        {"replay_complete": False},
        {"independent_processes": False},
        {"process_segments_per_pass": {**dict.fromkeys(probe.STAGE_ORDER, 1), "baseline_rep1": 2}},
        {
            "selected": {
                "strategy": "eager_nchw_autograd",
                "threads": 6,
                "baseline_threads": 6,
            }
        },
        {
            "replay_sequence_sha256": {
                **dict.fromkeys(probe.STAGE_ORDER, sha),
                "selected_rep1": "b" * 64,
            }
        },
    )
    for mutation in mutations:
        assert not probe.static_admission(**(valid | mutation))


def test_n_less_than_600_is_structurally_diagnostic_only() -> None:
    assert probe.diagnostic_verdict(admitted=False, n_pairs=599) == "DIAGNOSTIC_ONLY"
    assert probe.diagnostic_verdict(admitted=True, n_pairs=599) == "DIAGNOSTIC_ONLY"
    assert probe.diagnostic_verdict(admitted=False, n_pairs=600) == "NO-GO"


def _config(tmp_path: Path, *, mode: str = "measurement", n_pairs: int = 2) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": probe.WORKER_CONFIG_SCHEMA,
        "mode": mode,
        "stage": "baseline_rep0",
        "raw_path": str(tmp_path / "x.raw"),
        "raw_bytes": 1,
        "expected_raw_sha256": "r" * 64,
        "n_pairs": n_pairs,
        "seed": 0,
        "strategy": "eager_nchw_autograd",
        "intraop_threads": 6,
        "interop_threads": 1,
        "checkpoint_interval": probe.CHECKPOINT_INTERVAL,
        "checkpoint_interval_provenance": probe.CHECKPOINT_INTERVAL_PROVENANCE,
        "fingerprint_sha256": "f" * 64,
        "expected_weights_sha256": "w" * 64,
        "expected_torch_build": "test",
        "expected_python_executable": "/python",
        "expected_tool_sha256": "t" * 64,
        "expected_v4_tool_sha256": "v" * 64,
        "pythonhashseed": "0",
        "output_path": str(tmp_path / f"{mode}.json"),
        "measurement_path": str(tmp_path / "measurement.json"),
        "measurement_sha256": None,
        "method": probe.METHOD,
        "thread_environment": {},
    }
    return probe._with_sha(payload)


def _stage(config: dict[str, object], *, completed: int, terminal: bool) -> dict[str, object]:
    mode = str(config["mode"])
    n_pairs = int(config["n_pairs"])
    record = probe._empty_stage_record(config)
    record["completed_pairs"] = completed
    record["pair_sha256"] = ["a" * 64] * completed
    record["timings_ms"] = [1.0] * completed if mode == "measurement" else []
    record["process_segments"] = [
        {
            "child_id": "one",
            "pid": 1,
            "started_from_completed_pairs": 0,
            "binding": {"intraop_threads": 6, "interop_threads": 1},
            "python_executable": "/python",
            "pythonhashseed": "0",
            "started_at_utc": "2026-07-13T00:00:00Z",
        }
    ]
    if terminal:
        record["sequence_sha256"] = "b" * 64
        record["pair78_sha256"] = "c" * 64 if n_pairs > 78 else None
        record["total_argmax_pixels"] = n_pairs * probe.base.SEG_H * probe.base.SEG_W
        record["binding_before"] = {"intraop_threads": 6, "interop_threads": 1}
        record["binding_after"] = {"intraop_threads": 6, "interop_threads": 1}
        record["measurement_complete"] = mode == "measurement"
        record["replay_complete"] = mode == "replay"
        record["terminal_child_id"] = "one"
        record["terminal_pid"] = 1
        record["process_segments"][0]["completed_pairs"] = n_pairs
        record["process_segments"][0]["completed_at_utc"] = "2026-07-13T00:00:01Z"
    return probe._with_sha(record)


def test_last_pair_without_sequence_digest_is_recoverable_not_terminal(tmp_path: Path) -> None:
    config = _config(tmp_path)
    incomplete = _stage(config, completed=2, terminal=False)
    assert probe.validate_stage_record(incomplete, config=config) == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        probe.validate_stage_record(incomplete, config=config, require_complete=True)


def test_process_segment_is_banked_at_completed_zero_before_any_forward(tmp_path: Path) -> None:
    config = _config(tmp_path)
    record = probe._empty_stage_record(config)
    segment = {
        "child_id": "launch-child",
        "pid": 77,
        "started_from_completed_pairs": 0,
        "binding": {"intraop_threads": 6, "interop_threads": 1},
        "python_executable": "/python",
        "pythonhashseed": "0",
        "started_at_utc": "2026-07-13T00:00:00Z",
    }
    output = tmp_path / "measurement.json"
    probe._persist_process_launch(record=record, segment=segment, output_path=output)
    persisted = json.loads(output.read_text())
    assert persisted["completed_pairs"] == 0
    assert persisted["process_segments"] == [segment]
    assert persisted["measurement_complete"] is False
    assert persisted["sequence_sha256"] is None
    probe.validate_stage_record(persisted, config=config, require_complete=False)

    # Completion must update the persisted copy, not the caller's detached
    # segment mapping.
    probe._complete_process_segment(
        record=record,
        child_id="launch-child",
        pid=77,
        n_pairs=2,
        completed_at_utc="2026-07-13T00:00:01Z",
    )
    assert record["process_segments"][-1]["completed_pairs"] == 2
    assert record["process_segments"][-1]["completed_at_utc"] == "2026-07-13T00:00:01Z"
    assert "completed_pairs" not in segment
    with pytest.raises(RuntimeError, match="identity drift"):
        probe._complete_process_segment(
            record=record,
            child_id="different-child",
            pid=77,
            n_pairs=2,
            completed_at_utc="2026-07-13T00:00:02Z",
        )


def test_run_identity_survives_new_lock_pid_and_start_time(tmp_path: Path) -> None:
    argv = ["--raw", "x.raw", "--out", str(tmp_path / "receipt.json")]
    first = probe.load_or_create_run_identity(
        checkpoint_dir=tmp_path,
        raw_argv=argv,
        lock_owner={
            "pid": 10,
            "host": "host",
            "started_at_utc": "2026-07-13T00:00:00Z",
            "lock_path": str(tmp_path / "run.lock"),
        },
    )
    resumed = probe.load_or_create_run_identity(
        checkpoint_dir=tmp_path,
        raw_argv=argv,
        lock_owner={
            "pid": 99,
            "host": "host",
            "started_at_utc": "2026-07-13T01:00:00Z",
            "lock_path": str(tmp_path / "run.lock"),
        },
    )
    assert resumed == first
    assert resumed["initial_lock_owner"]["pid"] == 10
    assert resumed["sha256"] == probe.sha256_json(probe._without_sha(resumed))


def test_stage_terminal_binding_and_config_tamper_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    terminal = _stage(config, completed=2, terminal=True)
    assert probe.validate_stage_record(terminal, config=config, require_complete=True) == 2

    bad_binding = deepcopy(terminal)
    bad_binding["binding_after"]["intraop_threads"] = 1
    bad_binding = probe._with_sha(bad_binding)
    with pytest.raises(RuntimeError, match="binding custody"):
        probe.validate_stage_record(bad_binding, config=config, require_complete=True)

    bad_config = deepcopy(terminal)
    bad_config["config_sha256"] = "0" * 64
    bad_config = probe._with_sha(bad_config)
    with pytest.raises(RuntimeError, match="stage custody mismatch"):
        probe.validate_stage_record(bad_config, config=config)


def test_measurement_and_replay_stage_shapes_are_distinct(tmp_path: Path) -> None:
    measurement_config = _config(tmp_path, mode="measurement")
    replay_config = _config(tmp_path, mode="replay")
    measurement = _stage(measurement_config, completed=2, terminal=True)
    replay = _stage(replay_config, completed=2, terminal=True)
    assert len(measurement["timings_ms"]) == 2
    assert replay["timings_ms"] == []
    probe.validate_stage_record(measurement, config=measurement_config, require_complete=True)
    probe.validate_stage_record(replay, config=replay_config, require_complete=True)


def test_failure_markers_are_append_only_atomic_scoped_and_false_authority(tmp_path: Path) -> None:
    paths = [
        probe.write_failure_checkpoint(
            checkpoint_dir=tmp_path,
            fingerprint_sha256="f" * 64,
            stage="selected_rep0_replay",
            pair_index=78,
            expected_sha256="a" * 64,
            observed_sha256="b" * 64,
            flip_count=None,
            reason="unit-test drift",
            torch_build="2.12.test",
        )
        for _ in range(2)
    ]
    assert paths[0] != paths[1]
    assert all(path.is_file() for path in paths)
    payload = json.loads(paths[0].read_text())
    assert payload["argmax_flip_count"] is None
    assert payload["argmax_flip_count_authority"] == "UNAVAILABLE_SHA_MISMATCH_FAIL_CLOSED"
    assert payload["authority"] == probe.AUTHORITY
    assert "formulation" in payload["verdict_scope"]
    probe._validate_digest(payload, label="failure")
    assert not list(tmp_path.glob("*.tmp"))


def test_per_pair_replica_medians_follow_abba_pairing() -> None:
    records = {
        "baseline_rep0": {"timings_ms": [10.0, 20.0]},
        "selected_rep0": {"timings_ms": [4.0, 8.0]},
        "selected_rep1": {"timings_ms": [6.0, 10.0]},
        "baseline_rep1": {"timings_ms": [14.0, 24.0]},
    }
    baseline, selected = probe.per_pair_replica_medians(records)
    assert baseline == [12.0, 22.0]
    assert selected == [5.0, 9.0]


def test_only_binding_function_may_mutate_torch_thread_counts() -> None:
    tree = ast.parse(TOOL.read_text())
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"set_num_threads", "set_num_interop_threads"}:
            continue
        owner = parents.get(node)
        while owner is not None and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parents.get(owner)
        calls.append((node.func.attr, owner.name if isinstance(owner, ast.FunctionDef) else None))
    assert calls == [
        ("set_num_interop_threads", "_bind_static_threads"),
        ("set_num_threads", "_bind_static_threads"),
        ("set_num_interop_threads", "run_probe"),
    ]


def test_fingerprint_binds_seed_interop_canary_selection_policy_and_environment(tmp_path: Path) -> None:
    raw = tmp_path / "x.raw"
    raw.write_bytes(b"x")
    selection_key = {"sha256": "s" * 64}
    custody = {
        "raw_bytes": 1,
        "raw_sha256": "r" * 64,
        "weights_sha256": "w" * 64,
        "dependency_sha256": "d" * 64,
    }
    run_identity = probe._with_sha(
        {
            "schema": probe.RUN_IDENTITY_SCHEMA,
            "run_id": "run",
            "created_at_utc": "2026-07-13T00:00:00Z",
            "argv": [probe.active_python_executable(), str(probe.TOOL_PATH), "--raw", str(raw)],
            "initial_git": {"head": "h", "branch": "main"},
            "initial_lock_owner": {"pid": 1, "lock_path": "/lock"},
            "authority": probe.AUTHORITY,
        }
    )
    fingerprint = probe.build_run_fingerprint(
        raw=raw,
        n_pairs=600,
        seed=7,
        interop_threads=1,
        canary_indices=[0, 300, 599],
        selection_key=selection_key,
        selected=_selected(),
        start_custody=custody,
        policy_contracts={"x": {"process_lifecycle": {"method": probe.METHOD}}},
        out=tmp_path / "receipt.json",
        raw_argv=["--raw", str(raw), "--out", str(tmp_path / "receipt.json")],
        run_identity=run_identity,
    )
    payload = fingerprint["payload"]
    assert payload["seed"] == 7
    assert payload["interop_threads"] == 1
    assert payload["canary_indices"] == [0, 300, 599]
    assert payload["selected_arm"] == _selected()
    assert payload["method"] == probe.METHOD
    assert payload["checkpoint_interval_provenance"] == probe.CHECKPOINT_INTERVAL_PROVENANCE
    assert payload["worker_pythonhashseed"] == "7"
    assert payload["run_identity"] == run_identity
    assert fingerprint["sha256"] == probe.sha256_json(payload)


def test_cli_defaults_to_n600_and_worker_mode_is_internal() -> None:
    args = probe.parse_args(
        [
            "--raw",
            "does-not-need-to-exist.raw",
            "--out",
            "experiments/results/static-transfer/receipt.json",
        ]
    )
    assert args.n_pairs == 600
    assert args.out == (REPO_ROOT / "experiments/results/static-transfer/receipt.json").resolve()
    assert probe._parse_worker_path(["--_worker-config", "/durable/config.json"]) == Path(
        "/durable/config.json"
    )
    with pytest.raises(SystemExit):
        probe._parse_worker_path(["--_worker-config"])


def test_child_launch_preserves_virtualenv_executable_instead_of_resolving_symlink() -> None:
    assert probe.active_python_executable() == str(Path(sys.executable).absolute())
    source = TOOL.read_text()
    launch_line = next(line for line in source.splitlines() if "command = [active_python_executable()" in line)
    assert ".resolve()" not in launch_line


def _plan_contract_fixture() -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    selected = {"strategy": "eager_nchw_autograd", "threads": 1, "baseline_threads": 6}
    run_identity = probe._with_sha(
        {
            "schema": probe.RUN_IDENTITY_SCHEMA,
            "run_id": "run",
            "created_at_utc": "2026-07-13T00:00:00Z",
            "argv": [probe.active_python_executable(), str(probe.TOOL_PATH), "--raw", "x.raw"],
            "initial_git": {"head": "abc", "branch": "main"},
            "initial_lock_owner": {"pid": 10, "lock_path": "/durable/lock"},
            "authority": probe.AUTHORITY,
        }
    )
    payload = {
        "selected_arm": selected,
        "interop_threads": 1,
        "n_pairs": 600,
        "seed": 0,
        "worker_pythonhashseed": "0",
        "raw_sha256": "r" * 64,
        "weights_sha256": "w" * 64,
        "python_executable": probe.active_python_executable(),
        "python_executable_resolved_target": str(Path(sys.executable).resolve()),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity["sha256"],
    }
    fingerprint = {"payload": payload, "sha256": probe.sha256_json(payload)}
    receipt = {
        "selected_arm": selected,
        "axis": probe.AXIS,
        "verdict_scope": probe.receipt_verdict_scope(600),
        "labels": probe.EVIDENCE_LABELS,
        "authority": probe.AUTHORITY,
        "runtime": {
            "python": sys.version,
            "python_executable": probe.active_python_executable(),
            "python_executable_resolved_target": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mps_used": False,
            "cuda_used": False,
            "contest_cpu_timing_measured": False,
        },
        "custody": {
            "argv": run_identity["argv"],
            "git": run_identity["initial_git"],
            "lock_owner": run_identity["initial_lock_owner"],
        },
    }
    configs = {}
    for stage in probe.STAGE_ORDER:
        strategy, threads = probe.stage_strategy(stage, selected)
        for mode in probe.WORKER_MODES:
            configs[f"{stage}:{mode}"] = {
                "stage": stage,
                "mode": mode,
                "strategy": strategy,
                "intraop_threads": threads,
                "interop_threads": 1,
                "n_pairs": 600,
                "seed": 0,
                "pythonhashseed": "0",
                "expected_raw_sha256": "r" * 64,
                "expected_weights_sha256": "w" * 64,
                "fingerprint_sha256": fingerprint["sha256"],
                "method": probe.METHOD,
            }
    return receipt, fingerprint, configs


def test_plan_contract_rejects_selected_axis_runtime_and_config_tampering() -> None:
    receipt, fingerprint, configs = _plan_contract_fixture()
    probe.validate_static_plan_contract(
        receipt=receipt,
        fingerprint=fingerprint,
        configs=configs,
        expected_selected=receipt["selected_arm"],
    )

    selected_tamper = deepcopy(receipt)
    selected_tamper["selected_arm"]["threads"] = 3
    with pytest.raises(RuntimeError, match="selected_fingerprint"):
        probe.validate_static_plan_contract(
            receipt=selected_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=fingerprint["payload"]["selected_arm"],
        )

    axis_tamper = deepcopy(receipt)
    axis_tamper["axis"] = "[contest-CPU MEASURED]"
    with pytest.raises(RuntimeError, match="axis"):
        probe.validate_static_plan_contract(
            receipt=axis_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=receipt["selected_arm"],
        )

    runtime_tamper = deepcopy(receipt)
    runtime_tamper["runtime"]["contest_cpu_timing_measured"] = True
    with pytest.raises(RuntimeError, match="runtime_false_authority"):
        probe.validate_static_plan_contract(
            receipt=runtime_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=receipt["selected_arm"],
        )

    scope_tamper = deepcopy(receipt)
    scope_tamper["verdict_scope"] = "contest score and promotion authority"
    with pytest.raises(RuntimeError, match="verdict_scope"):
        probe.validate_static_plan_contract(
            receipt=scope_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=receipt["selected_arm"],
        )

    labels_tamper = deepcopy(receipt)
    labels_tamper["labels"]["timing_and_sha"] = "CONTEST_SCORE"
    with pytest.raises(RuntimeError, match="evidence_labels"):
        probe.validate_static_plan_contract(
            receipt=labels_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=receipt["selected_arm"],
        )

    python_tamper = deepcopy(receipt)
    python_tamper["runtime"]["python_executable"] = "/tampered/python"
    with pytest.raises(RuntimeError, match="runtime_python"):
        probe.validate_static_plan_contract(
            receipt=python_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=receipt["selected_arm"],
        )

    argv_tamper = deepcopy(receipt)
    argv_tamper["custody"]["argv"] = ["python", "tampered.py"]
    with pytest.raises(RuntimeError, match="custody_argv"):
        probe.validate_static_plan_contract(
            receipt=argv_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=receipt["selected_arm"],
        )

    git_tamper = deepcopy(receipt)
    git_tamper["custody"]["git"]["head"] = "tampered"
    with pytest.raises(RuntimeError, match="custody_git"):
        probe.validate_static_plan_contract(
            receipt=git_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=receipt["selected_arm"],
        )

    lock_tamper = deepcopy(receipt)
    lock_tamper["custody"]["lock_owner"]["pid"] = 999
    with pytest.raises(RuntimeError, match="custody_lock"):
        probe.validate_static_plan_contract(
            receipt=lock_tamper,
            fingerprint=fingerprint,
            configs=configs,
            expected_selected=receipt["selected_arm"],
        )

    config_tamper = deepcopy(configs)
    config_tamper["selected_rep0:measurement"]["intraop_threads"] = 3
    with pytest.raises(RuntimeError, match="config-plan:selected_rep0:measurement"):
        probe.validate_static_plan_contract(
            receipt=receipt,
            fingerprint=fingerprint,
            configs=config_tamper,
            expected_selected=receipt["selected_arm"],
        )


def _synthetic_pass_record(*, timing_ms: float, child_id: str, pid: int) -> dict[str, object]:
    return {
        "completed_pairs": 600,
        "timings_ms": [timing_ms] * 600,
        "pair_sha256": ["a" * 64] * 600,
        "sequence_sha256": "b" * 64,
        "pair78_sha256": "a" * 64,
        "measurement_complete": True,
        "replay_complete": True,
        "terminal_child_id": child_id,
        "terminal_pid": pid,
        "strategy": "eager_nchw_autograd",
        "intraop_threads": 6 if timing_ms > 500 else 1,
        "interop_threads": 1,
        "binding_before": {
            "intraop_threads": 6 if timing_ms > 500 else 1,
            "interop_threads": 1,
        },
        "binding_after": {
            "intraop_threads": 6 if timing_ms > 500 else 1,
            "interop_threads": 1,
        },
        "process_segments": [{"child_id": child_id, "pid": pid}],
    }


def test_derive_measurement_cannot_admit_duplicate_child_identity(monkeypatch, tmp_path: Path) -> None:
    measurements = {
        stage: _synthetic_pass_record(
            timing_ms=900.0 if stage.startswith("baseline") else 300.0,
            child_id="same-child",
            pid=123,
        )
        for stage in probe.STAGE_ORDER
    }
    replays = {
        stage: _synthetic_pass_record(
            timing_ms=900.0 if stage.startswith("baseline") else 300.0,
            child_id="same-child",
            pid=123,
        )
        for stage in probe.STAGE_ORDER
    }
    monkeypatch.setattr(probe, "sha256_file", lambda _path: "c" * 64)
    args = argparse.Namespace(n_pairs=600, checkpoint_dir=tmp_path)
    measurement, admitted = probe._derive_measurement(
        args=args,
        selected={"strategy": "eager_nchw_autograd", "threads": 1, "baseline_threads": 6},
        measurements=measurements,
        replays=replays,
    )
    assert admitted is False
    assert measurement["independent_full_replays"]["independent_processes"] is False
    assert measurement["independent_full_replays"]["unique_child_id_count"] == 1
    assert measurement["independent_full_replays"]["unique_pid_count"] == 1
