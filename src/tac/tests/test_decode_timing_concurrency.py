"""tac.decode_timing_concurrency: the frozen admission rule, receipt assembly, and the positive
control that an assembled quiesced receipt PASSES tac.decode_wall_clock (the dwc1 producers could
never reach that path: they wrote a total process count or null). The rule is ddm_pr10's
replacement: frozen + hashed before launch, quarter-core threshold over the whole window,
ancestors included, aggregate cap from P-cores, stage rates diagnostic only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tac.candidate_seal import measure_archive_identity, measure_runtime_digest
from tac.decode_timing_concurrency import (
    ADMISSION_RULE_SCHEMA,
    CONCURRENCY_SCHEMA,
    RAW_TIMING_SCHEMA,
    ConcurrencyError,
    ConcurrencyMonitor,
    ConcurrencyRule,
    PausedProcesses,
    ProcessRow,
    ancestors_of,
    assemble_local_receipt,
    classify,
    read_process_table,
    run_producer,
    sample_competitors,
    stage_diagnostics,
    stage_rate_deviations,
    summarize,
    tree_of,
    wait_until_quiet,
    write_calibration,
)
from tac.decode_wall_clock import (
    LOCAL_SCHEMA,
    SealContractError,
    build_decode_wall_clock,
    measure_receiver_digest,
    measure_t4_runtime_digest,
    validate_decode_wall_clock,
)

RULE = ConcurrencyRule(threshold_pcpu=25.0, visible_pcpu=5.0, ancestor_cap_pcpu=25.0,
                       aggregate_cap_pcpu=200.0, interval_seconds=0.2, settle_quiet_samples=3)
FROZEN_AT = "2026-09-10T12:00:00Z"
STARTED_AT = "2026-09-10T12:01:00Z"


def rows(*items: tuple[int, int, float, str]) -> list[ProcessRow]:
    return [ProcessRow(*item) for item in items]


TABLE = rows((1, 0, 0.3, "launchd"), (50, 1, 12.0, "claude"), (100, 50, 0.0, "monitor"),
             (200, 100, 0.0, "producer"), (201, 200, 380.0, "worker"), (300, 1, 30.0, "dashboard"),
             (301, 1, 9.5, "WindowServer"), (302, 1, 0.1, "idle"))
QUIET = rows((1, 0, 0.0, "launchd"), (100, 1, 0.0, "monitor"), (200, 100, 380.0, "worker"))


def test_tree_and_ancestors():
    assert tree_of(200, TABLE) == {200, 201}
    assert tree_of(100, TABLE) == {100, 200, 201}
    assert ancestors_of(100, TABLE) == [50, 1]


def test_rule_freezes_to_a_stable_hash():
    frozen = RULE.frozen()
    assert frozen["schema"] == ADMISSION_RULE_SCHEMA and frozen["threshold_pcpu"] == 25.0
    assert RULE.sha256() == ConcurrencyRule.from_frozen(frozen).sha256()
    assert ConcurrencyRule(threshold_pcpu=26.0, aggregate_cap_pcpu=200.0, interval_seconds=0.2).sha256() != RULE.sha256()
    with pytest.raises(ConcurrencyError, match="unknown schema"):
        ConcurrencyRule.from_frozen({"schema": "other"})
    with pytest.raises(ConcurrencyError, match="field missing"):
        ConcurrencyRule.from_frozen({"schema": ADMISSION_RULE_SCHEMA, "threshold_pcpu": 25.0})


def test_classify_applies_the_stated_rule():
    sample = classify(TABLE, rule=RULE, monitor_pid=100, producer_pid=200, label="t")
    assert [c["pid"] for c in sample["competing"]] == [300]
    assert [v["pid"] for v in sample["visible"]] == [300, 301]
    assert [a["pid"] for a in sample["excluded_ancestors_active"]] == [50]
    assert sample["other_pcpu_sum"] == pytest.approx(30.0 + 9.5 + 0.1)
    assert sample_competitors(sample, RULE) == sample["competing"]


def test_ancestor_counts_at_or_above_its_cap():
    table = rows(*[(r.pid, r.ppid, 25.0 if r.pid == 50 else r.pcpu, r.comm) for r in TABLE])
    sample = classify(table, rule=RULE, monitor_pid=100, producer_pid=200, label="t")
    assert sorted(((c["pid"], c.get("reason")) for c in sample["competing"]), key=str) == [
        (300, None), (50, "ancestor at or above cap")]
    assert len(sample_competitors(sample, RULE)) == 2


def test_aggregate_guard_counts_many_small_processes():
    table = TABLE + rows(*[(400 + i, 1, 20.0, f"small{i}") for i in range(10)])
    sample = classify(table, rule=RULE, monitor_pid=100, producer_pid=200, label="t")
    assert [c.get("reason") for c in sample["competing"]] == [None, "aggregate at or above cap"]


def _samples(labels_and_tables, *, start_epoch: float = 1_800_000_000.0, step: float = 20.0) -> list[dict]:
    samples = []
    for index, (label, table) in enumerate(labels_and_tables):
        sample = classify(table, rule=RULE, monitor_pid=100, producer_pid=200, label=label)
        sample["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_epoch))
        sample["monotonic"] = 1000.0 + index * step
        samples.append(sample)
    return samples


def _summary(labels_and_tables, **kw) -> dict:
    return summarize(_samples(labels_and_tables, **kw), rule=RULE, paused=[], monitor_command=["m"],
                     frozen_at_utc=FROZEN_AT, producer_started_at_utc=STARTED_AT)


def test_summarize_counts_the_whole_window_and_ignores_settle():
    summary = _summary([("settle_0001", TABLE), ("before", QUIET), ("during_0001", TABLE), ("after", QUIET)])
    assert summary["schema"] == CONCURRENCY_SCHEMA
    assert summary["admission_rule"] == RULE.frozen() and summary["admission_rule_sha256"] == RULE.sha256()
    assert summary["competing_process_count"] == 1 and summary["quiesced"] is False
    assert summary["samples_with_competition"] == ["during_0001"]
    assert summary["window_sample_count"] == 3 and summary["settle_sample_count"] == 1
    assert _summary([("settle_0001", TABLE), ("before", QUIET)])["quiesced"] is True
    with pytest.raises(ConcurrencyError):
        summarize(_samples([("settle_0001", TABLE)]), rule=RULE, paused=[], monitor_command=["m"],
                  frozen_at_utc=FROZEN_AT, producer_started_at_utc=STARTED_AT)


def test_wait_until_quiet_needs_consecutive_quiet_samples(monkeypatch):
    tables = iter([QUIET, QUIET, TABLE, QUIET, QUIET, QUIET, QUIET])
    monkeypatch.setattr("tac.decode_timing_concurrency.read_process_table", lambda: next(tables))
    monitor = ConcurrencyMonitor(RULE, monitor_pid=100)
    quiet = wait_until_quiet(monitor, settle_seconds=30.0)
    assert [s["label"] for s in quiet] == ["settle_0004", "settle_0005", "settle_0006"]
    assert len(monitor.samples) == 6
    busy = iter([TABLE] * 50)
    monkeypatch.setattr("tac.decode_timing_concurrency.read_process_table", lambda: next(busy))
    with pytest.raises(ConcurrencyError, match="did not quiesce"):
        wait_until_quiet(ConcurrencyMonitor(RULE, monitor_pid=100), settle_seconds=0.5)


def test_read_process_table_sees_this_process():
    table = read_process_table()
    assert any(row.pid == os.getpid() for row in table)


def test_paused_process_custody_records_identity_and_transitions():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with PausedProcesses([child.pid]) as paused:
            record = paused.records[0]
            assert record["pid"] == child.pid and record["stop_confirmed"] is True
            assert record["comm"] and record["start_time"] and record["stop_requested_at_utc"]
        assert record["resume_confirmed"] is True and record["resume_requested_at_utc"]
    finally:
        child.kill()
        child.wait()


def _t4_receipt(runtime: Path, path: Path, seconds: float) -> Path:
    archive_sha = measure_archive_identity(runtime / "archive.zip").sha256
    path.write_text(json.dumps({
        "passed": True, "returncode": 0, "expected_archive_sha256": archive_sha,
        "expected_runtime_tree_sha256": measure_t4_runtime_digest(runtime),
        "artifacts": {"contest_auth_eval.json": json.dumps({"inflate_elapsed_seconds": seconds}),
                      "modal_cuda_preflight.json": json.dumps({"torch_cuda_device_name": "Tesla T4"})}}))
    return path


def _raw_timing(runtime: Path, *, wall: float, pairs: int) -> dict:
    return {"schema": RAW_TIMING_SCHEMA, "axis": "[macOS-CPU advisory]", "measurement_kind": "cuda_path_equivalent_proxy",
            "binding": {"copied_runtime": str(runtime), "archive": {"sha256": measure_archive_identity(runtime / "archive.zip").sha256}},
            "report": {"pair_count": pairs}, "returncode": 0, "timed_out": False, "raw": {"sha256": "x"},
            "cold_start": True, "resumed": False, "cpu_threads": 4, "host": "synthetic-test-host",
            "platform": "synthetic", "command": ["producer", "--worker"], "wall_seconds": wall,
            "timed_scope": "synthetic", "public_entrypoint_executed": False, "public_shell_startup_included": False,
            "concurrency_before": {"concurrent_process_count": None}, "concurrency_after": {"concurrent_process_count": None}}


def _concurrency(path: Path, busy_samples: int) -> Path:
    tables = [("before", QUIET)] + [(f"during_{i + 1:04d}", TABLE) for i in range(busy_samples)] + [("after", QUIET)]
    path.write_text(json.dumps(_summary(tables)))
    return path


@pytest.fixture
def staged(tmp_path):
    from tac.tests.test_candidate_seal import _stage_candidate
    runtime, archive = _stage_candidate(tmp_path)
    root = tmp_path / "timing"
    root.mkdir()
    producer = root / "TIMING.json"
    producer.write_text(json.dumps(_raw_timing(runtime, wall=40.0, pairs=20)))
    return runtime, archive, root, producer


def test_assembled_quiesced_receipt_passes_the_validator(staged):
    runtime, archive, root, producer = staged
    concurrency = _concurrency(root / "CONCURRENCY.json", busy_samples=0)
    local = assemble_local_receipt(json.loads(producer.read_text()), json.loads(concurrency.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    assert local["schema"] == LOCAL_SCHEMA and local["margin_time_basis"] == "quiesced"
    assert local["frames"] == list(range(20)) and local["wall_seconds"] == 40.0
    block = local["concurrency"]
    assert block["competing_process_count"] == 0 and block["admission_rule_sha256"] == RULE.sha256()
    assert block["admission_rule_frozen_at_utc"] == FROZEN_AT and block["producer_started_at_utc"] == STARTED_AT
    assert local["receiver_sha256"] == measure_receiver_digest(runtime)
    assert local["runtime_sha256"] == measure_runtime_digest(runtime).sha256
    local_path = root / "local.json"
    local_path.write_text(json.dumps(local))
    t4 = _t4_receipt(runtime, root / "t4.json", 900.0)
    calibration = write_calibration(name="synthetic", local_receipt_path=local_path, t4_receipt_path=t4)
    assert calibration["cpu_to_t4_ratio"] == pytest.approx(900.0 / 1200.0)
    assert calibration["admission_rule_sha256"] == RULE.sha256()
    calibration_path = root / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    leg = build_decode_wall_clock(local_receipt_path=local_path, calibration_receipt_path=calibration_path,
                                  runtime_dir=runtime, archive_path=archive, candidate_t4_receipt_path=t4)
    problems, observed = validate_decode_wall_clock(leg, runtime_dir=runtime, archive_path=archive)
    assert problems == [] and observed["measured_t4_decode_seconds"] == 900.0


def test_competition_anywhere_in_the_window_is_refused(staged):
    runtime, archive, root, producer = staged
    concurrency = _concurrency(root / "CONCURRENCY.json", busy_samples=1)
    local = assemble_local_receipt(json.loads(producer.read_text()), json.loads(concurrency.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    assert local["margin_time_basis"] == "measured_concurrency_nonzero"
    assert local["concurrency"]["competing_process_count"] == 1
    assert local["concurrency"]["samples_with_competition"] == ["during_0001"]
    local_path = root / "local.json"
    local_path.write_text(json.dumps(local))
    t4 = _t4_receipt(runtime, root / "t4.json", 900.0)
    with pytest.raises(ConcurrencyError, match="quiesced"):
        write_calibration(name="synthetic", local_receipt_path=local_path, t4_receipt_path=t4)
    # the validator refuses too, independently of this module
    quiet = _concurrency(root / "C2.json", busy_samples=0)
    clean = assemble_local_receipt(json.loads(producer.read_text()), json.loads(quiet.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=quiet)
    clean_path = root / "clean.json"
    clean_path.write_text(json.dumps(clean))
    calibration_path = root / "calibration.json"
    calibration_path.write_text(json.dumps(write_calibration(name="synthetic", local_receipt_path=clean_path, t4_receipt_path=t4)))
    with pytest.raises(SealContractError, match="quiesced"):
        build_decode_wall_clock(local_receipt_path=local_path, calibration_receipt_path=calibration_path,
                                runtime_dir=runtime, archive_path=archive)


def test_assembler_cannot_loosen_or_forge_the_rule(staged):
    runtime, archive, root, producer = staged
    concurrency = _concurrency(root / "CONCURRENCY.json", busy_samples=1)
    doc = json.loads(concurrency.read_text())
    loosened = {**doc, "admission_rule": {**doc["admission_rule"], "threshold_pcpu": 100.0}}
    with pytest.raises(ConcurrencyError, match="hash mismatch"):
        assemble_local_receipt(json.loads(producer.read_text()), loosened,
                               producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    recounted = {**doc, "competing_process_count": 0, "quiesced": True}
    with pytest.raises(ConcurrencyError, match="differs from the sampler"):
        assemble_local_receipt(json.loads(producer.read_text()), recounted,
                               producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    late = {**doc, "admission_rule_frozen_at_utc": "2026-09-10T12:02:00Z"}
    with pytest.raises(ConcurrencyError, match="frozen after"):
        assemble_local_receipt(json.loads(producer.read_text()), late,
                               producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    unfrozen = {**doc, "producer_started_at_utc": None}
    with pytest.raises(ConcurrencyError, match="absent"):
        assemble_local_receipt(json.loads(producer.read_text()), unfrozen,
                               producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    with pytest.raises(ConcurrencyError, match=r"measured_concurrency\.v2"):
        assemble_local_receipt(json.loads(producer.read_text()), {**doc, "schema": "decode_wall_clock.measured_concurrency.v1"},
                               producer_receipt_path=producer, concurrency_receipt_path=concurrency)


def test_assemble_local_schema_keeps_the_producer_document(staged):
    runtime, archive, root, producer = staged
    producer_doc = {"schema": LOCAL_SCHEMA, "wall_seconds": 12.5, "frames": [0, 1], "cold_start": True,
                    "concurrency": {"competing_process_count": None}, "margin_time_basis": "unknown_host_concurrency",
                    "archive_sha256": "a", "receiver_sha256": "b"}
    producer.write_text(json.dumps(producer_doc))
    concurrency = _concurrency(root / "CONCURRENCY.json", busy_samples=0)
    local = assemble_local_receipt(producer_doc, json.loads(concurrency.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    assert local["wall_seconds"] == 12.5 and local["margin_time_basis"] == "quiesced"
    assert local["producer_margin_time_basis"] == "unknown_host_concurrency"
    assert local["producer_concurrency"] == {"competing_process_count": None}
    assert producer_doc["margin_time_basis"] == "unknown_host_concurrency", "input document must not be mutated"


def _stage_checkpoints(directory: Path, seconds: list[float], *, start_epoch: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    epoch = start_epoch
    for index, duration in enumerate([0.0, *seconds]):
        epoch += duration
        path = directory / f"stage_{25 * (index + 1):04d}.npz"
        path.write_bytes(b"x")
        os.utime(path, (epoch, epoch))
    return directory


def test_stage_rates_are_diagnostic_only(staged, tmp_path):
    runtime, archive, root, producer = staged
    base = 1_800_000_000.0
    seconds = [21.0, 21.2, 21.1, 26.0, 21.3, 21.2]
    checkpoints = _stage_checkpoints(tmp_path / "ck", seconds, start_epoch=base)
    rates = stage_rate_deviations(checkpoints, tolerance=0.05)
    assert rates["exceeding_stages"] == [125] and rates["median_seconds"] == pytest.approx(21.2)
    assert rates["excess_seconds"] == pytest.approx(4.9, abs=0.01)
    with pytest.raises(ConcurrencyError, match="too few"):
        stage_rate_deviations(_stage_checkpoints(tmp_path / "few", [21.0, 21.0], start_epoch=base))
    # a slow stage with quiet samples: the receipt stays quiesced (diagnostic only) but lists the stage
    tables = [("before", QUIET)] + [(f"during_{i + 1:04d}", QUIET) for i in range(6)] + [("after", QUIET)]
    concurrency = root / "CONCURRENCY.json"
    concurrency.write_text(json.dumps(_summary(tables, start_epoch=base)))
    local = assemble_local_receipt(json.loads(producer.read_text()), json.loads(concurrency.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency,
                                   stage_checkpoint_dir=checkpoints)
    diag = local["concurrency"]["stage_diagnostics"]
    assert local["margin_time_basis"] == "quiesced" and diag["role"].startswith("diagnostic")
    assert diag["exceeding_stages"] == [125] and diag["slowed_stages"][0]["processes_beside"] == []
    # and a competitor beside a slow stage is named, while the count comes from the whole-window rule
    busy = [("before", QUIET), ("during_0001", QUIET), ("during_0002", QUIET), ("during_0003", TABLE),
            ("during_0004", TABLE), ("during_0005", QUIET), ("during_0006", QUIET), ("after", QUIET)]
    summary = _summary(busy, start_epoch=base)
    verdict = stage_diagnostics(summary, rates, wall_seconds=sum(seconds) + 5.0, rule=RULE)
    assert [p["pid"] for p in verdict["slowed_stages"][0]["processes_beside"]] == [300, 301]
    assert summary["competing_process_count"] == 1


def test_sample_epochs_are_utc_and_dst_proof():
    from tac.decode_timing_concurrency import _sample_epochs
    epoch = 1_789_031_551.0  # 2026-09-10T09:12:31Z, a DST date in America/Chicago
    samples = [{"time_utc": "2026-09-10T09:12:31Z", "monotonic": 10.0}, {"time_utc": "x", "monotonic": 30.5}]
    assert _sample_epochs(samples) == [epoch, epoch + 20.5]


def test_run_producer_freezes_the_rule_before_the_producer_starts(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(0.8); print('done')"]
    rule = ConcurrencyRule(interval_seconds=0.2, threshold_pcpu=10_000.0, aggregate_cap_pcpu=1e9,
                           ancestor_cap_pcpu=1e9, settle_quiet_samples=2)
    code, summary = run_producer(command, cwd=tmp_path, env=dict(os.environ), stdout_path=tmp_path / "out.log",
                                 rule=rule, settle_seconds=5.0, paused_pids=[], timeout_seconds=30.0, monitor_command=["m"])
    assert code == 0 and (tmp_path / "out.log").read_text().strip() == "done"
    assert summary["schema"] == CONCURRENCY_SCHEMA and summary["admission_rule_sha256"] == rule.sha256()
    assert summary["admission_rule_frozen_at_utc"] <= summary["producer_started_at_utc"]
    labels = [sample["label"] for sample in summary["samples"]]
    assert labels[:2] == ["settle_0001", "settle_0002"] and "before" in labels and labels[-1] == "after"
    assert summary["competing_process_count"] == 0 and summary["quiesced"] is True and summary["paused_processes"] == []


def test_cli_assemble_calibrate_leg_round_trip(staged, capsys):
    from tools.quiesced_decode_timing import main
    runtime, archive, root, producer = staged
    concurrency = _concurrency(root / "CONCURRENCY.json", busy_samples=0)
    assert main(["assemble", "--producer-receipt", str(producer), "--concurrency", str(concurrency),
                 "--out", str(root / "local.json")]) == 0
    t4 = _t4_receipt(runtime, root / "t4.json", 900.0)
    assert main(["calibrate", "--name", "synthetic", "--local", str(root / "local.json"), "--t4-receipt", str(t4),
                 "--out", str(root / "calibration.json")]) == 0
    seal = root / "SEAL.json"
    seal.write_text("{}")
    assert main(["leg", "--local", str(root / "local.json"), "--calibration", str(root / "calibration.json"),
                 "--runtime-dir", str(runtime), "--archive", str(archive), "--candidate-t4-receipt", str(t4),
                 "--out", str(root / "leg.json"), "--sidecar-for", str(seal)]) == 0
    assert json.loads((root / "SEAL.json.decode_wall_clock.json").read_text()) == json.loads((root / "leg.json").read_text())
    out = capsys.readouterr().out
    assert "margin_time_basis=quiesced" in out and "problems=none" in out


@pytest.fixture
def v3_rule(monkeypatch):
    """Simulate the retained trace host; this is never a prospective freeze receipt."""
    monkeypatch.setattr("tac.decode_timing_concurrency.platform.system", lambda: "Darwin")
    monkeypatch.setattr("tac.decode_timing_concurrency.platform.machine", lambda: "arm64")
    monkeypatch.setattr("tac.decode_timing_concurrency.os.cpu_count", lambda: 18)
    monkeypatch.setattr("tac.decode_timing_concurrency.p_core_count", lambda: 6)
    return ConcurrencyRule()


def _burst_samples(rule, values, *, step=20.0):
    samples = []
    for i, (dasd, syspolicyd) in enumerate(values):
        table = QUIET + rows((501, 1, dasd, "/usr/libexec/dasd"), (502, 1, syspolicyd, "/usr/libexec/syspolicyd"))
        sample = classify(table, rule=rule, monitor_pid=100, producer_pid=200, label=f"during_{i:04d}")
        sample["monotonic"] = 1000.0 + i * step
        samples.append(sample)
    return samples


def _burst_summary(rule, samples):
    return summarize(samples, rule=rule, paused=[], monitor_command=["synthetic-test"],
                     frozen_at_utc=FROZEN_AT, producer_started_at_utc=STARTED_AT)


def test_v3_frozen_object_is_exact_pr11_json(v3_rule):
    import re

    memo = Path(__file__).resolve().parents[3] / ".omx/research/ddm_pr11_adjudicate_timing_admission_on_daemon_bursts_20260910.md"
    expected = json.loads(re.search(r"```json\n(.*?)\n```", memo.read_text(), re.S)[1])
    assert v3_rule.frozen() == expected
    assert ConcurrencyRule.from_frozen(expected).sha256() == v3_rule.sha256()
    changed = {**expected, "host_baseline_burst": {**expected["host_baseline_burst"], "max_bursts_in_window": 2}}
    with pytest.raises(ConcurrencyError, match="exact v3 class"):
        ConcurrencyRule.from_frozen(changed)
    with pytest.raises(ConcurrencyError, match="all four"):
        ConcurrencyRule.from_frozen({**expected, "host_constraints": {}})


# Hash-bound retained evidence: read only. No assembly, mutation, or authority upgrade.
RETAINED_TRACES = [
    ("move40_quiesced/CONCURRENCY.json", "f19efa1a1233e477643d92dcb91d37ec09bbeca5878d936b2fdae1ce126a86bb", 3, "during_0014", 14, 280.416),
    ("move40_quiesced2/CONCURRENCY.json", "040fe2c19399dfe7bd559a7683828273b7d362cba3cbd8e278e3a6af023fd611", 5, "during_0003", 14, 280.428),
    ("move40_quiesced3/CONCURRENCY_attempt3_refused.json", "63d55d1d3b05783c5cdbc0b0ece7c0562b0fd04541d40704036dc3a594b74202", 0, None, 10, 200.292),
    ("move40_quiesced4/CONCURRENCY.json", "f72baecc4d72fcf76aaf3807121f181824d5c4bc74a7f1a36c20510100b9728c", 0, None, 11, 220.320),
]


@pytest.mark.parametrize("suffix,digest,count,label,active,span", RETAINED_TRACES)
def test_retained_traces_counterfactual_only(v3_rule, suffix, digest, count, label, active, span):
    import hashlib

    path = Path("/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/receipts") / suffix
    if not path.is_file():
        pytest.skip(f"retained read-only trace not mounted: {path}")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    samples = json.loads(raw)["samples"]
    untouched = json.dumps(samples, sort_keys=True)
    result = _burst_summary(v3_rule, samples)
    assert result["competing_process_count"] == count
    assert result["host_baseline_allowance"]["valid"] is True
    burst, = result["host_baseline_bursts"]
    assert burst["active_samples"] == active
    assert burst["inclusive_span_seconds"] == pytest.approx(span, abs=0.0005)
    assert result["margin_time_basis"] == ("bounded_host_baseline" if count == 0 else "measured_concurrency_nonzero")
    if label:
        assert len(next(s for s in result["samples"] if s["label"] == label)["unwaived_competing"]) == count
    assert json.dumps(samples, sort_keys=True) == untouched
    assert path.read_bytes() == raw


@pytest.mark.parametrize("key,value", [("platform_system", "Linux"), ("machine", "x86_64"),
                                        ("logical_cpu_count", 16), ("performance_core_count", 8)])
def test_host_constraint_mismatch_disables_allowance(v3_rule, key, value):
    from dataclasses import replace

    rule = replace(v3_rule, host_constraints={**v3_rule.host_constraints, key: value})
    result = _burst_summary(rule, _burst_samples(rule, [(50.0, 0.0)]))
    assert result["host_baseline_allowance"]["checks"]["host_" + key] is False
    assert result["competing_process_count"] == 1
    assert result["samples"][0]["unwaived_other_pcpu_sum"] == 50.0


@pytest.mark.parametrize("comm,ppid", [("dasd", 1), ("/usr/libexec/dasd.extra", 1),
                                       ("/usr/libexec/dasd", 2), ("/usr/libexec/dasd", 1.0)])
def test_exact_member_identity_only(v3_rule, comm, ppid):
    sample = classify(QUIET + rows((501, ppid, 50.0, comm)), rule=v3_rule,
                      monitor_pid=100, producer_pid=200, label="before")
    assert sample["visible"][0]["host_baseline_member_match"] is False
    assert _burst_summary(v3_rule, [sample])["competing_process_count"] == 1


@pytest.mark.parametrize("values,step,bound", [
    ([(50.0, 0.0), (24.9, 0.0), (50.0, 0.0)], 20.0, "burst_count"),
    ([(50.0, 0.0)] * 16, 1.0, "active_samples_per_burst"),
    ([(50.0, 0.0)] * 2, 280.001, "inclusive_span"),
    ([(100.001, 0.0)], 20.0, "member_caps"),
    ([(0.0, 75.001)], 20.0, "member_caps"),
])
def test_global_bound_violation_waives_nothing(v3_rule, values, step, bound):
    result = _burst_summary(v3_rule, _burst_samples(v3_rule, values, step=step))
    assert result["host_baseline_allowance"]["checks"][bound] is False
    assert result["host_baseline_allowance"]["valid"] is False
    assert result["competing_process_count"] >= 1
    assert result["margin_time_basis"] == "measured_concurrency_nonzero"
    assert all(s["unwaived_other_pcpu_sum"] == s["other_pcpu_sum"] for s in result["samples"])


def test_inclusive_boundaries_and_combined_cap(v3_rule):
    # All member, combined, sample-count and span caps are met exactly.
    result = _burst_summary(v3_rule, _burst_samples(v3_rule, [(100.0, 75.0)] * 15))
    assert result["host_baseline_allowance"]["valid"] is True
    assert result["competing_process_count"] == 0
    assert result["host_baseline_bursts"][0]["inclusive_span_seconds"] == 300.0
    # Multiple exact-member rows make combined cap independently reachable.
    sample = classify(QUIET + rows((501, 1, 100.0, "/usr/libexec/dasd"),
                                   (502, 1, 75.0, "/usr/libexec/syspolicyd"),
                                   (503, 1, 0.1, "/usr/libexec/dasd")),
                      rule=v3_rule, monitor_pid=100, producer_pid=200, label="before")
    invalid = _burst_summary(v3_rule, [sample])
    assert invalid["host_baseline_allowance"]["checks"]["member_caps"] is True
    assert invalid["host_baseline_allowance"]["checks"]["combined_cap"] is False
    assert invalid["competing_process_count"] == 2


def test_activation_aggregate_and_subvisible_match(v3_rule):
    for value, basis in [(24.9, "quiesced"), (25.0, "bounded_host_baseline")]:
        result = _burst_summary(v3_rule, _burst_samples(v3_rule, [(value, 0.0)]))
        assert result["margin_time_basis"] == basis and result["competing_process_count"] == 0
    for ordinary, count in [(199.9, 0), (200.0, 1)]:
        table = QUIET + rows((501, 1, 25.0, "/usr/libexec/dasd"), (502, 1, 4.0, "/usr/libexec/syspolicyd"))
        table += rows(*[(600+i, 1, ordinary/20, "small") for i in range(20)])
        sample = classify(table, rule=v3_rule, monitor_pid=100, producer_pid=200, label="before")
        result = _burst_summary(v3_rule, [sample])
        assert len(result["samples"][0]["host_baseline_matches"]) == 2
        assert result["samples"][0]["unwaived_other_pcpu_sum"] == pytest.approx(ordinary)
        assert result["competing_process_count"] == count


def test_settle_uses_prefix_then_final_failure_refuses(v3_rule, monkeypatch):
    from dataclasses import replace

    rule = replace(v3_rule, interval_seconds=0.01)
    table = QUIET + rows((501, 1, 50.0, "/usr/libexec/dasd"))
    monkeypatch.setattr("tac.decode_timing_concurrency.read_process_table", lambda: table)
    monitor = ConcurrencyMonitor(rule, monitor_pid=100)
    assert len(wait_until_quiet(monitor, settle_seconds=0.1)) == 3
    assert all(s["host_baseline_provisional_allowance"]["valid"] for s in monitor.samples)
    assert all(s["unwaived_other_pcpu_sum"] == 0.0 for s in monitor.samples)
    result = _burst_summary(rule, monitor.samples + _burst_samples(rule, [(50.0, 0.0)] * 16, step=0.01))
    assert result["margin_time_basis"] == "measured_concurrency_nonzero"
    table = QUIET + rows((501, 1, 100.1, "/usr/libexec/dasd"))
    with pytest.raises(ConcurrencyError, match="did not quiesce"):
        wait_until_quiet(ConcurrencyMonitor(rule, monitor_pid=100), settle_seconds=0.02)


def test_assembled_bounded_receipt_passes_validator(staged, v3_rule):
    runtime, archive, root, producer = staged
    summary = _burst_summary(v3_rule, _burst_samples(v3_rule, [(100.0, 75.0)] * 15))
    concurrency = root / "CONCURRENCY.json"
    concurrency.write_text(json.dumps(summary))
    local = assemble_local_receipt(json.loads(producer.read_text()), summary,
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    assert local["margin_time_basis"] == "bounded_host_baseline"
    assert local["concurrency"]["host_baseline_allowance"] == summary["host_baseline_allowance"]
    assert local["concurrency"]["host_baseline_bursts"] == summary["host_baseline_bursts"]
    local_path = root / "local.json"
    local_path.write_text(json.dumps(local))
    t4 = _t4_receipt(runtime, root / "t4.json", 900.0)
    calibration = root / "calibration.json"
    calibration.write_text(json.dumps(write_calibration(name="synthetic", local_receipt_path=local_path, t4_receipt_path=t4)))
    leg = build_decode_wall_clock(local_receipt_path=local_path, calibration_receipt_path=calibration,
                                  runtime_dir=runtime, archive_path=archive, candidate_t4_receipt_path=t4)
    assert validate_decode_wall_clock(leg, runtime_dir=runtime, archive_path=archive)[0] == []
    for key in ["host_baseline_allowance", "host_baseline_bursts"]:
        forged = {**summary, key: {}}
        with pytest.raises(ConcurrencyError, match="adjudication differs"):
            assemble_local_receipt(json.loads(producer.read_text()), forged,
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    summary["samples"][0]["unwaived_other_pcpu_sum"] = 7.0
    with pytest.raises(ConcurrencyError, match="adjudication differs"):
        assemble_local_receipt(json.loads(producer.read_text()), summary,
                               producer_receipt_path=producer, concurrency_receipt_path=concurrency)


def test_v2_frozen_rule_keeps_original_hash_and_daemon_refusal(v3_rule):
    from dataclasses import replace

    from tac.decode_timing_concurrency import ADMISSION_RULE_V2

    old = replace(v3_rule, schema=ADMISSION_RULE_V2)
    assert old.sha256() == "b94474c6167ef51bd1e721b1928ab2a627b1a1821a448d899dd2c362a4eacdf1"
    assert ConcurrencyRule.from_frozen(old.frozen()).frozen() == old.frozen()
    result = _burst_summary(old, _burst_samples(old, [(50.0, 0.0)]))
    assert result["competing_process_count"] == 1
    assert result["margin_time_basis"] == "measured_concurrency_nonzero"


def test_allowed_burst_does_not_waive_ordinary_or_ancestor(v3_rule):
    table = QUIET + rows((501, 1, 50.0, "/usr/libexec/dasd"), (600, 1, 25.0, "ordinary"),
                         (50, 1, 25.0, "control-plane"))
    table = [ProcessRow(r.pid, 50 if r.pid == 100 else r.ppid, r.pcpu, r.comm) for r in table]
    sample = classify(table, rule=v3_rule, monitor_pid=100, producer_pid=200, label="before")
    result = _burst_summary(v3_rule, [sample])
    assert result["host_baseline_allowance"]["valid"] is True
    assert result["competing_process_count"] == 2
    assert {row["pid"] for row in result["samples"][0]["unwaived_competing"]} == {50, 600}


def test_captured_host_mismatch_and_row_tamper_refuse(v3_rule):
    samples = _burst_samples(v3_rule, [(50.0, 0.0)])
    samples[0]["host_constraints_observed"]["performance_core_count"] = 8
    result = _burst_summary(v3_rule, samples)
    assert result["competing_process_count"] == 1
    samples = _burst_samples(v3_rule, [(50.0, 0.0)])
    samples[0]["host_baseline_matches"][0]["pcpu"] = 20.0
    with pytest.raises(ConcurrencyError, match="differs from visible"):
        _burst_summary(v3_rule, samples)


def test_new_authority_requires_captured_host_and_matches(staged, v3_rule):
    _, _, root, producer = staged
    for key in ["host_constraints_observed", "host_baseline_matches"]:
        summary = _burst_summary(v3_rule, _burst_samples(v3_rule, [(50.0, 0.0)]))
        summary["samples"][0].pop(key)
        with pytest.raises(ConcurrencyError, match="lacks captured"):
            assemble_local_receipt(json.loads(producer.read_text()), summary,
                                   producer_receipt_path=producer, concurrency_receipt_path=root / "CONCURRENCY.json")
