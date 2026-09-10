"""tac.decode_timing_concurrency: the measured-concurrency rule, receipt assembly, and the
positive control that an assembled quiesced receipt PASSES tac.decode_wall_clock (the dwc1
producers could never reach that path: they wrote a total process count or null)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from tac.candidate_seal import measure_archive_identity, measure_runtime_digest
from tac.decode_timing_concurrency import (
    CONCURRENCY_SCHEMA,
    RAW_TIMING_SCHEMA,
    ConcurrencyError,
    ConcurrencyRule,
    ProcessRow,
    ancestors_of,
    assemble_local_receipt,
    attribute_competition,
    classify,
    read_process_table,
    run_producer,
    stage_rate_deviations,
    summarize,
    tree_of,
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

RULE = ConcurrencyRule(threshold_pcpu=25.0, visible_pcpu=5.0, ancestor_cap_pcpu=50.0,
                       aggregate_cap_pcpu=700.0, interval_seconds=0.2)


def rows(*items: tuple[int, int, float, str]) -> list[ProcessRow]:
    return [ProcessRow(*item) for item in items]


TABLE = rows((1, 0, 0.3, "launchd"), (50, 1, 12.0, "claude"), (100, 50, 0.0, "monitor"),
             (200, 100, 0.0, "producer"), (201, 200, 380.0, "worker"), (300, 1, 30.0, "dashboard"),
             (301, 1, 9.5, "WindowServer"), (302, 1, 0.1, "idle"))


def test_tree_and_ancestors():
    assert tree_of(200, TABLE) == {200, 201}
    assert tree_of(100, TABLE) == {100, 200, 201}
    assert ancestors_of(100, TABLE) == [50, 1]


def test_classify_applies_the_stated_rule():
    sample = classify(TABLE, rule=RULE, monitor_pid=100, producer_pid=200, label="t")
    assert [c["pid"] for c in sample["competing"]] == [300]
    assert [v["pid"] for v in sample["visible"]] == [300, 301]
    assert [a["pid"] for a in sample["excluded_ancestors_active"]] == [50]
    assert sample["other_pcpu_sum"] == pytest.approx(30.0 + 9.5 + 0.1)
    assert sample["producer_tree_size"] == 2 and sample["monitor_tree_size"] == 3


def test_ancestor_counts_above_its_cap():
    table = rows(*[(r.pid, r.ppid, 75.0 if r.pid == 50 else r.pcpu, r.comm) for r in TABLE])
    sample = classify(table, rule=RULE, monitor_pid=100, producer_pid=200, label="t")
    assert sorted(((c["pid"], c.get("reason")) for c in sample["competing"]), key=str) == [(300, None), (50, "ancestor above cap")]


def test_aggregate_guard_counts_many_small_processes():
    table = TABLE + rows(*[(400 + i, 1, 20.0, f"small{i}") for i in range(40)])
    sample = classify(table, rule=RULE, monitor_pid=100, producer_pid=200, label="t")
    reasons = [c.get("reason") for c in sample["competing"]]
    assert reasons == [None, "aggregate above cap"]
    assert sample["visible"] and all(v["pcpu"] < RULE.threshold_pcpu for v in sample["visible"][1:])


def test_summarize_takes_the_max_over_the_window_and_ignores_settle():
    quiet = classify(rows((1, 0, 0.0, "launchd"), (100, 1, 0.0, "monitor")), rule=RULE, monitor_pid=100,
                     producer_pid=None, label="before")
    busy = classify(TABLE, rule=RULE, monitor_pid=100, producer_pid=200, label="during_0001")
    settle_busy = {**busy, "label": "settle_0001"}
    summary = summarize([settle_busy, quiet, busy, quiet], rule=RULE, paused_pids=[7], monitor_command=["x"])
    assert summary["schema"] == CONCURRENCY_SCHEMA
    assert summary["competing_process_count"] == 1 and summary["pcpu_competing_process_count"] == 1
    assert summary["quiesced"] is False
    assert summary["samples_with_competition"] == ["during_0001"]
    assert summary["paused_pids"] == [7] and summary["sample_count"] == 4
    assert summary["window_sample_count"] == 3 and summary["settle_sample_count"] == 1
    only_settle = summarize([settle_busy, quiet], rule=RULE, paused_pids=[], monitor_command=["x"])
    assert only_settle["competing_process_count"] == 0 and only_settle["quiesced"] is True
    with pytest.raises(ConcurrencyError):
        summarize([settle_busy], rule=RULE, paused_pids=[], monitor_command=["x"])
    with pytest.raises(ConcurrencyError):
        summarize([], rule=RULE, paused_pids=[], monitor_command=["x"])


def _stage_checkpoints(directory: Path, seconds: list[float], *, start_epoch: float) -> Path:
    """Fake producer checkpoints: stage_NNNN.npz files whose mtimes encode the per-stage seconds."""
    directory.mkdir(parents=True, exist_ok=True)
    epoch = start_epoch
    for index, duration in enumerate([0.0, *seconds]):
        epoch += duration
        path = directory / f"stage_{25 * (index + 1):04d}.npz"
        path.write_bytes(b"x")
        os.utime(path, (epoch, epoch))
    return directory


def _timed_samples(labels_and_tables: list[tuple[str, list[ProcessRow]]], *, start_epoch: float, step: float) -> dict:
    samples = []
    for index, (label, table) in enumerate(labels_and_tables):
        sample = classify(table, rule=RULE, monitor_pid=100, producer_pid=200, label=label)
        sample["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_epoch))
        sample["monotonic"] = 1000.0 + index * step
        samples.append(sample)
    summary = summarize(samples, rule=ConcurrencyRule(interval_seconds=step, threshold_pcpu=25.0,
                                                      aggregate_cap_pcpu=700.0), paused_pids=[], monitor_command=["m"])
    return summary


def test_sample_epochs_are_utc_and_dst_proof():
    from tac.decode_timing_concurrency import _sample_epochs
    epoch = 1_789_031_551.0  # 2026-09-10T09:12:31Z, a DST date in America/Chicago
    samples = [{"time_utc": "2026-09-10T09:12:31Z", "monotonic": 10.0}, {"time_utc": "x", "monotonic": 30.5}]
    assert _sample_epochs(samples) == [epoch, epoch + 20.5]


QUIET = rows((1, 0, 0.0, "launchd"), (100, 1, 0.0, "monitor"), (200, 100, 380.0, "worker"))


def test_stage_rate_deviations_measure_the_slow_stage(tmp_path):
    base = 1_800_000_000.0
    directory = _stage_checkpoints(tmp_path / "ck", [21.0, 21.2, 21.1, 26.0, 21.3, 21.2], start_epoch=base)
    rates = stage_rate_deviations(directory, tolerance=0.05)
    assert rates["stage_count"] == 6 and rates["exceeding_stages"] == [125]
    assert rates["median_seconds"] == pytest.approx(21.2) and rates["max_deviation"] == pytest.approx(26.0 / 21.2 - 1)
    assert rates["covered_seconds"] == pytest.approx(sum([21.0, 21.2, 21.1, 26.0, 21.3, 21.2]))
    with pytest.raises(ConcurrencyError, match="too few"):
        stage_rate_deviations(_stage_checkpoints(tmp_path / "few", [21.0, 21.0], start_epoch=base))


def test_attribution_names_the_competitor_only_when_a_stage_slowed(tmp_path):
    base = 1_800_000_000.0
    seconds = [21.0, 21.2, 21.1, 26.0, 21.3, 21.2]
    rates = stage_rate_deviations(_stage_checkpoints(tmp_path / "ck", seconds, start_epoch=base), tolerance=0.05)
    # samples every 20 s from the start; the busy table shows up during the slow stage (t = 63..89) only
    tables = [("before", QUIET), ("during_0001", QUIET), ("during_0002", QUIET), ("during_0003", TABLE),
              ("during_0004", TABLE), ("during_0005", QUIET), ("during_0006", QUIET), ("after", QUIET)]
    summary = _timed_samples(tables, start_epoch=base, step=20.0)
    wall = sum(seconds) + 5.0
    verdict = attribute_competition(summary, rates, wall_seconds=wall, rule=RULE)
    assert verdict["competing_process_count"] == 1 and verdict["quiesced"] is False
    assert verdict["impact"]["exceeds"] is True and verdict["impact"]["excess_seconds"] == pytest.approx(4.9, abs=0.01)
    assert verdict["slowed_stages"][0]["stage"] == 125
    assert [p["pid"] for p in verdict["slowed_stages"][0]["processes_beside"]] == [300, 301]
    assert verdict["outside_stage_competitors"] == {}
    # the same competitor sampled while every stage ran at pace does not count, but stays on record
    flat = stage_rate_deviations(_stage_checkpoints(tmp_path / "flat", [21.0, 21.2, 21.1, 21.0, 21.3, 21.2],
                                                    start_epoch=base), tolerance=0.05)
    verdict = attribute_competition(summary, flat, wall_seconds=wall, rule=RULE)
    assert verdict["competing_process_count"] == 0 and verdict["quiesced"] is True
    assert verdict["pcpu_competing_process_count"] == 1 and verdict["slowed_stages"] == []
    assert verdict["impact"]["exceeds"] is False
    # a slow stage nobody was sampled next to still counts once, as unattributed
    quiet_summary = _timed_samples([(label, QUIET) for label, _ in tables], start_epoch=base, step=20.0)
    verdict = attribute_competition(quiet_summary, rates, wall_seconds=wall, rule=RULE)
    assert verdict["competing_process_count"] == 1
    assert verdict["slowed_stages"][0]["processes_beside"] == []
    # a competitor sampled OUTSIDE the instrumented stages (render phase) counts by the pcpu rule
    outside = _timed_samples([*[(label, QUIET) for label, _ in tables[:-1]], ("after", TABLE)], start_epoch=base, step=20.0)
    verdict = attribute_competition(outside, flat, wall_seconds=wall, rule=RULE)
    assert verdict["competing_process_count"] == 1 and list(verdict["outside_stage_competitors"]) == ["after"]
    # the same sample is harmless under the one-core default threshold (the dashboard sits at 30 %)
    verdict = attribute_competition(outside, flat, wall_seconds=wall, rule=ConcurrencyRule(aggregate_cap_pcpu=700.0))
    assert verdict["competing_process_count"] == 0


def test_read_process_table_sees_this_process():
    table = read_process_table()
    assert any(row.pid == os.getpid() for row in table)
    assert all(row.pcpu >= 0 for row in table)


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


def _concurrency(path: Path, count: int) -> Path:
    quiet = classify(rows((1, 0, 0.0, "launchd"), (100, 1, 0.0, "monitor")), rule=RULE, monitor_pid=100,
                     producer_pid=None, label="before")
    samples = [quiet] + ([classify(TABLE, rule=RULE, monitor_pid=100, producer_pid=200, label="during_0001")] * count)
    path.write_text(json.dumps(summarize(samples, rule=RULE, paused_pids=[], monitor_command=["m"])))
    return path


@pytest.fixture
def staged(tmp_path):
    from tac.tests.test_candidate_seal import _stage_candidate
    runtime, archive = _stage_candidate(tmp_path)
    return runtime, archive, tmp_path / "timing"


def test_assembled_quiesced_receipt_passes_the_validator(staged):
    runtime, archive, root = staged
    root.mkdir()
    producer = root / "TIMING.json"
    producer.write_text(json.dumps(_raw_timing(runtime, wall=40.0, pairs=20)))
    concurrency = _concurrency(root / "CONCURRENCY.json", count=0)
    local = assemble_local_receipt(json.loads(producer.read_text()), json.loads(concurrency.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    assert local["schema"] == LOCAL_SCHEMA and local["margin_time_basis"] == "quiesced"
    assert local["frames"] == list(range(20)) and local["wall_seconds"] == 40.0
    assert local["concurrency"]["competing_process_count"] == 0 and local["concurrency"]["receipt"]["path"] == str(concurrency)
    assert local["receiver_sha256"] == measure_receiver_digest(runtime)
    assert local["runtime_sha256"] == measure_runtime_digest(runtime).sha256
    local_path = root / "local.json"
    local_path.write_text(json.dumps(local))
    t4 = _t4_receipt(runtime, root / "t4.json", 900.0)
    calibration = write_calibration(name="synthetic", local_receipt_path=local_path, t4_receipt_path=t4)
    assert calibration["cpu_to_t4_ratio"] == pytest.approx(900.0 / 1200.0)
    calibration_path = root / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    leg = build_decode_wall_clock(local_receipt_path=local_path, calibration_receipt_path=calibration_path,
                                  runtime_dir=runtime, archive_path=archive, candidate_t4_receipt_path=t4)
    problems, observed = validate_decode_wall_clock(leg, runtime_dir=runtime, archive_path=archive)
    assert problems == [] and observed["measured_t4_decode_seconds"] == 900.0
    assert leg["projected_t4_decode_seconds"] == pytest.approx(900.0)


def test_nonzero_competition_is_refused_not_hidden(staged):
    runtime, archive, root = staged
    root.mkdir()
    producer = root / "TIMING.json"
    producer.write_text(json.dumps(_raw_timing(runtime, wall=40.0, pairs=20)))
    concurrency = _concurrency(root / "CONCURRENCY.json", count=1)
    local = assemble_local_receipt(json.loads(producer.read_text()), json.loads(concurrency.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    assert local["margin_time_basis"] == "measured_concurrency_nonzero"
    assert local["concurrency"]["competing_process_count"] == 1
    local_path = root / "local.json"
    local_path.write_text(json.dumps(local))
    t4 = _t4_receipt(runtime, root / "t4.json", 900.0)
    calibration_path = root / "calibration.json"
    calibration_path.write_text(json.dumps(write_calibration(name="synthetic", local_receipt_path=local_path, t4_receipt_path=t4)))
    with pytest.raises(SealContractError, match="quiesced"):
        build_decode_wall_clock(local_receipt_path=local_path, calibration_receipt_path=calibration_path,
                                runtime_dir=runtime, archive_path=archive)


def test_assemble_local_schema_keeps_the_producer_document(staged):
    runtime, archive, root = staged
    root.mkdir()
    producer_doc = {"schema": LOCAL_SCHEMA, "wall_seconds": 12.5, "frames": [0, 1], "cold_start": True,
                    "concurrency": {"competing_process_count": None}, "margin_time_basis": "unknown_host_concurrency",
                    "archive_sha256": "a", "receiver_sha256": "b"}
    producer = root / "producer.json"
    producer.write_text(json.dumps(producer_doc))
    concurrency = _concurrency(root / "CONCURRENCY.json", count=0)
    local = assemble_local_receipt(producer_doc, json.loads(concurrency.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    assert local["wall_seconds"] == 12.5 and local["margin_time_basis"] == "quiesced"
    assert local["producer_margin_time_basis"] == "unknown_host_concurrency"
    assert local["producer_concurrency"] == {"competing_process_count": None}
    assert local["producer_receipt"]["path"] == str(producer)
    assert producer_doc["margin_time_basis"] == "unknown_host_concurrency", "input document must not be mutated"


def test_assemble_rejects_foreign_or_unmeasured_blocks(staged):
    runtime, archive, root = staged
    root.mkdir()
    producer = root / "TIMING.json"
    producer.write_text(json.dumps(_raw_timing(runtime, wall=40.0, pairs=20)))
    concurrency = _concurrency(root / "CONCURRENCY.json", count=0)
    block = json.loads(concurrency.read_text())
    with pytest.raises(ConcurrencyError, match="measured_concurrency"):
        assemble_local_receipt(json.loads(producer.read_text()), {**block, "schema": "other"},
                               producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    with pytest.raises(ConcurrencyError, match="absent"):
        assemble_local_receipt(json.loads(producer.read_text()), {**block, "competing_process_count": None},
                               producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    with pytest.raises(ConcurrencyError, match="unknown producer schema"):
        assemble_local_receipt({"schema": "nope"}, block, producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    raw = _raw_timing(runtime, wall=40.0, pairs=0)
    with pytest.raises(ConcurrencyError, match="pair count"):
        assemble_local_receipt(raw, block, producer_receipt_path=producer, concurrency_receipt_path=concurrency)


def test_write_calibration_rejects_archive_mismatch(staged, tmp_path):
    runtime, archive, root = staged
    root.mkdir()
    producer = root / "TIMING.json"
    producer.write_text(json.dumps(_raw_timing(runtime, wall=40.0, pairs=20)))
    concurrency = _concurrency(root / "CONCURRENCY.json", count=0)
    local = assemble_local_receipt(json.loads(producer.read_text()), json.loads(concurrency.read_text()),
                                   producer_receipt_path=producer, concurrency_receipt_path=concurrency)
    local_path = root / "local.json"
    local_path.write_text(json.dumps(local))
    t4 = _t4_receipt(runtime, root / "t4.json", 900.0)
    doc = json.loads(t4.read_text())
    doc["expected_archive_sha256"] = "0" * 64
    t4.write_text(json.dumps(doc))
    with pytest.raises(ConcurrencyError, match="archive differs"):
        write_calibration(name="synthetic", local_receipt_path=local_path, t4_receipt_path=t4)


def test_run_producer_samples_a_real_process(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(0.8); print('done')"]
    code, summary = run_producer(command, cwd=tmp_path, env=dict(os.environ), stdout_path=tmp_path / "out.log",
                                 rule=ConcurrencyRule(interval_seconds=0.2, threshold_pcpu=10_000.0,
                                                      aggregate_cap_pcpu=1e9, ancestor_cap_pcpu=1e9),
                                 settle_seconds=5.0, paused_pids=[], timeout_seconds=30.0, monitor_command=["m"])
    assert code == 0 and (tmp_path / "out.log").read_text().strip() == "done"
    assert summary["schema"] == CONCURRENCY_SCHEMA and summary["sample_count"] >= 3
    labels = [sample["label"] for sample in summary["samples"]]
    assert labels[0].startswith("settle_") and "before" in labels and labels[-1] == "after"
    assert summary["competing_process_count"] == 0 and summary["quiesced"] is True


def test_cli_assemble_and_calibrate_round_trip(staged, capsys):
    from tools.quiesced_decode_timing import main
    runtime, archive, root = staged
    root.mkdir()
    producer = root / "TIMING.json"
    producer.write_text(json.dumps(_raw_timing(runtime, wall=40.0, pairs=20)))
    concurrency = _concurrency(root / "CONCURRENCY.json", count=0)
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
    sidecar = root / "SEAL.json.decode_wall_clock.json"
    assert json.loads(sidecar.read_text()) == json.loads((root / "leg.json").read_text())
    out = capsys.readouterr().out
    assert "margin_time_basis=quiesced" in out and "problems=none" in out
