"""Measured host concurrency for decode wall-clock timing receipts.

`tac.decode_wall_clock` admits a margin basis only when the local timing ran quiesced:
`concurrency.competing_process_count == 0` and `margin_time_basis == "quiesced"`.
The dwc1 timing producers recorded either the TOTAL number of processes on the host
(every `ps` row) or `null`; neither can ever meet that contract, so every leg they
produced was refused with "competing process count absent" even when the actual T4
decode of the same bytes completed with 270 s of margin.

This module measures COMPETING processes by a stated rule, samples them through the
whole producer run, and assembles the local receipt from the producer's own document
without retyping a single timing number.

Rule (recorded verbatim in every receipt; fixed 2026-09-10 after two measured runs):
  1. MEASURED IMPACT is the authority where the producer leaves stage checkpoints
     (`stage_NNNN.npz`, one per 25 pairs of the token stage): the run's median stage time
     is its own pace; the total excess over that pace, summed over stages, must stay
     within `impact_tolerance` of the wall (default 0.01 — thirty times inside the 0.7
     safety factor the 1,260 s limit already carries). Stages slower than the median by
     more than `stage_tolerance` (0.05) are listed with the processes sampled beside them.
     Two cold runs of the move 40 receiver showed why: the macOS activity scheduler
     `dasd` at ~96 % for five minutes moved the stage time by < 1 %, while two venv
     pythons at 67–100 % plus a git moved one stage by 10–22 %.
  2. Outside the instrumented stages (native build, verifier, render) the process samples
     decide: a process competes when its %CPU (the `ps` decaying average) is at least
     `threshold_pcpu` (default 100.0 = more than one full core, i.e. a multi-threaded job
     that can take P-cores from the four decode threads; a single-core background daemon
     cannot) and it is none of (a) the producer's tree, (b) the monitor's tree, (c) the
     monitor's ancestor chain (the control plane that launched the measurement), (d) pid 0.
     Excluded ancestors count above `ancestor_cap_pcpu` (default 100.0).
  3. An aggregate guard counts once more when the summed %CPU of every non-excluded
     process reaches `aggregate_cap_pcpu` (default 100 x (ncpu - 4) / 2) in any sample.
  Every process at or above `visible_pcpu` (default 5.0) is listed per sample, so a
  reviewer can re-decide from the receipt alone. Settle samples (the wait for a quiet
  host before the producer starts) are recorded but never counted. `quiesced` means the
  count is zero. Pausing a process (SIGSTOP) for the window is allowed only when the
  receipt lists it.
"""

from __future__ import annotations

import calendar
import itertools
import json
import os
import platform
import signal
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tac.candidate_seal import measure_archive_identity, measure_runtime_digest
from tac.decode_wall_clock import (
    CALIBRATION_SCHEMA,
    LOCAL_SCHEMA,
    measure_receiver_digest,
    receipt_reference,
)

RAW_TIMING_SCHEMA = "ddm_dwc1.raw_timing.v1"
CONCURRENCY_SCHEMA = "decode_wall_clock.measured_concurrency.v1"
T4_RUNTIME_DIGEST_DEFINITION = "tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest"
T4_SECONDS_FIELD = ["artifacts", "contest_auth_eval.json", "inflate_elapsed_seconds"]
T4_ARCHIVE_FIELD = ["expected_archive_sha256"]
T4_RUNTIME_FIELD = ["expected_runtime_tree_sha256"]
T4_HARDWARE_FIELD = ["artifacts", "modal_cuda_preflight.json", "torch_cuda_device_name"]


class ConcurrencyError(ValueError):
    """A receipt could not be assembled honestly."""


def _default_aggregate_cap() -> float:
    return 100.0 * max(1, (os.cpu_count() or 4) - 4) / 2


@dataclass(frozen=True)
class ConcurrencyRule:
    threshold_pcpu: float = 100.0
    visible_pcpu: float = 5.0
    ancestor_cap_pcpu: float = 100.0
    aggregate_cap_pcpu: float = field(default_factory=_default_aggregate_cap)
    interval_seconds: float = 20.0
    impact_tolerance: float = 0.01
    stage_tolerance: float = 0.05

    def as_dict(self) -> dict:
        return {**asdict(self), "ncpu": os.cpu_count(), "definition": (
            "measured impact first: total stage excess over the run's median pace <= impact_tolerance "
            "x wall (stages > stage_tolerance listed with the processes sampled beside them); outside "
            "the instrumented stages a process competes at %CPU >= threshold_pcpu unless it is in the "
            "producer tree, monitor tree, monitor ancestors (which count above ancestor_cap_pcpu), or "
            "pid 0; the summed %CPU of non-excluded processes counts once more at or above "
            "aggregate_cap_pcpu; every process >= visible_pcpu is listed; settle samples never count; "
            "quiesced iff the count is 0")}


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    ppid: int
    pcpu: float
    comm: str


def read_process_table() -> list[ProcessRow]:
    """One `ps` pass over every process; `comm` keeps only the executable path."""
    result = subprocess.run(["ps", "-axo", "pid,ppid,pcpu,comm"], capture_output=True, text=True,
                            timeout=10, check=True)
    rows: list[ProcessRow] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        rows.append(ProcessRow(int(parts[0]), int(parts[1]), float(parts[2]), parts[3] if len(parts) == 4 else ""))
    return rows


def tree_of(root: int, rows: list[ProcessRow]) -> set[int]:
    children: dict[int, list[int]] = {}
    for row in rows:
        children.setdefault(row.ppid, []).append(row.pid)
    seen, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen


def ancestors_of(pid: int, rows: list[ProcessRow]) -> list[int]:
    parent = {row.pid: row.ppid for row in rows}
    chain: list[int] = []
    while pid in parent and parent[pid] > 0 and parent[pid] not in chain:
        pid = parent[pid]
        chain.append(pid)
    return chain


def classify(rows: list[ProcessRow], *, rule: ConcurrencyRule, monitor_pid: int,
             producer_pid: int | None, label: str) -> dict:
    producer_tree = tree_of(producer_pid, rows) if producer_pid is not None else set()
    monitor_tree = tree_of(monitor_pid, rows)
    ancestors = set(ancestors_of(monitor_pid, rows))
    competing, visible, excluded, other_sum = [], [], [], 0.0
    for row in rows:
        if row.pid == 0 or row.pid in producer_tree or row.pid in monitor_tree:
            continue
        entry = {"pid": row.pid, "ppid": row.ppid, "pcpu": row.pcpu, "comm": row.comm}
        if row.pid in ancestors:
            if row.pcpu >= rule.visible_pcpu:
                excluded.append(entry)
            if row.pcpu > rule.ancestor_cap_pcpu:
                competing.append({**entry, "reason": "ancestor above cap"})
            continue
        other_sum += row.pcpu
        if row.pcpu >= rule.visible_pcpu:
            visible.append(entry)
        if row.pcpu >= rule.threshold_pcpu:
            competing.append(entry)
    if other_sum >= rule.aggregate_cap_pcpu:
        competing.append({"reason": "aggregate above cap", "other_pcpu_sum": other_sum})
    return {"label": label, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monotonic": time.monotonic(), "load_average": list(os.getloadavg()),
            "competing": competing, "visible": visible, "excluded_ancestors_active": excluded,
            "other_pcpu_sum": other_sum, "producer_tree_size": len(producer_tree),
            "monitor_tree_size": len(monitor_tree), "process_rows": len(rows)}


def summarize(samples: list[dict], *, rule: ConcurrencyRule, paused_pids: list[int],
              monitor_command: list[str]) -> dict:
    """Summarize the window: settle samples (the wait-for-quiet phase before the producer
    started) are recorded but never counted; the count is the max over before/during/after."""
    window = [sample for sample in samples if not sample["label"].startswith("settle_")]
    if not window:
        raise ConcurrencyError("no concurrency samples inside the measurement window")
    count = max(len(sample["competing"]) for sample in window)
    nonzero = [sample["label"] for sample in window if sample["competing"]]
    return {"schema": CONCURRENCY_SCHEMA, "competing_process_count": count, "pcpu_competing_process_count": count,
            "quiesced": count == 0, "attribution": "pcpu rule over the measurement window (settle samples excluded)",
            "load_average": window[0]["load_average"], "load_average_max": [
                max(sample["load_average"][i] for sample in window) for i in range(3)],
            "sample_count": len(samples), "window_sample_count": len(window),
            "settle_sample_count": len(samples) - len(window), "samples_with_competition": nonzero,
            "other_pcpu_sum_max": max(sample["other_pcpu_sum"] for sample in window),
            "visible_process_count_max": max(len(sample["visible"]) for sample in window),
            "rule": rule.as_dict(), "paused_pids": list(paused_pids), "monitor_command": list(monitor_command),
            "monitor_host": platform.node(), "ncpu": os.cpu_count(), "samples": samples}


STAGE_RATES_SCHEMA = "decode_wall_clock.stage_rates.v1"


def stage_rate_deviations(checkpoint_dir: Path, *, tolerance: float = 0.05, minimum_stages: int = 4) -> dict:
    """Measure contention directly: the producer's stage checkpoints (`stage_NNNN.npz`, one per
    25 pairs) give the decode rate over time; a stage slower than the run's median by more than
    `tolerance` is a measured slowdown, whoever caused it."""
    files = sorted(Path(checkpoint_dir).glob("stage_*.npz"))
    marks = [(int(path.stem.split("_")[1]), path.stat().st_mtime) for path in files]
    if len(marks) < minimum_stages + 1:
        raise ConcurrencyError(f"too few stage checkpoints in {checkpoint_dir}: {len(marks)}")
    stages = [{"stage": end, "start_epoch": t0, "end_epoch": t1, "seconds": t1 - t0}
              for (_, t0), (end, t1) in itertools.pairwise(marks)]
    median = statistics.median(stage["seconds"] for stage in stages)
    for stage in stages:
        stage["deviation"] = stage["seconds"] / median - 1.0
        stage["exceeds"] = stage["deviation"] > tolerance
    return {"schema": STAGE_RATES_SCHEMA, "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
            "stage_count": len(stages), "median_seconds": median, "tolerance": tolerance,
            "max_deviation": max(stage["deviation"] for stage in stages),
            "exceeding_stages": [stage["stage"] for stage in stages if stage["exceeds"]],
            "covered_seconds": marks[-1][1] - marks[0][1], "first_epoch": marks[0][1], "last_epoch": marks[-1][1],
            "stages": stages}


def _sample_epochs(samples: list[dict]) -> list[float]:
    base = calendar.timegm(time.strptime(samples[0]["time_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    origin = samples[0]["monotonic"]
    return [base + (sample["monotonic"] - origin) for sample in samples]


def _sample_competitors(sample: dict, rule: ConcurrencyRule) -> list[dict]:
    """Re-derive a sample's competitors under `rule` from its full `visible` list (>= 5 %), so the
    verdict does not depend on the threshold that happened to be in force at sampling time."""
    named = [entry for entry in sample.get("visible", []) if entry["pcpu"] >= rule.threshold_pcpu]
    named += [{**entry, "reason": "ancestor above cap"} for entry in sample.get("excluded_ancestors_active", [])
              if entry["pcpu"] > rule.ancestor_cap_pcpu]
    if sample.get("other_pcpu_sum", 0.0) >= rule.aggregate_cap_pcpu:
        named.append({"reason": "aggregate above cap", "other_pcpu_sum": sample["other_pcpu_sum"]})
    return named


def attribute_competition(concurrency: dict, rates: dict, *, wall_seconds: float,
                          rule: ConcurrencyRule | None = None) -> dict:
    """The verdict of the rule in the module docstring.

    Inside the instrumented stages the measured total excess over the run's own pace decides;
    outside them the re-derived process samples decide; an aggregate-over-cap sample counts
    anywhere. Zero means the decode ran at its own pace on an otherwise idle host.
    """
    rule = ConcurrencyRule() if rule is None else rule
    samples = concurrency["samples"]
    epochs = _sample_epochs(samples)
    window = [(index, sample) for index, sample in enumerate(samples) if not sample["label"].startswith("settle_")]
    interval = float(concurrency["rule"]["interval_seconds"])
    median = rates["median_seconds"]
    excess = sum(max(0.0, stage["seconds"] - median) for stage in rates["stages"])
    impact = excess / float(wall_seconds)
    slowed, stage_counts = [], []
    for stage in rates["stages"]:
        if not stage["exceeds"]:
            continue
        overlapping = [(index, sample) for index, sample in window
                       if stage["start_epoch"] - interval <= epochs[index] <= stage["end_epoch"] + interval]
        beside = {}
        for _, sample in overlapping:
            for entry in sample.get("visible", []):
                beside[entry["pid"]] = max(beside.get(entry["pid"], entry), entry, key=lambda e: e["pcpu"])
        competitors = max((len(_sample_competitors(sample, rule)) for _, sample in overlapping), default=0)
        stage_counts.append(max(1, competitors))
        slowed.append({"stage": stage["stage"], "deviation": stage["deviation"], "seconds": stage["seconds"],
                       "samples": [sample["label"] for _, sample in overlapping],
                       "processes_beside": sorted(beside.values(), key=lambda e: -e["pcpu"])})
    impact_count = max(stage_counts, default=1) if impact > rule.impact_tolerance else 0
    outside = [(index, sample) for index, sample in window
               if not rates["first_epoch"] <= epochs[index] <= rates["last_epoch"]]
    outside_hits = {sample["label"]: _sample_competitors(sample, rule) for _, sample in outside}
    outside_hits = {label: hits for label, hits in outside_hits.items() if hits}
    outside_count = max((len(hits) for hits in outside_hits.values()), default=0)
    aggregate = [sample["label"] for _, sample in window if sample.get("other_pcpu_sum", 0.0) >= rule.aggregate_cap_pcpu]
    count = impact_count + outside_count + (1 if aggregate else 0)
    pcpu_window = max((len(_sample_competitors(sample, rule)) for _, sample in window), default=0)
    return {"competing_process_count": count, "quiesced": count == 0,
            "attribution": "measured impact inside the instrumented stages; process samples outside them; "
                           "aggregate-over-cap samples anywhere (rule in verdict_rule)",
            "verdict_rule": rule.as_dict(), "pcpu_competing_process_count": pcpu_window,
            "impact": {"excess_seconds": excess, "wall_seconds": float(wall_seconds), "fraction": impact,
                       "tolerance": rule.impact_tolerance, "exceeds": impact > rule.impact_tolerance,
                       "covered_fraction": rates["covered_seconds"] / float(wall_seconds)},
            "slowed_stages": slowed, "outside_stage_competitors": outside_hits,
            "aggregate_over_cap_samples": aggregate,
            "stage_rates": {key: rates[key] for key in ("schema", "checkpoint_dir", "stage_count", "median_seconds",
                                                        "tolerance", "max_deviation", "exceeding_stages",
                                                        "covered_seconds")}}


class ConcurrencyMonitor:
    """Sample the process table at a fixed cadence for the life of one producer run."""

    def __init__(self, rule: ConcurrencyRule, *, monitor_pid: int | None = None):
        self.rule = rule
        self.monitor_pid = os.getpid() if monitor_pid is None else monitor_pid
        self.producer_pid: int | None = None
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def sample(self, label: str) -> dict:
        value = classify(read_process_table(), rule=self.rule, monitor_pid=self.monitor_pid,
                         producer_pid=self.producer_pid, label=label)
        with self._lock:
            self.samples.append(value)
        return value

    def start(self, producer_pid: int) -> None:
        self.producer_pid = producer_pid

        def loop() -> None:
            index = 0
            while not self._stop.wait(self.rule.interval_seconds):
                index += 1
                self.sample(f"during_{index:04d}")

        self._thread = threading.Thread(target=loop, name="concurrency-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)


def wait_until_quiet(monitor: ConcurrencyMonitor, *, settle_seconds: float, poll_seconds: float = 5.0) -> dict:
    """Do not start timing on a host that is still busy; give up after `settle_seconds`."""
    deadline = time.monotonic() + settle_seconds
    attempt = 0
    while True:
        attempt += 1
        sample = monitor.sample(f"settle_{attempt:04d}")
        if not sample["competing"]:
            return sample
        if time.monotonic() >= deadline:
            raise ConcurrencyError("host did not quiesce within the settle window: "
                                   + json.dumps(sample["competing"]))
        time.sleep(poll_seconds)


class PausedProcesses:
    """SIGSTOP the listed pids for the window; SIGCONT them no matter how the window ends."""

    def __init__(self, pids: list[int]):
        self.pids = list(pids)

    def __enter__(self) -> PausedProcesses:
        for pid in self.pids:
            os.kill(pid, signal.SIGSTOP)
        return self

    def __exit__(self, *_exc: object) -> None:
        for pid in self.pids:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


def run_producer(command: list[str], *, cwd: Path, env: dict[str, str], stdout_path: Path,
                 rule: ConcurrencyRule, settle_seconds: float, paused_pids: list[int],
                 timeout_seconds: float, monitor_command: list[str]) -> tuple[int, dict]:
    """Run one timing producer under the sampler; returns (returncode, concurrency summary)."""
    monitor = ConcurrencyMonitor(rule)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with PausedProcesses(paused_pids):
        wait_until_quiet(monitor, settle_seconds=settle_seconds)
        monitor.sample("before")
        with stdout_path.open("wb") as stream:
            process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            monitor.start(process.pid)
            try:
                code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
            finally:
                monitor.stop()
        monitor.sample("after")
    return code, summarize(monitor.samples, rule=rule, paused_pids=paused_pids, monitor_command=monitor_command)


def _fact(path: Path) -> dict:
    return receipt_reference(Path(path))


def assemble_local_receipt(producer_doc: dict, concurrency: dict, *, producer_receipt_path: Path,
                           concurrency_receipt_path: Path, stage_checkpoint_dir: Path | None = None,
                           rule: ConcurrencyRule | None = None) -> dict:
    """Turn a producer's document plus a measured concurrency block into a local.v1 receipt.

    Every timing field is copied from the producer document. Only the concurrency block and
    the margin basis are supplied here, and both point back at their measurement receipts.
    With `stage_checkpoint_dir` the count is impact-weighted (see `attribute_competition`);
    without it the strict pcpu count over the window stands.
    """
    if concurrency.get("schema") != CONCURRENCY_SCHEMA:
        raise ConcurrencyError("concurrency block must be a measured_concurrency.v1 document")
    verdict = {"competing_process_count": concurrency.get("competing_process_count"),
               "quiesced": concurrency.get("quiesced"), "attribution": concurrency.get("attribution"),
               "pcpu_competing_process_count": concurrency.get("pcpu_competing_process_count",
                                                                concurrency.get("competing_process_count"))}
    if stage_checkpoint_dir is not None:
        rule = ConcurrencyRule() if rule is None else rule
        rates = stage_rate_deviations(stage_checkpoint_dir, tolerance=rule.stage_tolerance)
        verdict = attribute_competition(concurrency, rates, wall_seconds=float(producer_doc["wall_seconds"]), rule=rule)
    count = verdict["competing_process_count"]
    if type(count) is not int or count < 0:
        raise ConcurrencyError("measured competing process count absent")
    quiesced = count == 0 and verdict.get("quiesced") is True
    block = {**verdict, "competing_process_count": count, "quiesced": quiesced,
             "load_average": concurrency["load_average"], "load_average_max": concurrency["load_average_max"],
             "sample_count": concurrency["sample_count"], "rule": concurrency["rule"],
             "paused_pids": concurrency["paused_pids"], "receipt": _fact(concurrency_receipt_path)}
    schema = producer_doc.get("schema")
    if schema == RAW_TIMING_SCHEMA:
        binding = producer_doc["binding"]
        report = producer_doc.get("report") or {}
        runtime_dir = Path(binding["copied_runtime"])
        pair_count = int(report.get("pair_count", 0))
        if pair_count < 1:
            raise ConcurrencyError("producer report carries no pair count")
        completed = (producer_doc.get("returncode") == 0 and not producer_doc.get("timed_out")
                     and "raw" in producer_doc)
        doc = {
            "schema": LOCAL_SCHEMA, "axis": str(producer_doc["axis"]).strip("[]"),
            "measurement_kind": producer_doc["measurement_kind"], "completed": completed,
            "cold_start": bool(producer_doc["cold_start"]), "resumed": bool(producer_doc["resumed"]),
            "checkpoint_resume": False, "checkpoint_resumed_from_frame": 0,
            "cpu_threads": producer_doc["cpu_threads"], "frames": list(range(pair_count)),
            "host": producer_doc["host"], "platform": producer_doc["platform"],
            "command": list(producer_doc["command"]), "wall_seconds": producer_doc["wall_seconds"],
            "timed_scope": producer_doc.get("timed_scope"),
            "public_entrypoint_executed": producer_doc.get("public_entrypoint_executed", False),
            "public_shell_startup_included": producer_doc.get("public_shell_startup_included", False),
            "runtime_dir": str(runtime_dir), "archive_path": str(runtime_dir / "archive.zip"),
            "runtime_sha256": measure_runtime_digest(runtime_dir).sha256,
            "receiver_sha256": measure_receiver_digest(runtime_dir),
            "archive_sha256": measure_archive_identity(runtime_dir / "archive.zip").sha256,
            "score_claim": False, "raw_timing_receipt": _fact(producer_receipt_path),
            "producer_concurrency": {"before": producer_doc.get("concurrency_before"),
                                     "after": producer_doc.get("concurrency_after")},
        }
        if doc["archive_sha256"] != binding["archive"]["sha256"]:
            raise ConcurrencyError("producer binding archive differs from the runtime archive")
    elif schema == LOCAL_SCHEMA:
        doc = dict(producer_doc)
        doc["producer_concurrency"] = doc.pop("concurrency", None)
        doc["producer_margin_time_basis"] = doc.pop("margin_time_basis", None)
        doc["producer_receipt"] = _fact(producer_receipt_path)
    else:
        raise ConcurrencyError(f"unknown producer schema: {schema!r}")
    doc["concurrency"] = block
    doc["margin_time_basis"] = "quiesced" if quiesced else "measured_concurrency_nonzero"
    return doc


def _field(document: object, keys: list[str]) -> object:
    value = document
    for key in keys:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict) or key not in value:
            raise ConcurrencyError("T4 receipt field missing: " + "/".join(keys))
        value = value[key]
    return value


def write_calibration(*, name: str, local_receipt_path: Path, t4_receipt_path: Path) -> dict:
    """Calibration = T4 seconds over the local 600-pair extrapolation of the SAME receiver+archive."""
    local = json.loads(Path(local_receipt_path).read_text())
    t4 = json.loads(Path(t4_receipt_path).read_text())
    if local.get("schema") != LOCAL_SCHEMA:
        raise ConcurrencyError("calibration needs a local.v1 receipt")
    frames = local["frames"]
    local600 = float(local["wall_seconds"]) * 600 / len(frames)
    seconds = float(_field(t4, T4_SECONDS_FIELD))
    if _field(t4, T4_ARCHIVE_FIELD) != local["archive_sha256"]:
        raise ConcurrencyError("T4 receipt archive differs from the local timing archive")
    return {"schema": CALIBRATION_SCHEMA, "name": name, "local_receipt": _fact(local_receipt_path),
            "t4_receipt": _fact(t4_receipt_path), "t4_seconds_field": T4_SECONDS_FIELD,
            "t4_archive_sha256_field": T4_ARCHIVE_FIELD, "t4_runtime_sha256_field": T4_RUNTIME_FIELD,
            "t4_hardware_field": T4_HARDWARE_FIELD, "t4_runtime_digest_definition": T4_RUNTIME_DIGEST_DEFINITION,
            "t4_timing_scope": "decode", "archive_sha256": local["archive_sha256"],
            "receiver_sha256": local["receiver_sha256"], "cpu_to_t4_ratio": seconds / local600,
            "local_600_seconds": local600, "measured_t4_seconds": seconds}


__all__ = [
    "CONCURRENCY_SCHEMA",
    "RAW_TIMING_SCHEMA",
    "STAGE_RATES_SCHEMA",
    "ConcurrencyError",
    "ConcurrencyMonitor",
    "ConcurrencyRule",
    "PausedProcesses",
    "ProcessRow",
    "ancestors_of",
    "assemble_local_receipt",
    "attribute_competition",
    "classify",
    "read_process_table",
    "run_producer",
    "stage_rate_deviations",
    "summarize",
    "tree_of",
    "wait_until_quiet",
    "write_calibration",
]
