"""Measured host concurrency for decode wall-clock timing receipts.

`tac.decode_wall_clock` admits a margin basis only when the local timing ran quiesced:
`concurrency.competing_process_count == 0` and `margin_time_basis == "quiesced"`.
The dwc1 timing producers recorded either the TOTAL number of processes on the host
(every `ps` row) or `null`; neither can ever meet that contract, so every leg they
produced was refused with "competing process count absent" even when the actual T4
decode of the same bytes completed with 270 s of margin.

This module measures COMPETING processes by a rule that is FROZEN AND HASHED BEFORE the
producer starts, samples the process table through the whole run, and assembles the local
receipt from the producer's own document without retyping a single timing number. The
assembler cannot override the rule: it copies the frozen rule from the sampling receipt and
refuses on any hash mismatch (ddm_pr10, 2026-09-10: a rule fixed after the runs were on
the table is the defect, whatever it says).

Admission rule (ddm_pr10's replacement, conservative by design — false-negative-prone,
never a claim that every counted process caused a measurable slowdown):
  1. A process competes when its %CPU (the `ps` decaying average) is at least
     `threshold_pcpu` (25.0 = a quarter core) and it is not the producer's tree, the
     monitor's tree, or pid 0. Excluded ancestors (the control plane that launched the
     measurement) compete at `ancestor_cap_pcpu` (25.0, same `>=`): ancestry is not
     permission to consume a quarter core during a calibration.
  2. An aggregate guard counts once more when the summed %CPU of every non-excluded
     process reaches `aggregate_cap_pcpu` (100 x (P-cores - 4 decode threads) = 200 on
     this 6-P-core host), closing the many-sub-threshold-process class.
  3. The rules apply to EVERY non-settle sample (before, during, after) including the
     checkpoint-instrumented stages; the count is the max over those samples; `quiesced`
     iff it is zero. Stage-rate deviations (`stage_tolerance` 0.05) are DIAGNOSTICS only —
     they never turn a counted competitor into zero.
  4. The producer starts only after `settle_quiet_samples` (3) consecutive quiet samples at
     the normal cadence. Settle samples are recorded but never counted.
  5. Every process paused for the window (SIGSTOP) carries identity and transition custody
     (pid, ppid, comm, start time, stop/resume requested + confirmed) in the receipt.
Every process at or above `visible_pcpu` (5.0) is listed per sample so a reviewer can
re-derive the count from the receipt alone.
"""

from __future__ import annotations

import calendar
import hashlib
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
CONCURRENCY_SCHEMA = "decode_wall_clock.measured_concurrency.v2"
STAGE_RATES_SCHEMA = "decode_wall_clock.stage_rates.v1"
ADMISSION_RULE_SCHEMA = "decode_wall_clock.admission_rule.v2"
T4_RUNTIME_DIGEST_DEFINITION = "tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest"
T4_SECONDS_FIELD = ["artifacts", "contest_auth_eval.json", "inflate_elapsed_seconds"]
T4_ARCHIVE_FIELD = ["expected_archive_sha256"]
T4_RUNTIME_FIELD = ["expected_runtime_tree_sha256"]
T4_HARDWARE_FIELD = ["artifacts", "modal_cuda_preflight.json", "torch_cuda_device_name"]
DECODE_THREADS = 4


class ConcurrencyError(ValueError):
    """A receipt could not be assembled honestly."""


def p_core_count() -> int:
    """Performance cores (macOS `hw.perflevel0.physicalcpu`); logical CPUs elsewhere."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return os.cpu_count() or DECODE_THREADS


def _default_aggregate_cap() -> float:
    return 100.0 * max(1, p_core_count() - DECODE_THREADS)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class ConcurrencyRule:
    threshold_pcpu: float = 25.0
    visible_pcpu: float = 5.0
    ancestor_cap_pcpu: float = 25.0
    aggregate_cap_pcpu: float = field(default_factory=_default_aggregate_cap)
    interval_seconds: float = 20.0
    settle_quiet_samples: int = 3
    stage_tolerance: float = 0.05

    def frozen(self) -> dict:
        """The rule as serialized into receipts; `sha256` covers exactly these fields."""
        return {"schema": ADMISSION_RULE_SCHEMA, **asdict(self), "decode_threads": DECODE_THREADS,
                "definition": (
                    "competing = %CPU >= threshold_pcpu for every process outside the producer tree, "
                    "monitor tree, and pid 0; monitor ancestors compete at %CPU >= ancestor_cap_pcpu; the "
                    "summed %CPU of non-excluded processes counts once more at or above aggregate_cap_pcpu; "
                    "applied to every non-settle sample; count = max over samples; quiesced iff 0; stage "
                    "rates are diagnostics only; producer starts after settle_quiet_samples consecutive "
                    "quiet samples")}

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.frozen(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_frozen(cls, body: object) -> ConcurrencyRule:
        if not isinstance(body, dict) or body.get("schema") != ADMISSION_RULE_SCHEMA:
            raise ConcurrencyError("admission rule absent or of an unknown schema")
        keys = ("threshold_pcpu", "visible_pcpu", "ancestor_cap_pcpu", "aggregate_cap_pcpu", "interval_seconds",
                "settle_quiet_samples", "stage_tolerance")
        try:
            return cls(**{key: body[key] for key in keys})
        except KeyError as exc:
            raise ConcurrencyError(f"admission rule field missing: {exc}") from exc


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
    """One sample: competitors under `rule`, plus everything visible so a reviewer can re-derive."""
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
            if row.pcpu >= rule.ancestor_cap_pcpu:
                competing.append({**entry, "reason": "ancestor at or above cap"})
            continue
        other_sum += row.pcpu
        if row.pcpu >= rule.visible_pcpu:
            visible.append(entry)
        if row.pcpu >= rule.threshold_pcpu:
            competing.append(entry)
    if other_sum >= rule.aggregate_cap_pcpu:
        competing.append({"reason": "aggregate at or above cap", "other_pcpu_sum": other_sum})
    return {"label": label, "time_utc": _utc_now(), "monotonic": time.monotonic(),
            "load_average": list(os.getloadavg()), "competing": competing, "visible": visible,
            "excluded_ancestors_active": excluded, "other_pcpu_sum": other_sum,
            "producer_tree_size": len(producer_tree), "monitor_tree_size": len(monitor_tree),
            "process_rows": len(rows)}


def sample_competitors(sample: dict, rule: ConcurrencyRule) -> list[dict]:
    """Re-derive a sample's competitors under `rule` from its full `visible` list (>= 5 %)."""
    named = [entry for entry in sample.get("visible", []) if entry["pcpu"] >= rule.threshold_pcpu]
    named += [{**entry, "reason": "ancestor at or above cap"} for entry in sample.get("excluded_ancestors_active", [])
              if entry["pcpu"] >= rule.ancestor_cap_pcpu]
    if sample.get("other_pcpu_sum", 0.0) >= rule.aggregate_cap_pcpu:
        named.append({"reason": "aggregate at or above cap", "other_pcpu_sum": sample["other_pcpu_sum"]})
    return named


def window_of(samples: list[dict]) -> list[dict]:
    return [sample for sample in samples if not sample["label"].startswith("settle_")]


def summarize(samples: list[dict], *, rule: ConcurrencyRule, paused: list[dict], monitor_command: list[str],
              frozen_at_utc: str, producer_started_at_utc: str | None) -> dict:
    """Settle samples are recorded but never counted; the count is the max over the window."""
    window = window_of(samples)
    if not window:
        raise ConcurrencyError("no concurrency samples inside the measurement window")
    counts = [len(sample_competitors(sample, rule)) for sample in window]
    count = max(counts)
    nonzero = [sample["label"] for sample, n in zip(window, counts, strict=True) if n]
    return {"schema": CONCURRENCY_SCHEMA, "admission_rule": rule.frozen(), "admission_rule_sha256": rule.sha256(),
            "admission_rule_frozen_at_utc": frozen_at_utc, "producer_started_at_utc": producer_started_at_utc,
            "competing_process_count": count, "quiesced": count == 0,
            "attribution": "frozen admission rule over every non-settle sample (max); competitors re-derived from "
                           "each sample's visible list",
            "load_average": window[0]["load_average"], "load_average_max": [
                max(sample["load_average"][i] for sample in window) for i in range(3)],
            "sample_count": len(samples), "window_sample_count": len(window),
            "settle_sample_count": len(samples) - len(window), "samples_with_competition": nonzero,
            "other_pcpu_sum_max": max(sample["other_pcpu_sum"] for sample in window),
            "visible_process_count_max": max(len(sample["visible"]) for sample in window),
            "paused_processes": list(paused), "monitor_command": list(monitor_command),
            "monitor_host": platform.node(), "ncpu": os.cpu_count(), "p_cores": p_core_count(), "samples": samples}


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


def wait_until_quiet(monitor: ConcurrencyMonitor, *, settle_seconds: float) -> list[dict]:
    """Start only after `settle_quiet_samples` CONSECUTIVE quiet samples at the normal cadence."""
    rule = monitor.rule
    deadline = time.monotonic() + settle_seconds
    quiet: list[dict] = []
    attempt = 0
    while True:
        attempt += 1
        sample = monitor.sample(f"settle_{attempt:04d}")
        quiet = [*quiet, sample] if not sample["competing"] else []
        if len(quiet) >= rule.settle_quiet_samples:
            return quiet
        if time.monotonic() >= deadline:
            raise ConcurrencyError("host did not quiesce within the settle window: "
                                   + json.dumps(sample["competing"]))
        time.sleep(rule.interval_seconds)


def _identity(pid: int) -> dict:
    out = subprocess.run(["ps", "-o", "pid=,ppid=,lstart=,stat=,comm=", "-p", str(pid)],
                         capture_output=True, text=True, timeout=10)
    line = out.stdout.strip()
    if out.returncode != 0 or not line:
        raise ConcurrencyError(f"paused process {pid} not found")
    parts = line.split(None, 2)
    rest = parts[2]
    start_time, stat_comm = rest[:24].strip(), rest[24:].split(None, 1)
    return {"pid": int(parts[0]), "ppid": int(parts[1]), "start_time": start_time,
            "stat": stat_comm[0], "comm": stat_comm[1] if len(stat_comm) > 1 else ""}


class PausedProcesses:
    """SIGSTOP the listed pids for the window with identity + transition custody; SIGCONT them
    no matter how the window ends."""

    def __init__(self, pids: list[int]):
        self.pids = list(pids)
        self.records: list[dict] = []

    def __enter__(self) -> PausedProcesses:
        for pid in self.pids:
            before = _identity(pid)
            record = {**before, "stop_requested_at_utc": _utc_now()}
            os.kill(pid, signal.SIGSTOP)
            time.sleep(0.2)
            after = _identity(pid)
            record["stop_confirmed"] = "T" in after["stat"] and after["start_time"] == before["start_time"]
            self.records.append(record)
            if not record["stop_confirmed"]:
                raise ConcurrencyError(f"could not confirm SIGSTOP on pid {pid}: stat {after['stat']!r}")
        return self

    def __exit__(self, *_exc: object) -> None:
        for record in self.records:
            record["resume_requested_at_utc"] = _utc_now()
            try:
                os.kill(record["pid"], signal.SIGCONT)
                time.sleep(0.2)
                after = _identity(record["pid"])
                record["resume_confirmed"] = "T" not in after["stat"] and after["start_time"] == record["start_time"]
            except (ProcessLookupError, ConcurrencyError):
                record["resume_confirmed"] = False
                record["resume_note"] = "process gone at resume"


def run_producer(command: list[str], *, cwd: Path, env: dict[str, str], stdout_path: Path,
                 rule: ConcurrencyRule, settle_seconds: float, paused_pids: list[int],
                 timeout_seconds: float, monitor_command: list[str]) -> tuple[int, dict]:
    """Run one timing producer under the sampler; returns (returncode, concurrency summary).
    The rule is frozen (hashed) here, before the settle wait and before the producer starts."""
    frozen_at = _utc_now()
    monitor = ConcurrencyMonitor(rule)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_at: str | None = None
    paused = PausedProcesses(paused_pids)
    with paused:
        wait_until_quiet(monitor, settle_seconds=settle_seconds)
        monitor.sample("before")
        with stdout_path.open("wb") as stream:
            started_at = _utc_now()
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
    return code, summarize(monitor.samples, rule=rule, paused=paused.records, monitor_command=monitor_command,
                           frozen_at_utc=frozen_at, producer_started_at_utc=started_at)


def _fact(path: Path) -> dict:
    return receipt_reference(Path(path))


def stage_rate_deviations(checkpoint_dir: Path, *, tolerance: float = 0.05, minimum_stages: int = 4) -> dict:
    """DIAGNOSTIC: the producer's stage checkpoints (`stage_NNNN.npz`, one per 25 pairs) give the
    decode rate over time; stages slower than the run's median by more than `tolerance` are listed.
    A uniform pace shift is invisible to this instrument (it uses the run's own median), which is
    why it never decides admission."""
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
            "excess_seconds": sum(max(0.0, stage["seconds"] - median) for stage in stages),
            "exceeding_stages": [stage["stage"] for stage in stages if stage["exceeds"]],
            "covered_seconds": marks[-1][1] - marks[0][1], "first_epoch": marks[0][1], "last_epoch": marks[-1][1],
            "stages": stages}


def _sample_epochs(samples: list[dict]) -> list[float]:
    base = calendar.timegm(time.strptime(samples[0]["time_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    origin = samples[0]["monotonic"]
    return [base + (sample["monotonic"] - origin) for sample in samples]


def stage_diagnostics(concurrency: dict, rates: dict, *, wall_seconds: float, rule: ConcurrencyRule) -> dict:
    """Per-stage slowdowns with the processes sampled beside them. Informational only."""
    samples = concurrency["samples"]
    epochs = _sample_epochs(samples)
    interval = rule.interval_seconds
    slowed = []
    for stage in rates["stages"]:
        if not stage["exceeds"]:
            continue
        beside: dict[int, dict] = {}
        labels = []
        for sample, epoch in zip(samples, epochs, strict=True):
            if sample["label"].startswith("settle_") or not (
                    stage["start_epoch"] - interval <= epoch <= stage["end_epoch"] + interval):
                continue
            labels.append(sample["label"])
            for entry in sample.get("visible", []):
                if entry["pid"] not in beside or entry["pcpu"] > beside[entry["pid"]]["pcpu"]:
                    beside[entry["pid"]] = entry
        slowed.append({"stage": stage["stage"], "deviation": stage["deviation"], "seconds": stage["seconds"],
                       "samples": labels, "processes_beside": sorted(beside.values(), key=lambda e: -e["pcpu"])})
    return {"schema": rates["schema"], "checkpoint_dir": rates["checkpoint_dir"], "stage_count": rates["stage_count"],
            "median_seconds": rates["median_seconds"], "tolerance": rates["tolerance"],
            "max_deviation": rates["max_deviation"], "excess_seconds": rates["excess_seconds"],
            "excess_fraction_of_wall": rates["excess_seconds"] / float(wall_seconds),
            "covered_fraction_of_wall": rates["covered_seconds"] / float(wall_seconds),
            "exceeding_stages": rates["exceeding_stages"], "slowed_stages": slowed,
            "role": "diagnostic only; never decides admission"}


def _hardware_fingerprint(producer_doc: dict) -> dict:
    """Carry the producer's fingerprint; when it recorded none and we are assembling ON the timing
    host, record the machine architecture so calibration and candidate identities compare like for
    like (`ddm_rlc1_public.py` writes `hardware_fingerprint`; `ddm_dwc1_move40_timing.py` did not)."""
    if "hardware_fingerprint" in producer_doc:
        return {"hardware_fingerprint": producer_doc["hardware_fingerprint"]}
    if producer_doc.get("host") == platform.node():
        return {"hardware_fingerprint": platform.machine()}
    return {}


def assemble_local_receipt(producer_doc: dict, concurrency: dict, *, producer_receipt_path: Path,
                           concurrency_receipt_path: Path, stage_checkpoint_dir: Path | None = None) -> dict:
    """Turn a producer's document plus a measured concurrency block into a local.v1 receipt.

    Every timing field is copied from the producer document. The admission rule is COPIED from
    the sampling receipt and its hash verified; nothing here can loosen it. The count is
    re-derived under that frozen rule over every non-settle sample and must equal the sampler's.
    """
    if concurrency.get("schema") != CONCURRENCY_SCHEMA:
        raise ConcurrencyError("concurrency block must be a measured_concurrency.v2 document (frozen rule)")
    rule = ConcurrencyRule.from_frozen(concurrency.get("admission_rule"))
    if concurrency.get("admission_rule_sha256") != rule.sha256() or rule.frozen() != concurrency.get("admission_rule"):
        raise ConcurrencyError("admission rule hash mismatch: the rule in the receipt is not the frozen rule")
    for key in ("admission_rule_frozen_at_utc", "producer_started_at_utc"):
        if not isinstance(concurrency.get(key), str) or not concurrency[key]:
            raise ConcurrencyError(f"{key} absent: the rule must be frozen before the producer starts")
    if concurrency["admission_rule_frozen_at_utc"] > concurrency["producer_started_at_utc"]:
        raise ConcurrencyError("admission rule was frozen after the producer started")
    window = window_of(concurrency["samples"])
    if not window:
        raise ConcurrencyError("no concurrency samples inside the measurement window")
    counts = [len(sample_competitors(sample, rule)) for sample in window]
    count = max(counts)
    if count != concurrency.get("competing_process_count"):
        raise ConcurrencyError("re-derived competitor count differs from the sampler's count")
    quiesced = count == 0
    block = {"competing_process_count": count, "quiesced": quiesced,
             "attribution": concurrency["attribution"], "admission_rule": concurrency["admission_rule"],
             "admission_rule_sha256": concurrency["admission_rule_sha256"],
             "admission_rule_frozen_at_utc": concurrency["admission_rule_frozen_at_utc"],
             "producer_started_at_utc": concurrency["producer_started_at_utc"],
             "samples_with_competition": [s["label"] for s, n in zip(window, counts, strict=True) if n],
             "load_average": concurrency["load_average"], "load_average_max": concurrency["load_average_max"],
             "sample_count": concurrency["sample_count"], "window_sample_count": len(window),
             "paused_processes": concurrency["paused_processes"], "receipt": _fact(concurrency_receipt_path)}
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
            **_hardware_fingerprint(producer_doc),
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
    if stage_checkpoint_dir is not None:
        rates = stage_rate_deviations(stage_checkpoint_dir, tolerance=rule.stage_tolerance)
        block["stage_diagnostics"] = stage_diagnostics(concurrency, rates, wall_seconds=float(doc["wall_seconds"]), rule=rule)
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
    if local.get("margin_time_basis") != "quiesced":
        raise ConcurrencyError("calibration needs a quiesced local receipt")
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
            "local_600_seconds": local600, "measured_t4_seconds": seconds,
            "admission_rule_sha256": local["concurrency"]["admission_rule_sha256"]}


__all__ = [
    "ADMISSION_RULE_SCHEMA",
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
    "classify",
    "p_core_count",
    "read_process_table",
    "run_producer",
    "sample_competitors",
    "stage_diagnostics",
    "stage_rate_deviations",
    "summarize",
    "tree_of",
    "wait_until_quiet",
    "window_of",
    "write_calibration",
]
