"""Measured host concurrency for decode wall-clock timing receipts.

The prospective v3 rule implements ddm_pr11's exact host-baseline burst amendment.
Quarter-core individual/ancestor and 200% aggregate tests cover every non-settle
sample. Only the exact PID-1 dasd/syspolicyd pair can be waived, after global host,
member, combined, burst-count, sample-count and inclusive-span checks pass.
Stage rates are retained diagnostics and never decide admission. The assembler
copies and verifies the producer's frozen rule and burst evidence, then recounts;
old receipts keep their original rules and cannot be upgraded retroactively.
"""

from __future__ import annotations

import calendar
import copy
import hashlib
import itertools
import json
import math
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
ADMISSION_RULE_SCHEMA = "decode_wall_clock.admission_rule.v3"
ADMISSION_RULE_V2 = "decode_wall_clock.admission_rule.v2"
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
    """The pr11 frozen v3 cap (two cores); never derived from a failed host query."""
    return 200.0


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


V3_DEFINITION = (
    'ordinary competitor = pcpu >= threshold_pcpu for every process outside producer tree, monitor tree, '
    'and pid 0; monitor ancestors compete at pcpu >= ancestor_cap_pcpu. The host-baseline allowance is '
    'disabled unless platform.system(), platform.machine(), os.cpu_count(), and p_core_count() exactly '
    'equal host_constraints. A host-baseline row matches only exact comm and ppid; a burst-active sample '
    'has at least one matching row at pcpu >= activation_pcpu. The allowance is valid only when matching '
    'rows stay pcpu <= their member caps, matching-row sum is <= combined cap in every sample, active '
    'samples form at most one consecutive run, active-sample count is <= max_active_samples_per_burst, '
    'and last.monotonic - first.monotonic + interval_seconds <= max_inclusive_span_seconds. If valid, '
    'matching rows do not compete and are subtracted from the aggregate; if any condition fails, no '
    'matching row is waived. Unwaived summed pcpu competes at >= aggregate_cap_pcpu. Apply to every '
    'non-settle sample; count = max over samples; admitted iff count == 0. Start after '
    'settle_quiet_samples consecutive samples with zero unwaived competitors; a later allowance failure '
    'refuses the final receipt. Stage rates are diagnostic only.'
)


def host_constraints() -> dict:
    """Exact running-host shape, captured when constructing the prospective rule."""
    return {"platform_system": platform.system(), "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(), "performance_core_count": p_core_count()}


def _host_baseline_burst() -> dict:
    """The exact pr11 envelope, with no vendor or basename exemptions."""
    return {'schema': 'decode_wall_clock.host_baseline_burst.v1', 'activation_pcpu': 25.0, 'members': [{'comm': '/usr/libexec/dasd', 'ppid': 1, 'pcpu_max_inclusive': 100.0}, {'comm': '/usr/libexec/syspolicyd', 'ppid': 1, 'pcpu_max_inclusive': 75.0}], 'combined_pcpu_max_inclusive': 175.0, 'max_bursts_in_window': 1, 'max_active_samples_per_burst': 15, 'max_inclusive_span_seconds': 300.0}


@dataclass(frozen=True)
class ConcurrencyRule:
    threshold_pcpu: float = 25.0
    visible_pcpu: float = 5.0
    ancestor_cap_pcpu: float = 25.0
    aggregate_cap_pcpu: float = field(default_factory=_default_aggregate_cap)
    interval_seconds: float = 20.0
    settle_quiet_samples: int = 3
    stage_tolerance: float = 0.05
    host_constraints: dict = field(default_factory=host_constraints)
    host_baseline_burst: dict = field(default_factory=_host_baseline_burst)
    schema: str = ADMISSION_RULE_SCHEMA

    def frozen(self) -> dict:
        """The rule as serialized into receipts; `sha256` covers exactly these fields."""
        body = asdict(self)
        if self.schema == ADMISSION_RULE_SCHEMA:
            return {**body, "decode_threads": DECODE_THREADS, "definition": V3_DEFINITION}
        body.pop("host_constraints")
        body.pop("host_baseline_burst")
        return {**body, "decode_threads": DECODE_THREADS,
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
        if not isinstance(body, dict) or body.get("schema") not in {ADMISSION_RULE_SCHEMA, ADMISSION_RULE_V2}:
            raise ConcurrencyError("admission rule absent or of an unknown schema")
        keys = ("threshold_pcpu", "visible_pcpu", "ancestor_cap_pcpu", "aggregate_cap_pcpu", "interval_seconds",
                "settle_quiet_samples", "stage_tolerance")
        try:
            if body["schema"] == ADMISSION_RULE_SCHEMA:
                keys += ("host_constraints", "host_baseline_burst")
            rule = cls(**copy.deepcopy({key: body[key] for key in keys}), schema=body["schema"])
            if rule.schema == ADMISSION_RULE_SCHEMA:
                if json.dumps(rule.host_baseline_burst, sort_keys=True) != json.dumps(_host_baseline_burst(), sort_keys=True):
                    raise ConcurrencyError("host-baseline envelope differs from the exact v3 class")
                expected = {"platform_system": str, "machine": str, "logical_cpu_count": int, "performance_core_count": int}
                if set(rule.host_constraints) != set(expected) or any(
                        type(rule.host_constraints[key]) is not kind for key, kind in expected.items()):
                    raise ConcurrencyError("host constraints must specify all four exact fields")
            return rule
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
    matched = []
    for row in rows:
        if row.pid == 0 or row.pid in producer_tree or row.pid in monitor_tree:
            continue
        entry = {"pid": row.pid, "ppid": row.ppid, "pcpu": row.pcpu, "comm": row.comm}
        entry["host_baseline_member_match"] = _member(entry, rule) is not None
        if entry["host_baseline_member_match"]:
            matched.append({**entry, "ancestor": row.pid in ancestors})
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
            "host_constraints_observed": host_constraints(), "host_baseline_matches": matched,
            "excluded_ancestors_active": excluded, "other_pcpu_sum": other_sum,
            "producer_tree_size": len(producer_tree), "monitor_tree_size": len(monitor_tree),
            "process_rows": len(rows)}


def _member(entry: dict, rule: ConcurrencyRule) -> dict | None:
    """Identity only; host eligibility and all bounds are adjudicated over the window."""
    if rule.schema != ADMISSION_RULE_SCHEMA:
        return None
    return next((member for member in rule.host_baseline_burst["members"]
                 if entry.get("comm") == member["comm"] and type(entry.get("ppid")) is int
                 and entry["ppid"] == member["ppid"]), None)


def _matched_rows(sample: dict, rule: ConcurrencyRule) -> list[dict]:
    # New samples also retain matching rows below visible_pcpu. Older traces can only
    # reconstruct their visible rows; those are counterfactual tests, never new authority.
    entries = [{**row, "ancestor": False} for row in sample.get("visible", [])]
    entries += [{**row, "ancestor": True} for row in sample.get("excluded_ancestors_active", [])]
    by_pid = {row["pid"]: row for row in entries if _member(row, rule) is not None}
    for row in sample.get("host_baseline_matches", []):
        if _member(row, rule) is None:
            raise ConcurrencyError("stored host-baseline row is not an exact member")
        if row["pid"] in by_pid:
            visible = by_pid[row["pid"]]
            if any(row.get(key) != visible.get(key) for key in ("comm", "ppid", "pcpu", "ancestor")):
                raise ConcurrencyError("host-baseline row differs from visible process evidence")
        by_pid[row["pid"]] = row
    return list(by_pid.values())


def adjudicate_host_baseline(samples: list[dict], rule: ConcurrencyRule) -> dict:
    """Adjudicate an entire non-settle window (or an observed settle prefix), without mutation."""
    envelope = rule.host_baseline_burst
    observed_host = host_constraints()
    checks = {"schema_v3": rule.schema == ADMISSION_RULE_SCHEMA}
    for key, value in rule.host_constraints.items():
        checks["host_" + key] = (type(observed_host.get(key)) is type(value)
                                  and observed_host.get(key) == value
                                  and all(type(s.get("host_constraints_observed", observed_host).get(key)) is type(value)
                                          and s.get("host_constraints_observed", observed_host).get(key) == value
                                          for s in samples))
    records, bursts = [], []
    previous_active = False
    for index, sample in enumerate(samples):
        matched = _matched_rows(sample, rule)
        member_checks = []
        for row in matched:
            cap = _member(row, rule)["pcpu_max_inclusive"]
            member_checks.append({"row": row, "pcpu_max_inclusive": cap,
                                  "passed": math.isfinite(row["pcpu"]) and 0 <= row["pcpu"] <= cap})
        total = sum(row["pcpu"] for row in matched)
        active = any(row["pcpu"] >= envelope["activation_pcpu"] for row in matched)
        record = {"label": sample["label"], "sample_index": index, "monotonic": sample["monotonic"],
                  "matched_rows": matched, "member_bounds": member_checks, "burst_active": active,
                  "matched_pcpu_sum": total,
                  "matched_other_pcpu_sum": sum(row["pcpu"] for row in matched if not row.get("ancestor", False)),
                  "combined_pcpu_max_inclusive": envelope["combined_pcpu_max_inclusive"],
                  "combined_cap_passed": math.isfinite(total) and total <= envelope["combined_pcpu_max_inclusive"]}
        records.append(record)
        if active:
            if not previous_active:
                bursts.append({"sample_indices": [], "labels": [], "first_monotonic": sample["monotonic"]})
            burst = bursts[-1]
            burst["sample_indices"].append(index)
            burst["labels"].append(sample["label"])
            burst["last_monotonic"] = sample["monotonic"]
        previous_active = active
    for burst in bursts:
        burst["active_samples"] = len(burst["sample_indices"])
        burst["inclusive_span_seconds"] = burst["last_monotonic"] - burst["first_monotonic"] + rule.interval_seconds
        burst["max_active_samples_per_burst"] = envelope["max_active_samples_per_burst"]
        burst["max_inclusive_span_seconds"] = envelope["max_inclusive_span_seconds"]
        burst["checks"] = {
            "active_samples": burst["active_samples"] <= envelope["max_active_samples_per_burst"],
            "inclusive_span": 0 < burst["inclusive_span_seconds"] <= envelope["max_inclusive_span_seconds"]}
    checks.update({
        "member_caps": all(bound["passed"] for record in records for bound in record["member_bounds"]),
        "combined_cap": all(record["combined_cap_passed"] for record in records),
        "burst_count": len(bursts) <= envelope["max_bursts_in_window"],
        "active_samples_per_burst": all(burst["checks"]["active_samples"] for burst in bursts),
        "inclusive_span": all(burst["checks"]["inclusive_span"] for burst in bursts),
        "monotonic_order": all(math.isfinite(s["monotonic"]) for s in samples)
                           and all(b["monotonic"] >= a["monotonic"] for a, b in itertools.pairwise(samples)),
    })
    return {"valid": all(checks.values()), "checks": checks, "bursts": bursts, "samples": records,
            "observed_host": observed_host, "burst_count": len(bursts),
            "max_bursts_in_window": envelope["max_bursts_in_window"]}


def sample_competitors(sample: dict, rule: ConcurrencyRule, *, allowance_valid: bool = False) -> list[dict]:
    """Recount with no waiver unless the complete window (or settle prefix) has passed."""
    waived = {row["pid"] for row in _matched_rows(sample, rule)} if allowance_valid else set()
    named = [entry for entry in sample.get("visible", [])
             if entry["pcpu"] >= rule.threshold_pcpu and entry["pid"] not in waived]
    named += [{**entry, "reason": "ancestor at or above cap"} for entry in sample.get("excluded_ancestors_active", [])
              if entry["pcpu"] >= rule.ancestor_cap_pcpu and entry["pid"] not in waived]
    other_sum = sample.get("other_pcpu_sum", 0.0)
    if allowance_valid:
        other_sum -= sum(row["pcpu"] for row in _matched_rows(sample, rule) if not row.get("ancestor", False))
    if other_sum >= rule.aggregate_cap_pcpu:
        named.append({"reason": "aggregate at or above cap", "other_pcpu_sum": other_sum})
    return named


def _adjudicate_window(samples: list[dict], rule: ConcurrencyRule) -> tuple[list[dict], dict, list[int]]:
    samples = copy.deepcopy(samples)
    window = window_of(samples)
    if not window:
        raise ConcurrencyError("no concurrency samples inside the measurement window")
    allowance = adjudicate_host_baseline(window, rule)
    for sample, record in zip(window, allowance["samples"], strict=True):
        sample["host_baseline_matches"] = record["matched_rows"]
        sample["host_baseline_burst_active"] = record["burst_active"]
        sample["unwaived_other_pcpu_sum"] = sample["other_pcpu_sum"] - (
            record["matched_other_pcpu_sum"] if allowance["valid"] else 0.0)
        sample["unwaived_competing"] = sample_competitors(sample, rule, allowance_valid=allowance["valid"])
    return samples, allowance, [len(sample["unwaived_competing"]) for sample in window]


def _margin_basis(count: int, allowance: dict) -> str:
    if count:
        return "measured_concurrency_nonzero"
    if not allowance["bursts"]:
        return "quiesced"
    return "bounded_host_baseline" if allowance["valid"] else "measured_concurrency_nonzero"


def window_of(samples: list[dict]) -> list[dict]:
    return [sample for sample in samples if not sample["label"].startswith("settle_")]


def summarize(samples: list[dict], *, rule: ConcurrencyRule, paused: list[dict], monitor_command: list[str],
              frozen_at_utc: str, producer_started_at_utc: str | None) -> dict:
    """Settle samples are recorded but never counted; the count is the max over the window."""
    samples, allowance, counts = _adjudicate_window(samples, rule)
    window = window_of(samples)
    count = max(counts)
    nonzero = [sample["label"] for sample, n in zip(window, counts, strict=True) if n]
    return {"schema": CONCURRENCY_SCHEMA, "admission_rule": rule.frozen(), "admission_rule_sha256": rule.sha256(),
            "admission_rule_frozen_at_utc": frozen_at_utc, "producer_started_at_utc": producer_started_at_utc,
            "competing_process_count": count, "quiesced": _margin_basis(count, allowance) == "quiesced",
            "margin_time_basis": _margin_basis(count, allowance),
            "host_baseline_allowance": allowance, "host_baseline_bursts": allowance["bursts"],
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
    prefix = []
    while True:
        attempt += 1
        sample = monitor.sample(f"settle_{attempt:04d}")
        prefix.append(sample)
        allowance = adjudicate_host_baseline(prefix, rule)
        sample["host_baseline_provisional_allowance"] = allowance
        competing = sample_competitors(sample, rule, allowance_valid=allowance["valid"])
        sample["unwaived_competing"] = competing
        sample["unwaived_other_pcpu_sum"] = sample["other_pcpu_sum"] - (
            allowance["samples"][-1]["matched_other_pcpu_sum"] if allowance["valid"] else 0.0)
        quiet = [*quiet, sample] if not competing else []
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
    rule = ConcurrencyRule.from_frozen(rule.frozen())
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
    if rule.schema == ADMISSION_RULE_SCHEMA and any(
            not isinstance(sample.get("host_constraints_observed"), dict)
            or set(sample["host_constraints_observed"]) != set(rule.host_constraints)
            or not isinstance(sample.get("host_baseline_matches"), list)
            for sample in concurrency["samples"]):
        raise ConcurrencyError("v3 sample lacks captured host or matched-row evidence")
    samples, allowance, counts = _adjudicate_window(concurrency["samples"], rule)
    window = window_of(samples)
    if rule.schema == ADMISSION_RULE_SCHEMA and (
            allowance != concurrency.get("host_baseline_allowance")
            or allowance["bursts"] != concurrency.get("host_baseline_bursts")
            or samples != concurrency["samples"]):
        raise ConcurrencyError("re-derived host-baseline adjudication differs from the sampler")
    count = max(counts)
    if count != concurrency.get("competing_process_count"):
        raise ConcurrencyError("re-derived competitor count differs from the sampler's count")
    basis = _margin_basis(count, allowance)
    quiesced = basis == "quiesced"
    block = {"competing_process_count": count, "quiesced": quiesced,
             "attribution": concurrency["attribution"], "admission_rule": concurrency["admission_rule"],
             "admission_rule_sha256": concurrency["admission_rule_sha256"],
             "admission_rule_frozen_at_utc": concurrency["admission_rule_frozen_at_utc"],
             "producer_started_at_utc": concurrency["producer_started_at_utc"],
             "samples_with_competition": [s["label"] for s, n in zip(window, counts, strict=True) if n],
             "load_average": concurrency["load_average"], "load_average_max": concurrency["load_average_max"],
             "sample_count": concurrency["sample_count"], "window_sample_count": len(window),
             "paused_processes": concurrency["paused_processes"], "receipt": _fact(concurrency_receipt_path)}
    if rule.schema == ADMISSION_RULE_SCHEMA:
        block["host_baseline_allowance"] = copy.deepcopy(concurrency["host_baseline_allowance"])
        block["host_baseline_bursts"] = copy.deepcopy(concurrency["host_baseline_bursts"])
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
    doc["margin_time_basis"] = basis
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
    if local.get("margin_time_basis") == "bounded_host_baseline":
        from tac.decode_wall_clock import _local

        _local(_fact(local_receipt_path))
    elif local.get("margin_time_basis") != "quiesced":
        raise ConcurrencyError("calibration needs a quiesced or valid bounded-host-baseline local receipt")
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
    "adjudicate_host_baseline",
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
