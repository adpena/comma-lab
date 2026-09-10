"""Measured decode budget receipts. Timing projections are advisory, never score authority.

The receiver identity differs from the seal runtime identity ONLY by replacing the
literal values of the two top-level archive pins. Payload-dependent work can still
change: inheritance records code identity, not a new timing measurement.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

from tac.candidate_seal import (
    SealContractError,
    measure_archive_identity,
    measure_runtime_digest,
    runtime_digest_skip_reason,
    sha256_file,
)

DECODE_WALL_CLOCK_SCHEMA = "candidate_decode_wall_clock.v1"
DECODE_WALL_CLOCK_SCHEMA_V2 = "candidate_decode_wall_clock.v2"  # ddm_pr11: mode "t4_direct"
T4_HARDWARE_NAMES = {"nvidia-t4", "tesla t4", "nvidia tesla t4", "nvidia t4"}
T4_CANONICAL_PATH = "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda"
T4_DIRECT_FIELDS = {
    "t4_seconds_field": ["artifacts", "contest_auth_eval.json", "inflate_elapsed_seconds"],
    "t4_archive_sha256_field": ["expected_archive_sha256"],
    "t4_runtime_sha256_field": ["expected_runtime_tree_sha256"],
    "t4_hardware_field": ["artifacts", "modal_cuda_preflight.json", "torch_cuda_device_name"],
    "t4_cuda_available_field": ["artifacts", "modal_cuda_preflight.json", "torch_cuda_available"],
}
T4_DIRECT_SCOPE = "cold_public_entrypoint_decode"
INHERITANCE_SCOPE = "identical normalized receiver code; candidate payload-dependent time not remeasured"
BOUNDED_BASIS = "bounded_host_baseline"
ADMISSION_RULE_V3 = "decode_wall_clock.admission_rule.v3"
LOCAL_SCHEMA = "decode_wall_clock.local.v1"
CALIBRATION_SCHEMA = "decode_wall_clock.calibration.v1"
LIMIT_SECONDS = 1260.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealContractError("decode_wall_clock: " + message)


def _number(value: object, label: str, *, zero: bool = False) -> float:
    _require(type(value) in (int, float), f"{label} must be a finite number")
    result = float(value)
    _require(math.isfinite(result) and (result >= 0 if zero else result > 0), f"{label} out of range")
    return result


def _read(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SealContractError(f"decode_wall_clock: cannot read {path}: {exc}") from exc
    _require(isinstance(doc, dict), f"{path} must contain an object")
    return doc


def receipt_reference(path: Path) -> dict:
    path = Path(path).resolve()
    _require(path.is_file(), f"receipt missing: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _receipt(ref: object) -> dict:
    _require(isinstance(ref, dict), "receipt reference must be an object")
    _require(isinstance(ref.get("path"), str) and bool(ref["path"]), "receipt path absent")
    path = Path(ref["path"])
    _require(path.is_file(), f"receipt missing: {path}")
    _require(sha256_file(path) == ref.get("sha256"), f"receipt drift: {path}")
    return _read(path)


def measure_receiver_digest(runtime_dir: Path) -> str:
    """Hash shipped files, normalizing only the values of explicit archive pin assignments."""
    root = Path(runtime_dir)
    _require(root.is_dir(), f"runtime missing: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "archive.zip" or runtime_digest_skip_reason(rel):
            continue
        data = path.read_bytes()
        if rel == "inflate.py":
            try:
                tree = ast.parse(data)
            except (SyntaxError, UnicodeDecodeError) as exc:
                raise SealContractError(f"decode_wall_clock: invalid inflate.py: {exc}") from exc
            lines = data.splitlines(keepends=True)
            offsets = [sum(map(len, lines[:i])) for i in range(len(lines))]
            edits = []
            seen = set()
            for node in tree.body:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name) or target.id not in {"ARCHIVE_SHA256", "ARCHIVE_BYTES"}:
                    continue
                _require(target.id not in seen, f"duplicate archive pin {target.id}")
                seen.add(target.id)
                value = node.value
                expected = str if target.id == "ARCHIVE_SHA256" else int
                _require(isinstance(value, ast.Constant) and type(value.value) is expected,
                         f"nonliteral archive pin {target.id}")
                edits.append((offsets[value.lineno - 1] + value.col_offset,
                              offsets[value.end_lineno - 1] + value.end_col_offset))
            for start, end in sorted(edits, reverse=True):
                data = data[:start] + b"<ARCHIVE_PIN>" + data[end:]
        rows.append((rel, len(data), hashlib.sha256(data).hexdigest()))
    _require(bool(rows), "receiver tree empty")
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _local(ref: object, *, require_margin_basis: bool = True) -> tuple[dict, float]:
    doc = _receipt(ref)
    _require(doc.get("schema") == LOCAL_SCHEMA, "unknown local timing schema")
    _require(doc.get("axis") == "macOS-CPU advisory", "local timing axis must be macOS-CPU advisory")
    _require(doc.get("measurement_kind") in {"public_entrypoint_decode", "cuda_path_equivalent_proxy"},
             "timing must cover complete decode work, not token-only parsing")
    _require(doc.get("completed") is True, "decode did not complete")
    _require(doc.get("cold_start") is True and doc.get("resumed", False) is False
             and doc.get("checkpoint_resume", False) is False
             and doc.get("checkpoint_resumed_from_frame", 0) == 0,
             "timing requires a cold start; resumed suffix timing is not a full decode")
    _require(type(doc.get("cpu_threads")) is int and doc["cpu_threads"] == 4, "timing requires exactly four CPU threads")
    frames = doc.get("frames")
    _require(isinstance(frames, list) and 20 <= len(frames) <= 600, "timing needs 20..600 consecutive pair frames")
    _require(all(type(x) is int for x in frames), "frame indices must be integers")
    _require(frames == list(range(frames[0], frames[0] + len(frames))) and frames[0] >= 0 and frames[-1] < 600,
             "pair frames must be consecutive within 0..599")
    _require(isinstance(doc.get("host"), str) and bool(doc["host"].strip()), "host absent")
    _require(isinstance(doc.get("command"), list) and bool(doc["command"])
             and all(isinstance(x, str) and x for x in doc["command"]), "command argv absent")
    concurrency = doc.get("concurrency")
    _require(isinstance(concurrency, dict), "concurrency object absent")
    count = concurrency.get("competing_process_count")
    _require((type(count) is int and count >= 0) or (count is None and not require_margin_basis),
             "competing process count absent")
    load = concurrency.get("load_average")
    _require(isinstance(load, list) and len(load) == 3, "three load averages required")
    for value in load:
        _number(value, "load average", zero=True)
    if require_margin_basis:
        basis = doc.get("margin_time_basis")
        if basis == BOUNDED_BASIS:
            _require(count == 0, "bounded host baseline needs competing_process_count == 0")
            rule = concurrency.get("admission_rule")
            _require(isinstance(rule, dict) and rule.get("schema") == ADMISSION_RULE_V3,
                     "bounded host baseline is admissible only under admission_rule.v3")
            digest = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
            _require(concurrency.get("admission_rule_sha256") == digest, "frozen admission rule hash mismatch")
            allowance = concurrency.get("host_baseline_allowance")
            _require(isinstance(allowance, dict) and allowance.get("valid") is True, "host baseline allowance not valid")
            checks = allowance.get("checks")
            _require(isinstance(checks, dict) and bool(checks) and all(v is True for v in checks.values()),
                     "every host baseline allowance check must be true")
        else:
            _require(basis == "quiesced" and count == 0,
                     "margin timing must be quiesced; competing measurements need measured normalization")
    for name in ("runtime_dir", "archive_path"):
        _require(isinstance(doc.get(name), str) and bool(doc[name]), f"{name} absent")
    root, archive = Path(doc["runtime_dir"]), Path(doc["archive_path"])
    _require(measure_runtime_digest(root).sha256 == doc.get("runtime_sha256"), "local runtime drift")
    _require(measure_receiver_digest(root) == doc.get("receiver_sha256"), "local receiver drift")
    _require(measure_archive_identity(archive).sha256 == doc.get("archive_sha256"), "local archive drift")
    seconds = _number(doc.get("wall_seconds"), "wall seconds")
    return doc, seconds * 600 / len(frames)


def _field(document: dict, keys: object, label: str) -> object:
    _require(isinstance(keys, list) and bool(keys) and all(isinstance(k, str) for k in keys), f"{label} field path absent")
    value = document
    for key in keys:
        if isinstance(value, str):
            value = json.loads(value)
        _require(isinstance(value, dict) and key in value, f"{label} receipt field missing: {key}")
        value = value[key]
    return value


def measure_t4_runtime_digest(runtime_dir: Path) -> str:
    """Use the worker's upload projection, preserving its distinct digest definition."""
    from experiments.contest_auth_eval import _runtime_dependency_manifest
    from tac.deploy.modal.auth_eval import modal_uploaded_submission_dir_runtime_manifest

    repo = Path(__file__).resolve().parents[2]
    manifest = _runtime_dependency_manifest(Path(runtime_dir) / "inflate.sh", repo / "upstream")
    return modal_uploaded_submission_dir_runtime_manifest(manifest)["runtime_tree_sha256"]


def _calibration(ref: object, *, require_margin_basis: bool = True) -> tuple[dict, float]:
    doc = _receipt(ref)
    _require(doc.get("schema") == CALIBRATION_SCHEMA, "unknown calibration schema")
    _require(isinstance(doc.get("name"), str) and bool(doc["name"].strip()), "calibration name absent")
    local, local600 = _local(doc.get("local_receipt"), require_margin_basis=require_margin_basis)
    t4 = _receipt(doc.get("t4_receipt"))
    # A hash authenticates source bytes, not the meaning of an arbitrary numeric field.
    # Bind the worker's actual decode/status fields so evaluate time cannot masquerade
    # as inflate time. Whole-row upper bounds need a separate source contract.
    _require(doc.get("t4_timing_scope") == "decode", "only completed decode calibration is supported")
    canonical_fields = {
        "t4_seconds_field": ["artifacts", "contest_auth_eval.json", "inflate_elapsed_seconds"],
        "t4_archive_sha256_field": ["expected_archive_sha256"],
        "t4_runtime_sha256_field": ["expected_runtime_tree_sha256"],
        "t4_hardware_field": ["artifacts", "modal_cuda_preflight.json", "torch_cuda_device_name"],
    }
    for field_name, expected_path in canonical_fields.items():
        _require(doc.get(field_name) == expected_path, f"{field_name} must name the canonical T4 source field")
    _require(t4.get("passed") is True and type(t4.get("returncode")) is int and t4["returncode"] == 0,
             "T4 calibration source must be a completed successful result")
    seconds = _number(_field(t4, doc.get("t4_seconds_field"), "T4 seconds"), "T4 receipt seconds")
    _require(doc.get("archive_sha256") == local["archive_sha256"], "calibration archive differs from local timing")
    _require(doc.get("receiver_sha256") == local["receiver_sha256"], "calibration receiver differs from local timing")
    _require(_field(t4, doc.get("t4_archive_sha256_field"), "T4 archive") == local["archive_sha256"],
             "T4 source archive differs from calibration")
    hardware = _field(t4, doc.get("t4_hardware_field"), "T4 hardware")
    _require(isinstance(hardware, str) and hardware.lower() in {"nvidia-t4", "tesla t4", "nvidia tesla t4", "nvidia t4"}, "source receipt must identify T4 hardware")
    _require(doc.get("t4_runtime_digest_definition") ==
             "tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest",
             "T4 runtime digest definition absent or unsupported")
    source_runtime_sha = _field(t4, doc.get("t4_runtime_sha256_field"), "T4 runtime")
    _require(source_runtime_sha == measure_t4_runtime_digest(Path(local["runtime_dir"])),
             "T4 source runtime differs from calibration receiver")
    ratio = seconds / local600
    _require(math.isclose(_number(doc.get("cpu_to_t4_ratio"), "CPU-to-T4 ratio"), ratio, rel_tol=1e-9),
             "CPU-to-T4 ratio must equal T4 seconds / local extrapolated seconds")
    local_host_identity = {key: local[key] for key in ("host", "platform", "hardware_fingerprint") if key in local}
    return {**doc, "measured_t4_seconds": seconds, "local_host_identity": local_host_identity}, ratio


def _candidate_t4(ref: object, runtime_dir: Path, archive_sha: str) -> tuple[float, str]:
    """Consume an exact candidate's completed decode or observed inflate timeout."""
    doc = _receipt(ref)
    _require(doc.get("expected_archive_sha256") == archive_sha, "candidate T4 archive differs")
    runtime_sha = measure_t4_runtime_digest(runtime_dir)
    _require(doc.get("expected_runtime_tree_sha256") == runtime_sha, "candidate T4 runtime differs")
    hardware = _field(doc, ["artifacts", "modal_cuda_preflight.json", "torch_cuda_device_name"], "candidate hardware")
    _require(isinstance(hardware, str) and hardware.lower() in {"nvidia-t4", "tesla t4", "nvidia tesla t4", "nvidia t4"},
             "candidate receipt does not identify T4 hardware")
    if doc.get("passed") is True and type(doc.get("returncode")) is int and doc["returncode"] == 0:
        seconds = _number(_field(doc, ["artifacts", "contest_auth_eval.json", "inflate_elapsed_seconds"],
                                 "candidate decode seconds"), "candidate T4 decode seconds")
        return seconds, "completed"
    _require(doc.get("passed") is False and type(doc.get("returncode")) is int and doc["returncode"] != 0,
             "candidate T4 failure status is missing")
    stderr = _field(doc, ["artifacts", "contest_auth_eval.stderr.log"], "candidate stderr")
    _require(isinstance(stderr, str), "candidate stderr must be text")
    match = re.search(r"ProcessGroupTimeout: Command .*inflate\.sh.* timed out after ([0-9.]+) seconds", stderr)
    _require(match is not None and "[inflate] TIMED OUT" in stderr, "candidate failure was not an observed inflate timeout")
    seconds = _number(float(match.group(1)), "observed inflate timeout")
    configured = _number(_field(doc, ["artifacts", "provenance.json", "inflate_timeout_seconds"],
                               "configured timeout"), "configured inflate timeout")
    _require(seconds == configured, "timeout log disagrees with configured timeout")
    _require(_field(doc, ["artifacts", "provenance.json", "archive_sha256"], "timeout archive") == archive_sha,
             "timeout provenance archive differs")
    _require(_field(doc, ["artifacts", "provenance.json", "inflate_runtime_manifest", "runtime_tree_sha256"],
                    "timeout runtime") == runtime_sha, "timeout provenance runtime differs")
    return seconds, "timeout_lower_bound"


def _cold_public_report(doc: dict) -> dict:
    """The receiver's own report line in the T4 stdout log proves a cold n600 public decode."""
    log = _field(doc, ["artifacts", "contest_auth_eval.stdout.log"], "T4 stdout log")
    _require(isinstance(log, str), "T4 stdout log must be text")
    report = None
    for line in log.splitlines():
        if '"checkpoint_resume"' in line and "{" in line:
            try:
                report = json.loads(line[line.index("{"):])
            except ValueError:
                continue
    _require(isinstance(report, dict), "cold receiver report absent from the T4 stdout log")
    _require(report.get("pair_count") == 600, "cold receiver report must cover 600 pairs")
    _require(report.get("checkpoint_resume") is False, "receiver report shows a checkpoint resume")
    decoder = report.get("token_decoder")
    _require(isinstance(decoder, dict) and decoder.get("checkpoint_resumed_from_frame") == 0,
             "token decoder resumed from a checkpoint")
    cache = report.get("token_cache")
    _require(isinstance(cache, dict) and cache.get("status") == "DISABLED", "token cache was not disabled")
    return report


def _t4_direct(ref: object, runtime_dir: Path, archive_path: Path, leg: dict) -> tuple[float, dict]:
    """ddm_pr11's t4_direct contract: a hash-bound completed cold public-entrypoint T4 decode of the
    exact archive/runtime is the timing authority; no local denominator, no ratio."""
    doc = _receipt(ref)
    for key in ("local_receipt", "calibration_receipt", "cpu_to_t4_ratio", "local_600_seconds", "ratio_definition"):
        _require(key not in leg, f"t4_direct legs carry no {key}")
    for key, path in T4_DIRECT_FIELDS.items():
        _require(leg.get(key) == path, f"{key} must name the canonical T4 source field")
    _require(leg.get("t4_runtime_digest_definition") == "tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest",
             "t4 runtime digest definition absent")
    _require(leg.get("t4_timing_scope") == T4_DIRECT_SCOPE, "t4_direct timing scope must be the cold public entrypoint decode")
    _require(leg.get("runtime_dir") == str(Path(runtime_dir).resolve()) and leg.get("archive_path") == str(Path(archive_path).resolve()),
             "t4_direct leg names a different runtime/archive")
    archive_sha = measure_archive_identity(archive_path).sha256
    _require(leg.get("archive_sha256") == archive_sha, "t4_direct archive differs from the current archive")
    _require(_field(doc, leg["t4_archive_sha256_field"], "T4 archive") == archive_sha, "T4 receipt archive differs from the leg")
    runtime_sha = measure_t4_runtime_digest(runtime_dir)
    _require(leg.get("t4_runtime_sha256") == runtime_sha, "t4_direct runtime projection differs from the current runtime")
    _require(_field(doc, leg["t4_runtime_sha256_field"], "T4 runtime") == runtime_sha, "T4 receipt runtime differs from the leg")
    _require(doc.get("passed") is True and type(doc.get("returncode")) is int and doc["returncode"] == 0,
             "t4_direct requires a completed successful T4 result")
    _require(doc.get("canonical_path") == T4_CANONICAL_PATH, "T4 receipt canonical path is not the contest CUDA path")
    _require(doc.get("inflate_sh_rel") == "inflate.sh", "T4 receipt did not run inflate.sh")
    hardware = _field(doc, leg["t4_hardware_field"], "T4 hardware")
    _require(isinstance(hardware, str) and hardware.lower() in T4_HARDWARE_NAMES, "T4 receipt does not identify T4 hardware")
    _require(_field(doc, leg["t4_cuda_available_field"], "T4 CUDA availability") is True, "T4 receipt reports CUDA unavailable")
    _require(_field(doc, ["artifacts", "contest_auth_eval.json", "n_samples"], "T4 n_samples") == 600, "T4 receipt is not the 600-sample eval")
    report = _cold_public_report(doc)
    seconds = _number(_field(doc, leg["t4_seconds_field"], "T4 decode seconds"), "T4 decode seconds")
    _require(_number(leg.get("measured_t4_decode_seconds"), "measured_t4_decode_seconds") == seconds
             and _number(leg.get("projected_t4_decode_seconds"), "projected_t4_decode_seconds") == seconds,
             "t4_direct seconds must equal the receipt's inflate_elapsed_seconds")
    return seconds, {"cold_report_pairs": report.get("pair_count"), "t4_hardware": hardware}


def build_t4_direct_leg(*, t4_receipt_path: Path, runtime_dir: Path, archive_path: Path) -> dict:
    """Construct ddm_pr11's t4_direct leg from a retained completed T4 receipt; refuses on any drift."""
    runtime_dir, archive_path = Path(runtime_dir).resolve(), Path(archive_path).resolve()
    doc = _receipt(receipt_reference(t4_receipt_path))
    seconds = _number(_field(doc, T4_DIRECT_FIELDS["t4_seconds_field"], "T4 decode seconds"), "T4 decode seconds")
    leg = {"schema": DECODE_WALL_CLOCK_SCHEMA_V2, "mode": "t4_direct", "score_claim": False,
           "candidate_t4_receipt": receipt_reference(t4_receipt_path),
           "runtime_dir": str(runtime_dir), "archive_path": str(archive_path),
           "archive_sha256": measure_archive_identity(archive_path).sha256,
           "receiver_sha256": measure_receiver_digest(runtime_dir),
           "t4_runtime_sha256": measure_t4_runtime_digest(runtime_dir),
           "t4_runtime_digest_definition": "tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest",
           **T4_DIRECT_FIELDS, "t4_timing_scope": T4_DIRECT_SCOPE,
           "measured_t4_decode_seconds": seconds, "projected_t4_decode_seconds": seconds,
           "limit_seconds": LIMIT_SECONDS}
    problems, _ = validate_decode_wall_clock(leg, runtime_dir=runtime_dir, archive_path=archive_path)
    _require(not problems, "; ".join(problems))
    return leg


def build_decode_wall_clock(*, local_receipt_path: Path, calibration_receipt_path: Path,
                            runtime_dir: Path, archive_path: Path, enforce_margin: bool = True,
                            candidate_t4_receipt_path: Path | None = None) -> dict:
    """Construct a measured leg and reject missing, stale, or over-budget evidence."""
    local_ref, calibration_ref = receipt_reference(local_receipt_path), receipt_reference(calibration_receipt_path)
    local, local600 = _local(local_ref, require_margin_basis=enforce_margin)
    _, ratio = _calibration(calibration_ref, require_margin_basis=enforce_margin)
    leg = {"schema": DECODE_WALL_CLOCK_SCHEMA, "mode": "measured", "score_claim": False,
           "local_receipt": local_ref, "calibration_receipt": calibration_ref,
           "receiver_sha256": local["receiver_sha256"], "archive_sha256": local["archive_sha256"],
           "local_600_seconds": local600, "cpu_to_t4_ratio": ratio,
           "ratio_definition": "T4_seconds / local_600_seconds",
           "projected_t4_decode_seconds": local600 * ratio, "limit_seconds": LIMIT_SECONDS}
    if candidate_t4_receipt_path is not None:
        leg["candidate_t4_receipt"] = receipt_reference(candidate_t4_receipt_path)
    problems, _ = validate_decode_wall_clock(leg, runtime_dir=runtime_dir, archive_path=archive_path)
    if enforce_margin:
        _require(not problems, "; ".join(problems))
    else:
        leg["validation_problems_at_construction"] = problems
    return leg


def inherit_decode_wall_clock(*, source_leg_path: Path, runtime_dir: Path, archive_path: Path,
                               pointer_archive_sha256: str) -> dict:
    """Inherit only a valid measured pointer leg whose normalized receiver code is identical."""
    source = _read(Path(source_leg_path))
    leg = {"schema": DECODE_WALL_CLOCK_SCHEMA, "mode": "inherited", "score_claim": False,
           "source_leg": receipt_reference(source_leg_path), "pointer_archive_sha256": pointer_archive_sha256,
           "receiver_sha256": measure_receiver_digest(runtime_dir),
           "archive_sha256": measure_archive_identity(archive_path).sha256,
           "projected_t4_decode_seconds": source.get("projected_t4_decode_seconds"),
           "limit_seconds": LIMIT_SECONDS, "source_mode": source.get("mode"),
           "inheritance_scope": INHERITANCE_SCOPE}
    if source.get("mode") == "t4_direct":
        leg["source_t4_runtime_sha256"] = source.get("t4_runtime_sha256")
        leg["candidate_t4_runtime_sha256"] = measure_t4_runtime_digest(runtime_dir)
    problems, _ = validate_decode_wall_clock(leg, runtime_dir=runtime_dir, archive_path=archive_path,
                                            pointer_archive_sha256=pointer_archive_sha256)
    _require(not problems, "; ".join(problems))
    return leg


def validate_decode_wall_clock(leg: object, *, runtime_dir: Path, archive_path: Path,
                              pointer_archive_sha256: str = "") -> tuple[list[str], dict]:
    """Re-read every receipt and source object; malformed input becomes a typed refusal."""
    observed = {}
    try:
        _require(isinstance(leg, dict), "leg must be an object")
        if leg.get("mode") == "t4_direct":
            _require(leg.get("schema") == DECODE_WALL_CLOCK_SCHEMA_V2, "t4_direct legs use candidate_decode_wall_clock.v2")
        else:
            _require(leg.get("schema") == DECODE_WALL_CLOCK_SCHEMA, "unknown leg schema")
        _require(leg.get("score_claim") is False, "timing cannot claim score authority")
        _require(leg.get("receiver_sha256") == measure_receiver_digest(runtime_dir), "candidate receiver differs from measurement")
        _require(leg.get("archive_sha256") == measure_archive_identity(archive_path).sha256, "candidate archive differs from timing leg")
        if leg.get("mode") == "t4_direct":
            seconds, facts = _t4_direct(leg.get("candidate_t4_receipt"), runtime_dir, archive_path, leg)
            observed.update(measured_t4_decode_seconds=seconds, measured_t4_kind="completed", **facts)
            _require(leg.get("limit_seconds") == LIMIT_SECONDS, "margin limit must be 0.7 * 1800 seconds")
            _require(seconds <= LIMIT_SECONDS, f"measured T4 decode {seconds:.6f}s exceeds {LIMIT_SECONDS}s")
            observed.update(mode="t4_direct", projected_t4_decode_seconds=seconds, limit_seconds=LIMIT_SECONDS)
            return [], observed
        if "candidate_t4_receipt" in leg:
            actual, kind = _candidate_t4(leg["candidate_t4_receipt"], runtime_dir, leg["archive_sha256"])
            observed.update(measured_t4_decode_seconds=actual, measured_t4_kind=kind)
            _require(actual <= LIMIT_SECONDS, f"measured T4 {kind} {actual:.6f}s exceeds {LIMIT_SECONDS}s")
        if leg.get("mode") == "measured":
            calibration_doc, _ = _calibration(leg.get("calibration_receipt"), require_margin_basis=False)
            actual = calibration_doc["measured_t4_seconds"]
            if (calibration_doc["archive_sha256"] == leg.get("archive_sha256")
                    and calibration_doc["receiver_sha256"] == leg.get("receiver_sha256")
                    and calibration_doc.get("t4_timing_scope") == "decode"):
                observed["measured_t4_decode_seconds"] = actual
                _require(actual <= LIMIT_SECONDS, f"measured T4 decode {actual:.6f}s exceeds {LIMIT_SECONDS}s")
        if leg.get("mode") == "inherited":
            source = _receipt(leg.get("source_leg"))
            _require(source.get("mode") in {"measured", "t4_direct"},
                     "inheritance must point directly to a measured or t4_direct leg (never to an inherited one)")
            _require(bool(pointer_archive_sha256) and leg.get("pointer_archive_sha256") == pointer_archive_sha256,
                     "inherited pointer identity absent or stale")
            _require(source.get("archive_sha256") == pointer_archive_sha256, "source measurement is not the pointer archive")
            if source.get("mode") == "t4_direct":
                source_runtime, source_archive = Path(source["runtime_dir"]), Path(source["archive_path"])
                _require(leg.get("source_t4_runtime_sha256") == source.get("t4_runtime_sha256"), "source T4 runtime identity absent")
                _require(leg.get("candidate_t4_runtime_sha256") == measure_t4_runtime_digest(runtime_dir),
                         "candidate T4 runtime identity absent or stale")
            else:
                local, _ = _local(source.get("local_receipt"), require_margin_basis=False)
                source_runtime, source_archive = Path(local["runtime_dir"]), Path(local["archive_path"])
            problems, _ = validate_decode_wall_clock(source, runtime_dir=source_runtime, archive_path=source_archive)
            _require(not problems, "invalid inherited source: " + "; ".join(problems))
            _require(source.get("receiver_sha256") == leg.get("receiver_sha256"), "inherited receiver code differs")
            _require(leg.get("inheritance_scope") == INHERITANCE_SCOPE, "inheritance scope statement absent")
            projected = _number(source.get("projected_t4_decode_seconds"), "source projection")
        else:
            _require(leg.get("mode") == "measured", "unknown timing mode")
            local, local600 = _local(leg.get("local_receipt"))
            calibration, ratio = _calibration(leg.get("calibration_receipt"))
            local_host_identity = {key: local[key] for key in ("host", "platform", "hardware_fingerprint") if key in local}
            _require(local_host_identity == calibration["local_host_identity"],
                     "candidate local host/platform differs from calibration host")
            _require(local["archive_sha256"] == leg.get("archive_sha256") and local["receiver_sha256"] == leg.get("receiver_sha256"),
                     "local receipt differs from timing leg identity")
            _require(math.isclose(_number(leg.get("local_600_seconds"), "local extrapolation"), local600, rel_tol=1e-9), "local extrapolation arithmetic mismatch")
            _require(math.isclose(_number(leg.get("cpu_to_t4_ratio"), "ratio"), ratio, rel_tol=1e-9), "ratio arithmetic mismatch")
            _require(leg.get("ratio_definition") == "T4_seconds / local_600_seconds", "ratio direction absent")
            projected = local600 * ratio
        _require(math.isclose(_number(leg.get("projected_t4_decode_seconds"), "projection"), projected, rel_tol=1e-9), "projection arithmetic mismatch")
        _require(leg.get("limit_seconds") == LIMIT_SECONDS, "margin limit must be 0.7 * 1800 seconds")
        _require(projected <= LIMIT_SECONDS, f"projected T4 decode {projected:.6f}s exceeds {LIMIT_SECONDS}s")
        observed.update(mode=leg["mode"], projected_t4_decode_seconds=projected, limit_seconds=LIMIT_SECONDS)
    except (SealContractError, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        return [str(exc)], observed
    return [], observed
