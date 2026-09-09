"""Fail-closed bindings for research instruments (ddm_pm2, catalog 415).

These gates do not calibrate or change measured values. Waivers and the legacy
process escape are loud, typed events; callers must retain the returned receipt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import re
import sys
import tokenize
from pathlib import Path

from tac.candidate_seal import SealContractError, measure_runtime_digest

REPO = Path(__file__).resolve().parents[2]
POINTER = REPO / ".omx/state/canonical_frontier_pointer.json"
DIGEST_DEFINITION = "tac.candidate_seal.measure_runtime_digest"


class ConfoundAlarm(ValueError):
    """A refused measurement, suitable for an uncaught nonzero CLI exit."""

    def __init__(self, kind: str, message: str, **details):
        self.record = dict(event="confound_alarm", kind=kind, message=message, **details)
        super().__init__(json.dumps(self.record, sort_keys=True))
        print(str(self), file=sys.stderr, flush=True)


def _event(event_name: str, **details) -> dict:
    record = dict(event=event_name, **details)
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)
    return record


def gates_disabled() -> bool:
    return os.environ.get("TAC_INSTRUMENT_GATES", "1") == "0"


def _escape() -> dict | None:
    if gates_disabled():
        return _event("instrument_gate_bypass", reason="TAC_INSTRUMENT_GATES=0", valid=False)
    return None


def _rationale(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    words = re.findall(r"[a-z]+", value.lower())
    if (
        len(value) < 16
        or len(words) < 3
        or re.search(r"\b(todo|tbd|placeholder|unknown|fixme)\b|<[^>]*>", value, re.I)
        or len(set(words)) < 3
    ):
        raise ConfoundAlarm("invalid_waiver", "Provide a concrete, non-placeholder rationale")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            raise ValueError("expected object")
        return obj
    except (OSError, ValueError) as exc:
        raise ConfoundAlarm("pointer_custody", f"Cannot read {path}: {exc}") from exc


def _positive(value, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfoundAlarm("invalid_pose", f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ConfoundAlarm("invalid_pose", f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def snapshot_pose_pointer(pointer_path: Path | None = None) -> dict:
    """Bind the CUDA baseline to its own source receipt before measuring."""
    bypass = _escape()
    if bypass:
        return bypass
    path = Path(pointer_path or POINTER)
    before = _sha(path) if path.is_file() else None
    pointer = _json(path)
    try:
        node = pointer["our_local_frontier_contest_cuda"]
        source = Path(node["source_path"])
        source = source if source.is_absolute() else REPO / source
        mirror = _json(source)
        archive_sha = node["archive_sha256"]
        if mirror["archive_sha256"] != archive_sha:
            raise ValueError("component receipt belongs to a different archive")
        receipt_path = Path(mirror["source_receipt"])
        receipt_sha = _sha(receipt_path)
        if receipt_sha != mirror["source_receipt_sha256"]:
            raise ValueError("promoted source receipt hash changed")
        if receipt_sha != node["extra"]["source_receipt_sha256"]:
            raise ValueError("source receipt differs from the pointer's pinned receipt")
        receipt = _json(receipt_path)
        if receipt["expected_archive_sha256"] != archive_sha or receipt["n_samples"] != 600:
            raise ValueError("source receipt is not this n600 archive")
        if receipt.get("passed") is not True:
            raise ValueError("source evaluation did not pass")
        if receipt["expected_runtime_tree_sha256"] != node["extra"]["runtime_tree_sha256"]:
            raise ValueError("source receipt belongs to a different runtime")
        d_pose = _positive(receipt["avg_posenet_dist"], "pointer d_pose", zero=True)
        if _sha(path) != before:
            raise ValueError("pointer moved while reading its receipt")
        return {
            "pointer_path": str(path),
            "pointer_sha256": before,
            "archive_sha256": archive_sha,
            "d_pose": d_pose,
            "receipt_path": str(receipt_path),
            "source_receipt_sha256": _sha(receipt_path),
            "modal_runtime_sha256": node["extra"]["runtime_tree_sha256"],
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        if isinstance(exc, ConfoundAlarm):
            raise
        raise ConfoundAlarm("pointer_custody", str(exc)) from exc


def _band(measured: float, reference: float, rationale: str | None, kind: str) -> dict:
    measured = _positive(measured, "measured d_pose", zero=True)
    reference = _positive(reference, "reference d_pose", zero=True)
    reason = _rationale(rationale)
    ratio = measured / reference if reference else (1.0 if measured == 0 else None)
    valid = ratio is not None and 1 / 3 <= ratio <= 3
    record = {"kind": kind, "measured": measured, "reference": reference, "ratio": ratio, "valid": valid}
    if not valid:
        message = "Pose base outside [1/3, 3]: likely missing --overlay or wrong tree"
        if reason is None:
            raise ConfoundAlarm(kind, message, **{k: v for k, v in record.items() if k != "kind"})
        return _event("instrument_gate_waiver", **record, rationale=reason)
    return dict(event="instrument_gate_pass", **record)


def check_pose_base(
    d_pose: float, *, rationale: str | None = None, pointer_path: Path | None = None, snapshot: dict | None = None
) -> dict:
    bypass = _escape()
    if bypass:
        return bypass
    reference = snapshot if snapshot is not None else snapshot_pose_pointer(pointer_path)
    try:
        if _sha(Path(reference["pointer_path"])) != reference["pointer_sha256"]:
            raise ConfoundAlarm("pointer_moved", "Pointer moved during base measurement; remeasure")
    except (KeyError, OSError) as exc:
        raise ConfoundAlarm("pointer_custody", f"Invalid pose snapshot: {exc}") from exc
    return {**_band(d_pose, reference["d_pose"], rationale, "pose_base_magnitude"), "pointer": reference}


def check_pose_pair(d_pose: float, reference: float, *, rationale: str | None = None) -> dict:
    return _escape() or _band(d_pose, reference, rationale, "pose_pair_magnitude")


def add_pose_gate_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pose-base-differs-because", default=None, help="explicit rationale for a pose base outside the pointer band"
    )


def add_coder_gate_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--coder-differs-because", default=None, help="explicit rationale for intentionally pricing a different coder"
    )


def check_pointer_coder(runtime_root: Path, *, rationale: str | None = None, pointer_path: Path | None = None) -> dict:
    """Resolve custody through the pointer receipt, never through a guessed tree."""
    bypass = _escape()
    if bypass:
        return bypass
    reason = _rationale(rationale)
    snapshot = snapshot_pose_pointer(pointer_path)
    manifest = _json(Path(snapshot["receipt_path"]).parent / "FIRE_MANIFEST.json")
    try:
        live_root = Path(manifest["runtime_dir"])
        if _sha(live_root / "archive.zip") != snapshot["archive_sha256"]:
            raise ValueError("live tree archive does not match the pointer")
        digests = manifest["stage3_runtime_digests"]
        seal = digests["seal_runtime"]
        modal = digests["modal_uploaded_runtime"]
        if (
            seal["digest_definition"] != DIGEST_DEFINITION
            or modal["runtime_tree_sha256"] != snapshot["modal_runtime_sha256"]
        ):
            raise ValueError("fire manifest is not bound to the pointer's runtime")
        live = measure_runtime_digest(live_root)
        if live.sha256 != seal["sha256"]:
            raise ValueError("pointer runtime tree drifted after promotion")
        encoder = measure_runtime_digest(Path(runtime_root))
        if not encoder.file_count:
            raise ValueError("encoder runtime tree is empty")
        if _sha(Path(snapshot["pointer_path"])) != snapshot["pointer_sha256"]:
            raise ValueError("pointer moved while hashing coder trees")
    except (KeyError, TypeError, ValueError, OSError, SealContractError) as exc:
        raise ConfoundAlarm("coder_custody", str(exc)) from exc
    record = {
        "runtime_root": str(runtime_root),
        "pointer_runtime_root": str(live_root),
        "encoder_sha256": encoder.sha256,
        "pointer_sha256": live.sha256,
        "digest_definition": DIGEST_DEFINITION,
        "valid": encoder.sha256 == live.sha256,
    }
    if not record["valid"]:
        if reason is None:
            raise ConfoundAlarm(
                "encoder_not_pointer_coder", "Wrong encoder tree; rebase or use --coder-differs-because", **record
            )
        return _event("instrument_gate_waiver", **record, rationale=reason)
    return dict(event="instrument_gate_pass", **record)


def check_experimental_coder(runtime_root: Path, *, rationale: str | None = None) -> dict:
    """New-codec CLIs must declare that the repository encoder is experimental.

    Checking only their source decoder tree would falsely certify a new encoder
    imported from experiments/. This explicit door covers that distinct shape.
    """
    bypass = _escape()
    if bypass:
        return bypass
    reason = _rationale(rationale)
    if reason is None:
        raise ConfoundAlarm("experimental_coder", "New codec pricing requires --coder-differs-because")
    source = check_pointer_coder(runtime_root, rationale=reason)
    return _event(
        "instrument_gate_waiver", valid=False, rationale=reason, kind="experimental_coder", source_binding=source
    )


POSE_BINDINGS = {
    "ddm_sj1_joint_admission.py": {"cmd_pose": "check_pose_base"},
    "ddm_rp1_pose.py": {"cmd_pose": "check_pose_base"},
    "ddm_fe1_admit_and_build.py": {
        "cmd_base_pose": "check_pose_base", "cmd_pose": "evaluate_base_codes"},
    "ddm_fe1_pose_price.py": {"cmd_price": "evaluate_base_codes"},
    "ddm_fe1_rebase.py": {"cmd_rebase": "evaluate_base_codes"},
    "ddm_fe1_cmp1_rebuild.py": {"main": "evaluate_base_codes"},
    "ddm_fe1_pass4_race.py": {"main": "evaluate_base_codes"},
    "ddm_rw1_renderer_edge_foldback.py": {"cmd_pose": "evaluate_base_codes"},
}
CODER_CLIS = (
    "ddm_tc1_mixer_pricing.py", "ddm_sm1_mixer_race.py",
    "ddm_rc3_shared_mixer_pricing.py", "ddm_rc3_shared_mixer_race.py",
)


def _call_name(node: ast.Call) -> str:
    return node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")


def _active_calls(node: ast.AST):
    """Ignore unused nested definitions and literal dead branches in the guard audit."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.If) and isinstance(child.test, ast.Constant):
            for statement in child.body if child.test.value else child.orelse:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    yield statement.value
                yield from _active_calls(statement)
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _active_calls(child)


def audit_instrument_gate_wiring(root: Path) -> list[str]:
    """Static deletion/order guard for declared current-vehicle entrypoints.

    Historical instruments retain their own vehicle baselines. Behavioral tests
    separately exercise refusal before persistence and compilation. This finite
    syntactic audit does not claim arbitrary Python control/dataflow coverage.
    """
    findings = []
    bindings = {**POSE_BINDINGS, "ddm_jg2_tail_reencode.py": {"_prepare": "check_pointer_coder"}}
    for name, functions in bindings.items():
        path = Path(root) / "experiments" / name
        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (OSError, SyntaxError) as exc:
            findings.append(f"{path}: cannot inspect instrument: {exc}")
            continue
        for function, guard in functions.items():
            node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function), None)
            if node is None:
                findings.append(f"{path}:{function}: missing function")
                continue
            comments = {t.start[0]: t.string for t in tokenize.generate_tokens(io.StringIO(source).readline)
                        if t.type == tokenize.COMMENT}
            line = comments.get(node.lineno, "")
            if "INSTRUMENT_BINDING_OK:" in line:
                try:
                    _rationale(line.split("INSTRUMENT_BINDING_OK:", 1)[1])
                    continue
                except ConfoundAlarm:
                    pass
            calls = sorted(_active_calls(node), key=lambda n: n.lineno)
            guards = [n for n in calls if _call_name(n) == guard]
            dangerous = (("compile_rc64", "load_route_b", "load_runtime") if guard == "check_pointer_coder"
                         else ("save", "write_text") if guard == "check_pose_base" else ())
            dangers = [n for n in calls if _call_name(n) in dangerous]
            if not guards or (dangers and guards[0].lineno >= dangers[0].lineno):
                findings.append(f"{path}:{function}: {guard} missing or after persistence/encode setup")
    for name in CODER_CLIS:
        path = Path(root) / "experiments" / name
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError) as exc:
            findings.append(f"{path}: cannot inspect pricing CLI: {exc}")
            continue
        mains = [n for n in tree.body if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)]
        calls = [n for block in mains for n in _active_calls(block)]
        guarded = [n for n in calls if _call_name(n) == "check_experimental_coder"]
        work = [n for n in calls if _call_name(n) in {"race", "run", "encode", "quantize", "supported_contexts", "globals"}]
        if not guarded or (work and guarded[0].lineno >= min(n.lineno for n in work)):
            findings.append(f"{path}: experimental coder gate missing or after pricing dispatch")
    return findings
