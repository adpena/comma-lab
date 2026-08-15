#!/usr/bin/env python3
"""Compose and optionally fire a watched HPAC continuation without hand-written argv.

The first supported family is the sealed RX2/WC2 HPAC wrapper.  The reference
trainer remains read-only.  Environment gates are extracted from its AST at
composition time, resource limits and watcher policy come from the parent
launch manifest, and the exact detached commands are sealed to ``launch.sh``
before either the endpoint closer or trainer is started.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TRAINER = REPO_ROOT / "tools/train_ddm_cl1_hpac_capacity.py"
SEALED_WRAPPER = REPO_ROOT / "tools/train_ddm_cl1_hpac_capacity_mps.py"
LAUNCHER = REPO_ROOT / "tools/launch_detached_process.py"
LOCAL_CLOSER = REPO_ROOT / "tools/local_endpoint_close.py"
HOT_STATE = REPO_ROOT / ".omx/state/main_hot_state.md"
FRONTIER_POINTER = REPO_ROOT / ".omx/state/canonical_frontier_pointer.json"

COMPOSITION_SCHEMA = "hpac_watched_continuation_composition.v1"
VERIFICATION_SCHEMA = "hpac_continuation_gate_chain_verification.v1"
HOT_STATE_SCHEMA = "main_hot_state_pointer_staleness_preflight.v1"
DONE_SCHEMA = "detached_local_process_done.v2"
_CHECKPOINT_RE = re.compile(r"qat_stage_end_epoch_(\d+)\.pt\Z")
_RECEIPT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_PROTECTED_TRAINER_FLAGS = frozenset(
    {
        "--device",
        "--epochs",
        "--out",
        "--profile",
        "--resume-from",
        "--save",
        "--stop-after-epoch",
    }
)


class ContinuationCompositionError(RuntimeError):
    """A continuation cannot be composed without weakening a sealed gate."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_text(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    if executable:
        tmp.chmod(0o755)
    tmp.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, _canonical_json(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuationCompositionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuationCompositionError(f"JSON root is not an object: {path}")
    return value


def _raises_training_error(node: ast.If) -> bool:
    for child in node.body:
        for nested in ast.walk(child):
            if not isinstance(nested, ast.Raise) or nested.exc is None:
                continue
            call = nested.exc
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Name) and func.id.endswith("TrainingError"):
                return True
    return False


def _env_get_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
            and func.value.attr == "environ"
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "getenv"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and node.value.attr == "environ"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


def parse_required_environment_gates(source_path: Path) -> dict[str, str]:
    """Extract ``env != required`` fail-closed raises from the trainer source."""

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    gates: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _raises_training_error(node):
            continue
        for compare in (part for part in ast.walk(node.test) if isinstance(part, ast.Compare)):
            if len(compare.ops) != 1 or not isinstance(compare.ops[0], ast.NotEq):
                continue
            left_key = _env_get_key(compare.left)
            right = compare.comparators[0]
            right_key = _env_get_key(right)
            required: str | None = None
            key: str | None = None
            if left_key and isinstance(right, ast.Constant) and isinstance(right.value, str):
                key, required = left_key, right.value
            elif right_key and isinstance(compare.left, ast.Constant) and isinstance(compare.left.value, str):
                key, required = right_key, compare.left.value
            if key is None or required is None:
                continue
            prior = gates.get(key)
            if prior is not None and prior != required:
                raise ContinuationCompositionError(
                    f"reference trainer has conflicting required values for {key}: {prior!r}, {required!r}"
                )
            gates[key] = required
    if not gates:
        raise ContinuationCompositionError(
            f"no fail-closed environment gates were parsed from {source_path}"
        )
    return gates


def parse_port_modes(wrapper_path: Path) -> dict[str, dict[str, Any]]:
    tree = ast.parse(wrapper_path.read_text(encoding="utf-8"), filename=str(wrapper_path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "PORT_MODES" for target in targets):
            continue
        raw = ast.literal_eval(node.value)
        if not isinstance(raw, dict) or not all(
            isinstance(key, str) and isinstance(value, dict) for key, value in raw.items()
        ):
            break
        return raw
    raise ContinuationCompositionError(f"cannot parse literal PORT_MODES from {wrapper_path}")


def parse_reference_cli_specs(source_path: Path) -> dict[str, bool]:
    """Return ``flag -> takes_value`` from literal argparse declarations."""

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    specs: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        flags = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        ]
        action = next(
            (
                kw.value.value
                for kw in node.keywords
                if kw.arg == "action"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ),
            "",
        )
        takes_value = action not in {"store_true", "store_false", "count", "help", "version"}
        for flag in flags:
            specs[flag] = takes_value
    if not specs:
        raise ContinuationCompositionError(f"cannot parse argparse flags from {source_path}")
    return specs


def _option_index(argv: Sequence[str], flag: str) -> int | None:
    for index, token in enumerate(argv):
        if token == flag or token.startswith(flag + "="):
            return index
    return None


def _option_value(argv: Sequence[str], flag: str) -> str:
    index = _option_index(argv, flag)
    if index is None:
        raise ContinuationCompositionError(f"parent argv lacks required option {flag}")
    token = argv[index]
    if token.startswith(flag + "="):
        return token.split("=", 1)[1]
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ContinuationCompositionError(f"parent argv option {flag} lacks a value")
    return argv[index + 1]


def _remove_option(argv: list[str], flag: str, *, takes_value: bool) -> None:
    while (index := _option_index(argv, flag)) is not None:
        token = argv[index]
        del argv[index]
        if token == flag and takes_value:
            if index >= len(argv):
                raise ContinuationCompositionError(f"option {flag} lacks a value")
            del argv[index]


def _set_option(argv: list[str], flag: str, value: str | None, *, takes_value: bool) -> None:
    _remove_option(argv, flag, takes_value=takes_value)
    argv.append(flag)
    if takes_value:
        if value is None:
            raise ContinuationCompositionError(f"option {flag} requires a value")
        argv.append(value)
    elif value is not None:
        raise ContinuationCompositionError(f"boolean option {flag} does not accept a value")


def _apply_overrides(
    argv: list[str], overrides: Sequence[str], cli_specs: Mapping[str, bool]
) -> None:
    for raw in overrides:
        flag, separator, value = raw.partition("=")
        if not flag.startswith("--"):
            raise ContinuationCompositionError(
                f"override must be --flag or --flag=value, got {raw!r}"
            )
        if flag in _PROTECTED_TRAINER_FLAGS:
            raise ContinuationCompositionError(f"override would violate sealed continuation field {flag}")
        if flag not in cli_specs:
            raise ContinuationCompositionError(f"override names no reference-trainer flag: {flag}")
        takes_value = cli_specs[flag]
        _set_option(argv, flag, value if separator else None, takes_value=takes_value)


def _rewrite_tree(value: Any, parent_root: Path, run_root: Path) -> Any:
    if isinstance(value, str):
        if str(parent_root) in value:
            return value.replace(str(parent_root), str(run_root))
        return value.replace(parent_root.name, run_root.name)
    if isinstance(value, list):
        return [_rewrite_tree(item, parent_root, run_root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_tree(item, parent_root, run_root) for key, item in value.items()}
    return value


def _watcher_config_path(manifest: Mapping[str, Any], kind: str) -> Path:
    watchers = manifest.get("watchers")
    if not isinstance(watchers, list):
        raise ContinuationCompositionError("parent launch manifest has no watcher list")
    matches = [row for row in watchers if isinstance(row, dict) and row.get("kind") == kind]
    if len(matches) != 1 or not isinstance(matches[0].get("config_path"), str):
        raise ContinuationCompositionError(f"parent manifest must name exactly one {kind} watcher")
    path = Path(matches[0]["config_path"]).expanduser().resolve(strict=False)
    if not path.is_file():
        raise ContinuationCompositionError(f"parent {kind} watcher config is absent: {path}")
    return path


def _parent_endpoint(log_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "{" not in line or "estimated_joint_bytes" not in line:
            continue
        try:
            row = json.loads(line[line.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, dict) and "epoch" in row and "estimated_joint_bytes" in row:
            try:
                normalized = {**row, "epoch": int(row["epoch"]), "estimated_joint_bytes": int(row["estimated_joint_bytes"])}
            except (TypeError, ValueError):
                continue
            rows.append(normalized)
    if not rows:
        raise ContinuationCompositionError(f"parent log has no joint-byte endpoint row: {log_path}")
    max_epoch = max(candidate["epoch"] for candidate in rows)
    row = [candidate for candidate in rows if candidate["epoch"] == max_epoch][-1]
    try:
        return {"epoch": row["epoch"], "joint_bytes": row["estimated_joint_bytes"], "row": row}
    except (TypeError, ValueError) as exc:
        raise ContinuationCompositionError("parent endpoint epoch/joint bytes are not integers") from exc


def locate_resume_checkpoint(parent_argv: Sequence[str], continuation_of_epochs: int) -> Path:
    save = Path(_option_value(parent_argv, "--save")).expanduser().resolve(strict=False)
    checkpoint_root = save.with_name(save.stem + ".checkpoints")
    matches: list[tuple[int, Path]] = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.rglob("qat_stage_end_epoch_*.pt"):
            match = _CHECKPOINT_RE.fullmatch(path.name)
            if match and path.is_file():
                matches.append((int(match.group(1)), path.resolve()))
    if not matches:
        raise ContinuationCompositionError(
            f"no qat_stage_end_epoch_*.pt exists under parent checkpoint root {checkpoint_root}"
        )
    newest_epoch = max(epoch for epoch, _ in matches)
    newest = [path for epoch, path in matches if epoch == newest_epoch]
    if len(newest) != 1:
        raise ContinuationCompositionError(
            f"ambiguous newest QAT checkpoint epoch {newest_epoch}: {[str(path) for path in newest]}"
        )
    if newest_epoch != continuation_of_epochs:
        raise ContinuationCompositionError(
            f"newest QAT checkpoint epoch {newest_epoch} != sealed continuation_of_epochs "
            f"{continuation_of_epochs}"
        )
    return newest[0]


def check_hot_state_staleness(
    hot_state_path: Path = HOT_STATE,
    frontier_pointer_path: Path = FRONTIER_POINTER,
) -> dict[str, Any]:
    """Warn-only comparison; unreadable state is loud but never a launch refusal."""

    receipt: dict[str, Any] = {
        "schema": HOT_STATE_SCHEMA,
        "mode": "WARN_ONLY",
        "hot_state_path": str(hot_state_path),
        "frontier_pointer_path": str(frontier_pointer_path),
        "status": "WARN_UNREADABLE",
        "warning": "",
    }
    try:
        text = hot_state_path.read_text(encoding="utf-8")
        section = text.split("## POINTER_LINE", 1)[1].split("\n## ", 1)[0]
        score_match = re.search(r"\bS\s+([0-9]+(?:\.[0-9]+)?)", section)
        if score_match is None:
            raise ValueError("POINTER_LINE has no S value")
        hot_score = float(score_match.group(1))
        pointer = _read_json(frontier_pointer_path)
        canonical_score = float((pointer.get("effective_frontier") or {})["score"])
        receipt.update({"hot_state_score": hot_score, "canonical_score": canonical_score})
        if math.isclose(hot_score, canonical_score, rel_tol=0.0, abs_tol=1e-15):
            receipt["status"] = "CURRENT"
        else:
            receipt["status"] = "WARN_STALE"
            receipt["warning"] = (
                f"main_hot_state POINTER_LINE S={hot_score:.17g} differs from canonical "
                f"effective_frontier={canonical_score:.17g}"
            )
    except (KeyError, IndexError, OSError, TypeError, ValueError, ContinuationCompositionError) as exc:
        receipt["warning"] = f"hot-state staleness comparison unavailable: {type(exc).__name__}: {exc}"
    return receipt


def _derived_run_root(parent_run_dir: Path, port_mode: str) -> Path:
    suffix = port_mode.removeprefix("full-mps-").replace("-", "_")
    return parent_run_dir.parent / f"{parent_run_dir.name}_{suffix}"


def _receipt_name(run_root: Path) -> str:
    raw = _RECEIPT_SAFE_RE.sub("_", f"hpac_continuation_{run_root.name}").strip("._-")
    if not raw:
        raise ContinuationCompositionError("run root cannot form a safe done-receipt name")
    return raw[:128]


def _validate_resource_budget(manifest: Mapping[str, Any]) -> dict[str, Any]:
    budget = manifest.get("resource_budget")
    if not isinstance(budget, dict):
        raise ContinuationCompositionError("parent launch manifest has no resource_budget object")
    required = ("measured_peak_rss_gib", "measured_thread_need", "walltime_cap_s")
    if any(key not in budget for key in required):
        raise ContinuationCompositionError(f"parent resource budget lacks one of {required}")
    requested_nice = manifest.get("requested_nice")
    if requested_nice is None:
        requested_nice = manifest.get("actual_nice")
    if requested_nice is None:
        requested_nice = 10
    peak_rss_gib = float(budget["measured_peak_rss_gib"])
    thread_need = int(budget["measured_thread_need"])
    walltime_cap_s = float(budget["walltime_cap_s"])
    if not math.isfinite(peak_rss_gib) or peak_rss_gib <= 0:
        raise ContinuationCompositionError("parent measured_peak_rss_gib must be finite and positive")
    if thread_need < 1:
        raise ContinuationCompositionError("parent measured_thread_need must be positive")
    if not math.isfinite(walltime_cap_s) or walltime_cap_s <= 0:
        raise ContinuationCompositionError("parent walltime_cap_s must be finite and positive")
    return {
        "measured_peak_rss_gib": peak_rss_gib,
        "measured_thread_need": thread_need,
        "walltime_cap_s": walltime_cap_s,
        "requested_nice": int(requested_nice),
    }


def _build_continuation_argv(
    parent_argv: Sequence[str],
    *,
    port_mode: str,
    mode: Mapping[str, Any],
    resume_checkpoint: Path,
    run_root: Path,
    overrides: Sequence[str],
) -> list[str]:
    argv = list(parent_argv)
    if len(argv) < 2 or Path(argv[1]).name != SEALED_WRAPPER.name:
        raise ContinuationCompositionError(
            f"parent argv is not the sealed wrapper {SEALED_WRAPPER.name}: {argv[:2]}"
        )
    specs = parse_reference_cli_specs(REFERENCE_TRAINER)
    _set_option(argv, "--port-mode", port_mode, takes_value=True)
    _set_option(argv, "--device", str(mode["device"]), takes_value=True)
    _set_option(argv, "--epochs", str(int(mode["epochs"])), takes_value=True)
    _set_option(argv, "--resume-from", str(resume_checkpoint), takes_value=True)
    _set_option(argv, "--resume-allow-trainer-drift", None, takes_value=False)
    save = run_root / "checkpoints" / f"{port_mode.replace('-', '_')}.pt"
    out = run_root / "reports/trainer.json"
    _set_option(argv, "--save", str(save), takes_value=True)
    _set_option(argv, "--out", str(out), takes_value=True)
    _apply_overrides(argv, overrides, specs)
    return argv


def _launcher_base(
    *,
    output_dir: Path,
    purpose: str,
    authority: str,
    done_receipt: str,
    nice: int,
    peak_rss_gib: float,
    thread_need: int,
    walltime_cap_s: float,
) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/launch_detached_process.py",
        "--output-dir",
        str(output_dir),
        "--cwd",
        str(REPO_ROOT),
        "--purpose",
        purpose,
        "--authority",
        authority,
        "--nice",
        str(nice),
        "--derive-resource-budgets",
        "--measured-peak-rss-gib",
        str(peak_rss_gib),
        "--measured-thread-need",
        str(thread_need),
        "--walltime-cap-s",
        str(walltime_cap_s),
        "--done-receipt",
        done_receipt,
        "--verify-alive-secs",
        "3",
    ]


def _launcher_environment(argv: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, token in enumerate(argv):
        if token != "--env":
            continue
        if index + 1 >= len(argv) or "=" not in argv[index + 1]:
            raise ContinuationCompositionError("launcher --env entry is malformed")
        key, value = argv[index + 1].split("=", 1)
        if key in values and values[key] != value:
            raise ContinuationCompositionError(f"launcher repeats conflicting environment gate {key}")
        values[key] = value
    return values


def compose_continuation(
    *,
    parent_run_dir: Path,
    port_mode: str,
    run_root: Path | None,
    overrides: Sequence[str],
    done_receipt: str | None,
    closer_deadline_s: float,
    closer_poll_s: float,
) -> dict[str, Any]:
    parent_run_dir = parent_run_dir.expanduser().resolve(strict=True)
    modes = parse_port_modes(SEALED_WRAPPER)
    mode = modes.get(port_mode)
    if mode is None:
        raise ContinuationCompositionError(f"unknown sealed wrapper mode {port_mode!r}")
    if not mode.get("resume_required") or "continuation_of_epochs" not in mode:
        raise ContinuationCompositionError(f"sealed wrapper mode is not a continuation: {port_mode}")
    run_root = (
        run_root.expanduser().resolve(strict=False)
        if run_root is not None
        else _derived_run_root(parent_run_dir, port_mode).resolve(strict=False)
    )
    if (
        run_root == parent_run_dir
        or parent_run_dir in run_root.parents
        or run_root in parent_run_dir.parents
    ):
        raise ContinuationCompositionError(
            "continuation run root and parent run must be disjoint, not equal or nested"
        )
    parent_manifest_path = parent_run_dir / "launcher/launch_manifest.json"
    parent_manifest = _read_json(parent_manifest_path)
    if parent_manifest.get("schema") != "detached_local_process_launch.v2":
        raise ContinuationCompositionError("parent launch manifest schema differs")
    parent_argv = parent_manifest.get("argv")
    if not isinstance(parent_argv, list) or not all(isinstance(item, str) for item in parent_argv):
        raise ContinuationCompositionError("parent manifest argv is malformed")
    continuation_of_epochs = int(mode["continuation_of_epochs"])
    resume_checkpoint = locate_resume_checkpoint(parent_argv, continuation_of_epochs)
    endpoint = _parent_endpoint(parent_run_dir / "launcher/run.log")
    if endpoint["epoch"] != continuation_of_epochs:
        raise ContinuationCompositionError(
            f"parent endpoint epoch {endpoint['epoch']} != continuation boundary {continuation_of_epochs}"
        )
    budget = _validate_resource_budget(parent_manifest)
    env_gates = parse_required_environment_gates(REFERENCE_TRAINER)
    trainer_argv = _build_continuation_argv(
        parent_argv,
        port_mode=port_mode,
        mode=mode,
        resume_checkpoint=resume_checkpoint,
        run_root=run_root,
        overrides=overrides,
    )
    for path in (
        Path(_option_value(trainer_argv, "--save")),
        Path(_option_value(trainer_argv, "--out")),
        Path(_option_value(trainer_argv, "--save")).with_name(
            Path(_option_value(trainer_argv, "--save")).stem + ".checkpoints"
        ),
    ):
        if path.exists():
            raise ContinuationCompositionError(f"continuation output already exists: {path}")

    launcher_dir = run_root / "launcher"
    liveness_path = launcher_dir / "generated_liveness.json"
    quality_path = launcher_dir / "generated_quality.json"
    liveness = _rewrite_tree(
        _read_json(_watcher_config_path(parent_manifest, "liveness")), parent_run_dir, run_root
    )
    quality = _rewrite_tree(
        _read_json(_watcher_config_path(parent_manifest, "quality")), parent_run_dir, run_root
    )
    quality["bar_value"] = endpoint["joint_bytes"]
    quality["bar_start_epoch"] = continuation_of_epochs + 1
    if isinstance(quality.get("phase_knee"), dict):
        quality["phase_knee"]["epoch"] = continuation_of_epochs + 1
    _atomic_json(liveness_path, liveness)
    _atomic_json(quality_path, quality)

    receipt_name = done_receipt or _receipt_name(run_root)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", receipt_name):
        raise ContinuationCompositionError(f"invalid done receipt name {receipt_name!r}")
    training_done_path = REPO_ROOT / f".omx/tmp/codex_runs/{receipt_name}.done"
    closer_suffix = ".endpoint-close"
    closer_receipt_name = receipt_name[: 128 - len(closer_suffix)] + closer_suffix
    closer_done_path = REPO_ROOT / f".omx/tmp/codex_runs/{closer_receipt_name}.done"
    for path in (training_done_path, closer_done_path):
        if path.exists() or path.with_name(path.name + ".armed.json").exists():
            raise ContinuationCompositionError(f"done-receipt identity is already active or complete: {path}")

    closer_output = run_root / "endpoint_closure"
    closer_cmd = _launcher_base(
        output_dir=closer_output / "launcher",
        purpose=f"armed local endpoint closer for {run_root.name}",
        authority="scorer-free local endpoint closure; never launches paid or scorer work",
        done_receipt=closer_receipt_name,
        nice=budget["requested_nice"],
        peak_rss_gib=1.0,
        thread_need=min(2, budget["measured_thread_need"]),
        walltime_cap_s=closer_deadline_s + 1800,
    )
    closer_cmd.extend(
        [
            "--",
            ".venv/bin/python",
            "tools/local_endpoint_close.py",
            "--run-root",
            str(run_root),
            "--done-receipt-path",
            str(training_done_path),
            "--output-dir",
            str(closer_output),
            "--deadline-s",
            str(closer_deadline_s),
            "--poll-s",
            str(closer_poll_s),
        ]
    )

    training_cmd = _launcher_base(
        output_dir=launcher_dir,
        purpose=f"watched sealed HPAC continuation {port_mode} from {parent_run_dir.name}",
        authority="local MPS training research signal; CPU identity race remains serialization authority",
        done_receipt=receipt_name,
        nice=budget["requested_nice"],
        peak_rss_gib=budget["measured_peak_rss_gib"],
        thread_need=budget["measured_thread_need"],
        walltime_cap_s=budget["walltime_cap_s"],
    )
    for key, value in env_gates.items():
        training_cmd.extend(["--env", f"{key}={value}"])
    training_cmd.extend(
        [
            "--arm-watchers",
            "--liveness-config",
            str(liveness_path),
            "--quality-config",
            str(quality_path),
            "--",
            *trainer_argv,
        ]
    )
    launcher_env = _launcher_environment(training_cmd)
    missing_or_drifted = {
        key: {"required": value, "launcher": launcher_env.get(key)}
        for key, value in env_gates.items()
        if launcher_env.get(key) != value
    }
    if missing_or_drifted:
        raise ContinuationCompositionError(
            f"source-parsed environment gates were not sealed in one pass: {missing_or_drifted}"
        )

    launch_script = launcher_dir / "launch.sh"
    script = (
        "#!/bin/sh\n"
        "set -eu\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        f"{shlex.join(closer_cmd)}\n"
        f"{shlex.join(training_cmd)}\n"
    )
    _atomic_text(launch_script, script, executable=True)
    hot_state = check_hot_state_staleness()
    composition: dict[str, Any] = {
        "schema": COMPOSITION_SCHEMA,
        "generated_utc": _utc_now(),
        "parent_run_dir": str(parent_run_dir),
        "parent_manifest": _file_record(parent_manifest_path),
        "run_root": str(run_root),
        "port_mode": port_mode,
        "sealed_mode": mode,
        "reference_trainer": _file_record(REFERENCE_TRAINER),
        "sealed_wrapper": _file_record(SEALED_WRAPPER),
        "launcher": _file_record(LAUNCHER),
        "local_closer": _file_record(LOCAL_CLOSER),
        "required_environment_gates_from_source": env_gates,
        "environment_gate_verification": {
            "status": "PASS",
            "source_required": env_gates,
            "launcher_assignments": {key: launcher_env[key] for key in env_gates},
        },
        "resource_budget_from_parent": budget,
        "resume_checkpoint": _file_record(resume_checkpoint),
        "parent_endpoint": endpoint,
        "watcher_configs": {
            "liveness": _file_record(liveness_path),
            "quality": _file_record(quality_path),
            "quality_bar_value": endpoint["joint_bytes"],
            "bar_start_epoch": continuation_of_epochs + 1,
        },
        "trainer_argv": trainer_argv,
        "closer_launcher_argv": closer_cmd,
        "training_launcher_argv": training_cmd,
        "training_done_receipt_path": str(training_done_path),
        "closer_done_receipt_path": str(closer_done_path),
        "hot_state_staleness_preflight": hot_state,
        "launch_script": _file_record(launch_script),
        "score_claim": False,
        "containment": "composer and closer launch no scorer, Modal, paid work, or auth evaluation",
    }
    composition_path = launcher_dir / "composition_manifest.json"
    _atomic_json(composition_path, composition)
    composition["composition_manifest"] = _file_record(composition_path)
    return composition


def _gate_refusal(message: str, env_gates: Mapping[str, str]) -> dict[str, Any]:
    env_match = re.search(r"\bset\s+([A-Z][A-Z0-9_]+)=([^\s]+)", message)
    if env_match:
        gate = env_match.group(1)
        cure = "SOURCE_PARSED_REQUIRED_ENVIRONMENT_ASSIGNMENT"
        return {
            "type": "GATE_REFUSED",
            "gate": gate,
            "required_value": env_gates.get(gate, env_match.group(2)),
            "known_cure_class": cure,
            "message": message,
        }
    if "run identity differs" in message:
        return {
            "type": "GATE_REFUSED",
            "gate": "resume_identity",
            "known_cure_class": "SEALED_WRAPPER_CONTINUATION_IDENTITY_ADAPTER",
            "message": message,
        }
    if "resume lineage entry" in message and "custody fields" in message:
        return {
            "type": "GATE_REFUSED",
            "gate": "resume_lineage_custody",
            "known_cure_class": "TYPED_LINEAGE_UNTOUCHED_WRAPPER_PROVENANCE_RECEIPT",
            "message": message,
        }
    return {
        "type": "GATE_REFUSED",
        "gate": "CL1TrainingError",
        "known_cure_class": "SOURCE_ADJUDICATION_REQUIRED",
        "message": message,
    }


def verify_gate_chain(
    *,
    log_path: Path,
    done_receipt_path: Path,
    start_offset: int,
    parent_epoch: int,
    timeout_s: float,
    poll_s: float,
    env_gates: Mapping[str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    observed: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(event: dict[str, Any]) -> None:
        identity = (event.get("type"), event.get("gate"), event.get("epoch"))
        if identity not in seen:
            seen.add(identity)
            observed.append(event)

    while True:
        try:
            with log_path.open("rb") as handle:
                size = handle.seek(0, os.SEEK_END)
                if size < start_offset:
                    raise ContinuationCompositionError(
                        f"run log shrank below launch offset {start_offset}: {log_path}"
                    )
                handle.seek(start_offset)
                lines = handle.read().decode("utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            lines = []
        for line in lines:
            if "CL1TrainingError:" in line:
                event = _gate_refusal(line.split("CL1TrainingError:", 1)[1].strip(), env_gates)
                add(event)
                return {
                    "schema": VERIFICATION_SCHEMA,
                    "outcome": event,
                    "events": observed,
                    "bounded_timeout_is_process_failure": False,
                }
            if "[continuation] epoch-extension identity reconciled" in line:
                add({"type": "RECONCILED"})
            if "{" not in line:
                continue
            try:
                row = json.loads(line[line.index("{") :])
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict):
                continue
            try:
                epoch = int(row.get("epoch"))
            except (TypeError, ValueError):
                continue
            if row.get("resume") is True:
                add({"type": "RESUMED", "epoch": epoch})
            if epoch > parent_epoch and "estimated_joint_bytes" in row and "phase" in row:
                event = {
                    "type": "FIRST_EPOCH_ROW",
                    "epoch": epoch,
                    "joint_bytes": int(row["estimated_joint_bytes"]),
                }
                add(event)
                return {
                    "schema": VERIFICATION_SCHEMA,
                    "outcome": event,
                    "events": observed,
                    "bounded_timeout_is_process_failure": False,
                }
        if done_receipt_path.is_file():
            try:
                done = json.loads(done_receipt_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                done = {}
            if isinstance(done, dict) and done.get("schema") == DONE_SCHEMA:
                event = {"type": "DEAD", "rc": int(done.get("rc", -1))}
                add(event)
                return {
                    "schema": VERIFICATION_SCHEMA,
                    "outcome": event,
                    "events": observed,
                    "bounded_timeout_is_process_failure": False,
                }
        if time.monotonic() - started >= timeout_s:
            event = observed[-1] if observed else {"type": "PENDING_BOUNDED"}
            return {
                "schema": VERIFICATION_SCHEMA,
                "outcome": event,
                "events": observed,
                "bounded_timeout_is_process_failure": False,
            }
        time.sleep(poll_s)


def _cmd_compose(args: argparse.Namespace) -> int:
    composition = compose_continuation(
        parent_run_dir=args.parent_run_dir,
        port_mode=args.port_mode,
        run_root=args.run_root,
        overrides=args.override,
        done_receipt=args.done_receipt,
        closer_deadline_s=args.closer_deadline_s,
        closer_poll_s=args.closer_poll_s,
    )
    hot = composition["hot_state_staleness_preflight"]
    if hot["status"].startswith("WARN"):
        print(f"WARN: {hot['warning']}", file=sys.stderr)
    if not args.fire:
        print(_canonical_json({"status": "COMPOSED_NOT_FIRED", **composition}), end="")
        return 0
    log_path = Path(composition["run_root"]) / "launcher/run.log"
    start_offset = log_path.stat().st_size if log_path.exists() else 0
    launch_script = Path(composition["launch_script"]["path"])
    proc = subprocess.run(
        ["/bin/sh", str(launch_script)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    fire_log = Path(composition["run_root"]) / "launcher/composer_fire.log"
    _atomic_text(fire_log, proc.stdout)
    verification = verify_gate_chain(
        log_path=log_path,
        done_receipt_path=Path(composition["training_done_receipt_path"]),
        start_offset=start_offset,
        parent_epoch=int(composition["parent_endpoint"]["epoch"]),
        timeout_s=args.verify_timeout_s,
        poll_s=args.verify_poll_s,
        env_gates=composition["required_environment_gates_from_source"],
    )
    verification.update(
        {
            "generated_utc": _utc_now(),
            "launch_script_rc": proc.returncode,
            "launch_script_log": _file_record(fire_log),
        }
    )
    verification_path = Path(composition["run_root"]) / "launcher/gate_chain_verification.json"
    _atomic_json(verification_path, verification)
    print(_canonical_json(verification), end="")
    if verification["outcome"]["type"] in {"GATE_REFUSED", "DEAD"}:
        cure = verification["outcome"].get("known_cure_class")
        if cure:
            print(
                f"gate={verification['outcome'].get('gate')} known_cure_class={cure}",
                file=sys.stderr,
            )
        return 4
    return 0 if proc.returncode == 0 else proc.returncode


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_gate_chain(
        log_path=args.log,
        done_receipt_path=args.done_receipt_path,
        start_offset=args.start_offset,
        parent_epoch=args.parent_epoch,
        timeout_s=args.timeout_s,
        poll_s=args.poll_s,
        env_gates=parse_required_environment_gates(args.reference_trainer),
    )
    print(_canonical_json(result), end="")
    return 4 if result["outcome"]["type"] in {"GATE_REFUSED", "DEAD"} else 0


def _cmd_hot_state(args: argparse.Namespace) -> int:
    result = check_hot_state_staleness(args.hot_state, args.frontier_pointer)
    if result["status"].startswith("WARN"):
        print(f"WARN: {result['warning']}", file=sys.stderr)
    print(_canonical_json(result), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compose = sub.add_parser("compose", help="Seal a continuation and optionally fire it.")
    compose.add_argument("--parent-run-dir", required=True, type=Path)
    compose.add_argument("--port-mode", required=True)
    compose.add_argument("--run-root", type=Path)
    compose.add_argument(
        "--override",
        action="append",
        default=[],
        help="Reference-trainer override as --flag or --flag=value; sealed fields are refused.",
    )
    compose.add_argument("--done-receipt")
    compose.add_argument("--closer-deadline-s", type=float, default=12 * 3600)
    compose.add_argument("--closer-poll-s", type=float, default=30)
    compose.add_argument("--verify-timeout-s", type=float, default=180)
    compose.add_argument("--verify-poll-s", type=float, default=2)
    compose.add_argument("--fire", action="store_true")
    compose.set_defaults(func=_cmd_compose)

    verify = sub.add_parser("verify", help="Run only the bounded gate-chain verifier.")
    verify.add_argument("--log", required=True, type=Path)
    verify.add_argument("--done-receipt-path", required=True, type=Path)
    verify.add_argument("--start-offset", required=True, type=int)
    verify.add_argument("--parent-epoch", required=True, type=int)
    verify.add_argument("--timeout-s", type=float, default=180)
    verify.add_argument("--poll-s", type=float, default=2)
    verify.add_argument("--reference-trainer", type=Path, default=REFERENCE_TRAINER)
    verify.set_defaults(func=_cmd_verify)

    hot = sub.add_parser("check-hot-state", help="Run the warn-only pointer staleness preflight.")
    hot.add_argument("--hot-state", type=Path, default=HOT_STATE)
    hot.add_argument("--frontier-pointer", type=Path, default=FRONTIER_POINTER)
    hot.set_defaults(func=_cmd_hot_state)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("closer_deadline_s", "closer_poll_s", "verify_timeout_s", "verify_poll_s"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "timeout_s") and (args.timeout_s < 0 or args.poll_s <= 0):
        parser.error("--timeout-s must be nonnegative and --poll-s must be positive")
    try:
        return int(args.func(args))
    except (ContinuationCompositionError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "schema": COMPOSITION_SCHEMA}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
