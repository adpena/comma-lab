#!/usr/bin/env python3
"""Submit or dry-run official Lightning Batch Jobs for pact lanes/evals."""
from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# `lightning_sdk.__init__` performs a PyPI version check on import unless this
# is set. Keep exact-eval tooling deterministic and avoid package-index network
# touches on import, especially during supply-chain incidents.
os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")

from tac.deploy.lightning.batch_jobs import (  # noqa: E402
    LightningAdjudicationSpec,
    LightningBatchJobsClient,
    archive_identity,
    make_exact_eval_spec,
    make_official_component_response_spec,
    lightning_sdk_job_name,
    mirror_local_component_response_artifact_dir,
    mirror_local_artifact_dir,
    mirror_ssh_component_response_artifact_dir,
    validate_local_component_response_artifact_dir,
    validate_local_artifact_dir,
)

SSH_AUTH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "KbdInteractiveAuthentication=no",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_env_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--env requires KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_metadata_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--queue-metadata requires KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _queue_metadata_from_args(args: argparse.Namespace) -> dict[str, str]:
    metadata = _parse_metadata_kv(args.queue_metadata)
    reason = getattr(args, "allow_skip_remote_preflight_reason", None)
    if reason:
        metadata["remote_preflight_skip_reason"] = str(reason).strip()
    return metadata


def _state_path(args: argparse.Namespace) -> Path | None:
    return Path(args.state_path) if args.state_path else None


def _client(args: argparse.Namespace) -> LightningBatchJobsClient:
    state_path = _state_path(args)
    if state_path is None:
        return LightningBatchJobsClient()
    return LightningBatchJobsClient(state_path=state_path)


def _adjudication_from_args(args: argparse.Namespace) -> LightningAdjudicationSpec | None:
    component_gate_args = (
        "max_posenet_dist",
        "max_segnet_dist",
        "baseline_posenet_dist",
        "baseline_segnet_dist",
        "max_posenet_relative",
        "max_segnet_relative",
    )
    if not args.adjudicate:
        if any(getattr(args, key, None) is not None for key in component_gate_args):
            raise SystemExit("component gates require --adjudicate")
        return None
    missing = []
    if args.baseline_score is None:
        missing.append("--baseline-score")
    if args.predicted_band is None:
        missing.append("--predicted-band LOW HIGH")
    if args.regression_threshold is None:
        missing.append("--regression-threshold")
    if missing:
        raise SystemExit("--adjudicate requires " + ", ".join(missing))
    return LightningAdjudicationSpec(
        baseline_score=args.baseline_score,
        predicted_band_low=args.predicted_band[0],
        predicted_band_high=args.predicted_band[1],
        regression_threshold=args.regression_threshold,
        baseline_archive_size_bytes=args.baseline_archive_bytes,
        max_posenet_dist=args.max_posenet_dist,
        max_segnet_dist=args.max_segnet_dist,
        baseline_posenet_dist=args.baseline_posenet_dist,
        baseline_segnet_dist=args.baseline_segnet_dist,
        max_posenet_relative=args.max_posenet_relative,
        max_segnet_relative=args.max_segnet_relative,
        component_reference_label=args.component_reference_label,
        delta_key=args.delta_key,
        max_sane_score=args.max_sane_score,
        allow_component_gate_forensic_success=(
            not args.fail_job_on_component_gate
        ),
    )


def _expected_archive_fields(args: argparse.Namespace) -> tuple[str | None, int | None]:
    expected_sha = args.expected_archive_sha256
    expected_bytes = args.expected_archive_size_bytes
    if args.infer_expected_archive:
        identity = archive_identity(args.archive)
        expected_sha = identity["archive_sha256"]
        expected_bytes = identity["archive_size_bytes"]
    return expected_sha, expected_bytes


def _expected_baseline_archive_fields(args: argparse.Namespace) -> tuple[str | None, int | None]:
    expected_sha = args.expected_baseline_archive_sha256
    expected_bytes = args.expected_baseline_archive_size_bytes
    if args.infer_expected_baseline_archive:
        identity = archive_identity(args.local_baseline_archive or args.baseline_archive)
        expected_sha = identity["archive_sha256"]
        expected_bytes = identity["archive_size_bytes"]
    return expected_sha, expected_bytes


def _repo_rel(path: str | Path) -> str:
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(f"path must stay inside repo for reproducible staging: {path}") from exc


def _safe_remote_repo_rel(value: str, *, field: str) -> str:
    rel = str(value).strip()
    if not rel:
        raise SystemExit(f"{field} must be a non-empty repo-relative path")
    if "\\" in rel or any(ch in rel for ch in "\r\n\0"):
        raise SystemExit(f"{field} contains unsafe path characters: {value!r}")
    if PurePosixPath(rel).is_absolute():
        raise SystemExit(f"{field} must be repo-relative, got absolute path: {value!r}")
    parts = rel.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SystemExit(f"{field} contains path traversal or unstable separators: {value!r}")
    for part in parts:
        if part == "__MACOSX" or part == ".DS_Store" or part.startswith("._") or part.startswith("."):
            raise SystemExit(f"{field} contains hidden or resource-fork path component: {value!r}")
    return rel


def _remote_repo_rel(path: str | None, *, repo_dir: str) -> str | None:
    if not path:
        return None
    raw = str(path).strip()
    if "\\" in raw or any(ch in raw for ch in "\r\n\0"):
        raise SystemExit(f"remote path contains unsafe characters: {path!r}")
    repo = str(repo_dir).rstrip("/")
    if raw.startswith(repo + "/"):
        return _safe_remote_repo_rel(raw[len(repo) + 1 :].strip("/"), field="remote repo path")
    if not PurePosixPath(raw).is_absolute():
        return _safe_remote_repo_rel(raw.strip("/"), field="remote repo path")
    return None


def _manifest_repo_paths(manifest: dict[str, object], *, manifest_path: Path) -> set[str]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SystemExit(f"source manifest missing files list: {manifest_path}")
    paths: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise SystemExit(f"source manifest files[{index}].path must be a string: {manifest_path}")
        rel = _safe_remote_repo_rel(str(item["path"]), field=f"source manifest files[{index}].path")
        if rel in paths:
            raise SystemExit(f"source manifest contains duplicate path: {rel}")
        paths.add(rel)
    return paths


def _require_remote_preflight_for_submit(args: argparse.Namespace, *, role: str) -> None:
    if args.dry_run:
        return
    if not args.studio:
        return
    if args.remote_preflight_ssh_target:
        return
    reason = str(getattr(args, "allow_skip_remote_preflight_reason", "") or "").strip()
    if reason:
        if len(reason) < 12:
            raise SystemExit("--allow-skip-remote-preflight-reason must be a specific auditable reason")
        return
    raise SystemExit(
        f"{role} non-dry-run Studio submit requires --remote-preflight-ssh-target. "
        "Use --allow-skip-remote-preflight-reason only for a documented break-glass image-backed "
        "or separately attested custody path."
    )


def _require_manual_artifact_override(args: argparse.Namespace, *, role: str) -> str | None:
    manual_remote = bool(getattr(args, "remote_artifact_dir", None))
    manual_without_state = not bool(getattr(args, "job_name", None))
    if not manual_remote and not manual_without_state:
        return None
    if not getattr(args, "allow_manual_artifact_dir", False):
        raise SystemExit(
            f"{role} SSH harvest must be state-derived. Manual --remote-artifact-dir "
            "requires --allow-manual-artifact-dir and --override-reason."
        )
    reason = str(getattr(args, "override_reason", "") or "").strip()
    if len(reason) < 12:
        raise SystemExit("--override-reason must be a specific auditable reason")
    return reason


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _parse_ssh_g(stdout: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, sep, value = line.partition(" ")
        if not sep:
            continue
        values.setdefault(key.lower(), []).append(value.strip())
    return values


def _last_ssh_value(values: dict[str, list[str]], key: str) -> str | None:
    entries = values.get(key)
    return entries[-1] if entries else None


def _expand_ssh_path(value: str) -> str:
    return str(Path(os.path.expandvars(os.path.expanduser(value))).resolve())


def _public_key_for_identity(identity_file: str) -> str:
    return _expand_ssh_path(identity_file) + ".pub"


def _identity_guidance(values: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for identity in values.get("identityfile", []):
        expanded = _expand_ssh_path(identity)
        if expanded in seen:
            continue
        seen.add(expanded)
        pub = _public_key_for_identity(identity)
        lines.append(
            "  - "
            f"identity={expanded} exists={Path(expanded).is_file()} "
            f"public_key={pub} public_key_exists={Path(pub).is_file()}"
        )
    return lines


def _ssh_policy_violations(values: dict[str, list[str]]) -> list[str]:
    violations: list[str] = []
    strict = str(_last_ssh_value(values, "stricthostkeychecking") or "").lower()
    if strict in {"false", "no", "off"}:
        violations.append(
            "StrictHostKeyChecking is disabled; use accept-new or yes for Lightning custody"
        )
    identity_files = values.get("identityfile", [])
    if identity_files and not any(Path(_expand_ssh_path(item)).is_file() for item in identity_files):
        violations.append("no resolved IdentityFile exists on disk")
    if identity_files and not any(Path(_public_key_for_identity(item)).is_file() for item in identity_files):
        violations.append("no resolved IdentityFile has a sibling .pub key for Lightning registration")
    return violations


def _ensure_ssh_auth_ready(
    ssh_target: str,
    *,
    ssh_bin: str = "ssh",
    connect_timeout: int = 15,
) -> None:
    target = str(ssh_target).strip()
    if not target or any(ch in target for ch in "\r\n\0"):
        raise SystemExit("Lightning SSH target must be non-empty and must not contain control characters")
    if target == "ssh.lightning.ai":
        raise SystemExit(
            "Lightning SSH target must be a ~/.ssh/config alias or user-qualified target, not bare ssh.lightning.ai"
        )
    config = subprocess.run(
        [ssh_bin, "-G", target],
        capture_output=True,
        text=True,
        check=False,
    )
    values = _parse_ssh_g(config.stdout) if config.returncode == 0 else {}
    probe = subprocess.run(
        [
            ssh_bin,
            *SSH_AUTH_OPTIONS,
            "-o",
            f"ConnectTimeout={int(connect_timeout)}",
            target,
            "true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    policy_violations = _ssh_policy_violations(values) if values else []
    if probe.returncode == 0 and not policy_violations:
        return
    lines = [
        f"Lightning SSH auth preflight failed for {target!r}; harvest blocked before copying artifacts.",
    ]
    if values:
        lines.append(
            "Resolved SSH endpoint: "
            f"user={_last_ssh_value(values, 'user')!r} "
            f"host={_last_ssh_value(values, 'hostname')!r} "
            f"identitiesonly={_last_ssh_value(values, 'identitiesonly')!r} "
            f"strict_host_key_checking={_last_ssh_value(values, 'stricthostkeychecking')!r}."
        )
        identity_lines = _identity_guidance(values)
        if identity_lines:
            lines.append("Resolved identity/public-key candidates:")
            lines.extend(identity_lines)
    elif config.stderr.strip():
        lines.append("ssh -G stderr: " + config.stderr.strip())
    if probe.stderr.strip():
        lines.append("SSH stderr: " + probe.stderr.strip())
    if policy_violations:
        lines.append("SSH policy violations: " + "; ".join(policy_violations))
    lines.extend(
        [
            "Fix: add the selected *.pub key to Lightning Studio/account SSH keys, or correct ~/.ssh/config User/IdentityFile.",
            "Verify with: ssh -o BatchMode=yes <alias> true",
            "Do not use the bare lightning CLI for discovery; keep using lightning-sdk repo wrappers.",
        ]
    )
    raise SystemExit("\n".join(lines))


def _safe_artifact_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "lightning_batch_job"


def _run_remote_supply_chain_preflight(
    *,
    ssh_target: str | None,
    job_name: str,
    repo_dir: str,
    python_bin: str,
    ssh_bin: str = "ssh",
    connect_timeout: int = 15,
) -> None:
    if not ssh_target:
        return
    _ensure_ssh_auth_ready(
        ssh_target,
        ssh_bin=ssh_bin,
        connect_timeout=connect_timeout,
    )
    target = str(ssh_target).strip()
    repo = str(repo_dir).strip()
    py = str(python_bin).strip() or ".venv/bin/python"
    if not repo or any(ch in repo for ch in "\r\n\0"):
        raise SystemExit("remote supply-chain preflight requires a safe non-empty --repo-dir")
    if not py or any(ch in py for ch in "\r\n\0"):
        raise SystemExit("remote supply-chain preflight requires a safe non-empty --python-bin")
    out = f".omx/state/lightning_batch_pre_submit_supply_chain_{_safe_artifact_stem(job_name)}.json"
    command = (
        "cd "
        + shlex.quote(repo)
        + " && "
        + shlex.quote(py)
        + " scripts/scan_lightning_supply_chain.py --json-out "
        + shlex.quote(out)
        + " --quiet --strict"
        + " > /dev/null"
        + " && cat "
        + shlex.quote(out)
    )
    result = subprocess.run(
        [
            ssh_bin,
            *SSH_AUTH_OPTIONS,
            "-o",
            f"ConnectTimeout={int(connect_timeout)}",
            target,
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    stdout_tail = result.stdout.strip()[-4000:]
    stderr_tail = result.stderr.strip()[-4000:]
    details = [
        f"remote Lightning supply-chain preflight failed for {job_name!r} on {target!r}; submit blocked.",
        f"repo_dir={repo!r}",
        f"python_bin={py!r}",
    ]
    if stdout_tail:
        details.append("remote stdout tail:\n" + stdout_tail)
    if stderr_tail:
        details.append("remote stderr tail:\n" + stderr_tail)
    raise SystemExit("\n".join(details))


def _parse_json_stdout(stdout: str) -> dict[str, object] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _run_local_supply_chain_scan() -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "scan_lightning_supply_chain.py"),
            "--repo-root",
            str(REPO_ROOT),
            "--quiet",
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _parse_json_stdout(result.stdout)
    return {
        "returncode": result.returncode,
        "payload": payload,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "ok": result.returncode == 0 and isinstance(payload, dict) and payload.get("status") == "OK",
    }


def _run_remote_supply_chain_scan(
    *,
    ssh_target: str,
    repo_dir: str,
    python_bin: str,
    artifact_stem: str,
    ssh_bin: str = "ssh",
    connect_timeout: int = 15,
) -> dict[str, object]:
    _ensure_ssh_auth_ready(
        ssh_target,
        ssh_bin=ssh_bin,
        connect_timeout=connect_timeout,
    )
    out = f".omx/state/lightning_doctor_remote_supply_chain_{_safe_artifact_stem(artifact_stem)}.json"
    command = (
        "cd "
        + shlex.quote(repo_dir)
        + " && "
        + shlex.quote(python_bin)
        + " scripts/scan_lightning_supply_chain.py --json-out "
        + shlex.quote(out)
        + " --quiet --strict > /dev/null && cat "
        + shlex.quote(out)
    )
    result = subprocess.run(
        [
            ssh_bin,
            *SSH_AUTH_OPTIONS,
            "-o",
            f"ConnectTimeout={int(connect_timeout)}",
            ssh_target,
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _parse_json_stdout(result.stdout)
    return {
        "returncode": result.returncode,
        "payload": payload,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "remote_json_out": out,
        "ok": result.returncode == 0 and isinstance(payload, dict) and payload.get("status") == "OK",
    }


def _component_response_plan_rel(root: Path, value: object, *, field: str) -> str:
    raw = str(value)
    if Path(raw).is_absolute():
        raise SystemExit(
            f"component-response submit blocked; perturbation plan {field} must be "
            f"relative to the plan file for remote reproducibility, got {raw!r}"
        )
    return _repo_rel(root / raw)


def _component_response_plan_artifacts(
    plan_path: Path,
    *,
    baseline_rel: str,
    ignore_top_baseline_json: bool = False,
) -> set[str]:
    payload = json.loads(plan_path.read_text())
    if isinstance(payload, list):
        plan = {"points": payload}
    elif isinstance(payload, dict):
        plan = payload
    else:
        raise SystemExit(f"perturbation plan must be a JSON object or list: {plan_path}")
    points = plan.get("points")
    if not isinstance(points, list) or not points:
        raise SystemExit(f"perturbation plan points must be a non-empty list: {plan_path}")
    root = plan_path.parent
    rels: set[str] = {baseline_rel, _repo_rel(plan_path)}
    top_baseline_json = plan.get("baseline_contest_auth_eval_json")
    if top_baseline_json is not None and not ignore_top_baseline_json:
        rels.add(
            _component_response_plan_rel(
                root,
                top_baseline_json,
                field="baseline_contest_auth_eval_json",
            )
        )
    nonzero = 0
    for index, raw in enumerate(points):
        if not isinstance(raw, dict):
            raise SystemExit(f"perturbation plan points[{index}] must be an object")
        epsilon = float(raw.get("epsilon"))
        archive_value = raw.get("archive")
        if archive_value is None and abs(epsilon) <= 1e-12:
            rels.add(baseline_rel)
        elif archive_value is None:
            raise SystemExit(f"perturbation plan points[{index}].archive is required for nonzero epsilon")
        else:
            rels.add(
                _component_response_plan_rel(
                    root,
                    archive_value,
                    field=f"points[{index}].archive",
                )
            )
        nonzero += int(abs(epsilon) > 1e-12)
        json_value = raw.get("contest_auth_eval_json")
        if json_value is not None:
            rels.add(
                _component_response_plan_rel(
                    root,
                    json_value,
                    field=f"points[{index}].contest_auth_eval_json",
                )
            )
    if nonzero <= 0:
        raise SystemExit("perturbation plan must include at least one nonzero response point")
    return rels


def _validate_component_response_submit_inputs(args: argparse.Namespace) -> None:
    if args.dry_run and not args.source_manifest and not args.local_perturbation_plan:
        return
    if bool(args.source_manifest) != bool(args.local_perturbation_plan):
        raise SystemExit(
            "component-response closure validation requires both "
            "--source-manifest and --local-perturbation-plan; omit both only "
            "for a topology-only dry-run"
        )
    if not args.source_manifest:
        raise SystemExit(
            "component-response submit requires --source-manifest from "
            "scripts/lightning_repro_workspace.py; use --dry-run until inputs are staged"
        )
    if not args.local_perturbation_plan:
        raise SystemExit(
            "component-response submit requires --local-perturbation-plan so "
            "the plan-listed archives can be checked against the staged manifest"
        )
    manifest_path = Path(args.source_manifest)
    manifest = _load_json_object(manifest_path, label="source manifest")
    manifest_paths = _manifest_repo_paths(manifest, manifest_path=manifest_path)
    baseline_rel = _remote_repo_rel(args.baseline_archive, repo_dir=args.repo_dir)
    plan_rel = _remote_repo_rel(args.perturbation_plan, repo_dir=args.repo_dir)
    if baseline_rel is None or plan_rel is None:
        raise SystemExit("--baseline-archive and --perturbation-plan must be inside --repo-dir for submit")
    baseline_json_rel = _remote_repo_rel(args.baseline_contest_auth_eval_json, repo_dir=args.repo_dir)
    required = _component_response_plan_artifacts(
        Path(args.local_perturbation_plan),
        baseline_rel=baseline_rel,
        ignore_top_baseline_json=baseline_json_rel is not None,
    )
    required.add(plan_rel)
    if baseline_json_rel is not None:
        required.add(baseline_json_rel)
    missing = sorted(required.difference(manifest_paths))
    if missing:
        raise SystemExit(
            "component-response submit blocked; staged source manifest does not include "
            "all required plan artifacts: " + ", ".join(missing[:20])
        )


def _validate_exact_eval_submit_inputs(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    if not args.studio:
        return
    if not args.source_manifest:
        raise SystemExit(
            "exact-eval non-dry-run Studio submit requires --source-manifest from "
            "scripts/lightning_repro_workspace.py so the archive bytes are known staged inputs"
        )
    archive_rel = _remote_repo_rel(args.archive, repo_dir=args.repo_dir)
    if archive_rel is None:
        raise SystemExit("--archive must be inside --repo-dir for exact-eval Studio submit")
    manifest_path = Path(args.source_manifest)
    manifest = _load_json_object(manifest_path, label="source manifest")
    manifest_paths = _manifest_repo_paths(manifest, manifest_path=manifest_path)
    if archive_rel not in manifest_paths:
        raise SystemExit(
            "exact-eval submit blocked; staged source manifest does not include "
            f"archive artifact: {archive_rel}"
        )
    metadata = _parse_metadata_kv(getattr(args, "queue_metadata", []) or [])
    baseline_json = metadata.get("baseline_json") or metadata.get("baseline_contest_auth_eval_json")
    if baseline_json:
        baseline_json_rel = _remote_repo_rel(baseline_json, repo_dir=args.repo_dir)
        if baseline_json_rel is None:
            raise SystemExit("exact-eval metadata baseline_json must be inside --repo-dir")
        if baseline_json_rel not in manifest_paths:
            raise SystemExit(
                "exact-eval submit blocked; staged source manifest does not include "
                f"metadata baseline_json artifact: {baseline_json_rel}"
            )


def cmd_exact_eval(args: argparse.Namespace) -> int:
    if not args.adjudicate:
        raise SystemExit(
            "exact-eval requires --adjudicate so every Lightning result has "
            "contest_auth_eval.json custody plus adjudication provenance"
        )
    _validate_exact_eval_submit_inputs(args)
    _require_remote_preflight_for_submit(args, role="exact-eval")
    if not args.dry_run:
        _run_remote_supply_chain_preflight(
            ssh_target=args.remote_preflight_ssh_target,
            job_name=args.job_name,
            repo_dir=args.repo_dir,
            python_bin=args.python_bin,
            ssh_bin=args.remote_preflight_ssh_bin,
            connect_timeout=args.remote_preflight_ssh_connect_timeout,
        )
    expected_sha, expected_bytes = _expected_archive_fields(args)
    spec = make_exact_eval_spec(
        name=args.job_name,
        archive_path=args.archive,
        repo_dir=args.repo_dir,
        upstream_dir=args.upstream_dir,
        output_dir=args.output_dir,
        machine=args.machine,
        studio=args.studio,
        image=args.image,
        python_bin=args.python_bin,
        max_runtime=args.max_runtime,
        env=_parse_env_kv(args.env),
        teamspace=args.teamspace,
        org=args.org,
        user=args.user,
        expected_archive_sha256=expected_sha,
        expected_archive_size_bytes=expected_bytes,
        queue_metadata=_queue_metadata_from_args(args),
        local_artifact_dir=args.local_artifact_dir,
        adjudication=_adjudication_from_args(args),
    )
    client = _client(args)
    record = client.submit(spec, dry_run=args.dry_run)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def cmd_component_response(args: argparse.Namespace) -> int:
    _validate_component_response_submit_inputs(args)
    _require_remote_preflight_for_submit(args, role="component-response")
    if not args.dry_run:
        _run_remote_supply_chain_preflight(
            ssh_target=args.remote_preflight_ssh_target,
            job_name=args.job_name,
            repo_dir=args.repo_dir,
            python_bin=args.python_bin,
            ssh_bin=args.remote_preflight_ssh_bin,
            connect_timeout=args.remote_preflight_ssh_connect_timeout,
        )
    expected_sha, expected_bytes = _expected_baseline_archive_fields(args)
    spec = make_official_component_response_spec(
        name=args.job_name,
        baseline_archive_path=args.baseline_archive,
        perturbation_plan_path=args.perturbation_plan,
        repo_dir=args.repo_dir,
        upstream_dir=args.upstream_dir,
        output_dir=args.output_dir,
        machine=args.machine,
        studio=args.studio,
        image=args.image,
        python_bin=args.python_bin,
        max_runtime=args.max_runtime,
        env=_parse_env_kv(args.env),
        teamspace=args.teamspace,
        org=args.org,
        user=args.user,
        baseline_contest_auth_eval_json=args.baseline_contest_auth_eval_json,
        inflate_sh=args.inflate_sh,
        video_names_file=args.video_names_file,
        expected_baseline_archive_sha256=expected_sha,
        expected_baseline_archive_size_bytes=expected_bytes,
        queue_metadata=_queue_metadata_from_args(args),
        local_artifact_dir=args.local_artifact_dir,
        max_relative_error=args.max_relative_error,
        zero_repro_tolerance=args.zero_repro_tolerance,
        min_observed_delta=args.min_observed_delta,
        allow_directional=args.allow_directional,
        require_passed=args.require_passed,
    )
    client = _client(args)
    record = client.submit(spec, dry_run=args.dry_run)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    client = _client(args)
    print(json.dumps(client.list_records(), indent=2, sort_keys=True))
    return 0


def cmd_validate_artifacts(args: argparse.Namespace) -> int:
    if args.mirror_dir:
        result = mirror_local_artifact_dir(
            args.artifact_dir,
            args.mirror_dir,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_archive_size_bytes=args.expected_archive_size_bytes,
            require_adjudication=args.require_adjudication,
            overwrite=args.overwrite,
        )
    else:
        result = validate_local_artifact_dir(
            args.artifact_dir,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_archive_size_bytes=args.expected_archive_size_bytes,
            require_adjudication=args.require_adjudication,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_validate_component_response_artifacts(args: argparse.Namespace) -> int:
    result = validate_local_component_response_artifact_dir(
        args.artifact_dir,
        expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
        require_passed=args.require_passed,
    )
    validation_path = Path(args.artifact_dir) / "official_component_response_artifact_validation.json"
    validation_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_harvest_component_response_local(args: argparse.Namespace) -> int:
    if args.mirror_dir:
        result = mirror_local_component_response_artifact_dir(
            args.artifact_dir,
            args.mirror_dir,
            expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
            expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
            require_passed=args.require_passed,
            overwrite=args.overwrite,
        )
    else:
        result = validate_local_component_response_artifact_dir(
            args.artifact_dir,
            expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
            expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
            require_passed=args.require_passed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_harvest_component_response_ssh(args: argparse.Namespace) -> int:
    _require_manual_artifact_override(args, role="component-response")
    if args.job_name:
        client = _client(args)
        _ensure_ssh_auth_ready(
            args.ssh_target,
            ssh_bin=args.ssh_bin,
            connect_timeout=args.ssh_connect_timeout,
        )
        result = client.harvest_ssh_component_response_artifacts(
            job_name=args.job_name,
            ssh_target=args.ssh_target,
            remote_dir=args.remote_artifact_dir,
            mirror_dir=args.mirror_dir,
            expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
            expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
            require_passed=args.require_passed,
            overwrite=args.overwrite,
            ssh_bin=args.ssh_bin,
            scp_bin=args.scp_bin,
            ssh_connect_timeout=args.ssh_connect_timeout,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    _ensure_ssh_auth_ready(
        args.ssh_target,
        ssh_bin=args.ssh_bin,
        connect_timeout=args.ssh_connect_timeout,
    )
    if not args.remote_artifact_dir or not args.mirror_dir:
        raise SystemExit(
            "harvest-component-response-ssh requires either --job-name with state, "
            "or both --remote-artifact-dir and --mirror-dir"
        )
    result = mirror_ssh_component_response_artifact_dir(
        ssh_target=args.ssh_target,
        remote_dir=args.remote_artifact_dir,
        mirror_dir=args.mirror_dir,
        expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
        require_passed=args.require_passed,
        overwrite=args.overwrite,
        ssh_bin=args.ssh_bin,
        scp_bin=args.scp_bin,
        ssh_connect_timeout=args.ssh_connect_timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_harvest_local(args: argparse.Namespace) -> int:
    client = _client(args)
    result = client.harvest_local_artifacts(
        job_name=args.job_name,
        artifact_dir=args.artifact_dir,
        mirror_dir=args.mirror_dir,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_archive_size_bytes=args.expected_archive_size_bytes,
        require_adjudication=args.require_adjudication,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_harvest_ssh(args: argparse.Namespace) -> int:
    _require_manual_artifact_override(args, role="exact-eval")
    client = _client(args)
    _ensure_ssh_auth_ready(
        args.ssh_target,
        ssh_bin=args.ssh_bin,
        connect_timeout=args.ssh_connect_timeout,
    )
    result = client.harvest_ssh_artifacts(
        job_name=args.job_name,
        ssh_target=args.ssh_target,
        remote_dir=args.remote_artifact_dir,
        mirror_dir=args.mirror_dir,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_archive_size_bytes=args.expected_archive_size_bytes,
        require_adjudication=args.require_adjudication,
        overwrite=args.overwrite,
        ssh_bin=args.ssh_bin,
        scp_bin=args.scp_bin,
        ssh_connect_timeout=args.ssh_connect_timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _dict_or_empty(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _latest_record_for_job(
    client: LightningBatchJobsClient,
    job_name: str,
) -> dict[str, object] | None:
    for record in reversed(client.list_records()):
        spec = _dict_or_empty(record.get("spec"))
        queue = _dict_or_empty(record.get("queue"))
        job = _dict_or_empty(record.get("job"))
        if (
            spec.get("name") == job_name
            or queue.get("job_name") == job_name
            or job.get("name") == job_name
        ):
            return record
    return None


def _record_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _record_job_name(record: dict[str, object]) -> str | None:
    spec = _dict_or_empty(record.get("spec"))
    queue = _dict_or_empty(record.get("queue"))
    job_record = _dict_or_empty(record.get("job"))
    return (
        _record_string(spec.get("name"))
        or _record_string(queue.get("job_name"))
        or _record_string(job_record.get("name"))
    )


def cmd_refresh_status(args: argparse.Namespace) -> int:
    client = _client(args)
    if args.all:
        if args.sdk_job_name:
            raise SystemExit("--sdk-job-name is only valid with --job-name")
        latest_by_job: dict[str, dict[str, object]] = {}
        skipped = []
        for record in client.list_records():
            job_name = _record_job_name(record)
            if not job_name:
                skipped.append({"reason": "missing_job_name", "record": record})
                continue
            if record.get("dry_run") and not args.include_dry_runs:
                skipped.append({"job_name": job_name, "reason": "dry_run"})
                continue
            latest_by_job[job_name] = record
        results = []
        failures = []
        for job_name in latest_by_job:
            single_args = argparse.Namespace(
                **{
                    **vars(args),
                    "all": False,
                    "job_name": job_name,
                    "sdk_job_name": None,
                }
            )
            try:
                record = _refresh_one_status(client, single_args)
            except Exception as exc:  # pragma: no cover - exercised via CLI
                failures.append(
                    {
                        "job_name": job_name,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            results.append(record)
        print(
            json.dumps(
                {
                    "refreshed_count": len(results),
                    "skipped_count": len(skipped),
                    "failure_count": len(failures),
                    "results": results,
                    "skipped": skipped,
                    "failures": failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if failures and args.fail_on_error else 0
    if not args.job_name:
        raise SystemExit("refresh-status requires --job-name or --all")
    result = _refresh_one_status(client, args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _refresh_one_status(
    client: LightningBatchJobsClient,
    args: argparse.Namespace,
) -> dict[str, object]:
    record = _latest_record_for_job(client, args.job_name)
    spec = _dict_or_empty(record.get("spec") if record else None)
    job_record = _dict_or_empty(record.get("job") if record else None)
    job_cls = client._job_cls or client._import_job_cls()
    sdk_job_name = (
        args.sdk_job_name
        or _record_string(job_record.get("name"))
        or lightning_sdk_job_name(args.job_name)
    )
    job = job_cls(
        name=sdk_job_name,
        teamspace=args.teamspace if args.teamspace is not None else _record_string(spec.get("teamspace")),
        org=args.org if args.org is not None else _record_string(spec.get("org")),
        user=args.user if args.user is not None else _record_string(spec.get("user")),
    )
    return client.refresh_status_from_job(job_name=args.job_name, job=job)


def _machine_to_record(machine: object) -> dict[str, object]:
    return {
        "name": getattr(machine, "name", None),
        "slug": getattr(machine, "slug", None),
        "instance_type": getattr(machine, "instance_type", None),
        "family": getattr(machine, "family", None),
        "accelerator_count": getattr(machine, "accelerator_count", None),
        "cost": getattr(machine, "cost", None),
        "interruptible_cost": getattr(machine, "interruptible_cost", None),
        "provider": getattr(machine, "provider", None),
        "wait_time": getattr(machine, "wait_time", None),
        "interruptible_wait_time": getattr(machine, "interruptible_wait_time", None),
    }


def _cluster_machine_to_record(machine: object) -> dict[str, object]:
    resources = getattr(machine, "resources", None)
    return {
        "name": getattr(machine, "instance_id", None),
        "slug": getattr(machine, "slug_multi_cloud", None),
        "instance_type": getattr(machine, "instance_id", None),
        "family": getattr(machine, "family", None),
        "accelerator_count": getattr(resources, "gpu", None) or getattr(resources, "cpu", None),
        "cost": getattr(machine, "cost", None),
        "interruptible_cost": getattr(machine, "spot_price", None),
        "provider": getattr(machine, "provider", None),
        "wait_time": getattr(machine, "available_in_seconds", None),
        "interruptible_wait_time": getattr(machine, "available_in_seconds_spot", None),
        "out_of_capacity": getattr(machine, "out_of_capacity", None),
    }


def _list_machine_rows(
    *,
    teamspace_name: str,
    org: str | None,
    user: str | None,
    cloud_accounts: list[str] | None = None,
    machine: str | None = None,
    gpu_only: bool = False,
) -> list[dict[str, object]]:
    from lightning_sdk import Teamspace

    teamspace = Teamspace(name=teamspace_name, org=org, user=user)
    selected_cloud_accounts = cloud_accounts or teamspace.cloud_accounts
    rows = []
    for cloud_account in selected_cloud_accounts:
        try:
            machines = teamspace.list_machines(cloud_account=cloud_account, machine=machine)
            machine_rows = [_machine_to_record(machine) for machine in machines]
        except AttributeError:
            # lightning-sdk 2026.4.10 expects an org-backed teamspace in
            # Teamspace.list_machines(). User-owned teamspaces need org_id="".
            from lightning_sdk.api import TeamspaceApi
            from lightning_sdk.machine import Machine

            machine_filter = None
            if machine:
                machine_filter = getattr(Machine, machine.upper(), Machine(machine, machine))
            raw = TeamspaceApi().list_machines(
                teamspace.id,
                cloud_accounts=[cloud_account],
                machine=machine_filter,
                org_id="",
            )
            machine_rows = [
                _cluster_machine_to_record(machine)
                for machine in raw
                if not getattr(machine, "out_of_capacity", False)
            ]
        if gpu_only:
            machine_rows = [
                row for row in machine_rows
                if row.get("family") not in {"CPU", "DATA-PREP", "DATA_PREP"}
            ]
        rows.append({"cloud_account": cloud_account, "machines": machine_rows})
    return rows


def cmd_list_machines(args: argparse.Namespace) -> int:
    rows = _list_machine_rows(
        teamspace_name=args.teamspace,
        org=args.org,
        user=args.user,
        cloud_accounts=args.cloud_account,
        machine=args.machine,
        gpu_only=args.gpu_only,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def _lightning_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("lightning", "pytorch-lightning", "lightning-sdk", "lightning_sdk"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _doctor_payload(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "tool": "scripts/launch_lightning_batch_job.py doctor",
        "recorded_at_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "package_versions": _lightning_package_versions(),
        "checks": {},
    }
    checks = payload["checks"]
    assert isinstance(checks, dict)

    local_supply_chain = _run_local_supply_chain_scan()
    checks["local_supply_chain"] = local_supply_chain

    ssh_target = getattr(args, "ssh_target", None)
    if ssh_target:
        try:
            _ensure_ssh_auth_ready(
                ssh_target,
                ssh_bin=args.ssh_bin,
                connect_timeout=args.ssh_connect_timeout,
            )
            checks["ssh_auth"] = {"ok": True, "target": ssh_target}
        except SystemExit as exc:
            checks["ssh_auth"] = {"ok": False, "target": ssh_target, "error": str(exc)}
    elif args.require_ssh:
        checks["ssh_auth"] = {"ok": False, "error": "--require-ssh set but --ssh-target was not provided"}
    else:
        checks["ssh_auth"] = {"ok": None, "status": "skipped"}

    if ssh_target and args.remote_supply_chain:
        try:
            checks["remote_supply_chain"] = _run_remote_supply_chain_scan(
                ssh_target=ssh_target,
                repo_dir=args.repo_dir,
                python_bin=args.python_bin,
                artifact_stem=args.run_id,
                ssh_bin=args.ssh_bin,
                connect_timeout=args.ssh_connect_timeout,
            )
        except SystemExit as exc:
            checks["remote_supply_chain"] = {
                "ok": False,
                "target": ssh_target,
                "repo_dir": args.repo_dir,
                "python_bin": args.python_bin,
                "error": str(exc),
            }
    elif args.require_remote_supply_chain:
        checks["remote_supply_chain"] = {
            "ok": False,
            "error": "--require-remote-supply-chain set but remote scan was not run",
        }
    else:
        checks["remote_supply_chain"] = {"ok": None, "status": "skipped"}

    if args.teamspace and args.machine_inventory:
        try:
            rows = _list_machine_rows(
                teamspace_name=args.teamspace,
                org=args.org,
                user=args.user,
                cloud_accounts=args.cloud_account,
                machine=args.machine,
                gpu_only=args.gpu_only,
            )
            machine_count = sum(
                len(row.get("machines", []))
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("machines"), list)
            )
            checks["machine_inventory"] = {
                "ok": machine_count > 0,
                "machine_count": machine_count,
                "rows": rows,
            }
        except Exception as exc:  # pragma: no cover - SDK/network failure branch
            checks["machine_inventory"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    elif args.require_machine_inventory:
        checks["machine_inventory"] = {
            "ok": False,
            "error": "--require-machine-inventory set but --teamspace was not provided or inventory disabled",
        }
    else:
        checks["machine_inventory"] = {"ok": None, "status": "skipped"}

    failed = [
        name
        for name, record in checks.items()
        if isinstance(record, dict) and record.get("ok") is False
    ]
    payload["status"] = "FAIL" if failed else "OK"
    payload["failed_checks"] = failed
    return payload


def cmd_doctor(args: argparse.Namespace) -> int:
    payload = _doctor_payload(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end="")
    return 1 if args.strict and payload.get("status") != "OK" else 0


def _add_state_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-path", default=None, help="Override local Lightning Batch Jobs state JSON.")


def _add_remote_preflight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--remote-preflight-ssh-target",
        default=None,
        help=(
            "SSH alias/target for a strict remote supply-chain scan before "
            "submitting a non-dry-run Studio-backed Batch Job."
        ),
    )
    parser.add_argument("--remote-preflight-ssh-bin", default="ssh")
    parser.add_argument(
        "--remote-preflight-ssh-connect-timeout",
        type=int,
        default=15,
        help="BatchMode SSH auth/preflight timeout in seconds.",
    )
    parser.add_argument(
        "--allow-skip-remote-preflight-reason",
        default=None,
        help=(
            "Break-glass reason allowing a non-dry-run Studio submit without "
            "--remote-preflight-ssh-target. Recorded in queue metadata."
        ),
    )


def _add_expected_archive_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-archive-sha256", default=None)
    parser.add_argument(
        "--expected-archive-size-bytes",
        "--expected-archive-bytes",
        dest="expected_archive_size_bytes",
        type=int,
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    exact = sub.add_parser("exact-eval")
    _add_state_arg(exact)
    exact.add_argument("--job-name", required=True)
    exact.add_argument("--archive", required=True, help="Archive path visible inside the Lightning job.")
    exact.add_argument("--repo-dir", required=True, help="Repo path visible inside the Lightning job.")
    exact.add_argument("--upstream-dir", required=True, help="Upstream scorer path visible inside the Lightning job.")
    exact.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Writable artifact output dir visible inside job. Default: "
            "<repo-dir>/experiments/results/lightning_batch/<job-name>."
        ),
    )
    exact.add_argument("--machine", default="T4")
    exact.add_argument("--studio", default=None, help="Lightning Studio name/env for Studio-backed jobs.")
    exact.add_argument("--image", default=None, help="Docker image for image-backed jobs.")
    exact.add_argument("--teamspace", default=None)
    exact.add_argument("--org", default=None)
    exact.add_argument("--user", default=None)
    exact.add_argument("--python-bin", default=".venv/bin/python")
    exact.add_argument("--max-runtime", type=int, default=3 * 60 * 60)
    exact.add_argument("--env", action="append", default=[], help="KEY=VALUE env override; repeatable.")
    exact.add_argument("--queue-metadata", action="append", default=[], help="KEY=VALUE audit metadata; repeatable.")
    exact.add_argument("--local-artifact-dir", default=None, help="Expected local mirror/harvest path for artifacts.")
    _add_remote_preflight_args(exact)
    exact.add_argument(
        "--source-manifest",
        default=None,
        help=(
            "Local lightning_repro_workspace.py manifest. Required for non-dry-run "
            "Studio exact-eval submit so the archive is known staged input."
        ),
    )
    _add_expected_archive_args(exact)
    exact.add_argument(
        "--infer-expected-archive",
        action="store_true",
        help="Compute expected archive SHA-256/bytes from --archive locally before submit/dry-run.",
    )
    exact.add_argument("--adjudicate", action="store_true", help="Append scripts/adjudicate_contest_auth_eval.py wiring.")
    exact.add_argument("--baseline-score", type=float)
    exact.add_argument("--baseline-archive-bytes", type=int)
    exact.add_argument("--predicted-band", nargs=2, type=float, metavar=("LOW", "HIGH"))
    exact.add_argument("--regression-threshold", type=float)
    exact.add_argument("--delta-key", default="score_delta_vs_baseline")
    exact.add_argument("--max-posenet-dist", type=float)
    exact.add_argument("--max-segnet-dist", type=float)
    exact.add_argument(
        "--baseline-posenet-dist",
        "--reference-posenet-dist",
        dest="baseline_posenet_dist",
        type=float,
        help="Reference PoseNet distortion for --max-posenet-relative.",
    )
    exact.add_argument(
        "--baseline-segnet-dist",
        "--reference-segnet-dist",
        dest="baseline_segnet_dist",
        type=float,
        help="Reference SegNet distortion for --max-segnet-relative.",
    )
    exact.add_argument(
        "--max-posenet-relative",
        type=float,
        help="Reject when avg_posenet_dist / reference_posenet_dist exceeds this ratio.",
    )
    exact.add_argument(
        "--max-segnet-relative",
        type=float,
        help="Reject when avg_segnet_dist / reference_segnet_dist exceeds this ratio.",
    )
    exact.add_argument("--component-reference-label", default="baseline")
    exact.add_argument("--max-sane-score", type=float, default=10.0)
    exact.add_argument(
        "--fail-job-on-component-gate",
        action="store_true",
        help=(
            "Return a failed Lightning job on component-gate violation. By "
            "default exact-eval jobs complete with non-promotable forensic "
            "artifacts when CUDA custody and adjudication succeeded."
        ),
    )
    exact.add_argument("--dry-run", action="store_true")
    exact.set_defaults(func=cmd_exact_eval)

    component = sub.add_parser("component-response")
    _add_state_arg(component)
    component.add_argument("--job-name", required=True)
    component.add_argument("--baseline-archive", required=True, help="Baseline archive path visible inside the Lightning job.")
    component.add_argument("--perturbation-plan", required=True, help="Official component-response plan path visible inside the Lightning job.")
    component.add_argument("--repo-dir", required=True, help="Repo path visible inside the Lightning job.")
    component.add_argument("--upstream-dir", required=True, help="Upstream scorer path visible inside the Lightning job.")
    component.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Writable artifact output dir visible inside job. Default: "
            "<repo-dir>/experiments/results/lightning_batch/<job-name>."
        ),
    )
    component.add_argument("--machine", default="T4")
    component.add_argument("--studio", default=None, help="Lightning Studio name/env for Studio-backed jobs.")
    component.add_argument("--image", default=None, help="Docker image for image-backed jobs.")
    component.add_argument("--teamspace", default=None)
    component.add_argument("--org", default=None)
    component.add_argument("--user", default=None)
    component.add_argument("--python-bin", default=".venv/bin/python")
    component.add_argument("--max-runtime", type=int, default=6 * 60 * 60)
    component.add_argument("--env", action="append", default=[], help="KEY=VALUE env override; repeatable.")
    component.add_argument("--queue-metadata", action="append", default=[], help="KEY=VALUE audit metadata; repeatable.")
    component.add_argument("--local-artifact-dir", default=None, help="Expected local mirror/harvest path for artifacts.")
    _add_remote_preflight_args(component)
    component.add_argument("--baseline-contest-auth-eval-json", default=None)
    component.add_argument("--inflate-sh", default="submissions/robust_current/inflate.sh")
    component.add_argument("--video-names-file", default="upstream/public_test_video_names.txt")
    component.add_argument("--expected-baseline-archive-sha256", default=None)
    component.add_argument(
        "--expected-baseline-archive-size-bytes",
        "--expected-baseline-archive-bytes",
        dest="expected_baseline_archive_size_bytes",
        type=int,
        default=None,
    )
    component.add_argument(
        "--infer-expected-baseline-archive",
        action="store_true",
        help="Compute expected baseline archive SHA-256/bytes locally before submit/dry-run.",
    )
    component.add_argument(
        "--local-baseline-archive",
        default=None,
        help="Local baseline archive to hash when --baseline-archive is a remote path.",
    )
    component.add_argument("--max-relative-error", type=float, default=0.35)
    component.add_argument("--zero-repro-tolerance", type=float, default=1e-7)
    component.add_argument("--min-observed-delta", type=float, default=1e-12)
    component.add_argument("--allow-directional", action="store_true")
    component.add_argument("--require-passed", action="store_true")
    component.add_argument(
        "--source-manifest",
        default=None,
        help=(
            "Local lightning_repro_workspace.py manifest. Required for non-dry-run "
            "component-response submit so plan archives are known staged inputs."
        ),
    )
    component.add_argument(
        "--local-perturbation-plan",
        default=None,
        help="Local perturbation plan used to verify plan-listed archives against --source-manifest before submit.",
    )
    component.add_argument("--dry-run", action="store_true")
    component.set_defaults(func=cmd_component_response)

    list_cmd = sub.add_parser("list")
    _add_state_arg(list_cmd)
    list_cmd.set_defaults(func=cmd_list)

    validate = sub.add_parser("validate-artifacts")
    validate.add_argument("--artifact-dir", required=True)
    validate.add_argument("--mirror-dir", default=None)
    _add_expected_archive_args(validate)
    validate.add_argument("--require-adjudication", action="store_true")
    validate.add_argument("--overwrite", action="store_true")
    validate.set_defaults(func=cmd_validate_artifacts)

    validate_component = sub.add_parser("validate-component-response-artifacts")
    validate_component.add_argument("--artifact-dir", required=True)
    validate_component.add_argument("--expected-baseline-archive-sha256", default=None)
    validate_component.add_argument(
        "--expected-baseline-archive-size-bytes",
        "--expected-baseline-archive-bytes",
        dest="expected_baseline_archive_size_bytes",
        type=int,
        default=None,
    )
    validate_component.add_argument("--require-passed", action="store_true")
    validate_component.set_defaults(func=cmd_validate_component_response_artifacts)

    harvest = sub.add_parser("harvest-local")
    _add_state_arg(harvest)
    harvest.add_argument("--job-name", required=True)
    harvest.add_argument("--artifact-dir", required=True)
    harvest.add_argument("--mirror-dir", default=None)
    _add_expected_archive_args(harvest)
    harvest.add_argument("--require-adjudication", action="store_true")
    harvest.add_argument("--overwrite", action="store_true")
    harvest.set_defaults(func=cmd_harvest_local)

    harvest_component = sub.add_parser("harvest-component-response-local")
    harvest_component.add_argument("--artifact-dir", required=True)
    harvest_component.add_argument("--mirror-dir", default=None)
    harvest_component.add_argument("--expected-baseline-archive-sha256", default=None)
    harvest_component.add_argument(
        "--expected-baseline-archive-size-bytes",
        "--expected-baseline-archive-bytes",
        dest="expected_baseline_archive_size_bytes",
        type=int,
        default=None,
    )
    harvest_component.add_argument("--require-passed", action="store_true")
    harvest_component.add_argument("--overwrite", action="store_true")
    harvest_component.set_defaults(func=cmd_harvest_component_response_local)

    harvest_ssh = sub.add_parser("harvest-ssh")
    _add_state_arg(harvest_ssh)
    harvest_ssh.add_argument("--job-name", required=True)
    harvest_ssh.add_argument("--ssh-target", required=True)
    harvest_ssh.add_argument(
        "--remote-artifact-dir",
        default=None,
        help="Remote Lightning artifact/output dir. Defaults to the recorded spec.remote_output_dir.",
    )
    harvest_ssh.add_argument(
        "--mirror-dir",
        default=None,
        help="Local mirror dir. Defaults to the recorded spec.local_artifact_dir.",
    )
    _add_expected_archive_args(harvest_ssh)
    harvest_ssh.add_argument("--require-adjudication", action="store_true")
    harvest_ssh.add_argument("--overwrite", action="store_true")
    harvest_ssh.add_argument("--allow-manual-artifact-dir", action="store_true")
    harvest_ssh.add_argument("--override-reason", default=None)
    harvest_ssh.add_argument("--ssh-bin", default="ssh")
    harvest_ssh.add_argument("--scp-bin", default="scp")
    harvest_ssh.add_argument(
        "--ssh-connect-timeout",
        type=int,
        default=15,
        help="BatchMode SSH auth preflight timeout in seconds.",
    )
    harvest_ssh.set_defaults(func=cmd_harvest_ssh)

    harvest_component_ssh = sub.add_parser("harvest-component-response-ssh")
    _add_state_arg(harvest_component_ssh)
    harvest_component_ssh.add_argument(
        "--job-name",
        default=None,
        help=(
            "State record name. When provided, remote/mirror dirs default "
            "from state and Studio outputs are mapped into SDK artifacts."
        ),
    )
    harvest_component_ssh.add_argument("--ssh-target", required=True)
    harvest_component_ssh.add_argument("--remote-artifact-dir", default=None)
    harvest_component_ssh.add_argument("--mirror-dir", default=None)
    harvest_component_ssh.add_argument("--expected-baseline-archive-sha256", default=None)
    harvest_component_ssh.add_argument(
        "--expected-baseline-archive-size-bytes",
        "--expected-baseline-archive-bytes",
        dest="expected_baseline_archive_size_bytes",
        type=int,
        default=None,
    )
    harvest_component_ssh.add_argument("--require-passed", action="store_true")
    harvest_component_ssh.add_argument("--overwrite", action="store_true")
    harvest_component_ssh.add_argument("--allow-manual-artifact-dir", action="store_true")
    harvest_component_ssh.add_argument("--override-reason", default=None)
    harvest_component_ssh.add_argument("--ssh-bin", default="ssh")
    harvest_component_ssh.add_argument("--scp-bin", default="scp")
    harvest_component_ssh.add_argument(
        "--ssh-connect-timeout",
        type=int,
        default=15,
        help="BatchMode SSH auth preflight timeout in seconds.",
    )
    harvest_component_ssh.set_defaults(func=cmd_harvest_component_response_ssh)

    refresh = sub.add_parser("refresh-status")
    _add_state_arg(refresh)
    refresh_group = refresh.add_mutually_exclusive_group(required=True)
    refresh_group.add_argument("--job-name", default=None, help="Local queue/spec job name.")
    refresh_group.add_argument(
        "--all",
        action="store_true",
        help="Refresh every non-dry-run record in the local state file.",
    )
    refresh.add_argument(
        "--sdk-job-name",
        default=None,
        help="Lightning SDK job name. Defaults to --job-name with underscores replaced by hyphens.",
    )
    refresh.add_argument(
        "--include-dry-runs",
        action="store_true",
        help="Include dry-run records when used with --all.",
    )
    refresh.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return non-zero from --all if any individual refresh fails.",
    )
    refresh.add_argument("--teamspace", default=None)
    refresh.add_argument("--org", default=None)
    refresh.add_argument("--user", default=None)
    refresh.set_defaults(func=cmd_refresh_status)

    machines = sub.add_parser("list-machines")
    machines.add_argument("--teamspace", required=True)
    machines.add_argument("--org", default=None)
    machines.add_argument("--user", default=None)
    machines.add_argument("--cloud-account", action="append", default=[])
    machines.add_argument("--machine", default=None, help="Optional SDK machine filter, e.g. T4 or L4.")
    machines.add_argument("--gpu-only", action="store_true")
    machines.set_defaults(func=cmd_list_machines)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--json-out", default=None)
    doctor.add_argument("--run-id", default="lightning_doctor")
    doctor.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return nonzero when any attempted or required check fails.",
    )
    doctor.add_argument("--ssh-target", default=os.environ.get("LIGHTNING_SSH_TARGET"))
    doctor.add_argument("--ssh-bin", default="ssh")
    doctor.add_argument("--ssh-connect-timeout", type=int, default=15)
    doctor.add_argument("--require-ssh", action="store_true")
    doctor.add_argument(
        "--remote-supply-chain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run strict remote supply-chain scan when --ssh-target is present.",
    )
    doctor.add_argument("--require-remote-supply-chain", action="store_true")
    doctor.add_argument("--repo-dir", default=os.environ.get("LIGHTNING_REMOTE_PACT", "/teamspace/studios/this_studio/pact"))
    doctor.add_argument("--python-bin", default=os.environ.get("LIGHTNING_BATCH_PYTHON_BIN", ".venv/bin/python"))
    doctor.add_argument("--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE"))
    doctor.add_argument("--org", default=os.environ.get("LIGHTNING_ORG"))
    doctor.add_argument("--user", default=os.environ.get("LIGHTNING_SDK_USER"))
    doctor.add_argument(
        "--machine-inventory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Query Lightning GPU machine inventory when --teamspace is present.",
    )
    doctor.add_argument("--require-machine-inventory", action="store_true")
    doctor.add_argument("--cloud-account", action="append", default=[])
    doctor.add_argument("--machine", default=None, help="Optional SDK machine filter, e.g. T4 or L4.")
    doctor.add_argument("--gpu-only", action="store_true", default=True)
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
