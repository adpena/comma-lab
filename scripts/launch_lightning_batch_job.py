#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ruff: noqa: I001
"""Submit or dry-run official Lightning Batch Jobs for pact lanes/evals."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

# `lightning_sdk.__init__` performs a PyPI version check on import unless this
# is set. Keep exact-eval tooling deterministic and avoid package-index network
# touches on import, especially during supply-chain incidents.
os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")

from tac.deploy.lightning.batch_jobs import (  # noqa: E402
    ARTIFACT_INFRA_FAILURE,
    ARTIFACT_VALIDATION,
    LIGHTNING_BATCH_STATE,
    LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
    LightningAdjudicationSpec,
    LightningBatchJobsClient,
    LightningStudioCloudAccountMismatchError,
    archive_identity,
    make_diagnostic_component_sensitivity_spec,
    make_exact_eval_spec,
    make_official_component_response_spec,
    lightning_sdk_job_name,
    mirror_local_component_sensitivity_artifact_dir,
    mirror_local_component_response_artifact_dir,
    mirror_local_artifact_dir,
    mirror_ssh_component_sensitivity_artifact_dir,
    mirror_ssh_component_response_artifact_dir,
    validate_local_component_sensitivity_artifact_dir,
    validate_local_component_response_artifact_dir,
    validate_local_artifact_dir,
    validate_studio_machine_class_pair,
)
from tac.optimizer.exact_dispatch_authority import active_dispatch_claim_present  # noqa: E402
from tac.optimizer.exact_readiness import claim_status_terminal  # noqa: E402
from tac.public_submission_refs import parse_public_pr_refs_csv  # noqa: E402
from tac.repo_io import json_text, read_json, write_json  # noqa: E402

SSH_AUTH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "KbdInteractiveAuthentication=no",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "TCPKeepAlive=yes",
    "-o",
    "ConnectionAttempts=3",
)
SSH_TRANSIENT_FAILURE_PATTERNS = (
    "connection reset by peer",
    "connection timed out",
    "connection closed by",
    "connection refused",
    "kex_exchange_identification",
    "operation timed out",
    "read: connection reset",
)
SSH_TRANSIENT_RETRY_ATTEMPTS = 4
SSH_TRANSIENT_RETRY_INITIAL_DELAY_S = 2.0


def _print_json(payload: object) -> None:
    print(json_text(payload), end="")


def _write_json(path: str | Path, payload: object) -> None:
    write_json(path, payload)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_EXACT_EVAL_PRE_SCORE_FAILURES = (
    {
        "terminal_class": "archive_validator_whitelist_block",
        "failure_class": "archive_validation_failure_before_score",
        "score_source": "none:archive_validator_whitelist_block",
        "reason": (
            "auth_eval.log shows the archive validator rejected a member or "
            "extension before contest_auth_eval.json could be written"
        ),
        "needles": (
            ("archive validator", "whitelist"),
            ("archive validator", "allowlist"),
            ("not in", "whitelist"),
            ("not in", "allowlist"),
            ("unexpected archive member",),
            ("archive member", "not allowed"),
        ),
    },
    {
        "terminal_class": "pr86_constriction_hpac_invalid_entropy_model",
        "failure_class": "archive_runtime_decode_failure_before_score",
        "score_source": "none:pr86_constriction_hpac_invalid_entropy_model",
        "reason": (
            "auth_eval.log shows PR86/constriction HPAC rejected an invalid "
            "entropy model before scoring"
        ),
        "needles": (
            ("pr86", "hpac", "invalid entropy model"),
            ("constriction", "hpac", "invalid entropy model"),
            ("hpac", "invalid entropy model"),
        ),
    },
    {
        "terminal_class": "inflate_returncode_failure",
        "failure_class": "inflate_failure_before_score",
        "score_source": "none:inflate_returncode_failure",
        "reason": (
            "auth_eval.log shows inflate.sh or the inflate stage returned "
            "non-zero before contest_auth_eval.json could be written"
        ),
        "needles": (
            ("inflate", "returncode"),
            ("inflate", "return code"),
            ("inflate", "returned non-zero"),
            ("inflate.sh", "non-zero exit status"),
            ("inflate.sh", "returned non-zero"),
            ("inflate failed",),
        ),
    },
)

_AUTH_LOG_SNIPPET_BYTES = 4096

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

_STALE_REMOTE_ARG_HINTS = {
    "--remote": "--remote-preflight-ssh-target for submit preflight, or scripts/lightning_exact_eval_repro.py --remote at the wrapper layer",
    "--rmote": "--remote in scripts/lightning_exact_eval_repro.py, or --remote-preflight-ssh-target in this launcher",
    "--remote-preflight-target": "--remote-preflight-ssh-target",
    "--remote-ssh-target": "--remote-preflight-ssh-target",
    "--ssh": "--ssh-target for harvest/doctor, or --remote-preflight-ssh-target for submit preflight",
    "--ssh-alias": "--ssh-target for harvest/doctor, or --remote-preflight-ssh-target for submit preflight",
    "--required-device": "belongs to scripts/adjudicate_contest_auth_eval.py; launch_lightning_batch_job.py exact-eval emits it internally when --adjudicate is used",
    "--required-samples": "belongs to scripts/adjudicate_contest_auth_eval.py; launch_lightning_batch_job.py exact-eval emits it internally when --adjudicate is used",
}


def _iter_parser_option_strings(parser: argparse.ArgumentParser) -> list[str]:
    options: set[str] = set()
    stack = [parser]
    while stack:
        current = stack.pop()
        for action in current._actions:
            options.update(action.option_strings)
            if isinstance(action, argparse._SubParsersAction):
                stack.extend(action.choices.values())
    return sorted(options)


def _unknown_arg_diagnostic(message: str, parser: argparse.ArgumentParser) -> str | None:
    prefix = "unrecognized arguments:"
    if prefix not in message:
        return None
    unknown = [item for item in message.split(prefix, 1)[1].split() if item.startswith("-")]
    if not unknown:
        return None
    known = _iter_parser_option_strings(parser)
    lines = ["Strict argparse rejected unknown option(s); use the real parser surface below:"]
    for item in unknown:
        stale_hint = _STALE_REMOTE_ARG_HINTS.get(item)
        nearest = difflib.get_close_matches(item, known, n=3, cutoff=0.62)
        if stale_hint:
            lines.append(f"  {item}: {stale_hint}")
        elif nearest:
            lines.append(f"  {item}: did you mean {', '.join(nearest)}?")
        else:
            lines.append(f"  {item}: no close known option")
    lines.append("Known options include: " + ", ".join(known[:80]))
    if len(known) > 80:
        lines.append(f"... plus {len(known) - 80} more; run the subcommand with --help for the scoped surface.")
    return "\n".join(lines)


class StrictArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        diagnostic = _unknown_arg_diagnostic(message, self)
        if diagnostic:
            message = message + "\n" + diagnostic
        super().error(message)


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


def _queue_metadata_from_args(args: argparse.Namespace) -> dict[str, object]:
    metadata = _parse_metadata_kv(args.queue_metadata)
    source_prs = metadata.get("source_prs") or metadata.get("source_public_prs")
    if source_prs:
        refs = parse_public_pr_refs_csv(source_prs)
        metadata["source_prs"] = ",".join(refs.keys())
        metadata["source_pr_urls"] = ",".join(str(ref["url"]) for ref in refs.values())
    source_manifest = getattr(args, "source_manifest", None)
    if source_manifest:
        manifest_path = Path(source_manifest)
        if manifest_path.is_file():
            manifest = _load_json_object(manifest_path, label="source manifest")
            identity = _source_manifest_identity(manifest, manifest_path=manifest_path)
            metadata.setdefault("source_manifest", str(source_manifest))
            metadata["source_manifest_sha256"] = str(identity["source_manifest_sha256"])
            metadata["source_manifest_file_sha256"] = str(identity["source_manifest_file_sha256"])
            if (
                getattr(args, "cmd", None) == "exact-eval"
                and not getattr(args, "dry_run", False)
                and getattr(args, "studio", None)
            ):
                closure = _source_manifest_runtime_closure_from_args(
                    args,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    identity=identity,
                )
                metadata["source_manifest_runtime_closure_sha256"] = closure["closure_sha256"]
                metadata["source_manifest_runtime_closure"] = closure
    reason = getattr(args, "allow_skip_remote_preflight_reason", None)
    if reason:
        metadata["remote_preflight_skip_reason"] = str(reason).strip()
    claim_reason = getattr(args, "allow_missing_dispatch_claim_reason", None)
    if claim_reason:
        metadata["dispatch_claim_skip_reason"] = str(claim_reason).strip()
    return metadata


_DISPATCH_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")


def _dispatch_claim_status_is_terminal(status: str) -> bool:
    """Backward-compatible wrapper around the canonical claim terminal check."""

    return claim_status_terminal(status)


def _require_dispatch_claim_for_submit(args: argparse.Namespace, *, role: str) -> None:
    if getattr(args, "dry_run", False):
        return
    if not getattr(args, "studio", None):
        return
    metadata = _parse_metadata_kv(getattr(args, "queue_metadata", []) or [])
    lane_id = (getattr(args, "dispatch_lane_id", None) or metadata.get("lane") or metadata.get("lane_id") or "").strip()
    skip_reason = (getattr(args, "allow_missing_dispatch_claim_reason", None) or "").strip()
    if not lane_id:
        if skip_reason:
            return
        raise SystemExit(
            f"{role} Studio submit requires --dispatch-lane-id or --queue-metadata lane=... "
            "plus a matching active .omx/state/active_lane_dispatch_claims.md row"
        )
    claims_path = Path(getattr(args, "dispatch_claims_path", None) or _DISPATCH_CLAIMS_PATH)
    job_name = str(getattr(args, "job_name", "")).strip()
    acceptable_job_ids = {job_name}
    if job_name:
        acceptable_job_ids.add(lightning_sdk_job_name(job_name))
    if active_dispatch_claim_present(
        lane_id=lane_id,
        dispatch_claims_path=claims_path,
        platform="lightning",
        instance_job_ids=acceptable_job_ids,
    ):
        return
    if skip_reason:
        return
    raise SystemExit(
        f"{role} Studio submit blocked: missing active dispatch claim for lane_id={lane_id} "
        f"job_name={job_name} in {claims_path}. Run tools/claim_lane_dispatch.py claim first, "
        "or pass --allow-missing-dispatch-claim-reason for an auditable break-glass override."
    )


def _require_lightning_identity_for_studio_submit(args: argparse.Namespace, *, role: str) -> None:
    if getattr(args, "dry_run", False):
        return
    if not getattr(args, "studio", None):
        if getattr(args, "image", None):
            return
        raise SystemExit(
            f"{role} non-dry-run Lightning submit requires explicit --studio "
            "or --image. Do not rely on SDK autodetection; pass --studio, "
            "--teamspace, and --user/--org for Studio-backed jobs."
        )
    if not getattr(args, "teamspace", None):
        raise SystemExit(
            f"{role} Studio submit requires --teamspace. "
            "Pass --studio, --teamspace, and --user/--org explicitly so the "
            "Lightning SDK namespace is deterministic before remote preflight."
        )
    if getattr(args, "org", None) or getattr(args, "user", None):
        return
    raise SystemExit(
        f"{role} Studio submit with --teamspace requires --user or --org. "
        "Lightning SDK cannot resolve user-owned teamspaces from a bare teamspace name; "
        "pass --user <lightning-user> or --org <lightning-org> before remote preflight."
    )


def _state_path(args: argparse.Namespace) -> Path | None:
    return Path(args.state_path) if args.state_path else None


def _client(args: argparse.Namespace) -> LightningBatchJobsClient:
    state_path = _state_path(args)
    if state_path is None:
        return LightningBatchJobsClient()
    return LightningBatchJobsClient(state_path=state_path)


def _submit_lightning_or_exit(
    client: LightningBatchJobsClient,
    spec: object,
    *,
    dry_run: bool,
) -> dict[str, object]:
    try:
        return client.submit(spec, dry_run=dry_run)
    except LightningStudioCloudAccountMismatchError as exc:
        raise SystemExit(str(exc)) from None


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
        required_device=args.eval_device,
        allow_component_gate_forensic_success=(
            args.allow_component_gate_forensic_success
            and not args.fail_job_on_component_gate
        ),
        allow_sane_score_forensic_success=(
            args.allow_sane_score_forensic_success
            and not args.fail_job_on_sane_score_gate
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
    return set(_manifest_entries_by_path(manifest, manifest_path=manifest_path))


def _manifest_entries_by_path(
    manifest: dict[str, object],
    *,
    manifest_path: Path,
) -> dict[str, dict[str, object]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SystemExit(f"source manifest missing files list: {manifest_path}")
    entries: dict[str, dict[str, object]] = {}
    for index, item in enumerate(files):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise SystemExit(f"source manifest files[{index}].path must be a string: {manifest_path}")
        rel = _safe_remote_repo_rel(str(item["path"]), field=f"source manifest files[{index}].path")
        if rel in entries:
            raise SystemExit(f"source manifest contains duplicate path: {rel}")
        entries[rel] = dict(item)
    return entries


def _source_manifest_identity(
    manifest: dict[str, object],
    *,
    manifest_path: Path,
) -> dict[str, object]:
    declared = manifest.get("manifest_sha256")
    without_self = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = _canonical_json_sha256(without_self)
    if declared is not None:
        if not isinstance(declared, str) or not _SHA256_HEX_RE.fullmatch(declared):
            raise SystemExit(f"source manifest has invalid manifest_sha256: {manifest_path}")
        if declared != canonical:
            raise SystemExit(
                "source manifest manifest_sha256 is stale: "
                f"{manifest_path} expected={declared} actual={canonical}"
            )
    return {
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": declared or canonical,
        "source_manifest_file_sha256": _sha256_file(manifest_path),
        "source_manifest_file_count": manifest.get("file_count"),
        "source_manifest_total_bytes": manifest.get("total_bytes"),
    }


def _manifest_entry_byte_sha(
    entries: dict[str, dict[str, object]],
    rel: str,
    *,
    manifest_path: Path,
) -> dict[str, object]:
    entry = entries.get(rel)
    if entry is None:
        raise SystemExit(
            "exact-eval submit blocked; staged source manifest does not include "
            f"required byte/SHA artifact: {rel}"
        )
    byte_count = entry.get("bytes")
    sha256 = entry.get("sha256")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise SystemExit(
            "exact-eval submit blocked; source manifest entry lacks integer bytes "
            f"for {rel}: {manifest_path}"
        )
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.fullmatch(sha256):
        raise SystemExit(
            "exact-eval submit blocked; source manifest entry lacks sha256 "
            f"for {rel}: {manifest_path}"
        )
    local_path = Path(REPO_ROOT) / rel
    if local_path.is_file():
        actual_bytes = local_path.stat().st_size
        actual_sha = _sha256_file(local_path)
        if actual_bytes != byte_count or actual_sha != sha256:
            raise SystemExit(
                "exact-eval submit blocked; source manifest byte/SHA mismatch "
                f"for {rel}: expected bytes={byte_count} sha256={sha256} "
                f"actual bytes={actual_bytes} sha256={actual_sha}"
            )
    return {"path": rel, "bytes": byte_count, "sha256": sha256}


def _source_manifest_runtime_closure_from_args(
    args: argparse.Namespace,
    *,
    manifest: dict[str, object],
    manifest_path: Path,
    identity: dict[str, object] | None = None,
) -> dict[str, object]:
    entries = _manifest_entries_by_path(manifest, manifest_path=manifest_path)
    archive_rel = _remote_repo_rel(args.archive, repo_dir=args.repo_dir)
    if archive_rel is None:
        raise SystemExit("--archive must be inside --repo-dir for exact-eval Studio submit")
    runtime_reqs = _exact_eval_runtime_requirements(args)
    metadata = _parse_metadata_kv(getattr(args, "queue_metadata", []) or [])
    optional_reqs: set[str] = set()
    for key in ("baseline_json", "baseline_contest_auth_eval_json", "archive_manifest", "cdo1_manifest"):
        value = metadata.get(key)
        if not value:
            continue
        rel = _remote_repo_rel(value, repo_dir=args.repo_dir)
        if rel is not None:
            optional_reqs.add(rel)
    required_paths = {archive_rel, *runtime_reqs, *optional_reqs}
    files = [
        _manifest_entry_byte_sha(entries, rel, manifest_path=manifest_path)
        for rel in sorted(required_paths)
    ]
    runtime_files = [
        _manifest_entry_byte_sha(entries, rel, manifest_path=manifest_path)
        for rel in sorted(runtime_reqs)
    ]
    inflate_rel = _remote_repo_rel(_default_exact_eval_inflate_sh(args), repo_dir=args.repo_dir)
    external_roots = _declared_runtime_dependency_roots(inflate_rel or "")
    identity = identity or _source_manifest_identity(manifest, manifest_path=manifest_path)
    payload: dict[str, object] = {
        "schema": "source_manifest_runtime_closure_v1",
        **identity,
        "archive": _manifest_entry_byte_sha(entries, archive_rel, manifest_path=manifest_path),
        "inflate_sh": inflate_rel,
        "declared_dependency_roots": external_roots,
        "required_path_count": len(files),
        "runtime_file_count": len(runtime_files),
        "runtime_total_bytes": sum(int(item["bytes"]) for item in runtime_files),
        "files": files,
        "runtime_files": runtime_files,
    }
    closure_basis = {key: value for key, value in payload.items() if key != "closure_sha256"}
    payload["closure_sha256"] = _canonical_json_sha256(closure_basis)
    return payload


def _default_exact_eval_inflate_sh(args: argparse.Namespace) -> str:
    return str(getattr(args, "inflate_sh", None) or "submissions/robust_current/inflate.sh")


def _declared_runtime_dependency_roots(inflate_rel: str) -> list[str]:
    """Resolve repo-relative PACT_RUNTIME_DEPENDENCY_ROOT literals for submit gates."""
    inflate_path = Path(REPO_ROOT) / inflate_rel
    if not inflate_path.is_file():
        return []
    roots: list[str] = []
    for line in inflate_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped[1:].strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if not stripped.startswith("PACT_RUNTIME_DEPENDENCY_ROOT="):
            continue
        rhs = stripped.split("=", 1)[1].strip()
        try:
            parsed = shlex.split(rhs, comments=True, posix=True)
        except ValueError as exc:
            raise SystemExit(
                "exact-eval submit blocked; invalid PACT_RUNTIME_DEPENDENCY_ROOT "
                f"shell literal in {inflate_rel}: {exc}"
            ) from exc
        if not parsed:
            raise SystemExit(
                f"exact-eval submit blocked; empty PACT_RUNTIME_DEPENDENCY_ROOT in {inflate_rel}"
            )
        raw_root = parsed[0]
        if "$" in raw_root or "`" in raw_root:
            raise SystemExit(
                "exact-eval submit blocked; PACT_RUNTIME_DEPENDENCY_ROOT must be a "
                f"repo-relative or repo-absolute literal, got {raw_root!r} in {inflate_rel}"
            )
        root_rel = _repo_rel(raw_root)
        root_dir = Path(REPO_ROOT) / root_rel
        if not root_dir.is_dir():
            raise SystemExit(
                "exact-eval submit blocked; PACT_RUNTIME_DEPENDENCY_ROOT does not "
                f"exist or is not a directory: {root_rel}"
            )
        roots.append(_safe_remote_repo_rel(root_rel, field="PACT_RUNTIME_DEPENDENCY_ROOT"))
    return list(dict.fromkeys(roots))


def _exact_eval_runtime_requirements(args: argparse.Namespace) -> set[str]:
    required: set[str] = set()
    inflate_rel = _remote_repo_rel(_default_exact_eval_inflate_sh(args), repo_dir=args.repo_dir)
    if inflate_rel is None:
        raise SystemExit("--inflate-sh must be inside --repo-dir for exact-eval Studio submit")
    required.add(inflate_rel)
    inflate_path = PurePosixPath(inflate_rel)
    local_config_env = Path(REPO_ROOT) / str(inflate_path.parent / "config.env")
    if inflate_rel == "submissions/robust_current/inflate.sh" or local_config_env.is_file():
        required.add(_safe_remote_repo_rel(str(inflate_path.parent / "config.env"), field="inflate config.env"))
    if inflate_rel != "submissions/robust_current/inflate.sh":
        runtime_dir = Path(REPO_ROOT) / str(inflate_path.parent)
        if runtime_dir.is_dir():
            for child in sorted(runtime_dir.rglob("*")):
                if not child.is_file():
                    continue
                rel_parts = child.relative_to(runtime_dir).parts
                if any(
                    part.startswith(".") or part.startswith("._") or part == "__pycache__"
                    for part in rel_parts
                ):
                    continue
                required.add(
                    _safe_remote_repo_rel(
                        str(PurePosixPath(inflate_rel).parent / PurePosixPath(*rel_parts)),
                        field=f"external inflate runtime file {'/'.join(rel_parts)}",
                    )
                )
        for root_rel in _declared_runtime_dependency_roots(inflate_rel):
            root_dir = Path(REPO_ROOT) / root_rel
            for child in sorted(root_dir.rglob("*")):
                if not child.is_file():
                    continue
                rel_parts = child.relative_to(root_dir).parts
                if any(
                    part.startswith(".") or part.startswith("._") or part == "__pycache__"
                    for part in rel_parts
                ):
                    continue
                required.add(
                    _safe_remote_repo_rel(
                        str(PurePosixPath(root_rel) / PurePosixPath(*rel_parts)),
                        field=f"PACT_RUNTIME_DEPENDENCY_ROOT file {'/'.join(rel_parts)}",
                    )
                )
    return required


_SOURCE_EMBEDDED_PAYLOAD_LITERAL_RE = re.compile(
    r"(?:b64decode|b85decode|a85decode|brotli\.decompress|lzma\.decompress|zlib\.decompress)\s*\(\s*([rubfRUBF]*[\"'])(?P<payload>.{65536,}?)(?<!\\)\1",
    re.DOTALL,
)


def _validate_no_source_embedded_payload_runtime(
    args: argparse.Namespace,
    *,
    archive_rel: str,
    runtime_rels: set[str],
) -> None:
    """Block public-replay exact evals that move charged payload into source.

    Public PR intake often replays external inflate runtimes for forensics. That
    is useful, but a tiny archive plus a giant base85/base64 Python literal is
    not a contest-faithful archive candidate for this repo: the score-affecting
    bytes are in runtime source, not archive.zip. Keep this guard in the submit
    path so a loophole replay cannot accidentally become promotion evidence.
    """
    if _remote_repo_rel(_default_exact_eval_inflate_sh(args), repo_dir=args.repo_dir) == "submissions/robust_current/inflate.sh":
        return
    waiver = str(getattr(args, "allow_source_embedded_payload_runtime_reason", "") or "").strip()
    archive_path = Path(REPO_ROOT) / archive_rel
    try:
        archive_bytes = archive_path.stat().st_size
    except OSError:
        archive_bytes = 0
    violations: list[str] = []
    runtime_source_bytes = 0
    for rel in sorted(runtime_rels):
        path = Path(REPO_ROOT) / rel
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh"}:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        runtime_source_bytes += len(raw)
        if path.suffix.lower() == ".py":
            text = raw.decode("utf-8", errors="ignore")
            match = _SOURCE_EMBEDDED_PAYLOAD_LITERAL_RE.search(text)
            if match:
                violations.append(
                    f"{rel} contains a >=64KiB encoded/decompressed string literal"
                )
    if archive_bytes and archive_bytes <= 1024 and runtime_source_bytes > 64 * 1024:
        violations.append(
            f"archive is {archive_bytes} bytes but external inflate source is "
            f"{runtime_source_bytes} bytes"
        )
    if violations and not waiver:
        raise SystemExit(
            "exact-eval submit blocked; external inflate runtime appears to "
            "carry source-embedded payload bytes instead of charging them in "
            "archive.zip: "
            + "; ".join(violations)
            + ". Quarantine as invalid/external or pass "
            "--allow-source-embedded-payload-runtime-reason with an auditable "
            "forensic-only reason."
        )


def _validate_t4_exact_eval_runtime_env(args: argparse.Namespace) -> None:
    machine = str(getattr(args, "machine", "") or "").lower()
    if "t4" not in machine and "g4dn" not in machine:
        return
    env = _parse_env_kv(getattr(args, "env", []) or [])
    torch_spec = str(env.get("INFLATE_TORCH_SPEC", "")).strip()
    if not torch_spec:
        raise SystemExit(
            "T4/g4dn exact-eval submit blocked; pass --env INFLATE_TORCH_SPEC=... "
            "so inflate-side torch cannot resolve to a CUDA-13 wheel on an older driver"
        )
    if "+cu124" in torch_spec:
        torchvision_spec = str(env.get("INFLATE_TORCHVISION_SPEC", "")).strip()
        if "+cu124" not in torchvision_spec:
            raise SystemExit(
                "T4/g4dn exact-eval submit blocked; cu124 torch pin requires "
                "--env INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124 "
                "so upstream scorer imports do not mix torch and torchvision builds"
            )
        extra_index = str(env.get("UV_EXTRA_INDEX_URL", "")).strip()
        strategy = str(env.get("UV_INDEX_STRATEGY", "")).strip()
        if "download.pytorch.org/whl/cu124" not in extra_index or strategy != "unsafe-best-match":
            raise SystemExit(
                "T4/g4dn exact-eval submit blocked; cu124 torch pin requires "
                "--env UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 "
                "and --env UV_INDEX_STRATEGY=unsafe-best-match"
            )


def _validate_exact_eval_archive_payload_preflight(args: argparse.Namespace) -> None:
    inflate_rel = _remote_repo_rel(_default_exact_eval_inflate_sh(args), repo_dir=args.repo_dir)
    if inflate_rel != "submissions/robust_current/inflate.sh":
        return
    archive = Path(args.archive)
    if not archive.is_file():
        archive_rel = _remote_repo_rel(args.archive, repo_dir=args.repo_dir)
        if archive_rel is not None:
            archive = Path(REPO_ROOT) / archive_rel
    if not archive.is_file():
        return
    from tac.submission_archive import validate_archive_seg_tile_actions_payloads

    try:
        errors = validate_archive_seg_tile_actions_payloads(archive)
    except Exception as exc:
        raise SystemExit(
            "exact-eval submit blocked; seg_tile_actions preflight crashed before GPU dispatch: "
            f"{exc}"
        ) from exc
    if errors:
        detail = "; ".join(errors[:8])
        raise SystemExit(
            "exact-eval submit blocked; seg_tile_actions preflight rejected candidate "
            f"before GPU dispatch: {detail}"
        )


def _validate_archive_manifest_dispatch_gate(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    if not archive.is_file():
        archive_rel = _remote_repo_rel(args.archive, repo_dir=args.repo_dir)
        if archive_rel is not None:
            archive = Path(REPO_ROOT) / archive_rel
    if not archive.is_file():
        return
    manifest_path = archive.with_name("manifest.json")
    if not manifest_path.is_file():
        return
    manifest = _load_json_object(manifest_path, label="candidate archive manifest")
    gate = manifest.get("exact_eval_dispatch_gate")
    if not isinstance(gate, dict):
        return
    if gate.get("required") is not True:
        return
    if gate.get("safe_for_exact_eval_dispatch") is True:
        return
    status = gate.get("status") or "unknown"
    blockers = gate.get("blockers") or []
    blocker_text = (
        ", ".join(str(item) for item in blockers[:8])
        if isinstance(blockers, list)
        else str(blockers)
    )
    raise SystemExit(
        "exact-eval submit blocked; candidate archive manifest exact_eval_dispatch_gate "
        f"is not safe_for_exact_eval_dispatch=true (status={status}). {blocker_text}"
    )


_STUDIO_SYMBOLIC_MACHINE_SUGGESTIONS = {
    "H100": "use a concrete Studio-compatible provider class, or submit an image-backed job on a matching cloud account",
    "H200": "use a concrete Studio-compatible provider class, or submit an image-backed job on a matching cloud account",
    "A100": "use a concrete Studio-compatible provider class, or submit an image-backed job on a matching cloud account",
    "A100_SXM4": "use a concrete Studio-compatible provider class, or submit an image-backed job on a matching cloud account",
    "L40S": "use g6e.4xlarge for the current Studio-backed AWS L40S route",
    "RTXP_6000": "use g7e.4xlarge for the current Studio-backed RTX PRO route",
    "RTXP_6000_X_2": "use g7e.12xlarge for the current Studio-backed dual RTX PRO route",
}
_STUDIO_GCP_MACHINE_PREFIXES = (
    "a2-",
    "a3-",
    "a4-",
    "g2-",
    "g4-standard-",
    "n1-",
    "v5",
    "v6",
)


def _validate_studio_machine_for_submit(args: argparse.Namespace) -> None:
    if getattr(args, "dry_run", False) or not getattr(args, "studio", None):
        return
    machine = str(getattr(args, "machine", "") or "").strip()
    if not machine:
        return
    key = machine.upper()
    if key in {"T4", "T4_SMALL"}:
        return
    suggestion = _STUDIO_SYMBOLIC_MACHINE_SUGGESTIONS.get(key)
    if suggestion:
        raise SystemExit(
            "Studio exact-eval submit blocked; symbolic accelerator "
            f"{machine!r} can be accepted by inventory but rejected by the "
            "Studio provider cluster at SDK submit time. "
            f"{suggestion}. Confirm with `launch_lightning_batch_job.py "
            "list-machines --teamspace <teamspace> --user <user> --gpu-only`."
        )
    lower = machine.lower()
    if lower.startswith(_STUDIO_GCP_MACHINE_PREFIXES) and not getattr(args, "cloud_account", None):
        raise SystemExit(
            "Studio exact-eval submit blocked; GCP machine "
            f"{machine!r} requires an explicit --cloud-account route. Without "
            "that route the Lightning SDK may submit through the default AWS "
            "cluster and fail before job creation."
        )
    try:
        validate_studio_machine_class_pair(
            machine,
            cloud_account=getattr(args, "cloud_account", None),
        )
    except ValueError as exc:
        raise SystemExit(f"Studio exact-eval submit blocked; {exc}") from exc


def _require_remote_preflight_for_submit(args: argparse.Namespace, *, role: str) -> None:
    if args.dry_run:
        return
    if not args.studio:
        return
    blocker = _ssh_target_shape_blocker(
        getattr(args, "remote_preflight_ssh_target", None),
        flag="--remote-preflight-ssh-target",
    )
    if blocker is None:
        return
    reason = str(getattr(args, "allow_skip_remote_preflight_reason", "") or "").strip()
    if reason:
        if len(reason) < 12:
            raise SystemExit("--allow-skip-remote-preflight-reason must be a specific auditable reason")
        return
    raise SystemExit(
        f"{role} non-dry-run Studio submit blocked: {blocker}. "
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
        payload = read_json(path)
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
    probe = _run_ssh_command_with_retries(
        [
            ssh_bin,
            *SSH_AUTH_OPTIONS,
            "-o",
            f"ConnectTimeout={int(connect_timeout)}",
            target,
            "true",
        ],
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
        lines.append(
            "ssh -G failed; this usually means the SSH alias is absent from ~/.ssh/config "
            "or the target hostname is misspelled."
        )
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


def _ssh_target_shape_blocker(target: str | None, *, flag: str) -> str | None:
    value = str(target or "").strip()
    if not value:
        return f"missing {flag}; non-dry-run Studio submit needs an SSH alias for remote preflight"
    if any(ch in value for ch in "\r\n\0"):
        return f"{flag} contains control characters"
    if value == "ssh.lightning.ai":
        return f"{flag} must be a ~/.ssh/config alias or user-qualified target, not bare ssh.lightning.ai"
    return None


def _remote_submit_readiness(args: argparse.Namespace, *, role: str) -> dict[str, object]:
    blockers: list[str] = []
    warnings: list[str] = []
    if getattr(args, "studio", None):
        blocker = _ssh_target_shape_blocker(
            getattr(args, "remote_preflight_ssh_target", None),
            flag="--remote-preflight-ssh-target",
        )
        skip_reason = str(getattr(args, "allow_skip_remote_preflight_reason", "") or "").strip()
        if blocker and skip_reason:
            warnings.append(f"{blocker}; break-glass skip reason is present")
        elif blocker:
            blockers.append(blocker)
    return {
        "ok": not blockers,
        "role": role,
        "blockers": blockers,
        "warnings": warnings,
        "required_ssh_flag": "--remote-preflight-ssh-target",
    }


def _safe_artifact_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "lightning_batch_job"


def _is_transient_ssh_failure(result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode == 0:
        return False
    if result.returncode != 255:
        return False
    combined = (str(result.stderr or "") + "\n" + str(result.stdout or "")).lower()
    return any(pattern in combined for pattern in SSH_TRANSIENT_FAILURE_PATTERNS)


def _run_ssh_command_with_retries(
    cmd: list[str],
    *,
    attempts: int = SSH_TRANSIENT_RETRY_ATTEMPTS,
    initial_delay_s: float = SSH_TRANSIENT_RETRY_INITIAL_DELAY_S,
) -> subprocess.CompletedProcess[str]:
    last: subprocess.CompletedProcess[str] | None = None
    delay = float(initial_delay_s)
    for attempt in range(max(1, int(attempts))):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        last = result
        if result.returncode == 0 or not _is_transient_ssh_failure(result):
            return result
        if attempt + 1 < attempts:
            time.sleep(delay)
            delay = min(delay * 2.0, 15.0)
    assert last is not None
    return last


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
    result = _run_ssh_command_with_retries(
        [
            ssh_bin,
            *SSH_AUTH_OPTIONS,
            "-o",
            f"ConnectTimeout={int(connect_timeout)}",
            target,
            command,
        ],
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
    result = _run_ssh_command_with_retries(
        [
            ssh_bin,
            *SSH_AUTH_OPTIONS,
            "-o",
            f"ConnectTimeout={int(connect_timeout)}",
            ssh_target,
            command,
        ],
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
    payload = read_json(plan_path)
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


def _validate_component_sensitivity_submit_inputs(args: argparse.Namespace) -> None:
    if args.dry_run and not args.source_manifest:
        return
    if not args.source_manifest:
        raise SystemExit(
            "component-sensitivity submit requires --source-manifest from "
            "scripts/lightning_repro_workspace.py so the baseline archive and "
            "profiling inputs are known staged artifacts; use --dry-run for a "
            "topology-only check"
        )
    manifest_path = Path(args.source_manifest)
    manifest = _load_json_object(manifest_path, label="source manifest")
    manifest_paths = _manifest_repo_paths(manifest, manifest_path=manifest_path)

    required: set[str] = set()
    baseline_rel = _remote_repo_rel(args.baseline_archive, repo_dir=args.repo_dir)
    if baseline_rel is None:
        raise SystemExit("--baseline-archive must be inside --repo-dir for component-sensitivity submit")
    required.add(baseline_rel)

    video_path = args.video or f"{str(args.upstream_dir).rstrip('/')}/videos/0.mkv"
    video_rel = _remote_repo_rel(video_path, repo_dir=args.repo_dir)
    if video_rel is None:
        raise SystemExit("--video must be inside --repo-dir for component-sensitivity submit")
    required.add(video_rel)

    if args.pair_weights:
        pair_weights_rel = _remote_repo_rel(args.pair_weights, repo_dir=args.repo_dir)
        if pair_weights_rel is None:
            raise SystemExit("--pair-weights must be inside --repo-dir for component-sensitivity submit")
        required.add(pair_weights_rel)

    missing = sorted(required.difference(manifest_paths))
    if missing:
        raise SystemExit(
            "component-sensitivity submit blocked; staged source manifest does not include "
            "all required profiling artifacts: " + ", ".join(missing[:20])
        )


def _validate_exact_eval_submit_inputs(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    if not args.studio:
        return
    _validate_studio_machine_for_submit(args)
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
    _source_manifest_identity(manifest, manifest_path=manifest_path)
    manifest_entries = _manifest_entries_by_path(manifest, manifest_path=manifest_path)
    manifest_paths = set(manifest_entries)
    if archive_rel not in manifest_paths:
        raise SystemExit(
            "exact-eval submit blocked; staged source manifest does not include "
            f"archive artifact: {archive_rel}"
        )
    _manifest_entry_byte_sha(manifest_entries, archive_rel, manifest_path=manifest_path)
    runtime_reqs = _exact_eval_runtime_requirements(args)
    runtime_missing = sorted(runtime_reqs.difference(manifest_paths))
    if runtime_missing:
        raise SystemExit(
            "exact-eval submit blocked; staged source manifest does not include "
            "inflate runtime closure: " + ", ".join(runtime_missing)
        )
    for rel in sorted(runtime_reqs):
        _manifest_entry_byte_sha(manifest_entries, rel, manifest_path=manifest_path)
    _validate_no_source_embedded_payload_runtime(
        args,
        archive_rel=archive_rel,
        runtime_rels=runtime_reqs,
    )
    _validate_exact_eval_archive_payload_preflight(args)
    _validate_archive_manifest_dispatch_gate(args)
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
        _manifest_entry_byte_sha(manifest_entries, baseline_json_rel, manifest_path=manifest_path)
    archive_manifest = metadata.get("archive_manifest") or metadata.get("cdo1_manifest")
    if archive_manifest:
        archive_manifest_rel = _remote_repo_rel(archive_manifest, repo_dir=args.repo_dir)
        if archive_manifest_rel is None:
            raise SystemExit("exact-eval metadata archive_manifest must be inside --repo-dir")
        if archive_manifest_rel not in manifest_paths:
            raise SystemExit(
                "exact-eval submit blocked; staged source manifest does not include "
                f"metadata archive_manifest artifact: {archive_manifest_rel}"
            )
        _manifest_entry_byte_sha(manifest_entries, archive_manifest_rel, manifest_path=manifest_path)
        _validate_score_payload_manifest_metadata(
            Path(REPO_ROOT) / archive_manifest_rel,
            metadata=metadata,
        )
    _validate_t4_exact_eval_runtime_env(args)


def _manifest_cdo1_overlay(manifest: dict[str, object]) -> dict[str, object] | None:
    overlay = manifest.get("cdo1_overlay")
    if isinstance(overlay, dict):
        return overlay
    archive_report = manifest.get("archive_report")
    if isinstance(archive_report, dict):
        nested = archive_report.get("cdo1_overlay")
        if isinstance(nested, dict):
            return nested
    return None


def _validate_score_payload_manifest_metadata(
    manifest_path: Path,
    *,
    metadata: dict[str, str],
) -> None:
    if not manifest_path.is_file():
        raise SystemExit(f"metadata archive_manifest is not readable locally: {manifest_path}")
    manifest = _load_json_object(manifest_path, label="score payload archive manifest")
    cdo1_overlay = _manifest_cdo1_overlay(manifest)
    if cdo1_overlay is None:
        return
    pair_index_basis = cdo1_overlay.get("pair_index_basis")
    if not isinstance(pair_index_basis, str) or not pair_index_basis:
        raise SystemExit(
            "exact-eval submit blocked; CDO1 archive manifest must record "
            "cdo1_overlay.pair_index_basis"
        )
    if pair_index_basis not in {"half_frame_pair_index", "video_frame_pair_index"}:
        raise SystemExit(
            "exact-eval submit blocked; CDO1 archive manifest has invalid "
            f"pair_index_basis={pair_index_basis!r}"
        )
    selected_pairs = cdo1_overlay.get("selected_pair_indices")
    if (
        not isinstance(selected_pairs, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in selected_pairs)
    ):
        raise SystemExit(
            "exact-eval submit blocked; CDO1 archive manifest must record "
            "selected_pair_indices as integers"
        )
    metadata_basis = metadata.get("pair_index_basis")
    if metadata_basis and metadata_basis != pair_index_basis:
        raise SystemExit(
            "exact-eval submit blocked; queue_metadata pair_index_basis does not "
            f"match archive manifest: {metadata_basis!r} != {pair_index_basis!r}"
        )


def cmd_exact_eval(args: argparse.Namespace) -> int:
    if not args.adjudicate:
        raise SystemExit(
            "exact-eval requires --adjudicate so every Lightning result has "
            "contest_auth_eval.json custody plus adjudication provenance"
        )
    _validate_exact_eval_submit_inputs(args)
    _require_dispatch_claim_for_submit(args, role="exact-eval")
    _require_lightning_identity_for_studio_submit(args, role="exact-eval")
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
        cloud_account=args.cloud_account,
        expected_archive_sha256=expected_sha,
        expected_archive_size_bytes=expected_bytes,
        queue_metadata=_queue_metadata_from_args(args),
        local_artifact_dir=args.local_artifact_dir,
        adjudication=_adjudication_from_args(args),
        component_trace=args.component_trace,
        component_trace_top_k=args.component_trace_top_k,
        eval_device=args.eval_device,
        inflate_sh=args.inflate_sh,
    )
    client = _client(args)
    record = _submit_lightning_or_exit(client, spec, dry_run=args.dry_run)
    if args.dry_run:
        record["submit_readiness"] = _remote_submit_readiness(args, role="exact-eval")
    _print_json(record)
    return 0


def cmd_component_response(args: argparse.Namespace) -> int:
    _validate_component_response_submit_inputs(args)
    _validate_studio_machine_for_submit(args)
    _require_dispatch_claim_for_submit(args, role="component-response")
    _require_lightning_identity_for_studio_submit(args, role="component-response")
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
        cloud_account=args.cloud_account,
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
    record = _submit_lightning_or_exit(client, spec, dry_run=args.dry_run)
    if args.dry_run:
        record["submit_readiness"] = _remote_submit_readiness(args, role="component-response")
    _print_json(record)
    return 0


def cmd_component_sensitivity(args: argparse.Namespace) -> int:
    _validate_component_sensitivity_submit_inputs(args)
    _validate_studio_machine_for_submit(args)
    _require_dispatch_claim_for_submit(args, role="component-sensitivity")
    _require_lightning_identity_for_studio_submit(args, role="component-sensitivity")
    _require_remote_preflight_for_submit(args, role="component-sensitivity")
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
    spec = make_diagnostic_component_sensitivity_spec(
        name=args.job_name,
        baseline_archive_path=args.baseline_archive,
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
        cloud_account=args.cloud_account,
        video_mkv=args.video,
        pair_weights_path=args.pair_weights,
        expected_baseline_archive_sha256=expected_sha,
        expected_baseline_archive_size_bytes=expected_bytes,
        queue_metadata=_queue_metadata_from_args(args),
        local_artifact_dir=args.local_artifact_dir,
        top_k_pairs=args.top_k_pairs,
        pair_batch=args.pair_batch,
        response_top_k=args.response_top_k,
        response_epsilons=args.response_epsilons,
        split_seed=args.split_seed,
        holdout_fraction=args.holdout_fraction,
        aggregate=args.aggregate,
        promotion_finite_difference=args.promotion_finite_difference,
        finite_difference_epsilon=args.finite_difference_epsilon,
        finite_difference_shard_index=args.finite_difference_shard_index,
        finite_difference_shard_count=args.finite_difference_shard_count,
    )
    client = _client(args)
    record = _submit_lightning_or_exit(client, spec, dry_run=args.dry_run)
    if args.dry_run:
        record["submit_readiness"] = _remote_submit_readiness(args, role="component-sensitivity")
    _print_json(record)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    client = _client(args)
    _print_json(client.list_records())
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
    _print_json(result)
    return 0


def cmd_validate_component_response_artifacts(args: argparse.Namespace) -> int:
    result = validate_local_component_response_artifact_dir(
        args.artifact_dir,
        expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
        require_passed=args.require_passed,
    )
    validation_path = Path(args.artifact_dir) / "official_component_response_artifact_validation.json"
    _write_json(validation_path, result)
    _print_json(result)
    return 0


def cmd_validate_component_sensitivity_artifacts(args: argparse.Namespace) -> int:
    result = validate_local_component_sensitivity_artifact_dir(
        args.artifact_dir,
        expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
    )
    validation_path = Path(args.artifact_dir) / "diagnostic_component_sensitivity_artifact_validation.json"
    _write_json(validation_path, result)
    _print_json(result)
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
    _print_json(result)
    return 0


def cmd_harvest_component_sensitivity_local(args: argparse.Namespace) -> int:
    if args.mirror_dir:
        result = mirror_local_component_sensitivity_artifact_dir(
            args.artifact_dir,
            args.mirror_dir,
            expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
            expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
            overwrite=args.overwrite,
        )
    else:
        result = validate_local_component_sensitivity_artifact_dir(
            args.artifact_dir,
            expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
            expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
        )
    _print_json(result)
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
        _print_json(result)
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
    _print_json(result)
    return 0


def cmd_harvest_component_sensitivity_ssh(args: argparse.Namespace) -> int:
    _require_manual_artifact_override(args, role="component-sensitivity")
    if args.job_name:
        client = _client(args)
        _ensure_ssh_auth_ready(
            args.ssh_target,
            ssh_bin=args.ssh_bin,
            connect_timeout=args.ssh_connect_timeout,
        )
        result = client.harvest_ssh_component_sensitivity_artifacts(
            job_name=args.job_name,
            ssh_target=args.ssh_target,
            remote_dir=args.remote_artifact_dir,
            mirror_dir=args.mirror_dir,
            expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
            expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
            overwrite=args.overwrite,
            ssh_bin=args.ssh_bin,
            scp_bin=args.scp_bin,
            ssh_connect_timeout=args.ssh_connect_timeout,
        )
        _print_json(result)
        return 0
    _ensure_ssh_auth_ready(
        args.ssh_target,
        ssh_bin=args.ssh_bin,
        connect_timeout=args.ssh_connect_timeout,
    )
    if not args.remote_artifact_dir or not args.mirror_dir:
        raise SystemExit(
            "harvest-component-sensitivity-ssh requires either --job-name with state, "
            "or both --remote-artifact-dir and --mirror-dir"
        )
    result = mirror_ssh_component_sensitivity_artifact_dir(
        ssh_target=args.ssh_target,
        remote_dir=args.remote_artifact_dir,
        mirror_dir=args.mirror_dir,
        expected_baseline_archive_sha256=args.expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=args.expected_baseline_archive_size_bytes,
        overwrite=args.overwrite,
        ssh_bin=args.ssh_bin,
        scp_bin=args.scp_bin,
        ssh_connect_timeout=args.ssh_connect_timeout,
    )
    _print_json(result)
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
    _print_json(result)
    return 0


def _log_matches_needles(log_lower: str, needles: tuple[tuple[str, ...], ...]) -> bool:
    return any(all(needle in log_lower for needle in group) for group in needles)


def _auth_log_tail_snippet(log_text: str) -> str:
    encoded = log_text.encode("utf-8", errors="replace")
    if len(encoded) <= _AUTH_LOG_SNIPPET_BYTES:
        return log_text
    return encoded[-_AUTH_LOG_SNIPPET_BYTES:].decode("utf-8", errors="replace")


def classify_exact_eval_missing_json_failure_from_auth_log(
    log_text: str,
) -> dict[str, object] | None:
    """Classify precise pre-score exact-eval failures from auth_eval.log.

    This never parses scores from human logs. It only refines missing
    ``contest_auth_eval.json`` diagnostics when the log identifies why scoring
    never reached JSON custody.
    """

    log_lower = log_text.lower()
    for failure in _EXACT_EVAL_PRE_SCORE_FAILURES:
        needles = failure["needles"]
        if _log_matches_needles(log_lower, needles):  # type: ignore[arg-type]
            return {
                "terminal_class": failure["terminal_class"],
                "failure_class": failure["failure_class"],
                "score_source": failure["score_source"],
                "reason": failure["reason"],
                "auth_eval_log_classification": "precise_pre_score_failure",
                "auth_eval_log_snippet_tail": _auth_log_tail_snippet(log_text),
            }
    return {
        "terminal_class": LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
        "failure_class": "runtime_or_harness_failure_before_score_json",
        "score_source": "none:missing_contest_auth_eval_json",
        "reason": (
            "contest_auth_eval.json is missing and auth_eval.log does not match "
            "a known precise pre-score failure signature"
        ),
        "auth_eval_log_classification": "missing_artifacts",
        "auth_eval_log_snippet_tail": _auth_log_tail_snippet(log_text),
    }


def refine_exact_eval_missing_json_failure(
    diagnostic: dict[str, object],
    *,
    artifact_dir: str | Path | None,
) -> dict[str, object]:
    if diagnostic.get("status") != "ARTIFACT_INFRA_FAILURE":
        return diagnostic
    if diagnostic.get("terminal_class") != LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS:
        return diagnostic
    if "contest_auth_eval.json" not in set(diagnostic.get("missing_required_files") or []):
        return diagnostic
    if artifact_dir is None:
        return diagnostic
    auth_log = Path(artifact_dir) / "auth_eval.log"
    if not auth_log.is_file():
        refined = dict(diagnostic)
        refined.setdefault("failure_class", "runtime_or_harness_failure_before_score_json")
        refined["auth_eval_log_classification"] = "missing_artifacts"
        refined["reason"] = (
            "contest_auth_eval.json is missing and auth_eval.log was not present "
            "in the harvested artifacts"
        )
        return refined
    classification = classify_exact_eval_missing_json_failure_from_auth_log(
        auth_log.read_text(encoding="utf-8", errors="replace")
    )
    if classification is None:
        return diagnostic
    refined = dict(diagnostic)
    refined.update(classification)
    refined["classified_at_utc"] = str(diagnostic.get("classified_at_utc") or _utc_now())
    refined["score_claim"] = False
    refined["method_evidence"] = False
    refined["promotion_eligible"] = False
    refined["evidence_grade"] = "invalid"
    refined["refined_from_terminal_class"] = LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS
    refined["recommended_action"] = (
        "Preserve the partial artifacts and logs as a pre-score exact-eval "
        "failure. Do not rank, retire, promote, or claim score without "
        "contest_auth_eval.json."
    )
    return refined


def _replace_latest_harvest_failure_in_state(
    *,
    state_path: Path,
    job_name: str,
    refined: dict[str, object],
) -> None:
    if not state_path.is_file():
        return
    client = LightningBatchJobsClient(state_path=state_path)

    def update_record(record: dict[str, object]) -> dict[str, object]:
        failures = list(record.get("artifact_failures") or [])
        if failures:
            failures[-1] = refined
            record["artifact_failures"] = failures
        record["status"] = "ARTIFACT_INFRA_FAILURE"
        if isinstance(refined.get("terminal_class"), str):
            record["terminal_class"] = refined["terminal_class"]
        history = list(record.get("status_history") or [])
        if history and isinstance(history[-1], dict):
            history[-1]["terminal_class"] = refined.get("terminal_class")
            history[-1]["source"] = "ssh_artifact_partial_failure_classification"
        record["status_history"] = history
        return record

    try:
        client.replace_latest_record_for_job(job_name, update_record)
    except KeyError:
        return


def _persist_harvest_failure_refinement(
    *,
    args: argparse.Namespace,
    refined: dict[str, object],
) -> None:
    ssh_source = _dict_or_empty(refined.get("ssh_source"))
    mirror_dir = ssh_source.get("mirror_dir") or getattr(args, "mirror_dir", None)
    if isinstance(mirror_dir, str) and mirror_dir:
        mirror = Path(mirror_dir)
        if mirror.exists():
            for name in (ARTIFACT_INFRA_FAILURE, ARTIFACT_VALIDATION):
                _write_json(mirror / name, refined)
    _replace_latest_harvest_failure_in_state(
        state_path=_state_path(args) or LIGHTNING_BATCH_STATE,
        job_name=args.job_name,
        refined=refined,
    )


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
    ssh_source = _dict_or_empty(result.get("ssh_source"))
    mirror_dir = ssh_source.get("mirror_dir") or args.mirror_dir
    refined = refine_exact_eval_missing_json_failure(result, artifact_dir=mirror_dir)
    if refined != result:
        result = refined
        _persist_harvest_failure_refinement(args=args, refined=result)
    _print_json(result)
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
            if record.get("status_reconciliation_required"):
                failures.append(
                    {
                        "job_name": job_name,
                        "error": "Lightning status reconciliation required",
                        "error_type": "StatusReconciliationRequired",
                        "status": record.get("status"),
                        "remote_observed_status": record.get("remote_observed_status"),
                        "status_anomalies": record.get("status_anomalies") or [],
                    }
                )
        _print_json(
            {
                "refreshed_count": len(results),
                "skipped_count": len(skipped),
                "failure_count": len(failures),
                "results": results,
                "skipped": skipped,
                "failures": failures,
            }
        )
        return 1 if failures and args.fail_on_error else 0
    if not args.job_name:
        raise SystemExit("refresh-status requires --job-name or --all")
    result = _refresh_one_status(client, args)
    _print_json(result)
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


def _attach_stop_request(
    client: LightningBatchJobsClient,
    *,
    job_name: str,
    stop_request: dict[str, object],
) -> dict[str, object]:
    index = client._find_record_index(job_name)
    records = client.list_records()
    record = dict(records[index])
    requests = list(record.get("stop_requests") or [])
    requests.append(stop_request)
    record["stop_requests"] = requests
    client._replace_record(index, record)
    return record


def _request_lightning_stop(job: object, *, timeout_seconds: float) -> object:
    if timeout_seconds <= 0:
        return job.stop()

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(signum: int, frame: object) -> None:  # pragma: no cover - signal plumbing
        raise TimeoutError("Lightning Job.stop() timed out")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return job.stop()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def cmd_stop_job(args: argparse.Namespace) -> int:
    client = _client(args)
    record = _latest_record_for_job(client, args.job_name)
    if record is None:
        raise SystemExit(f"Lightning Batch Job record not found: {args.job_name}")
    spec = _dict_or_empty(record.get("spec"))
    job_record = _dict_or_empty(record.get("job"))
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
    stop_request: dict[str, object] = {
        "schema_version": 1,
        "requested_at_utc": _utc_now(),
        "job_name": args.job_name,
        "sdk_job_name": sdk_job_name,
        "reason": args.reason,
        "timeout_seconds": args.timeout_seconds,
        "status_before": str(getattr(job, "status", None)),
    }
    try:
        result = _request_lightning_stop(job, timeout_seconds=float(args.timeout_seconds))
    except TimeoutError as exc:
        stop_request.update(
            {
                "stop_returned": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "status_after": str(getattr(job, "status", None)),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on provider SDK behavior
        stop_request.update(
            {
                "stop_returned": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "status_after": str(getattr(job, "status", None)),
            }
        )
    else:
        stop_request.update(
            {
                "stop_returned": True,
                "result_type": type(result).__name__,
                "status_after": str(getattr(job, "status", None)),
            }
        )
    _attach_stop_request(client, job_name=args.job_name, stop_request=stop_request)

    refresh_record = None
    refresh_error = None
    try:
        refresh_record = client.refresh_status_from_job(job_name=args.job_name, job=job)
    except Exception as exc:  # pragma: no cover - defensive around SDK state
        refresh_error = {"error_type": type(exc).__name__, "error": str(exc)}

    payload = {
        "stop_request": stop_request,
        "record": refresh_record,
        "refresh_error": refresh_error,
    }
    _print_json(payload)
    return 1 if not stop_request.get("stop_returned") else 0


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


def _is_sdk_machine_filter(machine: str | None) -> bool:
    if not machine:
        return False
    return re.fullmatch(r"[A-Z][A-Z0-9_]*", machine) is not None


def _filter_machine_rows(
    machine_rows: list[dict[str, object]],
    *,
    machine: str | None,
    sdk_filtered: bool,
) -> list[dict[str, object]]:
    if not machine or sdk_filtered:
        return machine_rows
    needle = machine.lower()
    matched = []
    for row in machine_rows:
        values = [
            row.get("name"),
            row.get("slug"),
            row.get("instance_type"),
            row.get("family"),
        ]
        if any(str(value).lower() == needle for value in values if value is not None):
            matched.append(row)
    return matched


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
    sdk_filtered = _is_sdk_machine_filter(machine)
    rows = []
    for cloud_account in selected_cloud_accounts:
        try:
            machines = teamspace.list_machines(
                cloud_account=cloud_account,
                machine=machine if sdk_filtered else None,
            )
            machine_rows = [_machine_to_record(machine) for machine in machines]
        except AttributeError:
            # lightning-sdk 2026.4.10 expects an org-backed teamspace in
            # Teamspace.list_machines(). User-owned teamspaces need org_id="".
            from lightning_sdk.api import TeamspaceApi
            from lightning_sdk.machine import Machine

            machine_filter = None
            if machine and sdk_filtered:
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
        machine_rows = _filter_machine_rows(
            machine_rows,
            machine=machine,
            sdk_filtered=sdk_filtered,
        )
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
    _print_json(rows)
    return 0


def _lightning_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("lightning", "pytorch-lightning", "lightning-sdk", "lightning_sdk"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _argparse_surface_summary(parser: argparse.ArgumentParser) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {"__main__": sorted(parser._option_string_actions)}
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, subparser in action.choices.items():
            summary[name] = sorted(subparser._option_string_actions)
    return summary


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

    parser_surface = _argparse_surface_summary(build_parser())
    checks["argparse_surface"] = {
        "ok": True,
        "commands": sorted(parser_surface),
        "remote_related_options": sorted(
            {
                option
                for options in parser_surface.values()
                for option in options
                if "remote" in option or "ssh" in option
            }
        ),
    }

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
        checks["ssh_auth"] = {
            "ok": False,
            "error": "--require-ssh set but --ssh-target/SSH alias was not provided",
        }
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
    if args.json_out:
        write_json(args.json_out, payload)
    _print_json(payload)
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


def _add_dispatch_claim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dispatch-lane-id",
        default=None,
        help="Lane id that must already be actively claimed in .omx/state/active_lane_dispatch_claims.md.",
    )
    parser.add_argument(
        "--dispatch-claims-path",
        default=str(_DISPATCH_CLAIMS_PATH),
        help="Markdown active-dispatch claim ledger to check before non-dry-run Studio submit.",
    )
    parser.add_argument(
        "--allow-missing-dispatch-claim-reason",
        default=None,
        help=(
            "Break-glass reason allowing a non-dry-run Studio submit without a "
            "matching active dispatch claim. Recorded in queue metadata."
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
    parser = StrictArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True, parser_class=StrictArgumentParser)

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
    exact.add_argument("--cloud-account", default=None)
    exact.add_argument("--python-bin", default=".venv/bin/python")
    exact.add_argument("--inflate-sh", default="submissions/robust_current/inflate.sh")
    exact.add_argument(
        "--allow-source-embedded-payload-runtime-reason",
        default=None,
        help=(
            "Forensic-only break-glass for public replay runtimes that embed "
            "large score-affecting payload literals in source instead of "
            "charging them in archive.zip."
        ),
    )
    exact.add_argument("--max-runtime", type=int, default=3 * 60 * 60)
    exact.add_argument("--env", action="append", default=[], help="KEY=VALUE env override; repeatable.")
    exact.add_argument("--queue-metadata", action="append", default=[], help="KEY=VALUE audit metadata; repeatable.")
    exact.add_argument("--local-artifact-dir", default=None, help="Expected local mirror/harvest path for artifacts.")
    _add_remote_preflight_args(exact)
    _add_dispatch_claim_args(exact)
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
    exact.add_argument(
        "--eval-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help=(
            "Exact auth-eval axis for experiments/contest_auth_eval.py. "
            "Default cuda preserves promotion-path behavior; cpu is a "
            "separate [contest-CPU] replay axis."
        ),
    )
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
        "--component-trace",
        action="store_true",
        help=(
            "After exact CUDA auth eval, emit diagnostic per-pair PoseNet/SegNet "
            "component_trace.json cross-checked against contest_auth_eval.json."
        ),
    )
    exact.add_argument("--component-trace-top-k", type=int, default=80)
    exact.add_argument(
        "--fail-job-on-component-gate",
        action="store_true",
        help=(
            "Return a failed Lightning job on component-gate violation. This "
            "is the default; the flag remains accepted for explicitness."
        ),
    )
    exact.add_argument(
        "--allow-component-gate-forensic-success",
        action="store_true",
        help=(
            "Explicit forensic-only escape hatch: return success for "
            "component-gate-only failures after writing non-promotable custody "
            "artifacts."
        ),
    )
    exact.add_argument(
        "--fail-job-on-sane-score-gate",
        action="store_true",
        help=(
            "Return a failed Lightning job when a finite exact-CUDA score is "
            "outside --max-sane-score. This is the default; the flag remains "
            "accepted for explicitness."
        ),
    )
    exact.add_argument(
        "--allow-sane-score-forensic-success",
        action="store_true",
        help=(
            "Explicit forensic-only escape hatch: return success for "
            "sane-score-gate failures after writing non-promotable custody "
            "artifacts."
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
    component.add_argument("--cloud-account", default=None)
    component.add_argument("--python-bin", default=".venv/bin/python")
    component.add_argument("--max-runtime", type=int, default=6 * 60 * 60)
    component.add_argument("--env", action="append", default=[], help="KEY=VALUE env override; repeatable.")
    component.add_argument("--queue-metadata", action="append", default=[], help="KEY=VALUE audit metadata; repeatable.")
    component.add_argument("--local-artifact-dir", default=None, help="Expected local mirror/harvest path for artifacts.")
    _add_remote_preflight_args(component)
    _add_dispatch_claim_args(component)
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

    sensitivity = sub.add_parser("component-sensitivity")
    _add_state_arg(sensitivity)
    sensitivity.add_argument("--job-name", required=True)
    sensitivity.add_argument("--baseline-archive", required=True, help="Baseline archive path visible inside the Lightning job.")
    sensitivity.add_argument("--repo-dir", required=True, help="Repo path visible inside the Lightning job.")
    sensitivity.add_argument("--upstream-dir", required=True, help="Upstream scorer path visible inside the Lightning job.")
    sensitivity.add_argument(
        "--video",
        default=None,
        help="Ground-truth video path visible inside the Lightning job. Default: <upstream-dir>/videos/0.mkv.",
    )
    sensitivity.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Writable artifact output dir visible inside job. Default: "
            "<repo-dir>/experiments/results/lightning_batch/<job-name>."
        ),
    )
    sensitivity.add_argument("--machine", default="T4")
    sensitivity.add_argument("--studio", default=None, help="Lightning Studio name/env for Studio-backed jobs.")
    sensitivity.add_argument("--image", default=None, help="Docker image for image-backed jobs.")
    sensitivity.add_argument("--teamspace", default=None)
    sensitivity.add_argument("--org", default=None)
    sensitivity.add_argument("--user", default=None)
    sensitivity.add_argument("--cloud-account", default=None)
    sensitivity.add_argument("--python-bin", default=".venv/bin/python")
    sensitivity.add_argument("--max-runtime", type=int, default=6 * 60 * 60)
    sensitivity.add_argument("--env", action="append", default=[], help="KEY=VALUE env override; repeatable.")
    sensitivity.add_argument("--queue-metadata", action="append", default=[], help="KEY=VALUE audit metadata; repeatable.")
    sensitivity.add_argument("--local-artifact-dir", default=None, help="Expected local mirror/harvest path for artifacts.")
    _add_remote_preflight_args(sensitivity)
    _add_dispatch_claim_args(sensitivity)
    sensitivity.add_argument("--pair-weights", default=None, help="Optional pair-weights file visible inside the Lightning job; otherwise uses --all-pairs.")
    sensitivity.add_argument("--top-k-pairs", type=int, default=64)
    sensitivity.add_argument("--pair-batch", type=int, default=2)
    sensitivity.add_argument("--response-top-k", type=int, default=16)
    sensitivity.add_argument(
        "--response-epsilons",
        default="-0.002,-0.001,-0.0005,0.0,0.0005,0.001,0.002",
    )
    sensitivity.add_argument("--split-seed", type=int, default=20260430)
    sensitivity.add_argument("--holdout-fraction", type=float, default=0.2)
    sensitivity.add_argument("--aggregate", choices=["sum", "mean", "max"], default="sum")
    sensitivity.add_argument(
        "--promotion-finite-difference",
        action="store_true",
        help=(
            "Run CUDA direct-renderer finite-difference maps instead of Fisher "
            "proxy maps. Still non-promotable until certified through official "
            "archive-response gates."
        ),
    )
    sensitivity.add_argument("--finite-difference-epsilon", type=float, default=0.001)
    sensitivity.add_argument("--finite-difference-shard-index", type=int, default=0)
    sensitivity.add_argument("--finite-difference-shard-count", type=int, default=1)
    sensitivity.add_argument("--expected-baseline-archive-sha256", default=None)
    sensitivity.add_argument(
        "--expected-baseline-archive-size-bytes",
        "--expected-baseline-archive-bytes",
        dest="expected_baseline_archive_size_bytes",
        type=int,
        default=None,
    )
    sensitivity.add_argument(
        "--infer-expected-baseline-archive",
        action="store_true",
        help="Compute expected baseline archive SHA-256/bytes locally before submit/dry-run.",
    )
    sensitivity.add_argument(
        "--local-baseline-archive",
        default=None,
        help="Local baseline archive to hash when --baseline-archive is a remote path.",
    )
    sensitivity.add_argument(
        "--source-manifest",
        default=None,
        help=(
            "Local lightning_repro_workspace.py manifest. Required for non-dry-run "
            "component-sensitivity submit so the baseline archive/video are known staged inputs."
        ),
    )
    sensitivity.add_argument("--dry-run", action="store_true")
    sensitivity.set_defaults(func=cmd_component_sensitivity)

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

    validate_sensitivity = sub.add_parser("validate-component-sensitivity-artifacts")
    validate_sensitivity.add_argument("--artifact-dir", required=True)
    validate_sensitivity.add_argument("--expected-baseline-archive-sha256", default=None)
    validate_sensitivity.add_argument(
        "--expected-baseline-archive-size-bytes",
        "--expected-baseline-archive-bytes",
        dest="expected_baseline_archive_size_bytes",
        type=int,
        default=None,
    )
    validate_sensitivity.set_defaults(func=cmd_validate_component_sensitivity_artifacts)

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

    harvest_sensitivity = sub.add_parser("harvest-component-sensitivity-local")
    harvest_sensitivity.add_argument("--artifact-dir", required=True)
    harvest_sensitivity.add_argument("--mirror-dir", default=None)
    harvest_sensitivity.add_argument("--expected-baseline-archive-sha256", default=None)
    harvest_sensitivity.add_argument(
        "--expected-baseline-archive-size-bytes",
        "--expected-baseline-archive-bytes",
        dest="expected_baseline_archive_size_bytes",
        type=int,
        default=None,
    )
    harvest_sensitivity.add_argument("--overwrite", action="store_true")
    harvest_sensitivity.set_defaults(func=cmd_harvest_component_sensitivity_local)

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

    harvest_sensitivity_ssh = sub.add_parser("harvest-component-sensitivity-ssh")
    _add_state_arg(harvest_sensitivity_ssh)
    harvest_sensitivity_ssh.add_argument(
        "--job-name",
        default=None,
        help=(
            "State record name. When provided, remote/mirror dirs default "
            "from state and Studio outputs are mapped into SDK artifacts."
        ),
    )
    harvest_sensitivity_ssh.add_argument("--ssh-target", required=True)
    harvest_sensitivity_ssh.add_argument("--remote-artifact-dir", default=None)
    harvest_sensitivity_ssh.add_argument("--mirror-dir", default=None)
    harvest_sensitivity_ssh.add_argument("--expected-baseline-archive-sha256", default=None)
    harvest_sensitivity_ssh.add_argument(
        "--expected-baseline-archive-size-bytes",
        "--expected-baseline-archive-bytes",
        dest="expected_baseline_archive_size_bytes",
        type=int,
        default=None,
    )
    harvest_sensitivity_ssh.add_argument("--overwrite", action="store_true")
    harvest_sensitivity_ssh.add_argument("--allow-manual-artifact-dir", action="store_true")
    harvest_sensitivity_ssh.add_argument("--override-reason", default=None)
    harvest_sensitivity_ssh.add_argument("--ssh-bin", default="ssh")
    harvest_sensitivity_ssh.add_argument("--scp-bin", default="scp")
    harvest_sensitivity_ssh.add_argument(
        "--ssh-connect-timeout",
        type=int,
        default=15,
        help="BatchMode SSH auth preflight timeout in seconds.",
    )
    harvest_sensitivity_ssh.set_defaults(func=cmd_harvest_component_sensitivity_ssh)

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

    stop = sub.add_parser("stop")
    stop.add_argument("--state-path", default=None, help="Override local Lightning Batch Jobs state JSON.")
    stop.add_argument("--job-name", required=True)
    stop.add_argument(
        "--sdk-job-name",
        default=None,
        help="Lightning SDK job name. Defaults to the recorded SDK name or --job-name with underscores replaced by hyphens.",
    )
    stop.add_argument("--teamspace", default=None)
    stop.add_argument("--org", default=None)
    stop.add_argument("--user", default=None)
    stop.add_argument("--reason", default=None)
    stop.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Bound Job.stop() so provider-side blocking does not hang local orchestration.",
    )
    stop.set_defaults(func=cmd_stop_job)

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
