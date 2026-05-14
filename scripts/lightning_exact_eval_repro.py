#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ruff: noqa: I001
"""Credential-safe reproducible Lightning exact-eval orchestrator.

This wrapper composes the two lower-level Lightning tools:

* ``scripts/lightning_repro_workspace.py`` for manifest-based Studio staging.
* ``scripts/launch_lightning_batch_job.py exact-eval`` for strict CUDA auth eval.

It intentionally does not contain SSH users, key paths, tokens, or Studio
credentials. Operators provide those via environment variables, SSH config
aliases, or CLI flags at runtime.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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

from tac.repo_io import json_text, read_json, sha256_file, write_json  # noqa: E402

DEFAULT_STATE_DIR = REPO_ROOT / ".omx/state"
DEFAULT_REMOTE_PACT = "/teamspace/studios/this_studio/pact"
DEFAULT_UPSTREAM_DIR = "/teamspace/studios/this_studio/upstream"
REQUIREMENTS_MODES = ("uv-sync", "verify-only", "no-install")

_STALE_REMOTE_ARG_HINTS = {
    "--rmote": "--remote",
    "--remote-preflight-ssh-target": "--remote on this wrapper; the wrapper forwards it to launch_lightning_batch_job.py",
    "--remote-preflight-target": "--remote",
    "--ssh-target": "--remote",
    "--ssh-alias": "--remote",
}


def _iter_parser_option_strings(parser: argparse.ArgumentParser) -> list[str]:
    return sorted(parser._option_string_actions)


def _unknown_arg_diagnostic(message: str, parser: argparse.ArgumentParser) -> str | None:
    prefix = "unrecognized arguments:"
    if prefix not in message:
        return None
    unknown = [item for item in message.split(prefix, 1)[1].split() if item.startswith("-")]
    if not unknown:
        return None
    known = _iter_parser_option_strings(parser)
    lines = ["Strict argparse rejected unknown option(s); use the real wrapper parser surface below:"]
    for item in unknown:
        stale_hint = _STALE_REMOTE_ARG_HINTS.get(item)
        nearest = difflib.get_close_matches(item, known, n=3, cutoff=0.62)
        if stale_hint:
            lines.append(f"  {item}: {stale_hint}")
        elif nearest:
            lines.append(f"  {item}: did you mean {', '.join(nearest)}?")
        else:
            lines.append(f"  {item}: no close known option")
    lines.append("Known options include: " + ", ".join(known))
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


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _repo_rel(path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    raw = Path(path)
    resolved = (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        return resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay inside repo for reproducible staging: {path}") from exc


def _archive_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _load_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite_number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out = float(value)
            if math.isfinite(out):
                return out
    return None


def _load_baseline(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    score = _finite_number(payload, "score_recomputed_from_components", "final_score")
    archive_bytes = payload.get("archive_size_bytes")
    if not isinstance(archive_bytes, int):
        archive_bytes = provenance.get("archive_size_bytes")
    if not isinstance(archive_bytes, int):
        archive_bytes = None
    return {
        "score": score,
        "archive_size_bytes": archive_bytes,
        "avg_posenet_dist": _finite_number(payload, "avg_posenet_dist", "pose_dist"),
        "avg_segnet_dist": _finite_number(payload, "avg_segnet_dist", "seg_dist"),
        "archive_sha256": provenance.get("archive_sha256"),
        "device": provenance.get("device"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
    }


def _parse_metadata(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--queue-metadata requires KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--queue-metadata key is empty in {item!r}")
        out[key] = value.strip()
    return out


def _fmt_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".17g")


def _append_optional(cmd: list[str], flag: str, value: str | int | float | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _manifest_entries_by_path(manifest: dict[str, Any], *, manifest_path: Path) -> dict[str, dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"source manifest missing files list: {manifest_path}")
    out: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"source manifest files[{index}] must be an object: {manifest_path}")
        rel = item.get("path")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"source manifest files[{index}].path must be a nonempty string: {manifest_path}")
        if rel in out:
            raise ValueError(f"source manifest contains duplicate path: {rel}")
        out[rel] = item
    return out


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_source_manifest_identity(manifest: dict[str, Any], *, manifest_path: Path) -> None:
    declared = manifest.get("manifest_sha256")
    if declared is None:
        return
    expected = _canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if declared != expected:
        raise ValueError(
            "exact-eval submit blocked; staged source manifest manifest_sha256 is stale: "
            f"{manifest_path} expected {declared!r} actual {expected!r}"
        )


def _assert_manifest_entry_matches_disk(entry: dict[str, Any], *, repo_root: Path, manifest_path: Path) -> None:
    rel = entry["path"]
    path = repo_root / rel
    if not path.is_file():
        raise ValueError(f"source manifest path is not a local file before submit: {rel}")
    expected_bytes = entry.get("bytes")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_bytes, int):
        raise ValueError(f"source manifest entry missing integer bytes for {rel}: {manifest_path}")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError(f"source manifest entry missing sha256 for {rel}: {manifest_path}")
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise ValueError(
            "exact-eval submit blocked; local file changed after staging source manifest: "
            f"{rel} expected bytes={expected_bytes} sha256={expected_sha} "
            f"actual bytes={actual_bytes} sha256={actual_sha}"
        )


def _public_preflight_path(plan: dict[str, Any]) -> str | None:
    metadata = plan.get("queue_metadata")
    if isinstance(metadata, dict):
        value = metadata.get("public_preflight")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _validate_public_replay_preflight(plan: dict[str, Any], *, repo_root: Path) -> None:
    preflight_rel = _public_preflight_path(plan)
    if preflight_rel is None:
        return
    preflight_path = repo_root / _repo_rel(preflight_rel, repo_root=repo_root)
    preflight = _load_json(preflight_path)
    if preflight.get("ready_for_exact_eval_dispatch") is not True:
        raise ValueError(
            "exact-eval submit blocked; public replay preflight is not ready_for_exact_eval_dispatch=true: "
            f"{preflight_rel}"
        )
    runtime = preflight.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"public replay preflight missing runtime object: {preflight_rel}")
    inflate_runtime = plan.get("inflate_runtime")
    inflate_rel = None
    if isinstance(inflate_runtime, dict):
        value = inflate_runtime.get("inflate_sh")
        if isinstance(value, str):
            inflate_rel = value
    if not inflate_rel:
        raise ValueError("exact-eval plan missing inflate_runtime.inflate_sh")
    preflight_inflate = runtime.get("inflate_sh")
    if preflight_inflate != inflate_rel:
        raise ValueError(
            "exact-eval submit blocked; public replay preflight runtime.inflate_sh does not match plan: "
            f"{preflight_inflate!r} != {inflate_rel!r}"
        )
    expected_inflate_sha = runtime.get("inflate_sh_sha256")
    actual_inflate_sha = sha256_file(repo_root / inflate_rel)
    if expected_inflate_sha != actual_inflate_sha:
        raise ValueError(
            "exact-eval submit blocked; public replay preflight inflate_sh_sha256 is stale: "
            f"{preflight_rel} expected {expected_inflate_sha!r} actual {actual_inflate_sha!r}"
        )
    inflate_text = (repo_root / inflate_rel).read_text(errors="replace")
    if "PACT_RUNTIME_DEPENDENCY_ROOT=" in inflate_text:
        runtime_manifest = runtime.get("runtime_manifest")
        if not isinstance(runtime_manifest, dict):
            raise ValueError(
                "exact-eval submit blocked; adapter declares PACT_RUNTIME_DEPENDENCY_ROOT "
                f"but public preflight has no runtime_manifest: {preflight_rel}"
            )
        roots = runtime_manifest.get("external_dependency_roots")
        if not isinstance(roots, list) or not roots:
            raise ValueError(
                "exact-eval submit blocked; adapter declares PACT_RUNTIME_DEPENDENCY_ROOT "
                f"but public preflight has no external_dependency_roots: {preflight_rel}"
            )
        artifacts = plan.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("exact-eval plan missing artifacts list")
        missing_roots: list[str] = []
        for root in roots:
            if not isinstance(root, dict):
                continue
            root_rel = root.get("repo_relative_root")
            if not isinstance(root_rel, str) or not root_rel:
                continue
            prefix = root_rel.rstrip("/") + "/"
            if not any(
                isinstance(artifact, str)
                and (artifact == root_rel or artifact.startswith(prefix))
                for artifact in artifacts
            ):
                missing_roots.append(root_rel)
        if missing_roots:
            raise ValueError(
                "exact-eval submit blocked; public replay preflight external runtime "
                "root is not staged by the plan: " + ", ".join(sorted(missing_roots))
            )


def _validate_staged_manifest_consistency(plan: dict[str, Any], *, repo_root: Path) -> None:
    manifest_rel = plan.get("paths", {}).get("manifest_out")
    if not isinstance(manifest_rel, str) or not manifest_rel:
        raise ValueError("exact-eval plan missing paths.manifest_out")
    manifest_path = repo_root / _repo_rel(manifest_rel, repo_root=repo_root)
    manifest = _load_json(manifest_path)
    _validate_source_manifest_identity(manifest, manifest_path=manifest_path)
    entries = _manifest_entries_by_path(manifest, manifest_path=manifest_path)
    missing: list[str] = []
    for rel in plan.get("artifacts", []):
        if not isinstance(rel, str):
            raise ValueError(f"exact-eval plan artifact path must be a string: {rel!r}")
        root = repo_root / _repo_rel(rel, repo_root=repo_root)
        if root.is_file() and rel not in entries:
            missing.append(rel)
        elif root.is_dir():
            prefix = rel.rstrip("/") + "/"
            if not any(path.startswith(prefix) for path in entries):
                missing.append(rel)
        elif not root.exists():
            raise ValueError(f"exact-eval plan artifact disappeared before submit: {rel}")
    if missing:
        raise ValueError(
            "exact-eval submit blocked; staged source manifest does not include plan artifact(s): "
            + ", ".join(sorted(missing))
        )
    for entry in entries.values():
        _assert_manifest_entry_matches_disk(entry, repo_root=repo_root, manifest_path=manifest_path)
    _validate_public_replay_preflight(plan, repo_root=repo_root)


def _ssh_target_shape_blocker(target: str | None, *, flag: str) -> str | None:
    value = str(target or "").strip()
    if not value:
        return f"missing {flag}; Studio submit needs an SSH alias so remote preflight can run"
    if any(ch in value for ch in "\r\n\0"):
        return f"{flag} contains control characters"
    if value == "ssh.lightning.ai":
        return f"{flag} must be a ~/.ssh/config alias or user-qualified target, not bare ssh.lightning.ai"
    return None


def _iter_inflate_runtime_files(runtime_dir: Path) -> list[Path]:
    """Return deterministic runtime files under an external inflate directory.

    Public PR replay/runtime packets often carry helpers below ``src/`` next to
    ``inflate.py``. A direct-sibling scan silently omits those files and lets a
    Lightning submit pass local closure checks while failing remotely at import
    time. Keep this recursive but conservative: hidden files and caches are not
    runtime dependencies.
    """
    if not runtime_dir.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(runtime_dir.rglob("*")):
        if not child.is_file():
            continue
        rel_parts = child.relative_to(runtime_dir).parts
        if any(
            part.startswith(".") or part.startswith("._") or part == "__pycache__"
            for part in rel_parts
        ):
            continue
        out.append(child)
    return out


def _declared_runtime_dependency_roots(inflate_rel: Path, *, repo_root: Path) -> list[str]:
    """Return repo-relative PACT_RUNTIME_DEPENDENCY_ROOT literals from an adapter.

    The directive may be in a shell assignment or an adjacent comment. It must
    be a literal repo path so staging can prove exact source custody before
    submitting a remote CUDA eval.
    """
    inflate_path = repo_root / inflate_rel
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
            raise ValueError(
                "invalid PACT_RUNTIME_DEPENDENCY_ROOT shell literal in "
                f"{inflate_rel}: {exc}"
            ) from exc
        if not parsed:
            raise ValueError(f"empty PACT_RUNTIME_DEPENDENCY_ROOT in {inflate_rel}")
        raw_root = parsed[0]
        if "$" in raw_root or "`" in raw_root:
            raise ValueError(
                "PACT_RUNTIME_DEPENDENCY_ROOT must be a repo-relative or "
                f"repo-absolute literal, got {raw_root!r} in {inflate_rel}"
            )
        root_rel = _repo_rel(raw_root, repo_root=repo_root).as_posix()
        if not (repo_root / root_rel).is_dir():
            raise ValueError(
                f"PACT_RUNTIME_DEPENDENCY_ROOT does not exist or is not a directory: {root_rel}"
            )
        roots.append(root_rel)
    return _dedupe_preserve(roots)


def _inflate_runtime_artifacts(inflate_rel: Path, *, repo_root: Path) -> list[str]:
    rels = [inflate_rel.as_posix()]
    config_rel = inflate_rel.parent / "config.env"
    if (repo_root / config_rel).is_file():
        rels.append(config_rel.as_posix())
    if inflate_rel.as_posix() == "submissions/robust_current/inflate.sh":
        return _dedupe_preserve(rels)
    runtime_dir = repo_root / inflate_rel.parent
    for child in _iter_inflate_runtime_files(runtime_dir):
        rels.append(child.relative_to(repo_root).as_posix())
    for root_rel in _declared_runtime_dependency_roots(inflate_rel, repo_root=repo_root):
        for child in _iter_inflate_runtime_files(repo_root / root_rel):
            rels.append(child.relative_to(repo_root).as_posix())
    return _dedupe_preserve(rels)


def build_parser() -> argparse.ArgumentParser:
    parser = StrictArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--archive", required=True, help="Local repo-relative archive to stage and evaluate.")
    parser.add_argument("--extra-artifact", action="append", default=[], help="Additional repo artifact to stage.")
    parser.add_argument("--baseline-json", default=None, help="Baseline contest_auth_eval.json for adjudication defaults.")
    parser.add_argument("--remote", default=_env_first("LIGHTNING_SSH_TARGET", "LIGHTNING_REMOTE", "REMOTE"))
    parser.add_argument("--remote-pact", default=os.environ.get("LIGHTNING_REMOTE_PACT", DEFAULT_REMOTE_PACT))
    parser.add_argument("--remote-archive-path", default=None, help="Override archive path visible inside the Batch job.")
    parser.add_argument("--upstream-dir", default=os.environ.get("LIGHTNING_UPSTREAM_DIR", DEFAULT_UPSTREAM_DIR))
    parser.add_argument(
        "--inflate-sh",
        default=os.environ.get("LIGHTNING_INFLATE_SH", "submissions/robust_current/inflate.sh"),
        help="Repo-relative inflate.sh to use for exact eval; public replay runs must not fall back to robust_current.",
    )
    parser.add_argument("--studio", default=os.environ.get("LIGHTNING_STUDIO"))
    parser.add_argument("--image", default=os.environ.get("LIGHTNING_IMAGE"))
    parser.add_argument("--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE"))
    parser.add_argument("--org", default=os.environ.get("LIGHTNING_ORG"))
    parser.add_argument("--sdk-user", dest="sdk_user", default=os.environ.get("LIGHTNING_SDK_USER"))
    parser.add_argument("--machine", default=os.environ.get("LIGHTNING_MACHINE", "T4"))
    parser.add_argument("--requirements-mode", choices=REQUIREMENTS_MODES, default=os.environ.get("LIGHTNING_REQUIREMENTS_MODE", "uv-sync"))
    parser.add_argument("--python-bin", default=os.environ.get("LIGHTNING_PYTHON_BIN"))
    parser.add_argument("--batch-python-bin", default=os.environ.get("LIGHTNING_BATCH_PYTHON_BIN"))
    parser.add_argument("--source", action="append", default=None, help="Override staged source path; repeatable.")
    parser.add_argument("--stage-workspace", action="store_true", help="Run manifest staging over SSH before queuing.")
    parser.add_argument("--require-stage-cuda", action="store_true", help="Require CUDA during SSH staging runtime check.")
    parser.add_argument("--stage-only", action="store_true", help="Stage and verify workspace, but do not queue a Batch job.")
    parser.add_argument("--submit", action="store_true", help="Submit the Lightning Batch Job. Default is dry-run queue only.")
    parser.add_argument(
        "--allow-unstaged-submit",
        action="store_true",
        help="Allow --submit without --stage-workspace when remote custody was verified separately.",
    )
    parser.add_argument(
        "--allow-skip-remote-preflight-reason",
        default=None,
        help=(
            "Break-glass reason passed to launch_lightning_batch_job.py when "
            "a Studio submit cannot run SSH remote preflight."
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--plan-out", default=None)
    parser.add_argument("--queue-record-out", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--local-artifact-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="Writable remote output dir. Defaults under remote pact results.")
    parser.add_argument("--plan-only", action="store_true", help="Write/print plan JSON without executing commands.")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE env override forwarded to the exact-eval Batch job.")
    parser.add_argument("--queue-metadata", action="append", default=[])
    parser.add_argument(
        "--dispatch-lane-id",
        default=None,
        help=(
            "Active lane id already claimed in .omx/state/active_lane_dispatch_claims.md; "
            "forwarded to launch_lightning_batch_job.py for non-dry-run submit guardrails."
        ),
    )
    parser.add_argument(
        "--dispatch-claims-path",
        default=None,
        help="Override dispatch-claim ledger path forwarded to launch_lightning_batch_job.py.",
    )
    parser.add_argument(
        "--allow-missing-dispatch-claim-reason",
        default=None,
        help="Auditable break-glass reason forwarded to launch_lightning_batch_job.py.",
    )

    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--baseline-archive-bytes", type=int)
    parser.add_argument("--predicted-band", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--regression-threshold", type=float)
    parser.add_argument("--delta-key", default="score_delta_vs_baseline")
    parser.add_argument("--max-posenet-dist", type=float)
    parser.add_argument("--max-segnet-dist", type=float)
    parser.add_argument("--baseline-posenet-dist", "--reference-posenet-dist", dest="baseline_posenet_dist", type=float)
    parser.add_argument("--baseline-segnet-dist", "--reference-segnet-dist", dest="baseline_segnet_dist", type=float)
    parser.add_argument("--max-posenet-relative", type=float)
    parser.add_argument("--max-segnet-relative", type=float)
    parser.add_argument("--component-reference-label", default=None)
    parser.add_argument("--max-sane-score", type=float, default=10.0)
    parser.add_argument(
        "--component-trace",
        action="store_true",
        help="Forward exact-eval component trace generation for per-pair diagnostics.",
    )
    parser.add_argument("--component-trace-top-k", type=int)
    return parser


def _resolve_plan_paths(args: argparse.Namespace, *, repo_root: Path) -> dict[str, str]:
    run_id = args.run_id or args.job_name
    return {
        "run_id": run_id,
        "manifest_out": args.manifest_out or f".omx/state/{run_id}_manifest.json",
        "local_supply_chain_scan": f".omx/state/{run_id}_local_lightning_supply_chain_scan.json",
        "plan_out": args.plan_out or f".omx/state/{run_id}_lightning_exact_eval_repro_plan.json",
        "queue_record_out": args.queue_record_out or f".omx/state/{run_id}_lightning_batch_record.json",
        "local_artifact_dir": args.local_artifact_dir or f"experiments/results/lightning_batch/{args.job_name}",
    }


def build_plan(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    if args.stage_only and not args.stage_workspace:
        raise ValueError("--stage-only requires --stage-workspace")
    if args.studio and args.image:
        raise ValueError("--studio and --image are mutually exclusive")
    if args.submit and not args.stage_workspace and not args.allow_unstaged_submit:
        raise ValueError("--submit requires --stage-workspace or --allow-unstaged-submit")
    if args.submit and not (args.studio or args.image):
        raise ValueError("--submit requires --studio or --image to avoid SDK-default ambiguity")
    if args.stage_workspace and not args.remote:
        raise ValueError("--stage-workspace requires --remote or LIGHTNING_SSH_TARGET")
    if args.stage_workspace:
        blocker = _ssh_target_shape_blocker(args.remote, flag="--remote")
        if blocker:
            raise ValueError(blocker)
    if args.submit and args.studio:
        blocker = _ssh_target_shape_blocker(args.remote, flag="--remote")
        skip_reason = str(args.allow_skip_remote_preflight_reason or "").strip()
        if blocker and not skip_reason:
            raise ValueError(
                blocker
                + "; pass --remote/LIGHTNING_SSH_TARGET before --submit so "
                "launch_lightning_batch_job.py receives --remote-preflight-ssh-target"
            )
        if blocker and len(skip_reason) < 12:
            raise ValueError("--allow-skip-remote-preflight-reason must be a specific auditable reason")

    archive_rel = _repo_rel(args.archive, repo_root=repo_root)
    archive_path = repo_root / archive_rel
    identity = _archive_identity(archive_path)
    inflate_rel = _repo_rel(args.inflate_sh, repo_root=repo_root)
    paths = _resolve_plan_paths(args, repo_root=repo_root)

    baseline_rel: Path | None = None
    baseline_payload: dict[str, Any] = {}
    if args.baseline_json:
        baseline_rel = _repo_rel(args.baseline_json, repo_root=repo_root)
        baseline_payload = _load_baseline(repo_root / baseline_rel)

    queue_needed = not args.stage_only
    baseline_score = args.baseline_score if args.baseline_score is not None else baseline_payload.get("score")
    baseline_archive_bytes = (
        args.baseline_archive_bytes
        if args.baseline_archive_bytes is not None
        else baseline_payload.get("archive_size_bytes")
    )
    baseline_posenet_dist = (
        args.baseline_posenet_dist
        if args.baseline_posenet_dist is not None
        else baseline_payload.get("avg_posenet_dist")
    )
    baseline_segnet_dist = (
        args.baseline_segnet_dist
        if args.baseline_segnet_dist is not None
        else baseline_payload.get("avg_segnet_dist")
    )
    component_reference_label = args.component_reference_label
    if component_reference_label is None:
        component_reference_label = baseline_rel.as_posix() if baseline_rel else "baseline"

    if queue_needed:
        missing = []
        if baseline_score is None:
            missing.append("--baseline-score or --baseline-json with score")
        if args.predicted_band is None:
            missing.append("--predicted-band LOW HIGH")
        if args.regression_threshold is None:
            missing.append("--regression-threshold")
        if args.max_posenet_relative is not None and baseline_posenet_dist is None:
            missing.append("--baseline-posenet-dist or --baseline-json with avg_posenet_dist")
        if args.max_segnet_relative is not None and baseline_segnet_dist is None:
            missing.append("--baseline-segnet-dist or --baseline-json with avg_segnet_dist")
        if missing:
            raise ValueError("exact-eval queue requires " + ", ".join(missing))

    declared_dependency_roots = _declared_runtime_dependency_roots(inflate_rel, repo_root=repo_root)
    inflate_artifact_rels = _inflate_runtime_artifacts(inflate_rel, repo_root=repo_root)
    artifact_rels = [archive_rel.as_posix(), *inflate_artifact_rels]
    for item in args.extra_artifact:
        artifact_rels.append(_repo_rel(item, repo_root=repo_root).as_posix())
    if baseline_rel is not None:
        artifact_rels.append(baseline_rel.as_posix())
    artifact_rels = _dedupe_preserve(artifact_rels)

    local_scan_cmd = [
        sys.executable,
        "scripts/scan_lightning_supply_chain.py",
        "--json-out",
        paths["local_supply_chain_scan"],
        "--strict",
        "--quiet",
    ]

    stage_cmd: list[str] | None = None
    if args.stage_workspace:
        stage_cmd = [
            sys.executable,
            "scripts/lightning_repro_workspace.py",
            "--remote",
            args.remote,
            "--remote-pact",
            args.remote_pact,
            "--run-id",
            paths["run_id"],
            "--manifest-out",
            paths["manifest_out"],
            "--requirements-mode",
            args.requirements_mode,
        ]
        if args.python_bin:
            stage_cmd.extend(["--python-bin", args.python_bin])
        if args.require_stage_cuda:
            stage_cmd.append("--require-cuda")
        if args.source is not None:
            for source in args.source:
                stage_cmd.extend(["--source", source])
        for artifact in artifact_rels:
            stage_cmd.extend(["--artifact", artifact])

    batch_python_bin = args.batch_python_bin
    if batch_python_bin is None:
        batch_python_bin = args.python_bin if args.requirements_mode == "verify-only" and args.python_bin else ".venv/bin/python"

    remote_archive = args.remote_archive_path or f"{args.remote_pact.rstrip('/')}/{archive_rel.as_posix()}"
    remote_output_dir = args.output_dir
    if remote_output_dir is not None and remote_output_dir.rstrip("/") == "/teamspace/jobs":
        raise ValueError("--output-dir must be writable; /teamspace/jobs is a read-only artifact view")
    if remote_output_dir is not None and remote_output_dir.startswith("/teamspace/jobs/"):
        raise ValueError(
            "--output-dir must not target /teamspace/jobs/...; that read-only artifact view "
            "is not a writable Studio workspace path"
        )

    metadata = {
        "wrapper": "scripts/lightning_exact_eval_repro.py",
        "run_id": paths["run_id"],
        "archive_rel": archive_rel.as_posix(),
        "source_manifest": paths["manifest_out"],
        "local_supply_chain_scan": paths["local_supply_chain_scan"],
    }
    if baseline_rel is not None:
        metadata["baseline_json"] = baseline_rel.as_posix()
    metadata.update(_parse_metadata(args.queue_metadata))

    queue_cmd: list[str] | None = None
    if queue_needed:
        assert baseline_score is not None
        assert args.predicted_band is not None
        assert args.regression_threshold is not None
        queue_cmd = [
            sys.executable,
            "scripts/launch_lightning_batch_job.py",
            "exact-eval",
            "--job-name",
            args.job_name,
            "--archive",
            remote_archive,
            "--repo-dir",
            args.remote_pact,
            "--upstream-dir",
            args.upstream_dir,
            "--inflate-sh",
            inflate_rel.as_posix(),
            "--machine",
            args.machine,
            "--python-bin",
            batch_python_bin,
            "--expected-archive-sha256",
            identity["sha256"],
            "--expected-archive-size-bytes",
            str(identity["size_bytes"]),
            "--local-artifact-dir",
            paths["local_artifact_dir"],
            "--adjudicate",
            "--baseline-score",
            _fmt_number(float(baseline_score)),
            "--predicted-band",
            _fmt_number(args.predicted_band[0]),
            _fmt_number(args.predicted_band[1]),
            "--regression-threshold",
            _fmt_number(args.regression_threshold),
            "--delta-key",
            args.delta_key,
            "--component-reference-label",
            component_reference_label,
            "--max-sane-score",
            _fmt_number(args.max_sane_score),
        ]
        if not args.submit:
            queue_cmd.append("--dry-run")
        if args.submit:
            queue_cmd.extend(["--source-manifest", paths["manifest_out"]])
            if args.remote:
                queue_cmd.extend(["--remote-preflight-ssh-target", args.remote])
            elif args.allow_skip_remote_preflight_reason:
                queue_cmd.extend(
                    [
                        "--allow-skip-remote-preflight-reason",
                        args.allow_skip_remote_preflight_reason,
                    ]
                )
            if args.dispatch_lane_id:
                queue_cmd.extend(["--dispatch-lane-id", args.dispatch_lane_id])
            if args.dispatch_claims_path:
                queue_cmd.extend(["--dispatch-claims-path", args.dispatch_claims_path])
            if args.allow_missing_dispatch_claim_reason:
                queue_cmd.extend(
                    [
                        "--allow-missing-dispatch-claim-reason",
                        args.allow_missing_dispatch_claim_reason,
                    ]
                )
        _append_optional(queue_cmd, "--state-path", args.state_path)
        _append_optional(queue_cmd, "--output-dir", remote_output_dir)
        _append_optional(queue_cmd, "--studio", args.studio)
        _append_optional(queue_cmd, "--image", args.image)
        _append_optional(queue_cmd, "--teamspace", args.teamspace)
        _append_optional(queue_cmd, "--org", args.org)
        _append_optional(queue_cmd, "--user", args.sdk_user)
        _append_optional(queue_cmd, "--baseline-archive-bytes", baseline_archive_bytes)
        _append_optional(queue_cmd, "--max-posenet-dist", args.max_posenet_dist)
        _append_optional(queue_cmd, "--max-segnet-dist", args.max_segnet_dist)
        _append_optional(queue_cmd, "--baseline-posenet-dist", baseline_posenet_dist)
        _append_optional(queue_cmd, "--baseline-segnet-dist", baseline_segnet_dist)
        _append_optional(queue_cmd, "--max-posenet-relative", args.max_posenet_relative)
        _append_optional(queue_cmd, "--max-segnet-relative", args.max_segnet_relative)
        if args.component_trace:
            queue_cmd.append("--component-trace")
        _append_optional(queue_cmd, "--component-trace-top-k", args.component_trace_top_k)
        for item in args.env:
            queue_cmd.extend(["--env", item])
        for key in sorted(metadata):
            queue_cmd.extend(["--queue-metadata", f"{key}={metadata[key]}"])

    commands = {
        "local_supply_chain_scan": local_scan_cmd,
        "stage_workspace": stage_cmd,
        "queue_exact_eval": queue_cmd,
    }
    return {
        "schema_version": 1,
        "tool": "scripts/lightning_exact_eval_repro.py",
        "generated_at_utc": _utc_now(),
        "job_name": args.job_name,
        "submit": bool(args.submit),
        "stage_workspace": bool(args.stage_workspace),
        "stage_only": bool(args.stage_only),
        "archive": {
            "local_path": archive_rel.as_posix(),
            "remote_path": remote_archive,
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        },
        "inflate_runtime": {
            "inflate_sh": inflate_rel.as_posix(),
            "declared_dependency_roots": declared_dependency_roots,
            "staged_artifacts": inflate_artifact_rels,
        },
        "baseline": {
            "json_path": baseline_rel.as_posix() if baseline_rel else None,
            "loaded": baseline_payload,
            "baseline_score": baseline_score,
            "baseline_archive_bytes": baseline_archive_bytes,
            "baseline_posenet_dist": baseline_posenet_dist,
            "baseline_segnet_dist": baseline_segnet_dist,
            "component_reference_label": component_reference_label,
        },
        "artifacts": artifact_rels,
        "paths": paths,
        "queue_metadata": metadata,
        "env": list(args.env),
        "component_trace": bool(args.component_trace),
        "component_trace_top_k": args.component_trace_top_k,
        "argparse_surface": _iter_parser_option_strings(build_parser()),
        "commands": commands,
        "command_strings": {
            key: shlex.join(value) if value else None
            for key, value in commands.items()
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _run(cmd: list[str], *, cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + shlex.join(cmd))
    if capture:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
        return proc
    return subprocess.run(cmd, cwd=cwd, text=True, check=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    plan_out = REPO_ROOT / plan["paths"]["plan_out"]
    _write_json(plan_out, plan)
    print(
        json_text(
            {"status": "PLAN_READY", "plan": str(plan_out), "submit": plan["submit"]}
        ),
        end="",
    )
    if args.plan_only:
        return 0

    _run(plan["commands"]["local_supply_chain_scan"], cwd=REPO_ROOT)
    if plan["commands"]["stage_workspace"]:
        _run(plan["commands"]["stage_workspace"], cwd=REPO_ROOT)
    if args.stage_only:
        return 0
    if args.submit:
        _validate_staged_manifest_consistency(plan, repo_root=REPO_ROOT)

    queue_cmd = plan["commands"]["queue_exact_eval"]
    if queue_cmd is None:
        raise RuntimeError("internal error: queue command is missing")
    queue_proc = _run(queue_cmd, cwd=REPO_ROOT, capture=True)
    try:
        queue_record = json.loads(queue_proc.stdout)
    except json.JSONDecodeError:
        queue_record = {"raw_stdout": queue_proc.stdout}
    queue_record_out = REPO_ROOT / plan["paths"]["queue_record_out"]
    _write_json(queue_record_out, queue_record)
    print(
        json_text(
            {
                "status": "QUEUED" if args.submit else "DRY_RUN_RECORDED",
                "record": str(queue_record_out),
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
