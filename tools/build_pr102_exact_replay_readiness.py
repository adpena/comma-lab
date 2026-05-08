#!/usr/bin/env python3
"""Build a no-dispatch PR102 public exact-replay readiness artifact.

The tool validates the source-sized intake manifest for PR102, verifies local
archive/runtime custody, and emits a fail-closed adapter plan for a later
``experiments/contest_auth_eval.py --device cuda`` run. It never runs CUDA,
never launches provider jobs, and never writes ``.omx/state``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import stat
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.audit_contract import AuditReport, audit_exit_code  # noqa: E402
from tac.repo_io import json_text, read_json, repo_relative, sha256_file  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "reverse_engineering/public_pr102_pr108_intake_20260508/manifest.json"
DEFAULT_ADAPTER_REL_PATH = (
    "experiments/results/public_pr102_exact_replay_adapter_20260508_codex/inflate.sh"
)
DEFAULT_LIGHTNING_LANE_ID = "pr102_public_exact_replay_t4"
DEFAULT_LIGHTNING_JOB_NAME = "pr102-hnerv-lc-v2-scale095-rplus1-exact"
DEFAULT_LIGHTNING_MACHINE = "g4dn.2xlarge"

NETWORK_OR_INSTALL_RE = re.compile(
    r"\b(pip\b.*\binstall|curl\b|wget\b|urllib\.request|urlopen|requests\.|https?://|git\s+clone)\b",
    re.IGNORECASE,
)
LOCAL_MODULE_NAMES = {
    "hnerv_model",
    "schema",
    "sidecar",
}
REQUIRED_RUNTIME_FILES = {
    "README.md",
    "compress.sh",
    "inflate.sh",
    "inflate.py",
    "hnerv_model.py",
    "schema.py",
    "sidecar.py",
}


def _repo_rel(path: Path, repo_root: Path) -> str:
    return repo_relative(path, repo_root)


def _path_from_manifest(raw: str, *, repo_root: Path, label: str) -> Path:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe repo-relative path: {raw!r}")
    return repo_root / Path(path)


def _entry_for_pr(manifest: dict[str, Any], pr_number: int) -> dict[str, Any]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest.entries must be a list")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("pr_number") == pr_number]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one PR{pr_number} entry, found {len(matches)}")
    return matches[0]


def _zip_member_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            data = zf.read(info.filename)
            records.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "compress_type": info.compress_type,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return records


def _validate_archive(entry: dict[str, Any], *, repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    archive_meta = entry.get("archive")
    if not isinstance(archive_meta, dict):
        return {}, ["missing_pr102_archive_metadata"]

    try:
        archive_path = _path_from_manifest(
            str(archive_meta.get("local_path") or ""),
            repo_root=repo_root,
            label="archive.local_path",
        )
    except ValueError as exc:
        return {}, [str(exc)]

    observed: dict[str, Any] = {
        "path": _repo_rel(archive_path, repo_root),
        "canonical_url": archive_meta.get("canonical_url"),
        "expected_bytes": archive_meta.get("bytes"),
        "expected_sha256": archive_meta.get("sha256"),
        "clean_clone_restore_required": True,
        "clean_clone_restore_note": (
            "adapter materialization does not embed archive payload bytes; clean clones must "
            "restore the archive from canonical_url or a verified source_manifest artifact "
            "and re-check size/SHA before exact replay"
        ),
    }
    if not archive_path.is_file():
        blockers.append("archive_local_path_missing")
        observed["exists"] = False
        return observed, blockers

    observed["exists"] = True
    observed["bytes"] = archive_path.stat().st_size
    observed["sha256"] = sha256_file(archive_path)
    if observed["bytes"] != archive_meta.get("bytes"):
        blockers.append("archive_size_mismatch")
    if observed["sha256"] != archive_meta.get("sha256"):
        blockers.append("archive_sha256_mismatch")

    try:
        members = _zip_member_records(archive_path)
    except BadZipFile:
        blockers.append("archive_bad_zip")
        members = []
    observed["members"] = members

    expected_members = archive_meta.get("members")
    if not isinstance(expected_members, list):
        blockers.append("archive_manifest_members_missing")
        expected_members = []
    if len(members) != len(expected_members):
        blockers.append("archive_member_count_mismatch")

    actual_by_name = {str(member.get("name")): member for member in members}
    if len(actual_by_name) != len(members):
        blockers.append("archive_duplicate_member_names")
    expected_by_name = {str(member.get("name")): member for member in expected_members if isinstance(member, dict)}
    for name, expected in expected_by_name.items():
        actual = actual_by_name.get(name)
        if actual is None:
            blockers.append(f"archive_member_missing:{name}")
            continue
        for field in ("file_size", "compress_size", "compress_type", "crc32", "sha256"):
            if actual.get(field) != expected.get(field):
                blockers.append(f"archive_member_{field}_mismatch:{name}")

    compressed_bytes = sum(int(member.get("compress_size") or 0) for member in members)
    observed["zip_overhead_bytes"] = observed.get("bytes", 0) - compressed_bytes
    if archive_meta.get("zip_overhead_bytes") != observed["zip_overhead_bytes"]:
        blockers.append("archive_zip_overhead_mismatch")

    return observed, sorted(dict.fromkeys(blockers))


def _materialized_runtime_fallback_root(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    adapter_rel_path: str,
) -> tuple[Path | None, dict[str, Any], list[str]]:
    info: dict[str, Any] = {}
    source_meta = entry.get("source_runtime_artifacts")
    if not isinstance(source_meta, dict):
        return None, info, ["missing_source_runtime_artifacts"]

    raw_runtime_source_path = source_meta.get("runtime_source_path")
    if not isinstance(raw_runtime_source_path, str) or not raw_runtime_source_path:
        return None, info, ["missing_runtime_source_path_for_materialized_fallback"]
    try:
        runtime_source_path = _safe_runtime_source_path(raw_runtime_source_path)
    except ValueError as exc:
        return None, info, [str(exc)]

    adapter_path = PurePosixPath(adapter_rel_path)
    if adapter_path.is_absolute() or ".." in adapter_path.parts:
        return None, info, [f"adapter_rel_path must be safe repo-relative path: {adapter_rel_path!r}"]

    fallback = repo_root / Path(adapter_path.parent) / "runtime_source" / Path(runtime_source_path)
    info["materialized_fallback_root"] = _repo_rel(fallback, repo_root)
    return fallback, info, []


def _runtime_source_root(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    adapter_rel_path: str,
) -> tuple[Path | None, list[str], dict[str, Any]]:
    source_meta = entry.get("source_runtime_artifacts")
    if not isinstance(source_meta, dict):
        return None, ["missing_source_runtime_artifacts"], {"source_resolution": "missing"}
    raw = source_meta.get("local_source_root")
    blockers: list[str] = []
    info: dict[str, Any] = {
        "source_resolution": "missing",
        "manifest_source_root": raw if isinstance(raw, str) and raw else None,
    }
    root: Path | None = None
    if isinstance(raw, str) and raw:
        try:
            root = _path_from_manifest(
                raw,
                repo_root=repo_root,
                label="source_runtime_artifacts.local_source_root",
            )
        except ValueError as exc:
            return None, [str(exc)], info
        info["manifest_source_root"] = _repo_rel(root, repo_root)
        if root.is_dir():
            info["source_resolution"] = "manifest_local_source_root"
            return root, [], info
        blockers.append("runtime_source_root_missing")
    else:
        blockers.append("missing_pr102_local_source_root")

    fallback, fallback_info, fallback_blockers = _materialized_runtime_fallback_root(
        entry,
        repo_root=repo_root,
        adapter_rel_path=adapter_rel_path,
    )
    info.update(fallback_info)
    if fallback_blockers:
        blockers.extend(fallback_blockers)
    elif fallback is not None and fallback.is_dir():
        info["source_resolution"] = "materialized_adapter_runtime_fallback"
        info["source_resolution_note"] = (
            "manifest local_source_root is unavailable; using committed adapter runtime_source copy"
        )
        return fallback, [], info
    return root, blockers, info


def _listed_runtime_files(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    source_meta = entry.get("source_runtime_artifacts")
    if not isinstance(source_meta, dict):
        return [], ["missing_source_runtime_artifacts"]
    files = source_meta.get("files")
    if not isinstance(files, list) or not files:
        return [], ["runtime_source_file_manifest_missing"]
    rows = [row for row in files if isinstance(row, dict)]
    if len(rows) != len(files):
        return rows, ["runtime_source_file_manifest_non_object_entry"]
    return rows, []


def _discover_runtime_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        if "__pycache__" in path.relative_to(root).parts:
            continue
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        files.append(path)
    return files


def _runtime_file_records(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    adapter_rel_path: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    root, root_blockers, root_info = _runtime_source_root(
        entry,
        repo_root=repo_root,
        adapter_rel_path=adapter_rel_path,
    )
    blockers.extend(root_blockers)
    rows, row_blockers = _listed_runtime_files(entry)
    blockers.extend(row_blockers)

    if root is None:
        return {"source_root": None, "files": [], **root_info}, sorted(dict.fromkeys(blockers))

    file_records: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    for row in rows:
        raw = row.get("path")
        if not isinstance(raw, str) or not raw:
            blockers.append("runtime_source_file_path_missing")
            continue
        rel = PurePosixPath(raw)
        if rel.is_absolute() or ".." in rel.parts:
            blockers.append(f"runtime_source_file_unsafe_path:{raw}")
            continue
        declared_paths.add(rel.as_posix())
        path = root / Path(rel)
        record: dict[str, Any] = {
            "path": rel.as_posix(),
            "repo_relative_path": _repo_rel(path, repo_root),
            "expected_sha256": row.get("sha256"),
            "git_blob_sha1": row.get("git_blob_sha1"),
            "role": row.get("role"),
        }
        if not path.is_file():
            blockers.append(f"runtime_source_file_missing:{raw}")
            record["exists"] = False
        else:
            record["exists"] = True
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)
            if record["sha256"] != row.get("sha256"):
                blockers.append(f"runtime_source_file_sha256_mismatch:{raw}")
        file_records.append(record)

    discovered_paths = {path.relative_to(root).as_posix() for path in _discover_runtime_files(root)}
    if not REQUIRED_RUNTIME_FILES.issubset(discovered_paths):
        missing = sorted(REQUIRED_RUNTIME_FILES - discovered_paths)
        blockers.extend(f"runtime_required_file_missing:{path}" for path in missing)
    if declared_paths != discovered_paths:
        extra = sorted(discovered_paths - declared_paths)
        missing = sorted(declared_paths - discovered_paths)
        blockers.extend(f"runtime_source_unmanifested_file:{path}" for path in extra)
        blockers.extend(f"runtime_source_manifested_file_not_discovered:{path}" for path in missing)

    source_meta = entry.get("source_runtime_artifacts") if isinstance(entry.get("source_runtime_artifacts"), dict) else {}
    checkout_root = root.parents[1] if len(root.parents) >= 2 else root
    manifest_payload = [
        f"{record.get('path')} {record.get('bytes')} {record.get('sha256')}"
        for record in sorted(file_records, key=lambda item: str(item.get("path")))
    ]
    runtime = {
        "runtime_source_path": source_meta.get("runtime_source_path"),
        "source_root": _repo_rel(root, repo_root),
        **root_info,
        "public_checkout_root": _repo_rel(checkout_root, repo_root),
        "file_count": len(file_records),
        "source_tree_sha256": hashlib.sha256(
            "\n".join(manifest_payload).encode("utf-8")
        ).hexdigest(),
        "files": file_records,
    }
    return runtime, sorted(dict.fromkeys(blockers))


def _python_external_imports(path: Path) -> tuple[list[str], str | None]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [], f"{exc.__class__.__name__}: {exc}"
    imports: set[str] = set()
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    external = sorted(
        name
        for name in imports
        if name not in stdlib and name not in LOCAL_MODULE_NAMES and name != "__future__"
    )
    return external, None


def _scan_runtime_risks(runtime: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    external_imports: set[str] = set()
    import_parse_errors: list[dict[str, str]] = []
    for row in runtime.get("files", []):
        rel_path = row.get("repo_relative_path")
        if not isinstance(rel_path, str):
            continue
        path = repo_root / rel_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if NETWORK_OR_INSTALL_RE.search(line):
                findings.append(
                    {
                        "path": rel_path,
                        "line": line_no,
                        "kind": "network_or_install_token",
                        "text": line.strip(),
                    }
                )
        if path.suffix == ".py":
            imports, error = _python_external_imports(path)
            external_imports.update(imports)
            if error:
                import_parse_errors.append({"path": rel_path, "error": error})

    required_modules = sorted(module for module in external_imports if module in {"brotli", "cv2", "numpy", "torch"})
    return {
        "manifest_compliance_risks": [],
        "network_or_install_findings": findings,
        "external_python_imports": sorted(external_imports),
        "required_python_modules_for_adapter_preflight": required_modules,
        "python_import_parse_errors": import_parse_errors,
        "adapter_dependency_policy": (
            "fail_closed_preinstalled_dependencies_only; the adapter must exit before public runtime import "
            "if required modules are unavailable"
        ),
    }


def _contest_auth_eval_command_text(tokens: list[str]) -> str:
    if not tokens:
        return ""
    continuation = " " + "\\" + "\n  "
    head = " ".join(tokens[:2])
    option_pairs = [" ".join(tokens[index : index + 2]) for index in range(2, len(tokens), 2)]
    return head + continuation + continuation.join(option_pairs)


def _command_text(tokens: list[str]) -> str:
    if not tokens:
        return ""
    chunks: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token.startswith("--") and current:
            chunks.append(" ".join(current))
            current = [token]
        else:
            current.append(token)
    if current:
        chunks.append(" ".join(current))
    return " \\\n  ".join(chunks)


def _adapter_planned_artifacts(
    *,
    adapter_rel_path: str,
    runtime: dict[str, Any],
    adapter_plan: dict[str, Any],
) -> list[str]:
    materialized = adapter_plan.get("materialized_files")
    if isinstance(materialized, list) and all(isinstance(item, str) for item in materialized):
        return list(materialized)

    adapter_path = PurePosixPath(adapter_rel_path)
    adapter_dir = adapter_path.parent
    runtime_source_path = _safe_runtime_source_path(str(runtime.get("runtime_source_path") or ""))
    planned = [
        adapter_path.as_posix(),
        (adapter_dir / "README.md").as_posix(),
    ]
    for row in runtime.get("files", []):
        rel = PurePosixPath(str(row.get("path") or ""))
        if not rel.as_posix() or rel.is_absolute() or ".." in rel.parts:
            continue
        planned.append((adapter_dir / "runtime_source" / runtime_source_path / rel).as_posix())
    return planned


def _readiness_artifact_defaults(adapter_rel_path: str) -> list[str]:
    adapter_dir = PurePosixPath(adapter_rel_path).parent
    return [
        (adapter_dir / "readiness.json").as_posix(),
        (adapter_dir / "readiness.md").as_posix(),
    ]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _build_lightning_exact_eval_runbook(
    *,
    archive: dict[str, Any],
    runtime: dict[str, Any],
    adapter_plan: dict[str, Any],
    adapter_rel_path: str,
) -> dict[str, Any]:
    """Return a no-dispatch Lightning source-manifest runbook for PR102."""
    lane_id = DEFAULT_LIGHTNING_LANE_ID
    job_name = DEFAULT_LIGHTNING_JOB_NAME
    readiness_artifacts = _readiness_artifact_defaults(adapter_rel_path)
    adapter_artifacts = _adapter_planned_artifacts(
        adapter_rel_path=adapter_rel_path,
        runtime=runtime,
        adapter_plan=adapter_plan,
    )
    source_manifest_expected = _dedupe(
        [
            str(archive.get("path") or ""),
            *adapter_artifacts,
            *readiness_artifacts,
        ]
    )
    source_manifest_expected = [item for item in source_manifest_expected if item]
    archive_sha = str(archive.get("sha256") or "")
    claim_command = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        lane_id,
        "--platform",
        "lightning",
        "--instance-job-id",
        job_name,
        "--agent",
        "${AGENT_ID}",
        "--predicted-eta-utc",
        "${ETA_UTC}",
        "--status",
        "eval",
        "--notes",
        f"PR102_exact_replay_archive_{archive_sha[:12]}_no_score_claim",
    ]
    wrapper_submit_command = [
        ".venv/bin/python",
        "scripts/lightning_exact_eval_repro.py",
        "--job-name",
        job_name,
        "--archive",
        str(archive.get("path") or ""),
        "--inflate-sh",
        adapter_rel_path,
        "--stage-workspace",
        "--remote",
        "${LIGHTNING_SSH_TARGET}",
        "--studio",
        "${LIGHTNING_STUDIO}",
        "--teamspace",
        "${LIGHTNING_TEAMSPACE}",
        "--sdk-user",
        "${LIGHTNING_SDK_USER}",
        "--requirements-mode",
        "verify-only",
        "--python-bin",
        ".venv/bin/python",
        "--machine",
        "${LIGHTNING_MACHINE:-" + DEFAULT_LIGHTNING_MACHINE + "}",
        "--baseline-score",
        "${EXACT_BASELINE_SCORE}",
        "--predicted-band",
        "${PREDICTED_LOW}",
        "${PREDICTED_HIGH}",
        "--regression-threshold",
        "${REGRESSION_THRESHOLD}",
        "--dispatch-lane-id",
        lane_id,
        "--dispatch-claims-path",
        ".omx/state/active_lane_dispatch_claims.md",
        "--extra-artifact",
        readiness_artifacts[0],
        "--extra-artifact",
        readiness_artifacts[1],
        "--queue-metadata",
        "source_prs=102",
        "--queue-metadata",
        f"pr102_readiness={readiness_artifacts[0]}",
        "--env",
        "INFLATE_TORCH_SPEC=torch==2.5.1+cu124",
        "--env",
        "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124",
        "--env",
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "--component-trace",
        "--component-trace-top-k",
        "96",
        "--submit",
    ]
    return {
        "lane_id": lane_id,
        "job_name": job_name,
        "claim_command": claim_command,
        "claim_command_text": _command_text(claim_command),
        "wrapper_submit_command": wrapper_submit_command,
        "wrapper_submit_command_text": _command_text(wrapper_submit_command),
        "readiness_artifacts": readiness_artifacts,
        "adapter_runtime_artifacts": adapter_artifacts,
        "source_manifest_expected_artifacts": source_manifest_expected,
        "guardrails": [
            "claim lane before submit; do not use --allow-missing-dispatch-claim-reason for normal PR102 replay",
            "clean clones must restore the external archive artifact and verify bytes/SHA before exact replay; the adapter only carries runtime source",
            "submit through scripts/lightning_exact_eval_repro.py with --stage-workspace so source_manifest is created and verified before launch",
            "do not require CUDA in the interactive Studio staging shell; the Batch runner performs the canonical CUDA preflight",
            f"use concrete Lightning machine {DEFAULT_LIGHTNING_MACHINE} on comma-lab AWS unless a refreshed machine inventory proves another alias works",
            "keep --requirements-mode verify-only unless a separate remote environment build is explicitly recorded",
            "preserve readiness JSON/markdown as explicit extra artifacts in the staged source manifest",
        ],
    }


def _adapter_script_text(
    *,
    runtime_source_root_rel: str,
    public_checkout_root_rel: str,
    module_name: str,
    required_modules: list[str],
) -> str:
    modules_literal = " ".join(required_modules)
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Source-sized public replay shim for PR102. Do not install packages here.
# PACT_RUNTIME_DEPENDENCY_ROOT = {runtime_source_root_rel}

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

find_repo_root() {{
  local dir="$HERE"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/experiments/contest_auth_eval.py" ]; then
      printf '%s\\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}}

if [ -n "${{PACT_REPO_ROOT:-}}" ]; then
  REPO_ROOT="$PACT_REPO_ROOT"
else
  REPO_ROOT="$(find_repo_root)" || {{
    echo "ERROR: could not find repo root; set PACT_REPO_ROOT" >&2
    exit 1
  }}
fi
PYTHON="${{PACT_PYTHON:-$REPO_ROOT/.venv/bin/python}}"
PUBLIC_SOURCE_ROOT="$REPO_ROOT/{public_checkout_root_rel}"
RUNTIME_SOURCE_ROOT="$REPO_ROOT/{runtime_source_root_rel}"

if [ "$#" -ne 3 ]; then
  echo "ERROR: expected DATA_DIR OUTPUT_DIR FILE_LIST arguments" >&2
  exit 2
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python not executable: $PYTHON" >&2
  exit 1
fi
if [ ! -d "$PUBLIC_SOURCE_ROOT" ] || [ ! -d "$RUNTIME_SOURCE_ROOT" ]; then
  echo "ERROR: PR102 public source runtime missing" >&2
  echo "PUBLIC_SOURCE_ROOT=$PUBLIC_SOURCE_ROOT" >&2
  echo "RUNTIME_SOURCE_ROOT=$RUNTIME_SOURCE_ROOT" >&2
  exit 1
fi

for module in {modules_literal}; do
  "$PYTHON" - "$module" <<'PY'
import importlib.util
import sys

module = sys.argv[1]
if importlib.util.find_spec(module) is None:
    raise SystemExit(f"ERROR: required PR102 runtime dependency missing: {{module}}")
PY
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PUBLIC_SOURCE_ROOT:$RUNTIME_SOURCE_ROOT"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${{line%.*}}"
  SRC="$DATA_DIR/${{BASE}}.bin"
  DST="$OUTPUT_DIR/${{BASE}}.raw"
  if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found" >&2
    exit 1
  fi
  cd "$PUBLIC_SOURCE_ROOT"
  "$PYTHON" -m "{module_name}" "$SRC" "$DST"
done < "$FILE_LIST"
"""


def _build_adapter_plan(
    *,
    archive: dict[str, Any],
    runtime: dict[str, Any],
    risks: dict[str, Any],
    adapter_rel_path: str,
) -> dict[str, Any]:
    runtime_root = str(runtime["source_root"])
    public_checkout_root = str(runtime["public_checkout_root"])
    module_name = "submissions.hnerv_lc_v2_scale095_rplus1.inflate"
    script_text = _adapter_script_text(
        runtime_source_root_rel=runtime_root,
        public_checkout_root_rel=public_checkout_root,
        module_name=module_name,
        required_modules=list(risks.get("required_python_modules_for_adapter_preflight") or []),
    )
    command = [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        str(archive["path"]),
        "--inflate-sh",
        adapter_rel_path,
        "--upstream-dir",
        "upstream",
        "--device",
        "cuda",
    ]
    return {
        "adapter_rel_path": adapter_rel_path,
        "runtime_dependency_root_directive": runtime_root,
        "module": module_name,
        "inflate_sh_sha256": hashlib.sha256(script_text.encode("utf-8")).hexdigest(),
        "inflate_sh_text": script_text,
        "contest_auth_eval_command": command,
        "contest_auth_eval_command_text": _contest_auth_eval_command_text(command),
        "dispatch_attempted": False,
        "score_claim": False,
    }


def build_pr102_exact_replay_readiness(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = REPO_ROOT,
    pr_number: int = 102,
    adapter_rel_path: str = DEFAULT_ADAPTER_REL_PATH,
) -> AuditReport:
    blockers: list[str] = []
    manifest: dict[str, Any] = {}
    try:
        loaded = read_json(manifest_path)
        if not isinstance(loaded, dict):
            raise ValueError("manifest root must be an object")
        manifest = loaded
        entry = _entry_for_pr(manifest, pr_number)
    except Exception as exc:
        entry = {}
        blockers.append(f"manifest_load_failed:{exc}")

    archive, archive_blockers = _validate_archive(entry, repo_root=repo_root) if entry else ({}, [])
    runtime, runtime_blockers = (
        _runtime_file_records(entry, repo_root=repo_root, adapter_rel_path=adapter_rel_path)
        if entry
        else ({}, [])
    )
    blockers.extend(archive_blockers)
    blockers.extend(runtime_blockers)

    risks = _scan_runtime_risks(runtime, repo_root=repo_root) if runtime else {}
    if isinstance(entry.get("compliance_risks"), list):
        risks["manifest_compliance_risks"] = list(entry["compliance_risks"])

    if risks.get("python_import_parse_errors"):
        blockers.append("runtime_python_import_parse_errors")

    if archive and runtime and not blockers:
        adapter_plan = _build_adapter_plan(
            archive=archive,
            runtime=runtime,
            risks=risks,
            adapter_rel_path=adapter_rel_path,
        )
    else:
        adapter_plan = {
            "adapter_rel_path": adapter_rel_path,
            "contest_auth_eval_command": [],
            "contest_auth_eval_command_text": "",
            "dispatch_attempted": False,
            "score_claim": False,
        }

    observations = entry.get("public_eval_observations") if isinstance(entry.get("public_eval_observations"), dict) else {}
    summary = {
        "pr_number": pr_number,
        "title": entry.get("title"),
        "url": entry.get("url"),
        "head_sha": entry.get("head_sha"),
        "manifest": _repo_rel(manifest_path, repo_root),
        "manifest_created_at_utc": manifest.get("created_at_utc"),
        "evidence_grade": manifest.get("evidence_grade"),
        "archive": archive,
        "runtime_source": runtime,
        "dependency_network_risks": risks,
        "public_eval_observations": observations,
        "local_exact_cuda_status": observations.get("local_exact_cuda_status"),
        "adapter_plan": adapter_plan,
        "lightning_exact_eval_runbook": _build_lightning_exact_eval_runbook(
            archive=archive,
            runtime=runtime,
            adapter_plan=adapter_plan,
            adapter_rel_path=adapter_rel_path,
        )
        if archive and runtime and not blockers
        else {},
        "next_status_after_this_artifact": "ready_to_materialize_adapter_then_run_exact_cuda_replay"
        if not blockers
        else "blocked_before_adapter_materialization",
    }
    return AuditReport(
        audit="pr102_exact_replay_readiness",
        readiness_key="adapter_plan_ready",
        ready=not blockers,
        blockers=tuple(sorted(dict.fromkeys(blockers))),
        summary=summary,
        score_claim=False,
        dispatch_attempted=False,
        metadata={
            "schema": "pr102_exact_replay_readiness_v1",
            "source_sized_artifact": True,
        },
    )


def materialize_adapter_plan(
    report: AuditReport,
    *,
    adapter_dir: Path,
    repo_root: Path = REPO_ROOT,
    overwrite: bool = False,
) -> AuditReport:
    if not report.ready:
        raise ValueError("refusing to materialize adapter for a failed readiness report")
    resolved = adapter_dir.resolve()
    omx_state = (repo_root / ".omx/state").resolve()
    try:
        resolved.relative_to(omx_state)
    except ValueError:
        pass
    else:
        raise ValueError("refusing to write adapter materialization under .omx/state")

    if resolved.exists() and any(resolved.iterdir()) and not overwrite:
        raise ValueError(f"adapter dir is not empty: {adapter_dir}")
    resolved.mkdir(parents=True, exist_ok=True)
    runtime_checkout = resolved / "runtime_source"
    runtime_source_rel = _safe_runtime_source_path(
        str(report.summary["runtime_source"]["runtime_source_path"])
    )
    runtime_root = runtime_checkout / Path(runtime_source_rel)
    copied_runtime_files = _copy_runtime_source_files(
        report,
        runtime_root=runtime_root,
        repo_root=repo_root,
    )
    adapter_plan = dict(report.summary["adapter_plan"])
    required_modules = list(
        report.summary["dependency_network_risks"].get(
            "required_python_modules_for_adapter_preflight"
        )
        or []
    )
    script_text = _adapter_script_text(
        runtime_source_root_rel=_repo_rel(runtime_root, repo_root),
        public_checkout_root_rel=_repo_rel(runtime_checkout, repo_root),
        module_name=str(adapter_plan["module"]),
        required_modules=required_modules,
    )
    inflate_sh = resolved / "inflate.sh"
    inflate_sh.write_text(script_text, encoding="utf-8")
    inflate_sh.chmod(inflate_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    readme = resolved / "README.md"
    readme.write_text(
        "# PR102 Exact Replay Adapter\n\n"
        "Source-sized, fail-closed adapter for later CUDA replay. This directory was "
        "materialized by `tools/build_pr102_exact_replay_readiness.py`, includes "
        "the public PR102 runtime source needed by `inflate.sh`, and does not "
        "contain archive payload bytes.\n",
        encoding="utf-8",
    )
    summary = dict(report.summary)
    adapter_plan["runtime_dependency_root_directive"] = _repo_rel(runtime_root, repo_root)
    adapter_plan["inflate_sh_text"] = script_text
    adapter_plan["inflate_sh_sha256"] = hashlib.sha256(script_text.encode("utf-8")).hexdigest()
    adapter_plan["materialized_files"] = [
        _repo_rel(inflate_sh, repo_root),
        _repo_rel(readme, repo_root),
        *copied_runtime_files,
    ]
    summary["adapter_plan"] = adapter_plan
    summary["lightning_exact_eval_runbook"] = _build_lightning_exact_eval_runbook(
        archive=summary["archive"],
        runtime=summary["runtime_source"],
        adapter_plan=adapter_plan,
        adapter_rel_path=str(adapter_plan["adapter_rel_path"]),
    )
    summary["next_status_after_this_artifact"] = (
        "adapter_materialized_ready_for_exact_cuda_replay_after_dispatch_claim"
    )
    return replace(report, summary=summary)


def _safe_runtime_source_path(raw: str) -> str:
    rel = PurePosixPath(raw)
    if not raw or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe runtime_source_path: {raw!r}")
    return rel.as_posix().rstrip("/")


def _copy_runtime_source_files(
    report: AuditReport,
    *,
    runtime_root: Path,
    repo_root: Path,
) -> list[str]:
    copied: list[str] = []
    for row in report.summary["runtime_source"].get("files", []):
        rel = PurePosixPath(str(row.get("path") or ""))
        source_rel = str(row.get("repo_relative_path") or "")
        if not rel.as_posix() or rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe runtime source file path: {rel}")
        source = repo_root / source_rel
        if not source.is_file():
            raise ValueError(f"runtime source file missing during materialization: {source_rel}")
        target = runtime_root / Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        copied.append(_repo_rel(target, repo_root))
    return copied


def render_markdown(report: AuditReport) -> str:
    payload = report.to_dict()
    summary = payload["summary"]
    archive = summary["archive"]
    runtime = summary["runtime_source"]
    risks = summary["dependency_network_risks"]
    adapter = summary["adapter_plan"]
    status = "PASS" if payload["adapter_plan_ready"] else "FAIL"

    lines = [
        "# PR102 Exact Replay Readiness - 2026-05-08",
        "",
        f"Status: `{status}`. Score claim: `false`. Dispatch attempted: `false`.",
        "",
        "## Archive Custody",
        "",
        f"- Path: `{archive.get('path')}`",
        f"- Bytes: `{archive.get('bytes')}`",
        f"- SHA-256: `{archive.get('sha256')}`",
        f"- Canonical URL: `{archive.get('canonical_url')}`",
        "",
        "## Runtime Source Files",
        "",
        f"- Source root: `{runtime.get('source_root')}`",
        f"- Source tree SHA-256: `{runtime.get('source_tree_sha256')}`",
        "",
        "| path | bytes | sha256 | role |",
        "| --- | ---: | --- | --- |",
    ]
    for row in runtime.get("files", []):
        role = str(row.get("role") or "")
        lines.append(
            f"| `{row.get('path')}` | `{row.get('bytes')}` | `{row.get('sha256')}` | {role} |"
        )

    lines.extend(
        [
            "",
            "## Dependency And Network Risks",
            "",
            f"- Adapter policy: `{risks.get('adapter_dependency_policy')}`",
            "- Required preinstalled Python modules: "
            f"`{', '.join(risks.get('required_python_modules_for_adapter_preflight') or [])}`",
            "",
            "Manifest compliance risks:",
        ]
    )
    for risk in risks.get("manifest_compliance_risks", []):
        lines.append(f"- {risk}")
    lines.append("")
    lines.append("Static network/install findings:")
    for finding in risks.get("network_or_install_findings", []):
        lines.append(f"- `{finding['path']}:{finding['line']}`: `{finding['text']}`")

    lines.extend(
        [
            "",
            "## Exact Replay Command",
            "",
            "Run only after ensuring dependencies are preinstalled and a remote dispatch claim is active:",
            "",
            "```bash",
            adapter.get("contest_auth_eval_command_text", ""),
            "```",
            "",
            "## Adapter Materialization",
            "",
        ]
    )
    materialized = adapter.get("materialized_files") or []
    if materialized:
        lines.extend(f"- `{path}`" for path in materialized)
    else:
        lines.append("- Not materialized in this report.")

    runbook = summary.get("lightning_exact_eval_runbook") or {}
    if runbook:
        lines.extend(
            [
                "",
                "## Lightning Exact Eval Source Manifest Runbook",
                "",
                f"- Lane id: `{runbook.get('lane_id')}`",
                f"- Job name: `{runbook.get('job_name')}`",
                "- Normal path: claim first, then submit only through the repro wrapper so "
                "`source_manifest.json` is created, SHA-verified, and forwarded to the launcher.",
                "",
                "Source manifest must include:",
            ]
        )
        for path in runbook.get("source_manifest_expected_artifacts", []):
            lines.append(f"- `{path}`")
        lines.extend(
            [
                "",
                "Claim before submit:",
                "",
                "```bash",
                str(runbook.get("claim_command_text") or ""),
                "```",
                "",
                "Submit through the wrapper:",
                "",
                "```bash",
                str(runbook.get("wrapper_submit_command_text") or ""),
                "```",
                "",
                "Guardrails:",
            ]
        )
        for guardrail in runbook.get("guardrails", []):
            lines.append(f"- {guardrail}")

    lines.extend(
        [
            "",
            "## Adapter Inflate Script",
            "",
            "```bash",
            adapter.get("inflate_sh_text", "").rstrip(),
            "```",
            "",
            "## Blockers",
            "",
        ]
    )
    if payload["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    else:
        lines.append("- None for source-sized adapter-plan readiness. Exact CUDA replay is still missing.")
    lines.append("")
    return "\n".join(lines)


def _write_output(path: Path, text: str, *, repo_root: Path) -> None:
    resolved = path.resolve()
    omx_state = (repo_root / ".omx/state").resolve()
    try:
        resolved.relative_to(omx_state)
    except ValueError:
        pass
    else:
        raise ValueError("refusing to write output under .omx/state")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--pr-number", type=int, default=102)
    parser.add_argument("--adapter-rel-path", default=DEFAULT_ADAPTER_REL_PATH)
    parser.add_argument("--materialize-adapter-dir", type=Path)
    parser.add_argument("--overwrite-adapter", action="store_true")
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    report = build_pr102_exact_replay_readiness(
        manifest_path=args.manifest,
        repo_root=repo_root,
        pr_number=args.pr_number,
        adapter_rel_path=args.adapter_rel_path,
    )
    if args.materialize_adapter_dir is not None:
        report = materialize_adapter_plan(
            report,
            adapter_dir=args.materialize_adapter_dir,
            repo_root=repo_root,
            overwrite=args.overwrite_adapter,
        )

    if args.format == "json":
        text = json_text(report.to_dict())
    elif args.format == "markdown":
        text = render_markdown(report)
    else:
        text = report.render_text(pass_detail=f"archive={report.summary.get('archive', {}).get('sha256')}")
        text += "\n"

    if args.output is not None:
        _write_output(args.output, text, repo_root=repo_root)
    else:
        print(text, end="")
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
