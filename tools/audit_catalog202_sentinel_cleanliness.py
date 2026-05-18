#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit whether a Modal recipe can use the Catalog #202 dirty-tree bypass.

Catalog #202 lets an operator bypass ``--require-clean-head`` only when the
explicit Modal sentinel set has been independently verified clean. This helper
does not dispatch, claim a lane, or approve spend. It turns that verification
into a byte-level artifact: effective sentinel list, SHA-256 ledger, git dirty
status, and an explicit pass/block verdict for the paired env attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - CLI failure path.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / ".omx" / "operator_authorize_recipes"
CATALOG202_BYPASS_INTENT_ENV_VAR = "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"  # OPERATOR_AUTHORIZE_CLEAN_BYPASS_OK:constant-only-no-env-set
CATALOG202_BYPASS_ATTESTATION_ENV_VAR = (
    "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED"
)
CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR = "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON"

# Keep synchronized with tools/operator_authorize.py::_MODAL_MOUNT_SET_PREFIXES.
MODAL_MOUNT_SET_PREFIXES: tuple[str, ...] = (
    "src/",
    "scripts/",
    "upstream/",
    "submissions/",
    "experiments/",
    "tools/",
    "pyproject.toml",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required; install pyyaml in the Pact venv.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"recipe did not parse to a mapping: {path}")
    return data


def resolve_recipe_path(recipe: str, *, repo_root: Path = REPO_ROOT) -> Path:
    candidate = Path(recipe)
    if candidate.exists():
        return candidate.resolve()
    path = repo_root / ".omx" / "operator_authorize_recipes" / f"{recipe}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def path_under_modal_mount_set(rel: str) -> bool:
    rel = rel.strip()
    while rel.startswith("./"):
        rel = rel[2:]
    return any(
        rel.startswith(prefix) if prefix.endswith("/") else rel == prefix
        for prefix in MODAL_MOUNT_SET_PREFIXES
    )


def _iter_declared_path_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [item for item in value if isinstance(item, str)]
    else:
        values = [str(value)]
    return [item.strip() for item in values if item.strip()]


def collect_sentinel_candidates(recipe: dict[str, Any]) -> list[str]:
    """Mirror operator_authorize.py's effective Modal sentinel candidate list."""

    paths: list[str] = [
        "experiments/modal_train_lane.py",
        "tools/operator_authorize.py",
        "tools/run_modal_smoke_before_full.py",
        "src/tac/deploy/modal/mount_manifest.py",
    ]
    remote_driver = recipe.get("remote_driver")
    if remote_driver:
        paths.append(str(remote_driver))
    modal_cfg = recipe.get("modal", {}) or {}
    if isinstance(modal_cfg, dict):
        for key in ("lane_script", "cost_band_trainer"):
            value = modal_cfg.get(key)
            if value:
                paths.append(str(value))
    trainer = recipe.get("required_input_files_trainer")
    paths.extend(_iter_declared_path_values(trainer))
    recipe_sentinels = recipe.get("sentinel_files") or []
    if isinstance(recipe_sentinels, list):
        for entry in recipe_sentinels:
            paths.extend(_iter_declared_path_values(entry))
    return paths


def effective_sentinel_files(
    recipe: dict[str, Any], *, repo_root: Path = REPO_ROOT
) -> tuple[list[str], list[str], list[str]]:
    """Return (effective, missing, outside_mount) sentinels."""

    effective: list[str] = []
    missing: list[str] = []
    outside_mount: list[str] = []
    seen: set[str] = set()
    for raw in collect_sentinel_candidates(recipe):
        rel = raw.strip()
        while rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel in seen:
            continue
        seen.add(rel)
        if not (repo_root / rel).is_file():
            missing.append(rel)
            continue
        if not path_under_modal_mount_set(rel):
            outside_mount.append(rel)
            continue
        effective.append(rel)
    return effective, missing, outside_mount


def git_status_paths(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git status failed")
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        rel = line[3:] if len(line) > 3 else ""
        if " -> " in rel:
            old, new = rel.split(" -> ", 1)
            out[old] = status
            out[new] = status
        elif rel:
            out[rel] = status
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sentinel_set_sha256(records: list[dict[str, Any]]) -> str:
    payload = {
        row["path"]: row["sha256"]
        for row in records
        if row.get("exists") and row.get("sha256")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_audit(recipe_name: str, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    recipe_path = resolve_recipe_path(recipe_name, repo_root=repo_root)
    try:
        recipe_relpath = str(recipe_path.relative_to(repo_root))
    except ValueError:
        recipe_relpath = str(recipe_path)
    recipe = _load_yaml(recipe_path)
    effective, missing, outside_mount = effective_sentinel_files(
        recipe, repo_root=repo_root
    )
    status = git_status_paths(repo_root)
    dirty_paths = sorted(status)
    dirty_path_set = set(dirty_paths)

    records: list[dict[str, Any]] = []
    for rel in effective:
        path = repo_root / rel
        st = path.stat()
        records.append(
            {
                "path": rel,
                "exists": True,
                "size_bytes": st.st_size,
                "sha256": sha256_file(path),
                "git_status": status.get(rel),
                "dirty_in_git": rel in dirty_path_set,
                "under_modal_mount_set": path_under_modal_mount_set(rel),
            }
        )

    dirty_sentinels = sorted(
        row["path"] for row in records if row.get("dirty_in_git")
    )
    effective_set = {row["path"] for row in records}
    dirty_mounted_non_sentinels = sorted(
        rel
        for rel in dirty_paths
        if path_under_modal_mount_set(rel) and rel not in effective_set
    )
    dirty_operator_side_paths = sorted(
        rel for rel in dirty_paths if not path_under_modal_mount_set(rel)
    )
    git_clean_blockers: list[str] = []
    snapshot_blockers: list[str] = []
    if missing:
        git_clean_blockers.append("catalog202_sentinel_file_missing")
        snapshot_blockers.append("catalog202_sentinel_file_missing")
    if outside_mount:
        git_clean_blockers.append("catalog202_sentinel_outside_modal_mount_set")
        snapshot_blockers.append("catalog202_sentinel_outside_modal_mount_set")
    if dirty_sentinels:
        git_clean_blockers.append("catalog202_sentinel_files_dirty_in_git")

    git_clean = not git_clean_blockers
    snapshot_ready = not snapshot_blockers
    set_hash = sentinel_set_sha256(records)
    return {
        "schema": "catalog202_sentinel_cleanliness_audit_v1",
        "generated_utc": _utc_now(),
        "recipe_name": recipe_path.stem,
        "recipe_path": recipe_relpath,
        "recipe_git_status": status.get(recipe_relpath),
        "lane_id": str(recipe.get("lane_id", "lane_unknown")),
        "platform": str(recipe.get("platform", "unknown")),
        "provider_dispatch_attempted": False,
        "lane_claim_opened": False,
        "score_claim": False,
        "promotion_eligible": False,
        "catalog202_env_contract": {
            "intent_env_var": CATALOG202_BYPASS_INTENT_ENV_VAR,
            "attestation_env_var": CATALOG202_BYPASS_ATTESTATION_ENV_VAR,
            "audit_json_env_var": CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR,
            "attestation_value_hint": f"catalog202_sentinel_audit:{set_hash}",
        },
        "sentinel_set_clean_for_catalog202": git_clean,
        "ready_for_catalog202_paired_env_attestation": git_clean,
        "attestation_blockers": git_clean_blockers,
        "sentinel_set_snapshot_stable_for_catalog202": snapshot_ready,
        "ready_for_catalog202_audit_backed_dirty_sentinel_attestation": snapshot_ready,
        "audit_backed_attestation_blockers": snapshot_blockers,
        "effective_sentinel_file_count": len(records),
        "effective_sentinel_files": [row["path"] for row in records],
        "sentinel_set_sha256": set_hash,
        "sentinel_records": records,
        "missing_sentinel_files": missing,
        "outside_modal_mount_sentinel_files": outside_mount,
        "dirty_worktree_path_count": len(dirty_paths),
        "dirty_sentinel_path_count": len(dirty_sentinels),
        "dirty_sentinel_paths": dirty_sentinels,
        "dirty_mounted_non_sentinel_path_count": len(dirty_mounted_non_sentinels),
        "dirty_mounted_non_sentinel_paths": dirty_mounted_non_sentinels,
        "dirty_operator_side_path_count": len(dirty_operator_side_paths),
        "dirty_operator_side_paths": dirty_operator_side_paths,
        "result_interpretation": (
            "Sentinel set is clean relative to git status; operator may use this "
            "artifact as evidence for the Catalog #202 paired env attestation, "
            "while still accepting that dirty non-sentinel files may be mounted."
            if git_clean
            else (
                "Sentinel files are dirty in git status, but the audit records "
                "the current effective sentinel paths and SHA-256s. A paid "
                "dispatch may use this artifact only through the audit-backed "
                "Catalog #202 path, which re-verifies these hashes at "
                "operator_authorize time."
                if snapshot_ready
                else "Catalog #202 paired env attestation is not defensible "
                "until missing/outside sentinel paths are resolved or "
                "explicitly removed from the recipe sentinel set."
            )
        ),
    }


def write_artifact(payload: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> Path:
    out_dir = repo_root / ".omx" / "state" / "catalog202_sentinel_cleanliness"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(payload["generated_utc"]).replace(":", "").replace("-", "")
    path = out_dir / f"{payload['recipe_name']}_{stamp}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="Recipe name or YAML path.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit JSON payload.")
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="Write .omx/state/catalog202_sentinel_cleanliness artifact.",
    )
    args = parser.parse_args(argv)

    payload = build_audit(args.recipe, repo_root=args.repo_root)
    if args.write_artifact:
        path = write_artifact(payload, repo_root=args.repo_root)
        print(f"[catalog202-sentinel-audit] wrote artifact: {path}", file=sys.stderr)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("=== Catalog #202 sentinel-clean audit ===")
        print(f"recipe: {payload['recipe_name']}")
        print(f"lane_id: {payload['lane_id']}")
        print(
            "sentinel_set_clean_for_catalog202: "
            f"{payload['sentinel_set_clean_for_catalog202']}"
        )
        print(f"attestation_blockers: {payload['attestation_blockers']}")
        print(
            "ready_for_catalog202_audit_backed_dirty_sentinel_attestation: "
            f"{payload['ready_for_catalog202_audit_backed_dirty_sentinel_attestation']}"
        )
        print(f"dirty_sentinel_path_count: {payload['dirty_sentinel_path_count']}")
        for rel in payload["dirty_sentinel_paths"]:
            print(f"  dirty_sentinel: {rel}")
    return 0 if payload["sentinel_set_snapshot_stable_for_catalog202"] else 1


if __name__ == "__main__":
    sys.exit(main())
