"""Disposition cs1 SSD-source debt without mutating either SSD.

The plan is appended to cs1's existing ledger before any local materialization.
Exact source bytes are stored as inert ``.blob`` objects so historical launchers
and negative fixtures cannot become active repo entrypoints.  Git preservation
is content-addressed: once committed, the original blob SHA-1 is reachable even
though the safe custody filename is not executable.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CS1_LEDGER = REPO / ".omx/research/ddm_cs1_ssd_code_certify_20260909.jsonl"
OTHER_OWED = REPO / ".omx/research/ddm_cs1_20260909/other_extensions_owed.json"
RUFF_RECEIPT = REPO / ".omx/research/ddm_sw1_20260910/blocked_ruff_before.json"
DESTINATION_ROOT = REPO / ".omx/research/ddm_sw1_20260910/recovered_ssd_blobs"
LANDING_PATCH = REPO / ".omx/research/ddm_sw1_20260910/landing.patch"
LANDING_MANIFEST = REPO / ".omx/research/ddm_sw1_20260910/landing_manifest.json"
FALLBACK_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sw1/receipts/commit_serializer_fallbacks"
)

HARD_RUFF_CODES = frozenset({"F821", "B023", "invalid-syntax"})
NON_REAL_PATH_TOKENS = (
    "/negative_control",
    "/negative_controls",
    "/retained_controls/",
    "/positive_control_",
    "/probe/test_",
    "_controls/",
    "_controls_v2/",
    "/source_hash_tamper/",
    "/source_hidden/",
    "/source_extra/",
)
PROTECTED_LIVE_TOKENS = (
    "/ddm_sj1_",
    "/ddm_rp1",
    "/ddm_bnd2",
    "/ddm_vr5",
    "/ddm_cmp2",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*=\s*['\"][^'\"]{12,}"),
)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def arm_directory(path: Path) -> str:
    parts = path.parts
    try:
        pact_index = parts.index("pact")
    except ValueError as exc:
        raise ValueError(f"path is outside a named pact SSD tree: {path}") from exc
    if pact_index + 1 >= len(parts):
        raise ValueError(f"path has no owner directory: {path}")
    return parts[pact_index + 1]


def owner_name(path: Path, prior_owner: str) -> str:
    directory = arm_directory(path)
    if prior_owner and prior_owner != "MAIN / source-owner reconciliation":
        return prior_owner.replace(" / MAIN", "")
    if directory.startswith("ddm_"):
        return directory
    match = re.match(r"scratch_([a-z0-9]+)", directory)
    if match:
        return f"ddm_{match.group(1)}"
    return f"MAIN:{directory}"


@functools.cache
def owner_memo(owner: str) -> tuple[str, str] | None:
    if owner.startswith("MAIN:"):
        token = owner.split(":", 1)[1].split("_", 1)[0]
        patterns = [f"ddm_{token}*.md"]
    else:
        short = "_".join(owner.split("_")[:2])
        patterns = [f"{owner}*.md", f"{short}*.md"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(
            path
            for path in (REPO / ".omx/research").glob(pattern)
            if path.is_file() and "charter" not in path.name.lower()
        )
    for memo in sorted(set(candidates)):
        rel = str(memo.relative_to(REPO))
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", rel], cwd=REPO, text=True
        ).strip()
        if re.fullmatch(r"[0-9a-f]{40}", commit):
            return rel, commit
    return None


def ruff_findings() -> dict[str, list[str]]:
    findings: dict[str, set[str]] = {}
    for row in json.loads(RUFF_RECEIPT.read_text()):
        findings.setdefault(row["filename"], set()).add(row["code"])
    return {path: sorted(codes) for path, codes in findings.items()}


def shell_syntax_ok(path: Path) -> bool:
    return subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, check=False
    ).returncode == 0


def classify(source_row: dict, ruff: dict[str, list[str]]) -> dict:
    source = Path(source_row.get("path") or source_row["representative_path"])
    data = source.read_bytes()
    expected_blob = source_row["blob_sha1"]
    if git_blob_sha1(data) != expected_blob:
        raise ValueError(f"source changed since cs1: {source}")
    expected_size = source_row.get("bytes", source_row.get("size_bytes"))
    if len(data) != expected_size:
        raise ValueError(f"source size changed since cs1: {source}")
    ext = source.suffix.lower()
    owner = owner_name(source, source_row.get("owner", ""))
    memo = owner_memo(owner)
    codes = ruff.get(str(source), []) if ext == ".py" else []
    reasons = []
    if memo is None:
        reasons.append("no committed owner memo")
    if any(token in str(source) for token in PROTECTED_LIVE_TOKENS):
        reasons.append("charter-protected live tree")
    if any(token in str(source) for token in NON_REAL_PATH_TOKENS):
        reasons.append("detector or adversarial control, not a real operational script")
    if set(codes) & HARD_RUFF_CODES:
        reasons.append("Ruff found invalid or correctness-risk Python")
    if ext == ".sh" and not shell_syntax_ok(source):
        reasons.append("bash -n failed")
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        reasons.append("secret-pattern review requires the source owner")
    supported = ext in {".py", ".sh", ".md", ".c", ".h", ".rs", ".toml", ".yaml", ".yml"}
    if not supported:
        reasons.append(f"unsupported source extension {ext}")
    destination = DESTINATION_ROOT / f"{expected_blob}.blob"
    disposition = "RETAIN_WITH_OWNER" if reasons else "PLANNED_EXACT_SOURCE_BLOB"
    return {
        "schema": "ddm_sw1_ssd_code_disposition.v1",
        "event_type": "disposition_plan",
        "source_ledger": str(CS1_LEDGER.relative_to(REPO)),
        "source_blob_sha1": expected_blob,
        "source_sha256": sha256(data),
        "source_bytes": len(data),
        "source_path": str(source),
        "source_extension": ext,
        "instance_count": source_row.get("instance_count", 1),
        "owner": owner,
        "owner_memo": memo[0] if memo else None,
        "owner_memo_commit": memo[1] if memo else None,
        "repo_relative_destination": str(destination.relative_to(REPO)) if not reasons else None,
        "disposition": disposition,
        "reason": "; ".join(reasons) if reasons else (
            "Exact authored bytes have committed owner evidence and passed two-pass eligibility review."
        ),
        "ruff_status": (
            "CLEAN" if ext == ".py" and not codes else
            "STYLE_OR_HYGIENE_FINDINGS_REVIEWED" if ext == ".py" else
            "NOT_APPLICABLE"
        ),
        "ruff_codes": codes,
        "shell_syntax": "PASS" if ext == ".sh" else "NOT_APPLICABLE",
        "review_pass_1": "exact hash, size, owner memo, syntax, Ruff and non-real/live-tree gates",
        "review_pass_2": "independent secret-pattern, inert-destination and destination-uniqueness review",
        "fire_trigger": (
            "Materialize exact bytes at the ledgered destination, rehash, then commit through the serializer."
            if not reasons
            else "Named owner supplies a clean exact source or a deterministic generated-artifact reproducer."
        ),
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def original_blocked(rows: list[dict]) -> list[dict]:
    base = [row for row in rows if row.get("disposition") == "BLOCKED" and not row.get("event_type")]
    if len(base) != 182 or len({row["blob_sha1"] for row in base}) != 182:
        raise ValueError("cs1 base ledger is not the expected 182 one-to-one BLOCKED set")
    return base


def plan() -> list[dict]:
    existing = load_jsonl(CS1_LEDGER)
    prior_sw1_rows = [
        row for row in existing if row.get("schema") == "ddm_sw1_ssd_code_disposition.v1"
    ]
    if prior_sw1_rows:
        raise ValueError(
            f"ddm_sw1 rows already exist ({len(prior_sw1_rows)}); refuse duplicate append"
        )
    ruff = ruff_findings()
    rows = [classify(row, ruff) for row in original_blocked(existing)]
    for item in json.loads(OTHER_OWED.read_text()):
        row = dict(item)
        row["path"] = row.pop("representative_path")
        row["bytes"] = row.pop("size_bytes")
        row["owner"] = "MAIN / source-owner reconciliation"
        planned = classify(row, ruff)
        planned["source_ledger"] = str(OTHER_OWED.relative_to(REPO))
        planned["prior_disposition"] = "OTHER_EXTENSION_OWED"
        rows.append(planned)
    destinations = [row["repo_relative_destination"] for row in rows if row["repo_relative_destination"]]
    if len(destinations) != len(set(destinations)):
        raise ValueError("two distinct rows target the same content-addressed destination")
    append_rows(CS1_LEDGER, rows)
    return rows


def materialize() -> list[dict]:
    all_rows = load_jsonl(CS1_LEDGER)
    plans = [row for row in all_rows if row.get("event_type") == "disposition_plan"]
    if len(plans) != 243:
        raise ValueError(f"expected 243 plan rows, found {len(plans)}")
    if any(row.get("event_type") == "disposition_final" for row in all_rows):
        raise ValueError("final rows already exist; refuse duplicate append")
    DESTINATION_ROOT.mkdir(parents=True, exist_ok=True)
    finals = []
    for row in plans:
        final = dict(row)
        final["event_type"] = "disposition_final"
        if row["disposition"] == "PLANNED_EXACT_SOURCE_BLOB":
            source = Path(row["source_path"])
            destination = REPO / row["repo_relative_destination"]
            data = source.read_bytes()
            if git_blob_sha1(data) != row["source_blob_sha1"] or sha256(data) != row["source_sha256"]:
                raise ValueError(f"source drifted after plan: {source}")
            if destination.exists():
                existing = destination.read_bytes()
                if existing != data:
                    raise ValueError(f"destination collision: {destination}")
            else:
                with destination.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            if git_blob_sha1(destination.read_bytes()) != row["source_blob_sha1"]:
                raise ValueError(f"materialized blob mismatch: {destination}")
            final["disposition"] = "LANDED_EXACT_SOURCE_BLOB_PENDING_COMMIT"
            final["reason"] = (
                "Exact source bytes are materialized at the pre-ledgered inert destination; Git reachability awaits serializer commit."
            )
            final["fire_trigger"] = "Serializer commit makes this exact blob reachable from a Git ref."
        finals.append(final)
    append_rows(CS1_LEDGER, finals)
    return finals


def reachable_blob_shas() -> set[str]:
    output = subprocess.check_output(
        ["git", "rev-list", "--objects", "--all"], cwd=REPO, text=True
    )
    return {line.split(" ", 1)[0] for line in output.splitlines() if line}


def finalize() -> list[dict]:
    """Append terminal custody rows only after every landed blob is ref-reachable."""
    all_rows = load_jsonl(CS1_LEDGER)
    finals = [row for row in all_rows if row.get("event_type") == "disposition_final"]
    if len(finals) != 243:
        raise ValueError(f"expected 243 final rows, found {len(finals)}")
    if any(row.get("event_type") == "disposition_commit_verified" for row in all_rows):
        raise ValueError("commit-verified rows already exist; refuse duplicate append")
    reachable = reachable_blob_shas()
    verified = []
    for row in finals:
        terminal = dict(row)
        terminal["event_type"] = "disposition_commit_verified"
        if row["disposition"] == "LANDED_EXACT_SOURCE_BLOB_PENDING_COMMIT":
            blob_sha = row["source_blob_sha1"]
            if blob_sha not in reachable:
                raise ValueError(f"source blob is not reachable from any Git ref: {blob_sha}")
            destination = row["repo_relative_destination"]
            commit = subprocess.check_output(
                ["git", "log", "-1", "--format=%H", "--", destination],
                cwd=REPO,
                text=True,
            ).strip()
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError(f"destination lacks a serializer commit: {destination}")
            terminal["disposition"] = "LANDED_EXACT_SOURCE_BLOB"
            terminal["destination_commit"] = commit
            terminal["reason"] = (
                "Exact authored source bytes are preserved as an inert, content-addressed "
                "blob reachable from a Git ref."
            )
            terminal["fire_trigger"] = "Terminal; no further action for this exact source blob."
        verified.append(terminal)
    append_rows(CS1_LEDGER, verified)
    return verified


def record_bundle_blocker() -> list[dict]:
    """Close pending rows against already-retained serializer fallback bundles."""
    all_rows = load_jsonl(CS1_LEDGER)
    finals = [row for row in all_rows if row.get("event_type") == "disposition_final"]
    if len(finals) != 243:
        raise ValueError(f"expected 243 final rows, found {len(finals)}")
    if any(row.get("event_type") == "disposition_commit_blocked" for row in all_rows):
        raise ValueError("commit-blocked rows already exist; refuse duplicate append")
    custody: dict[str, dict] = {}
    for receipt_path in sorted(FALLBACK_ROOT.glob("*/receipts.jsonl")):
        for receipt in load_jsonl(receipt_path):
            if receipt.get("event_type") != "git_object_write_denial_bundle_fallback":
                continue
            for item in receipt.get("files", []):
                custody[item["path"]] = receipt
    blocked = []
    for row in finals:
        terminal = dict(row)
        terminal["event_type"] = "disposition_commit_blocked"
        if row["disposition"] == "LANDED_EXACT_SOURCE_BLOB_PENDING_COMMIT":
            destination = row["repo_relative_destination"]
            receipt = custody.get(destination)
            if receipt is None:
                raise ValueError(f"no verified serializer fallback receipt for {destination}")
            terminal["disposition"] = "BUNDLE_READY_MAIN_MUST_LAND"
            terminal["fallback_bundle"] = receipt["bundle_path"]
            terminal["fallback_bundle_sha256"] = receipt["bundle_sha256"]
            terminal["fallback_format_patch"] = receipt["format_patch_path"]
            terminal["fallback_format_patch_sha256"] = receipt["format_patch_sha256"]
            terminal["fallback_commit"] = receipt["fallback_commit"]
            terminal["reason"] = (
                "Serializer rc=17: the sandbox denied Git object insertion. Exact bytes remain "
                "locally materialized and are also retained in a verified fallback bundle."
            )
            terminal["fire_trigger"] = (
                "MAIN with Git-object write authority verifies the bundle SHA, applies its format "
                "patch through the serializer, and reruns the authored-signal audit."
            )
        blocked.append(terminal)
    append_rows(CS1_LEDGER, blocked)
    return blocked


def write_landing_artifacts() -> list[dict]:
    """Compose the serializer's bounded fallback patches into one handoff mbox."""
    rows = load_jsonl(CS1_LEDGER)
    blocked = [row for row in rows if row.get("event_type") == "disposition_commit_blocked"]
    candidates = [row for row in blocked if row["disposition"] == "BUNDLE_READY_MAIN_MUST_LAND"]
    if len(candidates) != 223:
        raise ValueError(f"expected 223 bundle-ready rows, found {len(candidates)}")
    by_patch: dict[str, dict] = {}
    for row in candidates:
        by_patch[row["fallback_format_patch"]] = {
            "format_patch": row["fallback_format_patch"],
            "format_patch_sha256": row["fallback_format_patch_sha256"],
            "bundle": row["fallback_bundle"],
            "bundle_sha256": row["fallback_bundle_sha256"],
            "fallback_commit": row["fallback_commit"],
        }
    parts = []
    for patch_path, item in sorted(by_patch.items()):
        data = Path(patch_path).read_bytes()
        if sha256(data) != item["format_patch_sha256"]:
            raise ValueError(f"fallback format patch changed: {patch_path}")
        parts.append(data.rstrip(b"\n") + b"\n")
    combined = b"\n".join(parts)
    LANDING_PATCH.write_bytes(combined)
    manifest = {
        "schema": "ddm_sw1_landing_fallback.v1",
        "status": "BUNDLE_READY_MAIN_MUST_LAND",
        "reason": "Git object insertion was denied by the active sandbox (serializer rc=17).",
        "base_head": "951bcce1c788065120361413f54ac1a572284f9e",
        "candidate_blob_count": len(candidates),
        "batch_count": len(by_patch),
        "batch_limit": 25,
        "landing_patch": str(LANDING_PATCH.relative_to(REPO)),
        "landing_patch_sha256": sha256(combined),
        "fallback_batches": [by_patch[path] for path in sorted(by_patch)],
        "local_exact_blob_root": str(DESTINATION_ROOT.relative_to(REPO)),
        "source_ssds_mutated_by_disposition": False,
        "caveat": (
            "The serializer automatically wrote its fallback bundles under the Vertigo ddm_sw1 "
            "receipt root after Git object denial; those receipts must be retained."
        ),
        "apply_order": (
            "Verify all recorded SHA-256 values, then apply each fallback format patch as a "
            "separate serializer commit in listed order; rerun the full SSD audit afterward."
        ),
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    LANDING_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return candidates


def summary(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    return {"rows": len(rows), "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "plan",
            "materialize",
            "finalize",
            "record-bundle-blocker",
            "write-landing-artifacts",
        ),
    )
    args = parser.parse_args()
    actions = {
        "plan": plan,
        "materialize": materialize,
        "finalize": finalize,
        "record-bundle-blocker": record_bundle_blocker,
        "write-landing-artifacts": write_landing_artifacts,
    }
    rows = actions[args.action]()
    print(json.dumps(summary(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
