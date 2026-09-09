#!/usr/bin/env python3
"""Certify-or-block planner and executor for the ddm_vr3 raw reclaim.

The planner consumes the completed detached SHA census, emits one row per
chartered >=1 GiB blob, and admits the original two families or data-pinned VR4 retained chains.  Apply mode revalidates every certificate and lsof/reference gate before
unlinking a raw.  The archive, runtime, receipts, seals, and staged trees are
never mutation targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "ddm_vr3.reclaim_ledger.v1"
JOURNAL_SCHEMA = "ddm_vr3.reclaim_apply_journal.v1"
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
AP_ROOT = Path("/Volumes/APDataStore/pact")
RETAINED_STATUS = "RETAINED_REPRODUCER_VERIFIED_RAW_REHASH_AND_SAFETY_OWED"
LIVE_PREFIXES = ("ddm_sj1", "ddm_rp1", "ddm_bnd1", "ddm_gb2", "ddm_cmp2_compose")
TARGET_BYTES = 60 * (1 << 30)
AP1_ROOT = VERTIGO_ROOT / "cold_store/.omx/tmp/arm_receipts_local/ddm_ap1_residue_purchase_scorer/advisory"
JF2_ROOT = VERTIGO_ROOT / "cold_store/pact/ddm_jf2_terminal_diagonal_harvest/scorer"
AP1_TAGS = frozenset(
    {
        "carrier_l1",
        "carrier_l1_fixed_coder",
        "carrier_l2",
        "carrier_l2_fixed_coder",
        "carrier_l3",
        "carrier_l3_fixed_coder",
        "control_r2",
        "hpac_l1",
        "hpac_l2",
        "hpac_l3",
        "residual_l1",
        "residual_l2",
        "residual_l3",
        "semantic_l1_r2",
        "semantic_l2",
        "semantic_l3",
    }
)
JF2_TAGS = frozenset({"k002500_r2", "k040000_r2", "k060000_r2", "null_r2"})
PROTECTED_COMPONENT = "ddm_sj1_multipass_token_predistortion"
REFERENCE_SCOPES = (
    ".omx",
    "experiments",
    "tools",
    "docs",
    "src",
    "reports",
    "configs",
    ".ralph",
    "submissions",
)
LIVE_POINTER_REFERENCE_ROOT = VERTIGO_ROOT / PROTECTED_COMPONENT


class CertifyError(RuntimeError):
    """A fail-closed custody error."""


def valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CertifyError(f"expected JSON object: {path}")
    return payload


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_fsynced(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def df_row(path: Path) -> dict[str, Any]:
    stat = os.statvfs(path)
    return {
        "path": str(path),
        "available_bytes": int(stat.f_bavail * stat.f_frsize),
        "block_size": int(stat.f_frsize),
        "sampled_at_utc": utc_now(),
    }


def _selected_family(path: Path) -> tuple[str, str] | None:
    try:
        rel = path.relative_to(AP1_ROOT)
    except ValueError:
        pass
    else:
        if (
            len(rel.parts) == 4
            and rel.parts[0] in AP1_TAGS
            and rel.parts[1:]
            == (
                "work",
                "inflated",
                "0.raw",
            )
        ):
            return "AP1_TERMINAL_ADVISORY", rel.parts[0]
    try:
        rel = path.relative_to(JF2_ROOT)
    except ValueError:
        return None
    if (
        len(rel.parts) == 4
        and rel.parts[0] in JF2_TAGS
        and rel.parts[1:]
        == (
            "work",
            "inflated",
            "0.raw",
        )
    ):
        return "JF2_TERMINAL_ADVISORY", rel.parts[0]
    return None


def storage_root(path: Path) -> Path:
    for root in (VERTIGO_ROOT, AP_ROOT):
        if root in path.parents:
            return root
    raise CertifyError(f"outside SSD roots: {path}")


def _forbidden_target_reason(path: Path) -> str | None:
    if any(part.startswith(LIVE_PREFIXES) for part in path.parts):
        return "LIVE_POINTER_TREE_PROTECTED"
    try:
        root = storage_root(path)
    except CertifyError:
        return "OUTSIDE_SSD_ROOTS"
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return "RESOLVED_TARGET_OUTSIDE_VERTIGO_ROOT" if root == VERTIGO_ROOT else "RESOLVED_TARGET_OUTSIDE_AP_ROOT"
    if path.name != "0.raw" or path.is_symlink() or not path.is_file():
        return "TARGET_NOT_REGULAR_0_RAW"
    current = path.parent
    while current != root:
        if current.is_symlink():
            return "SYMLINK_ANCESTOR_BELOW_SSD_ROOT"
        if root not in current.parents:
            return "ANCESTOR_ESCAPES_SSD_ROOT"
        current = current.parent
    if any("SEAL" in part.upper() for part in path.parts):
        return "SEAL_TREE_NEVER_TOUCH"
    if "submissions" in path.parts:
        return "SUBMISSIONS_TREE_NEVER_TOUCH"
    if "experiments" in path.parts and "results" in path.parts and "submission_dir" in path.parts:
        return "CODEX_SUBMISSION_DIR_NEVER_TOUCH"
    return None


def _manifest_raw_record(manifest: dict[str, Any]) -> tuple[str, str]:
    payload = manifest.get("payload")
    files = payload.get("files") if isinstance(payload, dict) else manifest.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise CertifyError("decode receipt must name exactly one output file")
    record = files[0]
    name = str(record.get("relative_path") or record.get("path") or "")
    if Path(name).as_posix() != "0.raw":
        raise CertifyError(f"decode receipt output is not 0.raw: {name!r}")
    digest = str(record.get("sha256") or "")
    if not valid_sha256(digest):
        raise CertifyError("decode receipt lacks a full 0.raw SHA-256")
    field = "payload.files[0].sha256" if isinstance(payload, dict) else "files[0].sha256"
    return digest, field


def _verify_runtime_manifest(runtime: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(runtime.get("runtime_root") or ""))
    files = runtime.get("files")
    if not root.is_dir() or root.is_symlink():
        raise CertifyError(f"runtime root missing, not a directory, or a symlink: {root}")
    if not isinstance(files, list) or not files:
        raise CertifyError("runtime receipt has no file manifest")
    runtime_entries = list(root.rglob("*"))
    symlink_entries = [str(item) for item in runtime_entries if item.is_symlink()]
    if symlink_entries:
        raise CertifyError(f"runtime contains symlinks: {symlink_entries[:5]}")
    expected: dict[str, tuple[int, str]] = {}
    for record in files:
        if not isinstance(record, dict):
            raise CertifyError("runtime file row is not an object")
        rel = str(record.get("relative_path") or "")
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise CertifyError(f"unsafe runtime relative path: {rel!r}")
        if rel in expected:
            raise CertifyError(f"duplicate runtime relative path: {rel!r}")
        expected[rel] = (int(record.get("bytes") or -1), str(record.get("sha256") or ""))
    actual_paths = {
        item.relative_to(root).as_posix()
        for item in runtime_entries
        if item.is_file() and item.relative_to(root).as_posix() != "archive.zip"
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise CertifyError(f"runtime path-set drift: missing={missing[:5]} extra={extra[:5]}")
    for rel, (expected_bytes, expected_sha) in expected.items():
        target = root / rel
        if target.stat().st_size != expected_bytes:
            raise CertifyError(f"runtime byte-count drift: {target}")
        if not valid_sha256(expected_sha) or sha256_file(target) != expected_sha:
            raise CertifyError(f"runtime SHA-256 drift: {target}")
    for digest_name in (
        "runtime_tree_sha256",
        "runtime_content_tree_sha256",
        "runtime_files_sha256",
    ):
        if not valid_sha256(runtime.get(digest_name)):
            raise CertifyError(f"runtime receipt lacks {digest_name}")
    return {
        "runtime_root": str(root),
        "runtime_file_count": len(expected),
        "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": runtime.get("runtime_content_tree_sha256"),
        "runtime_files_sha256": runtime.get("runtime_files_sha256"),
        "runtime_file_manifest_verified_current": True,
        "runtime_path_set_equal": True,
    }


def certify_selected(path: Path, raw_sha256: str) -> dict[str, Any]:
    work = path.parent.parent
    receipt_path = work / "inflated_outputs_manifest.json"
    provenance_path = work / "provenance.json"
    eval_path = work / "contest_auth_eval.json"
    for required in (receipt_path, provenance_path, eval_path):
        if not required.is_file() or required.is_symlink():
            raise CertifyError(f"required receipt missing or symlinked: {required}")
    receipt = load_json(receipt_path)
    receipt_raw_sha, receipt_field = _manifest_raw_record(receipt)
    if receipt_raw_sha != raw_sha256:
        raise CertifyError(f"decode receipt/file SHA mismatch: receipt={receipt_raw_sha}:file={raw_sha256}")
    provenance = load_json(provenance_path)
    sys_argv = provenance.get("sys_argv")
    if not isinstance(sys_argv, list) or not sys_argv or not all(isinstance(arg, str) for arg in sys_argv):
        raise CertifyError("provenance lacks an exact sys_argv reproducer")
    archive = Path(str(provenance.get("archive_path") or ""))
    archive_sha = str(provenance.get("archive_sha256") or "")
    if not archive.is_file() or archive.is_symlink():
        raise CertifyError(f"reproducer archive missing or symlinked: {archive}")
    if not valid_sha256(archive_sha) or sha256_file(archive) != archive_sha:
        raise CertifyError(f"reproducer archive SHA-256 drift: {archive}")
    runtime = provenance.get("inflate_runtime_manifest")
    if not isinstance(runtime, dict):
        raise CertifyError("provenance lacks inflate_runtime_manifest")
    runtime_certificate = _verify_runtime_manifest(runtime)
    inflate_script = Path(str(provenance.get("inflate_script") or ""))
    inflate_script_sha = str(provenance.get("inflate_script_sha256") or "")
    if not inflate_script.is_file() or inflate_script.is_symlink():
        raise CertifyError(f"inflate script missing or symlinked: {inflate_script}")
    if not valid_sha256(inflate_script_sha) or sha256_file(inflate_script) != inflate_script_sha:
        raise CertifyError(f"inflate script SHA-256 drift: {inflate_script}")
    upstream_snapshot_sha = str(provenance.get("upstream_snapshot_sha256") or "")
    if not valid_sha256(upstream_snapshot_sha):
        raise CertifyError("provenance lacks upstream_snapshot_sha256")
    for flag, expected in (("--archive", str(archive)), ("--inflate-sh", str(inflate_script))):
        try:
            actual = sys_argv[sys_argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise CertifyError(f"reproducer argv lacks {flag}") from exc
        if actual != expected:
            raise CertifyError(f"reproducer argv {flag} drift: expected={expected}:actual={actual}")
    recorded_dir = receipt.get("inflated_dir")
    if not isinstance(recorded_dir, str):
        payload = receipt.get("payload")
        recorded_dir = payload.get("inflated_dir") if isinstance(payload, dict) else None
    aliases = [str(path), str(path.parent)]
    if isinstance(recorded_dir, str):
        aliases.extend([str(Path(recorded_dir) / path.name), recorded_dir])
    return {
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "archive_sha256_verified_current": True,
        "reproducer_argv": [str(provenance.get("effective_inflate_python") or "python"), *sys_argv],
        "inflate_script": str(inflate_script),
        "inflate_script_sha256": inflate_script_sha,
        "inflate_script_sha256_verified_current": True,
        "upstream_snapshot_sha256": upstream_snapshot_sha,
        **runtime_certificate,
        "decode_receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
            "raw_sha256_field": receipt_field,
            "raw_sha256": receipt_raw_sha,
            "equals_file_sha256": True,
        },
        "provenance_receipt": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
        },
        "eval_receipt": {"path": str(eval_path), "sha256": sha256_file(eval_path)},
        "reference_aliases": list(dict.fromkeys(aliases)),
    }


def repo_reference_hits(
    aliases_by_path: dict[str, list[str]],
    repo_root: Path,
    *,
    exclude_current: bool = True,
    observation_files: list[str] | None = None,
) -> dict[str, list[str]]:
    all_aliases = list(dict.fromkeys(alias for aliases in aliases_by_path.values() for alias in aliases))
    command = ["rg", "-n", "-F", "--no-messages", "--hidden"]
    for observed in observation_files or []:
        command.extend(["--glob", "!" + observed])
    if exclude_current:
        command.extend(
            [
                "--glob",
                "!ddm_vr3_reclaim_20260908.jsonl",
                "--glob",
                "!ddm_vr3_both_ssds_full_certify_or_block_reclaim_20260908.md",
                "--glob",
                "!**/ddm_vr3_census_20260908/**",
                "--glob",
                "!**/ddm_vr3_census.done*",
                "--glob",
                "!ddm_vr3_both_ssds_full_certify_or_block_reclaim.log",
                "--glob",
                "!ddm_vr5_generalized_certified_raw_reclaim_apply.log",
                # RECLAIM-CUSTODY ARTIFACTS ARE NOT CONSUMERS (MAIN 2026-09-10, the vr5
                # apply that refused all 17 rows): a reclaim arm's ledgers, journals,
                # inventories, memos, charters, final messages and serializer receipts
                # (format-patches copy the ledger verbatim) exist IN ORDER TO name the
                # path. A hit inside them certifies the row; it does not consume it.
                # Scoped to the ddm_vr<N> family so a real consumer elsewhere still hits.
                "--glob",
                "!**/ddm_vr[0-9]*/**",
                "--glob",
                "!**/ddm_vr[0-9]*_*.jsonl",
                "--glob",
                "!**/ddm_vr[0-9]*_*.md",
                "--glob",
                "!**/ddm_vr[0-9]*.log",
            ]
        )
    for alias in all_aliases:
        command.extend(["-e", alias])
    scopes = [scope for scope in REFERENCE_SCOPES if (repo_root / scope).exists()]
    # Explicit files bypass ignore rules: the live pointer may itself be gitignored.
    for name in (
        "canonical_frontier_pointer.json",
        "main_hot_state.md",
        "lane_registry.json",
        "active_lane_dispatch_claims.md",
    ):
        relative = ".omx/state/" + name
        if (repo_root / relative).is_file():
            scopes.append(relative)
    live_roots = {LIVE_POINTER_REFERENCE_ROOT}
    for root in (VERTIGO_ROOT, AP_ROOT):
        if root.is_dir():
            live_roots.update(p for p in root.iterdir() if p.name.startswith(LIVE_PREFIXES) and p.is_dir())
    live_scopes = [str(root) for root in sorted(live_roots) if root.exists()]
    commands = [command + scopes] if scopes else []
    if live_scopes:
        # Inspect live-tree control/receipt text without reading multi-GiB raw payloads.
        commands.append(
            [*command, "--no-ignore", "--glob", "*.{py,sh,json,jsonl,md,txt,yaml,yml,toml,log}", *live_scopes]
        )
    if not commands or not all_aliases:
        raise CertifyError("reference scan has no scopes or aliases")
    lines = []
    for scan in commands:
        try:
            completed = subprocess.run(scan, cwd=repo_root, text=True, capture_output=True, check=False, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CertifyError(f"reference scan unavailable: {exc}") from exc
        if completed.returncode not in (0, 1):
            raise CertifyError(f"reference scan failed rc={completed.returncode}: {completed.stderr}")
        lines.extend(completed.stdout.splitlines())
    return {
        path: sorted({line for line in lines if any(alias in line for alias in aliases)})
        for path, aliases in aliases_by_path.items()
    }


def lsof_plus_d(directory: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["lsof", "+D", str(directory)], text=True, capture_output=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CertifyError(f"lsof unavailable: {exc}") from exc
    if completed.returncode not in (0, 1) or completed.stderr.strip():
        raise CertifyError(f"lsof +D failed rc={completed.returncode}: {directory}: {completed.stderr.strip()}")
    output = completed.stdout.splitlines()
    open_rows = output[1:] if output and output[0].startswith("COMMAND") else output
    return {
        "command": ["lsof", "+D", str(directory)],
        "returncode": completed.returncode,
        "open_descriptor_rows": open_rows,
        "no_open_descriptors": not open_rows,
        "checked_at_utc": utc_now(),
    }


def process_gate(family: str) -> dict[str, Any]:
    """Require visible init/self and a successful owner search, never empty blind lsof."""
    if not re.fullmatch(r"ddm_[a-z0-9]+", family):
        raise CertifyError(f"invalid process owner: {family}")
    commands = [["ps", "-axo", "pid=,ppid=,command="], ["pgrep", "-fl", family + r"([^a-z0-9]|$)"]]
    receipts = []
    for command in commands:
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CertifyError(f"PROCESS_VISIBILITY_UNAVAILABLE:{exc}") from exc
        if result.stderr.strip() or result.returncode not in ((0,) if command[0] == "ps" else (0, 1)):
            raise CertifyError(f"PROCESS_VISIBILITY_UNAVAILABLE:{command[0]}:{result.stderr}")
        receipts.append({"command": command, "returncode": result.returncode})
        if command[0] == "ps":
            lines = [line.split(None, 2) for line in result.stdout.splitlines()]
            if any(len(line) != 3 or not line[0].isdigit() for line in lines):
                raise CertifyError("PROCESS_VISIBILITY_UNAVAILABLE:malformed ps")
            pids = {int(line[0]) for line in lines}
            if not {1, os.getpid()}.issubset(pids):
                raise CertifyError("PROCESS_VISIBILITY_UNAVAILABLE:init or self invisible")
            hits = [
                line for line in lines if int(line[0]) != os.getpid() and re.search(family + r"([^a-z0-9]|$)", line[2])
            ]
            if hits:
                raise CertifyError(f"LIVE_OWNER_PROCESS:{family}:{hits}")
        elif result.returncode == 0 or result.stdout.strip():
            raise CertifyError(f"LIVE_OWNER_PROCESS:{family}:{result.stdout}")
    return {"visible": True, "owner": family, "commands": receipts, "checked_at_utc": utc_now()}


def owner_family(row: dict[str, Any]) -> str:
    owner = str(row.get("owner", "")).split(" / ")[-1]
    if not re.fullmatch(r"ddm_[a-z0-9]+", owner):
        raise CertifyError("missing or invalid owning arm")
    if not any(part == owner or part.startswith(owner + "_") for part in Path(row["path"]).parts):
        raise CertifyError("owner/path mismatch")
    return owner


def pinned_file(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise CertifyError(f"pinned file missing or symlinked: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def verify_pin(pin: dict[str, str]) -> Path:
    path = Path(pin["path"])
    if pinned_file(path) != pin:
        raise CertifyError(f"PIN_SHA256_DRIFT:{path}")
    return path


def unique_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {row["path"]: row for row in rows}
    if not rows or len(result) != len(rows):
        raise CertifyError(f"empty or duplicate path ledger: {path}")
    return result


def certify_retained(source: dict[str, Any], rehash: dict[str, Any], closure: dict[str, Any]) -> dict[str, Any]:
    """Join VR4, MAIN's current hash, the terminal memo, and the current reproducer."""
    if source.get("certificate_status") != RETAINED_STATUS or not source.get("reproducer"):
        raise CertifyError("MISSING_RETAINED_REPRODUCER")
    path = Path(source["path"])
    forbidden = _forbidden_target_reason(path)
    if forbidden:
        raise CertifyError(forbidden)
    blockers = _stat_identity_blockers(path, source)
    if blockers:
        raise CertifyError(";".join(blockers))
    raw_sha = rehash.get("sha256")
    if (
        rehash.get("present") is not True
        or rehash.get("path") != str(path)
        or rehash.get("bytes") != source["bytes"]
        or not valid_sha256(raw_sha)
    ):
        raise CertifyError("CURRENT_REHASH_MISSING_OR_INVALID")
    if source.get("historical_sha256") and source["historical_sha256"] != raw_sha:
        raise CertifyError("HISTORICAL_RAW_SHA256_DRIFT")
    family = owner_family(source)
    if closure.get("family") != family or closure.get("disposition") != "CLOSED_ADVISORY_INSTANCE":
        raise CertifyError("OWNING_ARM_CLOSURE_MISSING")
    memo = verify_pin(closure["memo"])
    quote = closure.get("verdict_quote")
    if not isinstance(quote, str) or not quote.strip() or quote not in memo.read_text():
        raise CertifyError("CLOSURE_VERDICT_QUOTE_MISSING")
    current = certify_selected(path, str(raw_sha))
    if current != source["reproducer"]:
        raise CertifyError("RETAINED_REPRODUCER_DRIFT")
    return current


def combined_df(roots: list[Path]) -> dict[str, Any]:
    drives = [df_row(root) for root in roots]
    # APFS volumes may share a container; these charter drives must be distinct devices.
    if len({root.stat().st_dev for root in roots}) != len(roots):
        raise CertifyError("storage roots share device; combined capacity would double count")
    return {"available_bytes": sum(d["available_bytes"] for d in drives), "drives": drives, "sampled_at_utc": utc_now()}


def retained_revalidation(row: dict[str, Any]) -> dict[str, Any]:
    admission = row["retained_admission"]
    source = unique_rows(verify_pin(admission["source_ledger"]))[row["path"]]
    rehash = unique_rows(verify_pin(admission["rehash_ledger"]))[row["path"]]
    if row["family"] != owner_family(source) or row["sha256"] != rehash["sha256"]:
        raise CertifyError("ADMISSION_IDENTITY_DRIFT")
    current = certify_retained(source, rehash, admission["closure"])
    if {k: v for k, v in current.items() if k != "reference_aliases"} != row["reproducer"]:
        raise CertifyError("PLANNED_REPRODUCER_DRIFT")
    return current


def observation_exclusions(pins: list[dict[str, str]], repo_root: Path) -> list[str]:
    result = []
    for pin in pins:
        path = verify_pin(pin)
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        if any(c in relative for c in "*?[]!"):
            raise CertifyError("observation exclusion must be an exact file")
        result.append(relative)
    return result


def plan_retained(args: argparse.Namespace) -> int:
    sources = unique_rows(args.source_ledger)
    selected = [row for row in sources.values() if row.get("certificate_status") == RETAINED_STATUS]
    hashes = unique_rows(args.rehash_ledger)
    if set(hashes) != {row["path"] for row in selected}:
        raise CertifyError(f"MAIN_REHASH_INCOMPLETE:expected={len(selected)} actual={len(hashes)}")
    policy = load_json(args.closures)
    exclusions = observation_exclusions(policy["observation_files"], args.repo_root)
    output_relative = args.output_ledger.resolve().relative_to(args.repo_root.resolve()).as_posix()
    exclusions.append(output_relative)
    roots = sorted({storage_root(Path(row["path"])) for row in selected})
    before = combined_df(roots)
    rows = []
    aliases = {}
    for rank, source in enumerate(selected, 1):
        row = dict(source)
        path = source["path"]
        family = owner_family(source)
        closure = policy["closures"].get(family, {})
        blockers = []
        current = None
        try:
            current = certify_retained(source, hashes[path], closure)
        except (OSError, ValueError, KeyError, CertifyError) as exc:
            blockers.append(f"CERTIFICATE_REFUSED:{exc}")
        aliases[path] = source["reproducer"]["reference_aliases"]
        row.update(
            schema=SCHEMA,
            arm="ddm_vr5",
            family=family,
            inventory_rank=rank,
            sha256=hashes[path].get("sha256"),
            hash_status="HASHED_STABLE" if current else "BLOCKED",
            hash_completed_utc=None,
            hash_timestamp_note="MAIN receipt contains no timestamp",
            blockers=blockers,
            reproducer=None if current is None else {k: v for k, v in current.items() if k != "reference_aliases"},
            df_before=before,
            df_after=None,
            applied_at_utc=None,
            lsof_plan=None,
            score_claim=False,
            freed_bytes=0,
            certificate_complete=bool(current),
            certificate_status="RETAINED_CHAIN_CURRENT" if current else "BLOCKED",
            retained_admission={
                "source_ledger": pinned_file(args.source_ledger),
                "rehash_ledger": pinned_file(args.rehash_ledger),
                "closure": closure,
            },
            observation_files=policy["observation_files"],
            process_plan=policy.get("main_process_receipt", {"status": "LIVE_CHECK_REQUIRED_AT_APPLY"}),
            consumer_store=str(args.output_ledger),
            fire_trigger="MAIN harvest; outside sandbox with live process/reference/certificate gates",
        )
        rows.append(row)
    hits = repo_reference_hits(aliases, args.repo_root, observation_files=exclusions)
    for row in rows:
        row["reference_scan"] = {
            "hits": hits[row["path"]],
            "aliases": aliases[row["path"]],
            "checked_at_utc": utc_now(),
            "observation_exclusions": exclusions,
        }
        if hits[row["path"]]:
            row["blockers"].append("REPOSITORY_REFERENCE_HIT")
        row["planned_verdict"] = "DELETABLE" if not row["blockers"] else "BLOCKED:" + ";".join(row["blockers"])
        row["verdict"] = row["planned_verdict"]
        row["certificate_complete"] = not row["blockers"]
    atomic_jsonl(args.output_ledger, rows)
    admitted = [row for row in rows if row["planned_verdict"] == "DELETABLE"]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "deletable_rows": len(admitted),
                "deletable_bytes": sum(row["bytes"] for row in admitted),
                "ledger_sha256": sha256_file(args.output_ledger),
                "score_claim": False,
            }
        )
    )
    return 0


def _hash_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("path") or not row.get("status"):
            raise CertifyError("hash ledger row lacks path or status")
        rows[str(row["path"])] = row
    return rows


def plan(args: argparse.Namespace) -> int:
    inventory = load_json(args.inventory)
    candidates = inventory.get("candidates")
    if not isinstance(candidates, list) or inventory.get("scan_errors"):
        raise CertifyError("inventory missing candidates or carries scan errors")
    hashes = _hash_rows(args.hash_ledger)
    inventory_paths = [str(row["path"]) for row in candidates]
    if set(hashes) != set(inventory_paths) or len(hashes) != len(inventory_paths):
        raise CertifyError(f"hash census incomplete: inventory={len(inventory_paths)} stable_hashes={len(hashes)}")
    before = df_row(VERTIGO_ROOT)
    rows: list[dict[str, Any]] = []
    aliases_by_path: dict[str, list[str]] = {}
    for rank, item in enumerate(candidates, start=1):
        path = Path(str(item["path"]))
        hash_row = hashes[str(path)]
        for field in ("bytes", "mtime_ns", "device", "inode"):
            if int(item[field]) != int(hash_row[field]):
                raise CertifyError(f"inventory/hash mismatch {field}: {path}")
        selected = _selected_family(path)
        forbidden = _forbidden_target_reason(path)
        reproducer: dict[str, Any] | None = None
        blockers: list[str] = []
        hash_status = str(hash_row.get("status"))
        raw_sha = hash_row.get("sha256")
        hash_is_stable = hash_status == "HASHED_STABLE" and valid_sha256(raw_sha)
        if not hash_is_stable:
            blockers.append(f"HASH_NOT_STABLE:{hash_status}")
        if forbidden:
            blockers.append(forbidden)
        if selected is None:
            blockers.append("FULL_CERTIFICATE_NOT_ESTABLISHED_BEFORE_TARGET_BATCH")
            aliases = [str(path), str(path.parent)]
        elif hash_is_stable:
            try:
                reproducer = certify_selected(path, str(raw_sha))
                aliases = list(reproducer.pop("reference_aliases"))
            except (OSError, ValueError, KeyError, json.JSONDecodeError, CertifyError) as exc:
                blockers.append(f"CERTIFICATE_REFUSED:{type(exc).__name__}:{exc}")
                aliases = [str(path), str(path.parent)]
        else:
            aliases = [str(path), str(path.parent)]
        aliases_by_path[str(path)] = aliases
        rows.append(
            {
                "schema": SCHEMA,
                "inventory_rank": rank,
                "path": str(path),
                "bytes": int(hash_row["bytes"]),
                "sha256": raw_sha,
                "mtime_ns": int(hash_row["mtime_ns"]),
                "device": int(hash_row["device"]),
                "inode": int(hash_row["inode"]),
                "match_reasons": item["match_reasons"],
                "hash_status": hash_status,
                "hash_completed_utc": hash_row["hash_completed_utc"],
                "family": None if selected is None else selected[0],
                "candidate_tag": None if selected is None else selected[1],
                "reproducer": reproducer,
                "reference_scan": None,
                "lsof_plan": None,
                "live_pointer_protected": PROTECTED_COMPONENT in path.parts,
                "compression_verdict": (
                    "BLOCKED:SR3_CUSTODY_NAMESPACE_OR_NO_PER_TREE_KEEP_UNCOMPRESSED_PROTECTION_LIFT"
                ),
                "blockers": blockers,
                "planned_verdict": None,
                "verdict": None,
                "df_before": before,
                "df_after": None,
                "applied_at_utc": None,
            }
        )
    reference_hits = repo_reference_hits(aliases_by_path, args.repo_root)
    for row in rows:
        path = str(row["path"])
        hits = reference_hits[path]
        row["reference_scan"] = {
            "scopes": [scope for scope in REFERENCE_SCOPES if (args.repo_root / scope).exists()]
            + ([str(LIVE_POINTER_REFERENCE_ROOT)] if LIVE_POINTER_REFERENCE_ROOT.exists() else []),
            "aliases": aliases_by_path[path],
            "hits": hits,
            "no_repository_reference_hits": not hits,
            "checked_at_utc": utc_now(),
        }
        try:
            lsof = lsof_plus_d(Path(path).parent)
            row["lsof_plan"] = lsof
            if lsof["open_descriptor_rows"]:
                row["blockers"].append("LIVE_OPEN_DESCRIPTOR")
        except CertifyError as exc:
            row["blockers"].append(f"LSOF_SCAN_FAILED:{exc}")
        if hits:
            row["blockers"].append("REPOSITORY_REFERENCE_HIT")
        row["blockers"] = list(dict.fromkeys(row["blockers"]))
        row["planned_verdict"] = "DELETABLE" if not row["blockers"] else "BLOCKED:" + ";".join(row["blockers"])
        row["verdict"] = row["planned_verdict"]
    atomic_jsonl(args.output_ledger, rows)
    deletable = [row for row in rows if row["planned_verdict"] == "DELETABLE"]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "deletable_rows": len(deletable),
                "deletable_bytes": sum(row["bytes"] for row in deletable),
                "blocked_rows": len(rows) - len(deletable),
                "output_ledger": str(args.output_ledger),
                "output_ledger_sha256": sha256_file(args.output_ledger),
                "df_before": before,
            },
            sort_keys=True,
        )
    )
    return 0


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("schema") != SCHEMA for row in rows):
        raise CertifyError("invalid or empty reclaim ledger")
    return rows


def _stat_identity_blockers(path: Path, row: dict[str, Any]) -> list[str]:
    if not path.is_file() or path.is_symlink():
        return ["RAW_MISSING_OR_NOT_REGULAR"]
    stat = path.stat(follow_symlinks=False)
    blockers = []
    for field, actual in (
        ("bytes", stat.st_size),
        ("mtime_ns", stat.st_mtime_ns),
        ("device", stat.st_dev),
        ("inode", stat.st_ino),
    ):
        if int(row[field]) != int(actual):
            blockers.append(f"RAW_{field.upper()}_DRIFT")
    return blockers


def apply(args: argparse.Namespace) -> int:
    rows = _load_ledger(args.ledger)
    generalized = any("retained_admission" in row for row in rows)
    if generalized and not all("retained_admission" in row for row in rows):
        raise CertifyError("mixed legacy/generalized ledger refused")
    admitted_bytes = sum(int(row["bytes"]) for row in rows if row.get("planned_verdict") == "DELETABLE")
    if generalized and (args.target_bytes != admitted_bytes or admitted_bytes <= 0):
        raise CertifyError("generalized target must equal the full admitted byte set")
    if not generalized and int(args.target_bytes) < TARGET_BYTES:
        raise CertifyError(f"refusing to lower the charter target: got={args.target_bytes}:minimum={TARGET_BYTES}")
    if any(
        not valid_sha256(row.get("sha256")) or row.get("hash_status") != "HASHED_STABLE"
        for row in rows
        if row.get("planned_verdict") == "DELETABLE"
    ):
        raise CertifyError("every deletable ledger row must carry a stable full SHA-256 before apply")
    preapply_sha = sha256_file(args.ledger)
    if preapply_sha != args.expected_ledger_sha256:
        raise CertifyError(f"ledger identity mismatch: expected={args.expected_ledger_sha256}:actual={preapply_sha}")
    if len({row["path"] for row in rows}) != len(rows):
        raise CertifyError("duplicate apply paths")
    roots = sorted({storage_root(Path(row["path"])) for row in rows}) if generalized else [VERTIGO_ROOT]
    for row in rows:
        if row.get("verdict") == "DELETABLE":
            family = row["family"] if generalized else re.search(r"ddm_[a-z0-9]+", row["path"])[0]
            process_gate(family)
    baseline = min(int(row.get("df_before", {}).get("available_bytes")) for row in rows if row.get("df_before"))
    append_fsynced(
        args.journal,
        {
            "schema": JOURNAL_SCHEMA,
            "phase": "APPLY_START",
            "preapply_ledger_path": str(args.ledger),
            "preapply_ledger_sha256": preapply_sha,
            "target_bytes": int(args.target_bytes),
            "df": combined_df(roots),
            "written_at_utc": utc_now(),
        },
    )
    deleted = 0
    deleted_logical_bytes = sum(int(row["bytes"]) for row in rows if row.get("verdict") == "DELETED")
    for row in sorted(rows, key=lambda item: (-int(item["bytes"]), int(item["inventory_rank"]))):
        if row.get("planned_verdict") != "DELETABLE" or row.get("verdict") != "DELETABLE":
            continue
        current_df = combined_df(roots)
        net_freed = int(current_df["available_bytes"]) - baseline
        if net_freed >= int(args.target_bytes) and deleted_logical_bytes >= int(args.target_bytes):
            row["verdict"] = "BLOCKED:TARGET_MET_BEFORE_ROW"
            row["df_after"] = current_df
            continue
        path = Path(str(row["path"]))
        blockers = _stat_identity_blockers(path, row)
        if not generalized and _selected_family(path) is None:
            blockers.append("TARGET_NOT_IN_EXACT_APPLY_ALLOWLIST")
        forbidden = _forbidden_target_reason(path)
        if forbidden:
            blockers.append(forbidden)
        try:
            refreshed = retained_revalidation(row) if generalized else certify_selected(path, str(row["sha256"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError, CertifyError) as exc:
            blockers.append(f"CERTIFICATE_REVALIDATION_REFUSED:{type(exc).__name__}:{exc}")
            refreshed = None
        aliases = [str(path), str(path.parent)] if refreshed is None else list(refreshed.pop("reference_aliases"))
        exclusions = observation_exclusions(row.get("observation_files", []), args.repo_root)
        if generalized:
            exclusions.extend(
                [
                    args.ledger.resolve().relative_to(args.repo_root.resolve()).as_posix(),
                    args.journal.resolve().relative_to(args.repo_root.resolve()).as_posix(),
                ]
            )
        hits = repo_reference_hits({str(path): aliases}, args.repo_root, observation_files=exclusions)[str(path)]
        if hits:
            blockers.append("REPOSITORY_REFERENCE_HIT_AT_APPLY")
        raw_sha_verified_current = False
        if not blockers:
            if sha256_file(path) != row["sha256"]:
                blockers.append("RAW_SHA256_DRIFT_AT_APPLY")
            else:
                raw_sha_verified_current = True
        process = None
        try:
            family = row["family"] if generalized else re.search(r"ddm_[a-z0-9]+", row["path"])[0]
            process = process_gate(family)
            lsof = lsof_plus_d(path.parent)
            if lsof["open_descriptor_rows"]:
                blockers.append("LIVE_OPEN_DESCRIPTOR_AT_APPLY")
        except CertifyError as exc:
            lsof = None
            blockers.append(f"LSOF_SCAN_FAILED_AT_APPLY:{exc}")
        blockers.extend(_stat_identity_blockers(path, row))
        blockers = list(dict.fromkeys(blockers))
        append_fsynced(
            args.journal,
            {
                "schema": JOURNAL_SCHEMA,
                "phase": "PRE_DELETE",
                "path": str(path),
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "refreshed_reproducer": refreshed,
                "reference_hits": hits,
                "lsof": lsof,
                "process_gate": process,
                "raw_sha256_verified_current": raw_sha_verified_current,
                "blockers": blockers,
                "df": current_df,
                "written_at_utc": utc_now(),
            },
        )
        if blockers:
            row["verdict"] = "BLOCKED:" + ";".join(blockers)
            row["df_after"] = current_df
            atomic_jsonl(args.ledger, rows)
            continue
        path.unlink()
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        if path.exists() or path.is_symlink():
            raise CertifyError(f"raw still exists after unlink: {path}")
        after = combined_df(roots)
        row["verdict"] = "DELETED"
        row["df_after"] = after
        row["applied_at_utc"] = utc_now()
        row["apply_certificate"] = {
            "preapply_ledger_sha256": preapply_sha,
            "journal_path": str(args.journal),
            "lsof_no_open_descriptors": True,
            "repository_reference_hits": [],
            "reproducer_revalidated": True,
            "source_absent_after": True,
        }
        append_fsynced(
            args.journal,
            {
                "schema": JOURNAL_SCHEMA,
                "phase": "DELETED",
                "path": str(path),
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "df_before": current_df,
                "df_after": after,
                "written_at_utc": row["applied_at_utc"],
            },
        )
        atomic_jsonl(args.ledger, rows)
        deleted += 1
        deleted_logical_bytes += int(row["bytes"])
    final_df = combined_df(roots)
    for row in rows:
        if row.get("planned_verdict") == "DELETABLE" and row.get("verdict") == "DELETABLE":
            row["verdict"] = "BLOCKED:CERTIFIED_SET_EXHAUSTED_OR_TARGET_CHECK_ENDED"
            row["df_after"] = final_df
        if row.get("df_after") is None:
            row["df_after"] = final_df
    atomic_jsonl(args.ledger, rows)
    logical_bytes_deleted_total = sum(int(row["bytes"]) for row in rows if row.get("verdict") == "DELETED")
    measured_target_met = int(final_df["available_bytes"]) - baseline >= int(args.target_bytes)
    attributable_target_met = logical_bytes_deleted_total >= int(args.target_bytes)
    summary = {
        "schema": JOURNAL_SCHEMA,
        "phase": "APPLY_COMPLETE",
        "deleted_this_invocation": deleted,
        "deleted_rows_total": sum(row.get("verdict") == "DELETED" for row in rows),
        "logical_bytes_deleted_total": logical_bytes_deleted_total,
        "baseline_available_bytes": baseline,
        "final_df": final_df,
        "net_available_bytes_gained": int(final_df["available_bytes"]) - baseline,
        "target_bytes": int(args.target_bytes),
        "measured_df_target_met": measured_target_met,
        "attributable_logical_target_met": attributable_target_met,
        "target_met": measured_target_met and attributable_target_met,
        "final_ledger_sha256": sha256_file(args.ledger),
        "written_at_utc": utc_now(),
    }
    append_fsynced(args.journal, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["target_met"] else 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--inventory", type=Path, required=True)
    plan_parser.add_argument("--hash-ledger", type=Path, required=True)
    plan_parser.add_argument("--output-ledger", type=Path, required=True)
    retained_parser = subparsers.add_parser("plan-retained")
    retained_parser.add_argument("--source-ledger", type=Path, required=True)
    retained_parser.add_argument("--rehash-ledger", type=Path, required=True)
    retained_parser.add_argument("--closures", type=Path, required=True)
    retained_parser.add_argument("--output-ledger", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--ledger", type=Path, required=True)
    apply_parser.add_argument("--expected-ledger-sha256", required=True)
    apply_parser.add_argument("--journal", type=Path, required=True)
    apply_parser.add_argument("--target-bytes", type=int, default=TARGET_BYTES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan-retained":
            return plan_retained(args)
        return plan(args) if args.command == "plan" else apply(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, CertifyError) as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
