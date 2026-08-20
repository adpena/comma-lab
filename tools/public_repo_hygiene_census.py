#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inventory tracked public-repository local paths and their SHA pin consumers.

The output is evidence, not an automatic rewrite list. Every matching tracked file is
classified as executable/source (a), research/documentation (b), immutable/pinned (c),
or ignored/live-state (d). Class (d) is represented by explicit ``git check-ignore``
facts because ignored files are intentionally absent from the tracked-file denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

USER_LITERAL = b"/Users/adpena"  # ABSOLUTE_PATH_OK:census-detector-literal
VOLUME_LITERAL = b"/Volumes/"  # ABSOLUTE_PATH_OK:census-detector-literal
TOKENS = {
    "user_home": USER_LITERAL,
    "volume_root": VOLUME_LITERAL,
}
CLASS_A_PREFIXES = ("src/", "tools/", "experiments/", "scripts/")
PROTECTED_PREFIXES = (
    "submissions/robust_current/jg5_sub015_runtime/",
    "generations/gen5",
    "upstream/",
)
IGNORED_LIVE_PROBES = ("fleet.local.toml", ".env", ".env.local")
PLACEHOLDER_TOKENS = (b"$PACT_TIER1", b"$PACT_TIER2", b"$HOME", b"~/")
SHA256_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", re.IGNORECASE)


class CensusError(RuntimeError):
    """The Git tree or requested artifact could not be read deterministically."""


def _run(repo: Path, *argv: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        cwd=repo,
        check=check,
        capture_output=True,
    )


def _tree(repo: Path, ref: str) -> dict[str, str]:
    result = _run(repo, "git", "ls-tree", "-r", "-z", ref)
    rows: dict[str, str] = {}
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        meta, encoded_path = raw.split(b"\t", 1)
        _mode, kind, object_id = meta.decode("ascii").split()
        if kind == "blob":
            rows[encoded_path.decode("utf-8", errors="surrogateescape")] = object_id
    return rows


def _grep_paths(repo: Path, ref: str, literal: bytes) -> set[str]:
    result = _run(
        repo,
        "git",
        "grep",
        "-I",
        "-l",
        "-F",
        literal.decode("ascii"),
        ref,
        "--",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise CensusError(result.stderr.decode("utf-8", errors="replace"))
    prefix = f"{ref}:"
    return {
        line[len(prefix) :] if line.startswith(prefix) else line
        for line in result.stdout.decode("utf-8", errors="surrogateescape").splitlines()
        if line
    }


def _read_batch(
    repo: Path,
    object_ids: Iterable[str],
) -> dict[str, bytes]:
    ordered = tuple(dict.fromkeys(object_ids))
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    result: dict[str, bytes] = {}
    for object_id in ordered:
        process.stdin.write(object_id.encode("ascii") + b"\n")
        process.stdin.flush()
        header = process.stdout.readline().decode("ascii").strip().split()
        if len(header) != 3 or header[1] != "blob":
            process.kill()
            raise CensusError(f"could not read Git blob {object_id}: {header!r}")
        size = int(header[2])
        payload = process.stdout.read(size)
        if process.stdout.read(1) != b"\n":
            process.kill()
            raise CensusError(f"malformed git cat-file response for {object_id}")
        result[object_id] = payload
    process.stdin.close()
    returncode = process.wait()
    if returncode != 0:
        stderr = b"" if process.stderr is None else process.stderr.read()
        raise CensusError(stderr.decode("utf-8", errors="replace"))
    return result


def _pin_index(
    *,
    repo: Path,
    ref: str,
    target_digests: set[str],
) -> dict[str, list[str]]:
    """Find every tracked text reference to a hit file's exact SHA-256."""

    consumers: defaultdict[str, list[str]] = defaultdict(list)
    result = _run(
        repo,
        "git",
        "grep",
        "-I",
        "-n",
        "-E",
        "[0-9A-Fa-f]{64}",
        ref,
        "--",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise CensusError(result.stderr.decode("utf-8", errors="replace"))
    prefix = f"{ref}:"
    for raw_line in result.stdout.decode("utf-8", errors="ignore").splitlines():
        line = raw_line[len(prefix) :] if raw_line.startswith(prefix) else raw_line
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path, _line_no, text = parts
        for digest in set(SHA256_PATTERN.findall(text)) & target_digests:
            consumers[digest].append(path)
    return {digest: sorted(set(paths)) for digest, paths in consumers.items()}


def classify_path(
    path: str,
    *,
    content_sha256: str,
    pin_consumers: Mapping[str, Sequence[str]],
) -> tuple[str, str, list[str]]:
    """Classify one hit with immutable/pinned precedence over directory class."""

    if path.startswith(PROTECTED_PREFIXES):
        return "c", "protected_no_edit_surface", ["charter:ddm_sw1"]
    consumers = list(pin_consumers.get(content_sha256, ()))
    if consumers:
        return "c", "content_sha256_consumed_by_public_seal_or_receipt", consumers
    if path.startswith(CLASS_A_PREFIXES):
        return "a", "source_tool_experiment_or_script", []
    return "b", "research_documentation_or_other_tracked_evidence", []


def _top_directory(path: str) -> str:
    parts = Path(path).parts
    if path.startswith(".omx/research/"):
        return ".omx/research"
    return parts[0] if parts else "."


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact_row(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _ignore_facts(repo: Path) -> list[dict[str, object]]:
    facts = []
    for probe in IGNORED_LIVE_PROBES:
        result = _run(repo, "git", "check-ignore", "-v", "--", probe, check=False)
        facts.append(
            {
                "path": probe,
                "ignored": result.returncode == 0,
                "rule": result.stdout.decode("utf-8", errors="replace").strip(),
                "exists": (repo / probe).exists(),
            }
        )
    return facts


def build_census(repo: Path, ref: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    tree = _tree(repo, ref)
    hits_by_token = {name: _grep_paths(repo, ref, token) for name, token in TOKENS.items()}
    hit_paths = sorted(set().union(*hits_by_token.values()))
    needed_ids = [tree[path] for path in hit_paths]
    blobs = _read_batch(repo, needed_ids)
    digests = {
        path: hashlib.sha256(blobs[tree[path]]).hexdigest() for path in hit_paths
    }
    pin_consumers = _pin_index(
        repo=repo,
        ref=ref,
        target_digests=set(digests.values()),
    )
    rows: list[dict[str, object]] = []
    for path in hit_paths:
        payload = blobs[tree[path]]
        digest = digests[path]
        category, reason, consumers = classify_path(
            path,
            content_sha256=digest,
            pin_consumers=pin_consumers,
        )
        matches = {
            name: {
                "occurrences": payload.count(token),
                "matching_lines": sum(token in line for line in payload.splitlines()),
            }
            for name, token in TOKENS.items()
            if token in payload
        }
        rows.append(
            {
                "path": path,
                "class": category,
                "classification_reason": reason,
                "content_sha256": digest,
                "git_blob": tree[path],
                "matches": matches,
                "pinning_consumers": consumers,
            }
        )
    class_counts = Counter(str(row["class"]) for row in rows)
    top_directory_counts = Counter(_top_directory(str(row["path"])) for row in rows)
    summary: dict[str, object] = {
        "schema": "public_repo_hygiene_census.v1",
        "ref": ref,
        "commit": _run(repo, "git", "rev-parse", ref).stdout.decode("ascii").strip(),
        "tracked_file_denominator": len(tree),
        "matching_file_denominator": len(rows),
        "matching_file_counts_by_token": {
            key: len(value) for key, value in hits_by_token.items()
        },
        "matching_occurrence_counts_by_token": {
            key: sum(int(row["matches"].get(key, {}).get("occurrences", 0)) for row in rows)  # type: ignore[union-attr]
            for key in TOKENS
        },
        "class_counts": dict(sorted(class_counts.items())),
        "top_directory_counts": dict(top_directory_counts.most_common()),
        "class_c_rows": [
            {
                "path": row["path"],
                "content_sha256": row["content_sha256"],
                "pinning_consumers": row["pinning_consumers"],
            }
            for row in rows
            if row["class"] == "c"
        ],
        "ignored_live_state_facts": _ignore_facts(repo),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return summary, rows


def write_census(repo: Path, ref: str, out_root: Path) -> dict[str, object]:
    summary, rows = build_census(repo, ref)
    rows_path = out_root / "public_repo_hygiene_census_rows.jsonl"
    summary_path = out_root / "public_repo_hygiene_census_summary.json"
    _write_atomic(
        rows_path,
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        ),
    )
    summary["rows_artifact"] = _artifact_row(rows_path)
    _write_atomic(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {
        "summary": summary,
        "summary_artifact": _artifact_row(summary_path),
    }


def write_mapping(
    repo: Path,
    base: str,
    paths: Sequence[str],
    out_path: Path,
) -> dict[str, object]:
    rows = []
    old_tokens = (USER_LITERAL, VOLUME_LITERAL)
    for path in paths:
        prior = _run(repo, "git", "show", f"{base}:{path}", check=False)
        if prior.returncode not in (0, 128):
            raise CensusError(prior.stderr.decode("utf-8", errors="replace"))
        before = prior.stdout if prior.returncode == 0 else b""
        current_path = repo / path
        after = current_path.read_bytes() if current_path.is_file() else b""
        rows.append(
            {
                "path": path,
                "before_sha256": hashlib.sha256(before).hexdigest(),
                "after_sha256": hashlib.sha256(after).hexdigest(),
                "removed_local_literal_occurrences": sum(
                    max(before.count(token) - after.count(token), 0) for token in old_tokens
                ),
                "added_local_literal_occurrences": sum(
                    max(after.count(token) - before.count(token), 0) for token in old_tokens
                ),
                "added_placeholder_occurrences": sum(
                    max(after.count(token) - before.count(token), 0)
                    for token in PLACEHOLDER_TOKENS
                ),
            }
        )
    payload: dict[str, object] = {
        "schema": "public_repo_path_mapping_receipt.v1",
        "base": base,
        "base_commit": _run(repo, "git", "rev-parse", base).stdout.decode("ascii").strip(),
        "rows": rows,
        "totals": {
            "removed_local_literal_occurrences": sum(
                int(row["removed_local_literal_occurrences"]) for row in rows
            ),
            "added_local_literal_occurrences": sum(
                int(row["added_local_literal_occurrences"]) for row in rows
            ),
            "added_placeholder_occurrences": sum(
                int(row["added_placeholder_occurrences"]) for row in rows
            ),
        },
        "score_claim": False,
    }
    _write_atomic(
        out_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {"mapping": payload, "artifact": _artifact_row(out_path)}


def _fleet_inventory_tokens(inventory_path: Path) -> list[tuple[str, str]]:
    payload = tomllib.loads(inventory_path.read_text(encoding="utf-8"))
    hosts = payload.get("hosts")
    if not isinstance(hosts, dict):
        raise CensusError("fleet inventory has no [hosts] table")
    tokens: list[tuple[str, str]] = []
    for row in hosts.values():
        if not isinstance(row, dict):
            continue
        for kind in ("host", "ip"):
            value = row.get(kind)
            if isinstance(value, str) and value.strip():
                tokens.append((kind, value.strip()))
    return list(dict.fromkeys(tokens))


def write_fleet_scan(repo: Path, inventory_path: Path, out_path: Path) -> dict[str, object]:
    """Scan tracked/nonignored worktree files for exact private fleet coordinates."""

    tokens = _fleet_inventory_tokens(inventory_path)
    rows = []
    for kind, token in tokens:
        result = _run(
            repo,
            "git",
            "grep",
            "--untracked",
            "-I",
            "-n",
            "-F",
            token,
            "--",
            check=False,
        )
        if result.returncode not in (0, 1):
            raise CensusError(result.stderr.decode("utf-8", errors="replace"))
        matches = []
        for line in result.stdout.decode("utf-8", errors="surrogateescape").splitlines():
            parts = line.split(":", 2)
            if len(parts) < 2:
                continue
            matches.append({"path": parts[0], "line": int(parts[1])})
        rows.append(
            {
                "kind": kind,
                "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                "match_count": len(matches),
                "matches": matches,
            }
        )
    payload: dict[str, object] = {
        "schema": "public_repo_fleet_coordinate_scan.v1",
        "scope": "git tracked plus untracked nonignored working tree",
        "inventory_path_ignored": _run(
            repo,
            "git",
            "check-ignore",
            "-q",
            "--",
            inventory_path.relative_to(repo).as_posix(),
            check=False,
        ).returncode
        == 0,
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "token_count": len(rows),
        "matching_token_count": sum(bool(row["match_count"]) for row in rows),
        "match_count": sum(int(row["match_count"]) for row in rows),
        "rows": rows,
        "score_claim": False,
    }
    _write_atomic(
        out_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {"fleet_scan": payload, "artifact": _artifact_row(out_path)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    census = subparsers.add_parser("census")
    census.add_argument("--ref", default="HEAD")
    census.add_argument("--out-root", type=Path, required=True)
    mapping = subparsers.add_parser("mapping")
    mapping.add_argument("--base", required=True)
    mapping.add_argument("--path", action="append", required=True)
    mapping.add_argument("--out", type=Path, required=True)
    fleet = subparsers.add_parser("fleet")
    fleet.add_argument("--inventory", type=Path, required=True)
    fleet.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = args.repo.expanduser().resolve(strict=True)
    if args.command == "census":
        result = write_census(repo, args.ref, args.out_root.expanduser().resolve(strict=False))
    elif args.command == "mapping":
        result = write_mapping(
            repo,
            args.base,
            args.path,
            args.out.expanduser().resolve(strict=False),
        )
    else:
        result = write_fleet_scan(
            repo,
            args.inventory.expanduser().resolve(strict=True),
            args.out.expanduser().resolve(strict=False),
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
