"""Pure curation rules for the public-submission reverse-engineering tree.

The reverse-engineering directory is intentionally a curated surface, not a
second source tree. This module classifies recovered/orphaned material so
operators can promote reusable code into ``tac``, keep public-runtime copies as
forensics, and externalize raw artifacts without losing signal. It intentionally
does not import ``comma_lab``; lab-facing adapters wrap this module.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from tac.repo_io import json_text, repo_relative, sha256_file

DEFAULT_ROOT = Path("reverse_engineering")
ORPHAN_ROOT_NAME = "orphan_pyc_recovery_20260505_codex"
RAW_EXTENSIONS = {
    ".7z",
    ".avi",
    ".log",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".pth",
    ".pt",
    ".raw",
    ".safetensors",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
    ".zst",
}
BLOCKING_DISPOSITIONS = {"externalize_with_manifest", "manual_review"}
RELEASE_BLOCKING_DISPOSITIONS = BLOCKING_DISPOSITIONS | {
    "canonicalize_to_docs_or_ledger",
    "compare_and_promote_or_ledger",
    "compare_and_promote_to_tac",
    "external_forensics_manifest_only",
    "preserve_until_hand_rehydration",
    "preserve_until_source_disposition",
    "promote_thin_cli_or_ledger",
    "summarize_to_research_ledger",
}
IGNORED_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
DAMAGED_DECOMPILE_MARKERS = (
    "# Source Generated with Decompyle++",
    "# WARNING: Decompyle incomplete",
    "<NODE:",
    "pass# WARNING",
    "None(",
    "= [][",
)
RECOVERY_OPERATOR_SUFFIXES = (".PREFLIGHT_DEBT", ".QUARANTINED")


@dataclasses.dataclass(frozen=True)
class ReverseEngineeringRecord:
    relpath: str
    bytes: int
    sha256: str
    category: str
    disposition: str
    target: str | None
    live_relpath: str | None
    live_status: str | None
    reason: str


@dataclasses.dataclass(frozen=True)
class ReleaseResolutionRule:
    id: str
    action: str
    public_release: str
    note: str
    ledger_path: str | None = None
    category: str | None = None
    disposition: str | None = None
    relpath_prefix: str | None = None

    def matches(self, record: ReverseEngineeringRecord) -> bool:
        if self.category is not None and self.category != record.category:
            return False
        if self.disposition is not None and self.disposition != record.disposition:
            return False
        if self.relpath_prefix is not None and not record.relpath.startswith(self.relpath_prefix):
            return False
        return any((self.category, self.disposition, self.relpath_prefix))


def load_release_resolution_rules(path: Path | None) -> list[ReleaseResolutionRule]:
    """Load release-manifest rules that close public-release blockers."""

    if path is None:
        return []
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("release manifest schema_version must be 1")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("release manifest must contain nonempty rules")
    rules: list[ReleaseResolutionRule] = []
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            raise ValueError(f"release manifest rule {index} must be an object")
        match = raw.get("match")
        if not isinstance(match, dict):
            raise ValueError(f"release manifest rule {index} missing match object")
        for required in ("id", "action", "public_release", "note"):
            if not isinstance(raw.get(required), str) or not raw[required].strip():
                raise ValueError(f"release manifest rule {index} missing {required}")
        if raw["public_release"] not in {"exclude_raw", "publish_sanitized_summary", "publish_curated_source"}:
            raise ValueError(f"release manifest rule {index} has invalid public_release")
        if not any(match.get(key) for key in ("category", "disposition", "relpath_prefix")):
            raise ValueError(f"release manifest rule {index} must match category, disposition, or relpath_prefix")
        rules.append(
            ReleaseResolutionRule(
                id=raw["id"],
                action=raw["action"],
                public_release=raw["public_release"],
                note=raw["note"],
                ledger_path=raw.get("ledger_path"),
                category=match.get("category"),
                disposition=match.get("disposition"),
                relpath_prefix=match.get("relpath_prefix"),
            )
        )
    return rules


def _strip_recovery_header(data: bytes) -> bytes:
    lines = data.splitlines(keepends=True)
    cursor = 0
    while cursor < len(lines):
        text = lines[cursor].decode("utf-8", errors="replace").strip()
        if text.startswith("# pyc-recovery:") or text.startswith("# This is the canonical main-repo content"):
            cursor += 1
            continue
        if text.startswith("# Recovery spec preserved at:") or text.startswith("# Original STUB has been replaced"):
            cursor += 1
            continue
        break
    return b"".join(lines[cursor:])


def _same_ignoring_recovery_header(a: Path, b: Path) -> bool:
    try:
        return _strip_recovery_header(a.read_bytes()) == _strip_recovery_header(b.read_bytes())
    except UnicodeDecodeError:
        return False


def _same_trace_except_source_path(live_path: Path, recovered_path: Path) -> bool:
    """Return true when a recovered ARA trace only leaks private source paths."""

    try:
        live_lines = live_path.read_text(encoding="utf-8").splitlines()
        recovered_lines = recovered_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    if len(live_lines) != len(recovered_lines):
        return False
    for live_line, recovered_line in zip(live_lines, recovered_lines, strict=True):
        try:
            live = json.loads(live_line)
            recovered = json.loads(recovered_line)
        except json.JSONDecodeError:
            return False
        live_source = live.pop("source_path", None)
        recovered_source = recovered.pop("source_path", None)
        if live != recovered:
            return False
        if not isinstance(live_source, str) or not live_source.startswith("<operator-memory>/"):
            return False
        if not isinstance(recovered_source, str) or "/.claude/projects/" not in recovered_source:
            return False
    return True


def _operator_shadow_live_path(rel_after_orphan: str, repo_root: Path) -> str | None:
    for suffix in RECOVERY_OPERATOR_SUFFIXES:
        if not rel_after_orphan.endswith(suffix):
            continue
        candidate = rel_after_orphan.removesuffix(suffix)
        if (repo_root / candidate).is_file():
            return candidate
    return None


def _repo_rel(path: Path, repo_root: Path) -> str:
    return repo_relative(path, repo_root)


def _live_status(path: Path, repo_root: Path) -> tuple[str | None, str | None]:
    rel = _repo_rel(path, repo_root)
    prefix = f"{DEFAULT_ROOT.as_posix()}/{ORPHAN_ROOT_NAME}/"
    if not rel.startswith(prefix):
        return None, None
    live_rel = rel[len(prefix) :]
    live_path = repo_root / live_rel
    if live_path.is_dir():
        return live_rel, "live_is_directory"
    if not live_path.exists():
        return live_rel, "missing_in_main"
    if not live_path.is_file():
        return live_rel, "live_not_regular_file"
    if sha256_file(live_path) == sha256_file(path):
        return live_rel, "same_as_main"
    if _same_ignoring_recovery_header(live_path, path):
        return live_rel, "same_as_main_ignoring_recovery_header"
    return live_rel, "differs_from_main"


def _classify_orphan(rel_after_orphan: str, path: Path) -> tuple[str, str, str | None, str]:
    if rel_after_orphan.startswith(".omx/auto_memory_snapshot_"):
        return (
            "orphan_auto_memory_snapshot",
            "summarize_to_research_ledger",
            ".omx/research/",
            "Auto-memory snapshots are valuable context, but raw snapshots are not canonical reverse-engineering source.",
        )
    if rel_after_orphan.startswith(".omx/state/"):
        return (
            "orphan_provider_or_dispatch_state",
            "summarize_to_research_ledger",
            ".omx/research/",
            "Provider and dispatch state should be summarized; raw state remains local custody.",
        )
    if path.name.endswith(".recovery_spec.json"):
        target = rel_after_orphan.removesuffix(".recovery_spec.json")
        return (
            "orphan_recovery_spec",
            "preserve_until_source_disposition",
            target,
            "Recovery specs preserve decompiler/provenance signal until the matching source is promoted, ledgered, or retired.",
        )
    if rel_after_orphan.startswith("src/tac/tests/"):
        return (
            "orphan_tac_test_candidate",
            "compare_and_promote_or_ledger",
            rel_after_orphan,
            "Recovered reusable tests belong on main only after comparison against the live test surface.",
        )
    if rel_after_orphan.startswith("src/tac/"):
        return (
            "orphan_tac_candidate",
            "compare_and_promote_to_tac",
            rel_after_orphan,
            "Recovered reusable codec/runtime code belongs in tac after review and focused tests.",
        )
    if rel_after_orphan.startswith("experiments/results/"):
        return (
            "orphan_public_runtime_or_result_copy",
            "external_forensics_manifest_only",
            ".omx/research/",
            "Public-runtime/result copies are reverse-engineering evidence, not canonical source.",
        )
    if rel_after_orphan.startswith("experiments/"):
        return (
            "orphan_experiment_entrypoint_candidate",
            "promote_thin_cli_or_ledger",
            rel_after_orphan,
            "Experiment entry points should be thin wrappers over tac/comma-lab implementation or ledgered as forensics.",
        )
    if rel_after_orphan.startswith("scripts/") or rel_after_orphan.startswith("tools/"):
        return (
            "orphan_operator_tool_candidate",
            "compare_and_promote_or_ledger",
            rel_after_orphan,
            "Recovered operator tools need live-path comparison and visible gate/runbook wiring before promotion.",
        )
    if rel_after_orphan.startswith("reports/") or rel_after_orphan.startswith("docs/"):
        return (
            "orphan_report_or_site_candidate",
            "canonicalize_to_docs_or_ledger",
            rel_after_orphan,
            "Report/site material should become curated docs or a dated research ledger, not hidden recovery source.",
        )
    if rel_after_orphan.startswith("submissions/"):
        return (
            "orphan_submission_runtime_candidate",
            "compare_and_promote_or_ledger",
            rel_after_orphan,
            "Recovered submission runtime must pass contest-compliance review before promotion.",
        )
    return (
        "orphan_manual_review",
        "manual_review",
        rel_after_orphan,
        "No automatic disposition matched this orphan path.",
    )


def _is_damaged_decompile(path: Path) -> bool:
    if path.suffix not in {".py", ".sh"}:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in text for marker in DAMAGED_DECOMPILE_MARKERS)


def classify_file(path: Path, repo_root: Path, reverse_root: Path) -> ReverseEngineeringRecord:
    rel = _repo_rel(path, repo_root)
    size = path.stat().st_size
    digest = sha256_file(path)
    live_relpath, live_status = _live_status(path, repo_root)

    if path.suffix.lower() in RAW_EXTENSIONS:
        category = "raw_artifact_in_reverse_engineering"
        disposition = "externalize_with_manifest"
        target = ".omx/research/ or external artifact storage"
        reason = "Raw archives, media, tensors, checkpoints, and logs do not belong in the curated reverse-engineering tree."
    else:
        rel_to_reverse = path.resolve().relative_to(reverse_root.resolve()).as_posix()
        if rel_to_reverse.startswith(f"{ORPHAN_ROOT_NAME}/"):
            rel_after_orphan = rel_to_reverse[len(f"{ORPHAN_ROOT_NAME}/") :]
            operator_shadow_target = _operator_shadow_live_path(rel_after_orphan, repo_root)
            docs_trace_target = repo_root / rel_after_orphan
            if (
                rel_after_orphan == "docs/paper/ara/trace/events.jsonl"
                and docs_trace_target.is_file()
                and _same_trace_except_source_path(docs_trace_target, path)
            ):
                category = "orphan_report_private_path_shadow"
                disposition = "delete_after_manifest"
                target = rel_after_orphan
                reason = (
                    "Recovered ARA trace matches the live sanitized trace except for private absolute "
                    "operator-memory source paths; keep the sanitized docs trace as canonical."
                )
            elif operator_shadow_target is not None:
                category = "orphan_operator_tool_shadow"
                disposition = "delete_after_manifest"
                target = operator_shadow_target
                reason = (
                    "Recovered operator suffix copy is superseded by the canonical unsuffixed script; "
                    "the live script/runbook/tests carry the reusable signal."
                )
            elif _is_damaged_decompile(path):
                category = "orphan_damaged_decompile"
                disposition = "preserve_until_hand_rehydration"
                target = rel_after_orphan
                reason = (
                    "Recovered source contains decompiler damage markers or impossible pycdc constructs; "
                    "preserve it as intent/provenance until a hand-rehydrated implementation and tests exist."
                )
            else:
                category, disposition, target, reason = _classify_orphan(rel_after_orphan, path)
            if live_status == "same_as_main_ignoring_recovery_header":
                category = "orphan_duplicate_recovery_copy"
                disposition = "delete_after_manifest"
                target = live_relpath
                reason = (
                    "Recovered file matches live main after removing the pyc-recovery banner; "
                    "preserve the audit/spec signal, then delete this duplicate copy."
                )
        elif rel_to_reverse.startswith("public_frontier/recovered_runtime/"):
            category = "public_frontier_runtime_reference"
            disposition = "track_in_git"
            target = rel
            reason = (
                "Curated source-sized public-runtime reference under public_frontier; "
                "forensic evidence only, not score evidence or active experiment output."
            )
        elif path.name == ".gitignore" or path.suffix.lower() in {".md", ".json"}:
            category = "curated_reverse_engineering_surface"
            disposition = "track_in_git"
            target = rel
            reason = "Curated notes, manifests, and policy files are the intended reverse-engineering surface."
        else:
            category = "reverse_engineering_manual_review"
            disposition = "manual_review"
            target = rel
            reason = "Non-raw file outside the orphan tree needs explicit review before public tracking."

    return ReverseEngineeringRecord(
        relpath=rel,
        bytes=size,
        sha256=digest,
        category=category,
        disposition=disposition,
        target=target,
        live_relpath=live_relpath,
        live_status=live_status,
        reason=reason,
    )


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in IGNORED_SUFFIXES:
            continue
        if path.is_file():
            yield path


def audit_reverse_engineering_tree(
    repo_root: Path,
    *,
    reverse_root: Path | None = None,
) -> list[ReverseEngineeringRecord]:
    repo_root = repo_root.resolve()
    reverse_root = (repo_root / DEFAULT_ROOT if reverse_root is None else reverse_root).resolve()
    if not reverse_root.exists():
        raise FileNotFoundError(f"reverse-engineering root does not exist: {reverse_root}")
    return [classify_file(path, repo_root, reverse_root) for path in iter_files(reverse_root)]


def render_json(
    records: Sequence[ReverseEngineeringRecord],
    *,
    release_strict: bool = False,
    release_rules: Sequence[ReleaseResolutionRule] = (),
) -> str:
    blockers = (
        release_blocking_records(records, release_rules=release_rules)
        if release_strict
        else blocking_records(records)
    )
    payload = {
        "schema_version": 1,
        "tool": "tac.reverse_engineering_curation",
        "release_strict": release_strict,
        "total_files": len(records),
        "blocking_count": len(blockers),
        "release_resolution_rule_count": len(release_rules),
        "disposition_counts": dict(Counter(record.disposition for record in records)),
        "category_counts": dict(Counter(record.category for record in records)),
        "records": [dataclasses.asdict(record) for record in records],
    }
    return json_text(payload)


def blocking_records(records: Sequence[ReverseEngineeringRecord]) -> list[ReverseEngineeringRecord]:
    """Return records that should fail strict harness/preflight checks."""
    return [record for record in records if record.disposition in BLOCKING_DISPOSITIONS]


def release_blocking_records(
    records: Sequence[ReverseEngineeringRecord],
    *,
    release_rules: Sequence[ReleaseResolutionRule] = (),
) -> list[ReverseEngineeringRecord]:
    """Return records that must be resolved before a public release bundle."""

    blockers: list[ReverseEngineeringRecord] = []
    for record in records:
        if record.disposition not in RELEASE_BLOCKING_DISPOSITIONS:
            continue
        if any(rule.matches(record) for rule in release_rules):
            continue
        blockers.append(record)
    return blockers


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(records: Sequence[ReverseEngineeringRecord]) -> str:
    disposition_counts = Counter(record.disposition for record in records)
    category_counts = Counter(record.category for record in records)
    blockers = blocking_records(records)
    lines = [
        "# Reverse Engineering Tree Audit",
        "",
        f"- total_files: `{len(records)}`",
        f"- blocking_count: `{len(blockers)}`",
        "",
        "## Dispositions",
        "",
        "| disposition | files |",
        "|---|---:|",
    ]
    for disposition, count in sorted(disposition_counts.items()):
        lines.append(f"| `{disposition}` | {count} |")
    lines.extend(["", "## Categories", "", "| category | files |", "|---|---:|"])
    for category, count in sorted(category_counts.items()):
        lines.append(f"| `{category}` | {count} |")
    if blockers:
        lines.extend(["", "## Strict Blockers", "", "| disposition | file | reason |", "|---|---|---|"])
        for record in blockers:
            lines.append(
                f"| `{record.disposition}` | `{record.relpath}` | {_md_escape(record.reason)} |"
            )
    preserved = [
        record
        for record in records
        if record.disposition in {"preserve_until_hand_rehydration"}
    ]
    if preserved:
        lines.extend(
            [
                "",
                "## Preserved Recovery Signal",
                "",
                "| disposition | live status | file | target | reason |",
                "|---|---|---|---|---|",
            ]
        )
        for record in sorted(preserved, key=lambda item: item.relpath):
            lines.append(
                "| `{}` | `{}` | `{}` | `{}` | {} |".format(
                    record.disposition,
                    record.live_status or "",
                    record.relpath,
                    record.target or "",
                    _md_escape(record.reason),
                )
            )
    lines.extend(
        [
            "",
            "## Promotion Queue",
            "",
            "| disposition | live status | file | target | reason |",
            "|---|---|---|---|---|",
        ]
    )
    interesting = [
        record
        for record in records
        if record.disposition
        in {
            "compare_and_promote_to_tac",
            "compare_and_promote_or_ledger",
            "promote_thin_cli_or_ledger",
            "canonicalize_to_docs_or_ledger",
            "manual_review",
        }
    ]
    for record in sorted(interesting, key=lambda item: (item.disposition, item.relpath)):
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | {} |".format(
                record.disposition,
                record.live_status or "",
                record.relpath,
                record.target or "",
                _md_escape(record.reason),
            )
        )
    return "\n".join(lines) + "\n"
