# SPDX-License-Identifier: MIT
"""Artifact lifecycle classification + provenance-vs-state guard layer.

Meta-class prevention for the codex-discovered class spanning 5 findings (2026-05-08):

1. operator-approval scope leak — global selector flag froze into per-row state
2. public PR intake clones dirty — upstream source mutated in place
3. status.json poisoned by stale dirty-paths — transient porcelain frozen into committed JSON
4. rebuild_command.txt baked timestamps + approval — transient values hardcoded into "reusable" recipe
5. recovery_metadata.json overwritten timestamps — historical record mutated in place

The common class:
  Transient/global/upstream state being frozen or mutated into committed/forensic
  artifacts in ways that destroy provenance or leak authorization.

Dual error directions:
  A. Transient state TREATED AS provenance (frozen into committed artifacts).
  B. Provenance TREATED AS transient state (mutated in place).

Either direction destroys the audit trail or leaks authorization context.

The fix is a four-kind taxonomy + per-kind guards + a meta preflight gate.

Kinds
-----
- LIVE_STATE — transient session state. Should NEVER be committed.
- HISTORICAL_PROVENANCE — append-only record of what happened.
- LIVE_RECIPE — reusable instructions; baked transient values FORBIDDEN.
- DERIVED_OUTPUT — computed from current state; needs regeneration header.
- UNKNOWN — explicit "not yet classified"; flagged for human triage.

Cross-references
----------------
- Memory: ``feedback_codex_findings_meta_pattern_artifact_lifecycle_FIXED_20260508.md``
- Sister gates (specific): ``check_status_artifacts_no_stale_dirty_paths``,
  ``check_rebuild_commands_no_baked_runtime_state``,
  ``check_recovery_metadata_append_only`` (FIX-3+4 + FIX-5).
- Prior gates (extincted): ``check_operator_approval_must_be_lane_scoped``,
  ``check_public_pr_intake_clones_pristine``.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------


class ArtifactKind(StrEnum):
    """Lifecycle classification for a long-lived repo artifact."""

    LIVE_STATE = "LIVE_STATE"
    HISTORICAL_PROVENANCE = "HISTORICAL_PROVENANCE"
    LIVE_RECIPE = "LIVE_RECIPE"
    DERIVED_OUTPUT = "DERIVED_OUTPUT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, raw: str) -> ArtifactKind:
        try:
            return cls(raw)
        except ValueError as exc:
            raise ValueError(
                f"unknown ArtifactKind {raw!r} (valid: {[k.value for k in cls]})"
            ) from exc


# ---------------------------------------------------------------------------
# Registry — committed YAML maps glob pattern → kind
# ---------------------------------------------------------------------------


REGISTRY_RELPATH = ".omx/state/artifact_kind_registry.yaml"


@dataclass(frozen=True)
class RegistryEntry:
    pattern: str
    kind: ArtifactKind
    rationale: str
    append_fields: tuple[str, ...] = ()  # only meaningful for HISTORICAL_PROVENANCE


def load_registry(repo_root: Path | None = None) -> list[RegistryEntry]:
    """Load the committed artifact-kind registry. YAML is parsed by hand to
    avoid adding a yaml dependency for a simple flat schema."""
    root = Path(repo_root or REPO_ROOT)
    path = root / REGISTRY_RELPATH
    if not path.is_file():
        return []
    entries: list[RegistryEntry] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        if line.startswith("- ") or stripped.startswith("- "):
            # New entry start.
            if current is not None and "pattern" in current and "kind" in current:
                entries.append(_entry_from_dict(current))
            current = {}
            after = stripped[2:]
            if ":" in after:
                k, _, v = after.partition(":")
                current[k.strip()] = _parse_yaml_scalar(v.strip())
            continue
        if current is None:
            continue
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            current[k.strip()] = _parse_yaml_scalar(v.strip())
    if current is not None and "pattern" in current and "kind" in current:
        entries.append(_entry_from_dict(current))
    return entries


def _parse_yaml_scalar(v: str) -> Any:
    if (v.startswith('"') and v.endswith('"')) or (
        v.startswith("'") and v.endswith("'")
    ):
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip().strip("\"'") for p in inner.split(",")]
        return tuple(parts)
    return v


def _entry_from_dict(d: dict[str, Any]) -> RegistryEntry:
    fields_raw = d.get("append_fields", ())
    if isinstance(fields_raw, str):
        fields_raw = (fields_raw,)
    return RegistryEntry(
        pattern=str(d["pattern"]),
        kind=ArtifactKind.from_str(str(d["kind"])),
        rationale=str(d.get("rationale", "")),
        append_fields=tuple(fields_raw) if fields_raw else (),
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    path: str
    kind: ArtifactKind
    matched_pattern: str | None
    rationale: str
    append_fields: tuple[str, ...] = ()


class ArtifactClassifier:
    """Classify a path against the committed registry."""

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = Path(repo_root or REPO_ROOT)
        self.entries = load_registry(self.repo_root)

    def classify_path(self, path: Path | str) -> Classification:
        """Return classification for ``path`` relative to repo root.

        Patterns are matched with fnmatch. First match wins (registry order).
        Unmatched paths return UNKNOWN.
        """
        rel = self._rel(path)
        for entry in self.entries:
            if fnmatch.fnmatchcase(rel, entry.pattern):
                return Classification(
                    path=rel,
                    kind=entry.kind,
                    matched_pattern=entry.pattern,
                    rationale=entry.rationale,
                    append_fields=entry.append_fields,
                )
        return Classification(
            path=rel,
            kind=ArtifactKind.UNKNOWN,
            matched_pattern=None,
            rationale="no registry pattern matched",
            append_fields=(),
        )

    def _rel(self, path: Path | str) -> str:
        p = Path(path)
        if p.is_absolute():
            try:
                return p.relative_to(self.repo_root).as_posix()
            except ValueError:
                return p.as_posix()
        return p.as_posix()

    def assert_kind(self, path: Path | str, expected: ArtifactKind) -> None:
        """Raise if ``path``'s classification doesn't match ``expected``."""
        cls = self.classify_path(path)
        if cls.kind != expected:
            raise ArtifactLifecycleViolation(
                f"{cls.path}: expected kind {expected.value}, got {cls.kind.value} "
                f"(matched={cls.matched_pattern!r}, rationale={cls.rationale!r})"
            )


class ArtifactLifecycleViolation(Exception):
    """An artifact violates its declared lifecycle kind."""


# ---------------------------------------------------------------------------
# Per-kind guards
# ---------------------------------------------------------------------------


# Patterns that strongly suggest baked transient values in LIVE_RECIPE files.
_TRANSIENT_BAKE_PATTERNS = [
    # Baked timestamps that should be parameterized
    (r"--now-utc\s+20\d{2}-\d{2}-\d{2}T\d{2}", "baked --now-utc timestamp"),
    (r"--operator-approved-[a-zA-Z0-9_-]+(?!\s*\$|\s*\{)", "baked --operator-approved-* flag"),
    # /tmp paths in committed artifacts (per CLAUDE.md transient-evidence rule)
    (r"/tmp/[A-Za-z0-9_./-]+", "/tmp path baked into recipe"),
    # Hardcoded Vast.ai instance IDs
    (r"vastai[^\n]*\b\d{8,}\b", "hardcoded Vast.ai instance id"),
]


@dataclass
class GuardResult:
    path: str
    kind: ArtifactKind
    violations: list[str] = field(default_factory=list)


class LiveStateGuard:
    """LIVE_STATE files must be gitignored (never committed)."""

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = Path(repo_root or REPO_ROOT)

    def check(self, path: Path | str) -> GuardResult:
        rel = _rel_path(path, self.repo_root)
        violations: list[str] = []
        if _is_tracked_by_git(self.repo_root, rel):
            violations.append(
                f"LIVE_STATE file {rel!r} is tracked by git — must be gitignored"
            )
        return GuardResult(path=rel, kind=ArtifactKind.LIVE_STATE, violations=violations)


class ProvenanceGuard:
    """HISTORICAL_PROVENANCE files: forbids field mutation outside append_fields.

    Compares the prior-commit version (HEAD) to the working-tree version. Any
    top-level scalar that changed is flagged unless the file's registered
    ``append_fields`` lists the field as append-allowed.
    """

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = Path(repo_root or REPO_ROOT)

    def check(self, path: Path | str, append_fields: tuple[str, ...] = ()) -> GuardResult:
        rel = _rel_path(path, self.repo_root)
        violations: list[str] = []
        # Catalog #113 (2026-05-09): use text=False so binary HISTORICAL_PROVENANCE
        # files (.bin/.pt/.zip/.mkv/etc.) don't crash subprocess on UnicodeDecodeError.
        # We try to decode-and-JSON-parse below; binary content fails JSON parse
        # cleanly and exits the guard early.
        try:
            head = subprocess.run(
                ["git", "show", f"HEAD:{rel}"],
                cwd=self.repo_root,
                capture_output=True,
                text=False,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            head = None
        if head is None or head.returncode != 0:
            return GuardResult(
                path=rel, kind=ArtifactKind.HISTORICAL_PROVENANCE, violations=[]
            )
        abs_path = self.repo_root / rel
        if not abs_path.is_file():
            return GuardResult(
                path=rel, kind=ArtifactKind.HISTORICAL_PROVENANCE, violations=[]
            )
        try:
            head_text = head.stdout.decode("utf-8")
            head_payload = json.loads(head_text)
            wt_payload = json.loads(abs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return GuardResult(
                path=rel, kind=ArtifactKind.HISTORICAL_PROVENANCE, violations=[]
            )
        if not isinstance(head_payload, dict) or not isinstance(wt_payload, dict):
            return GuardResult(
                path=rel, kind=ArtifactKind.HISTORICAL_PROVENANCE, violations=[]
            )
        # Check every top-level scalar field for in-place mutation.
        for key, prev in head_payload.items():
            if key in append_fields:
                continue
            if isinstance(prev, (dict, list)):
                continue
            cur = wt_payload.get(key, prev)
            if cur != prev:
                violations.append(
                    f"HISTORICAL_PROVENANCE {rel!r} field {key!r} mutated "
                    f"({prev!r} -> {cur!r}); append-only — extend "
                    f"append_fields registry entry to permit OR add a new "
                    f"attempt entry instead of mutating in place"
                )
        return GuardResult(
            path=rel, kind=ArtifactKind.HISTORICAL_PROVENANCE, violations=violations
        )


class CommittedRangeProvenanceGuard:
    """HISTORICAL_PROVENANCE guard for already-committed ranges.

    ``ProvenanceGuard`` protects the pre-commit working tree by comparing
    ``HEAD:<path>`` to the file on disk. Once a bad mutation is committed,
    that comparison becomes self-equal. This guard closes the CI/review gap by
    comparing ``base_ref:<path>`` to ``head_ref:<path>``.
    """

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = Path(repo_root or REPO_ROOT)

    def check(
        self,
        path: Path | str,
        *,
        append_fields: tuple[str, ...] = (),
        base_ref: str,
        head_ref: str = "HEAD",
    ) -> GuardResult:
        rel = _rel_path(path, self.repo_root)
        before = _git_show_json(self.repo_root, base_ref, rel)
        after = _git_show_json(self.repo_root, head_ref, rel)
        if before is None or after is None:
            return GuardResult(
                path=rel, kind=ArtifactKind.HISTORICAL_PROVENANCE, violations=[]
            )
        if not isinstance(before, dict) or not isinstance(after, dict):
            return GuardResult(
                path=rel, kind=ArtifactKind.HISTORICAL_PROVENANCE, violations=[]
            )

        violations: list[str] = []
        append_field_set = set(append_fields)
        for key, prev in before.items():
            cur = after.get(key, prev)
            if key in append_field_set:
                if isinstance(prev, list) and isinstance(cur, list) and cur[: len(prev)] != prev:
                    violations.append(
                        f"HISTORICAL_PROVENANCE {rel!r} append field {key!r} "
                        f"mutated existing entries across {base_ref}..{head_ref}; "
                        "append-only fields may add entries but not rewrite prior entries"
                    )
                continue
            if isinstance(prev, (dict, list)):
                continue
            if cur != prev:
                violations.append(
                    f"HISTORICAL_PROVENANCE {rel!r} field {key!r} mutated "
                    f"across committed range {base_ref}..{head_ref} "
                    f"({prev!r} -> {cur!r}); append a new record instead of "
                    "rewriting provenance"
                )
        return GuardResult(
            path=rel, kind=ArtifactKind.HISTORICAL_PROVENANCE, violations=violations
        )


class RecipeGuard:
    """LIVE_RECIPE files must not contain baked transient values."""

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = Path(repo_root or REPO_ROOT)

    def check(self, path: Path | str) -> GuardResult:
        rel = _rel_path(path, self.repo_root)
        abs_path = self.repo_root / rel
        violations: list[str] = []
        if not abs_path.is_file():
            return GuardResult(path=rel, kind=ArtifactKind.LIVE_RECIPE, violations=[])
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return GuardResult(path=rel, kind=ArtifactKind.LIVE_RECIPE, violations=[])
        # Permit explicit historical-only annotation. Sister gate #112
        # (`check_rebuild_commands_no_baked_runtime_state`) accepts a broader
        # set of phrases — keep them aligned for cross-gate consistency.
        # Otherwise the META gate flags files the per-finding gate accepts,
        # which produces silent noise / false positives. 2026-05-09 alignment
        # under catalog #113 strict-flip cleanup.
        text_lower = text.lower()
        historical_markers = (
            "historical_recipe_only",
            "historical-recipe-only",
            "historical artifact",
            "do not replay",
            "frozen historical",
            "forensic reproduction",
        )
        if any(marker in text_lower for marker in historical_markers):
            return GuardResult(path=rel, kind=ArtifactKind.LIVE_RECIPE, violations=[])
        for pat, label in _TRANSIENT_BAKE_PATTERNS:
            for match in re.finditer(pat, text):
                snippet = match.group(0)
                violations.append(
                    f"LIVE_RECIPE {rel!r}: {label} -> {snippet!r}; "
                    f"parameterize ($ or {{}} placeholder) OR add HISTORICAL_RECIPE_ONLY header"
                )
        return GuardResult(path=rel, kind=ArtifactKind.LIVE_RECIPE, violations=violations)


_REGEN_HEADER_RE = re.compile(
    r"\b(generated_at|generated_utc)\b.*?\b(from_state_hash|source_state_hash|input_sha)\b",
    re.IGNORECASE | re.DOTALL,
)


class DerivedOutputGuard:
    """DERIVED_OUTPUT files need a regeneration header (within first 4 KB)."""

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = Path(repo_root or REPO_ROOT)

    def check(self, path: Path | str) -> GuardResult:
        rel = _rel_path(path, self.repo_root)
        abs_path = self.repo_root / rel
        violations: list[str] = []
        if not abs_path.is_file():
            return GuardResult(
                path=rel, kind=ArtifactKind.DERIVED_OUTPUT, violations=[]
            )
        try:
            head = abs_path.read_text(encoding="utf-8", errors="replace")[:4096]
        except OSError:
            return GuardResult(
                path=rel, kind=ArtifactKind.DERIVED_OUTPUT, violations=[]
            )
        if not _REGEN_HEADER_RE.search(head):
            violations.append(
                f"DERIVED_OUTPUT {rel!r}: missing regeneration header "
                f"(generated_at + from_state_hash within first 4KB) — "
                f"derived outputs must declare their regeneration provenance"
            )
        return GuardResult(
            path=rel, kind=ArtifactKind.DERIVED_OUTPUT, violations=violations
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rel_path(path: Path | str, repo_root: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return p.relative_to(repo_root).as_posix()
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def _is_tracked_by_git(repo_root: Path, relpath: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relpath],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _tracked_paths_matching_pattern(
    root: Path,
    pattern: str,
    *,
    tracked_relpaths: tuple[str, ...] | None = None,
) -> list[Path]:
    """Return tracked repo paths matching ``pattern`` without walking ignored trees.

    Historical-provenance patterns intentionally cover large experiment result
    globs. Strict preflight must audit committed provenance, not spend minutes
    enumerating ignored rebuildable output directories.
    """
    if pattern.startswith("/"):
        return []
    if tracked_relpaths is None:
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return []
        if result.returncode != 0:
            return []
        tracked_relpaths = tuple(result.stdout.splitlines())
    return [
        root / rel
        for rel in tracked_relpaths
        if fnmatch.fnmatchcase(rel, pattern)
    ]


def _git_show_json(repo_root: Path, ref: str, relpath: str) -> Any | None:
    # Catalog #113 (2026-05-09): use text=False so binary
    # HISTORICAL_PROVENANCE files don't crash on UnicodeDecodeError; decode
    # and JSON-parse explicitly inside the try/except so binary content
    # exits the guard early.
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{relpath}"],
            cwd=repo_root,
            capture_output=True,
            text=False,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Meta umbrella check (runs all guards on registry-classified files)
# ---------------------------------------------------------------------------


# FIX-4 2026-05-08: long-lived-artifact roots that the META audit enumerates
# unconditionally (catches artifacts the registry missed). Codex verification
# review caught that `run_meta_lifecycle_audit` only iterated REGISTERED
# patterns; new artifact paths outside the registry slipped past the META
# gate entirely. Now every tracked file under these roots is classified, and
# UNKNOWN (unregistered) classifications fail strict mode unless explicitly
# allowlisted at `.omx/state/artifact_classification_allowlist.json`.
LONG_LIVED_ARTIFACT_ROOTS: tuple[str, ...] = (
    "experiments/results/",
    ".omx/state/",
    ".omx/research/",
    "reports/",
    "docs/",
    "runtime-rs/",
    "cuda/",
    "submissions/robust_current/",
)
ALLOWLIST_RELPATH = ".omx/state/artifact_classification_allowlist.json"
LARGE_RESEARCH_JSON_THRESHOLD_BYTES = 50 * 1024 * 1024
LARGE_RESEARCH_JSON_MANIFEST_SCHEMA = "omx.large_research_artifact_manifest.v1"


def _load_classification_allowlist(repo_root: Path) -> set[str]:
    """Load explicit per-path allowlist for UNKNOWN long-lived artifacts.

    Format: {"allowlisted_paths": [{"path": "...", "reason": "..."}], ...}
    """
    p = repo_root / ALLOWLIST_RELPATH
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {entry["path"] for entry in data.get("allowlisted_paths", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return set()


def _enumerate_long_lived_tracked_paths(repo_root: Path) -> list[str]:
    """git ls-files restricted to long-lived-artifact roots."""
    try:
        result = subprocess.run(
            ["git", "ls-files", *LONG_LIVED_ARTIFACT_ROOTS],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def audit_unregistered_long_lived_artifacts(
    repo_root: Path | None = None,
    *,
    path_filter: set[str] | None = None,
) -> list[str]:
    """FIX-4: enumerate tracked long-lived artifacts; flag UNKNOWN classifications.

    Returns one violation string per unregistered tracked artifact under the
    long-lived roots that classifies as UNKNOWN AND is NOT in the explicit
    allowlist. Empty list = clean.

    Bug class fixed: codex verification 2026-05-08 caught that
    ``run_meta_lifecycle_audit`` only iterated REGISTERED patterns from the
    YAML registry. New long-lived artifacts outside the registry slipped
    past the META gate entirely, recreating the provenance-vs-state-confusion
    bug class for any future tracked artifact under
    `experiments/results/`, `.omx/state/`, `reports/`, `docs/`, etc.

    The fix enumerates `git ls-files` restricted to the long-lived roots
    and asserts every result is either a known kind (LIVE_STATE / HP /
    LIVE_RECIPE / DERIVED_OUTPUT) OR explicitly allowlisted with a reason.
    """
    root = Path(repo_root or REPO_ROOT)
    classifier = ArtifactClassifier(root)
    allowlist = _load_classification_allowlist(root)
    paths = _enumerate_long_lived_tracked_paths(root)
    violations: list[str] = []
    for relpath in paths:
        if path_filter is not None and relpath not in path_filter:
            continue
        if relpath in allowlist:
            continue
        classification = classifier.classify_path(relpath)
        if classification.kind is ArtifactKind.UNKNOWN:
            violations.append(
                f"UNKNOWN long-lived artifact {relpath!r}: not classified by "
                f"any registry pattern AND not in allowlist at "
                f"{ALLOWLIST_RELPATH}. Either add a registry pattern that "
                f"classifies it (LIVE_STATE / HISTORICAL_PROVENANCE / "
                f"LIVE_RECIPE / DERIVED_OUTPUT), or allowlist it with a "
                f"justification."
            )
    return violations


def audit_large_rebuildable_research_json_artifacts(
    repo_root: Path | None = None,
    *,
    path_filter: set[str] | None = None,
    threshold_bytes: int = LARGE_RESEARCH_JSON_THRESHOLD_BYTES,
) -> list[str]:
    """Flag bulky tracked research JSON unless compact custody is present.

    ``.omx/research/**/*.json`` is historical provenance by default, but huge
    generated payloads create source-control and disk-pressure failures unless
    the durable record is a compact manifest plus reproducible source contract.
    The audit intentionally targets tracked/indexed files, so new files are
    caught once staged while untracked scratch output can remain local.
    """
    root = Path(repo_root or REPO_ROOT)
    violations: list[str] = []
    for relpath in _enumerate_omx_research_json_paths(root):
        if path_filter is not None and relpath not in path_filter:
            continue
        if relpath.endswith((".compact_manifest.json", ".compact_summary.json")):
            continue
        path = root / relpath
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= threshold_bytes:
            continue
        violations.extend(
            _validate_large_research_json_manifest(
                root,
                relpath=relpath,
                size=size,
                threshold_bytes=threshold_bytes,
            )
        )
    return violations


def run_meta_lifecycle_audit(
    repo_root: Path | None = None,
    *,
    base_ref: str | None = None,
    head_ref: str = "HEAD",
    path_filter: set[str] | None = None,
    enumerate_unregistered: bool = False,
) -> list[str]:
    """Run all four guards against every registry-classified file under the repo.

    When ``path_filter`` is provided, only matching repo-relative paths are
    audited. This is how preflight blocks newly changed lifecycle violations
    without rewriting historical evidence artifacts during legacy migration.

    When ``enumerate_unregistered`` is True, ALSO enumerates `git ls-files`
    under long-lived-artifact roots and flags any tracked file that
    classifies as UNKNOWN (FIX-4 — closes the codex-verification gap where
    unregistered artifacts slipped past the META gate). Default is False
    pending operator allowlist baseline (~4016 currently-unregistered
    artifacts in the repo; call ``audit_unregistered_long_lived_artifacts``
    directly for the explicit unregistered-only audit, OR pass
    ``enumerate_unregistered=True`` once the allowlist at
    `.omx/state/artifact_classification_allowlist.json` is populated).

    Returns a list of violation strings (empty on clean).
    """
    root = Path(repo_root or REPO_ROOT)
    classifier = ArtifactClassifier(root)
    live_guard = LiveStateGuard(root)
    prov_guard = ProvenanceGuard(root)
    committed_prov_guard = CommittedRangeProvenanceGuard(root)
    recipe_guard = RecipeGuard(root)
    derived_guard = DerivedOutputGuard(root)

    violations: list[str] = []
    seen_patterns: set[str] = set()
    try:
        tracked_result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        tracked_relpaths: tuple[str, ...] = ()
    else:
        tracked_relpaths = (
            tuple(tracked_result.stdout.splitlines())
            if tracked_result.returncode == 0
            else ()
        )

    # For each registry pattern, glob-match the actual files and run the
    # appropriate guard.
    for entry in classifier.entries:
        # Avoid running the same pattern twice if listed multiple times.
        if entry.pattern in seen_patterns:
            continue
        seen_patterns.add(entry.pattern)
        # Audit durable repo artifacts, not the full ignored filesystem. Every
        # lifecycle kind is a source-control contract: LIVE_STATE must not be
        # tracked; HISTORICAL_PROVENANCE must be append-only once tracked;
        # LIVE_RECIPE / DERIVED_OUTPUT rules protect committed surfaces. Using
        # git-ls-files here keeps strict preflight bounded even when broad
        # registry patterns overlap large experiment result trees.
        paths = _tracked_paths_matching_pattern(
            root,
            entry.pattern,
            tracked_relpaths=tracked_relpaths,
        )
        for path in paths:
            rel = _rel_path(path, root)
            if path_filter is not None and rel not in path_filter:
                continue
            if entry.kind is ArtifactKind.LIVE_STATE:
                res = live_guard.check(path)
            elif entry.kind is ArtifactKind.HISTORICAL_PROVENANCE:
                res = prov_guard.check(path, append_fields=entry.append_fields)
                if base_ref:
                    committed_res = committed_prov_guard.check(
                        path,
                        append_fields=entry.append_fields,
                        base_ref=base_ref,
                        head_ref=head_ref,
                    )
                    res.violations.extend(committed_res.violations)
            elif entry.kind is ArtifactKind.LIVE_RECIPE:
                res = recipe_guard.check(path)
            elif entry.kind is ArtifactKind.DERIVED_OUTPUT:
                res = derived_guard.check(path)
            else:
                # UNKNOWN — surfaced separately if needed; not a guard target.
                continue
            for v in res.violations:
                violations.append(v)
    # FIX-4: also flag UNKNOWN-classified long-lived artifacts.
    if enumerate_unregistered:
        violations.extend(
            audit_unregistered_long_lived_artifacts(
                repo_root=root,
                path_filter=path_filter,
            )
        )
        violations.extend(
            audit_large_rebuildable_research_json_artifacts(
                repo_root=root,
                path_filter=path_filter,
            )
        )
    return violations


def _enumerate_omx_research_json_paths(repo_root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", ".omx/research"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if result.returncode != 0:
        return []
    return [
        rel
        for rel in result.stdout.splitlines()
        if (
            rel.startswith(".omx/research/")
            and rel.endswith(".json")
            and not rel.endswith(".jsonl")
        )
    ]


def _validate_large_research_json_manifest(
    repo_root: Path,
    *,
    relpath: str,
    size: int,
    threshold_bytes: int,
) -> list[str]:
    manifest_rel = _find_large_research_json_manifest(repo_root, relpath)
    if manifest_rel is None:
        return [
            f"LARGE_REBUILDABLE_RESEARCH_JSON {relpath!r}: {size} bytes exceeds "
            f"{threshold_bytes} bytes and no sibling compact "
            "manifest was found. Externalize rebuildable payloads and commit a "
            f"{LARGE_RESEARCH_JSON_MANIFEST_SCHEMA} manifest with source hash, "
            "byte count, rebuild/source provenance, summary path, and false "
            "authority fields set false."
        ]

    manifest_path = repo_root / manifest_rel
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{manifest_rel}: compact manifest for {relpath!r} is unreadable: {exc}"]
    if not isinstance(manifest, dict):
        return [f"{manifest_rel}: compact manifest for {relpath!r} is not a JSON object"]

    violations: list[str] = []
    if manifest.get("schema") != LARGE_RESEARCH_JSON_MANIFEST_SCHEMA:
        violations.append(
            f"{manifest_rel}: schema must be {LARGE_RESEARCH_JSON_MANIFEST_SCHEMA!r}"
        )
    if manifest.get("source_json_path") != relpath:
        violations.append(f"{manifest_rel}: source_json_path does not match {relpath!r}")
    if manifest.get("source_json_bytes") != size:
        violations.append(
            f"{manifest_rel}: source_json_bytes {manifest.get('source_json_bytes')!r} "
            f"does not match current size {size}"
        )
    actual_sha = _sha256_file(repo_root / relpath)
    if manifest.get("source_json_sha256") != actual_sha:
        violations.append(f"{manifest_rel}: source_json_sha256 is stale or missing")
    if not _has_rebuild_or_source_inputs(manifest):
        violations.append(
            f"{manifest_rel}: must include non-empty rebuild_command or source_inputs"
        )
    compact_summary_path = manifest.get("compact_summary_path")
    if not isinstance(compact_summary_path, str) or not compact_summary_path:
        violations.append(f"{manifest_rel}: compact_summary_path must be non-empty")
    elif not (repo_root / compact_summary_path).is_file():
        violations.append(
            f"{manifest_rel}: compact_summary_path {compact_summary_path!r} does not exist"
        )
    for authority_field in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if manifest.get(authority_field) is not False:
            violations.append(f"{manifest_rel}: {authority_field} must be false")
    return violations


def _find_large_research_json_manifest(repo_root: Path, relpath: str) -> str | None:
    source = Path(relpath)
    preferred = source.with_suffix(".compact_manifest.json").as_posix()
    appended = f"{relpath}.compact_manifest.json"
    for candidate in (preferred, appended):
        if (repo_root / candidate).is_file():
            return candidate
    return None


def _has_rebuild_or_source_inputs(manifest: dict[str, Any]) -> bool:
    rebuild_command = manifest.get("rebuild_command")
    if isinstance(rebuild_command, str) and rebuild_command.strip():
        return True
    source_inputs = manifest.get("source_inputs")
    return bool(
        isinstance(source_inputs, (list, tuple, dict)) and source_inputs
    ) or bool(isinstance(source_inputs, str) and source_inputs.strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _glob_under_root(root: Path, pattern: str) -> list[Path]:
    """Glob ``pattern`` (fnmatch-style relative to root) into actual paths.

    Skips absolute-path patterns (e.g. ``/tmp/codex_runs/**``) because they
    live outside the repo root and pathlib refuses non-relative patterns
    in glob(). Such patterns are still classifiable via fnmatch on the
    string path; they just don't get globbed into actual files.
    """
    if pattern.startswith("/"):
        # Absolute paths are out of repo scope; skip glob audit for them.
        return []
    try:
        return list(root.glob(pattern))
    except (ValueError, OSError, NotImplementedError):
        return []


__all__ = [
    "ALLOWLIST_RELPATH",
    "LONG_LIVED_ARTIFACT_ROOTS",
    "REGISTRY_RELPATH",
    "ArtifactClassifier",
    "ArtifactKind",
    "ArtifactLifecycleViolation",
    "Classification",
    "CommittedRangeProvenanceGuard",
    "DerivedOutputGuard",
    "GuardResult",
    "LiveStateGuard",
    "ProvenanceGuard",
    "RecipeGuard",
    "RegistryEntry",
    "audit_large_rebuildable_research_json_artifacts",
    "audit_unregistered_long_lived_artifacts",
    "load_registry",
    "run_meta_lifecycle_audit",
]
