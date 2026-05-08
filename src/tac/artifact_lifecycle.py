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
        try:
            head = subprocess.run(
                ["git", "show", f"HEAD:{rel}"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
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
            head_payload = json.loads(head.stdout)
            wt_payload = json.loads(abs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
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
        # Permit explicit historical-only annotation
        if (
            "HISTORICAL_RECIPE_ONLY" in text
            or "historical-recipe-only" in text.lower()
        ):
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


# ---------------------------------------------------------------------------
# Meta umbrella check (runs all guards on registry-classified files)
# ---------------------------------------------------------------------------


def run_meta_lifecycle_audit(
    repo_root: Path | None = None,
) -> list[str]:
    """Run all four guards against every registry-classified file under the repo.

    Returns a list of violation strings (empty on clean).
    """
    root = Path(repo_root or REPO_ROOT)
    classifier = ArtifactClassifier(root)
    live_guard = LiveStateGuard(root)
    prov_guard = ProvenanceGuard(root)
    recipe_guard = RecipeGuard(root)
    derived_guard = DerivedOutputGuard(root)

    violations: list[str] = []
    seen_patterns: set[str] = set()

    # For each registry pattern, glob-match the actual files and run the
    # appropriate guard.
    for entry in classifier.entries:
        # Avoid running the same pattern twice if listed multiple times.
        if entry.pattern in seen_patterns:
            continue
        seen_patterns.add(entry.pattern)
        # fnmatch + glob: convert pattern to glob walk under repo root
        for path in _glob_under_root(root, entry.pattern):
            if entry.kind is ArtifactKind.LIVE_STATE:
                res = live_guard.check(path)
            elif entry.kind is ArtifactKind.HISTORICAL_PROVENANCE:
                res = prov_guard.check(path, append_fields=entry.append_fields)
            elif entry.kind is ArtifactKind.LIVE_RECIPE:
                res = recipe_guard.check(path)
            elif entry.kind is ArtifactKind.DERIVED_OUTPUT:
                res = derived_guard.check(path)
            else:
                # UNKNOWN — surfaced separately if needed; not a guard target.
                continue
            for v in res.violations:
                violations.append(v)
    return violations


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
    "REGISTRY_RELPATH",
    "ArtifactClassifier",
    "ArtifactKind",
    "ArtifactLifecycleViolation",
    "Classification",
    "DerivedOutputGuard",
    "GuardResult",
    "LiveStateGuard",
    "ProvenanceGuard",
    "RecipeGuard",
    "RegistryEntry",
    "load_registry",
    "run_meta_lifecycle_audit",
]
