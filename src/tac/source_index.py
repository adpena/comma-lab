"""Shared source-tree inventory and parser cache for developer scanners.

This module is intentionally Python-first. It gives preflight-style scanners a
normal object to share file discovery, source text, and AST parses within one
process without monkeypatching ``pathlib``. Rust can replace the inventory
backend later, but the scanner contract stays small and testable here.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import fnmatch
import functools
import hashlib
import json
import os
import subprocess
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

_CURRENT_SOURCE_INDEX: contextvars.ContextVar[SourceIndex | None] = contextvars.ContextVar(
    "tac_current_source_index",
    default=None,
)
_TEXT_FACTS_CACHE_SCHEMA = "pact.source_text_facts.v26"
_DEFAULT_FACT_WORKERS = 8


@functools.lru_cache(maxsize=131_072)
def _safe_resolve(path: Path) -> Path:
    """Resolve ``path`` once per process for source-tree scanner keying.

    Preflight scanners repeatedly ask about the same few thousand source paths.
    `Path.resolve()` walks every path component and performs many `lstat` calls;
    caching keeps cold hosted-runner preflight bounded without changing the
    scanner contract.
    """

    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _path_key(path: Path) -> str:
    return _safe_resolve(path).as_posix()


def _source_index_fact_workers() -> int:
    """Return the worker budget for one SourceIndex fact extraction group.

    Preflight already runs independent checks concurrently. Letting every
    broad check also fan out to dozens of text-fact workers oversubscribes the
    machine and can make wall-clock latency worse than a sequential run. The
    default keeps intra-file parallelism but bounds nested fan-out; the env var
    exists for profiling on larger CI hosts.
    """

    raw = os.environ.get("PACT_SOURCE_INDEX_FACT_WORKERS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = _DEFAULT_FACT_WORKERS
    else:
        value = _DEFAULT_FACT_WORKERS
    return max(1, min(value, 32))


def _persistent_text_facts_enabled() -> bool:
    """Return whether this process should read/write persistent text facts."""

    raw = os.environ.get("PACT_SOURCE_INDEX_PERSISTENT_TEXT_FACTS", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    # Hosted CI starts from a cold checkout and discards `.omx/cache` after the
    # job, so serializing thousands of SourceTextFacts rows is pure overhead.
    return os.environ.get("GITHUB_ACTIONS", "").strip().lower() != "true"


_DEFAULT_TEXT_FACT_NEEDLES = frozenset(
    {
        "--disable-eval-roundtrip",
        "--device mps",
        "--device",
        "--inflate-sh",
        "--no-eval-roundtrip",
        "--archive",
        "--baseline-archive",
        "--device cpu",
        "--half-frame",
        "--pair-weights",
        "--perturbation-plan",
        "--remote-pact",
        "--repo-dir",
        "--required-device",
        "--required-samples",
        "--rmote",
        "--upstream-dir",
        "--video",
        "--with-uniward-delta",
        ".exists",
        ".round(",
        "AUTHORITATIVE_TAGS",
        "COMPLIANCE_APPROVED",
        "False",
        "F.interpolate",
        "KLDivLoss",
        "No additional approval needed",
        "PYTHONPATH",
        "READY-TO-LAUNCH",
        "caller is responsible",
        "deploy",
        "overrides " "this stub",
        "production",
        "wrapper",
        "WARN ",
        "WARNING:",
        "ZipFile",
        "'create'",
        "'instance'",
        '"create"',
        '"instance"',
        "[WARN]",
        "archive.zip",
        "args.output_dir",
        "batchmean",
        "build_baseline_archive",
        "cuda.is_available",
        "eval_roundtrip",
        "eval_roundtrip=False",
        "extractall(",
        "find ",
        "for path in",
        "from torch import cuda",
        "getattr(torch",
        "is_available",
        "isinstance",
        "json.dumps",
        "kl_distill_segnet_only",
        "kl_div",
        ".amrc",
        ".bin",
        ".mkv",
        ".mp4",
        ".pt",
        ".pth",
        ".raw",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".zip",
        "make_synthetic_pair_batch",
        "make_smoke",
        "make_synthetic",
        "mps",
        "pack_sparse_delta",
        "Path(__file__).resolve().parents",
        "compliance_status",
        "reconstruct_poses",
        "score_claim",
        "score_pair_components(",
        "dispatch_attempted",
        "sha256",
        "torch",
        "torch.frombuffer",
        "torch.load",
        "sys.path.insert",
        "tar",
        "_enforce_eval_roundtrip(args",
        "add_argument",
        "[contest-CUDA]",
        "contest-CUDA",
        "[contest-CPU",
        "contest-CPU",
        "[MPS-PROXY]",
        "MPS",
        "CPU",
        "advisory only",
        "GREEN",
        "RED",
        "KILL",
        "killed",
        "promote",
        "promoted",
        "FALSIFIED",
        "FALSIFICATION",
        "evidence_grade in {",  # CUSTODY_VALIDATOR_OK: source-index needle only
        "evidence_grade.lower() in {",  # CUSTODY_VALIDATOR_OK: source-index needle only
        "evidence_grade not in {",  # CUSTODY_VALIDATOR_OK: source-index needle only
        "evidence_grade.lower() not in {",  # CUSTODY_VALIDATOR_OK: source-index needle only
        'tag.startswith("[contest-CUDA")',  # CUSTODY_VALIDATOR_OK: source-index needle only
        'tag.startswith("[contest-CPU")',  # CUSTODY_VALIDATOR_OK: source-index needle only
        'evidence_tag.startswith("[contest-CUDA")',  # CUSTODY_VALIDATOR_OK: source-index needle only
        'evidence_tag.startswith("[contest-CPU")',  # CUSTODY_VALIDATOR_OK: source-index needle only
        "/tmp/",
        "_partial",
        "auth_eval",
        "cp ",
        "dispatched",
        "blessed",
        "install",
        "ln -s",
        "launch_lightning_batch_job.py",
        "launch",
        "mv ",
        "nohup",
        "path",
        "path=",
        "pgrep",
        "pgrep -f",
        "read",
        "read ",
        "rsync",
        "save_posterior",
        "self.pose_scorer(",
        "self.seg_scorer(",
        "scp",
        "-printf",
        "Standing instruction",
        "_posterior_lock",
        "ast.AnnAssign",
        "ast.Assign",
        ".omx/state/",
        ".commit-lock",
        "active_lane_dispatch_claims",
        "azure_active_vms",
        "commit-serializer",
        "continual_learning_posterior",
        "cuda_cpu_axis_profile_registry",
        "lane_c_compliance_attestations",
        "lane_maturity_audit",
        "lane_registry.json",
        "lightning_active_jobs",
        "lightning_active_sessions",
        "lightning_batch_jobs",
        "next_catalog_number",
        "review_policy",
        "vastai_active_instances",
        "existing.update(",
        "previous.update(",
        "prev.update(",
        "current.update(",
        "stored.update(",
        "on_disk.update(",
        "loaded.update(",
        "VALIDATOR_TOKENS",
        "VALIDATOR_PATTERNS",
        "VALIDATOR_FNS",
        "ACCEPT_TOKENS",
        "CUSTODY_TOKENS",
        "GATE_TOKENS",
        "GUARD_TOKENS",
        "zip",
        "/Users/",
        "/home/adpena/",
    }
)
_DEFAULT_CASEFOLD_TEXT_FACT_NEEDLES = frozenset(
    {
        # Only case-insensitive scanner prefilters belong here. Keeping this
        # intentionally tiny avoids copying and rescanning every source file for
        # hundreds of exact-match needles during cold preflight.
        "No additional approval needed",
        "READY-TO-LAUNCH",
        "Standing instruction",
        "caller is responsible",
        "deploy",
        "launch",
        "overrides " "this stub",
        "production",
        "wrapper",
    }
)
_DEFAULT_CASEFOLD_TEXT_FACT_NEEDLE_PAIRS = tuple(
    (needle, needle.casefold()) for needle in sorted(_DEFAULT_CASEFOLD_TEXT_FACT_NEEDLES)
)


def _source_line_count(text: str) -> int:
    """Return ``str.splitlines()`` line count without allocating all lines."""

    if not text:
        return 0
    count = text.count("\n")
    if not text.endswith(("\n", "\r")):
        count += 1
    return count


def _casefold_substrings_for_text(text: str) -> frozenset[str]:
    """Return casefolded prefilter needles present in ``text``."""

    if not _DEFAULT_CASEFOLD_TEXT_FACT_NEEDLE_PAIRS:
        return frozenset()
    folded_text = text.casefold()
    return frozenset(
        folded
        for _needle, folded in _DEFAULT_CASEFOLD_TEXT_FACT_NEEDLE_PAIRS
        if folded in folded_text
    )


@dataclass(frozen=True)
class SourceTextFacts:
    """Cheap one-pass lexical facts for a source file.

    These facts are intentionally limited to information that is safe to
    compute while the source text is already open: stable path metadata, line
    count and a small set of hot exact substrings used by
    broad preflight scanners. Checks still run their precise AST validation on
    candidate files; this object only prevents reopening every file for every
    scanner.
    """

    path: Path
    rel_path: str
    suffix: str
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    device: int
    line_count: int
    tokens: frozenset[str]
    substrings: frozenset[str]
    casefold_substrings: frozenset[str] = frozenset()

    def contains(self, needle: str) -> bool:
        """Return true when a known token or substring appears in this file."""

        return needle in self.substrings or needle in self.tokens

    def contains_all(self, needles: Sequence[str]) -> bool:
        """Return true when every known token/substr in ``needles`` appears."""

        return all(self.contains(needle) for needle in needles)

    def contains_any(self, needles: Sequence[str]) -> bool:
        """Return true when any known token/substr in ``needles`` appears."""

        return any(self.contains(needle) for needle in needles)

    def contains_casefold(self, needle: str) -> bool:
        """Return true when ``needle`` appears case-insensitively."""

        return needle.casefold() in self.casefold_substrings


@dataclass(frozen=True)
class ScannerQuery:
    """Declarative candidate-file query for broad scanner checks.

    Checks should describe the candidate set they need and let ``SourceIndex``
    perform one inventory/fact pass. This keeps policy checks readable while
    making it possible to fuse repeated scans and, later, replace lexical fact
    extraction with a native backend without changing check logic.
    """

    dirs: tuple[str | Path, ...]
    pattern: str
    require_all: tuple[str, ...] = ()
    require_any: tuple[str, ...] = ()
    require_all_casefold: tuple[str, ...] = ()
    require_any_casefold: tuple[str, ...] = ()
    exclude_any: tuple[str, ...] = ()


@dataclass
class SourceIndex:
    """Process-local source index for one repo root.

    ``SourceIndex`` is scoped by callers, usually through
    :func:`source_index_context`. It caches only read-only scanner inputs:
    recursive file lists, source text, and parsed Python ASTs. It deliberately
    does not watch for file mutations; create a fresh index for a fresh scan.
    """

    root: Path
    skip_parts: frozenset[str] = frozenset({"__pycache__", "comma_lab_public_export"})
    _file_cache: dict[tuple[tuple[str, ...], str], tuple[Path, ...]] = field(default_factory=dict, init=False)
    _files_by_pattern_cache: dict[
        tuple[tuple[str, ...], tuple[str, ...]], dict[str, tuple[Path, ...]]
    ] = field(default_factory=dict, init=False)
    _facts_group_cache: dict[tuple[tuple[str, ...], str], tuple[SourceTextFacts, ...]] = field(
        default_factory=dict,
        init=False,
    )
    _text_cache: dict[tuple[str, str | None, str | None], str] = field(default_factory=dict, init=False)
    _ast_cache: dict[str, ast.AST] = field(default_factory=dict, init=False)
    _text_facts_cache: dict[str, SourceTextFacts] = field(default_factory=dict, init=False)
    _substring_index: dict[tuple[tuple[tuple[str, ...], str], str], frozenset[Path]] = field(
        default_factory=dict,
        init=False,
    )
    _substring_index_groups: set[tuple[tuple[str, ...], str]] = field(default_factory=set, init=False)
    _scanner_query_cache: dict[str, tuple[SourceTextFacts, ...]] = field(default_factory=dict, init=False)
    _persistent_text_facts: dict[str, dict[str, object]] = field(default_factory=dict, init=False)
    _persistent_text_facts_dirty: bool = field(default=False, init=False)
    _persistent_text_facts_enabled: bool = field(default=True, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _key_locks: dict[tuple[object, ...], threading.Lock] = field(default_factory=dict, init=False, repr=False)
    _stats: dict[str, int] = field(
        default_factory=lambda: {
            "file_list_hits": 0,
            "file_list_misses": 0,
            "files_by_pattern_hits": 0,
            "files_by_pattern_misses": 0,
            "facts_group_hits": 0,
            "facts_group_misses": 0,
            "text_hits": 0,
            "text_misses": 0,
            "ast_hits": 0,
            "ast_misses": 0,
            "text_facts_hits": 0,
            "text_facts_misses": 0,
            "text_facts_persistent_hits": 0,
            "substring_index_hits": 0,
            "substring_index_misses": 0,
            "scanner_query_hits": 0,
            "scanner_query_misses": 0,
        },
        init=False,
    )

    def __post_init__(self) -> None:
        self.root = _safe_resolve(Path(self.root))
        self._persistent_text_facts_enabled = _persistent_text_facts_enabled()
        if self._persistent_text_facts_enabled:
            self._persistent_text_facts = self._load_persistent_text_facts()

    def files(self, dirs: Sequence[str | Path], *, pattern: str) -> tuple[Path, ...]:
        """Return sorted files under ``dirs`` matching ``pattern``.

        Relative directories are resolved under ``root``. Paths containing any
        configured ``skip_parts`` are excluded, matching the current preflight
        OSS-mirror and ``__pycache__`` policy.
        """

        dir_keys = tuple(self._dir_key(item) for item in dirs)
        key = (dir_keys, pattern)
        with self._lock:
            cached = self._file_cache.get(key)
            if cached is not None:
                self._stats["file_list_hits"] += 1
                return cached

        with self._key_lock("files", key):
            with self._lock:
                cached = self._file_cache.get(key)
                if cached is not None:
                    self._stats["file_list_hits"] += 1
                    return cached
                self._stats["file_list_misses"] += 1

            paths = self._files_via_rg(dirs, pattern)
            if paths is None:
                paths = self._files_via_os_walk(dirs, pattern)
            out = tuple(paths[key] for key in sorted(paths))
            with self._lock:
                self._file_cache[key] = out
            return out

    def files_by_pattern(
        self,
        dirs: Sequence[str | Path],
        *,
        patterns: Sequence[str],
    ) -> dict[str, tuple[Path, ...]]:
        """Return sorted matching files for multiple filename patterns.

        This is the source-index contract for cold single-pass inventory work:
        callers can request related suffix groups such as ``("*.py", "*.sh")``
        without paying one recursive walk per pattern. Results are grouped by
        the requested pattern and are intentionally byte-for-byte comparable to
        calling :meth:`files` for each pattern independently.
        """

        unique_patterns = tuple(dict.fromkeys(patterns))
        if not unique_patterns:
            return {}
        dir_keys = tuple(self._dir_key(item) for item in dirs)
        key = (dir_keys, unique_patterns)
        with self._lock:
            cached = self._files_by_pattern_cache.get(key)
            if cached is not None:
                self._stats["files_by_pattern_hits"] += 1
                return dict(cached)

        with self._key_lock("files_by_pattern", key):
            with self._lock:
                cached = self._files_by_pattern_cache.get(key)
                if cached is not None:
                    self._stats["files_by_pattern_hits"] += 1
                    return dict(cached)
                self._stats["files_by_pattern_misses"] += 1

            paths = self._files_via_rg_many(dirs, unique_patterns)
            if paths is None:
                paths = self._files_via_os_walk_many(dirs, unique_patterns)
            grouped: dict[str, list[Path]] = {pattern: [] for pattern in unique_patterns}
            for path in (paths[key] for key in sorted(paths)):
                for pattern in unique_patterns:
                    if fnmatch.fnmatch(path.name, pattern):
                        grouped[pattern].append(path)
            out = {pattern: tuple(grouped[pattern]) for pattern in unique_patterns}
            with self._lock:
                self._files_by_pattern_cache[key] = out
                for pattern, paths_for_pattern in out.items():
                    self._file_cache[(dir_keys, pattern)] = paths_for_pattern
            return dict(out)

    def read_text(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        """Read and cache source text for ``path``."""

        target = Path(path)
        key = (_path_key(target), encoding, errors)
        with self._lock:
            cached = self._text_cache.get(key)
            if cached is not None:
                self._stats["text_hits"] += 1
                return cached

        with self._key_lock("text", key):
            with self._lock:
                cached = self._text_cache.get(key)
                if cached is not None:
                    self._stats["text_hits"] += 1
                    return cached
                self._stats["text_misses"] += 1

            text = target.read_text(encoding=encoding, errors=errors)
            with self._lock:
                self._text_cache[key] = text
            return text

    def python_ast(self, path: str | Path) -> ast.AST:
        """Parse and cache a Python AST for ``path``."""

        target = Path(path)
        key = _path_key(target)
        with self._lock:
            cached = self._ast_cache.get(key)
            if cached is not None:
                self._stats["ast_hits"] += 1
                return cached

        with self._key_lock("ast", key):
            with self._lock:
                cached = self._ast_cache.get(key)
                if cached is not None:
                    self._stats["ast_hits"] += 1
                    return cached
                self._stats["ast_misses"] += 1

            tree = ast.parse(self.read_text(target), filename=str(target))
            with self._lock:
                self._ast_cache[key] = tree
            return tree

    def text_facts(self, path: str | Path) -> SourceTextFacts:
        """Return cached one-pass lexical facts for ``path``."""

        target = Path(path)
        key = _path_key(target)
        with self._lock:
            cached = self._text_facts_cache.get(key)
            if cached is not None:
                self._stats["text_facts_hits"] += 1
                return cached

        with self._key_lock("text_facts", key):
            with self._lock:
                cached = self._text_facts_cache.get(key)
                if cached is not None:
                    self._stats["text_facts_hits"] += 1
                    return cached
                self._stats["text_facts_misses"] += 1

            stat = target.stat()
            persistent = (
                self._persistent_text_facts.get(key)
                if self._persistent_text_facts_enabled
                else None
            )
            if self._persistent_row_matches(target, stat, persistent):
                facts = self._facts_from_persistent_row(target, stat, persistent)
                with self._lock:
                    self._text_facts_cache[key] = facts
                    self._stats["text_facts_persistent_hits"] += 1
                return facts

            text = self.read_text(target)
            facts = SourceTextFacts(
                path=target,
                rel_path=self.repo_relative(target),
                suffix=target.suffix,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,
                inode=stat.st_ino,
                device=stat.st_dev,
                line_count=_source_line_count(text),
                tokens=frozenset(),
                substrings=frozenset(
                    needle for needle in _DEFAULT_TEXT_FACT_NEEDLES if needle in text
                ),
                casefold_substrings=_casefold_substrings_for_text(text),
            )
            with self._lock:
                self._text_facts_cache[key] = facts
                if self._persistent_text_facts_enabled:
                    self._persistent_text_facts[key] = self._facts_to_persistent_row(facts)
                    self._persistent_text_facts_dirty = True
            return facts

    def save_persistent_text_facts(self) -> None:
        """Persist valid text facts for future preflight processes."""

        if not self._persistent_text_facts_enabled:
            return
        with self._lock:
            if not self._persistent_text_facts_dirty:
                return
            rows = dict(self._persistent_text_facts)
            self._persistent_text_facts_dirty = False

        path = self._persistent_text_facts_path()
        payload = {
            "schema": _TEXT_FACTS_CACHE_SCHEMA,
            "rows": rows,
        }
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            with self._lock:
                self._persistent_text_facts_dirty = True
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def facts_for_files(
        self,
        dirs: Sequence[str | Path],
        *,
        pattern: str,
        parallel: bool = True,
    ) -> tuple[SourceTextFacts, ...]:
        """Return text facts for every matching file.

        File discovery still happens once through :meth:`files`. Fact
        extraction is parallelized across files by default because each file is
        independent and the hot path is I/O plus regex tokenization.
        """

        group_key = self._files_group_key(dirs, pattern)
        with self._lock:
            cached = self._facts_group_cache.get(group_key)
            if cached is not None:
                self._stats["facts_group_hits"] += 1
                return cached

        with self._key_lock("facts_group", group_key):
            with self._lock:
                cached = self._facts_group_cache.get(group_key)
                if cached is not None:
                    self._stats["facts_group_hits"] += 1
                    return cached
                self._stats["facts_group_misses"] += 1

            paths = self.files(dirs, pattern=pattern)
            if not paths:
                rows: tuple[SourceTextFacts, ...] = ()
            elif not parallel or len(paths) < 32:
                rows = tuple(self.text_facts(path) for path in paths)
            else:
                workers = min(_source_index_fact_workers(), len(paths))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    rows = tuple(pool.map(self.text_facts, paths))
            with self._lock:
                self._facts_group_cache[group_key] = rows
            return rows

    def files_containing_substrings(
        self,
        dirs: Sequence[str | Path],
        *,
        pattern: str,
        substrings: Sequence[str],
        require_all: bool = True,
    ) -> tuple[Path, ...]:
        """Return files whose text facts contain all/any requested substrings.

        This is the query shape most broad preflight scans want. Internally it
        builds an inverted substring index once per source-index context, then
        uses set intersection/union rather than re-looping over every facts row
        in every check. Needles outside the default fact set are still handled
        correctly by checking cached source text, so adding a new scanner cannot
        silently produce an empty candidate set.
        """

        group_key = self._files_group_key(dirs, pattern)
        if not substrings:
            return self.files(dirs, pattern=pattern)
        facts_rows = self.facts_for_files(dirs, pattern=pattern)
        self._ensure_substring_index_for_group(group_key, facts_rows)
        unknown_misses: list[str] = []
        candidate_sets: list[frozenset[Path]] = []
        for substring in substrings:
            index_key = (group_key, substring)
            with self._lock:
                cached = self._substring_index.get(index_key)
                if cached is not None:
                    self._stats["substring_index_hits"] += 1
                    candidate_sets.append(cached)
                    continue
                self._stats["substring_index_misses"] += 1
            if substring in _DEFAULT_TEXT_FACT_NEEDLES:
                rows = frozenset(
                    facts.path for facts in facts_rows if facts.contains(substring)
                )
            else:
                unknown_misses.append(substring)
                continue
            with self._lock:
                self._substring_index[index_key] = rows
            candidate_sets.append(rows)
        if unknown_misses:
            buckets: dict[str, set[Path]] = {substring: set() for substring in unknown_misses}
            for facts in facts_rows:
                text = self.read_text(facts.path)
                for substring in unknown_misses:
                    if substring in text:
                        buckets[substring].add(facts.path)
            with self._lock:
                for substring, paths in buckets.items():
                    rows = frozenset(paths)
                    self._substring_index[(group_key, substring)] = rows
                    candidate_sets.append(rows)
        if require_all:
            matched = set(candidate_sets[0])
            for rows in candidate_sets[1:]:
                matched.intersection_update(rows)
        else:
            matched = set()
            for rows in candidate_sets:
                matched.update(rows)
        return tuple(sorted(matched, key=lambda item: item.as_posix()))

    def query_text_facts(self, query: ScannerQuery) -> tuple[SourceTextFacts, ...]:
        """Return facts for files matching a declarative scanner query."""

        cache_key = self.scanner_query_fingerprint(query)
        with self._lock:
            cached = self._scanner_query_cache.get(cache_key)
            if cached is not None:
                self._stats["scanner_query_hits"] += 1
                return cached

        with self._key_lock("scanner_query", cache_key):
            with self._lock:
                cached = self._scanner_query_cache.get(cache_key)
                if cached is not None:
                    self._stats["scanner_query_hits"] += 1
                    return cached
                self._stats["scanner_query_misses"] += 1

            facts_rows = self.facts_for_files(query.dirs, pattern=query.pattern)
            matched = {facts.path for facts in facts_rows}
            if query.require_all:
                matched.intersection_update(
                    self.files_containing_substrings(
                        query.dirs,
                        pattern=query.pattern,
                        substrings=query.require_all,
                        require_all=True,
                    )
                )
            if query.require_any:
                matched.intersection_update(
                    self.files_containing_substrings(
                        query.dirs,
                        pattern=query.pattern,
                        substrings=query.require_any,
                        require_all=False,
                    )
                )
            if query.require_all_casefold:
                matched.intersection_update(
                    self.files_containing_casefold_substrings(
                        query.dirs,
                        pattern=query.pattern,
                        substrings=query.require_all_casefold,
                        require_all=True,
                    )
                )
            if query.require_any_casefold:
                matched.intersection_update(
                    self.files_containing_casefold_substrings(
                        query.dirs,
                        pattern=query.pattern,
                        substrings=query.require_any_casefold,
                        require_all=False,
                    )
                )
            if query.exclude_any:
                matched.difference_update(
                    self.files_containing_substrings(
                        query.dirs,
                        pattern=query.pattern,
                        substrings=query.exclude_any,
                        require_all=False,
                    )
                )
            out = tuple(facts for facts in facts_rows if facts.path in matched)
            with self._lock:
                self._scanner_query_cache[cache_key] = out
            return out

    def query_files(self, query: ScannerQuery) -> tuple[Path, ...]:
        """Return paths for files matching a declarative scanner query."""

        return tuple(facts.path for facts in self.query_text_facts(query))

    def scanner_query_fingerprint(self, query: ScannerQuery) -> str:
        """Return a stable fingerprint for query semantics within this repo."""

        def _stable_unique(values: Sequence[str]) -> list[str]:
            return sorted(dict.fromkeys(str(value) for value in values))

        payload = {
            "schema": "pact.source_index.scanner_query.v1",
            "root": self.root.as_posix(),
            "dirs": sorted(dict.fromkeys(self._dir_key(item) for item in query.dirs)),
            "pattern": query.pattern,
            "require_all": _stable_unique(query.require_all),
            "require_any": _stable_unique(query.require_any),
            "require_all_casefold": _stable_unique(
                item.casefold() for item in query.require_all_casefold
            ),
            "require_any_casefold": _stable_unique(
                item.casefold() for item in query.require_any_casefold
            ),
            "exclude_any": _stable_unique(query.exclude_any),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def files_containing_casefold_substrings(
        self,
        dirs: Sequence[str | Path],
        *,
        pattern: str,
        substrings: Sequence[str],
        require_all: bool = True,
    ) -> tuple[Path, ...]:
        """Return files whose text contains all/any substrings ignoring case."""

        if not substrings:
            return self.files(dirs, pattern=pattern)
        facts_rows = self.facts_for_files(dirs, pattern=pattern)
        buckets: dict[str, set[Path]] = {substring.casefold(): set() for substring in substrings}
        unknown: list[str] = []
        known_needles = {
            needle.casefold() for needle in _DEFAULT_CASEFOLD_TEXT_FACT_NEEDLES
        }
        for substring in substrings:
            folded = substring.casefold()
            if folded in known_needles:
                for facts in facts_rows:
                    if facts.contains_casefold(substring):
                        buckets[folded].add(facts.path)
            else:
                unknown.append(substring)
        if unknown:
            for facts in facts_rows:
                folded_text = self.read_text(facts.path).casefold()
                for substring in unknown:
                    if substring.casefold() in folded_text:
                        buckets[substring.casefold()].add(facts.path)
        candidate_sets = [frozenset(buckets[substring.casefold()]) for substring in substrings]
        if require_all:
            matched = set(candidate_sets[0])
            for rows in candidate_sets[1:]:
                matched.intersection_update(rows)
        else:
            matched = set()
            for rows in candidate_sets:
                matched.update(rows)
        return tuple(sorted(matched, key=lambda item: item.as_posix()))

    def repo_relative(self, path: str | Path) -> str:
        """Return a stable POSIX path relative to this index root when possible."""

        target = _safe_resolve(Path(path))
        try:
            return target.relative_to(self.root).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss counters and current cache sizes."""

        return {
            **self._stats,
            "file_list_cache_entries": len(self._file_cache),
            "files_by_pattern_cache_entries": len(self._files_by_pattern_cache),
            "facts_group_cache_entries": len(self._facts_group_cache),
            "text_cache_entries": len(self._text_cache),
            "ast_cache_entries": len(self._ast_cache),
            "text_facts_cache_entries": len(self._text_facts_cache),
            "substring_index_entries": len(self._substring_index),
            "scanner_query_cache_entries": len(self._scanner_query_cache),
            "persistent_text_facts_entries": len(self._persistent_text_facts),
            "persistent_text_facts_enabled": int(self._persistent_text_facts_enabled),
        }

    def _dir_key(self, item: str | Path) -> str:
        path = Path(item)
        if path.is_absolute():
            resolved = _safe_resolve(path)
            try:
                return resolved.relative_to(self.root).as_posix().rstrip("/")
            except ValueError:
                return resolved.as_posix().rstrip("/")
        return path.as_posix().rstrip("/")

    def _files_group_key(
        self,
        dirs: Sequence[str | Path],
        pattern: str,
    ) -> tuple[tuple[str, ...], str]:
        return tuple(self._dir_key(item) for item in dirs), pattern

    def _resolve_dir(self, item: str | Path) -> Path:
        path = Path(item)
        if path.is_absolute():
            return path
        return self.root / path

    def _files_via_rg(
        self,
        dirs: Sequence[str | Path],
        pattern: str,
    ) -> dict[str, Path] | None:
        """Return matching files through ripgrep's parallel walker when safe."""

        roots: list[str] = []
        for item in dirs:
            path = Path(item)
            if path.is_absolute():
                resolved = _safe_resolve(path)
                try:
                    rel = resolved.relative_to(self.root)
                except ValueError:
                    return None
                if resolved.exists():
                    roots.append(rel.as_posix())
                continue
            resolved = self.root / path
            if resolved.exists():
                roots.append(path.as_posix())
        if not roots:
            return {}

        cmd = ["rg", "--files", "-g", pattern]
        for part in sorted(self.skip_parts):
            cmd.extend(["-g", f"!**/{part}/**"])
        cmd.extend(["-g", "!experiments/results/**"])
        cmd.extend(roots)
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode not in (0, 1):
            return None

        paths: dict[str, Path] = {}
        for raw_line in proc.stdout.splitlines():
            if not raw_line:
                continue
            path = self.root / raw_line
            if self._should_skip(path):
                continue
            if not path.is_file():
                continue
            paths[_path_key(path)] = path
        return paths

    def _files_via_rg_many(
        self,
        dirs: Sequence[str | Path],
        patterns: Sequence[str],
    ) -> dict[str, Path] | None:
        """Return files matching any pattern through one ripgrep invocation."""

        roots: list[str] = []
        for item in dirs:
            path = Path(item)
            if path.is_absolute():
                resolved = _safe_resolve(path)
                try:
                    rel = resolved.relative_to(self.root)
                except ValueError:
                    return None
                if resolved.exists():
                    roots.append(rel.as_posix())
                continue
            resolved = self.root / path
            if resolved.exists():
                roots.append(path.as_posix())
        if not roots:
            return {}

        cmd = ["rg", "--files"]
        for pattern in patterns:
            cmd.extend(["-g", pattern])
        for part in sorted(self.skip_parts):
            cmd.extend(["-g", f"!**/{part}/**"])
        cmd.extend(["-g", "!experiments/results/**"])
        cmd.extend(roots)
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode not in (0, 1):
            return None

        paths: dict[str, Path] = {}
        for raw_line in proc.stdout.splitlines():
            if not raw_line:
                continue
            path = self.root / raw_line
            if self._should_skip(path):
                continue
            if not path.is_file():
                continue
            paths[_path_key(path)] = path
        return paths

    def _files_via_os_walk(
        self,
        dirs: Sequence[str | Path],
        pattern: str,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for item in dirs:
            base = self._resolve_dir(item)
            if not base.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(
                    name
                    for name in dirnames
                    if name not in self.skip_parts
                )
                current = Path(dirpath)
                try:
                    rel_current = _safe_resolve(current).relative_to(self.root)
                except ValueError:
                    rel_current = current
                if rel_current.parts[:2] == ("experiments", "results"):
                    dirnames[:] = []
                    continue
                if rel_current.parts == ("experiments",):
                    dirnames[:] = [name for name in dirnames if name != "results"]
                for filename in sorted(filenames):
                    if not fnmatch.fnmatch(filename, pattern):
                        continue
                    path = current / filename
                    if self._should_skip(path):
                        continue
                    if not path.is_file():
                        continue
                    paths[_path_key(path)] = path
        return paths

    def _files_via_os_walk_many(
        self,
        dirs: Sequence[str | Path],
        patterns: Sequence[str],
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for item in dirs:
            base = self._resolve_dir(item)
            if not base.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(
                    name
                    for name in dirnames
                    if name not in self.skip_parts
                )
                current = Path(dirpath)
                try:
                    rel_current = _safe_resolve(current).relative_to(self.root)
                except ValueError:
                    rel_current = current
                if rel_current.parts[:2] == ("experiments", "results"):
                    dirnames[:] = []
                    continue
                if rel_current.parts == ("experiments",):
                    dirnames[:] = [name for name in dirnames if name != "results"]
                for filename in sorted(filenames):
                    if not any(fnmatch.fnmatch(filename, pattern) for pattern in patterns):
                        continue
                    path = current / filename
                    if self._should_skip(path):
                        continue
                    if not path.is_file():
                        continue
                    paths[_path_key(path)] = path
        return paths

    def _should_skip(self, path: Path) -> bool:
        parts = set(path.parts)
        if any(part in parts for part in self.skip_parts):
            return True
        try:
            rel = _safe_resolve(path).relative_to(self.root)
        except ValueError:
            rel = path
        return rel.parts[:2] == ("experiments", "results")

    def _ensure_substring_index_for_group(
        self,
        group_key: tuple[tuple[str, ...], str],
        facts_rows: tuple[SourceTextFacts, ...],
    ) -> None:
        with self._lock:
            if group_key in self._substring_index_groups:
                return

        with self._key_lock("substring_group", group_key):
            with self._lock:
                if group_key in self._substring_index_groups:
                    return
            buckets: dict[str, set[Path]] = {
                needle: set() for needle in _DEFAULT_TEXT_FACT_NEEDLES
            }
            for facts in facts_rows:
                for substring in facts.substrings:
                    bucket = buckets.get(substring)
                    if bucket is not None:
                        bucket.add(facts.path)
            with self._lock:
                for substring, paths in buckets.items():
                    self._substring_index[(group_key, substring)] = frozenset(paths)
                self._substring_index_groups.add(group_key)

    def _key_lock(self, *parts: object) -> threading.Lock:
        with self._lock:
            lock = self._key_locks.get(parts)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[parts] = lock
            return lock

    def _persistent_text_facts_path(self) -> Path:
        return self.root / ".omx" / "cache" / "source_text_facts.json"

    def _load_persistent_text_facts(self) -> dict[str, dict[str, object]]:
        try:
            payload = json.loads(
                self._persistent_text_facts_path().read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if payload.get("schema") != _TEXT_FACTS_CACHE_SCHEMA:
            return {}
        rows = payload.get("rows")
        if not isinstance(rows, dict):
            return {}
        return {str(key): value for key, value in rows.items() if isinstance(value, dict)}

    def _persistent_row_matches(
        self,
        target: Path,
        stat: os.stat_result,
        row: dict[str, object] | None,
    ) -> bool:
        if not isinstance(row, dict):
            return False
        return (
            row.get("path") == _path_key(target)
            and row.get("rel_path") == self.repo_relative(target)
            and row.get("suffix") == target.suffix
            and row.get("size_bytes") == stat.st_size
            and row.get("mtime_ns") == stat.st_mtime_ns
            and row.get("ctime_ns") == stat.st_ctime_ns
            and row.get("inode") == stat.st_ino
            and row.get("device") == stat.st_dev
            and isinstance(row.get("line_count"), int)
            and isinstance(row.get("substrings"), list)
        )

    def _facts_from_persistent_row(
        self,
        target: Path,
        stat: os.stat_result,
        row: dict[str, object],
    ) -> SourceTextFacts:
        substrings = row.get("substrings")
        tokens = row.get("tokens", [])
        return SourceTextFacts(
            path=target,
            rel_path=str(row["rel_path"]),
            suffix=str(row["suffix"]),
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
            inode=int(stat.st_ino),
            device=int(stat.st_dev),
            line_count=int(row["line_count"]),
            tokens=frozenset(str(item) for item in tokens if isinstance(item, str)),
            substrings=frozenset(
                str(item) for item in substrings if isinstance(item, str)
            ),
            casefold_substrings=frozenset(
                str(item).casefold()
                for item in row.get("casefold_substrings", substrings)
                if isinstance(item, str)
            ),
        )

    def _facts_to_persistent_row(self, facts: SourceTextFacts) -> dict[str, object]:
        return {
            "path": _path_key(facts.path),
            "rel_path": facts.rel_path,
            "suffix": facts.suffix,
            "size_bytes": facts.size_bytes,
            "mtime_ns": facts.mtime_ns,
            "ctime_ns": facts.ctime_ns,
            "inode": facts.inode,
            "device": facts.device,
            "line_count": facts.line_count,
            "tokens": sorted(facts.tokens),
            "substrings": sorted(facts.substrings),
            "casefold_substrings": sorted(facts.casefold_substrings),
        }


def get_current_source_index(repo_root: str | Path | None = None) -> SourceIndex | None:
    """Return the active source index when it matches ``repo_root``."""

    index = _CURRENT_SOURCE_INDEX.get()
    if index is None or repo_root is None:
        return index
    if index.root != _safe_resolve(Path(repo_root)):
        return None
    return index


@contextlib.contextmanager
def source_index_context(repo_root: str | Path) -> Iterator[SourceIndex]:
    """Install a shared source index for scanners in this context."""

    existing = get_current_source_index(repo_root)
    if existing is not None:
        yield existing
        return

    index = SourceIndex(Path(repo_root))
    token = _CURRENT_SOURCE_INDEX.set(index)
    try:
        yield index
    finally:
        index.save_persistent_text_facts()
        _CURRENT_SOURCE_INDEX.reset(token)


__all__ = [
    "ScannerQuery",
    "SourceIndex",
    "SourceTextFacts",
    "get_current_source_index",
    "source_index_context",
]
