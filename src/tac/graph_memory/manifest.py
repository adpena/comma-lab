# SPDX-License-Identifier: MIT
"""Source / completeness manifest for the graph-memory build (task #569, P0-1).

Why this exists (the 07-19 silent-collapse incident):
    A fresh ``graph_memory_recall.py --stats`` in a *linked git worktree*
    produced 3,157 nodes / 4,856 edges while the canonical-root cache held
    9,704 / 32,156 — the memory source silently resolved to a nonexistent
    worktree slug and ``parse_memory_files`` returned 0 with NO error. The
    cache publisher had no source-count manifest or collapse refusal, so a
    whole required source vanishing looked identical to a clean build.

    (The root-cause path bug is already fixed in ``build._canonical_repo_root``;
    this manifest is the SELF-PROTECT layer so ANY future required-source
    collapse — a different cause — becomes a loud diff instead of silence.)

What it does:
    * ``build_source_manifest`` probes every corpus source (files present on
      disk, bytes, latest mtime) and attributes graph nodes back to each
      source, so "scanned but emitted zero nodes" is visible per source.
    * ``check_completeness`` compares two consecutive manifests and warns
      (warn-only) when node/edge counts drop >20% or a previously-nonzero
      REQUIRED source collapses to zero / disappears.
    * ``publish_and_check`` is the wire-in the graph builder calls on every
      (re)build: append the manifest and emit any warnings to stderr. It is
      best-effort — it never breaks recall.

Deterministic + rebuildable: the manifest is derived from the same corpus the
graph indexes; the markdown/JSONL stores remain the source of truth.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .build import (
    _DAG_GLOB,
    _DEFERRAL_MD,
    _EQUATIONS_JSONL,
    _ORGAN_LEDGER,
    _RESEARCH_DIR,
    _TASKS_JSONL,
    REPO_ROOT,
    _canonical_repo_root,
    _memory_dir,
)
from .model import Graph

# Bump when the manifest schema or attribution logic changes materially.
MANIFEST_PARSER_VERSION = "1"

# Fraction drop in node/edge count (vs the previous manifest) that trips a
# warn-only completeness warning. 0.20 == 20% per the #569 P0-1 spec.
DEFAULT_DROP_THRESHOLD = 0.20

# Which sources are structurally REQUIRED (a collapse-to-zero is a bug) vs.
# legitimately-optional (absent in a fresh worktree is normal). The 07-19
# collapse was `memory` (required) -> this classification catches it.
_REQUIRED = frozenset({"memory", "dag", "equations"})


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Health of ONE corpus source at build time.

    ``units_present`` is the raw count scanned on disk (``*.md`` files for a
    directory source, JSONL lines for a registry, ``1`` for a single file);
    ``nodes_emitted`` is how many graph nodes were attributed back to this
    source. A large gap (or ``units_present > 0`` with ``nodes_emitted == 0``)
    is the silent-collapse signature.
    """

    name: str
    canonical_root: str
    required: bool
    exists: bool
    units_present: int
    bytes_present: int
    latest_source_mtime: float
    nodes_emitted: int
    parser_version: str = MANIFEST_PARSER_VERSION
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SourceSnapshot":
        return SourceSnapshot(
            name=d["name"],
            canonical_root=d.get("canonical_root", ""),
            required=bool(d.get("required", False)),
            exists=bool(d.get("exists", False)),
            units_present=int(d.get("units_present", 0)),
            bytes_present=int(d.get("bytes_present", 0)),
            latest_source_mtime=float(d.get("latest_source_mtime", 0.0)),
            nodes_emitted=int(d.get("nodes_emitted", 0)),
            parser_version=str(d.get("parser_version", MANIFEST_PARSER_VERSION)),
            skip_reason=str(d.get("skip_reason", "")),
        )


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """One published build's source-health record."""

    generated_at: str
    git_rev: str
    repo_root: str
    canonical_root: str
    node_count: int
    edge_count: int
    sources: tuple[SourceSnapshot, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sources"] = [s.to_dict() for s in self.sources]
        return d

    @staticmethod
    def from_dict(d: dict) -> "SourceManifest":
        return SourceManifest(
            generated_at=d.get("generated_at", ""),
            git_rev=d.get("git_rev", ""),
            repo_root=d.get("repo_root", ""),
            canonical_root=d.get("canonical_root", ""),
            node_count=int(d.get("node_count", 0)),
            edge_count=int(d.get("edge_count", 0)),
            sources=tuple(
                SourceSnapshot.from_dict(s) for s in d.get("sources", [])
            ),
        )

    def source(self, name: str) -> SourceSnapshot | None:
        for s in self.sources:
            if s.name == name:
                return s
        return None


@dataclass(frozen=True, slots=True)
class CompletenessWarning:
    """A warn-only completeness finding comparing two manifests."""

    kind: str  # node_count_drop | edge_count_drop | required_source_collapsed | required_source_absent
    source: str  # "" for graph-level, else the source name
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────── git / clock ───────────────────────────


def _git_rev(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return out
    except (OSError, subprocess.SubprocessError):
        return ""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────── node attribution ───────────────────────────


def _strip_line_anchor(source: str) -> str:
    """A node source may be ``path#L12-L34`` — return the bare path."""
    return source.split("#", 1)[0]


def _attribute_nodes(graph: Graph, roots: dict[str, Path]) -> dict[str, int]:
    """Count graph nodes whose ``source`` path belongs to each configured root.

    Directory roots match by prefix (memory dir); file roots match by equality
    on the bare (line-anchor-stripped) path. Shared/stub nodes with an empty
    source (entities, people, topics) are intentionally not attributed to a
    source — they are cross-cutting, and the collapse signal we care about is a
    REQUIRED authoritative source emitting zero.
    """
    counts: dict[str, int] = {name: 0 for name in roots}
    # Precompute normalized root strings once.
    dir_roots = {n: str(r) for n, r in roots.items() if r.is_dir()}
    file_roots = {n: str(r) for n, r in roots.items() if not r.is_dir()}
    for node in graph.nodes.values():
        src = node.source
        if not src:
            continue
        bare = _strip_line_anchor(src)
        # File roots first (exact match wins over a coincidental dir prefix).
        matched = False
        for name, root_str in file_roots.items():
            if bare == root_str:
                counts[name] += 1
                matched = True
                break
        if matched:
            continue
        for name, root_str in dir_roots.items():
            if bare.startswith(root_str):
                counts[name] += 1
                break
    return counts


def _resolve_dag_path(dag_path: Path | None) -> Path | None:
    if dag_path is not None:
        return dag_path
    dag_files = sorted(_RESEARCH_DIR.glob(_DAG_GLOB))
    return dag_files[-1] if dag_files else None


def _probe_source(
    name: str, root: Path | None, *, required: bool
) -> tuple[Path, int, int, float, bool, str]:
    """Return (canonical_root, units_present, bytes_present, latest_mtime, exists, skip_reason)."""
    if root is None:
        return Path(""), 0, 0, 0.0, False, "root_unresolved"
    if root.is_dir():
        md_files = [p for p in sorted(root.glob("*.md")) if p.name != "MEMORY.md"]
        units = len(md_files)
        total = 0
        latest = 0.0
        for p in md_files:
            try:
                st = p.stat()
                total += st.st_size
                latest = max(latest, st.st_mtime)
            except OSError:
                continue
        return root, units, total, latest, True, ("" if units else "dir_empty")
    if not root.is_file():
        return root, 0, 0, 0.0, False, "file_absent"
    try:
        st = root.stat()
    except OSError:
        return root, 0, 0, 0.0, False, "stat_failed"
    if root.suffix == ".jsonl":
        try:
            n_lines = sum(
                1 for ln in root.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()
            )
        except OSError:
            n_lines = 0
        return root, n_lines, st.st_size, st.st_mtime, True, ("" if n_lines else "file_empty")
    return root, 1, st.st_size, st.st_mtime, True, ""


# ─────────────────────────── manifest build ───────────────────────────


def build_source_manifest(
    graph: Graph,
    *,
    memory_dir: Path | None = None,
    dag_path: Path | None = None,
    equations_path: Path | None = None,
    tasks_path: Path | None = None,
    deferral_path: Path | None = None,
    organ_path: Path | None = None,
) -> SourceManifest:
    """Probe every corpus source and attribute ``graph`` nodes back to it."""
    roots: dict[str, Path | None] = {
        "memory": memory_dir or _memory_dir(),
        "dag": _resolve_dag_path(dag_path),
        "equations": equations_path or _EQUATIONS_JSONL,
        "tasks": tasks_path or _TASKS_JSONL,
        "deferral": deferral_path or _DEFERRAL_MD,
        "costate_organ": organ_path or _ORGAN_LEDGER,
    }
    # Attribution only needs real (present) roots.
    attrib_roots = {n: r for n, r in roots.items() if r is not None}
    node_counts = _attribute_nodes(graph, attrib_roots)

    snapshots: list[SourceSnapshot] = []
    for name, root in roots.items():
        required = name in _REQUIRED
        canon, units, nbytes, mtime, exists, skip = _probe_source(
            name, root, required=required
        )
        snapshots.append(SourceSnapshot(
            name=name,
            canonical_root=str(canon),
            required=required,
            exists=exists,
            units_present=units,
            bytes_present=nbytes,
            latest_source_mtime=mtime,
            nodes_emitted=node_counts.get(name, 0),
            skip_reason=skip,
        ))

    return SourceManifest(
        generated_at=_now_iso(),
        git_rev=_git_rev(REPO_ROOT),
        repo_root=str(REPO_ROOT),
        canonical_root=str(_canonical_repo_root()),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        sources=tuple(snapshots),
    )


# ─────────────────────────── completeness check ───────────────────────────


def check_completeness(
    prev: SourceManifest | None,
    curr: SourceManifest,
    *,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> list[CompletenessWarning]:
    """Warn-only diff of two consecutive manifests.

    Fires when:
      * total node OR edge count dropped by more than ``drop_threshold``;
      * a REQUIRED source that previously emitted >0 nodes now emits 0;
      * a REQUIRED source that previously existed is now absent.

    With no ``prev`` (first ever build) there is nothing to compare — returns [].
    """
    warnings: list[CompletenessWarning] = []
    if prev is None:
        return warnings

    def _drop(old: int, new: int) -> float:
        if old <= 0:
            return 0.0
        return (old - new) / old

    node_drop = _drop(prev.node_count, curr.node_count)
    if node_drop > drop_threshold:
        warnings.append(CompletenessWarning(
            kind="node_count_drop", source="",
            detail=(f"node count {prev.node_count} -> {curr.node_count} "
                    f"({node_drop:.0%} drop > {drop_threshold:.0%})"),
        ))
    edge_drop = _drop(prev.edge_count, curr.edge_count)
    if edge_drop > drop_threshold:
        warnings.append(CompletenessWarning(
            kind="edge_count_drop", source="",
            detail=(f"edge count {prev.edge_count} -> {curr.edge_count} "
                    f"({edge_drop:.0%} drop > {drop_threshold:.0%})"),
        ))

    for cs in curr.sources:
        ps = prev.source(cs.name)
        if ps is None:
            continue
        if cs.required and ps.nodes_emitted > 0 and cs.nodes_emitted == 0:
            warnings.append(CompletenessWarning(
                kind="required_source_collapsed", source=cs.name,
                detail=(f"required source '{cs.name}' emitted {ps.nodes_emitted} "
                        f"nodes previously, now 0 (root={cs.canonical_root}, "
                        f"units_present={cs.units_present}, skip={cs.skip_reason or 'none'})"),
            ))
        elif cs.required and ps.exists and not cs.exists:
            warnings.append(CompletenessWarning(
                kind="required_source_absent", source=cs.name,
                detail=(f"required source '{cs.name}' existed previously, now absent "
                        f"(root={cs.canonical_root}, skip={cs.skip_reason or 'none'})"),
            ))
    return warnings


# ─────────────────────────── publish / load ───────────────────────────


def manifest_path() -> Path:
    return REPO_ROOT / ".omx" / "state" / "graph_memory" / "source_manifest.jsonl"


def load_latest_manifest(path: Path | None = None) -> SourceManifest | None:
    """Return the last published manifest (latest row of the append-only JSONL)."""
    p = path or manifest_path()
    if not p.is_file():
        return None
    last: dict | None = None
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    except OSError:
        return None
    return SourceManifest.from_dict(last) if last is not None else None


def publish_manifest(manifest: SourceManifest, path: Path | None = None) -> Path:
    """Append ``manifest`` as one JSONL row (append-only history)."""
    p = path or manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    return p


def publish_and_check(
    graph: Graph,
    *,
    path: Path | None = None,
    warn: bool = True,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
    **build_kwargs,
) -> tuple[SourceManifest, list[CompletenessWarning]]:
    """Wire-in: build the manifest, compare to the last published one, append it.

    Warn-only: any completeness warning is written to stderr; nothing raises.
    The comparison uses the last row PRESENT BEFORE this build is appended.
    """
    p = path or manifest_path()
    prev = load_latest_manifest(p)
    curr = build_source_manifest(graph, **build_kwargs)
    findings = check_completeness(prev, curr, drop_threshold=drop_threshold)
    if warn and findings:
        for w in findings:
            print(
                f"[graph-memory completeness WARN] {w.kind}"
                + (f" [{w.source}]" if w.source else "")
                + f": {w.detail}",
                file=sys.stderr,
            )
    publish_manifest(curr, p)
    return curr, findings
