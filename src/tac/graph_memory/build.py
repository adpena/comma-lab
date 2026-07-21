# SPDX-License-Identifier: MIT
"""Build the graph-memory by parsing the real corpus (task #411).

Sources parsed (each is the SOURCE OF TRUTH; the graph indexes them):
  memory     ~/.claude/projects/<slug>/memory/*.md  — frontmatter + [[wikilinks]]
  dag        .omx/research/sub015_DAG_*.md           — ### / ## FEED-* blocks
  equations  .omx/state/canonical_equations_registry.jsonl — producers/consumers
  tasks      .omx/state/canonical_task_status.jsonl  — task subjects + blockers
  deferral   .omx/state/deferral_ledger.md           — D# rows + owners/triggers

Edges are SYNTHESIZED from the existing references (wikilinks, FEED->#task/eq/file,
producer/consumer, sister, supersedes) so the corpus becomes ONE navigable graph
without rewriting any FEED markdown. The wikilink markdown IS the Obsidian-
compatible substrate; this builder materializes the graph over it.

Deterministic: sorted iteration, content-derived ids, capped reads, no RNG.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from .model import Edge, Graph, Node

# ------------------------------------------------------------------ paths ----
REPO_ROOT = Path(__file__).resolve().parents[3]
_DAG_GLOB = "sub015_DAG_topaiml_reopen_and_pursuit_plan_*.md"
_RESEARCH_DIR = REPO_ROOT / ".omx" / "research"
_EQUATIONS_JSONL = REPO_ROOT / ".omx" / "state" / "canonical_equations_registry.jsonl"
_TASKS_JSONL = REPO_ROOT / ".omx" / "state" / "canonical_task_status.jsonl"
_DEFERRAL_MD = REPO_ROOT / ".omx" / "state" / "deferral_ledger.md"
_LANE_REGISTRY = REPO_ROOT / ".omx" / "state" / "lane_registry.json"
_DOCS_DIR = REPO_ROOT / "docs"
_DOCTRINE_ROOTS = (REPO_ROOT / "CLAUDE.md", REPO_ROOT / "AGENTS.md")

# Read caps: scoring/parsing on a bounded prefix is a deterministic heuristic
# (the DAG is ~1.7 MB; a memory description is small).
_BLOCK_SUMMARY_CAP = 600
_MEMORY_SUMMARY_CAP = 800

# ------------------------------------------------------------- extractors ----
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+\.md(?:#[^)\s]+)?)\)")
_PURE_NUMERIC_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
_NUMERIC_TUPLE_RE = re.compile(
    r"\(?\s*[+-]?\d+(?:\.\d+)?(?:\s*,\s*[+-]?\d+(?:\.\d+)?){1,}\s*\)?"
)
_COORDINATE_RE = re.compile(
    r"(?:[xyzhw]\s*[:=]\s*[+-]?\d+(?:\.\d+)?)(?:\s*,\s*"
    r"[xyzhw]\s*[:=]\s*[+-]?\d+(?:\.\d+)?)+",
    re.IGNORECASE,
)
# lever / task numeric refs: #247, #205 (2-4 digits, avoids color hex etc.)
_HASHREF_RE = re.compile(r"(?<![\w])#(\d{2,4})\b")
# file paths under our source trees
_FILE_RE = re.compile(
    r"(?<![\w./])((?:src/tac|tools|experiments|scripts|\.omx/research|docs)/[\w./-]+\.\w{1,4})"
)
# canonical equation ids look like snake_case ... _v<N>
_EQID_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+_v\d+)\b")
_FEED_REF_RE = re.compile(r"\bFEED-([A-Za-z0-9][\w-]*)")
_LANE_ID_RE = re.compile(r"\b(lane_[A-Za-z0-9][A-Za-z0-9_.-]*)\b")
_CATALOG_RE = re.compile(r"\bCatalog\s+#(\d{1,4})\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_VERDICT_RE = re.compile(
    r"\b(CONFIRMED|CONFIRM|NO-GO|NOGO|DEAD|KILL|FALSIFIED|VERDICT|DEFER(?:RED)?|"
    r"RESOLVED|CLOSED|DOMINATE[SD]?|SUPERSEDED|GO)\b"
)
# council / named people that recur as entities
_PEOPLE = (
    "Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Quantizr", "Hotz",
    "Selfcomp", "MacKay", "Ballé", "Balle", "Rudin", "Daubechies", "Boyd", "Tao",
    "Mallat", "Tishby", "Ballard", "Rao", "Atick", "Redlich", "Wyner", "Hinton",
    "Karpathy", "Schmidhuber", "Carmack", "Hassabis",
)
_PEOPLE_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in _PEOPLE) + r")\b")

_FEED_HDR_RE = re.compile(r"^#{2,3}\s+FEED-([A-Za-z0-9][\w-]*)")
_SISTER_RE = re.compile(r"[Ss]ister(?:\s+of)?:?\s*\[\[([^\]]+)\]\]")
_SUPERSEDE_RE = re.compile(r"[Ss]upersed(?:ed|es)?\s*(?:by)?:?\s*\[\[([^\]]+)\]\]")


def _clean(text: str, cap: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:cap]


def _canonical_repo_root() -> Path:
    """Resolve the CANONICAL repo root even from a linked git worktree.

    In a linked worktree ``REPO_ROOT`` is the worktree path, whose Claude
    memory slug does not exist -> ``parse_memory_files`` silently returned 0
    and the graph collapsed (MEASURED 9,704/32,156 -> 3,157/4,856 nodes/edges;
    the #566 crosswalk's P0 finding, 2026-07-19). Worktrees share the main
    repo's git common dir; its parent is the canonical root.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--path-format=absolute",
             "--git-common-dir"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        common = Path(out)
        if common.name == ".git":
            return common.parent
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return REPO_ROOT


def _memory_dir() -> Path:
    slug = str(_canonical_repo_root()).replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug / "memory"


def _mem_id(link_text: str) -> str:
    return "memory:" + link_text.strip()


def wikilink_target(raw: str) -> str | None:
    """Normalize one ``[[target|label]]`` and reject numeric false positives.

    Markdown uses double brackets for array/coordinate prose too.  Pure numbers,
    comma-separated numeric tuples, and named coordinate tuples are not graph
    entities and must never create dangling memory stubs.
    """
    target = raw.split("|", 1)[0].strip()
    if not target:
        return None
    if (
        _PURE_NUMERIC_RE.fullmatch(target)
        or _NUMERIC_TUPLE_RE.fullmatch(target)
        or _COORDINATE_RE.fullmatch(target)
    ):
        return None
    return target


def iter_wikilink_targets(text: str):
    """Yield normalized semantic wikilink targets from *text*."""
    for match in _WIKILINK_RE.finditer(text):
        target = wikilink_target(match.group(1))
        if target is not None:
            yield target


def _memory_node_id_for_path(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _mem_id(path.stem)
    name, _, _ = _parse_frontmatter(text)
    return _mem_id(name or path.stem)


def _is_memory_index(path: Path) -> bool:
    name = path.name.lower()
    return name == "memory.md" or (
        name.startswith("memory_")
        and ("cluster_" in name or "_cluster_" in name or "full_index_" in name)
    )


def _slugify_heading(text: str) -> str:
    text = re.sub(r"[`*_~]", "", text).strip().lower()
    text = re.sub(r"[^a-z0-9_ -]+", "", text)
    return re.sub(r"[-\s]+", "-", text).strip("-")


def _section_aliases(header: str, body: str) -> set[str]:
    aliases = {_slugify_heading(header)}
    for sep in (" — ", " – ", ": "):
        if sep in header:
            aliases.add(_slugify_heading(header.split(sep, 1)[0]))
    for line in body.splitlines():
        if not re.search(r"\bAnchors?\s*:", line, re.IGNORECASE):
            continue
        aliases.update(iter_wikilink_targets(line))
        aliases.update(
            token for token in re.findall(r"`([A-Za-z0-9][A-Za-z0-9_-]{2,})`", line)
            if "-" in token or "_" in token
        )
    return {a for a in aliases if a}


def _memory_file_aliases(path: Path) -> set[str]:
    """Concrete filename-derived aliases for a memory note (no fuzzy guessing)."""
    stem = path.stem
    aliases = {stem, stem.replace("_", "-")}
    for prefix in ("feedback_", "project_", "reference_"):
        if stem.startswith(prefix):
            short = stem.removeprefix(prefix)
            aliases.update({short, short.replace("_", "-")})
    return {alias for alias in aliases if alias}


# ------------------------------------------------------------- parse: memory -
def parse_memory_files(
    graph: Graph, memory_dir: Path | None = None, eq_ids: set[str] | None = None,
) -> int:
    """Parse memory/*.md files: node per file, [[wikilink]] + tag + sister edges."""
    eq_ids = eq_ids or set()
    mdir = memory_dir or _memory_dir()
    if not mdir.is_dir():
        return 0
    count = 0
    for path in sorted(mdir.glob("*.md")):
        if path.name == "MEMORY.md":
            continue  # the flat index — folded via topics below, not a node
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        name, mtype, desc = _parse_frontmatter(text)
        node_id = _mem_id(name) if name else "memory:" + path.stem
        graph.add_node(Node(
            id=node_id, ntype="memory",
            title=name or path.stem,
            summary=_clean(desc or "", _MEMORY_SUMMARY_CAP),
            source=str(path),
            attrs={"mtype": mtype or "memory"},
        ))
        for alias in _memory_file_aliases(path):
            alias_id = _mem_id(alias)
            if alias_id == node_id:
                continue
            graph.ensure_stub(alias_id, ntype="memory", title=alias)
            graph.add_edge(Edge(alias_id, node_id, "aliases", str(path)))
        count += 1
        # tag -> topic
        if mtype:
            topic_id = "topic:" + mtype
            graph.ensure_stub(topic_id, ntype="topic", title=mtype)
            graph.add_edge(Edge(node_id, topic_id, "tagged", str(path)))
        # verdict-carrying memory -> also a decision
        if _VERDICT_RE.search(desc or ""):
            graph.nodes[node_id].attrs.setdefault("has_verdict", True)
        # [[wikilinks]] -> links edges
        body = text
        for target in iter_wikilink_targets(body):
            tgt = _mem_id(target)
            if tgt == node_id:
                continue
            graph.ensure_stub(tgt, ntype="memory", title=target)
            etype = "links"
            graph.add_edge(Edge(node_id, tgt, etype, str(path)))
        # explicit sister / supersede
        for m in _SISTER_RE.finditer(body):
            tgt = _mem_id(m.group(1))
            graph.ensure_stub(tgt, ntype="memory", title=m.group(1).strip())
            graph.add_edge(Edge(node_id, tgt, "sister", str(path)))
        for m in _SUPERSEDE_RE.finditer(body):
            tgt = _mem_id(m.group(1))
            graph.ensure_stub(tgt, ntype="memory", title=m.group(1).strip())
            graph.add_edge(Edge(node_id, tgt, "supersedes", str(path)))
        # numeric #refs / file paths / equation ids / people in the BODY ->
        # entity references (memory files reference levers + files + eqs in prose)
        for m in _HASHREF_RE.finditer(body):
            ref_id = "ref:#" + m.group(1)
            graph.ensure_stub(ref_id, ntype="entity", title="#" + m.group(1))
            graph.add_edge(Edge(node_id, ref_id, "references", str(path)))
        for m in _FILE_RE.finditer(body):
            fid = "file:" + m.group(1)
            graph.ensure_stub(fid, ntype="entity", title=m.group(1))
            graph.add_edge(Edge(node_id, fid, "references", str(path)))
        for m in _EQID_RE.finditer(body):
            if m.group(1) in eq_ids:
                graph.add_edge(Edge(node_id, "eq:" + m.group(1), "references", str(path)))
                graph.add_edge(Edge(node_id, "eq:" + m.group(1), "equation_ref", str(path)))
        _synthesize_extended_refs(graph, node_id, body, eq_ids, str(path))
    return count


def parse_memory_indexes(graph: Graph, memory_dir: Path | None = None) -> int:
    """Synthesize note -> index ``indexed_by`` edges from markdown indexes."""
    mdir = memory_dir or _memory_dir()
    if not mdir.is_dir():
        return 0
    count = 0
    for path in sorted(p for p in mdir.glob("*.md") if _is_memory_index(p)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        index_id = "index:" + path.name
        graph.add_node(Node(
            id=index_id,
            ntype="index",
            title=path.stem,
            summary="Memory navigation index synthesized from local markdown links.",
            source=str(path),
            attrs={"index_kind": "root" if path.name == "MEMORY.md" else "cluster"},
        ))
        for match in _MARKDOWN_LINK_RE.finditer(text):
            raw = unquote(match.group(1)).split("#", 1)[0].strip("<>")
            target_path = (path.parent / raw).resolve()
            if target_path.suffix.lower() != ".md" or not target_path.is_file():
                continue
            try:
                target_path.relative_to(mdir.resolve())
            except ValueError:
                continue
            target_id = _memory_node_id_for_path(target_path)
            graph.ensure_stub(target_id, ntype="memory", title=target_path.stem)
            graph.add_edge(Edge(target_id, index_id, "indexed_by", str(path)))
            count += 1
    return count


def _parse_frontmatter(text: str) -> tuple[str, str, str]:
    """Return (name, type, description) from a memory-file YAML frontmatter."""
    if not text.startswith("---"):
        return "", "", ""
    end = text.find("\n---", 3)
    if end == -1:
        return "", "", ""
    fm = text[3:end]
    name = _fm_field(fm, "name")
    mtype = _fm_field(fm, "type")
    desc = _fm_field(fm, "description")
    return name, mtype, desc


def _fm_field(fm: str, key: str) -> str:
    # matches `key: value` or `key: "value"` (single-line; description may be a
    # long quoted string spanning the rest of the frontmatter until the next key)
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*(.*)$", fm)
    if not m:
        return ""
    val = m.group(1).strip()
    if val.startswith('"'):
        # quoted, possibly multi-line: capture through the closing quote
        rest = fm[m.start(1):]
        qm = re.match(r'"((?:[^"\\]|\\.)*)"', rest, re.DOTALL)
        if qm:
            return qm.group(1).replace('\\"', '"')
    return val.strip('"')


# --------------------------------------------------------------- parse: DAG --
def parse_dag_feeds(graph: Graph, dag_path: Path, eq_ids: set[str]) -> int:
    """Split the DAG into FEED-* blocks; a node per block + synthesized edges."""
    try:
        text = dag_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    lines = text.splitlines()
    # find block boundaries
    starts: list[tuple[int, str, str]] = []  # (lineno, slug, header)
    for i, line in enumerate(lines):
        m = _FEED_HDR_RE.match(line)
        if m:
            starts.append((i, m.group(1), line.lstrip("#").strip()))
    count = 0
    slug_seen: dict[str, int] = {}
    for idx, (ln, slug, header) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        block = "\n".join(lines[ln:end])
        # Line slicing excludes the complete heading regardless of LF/CRLF or heading whitespace.
        body = "\n".join(lines[ln + 1 : end])
        # disambiguate repeated slugs (an addendum/correction reuses a slug) so
        # every distinct block is preserved as its own node, not merged away.
        seen = slug_seen.get(slug, 0)
        slug_seen[slug] = seen + 1
        node_id = "feed:" + slug + (f"#{seen + 1}" if seen else "")
        verdict = _VERDICT_RE.search(header + " " + block[:400])
        graph.add_node(Node(
            id=node_id, ntype="decision" if verdict else "finding",
            title=_clean(header, 160),
            summary=_clean(body, _BLOCK_SUMMARY_CAP),
            source=f"{dag_path}#L{ln + 1}-L{end}",
            attrs={
                "date": (_DATE_RE.search(block).group(1) if _DATE_RE.search(block) else ""),
                "verdict": (verdict.group(1) if verdict else ""),
            },
        ))
        count += 1
        _synthesize_refs(graph, node_id, block, eq_ids, str(dag_path))
    return count


def _synthesize_refs(graph: Graph, node_id: str, block: str, eq_ids: set[str], source: str) -> None:
    """Add references/links/tagged edges from a block's mentions."""
    for target in iter_wikilink_targets(block):
        tgt = _mem_id(target)
        graph.ensure_stub(tgt, ntype="memory", title=target)
        graph.add_edge(Edge(node_id, tgt, "references", source))
    for m in _HASHREF_RE.finditer(block):
        ref_id = "ref:#" + m.group(1)
        graph.ensure_stub(ref_id, ntype="entity", title="#" + m.group(1))
        graph.add_edge(Edge(node_id, ref_id, "references", source))
    for m in _EQID_RE.finditer(block):
        eqid = m.group(1)
        if eqid in eq_ids:
            graph.add_edge(Edge(node_id, "eq:" + eqid, "references", source))
    for m in _FILE_RE.finditer(block):
        f = m.group(1)
        fid = "file:" + f
        graph.ensure_stub(fid, ntype="entity", title=f)
        graph.add_edge(Edge(node_id, fid, "references", source))
    for m in _PEOPLE_RE.finditer(block):
        pid = "person:" + m.group(1)
        graph.ensure_stub(pid, ntype="person", title=m.group(1))
        graph.add_edge(Edge(node_id, pid, "references", source))
    _synthesize_extended_refs(graph, node_id, block, eq_ids, source)


def _synthesize_extended_refs(
    graph: Graph, node_id: str, text: str, eq_ids: set[str], source: str,
) -> None:
    """Synthesize the cheap cross-corpus entity families with typed edges."""
    for match in _HASHREF_RE.finditer(text):
        ref_id = "ref:#" + match.group(1)
        graph.ensure_stub(ref_id, ntype="entity", title="#" + match.group(1))
        graph.add_edge(Edge(node_id, ref_id, "task_ref", source))
    for match in _EQID_RE.finditer(text):
        eqid = match.group(1)
        if eqid in eq_ids:
            graph.add_edge(Edge(node_id, "eq:" + eqid, "equation_ref", source))
    for match in _FEED_REF_RE.finditer(text):
        feed_id = "feed:" + match.group(1)
        graph.ensure_stub(feed_id, ntype="finding", title="FEED-" + match.group(1))
        graph.add_edge(Edge(node_id, feed_id, "feed_ref", source))
    for match in _LANE_ID_RE.finditer(text):
        lane_id = "lane:" + match.group(1)
        graph.ensure_stub(lane_id, ntype="lane", title=match.group(1))
        graph.add_edge(Edge(node_id, lane_id, "lane_ref", source))
    for match in _CATALOG_RE.finditer(text):
        catalog_id = "catalog:#" + match.group(1)
        graph.ensure_stub(catalog_id, ntype="catalog", title="Catalog #" + match.group(1))
        graph.add_edge(Edge(node_id, catalog_id, "catalog_ref", source))


def parse_section_documents(graph: Graph, paths: list[Path] | None = None) -> int:
    """Index doctrine headings and declared anchor slugs without editing docs."""
    if paths is None:
        paths = [p for p in _DOCTRINE_ROOTS if p.is_file()]
        if _DOCS_DIR.is_dir():
            paths.extend(sorted(_DOCS_DIR.rglob("*.md")))
    count = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        starts: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if match:
                starts.append((i, match.group(1).strip()))
        for idx, (line_no, header) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
            body = "\n".join(lines[line_no + 1:end])
            full_slug = _slugify_heading(header)
            if not full_slug:
                continue
            rel = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
            section_id = f"section:{rel}#{full_slug}"
            graph.add_node(Node(
                id=section_id,
                ntype="section",
                title=header,
                summary=_clean(body, _BLOCK_SUMMARY_CAP),
                source=f"{path}#L{line_no + 1}-L{end}",
            ))
            for alias in _section_aliases(header, body):
                alias_id = _mem_id(alias)
                graph.ensure_stub(alias_id, ntype="memory", title=alias)
                graph.add_edge(Edge(alias_id, section_id, "aliases", str(path)))
            _synthesize_extended_refs(graph, section_id, body, set(), str(path))
            count += 1
        for i, line in enumerate(lines, start=1):
            for match in _CATALOG_RE.finditer(line):
                cid = "catalog:#" + match.group(1)
                graph.add_node(Node(
                    id=cid,
                    ntype="catalog",
                    title="Catalog #" + match.group(1),
                    summary=_clean(line, 400),
                    source=f"{path}#L{i}",
                ))
    return count


def _research_node_id(path: Path) -> str:
    try:
        rel = path.relative_to(_RESEARCH_DIR).as_posix()
    except ValueError:
        rel = path.name
    return "research:" + rel


def parse_research_memos(graph: Graph, eq_ids: set[str], research_dir: Path | None = None) -> int:
    """Index dated research memos and their cross-memo/entity references."""
    root = research_dir or _RESEARCH_DIR
    if not root.is_dir():
        return 0
    count = 0
    files = sorted(p for p in root.rglob("*.md") if p.is_file())
    for path in files:
        if path.match(_DAG_GLOB):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        heading = next((m.group(1).strip() for line in text.splitlines()
                        if (m := re.match(r"^#{1,6}\s+(.+?)\s*$", line))), path.stem)
        node_id = _research_node_id(path)
        graph.add_node(Node(
            id=node_id,
            ntype="research",
            title=heading,
            summary=_clean(text, _BLOCK_SUMMARY_CAP),
            source=str(path),
        ))
        count += 1
        for target in iter_wikilink_targets(text):
            target_id = _mem_id(target)
            graph.ensure_stub(target_id, ntype="memory", title=target)
            graph.add_edge(Edge(node_id, target_id, "references", str(path)))
        for match in _MARKDOWN_LINK_RE.finditer(text):
            raw = unquote(match.group(1)).split("#", 1)[0].strip("<>")
            target_path = (path.parent / raw).resolve()
            if target_path.is_file() and target_path.suffix.lower() == ".md":
                try:
                    target_path.relative_to(root.resolve())
                except ValueError:
                    continue
                target_id = _research_node_id(target_path)
                graph.ensure_stub(target_id, ntype="research", title=target_path.stem)
                graph.add_edge(Edge(node_id, target_id, "memo_link", str(path)))
        _synthesize_extended_refs(graph, node_id, text, eq_ids, str(path))
    return count


# --------------------------------------------------------- parse: equations --
def parse_equations(graph: Graph, registry_path: Path) -> set[str]:
    """Node per equation + producer/consumer edges. Returns the set of eq ids."""
    eq_ids: set[str] = set()
    if not registry_path.is_file():
        return eq_ids
    for line in registry_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = d.get("equation_payload", d)
        eqid = payload.get("equation_id") or d.get("equation_id")
        if not eqid:
            continue
        node_id = "eq:" + eqid
        eq_ids.add(eqid)
        graph.add_node(Node(
            id=node_id, ntype="equation",
            title=payload.get("name") or eqid,
            summary=_clean(payload.get("one_line_summary", "") or payload.get("latex_form", ""), 400),
            source=str(registry_path),
            attrs={"residual": payload.get("predicted_vs_empirical_residual")},
        ))
        for prod in payload.get("canonical_producers", []) or []:
            pid = "file:" + prod
            graph.ensure_stub(pid, ntype="entity", title=prod)
            graph.add_edge(Edge(pid, node_id, "produces", str(registry_path)))
        for cons in payload.get("canonical_consumers", []) or []:
            cid = "file:" + cons
            graph.ensure_stub(cid, ntype="entity", title=cons)
            graph.add_edge(Edge(cid, node_id, "consumes", str(registry_path)))
    return eq_ids


# -------------------------------------------------------------- parse: tasks --
def parse_tasks(graph: Graph, tasks_path: Path) -> int:
    """Latest-row-wins task nodes + blocker edges (canonical_task_status.jsonl)."""
    if not tasks_path.is_file():
        return 0
    latest: dict[str, dict] = {}
    for line in tasks_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = d.get("task_id")
        if tid:
            latest[tid] = d  # later row wins
    for tid, d in sorted(latest.items()):
        node_id = "task:" + tid
        graph.add_node(Node(
            id=node_id, ntype="task",
            title=_clean(d.get("title", tid), 160),
            summary=_clean(d.get("event_notes", "") or d.get("status", ""), 400),
            source=str(tasks_path),
            attrs={"status": d.get("status", ""), "owner": d.get("owner", "")},
        ))
        numeric = re.fullmatch(r"#?(\d{1,4})", str(tid))
        if numeric:
            alias_id = "ref:#" + numeric.group(1)
            graph.ensure_stub(alias_id, ntype="entity", title="#" + numeric.group(1))
            graph.add_edge(Edge(alias_id, node_id, "aliases", str(tasks_path)))
        for b in d.get("blockers", []) or []:
            bl = str(b)
            m = _HASHREF_RE.search(bl)
            if m:
                ref_id = "ref:#" + m.group(1)
                graph.ensure_stub(ref_id, ntype="entity", title="#" + m.group(1))
                graph.add_edge(Edge(ref_id, node_id, "blocks", str(tasks_path)))
    return len(latest)


def parse_lanes(graph: Graph, registry_path: Path | None = None) -> int:
    """Canonical lane_registry rows -> resolvable lane nodes."""
    path = registry_path or _LANE_REGISTRY
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return 0
    lanes = payload.get("lanes", []) if isinstance(payload, dict) else []
    count = 0
    for lane in lanes:
        if not isinstance(lane, dict) or not lane.get("id"):
            continue
        lane_name = str(lane["id"])
        graph.add_node(Node(
            id="lane:" + lane_name,
            ntype="lane",
            title=lane_name,
            summary=_clean(str(lane.get("name", "")) + " " + str(lane.get("notes", "")), 400),
            source=str(path),
            attrs={"level": lane.get("level"), "phase": lane.get("phase")},
        ))
        count += 1
    return count


# ----------------------------------------------------------- parse: deferral --
def parse_deferrals(graph: Graph, ledger_path: Path) -> int:
    """Deferral-ledger table rows -> deferral nodes + task/#ref edges."""
    if not ledger_path.is_file():
        return 0
    count = 0
    for line in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("| D") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        did = cells[0]
        node_id = "deferral:" + did
        graph.add_node(Node(
            id=node_id, ntype="deferral",
            title=did + " " + _clean(cells[1], 100),
            summary=_clean(" ".join(cells[1:]), 400),
            source=str(ledger_path),
            attrs={"status": cells[-1][:60] if cells else ""},
        ))
        count += 1
        for m in _HASHREF_RE.finditer(line):
            ref_id = "ref:#" + m.group(1)
            graph.ensure_stub(ref_id, ntype="entity", title="#" + m.group(1))
            graph.add_edge(Edge(node_id, ref_id, "references", str(ledger_path)))
    return count


# ----------------------------------------------------- corpus source mtimes --
def corpus_sources(
    *,
    memory_dir: Path | None = None,
    dag_path: Path | None = None,
    equations_path: Path | None = None,
    tasks_path: Path | None = None,
    deferral_path: Path | None = None,
) -> list[Path]:
    """The concrete corpus files the graph indexes (existing ones only).

    Used by the auto-build cache (increment-2): the cached graph is stale the
    moment ANY of these is newer than the cache, so recall never serves a graph
    that has fallen behind the markdown source of truth.
    """
    srcs: list[Path] = []
    default_corpus = all(
        value is None
        for value in (memory_dir, dag_path, equations_path, tasks_path, deferral_path)
    )
    mdir = memory_dir or _memory_dir()
    if mdir.is_dir():
        srcs.extend(sorted(mdir.glob("*.md")))
    if default_corpus:
        if _RESEARCH_DIR.is_dir():
            srcs.extend(sorted(p for p in _RESEARCH_DIR.rglob("*.md") if p.is_file()))
        srcs.extend(p for p in _DOCTRINE_ROOTS if p.is_file())
        if _DOCS_DIR.is_dir():
            srcs.extend(sorted(p for p in _DOCS_DIR.rglob("*.md") if p.is_file()))
    if dag_path is None:
        dag_files = sorted(_RESEARCH_DIR.glob(_DAG_GLOB))
        dag_path = dag_files[-1] if dag_files else None
    trailing = [dag_path, equations_path or _EQUATIONS_JSONL,
                tasks_path or _TASKS_JSONL, deferral_path or _DEFERRAL_MD]
    if default_corpus:
        trailing.append(_LANE_REGISTRY)
    for p in trailing:
        if p is not None and Path(p).is_file():
            srcs.append(Path(p))
    return sorted(set(srcs))


def corpus_mtime(**kwargs) -> float:
    """Max mtime across all corpus sources (0.0 if there are none)."""
    mt = 0.0
    for p in corpus_sources(**kwargs):
        try:
            mt = max(mt, p.stat().st_mtime)
        except OSError:
            continue
    return mt


# ------------------------------------------------------------------- driver --
def build_graph(
    *,
    memory_dir: Path | None = None,
    dag_path: Path | None = None,
    equations_path: Path | None = None,
    tasks_path: Path | None = None,
    deferral_path: Path | None = None,
) -> Graph:
    """Parse the whole corpus into one Graph. Missing sources are skipped."""
    default_corpus = all(
        value is None
        for value in (memory_dir, dag_path, equations_path, tasks_path, deferral_path)
    )
    g = Graph()
    eq_path = equations_path or _EQUATIONS_JSONL
    eq_ids = parse_equations(g, eq_path)
    if dag_path is None:
        dag_files = sorted(_RESEARCH_DIR.glob(_DAG_GLOB))
        dag_path = dag_files[-1] if dag_files else None
    if dag_path is not None:
        parse_dag_feeds(g, dag_path, eq_ids)
    parse_memory_files(g, memory_dir, eq_ids)
    parse_memory_indexes(g, memory_dir)
    if default_corpus:
        parse_section_documents(g)
        parse_research_memos(g, eq_ids)
    parse_tasks(g, tasks_path or _TASKS_JSONL)
    if default_corpus:
        parse_lanes(g)
    parse_deferrals(g, deferral_path or _DEFERRAL_MD)
    parse_costate_organ_ledger(g)
    return g

# ---------------------------------------------------------- costate organ ----
_ORGAN_LEDGER = _RESEARCH_DIR / "costate_organ_trajectory_ledger.md"


def parse_costate_organ_ledger(graph: Graph, ledger_path: Path | None = None) -> int:
    """Index the #426 costate-organ trajectory ledger (triality-native memory): each
    FEED-426-organ block becomes a `finding` node; each named prototype regime becomes
    a typed `regime` node with edges to the run entity + the organ block. The ledger
    markdown REMAINS the source of truth (reconstruct-not-retrieve)."""
    p = ledger_path or _ORGAN_LEDGER
    if not p.exists():
        return 0
    try:
        from tac.witness_control.continual_costate import load_organ_memory
    except Exception:
        return 0
    mem = load_organ_memory(p)
    n = 0
    for rec in mem.records:
        stamp = str(rec.get("generated_at", "?"))
        block_id = f"finding:FEED-426-organ-{stamp}"
        run_ref = str(rec.get("run_ref", "?"))
        graph.add_node(Node(
            id=block_id, ntype="finding",
            title=f"FEED-426-organ-{stamp}",
            summary=(f"organ trajectory record run={run_ref}; "
                     f"{rec.get('n_intervals')} intervals; walk-forward winner "
                     f"{rec.get('winner_walkforward')}"),
            source=str(p), attrs={"run_ref": run_ref}))
        graph.ensure_stub(f"entity:{run_ref}", "entity", run_ref)
        graph.add_edge(Edge(block_id, f"entity:{run_ref}", "references", str(p)))
        graph.ensure_stub("entity:#426", "entity", "#426")
        graph.add_edge(Edge(block_id, "entity:#426", "references", str(p)))
        n += 1
    for proto in mem.prototype_library:
        rid = f"regime:{proto['name']}"
        graph.add_node(Node(
            id=rid, ntype="regime", title=proto["name"],
            summary=(f"costate regime prototype ({proto.get('scale')}, "
                     f"block_dim={proto.get('block_dim', 0)}, "
                     f"{proto.get('n_observations', 1)} obs; "
                     f"first {proto.get('first_run')}, last {proto.get('last_run')})"),
            source=str(p),
            attrs={"center": proto.get("center"),
                   "n_observations": proto.get("n_observations", 1)}))
        graph.add_edge(Edge(rid, "entity:#426", "references", str(p)))
        lr = proto.get("last_run")
        if lr:
            graph.ensure_stub(f"entity:{lr}", "entity", str(lr))
            graph.add_edge(Edge(rid, f"entity:{lr}", "references", str(p)))
        n += 1
    return n
