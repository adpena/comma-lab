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
from pathlib import Path

from .model import Edge, Graph, Node

# ------------------------------------------------------------------ paths ----
REPO_ROOT = Path(__file__).resolve().parents[3]
_DAG_GLOB = "sub015_DAG_topaiml_reopen_and_pursuit_plan_*.md"
_RESEARCH_DIR = REPO_ROOT / ".omx" / "research"
_EQUATIONS_JSONL = REPO_ROOT / ".omx" / "state" / "canonical_equations_registry.jsonl"
_TASKS_JSONL = REPO_ROOT / ".omx" / "state" / "canonical_task_status.jsonl"
_DEFERRAL_MD = REPO_ROOT / ".omx" / "state" / "deferral_ledger.md"

# Read caps: scoring/parsing on a bounded prefix is a deterministic heuristic
# (the DAG is ~1.7 MB; a memory description is small).
_BLOCK_SUMMARY_CAP = 600
_MEMORY_SUMMARY_CAP = 800

# ------------------------------------------------------------- extractors ----
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# lever / task numeric refs: #247, #205 (2-4 digits, avoids color hex etc.)
_HASHREF_RE = re.compile(r"(?<![\w])#(\d{2,4})\b")
# file paths under our source trees
_FILE_RE = re.compile(
    r"(?<![\w./])((?:src/tac|tools|experiments|scripts|\.omx/research|docs)/[\w./-]+\.\w{1,4})"
)
# canonical equation ids look like snake_case ... _v<N>
_EQID_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+_v\d+)\b")
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


def _memory_dir() -> Path:
    slug = str(REPO_ROOT).replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug / "memory"


def _mem_id(link_text: str) -> str:
    return "memory:" + link_text.strip()


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
        for m in _WIKILINK_RE.finditer(body):
            tgt = _mem_id(m.group(1))
            if tgt == node_id:
                continue
            graph.ensure_stub(tgt, ntype="memory", title=m.group(1).strip())
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
        # disambiguate repeated slugs (an addendum/correction reuses a slug) so
        # every distinct block is preserved as its own node, not merged away.
        seen = slug_seen.get(slug, 0)
        slug_seen[slug] = seen + 1
        node_id = "feed:" + slug + (f"#{seen + 1}" if seen else "")
        verdict = _VERDICT_RE.search(header + " " + block[:400])
        graph.add_node(Node(
            id=node_id, ntype="decision" if verdict else "finding",
            title=_clean(header, 160),
            summary=_clean(block[len(header):], _BLOCK_SUMMARY_CAP),
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
    for m in _WIKILINK_RE.finditer(block):
        tgt = _mem_id(m.group(1))
        graph.ensure_stub(tgt, ntype="memory", title=m.group(1).strip())
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
        for b in d.get("blockers", []) or []:
            bl = str(b)
            m = _HASHREF_RE.search(bl)
            if m:
                ref_id = "ref:#" + m.group(1)
                graph.ensure_stub(ref_id, ntype="entity", title="#" + m.group(1))
                graph.add_edge(Edge(ref_id, node_id, "blocks", str(tasks_path)))
    return len(latest)


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
    mdir = memory_dir or _memory_dir()
    if mdir.is_dir():
        srcs.extend(sorted(mdir.glob("*.md")))
    if dag_path is None:
        dag_files = sorted(_RESEARCH_DIR.glob(_DAG_GLOB))
        dag_path = dag_files[-1] if dag_files else None
    for p in (dag_path, equations_path or _EQUATIONS_JSONL,
              tasks_path or _TASKS_JSONL, deferral_path or _DEFERRAL_MD):
        if p is not None and Path(p).is_file():
            srcs.append(Path(p))
    return srcs


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
    g = Graph()
    eq_path = equations_path or _EQUATIONS_JSONL
    eq_ids = parse_equations(g, eq_path)
    if dag_path is None:
        dag_files = sorted(_RESEARCH_DIR.glob(_DAG_GLOB))
        dag_path = dag_files[-1] if dag_files else None
    if dag_path is not None:
        parse_dag_feeds(g, dag_path, eq_ids)
    parse_memory_files(g, memory_dir, eq_ids)
    parse_tasks(g, tasks_path or _TASKS_JSONL)
    parse_deferrals(g, deferral_path or _DEFERRAL_MD)
    return g
