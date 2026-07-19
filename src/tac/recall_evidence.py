# SPDX-License-Identifier: MIT
"""Fused RecallEvidence — one ranked surface over lexical + graph recall (task #569, P0-2).

Why this exists (the two-list problem):
    ``tools/corpus_query.py`` (lexical density retrieval) and
    ``tools/graph_memory_recall.py`` (reconstruct-not-retrieve graph traversal)
    each return their OWN result stream with an incomparable score scale. A
    decision-time consumer had to read TWO lists and eyeball-merge them, so a
    hit that ranked #2 lexically and #1 in the graph looked weaker than a #1
    lexical hit — cross-surface ranking was impossible.

This module normalizes both streams into one typed ``RecallEvidence`` row
(source surface, store, ref, path, score, hook line, rank) and reciprocal-rank
fuses them so a single ranked list is produced. The Cerebras ``k=60`` smoothing
constant is a candidate, not sacred (the crosswalk rank-6 caveat) — it is a
plain parameter here.

Design boundary: the fusion is dependency-injected. It consumes the *outputs*
of ``corpus_query.run_query`` (a dict) and ``graph_memory.reconstruct`` (a
``Reconstruction`` + its ``Graph``) — it does NOT import the tools/ CLI scripts,
so it stays a pure ``tac`` library. ``tools/recall_fused.py`` wires the two live
surfaces into it.

Determinism: pure functions of the injected results; stable tie-breaks on
``(-rrf_score, source_surface, ref)``. No RNG, no I/O.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tac.graph_memory.model import Graph
    from tac.graph_memory.recall import Reconstruction

# Reciprocal-rank-fusion smoothing constant. Source-reported (Cerebras KB), NOT
# proven Pact-optimal — exposed as a parameter so a gold-query suite can tune it.
DEFAULT_RRF_K = 60

_LINE_ANCHOR_RE = re.compile(r"#L\d+(?:-L\d+)?$")
_ID_PREFIX_RE = re.compile(r"^(?:file|memory|eq|ref|task|deferral|regime|topic|person|entity|feed):")


@dataclass(frozen=True, slots=True)
class RecallEvidence:
    """One normalized recall hit, comparable ACROSS retrieval surfaces.

    ``source_surface`` is the retriever ("corpus" | "graph"); ``store`` is the
    within-surface bucket (a corpus store name, or a graph node type);
    ``score`` is the retriever's own (non-comparable) score; ``rank`` is the
    1-based position WITHIN its surface; ``rrf_score`` is the cross-surface
    reciprocal-rank-fused score (0.0 until ``reciprocal_rank_fuse`` sets it).
    """

    source_surface: str
    store: str
    ref: str
    path: str
    score: float
    hook_line: str
    rank: int
    rrf_score: float = 0.0
    contributing_surfaces: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────── adapters ───────────────────────────


def evidence_from_corpus(result: dict) -> list[RecallEvidence]:
    """Adapt a ``corpus_query.run_query`` result dict into RecallEvidence rows.

    ``result["hits"]`` is already score-sorted (best first); we assign the
    1-based rank from that order and carry the first matching line as the hook.
    """
    rows: list[RecallEvidence] = []
    for i, hit in enumerate(result.get("hits", []), start=1):
        lines = hit.get("lines") or []
        rows.append(RecallEvidence(
            source_surface="corpus",
            store=str(hit.get("store", "")),
            ref=str(hit.get("ref", "")),
            path=str(hit.get("ref", "")),
            score=float(hit.get("score", 0.0)),
            hook_line=(lines[0] if lines else ""),
            rank=i,
        ))
    return rows


def evidence_from_reconstruction(recon: "Reconstruction", graph: "Graph") -> list[RecallEvidence]:
    """Adapt a graph ``Reconstruction`` into RecallEvidence rows.

    ``recon.nodes`` is weight-ordered (seeds first, then accumulated BFS
    weight); we rank by that order. The graph surface has no comparable scalar
    score, so ``score`` is a monotone POSITIONAL proxy (higher = earlier in the
    weight order) — labelled as such and used only for within-surface ranking;
    cross-surface comparison happens through ``rrf_score``, which depends only
    on rank.
    """
    payload = recon.to_dict(graph)
    nodes = payload.get("reconstructed_nodes", [])
    n = len(nodes)
    rows: list[RecallEvidence] = []
    for i, node in enumerate(nodes, start=1):
        summary = (node.get("summary") or "").strip()
        hook = summary[:200] if summary else str(node.get("title", ""))
        rows.append(RecallEvidence(
            source_surface="graph",
            store=str(node.get("ntype", "")),
            ref=str(node.get("id", "")),
            path=str(node.get("source", "")),
            score=float(n - i + 1),  # positional (weight-order) proxy
            hook_line=hook,
            rank=i,
        ))
    return rows


# ─────────────────────────── fusion ───────────────────────────


def _norm_key(ev: RecallEvidence) -> str:
    """A cross-surface identity key: same underlying artifact -> same key.

    Strips ``path#L..`` line anchors, a trailing `` :: FEED-header`` (corpus DAG
    ref), and graph id prefixes (``file:`` / ``memory:`` / ``eq:`` ...), then
    lowercases. Rows whose keys coincide (e.g. two surfaces citing the same file
    path) MERGE their RRF contributions; rows that do not simply keep their own.
    """
    raw = ev.path or ev.ref
    raw = raw.split(" :: ", 1)[0]
    raw = _LINE_ANCHOR_RE.sub("", raw)
    raw = _ID_PREFIX_RE.sub("", raw)
    return raw.strip().lower()


def reciprocal_rank_fuse(
    streams: list[list[RecallEvidence]],
    *,
    k: int = DEFAULT_RRF_K,
    weight: float = 1.0,
) -> list[RecallEvidence]:
    """Reciprocal-rank-fuse ranked RecallEvidence streams into ONE ranked list.

    For each row, its contribution is ``weight / (k + rank)``; rows across
    streams that normalize to the same key sum their contributions. The
    representative row kept per key is the one with the best (lowest) native
    rank; its ``rrf_score`` is the summed value and ``contributing_surfaces``
    records every surface that voted for it.

    NO row is dropped (exact-identifier preservation): every distinct artifact
    key appears exactly once in the output. Consumers take the top-k they want.
    """
    if k < 0:
        raise ValueError("rrf k must be non-negative")
    fused: dict[str, dict] = {}
    for stream in streams:
        for ev in stream:
            key = _norm_key(ev)
            if not key:
                continue
            contrib = weight / (k + ev.rank)
            slot = fused.get(key)
            if slot is None:
                fused[key] = {
                    "rep": ev,
                    "rrf": contrib,
                    "surfaces": {ev.source_surface},
                    "best_rank": ev.rank,
                }
            else:
                slot["rrf"] += contrib
                slot["surfaces"].add(ev.source_surface)
                if ev.rank < slot["best_rank"]:
                    slot["rep"] = ev
                    slot["best_rank"] = ev.rank
    out: list[RecallEvidence] = []
    for slot in fused.values():
        rep: RecallEvidence = slot["rep"]
        out.append(RecallEvidence(
            source_surface=rep.source_surface,
            store=rep.store,
            ref=rep.ref,
            path=rep.path,
            score=rep.score,
            hook_line=rep.hook_line,
            rank=rep.rank,
            rrf_score=round(slot["rrf"], 6),
            contributing_surfaces=tuple(sorted(slot["surfaces"])),
        ))
    out.sort(key=lambda e: (-e.rrf_score, e.source_surface, e.ref))
    return out


def fuse_recall(
    *,
    corpus_result: dict | None = None,
    reconstruction: "Reconstruction | None" = None,
    graph: "Graph | None" = None,
    k: int = DEFAULT_RRF_K,
) -> list[RecallEvidence]:
    """Convenience: adapt both live surfaces and fuse them into one ranked list.

    Either surface may be omitted (single-surface fusion is valid — it just
    reciprocal-rank-normalizes one stream). ``reconstruction`` requires
    ``graph`` to serialize.
    """
    streams: list[list[RecallEvidence]] = []
    if corpus_result is not None:
        streams.append(evidence_from_corpus(corpus_result))
    if reconstruction is not None and graph is not None:
        streams.append(evidence_from_reconstruction(reconstruction, graph))
    return reciprocal_rank_fuse(streams, k=k)
