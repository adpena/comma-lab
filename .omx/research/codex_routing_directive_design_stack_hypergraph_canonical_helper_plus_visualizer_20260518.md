# Codex routing directive: Design-stack hypergraph canonical helper + visualizer (Layer 1+2+3+4)
# Date: 2026-05-18
# Operator: approved 2026-05-18 ("Chain but how about a graph" + "All are approved")
# Per Catalog #245 canonical 4-layer pattern (sister of Modal call_id ledger + Codex→Claude inbox + Claude memory hermetic export)

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Subagent coherence-by-default" + "Meta-Lagrangian/Pareto solver" + Catalog #125 6-hook + #245 canonical 4-layer + #322 composition_alpha)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (1449 lines; THE 9×9 matrix IS the seed adjacency representation)
4. `src/tac/deploy/modal/call_id_ledger.py` (canonical 4-layer template)
5. `src/tac/codex_to_claude_inbox.py` (sister channel; mirror structure when it lands per routing directive 745fc2e19)
6. `tools/claim_catalog_number.py peek` returns next available: **333** at directive time
7. `src/tac/codec/wyner_ziv_layer.py` + `src/tac/wyner_ziv_deliverability/` + `src/tac/procedural_codebook_generator/` + `src/tac/null_space_exploiter/` (deterministic-byte-derivation subsystem; first-class typed node category in the hypergraph)

## CANONICAL 4-LAYER PATTERN

### LAYER 1 — Canonical helper at `src/tac/design_graph.py`

```python
# src/tac/design_graph.py
GRAPH_SCHEMA_VERSION = "design_stack_hypergraph_v1_20260518"
GRAPH_PATH = REPO_ROOT / ".omx" / "state" / "design_stack_hypergraph.json"
GRAPH_LOCK = GRAPH_PATH.with_suffix(".json.lock")

# Typed node categories (10 first-class types)
VALID_NODE_TYPES = frozenset({
    "design",                     # design memos at .omx/research/*_design_*_*.md
    "canonical_helper",           # src/tac/*.py canonical helpers
    "meta_gate",                  # Catalog #1...#332 STRICT preflight gates
    "probe",                      # tools/probe_*_disambiguator.py
    "substrate",                  # entries in lane_registry.json
    "venn_cell",                  # cells from 3-set/6-set Venn per Catalog #319/#322
    "posterior",                  # .omx/state/*_posterior.jsonl + sister state ledgers
    "consumer",                   # cathedral_autopilot_ranker / planner / dispatch
    "empirical_anchor",           # PR101_lc_v2 f174192aeadf... + DP1 sidecars + fec6 + etc.
    "deterministic_byte_derivation",  # Wyner-Ziv side-info + procedural_codebook_generator + null_space_exploiter + optical-flow side-info + FOE detection (META-category: derive bytes deterministically at inflate from contest seeds)
})

# Typed edge categories (7 first-class types; all directed)
VALID_EDGE_TYPES = frozenset({
    "produces_input_for",         # A → B; A produces signal B consumes
    "consumes_output_of",         # A → B; A consumes signal B produces (reverse direction; redundant but explicit for query convenience)
    "composes_with",              # A → B; A and B compose via Catalog #322 composition_alpha; weight = alpha_value + tier (additive/sub_additive/saturating/orthogonal/exclusive)
    "cycles_back_to",             # A → B; A's output feeds into B's input forming a cycle; weight = cycle_latency (per-LOOP / per-deliberation / per-dispatch)
    "gates_eligibility_of",       # A → B; A is a META gate that blocks B unless waiver/satisfied
    "waiver_eligible_via",        # A → B; A is a META gate, B is its canonical waiver pattern
    "empirically_anchors",        # A → B; A is an empirical anchor for B; weight = anchor_freshness_days + axis (CPU/CUDA/macOS-advisory)
})

# Typed hyperedges (3+ nodes; canonical for composition_alpha N-way per Catalog #322)
VALID_HYPEREDGE_TYPES = frozenset({
    "n_way_composition_alpha",    # {A, B, C, ...} → joint composition with weighted alpha
    "n_way_pareto_feasibility",   # {A, B, C, ...} → joint Pareto-feasible region per Dykstra alternating projection
    "n_way_venn_cell_stratification",  # {A, B, C, ...} → joint stratification over byte positions
})

class HypergraphRowValidationError(ValueError): pass
class HypergraphCorruptError(RuntimeError): pass

# Public API (mirror Catalog #245 + sister channels)
def add_node(*, node_id: str, node_type: str, source_path: str, metadata: dict, agent: str = "claude", **extra) -> dict: ...
def add_edge(*, src_node_id: str, dst_node_id: str, edge_type: str, weight: float | None = None, metadata: dict | None = None, agent: str = "claude", **extra) -> dict: ...
def add_hyperedge(*, node_ids: tuple[str, ...], hyperedge_type: str, weight: float | None = None, metadata: dict | None = None, agent: str = "claude", **extra) -> dict: ...

def query_critical_path(*, source: str | None = None, target: str | None = None, weight_attr: str = "predicted_delta_s") -> list[str]:
    """Longest weighted path through the DAG (after cycle-removal); identifies bottleneck for dispatch sequencing."""

def query_orphan_signals(*, direction: str = "producer_without_consumer") -> list[str]:
    """direction ∈ {producer_without_consumer, consumer_without_producer}; surfaces Catalog #711 sister analysis structurally."""

def query_hyperedge_compositions(*, contains_node: str | None = None, alpha_tier: str | None = None) -> list[dict]:
    """Returns N-way compositions matching filters; structural Catalog #322 lookup."""

def query_cycles(*, max_length: int | None = None) -> list[list[str]]:
    """Cycle detection via Tarjan SCC; surfaces continual-learning feedback loops vs deadlock candidates."""

def query_hook_coverage(*, hook_id: int = 1) -> dict:
    """Catalog #125 6-hook audit over the full graph; returns {hook: {producers: [...], consumers: [...], orphans: [...]}}"""

def query_dominator(*, node_id: str) -> set[str]:
    """Subgraph dominator analysis; surfaces downstream impact of a single node failure/change."""

def export_dot(*, output_path: Path | None = None) -> str:
    """DOT format export for graphviz rendering."""

def load_hypergraph_strict(path: Path | None = None) -> dict:
    """Catalog #138 fail-closed loader; quarantines corrupt to .corrupt.<utc>."""
```

**Implementation invariants** (mirror Catalog #245):
- fcntl-locked atomic writes per Catalog #131
- Strict-load per Catalog #138 (quarantines on parse failure)
- JSON byte-stable (sort_keys=True; ensure_ascii=False)
- 4-proc spawn-pool stress test in tests
- DAG sanity check: cycles_back_to edges flagged for cycle-handling discipline (not treated as DAG dependencies)

### LAYER 2 — CLI tool at `tools/render_design_graph.py`

```bash
# Render full graph to DOT → SVG via graphviz:
.venv/bin/python tools/render_design_graph.py render --output design_stack.svg

# Render to ASCII tree (terminal-friendly):
.venv/bin/python tools/render_design_graph.py render --format=ascii

# Critical-path query:
.venv/bin/python tools/render_design_graph.py critical-path --source pose_axis_t3_council --target frontier_displacement

# Orphan-signals audit (Catalog #711 sister):
.venv/bin/python tools/render_design_graph.py orphans --direction producer_without_consumer
.venv/bin/python tools/render_design_graph.py orphans --direction consumer_without_producer

# Hyperedge query (Catalog #322 N-way composition_alpha):
.venv/bin/python tools/render_design_graph.py hyperedges --alpha-tier sub_additive

# Cycle detection (continual-learning loops):
.venv/bin/python tools/render_design_graph.py cycles --max-length 5

# Hook coverage audit (Catalog #125 6-hook):
.venv/bin/python tools/render_design_graph.py hooks --hook 4

# Dominator analysis (downstream impact):
.venv/bin/python tools/render_design_graph.py dominator --node phase_1_fisher_precondition

# Add nodes/edges programmatically (used by other canonical helpers):
.venv/bin/python tools/render_design_graph.py add-node --id <id> --type design --source-path <path>
.venv/bin/python tools/render_design_graph.py add-edge --src <id> --dst <id> --type composes_with --weight 0.7
```

### LAYER 3 — STRICT preflight gate Catalog #333

```python
# Catalog #333: check_design_graph_hook_coverage_complete_or_orphans_declared
# Refuses repo state with hook-orphan signals (producer-without-consumer OR consumer-without-producer) UNLESS
# declared in canonical .omx/state/design_graph_orphan_waivers.json
# Same-line waiver: # DESIGN_GRAPH_ORPHAN_OK:<rationale>
```

Initial wire-in WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule". Strict-flip after first clean cycle of `query_hook_coverage()` returning zero orphans across all 6 hooks.

### LAYER 4 — operator_briefing.py wire-in

`tools/operator_briefing.py` extends to include `design_graph_node_count` + `design_graph_edge_count` + `design_graph_hyperedge_count` + `design_graph_orphan_count` + `design_graph_critical_path_length` in briefing output.

## SEED GRAPH (Codex bootstraps from)

Codex's first action after building the canonical helper: bootstrap the graph from today's 9 design landings + Codex's just-landed canonical helpers via:

```python
from tac.design_graph import add_node, add_edge, add_hyperedge

# 9 design nodes (parse Catalog #300 v2 frontmatter from each)
for memo_path in glob('.omx/research/*_design_*_20260518.md') + glob('.omx/research/grand_council_*_20260518.md'):
    metadata = parse_council_frontmatter(memo_path)
    add_node(node_id=basename(memo_path), node_type="design", source_path=memo_path, metadata=metadata)

# Canonical helpers (from src/tac/)
for helper_path in [
    "src/tac/master_gradient.py",
    "src/tac/codec/wyner_ziv_layer.py",
    "src/tac/wyner_ziv_deliverability/",
    "src/tac/procedural_codebook_generator/",
    "src/tac/null_space_exploiter/",
    "src/tac/optimization/substrate_composition_matrix.py",
    "src/tac/cost_band_calibration.py",
    "src/tac/canonical_task_status.py",
    # ... full enumeration per the synthesis memo §3.1
]:
    add_node(node_id=basename(helper_path), node_type="canonical_helper", source_path=helper_path, metadata={})

# Deterministic-byte-derivation META-category (operator's question highlighted this)
add_node(node_id="wyner_ziv_seed_subsystem", node_type="deterministic_byte_derivation",
         source_path="src/tac/wyner_ziv_deliverability/__init__.py",
         metadata={"members": ["wyner_ziv_layer", "wyner_ziv_deliverability", "procedural_codebook_generator", "null_space_exploiter", "optical_flow_side_info", "foe_detection"],
                   "canonical_principle": "Wyner-Ziv 1976 side-info at decoder + Atick-Redlich 1990 cooperative-receiver",
                   "deliverable_tiers": ["TIER_1_ZERO_COST", "TIER_2_CONSTANTS", "TIER_3_WAIVER_REQUIRED", "TIER_4_FORBIDDEN"]})

# Edges from 9×9 synthesis matrix (parse from .omx/research/cross_stack_synthesis_*_20260518.md §4)
for (src, dst, edge_type, weight) in parse_cross_pollination_matrix(synthesis_memo_path):
    add_edge(src_node_id=src, dst_node_id=dst, edge_type=edge_type, weight=weight)

# Hyperedges from Catalog #322 composition_alpha cells (parse from substrate_composition_matrix.json)
for cell in load_composition_matrix():
    add_hyperedge(node_ids=cell["substrate_ids"], hyperedge_type="n_way_composition_alpha",
                  weight=cell["alpha_value"], metadata={"alpha_tier": cell["alpha_tier"]})
```

## DETERMINISTIC-BYTE-DERIVATION SUBSYSTEM (operator-highlighted)

Per operator question 2026-05-18 ("wyner-ziv is still being considered as part of the seed and other procedural gen experiments right?"), THIS subsystem gets first-class typed-node category status:

```
deterministic_byte_derivation members:
├── wyner_ziv_layer            (src/tac/codec/wyner_ziv_layer.py)
├── wyner_ziv_deliverability   (src/tac/wyner_ziv_deliverability/ package; Tier 1-4 classification)
├── procedural_codebook_generator (src/tac/procedural_codebook_generator/; Codex landed 7c13abda3)
├── null_space_exploiter        (src/tac/null_space_exploiter/; Codex landed 7c13abda3; HIGHEST-EV per all design memos)
├── optical_flow_side_info      (planned per cheap-probe wave OP-1; deterministic optical-flow from contest video)
└── foe_detection               (planned per pose-axis council OP-6 LFV1 Telescope + LAPose)

Canonical principle: Wyner-Ziv 1976 side-info at decoder + Atick-Redlich 1990 cooperative-receiver
Deliverable tiers: TIER_1_ZERO_COST | TIER_2_CONSTANTS | TIER_3_WAIVER_REQUIRED | TIER_4_FORBIDDEN
```

Operator-routable queries this enables structurally:

```bash
# Which substrates have unconsumed deterministic_byte_derivation primitives ready to hoist?
.venv/bin/python tools/render_design_graph.py orphans --direction consumer_without_producer --filter-type deterministic_byte_derivation

# What's the dominator-set for a hypothetical Tier 1 hoist across all substrates?
.venv/bin/python tools/render_design_graph.py dominator --node wyner_ziv_seed_subsystem

# What N-way compositions involve the deterministic-byte-derivation META-category?
.venv/bin/python tools/render_design_graph.py hyperedges --contains-node wyner_ziv_seed_subsystem
```

## TESTS

`src/tac/tests/test_design_graph.py` covering:
1. Schema invariants (10 node types + 7 edge types + 3 hyperedge types + GRAPH_SCHEMA_VERSION pinned)
2. add_node / add_edge / add_hyperedge happy paths + validation rejections
3. query_critical_path / query_orphan_signals / query_hyperedge_compositions / query_cycles / query_hook_coverage / query_dominator
4. export_dot produces valid DOT format
5. load_hypergraph_strict raises HypergraphCorruptError on malformed JSON
6. Atomic writes (no .tmp leakage)
7. 4-proc spawn-pool stress test
8. JSONL byte-stable
9. CLI subprocess smoke per Layer 2 subcommand
10. STRICT preflight gate Catalog #333 fixtures
11. Sister Catalog #131 path registered (GRAPH_PATH in _SHARED_STATE_PATH_MARKERS)

Target: 35+ dedicated tests.

## CLAUDE.md catalog row 333

Per Catalog #176 template (after STRICT gate lands).

## DISCIPLINE (same as sister Catalog #245 + #331 + #332 directives)

Standard discipline applies. Codex builds following canonical 4-layer pattern.

## EXIT CRITERIA

- [ ] `src/tac/design_graph.py` exists with all public API functions
- [ ] `src/tac/tests/test_design_graph.py` 35+ tests pass
- [ ] `tools/render_design_graph.py` CLI runnable; all 8 subcommands
- [ ] Catalog #333 wired into `preflight_all(strict=False)`
- [ ] CLAUDE.md row 333 appended
- [ ] Lane `lane_design_stack_hypergraph_canonical_helper_20260518` at L1
- [ ] Seed graph bootstrap completes via Codex's first invocation (parses today's 9 design memos + canonical helpers + 9×9 matrix + composition_alpha cells)
- [ ] DOT export produces valid graphviz output
- [ ] First `query_orphan_signals()` returns 3 hook-CONSUMER-without-producer flags per cross-stack synthesis §8

— Main-Claude 2026-05-18 (operator-approved 4-layer canonical pattern + deterministic-byte-derivation META-category first-class per "wyner-ziv is still being considered as part of the seed")
