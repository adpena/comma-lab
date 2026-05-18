# Codex routing directive: canonical 3-set Venn classification helper package
# Date: 2026-05-18
# Operator: closure campaign master memo `closure_campaign_pursue_and_confirm_master_20260518.md` op-routable OPR-CLOSE-2 batch
# Authority: `.omx/research/n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518.md` (THE design memo with full canonical-vs-unique decision per layer + 9-dim checklist + cargo-cult audit + observability surface + Dykstra-feasibility predicted-band)
# Sister: closure_campaign_pursue_and_confirm_master_20260518.md §11.2 TOP-5 (OPR-CLOSE-2)
# Per CLAUDE.md "Subagent coherence-by-default" + Catalog #245 4-layer pattern + Catalog #290 + #294 + #303 + #305 + #296

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially Catalog #125 6-hook + #245 4-layer + #131 fcntl-locked JSONL)
2. `/Users/adpena/Projects/pact/AGENTS.md` (Claude × Codex role specialization; Codex executes builds; Claude designs)
3. `.omx/research/n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518.md` (THE AUTHORITY design memo)
4. `.omx/research/closure_campaign_pursue_and_confirm_master_20260518.md` (parent coordinator memo)
5. `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` §4 VENN row in 9×9 matrix (CONSUMES → FISHER/RIEM/TROP/POSEAXIS/FLOOR; ORTHO with Z8)
6. `src/tac/probe_outcomes_ledger.py` (sister canonical 4-layer helper pattern exemplar per Catalog #313)
7. `src/tac/deploy/modal/call_id_ledger.py` (sister canonical 4-layer helper pattern exemplar per Catalog #245)
8. `src/tac/council_continual_learning.py` (sister canonical helper pattern with fcntl-locked JSONL per Catalog #131)

## STRATEGIC CONTEXT

The 3-set Venn classification design memo specifies a canonical pair-region-class-frame-axis classifier that:
- Classifies every per-pair byte into one of 64 cells (3-set Venn = 2^6 = 64; the 3 sets are pair/region/class on each of frame_0/frame_1 axes per design memo §5)
- Computes per-cell density via empirical sparsity atlas
- Anchors on PR101_lc_v2 master-gradient `f174192aeadf...` per Catalog #316 frontier
- Predicts ΔS `-0.005` per Shannon's H(X|Y_6-set) lens (Catalog #287 axis tag `[prediction]`)
- Composes additively with FISHER + RIEM + TROP per cross-stack §4 matrix

This directive routes the full 4-layer canonical helper build to Codex.

## OP-1: Layer 1 — canonical helper module

**Target**: `src/tac/canonical_n_set_venn_classification/__init__.py` + sister modules

**Public API** (per design memo §6 canonical interface specification):

```python
# src/tac/canonical_n_set_venn_classification/__init__.py
from .region import Region, RegionSet
from .classifier import classify_pair_region_class, classify_archive_per_cell
from .cell_density import compute_cell_density, CellDensityMatrix
from .anchor_writer import append_n_set_venn_anchor, load_n_set_venn_anchors_strict
from .verdict import NSetVennClassificationVerdict

__all__ = [
    "Region", "RegionSet",
    "classify_pair_region_class", "classify_archive_per_cell",
    "compute_cell_density", "CellDensityMatrix",
    "append_n_set_venn_anchor", "load_n_set_venn_anchors_strict",
    "NSetVennClassificationVerdict",
]
```

**Canonical dataclasses** (frozen per design memo §5):

```python
@dataclass(frozen=True)
class Region:
    """One axis of the 3-set Venn (pair / region / class) on one frame axis (0 / 1)."""
    set_id: str  # "pair" | "region" | "class"
    frame_axis: int  # 0 | 1
    indices: tuple[int, ...]  # byte indices in this region

@dataclass(frozen=True)
class RegionSet:
    """Full 3-set × 2-frame = 6-axis classification scope."""
    pair_frame_0: Region
    pair_frame_1: Region
    region_frame_0: Region
    region_frame_1: Region
    class_frame_0: Region
    class_frame_1: Region
    archive_sha256: str  # canonical 64-char hex per Catalog #316
    archive_bytes: int

@dataclass(frozen=True)
class CellDensityMatrix:
    """64-cell density per Catalog #287 axis tag [prediction]."""
    cell_id_to_density: dict[str, float]  # cell_id format: "pf0:RF0:CF0" (binary 0/1 per axis)
    archive_sha256: str
    measurement_axis: str  # "[contest-CPU]" | "[contest-CUDA T4]" | "[macOS-CPU advisory]"
    hardware: str  # canonical per Catalog #190
    captured_at_utc: str
    provenance: dict  # canonical Provenance per Catalog #323
```

**Implementation requirements** (per design memo §6 + §10 9-dim checklist):

- `classify_pair_region_class(archive_sha256, pair_index, byte_index) -> str` returns cell_id in format `"pf0:RF0:CF0"` where each axis is 0 or 1
- `classify_archive_per_cell(archive_sha256, *, pair_count) -> CellDensityMatrix` iterates all per-pair bytes and computes 64-cell density
- `compute_cell_density(region_set, archive_bytes) -> CellDensityMatrix` deterministic given archive_sha256 + region_set
- `append_n_set_venn_anchor(matrix, *, sidecar_dir=".omx/state/n_set_venn_anchors/")` writes anchor to fcntl-locked JSONL per Catalog #131
- `load_n_set_venn_anchors_strict(sidecar_dir)` raises `NSetVennAnchorCorruptError` on JSON parse failure per Catalog #138

**Canonical-vs-unique decision per layer** (per Catalog #290 + design memo §canonical-vs-unique-decision-per-layer):

| Layer | Decision | Rationale |
|---|---|---|
| Archive grammar parsing | ADOPT canonical (`tac.master_gradient.parse_*_archive_bytes`) | Sister Catalog #167 routing |
| Per-pair iteration | ADOPT canonical (`tac.substrates._shared.pair_iterator`) | Standard contract |
| fcntl-locked JSONL persistence | ADOPT canonical (Catalog #131 helper) | Cross-substrate discipline |
| Provenance | ADOPT canonical (`tac.provenance.build_provenance_prediction`) | Catalog #323 |
| Cell density computation | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per-cell density formula is N-set-Venn-specific; canonical scorer-loss helper does not apply |
| 64-cell partitioning | FORK_BECAUSE_PRINCIPLED_MISMATCH | 3-set Venn structure is unique to this design memo |
| Region intersection | ADOPT canonical (Python set ops) | No FORK needed |

## OP-2: Layer 2 — CLI tool

**Target**: `tools/canonical_n_set_venn_classification_cli.py`

**CLI subcommands**:

```bash
# Compute 64-cell density for an archive
.venv/bin/python tools/canonical_n_set_venn_classification_cli.py compute \
    --archive-sha 6bae0201... --archive-path experiments/results/.../archive.zip \
    --pair-count 600 --output .omx/state/n_set_venn_anchors/anchor_<sha[:12]>_<utc>.json

# Query existing anchor
.venv/bin/python tools/canonical_n_set_venn_classification_cli.py query \
    --archive-sha 6bae0201... --format json

# Audit all anchors against frontier
.venv/bin/python tools/canonical_n_set_venn_classification_cli.py audit \
    --frontier-axis contest_cpu --summary
```

**Exit codes**: 0 clean / 1 strict-fail-on-violation / 2 CLI arg error / 3 corrupt sidecar.

## OP-3: Layer 3 — STRICT preflight gate

**Catalog #N**: claim transactionally via `.venv/bin/python tools/claim_catalog_number.py claim --commit-via-serializer --reason "VENN canonical helper STRICT gate"`. Use the returned # as `_CHECK_N_*` constants in `src/tac/preflight.py`.

**Gate function**: `check_canonical_n_set_venn_classification_use(strict: bool, verbose: bool) -> list`

**Refuses**: any source-text scan under `src/tac/`, `tools/`, `experiments/`, `scripts/` that calls a forbidden hand-rolled-Venn pattern token (`hand_rolled_venn_cell`, `inline_64_cell_partition`, `custom_region_intersection`) WITHOUT routing through `tac.canonical_n_set_venn_classification`. Acceptance:
- Imports `from tac.canonical_n_set_venn_classification import` token within ±10 lines OR
- Same-line `# CANONICAL_N_SET_VENN_BYPASS_OK:<rationale>` waiver (placeholder `<rationale>` / `<reason>` literal rejected)

**Wire-in**: `preflight_all(strict=False)` warn-only at landing; STRICT-flip after 0-violation verification per CLAUDE.md "Strict-flip atomicity rule".

**Tests**: ≥40 tests in `src/tac/tests/test_check_<N>_canonical_n_set_venn_classification_use.py` covering:
- Live-repo regression guard (≤5 ceiling at landing)
- Positive (hand-rolled pattern flagged in 3 surfaces)
- Negative (canonical import accepted)
- Waiver semantics (rationale accepted; placeholder rejected)
- Self-exempt (canonical helper file)
- Strict-mode raises with Catalog #N message
- Orchestrator wire-in regression guard

## OP-4: Layer 4 — integration wire-ins per Catalog #125 6-hook

**Hook #1 (sensitivity-map contribution)**: `tac.sensitivity_map` consumer extension at `src/tac/sensitivity_map/n_set_venn_contribution.py`:
```python
def add_n_set_venn_contribution_to_axis_weights(axis_weights, cell_density_matrix):
    """Per-cell density → per-axis weight contribution."""
    ...
```

**Hook #2 (Pareto constraint)**: 64-cell polytope constraint at `tac.pareto_64_cell_polytope_constraint`:
```python
def add_n_set_venn_polytope_constraint(pareto_solver, cell_density_matrix):
    """64-cell polytope contributes Pareto-feasibility constraint per design memo §7."""
    ...
```

**Hook #3 (bit-allocator)**: per-cell tier hook at `tac.bit_allocator.per_cell_tier`:
```python
def allocate_bits_per_cell_tier(cell_density_matrix, total_budget_bytes):
    """High-density cells get tier-1 bits; low-density cells tier-3."""
    ...
```

**Hook #4 (cathedral autopilot dispatch)**: extend `tools/cathedral_autopilot_autonomous_loop.py` with `adjust_predicted_delta_for_n_set_venn_class_v3` cascade:
```python
def adjust_predicted_delta_for_n_set_venn_class_v3(predicted_delta, archive_sha256, cell_class):
    """v3 cascade: HIGH_CELL_DENSITY → +0.05 reward; LOW_CELL_DENSITY → -0.10 penalty."""
    # Reads .omx/state/n_set_venn_anchors/ via load_n_set_venn_anchors_strict
    ...
```

**Hook #5 (continual-learning posterior update)**: `append_n_set_venn_anchor` writes per measurement to `.omx/state/n_set_venn_anchors/anchor_<sha[:12]>_<utc>.json` per Catalog #131 fcntl-locked discipline. Also call `tac.continual_learning.posterior_update_locked` for cross-substrate posterior.

**Hook #6 (probe-disambiguator)**: `tools/probe_n_set_venn_empirical_sparsity_atlas.py` — produces empirical sparsity atlas for any archive; disambiguates whether 64-cell partition is empirically meaningful (high entropy across cells) vs degenerate (most bytes in 1-2 cells).

## OP-5: Operator-facing audit tool per Catalog #305 observability surface

**Target**: `tools/audit_n_set_venn_classification_compliance.py`

Operator-runnable any time; emits JSON or `--summary` human-readable. Shows:
- Number of archives with n_set_venn anchors
- Per-cell density distribution (mean / std / min / max across all 64 cells)
- Top-5 highest-density cells across all archives (for FISHER consumer prioritization)
- Top-5 lowest-density cells (for TROP boundary consumer prioritization)

## OP-6: Memory entry

**Target**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_canonical_n_set_venn_classification_package_landed_<utc>.md`

Required sections (per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in declaration):
- 6-hook wire-in declaration (ALL 6 hooks WIRED or N/A_WITH_RATIONALE)
- Canonical-vs-unique decision per layer evidence (per Catalog #290)
- 9-dim checklist evidence (per Catalog #294)
- Cargo-cult audit per assumption (per Catalog #303)
- Observability surface declaration (per Catalog #305)
- Predicted ΔS band Dykstra-feasibility check (per Catalog #296)
- Empirical anchor evidence (1 anchor per archive in 6-archive matrix; per OP-AUDIT-1 OPR-CLOSE-1)

## OP-7: canonical_task_status row emission

Per closure verifier `tac.closure_completion_verifier` (sister design spec in closure_campaign_pursue_and_confirm_master_20260518.md §6): at landing of THIS routing directive, Codex MUST emit canonical_task_status row:

```python
from tac.canonical_task_status import upsert_task

upsert_task(
    task_id="codex_routing_directive_canonical_n_set_venn_classification_package_20260518::OP_1",
    title="Build Layer 1 canonical helper module",
    owner="codex",
    status="pending",
    predicted_cost_usd=0.0,
    source_design_memo=".omx/research/n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518.md",
)
# Repeat for OP_2 through OP_7
```

## OP-routables ranked (within this directive's scope)

| Rank | Op | Cost | EV |
|---|---|---|---|
| 1 | OP-1 Layer 1 canonical helper | ~2 days editor | ∞ (everything else depends on this) |
| 2 | OP-3 Layer 3 STRICT gate | ~0.5 days editor | high (extincts future hand-rolled-Venn class) |
| 3 | OP-4 Layer 4 6-hook integration | ~1 day editor | high (unblocks autopilot v3 cascade) |
| 4 | OP-2 Layer 2 CLI | ~0.5 days editor | medium |
| 5 | OP-5 audit tool | ~0.5 days editor | medium (operator observability) |
| 6 | OP-6 memory entry | ~0.5 hours editor | required closure |
| 7 | OP-7 canonical_task_status rows | ~5 min | required closure (closure verifier dependency) |

## Discipline checklist

- [ ] Catalog #229 premise verification BEFORE editing each file
- [ ] Catalog #206 checkpoint discipline every ~10 tool uses
- [ ] Catalog #117/#157/#174 commit serializer with POST-EDIT working-tree sha for EVERY file edit
- [ ] Catalog #126 lane pre-registered: extend `lane_canonical_n_set_venn_classification_package_20260518`
- [ ] Catalog #186 catalog # claimed transactionally via `--commit-via-serializer --reason`
- [ ] Catalog #290 canonical-vs-unique decision per layer in memory entry
- [ ] Catalog #294 9-dim checklist evidence in memory entry
- [ ] Catalog #303 cargo-cult audit per assumption in memory entry
- [ ] Catalog #305 observability surface in memory entry
- [ ] Catalog #125 6-hook wire-in declared in memory entry
- [ ] Catalog #314 absorption avoidance: scope is ONLY this routing directive's OPs; no other source files edited

Begin.
