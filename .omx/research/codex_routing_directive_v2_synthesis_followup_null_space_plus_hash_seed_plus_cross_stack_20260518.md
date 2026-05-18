# Codex routing directive v2: synthesis follow-up — null-space + hash-seed + cross-stack synergies
# Date: 2026-05-18
# Originating session: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
# Supersedes ITEM 5 of `codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518.md`
# Per CLAUDE.md "Subagent coherence-by-default" + AGENTS.md Claude×Codex feedback loop

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (full; honor NON-NEGOTIABLE markers)
2. `/Users/adpena/Projects/pact/AGENTS.md` (full; Claude×Codex feedback loop)
3. `.omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md` (**THE canonical scoping document for this directive — 1249 lines / 70 surfaces / 5 op-routables**)
4. `.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md` (contest-compliance verdict)
5. `.omx/research/codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518.md` (PRIOR directive; ITEMS 1-4 still apply; ITEM 5 SUPERSEDED by THIS directive)
6. `.omx/research/grand_council_symposium_inflate_py_extreme_compression_20260518.md` (parent symposium)

## SYNTHESIS VERDICT SUMMARY

Convergent prediction across THREE independent subagent waves (deep-research + asymptotic-stacking + synthesis):
- Final frontier potential: **`[0.122, 0.173]` [contest-CPU]** (10-13% improvement over current 0.19205)
- Wave 1-4 cumulative ΔS: `[-0.070, -0.019]` under realistic α-discount per Catalog #322
- Compliance-validated triad (hash-seed + weight-derived + null-space) accounts for `[-0.058, -0.016]` of the aggregate

## WORK ITEMS (priority order; reference synthesis memo § for full design)

### ITEM 5 (RE-SCOPED per OP-3) — `tac.procedural_codebook_generator` package

Per synthesis OP-3: build TWO sister canonical helpers under `src/tac/procedural_codebook_generator/`:

**`hash_seed_codebook_generator.py`** (~250 LOC + tests):
```python
# Public API (synthesis memo § "Hash-seed PRNG codebook generation"):
def emit_seed(target_codebook_shape: tuple[int, ...], target_distribution: str = "uniform_int8") -> bytes
def expand_seed_to_codebook(seed: bytes, target_shape: tuple[int, ...], target_distribution: str = "uniform_int8") -> np.ndarray
def verify_byte_mutation_smoke(seed_path: pathlib.Path, archive_path: pathlib.Path, mutation_offsets: Sequence[int]) -> bool  # Catalog #272 sister
```

Implementation: `numpy.random.Generator(numpy.random.PCG64(int.from_bytes(seed, 'big')))` for byte-identical deterministic codebook on CPU + CUDA. CRITICAL: validate `numpy.random.Generator` cross-platform determinism via spawn-pool test on both Apple Silicon + Linux x86_64 (HNeRV parity L9 runtime closure).

**`weight_derived_codebook_generator.py`** (~250 LOC + tests):
```python
def derive_codebook_from_archive_bytes(archive_path: pathlib.Path, source_member: str, target_shape: tuple[int, ...]) -> np.ndarray
def freeze_source_member_sha256(archive_path: pathlib.Path, source_member: str) -> str  # cite in Provenance per Catalog #323
def verify_no_new_bytes_added(before_archive: pathlib.Path, after_archive: pathlib.Path) -> bool  # MANDATORY contest-compliance gate
```

Implementation: `hashlib.sha256` on the frozen source member's bytes → PRNG seed → codebook. Source member MUST be inside `archive.zip` already (no external state per maintainer precedent). The frozen-SHA discipline is critical — any mutation of the source member invalidates the codebook.

**Sister probe**: `tools/probe_procedural_codebook_generator_validation.py` (~150 LOC):
- Empirical validation on NSCS06 v8 chroma palette replacement (~7.5KB → 8-byte seed)
- Byte-mutation smoke per Catalog #272 (mutate seed bytes; verify inflate output changes)
- contest-CPU dispatch: ~$3-5 envelope (Modal A10G smoke; 5-epoch validation)

**Catalog # claim**: NEW STRICT preflight gate `check_procedural_codebook_generator_canonical_use` to refuse any substrate that ships pre-baked constants. Claim via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "procedural_codebook_generator canonical-use enforcement per contest-compliance"`. WARN-ONLY initial wire-in.

**Cross-stack wire-in**: integrate with `tac.wyner_ziv_deliverability.proof_builder` (Catalog #319 Q1) — hash-seed bytes count as `TIER_1_ZERO_COST` per the deliverability prober contract. This converts NSCS06 v7 chroma palette from TIER-2 (deferred, ~7.5KB charged) to TIER-1 (8 bytes charged → effectively zero-cost).

Expected aggregate ΔS: `[-0.018, -0.004]` across 5 substrates.

### ITEM 6 (NEW per OP-2) — `tac.null_space_exploiter` canonical helper (HIGHEST single-item EV)

Per synthesis OP-2: build canonical helper under `src/tac/null_space_exploiter/`:

```python
# Public API (synthesis memo § "Master-gradient null-space exploitation roadmap"):
def compute_null_space_basis(master_gradient: np.ndarray, axis_threshold: float = 0.95) -> np.ndarray
    # Returns orthonormal basis of the null subspace where (seg_grad, pose_grad) co-rank-degenerate (cos > threshold)
def project_modifications_onto_null_space(modifications: np.ndarray, null_basis: np.ndarray) -> np.ndarray
    # Aligned modifications produce ZERO axial ΔS
def compute_score_safe_byte_reduction(archive_bytes: bytes, master_gradient: np.ndarray, null_basis: np.ndarray, target_reduction_bytes: int) -> bytes
    # The frontier-breaking primitive: reduce archive bytes via null-space-aligned modifications
```

Empirical anchor: cos(seg, pose) = 0.8973 on fec6 archive (Fields-Medal slot 1). The 1D null space carries [-0.040, -0.012] aggregate potential ΔS across 4 frontier archives once ITEM 3 (extractor extension) lands.

**Sister probe**: `tools/probe_null_space_exploiter_validation.py`:
- Empirical null-space identification on fec6 baseline
- Byte-reduction validation: target 5-10% archive shrinkage at predicted-zero-ΔS
- contest-CPU dispatch on PR101 fec6: ~$3 envelope

**Wire-in** per synthesis hook #4 cathedral autopilot gap closure: add `null_space_safe_reduction` reward factor to autopilot reweight v2 cascade per Catalog #319.

Expected aggregate ΔS: `[-0.040, -0.012]` — **HIGHEST EV single op-routable** in the entire synthesis.

### ITEM 7 (NEW per OP-4) — per-pair master gradient wire-in audit + 6 wire-in closures

Per synthesis OP-4: audit per-pair master gradient consumption across 6 named sites + close gaps:

1. `tac.bit_allocator.allocate_per_pair` (currently doesn't consume; should)
2. `tac.optimization.field_equation_planner.field_row` (consumes Pareto-relevant constraints; needs per-pair fp64 integration)
3. `tac.xray.*` primitive registry (13 primitives; only 5 currently consume per-pair signal)
4. `tac.continual_learning.posterior_update_locked` (per-pair-keyed posterior NOT YET landed; Wave 3 #802)
5. Cross-correlation between per-pair × per-deliberation surfaces (Rashomon ensemble per Catalog #252)
6. cathedral autopilot v2 cascade per-pair reweight (Catalog #319)

Output: `.omx/research/per_pair_master_gradient_wire_in_audit_20260518.md` with per-site verdict (ACTIVE / DORMANT / DESIGN-ONLY / NOT-APPLICABLE) + closure design per DORMANT site.

Expected aggregate ΔS from closures: `[-0.008, -0.002]` (validation signal routing improvement).

### ITEM 8 (NEW per OP-5) — multi-granularity sensitivity tensor (pair × byte × class × axis) in DuckDB

Per synthesis OP-5: extend `tac.canonical_duckdb.per_byte_sensitivity_ext` to support 4-tuple keying `(pair_id, byte_offset, class_id, axis)`.

Schema addition:
```sql
CREATE TABLE multi_granularity_sensitivity (
  archive_sha256 TEXT NOT NULL,
  pair_id INTEGER NOT NULL,
  byte_offset INTEGER NOT NULL,
  class_id INTEGER NOT NULL,  -- 5 SegNet classes
  axis TEXT NOT NULL,  -- 'seg' / 'pose' / 'rate'
  sensitivity_fp64 DOUBLE NOT NULL,
  derived_at_utc TIMESTAMP NOT NULL,
  PRIMARY KEY (archive_sha256, pair_id, byte_offset, class_id, axis)
);
```

Cross-stack synergy: enables per-class chroma-aware quantization (NSCS06 v7 pattern) + per-pair difficulty-aware Wyner-Ziv hoist (Catalog #319) + per-class composition cells (Catalog #322).

Expected aggregate ΔS per dispatch: `[-0.005, -0.001]` (better signal routing).

### ITEM 9 (NEW per cross-stack synergy #2) — NSCS06 v7 chroma palette → hash-seed replacement

THE CANONICAL TEST CASE for ITEM 5 (`hash_seed_codebook_generator`). Per synthesis cross-stack synergy #2:

NSCS06 v7 per-class chroma anchors (per `feedback_pre_rigor_kill_defer_falsified_inventory_landed_20260517.md` #864 symposium) currently ship as ~7.5KB per-class chroma palette per substrate. Replace with 8-byte PRNG seed + inflate-time codebook expansion.

Net effect:
- Archive bytes: -7.4 KB per substrate × 33 substrates × stacking discount ≈ -1.5 to -3 KB aggregate
- Rate term reduction: `25 * 2_000 / 37_545_489` ≈ **`-0.0013` ΔS guaranteed**
- Stacked with other items: composes orthogonally (per Catalog #322 anti-phantom; verify with N×N pairwise α)

Empirical validation gate: $3 Modal A10G smoke + paired contest-CPU GHA Linux x86_64 anchor. If ΔS ≥ -0.001 on PR101 fec6 + sister substrate, roll out to 32+ remaining substrates.

This is the FIRST empirical anchor for the entire hash-seed family — gates all of ITEM 5's downstream contest-CUDA dispatches.

## DISCIPLINE REMINDERS

All discipline from prior directive (codex_routing_directive_inflate_py_plus_*) still applies:
- Catalog #117 + #157 + #174 commit serializer + POST-EDIT sha
- Catalog #186 catalog-claim transactionality (do NOT pre-claim #s)
- Catalog #206 checkpoint every ~10 tool uses
- Catalog #229 premise verification
- Catalog #272 byte-mutation smoke per archive change
- Catalog #287 [empirical:<path>] tag every claim
- Catalog #292 explicit assumption-statement per architectural decision
- Catalog #295 inflate.py works with empty PYTHONPATH
- Catalog #305 observability surface section in every new design memo
- Catalog #313 probe-outcomes ledger consultation
- Catalog #314 declare files_touched up-front (sister-subagent absorption-pattern protection)
- Catalog #319 v2 cascade ITEM 6 null_space_exploiter wire-in MANDATORY
- Catalog #323 ProvenanceKind enum extensions (ITEM 2 from prior directive — required before ITEM 5 deploys)

## RECOMMENDED SEQUENCING

Phase 1 (DEFENSIVE — blocks downstream substrate engineering risk):
- ITEM 1 + ITEM 2 from prior directive (compliance rationale + ProvenanceKind enum) — ~5-7h, $0

Phase 2 (UNLOCK):
- ITEM 3 from prior directive (master-gradient extractor extension) — ~24-32h, $0
- ITEM 7 (per-pair wire-in audit) — ~16-24h, $0

Phase 3 (FRONTIER-BREAKING):
- ITEM 6 (null_space_exploiter; HIGHEST EV single item) — ~16-24h + $3, [-0.040, -0.012] ΔS
- ITEM 5 (procedural_codebook_generator; cross-stack with ITEM 9) — ~24-32h + $5-15, [-0.018, -0.004] ΔS
- ITEM 9 (NSCS06 v7 chroma → hash-seed) — VALIDATION ANCHOR for ITEM 5; ~6-8h + $3, [-0.0013] guaranteed ΔS

Phase 4 (OPTIMIZATION):
- ITEM 8 (multi-granularity sensitivity tensor) — ~32-40h, $0, [-0.005, -0.001] per dispatch
- ITEM 4 from prior directive (OP-1+OP-2+OP-5 reviewability) — ~6h, $0, reviewability-only

## OPERATOR ACKNOWLEDGEMENT

Per AGENTS.md role division: this is Claude's DESIGN deliverable. Codex executes per next /goal or pre-flight directive scan. Acknowledge receipt of THIS directive in your next checkpoint via `tools/subagent_checkpoint.py --notes "incorporated codex_routing_directive_v2_synthesis_followup_20260518"`.

— Main-Claude (relayed on behalf of operator)
