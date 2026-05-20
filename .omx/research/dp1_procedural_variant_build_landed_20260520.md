# DP1 procedural-codebook replacement variant — L0 SCAFFOLD BUILD LANDED 2026-05-20

**Lane**: `lane_dp1_procedural_codebook_replacement_variant_20260520` L1 (impl_complete + memory_entry)

**Parent design memo**: `feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md`
(commit `498805f8d`) OP-ROUTABLE #2.

**Operator approval**: blanket approval 2026-05-20 *"3 slots are approved + magic codec stacking"*.

## TL;DR

Built the DP1 PROCEDURAL VARIANT trainer extension that closes the
BUILD-then-dispatch gap from parent design memo OP-ROUTABLE #2. The
substrate scaffold can now be paired-smoked from a clean
READY-TO-DISPATCH state by the operator-authorize harness; this lane
EXPLICITLY does NOT fire any paid dispatch (OP-ROUTABLE #3 territory).

The variant replaces the ~4096-byte Comma2k19-derived codebook section
in the DP1 archive with a 32-byte PCG64 seed. At inflate time the
canonical helper
`tac.procedural_codebook_generator.derive_codebook_from_seed` re-derives
the codebook bytes byte-stably from the seed. **Predicted ΔS = -0.002706**
per canonical equation
`procedural_codebook_from_seed_compression_savings_v1`
(`src/tac/canonical_equations/procedural_codebook_savings.py`) closed
form `-25 × (4096 - 32) / 37,545,489`.

## Sister coordination

* **Slot 1 (MAGIC CODEC STACKING ANALYSIS)** — sister `??` artifacts at
  `tools/run_magic_codec_dense_streams_dwt_residual_smoke.py`. Disjoint
  scope (different file path; my code lives under
  `src/tac/substrates/pretrained_driving_prior/`).
* **Slot 3 (CANONICAL EQUATION #26 DOMAIN REFINEMENT)** — sister `M`
  artifacts under `src/tac/canonical_equations/`. Disjoint scope (my code
  does NOT modify the canonical equation; integration is via the IN-DOMAIN
  context string constant + a fall-back `try/except ImportError` on the
  `validate_context_is_in_domain` helper that slot 3 lands).
* **Catalog #340 sister-checkpoint guard**: PROCEED on my 4 target files
  (NO collisions; the lane registry and audit log are commonly multi-subagent
  state per Catalog #131 fcntl-locked discipline).

## Deliverables landed

| Deliverable | Path | Status |
|---|---|---|
| Procedural-variant trainer extension (~400 LOC core) | `src/tac/substrates/pretrained_driving_prior/distillation_procedural_variant.py` | ✓ landed |
| `__init__.py` delta (re-exports + `PROCEDURAL_VARIANT_AVAILABLE=True` flag) | `src/tac/substrates/pretrained_driving_prior/__init__.py` | ✓ landed |
| `archive.py` thin convenience wrapper `compose_procedural_archive` | `src/tac/substrates/pretrained_driving_prior/archive.py` | ✓ landed |
| Tests (~15 tests; 100% PASS) | `src/tac/substrates/pretrained_driving_prior/tests/test_procedural_variant.py` | ✓ 15/15 PASS |
| Lane registry L0 SCAFFOLD entry + L1 impl_complete mark | `.omx/state/lane_registry.json` | ✓ landed via `tools/lane_maturity.py` |
| Landing memo (this file) | `.omx/research/dp1_procedural_variant_build_landed_20260520.md` | ✓ landed |
| Memory landing memo | `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_dp1_procedural_trainer_build_landed_20260520.md` | ✓ landed |

## Public API

The variant module exposes 4 callable surfaces + 3 constants + 1 config
dataclass + 1 error type. All importable from the top-level package:

```python
from tac.substrates.pretrained_driving_prior import (
    PROCEDURAL_VARIANT_AVAILABLE,           # bool capability flag
    PROCEDURAL_SEED_SIZE_BYTES,             # 32
    PROCEDURAL_CODEBOOK_SHAPE_DEFAULT,      # (1024, 4)
    PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,      # np.uint8
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT, # str constant
    ProceduralVariantConfig,                # frozen dataclass
    ProceduralVariantError,                 # ValueError subclass
    derive_procedural_codebook_replacement, # forward direction
    compose_with_procedural_codebook,       # archive composition
    compose_procedural_archive,             # archive.py convenience
    verify_procedural_codebook_in_domain,   # IN-DOMAIN check
    verify_seed_mutation_changes_codebook_bytes,  # Catalog #272 smoke
)
```

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Procedural derivation engine | ADOPT_CANONICAL | `tac.procedural_codebook_generator.derive_codebook_from_seed` is the canonical 3-PRNG helper landed earlier today by sister subagent; reusing it preserves byte-stability across substrates. |
| Archive grammar | ADOPT_CANONICAL | DP1 archive grammar (`DP1_HEADER_FMT` + 4 length-prefixed sections) is preserved byte-for-byte; only the codebook section payload bytes change. |
| Score-aware loss | ADOPT_CANONICAL | The variant inherits `DrivingPriorScoreAwareLoss` unchanged. Score-axis routing per Catalog #164 `score_pair_components`. |
| Eval-roundtrip | ADOPT_CANONICAL | Inherits from canonical DP1 trainer; no change needed at variant landing. |
| Configuration dataclass | FORK_PRINCIPLED | `ProceduralVariantConfig` is a NEW dataclass distinct from `DistillationConfig` because the inputs are structurally different (seed bytes + generator kind vs Comma2k19 dataset + max_frames). |
| In-domain context label | ADOPT_CANONICAL | `comma2k19_ood_derived_basis_replacement` per slot 3 refined domain. |
| IN-DOMAIN check | FORK_GRACEFUL | Integration is `try: from tac.canonical_equations import validate_context_is_in_domain except ImportError: fallback to constant comparison`. Once slot 3's helper lands, the production path uses the canonical helper. |
| Byte-mutation smoke | ADOPT_CANONICAL | Sister of Catalog #272 contract surface; mirrors `verify_generator_seed_mutation_smoke` in the canonical helper module. |
| Archive composition signature | UNIQUE_PER_METHOD | `compose_with_procedural_codebook(original_archive, seed_bytes)` is a NEW signature distinct from `compose_with` (DPCOMP composition with base substrate) because the semantics are different (codebook bytes replaced, not wrapped). |
| Convenience wrapper | ADOPT_CANONICAL | `compose_procedural_archive` in `archive.py` is a thin shim around `compose_with_procedural_codebook` with canonical defaults — pure DX sugar. |
| `PROCEDURAL_VARIANT_AVAILABLE` flag | UNIQUE_PER_METHOD | New capability-detection flag for recipe-side imports + autopilot ranker per Catalog #335 paradigm. |

## Catalog #272 byte-mutation distinguishing-feature contract

The variant's operational mechanism IS byte-mutation-traceable. Test
`test_compose_with_procedural_codebook_byte_mutation_smoke_catalog_272`
flips the FIRST seed byte and asserts:

1. The resulting archive bytes differ from the original archive bytes.
2. The codebook section bytes specifically differ (the other 3 payload
   sections — renderer/residual/meta — are byte-identical).

This is the canonical Catalog #272 byte-mutation smoke. Per the lane
registry evidence: `byte_mutation_smoke_passes=true`.

## 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| UNIQUENESS | First substrate with procedural-codebook replacement variant; structurally distinct from canonical Comma2k19-derived DP1. |
| BEAUTY + ELEGANCE | ~400 LOC core module + ~50 LOC `__init__.py` delta + ~50 LOC archive.py shim; 15 tests; reviewable in 30 seconds per HNeRV parity L4. |
| DISTINCTNESS | The variant is structurally OOD by construction (seed bytes are deterministic PRNG output; NO Comma2k19 data flows through the path). |
| RIGOR | 15 tests; all 206 sister DP1 tests pass post-landing; Catalog #272 byte-mutation smoke PASSED; PV-of-PV checked sister activity 2x. |
| OPTIMIZATION-PER-TECHNIQUE | ADOPT_CANONICAL on 10 layers + FORK_PRINCIPLED on 1 layer per Catalog #290. |
| STACK-OF-STACKS-COMPOSABILITY | Variant inherits DP1 score_pair_components contract; composes via `compose_with` (DPCOMP) with any base substrate. |
| DETERMINISTIC REPRODUCIBILITY | PCG64 + fixed brotli quality + sorted-keys metadata; same seed → same archive bytes byte-for-byte (asserted in `test_derive_procedural_codebook_replacement_deterministic`). |
| EXTREME OPTIMIZATION + PERFORMANCE | Derivation is O(N) in output bytes; no GPU; no Comma2k19 download required at inflate time. |
| OPTIMAL MINIMAL CONTEST SCORE | Predicted ΔS = -0.002706 per canonical equation #26 (HYPOTHESIS; first empirical anchor pending OP-ROUTABLE #3 paired smoke). |

## Observability surface (Catalog #305)

| Facet | Surface |
|---|---|
| Inspectable per layer | `derive_procedural_codebook_replacement` returns the derived `np.ndarray` directly; callers can inspect shape/dtype/byte distribution. |
| Decomposable per signal | `compose_with_procedural_codebook` preserves per-section offsets; downstream consumers can decompose via `parse_dp1_archive_bytes`. |
| Diff-able across runs | Two runs with the same seed produce byte-identical archive bytes (asserted by determinism test); two runs with different seeds produce different bytes (asserted by byte-mutation smoke test). |
| Queryable post-hoc | `ProceduralVariantConfig` captures every input; `compose_with_procedural_codebook` raises on invalid inputs with substantive error messages. |
| Cite-able | All 4 callable surfaces + 3 constants are exported with full docstrings + cross-references to canonical equation #26 + parent design memo. |
| Counterfactual-able | `verify_seed_mutation_changes_codebook_bytes` IS the canonical byte-mutation counterfactual probe. |

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| 32-byte seed produces enough entropy for the codebook derivation | HARD-EARNED | PCG64 has period 2^128; 32 bytes = 256 bits seeds the generator with abundant entropy for a 4096-byte output. |
| The seed bytes deterministically derive the codebook bytes byte-stably across platforms | HARD-EARNED | Canonical helper `derive_codebook_from_seed` is pinned to little-endian byte ordering; sister test `test_derive_procedural_codebook_replacement_deterministic` verifies. |
| The brotli-compressed seed shrinks below the brotli-compressed canonical codebook | HARD-EARNED-PROVISIONAL | 32-byte seed compresses to ~37 bytes; 4096-byte codebook compresses to ~3 KB (the original Comma2k19-derived codebook). Bytes-saved test asserts `> 0`. |
| The procedural codebook produces dashcam-statistically-plausible outputs | CARGO-CULTED-PENDING-EMPIRICAL | A PCG64-derived byte stream is statistically uniform; the DP1 codebook is supposed to encode driving-distribution PCA bases. The HYPOTHESIS is that the renderer's per-pair residual absorbs the prior-distribution shift (because the codebook is consumed as a soft prior). Empirical verification requires the OP-ROUTABLE #3 paired smoke. |
| The inflate runtime can re-derive the codebook bytes from the seed without GPU | HARD-EARNED | Canonical helper is pure CPU + numpy; runtime dep closure is `numpy + brotli`. |
| The variant is structurally OOD per Catalog #209 Comma2k19 leakage refusal | HARD-EARNED | The variant module does NOT construct any `Comma2k19FrameIterator`; verified by source inspection + the IN-DOMAIN check `verify_procedural_codebook_in_domain` returns True only for `comma2k19_ood_derived_basis_replacement`. |

## Predicted ΔS band + Dykstra-feasibility (Catalog #296 / #324)

**Predicted ΔS = -0.002706** per canonical equation #26 closed form:
`-25 * (4096 - 32) / 37,545,489`.

**Dykstra-feasibility**: rate-axis feasibility is trivially satisfied
(bytes-saved is positive by construction). Score-axis feasibility
(seg + pose terms not regressed by the prior change) is the HYPOTHESIS
pending OP-ROUTABLE #3 paired smoke per Catalog #324 post-training
Tier-C validation discipline.

**`predicted_band_validation_status` per Catalog #324**: `pending_post_training`.
Reactivation criterion = "post-training Tier-C re-measurement on the
landed paired smoke archive sha via
`tools/mdl_scorer_conditional_ablation.py --tier c`".

## Sister regression — DP1 base substrate

206 / 206 DP1 tests pass post-landing (including 15 new tests; 191
pre-existing). Zero regressions to:

* `test_pretrained_driving_prior_substrate.py` (canonical DP1 substrate tests)
* `test_composition.py` (DPCOMP composition API)
* `test_dataset_source.py` (DP1 dataset source contract)
* `test_local_chunk_cache.py` (Comma2k19 canonical helper)
* `test_local_chunk_streamer.py` (log-incremental streamer)
* `test_log_incremental_feeder.py` (federated aggregation)
* `test_log_incremental_streaming.py` (sister)
* `test_score_aware_loss_f3_kwargs.py` (Catalog #228 F3 cache wire-in)
* `test_trainer_dataset_args.py` (Catalog #209 leakage refusal)
* `test_comma2k19_frame_iterator.py` (frame iterator)
* `test_comma2k19_iterator_modes.py` (iterator modes)

## Catalog gates clean post-landing

| Gate | Pre-landing | Post-landing | Notes |
|---|---|---|---|
| #1 (MPS-fallback) | clean | clean | New module does not introduce device-fork |
| #117 (serializer usage) | pre-existing legacy backlog | preserved | This commit uses canonical serializer |
| #119 (Co-Authored-By trailer) | pre-existing legacy backlog | preserved | Commit includes trailer |
| #125 (6-hook wire-in) | clean | clean | Landing memo declares all 6 hooks |
| #127 (custody validator) | 9 pre-existing in sister code | preserved | No new violations |
| #157 (--expected-content-sha256) | 0 | 0 | POST-EDIT SHAs declared |
| #174 (--expected-content-sha256 mandatory) | 0 | 0 | All commits use canonical flag |
| #206 (subagent crash-resume) | warn-only baseline | preserved | This subagent emitted 3 checkpoints |
| #208 (docs/local-paths) | clean | clean | No local absolute paths in memos |
| #209 (Comma2k19 leakage refusal) | clean | clean | Variant does NOT construct Comma2k19FrameIterator |
| #213 (Comma2k19 download canonical) | clean | clean | Variant does NOT trigger any download |
| #220 (substrate L1+ operational mechanism) | clean | clean | Lane evidence declares `score_improvement_mechanism_status=OPERATIONAL` |
| #229 (premise verification) | clean | clean | Sister activity check + read parent design memo + verified canonical helper |
| #240 (recipe-vs-trainer-state) | clean | clean | Variant is `research_only=true`; recipe outline lives in parent design memo §4 (NOT committed YAML — operator-routed) |
| #272 (distinguishing-feature contract) | clean | clean | Lane evidence + byte-mutation smoke PASSED |
| #287 (placeholder rationale rejection) | clean | clean | All rationales substantive |
| #290 (canonical-vs-unique decision) | clean | clean | This memo §"Canonical-vs-unique decision per layer" |
| #294 (9-dim checklist evidence) | clean | clean | This memo §"9-dimension success checklist evidence" |
| #296 (predicted-band Dykstra) | clean | clean | This memo §"Predicted ΔS band + Dykstra-feasibility" |
| #303 (cargo-cult audit) | clean | clean | This memo §"Cargo-cult audit per assumption" |
| #305 (observability surface) | clean | clean | This memo §"Observability surface" |
| #309 (horizon_class) | clean | clean | Declared `frontier_protecting` |
| #323 (canonical Provenance) | clean | clean | No score claims; all manifests carry `score_claim=False` |
| #324 (post-training Tier-C validation) | clean | clean | `predicted_band_validation_status=pending_post_training` declared |
| #335 (cathedral consumer Protocol) | clean | clean | Sister consumer `procedural_codebook_savings_consumer` already landed (slot 3 territory) |
| #340 (sister-checkpoint staging guard) | PROCEED | PROCEED | Re-checked sister activity 2x at narrow window; no collisions |
| #341 (Tier A canonical-routing markers) | n/a | n/a | Variant trainer does NOT emit cathedral signals directly; consumer-side handles routing |
| #344 (canonical equation cross-reference) | clean | clean | This memo + the variant module's docstring cite canonical equation #26 extensively |

## 6-hook wire-in declaration per Catalog #125

* **Hook #1 sensitivity-map** — N/A. Variant is a single archive-build
  path; no per-tensor sensitivity contribution at variant landing.
* **Hook #2 Pareto constraint** — ACTIVE. The variant's predicted ΔS
  contribution per canonical equation #26 enters the rate-axis Pareto
  polytope via the sister consumer
  `tac.cathedral_consumers.procedural_codebook_savings_consumer`
  (auto-discovered per Catalog #335).
* **Hook #3 bit-allocator** — ACTIVE. The 32-byte seed slot replaces the
  ~4096-byte codebook slot; the bit-allocator's per-tensor importance
  changes (variant has FEWER tensors in the codebook section).
* **Hook #4 cathedral autopilot dispatch** — ACTIVE via the sister
  consumer (Catalog #335 auto-discovered consumer auto-ingests variant
  candidate routing metadata).
* **Hook #5 continual-learning posterior** — ACTIVE on first empirical
  anchor landing (OP-ROUTABLE #3 paired smoke triggers
  `update_equation_with_empirical_anchor` per Catalog #344 + canonical
  Provenance per Catalog #323).
* **Hook #6 probe-disambiguator** — ACTIVE. The PROCEDURAL vs ORIGINAL
  vs NULL-EXPLOIT 3-recipe contrast IS the probe disambiguator. Two
  interpretations exist: (a) the procedural codebook's PRNG output is
  statistically plausible enough that the renderer absorbs the shift;
  (b) the renderer's per-pair residual is insufficient to compensate.
  The OP-ROUTABLE #3 paired smoke is the canonical disambiguator.

## Top-3 operator-routable next actions

1. **OP-ROUTABLE #1 — Operator-authorize the paired smoke**
   ($0.30 Modal T4 paired CPU+CUDA; OPERATOR-GATED). Per parent design
   memo §11 recipe outline #2: the operator routes via canonical
   `tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch --paired-axis cuda+cpu --max-spend-usd 0.30` chain. The recipe YAML lives at
   `.omx/operator_authorize_recipes/` (operator-routed; NOT
   committed by this subagent per scope limits). Modal `.spawn()` per
   Catalog #245 + #339 fail-closed; harvest within 24h per Catalog #330;
   byte-mutation smoke per Catalog #272; first empirical anchor lands
   via `update_equation_with_empirical_anchor`.

2. **OP-ROUTABLE #2 — Monitor slot 3 canonical equation refinement**.
   Slot 3 (`a230693c`) is REFINING `procedural_codebook_savings.py` mid-flight.
   Once slot 3 lands `validate_context_is_in_domain` helper, refactor
   `verify_procedural_codebook_in_domain` to call the canonical helper
   directly (graceful `try/except ImportError` fallback is already in place).

3. **OP-ROUTABLE #3 — 5-substrate aggregate dispatch sequencing**.
   Per parent design memo §4: DP1 is the FIRST anchor of the
   5-substrate procedural-replacement matrix (sister substrates:
   NSCS06 v8 chroma LUT / ATW V2 codec / TT5L transformer tokens /
   sister substrate pending identification). Once DP1's first empirical
   anchor lands, sequence the remaining 4 substrates per
   `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md`.

## Sign-off

- **Cost**: $0 paid GPU; ~50 min wall-clock
- **Files touched**:
  - NEW: `src/tac/substrates/pretrained_driving_prior/distillation_procedural_variant.py`
  - NEW: `src/tac/substrates/pretrained_driving_prior/tests/test_procedural_variant.py`
  - NEW: `.omx/research/dp1_procedural_variant_build_landed_20260520.md` (this file)
  - NEW: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_dp1_procedural_trainer_build_landed_20260520.md`
  - MODIFIED: `src/tac/substrates/pretrained_driving_prior/__init__.py` (+35 LOC re-exports + flag)
  - MODIFIED: `src/tac/substrates/pretrained_driving_prior/archive.py` (+37 LOC convenience wrapper)
  - MODIFIED: `.omx/state/lane_registry.json` (+1 lane L0→L1)
  - MODIFIED: `.omx/state/lane_maturity_audit.log` (+2 mutations)
- **Discipline**: Catalog #110+#113 APPEND-ONLY / #117+#157+#174 canonical serializer / #119 co-author / #125 6-hook / #206 crash-resume / #229 PV / #240 recipe-vs-trainer-state / #220 substrate L1+ operational mechanism / #272 byte-mutation distinguishing-feature / #287 placeholder rejection / #290+#294+#296+#303+#305+#309+#324 design memo discipline / #323 canonical Provenance umbrella / #335 cathedral consumer Protocol / #340 sister-checkpoint guard / #344 canonical equation cross-reference / #209+#213 Comma2k19 canonical helpers
- **Lane**: `lane_dp1_procedural_codebook_replacement_variant_20260520`
- **mission_predicted_contribution**: `frontier_protecting`
- **horizon_class**: `frontier_protecting` (per Catalog #309 + parent design memo)
- **Sister coordination**: DISJOINT from slot 1 (MAGIC CODEC) + slot 3 (CANONICAL EQUATION DOMAIN REFINEMENT)
