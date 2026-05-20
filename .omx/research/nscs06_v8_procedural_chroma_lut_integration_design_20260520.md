---
council_tier: T1
council_attendees: [Working-Group]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "NSCS06 v8 chroma LUT is 4096 bytes (parent memo §4 #1 estimate)"
    classification: CARGO-CULTED
    rationale: "ACTUAL NSCS06 v3 (already landed at commit 4292c8ce2) has chroma palette = NUM_SEGNET_CLASSES * 3 = 15 bytes, not 4096. The '~4 KB chroma LUT' in parent memo refers to a HYPOTHETICAL v8 redesign that does not match current code. Predicted ΔS must be recalculated against either (a) actual current 15-byte palette OR (b) a documented v8 architectural pivot that introduces a 4 KB chroma codebook."
  - assumption: "NSCS06 substrate paradigm is dispatchable for paid empirical smoke"
    classification: HARD-EARNED-EMPIRICALLY-FALSIFIED-PER-CATALOG-307-308
    rationale: "Per parent memo §4 #1 verbatim 'NSCS06 substrate paradigm is currently DEFER per Catalog #307/#308 implementation-falsification; reactivation criterion is the v8 redesign with procedural-codebook replacement for chroma LUT.' This design memo IS that reactivation criterion candidate; it cannot grant its own reactivation."
  - assumption: "Catalog #325 per-substrate symposium recency permits paid dispatch"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Latest NSCS06 symposium dated 2026-05-16 (4 days before today 2026-05-20). Catalog #325 requires ≥14 days OR fresh symposium per the canonical 6-step contract. Current state BLOCKS paid dispatch until 2026-05-30 OR a fresh symposium lands. This design memo is DESIGN-ONLY; it documents the integration architecture pending the gating sequence."
  - assumption: "8-byte chroma seed (existing v3) is sufficient for v8 4 KB hypothetical chroma codebook"
    classification: HARD-EARNED-PARTIALLY (8B sufficient for entropy of 5-class × 3-byte uniform palette; INSUFFICIENT entropy for hypothetical 1024×4 = 4096-byte LUT per Shannon's source-coding theorem; need 32-byte seed for 4096-byte codebook per task description spec)
council_decisions_recorded:
  - "op-routable #1: DESIGN-ONLY memo landed; predicted ΔS recalculated per ACTUAL byte counts (TWO scenarios: scenario A = existing v3 palette; scenario B = hypothetical v8 4 KB LUT)"
  - "op-routable #2: paid empirical smoke BLOCKED by Catalog #325 (symposium <14 days) AND Catalog #307/#308 (NSCS06 paradigm DEFER) — gating sequence documented per §7"
  - "op-routable #3: 5-substrate matrix sibling subagent referenced in parent prompt; sibling has NOT landed at time of this memo (verified empirically via .omx/research/ scan); coordination contract documented per §11"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: nscs06_carmack_hotz_strip_everything_v8_chroma_lut_procedural
deferred_substrate_retrospective_due_utc: "2026-06-19T22:00:00Z"
related_deliberation_ids:
  - feedback_procedural_codebook_generator_build_landed_20260520
  - procedural_codebook_generator_null_exploit_design_20260520
  - grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516
horizon_class: frontier_pursuit
---

<!--
Catalog #344 canonical equation cross-reference:
`procedural_codebook_from_seed_compression_savings_v1` (registered 2026-05-20
via sister commit 5c1af7ba6; registry row count 25 → 26 at landing). This memo
designs the FIRST empirical anchor for that equation per the canonical
accumulation playbook documented in §6 below. The equation's
predicted_only-pending-empirical hypothesis-anchor will be extended via
`tac.canonical_equations.update_equation_with_empirical_anchor` once the
paid smoke per §6 Stage 3 lands.
-->

# NSCS06 v8 Procedural Chroma-LUT Integration Design

**Lane**: `lane_wave_3_nscs06_v8_procedural_chroma_lut_integration_design_20260520` L1 (impl_complete=false; design-only; substrate code unchanged per task scope)

**Parent op-routable**: PROCEDURAL-CODEBOOK BUILD landing memo `feedback_procedural_codebook_generator_build_landed_20260520.md` Top-3 op-routable #1 ("NSCS06 v8 chroma-LUT paired smoke — first empirical anchor; build minimum-viable NSCS06 v8 substrate with chroma LUT replaced by `derive_codebook_from_seed(seed_bytes=32B, output_shape=(1024,4), dtype=np.uint8, generator_kind='pcg64')`")

**Operator framing (verbatim 2026-05-20)**: WAVE-3-NSCS06-V8-PROCEDURAL-CHROMA-LUT-INTEGRATION-DESIGN task description: "Design the FIRST empirical anchor for canonical equation `procedural_codebook_from_seed_compression_savings_v1`"

**Scope**: DESIGN-ONLY. Per task description SCOPE LIMITS: NO production code mutation, NO recipe creation, NO paid dispatch, NO push to origin, NO TaskList mutation.

## §1 — Substrate state survey (Catalog #229 premise verification)

The task description prompts a design against the parent memo §4 #1 enumeration which states "NSCS06 v8 chroma LUT (~4 KB constants → ~32 B seed = ~3.968 KB saved → predicted ΔS `25 * 3968 / 37545489 = -0.00264`)". Empirical state of the substrate at landing 2026-05-20:

**Current NSCS06 v3 (already landed at commit 4292c8ce2, symposium 2026-05-16)**:

- `src/tac/substrates/nscs06_carmack_hotz_strip_everything/archive.py`: schema v3 ALREADY supports `CHROMA_SCHEMA_VERSION_SEEDED_CHROMA = 3` with `CHROMA_SEED_BYTES = 8`
- Existing `emit_chroma_palette_seed()` + `expand_chroma_seed_to_palette()` already use `numpy.random.PCG64` (sister of `tac.procedural_codebook_generator.hash_seed_codebook_generator.expand_seed_to_codebook`)
- Existing chroma palette shape: `(NUM_SEGNET_CLASSES, CHROMA_BYTES_PER_CLASS) = (5, 3) = 15 bytes`
- Other LUT/codebook candidates in v3 archive: `PALETTE` (16 bytes uint8) + `CDF` table (`NUM_SEGNET_CLASSES × (PALETTE_SIZE+1) × 2 = 5 × 17 × 2 = 170 bytes`)

**Critical premise correction**: the parent memo §4 #1 estimate of "~4 KB chroma LUT" does NOT match the current NSCS06 v3 architecture. The 4 KB figure is HYPOTHETICAL for a v8 redesign with a substantially larger chroma codebook.

**Two distinct scenarios** for the integration design:

| Scenario | Source | N_codebook | K_seed | Bytes saved | Predicted ΔS |
|----------|--------|-----------:|-------:|------------:|-------------:|
| A: existing v3 palette migration to canonical helper | code 4292c8ce2 | 15 | 8 | 7 | −0.00000466 |
| B: hypothetical v8 4 KB chroma codebook with 32 B seed | task description + parent §4 | 4096 | 32 | 4064 | −0.00271 |
| C: v3 chroma seed migration FROM numpy.PCG64 TO canonical 3-PRNG-kind helper | architectural | 0 byte savings | 0 byte savings | 0 | 0 (canonical-helper-routing pure refactor; observability-only) |

Scenario A is structurally too small to register as a meaningful empirical anchor. Scenario C is canonical-helper-routing refactor (Catalog #335 sister discipline; the existing v3 code uses sister `hash_seed_codebook_generator`; migration to the new `seed_derived_codebook` would be observability-only with no byte savings). **Scenario B is the only one matching the task description's predicted ΔS = -0.00271 and registered canonical equation `procedural_codebook_from_seed_compression_savings_v1` per-substrate prediction `_NSCS06_V8_PREDICTED_DELTA_S = -25 * (4096-32) / 37_545_489 = -0.00270749`** (registered in `src/tac/canonical_equations/procedural_codebook_savings.py:76-78`).

This design memo proceeds against **Scenario B** with explicit acknowledgment that the v8 redesign introducing the 4 KB chroma codebook is a substrate-engineering bet that has NOT been validated empirically (Tier-C density unknown; HNeRV parity discipline lesson L1 score-aware-training constraint un-tested at the 4 KB chroma codebook scale).

## §2 — Architectural integration sketch (substrate-engineering scope; design-only)

### 2.1 — v8 chroma-codebook structural extension

Hypothetical v8 grammar EXTENDS v3 with a chroma codebook (4 KB) generated from a 32-byte seed. The seed lives INSIDE archive.zip per the canonical compliance citation chain (`upstream/evaluate.py` line 63 + Catalog #213 sister + Catalog #272 byte-mutation smoke). The 4 KB codebook is derived at inflate time via `tac.procedural_codebook_generator.seed_derived_codebook.derive_codebook_from_seed(seed_bytes=<32B archive member>, output_shape=(1024, 4), dtype=np.uint8, generator_kind="pcg64")`.

```python
# (DESIGN-ONLY snippet; NOT landed in substrate code)

# NEW v8 fields in CarmackHotzArchive dataclass:
# chroma_codebook_seed: bytes  # 32 bytes; archive-charged
# chroma_codebook: np.ndarray  # (1024, 4) uint8; DERIVED at inflate from seed

# v8 archive grammar header EXTENSION:
# CHROMA_CODEBOOK_SEED_LEN(2) u16  # = 32
# CHROMA_CODEBOOK_SHAPE_H(2)  u16  # = 1024
# CHROMA_CODEBOOK_SHAPE_W(2)  u16  # = 4
# CHROMA_CODEBOOK_SEED       ...   # 32 bytes pcg64 seed

# inflate.py BUILD:
# from tac.procedural_codebook_generator import derive_codebook_from_seed
# chroma_codebook = derive_codebook_from_seed(
#     seed_bytes=archive.chroma_codebook_seed,
#     output_shape=(1024, 4),
#     dtype=np.uint8,
#     generator_kind="pcg64",
# )
```

### 2.2 — Distinguishing-feature integration contract (Catalog #272)

Per the gate's canonical 4-step contract:
1. `distinguishing_feature_name`: `chroma_codebook_procedural_pcg64_seed_derived`
2. `distinguishing_bytes_path`: `archive.chroma_codebook_seed` (32 bytes at fixed offset)
3. `inflate_consumer_function`: `tac.substrates.nscs06_carmack_hotz_strip_everything.inflate.synthesize_chroma_from_codebook` (HYPOTHETICAL; does not exist yet)
4. `byte_mutation_smoke_passes`: PENDING — `tools/verify_distinguishing_feature_byte_mutation.py` MUST verify that mutating ANY byte of the 32-byte seed produces RENDERED FRAME CHANGES per the inflate path

This is the canonical "mutate seed → re-inflate → frames change" smoke. If the smoke FAILS (i.e. inflate renders identical frames regardless of seed bytes), the v8 design is a `research_only=true` substrate per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable.

### 2.3 — Why pcg64 (not xorshift or lcg)

Per the canonical helper's `DEFAULT_GENERATOR_KIND` rationale: pcg64 has highest entropy + statistical rigor (O'Neill 2014). For a 4 KB codebook the additional ~50 LOC inflate runtime cost (~80 LOC pcg64 vs ~25 LOC lcg) is acceptable within the HNeRV parity discipline lesson L4 budget (≤100 LOC inflate ideal; ≤200 LOC with waiver). xorshift would also suffice; pcg64 chosen per task description spec.

## Canonical-vs-unique decision per layer

(Catalog #290; section §3 of this memo)

| Layer | Canonical helper available? | Decision | Rationale |
|-------|----------------------------|----------|-----------|
| archive grammar (header layout) | NO (substrate-specific) | UNIQUE | CH06 grammar is substrate-unique; v8 adds 3 fields to v3's 36-byte header |
| chroma codebook derivation | YES (`tac.procedural_codebook_generator.derive_codebook_from_seed`) | ADOPT CANONICAL | Newly landed; serves the substrate; no principled mismatch |
| scorer-preprocess routing | YES (`tac.substrates._shared.score_aware_common.score_pair_components`) | ADOPT CANONICAL | Per HNeRV parity L8 + Catalog #164 |
| auth-eval routing | YES (`tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`) | ADOPT CANONICAL | Per Catalog #226 + #223 + #221 |
| inflate device selection | YES (`tac.substrates._shared.inflate_runtime.select_inflate_device`) | ADOPT CANONICAL | Per Catalog #205 |
| EMA / score-aware loss | YES (Catalog discipline) | ADOPT CANONICAL | Per CLAUDE.md "EMA — NON-NEGOTIABLE" + "eval_roundtrip — NON-NEGOTIABLE" |
| Modal NVML env block | YES (`tac.deploy.modal.runtime` constants) | ADOPT CANONICAL | Per Catalog #244 |
| Modal mount manifest | YES (`tac.deploy.modal.mount_manifest.build_training_image`) | ADOPT CANONICAL | Per Catalog #153 |
| 6-hook wire-in | YES (Catalog #125 + #305 + #294 etc.) | ADOPT CANONICAL | Substrate engineering must inherit ALL 6 hooks per discipline |

Operator-facing decision: **adopt-canonical-by-default across all infrastructure layers; substrate-engineering is concentrated in the v8 grammar extension + the chroma-synthesis logic that consumes the codebook**. This minimizes the substrate-engineering surface while satisfying the UNIQUE-AND-COMPLETE-PER-METHOD non-negotiable (substrate engineering happens ONCE per architecture class; bolt-ons happen many times — the canonical helpers are the bolt-ons here, the substrate engineering is the v8 chroma codebook architectural pivot).

## 9-dimension success checklist evidence

(Catalog #294; section §4 of this memo)

| Dimension | Evidence at design time | Action gates |
|-----------|------------------------|--------------|
| 1. UNIQUENESS | v8 4 KB chroma codebook is structurally distinct from v3 (15 B palette) | empirical Tier-C density measurement post-training (Catalog #324) |
| 2. BEAUTY + ELEGANCE | seed-derived 32 B → 4 KB codebook via pcg64 is PR101-style 30-sec-reviewable | inflate.py LOC budget ≤200 (HNeRV L4 waiver); current v3 inflate.py = 221 LOC (already over; v8 must NOT add LOC over budget) |
| 3. DISTINCTNESS | distinct from sister substrate ATW V2 / TT5L / DP1 procedural-codebook surfaces (different score-relevant byte slots) | sister 5-substrate matrix design sibling subagent will document orthogonality |
| 4. RIGOR | this memo's Assumption-Adversary verdicts surface 4 cargo-culted assumptions (parent memo's 4 KB estimate; NSCS06 paradigm dispatchability; symposium recency; seed entropy sufficiency) | premise verification (Catalog #229) + cargo-cult audit per §10 |
| 5. OPTIMIZATION PER TECHNIQUE | per layer analysis in §3 (UNIQUE-AND-COMPLETE-PER-METHOD non-negotiable) | empirical paired smoke per §6 |
| 6. STACK-OF-STACKS-COMPOSABILITY | composes additively with PR101 fec6 (Wave 3) and ATW V2 (Wave 3) procedural-replacement smokes; aggregate ΔS prediction `-0.013` per parent memo | per-substrate empirical anchors before composition smoke |
| 7. DETERMINISTIC REPRODUCIBILITY | seed-derived bytes are byte-stable per `tac.procedural_codebook_generator.derive_codebook_from_seed` contract (cross-platform via explicit little-endian) | per Catalog #272 byte-mutation smoke |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | inflate-time codebook derivation adds <1 ms wall-clock (4096 bytes × pcg64 8 bytes/step = 512 steps × ~1 μs/step = ~0.5 ms) | empirical wall-clock measurement at smoke |
| 9. OPTIMAL MINIMAL CONTEST SCORE | predicted ΔS = −0.00271 per canonical equation; competitive contribution to aggregate `-0.013` prediction across 5 substrates | empirical paired smoke per §6 |

## Cargo-cult audit per assumption

(Catalog #303; section §5 of this memo)

Operator's hard-earned-vs-cargo-culted classification per the addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`):

1. **"NSCS06 v8 chroma codebook is 4096 bytes"** — CARGO-CULTED (parent memo §4 #1 estimate; NOT empirically validated against any v8 design memo). UNWIND-TEST: actually measure what byte budget the v8 score-aware training requires; the 4 KB figure should be derived from per-class entropy × NUM_SEGNET_CLASSES × spatial-block-size empirical study, not assumed.
2. **"pcg64 is the optimal PRNG kind for chroma codebook derivation"** — CARGO-CULTED (chosen per task description spec; no per-substrate empirical study). UNWIND-TEST: compare xorshift vs lcg vs pcg64 paired smoke; if xorshift produces equally low ΔS at lower LOC budget, prefer xorshift per HNeRV parity L4.
3. **"seed-derived bytes preserve score relative to trained bytes"** — CARGO-CULTED (assumed via Shannon's source-coding theorem but UN-TESTED on the contest's per-class chroma sensitivity surface). UNWIND-TEST: per Catalog #272 byte-mutation smoke + Tier-C density post-training validation per Catalog #324.
4. **"single 32-byte seed has sufficient entropy for 4 KB codebook"** — HARD-EARNED-PARTIALLY (32-byte seed = 256 bits entropy; 4 KB codebook contains at most 4096 × 8 = 32,768 bits if every byte is independent; the seed is sufficient ONLY if the codebook has structural redundancy of ratio ≥ 128:1 which is consistent with uniform_int8 RNG output). UNWIND-TEST: if v8 codebook has structured content (e.g. spatially-coherent chroma values), 32-byte seed may be insufficient and need replacement with a small generative-model architecture per `tac.substrates.pretrained_driving_prior` (DP1) sister discipline.
5. **"Catalog #325 ≥14-day window can be bypassed via operator-frontier-override"** — HARD-EARNED-EMPIRICALLY-VERIFIED (per CLAUDE.md "Mission alignment" Consequence 1; the override IS the canonical escape hatch but it requires operator-verbatim quote in `council_override_rationale` frontmatter field). UNWIND-TEST: do NOT invoke override for this substrate — the ≥14-day window is a RIGOR-OVERHEAD constraint, but NSCS06's prior empirical anchors (v6 = 105.15 contest-CUDA; v7 = 58.89; v8 hypothetical with chroma-LUT replacement) suggest paradigm-falsification risk that the ≥14-day window's deliberation-cadence requirement was designed to surface.
6. **"v8 chroma codebook is the highest-EV procedural-replacement target"** — CARGO-CULTED (parent memo §4 ranked NSCS06 first by predicted ΔS contribution; ranking is observability-only). UNWIND-TEST: a sister 5-substrate matrix would compare NSCS06 v8 vs ATW V2 vs TT5L vs DP1 vs sister; if a sister substrate has higher empirical ΔS contribution at lower engineering cost, prefer the sister.
7. **"This design memo can be the first empirical anchor for the canonical equation"** — CARGO-CULTED (a DESIGN memo is not an EMPIRICAL anchor; the equation's first empirical anchor requires Stage 1+2 of the playbook in §6 to LAND with measured Stage 3 ΔS). UNWIND-TEST: this memo is the integration design; the first empirical anchor is whatever paid smoke lands AFTER Catalog #325 + #307/#308 gating clears.

## §6 — Empirical-anchor accumulation playbook (CANONICAL)

The canonical sequence to extend `procedural_codebook_from_seed_compression_savings_v1` with NSCS06 v8 as its first empirical anchor:

**Pre-Stage 0**: Catalog #325 + Catalog #307/#308 gating MUST clear before ANY paid dispatch.
  - Catalog #325: fresh per-substrate symposium memo with verdict ∈ {PROCEED, PROCEED_WITH_REVISIONS} within 14-day window (canonical 6-step contract). Latest 2026-05-16 symposium expires for THIS purpose on 2026-05-30 OR is superseded by a fresh symposium.
  - Catalog #307/#308: kill verdict on NSCS06 paradigm must distinguish PARADIGM-LEVEL FALSIFICATION (paradigm KILL) vs IMPLEMENTATION-LEVEL FALSIFICATION (specific v6/v7 implementation FALSIFIED but paradigm INTACT). v7 result (105.15 → 58.89 = 44% improvement) per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" suggests paradigm-intact-iterative-rescue Tier 1. v8 chroma-LUT-procedural-replacement is the next iteration in that trajectory.

**Stage 1: ORIGINAL archive build + eval (BASELINE)**
  - Build NSCS06 v8 ORIGINAL archive (v8 grammar WITH 4 KB chroma codebook STORED as raw bytes, NOT seed-derived)
  - Dispatch paired CPU+CUDA `upstream/evaluate.py` per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
  - Record: `score_v8_original_contest_cpu` + `score_v8_original_contest_cuda` + `archive_bytes_v8_original` + `archive_sha256_v8_original`
  - Tag: `[contest-CPU GHA Linux x86_64]` + `[contest-CUDA T4 or higher]` per Catalog #127 custody routing
  - Estimated cost: $0.50 paired (Modal T4 smoke 100 epochs)

**Stage 2: PROCEDURAL archive build + eval (TREATMENT)**
  - Build NSCS06 v8 PROCEDURAL archive (v8 grammar WITH 32-byte seed REPLACING 4 KB chroma codebook; chroma codebook DERIVED at inflate via `derive_codebook_from_seed`)
  - SAME training pipeline + same chroma codebook bytes at compress time (seeded so the trained model converges to the same codebook bytes the seed produces)
  - Dispatch paired CPU+CUDA eval on the new archive
  - Record: `score_v8_procedural_contest_cpu` + `score_v8_procedural_contest_cuda` + `archive_bytes_v8_procedural` + `archive_sha256_v8_procedural`
  - Verify: `archive_bytes_v8_procedural ≈ archive_bytes_v8_original - 4064` (within compression-stream-boundary noise)

**Stage 3: ΔS empirical measurement**
  - Compute: `delta_s_empirical_cpu = score_v8_procedural_contest_cpu - score_v8_original_contest_cpu`
  - Compute: `delta_s_empirical_cuda = score_v8_procedural_contest_cuda - score_v8_original_contest_cuda`
  - Predicted: `delta_s_predicted = -0.00271` (canonical equation)
  - Compute residual: `residual_cpu = delta_s_empirical_cpu - delta_s_predicted`; `residual_cuda = delta_s_empirical_cuda - delta_s_predicted`
  - Compute z-score: `residual_zscore = |residual| / sigma_predicted` (where `sigma_predicted` derives from the equation's domain_of_validity ± measurement noise estimate)

**Stage 4: Canonical equation extension via `update_equation_with_empirical_anchor`**
  - Construct `EmpiricalAnchor` with:
    - `anchor_id`: `nscs06_v8_chroma_codebook_paired_cpu_cuda_first_empirical_anchor_<UTC>`
    - `measurement_utc`: paired-eval completion timestamp
    - `inputs`: `{substrate_id: "nscs06_v8_chroma_codebook_procedural", n_codebook_bytes: 4096, k_seed_bytes: 32, generator_kind: "pcg64", archive_sha256_v8_original, archive_sha256_v8_procedural}`
    - `predicted_output`: `{delta_s: -0.00270749}`
    - `empirical_output`: `{delta_s_cpu: <measured>, delta_s_cuda: <measured>, archive_bytes_saved: <measured>}`
    - `residual`: `max(|residual_cpu|, |residual_cuda|)` (worst-axis residual)
    - `provenance`: per Catalog #323 canonical Provenance with axis-tag + hardware-substrate
  - Append via `tac.canonical_equations.update_equation_with_empirical_anchor("procedural_codebook_from_seed_compression_savings_v1", anchor)` per Catalog #344 sister discipline
  - Verify: registry row count 26 (preserved; the canonical equation gets a NEW anchor row appended to its `empirical_anchors` tuple, NOT a new equation row)

**Stage 5 (downstream): canonical-equation auto-recalibration**
  - The canonical equation's `next_recalibration_trigger = RECALIBRATE_ON_NEW_ANCHORS` per Catalog #344 fires automatically on the next consumer call (`tac.canonical_equations.recalibrate_equation_from_anchors`)
  - Updated `predicted_vs_empirical_residual` dict surfaces the new residual for the autopilot cathedral_consumers loop
  - Sister consumer `procedural_codebook_savings_consumer` automatically re-routes the updated predicted ΔS to the cathedral autopilot ranker per Catalog #335 + Catalog #341 Tier A markers

## §7 — Operator-routable decision matrix

Per Catalog #325 + Catalog #324 + Catalog #307/#308 + paid-dispatch gating cascade:

```
┌─ Decision: should we paid-dispatch NSCS06 v8 procedural chroma-LUT smoke? ─┐
│                                                                             │
│  1. Does NSCS06 v8 substrate code exist?                                    │
│     NO  → DECISION: BUILD substrate variant first (research_only:true       │
│          scaffold per CLAUDE.md "Substrate scaffolds MUST be COMPLETE       │
│          OR RESEARCH-ONLY"). Spawn sister subagent for build with           │
│          dispatch_enabled:false initial recipe. Cost: $0 (build only).      │
│     YES → proceed to step 2.                                                │
│                                                                             │
│  2. Has Catalog #325 per-substrate symposium fired within 14 days?          │
│     NO  → DECISION: DEFER pending symposium re-cadence. Spawn               │
│          per-substrate sextet pact (Shannon LEAD / Dykstra CO-LEAD /        │
│          Rudin CO-LEAD / Daubechies CO-LEAD / Yousfi / Fridrich /           │
│          Contrarian / Assumption-Adversary + grand council attendees        │
│          per topic: Quantizr / Hotz / Selfcomp / Ballé / MacKay).           │
│          Symposium memo MUST satisfy the canonical 6-step contract.         │
│     YES → proceed to step 3.                                                │
│                                                                             │
│  3. Does Catalog #307/#308 paradigm-vs-implementation classification        │
│     for NSCS06 substrate class permit v8 reactivation?                      │
│     NO  → DECISION: DEFER pending higher-tier deliberation (T3 grand        │
│          council per CLAUDE.md "Recursive adversarial review protocol").    │
│          Per CLAUDE.md "Forbidden premature KILL": NSCS06 paradigm is       │
│          DEFERRED-pending-research, never KILLED. Reactivation criteria     │
│          (v8 chroma-LUT-procedural-replacement) is THIS design memo.        │
│     YES → proceed to step 4.                                                │
│                                                                             │
│  4. Is the per-substrate Tier-C density measurement (Catalog #324)          │
│     post-training validation status `validated_post_training` OR            │
│     `pending_post_training` with reactivation criteria pinned?              │
│     NO  → DECISION: DEFER pending Tier-C validation; recipe MUST            │
│          declare `predicted_band_validation_status: pending_post_training`  │
│          + reactivation criterion: "post-training Tier-C re-measurement     │
│          on archive sha256 <pending>".                                      │
│     YES → proceed to step 5.                                                │
│                                                                             │
│  5. Has the Catalog #272 byte-mutation smoke verified that seed mutation    │
│     produces frame-level changes?                                           │
│     NO  → DECISION: BUILD byte-mutation smoke (cheap CPU; ~$0.05            │
│          local or ~$0.10 Modal CPU). Smoke validates that the seed         │
│          bytes are NOT a no-op per Catalog #220 + Catalog #139 sister.      │
│     YES → proceed to step 6 — READY TO PAIRED-SMOKE.                        │
│                                                                             │
│  6. ALL prerequisites satisfied → DECISION: READY TO PAIRED-SMOKE.          │
│     Estimated cost: $0.50-1.00 paired CPU+CUDA Modal T4/A10G smoke           │
│     100 epochs per Stage 1+2 of §6 playbook. Operator authorizes via        │
│     `tools/operator_authorize.py --recipe substrate_nscs06_v8_procedural_   │
│     chroma_lut_modal_paired_dispatch.yaml`.                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Current state at landing of this memo (2026-05-20)**:

| Step | Verdict |
|------|---------|
| 1. Substrate code exists? | NO — v3 exists (8B seed → 15B palette); v8 hypothetical 4 KB codebook does NOT exist in code |
| 2. Catalog #325 symposium ≥14 days? | NO — latest symposium 2026-05-16 (4 days ago); window expires for THIS purpose 2026-05-30 |
| 3. Catalog #307/#308 paradigm permits v8 reactivation? | DEFER — NSCS06 paradigm currently DEFER per parent memo §4 #1; this memo IS the reactivation criterion candidate; cannot grant own reactivation |
| 4. Tier-C density validated? | N/A (substrate does not yet exist) |
| 5. Byte-mutation smoke fired? | N/A (substrate does not yet exist) |
| 6. READY TO PAIRED-SMOKE? | **NO. STATUS = BUILD pending Steps 1+2+3.** |

**Operator-routable next-action (Top-3 in §13)**:
1. Spawn sister subagent for v8 substrate BUILD (Step 1) — substrate code lands as `research_only:true` scaffold; estimated $0 GPU + ~2-4h subagent wall-clock.
2. Schedule per-substrate symposium for NSCS06 v8 chroma-LUT procedural-replacement integration (Step 2) — T2 sextet pact at minimum; T3 grand council if paradigm-class question surfaces per Catalog #307/#308. Estimated 1 council session.
3. Reserve $0.50-1.00 paired Modal T4 smoke budget for Stages 1+2 of §6 playbook (Step 6) — paid dispatch only AFTER Steps 1+2+3+4+5 clear.

## Observability surface

(Catalog #305; section §8 of this memo)

Per CLAUDE.md "Max observability — non-negotiable" 6-facet definition:

1. **Inspectable per layer**: every layer's input + output captured at runtime:
   - Compress-time chroma-LUT bytes BEFORE seed search (in trainer's archive_build_pass)
   - Seed bytes EMITTED (32 B; observed via `archive.chroma_codebook_seed`)
   - Codebook DERIVED at inflate (4096 B; observed via `derive_codebook_from_seed` return value)
   - Per-frame RGB rendered from codebook + class labels (observed via inflate.py output)

2. **Decomposable per signal**:
   - `final_score = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37_545_489`
   - Per-axis decomposition: `d_seg_v8_procedural - d_seg_v8_original` (seg-axis ΔS)
   - `sqrt(10 * d_pose_v8_procedural) - sqrt(10 * d_pose_v8_original)` (pose-axis ΔS)
   - `25 * (archive_bytes_v8_procedural - archive_bytes_v8_original) / 37_545_489` (rate-axis ΔS — this is the predicted source of the savings)

3. **Diff-able across runs**:
   - `archive_sha256_v8_original` vs `archive_sha256_v8_procedural` (byte-level diff)
   - Rendered frames byte-for-byte diff via `tools/verify_distinguishing_feature_byte_mutation.py`
   - Score-component breakdown JSON diff via `upstream/evaluate.py` per-pair output

4. **Queryable post-hoc**:
   - Modal call_id ledger row per Catalog #245 (`.omx/state/modal_call_id_ledger.jsonl`)
   - Recipe persisted at `.omx/operator_authorize_recipes/substrate_nscs06_v8_procedural_chroma_lut_modal_paired_dispatch.yaml` (DESIGN; not landed)
   - Per-anchor canonical equation row persisted at `.omx/state/canonical_equations_registry.jsonl`

5. **Cite-able**:
   - Each archive sha256 cited inline in landing memo
   - Each Modal call_id cited in `modal_call_id_ledger.jsonl`
   - Each commit_sha + author + UTC timestamp captured per CLAUDE.md "Apples-to-apples evidence discipline"

6. **Counterfactual-able**:
   - Catalog #272 byte-mutation smoke: mutate 1 byte of seed → re-inflate → frames change (proves operational consumption)
   - Catalog #139 packet compiler no-op detector: mutate at random offset → score changes (proves bytes consumed)
   - Catalog #105 no-op provenance: provenance.json records mutate-and-detect verdict

## §9 — Predicted ΔS band per Catalog #324 (post-training Tier-C validation discipline)

**Predicted ΔS band**: `[-0.00271, -0.00271]` (point estimate from canonical equation; bounds tightening pending empirical residual measurement)

**predicted_band_validation_status**: `pending_post_training` per the canonical contract.

**Reactivation criterion**: post-training Tier-C density re-measurement on landed archive sha256 (TBD post-Stage-1 + Stage-2 of §6) via `tools/mdl_scorer_conditional_ablation.py --tier c`.

**Justification**: the predicted ΔS = -0.00271 is derived from `_NSCS06_V8_PREDICTED_DELTA_S = -25 * (4096-32) / 37_545_489` in the canonical equation. This is a CANONICAL-FORMULA-DERIVED prediction with NO Tier-C density input. Per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density (the phantom-predicted-band trap)": this prediction is NOT derived from random-init Tier-C density (which would be a phantom prediction); it is derived from the canonical contest rate-term formula assuming byte-count savings of `N_codebook - K_seed`. The prediction is admissible at Tier-C-pending status because:

1. The bytes-saved figure (4064) is structurally guaranteed if the v8 grammar replaces 4096-byte raw codebook with 32-byte seed (modulo compression-stream-boundary noise).
2. The score change comes ENTIRELY from the rate-term axis (assuming seg + pose distortions are preserved by seed-derived codebook IF the trainer converges to identical bytes the seed produces).
3. The Tier-C validation post-training measures whether the trainer DID converge to those bytes — a phantom-Tier-C-prediction would assume convergence; the actual paired smoke MEASURES it.

**Dykstra-feasibility check** (Catalog #296):
- Constraint 1 (rate ≤ R): satisfied trivially (saving bytes only loosens the rate constraint)
- Constraint 2 (d_seg ≤ S): MUST be empirically validated — if seed-derived chroma codebook produces different per-class chroma values than the trained codebook, d_seg may increase. The Dykstra-feasibility prediction assumes trainer-codebook == seed-derived-codebook BYTE-FOR-BYTE (which IS the canonical formula's assumption).
- Constraint 3 (d_pose ≤ P): satisfied trivially (chroma codebook does not affect pose axis directly; PoseNet operates on YUV6 which derives from luma + chroma)
- ALTERNATING PROJECTIONS: the projection onto rate-feasible region is direct (subtract 4064 bytes); the projection onto seg-feasible region requires the trainer convergence assumption; the projection onto pose-feasible region is satisfied automatically. The Dykstra alternating-projections sequence converges immediately if the trainer assumption holds.

## §10 — Sister-coordination contract (Catalog #230 + Catalog #302)

**Sister 5-substrate matrix design sibling** (referenced in task description as "sister-DISJOINT from 5-substrate matrix design"):

- Empirical state at landing: the 5-substrate matrix design memo does NOT exist yet at `.omx/research/` (verified via empirical scan 2026-05-20T22:54Z)
- Expected sibling scope per parent memo §4: 5 substrates (NSCS06 v8 chroma LUT + ATW V2 codec + TT5L + DP1 + sister) with aggregate predicted ΔS -0.013
- Expected sibling artifact: `.omx/research/procedural_codebook_5_substrate_matrix_design_20260520.md` OR equivalent

**Sister-collision verdict**: DISJOINT
- This memo is NSCS06-V8-ONLY (substrate-specific integration design)
- Sister memo is CROSS-SUBSTRATE MATRIX (aggregate composition design)
- Different memo paths; different scope; no Catalog #248 conflict-marker risk

**Coordination contract** (if sibling lands during my work):
- Sister 5-substrate matrix MAY reference this NSCS06 v8 design as the canonical reference for the NSCS06 row
- Sister 5-substrate matrix MAY supersede this memo's canonical-vs-unique decisions IF cross-substrate composition imposes constraints not visible at NSCS06-only scope
- This memo MAY be EXTENDED (APPEND-ONLY per Catalog #110/#113 HISTORICAL_PROVENANCE) with a `## Sister 5-substrate matrix integration notes` section if the sibling lands and produces routing decisions affecting NSCS06 v8

**Sister-checkpoint guard verdict** (Catalog #340): PROCEED at landing (no sister activity on memo target path within 6-hour window per `tools/check_sister_files_recently_landed.py`)

## §11 — Discipline citations (binding)

- Catalog #117 + #157 + #174 commit serializer with POST-EDIT --expected-content-sha256
- Catalog #119 Co-Authored-By trailer (auto-appended by serializer)
- Catalog #125 6-hook wire-in declaration per §12
- Catalog #185 META drift sister regression
- Catalog #186 catalog-claim committed via canonical serializer (N/A; no new catalog # claimed in this DESIGN-ONLY memo)
- Catalog #206 crash-resume discipline (4 checkpoints emitted)
- Catalog #229 premise-verification (9 PVs HARD-EARNED-VERIFIED per §1 substrate state survey)
- Catalog #287 placeholder-rationale rejection (memo uses substantive rationales)
- Catalog #290 canonical-vs-unique decision per layer §3
- Catalog #292 per-deliberation assumption surfacing in council_assumption_adversary_verdict frontmatter
- Catalog #294 9-dimension success checklist evidence §4
- Catalog #296 Dykstra-feasibility for predicted band §9
- Catalog #300 v2 council frontmatter (T1 design memo at substrate-engineering scope)
- Catalog #303 cargo-cult audit per assumption §5
- Catalog #305 observability surface §8
- Catalog #309 horizon_class: `frontier_pursuit` in frontmatter
- Catalog #323 canonical Provenance (every score-claim in §6 playbook carries axis + hardware substrate + evidence_grade)
- Catalog #324 predicted_band_validation_status: pending_post_training per §9
- Catalog #325 per-substrate symposium recency: latest 2026-05-16 = 4 days ago; BLOCKS paid dispatch until 2026-05-30 OR fresh symposium; DESIGN-ONLY does not trigger gate
- Catalog #340 sister-checkpoint guard: PROCEED verdict at write time per §10
- Catalog #344 canonical equation cross-reference: HTML comment after frontmatter
- Catalog #110 + #113 APPEND-ONLY HISTORICAL_PROVENANCE: this memo is NEW (no prior body to preserve); future Sister 5-substrate matrix integration notes section MAY be APPENDED only
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable: gating cascade per §7 STRICTLY enforced
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable: §7 step 2 requires fresh symposium with canonical 6-step contract
- CLAUDE.md "Forbidden premature KILL without research exhaustion": NSCS06 paradigm DEFER, not KILL; v8 chroma-LUT procedural-replacement IS the canonical reactivation path
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": §3 canonical-vs-unique decisions per layer enforce per-substrate optimal engineering

## §12 — 6-hook wire-in declaration per Catalog #125

| Hook | Status | Surface |
|------|--------|---------|
| #1 sensitivity-map | ACTIVE-DESIGN | the 32-byte seed slot is the canonical sensitivity surface for the chroma-LUT byte-saving direction; per-byte master-gradient anchors on the seed bytes inform the per-X plan via `tac.master_gradient_consumers` |
| #2 Pareto constraint | ACTIVE-DESIGN | §9 Dykstra-feasibility analysis articulates rate-axis loosening + seg-axis preservation conditional on trainer convergence; explicit Pareto-feasible-region intersection check |
| #3 bit-allocator | ACTIVE-DESIGN | 4064 bytes hoisted from chroma codebook to seed is a per-substrate bit-allocator decision that feeds the substrate's overall byte budget |
| #4 cathedral autopilot dispatch | ACTIVE-DESIGN | sister cathedral_consumer `procedural_codebook_savings_consumer` (auto-discovered per Catalog #335) emits Tier A markers for this substrate's predicted ΔS; the equation's `predicted_vs_empirical_residual` post-Stage-3 anchor feeds the autopilot ranker |
| #5 continual-learning posterior | ACTIVE-DESIGN | canonical equation `procedural_codebook_from_seed_compression_savings_v1` extension per Stage 4 of §6 playbook; `tac.canonical_equations.update_equation_with_empirical_anchor` is the canonical posterior write surface |
| #6 probe-disambiguator | ACTIVE-DESIGN | the canonical disambiguator IS Catalog #272 byte-mutation smoke + Catalog #324 post-training Tier-C validation + Catalog #325 per-substrate symposium; collectively they disambiguate among (a) seed-derived bytes preserve score, (b) seed-derived bytes degrade score (canonical formula prediction was wrong), (c) seed-derived bytes are no-op (trainer did not converge) |

## §13 — Top-3 operator-routable next-actions

1. **Spawn sister subagent for NSCS06 v8 substrate BUILD** (Step 1 of §7 cascade). Scope: extend `src/tac/substrates/nscs06_carmack_hotz_strip_everything/{archive,codec,inflate}.py` with v8 schema (CH06_SCHEMA_VERSION_PROCEDURAL_CHROMA_CODEBOOK = 4 OR equivalent); add `derive_codebook_from_seed` integration at inflate; substrate trainer enters `research_only=true` state with `dispatch_enabled=false` recipe. Estimated cost: $0 GPU + ~2-4h subagent wall-clock. NO paid dispatch in this lane.

2. **Schedule fresh NSCS06 v8 per-substrate symposium per Catalog #325** (Step 2 of §7 cascade). Scope: T2 sextet pact (Shannon LEAD / Dykstra CO-LEAD / Rudin CO-LEAD / Daubechies CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary) + grand council attendees per topic (Quantizr / Hotz / Selfcomp / Ballé / MacKay / Time-Traveler). Symposium memo MUST satisfy the canonical 6-step contract per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium". Estimated cost: 1 council session + ~1h memo writing. Resolves Catalog #325 14-day window.

3. **Reserve $0.50-1.00 paired Modal T4/A10G smoke budget for Stages 1+2 of §6 playbook** (Step 6 of §7 cascade). Pre-conditions: Steps 1+2+3+4+5 of §7 all cleared. Recipe: `.omx/operator_authorize_recipes/substrate_nscs06_v8_procedural_chroma_lut_modal_paired_dispatch.yaml` (DESIGN; lands with substrate BUILD in op-routable #1). Verifies `procedural_codebook_from_seed_compression_savings_v1` first empirical anchor per §6 Stage 4.

## §14 — Blockers

1. **NSCS06 v8 substrate code does NOT exist** — task description's 4096-byte chroma LUT is HYPOTHETICAL; current v3 has 15-byte palette. Operator-routable #1 (substrate BUILD) is the prerequisite.

2. **Catalog #325 14-day symposium window NOT satisfied** — latest 2026-05-16 = 4 days ago. Operator-routable #2 (fresh symposium) is the prerequisite.

3. **Catalog #307/#308 paradigm-vs-implementation classification for NSCS06 reactivation NOT explicit** — parent memo §4 #1 states NSCS06 paradigm is DEFER per Catalog #307/#308; this memo cannot grant its own reactivation. T3 grand council deliberation per CLAUDE.md "Council hierarchy: 4-tier protocol" is the canonical resolution path.

4. **Sister 5-substrate matrix design sibling memo does NOT exist** at landing — coordination contract per §10 documented in case sibling lands during operator-review window.

## §15 — Closing

This DESIGN-ONLY memo documents the integration architecture for NSCS06 v8 chroma-LUT procedural-replacement as the FIRST EMPIRICAL ANCHOR for canonical equation `procedural_codebook_from_seed_compression_savings_v1`. The canonical 5-stage empirical-anchor accumulation playbook (§6) is the operator-routable sequence for converting this design into a measurable per-substrate predicted-vs-empirical residual.

**Status**: DESIGN ONLY. Substrate code NOT modified. Recipe NOT created. Paid dispatch NOT authorized. Gating cascade per §7 BLOCKS paid dispatch on 3 Catalog gates (#325 + #307/#308 + #324). Operator-routable next-actions per §13 are sequenced to clear the gating cascade.

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion"**: NSCS06 substrate paradigm is DEFER, not KILL. This memo IS the canonical reactivation criterion candidate per parent memo §4 #1. The v8 chroma-LUT procedural-replacement architecture is the canonical research path forward; its empirical validation extends the canonical equation and informs the autopilot cathedral_consumers loop without prematurely killing the substrate paradigm.

**End of design memo.**
