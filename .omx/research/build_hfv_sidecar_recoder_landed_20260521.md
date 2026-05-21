<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:cross_session_reference_to_canonical_frontier_pointer_anchors_fec6_pr101_cpu_0_192051_and_pr106_format0d_cuda_0_205330_per_canonical_frontier_pointer_json_2026-05-15_through_2026-05-21_plus_predicted_rate_only_delta_minus_0_0093_per_OVERNIGHT_S_landing_memo_decision_6 -->
<!-- FORMALIZATION_PENDING:overnight_x2_canonical_recoder_builder_landing_memo_references_existing_canonical_equation_356_lifecycle_per_OVERNIGHT_P_registration_commit_4f8e754cf_no_new_equation_registration_required_per_catalog_344_structural_disambiguator_scope -->
---
schema: subagent_landing_memo_v1
topic: build_hfv_sidecar_recoder_canonical_lossless_byte_reducer_overnight_x2
created_at_utc: 2026-05-21T14:42:00Z
author: claude:overnight_x2_build_hfv_sidecar_recoder_20260521
lane_id: lane_overnight_x2_build_hfv_sidecar_recoder_20260521
mission_contribution: frontier_breaking_enabler
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
dispatch_attempted: false
paid_dispatch_attempted: false
evidence_grade: "[prediction]"
predicted_band_validation_status: pending_post_training
council_tier: T1
council_attendees:
  - Carmack       # MVP-first phasing arbiter per CLAUDE.md amendment be125b878
  - Assumption-Adversary  # cargo-cult classification per Catalog #292
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "A canonical HFV sidecar recoder is necessary AND sufficient to unblock the OVERNIGHT-S Path 3 prerequisite for HFV cascade PR110-rebase"
    classification: HARD-EARNED-WITH-CAVEAT
    rationale: |
      The recoder IS necessary (OVERNIGHT-S Decision 6 op-routable
      explicitly names it as Path 3 prerequisite). It is NOT sufficient
      for frontier-lowering on its own: the predicted rate-only delta
      is approximately -0.0093 per OVERNIGHT-S T3 §5.5 closed-form (and
      empirically -0.0159 via this build's combined-strategy smoke on
      live seed_top16; even better than the OVERNIGHT-S prediction).
      However the component-gain hurdle for HFV cascade PR110-rebase
      to be FRONTIER-LOWERING is +0.0161 per the OVERNIGHT-S calculation
      (the dense HFV1 sidecar produced -0.144 / -0.148 component
      regression vs fec6 / pr106 frontier). A purely byte-shrinking
      recoder cannot bridge a component-gain hurdle. The recoder IS a
      structural prerequisite that unblocks the COMBINED test (sensitivity-
      weighted seed + recoded sidecar + paired Modal smoke); the test
      itself remains DEFER-pending-research per CLAUDE.md "Forbidden
      premature KILL without research exhaustion" until both Builder 1
      (sister OVERNIGHT-X1 sensitivity-weighted foveation_params
      generator) AND a component-gain breakthrough land.
  - assumption: "The recoder should target ~10,000 bytes (~58% reduction) per OVERNIGHT-S Path 3 spec"
    classification: HARD-EARNED-CONSERVATIVELY-EXCEEDED
    rationale: |
      OVERNIGHT-S Decision 3 specified "Sidecar recoder (60% byte
      reduction; 24KB -> ~10KB)". Empirical PV during this build:
      combined strategy achieves 99.73% reduction on synthetic identity
      (24016 -> 64 bytes) AND 99.48% reduction on LIVE seed_top16
      (24016 -> 126 bytes). The OVERNIGHT-S target was conservative;
      brotli quality=11 on a 1200-row sparse-pair sidecar achieves
      effectively the information-theoretic floor (the dense bytes
      contain only ~1-16 unique 20-byte rows so the Kolmogorov complexity
      is O(unique_rows * 20) plus a small framing overhead).
  - assumption: "The recoder MUST be bit-identical lossless per CLAUDE.md 'Apples-to-apples evidence discipline'"
    classification: HARD-EARNED-STRUCTURALLY-ENFORCED
    rationale: |
      Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #229
      PV: any recoder that produces frame-rendering-different output is
      structurally INVALID for contest compliance because the inflate
      runtime's downstream foveation transform consumes the row values
      directly (alpha, radius, power, origin_x, origin_y). A lossy
      recoder would shift component scores in unpredictable ways.
      Therefore: the canonical recoder MUST be bit-identical lossless;
      the ``--verify-roundtrip`` flag defaults ON; per-component
      quantization is INTENTIONALLY excluded from the canonical strategy
      list (it would be lossy at 16-bit per component; alpha/radius/power
      need float32 precision for frame-rendering parity). Structurally
      enforced via the round-trip test parametrization across all 3
      canonical lossless strategies (entropy_brotli / sparse_delta /
      combined).
council_decisions_recorded:
  - "Decision 1: Build tools/build_hfv_sidecar_recoder.py (~700 LOC) implementing 3 canonical lossless encoding strategies (entropy_brotli / sparse_delta / combined) with paired INVERSE decoders + bit-identical round-trip verification via --verify-roundtrip flag (default ON)"
  - "Decision 2: New canonical HFRC magic (b'HFRC') distinct from HFV1 (dense) and HFV2 (sparse-pair); inflate runtime magic-byte dispatch is the canonical Carmack pattern (the data tells you what it is)"
  - "Decision 3: Canonical equation #356 (hfv2_sparse_pair_sidecar_replacement_savings_v1) is the predicted-rate-savings reference; closed-form deltaS = -25 * (N_dense - N_recoded) / 37_545_489 per OVERNIGHT-P registration commit 4f8e754cf"
  - "Decision 4: Catalog #287/#323 canonical Provenance defaults threaded: axis_tag='[prediction]' + promotable=False + score_claim=False + canonical_equation_reference=#356; NOT a score claim until paired CUDA + CPU empirical anchors land per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE'"
  - "Decision 5: 35 dedicated tests landed in src/tac/tests/test_build_hfv_sidecar_recoder.py covering: (a) bit-identical round-trip across 3 strategies on synthetic identity + sparse fixtures; (b) size-reduction (output < input for all canonical strategies); (c) combined strategy meets 10KB target on both synthetic and 16-sparse fixtures; (d) graceful failure on malformed input (truncated, wrong magic, size mismatch); (e) deterministic output (sha256 deterministic); (f) Catalog #287/#323 Provenance discipline; (g) canonical equation #356 formula; (h) smoke mode runs without paid GPU; (i) --dry-run does not write output; (j) --no-verify-roundtrip flag; (k) live seed_top16 integration; (l) strategy-byte dispatch; (m) varint round-trip"
  - "Decision 6: $0 spent (Carmack MVP-first Step 1 FREE local CPU smoke + design + tests + landing memo only); no paid GPU dispatch fired; bit-identical round-trip structurally verified empirically on both synthetic + LIVE fixtures"
  - "Decision 7: Path 3 prerequisite OVERNIGHT-S Decision 6 (b) SATISFIED; cascade re-fire (sensitivity-weighted seed via sister OVERNIGHT-X1 + recoded sidecar via THIS build + paired Modal smoke) remains DEFER per CLAUDE.md 'Forbidden premature KILL' until either (i) OVERNIGHT-X1 lands its Builder 1 + a component-gain breakthrough proves the combined test can bridge the +0.0161 component-gain hurdle, OR (ii) operator pivots per OVERNIGHT-S Decision 6 (d) to substrate-class-shift cascade (DP1 + NSCS06 v8 + 5-substrate matrix + STC residual sidecar over A1)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: null
substrate_alias: null
---

# OVERNIGHT-X2: Canonical HFV Sidecar Recoder LANDED

## Summary

Built `tools/build_hfv_sidecar_recoder.py` (canonical lossless HFV sidecar
recoder) per operator directive 2026-05-21 *"Build the 2/3 unbuilt"* and
OVERNIGHT-S landing memo `pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_landed_20260521.md`
Path 3 prerequisite (commit `079edcfdd`).

The recoder shrinks the canonical `foveation_params.bin` HFV1 sidecar
from the dense 24,016-byte representation to a compact HFRC recoded
sidecar via three canonical lossless strategies (`entropy_brotli`,
`sparse_delta`, `combined`). Bit-identical round-trip is structurally
enforced.

## Carmack MVP-first 5-step compliance

Per CLAUDE.md amendment `be125b878` ("Carmack MVP-first phasing"):

1. **FREE local CPU smoke first** — DONE. The synthetic-fixture smoke
   runs locally in ~1ms; the LIVE seed_top16 fixture smoke runs in
   ~2ms. No paid GPU dispatch fired.

2. **Smoke MUST falsifiably challenge** — DONE. Predicted size
   reduction was ~58% (OVERNIGHT-S Decision 3). Empirical result: 99.73%
   reduction on synthetic identity (24016 → 64 bytes) and 99.48% on
   LIVE seed_top16 (24016 → 126 bytes). The falsifying outcome would
   have been (a) recoded sidecar LARGER than original, OR (b) round-trip
   failure, OR (c) failure to meet the ≤10,000-byte target. NONE of
   these triggered.

3. **Catalog #344 reference** — DONE. Canonical equation #356
   (`hfv2_sparse_pair_sidecar_replacement_savings_v1`) IN-DOMAIN per
   OVERNIGHT-P registration commit `4f8e754cf`. Closed-form:
   `predicted_archive_bytes_saved = -25 * (N_dense - N_recoded) / 37_545_489`.

4. **Land verdict in same commit batch** — DONE. Recoder source +
   tests + landing memo all in one canonical-serializer commit.

5. **Re-route operator priority queue** — DONE via this landing memo
   "Operator-routable next-step" section.

## Empirical results

### Synthetic fixture (identity foveation, 1200 frames, 1 sparse pair)

| Strategy        | Output bytes | Reduction | Under 10KB target | Round-trip |
|-----------------|--------------|-----------|-------------------|------------|
| entropy_brotli  | 96           | 99.60%    | YES               | YES        |
| sparse_delta    | 60           | 99.75%    | YES               | YES        |
| combined        | 64           | 99.73%    | YES               | YES        |

### LIVE fixture (`seed_top16_component_hardpairs/foveation_params.bin`; 16 sparse pairs)

| Field                       | Value                                                                |
|-----------------------------|----------------------------------------------------------------------|
| Strategy                    | combined                                                             |
| Input bytes                 | 24,016                                                               |
| Output bytes                | 126                                                                  |
| Reduction                   | 99.48%                                                               |
| Under 10KB target           | YES                                                                  |
| Round-trip verified         | YES (sha256 match: `f1dbcf02973957b4`...)                            |
| Predicted rate savings      | `-0.0159 [prediction]` per canonical equation #356                   |
| Sha256 input                | `f1dbcf02973957b4f4e30bd2638629051f4a986c3010bed32c954f500c6d1551`   |
| Sha256 output               | `f4c9ed4c61f37489ec9f943a373a4f54ffc424bd2aca0042fb8bca80c77aa365`   |
| Canonical artifact          | `experiments/results/lane_overnight_x2_build_hfv_sidecar_recoder_20260521/seed_top16_recoder_smoke_report_combined.json` |
| Recoded sidecar artifact    | `experiments/results/lane_overnight_x2_build_hfv_sidecar_recoder_20260521/seed_top16_foveation_params_recoded_combined.bin` |

### Predicted-vs-OVERNIGHT-S delta calibration

OVERNIGHT-S Decision 3 predicted rate-only delta `-0.0093 [prediction]`
based on 60% byte reduction (24KB → 10KB target). This build's combined
strategy on LIVE seed_top16 achieves 99.48% reduction (24KB → 126
bytes), yielding **`-0.0159 [prediction]`** — ~1.7× better than the
OVERNIGHT-S prediction. Sister to canonical equation #356; would be a
candidate for empirical anchor registration if/when paired CUDA + CPU
empirical landing happens.

## Catalog #325 6-step contract

### 1. Cargo-cult audit per assumption (Catalog #303)

| Assumption                                                            | Classification | Unwind path                                                      |
|-----------------------------------------------------------------------|----------------|------------------------------------------------------------------|
| HFV1 dense foveation params are highly redundant (97-100% identity)   | HARD-EARNED    | Empirical PV across 5 live foveation_params.bin variants confirms |
| brotli quality=11 captures near-Kolmogorov-complexity entropy         | HARD-EARNED    | Information-theoretic: 1-16 unique 20-byte rows → ~50-150 bytes  |
| Per-component float32 precision MUST be preserved bit-identically     | HARD-EARNED    | Apples-to-apples evidence + downstream foveation transform contract |
| Magic-byte dispatch (HFV1/HFV2/HFRC) is the canonical pattern         | HARD-EARNED    | Carmack canonical "the data tells you what it is" pattern         |
| Recoder ALONE bridges the HFV cascade frontier-tying hurdle            | CARGO-CULTED   | OVERNIGHT-S T3 §5.5: byte reduction is necessary but NOT sufficient; component-gain hurdle remains +0.0161 |

### 2. 9-dimension success checklist evidence (Catalog #294)

| Dim | Evidence                                                                |
|-----|-------------------------------------------------------------------------|
| 1. UNIQUENESS                       | Byte-level recoder distinct from HFV2 sparse-pair canonical (semantic-level) |
| 2. BEAUTY + ELEGANCE                | ~700 LOC; 30-sec-reviewable per HNeRV parity L7; one-purpose CLI |
| 3. DISTINCTNESS                     | NEW HFRC magic + 3 strategies + strategy-byte auto-dispatch |
| 4. RIGOR                            | 35/35 tests PASS + bit-identical round-trip structurally enforced + canonical Provenance |
| 5. OPTIMIZATION PER TECHNIQUE       | brotli quality=11 max; varint LEB128; struct-pack canonical |
| 6. STACK-OF-STACKS COMPOSABILITY    | Composes with HFV1 dense + HFV2 sparse (orthogonal byte-vs-semantic axes) |
| 7. DETERMINISTIC REPRODUCIBILITY    | Deterministic-output test pinned; sha256 stable across runs |
| 8. EXTREME OPTIMIZATION             | <2ms wall-clock for 24KB input; ~100× faster than typical Modal dispatch overhead |
| 9. OPTIMAL MINIMAL CONTEST SCORE    | Rate-only delta predicted `-0.0159 [prediction]` — necessary but not sufficient per AA verdict |

### 3. Observability surface (Catalog #305)

- **Inspectable per layer**: parse_hfv1 + recode_foveation_params + decode_recoded_sidecar are independently invocable.
- **Decomposable per signal**: per-strategy verdicts; per-fixture verdicts; verdict dataclass exposes 18 fields.
- **Diff-able across runs**: deterministic-output test pins sha256; runs can be diffed via sha256_output.
- **Queryable post-hoc**: --report-out-json emits canonical machine-readable verdict JSON.
- **Cite-able**: every verdict carries canonical_equation_reference + lane_id + sha256_input.
- **Counterfactual-able**: --no-verify-roundtrip + --dry-run flags enable counterfactual probes.

### 4. Sextet pact deliberation

T1 Working Group (Carmack + Assumption-Adversary) — sextet-pact not
required at T1 per CLAUDE.md "Council hierarchy: 4-tier protocol".
Carmack MVP-first arbitrates the build vs design tradeoff; Assumption-
Adversary classifies the 5 surfaced assumptions HARD-EARNED-vs-CARGO-
CULTED per Catalog #292 frontmatter.

### 5. Per-substrate reactivation criteria pinned

This build is a TOOL not a substrate (no `lane_class=substrate_engineering`
or `lane_class=substrate_class_shift` declaration). Reactivation criteria
for the SISTER cascade re-fire (HFV cascade PR110-rebase + sensitivity-
weighted seed + recoded sidecar + paired Modal smoke):

1. **OVERNIGHT-X1 Builder 1 lands** (sister DISJOINT): sensitivity-weighted
   foveation_params generator that produces NON-uniform seed bytes per
   M_contest gradients.
2. **Component-gain breakthrough**: empirical evidence that the combined
   (sensitivity-seed + recoded) cascade closes the +0.0161 component-gain
   hurdle vs the OVERNIGHT-K uniform-seed empirical baseline (CPU 0.336724
   / CUDA 0.353177). Per CLAUDE.md "Forbidden premature KILL without
   research exhaustion" + OVERNIGHT-S Decision 4.
3. **OR pivot per OVERNIGHT-S Decision 6 (d)** to substrate-class-shift
   cascade (DP1 + NSCS06 v8 + 5-substrate matrix + STC residual sidecar
   over A1) per T3 symposium §5 Tier-1.

### 6. Catalog #324 post-training Tier-C validation discipline

This build emits PREDICTED rate-only delta carrying `[prediction]` axis
tag + `promotable=False` + `score_claim=False` per Catalog #287/#323.
`predicted_band_validation_status: pending_post_training` — the actual
contest score delta requires paired Linux x86_64 + NVIDIA empirical
landing on the actual archive bytes. Catalog #324 sister recipe
declaration not yet relevant (the recoder is a TOOL not a substrate
recipe).

## 6-hook wire-in declaration (Catalog #125)

- **Hook 1 (sensitivity-map contribution)**: N/A — recoder is a byte-
  reducer at the inflate-sidecar surface; downstream sensitivity-map
  consumers consume the PR110 archive surface, not the foveation-params
  internal representation.
- **Hook 2 (Pareto constraint)**: ACTIVE — adds the byte-level entropy
  axis to the meta-Lagrangian / Pareto solver per CLAUDE.md
  "Meta-Lagrangian/Pareto solver" non-negotiable. The recoded sidecar
  bytes ARE the optimization variable on the rate axis; canonical
  equation #356 IS the closed-form constraint.
- **Hook 3 (bit-allocator hook)**: ACTIVE — the recoder IS a per-tensor
  importance allocator at the byte level; future bit-allocator consumers
  can route through `recode_foveation_params(..., strategy='combined')`
  to allocate rate budget.
- **Hook 4 (cathedral autopilot dispatch hook)**: PENDING — the recoder
  is a TOOL (not yet wired into cathedral autopilot ranker); future
  cathedral consumer wrapping per Catalog #335 canonical contract would
  surface this build as an auto-discoverable dispatch candidate.
- **Hook 5 (continual-learning posterior update)**: PENDING — when paired
  CUDA + CPU empirical anchors land, canonical equation #356 can absorb
  this build's predicted-vs-empirical residual via
  `tac.canonical_equations.registry.update_equation_with_empirical_anchor`.
- **Hook 6 (probe-disambiguator)**: N/A — the recoder produces ONE
  canonical bit-identical lossless output per strategy; no defensible
  alternative interpretations.

## Sister coherence verification

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #340 sister-
checkpoint guard:

- **OVERNIGHT-X1** (`overnight_x1_*`; lane `lane_overnight_x1_build_sensitivity_weighted_foveation_params_generator_20260521`): builds `tools/build_sensitivity_weighted_foveation_params_generator.py` (companion Builder 1 for the OVERNIGHT-S combined-cascade re-fire). **DISJOINT** from my scope (`tools/build_hfv_sidecar_recoder.py`).
- **OVERNIGHT-V** (NSCS06 v8 Phase 2 BUILD): substrate trainer + recipe + ledger. **DISJOINT**.
- **OVERNIGHT-W** (STC residual sidecar over A1 design): research memo only. **DISJOINT**.
- **OVERNIGHT-Y** (STC 3a A1 residual entropy probe build): probe build. **DISJOINT**.
- **OVERNIGHT-R** (DP1 3rd-attempt re-dispatch): different recipe + state. **DISJOINT**.
- **OVERNIGHT-T** (NSCS06 v8 Phase 2 revision 1+4 T1 working group): different memos. **DISJOINT**.
- **OVERNIGHT-U** (PR110 stacking cascade top-5 in-domain to fec6 frontier): different artifact directory. **DISJOINT**.

All sister subagent in-progress checkpoints inspected; zero file overlap
with `tools/build_hfv_sidecar_recoder.py`, `src/tac/tests/test_build_hfv_sidecar_recoder.py`,
`.omx/research/build_hfv_sidecar_recoder_landed_20260521.md`, or
`experiments/results/lane_overnight_x2_build_hfv_sidecar_recoder_20260521/**`.

## Operator-routable next-step

Per OVERNIGHT-S landing memo Decision 6 operator-routable redirect:

**A.** **Wait for sister OVERNIGHT-X1 (Builder 1)** to land its
`tools/build_sensitivity_weighted_foveation_params_generator.py` (in
flight at landing of THIS memo per sister checkpoint inspection). Once
landed: combined-cascade prerequisites (a) + (b) both satisfied; paired
Modal smoke test ready to fire BUT predicted to STILL underperform
frontier per OVERNIGHT-S T3 §5.5 (combined paths +0.05-0.08 above
frontier per linear extrapolation). Component-gain breakthrough required
before paid dispatch is justifiable per Carmack MVP-first.

**B.** **OR pivot per OVERNIGHT-S Decision 6 (d)** to substrate-class-
shift cascade (DP1 + NSCS06 v8 + 5-substrate matrix + STC residual
sidecar over A1) per T3 symposium §5 Tier-1 canonical frontier-lowering
paradigm. THIS recoder remains available as a stack-of-stacks composable
primitive for ANY future cascade involving foveation_params sidecars.

## Discipline checklist

- Catalog #229 PV (read 7 reference files; verified empirical foveation_params shape; verified canonical equation #356 exists in registry)
- Catalog #117/#157/#174 canonical serializer (next commit via tools/subagent_commit_serializer.py with POST-EDIT --expected-content-sha256)
- Catalog #119 Co-Authored-By trailer (auto-appended by canonical serializer)
- Catalog #125 6-hook wire-in declaration (above)
- Catalog #126 lane pre-registered (`lane_overnight_x2_build_hfv_sidecar_recoder_20260521` in prompt + this memo)
- Catalog #206 (3 checkpoints emitted: step 1 PV, step 2 build+tests, step complete with this landing memo)
- Catalog #229 premise verification (35/35 tests PASS; bit-identical round-trip empirically verified on synthetic + live fixtures)
- Catalog #287 placeholder-rationale rejection (NO placeholder `<rationale>` / `<reason>` literals in this memo)
- Catalog #292 per-deliberation assumption surfacing (5 assumptions classified above)
- Catalog #300 council deliberation v2 frontmatter (T1 attendees + verdict + assumption-adversary verdict + decisions recorded)
- Catalog #303 cargo-cult audit per assumption (above)
- Catalog #305 observability surface (above)
- Catalog #316 frontier pointer N/A (no frontier-score claim in this memo; only `[prediction]` rate-only delta)
- Catalog #323 canonical Provenance umbrella (verdict dataclass + JSON output carry axis_tag + promotable + score_claim + canonical_equation_reference)
- Catalog #340 sister-checkpoint guard (sister subagent in-flight checkpoints inspected; zero file overlap)
- Catalog #344 canonical equation #356 IN-DOMAIN reference (per OVERNIGHT-P registration; closed-form formula matches)
- Carmack MVP-first 5-step (above)

## Cost summary

- **GPU**: $0 (Carmack MVP-first Step 1 FREE local CPU smoke + tests + landing memo)
- **Wall-clock**: ~1.2 hours (PV + design + scaffold + 3 strategies + INVERSE decoders + 35 tests + smoke + landing memo)
- **Token usage**: ~moderate (no nested subagent spawning; no Modal/Vast invocation)

## Artifacts

- Source: `tools/build_hfv_sidecar_recoder.py` (~700 LOC)
- Tests: `src/tac/tests/test_build_hfv_sidecar_recoder.py` (35 tests; 100% PASS)
- Canonical smoke report: `experiments/results/lane_overnight_x2_build_hfv_sidecar_recoder_20260521/seed_top16_recoder_smoke_report_combined.json`
- Recoded sidecar artifact: `experiments/results/lane_overnight_x2_build_hfv_sidecar_recoder_20260521/seed_top16_foveation_params_recoded_combined.bin`
- Landing memo: `.omx/research/build_hfv_sidecar_recoder_landed_20260521.md` (this file)
- Lane registry: `lane_overnight_x2_build_hfv_sidecar_recoder_20260521`
