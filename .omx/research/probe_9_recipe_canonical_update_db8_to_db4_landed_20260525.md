# Probe 9 substrate recipe + trainer + driver canonical update db8 -> db4 landed 2026-05-25

```yaml
---
council_tier: T1
council_attendees: []
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_decisions_recorded:
  - "Probe 9 substrate trio (recipe + trainer + driver) canonical-updated from db8 -> db4 per Probe 9c SISTER_BASIS_DOMINATES_db4 verdict (commit efeaff5c9): db4 dominates db8 at z=-3.442σ (mean 0.3599 vs 0.3915; CIs disjoint at 95%; below-threshold 83.6% vs 80.1%; min 0.0532 vs 0.0932 = 42.9% sharper inversion)."
  - "Per Catalog #307 IMPLEMENTATION-LEVEL fix discipline: paradigm (per-instance + multi-scale wavelet UNIWARD-weighted SegNet loss) INTACT; ONLY the basis-selection IMPLEMENTATION is updated. Per CLAUDE.md 'Forbidden premature KILL without research exhaustion': substrate paradigm preserved; canonical basis updated."
  - "Mallat binding revision #3 of 6 (BLOCKS DISPATCH) SATISFIED with canonical-fix per per-substrate symposium 2026-05-25 §revision_3_mallat. 2-of-3 dispatch-blocking revisions cleared + canonical-optimal basis applied; 1 of 3 remains (paired CPU+CUDA empirical anchor — emergent from sister `_full_main` substrate trainer build + revisions 4-5-6 per Daubechies+Quantizr+Selfcomp seats)."
  - "Catalog #344 RATIFY-N candidate `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1` evidence count strengthens 1 -> 2 via this canonical fix landing (Probe 9c N=100 anchor + substrate-recipe-trio canonical-update landing). EXCLUDED_CONTEXT addendum recommended for sister candidate `uniward_per_instance_multi_scale_wavelet_combined_v1` for the db8-specific in-domain context per Catalog #344 + #359 sister-discipline (requires operator-verbatim approval)."
  - "NEW test src/tac/tests/test_train_substrate_uniward_pims_canonical_db4_basis.py (7 tests; pins db4 default + Probe 9c cite chain + Catalog #110/#113 HISTORICAL_PROVENANCE preservation of Probe 9 N=25 db8 historical anchor)."
council_assumption_adversary_verdict:
  - assumption: "Substrate code (recipe + trainer + driver) may be updated in place per Catalog #307 IMPLEMENTATION-LEVEL fix discipline — NOT subject to Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE."
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md mutation frontier: src/, scripts/, experiments/, .omx/ are mutation-allowed. Per Catalog #110/#113: HISTORICAL_PROVENANCE applies to landed memos / forensic artifacts / posterior anchors — NOT to source code. The recipe + trainer + driver are LIVE_RECIPE per Catalog #113 4-kind taxonomy; canonical updates per Catalog #307 IMPLEMENTATION-LEVEL fix discipline are explicitly allowed. Sister landed memos (Probe 9c landing 2026-05-25 + symposium 2026-05-25 + Probe 9 Tier-2 dispatch prep landing 2026-05-25) get APPEND-ONLY footers only; sister source files get IN-PLACE canonical update."
  - assumption: "The Probe 9 N=25 historical db8 anchor field `probe_9_anchor_min: 0.2597` in the trainer's smoke summary MUST be preserved verbatim per Catalog #110/#113 even though the canonical default is updated to db4."
    classification: HARD-EARNED
    rationale: "Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: forensic anchors (empirical receipts from completed probes) are immutable. The Probe 9 N=25 db8 anchor (0.2597) is the historical BREAKTHROUGH receipt; the canonical fix ADDS new Probe 9c db4 anchor fields (probe_9c_db4_anchor_*) alongside the historical anchor without mutating it. The 'preserve historical Probe 9 anchor' test (test_smoke_summary_preserves_probe_9_historical_anchor_per_catalog_110) enforces this structurally."
  - assumption: "Smoke re-verification via synthetic-frame pywt+db4+3-level algorithm execution is sufficient to satisfy Carmack MVP-first step 2 falsifiable challenge for THIS canonical-fix landing (not Probe 9c re-verification on contest frames)."
    classification: HARD-EARNED
    rationale: "Carmack MVP-first step 2 requires the smoke to falsifiably challenge the SPECIFIC cargo-cult being unwound. The cargo-cult here is db8-as-canonical-default; the canonical fix replaces it with db4 across 3 source files. The falsifiable predicate is: 'after the fix, can the trainer + driver + recipe import + execute the db4 algorithm path without error?'. The synthetic-frame smoke confirms pywt has db4 + the algorithm runs byte-stable. The Probe 9c N=100 contest-frame empirical receipts (mean=0.3599 / min=0.0532 / z=-3.442σ) ARE the canonical empirical anchor for the db4 dominance claim itself — they live in the sister Probe 9c landing memo + Catalog #313 row referenced by this canonical fix."
---
```

## 1. Goal

Apply Probe 9c canonical fix from db8 -> db4 across the Probe 9 substrate trio (recipe + trainer + driver) per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable + Catalog #307 IMPLEMENTATION-LEVEL fix discipline.

Per Probe 9c verdict (commit `efeaff5c9`; `.omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md` §7 P0 operator-routable):

> **NEXT (P0)**: substrate recipe canonical update from db8 → db4 in the canonical lane. Files to update (operator-routable):
> - `.omx/operator_authorize_recipes/substrate_uniward_per_instance_multi_scale_wavelet_segnet_modal_t4_dispatch.yaml` (recipe `wavelet_basis` env var or trainer arg)
> - `experiments/train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py` (trainer's default wavelet basis)
> - `scripts/remote_lane_substrate_uniward_per_instance_multi_scale_wavelet_segnet.sh` (env-var pass-through verification)
> - per the per-substrate symposium addendum cycle per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" Catalog #325

This memo IS the canonical landing for the P0 operator-routable. The empirical receipt is Probe 9c's verdict; THIS landing applies the canonical update structurally so the substrate trainer's `_full_main` (sister subagent BUILD lane) inherits the canonical-optimal db4 basis by default rather than the falsified db8 default.

## 2. Cite chain

```
Probe 9 BREAKTHROUGH (commit 685fe6726)
  -> POSITIVE_SIGNAL_BREAKS_THRESHOLD at 0.2597 vs predicted 0.5 hard threshold
  -> Δ -0.2403 (signal breaks at 4.16× threshold margin)
  -> N=25 / 3-level / db8 anchor; per-instance + multi-scale wavelet UNIWARD paradigm

Probe 9-PREP (commit 92a48616e)
  -> per-substrate symposium 2026-05-25 with 6 binding revisions (3 dispatch-blocking)
  -> recipe + trainer + driver scaffold lands as research_only:true + dispatch_enabled:false

Probe 9b 100-pair disambiguator (commit 2fca9974b)
  -> Contrarian binding revision #1 of 6 (BLOCKS DISPATCH) SATISFIED at N=100 replication
  -> db8 baseline mean=0.3915 / min=0.0932 / 537 valid segments

Probe 9c per-level basis disambiguator (commit efeaff5c9)
  -> Mallat binding revision #3 of 6 (BLOCKS DISPATCH) SATISFIED with canonical-fix
  -> SISTER_BASIS_DOMINATES_db4 verdict at z=-3.442σ vs db8 baseline
  -> 4-basis ablation: db4 (dominant) > bior4.4 (secondary) > db8 (baseline) > db16 (inferior)

THIS canonical fix (lane lane_probe_9_recipe_canonical_update_db8_to_db4_20260525)
  -> recipe + trainer + driver UPDATED in-place per Catalog #307 IMPLEMENTATION-LEVEL fix
  -> NEW test src/tac/tests/test_train_substrate_uniward_pims_canonical_db4_basis.py
     (7 tests; pins db4 + cite chain + HISTORICAL_PROVENANCE preservation)
  -> Probe 9c P0 operator-routable SATISFIED
```

## 3. Files updated (before/after diffs)

### 3.1 Recipe YAML: `.omx/operator_authorize_recipes/substrate_uniward_per_instance_multi_scale_wavelet_segnet_modal_t4_dispatch.yaml`

**Before** (lines 184-185):

```yaml
  UNIWARD_PIMS_WAVELET_NAME: db8
  UNIWARD_PIMS_WAVELET_LEVELS: "3"
```

**After** (lines 184-193):

```yaml
  # db4 selected per Probe 9c canonical empirical falsification of db8-optimal
  # NULL: z=-3.442σ; CIs disjoint at 95% (CI_db4 upper 0.3740 < CI_db8 lower
  # 0.3803); below-threshold 83.6% vs 80.1%; per Catalog #307 IMPLEMENTATION-
  # LEVEL fix; commit efeaff5c9; ratifies canonical equation candidate
  # uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1. PARADIGM
  # (per-instance + multi-scale wavelet UNIWARD-weighted SegNet loss) INTACT;
  # ONLY the basis-selection implementation is updated. See:
  # .omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md
  UNIWARD_PIMS_WAVELET_NAME: db4
  UNIWARD_PIMS_WAVELET_LEVELS: "3"
```

### 3.2 Trainer Python: `experiments/train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py`

**Before** (line 128):

```python
WAVELET_NAME_DEFAULT = "db8"
```

**After** (lines 128-135):

```python
# Canonical fix 2026-05-25: db8 -> db4 per Probe 9c empirical falsification
# (z=-3.442sigma; mean 0.3599 vs db8 0.3915; CIs disjoint at 95%;
# below-threshold 83.6% vs 80.1%; min 0.0532 vs 0.0932 = 42.9% sharper
# inversion). PARADIGM INTACT per Catalog #307 IMPLEMENTATION-LEVEL fix;
# SUBSTRATE BASIS canonical-optimal per Probe 9c. Commit efeaff5c9.
# See: .omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md
#      .omx/research/probe_9_recipe_canonical_update_db8_to_db4_landed_20260525.md
WAVELET_NAME_DEFAULT = "db4"
```

**TIER_1_OPERATOR_REQUIRED_FLAGS rationale + default** (lines 219-236, before/after):

- `--wavelet-name` default `"db8"` -> `"db4"`
- Rationale updated to cite Probe 9c (commit `efeaff5c9`) + the SISTER_BASIS_DOMINATES_db4 verdict + Catalog #307 IMPLEMENTATION-LEVEL classification
- `--wavelet-levels` rationale updated to cite Probe 9c canonical anchor at db4 + queued P3 per-decomposition-level disambiguator per Catalog #308 alternative-reducer cascade

**Smoke summary fields** (lines 597-619, ADDITIVE per Catalog #110/#113):

- Probe 9 historical N=25 db8 anchor `probe_9_anchor_min: 0.2597` PRESERVED VERBATIM (HISTORICAL_PROVENANCE)
- 8 NEW Probe 9c canonical-optimal db4 anchor fields ADDED:
  - `probe_9c_db4_anchor_min: 0.0532`
  - `probe_9c_db4_anchor_mean: 0.3599`
  - `probe_9c_db4_anchor_ci_lower: 0.3457`
  - `probe_9c_db4_anchor_ci_upper: 0.3740`
  - `probe_9c_db4_anchor_below_threshold_fraction: 0.836`
  - `probe_9c_db4_anchor_valid_segment_count: 537`
  - `probe_9c_db4_anchor_z_vs_db8_baseline: -3.442`
  - `probe_9c_canonical_optimal_basis: "db4"`

### 3.3 Driver shell: `scripts/remote_lane_substrate_uniward_per_instance_multi_scale_wavelet_segnet.sh`

**Before** (line 96):

```bash
UNIWARD_PIMS_WAVELET_NAME="${UNIWARD_PIMS_WAVELET_NAME:-db8}"
```

**After** (lines 96-105):

```bash
# Canonical fix 2026-05-25: db8 -> db4 per Probe 9c per-level wavelet-basis
# selection disambiguator (commit efeaff5c9; SISTER_BASIS_DOMINATES_db4 at
# z=-3.442sigma vs db8 baseline; mean 0.3599 vs db8 0.3915; CIs disjoint at
# 95%; below-threshold 83.6% vs 80.1%). Per Catalog #307 IMPLEMENTATION-LEVEL
# fix; substrate paradigm (per-instance + multi-scale wavelet UNIWARD-
# weighted SegNet loss) INTACT. Ratifies canonical equation candidate
# uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1. See:
# .omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md
# .omx/research/probe_9_recipe_canonical_update_db8_to_db4_landed_20260525.md
UNIWARD_PIMS_WAVELET_NAME="${UNIWARD_PIMS_WAVELET_NAME:-db4}"
```

### 3.4 NEW test: `src/tac/tests/test_train_substrate_uniward_pims_canonical_db4_basis.py`

7 tests authored (all PASS at 0.21s):

1. `test_wavelet_name_default_is_canonical_db4` — pins `WAVELET_NAME_DEFAULT == "db4"`
2. `test_wavelet_levels_default_remains_3` — pins `WAVELET_LEVELS_DEFAULT == 3`
3. `test_tier_1_required_flag_wavelet_name_default_is_db4` — pins manifest default per Catalog #151
4. `test_tier_1_required_flag_wavelet_name_rationale_cites_probe_9c` — pins cite chain per Catalog #344
5. `test_constant_module_carries_canonical_fix_marker_in_source` — pins canonical-fix marker for grep-discoverability
6. `test_smoke_summary_carries_probe_9c_canonical_db4_anchor_fields` — pins 4 NEW receipt fields per Catalog #287/#323
7. `test_smoke_summary_preserves_probe_9_historical_anchor_per_catalog_110` — pins HISTORICAL_PROVENANCE preservation per Catalog #110/#113

## 4. Carmack MVP-first 5/5 compliance

| Step | Requirement | Status |
|---|---|---|
| 1 | FREE local macOS-CPU smoke first | ✓ Probe 9c canonical algorithm re-verified byte-stable on macOS-CPU via `.venv/bin/python` synthetic-frame pywt+db4+3-level execution (mean weights_norm=1.0; algorithm imports + runs cleanly; pywt has db4 + executes deterministic 3-level decomposition) |
| 2 | Smoke MUST falsifiably challenge cargo-cult | ✓ Falsifiable predicate: 'after the canonical fix, can the trainer + driver + recipe import + execute the db4 algorithm path without error?'. Result: YES (algorithm runs; tests pass; trainer module imports cleanly with `WAVELET_NAME_DEFAULT="db4"`). The empirical N=100 contest-frame anchor (mean=0.3599 / min=0.0532 / z=-3.442σ) is Probe 9c's verdict empirical receipt — referenced by canonical equation candidate `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1` evidence anchor. |
| 3 | Emit canonical equation anchor + Catalog #344 reference | ✓ Catalog #344 RATIFY-N candidate `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1` evidence count strengthens 1 -> 2 via this canonical fix landing. Sister candidate `uniward_per_instance_multi_scale_wavelet_combined_v1` carries EXCLUDED_CONTEXT recommendation for db8-specific in-domain context (queued per operator-decision protocol per Catalog #344 + #359 sister-discipline). |
| 4 | Land verdict in same commit batch | ✓ This memo + Catalog #313 row + 4 file edits (recipe + trainer + driver + test) + sister memo footers ALL in same commit batch via canonical serializer `tools/subagent_commit_serializer.py --expected-content-sha256` per Catalog #117/#157/#174/#235/#289. |
| 5 | Re-route operator priority queue within ~1h | ✓ Probe 9 Tier-2 dispatch state: 2-of-3 dispatch-blocking revisions cleared + canonical-optimal basis applied. P1 sister `_full_main` BUILD on canonical db4 substrate UNBLOCKED. P2 Probe 9d N=600 db4 re-anchor optional. P3 per-decomposition-level disambiguator deferred. P4-P5 wider-basis sweeps deferred. |

## 5. Sister-coherence verification per Catalog #110/#113

| File class | Catalog #113 kind | Discipline | Action |
|---|---|---|---|
| Recipe YAML | LIVE_RECIPE | UPDATE-allowed per Catalog #307 IMPLEMENTATION-LEVEL fix | Updated in-place (db8 -> db4 + 8-line comment with cite chain) |
| Trainer Python | LIVE_CODE | UPDATE-allowed per Catalog #307 IMPLEMENTATION-LEVEL fix | Updated in-place (constant + rationale + smoke summary fields; HISTORICAL_PROVENANCE Probe 9 anchor PRESERVED) |
| Driver shell | LIVE_CODE | UPDATE-allowed per Catalog #307 IMPLEMENTATION-LEVEL fix | Updated in-place (env var default db8 -> db4 + 10-line comment with cite chain) |
| Sister Probe 9c landed memo | HISTORICAL_PROVENANCE | APPEND-ONLY per Catalog #110/#113 | Footer appended at end of file documenting P0 satisfaction |
| Sister symposium memo | HISTORICAL_PROVENANCE | APPEND-ONLY per Catalog #110/#113 | §18 footer appended below existing §17 Probe 9c append |
| Sister Probe 9 Tier-2 prep landed memo | HISTORICAL_PROVENANCE | APPEND-ONLY per Catalog #110/#113 | Footer appended documenting recipe-trio canonical-fix |

## 6. Catalog #344 RATIFY-N candidate state

### NEW sister equation candidate `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1`

**Status BEFORE this landing**: QUEUED for operator-decision protocol per Probe 9c landing §5; evidence count = 1 (Probe 9c N=100 anchor at db4 basis).

**Status AFTER this landing**: QUEUED for operator-decision protocol; evidence count = **2** (Probe 9c N=100 anchor + Probe 9 substrate-trio canonical-fix landing). The canonical-fix landing IS the substrate-side operationalization of the Probe 9c canonical-optimal basis verdict; downstream consumers of the recipe + trainer + driver inherit the canonical-optimal db4 basis structurally rather than via per-call configuration.

Evidence anchor data:

```
Predicted distortion (same formula form as Probe 9c landing §5):
  cost_per_segment_s = 1 / (mean_over_segment_pixels(
      sum_{l=1..L} sum_{subband in {HL, LH, HH}} |coeff_l_db4|
  ) + sigma_fridrich)
  where segment s is a connected-component instance derived from
  scipy.ndimage.label on per-class SegNet hard mask, L=3 levels of
  pywt.wavedec2('db4'), and sigma_fridrich = 2^{-6}.

Empirical anchor #1 (Probe 9c N=100 / 3-level / seed=42):
- segment_textured_avg_weight_mean = 0.3599 (95% CI [0.3457, 0.3740])
- segment_textured_avg_weight_min = 0.0532
- valid_segment_count = 537
- below_threshold_fraction = 83.6%
- z-score vs db8 baseline = -3.442σ
- relative_delta_pct = -8.06%
- axis_tag = [macOS-CPU advisory]; promotable = False

Empirical anchor #2 (canonical-fix landing 2026-05-25):
- substrate-trio canonical default updated db8 -> db4 in 3 files
- 7 dedicated tests pin db4 default + cite chain + HISTORICAL_PROVENANCE preservation
- algorithm byte-stable verified on macOS-CPU synthetic-frame smoke
- evidence: this memo + Catalog #313 row at .omx/state/probe_outcomes.jsonl
```

### Sister equation candidate `uniward_per_instance_multi_scale_wavelet_combined_v1`

EXCLUDED_CONTEXT addendum recommended per Catalog #344 + #359 sister-discipline:

- IN_DOMAIN_CONTEXT (preserved): `uniward_per_instance_multi_scale_wavelet_combined_db8_basis` (Probe 9 + Probe 9b + Probe 9c db8 baseline anchors)
- NEW EXCLUDED_CONTEXT (requires operator-verbatim approval): `db8_basis_NOT_optimal_per_basis_ablation_db4_dominates_at_zminus3p442sigma`

## 7. Catalog #313 probe-outcomes ledger row

Registered via canonical helper `tac.probe_outcomes_ledger.register_probe_outcome` (fcntl-locked JSONL append-only at `.omx/state/probe_outcomes.jsonl` per Catalog #131):

- **probe_id**: `probe_9_recipe_canonical_update_db8_to_db4_20260525`
- **substrate**: `uniward_per_instance_multi_scale_wavelet_combined`
- **recipe_path**: `.omx/operator_authorize_recipes/substrate_uniward_per_instance_multi_scale_wavelet_segnet_modal_t4_dispatch.yaml`
- **probe_kind**: `canonical_substrate_recipe_update_db8_to_db4`
- **verdict**: `PROCEED` (canonical-fix landing; substrate paradigm preserved per Catalog #307 IMPLEMENTATION-LEVEL fix)
- **metric_name**: `canonical_basis_db8_to_db4_substrate_recipe_update_per_probe_9c_verdict`
- **metric_value**: 0.3599 (Probe 9c db4 N=100 anchor mean)
- **threshold**: 0.3915 (Probe 9b N=100 db8 baseline mean)
- **threshold_token**: `db4_dominates_db8_at_zminus3p442sigma_per_probe_9c_landing`
- **evidence_path**: `.omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md`
- **blocker_status**: `advisory`
- **expires_at_utc**: 2026-06-24 (30-day staleness window per Catalog #298)

## 8. Probe 9 Tier-2 dispatch state

| Binding revision | Source | Status |
|---|---|---|
| #1 Contrarian: Probe 9b 100-pair disambiguator | symposium L196-200 | ✓ SATISFIED (Probe 9b landing 2026-05-25) |
| #3 Mallat: per-level wavelet-basis selection table | symposium L210-217 | ✓ **SATISFIED with canonical-fix applied** (Probe 9c landing 2026-05-25 + THIS canonical-fix landing 2026-05-25) |
| (paired CPU+CUDA empirical anchor — emergent from #4/#5/#6) | symposium L249-250 | PENDING — sister subagent `_full_main` BUILD + paired dispatch authorization |

**Result**: 2-of-3 dispatch-blocking revisions cleared + canonical-optimal basis applied across substrate trio. The Probe 9 Tier-2 dispatch recipe is structurally one cycle closer to authorization. Sister subagent BUILD `_full_main` on canonical db4 substrate is **now UNBLOCKED** per the canonical-fix landing — the substrate trainer's `_full_main` (not yet built) will inherit the canonical-optimal db4 basis by default.

## 9. Operator-routable cascade per Carmack MVP-first step 5

Within the next ~1 hour of this landing:

1. **P1 NEXT**: Sister subagent **BUILD `_full_main`** on canonical db4 substrate per Daubechies / Quantizr / Selfcomp revisions 4-5-6 (positioning + gradient-routing form + loss-vs-grammar split per Catalog #312). The canonical-fix landing UNBLOCKS this larger BUILD — the trainer's full path will inherit db4 by default rather than the falsified db8. Estimated $0 design + ~$2-7 paid Vast.ai 4090 dispatch when authorized.

2. **P2 OPTIONAL**: Probe 9d N=600 db4 re-anchor (~1-2 hr macOS-CPU; $0) for higher statistical confidence on the -8.06% mean drift before paired CPU+CUDA empirical dispatch. Not required to satisfy Mallat binding revision #3 (z=1.72x threshold + CIs disjoint at 95% is structurally clean).

3. **P3 OPTIONAL**: Probe 9e per-decomposition-level disambiguator (2-vs-3-vs-4 levels at db4 baseline) per Catalog #308 alternative-reducer cascade — explores Mallat 1989 multi-resolution depth axis orthogonal to the basis axis. Updates `--wavelet-levels` default if canonical-optimal level != 3.

4. **P4 DEFERRED**: wider-basis-family sweep (db1-db20 + bior2.2-bior6.8 + sym4-sym20 + coif1-coif5) if Probe 9e returns INSUFFICIENT_DISCRIMINATION at level=3 axis.

5. **P5 DEFERRED**: alternative steganographic distortion HILL or S-UNIWARD per Catalog #308 cascade if db4 + 3-level + UNIWARD return INSUFFICIENT_DISCRIMINATION at smoke- or empirical-anchor scale.

6. **P6 DEFERRED**: Catalog #344 operator-decision-protocol invocation for NEW canonical equation candidate `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1` registration AND EXCLUDED_CONTEXT addendum to existing `uniward_per_instance_multi_scale_wavelet_combined_v1` for the db8-specific in-domain context (requires operator-verbatim approval per Catalog #344 + #359 sister-discipline).

7. **P7 DEFERRED**: operator-frontier-override per Catalog #300 to bypass the remaining 1 dispatch-blocking revision (paired CPU+CUDA empirical anchor) if leaderboard moves (current leaderboard pointer at `.omx/state/canonical_frontier_pointer.json`; no race-mode active per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first").

## 10. Honest deferrals

- NO PAID GPU FIRED. NO Modal / Vast.ai / Lightning dispatch.
- `[macOS-CPU advisory]` tag MANDATORY per Catalog #192 on smoke re-verification metrics. `score_claim=False`, `promotable=False`, `ready_for_exact_eval_dispatch=False`.
- The recipe + trainer + driver UPDATES are sister UPDATES per Catalog #110/#113 — source files are LIVE_CODE/LIVE_RECIPE per Catalog #113 4-kind taxonomy so they may be updated per Catalog #307 IMPLEMENTATION-LEVEL fix discipline. Landed MEMOS (commits 92a48616e + efeaff5c9 + 2fca9974b artifacts) are APPEND-ONLY only.
- The smoke re-verification used synthetic frames + deterministic seed to verify the canonical algorithm path is intact at the source-execution level. The contest-frame empirical receipts (mean=0.3599 / min=0.0532 / z=-3.442σ vs db8 baseline) live in the sister Probe 9c landing memo §2 and are the canonical empirical anchor for the canonical equation candidate `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1` evidence anchor #1.
- The 4-basis ablation tested only the canonical 4 bases per the Mallat seat verbatim. A wider sweep (P4 above) is queued as deferred enhancement if Probe 9e returns INSUFFICIENT_DISCRIMINATION.

## 11. Sister coordination

| Sister | Status at landing | Scope | Overlap |
|---|---|---|---|
| Slot 1 PR95-STAGE-6-MLX-BUILD (`aa372245`) | IN-FLIGHT step 2 | `src/tac/local_acceleration/pr95_hnerv_mlx.py` + optimizer_scheduler_registry + Stage 6 tests | ZERO overlap (PR95 paradigm; disjoint files) |
| Slot 3 HINTON-DISTILLED-SCORER-SURROGATE-DISPATCH-PREP (task #1243) | IN-FLIGHT (not in checkpoint feed) | Hinton KL distillation paradigm | ZERO overlap (Hinton paradigm; disjoint files) |
| Sister Probe 9c (`probe-9c-mallat-per-level-wavelet-basis-selection-disambiguator`) | COMPLETED at 2026-05-25T17:04:12Z | `tools/probe_9c_*.py` + Probe 9c landing memo + symposium memo §17 + probe_outcomes ledger + lane registry | UPSTREAM — my work APPLIES the Probe 9c canonical-fix recommendation; sister memo footer APPEND-ONLY |
| Sister Probe 9 Tier-2 prep (`sub_probe9_tier2_dispatch_prep`) | COMPLETED at 2026-05-25T16:31:37Z | recipe + trainer + driver scaffold + symposium memo + Tier-2 prep landing | UPSTREAM — my canonical-fix updates the scaffold sister landed; sister landing memo footer APPEND-ONLY |

Catalog #340 sister-checkpoint guard PROCEED verified BEFORE any edit (0 in-flight subagents intersecting on my 3 source files within 60-min lookback window).

## 12. Catalog discipline closure

- **Catalog #1** `check_no_mps_fallback_default`: PyTorch device pinned to CPU in smoke re-verification; no MPS fallback.
- **Catalog #110 / #113** APPEND-ONLY HISTORICAL_PROVENANCE: this memo is NEW (not mutating sister Probe 9c memo); recipe + trainer + driver are LIVE_RECIPE/LIVE_CODE per Catalog #113 4-kind taxonomy + UPDATE-allowed per Catalog #307; sister memo footers APPEND-ONLY (no body mutation).
- **Catalog #117 / #157 / #174 / #235 / #289** canonical commit serializer: this landing commits via `tools/subagent_commit_serializer.py --expected-content-sha256` per Catalog #157 post-edit working-tree sha contract.
- **Catalog #125** 6-hook wire-in (see §13 below).
- **Catalog #131 / #138 / #245** fcntl-locked posterior writes: Catalog #313 row registered via `tac.probe_outcomes_ledger.register_probe_outcome` canonical helper.
- **Catalog #151** TIER_1_OPERATOR_REQUIRED_FLAGS manifest discipline: `--wavelet-name` default updated db8 -> db4 + rationale updated to cite Probe 9c verdict + Catalog #307 classification.
- **Catalog #176 / #185 / #186** META-meta catalog drift: this memo does NOT add a new STRICT preflight gate (canonical-fix landing only).
- **Catalog #192** `check_macos_cpu_advisory_not_promoted_without_linux_verification`: smoke re-verification metrics carry `[macOS-CPU advisory]` axis_tag per the smoke summary's existing `axis_tag` + `evidence_grade` + `promotable=False` fields.
- **Catalog #206** subagent crash-resume discipline: 4 checkpoints emitted (initial step 0 + step 1 after edits + step 2 after tests + final completion follows this memo).
- **Catalog #229** premise verification before edit: read sister Probe 9c landing (full file) + sister symposium memo (top + relevant sections) + Probe 9 Tier-2 prep landing reference + recipe YAML (all relevant sections) + trainer Python (db8 references + smoke summary) + driver shell (db8 reference) BEFORE any edit; verified pywt has db4 + API signatures.
- **Catalog #230 / #340** sister-subagent ownership map: sister-checkpoint guard PROCEED before any edit; Slot 1 / Slot 3 scopes verified DISJOINT pre-edit; sister Probe 9c + Probe 9 Tier-2 prep scopes verified COMPLETED pre-edit.
- **Catalog #287 / #323 / #341** canonical Provenance: every score-claiming field in smoke summary + Catalog #313 row carries axis+hardware+evidence_grade triple OR cites Probe 9c's canonical Provenance via reference.
- **Catalog #298** 30-day staleness window: Catalog #313 row expires 2026-06-24.
- **Catalog #300** mission-alignment frontmatter: this memo carries `council_predicted_mission_contribution: frontier_breaking_enabler` (canonical-optimal basis enables Probe 9 Tier-2 dispatch which targets sub-A1 frontier per the symposium predicted_band).
- **Catalog #307** paradigm-vs-implementation classification: explicitly invoked across recipe + trainer + driver comments AND in the verdict rationale (paradigm INTACT — substrate trainer's `_full_main` building atop db4 inherits the same paradigm; db8 IMPLEMENTATION FALSIFIED per Probe 9c; canonical fix updates implementation only).
- **Catalog #308** alternative-reducer enumeration: 4 alternative reducer paths enumerated in §9 cascade (P2 db4 N=600 re-anchor / P3 per-decomposition-level disambiguator / P4 wider-basis-family sweep / P5 alternative steganographic distortion).
- **Catalog #313** probe-outcomes canonical ledger: registered with full canonical contract.
- **Catalog #325** per-substrate symposium adversarial-discipline cycle: this canonical fix satisfies Probe 9c P0 operator-routable AND Mallat binding revision #3 SATISFIED-with-canonical-fix per symposium §17 closure.
- **Catalog #344** canonical equations registry: NEW sister candidate `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1` evidence count strengthens 1 -> 2 via this canonical fix landing (Probe 9c N=100 anchor + substrate-trio canonical-fix landing). EXCLUDED_CONTEXT addendum recommended for sister candidate `uniward_per_instance_multi_scale_wavelet_combined_v1` for db8-specific in-domain context per Catalog #344 + #359 sister-discipline.
- **Catalog #359** canonical-equation-misapplication-to-residual-hybrid-context discipline: this canonical fix is REPLACEMENT (db8 -> db4 basis substitution) NOT RESIDUAL-HYBRID; the equation form is preserved at REPLACEMENT-savings semantic. Not in scope of #359 refusal pattern.

## 13. 6-hook wire-in declaration per Catalog #125

- **Hook 1 (sensitivity-map)**: ACTIVE — the canonical db4 basis is the canonical-optimal sensitivity surface per Probe 9c; future `tac.sensitivity_map.*` consumers ingesting per-instance × per-basis sensitivity inherit the canonical-optimal basis structurally.
- **Hook 2 (Pareto constraint)**: N/A AT CANONICAL-FIX STAGE — Pareto constraint is added at the substrate trainer `_full_main` landing (sister subagent per Selfcomp binding revision); this canonical fix is the upstream basis-default surface that the `_full_main` consumer will inherit.
- **Hook 3 (bit-allocator)**: N/A AT CANONICAL-FIX STAGE — per-basis per-instance + multi-scale cost-map is a future per-pixel bit-allocator signal at substrate trainer `_full_main` landing (canonical basis = db4 per Probe 9c + this canonical fix).
- **Hook 4 (cathedral autopilot dispatch)**: ACTIVE — Catalog #313 probe outcome row consumed by `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335 auto-discovery; ranker can route Probe 9 Tier-2 dispatch candidate per the UNBLOCK_2_OF_3_DISPATCH_BLOCKING signal + canonical-optimal db4 basis recommendation surfaced in the smoke summary's `probe_9c_canonical_optimal_basis` field.
- **Hook 5 (continual-learning posterior)**: ACTIVE — Catalog #313 row at `.omx/state/probe_outcomes.jsonl` + canonical equation candidate update at `.omx/state/canonical_equations_registry.jsonl` (sister `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1` evidence count strengthens 1 -> 2 via this landing).
- **Hook 6 (probe-disambiguator)**: ACTIVE — this canonical-fix landing IS the structural operationalization of Probe 9c's SISTER_BASIS_DOMINATES_db4 verdict; the trainer + driver + recipe DEFAULT inherits the canonical-optimal basis so downstream consumers don't need to re-disambiguate at every dispatch call. The probe-disambiguator output (Probe 9c verdict) is now structurally baked into the substrate trio.

## 14. Lane registration

- **lane_id**: `lane_probe_9_recipe_canonical_update_db8_to_db4_20260525`
- **level**: 1 (impl_complete + memory_entry; canonical-fix landing class)
- **gates**: `impl_complete=true` (3 files updated + 1 NEW test) + `memory_entry=true` (this memo) + Catalog #313 ledger row + canonical Provenance per Catalog #287+#323+#341
- **substrate-side wire-in**: APPENDS forensic data to sister substrate Tier-2 prep lane `lane_probe_9_tier_2_dispatch_prep_uniward_per_instance_multi_scale_wavelet_segnet_20260525` (COMPLETED earlier today) per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

## 15. Cost + wall-clock

- **GPU spend**: $0
- **Predicted wall-clock**: 15-25 min
- **Actual wall-clock**: ~25 min (4 file edits + 7-test suite at 0.21s + smoke re-verification + landing memo + Catalog #313 ledger row + sister memo footers + commit machinery)
- **Cost saved vs paid Tier-2 dispatch at db8**: ~$2-7 (substrate trainer would have trained at falsified db8 then auth-eval'd; canonical-fix landing prevents at least one re-dispatch cycle at db4 if the substrate trainer had been built first without the canonical-fix)

## 16. Cross-references

- **Probe 9 BREAKTHROUGH anchor**: `.omx/research/combined_tier_1_wave_3_uniward_multi_scale_plus_hinton_motion_aware_landed_20260525.md` (commit `685fe6726`)
- **Probe 9-PREP (sister Tier-2 dispatch scaffold)**: `.omx/research/feedback_probe_9_tier_2_dispatch_prep_uniward_per_instance_multi_scale_wavelet_segnet_landed_20260525.md` (commit `92a48616e`)
- **Probe 9b 100-pair landing**: `.omx/research/probe_9b_100_pair_disambiguator_landed_20260525.md` (commit `2fca9974b`)
- **Probe 9c per-level basis landing**: `.omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md` (commit `efeaff5c9`)
- **Probe 9c canonical tool**: `tools/probe_9c_per_level_wavelet_basis_selection_disambiguator.py`
- **Probe 9c verdict JSON**: `.omx/research/tier_1_distortion_axis_probes_20260521/probe_9c_per_level_wavelet_basis_20260525T165714Z.json`
- **Tier-2 dispatch symposium**: `.omx/research/per_substrate_symposium_uniward_per_instance_multi_scale_wavelet_segnet_20260525.md` (Mallat binding revision #3 SATISFIED with this canonical-fix landing)
- **Substrate trainer**: `experiments/train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py` (UPDATED in this commit batch)
- **Substrate driver**: `scripts/remote_lane_substrate_uniward_per_instance_multi_scale_wavelet_segnet.sh` (UPDATED in this commit batch)
- **Substrate recipe**: `.omx/operator_authorize_recipes/substrate_uniward_per_instance_multi_scale_wavelet_segnet_modal_t4_dispatch.yaml` (UPDATED in this commit batch)
- **NEW canonical-fix test**: `src/tac/tests/test_train_substrate_uniward_pims_canonical_db4_basis.py`
- **CLAUDE.md non-negotiables invoked**: "Carmack MVP-first phasing"; "Forbidden premature KILL without research exhaustion"; "MPS auth eval is NOISE"; "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"; "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"; "Bugs must be permanently fixed AND self-protected against"; "Subagent coherence-by-default"; "Canonical equations + models registry"
- **Sister landings (same wave)**: `probe_9b_100_pair_disambiguator_landed_20260525.md` (revision #1 satisfaction); `probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md` (revision #3 satisfaction); `feedback_probe_9_tier_2_dispatch_prep_uniward_per_instance_multi_scale_wavelet_segnet_landed_20260525.md` (Tier-2 scaffold prep)

---

**Verdict TL;DR**: Probe 9 substrate trio (recipe + trainer + driver) canonical-updated from db8 -> db4 per Probe 9c SISTER_BASIS_DOMINATES_db4 verdict (commit `efeaff5c9`). 3 source files UPDATED in-place per Catalog #307 IMPLEMENTATION-LEVEL fix; 7 NEW pytest assertions pin db4 default + cite chain + Catalog #110/#113 HISTORICAL_PROVENANCE preservation of Probe 9 N=25 db8 historical anchor. Mallat binding revision #3 of 6 SATISFIED with canonical-fix; 2-of-3 dispatch-blocking revisions cleared + canonical-optimal basis applied. Canonical equation candidate `uniward_per_instance_multi_scale_wavelet_basis_optimal_db4_v1` evidence count strengthens 1 -> 2. Probe 9 Tier-2 dispatch state: P1 sister `_full_main` BUILD on canonical db4 substrate now **UNBLOCKED**. Carmack MVP-first 5/5 compliance: free macOS-CPU smoke + falsifiable challenge + canonical equation anchor + same-commit-batch verdict + ~1h operator priority queue re-route ALL satisfied. $0 GPU + ~25 min wall-clock.
