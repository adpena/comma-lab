# Build-wave: top duty-to-measure levers (#377, 2026-07-09)

**Agent:** BUILD agent under #377 ("Build all unbuilt", operator GO 2026-07-09).
**Scope:** the top-3 UNBUILT-marked duty-to-measure levers from `tools/costate_digest.py`.
**$0, no GPU, no launches.** Pointer 0.19110 [contest-CPU] UNMOVED (means).

## TL;DR — per-lever disposition

| Lever | Digest rank | Disposition |
|---|---|---|
| #121 d_seg-aware taper | 73% | **ALREADY-BUILT + ALREADY-HELD** (did NOT rebuild) — owed a measurement |
| #169 horizon-weighted margin | 43.8% | **ALREADY-BUILT + ALREADY-HELD + COMPOSED** (did NOT rebuild) — owed a measurement |
| D18 latent-table truncate (k90) | 2.4% | **DEFERRED — named blocker: no v7 final ckpt** (machinery already exists) |

The mission's "unbuilt" premise (from the digest's `~` marker) was **FALSE for the top two**. Per the
operating manual §1 (read the real ask) + §4 (verify by re-deriving, not recognizing), the premise was
attacked before building and falsified against the actual artifacts. Building either would be a
forbidden duplicate (NO-FAKE #7). The real build was an **apparatus fix** that makes the duty-to-measure
system correctly recognize the two held levers.

## Evidence (MEASURED via source inspection + tests)

### #121 — `DsegAwareTaper` (`src/tac/witness_dsl/curriculum_dsl.py:1944`)
- Compiles `--dseg-aware-taper` + `-strength/-scale/-floor` (factory `overrides`).
- All 4 flags **MAPPED** in `lever_registry.completeness()` (not in `.unmapped`).
- **WIRED** into the levelset trainer: argparse `experiments/train_levelset_witness_realized_through_R_mlx.py:10449`, cfg propagation `:691-694/:846-849`, render-AA guard `:3081`.
- Reference twin: `src/tac/boundary_math/dseg_aware_fourier_taper.py`.
- Byte-identical default-OFF (`--dseg-aware-taper` default False → branch never built).

### #169 — `HorizonWeightedMargin` (`curriculum_dsl.py:3192`)
- Compiles `--seg-horizon-margin-{weight,target,lo,hi,rows...,start-epoch}`; argparse `:10929`.
- **LIVE in the trainer loss** `:4988-4991`: `hz_term = mx.sum(_hz_hinge)/(mx.sum(_hz_mask)+1e-6); L = L + hz_w*hz_term` — the one-sided hinge `relu(m_target − m_wit)` on the shared through-R witness GT-class margin, stratified to the θ-independent horizon band × GT-margin ∈ [lo,hi]. Composes onto the existing #141/#274 margin/spike machinery (same SegNet forward, 0 archive bytes).
- Reference twin: `src/tac/boundary_math/horizon_weighted_margin.py`.
- Byte-identical default-OFF (`weight=0.0` default → branch skipped).
- Tests: `boundary_math/tests/test_horizon_weighted_margin.py` + `test_dseg_aware_fourier_taper.py` + `tests/test_v752_owed_gates.py` = **58 passing**.

## Root cause of the false `~unbuilt` marker

`duty_to_measure_ranked` (`activation_ledger.py`) computes `registered = (significance_key ∈ factory_names)`.
The relative-significance store (`.omx/state/lever_relative_significance.jsonl`) keyed these rows by
task-# names (`d_seg_aware_taper_121`, `horizon_weighted_margin_169`) — recorded when they were
duty-to-BUILD findings and never reconciled to the now-held factory names (`DsegAwareTaper`,
`HorizonWeightedMargin`). So a built+held+wired lever was ranked `registered=False` → digest `~=unbuilt`
(duty-to-BUILD) instead of `*=never-fired` (duty-to-MEASURE). This is orphaned signal per CLAUDE.md
"'Off' is a tracked queue" + "Results must become system intelligence".

## The build (apparatus, `src/tac/witness_dsl/activation_ledger.py`)

Read-time **significance-key canonicalization**: `canonicalize_significance_keys(sig, factory_names)`
+ guarded `_SIGNIFICANCE_LEVER_ALIASES` map. Properties:
- Applies an alias **only when its target IS a real held factory** — if the factory is renamed/removed,
  the row correctly reverts to a build gap (fail-safe).
- **APPEND-ONLY store NOT rewritten** — canonicalization is at read time (history preserved).
- **Preserves an explicit canonical row** (latest-wins) — never clobbers a real `DsegAwareTaper` sig row.
- Pure + idempotent (returns a new dict; input unmutated).
- Wired into `duty_to_measure_ranked` (single call site after `_read_significance`).

**Result:** `tools/costate_digest.py` now reads `DsegAwareTaper* 73%, HorizonWeightedMargin* 43.8%`
(`*` = never-fired = duty-to-MEASURE). D18's `latent_table_truncate_d18_k90` intentionally NOT aliased
(it is a byte-close-tool lever, not a DSL factory) → stays a finding.

**Durable follow-through (noted, not a blocker):** new significance rows for a HELD lever should be
recorded under the canonical factory name from the start; the alias map is legacy reconciliation only.
Adding a NEW canonical-named sig row while a legacy row still exists would re-orphan the legacy row
(the guard leaves it) — so do NOT dual-key; migrate.

## D18 — deferred with named blocker

- **Blocker:** no v7 FINAL checkpoint (run `dry_start`, `best=NONE`, no `levelset_best.json`). The A/B
  "truncate `code` to measured k90 columns → real Δbytes vs Δd_seg/Δd_pose" consumes the FINAL ckpt
  `code` table + the `{stage:mod_dim_dynamics}` k90 series; neither exists yet.
- **NOT unbuilt:** the truncation-at-export machinery exists — `tools/witness_code_pca_byteclose.py`
  (PCA-K sweep `--ks`, reconstructs codes from DEQUANTIZED PCA rep, measures rate-vs-realized-d_seg
  Pareto via the deploy-faithful realized verdict) + the k90 sensor
  `src/tac/boundary_math/mod_dim_dynamics.py` (`k_energy_cutoff`→k90, `truncate_bytes_estimate`).
- **Only-missing-wire (premature to build now):** auto-feed measured k90 from telemetry into `--ks`
  (the sweep already covers any K incl. k90); un-validatable without the ckpt.

## Tests

- `src/tac/tests/test_activation_ledger.py`: +5 (24 total pass) — move-legacy-onto-factory ·
  no-op-when-target-not-a-factory · preserve-explicit-canonical-row · pure+idempotent · end-to-end
  ranked-marks-registered-not-unbuilt.
- Regression green: `test_relative_significance_ranking.py` (23) · `test_costate_digest_ncde.py` +
  `test_costate_percost_ranking.py` (23) · the 58 #121/#169 lever tests · ruff clean.

## Triality

- **DSL leg:** factories already present (`DsegAwareTaper`, `HorizonWeightedMargin`) — no new factory owed.
- **DAG leg:** `sub015_DAG_*` FEED-buildwave-dtm (2026-07-09).
- **Equations leg:** N/A until measured — the two levers carry est ΔS (#169: 0.012–0.024 oracle) but no
  byte-closed n600 row; the canonical-equation anchor lands only when a real through-R measurement exists
  (their duty-to-MEASURE, now correctly surfaced by this fix).

## Canonical equations (Catalog #344)
# FORMALIZATION_PENDING: duty-to-measure ledger memo — no measured rows; each lever registers its equation on its first measured firing per the activation-ledger discipline.
