# UGC terminal-polish same-budget A/B — measured receipt (2026-07-12)

## Result

**MEASURED winner: `(1+1)-ES`.** At the pinned 64 exact-function-evaluation search budget,
`(1+1)-ES` reached the exact-enumeration optimum on the six-bit active support:
`Delta S = -8.229284904293088e-6`, or `1.285825766295795e-7` improvement per search
evaluation. UGC reached `Delta S = -7.524000882441762e-6`, or
`1.175625137881525e-7` per evaluation: **8.57041687162139% less progress per evaluation**
than the control. UGC tied plain DisARM and RLOO and missed bit 1.

**Verdict: `UGC_LOSES_INSTANCE_FORMULATION_SCOPED`.** Do not route UGC as the default
#396/#400 terminal-polish estimator from this receipt. The negative is scoped to the pinned
archive, six direction-pinned pair-local bits, probability geometry, seed, and 64-call budget.
It is not a family-level claim that UGC is dead.

## Same-budget table

Every variance arm and every search arm consumed exactly 64 calls to the same exact discrete
objective. Every final mask was freshly rendered and checked in the canonical 16-pair frozen
CPU-torch scorer layout; `seg_cell_maxabs=0`, `pose_cell_maxabs=0`, and score-composition
`residual=0` for all arms.

| estimator | MEASURED matched-call estimator trace variance | DERIVED exhaustive trace variance | MEASURED Delta S | MEASURED improvement / search eval | MEASURED wall-clock, variance + search + verify (s) | final mask |
|---|---:|---:|---:|---:|---:|---|
| `(1+1)-ES` control | N/A; proposal-gain variance `1.274586766056233e-12` (63 proposals after one counted reference call) | N/A | `-8.229284904293088e-6` | `1.285825766295795e-7` | `56.84323175007012` | `111111` |
| UGC | `1.414699051736422e-12` (21 samples + 1 padding call) | `1.236434309147681e-12` | `-7.524000882441762e-6` | `1.175625137881525e-7` | `47.20618595904670` | `101111` |
| plain DisARM | `1.397608656162014e-12` (32 samples) | `1.812328308876038e-12` | `-7.524000882441762e-6` | `1.175625137881525e-7` | `46.56612658291124` | `101111` |
| RLOO | `2.561923075338957e-12` (32 samples) | `2.195125977146556e-12` | `-7.524000882441762e-6` | `1.175625137881525e-7` | `47.02240820694715` | `101111` |
| exact enumeration | `0` | `0` | `-8.229284904293088e-6` | `1.285825766295795e-7` | `46.61348766786978` | `111111` |

The wall-clock column excludes the one-time shared direction sweep and sums the three separately
timed per-arm phases recorded in the receipt. It is not a contest-runtime claim.

## What the variance measurement says

- **MEASURED, matched 64 calls:** the single seeded empirical UGC trace-variance estimate is
  `1.223`% above plain DisARM. This finite-sample comparison is noisy because UGC's three objective
  calls per sample permit only 21 samples, versus 32 two-call DisARM samples.
- **DERIVED from the exact 64-state objective table:** exhaustive expectation over the estimator's
  finite sampling distribution gives UGC trace variance `1.236434309147681e-12` versus DisARM
  `1.812328308876038e-12`, a **31.77647211644078% reduction**. The exact estimator means agree
  coordinatewise with the exact Bernoulli-logit gradient to numerical precision.
- **MEASURED search consequence:** the variance reduction did not buy better fixed-call progress.
  UGC spends three exact calls per gradient sample (DisARM antithetic pair plus bitflip-1 boundary
  draw), leaving fewer monotone-gated proposals. It accepted five proposals but never activated
  bit 1; `(1+1)-ES` and exact enumeration reached `111111`.

Thus the boundary-variance mechanism is present on this objective, while the proposed UGC search
formulation loses after function-evaluation cost is included. Lower per-sample variance and higher
fixed-call terminal-polish progress are distinct laws.

## Exact authority and custody

- **Axis:** `[macOS-CPU advisory . frozen CPU-torch exact cells . NON-PROMOTABLE]`.
- **Score claim:** false. This is a local exact-cell A/B, not `upstream/evaluate.py` contest-CPU or
  contest-CUDA evidence. The frontier pointer is unmoved.
- **n600 authority composition:** all 600 cached base Seg/Pose cells enter every objective value;
  the active mask changes six pair-local cells only. The nonlinear Pose term is recomputed from the
  n600 mean on every mask. Archive bytes are re-packed and measured for every mask.
- **Frozen completed fixture:**
  `experiments/results/click_polish_399_campaign/candidate_archive.zip`, SHA-256
  `9c2afa96abdd6fa401bbdfa7a29a7f26ef67c70540656b6fd9ffd87d0bb91d6c`, `177169` bytes.
  The harness refuses a non-`STOP` fixture or a SHA mismatch. The active/live import campaign was
  not read or mutated.
- **Base exact S:** `0.19081182131424618`; base `d_seg=0.0005569203690295884`,
  `d_pose=0.00002941300304617774`.
- **Active support:** pairs `(144,147,150,153,156,159)`, one direction-pinned byte edit per pair,
  selected by a complete `28 columns x {+1,-1}` exact scorer sweep in canonical 16-pair chunks.
- **Probability geometry:** `(1/24,1/24,1/24,1/2,1/2,1/2)` with coordinatewise UGC threshold
  `tau=1/(2K)=1/12`; seed `396400`; `K=6`.
- **Budgets:** 64 exact calls for variance and 64 exact calls for search for every arm. Padding calls
  are counted. Exact enumeration has exactly `2^6=64` states.
- **Resumability:** direction rows, arm snapshots, RNG state, accepted-proposal JSONL, sealed arm
  receipts, and the aggregate receipt are durable under
  `experiments/results/ugc_terminal_polish_ab_20260712/`.
- **Receipt:**
  `experiments/results/ugc_terminal_polish_ab_20260712/measurement_receipt.json`.

No paid dispatch occurred and no live process/run was touched.

## Implementation and tests

The existing `tac.through_r.mc_finisher` now supplies a single direction-pinned exact-objective
harness with selectable `(1+1)-ES`, UGC, DisARM, RLOO, and exact-enumeration arms. UGC switches
coordinatewise: DisARM in the Bernoulli interior and bitflip-1 when
`min(p_i,1-p_i) < 1/(2K)`. The existing exact monotone gate remains the sole acceptance authority.

Positive, negative, edge, deterministic-resume, exact-budget, exact-pair-local composition, and
unbiasedness tests are in `src/tac/through_r/tests/test_mc_finisher_ugc.py`. The unbiasedness test
checks the Monte Carlo mean against the brute-force exact Bernoulli-logit finite-difference gradient
on a tiny synthetic objective; exhaustive estimator moments provide the stronger finite-support
cross-check used above.

The first adversarial review found and fixed one receipt bug: the ES proposal-gain diagnostic had
reported budget `B` while making an uncounted reference-value call (`B+1` actual). The reference is
now counted, an actual-call regression pins this contract, the stale receipt is rejected, and the
remeasured ES row has 63 proposal samples after one reference call. No result above uses the stale
row.

## Triality and routing

- **Equations:** `ugc_terminal_polish_variance_cost_progress_separation_v1` records that the
  variance ratio and fixed-call progress ratio are separate quantities.
- **DAG:** `FEED-ugc-terminal-polish-ab-396-400` records the measured route decision.
- **DSL:** N/A. UGC did not become the default terminal-polish estimator, so adding a witness DSL
  lever would promote a losing scoped formulation.
- **#396/#400 route:** keep `(1+1)-ES` as the default control and exact enumeration when support is
  locally enumerable. UGC remains an opt-in research arm.

## Claim labels and sources

- **FROM-LITERATURE:** UGC's coordinatewise DisARM/bitflip-1 construction and unbiasedness claim:
  Kunes et al., arXiv:2208.06124. Plain DisARM structure: Dong et al., arXiv:2006.10680; the
  permissive reference inspected was Google Research's Apache-2.0 `disarm` implementation.
- **MEASURED:** exact objective values, empirical matched-call variances, accepted masks, function
  evaluation counts, and wall clocks in the receipt above.
- **DERIVED:** exhaustive finite-support estimator moments and ratios computed from the measured
  64-state exact objective table, without additional scorer calls.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
`.omx/research/policy_gradient_variance_reduction_survey_20260712.md`;
`.omx/research/mc_finisher_396_design_20260710.md`;
`.omx/research/mc_finisher_diagonal_build_20260710.md`;
`src/tac/through_r/mc_finisher.py`; `tools/click_polish_exact_search.py`;
`tools/click_polish_local.py`; `reports/latest.md`; `.omx/state/lane_registry.json`;
`.omx/state/subagent_progress.jsonl`; `.omx/state/master_gradient_anchors.jsonl`;
`.omx/state/modal_call_id_ledger.jsonl`; `.omx/state/cost_band_posterior.jsonl`;
`.omx/state/continual_learning_posterior.jsonl`; latest sister findings/session/design/council memos;
the STOP-sealed campaign fixture and its authority/locality/ledger receipts; Kunes et al.; Dong et
al.; Google Research `disarm`.
