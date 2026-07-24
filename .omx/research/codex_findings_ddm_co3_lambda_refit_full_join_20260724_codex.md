# Codex findings — DDM CO3 N600 lambda refit and full join

UTC: 2026-07-24  
Lane: `ddm_co3_lambda_refit_full_join`  
Status: `IMPLEMENTED_ADVISORY_MAIN_REVIEW_REQUIRED`  
Authority: `_dev`, `research_only=true`, `execution_allowed=false`,
`actuation=NONE`, `score_claim=false`, `promotion_eligible=false`.

## Verdict

The preregistered N600 pair-held-out admission bar is met, so the selected
`[advisory-heuristic]` may upgrade the campaign's measurement-duty ordering.
It does **not** authorize pair actuation: only 15/600 pairs have direct positive
MS4D Fisher precision, so 585 pair rankings remain explicitly unranked.

All metrics below are `[macOS-CPU frozen-scorer advisory]` and computed only
from concatenated out-of-fold predictions over exact pair IDs 0..599.

| Candidate | Held-out NDCG@4 | Held-out Spearman | Disposition |
|---|---:|---:|---|
| factorized refit | 0.2222207523 | 0.6986970345 | measured control |
| factorized + MS4D interactions | **1.0000000000** | **0.8607149751** | selected |
| G4 regime-conditional | 1.0000000000 | 0.8580703066 | lower Spearman |
| close-form nested mixture | 0.9180529396 | 0.8606129895 | measured, not selected |
| small monotone/GB | — | — | preregistered trigger not met; no family negative |

Admission: `1.0 >= 0.75`, held-out only. The prior full-join factorized control
(`NDCG@4=0.1955706570`, `rho=0.7476669456`) remains retained as history, not
silently overwritten.

## Collapse localization

The simple factorization fails to represent the realized N600 response
geometry. Measured EV1 response features and MS4D interactions recover the
global top-four ordering, but the recovery is not uniform:

- `MyCar`: 20 pairs, mean absolute innovation `0.1022806267` (worst stratum),
  NDCG@4 `1.0`, rho `0.7781336587`.
- `Road`: 288 pairs, mean absolute innovation `0.0250909953`, NDCG@4 `0.0`,
  rho `0.7574632039`. This is a real localized ranking failure even though the
  global NDCG@4 is perfect.
- `Undrivable`: 60 pairs, NDCG@4 `0.5180392788`, rho `0.5988887348`.
- margin decile D10: mean absolute innovation `0.0891269333`, rho
  `0.6335423103`.
- hardness decile D10: mean absolute innovation `0.0732745216`, rho
  `0.8323494559`.

The global top four are pair IDs `452, 75, 313, 446`; pair `327` is the first
false positive at rank five with realized target zero. This is an
instance/backtest result, not prospective cross-family generalization.

The requested G4 pair-class row is honest but non-separating: all 600 dominant
labels are `STATIC_IN_IMAGE`, derived from G3 boundary/interior counts crossed
with aggregate G4 class masses. No exact pair-level G4 class authority exists,
so no pairwise G4-class separability is claimed.

## Pantheon self-checks

- Pontryagin/Bellman adjacent-lambda residual:
  `AWAITING_J8F_MEASUREMENT`; pair OOF rows are not a Bellman trajectory.
- Organ/RD1 dual consistency:
  `AWAITING_NON_NULL_MATCHED_RD1_DUALS`; the hash-verified RD1 receipt has
  162 typed dual rows and 0 actionable prices. Null was not coerced to zero.
- Wallace/MML precision: 15/600 intervals; overlap or missing precision emits
  `TIED_OVERLAPPING_OR_MISSING_INTERVAL` or `UNRANKED_PRECISION_OWED`.
- Kalman innovation diagnostic: lag-one correlation `0.0667958019`;
  `NO_LARGE_LAG_ONE_COLOR_DETECTED`, scoped to lag one only.
- Rudin explainability: reused
  `tac.autopilot_rudin_daubechies.falling_rule_list.FallingRuleList`.
  GOSDT/SLIM were reasoned-excluded because their fixed action/feature schemas
  are not the N600 lambda feature schema.
- Compression progress per effort and COVER/bandit allocation remain
  `AWAITING_J8F_MEASUREMENT` / `DESIGN_ONLY_NOT_ACTUATED`.

Oracle facade coverage is fresh: margin-Fisher `1200` bucket rows, `25` direct
blocks over `15` pair IDs; pose tube `600` pair rows; stationarity `5` strata,
`3` temporal classes, and flip mass `4,011,236`.

## One-digest integration

Receipt file SHA-256:
`896f18818afef8d54eba3b20e55fe339a8ef5caf08a27654bec4b1e4eed9b1f8`.

Canonical receipt content SHA-256:
`c8085d38db27cb05e92e88892827bb36c78346196de485008107ad42b496c1f3`.

Campaign state digest:
`bf68a2614ef72da810f8ae6b6c9e3e9aa95b9840a5ffbd75a99fb7e648282d0f`.

That exact state digest is carried by the digest, dashboard, duty queue, and
lever-registry activation nag. Current duties remain non-actuating:

1. `J8F_MEASURE_CLASS_E_TELEMETRY`
2. `CO3_LAMBDA_RANKER_FISHER_PRECISION_CLOSURE`
3. `MS2R_TOLERANCE_CAPPED_DIMENSION_PRICING`

The J8F blocker is preserved. The new precision blocker is
`BLOCKED_PAIR_LEVEL_MS4D_FISHER_PRECISION_585`.

## Adversarial review and verification

Two clean Codex review passes were recorded for every changed Python file.
Pass 1 found and fixed a real custody defect: the RD1 null check had described
0 actionable prices without hashing the RD1 authority into the refit receipt.
Pass 2 added fail-closed receipt-tamper, J8F-preservation, row-firewall, and
non-actionable top-diagnostic checks.

- Focused refit/campaign/organ suite: `35 passed`.
- Broader DDM sweep: `56 passed`, `1` unrelated failure in the existing MS2R
  live replay because its immutable `01_oracle_admission.json` differs on
  resume. Verdict scope: existing MS2R stage-custody surface only; it does not
  falsify CO3.
- Ruff: clean on the new and directly changed CO3 Python surfaces.
- Compileall: clean.
- Receipt regeneration: byte-identical after the final code change.

## STORES CONSULTED

- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/master_gradient_anchors.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`
- `.omx/state/cost_band_posterior.jsonl`
- `.omx/state/continual_learning_posterior.jsonl`
- `.omx/state/canonical_frontier_pointer.json`
- latest CO2 findings/equations/DAG/done artifacts
- G3, G4, EV1, MS4D, RD1 receipts and scorer-value-oracle facade
- per-arm and broadcast Codex inboxes through `2026-07-24T22:30:57Z`

Pointer delta: none. No score, launch, provider, GPU, archive mutation, or
promotion action occurred. MAIN landing review is required.
