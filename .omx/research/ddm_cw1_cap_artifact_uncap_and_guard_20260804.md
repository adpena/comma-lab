---
schema: ddm_cw1_cap_artifact_guard.v1
date_utc: 2026-08-04
arm: ddm_cw1
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
tokens: [no-triality, p0-ledger-ok]
head_at_measure_start: 71a790be2e
---

# ddm_cw1 - cap artifacts: stop reasons before verdicts

## Answer First

`#935` is a real cap artifact.  The old sq1 solved-paint receipt stopped at a
25-step cap; rerunning the same n32 selected pairs with explicit convergence
telemetry and a 50-step cap improved all 32 rows, but still did not produce a
convergence verdict.

Measured denominator: n=32 sq1 selected pairs.

| receipt | steps | after flips | eta on described band | stop-reason rows |
|---|---:|---:|---:|---:|
| `sq1_stage_n32.json` | 25 | 8,541 | 0.789509594883 | 0/32 explicit |
| `sq1_stage_n32_uncap50_cw1.json` | 50 | 6,841 | 0.862004264392 | 32/32 explicit |

Delta: 1,700 fewer realized flips than the 25-step receipt; 32/32 pairs
improved.  Stop reasons at step 50: `iteration_cap_best_at_cap` 31/32,
`iteration_cap_before_plateau` 1/32, converged 0/32.  This is a better floor,
not a convergence certificate.

Receipt custody:

| artifact | bytes | sha256 |
|---|---:|---|
| `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap50_cw1.json` | 131,241 | `b412bb4d1a31e83a4d31a0600102dba0b1549bc02b27c89ade725d3b89676998` |
| `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32_uncap50_cw1.json` | 6,281 | `444b0c248229abb76264c51c192cef9b4b1322f38561c4e8549dc174b102e7b9` |

The aggregate's subset-scoped printout reprices the measured eta as
`gross +0.36151`, `net -0.11679 S`, but this is still n32, subset-scoped,
pose-collateral-heavy, and non-promotable.  No exact archive was built or run.

## Commands Run

Primary #935 measurement:

```bash
.venv/bin/python experiments/ddm_sq1_stage_decomposition_and_solved_paint.py \
  --sub-dir /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2 \
  --gt-mkv upstream/videos/0.mkv \
  --pairs-npy /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_selected_pairs.npy \
  --out /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap50_cw1.json \
  --threads 6 \
  --steps 50 \
  --eval-every 5 \
  --convergence-patience-evals 3 \
  --resume
```

Aggregation:

```bash
.venv/bin/python experiments/ddm_sq1_aggregate.py \
  /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_eta_seg_n32.json \
  /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap50_cw1.json \
  /Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32_uncap50_cw1.json
```

## #938 Guard

Landed a code-derived class guard:
`tools/check_no_silent_cap_defaults.py`.

Current denominator from the generated baseline:

| scanned files | cap-default sites | grandfathered silent sites | parse errors |
|---:|---:|---:|---:|
| 21,674 | 321 | 317 | 0 |

Baseline artifact:
`.omx/research/ddm_cw1_silent_cap_defaults_baseline_20260804.json`,
185,175 bytes,
`e8238a91c8adc067949066544995ff24a947cc50032bd6367192ebefdfffb3a1`.

Gate command:

```bash
.venv/bin/python tools/check_no_silent_cap_defaults.py \
  --baseline .omx/research/ddm_cw1_silent_cap_defaults_baseline_20260804.json
```

Result: `new_silent=0`.

Scope: this refuses new silent cap defaults relative to the current baseline.
It does not remediate the 317 grandfathered silent sites.

## #874 Fold

`#874` is folded into two existing surfaces rather than reopened as an isolated
grep count:

1. The existing termination-census law
   `tac.canonical_equations.ddm_os1_termination_census_from_cost_proxy_v1`
   still owns the damped-GN stop-state reconstruction class, and its tests were
   rerun in this landing.
2. The new `check_no_silent_cap_defaults.py` guard owns the "new argparse cap
   default with no stop-reporting vocabulary" class.

This arm did not mutate the `ddm_os1` law.

## #850 Fold

`#850` was not rerun.  The handed statement that the live pose solve was still
hard-capped at 2-3 relinearisations is stale in the current corpus:
`ddm_ss1_selection_vs_search_20260803.md` reports the terminal-pose-GN instance
as cured/off the live chain and remeasures the claimed 13-23% tail as 1.2% per
relinearisation at the shipped bound; `ddm_qd2_rebaseline_against_cx1_20260803.md`
and `ddm_pu1_pose_underpricing_and_tail_20260803.md` classify the `#850/#873/#882`
pose delta as one `pj2` run triple-stamped and already absorbed into `cx1`.

Disposition: FOLDED, not queued.  Rerunning it as a live cap artifact here would
reopen a stale/off-chain premise instead of measuring the current vehicle.

## Fire Orders

- `#935` extension: QUEUED.  If more local scorer-light time is allocated, rerun
  the same command to a higher cap on the same n32 denominator, first `--steps
  100 --eval-every 5 --convergence-patience-evals 3`, writing a new receipt
  path.  A result is admissible only with explicit stop-reason counts; if rows
  remain `iteration_cap_*`, quote it as a higher floor, not convergence.
- `#935` n600 promotion: QUEUED BEHIND SCORER SLOT.  Fire only after the current
  full-n600 owner releases the slot, chunk <=120, and keep the same stop-reason
  schema.  Do not promote from the n32 receipt.
- Existing 317 silent defaults: QUEUED AS HYGIENE, not a frontier row.  Use the
  baseline JSON's `sites` list to rank by live-chain consumer before editing;
  each remediation must add a real stop-report surface, not just a marker string.

## Boundaries

- No exact archive was built.
- No `upstream/evaluate.py` run was made.
- No contest-CPU/CUDA claim was made.
- No MPS authority was used.
- The n32 sq1 selected-pair population is not n600 and not a promotable score.
- The own-vehicle frontier is unchanged.

