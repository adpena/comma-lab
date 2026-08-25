# DDM S1E both-OFF floor adjudicator

Date: 2026-08-25  
Harness task: #1270  
Instrument: `tools/s1a_off_floor_adjudicator.py`  
Instrument schema: `ddm_s1e_off_floor_adjudicator.v1`  
Row schema: `ddm_s1e_off_floor_checkpoint_row.v1`

## Verdict

The executed positive control is `INCOMPLETE_DATA` because seed 20260815 reached its epoch-65 endpoint
but seed 20260816 had not yet emitted `STAGE_CONTROLLER_RESULT.json`. Within the finished seed-20260815
instance, every one of 14 retained checkpoints refused the preregistered GB1 renderer corner. No point
crossed it. The closest point was epoch 65 at composed delta **+0.1554134085557307 S**.

This is an n60, fixed-evenly-strided `[Darwin-mps frozen-scorer advisory]` screen. It is not a contest
score, not a population negative, not promotion-eligible, and not a submission candidate. The instrument
sets `score_claim=false`, `promotion_eligible=false`, and `submission_candidate=false` globally and on
every row.

## Positive-control table

Every row uses the real controller-selected serialized packet size, 38,847 B. Relative to the 30,856 B
GB1 renderer block, signed `bytes_shed` is **-7,991 B**: the Stage-A packet is larger, not smaller. At the
exact exchange rate `25 / 37,545,489 = 6.658589531221714e-7 S/B`, the signed rate credit is
`-0.005320878894399272 S`; subtracting that negative credit adds the same amount as a rate penalty.

| Seed | Epoch | hard d_seg | d_pose | Seg damage S | Pose damage S | Rate credit S | Composed delta S | Crossed? |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 20260815 | 1 | 0.000832282181 | 0.049302700907 | 0.063089218136 | 0.694177592427 | -0.005320878894 | 0.762587689458 | no |
| 20260815 | 5 | 0.000843471964 | 0.016668861732 | 0.064208196389 | 0.400293945552 | -0.005320878894 | 0.469823020835 | no |
| 20260815 | 10 | 0.000818634056 | 0.013199714944 | 0.061724405649 | 0.355332891538 | -0.005320878894 | 0.422378176081 | no |
| 20260815 | 15 | 0.000789811893 | 0.009770474397 | 0.058842189290 | 0.304596352741 | -0.005320878894 | 0.368759420925 | no |
| 20260815 | 20 | 0.000758277019 | 0.008377769031 | 0.055688701949 | 0.281462532203 | -0.005320878894 | 0.342472113046 | no |
| 20260815 | 25 | 0.000788455538 | 0.007120902184 | 0.058706553799 | 0.258868958166 | -0.005320878894 | 0.322896390859 | no |
| 20260815 | 30 | 0.000769382052 | 0.004387653433 | 0.056799205166 | 0.201486035166 | -0.005320878894 | 0.263606119226 | no |
| 20260815 | 35 | 0.000764973985 | 0.005733245518 | 0.056358398550 | 0.231460738230 | -0.005320878894 | 0.293140015674 | no |
| 20260815 | 40 | 0.000809309189 | 0.003595446469 | 0.060791918921 | 0.181635397594 | -0.005320878894 | 0.247748195410 | no |
| 20260815 | 45 | 0.000790998689 | 0.002502114512 | 0.058960868890 | 0.150199507641 | -0.005320878894 | 0.214481255425 | no |
| 20260815 | 50 | 0.000794728578 | 0.002220668830 | 0.059333857760 | 0.141037859056 | -0.005320878894 | 0.205692595711 | no |
| 20260815 | 55 | 0.000810750353 | 0.001649853424 | 0.060936035268 | 0.120465392228 | -0.005320878894 | 0.186722306391 | no |
| 20260815 | 60 | 0.000801849354 | 0.001491129748 | 0.060045935359 | 0.114130595678 | -0.005320878894 | 0.179497409931 | no |
| 20260815 | 65 | 0.000814988860 | 0.000935357297 | 0.061359885993 | 0.088732643668 | -0.005320878894 | 0.155413408556 | no |

The preregistered prior-law prediction survived this positive control: pose damage exceeded seg damage
at all 14/14 tested checkpoints. There were zero seg-dominant falsifying points. This is an instance-scoped
observation on one evenly-strided n60 advisory seed, not a two-seed or population law.

## Arithmetic and typed schema

For each joined retained checkpoint/evaluation pair, the instrument emits:

```text
bytes_shed = 30,856 - controller.cheap_to_shrink_ladder.base_bytes
seg_damage_S = 100 * (hard_d_seg - 0.00020139)
pose_damage_S = sqrt(10 * d_pose) - sqrt(10 * 6.37e-6)
rate_credit_S = bytes_shed * (25 / 37,545,489)
composed_delta_S = seg_damage_S + pose_damage_S - rate_credit_S
point_crosses_renderer_corner = composed_delta_S < 0
```

The byte numerator is accepted only after the chosen allocation, allocation SHA, selection SHA, policy,
quantization-race row, ladder `base_bytes`, gate passes, retained student-packet bytes, and student-packet
SHA bind one another. Candidate ZIP size is retained only as provenance and never substituted for the
chartered renderer-packet numerator.

Global verdict precedence is:

1. `CORNER_CROSSED_AT_LEAST_ONE_POINT` if any joined point has negative composed delta.
2. `ENTERED_AND_REFUSED_ALL_POINTS` if both epoch-65 endpoints are complete and no point crosses.
3. `INCOMPLETE_DATA` otherwise.

The global verdict is copied into every typed row. A partially materialized seed store is typed as
incomplete; malformed schemas, mismatched bindings, missing retained files, and byte/SHA mismatches fail
closed instead of becoming `INCOMPLETE_DATA`.

## Executed receipt

- Path: `/Volumes/APDataStore/pact/ddm_s1e_off_floor_adjudicator/seed_20260815_positive_control.json`
- Bytes: 120,494
- SHA-256: `3a6dae41b141e1fe6e41f3f5eba8f88072fd3b1a04123c87fbc6f043ecfa94da`
- Rows: 14, epochs 1, 5, 10, ..., 65 for seed 20260815
- Seed 20260815: `ENDPOINT_COMPLETE`
- Seed 20260816: `MISSING_STAGE_CONTROLLER_RESULT`
- Global verdict: `INCOMPLETE_DATA`
- Corner crossings: 0
- Prior-law falsifiers: 0

The receipt live-content-verifies SHA-256 for the controller, evaluations, checkpoints, candidate archive
and repeat, member, semantic stream, student packet, parse-back transcript, and section-preservation
receipt. The two large per-checkpoint receiver/scorer caches are verified live for existence and byte
length; their producer SHA-256 receipts are preserved inside each independently hashed evaluation JSON
and labeled `producer_receipt_in_hashed_evaluation_plus_live_size_verified`. They were not independently
rehash-read by this adjudicator.

## Review and controls

- `ruff check`: pass.
- Behavioral controls: 12 passed.
- Two consecutive clean review passes are recorded for all 16 entities in the instrument and all 16
  entities in its test module. No review override was used.
- Controls cover both complete/refused and crossed verdicts, incomplete second-seed states, controller
  rather than ZIP byte arithmetic, prior-law counting, quantization-race mismatch, non-strided subset
  rejection, score-claim rejection, output exclusion from the live training tree, retained-size tamper,
  and same-size retained student-packet SHA tamper.
- The live training tree and `experiments/ddm_wd3_scorer_aware_width_distillation.py` remained read-only.
  No Modal job or full-n600 scorer job was fired.

## RECALL EVIDENCE

I searched `.omx/research/`, the canonical research index/DAG, hot state, task-ledger surfaces, and the
canonical-equations registry by content using `s1a`, `wd3`, `renderer`, `break-even`, `30856`,
`6.658`, `off floor`, `hard_d_seg`, `d_pose`, `prefix`, and `evenly-strided`. In addition to all chartered
seeds, the governing results were:

- `.omx/research/ddm_s1_trained_renderer_diagonal_20260825.md` supplied the preregistered GB1 damage
  expression and both-OFF/ON sequencing.
- `.omx/research/ddm_s1a_stage_a_adapter_20260825.md` and the actual retained evaluation rows showed that
  the live n60 selection is pair IDs `0,10,...,590`, not a contiguous prefix. WD3's selection contract
  independently labels it fixed evenly strided. This changed the instrument: it rejects any prefix or
  other subset, declares the exact selection on every row, and does not transfer NA2's prefix-bias
  multiplier to these rows. The MPS/n60 advisory boundary still binds.
- `.omx/research/ddm_ar1b_archive_residue_purchase_20260822.md` and
  `.omx/research/ddm_wa1_week_audit_gestalt_toy_orphan_synergy_20260825.md` corroborated the measured
  30,856 B GB1 renderer block.
- `.omx/research/ddm_tx1_exchange_rate_cashout_20260823.md`, the canonical-equations registry, and hot
  state corroborated the exact `25/37,545,489` exchange rate. The code uses the exact ratio, not the
  rounded `6.658e-7` display value.
- `.omx/research/ddm_fb1_sub012_feasibility_bound_20260823.md` closes submission framing: even
  renderer-to-zero at fixed GB1 distortion remains about 0.12757, so Stage A is only an input to Stage B.
- `.omx/research/ddm_w72_distortion_advisory_20260823.md` and
  `.omx/research/ddm_dg2_diagonal_distortion_verdict_20260824.md` reinforced that pose damage must remain
  explicit rather than being collapsed into a seg-only selector.

## Handoff

Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Owner: MAIN. Consumer store:
`/Volumes/APDataStore/pact/ddm_s1e_off_floor_adjudicator/both_off_endpoint.json`. Fire only when seed
20260816 has a complete `STAGE_CONTROLLER_RESULT.json`, joined `wd3_epoch_0065.pt`, and
`epoch_0065_n60.json`. MAIN then reruns the committed instrument before authorizing ON seed 20260815.

The executed receipt did not move the frontier: **gb1 — S 0.14811799921260607 @ 180,215 B
[contest-CUDA T4, n600], archive sha256 ba1f38301a45308e99ecc6fb86ff9bf7d212e5c9cb023c13cef410620761e3a4.**
