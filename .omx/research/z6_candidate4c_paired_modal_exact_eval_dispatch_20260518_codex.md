# Z6 Candidate 4c Paired Modal Exact-Eval Dispatch - 2026-05-18

## Status

Race Mode was active and the full-600 zero-epoch Candidate 4c archive pair had
a byte-closed post-harvest exact-eval handoff. Codex actuated the paired Modal
auth-eval handoff for both archive modes:

- full FiLM predictor archive
- identity-predictor disambiguator archive

This is a score-bearing eval dispatch attempt, not a score claim. All four
axes remain `score_claim=false` and `promotion_eligible=false` until recovered
Modal artifacts are adjudicated.

## Pre-dispatch custody

Live claim state before dispatch:

```text
CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=507 unparsable_timestamp=0 invalid_lane_id=0
```

Plan artifacts:

- `.omx/state/candidate4c_launch_packet/candidate4c_full_paired_modal_plan_20260518T110000Z.json`
- `.omx/state/candidate4c_launch_packet/candidate4c_identity_paired_modal_plan_20260518T110000Z.json`

No existing promotable anchors were found for either archive:

- `axes_skipped_due_to_existing_anchor.contest_cuda=false`
- `axes_skipped_due_to_existing_anchor.contest_cpu=false`

Archive/runtime custody:

| mode | archive bytes | archive sha256 | CUDA runtime tree | CPU runtime tree | runtime content tree |
|---|---:|---|---|---|---|
| full | `211866` | `5b371490b4459b85e95e6173653fc1b9aa78010681862ec51111166a6c867c4b` | `09f53c44b021314639a94f48b4ceb143fe3b279ab88e5c0b2dc19ad27a3ae242` | `f888831ce915d37c2a6a3069646a02291569a948a1c2108b1c9ff76b1e3ab57b` | `141af7545581cf5e0eca4b096674c0d36ce57f445c06ecad1e5df2aff49eb02c` |
| identity | `212047` | `e6cd9bf67ca68bcdf93aa0e804435b75b813e420d5e3964b3a6cb6cee28e3589` | `09f53c44b021314639a94f48b4ceb143fe3b279ab88e5c0b2dc19ad27a3ae242` | `f888831ce915d37c2a6a3069646a02291569a948a1c2108b1c9ff76b1e3ab57b` | `141af7545581cf5e0eca4b096674c0d36ce57f445c06ecad1e5df2aff49eb02c` |

## Detached Modal calls

| mode | axis | lane id | instance/job id | call id | output dir |
|---|---|---|---|---|---|
| full | `[contest-CUDA]` | `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_full_contest_cuda` | `candidate4c_full_5b371490b445_cuda` | `fc-01KRXC3V6N13J3H9R5XZSXHPQ1` | `experiments/results/modal_auth_eval/candidate4c_full_5b371490b445_cuda` |
| full | `[contest-CPU]` | `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_full_contest_cpu` | `candidate4c_full_5b371490b445_cpu` | `fc-01KRXC4EZ3GY615KF1EJE33VZ2` | `experiments/results/modal_auth_eval_cpu/candidate4c_full_5b371490b445_cpu` |
| identity | `[contest-CUDA]` | `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_identity_contest_cuda` | `candidate4c_identity_e6cd9bf67ca6_cuda` | `fc-01KRXC3WYZKE2ZEE04R7P714KE` | `experiments/results/modal_auth_eval/candidate4c_identity_e6cd9bf67ca6_cuda` |
| identity | `[contest-CPU]` | `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_identity_contest_cpu` | `candidate4c_identity_e6cd9bf67ca6_cpu` | `fc-01KRXC4M333B38CRV7Q6HXNVN1` | `experiments/results/modal_auth_eval_cpu/candidate4c_identity_e6cd9bf67ca6_cpu` |

Post-dispatch live claim summary:

```text
CLAIM_SUMMARY active=4 stale_nonterminal=0 terminal_latest=507 unparsable_timestamp=0 invalid_lane_id=0
```

Initial recovery attempt at `2026-05-18T11:05:20Z` returned `status=pending`
for all four calls.

## 2026-05-18T11:07Z CUDA recovery

Both `[contest-CUDA]` calls recovered and wrote terminal dispatch-claim rows:

- full: `completed_contest_cuda_modal_auth_eval_recovered`
- identity: `completed_contest_cuda_modal_auth_eval_recovered`

Recovered artifacts:

- full:
  `experiments/results/modal_auth_eval/candidate4c_full_5b371490b445_cuda/contest_auth_eval.json`
- identity:
  `experiments/results/modal_auth_eval/candidate4c_identity_e6cd9bf67ca6_cuda/contest_auth_eval.json`

CUDA result review:

| mode | score | seg | pose | bytes | rate term | pose term | seg term |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | `90.58142803863508` | `0.50482631` | `159.66197205` | `211866` | `0.14107287296218196` | `39.9577241656729` | `50.482631` |
| identity | `90.58427695093009` | `0.50482631` | `159.68377686` | `212047` | `0.14119339343269705` | `39.96045255749739` | `50.482631` |

`[contest-CUDA]` identity-minus-full:

- score: `0.0028489122950077217`
- SegNet: `0`
- PoseNet: `0.021804809999991903`
- rate: `0.0000048208188206042`
- archive bytes: `181`

Classification:

- `measured_config_regression_zero_epoch_control`
- `full_lower_than_identity_on_contest_cuda_but_below_delta`
- `not_promotion_authority`

The zero-epoch archive pair is now proven bad on `[contest-CUDA]`. This does
not kill Candidate 4c as a trained method; it blocks any accidental promotion
of the zero-epoch handoff packet and preserves the exact anchor for scorer
geometry and no-op/disambiguator review.

Current active claims after CUDA recovery:

```text
active=2
active lanes: full [contest-CPU], identity [contest-CPU]
```

## 2026-05-18T11:11Z CPU recovery and terminal claim closure

Both `[contest-CPU]` calls recovered and wrote terminal dispatch-claim rows:

- full: `completed_contest_cpu_modal_auth_eval_recovered`
- identity: `completed_contest_cpu_modal_auth_eval_recovered`

Final live claim summary after CPU recovery:

```text
CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=511 unparsable_timestamp=0 invalid_lane_id=0
```

Recovered artifacts:

- full:
  `experiments/results/modal_auth_eval_cpu/candidate4c_full_5b371490b445_cpu/contest_auth_eval.json`
- identity:
  `experiments/results/modal_auth_eval_cpu/candidate4c_identity_e6cd9bf67ca6_cpu/contest_auth_eval.json`

CPU result review:

| mode | score | seg | pose | bytes | rate term | pose term | seg term |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | `90.57816474855734` | `0.5048244` | `159.63742065` | `211866` | `0.14107287296218196` | `39.95465187559516` | `50.48244` |
| identity | `90.58102532784203` | `0.5048244` | `159.65931702` | `212047` | `0.14119339343269705` | `39.95739193440933` | `50.48244` |

`[contest-CPU]` identity-minus-full:

- score: `0.0028605792846860822`
- SegNet: `0`
- PoseNet: `0.021896370000007437`
- rate: `0.0000048208188206042`
- archive bytes: `181`

Final classification:

- `paired_contest_cuda_cpu_zeroepoch_control_recovered`
- full FiLM lower than identity on both axes
- both axis deltas below `decision_delta_s=0.005`
- zero-epoch packet measured bad, not promotion material
- trained Candidate 4c remains unresolved by this control

## Recovery commands

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval/candidate4c_full_5b371490b445_cuda

.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval_cpu/candidate4c_full_5b371490b445_cpu

.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval/candidate4c_identity_e6cd9bf67ca6_cuda

.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval_cpu/candidate4c_identity_e6cd9bf67ca6_cpu
```

Recovery must close each matching dispatch claim with a terminal row. Until
then, do not re-dispatch these four lane ids.

## Interpretation

This turn advanced Candidate 4c from byte-closed local exact-eval handoff to
four active Modal auth-eval calls. The next score-moving action is harvest and
adjudication of the returned artifacts:

- recompute formula from recovered components;
- keep `[contest-CUDA]` and `[contest-CPU]` axes separate;
- compare full vs identity under matched archive/runtime custody;
- update posterior and dispatch/lane ledgers only from harvested artifacts;
- terminalize each active claim on success or failure.
