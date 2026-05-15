# D4 Smoke Failure Review - Temp Auth-Eval Evidence Path - 2026-05-15

score_claim: `false`
promotion_eligible: `false`
axis: `[contest-CUDA]` attempted, no score emitted

## Run

- lane: `lane_d4_wyner_ziv_frame_0_substrate_20260514`
- job: `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T051529Z__smoke__50ep`
- Modal call: `fc-01KRN0ZTJ14WHG94C4VDXXTMR7`
- recovered directory:
  `experiments/results/lane_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T051529Z__smoke__50ep_modal`
- terminal claim: `failed_modal_training_rc_1`
- elapsed: `962s`
- cost-band anchor: `$0.15762615123505003` on T4

## Classification

`infrastructure_guardrail_failure`

The run trained and built a WZF01 archive, but exact auth eval did not start
because the trainer passed a temp-storage JSON path into `contest_auth_eval.py`:

```text
contest_auth_eval evidence path is under temp storage:
/tmp/pact/lane_substrate_d4_wyner_ziv_frame_0_results/output/auth_eval.json.
Choose a durable repo/provider work dir or pass --allow-temp-work-dir for
diagnostic scratch only.
```

This is not a model-score result and must not update D4's score posterior
except as a runtime/config failure.

## Root Cause

Modal lane trainers execute from a writable `/tmp/pact` copy. D4 wrote
`auth_eval.json` directly under `args.output_dir`, which mapped to
`/tmp/pact/lane_substrate_d4_wyner_ziv_frame_0_results/output`. The hardened
`contest_auth_eval.py` contract correctly rejects score-grade evidence paths
under temp storage.

## Fix

`experiments/train_substrate_d4_wyner_ziv_frame_0.py` now resolves two auth-eval
paths:

- `auth_eval_gate_json_path`: non-temp path used for `contest_auth_eval.py`
  custody validation, defaulting to `/root/d4_wyner_ziv_frame_0_auth_eval/...`
  when `output_dir` is under temp storage.
- `auth_eval_json_path`: harvested local copy under the lane output directory,
  written only after the gate JSON has succeeded.

The trainer also records both paths in provenance. Focused tests cover the
temp-workspace and durable-output cases.

## Reactivation

Refire D4 after this fix with the same 50-epoch, max-pairs 200 smoke recipe.
The previous archive is useful only as a training artifact; it lacks a valid
auth-eval score and is not promotion/rank/kill evidence.
