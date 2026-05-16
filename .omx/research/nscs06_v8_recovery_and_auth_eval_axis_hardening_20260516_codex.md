# NSCS06 v8 Modal recovery and auth-eval axis hardening

Date: 2026-05-16
Agent: Codex
Lane: `lane_nscs06_v8_path_b_wavelet_residual_substrate_build_20260516`
Job: `substrate_nscs06_v8_path_b_wavelet_modal_t4_dispatch_20260516T184838Zsmoke_retry`
Modal call id: `fc-01KRS1XP83TE8BV6QBJ52MC1MG`

## Recovery receipt

`experiments/modal_recover_lane.py` recovered the detached Modal training call
and appended the terminal dispatch claim:

```text
status: completed_modal_training_recovered_no_score_claim
score_claim: false
promotion_eligible: false
elapsed_seconds: 1207.326888717
estimated_cost_usd: 0.1978674623175083
```

Current dispatch-claim summary after recovery:

```text
CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=925 unparsable_timestamp=0 invalid_lane_id=4
```

The recovered raw artifacts live under the ignored Modal result directory:

```text
experiments/results/lane_substrate_nscs06_v8_path_b_wavelet_modal_t4_dispatch_20260516T184838Zsmoke_retry_modal/
```

This memo is the durable tracked no-signal-loss surface for the recovery.

## Advisory auth-eval result

The recovered auth-eval JSON is a diagnostic Modal CPU artifact, not a
promotion result and not a method kill:

```text
path: experiments/results/lane_substrate_nscs06_v8_path_b_wavelet_modal_t4_dispatch_20260516T184838Zsmoke_retry_modal/lane_nscs06_v8_path_b_wavelet_results/output/contest_auth_eval_cpu.json
sha256: 1036e3cde8f7e9dd138ddd6acccc2642815d55af498e56452001214ada842280
score_axis: diagnostic_cpu
evidence_grade: B
score_claim: false
score_claim_valid: false
promotion_eligible: false
exact_cuda_eval_complete: false
rank_or_kill_eligible: false
diagnostic_blocker: modal_training_wrapper_auth_eval_advisory_only
canonical_score: 104.98298682733697
avg_posenet_dist: 88.42543793
avg_segnet_dist: 0.75236231
archive_size_bytes: 15530
evaluate_elapsed_seconds: 1024.9264560679999
inflate_elapsed_seconds: 3.9221402720000356
contest_auth_eval_elapsed_seconds: 1040.852344851
```

Custody:

```text
archive.zip bytes: 15530
archive.zip sha256: 03ef568bc918f22c10fed33dbc3e43b29d6bdc4d0243e15c1bb8b616f1da9530
0.bin bytes: 35547
0.bin sha256: 19524bb3a1439fc8cbc981010e28e3613eab7941b2370ebacb6984d5b97d0b0f
run.log sha256: e4de46605ebe0be4092f2bcc7049da5076fb8e321ec6a6fe7cb81bf138a85a53
runtime_tree_sha256: e960056793e19bc08db6a7d6b3657582f656ba3728d34d808c187e469cb66270
inflated_raw_aggregate_sha256: 0e6399dc6acc14fe85bcb8ae31fb4ee331fea5a6151c3ff184e210d27186cce9
```

Classification:

```text
measured-config regression / diagnostic CPU only / no score claim /
not promotion eligible / not rank-or-kill eligible / not a lane death.
```

The score is dominated by component collapse and must be investigated with
xray tooling before any broader conclusion about the wavelet residual mechanism.

## Bug found

The Modal wrapper forces `AUTH_EVAL_DEVICE=cpu` for advisory auth eval. The
trainer and remote lane script still expected `contest_auth_eval_cuda.json`
because they derived the expected file from `NSCS06_V8_DEVICE=cuda` rather than
from the actual auth-eval device.

The canonical `gate_auth_eval_call` prevented a phantom CUDA file by rewriting
the output to `contest_auth_eval_cpu.json`, but the remote script then logged:

```text
auth_eval_artifact_missing path=/tmp/pact/lane_nscs06_v8_path_b_wavelet_results/output/contest_auth_eval_cuda.json
```

That was an engineering/custody bug in result harvesting and completion
labelling, not a substrate result.

## Fix landed

The follow-up code change makes NSCS06 v8 auth-eval custody device-aware:

- `experiments/train_substrate_nscs06_v8_path_b_wavelet.py` now derives
  `contest_auth_eval_{auth_eval_device}.json` from the explicit auth-eval
  device, passes `auth_eval_device=...`, and opts into
  `return_non_cuda_result=True`.
- The trainer provenance now preserves generic `auth_eval_score`,
  `auth_eval_score_axis`, `auth_eval_device`, `auth_eval_evidence_grade`,
  `auth_eval_score_claim_valid`, and the full `auth_eval_result` object.
- Contest-CUDA posterior updates and score claims remain gated on a valid
  `contest_cuda` result only.
- `scripts/remote_lane_substrate_nscs06_v8_path_b_wavelet.sh` now derives the
  expected auth-eval artifact from `AUTH_EVAL_DEVICE` first and labels the
  completion marker from the JSON `score_axis` and `score_claim_valid` fields.
  Modal CPU advisory artifacts are logged as
  `[training-artifact-no-score-claim]`, never as `[contest-CUDA]`.
- Regression tests assert the trainer is device-aware and the remote script has
  no unconditional contest-axis completion tag.

## L5/L5 v2 relevance

The recovered terminal claim removes the unrelated active-claim blocker that
was visible while preparing the L5 v2 paired-measurement path. L5 v2 still must
clear its own probe-observation, paired CPU/CUDA plan, source-schedule freshness,
side-info effect-curve, and timing-custody gates before any dispatch or score
claim.

The broader rule applies to all L5 staircase and non-HNeRV frontier lanes:
paired CPU/CUDA evidence is useful only when the artifact name, JSON content,
completion marker, lane claim, and promotion axis all agree. Any mismatch is a
custody bug until fixed and reviewed.

## Next actions

1. Rerun NSCS06 v8 only after the fixed script is present in the Modal runtime.
2. Before declaring NSCS06 v8 falsified, xray the inflated raw output against
   PoseNet/SegNet components and inspect whether collapse comes from the
   grayscale/channel path, class-context coding, or upsample/inflate geometry.
3. Do not use this Modal CPU artifact for ranking, promotion, or kill decisions.
4. Keep L5 v2 priority on byte-closed paired measurement and exact axis-labelled
   custody rather than local-basin polish.
