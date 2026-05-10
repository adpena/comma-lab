# T1 Modal guard failed before training: missing A1 canonical payload

Generated: `2026-05-10T09:35:00Z`

## Verdict

`t1_balle_modal_guard_a3311268_20260510T0831Z` closed terminally with no score
claim. This is an infrastructure/export-custody failure, not a model result.

Classification: `remote_script_failed / missing_canonical_a1_payload`.

## Custody

- Lane id: `t1_balle_128k_endtoend`
- Modal app id: `ap-vPBYzG1bLZRRfMubZKWkQK`
- Modal call id: `fc-01KR8GACB3NCW5TNG1E9YFPXHM`
- Metadata:
  `experiments/results/t1_balle_modal_guard_a3311268_20260510T0831Z/modal_metadata.json`
- Harvest summary:
  `experiments/results/t1_balle_modal_guard_a3311268_20260510T0831Z/harvest_summary.json`
- Harvested logs:
  `experiments/results/t1_balle_modal_guard_a3311268_20260510T0831Z/harvested_artifacts/modal_logs/remote_lane_t1_balle_endtoend.stdout.log`
- Terminal claim status: `failed_t1_modal_recovered_no_score_claim`

## Failure

Remote Stage 5 reached score-domain training and failed immediately:

```text
tac.paradigm_delta_epsilon_zeta.frozen_a1_encoder.FrozenA1EncoderError:
canonical A1 directory/symlink not found at
/workspace/pact/experiments/results/A1_canonical; operator must run
tools/designate_canonical_a1.py before T1 scaffold can load
```

Local state has `experiments/results/A1_canonical` as a symlink, but the Modal
T1 mount set did not include the symlink target or a declared canonical A1
payload. The remote worker therefore saw no canonical A1 directory.

## Evidence boundary

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- No archive was produced.
- No exact auth eval was run.
- No CUDA score exists.

The guard correctly failed closed and appended a terminal lane-claim row.

## Required fix before T1 rerun

Do not rerun this exact dispatch. The next T1 actuator must do one of:

1. Fail locally before dispatch if `experiments/results/A1_canonical` cannot be
   mounted as a real directory with required archive/checkpoint/latent files.
2. Materialize a compact canonical A1 payload bundle and mount it explicitly.
3. Run an explicit canonical-A1 designation/materialization step before Modal
   `.spawn()`, then record the payload paths and SHA-256s in Modal metadata.

Only after that fix should T1 be re-claimed and rerun.
