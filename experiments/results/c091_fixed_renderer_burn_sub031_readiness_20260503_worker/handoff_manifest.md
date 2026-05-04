# Trained Renderer Transplant Dispatch Handoff - 2026-05-03

Evidence grade: empirical dry-run planning only. Score claim: false. Remote dispatch: none.

## Custody

- Source archive: `experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip`
- Source bytes: `276485`
- Source SHA-256: `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8`
- Renderer export: `None`
- Renderer export SHA-256: `None`

## Break-Even

- Current score: `0.31514430182167497`
- Strict target: `< 0.31`
- Bytes to save at unchanged distortion: `7726`
- Max byte-only crossing archive bytes: `268759`

## Readiness

- Exact-eval dispatch ready: `False`
- Blockers: `candidate archive is not available yet`; `no terminal Modal renderer export is available locally`; `recover the active Modal call, then provide the recovered QZS3 renderer export path`; `run transplant preflight to build candidate archive`

## Next Commands

```bash
.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KQP9K42CAWJH7XEV4KC0V28M
.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KQP9T1VD14785MG63H7JM5VK
.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KQP9T19Y7PMDETDN99WDMF2W
.venv/bin/python experiments/build_renderer_shrink_candidate.py --source-archive experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip --renderer-export '<recovered_renderer_qzs3.bin>' --output-dir experiments/results/c091_fixed_renderer_burn_sub031_readiness_20260503_worker/preflight --qzs3-block-sizes 64,128,256,512,1024
.venv/bin/python experiments/preflight_renderer_transplant_pose_safety.py --source-archive experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip --candidate-archive '${CANDIDATE_ARCHIVE}' --output-json 'experiments/results/c091_fixed_renderer_burn_sub031_readiness_20260503_worker/pose_safety/${CANDIDATE_ID}.json'
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id c091_fixed_renderer_burn_sub031_transplant --platform lightning --instance-job-id 'exact_eval_${CANDIDATE_ID}_trained_transplant_READY' --agent codex:gpt-5 --status eval --notes 'trained_renderer_transplant candidate=${CANDIDATE_ID} archive_sha256=${ARCHIVE_SHA256}'
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval --job-name 'exact_eval_${CANDIDATE_ID}_trained_transplant_READY' --archive '${CANDIDATE_ARCHIVE}' --repo-dir /teamspace/studios/this_studio/pact --upstream-dir /teamspace/studios/this_studio/pact/upstream --machine g7e.4xlarge --adjudicate --baseline-score 0.31514430182167497 --baseline-archive-bytes 276485 --predicted-band 0.0 10.0 --regression-threshold 10.0 --infer-expected-archive --dispatch-lane-id c091_fixed_renderer_burn_sub031_transplant --queue-metadata lane_id=c091_fixed_renderer_burn_sub031_transplant --queue-metadata 'candidate_id=${CANDIDATE_ID}' --queue-metadata source_archive_sha256=79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8 --queue-metadata 'trained_renderer_sha256=${RENDERER_SHA256}' --queue-metadata purpose=trained_renderer_transplant_dispatch --component-trace --component-trace-top-k 80 --max-sane-score 10.0 --dry-run
```
