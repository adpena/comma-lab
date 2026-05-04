# Trained Renderer Transplant Dispatch Handoff - 2026-05-03

Evidence grade: empirical dry-run planning only. Score claim: false. Remote dispatch: none.

## Custody

- Source archive: `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip`
- Source bytes: `276481`
- Source SHA-256: `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- Renderer export: `None`
- Renderer export SHA-256: `None`

## Break-Even

- Current score: `0.31516575028285976`
- Strict target: `< 0.314`
- Bytes to save at unchanged distortion: `1751`
- Max byte-only crossing archive bytes: `274730`

## Readiness

- Exact-eval dispatch ready: `False`
- Blockers: `candidate archive is not available yet`; `no terminal Modal renderer export is available locally`; `recover the active Modal call, then provide the recovered QZS3 renderer export path`; `run transplant preflight to build candidate archive`

## Next Commands

```bash
.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KQP9K42CAWJH7XEV4KC0V28M
.venv/bin/python experiments/build_renderer_shrink_candidate.py --source-archive experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip --renderer-export '<recovered_renderer_qzs3.bin>' --output-dir experiments/results/trained_renderer_transplant_recovery_worker_20260503/c091_readiness/preflight --qzs3-block-sizes 32,48,64,96,128 --force
.venv/bin/python experiments/preflight_renderer_transplant_pose_safety.py --source-archive experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip --candidate-archive '${CANDIDATE_ARCHIVE}' --output-json 'experiments/results/trained_renderer_transplant_recovery_worker_20260503/c091_readiness/pose_safety/${CANDIDATE_ID}.json'
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id c091_trained_renderer_self_compression_transplant --platform lightning --instance-job-id 'exact_eval_${CANDIDATE_ID}_trained_transplant_READY' --agent codex:gpt-5 --status eval --notes 'trained_renderer_transplant candidate=${CANDIDATE_ID} archive_sha256=${ARCHIVE_SHA256}'
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval --job-name 'exact_eval_${CANDIDATE_ID}_trained_transplant_READY' --archive '${CANDIDATE_ARCHIVE}' --repo-dir /teamspace/studios/this_studio/pact --upstream-dir /teamspace/studios/this_studio/pact/upstream --machine g7e.4xlarge --adjudicate --baseline-score 0.31516575028285976 --baseline-archive-bytes 276481 --predicted-band 0.0 10.0 --regression-threshold 10.0 --infer-expected-archive --dispatch-lane-id c091_trained_renderer_self_compression_transplant --queue-metadata lane_id=c091_trained_renderer_self_compression_transplant --queue-metadata 'candidate_id=${CANDIDATE_ID}' --queue-metadata source_archive_sha256=03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746 --queue-metadata 'trained_renderer_sha256=${RENDERER_SHA256}' --queue-metadata purpose=trained_renderer_transplant_dispatch --component-trace --component-trace-top-k 80 --max-sane-score 10.0 --dry-run
```
