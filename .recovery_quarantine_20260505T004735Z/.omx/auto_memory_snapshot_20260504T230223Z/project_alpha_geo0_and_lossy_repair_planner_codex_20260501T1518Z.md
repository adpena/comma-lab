# Codex Memory - Alpha Geo0 Custody And Lossy Repair Planner - 2026-05-01T15:18Z

Context:

- Project: `/Users/adpena/Projects/pact`.
- Source progress ledger:
  `.omx/research/shannon_floor_nextwave_telemetry_and_research_20260430_codex.md`.
- Operating rule: CUDA auth eval is the only score truth; Alpha planner outputs
  are empirical and non-promotable until concrete archive bytes pass exact CUDA
  auth eval.

Alpha Geo0 diagnostic custody:

- Failed Lightning job:
  `alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z`.
- Local mirror:
  `experiments/results/lightning_batch/alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z/`.
- Manual SSH mirror was used because this custom Alpha diagnostic job did not
  emit `lightning_queue_metadata.json`, which the state-derived harvest wrapper
  expected. Only small JSON/log artifacts were copied.
- T4 preflight was clean: `cuda_available=true`, `device_name=Tesla T4`,
  `gpu_t4_match=true`.
- NVDEC probe passed.
- Failure stage: `decode_candidate_masks`.
- Failure: `RuntimeError: diagnose_nerv_geometry failed`.
- Geometry: global mask disagreement `0.012303911844889322`
  (`2,902,857 / 235,929,600` pixels).
- The job records `score_claim=false`, `promotion_eligible=false`,
  `passed=false`.
- Classification: scoped measured-implementation geometry failure for this
  Alpha Geo0/jsonfix40 candidate. Not a family kill and not score evidence.

Alpha lossy sparse-repair planner:

- Output directory:
  `experiments/results/alpha_lossy_repair_budget_planner_20260501/`.
- Report:
  `experiments/results/alpha_lossy_repair_budget_planner_20260501/alpha_lossy_repair_budget_plan.json`.
- Candidate specs:
  `experiments/results/alpha_lossy_repair_budget_planner_20260501/candidate_archive_specs/`.
- Command:
  `.venv/bin/python experiments/alpha_lossy_repair_budget_planner.py --output-dir experiments/results/alpha_lossy_repair_budget_planner_20260501 --lossy-base-bytes 60000,80000,100000,150000,250000 --max-specs 24 --force`.
- Planner summary: `104` budget records, `24` candidate archive specs.
- Evidence boundary: empirical/non-promotable; no archives built, no scorer
  loaded, no remote jobs launched, and exact CUDA auth eval is required before
  any score/rank/promotion claim.
- Verification:
  `.venv/bin/python -m py_compile experiments/alpha_lossy_repair_budget_planner.py src/tac/tests/test_alpha_lossy_repair_budget_planner.py`
  and
  `.venv/bin/python -m pytest src/tac/tests/test_alpha_lossy_repair_budget_planner.py -q`
  passed with `5 passed in 0.08s`.

Live exact-eval context:

- T4 promotion hedge jobs for identical `owv3_0120_stack` archive
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`,
  `609,963` bytes:
  `exact_eval_owv3_0120_stack_t4_20260501T150652Z` and
  `exact_eval_owv3_0120_stack_t4aws_20260501T151050Z`.
- As of `2026-05-01T15:15:54Z`, both jobs were running. Harvest first valid
  adjudicated result with the state-derived wrapper, then stop the duplicate if
  it remains active.
