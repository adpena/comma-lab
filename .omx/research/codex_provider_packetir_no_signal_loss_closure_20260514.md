# Codex Provider And PacketIR Closure Update - 2026-05-14

## Scope

Execution pass after commit `220323512`, focused on no-signal-loss provider
state and PR106/R2 PacketIR closure.

## Landed engineering

- Hardened `tools/harvest_modal_calls.py` so an incomplete or failed
  `modal_training_terminal_claim.json` marker no longer masks stronger local
  harvest evidence (`rc`, elapsed time, artifact count, crash kind, or payload
  files). This prevents re-running the harvester against provider-GC'd calls
  from overwriting existing terminal summaries with weaker `NotFoundError`
  summaries.
- Hardened `tools/trigger_gha_cpu_eval.py` so diagnostic `--skip-trigger` and
  workflow-dispatch failures append terminal claim rows for the claimed
  `lane_id` / `instance_job_id` when no GitHub Actions run exists. This closes
  the phantom-active-claim failure mode.
- Extended `tools/build_pr106_r2_packetir_exact_closure.py` to merge a separate
  PacketIR identity proof into closure input shape. This lets HLM1/xmember
  candidate manifests close without one-off glue.
- Hardened `scripts/pre_submission_compliance_check.py` so provider-temp
  `inflate_script` paths are sanitized as `submission_dir/inflate.sh` instead
  of falling back to `submissions/pr103_pr106_final_runtime/inflate.sh`.
  Contest-final dispatch-claim runtime binding now uses the operator-supplied
  scored runtime tree when provided, not the mutable local release surface tree.
- Registered `PR106-R2-HDM4-HLM1-XMEMBER` in the HNeRV scorecard and
  `tools/operator_briefing.py` phase-1 packet list, and extended all-lanes Gate
  #26 to prove PacketIR/runtime consumption for the xmember archive.
- Hardened terminal exact-eval suppression so operator packets can fail closed
  on "same lane + same archive + terminal result" even when the local runtime
  tree drifted after the scored run. The xmember packet is now suppressed with a
  runtime-mismatch terminal blocker rather than exposing stale submit commands.
- Fixed `tools/build_pr106_hlm1_exact_eval_packet.py` refresh steps so
  non-default result directories, manifests, lane IDs, output dirs, and approval
  state are preserved. The prior xmember refresh command would have silently
  regenerated the default HLM1 packet into the xmember path.

## Artifact closure

- Built HLM1 xmember closure:
  `experiments/results/pr106_r2_hdm4_hlm1_xmember_closure_20260514_codex/closure.json`
- Refreshed HNeRV scorecard:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/scorecard.json`
  now selects `PR106-R2-HDM4-HLM1-XMEMBER` as the internal exact-CUDA
  score-lowering frontier.
- Refreshed strict xmember compliance:
  `experiments/results/pr106_r2_hdm4_hlm1_xmember_candidate_20260514_codex/pre_submission_compliance.contest_final.strict_20260514_codex.json`
  passes and records sanitized provider runtime custody without the PR103
  fallback path.
- Classification:
  `exact_measured_improves_packetir_source_cuda`
- Candidate archive:
  `391400008b69e66f8bd522f4eb2a53c465e58a17e536d171caf039f9e51e874f`
  (`186415` bytes)
- Source archive:
  `8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0`
  (`186423` bytes)
- Exact CUDA delta vs source:
  `-0.000005326871624966589`
- Closure blockers:
  none
- Duplicate-dispatch blocker:
  `same_candidate_archive_already_exact_evaluated`
- Operator packet state:
  `ready_for_submit=false`; terminal blocker is
  `same_lane_terminal_runtime_mismatch_for_same_archive` because the local
  report-only release surface changed after the exact CUDA run. The scored
  runtime remains `c25a4c1e4ee047b7b669978bc4c139327e37db83592a8cda32d086a579577ccf`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_gha_cpu_eval_harness.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_build_pr101_finetuned_archive_codec_dir.py src/tac/tests/test_modal_training_harvest_summary.py -q`
  - `56 passed`
- `.venv/bin/python tools/harvest_modal_calls.py --repo-root .`
  - all `73` Modal metadata rows classify as harvested in plan-only mode
- `.venv/bin/python scripts/pre_submission_compliance_check.py ... --contest-final --strict`
  - xmember strict compliance passed; sanitized eval command uses
    `submission_dir/inflate.sh`
- `.venv/bin/python experiments/build_hnerv_frontier_scorecard.py ... PR106-R2-HDM4-HLM1-XMEMBER=...`
  - scorecard regenerated with the xmember exact CUDA row and xmember section
    profile

## Interpretation

This is a small measured rate win, not a new score-floor breakthrough. Its
value is closure discipline: the exact archive/runtime/axis key is now
recorded as already measured, so future work should move to a new consumed-byte
mutation or a larger sidecar/compression candidate rather than redispatching
the same packet.
