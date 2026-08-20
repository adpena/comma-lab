# 09 — Commands to run

## Shared parse-back selection tests

```bash
pytest -q src/tac/tests/test_long_training_archive_selection.py   -k "parseback or replay_required"
```

## HiNeRV target-region birth smoke

```bash
python tools/trace_nerv_crux.py   --family hi_nerv   --repo-ref 69b4c523d2696978cae4423d95488d15d851e8cd   --pairs 4   --steps 64   --trace init,forward,scorer,grad,update,fakequant,archive,parseback   --require-metric selection_health_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass   --out experiments/results/hinerv_target_region_birth_trace_20260606/trace.jsonl
```

## HiNeRV scoped birth actuator smoke

```bash
python tools/run_hinerv_target_region_birth_smoke.py   --pairs 4   --steps 64   --update-patterns head_rgb_1,feature_grids,fine_injector   --require-frontier-margin-improvement   --require-pose-trust-region   --out experiments/results/hinerv_target_region_birth_smoke_20260606
```

## SNeRV full TUB source-forward closure

```bash
python tools/prove_snerv_tub_full_source_forward_closure.py   --official-repo /Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/SNeRV   --checkpoint /Volumes/VertigoDataTier/pact/snerv_fixtures/snerv_t_official.pt   --require-output2-decoder   --require-temporal-encoder   --out .omx/research/snerv_official_tub_full_source_forward_closure_20260606.json
```

## Section value-per-byte

```bash
python tools/profile_nerv_section_value.py   --before experiments/results/run_a/archive.zip   --after experiments/results/run_b/archive.zip   --before-replay experiments/results/run_a/replay.json   --after-replay experiments/results/run_b/replay.json   --out experiments/results/run_b/section_value.json
```

## Long-run preflight gate

```bash
python tools/validate_nerv_long_run_gate.py   --training-artifact experiments/results/<run>/training_artifact.json   --require-parseback-selection   --require-target-region-birth   --require-pose-marginal   --require-section-value   --require-receiver-proof
```
