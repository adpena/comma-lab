# DP1 Engineering Status Gate - 2026-05-15

## Artifact

New guard:

```bash
PYTHONPATH=src .venv/bin/python tools/dp1_engineering_status_gate.py \
  --strict \
  --output-json .omx/research/dp1_engineering_status_gate_20260515_codex.json
```

Live result:

```text
engineering_status=implemented_proxy_ready_untrained_real_prior_missing
classification=untrained_unpromoted_promising_substrate
false_claims=0
next_gate=dp1_comma2k19_onechunk_cpu_advisory_source_custody
next_gate_status=blocked_until_source_supplied
```

The gate composes the current DP1 planning, smoke, Tier-C advisory, and tiny
full CPU-advisory artifacts. It fails closed on `score_claim=true`,
`score_claim_valid=true`, `promotion_eligible=true`, or `rank_or_kill_eligible`
without contest score evidence.

## Current Status

DP1 is implemented enough to smoke, parse, probe, run Tier-C advisory analysis,
apply archived codebook bytes at inflate time, and execute a tiny full CPU
advisory path. It has not produced a real Comma2k19-trained prior, a
promotion-grade deployment packet, or a legitimate contest CPU/CUDA score.

The live status packet records:

- `smoke_archive_materialized=true`
- `tier_c_advisory_present=true`
- `tiny_full_cpu_advisory_ran=true`
- `real_dataset_source_ready=false`
- `exact_score_artifact_present=false`

Blocking evidence:

- `dp1_real_dataset_source_manifest_missing`
- `dp1_trained_real_prior_missing`
- `dp1_paired_contest_cpu_cuda_exact_eval_missing`

## Next Executable Gate

The next DP1 action is still the no-score, source-custody probe:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python \
  experiments/train_substrate_pretrained_driving_prior.py \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --dataset-name comma2k19 \
  --comma2k19-chunks-dir "$DPP_COMMA2K19_CHUNKS_DIR" \
  --epochs 1 \
  --batch-size 1 \
  --max-pairs 4 \
  --val-pair-count 1 \
  --max-distillation-frames 128 \
  --max-distillation-chunks 1 \
  --skip-auth-eval \
  --output-dir experiments/results/dp1_comma2k19_onechunk_cpu_advisory_<UTC>
```

Success is not a score. Success means a `dp1_dataset_source_manifest.v1` with
`dataset_name=comma2k19`, complete SHA-256 coverage or a pinned prebuilt
codebook source mode, `score_claim=false`, `promotion_eligible=false`, and a
written `archive.zip` plus `provenance.json`.

Only after that source-custody artifact exists should DP1 move to a provider
timing smoke, then byte-closed paired `[contest-CUDA]` and `[contest-CPU]` eval.
