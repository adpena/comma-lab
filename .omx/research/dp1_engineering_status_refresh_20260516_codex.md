# DP1 Engineering Status Refresh - 2026-05-16

## Command

```bash
.venv/bin/python tools/dp1_engineering_status_gate.py \
  --output-json experiments/results/dp1_status_20260516_codex/status.json
```

## Current Classification

- `classification`: `untrained_unpromoted_promising_substrate`
- `engineering_status`: `implemented_proxy_ready_untrained_real_prior_missing`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `ready_for_exact_eval_dispatch`: `false`
- `rank_or_kill_eligible`: `false`

## Confirmed Capabilities

- Planning/readiness manifest exists.
- DP1 smoke archive materialized:
  - bytes: `11924`
  - sha256: `e042d49c09a8e6e16dc0a2b00ef3b98bcccda1c04eb829684cc53c9ed6f0c764`
  - evidence grade: `[proxy]`
- Tiny full CPU advisory artifact exists:
  - bytes: `25914`
  - sha256: `e4918b420c7b40379e432a11beb5671430f96cbaf68fa7bae70423ae0af2fc0b`
  - evidence grade: `[proxy]`
- Tier-C advisory artifact exists.

## Blockers

- `dp1_real_dataset_source_manifest_missing`
- `dp1_trained_real_prior_missing`
- `dp1_paired_contest_cpu_cuda_exact_eval_missing`
- `source_custody_preflight.status=blocked_source_dir_not_configured`

## Next Executable Gate

`dp1_comma2k19_onechunk_cpu_advisory_source_custody`

Purpose: produce the first real-source DP1 training/runtime custody artifact
without a score claim.

Command template from the status gate:

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

Success criteria:

- `dataset_source_manifest.schema=dp1_dataset_source_manifest.v1`
- `dataset_name=comma2k19`
- selected chunk SHA-256 coverage complete, or prebuilt-codebook source mode
- source-custody preflight passed for local chunks
- `score_claim=false`
- `promotion_eligible=false`
- `archive.zip` and `provenance.json` written

## Evidence Boundary

This refresh is a status artifact only. It does not promote DP1, does not
retire DP1, does not claim score movement, and does not convert proxy or
macOS-CPU advisory evidence into contest authority.
