# Codex Findings - ATW2 Full-Candidate Generation Local Blocker

**Timestamp (UTC)**: 2026-05-21T06:24:00Z  
**Scope**: Probe whether the local machine can produce a 600-pair ATW2 candidate archive for the CDF compaction lane.  
**Verdict**: BLOCKED_LOCAL_NO_CUDA_FOR_FULL_ATW2_CANDIDATE

## Summary

The ATW2 CDF compaction stack is implementation-ready but the local inventory has no full ATW2 candidate archive. Codex probed the next obvious step: generate a 600-pair ATW2 archive locally from the full trainer with auth eval skipped.

The probe failed closed before writing an archive:

- CPU full training is refused by the canonical substrate trainer policy.
- Local CUDA is unavailable.
- Required local inputs are present, so this is a compute-substrate blocker, not missing custody.

No score claim, promotion claim, or exact-eval readiness claim is made.

## Commands

CPU full-candidate probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/train_substrate_atw_codec_v2.py \
  --output-dir experiments/results/atw2_full_candidate_probe_20260521T0624Z \
  --epochs 0 --device cpu --skip-auth-eval --max-pairs 600
```

Result:

```text
[atw_codec_v2] --device cpu is permitted only with --smoke per CLAUDE.md 'MPS auth eval is NOISE' + 'EMA — non-negotiable' + full-training-needs-CUDA convention. Use --device cuda for promotion-grade training. CPU smoke is allowed only when deterministic-bytes acceptable.
```

CUDA full-candidate probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/train_substrate_atw_codec_v2.py \
  --output-dir experiments/results/atw2_full_candidate_probe_cuda_20260521T0624Z \
  --epochs 0 --device cuda --skip-auth-eval --max-pairs 600
```

Result:

```text
[atw_codec_v2] --device cuda requested but cuda not available
```

## Local Custody Check

Present:

- `upstream/videos/0.mkv` - 36 MB
- `upstream/models/segnet.safetensors` - 37 MB
- `upstream/models/posenet.safetensors` - 53 MB

No output directory was created for either failed probe.

## Interpretation

The blocker for ATW2 CDF full-candidate compaction is now narrower:

1. Not scanner/compactor implementation.
2. Not missing local upstream video/model custody.
3. Not local CPU, because full ATW2 training intentionally refuses CPU.
4. It is a CUDA/provider dispatch problem.

The next frontier-moving action for this lane is a claimed CUDA provider timing smoke or full-candidate archive build for ATW2, with `--skip-auth-eval` acceptable for archive generation custody and no score claim. Once a 600-pair ATW2 `archive.zip` exists, rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/compact_atw2_cdf_candidates.py \
  <candidate-root> --device cpu --full-candidate-only
```

If no CUDA provider is available, the ATW2 CDF lane should remain blocked and effort should shift to a lane with an existing full candidate archive or a local-CPU/MPS-safe timing smoke that does not violate promotion-custody rules.
