# Latest Report — 2026-04-25 (Rounds 23-26 hardening)

## Status
- Last verified contest-compliant auth: 2.01 (April 21 e2e through inflate.sh → upstream evaluate.py)
- SHIRAZ A100 mid-training (Phase 3 ETA ~30 min from 14:41Z)
- WILDE+GREEN killed earlier today after WILDE failure mode (TTO frames at GT range) discovered

## Round 23-26 hardening complete
- 19 CRITICAL bugs fixed in the deployment chain
- 23/23 preflight tests pass
- preflight_arity + preflight_profiles + check_codebase_drift clean against live repo
- `pipeline.py compress --profile X --video Y --checkpoint Z` is the canonical entry point
- Explicit CLI flags now correctly win over profile values
- Typo'd profile keys fail loud (no more silent-drop SHIRAZ class)
- I4LZ weight compression guarded (no arch header → falls back to FP4 for non-default arch)
- Full provenance in pipeline_config.json (git hash + GPU + PyTorch + platform + timestamp)
- deploy_vastai.py preflight check 6 detects stale ad-hoc run_pipeline.sh on remote

## Open architectural gaps (planned)
- train_distill.py has no --profile flag (profile dicts are documentation today)
- No deterministic CUDA / numpy / random seeding
- No data hashes / no uv.lock
- No CI runs the new tests
- No deploy_vastai.py download subcommand

## NUCLEAR queue (zero-GPU, run today)
1. Half-frame mask AV1 sweep at 384×512 (CRF 24..48)
2. SHIRAZ Phase 3 → immediate auth eval
3. Even-from-odd warp at inflate (Quantizr paradigm)

## Score projections (honest)
- Realistic best stacked with current architectures in 8 days: 0.8-1.4
- Beating Quantizr (0.33) needs new architecture family or training breakthrough
- Half-frame masks: rate -0.12 to -0.18 (high confidence)
- MXLZ renderer: rate -0.06 to -0.09 (medium confidence; arch_header guard added)
- Engineered corrections: SegNet -0.05 to -0.10 (low confidence, no auth validation yet)
- v2 beneficial_quant_noise: SegNet -0.10 to -0.20 (speculative, zero empirical backing)
