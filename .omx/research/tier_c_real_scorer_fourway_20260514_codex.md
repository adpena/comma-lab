# Tier-C real-scorer four-way harvest - 2026-05-14

## Scope

This checkpoint preserves the first real-scorer Tier-C four-way local run after
adding DP1 archive decode support to `tools/mdl_scorer_conditional_ablation.py`.

Evidence axis:

- `[macOS-CPU advisory only]`
- real scorer, pair-sampled with `--pair-samples 1`
- no score claim
- no promotion eligibility

## Command

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python \
  tools/run_tier_c_with_real_scorer.py \
  --execute \
  --pair-samples 1 \
  --scorer-batch-size 1 \
  --output-dir experiments/results/tier_c_real_scorer_fourway_codex_execute1_dp1fix_20260514
```

Result: success in `77.306s`.

## Artifacts

- manifest:
  `experiments/results/tier_c_real_scorer_fourway_codex_execute1_dp1fix_20260514/tier_c_real_scorer_manifest.json`
  - sha256:
    `394d1dbc0da5c41fc49c7cbcf4907f7f92239cb0d7330d4ef1c46f94c794d894`
- markdown summary:
  `experiments/results/tier_c_real_scorer_fourway_codex_execute1_dp1fix_20260514/tier_c_real_scorer_manifest.md`
  - sha256:
    `1a1697203fa3ffd5731ec5e3c9e9d4465ca87ae78ac74eb9e9783b4436391e20`

Per-archive JSONs:

- A1:
  `a1_tier_c_real_scorer.json`
  - sha256:
    `5822a603dc823f22abc35793455b1176e380bbba3035aa9e8a01806988bf84b6`
- PR106 r2:
  `pr106_r2_tier_c_real_scorer.json`
  - sha256:
    `ae26ecdd8749f83c8ee1f798ba2663367170d9add809eed35557810ec7e8f33e`
- IBPS1 C6 5ep:
  `ibps1_c6_5ep_tier_c_real_scorer.json`
  - sha256:
    `55b1532c1cc7a8e5ecea78e0df12570ed667aaba3bdd8ace82348e4164fae3e9`
- DP1 smoke:
  `dp1_smoke_tier_c_real_scorer.json`
  - sha256:
    `3e173e9f5fd27fd0555457b8714e2d50bad40ad3052971baef2e19157b803369`

## Result snapshot

| label | bytes | baseline components | verdict | density | latent sigma1 |
| --- | ---: | ---: | --- | ---: | ---: |
| A1 | 178262 | 0.1287940 | within_class | 0.99080177 | 2.96045 |
| PR106 r2 | 186822 | 0.15050685 | within_class | 0.97910983 | 4.68029 |
| IBPS1 C6 5ep | 224481 | 54.6717 | within_class | 0.87252306 | 0.308543 |
| DP1 smoke | 12032 | 104.2382 | across_class | 0.13258188 | 0.0 |

Interpretation:

- This is an evidence-quality upgrade for Tier-C density estimation.
- It is not a score claim.
- The one-pair run is too small to rank archives or promote C6/DP1.
- DP1 decode support is now present, so future real-scorer Tier-C sweeps can
  include DP1 instead of failing closed at `grammar dp1 not supported`.

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_mdl_ablation_tier_c_dp1.py \
  src/tac/tests/test_run_tier_c_with_real_scorer.py -q
```

Result: `29 passed`.
