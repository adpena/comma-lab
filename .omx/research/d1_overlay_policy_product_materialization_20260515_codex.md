# D1 Overlay Policy Product Materialization - 2026-05-15

This is a byte-closed D1 candidate ledger, not a score claim.

## Code Path

- `tools/build_d1_overlay_policy_candidates.py` now sweeps channel policy,
  amplitude attenuation, and sign schedule as D1POLY1 metadata-only variants.
- `src/tac/tests/test_build_d1_overlay_policy_candidates.py` verifies that the
  product sweep writes deterministic archives whose embedded D1 metadata matches
  the requested tuple.

## Materialized Product Sweep

Command:

```bash
PYTHONPATH=src .venv/bin/python tools/build_d1_overlay_policy_candidates.py \
  --d1-bin experiments/results/d1_fused_runtime_smoke_20260515T165614Z_codex/d1_polytope.bin \
  --a1-bin experiments/results/d1_fused_runtime_smoke_20260515T165614Z_codex/a1.bin \
  --output-dir experiments/results/d1_overlay_policy_product_candidates_20260515_codex \
  --policies rgb,green,neg_green,rb_pos_g_neg \
  --amplitude-scales 0.5,1.0 \
  --sign-policies payload,negate_payload,alternating_pairs
```

Summary:

- candidate count: `24`
- output manifest:
  `experiments/results/d1_overlay_policy_product_candidates_20260515_codex/d1_overlay_policy_candidates_manifest.json`
- channel policies: `rgb`, `green`, `neg_green`, `rb_pos_g_neg`
- amplitude scales: `0.5`, `1.0`
- sign policies: `payload`, `negate_payload`, `alternating_pairs`
- smallest archive:
  `d1_overlay_channel_green_amp_0p5_sign_payload`, `3993` bytes,
  SHA-256 `d3a4619e96c90688e38fefe4eadd792ff1d3b5c9d4f94c9c14efeb6a77e09ff3`
- archive byte range: `3993` to `4014`

## Authority Boundary

All D1 policy-product candidates keep:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Promotion still requires the same archive/runtime bytes to receive paired exact
`[contest-CUDA]` and `[contest-CPU]` auth eval plus result review.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests \
  src/tac/tests/test_d1_segnet_margin_polytope_dispatch_config.py
# 119 passed

.venv/bin/ruff check \
  tools/build_d1_overlay_policy_candidates.py \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  experiments/train_substrate_d1_segnet_margin_polytope.py \
  src/tac/substrates/d1_segnet_margin_polytope/overlay.py \
  src/tac/substrates/d1_segnet_margin_polytope/inflate.py \
  src/tac/substrates/d1_segnet_margin_polytope/__init__.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_overlay_and_shrink.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py
# All checks passed
```
