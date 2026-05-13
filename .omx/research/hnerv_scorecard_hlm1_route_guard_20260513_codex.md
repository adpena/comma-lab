# HNeRV Scorecard HLM1 Route Guard (2026-05-13)

## Summary

The HNeRV frontier scorecard was refreshed to include the exact CUDA
HDM4+HLM1 result:

- label: `PR106-R2-HDM4-HLM1`
- axis: `[contest-CUDA]`
- score: `0.20638030907530963`
- archive bytes: `186423`
- archive SHA-256:
  `8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0`
- eval artifact:
  `experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.adjudicated.json`

This fixes a no-signal-loss bug class: all-lanes preflight previously passed
while the generated scorecard still routed internal score-lowering work through
HDM4 at `0.20642625334307507`, hiding the lower HLM1 exact CUDA artifact.

## Guardrail

`tools/audit_hnerv_frontier_scorecard.py` now accepts
`--required-eval LABEL=PATH`. The audit fails closed if a required exact CUDA
eval is missing from scorecard rows, has a mismatched archive/score/artifact
identity, is not full-sample T4 CUDA evidence, or is lower than the declared
internal score-lowering frontier but not selected.

`tools/all_lanes_preflight.py` threads the known HLM1 exact CUDA eval through
that required-eval check. Gate #6 now reports:

```text
hnerv frontier scorecard: PASS (12 rows, 2 payload groups, 36 follow-up targets, internal score-lowering=PR106-R2-HDM4-HLM1 (0.20638030907530963))
```

## Canonicality Boundary

The HLM1 row is intentionally an internal score-lowering frontier, not public
promotion authority. The eval artifact still carries explicit
`promotion_blockers`, so `experiments/build_hnerv_frontier_scorecard.py` now
treats a nonempty `promotion_blockers` list as a canonical-frontier blocker.
This preserves the public/canonical frontier while still allowing exact-CUDA
optimizer routing through the lower HLM1 artifact.

## Refreshed Targets

- Canonical/public frontier remains `PR103-ac-repack` at
  `0.20898105277982337` because HLM1 still has promotion blockers.
- Internal score-lowering frontier is `PR106-R2-HDM4-HLM1` at
  `0.20638030907530963`.
- Next internal byte target remains the `inner_decoder_packed_brotli` section:
  `169990` bytes, SHA-256
  `76a1156369b6f3a54c011261137684ec1b4f70331e2d4335dea8761e5d28aa06`.
- HLM1 latent section is now represented separately in the scorecard profile:
  `15780` bytes, SHA-256
  `c5a6bfc157fcef474a21a3bbbb687e2bea2e99ad3d6195f78563a397e8f21ee4`.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py \
  src/tac/tests/test_build_hnerv_frontier_scorecard.py \
  src/tac/tests/test_check_193_substrate_auth_eval_claim_boundary.py \
  src/tac/tests/test_trainer_skeleton.py \
  src/tac/substrates/siren/tests/test_siren_roundtrip.py \
  src/tac/tests/test_auth_eval_result.py \
  src/tac/tests/test_check_182_substrate_recipe_target_modes.py
```

Result: `97 passed in 1.63s`.

```bash
.venv/bin/python tools/audit_hnerv_frontier_scorecard.py \
  --scorecard experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/scorecard.json \
  --required-eval PR106-R2-HDM4-HLM1=experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.adjudicated.json
```

Result: `PASS`, internal score-lowering frontier
`PR106-R2-HDM4-HLM1 (0.20638030907530963)`.

```bash
.venv/bin/python tools/all_lanes_preflight.py --timings --timeout-s 30
```

Result: `ALL 31 PREFLIGHT CHECKS PASSED`, wall `2.34s`.

## Next Score-Lowering Work

Do not spend another exact eval on this unchanged HLM1 archive. The next useful
work is byte-different and score-affecting:

1. Semantic transform of the HLM1/HDM4 `inner_decoder_packed_brotli` section.
2. HNeRV parity training and PR95/PR100/PR101 eval-roundtrip discipline.
3. SIREN/Balle/CompressAI substrate trainers only after real trainer, recipe,
   archive grammar, and exact-eval path exist.
4. PacketIR compiler passes that emit charged archives and preserve exact CUDA
   score authority.
