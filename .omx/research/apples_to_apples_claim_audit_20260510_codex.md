# Apples-to-apples claim audit (2026-05-10)

`score_claim=false`; `dispatch_attempted=false`; `remote_gpu_run=false`.

## Why this exists

The PR103 arithmetic retarget pass initially risked a false conclusion:
reconstructing PR103's source q8 histogram produced a no-op, but that does not
test whether nearby q8 histogram coordinates can improve byte count. A corrected
coordinate probe found small byte-positive moves. This memo records the
resulting audit rule for current and recent claims:

> A result may only falsify, exhaust, or optimize the exact axis it measured:
> same archive/runtime identity, same scorer/eval substrate, same proxy or exact
> objective, same byte grammar, same candidate family, and same promotion gate.

Anything else is a trust-region update, not a lane kill.

## Audit classes to preserve

| Claim family | Safe conclusion | Unsafe conclusion to avoid | Current action |
|---|---|---|---|
| PR103 arithmetic retarget | Baseline source-rule histogram reconstruction is a no-op. Small q8 coordinate changes can save `1-2` local bytes in histogram Brotli only. | "PR103 arithmetic retarget is exhausted" or "model-gap estimates are false." | Keep PR103 AC work open; require runtime adapter plus multi-coordinate/multi-stream search before archive materialization. |
| A1/PR101 bias sweeps | Measured A1/PR101 bias-runtime identities regressed or failed to beat their exact anchors on the measured CPU/CUDA axis. | "A1 is globally optimal" or "inflate-time arithmetic has no remaining value." | Limit to the tested coordinate grid and exact archive/runtime identities; reopen only with new byte/runtime identity and proof. |
| A1 CPU medal-band | A1 is strong on `[contest-CPU GHA Linux x86_64]`; paired Modal T4 CUDA is materially worse. | "CPU medal-band implies submission readiness" or "rounds to gold" without CUDA. | Keep CPU and CUDA axes separate; submission requires exact CUDA custody and policy approval. |
| A5 scalar/q-bit schedules | Current q7/qsum schedules are exact-CUDA negatives. | "A5 is dead" or "frame-conditional/channel allocation is exhausted." | Reactivate only through score-domain channel allocation, training-time q-bit noise, or runtime-consumed packet deltas. |
| Track 4 UNIWARD/STC/Hessian | v1/v2 measured configs are negative on their measured axes. | "Hessian/UNIWARD family is killed." | Keep deferred with stricter reactivation criteria; do not dispatch old identities. |
| PR103 raw hidden-gem deletion | Raw range-stream deletion collapsed on exact CUDA and is non-grammar-preserving. | "PR103 arithmetic custom-codec work is dead." | Keep grammar-aware AC transforms open; reject raw deletion without symbol-roundtrip/runtime proof. |
| MPS/Kaggle/macOS sweeps | Useful for ranking, curves, and configuration discovery. | Treat as auth eval or lane status evidence. | Promote only from byte-closed packet custody plus `[contest-CPU]` or `[contest-CUDA]` as labeled. |

## New rule for future ledgers

Any memo using `falsified`, `exhausted`, `optimal`, `local optimum`, `do not
spend`, or `submission ready` must include an apples-to-apples scope line:

```text
Scope: <candidate family> / <archive SHA or packet identity> /
<runtime SHA or runtime class> / <evidence axis> / <objective> /
<what remains untested>
```

Examples:

- Good: "Measured config `archive_sha=...`, runtime tree `...`, `[contest-CUDA
  T4]`, exact evaluator, 600 samples: negative. Untested: score-domain
  training-time q-bit noise."
- Bad: "A5 exhausted" after one q-bit exact negative.
- Good: "Baseline q8 histogram reconstruction no-ops. Untested: q8 coordinate
  search, multi-stream search, alternate range model, runtime-adapter overhead."
- Bad: "PR103 arithmetic retarget falsified."

## Immediate correction

The PR103 arithmetic plan was corrected in
`.omx/research/pr103_arithmetic_transform_plan_20260510_codex.md`:

- baseline retarget probes are classified as source-rule reconstruction;
- coordinate-search probes across the top-5 streams found local `-1` to `-2`
  byte moves;
- all PR103 AC artifacts remain `score_claim=false` and
  `ready_for_exact_eval_dispatch=false`.

## Verification

```bash
rg -n "FALSIFIED|falsified|exhausted|optimal|local optimum|do not spend|no-op|contest-CPU|contest-CUDA|macOS|MPS" \
  .omx/research/current_score_lowering_roadmap_20260510_codex.md \
  .omx/research/*20260510*.md .omx/research/*20260509*.md

.venv/bin/python -m pytest \
  src/tac/tests/test_hnerv_pr103_lc_ac_schema.py \
  src/tac/tests/test_pr103_arithmetic_transform_plan.py \
  tests/test_plan_pr103_arithmetic_transform_cli.py -q
# 17 passed
```
