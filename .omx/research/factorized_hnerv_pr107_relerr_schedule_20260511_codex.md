# Factorized HNeRV PR107 rel-error schedule (2026-05-11)

## Scope

Operator goal: keep score-lowering pressure focused on exact, composable work
without repeating the HNeRV parity mistake of building research artifacts that
cannot become scored packets.

This pass added a CPU-only rank/error/byte planner:

- `tools/plan_factorized_hnerv_relerr_schedule.py`
- `src/tac/tests/test_plan_factorized_hnerv_relerr_schedule.py`

The tool measures actual post-quantization SVD reconstruction error and
isolated per-tensor brotli savings, then emits a
`tools/build_factorized_hnerv_archive.py --plan-config` only when the schedule
has positive byte savings under the configured rel-error cap. It never marks a
row exact-eval dispatch-ready.

## Real PR107 substrate result

Command:

```bash
.venv/bin/python tools/plan_factorized_hnerv_relerr_schedule.py \
  --substrate-archive experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/archive.zip \
  --json-out .omx/research/artifacts/factorized_hnerv_pr107_relerr_schedule_20260511_codex/plan.json \
  --markdown-out .omx/research/artifacts/factorized_hnerv_pr107_relerr_schedule_20260511_codex/plan.md \
  --print-markdown
```

Result:

- score claim: `false`
- exact-eval dispatch-ready: `false`
- rel-error caps checked: `0.02`, `0.04`, `0.06`, `0.08`, `0.10`, `0.15`
- selected tensors: none at every cap
- isolated byte savings: `0` at every cap

## Adversarial classification

This falsifies the immediate post-hoc-SVD shortcut on the PR107 substrate. It
does **not** kill factorized HNeRV as a family.

The correct interpretation is:

- The packet/runtime/export surface exists.
- The current public PR107 weights do not expose a rel-error-capped positive
  isolated brotli schedule under the tested caps.
- Exact CUDA dispatch would be arbitrary until a factorized schedule has both
  positive charged-byte movement and a score-domain trust reason.

Verdict:
**DEFERRED-pending-low-rank-trained-substrate-or-score-domain-QAT**.

## Reactivation criteria

Reopen this lane when one of these is true:

1. A low-rank-structured HNeRV/Ballé substrate is trained with the factorized
   packet grammar in the loop.
2. QAT or distillation explicitly penalizes factor rank while preserving
   score-domain SegNet/PoseNet losses and differentiable eval-roundtrip.
3. A non-SVD factor family, grouped/depthwise rewrite, tensor-train, Tucker,
   LoRA-style delta, or learned hyper-decoder produces positive charged-byte
   savings with a rel-error cap and runtime-consumed bytes.
4. Exact CPU/CUDA paired eval with raw-output custody proves a higher-relerr
   schedule is score-safe despite tensor RMS risk.

## Planner correction

`tools/plan_sub017_cpu_frontier.py` now treats the active blocker as
`posthoc_factorized_hnerv_relerr_safe_schedule_missing` instead of the stale
`factorized_hnerv_runtime_not_implemented`. The runtime/compiler exists; the
measured schedule does not yet justify score dispatch.
