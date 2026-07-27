# Codex TIER-0 session anchor — 2026-07-27 17:49 UTC

## Pointer-delta honesty

- Canonical frontier target: **0.172**.
- Best local custody row: **0.1880443979880752** on contest-CPU; it is not a
  fresh same-object G111 submission candidate.
- Pointer movement this session: **0.000**. No archive was promoted and no
  scorer result was inferred from a proxy.

## Completed and closed; do not rediscover

- G110 public receiver/archive closure, clean public runtime, coder
  arbitration, and atomic release materialization are completed.
- G117 parsed selector, G119 pose refit/hardening, G120 production authority
  and governed dry-run gate, and G121 exhaustive/live harvest handoff are
  completed.
- `pact-g111-sparse-seed-root-state-custody-20260727` is **completed/green**:
  implementation commit `e1bfb97d06`, task-closure commit `2b06dc666c`.
- The completed G111 child now covers sparse protected-seed and independent
  optimizer custody, additive legacy-v2 behavior, schema-aware primary
  optimizer-family checks, atomic armed-empty/observed Polyak semantics,
  exact streaming island detection, nonfinite/empty-support refusal, and
  refusal of false v3 complete-trajectory claims.
- Validation: 80 focused island/Polyak/sparse/fresh tests passed with one
  unrelated stale-DSL test deselected; 63 checkpoint tests passed with four
  documented governed-admission fixtures deselected; Ruff F821/F823,
  `py_compile`, JSON parse, diff check, and independent P0 review passed.

## Still open; do not overclaim

- `pact-g111-complete-trainable-state-resume-20260727` remains
  **in_progress**.
- Exact blocker: implement the real 14-domain checkpoint writer/restore
  adapters and component-specific required-key plus reverse-coverage proof
  for every active trajectory domain. The current skeleton deliberately
  cannot certify completeness.
- `pact-g111-current-typed-clean-dry-start` remains **blocked/red** on that
  parent. Historical v6 and v7 are preserved failure evidence and cannot be
  reused to clear a fresh v8 gate.
- `pact-g111-first-real-n600-capstone-run` remains **blocked** on the full
  trajectory proof and fresh v8 crash-resume dry-start.

## Exact next executable edge

1. Implement and test the 14-domain writer/restore adapters with reverse
   coverage; mark only bounded children completed as they close.
2. When and only when the parent is complete, run a fresh v8 full-n600
   uninterrupted/resumed dry-start on the SSD tier.
3. Run the real n600 producer, G121 harvest, G119 pose refit, G110 atomic
   materialization and double decode.
4. Evaluate the exact archive bytes on upstream contest-CPU and contest-CUDA;
   promote only a receiver-closed score below 0.172.

## Closeness verdict

The post-producer path is structurally connected, so the remaining critical
path is narrow. Operationally this is **not one evaluator call away**: one
state-custody implementation phase, one fresh physical resume proof, and one
real n600 build/eval phase remain. No current completed epoch provides an
honest wall-clock ETA.

The old G54 low-distortion envelope remains research-only but useful:
preserved distortion would require fewer than 187,563 archive bytes to cross
0.172. It is a byte budget and acquisition signal, not a candidate or a
probability of success.

## Triality

- DSL: governed launch and current typed-hash custody remain mandatory.
- DAG: full trajectory proof -> fresh v8 -> real n600 compile -> harvest/refit
  -> atomic archive -> double decode -> exact CPU/CUDA scorer.
- Equations: minimize
  `100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`, with no
  independent arbitrary segment/pose/rate thresholds.

## STORES CONSULTED

- `reports/latest.md`
- `.omx/state/canonical_frontier_pointer.json`
- `.omx/state/canonical_task_status.jsonl`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- G111 physical v6/v7 receipts on `/Volumes/VertigoDataTier/pact`
- G54 low-distortion rate-budget receipt
