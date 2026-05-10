# Exact-ready terminal-evidence audit (2026-05-10)

## Scope

This ledger records the stale exact-ready queue class found after the Modal
exact-CUDA tranche for PR101 bias-refine, PR103 hidden-gem, and PR106 q10.
The score-lowering goal is to prevent duplicate exact-eval spend and force the
queue to advance to new byte-changing or train-time artifacts after terminal
evidence lands.

## New durable surfaces

- `src/tac/optimizer/exact_ready_audit.py`
- `tools/audit_exact_ready_queues.py`
- `src/tac/tests/test_optimizer_exact_ready_audit.py`
- `tests/test_audit_exact_ready_queues_cli.py`
- `tools/operator_briefing.py` now suppresses exact-eval packet rows with
  same-lane/same-archive terminal evidence from
  `.omx/state/active_lane_dispatch_claims.md`.
- `tools/operator_briefing.py` now exposes the exact-ready queue audit in both
  human and JSON output so stale rows are visible from the normal operator
  flow.

The audit reuses `tac.optimizer.exact_readiness.terminal_claim_result_conflicts`
so stale persisted queues and newly generated promotions share the same
mathematical predicate:

```text
stale ⇔ ready_for_exact_eval_dispatch
      ∧ same lane_id
      ∧ same archive_sha256
      ∧ (same runtime_tree_sha256 when score_affecting_runtime_changed=true)
      ∧ terminal exact-CUDA evidence is negative
        or terminal score is not below active floor
        or terminal score already beat the active floor
```

Infrastructure failures such as missing runtime dependencies remain rerunnable.
Same-archive candidates with a different score-affecting runtime tree remain
eligible for exact eval; they are not suppressed by archive SHA alone.

## Current audit result

Command:

```bash
.venv/bin/python tools/audit_exact_ready_queues.py \
  --format json --warn-only \
  --output experiments/results/optimizer_candidate_queue_20260510_codex/exact_ready_terminal_evidence_audit_20260510_codex.json
```

Result:

- `queue_count=5`
- `stale_ready_row_count=3`
- `passed=false`

Stale exact-ready rows:

- `pr101_bias_refine_exact_ready_queue.json`
  - lane `pr101_kaggle_proxy_runtime_packet_exact_eval`
  - archive `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
  - runtime `53e51052c71f97354ee1a48abddd006d98e87c795e714a0d5d4f8e0b5dc75a88`
  - terminal score `0.22650343150032118 >= 0.2089810755823297`
- `pr103_hidden_gem_exact_ready_queue.json`
  - lane `pr103_ac_hidden_gem`
  - archive `8274e88c0ab1d26a06470a0730d17fe004556afa564460cf1c05624ff6060278`
  - terminal exact-CUDA negative: `41.34951661322894`
- `pr106_q10_exact_ready_queue.json`
  - lane `pr106_q10_151byte_brotli`
  - archive `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
  - terminal exact-CUDA negative for frontier promotion: `0.20936498680571203`

Non-stale same-archive rows:

- `pr101_proxy_sweep/exact_eval_promotion/exact_ready_queue.json`
- `pr101_kaggle_proxy_exact_ready_queue.json`

Those rows share the PR101 archive SHA but have different score-affecting
runtime trees than the terminal PR101 bias-refine exact-CUDA run. They remain
eligible only as runtime-patch candidates with fresh dispatch claims; their
proxy score is not a score claim.

## Operator-briefing effect

Command:

```bash
.venv/bin/python tools/operator_briefing.py --json --top 3
```

The PR106 q10 row now contains:

```json
{
  "ready_for_submit": false,
  "terminal_exact_eval_evidence_blockers": [
    "same_lane_terminal_negative_for_same_archive:pr106_q10_151byte_brotli:pr106_q10_exact_cuda_modal_20260510T173900Z:completed_contest_cuda_auth_eval_negative"
  ]
}
```

## Score-lowering implication

Do not redispatch PR101 bias-refine, PR103 hidden-gem, or PR106 q10 for the
same archive and runtime identity. PR106 q10 remains a useful
grammar-preserving recode anchor; PR103 hidden-gem remains a catastrophic
range-stream perturbation negative; PR101 bias-refine remains a CUDA-drift
anchor. Same-archive PR101 runtime-patch rows require fresh runtime-tree
custody and their own exact CUDA dispatch before any promotion. The next queue
should move to:

1. harvest/classify active T1 Ballé Modal run;
2. grammar-preserving HNeRV custom-codec packet compiler work with raw/tensor
   equivalence proof;
3. CUDA-calibrated PR101/A1 searches only when runtime-consumption proof and
   drift posterior are present.
