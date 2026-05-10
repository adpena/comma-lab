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
- `src/tac/optimizer/exact_ready_audit.py` now also recomputes live archive
  SHA-256 and runtime-tree SHA-256 for ready rows that declare
  `archive_path`/`candidate_archive_path` and `submission_dir`. A ready row is
  blocked when its persisted metadata no longer matches the bytes that would
  actually dispatch.
- Recursive hardening pass after fresh-eyes review: exact-ready audit now
  fails closed when a ready row lacks a live archive path, validates archive
  byte fields, strict ZIP status, `inflate.sh` executable bit, `report.txt`,
  archive manifest SHA/size/member closure, and declared `inflate_sh_path`.
  Terminal suppression now also recognizes older `completed_contest_cuda`
  statuses with score evidence, and runtime SHA disambiguation no longer
  depends solely on the fragile `score_affecting_runtime_changed` boolean when
  both candidate and terminal runtime SHAs are present.
- `tools/operator_briefing.py --json --skip-pareto` now still emits
  `exact_ready_queue_audit`; queue hygiene is an operator/dispatch invariant,
  not a Pareto-table-only subsection.
- Follow-up cleanup pass: `tools/audit_exact_ready_queues.py` can now write
  and apply a durable suppression/retraction manifest. Raw queue JSON is left
  untouched; the manifest records why each stale row is not dispatchable and
  reports residual unresolved stale rows separately from classified rows.
- Canonicalization follow-up: suppression/retraction logic lives in
  `tac.optimizer.exact_ready_audit` (`build_suppression_manifest`,
  `load_suppression_manifest`, `apply_suppression_manifest`); the CLI and
  `tools/operator_briefing.py` are thin callers.
- `tools/operator_briefing.py --json --skip-pareto` now applies the manifest
  automatically and reports `raw_stale_ready_row_count=5`,
  `suppressed_ready_row_count=5`, `stale_ready_row_count=0`.
- Durable manifest:
  `.omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json`.

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

## Current cleanup result

Command:

```bash
.venv/bin/python tools/audit_exact_ready_queues.py \
  --format json \
  --write-suppression-manifest .omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json
```

Result:

- `queue_count=5`
- `raw_stale_ready_row_count=5`
- `suppressed_ready_row_count=5`
- `stale_ready_row_count=0`
- `passed=true`

Suppressed/retracted exact-ready rows:

- `pr101_proxy_sweep/exact_eval_promotion/exact_ready_queue.json`
  - lane `pr101_kaggle_proxy_runtime_packet_exact_eval`
  - archive `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
  - declared runtime `748b99d3bb63372eeff8f784cd0b9589a3b05c781e3517cd426810dfe48a5382`
  - live runtime `84afb14b741a7250046e6956b00710be02615b8cc500551f77576546245dfaf2`
  - classification: `retracted_stale_live_runtime_metadata`
  - operator action: `retract_stale_exact_ready_row`
  - blocker:
    `ready_row_runtime_tree_sha_mismatch:84afb14b741a7250046e6956b00710be02615b8cc500551f77576546245dfaf2!=748b99d3bb63372eeff8f784cd0b9589a3b05c781e3517cd426810dfe48a5382`
- `pr101_bias_refine_exact_ready_queue.json`
  - lane `pr101_kaggle_proxy_runtime_packet_exact_eval`
  - archive `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
  - runtime `53e51052c71f97354ee1a48abddd006d98e87c795e714a0d5d4f8e0b5dc75a88`
  - classification: `retired_by_terminal_score_not_below_active_floor`
  - terminal score `0.22650343150032118 >= 0.2089810755823297`
- `pr101_kaggle_proxy_exact_ready_queue.json`
  - lane `pr101_kaggle_proxy_runtime_packet_exact_eval`
  - archive `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
  - runtime `84afb14b741a7250046e6956b00710be02615b8cc500551f77576546245dfaf2`
  - classification: `retired_by_terminal_exact_cuda_negative`
  - terminal exact-CUDA negative: `0.22688160652506983`
- `pr103_hidden_gem_exact_ready_queue.json`
  - lane `pr103_ac_hidden_gem`
  - archive `8274e88c0ab1d26a06470a0730d17fe004556afa564460cf1c05624ff6060278`
  - classification: `retired_by_terminal_exact_cuda_negative`
  - terminal exact-CUDA negative: `41.34951661322894`
- `pr106_q10_exact_ready_queue.json`
  - lane `pr106_q10_151byte_brotli`
  - archive `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
  - classification: `retired_by_terminal_exact_cuda_negative`
  - terminal exact-CUDA negative for frontier promotion: `0.20936498680571203`

The PR101 runtime-patch row that previously stayed non-stale is now also
retired by the later Modal T4 terminal negative row:
`pr101_kaggle_proxy_runtime_packet_exact_cuda_modal_20260510T194142Z`.

Live-custody facts now confirm the five stale rows have valid archive bytes,
strict ZIPs, executable `inflate.sh`, present reports, archive manifests, and
runtime manifests; they are stale because of terminal evidence or runtime
metadata mismatch, not because the audit is unable to inspect them. The raw
queue artifacts are preserved, but the manifest sets `dispatch_allowed=false`
for each classified identity.

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
anchor. Same-archive PR101 runtime-patch rows require fresh live runtime-tree
custody and their own exact CUDA dispatch before any promotion; stale metadata
must not enter the dispatch queue. The next queue should move to:

1. harvest/classify active T1 Ballé Modal run;
2. grammar-preserving HNeRV custom-codec packet compiler work with raw/tensor
   equivalence proof;
3. CUDA-calibrated PR101/A1 searches only when runtime-consumption proof and
   drift posterior are present.

## Unified solver wire-in

`research_only=true` for this cleanup/classification landing. It does not add a
new codec, sensitivity map, Pareto constraint, bit allocator, or dispatchable
candidate. Its solver-stack effect is negative custody: suppress/retract stale
exact-ready identities so future dispatch tools and operators move to new
charged-byte/runtime evidence instead of duplicate GPU spend.
