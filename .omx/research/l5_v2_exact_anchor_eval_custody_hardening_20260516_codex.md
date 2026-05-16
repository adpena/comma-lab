# L5-v2 exact-anchor eval custody hardening (2026-05-16)

## Problem

The L5-v2 `exact_anchor_or_diagnostic_pair` gate checked paired axis identity,
devices, component deltas, anchor type, `score_claim=false`, and evidence-grade
semantics. It did not call the shared exact-eval custody validator for exact
anchor rows, so a malformed exact row could omit contest sample count, score
formula closure, archive byte count, hardware, auth-eval command, or durable
log/artifact files while still satisfying the gate's local subset.

## Fix

For `anchor_type="exact"` rows, the gate now calls
`validate_exact_eval_evidence()` with:

- expected axis (`contest_cpu` or `contest_cuda`)
- shared archive SHA and runtime tree SHA
- required artifact path, hardware, auth-eval command, log path, and devices
- explicit `repo_root` artifact base directory

Diagnostic anchors remain diagnostic-only: they still require diagnostic
evidence grade and reason text, but they are not promoted into score authority.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q
```

Result: `27 passed`.

## Authority

This is a dispatch/promotion guardrail. It does not create a score claim. It
prevents L5-v2 exact-anchor gate satisfaction unless the row has contest formula
closure and durable CPU/CUDA artifact custody.
