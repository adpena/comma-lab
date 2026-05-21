# Codex Findings: Subagent Review Triage

**Date (UTC):** 2026-05-21T02:10:05Z
**Head verified:** `06b69b8ed3386aa6a8420758057f33cacd6339b9`
**Scope:** inherited subagent findings for PR101/PR106 sidecar grammar and LL scorer-response authority fields.

## Verdict

The inherited findings were reviewed against current `origin/main` and are stale
or already covered by newer code. No patch is required on head `06b69b8ed`.

## Finding 1: PR106 stale framing metadata after PR101 exact-radix width change

**Subagent claim:** PR106 packet paths fail after PR101 `encode_ranked_no_op_sidecar`
started using exact-radix dimension bytes and actual no-op rank width.

**Current-head verification:** not reproduced.

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py
```

Result: `34 passed in 0.44s`.

The full paired grammar regression also passes:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_packet_compiler_pr101_sidecar_grammar.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_scorer_response_dataset.py
```

Result: `101 passed in 0.95s`.

## Finding 2: LL scorer-response authority laundering

**Subagent claim:** pair #4 seed-boundary evidence can omit or contradict
`score_claim=false` and still be normalized into a non-promotional dataset.

**Current-head verification:** already fixed on current main.

`src/tac/optimization/scorer_response_dataset.py` requires every listed
authority field to be present and exactly `False` unless a caller explicitly
opts into legacy missing-field normalization. Pair #4 boundary normalization
requires:

- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `rank_or_kill_eligible=false`
- `promotable=false`

`tools/run_magic_codec_pair_4_procedural_seed_orthogonality_smoke.py` also
emits `score_claim=false` directly, not only `score_claim_valid=false`.

Focused proof:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py
```

Result: `15 passed in 0.17s`.

## Working-Tree Note

The unrelated untracked file
`.omx/research/hf_jobs_billing_unblock_523_hinton_surrogate_20260520.md` was
left untouched.
