# L5 v2 anchor semantics hardening (2026-05-16)

## Scope

- `src/tac/optimization/l5_staircase_v2.py`
- `src/tac/tests/test_l5_staircase_v2.py`

## Finding

The L5 v2 `exact_anchor_or_diagnostic_pair` gate validated paired CPU/CUDA
identity and component-delta fields, but `anchor_type` alone controlled whether
a row was treated as `exact` or `diagnostic`. A row could therefore present an
`anchor_type=exact` or `anchor_type=diagnostic` label without explicit
no-score-claim semantics, without evidence-grade semantics, and without a
diagnostic reason.

For L5-v2 architecture lock-in this is too weak: exact anchors, diagnostic
anchors, and proxy/advisory rows must remain distinguishable in the artifact
itself, not only in surrounding prose.

## Change

When the gate requires anchor semantics:

- every anchor row must carry `score_claim=false`;
- exact rows must carry an evidence grade containing `contest`;
- diagnostic rows must carry an evidence grade containing `diagnostic`;
- diagnostic rows must also provide a non-empty `diagnostic_reason`.

Regression coverage now mutates a valid anchor artifact so CUDA tries to claim
score from a proxy evidence grade and CPU switches to diagnostic without a
reason; the gate refuses both.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_autopilot_dispatch_ranking.py -q
.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py
.venv/bin/python -m py_compile src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py
```

Results:

- `26 passed`
- `58 passed`
- `ruff`: clean
- `py_compile`: clean

## Evidence Semantics

No score claim, no promotion, and no dispatch authorization. This hardens the
L5-v2 gate so anchor rows cannot masquerade as exact or diagnostic evidence by
label alone.
