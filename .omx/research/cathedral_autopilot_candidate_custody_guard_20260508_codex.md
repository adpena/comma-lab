# Cathedral Autopilot Candidate Custody Guard - 2026-05-08

## Scope

Task: adversarially review score-lowering candidate tooling and implement one
focused, local, OSS/composable improvement. No GPU jobs were dispatched.

Owned surfaces touched:

- `tools/cathedral_autopilot.py`
- `src/tac/optimization/candidate_evidence_contract.py`
- related tests under `src/tac/tests/`

## Adversarial Finding

`tools/cathedral_autopilot.py` documented that missing custody fields must fail
closed, but `_is_explicitly_promotable_evidence()` could accept an
exact-CUDA-looking row using only booleans plus a marker such as
`[contest-CUDA]`. A malformed or prematurely harvested row could therefore
become a promotable empirical anchor without archive SHA, runtime-tree SHA,
positive archive bytes, or a CUDA score.

That is a planner bug, not an eval result. It can move ranker state before the
candidate has enough custody to support exact-eval implications.

## Implemented Improvement

Added a pure reusable contract helper:

`src/tac/optimization/candidate_evidence_contract.py`

The helper emits machine-readable blockers and requires all of the following
before a row can be active promotable exact-CUDA evidence:

- explicit `score_claim`, `promotion_eligible`, `rank_or_kill_eligible`, and
  `ready_for_exact_eval_dispatch` booleans;
- exact-CUDA evidence marker;
- no proxy/planning/research-signal marker;
- no negative/retired/deferred/retracted verdict marker;
- positive archive bytes;
- valid 64-hex archive SHA-256;
- valid 64-hex runtime-tree SHA-256;
- CUDA score field;
- no source dispatch blockers.

`tools/cathedral_autopilot.py` now consumes the helper. Non-promotable
empirical anchors still update planning visibility and validation queues, but
they are blocked from active ranking/promotion and carry the specific custody
blockers.

## Local Verification

Command:

```bash
.venv/bin/python -m pytest src/tac/tests/test_candidate_evidence_contract.py src/tac/tests/test_cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot_proxy_guards.py
```

Result: 39 passed.

## Next Exact-Eval Implication

The next exact-eval harvester or candidate builder that wants cathedral
autopilot promotion must stamp `archive_sha256`, `runtime_tree_sha256`,
`empirical_archive_bytes`, and `score_contest_cuda` on the evidence row. A row
without those fields can still appear in the validation queue as a high-upside
planning signal, but it will not rank as a score-lowering empirical anchor.
