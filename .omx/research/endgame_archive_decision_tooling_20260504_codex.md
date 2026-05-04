# Endgame Archive Decision Tooling - 2026-05-04

Scope: local-only, byte-level decision support for PR85/STBM/PR91/PR92-family
archives. No remote GPU dispatch, no training, no scorer load, no upstream
scorer edits, and no score claim were performed.

## Artifact

New deterministic tool:

- Module: `src/tac/endgame_archive_decision.py`
- CLI: `experiments/profile_endgame_archive_decision.py`
- Focused tests: `src/tac/tests/test_endgame_archive_decision.py`
- Output JSON:
  `experiments/results/endgame_archive_decision_20260504_codex/endgame_archive_decision_profile.json`
- Output Markdown:
  `experiments/results/endgame_archive_decision_20260504_codex/endgame_archive_decision_profile.md`

Command:

```text
.venv/bin/python experiments/profile_endgame_archive_decision.py \
  --candidate PR85=experiments/results/public_pr85_intake_20260503_codex/archive.zip \
  --candidate PR85_STBM1BR=experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip \
  --candidate PR91_HPM1=experiments/results/public_pr91_intake_20260504_codex/archive.zip \
  --candidate PR92_RSB1=experiments/results/public_pr92_intake_20260504_codex/archive.zip \
  --frontier-label PR85_STBM1BR \
  --json-out experiments/results/endgame_archive_decision_20260504_codex/endgame_archive_decision_profile.json \
  --markdown-out experiments/results/endgame_archive_decision_20260504_codex/endgame_archive_decision_profile.md
```

Evidence grade: `byte_level_decision_support_only`. The report is not score
evidence. Exact score truth remains the contest CUDA path:
`archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Trajectory Forecast

Current local exact frontier remains PR85 + STBM1BR lossless mask recode:

- Archive bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Exact T4 score already recorded elsewhere:
  `0.25369011029397787`

Public PR85/PR92 anatomy has converged on the same PR85-family compact `x`
bundle: QMA9 mask, QH0 model, P1D1 pose, small Brotli side streams, and a
randmulti tail. PR92 adds a 386 byte `RSB1` side-action member and replaces the
legacy randmulti tail with `RMB1`. That direction is anatomically useful, but
it does not beat the STBM mask-recode frontier by rate alone.

PR91/HPM1 is still the largest known byte signal in this family. The profiler
confirms its HPM1 mask contract parses locally:

- HPM1 mask bytes: `145087`
- HPM1 token bytes: `116796`
- HPM1 HPAC bytes: `28243`

Versus the STBM frontier, PR91's mask replacement is `-7352` archive bytes if
runtime/parity gates are solved. That is the highest local rate opportunity in
this four-archive slice, but it remains blocked by the HPM1 entropy/probability
contract documented in the PR91 parity ledger.

## Tool Findings

All four profiled archives passed strict local ZIP custody checks and cheap
segment validation. The tool records central/local name consistency, duplicate
member blockers, segment entropy, SHA-256, codec labels, and contract probes.

Key deltas against `PR85_STBM1BR`:

| Candidate | Archive delta | Primary delta | Side-info delta | ZIP overhead delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `PR85` | `+6572` | `+6572` | `0` | `0` | old QMA9 mask is strictly worse by rate |
| `PR91_HPM1` | `-7352` | `-7352` | `0` | `0` | byte-positive, blocked by mask runtime/parity |
| `PR92_RSB1` | `+6760` | `+6296` | `+386` | `+78` | worse than STBM frontier |

PR92 detailed side/tail accounting:

- `RMB1` randmulti segment bytes: `15825`
- Legacy PR85/STBM randmulti segment bytes: `16101`
- Tail segment saving: `-276`
- `RSB1` side-action member bytes: `386`
- Additional ZIP overhead from the extra member: `78`
- Net `randmulti+side_info` transplant estimate onto STBM:
  `+188` bytes

PR92 `RMB1` validation:

- Mask Brotli bytes: `4478`
- Values Brotli bytes: `11341`
- Decoded mask bytes: `6225`
- Row count: `83`
- Selected value count: `13496`

PR92 `RSB1` validation:

- Action count: `600`
- Table id: `1`
- Raw action bytes: `600`
- Unique actions: `40`

Interpretation: PR92's side-action/tail idea is not a rate-only stack candidate
on top of STBM. It needs component improvement evidence or a new side-action
encoding that beats the `+188` byte overhead before any exact-eval dispatch
plan is worth writing.

## Top 3 Endgame Actions

1. Main orchestrator should keep exact-eval capacity off PR92-on-STBM
   `RMB1+RSB1` transplants unless a new artifact first shows component benefit
   or reduces the side-action overhead below break-even. The current rate-only
   estimate is `+188` bytes, so dispatching it for score would be a late-contest
   stall.

2. Continue PR91/HPM1 parity recovery as the main rate-upside lane. The local
   byte contract is clean and the mask replacement is `-7352` bytes versus the
   STBM frontier, but promotion requires runtime parity and then a claimed
   exact CUDA auth eval. Slow HPM1 prefix profiling should not consume the
   critical path unless it directly repairs the HPAC probability/token contract.

3. Use `profile_endgame_archive_decision.py` as a pre-dispatch byte gate for
   PR85-family public or local variants. A candidate should not reach a lane
   claim unless the tool shows strict ZIP validity, known segment/side-info
   validation, and a byte-negative or component-motivated transplant estimate
   against the STBM frontier.

## Recursive Review

- OSS hygiene: new code is deterministic, local-path agnostic, uses standard
  argparse, emits stable sorted JSON, and contains no credentials, provider
  endpoints, or hard-coded operator account data.
- Reproducibility: tests construct deterministic ZIPs with fixed timestamps and
  stored members; report output contains no wall-clock timestamp or random id.
- Contest compliance: tool is byte-only; it does not inflate videos, load
  scorers, modify upstream scorer files, dispatch jobs, or promote scores.
- Paper citability: report JSON records archive bytes/SHA-256, segment bytes,
  segment SHA-256, codec labels, and evidence grade. Claims here are marked
  byte-level decision support, not CUDA score evidence.
- Secret leakage: output and ledger contain local artifact paths and public
  archive anatomy only. No private provider URL, token, SSH target, or account
  metadata is written.

Verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_endgame_archive_decision.py -q
```

Result: `3 passed in 0.09s`.
