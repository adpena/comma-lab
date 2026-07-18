# Codex session summary — V10 factor 2 uint8 lattice — 2026-07-18

## Pointer and scope

`0.1910828242 [contest-CPU Linux x86_64]` is unchanged. This session built and
measured a research-only local factor-2 primitive. It launched no training,
paid/GPU work, full n600 scorer pass, evaluator, submission, or frontier
mutation, and it did not write the sacred c2 result tree.

## Durable landing set

- exact rational bounded uint8 preimage solver with typed proof/candidate and
  hard-oracle repair statuses;
- 99 behavioral tests, including hidden-source refusal, brute-force/certificate
  regressions, numeric/immutability/overflow guards, fail-closed resume and
  source custody, parser guards, cycle handling, and decoded uint8 through
  actual frozen CPU SegNet;
- bounded resumable n6 measurement tool and exact JSON/Markdown receipts;
- triality DAG FEED and non-registered canonical-equation candidate;
- factor-2 completeness delta and adversarial findings memo.

## Measured result

On deterministic pairs `[90,175,277,381,424,573]` at
`[macOS-CPU advisory subset]`, all `3,538,944` RGB channel-blocks and all six
frame aggregates were exact. `clip(round(B(y)))` had maximum `A` error
`63.824981689453125` and 520 hard Seg mismatches (`d_seg=0.0004408094618`).
The parsed exact candidate had zero integer-numerator residual, zero Seg
mismatches, recovered 520/520 failed cells, and introduced zero regressions.
Its `11,346,894`-byte sidecar is not a rate win or contest archive.

## Adversarial correction

The initial adjacent-corner completeness premise was falsified before code and
replaced with exact gcd-pruned bounded DFS. The first independent code review
was not clean: it found source-copy control, false hard-feasibility, early cycle
exit, proof/provenance accounting, missing real-SegNet canary, and status-name
bugs. A later full-landing pass caught a second fail-open class: malformed exact
scalars could be coerced into a certificate, and the equation candidate asserted
existence unconditionally. Later passes found widened/extreme tolerance,
coercible array dtypes, overflow, mutable certificates, changing oracle shapes,
unbounded fallback arity, and serializer asymmetry. A final wrapper pass found
stored resume metrics, import-path identity, aggregate-payload identity, and a
second-filesystem preflight could fail open. All were fixed; resume now
re-derives prior rows and the executed frozen scorer paths are byte/path-bound.
The test count rose from 12 to 99. The final receipt SHA is
`665ce8ecd789...`, bound to solver `5039902d8de5...`, tests
`c4a532f7ba5c...`, and tool `51103ef9a97f...`.

Four earlier receipts were losslessly preserved on the SSD with
machine-readable move manifests rather than overwritten. The fourth is
`preseal_receipt_tool_b32ebb8c.json` (SHA `88e9b9b31f34...`; move-manifest SHA
`f8b924787056...`); the third is `preseal_receipt_solver_5897df7b.json` (SHA
`afdbdd761c03...`; manifest SHA `03b097f6655d...`). All four are explicitly
superseded and non-authoritative for landing. Reviews also caught and corrected
stale run-specific fields before the final JSON/Markdown binding.

## Honest disposition

Completeness factor 2 is `HAVE (advisory local primitive)` / `PARTIAL`, not
complete or adopted. The full family remains open. Full n600 governed replay,
Pose/both-frame interaction, counted receiver/archive rate closure, identical
contest-CPU/CUDA replay, and independent MAIN landing review remain owed.

The isolated branch is subject to the canonical three-clean-review and commit
serializer gates. Their completion does not substitute for MAIN's review of
the algorithm, hash custody, frozen-scorer path, factor disposition, and
remaining blockers.
