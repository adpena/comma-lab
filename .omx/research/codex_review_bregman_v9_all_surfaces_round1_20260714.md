# Round-1 adversarial review — Bregman all-surfaces V9·CGauge — 2026-07-14

Pointer: `0.1910828242 [contest-CPU Linux x86_64]`; local PR128
`0.1880443979880752` remains non-submission. **UNCHANGED.** This review has no
score, launch, promotion, evaluator, or MPS authority.

## Disposition

**BREGMAN FUNCTIONAL VERDICT: PASS. TRANSACTIONAL V9 FREEZE: OWNER SOURCE-SEAL BLOCKED.**

The Bregman helpers, equations, receipt, and standalone sealed DSL policy pass
fresh and independent review, including three clean `71 passed` runs. The
canonical basis source stabilized, but a later exclusive-owner edit reopened
the scientific-declaration seal. The latest end-to-end V9 group is `16 passed,
2 failed, 24 errors`; every failure/error is rooted in the fail-closed mismatch
`expected 6cfa9845...`, `live 5c926130...`, before the Bregman assertions are
reached. The previously observed owner test whose literal expected hashes
predate final source-closure hardening also remains unresolved. Neither issue
authorizes this lane to patch the exclusive provenance/config hot files; both
were routed to the owner.

## Fresh-eyes findings and disposition

| ID | Initial finding | Severity | Disposition |
|---|---|---:|---|
| R1 | Live V9 selected an unsealed default Bregman policy | P1 | **FIXED:** `spec_v9_cgauge` selects `sealed_v9_bregman_geometry_policy`; local receipt and durable binding are exact-content validated |
| R2 | Nested Bregman metric could drift from top-level optimal metric | P1 | **FIXED by owner:** config provenance requires exact nested/top-level binding equality; mutation regression exists |
| R3 | Missing nullable metric keys were accepted as explicit `None` | P1 | **FIXED by sibling/owner:** verifier checks missing keys separately; regression deletes a nullable key and refuses |
| R4 | `resolved_at` made repeat compile hashes volatile | P2 | **FIXED by owner:** volatile field removed before hashing; prior fresh repeats matched |
| R5 | Chernoff receipt did not bind endpoint/support semantics and used an unstable difference of KL values | P2 | **FIXED:** endpoint/support byte hashes and point IDs are required; direct likelihood-ratio bisector residual is used; float64 exhaustion refuses |
| R6 | Caratheodory reduction depended on an arbitrary LAPACK/SVD nullspace basis | P2 | **FIXED:** fixed source-order `D+2` selection, pure-scalar deterministic RREF, explicit rank-ambiguity refusal, exact eliminand zero, compensated mass/moment checks; SVD is forbidden by test |
| R7 | Measurement receipt lacked source/runtime/command custody, later developed a self-hash cycle, and initially did not revalidate its claimed source closure | P1/P2 | **FIXED:** git base, exact command, runtime, source hashes, and authority fields are bound and recomputed by the sealed policy. Policy source is normalized only at its two embedded receipt/binding hash literals; all other policy bytes are hashed. Rehashed source-map and normalized-source substitutions refuse |
| R7b | Durable binding validation checked only selected semantic fields, so a rehashed substituted surface payload could pass | P1 | **FIXED:** the sealed policy reconstructs the complete canonical receipt-only binding and requires exact payload equality; adversarial regression refuses a rehashed surface mutation |
| R8 | Spatial KL changes lacked direct resolution-invariance tests; old equation prose said CE equals KL | P2/P3 | **FIXED:** U-DIE, NSCS02, and pause/distill resolution tests exist; prose now states `CE(p,q)=H(p)+KL(p||q)` |
| R9 | Concurrent scientific-declaration and basis-source drift invalidated earlier end-to-end seals | P1 transactional | **REOPENED, EXCLUSIVE OWNER:** basis source is stable, but declaration seal now expects `6cfa9845...` while live table hashes `5c926130...`; strict V9 refuses before Bregman assertions |
| R10 | Owner strict test pins superseded Bregman receipt/binding hashes | P1 transactional | **OPEN, EXCLUSIVE OWNER:** previously observed after R9 was temporarily closed; current R9 refusal prevents this assertion from being reached |

## Receipt custody

- Deterministic measurement receipt:
  `.omx/research/bregman_v9_all_surfaces_measurement_20260714.json`, SHA-256
  `12b82ca3f9809339746cc03b48a3237643861dec9e9baec19852a184fa7f358c`.
  Two complete executions produced identical bytes.
- Durable V9 handoff binding:
  `.omx/research/bregman_v9_all_surfaces_binding_20260714.json`, SHA-256
  `bdc01ae586c4467b18ebf4deee206242426ddc51da3dc47d8a2b8fff6cab8481`.
  A `--reuse-existing-receipt` reconstruction produced identical bytes.
- Opt-in local timing advisory:
  `.omx/research/bregman_v9_all_surfaces_timing_advisory_20260714.json`,
  SHA-256
  `004c9dbc72e1082e82b0eaf0608173bf6f770c552a3e6cdf0c93391feed1b897`.
  It measured `112.63732662919126x` for repeated synthetic log-ratio sums after
  excluding one-time Caratheodory setup. It is deliberately not a canonical
  constant or live-throughput verdict; the deterministic result is `600→5`,
  exactly `120x` point-count reduction.

## Verification evidence at review freeze

- Bregman/metric/equation/receipt/policy group: three clean fresh-process runs,
  `71 passed` each.
- KL consumer and shape-custody group: `193 passed`.
- Strict KL-reduction selector: `1 passed, 306 deselected`, zero violations.
- Owned new-file Ruff check: green after required formatting.
- Standalone sealed policy: exact receipt/binding hashes accepted; metric ID
  and nested metric ID both `argmax_native_vjp_fidelity_v1`; activation false;
  gauge status `GAUGE_IMPLEMENTATION_CUSTODY_GAP`.
- End-to-end V9 group: `16 passed, 2 failed, 24 errors`; all red outcomes are
  rooted in the exclusive-owner declaration seal mismatch before Bregman
  assertions. The stale receipt/binding literals remain known but unexercised
  under the current earlier refusal.

The headless process prints an existing Metal-device atexit warning after some
CPU-only tests. It is not a test failure and grants no MPS authority.

## Serializer disposition

The exact owned-new-file serializer attempt refused with `rc=6` before
staging or committing anything. Its dependency fence found that sibling-owned
`src/tac/scorer_surrogate/vjp_fidelity.py` is still absent from `HEAD`; landing
this lane alone would create a clean-HEAD-breaking partial commit and would
misattribute the sibling metric implementation. The index remained empty and
`HEAD` remained `927ef10723`. Exact file hashes and hunk-only shared-file
instructions are in
`.omx/research/codex_harvest_bregman_v9_all_surfaces_20260714_codex.md`.

## Freeze rule

Do not call the complete V9 strict group green until R9 and R10 are repaired by
the exclusive owner and the group passes against the final receipt hashes.
No other Bregman functional finding remains open at this review point.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; V9 DSL/policy/config/test source; Bregman helper,
equation, receipt, and test source; retained measurement artifacts; fresh
independent reviewer report; lane/subagent progress; watched arm and fleet
inboxes.
