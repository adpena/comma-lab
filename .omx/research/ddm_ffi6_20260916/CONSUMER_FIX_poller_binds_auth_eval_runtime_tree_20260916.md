# Consumer fix — the harvest poller's terminal claim row binds the auth-eval runtime-tree sha

`# FORMALIZATION_PENDING: a consumer-fix adjudication for one frozen-contract row. It changes no
definition, no digest, and no term of the score; it records why one pinned consumer of the pr19
definition moved, and re-pins the implementation manifest. Nothing here is a law to formalize.`

**Axis: `[source + exact digest arithmetic]`. No score claim, no promotion claim, no fire ordered by
this memo. Frontier line: composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600]
(move 53) — unmoved by this row.**

## The drift (MEASURED on the live tree, 2026-09-16)

The frozen contract's newest row before this one is the pr19 consumer-fix batch
(`pr19_identity_class_envelope_consumers_20260911`, implementation commit
`a9fd1270495522f5335737ac56643cf8ace8e670`). Being the last row, its implementation manifest
(`.omx/research/ddm_pr19_20260911/PREFIRE_IMPLEMENTATION_MANIFEST_PR19.json`) is the one the
contract holds to the **live** bytes: `tac.candidate_seal._pf_contract` walks the amendment list and
passes `require_live=True` for `index == len(amendments) - 1`, which runs `_pf_blob` on every path.

I re-measured all 15 pinned consumers against their live bytes. **Fourteen are live-clean. One is
not:**

| path | pinned at `a9fd1270` | live (== `HEAD`) |
|---|---|---|
| `tools/modal_harvest_poller.py` | `2a98ab6276a7b1692d822b7eb1cd0a41f77cdfc419c7155a758258839905ef4f`, 36,333 B | `bc7383343a5f918b946d358f64bb18d79328da45dad556bc2604a26349324e8e`, 39,394 B |

`git log a9fd1270..HEAD -- tools/modal_harvest_poller.py` names exactly one commit:
**`da66353b7ac0e884e5da71ee5f342e38c0f8e4be`** (2026-09-11T22:03:06-05:00), touching that file and
`src/tac/tests/test_pointer_move_packet.py` and nothing else. No commit has touched the poller since.
The working tree is clean for it, so live == HEAD == `da66353b7`'s blob.

The consequence is candidate-independent: **no pre-fire intent could be produced for any candidate.**
mrs5 hit it for real on 2026-09-16T19:17:58Z with truthful evidence at the pointer's own bytes —
`PREFIRE_CONTRACT_DRIFT_REFUSED: live file differs from committed blob:
/Users/adpena/Projects/pact/tools/modal_harvest_poller.py`, retained at
`.omx/research/ddm_mrs5_20260916/prefire_attempt/PREFIRE_REFUSAL_43ca2b00aebd4c4296d8356283327f0c.json`
(324 B, sha `8df50f1c0ee52069bf3eafef2060b7d63d0d0da68a07571e9ad80ee30986b346`). **That refusal is the
contract working.** This row is what answers it — through the contract's own amendment procedure,
not around it.

## What moved in the consumer, re-derived from the diff (not from the commit message)

`da66353b7` adds one private helper and rewrites eight lines of one existing note builder:

1. **`_auth_eval_provenance_runtime_tree_sha256(result)`** (new, ~35 lines). It reads
   `result["artifacts"]["contest_auth_eval.json"]` — a JSON string or an already-parsed object — and
   returns the first 64-hex `runtime_tree_sha256` found at the payload root, under `provenance`, or
   inside an `inflate_runtime_manifest` at either scope. Its docstring says it mirrors
   `scripts/pre_submission_compliance_check.py::_runtime_tree_candidates`, and it does: same four
   lookup sites, same hex shape. **Every failure path returns `None`** — not a dict, missing
   artifact, `JSONDecodeError`, non-hex — so the caller falls back to the receipt's own field rather
   than forging a binding.
2. **`canonical_terminal_claim_notes`** previously wrote
   `runtime_tree = result.get("expected_runtime_tree_sha256") or result.get("runtime_tree_sha256")`.
   It now prefers the auth-eval provenance value and, *when the two differ*, emits the old value
   separately as `t4_runtime_digest_sha256=`. Both are copied verbatim from the receipt; neither is
   re-derived or truncated.

The defect it cures is real and was measured elsewhere: a first-measurement receipt carries **two**
runtime-tree digests under **different definitions** — the top-level `expected_runtime_tree_sha256`
is the timing leg's `tac.decode_wall_clock.measure_t4_runtime_digest`, while the compliance
checker's `auth_eval_runtime_tree_expected_match` reads the auth-eval provenance value. On a normal
receipt they agree; on the move-48 packet they did not, so the terminal claim row bound the timing
digest and no single `--expected-runtime-tree-sha256` could satisfy both checks (89/93 either way).

**Executed proof, on the live bytes (MEASURED):**
`src/tac/tests/test_pointer_move_packet.py::test_terminal_claim_note_binds_the_auth_eval_runtime_tree_on_a_first_measurement_receipt`
— 1 passed; the whole `test_pointer_move_packet.py` suite — 20 passed.

## Why this is a consumer fix and not a definition change

`definition_change: false`, and it is false in substance, not only in the field:

- The poller is a **harvest-side note writer**. It runs after a measurement returns. It is not on the
  intent-production path, not on the authorization path, and not on the fire path. Nothing it emits
  enters `candidate_prefire_intent.v1`, `candidate_prefire_timing_risk.v1`, or any digest the
  contract compares.
- **No digest definition moved.** `tac.candidate_seal.measure_prefire_risk_receiver_digest.v1`,
  `tac.decode_wall_clock.measure_receiver_behavior_digest.v2`, and the pr19 identity-class envelope
  (`tac.candidate_seal.validate_prefire_risk.identity_class_envelope.v1`) are all untouched by
  `da66353b7`; the diff touches one tool and one test file.
- **Direction of strictness.** The change makes the terminal claim row bind the value the compliance
  checker actually reads, and keeps the other digest instead of discarding it. It removes an
  ambiguity; it relaxes no check.
- The definition parent is therefore unchanged: `05aa14352f01ac356387c40577cd6a2d94eff37b7d010ec1df3c9d3b60556da8`
  (the pr19 identity-class envelope amendment), exactly as the pr19 consumer-fix row names it.

## Attacking it

*Could the new helper bind a forged value?* It copies a 64-hex string out of the harvested receipt's
own embedded auth-eval JSON and refuses anything else by returning `None`. It computes nothing.

*Could preferring the auth-eval value lose the timing digest?* No — the differing case is exactly the
case that adds `t4_runtime_digest_sha256=`. Both values survive in the row.

*Does re-pinning weaken the drift control?* No. The control is "live bytes must equal the named
commit's blob." This row changes **which commit** is named for one file, to a commit already on
`main`, already reviewed, already tested, and already an ancestor of `HEAD`. Any future edit to any
of the 15 consumers refuses again, unchanged.

## What this row does NOT do

No definition change. No new amendment id. No change to `PREFIRE_IMPLEMENTATION_PATHS` (still the
same 15 canonical paths, sorted). The other 14 pins are re-stated at their unchanged values.
Appending this row invalidates in-flight intents (pr17 NO-GRACE) — there are none: the producer has
been refusing at gate 0 since 2026-09-11, so nothing is in flight to invalidate.

It also does **not** answer the two refusals that sit behind gate 0 for a receiver-only candidate at
the pointer's own bytes (mrs5's §5 gates 1 and 2). Those are a contract-shape question for the
pr-family, recorded in `.omx/research/ddm_ffi6_prefire_contract_consumer_pin_amendment_20260916.md`
and deliberately not patched here.
