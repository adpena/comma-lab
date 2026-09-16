# ddm_ffi6 — the frozen pre-fire contract's consumer pins, re-pinned for the 2026-09-11 poller fix; intents are producible again (2026-09-16)

<!-- # FORMALIZATION_PENDING: this arm lands one typed consumer-fix row in the frozen contract's own
amendment list and reports two contract-shape questions for the pr-family. It registers no canonical
equation: a consumer pin is custody bookkeeping, not a law, and the two open questions are explicitly
handed to an adjudicator rather than answered here. -->

**Axis: `[source + exact digest arithmetic]` and `[macOS-CPU advisory]` for the producer run. No score
claim, no promotion claim, no Modal, no fire, no authorization, no packet, no pointer move. $0.**

**The finding in one line (MEASURED): gate 0 no longer fires.** Before this arm, every candidate's
pre-fire intent refused at `PREFIRE_CONTRACT_DRIFT_REFUSED` because one of the 15 pinned consumers had
moved on `main`. One typed `prefire_contract_consumer_fix.v1` row — the contract's own cure — restores
production. A real producer run now advances past the contract gate and stops at the candidate-specific
gate, which is the honest, recorded gap.

## 1. The drift (MEASURED)

`tac.candidate_seal._pf_contract` walks the frozen amendment list and passes `require_live=True` for
`index == len(amendments) - 1`. That means the **newest** row's implementation manifest is the one held
to the working tree's live bytes, through `_pf_blob`. Before this arm the newest row was pr19's
consumer-fix batch (`pr19_identity_class_envelope_consumers_20260911`, implementation commit
`a9fd1270495522f5335737ac56643cf8ace8e670`, manifest
`.omx/research/ddm_pr19_20260911/PREFIRE_IMPLEMENTATION_MANIFEST_PR19.json`).

I re-measured all 15 pins against live bytes. **14 live-clean; exactly one drifted:**

| file | pinned at `a9fd1270` | live (== `HEAD`) |
|---|---|---|
| `tools/modal_harvest_poller.py` | `2a98ab6276a7b1692d822b7eb1cd0a41f77cdfc419c7155a758258839905ef4f`, 36,333 B | `bc7383343a5f918b946d358f64bb18d79328da45dad556bc2604a26349324e8e`, 39,394 B |

`git log a9fd1270..HEAD -- tools/modal_harvest_poller.py` returns exactly one commit:
**`da66353b7ac0e884e5da71ee5f342e38c0f8e4be`** (2026-09-11T22:03:06-05:00), touching that file and
`src/tac/tests/test_pointer_move_packet.py` only. Nothing has touched it since; the working tree is
clean for it, so live == HEAD == that commit's blob. The freeze record itself was never inconsistent —
every row's sha still matched its own committed blob. **The tree moved on, not the record.**

The change itself (DERIVED by reading the diff, not the commit message): a new private helper
`_auth_eval_provenance_runtime_tree_sha256(result)` reads the 64-hex `runtime_tree_sha256` off the
harvested receipt's embedded `contest_auth_eval.json` — the value
`scripts/pre_submission_compliance_check.py::_runtime_tree_candidates` actually reads — and returns
`None` on every failure path; `canonical_terminal_claim_notes` then prefers it and, when the timing
leg's `expected_runtime_tree_sha256` differs, carries that one separately as
`t4_runtime_digest_sha256=`. Both values are copied, never re-derived. It is a **harvest-side note
writer**: not on the intent-production, authorization, or fire path, and it touches no digest
definition. Its test passes on the live bytes:
`test_terminal_claim_note_binds_the_auth_eval_runtime_tree_on_a_first_measurement_receipt` — 1 passed;
`test_pointer_move_packet.py` — 20 passed (MEASURED).

## 2. The amendment row I produced

The contract's amendment procedure needs no code. `_pf_freeze_history` accepts an appended
`prefire_contract_consumer_fix.v1` row with an exact ten-key field set; `_pf_contract` then re-pins the
live check to that row's manifest. **No code change was required anywhere, so the charter's STOP branch
did not fire.**

| field | value |
|---|---|
| `schema` | `prefire_contract_consumer_fix.v1` |
| `fix_batch_id` | `ffi6_poller_auth_eval_runtime_tree_pin_20260916` |
| `previous_row_sha256` | `d9553f662a04428f…` (`prefire_digest` of pr19's consumer-fix row) |
| `definition_parent_sha256` | `05aa14352f01ac356387c40577cd6a2d94eff37b7d010ec1df3c9d3b60556da8` — pr19's identity-class envelope, **unchanged** |
| `definition_change` | `false` |
| `consumer_fixes[0].fix_id` | `ffi6-poller-auth-eval-runtime-tree-pin` |
| `consumer_fixes[0].causal_commit` | `da66353b7ac0e884e5da71ee5f342e38c0f8e4be` |
| `consumer_fixes[0].tests` | `src/tac/tests/test_pointer_move_packet.py::test_terminal_claim_note_binds_the_auth_eval_runtime_tree_on_a_first_measurement_receipt` |
| `implementation_commit` | `afc2b294b8709a7031eddb8cc49a90457ce996b6` |
| `implementation_manifest` | `.omx/research/ddm_ffi6_20260916/PREFIRE_IMPLEMENTATION_MANIFEST_FFI6_POLLER.json`, 2,251 B, `849e47963952983f74945e112bb018b79b484a90bc9949d2cc981e4a43dec687` |
| `adjudication_memo` | `.omx/research/ddm_ffi6_20260916/CONSUMER_FIX_poller_binds_auth_eval_runtime_tree_20260916.md`, 7,447 B, `a39aacb8d3c42b66931b9077be3ca06c0bca725c26508172ca739ec0bdfa1876` |
| `score_claim` | `false` |
| row digest | `f1c76c8b11dd61a03410db298adf03cfc31ba0f8a714b229617be8cf540f9795` |

The manifest re-states the same **15** canonical `PREFIRE_IMPLEMENTATION_PATHS`, sorted, with
`{path, bytes, sha256}` — 14 at their unchanged values, the poller at `bc733834…`/39,394 B. It is
serialized byte-for-byte the way pr19's is (`json.dumps(rows, indent=1, ensure_ascii=False) + "\n"`).

**Why two commits (DERIVED from the validator, and from how pr18/rlc5 did it).** `_pf_contract` runs
`_pf_blob(repo, amendment_commit, adjudication_memo)`, so the row's memo and manifest must already be
committed at the commit the row names; the row cannot name a commit that does not yet exist. So commit A
lands the two pinned objects, and commit B appends the row naming commit A. That is the same two-step
shape as amendment[8] (`causal d1ed3674b`, `implementation 79a50df08`).

**The append is pure.** `git diff --numstat` on the frozen file: **30 insertions, 0 deletions**. All 11
prior rows re-digest identically and every top-level key is unchanged (MEASURED). I edited contract
**data** — the frozen receipt's own `amendments` list, which is exactly where the contract says a drift
refusal is cured. I edited no contract **code**: `src/tac/candidate_seal.py`,
`src/tac/decode_wall_clock.py`, and `tools/modal_harvest_poller.py` are untouched (`git diff` empty for
all three).

## 3. Receipts, before and after

`GATE0_RECEIPT_BEFORE.json` / `GATE0_RECEIPT_AFTER.json` hold these in full.

**BEFORE** (at `HEAD` `92afd5cf7f0d39d…`, 11 rows). A contract block built exactly as
`build_prefire_intent` builds it, from the live frozen file, run through `_pf_contract`:

```
PREFIRE_CONTRACT_DRIFT_REFUSED: live file differs from committed blob:
/Users/adpena/Projects/pact/tools/modal_harvest_poller.py
```

— the same detail, verbatim, as mrs5's real producer refusal
(`PREFIRE_REFUSAL_43ca2b00aebd4c4296d8356283327f0c.json`, 2026-09-16T19:17:58Z). 14/15 pins clean.

**AFTER** (at `HEAD` `5a603bf07ca7e44…`, 12 rows). Same probe: **every contract drift check PASSES** —
freeze history valid over 12 rows, every manifest row matches its owner commit's blob, the newest row's
15 pins all match live (15/15), the amendment memo matches its commit's blob, the pr12 memo pin holds.
The probe then stops on `KeyError 'evidence'` because a stub intent carries no evidence block; that is
downstream of every drift check.

**The real producer run (MEASURED, the proof the charter asked for).**
`tools/make_candidate_seal.py --first-fire-intent` on mrs5's real inputs
(`--runtime-dir $PWD/submissions/mrs5`, mrs5's eight evidence files, `--admit-bar-net-ds 0.0`,
`--bar-tolerance 0`, both axes `contest_cuda`), output into this arm's own probe directory:

```
PREFIRE_IDENTITY_DRIFT_REFUSED: receiver/archive pins disagree
rc=3
```

Receipt: `.omx/research/ddm_ffi6_20260916/prefire_probe/PREFIRE_REFUSAL_0b5b7b2ce059402db56e7558a377ed18.json`
(258 B). **`PREFIRE_CONTRACT_DRIFT_REFUSED` is gone.** No intent object was created — the producer
deletes a failed object — so nothing is in flight and pr17 NO-GRACE invalidates nothing.

**Regression control:** `test_candidate_prefire_intent.py`, `test_candidate_seal.py`,
`test_decode_wall_clock_t4_direct.py`, `test_decode_wall_clock.py`, `test_pointer_move_packet.py` —
**297 passed** after the append (MEASURED).

## 4. The gap behind gate 0 — questions for a pr-family memo, not patched here

These are mrs5's gates 1 and 2. I reproduced both on the live tree and changed neither. They are
candidate-specific, they are about a **receiver-only candidate at the pointer's exact bytes**, and they
are the contract's shape, so an adjudicator owns them.

**Gate 1, verbatim (MEASURED):**

```
PREFIRE_RISK_EVIDENCE_REFUSED: both archive pins required
```

raised by `tac.candidate_seal.measure_prefire_risk_receiver_digest(submissions/mrs5)`. Its sister
normalizer `tac.decode_wall_clock.measure_receiver_digest` accepts the same directory and returns
`6ee7beeeee7ee4ec893d71a3126e84b47be2a5cda9233b718a58ae8b2266dccc`.

> **Question 1 for the pr-family.** Two normalizers in one contract family disagree about the same
> directory: `tac.candidate_seal._materialize_prefire_receiver_rows` *requires* the top-level
> `ARCHIVE_SHA256`/`ARCHIVE_BYTES` assignments before it normalizes them to `<ARCHIVE_PIN>`, and
> `tac.decode_wall_clock._receiver_rows` performs the same normalization without requiring them. Is the
> stricter side the intended identity rule — i.e. does the risk digest deliberately have **no identity
> class for a pinless receiver**, so that pricing one from a pin-bearing measured leg is a transfer the
> contract refuses on purpose — or is the requirement an artifact that should be relaxed to match the
> sister? **My reading (INFERRED, and it agrees with mrs5's): a working control.** The two digests carry
> different definition names the contract never interchanges, and the difference is one explicit
> `_pf_require`, not an environment. But the contract does not say so in words anywhere I could find, so
> a reader cannot tell a designed refusal from an oversight without reading both functions.

**Gate 2, verbatim (MEASURED at the producer):**

```
PREFIRE_IDENTITY_DRIFT_REFUSED: receiver/archive pins disagree
```

from `src/tac/candidate_seal.py:2044`,
`_pf_require(check_pin_consistency(root, archive_path=archive).ok, code, "receiver/archive pins disagree")`.
`check_pin_consistency(submissions/mrs5)` returns `verdict='PIN_ABSENT'`, `ok=False` (MEASURED).

Behind it sits the arithmetic half, which the producer never reached this time because identity fires
first. `_pf_pointer` at `:2072` requires `_pf_number(bar["net_dS_threshold"], …) < 0` **strictly**, and
at `:2080` requires `ds < bar["net_dS_threshold"]`. For a receiver-only candidate at the pointer's own
bytes, `derived_net_dS = 25 × (179,286 − 179,286) / 37,545,489 = 0.0` exactly, and no negative threshold
is above zero.

> **Question 2 for the pr-family.** The frozen `candidate_prefire_intent.v1` is shaped for a candidate
> that **improves the score**: a strictly negative admit bar and a rate delta below it. A receiver-only
> change at identical archive bytes improves nothing by construction — its purpose is decode cost and
> reviewability — so it has **no expressible admit bar**, and `completed_t4_receiver_delta`'s one-sided
> `max(0, ceiling/base − 1)` clamp has no field for a ratio below 1, so mrs5's measured 6.8% *cheaper*
> decode projects as unchanged. tc4/pr19 already hold that a receiver change needs its own measured leg
> and that the intent chain exists for exactly that. Does the intent object therefore grow a
> receiver-only mode (net dS = 0 admissible, bar expressed on decode seconds rather than S), or is a
> receiver-only candidate meant to reach a measured leg by some other door the contract already has?
> **I did not answer this and did not patch toward either branch.**

> **Question 3, smaller.** Is `PIN_ABSENT` itself the intended verdict for a deliberately unpinned
> minimal receiver, or should `check_pin_consistency` distinguish "pins disagree" from "no pins
> declared"? The refusal text says "pins disagree" for a tree that declares none. That is a wording
> question with an evidence consequence: a reader of the receipt cannot tell the two apart.

**verdict_scope for §4:** these are statements about the frozen `candidate_prefire_intent.v1` /
`candidate_prefire_timing_risk.v1` shapes as of commit `a47543199` plus its twelve landed amendment
rows, evaluated on **one** candidate — a receiver-only tree at the pointer's exact archive bytes with no
declared archive pin. They say nothing about score-moving candidates, which is what every prior intent
has been, and they are not a claim that any of the three refusals is a defect.

## 5. Boundaries honoured

No Modal, no fire, no authorization, no nonce, no packet, no PR, no push, no `authorize_*`, no scorer,
no timing sample, no pointer move, no promotion. No writes to `/Volumes/APDataStore/pact/ddm_pd7/` or
`ddm_mrs5_fire/`, and no writes to either SSD at all. No edits to `upstream/`, sealed trees, PR trees,
`submissions/mrs5`, or any prior arm's evidence directory. No edits to contract **code**:
`src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`, `tools/modal_harvest_poller.py` are
byte-unchanged. The one contract **data** edit is the append this arm was chartered to produce, and it is
pure append. Prior amendment rows were not rewritten. The producer run created no lifecycle object; its
typed refusal is retained. All new files live under `.omx/research/ddm_ffi6_20260916/`.

## 6. NEXT_IF_RESUMED

- **QUEUED; owner pr-family; trigger: the next receiver-only candidate.** Adjudicate questions 1–3
  above. Until then a receiver-only candidate at the pointer's own bytes has no door, and gate 0 being
  open does not change that.
- **QUEUED; owner MAIN; trigger: the next score-moving candidate.** The producer is unblocked for every
  candidate that carries an archive pin and a strictly negative bar; no further contract work is owed.
- **STANDING; owner whoever next moves a pinned consumer.** The fifteen pins are custody, not a wall:
  re-pinning is one appended row plus two commits, and it needs no code. Twelve rows now; the newest is
  `f1c76c8b…`.

## 7. DEAD-ENDS

- Using rlc5's committed `CANDIDATE_PREFIRE_INTENT_v4.json` as the before/after probe: it refuses
  earlier, at `"latest exact frozen amendment required"`, because it pins an older row. It cannot show
  the live-drift detail. Replaced by a contract block built the way `build_prefire_intent` builds it.
- Naming `da66353b7` as the row's `implementation_commit`: the validator requires the row's adjudication
  memo to be blob-identical **at that commit**, and a memo written today is not in a commit from
  2026-09-11. The two-commit shape is forced by the contract, not chosen.
- Editing `candidate_seal.py` or the poller to make the refusal go away: forbidden, unnecessary, and the
  exact "patch around the control" this contract exists to prevent.

composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53) — unmoved by this arm.
