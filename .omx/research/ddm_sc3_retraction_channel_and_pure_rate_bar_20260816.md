---
arm: ddm_sc3
title: "The arm-queue plan surface now has a retraction channel, corrections reach the reader mechanically, and the sub-0.15 pure-rate bar is a registered DERIVATION — whose four-base identity claim I had to correct at its own first base"
utc: 2026-08-16
axis: "[local-CPU $0 apparatus + exact decimal arithmetic over MEASURED contest-CUDA receipts] — NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] — UNMOVED by this unit"
verdict_scope_default: "stated inline per claim"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_sc3 — the retraction channel, and the bar that cannot go stale

STORES CONSULTED (primary, re-derived not quoted):
`.omx/state/codex_arm_queue.next_if_resumed.jsonl` (248 rows at entry) ·
`tools/codex_arm_queue.py` · `tools/costate_digest.py` · `tools/modal_endpoint_close.py` ·
`tools/build_ddm_uf1_refresh_registry.py` · `.omx/state/canonical_frontier_pointer.json` ·
`.omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md` ·
`.omx/research/ddm_mc36_dual_axis_t4_verdict_20260814.md` ·
`.omx/research/ddm_hv1_harvest_compose_ep508_20260815.md` ·
memories [[structural_beats_procedural_and_the_detector_that_zeroes_on_the_cure_20260803]]
[[corrections_land_in_bodies_headlines_keep_the_stale_number_20260805]]
[[orphan_sweeps_that_do_not_write_the_store_are_the_disease_20260803]].

## ANSWER FIRST

**JOB 1 — BUILT AND IN USE. The channel is additive, no reader broke.** The plan surface
now carries an append-only retraction row. Correcting a source memo and re-extracting it
**auto-retracts** the pre-correction row, so a fix at the source reaches the consumer
without anyone remembering to do it. Four real rows are retracted: `rx1`, `rx2`, `sp2`
SUPERSEDED (the `< 186,269 B` bar), `wd2` AMEND_REQUIRED (the `15,157 B` rung — a row I
found myself; `fb1` named the file but not the queue row). The live reader serves
**245 of 248**, and says so on its own line.

**JOB 2 — REGISTERED, WITH A CORRECTION TO THE CLAIM THAT MOTIVATED IT.**
`sub015_pure_rate_archive_byte_bar_v1` is a DERIVATION off the live pointer, not a
literal. Re-deriving at 44 digits **falsified `fb1`'s four-base identity at its own first
base**: MC36 / e480b v2 / hv1 agree to **7.23e-11 B** (bar **168,345.5977 B**), but
**cp135 gives 168,297.5395 B — 48.058 B lower**, because `cp135 → MC36` was a DISTORTION
move (MC36's own verdict: seg −37 flips = −3.136529e-5 S). The pure-rate caveat is not a
footnote to the invariant; it is the condition that generates it, and cp135 is the worked
proof. I registered the corrected scope with the counter-example embedded so nobody can
re-broaden it.

**THE POINTER DID NOT MOVE. This is apparatus, not goal progress.**

---

## A. JOB 1 — the retraction channel

### A.1 The defect, re-derived at source

`row_id = sha256(schema | arm | source_path | line_start | block_sha256)`
(`tools/codex_arm_queue.py::extract_next_if_resumed`). The stored schema is
16 fields with **no** supersede / stale / retract / active / status field — I enumerated
them off the live store, not off the docstring. So a corrected memo minted a second row
and the first one lived forever beside it, both served.

MEASURED hazard, independently reproduced from the live store: rows 226 / 227 / 235 carry
`archive below 186,269 B` against a 182,759 B live archive. A candidate at that bar passes
while scoring **+0.002337165 worse** than what we ship — 233.7× the 1e-5 naming bar, in
the anti-conservative direction.

### A.2 The design, and the two constraints that shaped it

- **Append-only.** A retraction is a NEW row with its own schema string
  (`codex_arm_queue.next_if_resumed.retraction.v1`) keyed by `target_row_id`. No stored
  row is ever mutated or deleted — pinned by a test that compares the store byte-for-byte
  before and after.
- **Additive / legacy-compatible.** Retraction rows carry a different schema, so any
  pre-existing reader filtering `schema == ...next_if_resumed.v1` simply does not see
  them. The 248 legacy rows load unchanged (`retracted=False`).
- **Never a silent drop.** A retraction that shrinks the queue without saying so is the
  vacuity==pass disease. `next_if_resumed_debt()` reports the DENOMINATOR — total, live,
  superseded, amend-required, and the reasons — and the costate line prints
  `retracted: N superseded (hidden), M amend-required`.
- **Two dispositions, because I measured two shapes.** `SUPERSEDED` (the whole point of
  the row is the dead bar) is hidden by default. `AMEND_REQUIRED` (one stale clause beside
  live follow-ons — exactly `wd2`) stays SERVED and stamped: hiding it to suppress one
  clause would trade a loud defect for a silent one.
- **Fails closed.** Retracting an unknown `row_id` is REFUSED — a retraction that targets
  nothing looks like a cleared hazard, which is worse than none. Placeholder reasons are
  refused (Catalog #287 sister discipline), as are a missing citation and an unknown
  disposition. Every refusal path is tested and leaves the row live.

### A.3 The mechanical half — this is the part that makes the class stop recurring

`_auto_retract_reextracted_block` fires inside the extractor: same arm, same source path,
same start line, **different block hash** ⇒ the prior row is retracted with the two block
hashes named. Scoped deliberately tight — a different source file is a different plan, not
a correction, and must not be retracted by inference (pinned by a negative-control test).

### A.4 Readers — what I mutation-tested, and what I did not

| surface | role | mutation-tested? |
|---|---|---|
| `tools/costate_digest.py::section_arm_next_if_resumed` | **the reader** | **YES** — plant retraction ⇒ row gone + line says so; remove retraction ⇒ row returns. Plus an AMEND_REQUIRED control proving it stays served. |
| `tools/codex_arm_queue.py` (`load_next_if_resumed`, `next_if_resumed_debt`, CLI) | reader API + producer | **YES** — 8 tests: plant/remove cycle, append-only byte check, four refusal paths, auto-retract positive + negative control, debt denominator, legacy load. |
| `tools/modal_endpoint_close.py` | **PRODUCER only** — extracts blocks from memo TEXT, never parses the JSONL | **YES, as a producer**: a closure re-run after a memo correction auto-retracts the stale row (1 test). Not tested as a reader because it is not one. |
| `tools/build_ddm_uf1_refresh_registry.py` | **NOT A READER AT ALL** | n/a — see A.5 |

I swept every `.py` / `.sh` / `.md` under `tools`, `src`, `scripts`, `experiments`,
`.claude` for the store path. Beyond the three tools, only three `experiments/` files
mention it, and all three only as a `"consumer_store"` STRING inside their own receipt
JSON (`ddm_pk2`, `ddm_pk3`, `ddm_rx2_midrun_serialization_probe`). None parses it.

### A.5 Correction to the charter: `build_ddm_uf1_refresh_registry.py` is not a reader

My charter listed it among the readers. It is not. It contains **zero** references to
`codex_arm_queue`; its `_next_if_resumed()` returns a hardcoded markdown string that it
writes to its OWN per-arm `NEXT-IF-RESUMED.md` output file. Same words, different object —
the retrieval-hazard class the queue's own module docstring warns about for
`dispatch_queue.md`. Nothing to wire.

### A.6 Used on the live store

```
before   248 plan rows served
after    245 served | superseded 3 (hidden) | amend-required 1
```

`rx1` / `rx2` / `sp2` SUPERSEDED; `wd2` AMEND_REQUIRED. Each reason prices the hazard,
names the direction (anti-conservative vs conservative), and points at the derivation to
re-file against rather than at another literal.

---

## B. JOB 2 — the pure-rate byte bar as a LawRef

### B.1 The form

```
distortion   = S_base - 25*B_base / D          (the seg+pose legs, S units, exact by subtraction)
bar_bytes    = (0.15 - distortion) * D / 25
required_cut = B_base - bar_bytes
```

`pure_rate_byte_bar_from_pointer()` reads `.omx/state/canonical_frontier_pointer.json` and
returns the bar **together with the base it used** — sha, bytes, axis, measured-at — so a
receipt shows WHICH frontier the bar came from instead of asserting a bare number. It
fails closed on a malformed pointer and REFUSES an upstream-only `effective_frontier`
(we do not hold that archive; inventing a byte bar off it would be false authority).

Recovering the distortion leg by subtraction matters: published seg legs are rounded to 6
significant figures while composed S is carried to 17, so no rounded component enters.

### B.2 The four-base check — MEASURED, and it corrects `fb1`

| base | S | bytes | seg+pose | bar (B) |
|---|---|---:|---|---:|
| cp135 | 0.16195513827824176 | 186,252 | 0.0379375765413311021 | **168,297.5395** |
| MC36 Variant C | 0.1619344578804448 | 186,269 | 0.0379055765413310652 | 168,345.5977 |
| e480b v2 | 0.1600920261571558 | 183,502 | 0.0379055765413311133 | 168,345.5977 |
| hv1 ep0634 | 0.15959729295498598 | 182,759 | 0.0379055765413310666 | 168,345.5977 |

- MC36 / e480b v2 / hv1 agree to **7.23e-11 B**. Decode-identical distortion — CONFIRMED,
  and tighter than the 1e-15 the charter asked me to verify.
- **cp135 does not.** Its seg+pose is **+3.2e-5** higher and its bar **48.058 B** lower.
  Mechanism, MEASURED at source: `ddm_mc36_dual_axis_t4_verdict_20260814.md` records the
  MC36 seg leg as **−37 flips = −3.136529e-5 S**. `cp135 → MC36` was a distortion move, so
  it is precisely the step the pure-rate caveat excludes.

`fb1` wrote "re-derived off all four bases; identical to four decimal places from every
one of them". That is FALSE at cp135, and the counter-example sits inside the lineage
`fb1` itself named. I register the corrected scope — the frozen-distortion sub-lineage —
with cp135 embedded as the worked counter-example in the anchor.

### B.3 The caveat, encoded rather than mentioned

`domain_of_validity.does_not_apply_to` names d_seg, d_pose and "re-measure", and a test
asserts those tokens are present — so a future edit that quietly drops the caveat fails.
`D = 37,545,489` is an INPUT, not a constant of nature (Catalog #812: evaluate.py sums
`rglob('*')` over `videos/`); passing a different denominator is supported and tested.

### B.4 A smaller inconsistency I found and did NOT resolve

The widely-quoted MC36 S `0.1619344578804448` and the value implied by cp135 plus MC36's
own measured net ΔS (`−1.99799e-5` ⇒ `0.16193515837824176`) disagree by **7.005e-7 S**.
That is 0.07× the 1e-5 naming bar, so it moves no verdict and I did not chase it — but it
is recorded in the anchor, because a future arm computing a tight MC36-based seg claim
must go to the primary T4 receipt rather than to either composed literal.

---

## C. Two things I found on the way and did NOT fix

1. **`section_arm_next_if_resumed` is not printed in the live digest at all today.**
   `tools/costate_digest.py::build_digest` calls it inside the `else:` branch — the
   historical fallback used only when no complete live DDM source fleet is available.
   With the DDM fleet live (today), the arm-queue plan surface and my retraction-debt
   line are both invisible at SessionStart. The retraction filter is correct and tested;
   its OUTPUT just does not currently reach the operator through that path. The fix is one
   line in the `ddm_live` branch, but changing what SessionStart prints for every session
   is an operator-facing change outside an apparatus arm's charter, so I flag it rather
   than make it. `codex_arm_queue.py next` reaches the same data with zero risk.
2. **Three `tools/tests/test_modal_endpoint_close.py` tests fail, PRE-EXISTING.** I proved
   it by stashing only my own files and re-running at HEAD: the identical three fail. Cause
   is a stale `lane_ac1_test` active dispatch claim (age 60.5 h) in live state, not code.
   Clearing it needs a terminal `stale_*` row from whoever owns that claim.

---

## D. Adversarial self-review — where this unit could be wrong

1. **`AMEND_REQUIRED` is a judgement call I made, not a measurement.** I split the
   dispositions because `wd2`'s row has three clauses and only one is stale. A reviewer
   could argue any stale clause should hide the whole row. My reason: hiding a row with
   two live follow-ons to suppress one stale clause is the silent-drop disease. But the
   cost is real — an `AMEND_REQUIRED` row is still served, so a careless reader can still
   consume the stale clause. The stamp is the only thing standing between them.
   **Falsifier:** an arm that consumes the `15,157 B` clause from row 243 after this
   landing. That would prove the disposition should have been SUPERSEDED.
2. **Auto-retraction is scoped to `(arm, source_path, line_start)`.** A correction that
   MOVES the block to a different line, or lands in a new timestamped
   `arm_final_messages/` file, will NOT auto-retract its ancestor. That is deliberate —
   inferring supersession across files would retract genuinely distinct plans — but it
   means the mechanical half covers in-place corrections only. Cross-file supersession
   still needs a hand-filed retraction. **Falsifier:** a corrected memo landing as a new
   timestamped file whose ancestor stays live; I expect this to happen and it is the known
   gap, not a surprise.
3. **I retracted four rows on ONE arm's evidence plus my own re-derivation.** I
   independently reproduced the stale bar in rows 226/227/235 and the live pointer values,
   so the SUPERSEDED calls rest on primary artifacts. But I did NOT verify that the rx1 /
   rx2 / sp2 chains are still wanted at all — `fb1` raised the same doubt. If MAIN has
   closed the RX2 chain, the correct action was to close the rows, not retract them.
   Retraction is the weaker, reversible move, which is why I chose it.
4. **My reader sweep is a text sweep.** I matched the store's path string and the
   `next_if_resumed` identifier across `tools`, `src`, `scripts`, `experiments`, `.claude`.
   A reader that builds the path by concatenation, or reaches it through a config value,
   would be invisible to it. **Falsifier:** any consumer of that JSONL not in my A.4 table.
5. **The `48.058 B` cp135 offset depends on the quoted cp135 and MC36 composed S values.**
   §B.4 shows those two literals are themselves mutually inconsistent by 7.005e-7 S. The
   cp135 offset is ~68,000× that inconsistency, so the sign and scale are safe, but the
   last digits of `168,297.5395` inherit it. The FROZEN-lineage bar `168,345.5977` does
   not — it agrees across three independently published rows.
6. **The channel does not stop a stale bar being WRITTEN.** It only lets a correction
   reach the reader afterwards. The write-side cure — fire orders calling
   `pure_rate_byte_bar_from_pointer()` instead of typing a number — is now cheap (the law
   is registered and tested) but is not enforced anywhere. That is the deferred item my
   charter told me not to open, and this landing lowers its cost without closing it.

---

## NEXT_IF_RESUMED

- **`QUEUED-WITH-A-FIRE-ORDER`** — owner: **MAIN**. Fire trigger: **before any RX2-chain
  candidate is adjudicated.** Action: decide whether the rx1 / rx2 / sp2 chains are still
  wanted. If yes, re-file their fire orders against
  `tac.canonical_equations.sub015_pure_rate_archive_byte_bar_20260816.pure_rate_byte_bar_from_pointer()`
  and drop the retraction by writing the corrected block into the source memo (which now
  auto-retracts the stale row). If no, mark the chains closed. The rows are retracted, so
  nothing fires off them meanwhile. **$0.**
- **`QUEUED-WITH-A-FIRE-ORDER`** — owner: **MAIN or a costate-owning arm**. Consumer
  surface: `tools/costate_digest.py::build_digest`, the `ddm_live` branch. Fire trigger:
  the next time SessionStart output is being curated. Action: add
  `section_arm_next_if_resumed()` to the `ddm_live` branch so the plan surface and its
  retraction debt are visible in live mode (§C.1). One line; I did not take it because it
  changes operator-facing SessionStart output. **$0.**
- **`QUEUED-WITH-A-FIRE-ORDER`** — owner: whoever owns the `lane_ac1_test` dispatch claim.
  Fire trigger: immediately — it fails three tests on every run. Action: append a terminal
  `stale_*` row for `lane_ac1_test` / `modal:ac1-test` (§C.2). **$0.**
- **`DEFERRED, blocker named`** — owner: an apparatus arm, not this one. Make fire orders
  READ the pointer at fire time rather than copy a snapshot. Blocker unchanged from `fb1`:
  Catalog #299 gate quota plus an undecided gate-vs-lint call. **This landing lowers the
  cost** — the derivation is registered, tested, and callable, so the remaining work is the
  enforcement surface, not the arithmetic. **$0.**
- **`DEFERRED, blocker named`** — owner: a future queue-hygiene arm. Cross-file
  supersession (§D.2): a corrected memo landing as a NEW timestamped
  `arm_final_messages/` file does not auto-retract its ancestor. Blocker: any
  cross-file rule risks retracting genuinely distinct plans, so it needs a measured
  decision on what "the same plan" means, not a guess. **$0.**

**Own-vehicle frontier: hv1 ep0634 `S = 0.15959729295498598 @ 182,759 B`
`[contest-CUDA T4 n600]` — UNMOVED. This unit is apparatus and moved no score.**
