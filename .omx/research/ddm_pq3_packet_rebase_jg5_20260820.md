# ddm_pq3 — the submission packet is rebased onto the first sub-0.15 exact row

`date_utc: 2026-08-20` · `owner: ddm_pq3` ·
`axis: [contest-CUDA T4, n600] for the score; packet work is byte-exact and scorer-free` ·
`score_claim: false (this arm measured nothing; it re-custodied a measured row)` ·
`frontier_moved: false (the pointer already carried jg5 when this arm started)`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`.**
Unmoved by this arm — it was already there.

---

## THE ANSWER, FIRST

The prepared contest submission packet is rebased from the superseded ck1 candidate onto
**jg5, archive `f3bce5d2…` / 180,625 B, `[contest-CUDA T4, n600]` S = 0.14839100138338618** —
the first sub-0.15 row in this packet's history, and 0.0136 below the best ranked public
entry (PR #135, 0.162).

Compliance is **83 GREEN / 4 RED of 87** under strict `--contest-final`, the same red SET as
generation 4 (structural / CPU-axis / declared-dependency / operator-gated hosting). The
packet is **PREPARED, FROZEN-READY, NOT SUBMITTED**. No push, hosting, or PR action of any
kind was taken.

## THE CHARTER WAS WRONG ABOUT ITS OWN STARTING POINT, AND RECALL CAUGHT IT

The charter named the pq1/pq2 bundle as the rebase source. **It was three generations
stale.** A generation 4 (ck1, 177,182 B, S 0.15710198138050818) had been staged on
2026-08-19, and review rounds had run through **12**. Rebasing from pq1/pq2 would have
discarded the round 5–12 hardening — the census guard, the receipt-freshness law, the
sign-determinacy arithmetic fix, the network-dependency disclosure.

This is the m44 discipline paying for itself: never recall from working memory alone. The
real work was **generation 5**, not a rebase of generation 2.

## WHAT WAS MEASURED, NOT ASSERTED

Every identity claim in the packet was re-derived rather than copied:

| Claim | How it was proved |
|---|---|
| Archive is the scored archive | `f3bce5d2…` re-hashed from disk; 180,625 B |
| Staged tree IS the evaluated tree | 33/33 manifest rows re-hashed **after the copy**, and `runtime_tree_sha256` **re-derived from the staged rows** = `2103073d…` = the receipt's = the T4-asserted = the pointer's |
| The score | Recomputed from all three components; the evaluator's own `final_score` field says `0.15` |
| The one-line GPU flip is not free | Simulated the edit and measured the tree hash move `2103073d…` → `75a1aeef…` |

**Two shas that looked contradictory were not.** The candidate seal records runtime
`a5d23cee…` at 34 files; the pointer and the T4 row record `2103073d…` at 33. Reconciled by
reading the producer at source: the seal hashes a 34-file tree including `archive.zip`; the
manifest hash is a canonical-JSON digest over the 33 declared rows. 33 + archive = 34.
Different definitions, no conflict — but that is a thing to *check*, not to assume.

## FOUR FINDINGS THIS ARM SURFACED

### 1. The harvested authority receipts are Python `bytes` reprs on disk

`report.txt`, `contest_auth_eval.json`, `provenance.json` and three siblings all begin with
`b'` and carry `\n` as two literal characters. The harvest path persisted
`repr(payload_bytes)` rather than the payload.

The encoding is lossless, so the files were decoded and **each decode proved round-trip
exact** before use (`HARVEST_DECODE_RECEIPT.json`). The shipped `report.txt` is therefore
the evaluator's own 664 bytes, not a re-authoring. **But a consumer that reads
`contest_auth_eval.json` with `json.load` gets a `JSONDecodeError`, and one that ships
`report.txt` verbatim ships a Python repr into a public PR body.** This is a harvester
defect worth fixing upstream of the packet; it is routed to MAIN rather than patched here.

### 2. The one-line GPU-routing flip costs a T4 row — the variants are NOT symmetric

The charter asked for both routing variants "as a one-line flip". `inflate.sh` is row 8 of
the pinned 33-row runtime manifest, so **any** edit moves the manifest-derived tree hash.
Measured: `2103073d…` → `75a1aeef…`. The 0.14839100138338618 row does not apply to the
edited tree.

So variant (b) (current/auto, staged) costs zero, and variant (a) (gpu-required-explicit)
costs a **new T4 exact-eval row plus a full re-stage**. Delivering them as a symmetric flip
would have been the convenient answer rather than the true one. Full pricing in
`GPU_ROUTING_VARIANTS.md`; the decision is reserved to the operator.

This is the round-11 F2(a) lesson in its original form: editing manifest-pinned bytes ships
what the exact evaluation never evaluated. The lineage has already paid for that lesson once
— the prior candidate scored **79.40** under one receiver tree and **0.157** under another
on byte-identical archive bytes.

### 3. My own reader nearly produced a false catastrophic result

Parsing the compliance receipt for an `ok` key — which the rows do not have; they carry
`passed` — scored the packet **0 GREEN / 87 RED**. Every "red" had detail `None`, which is
what gave it away. The true result was 79/87 at that moment.

The checker also writes its JSON and exits `1` in strict mode with **no stdout at all**, so
an empty log reads like a crash. Both facts are now recorded in `COMPLIANCE_RUNBOOK.md` so
the next reader does not raise the same false alarm. Same genus as the jg4 advisory-gate
false refusal: **verify the instrument before believing its verdict.**

### 4. AppleDouble contamination recurred — and the cure is an ORDERING law

Staging produced a clean tree. Writing the four public docs onto the ExFAT volume then
caused macOS to create **51** `._*` sidecars across the generations tree, which
`packet_census_guard.py` caught with exact paths.

The structural half is already right: the new stager selects files **by the manifest**, so
the 27 `.pyc` and 2 sidecars sitting in the *source* tree could not enter the packet by
construction, and are reported with exact paths rather than silently dropped. What was
missing is ordering: **purge, then census, then buy the receipt, with no writes in between.**
Recorded in the runbook.

## WHAT I BUILT RATHER THAN SCRIPTED

`tools/stage_contest_submission_packet.py` — generations 2, 3, 4 and 5 were each staged by a
fresh ad-hoc script, and generation 4's arm recorded its own defect: its census *"filtered
AppleDouble out of BOTH sides of its comparison … a check that excludes a file class cannot
certify it."* Fourth repetition, second contamination incident. Per the canonical-chain rule
and the least-hand-typing law, staging is now one reviewed tool with the invariants wired in:
manifest-driven copy, re-hash after copy, tree hash re-derived from the staged rows, census
with its denominator, fail-closed with the output removed on any mismatch.

**It has no tests. That is owed and named**, and round 13 is told to re-run its proof
independently rather than trust it.

## COMPLIANCE: FOUR REDS CURED AT SOURCE, FOUR REMAIN

The first receipt measured **79/87 with 8 reds**. Four were fixed rather than argued away:
the public-source repo link (dropped when I rewrote the PR body) was restored, and three
dispatch-claim rows were cured by appending one conforming terminal claim binding **both**
full 64-character shas — the prior row carried an 8-character prefix and a status outside
the checker's accepted set.

That appended row turned three reds green, and **the arm that benefits from a green should
not be the only one who says so** — it is flagged for independent verification in round 13.

The four survivors are the generation-4 set, unchanged: the structural raw-promotion
blockers, the absent CPU row, the `inflate.sh:27` Brotli bootstrap under the e4 precedent,
and operator-gated hosting.

## THE HONEST STATE

The packet is ready to freeze and **is not submittable by any arm**. Five things stand
between it and publication, itemized in `FREEZE_CHECKLIST.md`: the operator's one-line
confirm, the GPU-routing decision, an optional wc2 corrector fold, an optional rr5 rider
fold, and a hosted archive URL. The review counter is **0 of 5** and this arm cannot clear
it, having staged the packet.

The largest open risk is **not** the score — that is measured on the exact submitted bytes.
It is the **evaluation-time budget**: inflation took 1,419.9 s and evaluation 51.4 s of a
1,800 s job wall, leaving 328.7 s for checkout, dependencies and download. Against the
derived CUDA residual window `[890.6, 1430.6] s` this fits only the optimistic end, by about
**10.7 s**. Graded **WARN, not PASS**, and disclosed in the report, the README and the PR
body rather than left for a judge to discover as a timeout.

On the CPU path the projection is **1,414–1,913 s against a `[1,044, 1,332] s` residual** —
over budget in every corner — and the prior lineage measured contest-CPU inflation at
3,422.7 s. No CPU row exists on these bytes and none is claimed.

### The rr5 rider: evaluated, and the default is DO NOT FOLD

The charter cited −1.85e-4 for the rider. rr5's own memo corrects that: ra2 measured
`+263 B raw / ~230 B realised` and labelled the realised half **PROVISIONAL**; a later memo
added an unreceipted 48 B leg to reach ~278 B ⇒ −1.85e-4, and downstream memos inherited the
sum as "MEASURED". The lossless claim is sound; the size was overstated — cross-regime
constant transfer, one memo downstream.

Decisive point regardless of size: **lossless does not mean free.** The rider changes archive
bytes, so even with decode-identity proven the resulting score would be **DERIVED**, not a
measured row — the NO-FAKE #8 surrogate trap. Default stands: ship the measured `f3bce5d2`
bytes.

## WHAT THIS UNIT DID NOT DO

It did not move the exact pointer. The sub-0.15 row was won by `ddm_jg5`; this arm
re-custodied it into a publishable, compliance-checked, honestly-attributed packet. Custody
is means, not end — but the end here is a submission, and a measured row that cannot be
published is not yet a result.
