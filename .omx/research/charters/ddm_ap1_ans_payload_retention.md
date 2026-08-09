# ddm_ap1 — The −2,120 B ANS win is a LENGTH, not a payload. Re-run and keep the bytes.

**Owner:** codex arm · scorer-free · `[macOS-CPU advisory]` · `score_claim=false` · no Modal

## THE DEFECT, OWNED

`ans_n600/ans_real_n600.py` measured `len(enc.get_compressed().tobytes())` and then **discarded the
bytes** — line 37 is literally `del enc`; line 41 has the same shape for the ANS arm. Found by
`ddm_rc1_receiver` (`5de03569ad`), recorded as OWED item 5 in
`.omx/research/ddm_pr130_reproduce_20260809/ANS_REAL_TABLE_MEASUREMENT.md`.

The measurement is REAL. The faithfulness control passed, and the range arm reproduced the shipped
token stream length **116,980 B exactly**. But **no ANS words were retained, so no archive can be
assembled from it.** The composed 188,029 B / `S=0.170128405876608123` is arithmetic over a measured
length, not a built object.

Measured, for your budget:

| quantity | value |
|---|---:|
| frames | 600 |
| ideal (cross-entropy of the real tables) | 114,851.8 B |
| range (PR130's shipped coder) | 116,980 B — **+1.8530%** over ideal |
| **ANS** | **114,860 B — +0.0071% over ideal** |
| Δ | **−2,120 B** → ΔS ≈ −0.0014123 |
| run time | 681 s |
| conditional tables | ~4.7 GB if materialized float32 |

## YOUR JOB

Re-run the n600 ANS encode **retaining the words**, so a real archive can be built from them.

1. **Retain the payload.** SSD atomic int16 chunks under
   `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/`, with a resume receipt (the run is 681 s and
   the tables are large — per the resumability non-negotiable, it must survive a crash).
2. **Round-trip it.** Decode the retained words back and prove the symbol sequence is identical to
   what was encoded. A retained payload that does not decode is worth nothing.
3. **Re-run the range arm too, retained**, and prove its serialized bytes equal the shipped
   116,980 B **byte-for-byte**, not just in length. The prior run proved LENGTH equality only and
   said so honestly; close that gap or state plainly that you could not.
4. **Then hand `cx2` a real object**: the retained ANS token stream + its sha256 + its exact byte
   count, at a path that survives.

Encode-side memory is the known hard part: ANS encodes backwards, so it wants the conditional
tables materialized where the current encoder streams them. `probability_table` returns float32
(2.197 GiB of tables + 471,859,200 B of int32 symbols). **The model first quantizes logits to
int16** — spilling those exact codes halves the table field to 1,179,648,000 B. Reverse-chunk
helpers are already built and tested; reuse them (internal-leverage authority: they are ours, use
them off the shelf, extend them if they do not fit).

## OPTIMAL FORM

Reference form: a full n600 encode with retained, round-trip-verified payload for BOTH coders, at
the pinned `constriction` 0.5.0 that the receiver already dispatches. Declared reductions: SCOPE
only — none planned; the run is already n600. A MECHANISM reduction (fewer frames, synthetic
tables, unpinned coder version) is a TOY BRACKET and cannot produce the object cx2 needs.

Provenance pins (verify each; a pin that does not reproduce is a STOP):
- the prior receipt `ans_n600/ans_vs_range_n600_result.json` under
  `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/`
- archive sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, 191,052 B
- the receiver's coder dispatch (`absent → legacy`, `r7_smevr_v1 → SMEVR`, anything else → REFUSED)
  landed at `5de03569ad`; the ANS/Range dispatch is already built — do not rebuild it

## HARD RULES

- Launch through `tools/launch_detached_process.py`. It now refuses (rc=5) argv matching the fleet
  reaper predicate — **the first attempt at this exact run died at frame 300/600 (337 s)** because
  it was launched from a scratchpad path containing `claude-501`, which the launchd reaper matches
  on a word boundary. Do not reintroduce that argv.
- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- `upstream/` immutable. The intake clone at `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/`
  is READ-ONLY — copy out to work.

## DELIVERABLE

A retained, round-trip-verified ANS token payload with path + sha256 + exact bytes, and the
range-arm byte-identity result. Say plainly which of the two byte-identity proofs you closed and
which you could not.

---

## ERRATUM — MAIN, 2026-08-09 (the charter author correcting his own false pin)

**The SMEVR selector pin in "Provenance pins" above is WRONG and is hereby WITHDRAWN.**

I wrote that the receiver's coder dispatch (`absent → legacy`, `r7_smevr_v1 → SMEVR`, anything else
→ REFUSED) "landed at `5de03569ad`". `ddm_ap1` checked at source and found **that selector does not
exist at that commit.** The charter's own STOP rule — "a pin that does not reproduce is a STOP" —
fired, and the arm was right to stop on it rather than proceed around it.

**The mechanism of my error:** that three-case dispatch is real, but it belongs to OUR OWN-VEHICLE
TR1 receiver (task #858, re-derived at source by `sv2` and pinned by a mutation-checked test). It is
NOT the PR130-reproduction receiver at `5de03569ad`. I carried the detail across from working memory
and attached it to the wrong object. This is the recall-before-decide law applied to charter
authorship: **verify every pin at source before writing it into a charter**, because an arm that
honours a false pin either stops (best case, what happened here) or builds on sand.

**What survives unchanged:** everything AP1 actually measured. The retained payloads, both identity
proofs, the constriction 0.5.0 pin, and the decode verification are independent of the false pin and
stand on their own receipts.

**Charter-completeness, stated honestly:** AP1 is NOT fully charter-complete. Beyond the withdrawn
pin, its producer lacked the governed launcher, whole-job resumability, and direct canonical-helper
reuse. The arm reported this instead of fabricating the receipts, which is the correct outcome —
those legs remain OWED, not done.

**Duplication note, also mine:** `ddm_dt1` necessarily encoded and retained an ANS payload in order
to measure ANS decode time, so this charter overlapped work already in flight. I fired it without
checking that. `dt1`'s retained pair is the cleaner source and is what `cx2` should consume.
