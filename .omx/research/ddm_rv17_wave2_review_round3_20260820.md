# ddm_rv17 — WAVE 2, ROUND 3: two MED findings, one of them against my own round-2 CLEAN; counter RESETS to 0/3

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · fresh-lens round over the unchanged wave-2 set.

## THE ANSWER, FIRST

**Counter resets to 0/3.** Two MED findings, from the two lenses that had not been applied: the
**consumer surface** and **my own round-2 output**.

**RV17-W2-F12 — my round-2 CLEAN over-credited the cure batch.** W2-F9 is not cured, and the erratum
says so in its own words: **"§3 STANDS IN FULL."** The 3.97× survives in ANSWER-FIRST (line 22), the
§3 table (line 104), and §7 (line 217). In round 2 I ran a *number-stability* check — "did any value
move under a relabel?" — and read `3.97 (5 occurrences)` as evidence that nothing had drifted. The
check I owed was *finding-closure*: is the finding **about** that number closed? It is not. I used
the right instrument for the wrong question, which is the round-18 lesson repeating.

**RV17-W2-F13 — W2-F3 is uncured at the surface where it does damage.** `MEMORY.md:84` still reads:

> `−log2p DIRECTION-DEPENDENT (fs2): 0.88× away / 0.09× TOWARD argmax — price token levers by REAL re-encode`

And **neither** the index line **nor** the topic memory file carries the range or jg5's own figure —
`grep -c '0.813\|0.773'` returns **0** in both. So the constant W2-F3 showed to be mis-sourced —
jg2's 3-pair, ~58-token number attributed to jg5's 8,654 tokens, and the top of a 0.773–0.877 range —
remains the **standing rule every future consumer reads**, stripped of the provenance that would let
them notice.

That is precisely the class fs1 fell into with `na10:562`: a constant consumed downstream without
its source sentence. The wave found that defect, and the wave's own cure left an instance of it live
one level out.

---

## LENS 1 — CONSUMER-SIDE — **FINDING (W2-F13)**

Asking *"who reads these numbers next, and would they mis-consume them the way fs1 mis-consumed
na10?"* is the question this batch most needed, and the answer is yes for the fs2 factors.

The memory index line is the highest-traffic consumer surface in this project — it is loaded every
session, and its entries are one-line hooks by design. A one-line hook carrying a point estimate
with no range and no provenance is a `na10:562` waiting to happen: the next arm to price a token
lever will read `0.88×`, and nothing in its path says *jg5 measured 0.813, jg2 measured 0.877 on 3
pairs, jg3 measured 0.773 and is `delta_trustworthy: false`.*

**Corroboration from the other side:** fs3 independently found the same contradiction *inside fs2's
own memo* — the 0.8133 its numbers produce sitting beside the 0.877 it prints. Two arms reaching the
same defect from opposite directions is the strongest evidence that this is structural rather than a
transcription slip.

**CURE:** restate the hook as the range with its spread — *"0.77–0.88× away (jg5 8,654 tok = 0.813;
jg2 3 pairs = 0.877), 0.09× toward"* — so a consumer cannot take the maximum as the rule. The
topic file should carry the three measurements and their weights.

## LENS 4 — MY OWN ROUND-2 OUTPUT — **FINDING (W2-F12)**

The over-credit was specific and worth naming precisely, because the mechanism is reusable:

| what I checked in round 2 | what it establishes | what it does **not** |
|---|---|---|
| `5.667 (7) · 3.97 (5) · 2.909 (3) · S (2)` unchanged | no number moved under a relabel | whether the **finding about** a number was closed |

W2-F10 (scope inflation) I verified as cured by inspecting the surfaces it named. W2-F9 I never
re-opened at all — I let a stability check stand in for a closure check. The erratum's own banner
would have told me: *"§3 STANDS IN FULL."*

**W2-F9 remains open on the merits.** 3.97 = cheapest ÷ **median** credit, a statement about the
median pair; the claim it headlines is a **blanket** move whose additive break-even is the **mean**
(2.909 B/pair → **1.95×**). The conclusion survives — fs3's 5.766 B/pair composition settles that
independently — but the published magnitude is the median-derived one in three places, and the
erratum declined to touch it.

## LENSES 2 AND 3 — not the deepest cut here, and I am saying so rather than padding

Retention I re-derived at length in rounds 1b/1c: fs2 **86/86** with zero mismatches, fs1's four
arrays matching the result JSON, em1's payload sha matching. Re-walking them this round would be a
re-walk, which the cadence explicitly excludes. Negative-space on em1 I flagged honestly in round 1
and it stands unchanged: **I verified the payload's identity and reality, not the 0/46 reachability
arithmetic** — that remains the one leg of em1's three-leg NOT-LIVE adjudication I have never
re-derived, and it should be before #1147's closure is cited as settled. I am carrying it forward
rather than converting it into a finding, because its status has not changed since I first recorded
it.

---

## COUNTER

**0 / 3 — reset.** Round 2's clean pass does not carry, and it should not: it was earned on a
narrower question than the one the round was for.

The pattern this round exposes is the one worth keeping. Wave 2's findings were all *"the
measurement is sound, the generalization is not."* Round 3 finds the same shape in the **cures**:
the fs1 erratum corrected the surfaces W2-F10 named and left W2-F9's number standing with an
explicit "STANDS IN FULL"; W2-F3's correction reached the memo and not the memory line. **A cure
scoped to the findings it was handed is not the same as a cure scoped to the defect** — and I made
the reviewer's version of the identical error by checking the numbers I had listed instead of the
findings I had raised.

Six of my own outputs have now been corrected in this campaign. This one I caught myself, which is
the only reason it is in this memo rather than the next one.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round3_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — gen6 frozen, #1111 operator-HELD.
