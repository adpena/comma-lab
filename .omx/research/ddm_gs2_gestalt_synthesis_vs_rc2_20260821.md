# ddm_gs2 — Gestalt synthesis: negatives × week-signal × equations × lineages vs the rc2 frontier

Date: 2026-08-21 · Author: MAIN (Fable) · Inputs: ddm_na11 (378246c832) · ddm_pw2 (602ad5c372) ·
ddm_eq1 (af0eb94819) · MAIN pointer-receipt decomposition · sr2f de-blur sign row IN FLIGHT.
Baseline everywhere: [baseline:rc2=0.14827847122030852 @ 180,456 B, contest-CUDA T4 n600,
archive df7fd266…]. STORES CONSULTED: the three arm memos (sha-committed), canonical pointer,
canonical_task_status.jsonl, retained receipts cited therein.

## 0. The operating point (MEASURED, recomputed from components)

rate 0.120158 (81.0%) · seg 0.020139 (13.6%) · pose 0.007981 (5.4%).
Exchange rates AT rc2: 1,000 B = 6.66e-4 S · 1 seg flip = 8.48e-7 S ≈ 1.27 B · pose 1e-7 = 94 B-eq.
Pose marginal is now 6.26× seg's per unit distortion (up from 2.71× at PR106) — but pose ROUTING
is retired: pose can close at most 43.7% of the gap to 0.13 alone. Floor band 0.07–0.13; zeroing
BOTH distortion legs lands S = 0.120158 — inside the band's top. 0.13 is reachable from here
without touching rate representation; 0.07 is not.

## 1. Convergent patterns (each held by ≥2 independent arms)

**P1 — Compose ≠ stack; joint solves beat splices ~80×.** Move ladder: micro/splice moves land
2e-5…1.3e-4 each; the jg three-way {edit,drop,keep} JOINT waterfill landed −8.1e-3 in one step.
pw2's matrix receipts: jg4's naive merge of two finished candidates scored 0.3192 while jg5's
joint admission scored 0.14838; mc36 beat its own naive union 3.705×. na11 independently: the
qs1/qs5 re-prices fold into the live joint solve as PROPOSAL CLASSES, never a parallel chain.

**P2 — Additive correction is dead at this flip density; the live mechanism classes are
subtractive · conditioning · joint-solve · zero-byte decode-side operators.** ET5 measured
84.48 B/flip vs the 1.27 B/flip waterline (66× over). The seg-edit actuator is at its measured
marginal stopping point (0.662 value/cost, 8.91× degraded from its own set average) — no
re-pricing restarts it. Every closed family of the last 10 days (qs, sa1, SD1M, pk3/pk4, bands)
was additive.

**P3 — First-order models over-promise 5–10×; realization discounts are the norm.**
Representation levers realize 8.7–21.3% of their first-order models across four independent
mechanisms (fs2 8.72% · r2b 9.46% · compensated edit 10.5% · pz4r 21.27%). Toward-argmax coding
credits are 91.3% phantom. Compensation REACH at n=454 cancels 99.725% (21.8× worse than the n=3
figure). Foreign-body projections measured on the live body flip sign (#869: −113,555 B claimed →
+9.0e-3 measured). RANKING RULE going forward: divide every unrealized first-order claim by ~5–10
before it enters a queue.

**P4 — Laws age faster than the frontier moves; the same precondition can move rows in OPPOSITE
directions.** Pose-compensation PRICE collapsed 129× (reopening compensation-dominated refusals)
while compensation REACH degraded 21.8× (killing reliance on it) — net ≈ zero. Three laws retired
from citation (pose second wall SATISFIED · witness flicker floor lineage-scoped · HWM ceiling
26× stale). Four "BANKED" headlines were stale (qs2/re1 already consumed by mc36 · t1h REFUSED,
anti-transfers 6.31× · mz2 rungs refused). Every operating-point-dependent allocation law must be
re-derived at the current point or it poisons the queue — the eu4 pose-first doctrine is the
worked example (correct then, retired for routing now).

**P5 — The remaining mass sits in exactly three named places.**
(a) **The manufactured-seg pool**: 95.9% of the seg debt (22,783 round-trip flips) is
render/re-segment loss, not label error. Two direct attacks exist and neither has a realized
number: the zero-byte de-blur operator (sr2f LIVE, ceiling −0.019, sign unmeasured) and the ec1
edge-conditioned CONDITIONING adjustment (ceiling −0.019 net of rate, AUROC 0.9957 measured,
price 1,707 B on the CP135 body, realized Δd_seg NEVER measured — designed 08-14, never got a slot).
Seg is the only axis whose ceiling (0.0201) exceeds the gap to the floor band (0.0183).
(b) **Rate REPRESENTATION**: coding is measured CLOSED (−5 B headroom across 4 mechanisms × 4
sections); representation is OPEN but discount-heavy — best unfired (carrier shave) is −0.0148
first-order ≈ −0.0031 realized-discounted.
(c) **The CPU-axis GT fork**: 100.02% of the CPU-vs-CUDA gap is the GT decoder lineage; a joint
(DALI ∥ PyAV) carrier re-solve is worth up to −0.030 on the PUBLIC-LEADERBOARD axis; machinery
exists at 0.0991 B/pair; gated on the cd1 corrector port (CPU decode wall 3,037.6 s over).

## 2. Patterns of patterns

**M1 — One doctrine unifies every win of the sixteen moves: FREE STRUCTURE FIRST, JOINT SOLVE
SECOND, COUNTED BYTES LAST.** The native port (free wall-clock), the CPR1 rider (free bytes), the
de-blur operator (free, rule-118), the jg waterfill (joint), the Schur compensation (in-compile,
joint). The losing shape, closed at FAMILY scope again and again, is counted bytes spent on
additive corrections against sparse targets. This is the sparse×learned-prior law, the
placement-beats-amount law (26×), and the one-waterfill law converging into one sentence.

**M2 — The binding constraint has migrated from physics to PLUMBING.** The week's largest
residual deltas are blocked by: a hardcoded array index (pose compose tool), a training slot
never granted (ec1), a dead pid nobody re-checked for 5 days (sr1 FO-1), a favourable reopen
audited twice and fired never (cb1). The audits' top yields were not new mathematics — they were
sealed $0 rows sitting unfired. The follow-ons-fire-at-harvest law, measured at corpus scale.

**M3 — Verification is asymmetric: negatives strengthen, positives deflate.** 17 of 23 negatives
STAND (5 more strongly) under sharper instruments; reopens yielded only 1e-5-class rows. The
corpus's residual value is NOT in mis-graded negatives — it is in never-measured cells (the two
seg attacks, the CPU-axis solve). Audit effort should chase unmeasured ceilings, not re-litigate
measured floors.

## 3. Ranked frontier-lowering queue (deduped, discounted, at rc2)

| # | opportunity | mech class | ΔS (honest) | cost | status/gate |
|---|---|---|---|---|---|
| 1 | sr1 FO-1 de-blur SIGN | free decode-side op | −1.9e-4…−1.9e-2 IF live; sign unknown | $0 | **sr2f RUNNING** |
| 2 | ec1 edge-conditioned seg | conditioning | ceiling −0.019 net; realized UNMEASURED | re-pin + training slot | prep arm firing now |
| 3 | fx2 19-member rebuild (na11 R2) | subtractive/free | −5.77e-5 | rebuild + $0.16 | decode-margin refusal dissolved by rc2's measured 323.5 s slack |
| 4 | pose re-solve on rc2 body | joint solve | ≤ ~7e-5, **CONTESTED** | small tool fix + $0.16 | eq1 vs pw2/na11 disagree whether rc2 already carries the up2 codes — resolve by receipt comparison BEFORE firing (registered) |
| 5 | joint (DALI∥PyAV) carrier solve | joint solve, CPU axis | −0.021…−0.030 [contest-CPU] | gated on cd1 corrector port | the public-leaderboard axis lever |
| 6 | cb1 hood arithmetic (na11 R1) | $0 arithmetic | −2.34e-5 | ~140 s | na11 registered ddm_na11_R1 |
| 7 | rate-representation pool | representation | ≈ −0.003 realized-discounted | varies | carrier shave first; #1162 wall lever |

qs1/qs5 re-prices (na11 R3): fold into the live cw1/F1 joint solve as proposal classes — never a
parallel candidate chain (P1).

## 4. Corrections of record (NO-FAKE / stale-headline)

1. MAIN's charter premise "TR1 line holds a 130,875 B body 49,581 B below rc2" is a CATEGORY
   ERROR — trainer-internal MPS telemetry proxy, no pose term, no archive; the lineage's real
   archive is 182,759 B, worse than rc2 on all three components; rc2 IS that lineage's
   descendant. `hpac_mc36_joint_descent_law_v1`'s "132,798 B floor" is a fit to the same proxy —
   never quote it as an archive floor.
2. Stale-BANKED rows corrected: qs2 + re1 CONSUMED by mc36 (event-id proven) · ra2 rider ≡ rr5,
   shipped in rc2 · #869 adaptive map REFUTED on the live body (+9.0e-3) · t1h REFUSED
   (anti-transfers 6.31×) · mz2 q3/q4 REFUSED. rc4's 99.9874% figure is refuted — do not quote.
3. na10 reopens ps1u and ps135b DOWNGRADED to STANDS on arithmetic (ps1u +3.60e-4 wrong-way even
   at pose=0).
4. Laws retired from citation: pose second wall (SATISFIED) · witness flicker floor
   (lineage-scoped) · HWM ΔS ceiling (26× stale). CLAUDE.md's seg-vs-pose marginal table owes the
   rc2 row (pose marginal 6.26× seg, operating-point-dependent).
5. Registry hygiene owed: 27/449 canonical equations have no resolvable code consumer (17
   rate-axis); ledger loader reports 2 permanently unreadable task rows (repair is an append).

## 5. What this changes

The audit did not find a hidden 1e-2 sitting in a drawer — it found that the two largest credible
ceilings (both ≈ −0.019, both on the manufactured-seg pool) have never had their realized number
measured, and that the campaign's own doctrine (free-structure → joint-solve → counted-bytes)
predicts exactly where the next structural leap lives: a JOINT solve that includes the de-blur
operator and/or conditioning as free decode-side structure, on the axis (seg) whose ceiling
exceeds the remaining gap to the floor band. Rate representation is the 0.07-era campaign; the
0.13-era campaign is seg-conditioning + the CPU-axis GT fork.

## 6. Errata (2026-08-21, post-ec2p — append-only)

Rank-2's premise is REFUTED by a recovered orphan: ec1's realized Δd_seg WAS measured (T4,
2026-08-14, full n600) and came back NET-NEGATIVE — −40,779 flips (12,075 fixed = 10.5× over
break-even; 52,854 collateral = 4.377× the fixes) [CP135 body, COMPONENT-ONLY, INSTANCE scope].
The measurement was orphaned (poller crash, no memo), so eq1 ranked it #1 on a falsified premise
— M2 (plumbing-not-physics) measured at the head of this memo's own queue. Price re-pinned
1,707 → 1,176 B on rc2 (−31.1%). ec1 re-ranks from "largest un-cashed claim" to
"collateral-suppression redesign" (collateral must fall 4.93× at constant gross; pose unmeasured).
The seg attack path sharpens M1: conditioning without collateral priced inside the objective
joins additive correction in the dead column; the live route is collateral-constrained JOINT
solves + the de-blur operator (sr2f sign still pending). See FEED-ec2p + ddm_ec2p memo.

## 7. Errata 2 (2026-08-21, post-sr2f — append-only)

Rank-1 is RESOLVED: the de-blur sign was measured 08-16 (a1s, CLOSED_NET_LOSS + FAMILY_CLOSED)
— the row this memo ranked #1 was ALREADY FIRED, its ledger row stale at pending. sr2f re-priced
the only reopening term on rc2: pose drift 8.70× incumbent = 1.967× break-even granting 100% seg
recovery → family CONFIRMED CLOSED (operator×scene property, vehicle-independent to 0.33%).
Combined with Errata 1: BOTH top-ranked cells were resolved-on-disk before this audit ranked
them — M2 (plumbing-not-physics) was not just a pattern IN the queue, it WAS the queue's top-2.
The manufactured-seg pool's only live route is now the render-side objective with pose AND
collateral priced inside (a1s FO-C ≡ ec1 redesign ≡ js8 joint family). Actionable head: fx2
rebuild × dx1 CABAC composed candidate; then the contested pose-codes receipt comparison.
