# FULL-STACK SUBMISSION DESCRIPTION — what ships, whose each part is, and the recursive verification pass (operator-requested 2026-08-31; #1363 draft-input at the #1111 boundary)

Date: 2026-08-31 · Author: MAIN · `score_claim` only where axis-labeled · Sources verified THIS
turn: `BORROWED_SUBSTRATE_ACCOUNTING.md` gen-6 (append-only, read §§8–10) ·
`CONTRIBUTION_ETIQUETTE.md` (retained Yousfi comment corpus) · `GENERATION_LOG.md` tail · hot
state pointer chain · `ddm_llm_policy_intake_20260831.md`. This memo DESCRIBES; the operator
writes all public text (contest LLM policy, 2026-08-31).

## 1. The vehicle at one glance

The archive decodes in three stages: (1) a compressed-model container unpacks the **semantic
renderer** (tokens → RGB frames) and the **pose carrier** (int12 coefficient lattice → PoseNet
motion); (2) the **HPAC probability model + RC64 range coder** decode the semantic **token
stream** (the per-position class field, 117,964,800 tokens over 600 pairs); (3) our
**receiver/assembly layer** applies container un-transforms (plane2, tile48×groupbin8),
the decode-time probability corrector, the compensation blob, and the pose-carrier splice,
then renders the 1,200 frames the frozen scorers see. Live pointer body: **afr1, 180,002 B,
S 0.14797617125559104 [contest-CUDA T4 n600]**, sha `cbb8d928…d405bf25` — below the public
leaderboard best (PR #135 at 0.162) on the same axis.

## 2. Lineage (every link an archive we hold or measured)

contest frame (videos, scorers, evaluate.py — nobody's contribution) → **PR #130
`semantic-pose-HPAC_CPR1`, Fesal Fayed** (the base learned vehicle, 191,052 B) → **PR #133
`cpr1_cbq_matched8`, JasonMo123** (transitively via #135) → **PR #135
`semantic-pose-HPAC_CPR1_polished`, Shreyan Mohanty** (0.162, rank 1) → **our 23 admitted
pointer moves** (08-14 → 08-31): micro-edit/compensation era (MC36 → keep01) → composed
candidates (sz1, ck1/ck2 plane2) → model-axis (ma1/to1 tail-override) → pose solves (up2/up3,
0 B) → joint waterfill (jg5) → rc2 (frozen packet candidate) → coder/container collection era
(gb1 native-identity, jt22/jt23, lb1 patch192×bank joint re-encode, afr1 tile48×groupbin8).

## 3. The four-class accounting (the gen-6 table, per archive section)

**Theirs (`inherited-substrate`, attribution mandatory, no originality claimed):** the
compressed model container · the HPAC probability-model ARCHITECTURE · the residual payload +
table codes · the RC64 range-coder backend (both encoder side and shipped receiver side).

**Theirs-idea / our-implementation (`mechanism-adopt-with-attribution`):** the semantic
renderer STATE (our format over PR #135's values; values lossily changed since gen-4 — the
earlier "byte-identical after decode" claim was WITHDRAWN when it stopped being true) · the
pose carrier STATE (their solver form, our binding, their lattice re-solved — 6,713 of 7,200
coordinates) · the compensation blob (edit-then-recompensate is PR #135's pattern) · the RC64
token stream (our model-axis work over their coder).

**Ours (`ours-original`, each with a receipt):** the receiver binding / assembly / custody
layer · the end-to-end compression entry point (`ddm_pq2_compress_e2e.py`; VERIFIED label
scoped to gen-3 bytes, honestly not carried forward) · the **joint admission waterfill**
(admit a seg edit only if it pays for the pose it costs; carrier re-solved against the
candidate's own edited renders — 0 counted bytes, it decides values inside existing sections)
· the retrained **HPAC probability object** (their architecture, retrained here on OUR label
field) · the build-chain instrument suite (jg2 exact tail re-encoder; edit-cost superposition
law; plane2 container transform −657 B; to1 tail-override −105 B; ma1 within-miss corrector —
the shipped `runtime/free_corrector.py`; up2/br1 pose GN solves, 0 B, 0 pairs worsened; up3
un-interleave + Rice splice; jg4 checkpoint fix) · the custody apparatus (candidate seal,
canonical score arithmetic, packet stager + census guard, dual-axis materializer) · and the
five post-freeze moves rc2→afr1, ALL of this class: lossless container/coder re-encodes and
the in-compile Schur pose compensation, no new learned artifact, distortion bit-identical.

**The one-line honest summary (gen-6 verbatim in spirit):** the learned VEHICLE is
PR 130/135's and we do not claim it; what is ours is the DECISION LAYER over it — which
values to perturb, how to admit edits jointly against pose damage, how to re-encode and
re-lay-out the sections losslessly — plus the retrained probability object and the
measurement/custody machinery that makes every number checkable rather than believed.

**Prior-art disclosures we carry:** PR #138 `opal_v1` published the decode-time-corrector
mechanism CLASS first (our packet states CONCURRENT INDEPENDENT DEVELOPMENT, design receipt
07-22, no priority claim); edit-then-recompensate is PR #135's. The level-set/task-space
witness line (our largest original research object) is named in the ledger as
RESEARCH-ONLY — it is NOT in these bytes and never byte-closed below the pointer.

## 4. Recursive verification pass (what I traced, and what it caught)

(a) Accounting doc read at its LIVE section (gen-6) per its own append-only banners — the
doc's history itself records two headline-vs-body incidents and resolves them; (b) ancestry
chain cross-checked against the hot-state 23-move ledger — the 5 post-freeze moves are all
`ours-original`-class lossless mechanisms, so the borrow/own boundary is UNCHANGED since
gen-6 while the BYTES changed (packet re-swap rc2→afr1 + accounting §11 amendment are OWED at
the submit boundary, now recorded in #1111); (c) the claim arithmetic recomputed from
components (S = seg 0.020139 + pose 0.0079812 + rate 0.1198559 ✓); (d) axis honesty: CUDA is
the claimed axis; contest-CPU on the gen-5 bytes measured INFEASIBLE (4,369.6 s vs 1,800 s
wall) and afr1's CPU leg is RECORD-WITH-REASON, no CPU score claimed; (e) the pv1 provenance
audit's 26-claim sweep already forced the two corrections against us (byte-identity
withdrawal; `ours-original` residual label withdrawal) — both survive in the live table.

## 5. The manner Yousfi's record demands (retained receipts, CONTRIBUTION_ETIQUETTE.md)

Follow the template literally; answer competitive-or-new PLAINLY and narrowly (PR #108 closed
for established-tricks-without-novelty; PR #110 asked for an easy-to-understand answer) —
ours: "competitive: measured below the leaderboard best on the CUDA axis; new mechanisms:
the joint admission waterfill + lossless container/coder collection layer, everything else
credited per-section"; host archive.zip OUTSIDE the repo; credit lineage at verifiable depth
(the per-section table IS this); keep axes separate with exact hardware; keep the body ~15
lines with detail linked; ONE PR, no serial submissions, no private channels; and NEW
(2026-08-31): the operator personally writes the PR description and all public comments —
this memo is input material only.

## 6. Consumers

#1111 (submit boundary: re-swap + accounting §11 + compliance re-buy + this memo + the policy
memo as REQUIRED inputs) · #1363 (draft-input obligation: SATISFIED by this memo at the
input-material level) · the operator's own final text.
