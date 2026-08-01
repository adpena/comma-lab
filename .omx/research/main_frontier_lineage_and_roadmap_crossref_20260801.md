# MAIN cross-reference: what produced our best score, what we built since, where the frontier is

`date_utc: 2026-08-01` · `author: MAIN (Opus 5)` · `pointer: 0.1910828242 [contest-CPU] UNMOVED` ·
`score_claim: false` · $0 read-only, **0 scorer forwards**. Operator 07-31: *"Read more of the files
and cross reference against the OMX DAG and record and findings and everything. And I need you to
really understand what produced our best score and all the work that we've done since then and what
the frontier is and what the roadmap and next steps are."*

Sources READ (not recalled): `.omx/state/canonical_frontier_pointer.json` ·
`.omx/state/continual_learning_posterior.json` · `src/tac/optimization/ddm_tr1_runtime.py` (full) ·
`src/tac/optimization/ddm_runtime_exporter.py` (header) ·
`.omx/research/ddm_eg1_tr1_rehearsal_20260728.json` ·
`.omx/research/ddm_ja1_order_of_operations_dag_20260731.json` ·
DAG `sub015_DAG_…_20260611.md` FEED index + FEED-cn3 in full.

---

## 1. WHAT PRODUCED OUR BEST SCORE — the honest answer is *borrowed substrate*

| row | S | bytes | rate | distortion | what it is |
|---|---:|---:|---:|---:|---|
| **best SUBMITTABLE** | **0.1910828242** | 177,169 | 0.11797 | 0.07311 | `lane_clickpolish_pr110_frontier_20260710`, sha `ad02b012…` — **PR110 borrowed substrate + OUR exact-score-gated latent click-polish (#399)** |
| local CPU bank | 0.1880443980 | 176,564 | 0.11757 | 0.07048 | `lane_pr128_click_import_**NONSUBMISSION**_defensive_bank`, sha `196acd18…` — NO-FAKE #7 forbids shipping as original |
| best CUDA | 0.2053300290 | 186,876 | 0.12443 | 0.08090 | PR106 lineage, **2026-05-16 — 2.5 months stale** |
| **THE BAR** | **0.172141** | 190,952 | 0.12715 | **0.04499** | PR130 `semantic-pose-HPAC_CPR1`, external |
| **OUR OWN VEHICLE** | 0.9639878 | 360,238 | 0.23987 | 0.72412 | v4d = TR1 token base + composed rungs `[macOS-CPU advisory]` |

**Every score we can point to came from a borrowed carrier.** Our own from-scratch vehicle is
**2.03× worse on rate and 9.9× worse on distortion** than our own borrowed best.

## 2. ⚡ NEW FINDING A — the gap to the bar on the BORROWED line is DISTORTION, not rate

Checkable by subtraction from numbers already in the pointer: **PR130 spends MORE bytes than our best
submittable (190,952 vs 177,169) and still wins**, because its distortion is 0.04499 vs our 0.07311.
Δrate **+0.00918** (we are cheaper); Δdistortion **−0.02812**; net −0.01894 = 0.172141 − 0.1910828. ✓

MEMORY.md's standing row *"07-27: gap=RATE"* is TRUE for the **v10 describe-line** (which solved
distortion to d_seg 1.52e-4 and could not fit the description in the box) and **FALSE for the
borrowed line**. Two lineages, opposite gap structures; the framing does not transfer.

## 3. ⚡ NEW FINDING B — the pointer has no submittable-vs-bank distinction

`our_local_frontier_contest_cpu` = **0.18804**, whose own `architecture_class` says
`…_NONSUBMISSION_defensive_bank_…`. Our best SUBMITTABLE (0.1910828242) **appears nowhere in the
pointer's structured fields** — it is recoverable only by scanning the posterior. So the declared SoT
for "our local frontier" holds a row we may not ship, and the row we *may* ship is not represented.
`effective_frontier` still computes correctly (min → 0.172 external), but any consumer asking
"what is our best shippable archive?" gets the wrong answer. **Schema gap, not a data error.**

## 4. ⚡ NEW FINDING C — the CUDA axis is 2.5 months stale

Last `[contest-CUDA]` row: 2026-05-16, PR106 lineage. Every candidate built since (click-polish,
PR128 import, the whole DDM/TR1 arc) has only ever been measured on CPU. Defensible — the contest
ranks CPU — but CLAUDE.md's *"Submission auth eval — BOTH CPU AND CUDA"* is **unmet for everything
current**, and would be a submission blocker the moment a row is ready.

## 5. WHAT WE BUILT SINCE — three lineages, not one

1. **v10 describe-line / inverse-solve** (07-19→07-24). SOLVED distortion (q1 d_seg **1.52e-4**, far
   under the 1.72e-3 need) but the *description* of the solution never fit the box. Gap = RATE.
   Plane-storage family measured RATE-DEAD.
2. **DDM describe/correct/descend** (07-23→07-28). Grammar induction · correction stacks · lattice
   solves · ms/rg/j chains. Exporter = `ddm_runtime_exporter.py` (DDM v15/J2: lane program,
   worldsheet G1, class templates). Converged onto "no correction strictly improves realized joint ΔS."
3. **TR1 trained partition→pixel renderer** (07-28→now) — THE live vehicle. Mechanism in
   `[[tr1-architecture-mechanism-read-from-the-receiver]]`: **tokens 99.0% of a 504,736 B archive**,
   LOTTO renderer 3,341 B, selector 535 B, pose_stub 83 B **INERT with frame0 = zeros**.
   v4d is a composed candidate on this token base (`ja1`: *"v4d-cheap (NO token-base change)"*).

## 6. THE FRONTIER — cn3's decisive arithmetic (independently re-derived here)

Banked pose fallback **0.12689** + best byte-closed rate ever built (Knee-B 174,578 B = **0.11624**)
= **0.24313 > bar 0.172141 with ZERO seg budget** → excess **+0.07099**.
**No composition of currently-banked components reaches the bar.** Excluded *a priori*, not merely
sub-optimal — this is the arithmetic behind the standing `NO AXIS PRIORITY` law.
Debts to gc13 corner C: **seg 0.371179 (7.19×) · pose 0.277677 (19.19×) · rate 0.153305 (2.77×)**.

Feasibility identity, free from non-negativity: **S ≥ 100·d_seg** ⇒ sub-0.172141 REQUIRES
**d_seg < 0.00172141**. Burn best 0.0038892 is 2.26× high; v4d's seg term alone (0.431179) already
exceeds the bar regardless of pose and rate.

## 7. THE ROADMAP — cn3's ranked next-3, all labelled MEANS, none crosses the bar

1. **FIRE BURN-2 at the window_03 boundary instead of a window_04.** Already BUILT+TESTED (4 commits
   `f28e427dd9` `e8d531e735` `4bdd72a2f7` `d138df0c00`), **never fired** — the largest signal loss of
   the week is a *build*, not a finding. With `bias_correction` explicitly declared it answers three
   pre-registered questions in one slot (seg capacity · gc15's one-field falsifier · gc14's R1 control).
   Falsifier with teeth: bc1 measured from-birth ep399 at d_seg 0.005169, **worse** than warm 0.004264.
2. **RE-ANCHOR ja1 at the v4d base ($0)** — its ranks 1–4 were all consumed by v4d, so every charter
   spawned since reads a table whose top four rows are spent. A stale *allocator* misleads every
   decision after it.
3. **TERMINAL POSE SOLVE**, routed CONTENT-limited (784× per-pair spread, 90% of mass in 88 pairs)
   ⇒ QA68 expert-menu (UNBUILT) outranks QA65 finer quanta.

**AND: window_03 has REVERSED** — trough ep879 0.0039510 → ep914 0.0043281, seven consecutive
monotone rises, OLS t = **+6.28** on gc14's own pre-registered window ⇒ B5-C fires, **window_04 must
NOT run**.

## 8. THE APPARATUS ROOT CAUSE — and it is the same shape three times over

cn3: three anti-orphan gates all report CLEAN **because each scans the wrong set** —
(i) `check_codex_findings_memos_consumed`: **0 of 1,260 files in scope** (mtime<3d filter);
(ii) `lever_registry.completeness()`: ASTs **one** file, ~180 `witness_dsl/*` modules invisible;
(iii) #396: correct scan set, **433 violations**, never strict-flipped.
⇒ **NEW LAW: a gate's LIVE-COUNT-0 is meaningless until its DENOMINATOR is asserted.**

**MAIN's cross-reference: this is the same defect I found independently as #842** — 502 of 502
`preflight_all` gate call sites sit inside `if check_codebase:` and are SKIPPED on a normal commit.
Same shape again in my own tooling twice today: a zsh `--include=*.py` glob error meant grep **never
ran**, and its empty output was indistinguishable from a clean negative. **Four instances, one law:
an empty result is not evidence until the command is proven to have run over a stated denominator.**
