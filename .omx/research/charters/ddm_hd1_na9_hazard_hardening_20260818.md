# ddm_hd1 — fix + self-protect + automate + dynamic + recursively harden the na9 hazards

**Operator binding 2026-08-18 verbatim: "Must fix and self protect against and make automated and
dynamic and recursively hardened and polished."** Consumes na9's memo
(.omx/research/ddm_na9_gestalt_negative_audit_20260818.md) — read it FIRST, plus the
freeze-and-constrain memory (~/.claude/.../memory/freeze_and_constrain_through_engineering_20260818.md)
and the dy1 scope-law (derived-at-consumption is the default config semantic). Two-landing law
binds every fix: the cure + the gate that refuses re-introduction; m50 binds every gate:
EXECUTED positive controls in BOTH directions (red on the disease, green on the cure).

Standing laws: NO Modal/paid · NO scorer runs · NO Metal/MLX-GPU · upstream/ READ-ONLY ·
serializer commits w/ POST-EDIT sha · .py = 2 review-tracker passes, NEVER REVIEW_GATE_OVERRIDE
on .py · NO AI attribution · .venv/bin/python (bare python not on PATH).

## The four fixes (ranked by liveness)

H1 — **LIVE HAZARD: the latched receiver-pin literal in experiments/ddm_pq2_compress_e2e.py.**
na9 verified the r3 cure is present + fail-closed BUT hardcoded to rr4's archive — it refuses
fx1's (65c75d7f…, 180,601 B) and sa1's (67422cf0…, 178,272 B) candidates by construction.
FIX (dynamic): derive the pin AT COMPOSE TIME from the candidate archive being staged — the
invariant becomes "the staged runtime's inflate.py ARCHIVE_SHA256/ARCHIVE_BYTES constants MUST
match the staged archive" (a consistency constraint, not a latched value), re-pinning the
constants as part of staging exactly as MAIN did by hand for t1h r4. SELF-PROTECT: a test that
(a) REFUSES a tree whose pin mismatches its archive, (b) PASSES a correctly re-pinned non-rr4
tree (use the retained fx1 candidate_runtime as the real fixture). This is the first brick of
the #1115 seal contract — build it as the seal validator's pin-consistency check so #1115
consumes it, not a parallel twin.

H2 — **The corrections index date horizon (na9 F6).** Max source date 08-05, ZERO after — blind
to the live vehicle while reading as comprehensive. FIX (automated+dynamic): find why ingestion
stopped (root-cause, not just re-run), re-run the indexer over the full corpus, and add a
FRESHNESS BANNER to every query output (max_source_date + source count printed unconditionally —
silence made loud) + a warn when max_source_date < now−48h. RECURSIVE: the freshness check
itself gets a control (feed it a stale fixture → must warn).

H3 — **Closure-names-wrong-object (na9 F5, 4 instances, 4th cost a live win).** FIX AT SOURCE:
narrow dc1's "coder axis is CLOSED (FAMILY)" to what was measured ("coder-SWAP at FIXED
probabilities closed; the probability-model axis was NOT measured — fx1 moved it −560 B") in
the dc1 memo (append-only correction block) + nx1's promotion row + the hot-state sentence
"POST-HOC BYTE SURGERY ON THIS ARCHIVE IS EXHAUSTED" scoped to the lossless axis ra2 measured
(edit via tools/main_hot_state.py). SELF-PROTECT (build): extend the landed headline-vs-body
detector (au1's instrument) with a SCOPE-WORD pass — flag FAMILY/CLOSED/EXHAUSTED/DEAD sentences
whose own evidence rows name a narrower object; run it over the live corpus once, report count
w/ honest denominators; warn-only.

H4 — **na9's cheap retained-signal items:** (a) the 600×6 per-pair pose array retention at the
eval harness (~14 KB, $0 — wire it so the NEXT eval keeps it; ALWAYS KEEP THE PAYLOAD);
(b) the ensemble-calibrated-falsifier law from the LR-rung finding (stochastic endpoints need
seed-ensemble bands) — register as a canonical anti-pattern row if absent.

## Deliverables
1. H1-H4 landed w/ executed both-direction controls per gate (show the control outputs in the memo).
2. Memo .omx/research/ddm_hd1_na9_hazard_hardening_20260818.md — per-fix {cure commit · gate ·
   controls executed · what remains}; STORES CONSULTED; honest partial if any fix exceeds scope.
3. Final message: per-fix status + anything MAIN must adjudicate. End with the frontier line
   (S 0.15853325034789678 @ 181,161 B [contest-CUDA T4 n600]) + your unit did not move it.

## PRIOR-LAW PREDICTION (charter-time)
H1 is a ~50-line staging change + 2 controls (the fx1 fixture makes it real); H2's root cause is
most likely an ingestion path/glob that a directory rename orphaned around 08-05; H3's scope-word
pass will flag 10-40 sentences of which most are honest (the instrument must report the
denominator, not assume disease); H4a is a ≤10-line wire. If H2's root cause is instead a data
dependency that CANNOT be re-run cheaply, say so and land the freshness banner alone — the
banner is the self-protection; the backfill is the repair.
