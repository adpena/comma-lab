# ddm_m1r4b — M1 FIRE GATE, REVIEW PASS (round 4), LENS B: SCIENCE + OBJECTIVE FIT

You are one of THREE concurrent independent review passes over the same sealed M1 artifact.
Your lens is **SCIENCE**: is this the RIGHT burn — right objective, right schedule, right stopping
rule, right expected value — not merely a correctly-plumbed one. Sister arms carry the MECHANICS
lens and the ARITHMETIC/AUTHORITY lens.

## Why this round exists (state, verbatim)

The M1 n120 receiver fire is gated on a **3-consecutive-clean-pass** counter.
- Round 2 (`ddm_m1r2_20260808/M1R2_REVIEW.md`) returned `FINDINGS_RESET_COUNTER` with 2 findings.
- Round 3 (`ddm_m1r3_20260808/M1R3_REVIEW.md`) returned `CLEAN_PASS_3_OF_3` on the science questions.
- **Both round-2 findings have since been CURED and committed**; the artifact CHANGED, so round 3's
  clean verdict does NOT carry forward. The counter is at **0/3**. You are pass 1-of-3 of the fresh
  sequence.

**CONCURRENCY CAVEAT (binding):** all three passes review the SAME frozen state. If ANY pass reports
ANY finding, ALL THREE are void and the counter stays at 0. Do not soften a finding to protect the
sequence — a false CLEAN is far more expensive than a reset.

**READ ROUND 3 FIRST**, then deliberately try to break it. Round 3 cleared these science questions
once; your job is not to re-run its checklist but to find what a clean pass missed. A second clean
pass that merely agrees with the first is worth very little.

## The cures that changed the artifact under round 3

1. **M1R2-F1:** `one_sample_flip_S` was stale (derived from n=115) while the same dict carried the
   live n=120 inputs. Re-derived in place to `4.238552517361111e-06`; `marginal_bar_S_per_step`
   re-derived to `8.477105034722223e-08`. Commit `ff43f9dcf3`. **This changed the STOPPING RULE's
   threshold by ~4.2%** — which is squarely in your lens.
2. **M1R2-F2:** the fire-guard's probe/fire equivalence tuple was incomplete; cured structurally in
   `tools/mx1_fire_guard.py` (commit `1381ac84cb`). Mechanics lens owns the cure itself.

## Scope (pinned; re-verify these shas yourself before reading)

| file | sha256 (first 16) |
|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `90cf28d390999ef9` |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `8bad6a6b8be1b201` |
| `tools/mx1_fire_guard.py` | `60fc0501a65d8d09` |
| `tools/ddm_seal_orchestrator.py` | `11c4368f009afc31` |

**REVIEW SUBJECT (immutable).** The reviewed subject is the CONTENT of the four files above,
keyed by their sha256 — **NOT the live repository HEAD**. The subject's tree is that of commit
`393d67d016` (ROUND 5: the m1c1 amendment 393d67d016 is IN the subject). Live HEAD WILL advance during your pass (this charter's own commit, sister-arm
receipts, MAIN's own notes); that is EXPECTED and is NOT a finding. Descendant commits that touch
only charter files, review receipts, or MAIN's notes are OUTSIDE the reviewed subject by construction.

**The binding stop test:** re-hash the four files yourself. If any of the four CONTENT shas differs
from the table, STOP and report that as finding #1 — the subject moved under the review and the pass
is void. A differing live HEAD **alone is not such a mismatch and must not stop the pass**.

(Provenance-clause v2, repaired after M1R4A-F1: v1 pinned live HEAD, which a charter's own commit
necessarily invalidates — an unstable freeze key in a shared worktree.)

## Your questions (re-DERIVE; do not confirm)

**B1. The stopping rule now that its threshold moved.** The event predicate stops on marginal
ΔS/step falling below `marginal_bar_S_per_step`. Re-derive that bar from ITS OWN inputs (do not
copy the ticket's number). Then ask the harder questions: is a marginal-rate bar the right stopping
rule for a curve whose descent is known to be **event-punctuated, not smooth**? What happens if the
trajectory sits below the bar for a window and then re-accelerates (the plateau-then-drop shape
this vehicle has shown before)? Does the predicate's window/averaging make it robust to that, or
does it stop us on the flat part of a staircase? Read `evaluate_trajectory_stop` at source.

**B2. The safety cap is a cap, not a plan.** The ticket carries `3250` as a SAFETY CAP. If the burn
terminates at the cap rather than on the event, what does that mean scientifically — and does the
ticket say what we do in that case? An un-planned cap-exit is the censored-cap genus (a stop with
no typed receipt is a silent instrument).

**B3. Is n120 the right population?** The standing law is n≥120 STRATIFIED-RANDOM, never a prefix,
because a prefix of this clip is a temporally-correlated SCENE BLOCK: measured pose prefixes run
**2.5–4.2× HARDER** than the population while seg prefixes run **0.96× easier**, so a prefix biases
the two axes in OPPOSITE directions. Verify from the ticket + trainer source how the 120 pairs are
actually selected. If it is a prefix or a strided sample rather than a seeded stratified random
draw, that is a finding — and say which axis it biases and in which direction.

**B4. Objective fit.** What exactly does this burn optimize, and is that the quantity that moves the
composed score? Our measured gap decomposition against the PR130 bar is **seg 0.4010 (~69%) +
pose 0.0693 + rate ~0.1110**. If the burn's objective is not aimed at the dominant term, say so
plainly and quantify the expected-value mismatch. Beware the sunk-cost read: "the ticket is sealed"
is not an argument.

**B5. The lr, the EMA, and the schedule.** `lr = 2e-7` is SOURCE-VERIFIED from the PR130 repro
`train.sh:113` — but a constant verified in ANOTHER vehicle's batch geometry is a BORROWED CONSTANT
until re-derived at OUR batch geometry (constants-are-poison; cross-regime transfer is a named
recurring defect here). Does the ticket re-derive it, or transfer it? Same question for the EMA
decay (the canonical law is `ema_decay_run_geometry_v1` — decay follows from steps/epoch × horizon,
NOT a flat 0.997) and for any curriculum/stage boundary the config inherits.

**B6. The sigma-calibration protocol.** The ticket's seal includes a σ protocol (5× n120 5-step runs
+ an fp32 reference → `sanity_sigma_measured`, with 3 seal falsifiers). Is that protocol capable of
measuring what it claims — i.e. is 5 runs × 5 steps enough to separate real descent from run-to-run
noise at the bar in B1? Compute the implied resolution and compare it to the bar. If the noise floor
is at or above the bar, the stopping rule is unfalsifiable and that is a LAUNCH-BLOCKING finding.

**B7. What does this burn's failure look like?** Name the two or three most likely ways it produces
a result we cannot use (not "it crashes" — MECHANICS owns that; scientific unusability: confounded,
un-attributable, or measuring the instrument rather than the vehicle). For each, is there a cheap
pre-registration NOW that would make the outcome interpretable either way?

## Rules

- **Re-derive, don't confirm.** Quote file:line. `UNDETERMINED` is an acceptable answer with a named
  missing input; a guess is not.
- Label every number MEASURED / DERIVED / INFERRED / ASSUMED. Do not restate a ticket number as
  though you had checked it.
- **Never invent** CLI flags, API names, or VALUES.
- **Do NOT fire the burn.** Do NOT launch any Metal job. Do NOT modify the ticket or the trainer.
- Commit ONLY your own receipt, via `tools/subagent_commit_serializer.py` with POST-EDIT
  `--expected-content-sha256` per file, tags `[no-triality] [p0-ledger-ok]`, NO Claude/AI attribution
  and no Co-Authored-By trailer.

## Deliverable

Write `.omx/research/ddm_m1r4_20260808/M1R5B_REVIEW.md` with:

1. First line EXACTLY one of `VERDICT: CLEAN_PASS` / `VERDICT: FINDINGS_RESET_COUNTER`.
2. The sha re-verifications.
3. Per question B1–B7: what you derived, from which file:line, conclusion, and honesty label.
4. Per finding: severity, MECHANISM, failure scenario, smallest correct cure. Most-severe first.
5. An explicit statement of what round 3 checked that you deliberately did NOT re-check, and what
   you checked that round 3 did not.

## OPTIMAL FORM

- **Reference form:** a full-depth adversarial pre-launch science review that treats the sealed
  ticket as a hypothesis to falsify, at the bar of the round-2 pass that found the stale-constant
  defect inside a previously-cleared artifact.
- **Scope reductions (legal):** bounded to the pinned files and their transitive reads; no new
  measurement commissioned; σ-protocol adequacy assessed analytically, not by running it.
- **Mechanism reductions:** NONE. Checklist-confirming round 3 rather than attacking it is a silent
  mechanism reduction and invalidates the pass.
- **Provenance pins:** the four CONTENT shas above = the immutable subject; subject tree =
  commit `1381ac84cb`; live HEAD is explicitly NOT a pin (see provenance-clause v2).
