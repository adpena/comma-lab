# ddm_m1r4a — M1 FIRE GATE, REVIEW PASS (round 4), LENS A: MECHANICS + SEAL GATES

You are one of THREE concurrent independent review passes over the same sealed M1 artifact.
Your lens is **MECHANICS**: the config → argv → guard → preflight → resume → memory chain.
Two sister arms carry the SCIENCE lens and the ARITHMETIC/AUTHORITY lens. Do not duplicate
them; if you find something in their territory, report it anyway (a finding is a finding).

## Why this round exists (state, verbatim)

The M1 n120 receiver fire is gated on a **3-consecutive-clean-pass** counter.
- Round 2 (`ddm_m1r2_20260808/M1R2_REVIEW.md`) returned `FINDINGS_RESET_COUNTER` with 2 findings.
- Round 3 (`ddm_m1r3_20260808/M1R3_REVIEW.md`) returned `CLEAN_PASS_3_OF_3` on the science questions.
- **Both round-2 findings have since been CURED and committed** (see below). The counter is at **0/3**.
- You are pass 1-of-3 of the fresh sequence against the **cured** artifact.

**CONCURRENCY CAVEAT (binding):** all three passes review the SAME frozen state. If ANY pass
reports ANY finding, ALL THREE passes are void and the counter stays at 0 — a later pass cannot
inherit an earlier pass's clean verdict when the artifact changed underneath it. So: do not
soften a finding to protect the sequence. A false CLEAN is far more expensive than a reset.

## The cures you are reviewing (do not assume they are correct)

1. **M1R2-F1 — stale `one_sample_flip_S`.** The ticket's event predicate carried
   `one_sample_flip_S = 4.4228e-06`, derived from a stale n=115 population, while the SAME dict
   carries the inputs for the live n=120 population. Cured by re-deriving in place:
   `one_sample_flip_S = 4.238552517361111e-06`, and `marginal_bar_S_per_step` re-derived to
   `8.477105034722223e-08`. Commit `ff43f9dcf3`.
2. **M1R2-F2 — fire-guard receipt-tuple incompleteness.** `tools/mx1_fire_guard.py` compared a
   hand-listed subset of fire-argv flags for probe/fire equivalence, so a future trainer flag
   could silently fall outside the equivalence claim. Cured **structurally**: equivalence keys
   hoisted to module constants, `_EQUIVALENCE_EXCLUSIONS` names every excluded flag WITH its
   reason, and `_validate_flag_classification` REFUSES any argv containing a flag in neither
   set. Commit `1381ac84cb`.

## Scope (pinned; re-verify these shas yourself before reading)

| file | sha256 (first 16) |
|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `9c8373b5b352cacc` |
| `tools/mx1_fire_guard.py` | `60fc0501a65d8d09` |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `1ef18faf37e2f171` |
| `tools/ddm_seal_orchestrator.py` | `e592cb36fb00d502` |

**REVIEW SUBJECT (immutable).** The reviewed subject is the CONTENT of the four files above,
keyed by their sha256 — **NOT the live repository HEAD**. The subject's tree is that of commit
`1381ac84cb`. Live HEAD WILL advance during your pass (this charter's own commit, sister-arm
receipts, MAIN's own notes); that is EXPECTED and is NOT a finding. Descendant commits that touch
only charter files, review receipts, or MAIN's notes are OUTSIDE the reviewed subject by construction.

**The binding stop test:** re-hash the four files yourself. If any of the four CONTENT shas differs
from the table, STOP and report that as finding #1 — the subject moved under the review and the pass
is void. A differing live HEAD **alone is not such a mismatch and must not stop the pass**.

(Provenance-clause v2, repaired after M1R4A-F1: v1 pinned live HEAD, which a charter's own commit
necessarily invalidates — an unstable freeze key in a shared worktree.)

## Your questions (re-DERIVE; do not confirm)

Read the code and the ticket and answer each from the source, not from the ticket's own prose.

**A1. The fire argv actually runs what the ticket claims.** Trace every flag in the ticket's fire
argv to its `add_argument` in the trainer and to the code that CONSUMES the parsed value. Any
flag that parses but reaches no consumer is an inert-lever finding (NO-FAKE #1 territory).

**A2. The equivalence claim is sound after the cure.** The guard's real claim is *"the fire runs
at the config whose peak memory the mem-probe measured."* For EVERY key in
`_EQUIVALENCE_EXACT_KEYS` and `_EQUIVALENCE_FLOAT_KEYS`: does equality of that key actually
bear on peak memory? For EVERY key in `_EQUIVALENCE_EXCLUSIONS`: is the stated reason TRUE — i.e.
can that flag differ between probe and fire without changing peak memory? Find me a flag whose
exclusion reason is wrong. `steps` and `seed` are the easy ones; look hardest at
`eval_every`, `checkpoint_every`, `resume_from`, and `mem_probe_steps`.

**A3. The classification gate cannot be trivially satisfied.** Can an argv reach `evaluate_guard`
in a shape where `_flag_value_map` returns fewer flags than are actually passed (so unclassified
flags are invisible)? Consider: `--flag=value` vs `--flag value`, repeated flags, `--` separators,
`safe_run` wrapping, positional args, and abbreviated/prefix flag forms if argparse allows them.

**A4. Resume + per-stage checkpoint durability.** Per the standing non-negotiable, the fire must
be crash-resumable from disk and must preserve a byte-close-loadable checkpoint at EVERY stage
boundary under a DISTINCT stage-encoded filename (no overwrite), with the EMA shadow saved, written
atomically. Verify from the trainer source that the ticket's config actually produces this. A
loop-end-only save is a launch-blocking finding.

**A5. Memory preflight is measured, not projected.** Confirm the mem-probe path exercises the REAL
saturated config (fp16 + per-step hygiene + chunk cache, Q3 fallback) at the fire's own scale, and
that the guard reads the probe's MEASURED peak rather than a formula. Recall the #205 precedent:
a B=8 throughput gate handed out a false green and the n600 run OOM'd at 90 GB with no checkpoint.

**A6. The seal orchestrator's own gates.** Does `ddm_seal_orchestrator.py` fail CLOSED on a missing
or malformed receipt, or can an absent file read as a pass? Vacuity == pass is a standing genus here.

## Rules

- **Re-derive, don't confirm.** Quote file:line for every claim. If you cannot reach a conclusion
  from the source, say `UNDETERMINED` and name exactly what you would need — do not guess.
- **Never invent** CLI flags, API names, or VALUES. Grep `add_argument` / `def` before asserting any.
- **Do NOT fire the burn.** Do NOT launch any Metal job. Do NOT modify the ticket, the guard, the
  trainer, or the orchestrator. You are a reviewer; MAIN owns the cures.
- Bounded shell reads and greps are fine. If you must run the guard, run it in a way that cannot
  write into the sealed run dir.
- Commit ONLY your own receipt, via `tools/subagent_commit_serializer.py` with POST-EDIT
  `--expected-content-sha256` per file, tags `[no-triality] [p0-ledger-ok]`, and NO Claude/AI
  attribution or Co-Authored-By trailer (commits are the operator's alone).

## Deliverable

Write `.omx/research/ddm_m1r4_20260808/M1R4A_REVIEW.md` with:

1. A first line that is EXACTLY one of:
   - `VERDICT: CLEAN_PASS` — you re-derived every question above and found nothing.
   - `VERDICT: FINDINGS_RESET_COUNTER` — one or more findings.
2. The four sha re-verifications.
3. Per question A1–A6: what you derived, from which file:line, and your conclusion.
4. Per finding: severity (LAUNCH-BLOCKING / HIGH / MEDIUM / LOW), the MECHANISM (not just the
   symptom), the exact failure scenario, and the smallest correct cure. Rank most-severe first.
5. An explicit statement of what your lens did NOT cover.

## OPTIMAL FORM

- **Reference form:** an adversarial pre-launch mechanics review at full source depth over a frozen,
  sha-pinned artifact set — the same bar as the round-2 pass that found both real defects.
- **Scope reductions (legal):** review is bounded to the four pinned files plus whatever they
  transitively require you to read. No new measurement is commissioned.
- **Mechanism reductions:** NONE. This is a full-depth re-derivation, not a checklist pass. If you
  find yourself confirming the ticket's own prose rather than deriving from source, you have
  silently reduced the mechanism and the pass is invalid.
- **Provenance pins:** the four CONTENT shas above = the immutable subject; subject tree =
  commit `1381ac84cb`; live HEAD is explicitly NOT a pin (see provenance-clause v2).
