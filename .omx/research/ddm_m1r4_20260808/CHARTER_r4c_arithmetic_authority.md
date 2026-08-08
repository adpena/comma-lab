# ddm_m1r4c — M1 FIRE GATE, REVIEW PASS (round 4), LENS C: ARITHMETIC + AUTHORITY + PROVENANCE

You are one of THREE concurrent independent review passes over the same sealed M1 artifact.
Your lens is **NEW this round**, and it exists because of a measured fact about our own defects.

## Why THIS lens exists (read this; it is the whole point of the arm)

Every real defect found in the M1 artifact this week has the SAME genus:

> **a value that should have been DERIVED or IMPORTED was instead frozen as a literal.**

Five instances in two days, across totally different surfaces:
1. `one_sample_flip_S = 4.4228e-06` — a literal derived from a stale n=115 population, sitting in
   the SAME dict as the live n=120 inputs that would have produced `4.238552517361111e-06`.
2. `verdict_scope: "n32 arm ..."` — a hardcoded population string in a renderer that already had
   `pair_ids` in scope; the emitted verdicts claimed the wrong n.
3. An arm-model pin frozen as a literal in a second spawn surface, so a "never again" guarantee was
   true at one call site and FALSE at another with more callers.
4. A reasoning-effort constant hardcoded instead of selected per task.
5. Duplicated delegate defaults — a drift twin of a constant that had a rightful owner.

The two standing review lenses (MECHANICS, SCIENCE) both **cleared artifacts that contained these
defects**. Neither lens is aimed at this genus. You are. Hunt it specifically, everywhere, and
report the general cure (make it a derivation or an import) — never just the corrected literal.

## Round state

The M1 n120 receiver fire is gated on a **3-consecutive-clean-pass** counter, now at **0/3** after
round 2's two findings were cured (commits `ff43f9dcf3` and `1381ac84cb`). You are pass 1-of-3 of
the fresh sequence, alongside a MECHANICS arm and a SCIENCE arm.

**CONCURRENCY CAVEAT (binding):** all three passes review the SAME frozen state. If ANY pass reports
ANY finding, ALL THREE are void and the counter stays at 0. Do not soften a finding to protect the
sequence — a false CLEAN is far more expensive than a reset.

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

## Your questions

**C1. THE SWEEP — every number in the ticket, re-derived from its own inputs.**
Walk the ticket end to end. For EVERY numeric literal and every population/scope STRING, classify it:

| class | meaning | acceptable? |
|---|---|---|
| DERIVED-IN-PLACE | computed at use time from inputs present in scope | yes |
| IMPORTED | read from a named registry / LawRef / canonical equation / pinned receipt | yes, if the pin resolves |
| MEASURED-PINNED | a measurement, with an artifact path + sha that you verified exists | yes |
| FROZEN-LITERAL | a value whose derivation inputs ARE available but which is written as a constant | **FINDING** |
| ORPHAN-LITERAL | a value with no traceable derivation or receipt at all | **FINDING** |
| BORROWED | verified in ANOTHER vehicle's regime and transferred without re-derivation | **FINDING** |

Report the FULL table, not just the findings — the denominator is part of the deliverable. A bare
count of findings with no denominator is the vacuity genus (`skip == green`).

Pay special attention to any literal that sits in the same structure as the inputs that would
produce it. That is the exact shape of M1R2-F1, and it is the cheapest defect in the world to miss
because the number *looks* authoritative.

**C2. AUTHORITY LABELS.** For every claim in the ticket and in `MAIN_FINDING_R0_verdict_scope_constant.md`
that carries or implies an authority (`[macOS-CPU advisory]`, `[macOS-MLX research-signal]`,
`[contest-CPU]`, `[contest-CUDA]`, "MEASURED", "DERIVED", "verified"): is the label CORRECT for the
evidence behind it? Standing law: MPS/MLX is NEVER score authority; macOS-CPU is advisory, not
contest-CPU; only `upstream/evaluate.py` on the exact archive bytes is a score. An over-claimed label
is a NO-FAKE #8 finding regardless of whether the number itself is right.

**C3. SCOPE STRINGS.** Every `verdict_scope`, population descriptor, and "n=" claim in the pinned
files: does it match what the code actually iterates at run time? Three sites in the renderer were
just cured by deriving from `len(pair_ids)`; two `n32` strings were deliberately LEFT because they
are help-text prose about historical arms. Verify BOTH halves of that judgement — that the cured
sites derive correctly, AND that the left-alone sites really are prose and not live verdict scope.

**C4. DRIFT TWINS.** Does any constant in the pinned files exist in TWO places, such that editing
one leaves the other stale? Include: values duplicated between the ticket and the trainer defaults;
between the guard and the trainer; between a docstring/help string and the code it describes;
between a receipt and the artifact it describes. Name the rightful OWNER for each twin and the
import that should replace the copy.

**C5. RECEIPT RESOLVABILITY.** Every artifact path, sha, and receipt reference in the ticket: does
it resolve on disk RIGHT NOW, and does the content hash match the claim? A pinned sha that no longer
resolves is a stale-pin finding. Report any `/tmp` path in persisted evidence as a finding on sight
(transient evidence is unreproducible by any other agent or host).

**C6. THE ARITHMETIC ITSELF.** Recompute, by hand, every derived quantity in the event predicate and
the seal gates — including `one_sample_flip_S`, `marginal_bar_S_per_step`, the microbatch derivation,
and any S-arithmetic (`S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37_545_489`). Show your work.
Do not accept a number because it appears twice; a value copied into two fields is one claim, not two.

**C7. THE SELF-CHECK.** Turn the lens on itself: is there anything in THIS charter that you had to
take on faith, where the underlying value could be re-derived instead? If yes, that is a finding
about the review apparatus and I want it.

## Rules

- **Re-derive, don't confirm.** Quote file:line and show the arithmetic. `UNDETERMINED` with a named
  missing input is acceptable; a guess is not.
- **Never invent** CLI flags, API names, or VALUES. Grep `add_argument` / `def` / the registry before
  asserting any name or number exists.
- Report the DENOMINATOR everywhere — "N literals examined, M findings", never a bare finding count.
- **Do NOT fire the burn.** Do NOT launch any Metal job. Do NOT modify the ticket, the guard, the
  trainer, or the orchestrator. MAIN owns the cures.
- Commit ONLY your own receipt, via `tools/subagent_commit_serializer.py` with POST-EDIT
  `--expected-content-sha256` per file, tags `[no-triality] [p0-ledger-ok]`, NO Claude/AI attribution
  and no Co-Authored-By trailer.

## Deliverable

Write `.omx/research/ddm_m1r4_20260808/M1R4C_REVIEW.md` with:

1. First line EXACTLY one of `VERDICT: CLEAN_PASS` / `VERDICT: FINDINGS_RESET_COUNTER`.
2. The sha re-verifications.
3. **The full C1 classification table** (every literal, classified, with its derivation or receipt).
4. Per question C2–C7: what you derived, from which file:line, conclusion.
5. Per finding: severity (LAUNCH-BLOCKING / HIGH / MEDIUM / LOW), the MECHANISM, the failure
   scenario, and the smallest STRUCTURAL cure (a derivation or an import — never "change the number").
   Most-severe first.
6. A closing paragraph: is the frozen-literal genus DRAINED in this artifact, or did you stop early?
   Say which, honestly. "I found none" and "I did not finish the sweep" are different claims.

## OPTIMAL FORM

- **Reference form:** an exhaustive constant-provenance audit at the element level — every literal
  in the artifact classified against the value-provenance ladder, with the arithmetic recomputed
  independently. This is the bar the two standing lenses did NOT meet, which is why five defects of
  this genus survived them.
- **Scope reductions (legal):** bounded to the four pinned files and their transitive derivation
  sources; no new measurement commissioned; recomputation is analytic.
- **Mechanism reductions:** NONE. Sampling the literals instead of enumerating them, or reporting
  findings without the denominator, is a silent mechanism reduction and invalidates the pass.
- **Provenance pins:** the four CONTENT shas above = the immutable subject; subject tree =
  commit `1381ac84cb`; live HEAD is explicitly NOT a pin (see provenance-clause v2).
