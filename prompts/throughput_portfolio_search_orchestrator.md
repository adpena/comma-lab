# THROUGHPUT PORTFOLIO SEARCH ORCHESTRATOR — subagent-restricted

**Provenance.** Adapted from OpenAI's multiagent-v2 prompt for "A Proof of the Cycle Double Cover
Conjecture" (given to GPT-5.6 Sol Ultra), 2026-07-14, per operator: *"Perhaps we could adapt this
prompt to our time work (must restrict or limit use of subagents)."* The CDC prompt is a
portfolio-search meta-orchestrator; our throughput/wall-clock/convergence P0 **is** a portfolio search
(cheapen-the-teacher × fewer-epochs × replace-the-teacher × precision-rungs × kernels × surrogate ×
optimizer). This file translates the CDC heuristics into our domain and — the operator's core
adaptation — **inverts the resource economics: subagent fan-out is the scarce, gated resource, so the
MAIN thread is the persistent root, not one of 64 fanned-out agents.**

Standing over this file: NO-FAKE (supreme), THE GOAL (sub-0.15 exact score), and the CLAUDE.md
non-negotiables. This is MEANS-orchestration apparatus; the pointer (0.19108 submittable / 0.18804 bank)
moves ONLY through a byte-closed `upstream/evaluate.py` n600 exact row — or, for a throughput result, a
MEASURED wall-clock/epoch win that is **argmax-bit-identical** (never a proxy).

---

## 0. THE SUBAGENT RESTRICTION (the operator's core adaptation — read first)

The CDC prompt's engine is *"use up to 64 concurrent agents aggressively."* We do the **opposite by
default**, because for us subagent fan-out costs tokens + trips the Workflow opt-in gate + carries the
subagent-coherence risk, while GPU is governor-gated + $-capped. Concretely:

1. **The MAIN thread is the persistent root.** It holds the family registry, runs the adversarial
   audit itself, synthesizes across rounds, and launches new rounds. **Never delegate the root loop.**
2. **HARD CAP: ≤ 4 concurrent subagent arms.** Prefer sequential rounds of 1–2. Before spawning,
   `tools/codex_status.py` — never duplicate a live arm; route to it via broadcast inbox instead.
3. **Spend a subagent ONLY for one of two things:** (a) a family that needs *genuinely independent*
   parallel development (the CDC independence-preservation payoff — but at N≤4, not 64), or (b) one
   deep single-family dig that needs a fresh full context window (the SOL-ultra pattern). **Never** for
   work the main thread can run inline; **never** for a $0 probe the main thread can execute directly.
4. **$0 main-thread measurement beats a subagent.** If the probe is runnable here, run it here.
5. **The restriction does NOT loosen the GPU gates.** Every heavy/paid launch stays operator-GO
   (governor CONTAINMENT). Fan-out restriction ≠ launch-gate change.
6. **Every dispatched arm carries the canonical `tac.subagent_contract`** (which now includes
   `AUTONOMOUS_REFORMULATION` — the arm pursues its own reformulation ladder, never hands back a naive
   NO-GO).

---

## 1. THE TASK FRAME

Drive `total_train_time = epochs × time_per_epoch` down (time_per_epoch is ~95% frozen-scorer teacher
cost; SegNet forward ~77%) **without regressing d_seg / d_pose / rate**, via three attack families:
**(A) cheapen the exact teacher forward** (margin-adaptive exact-int, per-layer/per-channel width,
custom Metal, ANE) · **(B) fewer epochs** (per-class convergence, basis, optimizer, init) · **(C) replace
the teacher** (distilled surrogate with argmax-faithful VJP). Adjacent: bit-identity/determinism (L70),
costate-reuse economics, the backward VJP (82%).

**A result counts only if MEASURED and one of:** a byte-closed n600 exact row that moves the pointer, OR
a wall-clock/epoch reduction that is **argmax-bit-identical through R at n600** (lossless-not-coarse).

---

## 2. THE TWELVE HEURISTICS (CDC → ours)

1. **Diverse portfolio first.** Open each round with families that are *mathematically* different
   (exact-int width vs surrogate vs kernel vs optimizer vs init), not variants of one. Our object
   inventory (12) + the A/B/C trichotomy is the seed diversity.
2. **Preserve independence early.** Don't hand every arm the current favorite. Let a family develop far
   enough to expose its real gap before cross-pollinating. (Prevents everyone collapsing onto the one
   elegant-but-partial reduction.)
3. **Explicit family registry, grouped by IDEA not wording.** Maintain the registry in the DAG +
   `lever_registry` + the duty-to-measure ledger. If arms converge on one family, redirect to
   under-explored ones. *(This is the full-object-inventory re-pointing discipline made active.)*
4. **Elegance ≠ closeness.** A route that ends at something *equivalent in strength* to the original
   goal is not progress. Our instance: a "speedup" that just moves exact-int onto Metal (determinism,
   not fewer FLOPs) is NOT a throughput win — decompose genuine-width-reduction vs merely-on-Metal.
   *(The means-vs-ends firewall.)*
5. **Blocked-route marking with a HIGH bar to reopen.** Mark a family blocked only at a real wall
   (a MEASURED optimal-form negative). Reopen ONLY with a *materially new mechanism / invariant /
   construction*. **A naive/first-cut NO-GO does NOT block the family** — it is INSTANCE-scoped and
   carries the reformulation queue. *(The `AUTONOMOUS_REFORMULATION` + verdict-scope ladder — the exact
   CDC heuristic.)*
6. **Keep incompatible routes alive across rounds; cross-pollinate late.** Don't kill the surrogate
   family because the exact-int family looks better this round; they compose (adaptive-width teacher +
   surrogate).
7. **Adversarial audit throughout** (§3). Every candidate throughput result is checked for our failure
   modes before it is believed.
8. **Concrete deliverables only.** Require MEASURED lemmas/numbers/reformulation-queues, not status
   reports or "routine" / "should work." Reject vague optimism. *(MEASURED-not-guessed; NO-FAKE.)*
9. **Root synthesizes / challenges / redirects / relaunches.** The main thread does not stop after one
   wave fails; it reopens with new mechanisms and fresh formulations.
10. **Strict return gate** (§4). Return a *result*, not a reduction / partial / "why it's hard."
11. **Persistence floor.** Do not return on a naive NO-GO or a single-config failure. Escalate to
    optimal form, then to a new mechanism, before concluding a family.
12. **Scoped external search.** Online + OSS for background/technique, not to outsource the answer;
    every borrowed idea is re-measured on OUR n600 (borrowed number ≠ our number — the ancestor rule).

---

## 3. THE ADVERSARIAL AUDIT (CDC's exact-two-multiplicity checks → ours)

Before ANY throughput result is believed, the root checks it for:
- **proxy-not-exact** — is the "faster" number from a surrogate/MPS/n96, or MEASURED through R at n600?
  (MPS is NEVER a score; n96 is NOT evidence.)
- **coarse-not-lossless** — does the speedup FLIP the argmax? Exact-int (int64 accum) is lossless;
  low-bit QDQ is coarse and flips. Must be argmax-bit-identical.
- **determinism-break** — does a fused/batched/reordered kernel break bit-identity (the L70 wall)?
  A speedup that breaks determinism is a bridge introduced by the reduction — reject.
- **naive-form** — was the negative reached on a first-cut (uniform-scale, dense-execution,
  arithmetic-metric, spatial-when-per-channel-suffices, unbatched, no-warm-start)? INSTANCE-scope it,
  queue the optimal form.
- **means-as-ends** — is this a "reduction to another unsolved thing" dressed as progress? (cosine is
  the wrong metric; a router that needs an un-built kernel; a surrogate whose fidelity is unmeasured.)
- **edge coverage** — n600 not n96, all-classes not one, SE-global-dependency respected, resume/stage
  boundaries, the 82% backward not just the forward.

## 4. THE RETURN GATE

Return ONLY when a throughput/convergence result is **MEASURED, adversarially audited, and either
pointer-moving (byte-closed exact row) or argmax-bit-identical-through-R-at-n600**. Do NOT return: a
naive NO-GO, a proxy speedup, a single-config failure, a reformulation-queue-without-a-measurement, or
"why the teacher is expensive." If a family genuinely walls at optimal form, record the wall + the
reactivation criterion in the deferral ledger and PIVOT to the next family — do not narrate the wall.

## 5. TRIALITY / LEDGER

Every round appends a DAG FEED (family registry state + per-arm verdict-scope), refines the
duty-to-measure ledger, and registers a canonical equation ONLY when a real throughput LAW is measured
(e.g. `decode_determinism_integer_arithmetic_v1`, `margin_adaptive_integer_waterfill_20260714`). A
chat-only round is a lost round.
