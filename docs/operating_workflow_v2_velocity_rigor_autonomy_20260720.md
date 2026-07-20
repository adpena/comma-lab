# Operating Workflow v2 — velocity × rigor × autonomy (BINDING)

**Source:** operator directive 2026-07-20 verbatim: *"Our workflow should be updated to include
all of that and all of this and accelerate the velocity and encourage rigorous and autonomous
work."* — canonicalizing the 2026-07-20 boundary-displacement-convergence day (16+ custodied
landings, 5 measured convergences, 2 registered laws, 0 unforced idle ticks after correction).
Binds this agent and every subagent alongside `docs/operating_manual_craft_handoff.md` (the
craft layer; this doc is the LOOP layer). Sisters: CLAUDE.md §OPERATOR PRIORITY,
`tac.subagent_contract` (the enforcement surface for dispatch prompts).

## 1. THE LOOP (every tick, in this order)

1. **CUSTODY FIRST.** Landed arms before new work: verify each claim by RE-DERIVATION from the
   arm's primary artifacts (never by reading its summary), disposition every finding with
   `--consumed-by` naming a live consumer, route findings same-turn. A reviewed-stamp without a
   named consumer is the orphan bug class (`codex_findings_disposition_is_not_consumption`).
2. **FLEET AT USEFUL CAPACITY** (soft cap 4): exactly ONE arm owns the critical path; free slots
   carry $0 measurement accelerants with PRE-REGISTERED verdict tests that shape the NEXT hop
   (successor vehicle, re-pricing measurements, duty-to-measure queue drains). Accelerants NEVER
   duplicate the critical arm's named gaps — collision is negative work.
3. **PRE-STAGE THE NEXT HOP** before the current one lands: verify downstream readiness while
   upstream runs (e.g. Modal exact-eval harness/ledger/envelope checked BEFORE the candidate
   archive exists). The goal is zero latency between a landing and its consumption.
4. **QUIET-BOUNDARY CONSOLIDATION.** Branch reconciles, batch equation registrations, memory
   compression, gate backfills fire when the critical path pauses — never during (merge-surface
   churn under a live lineage arm is self-inflicted risk).

## 2. DISPATCH DOCTRINE (what goes INTO every arm)

- **Richest-signal, optimal-form context** (operator 2026-07-20): named artifact endpoints ARE
  the spec; inbox packets over prose; canonical equations consumed BY ID; the exact control
  numbers and gates inline. Starving an arm of context to save prompt bytes is a false economy.
- **Full online research + OSS authority in every prompt.** "$0 local" bounds SPEND, never
  information. Most of our subproblems are explored territory (matrix calculus, lattice sieving,
  camera geometry) — an arm that re-derives what a literature lookup settles is burning tokens.
- **Master-thesis framing + tiebreak law** in every solve arm: invert the frozen contest
  information space over (formulation × realization × completeness), jointly optimal
  (score, wall-clock); score-neutral choices ALWAYS resolve to least complexity.
- **Contract blocks** from `tac.subagent_contract.standard_contract()` — never re-typed by hand.

## 3. SPAWN PREFLIGHT (three checks, ~seconds each, before ANY launch/relaunch)

(a) **INTERFACE** — flag names AND values against the target's real argparse;
(b) **PATHS** — executable + entry script exist at the TARGET cwd;
(c) **BRANCH-STATE** — arm-built dependencies merged where you're launching from.
Structural backstops exist (`codex_delegate` model validator; `launch_detached_process` rc=2
fail-fast) but the checks are the discipline; the backstops are the last line. Cost of skipping
is counted on EVERY axis: tokens, wall-clock, attention, TRUST
(`launcher_invocation_preflight_verify_interface_paths_branch_state_20260720`).

## 4. RIGOR FLOOR (non-negotiable on every deliverable)

Two named doctrines govern EVERYWHERE (operator 2026-07-20: *"no binary interpretations or
naive or toy. Einsteinian and Kolmogorov everywhere"*):
- **EINSTEINIAN** — no binary verdicts, no naive/toy realizations graded as family verdicts;
  a negative is steering signal to REFORMULATE (change coordinates, change the formulation),
  never a kill; every verdict is scoped and every result re-enters the solve as a constraint.
- **KOLMOGOROV** — every floor/cost claim is about INTRINSIC complexity of what the score
  actually sees, at optimal form; a wall measured on a suboptimal realization is a statement
  about that realization's description, not the object's K; min-description is the rate spine
  and the complexity tiebreak (§2) is its wall-clock shadow.

Concretely:
- **Decompose every headline** — a bare composite is UNMEASURED; ship per-class / per-stratum /
  per-section / per-term splits or build the splitter.
- **verdict_scope on every negative** (INSTANCE < FORMULATION < FAMILY < PARADIGM); one failed
  formulation never kills a family (the Einsteinian clause, #578 binding).
- **Measured / derived / labeled, never guessed**; NO-FAKE supreme; typed custody for
  derivative-vs-secant class distinctions; assumptions labeled inline or the result is fake.
- **Own-mistake accounting on all axes** — tokens, wall-clock, attention, trust — never only
  the flattering one.

## 5. AUTONOMY CONTRACT (what fires without asking)

- **Standing-GO gates auto-fire when their preconditions clear** (recorded per-directive, e.g.
  #578: governed relaunch on rc=6 clear + memory preflight; sealed-ticket chain launches; Modal
  exact-eval ≤$20 single-flight for byte-closed candidates). Every autonomous firing is reported
  with config hash + preflight receipts. Governor REFUSE is authoritative — never bypassed.
- **Costate digest at session start** = the SENSE organ; the duty-to-measure queue is the
  default source of accelerant work for free fleet slots.
- **Not covered, ever** (per-ask): beyond-envelope spend, stopping live runs, live-config edits.
- Autonomy is priced in operator trust; the preflight discipline (§3) is what keeps it funded.

## 6. CANONICALIZATION (same-turn, quadrality)

Every measured finding lands in ALL FOUR representational legs the SAME turn it is custodied:
DAG FEED (with verdict_scope) ↔ DSL/tools (a lever or a consumer) ↔ canonical equations
(registered law or FORMALIZATION_PENDING with rationale) ↔ tasks/ledgers (disposition,
--consumed-by). A chat-only insight is a lost insight; a drift between legs is campaign-level
forgetting. Memory files for corrections/doctrine land in the SAME turn as the correction.

## 7. ENFORCEMENT

- `tac.subagent_contract` carries §2's blocks (extension owed → arm
  `workflow_contract_extension`, 2026-07-20); dispatchers compose from it, never hand-type.
- §3 is backstopped in `tools/codex_delegate.py` + `tools/launch_detached_process.py`.
- §1/§4/§5/§6 are audited at custody time; the tick protocol (ROADMAP TICK v2 cron) executes §1.
- This doc is pointer-referenced from memory
  (`workflow_v2_velocity_rigor_autonomy_20260720.md`) so it loads via MEMORY.md at session start.
