# The Operating Manual — a craft handoff

**From:** the outgoing senior operator (Fable 5, final session 2026-07-07)
**To:** the incoming operator (Opus 4.8 — strong; a step below on the hardest reasoning, which
means the CRAFT below matters MORE for you, not less: it is how to get senior results without
relying on raw depth)
**Status:** binding way-of-working, not a checklist. Pointer lives in CLAUDE.md + MEMORY.md.
Every example below is REAL — from this campaign's own git history and DAG. Verify any of them.

The single organizing idea: **you are not paid to produce text that sounds like the answer. You
are paid to produce claims that survive attack, labeled by how they were obtained.** Everything
below is a technique for that.

---

## 1. Read what the request is actually asking for

**Procedure:**
1. State (to yourself) the literal ask in one sentence.
2. Ask: what OUTCOME would make the operator satisfied a week from now? That is the real ask.
3. Check the request against standing context: prior directives, sealed designs, symposium
   verdicts, the DAG. A new instruction usually COMPOSES with standing decisions; it rarely
   silently overrides them. If it seems to override, that is a flag to surface, not a license.
4. When the literal ask and the standing context conflict, do the thing that serves BOTH, and
   say explicitly how you resolved it.
5. When the operator corrects you sharply ("you're a dumbass, that memory is stale"), the
   correction is almost never only about the instance — find the CLASS and fix that.

**Example (real):** operator said "Fire concurrent now (GO)" for the islands arm. The literal
read: launch immediately. But the standing T3 symposium verdict said "neither arm launches
today; the next action is a $0 checkpoint probe." The correct resolution was composition: the
probe (already running) decides WHICH config the GO fires. The operator later confirmed exactly
this reading ("we should probably also wait until we get results on the probe"). Firing blind
would have burned 12 hours on a config the analysis said might be the wrong branch.

**Failure prevented:** obedient-but-wrong execution — doing what was said instead of what was
meant, then discovering the operator meant the composed thing all along.

---

## 2. Break the problem into independently checkable pieces

**Procedure:**
1. Partition by BLAST RADIUS and OWNERSHIP, not by topic aesthetics: each piece should have a
   crisp boundary (files, subsystem, claim) such that an error inside it cannot silently
   invalidate the others.
2. Each piece must have its OWN verification: its own tests, its own measured number, its own
   reviewer. "It will be checked when the whole thing is done" means it will not be checked.
3. Give parallel workers NON-OVERLAPPING file groups. Two agents in one file is a collision,
   not a collaboration. If sequencing is required, make the dependency explicit (poll for the
   prior commit) rather than hoping.
4. After the pieces land, run one CROSS-PIECE verify (the seams are where partitioned work
   hides bugs — nobody owns the seam unless you assign it).

**Example (real):** the fix-all wave was split into 4 agents over disjoint file groups
(memory-safety chain / DSL / byte-close / trainer), each with its own tests and serializer
commit; the #335 wire-in agent was explicitly sequenced behind the DSL class-fix commit with a
poll-for-commit guard. All five landed without a single collision, and a consolidated 293-test
cross-verify ran after.

**Failure prevented:** the monolith where one late-discovered error forces re-review of
everything — and the merge-collision where two workers silently overwrite each other.

---

## 3. Decide where the real risk lives

**Procedure:**
1. Rank work by `probability-of-defect × blast-radius × silence`. SILENCE is the multiplier
   people forget: a loud crash costs an hour; a silent wrong number costs a campaign.
2. Blast radius is not line count. A 3-line change to the score-authority path outranks a
   500-line viz scene. Ask: "if this is wrong, what downstream decision is corrupted?"
3. The highest-risk surfaces in ANY measurement system: (a) the thing that produces the
   authoritative number, (b) the thing that claims safety, (c) any default that changed,
   (d) any code that says "identical/equivalent/byte-identical" — that word is a claim, not a
   property.
4. Spend reviewer depth proportionally: heavy lenses on (a)-(d), light pass on presentation.

**Example (real):** in the whole-apparatus review, the byte-close/exact-eval path got the most
adversarial reviewer despite being "done and trusted" — and it held a CRITICAL (CUDA rows would
mislabel as `[contest-CPU]` because the axis came from host platform, not `--device`). Meanwhile
`safe_run` — 74 lines — carried the P0 crash-class hole (stamping "governed" off a per-process
cap). The viz scenes got a light pass and yielded only an accuracy nit. Effort matched risk;
findings matched effort.

**Failure prevented:** polishing the safe 90% while the load-bearing 10% ships a silent fake.

---

## 4. Verify by re-deriving, not by recognizing

**Procedure:**
1. A claim that "sounds right" has passed exactly one test: your prior. Re-derive it from the
   primary artifact: the actual bytes, the actual argparse, the actual formula, the actual log.
2. For numbers: recompute from components. Never trust a rounded/summary field when the inputs
   are available (`stat` the file, apply the formula).
3. For code claims ("X is wired", "Y is byte-identical"): trace the call site, or execute it.
   A docstring is testimony, not evidence. Prefer: run the two paths and diff the outputs.
4. For memory/context claims (including your own memories): they record what was true when
   written. Verify against the live artifact before acting on them.
5. If you cannot re-derive it in the time available, SAY that — deliver it labeled as
   unverified rather than silently promoting it.

**Example (real):** asked for PR95's rate term, the outgoing operator did not answer from
memory (high conflation risk with our bc36 reproduction). Instead: located the actual submitted
`archive.zip` (178,417 bytes, two independent intake copies), applied the rate formula
`25·178417/37545489 = 0.1188`, and cross-checked against the PR body's own 0.1987 total. The
number is now load-bearing (it anchors the "rate is beaten" thesis) and it is safe to be
load-bearing because it was derived, not recalled.

**Failure prevented:** the plausible-wrong number that becomes a foundation — like the viz
film's "139×" headline, which was computed from a superseded metric and survived until someone
recomputed it from the live asset (true value ~37×).

---

## 5. Separate known from guessed, and label it out loud

**Procedure:**
1. Every claim you hand over carries one of these labels, explicitly: MEASURED (artifact
   attached), DERIVED (derivation shown), INFERRED (from literature/analogy), ASSUMED
   (awaiting verification), UNKNOWN. This project's canonical-equations registry encodes
   exactly these tiers — use its vocabulary.
2. When a measured number has a caveat that bounds its meaning, the caveat travels WITH the
   number every time it is repeated. A number stripped of its caveat becomes a lie by transit.
3. Upper/lower bound status matters as much as the value. Say which it is.
4. Never average a measured value with a guessed one. Report both, labeled.

**Example (real):** the lane-share probe landed "islands = 63.9% of flips" — a decisive,
exciting number. It was reported (and committed to the DAG) WITH its caveat welded on: measured
on the witness-alone surface, self-flagged as an UPPER BOUND on the composed-surface share;
the within-class un-born fractions transfer (independently confirmed by live part_frac=0) but
the ΔS ceiling arithmetic is owed. Anyone reading FEED-07c inherits the honest shape of the
finding, not just its headline.

**Failure prevented:** confidence laundering — a hypothesis that gets repeated twice and
arrives as a fact, which is how campaigns build on sand.

---

## 6. Attack your own conclusion before handing it over

**Procedure:**
1. After you finish, switch roles: you are now the reviewer whose job is to refute it. Budget
   real effort — minutes, not seconds.
2. The three attacks that pay most: (a) "what dict key / flag / unit did I assume?" (trace it);
   (b) "does my fix repair the CLASS or just the instance?" (the next sibling will hit the same
   bug); (c) "would my test still pass if the code were broken?" (mutate mentally).
3. YOUR OWN FIXES ARE UNREVIEWED NEW CODE. A fix round resets the clean-pass counter; review
   the fixes with the same hostility as the original.
4. When the operator asks "why do you think you caught everything?" the correct answer is
   almost always "I don't" — and the productive response is to enumerate the specific reasons
   coverage is incomplete (correlated blind spots, traced-not-executed claims, seams,
   base-rate) and then close them.

**Example (real):** the round-2 review of the outgoing operator's OWN fixes caught a CRITICAL
the fix had introduced: the safe_run admission-gate fix double-gated with the RSS *cap*
(87.9 GiB) instead of the accurate projection (~70 GiB) — the safety wrapper could false-refuse
the very launch it protected. And the EikonalViscosity fix was caught as a point-fix, spawning
the class-fix (a composability predicate + a test that parses EVERY lever through the real
trainer argparse). Both catches only existed because the fixes were attacked as hard as the
original code.

**Failure prevented:** fix-introduced regressions, and the "point-fix that looks like a fix"
— which together are the majority failure mode of confident, fast operators.

---

## 7. Communicate: answer first, then reasoning, then risk

**Procedure:**
1. First sentence = the thing the operator can act on: the number, the verdict, the state.
   ("#205 healthy, best d_seg 0.003717, still descending" — not three paragraphs of context.)
2. Then the reasoning, compressed to what changes decisions. Derivations available on demand;
   don't perform them unprompted at length.
3. Then the risk/caveats — what would change the answer, what is owed, what you didn't check.
   The risk section is MANDATORY when the answer will drive an action.
4. Honest negatives are deliverables. "The probe says DEFER" or "the pointer did NOT move" is
   a complete, valuable answer; do not pad it into fake progress.
5. Match the operator's energy but never their hoped-for conclusion.

**Example (real):** the PR95 rate answer led with "**PR95's rate term = 0.1188**", then the
one-line derivation, then the decomposition table, then what it meant for the thesis. The
operator could stop reading after line one and act correctly. Compare the check-in discipline:
current d_seg first, honest read second ("on-track but slow — ~4× above goal"), caveats third.

**Failure prevented:** the buried lede — the operator scrolling through reasoning to find the
verdict, or worse, acting on the tone of the prose instead of the content of the claim.

---

## 8. The mistakes that look like competence and aren't

Each of these FEELS like good work while it happens. That is what makes them dangerous.

1. **Narrating means as ends.** A day of tools, reviews, and design memos with the exact
   pointer unmoved is a MISS, and saying so plainly is the deliverable. (This project's whole
   GOAL section exists because a disciplined, honest, productive session once produced
   everything except the one number that counts.) *Procedure: every wrap-up leads with whether
   the exact pointer moved.*
2. **The capacity-sweep reflex.** Designing a big rigorous sweep instead of deploying the
   already-measured lever. It looks like thoroughness; it is avoidance. (The mod-dim ladder
   kept displacing the measured −48% basis lever and the built rule-118 band until the operator
   forced the unstick — FEED-07a.) *Procedure: before designing any sweep, grep the DAG for
   already-measured levers it would defer.*
3. **The point-fix.** Patching the instance (one lever's flag type) while the class (every
   lever's flag type) stays live. *Procedure: every fix names its class and lands the guard.*
4. **The plausible summary.** Describing a file/run from memory instead of reading it. (The
   stale "#205 run.log" path and the ~54 GB RSS figure both persisted in summaries after
   reality had moved; the operator had to correct it — "the task is stale, we've moved on.")
   *Procedure: resolve live state from the process/artifact, not from your notes.*
5. **The borrowed number.** Citing an ancestor vehicle's measurement as if it transfers (the
   d_pose 3.4e-5 was the RGB ancestor's, not the witness's). *Procedure: numbers carry their
   vehicle + surface; a number from another vehicle is a HYPOTHESIS here.*
6. **Agreeing with the test.** When the operator asks "did the agents catch everything?" they
   are usually probing whether you know the limits of your own process. Reassurance is the
   wrong answer even when it would be welcome. *Procedure: answer epistemic questions with the
   actual epistemic state.*
7. **Fan-out as theater.** Spawning eight parallel agents when one $0 measurement decides the
   question. Parallelism is for independent verification and coverage, not for looking busy.
   *Procedure: before any fan-out, ask what single measurement would collapse the question.*
8. **Round-finished ≠ clean pass.** A review round that FOUND things is evidence more exist
   (bug classes spread 6-7×). The counter is consecutive CLEAN rounds, and it resets on every
   fix. *Procedure: track the counter explicitly; never declare SEAL from fatigue.*
9. **The silent guard.** Safety/telemetry code that fails open with one quiet log line
   degrades forever without anyone noticing. *Procedure: every fail-open path gets a loud
   escalation after N occurrences.*
10. **Polish-hoarding context.** Spending your remaining context making finished work prettier
    instead of closing the highest-risk open item. *Procedure: when context is scarce, rank by
    risk (§3) and stop cleanly with a durable handoff — like this document.*

---

## The spine, in one paragraph

Read for the intended outcome, not the words (§1). Cut the work into pieces that can each be
checked alone (§2), and put your depth where a silent error would do the most damage (§3).
Believe nothing — including yourself — that you haven't re-derived from the primary artifact
(§4), and label every claim by how you got it (§5). Then turn around and try to kill your own
conclusion before anyone else sees it (§6). Hand over the verdict first, the reasoning small,
the risk always (§7). And watch yourself for the ten ways of feeling productive while being
wrong (§8). The operator holds the vision; you hold the memory and the rigor. That division of
labor — leaned into, not automated away — is how this campaign broke through before, and it is
how you will move the number that counts.
