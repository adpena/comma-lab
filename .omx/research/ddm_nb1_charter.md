# ddm_nb1 — EXHAUSTIVE negative-result audit under four implementation-form lenses

Operator directive 2026-08-09 verbatim: *"Must have codex audit all negative results
and especially look for naive or toy or generic basis or otherwise not optimal."*

You are auditing **the whole negative corpus**, not a sample. The question for every
negative is ONE question, asked four ways:

> Was the implementation that produced this negative at its family's OPTIMAL FORM —
> or did we falsify a naive/toy/generic/convenient *instance* and record it as if we
> had falsified the *family*?

Catalog #307 is the governing law: a negative from a janky prototype is
**IMPLEMENTATION-LEVEL falsified, PARADIGM INTACT, RE-OPENED**. A negative from an
optimal-form implementation is a real family verdict. Your job is to separate them,
exhaustively, and to say how many you could not reach.

## OPTIMAL FORM (required block — this charter's own honesty)

- REFERENCE form for this task = an exhaustive, denominator-reported, loop-until-dry
  audit over the machine corpora, adjudicated rows (not string hits).
- SCOPE reductions (legal, must be declared): you may bound by wall-clock and report
  the exact unreached remainder with its denominator.
- MECHANISM reductions (require an explicit TOY-BRACKET declaration and forfeit the
  family verdict): substituting keyword-grep for adjudication; sampling without
  reporting the denominator; grading from headlines instead of bodies.
- Provenance pins: cite every negative by file path + line/row id + commit sha.

## THE FOUR LENSES (the operator's exact words, made operational)

**L1 — NAIVE.** First-pass implementation; the family's known cures were never
applied. Anchor: the charter-time optimal-form law
(`naive_first_pass_born_at_charter_time_optimal_form_law_20260806`) — naive
implementations are BORN at charter time, so check the charter that spawned the
negative, not only the code. Also: `realization_gap_is_fixable_through_actual_S_R_GT_20260806`
— a NAIVE element is a TODO, never a wall.

**L2 — TOY.** Scope too small to support the verdict it was used for: subset n,
truncated epochs, capped iterations, prefix instead of stratified-random. Two hard
sub-checks:
 - **prefix bias** — m88/m96: a prefix of this population is a DIFFERENT population,
   and the bias SIGN INVERTS by axis (pose prefixes 2.5–4.2× HARDER, seg ≈0.96×
   easier, rate ≈neutral). Any pose NO-GO drawn on a prefix is the false-negative
   shape. na4 measured the rate leg; use the completed axis triple.
 - **censored caps** — a solve that stopped AT its bound while still descending is
   not a verdict (precedents: #850 pose GN 2–3 relins no convergence test; #935 sq1
   25/25 steps on 31/32 pairs; ca1's census 89 cap sites / 83 silent).

**L3 — GENERIC BASIS** *(the lens the operator named explicitly; ty1 did NOT carry it)*.
A generic default standing in for a derived or raced choice. Anchor: the
GENERIC-TRIPLE law — **a generic basis/metric/coder is a CONTROL, not a treatment;
it must be derived-or-raced.** Sweep for negatives whose treatment arm used: identity
/ Fourier / cosine basis where a derived or curvelet/shearlet basis was the actual
claim; Euclidean distance where the Fisher/S-geometry pullback is the right metric
(`cosine_is_hardly_ever_optimal_prefer_fisher_derived_basis_20260803`); a default
coder where a per-surface race was owed (#940: races-not-reputation, vindicated in
BOTH directions in one day); default optimizer/init/schedule where the DE-derived law
exists. na2 and ba31 already carry this lens at their own scope — DEDUPE, then extend.

**L4 — OTHERWISE NOT OPTIMAL (the engineering basis).** The solver form, the metric,
the realization path, or the harness was chosen for convenience rather than derived.
Includes: measuring on a proxy when the real S/R/GT/lattice path was available;
borrowed constants from a retired vehicle (constants-are-poison; the m34 no-old-lineage
ban); cross-regime constant transfer
(`cross-regime-constant-transfer-genus-finishing-stage`); wrong-object floors
(`ddm_pc2` §11: the floor you divide by decides the answer).

## MANDATORY DEDUPE — do NOT re-run these

Prior negative/naive audits, all landed. Read them FIRST; cite what each already
settled; only report rows they did not reach or graded under a lens they lacked:
`ddm_ty1_20260806/` (37/37 bounded — start from its `NEXT_IF_RESUMED.md` item 1) ·
`ddm_na1_negative_audit_20260802.md` · `ddm_na2_negative_audit_20260803.md` ·
`ddm_na3_negative_audit_20260804.md` · `ddm_ba29/ba30/ba31_negative*surfaces*.md` ·
`ddm_ea1_einsteinian_negative_audit_20260730.md` ·
`ddm_ng1_negative_signal_mining_20260729.md` ·
`ddm_rv1_conditional_validity_regrade_DAG_FEED_20260728.md` ·
`ddm_hg1_negatives_as_geometry_20260803.md`.

ty1's own named limit is your starting denominator: it "does not exhaust AU1's
11,840 correction candidates or VO2's 4,630 registry rows."

## THE CORPUS (report the DENOMINATOR for each; never a bare count)

`au1_corrections_index.jsonl` (11,840 rows / 7,592 memos) ·
`au1_headline_vs_body.jsonl` (8,157) · VO2 registry (4,630) ·
`.omx/state/probe_outcomes_ledger.jsonl` (662 live rows, 23 callers) ·
`.omx/research/**` verdict-bearing memos · the task ledger's own negative rows.
A skipped file is NOT a pass — vacuity==pass is a named genus. If you cannot reach a
stratum, say so with its size.

## METHOD

1. **Enumerate** the negative population and state its denominator per corpus.
2. **Adjudicate** each row against L1–L4 from the BODY, not the headline (stale
   headlines survive corrected bodies — that genus is measured).
3. **Grade** per Catalog #307: `PARADIGM_FALSIFIED` (optimal form, real family
   verdict) | `IMPLEMENTATION_FALSIFIED_REOPEN` (name the lens, the cure, the
   reactivation measurement, and its cost-to-falsify) | `UNREACHED` (with reason).
4. **Loop until dry** — single-pass audits are PARTIAL by definition
   (`fractal_audit_standard_single_pass_audits_are_partial_20260806`). Seal only on a
   round that adds zero new rows; include a self-audit round.
5. **Rank** the RE-OPEN rows by (stakes in S units against the measured gap
   decomposition: seg 0.4015 / pose 0.2776 / rate 0.1126 vs the PR130 floor 0.172141)
   ÷ (cost-to-falsify). Name the consumer for each.

## HONESTY BARS

- Every row cites its evidence at source. No claim from working memory.
- Negative-EXISTENCE claims ("no such case") require exhaustive search or the
  explicit words "did not find in <scope>" — this is our #1 false-claim class.
- N same-defect negatives are ONE instance, never family convergence.
- Do not invent a cure you cannot name a measurement for.
- If a prior audit already graded a row correctly, say so and move on — re-stamping
  is the trap.

## BOUNDARIES

Scorer-free. No Metal/MPS/CUDA. No scorer slot. No eval, dispatch, launch, archive
build, or promotion. No upstream edit. No public-PR-intake edit. READ-ONLY on
`/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/`. Commit via
`tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`,
tags `[no-triality] [p0-ledger-ok]`, and NO AI/Co-Authored-By trailer. Two
`review_tracker.py mark-file` passes for any `.py`.

## NO SIGNAL LOSS / NO ORPHAN SIGNAL (operator directive 2026-08-09, BINDING)

An audit that grades rows into a memo and stops **is itself the orphan bug** it was
sent to find. Measured genus: 68.6% of banked decimals reach NO consumer — the
headlined number gets banked, the rest of the same measurement does not
(`orphan_sweeps_that_do_not_write_the_store_are_the_disease_20260803`). Sister laws:
`follow_on_work_fires_immediately_or_it_is_orphan_poison_20260803` · #870 the
NAMED-$0-FOLLOW-ON-NEVER-RUN class · "you own everything — every row exits OWNED."

Therefore, binding on this arm:

1. **Every adjudicated row exits with an OWNER and a DISPOSITION.** Exactly one of:
   `FIRED` (you did it now) · `FOLDED` (written into a real consumer surface) ·
   `QUEUED-WITH-FIRE-ORDER` (named trigger + named owner) · `DEFERRED` (named,
   MEASURED blocker + the condition that reopens it). "Unowned", "MAIN to route",
   and "your call" are FORBIDDEN dispositions.
2. **Route to STORES, not prose.** A finding is consumed only when it reaches a
   machine-readable surface a consumer actually reads: `probe_outcomes_ledger`
   (23 callers, 662 live rows — the surface that WON), the task ledger,
   `tac.canonical_equations`, a DSL `Lever`, the lane registry, or the costate
   duty-queue. A `.md` alone is a bridge artifact, and must name the store that
   should absorb it next.
3. **Route the WHOLE measurement, not the headline.** If a row carries several
   measured quantities, every one gets a consumer or an explicit "no consumer
   exists, here is the one that should."
4. **Report the routing denominator.** State: rows adjudicated / rows routed /
   rows left unrouted, and why. An unrouted remainder is allowed; a SILENT one is not.
5. **Reciprocal check — the reverse direction.** Sweep for negatives whose *cure was
   already built and never wired* (built-elsewhere-unwired is a real, tracked build
   grade). A negative standing while its cure sits unwired is orphan signal on both
   ends and ranks at the top of the RE-OPEN table.

## DELIVERABLE

`.omx/research/ddm_nb1_<UTC>/` containing: `NB1_FINDINGS.md` (denominators, per-lens
tallies, ranked RE-OPEN table), `NB1_ROWS.jsonl` (one adjudicated row per negative,
typed), `NEXT_IF_RESUMED.md`, `RECEIPT.json`. Final message: the denominator, the
count graded per lens, the top-3 RE-OPEN rows by stakes÷cost with their named
reactivation measurement, and the honest unreached remainder.
