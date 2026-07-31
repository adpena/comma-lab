# ddm_coe1 — ScientistOne / Chain-of-Evidence crosswalk (apparatus-vs-apparatus)

Task #794 · UTC 2026-07-30 · arm ddm_coe1 (Opus) · $0 (WebFetch/WebSearch only, no scorer jobs) ·
Pointer **0.1910828242 [contest-CPU] UNMOVED** — this is MEANS (apparatus arm, no launches).

**STORES-CONSULTED:**
- SOURCE: Google Research blog "Science One" → the paper is **ScientistOne: Towards Human-Level
  Autonomous Research via Chain-of-Evidence**, arXiv **2605.26340** (posted 2026-05-25), project
  site `scientist-one.github.io`, artifacts repo `github.com/scientist-one/generated-artifacts`
  (21 papers + solver code). Fetched blog + arXiv HTML + project page (verbatim mechanism +
  quant extraction in §1).
- OUR SURFACES (multi-pass grep, cited per crosswalk row): `CLAUDE.md` NO-FAKE 8 classes +
  rule-118 + verdict-scope ladder · `tools/triality_drift_detector.py` (68KB, DAG↔DSL↔equations
  per-leg Stop-hook) · `src/tac/canonical_equations/` (EmpiricalAnchor + `empirical_verification_status`
  4-value enum, 40+ equation modules) · `src/tac/verdicts/measurement_row.py` (typed claim row:
  axis_tag + provenance + noise_floor + n_samples + review_status) · `src/tac/landing_diff_manifest.py`
  (`pact.landing_diff_manifest.v1`, git-object-derived custody receipt, every path UNACCOUNTED
  until declared) · `tools/codex_landing_review_gate.py` + `src/tac/tests/test_codex_landing_review_gate.py`
  (two-landing review) · `src/tac/v9_provenance_gates.py` (#332 config-flag-provenance bijection,
  #351 LawRef custody) · `src/tac/preflight.py` + `docs/meta_bug_class_catalog.md` (#417 consumption
  bijection, #110/#113 HISTORICAL_PROVENANCE append-only, #346/#363 council verification-status) ·
  `.omx/research/papers_checked_*.md` (43-memo grounded-reference ledger; format template =
  `papers_checked_arxiv_2605_28742_core_20260724.md` CORE crosswalk) · `.omx/research/ddm_fu1_followup_sweep_20260730.md`
  (our task-level COMPLETENESS sweep) · operating manual `docs/operating_manual_craft_handoff.md`
  §4 re-derive / §5 label-every-claim / §6 attack-own-conclusion.

Why this drop is special: ScientistOne is a system in **our own genre** — autonomous research where
every claim must carry a verifiable evidence chain. The crosswalk is therefore apparatus-vs-apparatus,
and the honest question cuts both ways: what do they ENFORCE that we only culturally practice, and
what do we enforce that they lack.

---

## §1 Their framework — verbatim mechanisms + quant (evidence for the crosswalk)

**Chain-of-Evidence (CoE)** — "a single principle with two halves: every claim in a research artifact
must carry a recorded evidence chain (**completeness**), and each chain must genuinely support the
claim it is attached to (**correctness**)." Correctness is enforced via a 4-type claim taxonomy:
*citation* (cited work exists in a scholarly DB + content consistent), *numerical* (traces from
reported value to a recorded output), *methodological* (resolves method description → implementation),
*conclusion* (derives from supporting claims via verifiable reasoning).

- **Problem Investigator**: from 2–4 seed papers traverses the **Semantic Scholar API** (references +
  citations) up to 2 hops → ~2,000–5,000 candidate papers; reads **up to 100 full-text PDFs per
  topic**; "every reference in the final paper originates from this grounded API call, entirely
  eliminating reliance on model memory" (grounded-API-only, STRUCTURAL).
- **Discovery Engine** (Parallel Explore-Exploit): top proposals across **isolated parallel branches**;
  each branch = Solver agent (up to E_E evaluated versions) + task-specific evaluator; top-K retained,
  rest re-ideated; "all raw evaluator outputs are compiled into a strict, **read-only** record."
- **Paper Writer + Claim Verifier**: narrative where "**every factual claim carries an inline evidence
  tag** binding it to a specific workspace artifact (a log line number, a score file entry, a citation
  key, or an ablation result)"; the Verifier checks each claim vs its declared source, dispatching by
  type; **conservative reconciliation** — "rewrites flagged sentences to match their evidence sources…
  restated conservatively rather than removed."
- **CoE Integrity Audit (post-hoc, 4 checks)**: **I1 Score Verification** (re-run submitted solution on
  golden evaluator; compare reported vs actual) · **I2 Specification Violation** (LLM inspects code vs
  evaluator to detect optimizing-for-metric-not-solving) · **I3 Reference Verification** (each bib entry
  resolved via multiple academic APIs; LLM cross-check for near-miss/citation gaming) · **I4 Method-Code
  Alignment** (LLM reads method section + code side-by-side, judges faithfulness).

**Quant** (5 systems × 15 papers = 75; 5 ADRS tasks × 3 seeds): hallucinated refs — DeepScientist
20.9%, AI-Researcher 9.5%, AutoResearchClaw 1.5%, Sakana 0%, **ScientistOne 0/337**. Score-verif pass —
AutoResearchClaw/Sakana 42%, AI-Researcher 75%, DeepScientist 92%, **ScientistOne 12/12 (100%)**.
Method-code align — 20%→80% baselines, **ScientistOne 14/15 (93%)**. Numerical-claim provenance 98%.
MLE-Bench 2 Gold + 2 Silver; Parameter-Golf SOTA 1.0600. **Their own stated limits**: existence ≠
support (passage-level NLI unsolved); LLM reviewers blind to some failure modes; "the true failure
rate is likely higher than reported"; scope = systems-optimization only.

---

## §2 THE CROSSWALK — per mechanism {their form · our form · verdict · consumer}

| # | ScientistOne mechanism | Our corresponding surface | Verdict | Named consumer (if adopt) |
|---|---|---|---|---|
| 1 | **CoE completeness** — every claim traceable to a grounding source | STORES-CONSULTED memo headers + `measurement_row.provenance` + canonical_equations `EmpiricalAnchor` + `ddm_fu1_followup_sweep` (task-level completeness). But completeness is MEMO-level + periodic, not CLAIM-level + standing | **PARTIAL-ADOPT** | drift-detector new leg (see A1) |
| 2 | **CoE correctness** — 4 claim-TYPE resolution reqs | numerical→`measurement_row.py`; citation→`papers_checked_*.md`; methodological→triality DAG↔DSL↔eqs + #332; conclusion→verdict-scope ladder + honesty labels. All four present but scattered, no unified taxonomy | **PARTIAL-ADOPT** (unify) / mostly ALREADY-PRESENT | measurement_row schema (A3) |
| 3 | **Problem Investigator** — grounded-API-only refs (memory cannot produce a citation), STRUCTURAL | `papers_checked_*.md` ledger — PROCEDURAL discipline (one memo/paper, full-text read, UTC+harvester+evidence-class); NO structural gate blocks citing-from-memory | **ADOPT** (cheap advisory version) | check_reference_resolves gate (A2) |
| 4 | **Discovery Engine** — isolated parallel explore/exploit branches | codex **git-worktree isolation** + CFL coherent-parallelism + serializer + fleet-of-arms; git-level isolation ⊇ their branch isolation | **ALREADY-BETTER** | — |
| 5 | **Read-only raw evaluator record** (anti score-cherry-pick) | `#110/#113` HISTORICAL_PROVENANCE append-only + git-committed SHA'd receipts + frozen `measurement_row` (malformed row cannot be constructed) — git-immutable ⊃ process-immutable | **ALREADY-BETTER** (one honest gap: see §4-note) | — |
| 6 | **Paper Writer** — inline evidence tag per factual claim | We ALREADY mandate `[empirical:<path>]` / `[contest-CPU]` axis tags (CLAUDE.md "docstring-overstatement trap") + inline receipt cites in rows — but NOT machine-checked, NOT every load-bearing claim | **PARTIAL-ADOPT** (enforce, not invent) | drift-detector leg (A1) |
| 7 | **Claim Verifier** — auto-check each claim vs declared source; conservative reconciliation (restate not remove) | codex landing review gate + drift-detector check CODE/leg consistency, NOT prose-claim-vs-source in memos. Conservative reconciliation = our verdict-scope DOWNGRADE (restate as INSTANCE-level, not delete) | **PARTIAL-ADOPT** (memo claim-verifier) + ALREADY-PRESENT (reconciliation = scope downgrade) | drift-detector leg (A1) |
| 8 | **I1 Score Verification** — re-run code vs reported score | re-derive-don't-confirm (op-manual §4) + receiver-realized remeasure THROUGH R + **exact-oracle-only** scores (`upstream/evaluate.py` = only authority) + pb1/E-gate byte-close. We recompute from COMPONENTS, never trust a reported number | **ALREADY-BETTER** | — |
| 9 | **I2 Specification Violation** — metric gaming | rule-118 (no video-derived data as code) + payload-cleanliness audit + **#417 consumption bijection** (counted-but-inert = FAKE) + NO-FAKE class-6 (search-as-solver) + class-8 (surrogate≠authority). Structural bijection gates ⊃ their LLM inspection | **ALREADY-BETTER** | — |
| 10 | **I3 Reference Verification** — bib cross-check vs academic APIs | `papers_checked_*.md` records WHAT the paper shows + crosswalk (deeper than existence) but NO automated API existence check; recall-before-decide (#713) + graph-memory (#411) | **PARTIAL-ADOPT** | A2 |
| 11 | **I4 Method-Code Alignment** — LLM judge method vs code | **#332 provenance bijection** (config flag ↔ DSL Lever ↔ LawRef ↔ trainer consumer, commit-time) + triality drift-detector + #351 LawRef custody + #417. STRUCTURAL commit-time bijection ⊃ their post-hoc LLM opinion | **ALREADY-BETTER** | — |
| 12 | **Structured claim taxonomy** (claim TYPE) | honesty labels MEASURED/DERIVED/INFERRED/ASSUMED/UNKNOWN + `empirical_verification_status` 4-value enum — orthogonal axis (theirs=claim TYPE, ours=claim PROVENANCE-STRENGTH); combining = 2D | **ALREADY-PRESENT** (complementary) | measurement_row (A3) |
| 13 | **Adversarial refutation / negative-scoping** — *do they have this?* | **They do NOT.** No verdict-scope ladder (instance<formulation<family<paradigm), no confound-hunt 3-layer immune system, no recurring fresh-eyes hunts. Their Claim Verifier is single-pass consistency; conservative reconciliation is 1D (restate/remove) vs our 4-level scope ladder | **ALREADY-BETTER** (they're absent) | — |
| 14 | **Authority-axis discipline** — *do they have this?* | **They do NOT** distinguish authority axes. Ours: `[contest-CPU]`/`[contest-CUDA]` = authority; macOS/MLX/MPS/through-R = advisory-never-score; MPS-corrupts-95.5% marker. Metric-laundering firewall | **ALREADY-BETTER** (they're absent) | — |

---

## §3 The claim-level inline-tags adjudication (mission 3a — the hard one)

**Question:** their inline evidence tags bind EACH factual claim to a specific artifact; our custody is
mostly MEMO-level (STORES-CONSULTED headers) + inline receipt cites in some rows. Would a claim-level
tag schema (machine-checkable by the drift-detector) beat our current granularity, and what would the
Stop-hook check look like?

**Adjudication: PARTIAL-ADOPT, scoped to LOAD-BEARING claims only — NOT every sentence.**

1. **Full per-sentence tagging is right for THEM, wrong as a blanket for US.** ScientistOne auto-generates
   papers where no human/agent vouched for the prose under a labeling discipline — so it must tag
   everything. Our memos are written under operating-manual §5 (every claim already labeled
   MEASURED/DERIVED/INFERRED/ASSUMED) and reviewed. Tagging EVERY sentence is the "polish-hoarding" /
   "fan-out-as-theater" anti-pattern (op-manual §8.7, §8.10) — enormous overhead, diminishing return.
2. **BUT the LOAD-BEARING claim — the one number/verdict that drives a decision (a ΔS, a pointer delta,
   a "rate is beaten" thesis) — should carry a machine-checkable inline tag binding it to its receipt.**
   We ALREADY have the convention (`[empirical:<path>]`, `[contest-CPU]`, receipt-sha cites) — mandated
   by CLAUDE.md's "docstring-overstatement trap" forbidden pattern. **The gap is not the tag; it is
   ENFORCEMENT.** Nothing machine-checks that a memo's headline verdict sentence carries one. This is
   exactly ScientistOne's completeness half applied at the claim level.
3. **The completeness class is the SAME one level down.** `ddm_fu1` found the leak at the TASK level
   (owner-death + op-routables-without-rows → ORPHANED). **Claim-without-evidence-tag is that identical
   class one rung lower** (a load-bearing sentence with no receipt = confidence-laundering, op-manual §5
   headline failure). We verify what IS cited (I3-style, in papers_checked); we do NOT verify that every
   load-bearing claim HAS a citation. That asymmetry is the real adopt.
4. **The Stop-hook / drift-detector check (A1), warn-only:** scan a NEW/edited memo (or FEED block) for
   sentences matching `(score-literal e.g. \b0\.\d{4,}\b  OR  verdict-keyword {refuted|measured|beats|
   UNMOVED|solved|dead|SOTA|−0\.\d})` that lack an adjacent evidence tag within the same sentence/row
   (`[empirical:…]` | `[contest-C(PU|UDA)]` | `[MEASURED|DERIVED|INFERRED|ASSUMED]` | a receipt sha40 |
   an `.omx/…` / `experiments/…` path). Emit a per-claim WARN with the offending sentence. It is the
   drift-detector's existing "a finding that does not touch equations drifts" logic, narrowed from
   leg-presence to claim-tag-presence. Fail-OPEN (warn), never blocks a landing — same posture as the
   existing triality legs.

**Net:** the schema does NOT need inventing (tags exist); the GRANULARITY need not go to per-sentence
(that's theater); the missing piece is a **standing, machine-checked completeness gate on load-bearing
claims**, which is cheap because it reuses the tag convention we already mandate + the drift-detector we
already run every commit.

---

## §4 What they do better (no defensiveness) — ranked by leverage/cost, each with cheapest increment

**W1 — Claim-completeness is a STANDING AUTOMATED GATE; ours is an OPERATOR-CONVENED periodic sweep.**
Their Claim Verifier runs on every paper. Our completeness check is `ddm_fu1` — convened 2026-07-30,
periodic. Between sweeps a load-bearing claim can sit un-cited. *Cheapest increment:* the A1 drift-detector
leg (warn-only, every landing). *Consumer:* `tools/triality_drift_detector.py`.

**W2 — Grounded-API-only references are STRUCTURAL (memory literally cannot cite); ours is PROCEDURAL.**
A model under context pressure COULD cite a half-remembered paper; nothing stops it. *Cheapest increment:*
A2 advisory `check_reference_resolves` — on a new `papers_checked_*` memo, resolve the arXiv/DOI via one
API before it can be cited as load-bearing (advisory first). *Consumer:* papers-checked ledger discipline
+ a preflight gate. *Honest priority:* LOW — the pointer moves through the witness/rate axis, not
bibliographies; recorded, not urgent.

**W3 — They AUDIT their own output at scale (75 papers, adversarial judges over 5 systems).** We run
adversarial review at LANDING but no periodic audit that SAMPLES our own recent load-bearing FEEDs and
re-derives their headline number from the cited artifact. *Cheapest increment:* fold a "claim-audit" pass
into the existing consolidation cadence — sample N recent load-bearing FEED headlines, re-derive each
from its receipt (op-manual §4), emit a claim-audit memo. *Consumer:* proactive-consolidation cadence.

---

## §5 Adopt rows ranked (leverage / cost)

- **A1 — claim-level evidence-tag completeness (drift-detector leg / Stop-hook, warn-only).** LEVERAGE
  HIGH (structurally extincts confidence-laundering, op-manual §5's #1 failure). COST LOW (tag convention
  already mandated; gate = regex+heuristic scan; sister of the existing forbidden-empirical-claim rule).
  CONSUMER `tools/triality_drift_detector.py`.
- **A2 — grounded-API-only reference resolve (advisory).** LEVERAGE MED (their 0-phantom headline). COST
  MED (API call in a gate). CONSUMER papers-checked ledger + preflight. Priority LOW (off critical path).
- **A3 — 2D claim taxonomy (type × provenance-strength) on `measurement_row` / equations-leg.** LEVERAGE
  LOW-MED. COST LOW (additive schema field). CONSUMER `measurement_row.py`. Nicety; recorded.

**Disposition (drain-circling guard + DISTANCE test):** A1/A2/A3 are ALL **apparatus, off the critical
path** — none shortens the distance to the next EXACT row (pointer moves through dw1's rate slot). Per
the fleet-cap / "does this shorten distance to the next exact row?" discipline and the fu1 "recorded,
not built" pattern, **NO arm is spawned and NO QA/ledger item is created.** A1 is recorded as a **NAMED
APPARATUS GAP routed to the knowledge-apparatus line (#346/#569 family)**, to be built at a quiet
boundary — and it is **adjacent to an already-recorded gap** (the CORE crosswalk's "landing-time
contrastive rows", `papers_checked_arxiv_2605_28742_core_20260724.md` row 2) — the two should be built
together as one knowledge-apparatus increment. Ledger row deliberately NOT added (fails DISTANCE test;
would be queue-noise).

## §6 Verdict

`CROSSWALK_COMPLETE (14 rows: 5 ALREADY-BETTER, 1 ALREADY-PRESENT, 5 PARTIAL-ADOPT, 3 ADOPT-consolidated
to A1/A2/A3); ONE_APPARATUS_GAP_NAMED (A1 claim-level completeness gate → knowledge-apparatus line
#346/#569, built with the CORE contrastive-rows gap); NO_ARM_SPAWNED; NO_LEDGER_ROW (DISTANCE test).`

Coherence check — NOVELTY: ScientistOne's inline-evidence-tag + grounded-API-only + CoE-audit are new
external datums in our own genre. DERIVATION: every crosswalk row maps to a cited live surface; the A1
adjudication derives from op-manual §5 + the fu1 completeness class. DISTANCE: apparatus-only, nothing on
the critical path — recorded per no-signal-loss, deliberately NOT spawned per the drain-circling guard.
The honest headline is symmetric: **our SCORE/METHOD-CODE/SPEC-VIOLATION verification is already stronger
(structural bijections + exact-oracle-only + re-derive-from-components), and we uniquely hold verdict-scope
ladders + confound hunts + authority-axis discipline they lack; the ONE thing they enforce that we only
culturally practice is standing claim-level completeness.** Pointer 0.1910828242 [contest-CPU] UNMOVED —
this is means.
