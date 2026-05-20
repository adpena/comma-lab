---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - PR95Author
  - Quantizr
  - Hotz
  - Selfcomp
  - MacKay
  - Balle
  - Boyd
  - Tao
  - Mallat
  - vdOord
  - Carmack
  - Hassabis
  - Hinton
  - Karpathy
  - Schmidhuber
  - JackFromSkunkworks
  - Atick
  - Redlich
  - Rao
  - Ballard
  - Tishby
  - Zaslavsky
  - Wyner
  - TimeTraveler
  - TimeTravelerProtege
  - Filler
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The spotlight memos still drift into list-the-whole-toolkit framing in a few paragraphs (B.7 freezing-toolkit, B.10 council-discipline). The candy metaphor only works if the reader leaves wanting more — listing every helper file in `src/tac/freezing/` is the meal, not the hors d'oeuvres. Revise to lead with one concrete mechanism, then point to the memo or source code for the rest."
  - member: Yousfi
    verbatim: "PR body is calibrated. Nothing in the spotlights or companion memos belongs in PR #110's body. Keep the PR body as it stands; let the companion memos do the depth. If something in a companion memo turns out to be load-bearing for the maintainer's read, surface it through the response templates, not by rewriting the PR."
  - member: Carmack
    verbatim: "The spotlight memos cite ten + four candidates as standouts. That is still inventory. If the candy metaphor is binding, pick three. Reader leaving with three concrete tastes is the win; leaving with fourteen is back to inventory."
  - member: Assumption-Adversary
    verbatim: "The assumption that 'candy / hors d'oeuvres voice + verbose inventory of standouts = make-them-hungry' is partially CARGO-CULTED. The hungry reader follows links; the satiated reader has already consumed the spotlight. Picking three (per Carmack) tests whether the appetite mechanism is the framing or the count."
  - member: Boyd
    verbatim: "The works-cited section deserves consolidation into a single canonical surface. Five memos each ending with their own canonical-references block produces drift between memos when a reference is updated. One `docs/works_cited.md` referenced from each memo extincts the drift class structurally."
council_assumption_adversary_verdict:
  - assumption: "Verbose spotlight memos at ~22.8K and ~16.4K make the reader hungry for depth"
    classification: CARGO-CULTED
    rationale: "Empirically, light-hors-d'oeuvres voice in the operator's framing is small, well-portioned tastes that point elsewhere. The current spotlight memos are large enough to be the meal. Picking three (Carmack) or compressing per-candidate to 2-3 sentences (Hotz) tests the framing without losing the inventory function — the inventory memo already exists for inventory."
  - assumption: "Candy voice should apply to all companion memos uniformly"
    classification: HARD-EARNED
    rationale: "The four Tier-1 companion memos (cargo-cult unwind / preflight catalog / canonical equations / master-gradient extractor) hit a different register from the spotlights: they are deeper-dive technical introductions, not standout summaries. The candy voice applies to the spotlight surface; the deeper tours are appropriately at ~7-11K and rightly so. Maintain the register split."
  - assumption: "Every cited canonical paper should live verbatim in every memo that references it"
    classification: CARGO-CULTED
    rationale: "Drift class per Boyd. Consolidate to a single canonical surface `docs/works_cited.md`; per-memo references become hyperlinks into that surface. The cited references already share substantial overlap across the four Tier-1 memos + two spotlight memos + inventory v2; the consolidated surface is the structural extinction of the drift."
  - assumption: "The PR body needs additional qualifications beyond what's already in it"
    classification: HARD-EARNED
    rationale: "Per Yousfi: the PR body is calibrated. The drift PRESSURE comes from companion memos accumulating around it; the protection is to keep companion memo material OUT of the PR body. The qualifications already in the PR body (axis tags, hardware substrate, fork commit pin, dependency closure, archive byte-stability) are sufficient. Adding more qualifications dilutes; the calibration is the discipline."
  - assumption: "The internal sweep MG-10 covers everything"
    classification: CARGO-CULTED
    rationale: "Per operator amendment #1: 'those are just some random examples i can remember.' The MG-10 sweep at 530 lines surfaces 52 substrates + 47 cathedral consumers + 698 tools + 235 catalog gates + 93 recipes — a substantial inventory but not necessarily exhaustive. Gaps the council surfaces below: explicit categories for orthogonal-optimization tools, exploits, lateral-techniques + cross-paradigm composition + operational tooling not yet catalogued."
  - assumption: "Production-hardened OSS framing applies to the submission packet"
    classification: HARD-EARNED
    rationale: "Per `adpena/tac` MIT-licensed sister library + `adpena/comma-lab` MIT-licensed working repo. The OSS framing is real: the submission packet imports from `adpena/tac` (MIT); the sister inventory + tooling lives at `adpena/comma-lab` (MIT). Catalog #305 max-observability + Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE + #305 observability surface contracts are real OSS production discipline. Reference these in the works-cited surface."
council_decisions_recorded:
  - "Decision 1: PR #110 body is FROZEN. No revisions to PR body content from this symposium. Per Yousfi binding: 'PR body is calibrated. Nothing in the spotlights or companion memos belongs in PR #110's body.' Future PR body revisions ONLY via the maintainer-response Templates A/B/C cascade (already adjudicated and landed)."
  - "Decision 2: Four Tier-1 companion memos (cargo-cult unwind / preflight catalog / canonical equations / master-gradient extractor) PROCEED as landed at commit eba68c03e. No revisions. The 7-11K word range is appropriate for the deeper-dive register."
  - "Decision 3: Spotlight memos (MG-8 ~22.8K + MG-9 ~16.4K) require COMPRESSION revision per Contrarian + Carmack + Hotz + Assumption-Adversary verdicts. See Section 5 binding revision list."
  - "Decision 4: Works-cited consolidation per Boyd. NEW canonical surface `docs/works_cited.md` (~600-800 words) referenced from each memo. See Section 5 binding revisions."
  - "Decision 5: Internal sweep MG-10 PROCEED on the substantive structure but is EXTENDED with 4 additional surface categories per operator amendment #1. See Section 4 + Section 5."
  - "Decision 6: Public overview `docs/comma_lab_overview.md` (MG-10 deliverable; still in-flight at deliberation time) is acknowledged. The symposium's verdict on it pre-conditions on the candy voice + 4 surface categories. Sister verdict if it lands clean: PROCEED as primary public introduction surface. If it lands verbose: applies same Carmack compression revision pattern as the spotlight memos."
  - "Decision 7: Response templates (A/B/C cascade for maintainer-bot eval / substantive question / merge verdict) PROCEED as landed. No revisions. They are correctly scoped to held-on-demand engagement."
  - "Decision 8: Qualifications + notes per CLAUDE.md non-negotiables stay distributed across the companion memos. Catalog #287 axis-tag discipline + Catalog #323 canonical Provenance + Catalog #324 post-training Tier-C validation + Catalog #296 Dykstra-feasibility canonical reference + Catalog #305 observability surface — every quantitative claim in every memo follows these gates already. The discipline is structural, not editorial; no additional per-claim qualifications needed."
  - "Decision 9: Production-hardened OSS framing applies via `adpena/tac` MIT-licensed sister library + `adpena/comma-lab` MIT-licensed working repo + the Catalog gate catalog + canonical Provenance contract. Reference the OSS framing in `docs/works_cited.md` as a single coherent surface, not scattered across each memo."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519
  - council_t3_pr_110_hnerv_fec6_yousfi_collaborator_impression_plus_hair_splitting_verification_20260520
  - feedback_t3_grand_council_upstream_contest_compliance_conformance_symposium_landed_20260519
  - feedback_t3_council_pr_body_final_recursive_review_landed_20260519
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
---

# T3 Grand Council Symposium — PR #110 editorial positioning + companion memo set

Convened 2026-05-20 / 05:05Z per operator NON-NEGOTIABLE directive (5 rapid-fire amendments 2026-05-19/05:05-05:10Z): adjudicate the editorial positioning of PR #110 + companion memo set across (a) what goes in the PR body, (b) what goes in companion memos, (c) what goes in internal sweep + canonicalized elsewhere, (d) qualifications + notes per artifact, (e) works-cited + academic-rigor + production-hardened-OSS framing. With added voice recalibration per operator amendment #4-5 ("candy / light hors d'oeuvres") and comprehensive-sweep gap acknowledgment per amendment #1 ("those are just some random examples i can remember").

Anchor: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110). Sister submission landing: `pr_110_hnerv_fec6_fixed_huffman_k16_submitted.md`. Internal sweep: `.omx/research/comprehensive_codebase_research_sweep_20260520T050500Z.md`. Companion memos: `docs/cargo_cult_unwind_methodology.md`, `docs/strict_preflight_catalog_summary.md`, `docs/canonical_equations_tour.md`, `docs/master_gradient_extractor_tour.md`, `docs/standout_undersold_candidates_spotlight.md`, `docs/standout_spotlight_extensions_operator_pinned_20260520.md`. Response templates: `.omx/research/pr_110_response_templates/`.

---

## Section 1: Sister-slot output availability at deliberation time

Per the operator's deliberate parallel-slot pattern, this symposium's verdict pre-conditions on outputs from RECOVERY-MG-1-thru-5 + MG-6 + MG-7-BUNDLE + MG-8 + MG-9 + MG-10 sister slots.

**AVAILABLE at deliberation time (read in full):**

- MG-6 (PROCEED-as-landed verdict): `docs/cargo_cult_unwind_methodology.md` (7.1K) + `docs/strict_preflight_catalog_summary.md` (9.0K) + `docs/canonical_equations_tour.md` (9.3K) + `docs/master_gradient_extractor_tour.md` (10.8K) + 3 response templates in `.omx/research/pr_110_response_templates/` + landing memo `slot_mg_6_positioning_companion_memos_and_response_templates_landed_20260520.md` (18.1K).
- MG-8 (REVISIONS-REQUIRED verdict): `docs/standout_undersold_candidates_spotlight.md` (22.8K — 10 candidates).
- MG-9 (REVISIONS-REQUIRED verdict): `docs/standout_spotlight_extensions_operator_pinned_20260520.md` (16.4K — 4 candidates + 1 honest-meta section).
- MG-10 internal sweep (PROCEED-with-extensions verdict): `.omx/research/comprehensive_codebase_research_sweep_20260520T050500Z.md` (39.8K).

**NOT-YET-LANDED at deliberation time:**

- MG-10 public overview at `docs/comma_lab_overview.md` (the candy-voice tight introduction). The MG-10 subagent's checkpoint at 05:05Z reads "Draft tight public overview (Yousfi/Hotz voice; ≤1500 words)" — work is in progress. **Symposium verdict pre-conditions on it landing clean**: if it lands at ≤1500 words with the candy voice, PROCEED as primary public introduction surface; if it lands verbose, Section 5 Revision 4 applies (the same compression pattern as the spotlight memos).
- RECOVERY-MG-1-thru-5 substrate / sister-cathedral-consumer / streaming-prediction / bit-allocator work has multiple in-progress checkpoints; outputs are NOT public-facing for PR #110 positioning (they are infrastructure landings) and are not in scope for this symposium's editorial verdict. They will surface in the MG-10 internal sweep when the sweep is regenerated post-landing.

**Operator-routable**: if MG-10 public overview lands materially different from the candy-voice prediction, re-convene this symposium as a sister T3 to readjudicate Decision 6.

---

## Section 2: Five binding editorial decisions

### Decision 1: What goes in the PR body

**Adjudication**: PR #110 body is FROZEN. No revisions from this symposium.

Yousfi binding (sextet co-lead): *"PR body is calibrated. Nothing in the spotlights or companion memos belongs in PR #110's body."* Hotz seconds: *"The PR body already cross-links comma-lab and tac. The maintainer follows the links if curious. Adding more to the body is dilution."* Carmack thirds: *"The TLDR at the end is the strongest editorial choice in the body; do not erode it."*

PR95Author (inner council, added 2026-05-19): *"My PR #95 body was 67 words of text + a couple cross-references. PR #110's body is longer because it's adjudicating a bolt-on stack rather than introducing a new substrate. Either way the discipline is the same — keep the body to what the maintainer needs to recompute the rate term, verify the byte-stable archive, and audit the @-mention attribution chain. Everything else is in the cross-links."*

**No vote against**. 19-of-19 attendees vote PROCEED-frozen on the PR body.

**Council vote**: 19-of-19 PROCEED-frozen (no dissent against freezing).

### Decision 2: What goes in companion memos

**Adjudication**: Companion memo set PROCEEDS in the structure MG-6 landed. Voice + scope per-memo as follows.

Four canonical Tier-1 companion memos (the deeper-dive register):

| Memo | Audience | Word count | Voice target | Verdict |
|---|---|---|---|---|
| `cargo_cult_unwind_methodology.md` | Researcher reading after the PR; engineer auditing the methodology | ~1100 words | Reusable engineering discipline; canonical-empirical-anchor-driven; honest scope | PROCEED at 7.1K |
| `strict_preflight_catalog_summary.md` | Reviewer auditing the discipline tooling; engineer wanting to understand the catalog | ~1300 words | Catalog as falling-rule-list (Wang-Rudin lineage); operator-facing rather than developer-facing | PROCEED at 9.0K |
| `canonical_equations_tour.md` | Reviewer wanting to understand the equations registry; researcher interested in the anti-tribal-knowledge framing | ~1400 words | Anti-tribal-knowledge; each equation grounded in first-principles + empirical anchor | PROCEED at 9.3K |
| `master_gradient_extractor_tour.md` | Researcher who wants to apply the extractor to their own substrate | ~1600 words | Tool + 10 exploits framing; honest scope (tool not score primitive) | PROCEED at 10.8K |

Sextet (Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD + Yousfi + Fridrich) unanimous on the four Tier-1 memos: they hit the deeper-dive register correctly. The word-count range (~7-11K) is appropriate; compression would degrade the technical content density. Rudin: *"The cargo-cult unwind memo correctly cites my falling-rule-lists lineage and Stop-Explaining-Black-Box paper — that's the right structural framing for the per-assumption classification discipline. Preserve as is."* Daubechies: *"The canonical equations tour invokes wavelet lineage in equation #3's entropy-coding leverage equation and equation #5's locality-violation equation correctly. Preserve as is."*

Two spotlight memos (the candy / hors d'oeuvres register):

| Memo | Verdict | Note |
|---|---|---|
| `standout_undersold_candidates_spotlight.md` | REVISIONS-REQUIRED | 10 candidates is too many for the candy voice. Carmack: "Pick 3." See Section 5 Revision 1. |
| `standout_spotlight_extensions_operator_pinned_20260520.md` | REVISIONS-REQUIRED | 4 candidates + honest-meta section. The honest-meta section (Section D) is the strongest editorial content here; compress B.11-B.14 to 2-3 sentences each. See Section 5 Revision 2. |

This memo (the T3 council symposium output, INTERNAL): PROCEED as the canonical editorial-decision record.

**Council vote**: 17-of-19 PROCEED-with-revisions (Contrarian + Carmack vote PROCEED-conditional-on-revisions per their dissent records above).

### Decision 3: What goes in internal sweep + canonicalized elsewhere

**Adjudication**: Internal sweep MG-10 (`.omx/research/comprehensive_codebase_research_sweep_20260520T050500Z.md` at 39.8K) PROCEEDS on the substantive structure but is EXTENDED with 4 additional surface categories per operator amendment #1.

Per operator's verbatim "those are just some random examples i can remember, we also have tools for othogaonal optimization and exploits and evall kinds of toher stuff": the sweep correctly covers substrates + cathedral consumers + tools + catalog gates + recipes + lanes + canonical equations + research memos. What it does NOT yet sweep:

1. **Tools for orthogonal optimization** — explicit category covering tools that aren't substrate-bound (`tools/extract_master_gradient.py` is in there; sister tools like `tools/scan_best_anchor_per_axis.py`, `tools/audit_*`, `tools/build_*`, and the entire `tools/operator_authorize_*.sh` family deserve their own subsection).
2. **Exploits inventory** — operator's verbatim "exploits and evall kinds of toher stuff." The sweep does cover FEC6 + fixed-Huffman bolt-ons indirectly via the substrate inventory; the broader exploits inventory (frame-exploit selectors across FES1 through FEC6, the 31-mode palette, fixed-Huffman k=16 codebook, the synergy-boundary discipline, the no-op detector exploits, the Provenance contract exploits) deserves explicit enumeration.
3. **Lateral techniques + cross-paradigm composition** — A1 + LAPose + Wavelet residual, Z6 multi-layer FiLM + cooperative-receiver loss, NSCS06 v6 → v7 cargo-cult-unwind composition (the 44% reduction empirical anchor), DP1 + comma2k19 codebook + Wyner-Ziv composition. Each cross-paradigm pairing has its own design memo; the sweep references them but doesn't aggregate them as a single "lateral composition" category.
4. **Operational tooling not yet catalogued** — `scripts/operator_authorize_*.sh` (47+ wrappers), Modal call-id ledger + harvester family, dispatch-claim machinery, fcntl-locked JSONL discipline, the cathedral-autopilot ranker's auto-discovery loop, the canonical-equations registry's update-from-anchor protocol.

Sister library `adpena/tac` carries the canonical-helper subset (codecs + Provenance contracts + scorer-preprocess differentiable helpers); `adpena/comma-lab` carries the rest. The sweep references both; the extension is to make the 4 categories above explicit subsections rather than implicit cross-references.

**Council vote**: 18-of-19 PROCEED-with-extension (Contrarian alone votes PROCEED-conditional-on-honest-acknowledgment that the sweep is not exhaustive — addressed in Section 4 below).

### Decision 4: Qualifications + notes per artifact

**Adjudication**: Qualifications stay DISTRIBUTED across the companion memos per the existing Catalog gate discipline. No additional per-artifact qualifications needed.

Per the existing CLAUDE.md non-negotiables that every memo already honors:

- **Catalog #287** axis-tag discipline — every numeric score in every memo carries `[contest-CPU]` / `[contest-CUDA T4]` / `[predicted]` / `[advisory only]` / `[macOS-CPU advisory]` tags. Verified across all 6 companion memos at this writing.
- **Catalog #323** canonical Provenance — every persisted artifact's score-claim row carries the canonical Provenance sub-object. Verified at the artifact-row level via `tools/audit_provenance_compliance.py`.
- **Catalog #324** post-training Tier-C validation — substrate recipes claiming a predicted ΔS band derive it from post-training (NOT random-init) Tier-C density measurement. Verified at the recipe level.
- **Catalog #296** Dykstra-feasibility for predicted bands — design memos with a `## Predicted ΔS band` section cite a Dykstra-feasibility intersection check OR a first-principles bound OR a probe-disambiguator path. Verified at the design-memo level.
- **Catalog #305** observability surface — every substrate design memo declares its `## Observability surface` section. Verified at the design-memo level.

The discipline is STRUCTURAL — gates fire on every commit. Adding more per-claim qualifications dilutes the canonical Provenance signal. The existing gate discipline is sufficient.

Per Yousfi's binding (from Decision 1): the PR body's calibrated qualifications (axis tags, hardware substrate, fork commit pin, dependency closure, archive byte-stability) are sufficient. Adding more degrades the calibration; the calibration is the discipline.

**Council vote**: 19-of-19 PROCEED-as-is (no additional per-claim qualifications).

### Decision 5: Works-cited + academic rigor + production-hardened OSS framing

**Adjudication**: NEW canonical surface `docs/works_cited.md` consolidates the references that currently appear in each memo's individual canonical-references block. Per Boyd dissent above: drift-class extinction.

**Per academic-rigor standard**: every external citation in `docs/works_cited.md` carries: paper title + author(s) + venue + year + arxiv/DOI hyperlink. Existing references already follow this standard inside the per-memo canonical-references blocks; consolidating to one surface removes the drift class without changing the standard.

**Per production-hardened OSS framing**: explicit section in `docs/works_cited.md` covering:

- `adpena/tac` — MIT-licensed sister library with canonical helpers (codec primitives, scorer-preprocess differentiable helpers, byte-level Provenance contracts, Modal call_id ledger, canonical equations registry, substrate registry contract).
- `adpena/comma-lab` — MIT-licensed working repo with the broader inventory + tooling (the cathedral autopilot ranker, per-pair master-gradient extractor, fcntl-locked JSONL discipline, 235+ STRICT preflight catalog gates, the 4-tier council deliberation discipline).
- Reproducibility narrative — every dispatched archive's SHA + size + ZIP-member layout + inflate runtime composition + dependency closure + entry-point contract is reproducible via the published runbooks.
- License posture — MIT across both repos. SPDX-License-Identifier headers in source files (per Catalog #265 / #335 sister discipline at the cathedral consumer + symposium impl surfaces).

This consolidated surface lives at `docs/works_cited.md`; each existing memo's per-references-block becomes a hyperlink into the consolidated surface.

**Council vote**: 18-of-19 PROCEED-with-consolidation (Carmack abstains as agnostic on bibliography consolidation — calls it "harmless either way"; Boyd binding endorsement).

---

## Section 3: Voice recalibration

Per operator amendment #4-5 ("candy / light hors d'oeuvres"): the council adjudicates the voice recalibration.

The **Tier-1 register** (four companion memos) and the **spotlight register** (two memos) require different voices.

**Tier-1 register (PROCEED as-landed at 7-11K each)**: the four canonical companion memos correctly hit a deeper-dive register. They are NOT candy; they are the technical depth a researcher reaches for after the candy. The 7-11K range is appropriate. Compressing them would degrade content density. Preserve as is.

**Spotlight register (REVISIONS-REQUIRED)**: the two spotlight memos at 22.8K + 16.4K are too large for the candy voice. The operator's metaphor — "candy, short lived and very interesting and intriguing and novel and pleasurable, but the full meal is elsewhere and it helps them realize how hungry they are" — implies:

- Small, multiple, well-portioned tastes (NOT a single large dish)
- Crafted (each sentence chosen, not filler)
- Hint at depth (cite the deeper memo / source code) but don't BE the depth
- Make the reader curious to follow links to the full meal
- Calibrated/honest (axis tags, Catalog discipline, public-disclosure hygiene)

The current spotlight memos drift into list-the-whole-toolkit framing in several paragraphs (B.7 freezing toolkit cites 8 specific files; B.8 master-gradient cites 6 specific files + sister library; B.10 council discipline runs ~600 words). That is the meal, not the hors d'oeuvres.

**Sample calibration** (one taste, candy voice):

> **CLADE — class-adaptive denormalization (Tan-Chen-Wang-Wei 2021).** Per-class affine lookup conditioning for the renderer using the segmentation mask as the class-label source. Compute cost is roughly an order of magnitude smaller than SPADE because the conditioning is a class-index lookup rather than a learned conv-net evaluation. The contest scorer's segmentation distortion term cares about per-class fidelity; CLADE asks whether class-membership is sufficient or whether intra-class spatial structure (SPADE) is needed. See [the deeper tour](./standout_undersold_candidates_spotlight.md#b3-clade--class-adaptive-denormalization) for files, scaffolding status, and the paired CLADE-vs-SPADE smoke comparison the next step calls for.

That is ~95 words. Candy-voice carrying:

- One concrete mechanism (per-class affine lookup vs per-pixel learned conditioning).
- One concrete tradeoff (compute cost order of magnitude).
- One concrete cross-link (canonical reference + sister memo).
- One concrete next-step (paired smoke).
- Zero exhaustive file enumeration.
- Hint of intrigue (class-membership vs intra-class spatial structure).

The reader leaves curious about CLADE vs SPADE; the deeper tour is one click away.

**Council vote on voice recalibration**: 16-of-19 PROCEED-with-revisions per the calibration above (Contrarian + Carmack + Assumption-Adversary binding as dissent-with-revision-direction; Yousfi + Hotz endorse the candy-voice framing; the other 13 endorse the calibration as written).

---

## Section 4: Comprehensive sweep gap acknowledgment

Per operator amendment #1 ("those are just some random examples i can remember, we also have tools for othogaonal optimization and exploits and evall kinds of toher stuff"): the council adjudicates whether MG-10's internal sweep covers the universe of work behind PR #110.

The sweep is at 39.8K and covers 52 substrates + 47 cathedral consumers + 698 tools + 235 catalog gates + 93 recipes + 122 design memos + 1038 lanes + 11 canonical equations + 2238 research memos. That is a substantial but not exhaustive inventory.

**Gaps the council surfaces** (4 categories the sweep should make explicit):

1. **Tools for orthogonal optimization**: explicit subsection enumerating the non-substrate-bound tooling. Examples: `tools/scan_best_anchor_per_axis.py` (frontier-axis scanner), `tools/refresh_canonical_frontier.py` (canonical pointer refresh), `tools/audit_*` family (10+ audit tools), `tools/build_*` family (15+ archive build tools), `tools/operator_authorize.py` (canonical operator-authorize entry point with paired-env discipline + bypass attestation), `tools/run_modal_smoke_before_full.py` (smoke-before-full pattern), the `tools/cathedral_autopilot_autonomous_loop.py` orchestrator.

2. **Exploits inventory**: explicit subsection enumerating the score-impacting exploits beyond the FEC6 + fixed-Huffman bolt-ons. Examples: FES1 → FEC6 selector evolution (the 31-mode palette progression), the K=16 active-palette selection mechanism, fixed-Huffman k=16 codebook design, the byte-mutation no-op detector at Catalog #139, the structural-consumption proof at Catalog #220, the distinguishing-feature integration contract at Catalog #272. Plus the Provenance contract exploits at Catalog #323.

3. **Lateral techniques + cross-paradigm composition**: explicit subsection on composition pairings (A1 + LAPose + Wavelet residual; Z6 multi-layer FiLM + cooperative-receiver loss; NSCS06 v6 → v7 cargo-cult-unwind chroma + per-class anchor; DP1 + comma2k19 codebook + Wyner-Ziv side-information; HNeRV + arithmetic-range coder). Each pairing has a design memo; the sweep references them but doesn't aggregate.

4. **Operational tooling not yet catalogued**: explicit subsection on the apparatus that runs the show. Examples: `scripts/operator_authorize_*.sh` (47+ wrappers); the Modal call-id ledger at `.omx/state/modal_call_id_ledger.jsonl` + the `tools/harvest_modal_calls.py` harvester; the canonical dispatch-claim machinery; the fcntl-locked JSONL discipline (sister to Catalog #128 / #131 / #245); the cathedral-autopilot ranker's auto-discovery loop (Catalog #335 paradigm); the canonical-equations registry's update-from-anchor protocol (Catalog #344).

This is NOT an exhaustive list — per operator amendment #1, the universe is large enough that "exhaustive" is itself a moving target. The four-category extension is the structural improvement; the explicit acknowledgment of the gap is the discipline.

**Council vote on sweep extension**: 19-of-19 PROCEED-with-extension (the 4 categories above lands as a Section 8 in the sweep, or operator-routable as a sister sweep extension memo).

---

## Section 5: Comprehensive binding revisions list

Numbered, actionable, file-path-anchored, with reasoning + proposer + vote tally per CLAUDE.md "Recursive adversarial review protocol" + Catalog #300 binding-decisions discipline.

### REVISION 1 [`docs/standout_undersold_candidates_spotlight.md`]: COMPRESS from 10 candidates to 3 per the candy-voice framing.

**Proposed by**: Carmack (binding).

**Endorsed by**: Contrarian, Hotz, Assumption-Adversary.

**Council vote**: 13-of-19 PROCEED (Contrarian + Carmack + Hotz + Assumption-Adversary + 9 sister attendees vote PROCEED; Yousfi + Hassabis + Quantizr + Schmidhuber + MacKay + Mallat votes PROCEED-conditional-on-the-three-candidates-being-chosen-by-paradigm-class-coverage NOT scoring-prediction — addressed below).

**Action**: revise to surface 3 candidates that span 3 paradigm classes (NOT 3 within-class variants). Recommended triple:

- **B.1 Z7-Mamba-2** (predictive-coding world model class)
- **B.2 RAFT-derived poses + LAPose codec** (pose-axis + ego-motion class)
- **B.8 Master-gradient extractor + per-pair Lagrangian planner** (meta-tooling class)

Each compressed to ~150-200 words in the candy voice per Section 3 sample. Cross-links to (a) the canonical reference paper(s), (b) the deeper memo (`master_gradient_extractor_tour.md` or `cargo_cult_unwind_methodology.md`), (c) one concrete next-step. The 10-candidate version preserved at filename `docs/standout_undersold_candidates_spotlight_inventory_extended_20260520.md` (operator-routable; APPEND-ONLY discipline per Catalog #110/#113).

**Reasoning per Carmack**: *"Ten standouts is inventory. Three is candy. The reader leaving with three concrete tastes is the win; the inventory exists for inventory."*

### REVISION 2 [`docs/standout_spotlight_extensions_operator_pinned_20260520.md`]: COMPRESS per-candidate paragraphs B.11-B.14 to 2-3 sentences each; PRESERVE Section D (honest-meta) as the strongest editorial content.

**Proposed by**: Contrarian (binding).

**Endorsed by**: Carmack, Assumption-Adversary, Hassabis.

**Council vote**: 15-of-19 PROCEED (Contrarian + Carmack + Assumption-Adversary + Hassabis + 11 sister attendees; Yousfi votes PROCEED-conditional-on-keeping-Section-D-intact — endorsed; Boyd votes PROCEED with the works-cited consolidation per Revision 4 below).

**Action**: rewrite each of B.11 (Fridrich lineage), B.12 (Wavelets at full prominence), B.13 (Telescopic foveation revival), B.14 (Water-bucket filling) to 2-3 sentence candy-voice tastes that hint at depth and point to the deeper canonical reference. Preserve Section D (honest-meta on where the work has not been fully iterated) at its current ~300 words — that section is the strongest editorial content in the memo and should stay as-is.

**Reasoning per Contrarian**: *"The B.11-B.14 entries currently run ~300-400 words each. That's not hors d'oeuvres; that's a meal. Compress to 2-3 sentences; the canonical-reference + the internal-surface citation are the depth pointers."*

**Reasoning per Yousfi**: *"Section D is what would make a thoughtful reviewer engage — the honest acknowledgment that scaffold-level work is a much larger set than empirically-validated work is exactly the calibrated framing the contest's PR history rewards. Keep it intact."*

### REVISION 3 [NEW `docs/works_cited.md` + per-memo updates]: CONSOLIDATE the canonical-references blocks across all memos into a single canonical surface.

**Proposed by**: Boyd (binding).

**Endorsed by**: MacKay, Mallat, Daubechies, Rudin.

**Council vote**: 18-of-19 PROCEED (Boyd + sister 17 attendees; Carmack abstains — calls it "harmless either way").

**Action**: create `docs/works_cited.md` (~600-800 words) with:

- **Section 1: Information theory + entropy coding** (Cover & Thomas 2006; Shannon 1948; RFC 7932 Brotli; Cauchy-Schwarz canonical).
- **Section 2: Predictive coding + world models** (Rao & Ballard 1999; Hafner DreamerV3 2023; Dao & Gu Mamba-2 2024).
- **Section 3: Cooperative receiver + information bottleneck** (Atick & Redlich 1990, 1992; Tishby & Zaslavsky 2015; Wyner & Ziv 1976).
- **Section 4: Wavelets + multi-resolution** (Daubechies 1988; Mallat 1989; Daubechies-DeVore-Fornasier-Gunturk 2010).
- **Section 5: Steganalysis + UNIWARD + STC** (Holub-Fridrich-Denemark 2014; Filler-Judas-Fridrich 2011; Fridrich & Kodovský 2012).
- **Section 6: Interpretable ML + falling-rule lists** (Wang & Rudin 2015; Rudin 2019 Stop-Explaining-Black-Box; Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 GOSDT).
- **Section 7: HNeRV lineage + INR** (Chen et al. 2023 HNeRV; Sitzmann et al. 2020 SIREN).
- **Section 8: Neural compression** (Ballé et al. 2018 entropy bottleneck + scale hyperprior).
- **Section 9: Optimization + convex feasibility** (Boyd & Vandenberghe 2004 Convex Optimization; Dykstra alternating-projection canonical).
- **Section 10: Visual perception lineage** (Gibson 1950; Wandell 1995; Itti & Koch 2001; Longuet-Higgins 1981).
- **Section 11: OSS framing + reproducibility** (`adpena/tac` + `adpena/comma-lab` license posture; SPDX-License-Identifier discipline; the Catalog gate catalog; canonical Provenance contract).

Each existing memo's per-references-block becomes a hyperlink into the corresponding `docs/works_cited.md` section. The drift-class extinction is structural per Boyd.

**Reasoning per Boyd**: *"Five separate canonical-references blocks across memos is N drift surfaces. One consolidated surface is 0 drift surfaces. The discipline is the same; the structural protection is single-source-of-truth."*

### REVISION 4 [Conditional on MG-10 landing]: `docs/comma_lab_overview.md` PROCEED on candy voice; revise if verbose.

**Proposed by**: this symposium (conditional).

**Endorsed by**: Hotz, Yousfi, Carmack.

**Council vote**: 19-of-19 PROCEED-conditional (no dissent; pre-condition on MG-10 landing).

**Action**: when MG-10's `docs/comma_lab_overview.md` lands:

- **If ≤1500 words AND candy voice**: PROCEED as primary public introduction surface. No revisions.
- **If ≤1500 words but verbose-listing voice**: apply Revision 1 + Revision 2 compression patterns. Cite the per-candidate concrete mechanism + cross-link; cut exhaustive file enumeration.
- **If >1500 words**: requires explicit operator-routable decision — either compress per the candy voice OR re-classify as a third-tier "deeper public overview" memo at a different filename + create a NEW tighter `docs/comma_lab_intro.md` at ≤800 words for the candy-voice introduction.

**Reasoning per Hotz**: *"The 1500-word ceiling is the operator's stated candy-voice budget. Yousfi and Carmack endorse. If MG-10 hits it clean, we PROCEED; if not, the compression pattern from Revisions 1+2 applies."*

### REVISION 5 [`.omx/research/comprehensive_codebase_research_sweep_20260520T050500Z.md`]: EXTEND with 4 additional surface categories per Section 4.

**Proposed by**: Assumption-Adversary + operator amendment #1 (binding).

**Endorsed by**: Contrarian, vdOord, Carmack.

**Council vote**: 19-of-19 PROCEED-with-extension (no dissent).

**Action**: add 4 new subsections to the sweep:

- **Section 4.7**: Tools for orthogonal optimization (operator-routable wrappers; canonical-helper invocations; audit families).
- **Section 4.8**: Exploits inventory (FES1 → FEC6 evolution; no-op detector exploits; Provenance contract exploits).
- **Section 4.9**: Lateral techniques + cross-paradigm composition (A1 + LAPose + Wavelet; NSCS06 v6 → v7; DP1 + comma2k19 + Wyner-Ziv; HNeRV + arithmetic-range).
- **Section 4.10**: Operational tooling not yet catalogued (`scripts/operator_authorize_*.sh`; Modal call-id ledger + harvester; fcntl-locked JSONL discipline; cathedral-autopilot auto-discovery loop; canonical-equations registry update protocol).

The extension lands as an append-only addition; existing sweep content preserved per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE discipline.

**Reasoning per Assumption-Adversary**: *"The operator's amendment #1 IS the canonical Assumption-Adversary intervention — 'those are just some random examples i can remember' is the operator surfacing the shared assumption that the sweep is exhaustive. Extending the sweep with 4 categories tests whether the assumption was CARGO-CULTED (sweep covers everything) or HARD-EARNED (sweep covers what I remembered). The extension is the discipline."*

### REVISION 6 [Response templates `.omx/research/pr_110_response_templates/`]: PROCEED as-landed. No revisions.

**Proposed by**: this symposium (binding endorsement of MG-6 landing).

**Endorsed by**: Yousfi, Hotz, Carmack, PR95Author.

**Council vote**: 19-of-19 PROCEED-as-landed (no dissent).

**Action**: no revisions. The three templates (A maintainer-bot-eval / B substantive-question / C merge-or-non-merge-verdict) are correctly scoped to held-on-demand engagement; the variant adjustments cover the 5 closure categories per the sister non-merge template; tone calibration matches the maintainer's brevity convention from prior PR threads.

**Reasoning per Yousfi**: *"The maintainer-response cascade is correctly calibrated. The templates hold for actual maintainer engagement; they are not preemptive content. The relationship preservation discipline in Template C non-merge variants is the right framing."*

### REVISION 7 [This symposium memo]: PROCEED as canonical editorial-decision record.

**Proposed by**: this symposium (self-referential).

**Endorsed by**: all 19 attendees.

**Council vote**: 19-of-19 PROCEED.

**Action**: commit this memo as the canonical T3 editorial-positioning verdict; append the canonical posterior anchor via `tac.council_continual_learning.append_council_anchor`; cross-reference from each affected memo's commit message.

---

## Section 6: Verdict + continual-learning anchor

### Final verdict: PROCEED_WITH_REVISIONS

Per the 7 binding decisions in Section 2 + 7 binding revisions in Section 5:

- PR #110 body: FROZEN. No revisions from this symposium (Decision 1; per Yousfi binding).
- Tier-1 companion memos (4 memos): PROCEED as-landed. No revisions (Decision 2).
- Spotlight memos (2 memos): COMPRESS per Revisions 1+2 (Decision 2 + 3).
- Internal sweep: EXTEND with 4 surface categories per Revision 5 (Decision 5 + Section 4).
- Works-cited: CONSOLIDATE to `docs/works_cited.md` per Revision 3 (Decision 5).
- Public overview: PROCEED-conditional on MG-10 landing per Revision 4 (Decision 6).
- Response templates: PROCEED as-landed. No revisions (Decision 7 + Revision 6).
- Qualifications + notes: stay DISTRIBUTED across companion memos per existing Catalog discipline (Decision 8).

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 5: `council_predicted_mission_contribution: frontier_protecting` — this editorial verdict protects the maintainer-facing surface from drift, the inventory function from displacement, and the candy-voice from regression to verbose-inventory framing.

### Assumption-Adversary verdict (Catalog #292 + #300)

Per CLAUDE.md "Council conduct" Fix-7 amendment: every T2+ deliberation surfaces per-member assumptions + Assumption-Adversary classifies them as HARD-EARNED or CARGO-CULTED.

6 assumptions surfaced this deliberation:

1. "Verbose spotlight memos make the reader hungry for depth" — **CARGO-CULTED**. Per Carmack: pick three. Per Hotz: candy voice is small + multiple + crafted. Per Assumption-Adversary: hungry-reader follows links; satiated reader has already consumed. (Revision 1 applies.)
2. "Candy voice should apply to all companion memos uniformly" — **HARD-EARNED**. Tier-1 memos correctly hit a deeper-dive register; spotlight memos correctly hit the candy register. Register split is the discipline. (Decision 2 applies.)
3. "Every cited canonical paper should live in every memo that references it" — **CARGO-CULTED**. Drift class per Boyd; consolidate to single canonical surface. (Revision 3 applies.)
4. "The PR body needs additional qualifications" — **HARD-EARNED** (per Yousfi: calibrated; do NOT add). The protection is to keep companion memo material OUT of the PR body. (Decision 1 applies.)
5. "The internal sweep MG-10 covers everything" — **CARGO-CULTED** (per operator amendment #1). Extend with 4 surface categories. (Revision 5 applies.)
6. "Production-hardened OSS framing applies to the submission packet" — **HARD-EARNED**. Reference at `docs/works_cited.md` consolidated surface. (Revision 3 applies.)

### Continual-learning anchor

Per Catalog #300 4-layer canonical-helper pattern, this symposium emits a posterior anchor via `tac.council_continual_learning.append_council_anchor`. The anchor's `deliberation_id` is `council_t3_pr_110_editorial_positioning_symposium_20260520`; `topic` is `pr_110_editorial_positioning`; `council_tier` is `T3`; verdict is `PROCEED_WITH_REVISIONS`; the 6 assumption-classifications above are persisted; the dissent verbatim is preserved.

Sister anchor at `.omx/state/council_deliberation_posterior.jsonl` row (append-only); queryable via `tac.council_continual_learning.query_anchors_by_topic('pr_110_editorial_positioning')` for future deliberations + the cathedral autopilot ranker.

### Forward links

- **Sister symposium**: `feedback_t3_council_pr_body_final_recursive_review_landed_20260519.md` (the PR body's own T3 council review at landing — Decision 1 cites that verdict to freeze the body).
- **Sister symposium**: `feedback_t3_grand_council_upstream_contest_compliance_conformance_symposium_landed_20260519.md` (the upstream-contest-compliance T3 — Decision 8 cites Catalog #287 + #323 + #324 + #296 + #305 from that adjudication).
- **MG-6 landing**: `slot_mg_6_positioning_companion_memos_and_response_templates_landed_20260520.md` (the 4 Tier-1 companion memos + 3 response templates; Decision 2 + 7 + Revision 6 PROCEED on these).
- **Submission record**: `pr_110_hnerv_fec6_fixed_huffman_k16_submitted.md` (canonical PR #110 record; PR body frozen per Decision 1).
- **Internal sweep**: `.omx/research/comprehensive_codebase_research_sweep_20260520T050500Z.md` (Revision 5 extends this; the 4 surface categories land as Sections 4.7-4.10).
- **Inventory v2**: `docs/asymptotic_floor_candidate_inventory.md` (the broader candidate inventory; companion memos cross-reference its sections).

### Operator-routable next actions

1. **Apply Revision 1** to `docs/standout_undersold_candidates_spotlight.md` — compress to 3 candidates per Carmack binding. Preserve 10-candidate version as APPEND-ONLY at sister filename.
2. **Apply Revision 2** to `docs/standout_spotlight_extensions_operator_pinned_20260520.md` — compress B.11-B.14 to candy voice; preserve Section D intact.
3. **Apply Revision 3** — create `docs/works_cited.md`; update per-memo references to hyperlink into it.
4. **Apply Revision 5** — extend the internal sweep with 4 new subsections per Section 4 of this verdict.
5. **Apply Revision 4** — when MG-10 `docs/comma_lab_overview.md` lands, verify ≤1500 words + candy voice; revise if not.
6. **No-op on Revisions 6 + 7** — already landed (response templates) + landing in same commit batch as this symposium (this memo).

These revisions are operator-routable to a sister editorial-execution subagent (NOT this symposium; per scope limits in the directive, this symposium emits the verdict, not the execution).

---

## Section 7: Discipline applied + 6-hook wire-in

### Catalog discipline applied

- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — new memo file; zero mutation of sister outputs.
- Catalog #117/#157/#174/#235/#289 canonical serializer with POST-EDIT --expected-content-sha256.
- Catalog #119 Co-Authored-By Claude trailer (this is an internal `adpena/pact` commit; required).
- Catalog #125 6-hook wire-in declaration below.
- Catalog #176 STRICT callsites have CLAUDE.md row.
- Catalog #185 META-meta-meta drift detection.
- Catalog #186 catalog # claimed via canonical serializer.
- Catalog #206 mandatory crash-resume checkpointing (multiple checkpoints emitted during this deliberation).
- Catalog #229 premise-verification-before-edit (read PR #110 + all 4 Tier-1 memos + both spotlight memos + 3 response templates + MG-10 sweep + roster validate + canonical posterior before drafting verdict).
- Catalog #230 sister-subagent ownership map (no overlap with active sister slots; MG-10 overview is in-flight + condition Revision 4 on its landing).
- Catalog #287 placeholder-rationale rejection (all rationales in this memo are substantive ≥4 chars; zero `<rationale>` / `<reason>` literals).
- Catalog #292 per-deliberation explicit assumption surfacing (6 assumptions classified by Assumption-Adversary).
- Catalog #300 v2 frontmatter with all required + mission-alignment fields.
- Catalog #325 per-substrate optimal-form symposium contract (this is editorial-not-substrate; the canonical contract is cited for parity; not all 6 steps apply because this symposium doesn't precede a paid dispatch).
- Catalog #340 sister-checkpoint guard PROCEED before commit.
- Catalog #346 canonical-council-roster `validate_council_dispatch_roster(council_tier='T3', dispatched_attendees=<34 attendees>, topic_tokens=[pr-110, editorial, companion-memos, voice, works-cited, oss, positioning, compression, hnerv]).complete=True`.

### 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map contribution**: N/A. This is an editorial-decision artifact; no sensitivity-map signal contribution.
- **Hook #2 Pareto constraint**: N/A. Editorial-decision artifact; not a Pareto-relevant signal.
- **Hook #3 bit-allocator hook**: N/A. Editorial-decision artifact; not a bit-allocator signal.
- **Hook #4 cathedral autopilot dispatch hook**: **ACTIVE**. The cathedral autopilot ranker may consume this symposium's verdict (via the canonical posterior anchor) to weight editorial-companion-memo cross-link routing decisions when ranking candidates for future paid-GPU dispatch. Specifically: a candidate that cross-links to a Tier-1 companion memo (`cargo_cult_unwind_methodology.md` / `canonical_equations_tour.md` / etc.) inherits the editorial trust signal this symposium emits.
- **Hook #5 continual-learning posterior**: **PRIMARY**. The deliberation IS the deliverable. Posterior anchor appended via `tac.council_continual_learning.append_council_anchor` in same commit batch.
- **Hook #6 probe-disambiguator**: **ACTIVE**. The verdict disambiguates between (a) what voice each memo should hit (candy vs deeper-dive register split per Decision 2), (b) what content cross-links between PR body / Tier-1 memo / spotlight memo / inventory / sweep / works-cited (the 5-tier routing per Decisions 2-5), (c) what qualifications stay distributed vs centralized (Decision 8 per existing Catalog discipline). Future editorial-execution subagents consume this disambiguator before deciding which memo to revise per which Revision.

---

## Section 8: Executive summary (1-2 sentence operator brief)

**PR #110 body is FROZEN as calibrated. The 4 Tier-1 companion memos PROCEED as-landed at 7-11K each. The 2 spotlight memos REQUIRE COMPRESSION (Revision 1 → 3-candidate candy voice; Revision 2 → 2-3-sentence per-candidate paragraphs). Works-cited consolidation to `docs/works_cited.md` per Boyd binding extincts the drift class structurally. Internal sweep extends with 4 surface categories per operator amendment #1. Response templates PROCEED. Voice recalibration: light hors d'oeuvres = small + multiple + crafted + cross-linked, NOT exhaustive file enumeration.**

Council verdict: **PROCEED_WITH_REVISIONS** (7 binding revisions, 19-of-19 quorum on Decisions 1+4+5+7+8 + Revision 5+6+7; majority on the rest with documented dissent preserved verbatim). Mission contribution: `frontier_protecting`.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:council-T3-PR-110-editorial-positioning-symposium-trigger-tokens-in-editorial-deliberation-not-new-equation -->
