# SLOT-MG-6 positioning companion memos + PR #110 response templates LANDED 2026-05-20

**Operator question 2026-05-19/04:25Z verbatim**: *"Are we selling any of our research or innovations or OSS or otherwise short in our PR and longer term strategy for conversation with yousfi and comma ai in the comments and via email?"*

**Parent analysis verdict**: Yes — 8-10 places where PR #110 (and the surrounding PR comment / email / Discord engagement surface) under-sells the research / OSS / methodology assets the broader work has produced. This slot lands the structural remediation: 4 PUBLIC companion memos in `docs/` + 3 INTERNAL PR-comment response templates in `.omx/research/pr_110_response_templates/` + 1 inventory cross-link update.

**Lane**: `lane_slot_mg_6_positioning_companion_memos_20260519` L1 (impl_complete + memory_entry).

**Successor of**: predecessor `slot_mg_6_positioning_companion_memos_and_response_templates` crashed at credit-exhaustion 2026-05-20 ~04:45Z with ZERO files landed. This RESPAWN-MG-6 ran the full original brief from scratch per Catalog #206 crash-resume protocol.

---

## What landed

### TIER 1 — 4 public companion memos in `docs/`

All four written with: zero performative language (no "novel" / "groundbreaking" / "best-in-class" / "innovative" / "exciting" / "we're proud" / "first ever") + zero Claude / Anthropic / AI-assisted / Co-Authored / CLAUDE.md / .omx/research/ / .omx/state/ tokens in file content + honest empirical posture + canonical citations where they earn weight + cross-links to inventory + PR #110 + sister library `adpena/tac`.

1. **`docs/cargo_cult_unwind_methodology.md`** (~860 words) — generalizes the NSCS06 v6 → v7 = 44% improvement in one iteration as a reusable substrate-paradigm-rescue discipline. Sections: what it is + empirical anchor + general recipe + where it doesn't apply + internal tooling. Cites Wang & Rudin 2015 *Falling Rule Lists* (AISTATS) + Rudin 2019 *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions* (Nat Mach Intell). Honest framing: empirically validated on ONE substrate refactoring; promising but not universally proven.

2. **`docs/strict_preflight_catalog_summary.md`** (~1100 words) — browseable summary of the ~300 strict-preflight catalog gates. Sections: what it is + the two-landings discipline + 7 categories with representative gates + the META-meta gates (#118 / #159 / #176 / #185 / #186 / #299) + anti-pattern this prevents + honest scope. Cites Rudin 2019 + Wang & Rudin 2015 for the interpretable-ML lineage. Honest framing: engineering rigor, not contest-score directly.

3. **`docs/canonical_equations_tour.md`** (~1200 words) — tour of the 6 initial canonical equations with first-principles derivations: Brotli cascade bounded per stream / MPS drift architecture-class dependent / per-byte leverage uniformly distributed / per-pair master-gradient Taylor + Cauchy-Schwarz / master-gradient locality violation by codec / canonical frontier pointer. Cites Cover & Thomas 2006 *Elements of Information Theory* + Cauchy-Schwarz inequality + RFC 7932 (Brotli). Honest framing: 6 equations is small; the discipline matters more than the count; framework is anti-tribal-knowledge.

4. **`docs/master_gradient_extractor_tour.md`** (~1500 words) — tool tour with 3 operating-point surfaces (`M_contest` / `M_archive` / `M_inflated`) + 10 exploits (per-pair difficulty atlas / score-weighted reconstruction error / top-K byte ranking / bottom-K free-entropy bytes / per-class chroma allocation citing NSCS06 v6 → v7 anchor / substrate-fit diagnostic / Cramér-Rao floor estimate / bit-level score-critical bits / per-pair gradient clustering / streaming master-gradient during training) + example outputs against FEC6 / PR101 / PR106 anchors + how-to-use-it-for-your-own-substrate workflow. Cites canonical equations 4 and 5 from the sister tour. Honest framing: tool, not contest-score primitive.

### TIER 2 — 3 internal PR comment response templates in `.omx/research/pr_110_response_templates/`

Sole-author voice as Alejandro Peña per `user_pr_attribution.md`. May reference canonical Catalog / CLAUDE.md freely (INTERNAL scope). Held until actual maintainer activity warrants deployment.

1. **`response_to_maintainer_bot_eval_comment_20260520.md`** — ~50 words. Acknowledges maintainer-bot CPU+CUDA score posting + cross-references inventory + sister library + offers context. Tone: brief, technical, warm. Includes 3 variant adjustments (exact match / slight drift / substantive difference / CUDA-only).

2. **`response_to_substantive_yousfi_or_hotz_question_20260520.md`** — ~200-400 words scaffold with placeholders. Sections: acknowledgment / technical depth / next-step engagement (workshop paper / tooling / openpilot generalization / contest-policy deferred) / closing preserving long-term relationship. Scaffold-specific guidance for length calibration + tone + citation discipline + anti-patterns to refuse.

3. **`response_to_merge_or_nonmerge_verdict_20260520.md`** — C-merge (~80 words; gracious thank-you + sister library + email) + C-non-merge with 5 sub-templates per closure categories B1 score-gap / B2 not-innovative / B3 post-deadline / B4 runner-busy / B5 modifications (each ~120 words). References the prior 5-category non-merge template at `.omx/research/pr_submission_yousfi_non_merge_response_template_20260519T182635Z.md`. All sub-templates calibrated for long-term-relationship-first posture.

### Inventory cross-link update

`docs/asymptotic_floor_candidate_inventory.md` Section H "Reproducibility and cross-links" extended with a new "Companion methodology + tooling tours" sub-section linking the 4 new public memos. The pre-existing 3 cross-links (submission packet / sister library / this repo) preserved verbatim per Catalog #110/#113 APPEND-ONLY discipline; the new sub-section is operationally a curated index extension.

---

## Positioning shifts achieved

**Before**: the inventory memo + PR #110 body collectively cited the cargo-cult unwind methodology, the strict preflight catalog, the canonical equations registry, and the master-gradient extractor as parenthetical references inside the broader inventory. A reviewer who wanted to learn what any of those things were had to grep across multiple committed memos, file docstrings, and internal landing reports — most of which were under-disclosed for OSS-promotional purposes.

**After**: each of the four assets has a dedicated, browseable public memo with first-principles citations, empirical anchors, honest scope statements, and cross-links between the four memos + the inventory + the sister library + the submission packet. A reviewer landing on the inventory memo's Section H can click directly into any methodology / tooling tour without context-switching.

**Specific under-disclosures the public memos extinct**:

1. The cargo-cult unwind methodology was only narrated in the NSCS06 v6 → v7 commit messages + the inventory's Section C.10. Now: dedicated `docs/cargo_cult_unwind_methodology.md` with reusable recipe + Wang+Rudin + Rudin 2019 citations.
2. The strict preflight catalog was an internal artifact of the engineering discipline; reviewers had no entry point into what it was, why it exists, or how the META-meta gates work. Now: dedicated `docs/strict_preflight_catalog_summary.md` with the 7-category overview + the 6 META-meta gates + the structural-extinction anti-pattern.
3. The canonical equations registry sat in `.omx/state/canonical_equations_registry.jsonl` as an unreviewable JSONL with zero public entry point. Now: dedicated `docs/canonical_equations_tour.md` with first-principles derivations of all 6 initial equations + Cover & Thomas + Cauchy-Schwarz + RFC 7932 citations.
4. The master-gradient extractor and its 10 exploits were buried in `tools/extract_master_gradient.py` docstrings + research memos. Now: dedicated `docs/master_gradient_extractor_tour.md` with operator-facing tool tour + workflow guidance for any substrate designer.

**Long-term relationship preservation**: the 3 internal response templates ensure that when maintainer engagement (eval comments / substantive questions / merge or non-merge verdicts) happens, the responses are calibrated for long-term relationship preservation rather than reactive defensiveness. Each template explicitly closes with open offers of continued engagement (email / Discord / workshop paper collaboration / openpilot generalization) rather than closure-category disputation.

---

## Sister coordination

At preflight time, ~20 sister subagents were in flight across disjoint scopes (cathedral consumer canonical contracts at hooks #5/#6 closure, V8 Faiss substrate scaffolding, master-gradient slot MG-1/MG-2/MG-3/MG-4/MG-5 building uncertainty / Bayesian posterior / multi-granularity / per-pair difficulty atlas / streaming prediction, B1 E8 SGLD scope fix, DreamerV3 RSSM per-substrate symposium, T3 PR-110 Yousfi-collaborator symposium, PR-body draft v2, D3 compliance gate). All sisters were `in_progress` on disjoint file sets per their checkpoint declarations:

- Slot MG-6 (this slot) owns: `docs/cargo_cult_unwind_methodology.md` + `docs/strict_preflight_catalog_summary.md` + `docs/canonical_equations_tour.md` + `docs/master_gradient_extractor_tour.md` + `.omx/research/pr_110_response_templates/*.md` + the Section H cross-link extension of `docs/asymptotic_floor_candidate_inventory.md`.
- Sister-subagent files: none of the in-flight sisters declared any of the above paths in their `files_touched` per the canonical checkpoint JSONL. Catalog #340 sister-checkpoint guard expected to PROCEED on commit.

The inventory memo Section H cross-link extension is the only modification to a sister-touched file (the sister `slot_amend_asymptotic_floor_inventory_v2_20260520` was in-progress on the inventory). Per Catalog #110/#113 APPEND-ONLY: my change is purely additive at the end of Section H (NEW sub-section title + 4 NEW cross-link bullets); the pre-existing 3 cross-links are preserved verbatim; no body content of any other section is mutated. If the sister had landed first, my edit would auto-conflict at the EOF and force a checkpoint-aware retry per Catalog #157's `--expected-content-sha256` discipline.

---

## Discipline applied

- **Catalog #229 PV (premise verification before edit)**: read 6 source files in pre-flight (CLAUDE.md sections / user_pr_attribution.md / feedback_forbidden_claude_attribution / pr_110_hnerv_fec6_fixed_huffman_k16_submitted.md / docs/asymptotic_floor_candidate_inventory.md / tools/extract_master_gradient.py + tools/master_gradient_xray.py headers + .omx/state/canonical_equations_registry.jsonl). Verified working directory IS `adpena/comma-lab` repo (origin remote confirms). Verified target directories: `docs/` exists; `.omx/research/pr_110_response_templates/` created.
- **Catalog #206 crash-resume**: predecessor checkpoint read at start (returned no rows; predecessor crashed before first checkpoint). Emitted 5 checkpoints during this run (steps 1-5; step 5 = complete).
- **Catalog #117 / #157 / #174 / #235 / #289 canonical serializer**: all commits via `tools/subagent_commit_serializer.py` with `--expected-content-sha256` per Catalog #157 pre-pre-lock hash discipline. Post-edit working-tree shas captured AFTER all edits, BEFORE serializer invocation. Catalog #119 Co-Authored-By Claude trailer auto-appended by serializer (INTERNAL adpena/comma-lab repo per `user_pr_attribution.md` scope split).
- **Catalog #110 / #113 APPEND-ONLY**: all 7 new files are NEW (zero mutation of existing files except the inventory Section H cross-link extension which is purely additive at EOF). The 3 pre-existing cross-links in Section H preserved verbatim; the new sub-section is an additive operationally-curated index extension.
- **Catalog #340 sister-checkpoint guard**: expected PROCEED on commit (no sister has any of my files in their declared `files_touched`).
- **Catalog #287 placeholder-rationale rejection**: zero `<rationale>` / `<reason>` literals anywhere in the 7 new files or this landing memo. All cross-references are concrete file paths or URL anchors.
- **Catalog #208 zero local-absolute-paths in docs/**: zero `/Users/adpena/`, zero `/tmp/`, zero `~/.claude/` paths in the 4 PUBLIC memos. The 3 INTERNAL templates may contain canonical Catalog # / CLAUDE.md references per their INTERNAL scope.
- **Zero performative language in public memos**: grep-verified zero matches of "novel" / "groundbreaking" / "best-in-class" / "innovative" (as a self-descriptor; "innovative" appears once in `master_gradient_extractor_tour.md` quoting the PR #108 closure rubric, which is the maintainer's verbatim framing) / "exciting" / "we're proud" / "first ever".
- **Zero Claude / Anthropic / AI tokens in public memos**: grep-verified zero matches of "Claude" / "Anthropic" / "AI-assisted" / "Co-Authored" / "claude.com" / "anthropic.com" / ".claude/" in the 4 public memo bodies.
- **CLAUDE.md "Public Disclosure Hygiene"**: zero credentials, zero private infrastructure URLs, zero local absolute paths, zero raw provider logs in the public memos.
- **CLAUDE.md "Strategic Secrecy" applied generatively**: shared cross-links to the public methodology memos generously rather than withholding the broader work behind closure-status.
- **CLAUDE.md "Apples-to-apples evidence discipline"**: every empirical score literal in the public memos is axis-tagged (`[contest-CUDA T4]`, `[contest-CPU]`) per the canonical evidence-discipline contract.

---

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = **N/A** (positioning artifacts; no algorithmic signal contribution to the solver stack)
- hook #2 Pareto constraint = **N/A** (no Pareto-relevant signal)
- hook #3 bit-allocator = **N/A** (no bit-allocator signal)
- hook #4 cathedral autopilot dispatch = **N/A** (no autopilot consumer)
- hook #5 continual-learning posterior = **N/A** (positioning artifacts, not measurements)
- hook #6 probe-disambiguator = **N/A** (no probe-disambiguator; the memos and templates ARE the canonical disambiguators between under-disclosed and disclosed positioning at the OSS-readiness surface)

All six hooks are N/A by construction: this slot lands positioning artifacts (browseable public memos + held internal response templates + 1 inventory cross-link), not algorithmic primitives. The structural value is in the operator-facing OSS-readiness surface — a reviewer engaging with PR #110 / the comma.ai Discord / direct email now has dedicated entry points into the cargo-cult unwind methodology, the strict preflight catalog, the canonical equations registry, and the master-gradient extractor tool, none of which had public entry points before this slot.

---

## Operator next steps

1. **Optional review pass**: read the 4 public memos to confirm tone + scope calibration before any maintainer engagement. The internal templates are scaffold-only and require operator placeholder-fill before any deployment.
2. **Deploy templates only on actual maintainer activity**: do not preemptively post the templates to PR #110. Hold until the maintainer-bot posts eval scores (Template A), @YassineYousfi or @geohot or similar posts a substantive technical question (Template B), or an explicit verdict (Template C) lands.
3. **Sister-subagent coordination**: the sister `slot_amend_asymptotic_floor_inventory_v2_20260520` may land additional inventory amendments; if its edit conflicts with my Section H extension, the canonical serializer's `--expected-content-sha256` will refuse my commit and force a re-base against the landed amendments. Both edits should compose cleanly (sister's amendments are V2 of inventory body; my edit is V1 → V2 of Section H cross-links).
4. **Future positioning surfaces**: this slot does NOT remediate every under-disclosure. The sister library `adpena/tac` README + the comma-lab top-level README still under-disclose. Operator-routable: a follow-up slot to extend both READMEs to surface the 4 new public memos as primary entry points.

---

## File paths + word counts

| File | Path | Word count |
|---|---|---|
| Memo 1 (cargo-cult unwind) | `docs/cargo_cult_unwind_methodology.md` | ~860 |
| Memo 2 (catalog summary) | `docs/strict_preflight_catalog_summary.md` | ~1100 |
| Memo 3 (canonical equations tour) | `docs/canonical_equations_tour.md` | ~1200 |
| Memo 4 (master-gradient tour) | `docs/master_gradient_extractor_tour.md` | ~1500 |
| Template A (bot eval) | `.omx/research/pr_110_response_templates/response_to_maintainer_bot_eval_comment_20260520.md` | ~300 (incl. variant adjustments + tone notes + discipline) |
| Template B (substantive Q) | `.omx/research/pr_110_response_templates/response_to_substantive_yousfi_or_hotz_question_20260520.md` | ~700 (incl. scaffold guidance + cross-links + discipline) |
| Template C (verdict) | `.omx/research/pr_110_response_templates/response_to_merge_or_nonmerge_verdict_20260520.md` | ~900 (incl. C-merge + 5 C-non-merge sub-templates + tone notes + discipline) |
| Inventory cross-link extension | `docs/asymptotic_floor_candidate_inventory.md` (Section H additive) | +160 |
| This landing memo | `.omx/research/slot_mg_6_positioning_companion_memos_and_response_templates_landed_20260520.md` | ~1400 |
| **Total new content** | — | **~8120 words** |

---

## Executive summary (one sentence)

PR #110 was under-selling four research / OSS / methodology assets (cargo-cult unwind methodology, strict preflight catalog, canonical equations registry, master-gradient extractor); SLOT-MG-6 lands four browseable public companion memos in `docs/` plus three calibrated internal PR comment response templates in `.omx/research/pr_110_response_templates/` plus one inventory cross-link update — together extincting the under-disclosure for any maintainer / researcher / future-collaborator who engages with the submission or the broader work.

---

## Related

- `pr_110_hnerv_fec6_fixed_huffman_k16_submitted.md` (canonical PR #110 record this slot positions against)
- `pr_submission_yousfi_non_merge_response_template_20260519T182635Z.md` (prior 5-category non-merge template; sister to Template C)
- `docs/asymptotic_floor_candidate_inventory.md` (Section H extended; broader candidate inventory the public memos cross-link into)
- `user_pr_attribution.md` (sole-author voice binding for the 3 internal templates)
- `forbidden_claude_attribution_in_public_pr_surfaces.md` (zero-Claude binding for the 4 public memos)
