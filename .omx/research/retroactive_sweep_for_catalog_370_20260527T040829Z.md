<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113. Retroactive sweep per Catalog #348 for newly-landed Catalog #370 (canonical-submission-pipeline Phase 8 STRICT preflight gate). -->
<!-- HISTORICAL_SCORE_LITERAL_OK: this memo cites the 2026-05-19 PR101 submission anchor; no NEW score literal claims. -->

# Retroactive sweep for Catalog #370 — canonical-submission-pipeline Phase 8 STRICT preflight gate

**Date:** 2026-05-27T04:08:29Z
**Per:** CLAUDE.md "Operator gates must be wired and used" non-negotiable + Catalog #348 "EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP" self-protection.
**Operator NON-NEGOTIABLE 2026-05-26:** 9th canonical-automated-submission-packet-bundling amendment Layer 6 + 12th canonicalization × standardization × ease-of-contest-compliance trinity.

## 1. Bug-class symptom signature

A `submissions/*/` directory contains PR-facing artifacts (`PR_BODY*.md` / `PR_DESCRIPTION.md` / `README.md` with PR-title or PR-body sentinel tokens) without ALL FOUR canonical verdict sidecars present + clean. Specifically:

1. **PR-facing trigger** — submission_dir contains `PR_BODY*.md` OR `PR_DESCRIPTION.md` (existence alone qualifies) OR `README.md` whose body contains at least one of the canonical PR-facing sentinel tokens (`# PR ` / `## Submission` / `## Score` / `## Reproducibility` / `## Attribution` / `## Citations` / `[contest-CPU]` / `[contest-CUDA]` / `commaai/comma_video_compression_challenge` / etc.)
2. **Missing or unclean canonical verdict sidecar(s)** — one or more of:
   - Phase 4 `SubmissionBundleResult` (`submission_bundle_result*.json` with `overall_pass: true`)
   - Phase 5 `LintVerdict` (`lint_verdict*.json` with `overall_clean: true`)
   - Phase 6 `ComplianceVerdict` (`compliance_verdict*.json` OR `compliance_report_*.json` OR `pre_submission_compliance.contest_final.json` with `overall_clean: true`)
   - Phase 7 `PairedAuthEvalVerdict` (`paired_auth_eval_verdict*.json` OR `dual_eval_adjudicated.json` with `verdict: PAIRED_PASS` OR `overall_pass: true`)
3. **No opt-out** — submission is NOT under the canonical exempt paths (`submissions/exact_current/` per CLAUDE.md mutation frontier; `_intake_` vendored clones).
4. **No waiver** — no same-line `# PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK:<rationale>` waiver with non-placeholder rationale (≥4 chars) in README.md or PR_BODY*.md first 30 lines.

The bug-class anchor: 2026-05-19 PR101 submission_dir was hand-edited across 4 sister subagents (Slot K + L + M + J) over ~3h wall-clock with no canonical pipeline structurally enforcing the 4-verdict chain. Per Phase 1 spec memo §1 Why: *"6+ phases × multiple iterations × manual hand-editing per surface = the canonical anti-pattern."*

## 2. Pre-fix window

The bug-class was empirically demonstrated **multiple times** across the contest:

- **2026-05-19 PR101 submission cascade** — 4 sister subagents over ~3h wall-clock per `feedback_pr_submission_d5_prerequisites_executed_landed_20260519T182635Z.md`: T3 symposium PROCEED_WITH_REVISIONS verdict (commit `eac8a3a7f`); 6-of-8 prerequisites EXECUTED + 2 OPERATOR-GATED remaining. NO single canonical helper governed the full lifecycle; each phase was hand-invoked.
- **PR111-candidate planning waves** (cascade A / cascade C' / NSCS06 v8 in flight at landing): each candidate planning requires re-implementing the 4-verdict chain ad-hoc. Without Catalog #370, future PR111+ submissions would silently repeat the same hand-editing failure mode.

Per the 9th canonical-automated-submission-packet-bundling amendment: the structural extinction at this gate prevents the bug class from recurring. Phase 8 (this landing) is the STRICT gate at Layer 6; Phase 7 (paired_auth_eval, in-flight sister subagent) lands Layer 5; Phase 6 (compliance) + Phase 5 (linter) + Phase 4 (builder) + Phase 3 (archive_grammar) + Phase 2 (compression_pipeline) all landed earlier in the canonical-submission-pipeline cascade.

## 3. Historical-KILL/DEFER/FALSIFY search results

Searched repo for all PR-facing artifact patterns across `submissions/*/`:

```bash
grep -rln "## Submission\|## Score\|## Reproducibility\|## Citations\|commaai/comma_video_compression_challenge\|\[contest-CPU\]\|\[contest-CUDA\]" submissions/*/README.md submissions/*/PR_BODY*.md 2>/dev/null
```

Findings at landing (live count = 4 in-scope PR-facing submissions without canonical 4-verdict chain):

1. **`submissions/a1/`** — A1 substrate submission predates the canonical pipeline. README.md is PR-facing (`## Score` + `[contest-CPU]` + `[contest-CUDA]` tokens). Phase 4/5/6/7 sidecars all missing. Operator-routable resolution: (a) invoke each canonical helper (`tools/submission_bundle_cli.py` + `tools/submission_linter_cli.py` + `tools/submission_compliance_cli.py` + `tools/paired_auth_eval_cli.py`); (b) add waiver `# PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK:a1_submission_predates_canonical_pipeline_2026_05_13_anchor_no_re_submission_planned`; OR (c) strip PR-facing tokens from README.md if A1 submission is research-only at this point.
2. **`submissions/pr106_latent_sidecar_r2/`** — PR106 substrate R2 submission predates the canonical pipeline. Same resolution path as A1.
3. **`submissions/pr106_latent_sidecar_r2_pr101_grammar/`** — PR106 + PR101 grammar composition predates the canonical pipeline. Same resolution path.
4. **`submissions/robust_current/`** — `robust_current` substrate predates the canonical pipeline. Same resolution path.

All 4 violations are CARRY-FORWARD-FROM-2026-05-19 (predate today's canonical-submission-pipeline landing cascade). NONE are NEW bug-class anchors landed by this session.

**No historical KILL / DEFER / FALSIFY memos cite the canonical-submission-pipeline absence as the structural cause.** Prior submission landings cited individual phase blockers (D3 hosting / D5 paired auth-eval / etc.) without naming the META-class root cause (lack of canonical pipeline). The META-class is now structurally extincted at Layer 6 (STRICT gate) per Phase 1 spec memo.

Per the 9th canonical-automated-submission-packet-bundling amendment: prior recurrences manifested as different surface symptoms (PR95 review brevity discipline / PR101 attribution chain / PR102 medal-class precedent / PR103 third-prize ordering / etc.). The structural extinction at #370 prevents future recurrences at the recipe + canonical-4-verdict-chain surface.

## 4. Per-finding RE-EVAL-priority assignment

| Historical Finding | RE-EVAL Priority | Rationale |
|---|---|---|
| 2026-05-19 PR101 submission cascade (commit `eac8a3a7f` + sister) | LOW (closed in-place; structural fix landed) | The 6-of-8 prerequisites EXECUTED + 2 OPERATOR-GATED REMAINING state is preserved per Catalog #110/#113 APPEND-ONLY. The structural fix at #370 prevents the underlying anti-pattern (4 sister subagents × ~3h wall-clock per submission) from recurring. No re-eval of the historical state required. |
| `submissions/a1/` baseline (2026-05-13) | DEFERRED-pending-operator | A1 substrate carries `[contest-CPU] = 0.19285` paired CPU+CUDA empirical evidence (per `submissions/a1/dual_eval_adjudicated.json`); the submission predates Phase 4-7 canonical helpers. Operator routes: (a) generate 4 canonical sidecars; (b) add waiver; (c) declare research-only via README token strip. |
| `submissions/pr106_latent_sidecar_r2*/` baselines | DEFERRED-pending-operator | Same as A1: empirical contest-axis evidence predates canonical pipeline; operator-routable resolution per the 3 paths. |
| `submissions/robust_current/` baseline | DEFERRED-pending-operator | `robust_current` is the contest's robust archive target per CLAUDE.md "Primary duties #2"; the submission predates canonical pipeline; operator-routable resolution per the 3 paths. |
| Cascade A FEC10 hybrid PR111-candidate (in flight) | HIGH (use canonical pipeline from PR111) | First future PR submission MUST go through the canonical Phase 4 → 5 → 6 → 7 → 8 chain per Phase 1 spec memo Phase 10 acceptance criterion #3. |
| NSCS06 v8 PR111-candidate (in flight) | HIGH (same as Cascade A) | Same as Cascade A FEC10 hybrid. |
| Cascade C' PR111-candidate (in flight) | HIGH (same as Cascade A) | Same as Cascade A FEC10 hybrid. |

## 5. Sister Catalog gates protecting the same META-class

Per Phase 1 spec memo §4 catalog cross-references matrix, Catalog #370 closes the 7-layer architecture at Layer 6 (STRICT gate). The 4 canonical verdicts each route their own sister gates:

- **Phase 4 builder**: Catalog #146 (contest-compliant inflate runtime template) + Catalog #205 (canonical `select_inflate_device`) + Catalog #295 (PYTHONPATH self-containment) + Catalog #361 (Modal artifact filter preserves `output/submission/`)
- **Phase 5 linter**: Catalog #208 (docs no-local-absolute-paths) + Catalog #287 (placeholder-rationale rejection)
- **Phase 6 compliance**: Catalog #127 (authoritative-tag custody) + Catalog #152 (operator wrapper validates required input files) + Catalog #192 (macOS-CPU non-promotion) + Catalog #221 (auth-eval result artifact fail-closed) + Catalog #226 (canonical `gate_auth_eval_call`) + Catalog #240 (recipe-vs-trainer-state consistency) + Catalog #266 (archive bytes consumed by inflate)
- **Phase 7 paired_auth_eval**: Catalog #245 (Modal call_id ledger) + Catalog #313 (probe-outcomes ledger) + Catalog #339 (post-spawn fail-closed) + Catalog #360 (pre-spawn fatal observability) + Catalog #192 (macOS-CPU non-promotion)
- **Cathedral consumer**: Catalog #335 (canonical contract) + Catalog #336/#337 (auto-discovery + invoke) + Catalog #341 (Tier A canonical-routing markers)
- **META-meta**: Catalog #176 (STRICT-callsites-have-CLAUDE.md-row) + Catalog #185 (Live count: 0 verified empirically) + Catalog #299 (quota brake under 400) + Catalog #348 (retroactive sweep for new gate; THIS memo satisfies it)

Together they extinct the canonical-submission-pipeline-absence bug class structurally at 7 ORTHOGONAL surfaces: Layer 0 (compression_pipeline) + Layer 1 (archive_grammar) + Layer 2 (builder) + Layer 3 (linter) + Layer 4 (compliance) + Layer 5 (paired_auth_eval) + Layer 6 (STRICT gate — THIS landing). Phase 10 first PR111-candidate end-to-end regression unifies the 7 layers via the operator-runbook end-to-end CLI (`tools/operator_pr_submission_full_lifecycle.py`).

## 6. Cross-references

- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- CLAUDE.md "Operator gates must be wired and used" non-negotiable
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- CLAUDE.md "Public Disclosure Hygiene" non-negotiable
- CLAUDE.md "Frontier scores are pointer-only" (Catalog #343 sister) — `dual_eval_adjudicated.json` acceptance preserves canonical-pointer routing
- Phase 1 spec memo: `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`
- Phase 2 landing: `feedback_phase_2_submission_compression_pipeline_canonical_landed_20260526.md`
- Phase 3 landing: `feedback_phase_3_submission_archive_grammar_canonical_landed_20260526.md`
- Phase 4 landing: `feedback_phase_4_submission_bundle_canonical_landed_20260526.md`
- Phase 5 landing: `feedback_phase_5_submission_linter_canonical_helper_landed_20260526.md`
- Phase 6 landing: `feedback_phase_6_submission_compliance_canonical_helper_landed_20260526.md`
- Phase 7 in-flight subagent: `phase7_paired_183A1060` (sister-disjoint scope; lands Layer 5)
- 2026-05-19 PR101 cascade anchor: `feedback_pr_submission_d5_prerequisites_executed_landed_20260519T182635Z.md`
- Catalog #348 retroactive-sweep discipline: `.omx/research/retroactive_sweep_for_catalog_348_20260519T202900Z.md`
- PR submission helper: `feedback_pr_95_quantizr_study_citations_landed_20260519.md` (the PR 95 medal-class brevity discipline this gate's PR-facing sentinel detection inherits)

## 7. Discipline declarations

- Catalog #229 PV: full read of Phase 1 spec memo §3 Phase 8 + Phase 5/6 sister landings + ComplianceVerdict shape + canonical wire-in pattern + `submissions/` scope + JSON sidecar conventions BEFORE drafting
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is NEW; zero mutations of prior memos
- Catalog #287 substantive-rationale rejection — placeholder literals `<rationale>` / `<reason>` + bare-no-rationale + <4-char rationales all rejected throughout the gate's waiver mechanism
- Catalog #348 4-field contract: bug-class symptom signature ✓ + pre-fix window ✓ + historical search results ✓ + per-finding RE-EVAL-priority assignment ✓
- Catalog #299 quota brake: current catalog # is #370 well under 400 quota
- Catalog #176 META-meta: CLAUDE.md catalog row added in same commit batch (`370. \`check_no_pr_submission_without_canonical_compliance_verdict\` ...`)
- Catalog #185 META-meta-meta: Live count: 4 (warn-only baseline canonical per Phase 1 spec memo prediction) verified empirically + bounded test ≤10 in test suite

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
