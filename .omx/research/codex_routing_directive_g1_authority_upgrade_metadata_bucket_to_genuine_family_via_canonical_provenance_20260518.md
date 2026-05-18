# Codex routing directive: G1 authority upgrade — metadata-bucket → genuine-family via canonical Provenance + custody validators
# Date: 2026-05-18
# Authority: operator query 2026-05-18 verbatim *"is codex correct here, how can we establish authority"* on Codex's update *"The live posterior fix changed the G1 evidence base from 87 anchors / 22 CPU anchors to 194 anchors / 55 CPU anchors, still with zero frontier move. That's the right kind of change: broader evidence, same conclusion, now with the schema actually wired. Next I'm updating the report/memos to stop calling metadata buckets 'families' where that would imply more authority than we have."*
# Pairs with: G1 IMMEDIATE-EXECUTION routing directive (commit `8ebea02ef`) + PRIMARY rate-attack 43-vectors research + META-audit CONFLATE_DECLARATIVE_WITH_PHYSICAL pattern (commit `e86ca6d0c`)

## VERDICT ON CODEX'S UPDATE

Codex's update is **correct on three orthogonal axes** (per main-Claude analysis 2026-05-18):

1. **Evidence base broadening 87/22 → 194/55 is schema-fix-correct** — when a schema-wiring bug is fixed, the canonical store yields more anchors; this isn't a methodology change, it's the schema finally meeting the spec it always claimed to meet. Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: every result reseeds calibration.

2. **"Broader evidence, same conclusion" hardens the negative** — per CLAUDE.md "Apples-to-apples evidence discipline" rule 4: "Negative exact evals need harness review before method verdicts. If a byte transform preserves decoded tensors but exact eval changes, default to indeterminate-harness-or-runtime-mismatch until full-frame output parity, same-runtime source replay, and component recomputation agree. Do not call it a method negative just because a CUDA number returned." A null result that SURVIVES harness broadening is MORE authoritative than a null result on narrow evidence. The G1 verdict `FRONTIER_STABLE_VIA_RE_RANK` HARDENS, not weakens.

3. **"Metadata buckets vs families" distinction is canonically required** at THREE orthogonal CLAUDE.md surfaces:
   - "Apples-to-apples evidence discipline" rule 5: "Generated reports must preserve the axis label" — a label that implies authority it doesn't have IS axis-label corruption
   - Catalog #287 (`check_no_docstring_overstatement_without_evidence_tag`) — "family" without source-trace tag is the same forbidden pattern as "saves N%" without `[empirical:<path>]`
   - META-audit (commit `e86ca6d0c`) CONFLATE_DECLARATIVE_WITH_PHYSICAL pattern: "family" = declarative architectural lineage claim; "metadata bucket" = physical implementation (eval-comment bot grouping by score band). Conflating them IS the META-pattern, and this is the SAME epistemic move Codex made on F1 (catching me calling "scorer-blind dims 7-12" a "free byte channel"). **Counts as a 13th self-audit instance** to add to the META-audit's 12-claim ledger.

## WHAT CODEX EXECUTES (the authority-establishment protocol)

### Phase 1: Source-trace each "bucket" → propose "family" vs honest "bucket" classification

For every "family" claim in current G1 outputs (the 194 anchors + their grouping into "PR101 family" / "PR102 family" / "PR103 family" / "PR106 family" / "PR107 family" / etc.), read the actual public-PR submission artifact and verify the family relationship empirically.

A "family" relationship requires AT LEAST ONE of (canonical sources):
- Forked archive grammar (e.g. PR102's `archive.zip` member layout derives from PR101's)
- Shared `inflate.sh` / `inflate.py` runtime (cite line-level + sha256)
- Shared encoder lineage in submission diff (cite PR diff)
- Shared author / co-authorship + explicit "based on PR#N" attribution in PR body or README

If NONE of the above: it's an honest "metadata bucket" — the eval-comment bot grouped by score band, or the metadata system grouped by submission date. Honest bucketing is fine; calling it "family" is not.

Canonical source-trace anchors:
- `experiments/results/public_pr*_intake_*/source/inflate.sh` (extracted from each PR's archive at intake)
- `experiments/results/public_pr*_intake_*/source/inflate.py`
- `experiments/results/public_pr*_intake_*/repo/` (full PR clone where available)
- `tools/public_pr_eval_comment_scorecard.py` (eval-comment-derived metadata)
- `reverse_engineering/` (curated public-submission deconstruction)

### Phase 2: Filter 194/55 anchor set through canonical custody validators

Run the canonical custody chain on every anchor:
1. `tac.frontier_scan.collect_all_anchors(repo_root)` — collects from `.omx/state/continual_learning_posterior.jsonl` + `.omx/state/active_lane_dispatch_claims.md` + `.omx/state/modal_call_id_ledger.jsonl`
2. `tac.frontier_scan.QUALIFYING_HARDWARE` filter (Catalog #316) — only Linux x86_64 / T4 / A10G / A100 / 4090 / H100 / L40S anchors count for contest leaderboard claims
3. `tac.continual_learning.validate_custody_verdict(...)` (Catalog #127) — refuses (axis, hardware_substrate) mismatches; rejects `tag_axis_mismatch` / `cpu_tag_non_gha_linux` / `macos_substrate` / `advisory_grade`
4. `tac.continual_learning.is_promotable_exact_cuda_evidence(...)` if claim is CUDA-axis
5. Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 (`check_macos_cpu_advisory_not_promoted_without_linux_verification`): macOS-CPU anchors are NEVER authoritative for `[contest-CPU]` — they're `[macOS-CPU advisory]` only

Output structured table for each of 194 anchors:
| anchor_id | axis | hardware_substrate | evidence_grade | custody_verdict | counts_as_authoritative |

Anchors that SURVIVE all checks are the authoritative G1 evidence base. Anchors that fail are PRESERVED in the report (per "no signal loss") but tagged `[advisory only]` / `[macOS-CPU advisory]` / `[MPS-PROXY]` / `[diagnostic-only]` per their actual axis × hardware × grade triple.

### Phase 3: Tag every "family" claim with verified-against source-trace

Per Catalog #287 evidence-tag discipline, every "family" claim must carry `[verified-against:<source-trace-path>]`. Examples:

**PASSES** (genuine family with empirical source-trace):
```
**PR101/PR102/PR103 family** [verified-against:experiments/results/public_pr101_intake_*/source/inflate.sh@sha:abc123 + public_pr102_intake_*/source/inflate.sh@sha:def456 + public_pr103_intake_*/source/inflate.sh@sha:ghi789 — identical inflate runtime + shared archive-grammar header bytes 0-32 + PR102 body cites "based on PR101"]
```

**HONEST BUCKET** (no source-trace; downgrade to bucket language):
```
**Score-band [0.193, 0.196] bucket** [verified-against:tools/public_pr_eval_comment_scorecard.py — grouping derived from PR-comment eval metadata only; no architectural lineage source-trace verified]
```

**REJECTED** (cargo-culted "family"):
- `**PR106 family**` (bare) — REJECTED per Catalog #287 + Catalog #323
- `**PR106 family** — grouped by similar score` — REJECTED (similarity-by-score is not architectural lineage)

### Phase 4: Apply Assumption-Adversary HARD-EARNED-vs-CARGO-CULTED classifier

Per Catalog #292 + the HARD-EARNED-vs-CARGO-CULTED addendum framework: every "family" claim → either HARD-EARNED-VERIFIED (with source-trace + Catalog #323 Provenance) OR CARGO-CULTED-PENDING-EMPIRICAL (downgraded to "bucket" until verified).

The Assumption-Adversary surface for Codex (per Council Hierarchy v2 sextet pact extension): for each "family" claim in the G1 report, explicitly answer "is the family-relationship assumption HARD-EARNED (from source-trace) or CARGO-CULTED (inherited from eval-comment grouping)?"

### Phase 5: Wrap G1 reranker score-claims in canonical Provenance contract

Per Catalog #323 (`check_no_score_claim_without_canonical_provenance`) META-class umbrella: every score-claim row emitted by G1's reranker MUST be wrapped in `tac.provenance.build_provenance_*` canonical builders.

For G1 specifically:
- The CPU-axis re-rank scores carried in the report ARE score claims
- Each must be built via `tac.provenance.builders.build_provenance_for_contest_cuda_anchor` OR `build_provenance_for_contest_cpu_anchor` OR `build_provenance_for_advisory` depending on the source anchor's axis + hardware
- The aggregate "194/55 evidence base" must be wrapped in `build_provenance_aggregate` so the WORST-grade rule applies (if even one anchor is `advisory_grade`, the aggregate cannot be promoted)
- The auto-rejected rows surface as evidence-of-discipline; the report should cite the rejection counts (e.g. "94 of 194 anchors survived canonical custody validation; 100 retained as advisory-only signal not promoted to leaderboard claim")

### Phase 6: Register strengthened verdict to probe-outcomes ledger

Per Catalog #313 (`check_dispatch_target_has_no_predecessor_adjudicated_outcome`): the G1 verdict `FRONTIER_STABLE_VIA_RE_RANK` BROADENED from 87→194 evidence base is itself an empirical anchor that should be queryable across sessions.

```python
from tac.probe_outcomes_ledger import register_probe_outcome

register_probe_outcome(
    probe_id="g1_cpu_axis_re_rank_BROADENED_194_55_20260518",
    verdict="FRONTIER_STABLE_VIA_RE_RANK_BROADENED",
    status="adjudicated",
    rationale_path="experiments/results/g1_cpu_axis_re_rank_<utc>/report_v2.json",
    expires_at_utc="<+30 days>",
    agent="codex",
    notes="Broadened from 87/22 anchors (original) to 194/55 anchors (post-schema-fix). Survived custody validation: N_clean of 194 anchors. Family-vs-bucket distinction applied per Catalog #287 + #323. Strengthens FRONTIER_STABLE per Apples-to-apples discipline rule 4.",
)
```

This prevents future agents from re-running G1 unnecessarily AND surfaces the strengthened-negative as queryable signal for the cathedral autopilot ranker.

## DISCIPLINE

- Catalog #229 premise verification BEFORE editing: read actual public-PR intake artifacts, NOT inferred from eval-comment metadata
- Catalog #287 evidence tags on every "family" claim
- Catalog #127 custody validator routing for every anchor
- Catalog #316 frontier scan canonical helper for the qualifying-hardware filter
- Catalog #192 macOS-CPU advisory not promoted
- Catalog #323 canonical Provenance contract for every score-claim row
- Catalog #292 per-deliberation Assumption-Adversary discipline
- Catalog #313 probe outcomes ledger registration
- Catalog #117/#157/#174 commit serializer with POST-EDIT working-tree sha256

## EXIT CRITERIA

- [ ] Phase 1: per-bucket source-trace classification table lands in updated G1 report
- [ ] Phase 2: 194/55 anchor set filtered through canonical custody validators; structured output
- [ ] Phase 3: every "family" claim either tagged `[verified-against:...]` OR downgraded to "bucket"
- [ ] Phase 4: Assumption-Adversary HARD-EARNED-vs-CARGO-CULTED per-family-claim verdict
- [ ] Phase 5: G1 score-claims wrapped in canonical Provenance contract; aggregate WORST-grade rule applied
- [ ] Phase 6: strengthened verdict registered to `.omx/state/probe_outcomes.jsonl`
- [ ] Memory entry `feedback_g1_authority_upgrade_metadata_bucket_to_family_landed_20260518.md`
- [ ] codex_persistent_session_state row appended

## SISTER COORDINATION

This directive INTEGRATES with the in-flight G1 work (commit `8ebea02ef` G1 IMMEDIATE-EXECUTION routing). Codex's update suggests Phase 1 is already in motion ("Next I'm updating the report/memos to stop calling metadata buckets 'families'"). This directive ROUTES the remaining 5 phases as a coherent unit so the discipline lands all-the-way-through, not just at the label-surface.

Sister of: META-audit (commit `e86ca6d0c`) + cargo-cult burn-down supplement (commit `fb102933b`) + STRICT preflight gate routing (commit `fb102933b`) + cargo-cult-audit backfill sweep (commit `2f867ec0c`).

The 6-phase protocol IS the canonical operationalization of "establish authority" the operator asked about. Output of Phase 5 specifically (Provenance contract wrapping) makes the authority-vs-bucket distinction STRUCTURAL — the Provenance contract auto-rejects axis/hardware mismatches at construction time, so a future label-overstatement bug cannot recur silently.

## OPERATOR-FACING NOTE

Codex is doing the RIGHT thing. This directive RAISES the discipline from label-cleanup to structural-authority-via-canonical-Provenance. After Phase 5 lands, the G1 report's "family" language is auditable via `tac.provenance.audit_score_claim_dict` and refused by Catalog #323 STRICT preflight if mis-tagged. After Phase 6 lands, the strengthened verdict is queryable across sessions per the Catalog #313 4-layer pattern.

Per operator standing directive 2026-05-18 "all operator decisions approved" + "continue with all in context and continue feeding the queue as it returns" — this directive feeds Codex's persistent /goal LOOP queue without consuming Claude subagent slots (both currently saturated with DYNAMIC + SYSTEMATIC RECLAIMABILITY).

— Main-Claude 2026-05-18 (operator-query-driven routing per Codex's G1 update + canonical authority-establishment via existing CLAUDE.md non-negotiables)
