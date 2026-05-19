---
schema: council_deliberation_v2
deliberation_id: codex_findings_t3_grand_council_synthesis_20260518T145900Z_codex
topic: "Codex xhigh adversarial review of T3 grand council synthesis memo — contest-axis authority hardening for master-gradient + Provenance discipline"
council_tier: T3
council_attendees:
  - Cicero
  - Contrarian
  - Assumption-Adversary
  - Shannon
  - Yousfi
  - Fridrich
  - MacKay_memorial
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Cicero
    verbatim: "P1: stale master-gradient contest-axis authority — live state carried measurement_axis=[contest-CPU] on a macOS advisory 8-pair subset row; consumers filtering by axis could accidentally treat a diagnostic local row as contest-CPU authority. P1: dead posterior CLI syntax in memo. P2: composition #3 language outruns evidence. P2: sensitivity_mask_aware_quantizr_v1 is a next-action candidate not a landing result. P2: HF Jobs / Trackio publication should stay private/operator-approved. P3: evidence labels should remain inline near frontier/medal-band/submission-ready/auto-promote/score-gap language."
council_assumption_adversary_verdict:
  - assumption: "measurement_axis label alone is sufficient to establish contest authority"
    classification: CARGO-CULTED
    rationale: "contest authority requires JOINT (axis, hardware, pair_count, scored_archive_sha, runtime_custody) verification; label without underlying custody is a phantom-score class instance per CLAUDE.md FORBIDDEN_PATTERNS"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_decisions_recorded:
  - "added tac.master_gradient.contest_axis_authority_violation_reason / is_authoritative_axis_anchor / latest_rejected_contest_axis_anchor_for_archive / unresolved_contest_axis_authority_violations"
  - "tightened authority filter: contest-CUDA requires CUDA/GPU hardware; contest-CPU requires Linux CPU hardware"
  - "append-only correction row re-labels fec6 local subset to advisory grade"
council_decisions_landed_via_catalog: ["Catalog #327"]
---

# Codex Findings: T3 Grand Council Synthesis Adversarial Review - 2026-05-18T14:59:00Z

## Scope

Target memo:
`.omx/research/council_t3_grand_council_synthesis_all_research_eureka_engineering_meta_20260518.md`.

Requested mode: xhigh adversarial bug hunt, research memo review, no cargo-culting,
full-stack integration, CPU and GPU evidence separated, Cathedral autopilot as
canonical consumer.

## Cicero Findings

1. **P1: stale master-gradient contest-axis authority.** The target memo correctly
   caveats the fec6 master-gradient xray, but live state carried
   `measurement_axis="[contest-CPU]"` on a macOS advisory 8-pair subset row.
   Any consumer filtering by axis could accidentally treat a diagnostic local
   row as contest-CPU authority.

2. **P1: dead posterior CLI syntax.** The memo text references
   `tools/extract_master_gradient.py --target local-cpu`, but the live parser
   has no `--target` flag. Per the append-only posterior discipline, this
   remains a documentation/correction-record follow-up; no direct JSONL mutation
   was made in this pass.

3. **P2: Composition #3 language outruns evidence.** Treat it as a hypothesis
   until archive/runtime/component evidence lands on a matching axis.

4. **P2: `sensitivity_mask_aware_quantizr_v1` is a next-action candidate, not a
   landing result.** It can route per-byte planning, but not a score claim.

5. **P2: HF Jobs / Trackio publication should stay private/operator-approved
   until publication hygiene is explicitly routed.**

6. **P3: evidence labels should remain inline near frontier, medal-band,
   submission-ready, auto-promote, and score-gap language.**

## Fix Landed

- Added `tac.master_gradient.contest_axis_authority_violation_reason`,
  `is_authoritative_axis_anchor`,
  `latest_rejected_contest_axis_anchor_for_archive`, and
  `unresolved_contest_axis_authority_violations`.
- Tightened the authority filter after a second xhigh adversarial review:
  contest-CUDA requires CUDA/GPU hardware, contest-CPU requires Linux CPU
  hardware, contest rows require full pair counts plus scored archive SHA/byte
  custody and measurement call/runtime custody, and append-only corrections only
  suppress the same gradient artifact rather than any later row on the same
  archive.
- Appended a canonical correction row that re-labels the fec6 local subset
  anchor as `[macOS-CPU advisory]` instead of mutating historical JSONL.
- Wired Catalog #327 strict preflight:
  `check_master_gradient_contest_axis_requires_authoritative_custody`.
- Wired all live consumers found in this pass:
  Cathedral autopilot rank-time diagnostics, per-pair/aggregate
  `tac.master_gradient_consumers`, per-X byte planner, and canonical DuckDB
  per-byte sensitivity backfill.
- Added DuckDB source-axis/hardware/method/evidence provenance columns so
  advisory sensitivity rows remain diagnostic after materialization.
- Preserved per-pair master gradients as first-class diagnostic signals for
  training, compress-time planning, and guarded inflate-time planning. The
  hard boundary is authority: advisory/subset/local rows can guide design, but
  cannot rank, kill, promote, or claim `[contest-CPU]` / `[contest-CUDA]`
  without matching custody.

## Sidecar Review Closure

- **Avicenna consumer inventory:** no direct unfiltered authority consumers
  remained. Training, compress-time, and inflate-time shims route through
  `load_per_pair_gradient_from_anchor`; Cathedral, per-X, and DuckDB are wired
  directly.
- **Dalton adversarial review:** fixed P1 CPU/GPU axis mismatch, P1
  over-broad correction grouping, P2 DuckDB provenance loss, and P2 Cathedral
  different-axis diagnostics. The remaining token-based source-contract caveat
  is backed by behavioral tests for each discovered consumer path.

## Primary-Source Research Context

- Koh and Liang, *Understanding Black-box Predictions via Influence Functions*
  (arXiv:1703.04730), supports the general pattern that gradient/influence
  signals can identify high-leverage training/example effects, but those
  signals remain estimator-dependent and require careful provenance:
  https://arxiv.org/abs/1703.04730
- HNeRV / implicit neural video representation work motivates using learned
  representation internals for video compression planning, but it does not
  justify crossing evidence axes without exact replay/runtime custody:
  https://arxiv.org/abs/2304.02633

## Evidence Discipline

No score claim, no promotion claim, no paid dispatch, and no direct mutation of
canonical posterior JSONL occurred in this pass. The live false-authority count
for effective master-gradient contest-axis rows was verified as zero after the
append-only correction.
