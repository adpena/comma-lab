# Codex Routing Directive — Task Triage Batch — 2026-05-20T12:06:07Z

> Companion to `.omx/research/task_triage_inventory_20260520T120607Z.md`.
> Lists 2 CODEX-routed tasks with canonical `codex /goal` prompts.
> **Operator fires `codex /goal` independently — this directive memo IS the
> input artifact.**

## Routing rationale (per task brief Rule 2)

Both CODEX-routed tasks (#11 + #12 from triage inventory) qualify because:
- Deep research / premise revision / extractor extension needed
- Output is design memo + canonical-helper extension (NOT operator-decision-blocking code edits)
- Multi-hour wall-clock with xhigh reasoning benefit
- No urgent operator dependency (Codex /goal runs autonomously)

## Codex Goal 1 — OP-SYN-1 master-gradient 6-archive extension (Triage task #11)

**Task ID**: `codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518::OP_SYN_1`

**Why-Codex-not-subagent**: extending the master-gradient extractor to handle 3 missing archive families (DP1 deterministic tensor-span serializer + PR106 format0d operator-response projector + PR107 latent codec Jacobian) requires deep research into each archive's grammar + careful Jacobian/projector design that benefits from external-perspective rigor. Codex xhigh has demonstrated capability on master-gradient design (sister landings MG-1 through MG-19 + Cable D consumers 7-14).

**Canonical `codex /goal` prompt (~300 words)**:

```
Goal: extend `tools/extract_master_gradient.py` (and the canonical helper
package `tac.master_gradient`) to handle the 3 missing archive families
blocking OP-SYN-1 per the routing directive
`.omx/research/codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md`.

Pre-flight (Catalog #229):
- read tools/extract_master_gradient.py (current state)
- read src/tac/master_gradient/ canonical helper package
- read recent landing memos:
  - feedback_cable_d_consumers_7_14_solver_wire_in_hooks_123_landed_20260519.md
  - feedback_cable_d_consumers_7_14_hooks_5_6_closure_landed_20260519.md
  - feedback_master_gradient_canonical_helper_landed_with_cathedral_autopilot_wirein_20260517.md
- read sister projector landings:
  - PR106 format0d primary packed-HNeRV projector (lineage commits ef99f489f, 151941885)
  - PR107 latent codec
- read .omx/research/codex_findings_overconservative_authority_bottlenecks_20260519T014528Z_codex.md
  for the partner-active surface boundary

Deliverables:
1. DP1 deterministic tensor-span serializer projector — design memo
   .omx/research/op_syn_1_dp1_projector_design_<UTC>_codex.md
2. PR106 format0d operator-response projector — design memo
   .omx/research/op_syn_1_pr106_format0d_response_projector_design_<UTC>_codex.md
3. PR107 latent codec Jacobian (or zero-grad v2) — design memo
   .omx/research/op_syn_1_pr107_latent_jacobian_design_<UTC>_codex.md
4. Implementation landing of all 3 projectors via canonical
   subagent_commit_serializer with --expected-content-sha256
5. Tests (≥10 per projector covering happy path + edge cases + Catalog #318
   raw-byte-authority-not-landed gate compliance)
6. Catalog #354 master-gradient exploit consumer bundle compliance check
   passes (extension must not break the 8-consumer bundle)
7. Update probe_outcomes.jsonl with PROCEED verdict on each projector after
   tests green
8. Update canonical_task_status.jsonl ledger with OP_SYN_1 status=completed
   only after ALL 3 projectors land + tests pass

Discipline (per CLAUDE.md non-negotiables):
- Catalog #318 raw-byte-authority guard (no byte-level FD over ZIP+entropy-coded packets)
- Catalog #157/#174 commit-serializer --expected-content-sha256 mandatory
- Catalog #287 evidence-tag for every claim
- Catalog #323 canonical Provenance on every score-claim artifact
- 6-hook wire-in declaration per Catalog #125 in landing memo
```

**Estimated cost + wall-clock**: $0 GPU; 4-6 hours xhigh wall-clock per design memo + commit cycle (~12-18h total).

**Operator routing instruction**:
```
codex /goal --skill codex-cli-runtime \
    --input .omx/research/codex_routing_directive_task_triage_batch_20260520T120607Z.md \
    --goal "OP-SYN-1 master-gradient 6-archive extension per Codex Goal 1 section"
```

---

## Codex Goal 2 — B1 rate-attack vector 3 (HEVC pivot + 3 custody artifacts) (Triage task #12)

**Task ID**: `codex_routing_directive_rate_attack_vector_3_b1_contest_video_codebook_20260518::PHASE_1_PROBES`

**Why-Codex-not-subagent**: B1 directive premise was empirically FALSIFIED (canonical video is HEVC/YUV420p, not AV1 as the directive assumed). This requires (a) revising the original directive memo per Catalog #229 premise-verification + Catalog #307 paradigm-vs-implementation falsification, (b) re-deriving the rate-attack lane on HEVC bytes, (c) landing 3 missing custody artifacts. Deep-research scope with cross-domain triangulation (HEVC codec internals + rendered frontier query custody + lane registry schema) suited to Codex xhigh.

**Canonical `codex /goal` prompt (~300 words)**:

```
Goal: revise B1 rate-attack vector 3 directive premise + land 3 missing
custody artifacts per probe-outcomes blockers list:
- directive_av1_premise_false_actual_codec_is_not_av1
- requires_t4_dali_decode_identity
- requires_rendered_frontier_query_custody
- lane_registry_missing_exact_b1_lane

Pre-flight (Catalog #229):
- read .omx/research/codex_routing_directive_rate_attack_vector_3_b1_contest_video_codebook_20260518.md
- read tools/extract_master_gradient.py B1 probe machinery (landed; tests green)
- read upstream/videos/0.mkv via ffprobe to verify HEVC/YUV420p empirically
- read .omx/state/lane_registry.json for any B1-prefix lane entries
- read recent B1 landings:
  - feedback_b1_e7_vq_k_sweep_remediated_dispatch_landed_20260519.md
  - feedback_b1_e8_sgld_trainer_scope_fix_op2_landed_20260519.md

Deliverables:
1. Revised B1 directive memo per CLAUDE.md "Forbidden premature KILL":
   .omx/research/codex_routing_directive_rate_attack_vector_3_b1_HEVC_pivot_<UTC>_codex.md
   - explicitly classify the original AV1-premise FALSIFICATION as
     IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307 (the rate-attack
     paradigm is intact; the codec assumption was wrong)
   - re-derive HEVC rate-attack canvas (B-frame structure + intra-coding
     density + IDR cadence)
   - cite-chain to original directive + Catalog #307 paradigm-vs-implementation
     non-negotiable
2. T4 DALI decode identity probe — land probe script + record outcome via
   tac.probe_outcomes_ledger.register_probe_outcome
3. Rendered frontier query custody — design + land canonical helper for
   querying the current frontier per Catalog #316
4. Lane registry exact-B1-lane entry — add via tools/lane_maturity.py
   add-lane lane_b1_hevc_rate_attack_phase_1 --name "B1 HEVC rate-attack
   Phase 1 probes" --phase 1
5. Update probe_outcomes.jsonl with PROCEED verdict on PHASE_1_PROBES once
   all 3 custody artifacts land + tests green
6. Update canonical_task_status.jsonl ledger with status=completed

Discipline:
- Catalog #287/#323/#343/#229/#292 standard
- Catalog #307 paradigm-vs-implementation classification mandatory
- Catalog #308 ≥3 alternative probe methodologies if reactivation criterion
  is on a substrate-class boundary
- Catalog #125 6-hook wire-in declaration in landing memo
- per CLAUDE.md "Forbidden premature KILL": this revises the directive, does
  NOT kill the lane
```

**Estimated cost + wall-clock**: $0 GPU + $0.30-0.50 for T4 DALI decode identity probe if dispatched; 3-5 hours xhigh wall-clock.

**Operator routing instruction**:
```
codex /goal --skill codex-cli-runtime \
    --input .omx/research/codex_routing_directive_task_triage_batch_20260520T120607Z.md \
    --goal "B1 rate-attack vector 3 HEVC pivot + 3 custody artifacts per Codex Goal 2 section"
```

---

## Summary

| # | Task ID | Cost | Wall-clock | Output deliverable |
|---|---------|------|------------|-----|
| 1 | OP_SYN_1 master-gradient 6-archive extension | $0 | 12-18h | 3 design memos + projector impls + tests |
| 2 | B1 rate-attack vector 3 HEVC pivot | $0-$0.50 | 3-5h | revised directive + 3 custody artifacts + lane registration |

**Total operator-routable cost: $0-$0.50.**

Both Codex goals are independent + parallel-safe (no sister overlap). Operator may fire both in parallel (Codex /goal runs disjoint sessions).


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:codex-routing-directive-task-triage-batch-trigger-tokens-describe-task-states-not-new-empirical-finding -->
