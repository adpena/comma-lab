# Codex Findings - Latest Design Memo Implementation Intake

**UTC:** 2026-05-20T03:10:38Z  
**Owner:** codex  
**Lane:** `lane_codex_latest_design_memo_implementation_intake_20260520`  
**Scope:** latest `.omx/research` design memos, routing directives, and Codex task-status rows with implementation obligations as of this pass.  
**Score claim:** none  
**Promotion eligible:** false  
**Ready for dispatch:** false

## Reviewed Surfaces

- `.omx/research/v1_faiss_v4_probe_plus_v8_design_landed_20260519.md`
- `.omx/research/v1_faiss_v8_learned_compression_faiss_design_20260519.md`
- `.omx/research/pr101_fec6_frontier_packetir_matrix_20260519_codex.md`
- `.omx/research/codex_findings_pr101_packetir_frontier_cpu_authority_20260519T220244Z_codex.md`
- `.omx/research/sigma_15_per_substrate_sweep_design_20260519T211927Z_codex.md`
- `.omx/research/codex_routing_directive_codec_py_refactor_with_byte_identity_verification_20260519T211500Z.md`
- `.omx/research/codex_findings_catalog204_item4_dispatch_refused_by_probe_outcome_20260519T211313Z_codex.md`
- `.omx/state/canonical_task_status.jsonl`
- `.omx/state/subagent_progress.jsonl`

## Findings

### 1. V1 Faiss V8 Is Delegated, But It Is Not Dispatchable Or Scaffold-Ready As-Written

The V8 design memo is the newest Claude design surface that actually names new Codex-buildable files:

- `experiments/train_substrate_v8_learned_compression_faiss.py`
- `submissions/v8_learned_compression_faiss/inflate.py`
- `submissions/v8_learned_compression_faiss/inflate.sh`
- `.omx/operator_authorize_recipes/substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml`

Those files do not exist yet. The memo also marks the design surface as `research_only: true` and `dispatch_enabled: false`; therefore the only safe Codex next step is a $0 implementation-readiness pass or scaffold patch, not provider spend.

Premise/API audit found that most cited canonical helpers exist, but three memo references need correction before implementation:

- `tac.substrates._shared.score_aware_common.score_pair_components` is stale. The importable helper is `tac.substrates.score_aware_common.score_pair_components` / `score_pair_components_dispatch`.
- `compute_pq_mi_verdict` is not a reusable `tac.*` helper; it currently lives in `tools/probe_atw_v2_1_faiss_pq_disambiguator.py` and is consumed by the V4 probe tool.
- `tac.provenance.build_provenance_for_contest_archive_byte_member` is not importable. The canonical available builder is `tac.provenance.build_provenance_for_archive_member`.

Implementation verdict: **PROCEED_TO_PREMISE_FIX_AND_SCAFFOLD**, no dispatch, no score claim. The first code patch should either promote/rehome the MI verdict helper into reusable `tac` code or explicitly keep the V8 smoke's MI computation tool-local with no false canonical-helper claim.

### 2. V1 Faiss V4 Probe Is Complete Evidence, Not A Codex Build Queue

The V4 landing memo closes the free probe and V8 design-writing routables. It records V4 as `[macOS-CPU advisory only]`, with no contest score authority. It does not authorize V8 training or Modal dispatch from this Codex loop.

Implementation verdict: **NO_REOPEN**. Consume the V4 evidence as a premise for V8 only.

### 3. PR101/FEC6 PacketIR Has One Active Local Closure Item, Not A PR Submission Runtime Edit

Canonical task status still shows `operator_packetir_compiler_pr101_fec6_20260519::IDENTITY_AND_QUEUE` as `in_progress`. The existing PacketIR matrix says the safe remaining local item is `local_identity_profile_smoke`; exact eval remains blocked on candidate queue and operator authorization.

Current active sister-agent map includes `claude_slot_rr_d3_compliance_gate_clearance_20260520`, touching the PR101/FEC6 submission directory and compliance gate. Codex should avoid further edits to:

- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/`
- `scripts/pre_submission_compliance_check.py`
- PR body/release packet files

Implementation verdict: **LOCAL_ONLY_AFTER_SISTER_CLEARANCE**. If picked up next, run or finish a local identity/profile smoke against the PacketIR artifacts without modifying the PR runtime surface.

### 4. Codec Refactor Directive Is Already Completed

The codec refactor directive was registered from `.omx/research/codex_routing_directive_codec_py_refactor_with_byte_identity_verification_20260519T211500Z.md`. Canonical task status records completion via commits `11500bbfe`, `c5a01413e`, and `184340adb`.

Implementation verdict: **NO_REOPEN**. Do not re-refactor `src/codec.py` while the PR compliance/runtime surface is active. Only regenerate byte-identity proof after the current PR runtime stabilizes.

### 5. Sigma=15 Sweep Design Is A Small Future Patch, Not A Current Dispatch Lane

The sigma design memo already corrected the grid and hardened `scripts/remote_lane_fr_mm_sigma_sweep.sh`. Remaining buildable items are:

- add `lut_sigma` plumbing to SegMap fixed-soft/LCT paths;
- add Szabolcs builder/inflate sigma config or prebuilt-LUT discipline;
- keep SCPP on a separate integer block-FP cutoff sweep.

Implementation verdict: **LOWER_PRIORITY_PATCH_QUEUE**. This should not displace V8 premise/scaffold work or PacketIR local closure unless the operator explicitly prioritizes sigma sweeps.

### 6. Paid Dispatch Items Remain Blocked

Catalog #204 ITEM_4 is blocked by a same-day Catalog #313 predecessor `DEFER` on `harvest_e8_sgld_1_instant_crash_20260519`. The live task ledger also still contains paid-dispatch rows for C6/Z6/STC follow-ons that require fresh claim/preflight/operator-authority before spend.

Implementation verdict: **DO_NOT_DISPATCH_FROM_THIS_REVIEW**. A Codex implementation pass may build missing local guards or fresh probes, but should not bypass the predecessor ledger or launch providers without explicit operator authorization.

## Priority Queue For Codex

1. **V8 premise-fix scaffold:** correct stale helper references and build the minimum importable V8 trainer/inflate scaffold behind `research_only` / `dispatch_enabled=false`; include focused tests that prevent phantom-helper authority.
2. **PacketIR local closure:** once the active PR compliance sister lane is clear, finish the local identity/profile smoke and mark `operator_packetir_compiler_pr101_fec6_20260519::IDENTITY_AND_QUEUE` terminal in canonical task status.
3. **Sigma small patches:** add explicit `lut_sigma` plumbing for SegMap fixed-soft/LCT and separate SCPP integer cutoff sweep only after higher-EV frontier surfaces are not blocking.

## Authority Boundary

This review is an intake and adversarial-premise artifact. It does not authorize paid dispatch, exact eval, PR submission, promotion, rank/kill decisions, or score claims. The only immediately safe implementation work is local, additive, and guarded by the helper/path corrections above.
