# Harvest summary — NSCS06 v8 Path B + STC v2 disambiguator Modal T4 smokes (2026-05-16T22:12Z)

**Subagent:** `subagent_a_harvest_smokes_20260516`
**Lane:** `lane_harvest_nscs06_v8_and_stc_v2_smokes_20260516`
**Charter:** SUBAGENT A — HARVEST + apply per-decision-tree verdicts; disjoint scope per Catalog #230 (no source code, no preflight.py, no CLAUDE.md, no substrate trainers; writes only to ledger / posterior / memo Appendices / experiments/results/).

## Executive summary

| Lane | Modal call_id | rc | elapsed | Score | Axis | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| NSCS06 v8 Path B Wavelet | `fc-01KRRJNSXCJ48W4DW53YE02PAE` | 0 | 21.8 min | **58.892** | `[diagnostic-CPU Modal advisory only]` | **DEFER-pending-grand-council** (per design memo Section 18; band `>40`) |
| STC v2 disambiguator | `fc-01KRSB76H04HM4958V2HX2JZZ4` | 25 | 1.6 sec | (no archive) | (no axis) | **UNARBITRATED-pending-infrastructure-fix** (Lane A anchor missing) |

## NSCS06 v8 Path B — per Section 18 decision tree verdict

- Canonical score: **58.892** (pose 30.9 + seg 25.3 + rate 2.7) `[diagnostic-CPU Modal advisory only]` — Modal CPU is NOT 1:1 GHA Linux x86_64 per CLAUDE.md submission-auth-eval mandate
- Archive sha256: `af80dc76b802e9b096cdd3bc5ca412d3ccb8a17ca81fb1382c9c11c2c7ca120a` (4,014,234 bytes; n_samples=600)
- Phantom-score-fix (Catalog #249) correctly auto-redirected output filename CUDA→CPU after `AUTH_EVAL_DEVICE=cpu` was observed; no false `[contest-CUDA]` lie shipped
- Design memo Section 18 decision tree band `>40` → **DEFER pending grand-council symposium ratification of next path**
- NOT KILLED per CLAUDE.md "Forbidden premature KILL without research exhaustion" — only this specific (100ep, T4, CPU auth-eval, single (DB4, depth=2) wavelet config) is retired; CUDA paired eval / different DWT / per-class-conditional CDF / Wyner-Ziv budget sweep remain plausible
- Pose contribution dominates (52%) — consistent with v7 cargo-cult #2 PoseNet-NOT-translation-invariant finding; SegNet contribution (43%) consistent with chroma-strip Y=R=G=B replication anchor (v7 cargo-cult #1)
- Cost: $0.215 actual (Modal T4 21.8 min × $0.59/hr); prediction_in_band=false (predicted $10-20, actual $0.21 because auth-eval ran on CPU not GPU)

## STC v2 disambiguator — UNARBITRATED per Section 2.3 decision tree

- Fast pre-train FATAL: `Lane A anchor archive missing at /tmp/pact/experiments/results/lane_a_landed/archive_lane_a.zip`
- No STC encode pass → no `stcb_bytes` measurement → decision tree did NOT fire
- Per CLAUDE.md "Forbidden premature KILL": single infrastructure failure does NOT exhaust STC v2 research; lane is NOT FALSIFIED
- Codex sister `tools/harvest_modal_calls.py --from-ledger --execute` ALREADY appended terminal claim + cost-band anchor ($0.00027) + cathedral evidence row (`failed_modal_training_rc_25`); per Catalog #230 this subagent did NOT re-do that work
- Reactivation criteria: provide Lane A anchor archive + `renderer.bin` / `optimized_poses.pt` in remote worker contract; verify mounted path before provider launch; Catalog #152 `required_input_files` manifest entry likely the canonical fix

## Artifacts persisted

- NSCS06 v8: `experiments/results/lane_nscs06_v8_path_b_wavelet_modal_t4_20260516T142210Z_modal/artifacts/{contest_auth_eval_cpu.json, contest_auth_eval.json, inflated_outputs_manifest.json, provenance.json, run.log, modal_live_metadata.json, modal_worker_head_ledger.json}`
- STC v2: `experiments/results/lane_stc_v2_disambiguator_modal_t4_20260516T213028Z_modal/artifacts/{modal_lane_*.log, run.log, modal_live_metadata.json, modal_worker_head_ledger.json}`

## Ledger + posterior + cost-band updates (this subagent)

- **Modal call_id ledger** (`.omx/state/modal_call_id_ledger.jsonl`): appended `harvested` event_type row for `fc-01KRRJNSXCJ48W4DW53YE02PAE` with `score=58.892`, `score_axis="diagnostic_cpu"`, `evidence_grade="diagnostic-CPU-Modal-advisory"`, `archive_sha256=af80...c120a`, `archive_bytes=4014234`. STC v2 ledger row already complete (codex sister wrote `failed` event_type).
- **Cost-band posterior** (`.omx/state/cost_band_posterior.jsonl`): appended NSCS06 v8 anchor (`outcome=harvested_partial`, actual_cost_usd=$0.215). STC v2 anchor already complete (codex sister wrote `outcome=failed_dispatch`).
- **Continual-learning posterior** (`.omx/state/continual_learning_posterior.jsonl`): appended NSCS06 v8 ContestResult; **correctly REFUSED** (refused_anchor_count incremented) because `evidence_tag="[diagnostic-CPU Modal advisory only]"` is NOT in `AUTHORITATIVE_TAGS={[contest-CUDA], [contest-CPU], [contest-CPU GHA], [contest-CPU GHA Linux x86_64]}` per Catalog #127. The row is forensically preserved.

## Design memo Appendix updates (this subagent)

- NSCS06 v8: appended **Appendix A** (~110 lines) to `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md` per Catalog #110 + #113 HISTORICAL_PROVENANCE APPEND-ONLY (pre-existing body content UNCHANGED)
- STC v2: appended **Appendix B** (~90 lines) to `.omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md` per same discipline

## 6-hook wire-in completion (per Catalog #125)

| Hook | NSCS06 v8 | STC v2 |
| --- | --- | --- |
| 1. Sensitivity-map contribution | ACTIVE (pose/seg/rate components) | N/A (no encode pass) |
| 2. Pareto constraint | ACTIVE (point at (4.01 MB, 0.253, 95.76) far from frontier polytope) | N/A (no archive bytes) |
| 3. Bit-allocator hook | N/A (fixed per-subband CDFs) | N/A (STC no allocator) |
| 4. Cathedral autopilot dispatch hook | ACTIVE (call_id ledger + cost-band + posterior all consumable) | ACTIVE (codex sister wrote terminal row) |
| 5. Continual-learning posterior update | ACTIVE (REFUSED row; forensically preserved) | N/A (no score; codex sister wrote no-score evidence) |
| 6. Probe-disambiguator | ACTIVE (Section 18 decision tree fired band `>40`) | ACTIVE (Section 2.3 decision tree did NOT fire — UNARBITRATED) |

## Cross-references

- NSCS06 v8 design memo: `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md` (Appendix A appended)
- STC v2 batched memo: `.omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md` (Appendix B appended)
- Sister codex harvest closure: `.omx/research/stc_v2_modal_harvest_no_signal_loss_20260516_codex.md`
- This subagent's checkpoint: `.omx/state/subagent_progress.jsonl` (subagent_id=`subagent_a_harvest_smokes_20260516`)
- This subagent's lane registration (pending): `lane_harvest_nscs06_v8_and_stc_v2_smokes_20260516`

## Disjoint scope honored (per Catalog #230)

- Read-only on source code, preflight.py, CLAUDE.md, substrate trainers
- Append-only on `.omx/state/*.jsonl` (call_id ledger / cost-band posterior / continual-learning posterior) via canonical fcntl-locked helpers per Catalog #128 / #131 / #245
- Append-only on `.omx/research/*_design_*.md` Appendix sections per Catalog #110 / #113 HISTORICAL_PROVENANCE
- New write under `experiments/results/lane_nscs06_v8_path_b_wavelet_modal_t4_20260516T142210Z_modal/artifacts/` + `experiments/results/lane_stc_v2_disambiguator_modal_t4_20260516T213028Z_modal/artifacts/` (DERIVED_OUTPUT per Catalog #113; OK)
- Sister subagents B/C/D/E/F own different scopes; no overlap

## Per CLAUDE.md non-negotiables honored

- **Apples-to-apples evidence discipline:** every score is axis-tagged (`[diagnostic-CPU Modal advisory only]` — explicit non-promotability)
- **Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE:** Modal CPU correctly NOT tagged as `[contest-CPU]` (which requires GHA Linux x86_64)
- **Forbidden premature KILL:** NSCS06 v8 verdict is DEFER (single config retired; reactivation paths enumerated); STC v2 verdict is UNARBITRATED (infrastructure failure, not method exhaustion)
- **Forbidden misleading-directory-name (Catalog #249):** phantom-score-fix correctly fired auto-redirect; loud `[phantom-score-fix]` warning preserved in run.log
- **Modal `.spawn()` HARVEST OR LOSE:** both call_ids harvested within ~7 hours of dispatch (well within 24h TTL)
- **Bugs must be permanently fixed AND self-protected against:** N/A — this is a harvest+verdict subagent, not a fix subagent; the underlying STC v2 infrastructure bug class (missing required input file) is already self-protected by Catalog #152

## Op-routables (ranked by EV/$)

| Rank | Action | Cost | EV |
| --- | --- | --- | --- |
| 1 | Pre-grand-council brief: NSCS06 v8 Path B Modal T4 CPU smoke landed 58.89 [diagnostic-CPU advisory]; pose dominant; recommend pivot to design memo Section 18 Path C hybrid-neural OR re-dispatch with CUDA auth-eval for paired axis | $0 (brief only) | High (informs next substrate selection) |
| 2 | STC v2 reactivation: fix `scripts/remote_lane_substrate_stc_v2.sh` to declare Lane A anchor as required_input_file (Catalog #152 manifest entry); re-dispatch the cheapest CUDA disambiguator ($0.20 per Section 2.3) | $0.20 | High (closes the STC v2 vs AV1 cost question for free) |
| 3 | Council deliberation: NSCS06 v8 verdict `>40` → choose between (a) re-dispatch CUDA paired, (b) Path C hybrid-neural pivot, (c) absorb pose-NOT-translation-invariant finding into next NSCS lineage | $0 (deliberation only) | High |
| 4 | Update lane registry: mark `lane_harvest_nscs06_v8_and_stc_v2_smokes_20260516` at L1 with gates {impl_complete, memory_entry} via `tools/lane_maturity.py mark` | $0 | Medium (registry hygiene) |
