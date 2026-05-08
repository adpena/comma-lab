# Cross-Paradigm Dispatch Readiness Review - Worker D - 2026-05-08

Scope: adversarial claim-hygiene and next-dispatch-readiness review for
cross-paradigm, Omega-OPT, and score-lowering evidence rows.

Inputs inspected:

- `reports/cathedral_autopilot_evidence.jsonl`
- `reports/omega_opt_linear_stack_packet_20260508.json`
- `.omx/research/omega_opt_anchor_discipline_20260508_codex.md`
- `.omx/research/monolithic_packet_candidate_bridge_20260508_codex.md`
- `.omx/research/frontier_monolithic_budget_assumption_audit_20260508_codex.md`
- `.omx/research/frontier_dispatch_readiness_audit_20260508_codex.md`
- `.omx/research/lightning_round3_harvest_and_runtime_failures_20260508_codex.md`
- `.omx/research/lossy_coarsening_exact_cuda_adversarial_review_20260508_worker_b.md`
- `.omx/research/lossy_coarsening_exact_cuda_result_review_20260508_codex.json`
- relevant raw manifests under `reports/raw/` and monolithic candidate manifests under
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/`

No dispatch was attempted. No code was changed.

## Current Scoring Floor

The active local A++ HNeRV rate anchor is PR103-on-PR106:

- archive:
  `experiments/results/pr103_repack_pr106_standalone_20260507/exact_eval_static_release_surface/archive.zip`
- archive bytes: `185578`
- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- exact T4 score: `0.20898105277982337` from
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
- non-rate budget already consumed by distortion terms:
  `0.08541228`

This floor supersedes PR101-only comparisons and PR106x-lowlevel rate-only
comparisons. A rate-only candidate above `185578` bytes is not worth exact CUDA
spend unless it intentionally changes scorer-visible output and has component
risk gates.

## Claim-Hygiene Findings

1. `reports/omega_opt_linear_stack_packet_20260508.json` is correctly
   fail-closed. Even when materialized from a source plan containing predicted
   `0.130` score language, the packet has `score_claim=false`,
   `promotion_eligible=false`, `rank_or_kill_eligible=false`,
   `ready_for_exact_eval_dispatch=false`, and blockers for missing archive
   bytes/SHA, runtime packet, inflate path, exact CUDA JSON, 1:1 anchor, and
   per-layer runtime-consumption proof.

2. `.omx/research/omega_opt_anchor_discipline_20260508_codex.md` has the
   right control-plane semantics: every nested Omega-OPT score is a hypothesis
   until a matching byte-closed packet and exact CUDA auth eval exist. The
   ledger correctly preserves the designs without allowing promotion, ranking,
   dispatch, or family kill.

3. `reports/cathedral_autopilot_evidence.jsonl` is mixed historical state. The
   later rows mostly carry fail-closed fields, but early rows still contain
   stale labels such as `FALSIFIED`, `STILL-FALSIFIED`, `CONDITIONAL`, and
   proxy-positive phrases without full row contracts. Consumers must prefer the
   latest superseding row with explicit `score_claim=false` /
   `ready_for_exact_eval_dispatch=false` fields, not the earliest row for a
   technique.

4. The only score-grade row in the reviewed JSONL is
   `lossy_coarsening_analytical`, and the final reviewed status is
   `contest-CUDA A-negative`: exact T4 score `0.351718793322788`, archive
   bytes `156404`, archive SHA
   `ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28`.
   This retires only the measured `per_tensor_K_budget=0.05` direct PR101
   coarsening config. It is not a lossy-coarsening family kill.

5. The monolithic bridge work is the correct substrate direction. PR101 and
   PR106-style HNeRV frontier packets are single-member ZIP archives with
   parser-proven internal sections. Valid future stack work must mutate
   parser-proven sections, record old/new section SHA-256s and archive SHA-256s,
   prove the runtime consumes the changed bytes, then run exact CUDA auth eval.
   ZIP-member-level mask/pose/component budgets are invalid on these packets.

## Closest Exact-Evaluable Candidates

| Rank | Candidate | Current evidence | Why it is close | Blocking issue | Readiness verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | `arch_shrink_x0.4_lightning` | CPU byte anchor `83571` bytes; active Lightning claim `arch-shrink-x0-4-lightning-20260508T024304Z` | It is already in the exact-eval pipeline through another agent's active claim. Its rate budget leaves a large non-rate break-even margin if the trained packet avoids component collapse. | Do not duplicate dispatch. Poll/harvest only; if exact JSON lands, enforce archive bytes/SHA/runtime-tree custody and component recomputation. | Closest operationally; blocked on existing active job harvest. |
| 2 | `cross_paradigm_admm_continuous_k_then_op1_finalizer` | CPU faithful cross-paradigm row, `137531` bytes, `rel_err_pre_finalizer=0.0415`, score-affecting bytes changed | It is the best real-substrate cross-paradigm byte anchor. Against the current floor, `137531` bytes gives rate `0.091576248` and allows non-rate up to `0.117404805`, which is enough margin if distortion stays near PR101-level. | No byte-closed archive, no runtime consumer, no exact CUDA. The lossy-coarsening A-negative proves rel_err alone is not a score proxy. | Best next build candidate after adding monolithic packet closure and score-aware tensor safeguards. |
| 3 | `pr106x_lgblock16_monolithic_section_candidate` / monolithic bridge path | Built archive `186079` bytes, SHA `866dc135e9168d61fab02b6b1c218c4b1d6eed779154a6dc3095fd05e48024f2`, parser-proven `decoder_packed_brotli` replacement, old/new section SHAs | This is the cleanest archive-construction proof: actual monolithic packet, deterministic rebuilt archive, and section-level custody exist. | Missing runtime-consumption proof and active lane claim. Also it is rate-only and `501` bytes above the active PR103-on-PR106 floor, so dispatching this exact PR106x candidate would waste score wall-clock unless used only as a control or ported onto the active anchor. | Closest structurally, not a score-lowering dispatch target as-is. |

Honorable mention: `joint_admm_x_continuous_lossy_coarsening` has useful
allocation evidence (`153639` bytes at the 0.0386 target, and lower-distortion
`0.01` / `0.02` rows), but the exact lossy-coarsening negative shows that this
family needs score-aware per-tensor weights or a retrain/repair loop before new
exact-eval spend.

## Invalid Or Proxy Traps

1. `omega_opt_linear_stack_post_hoc_composition` at `41303` bytes is a byte
   hypothesis, not a packet. It composes arch shrink, IMP, lossy coarsening,
   and Brotli post-hoc with no retrain, no runtime decoder packet, no
   layer-by-layer consumed-byte proof, and no exact CUDA. The predicted
   `0.130` score must remain non-ranking.

2. Any PR101/PR106 row that implies separate mask, pose, renderer, or component
   ZIP members is invalid under the monolithic packet finding. The actionable
   surfaces are parser-proven internal sections such as PR101 `decoder_blob`,
   `latent_blob`, `sidecar_blob` and PR106 `decoder_packed_brotli`,
   `latents_and_sidecar_brotli`, or a new charged sidecar/runtime packet.

3. `lossy_coarsening_analytical` proxy-positive language is now calibrated by
   an exact negative. The CPU/MPS proxy row said byte-lower than PR101 and
   carried a predicted `[0.18, 0.22]` band, but exact CUDA scored
   `0.351718793322788`. Future coarsening rows cannot dispatch from rel_err
   alone; they need score-aware tensor sensitivity, lower-risk budgets, retrain
   or repair, byte-closed runtime packet, and exact CUDA.

4. HStack/VStack, per-tensor codec choice, and Joint-ADMM rows remain
   planning-only unless the emitted bytes become real monolithic section
   replacements. `bytes_out`, CSR envelopes, codec IDs, or materialized blobs
   outside a consumed archive section are not dispatchable.

5. PR106x lgblock16 is a valid archive-construction control but an invalid
   score-lowering target as-is. It is rate-only, not below the `185578` byte
   active floor, and lacks runtime proof plus lane claim.

## Next Tranche Recommendation

Build one monolithic packet closure and floor gate, then use it as the only
path from CPU/Omega/cross-paradigm byte anchors to exact dispatch.

The gate should take a materialized replacement payload and refuse readiness
unless all of these are true:

- target archive is the active A++ floor or an explicitly justified
  scorer-changing substrate;
- target section is parser-proven with offset, length, old bytes, and old
  SHA-256;
- replacement bytes exist on disk and are not a no-op;
- rebuilt archive bytes and SHA-256 are recorded;
- changed section SHA-256s are echoed by an actual runtime/inflate log;
- runtime proof is bound to the candidate archive SHA and command/log SHA;
- an active Level-2 dispatch claim is exported and matches lane/job identity;
- static compliance passes;
- rate-only candidates above `185578` bytes are refused before GPU spend;
- CPU/MPS/proxy rows cannot set `ready_for_exact_eval_dispatch=true`.

This single guard/build path reduces score wall-clock risk more than another
standalone proxy sweep because it converts the best CPU anchors into actual
contest packets while blocking the three recurring waste modes: non-consumed
bytes, monolithic-layout violations, and rate-only above-floor dispatches.
