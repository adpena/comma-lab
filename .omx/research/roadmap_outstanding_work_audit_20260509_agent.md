# Roadmap Outstanding Work Audit - 2026-05-09

<!-- generated_at: 2026-05-09T10:54:57Z -->
<!-- evidence_grade: roadmap_audit; no score claim; no code edits -->

## Scope

Audited current roadmap/state surfaces on `main`:

- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/lane_registry.json`
- `reports/latest.md`
- `reports/phase_a_pareto_20260508.md`
- latest relevant `.omx/research/*20260509*.md` for HNeRV/PR lessons, A1,
  A5, PARADIGM-dezeta, Track 4, and Lane 12-v2.

This audit is evidence-grade aware. `[contest-CUDA]` and `[contest-CPU]` are
kept as distinct axes. macOS CPU, CPU-prep, source-forensics, and prediction
rows are planning signals only unless paired archive/runtime custody exists.

## Current Anchors And Outstanding Work

| Area | Current evidence | Outstanding work / blockers |
|---|---|---|
| HNeRV / public PR lessons | Source/binary forensics, not a new score claim. PR100/101/102/103 share byte-identical decoder weights from PR95; the missing leverage is PR95's training stack, eval-roundtrip inner loop, differentiable `rgb_to_yuv6`, exact-archive checkpoint selection, per-pair sidecars, and inflate-time constants. Sources: `hnerv_forensics_critical_findings_for_a1a9359d_20260509.md`, `hnerv_leaderboard_binary_forensics_dossier_20260509.md`, `hnerv_lessons_docs_adversarial_review_20260509_codex.md`. | Reproduce PR95 training stack or run a cheaper parity smoke before treating new HNeRV-derived substrates as comparable. Add a renderer-trainer gradcheck proving nonzero `d_seg` and `d_pose` gradients through scorer preprocess. Verify A1-vs-PR101 byte identity with section SHA manifests and same-archive runtime-framing controls before more cross-archive composition. |
| A1 score-gradient | Paired exact anchor exists: `0.19284757743677347` `[contest-CPU]` on GHA Linux x86_64 and `0.2263520234784395` `[contest-CUDA]` on Modal T4, archive `87ec7ca5...492b5`, `178262 B`. Best-proxy refire duplicated the same archive/score. Bias V2 half-magnitude exact CPU regressed to `0.194295755690`. Sources: `phase_a1_latent_aligned_contest_cpu_anchor_20260509_codex.md`, `phase_a1_best_proxy_modal_harvest_20260509_codex.md`, `a1_inflate_bias_sweep_exact_cpu_review_20260509_codex.md`. | Do not relaunch the same A1 basin or spend duplicate CPU evals. Next A1 work must emit a byte-different runtime-consumed packet via score-domain/SegNet-boundary validation, same-archive runtime-constant coordinate search, or full sidecar resample. Future GHA CPU sweeps are gated by the in-flight directive fixing prefix-unsafe submission matching in `tools/dispatch_cpu_eval_via_github_actions.py`. |
| A5 frame-conditional q-bits | Runtime packet path is real, but evidence is macOS CPU advisory. Best local A5 point is binary q-bit side-info at `178095 B`, advisory `0.20101191630821824`, still about `0.00816` worse than A1. Channel DP qsum200 is byte-positive at `178014 B` but worsens to `0.2016517950045961` advisory. Sources: `a5_binary_qbits_sideinfo_improvement_20260509_codex.md`, `a5_channel_qbits_dp_schedule_negative_20260509_codex.md`, `a5_segnet_boundary_margin_scalar_negative_20260509_codex.md`. | No exact eval spend for current scalar/global/latent-MSE schedules. Reactivate only with local score-domain channel allocation, local boundary placement inside pairs, or training-time q-bit noise that changes archive SHA and moves advisory within `0.001` of A1 or saves enough bytes to pay for measured distortion. |
| Track 4 UNIWARD/STC/Hessian on A1 | v1 is `[contest-CPU]` negative: best `blocks4_7bit` scored `0.19869389522684905` vs A1 `0.19284757743677347`, saving only `359 B`. This retires the measured v1 config only. A bug-class fix landed per memo: score-gradient saliency option, cliff-zone gate, strict preflight, 3 clean review, and 30 tests. Sources: `track4_uniward_stc_hessian_a1_contest_cpu_anchor_20260509.md`, `track4_reactivation_options_for_council_20260509.md`, `track4_bug_class_fix_3_clean_pass_review_20260509.md`. | Do not spend CUDA on v1 candidates. If continuing Track 4, use `--saliency-source score_gradient`, prefer full 600-pair saliency for authoritative ranking, and require cliff-zone clearance plus dispatch claim. Registry/report surfaces are stale: they still show strict_preflight/three_clean false or pending GHA for `blocks4_7bit` even though the contest-CPU negative and fix memo exist. |
| PARADIGM-dezeta | Phase 1 produced local targets and A1 EMA smoke only: `[empirical_planning; local CPU sanity loop]`, no score claim, no dispatch. Phase 2/3 review says ready for exact eval dispatch is `false`. Sources: `deltaepszeta_phase1_targets_and_smoke_20260509_codex.md`, `paradigm_dezeta_phase2_3_plan_review_20260509_codex.md`. | Blocked on a runtime-consumed dezeta archive compiler, score-domain/boundary-aware surrogate, calibrated typed artifacts for T7/T8/T10, and byte-different packet custody. Target tables, LEPR/ZETA blobs, and checkpoints outside a scored archive are not dispatchable. Do not apply the HNeRV CPU/CUDA gap to this new architecture class without paired anchors. |
| Lane 12-v2 NeRV-as-renderer | Phase A design/scaffold only, `[predicted]` score band, no empirical score. It correctly rescope v1 from mask logits to full RGB renderer and declares archive grammar. Source: `lane_12_v2_nerv_as_renderer_phase_a_design_20260509.md`. | Phase B is blocked by self-contained contest runtime, differentiable scorer-preprocess gradcheck, PR95/PR100 training-stack parity or deliberate deviation, exact packet builder plus no-op proof, and paired `[contest-CUDA]`/`[contest-CPU]` once shippable. Keep Lane 12-v1 deferred/superseded, not killed. |

## Dispatch And State Blockers

`tools/claim_lane_dispatch.py summary` reports:

- active nonterminal: `pr107_apogee_cpu_auth_eval_linux_x86_64`
  (`PRESTAGE:pr107-cpu-eval-lightning-20260508-PLACEHOLDER`,
  `pending_authorization`);
- stale nonterminal: `apogee_int6_contest_cuda_anchor`
  (`PRESTAGE:apogee-int6-cuda-anchor-20260508-PLACEHOLDER`);
- stale nonterminal: `pr101_admm_step6_no_dead_k`
  (`PRESTAGE:admm-no-dead-k-20260508-PLACEHOLDER`).

Before any new dispatch, close or explicitly supersede the two stale
preauthorization rows and re-check the active PR107 CPU eval row.

The 2026-05-09 round-2 directive adds three cross-cutting blockers before
trusting concurrent harvest or posterior routing:

1. `tools/dispatch_cpu_eval_via_github_actions.py` has a prefix-unsafe
   submission-name matching failure class. All future GHA CPU rows need
   post-fix custody verification.
2. `src/tac/continual_learning.py` must validate tag, axis, hardware substrate,
   and metadata together; tag-only `[contest-CPU]`/`[contest-CUDA]` acceptance
   is unsafe.
3. Posterior writes need a locked transactional update path to avoid dropping
   empirical anchors under parallel harvest.

The companion coordination ledger
`gha_dispatcher_high1_fix_status_20260509.md` currently says the GHA
dispatcher HIGH 1 fix is `IN-FLIGHT`, not `LANDED`; treat future concurrent
GHA CPU dispatches accordingly.

## Stale Or Inconsistent Roadmap Entries

- `reports/latest.md` has fresh May 9 content but its title still names the
  May 4 PR106 adapter status. Later sections correctly supersede PR106/PR106x
  with PR103-on-PR106 as the active local HNeRV rate anchor.
- The old "Updated Next Queue" still lists Lane Omega-W-V3, int5, int6, and
  SJ-KL as a May 4 dispatch matrix. That queue is stale relative to later
  evidence: int5 is deferred by basin-parity failure, int6 is the safer
  candidate but was blocked by capacity/credit/preauth state, and PR106-based
  work must be labeled predecessor-substrate unless rebuilt against the current
  PR103-on-PR106 anchor.
- `reports/latest.md` itself flags `docs/paper/04_results.md` as a stale PR95
  frontier table. Do not publish from that paper table without refresh.
- `.omx/state/lane_registry.json` is stale for Track 4: the registry still
  says `strict_preflight=false`, `three_clean_review=false`, and "pending GHA
  dispatch" for `blocks4_7bit`, while the May 9 memos record `[contest-CPU]`
  negative evidence and a landed bug-class fix/3-clean review.
- `.omx/state/lane_registry.json` is inconsistent for T9: it records
  `lane_t9_cross_archive_substrate_composition` as L1 with implementation
  complete, but operator decision/directive says DEFER T9 or rescope to
  single-axis A1 branching, with no current cross-archive implementation work.
- `.omx/state/lane_registry.json` now marks `lane_12_nerv_mask_codec`
  `research_only=true` with Lane 12-v2 supersession criteria, but the same row
  still carries Level 2 historical gates and notes saying "Audit: Level 1".
  Roadmaps should treat it as forensic/superseded, not an active score lane.
- Repo-root `MEMORY.md` is absent, while AGENTS requires reading top-10
  `MEMORY.md` entries. Agents currently have to rely on external operator
  memory plus current ledgers; this is a process-surface inconsistency.

## Near-Term Audit Recommendation

Highest priority is not another broad strategy pass. Close the state surfaces
that can misroute work:

1. Fix/verify the GHA CPU dispatcher custody bug before new A1/A5/Track 4 CPU
   sweeps.
2. Update lane registry entries for Track 4, T9, and Lane 12-v1/v2 so the
   deduplication layer matches May 9 evidence.
3. Reconcile `reports/latest.md` by replacing the stale May 4 queue with the
   current blocker list above.
4. Only then spend eval budget on byte-different, runtime-consumed packets with
   exact claim rows and paired-axis custody where the result will affect
   frontier or medal-band decisions.
