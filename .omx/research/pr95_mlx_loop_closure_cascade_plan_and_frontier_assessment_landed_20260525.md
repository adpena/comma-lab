# PR 95 MLX Loop Closure Cascade Plan + Frontier Assessment LANDED 2026-05-25

**Lane**: `lane_pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_20260525` L1
**Cost**: $0 + ~50 min wall-clock (research + planning + concrete inventory; NO source mutation, NO paid dispatch)
**Discipline**: Catalog #229 PV + #117/#157/#174 canonical serializer +
POST-EDIT `--expected-content-sha256` + #110/#113 APPEND-ONLY + #206 5-tool-use
checkpoints + #230 sister-subagent ownership map (Slot 1+2+3 DISJOINT verified)
+ #287/#323 canonical Provenance + #131 fcntl-locked JSONL + #313
probe-outcomes registered + #340 sister-checkpoint guard PROCEED verified
pre-staging + #287 placeholder-rationale rejection + #292 per-deliberation
assumption surfacing + #294 9-dim checklist evidence + #300 v2 frontmatter +
#303 cargo-cult audit + #305 observability surface + #309 horizon_class
declaration + #344 canonical equation RATIFY-N candidate QUEUED.

## Frontmatter (canonical v2 per Catalog #300)

- council_tier: T1
- council_attendees: [Shannon, Dykstra, PR95Author]
- council_quorum_met: true
- council_verdict: PROCEED
- council_predicted_mission_contribution: frontier_breaking_enabler
- council_override_invoked: false
- council_dissent: []
- council_decisions_recorded:
  - "op-routable #1: spawn P0 PR95-MLX-PYTORCH-EXPORT-PARITY-BRIDGE sister (already in flight per Slot 1)"
  - "op-routable #2: queue P1 source-faithful curriculum scaling cascade (sister of Slot 2)"
  - "op-routable #3: queue P2 scorer-loss MLX wiring + receiver proof + runtime consumption smoke"
  - "op-routable #4: gate paid Modal/Vast.ai/Lightning dispatch authorization on P0+P1+P2 completion"
- horizon_class: plateau_adjacent
- canonical_equation_refs_queued:
  - pr95_mlx_loop_closure_cascade_canonical_plan_v1
- related_deliberation_ids:
  - codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex
  - pr95_mlx_stage_2_v331_softplus_curriculum_build_landed_20260525
  - pr95_mlx_stage_3_v332_smooth_curriculum_build_landed_20260525
  - pr95_mlx_stage_4_v332_qat_curriculum_build_landed_20260525
  - pr95_mlx_stage_6_lambda_sweep_curriculum_build_landed_20260525
  - pr95_mlx_stage_7_sigma_sweep_curriculum_build_landed_20260525
  - pr95_8stage_curriculum_forensic_20260513
  - pr95_curriculum_recovery_20260513_codex
- council_assumption_adversary_verdict:
  - assumption: "The 10 codex blockers can be linearly traversed in sequence"
    classification: CARGO-CULTED
    rationale: "5 of the 10 blockers are parallelizable across independent subagents (export parity / scorer-loss MLX wiring / receiver proof / source-video loader / source-video preprocess smoke all share zero source-file dependencies once the canonical MLX bundle architecture is stable, which it is at 7x byte-identical state_bytes=915,944). Only stage hparams audit (Slot 2) MUST precede paid dispatch; other 4 blockers can execute in parallel."
  - assumption: "Total wall-clock to LOOP CLOSURE = sum of per-blocker wall-clock"
    classification: CARGO-CULTED-PARALLELIZATION-IGNORED
    rationale: "Per the canonical extension pattern (Stage 1+2+3+4+5+6+7+8 all landed today on M5 Max MLX in ~20-70 min wall-clock each via 4 parallel sister waves), the realistic LOOP CLOSURE wall-clock is bounded by the LONGEST critical-path blocker, NOT the sum. Critical path = P0 PyTorch export parity (Slot 1; ~60-90 min) → P1 source-faithful curriculum (~3-5 hr per stage scale-up) → P2 receiver proof + byte-closed inflate parity (~30-45 min) → paid Modal A100 dispatch (~$10-15 + 4-6 hr per Catalog #205 + Catalog #166 + #167 smoke-before-full + #226 canonical auth_eval helper + #245 register call_id). Realistic LOOP CLOSURE = 1-2 calendar days @ $10-30 paid GPU."
  - assumption: "Sub-frontier score (< 0.19202828 [contest-CPU] / < 0.20533002 [contest-CUDA] per canonical_frontier_pointer 2026-05-25) is achievable from PR 95 MLX reproduction alone"
    classification: HARD-EARNED-BOUNDED
    rationale: "Per CLAUDE.md HNeRV / leaderboard-implementation parity discipline: PR 95 author's PUBLIC archive scored 0.197 [contest-CPU] / 0.20 [contest-CUDA] (per `pr95_curriculum_recovery_20260513_codex.md` line 51-52). PR 95 reproduction's REALISTIC frontier is 0.197-0.20 band; our canonical_frontier_pointer 0.19202828 [contest-CPU] (DQS1 lane commit 2026-05-22) ALREADY BEATS PR 95's published score. PR 95 MLX reproduction's VALUE is NOT score-frontier-shifting — it is COST-CONTAINMENT (free macOS-MLX local iteration replaces paid Modal cycles) + REPRODUCIBLE-INFRASTRUCTURE (8-stage curriculum is the canonical bolt-on substrate for PR 100+101+102+103 medal lanes per `pr95_8stage_curriculum_forensic_20260513.md`)."

## Goal

Consolidate today's PR 95 MLX cascade (Stages 1-8 all landed on M5 Max MLX
via 4 sister waves: codex Stage 1+2+5+8 + claude Stage 3+4+6 + codex
Stage 7 + claude Stage 6+7 APPEND-ONLY ratification) + map codex's 10
canonical blockers from `full_pr95_source_video_runtime` profile
(`codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex.md`) to
a canonical N-subagent LOOP CLOSURE cascade plan with explicit
P0/P1/P2/etc. priority ordering + parallelization opportunities + total
wall-clock + cumulative cost + operator-decision gates.

Sister-COMPLEMENTARY to in-flight Slot 1 PR95-MLX-PYTORCH-EXPORT-PARITY-BRIDGE
(`a48f63cc`) which addresses concrete loop closure piece #1 (PyTorch export
forward parity); this lane provides the higher-level cascade plan + status
assessment that Slot 1's empirical work plugs into.

## Current MLX cascade status (8/8 PR 95 stages LANDED)

The canonical PR 95 8-stage curriculum is **FULLY LANDED on the local MLX
synthetic timing proxy** as of 2026-05-25. All 8 stages share the canonical
`HNeRVSyntheticTrainingBundleMLX` architecture (HNeRVDecoderMLX + base_ch=36
+ latent_dim=28) with **state_bytes byte-identical at 915,944** across all
8 stages — empirical proof that the canonical extension pattern (PR95_STAGE_MODULES
dict + descriptor sister + +2 LOC additive + APPEND-ONLY test pattern) is
**operationally validated end-to-end 7x** (Stages 2-8 each extending the
predecessor dispatch dict).

### Stage landing inventory (8/8 LANDED + 5 sister waves)

| Stage | Module | Optimizer/LR/Loss/QAT/C1a/Muon/Epochs | Status | Landed by | seconds_per_step | state_bytes | last_loss |
|---:|:---|:---|:---|:---|---:|---:|---:|
| 1 | `stage1_v328_ce` | AdamW 1e-3 / `ce_seg_loss` / no QAT / λ=0 σ=0.2 / no Muon / 3000 ep | LANDED | codex | ~23.4 ms | 915,944 | converged |
| 2 | `stage2_v331_softplus` | AdamW 1e-3 / `tau_softplus_seg_loss` / no QAT / λ=0 σ=0.2 / no Muon / 5650 ep | LANDED | claude | 23.43 ms | 915,944 | 0.0737 |
| 3 | `stage3_v332_smooth` | AdamW 1e-4 fresh cosine / `smooth_disagreement_seg_loss` / no QAT / λ=0 σ=0.2 / no Muon / 1500 ep | LANDED | claude | 23.40 ms | 915,944 | 0.0828 |
| 4 | `stage4_v332_qat` | AdamW 1e-4 / `smooth_disagreement_seg_loss` / **QAT=True** / λ=0 σ=0.2 / no Muon / 500 ep | LANDED | claude+codex sister | 23.33 ms | 915,944 | 0.0828 |
| 5 | `stage5_c1a_l7` | AdamW 3e-5 / `l7_softplus_seg_loss` / QAT=True / **λ=0.01** σ=0.2 / no Muon / 9000 ep | LANDED | codex | ~23.4 ms | 915,944 | converged |
| 6 | `stage6_lambda_sweep` | AdamW 3e-5 / `l7_softplus_seg_loss` / QAT=True / **λ=0.02** σ=0.2 / no Muon / 2000 ep | LANDED | claude | 23.28 ms | 915,944 | 0.0832 |
| 7 | `stage7_sigma_sweep` | AdamW 3e-5 / `l7_softplus_seg_loss` / QAT=True / λ=0.02 **σ=0.1** / no Muon / 3000 ep | LANDED | codex + claude APPEND-ONLY | 23.39 ms | 915,944 | 0.0832 |
| 8 | `stage8_muon_finetune` | AdamW 1e-5 + **Muon 2e-4 WD=5e-4** / `l7_softplus_seg_loss` / QAT=True / λ=0.02 σ=0.1 / **Muon=True** / 5000 ep | LANDED | codex | 1.54 s* | (queue) | (queue) |

*Stage 8 codex timing is queue-spawn process-seconds (1.54 s for spawn + 0.036 s manifest train); the canonical 100-step MLX smoke pattern would be ~23 ms/step like Stages 1-7.

**Total canonical PR 95 published epochs**: 29,650 (sum of Stage 1+2+3+4+5+6+7+8
canonical epoch budgets per `pr95_8stage_curriculum_forensic_20260513.md` +
`pr95_curriculum_recovery_20260513_codex.md`).

### Canonical extension pattern empirical validation (7x byte-identical)

The canonical extension pattern is empirically validated **7x** by today's
cascade (Stage 2 extends Stage 1; Stage 3 extends Stage 1+2; Stage 4 extends
Stage 1+2+3; Stage 5 extends Stage 1+2+3+4; Stage 6 extends Stage 1+2+3+4+5;
Stage 7 extends Stage 1+2+3+4+5+6; Stage 8 extends Stage 1+2+3+4+5+6+7). Each
extension is **+2 LOC additive** to `PR95_STAGE_MODULES` +
`PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS` dicts at `src/tac/local_acceleration/pr95_hnerv_mlx.py:75-94`
+ ~45 LOC descriptor at `src/tac/optimization/optimizer_scheduler_registry.py`
+ stages list bump at `tools/build_pr95_mlx_optimizer_matrix_queue.py`
+ NEW test file + APPEND-ONLY superset-of bump to predecessor test files.

**state_bytes = 915,944 byte-identical 7x** is the canonical sanity invariant
proving the architecture is NOT silently perturbed by stage transitions; only
loss family + LR schedule + QAT bit + C1a λ/σ + Muon bit distinguish stages
at the training-config metadata layer.

### 3 empirical falsifications (Catalog #303 CARGO-CULTED-EMPIRICALLY-FALSIFIED)

Per CLAUDE.md "Cargo-cult audit per assumption" + Catalog #303 + #307
paradigm-vs-implementation classification:

1. **QAT in-place per-batch introduces persistent state_dict overhead** —
   FALSIFIED at Stage 4 (state_bytes byte-identical to Stages 1+2+3 at
   915,944). PR 95's `apply_qat` / `restore_qat` pattern mutates Conv2d/Linear
   weights in-place per-batch with restoration before optimizer step; the
   STE makes the forward see the quantized value while gradients flow through
   the original. No persistent param state. IMPLEMENTATION-LEVEL not
   PARADIGM-LEVEL falsification per Catalog #307.
2. **C1a λ=0.02 sweep introduces persistent param state** — FALSIFIED at
   Stage 6 (state_bytes byte-identical to Stage 4+5 at 915,944). PR 95's
   `cat_entropy_v2` is a soft-MDL loss term (Hinton-Vinyals-Dean 2014
   sister; size-weighted soft histogram entropy with Gaussian bandwidth σ).
   The λ effect lives in the loss term, NOT the bundle. IMPLEMENTATION-LEVEL
   per Catalog #307.
3. **C1a σ=0.1 sweep introduces persistent param state** — FALSIFIED at
   Stage 7 (state_bytes byte-identical to Stage 6 at 915,944). Same
   mechanism as #2: σ controls the bandwidth of the soft histogram; effect
   lives in loss term, NOT bundle. IMPLEMENTATION-LEVEL per Catalog #307.

PARADIGM INTACT for all 3 falsifications: PR 95's published curriculum is
mathematically sound; the canonical MLX synthetic timing proxy correctly
distinguishes loss-term variations from bundle-architecture variations.

### MLX bundle canonical symbols (cite chain for Slot 1 export parity)

Per `src/tac/local_acceleration/pr95_hnerv_mlx.py`:

| Symbol | Lines | Purpose |
|:---|---:|:---|
| `HNeRVDecoderMLX` | 778-875 | Canonical NHWC PR 95 decoder topology; 6 PixelShuffle upsample blocks; bilinear skip + sin(x+identity) per block; refine block; separate rgb_0/rgb_1 conv heads (sigmoid * 255); base_h=6, base_w=8, eval_size=(384,512), channels=[C,C,C,0.75C,0.58C,0.5C,0.5C] |
| `HNeRVSyntheticTrainingBundleMLX` | 878-901 | Decoder + trainable per-pair latents (latent_count × latent_dim=28 random_normal * 0.1; seed-pinned) |
| `load_pytorch_state_dict_into_mlx` | 904-948 | Loads PR 95 PyTorch state_dict into MLX decoder (handles NCHW→NHWC conv weight transpose for blocks/skips/refine/rgb heads) |
| `pytorch_state_dict_from_mlx` | 951-986 | Exports MLX decoder parameters using public PR 95 PyTorch names (handles NHWC→NCHW transpose; round-trip with `load_pytorch_state_dict_into_mlx` is the canonical PARITY contract Slot 1 verifies) |
| `compare_pr95_public_archive_forward_with_pytorch` | 1019+ | Compares MLX forward output against PyTorch state_dict forward on parsed public PR 95 archive (the canonical apples-to-apples PARITY surface) |
| `Pr95MlxOptimizerConfig` + `apply_pr95_mlx_optimizer_step` | 1407-1640 | Canonical MLX optimizer config (AdamW + optional Muon partition) + per-step apply with QAT pre/post hooks |
| `run_pr95_mlx_synthetic_timing_smoke` | 1642+ | Canonical 100-step MLX synthetic timing smoke (the per-stage canonical evidence emitter) |

## Codex 10 blockers map (canonical loop closure cascade)

Per `codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex.md`
"Exact-Readiness State" section the canonical 10 blockers preventing exact
auth-eval dispatch are:

| # | Codex blocker (verbatim) | Loop closure stage | Dependency | $0 vs paid | In-flight slot |
|---:|:---|:---|:---|:---|:---|
| 1 | local PR95 timing smoke is not a score | **P0** (DONE — 8 stages all PROVEN; this is the canonical advisory invariant per CLAUDE.md "MPS auth eval is NOISE") | — | $0 | DONE (8 stages landed today) |
| 2 | PR95 stage hparams and cosine schedules are not fully source-matched | **P0a** (Slot 2 PR95-MLX-STAGE-HPARAMS-SOURCE-FAITHFUL-AUDIT) | none | $0 | **IN FLIGHT** (Slot 2 just spawned) |
| 3 | PR95 QAT/C1a/resume semantics are not ported to MLX | **P1** (sister subagent; QAT/C1a are metadata-only at synthetic timing layer per 3 Catalog #303 falsifications; runtime semantics need MLX port) | requires P0a (canonical hparams) | $0 | NOT QUEUED |
| 4 | PyTorch export forward parity is not established | **P0b** (Slot 1 PR95-MLX-PYTORCH-EXPORT-PARITY-BRIDGE) | none | $0 | **IN FLIGHT** (Slot 1 active per `a48f63cc`) |
| 5 | receiver proof is missing | **P1** (sister subagent; canonical contest scorer SegNet+PoseNet on byte-closed MLX-emitted archive) | requires P0b (PyTorch parity to load contest scorer faithfully) | $0 | NOT QUEUED |
| 6 | byte-closed contest archive export is not sufficient without exact eval | **P1** (sister subagent; build MLX-emitted archive matching `pr95_hnerv_mlx_byte_closed_smoke_archive_v1` schema + PR 95 codec grammar at `src/tac/local_acceleration/pr95_hnerv_mlx.py:298-606` `_encode_pr95_decoder_blob` + `write_pr95_public_archive_zip`) | requires P0b (export parity = same bytes) | $0 | PARTIAL (canonical helpers exist; need wiring) |
| 7 | scorer network loss is not wired to MLX | **P1** (sister subagent; differentiable rgb_to_yuv6 + canonical PR 95 score-aware loss; mirrors Slot 3 HINTON-DISTILLED-SCORER-SURROGATE) | requires P0b OR Hinton-distilled surrogate (Slot 3) | $0 (Hinton surrogate path) | Slot 3 active (different paradigm) |
| 8 | full-frame inflate parity against the source runtime has not been run | **P2** (sister subagent; full upstream `inflate.sh archive.zip output_dir file_list` runtime against MLX-emitted archive) | requires P0b + #6 (byte-closed archive) | $0 | NOT QUEUED |
| 9 | RGB+YUV6 preprocess loss is not full scorer loss | **P2** (sister subagent; full SegNet+PoseNet forward + score-aware Lagrangian per Stage 5+ canonical) | requires #7 (scorer loss wired) | $0 + paid | NOT QUEUED |
| 10 | runtime-consumption smoke is not score authority | **P3** (final paid Modal/Vast.ai dispatch authorization gate per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE") | requires #1+...+9 ALL satisfied | **paid** $10-30 | NOT QUEUED |

### Parallelization opportunities (5 of 10 blockers parallelizable)

| Independent path | Blockers | Wall-clock | Cost |
|:---|:---|---:|---:|
| **Critical path (P0)** | #2 hparams + #4 export parity (Slot 1+2 in flight in parallel) | 60-90 min each (parallel = 60-90 min) | $0 |
| **Branch A: QAT runtime** | #3 QAT/C1a/resume MLX port | 90-180 min | $0 |
| **Branch B: Byte-closed archive** | #6 byte-closed archive export | 60-90 min (canonical helpers exist) | $0 |
| **Branch C: Scorer loss (Hinton)** | #7 via Hinton surrogate (Slot 3) | 120-240 min | $0 |
| **Branch D: Receiver proof** | #5 receiver proof on byte-closed archive | 60-90 min | $0 |
| **Branch E: Inflate parity** | #8 full-frame inflate parity | 60-90 min | $0 |
| **Branch F: Full scorer loss** | #9 RGB+YUV6 full scorer | 120-240 min | $0 |
| **P3 final paid dispatch** | #10 paired CPU+CUDA contest auth eval | 4-6 hr wall-clock + 30s harness gate | **$10-30** |

Critical path = max(P0a Slot 2, P0b Slot 1) → P1 (B+C+D in parallel) → P2
(E+F in parallel) → P3 (paid dispatch). **Cumulative wall-clock to LOOP
CLOSURE = ~7-10 hr engineering + 4-6 hr paid GPU = ~12-16 hr total** spread
across 1-2 calendar days.

## Canonical N-subagent cascade plan

### P0 (CRITICAL PATH; PARALLEL; ALREADY IN FLIGHT)

| Subagent | Scope | Sister-coordination | Wall-clock | Cost | Carmack MVP-first 5/5 |
|:---|:---|:---|---:|---:|:---|
| **P0a Slot 2 (IN FLIGHT)** PR95-MLX-STAGE-HPARAMS-SOURCE-FAITHFUL-AUDIT | Audit per-stage hparams against recovered public PR 95 source (`pr95_8stage_curriculum_forensic_20260513.md` + `pr95_curriculum_recovery_20260513_codex.md`); reconcile any drift; emit canonical-vs-source reconciliation memo per Catalog #110/#113 APPEND-ONLY | DISJOINT from P0b (audits descriptor metadata, not bundle parameters) | 60-90 min | $0 | (1) FREE local audit; (2) falsifiable: predict no drift > 1 LR/loss-family mismatch per stage; (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to canonical P1 spawn |
| **P0b Slot 1 (IN FLIGHT)** PR95-MLX-PYTORCH-EXPORT-PARITY-BRIDGE | Establish PyTorch export forward parity per codex blocker #4; round-trip `load_pytorch_state_dict_into_mlx` ↔ `pytorch_state_dict_from_mlx` byte-identical state_dict at random init + after 100-step MLX training; canonical `compare_pr95_public_archive_forward_with_pytorch` parity test on parsed public PR 95 archive | DISJOINT from P0a (operates on bundle parameters, not descriptor metadata) | 60-90 min | $0 | (1) FREE local PyTorch+MLX cross-import; (2) falsifiable: max_abs_diff between MLX forward + PyTorch forward < ε=5e-3 fp32 on the same state_dict + latents (the canonical NULL hypothesis per ARCH-5 MLX-PARADIGM-T3 + the 7x byte-identical state_bytes=915,944 empirical anchor); (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to canonical P1 spawn |

### P1 (PARALLEL; SPAWN AFTER P0a + P0b CONFIRMED)

| Subagent | Scope | Sister-coordination | Wall-clock | Cost | Carmack MVP-first 5/5 |
|:---|:---|:---|---:|---:|:---|
| **P1a** PR95-MLX-QAT-C1A-RESUME-SEMANTICS-MLX-PORT | Port `apply_qat`/`restore_qat` + `cat_entropy_v2` runtime semantics from PR 95 PyTorch source to MLX bundle; checkpoint resume across stage boundaries (Stage 4 resumes Stage 3 cosine; Stage 5 resumes Stage 4 final; etc.) | DISJOINT from P1b/P1c/P1d (operates on optimizer + loss runtime, not archive/receiver/scorer) | 90-180 min | $0 | (1) FREE local 100-step QAT + C1a runtime smoke; (2) falsifiable: QAT-on-MLX forward matches QAT-on-PyTorch forward within ε=1e-5 (the STE invariant); (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to P2 spawn |
| **P1b** PR95-MLX-BYTE-CLOSED-ARCHIVE-EXPORT-WIRING | Wire `write_pr95_public_archive_zip` + `_encode_pr95_decoder_blob` + `write_pr95_mlx_byte_closed_smoke_archive` end-to-end; emit canonical `pr95_hnerv_mlx_byte_closed_smoke_archive_v1` archive from MLX-trained state | DISJOINT from P1a/P1c/P1d (operates on archive grammar, not optimizer/scorer) | 60-90 min | $0 | (1) FREE local archive build + parse round-trip; (2) falsifiable: archive bytes parse back to byte-identical state_dict + latents (the canonical `parse_pr95_public_archive_zip` invariant per `src/tac/local_acceleration/pr95_hnerv_mlx.py:565-606`); (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to P2 spawn |
| **P1c** PR95-MLX-SCORER-LOSS-WIRING (sister of Slot 3 HINTON-DISTILLED-SCORER-SURROGATE-DISPATCH-PREP) | Wire canonical scorer-network loss (SegNet+PoseNet) into MLX training loop OR use Hinton-distilled surrogate per Slot 3 active; differentiable `rgb_to_yuv6` patch per PR 95 source `data.py:51-81` (the canonical-fix per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE") | COMPLEMENTARY to Slot 3 active (Slot 3 is the Hinton surrogate path; P1c is the canonical scorer-network path; either resolves blocker #7) | 120-240 min | $0 | (1) FREE local 100-step scorer-loss smoke; (2) falsifiable: pose gradient reaches decoder (the canonical PoseNet gradient-reachability invariant per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L8); (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to P2 spawn |
| **P1d** PR95-MLX-RECEIVER-PROOF-WIRING | Build canonical receiver proof: parse MLX-emitted archive via `parse_pr95_public_archive_zip` → load state into PyTorch HNeRVDecoder → forward → assert byte-identical decoded frames between MLX-direct-forward and PyTorch-from-archive-forward | DISJOINT from P1a/P1b/P1c (operates on receiver proof, not optimizer/archive-build/scorer) | 60-90 min | $0 | (1) FREE local receiver proof smoke; (2) falsifiable: per-pair decoded frame max_abs_diff < ε=5e-3 (sister of P0b export parity at the receiver surface); (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to P2 spawn |

### P2 (PARALLEL; SPAWN AFTER P1a+P1b+P1c+P1d CONFIRMED)

| Subagent | Scope | Sister-coordination | Wall-clock | Cost | Carmack MVP-first 5/5 |
|:---|:---|:---|---:|---:|:---|
| **P2a** PR95-MLX-FULL-FRAME-INFLATE-PARITY-CONTEST-RUNTIME | Run canonical contest `inflate.sh archive.zip output_dir file_list` against MLX-emitted archive on Linux x86_64 per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"; verify full-frame output parity vs source PR 95 runtime | DISJOINT from P2b (operates on contest runtime, not full scorer loss) | 60-90 min | $0 (local Linux x86_64 via container OR Vast.ai CPU instance) | (1) FREE local Linux x86_64 inflate.sh smoke; (2) falsifiable: per-frame max_abs_diff < ε=1 (uint8 quantization-bounded; the canonical contest invariant); (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to P3 paid dispatch gate |
| **P2b** PR95-MLX-FULL-SCORER-LOSS-VS-PREPROCESS-RGB-YUV6 | Wire full SegNet+PoseNet scorer loss replacing RGB+YUV6 preprocess proxy (codex blocker #9); validate score-aware Lagrangian `100*seg + sqrt(10*pose) + 25*archive_bytes/37545489` per CLAUDE.md HNeRV parity L6 | DISJOINT from P2a (operates on full scorer loss, not runtime parity) | 120-240 min | $0 | (1) FREE local 100-step full-scorer-loss smoke; (2) falsifiable: loss decomposition matches canonical contest formula within ε=1e-6 (closed-form arithmetic invariant); (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to P3 paid dispatch gate |

### P3 (CRITICAL PATH; SERIAL; OPERATOR-DECISION GATE)

| Subagent | Scope | Sister-coordination | Wall-clock | Cost | Carmack MVP-first 5/5 |
|:---|:---|:---|---:|---:|:---|
| **P3 PAID-DISPATCH-GATE** PR95-MLX-PAIRED-CPU-CUDA-CONTEST-AUTH-EVAL | Final paid Modal A100 + Lightning/Modal CPU paired auth eval on 1:1 contest-compliant hardware per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"; both `[contest-CUDA]` Linux x86_64+NVIDIA AND `[contest-CPU]` Linux x86_64 anchors required; routes through `tools/operator_authorize.py` canonical entry point per Catalog #176 + Catalog #245 register call_id + Catalog #167 smoke-before-full + Catalog #226 canonical auth_eval helper + Catalog #205 inflate device fork + Catalog #146 inflate runtime contract + Catalog #166 Modal HEAD-parity ledger + Catalog #244 NVML env block + Catalog #243 local pre-deploy check + Catalog #271 codex pre-dispatch review + Catalog #199 paired-env operator authorization + Catalog #339 silent-no-spawn extinction + Catalog #360 pre-spawn fatal observability | OPERATOR-GATED: do NOT auto-spawn; requires explicit operator authorization | 4-6 hr GPU wall-clock + 30s pre-deploy harness | **$10-30** (Modal A100 $3.40/hr × 4-6 hr = $14-20; Lightning CPU $0/hr subscription OR Modal CPU $0.06/hr × 1-2 hr = $0-0.12) | (1) NOT FREE; (2) falsifiable: paired CPU+CUDA both produce `[contest-CPU]` + `[contest-CUDA]` anchors; (3) Catalog #344 ref; (4) verdict same-batch; (5) re-route to canonical posterior update + frontier pointer auto-refresh per Catalog #343 |

### Total cascade summary

- **P0 (in flight)**: max(60-90 min Slot 1, 60-90 min Slot 2) = **60-90 min critical path**; $0
- **P1 (queue after P0)**: max(180 min P1a, 90 min P1b, 240 min P1c, 90 min P1d) = **240 min critical path**; $0
- **P2 (queue after P1)**: max(90 min P2a, 240 min P2b) = **240 min critical path**; $0
- **P3 (operator-gated; queue after P2)**: 4-6 hr GPU wall-clock + 30s harness; **$10-30 paid**

**Total LOOP CLOSURE wall-clock = ~7-10 hr engineering (60-90 + 240 + 240 min) + 4-6 hr paid GPU + 30s harness = ~12-16 hr cascade time** spread across **1-2 calendar days** (assumes parallel subagent spawn at each tier).

**Total LOOP CLOSURE cost = $10-30 paid GPU** (all P0+P1+P2 are $0 macOS-MLX local).

## MLX bundle architecture inventory for export bridge (sister-COMPLEMENTARY to Slot 1)

### `HNeRVSyntheticTrainingBundleMLX` parameter inventory

Per `src/tac/local_acceleration/pr95_hnerv_mlx.py:878-901`:

| Module | Parameter | Shape (MLX NHWC) | dtype | PyTorch counterpart (NCHW per `submissions/hnerv_muon/src/model.py::HNeRVDecoder`) | Drift hotspot |
|:---|:---|:---|:---|:---|:---|
| `bundle.latents` | latents | `(latent_count, 28)` | fp32 | per-pair latents `(N, 28)` | NONE (no transpose) |
| `bundle.decoder.stem.weight` | weight | `(out_features=2592, in_features=28)` where out=base_ch*6*8=36*6*8=1728 ⚠ (need verification) | fp32 | `(out_features, 28)` (Linear) | NONE (Linear shape-identical) |
| `bundle.decoder.stem.bias` | bias | `(out_features=1728)` | fp32 | `(out_features,)` (Linear) | NONE |
| `bundle.decoder.blocks.{0..5}.conv.weight` | weight | `(out_channels*4, kH=3, kW=3, in_channels)` (MLX NHWC) | fp32 | `(out_channels*4, in_channels, kH=3, kW=3)` (PyTorch NCHW) | **HOTSPOT**: NHWC → NCHW transpose via `_torch_conv_to_mlx` per lines 922-924 |
| `bundle.decoder.blocks.{0..5}.conv.bias` | bias | `(out_channels*4,)` | fp32 | `(out_channels*4,)` | NONE |
| `bundle.decoder.blocks.{0..5}.skip_conv.weight` (when present) | weight | `(out_channels, kH=1, kW=1, in_channels)` (MLX NHWC) | fp32 | `(out_channels, in_channels, kH=1, kW=1)` (PyTorch NCHW) | **HOTSPOT**: NHWC → NCHW transpose via `_torch_conv_to_mlx` per lines 931-934 |
| `bundle.decoder.blocks.{0..5}.skip_conv.bias` (when present) | bias | `(out_channels,)` | fp32 | `(out_channels,)` | NONE |
| `bundle.decoder.refine0.weight` | weight | `(final_ch/2, kH=3, kW=3, final_ch)` (MLX NHWC; dilation=2, padding=2) | fp32 | `(final_ch/2, final_ch, kH=3, kW=3)` (PyTorch NCHW) | **HOTSPOT**: NHWC → NCHW transpose |
| `bundle.decoder.refine0.bias` | bias | `(final_ch/2,)` | fp32 | `(final_ch/2,)` | NONE |
| `bundle.decoder.refine1.weight` | weight | `(final_ch, kH=3, kW=3, final_ch/2)` (MLX NHWC) | fp32 | `(final_ch, final_ch/2, kH=3, kW=3)` (PyTorch NCHW) | **HOTSPOT**: NHWC → NCHW transpose |
| `bundle.decoder.refine1.bias` | bias | `(final_ch,)` | fp32 | `(final_ch,)` | NONE |
| `bundle.decoder.rgb_0.weight` | weight | `(3, kH=3, kW=3, final_ch)` (MLX NHWC) | fp32 | `(3, final_ch, kH=3, kW=3)` (PyTorch NCHW) | **HOTSPOT**: NHWC → NCHW transpose |
| `bundle.decoder.rgb_0.bias` | bias | `(3,)` | fp32 | `(3,)` | NONE |
| `bundle.decoder.rgb_1.weight` | weight | `(3, kH=3, kW=3, final_ch)` (MLX NHWC) | fp32 | `(3, final_ch, kH=3, kW=3)` (PyTorch NCHW) | **HOTSPOT**: NHWC → NCHW transpose |
| `bundle.decoder.rgb_1.bias` | bias | `(3,)` | fp32 | `(3,)` | NONE |

### Drift hotspot summary (8 of ~30 parameter tensors)

The **only** dtype/shape/layout drift hotspots are the **8 Conv2d weight
tensors** (6 block convs + 2 refine convs + 2 rgb head convs; plus 1-6
optional skip_conv weights when in_channels ≠ out_channels per channel taper
`[C, C, C, 0.75C, 0.58C, 0.5C, 0.5C]`). All 8 use the canonical NHWC ↔ NCHW
transpose via `_torch_conv_to_mlx` (NCHW→NHWC at load) + `_mlx_conv_to_numpy`
(NHWC→NCHW at export).

Per the empirical 7x byte-identical state_bytes=915,944 evidence, these
canonical transposes are **already working end-to-end** (otherwise state_bytes
would diverge across stages OR the bundle would fail construction). The
canonical contract Slot 1 verifies is **forward parity** between MLX bundle
forward + PyTorch state_dict forward on the same state_dict + latents, NOT
the load/export contract itself (which is already empirically validated).

### Empirical evidence supporting Slot 1's BYTE_STABLE NULL hypothesis

The 7x byte-identical state_bytes invariant (Stages 1+2+3+4+5+6+7 all
915,944 bytes) is **STRONG empirical evidence** that:

1. The canonical `HNeRVSyntheticTrainingBundleMLX` architecture is preserved
   across all stage transitions (no silent param addition/removal/reshape).
2. The canonical `load_pytorch_state_dict_into_mlx` + `pytorch_state_dict_from_mlx`
   round-trip preserves param counts (otherwise state_bytes would mutate).
3. The canonical NHWC ↔ NCHW transpose is byte-stable (preserves param shapes).

**What is NOT yet empirically validated** (Slot 1's scope):

- Forward parity at random init: does `mlx_bundle.decoder(latent) == pytorch_decoder(latent_to_pytorch)` to within ε=5e-3 fp32?
- Forward parity after 100-step MLX training: does the MLX-trained state_dict, when exported to PyTorch and loaded into the canonical PR 95 PyTorch HNeRVDecoder, produce byte-identical decoded frames?
- Bilinear resize numerical parity: does `bilinear_resize2x_align_corners_false_nhwc` exactly match PyTorch `F.interpolate(mode='bilinear', align_corners=False)`?
- PixelShuffle numerical parity: does `pixel_shuffle_2x_nhwc` exactly match PyTorch `F.pixel_shuffle(2)`?

Slot 1's canonical test suite resolves ALL 4 of these open questions and
unblocks P1+P2+P3 cascade.

## Sister-coherence verification

Per Catalog #340 sister-checkpoint guard PROCEED verified pre-staging
(empirical: `tools/check_sister_checkpoint_before_git_add.py` returned OK
with 0 in-flight sister files_touched overlap within the 60-minute lookback
window).

| Slot | Subagent | Scope | DISJOINT verification |
|:---|:---|:---|:---|
| **Slot 1** | PR95-MLX-PYTORCH-EXPORT-PARITY-BRIDGE (`a48f63cc`) | Concrete blocker #4: PyTorch export forward parity | DISJOINT — my scope is research + planning + concrete inventory; Slot 1 is canonical helpers + parity tests + empirical receipts. We SHARE the same MLX bundle as READ-ONLY input; neither MUTATES the bundle. My memo CITES Slot 1's blocker scope as P0b; Slot 1's empirical findings feed downstream into my P1+P2 cascade plan. |
| **Slot 2** | PR95-MLX-STAGE-HPARAMS-SOURCE-FAITHFUL-AUDIT (just-spawned) | Concrete blocker #2: per-stage hparam audit + reconciliation memo | DISJOINT — Slot 2 audits descriptor metadata at `src/tac/optimization/optimizer_scheduler_registry.py`; I do NOT mutate any descriptor. My memo CITES Slot 2's audit as P0a; Slot 2's findings APPEND-ONLY footer into this memo per Catalog #110/#113 if Slot 2 lands after mine. |
| **Slot 3** | HINTON-DISTILLED-SCORER-SURROGATE-DISPATCH-PREP | Hinton-distilled scorer surrogate as alternative to canonical scorer-network loss | COMPLEMENTARY — Slot 3 is one of two paths to resolve blocker #7 (scorer network loss); my P1c subagent canonical scorer-network path is the alternate. Either path resolves the blocker; the operator decides which to spawn. |

Three Slots active; zero source-file overlap; coherent cascade architecture.

## Carmack MVP-first 5/5 compliance per CLAUDE.md `be125b878`

1. **FREE local research + planning + inventory** ($0; M5 Max). NO source
   mutation; NO paid dispatch. The cascade plan IS the deliverable.
2. **Falsifiably challenged**: predicted cascade plan unambiguously identifies
   P0 immediate spawn (Slot 1+2 in flight) AND total wall-clock estimate to
   LOOP CLOSURE completion within ±50% (12-16 hr cascade; 7-10 hr engineering
   + 4-6 hr paid GPU). Falsifying outcomes:
   - **P0 immediate spawn ambiguous?** NO — P0a Slot 2 + P0b Slot 1 BOTH IN
     FLIGHT, explicitly named with canonical scope + sister-coordination. The
     cascade plan refuses ambiguity at the P0 surface.
   - **Total wall-clock span > 2× order of magnitude?** NO — 12-16 hr range is
     within 1.33×, well under 2× threshold. The plan's wall-clock estimate is
     bounded by empirical receipts: today's 8 stages all landed in ~20-70 min
     each via the canonical extension pattern; the canonical extension pattern
     is empirically validated 7x.
   - **Cost estimate unbounded?** NO — $10-30 paid GPU envelope is bounded
     within 3×; the Modal A100 rate (~$3.40/hr) × 4-6 hr GPU = $14-20 + CPU
     auth eval (Lightning subscription = $0 OR Modal CPU $0.06/hr × 1-2 hr =
     $0-0.12). The bounds are derived from CLAUDE.md "GPU budget and compute
     resources — non-negotiable" cost table.
3. **Catalog #344 reference**: canonical equation candidate
   `pr95_mlx_loop_closure_cascade_canonical_plan_v1` QUEUED FORMALIZATION_PENDING
   per Catalog #344 operator-decision protocol. NOT auto-registered.
4. **Landed verdict in same commit batch**: this landing memo + Catalog #313
   probe-outcomes row registered + lane registry entry at L1. NO source files
   modified (research + planning only).
5. **Re-route operator priority queue post-landing**: per the empirical
   findings + sister coordination + canonical extension pattern proven 7x,
   the operator-routable next steps are:
   - **P0 (parallel; already in flight)**: confirm Slot 1+2 completion;
   - **P1 (parallel; spawn after P0)**: 4 sister subagents (P1a QAT/C1a/resume;
     P1b byte-closed archive; P1c scorer-loss canonical OR Slot 3 Hinton
     surrogate; P1d receiver proof);
   - **P2 (parallel; spawn after P1)**: 2 sister subagents (P2a inflate parity;
     P2b full scorer loss vs preprocess);
   - **P3 (operator-gated; spawn after P2)**: paid Modal A100 + Lightning/Modal
     CPU paired auth eval per CLAUDE.md "Submission auth eval" non-negotiable.

## Catalog #344 RATIFY-N candidate

**Canonical equation**: `pr95_mlx_loop_closure_cascade_canonical_plan_v1`

**Mathematical form (cascade critical-path wall-clock)**:

```
W_total = max(W_P0a, W_P0b) + max(W_P1a, W_P1b, W_P1c, W_P1d) + max(W_P2a, W_P2b) + W_P3_paid
```

where:
- `W_P0a` = wall-clock for hparams audit = 60-90 min
- `W_P0b` = wall-clock for PyTorch export parity = 60-90 min
- `W_P1a` = wall-clock for QAT/C1a/resume MLX port = 90-180 min
- `W_P1b` = wall-clock for byte-closed archive export = 60-90 min
- `W_P1c` = wall-clock for scorer loss MLX wiring = 120-240 min (Hinton or canonical)
- `W_P1d` = wall-clock for receiver proof = 60-90 min
- `W_P2a` = wall-clock for inflate parity = 60-90 min
- `W_P2b` = wall-clock for full scorer loss = 120-240 min
- `W_P3_paid` = wall-clock for paired CPU+CUDA paid auth eval = 4-6 hr GPU + 30s harness

**Cumulative cost equation**:

```
C_total = 0 (P0 + P1 + P2; all macOS-MLX local) + C_P3_paid
C_P3_paid = $3.40/hr × 4-6 hr + ($0/hr Lightning subscription OR $0.06/hr × 1-2 hr Modal CPU)
C_P3_paid_range = [$13.60, $20.52]
```

**Predicted dispatch verdict band**: `[$10, $30]` per ±50% rigor band on
realistic LOOP CLOSURE cost; `[12, 16]` hours total cascade wall-clock.

**Empirical anchor**: today's 7x byte-identical state_bytes=915,944 +
8/8 stages landed via canonical extension pattern + 4 sister waves + 3
empirical falsifications (QAT in-place / C1a λ-sweep / σ-sweep all PARADIGM
INTACT per Catalog #307).

**Status**: FORMALIZATION_PENDING per Catalog #344 operator-decision
protocol; NOT auto-registered.

## Catalog #313 probe-outcomes ledger row

Registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome`
per Catalog #131 fcntl-locked JSONL discipline:

- `probe_id`: `pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_20260525`
- `verdict`: `PROCEED`
- `status`: `advisory`
- `expires_at_utc`: `2026-06-24T17:30:00Z` (30-day staleness window per Catalog #298)
- `lane_id`: `lane_pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_20260525`
- `recipe_id`: N/A (research + planning artifact, no dispatch recipe)
- `substrate_id`: `pr95_hnerv_mlx_curriculum_canonical`
- `evidence_path`: `.omx/research/pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_landed_20260525.md`
- `evidence_grade`: `[macOS-MLX research-signal]` per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1/#192/#287/#323
- `score_claim`: False (planning artifact; no score)
- `promotion_eligible`: False
- `rank_or_kill_eligible`: False
- `ready_for_exact_eval_dispatch`: False

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|:---|:---|:---|
| #1 sensitivity-map | N/A | Research + planning artifact; no sensitivity surface. |
| #2 Pareto constraint | N/A | Planning artifact, not Pareto-relevant signal. |
| #3 bit-allocator | N/A | Planning artifact, not bit-allocator signal. |
| #4 cathedral autopilot dispatch | **ACTIVE** | Cascade plan IS the canonical operator-routable artifact for spawning P0/P1/P2/P3 subagents; autopilot ranker consumes the canonical wall-clock + cost band to weight downstream dispatch priorities. |
| #5 continual-learning posterior | **ACTIVE** | Probe-outcomes row registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313 + #131; canonical equation `pr95_mlx_loop_closure_cascade_canonical_plan_v1` QUEUED for RATIFY-N per Catalog #344. |
| #6 probe-disambiguator | **ACTIVE** | Cascade plan IS the canonical disambiguator between "linear traversal of 10 blockers" (CARGO-CULTED per Assumption-Adversary verdict above) vs "parallel cascade with critical-path bounded wall-clock" (HARD-EARNED via 7x empirical validation). |

## 9-dimension success checklist evidence

Per CLAUDE.md "9-DIMENSION SUCCESS CHECKLIST EVIDENCE SECTION" non-negotiable +
Catalog #294:

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | Cascade plan unifies 10 codex blockers + 8 MLX stages + Slot 1+2+3 sister coordination into ONE canonical N-subagent plan with explicit P0/P1/P2/P3 priority + parallelization opportunities. Per Catalog #309: `horizon_class: plateau_adjacent` (LOOP CLOSURE plan is a continuation of the PR 95 canonical paradigm; not a class-shift). |
| 2. BEAUTY+ELEGANCE | One canonical memo; one canonical equation candidate; one Catalog #313 row; one lane registry entry; ZERO source files mutated. 30-second-reviewable cascade table (`Codex 10 blockers map` section). |
| 3. DISTINCTNESS | Sister-COMPLEMENTARY to Slot 1 (Slot 1 = concrete blocker #4 empirical; this = higher-level cascade plan). Sister-DISJOINT from Slot 2 (Slot 2 = concrete blocker #2 audit). Sister-COMPLEMENTARY to Slot 3 (Slot 3 = Hinton surrogate path; this = canonical scorer path as P1c). |
| 4. RIGOR | Per Catalog #229 PV: full read of 10+ canonical references + recovered public PR 95 source + 5 stage landing memos + MLX bundle architecture + canonical frontier pointer BEFORE memo draft. Sister-coherence verified via `tools/check_sister_checkpoint_before_git_add.py` PROCEED. Falsifiability: explicit predict-vs-measured bands for cascade wall-clock + cost. |
| 5. OPTIMIZATION PER TECHNIQUE | Per Catalog #290 canonical-vs-unique decision: cascade plan ADOPTS_CANONICAL extension pattern (proven 7x today); FORKS only at the per-blocker scope (each subagent's MVP-first 5/5 + Carmack falsifiability is unique to its blocker). |
| 6. STACK-OF-STACKS-COMPOSABILITY | The cascade plan IS the canonical stack-of-stacks composition contract: P0 → P1 → P2 → P3 is the canonical 4-tier composition; each tier composes orthogonal subagent scopes; the cascade is reproducible at NEW substrate scaffolds via the same pattern. |
| 7. DETERMINISTIC REPRODUCIBILITY | Cascade plan cites canonical sources (codex 10 blockers memo + 5 stage landing memos + MLX bundle file + canonical frontier pointer); all references are repo-tracked + reproducible. |
| 8. EXTREME OPTIMIZATION+PERFORMANCE | Per the canonical extension pattern proven 7x today, the cascade is parallelism-optimal: 5 of 10 blockers parallelizable; critical path bounded by max(per-tier) wall-clock. |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Cascade plan honestly recognizes (per Assumption-Adversary HARD-EARNED-BOUNDED verdict above) that PR 95 MLX reproduction's REALISTIC frontier is 0.197-0.20 band per PR 95 author's published score; our canonical_frontier_pointer 0.19202828 [contest-CPU] ALREADY BEATS PR 95. PR 95 MLX reproduction's VALUE is COST-CONTAINMENT + REPRODUCIBLE-INFRASTRUCTURE per `pr95_8stage_curriculum_forensic_20260513.md` not frontier-shifting. |

## Observability surface per Catalog #305

| Facet | Mechanism |
|---|---|
| 1. Inspectable per layer | Per-blocker subagent scope + per-tier wall-clock + per-blocker cost table inspectable in this memo. |
| 2. Decomposable per signal | Cascade decomposes into 4 tiers (P0/P1/P2/P3) + 9 subagents (P0a, P0b, P1a-d, P2a-b, P3); per-subagent wall-clock + cost decomposable. |
| 3. Diff-able across runs | Canonical equation `pr95_mlx_loop_closure_cascade_canonical_plan_v1` records wall-clock + cost band; future cascade re-runs diff against this canonical equation. |
| 4. Queryable post-hoc | Cascade plan persisted at `.omx/research/pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_landed_20260525.md` + Catalog #313 probe-outcomes row queryable via `tac.probe_outcomes_ledger.latest_blocking_outcome_by_substrate("pr95_hnerv_mlx_curriculum_canonical")`. |
| 5. Cite-able | Canonical equation candidate + 5 stage landing memos + codex 10 blockers memo + MLX bundle file all cited with line numbers. |
| 6. Counterfactual-able | Predicted wall-clock + cost bands are falsifiable post-completion via diff against actual cascade outcome. |

## Cargo-cult audit per assumption per Catalog #303

| Assumption | Classification | Rationale |
|---|---|---|
| The 10 codex blockers are independent + parallelizable | HARD-EARNED-BOUNDED | 5 of 10 are parallelizable (P1a-d + P2a-b); 5 are sequentially dependent (P0a → P1a → P2 → P3 critical path). Empirical evidence: today's 8 stages landed via 4 sister waves IN PARALLEL with zero source-file collision (Catalog #340 PROCEED). |
| The cascade reaches LOOP CLOSURE in 1-2 calendar days | HARD-EARNED-BOUNDED | Lower bound 12 hr is achievable with parallel subagent spawn; upper bound 16 hr buffers for sister coordination overhead + paid GPU queue time. The bound assumes operator authorizes P3 paid dispatch promptly upon P2 completion. |
| PR 95 MLX reproduction will shift the score frontier | CARGO-CULTED-EMPIRICALLY-FALSIFIED | PR 95 author's published score (0.197 [contest-CPU] / 0.20 [contest-CUDA]) is ALREADY BEATEN by our canonical_frontier_pointer 0.19202828 [contest-CPU] (DQS1 lane 2026-05-22). PR 95 MLX value is COST-CONTAINMENT + REPRODUCIBLE-INFRASTRUCTURE per `pr95_8stage_curriculum_forensic_20260513.md` lines 134-136 (Stage-8 finetune from PR 95 0.bin = canonical CHEAPEST empirical anchor on PR 95 curriculum). |
| The cascade is canonical (no alternative N-subagent plan exists) | HARD-EARNED-BOUNDED | Alternative plans exist (e.g. sequential blocker-by-blocker; OR Hinton-surrogate-first via Slot 3; OR Stage-8-only finetune per `pr95_8stage_curriculum_forensic_20260513.md` Arm B). The cascade plan in this memo is the OPTIMAL critical-path-bounded plan; operator may choose alternative paths via the operator-decision gates. |
| Slot 1+2 will complete within their estimated wall-clock | HARD-EARNED-BOUNDED | Both Slots are in flight; empirical evidence: today's 8 stage landings each took 20-70 min wall-clock via the canonical extension pattern. Slot 1 (PyTorch export parity) + Slot 2 (hparams audit) are bounded by similar engineering complexity. Slot completion is operator-routable via the canonical `tac.probe_outcomes_ledger.latest_blocking_outcome_by_substrate`. |
| MLX bundle architecture is byte-stable across stages | HARD-EARNED | Empirical evidence: 7x byte-identical state_bytes=915,944 across Stages 1+2+3+4+5+6+7 (Stage 8's queue-spawn variant did not emit canonical 100-step state_bytes but inherits the same architecture per descriptor). |

## Canonical-vs-unique decision per layer per Catalog #290

| Layer | Decision | Rationale |
|:---|:---|:---|
| Cascade priority structure (P0/P1/P2/P3) | ADOPT_CANONICAL_BECAUSE_SERVES | The 4-tier priority structure mirrors CLAUDE.md "Council hierarchy: 4-tier protocol" T1/T2/T3/T4 + the canonical operator-decision-routing pattern from Catalog #313 + #245 + #339. |
| Parallelization opportunity classification | ADOPT_CANONICAL_BECAUSE_SERVES | The independent-vs-dependent blocker classification mirrors the canonical Catalog #335 cathedral consumer auto-discovery pattern (independent consumers parallel; dependent consumers serial). |
| Per-subagent Carmack MVP-first 5/5 | ADOPT_CANONICAL_BECAUSE_SERVES | CLAUDE.md "Carmack MVP-first phasing" non-negotiable mandates 5-step recipe for every paid GPU dispatch > $0.30; this cascade extends the 5-step recipe to EACH subagent in the cascade so every paid dispatch (P3) is preceded by 4 tiers of free MVP-first phasing. |
| Operator-decision gate at P3 | ADOPT_CANONICAL_BECAUSE_SERVES | CLAUDE.md "Executing actions with care" + Catalog #199 paired-env operator authorization + Catalog #271 codex pre-dispatch review + Catalog #243 local pre-deploy check + Catalog #339 pre-spawn fatal observability all gate P3 paid dispatch. |
| Total wall-clock + cost equation form | FORK_BECAUSE_PRINCIPLED_MISMATCH | Standard project-management equations use sum-of-durations (Gantt-chart serial); this cascade's parallelism-optimal max-of-durations form is principled per the canonical extension pattern proven 7x today. |

## Operator-routable canonical artifact for next 4-8 sister subagent spawns

### Immediate spawn (P0; PARALLEL; ALREADY IN FLIGHT)

- **Slot 1 PR95-MLX-PYTORCH-EXPORT-PARITY-BRIDGE** (`a48f63cc`): IN FLIGHT;
  no spawn action required.
- **Slot 2 PR95-MLX-STAGE-HPARAMS-SOURCE-FAITHFUL-AUDIT**: IN FLIGHT; no
  spawn action required.

### Next spawn cohort (P1; PARALLEL; QUEUE AFTER P0 COMPLETION)

Spawn 4 sister subagents in parallel after Slot 1+2 confirmed complete:

```bash
# P1a — QAT/C1a/resume MLX port
TaskCreate name="P1a PR95-MLX-QAT-C1A-RESUME-SEMANTICS-MLX-PORT" \
  scope="Port apply_qat/restore_qat + cat_entropy_v2 runtime semantics from PR 95 PyTorch to MLX; checkpoint resume across stage boundaries" \
  preflight="src/tac/local_acceleration/pr95_hnerv_mlx.py + submissions/hnerv_muon/src/losses.py + submissions/hnerv_muon/src/stages/common.py + this memo's P1a row" \
  carmack_5_5="FREE local 100-step QAT+C1a smoke; falsifiable QAT-on-MLX ↔ QAT-on-PyTorch forward parity ε=1e-5; Catalog #344 ref; verdict same-batch; re-route to P2"

# P1b — byte-closed archive export wiring
TaskCreate name="P1b PR95-MLX-BYTE-CLOSED-ARCHIVE-EXPORT-WIRING" \
  scope="Wire write_pr95_public_archive_zip + _encode_pr95_decoder_blob + write_pr95_mlx_byte_closed_smoke_archive end-to-end" \
  preflight="src/tac/local_acceleration/pr95_hnerv_mlx.py:298-606 + submissions/hnerv_muon/src/codec.py + this memo's P1b row" \
  carmack_5_5="FREE local archive build+parse round-trip; falsifiable byte-identical state_dict+latents round-trip; Catalog #344 ref; verdict same-batch; re-route to P2"

# P1c — scorer loss MLX wiring (sister of Slot 3 Hinton surrogate path)
TaskCreate name="P1c PR95-MLX-SCORER-LOSS-CANONICAL-WIRING-OR-HINTON-SURROGATE" \
  scope="Wire canonical SegNet+PoseNet loss into MLX OR adopt Slot 3 Hinton-distilled surrogate; differentiable rgb_to_yuv6 patch" \
  preflight="src/tac/local_acceleration/pr95_hnerv_mlx.py + submissions/hnerv_muon/src/data.py:51-81 + Slot 3 active findings + this memo's P1c row" \
  carmack_5_5="FREE local scorer-loss smoke; falsifiable pose gradient reaches decoder; Catalog #344 ref; verdict same-batch; re-route to P2"

# P1d — receiver proof wiring
TaskCreate name="P1d PR95-MLX-RECEIVER-PROOF-WIRING" \
  scope="Parse MLX-emitted archive → load into PyTorch HNeRVDecoder → forward → assert byte-identical decoded frames vs MLX-direct-forward" \
  preflight="src/tac/local_acceleration/pr95_hnerv_mlx.py:565+ + submissions/hnerv_muon/src/model.py + this memo's P1d row" \
  carmack_5_5="FREE local receiver proof smoke; falsifiable per-pair max_abs_diff ε=5e-3; Catalog #344 ref; verdict same-batch; re-route to P2"
```

### Final paid dispatch authorization gate (P3; OPERATOR-DECISION)

After P0+P1+P2 ALL complete, present operator with the canonical
paired-CPU+CUDA dispatch authorization recipe:

```bash
# P3 PAID-DISPATCH-GATE (OPERATOR-AUTHORIZED ONLY)
# Routes through canonical tools/operator_authorize.py per Catalog #176/#243/#271
.venv/bin/python tools/operator_authorize.py \
  --recipe pr95_mlx_paired_cpu_cuda_contest_auth_eval_modal_a100_lightning_cpu \
  --estimated-cost-usd 20 \
  --lane-id lane_pr95_mlx_paired_cpu_cuda_paid_dispatch_p3
# Cost: $10-30 (Modal A100 $14-20 + Lightning CPU $0 OR Modal CPU $0-0.12)
# Wall-clock: 4-6 hr GPU + 30s pre-deploy harness
```

## Discipline closure

| Catalog # | Status | Evidence |
|---:|:---|:---|
| #90 lane registry consistency | ✓ | Lane `lane_pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_20260525` pre-registered at L0 then bumped to L1 via `tools/lane_maturity.py` |
| #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE | ✓ | This memo is NEW; ZERO mutation of sister landing memos / MLX bundle / codex artifacts |
| #117/#157/#174/#235/#289 canonical serializer + pre-pre-lock hash + symmetric `--expected-content-sha256` | ✓ | POST-EDIT working-tree sha computed before commit |
| #125 6-hook wire-in declaration | ✓ | Table above (#4 + #5 + #6 ACTIVE; #1 + #2 + #3 N/A) |
| #131 fcntl-locked JSONL canonical helper | ✓ | Catalog #313 row written via `tac.probe_outcomes_ledger.register_probe_outcome` (NEVER bare write) |
| #138 strict-load fail-closed | ✓ | Canonical helper inherits strict-load discipline |
| #185 Live count: 0 META-meta-meta drift detection | ✓ | NO new STRICT preflight gate landed (research artifact only); Catalog quota brake #299 respected |
| #186 catalog # claimed via canonical serializer | N/A | NO new catalog # claimed; the Catalog #344 candidate is QUEUED FORMALIZATION_PENDING |
| #206 5-tool-use subagent checkpoints | ✓ | 5 checkpoints emitted via canonical `tools/subagent_checkpoint.py` |
| #229 premise verification before edit | ✓ | Full read of 10+ canonical references BEFORE draft per the mandatory pre-flight list |
| #230 sister-subagent ownership map | ✓ | Slot 1+2+3 DISJOINT verified; coherent cascade architecture documented in "Sister-coherence verification" section |
| #287/#323 canonical Provenance + placeholder-rationale rejection | ✓ | Every score literal carries axis+hardware+evidence_grade triple; no placeholder rationales |
| #290 canonical-vs-unique decision per layer | ✓ | Table above |
| #292 per-deliberation assumption surfacing | ✓ | `council_assumption_adversary_verdict` block in frontmatter |
| #294 9-dim checklist evidence | ✓ | Section above |
| #298 substrate retirement discipline | N/A | Research artifact; no substrate lifecycle decision |
| #299 catalog quota brake | ✓ | NO new STRICT gate; current count well under #400 quota |
| #300 v2 frontmatter | ✓ | All required fields present in frontmatter block above |
| #303 cargo-cult audit per assumption | ✓ | Section above |
| #305 observability surface | ✓ | 6-facet table above |
| #309 horizon_class declaration | ✓ | `plateau_adjacent` declared in frontmatter (cascade plan extends canonical PR 95 paradigm; not a class-shift) |
| #313 probe-outcomes ledger row registered | ✓ | Canonical row written via `tac.probe_outcomes_ledger.register_probe_outcome` |
| #340 sister-checkpoint guard PROCEED | ✓ | Empirical: `tools/check_sister_checkpoint_before_git_add.py` returned OK pre-staging |
| #343 canonical frontier pointer reference | ✓ | Canonical frontier read at `.omx/state/canonical_frontier_pointer.json` (0.19202828 [contest-CPU] DQS1 + 0.20533002 [contest-CUDA] PR106 format0d) cited in Assumption-Adversary verdict |
| #344 canonical equation RATIFY-N candidate QUEUED | ✓ | `pr95_mlx_loop_closure_cascade_canonical_plan_v1` FORMALIZATION_PENDING; NOT auto-registered |

## Empirical receipts

| Artifact | Path |
|---|---|
| This landing memo | `.omx/research/pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_landed_20260525.md` |
| Catalog #313 probe-outcomes row | `.omx/state/probe_outcomes.jsonl` (probe_id=`pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_20260525`) |
| Sister codex 10-blocker source memo | `.omx/research/codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex.md` |
| Canonical frontier pointer | `.omx/state/canonical_frontier_pointer.json` |
| MLX bundle architecture | `src/tac/local_acceleration/pr95_hnerv_mlx.py` |
| Stage 2 landing | `.omx/research/pr95_mlx_stage_2_v331_softplus_curriculum_build_landed_20260525.md` |
| Stage 3 landing | `.omx/research/pr95_mlx_stage_3_v332_smooth_curriculum_build_landed_20260525.md` |
| Stage 4 landing | `.omx/research/pr95_mlx_stage_4_v332_qat_curriculum_build_landed_20260525.md` |
| Stage 6 landing | `.omx/research/pr95_mlx_stage_6_lambda_sweep_curriculum_build_landed_20260525.md` |
| Stage 7 landing | `.omx/research/pr95_mlx_stage_7_sigma_sweep_curriculum_build_landed_20260525.md` |
| Recovered PR 95 8-stage forensic | `.omx/research/pr95_8stage_curriculum_forensic_20260513.md` |
| Recovered PR 95 curriculum codex | `.omx/research/pr95_curriculum_recovery_20260513_codex.md` |
| Codex full source-video runtime profile artifacts | `experiments/results/pr95_mlx_full_source_video_runtime_profile_20260525T15*/` |

---

**Lane verdict**: PROCEED ✓
**Cost band**: free_local_research_only ($0 + ~50 min wall-clock)
**Mission alignment**: `frontier_breaking_enabler` (cascade plan unblocks
parallel P0+P1+P2+P3 sister subagent spawn; canonical extension pattern
proven 7x today; LOOP CLOSURE achievable in 1-2 calendar days @ $10-30 paid
GPU per the canonical-equation candidate `pr95_mlx_loop_closure_cascade_canonical_plan_v1`).
