<!--
generated_at: 2026-05-08T23:26:58Z
from_state_hash: 61c3b8cce4f221ca37988c2a382ff59568df7daa
regenerated_by: claude:fix-all-bugs-pass
-->

# Latest Report - 2026-05-04 PR106 belt_and_suspenders adapter contest-faithful status

## 2026-05-08 (evening) — Recursive hardening + Phase A ablation pass

**Headline:** 11 STRICT/warn preflight gates landed, ~395 violations
extincted, META-META commit-machinery protections live, 4018 long-lived
artifacts classified, Strategic Secrecy Rule retired (contest is over),
Phase A ablations A1–A4 + A3-alt anchored.

**Preflight gates (Catalog #109 → #119):**
- #109 public PR intake clones pristine — STRICT @ 0
- #110 recovery_metadata append-only — STRICT @ 0
- #111 status no-stale-dirty-paths — STRICT @ 0
- #112 rebuild no-baked-state — STRICT @ 0
- #113 META artifact_lifecycle umbrella (`enumerate_unregistered=True` strict-flip after 4018-path classification) — STRICT @ 0 on changed-paths scope
- #114 PR101 synthetic-target gate — STRICT @ 0
- #115 packet-blocker evidence-matches — STRICT @ 0
- #117 serializer-uses-lock — warn-only (legacy backlog clearing naturally)
- #118 catalog-no-duplicate-numbers — STRICT @ 0
- #119 co-author-trailer — warn-only (legacy backlog)

**META-META commit-machinery (commit `4695d222`):**
- FIX-1: working-tree pre/post-lock hash check (concurrent-edit-leak detection, refuses rc=3)
- FIX-2: `tools/claim_catalog_number.py` atomic via fcntl on `.omx/state/next_catalog_number.txt`
- FIX-3: serializer auto-appends Co-Authored-By trailer (idempotent)
- FIX-4: `audit_unregistered_long_lived_artifacts()` enumerates `git ls-files` under LONG_LIVED_ARTIFACT_ROOTS

**Strategic Secrecy Rule retired** (commit `e6806fa0`): contest is over.
Code-level cross-references in `src/tac/deploy/`, `optimal_stack_orchestrator.py`,
and `tools/oss_publish_staging.py` updated to descriptive language without
rule citation (commit `648b498c`).

**Auto-fork-PR tooling for GHA CPU eval** (commit `406b4211`):
- `tools/create_fork_pr_for_submission.py` clones fork, branches, copies
  submission_dir to `submissions/<name>/`, opens draft PR, returns PR number
- `tools/dispatch_cpu_eval_via_github_actions.py --auto-create-fork-pr` flag
  closes the runtime-contract gap from `pr102_cpu_eval_gha_runtime_contract_failure_20260508_codex.md`

**Phase A Pareto (current state, see `reports/phase_a_pareto_20260508.md`):**

| Lane | Archive bytes | Δ vs brotli (178,144 B) | Verdict |
|---|---:|---:|---|
| A0 mdl_baseline | — | — | byte_proxy_only_deterministic |
| A1 score_gradient | (dispatch tooling landed; infra-blocked on Lightning GPU/Vast.ai credit) | — | dispatchable |
| A2 xavier_l2 | 156,344 | -21,800 | FALSIFIED proxy (-3,635 B regression vs uniform) |
| A3-alt mallat_wavelet | 156,344 | -21,800 | incremental_improvement_insufficient (Mallat > Xavier in 2/4 cells; both fail uniform) |
| A4 charm_hyperprior toy | byte-tight (CARM2 wire format) | — | dispatch-ready ($15 Lightning T4 awaiting authorization) |
| ADMM_lossy_coarsening_baseline | 147,285 | -30,859 | Path B baseline; -28 KB savings @ 4-5% rel_err |

**Class-level finding:** TWO weight-domain importance proxies (Xavier-L2 +
Mallat wavelet) have now failed to beat uniform on PR101's near-iid
substrate. Future Decision 3 reactivation MUST use score-domain
(Hessian-trace, score-gradient) or byte-domain (compression-hardness)
proxies, NOT a third weight-domain proxy. Per CLAUDE.md "KILL is LAST
RESORT", the lane is `incremental_improvement_insufficient`, not killed —
reactivation criteria documented per memo.

**A1 + A4 + A3-alt review verdict:**
- A1 dispatch tooling: 3 CRITICAL fixes applied (`load_differentiable_scorers`
  signature, canonical `simulate_eval_roundtrip` resize cycle, stale claim
  closure structural fix), 2 Medium fixes, 1 advisory closed. Re-fire
  ready when Lightning GPU attached or Vast.ai topped up.
- A4 ChARM range coder: 4 Medium fixes applied (deterministic ZIP, CARM2
  framing docs, dead param), 4 Low findings closed. R1-1 entropy caveat
  documented. NOT a dispatch blocker.
- A3-alt Mallat: see Pareto table above.

**Phase A subagents (LANDED):**
- A4-alt Filler STC pose codec (`75c99b84`) — 559+ LOC + 27 tests pass.
  Δ vs PD-V2 on smooth-walk fixture: **−400 B (−9.17%)**; idle-dominant
  +52% (expected — AC exploits qint=0 dominance). Verdict
  `byte_anchor_landed`. Ledger:
  `feedback_pr101_pose_filler_stc_byte_anchor_landed_20260508.md`.
  First empirical anchor at
  `experiments/results/pr101_pose_filler_stc_20260508T194527Z/`.
- A5 frame-conditional bit budget — byte-closed runtime packet landed, but
  the current complexity-only schedule is a measured-config negative:
  macOS CPU advisory score **1.937884** (`pose=0.078646`,
  `seg=0.009361`, `172,615 B`). Reactivate with score-domain q-bit
  allocation, not the η=4 complexity schedule.
- A6 Selfcomp block-FP × Ballé hyperprior compose — measured-config
  negative. Best compose **B=64, sq=uint8 = 214,035 B**, which loses to
  PR101 brotli by +35,891 B. Treat this as a scoped proxy retirement for
  the current max-abs-scale conditional Gaussian range-coder, not as a
  score-lowering lane and not as a family kill. Real Selfcomp per-channel
  block-FP, learned ChARM, tensor-aware PMFs, and byte-map-preserving
  arithmetic rewrites remain valid reactivation paths. Linear σ
  map cannot match brotli's adaptive context modelling on PR101's
  near-iid INT8 stream.
- PHASE 4 INTEGRATION paper harness — landed.

**Phase A in flight:** none. All 7 lanes anchored.

**Inline closure (post-A4-alt push, `0f50b5c5`):**
After A4-alt landed at `75c99b84`, full-codebase preflight surfaced 2
pre-existing `check_training_scripts_have_auth_eval` violations on the A1
+ A4 training scripts (training scripts that save a checkpoint must invoke
auth_eval OR define an explicit `--no-auth-eval-on-best` opt-out flag).
The opt-out is now no longer allowed to be default-on; default-on
exemptions are treated as guard bypasses, and downstream lanes remain
responsible for exact auth eval. Sister fix: SyntaxWarning escape
`\|Δ\|` removed from
`tools/pr101_omega_opt_per_tensor_codec_choice_empirical.py:35`.

**Awaiting operator:**
- A1 dispatch ($8 Lightning T4): Lightning GPU attach OR Vast.ai credit topup
- A4 dispatch ($15 Lightning T4): authorization with documented R1-1 caveat
- PARADIGM-δεζ (#307) + PHASE 4 INTEGRATION (#308): major work needing strategic alignment

---

## 2026-05-08 — CUDA vs CPU auth eval split: leaderboard ranks by CPU axis

**Headline:** the contest's `upstream/evaluate.py` produces two distinct
authoritative score axes for the same archive bytes — `--device cuda` and
`--device cpu` — and the public leaderboard ranks by the **CPU** score, not
CUDA. We discovered this 2026-05-08, four days after the deadline closed.

**Empirical content** (5 paired bot comments, no new compute):

| | mean | σ | range |
|---|---:|---:|---|
| `R_pose = pose_cuda / pose_cpu` | 5.04 | 0.10 | [4.97, 5.21] |
| `R_seg  = seg_cuda  / seg_cpu`  | 1.17 | 0.01 | [1.16, 1.18] |
| `Δscore = score_cuda − score_cpu` | 0.0330 | 0.0004 | [0.0325, 0.0335] |

70% of the gap sits in pose, 30% in seg, 0% in rate (rate term bit-identical).

**Mechanism hypothesis (3-axis additive):** (a) decoder split — `DaliVideoDataset`
for CUDA hardware NVDEC vs `AVVideoDataset` for CPU libavcodec software decode
produces different RGB uint8 from the same `.mkv`; (b) PoseNet FP32 conv-stack
noise floor with `σ²_cuda ≈ K · L · ε² · ||x||²`, predicting R_pose ≈ 5.0 at
medal-band pose<sub>cpu</sub> = 3.5e-5 — numerically perfect match for observed
5.04; (c) Hydra-head MLP. The original "FastViT TF32-attention compounding"
hypothesis was falsified on three independent grounds (RepMixer not attention,
T4 has no TF32 hardware, PyTorch default `allow_tf32=False`).

**Score-formula amplification:** `sqrt(10·d_pose)` softens the 5× pose ratio
to 2.24× in absolute pose-term contribution, but pose contribution is *larger*
in absolute score points on CUDA than on CPU (0.0417 vs 0.0187). At PR106's
operating point the May 4 race postmortem rule "pose 2.71× more marginal than
seg" *inverts* on the leaderboard substrate — SegNet becomes ~4× more
leaderboard-marginal than pose.

**Predicted CPU scores for our existing artifacts:**

| Archive | CUDA score | Predicted CPU score | Predicted leaderboard rank |
|---|---:|---:|---|
| Our PR #107 apogee | 0.22936 | 0.1966358879 (verified Linux x86_64 CPU) | just outside silver/bronze band |
| PR103-on-PR106 AC repack | 0.20898 | ~0.176 | above current top |
| PR102 hardened replay | 0.22839 | 0.19538 (verified) | bronze (verified) |
| PR104 hardened replay | 0.23114 | ~0.198 | non-frontier |

**Operational rule (CLAUDE.md non-negotiable):** every shippable archive gets
dual-eval — authoritative `[contest-CUDA]` and `[contest-CPU]` axes on
Linux x86_64 hardware that is 1:1 contest-compliant with the GitHub Actions
CI runner. Apple Silicon CPU eval is `[macOS-CPU advisory only]`, never
`[contest-CPU]`. Tag distinctness enforced by Check D / B7.

**Strategic exploitation prescriptions:**

1. **CPU-axis pose trust-region loss**: ablate a Huber-style pose loss where
   τ ≈ sqrt(observed HNeRV CPU pose band). This is a training-time hypothesis,
   not proof that pose below τ is free; paired CPU/CUDA exact eval must decide.
2. **Leaderboard-aware Lagrangian**: `tac.score_geometry target_axis="cpu_leaderboard"`
   reweights pose marginal × 0.20, seg marginal × 0.86 before ranking
   dispatch candidates.
3. **Training-time SegNet boundary robustness**: make rendered frames more
   robust around class boundaries. Do not run SegNet or smooth scorer logits in
   `inflate.py`; submission inflate remains scorer-free.
4. **Calibrated noise injection in training**: σ ≈ 1.7e-3 per-op-equivalent.
   Tightens R<sub>pose</sub> from ~5 to ~3.5.

**Documentation surfaces:**

- OSS findings: `docs/findings/cuda_cpu_auth_eval_split_20260508.md`
- Internal methodology: `docs/writeup/cuda_cpu_drift_methodology.md`
- Paper: `docs/paper/00_abstract.md` (4th contribution), `docs/paper/04_results.md` §4.8,
  `docs/paper/06_related_work.md` §6.8, `docs/paper/07_discussion.md` §7.9
- Site: `reports/graphs/site/cuda_cpu_split_post.md`,
  `reports/graphs/site/judges_one_pager.md` (post-deadline section)
- README addendum: project root `README.md`
- Memory: `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md`,
  `feedback_cuda_cpu_drift_sweep_research_design_20260508.md`,
  `feedback_cuda_cpu_pose_drift_mechanism_deep_dive_20260508.md`,
  `project_pr102_replay_drift_0_228_vs_claimed_0_195_20260508.md`,
  `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`,
  `feedback_representation_integration_gates_landed_20260508.md`
- Research: `.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md`,
  `.omx/research/cuda_cpu_pose_drift_mechanism_deep_dive_20260508_claude.md`,
  `.omx/research/representation_integration_gap_audit_20260508_codex.md`,
  `.omx/research/public_replay_drift_hypothesis_20260508_codex.md`

**Operator next-action menu (P0 → P3):**

1. **P0** — Modal CPU smoke replay of any one HNeRV archive (PR102 ideal:
   verified 0.19538). $0.12, 90 min. Validates the Modal CPU substrate is
   1:1 with the contest CI's CPU axis.
2. **P0** — PR #107 apogee CPU replay is complete via GitHub Actions Linux
   x86_64: `0.1966358879`. Next work is paired candidate sweeps, not confirming
   this prediction.
3. **P1** — 3-PR cross-family smoke (PR106, PR104, PR91). $1.26, 3-6 hours.
   Decisive between hypotheses H1/H2/H3/H4 in the sweep design.
4. **P1** — `tac.score_geometry target_axis="cpu_leaderboard"` flag (1 LOC,
   $0). Reweights cathedral autopilot for leaderboard-targeted dispatch.
5. **P2** — Decoder-vs-FP32 discriminator microbench: dump PyAV-decoded
   tensors on CPU, feed those shared tensors through CUDA forward, and compare
   against CUDA+DALI and CPU+PyAV. Do **not** force `AVVideoDataset` under
   `--device cuda`; upstream `AVVideoDataset` asserts a non-CUDA device.
   ~$0.25 / 1-hour T4. Decisive between decoder-mismatch and FP32-floor
   contribution to the gap.
6. **P3** — Layer-by-layer PoseNet/Hydra drift profile to falsify / confirm
   shared-input additive-noise model. This is RepMixer/conv/MLP drift, not a
   FastViT attention-softmax probe.

## 2026-05-08 — Evidence-grade drift supersession after PR102/PR104/PR106

- PR102 hardened replay is exact T4 CUDA A++ at `0.22839372989108092`, but it confirms public CPU/leaderboard drift and is not a new archive-byte win or local frontier.
- PR104 is no longer an evidence hole: hardened exact T4 CUDA replay landed at `0.23113446620399658`, non-frontier versus the active PR103-on-PR106 anchor.
- PR106 UNIWARD-Lagrangian `rms_target=0.05` exact T4 CUDA landed at `0.3371617511972341` and is `A-negative scoped forensic` for that measured runtime packet only: no promotion, no rank-frontier use, no family kill.
- Treat `0.20454` / `178873` as an unanchored formula projection in older roadmap/autopilot surfaces unless a matching exact CUDA artifact is cited. Prediction, CPU, MPS, proxy, and byte-only rows cannot promote, rank, kill, or dispatch without exact packet custody.
- Supersession ledger: `.omx/research/evidence_grade_drift_pr102_pr104_pr106_codex_20260508.md`.

## 2026-05-08 — Path B Ω-OPT 6/8 anchored; cross-paradigm 137,531 B figure RETRACTED to byte_proxy_only_NOT_deployable (REVIEW-ENG C1)

### Frontier state

- **Local A++ HNeRV rate anchor (active)**: PR103-on-PR106 = `0.2089810755823297` [contest-CUDA] / `185,578` bytes / SHA256 `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- **Public-leaderboard medal band** (May 4 race): 0.193 / 0.195 / 0.195 (PR #101 / #103 / #102)
- **Phase 4 INTEGRATION predicted band** (commit e8ae721c): 0.155-0.175 [predicted-band, NOT contest-CUDA]; 0.140-0.180 conservative
- **Theoretical Shannon floor target** (per CLAUDE.md "Frontier target"): keep advancing toward 0.155 and below

### CPU-prep empirical anchors (real PR101 substrate, NOT contest-CUDA)

| Composition | Bytes | vs anchor 185,578 | Source |
|---|---:|---:|---|
| ~~Path_B_step6_ADMM_x_continuous_K_then_Op1 (cross-paradigm)~~ **RETRACTED** | ~~137,531~~ | n/a | commit 8d33d5c1; per REVIEW-ENG C1 this is `len(blob_op1)` byte-proxy of the Op1 re-encode of the dequantized fp32 substrate, NOT a byte-closed archive. `cuda_eval_worth_testing=False`; dispatch_blocker `137531_byte_proxy_not_byte_closed_archive`. WIRE-DECODER subagent owns the deployable composition. |
| Path B step 6 ADMM standalone (byte-closed candidate) | 153,699 | -31,879 | commit 82bfc648 [CPU-prep] (rel_err 4.15%); dispatch_blocker `apogee_int6_contest_cuda_anchor_required_first` (REVIEW-ENG C3) |
| Path B step 5 Joint-ADMM Lagrangian | 150,000 | -35,578 | commit b8aa5c43 [CPU-prep] (rel_err 4.36%) |
| Path B step 4 HStack codec-CHOICE | 156,344 | -29,234 | commit 4f2cfd55 [CPU-prep] (analytical, rel_err 3.86%) |
| Op2_alone (canonical 8-stack winner) | 161,942 | -23,636 | [CPU-prep, lossless] |
| Op1_alone | 162,202 | -23,376 | [CPU-prep, lossless] |
| Op3_int6 → Op1 (substrate-mismatch BALLOONED) | 309,470 | +123,892 | [CPU-prep, mismatch evidence] |

All CPU-prep rows: `score_claim = False`, `promotion_eligible = False`, `ready_for_exact_eval_dispatch = False`. The retracted 137,531 B figure is a byte-proxy of the standalone Op1 re-encode on a dequantized fp32 substrate AFTER ADMM coarsening; it does NOT include the K side-info, fp16 scales, or PR101 latent_blob/sidecar that an actual archive must carry, and there is no inflate.py that can read this composition end-to-end. The 153,699 B Path-B-step-6 standalone IS byte-closed (forked inflate.py + matching encoder); REVIEW-ENG C3 attaches `apogee_int6_contest_cuda_anchor_required_first` to the same row because 4.15% rel_err → score mapping is unmeasured.

### Empirical findings

- **Allocation MECHANISM dominates codec-basis on PR101** — Joint-ADMM Lagrangian (step 5) beats greedy by 12-65 KB at rms_target=0.05; step 6 confirms (continuous-K basis at 152,420 B@4.33% within 2,420 B of step 5)
- **PR101 codec lane saturated at ~150 KB byte-floor at 4-5% rel_err** — remaining headroom requires CUDA distortion validation, not codec cleverness
- **Substrate-mismatch in BOTH directions** — Op3 designed for HNeRV/PR106 ballooned PR101 archives by +147K-200K B; PR101 byte-maps yield only -241 B on PR106. Codecs are NOT portable across substrates without retuning.
- **Composability taxonomy** — STACKABLE: Op3, β (transforms_state_dict=True); SUBSTITUTIONAL: Op1, Op2, γ (independent terminal codec choices)

### CUDA-confirmed negative this cycle

- **lossy_coarsening_analytical** Lightning T4 dispatch = `0.3517 [contest-CUDA A-negative]` at 156,404 archive bytes. The byte-anchor 156,344 B @ 3.86% rel_err did NOT translate to predicted 0.189 score. Status: `measured_config_retired` (per-tensor K budget 0.05). Reactivation requires scorer-aware retrain.

### In flight

- **arch_shrink_x0.4 Q-FAITHFUL retrain** Lightning T4 (active job
  `arch-shrink-x0-4-lightning-20260508T024304Z`) — still running at
  `2026-05-08T11:53:01Z`; no terminal `contest_auth_eval.json` yet. First
  exact CUDA architecture-lane anchor remains pending harvest.

### Adversarial review gate active

4 commits pending clearance: 256d6fe1 (Lightning canonical bootstrap fix), 82bfc648 (ADMM byte-closed candidate), e8ae721c (Phase 4 design memo), 8d33d5c1 (cross-paradigm winner anchor).

### Bug class extincted

- `forbidden_remote_bootstrap_inline` re-violation closed by commit 256d6fe1: 7 sequential dep-discovery failures (uv → ensurepip → cu124 → find → brotli → timm) replaced with one canonical `bash scripts/remote_archive_only_eval.sh` invocation. The wrapper installs full dep closure (uv + ffmpeg + scorer deps) and auto-pins INFLATE_TORCH_SPEC by driver version.

### Next dispatches (per `.omx/state/next_experiments.md`)

1. ~~137,531 B cross-paradigm winner CUDA dispatch~~ **RETRACTED** (REVIEW-ENG C1); WIRE-DECODER subagent in flight to build deployable composition with matching inflate.py
2. 153,699 B ADMM byte-closed candidate CUDA dispatch — pending review clearance + apogee_int6 [contest-CUDA] anchor (REVIEW-ENG C3)
3. PARADIGM-δεζ Phase 2 GPU implementation — pending arch_shrink result + apogee_int6 [contest-CUDA] precondition

---

## Cross-PR Lineage Finding — 2026-05-07T16:00Z

**The May 4 medal-band entries are bolt-ons on PR #100's substrate, not bespoke codecs.** Bit-level deconstruction of the gold/silver/bronze archives shows each medal entry inherits PR #100's archive layout (BradyMeighan, hnerv_lc_v2, 0.1954) and adds a small focused delta:

- **Gold (PR #101 SajayR, 0.193)**: PR #100 substrate + schema-driven decoder + split-Brotli streams + per-tensor byte-map permutations
- **Silver (PR #103 rem2, 0.195)**: PR #100 substrate + arithmetic-coding bolt-on (241 LOC in 2 files)
- **Bronze (PR #102 EthanYangTW, 0.195)**: PR #100 archive bytes + inference-time scale 0.0095 + frame-0 nudges (zero new codec work)

**Strategic implication**: at this score band the contest does not reward from-scratch codec design. Once one team makes its inflate/compress code public, every other team can read it and start bolting on. Engineering velocity wins.

**Substrate-mismatch corollary**: PR101's per-tensor byte-maps were tuned against PR101's fine-tuned weights. On the PR106 substrate they yield only -241 bytes (33× shortfall vs the -7,963 bytes on PR101's own substrate). Codec wins are not portable across substrates without retuning. Surfaced into:

- `.omx/research/findings.md` (top entry, 2026-05-07)
- `docs/paper/06_related_work.md` (new §6.4 "PR lineage and bolt-on engineering")
- `.omx/research/pr_extended_bit_level_lineage_pr95_pr100_pr101_pr103_20260507_claude.md` (full byte-level evidence)
- `.omx/research/four_way_stack_composition_contract_20260507_claude.md` (canonical reuse pattern via `tac.codec_pipeline.CodecOp`)

**Why now**: per user 2026-05-07: "that finding about the PR submission using another submissions stuff is interesting we should mention that in our writeup and everywhere and findings".

## Supersession — 2026-05-08 Monolithic Layout And Active HNeRV Rate Anchor

The PR106x wording in the 2026-05-07 addenda below is historical. Later exact
eval and layout work promotes PR103-on-PR106 as the active local A++ HNeRV
rate anchor: strict formula score `0.2089810755823297`, report-reconstructed
score `0.20898105277982337`, `185578` bytes, SHA-256
`ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`.
PR106/PR106x remains a predecessor substrate/control; rate-only candidates
must beat `185578` bytes or explicitly stack with a scorer-changing packet.

The monolithic-layout correction also supersedes any member-name-derived
pose/mask/renderer budget assumption for PR101/PR106-style HNeRV archives.
Those archives have single ZIP members and parser-proven internal sections;
there is no separate PR106 pose or mask ZIP member to replace.

## Track A/B/C/D/E/F/G/H Addendum — 2026-05-07T11:00Z

**Superseded historical wording**: this addendum originally treated
`PR106x-lowlevel-brotli` (`0.20935073680571203`, `186080` bytes) as the
current frontier. It is now a predecessor exact control below the active
PR103-on-PR106 anchor above.

### Parallel-track session deliverables

| Track | Artifact | Commit | Status |
|-------|----------|--------|--------|
| A | apogee_int6 Lightning T4 dispatch (predicted [0.190, 0.215]) | staging dea0c7a6 / outcome 5c219803 | **BLOCKED** Lightning AWS T4 capacity persistent (3 attempts 10:16-10:49Z); Vast.ai out of credit. Operator decision: reload credits, pivot to Azure $200, or wait. |
| B | Wave-Ω Ω-1 CUDA scorer wiring | aa6a464d | **DONE** — JFG+SegNet+PoseNet differentiable score_fn at experiments/build_sjkl_residual.py:179; 10 new tests + 60 SJKL suite green |
| C | scorer_basin_parity_gate tooling + apogee_int6 evidence | e991cdcd, f04b7d5b | **DONE** — int6 PASSED (pose_delta 30× below threshold); reusable for any apogee_intN |
| D | PARADIGM-δεζ Phase 1 scaffolding (3 modules, lane registry, pipeline.py flags) | 6fe8125b | **DONE** — 3 lanes registered, 53 tests pass, 540+432+493 LOC stub modules |
| E | Wave-Ω Ω-2 NeRV mask inflate branch | (pre-existing) | **DUPLICATE-OF-EXISTING** — already wired at inflate_renderer.py:1780+; 4th stale-blueprint signal of session |
| F | Wave-Ω Ω-3 block-FP JFG transplant module | in flight (subagent aca10101ef68a5607) | Predicted -0.035 score (~52KB savings); CPU-only |
| G | apogee_int5 + int7 basin-parity evidence | (artifact-only) | **DONE** — int7 PASS (rel_err 0.79%, safe), int5 FAIL (rel_err 3.31%, catastrophic regime); int5 DEFERRED-pending-research |
| H | reports/latest.md + lane registry update | (this commit) | **DONE** — apogee_int6/int7 marked strict_preflight=true (L2) |

### Empirical apogee_intN safety boundary (pinned 2026-05-07)

| Variant | Bytes | rel_err % | basin-parity | Verdict |
|---------|-------|-----------|--------------|---------|
| int4 | 109996 | 7.09 | (skipped) | Falsified at 1.4287 [contest-CUDA T4] |
| int5 | 154555 | 3.31 | **FAIL** (pose_delta 2.26× threshold) | DEFERRED-pending-research |
| **int6** | **170450** | **1.55** | **PASS** (pose_delta 30× below) | **Safe — primary dispatch candidate** |
| **int7** | **205158** | **0.79** | **PASS** (pose_delta 18× below) | **Safe — control dispatch candidate** |
| int8 | 187731 | 0.24 | (lossless-class) | Calibration anchor (0.2112) |

The boundary between safe and catastrophic regimes lies between rel_err 1.55% and 3.31%. Future PTQ schemes should target ≤2.0% rel_err to safely cross.

### Fan-out plan when GPU returns

Per CLAUDE.md race-mode rigor inversion + parallel-dispatch-first non-negotiable: dispatch **{int6, int7}** in parallel via `tools/parallel_dispatch_top_k.py` for double-anchor calibration of the safe regime. Each candidate has its readiness-evidence-json staged; predispatch_sanity ALL 5 GATES PASS (exit 0) for both.

Expected outcomes (from predicted-band, NOT contest-CUDA):
- int6: [0.190, 0.215] — historical comparison against the superseded 0.20935
  PR106x frontier; current rate-only anchor is PR103-on-PR106 at
  `0.2089810755823297`
- int7: [0.198, 0.208] — control variant (smaller savings, near-lossless distortion)

If both land in their predicted bands, that's two new lossy anchors at rel_err 0.79% and 1.55% — completes the apogee_intN curve calibration in the safe regime (currently only int8 at 0.24% and int4 at 7.09%; the gap between is exactly what int6/int7 fill).

### Operator decisions pending

1. Vast.ai credit reload (~$25) → unblocks immediate dispatch
2. Azure $200 free-credits pivot (per task #312, wired but unused; needs `az login`)
3. AWS spot $100 free-credits pivot (per CLAUDE.md memory; less canonical)
4. Wait for Lightning T4 AWS capacity to return (free; unknown ETA)

## Worker D Addendum - 2026-05-07

Superseded historical scorecard routing treated
`PR106x-lowlevel-brotli` as the lowest exact CUDA row already in custody:
score `0.20935073680571203`, bytes `186080`, archive SHA-256
`b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`,
eval artifact
`experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json`.
This is an existing exact lossless-repack control, not a new family claim, and
is now below the active PR103-on-PR106 A++ HNeRV rate anchor (`185578` bytes).

The historical next exact-evaluable score-lowering target closest to that
PR106x predecessor frontier was the same archive's `decoder_packed_brotli`
section: `170127` charged bytes,
section SHA-256
`07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`.
Next gate is a byte-different archive with old/new section SHA-256 and charged
byte proof, followed by exact CUDA auth eval only after the Level 2 lane claim.
Supersession: current rate-only work must compare against the PR103-on-PR106
`185578`-byte floor.
No dispatch was attempted for this update. Durable note:
`.omx/research/hnerv_frontier_hidden_gem_ranking_20260507_worker_d.md`.

## Superseded May 4 Exact Public Frontier

**Superseded May 4 public-frontier floor**: PR106 `belt_and_suspenders`
adapter replay = `0.20945673680571203` [A++]. This remains source-substrate
and predecessor evidence, but the active local HNeRV rate anchor is now
PR103-on-PR106 at `0.2089810755823297` / `185578` bytes.

- Evidence: `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json`
- Adapter: `experiments/results/public_runtime_adapters_20260504_codex/pr106_belt_and_suspenders_adapter/inflate.sh`
- Archive bytes/SHA: `186239`, `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- Components: SegNet `0.067142000` (avg 0.00067142), PoseNet `0.018305737` (avg 0.0000335), Rate `0.124009000`, samples `600`, device Tesla T4 CUDA, gpu_t4_match=true
- Runtime adapter wraps the public PR106 inflate path through repo-managed `.venv/bin/python` with brotli closure; preserves archive bytes
- Authority: `.omx/research/public_hnerv_frontier_deconstruction_20260504_codex.md`

**Predecessor (now superseded)**: PR100 HNeRV-LC-v2 adapter replay `0.22826947142244708` [A++].

**Score gap unlocked**: PR106 beats PR100 by `0.01881273461673505` and beats PR101 (the earlier exact best at the time of last report) by `0.01689658`. The dominant driver is PoseNet contribution `0.018306` vs PR101's `0.041355` — a ~2.3× pose-distance reduction in exchange for +5,314 archive bytes (rate cost +0.0053). Supersession after the monolithic-layout finding: this is a score-component comparison, not proof of a separate pose ZIP member or member-level pose budget.

### Cross-validating x-repack confirmation

Two byte-different deterministic ZIP-member-name repacks (changing `0.bin` → `x`, saving 8 ZIP-header bytes) reproduce the predicted `~6.66e-7/byte` rate-only effect within float epsilon:

- PR106 adapter `0.20945673680571203` vs PR106 xrepack `0.20945123680571204` — Δ = `5.500e-6` for 8 bytes saved (matches `25/37545489 × 8 = 5.326e-6`).
- PR105 adapter `0.23043732986984997` vs PR105 xrepack `0.23043182986984995` — same Δ.

This is decisive evidence the runtime adapter path is byte-faithful and our public-frontier intake gate is calibrated.

The PR100 adapter replay supersedes the PR95 stem-permutation repack
(`0.23089404465634825`, bytes `178277`, SHA
`e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`) by
`0.00156292999674471` score points despite adding `115` archive bytes. The
win comes from improved SegNet distance with a small PoseNet/rate tradeoff.
The adapter changes the runtime call contract for exact contest replay while
preserving the PR98 archive bytes; cross-run comparisons must include the
runtime tree hash above. It also supersedes both the conservative PR95 repack
(`0.23091954465634829`, bytes `178321`, SHA
`2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`) and the
exact public PR95 replay
(`0.23098329465634826`, bytes `178417`, SHA
`e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`). It
saves `44` bytes versus the conservative repack and `140` bytes versus the
public PR95 replay. It also supersedes the previous
PR85-family exact anchor, `PR85+STBM1BR+PR92/RMB1` at
`0.2535063602939779`. None of these rows is a Shannon-floor attainment claim;
the next tranche targets PR98/PR99 HNeRV deconstruction, PR91/HPM1 parity
recovery, and public-release hygiene.

## Public Context

| Entry | Status | Score signal | Evidence use |
|---|---:|---:|---|
| PR #98 HNeRV/Muon adapter replay | Public-source adapter exact T4 | Exact T4 `0.22826947142244708`, bytes `178981` | Superseded exact predecessor; PR100 source attribution required |
| PR #99 HNeRV/Muon LC adapter replay | Public-source adapter exact T4 | Exact T4 `0.2297226895103603`, bytes `178546` | Superseded exact A++ PR98-family predecessor; PR99 source attribution required |
| PR95 stem-permutation repack | Internal exact T4 | Exact T4 `0.23089404465634825`, bytes `178277` | Superseded exact predecessor |
| PR95 conservative repack | Internal exact T4 | Exact T4 `0.23091954465634829`, bytes `178321` | Superseded exact predecessor |
| PR #95 HNeRV/Muon public archive | Public/open replay | Exact T4 `0.23098329465634826`, bytes `178417`; public body reports `0.1987048012202245` | Superseded exact source anchor; body/CPU score is external only |
| PR #96 rem2_HNeRV | Open/self-reported | Public body reports `0.20567121179282477`, bytes `186631`; no local exact replay in this packet | External frontier signal only until exact CUDA replay |
| PR85+STBM1BR+PR92/RMB1 | Internal exact T4 | Exact T4 `0.2535063602939779`, bytes `229480` | Superseded PR85-family exact anchor |
| PR85 + STBM1BR mask recode | Internal exact T4 | Exact T4 `0.25369011029397787`, bytes `229756` | Superseded exact PR85-family anchor |
| PR #91 HPM1 hybrid | Open/self-reported | PR reports `0.24879480490416128`; local replay currently fails HPM1 entropy decode | External signal only until parity and exact CUDA replay |
| PR #85 adaptive masking joint frame model | Public/open replay | Exact T4 `0.25806611029397786`, bytes `236328` | Superseded exact anchor and source for STBM recode |
| PR #84 QMA9/no-router | Public/open replay | Exact T4 `0.2751401491321396`, bytes `215735` | Superseded public replay frontier; source attribution required |
| PR #81 QMA9/range-mask | Public/open replay | Exact T4 `0.2812078926981712`, bytes `215960` | Superseded exact frontier; QMA9 mask stream anchor |
| PR #82 Henosis | Public/open replay | Exact T4 `0.2983246102939779`, bytes `296789` | Non-frontier A++ transfer target |
| PR #79 `qpose14_r55_segactions_minp` | Official leaderboard top row at rounded `0.31` | Internal exact replay/repack frontier `0.31453355357318635` before PR81 | Superseded by PR81 exact replay |
| PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` | Open external PR | Reports rounded `0.31`, bytes `276564`, PoseNet `0.00048597`, SegNet `0.00061000` | External target; mask segment used in C-067 with attribution |
| PR #65 `henosis_qz_n3z_r25_clean` | Open external PR | Reports `284425` bytes, local replay `0.31968`/`0.3600` | Side-channel correction motivation; multi-stage residual paradigm reverse-engineered |
| PR #63/#64 public-floor lineage | Merged/visible | `0.32`/`0.33` rounded band | Basin anatomy and packer transfer |
| C-067 | Internal exact A++ superseded | Recomputed `0.31561703078448233`, bytes `276214` | Historical PR67 fixed-slice frontier |
| C-063 | Internal exact A++ superseded | Recomputed `0.3156230307844823`, bytes `276223` | Predecessor in PR67 comparison chain |
| C-059 | Internal exact A++ superseded | Recomputed `0.3157055307844823`, bytes `276347` | Predecessor in lossless-repack chain |
| C-058 | Internal exact A++ superseded | Recomputed `0.3157555307844823`, bytes `276422` | Active-subspace byte micro-frontier predecessor |
| C-057 | Internal exact A++ superseded | Recomputed `0.3157562807844823`, bytes `276423` | Anisotropic-basis pose comparison anchor |
| PR #68 `loophole_v2` | Closed external PR | Proof-of-concept moving payload to script | Quarantine as `invalid`/`external_quarantine`; archive-metering loophole risk |
| PR #69 `houdini` | Open external PR | No maintainer-filled eval report at inspection time | Quarantine as `external_quarantine`; unverified boundary experiment |
| PR #70 `mask_decoder` | Open external PR | Reports rounded `0.19`, bytes `57329`, author states bytes moved into `inflate.py` | Quarantine as `invalid`/`external_quarantine`; not leaderboard-comparable |

PR #67 remains the most relevant contest-faithful external source. C-067 locally evaluates a charged fixed-slice candidate that uses the PR67 mask segment with C-059 model/pose bytes; the local exact T4 score is A++ evidence for the exact archive bytes and simultaneously requires PR67 source attribution for the mask segment. PR #68/#69/#70 are useful only for compliance hardening: they show why payload closure must meter every score-affecting byte and why our reports must reject script-side payloads, hidden sidecars, malformed ZIP reliance, or uncharged runtime data.

## Frontier Custody (last 24h)

Codex pose-manifold water-fill lineage (C-058 → C-059 → C-063 → C-067) on H100 NVL/SXM with Lightning Tesla T4 promotion. All micro-frontier improvements are pure charged-rate (PoseNet/SegNet identical, byte deltas only), reflecting the C-059-basin packer-and-layout exhaustion. Q-FAITHFUL successor work continues separately as `B`/`A-negative` evidence (zoom-runtime fix verified; measured snapshot still PoseNet-collapsed at `22.1476`); retire only that snapshot/export, not the future QAT++ or geometry-trained architecture.

CMG2 exact T4 wave landed as `A-negative scoped forensic` (plain 2x2: `2.295` at `194020` bytes; top512 AMR1 repair: `2.125` at `248074` bytes; top256 AMR1 repair: `2.223` at `219850` bytes). These retire the measured nearest-neighbor CMG2 base + hand-picked AMR1 repair only; learned/predictive/row-span/geometry-preserving mask grammars remain open. Predictive mask-grammar row-span probe is `empirical_byte_probe_only` (best `63212` charged bytes for `row_span_stride4_class_predictor`+`lzma6` on PR67 mask, `-156260` vs current `219472`-byte segment) and motivates CMG3 closed-archive implementation as the next charged-evidence step.

Current observability tightens the next tranche. C-067 byte accounting is
control-plane evidence only, but it shows the exact unchanged-distortion
sub-`0.300` crossing requires `23454` fewer bytes (`23455` buffered), with
stream pressure concentrated in `masks.mkv` (`219472` bytes) and `renderer.bin`
(`55965` bytes); `optimized_poses.bin` is only `677` bytes and ZIP overhead is
`100` bytes. The generated markdown/PNG profiles live at
`experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md`
and `experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.png`.

PMG atomtop4068 is now an L40S CUDA `A-negative scoped forensic`, not a T4
frontier row: `experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/contest_auth_eval.json`
recomputed score `28.41411894150047`, archive `195762` bytes, SHA
`2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497`,
PoseNet `62.34251404`, SegNet `0.03315286`, `600` samples, NVIDIA L40S. It
retires PMG row-run-only rescue for this implementation and blocks another
PMG row-run-only T4 promotion; it does not kill learned mask grammar,
multimask reconciliation, atom planning, or pose-conditioned residuals.

SJ-KL moved from theory toward production integration without becoming score
evidence. The robust-current runtime can consume charged `sjkl.bin`
additively with shape checks and no scorer imports, and the target-slot bug is
closed by using JointFrameGenerator pair slot `0` (`fake1`) consistently. The
smoke manifest
`experiments/results/sjkl_tensor_prep_c067_smoke_20260502/sjkl_pair_tensor_prep_manifest.json`
prepared `4` pairs; the full manifest
`experiments/results/sjkl_tensor_prep_c067_full_20260502/sjkl_pair_tensor_prep_manifest.json`
prepared `600` pairs. Both are `build_tensor_prep_only`,
`score_claim=false`, and `promotion_eligible=false`.

## Submission Pipeline

Historical submission-packet context: the PR100 adapter replay was the May 4
submission packet champion in this section, but it is not the current local
HNeRV score/rate anchor. Current local HNeRV anchor remains PR103-on-PR106 as
listed above. The PR98 packet is:

- packet:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`
- archive:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter/archive.zip`
  (`178981` bytes, SHA
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`)
- inflate:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter/inflate.sh`
  plus the copied PR100 HNeRV-LC-v2 adapter runtime source tree
- attribution: PR #98 public HNeRV/Muon archive and runtime source
- runtime custody:
  `inflate_runtime_manifest.runtime_tree_sha256=ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- compliance audit: strict pre-submission gate passed with `78` checks and no
  failed checks in
  `experiments/results/submission_packet_pr100_adapter_20260504/pre_submission_compliance.json`

The previous PR95 stem-permutation upload directory is now superseded for score
wording:

- packet:
  `experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm`
- archive:
  `experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm/archive.zip`
  (`178277` bytes, SHA
  `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`)
- inflate:
  `experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm/inflate.sh`
  plus the copied PR95 HNeRV/Muon runtime source tree
- attribution: PR #95 public HNeRV/Muon archive and runtime source
- runtime custody:
  `inflate_runtime_manifest.runtime_tree_sha256=a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7`
- compliance audit: strict pre-submission gate passed with `80` checks and no
  failed checks in
  `experiments/results/submission_packet_pr95_stemperm_20260504/pre_submission_compliance.json`

Public Apogee surfaces are prepared but not score authorities:

- PR body template: `docs/submission_template.md`
- public supplement plan:
  `docs/runbooks/apogee_public_supplement_20260502.md`
- Lightning notebook skeleton: `notebooks/apogee_lightning_supplement.ipynb`
- Cloudflare Pages sanitized bundle/runbook:
  `reports/graphs/public_site/` from `reports/graphs/build_public_site_bundle.py`
  and `reports/graphs/deploy_cloudflare_pages.md`

Before publishing, run strict public-release hygiene over those exact surfaces
and replace only sanitized placeholders such as `${LIGHTNING_SUPPLEMENT_URL}`,
`${CLOUDFLARE_PAGES_URL}`, `${APOGEE_ARCHIVE_ZIP_URL}`, and
`${APOGEE_RELEASE_MANIFEST}`. Do not publish private Lightning/Vast job links,
raw `.omx/state`, local absolute paths, or provider transcripts.

## Report Pipeline Contract

Every public or judge-facing packet should be generated from structured rows with these sections:

1. `frontier_summary` - only Grade `A++`/`A` rows; current default is PR103-on-PR106, with PR106, PR104, PR102, PR100 adapter replay, PR95 stem-permutation repack, PR95 conservative repack, PR95 public exact replay, PR85+STBM/RMB1, PR85, PR84, PR81, and C-067 retained as predecessor/context rows.
2. `public_external_context` - PR96 unresolved or self-reported claims, PR95 body/CPU score, PR91/PR67/PR65/PR63/PR64 anatomy and claimed scores tagged `external`.
3. `quarantined_exploit_context` - PR68/PR69/PR70-style sidecar or rule-boundary evidence tagged `invalid` or `external_quarantine`.
4. `exact_artifact_table` - archive path, SHA, bytes, eval JSON, device, samples, component values, recomputed score, evidence tag, allowed use, and `inflate_runtime_manifest.runtime_tree_sha256` for cross-run comparisons.
5. `negative_results` - exact scoped regressions only [contest-CUDA]; no broad method kills from proxy, CPU/MPS, byte-only, or exploit evidence (these are `[advisory only]` device classes).
6. `submission_checklist` - payload closure, deterministic ZIP, no scorer patches, no sidecars, no hidden files, `archive.zip -> inflate.sh -> upstream/evaluate.py`, CUDA/T4/equivalent proof, inflate budget, review signoff.
7. `next_wave_roadmap` - CMG3 closed row-span archive + exact CUDA gate, predictive/lossy mask grammar atoms, Q-FAITHFUL successor geometry, charged pose-basis atoms, hard-pair temporal windows, payload-efficient residuals, and packer/layout atoms; all blocked from promotion until exact CUDA archive evidence exists.

## Caveats

- Supersession: PR103-on-PR106 is the active local HNeRV rate anchor; PR100 adapter replay, PR95 stem-permutation repack, PR95 conservative repack, PR95 public exact replay, PR85+STBM/RMB1, PR85, PR84, PR81, and C-067 are predecessor rows. None is a Shannon-floor attainment claim.
- Public PRs and GitHub comments are external design signals unless we have exact archive bytes, SHA, CUDA eval JSON, component recomputation, and custody.
- PR #70's low reported score is non-comparable under our compliance policy because the public PR text says score-affecting bytes were moved from `archive.zip` into `inflate.py`.
- C-067 carries an external-source attribution requirement (PR #67 mask segment); the local score claim is A++ evidence for the charged archive bytes, but the mask source remains externally attributed per `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`.
- The H100 NVL diagnostic of the same C-067 archive bytes scored `0.36295` earlier; this is a runtime-custody warning, not a contradiction. Cross-run comparisons require matching `inflate_runtime_manifest.runtime_tree_sha256`.

## Cross-references

- Paper draft: `docs/paper/04_results.md` (stale PR95 frontier table; update before publication)
- Codex writeup ledger: `.omx/research/submission_writeup_integration_20260502_codex.md`
- Working notes: `reports/writeup_working.md` (live operating point)
- Submission pipeline runbook: `docs/runbooks/contest_submission_pipeline_20260502.md`
- Reverse-engineering refs: `reports/raw/leaderboard_intel_20260501/` (PR #65/#67 inflate.py + archive.zip + line_search.py)
- Memory: `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md`, `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`

## Deadline

Contest deadline: **May 3 11:59 PM AOE = May 4 06:59 AM CDT (May 4 11:59 UTC)**. This report is now in final handoff mode for the confirmed PR100 adapter champion, not pre-deadline queue planning.

## Internal sub-frontier launch-ready candidate — Lane Ω-W-V3 (post-deadline)

**Lane Ω-W-V3** (water-filling codec v2 → PR106 HNeRV decoder) is fully scaffolded as of 2026-05-04 and ready for ~$0.30 GPU dispatch on Vast.ai 4090 (or T4-equivalent). Council 8/10 GO; predicted band [0.194, 0.204] [prediction, NOT contest-CUDA].

### Pipeline (5 commits today)

| Stage | File | Commit | Purpose |
|---|---|---|---|
| 1 | `experiments/extract_pr106_decoder.py` | `45149f21` | Unpack PR106 0.bin → 28 tensors / 228,958 params + 600×28 latents |
| 2 | `experiments/build_sensitivity_map_pr106.py` | `5ca67264` | Per-channel β-Fisher (CUDA-required); CPU stub OK |
| 3 | `experiments/repack_pr106_with_water_filling.py` | `b2f958a4` | Per-Conv2d water-fill OWV2 + brotli-int8 fallback → apogee_v2 0.bin |
| 4 | `submissions/apogee_v2/inflate.{py,sh}` + src | `c7f237eb` | Inverse parser; **round-trip VERIFIED** on real bytes |
| 5 | `scripts/remote_lane_omega_w_v3_pr106.sh` | `ac3053ef` | 4-stage GPU dispatch wrapper |

### Stub-mode preview (CPU, all-ones sensitivity) [empirical:experiments/results/apogee_v2_repack_20260504_claude/repack_metadata.json]

| Metric | PR106 (superseded public-frontier replay/control) | Apogee-v2 stub | Δ |
|---|---:|---:|---:|
| Archive bytes | 186,239 | 164,087 | **−22,152 (−11.9%)** |
| Rate-component score | 0.124 | 0.109 | **−0.01475** |
| Predicted total score (if distortion holds) | 0.20945673 | **~0.1947** | **SUB-0.20** |

Real β-Fisher (CUDA dispatch) could shift bytes among layers and improve further. Distortion delta is UNKNOWN until contest_auth_eval runs on T4 — predicted band [0.194, 0.204] per audit assumes the per-channel water-fill quantization preserves PR106's PoseNet sensitivity (the dominant attack surface at 0.018306).

### Operator one-liner (when GPU dispatch approved)

```bash
PR106_ARCHIVE=experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
bash scripts/remote_lane_omega_w_v3_pr106.sh
```

Anchor: `experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_01_water_filling_codec_v2_pr106_decoder.md` (Council 8/10 GO).

## Lane #04 intN expansion (NEW 2026-05-04 evening) [empirical:experiments/repack_pr106_with_intN_block_fp.py]

The Lane #04 ternary-block_fp falsification was revived as a Pareto sweep over signed intN block-FP variants (int4..int8). Verified end-to-end byte-decodable locally (CPU stub mode):

| Variant | Magic | Archive bytes | Rate Δ | Rel err per weight | Risk | Predicted band [prediction, NOT contest-CUDA] | Pareto |
|---|---|---:|---:|---:|---|---|---|
| int4 | `0xA4` | 109,996 | −0.0508 | 7.1% | HIGH | [0.155, 0.180] | FRONTIER |
| **int5** | **`0xA5`** | **154,555** | **−0.0211** | **3.3%** | **MEDIUM** | **[0.180, 0.196] — sweet spot** | FRONTIER |
| int6 | `0xA6` | 170,450 | −0.0105 | 1.55% | LOW | [0.190, 0.204] | FRONTIER |
| int7 | `0xA7` | 205,158 | +0.0126 | 0.79% | VERY LOW | [0.198, 0.208] | **DOMINATED by int8** |
| int8 | `0xA8` | 187,731 | +0.0010 | 0.24% | ALMOST LOSSLESS | [0.196, 0.207] | FRONTIER |

**Pareto-domination finding (NEW)**: int7 is +18,919 bytes vs int8 for the *same* VERY-LOW distortion class. Non-byte-aligned int7 packing requires shift/mask boilerplate per block whereas int8 fits cleanly. Reactivation criterion only if a future packer eliminates the bit-aligned overhead (e.g., 2× int7 → 14-bit aligned, or context model on the bit-stream).

**Lane #04 closed-loop chain — all four components landed this session:**
- Producer: `experiments/repack_pr106_with_intN_block_fp.py` (`82ca9456`) — generic `--bits N`
- Inflate adapter: `submissions/apogee_intN/inflate.{py,sh}` + vendored `src/intn_codec.py` (`62e3a51c`)
- Dispatch wrapper: `scripts/remote_lane_apogee_intN.sh` (`5c31755e`) — operator picks bits via `APOGEE_INTN_BITS` env
- Decision matrix doc: `docs/pr106_stacking_decision_table_20260504.md` (`7b354a1a`)

**Operator one-liners** (Vast.ai 4090 ~$0.30 / 30min each):

```bash
APOGEE_INTN_BITS=5 bash scripts/remote_lane_apogee_intN.sh   # sweet spot
APOGEE_INTN_BITS=6 bash scripts/remote_lane_apogee_intN.sh   # safe fallback
APOGEE_INTN_BITS=4 bash scripts/remote_lane_apogee_intN.sh   # high-risk high-reward
APOGEE_INTN_BITS=8 bash scripts/remote_lane_apogee_intN.sh   # almost-lossless calibration
```

For full operator-ready `launch_lane_on_vastai.py full ...` commands per
Pareto-frontier variant: `.venv/bin/python tools/apogee_intN_pareto.py`.

## Dispatch feedback trio (NEW 2026-05-04 evening)

A complete pre-/in-/post-dispatch tooling stack now lives in `tools/`. Each
piece reads on-disk artifacts independently — no shared state, no orchestrator.

| Tool | Phase | Reads | Output |
|---|---|---|---|
| `tools/apogee_intN_pareto.py` | **pre-dispatch** | `experiments/results/apogee_int*_repack_*/repack_metadata.json` | Pareto-frontier table + `launch_lane_on_vastai.py full ...` one-liners per non-dominated bits config |
| `tools/score_dashboard.py` | **post-landing (general)** | `experiments/results/**/contest_auth_eval*.json` | Sorted view (best score first) of every contest_auth_eval ever produced; non-CUDA rows marked `*` per CLAUDE.md MPS-auth-eval-is-NOISE |
| `tools/predicted_vs_actual_reconciler.py` | **post-landing (apogee_intN)** | both of the above | Per-bits in-band/out-of-band check + beats-PR106 verdict |

10 regression tests pass (5 per tool that emits subprocess one-liners), all
catching dead-flag-wiring (CLAUDE.md NEVER-INVENT-CLI-FLAGS), schema drift,
band malformation, and stale wrapper-path renames. Smoke-verified on the
live empirical manifests for all 5 bits configs.

**Lane registry promotions (Check 90 STRICT validates clean)**: 4 apogee_intN
lanes now at L2 INTEGRATION (impl_complete + real_archive_empirical satisfied).
int7 also registered at L2 with a `Reactivation: <criteria>` note per the
"KILLED lanes get registry entries too" discipline. Pre-existing stale-path
findings on `lane_owv3_0120_stack` and `lane_line_search_pose_refinement`
also repaired this session.

## Recovery + Lane SJ-KL launchability (NEW 2026-05-04 evening)

A defensive-validation pass uncovered 8 lost helpers from a subagent-worktree-cleanup bug class (subagent built helpers + wrapper, only the wrapper got committed before quota; helpers wiped on cleanup). Full recovery doc: `docs/recovery_report_20260504.md`. Bug class structurally extinct via two new preflight checks (PCC9 + PCC9b, warn-only initially) that scan shell scripts + test files for runtime-executed file references that don't resolve to disk.

**Lane SJ-KL C067 is now end-to-end runtime-decodable** after the 6 lost SJ-KL Python modules were rebuilt from runbook + addendum spec:
- `src/tac/sjkl_basis.py` — full codec library (basis + alpha-block V1+V2 + full payload + runtime aliases), 26 tests pass
- `experiments/build_sjkl_residual.py` — Lanczos top-K Fisher-info basis + per-pair alpha quantization → sjkl.bin
- `experiments/prepare_sjkl_pair_tensors.py` — frame-pair I/O orchestration
- `experiments/build_sjkl_c067_archive.py` — `top_level_sibling` layout fully implemented (preserves source `p` bytes exactly)
- Plus the recovered `tools/claim_lane_dispatch.py` (full) + `scripts/ensure_remote_uv.sh` (canonical uv installer)

End-to-end pipeline verified locally: prepare → build_residual → 942-byte sjkl.bin → build_archive → charged ZIP `{p untouched, sjkl.bin charged}` → decode via `tac.sjkl_basis`: byte-faithful basis + alpha extraction.

**Operator one-liner** (Vast.ai 4090, separate base archive from PR106-stacking lanes):

```bash
bash scripts/remote_lane_sjkl_c067.sh
```

## Strategic findings — late 2026-05-04 dashboard-mining session

After the dashboard log-parser fix (commit dbb0032d, +293 score visibility),
three cross-PR analyses produced a coherent strategic picture:

1. **PR106 vs PR101 (+0.017 gap) is training-recipe, NOT codec** — PR106 uses
   an 8-stage training pipeline (`stage1_v328_ce` → `stage8_muon_finetune`)
   that drops pose distortion 5× to 0.000034. Our codec-side work is correctly
   anchored on the best-trained decoder. See
   `docs/pr106_vs_pr101_training_recipe_finding_20260504.md`.

2. **PR97 anti-pattern: pose marginal value > seg at PR106's operating point**
   — PR97 traded pose for seg (-65% seg, +18× pose) and lost 0.042 net. The
   marginal sensitivity formula `5/sqrt(10*pose_avg)` evaluates to 271 at
   PR106's pose_avg=0.000034 vs SegNet's constant 100, so pose marginal value
   is 2.71× seg. CLAUDE.md's "SegNet 77× more important" heuristic was at the
   OLD 1.x operating point — at PR106's level the marginal value FLIPS. Our
   pre-registered sidechannel lanes both target pose, validating direction.
   See `docs/pr97_anti_pattern_pose_vs_seg_marginal_20260504.md`.

3. **PR family evolution: 4 paradigm-shift eras** —
   - Era 1 (PR63-64, qpose14): ~287KB / 0.325-0.331
   - Era 2 (PR81-85, range_mask): ~215-240KB / 0.258-0.298 (-0.05)
   - Era 3 (PR95-105, HNeRV): ~178KB / 0.226-0.231 (-0.05)
   - Era 4 (PR106, HNeRV+8-stage): 186KB / 0.2095 (-0.017)

   Within-era variance: ±0.005 (codec polish). Between-era jumps: -0.05 to
   -0.10 (architecture/training paradigm shifts). Our lanes operate within
   Era 4 — predicted total post-stacking range 0.180-0.205. Beating PR106 by
   0.05+ requires Era 5 (multi-week research-grade work). See
   `docs/pr_family_evolution_timeline_20260504.md`.

**Master INDEX**: `docs/INDEX_score_aware_sidechannel_thread_20260504.md` ties
together the 9-memo paradigm thread + 6 sidechannel variants + 2 pre-registered
PR106-stacking lanes (`lane_pr106_latent_sidecar` + `lane_pr106_yshift_sidechannel`).

**Operator handoff snapshot**: `docs/operator_handoff_snapshot_20260504.md`
captures the 4 actionable choices with recommendation (Choice 1: dispatch
apogee_int5 at $0.30 — RECOMMENDED).

## Updated Next Queue

1. Use `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter` as the release packet unless a newer exact T4 A++ packet is explicitly promoted.
2. Keep PR100 source attribution and the exact T4 custody block with every public/judge-facing score claim.
3. Run strict public-release hygiene on the exact PR body, notebook, and site bundle before publishing URLs.
4. Keep PR96, PR91/HPM1, and any public body/CPU scores in external context until exact CUDA replay lands.
5. **Lane Ω-W-V3 GPU dispatch (~$0.30)** — council 8/10 GO, predicted [0.194, 0.204] [prediction, NOT contest-CUDA]. Operator approval gate plus fresh dispatch claim and exact artifact custody are required before score use.
6. **Lane #04 int5 GPU dispatch (~$0.30)** — predicted [0.180, 0.196] [prediction, NOT contest-CUDA], MEDIUM risk, sweet spot of the int-N Pareto. Worth testing only through normal exact-eval gates.
7. **Lane #04 int6 GPU dispatch (~$0.30)** — predicted [0.190, 0.204] [prediction, NOT contest-CUDA], LOW risk fallback if int5 distortion exceeds tolerance.
8. **Lane SJ-KL C067 GPU dispatch (~$0.30)** — uses C067 base archive (not PR106), so it is independent from the PR106-stacking lanes. Predicted score band TBD pending first exact run.

**Total dispatch matrix: 4 lanes × ~$0.30 = ~$1.20** for a historical high-EV
sweep proposal that would test whether any lane can land below PR106's exact
`0.20945673680571203`. No prediction row counts as contest-CUDA evidence until
the exact archive/runtime is harvested and reviewed.
