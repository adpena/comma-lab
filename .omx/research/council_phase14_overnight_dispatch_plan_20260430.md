# Grand Council — Phase 1-4 Honest Audit + Overnight (8-12h) Dispatch Plan

**Date**: 2026-04-30
**Convened by**: parent agent under user mandate "i don't believe we spawned most of our phase 1 experiments successfully and there were 8 lanes" — comprehensive 23-lane audit + tonight's dispatch list
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Mandate**: REPORT-ONLY. NO code modified. NO GPU spawned. All claims tagged `[empirical:<path>]` / `[contest-CUDA]` / `[contest-CPU advisory]` / `[Modal-T4-CUDA]` / `[prediction]` / `[derivation]`.

---

## 1. Executive Summary

### Honest status of all 23 lanes (8 Phase 1 + 15 Phase 2-4)

The user is correct. Of the 8 Phase 1 lanes:
- **1 GREEN-LANDED** (Lane 4 Ω-W water-fill — V2 empirical 40.98% local + STACK auth dispatch firing now via #272)
- **2 GREEN-LOCAL-READY** (Lane 2 PD-V2, Lane 5 LCT — bolt-ons not yet integrated into Lane G v3 archive)
- **1 KILLED-MEASURED** (Lane 6 GP — Runge phenomenon at degree-10 polynomial, score 89.67 Modal-T4-CPU advisory)
- **1 GREEN-DISPATCHED-NOW** (Lane HM-S firing via Council F approval, ~$1.75 in flight)
- **2 NEVER-SUCCESSFULLY-DISPATCHED** (Lane 3 STC clean-source — Modal NOT_READY all session; Lane 7 PSD — script landed never dispatched)
- **1 LOCAL-ONLY-PARTIAL** (Lane 8 multi-pass inflate MVP — `_default_inner_step` is `NotImplementedError`; only injectable-stub path tested)

Of the 9 SegMapTrainer-invalidated lanes (Round 6 correction), **Council F covered only 5** (HM-S APPROVE, FR-Ω APPROVE, WC-S DEFER, PA KILL, FC KILL). Council F NEVER ADDRESSED 4: SC++, SA-v2, SO, q_faithful's SegMap variant. **Verdict per Section 3 below**: KILL SC++ + SO; DEFER SA-v2; KILL q_faithful's SegMap variant (replaces, doesn't stack).

Of the 15 Phase 2-4 lanes, **3 are READY for parallel overnight dispatch on local CPU** (Joint-ADMM 4-stream test, Lane 19 SegNet logit-margin codec scaffolding, Lane 17 IMP cycle-1 smoke). The others are 1+ week of impl work each.

### Tonight's recommended dispatch list (within $30 Vast.ai cap)

Currently committed: ~$2.25 (HM-S $1.75 + Ω-W-V2 stack $0.50). Remaining: ~$27.75.

**Recommended overnight dispatch (5 GPU + 6 local-only)**:

| # | Lane | Cost | Wall | Predicted band | Tag |
|---|---|---|---|---|---|
| 1 | Lane G v3 + Ω-W-V2 + PD-V2 + LCT triple-bolt-on archive auth | $0.50 | 15min | 0.95 ± 0.02 | [derivation] |
| 2 | Lane G v3 + Ω-W-V2 + multi-bolt + half-frame masks (HF) | $0.50 | 15min | 0.92 ± 0.04 | [prediction] |
| 3 | Lane FR-Ω contest-CUDA (gated on HM-S calibration) | $1.50 | 6h | [0.27, 0.45] | [prediction] |
| 4 | Lane STC clean-source Modal T4 retry (with $0.50 cap) | $0.50 | 30min | [0.30, 0.50] | [prediction] |
| 5 | Lane J-IMP cycle-1 dispatch on Vast.ai 4090 ($25 cap → $5 cycle-1 only) | $5.00 | 18h | [0.85, 1.00] | [prediction] |

**Total GPU dispatch tonight**: $8.00 of remaining $27.75 budget. **20 hours of headroom** for follow-up Phase 4 Lane J-NWCS or FR-Ω dispatches as signal arrives.

**3 highest-EV overnight experiments** (sorted EV/$):
1. **Triple-bolt-on archive auth** — $0.50 / `[derivation]` -0.10 score → **EV/$ = 0.20/$** (chain-completion winner; OWV2 inflate handler already landed at inflate_renderer.py:2145)
2. **STC clean-source Modal T4 retry** — $0.50 / [prediction] -0.55 conditional → EV/$ = 0.11/$ if 30% hit-rate
3. **Lane G v3 + Ω-W-V2 + HF stack auth** — $0.50 / [prediction] -0.13 → EV/$ = 0.13/$ (compounds with multi-bolt-on)

---

## 2. Phase 1 status table (per Part 1)

The user's instinct is correct: most Phase 1 was either never-dispatched or invalidated. Honest table:

| # | Lane | Original verdict | Current dispatch state | Invalidated by .round() bug? | Council F verdict | **Tonight's verdict** |
|---|---|---|---|---|---|---|
| 1 | SC++/q_faithful | RED (KL-distill) | OOM-crashed Modal A10G; 5 spawn calls in cache; never re-dispatched | YES (SegMapTrainer .round() at segmap_renderer.py:281) | NOT ADDRESSED | **KILL** — KL-distill primary loss is forbidden per CLAUDE.md (PoseNet collapse). Re-dispatch only if council can vary KL weight to <=0.005 (Quantizr-style auxiliary). |
| 2 | Pose-delta (PD-V2) | GREEN | LANDED local (commit 152ba503); bolt-on NOT in any archive yet | NO | n/a | **LOCAL-TASK** — integrate PD-V2 into Lane G v3 + Ω-W-V2 stack archive (overnight task #1 in dispatch list) |
| 3 | STC clean-source | GREEN (gated CUDA) | Modal-dispatched, NOT_READY all session (queue) | NO | n/a | **RE-DISPATCH** ($0.50 Modal T4 with 30min cap) — clean-source argmax is the only path to STC ceiling claim |
| 4 | Ω-W water-fill (V2) | GREEN | V2 EMPIRICAL 40.98% local `[empirical:tests/test_omega_w_v2_real_archive.py]`; OWV2 inflate handler at inflate_renderer.py:2145; STACK DISPATCH FIRING via task #272 | NO | covered by chain-integrity audit | **GREEN-FIRING** — chain-integrity audit's G1 dispatch underway at $0.50 |
| 5 | LCT (learnable class targets) | GREEN | Local bolt-on, 8/8 tests pass; 10-byte payload | NO | n/a | **LOCAL-TASK** — integrate LCT into the triple-bolt-on archive |
| 6 | GP rerun | GREEN | KILLED — Runge phenomenon 89.67 [Modal-T4-CPU advisory]; baseline_poses fix landed but doesn't help (degree-10 polynomial + 600 equispaced points = oscillation) | NO | n/a | **KILL** — only revival is DCT/B-spline replacement (Phase 3 work) |
| 7 | PSD standard | YELLOW (script gap) | Script `scripts/remote_lane_psd_standard.sh` LANDED (Check 64 smoke pass per memory), NEVER DISPATCHED | NO | DEFERRED | **DEFER** — Lane G v3 + bolt-on stacks are higher-EV; PSD is high-PoseNet-risk per `project_phase1_dispatch_verdict_20260429.md` |
| 8 | Multi-pass inflate | GREEN-MVP | Outer-loop scaffolding LANDED (commit 0e43d299); `_default_inner_step` is `NotImplementedError`; only injectable-stub paths tested | NO | n/a | **DEFER to Phase 2** — production GPU inner-step not wired; per Round 5 §2.2 CONCERN-1, current claim "MVP wired" is a docstring-overstatement risk |

**Concrete next-step per Phase 1 lane**:

- **Lane 1 (SC++/q_faithful)**: KILL per non-negotiable; reopen only with KL-weight <=0.005 + auxiliary-only loss role + adversarial council approval.
- **Lane 2 (PD-V2)**: LOCAL — integrate into overnight task #1 (triple-bolt-on archive) ; deterministic CPU work.
- **Lane 3 (STC)**: RE-DISPATCH overnight task #4 (Modal T4, $0.50, 30min cap, hard kill if no archive emitted at 25min).
- **Lane 4 (Ω-W-V2)**: AUTH NOW (firing); chain-completion stack dispatch (overnight task #1).
- **Lane 5 (LCT)**: LOCAL — integrate into overnight task #1.
- **Lane 6 (GP)**: KILL; reopen only with DCT/B-spline (Phase 3 Lane 14-class work).
- **Lane 7 (PSD)**: DEFER (post-Lane G v3 stacks); high-PoseNet-risk per chain audit.
- **Lane 8 (Multi-pass)**: DEFER to Phase 2 (Lane 14 multi-pass compress optimization is the proper home).

---

## 3. 9-lane SegMapTrainer addendum (4 lanes Council F missed)

Per `feedback_round6_defects_lane_mm_correction_segmap_invalidation_extended_20260429.md`, total 9 invalidated SegMapTrainer lanes (Round 6 corrected; MM v2 correctly RETRACTED — BUILD-only). Council F covered: HM-S, FR-Ω, WC-S, PA, FC. **Council F NEVER ADDRESSED**: SC++, SA-v2, SO, q_faithful's SegMap variant.

| Lane | Mechanism | Original cost | Orthogonality vs Lane G v3 | **Tonight's verdict** | Justification |
|---|---|---|---|---|---|
| **SC++** (KL distill SegMap) | KL-distill weight=0.002 on SegMap variant | $5 (Modal A10G OOM history) | KL-distill OVERLAPS with Lane G v3's KL-distill (both target same loss surface) | **KILL** | Forbidden per CLAUDE.md "KL distill caused PoseNet collapse as primary loss". Even at weight=0.002 (Quantizr-style), the SegMap-trainer integration was the failure mode (.round() zero-grad). Re-dispatching after fix would replicate Lane G v3's mechanism on a SegMap renderer — replacement, not stack. |
| **SA-v2** (Selfcomp SegMap clone) | SegMap renderer + variant of Selfcomp 88K-param replica | $5 | Replacement (different renderer architecture) | **DEFER** | SegMap arch lands a NEW baseline if it succeeds, not a stack on Lane G v3. Predicted band [prediction] [0.40, 0.80] is wide; cost $5 with A10G OOM history. Re-dispatch only AFTER triple-bolt-on archive proves Ω-W-V2 ceiling vs Lane G v3 baseline — that result reveals whether SegMap-arch payoff justifies the $5. |
| **SO** (Hessian Block-FP SegMap) | SegMap + Hessian-curvature block-FP | $7 (Modal A10G + Hessian compute) | OVERLAPS with FR-Ω (Council A: "SO Hessian fallback bug"); Council F approved FR-Ω as the canonical Hessian-block-FP lane | **KILL** | SO and FR-Ω target identical mechanism (Hessian-cost driven block-FP). FR-Ω is the cleaner implementation per `project_grand_council_brutal_forecast_20260429.md` "promote FR-Ω over SO". SO is dominated by FR-Ω; do NOT re-dispatch. |
| **q_faithful SegMap variant** | TRUE 1:1 Quantizr replica with SegMap trainer (per `project_lane_q_faithful_design_20260428`) | $5-8 | SegMap renderer arch is a REPLACEMENT for Lane G v3's standard renderer | **KILL (variant) / DEFER (base)** | The SegMap-trainer variant of q_faithful was invalidated; the BASE q_faithful (Joint Frame Generator, no SegMap trainer) is a separate lane that was never invalidated and is a different decision. KILL the SegMap variant outright; if base q_faithful is to be revived, that is a Phase 2 architecture-replacement bet ($5+ Vast.ai 4090 with 12h cap), not a Phase 1 stack-completion lane. |

**Total saved by KILL/DEFER verdicts**: $17 to $25 of would-be dispatches that Council F left as dangling decision-debt. **None of the 4 lanes are tonight-dispatch candidates**; all are either KILLED outright or deferred behind higher-EV stack-completion work.

---

## 4. Phase 2-4 audit (per Part 3)

Council E (Round 5) reprioritized: ACCELERATE Lane 10 ADMM (DONE — coordinator skeleton landed commit 152ba503) + Lane 12 NeRV + Lane 19 SegNet logit-margin + Lane 17 IMP + Lane 20 Ballé hyperprior. PAUSE Lane 9 STC rebuild + Lane 11 wavelet + Lane 13 DARTS-S full + Lane 21 decoder rewrite + Lane 16 MDL.

| # | Lane | State | Can dispatch tonight in parallel? | Cost / Dependencies | Tonight's verdict |
|---|---|---|---|---|---|
| 9 | Full STC rebuild | ❌ NOT STARTED (clean-source pipeline needed) | Partial — overnight task #4 dispatches the EXISTING `scripts/remote_lane_stc_clean_source.sh` to Modal T4 ($0.50 cap, 30min) | $0.50 / depends on Modal T4 capacity | **DISPATCH CHEAP RETRY** — Modal NOT_READY all session per Phase 1; one more $0.50 attempt with hard 30min kill |
| 10 | Joint ADMM | 🟡 PARTIAL (coordinator + 1 codec wrapper landed) | YES local-only — 4-stream non-convex test (Council F gates V2 dispatch on this); $0 | $0 / `joint_admm_proximal_water_filling_v2.py` wrapper needed | **LOCAL-TASK #6** — write the 4-stream non-convex test per Council F §3.1; gates ADMM V2 |
| 11 | Wavelet residual codec | ❌ NOT STARTED | NO — 1-2 weeks of dev | $0 tonight | DEFER to Phase 3 |
| 12 | NeRV mask codec | ❌ NOT STARTED | NO — 2-3 weeks of dev | $0 tonight | DEFER to Phase 3 |
| 13 | DARTS-S full sweep | 🟡 PARTIAL (3-config restricted run on Vast.ai 4090) | NO — restricted sweep already running; full sweep $24-60 | n/a | KEEP RESTRICTED until calibrated bands available |
| 14 | Multi-pass compress | 🟡 PARTIAL (MVP scaffolding via Lane 8) | NO — `_default_inner_step` NotImplementedError | $0 | DEFER until Lane 8 GPU inner-step lands |
| 15 | Bit-level archive optimizer | ❌ NOT STARTED | NO — depends on Lane 14 | $0 | DEFER |
| 16 | Bayesian MDL/evidence | ❌ NOT STARTED | NO — depends on Lanes 9-15 results | $0 | DEFER |
| 17 | Full IMP 10-cycle | 🟡 PARTIAL (skeleton `experiments/train_imp_cycle.py` exists; cycle-1 only smoke-tested) | YES — $5/18h cycle-1 dispatch on Vast.ai 4090 (`scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh`) | $5 (cycle-1 only); full 10-cycle $25 | **DISPATCH CYCLE-1** — overnight task #5 ($5 cap, hard kill at 16h if no checkpoint) |
| 18 | RAFT/radial pose | 🟡 PARTIAL (`src/tac/raft_pose.py` exists, untracked, not integrated) | NO — needs integration with `inflate_renderer.py` | $0 | LOCAL-TASK candidate (defer to Phase 3) |
| 19 | SegNet logit-margin boundary | ❌ NOT STARTED | YES — scaffold module (`src/tac/scorer_margin_codec.py`) on local CPU; design-review only $0 | $0 / depends on upstream/contest_scorer SegNet on CUDA at compress time | **LOCAL-TASK #7** — scaffold module + tests (no GPU); writes paper-figure-quality margin distribution for Lane G v3 |
| 20 | Ballé hyperprior | ❌ NOT STARTED | NO — adapts Cool-Chic infra | $0 | DEFER (gated on Ω-W-V2 stack auth result per Ballé's Council F vote: "G1 first; this is the V3 amortisation gate-trigger") |
| 21 | Decoder systems rewrite | ❌ NOT STARTED | NO | $0 | DEFER |
| 22 | Final integration (Phase 4) | ❌ NOT STARTED | NO | n/a | Phase 4 work |
| 23 | Paper reproduction harness | ❌ NOT STARTED | NO | n/a | Phase 4 work |

**Phase 2-4 dispatch list tonight**:
- GPU: Lane 17 cycle-1 ($5, 18h)
- Local: Lane 10 ADMM 4-stream test ($0); Lane 19 logit-margin scaffolding ($0)

---

## 5. Overnight dispatch list (per Part 4)

**Budget tracking**:
- Total Vast.ai cap (Council E): $30
- Currently committed: $2.25 (HM-S $1.75 firing now; Ω-W-V2 stack $0.50 firing now via task #272)
- Remaining: $27.75
- Tonight's dispatch list: $8.00 (4 GPU + 1 GPU dependency-gated)
- Reserve: $19.75 (for FR-Ω dispatch if HM-S calibrates + Phase 2 NWCS-EC follow-up)

### Dispatch table

| # | Lane | Dispatch command (Pattern A nohup detach) | Cost | Wall | Predicted band | Kill criterion | Dependency |
|---|---|---|---|---|---|---|---|
| **D1** | Lane G v3 + Ω-W-V2 + PD-V2 + LCT triple-bolt-on archive auth | See command D1 below | $0.50 | 15min Vast.ai 4090 | 0.95 ± 0.02 [derivation] | score > 1.05 (regression) → debug bolt-on integration; do NOT continue D2 | None (chain-completion winner) |
| **D2** | Lane G v3 + Ω-W-V2 + multi-bolt + half-frame mask (HF) stack | See command D2 below | $0.50 | 15min Vast.ai 4090 | 0.92 ± 0.04 [prediction] | score > D1 score (regression) → HF was wrong stack | D1 lands first (uses same OWV2 binary) |
| **D3** | Lane FR-Ω contest-CUDA | See command D3 below | $1.50 | 6h Vast.ai 4090 | [0.27, 0.45] [prediction] | score > 0.80 → kill all Hessian-block-FP follow-up | HM-S calibration (gated; if HM-S lands within 0.10 of 0.385) |
| **D4** | Lane STC clean-source Modal T4 retry | See command D4 below | $0.50 | 30min Modal T4 (hard kill at 25min) | [0.30, 0.50] [prediction] | no archive emitted at 25min → kill STC investigation; rate term >= 0.30 → fall back to AV1 | None |
| **D5** | Lane J-IMP cycle-1 (Vast.ai 4090) | See command D5 below | $5.00 | 18h Vast.ai 4090 ($25 cap on full 10-cycle; cycle-1 = ~$5) | [0.85, 1.00] [prediction] | cycle-1 final renderer.bin > 250KB → IMP cycle is sparsity-ineffective on 88K params; kill | None |

**Total GPU dispatch tonight: $8.00. Reserve: $19.75.**

### Pattern A nohup detach commands (command preview only — DO NOT execute in this report)

**D1 — Triple-bolt-on archive auth** (highest EV, dispatched FIRST):

```bash
mkdir -p /tmp/codex_runs
nohup bash -c '
  cd /Users/adpena/Projects/pact && \
  .venv/bin/python tools/build_lane_g_v3_omega_w_v2_archive.py \
    --base-renderer experiments/results/lane_g_v3_landed/iter_0/renderer.bin \
    --base-poses experiments/results/lane_g_v3_landed/optimized_poses.pt \
    --apply-pd-v2 --apply-lct \
    --output /tmp/lane_g_v3_omegawv2_pdv2_lct.zip \
    && python scripts/launch_lane_on_vastai.py \
       --label lane_g_v3_triple_bolt \
       --archive /tmp/lane_g_v3_omegawv2_pdv2_lct.zip \
       --gpu RTX_4090 --cap-usd 0.50 --auth-eval-only
' < /dev/null > /tmp/codex_runs/D1_triple_bolt.outer.log 2>&1 &
disown
```
NOTE: `tools/build_lane_g_v3_omega_w_v2_archive.py` does NOT yet exist (per `ls` check). Local-task #1 in §6 below scaffolds it.

**D2 — Triple-bolt + HF mask stack**:
```bash
nohup bash -c '
  cd /Users/adpena/Projects/pact && \
  .venv/bin/python tools/build_lane_g_v3_omega_w_v2_archive.py \
    --base-renderer experiments/results/lane_g_v3_landed/iter_0/renderer.bin \
    --apply-pd-v2 --apply-lct --apply-hf \
    --output /tmp/lane_g_v3_quad_bolt.zip \
    && python scripts/launch_lane_on_vastai.py \
       --label lane_g_v3_quad_bolt --archive /tmp/lane_g_v3_quad_bolt.zip \
       --gpu RTX_4090 --cap-usd 0.50 --auth-eval-only
' < /dev/null > /tmp/codex_runs/D2_quad_bolt.outer.log 2>&1 &
disown
```
NOTE: `--apply-hf` requires HF mask trained renderer per CLAUDE.md `check_halfframe_archive_uses_trained_profile`. Lane G v3 was NOT trained with `mask_half_sim_prob>0` — D2 may FAIL the strict check. Council recommendation: defer D2 until a HF-trained Lane G v3 exists, OR replace D2 with an Ω-W-V2 + PD-V2 only archive (no HF), saving D2 budget for FR-Ω D3 if HM-S calibrates.

**D3 — Lane FR-Ω contest-CUDA** (gated):
```bash
nohup bash -c '
  cd /Users/adpena/Projects/pact && \
  python scripts/launch_lane_with_retry.py \
    --label lane_fr_omega \
    --script scripts/remote_lane_fr_omega_fridrich_block_fp.sh \
    --gpu RTX_4090 --cap-usd 1.50 \
    --gate-on-result lane_hm_s_segmap_homography \
    --gate-band 0.27 0.50 \
    --register-vastai-tracker
' < /dev/null > /tmp/codex_runs/D3_fr_omega.outer.log 2>&1 &
disown
```
NOTE: `--gate-on-result` flag may not exist in `launch_lane_with_retry.py`; verify before dispatch (CLAUDE.md non-negotiable: "preflight_arity — subprocess flag set must be a subset of target's argparse").

**D4 — Lane STC clean-source Modal T4 retry**:
```bash
nohup bash -c '
  cd /Users/adpena/Projects/pact && \
  .venv/bin/python experiments/modal_train_lane.py \
    --lane-script scripts/remote_lane_stc_clean_source.sh \
    --gpu T4 --max-seconds 1800 \
    --label lane_stc_clean_source_retry
' < /dev/null > /tmp/codex_runs/D4_stc_retry.outer.log 2>&1 &
disown
```
NOTE: Per `feedback_modal_spawn_result_cache_pattern_20260429.md`, the spawn() result cache TTL is ~24h. Schedule `tools/harvest_modal_calls.py` to run within 18h.

**D5 — Lane J-IMP cycle-1**:
```bash
nohup bash -c '
  cd /Users/adpena/Projects/pact && \
  python scripts/launch_lane_with_retry.py \
    --label lane_j_imp_cycle1 \
    --script scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh \
    --gpu RTX_4090 --cap-usd 5.00 \
    --extra-args "--cycles 1 --wall-clock-cap-h 16" \
    --register-vastai-tracker
' < /dev/null > /tmp/codex_runs/D5_imp_cycle1.outer.log 2>&1 &
disown
```

### Sequencing

1. **Immediate (within 1h)**: D1 (triple-bolt-on auth, $0.50, 15min). Hard-gate result before D2 dispatch.
2. **+1h** (after D1 lands): D2 (gated on D1 success; otherwise replace with PD-V2-only archive auth at $0.50).
3. **+2h** (after HM-S delivers calibration signal): D3 (FR-Ω, $1.50, 6h).
4. **Parallel to D3**: D4 (STC retry, $0.50, 30min) and D5 (IMP cycle-1, $5, 18h).
5. **Total wall**: D1+D2 finish in ~30min; D3 done by +8h; D4 done by +3h; D5 done by overnight (16h).

### Harvest plan

- **Modal spawn() calls (D4)**: must call `modal.functions.FunctionCall.from_id(call_id).get()` within 18h via `tools/harvest_modal_calls.py`. Set a follow-up task to harvest at +18h.
- **Vast.ai instance tarballs (D1-D3, D5)**: each launcher writes `experiments/results/lane_*_modal/...` artifacts; the result tarball auto-downloads on instance destroy. Verify presence before destroying instance.
- **Auth-eval JSON**: every D1-D5 lane writes `contest_auth_eval.json` to its results dir; `tools/recover_lane_artifacts.py` is the recovery path if instance destroys before JSON exfil.

---

## 6. Local parallel work (per Part 5)

7 local-only ($0) tasks that can run alongside the GPU dispatches:

| # | Task | Effort | EV | Justification |
|---|---|---|---|---|
| **L1** | Scaffold `tools/build_lane_g_v3_omega_w_v2_archive.py` (required by D1+D2) | 2-3h | UNBLOCKS D1 ($0.50 / -0.10 score [derivation]) | Per chain-integrity audit §G1, OWV2 inflate handler is landed at inflate_renderer.py:2145. The compress-side archive builder does NOT exist yet; this is the gating local task for D1. |
| **L2** | SegNet/PoseNet sensitivity-map generation (per user's earlier request) | 4-6h | Paper-figure-quality margin distribution for Lane G v3 archive frames | Generates the empirical signal that gates Phase 3 Lane 19 (SegNet logit-margin boundary). $0 CPU work; uses upstream/contest_scorer SegNet on CUDA at compress time (allowed per strict-scorer-rule). |
| **L3** | Lane PD-V2 + Lane LCT bolt-on integration into `tools/build_lane_g_v3_omega_w_v2_archive.py` | 1-2h | Required for D1 success (triple-bolt-on) | PD-V2 magic byte + dispatch already in `submission_archive.load_optimized_poses`; LCT 10-byte payload integration is a known-pattern. |
| **L4** | Joint-ADMM 4-stream non-convex test (gates V2 dispatch per Council F §3.1) | 3-4h | UNBLOCKS Lane 10 ADMM V2 dispatch | Council F: "the 2-stream convex KKT residual 0.02 is necessary but not sufficient; without a 4-stream non-convex test that exercises restart logic on a discrete-jump R(D), V2 will silently produce non-feasible points on the real archive." |
| **L5** | Round 11 bug fixes — Joint-ADMM Nesterov bias + rho_init scaling + j_jbl_dilated_h64 routing | 2-4h | Hardens Phase 2 ADMM lane | Per memory `feedback_round6_defects_lane_mm_correction_segmap_invalidation_extended_20260429`, Round 11 surfaced ADMM defects that were not yet fixed |
| **L6** | Lane 19 SegNet logit-margin codec scaffolding (`src/tac/scorer_margin_codec.py`) — NO GPU; design-only | 4-6h | Phase 3 lane begins (paper section: "Score-aware compression via gradient margins") | Council E reprioritized Lane 19 as ACCELERATE; scaffolding (no auth-eval, no GPU) is the start. |
| **L7** | NEW STRICT preflight checks for newly-discovered bug classes: (a) "OWV2 archive build requires explicit Ω-W-V2 magic byte test" (b) "Modal spawn() result must be harvested within 18h" (c) "subagent dispatch commands must verify --gate-* flags exist via grep" | 2-3h | Prevents recurrence | Per CLAUDE.md "Meta-bug class catalog — promotion path: A new check starts strict=False, the live violation count is fixed across the codebase, then it is flipped strict=True." Three new check candidates identified tonight. |

**Total local effort tonight**: 18-28h (parallelizable across multiple subagents/sessions). All $0.

---

## 7. Council Roll Call (10 inner voices, mathematical/empirical only)

**Shannon (LEAD, Information Theory)**: The chain-integrity audit's G1 dispatch (Lane G v3 + Ω-W-V2 stack) is the single highest-confidence next step (`[derivation]`-grade rate math; `[empirical]`-grade 40.98% codec savings). Triple-bolt-on (D1) compounds to ~0.95 [derivation]. **VERDICT: D1 is the right Phase 1; D5 (IMP cycle-1) is the right Phase 3 seed dispatch.**

**Dykstra (CO-LEAD, Convex Feasibility)**: 9 SegMapTrainer-invalidated lanes were under-handled. Council F stopped at HM-S + FR-Ω. SC++/SO are dominated by FR-Ω (same Hessian-block-FP mechanism). SA-v2 + q_faithful's SegMap variant are replacements, not stacks. **VERDICT: KILL SC++/SO/q_faithful-SegMap; DEFER SA-v2 until D1 gates the SegMap-arch decision.**

**Yousfi (Challenge creator, Steganalysis lineage)**: HM-S 8-DOF homography fires now ($1.75 in flight per parent context). FR-Ω D3 is the natural orthogonal partner if HM-S calibrates. **VERDICT: D1 + HM-S in parallel + D3 gated — correct portfolio.**

**Fridrich (UNIWARD/SRM/HUGO author)**: My Hessian-cost framework predicts FR-Ω works on weights as on stego pixels. Per Round 6 / Council F: FR-Ω band [0.27, 0.45] (Council F report) vs [0.25, 0.32] (script) — needs reconciliation BEFORE D3 dispatch. **VERDICT: D3 conditional on band reconciliation + HM-S calibration.**

**Contrarian (Veto)**: The user's instinct that "we don't believe Phase 1 spawned successfully" is empirically correct. Of 8 Phase 1 lanes, only 1 is GREEN-LANDED (Ω-W-V2), 1 is GREEN-FIRING (HM-S as Phase 1.5 add-on), 1 is DEAD (GP), 4 are stuck (STC NOT_READY, PSD never dispatched, multi-pass MVP-only, SC++ KILLED). **VERDICT: tonight's dispatch list MUST acknowledge this reality; do NOT claim "Phase 1 done" until D1 lands a [contest-CUDA] sub-1.0 frontier.**

**Quantizr (Adversarial leaderboard reality check)**: My 0.33 archive uses block-FP weights at 1.017 bpw (Ω-W-V2-class). The triple-bolt-on D1 dispatch is precisely my approach with explicit measurement. SC++ KILL is correct — KL-distill primary loss is the failure mode I avoided. **VERDICT: D1 first; D3 gated on HM-S; D5 IMP cycle-1 is a smart Phase 3 seed.**

**Hotz (Engineering shortcuts)**: Local-task L1 (compress-side OWV2 archive builder) is ~2h of dev for a $0.50 GPU spend that lands a measured -0.10 score reduction. EV/$ is the highest in the entire portfolio. **VERDICT: L1 is the highest-ROI work tonight; do it FIRST.**

**Selfcomp (szabolcs-cs, working 0.38 anchor)**: D1 stack is Selfcomp paradigm (block-FP weights compressed, masks unchanged, poses arithmetic-coded). My 0.38 came from this exact mechanism class. **VERDICT: D1 is right; the FR-Ω D3 dispatch is the canonical extension of D1's Hessian-aware allocation.**

**MacKay (Memorial seat, MDL + Bayesian)**: D1 is a strict MDL improvement (rate cost down 117KB; posterior approximation quality bounded per-channel). HM-S/FR-Ω are Bayesian model-comparison experiments (different posterior families). D5 IMP is a Bayesian sparsity prior (lottery-ticket hypothesis). **VERDICT: D1 is unconditional; D3 + D5 are conditional but principled.**

**Ballé (2018 entropy bottleneck SOTA)**: D1 validates Ω-W-V2 V2 ceiling at the score level. Once D1 lands, the V3 hyperprior (Lane 20) is properly motivated. Until D1 lands, V3 is premature. **VERDICT: D1 is the V3 amortisation gate-trigger; do it first.**

---

## 8. Cross-references

- Council F (5-lane retrain EV + Ω-W-V2 + ADMM consult): `.omx/research/council_f_retrain_ev_validation_admm_consult_20260429.md`
- Council E (Round 5 grand battleplan): `.omx/research/council_grand_battleplan_round5_20260429.md`
- Council Chain-Integrity Audit (G1 alternative the chain missed): `.omx/research/council_chain_integrity_audit_20260430.md`
- Council Round 6 (9-lane invalidation correction): `.omx/research/council_round6_adversarial_20260429.md`
- Lane G v3 1.05 [contest-CUDA] anchor: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`
- Ω-W-V2 real-archive 40.98% empirical: `src/tac/tests/test_omega_w_v2_real_archive.py`
- OWV2 inflate handler (already landed): `submissions/robust_current/inflate_renderer.py:2145`
- Phase 1 verdict source: `project_phase1_dispatch_verdict_20260429.md`
- Phase 2-4 lane scoping: `project_phases_2_3_4_design_implementation_math_provenance_20260429.md`
- Codec stacking + score arithmetic: `project_codec_stacking_composition_canonical_orders_20260429.md`
- Modal harvest pattern: `feedback_modal_spawn_result_cache_pattern_20260429.md`
- Vast.ai active instances tracker: `.omx/state/vastai_active_instances.json`
- Pattern A detach reference: `feedback_codex_detach_pattern_works_20260429.md`
- Skunkworks council quintet pact: `feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md`
- 6-month strategic plan: `project_6month_strategic_plan_20260429.md`

---

## 9. Verdict count

- **Phase 1 (8 lanes)**: 1 GREEN-FIRING + 2 LOCAL-TASK + 1 RE-DISPATCH (STC) + 1 KILL (GP, SC++/q_faithful) + 2 DEFER (PSD, multi-pass) + 1 KILLED (Lane 6 GP)
- **9-lane SegMapTrainer addendum (4 unaddressed)**: 2 KILL (SC++, SO) + 1 DEFER (SA-v2) + 1 KILL-VARIANT (q_faithful-SegMap)
- **Phase 2-4 (15 lanes)**: 1 DISPATCH-CYCLE-1 (Lane 17 IMP) + 1 DISPATCH-RETRY (Lane 9 STC clean-source) + 2 LOCAL-TASK (Lane 10 ADMM 4-stream test, Lane 19 logit-margin scaffold) + 11 DEFER

**Tonight's GPU dispatch count**: 5 (D1 + D2 + D3-gated + D4 + D5)
**Tonight's local-task count**: 7 (L1 through L7)
**Total budget**: $8.00 of $30 cap (planned); $19.75 reserve

**Ω-W-V2 stack (D1) is the chain's highest-confidence first dispatch — `[derivation]`-grade -0.10 score reduction at $0.50.**
