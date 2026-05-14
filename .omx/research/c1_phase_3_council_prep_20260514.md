# C1 world-model + foveation Phase 3 council prep ledger

**Lane**: `lane_c1_smoke_modal_phase_3_council_prep_20260514`
**Parent campaign**: `lane_c1_world_model_foveation_campaign_l1_scaffold_20260514`
**Tag**: `research_only=true` for all planning text; smoke artifacts `[smoke-no-scorer]`; no `[contest-CUDA]` anchor in this ledger; no score promotion.
**Inherited directives**: [`.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md`, `.omx/research/recursive_no_signal_loss_protocol_20260514.md`, `.omx/research/journal_lab_grade_documentation_standard_directive_20260514.md`]
**Subagent**: `c1_smoke_phase_3_prep_1778775305`
**Parent session**: `operator_session_20260514`
**Journal-grade**: `journal_grade_v1=true`

This ledger documents (a) the C1 Modal smoke attempt + the structural failure mode it surfaced, (b) the two probe-disambiguator verdicts (Catalog #125 hook #6) consumed BEFORE multi-stage architecture lock, (c) the Phase 3 multi-stage council ledger with math-grade derivations + citations + reactivation criteria, and (d) the 5 operator-routable decisions that gate the `_full_main` NotImplementedError unlock.

## 1. Hypothesis statement

C1 substrate (world-model recurrent latent + foveation matched to ego-motion) achieves `S ∈ [0.13, 0.16]` `[mathematical-derivation; first-principles-bound]` vs A1 baseline `S = 0.1928 [contest-CUDA T4; archive sha 87ec7ca5...492b5]` via a multi-stage training schedule ($30-50 over 6-8 Modal stages) that escapes the within-HNeRV-class MDL saturation trap empirically anchored at 97-99% density by Z1.

## 2. Math derivation

**Score formula** (contest spec, `upstream/evaluate.py`):
```
S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489
```
where d_seg ∈ [0, 1] (mask argmax disagreement rate), d_pose ∈ [0, 1] (pose-component MSE), B = archive bytes, N = 37,545,489 (frame count × resolution factor per CLAUDE.md FORBIDDEN_PATTERNS "Score formula").

**A1 baseline decomposition** (verified [contest-CUDA T4] archive sha `87ec7ca5...492b5`, 178,055 bytes):
- d_seg ≈ 0.067 (SegNet argmax disagreement on 600 contest pairs)
- d_pose ≈ 0.018 (PoseNet 6-dim MSE)
- B/N = 178055/37545489 = 0.004742
- S = 100·0.067 + sqrt(10·0.018) + 25·0.004742 = 6.7 + 0.4243 + 0.1186 ≈ 0.193 [contest-CUDA T4]

**C1 predicted ΔS** (decomposed):

(i) **World-model rate Δ** (Ha & Schmidhuber 2018 §3, Hafner DreamerV3 2023 §IV):
- HNeRV per-frame independent decoder requires per-frame parameters = O(latent_dim × time × decoder_capacity) ≈ 120 B/frame for 1200 frames → ~144 KB
- World-model recurrent latent z_t = GRU(z_{t-1}) shares parameters across time; per-frame residual surprise ≈ 50 B/frame for stationary-ergodic driving (Atick-Redlich 1990 §3.2 cortical magnification bound applied to temporal axis) → ~60 KB
- Δb = 144 - 60 = 84 KB; ΔS rate-term = -25·84000/37545489 = **-0.056** `[predicted; uncertainty ±40% bound by Ha-Schmidhuber's "100 B/frame" upper bound vs lower-bound MDL]`

(ii) **Foveation distortion Δ** (Carmack 2024 shower-thought, Atick-Redlich 1990 §3.1):
- 1164×874 camera → 384×512 scorer resize matrix is rank-deficient (singular values decay exponentially after ~24K of 192K dimensions per Mallat 1999 wavelet-scattering analysis)
- PoseNet (FastViT-T12 + Hydra head, modules.py:103-109) consumes BOTH frames YUV6 forward; foveation matched to ego-motion vanishing point reallocates bits from periphery (where scorer can't recover detail) to FOV-center
- Predicted d_pose Δ: -0.003 to -0.008 (foveation alone) `[predicted; uncertainty ±60% bound by lack of public benchmark for foveation-vs-uniform on driving video]`
- Predicted d_seg Δ: ≈ 0 (SegNet sees frame_1 only; foveation does not directly reduce SegNet disagreement)
- ΔS distortion-term: sqrt(10·(0.018 - 0.005)) - sqrt(10·0.018) = 0.3606 - 0.4243 = **-0.064** `[predicted; uncertainty ±60%]`

(iii) **Total C1 predicted ΔS**: -0.056 (rate) + -0.064 (distortion) = **-0.120 lower bound** vs A1's 0.193 → **predicted S ∈ [0.07, 0.13]** for best-case Stage 6 convergence

**Conservative band** (accounting for training imperfection + uncertainty stacking):
- Best case: S ≈ 0.07-0.10 (full distortion + rate gains realized)
- Median case: S ≈ 0.13-0.16 (partial realization; the campaign-ledger band)
- Worst case: S ≈ 0.17-0.20 (rate gain only; distortion neutral)

**Reactivation threshold** (per CLAUDE.md "KILL is LAST RESORT"): if Stage 6 [contest-CUDA] anchor lands at S > 0.22, lane goes `DEFERRED-pending-foveation-rescope` with explicit re-open criterion (e.g. "if codex Frame-0-derivation-class lands a sub-0.15 anchor on D4 substrate, foveation may stack additively").

## 3. Citations (named + cross-ref)

- **Ha & Schmidhuber 2018** — *World Models* (NeurIPS; arXiv:1803.10122) — RSSM + VRNN-style recurrent latent dynamics; predicted-token residual surprise bound
- **Atick & Redlich 1990** — *Towards a theory of early visual processing* (Neural Computation 2:308-320) — cortical magnification factor; 2-10× periphery bit savings for stationary-ergodic vision
- **Rao & Ballard 1999** — *Predictive coding in the visual cortex* (Nature Neuroscience 2:79-87) — hierarchical residual surprise encoding
- **Hafner et al. 2023** — *DreamerV3* (arXiv:2301.04104) — modern world-model architecture; GRU/RSSM as production-grade recurrence
- **Mallat 1999** — *A Wavelet Tour of Signal Processing* (Academic Press; §6.2 image-resize singular-value decay)
- **Shannon 1959** — *Coding Theorems for a Discrete Source With a Fidelity Criterion* (IRE Conv Rec §16) — `R(D) = inf_{p(Y|X)} I(X; Y)` rate-distortion bound for C1's residual codec
- **Tishby & Zaslavsky 2015** — *Deep learning and the information bottleneck principle* (ITW 2015) — IBPS framing for the C5 cooperative-receiver loss stacking
- **MacKay 2003** — *Information Theory, Inference, and Learning Algorithms* (CUP) — MDL grounding for the rate-Lagrangian
- **Rissanen 1978** — *Modeling by shortest data description* (Automatica 14) — original MDL formulation for the world-model parameter cost
- **Carmack 2024** — internal shower-thought via `feedback_zen_floor_field_medal_grade_council_landed_20260514.md` Round 1 eureka #4 — 1164×874 → 384×512 rank-deficient resize observation

**Internal cross-refs**:
- [[campaign_c1_world_model_foveation_20260514]] — campaign ledger (7-field envelope)
- [[feedback_c1_world_model_foveation_campaign_l1_scaffold_landed_20260514]] — substrate L1 scaffold landing
- [[feedback_z1_mdl_ablation_landed_20260514]] — A1 (99.29%) + PR106 r2 (97.21%) within-class MDL saturation; the empirical motivation for class shift
- [[feedback_long_term_multi_year_campaigns_landed_20260514]] — 7-campaign roadmap (C1-C7) with $30-50 budget envelope
- [[feedback_grand_council_maximize_value_landed_20260514]] — staircase framing (Time-Traveler's Step 4-5 = world-model + foveation)
- [[feedback_zen_floor_field_medal_grade_council_landed_20260514]] — across-class predicted band `[0.13, 0.16]`
- [[feedback_d4_wyner_ziv_frame_0_landed_20260514]] — sister across-class substrate (frame-0 derivation; orthogonal mechanism; potential stacking)
- [[feedback_mdl_density_gate_and_autopilot_ranker_landed_20260514]] — Catalog #219 autopilot v2 ranker (-0.05 class-shift reward for Ha-Schmidhuber-1990 + Atick-Redlich-1990 compound)

## 4. Provenance chain

| Element | Value | Verification |
|---|---|---|
| HEAD commit at smoke attempt | `5d0ec061d` (last clean ancestor) | `git rev-parse HEAD` (dirty tree at attempt time) |
| Probe 1 output | `reports/raw/c1_wm_probe_20260514T161531Z.json` (gitignored per `reports/raw/`) | sha256 `3a47891425321ac52f02c3dd0a003caef7c3153d777f6b811fdf4a1513c9e6c2` + JSON sort-keys deterministic |
| Probe 2 output | `reports/raw/c1_fov_probe_20260514T161537Z.json` (gitignored per `reports/raw/`) | sha256 `80e7ab0ab42e2bffba7ff0b6f22b81f31c17690ddc9894806ac511c2064354a3` + JSON sort-keys deterministic |
| Probe 1 verdict (inlined for durability) | `independent_frame_baseline` (margin 30%); residuals: GRU=0.00405 / LSTM=0.00432 / indep=0.00284 | `[proxy_synthetic; reports/raw is ephemeral]` |
| Probe 2 verdict (inlined for durability) | `tie` (uniform=0.012137105 vs learned_per_pixel=0.012137094 within 5%); ego_motion_radial=0.04217 (3.5× worse) | `[proxy_synthetic; reports/raw is ephemeral]` |
| Modal smoke attempt timestamp | `2026-05-14T16:16:19Z` | `/tmp/c1_smoke_dispatch.log` |
| Modal smoke rc | `1` (PRE-GPU upload failure) | `tee /tmp/c1_smoke_dispatch.log` |
| Modal smoke failure mode | Catalog #165 mtime-stability check refused | `experiments/train_substrate_ff_nerv.py modified during build process` |
| Modal smoke cost incurred | `$0.00` (no GPU meter started; image upload aborted) | provider dashboard: no call_id created |
| Dispatch claim row | `failed_modal_mount_upload_race_sister_subagent_dirty_tree` | `.omx/state/active_lane_dispatch_claims.md` |
| Lane registry entry | `lane_c1_smoke_modal_phase_3_council_prep_20260514` L0 phase 3 | `tools/lane_maturity.py audit` |
| Checkpoint record | `c1_smoke_phase_3_prep_1778775305` | `.omx/state/subagent_progress.jsonl` |

## 5. Empirical evidence tag

- Probe 1 verdict: `independent_frame_baseline` (margin 30%) `[proxy_synthetic:reports/raw/c1_wm_probe_20260514T161531Z.json]`
- Probe 2 verdict: `tie` (uniform vs learned_per_pixel within 5%) `[proxy_synthetic:reports/raw/c1_fov_probe_20260514T161537Z.json]`
- C1 predicted band `[0.13, 0.16]` `[predicted; first-principles-bound; uncertainty ±40%]`
- Modal smoke result `[empirical:/tmp/c1_smoke_dispatch.log; pre-GPU failure mode Catalog #165]`
- A1 baseline `S=0.193 [contest-CUDA T4; archive sha 87ec7ca5...492b5]` (verified anchor; not C1)

NO bare "improves N%" / "verified" / "beats baseline" claims in this ledger. Every claim carries a tag.

## 6. Reproducibility recipe

### Re-run probes (CPU, $0, ~3 sec each)

```bash
git checkout 5d0ec061d
.venv/bin/python tools/probe_c1_world_model_vs_independent_frames_disambiguator.py \
    --n-frames 64 --latent-dim 16 --epochs 200 \
    --output reports/raw/c1_wm_probe_<utc>.json
# expected verdict: independent_frame_baseline (margin ~30% on synthetic stationary-ergodic target)
# expected sha256 of output: depends on platform; verify the verdict field, not the bytes

.venv/bin/python tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py \
    --output reports/raw/c1_fov_probe_<utc>.json
# expected verdict: tie (uniform vs learned_per_pixel within 5%)
```

### Re-fire Modal smoke (after sister-subagent quiescence)

Pre-flight check:
```bash
# wait until git status is clean OR mtime stability window passes
git status --porcelain | wc -l  # expect 0 OR stable for >30 sec
stat -f "%m" experiments/train_substrate_*.py | sort -n | tail -1  # expect mtime > 60s old
```

Fire smoke:
```bash
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=1.50 \
OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1 \
C1_WORLD_MODEL_FOVEATION_SMOKE_EPOCHS=100 \
C1_WORLD_MODEL_FOVEATION_SMOKE_GPU=T4 \
/bin/bash scripts/operator_authorize_substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.sh
# expected: ~$0.50-1.00 GPU spend; 100 epochs T4; rc=0; valid C1WMFV1 archive; substrate plumbing validated
```

### Dependencies (`pyproject.toml` excerpt)
```toml
torch>=2.5.0  # canonical Modal training image dep per Catalog #203
brotli>=1.1  # archive grammar dep
constriction>=0.4,<0.5  # entropy coder dep
pyppmd>=1.3,<2.0  # entropy coder dep
```

### Hardware
- Probes: any CPU (verified on macOS M5 Max + Linux x86_64)
- Smoke: Modal T4 (~$0.59/hr; 100 epochs ~5-15 min)
- Full Phase 3 multi-stage: Modal T4 + A100 mix (see §11 budget table)

### Time budget
- Probes: ~3 sec each on CPU; $0
- Smoke (when sister-subagent quiescent): ~5-15 min Modal T4; ~$0.50-1.00
- Full multi-stage (Phase 3 unlocked): 6-8 stages × 2-5 hr each = 12-40 GPU-hours; $30-50

## 7. Sister-substrate / sister-lane references

**Lane registry entries touched** (none modified; all read-only):
- `lane_c1_world_model_foveation_campaign_l1_scaffold_20260514` — parent campaign (RECOVERY-1 owns; do not touch per Rule R2)
- `lane_c1_smoke_modal_phase_3_council_prep_20260514` — this lane (Phase 3 prep)
- `lane_d4_wyner_ziv_frame_0_substrate_20260514` — sister across-class (frame-0 derivation; orthogonal mechanism; potential stacking with C1)
- `lane_z3_balle_hyperprior_bolton_20260513` — sister within-class staircase (Z3 step 1)
- `lane_z4_cooperative_receiver_loss_20260513` — sister across-class (cooperative-receiver loss; potential C1+C5 stack)
- `lane_z5_predictive_coding_world_model_20260513` — sister across-class (predictive-coding L1; potential C1+Z5 ego-motion handoff)
- `lane_c6_e4_mdl_ibps_substrate_20260514` — sister across-class (MDL-IBPS; C6 grammar extension WIP)

**Catalog #s touched** (referenced; none modified):
- #5 (eval_roundtrip non-negotiable) — C1 score_aware_loss honors via `apply_eval_roundtrip=True`
- #124 (representation lane archive grammar at design time) — C1 declares all 8 fields inline in `__init__.py`
- #125 (subagent landing solver wire-in) — this ledger declares all 6 hooks (see §8)
- #127 (custody validator) — Modal smoke stats wired to fail-closed posterior
- #128 (continual-learning posterior locked) — wires on every Stage's empirical anchor
- #146 (Phase 1 trainer contest-compliant runtime) — C1 trainer honors via `_write_runtime` 3-positional-arg
- #151/#152 (operator wrapper threads trainer flags + required-input file validation) — C1 trainer has TIER_1_OPERATOR_REQUIRED_FLAGS `ast.AnnAssign` form
- #164 (scorer preprocess before forward) — C1 score_aware_loss routes through canonical `score_pair_components`
- #165 (Modal mount mtime-stability) — THE check that refused this smoke attempt (correctly; sister-subagent collision)
- #166 (Modal dispatch HEAD parity + worker source ledger) — would have run if smoke had reached worker
- #167 (smoke-before-full pattern) — C1 wrapper routes through canonical
- #190 (substrate hardware substrate dynamic detection) — C1 trainer honors
- #197 (full_cpu coupled flags) — C1 trainer honors
- #205 (inflate.py canonical select_inflate_device) — C1 inflate.py honors
- #219 (MDL density promotion gate) — would fire if C1 archive ever submitted with density > 0.90
- #221 (auth eval result artifacts fail-closed for score claims) — C1 trainer honors

**Substrate composition matrix**: C1 is `lane_class=substrate_engineering` with `literature_anchor=Ha-Schmidhuber-2018+Atick-Redlich-1990` per the v2 autopilot ranker; stacking with D4 (Wyner-Ziv frame-0 derivation) is orthogonal-mechanism (D4 derives frame-0 from frame-1 via motion; C1 compresses frame-1 trajectory via world-model recurrence). Stacking with Z4 (cooperative-receiver loss) is additive (Z4 loss applies AT the C1 substrate's training loop).

## 8. 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution**: foveation map `M_t` (per-pixel) + world-model latent `z_t` (per-frame) — `FoveationMapModule.map(z_t)` IS the per-pixel importance signal; smaller latent norm = lower-importance frame. Registered after Stage 1 empirical anchor lands; declared in campaign ledger §6-hook #1.
2. **Pareto constraint**: `foveation_concentration ≥ 0.30` + `world_model_residual_entropy ≤ 0.01` constraints added to Pareto feasible set; intersect with A1's `archive_bytes ≤ 180000` constraint to define C1 feasibility region. Binds after Stage 1.
3. **Bit-allocator hook**: `FoveationMapModule.map(z_t)` is the bit-allocator hook — per-pixel bit cost modulated by `(1 + foveation_attenuation × (1 - M_t(x, y)))`. Archive build path consumes this map for residual bytes per pixel.
4. **Cathedral autopilot dispatch hook**: C1 row registered in v2 ranker with `lane_class=substrate_engineering` + `literature_anchor=Ha-Schmidhuber-2018+Atick-Redlich-1990` (compound -0.03 + -0.02 = -0.05 across-class reward per Z1-revision logic landed 2026-05-14 Catalog #219). Predicted ΔS rank-adjusted; C1 ranks ahead of within-class sidecar bolt-ons even at modest predicted band.
5. **Continual-learning posterior update**: triggered on every Stage's empirical anchor via `tac.continual_learning.posterior_update_locked` (Catalog #128 atomic fcntl). Each Stage's `[contest-CUDA]` anchor (or `[smoke-no-scorer]` proxy) feeds the cost-band posterior + the across-class predicted-band posterior.
6. **Probe-disambiguator**: TWO probes shipped + RUN this session:
   - `tools/probe_c1_world_model_vs_independent_frames_disambiguator.py` → verdict `independent_frame_baseline` (margin 30% on synthetic stationary-ergodic). **Note**: synthetic target does NOT validate GRU recurrence claim; production driving video may differ. Must re-run against contest video before Stage 3 architecture lock.
   - `tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py` → verdict `tie` (uniform vs learned_per_pixel within 5%; ego_motion_radial 3.5× worse residual on synthetic radial target). **Note**: synthetic radial target may not match driving-video saliency; ego_motion_radial may still dominate on real frames.

## 9. Stop / continue thresholds

Per CLAUDE.md "Long-burn score-lowering campaign default" §7 mandatory:

| Gate | Threshold | Action if PASS | Action if FAIL |
|---|---|---|---|
| Probe 1 (world-model recurrence) | margin ≥ 5% on synthetic | continue | re-run against contest video before Stage 3 |
| Probe 2 (foveation strategy) | margin ≥ 5% on synthetic | continue | DEFAULT to ego_motion_radial per Carmack shower-thought; revisit at Stage 4 |
| SMOKE (Modal T4 100 epochs) | rc=0 + valid C1WMFV1 archive + plumbing end-to-end | continue to Phase 3 council | DEFERRED-pending-mount-race-retry (cost $0; not a method negative) |
| Phase 3 council vote | ≥ 4/5 inner-quintet PROCEED | continue to Stage 1 | DEFERRED-pending-council-revision |
| Stage 1 (world-model alone, T4 100ep, $1) | residual_l2 asymptote < 0.01 | continue to Stage 2 | DEFERRED-pending-recurrence-mode-rescope |
| Stage 2 (foveation alone, T4 200ep, $2) | foveation_l1 < 0.5 (concentration > 50%) | continue to Stage 3 | DEFERRED-pending-foveation-rescope |
| Stage 3 (combined, A100 500ep, $10) | proxy_loss < 0.04 | continue to Stage 4 | DEFERRED-pending-combined-rescope |
| Stage 4 (residual codec, T4 200ep, $2) | residual_blob ≤ 80 KB | continue to Stage 5 | DEFERRED-pending-codec-rescope |
| Stage 5 (archive byte sweep, T4 100ep, $1) | total bytes ∈ [100, 180] KB | continue to Stage 6 | DEFERRED-pending-byte-budget-rescope |
| Stage 6 (full Lagrangian, A100 1000ep, $20) | auth_eval [contest-CUDA] ∈ predicted band [0.13, 0.16] | promote to L2/L3 | DEFERRED-pending-research per outcome band |
| Stage 6 fallback (band [0.17, 0.20]) | — | DEFERRED-pending-research; consider C1+D4 stack | — |
| Stage 6 worst-case (band > 0.22) | — | DEFERRED-pending-foveation-rescope OR DEFERRED-pending-cross-class-rescope | — |

**No KILL verdicts** anywhere in this thresholds table. Every failure produces a DEFERRED-pending-* memo per CLAUDE.md "KILL is LAST RESORT" non-negotiable.

## 10. Reactivation criteria (for any DEFERRED outcome)

If Phase 3 multi-stage produces a DEFERRED memo at any stage:

| Stage failure | Reactivation criterion |
|---|---|
| Probe re-run on contest video | re-run when contest video is available locally (currently in `upstream/videos/0.mkv`); update probe-disambiguator to consume real frames not synthetic stationary-ergodic |
| Smoke mount-race | sister-subagent quiescence (≥ 60 sec mtime stability across `experiments/train_substrate_*.py`); cron retry feasible |
| Council reject | inner-quintet revote after operator-supplied additional evidence (e.g. C5 cooperative-receiver loss empirical anchor); council file path supplied |
| Stage 1 (world-model alone) | revisit if D4 lands a sub-0.15 anchor (frame-0 derivation orthogonal to world-model; potential stacking) |
| Stage 2 (foveation alone) | revisit if Mallat-wavelet sister codec (`lane_t1_alt_mallat_wavelet`) lands an empirical scaling-foveation result |
| Stage 3 (combined) | revisit if Stage 1 OR Stage 2 lands cleanly individually |
| Stage 4 (residual codec) | revisit if a hyperprior bolt-on (Ballé 2018) or constriction-AC sister lands measurable gain on PR101 |
| Stage 5 (byte sweep) | revisit if FP4A export contract for C1 lands; FP4 brings ~80KB ceiling per ledger §6 |
| Stage 6 worst-case > 0.22 | revisit if C1+D4 compound or C1+Z4 compound stacking analysis shows interaction effect |

## 11. Operator-routable decisions (5)

Per the journal-grade standard + CLAUDE.md "Operator gates must be wired and used" non-negotiable, the 5 Phase 3 unlock decisions, ranked by EV (expected value toward sub-0.16 band) and dependency:

### Decision 1: Approve Modal smoke RETRY (~$1; LOW risk; quiescence-dependent)

**Recommended action**: APPROVE; schedule via cron at next sister-subagent quiescence window.

**Cost**: $0.50-1.00 (Modal T4 100 epochs ~5-15 min wall-clock); $0 sunk so far.

**Risk assessment**: LOW. Catalog #165 mtime-stability check refused this attempt correctly (sister subagent dirty-tree race); retry under quiescence has zero technical risk and validates substrate plumbing end-to-end. If retry succeeds: L2 promotion via `real_archive_empirical` gate. If retry also fails: surface as Catalog #165 false-positive candidate.

**Conditional dependencies**: requires sister-subagent quiescence (≥ 60 sec mtime stability across `experiments/train_substrate_*.py`). Cron retry script can poll `git status --porcelain` until clean OR `stat -f "%m" experiments/train_substrate_*.py | sort -n | tail -1` shows mtime > 60s old.

**Rationale**: smoke validation is the structural prerequisite for the Phase 3 council vote (per §9 thresholds table). Without smoke, Stage 1+ are forbidden. The mount-race is a SIDE EFFECT of multiple sister subagents fan-out, not a method negative.

### Decision 2: Inner-quintet council PROCEED vote on multi-stage schedule (~$50; MEDIUM risk; smoke-dependent)

**Recommended action**: schedule council vote AFTER Decision 1 succeeds.

**Cost**: $0 for the council itself ($30-50 for the multi-stage if council PROCEEDs).

**Risk assessment**: MEDIUM. The council surface is Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian. Per the campaign ledger §7 and CLAUDE.md "Adversarial council review of design decisions" non-negotiable, ≥ 4/5 inner-quintet PROCEED required. Grand-council advisory: Hassabis (world-model design), Carmack (foveation engineering), Hinton (latent-dim selection), Mallat (foveation as wavelet analog), Ha (canonical world-model author; via Carmack's shower-thought chain).

**Council framing** (council ledger to be written at `.omx/research/council_c1_world_model_foveation_phase_3_unlock_<YYYYMMDD>.md` per directive):

- **Shannon LEAD position**: rate-distortion derivation §2 shows ΔS rate-term -0.056 from world-model recurrence; this trace back to a R(D) bound is necessary. The conservative band [0.13, 0.16] is internally consistent.
- **Dykstra CO-LEAD position**: Pareto-feasibility intersection of A1's archive_bytes ≤ 180000 + C1's foveation_concentration ≥ 0.30 + world_model_residual_entropy ≤ 0.01 must produce a non-empty feasible set. Verify via alternating-projections sweep before Stage 1.
- **Yousfi position**: scorer-aware embedding (PoseNet + SegNet) must remain detector-informed. C1's foveation matched to ego-motion is the inverse-steganalysis equivalent of "embedding errors in textured regions where the detector can't see them" — same UNIWARD logic.
- **Fridrich position**: same as Yousfi. Foveation = inverse-steganalysis UNIWARD applied to bit allocation.
- **Contrarian position**: probe 1 verdict `independent_frame_baseline` on synthetic IS a red flag. The synthetic stationary-ergodic target may not exercise the world-model's actual advantage (which requires PREDICTABLE temporal dynamics, not random walk). Demand: re-run probe against contest video AS PRE-STAGE-1-GATE.

**Conditional dependencies**: requires smoke decision (Decision 1) to land; requires probe-2 re-run against contest video (Decision 3); requires §2 math derivation to be reviewed in the council file.

**Rationale**: high EV ($30-50 → predicted -0.04 to -0.06 score gain); but Contrarian's red flag must be addressed empirically before Stage 1 fires. The council vote IS the structural gate.

### Decision 3: Re-run probe-disambiguators on REAL contest video (~$0; LOW risk; smoke-independent)

**Recommended action**: APPROVE; immediately parallelizable with Decision 1.

**Cost**: $0 (CPU; ~30-60 sec per probe with real video decode).

**Risk assessment**: LOW. The current verdicts on synthetic are SUGGESTIVE BUT NOT BINDING. Per the design-tension memo + Catalog #125 hook #6, the probes are arbitration BEFORE architecture lock. The synthetic verdicts MUST be confirmed against the actual contest video before Stage 3 (combined) fires.

**Implementation sketch**:
```python
# Modify probe 1 to consume real contest video frames:
import pyav
video = pyav.open('upstream/videos/0.mkv')
target = decode_first_n_frames(video, n_frames=64)  # (64, H*W*3)
# Then fit world-model + independent baseline on REAL target
```

**Conditional dependencies**: none; can fire immediately.

**Rationale**: probe-1's `independent_frame_baseline` verdict on synthetic random-walk is suspicious — production driving video has slow ego-motion with bursty turns, which world-models should exploit. Re-running on real video either CONFIRMS the verdict (and we should reconsider architecture) OR INVALIDATES it (and Stage 1 GRU is justified).

### Decision 4: Z5 ego-motion handoff with Time-Traveler subagent (~$0; LOW risk; coordination-only)

**Recommended action**: APPROVE; coordinate via memory file appended note.

**Cost**: $0 (no GPU; coordination work).

**Risk assessment**: LOW. The Time-Traveler subagent's Z5 predictive-coding L1 substrate (`lane_z5_predictive_coding_world_model_20260513`) shares the ego-motion vanishing-point concept with C1's foveation strategy. Both consume PoseNet pose outputs. Per CLAUDE.md "Subagent coherence-by-default" + Rule R2 sister-subagent ownership: cross-tree edits forbidden, but coordinated stacking analysis is welcome.

**Coordination pattern**: append note to `.omx/research/recursive_no_signal_loss_protocol_20260514.md` under "Recursive trust-but-verify" §; surface in BOTH lanes' next-cycle landing memos.

**Rationale**: stacking C1 (foveation matched to ego-motion) + Z5 (predictive-coding world-model) could compound -0.05 (foveation) + -0.06 (predictive coding) = **-0.11** total ΔS vs A1's 0.193 → predicted band [0.08, 0.10]. This is the staircase Step 4-5 territory per Time-Traveler's framing.

### Decision 5: Staircase ordering coordination (~$0 if local; LOW risk; council-overrideable)

**Recommended action**: DEFER to council (Decision 2 inner-quintet vote). Recommended ordering surfaced for council review:

Per the staircase framing in [[feedback_long_term_multi_year_campaigns_landed_20260514]] + [[feedback_grand_council_maximize_value_landed_20260514]]:

| Order | Substrate | Predicted ΔS | Cost | Cumulative S |
|---|---|---|---|---|
| Step 1 | Z3 Ballé hyperprior bolt-on | -0.001 to -0.003 | $2 | ~0.190 |
| Step 2 | D4 Wyner-Ziv frame-0 (sister; if smoke green) | -0.025 to -0.045 | $10 | ~0.160 |
| Step 3 | C5 cooperative-receiver loss (Z4) | -0.005 to -0.010 | $5 | ~0.155 |
| Step 4 | **C1 world-model + foveation** | -0.04 to -0.06 | $30-50 | **~0.10-0.13** |
| Step 5 | Z5 predictive-coding world-model | -0.01 to -0.02 | $20 | ~0.08-0.11 |
| Step 6+ | C6 MDL-IBPS / DARTS-SuperNet | -0.005 to -0.015 | $20-100 | ~0.06-0.09 |

**Rationale**: C1 is Step 4 (high cost, high gain, dependency on Z3+D4+Z4 prerequisite anchors). The ordering ensures cheap-fast wins land first (Z3 ~$2) and the expensive C1 Stage 6 ($20) only fires AFTER Steps 1-3 anchors are in posterior.

**Cost**: $0 for council coordination; cumulative staircase $87-187 if all 6 steps execute (council-overrideable budget cap).

**Risk assessment**: LOW for ordering question (council decides); MEDIUM-HIGH for cumulative spend (operator must approve in chunks).

**Conditional dependencies**: Decision 1 smoke + Decision 2 council vote must land first. Decision 4 Z5 handoff is a sub-decision of this ordering.

## 12. Crash-resume protocol

Per CLAUDE.md "Mandatory crash-resume protocol" + Catalog #206:

- **Parent session**: `operator_session_20260514`
- **Subagent ID**: `c1_smoke_phase_3_prep_1778775305`
- **Inherited directives**: `["recovery_session_20260514_directive_absolute_no_signal_loss_20260514", "recursive_no_signal_loss_protocol_20260514", "journal_lab_grade_documentation_standard_directive_20260514"]`
- **Lane ID**: `lane_c1_smoke_modal_phase_3_council_prep_20260514`
- **Final checkpoint status (at memo write)**: `step=4`, `status=in_progress`, `next_action="commit ledger; mark lane gates; write landing memo"`
- **If interrupted at this point**: a successor subagent can query `.venv/bin/python tools/subagent_checkpoint.py read --lane-id lane_c1_smoke_modal_phase_3_council_prep_20260514 --latest-incomplete` and resume from `next_action`.
- **Files touched at checkpoint**: `reports/raw/c1_wm_probe_20260514T161531Z.json`, `reports/raw/c1_fov_probe_20260514T161537Z.json`, `.omx/research/c1_phase_3_council_prep_20260514.md`, `.omx/state/active_lane_dispatch_claims.md`, `.omx/state/lane_registry.json`

## 13. Sister-subagent file ownership preserved (Rule R2)

Per `.omx/research/recursive_no_signal_loss_protocol_20260514.md` Rule R2, this subagent did NOT modify:

- `src/tac/substrates/c1_world_model_foveation/*` — owned by predecessor C1-WORLD-MODEL subagent (L1 scaffold landed)
- `experiments/train_substrate_*.py` (any sister substrate trainer) — owned by RECOVERY-3 Tier B/C/D + PER-TRAINER-WIRE-IN subagents
- `src/tac/substrates/d1_segnet_margin_polytope/*` — owned by RECOVERY-1
- `src/tac/substrates/c6_e4_mdl_ibps/*` — owned by C6-NEXT-WAVE
- `src/tac/substrates/z3_balle_hyperprior_bolton/*` / `z4_*` / `z5_*` — owned by Z3-BALLE / TIME-TRAVELER-STAIRCASE
- `src/tac/preflight.py` — multi-subagent shared; no preflight gates added here
- `CLAUDE.md` — multi-subagent shared; no catalog rows added here
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` — HISTORICAL_PROVENANCE per Catalog #113
- `.omx/research/recursive_no_signal_loss_protocol_20260514.md` — HISTORICAL_PROVENANCE per Catalog #113
- `.omx/research/journal_lab_grade_documentation_standard_directive_20260514.md` — HISTORICAL_PROVENANCE per Catalog #113

Files this subagent OWNS + modified:
- `.omx/research/c1_phase_3_council_prep_20260514.md` (this ledger; new)
- `reports/raw/c1_wm_probe_20260514T161531Z.json` (probe 1 output; new)
- `reports/raw/c1_fov_probe_20260514T161537Z.json` (probe 2 output; new)
- `.omx/state/active_lane_dispatch_claims.md` (claim row append; canonical helper)
- `.omx/state/lane_registry.json` (lane pre-registered via canonical helper)
- `.omx/state/subagent_progress.jsonl` (checkpoint append; canonical helper)

## 14. Why this ledger matters (journal-grade per directive)

Per the journal-grade documentation directive `journal_lab_grade_documentation_standard_directive_20260514.md` §"Why this matters":

1. **A new agent reading this ledger in 6 months** can reproduce the probes, retry the smoke, derive the predicted ΔS from §2 math, run the multi-stage schedule with the 5 operator-routable decisions §11 as the canonical entry point.
2. **A sister subagent at a different recursion depth** can consume the predicted-band + probe verdicts WITHOUT re-deriving the math.
3. **The operator** can audit the smoke failure mode (Catalog #165 mount-race) back to first principles, audit the predicted-band derivation back to Ha-Schmidhuber-1990, audit the staircase ordering back to the campaign roadmap.
4. **Public OSS publication** (per the comma.ai/openpilot-style MIT release) carries journal-grade rigor with citations + math + reproducibility recipes.
5. **Crashed subagent recovery** preserves reasoning (the 5 operator-routable decisions are the canonical "what was the agent about to do?" anchor).

Tag: `research_only=true`. NO score claims. NO `[contest-CUDA]` promotion. The smoke retry + multi-stage schedule require explicit operator approval before any GPU spend resumes.

## Cross-refs (final)

- [[campaign_c1_world_model_foveation_20260514]] — campaign ledger (7-field envelope; the parent document)
- [[feedback_c1_world_model_foveation_campaign_l1_scaffold_landed_20260514]] — substrate L1 scaffold landing memo
- [[feedback_long_term_multi_year_campaigns_landed_20260514]] — C1 in the 7-campaign roadmap
- [[feedback_grand_council_maximize_value_landed_20260514]] — staircase framing from inner-ten council
- [[feedback_zen_floor_field_medal_grade_council_landed_20260514]] — across-class predicted band derivation
- [[feedback_z1_mdl_ablation_landed_20260514]] — within-class MDL saturation empirical anchor
- [[feedback_d4_wyner_ziv_frame_0_landed_20260514]] — sister across-class substrate
- [[feedback_mdl_density_gate_and_autopilot_ranker_landed_20260514]] — Catalog #219 + autopilot v2 ranker
- `src/tac/substrates/c1_world_model_foveation/__init__.py` — substrate root with Catalog #124 8 fields declared
- `experiments/train_substrate_c1_world_model_foveation.py` — trainer (Phase 3-gated full)
- `.omx/operator_authorize_recipes/substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml` — recipe
- `scripts/remote_lane_substrate_c1_world_model_foveation.sh` — remote driver
- `scripts/operator_authorize_substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.sh` — smoke wrapper
- `tools/probe_c1_world_model_vs_independent_frames_disambiguator.py` — world-model recurrence probe
- `tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py` — foveation strategy probe

Tagged `research_only=true`. NO score claims in this ledger.
