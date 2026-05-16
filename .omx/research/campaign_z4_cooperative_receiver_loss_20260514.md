# Campaign ledger: Z4 cooperative-receiver loss (staircase Step 2)

`research_only=false` — L1 SCAFFOLD; Phase 2 council approval required to lift `_full_main` `NotImplementedError`. NO score claims. NO dispatch this turn.

Per CLAUDE.md "Long-burn score-lowering campaign default — NON-NEGOTIABLE":

## 1. lane_id + dispatch-claim plan

- **Lane**: `lane_z4_cooperative_receiver_loss_step2_20260514`
- **Pre-registered at L0** via `tools/lane_maturity.py add-lane`. Promoted to L1 after this landing (impl_complete + memory_entry + strict_preflight via Catalog #124).
- **Dispatch-claim plan**: per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION" non-negotiable, the smoke + full will each claim via `tools/claim_lane_dispatch.py claim --lane-id <lane> --status active_<provider>_smoke|full` before any GPU spend. Single-lane claim per dispatch; 24h TTL; terminal row appended on completion (success or fail). NO concurrent same-`lane_id` dispatches.

## 2. Source evidence + score-lowering hypothesis

- **Source evidence**:
  - Atick & Redlich (1990) "Towards a theory of early visual processing" — the original cooperative-receiver theorem: when the receiver is known to compute `f_R(X)`, the bit budget shrinks from `H(X)` to `H(X | f_R(X))`.
  - Time-Traveler peer-seat council (`feedback_grand_council_maximize_value_landed_20260514.md`): 10/11 STAIRCASE recommendation; Step 2 cooperative-receiver loss is the dominant marginal-value-per-byte intervention at the PR101+Z3 operating point.
  - Zen-floor field-medal council (`feedback_zen_floor_field_medal_grade_council_landed_20260514.md`): Step 2 predicted band [0.180, 0.188].
  - CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)": at PR106 frontier, pose marginal is 2.71× SegNet's; the cooperative-receiver loss is dominated by aligning gradient flow with the pose-axis distortion (not pixel-MSE).

- **Primary URLs / identifiers (retrieved 2026-05-16)**:
  - Atick-Redlich efficient coding:
    https://doi.org/10.1162/neco.1990.2.3.308
  - Tishby-Zaslavsky information-bottleneck framing:
    https://arxiv.org/abs/1503.02406 and https://arxiv.org/abs/1703.00810

- **Claim boundary**:
  objective analogy only: these sources are planning priors, not Pact score
  evidence.
  Atick-Redlich / information-bottleneck sources support the objective
  analogy only. They do not prove that the Pact scorer shrinks archive bytes
  or score. Z4 must still emit a byte-closed archive, keep scorer-free inflate,
  compare against a source-matched Z3/A1 baseline, record archive SHA and
  runtime tree/content SHA, recompute components, and run paired CPU/CUDA exact
  eval before any promotion wording.

- **Hypothesis**:
  Training the Z3+A1 substrate against `L = α·B/N + β_seg·d_seg + γ_pose·sqrt(d_pose) + λ_pixel·MSE_pixel` with `λ_pixel = 0` (pure cooperative-receiver) reaches **score band [0.180, 0.188]** on contest-CUDA T4 vs Z3's predicted [0.188, 0.193] baseline. Δ predicted: −0.005 to −0.010 vs Z3.

  The gain hypothesis is dominated by gradient alignment: pixel-MSE may waste
  bits on perceptually relevant but scorer-irrelevant texture, while scorer
  distortions are exactly what the contest measures. This remains a Pact
  planning hypothesis until the paired exact-eval packet above exists.

## 3. Timing-smoke command (~$0.50 Modal T4 1-epoch smoke)

```bash
# Local smoke (no GPU; ~30s on M5 Max):
.venv/bin/python experiments/train_substrate_z4_cooperative_receiver_loss.py \
    --output-dir experiments/results/z4_smoke_$(date -u +%Y%m%dT%H%M%SZ) \
    --epochs 3 --device cpu --smoke

# Modal T4 smoke ($0.50, ~10 min):
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.yaml \
    --smoke-only \
    --operator-approved 'adpena:<UTC>'
```

Expected wall-clock: ~10 min on Modal T4 (cost band $0.50 p50 fallback per CLAUDE.md "Modal scheduling can take HOURS"). Smoke validates substrate loads, scorer hooks fire, gradient flows through cooperative-receiver loss, archive packs + parses + inflates roundtrip, no NaN watchdog tripped.

## 4. Full-run command (~$5 Modal T4 200-epoch full) — REQUIRES PHASE 2 COUNCIL APPROVAL

The `_full_main` body is currently `NotImplementedError("Phase 2 council approval required")`. Operator-routable decision in §7. After council approval:

```bash
# Modal T4 full run ($5, ~17 min wall-clock once scheduled):
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_z4_cooperative_receiver_loss_modal_t4_dispatch \
    --operator-approved 'adpena:<UTC>' \
    --auto-fan-out

# Resume support: --resume-from-checkpoint experiments/results/z4_<run>/checkpoint_epoch_N.pt
```

## 5. Live provider rate + cost model

- **Modal T4**: $0.59/hr × (~10 min smoke + ~17 min full) ≈ **$0.50 smoke + $5 full = $5.50 total**.
- **Vast.ai 4090 (cheaper alt)**: $0.25/hr × similar wall-clock; total ~$2.50.
- **Lightning T4**: ~$0.40/hr; total ~$3.50.
- **Empirical anchor**: Z3 sister substrate's first smoke (lane_z3_balle_hyperprior_bolton_campaign_20260514) — if/when Z3 smoke lands, update this row with measured (smoke_wall_clock × hourly_rate).

Per CLAUDE.md "Long-burn score-lowering campaign default": "Budget uncertainty is not a reason to defer. If cost is unknown, run or prepare the smallest faithful timing smoke so cost becomes measured GPU-hours." The CPU smoke is the smallest faithful timing test; $0 GPU spend.

## 6. Byte-closed archive/export/inflate plan

- **Archive grammar**: Z4CR1 monolithic single-file `0.bin` (extends Z3HP1). 25-byte header + encoder blob (brotli fp16) + decoder blob (brotli fp16) + int8 latent blob + JSON meta with `cooperative_receiver_meta` provenance tag. Implemented at `src/tac/substrates/z4_cooperative_receiver_loss/archive.py`.
- **Inflate runtime**: `src/tac/substrates/z4_cooperative_receiver_loss/inflate.py` ≤ 200 LOC; no scorer imports; CPU/CUDA-agnostic via `select_inflate_device` (Catalog #205); 3-positional-arg contract per Catalog #146.
- **Export contract**: trainer `_full_main` (post-council) emits archive.zip via the canonical `_build_archive_zip` deterministic-ZIP helper (Catalog #19 sister pattern); writes `submission_dir/{inflate.sh, inflate.py, src/tac/substrates/z4_cooperative_receiver_loss/...}` per the canonical `_write_runtime` pattern.
- **Tests confirm roundtrip**: `test_archive_roundtrip_preserves_latents`, `test_inflate_one_video_writes_raw`, `test_archive_cooperative_receiver_meta_tag_present`. 31 dedicated tests passing.

## 7. Stop/continue thresholds

### Smoke gate (smoke-before-full per Catalog #167)
- **GREEN** (advance to full): smoke completes rc=0; archive bytes in [50_000, 200_000]; CPU smoke training loss converges (not NaN); inflate roundtrip succeeds.
- **YELLOW** (operator review): smoke completes but archive bytes outside band OR training loss > 10× expected at end of smoke epochs.
- **RED** (DEFERRED-pending-research): any of NaN, archive roundtrip failure, scorer-hook crash.

### Mid-stage gate (epoch 100 of full)
- **GREEN**: cooperative-receiver loss term decreasing monotonically; pose distortion approaching scorer-aware regime.
- **YELLOW**: cooperative-receiver loss flat after 100 epochs — operator-route decision (continue vs early-stop and retire as `measured-config-retired`).
- **RED**: cooperative-receiver loss diverging — STOP, save checkpoint, DEFERRED-pending-research per CLAUDE.md "KILL is the LAST RESORT".

### Export gate (after EMA shadow saved)
- **GREEN**: archive packs to deterministic bytes; CPU local inflate parity passes; archive bytes within [50_000, 200_000].
- **YELLOW**: archive bytes outside band — operator decides repack with smaller decoder OR retire.
- **RED**: archive determinism fails OR inflate-runtime LOC > 200 — STOP, audit grammar; KILL only with grand-council consensus per CLAUDE.md "KILL = LAST RESORT".

### Exact eval gate (CUDA auth eval on EMA shadow)
- **GREEN** (mark `contest_cuda` gate): contest-CUDA T4 score in [0.175, 0.195] (target [0.180, 0.188] + 0.007 tolerance for scorer numerical drift on T4).
- **YELLOW** (operator review): score in [0.195, 0.220] — operator-route as `measured-config-retired-suboptimal`; do not mark `contest_cuda` gate; document reactivation criteria.
- **RED** (DEFERRED-pending-research): score > 0.220 — DEFER per CLAUDE.md "KILL = LAST RESORT". Reactivation criteria: empirical anchor of Z3 baseline on same hardware; ablation showing pixel-MSE-only training yields the same score (refutes cooperative-receiver hypothesis); OR identify specific bug (NaN, scorer-roundtrip mismatch).

## 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map**: cooperative-receiver gradient norm IS the per-tensor importance signal; register `sensitivity_map.cooperative_receiver_grad_v1` post Phase 2 dispatch.
2. **Pareto constraint**: `cooperative_receiver_loss ≤ ε_scorer` intersected with Z3 rate/distortion polytope; register `tac.pareto.cooperative_receiver_v1` post-smoke.
3. **Bit-allocator hook**: `(β_seg + γ_pose)` IS the Lagrangian for per-tensor bit-allocation; register `bit_allocator.cooperative_receiver_v1` post-smoke.
4. **Cathedral autopilot dispatch hook**: recipe registered at `.omx/operator_authorize_recipes/substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.yaml`; gated by Catalog #167 smoke-before-full. Ranker v2 (Catalog #219) reads `literature_anchor=Atick-Redlich1990` and applies the canonical -0.01 to -0.02 class-shift reward via `apply_z1_empirical_revision_to_candidate_delta`.
5. **Continual-learning posterior**: every Z4 empirical anchor seeds the posterior via `posterior_update_locked` (Catalog #128). The paired `(L_pixel, L_scorer)` measurement is the canonical disambiguator data point.
6. **Probe-disambiguator**: `tools/probe_z4_cooperative_receiver_vs_marginal_disambiguator.py` (planned post Phase 2). Sweeps `λ_pixel` ∈ [0.0, 0.5, 1.0] at fixed `β_seg + γ_pose`. Returns: "Atick-Redlich wins" (λ=0 best), "marginal-gradient-alignment wins" (λ=0 and λ=1 tied), or "ambiguous" (λ=0.5 best — partial regime).

## Catalog #124 8 archive-grammar fields

All 8 declared inline in `src/tac/substrates/z4_cooperative_receiver_loss/__init__.py` so AST walker observes them at module-import time:

1. `archive_grammar`: monolithic single-file `0.bin` extends Z3HP1
2. `parser_section_manifest`: Z4CR1 header + 4 blobs + meta JSON
3. `inflate_runtime_loc_budget`: ≤200 LOC substrate-engineering waiver
4. `runtime_dep_closure`: torch + brotli + constriction
5. `export_format`: Z4CR1 monolithic single-zip-member `0.bin`
6. `score_aware_loss`: `CooperativeReceiverScoreAwareLoss` via `score_pair_components` (Catalog #164)
7. `bolt_on_loc_budget`: `lane_class=substrate_engineering` (HNeRV L7)
8. `no_op_detector_planned`: training-only change; bytes WILL differ from Z3 (different trained weights); empirical no-op detector verifies decoded RGB differs

## HNeRV parity discipline (CLAUDE.md non-negotiable 13 lessons)

1. ✅ Score-aware substrate via canonical `score_pair_components` (Catalog #164)
2. ✅ Export-first design (archive grammar declared before training)
3. ✅ Monolithic single-file `0.bin`
4. ✅ Inflate.py ≤ 200 LOC (substrate-engineering waiver) — actual: ~145 LOC
5. ✅ Full renderer architecture (RGB out, not mask-only)
6. ✅ Score-domain Lagrangian per Catalog #164 (NOT rel_err²)
7. ✅ `lane_class=substrate_engineering` per HNeRV L7
8. ⏳ Eval-roundtrip + differentiable YUV6 (`patch_upstream_yuv6_globally` before `load_differentiable_scorers`; Catalog #187) — enforced in `_full_main` once council lifts NotImplementedError
9. ⏳ Runtime closure (`inflate.sh` signature; dependency closure) — emit in `_full_main`'s `_write_runtime` helper
10. N/A — Mask/pose coupling gate (Z4 is full renderer, not mask codec)
11. ⏳ No-op detector — empirical; runs after first smoke completes
12. ✅ Single-LOC-per-LOC review discipline — `score_aware_loss.py` is ~210 LOC, every line reviewable in 30s
13. ✅ KILL = LAST RESORT (per CLAUDE.md non-negotiable; default verdict is DEFERRED)

## Cross-references

- **Step 1 sister**: `lane_z3_balle_hyperprior_bolton_campaign_20260514` (Balle hyperprior; byte basis for Z4)
- **Step 3 sister**: `lane_z5_predictive_coding_world_model_step3_20260514` (Z5 builds on Z4)
- **C5 mature cooperative-receiver lane**: `lane_wyner_ziv_cooperative_receiver_substrate_20260513` (DISCUS-class; structurally orthogonal to Z4)
- **Grand council (Time-Traveler peer seat)**: `feedback_grand_council_maximize_value_landed_20260514.md`
- **Zen-floor council**: `feedback_zen_floor_field_medal_grade_council_landed_20260514.md`
- **Long-burn campaign roadmap**: `feedback_long_term_multi_year_campaigns_landed_20260514.md` (C5)
- **Z1 ablation evidence**: `feedback_z1_mdl_ablation_landed_20260514.md`
- **Forbidden representation patterns** (CLAUDE.md): respected by export-first design + score-aware substrate

Lane: `lane_z4_cooperative_receiver_loss_step2_20260514`
Status: L1 SCAFFOLD; Phase 2 dispatch approval required to lift `_full_main` `NotImplementedError`.
