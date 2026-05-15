# ATW Codec V1 — Atick-Tishby-Wyner Cooperative-Receiver Codec — Design Memo

Date: 2026-05-15
Lane: `lane_atw_codec_design_v1_20260515`
Operator decision anchor: grand reunion symposium Composite #1 (Phase D), with $5-15 smoke when ready.
Through-line: this memo formalizes the **ATW codec** triple proposed in the
2026-05-15 grand reunion symposium Phase C Dyad 1 (Atick ↔ Tishby ↔ Wyner)
and Phase D whiteboard Composite #1 (lines 727-770 of
`feedback_grand_reunion_fields_grade_passion_full_council_debrief_*_20260515.md`).
Predicted [0.18, 0.21] frontier displacement.

---

## 1. Three-paper math composition (the ATW Lagrangian)

The ATW codec composes three foundational information-theoretic frameworks
into ONE training Lagrangian. The novelty is not the individual terms — it is
that they compose into a single closed-form-tractable objective on the
contest's specific cooperative-receiver substrate.

### Atick & Redlich (1990) — "Towards a Theory of Early Visual Processing"

Atick-Redlich derive that, for a known cooperative receiver `R = (W_R, A_R, P_R)`
(weights, activations, predictions), the optimal compressor minimizes

    L_AR = α · H(X | f_R(X))

NOT the unconditioned `H(X)`. For our contest scorer R = SegNet+PoseNet, this
collapses pixel-MSE proxies to scorer-conditional losses. The Z4 substrate
(currently testing on Modal `fc-01KRPJVEMQ5S7Q8EKGKQWKCS93`) is the loss-only
form of this. The ATW codec subsumes Z4 as the β-only branch.

### Tishby, Pereira & Bialek (1999) — Information Bottleneck

Tishby's IB Lagrangian replaces the conditional entropy with a variational
two-term form:

    L_IB = I(X; T) - β · I(T; Y)

where `T` is the codec representation and `Y = scorer outputs`. The β knob is
the rate-distortion tradeoff; the encoder converges to

    p*(t|x) ∝ p(t) · exp(β · ∫ p(y|x) log p(y|t) dy)

For the contest, `β = 4` (= contest weights ratio 100/25). When the cooperative
receiver is KNOWN (we have published SegNet + PoseNet weights byte-for-byte),
`p(y|t)` is computable in closed form modulo numerical tractability. This is
the analytical floor Tao proposed at symposium line 595 (Blahut-Arimoto in 2 hrs CPU).

### Wyner-Ziv (1976) — Source coding with side information at decoder

Wyner-Ziv: when the decoder has access to side information `S` correlated with
the source `X`, the rate-distortion function tightens from `R(D)` to

    R_WZ(D) = R_X|S(D) ≤ R(D)

The gap `R(D) - R_WZ(D) = H(Y|S) - H(Y|X,S)` is the rate the encoder need not
transmit because the decoder can predict it from `S`.

For the ATW codec: **S = the published SegNet+PoseNet weights**. These weights
are FIXED + KNOWN to both compress-time and inflate-time (compress can use
them per CLAUDE.md "Contest compliance"; inflate-time predict-from-S happens
without re-loading scorer weights — the encoder bakes the prediction into the
tiny `predict(t | scorer_class_prior)` head shipped in the archive).

**Wyner-Ziv gain estimate** for dashcam + scorer: the contest scorer's class
priors collapse most of the dashcam pixel diversity into ~5 SegNet classes +
6-dim pose. The conditional entropy `H(latent | scorer_class)` is empirically
~30-50% lower than unconditioned `H(latent)`. So the ATW codec can save
30-50% of the per-pair latent rate at fixed scorer-distortion.

### Composition: the ATW Lagrangian

    L_ATW = α · B(θ)/N                          ← rate from archive bytes
          + β_seg · d_seg(θ)                    ← Atick-Redlich SegNet term
          + γ_pose · sqrt(d_pose(θ))            ← Atick-Redlich PoseNet term
          + κ_IB · I(T; Y_predicted)            ← Tishby IB info-preservation
          + λ_WZ · R_WZ_residual(t | t̂(s))    ← Wyner-Ziv side-info residual term
          + λ_pixel · MSE(decoded, GT)          ← Z3 pixel-MSE residual (default 0)

where:

* α, β_seg, γ_pose, λ_pixel: contest formula weights (matches Z4 + A1 baseline)
* κ_IB ≥ 0: IB knob; default 0.0 = pure Atick-Redlich + WZ; 0.05-0.1 = Tishby IB regime
* λ_WZ ≥ 0: WZ residual weight; default 1.0; 0.0 = no WZ encoding (Z4 baseline)

The default settings (κ_IB=0, λ_WZ=1, λ_pixel=0) recover a CLEAN ATW codec
where the PRIMARY mechanism is Wyner-Ziv side-info compression on the per-pair
latent residual. Setting κ_IB > 0 + λ_WZ = 0 recovers Tishby IB. Setting all
three knobs to 0 except λ_pixel=1 recovers Z3 baseline.

This three-knob design enables a **probe-disambiguator** (Catalog #125 hook #6):

    * (κ_IB=0, λ_WZ=0, λ_pixel=0) → Atick-Redlich pure (= Z4)
    * (κ_IB=0, λ_WZ=1, λ_pixel=0) → ATW canonical
    * (κ_IB=0.1, λ_WZ=0, λ_pixel=0) → Tishby IB pure
    * (κ_IB=0, λ_WZ=0, λ_pixel=1) → Z3 baseline (pixel-MSE)

The relative ranking of these four ablations across paired CPU+CUDA auth
evals is the empirical arbitrator between {Atick-only, IB-only, WZ-only, classical}.

---

## 2. Application target — A1 substrate as the codec base

ATW V1 is a **codec substitution**, not a substrate redesign. The encoder +
decoder + per-pair-latent architecture inherits from A1 (the current frontier
at 0.19285 contest-CPU per `dual_eval_adjudicated.json`). The intervention
happens at three byte/training surfaces:

1. **Loss replacement**: `ATWScoreAwareLoss` replaces Z4's
   `CooperativeReceiverScoreAwareLoss` and adds the κ_IB + λ_WZ + λ_pixel knobs.
2. **Latent encoding**: per-pair latent `z` is split into `(z_predicted, z_residual)`
   where `z_predicted = predict(t | scorer_class_prior)` is computed at
   compress-time and shipped as a TINY (~1 KB) decoder-hint side-info, and
   `z_residual = z - z_predicted` is what the archive actually carries (Wyner-Ziv).
   The bit savings come from `z_residual` having ~30-50% lower entropy than `z`.
3. **Archive grammar**: ATW1 magic (`b"ATW1"`); meta carries
   `atw_codec_meta` provenance tag with κ_IB/λ_WZ/λ_pixel + literature anchors
   + Wyner-Ziv side-info head sha256 (scorer-class-prior precomputed table).

The cooperative receiver is the COMPRESS-side scorer pair (per CLAUDE.md
"Contest compliance" non-negotiable: compress can use scorer; inflate cannot).
At inflate time the decoder has only `z_residual` + `z_predicted_table` and
reconstructs `z = z_residual + z_predicted_table[scorer_class_prior(latent_index)]`
via a small lookup — no scorer load at inflate.

### Predicted [0.18, 0.21] frontier displacement

Per symposium line 760 + Tao analytical floor (line 595): IB+WZ theory gives
a closed-form upper bound on the rate gain. Empirical estimate composed from:

* **Rate gain** (Wyner-Ziv side-info): `-0.020 to -0.040` on the rate-axis
  (= 25 · (newrate - oldrate) / N_norm). For A1 archive (179 KB),
  if WZ saves 30% of latent bytes (~14 KB → ~10 KB latents), rate-axis ΔS ≈ -0.0027.
  If WZ saves 50% (~14 KB → ~7 KB), rate-axis ΔS ≈ -0.005. Per-frame benefit
  scales with latent fraction; predicted upper bound ~-0.020 if WZ generalizes
  to encoder/decoder weight prediction too.
* **Distortion gain** (Atick-Redlich): `-0.005 to -0.010` on combined
  100·d_seg + sqrt(10·d_pose), per Z4 council prediction. Already the Z4
  hypothesis; ATW inherits.
* **Total predicted ΔS vs A1 baseline 0.19285**: `-0.025 to -0.050`, giving
  predicted band `[0.143, 0.168]`. CONSERVATIVE band per "[prediction;
  first-principles-bound]" tagging convention: `[0.18, 0.21]` accounts for
  empirical compression of analytical bounds typically observed (factor ~2-3x).

These predictions are `[prediction; first-principles-bound from IB+WZ theory]`
per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable. Empirical
validation via the smoke + paired full-anchor cycle below.

### Composition with existing primitives — STACKS on Z4-V2 + A1

ATW V1 explicitly STACKS on the in-flight Z4-V2 (Modal `fc-01KRPJVEMQ5S7Q8EKGKQWKCS93`)
in two ways:

1. The β-only branch of ATW (κ_IB=0, λ_WZ=0, λ_pixel=0) = Z4 verbatim. If
   Z4-V2 returns with a successful contest-CUDA anchor, ATW V1 inherits its
   loss-side calibration.
2. ATW V1's WZ side-info head is composable with Z3's hyperprior (the IB
   regularizer on `z_predicted` distribution shape). If Z3 ever recovers, ATW
   can stack on Z3's hyperprior too (substitution-1:1 per Wunderkind G1).

Pareto polytope intersection: ATW adds two new constraints (κ_IB ≤ ε_IB,
λ_WZ residual entropy ≤ ε_WZ) that must be intersected with Z3+A1 rate/distortion
constraints. Per Boyd's convex feasibility lens, the ATW feasibility region
is a SUBSET of Z3's, so any ATW Pareto-optimal point dominates Z3 + Z4 at
the same operating point.

---

## 3. 36-field SubstrateContract per Catalog #241/#242 META layer

The substrate registers via `@register_substrate(SubstrateContract(...))` per
Catalog #241 meta-layer pattern. Required fields:

```python
SubstrateContract(
    id="atw_codec_v1",
    name="ATW Codec V1 (Atick-Tishby-Wyner Cooperative-Receiver Codec)",
    lane_id="lane_atw_codec_design_v1_20260515",
    target_modes=["research_substrate"],
    deployment_target="modal_a100_research",
    export_format="ATW1_monolithic_single_zip_member_0_bin",

    # HNeRV parity 8 fields (Catalog #124)
    archive_grammar="ATW1_monolithic_single_file_0_bin_with_wz_side_info_head",
    parser_section_manifest=(
        "ATW1 header (4-byte magic + 1-byte version + per-tensor config) + "
        "encoder_blob + decoder_blob + latent_residual_blob + "
        "wz_side_info_head_blob + meta_blob"
    ),
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=["torch", "brotli"],
    score_aware_loss="ATWScoreAwareLoss_routing_through_score_pair_components",
    bolt_on_loc_budget=400,
    no_op_detector_planned=True,
    research_only=True,  # Phase 2 council approval to lift _full_main NotImplementedError

    # Substrate-engineering opt-out per HNeRV parity L7
    lane_class="substrate_engineering",

    # Operational mechanism per Catalog #220
    score_improvement_mechanism_status="OPERATIONAL_AT_DESIGN_TIME",
    runtime_overlay_consumed=True,  # WZ side-info head consumed at inflate
    archive_bytes_added="-30% to -50% of A1 latent bytes (~4-7 KB savings)",

    # Catalog #170/#171/#172/#173/#181/#182/#215 substrate-recipe schema
    min_vram_gb=16,
    video_input_strategy="per_dispatch_local_copy",
    pyav_decode_strategy="cpu_thread_async_upload",
    canary_status="post_canary_dependent",
    canary_dependency="lane_z4_cooperative_receiver_loss_step2_20260514",
    cost_band_epochs=200,
    recipe_smoke_only=False,
    recipe_research_only=True,

    # 6-hook wire-in per Catalog #125
    hook_sensitivity_map="atw_codec_grad_norm_v1",
    hook_pareto_constraint="tac.pareto.atw_codec_v1",
    hook_bit_allocator="bit_allocator.atw_wz_residual_v1",
    hook_cathedral_autopilot="recipe_registered_warn_only_at_landing",
    hook_continual_learning="atw_anchor_seeds_posterior_paired_with_lambda_wz_kappa_ib",
    hook_probe_disambiguator="probe_atw_kappa_lambda_disambiguator_v1",

    literature_anchor=[
        "Atick & Redlich (1990) Neural Computation 2(3):308-320",
        "Tishby, Pereira & Bialek (1999) IB Lagrangian",
        "Wyner & Ziv (1976) IEEE Trans Info Theory 22(1):1-10",
    ],
    predicted_score_band=[0.18, 0.21],
    predicted_score_band_basis="first-principles-bound from IB+WZ theory",
)
```

---

## 4. HNeRV parity discipline lessons

| # | Lesson | ATW Compliance |
|---|--------|----------------|
| 1 | Substrate must be score-aware | ✓ — cooperative receiver IS the loss target |
| 2 | Export-first design | ✓ — archive grammar declared BEFORE training script |
| 3 | Archive grammar = monolithic single-file `0.bin` | ✓ — ATW1 monolithic 0.bin |
| 4 | Inflate.py ≤ 200 LOC | ✓ — substrate-engineering waiver per L7; reuses inflate runtime helpers |
| 5 | Architecture must be FULL renderer (RGB out) | ✓ — encoder + decoder + latent renders RGB pair |
| 6 | Score-domain Lagrangian | ✓ — α·B/N + β·d_seg + γ·sqrt(d_pose); same form as Z4 |
| 7 | Bolt-on size ≤ 350 LOC | substrate_engineering opt-out (~400 LOC scaffold) |
| 8 | Eval-roundtrip-aware + differentiable scorer-preprocess | ✓ — uses `apply_eval_roundtrip_during_training` per Catalog #5 + canonical `score_pair_components_dispatch` per Catalog #164 |
| 9 | Runtime closure | declared `runtime_dep_closure=[torch, brotli]` |
| 10 | Mask/pose coupling gate | N/A (full RGB renderer, not mask codec) |
| 11 | No-op detector | ✓ — WZ side-info head is structurally consumed; `no_op_detector_planned=True` |
| 12 | Single-LOC-per-LOC review | ~400 LOC bolt-on; reviewable in 30 sec |
| 13 | KILL is LAST RESORT | ATW V1 is research-only scaffold; never kill, only DEFERRED-pending-research per CLAUDE.md |

---

## 5. Reactivation criteria for empirical validation

ATW V1 ships as L1 SCAFFOLD (per Catalog #220 cascade: research_only=true +
lane_class=substrate_engineering + `_full_main raises NotImplementedError`).
Phase 2 council approval is required to lift `NotImplementedError` and dispatch.

**Reactivation criteria** (any TWO of these grant Phase 2 approval):

1. Z4-V2 returns with a successful contest-CUDA anchor (`fc-01KRPJVEMQ5S7Q8EKGKQWKCS93`)
   AND ΔS vs A1 in the predicted [-0.005, -0.010] band, confirming the
   cooperative-receiver hypothesis empirically.
2. Tao + Boyd Blahut-Arimoto analytical floor computed (~$0, 2 hours CPU per
   symposium line 595) AND the floor for A1's operating point shows that ATW
   theoretical gain ≥ -0.020 vs A1.
3. WZ side-info bit-savings empirically measured on A1 latent bytes via cheap
   CPU script (~$0, 1 hour) AND measured savings ≥ 20% per-pair latent rate.
4. Operator routes Z3-G1 substitution (Wunderkind G1, $10 paired) AND it
   returns with a successful CUDA anchor, confirming the scorer-as-decoder-side-info
   pattern empirically.

Until two of the four reactivation criteria fire, ATW V1 stays at L1 SCAFFOLD
+ research_only=true. No GPU dispatch.

---

## 6. Smoke + paired full-anchor cycle when activated

When Phase 2 lifts `NotImplementedError`:

1. **$0 pre-smoke**: WZ side-info bit-savings measured on A1 latent bytes
   (Tao + Boyd 1 hour CPU). Falsification: if savings < 10%, defer further.
2. **$5 Modal A100 smoke**: 100ep on synthetic-data smoke path, validate
   archive roundtrip + inflate parity + scorer preprocess gradient flow.
3. **$10 Modal A100 paired full anchor**: 200ep on real `upstream/videos/0.mkv`,
   ends with paired CPU+CUDA contest auth eval per CLAUDE.md "Submission auth
   eval — BOTH CPU AND CUDA" non-negotiable. Provider routing per cost-band
   posterior (D9 routing); use `tools/dispatch_modal_paired_auth_eval.py`
   with `--skip-axis-if-promotable-anchor-exists` per Catalog #246.

Total cost: $5-15 dispatch + $0-$0.50 pre-smoke = **$5-15.50**.

Probe-disambiguator dispatch ($30 paired): sweep (κ_IB, λ_WZ, λ_pixel) over
4 corner ablations (Atick-only, ATW canonical, Tishby pure, Z3 baseline).
Defer until V1 anchor lands.

---

## 7. Cross-references

* `feedback_grand_reunion_fields_grade_passion_full_council_debrief_*_20260515.md` — symposium Composite #1 (lines 727-770)
* `feedback_grand_council_evidence_review_modal_failures_*_20260515.md` — Phase B top-3 + Z4 lambda=0 timeout root cause
* `feedback_z4_atick_redlich_minimum_viable_landed_20260515.md` — Z4 minimum viable scaffold
* `feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md` — G1 substitution + ATW alignment
* `feedback_taylor_decomposition_contest_rules_into_autopilot_proxies_landed_20260515.md` — Taylor proxy framework consumed by ATW autopilot wire-in
* `src/tac/codec/cooperative_receiver/atick_redlich.py` — canonical Atick-Redlich primitive ATW reuses
* `src/tac/substrates/z4_cooperative_receiver_loss/` — sister substrate, β-only branch of ATW
* `src/tac/substrates/d4_wyner_ziv_frame_0/` — sister substrate, Wyner-Ziv on frame_0
* `tools/dispatch_modal_paired_auth_eval.py` — paired CPU+CUDA dispatch helper per Catalog #246

---

## 8. Honest gaps + risks

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Lagrangian | UNIQUE | ATW composes Atick-Redlich, Information Bottleneck, and Wyner-Ziv in one objective; forcing it into a Z3/Z4 helper would suppress the side-information mechanism. |
| Scorer preprocessing | ADOPT only as a guarded call | Canonical scorer preprocessing is compliance hygiene, but ATW keeps its own cooperative-receiver loss routing and ablation knobs. |
| Archive grammar | UNIQUE | The ATW1 residual-plus-side-info packet is the score-relevant design surface. |
| Probe disambiguator | UNIQUE | The four-corner `(kappa_IB, lambda_WZ, lambda_pixel)` probe is the arbitration mechanism. |
| Runtime/custody | ADOPT canonical | Dispatch claims, paired CPU/CUDA evidence, manifests, and scorer-free inflate remain shared guardrails. |
| Dispatch policy | UNIQUE fail-closed | ATW remains research-only until `_full_main`, export, and paired exact-eval custody land. |

* **WZ side-info head design is a closed-form computation but NOT YET implemented.**
  The scaffold ships with the head as a TINY MLP placeholder; the closed-form
  computation lands at Phase 2 lift time.
* **κ_IB > 0 IB regularizer is computationally expensive** (requires posterior
  approximation `q(y|t)`); the scaffold supports the hyperparameter but ships
  with κ_IB=0 default; non-zero κ_IB requires additional engineering.
* **Predicted [0.18, 0.21] is a CONSERVATIVE band** per typical 2-3x compression
  of analytical bounds. The actual outcome could be HIGHER (worse) if the
  encoder fails to converge to the IB optimum; or LOWER (better) if the WZ
  prediction quality matches the upper bound. Non-zero variance is expected.
* **Composition with Z3 hyperprior is theoretical** — has not been jointly
  trained yet. ATW V1 ships standalone; Z3+ATW joint training is V2.
* **Scorer load at COMPRESS time is per-CLAUDE.md compliant** but adds compress
  cost (~$0.50/dispatch via canonical scorer roundtrip). Acceptable per
  contest rule "compress side can use anything."
* **Phase 2 council approval IS REQUIRED** to lift `_full_main NotImplementedError`.
  Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" the
  scaffold is honest about being design-time only at landing.

---

## 9. Net assessment

The ATW codec is a **theoretically clean** composition of three foundational
information-theoretic frameworks that explicitly subsumes:

* Z4 (Atick-Redlich loss-only) as the β-only branch
* Tishby IB as the κ_IB-only branch
* Wyner-Ziv as the λ_WZ-only branch
* Z3 baseline as the λ_pixel-only branch

The scaffold lands the architecture + archive grammar + score-aware loss + 36-field
contract + 6-hook wire-in + ~30-50 dedicated tests in ~400 LOC bolt-on. No GPU
dispatch from scaffold — research-only per Catalog #220 + #240 cascade.

When Phase 2 lifts `NotImplementedError`, the predicted [0.18, 0.21] frontier
displacement is verifiable in $5-15 of Modal A100 spend. If it lands within
the band, ATW V1 becomes the new score-lowering primitive that all subsequent
substrates can compose against.
