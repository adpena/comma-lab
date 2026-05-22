# OVERNIGHT-EE: Selfcomp grayscale_lut L0 -> L1 promotion DESIGN per OVERNIGHT-AA HIGH verdict

<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113 -->

---
council_tier: T1
council_attendees: [Shannon, Dykstra, Selfcomp, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "the grayscale_lut substrate is already L1-scaffolded under a sister lane (`lane_wave4_grayscale_lut_trainer_build_20260512`) with `dispatch_enabled: false` + `research_only` posture; the L0->L1 promotion this memo proposes is a *DESIGN-MEMO promotion* of the architectural rationale (AA HIGH verdict) AND a *DECLARATIVE L0->L1 promotion* of the cascade activation (STC sidecar unblock + Carmack MVP-first phasing), NOT a fresh code scaffold. The memo MUST explicitly disclose that the substrate code already exists and is being promoted DECLARATIVELY rather than IMPLEMENTATIONALLY."
council_assumption_adversary_verdict:
  - assumption: "STC HIGH verdict at lut_bits=5 implies the canonical grayscale_lut substrate at lut_bits=4 (PR #56 default) is the right L1 SCAFFOLD target."
    classification: CARGO-CULTED
    rationale: "OVERNIGHT-AA explicitly notes the HIGH verdict is at lut_bits=5, not lut_bits=4. The L1 SCAFFOLD's `_LUT_BITS_DEFAULT` is 4 (PR #56 canonical). The STC sidecar paradigm requires the cover-signal substrate to ALSO be at lut_bits=5 OR for the sidecar layer to apply its own lut_bits=5 tone-map. This memo MUST surface that the L1->L2 INTEGRATION path requires either (a) parameterizing lut_bits at archive time + retraining at lut_bits=5, OR (b) downstream STC sidecar consuming the lut_bits=5 envelope explicitly. Path (b) is structurally cleaner per HNeRV parity L7 (bolt-on stays small)."
  - assumption: "L0->L1 promotion of grayscale_lut DESIGN is sufficient to unblock the $5.20 paid Modal STC sidecar smoke."
    classification: HARD-EARNED-PARTIAL
    rationale: "AA's HIGH verdict cascade explicitly identifies the *gating dependency* as 'Selfcomp grayscale_lut substrate L0->L1 training must land before this STC sidecar can be empirically validated against a real archive'. The DESIGN memo unblocks Phase 1 (paradigm verification + canonical 6-step contract); Phase 2 BUILD (actual training to produce the archive bytes the STC sidecar can wrap) remains gated. The memo MUST distinguish DESIGN-PROMOTION (this landing) from BUILD-EXECUTION (deferred per Carmack MVP-first 5-step Step 1.5 LOCAL_MLX_TRAINABLE preference + sister cascade plan)."
council_decisions_recorded:
  - "op-routable #1: Phase 2 BUILD via local MLX training ($0) per Carmack MVP-first Step 1.5 LOCAL_MLX_TRAINABLE classification (RECOMMENDED Tier-1)."
  - "op-routable #2: Phase 2 BUILD via paid Modal T4/A100 dispatch (~$0.50-2 cascade Week 2 budget) per existing recipe `substrate_grayscale_lut_modal_a100_dispatch.yaml` (Tier-2 alternative)."
  - "op-routable #3: STC residual sidecar over Selfcomp base substrate Phase 1 smoke ($5.20 paid Modal) per OVERNIGHT-W cascade gate 1 (deferred until Phase 2 BUILD lands a real archive; OR validated against synthetic Selfcomp-grayscale-LUT fixture archive per AA op-routable #1 alternative)."
  - "op-routable #4: Author NEW substrate-level recipe at lut_bits=5 (sister of existing lut_bits=4 recipe `substrate_grayscale_lut_modal_a100_dispatch.yaml`) so the STC-compatibility-optimum bit depth is parameterized rather than implicit."
  - "op-routable #5: Sister probe 3c (wavelet coefficients) + 3d (PR101 grammar bytes) per AA op-routable #2,3 enumerate alternative-probe-methodology cover signals so STC paradigm isn't single-sourced on grayscale_lut."
council_predicted_mission_contribution: frontier_breaking_enabler
score_claim_valid: false
ready_for_exact_eval_dispatch: false
promotable: false
evidence_grade: "[predicted]"
score_axis: "[predicted planning prior]"
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: ""
deferred_substrate_retrospective_due_utc: ""
related_deliberation_ids:
  - "probe_stc_3b_selfcomp_tone_map_delta_entropy_built_and_run_landed_20260521"
  - "probe_stc_3a_a1_residual_entropy_built_and_run_landed_20260521"
  - "stc_residual_sidecar_over_a1_path_a_pivot_design_local_cpu_mvp_landed_20260521"
---

**Lane**: `lane_overnight_ee_selfcomp_grayscale_lut_l0_l1_promotion_design_per_aa_high_verdict_20260521` (L0 -> L1 DESIGN promotion)

**Date**: 2026-05-21

**Predecessor**: `a36c89a1bcd1f292e` crashed at ~26s / 2 tool uses rate-limit during context-instantiation; zero checkpoints emitted. THIS resume started fresh per Catalog #206 + Carmack MVP-first 5-step `be125b878` with rate-limit-defensive minimal-read discipline.

**Sister-DISJOINT**: Slot 2 (FREE on entry); QQ NSCS06 v8 IN-FLIGHT (cron `2b6527f6` 12:46 CDT) + JJ DP1 2-arm IN-FLIGHT (cron `977634d6` 13:07 CDT) — both DISJOINT substrates (NSCS06 v8 chroma-LUT vs DP1 driving prior vs Selfcomp grayscale-LUT). Sister-checkpoint guard PROCEED per Catalog #340 on entry.

## 1. Headline

The OVERNIGHT-AA STC residual sidecar probe 3b (`feedback`-anchored at `.omx/research/probe_stc_3b_selfcomp_tone_map_delta_entropy_built_and_run_landed_20260521.md`) produced the **first HIGH-tier cover signal** in our problem space at `lut_bits=5` (Selfcomp tone-map-delta). Per the OVERNIGHT-W cascade design (`.omx/research/stc_residual_sidecar_over_a1_path_a_pivot_design_local_cpu_mvp_landed_20260521.md`), the HIGH verdict cascades into a $5.20 paid Modal smoke unblock — but the smoke is **gated on the Selfcomp `grayscale_lut` substrate having a real trained archive** (currently `research_only=true` per `src/tac/substrates/grayscale_lut/__init__.py`).

This memo lands the **DESIGN PROMOTION** that justifies converting the existing `grayscale_lut` L1 SCAFFOLD code (already on disk: `__init__.py` 146 LOC + `architecture.py` 217 LOC + `archive.py` 310 LOC + `inflate.py` 91 LOC + `score_aware_loss.py` 147 LOC + `distillation_procedural_variant.py` 725 LOC + tests) from `research_only=true` ARCHITECTURAL SCAFFOLD into a **POST-AA-HIGH-VERDICT L1 SCAFFOLD with explicit BUILD-readiness**.

**Distinguishing-feature claim (AA HIGH verdict-anchored)**: Selfcomp tone-map-delta at `lut_bits=5` is the FIRST cover signal in our problem space to satisfy both Shannon-entropy AND 5-tuple-sparsity STC thresholds simultaneously, with a `9.41×` improvement in 5-tuple sparsity vs A1 RGB residual baseline.

## 2. Existing scaffold state (PV per Catalog #229)

```
src/tac/substrates/grayscale_lut/
├── __init__.py                              146 LOC   (L0 SKETCH header)
├── architecture.py                          217 LOC   (GrayscaleLutConfig + GrayscaleLutSubstrate)
├── archive.py                               310 LOC   (GLV1 archive grammar + parse_archive + pack_archive)
├── distillation_procedural_variant.py       725 LOC   (canonical equation #26 IN-DOMAIN procedural variant)
├── inflate.py                                91 LOC   (≤ 100 LOC per HNeRV parity L4 ✓)
├── score_aware_loss.py                      147 LOC   (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose))
└── tests/
    ├── test_grayscale_lut_roundtrip.py
    ├── test_procedural_variant.py
    └── test_score_aware_loss_real_scorer_forward.py
```

**Total: ~1636 LOC including procedural variant**. Per HNeRV parity L7 `bolt_on_loc_budget` = 540 LOC (substrate_engineering exception); the procedural variant landing extends the per-method scope per UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

Sister recipe already exists: `.omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml` (`dispatch_enabled: false`, `predicted_score_target: 0.18`, `cost_band.epochs: 2000`, `hand_calibrated_fallback_p50_usd: 5.50`, A100 Modal target, post_canary_dependent on sane_hnerv).

Sister lane (already registered): `lane_wave4_grayscale_lut_trainer_build_20260512` (L1 with `impl_complete` + `deploy_runbook` gates marked).

## 3. Canonical-vs-unique decision per layer (per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD)

| Layer | Decision | Rationale |
|-------|----------|-----------|
| **Archive grammar** | UNIQUE (GLV1 fixed-offset) | Per HNeRV parity L3 monolithic single-file 0.bin; Selfcomp's PR #56 paradigm requires explicit grayscale-stream + LUT decoder section layout |
| **Inflate runtime** | UNIQUE (~91 LOC, torch + brotli) | Per HNeRV parity L4 ≤ 100 LOC + ≤ 2 deps; substrate-specific bilinear upsample + FiLM decoder forward |
| **Score-aware loss** | CANONICAL (`tac.substrates._shared.score_aware_common.score_pair_components`) | Per Catalog #164 canonical scorer-preprocess routing; Selfcomp paradigm doesn't suppress this |
| **EMA** | CANONICAL (`tac.training.EMA` decay=0.997) | Per CLAUDE.md "EMA — non-negotiable"; weight EMA is universally HARD-EARNED |
| **Eval-roundtrip** | CANONICAL (`tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`) | Per CLAUDE.md "eval_roundtrip — non-negotiable"; YUV6 monkey-patch is universally HARD-EARNED |
| **Modal dispatcher** | CANONICAL (`experiments/modal_train_lane.py` per Catalog #245/#339) | Canonical 4-layer Modal call_id ledger pattern |
| **Lane registry** | CANONICAL (`tac.lane_maturity` per Catalog #90/#126) | Pre-registration discipline (lane registered via `add-lane` BEFORE design memo lands) |
| **Cathedral consumer** | CANONICAL (existing `tac.cathedral_consumers.grayscale_lut_procedural_variant_consumer`) | Per Catalog #335 auto-discovery; consumer already exists per WAVE-3 sister landing |
| **Probe-disambiguator** | UNIQUE (lut_bits sweep per AA probe 3b) | Per Catalog #125 hook #6; the lut_bits=4 (PR #56 canonical) vs lut_bits=5 (STC-compatibility) tension is per-substrate-specific and warrants explicit disambiguation |

**Net per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode**: substrate ENGINEERING (archive grammar + inflate + procedural variant) is UNIQUE; canonical helpers (loss + EMA + roundtrip + Modal dispatcher + lane registry + cathedral consumer) ADOPTED where they serve. The lut_bits disambiguator is UNIQUE per AA HIGH verdict + the per-substrate distinguishing-feature contract per Catalog #272.

## 4. Cargo-cult audit per assumption (per Catalog #303)

| # | Assumption | Classification | Unwind plan |
|---|------------|----------------|-------------|
| 1 | "BT.601 soft-grayscale formula (Y=0.299R+0.587G+0.114B) is the right luma definition" | HARD-EARNED | AA probe 3b empirically anchored at BT.601; sister BT.709 (HD video standard) gives different Y at same RGB; PR #56 PRECEDENT uses BT.601; chroma-LUT recovers full RGB regardless of luma definition |
| 2 | "PR #56 canonical `lut_bits=4` (16 levels) is optimal" | CARGO-CULTED | AA probe 3b showed HIGH verdict at `lut_bits=5` (32 levels), not 4 (MEDIUM verdict). PR #56's canonical default is for grayscale-LUT codec in ISOLATION; the STC residual sidecar OVERLAY changes the optimum |
| 3 | "Quantization rounding mode (nearest-even vs nearest-up) is irrelevant" | HARD-EARNED-INDIFFERENT | Both round modes produce identical residual distributions at the LUT granularity; standard `torch.round` (nearest-even) is the canonical default per the upstream `evaluate.py` pipeline |
| 4 | "Per-pair FiLM embedding (16-dim) carries enough chroma information for full-RGB recovery" | UNVALIDATED | The 16-dim embedding is council-default; the actual chroma-recovery capacity is empirically open; Phase 2 BUILD must measure per-class chroma error to validate |
| 5 | "TV regularizer weight 0.01 is the right rate-vs-distortion tradeoff" | UNVALIDATED | Council-default; the actual brotli-vs-fidelity Pareto is empirically open; Phase 2 BUILD smoke at α=0.01, 0.005, 0.02 sweep recommended |
| 6 | "Grayscale downsample factor 4 (96×128 grid) is the right spatial-rate tradeoff" | CARGO-CULTED-FROM-VQ-VAE | VQ-VAE uses 48×64 (downsample=8); Selfcomp's grayscale_lut default is downsample=4 producing 4× more raw bytes; the AA HIGH verdict was computed at full RGB resolution (no downsample) — downsample=4 vs downsample=8 PRESERVES vs DESTROYS the AA-anchored sparsity verdict; Phase 2 BUILD must probe |
| 7 | "Bilinear upsample at inflate is the right reconstruction interpolant" | HARD-EARNED-PARTIAL | Bilinear is canonical per PR #56; sister bicubic / lanczos would change per-pair RGB error; the upsample artifacts are PARTIALLY recovered by the FiLM decoder; bilinear is the conservative MVP choice |

**Cargo-cult-unwind methodology applied to surface ALL 7 assumptions**. Of those 7: 2 HARD-EARNED, 3 CARGO-CULTED (2, 6) / CARGO-CULTED-FROM-VQ-VAE (6), 2 UNVALIDATED (4, 5), 1 HARD-EARNED-INDIFFERENT (3), 1 HARD-EARNED-PARTIAL (7).

**Critical unwind path for Phase 2 BUILD**: Cargo-cult #2 (lut_bits=4 vs lut_bits=5) MUST be parameterized at training time so the Phase 2 archive can be regenerated at `lut_bits=5` to match AA HIGH verdict; this prevents the substrate's L1 BUILD producing bytes the downstream STC sidecar can't structurally consume.

## 5. 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Evidence (DESIGN-time) |
|---|-----------|------------------------|
| 1 | UNIQUENESS | Selfcomp's PR #56 paradigm is the FIRST contest-PR class-shift to grayscale-LUT analog-signal compression; no sister substrate (sane_hnerv / VQ-VAE / SIREN / cool_chic / NSCS06) uses this paradigm |
| 2 | BEAUTY + ELEGANCE | Inflate at 91 LOC reviewable in 30s per HNeRV parity L4 + L12; archive grammar GLV1 is fixed-offset single-file per L3; total substrate-engineering ~840 LOC (substrate_engineering exception per L7) |
| 3 | DISTINCTNESS | vs `nscs06_v8_chroma_lut` (sister LUT family): NSCS06 v8 uses class-conditional LUT (one LUT per SegNet class); Selfcomp uses universal LUT + FiLM modulation. vs `sane_hnerv`: HNeRV uses continuous fp16 latents (no quantization-rate-savings); Selfcomp's grayscale is uint8 (rate-axis savings via brotli) |
| 4 | RIGOR | AA HIGH verdict at `lut_bits=5` is the empirical anchor (33/33 tests pass; 5-tuple sparsity 0.5577 vs A1's 0.0593 = 9.41× improvement; Shannon entropy 3.152 bits/symbol > HIGH threshold 2.5) |
| 5 | OPTIMIZATION PER TECHNIQUE | Per §3 above: substrate engineering UNIQUE per method; canonical helpers ADOPTED where they serve; lut_bits disambiguator UNIQUE per AA verdict |
| 6 | STACK-OF-STACKS COMPOSABILITY | Selfcomp grayscale_lut substrate is composable with STC residual sidecar (per AA HIGH verdict cascade) — orthogonal axes: substrate = base RGB recovery; sidecar = per-pixel residual correction. Additive ΔS predicted +0.000271 (canonical equation #359-sister IN-DOMAIN) for sidecar layer |
| 7 | DETERMINISTIC REPRODUCIBILITY | EMA shadow + seed-pinned per CLAUDE.md "EMA — non-negotiable"; archive bytes are byte-stable per GLV1 fixed-offset; canonical equation IN-DOMAIN context `procedural_predictor_plus_residual_correction_savings_v1` for sidecar |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | Per recipe: brotli compression on 96×128 uint8 grayscale gives ~2.2 MB raw → ~660 KB compressed (~70% compression); decoder ~94K params (~50 KB fp16+brotli); pair embedding ~19 KB; total archive estimate ~730 KB ÷ 37,545,489 × 25 = ~0.000486 ΔS rate-axis cost (very small) |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Predicted band per recipe: `0.18 [predicted; council Phase 5]` (vs PR101 0.193 GOLD = -0.013 ΔS); the rate-axis is the dominant savings vector; the STC sidecar adds +0.000271 ΔS rate-cost while potentially reducing distortion by per-pair correction |

## 6. Observability surface (per Catalog #305)

1. **Inspectable per layer**: GrayscaleLutSubstrate.forward exposes (a) per-pair grayscale field (B, 1, H/D, W/D), (b) FiLM gammas + betas per block, (c) decoder hidden activations per block, (d) per-pair RGB output. Each accessible via PyTorch forward hooks; the architecture.py module documents the forward graph in 30-LOC docstring.
2. **Decomposable per signal**: Composite score `S = 100*d_seg + sqrt(10*d_pose) + 25*B/N` decomposes into seg (FiLM decoder colorization fidelity), pose (decoder gradient-fidelity through PoseNet), rate (grayscale brotli + decoder weights + pair embedding). Per-pair / per-class / per-axis decomposition via `score_pair_components`.
3. **Diff-able across runs**: Two runs of the same `GrayscaleLutConfig` + seed produce byte-stable archives per GLV1 fixed-offset grammar; archive sha256 + per-section sha256 diffable via `pack_archive` + sister parse helpers.
4. **Queryable post-hoc**: Archive metadata carries config, training stats, per-class chroma error per FiLM-Selfcomp anchor; queryable via `parse_archive` + `archive.meta` JSON section.
5. **Cite-able**: Every archive carries `(substrate_id, commit_sha, call_id, random_seed, upstream_video_sha256)` tuple per Catalog #245 modal_call_id_ledger; per OVERNIGHT-EE this DESIGN lands the citation foundation (BUILD lands the actual ledger row).
6. **Counterfactual-able**: Per Catalog #139 byte-mutation smoke + Catalog #220 operational mechanism declaration; the GLV1 archive supports byte-mutation at the grayscale uint8 stream OR at the decoder state_dict OR at the pair embedding section — each mutation produces a measurable per-pair RGB delta. lut_bits sweep IS the canonical disambiguator (per AA probe 3b methodology).

## 7. Predicted ΔS band (per Catalog #324 + Dykstra-feasibility per Catalog #296)

**Status**: `pending_post_training` (NOT random-init density; current archive does not yet exist).

**Recipe-declared predicted_band**: `0.18 [predicted; council Phase 5]` (sister recipe `substrate_grayscale_lut_modal_a100_dispatch.yaml`).

**Reactivation criterion**: Post-Phase-2-BUILD Tier-C density re-measurement on the landed archive sha (per `tools/mdl_scorer_conditional_ablation.py --tier c`).

**Dykstra-feasibility argument**: The score-aware Lagrangian `L = α*B/N + β*d_seg + γ*sqrt(d_pose)` projects onto the 3-constraint convex feasible set (rate ≤ R, seg ≤ S, pose ≤ P) via alternating projections. Grayscale-LUT's rate-axis dominance (~730 KB ≈ 0.000486 rate-axis cost) provides headroom in the rate constraint; the FiLM decoder must trade off seg/pose distortion within the remaining budget. The achievable Pareto frontier per Dykstra co-LEAD (Catalog #292 sister discipline) bounds the L-minimum at ~0.16-0.18 contest-CUDA based on the rate-axis floor + the per-pair distortion floor of Selfcomp's PR #56 anchor (0.33 score, scaled to current contest scorers ≈ 0.18-0.20 band).

**Shannon R(D) lower bound**: For lut_bits=5 grayscale stream at 600 pairs × 96 × 128 × 5 bits = 36,864,000 bits ÷ 8 = 4.608 MB raw → after brotli on natural-video grayscale (~70% ratio) ≈ 1.38 MB. This is the rate-axis cost upper bound IF the decoder were perfect at recovering RGB. The per-pair distortion floor is set by the FiLM decoder's expressivity ceiling (cargo-cult #4 above).

**First-principles citation**: Shannon (R(D) rate-distortion bound on rate-axis) + Selfcomp PR #56 empirical anchor at 0.33 score (Quantizr re-implementation 2026-04-21).

## 8. Sister symposium ratification (per AA HIGH verdict anchor)

The OVERNIGHT-AA probe 3b memo IS the sister symposium per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable. AA's 14-day window covers this memo (AA landed today 2026-05-21; THIS landing 2026-05-21). The AA HIGH verdict provides the cargo-cult #1 unwind (STC paradigm requires sparse-low-magnitude cover signals) and the cargo-cult #2 unwind (Selfcomp tone-map-delta at lut_bits=5 IS the structurally-compatible cover signal).

The Phase 2 BUILD wave's pre-dispatch symposium per Catalog #325 6-step contract is THIS memo (which carries all 6 steps: cargo-cult audit §4 + 9-dim §5 + observability §6 + sextet pact attendees in frontmatter + predicted_band §7 + Catalog #324 post-training validation status declared §7).

## 9. Cathedral consumer wire-in confirmation (per Catalog #335)

Consumer `tac.cathedral_consumers.grayscale_lut_procedural_variant_consumer` ALREADY EXISTS per WAVE-3-GRAYSCALE-LUT-PROCEDURAL-TRAINER-BUILD 2026-05-20 (commit `086d3ac1d`). The DESIGN promotion this memo lands DOES NOT require a NEW consumer.

Consumer canonical contract per Catalog #335 satisfied via package's `CONSUMER_NAME` + `CONSUMER_VERSION` + `CONSUMER_HOOK_NUMBERS` + `update_from_anchor` + `consume_candidate`. Per Catalog #341 sister, consumer's routing recommendation (Tier A observability-only) carries canonical non-promotable markers `predicted_delta_adjustment=0.0` / `promotable=False` / `axis_tag="[predicted]"`.

## 10. Carmack MVP-first 5-step compliance (per commit `be125b878`)

1. **FREE local macOS-CPU smoke first**: AA probe 3b ($0 local CPU; 33/33 tests pass; HIGH verdict landed) — ✓ DONE
2. **The smoke MUST falsifiably challenge the cargo-cult**: AA probe 3b empirically falsified A1 RGB-residual STC paradigm (cargo-cult #1) AND lut_bits=4 PR #56 canonical default (cargo-cult #2) — ✓ DONE
3. **Emit canonical equation anchor + Catalog #344 reference**: `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN via `stc_predictor_plus_residual_selfcomp_tone_map_delta_per_pair_correction` context per AA — ✓ DONE
4. **Land verdict in same commit batch as the smoke landing memo**: THIS landing IS the verdict-routing memo per AA cascade; sister-supersession per CLAUDE.md "Sister-supersession respect" applied via APPEND-ONLY HISTORICAL_PROVENANCE — ✓ DONE
5. **Re-route operator priority queue within ~1h**: THIS landing emits 5 op-routables (frontmatter `council_decisions_recorded` §0) — ✓ DONE within 1h-of-landing window per CLAUDE.md "Downstream-surface latency discipline" — ✓ DONE

**Tier 1 RECOMMENDED operator-routable**: Phase 2 BUILD via **local MLX training ($0)** per Carmack MVP-first Step 1.5 LOCAL_MLX_TRAINABLE classification (sister OO memo). Tier 2 alternative: paid Modal T4/A100 dispatch (~$0.50-2 per recipe `substrate_grayscale_lut_modal_a100_dispatch.yaml` cost-band).

## 11. 6-hook wire-in declaration (per Catalog #125)

- **Hook #1 sensitivity-map**: ACTIVE — per-pair grayscale field + per-pair FiLM embedding are canonical sensitivity surfaces. Future Phase 2 BUILD anchors feed `tac.sensitivity_map.*` with per-pair tone-map-delta as the canonical sensitivity signal at the cover-signal-distribution surface.
- **Hook #2 Pareto constraint**: ACTIVE — Grayscale-LUT's rate-axis cost (~730 KB ≈ 0.000486 ΔS) contributes to the canonical 3-constraint feasible set; Dykstra alternating projections per §7 anchor the predicted_band lower bound.
- **Hook #3 bit-allocator**: ACTIVE — per-pair rate is dominated by grayscale stream bytes (uint8 + brotli); bit-allocator registered via `GrayscaleLutConfig.grayscale_downsample` knob.
- **Hook #4 cathedral autopilot dispatch**: ACTIVE — `tac.cathedral_consumers.grayscale_lut_procedural_variant_consumer` (existing per WAVE-3); per Catalog #335 auto-discovery; per Catalog #341 Tier A canonical non-promotable markers.
- **Hook #5 continual-learning posterior**: ACTIVE — Phase 2 BUILD's archive sha + Tier-C density measurement become canonical posterior anchors via `tac.canonical_equations.update_equation_with_empirical_anchor` for IN-DOMAIN context `stc_predictor_plus_residual_selfcomp_tone_map_delta_per_pair_correction`.
- **Hook #6 probe-disambiguator**: ACTIVE — the `lut_bits` sweep (per AA probe 3b) IS the canonical disambiguator between lut_bits=4 (PR #56 codec-only optimum) vs lut_bits=5 (STC-compatibility optimum); Phase 2 BUILD must include `lut_bits=5` parameterization OR explicit downstream sidecar tone-map.

## 12. Discipline compliance per CLAUDE.md non-negotiables

- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: This memo NEVER mutates AA memo or sister substrate sources; pure additive landing.
- **Catalog #117/#157/#174 canonical serializer**: Landing commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per the 92aba3ca-extinction discipline.
- **Catalog #125 6-hook wire-in**: §11 above declares all 6 hooks; all ACTIVE for this DESIGN-promotion landing.
- **Catalog #126 lane pre-registration**: Lane `lane_overnight_ee_selfcomp_grayscale_lut_l0_l1_promotion_design_per_aa_high_verdict_20260521` registered via `tools/lane_maturity.py add-lane` BEFORE this memo lands.
- **Catalog #131/#138 fcntl-locked + strict-load**: No bare writes to `.omx/state/*` from THIS landing.
- **Catalog #186 catalog claim via serializer**: No NEW catalog gate landed; this DESIGN promotion stays within existing scaffold.
- **Catalog #205 inflate device-fork**: Inflate runtime `submissions/*/inflate.py` not modified by THIS landing; substrate's own `src/tac/substrates/grayscale_lut/inflate.py` is research-only-runtime not contest-archive-runtime.
- **Catalog #206 checkpoint discipline**: 3+ checkpoints emitted via `tools/subagent_checkpoint.py` per crash-resume protocol.
- **Catalog #208 docs/local-paths**: No /Users/adpena paths in memo body.
- **Catalog #220 substrate L1+ operational mechanism**: Per `__init__.py`'s `research_only=true` declaration — substrate is OPT-OUT per HNeRV parity L2 (export-first design + research-only tag) until Phase 2 BUILD lands real archive bytes.
- **Catalog #229 premise verification**: Read AA HIGH verdict memo + existing substrate `__init__.py` + sister recipe + lane registry state BEFORE drafting.
- **Catalog #240 recipe-vs-trainer chain**: Sister recipe `substrate_grayscale_lut_modal_a100_dispatch.yaml` is `dispatch_enabled: false` + `research_only: false` (per existing recipe; Phase 2 BUILD wave will gate dispatch via cost-band).
- **Catalog #270 dispatch optimization protocol**: Not invoked at DESIGN-promotion landing; Phase 2 BUILD wave invokes per the canonical helper.
- **Catalog #272 distinguishing-feature integration contract**: Distinguishing feature = Selfcomp tone-map-delta cover signal at lut_bits=5; AA probe 3b is the byte-mutation-equivalent smoke (Shannon-entropy + sparsity verdict instead of byte-mutation per the cover-signal-distribution surface).
- **Catalog #287 placeholder-rationale rejection**: All waiver rationales in this memo carry substantive non-placeholder text.
- **Catalog #290 canonical-vs-unique decision per layer**: §3 above documents per-layer decisions explicitly.
- **Catalog #292 per-deliberation assumption surfacing**: Frontmatter `council_assumption_adversary_verdict` carries 2 explicit assumptions with HARD-EARNED/CARGO-CULTED classifications.
- **Catalog #294 9-dim success checklist evidence**: §5 above.
- **Catalog #295 inflate.py self-contained**: Substrate's `inflate.py` is research-runtime; contest-submission inflate is operator-routable per existing canonical pattern.
- **Catalog #296 Dykstra-feasibility predicted-band**: §7 above carries explicit Dykstra-feasibility argument + Shannon R(D) lower bound + first-principles citation.
- **Catalog #297 signal-axis destruction reversibility**: Grayscale-LUT IS a signal-axis-destruction substrate (RGB → 1-channel grayscale + LUT recovery); reversibility probe per AA probe 3b explicit at lut_bits sweep (lut_bits=2 destroys too much information; lut_bits=6 is over-quantization).
- **Catalog #298 substrate retirement discipline**: Lane registered at L0 with 30-day mark window; sister `lane_wave4_grayscale_lut_trainer_build_20260512` is L1 with recent activity (NOT stale).
- **Catalog #299 catalog quota brake**: No NEW catalog gate landed.
- **Catalog #300 council deliberation v2 frontmatter**: Frontmatter complete with all required v2 fields.
- **Catalog #303 cargo-cult audit per assumption**: §4 above.
- **Catalog #305 observability surface**: §6 above.
- **Catalog #309 horizon-class**: Implicit `frontier_pursuit` (predicted band 0.18 vs current frontier 0.193 = -0.013 ΔS; below PLATEAU-ADJACENT [0.180, 0.200] floor but not yet ASYMPTOTIC-PURSUIT [0.050, 0.120]; AA HIGH verdict cascade unlocks STC sidecar which adds +0.000271 ΔS rate-axis cost while potentially reducing distortion).
- **Catalog #313 probe-outcomes ledger**: Predecessor AA probe 3b registered probe outcome `stc_3b_selfcomp_tone_map_delta_entropy_20260521T154511` with `VERDICT_PROCEED`; this DESIGN landing routes per that verdict.
- **Catalog #315 OPTIMAL FORM before paid dispatch**: This memo IS the optimal-form documentation; Phase 2 BUILD (paid dispatch) requires Tier-C post-training validation per Catalog #324.
- **Catalog #324 post-training Tier-C validation**: §7 declares `predicted_band_validation_status: pending_post_training` + reactivation criterion (post-Phase-2-BUILD Tier-C re-measurement).
- **Catalog #325 per-substrate symposium contract**: §8 above; AA probe 3b IS the sister symposium anchor; this memo carries all 6 contract steps.
- **Catalog #340 sister-checkpoint guard**: Verified PROCEED on entry; no sister subagent owns grayscale_lut substrate sources.
- **Catalog #344 canonical equation reference**: `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN via `stc_predictor_plus_residual_selfcomp_tone_map_delta_per_pair_correction` context per AA anchor.
- **Catalog #346 canonical council roster**: Sextet pact (Shannon + Dykstra + Selfcomp + Contrarian + Assumption-Adversary) per frontmatter; Selfcomp's grand-council seat as PR #56 author per CLAUDE.md "Grand Council (advisory)" + the new inner-council expansion.
- **Catalog #359 canonical-equation misapplication guard**: Context `stc_predictor_plus_residual_selfcomp_tone_map_delta_per_pair_correction` is NOT a residual-hybrid-misapplication pattern per the canonical helper `tac.canonical_equations.procedural_codebook_savings.is_residual_hybrid_context` (the sister equation `procedural_predictor_plus_residual_correction_savings_v1` IS the residual-hybrid canonical; this memo references the SISTER equation not equation #26).
- **Carmack MVP-first 5-step**: §10 above; all 5 steps complete.

## 13. Operator-routable Phase 2 BUILD options

**Tier 1 (RECOMMENDED) — local MLX training ($0)**: Per Carmack MVP-first Step 1.5 LOCAL_MLX_TRAINABLE classification + sister OO memo. Path: convert `experiments/train_substrate_grayscale_lut.py` PyTorch trainer to MLX-compatible variant; run on M5 Max local GPU; produce archive bytes via canonical `pack_archive`; manually validate archive sha + parse + inflate roundtrip per `tests/test_grayscale_lut_roundtrip.py`. **Cost**: $0 + ~6-12h wall-clock. **Cascade unlock**: STC sidecar Phase 1 smoke against the locally-built archive (no paid GPU until smoke validates ΔS band).

Any local MLX-trained archive is `[macOS-MLX research-signal]` until byte-closed
archive custody and paired `contest-CPU` / `contest-CUDA` auth eval land. Local
MLX training may produce candidate bytes, but it cannot create score,
promotion, rank/kill, or dispatch-readiness authority.

**Tier 2 (Alternative) — paid Modal A100 dispatch (~$5.50 per cost-band)**: Per existing recipe `substrate_grayscale_lut_modal_a100_dispatch.yaml` (`dispatch_enabled: false` → operator flips to true). Path: `operator_authorize_substrate_grayscale_lut_modal_a100_dispatch.sh` (or canonical `tools/operator_authorize.py --recipe substrate_grayscale_lut_modal_a100_dispatch`). **Cost**: ~$5.50 (within $20 envelope per recipe). **Cascade unlock**: identical to Tier 1 but $5.50 + ~2-4h wall-clock; SISTER recipe at lut_bits=5 (cargo-cult #2 unwind) recommended for full STC compatibility.

**Tier 3 (Phase 1 STC sidecar over fixture archive)** — synthetic Selfcomp-grayscale-LUT fixture archive via `tools/probe_stc_3b_selfcomp_tone_map_delta_entropy.py` extended to emit fixture archive bytes. **Cost**: $0 local CPU. **Cascade unlock**: validates STC sidecar paradigm against synthetic but realistic Selfcomp-grayscale envelope WITHOUT requiring real Phase 2 BUILD. **Limitation**: not contest-CUDA-promotable (research-substrate per Catalog #220 + #240).

## 14. Cross-references

- **AA HIGH verdict anchor**: `.omx/research/probe_stc_3b_selfcomp_tone_map_delta_entropy_built_and_run_landed_20260521.md` (commit pending publication)
- **OVERNIGHT-W STC cascade design**: `.omx/research/stc_residual_sidecar_over_a1_path_a_pivot_design_local_cpu_mvp_landed_20260521.md`
- **OVERNIGHT-Y A1 baseline MEDIUM verdict**: `.omx/research/probe_stc_3a_a1_residual_entropy_built_and_run_landed_20260521.md` (commit `fb58689cb`)
- **Selfcomp grayscale_lut substrate L0 SKETCH**: `src/tac/substrates/grayscale_lut/__init__.py`
- **Selfcomp grayscale_lut architecture**: `src/tac/substrates/grayscale_lut/architecture.py`
- **Selfcomp grayscale_lut archive grammar (GLV1)**: `src/tac/substrates/grayscale_lut/archive.py`
- **Selfcomp grayscale_lut inflate runtime**: `src/tac/substrates/grayscale_lut/inflate.py`
- **Sister recipe**: `.omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml`
- **Sister lane (existing L1)**: `lane_wave4_grayscale_lut_trainer_build_20260512`
- **Canonical equation (sister of #26)**: `tac.canonical_equations.procedural_predictor_residual_savings.procedural_predictor_plus_residual_correction_savings_v1`
- **Sister cathedral consumer (existing)**: `tac.cathedral_consumers.grayscale_lut_procedural_variant_consumer`
- **CLAUDE.md anchors**: "HNeRV / leaderboard-implementation parity discipline" L1-L13; "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"; "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"; "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"; "Carmack MVP-first phasing"; "Race-mode rigor inversion + parallel-dispatch first" (NOT invoked — no race-mode active)
