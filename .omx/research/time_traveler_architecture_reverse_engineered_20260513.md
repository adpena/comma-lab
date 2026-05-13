---
title: Time-Traveler L5 Autonomy Architecture (Reverse-Engineered)
date: 2026-05-13
status: planning artifact — design synthesis
score_claim: false
evidence_axes: [time-traveler-prediction, mathematical-derivation, literature-prediction]
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: true
hnerv_parity_audit: design-time only (substrate not yet built)
---

# Time-Traveler L5 Autonomy Architecture — Reverse-Engineered for Contest

## Provenance

Operator directive 2026-05-13: "a time traveler from the future who solved self-driving full autonomous using a single comma ai hardware unit alone to share the secret of secret secrets with us".

The premise: if someone from the future solved L5 autonomy on comma.ai's single ~2W ARM+NPU device (Snapdragon 845, 4GB RAM, IMX390 camera, no cloud), what architecture would they have used? Reverse-engineering that architecture tells us what we're missing NOW.

Operator confirmation (2026-05-13, post-synthesis): "The time-traveler frame is the killer insight... reverse-engineering that architecture tells us what we're missing NOW. Their key insight is likely: cooperative-receiver architecture, foveation matched to ego-motion, differentiable world model, sub-100K params, predictive coding."

This memo formalizes that synthesis.

## The Architecture

```
TimeTravelerArchive (target: 95-110 KB total)
│
├── STAGE 1: WORLD MODEL (~55-70 KB, encoded ONCE)
│   ├── scene_geometry_prior          ~8 KB
│   ├── ego_motion_dynamics_prior     ~3 KB
│   ├── segmentation_class_palette    ~2 KB
│   ├── foveation_grid                ~2 KB
│   ├── predictive_decoder            ~35 KB
│   └── differentiable_physics_op     ~10 KB
│
├── STAGE 2: PER-PAIR SIDE INFO (~25-35 KB = ~45 bytes/pair × 600 pairs)
│   ├── pose_delta_SE3_lie_algebra    12 bytes/pair
│   ├── segnet_argmax_boundary_only   18 bytes/pair
│   ├── hf_residual_dsss              6 bytes/pair
│   └── prediction_error_residual     9 bytes/pair
│
├── STAGE 3: ARITHMETIC CODING STATE (~10 KB)
│   ├── conditional_model_params      ~5 KB
│   ├── section_offsets               ~3 KB
│   └── checksum_hashes               ~2 KB
│
└── STAGE 4: HEADER (~2 KB)
    └── magic + lengths + grammar version
```

**Predicted contest-CPU score: 0.150-0.170** `[time-traveler-prediction]`

## Why this beats PR101 0.193

| Component        | PR101 budget    | Time-traveler   | Why TT wins                         |
|------------------|----------------:|----------------:|-------------------------------------|
| Decoder weights  | ~110 KB         | ~35 KB          | Sub-100K params; Tikhonov regularization |
| Latents          | ~67 KB          | 0 KB            | Replaced by prediction-error only   |
| Per-pair side info | 0 KB          | ~30 KB          | Foveation + boundary + HF + residual |
| World-model prior | 0 KB           | ~25 KB          | Encodes physics+dynamics once       |
| **Total**        | **178 KB**      | **~100 KB**     | **44% smaller**                     |

## The Five First-Principles Design Moves

### 1. Cooperative-Receiver Theorem (Atick-Redlich 1990)

The scorer (SegNet + PoseNet) is known + fixed + public. The optimal encoder maximizes MI(archive_bytes; scorer_output) under rate constraint. The time-traveler didn't train a generic compressor; they trained a "video-against-this-specific-scorer" encoder.

Mathematical statement (Atick-Redlich efficient-coding theorem applied):
```
maximize MI(B; S(B)) subject to |B| ≤ R
```
where B = archive bytes, S = fixed scorer, R = rate budget.

This is structurally different from "maximize MI(B; V_GT)" which is the standard compression objective. The cooperative-receiver bound is much tighter because S(V_GT) has lower entropy than V_GT itself.

Cite: Atick & Redlich (1990), "Towards a theory of early visual processing", Neural Computation 2:308-320. https://doi.org/10.1162/neco.1990.2.3.308

### 2. Predictive Coding Hierarchy (Rao-Ballard 1999, Friston free-energy)

Cortex doesn't encode photons; it encodes prediction error against a learned generative model. The retina-V1-V4 hierarchy is INTERNAL FEEDBACK: each layer predicts the next, and only the PREDICTION ERROR propagates upward.

Apply to archive: world model is in Stage 1 (~60 KB, encoded once). Per-pair information is just the part the world model cannot predict (~45 bytes/pair).

Mathematical justification: prediction error has lower entropy than raw signal by the data-processing inequality H(R) ≤ H(V) where R = V - V_predicted and V_predicted is a deterministic function of past observations.

Cite: Rao & Ballard (1999), "Predictive coding in the visual cortex", Nature Neuroscience 2:79-87. https://doi.org/10.1038/4580
Cite: Friston (2010), "The free-energy principle: a unified brain theory?", Nature Reviews Neuroscience 11:127-138. https://doi.org/10.1038/nrn2787

### 3. Foveation Matched to Ego-Motion (Gibson 1950, Lee 1976, LAPose canvas)

Dashcam attention is centered on the focus of expansion (FOE). Pixels near FOE need high fidelity (vehicles, lane markers, brake lights). Pixels far from FOE can be encoded cheaply (sky, road shoulders, peripheral parked cars).

Log-polar foveation grid: 2 KB encodes the spatial weighting map. Provides 5-10× effective resolution gain on score-relevant regions vs uniform encoding.

Connect to LAPose foveation atom manifest already in repo (`tools/build_lapose_foveation_atom_manifest.py`).

Cite: Gibson (1950), "The Perception of the Visual World"; Lee (1976), "A theory of visual control of braking based on information about time-to-collision", Perception 5:437-459. https://doi.org/10.1068/p050437

### 4. Differentiable World Model (= encode physics, not pixels)

World state for dashcam:
- 6-DoF ego pose (12 bytes at fp16 with delta-coding)
- Road plane parameters (4-8 bytes per scene segment)
- Vehicle positions + IDs (variable, ~20 bytes per detection)
- Lane markers (parametric, ~10 bytes per lane)

Total: ~50-100 bytes per scene state. Render via differentiable physics + small MLP. Compare to encoding 384×512×3 RGB pixels per frame (~590 KB raw, ~30 KB with H.265).

Connect to Council F's O5 MDL Program-Plus-Patches: the world model IS the program; per-pair side info IS the patch.

### 5. Sub-100K Params Properly Trained

Information-theoretic argument: every decoder parameter costs ~4 bits at FP4 quantization.
- PR101: 229K params = 114 KB of decoder cost
- Quantizr: 88K params → 0.33 contest-CPU (close to PR101 architecture-wise; loses ~0.14 from no curriculum + no score-aware)
- Time-traveler: 60K params + world model + foveation = 35 KB decoder + 25 KB world model = 60 KB representation cost vs PR101's 114 KB

Trade params for STRUCTURE. The Tikhonov regularization principle says under-parameterized models with good prior beat over-parameterized models without prior, given the same training budget.

Mathematical: model description length L = K(decoder) + K(latents | decoder). Minimizing L under fixed distortion = optimal description-length-aware training. Subject to: K(decoder) decreases monotonically with #params; K(latents | decoder) is minimized when decoder is well-trained on world structure.

## Concrete Connection to Our Session's Existing Primitives

| Time-Traveler stage | Existing session work | Lane status |
|---|---|---|
| Score-aware encoder training | PR95 8-stage curriculum (F1 ported); IGLT (in flight) | F1 Phase 1+2 LANDED |
| Predictive coding | Council F O5 MPPA + Wyner-Ziv (SE-1, ancient elder) | DESIGN |
| Foveation grid | A1+LAPose substrate (L1) + φ1 SABOR audit (CONFIRMED) | L1 smoke-ready |
| Differentiable physics | Alien-tech NCA + Constructor-DAG + MERA | DESIGN |
| Sub-100K params | PR101 LoRA/DoRA r=8 → 17K trainable params | L1 build |
| Pose-delta AC | PR93 pose codec primitive | LANDED |
| HF byte-stuffing | φ3 S2SBS substrate (CONFIRMED 97KB/frame capacity) | AUDIT LANDED, substrate DESIGN |
| Boundary-only segnet residual | φ1 SABOR substrate (CONFIRMED 99.27% stability) | AUDIT LANDED, substrate DESIGN |
| SE-4 JSCC scorer-conditional coding | Ancient elder SE-4 (in flight from IMPL-A) | IMPL in flight |
| CTW arithmetic | Codex CTW + ancient elder SE-2 | DESIGN |

**This means we're CLOSER than the literature suggests** — the time-traveler architecture is buildable from primitives we already have. They just need to be COMPOSED into one coherent archive grammar.

## The One Piece Missing

**Differentiable-physics renderer + ego-motion prior** as a substrate class. The closest existing work is LAPose motion atoms + scene-geometry priors, but these need to be unified into ONE substrate with archive grammar.

Build effort estimate: 2-3 days
- Day 1: Differentiable physics renderer (small MLP + positional encoding + Lie algebra pose transform)
- Day 2: Ego-motion dynamics prior (typical highway dynamics; Markov model on pose deltas)
- Day 3: Integration with predictive-coding loop + archive grammar

Dispatch cost: $3-8 Modal A100 smoke + full
Predicted Δscore if combined with existing primitives: -0.020 to -0.040 (toward sub-0.180)

## Falsification Criteria

This architecture is falsified if:
- Empirical contest-CPU > 0.190 after building + training + auth-eval
- Decoder + world-model bytes can't compress below 60 KB while preserving scorer-equivalence
- Differentiable-physics module increases inflate.py LOC budget above 200 (HNeRV parity lesson 4)
- Per-pair side info exceeds 50 bytes/pair (rate budget breach)

## Reactivation criteria

If falsified at first empirical anchor:
- Defer differentiable-physics; try sub-100K-param + foveation + score-aware ONLY (cheaper version)
- If THAT still doesn't beat 0.190: HNeRV-family is the ceiling per Council G

## Cross-references

- Time-traveler-frame originator: this session's synthesis (Claude main thread 2026-05-13)
- Council F first-principles original score-lowering memo (commit `896f1d79`)
- Council G HNeRV meat-on-bone deep-dive memo (commit `896f1d79`)
- φ1 SABOR audit memo (commits `c075b84c` + `1a726794`)
- φ3 S2SBS audit memo (commits `b69d6750` + `a647008f`)
- F1 PR95 forensic memo (commit `ce8fdcc7`)
- F1 substrate engineering (commit `3074f7f6`)
- Zen-state frontier deep-math memo (commit `3d867a0d`)
- Alien-technology unknown-unknowns memo (commit `a1ac138f`)
- Ancient-elder polymath memo (commit `39686bcf`)
- Codex frontier-innovation roadmap (`.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md`)

## 6-hook wire-in (per Catalog #125)

1. Sensitivity-map: the per-pair side info budget (45 bytes/pair) should be allocated by score-component-sensitivity (Catalog #123). Hook: register cost-band sensitivity ranking.
2. Pareto constraint: world-model + per-pair info ≤ 100 KB total. Hook: register Pareto constraint in `tac.pareto_*`.
3. Bit-allocator: Stage 1 vs Stage 2 budget split is Fisher-water-fillable. Hook: register allocator.
4. Cathedral autopilot dispatch: time-traveler substrate L0 SKETCH → register as candidate. Hook: lane registry add-lane after L0 ready.
5. Continual-learning posterior: empirical first-anchor result updates prior on time-traveler-prediction. Hook: append to cost-band posterior.
6. Probe-disambiguator: 2 defensible interpretations (full differentiable-physics vs. cheap-foveation-only). Hook: probe tool to disambiguate via cheap macOS-CPU smoke first.

## Verdict

DEFERRED-pending-empirical. NO score claim. NO promotion. NO archive bytes built yet. This is a design synthesis that points the next dispatch direction.

Recommended next operator decision: authorize a $0-3 build of the time-traveler architecture as a NEW substrate (`lane_time_traveler_l5_autonomy_substrate_20260513`), then smoke at macOS-CPU ($0), then dispatch at Modal A100 ($3-8) if smoke passes predicted band [0.150, 0.180].
