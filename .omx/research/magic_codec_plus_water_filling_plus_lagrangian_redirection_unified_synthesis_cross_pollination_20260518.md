# Magic Codec + Water-Filling + Lagrangian Redirection — unified synthesis + cross-pollination across ablation / quantization / adversarial training / codec
# Date: 2026-05-18
# Authority: operator query 2026-05-18 verbatim *"remember the magic codec and water bucket filling and Lagrangian redirection work and maybe consider the principles and concepts and problems and solutions and manifolds involved and how they might apply to the ablation work and quantization and adversarial training and/or codec"*
# Lane: `lane_magic_codec_water_filling_lagrangian_unified_synthesis_cross_pollination_20260518` L0 (pre-registration; main-thread synthesis pending follow-on subagent for deep unification)

## STRATEGIC FRAMING

Three canonical primitives sit in `tac/` as separate-looking helpers but are mathematically unified by a single convex-optimization meta-structure. Surfacing this unification clarifies what's already wired and where cross-pollination is structurally cheap.

The three primitives:

1. **Magic Codec** (canonical: `tac.magic_codec` per task #511 / #525 / #556 / #467; commits 90bca47ff sister-ref). Auto-selector across multiple codecs per per-tensor characteristics (entropy / sparsity / dynamic range / score-sensitivity). Discovery + per-tensor optimal codec selection.

2. **Water-Filling (Ω-W)** (canonical: `tac.water_filling` per task #233 / #244 / #272 / #356; Boyd-style closed-form KKT solution; empirically validated -200 to -450 bp on eligible substrates per Lane Ω-W-V2/V3). Allocate bits across multiple resources (tensors / pixels / frequencies) per concave rate-distortion curve.

3. **Lagrangian Redirection** (canonical: `tac.run_admm` + `tac.meta_lagrangian_search` + `tac.per_pair_optimal_treatment_plan_via_lagrangian_dual` per task #245 / #364 / #790; Catalog #319 Q3 autopilot v2 cascade is the canonical consumer). Dual decomposition over {repr, predict, quant, entropy} via shared multipliers + alternating updates.

## THE UNIFICATION

All three solve the SAME canonical problem:

```
maximize    Σ_i utility_i(x_i)        # concave per-resource utility
subject to  Σ_i cost_i(x_i) ≤ B       # linear budget constraint
            x_i ∈ Choice_i             # per-resource feasible set
```

The solution is the KKT system:

```
∂utility_i / ∂x_i = λ * ∂cost_i / ∂x_i    for all active i
λ ≥ 0; complementary slackness            for the budget constraint
x_i ∈ Choice_i                             for each resource
```

The three primitives differ ONLY in which surface they operate:

| Primitive | Resource (x_i) | Utility | Cost | Choice_i | Solver |
|---|---|---|---|---|---|
| **Magic Codec** | per-tensor codec selection | score improvement per codec | bytes per codec | discrete: {brotli, lzma, arithmetic, FP4, AC, ...} | discrete enumeration + greedy |
| **Water-Filling** | per-tensor bits | -d(distortion)/d(bits) (concave) | bits | continuous: [0, max_bits] | closed-form KKT (water level λ) |
| **Lagrangian (ADMM)** | per-block primal variables | joint reconstruction quality | rate + distortion | continuous + non-convex | alternating + shared multipliers |

**The mathematical unification:** they're all KKT-stationary points on the Pareto frontier of {rate, distortion, score}. Different solvers; same manifold; same dual variables.

## MANIFOLD STRUCTURE

The unified manifold is the **Pareto frontier of {rate, d_seg, d_pose}** under the contest score `25*R + 100*d_seg + sqrt(10*d_pose)`. Every primitive is a SLIDE along this frontier:

- **Water-Filling** slides along the rate axis at fixed quality (or vice versa) by reallocating bits across tensors
- **Lagrangian/ADMM** slides toward a chosen operating point by setting {λ_rate, λ_seg, λ_pose} multipliers
- **Magic Codec** slides ALONG the discrete-choice axis (which codec used per tensor) — orthogonal to the continuous bit-allocation axis

The three are **composable**: magic codec picks the codec per tensor → water-filling allocates bits within each codec's parameter space → Lagrangian/ADMM globally coordinates across {repr, predict, quant, entropy} blocks.

## CROSS-POLLINATION TO 4 SURFACES

### 1. Ablation work (substrate weight removal)

**Insight**: Ablation IS a discrete choice over {keep_weight_i, remove_weight_i} per weight. Currently treated as hand-designed (IMP cycles, magnitude pruning, etc.) — should be magic-codec-driven OR water-filling-driven OR Lagrangian-driven per the operator's chosen frame:

- **Magic Codec frame**: each weight has a "compressed representation" with multiple codec options ({drop, FP4, INT8 per-channel, codebook-VQ}); magic codec picks per-weight
- **Water-Filling frame**: weight importance (Fisher/Hessian diagonal) is the per-weight utility; bit-budget allocation across weights = which to keep at high precision
- **Lagrangian frame**: ablation IS a constraint (max-removed-fraction ≤ ρ); Lagrangian dual gives the soft policy (per-weight λ * importance ≥ threshold → keep)

**Cross-pollination opportunity**: replace hand-designed IMP magnitude-pruning with water-filling-driven sensitivity-aware pruning. The `tac.sensitivity_map` per-axis-weight surface already produces the per-weight utility scores; feed them into `tac.water_filling.allocate_bits(scores, total_bits=B)` to get optimal ablation pattern. Catalog #305 observability surface declares this as ACTIVE for any new ablation work.

### 2. Quantization (FP4 / FP8 / INT4 / per-channel scales)

**Insight**: Current quantization is hand-designed per substrate (e.g. PR101 uses FP4 + per-channel scales; PR106 uses int4 ternary; etc.). This is the SAME math as water-filling + magic codec:

- **Magic Codec frame**: per-tensor quantization scheme is a discrete choice (FP4 / FP8 / INT4 / INT8 / per-channel / block-FP / codebook-VQ). Magic codec selects per-tensor based on (entropy, sparsity, dynamic-range, score-sensitivity).
- **Water-Filling frame**: bits-per-value allocation across tensors. Total bits = B; per-tensor bit budget chosen to maximize Σ_i quality_i(bits_i). For Gaussian-like tensors with concave quality-vs-bits curve, water-filling is closed-form optimal.
- **Lagrangian frame**: joint quantization-entropy-coding optimization via ADMM. Block 1: quantization parameters; Block 2: entropy coder; shared λ for the rate constraint.

**Cross-pollination opportunity**: build a `tac.quantization.water_filling_aware_quantize(tensor_dict, total_bits, sensitivity_map)` helper that takes per-tensor sensitivity (from `tac.sensitivity_map`) + applies water-filling per Boyd's KKT solution + selects per-tensor codec via `tac.magic_codec.auto_select`. Replaces hand-designed per-tensor quantization with unified canonical allocator. Predicted to improve per-archive compression by 5-15% at equal distortion per the Lane Ω-W-V2 +200-450 bp empirical anchor.

### 3. Adversarial training (UNIWARD / inverse-steganalysis / scorer-aware)

**Insight**: UNIWARD literally IS water-filling in the perturbation-budget direction:

> "UNIWARD: errors in textured regions are undetectable. Weight loss by inverse local variance."

The "weight loss by inverse local variance" IS the per-pixel utility curve for water-filling. The total perturbation budget B is the constraint. The KKT solution puts MORE perturbation in high-variance (textured) regions and LESS in smooth regions. This is the EXACT same math as Lane Ω-W's bit allocation across tensors.

The cross-pollination is structural:

- **Magic Codec frame**: per-region adversarial perturbation TYPE selection (texture-aware UNIWARD / boundary-aware SCT-style / scorer-class-targeted). Magic codec picks per-region.
- **Water-Filling frame**: per-pixel perturbation budget allocation. Sum-perturbation ≤ B; per-pixel utility = scorer-blindness (high in textured regions, low in smooth). Water-filling = optimal UNIWARD.
- **Lagrangian frame**: adversarial objective IS a Lagrangian (max distortion-to-scorer subject to L∞ detectability constraint). The "redirection" is gradient ascent on the inverse-scorer, dual-coordinated with a detectability constraint.

**Cross-pollination opportunity**: factor the canonical UNIWARD `tac.uniward` helper into a thin wrapper over `tac.water_filling.allocate(per_pixel_utility=inverse_variance, total_perturbation=B)`. Same code-path as Lane Ω-W bit allocation; just different utility function. Then sister inverse-steganalysis variants (SCT-style / scorer-class-targeted / boundary-aware) become ALTERNATIVE per-pixel utility functions that all flow through the same canonical allocator. Reduces 5+ adversarial-training helpers to 1 canonical surface + per-method utility-curve plugins.

### 4. Codec (in-flight TT5L V2 + Z7-Mamba-2 + cargo-cult resurrection)

**Insight**: Each substrate produces a TENSOR ZOO that should be magic-codec-driven + water-filling-allocated:

- **TT5L V2** tensors: VGGT pointmap latents + RSSM categorical latents + cooperative-receiver foveation map + DUSt3R outputs. Each has DIFFERENT statistics; each should get DIFFERENT codec.
- **Z7-Mamba-2** tensors: SSM state vectors (long-range structured) + per-pair residuals (sparse) + latent_init (small + dense). Each has different statistics.
- **A1 base archive**: latent + decoder weights + pose deltas. Already partially magic-codec-driven.

The TT5L V2 + Z7-Mamba-2 integration audits both flagged DP1 codebook initialization as HIGH-PAIR-CANDIDATE. The DP1 codebook is the LEARNED magic-codec entry — `Choice_i` = "use DP1 codebook for tensor i" vs "use raw bytes" vs "use FP4." Magic codec auto-selection across DP1-codebook-vs-alternatives IS the canonical cross-substrate composition primitive.

**Cross-pollination opportunity**: extend the JUST-LANDED `tac.master_gradient_archive_parsers` to feed per-tensor sensitivity into a unified `tac.water_filling_plus_magic_codec.allocate(tensor_zoo, total_bits)` surface that:
1. Reads each tensor's per-byte sensitivity (master_gradient output)
2. Runs magic_codec auto-selection per tensor (discrete codec choice)
3. Runs water-filling within each codec's parameter space (continuous bit allocation)
4. Runs ADMM globally to coordinate across substrate blocks (joint rate-distortion-score)

The Catalog #319 Q3 autopilot v2 cascade already routes through `predicted_delta_s_via_lagrangian_dual` — this surface IS the unified entry point. Wiring water-filling + magic codec into the cascade closes the producer-consumer loop end-to-end.

## OPERATOR-FACING SUMMARY

The three primitives (magic codec / water-filling / Lagrangian redirection) are different MANIFEST surfaces of the SAME convex-optimization-with-concave-utility-and-linear-budget-constraint math. Cross-pollination across:

- **Ablation** = water-filling-driven sensitivity-aware pruning
- **Quantization** = `tac.water_filling_aware_quantize` unified canonical allocator
- **Adversarial training** = UNIWARD-as-water-filling refactor; sister inverse-steganalysis variants as utility-curve plugins
- **Codec** = TT5L V2 / Z7-Mamba-2 / DP1 cross-substrate composition via unified `tac.water_filling_plus_magic_codec` surface fed by master-gradient sensitivity

The UNIFICATION is structurally cheap because:
1. `tac.water_filling` + `tac.run_admm` + `tac.magic_codec` already exist as separate canonical helpers
2. `tac.sensitivity_map` per-axis-weight surface already produces per-tensor utility scores
3. `tac.master_gradient_archive_parsers` (JUST LANDED at `117ac364d`) produces per-tensor sensitivity for 8 archive grammars
4. Catalog #319 Q3 autopilot v2 cascade already routes through `predicted_delta_s_via_lagrangian_dual` — the canonical entry point
5. The 9-dimension success checklist evidence section per Catalog #294 forces this kind of unification at every substrate design

The next-step subagent should:
1. Audit the 4 canonical helpers' current API surfaces (water_filling / run_admm / magic_codec / meta_lagrangian_search) for parity
2. Design the unified `tac.water_filling_plus_magic_codec_plus_admm.allocate(...)` canonical surface
3. Identify 3-5 specific cross-pollination integration points (ablation / quantization / adversarial / codec) with predicted ΔS bands
4. Propose Codex routing directives for each cross-pollination

## CROSS-REFERENCES

- Lane Ω-W water-filling (#233; commits ref'd in task list): canonical water-filling implementation
- Lane Joint-ADMM (#245): Boyd-style coordinator across {repr, predict, quant, entropy}
- Magic codec auto-selector + meta-codec discovery (#511 AA; #525 NN; #556 SSS; #467 #2 per-frame difficulty map + frame-conditional codec): canonical magic-codec implementation
- Meta-Lagrangian search engine (#364; commit ref): automated extreme-optimization pipeline
- `tac.per_pair_optimal_treatment_plan_via_lagrangian_dual` (#790): per-pair Lagrangian dual; Catalog #319 Q3 v2 cascade canonical entry point
- `tac.sensitivity_map` (#275 + #586 COUNCIL-A1): per-axis-weight reweighting API
- `tac.master_gradient_archive_parsers` (#887 + commit `117ac364d`): JUST-LANDED canonical facade
- Cathedral autopilot v2 cascade (Catalog #319 Q3 + #322 v2 reweight): canonical consumer
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE": canonical discipline for this work class
- HNeRV parity discipline L6: "Score-domain Lagrangian (not weight-domain proxies)" — the unified framework IS the canonical score-domain Lagrangian
- DEEP-RESEARCH-WAVE convergent-truth tuples (commit `d7cd940d4`): Shannon↔IB↔RD↔MDL↔Bayesian↔Kolmogorov + Mallat↔Daubechies↔Donoho↔CRT all reduce to convex-optimization-with-concave-utility-and-linear-budget at the canonical surface

## NEXT-STEP ROUTING (when subagent slot frees)

Spawn comprehensive design subagent: `lane_magic_codec_water_filling_lagrangian_unified_synthesis_design_20260518`:
- Read this memo + the 4 canonical helpers' source code
- Build unified `tac.water_filling_plus_magic_codec_plus_admm` canonical surface design
- Audit cross-pollination integration points per 4 surfaces (ablation / quantization / adversarial / codec)
- Predict ΔS bands per integration with Catalog #296 Dykstra-feasibility checks
- Propose Codex routing directives for each cross-pollination
- Council deliberation v2 per Catalog #300 (T2 sextet + grand council: Boyd / Dykstra / Shannon / van den Oord / Hinton / MacKay)
- Sister to cargo-cult resurrection symposiums (which produced composable candidates)

— Main-Claude 2026-05-18 (operator-query-driven synthesis per *"remember the magic codec and water bucket filling and Lagrangian redirection work and maybe consider the principles and concepts and problems and solutions and manifolds involved and how they might apply to the ablation work and quantization and adversarial training and/or codec"*)
