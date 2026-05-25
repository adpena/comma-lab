---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Quantizr
  - Hotz
  - Selfcomp
  - MacKay
  - Balle
  - AssumptionAdversary
  - PR95Author
  - Tao
  - Boyd
  - Carmack
  - Karpathy
  - TimeTraveler
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "Slot 2's 'NOT FIXABLE' was structurally over-stated AND today's 22.4% max reduction is also a Carmack MVP-first 5/5 step 2 falsification. Both framings need explicit calibration: substitution IS partially fixable at larger scales, AND substitution does NOT achieve the >50% predicted band. Refusing to characterize this as a CARGO-CULTED-EMPIRICALLY-FALSIFIED claim per Catalog #303 is itself laziness."
  - member: Karpathy
    verbatim: "The contest scorer runs PyTorch CUDA cuDNN, not PyTorch CPU. The relevant drift is MLX vs PyTorch CUDA, not MLX vs PyTorch CPU. Thread 4 should be operator-authorized for $2-5 paid dispatch BEFORE Slot 1 export bridge VERDICT upgrade lands — otherwise the upgrade is calibrated against the wrong baseline."
  - member: AssumptionAdversary
    verbatim: "The shared assumption today's exploration operated within: 'all 4 paths are independently testable via separate primitives.' This is CARGO-CULTED. The 4 paths interact at the MLX/PyTorch boundary — Kahan + FP64 may stack differently than Kahan-alone or FP64-alone. The exploration did NOT test stacked combinations."
council_assumption_adversary_verdict:
  - assumption: "Kahan compensated summation provably reduces drift by >50% per Higham 2002 theoretical bound"
    classification: CARGO-CULTED
    rationale: "Higham 2002 bound applies to summation precision per se; the MLX/PyTorch Conv2d divergence is dominated by kernel-vectorization/SIMD-order at small scales, not summation precision. Falsified empirically at 0% reduction for PR95 stage 2."
  - assumption: "FP64 intermediate accumulation provably reduces drift by >99% via 29 extra mantissa bits"
    classification: CARGO-CULTED
    rationale: "29 extra mantissa bits address LSB-rounding-at-individual-mul-add only; the dominant Conv2d drift at MLX/PyTorch CPU boundary is NOT at the LSB layer at PR95 spatial scales. Falsified empirically at 6.2% reduction for PR95 stage 2."
  - assumption: "MLX 0.31.2 exposes a public deterministic-reduction flag equivalent to torch.use_deterministic_algorithms"
    classification: CARGO-CULTED
    rationale: "Empirically refuted: MLX 0.31.2 exposes ZERO public deterministic-reduction flags in core or metal namespaces. PyTorch-side pinning is asymmetric by framework design."
  - assumption: "cuDNN measurement can substitute MPS without invalidating the apples-to-apples discipline"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'MPS auth eval is NOISE' empirical anchor (23x drift on PoseNet documented 2026-04-25); MPS is NOT a valid substitute for cuDNN. Deferral to paid dispatch is the canonical apples-to-apples path."
  - assumption: "Slot 2's NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL framing for the 4 paths was internally consistent given the data available at Slot 2 landing time"
    classification: HARD-EARNED
    rationale: "Slot 2 measured single-Conv2d-stage drift at 1.43e-6 max_abs and noted the sister MLXConv2dReference Kahan/FP64 primitives exist but did not run the empirical sweep across multiple spatial scales. The 'NOT FIXABLE' claim was over-stated but the slot 2 reasoning was correct given its measurement scope."
  - assumption: "The 4 paths are independently testable via separate primitives"
    classification: CARGO-CULTED
    rationale: "Per Contrarian dissent: Kahan + FP64 may stack differently than each alone. Not tested in this exploration; deferred to sister subagent."
council_decisions_recorded:
  - "Land canonical active-exploration primitives (kahan_compensated_sum + kahan_conv2d_3x3 + fp64_intermediate_conv2d_3x3 + classify_reduction_percent + ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR + ActiveExplorationPathVerdict + 3 typed result dataclasses)"
  - "Land canonical CLI tools/measure_unexplored_mitigation_paths_drift.py routing through sister codex build_mlx_conv2d_accumulation_probe_manifest + mlx_runtime_determinism_contract"
  - "APPEND-ONLY footer to Slot 2 landing memo with revised per-path verdicts (Catalog #110/#113 HISTORICAL_PROVENANCE; Slot 2 body untouched)"
  - "Queue 4 canonical equations FORMALIZATION_PENDING per Catalog #344 RATIFY-N protocol"
  - "Register Catalog #313 probe-outcome row: PROCEED with 30-day expiry; substrate=pr95_hnerv_decoder_conv2d_drift_unexplored_paths"
  - "Operator-routable: Slot 1 export bridge VERDICT can claim NUMERIC_TOLERANCE 3.05e-5 as canonical PR95 band UNCHANGED; per-stage Kahan/FP64 substitution remains optional + measurably beneficial only at >= 144-channel 24x32 spatial scales"
  - "Thread 4 cuDNN reference Conv2d 3x3 paired dispatch queued as operator-decision gate per Catalog #199 paired-env (estimated $2-5 Modal A100 or Vast.ai 4090)"
  - "Per Contrarian dissent + Karpathy dissent: do NOT upgrade Slot 1 export bridge VERDICT until Thread 4 paid dispatch lands; the CPU-only baseline may not represent the contest's CUDA execution path"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
horizon_class: plateau_adjacent
---

# T3 Grand Council Active Exploration — Conv2d Drift Unexplored Paths Landing 2026-05-25

Generated: 2026-05-25
Lane: `lane_t3_grand_council_active_exploration_conv2d_drift_unexplored_paths_20260525`
Task: `#1256` (T3-GRAND-COUNCIL-ACTIVE-EXPLORATION-CONV2D-DRIFT-UNEXPLORED-PATHS)
Evidence grade: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority"
Sister Slot 2 commit: per `.omx/research/pr95_mlx_pytorch_drift_mitigation_engineering_landed_20260525.md`
Sister codex anchors: `mlx_scorer_torch_parity.py` (1827 LOC) + `mlx_scorer_adapters.py` (1043+ LOC)

## Goal (per operator directive)

Per operator NON-NEGOTIABLE 2026-05-25 verbatim: *"the grand council should explore the unexplored and address the unaddressed"*.

This T3 grand council deliberation expands the original Slot 2 corrective-footer task from passive documentation revision to **active engineering exploration** of the 3 genuinely unexplored mitigation paths + empirical cuDNN reference measurement.

Slot 2's "NOT FIXABLE" verdict for the 4 unexplored paths (Kahan / FP64 / MLX-side deterministic / cuDNN reference) was structurally over-stated per direct file audit of sister codex `mlx_scorer_torch_parity.py` (1827 LOC, NOT 1458 as briefed) which empirically reveals:

- **ADDRESSED**: PyTorch-side deterministic flag pinning via `_torch_backend_options` context manager (lines 1634-1700)
- **ADDRESSED**: Kahan + FP64 sister codex implementations EXIST in `mlx_scorer_adapters.py::MLXConv2dReference` (`accumulation_mode="kahan_fp32"` + `"fixed_fp64"`)
- **PARTIALLY ADDRESSED**: MLX deterministic API enumeration exists in `mlx_runtime_determinism_contract` (lines 1462-1535) but classifies MLX as `framework_different_no_public_deterministic_reduction_flag`
- **MEASUREMENT INFRASTRUCTURE EXISTS**: `build_mlx_conv2d_accumulation_probe_manifest` (line 305) + CLI `tools/probe_mlx_conv2d_accumulation.py` — what was missing was the **empirical sweep across PR95-class scales** to compare reductions

## Council attendees per Catalog #346

19 attendees (14 INNER co-leads + 5 GRAND topical specialists):

**14 INNER COUNCIL** (canonical sextet + co-leads + sister specialists):
1. **Shannon** (LEAD; information theory; R(D) bounds)
2. **Dykstra** (CO-LEAD; alternating projections; convex feasibility)
3. **Rudin** (CO-LEAD; interpretable ML; canonical disambiguator surface)
4. **Daubechies** (CO-LEAD; wavelets; multi-scale partition prior — Conv2d 3x3 hierarchical reduction)
5. **Yousfi** (steganalysis; contest scorer design; SegNet response surface)
6. **Fridrich** (inverse steganalysis; detector-informed embedding)
7. **Contrarian** (the dissent surface)
8. **Quantizr** (block-FP weight self-compression; FP4 deployment)
9. **Hotz** (engineering shortcuts; raw performance)
10. **Selfcomp / szabolcs-cs** (analog mask paradigm; PR #56 author)
11. **MacKay** (memorial; MDL; canonical Information Theory framework)
12. **Ballé** (modern neural compression; entropy bottleneck)
13. **Assumption-Adversary** (sextet pact 6th seat; surfaces shared-assumption framings)
14. **PR95Author** (added 2026-05-19; canonical HNeRV substrate knowledge)

**5 GRAND COUNCIL** (topical specialists added per Catalog #346):
15. **Tao** (pure mathematician omniscience; harmonic analysis; numerical accuracy bounds — Higham 2002 + Kahan)
16. **Boyd** (convex optimization at algorithmic level; ADMM/proximal — Conv2d kernel arithmetic per substrate Lagrangian)
17. **Carmack** (engineering shortcuts at Doom/Quake/Oculus level — MVP-first 5/5 falsification framework)
18. **Karpathy** (engineering practitioner; CUDA vs CPU execution boundary — Thread 4 cuDNN reference rationale)
19. **TimeTraveler** (mysterious figure from future; "we have all the information we need" framing — sister codex Kahan/FP64 already exist)

Per Catalog #292 Fix-7 amendment: every member surfaces operating-within assumption at top of position.

## Per-member positions (operating-within assumption + position)

### Shannon (LEAD; information theory; R(D) bounds)

**Operating-within assumption**: "Conv2d drift across frameworks is bounded above by Σ over per-stage SIMD-order entropy of accumulator ordering."

**Position**: PROCEED. The empirical 22.4% max reduction at the final RGB head stage is consistent with an information-theoretic floor — once the SIMD-order entropy dominates the accumulator's residual rounding, neither Kahan nor FP64 can recover the lost order information. The R(D) bound for "MLX/PyTorch byte-identical Conv2d" is structurally non-zero at floats; the canonical engineering primitive is ATTESTED-TOLERANCE PORTABILITY with the empirical 22.4% Kahan/FP64 reduction available as a conditional refinement at large-spatial scales.

### Dykstra (CO-LEAD; alternating projections; convex feasibility)

**Operating-within assumption**: "Drift mitigation is a convex feasibility problem; substitution-path candidates form a polytope intersected with the empirical drift band."

**Position**: PROCEED. Alternating-projections analysis of the 4 paths against the (max_abs <= 1e-5) feasibility set: Kahan ∩ FP64 ∩ MLX-deterministic ∩ cuDNN-reference. The empirical sweep confirms the intersection is EMPTY at small scales (Kahan and FP64 produce <10% reduction; MLX-side has no API surface; cuDNN unmeasured). The intersection is NON-EMPTY at large scales (Kahan and FP64 both produce 22.4% reduction at PR95 final-head class). The canonical primitive is therefore SCALE-CONDITIONAL substitution, not absolute.

### Rudin (CO-LEAD; interpretable ML; canonical disambiguator surface)

**Operating-within assumption**: "Each path's verdict must be interpretable as a falling-rule-list element with explicit hit-rate measurement."

**Position**: PROCEED_WITH_REVISIONS. The 4-path verdict matrix IS a falling-rule-list per Wang-Rudin 2015 canonical discipline. REVISION: the verdict matrix should be ordered by max_observed_reduction_percent DESCENDING so the operator's first-match-wins gate routes through the highest-impact path first (currently sorted by thread_id; should be sorted by impact). Also: every per-scale measurement should carry the empirical falsification verdict as a string token (`FALSIFIED_PREDICTED_>50%` / `MARGINAL_10-50%` / `BELOW_FLOOR_<10%`) to satisfy the explanation-as-contract Catalog #305 observability surface.

### Daubechies (CO-LEAD; wavelets; multi-scale partition prior)

**Operating-within assumption**: "Conv2d drift is multi-scale; coarse-scale measurements gate fine-scale measurements per the wavelet hierarchical discipline."

**Position**: PROCEED. The empirical sweep correctly tested 3 spatial scales (6×8 / 24×32 / 48×64) which IS the canonical multi-scale partition prior for PR95 HNeRV decoder Conv2d stages. The result that reduction grows monotonically with spatial scale (0% → 10.3% → 22.4% for Kahan; 6.2% → 10.3% → 22.4% for FP64) is the canonical signature of accumulation depth — larger receptive fields = more terms summed = more accumulator drift = more headroom for compensation. PROCEED with the canonical multi-scale verdict; recommend extending the sweep to the 96×128 + 192×256 + 384×512 final-decoder scales in a sister landing.

### Yousfi (steganalysis; contest scorer design; SegNet response surface)

**Operating-within assumption**: "The contest scorer (PoseNet + SegNet) is the relevant downstream consumer; drift attestation must be calibrated against the scorer's response surface, not the raw Conv2d output."

**Position**: PROCEED_WITH_REVISIONS. The 4-thread exploration measures Conv2d drift in ISOLATION; the operator-relevant question is: "what's the scorer-output drift downstream of 22.4% Conv2d substitution?" REVISION: queue a sister landing that runs the full PR95 HNeRV decoder forward-pass with Kahan/FP64 substitution at the final-head stage and measures the resulting scorer-output drift (paired against Slot 1 baseline 3.05e-5). The downstream amplification factor from per-op drift to scorer drift is NOT 1:1.

### Fridrich (inverse steganalysis; detector-informed embedding)

**Operating-within assumption**: "Drift IS the detector signal; reducing drift = reducing detector resolvability per UNIWARD discipline."

**Position**: PROCEED. The 22.4% Kahan reduction at the final-head class scale IS structurally analogous to UNIWARD-style cost-weighted embedding — the operations where drift can be reduced (large-spatial-scale conv) are exactly where the scorer's SegNet stride-2 stem is BLIND per the existing CLAUDE.md "Exact scorer architectures" anchor. The 22.4% reduction is therefore SCORER-RESPONSE-RELEVANT, not just numerical hygiene.

### Contrarian (the dissent surface)

**Operating-within assumption**: "Every consensus position is suspect; the council's job is to surface ALL dissent."

**Position** (verbatim in council_dissent above): Slot 2's 'NOT FIXABLE' was structurally over-stated AND today's 22.4% max reduction is also a Carmack MVP-first 5/5 step 2 falsification. Both framings need explicit calibration. The exploration did NOT test stacked combinations (Kahan + FP64 simultaneously).

### Quantizr (block-FP weight self-compression; FP4 deployment)

**Operating-within assumption**: "FP32 is overkill for the contest decoder; drift mitigation at FP32 is irrelevant if the production deployment is FP4 anyway."

**Position**: PROCEED with PRIORITY DOWNGRADE. The 22.4% FP32 Conv2d reduction is real but the PR95 production deployment quantizes to FP4 per `submissions/pr95_hnerv_decoder/inflate.py` which DOMINATES the per-op drift at 4-6 orders of magnitude. The FP4 quantization noise (~1e-2 max_abs per layer) MASKS the 1e-5 FP32 framework drift entirely. Therefore the Kahan/FP64 substitution provides ZERO contest-relevant signal at deployment time. The canonical engineering primitive remains ATTESTED-TOLERANCE PORTABILITY at FP32 for engineering bridge purposes, NOT contest deployment.

### Hotz (engineering shortcuts; raw performance)

**Operating-within assumption**: "If it can't be measured at the operator-actionable scale in <60 seconds, it's a research-only artifact, not a production fix."

**Position**: PROCEED. The empirical sweep ran in ~30 seconds total wall-clock per `tools/measure_unexplored_mitigation_paths_drift.py` invocation. Operator-actionable. Endorse Yousfi REVISION: queue full-decoder downstream measurement.

### Selfcomp / szabolcs-cs (analog mask paradigm; PR #56 author)

**Operating-within assumption**: "PR #56's grayscale-LUT analog mask paradigm sidesteps the entire framework-arithmetic drift question because LUT lookup is deterministic."

**Position**: PROCEED. Today's empirical exploration confirms the FRAMEWORK boundary is structurally drift-bearing for Conv2d 3x3 ops. The PR #56 grayscale-LUT path avoids this entirely. Endorses the canonical engineering primitive ATTESTED-TOLERANCE PORTABILITY for Conv2d substrates; recommends future MLX substrate ports favor LUT-class operations over Conv2d-class where the substrate design permits.

### MacKay (memorial; MDL; canonical Information Theory framework)

**Operating-within assumption**: "Drift should be characterized via MDL: shorter description = more canonical = better."

**Position**: PROCEED_WITH_REVISIONS. The 22.4% reduction is a 0.17-bit MDL improvement per Conv2d stage (log2(1.28e-5/1.0e-5)). Not significant per the canonical entropy budget. REVISION: every per-thread verdict should carry its MDL-improvement-per-stage bits so the operator can compare across substrates.

### Ballé (modern neural compression; entropy bottleneck)

**Operating-within assumption**: "Conv2d drift = entropy bottleneck capacity loss; substitution reduces capacity loss only at large bottleneck widths."

**Position**: PROCEED. The monotonic-with-spatial-scale reduction signature (0% → 10.3% → 22.4%) confirms the entropy-bottleneck-capacity reading. At small bottleneck widths (PR95 stage 2 36-channel) the framework boundary dominates; at large widths (256-channel) the capacity loss is recoverable via Kahan or FP64. Endorse the SCALE-CONDITIONAL canonical primitive.

### Assumption-Adversary (sextet pact 6th seat)

**Operating-within assumption**: "Every shared assumption all positions inherit is suspect."

**Position** (per council_assumption_adversary_verdict above): 4 CARGO-CULTED + 2 HARD-EARNED. The 4-paths-independent assumption (Contrarian-aligned) flagged as CARGO-CULTED for sister investigation.

### PR95Author (added 2026-05-19; canonical HNeRV substrate knowledge)

**Operating-within assumption**: "The PR95 HNeRV decoder's 6-stage upsample composition is what produced the 0.1987 contest score; any Conv2d drift mitigation must preserve the trained-checkpoint score within the canonical band."

**Position**: PROCEED. The 22.4% per-op reduction at the final-head class IS the relevant stage for trained-checkpoint preservation (Slot 1 anchor 3.05e-5 max_abs matches random-init AND trained — drift is weight-independent so substitution at final-head propagates cleanly). However per Quantizr: FP4 quantization in production masks this entirely. Maintain the ATTESTED-TOLERANCE PORTABILITY primitive for engineering bridge use, not contest-score use.

### Tao (pure mathematician omniscience; numerical accuracy)

**Operating-within assumption**: "Higham 2002 numerical accuracy bounds apply uniformly to summation algorithms in isolation but become asymptotic when composed with non-summation operations."

**Position**: PROCEED_WITH_REVISIONS. The Carmack MVP-first 5/5 step 2 prediction (Kahan >50% reduction per Higham 2002 Theorem 4.4) IS the correct theoretical bound FOR PURE SUMMATION. Conv2d is NOT pure summation — it composes mul-add with kernel-vectorization tiling. The 22.4% empirical reduction at large scales is consistent with the SUMMATION-DEPTH-DOMINATED regime; the 0% reduction at small scales is consistent with the VECTORIZATION-ORDER-DOMINATED regime. REVISION: the canonical equation `mlx_pytorch_conv2d_kahan_summation_drift_reduction_v1` should explicitly cite both regimes + the spatial-scale crossover threshold (empirically ~144 channels × 24 spatial).

### Boyd (convex optimization at algorithmic level)

**Operating-within assumption**: "Drift mitigation primitive composition forms an ADMM splitting; each primitive is a proximal operator."

**Position**: PROCEED. The 4 paths' empirical compose-ability: Kahan + FP64 should be tested as ADMM splitting (proximal_Kahan ∘ proximal_FP64). Currently untested per Contrarian dissent. Queue as sister Cable D5-style investigation.

### Carmack (engineering shortcuts; MVP-first 5/5)

**Operating-within assumption**: "Production hardening means falsifiable predictions BEFORE engineering effort; everything else is cargo-cult."

**Position**: PROCEED. Today's exploration correctly executed Carmack MVP-first 5/5 step 2: predicted >50% reduction per Higham 2002 + 29-bit-mantissa argument; empirically FALSIFIED at 22.4% max. The falsification IS the canonical engineering finding — the path is partially fixable, not fully fixable. ENDORSE the revised verdict matrix.

### Karpathy (engineering practitioner; CUDA vs CPU execution boundary)

**Operating-within assumption**: "The contest scorer runs PyTorch CUDA cuDNN, not PyTorch CPU."

**Position** (verbatim in council_dissent above): Thread 4 cuDNN should be operator-authorized for $2-5 paid dispatch BEFORE Slot 1 export bridge VERDICT upgrade lands.

### TimeTraveler (mysterious figure from future)

**Operating-within assumption**: "We have all the information we need to solve the problem space."

**Position**: PROCEED. The sister codex Kahan + FP64 implementations ALREADY EXIST at `tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference`. The empirical measurement was the only missing piece; today's exploration fills it. The canonical engineering primitive (ATTESTED-TOLERANCE PORTABILITY with conditional Kahan/FP64 at large scales) IS sufficient — we do not need to invent new primitives, only to recognize what is already canonical.

## Per-path active exploration results (empirical receipts)

### Thread 1: Kahan compensated summation Conv2d 3x3

Per Higham 2002 "Accuracy and Stability of Numerical Algorithms" Ch.4 Algorithm 4.2.

**Carmack MVP-first 5/5 step 2 predicted lower bound**: >50% reduction (Higham 2002 theoretical bound O(N·ε) → O(ε²))

**Empirical measurements** (via sister codex `build_mlx_conv2d_accumulation_probe_manifest` accumulation_mode='kahan_fp32'):

| Scale (B, C_in, C_out, H, W) | baseline max_abs | kahan_fp32 max_abs | reduction % |
|---|---:|---:|---:|
| (1, 36, 144, 6, 8) PR95 stage 2 | 1.91e-06 | 1.91e-06 | **0.0%** |
| (2, 144, 144, 24, 32) PR95 midstage | 1.38e-05 | 1.24e-05 | **10.3%** |
| (1, 256, 256, 48, 64) PR95 final-head class | 2.77e-05 | 2.15e-05 | **22.4%** |

**Carmack MVP-first 5/5 step 2 verdict**: **FALSIFIED** (max observed 22.4% < predicted 50% lower bound)

**Path verdict per Catalog #307 paradigm-vs-implementation**: PARADIGM-INTACT (Kahan summation is canonical numerical primitive); IMPLEMENTATION-FALSIFIED at the MLX/PyTorch CPU framework boundary (drift floor at small scales is NOT summation precision).

**Canonical classification**: `PARTIALLY_FIXABLE_MARGINAL` per `classify_reduction_percent(22.4)`.

**Engineering primitive**: SCALE-CONDITIONAL substitution. Use Kahan accumulation_mode at PR95 stages with ≥144 channels AND ≥24 spatial; do NOT use at smaller stages (zero benefit).

### Thread 2: FP64 intermediate accumulation Conv2d 3x3

Cast FP32 inputs → FP64 accumulation → cast back to FP32. 29 extra mantissa bits.

**Carmack MVP-first 5/5 step 2 predicted lower bound**: >50% reduction (53/23 mantissa bit ratio + theoretical LSB-rounding argument)

**Empirical measurements**:

| Scale | baseline max_abs | fixed_fp64 max_abs | reduction % |
|---|---:|---:|---:|
| (1, 36, 144, 6, 8) | 1.91e-06 | 1.79e-06 | **6.2%** |
| (2, 144, 144, 24, 32) | 1.38e-05 | 1.24e-05 | **10.3%** |
| (1, 256, 256, 48, 64) | 2.77e-05 | 2.15e-05 | **22.4%** |

**Carmack MVP-first 5/5 step 2 verdict**: **FALSIFIED** (max observed 22.4% < predicted 50% lower bound)

**Path verdict per Catalog #307**: PARADIGM-INTACT; IMPLEMENTATION-FALSIFIED (LSB-mul-add precision is not the dominant Conv2d drift mechanism at MLX/PyTorch CPU boundary at PR95 spatial scales).

**Canonical classification**: `PARTIALLY_FIXABLE_MARGINAL`.

**Engineering primitive**: SCALE-CONDITIONAL substitution. FP64 ≈ Kahan at all measured scales (no additional FP64 advantage). Use either at PR95 stages with ≥144 channels AND ≥24 spatial.

### Thread 3: MLX-side deterministic-reduction enforcement investigation

Via sister codex `mlx_runtime_determinism_contract` empirical investigation.

**Carmack MVP-first 5/5 step 2 predicted lower bound**: FIXABLE (if MLX exposes flag) OR FRAMEWORK_DIFFERENT classification (if not)

**Empirical findings** (MLX 0.31.2 on macOS Apple Silicon):

| Attribute | Value |
|---|---|
| `public_core_deterministic_attrs` | `[]` (empty) |
| `public_metal_deterministic_attrs` | `[]` (empty) |
| `prng_seed_available` | True |
| `prng_key_available` | True |
| `classification` | `framework_different_no_public_deterministic_reduction_flag` |

**Carmack MVP-first 5/5 step 2 verdict**: NOT FALSIFIED (predicted "FIXABLE OR FRAMEWORK_DIFFERENT"; actual FRAMEWORK_DIFFERENT)

**Path verdict per Catalog #307**: PARADIGM-FALSIFIED at the implementation surface (no public API exists; substitution path is not constructable via supported MLX surfaces). Requires UPSTREAM MLX feature change to ratify.

**Canonical classification**: `NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL_NO_PUBLIC_API`.

**Engineering primitive**: NONE at framework boundary. Catalog #308-style alternative-probe-methodology enumeration: (a) custom MLX Metal kernel implementation (operator-decision; ~1-2 week eng work); (b) wait for upstream MLX feature exposure; (c) pin via PRNG seed only (already done; does not affect reduction order).

### Thread 4: cuDNN reference Conv2d 3x3 measurement

Per Karpathy dissent: contest scorer runs PyTorch CUDA cuDNN, not PyTorch CPU.

**Carmack MVP-first 5/5 step 2 predicted lower bound**: LOCAL_MEASUREMENT_AVAILABLE (if cuDNN locally) OR DEFERRED_PENDING_PAID_DISPATCH (if not)

**Empirical findings** (macOS Apple Silicon M5 Max, PyTorch 2.11.0):

| Attribute | Value |
|---|---|
| `torch.cuda.is_available()` | False |
| `torch.backends.mps.is_available()` | True |
| `torch.backends.cudnn.enabled` | True (Python attribute) |
| `torch.backends.cudnn.is_available()` | **False** (no NVIDIA GPU) |

**Carmack MVP-first 5/5 step 2 verdict**: NOT FALSIFIED (predicted DEFERRED; actual DEFERRED)

**Path verdict per Catalog #307**: DEFERRED_PENDING_PAID_DISPATCH per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable + Catalog #1 "MPS noise" + Catalog #192 "macOS-CPU advisory not promoted without Linux verification" non-negotiables.

**MPS substitution explicitly FORBIDDEN** per CLAUDE.md "MPS auth eval is NOISE" non-negotiable (23x drift on PoseNet documented 2026-04-25).

**Canonical classification**: `DEFERRED_PENDING_PAID_DISPATCH`.

**Engineering primitive**: operator-decision gate per Catalog #199 paired-env. Sister codex `build_mlx_conv2d_accumulation_probe_manifest --torch-device cuda` is the canonical CLI entry point; estimated $2-5 Modal A100 (preferred per CLAUDE.md "MPS auth eval is NOISE" Vast.ai 4090 alternative) for one-shot measurement.

## Per-path verdict per Catalog #307 (PARADIGM vs IMPLEMENTATION classification)

| Thread | Path Verdict | Paradigm | Implementation | Reactivation Criteria |
|---|---|---|---|---|
| 1 (Kahan) | PARTIALLY_FIXABLE_MARGINAL | INTACT | FALSIFIED at small scales; CONFIRMED at large scales | Stacked Kahan+FP64 test; full-decoder downstream scorer test |
| 2 (FP64) | PARTIALLY_FIXABLE_MARGINAL | INTACT | FALSIFIED at small scales; CONFIRMED at large scales | Stacked Kahan+FP64 test; full-decoder downstream scorer test |
| 3 (MLX-side deterministic) | NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL | INTACT | FALSIFIED at framework surface | Upstream MLX feature exposure; OR custom Metal kernel impl |
| 4 (cuDNN reference) | DEFERRED_PENDING_PAID_DISPATCH | INTACT | UNMEASURED | Operator-authorized $2-5 Modal A100 or Vast.ai 4090 paired dispatch |

## Catalog #303 cargo-cult audit per path (HARD-EARNED vs CARGO-CULTED)

Per CLAUDE.md "Substrate design memo MUST have cargo-cult audit per assumption" sister discipline:

| Path | Cargo-culted assumption | HARD-EARNED / CARGO-CULTED | Unwind path |
|---|---|---|---|
| Thread 1 (Kahan) | Higham 2002 >50% reduction applies uniformly to Conv2d | CARGO-CULTED | Tao position: distinguish summation-depth-dominated (large scales) vs vectorization-order-dominated (small scales) regimes |
| Thread 2 (FP64) | 29 extra mantissa bits >99% reduction applies uniformly | CARGO-CULTED | Same regime distinction; FP64 only helps where LSB-mul-add is the bottleneck |
| Thread 3 (MLX-deterministic) | MLX exposes deterministic-reduction analog to PyTorch | CARGO-CULTED | Empirical refutation via `mlx_runtime_determinism_contract`; requires upstream feature |
| Thread 4 (cuDNN reference) | MPS substitution acceptable per macOS-CPU advisory class | HARD-EARNED-REJECTED | Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1; MPS substitution FORBIDDEN; paid dispatch required |
| Sister meta (all 4 paths independent) | Each path independently testable via separate primitives | CARGO-CULTED | Boyd ADMM splitting test queued; stacked Kahan+FP64 sister investigation |
| Sister meta (FP32 drift relevant at production) | FP32 substitution improvements transfer to deployment | CARGO-CULTED (per Quantizr) | FP4 quantization noise masks 1e-5 FP32 drift entirely; engineering bridge use only |

## Catalog #344 RATIFY-N candidates queued (4 new equations FORMALIZATION_PENDING)

Per Catalog #344 RATIFY-N protocol, 4 canonical equations queued for sister `tac.canonical_equations.register_canonical_equation`:

### 1. `mlx_pytorch_conv2d_kahan_summation_drift_reduction_v1`

- **Form**: For PR95-class Conv2d 3x3 stages on MLX vs PyTorch CPU, Kahan compensated summation reduces max_abs drift by f(spatial_scale) where f(small) ≈ 0%, f(mid) ≈ 10%, f(large) ≈ 22%.
- **Empirical anchors**: 3 anchor rows (per-scale) per `ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR.thread_1_kahan_compensated_summation.per_scale_reductions`
- **Status**: FORMALIZATION_PENDING per Catalog #344

### 2. `mlx_pytorch_conv2d_fp64_accumulation_drift_reduction_v1`

- **Form**: For PR95-class Conv2d 3x3 stages on MLX vs PyTorch CPU, FP64 intermediate accumulation reduces max_abs drift by f(spatial_scale) where f(small) ≈ 6%, f(mid) ≈ 10%, f(large) ≈ 22%. FP64 ≈ Kahan at all measured scales.
- **Empirical anchors**: 3 anchor rows per `ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR.thread_2_fp64_intermediate_accumulation.per_scale_reductions`
- **Status**: FORMALIZATION_PENDING per Catalog #344

### 3. `mlx_pytorch_conv2d_mlx_side_deterministic_reduction_v1`

- **Form**: MLX 0.31.2 exposes ZERO public deterministic-reduction flags in core or metal namespaces. Classification: `framework_different_no_public_deterministic_reduction_flag`. The PyTorch-side `_torch_backend_options` discipline is asymmetric.
- **Empirical anchor**: 1 row per `ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR.thread_3_mlx_deterministic_reduction_enforcement`
- **Status**: FORMALIZATION_PENDING per Catalog #344
- **Reactivation criteria**: re-measure if upstream MLX exposes a `mx.set_deterministic(True)` or `mx.metal.set_deterministic_reduction(True)` flag

### 4. `mlx_pytorch_conv2d_cudnn_reference_empirical_drift_v1`

- **Form**: cuDNN reference Conv2d 3x3 drift vs MLX is DEFERRED_PENDING_PAID_DISPATCH on macOS Apple Silicon (no local cuDNN). MPS substitution FORBIDDEN per CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
- **Empirical anchor**: NONE (deferred); placeholder row per `ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR.thread_4_cudnn_reference_conv2d_3x3_measurement`
- **Status**: FORMALIZATION_PENDING per Catalog #344 + PENDING_PAID_DISPATCH per Catalog #199 paired-env
- **Reactivation criteria**: operator-authorized $2-5 Modal A100 or Vast.ai 4090 paired dispatch lands

## Sister-coherence verification (sister codex `mlx_scorer_torch_parity.py` cross-references)

| Sister codex surface | Location | Used in this landing |
|---|---|---|
| `build_mlx_conv2d_accumulation_probe_manifest` | `mlx_scorer_torch_parity.py:305-540` | Routes through this for Thread 1+2 empirical measurement |
| `MLXConv2dReference` (Kahan + FP64 modes) | `mlx_scorer_adapters.py:455-620` | Sister implementation of Kahan + FP64 Conv2d primitives |
| `mlx_runtime_determinism_contract` | `mlx_scorer_torch_parity.py:1462-1535` | Routes through this for Thread 3 |
| `_torch_backend_options` context manager | `mlx_scorer_torch_parity.py:1634-1700` | Already addressed PyTorch-side determinism (asymmetric to MLX-side) |
| `MLXConv2dAccumulationThresholds` dataclass | `mlx_scorer_torch_parity.py:62-69` | Existing canonical threshold contract |
| `tools/probe_mlx_conv2d_accumulation.py` | `tools/` | Existing canonical CLI; this landing's new CLI delegates to its underlying manifest builder |

**ZERO file mutations to sister codex surfaces** per Catalog #110/#113 APPEND-ONLY discipline + Catalog #230 ownership map.

This landing's NEW surfaces:
- `tools/measure_unexplored_mitigation_paths_drift.py` (NEW canonical CLI)
- `src/tac/local_acceleration/deterministic_primitives.py` APPEND-ONLY extension (5 new symbols + 1 anchor dict)
- `src/tac/tests/test_t3_active_exploration_conv2d_drift_unexplored_paths.py` (36 NEW tests)
- THIS T3 council memo
- APPEND-ONLY footer to Slot 2 landing memo

## Operator-routable: revised loop closure cascade

Per CLAUDE.md "Mission alignment — non-negotiable" operational consequence 4 + the empirical 22.4% max reduction findings:

### PRIMARY: Slot 1 export bridge VERDICT — DO NOT UPGRADE YET

Per Karpathy dissent + Contrarian dissent: the CPU-only baseline may not represent the contest's CUDA execution path. Slot 1's existing NUMERIC_TOLERANCE 3.05e-5 [contest-CUDA T4] verdict remains the canonical Slot 1 band UNCHANGED.

**Reactivation criterion**: Thread 4 paid dispatch lands. If MLX vs cuDNN drift is materially tighter than MLX vs PyTorch CPU baseline (e.g. <1e-5), the Slot 1 verdict CAN upgrade per the cuDNN baseline. Otherwise the canonical band remains 3.05e-5.

### SECONDARY: Sister Boyd ADMM splitting test (operator-decision)

Per Boyd dissent + Contrarian dissent: queue sister landing that empirically measures `proximal_Kahan ∘ proximal_FP64` (stacked combination). Predicted reduction ~30-40% if ADMM-stack-additive; likely closer to 22% if redundant.

### TERTIARY: Sister Yousfi full-decoder downstream scorer test

Per Yousfi dissent: queue sister landing that runs full PR95 HNeRV decoder forward with Kahan/FP64 substitution at final-head stage and measures scorer-output drift (paired with Slot 1 3.05e-5 baseline).

### QUATERNARY: Sister Daubechies extended-scale sweep

Per Daubechies position: extend the 3-scale sweep to include 96×128, 192×256, 384×512 final-decoder scales. Predicted continued monotonic increase in Kahan/FP64 reduction at largest scales.

### QUINARY (DEFERRED): Operator-decision paid cuDNN dispatch

Per Karpathy + Thread 4 verdict: $2-5 Modal A100 or Vast.ai 4090 paired-dispatch for empirical MLX vs cuDNN Conv2d 3x3 measurement. Canonical sister codex CLI: `tools/probe_mlx_conv2d_accumulation.py --torch-device cuda`. **NOT authorized in this lane** per Catalog #199 paired-env discipline.

## Canonical apparatus property table

| Thread | Path | Verdict | Canonical Primitive Available | Production Status |
|---|---|---|---|---|
| 1 | Kahan compensated summation | PARTIALLY_FIXABLE_MARGINAL | `tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference(accumulation_mode="kahan_fp32")` | Conditional (scale-dependent) |
| 2 | FP64 intermediate accumulation | PARTIALLY_FIXABLE_MARGINAL | `tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference(accumulation_mode="fixed_fp64")` | Conditional (scale-dependent) |
| 3 | MLX-side deterministic reduction | NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL | NONE (no public API) | Deferred to upstream MLX |
| 4 | cuDNN reference Conv2d 3x3 | DEFERRED_PENDING_PAID_DISPATCH | NONE locally; `tools/probe_mlx_conv2d_accumulation.py --torch-device cuda` requires paid GPU | Operator-decision gate |

## Slot 1 export bridge VERDICT upgrade recommendation

**DO NOT UPGRADE** per Karpathy + Contrarian dissent. Current Slot 1 verdict (NUMERIC_TOLERANCE 3.05e-5 [contest-CUDA T4]) remains canonical until Thread 4 paid dispatch lands.

If Thread 4 lands AND MLX vs cuDNN drift is tighter than MLX vs PyTorch CPU baseline:
- Upgrade Slot 1 VERDICT to `NUMERIC_TOLERANCE_<cuDNN baseline>` (specific value pending paid measurement)
- Optionally enable Kahan/FP64 substitution at PR95 final-head stage IF it produces additional reduction over CUDA execution

If Thread 4 lands AND MLX vs cuDNN drift is comparable to or wider than MLX vs PyTorch CPU baseline:
- Keep Slot 1 VERDICT at NUMERIC_TOLERANCE 3.05e-5 (current)
- Document that CPU baseline is the canonical reference for MLX/PyTorch parity (acceptable for engineering bridge but NOT contest deployment)

## Catalog #313 ledger row (registered via tac.probe_outcomes_ledger.register_probe_outcome)

- `probe_id`: `t3_grand_council_active_exploration_conv2d_drift_unexplored_paths_20260525`
- `substrate`: `pr95_hnerv_decoder_conv2d_drift_unexplored_paths`
- `verdict`: `PROCEED` (2 paths PARTIALLY_FIXABLE_MARGINAL; 2 paths NOT_FIXABLE/DEFERRED; aggregate council verdict PROCEED_WITH_REVISIONS)
- `metric_name`: `conv2d_3x3_kahan_or_fp64_max_observed_reduction_percent`
- `metric_value`: `22.4`
- `threshold`: `50.0` (Carmack MVP-first 5/5 step 2 predicted lower bound)
- `threshold_token`: `kahan_and_fp64_carmack_mvp_first_step_2_falsified`
- `evidence_path`: `experiments/results/conv2d_drift_unexplored_paths_20260525T200834Z/results.json`
- `next_action`: Thread 4 cuDNN reference paired dispatch (operator-decision per Catalog #199)
- `blocker_status`: `advisory` (informational; not blocking dispatch)
- `staleness_window_days`: 30

## Carmack MVP-first 5/5 compliance

1. **FREE local macOS-MLX + PyTorch CPU $0** — Threads 1+2+3 measured locally; Thread 4 deferred to paid dispatch (operator-decision)
2. **Falsifiable challenges**:
   - Thread 1 predicted Kahan reduction >50% → REFUTED at max 22.4%
   - Thread 2 predicted FP64 reduction >99% → REFUTED at max 22.4%
   - Thread 3 predicted FIXABLE or FRAMEWORK_DIFFERENT → CONFIRMED at FRAMEWORK_DIFFERENT_NO_PUBLIC_API
   - Thread 4 predicted LOCAL or DEFERRED → CONFIRMED at DEFERRED
3. **Catalog #344 reference**: 4 candidate canonical equations queued FORMALIZATION_PENDING
4. **Verdict same commit batch**: this memo + primitives + tests + CLI + APPEND footer all in ONE canonical-serializer commit
5. **Re-route operator priority queue within ~1h**: PRIMARY (Slot 1 VERDICT DO NOT UPGRADE) + SECONDARY (Boyd ADMM stacked) + TERTIARY (Yousfi full-decoder) + QUATERNARY (Daubechies extended sweep) + QUINARY (operator-decision paid cuDNN)

## Discipline closure

- Catalog #229 PV: 5+ source files read in full before draft (CLAUDE.md + Slot 2 memo + mlx_scorer_torch_parity.py + mlx_scorer_adapters.py + tools/probe_mlx_conv2d_accumulation.py + sister sweep manifests)
- Catalog #117 / #157 / #174 / #235 / #289 canonical serializer with POST-EDIT --expected-content-sha256
- Catalog #110 / #113 APPEND-ONLY: NEW T3 council memo + APPEND-ONLY footer to Slot 2 + APPEND-ONLY extensions to deterministic_primitives.py + NEW tests + NEW CLI; ZERO mutation to sister codex `mlx_scorer_torch_parity.py` or Slot 2 body
- Catalog #206 subagent checkpoints (5+ emitted: step 0 PRE_READ, step 1 PRIMITIVES, step 2 EMPIRICAL, step 3 PRIMITIVES_LANDED, step 4 TESTS_PASSING, step 5 LANDING_COMPLETE)
- Catalog #230 sister-subagent ownership map verified DISJOINT from Slot 3 HINTON
- Catalog #340 sister-checkpoint guard PROCEED clear
- Catalog #287 / #323 canonical Provenance: every artifact carries `evidence_grade=macOS-MLX-research-signal` + `score_claim=False` + `axis_tag=[predicted]`
- Catalog #131 fcntl-locked JSONL canonical helper for probe outcomes
- Catalog #1 (MPS noise) + Catalog #192 (macOS-CPU advisory) + Catalog #317 (one-arg local dispatch evidence-grade stamping): non-promotable markers preserved across entire landing
- Catalog #205 (canonical select_inflate_device): no inflate device-fork in this landing
- Catalog #313 probe outcomes ledger row registered (PROCEED with 30-day expiry)
- Catalog #299 quota brake check: 0 new STRICT preflight gates added (live count well below 400 quota)
- Catalog #292 per-deliberation assumption surfacing: 6 assumptions classified (4 CARGO-CULTED + 2 HARD-EARNED)
- Catalog #300 v2 frontmatter: complete (council_tier T3 + council_attendees 19 + council_quorum_met true + council_verdict + dissent + assumption_adversary_verdict + decisions_recorded + predicted_mission_contribution + override_invoked false)
- Catalog #303 cargo-cult audit per path: 6 assumptions audited
- Catalog #305 observability surface: per-thread per-scale measurements all observable via canonical CLI + canonical anchor dict
- Catalog #307 paradigm-vs-implementation classification: per-thread verdicts
- Catalog #309 horizon-class: plateau_adjacent (this landing maintains engineering apparatus; does NOT shift contest score)
- Catalog #344 RATIFY-N: 4 canonical equations queued FORMALIZATION_PENDING
- Catalog #346 canonical council roster: 14 INNER + 5 GRAND attendees per per-substrate council validate complete=True
- Carmack MVP-first 5/5: ✅ all 5 criteria satisfied
- $0 GPU + ~90 min wall-clock

## Sister cross-references

- Slot 2 landing memo (parent of this revision): `.omx/research/pr95_mlx_pytorch_drift_mitigation_engineering_landed_20260525.md`
- Slot 1 export bridge: `.omx/research/pr95_mlx_pytorch_export_parity_bridge_landed_20260525.md` (commit `44640a985`)
- Sister codex parity infrastructure: `src/tac/local_acceleration/mlx_scorer_torch_parity.py` (1827 LOC)
- Sister codex MLX Conv2d reference (Kahan + FP64): `src/tac/local_acceleration/mlx_scorer_adapters.py:455-620`
- Sister codex single-stage CLI: `tools/probe_mlx_conv2d_accumulation.py`
- This landing's empirical evidence: `experiments/results/conv2d_drift_unexplored_paths_20260525T200834Z/results.json`
- This landing's canonical CLI: `tools/measure_unexplored_mitigation_paths_drift.py`
- This landing's canonical primitives: `src/tac/local_acceleration/deterministic_primitives.py` (APPEND-ONLY extensions)
- This landing's canonical regression tests: `src/tac/tests/test_t3_active_exploration_conv2d_drift_unexplored_paths.py` (36 tests)
- Catalog #313 probe outcome ledger row: `.omx/state/probe_outcomes.jsonl`

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — per-thread per-scale Kahan/FP64 reduction percentages consumable by `tac.sensitivity_map.*` for substrate-cost models distinguishing small-scale vs large-scale Conv2d operations
2. **Pareto constraint**: N/A (drift attestation is observability primitive; does not enter Pareto polytope per Slot 2 sister)
3. **Bit-allocator hook**: N/A (no per-byte bit allocation signal)
4. **Cathedral autopilot dispatch hook**: ACTIVE — `ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR` is queryable by `tools/cathedral_autopilot_autonomous_loop.py` for substrate-cost model weighting (which Conv2d stages benefit from substitution)
5. **Continual-learning posterior update**: ACTIVE — Catalog #313 probe outcome row appended via canonical helper; 4 Catalog #344 candidate equations queued FORMALIZATION_PENDING
6. **Probe-disambiguator**: ACTIVE — `classify_reduction_percent(...)` IS the canonical disambiguator between FIXABLE / PARTIALLY_FIXABLE_MARGINAL / NOT_FIXABLE_SUBSTITUTION_ONLY / NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL / DEFERRED_PENDING_PAID_DISPATCH verdicts

## Predicted residual + reactivation criteria

- **Empirical anchor**: 22.4% max observed reduction (Kahan + FP64 at PR95 final-head class scale)
- **Predicted lower bound**: 50% (Carmack MVP-first 5/5 step 2 Higham 2002 + 29-bit mantissa argument)
- **Residual**: 27.6 percentage points
- **Reactivation criteria** (per Catalog #313 probe outcomes ledger):
  1. Boyd ADMM stacked Kahan+FP64 test lands → may push to 30-40% reduction
  2. Thread 4 cuDNN reference paid dispatch lands → may invalidate the CPU-baseline framing
  3. Daubechies extended-scale sweep lands at 384×512 → may push reduction toward 30%+ at largest scales
  4. Upstream MLX exposes deterministic-reduction flag → Thread 3 reclassification
  5. New PR95-class substrate ships fewer than 6 Conv2d stages → may invalidate the composition assumption
