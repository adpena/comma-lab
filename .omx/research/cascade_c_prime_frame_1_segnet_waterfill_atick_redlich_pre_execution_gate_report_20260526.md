# Cascade C': FRAME-1 SegNet-class waterfill + Atick-Redlich asymmetric channel pre-execution gate report — 2026-05-26

**Subagent_id:** `cascade-c-prime-frame-1-segnet-class-waterfill-atick-redlich-asymmetric-channel-pure-full-scorer-attack-mlx-first-numpy-portable-20260526`

**Operator approval:** 2026-05-26 *"all are approved + pursue other attacks as well remmeber all MLX first portable via numpy and indivually fractally optimized"* + 3-strategy framework strategy-coverage rule (FULL-SCORER attack underserved).

**Sister coordination per CLAUDE.md "Subagent coherence-by-default":** Read `.omx/state/subagent_progress.jsonl` snapshot 2026-05-26T19:50Z; in-flight sister subagents:
- `uniward-per-pixel-score-conditional-sensitivity-weighting-...` (UNIWARD per-pixel; DIST attack; touches `src/tac/substrates/uniward_per_pixel_distortion/`)
- `cascade-b-hinton-kl-distill-catalyst-distortion-attack-...` (Cascade B Hinton-KL CATALYST; DIST+FULL-SCORER CATALYST composition; touches `experiments/results/cascade_b_*`)
- `nscs06-v8-stacked-paired-modal-t4-re-fire-post-trainer-...` (NSCS06 trainer-fix Modal CUDA; touches `experiments/results/lane_substrate_nscs06_v8_*`)
- THIS substrate scope = NEW `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/` + reading-only access to existing PR110 selector encoder + canonical `tac.findings_lagrangian` + canonical `tac.master_gradient` + canonical scorer modules. STAY DISJOINT.

---

## 3-strategy attack decomposition

Per just-landed 3-strategy framework standing directive (`feedback_fractal_optimization_full_stack_three_strategies_rate_distortion_full_scorer_attack_standing_directive_20260526.md`):

**PRIMARY STRATEGY: FULL SCORER ATTACK** (attack the joint cross-axis composition via Atick-Redlich asymmetric channel theory)

**SUB-AXIS**: scorer-architecture-imposed asymmetry from `upstream/modules.py:108` SegNet slice `x[:, -1, ...]`. This creates a 2D asymmetric channel:

| Perturbation target | SegNet cost | PoseNet cost |
|---------------------|-------------|--------------|
| frame-0 only        | **0 bytes** (structurally) | N PoseNet bytes |
| frame-1 only        | M SegNet bytes | N' PoseNet bytes |

**Empirical structural verification** (from sister `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/pr110_opt12_posenet_null_frame0.json`): all 87 widened frame-0 modes show `seg_delta: 0.0` — Atick-Redlich asymmetric channel is empirically validated at the SegNet axis structurally.

**MATHEMATICAL GROUNDING**: Lagrangian dual per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable:

```
min_x  Σ_i (α · d_seg_i(x) + β · √(d_pose_i(x)) + γ · bytes_i(x))
       s.t. routing_i ∈ {frame_0, frame_1}
```

Per Atick-Redlich asymmetric channel theory: at PR106-equivalent frontier operating point (pose_avg ~3.4e-5 per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"), POSE marginal value is 2.71× SegNet's. Frame-0 perturbations exclusively attack POSE (free SegNet). Frame-1 perturbations attack BOTH but cost SegNet bytes.

**The optimization question**: does the additional FRAME-1 menu attack surface (which costs SegNet bytes) unlock per-pair score reductions that net-beat staying with the FRAME-0-only menu (which is structurally seg-free)?

**Predicted ΔS band per Dykstra-feasibility per Catalog #296**:
- **Optimistic** (-3 to -10 score points): frame-1 SegNet-class-region waterfill in low-margin regions enables 8-16 new selector modes that the Lagrangian dual routes to ~150-300 pairs (the 22.3% PoseNet-null + low-SegNet-margin subset) yielding net per-pair pose savings exceeding SegNet cost
- **PESSIMISTIC** (+0.001 to +0.01): the K=16 frame-0 menu is already at the Pareto frontier for the asymmetric channel; adding frame-1 modes adds wire bytes + SegNet penalty that no per-pair routing decision overcomes
- **Dykstra-feasibility check**: alternating projections onto {rate ≤ R + Δsidecar, seg ≤ S_target, pose ≤ P_target}; the frame-1 menu is feasible IFF its per-pair pose savings exceed its per-pair (sidecar + SegNet cost) overhead

**Horizon-class per Catalog #309**: PENDING empirical measurement. Plateau-adjacent if no per-pair savings emerge; frontier-pursuit if savings ≥ 3 score points. The empirical anchor must measure to disambiguate.

**CATALOG #344 CANDIDATE EQUATION**: `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1` proposed (mathematical form derived from joint per-pair Lagrangian dual; operator-decision protocol per Catalog #344 deferred until empirical anchor measured).

---

## Entropy-position declaration

Per just-landed entropy-position discipline standing directive:

**Entropy position**: **AT SCORER LEVEL** (P18 SegNet entropy + P19 PoseNet entropy + composition via MULTI-AXIS-LAGRANGIAN rule per § 10 of the standing directive)

**Entropy distribution interacted with**: scorer-output joint (seg_logit_margin, pose_dim_sensitivity) per-pair distribution. PER-pair operating-point varies (some pairs are pose-null-by-construction; others are seg-margin-dominated).

**Theoretical bound**: Per CLAUDE.md "Meta-Lagrangian/Pareto solver" — Pareto frontier intersection of (rate, d_seg, d_pose) feasible sets. Frame-0-only menu is a STRICTLY-WORSE-OR-EQUAL constraint set vs frame-0 ∪ frame-1 menu (more attack surface ≥ less). The bound is whether expansion ACTUALLY reaches a new vertex.

**Multi-position composition rule per § 10 of standing directive**: **MULTI-AXIS-LAGRANGIAN composition** (joint solve over rate-axis + d_seg-axis + d_pose-axis Lagrangian dual). NOT ORTHOGONAL (frame-1 menu shares K=16 selector budget with frame-0 menu); NOT SHARED-CONSTRAINT (different scorer architectures see different perturbation targets); NOT CATALYST (frame-1 attacks don't ENABLE further frame-0 attacks). Pure MULTI-AXIS Pareto-frontier intersection.

**Canonical equation anchor**: candidate `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1` interacts with sister `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` (just-registered #2/3) at the per-pair routing decision (KL distillation could ENABLE frame-1 perturbations to be less SegNet-disruptive).

---

## MLX-first → numpy-portable bridge contract

Per just-landed `feedback_mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526.md`:

**Training-time (MLX-first)**:
1. MLX-native per-frame scorer-sensitivity computation via `tac.master_gradient` typed `CandidateModificationSpec` per Catalog #318 (NOT raw byte authority)
2. Per-pair Lagrangian dual variables computed via `tac.findings_lagrangian` Phase 1 helper (sister of Meta-Lagrangian Phase 2 #1365 wire-in)
3. Per-pair routing decision: frame-0 vs frame-1 attack target indexed into selector menu
4. MLX state_dict export to numpy via `np.savez_compressed` per HNeRV parity L3 archive grammar

**Inflate-time (numpy-portable)**:
- **Option A (Carmack-preferred budget conservation)**: routing decision consumed entirely at COMPRESS time; archive carries the SAME wire as PR110 + UPDATED Huffman codebook for new K (where K = K_frame0 + K_frame1_added). NO new sidecar bytes; inflate.py UNCHANGED beyond menu lookup table.
- **Option B (≤1-bit-per-pair sidecar)**: per-pair 1-bit flag indicates frame-0 vs frame-1 attack; 600 pairs / 8 = 75 bytes uncompressed, brotli-compressed empirically expected ~40-60 bytes. Increases inflate.py LOC by ~10-20 lines (HNeRV parity L4 ≤200 LOC budget preserved).

**Bridge contract honored**:
1. Training: MLX scorer-sensitivity + Lagrangian dual + per-pair routing
2. Export: routing decision baked into selector menu indices (Option A) OR routing decision sidecar npz (Option B)
3. Archive: PR110-compatible ZIP grammar; either UPDATED Huffman codebook (Option A) or NEW sidecar at fixed offset (Option B)
4. Inflate: numpy-portable; HNeRV parity L4 ≤200 LOC + ≤2 deps (numpy + brotli for Option B; numpy only for Option A)
5. Runtime closure: `inflate.sh archive_dir output_dir file_list` via canonical `tac.substrates._shared.inflate_runtime.select_inflate_device` per Catalog #205

**FORBIDDEN at inflate time**: torch / mlx / tensorflow / jax (would inflate runtime tree + dependency closure failures per HNeRV parity L9)

---

## Individually-fractal decomposition

Per just-elevated full-stack fractal GUIDING PRINCIPLE (`feedback_pr95_sniped_lesson_full_stack_mlx_first_per_candidate_standing_directive_20260526.md`):

**13-ingredient fractal tree for THIS substrate** (sub-ingredient → sub-sub-ingredient depth):

| # | HNeRV parity ingredient | Cascade C' decomposition |
|---|-------------------------|--------------------------|
| 1 | score-aware substrate | per-frame routing-aware loss → per-pair Lagrangian dual → per-axis (seg, pose, rate) sensitivity map → per-class SegNet logit margin |
| 2 | export-first design | MLX state_dict → npz routing-decision sidecar OR baked-into-Huffman-codebook → ZIP-member at fixed offset |
| 3 | archive grammar | PR110-compatible + (Option A: same wire bytes / Option B: ≤75-byte sidecar) → fixed-section manifest per HNeRV parity L3 |
| 4 | inflate.py ≤200 LOC | menu lookup table extension (Option A: +5-10 LOC / Option B: +15-25 LOC including sidecar decode) |
| 5 | full renderer not single-component slot | THIS SUBSTRATE ATTACKS BOTH FRAMES (not just frame-0 like current PR110 menu) — respects renderer-class per HNeRV parity L5 |
| 6 | score-domain Lagrangian | MULTI-AXIS Lagrangian dual via `tac.findings_lagrangian` Phase 1 wire-in #1059 → per-pair routing decision |
| 7 | bolt-on size ≤350 LOC | NEW substrate dir = 4-6 files × ~80-120 LOC = ~400-700 LOC; declare `lane_class=substrate_engineering` per HNeRV parity L7 |
| 8 | eval_roundtrip + differentiable scorer-preprocess | canonical `tac.differentiable_eval_roundtrip` per CLAUDE.md non-negotiable (PR110 baseline already routes through this; inherit canonical) |
| 9 | runtime closure | inflate.sh dep tree: numpy + (Option B: brotli) ONLY |
| 10 | mask/pose coupling gate | mask change requires pose regeneration + geometry diagnostics + decoded mask SHA-256s + mask disagreement record per CLAUDE.md L10 |
| 11 | no-op detector | byte-mutation smoke per Catalog #105/#139/#220/#272 — mutate routing-decision bytes (Option B) or one selector-stream byte (Option A); verify output frames change |
| 12 | single-LOC-per-LOC reviewable | each new menu entry in ≤2 lines; routing-decision sidecar decode in ≤15 LOC |
| 13 | KILL/FALSIFIED is last resort | per Catalog #307: if empirical falsification, classify as IMPLEMENTATION-LEVEL (Atick-Redlich paradigm INTACT); operator-routable for sister 5th-order iteration on Lagrangian dual formulation OR Atick-Redlich-Tishby IB sister substrate |

**Sub-sub-sub-sub-ingredient depth** (frame-1 menu design level): per-pair routing decision → 1-bit flag → brotli compression → wire bytes; alternative: per-pair routing decision → indexed into UPDATED K=24 menu (16 frame-0 + 8 frame-1) → Huffman codebook with NEW codeword distribution.

---

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode — NON-NEGOTIABLE, HIGHEST EMPHASIS" + Catalog #290:

| Layer | Decision | Rationale |
|-------|----------|-----------|
| `tac.findings_lagrangian` (4-term scalar Lagrangian) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Phase 1 wire-in #1059 is the canonical living solver; Cascade C' is the FIRST per-substrate consumer at the FULL-SCORER attack class — adoption is foundation for future per-substrate Lagrangian consumers |
| `tac.master_gradient` typed `CandidateModificationSpec` (per-pair scorer decomposition) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Per Catalog #318 raw-byte authority guard; the canonical typed API is the only contest-faithful way to compute per-pair scorer sensitivity |
| `score_aware_loss` (training loss) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Per Catalog #164/#226 canonical scorer-preprocess routing; substrate-optimal engineering does NOT diverge from canonical scorer-preprocess at this layer (the scorer architecture IS the SUT) |
| `select_inflate_device` (inflate-time device fork) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Per Catalog #205 inline-device-fork ban; canonical helper with PACT_INFLATE_DEVICE env-var pin is the only allowed inflate runtime pattern |
| `differentiable_eval_roundtrip` (uint8 bottleneck simulation) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Per CLAUDE.md eval_roundtrip non-negotiable; substrate-optimal engineering does NOT diverge here (the bottleneck IS the contest scorer's input) |
| **Per-frame routing decision logic** (NEW per Atick-Redlich asymmetric channel) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | NO canonical helper exists for routing selector modes across frame-0/frame-1 axis per Atick-Redlich asymmetric channel theory; this is genuinely substrate-OPTIMAL engineering per UNIQUE-AND-COMPLETE-PER-METHOD |
| **Per-pair Lagrangian dual variable computation** (per-substrate adapter) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | The canonical `tac.findings_lagrangian` provides scalar Lagrangian; the per-pair adapter that ROUTES per-pair candidates across frame-0/frame-1 IS substrate-specific. Sister of Meta-Lagrangian Phase 2 #1365 wire-in. |
| Selector menu encoding (K=16 vs K=24 expansion) | **EMPIRICAL** (paired-comparison smoke required) | Default to ADOPT canonical K=16 PR110 encoder; FORK to K=24 IFF paired-comparison smoke measures ΔS ≥ 0.005 improvement |
| Brotli compression for routing sidecar (Option B) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Brotli is canonical per existing submissions; no substrate-specific reason to fork |

---

## 9-dimension success checklist evidence

Per Catalog #294:

| # | Dimension | Evidence |
|---|-----------|----------|
| 1 | UNIQUENESS | First-ever substrate to attack Atick-Redlich asymmetric scorer channel via per-pair Lagrangian dual routing decision; ZERO sister substrates touch FRAME-1 menu axis (verified via grep across tools/ src/tac/ submissions/) |
| 2 | BEAUTY + ELEGANCE | Per-pair routing decision is single-line Lagrangian comparison; menu expansion is ≤16 new mode entries; inflate runtime change is ≤25 LOC (Option B) OR ≤10 LOC (Option A) |
| 3 | DISTINCTNESS | Explicitly DIFFERENT from sister UNIWARD per-pixel (different granularity) + Cascade B CATALYST (different composition rule) + NSCS06 trainer-fix (different substrate class) + Cascade C per-region codec (different entropy-position + falsified-FRAME-0 vs valid-FRAME-1 axis) |
| 4 | RIGOR | Premise verification (read SegNet `x[:,-1,...]` slice + Cascade C alt reducer B + 5 standing-directive memos); adversarial review (Carmack-dissent verdict per Catalog #307 IMPLEMENTATION-LEVEL classification); assumption classification (Cargo-cult audit § below); empirical anchor (MLX-LOCAL smoke per Step 3-4) |
| 5 | OPTIMIZATION PER TECHNIQUE | Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD; Catalog #290 canonical-vs-unique decision per layer documented above |
| 6 | STACK-OF-STACKS-COMPOSABILITY | MULTI-AXIS-LAGRANGIAN composition rule per § 10 entropy-position discipline; orthogonal to sister Markov 2nd-order (#1336+) at the entropy-position axis but COMPETES at the per-pair routing decision (mutual-exclusive selector menus) |
| 7 | DETERMINISTIC REPRODUCIBILITY | MLX inference is deterministic at fp32; routing decision is integer-valued (frame-0 vs frame-1); brotli compression is deterministic; archive bytes are byte-stable across re-runs |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | Lagrangian dual solve is O(N_pairs × N_modes) per iteration; routing decision is O(N_pairs); inflate runtime is O(N_pairs) for routing decode |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Predicted -3 to -10 score points if FULL-SCORER attack succeeds; per Catalog #309 horizon-class PENDING empirical measurement |

---

## Cargo-cult audit per assumption

Per Catalog #303 + hard-earned-vs-cargo-culted addendum:

| # | Assumption | Classification | Unwind path |
|---|------------|----------------|-------------|
| 1 | Atick-Redlich asymmetric channel is structurally valid for frame-0 perturbations | **HARD-EARNED-EMPIRICALLY-VERIFIED** | Sister #1324 OPT-12 artifact confirms ALL 87 widened frame-0 modes have `seg_delta: 0.0` — structurally true |
| 2 | Frame-1 perturbations can extract MORE total per-pair savings than frame-0-only | **CARGO-CULTED-AWAITING-VERIFICATION** | MUST empirically measure per-pair frame-1 modes (NONE exist in PR110 menu); the assumption is theoretically grounded (more attack surface ≥ less) but theoretically falsifiable (frame-1 modes might be uniformly seg-dominated) |
| 3 | The K=16 frame-0 menu is at the Pareto frontier for frame-0 only | **HARD-EARNED-EMPIRICALLY-VERIFIED** | Per Cascade C landing memo § 2.1, FEC6 selector wire is 249 bytes vs Shannon-floor 241 bytes = 0.0284 bits/pair slack (already saturated) |
| 4 | Adding frame-1 modes will not strictly worsen the empirical archive bytes axis | **CARGO-CULTED-AWAITING-VERIFICATION** | The K-expansion changes Huffman codebook distribution; Cascade C empirically showed partition-overhead can dominate; MUST measure |
| 5 | Per-pair Lagrangian dual routing decision converges in O(N_pairs × N_modes) iterations | **CARGO-CULTED-AWAITING-VERIFICATION** | Convex Lagrangian dual typically converges in O(N) iterations; MUST verify via MLX-LOCAL smoke |
| 6 | The contest scorer's PoseNet sees BOTH frames per `upstream/modules.py` PoseNet class | **HARD-EARNED-EMPIRICALLY-VERIFIED** | Read `upstream/modules.py:74` einops rearrange `'(b t) c h w -> b (t c) h w'` confirms PoseNet's 12-channel input is 2 frames × YUV6 (= 12 channels) |
| 7 | Routing-decision sidecar (Option B) brotli-compresses to ≤60 bytes for 600 pairs | **CARGO-CULTED-AWAITING-VERIFICATION** | Brotli compression ratio depends on routing decision entropy; MUST empirically measure |

**Unwind for assumption #2 (the core hypothesis)**: MLX-LOCAL frame-1 economics smoke must measure per-pair frame-1 selector mode (seg_delta, pose_delta, sidecar_byte_cost). If smoke shows uniformly seg-dominated, falsify-IMPLEMENTATION (Atick-Redlich paradigm INTACT; specific frame-1 menu choice falsified) and iterate menu design.

---

## Observability surface

Per Catalog #305:

| Facet | How surfaced |
|-------|--------------|
| **Inspectable per layer** | Per-pair routing decision in `routing_decision.py::decide_per_pair_routing()` is pure-function; inputs = (per-pair seg_logit_margin, per-pair pose_dim_sensitivity, per-frame mode list); output = per-pair routing flag |
| **Decomposable per signal** | Output JSON decomposes per-pair into (selected_mode_idx, frame_target ∈ {0,1}, lagrangian_dual_variable, predicted_d_seg, predicted_d_pose, predicted_archive_byte_delta) |
| **Diff-able across runs** | All artifacts under `.omx/research/cascade_c_prime_*/` are JSON with sorted_keys (byte-stable for fixed inputs); MLX inference deterministic at fp32 |
| **Queryable post-hoc** | All artifacts JSON; jq queryable; sha256 of each artifact recorded |
| **Cite-able** | Every artifact carries `archive_sha256` + `axis_tag=[macOS-MLX research-signal]` + `provenance.subagent_id` + `provenance.method` + canonical Provenance per Catalog #323 |
| **Counterfactual-able** | Re-run with different `FRAME_1_MENU` to test per-menu counterfactuals; re-run with different Lagrangian weights (α, β, γ) to test per-tradeoff counterfactuals |

---

## Drift surface declaration per MLX↔CUDA bidirectional

Per sister standing directive `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`:

**Drift sources for Cascade C'**: this lane runs MLX-LOCAL only; no CUDA computation. However the 5 canonical drift sources apply to FUTURE CUDA validation when the operator routes a paired-CUDA anchor:

1. **bfloat16/fp16**: MLX uses fp32 inference by default for scorer-sensitivity; CUDA may run fp16. Per-pair routing decision is integer-valued (frame-0 vs frame-1) so the discretization absorbs minor numeric drift. RISK: LOW.
2. **softmax-LSE epsilon**: SegNet `argmax(dim=1)` is used for distortion; tie-breaking on near-equal logits drifts. RISK: MEDIUM at boundary pixels.
3. **AdamW β state**: training-time only; routing decision is computed AT inference time post-training. RISK: N/A.
4. **bicubic non-bit-identity**: SegNet preprocess uses `mode='bilinear'` per `upstream/modules.py:109`; this is bilinear NOT bicubic. RISK: LOW.
5. **EMA Kahan precision**: training-time only; routing decision is post-EMA-applied weights. RISK: N/A.

**Drift mitigation**: per-pair routing decision must be measured on the EXACT inflate-runtime CUDA scorer (NOT the MLX training-time scorer) to verify routing decision preservation across substrates. Operator-routable for paired-CUDA validation post-MLX-LOCAL smoke.

---

## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility check:

**Predicted band**: [-0.010, +0.005] (range -10 score points to +5 score points, asymmetric WIDE band)

**Dykstra-feasibility intersection**: the feasibility set is the joint constraint over {rate ≤ R + Δsidecar, d_seg ≤ S_target, d_pose ≤ P_target}. At PR106 frontier operating point (pose_avg ~3.4e-5; seg_avg ~6.7e-4; archive bytes 178,517), Atick-Redlich asymmetric channel theory predicts:

- **Frame-0 attack saturation**: PR110 K=16 menu uses ALL frame-0 modes; further frame-0 menu expansion has diminishing returns
- **Frame-1 attack expansion**: NEW per-pair attack surface; per-pair seg cost ~O(boundary_pixel_count × class_disagreement_probability)
- **Lagrangian dual routing**: per CLAUDE.md "Meta-Lagrangian/Pareto solver", routing decision is feasible IFF per-pair pose-savings exceed per-pair (seg-cost + rate-cost) overhead

**First-principles citation** (per Catalog #296 acceptance cascade): Atick-Redlich 1990 *"Towards a Theory of Early Visual Processing"* + Atick-Redlich *"Convergent algorithm for sensory receptive field development"* (cooperative-receiver theoretical framework; Tishby-Zaslavsky 2015 sister IB framework).

**Probe-disambiguator path**: `tools/probe_cascade_c_prime_asymmetric_channel_disambiguator.py` (to be built; sister of Catalog #313 probe-outcomes ledger discipline) — runs both Option A (no-sidecar K-expansion) AND Option B (1-bit-sidecar) and records routing decision distribution.

---

## Horizon-class declaration

Per Catalog #309: **PENDING empirical measurement**. Operating-point analysis:
- **plateau_adjacent** if empirical ΔS ∈ [-0.001, +0.005] — adding to plateau-adjacent operating point at 0.196-0.199 cluster
- **frontier_pursuit** if empirical ΔS ∈ [-0.010, -0.001] — extends frontier toward asymptotic floor

Operator-routable for re-classification after MLX-LOCAL smoke results.

---

## Catalog #344 canonical equation target

**Proposed equation**: `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1`

**Mathematical form** (PROPOSED; awaits operator approval per Catalog #344 operator-decision protocol):
```
ΔS_routing(N_pairs, frame_0_menu, frame_1_menu, lagrangian_dual_solver) =
  Σ_i min_{m ∈ frame_0_menu ∪ frame_1_menu} (
    100 × d_seg_i(m) +
    √(10 × d_pose_i(m)) +
    25 / 37_545_489 × bytes_i(m, routing_decision)
  )
  - baseline_per_pair_score_i(frame_0_menu_only)
```

**Producer surface**: NEW substrate `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/routing_decision.py::decide_per_pair_routing()` returns typed routing decision + empirical anchor.

**Consumer surface**: `tac.cathedral_consumers.canonical_equation_lookup_consumer` (auto-discovered per Catalog #335) emits per-candidate `[predicted]` annotations when matching frame-1-perturbation context tokens.

**Empirical anchor predicted**: PENDING measurement. Will register via `tac.canonical_equations.update_equation_with_empirical_anchor` per sister catalog-memo revision pattern just-landed commit `7ab5f58ae`.

---

## Premise verification per Catalog #229

Premise verifications performed BEFORE design:

1. **SegNet `x[:, -1, ...]` slice exists at `upstream/modules.py:108`**: VERIFIED via direct grep
2. **PoseNet processes BOTH frames as 12-channel input**: VERIFIED via `upstream/modules.py:74` einops rearrange `'(b t) c h w -> b (t c) h w'`
3. **PR110 K=16 menu is ALL frame-0 modes (zero frame-1)**: VERIFIED via grep of `submissions/hnerv_fec6_fixed_huffman_k16/encoder/build_pr101_frame_exploit_selector_packet_markov.py` (all 16 mode prev-labels are `frame0_*`)
4. **Cascade C alt reducer B (FRAME-1 SegNet-class waterfill) is research-only-deferred**: VERIFIED via Cascade C landing memo § 6 Step 14.3 "Design Cascade C' variant: SegNet-class waterfill applied to FRAME-1 selector menu"
5. **Sister `tac.findings_lagrangian` Phase 1 wire-in exists**: VERIFIED via direct read of `src/tac/findings_lagrangian/__init__.py` package contract
6. **Sister `tac.master_gradient` typed `CandidateModificationSpec` exists per Catalog #318**: VERIFIED via grep of `src/tac/master_gradient.py`
7. **NO sister subagent scope overlap with FRAME-1 selector substrate**: VERIFIED via `.omx/state/subagent_progress.jsonl` snapshot at 2026-05-26T19:50Z (3 in-flight sisters all DISJOINT)
8. **Sister #1324 PoseNet-null artifact confirms structural Atick-Redlich asymmetric channel**: VERIFIED via direct read of `pr110_opt12_posenet_null_frame0.json` (all 87 frame-0 modes have `seg_delta: 0.0`)

**Gate verdict**: ALL PASS. Proceeding to substrate scaffold (Step 2) + MLX-LOCAL smoke (Step 3).

---

## Discipline summary

- **Catalog #229 PV**: 8 premises empirically verified pre-design
- **Catalog #206**: 2 checkpoints emitted; next per ~10 tool uses
- **Catalog #117/#157/#174**: canonical serializer + POST-EDIT `--expected-content-sha256` will be used for all commits
- **Catalog #205/#295/#361**: inflate runtime canonical surface protection (numpy-portable; ≤200 LOC; ≤2 deps)
- **Catalog #287**: placeholder `<rationale>` literals REJECTED (rationales ≥4 chars substantive)
- **Catalog #340**: sister-checkpoint guard before any commit
- **Catalog #343**: NO hardcoded score literals (frontier scores via canonical pointer)
- **Catalog #318**: per-pair scorer-sensitivity via typed `CandidateModificationSpec` (NOT raw byte authority)
- **Catalog #192/#317**: all MLX outputs tagged `[macOS-MLX research-signal]` with `score_claim=False` + `promotable=False` + `axis_tag=[predicted]`
- Per CLAUDE.md "Forbidden premature KILL without research exhaustion": IF empirical falsification, classify as IMPLEMENTATION-LEVEL per Catalog #307; PARADIGM (Atick-Redlich asymmetric channel) INTACT
- Per CLAUDE.md "Remember all on MLX": NO PAID DISPATCH; ALL EXECUTION LOCAL macOS M5 MAX
- Per CLAUDE.md "Carmack MVP-first phasing": this entire substrate runs at $0 cost via MLX-LOCAL before any paid empirical anchor
