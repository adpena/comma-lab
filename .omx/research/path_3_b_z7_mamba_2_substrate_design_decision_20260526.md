---
council_tier: T1
council_attendees: [Shannon, Dykstra, AssumptionAdversary, Contrarian, Dao-Gu-advisory, Hafner-advisory]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "Path (c) is the right verdict given the audit findings, but the bolt-on temptation will recur at Phase 3 if the L0 SCAFFOLD design memo doesn't explicitly enumerate which canonical helpers it ADOPTS (per the canonical-vs-unique decision rule's OBVIOUS-FIT branch) vs FORKS. The Phase 3 memo must carry the section header `## Canonical-vs-unique decision per layer` per Catalog #290 and itemize EVERY layer."
council_assumption_adversary_verdict:
  - assumption: "Path (c) FRESH SUBSTRATE will produce a coherent end-to-end Mamba-2 substrate within the L0 SCAFFOLD deliverable budget"
    classification: HARD-EARNED
    rationale: "L0 SCAFFOLD per the brief = design memo + skeleton + memos; no paid GPU. Coherence is a design-time deliverable, not a runtime claim. Per Catalog #294 9-dim checklist Dim 7 (deterministic reproducibility), the L0 SCAFFOLD is structurally feasible at $0."
  - assumption: "Fresh substrate forfeits paired-comparison cleanliness with Z7-LSTM/GRU sister"
    classification: CARGO-CULTED
    rationale: "Paired-comparison cleanliness was an inherited Z7 symposium Revision #2 binding pattern that assumed substrate-class-shift required holding all variables constant except the predictor primitive. The 2026-05-26 binding directive #1 (MLX-first + design-the-whole-stack) supersedes this — paired-comparison is no longer the dominant evaluation criterion; ABSOLUTE score against PR110 fec6 frontier 0.1928 is."
council_decisions_recorded:
  - "op-routable #1: Phase 3 L0 SCAFFOLD memo MUST carry Canonical-vs-unique decision per layer section per Catalog #290 (Contrarian's binding revision)"
  - "op-routable #2: Phase 3 L0 SCAFFOLD does not delete the existing time_traveler_l5_z7_mamba2 substrate dir (sister z7_mamba2_mlx_scaffold_ext_20260526 actively building); creates NEW substrate dir per HNeRV parity L7 substrate-engineering split"
  - "op-routable #3: declare research_only=true + canonical Provenance non-promotable markers per Catalog #1/#192/#317 since L0 is MLX-first design + scaffold only"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: z7_mamba2_v2_fresh_substrate_mlx_first_design
related_deliberation_ids:
  - path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526
  - z7_mamba2_substrate_design_memo_20260518
  - feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515
  - feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515
---

# Path 3 candidate B' — Z7-Mamba-2 PHASE 2 substrate-design decision

**Lane:** `lane_path_3_b_prime_z7_mamba_2_cargo_cult_first_20260526` (L0)
**Phase 1 input:** `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
**Cost:** $0 (decision-only)
**Wall-clock:** Phase 2 ~25 min

## 0. The decision per the brief's 3-path enumeration

Per the brief, choose ONE of:
- Path (a) JUSTIFIED-EXTEND with canonical adoption (all assumptions HARD-EARNED)
- Path (b) JUSTIFIED-EXTEND with explicit FORK (SOME assumptions CARGO-CULTED; preserve hard-earned core)
- Path (c) FRESH SUBSTRATE DESIGN from first principles (MAJORITY assumptions CARGO-CULTED OR Mamba-2 architecture too different to extend coherently)

## 1. The Phase 1 audit's evidence

Per Phase 1 audit memo:
- 8 NEW CARGO-CULTED assumptions surfaced (beyond CC-1..CC-10 already in 2026-05-18 design memo)
- 0 NEW HARD-EARNED assumptions
- 2 NEW HARD-EARNED-PARTIAL assumptions (decompose-and-fork)

The 8 NEW CARGO-CULTED assumptions span 4 orthogonal architectural axes:
- **Decoder axis**: CC-A (Z6 decoder force-fit)
- **Latent dimensionality axis**: CC-B (latent_dim=24) + CC-C (ego_motion_dim=8)
- **Training-pathway axis**: CC-D (stability-as-optimizer-fix) + CC-G (sequential autoregress instead of parallel scan) + CC-F (MLX-first scope misclassification)
- **Archive grammar axis**: CC-J (Z7MCM2 inheritance)

The remaining 2 NEW HARD-EARNED-PARTIAL items (CC-E + CC-H) require decompose-and-fork at the same orthogonal axes (Wyner-Ziv channel sizing + IB scalar).

## 2. The decision — **Path (c) FRESH SUBSTRATE DESIGN**

### Justification per the brief's binding criteria

The brief specifies Path (c) is chosen when "MAJORITY of assumptions classified CARGO-CULTED OR Mamba-2's optimal architecture is too different from existing scaffold to extend coherently."

The 8 NEW CARGO-CULTED assumptions are a clear MAJORITY of the new audit findings (8/10 = 80%). When added to the 2026-05-18 design memo's CC-1..CC-10 (5 of those 10 were CARGO-CULTED or PENDING), the cumulative CARGO-CULTED count is 13/20 = 65%.

The 4 orthogonal architectural axes the CARGO-CULTED assumptions span are ALL non-trivial structural decisions. Specifically:
- **Decoder axis** (CC-A): Mamba-2's selective-state-space output is a time-series (latent stream) that the Z6 decoder treats as a per-pair-independent image embedding. The architectural mismatch is FUNDAMENTAL: PixelShuffle decoder does not consume Mamba-2's distinguishing temporal structure.
- **Latent dimensionality axis** (CC-B + CC-C): the latent_dim=24 + ego_motion_dim=8 budget was sized for GRU's hidden_dim=128; Mamba-2's d_model=64 (sister-halved) creates a gradient bottleneck at the output_projection's 64→24 compression. This is an ARCHITECTURE mismatch, not a hyperparameter fork.
- **Training-pathway axis** (CC-D + CC-G + CC-F): the existing substrate's sequential autoregress force-fits Mamba-2 into GRU's per-pair loop, bypassing Mamba-2's distinguishing parallel-scan capability. The stability blocker IS a downstream symptom of training-pathway force-fit.
- **Archive grammar axis** (CC-J): Z7MCM2 grammar inherits Z7PCWM1's predictor_blob layout that wastefully ships Mamba-2's procedurally-regenerable A_log matrix.

### Why NOT Path (a) — JUSTIFIED-EXTEND with canonical adoption

Path (a) would require ALL existing assumptions classified HARD-EARNED. The 8 NEW CARGO-CULTED items make Path (a) structurally false-by-construction.

### Why NOT Path (b) — JUSTIFIED-EXTEND with explicit FORK

Path (b) is feasible — could fork only the 4-6 most-impactful CARGO-CULTED assumptions while keeping the rest CANONICAL-ADOPT. But the operator's binding directive #1 explicitly elevates *"design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"* — this is structurally a Path (c) directive, NOT a Path (b) directive.

Path (b) is the predecessor's failure mode: produce some forks + retain some canonical adoption + claim "extension is justified by audit." The operator's binding directive #2 forbids this *unless* the rigorous adversarial cargo-cult pass has been done first AND the pass concludes extension is justified. The pass DID complete, and it concluded extension is NOT justified.

### Predicted ΔS band — Path (c) per Catalog #309 horizon_class

Per Phase 1 §5 Dykstra-feasibility check:
- **P10 lower bound:** ΔS = -0.005 (partial unwind)
- **P50 median:** ΔS = -0.018 (substantive unwind across 4 axes)
- **P90 upper bound:** ΔS = -0.040 (all 8 unwinds realize; substrate sits below PR110 fec6 frontier 0.1928 at ~0.155)

**horizon_class declaration: frontier_pursuit** (P50 ΔS = -0.018 puts the substrate's predicted score at 0.1928 - 0.018 = 0.175, which sits at the upper-region of frontier_pursuit per Catalog #309 boundaries [0.120, 0.180]). If P90 realizes, the substrate enters asymptotic_pursuit [0.050, 0.120] — but the L0 SCAFFOLD's claim is FRONTIER_PURSUIT to remain honest about the predicted P50.

## 3. The implementation roadmap for Phase 3

### NEW substrate directory (canonical naming per Catalog #110/#113 APPEND-ONLY)

Per op-routable #2 (Contrarian's binding revision): DO NOT delete or mutate `src/tac/substrates/time_traveler_l5_z7_mamba2/` — the sister `z7_mamba2_mlx_scaffold_ext_20260526` is actively building there. The existing scaffold + sister extension are PRESERVED as historical Z7-Mamba-2-v1 (per HNeRV parity L7 substrate-engineering split; substrate engineering happens ONCE per architecture class).

NEW substrate directory: `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` (the v2 prefix structurally encodes the substrate-class-shift). This is the L0 SCAFFOLD skeleton's home.

### L0 SCAFFOLD deliverables (Phase 3 binding)

1. **Design memo** at `.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md` with all 7 required sections:
   - Catalog #300 v2 frontmatter (T1; PROCEED; frontier_breaking; frontier_pursuit)
   - `## Canonical-vs-unique decision per layer` (Catalog #290) — itemize every layer
   - `## 9-dimension success checklist evidence` (Catalog #294) — per-dim evidence
   - `## Cargo-cult audit per assumption` (Catalog #303) — reference Phase 1 audit memo
   - `## Observability surface` (Catalog #305) — all 6 facets
   - `## Predicted ΔS band` (Catalog #296) — Dykstra-feasibility intersection check
   - `horizon_class: frontier_pursuit` (Catalog #309)
   - Mamba-2 substrate math + SSD canonical formulation + MLX-implementation roadmap

2. **L0 SCAFFOLD skeleton** at `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` with:
   - `__init__.py` — SPDX header + `# LEGACY_SUBSTRATE_PRE_META_LAYER:<substantive rationale>` waiver per Catalog #241 (non-placeholder rationale per Catalog #287)
   - `architecture.py` — MLX Mamba-2 architecture (selective state-space recurrence + temporal-conv decoder pre-stage) per CC-A unwind
   - `mamba2_temporal_decoder.py` — NEW decoder optimized for Mamba-2's temporal latent stream per CC-A unwind
   - `training_curriculum.py` — MLX training schedule (chunk-parallel scan per CC-G unwind; A_log init scheme per CC-D unwind; latent_dim ∈ {16, 32, 48} sweep ready)
   - `archive.py` (Z7MCM3 grammar) — procedurally-regenerable A_log per CC-J unwind; cosine-similarity-quantized B/C
   - `inflate_runtime.py` — HNeRV parity L4 ≤200 LOC budget; scorer-free; CUDA/CPU agnostic via canonical `select_inflate_device` per Catalog #205
   - `tests/__init__.py` + `tests/test_basic.py` — shape + smoke tests

3. **Landing memo** at `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md` per Catalog #229 PV discipline:
   - PV-N empirical pre-edit state verification
   - Sister coordination per Catalog #230 (5 sisters from brief + the in-flight z7_mamba2 sister)
   - 6-hook wire-in declaration per Catalog #125
   - Mission contribution per Catalog #300

### Per-layer canonical-vs-unique decisions (Phase 3 binding)

| Layer | Decision | Rationale |
|---|---|---|
| Predictor primitive | UNIQUE-FORK | Mamba-2 selective state-space + SSD parallel-scan per CC-G unwind |
| Decoder | UNIQUE-FORK | `Mamba2TemporalDecoder` with Conv1D(d_conv=4) temporal pre-stage matching Mamba-2 window per CC-A unwind |
| Latent dim | UNIQUE-FORK | Default 32 (was 24); curriculum supports sweep ∈ {16, 32, 48} per CC-B unwind |
| Ego motion dim | UNIQUE-FORK | Default 16 (was 8); Wave N+1 paired smoke at {4, 8, 16, 24} per CC-C unwind |
| Context conditioner | CANONICAL-ADOPT | `LatentAffineContextConditioner` IS empirically validated; CC-5 HARD-EARNED inherited |
| Training pathway | UNIQUE-FORK | Chunk-parallel SSD-scan on CUDA; sequential reference on MPS per CC-G unwind |
| A_log init scheme | UNIQUE-FORK | Configurable ∈ {Z+1 (Mamba-2 default), HiPPO-like, log-uniform} per CC-D unwind |
| Stateful mode | CANONICAL-ADOPT (with WARN) | Default stateful=True per Wyner-Ziv pattern (CC-7 HARD-EARNED); but WARN at L1 disambiguator that channel may be over-provisioned per CC-E HARD-EARNED-PARTIAL |
| Archive grammar (Z7MCM3) | UNIQUE-FORK | Procedurally-regenerable A_log; cosine-quantized B/C; conv1d kernel-quantized per CC-J unwind |
| Loss formulation | CANONICAL-ADOPT (with FORK on ib_scale) | Contest-formula rate+seg+sqrt(pose) HARD-EARNED; ib_scale forkable per-substrate per CC-H HARD-EARNED-PARTIAL |
| Scorer routing | CANONICAL-ADOPT | `score_pair_components_dispatch` HARD-EARNED per Catalog #164 + #190 + CC-I HARD-EARNED inherited |
| F3 GTScorerCache | CANONICAL-ADOPT | Tier-1 engineering primitive HARD-EARNED per Catalog #228 + CC-I HARD-EARNED inherited |
| Inflate runtime | UNIQUE-IMPL (CANONICAL contract) | NEW file; CANONICAL `select_inflate_device` per Catalog #205; HNeRV parity L4 ≤200 LOC |
| eval_roundtrip | CANONICAL-ADOPT | HARD-EARNED CLAUDE.md non-negotiable |
| EMA decay | CANONICAL-ADOPT | 0.997 HARD-EARNED CLAUDE.md non-negotiable (sister-canonical) |
| MLX-iteration scope | UNIQUE-DESIGN | MLX-first per binding directive #1; explicitly NOT a stability-validation surface per CC-F |

**Net: 7 UNIQUE-FORK + 7 CANONICAL-ADOPT + 1 UNIQUE-IMPL + 1 UNIQUE-DESIGN = balanced design.** The substrate is NOT pure-greenfield (canonical adoption preserved at score-axis + HARD-EARNED non-negotiables + Tier-1 engineering primitives) AND NOT pure-bolt-on (unique forks at every CARGO-CULTED axis per Phase 1 audit).

## 4. The Mamba-2 substrate math (Phase 3 design-memo content preview)

### Selective state-space recurrence (per Dao-Gu 2024)

State-space model (continuous-time):
```
h'(t) = A h(t) + B x(t)
y(t) = C h(t)
```

Discretized via zero-order hold (per Mamba-2 §3):
```
A_bar = exp(dt * A)
B_bar = dt * B
h_t = A_bar * h_{t-1} + B_bar * x_t
y_t = C * h_t
```

Selective: B, C, dt are INPUT-CONDITIONED (functions of x_t), so the recurrence becomes data-dependent. A is parameterized as `A = -exp(A_log)` to keep eigenvalues negative (decay).

### SSD parallel-scan (Mamba-2 §4)

The selective-state-space recurrence has a Structured State-Space Duality (SSD) form that allows the per-pair-sequential update to be reformulated as a chunk-parallel scan:
```
[h_0, h_1, ..., h_K] = scan(B_bars, A_bars, x_inners)
```
This is O(log K) parallel time on CUDA (via the canonical `selective_scan_fn` kernel), vs O(K) sequential time. On 600-pair sequences with K=64 chunk size, SSD scan is ~5-10× faster than sequential autoregress.

### Substrate-specific exploitation

The fresh substrate's `training_curriculum.py` uses chunk-parallel SSD-scan during training (per CC-G unwind), and the runtime inflate uses sequential (per HNeRV parity L4 ≤200 LOC budget + CUDA/CPU agnostic constraint). The byte-stable invariant: chunk-parallel scan + sequential unroll produce IDENTICAL hidden states (this is the SSD theorem).

### Temporal-conv decoder pre-stage (CC-A unwind)

The fresh substrate's decoder takes the (num_pairs=600, latent_dim=32) latent stream and applies:
```
z_temporal = Conv1D(d_conv=4, in_channels=32, out_channels=32)(z_stream)  # along pair-axis
z_per_pair = z_temporal[t]  # for each pair t
rgb_0, rgb_1 = SpatialDecoder(z_per_pair)  # PixelShuffle stack matching Z6 spatial path
```

The Conv1D pre-stage matches Mamba-2's `d_conv=4` selective-state-space temporal window, so the decoder consumes per-pair latents WITH the temporal context Mamba-2 produced (not in isolation as Z6 decoder does).

### MLX-implementation roadmap

Per the binding directive #1 (MLX-first):
- `architecture.py` is MLX-native; reference PyTorch implementation lives in `_mamba2_reference_torch.py` for byte-stable export bridge per predecessor's state_dict-key-parity work.
- `training_curriculum.py` uses MLX optimizer + MLX dataloader for $0 macOS iteration.
- `archive.py` is MLX-native pack/unpack with PyTorch-compatible state_dict export for byte-stable handoff.
- `inflate_runtime.py` is PyTorch-only (HNeRV parity L4 + canonical `select_inflate_device`).
- Gate at `tools/gate_mlx_candidate_contest_equivalence.py` MUST be invoked BEFORE any paid CUDA dispatch consideration.

## 5. Sister coordination per Catalog #230 (verified)

- Sister A (`subagent_a_dreamer_v3_rssm_20260526T065116Z_10444`): building `src/tac/substrates/dreamer_v3_rssm/` — DISJOINT
- Sister D (`af6ca73c5a7fc40f4` / `lane_z6_predictive_coding_mlx_scaffold_20260526`): building `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` — DISJOINT
- Sister E (`a35f9f86781aaaa4f` / `lane_path_3_e_boost_nerv_against_pr110_20260526`): building `src/tac/substrates/boost_nerv_pr110_residual/` — DISJOINT
- Sister C' (`path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526`): cargo-cult-first at `src/tac/substrates/nscs06_v8_chroma_lut/` — DISJOINT
- Sister `z7_mamba2_mlx_scaffold_ext_20260526`: building `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py` — DISJOINT (NEW substrate dir is `z7_mamba2_v2_fresh_substrate/`)

**Net: 0 file overlap.** All my Phase 3 work goes to NEW substrate dir `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` + NEW `.omx/research/*.md` memos.

## 6. 6-hook wire-in declaration per Catalog #125 (Phase 2 deliverable)

1. Sensitivity-map: N/A at Phase 2 (decision-only)
2. Pareto constraint: ACTIVE — 4-axis decomposition from Phase 1 §5 IS the Pareto polytope shape
3. Bit-allocator hook: N/A at Phase 2 (Phase 3 archive grammar redesign declares)
4. Cathedral autopilot dispatch: N/A at Phase 2 (Phase 3 L0 SCAFFOLD design memo declares)
5. Continual-learning posterior: ACTIVE — anchor will be appended via `tac.council_continual_learning.append_council_anchor` upon commit
6. Probe-disambiguator: ACTIVE — the design-decision IS the canonical disambiguator between Path (a) / (b) / (c)

## 7. Exit criteria

- ✓ Path (a)/(b)/(c) explicitly chosen with binding justification (Path (c))
- ✓ Per-layer canonical-vs-unique decision table (16 layers itemized)
- ✓ Predicted ΔS band per Catalog #296 with Dykstra-feasibility intersection (referenced from Phase 1 §5)
- ✓ horizon_class declaration per Catalog #309 (frontier_pursuit; could upgrade to asymptotic_pursuit if P90 realizes)
- ✓ MLX-implementation roadmap per binding directive #1
- ✓ Mamba-2 substrate math summary
- ✓ Sister coordination per Catalog #230 (0 file overlap)
- ✓ 6-hook wire-in declaration per Catalog #125
- → Phase 3 L0 SCAFFOLD binding directives (see §3)

## 8. Cross-references

- Phase 1 input: `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Predecessor stability memo: `.omx/research/z7_mamba_2_stability_design_space_20260518.md`
- 2026-05-18 design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Canonical operating mode: CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
- Canonical classification: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- CLAUDE.md non-negotiables: "HNeRV / leaderboard-implementation parity discipline" L7 substrate-engineering split + "Forbidden premature KILL"
