<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Atick
  - Redlich
  - Ballard
  - Rao
  - Tishby
  - Zaslavsky
  - Wyner
  - AssumptionAdversary
  - Contrarian
  - PR95Author
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "Phase 3 design memo + scaffold must produce BYTE-MUTATION smoke per Catalog #139/#272 on every section from byte-zero. The cdf_table_blob bug class must NOT recur at a sister section."
council_assumption_adversary_verdict:
  - assumption: "Atick-Redlich SINGLE substrate-optimal anchor with ego-motion FOE projection is the Layer 1 META-unwind binding architectural choice"
    classification: HARD-EARNED
    rationale: "Per Phase 2 design-decision memo + Phase 1 audit Layer 1 META-unwind + Ballard 2007 ego-motion-conditioning + Catalog #311 predictive-coding-substrate-design-has-ego-motion-conditioning + Catalog #310 F-asymptote-class-shift-not-bolt-on. The substrate IS framed as a class-shift (cooperative-receiver loss against ego-motion-conditioning), NOT a bolt-on of WZ-residual to a Z4 skeleton."
council_decisions_recorded:
  - "op-routable #1: L0 scaffold package src/tac/substrates/atw_v2_cooperative_receiver_v2/ with 6 files per Phase 2 binding deliverables"
  - "op-routable #2: MLX renderer + numpy reference + PyTorch parity reference + isolated _training_only.py"
  - "op-routable #3: archive grammar NEW magic ATWv2CR2; NO dead sections; byte-mutation verifiable"
  - "op-routable #4: inflate.py ≤200 LOC per HNeRV parity L4; canonical select_inflate_device per Catalog #205"
  - "op-routable #5: _full_main raises NotImplementedError per Catalog #240(c)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
deferred_substrate_id: atw_v2_cooperative_receiver_v2
related_deliberation_ids:
  - path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526
  - atw_codec_v2_cooperative_receiver_full_stack_design_20260516
  - atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
chosen_path: "(b) JUSTIFIED-EXTEND with explicit FORK on cargo-culted assumptions"
audit_evidence_tag: "[macOS-MLX research-signal]"
---

# Path 3 candidate H — ATW V2 cooperative-receiver Phase 3 substrate design

**Lane:** `lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526` L1
**Subagent:** `path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526`
**Cost:** $0
**Wall-clock:** Phase 3 ~90 min

## 0. Phase 2 input + binding decisions

- Path (b) FORK selected (commit `683878854`)
- Layer 1 META-unwind: Atick-Redlich 1990 as SINGLE substrate-optimal anchor
- Wyner-Ziv DROPPED (preconditions violated)
- Tishby IB DEMOTED to advisory cross-check
- NEW conditioning variable: ego-motion FOE projection (Ballard 2007 + Catalog #311) replaces per-class softmax (D4-falsified)
- NEW package: `src/tac/substrates/atw_v2_cooperative_receiver_v2/` (preserves existing `atw_codec_v2/` per Catalog #110/#113)

## 1. Canonical-vs-unique decision per layer (Catalog #290 NON-NEGOTIABLE)

| Layer | Canonical adoption candidate | Decision | Rationale |
|---|---|---|---|
| Inflate device selector | `tac.substrates._shared.inflate_runtime.select_inflate_device` per Catalog #205 | ADOPT_CANONICAL_BECAUSE_SERVES | Universal across substrates; CPU/CUDA reproducibility |
| Auth-eval routing | `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` per Catalog #226 | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical CLI flags; phantom-score prevention |
| NVML/CUDA env block | `tac.deploy.modal.runtime` canonical 3-export per Catalog #244 | ADOPT_CANONICAL_BECAUSE_SERVES | Sister bug-class extincted at driver surface |
| Scorer preprocess | canonical `score_pair_components` per Catalog #164 / `tac.losses.scorer_loss_terms_btchw` | ADOPT_CANONICAL_BECAUSE_SERVES | Strict-scorer-rule + differentiable-scorer-preprocess |
| Score-aware loss canonical primitive | `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` per Catalog #164 | ADOPT_CANONICAL_BECAUSE_SERVES + EXTEND | Atick-Redlich framing is the binding anchor; extend for ego-motion conditioning |
| Conditioning variable | ATW V2 v1 used per-class softmax (DETERMINISTIC SHARED CONDITIONING reframing) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per Phase 1 CC-7 unwind + CC-1 unwind: per-class softmax falsified by D4 INDEPENDENT (I=6.4e-3); ego-motion FOE projection per Ballard 2007 + Catalog #311 hypothesized to score-improve |
| Encoder | ATW V2 v1 Conv+Linear(3→64→24) | FORK_BECAUSE_SUPPRESSES | hidden_dim=64 cargo-culted from V1 inheritance per CC-6; substrate-optimal capacity TBD; default Phase 3 = medium-capacity HNeRV-style stem (will tune in Wave N+1) |
| Per-pair latent | latent_dim=24 inherited from Z4/Z6 | FORK_BECAUSE_SUPPRESSES | Per CC-6: not measured against cooperative-receiver gradient surface; default Phase 3 = 32-dim (will Pareto-sweep in Wave N+1) |
| Decoder | ATW V2 v1 PixelShuffle×6 (NeRV-style) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per CC-5: PixelShuffle never optimized for cooperative-receiver loss; HNeRV-style (PR101 medal-class proven canonical) is the default Phase 3 choice |
| Archive grammar | ATW V2 v1 ATW2 9-section with dead cdf_table_blob | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per CC-8 EMPIRICALLY FALSIFIED: cdf_table_blob is DECODE-OPAQUE per codex 057130de4; NEW grammar magic `ATWv2CR2` (cooperative-receiver V2); REMOVE cdf_table_blob; ALL remaining sections byte-mutation verifiable from byte-zero |
| Inflate runtime | ATW V2 v1 8.6KB inflate.py | FORK_BECAUSE_PRINCIPLED_MISMATCH | NEW substrate; rewrite from byte-zero; ≤200 LOC per HNeRV L4; pure-numpy + canonical select_inflate_device + canonical-fail-closed-runtime-token per Catalog #146 |
| Training loss | ATW V2 v1 additive triple-Lagrangian (Variant A/B) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per CC-4 unwind: drop additive triple-binding; SINGLE-theorem-anchor Atick-Redlich loss subject to Pareto-feasibility constraints (Dykstra alternating projections, NOT additive scalar Lagrangian) |
| MLX iteration surface | none in V1 (PyTorch-native only) | ADOPT_CANONICAL_BECAUSE_SERVES + EXTEND | Per operator directive #1 MLX-first reframing; canonical pattern from Z6 mlx_renderer.py + pr95_hnerv_mlx + Catalog #1265 contest-equivalence gate |
| Substrate registry | `tac.substrate_registry` per Catalog #241 + #242 | ADOPT_CANONICAL_BECAUSE_SERVES | Universal; canonical 36-field contract |
| Cathedral consumer | `tac.cathedral.consumer_contract` per Catalog #335 | ADOPT_CANONICAL_BECAUSE_SERVES | TIER_A observability-only at L0; TIER_B promotion deferred to Phase 4+ post-D4-equivalent-probe |

## 2. 9-dimension success checklist evidence (Catalog #294 NON-NEGOTIABLE)

| Dim | Evidence at L0 SCAFFOLD |
|---|---|
| **1 UNIQUENESS** | Class-shift substrate per Catalog #310: ego-motion-conditioned cooperative-receiver loss is NOT bolt-on to A1/PR101 (no PR101-derivative substrate uses ego-motion FOE projection as conditioning variable for class-conditional rate term R(D|Y_ego_motion)) |
| **2 BEAUTY+ELEGANCE** | Single-theorem-anchor (Atick-Redlich) + single-conditioning-variable (ego-motion FOE) + 6-file scaffold + ≤200-LOC inflate; reviewable per layer in 30 seconds |
| **3 DISTINCTNESS** | Explicitly distinct from sister NSCS06 v8 chroma_lut (per-class chroma prior for grayscale-LUT substrate; canonical equation #26 IN-DOMAIN) + Z6 (predictive coding via FiLM predictor) + Z7-Mamba-2 (selective-state-space recurrence); this substrate's distinguishing feature is ego-motion conditioning at the rate-axis per R(D|Y) framework |
| **4 RIGOR** | Phase 1 audit (17 CCs across 4 layers; HARD-EARNED-vs-CARGO-CULTED per Catalog #303) + Phase 2 design decision (Path b FORK selection per Catalog #290 falling-rule) + Phase 3 design memo (THIS) + L0 SCAFFOLD with byte-mutation tests from byte-zero per Catalog #139/#272; D4-equivalent probe planned per Phase 4 |
| **5 OPTIMIZATION-PER-TECHNIQUE** | 14 layers + per-layer canonical-vs-unique decision (§1 above); 5 ADOPT + 9 FORK; substrate-optimal engineering per Catalog #290 |
| **6 STACK-OF-STACKS-COMPOSABILITY** | Substrate is single-knob (Atick-Redlich loss); composes with sister substrates as a class-shift candidate in cathedral autopilot rank ordering; not bolt-on |
| **7 DETERMINISTIC REPRODUCIBILITY** | Archive bytes deterministic across reseeds (per Catalog #166 Modal HEAD-parity); MLX→PyTorch state_dict export byte-stable per #1251 bridge; numpy reference bit-exact for FORWARD/INFLATE path per Axis 3 |
| **8 EXTREME OPTIMIZATION+PERFORMANCE** | MLX-first iteration loop per operator directive #1 → $0 macOS iteration before paid CUDA; per pr95 MLX→PyTorch full-decoder downstream-scorer drift ~1e-2 (within Catalog #1265 gate threshold WITH precise-mode mitigations); HNeRV-style decoder canonical-medal-class-proven |
| **9 OPTIMAL MINIMAL CONTEST SCORE** | Predicted ΔS via Dykstra-feasibility framework: rate-axis savings 3-15% from R(D|Y_ego_motion) bound vs unconditional R(D); seg/pose-axis 0-5% from cooperative-receiver loss alignment; total predicted ΔS [TBD-pending-Phase-3-Dykstra-feasibility-computation]; baseline = current PR101 frontier per `.omx/state/canonical_frontier_pointer.json` |

## 3. Cargo-cult audit per assumption (Catalog #303 NON-NEGOTIABLE)

The Phase 3 substrate design honors the Phase 1 audit findings:

- 13 CARGO-CULTED CCs from Phase 1 → unwound per Phase 2 design decision
- 4 HARD-EARNED CCs from Phase 1 → preserved per canonical-share-when-serves
- NEW CCs surfaced in Phase 3 design (this memo):

| # | Assumption | Classification | Unwind |
|---|---|---|---|
| **P3-CC-1** | "Ego-motion FOE projection is INFORMATIVE for per-pair latent conditioning on dashcam video" | **HARD-EARNED-PARTIAL** | Per Ballard 2007 + Catalog #311: ego-motion IS the dominant continuous-time signal in dashcam video (vehicle is moving; everything is parallax-shifting). The PARTIAL part: NOT empirically verified for THIS contest; Phase 4 D4-equivalent probe required BEFORE any paid CUDA. |
| **P3-CC-2** | "Class-conditional rate term R(D|Y_ego_motion) achieves 3-15% rate-axis savings vs unconditional R(D)" | **CARGO-CULTED-PENDING-EMPIRICAL** | Range estimated from Shannon R(D) literature on conditional source coding for video; NOT empirically measured for this substrate. Phase 4 paired smoke with/without ego-motion conditioning at SAME archive bytes. |
| **P3-CC-3** | "HNeRV-style decoder is the substrate-optimal choice for Atick-Redlich cooperative-receiver loss gradient surface" | **CARGO-CULTED-PENDING-EMPIRICAL** | HNeRV is proven canonical for medal-class scores (PR101). NOT empirically measured against Atick-Redlich loss gradient surface specifically. Phase 4 paired smoke vs Cool-Chic + continuous-time Atick decoder. |

## 4. Observability surface (Catalog #305 NON-NEGOTIABLE)

| Facet | Surface |
|---|---|
| 1 Inspectable per layer | MLX renderer exposes per-layer activation hooks via `register_forward_hook` pattern; per-pair latent / per-pair ego-motion / per-pair conditioning embedding all queryable; PyTorch parity reference for cross-check |
| 2 Decomposable per signal | Loss decomposes into (rate term `25·B/N`, seg term `100·d_seg`, pose term `sqrt(10·d_pose)`, cooperative-receiver MI term); per-step JSONL emission |
| 3 Diff-able across runs | Archive byte-deterministic per Catalog #166 + reproducible per `(seed, commit_sha, upstream_snapshot_sha256)` tuple |
| 4 Queryable post-hoc | All artifacts machine-readable JSON; `experiments/results/lane_path_3_h_atw_v2_cooperative_receiver_v2_*/` carries `auth_eval_*.json` + provenance + per-pair-conditioning trace |
| 5 Cite-able | Every artifact carries canonical Provenance per Catalog #323 (call_id / lane_id / archive sha256 / upstream snapshot sha256) |
| 6 Counterfactual-able | Byte-mutation smoke per Catalog #139/#272 on EVERY archive section from byte-zero; tests assert NON-ZERO max_abs_raw_byte_delta per section |

## 5. Dykstra-feasibility predicted-band check (Catalog #296 NON-NEGOTIABLE)

Predicted ΔS band derivation:

- Atick-Redlich cooperative-receiver loss minimizes -I(X; R(decoder(z))) subject to:
  - **Constraint 1**: rate `25·B/37_545_489 ≤ R_max` where R_max = current PR101 frontier rate budget
  - **Constraint 2**: HNeRV parity L1-L13 non-negotiables (eval_roundtrip + EMA + canonical scorer routing + …)
  - **Constraint 3**: archive grammar per Catalog #146 (single ZIP member `0.bin`; ≤200-LOC inflate; pure-numpy fallback)
  - **Constraint 4**: contest formula `S = 100·d_seg + sqrt(10·d_pose) + 25·B/N`

- Dykstra alternating-projections: iterate (project onto C1; project onto C2; project onto C3; project onto C4) to find Pareto vertex
- Predicted ΔS band: pending Phase 4 empirical Dykstra computation; first-principles bound = `[-0.020, -0.005]` ([macOS-MLX research-signal] only; NOT [contest-CUDA] until Phase 4)

## 6. Horizon class declaration (Catalog #309 NON-NEGOTIABLE)

`horizon_class: frontier_pursuit` per Catalog #309. Rationale: predicted ΔS band `[-0.020, -0.005]` falls in the frontier-pursuit zone [0.120, 0.180] when applied to current PR101 frontier (~0.192 baseline + predicted ΔS = ~0.172-0.187). Not asymptotic_pursuit (would require sub-0.120 prediction); not plateau_adjacent (predicted ΔS exceeds plateau noise floor).

## 7. Math + scientific + engineering rigor per layer (per operator directive #3 Axis 1 NEW)

| Layer | Math | Scientific citation | Engineering anchor |
|---|---|---|---|
| MLX encoder | Conv2d → linear projection; FP32 precise math per pr95 MLX drift mitigation | Atick-Redlich 1990 channel-input characterization | Per Z6 mlx_renderer.py canonical pattern + Catalog #1265 contest-equivalence gate |
| Per-pair latent | latent_dim=32 default (Pareto vertex search in Wave N+1) | Schmidhuber 2009 compression-as-intelligence latent-capacity bound | HNeRV parity L1-L13 + Catalog #220 operational mechanism |
| Ego-motion FOE projection | per-pair PoseNet pose-delta dominant-direction projection: `Y = pose_delta / ||pose_delta||` (3-component unit vector for translation; 3-component unit vector for rotation) | Ballard 2007 *"Embodied cognition and visual perception"* + Atick-Redlich 1990 + Catalog #311 ego-motion-conditioning + Catalog #310 F-asymptote-class-shift | NEW `tac.codec.cooperative_receiver.ego_motion_foe_projection` canonical primitive (Phase 3 lands stub; Phase 4 lands operational compute) |
| Conditional rate term | R(D|Y_ego_motion) per Wyner-Cover conditional source coding for known Y | Wyner 1976 (advisory; reframed as DETERMINISTIC SHARED CONDITIONING per Phase 2 CC-1 unwind) + Cover-Thomas 2006 §5.5 | Per-pair class-conditional CDF table (substrate-specific; byte-mutation verifiable per Catalog #139) |
| MLX decoder | HNeRV-style (Conv+Upsample+Conv pattern); PixelShuffle factor 2; precise FP32 | HNeRV (Chen et al. 2023 arXiv:2304.02633) + Atick-Redlich 1990 fixed-receiver framework | Per pr95_hnerv_mlx canonical pattern + Catalog #1265 + PR101 medal-class proven inheritance |
| Scorer-conditioning forward | per-pair ego-motion projection → per-pair conditioning embedding → per-pair rate-adjusted latent | Ballard 2007 ego-motion + Atick-Redlich 1990 cooperative-receiver | NEW (substrate-specific) |
| Cooperative-receiver loss | L_AR = -I(X; R(decoder(z))) per Atick-Redlich 1990 + multi-task variant per Tishby-Zaslavsky 2015 advisory | Atick-Redlich 1990 + Tishby-Zaslavsky 2015 + Schmidhuber 2009 compression-as-intelligence cross-check | Canonical `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` + Catalog #164 + Catalog #297 reversibility probe (signal-axis preservation) |
| Archive grammar (NEW ATWv2CR2 magic) | header + encoder_blob + decoder_blob + cond_embed_blob + ego_motion_proj_blob + per_pair_latent_blob + class_cond_cdf_blob + meta_blob (8 sections; ALL byte-mutation verifiable from byte-zero) | Shannon R(D|Y) conditional source coding bound | Substrate-specific magic; NO dead sections (cdf_table_blob lesson learned per Catalog #220/#272); pure-numpy parser; Catalog #146 contest-compliant |
| Inflate runtime | per-pair latent + ego-motion-conditional decode; pure-numpy fallback; PyTorch for canonical select_inflate_device | Per Catalog #146 contest-compliant template | ≤200 LOC per HNeRV parity L4; canonical Catalog #205 select_inflate_device |

## 8. MLX drift minimization per primitive (per operator directive #3 Axis 2 NEW)

Per pr95 MLX drift research + engineering anchor + downstream-scorer anchor:

| MLX primitive | Expected drift bound vs PyTorch | Mitigation in Phase 3 scaffold | Catalog #1265 gate threshold |
|---|---|---|---|
| `mx.conv2d` (encoder + decoder Conv2d blocks) | 1e-5 to 1e-3 max_abs (FP32 fast-math reassociation) | `precise=True` mode; per-primitive comparison vs torch.conv2d via test_basic.py | Below threshold WITH mitigation |
| `mx.linear` (latent projection, conditioning embedding) | 1e-5 to 1e-3 (FMA reassociation) | `precise=True`; explicit FP32 dtypes; numpy `@` reference verification | Below threshold WITH mitigation |
| `mx.relu` / `mx.maximum` | bit-exact | None needed | Below threshold |
| `mean(axis=...)` (encoder spatial pool + softmax norm) | 1e-4 max_abs (reduction order) | Cast→FP64→reduce→cast-back pattern; verify via numpy reference | Below threshold WITH mitigation |
| `mx.softmax` (rate-conditional CDF normalization) | 1e-4 max_abs (reduction order) | Numerically-stable formulation; explicit axis; numpy reference | Below threshold WITH mitigation |
| `mx.sigmoid` (decoder output) | ~1e-6 | None typically needed | Below threshold |
| Cross-pair conditioning recurrence (if used) | RECURRENCE compounds drift | Compute per-pair independently; do NOT chain hidden state across pairs in L0 scaffold; if Phase 4 chains, periodic resync to numpy reference | Below threshold WITH per-pair independence |
| End-to-end full-decoder + scorer pass | 1e-3 to 1e-2 final score drift per pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526 | Catalog #1265 gate threshold 1e-3 = pass conditional on precise-mode mitigations | Pass WITH mitigations (per #1258 corrected empirical anchor 0.000011 score-drift) |

L0 SCAFFOLD test_basic.py MUST emit: (1) MLX↔PyTorch per-primitive max_abs measurement table; (2) MLX↔numpy reference parity per primitive; (3) MLX↔PyTorch end-to-end full-decoder + scorer pass max_abs measurement; (4) Catalog #1265 gate threshold verification.

## 9. Portability via numpy per primitive (per operator directive #3 Axis 3 NEW)

| MLX primitive | numpy reference | Portability status |
|---|---|---|
| `mx.conv2d` | `scipy.signal.correlate2d` per channel OR FFT-based conv via `np.fft` | Portable; slow on CPU; document non-promotable-on-CPU for production-perf-critical paths |
| `mx.linear` | `numpy @` operator | Bit-exact; trivially portable |
| `mx.relu` / `mx.maximum` | `np.maximum(x, 0)` | Bit-exact; trivially portable |
| `mx.sigmoid` | `1 / (1 + np.exp(-x))` (numerically stable for moderate inputs) | Portable; tolerance for FP overflow at extreme inputs |
| `mx.softmax` | numerically-stable: `e = np.exp(x - x.max(axis, keepdims=True)); e / e.sum(axis, keepdims=True)` | Bit-exact (algorithmically); trivially portable |
| `mean(axis=...)` | `np.mean(axis=axis)` | Bit-exact; trivially portable |
| Per-pair ego-motion projection | numpy vector ops (`np.linalg.norm`, broadcast division) | Bit-exact; trivially portable |
| Cooperative-receiver loss compute (PyTorch only) | NON-PORTABLE | Isolated in `_training_only.py` per documented exception (operator directive #3 acceptance) |
| Inflate runtime (production-critical) | Pure-numpy + PyTorch for canonical `select_inflate_device` | Per Catalog #146 contest-compliant; production-portable to CPU-only systems |
| Archive grammar parser | Pure-numpy + `struct.unpack` | Trivially portable; CPU-only systems supported |

**Decision**: forward path + inflate path FULLY portable to numpy. Training loss path explicitly non-portable (PyTorch only); isolated in `_training_only.py` per operator directive #3 acceptance.

## 10. L0 SCAFFOLD package structure

```
src/tac/substrates/atw_v2_cooperative_receiver_v2/
├── __init__.py                # SPDX + Catalog #241 contract + Catalog #124 8 fields
├── mlx_renderer.py            # MLX-native encoder + decoder + scorer-conditioning forward
├── numpy_reference.py         # numpy reference for FORWARD/INFLATE path (Axis 3)
├── _torch_compat_reference.py # PyTorch parity reference for MLX↔PyTorch drift (Axis 2)
├── _training_only.py          # PyTorch training loss compute (non-portable; isolated)
├── archive.py                 # NEW ATWv2CR2 grammar; NO dead sections
├── inflate.py                 # ≤200 LOC pure-numpy + canonical select_inflate_device
├── registered_substrate.py    # Catalog #241 + #240(c) raises NotImplementedError
└── tests/
    ├── __init__.py
    └── test_basic.py          # Catalog #91 roundtrip + Catalog #139 byte-mutation + MLX↔PyTorch + MLX↔numpy
```

## 11. 6-hook wire-in declaration (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map** (hook #1): NEW `tac.sensitivity_map.atw_v2_cooperative_receiver_ego_motion_conditioning` — STUB at L0 scaffold; consumed by Phase 4 D4-equivalent probe; reports MI(z; Y_ego_motion) per-pair
2. **Pareto constraint** (hook #2): NEW `tac.pareto.atw_v2_cooperative_receiver_ego_motion_conditional_rate` — R(D|Y_ego_motion) constraint replacing V1's R_WZ ≥ H(latent | scorer_class)
3. **Bit-allocator hook** (hook #3): NEW `bit_allocator.atw_v2_cooperative_receiver_ego_motion_v1` — per-pair archive bytes by ego-motion conditioning entropy
4. **Cathedral autopilot dispatch hook** (hook #4): NEW recipe `.omx/operator_authorize_recipes/substrate_atw_v2_cooperative_receiver_v2_modal_t4_dispatch.yaml`; `dispatch_enabled: false` + `research_only: true` at landing per Catalog #240(c); auto-discovered per Catalog #335 dual-tier (TIER_A observability-only)
5. **Continual-learning posterior update** (hook #5): Phase 3 landing memo emits council anchor via `tac.council_continual_learning.append_council_anchor` with `deferred_substrate_id="atw_v2_cooperative_receiver_v2"`
6. **Probe-disambiguator** (hook #6): NEW `tools/probe_atw_v2_cooperative_receiver_ego_motion_conditioning_disambiguator.py` — D4-equivalent for ego-motion conditioning surface; smoke run deferred to Phase 4

## 12. Cross-references

- Phase 1 audit: `path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526.md` (commit `06ea98483`)
- Phase 2 design decision: `path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526.md` (commit `683878854`)
- v1 cdf_table_blob falsification: `atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521.md`
- Sister B' Z7-Mamba-2: `path_3_b_z7_mamba_2_substrate_design_decision_20260526.md`
- Sister C' NSCS06 v8 chroma_lut: `path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md`
- Canonical Z6 mlx_renderer pattern: `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py`
- Canonical pr95_hnerv_mlx pattern: `src/tac/local_acceleration/pr95_hnerv_mlx.py`
- Canonical equation #26 EXCLUDED `direct_byte_substitution_on_decode_opaque_raw_sections`: `.omx/state/canonical_equations_registry.jsonl`

## 13. APPEND-ONLY footer per Catalog #110/#113

This Phase 3 design memo is APPEND-only. The Phase 1 audit + Phase 2 design decision + all predecessor memos are PRESERVED byte-for-byte.

**Mission contribution per Catalog #300**: `frontier_breaking_enabler`

**Lane**: `lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526` L1 (impl_complete: Phase 1 audit + Phase 2 design decision + Phase 3 design memo)
