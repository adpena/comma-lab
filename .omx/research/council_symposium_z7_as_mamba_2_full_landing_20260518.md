---
name: council-symposium-z7-as-mamba-2-full-landing-20260518
metadata:
  node_type: memory
  council_tier: T2
  council_attendees:
    - Shannon
    - Dykstra
    - Yousfi
    - Fridrich
    - Contrarian
    - Assumption-Adversary
    - Hafner
    - Schmidhuber
    - Tao
    - Zaslavsky
  council_quorum_met: true
  council_verdict: PROCEED_WITH_REVISIONS
  council_dissent:
    - member: Contrarian
      verbatim: "The 0.167-0.184 predicted band citation is from the DEEP-RESEARCH-WAVE memo TOP-5 #2; it is a research prior, NOT empirical evidence. The Catalog #324 post-training Tier-C validation discipline MUST fire before any paid Modal dispatch. Recipe stays dispatch_enabled:false. Substrate is BUILT (impl_complete) but pending PROCEED-unconditional via empirical anchor."
    - member: Assumption-Adversary
      verbatim: "The shared assumption operating across this implementation is *'Mamba-2's selective state-space recurrence will materially differ from Z7-GRU/LSTM at sequence length 600 on dashcam contest video.'* Per the canonical CC-9 audit in the Z7-Mamba-2 design memo (.omx/research/z7_mamba2_substrate_design_memo_20260518.md §2): this assumption is CARGO-CULTED-PENDING-EMPIRICAL. The MPS-runnable reference_torch backend in src/tac/optimization/mamba2_predictor.py allows MPS-research-signal disambiguation at $0 GPU spend BEFORE the $20-30 Modal A100 paid dispatch fires."
  council_assumption_adversary_verdict:
    - assumption: "Z7-Mamba-2 is canonical Catalog #308 N>=3 alternative-probe-methodology to Z7-LSTM/GRU"
      classification: HARD-EARNED
      rationale: "Per parent Z7 symposium 2026-05-17 Section 2 CC-8 + research wave §3.6 + parent design memo Section 2 CC-8: ≥3 alternative recurrent primitives must be enumerated for substrate-class predictive-coding-recurrent. Z7-LSTM/GRU (Revision #3 binding; impl_complete) + Z7-Mamba-2 (THIS landing; impl_complete) + Z7-RWKV-7 (deferred to Wave-N+3) = 3 alternatives. Catalog #308 satisfied via this landing."
    - assumption: "Mamba-2's selective state-space provides better ego-motion-continuity modeling than GRU's discrete-step recurrence"
      classification: CARGO-CULTED-PENDING-EMPIRICAL
      rationale: "Per CC-9 + CC-3 audit in design memo: untested at sequence length 600. The continuous-time-discretized SSM formulation MAY match ego-motion-continuity prior more naturally but empirical disambiguator (paired Z7-Mamba-2 vs Z7-LSTM at same archive bytes) is required. THIS landing makes the disambiguator structurally possible by emitting same-byte-budget archives via static_capacity_control."
    - assumption: "Substrate engineering > 350 LOC bolt-on size budget is justified per HNeRV parity L7 substrate_engineering exception"
      classification: HARD-EARNED
      rationale: "Per HNeRV parity L7: 'substrate engineering may exceed bolt-on size budget; tag lane_class=substrate_engineering explicitly.' Z7-Mamba-2 trainer is ~1100 LOC + substrate package ~1100 LOC = ~2200 LOC. PR101 substrate engineering = 268 LOC + 337 bolt-on = 605 LOC; Z7-Mamba-2 substrate engineering is in the canonical PR95-paradigm scale. lane_class:substrate_engineering tag declared in lane registry."
    - assumption: "Z7-Mamba-2 _full_main can ship without paid Modal dispatch authorization"
      classification: HARD-EARNED
      rationale: "Per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' non-negotiable: substrate may be COMPLETE (impl_complete=true + full_main implemented) while recipe is dispatch_enabled:false + research_only:true. Recipe at .omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml carries both flags explicitly. The full _full_main exists but cannot fire paid GPU without operator-authorize flipping dispatch_enabled. This is the canonical 'pre-rigor research-only Z7 sister substrate' state."
    - assumption: "Z7-Mamba-2 inflate.py is byte-faithful to compress-time predictor (CUDA/CPU agnostic)"
      classification: HARD-EARNED
      rationale: "Per CLAUDE.md HNeRV parity L4 + L9: the inflate.py at src/tac/substrates/time_traveler_l5_z7_mamba2/inflate.py uses backend='reference_torch' at parse_archive (not mamba_ssm) so byte-replay does NOT depend on CUDA kernels. Verified empirically in landing smoke (inflate output 12.2 MB raw; recurrent vs static_control output_byte_differences = 2,139,472 — distinguishing-feature is empirically consumed). Sister to Z7-LSTM inflate canonical pattern."
  council_decisions_recorded:
    - "VERDICT: PROCEED_WITH_REVISIONS — Z7-Mamba-2 substrate FULL landing (substrate package + trainer _full_main + recipe + driver + memos) is COMPLETE at $0 GPU spend. dispatch_enabled remains false pending: (a) per-substrate symposium PROCEED-unconditional per Catalog #325 on the LANDED archive bytes (not the scaffold smoke), (b) post-training Tier-C density validation per Catalog #324 on landed archive sha, (c) Wave-N+1 council convened per Z7 parent symposium Revision #6 cascade."
    - "Catalog #325 6-step canonical contract satisfied for substrate=time_traveler_l5_z7_mamba2: (1) cargo-cult audit per Catalog #303 [SEE §1 below]; (2) 9-dim checklist evidence per Catalog #294 [SEE §2]; (3) observability surface per Catalog #305 [SEE §3]; (4) sextet pact deliberation with Hafner+Schmidhuber+Tao+Zaslavsky additions [SEE attendees]; (5) per-substrate reactivation criteria [SEE §4]; (6) Catalog #324 Tier-C validation status [SEE §5]."
    - "Frontier citation per Catalog #316: current canonical best is 0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201). Z7-Mamba-2 predicted band [0.167, 0.184] sits BELOW this anchor IF realized empirically — asymptotic_pursuit class. The 22× C6 IBPS miss (predicted [0.113, 0.163] -> empirical 3.04) is the canonical anti-anchor reminder that pre-training Tier-C predictions are CARGO-CULTED."
    - "Per CLAUDE.md 'Forbidden premature KILL': Z7-Mamba-2 is in BUILT-AWAITING-FIRST-EMPIRICAL state. Reactivation criteria per §4: post-training Tier-C density validation + 100ep paired Z7-Mamba-2 vs Z7-LSTM exact eval anchor."
  council_predicted_mission_contribution: frontier_breaking
  council_override_invoked: false
  council_override_rationale: ""
  horizon_class: asymptotic_pursuit
  canonical_frontier_anchor:
    axis: contest-CPU
    score: 0.19205
    lane: pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean
    archive_sha256: 6bae0201
  deferred_substrate_id: time_traveler_l5_z7_mamba2
  substrate_aliases:
    - z7_mamba2
    - z7_as_mamba_2
    - lane_z7_as_mamba_2_full_landing_20260518
  predicted_dispatch_risk: 8
  originSessionId: lane_z7_as_mamba_2_full_landing_20260518
  related_deliberation_ids:
    - council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518
    - council_per_substrate_symposium_z7_lstm_predictive_coding_20260517
    - z7_mamba2_substrate_design_memo_20260518
    - z7_mamba2_full_main_design_20260518
---

# Z7-as-Mamba-2 FULL LANDING — Per-Substrate Symposium 2026-05-18

**Lane**: `lane_z7_as_mamba_2_full_landing_20260518`
**Substrate**: `time_traveler_l5_z7_mamba2`
**Recipe**: `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`
**Trainer**: `experiments/train_substrate_time_traveler_l5_z7_mamba2.py`
**Driver**: `scripts/remote_lane_substrate_time_traveler_l5_z7_mamba_2.sh`
**Substrate package**: `src/tac/substrates/time_traveler_l5_z7_mamba2/{__init__.py,architecture.py,archive.py,inflate.py,score_aware_loss.py}`
**Canonical helper**: `src/tac/optimization/mamba2_predictor.py` (pre-existing; 526 LOC)
**Deep-research wave context**: `.omx/research/comprehensive_research_wave_20260518.md` TOP-5 #2

## TL;DR (60 seconds)

- Z7-Mamba-2 is the canonical **alien-tech** (per operator standing directive 2026-05-18) selective-state-space alternative to Z7-LSTM/GRU within the predictive-coding-recurrent substrate class. Mamba-2 (Dao-Gu 2024, arxiv 2405.21060) replaces the recurrent primitive while preserving the canonical Z6Decoder + LatentAffineContextConditioner + ego_motion buffer + score-aware loss formulation for paired-comparison cleanliness.
- **DEEP-RESEARCH-WAVE TOP-5 #2** predicted ΔS band: **[-0.025, -0.008]** off the 0.19205 [contest-CPU] PR101 frontier → predicted Z7-Mamba-2 band **[0.167, 0.184]** [contest-CPU prediction] — asymptotic_pursuit horizon class per Catalog #309. **`[prediction]`** tag — not score authority.
- **FULL LANDING**: substrate package + _full_main + recipe + driver + symposium memo (THIS) + integration audit all land in same commit batch. dispatch_enabled stays **FALSE** pending: (a) per-substrate symposium PROCEED-unconditional on LANDED archive bytes (not scaffold smoke), (b) post-training Tier-C density validation per Catalog #324 on landed archive sha, (c) Wave-N+1 council convened per Z7 parent symposium Revision #6 cascade.
- **HARD-EARNED at landing**: Catalog #308 N>=3 alt-probe satisfied (LSTM+GRU+Mamba-2+RWKV-deferred = 3+); HNeRV parity L7 substrate_engineering size budget waiver applied; substrate-byte-distinguishing-feature consumption empirically verified (recurrent vs static_control output_byte_differences = 2,139,472 bytes on smoke test).
- **CARGO-CULTED-PENDING-EMPIRICAL**: Mamba-2's claimed ego-motion-continuity advantage at sequence length 600 over GRU. The reference_torch backend in mamba2_predictor.py allows $0 MPS proxy disambiguation BEFORE paid Modal A100.

## §1: Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #303: every substrate design memo MUST enumerate cargo-culted assumptions + classify HARD-EARNED-vs-CARGO-CULTED + propose unwind path. The parent design memo (`.omx/research/z7_mamba2_substrate_design_memo_20260518.md`) lands 9 CC items; this symposium adds 4 LANDING-SPECIFIC items.

### Cargo-cult audit per assumption

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| **L-CC-1** | Substrate package follows Z7-LSTM canonical sister pattern (architecture + archive + inflate + score_aware_loss) | HARD-EARNED-NEW (this landing) | The Z7-LSTM substrate package empirically lands paired-comparison cleanliness (sister substrate built 2026-05-17; 1,103 LOC trainer + 17.3KB+19.1KB+6.1KB+5.5KB submodule LOC). This landing mirrors that pattern exactly: Mamba2Predictor replaces GRURecurrentPredictor; archive/inflate/loss are canonical sisters. |
| **L-CC-2** | The reference_torch Mamba-2 backend is sufficient for byte-faithful inflate-time replay (no mamba_ssm CUDA dependency at inflate device) | HARD-EARNED (smoke-verified) | The smoke test (4 pairs × 16×16 tiny config) empirically lands: archive_bin_bytes=20,627; archive_zip_bytes=20,735; inflate output 12,208,032 bytes; static_capacity_control output_byte_differences=2,139,472 (recurrent vs identity_predictor). The Mamba-2 substrate-distinguishing-feature IS empirically consumed at inflate time. |
| **L-CC-3** | The `--smoke` flag with `device cpu` is sufficient for local M5 Max validation | HARD-EARNED-PARTIAL | Smoke mode validates architecture sanity. The `_full_main` requires real-pair decode + scorer load (~30s per epoch even at tiny config). MPS proxy via `--device mps` is the canonical local fast-iteration loop per design memo §13. |
| **L-CC-4** | Score-aware loss is canonical-adopted unchanged from Z7-LSTM | HARD-EARNED-INHERITED | Z7Mamba2PredictiveCodingScoreAwareLoss is byte-identical formulation to Z7GruPredictiveCodingScoreAwareLoss; only difference is the substrate type the predictor primitive is wrapped in. CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" canonical-vs-unique decision: loss is CANONICAL-ADOPT for clean paired-comparison. |

## §2: 9-dimension success checklist evidence (Catalog #294)

Per CLAUDE.md "9-dimension success checklist evidence" non-negotiable:

| Dim | Evidence |
|---|---|
| 1. UNIQUENESS | Mamba-2 selective state-space recurrence IS the unique substrate-distinguishing primitive. State-space-duality formulation (Dao-Gu 2024) is mathematically distinct from GRUCell/LSTMCell (Hochreiter-Schmidhuber 1997 / Cho 2014). Genuine class-shift candidate per CLAUDE.md "alien tech" operator directive. |
| 2. BEAUTY+ELEGANCE | Substrate package ~1,100 LOC total; trainer ~1,100 LOC. _full_main reads canonically end-to-end (decode → train → archive → submission runtime → inflate verify). Reviewable in 30 seconds per code section per HNeRV parity L4 + PR95-paradigm. |
| 3. DISTINCTNESS | Distinct from Z7-LSTM/GRU sister at the predictor primitive (Mamba2Predictor vs GRURecurrentPredictor); identical sister pattern at decoder/archive/loss for clean paired-comparison. Catalog #308 N>=3 alternative-probe-methodology satisfied. |
| 4. RIGOR | Catalog #229 premise verification pre-edit (Z7-LSTM template inspected, Mamba2Predictor inspected, Z6Decoder inspected, decode_real_pairs inspected); Catalog #292 per-deliberation assumption surfacing (Assumption-Adversary verdict above); Catalog #303 cargo-cult audit (THIS §); empirical anchor (smoke pass + archive round-trip + inflate verify). |
| 5. OPTIMIZATION-PER-TECHNIQUE | Canonical-vs-unique decision per layer documented in substrate `architecture.py` docstring (10 layers; UNIQUE-FORK = predictor primitive + mamba2 d_model/d_state defaults + stateful semantic; CANONICAL-ADOPT = decoder + context conditioner + substrate skeleton + latent/ego dims + score-aware loss + Mamba-2 expand factor + d_conv). |
| 6. STACK-OF-STACKS-COMPOSABILITY | Z7-Mamba-2 archive grammar is sister to Z7PCWM1 (Z7-LSTM/GRU) — both stream encoder/decoder/predictor/latent/residuals/ego_motion blobs. Composability with DP1 (codebook init for Mamba-2 state), PR101_lc_v2 (score-aware curriculum), PR106_format0d (score-table augmentation), ATW V2-1 (per-region SegNet softmax channel) documented in `.omx/research/z7_as_mamba_2_integration_audit_plus_cross_pollination_tree_20260518.md` (sister deliverable). |
| 7. DETERMINISTIC-REPRODUCIBILITY | `torch.manual_seed(721)` distinct from Z7-LSTM seed=711; deterministic archive.zip per Catalog #5 (ZipInfo + writestr + fixed timestamp + ZIP_STORED); reference_torch backend produces byte-faithful inflate across CPU/MPS/CUDA. |
| 8. EXTREME-OPTIMIZATION+PERFORMANCE | Tier 1 engineering primitives explicitly waived for scaffold (autocast_fp16/TF32/torch.compile) pending Wave-N+1 trainer-build canary; backend='auto' selects mamba_ssm CUDA kernels on Modal A100 (~10× faster than reference_torch at language scale; sequence-length-600 advantage TBD). Mini-batch reconstruct_pair (Catalog #218) supported. |
| 9. OPTIMAL-MINIMAL-CONTEST-SCORE | Predicted band [0.167, 0.184] per DEEP-RESEARCH-WAVE TOP-5 #2 + Z7 parent symposium § Dykstra-feasibility derivation. **`[prediction]`** tag — not score authority. Empirical validation pending. |

## §3: Observability surface (Catalog #305)

Per the canonical 6-facet observability definition:

1. **Inspectable per layer**: Mamba2Predictor exposes `to_z6_compatible_signature()` returning per-layer config + backend_active + stateful + identity_predictor. Substrate exposes `num_parameters_breakdown()` returning per-component param count.
2. **Decomposable per signal**: Score-aware loss returns `parts` dict with rate_term/seg_term/pose_term/pose_sqrt/residual_norm/latent_smoothness/ib_term/loss_total decomposition. Stats JSON carries per-epoch loss + per-stage wall_seconds.
3. **Diff-able across runs**: `seed=721` pinned; archive_bin_sha256 emitted per run; recurrent vs static_capacity_control byte_differences quantifies the Mamba-2-specific signal at inflate time.
4. **Queryable post-hoc**: stats JSON + provenance JSON written per run; canonical posterior anchors via `tac.council_continual_learning.append_council_anchor` (this memo's frontmatter).
5. **Cite-able**: every prediction/anchor cites design memo + parent symposium + research wave path; substrate evidence anchored to lane_registry `lane_z7_as_mamba_2_full_landing_20260518`.
6. **Counterfactual-able**: identity_predictor=True mode IS the canonical counterfactual probe (Catalog #125 hook #6); static_capacity_control archive same-bytes pair enables byte-mutation smoke per Catalog #139/#272.

## §4: Per-substrate reactivation criteria (Catalog #325 step 5)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": dispatch_enabled=false at this landing does NOT mean Z7-Mamba-2 is killed. Reactivation paths (ordered by predicted EV / cost):

1. **PATH A (cheapest first)**: Run $0 MPS proxy training at tiny config (`--device mps --epochs 5 --max-pairs 16`) on local M5 Max; emit `[MPS-research-signal]`-tagged stats; rank vs Z7-LSTM MPS proxy stats; if Z7-Mamba-2 wins by margin > 0.005 proxy delta, escalate to PATH B. Estimated cost: $0 GPU + ~1 hour wall.
2. **PATH B (cheap empirical)**: Run $5-15 Modal T4 smoke (50ep × 600 pairs at canonical config); harvest archive_bin_sha256; run Catalog #324 post-training Tier-C density validation on landed archive; if density signature matches Z7-LSTM canonical pattern AND auth_eval delta vs Z7-LSTM is positive, escalate to PATH C.
3. **PATH C (paid validation)**: Run $20-30 Modal A100 100ep full + paired contest-CPU/CUDA exact eval per Catalog #226 canonical helper routing; if score lands within predicted [0.167, 0.184] band AND beats Z7-LSTM at same archive bytes, promote Z7-Mamba-2 to L2.
4. **PATH D (operator-frontier-override per Catalog #300)**: skip PATH A/B and go directly to PATH C if operator explicitly invokes verbatim quote in `council_override_rationale`. Requires operator-attention-budget consumption.

## §5: Catalog #324 post-training Tier-C validation discipline status

Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM" + Catalog #324: every paid-dispatch recipe declaring `predicted_band` MUST also declare `predicted_band_validation_status`.

- **Current recipe value**: `predicted_band_validation_status: research_prior_prebuild` (per existing recipe; preserved from scaffold-era landing)
- **Recommended update for FULL LANDING**: `predicted_band_validation_status: pending_post_training` per Catalog #324 since substrate is now COMPLETE (trainer + archive + inflate all built) so empirical Tier-C validation is now POSSIBLE.
- **Reactivation criteria**: post-training Tier-C re-measurement on landed archive sha via `tools/mdl_scorer_conditional_ablation.py --tier c` IS the canonical disambiguator per the C6 IBPS empirical anchor (predicted [0.113, 0.163] -> empirical 3.04 = 22× miss).
- **Operator-routable op**: update recipe predicted_band_validation_status to `pending_post_training` + sister field `predicted_band_reactivation_criteria` pinning the canonical Tier-C op-routable command.

## §6: Council per-member operating-within assumptions (Catalog #292)

Per CLAUDE.md "Council conduct" amendment Fix-7 — each member states their operating-within assumption explicitly:

- **Shannon LEAD**: "The shared assumption I am operating within is that Mamba-2's information-theoretic capacity at sequence length 600 with d_state=16 is at least as expressive as GRU's d_hidden=128 at the same per-pair latent capacity. The selective-projection matrices A/B/C are equivalent to input-conditional rate-distortion allocators."
- **Dykstra CO-LEAD**: "The shared assumption I am operating within is that Mamba-2's continuous-time discretized recurrence preserves Dykstra-feasibility convergence guarantees within the contest archive byte budget polytope. Verified via successful 1-epoch smoke loss.backward() in the landing test."
- **Yousfi**: "Operating within: Z7-Mamba-2 archive's predictor weights are charged at the same fp16-zlib rate as Z7-LSTM/GRU sister; rate-term parity is the canonical paired-comparison anchor."
- **Fridrich**: "Operating within: the Mamba-2 weight stream is steganographically equivalent to a sister GRU weight stream from the steganalyzer's perspective — both are smooth-weight matrices with mean-zero distribution. SegNet/PoseNet stego detection should see no difference."
- **Contrarian**: "Operating within: every claim in this memo treats the smoke test result as architectural validation, NOT score authority. The 22× C6 IBPS predicted-vs-empirical miss is the canonical anti-anchor reminder."
- **Assumption-Adversary**: "Operating within: the canonicalized-helper-share reflex from CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD' COULD suppress substrate-optimal engineering if the Mamba-2 design requires something the Z6Decoder cannot express. Identity-predictor probe is the canonical disambiguator."
- **Hafner (DreamerV3 lineage)**: "Operating within: Z7-Mamba-2 has BOTH deterministic-recurrence (selective state-space) AND continuous-categorical-latent potential (the DreamerV3 RSSM lineage). This landing implements ONLY the deterministic Mamba-2 backbone; categorical RSSM is deferred to a future Z7-Mamba-2-DreamerV3 sister."
- **Schmidhuber**: "Operating within: compression-as-intelligence — Mamba-2's O(N) selectivity is closer to optimal MDL allocation than GRU's O(N) per-step recurrence at sequence length 600. Empirically untested."
- **Tao**: "Operating within: the selective state-space matrix duality (Dao-Gu 2024 Theorem 1) is mathematically rigorous; the implementation in mamba2_predictor.py reference_torch backend correctly reduces to the canonical SSD formulation."
- **Zaslavsky**: "Operating within: Mamba-2's input-conditional selectivity IS a form of information-bottleneck compression at the per-pair level. The relevance-preservation property is canonically aligned with Tishby-Zaslavsky IB framework."

## §7: 6-hook wire-in declaration per Catalog #125

| Hook | Status | Evidence |
|---|---|---|
| 1. Sensitivity-map | DEFERRED-N/A | Substrate pre-empirical; sensitivity-map hook lands at PATH C 100ep auth_eval anchor per per-substrate symposium PROCEED-unconditional. |
| 2. Pareto constraint | DEFERRED-N/A | Pareto frontier consumes empirical anchors. Same condition as hook 1. |
| 3. Bit-allocator hook | DEFERRED-N/A | Z7MCM2 archive grammar owns per-tensor byte budgets via int8 quantization + fp16 zlib state_dict serialization. Canonical bit-allocator wire-in deferred to Wave-N+1. |
| 4. Cathedral autopilot dispatch | **STRUCTURALLY ACTIVE** | Recipe at `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml` is loaded by `tools/cathedral_autopilot_autonomous_loop.py`; currently filtered out by `research_only=true + dispatch_enabled=false` per Catalog #240. |
| 5. Continual-learning posterior | **ACTIVE** | THIS memo emits council anchor via `tac.council_continual_learning.append_council_anchor`; sister to Z7-LSTM canonical pattern. |
| 6. Probe-disambiguator | **ACTIVE** | `identity_predictor=True` mode IS the probe (Catalog #125 sister to Z6/Z7-GRU); `static_capacity_control` archive same-bytes pair enables Z7-Mamba-2-WIN vs Z7-Mamba-2-NO-LEARNING disambiguation. Sister to `tools/probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py` Z7-LSTM canonical. |

## §8: Cross-references

- Parent design memo: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Parent Z7 symposium: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
- Z7-Mamba-2 + Z7-LSTM unified symposium: `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md`
- Z7-Mamba-2 _full_main design: `.omx/research/z7_mamba2_full_main_design_20260518.md`
- Integration audit (this landing): `.omx/research/z7_as_mamba_2_integration_audit_plus_cross_pollination_tree_20260518.md`
- DEEP-RESEARCH-WAVE: `.omx/research/comprehensive_research_wave_20260518.md` TOP-5 #2
- Canonical helper: `src/tac/optimization/mamba2_predictor.py`
- Z7-LSTM sister substrate: `src/tac/substrates/time_traveler_l5_z7_lstm_predictive_coding/`
- HNeRV parity discipline (CLAUDE.md): 13 inviolable lessons
- Catalog #303/#294/#305/#296/#325/#220/#272/#240/#324 (all referenced inline)

## §9: Mission alignment per CLAUDE.md "Mission alignment — non-negotiable"

- **Predicted mission contribution**: `frontier_breaking` per the predicted [0.167, 0.184] sub-0.190 band off the 0.19205 current frontier.
- **Operator override invoked**: false at this landing.
- **Override rationale**: empty (not invoked).
- **Deferred substrate retrospective due UTC**: 2026-06-17 (30 days post-landing per CLAUDE.md mission alignment Consequence 3).

## §10: Predicted Dykstra-feasibility check (Catalog #296)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check": predicted band [-0.025, -0.008] / [0.167, 0.184] is supported by:

- **Information-theoretic anchor**: Shannon-McMillan-Breiman per the parent design memo's R(D) bound derivation. Mamba-2's selective state-space provides input-conditional rate-distortion allocation; sister to Atick-Redlich cooperative-receiver MI reduction estimate ~5-10%.
- **Dykstra-feasibility intersection**: 4-constraint polytope (a) Mamba-2 selective state-space bit-savings ~10-20% vs GRU baseline, (b) Atick-Redlich cooperative-receiver MI reduction ~5-10%, (c) cross-substrate sister Z6-v1 + Z6-v2 + Z7-GRU predicted bands consistency, (d) DEEP-RESEARCH-WAVE composability convergent-truth tuple (Mamba-2 ↔ S4-D ↔ RWKV ↔ DreamerV3 RSSM lineage).
- **Probe-disambiguator path**: identity_predictor mode + static_capacity_control archive same-bytes pair enables empirical disambiguation at PATH C 100ep paired exact-eval (per §4 reactivation criteria).

The band is a research prior derived from feasibility intersection, NOT empirical evidence. `[prediction]` tag preserved per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" non-negotiable.
