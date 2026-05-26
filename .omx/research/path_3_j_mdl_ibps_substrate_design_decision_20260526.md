---
schema: per_substrate_design_decision_v1
deliberation_id: path_3_j_mdl_ibps_substrate_design_decision_20260526
topic: "Path 3 candidate J — substrate-design decision (Phase 2 of 3) per binding operator directive 2026-05-26 #1 'substrate-design + curriculum + optimize whole stack around it' driven by Phase 1 cargo-cult audit findings"
review_kind: per_substrate_design_decision_T2
review_date: "2026-05-26"
lane_id: lane_path_3_j_mdl_ibps_information_bottleneck_cargo_cult_first_20260526
substrate_id: path_3_j_mdl_ibps
substrate_alias: mdl_ibps_j
parent_substrate_id: c6_e4_mdl_ibps
deferred_substrate_id: path_3_j_mdl_ibps
horizon_class: frontier_pursuit
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - MacKay        # MDL framework canonical (PRIMARY)
  - Tishby        # IB framework canonical (memorial)
  - Zaslavsky     # active Tishby-lineage
  - Higgins-memorial  # β-VAE β-tuning canonical
  - Belghazi-memorial  # MINE MI-estimation canonical
  - Hafner        # DreamerV3 RSSM categorical posterior canonical
  - Contrarian
  - Assumption-Adversary
  - PR95Author    # contest-experience canonical
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
predicted_band_validation_status: pending_post_training
predicted_band: null
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: true
dispatch_enabled: false
related_deliberation_ids:
  - path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526
  - council_t3_cargo_cult_resurrection_c6_ibps_v2_20260519
  - c6_ibps_post_training_tier_c_remeasurement_landed_20260519
catalog_anchors: [290, 296, 305, 309, 324, 325, 307, 308, 110, 113, 229, 287, 292, 303]
mission_contribution: frontier_breaking_enabler
---

# Path 3 J=MDL-IBPS — Substrate-design decision (Phase 2 of 3)

**Predecessor:** Phase 1 cargo-cult audit `.omx/research/path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526.md` (11 CCs classified; CC-J-2 + CC-J-3 + CC-J-5 + CC-J-6 enumerate substrate-design decision space).

**This memo's mandate:** per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + operator binding directive 2026-05-26 #1 (substrate-design + curriculum + optimize whole stack), choose substrate-design path (a)/(b)/(c) per Catalog #290 with explicit justification from Phase 1 unwind paths.

---

## ## Operating-within assumption-statement (per Catalog #292)

**The shared assumption I am operating within for this design:** *"The substrate-design optimal point in J's design space is structurally DISTINCT from sister F=Z8 (hierarchical IB binding) and sister A=DreamerV3 (categorical posterior); J occupies a NICHE in the cargo-cult-resurrection design space that combines (i) MINE-based tight MI lower bound for the I(z; frames) regularizer with (ii) HYBRID procedural-coord-MLP-decoder with per-pair sparse-IB-regularized residual modulation."*

HARD-EARNED basis: CC-J-2 unwind enumerates {(a) higher-dim continuous + MINE; (b) discrete categorical; (c) sparse-Laplacian; (d) hierarchical+discrete hybrid}; sister A=DreamerV3 LANDED carries (b); sister F=Z8 LANDED carries hierarchical primitive; J's distinctness emerges from combining (a) MINE + (c) sparse-Laplacian + hybrid procedural decoder. CC-J-4 unwind cites Belghazi 2018 MINE as the canonical tight-MI estimator; CC-J-5 unwind cites K=COIN++ procedural+FiLM as the hybrid sister. CC-J-3 + CC-J-6 unwinds fully address β-tuning + decoder-resolution mechanisms.

CARGO-CULTED component (Assumption-Adversary): the substrate-design FROM-FIRST-PRINCIPLES approach RISKS producing a kitchen_sink (PR105 anti-pattern 1776 LOC LOST to rem2's 241 LOC silver per CLAUDE.md "Race-mode rigor inversion"). The mitigation: explicit LOC budget per HNeRV parity L7 (substrate-engineering exceeds bolt-on cap with rationale) + 30-second-reviewable per file discipline. J's TOTAL substrate LOC budget ≤900 (per L7 substrate-engineering allowance comparable to F=Z8 28.4K renderer + 7K archive + 7K inflate + 12K __init__ scaffolds).

---

## ## Path decision

### Considered paths

Per Catalog #290 canonical-vs-unique decision framework, J considers FOUR distinct substrate-design paths:

**Path (a) — DISCRETE-CATEGORICAL-MINE-HYBRID** (CHOSEN; see below for rationale)
- Reducer: discrete categorical posterior at K=16 per group × G=12 groups = log2(16^12) = 48 bits per latent sample
- MI estimator: MINE-based tight lower bound on I(z; frames) replacing variational KL upper bound (CC-J-4 unwind)
- Decoder: HYBRID procedural-coord-MLP + per-pair categorical-index modulation (FiLM-style; sister K=COIN++ pattern but with discrete categorical not continuous FiLM)
- Output resolution: FULL 384×512 (CC-J-6 unwind addresses Nyquist+SegNet-boundary-frequency-content)
- β-tuning: empirical sweep `{1e-5, 1e-4, 1e-3, 1e-2}` per CC-J-3 + Higgins-memorial verdict (3 orders of magnitude below C6 v1 0.01)
- Bit budget: per-pair ~6 bytes (12 groups × 4 bits = 48 bits = 6 bytes) × 600 pairs = ~3.6 KB total per-pair latent
- Decoder bytes: ~50 KB (procedural coord-MLP base + FiLM modulation matrices)
- Total archive target: ~55-65 KB (substantially smaller than C6 v1's ~225 KB anchor)

**Path (b) — HIGHER-DIM-CONTINUOUS-WITH-MINE** (ALTERNATIVE; deferred)
- Reducer: continuous Gaussian posterior at d_z=64 (vs C6 v1's 24); more capacity
- MI estimator: MINE-based tight lower bound (CC-J-4 unwind)
- Distinct from path (a) by retaining continuous posterior; sacrifices interpretability per Catalog #273-#278 Rudin-Daubechies
- Reason for deferral: continuous posterior has the SAME collapse mechanism that doomed C6 v1; even with tighter MI estimator, the per-sample bit capacity at d_z=64 (~64 × log2(σ²/σ²_q) ≤ ~100 bits) is still BELOW the per-pixel SegNet I(X;Y) ceiling
- Reactivation: if path (a) shows MINE-vs-KL gap is the dominant drag, path (b) becomes EV-positive

**Path (c) — SPARSE-LAPLACIAN-PRIOR** (Path B5 sister; deferred)
- Reducer: continuous Gaussian posterior with SPARSE-Laplacian prior (vs C6 v1's N(0, I) standard normal)
- Encourages dimension-wise sparsity → effective d_z ≪ nominal d_z
- MacKay-canonical extension per Path B5 in T3 v2 symposium
- Reason for deferral: sparsity discipline reduces realized d_z below the bit-budget requirement; addresses CC-J-2 only weakly

**Path (d) — HIERARCHICAL+DISCRETE HYBRID (B4 Ballard-canonical)** (deferred to sister F=Z8 territory)
- B1 hierarchical IB + B2 RSSM categorical posterior combined
- Reason for deferral: sister F=Z8 LANDED commit 5ff5d2ab9 already operationalizes hierarchical binding via canonical quadruple; J duplicating would violate Catalog #230 sister-subagent ownership map and risk Catalog #302 sister-subagent-scope-overlap

### CHOSEN: Path (a) — DISCRETE-CATEGORICAL-MINE-HYBRID

**Justification per Catalog #290:**

1. **DISTINCTNESS from sisters** — explicit FORK from A=DreamerV3 (which uses K=256 × G=24 = 192 bits/sample at the substrate latent surface; J uses K=16 × G=12 = 48 bits/sample, ~4× tighter bottleneck targeting MDL-optimal point); FORK from F=Z8 (which uses hierarchical Rao-Ballard quadruple binding; J uses single-scale categorical+MINE+sparse with procedural decoder); FORK from K=COIN++ (which uses continuous FiLM modulation; J uses DISCRETE CATEGORICAL modulation indices for both quantization-friendliness and interpretability per Rudin-Daubechies discipline).

2. **UNIQUE-AND-COMPLETE-PER-METHOD bind** — J binds (i) MINE tight-MI estimator + (ii) DISCRETE categorical posterior + (iii) HYBRID procedural-coord-MLP-decoder + (iv) full-resolution 384×512 decoder + (v) empirical β-sweep + (vi) sparse-Laplacian regularizer on FiLM modulation matrices into ONE substrate; no sister substrate carries this combination.

3. **OPTIMIZE WHOLE STACK around the substrate** — J's curriculum design (Phase 3) optimizes ALL of: scorer-aware loss routing (canonical Catalog #164 helper); β-sweep schedule (operator-tunable); decoder FiLM-modulation initialization (random orthogonal); per-pair categorical-index posterior initialization (random uniform → softens to learned distribution via Gumbel-Softmax); MINE-network training schedule (alternating with substrate parameters per Belghazi 2018 protocol); EMA shadow + eval_roundtrip + canonical scorer-preprocess routing — ALL canonical-preserved per Phase 1 HARD-EARNED classifications CC-J-7/8/9.

4. **Cargo-cult unwind COMPLETENESS** — Path (a) explicitly addresses ALL Phase 1 unwinds: CC-J-1 (no random-init Tier-C extrapolation; recipe `pending_post_training`); CC-J-2 (discrete categorical replaces continuous Gaussian); CC-J-3 (β-sweep empirical, not 0.01 default); CC-J-4 (MINE replaces variational KL upper bound); CC-J-5 (HYBRID procedural+per-pair-modulation replaces purely procedural OR purely content-adaptive); CC-J-6 (full 384×512 decoder, no bilinear blur); CC-J-7/8/9 (preserve canonical eval_roundtrip + EMA + Modal A10G); CC-J-10 (MLX-first scaffold); CC-J-11 (numpy sister reference).

5. **30-second per-file reviewable** — Path (a)'s 7-file L0 SCAFFOLD (mlx_renderer.py + numpy_reference.py + archive.py + inflate.py + __init__.py + tests/test_basic.py + ib_loss_mine.py) MUST each be ≤300 LOC per HNeRV parity L4 + L12 discipline.

6. **Predicted ΔS band reactivation** — predicted_band remains NULL in this Phase 2 decision; Phase 3 L0 SCAFFOLD recipe declares `predicted_band_validation_status: pending_post_training` per Catalog #324; any band claim deferred to post-smoke Tier-C anchor.

---

## ## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton (`tac.substrates._shared.trainer_skeleton.device_or_die` + canonical TF32/CUDA discipline) | ADOPT canonical | CC-J-7/8/9 HARD-EARNED |
| Scorer loss helper (`tac.substrates._shared.score_aware_common.score_pair_components`) | ADOPT canonical | Catalog #164 differentiability invariant; HARD-EARNED |
| eval_roundtrip (`tac.differentiable_eval_roundtrip`) | ADOPT canonical | CC-J-7 HARD-EARNED; CLAUDE.md non-negotiable |
| EMA shadow (`tac.training.EMA` decay 0.997) | ADOPT canonical | CC-J-8 HARD-EARNED; CLAUDE.md non-negotiable |
| Canonical scorer-preprocess routing | ADOPT canonical | Catalog #164 |
| Modal A10G `min_smoke_gpu` | ADOPT canonical | CC-J-9 HARD-EARNED |
| Inflate runtime (`select_inflate_device`) | ADOPT canonical | Catalog #205 |
| Auth eval helper (`gate_auth_eval_call`) | ADOPT canonical | Catalog #226 |
| Hardware substrate detection (`detect_hardware_substrate`) | ADOPT canonical | Catalog #190 |
| **MINE network for tight I(z; frames) lower bound** | **UNIQUE** | CC-J-4 unwind; Belghazi 2018; not in any canonical helper; ~80 LOC dedicated network + training schedule |
| **Discrete categorical posterior (K=16 × G=12)** | **UNIQUE FORK from sister A=DreamerV3 (K=256 × G=24)** | CC-J-2 unwind; smaller categorical alphabet matched to per-pair bit-budget target ~6 bytes |
| **HYBRID procedural-coord-MLP base + discrete categorical FiLM modulation** | **UNIQUE FORK from sister K=COIN++ (continuous FiLM) + UNIQUE FORK from C6 v1 (per-pair latent)** | CC-J-5 unwind; combines bit-efficiency of procedural with content-adaptivity of per-pair modulation |
| **Full 384×512 decoder output** | **UNIQUE FORK from C6 v1 (48×64 + bilinear upsample)** | CC-J-6 unwind; addresses Nyquist + SegNet-boundary-frequency-content |
| **Empirical β-sweep `{1e-5, 1e-4, 1e-3, 1e-2}`** | **UNIQUE FORK from C6 v1 (β=0.01 default)** | CC-J-3 unwind; Higgins-memorial verdict |
| **Sparse-Laplacian regularizer on FiLM modulation matrices** | **UNIQUE** | Path B5 MacKay-canonical influence; encourages categorical-index sparsity; ~50 LOC |
| Archive grammar (MDLIBPS-J1) | UNIQUE | Substrate-engineering; new grammar for categorical-index per-pair modulation + MINE-network state |
| MLX renderer (`mlx_renderer.py`) | UNIQUE per AMENDMENT #3 axis 2 | Sister coin_pp/z8 patterns; J's specific architecture |
| Numpy reference (`numpy_reference.py`) | UNIQUE per AMENDMENT #3 axis 3 | Sister coin_pp pattern; J's specific architecture |

**Bolt-on vs substrate-engineering split per HNeRV parity L7:** J is **substrate-engineering** (NEW architecture class composing MINE + discrete categorical + procedural-FiLM-hybrid + sparse-Laplacian regularizer). LOC budget exceeds bolt-on cap explicitly. The 6 UNIQUE / UNIQUE FORK decisions ARE the substrate-optimal engineering surface.

---

## ## Curriculum design (optimized stack around the substrate)

Per binding operator directive 2026-05-26 #1 verbatim: *"design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"*.

J's full-stack curriculum:

### Stage 0: MLX-first L0 SCAFFOLD smoke (CPU; free; Phase 3)
- MLX renderer + numpy reference parity test (axis 2 + axis 3 discipline)
- MLX↔PyTorch parity max_abs measurement per channel + per layer per Catalog #1265 gate threshold 0.001
- Byte-deterministic archive grammar round-trip test per Catalog #146 + #220 + #139

### Stage 1: MLX smoke trainer (CPU on local macOS Apple Silicon; ~10-30 min wall-clock; $0)
- 100ep MLX-side training on synthetic 8-pair subset for shape validation
- `_full_main raises NotImplementedError` per Catalog #240 (c) opt-out pre-substrate-engineering
- Smoke result tagged `[macOS-MLX research-signal]` per Catalog #192/#317

### Stage 2: MLX-gate validation (CPU; ~5-10 min wall-clock; $0)
- `tools/gate_mlx_candidate_contest_equivalence.py --substrate path_3_j_mdl_ibps` threshold 0.001
- PASS verdict required BEFORE Stage 3 paid CUDA dispatch authorization

### Stage 3: Modal A10G β-sweep smoke (50ep; $5-15 per arm × 4 arms = $20-60 total)
- β sweep across `{1e-5, 1e-4, 1e-3, 1e-2}` (4 arms; canonical Higgins 2017 method)
- Each arm: 50ep smoke with FULL substrate (mlx-trained checkpoint exported to PyTorch via canonical helper)
- Per-arm artifact: contest-CUDA + paired contest-CPU auth eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- Per-arm post-training Tier-C density re-measurement per Catalog #324

### Stage 4: Post-training Tier-C ablation (CPU; ~5-10 min wall-clock; $0)
- Run `tools/mdl_scorer_conditional_ablation.py --archive <best-β-arm-archive> --tier c` per Catalog #324
- VERDICT: ACROSS_CLASS (density < 0.30) → J achieves substrate-class shift → Stage 5 full dispatch authorized
- VERDICT: WITHIN_CLASS (density ≥ 0.70) → J fails class-shift; Catalog #307 implementation-level falsification recorded; sister substrate path re-evaluation per Catalog #308 ≥3 alternative methodologies

### Stage 5: Modal A10G 500-1000ep full dispatch (deferred until Stage 4 ACROSS_CLASS verdict)
- Per-substrate symposium per Catalog #325 re-affirmation
- Operator-frontier-override per Catalog #199 if dispatch authorized
- Predicted-band declaration per Catalog #296 Dykstra-feasibility OR Shannon R(D) bound OR probe-disambiguator

### Stage 6: Contest-CPU paired auth eval (Linux x86_64; CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable)
- After Stage 5 paid CUDA dispatch lands archive, run contest-CPU auth eval on EXACT archive bytes
- Promotion language reserved until BOTH `[contest-CUDA]` AND `[contest-CPU]` axes anchored

---

## ## 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Path (a) DISCRETE-CATEGORICAL-MINE-HYBRID evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | Path (a)'s discrete categorical + MINE + sparse-Laplacian combination targets ACROSS_CLASS Tier-C verdict; reactivation criterion at Stage 4 |
| 2 | BEAUTY+ELEGANCE (30-sec reviewable) | 7-file scaffold each ≤300 LOC; HNeRV parity L4 + L12 |
| 3 | DISTINCTNESS | FORK from A=DreamerV3 (K=256 not K=16; 192 bits vs 48 bits); FORK from F=Z8 (hierarchical not single-scale); FORK from K=COIN++ (continuous FiLM not categorical); FORK from C6 v1 (continuous Gaussian not categorical, MINE not KL, full-res not 48×64) |
| 4 | RIGOR (premise + adversarial + assumption + empirical) | Phase 1 audit (11 CCs); Phase 2 (this memo); Phase 3 L0 SCAFFOLD + empirical anchor |
| 5 | OPTIMIZATION-PER-TECHNIQUE | Canonical helpers PRESERVED (CC-J-7/8/9); UNIQUE primitives substrate-optimal-engineered (CC-J-1/2/3/4/5/6/10/11) |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Per Catalog #319 deliverability proof: J's archive bytes potentially compose with sister substrates via Wyner-Ziv pipeline-stage per sister M=Wyner-Ziv candidate brief; orthogonality TBD post-Stage-3 empirical |
| 7 | DETERMINISTIC-REPRODUCIBILITY | Archive grammar MDLIBPS-J1 byte-deterministic; seed pinning; canonical Catalog #146 |
| 8 | EXTREME-OPTIMIZATION+PERFORMANCE | MLX-first per AMENDMENT #3 axis 2; numpy reference per axis 3; Catalog #1265 gate |
| 9 | OPTIMAL-MINIMAL-CONTEST-SCORE | CC-J-1/2/3/6 unwinds directly address SegNet-collapse mechanism; β-sweep + full-res decoder + MINE-tight-MI; predicted band reactivation criterion via Stage 4 Tier-C |

---

## ## Observability surface (per Catalog #305)

6-facet observability surface for J Path (a):

1. **Per-layer inspection.** Inspect IB encoder per-layer activations; per-pair categorical-index distribution histogram (12 groups × 16 categories = 192 cells); MINE network forward; FiLM-modulated procedural-coord-MLP per-pixel intermediate values. Hooks via `tac.xray.<lens>`.
2. **Per-signal decomposition.** Per-arm β decomposition: (β, d_seg, d_pose, rate, IB_loss_via_MINE, IB_loss_via_KL_upper_bound, MINE_KL_gap, total_score) emitted per Stage 3 smoke arm.
3. **Run-to-run diff.** Byte-deterministic MDLIBPS-J1 archive grammar; `tools/diff_auth_eval_results.py` (planned per observability audit highest-ROI list).
4. **Post-hoc query interface.** `experiments/results/<lane>/observability/<arm>/*.jsonl`; `.omx/state/continual_learning_posterior.jsonl` queryable per `tac.continual_learning.query_*`.
5. **Cite-chain.** Every score row: `(substrate_id=path_3_j_mdl_ibps, commit_sha, modal_call_id, β, K=16, G=12, mine_network_loc=80, seed)` per Catalog #245.
6. **Counterfactual hooks.** Per-section byte-offset addressability for MDLIBPS-J1: header / mine_network_blob / procedural_decoder_blob / per_pair_categorical_indices_blob / film_modulation_matrices_blob / meta_blob. Catalog #139 + #272 + #105 byte-mutation surface; "what if this byte changed?" probes without re-training.

---

## ## NEW 3-axis discipline per AMENDMENT #3

### Axis 1: Math + scientific + engineering rigor per layer (Path (a))

Per Phase 1 audit Axis 1 table (preserved); Path (a)-specific additions:

| Layer | Math | Scientific | Engineering | Classification |
|---|---|---|---|---|
| MINE network | Donsker-Varadhan dual representation; Belghazi 2018 | Poole et al. 2019 enumerated tradeoffs | Standard MLP critic; ~80 LOC | HARD-EARNED (3-axis) |
| Discrete categorical posterior (Gumbel-Softmax) | Jang et al. 2016 Gumbel-Softmax; Maddison et al. 2016 Concrete distribution | Hafner 2024 DreamerV3 categorical posterior; sister A LANDED | Standard reparametrization trick; ~50 LOC | HARD-EARNED (3-axis) |
| Procedural-coord-MLP + FiLM | Perez et al. 2017 FiLM; Mildenhall 2020 NeRF coord-MLP | Sister K=COIN++ canonical pattern (LANDED) | Sister K=COIN++ scaffold; FORK to discrete | HARD-EARNED (3-axis) |
| Sparse-Laplacian regularizer | MacKay 2003 ITILA ch. 28; Olshausen-Field 1996 sparse coding | Path B5 MacKay-canonical T3 v2 symposium verbatim | L1-regularizer + Laplacian prior MLE; ~30 LOC | HARD-EARNED (3-axis) |
| Empirical β-sweep | Higgins 2017 β-VAE empirical-β-tuning | Higgins-memorial T3 v2 symposium verbatim | Loop over β ∈ {1e-5, 1e-4, 1e-3, 1e-2}; 4 arms | HARD-EARNED (3-axis) |

### Axis 2: MLX drift minimization per primitive (Path (a))

Per Phase 1 audit Axis 2 table (preserved); Path (a)-specific additions:

| MLX primitive | Expected drift bound vs PyTorch | Mitigation strategy |
|---|---|---|
| `mx.random.gumbel` (for Gumbel-Softmax reparametrization) | < 1e-5 with explicit fp32 sampling | `mx.random.gumbel(shape, dtype=mx.float32)`; sister A=DreamerV3 pattern |
| `mx.softmax` (categorical posterior) | < 1e-6 with epsilon | Always pass `eps=1e-12`; canonical pattern |
| MINE network critic forward | < 1e-5 with fp32 accumulation | Explicit dtype casts; standard MLP |
| FiLM modulation (scale + shift) | < 1e-6 with broadcasting | Standard MLX broadcasting; no drift expected |
| Procedural coord-MLP per-pixel forward | < 1e-5 with fp32 sinusoidal positional encoding | Sister K=COIN++ pattern; `mx.sin` + `mx.cos` fp32 |
| Sparse-Laplacian L1 reduction | < 1e-6 | Standard `mx.abs(x).sum()` |

Smoke test threshold: 0.001 per Catalog #1265 gate (90× margin over 0.000011 empirical anchor).

### Axis 3: Portability via numpy per primitive (Path (a))

Per Phase 1 audit Axis 3 (preserved); Path (a)'s `numpy_reference.py` includes:

- MINE network forward (numpy linear + ReLU)
- Discrete categorical posterior sampling (numpy `np.random.choice` with softmax probs)
- Procedural coord-MLP forward (numpy linear + sinusoidal positional encoding)
- FiLM modulation (numpy broadcasting)
- Sparse-Laplacian L1 (numpy `np.abs(x).sum()`)
- IB loss via MINE estimator (numpy)
- IB loss via canonical KL upper bound (numpy; for comparison test)

Test parity: MLX↔numpy ≤ 1e-5 per primitive (tighter than MLX↔PyTorch ≤ 0.001 because both use IEEE 754).

---

## ## Horizon-class classification (per Catalog #309)

**horizon_class: frontier_pursuit**

Justification: per Phase 1 audit; Path (a) targets BELOW canonical frontier 0.192051 [contest-CPU]; the DISCRETE-CATEGORICAL+MINE+HYBRID+FULL-RES combination addresses ALL Phase 1 unwinds; predicted band reactivation criterion post-Stage-4 Tier-C ACROSS_CLASS verdict; if ACROSS_CLASS realized, frontier_pursuit band [0.120, 0.180] is plausible; if WITHIN_CLASS, asymptotic_pursuit reclassification per Catalog #307.

---

## ## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

- Stage 4 Tier-C ACROSS_CLASS verdict (density < 0.30) → Stage 5 full dispatch authorized
- Stage 4 Tier-C WITHIN_CLASS verdict (density ≥ 0.70) → Catalog #307 implementation-level falsification recorded; Catalog #308 ≥3 alternative methodologies pre-existing (sister Paths B1/B3/B4/B5/B6 enumerated in T3 v2 symposium); NO KILL of J or IB paradigm
- ANY paid dispatch authorization requires per-substrate symposium per Catalog #325 + operator-frontier-override per Catalog #199

---

## ## Sister coordination per Catalog #230

Per the inventory brief at landing time + this session's sister-checkpoint state:

- A=DreamerV3 RSSM (LANDED) — categorical posterior K=256 × G=24 = 192 bits/sample; J FORKS to K=16 × G=12 = 48 bits/sample
- F=Z8 hierarchical predictive coding (LANDED) — Tishby IB hierarchical binding; J FORKS to single-scale + MINE
- K=COIN++ (in-flight) — continuous FiLM modulation; J FORKS to DISCRETE categorical FiLM
- C6 v1 (PARENT; phantom_random_init recipe) — continuous Gaussian + 24-dim + 48×64 + β=0.01 + variational KL; J FORKS on ALL primary axes

No collision with sister-subagent ownership maps; no Catalog #302 sister-subagent-scope-overlap.

---

## ## Cross-references

- `.omx/research/path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526.md` (Phase 1 audit; CC-J-1 to CC-J-11 enumerated)
- `.omx/research/c6_e4_mdl_ibps_cargo_cult_unwind_design_20260516.md` (predecessor C6 unwind)
- `.omx/research/council_t3_cargo_cult_resurrection_c6_ibps_v2_20260519.md` (T3 v2 symposium; Path B1-B6 enumerated)
- `.omx/research/c6_ibps_post_training_tier_c_remeasurement_landed_20260519.md` (Catalog #324 anchor)
- `src/tac/substrates/coin_pp_implicit_neural_representation/` (sister K=COIN++ scaffold; 3-axis pattern)
- `src/tac/substrates/dreamer_v3_rssm/` (sister A=DreamerV3 RSSM scaffold; categorical pattern)
- Belghazi et al. 2018 "MINE: Mutual Information Neural Estimation"
- Jang et al. 2016 "Categorical Reparameterization with Gumbel-Softmax"
- Higgins et al. 2017 "β-VAE"
- Hafner et al. 2024 "DreamerV3"
- Perez et al. 2017 "FiLM: Visual Reasoning with a General Conditioning Layer"
- MacKay 2003 ch. 28 sparse-Laplacian prior
- Olshausen-Field 1996 sparse coding

---

**Status:** PHASE 2 DESIGN DECISION LANDED 2026-05-26. Path (a) DISCRETE-CATEGORICAL-MINE-HYBRID CHOSEN. Phase 3 L0 SCAFFOLD proceeds inline.

**Operator-routable:** Phase 3 L0 SCAFFOLD lands the substrate package + memos; no operator-decision-required action at this decision boundary; any future paid dispatch requires per-substrate symposium per Catalog #325.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
