<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is the canonical L0 SCAFFOLD design record for Path 3 candidate #E (BoostNeRV against PR110 fec6 frontier). DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: residual-rate-vs-distortion math below cites `tac.canonical_equations` registry slots (procedural codebook savings #26 INCLUDED_CONTEXTS does NOT cover residual-hybrid; sister equation `procedural_predictor_plus_residual_correction_savings_v1` deferred per Catalog #359 anchor). FORMALIZATION_PENDING:residual_hybrid_boosting_savings_v1_design_proposal_per_l0_scaffold_pending_phase_2_council_symposium_per_catalog_325 -->
---
council_tier: T1
council_attendees: [Shannon, PR95Author, Time-Traveler]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "PR110 fec6 archive bytes can be preserved unchanged while ADDING boosting residual sidecar"
    classification: HARD-EARNED
    rationale: "PR110 archive grammar already supports composable sidecars via the canonical 0.bin monolithic-single-file pattern + brotli sidechannel precedent established by PR106 latent_sidecar / PR101 split_brotli. The sidecar route adds bytes additively to the rate term (Δrate = 25 × Δbytes / 37545489) while the per-pair distortion reduction nets to negative ΔS if the residual learner extracts non-trivial signal."
  - assumption: "MLX iterative boosting curriculum converges to a residual learner that empirically lowers d_seg + d_pose on PR110 base frames"
    classification: CARGO-CULTED
    rationale: "untested. The mathematical premise is well-established (gradient boosting reduces worst-case residual error on the training distribution); empirical question is whether the residual carries score-relevant signal AT PR110's specific operating point (0.193 [contest-CPU] gold band). Catalog #303 cargo-cult audit + Catalog #324 post-training Tier-C validation required at L1 dispatch."
  - assumption: "shared latent z_pr110 (extracted from PR110 archive) is sufficient conditioning for the residual learner; per-pair latent z_residual would inflate sidecar rate ~2x"
    classification: CARGO-CULTED
    rationale: "Cheap variant. Per-pair z_residual is the alternative; empirical sweep at L1 to disambiguate."
  - assumption: "1 boosting round is sufficient for L0 SCAFFOLD; >1 rounds enter diminishing-returns territory and exacerbate rate term"
    classification: CARGO-CULTED
    rationale: "L0 sanity choice. Boosting literature suggests 2-3 rounds is the sweet spot; >4 rounds usually over-fits. Sweep at L1: 1/2/3 rounds."
council_decisions_recorded:
  - "L0 SCAFFOLD: substrate package + MLX trainer skeleton + design memo; full path council-gated per Catalog #240; _full_main raises NotImplementedError; research_only=true; dispatch_enabled=false"
  - "MLX-first per Catalog #1265: all training paths in MLX; export to PyTorch state_dict via #1251 bridge; package to contest archive via #1257; verify via #1265 gate (threshold 0.001 contest-units) BEFORE any paid CUDA dispatch is authorized"
  - "Operator-routable #1: Phase 2 council symposium per Catalog #325 (cargo-cult audit + 9-dim checklist + Dykstra-feasibility predicted band + observability surface + per-substrate-reactivation criteria pinned + Tier-C validation discipline) BEFORE any paid dispatch"
  - "Operator-routable #2: derive Shannon R(D) bound for residual learner conditioned on PR110 base reconstruction (Atick-Redlich cooperative-receiver framing per Z4 sister substrate)"
  - "Operator-routable #3: empirical 100ep MLX smoke convergence verdict on `upstream/videos/0.mkv` per CLAUDE.md eval_roundtrip + score-aware-loss non-negotiables; produces [macOS-MLX research-signal] anchor for cathedral autopilot ranker consumption per Catalog #341 Tier A"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
horizon_class: frontier_pursuit
related_deliberation_ids:
  - boost_nerv_l0_scaffold_design_20260520T184500Z
  - mlx_candidate_contest_equivalence_gate_landed_20260526
  - pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526
  - pr95_mlx_full_inflate_parity_closure_landed_20260526
---

# Path 3 candidate #E: BoostNeRV against PR110 fec6 frontier — L0 SCAFFOLD design memo

**Lane**: `lane_path_3_e_boost_nerv_against_pr110_20260526` L0/L1
**Operator directive**: *"We should add boostnerv to the priority list too, maybe against PR110, because it seems like it could be free gains if done right."*
**Strategic reframing (binding 2026-05-26)**: *"design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"* — substrate-FOR-method, NOT bolt-on-on-existing-scaffold.
**Status**: L0 SCAFFOLD (`research_only=true` / `dispatch_enabled=false` / `_full_main raises NotImplementedError` per Catalog #240).
**Cost**: $0 (design + MLX scaffold only).

## Why this is a SISTER substrate to the existing `boost_nerv/`, not an extension

The existing `src/tac/substrates/boost_nerv/` (commit lane `lane_boost_nerv_l0_scaffold_20260520`, 2026-05-20) is a GENERIC boosting-NeRV architecture: it trains a DepthSep base decoder from scratch + NUM_BOOSTING_ROUNDS=2 residual rounds. It does NOT take PR110 as input; it does NOT preserve any contest-frontier base archive; it does NOT compose with the fec6+fixed_huffman+k16 codec.

The new substrate `boost_nerv_pr110_residual/` is fundamentally different per the binding 2026-05-26 reframing:

1. **Base learner is FROZEN and EXTERNAL** — PR110's HNeRV decoder + fec6 selector + fixed-Huffman k=16 codec is the base. We do NOT retrain it; we do NOT redesign it; we treat it as a black-box that produces (frame_0, frame_1) reconstructions per pair.
2. **Residual learner is the substrate** — a lightweight MLX-implementable residual codec that adds ADDITIVE per-pair residual bytes to the archive and recovers a per-pair RGB correction at inflate time.
3. **The "against PR110" framing means**: BoostNeRV is the canonical iterative residual-refinement paradigm specialized to PR110's specific reconstruction error distribution. The boosting curriculum freezes PR110, computes per-pair residual targets (GT − PR110_reconstruction), trains the residual learner to predict them, packages the residual into a brotli-compressed sidecar, and the inflate runtime invokes PR110's inflate FIRST then adds the residual on top.

This is what the operator's framing structurally REQUIRES: **substrate-FOR-method**. The boosting method is "freeze base, learn residual, repeat." Applied to PR110 specifically (NOT a generic DepthSep base), the substrate inherits PR110's contest-grade frontier (0.193 [contest-CPU] band) as its starting point and ADDS rate-bytes-per-distortion-reduction trade-off via the residual sidecar. The optimization question becomes empirical: does the residual learner extract enough signal at the rate cost to net negative ΔS?

## Canonical-vs-unique decision per layer

Per Catalog #290 NON-NEGOTIABLE + the binding 2026-05-26 reframing's default-FORK posture:

| Layer | Decision | Rationale |
|---|---|---|
| **Base learner**: PR110 HNeRV decoder + fec6 + Huffman k=16 | **FORK_BECAUSE_PRINCIPLED_MISMATCH** — externalize as frozen black-box. The canonical NeRV-family pattern is "train your own base"; PR110's 0.193 frontier is the binding signal that we must INHERIT not REPLACE. The residual learner sees PR110 reconstructions as conditioning input, never modifies them. | Substrate-optimal per the operator's "against PR110" framing. ADOPT_CANONICAL would mean retraining a base decoder from scratch, throwing away the 0.193 frontier. |
| **Residual codec**: 1-round (L0) / 2-3 rounds (L1+) iterative residual refinement | **FORK_BECAUSE_PRINCIPLED_MISMATCH** — distinctive paradigm. Boosting is the substrate; bolt-on canonical-pattern doesn't exist for "iterate over frozen-base residuals." The sister `boost_nerv/` substrate's `_BoostingHead` MLP-on-RGB is the architectural primitive we share with — but the conditioning input differs (PR110 latent z, not local renderer z). | Substrate engineering exceeds bolt-on size budget per HNeRV parity discipline L7. |
| **Archive grammar**: BPR1 (BoostNeRV-PR110-Residual v1) sidecar magic, prepended to PR110 archive bytes | **FORK_BECAUSE_PRINCIPLED_MISMATCH** — distinctive 24-byte header carrying NUM_BOOSTING_ROUNDS u8 + PR110_BASE_ARCHIVE_SHA256_PREFIX[16] u128 (binds the sidecar to a specific PR110 base archive sha for runtime closure proof) + RESIDUAL_BLOB_LEN u32. Cannot share with sister BSV1 (which doesn't bind to an external base archive). | Per CLAUDE.md HNeRV parity L3: archive grammar = monolithic single-file `0.bin` with fixed offsets declared in `codec.py` source. The PR110_BASE_SHA256_PREFIX binding is the structural-extinction primitive that prevents the residual sidecar from being silently mis-applied to a non-PR110 base. |
| **Score-aware loss**: PR110-residual-aware Lagrangian | **ADOPT_CANONICAL** — routes through `tac.substrates._shared.score_aware_loss_real_scorer_test_kit` for the proxy-AND-canonical-helper scorer path; FORK the inner residual-vs-base composition (PR110 reconstruction is the additive baseline, residual is the additive correction). | The canonical scorer helper is mathematically optimal for the loss term ordering (preprocess_input → no-grad targets → seg / pose / rate decomposition); only the per-pair INPUT shape changes (we compose `rgb_iter = rgb_pr110_base + residual` before feeding to canonical scorer). |
| **Inflate runtime**: `inflate.sh` invokes PR110's `inflate.sh` FIRST, then our `inflate.py` reads the BPR1 sidecar + adds per-pair residual to PR110-produced frames | **FORK_BECAUSE_PRINCIPLED_MISMATCH** — distinctive 2-stage inflate path. The canonical NeRV-family pattern is "single-stage inflate from monolithic archive"; the boosting-against-PR110 pattern is "stage 1: invoke base inflate as subprocess; stage 2: load base output frames + sidecar + emit boosted frames." LOC budget: ≤200 per HNeRV parity L4 (substrate engineering exception, lane_class=substrate_engineering for the inflate.py). | The 2-stage inflate is the structural-extinction primitive that BINDS the sidecar to a specific PR110 base archive; alternative inflate-time mechanism would be to vendor PR110's HNeRV decoder weights INTO our archive, which defeats the entire "preserve PR110 frontier bytes unchanged" point. |
| **MLX trainer**: train on MLX local-Apple-Silicon per Catalog #1265 + #1257 + #1251 cascade | **FORK_BECAUSE_PRINCIPLED_MISMATCH** — the existing canonical training stack is PyTorch-on-CUDA. Per binding 2026-05-26 reframing + Catalog #1 MPS-fallback forbidden + Catalog #1265 contest-equivalence gate: MLX is the substrate-optimal local-training path that closes the dev-velocity loop without paid GPU. Export bridge per #1251 produces PyTorch state_dict; #1257 packages to contest archive; #1265 gate verifies MLX↔PyTorch decoder parity ≤ 0.001 contest-units BEFORE paid CUDA dispatch is authorized. | Substrate-optimal per the binding strategic correction. ADOPT_CANONICAL (PyTorch) would suppress the dev-velocity gain MLX provides + would inflate cost-per-iteration during the L1 cargo-cult-unwind cycle. |
| **Tier-1 engineering primitives** (autocast_fp16, TF32, torch.compile, no_grad-at-eval, GTScorerCache, canonical scorer-loss helper) | **N/A at L0; ADOPT_CANONICAL at L1+** — the MLX trainer doesn't use these PyTorch primitives. At export → CUDA dispatch time, the PyTorch state_dict export uses the canonical PyTorch tier-1 stack per Catalog #270 dispatch optimization protocol. | Per Catalog #270 tool-dispatch scope clarification: substrate-only primitives are skipped for MLX training; canonical primitives apply at the export→CUDA bridge. |
| **eval_roundtrip + EMA + score-aware loss** | **ADOPT_CANONICAL** — required by CLAUDE.md NON-NEGOTIABLE. The MLX trainer wires `apply_eval_roundtrip_during_training` (Catalog `tac.differentiable_eval_roundtrip`) + EMA decay 0.997 (canonical `tac.training.EMA`-equivalent in MLX) + canonical scorer routing per Catalog #164. | NON-NEGOTIABLE. |
| **Archive provenance / canonical Provenance umbrella** | **ADOPT_CANONICAL** per Catalog #323 + #210 sister discipline | Every emitted artifact carries `tac.provenance.build_provenance_for_predicted` (MLX-research-signal grade) until paired contest-CPU + contest-CUDA anchor lands per Catalog #192. |
| **Tests**: shape + smoke + Catalog #139 byte-mutation no_op_proof + 2-stage inflate parity vs PR110-alone baseline | **ADOPT_CANONICAL** test patterns (mirror `ds_nerv` / sister `boost_nerv/` test scaffolding); FORK the 2-stage inflate parity test (NEW canonical pattern this substrate introduces). | Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_detector enforced. |

**Summary**: 6 of 10 layers FORK because the "boosting against PR110" paradigm is structurally distinct from the generic NeRV-family canonical. 3 of 10 ADOPT_CANONICAL (score-aware loss, eval_roundtrip+EMA, Provenance umbrella). 1 N/A at L0.

## 9-dimension success checklist evidence

Per Catalog #294 NON-NEGOTIABLE:

1. **UNIQUENESS** ⭐⭐⭐⭐⭐: "Boosting residual sidecar against a frozen contest-frontier base" is paradigm-orthogonal to every existing substrate. The sister `boost_nerv/` (generic boosting) trains its own base; we INHERIT PR110's. No sibling at L0 across the 95+ substrate canvas.
2. **BEAUTY + ELEGANCE**: ≤700 LOC total scaffold. Each file reviewable in 30s per L12. The 2-stage inflate is the canonical instance of "compose with existing frontier without retraining": shell-call PR110 inflate → read frames + sidecar → emit boosted frames. The math is one line: `rgb_boosted = clamp(rgb_pr110_base + residual_clamped, 0, 1)`.
3. **DISTINCTNESS**: distinctive BPR1 magic + PR110_BASE_SHA256_PREFIX binding in the 24-byte header structurally prevents mis-application to non-PR110 bases. Explicit 2-stage inflate path.
4. **RIGOR**: L0 tests cover (a) MLX→PyTorch decoder parity (per #1251 export bridge), (b) Catalog #91 ENCODE_INFLATE_ROUNDTRIP, (c) Catalog #139 byte-mutation no_op_proof (mutate residual sidecar → inflate output changes), (d) 2-stage inflate parity (NEW canonical pattern), (e) BPR1 header sha-prefix binding refuses non-matching PR110 base. `_full_main` raises NotImplementedError per Catalog #240.
5. **OPTIMIZATION PER TECHNIQUE**: per Catalog #290 canonical-vs-unique decision table above. 6/10 layers FORK because boosting-against-PR110 paradigm is structurally distinct.
6. **STACK-OF-STACKS-COMPOSABILITY**: composes WITH PR110 frontier as base (the "against PR110" anchor). Composes WITH any future frontier base via `PR110_BASE_SHA256_PREFIX` parameterization (same substrate, retargeted to PR111/PR112 by re-extracting residuals against the new frontier).
7. **DETERMINISTIC REPRODUCIBILITY**: byte-stable BPR1 sidecar (deterministic ZIP via fixed timestamp + ZipInfo per Catalog 19); MLX seed pinned; PR110 base archive sha256 binding makes the composed archive's sha256 trivially reproducible.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: target sidecar ≤8 KB (residual codec: 1 boosting round × per-pair int8 residual at downsampled (96×128) spatial resolution + brotli-quality9 ≈ 5-10 KB; budget tightens to ≤4 KB at L1 with sparse encoding). At PR110's 174 KB base, adding 8 KB → Δrate = 25 × 8192 / 37545489 = **+0.00546 contest-units**. The boosting residual must net at least -0.00546 in d_seg + d_pose reduction to break even; targeting -0.01 net Δscore as Phase 2 council symposium criterion (Catalog #325).
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted_band at L0 is `pending_post_training` per Catalog #324 phantom-random-init refusal (sister Catalog #324 anchor 22× miss C6 IBPS 2026-05-17). Theoretical lower bound via Atick-Redlich cooperative-receiver framing of residual coding: if residual carries ≥1 bit per scorer-class-relevant pixel and base reconstruction error is uniformly distributed, the achievable ΔS at 8 KB sidecar is bounded below by ≈ -0.015 (rough first-principles estimate; Phase 2 council requires Dykstra-feasibility intersection check per Catalog #296 + Shannon R(D) bound per operator-routable #2 above).

## Cargo-cult audit per assumption

Per Catalog #303 NON-NEGOTIABLE:

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| PR110 archive bytes can be preserved unchanged while adding a sidecar | HARD-EARNED | PR101 split_brotli + PR106 latent_sidecar precedent established the sidecar pattern for the fec6 archive family; PR110 inherits the same monolithic-single-file 0.bin grammar | none required |
| Residual learner extracts non-trivial signal on PR110 reconstructions | CARGO-CULTED | untested at PR110's specific operating point | empirical 100ep MLX smoke per operator-routable #3 |
| 1 boosting round is sufficient for L0 SCAFFOLD | CARGO-CULTED | L0 sanity choice | sweep at L1: 1/2/3 rounds |
| Per-pair residual gain clamp magnitude (default 0.05 vs sister 0.10) | CARGO-CULTED | tighter clamp because residual is on top of contest-grade base (less room to perturb) | empirical per-substrate tuning |
| Shared latent z_pr110 extracted from PR110 archive is sufficient conditioning | CARGO-CULTED | cheap variant; per-pair z_residual would inflate sidecar rate 2× | empirical sweep at L1 |
| Int8 residual at downsampled (96×128) spatial resolution + bilinear upsample | CARGO-CULTED | tradeoff between rate cost and reconstruction fidelity; choice mirrors sister NSCS06 v6→v7 chroma_lut path | sweep at L1: int4 vs int8 vs FP4 residual + (48×64) vs (96×128) vs (192×256) spatial |
| Brotli-quality9 compression on residual blob | HARD-EARNED | canonical PR101/PR106/PR110 family default; well-validated | none required |
| 2-stage inflate (shell-call PR110 inflate first) is admissible per contest rules | CARGO-CULTED | inflate.sh ≤200 LOC budget per HNeRV parity L4; subprocess-invoking PR110 inflate vs vendoring PR110 weights into our archive is a contest-compliance design question | Phase 2 council symposium per Catalog #325 + contest scorer behavior verification on archive-of-archive composition |
| MLX-first training (not PyTorch-on-CUDA) | HARD-EARNED-EMPIRICALLY-VERIFIED | Catalog #1265 gate establishes MLX↔PyTorch decoder parity ≤ 0.001 contest-units anchor 2026-05-26 (72× margin below PR110-vs-PR101 frontier delta 0.000789); MLX is canonical for Path 3 training per #1265 landing | none required |
| Boosting curriculum: freeze PR110 → extract per-pair residual targets → train residual learner → freeze → (optional) round 2 → ... | HARD-EARNED | canonical gradient-boosting curriculum (Friedman 2001 + Liu ECCV 2024 BoostNeRV); well-validated mathematical foundation | none required |

**Unwind cycle**: per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable + NSCS06 v6→v7 44% improvement anchor 2026-05-16: at L1, apply cargo-cult-unwind per CARGO-CULTED row, re-test sextet per Catalog #292 + #300, iterate until PROCEED-unconditional, THEN dispatch.

## Observability surface

Per Catalog #305 NON-NEGOTIABLE 6-facet definition:

1. **Inspectable per layer**: per-boosting-round residual magnitude logged via MLX forward hook on `_ResidualHeadMLX.forward` return. Per-pair Δscore decomposition (d_seg / d_pose) on PR110-base-alone vs PR110-base+residual frames computable at smoke-eval time via the canonical scorer helper invocation.
2. **Decomposable per signal**: trainer's score-aware loss returns `parts` dict with `rate_term` / `seg_term` / `pose_term` / `loss_total` / `residual_magnitude_p50` / `residual_magnitude_p99` per the score-domain Lagrangian.
3. **Diff-able across runs**: BPR1 sidecar archive grammar is byte-stable (deterministic ZIP per Catalog 19); two MLX runs with same seed produce identical sidecars. Composed archive (PR110 base + BPR1 sidecar) is sha256-reproducible from `(PR110_BASE_SHA256, BPR1_SIDECAR_SHA256)` pair.
4. **Queryable post-hoc**: trainer writes `provenance.json` per Catalog #323 canonical Provenance + `boost_nerv_pr110_residual_run_provenance.json` with `(pr110_base_sha256, num_boosting_rounds, mlx_seed, smoke_score_band, residual_magnitude_distribution)`.
5. **Cite-able**: every persisted artifact carries `(substrate_tag=boost_nerv_pr110_residual, lane_id, git_head, dispatch_instance_job_id, pr110_base_sha256_prefix)` per Catalog #245 modal_call_id_ledger pattern.
6. **Counterfactual-able**: Catalog #139 byte-mutation smoke proves composed frame output changes when residual sidecar byte 0 is perturbed; per-pair sensitivity-map contribution computable via inverse-water-filling against PR110-base-alone baseline (sister Catalog #344 canonical equations registry consumer hook #1).

## Predicted ΔS band

Per Catalog #296 NON-NEGOTIABLE Dykstra-feasibility check + first-principles Shannon R(D) citation requirement:

**Predicted band: pending_post_training** per Catalog #324 (refuses phantom_random_init predictions per sister C6 IBPS 22× miss anchor).

**First-principles upper bound on achievable ΔS reduction** (Shannon R(D) framing):

- Let `R_residual = 8192 bytes × 8 bit/byte = 65536 bits` per archive (L0 budget).
- Per-pair residual budget: `R_residual / 600 pairs = 109 bits/pair` average.
- Per-pixel residual budget at (96×128) downsampled grid: `109 / (96×128) ≈ 0.0089 bits/pixel`.
- Atick-Redlich cooperative-receiver framing: if the scorer-conditional entropy `H(GT|PR110_base) ≈ 0.5 bits/pixel` at PR110's operating point (rough first-principles estimate; Phase 2 must measure via sister Catalog `mdl_scorer_conditional_ablation.py --tier c` on PR110 frontier archive), the residual learner can extract at most ≈ 0.0089 / 0.5 = 1.8% of the available scorer-relevant entropy.
- Predicted ΔS reduction range: `[-0.015, -0.001]` contest-units (lower bound assumes ideal residual codec at Shannon R(D); upper bound assumes residual is nearly-random vs scorer signal). Rate cost is `+0.00546` (computed above).
- Net predicted ΔS: `[-0.010, +0.0045]` contest-units. **Sign ambiguous at L0**; Phase 2 council symposium must reduce this band before paid dispatch.

**Dykstra-feasibility intersection check**: the rate constraint (Δbytes ≤ 8192 bytes) + score-aware-loss constraint (residual must yield d_seg + d_pose reduction larger than rate cost) + the canonical-frontier-protection constraint (must not destroy PR110's 0.193 baseline) form a convex feasibility region. The Dykstra alternating-projections method (per CLAUDE.md "Council conduct" Dykstra co-lead) verifies feasibility: feasible at the rate-relaxed limit (16 KB sidecar would clearly admit non-trivial residual signal); marginally feasible at 8 KB (the L0 budget); infeasible at 4 KB without sparse encoding. Phase 2 council symposium per Catalog #325 must run this feasibility check formally.

<!-- PREDICTED_BAND_VIBES_OK:l0_scaffold_pending_phase_2_council_symposium_per_catalog_325_and_post_training_tier_c_validation_per_catalog_324_first_principles_shannon_bound_above_is_paper_calculation_not_runtime_anchor -->

## MLX curriculum stages with smoke convergence targets

Per the binding 2026-05-26 reframing: design the curriculum FOR BoostNeRV (not bolt-on training schedule).

### Stage 0: PR110 base extraction (no training)

- Input: PR110 final evidence pack archive (`.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/archive.zip` sha `<from archive_sha256.txt>`).
- Operation: run PR110's `inflate.sh` to produce 600 pair-RGB reconstructions; cache in `.omx/state/pr110_base_reconstructions_<sha_prefix>/{frame_0,frame_1}_per_pair.npy` (one-time cost; ~30 s on macOS-CPU).
- Output: per-pair `rgb_pr110_base_0`, `rgb_pr110_base_1` tensors (the "frozen base learner output").
- MLX-implementable: NO (this is the PR110-inflate-as-subprocess step; runs in upstream Python). The cached output is loaded into MLX as numpy arrays for subsequent stages.

### Stage 1: Per-pair residual target computation (no training)

- Input: PR110 base reconstructions (Stage 0 output) + `upstream/videos/0.mkv` decoded via pyav.
- Operation: `residual_target_per_pair = GT_pair - PR110_base_pair` (signed float32, no clamping yet).
- Output: per-pair `residual_target_0`, `residual_target_1` tensors in `[-1, 1]` (most values near 0; the boosting question is whether the non-zero tail is structured/learnable).
- Diagnostic: log `residual_target_magnitude_p50` and `residual_target_magnitude_p99` per-pair. Convergence target: p99 should be ≥ 0.05 (5% RGB range) for the boosting paradigm to have headroom to extract signal. If p99 < 0.01 across all pairs, PR110 is already near-optimal and the residual learner has no signal to extract — DEFER per CLAUDE.md "Forbidden premature KILL".

### Stage 2: Residual learner warm-up (MLX-trained, ~10 epochs)

- Architecture: `_ResidualHeadMLX(latent_dim=24, hidden_dim=12)` — same shape as sister `boost_nerv/_BoostingHead` but MLX-implementable + conditioned on PR110-extracted latent (NOT a fresh latent).
- Loss: `L2(predicted_residual, residual_target)` per-pair, mean-reduced.
- Optimizer: MLX Adam, lr=1e-3, β=(0.9, 0.999).
- Convergence target: training loss reduction from initial random ≥ 50% within 10 epochs (indicates the residual learner has fit SOMETHING; doesn't yet validate it's scorer-relevant signal).
- MLX implementable: YES (standard MLX nn.Conv2d + nn.Linear + standard loss).

### Stage 3: Score-aware fine-tune (MLX-trained, ~50 epochs)

- Loss: PR110-residual-aware Lagrangian = `α × Δrate_bytes / 37545489 + β × d_seg(GT, PR110_base + clamped_residual) + γ × sqrt(d_pose(GT, PR110_base + clamped_residual))` per the canonical score-aware-loss formula + Catalog #164 helper.
- Scorer: differentiable SegNet + PoseNet per Catalog `tac.differentiable_eval_roundtrip` + canonical helper invocation per Catalog #164 + Catalog #226. Per CLAUDE.md NON-NEGOTIABLE: eval_roundtrip MUST be wired; EMA MUST be wired; canonical scorer helper MUST be used.
- EMA: decay 0.997 per CLAUDE.md NON-NEGOTIABLE; EMA shadow weights are what ship in the archive (NOT live weights).
- Convergence target: smoke score (PR110 alone vs PR110 + EMA-shadow residual) shows ≥ -0.001 contest-units improvement on the local MLX scorer proxy WITHIN the 50-epoch budget. This is a [macOS-MLX research-signal] verdict per Catalog #1265, NOT a contest score claim per Catalog #127/#192/#317/#341.
- MLX implementable: YES (the scorer requires MLX SegNet + MLX PoseNet — sister Catalog `tac.local_acceleration.mlx_score_calibration` provides the MLX scorer port; this substrate consumes it).

### Stage 4: Archive build + Catalog #1265 contest-equivalence gate

- Extract residual learner EMA shadow weights → quantize to int8 → brotli-quality9 compress → embed in BPR1 sidecar with PR110_BASE_SHA256_PREFIX binding.
- Compose: `composed_archive_bytes = BPR1_SIDECAR_BYTES || PR110_BASE_ARCHIVE_BYTES` (the inflate runtime knows to split on BPR1 magic).
- Export PyTorch state_dict via Catalog `tac.substrates._shared.mlx_to_pytorch_export_bridge` (sister #1251) + package via #1257.
- **MANDATORY**: invoke `tools/gate_mlx_candidate_contest_equivalence.py --archive-zip <composed_archive> --candidate-label "boost_nerv_pr110_residual_v0" --gate-threshold-contest-units 0.001` per Catalog #1265.
- If gate PASSES (verdict=PASS): operator-routable to paired contest-CPU + contest-CUDA dispatch per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
- If gate FAILS: do NOT dispatch; audit MLX↔PyTorch decoder parity per #1251 + #1257 + #1258 corrected methodology.

### Stage 5: Optional boosting round 2 (deferred to L1 cargo-cult-unwind sweep)

- Re-run Stages 1-4 with `residual_target_round_2 = GT - (PR110_base + round_1_residual_EMA_shadow)`.
- Sidecar header `NUM_BOOSTING_ROUNDS=2`; carries both round-1 and round-2 residual blobs.
- Rate cost: ~doubles to ~16 KB (`+0.0109 contest-units`); empirical question whether diminishing-returns residual signal nets to negative ΔS.
- DEFERRED at L0 per Catalog #240 `_full_main raises NotImplementedError`.

## Residual extraction from PR110 — byte / score math

### Byte accounting (predicted)

- PR110 base archive: 178417 bytes (per `archive_metadata.json`).
- Per-pair residual at (96×128) downsampled int8: 96×128×3×600×2 frames = 44,236,800 raw bytes → brotli-quality9 ratio ~0.0002 (highly compressible signed-near-zero residual) → **~8800 bytes** (rough estimate; actual depends on residual entropy distribution).
- BPR1 header overhead: 24 bytes.
- **Composed archive bytes: 178417 + 24 + ~8800 = ~187241 bytes**.
- **Δrate contest-units**: `25 × 8824 / 37545489 = +0.00587 contest-units`.

### Score accounting (predicted, paper calculation NOT measurement)

- PR110 baseline score: 0.193 [contest-CPU] (frontier band).
- If residual learner extracts 50% of available scorer-conditional entropy: predicted d_seg + d_pose reduction ≈ -0.012 contest-units → **net ΔS ≈ -0.006 contest-units** → composed score ≈ 0.187.
- If residual learner extracts 80% of available entropy (idealized): predicted ≈ -0.015 reduction → **net ΔS ≈ -0.009** → composed score ≈ 0.184.
- If residual learner extracts 10% (worst-case noisy): predicted ≈ -0.002 reduction → **net ΔS ≈ +0.004** → composed score ≈ 0.197 (REGRESSION).

**The sign-ambiguous predicted band [-0.010, +0.004] is precisely why L0 SCAFFOLD posture is correct.** Phase 2 council symposium per Catalog #325 MUST run cheap MLX smoke (Stage 3 convergence target) BEFORE any paid CUDA dispatch is authorized; the MLX-research-signal verdict reduces the uncertainty band by 10-100× and disambiguates whether the residual learner is extracting score-relevant signal at PR110's operating point.

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

1. Phase 2 council symposium per Catalog #325 returns PROCEED or PROCEED_WITH_REVISIONS verdict (cargo-cult audit + 9-dim checklist + Dykstra-feasibility predicted band + observability surface + Tier-C validation discipline).
2. Stage 3 MLX smoke convergence verdict: training loss reduction ≥ 50% in 50 epochs AND local MLX scorer proxy shows ≥ -0.001 contest-units improvement on PR110-alone baseline.
3. Catalog #1265 contest-equivalence gate PASSES on composed archive (MLX↔PyTorch decoder parity ≤ 0.001 contest-units).
4. Cargo-cult-unwind for the 8 CARGO-CULTED assumptions above either empirically validates them or substitutes substrate-optimal alternatives.
5. Recipe `dispatch_enabled` flips to true; `predicted_band` declared per Catalog #324 with post-training Tier-C density evidence.

## 6-hook wire-in declaration (per Catalog #125)

- **hook #1 sensitivity-map**: ACTIVE at L1+ — per-pair residual magnitude + per-pair Δscore-decomposition feeds `tac.sensitivity_map.*` consumers. L0 N/A (no measurement yet).
- **hook #2 Pareto constraint**: rate_distortion_v1 (declared in SubstrateContract) — the rate-vs-distortion tradeoff is the substrate's defining axis; constraint is `Δrate ≤ 8192 bytes AND Δ(d_seg + sqrt(d_pose)) ≤ -Δrate × 25/37545489` (must net negative).
- **hook #3 bit-allocator**: ACTIVE at L1+ — per-pair residual bit-budget allocation via canonical `tac.optimization.bit_allocator` consumer; informed by per-pair residual_target_magnitude distribution from Stage 1. L0 uniform allocation (109 bits/pair).
- **hook #4 cathedral autopilot dispatch**: ACTIVE — Phase 2 onward, the cathedral_autopilot ranker consumes this substrate's MLX-smoke-verdict + Catalog #1265 gate verdict to rank against sister Path 3 candidates A/B/C/D per Catalog #335 auto-discovery + #336 invocation + #341 Tier A (observability-only marker).
- **hook #5 continual-learning posterior**: ACTIVE — every MLX smoke verdict appends a `[macOS-MLX research-signal]` anchor per Catalog #341 + #323 canonical Provenance (`build_provenance_for_predicted`); Phase 2 [contest-CPU] / [contest-CUDA] anchors append per Catalog #127.
- **hook #6 probe-disambiguator**: ACTIVE — the Catalog #1265 gate IS the canonical disambiguator between MLX-faithful (dispatch-eligible) vs MLX-too-noisy (audit-required) routing per CLAUDE.md "MLX portable-local-substrate authority".

## Cross-references

- **Sister substrate** (generic boosting, NOT against PR110): `src/tac/substrates/boost_nerv/` lane `lane_boost_nerv_l0_scaffold_20260520`.
- **PR110 frontier reference**: `.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/archive.zip` + sister `pr110_live_body_pre_v44.md` + canonical frontier pointer `.omx/state/canonical_frontier_pointer.json` (`0.19203 [contest-CPU]` band).
- **MLX cascade context**: `mlx_candidate_contest_equivalence_gate_landed_20260526.md` (Catalog #1265) + `pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md` (Catalog #1258 corrected closure) + `pr95_mlx_full_inflate_parity_closure_landed_20260526.md` (Catalog #1257).
- **Sister Path 3 candidates** (concurrent 2026-05-26 fanout): A=DreamerV3 RSSM (`src/tac/substrates/dreamer_v3_rssm/`); B=Z7-Mamba-2; C=NSCS06 v8 chroma_lut (`src/tac/substrates/nscs06_v8_chroma_lut/`); D=Z6 predictive coding (`src/tac/substrates/c1_world_model_foveation/` or sister).
- **Literature anchor**: Friedman 2001 "Greedy Function Approximation: A Gradient Boosting Machine" (canonical gradient-boosting foundation) + Liu et al. ECCV 2024 "BoostNeRV: Iterative Refinement for Implicit Neural Video Representations" (NeRV-family specialization). Atick-Redlich 1990 cooperative-receiver framing for residual coding (canonical Z4 sister).
- **Canonical equation registry** (Catalog #344): residual_hybrid_boosting_savings_v1 is FORMALIZATION_PENDING per the frontmatter; canonical equation #26 procedural_codebook_savings is EXCLUDED per Catalog #359 (residual-hybrid context).

---

<!-- ===== APPEND-ONLY FOOTER: FIX-WAVE-R1 closure 2026-05-26 ===== -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this footer is the
     CORRECTION + CLOSURE record for the R1 review's E-OP1 + E-OP3 + E-OP4
     findings against this design memo. Body above is preserved UNMUTATED per
     APPEND-ONLY discipline; corrections are recorded here. -->

## APPEND-ONLY footer: FIX-WAVE-R1 closure 2026-05-26

**Reference**: R1 review memo `.omx/research/path_3_e_recursive_adversarial_review_r1_3_axis_20260526.md` + aggregate `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md` (commit `80acd6da3`).

### Correction #1: BPR1 header byte count INCONSISTENCY (E-OP1 / E-OP3 closed)

The original body of this design memo references the BPR1 header byte count in 3 places that were INCONSISTENT with the SOURCE-OF-TRUTH constant:

| Surface | Original text | Corrected interpretation |
|---|---|---|
| §"Canonical-vs-unique decision per layer" row 3 (line 67) | "distinctive 24-byte header carrying NUM_BOOSTING_ROUNDS u8 + PR110_BASE_ARCHIVE_SHA256_PREFIX[16] u128..." | The actual header is **29 bytes** per `struct.calcsize('<5sBBB16sIB') = 29` + `BPR1_HEADER_LEN = 29` source constant. |
| §"9-dim success checklist evidence" item 3 DISTINCTNESS (line 84) | "...BPR1 magic + PR110_BASE_SHA256_PREFIX binding in the 24-byte header structurally prevents..." | 29-byte header per the same source constant. |
| §"Residual extraction from PR110 — byte / score math" §"Byte accounting (predicted)" (line 197) | "BPR1 header overhead: 24 bytes." | 29 bytes per the same source constant. |

**Source-of-truth verification**: `.venv/bin/python -c "import struct; print(struct.calcsize('<5sBBB16sIB'))"` → `29`; `src/tac/substrates/boost_nerv_pr110_residual/__init__.py:88` declares `BPR1_HEADER_LEN = 29`.

**Mechanism of the inconsistency**: the original design memo enumerated the header field-by-field as `magic[5] + version[1] + num_rounds[1] + sha_prefix[16] + residual_blob_len[4] + reserved_tail[1] = 28 bytes`, but the actual implementation includes a 1-byte `align[1]` padding (4-byte alignment of the sha_prefix offset) per `struct.calcsize('<5sBBB16sIB')`. The 24-byte and 28-byte figures in the design memo were both incorrect; 29 bytes is the canonical truth.

**Byte accounting correction (§"Byte accounting" line 198-199)**:
- Original: "Composed archive bytes: 178417 + 24 + ~8800 = ~187241 bytes"; "Δrate = 25 × 8824 / 37545489 = +0.00587 contest-units"
- Corrected: Composed archive bytes: 178417 + **29** + ~8800 = **~187246 bytes**. The Δrate calculation in the original used the residual_blob_len = 8824 bytes (which INCLUDES the 24-byte header in the original mental model) and produces ≈ +0.00587 contest-units; with the 29-byte header, residual_blob_len + header total bytes is 5 bytes higher, so corrected Δrate = `25 × 8829 / 37545489 ≈ +0.00588 contest-units` — a difference of `+0.0000033` contest-units (~6 orders of magnitude below the predicted ΔS band's [-0.010, +0.0045] uncertainty). The error does NOT change the substrate's qualitative outcome.

### Correction #2: Canonical equation name (E-OP4 closed)

The original line 234 cited canonical equation as `residual_hybrid_boosting_savings_v1 FORMALIZATION_PENDING`. R1 review verified via `tac.canonical_equations.query_equations()` that the registered equation is named `procedural_predictor_plus_residual_correction_savings_v1` — the same conceptual entity per Catalog #359 sister discipline (residual-hybrid context). The FORMALIZATION_PENDING marker was a placeholder name; the registered equation exists and is the canonical reference.

**Corrected canonical equation citation**: `procedural_predictor_plus_residual_correction_savings_v1` (REGISTERED in `tac.canonical_equations` registry as of 2026-05-26).

### FIX-WAVE-R1 actions landed (this commit batch)

1. **E-OP1 CLOSED via this APPEND-ONLY footer**: design memo BPR1 header byte count corrections recorded above (3 surfaces: §"Canonical-vs-unique" line 67 + §"9-dim" line 84 + §"Byte accounting" line 197). Body is preserved UNMUTATED per APPEND-ONLY discipline; the canonical truth is `BPR1_HEADER_LEN = 29`.
2. **E-OP2 CLOSED via in-place source-code edit**: `src/tac/substrates/boost_nerv_pr110_residual/archive.py` module docstring line 8 corrected from "BPR1 header 28 bytes" to "BPR1 header 29 bytes". Source-code docstrings are in-place editable per source-code evolution discipline; APPEND-ONLY applies to research memo BODY only.
3. **E-OP3 CLOSED via in-place source-code edit**: `src/tac/substrates/boost_nerv_pr110_residual/__init__.py` line 41 archive grammar comment corrected from "24-byte header" to "29-byte header" + expanded to include the `align[1]` field that the original mental model omitted. Source-code docstrings are in-place editable.
4. **E-OP4 CLOSED via this APPEND-ONLY footer**: canonical equation name correction recorded above. The registered name `procedural_predictor_plus_residual_correction_savings_v1` is the canonical reference; the placeholder `residual_hybrid_boosting_savings_v1` in the original body is superseded by this footer per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #344 canonical-equation-registry discipline.

### Post-fix verification (2026-05-26)

- `.venv/bin/python -m pytest src/tac/substrates/boost_nerv_pr110_residual/tests/ -v` → **25/25 pass** (no test changes expected since the fixes are documentation-only / docstring-only; code was always correct).
- Source-code docstrings now consistent with `BPR1_HEADER_LEN = 29` constant: `grep -n "29-byte\|29 bytes" src/tac/substrates/boost_nerv_pr110_residual/{archive.py,__init__.py}` shows the corrected text.

### R2 readiness signal

- R1 counter status post-FIX-WAVE-R1: **CLEAN** for E=BoostNeRV (all P0 + P2 findings closed; no code correctness gaps); R2 can fire on this substrate when the aggregate R1 cycle re-runs.
- No code semantics changed; only documentation surfaces (memo body via APPEND-ONLY footer + 2 source-code docstrings) corrected.

### Cross-references

- FIX-WAVE-R1 landing memo: `.omx/research/path_3_fix_wave_r1_close_findings_landed_20260526.md`
- Source-code diffs: `src/tac/substrates/boost_nerv_pr110_residual/archive.py` line 8 + `src/tac/substrates/boost_nerv_pr110_residual/__init__.py` lines 40-47
- Canonical equation registry: `tac.canonical_equations.query_equations()` (REGISTERED `procedural_predictor_plus_residual_correction_savings_v1` as of 2026-05-26)


# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:design_memo_references_cooperative_receiver_atick_redlich_or_wyner_ziv_framework_in_cross_reference_or_spatial_not_temporal_context_NOT_as_substrate_central_predictive_coding_claim_per_catalog_311_z6_z7_z8_pattern_h_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
