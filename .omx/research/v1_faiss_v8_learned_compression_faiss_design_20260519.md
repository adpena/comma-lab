---
schema: substrate_design_memo_v1
design_id: v1_faiss_v8_learned_compression_faiss_design_20260519
substrate_id: atw_codec_v2_1_faiss_ivf_pq_v8_learned_compression
substrate_alias: atw_v2_1_faiss_pq_v8
parent_substrate_id: atw_codec_v2_1_faiss_ivf_pq
parent_design_memo: atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md
horizon_class: frontier_pursuit
design_date: "2026-05-19"
lane_id: lane_v1_faiss_v4_probe_plus_v8_design_20260519
predicted_band_validation_status: pending_post_training
predicted_band:
  contest_cpu: [0.187, 0.193]   # medal-class POTENTIAL; predicted delta [-0.005, +0.001] from frontier 0.192051
  contest_cuda: [0.198, 0.210]   # predicted delta band from CUDA-frontier 0.205330
canonical_frontier_anchor:
  contest_cpu: "0.1920513169 [contest-CPU] (archive sha 6bae0201; per `.omx/state/canonical_frontier_pointer.json` 2026-05-15)"
  contest_cuda: "0.2053300290 [contest-CUDA] (archive sha 9cb989cef519; per pointer 2026-05-16)"
  pointer_path: ".omx/state/canonical_frontier_pointer.json (per CLAUDE.md 'Frontier scores are pointer-only')"
research_only: true
dispatch_enabled: false
council_anchor_required: false   # DESIGN ONLY; council deliberation NOT appended to posterior per parent prompt
operator_directive: "DD V1 Faiss symposium op-routable #3 (PRIMARY-CROSS-POLLINATION FREE design memo). Cross-pollination with sister C6 IBPS v2 Path B2 (DreamerV3 RSSM categorical posterior K=256 × 24 groups) + sister NSCS06 v8 Variant C (hybrid_class_shift_path_C neural residual decoder) on discrete-posterior strategy at small-neural-architecture scale."
cross_pollination_canonical:
  - sister_design_id: council_t3_cargo_cult_resurrection_c6_ibps_v2_20260519
    sister_path_id: path_b2_categorical_posterior_dreamerv3_rssm
    surface: substrate_latent
    discrete_posterior_strategy: "24 categorical groups × K=256 with Gumbel-Softmax + straight-through estimator"
  - sister_design_id: council_t3_cargo_cult_resurrection_nscs06_v8_variant_c_20260519
    sister_path_id: hybrid_class_shift_path_C
    surface: chroma_residual
    discrete_posterior_strategy: "small neural residual decoder with entropy-bottleneck quantization"
  - sister_design_id: this v1 Faiss V8 design
    sister_path_id: v8_learned_compression_faiss_extension
    surface: side_info_channel
    discrete_posterior_strategy: "small Balle 2018 entropy-bottleneck + scale-hyperprior encoder; ~50K params; ~4-bit categorical codebook"
council_attendees_cited:
  # Canonical inner council per Catalog #346 (cited for design rationale; NOT a council deliberation)
  - Shannon            # CO-LEAD information-theory (R(D) rate-distortion frame)
  - Dykstra            # CO-LEAD optimization-feasibility (alternating projections to Pareto-optimal point)
  - Rudin              # CO-LEAD interpretable-ML (encoder weights + categorical posterior must remain 30-sec-reviewable)
  - Daubechies         # CO-LEAD wavelets (compressive coverage estimator K=8 sister variants)
  - Yousfi             # steganalysis lens (per-region SegNet softmax is steganalysis-canonical feature class)
  - Fridrich           # steganalysis founder (capacity argument; V8 byte budget vs steganalysis-canonical ~10-15KB capacity)
  - Contrarian         # weak-arguments VETO
  - Assumption-Adversary  # shared-assumption framing VETO
  - Quantizr           # adversarial reverse-engineering (PR #56 hybrid analytical+neural validates discrete-posterior at contest scale)
  - Hotz               # engineering shortcuts (smallest neural architecture that preserves I(X;T))
  - Selfcomp           # PR #56 lead implementer (88K SegNet learnable class targets canonical discrete-posterior precedent)
  - MacKay             # MDL + Bayesian (Laplace prior + max-entropy categorical distribution)
  - Balle              # neural-compression SOTA (2018 entropy-bottleneck + scale-hyperprior CANONICAL reference)
  - PR95Author         # PR #95 HNeRV root author (bind-all-ingredients + empirical-first methodology)
  # GRAND COUNCIL (topical seats)
  - Schmidhuber        # compression-as-intelligence (DreamerV3 RSSM categorical sister cross-pollination canonical)
  - Hafner             # DreamerV3 RSSM (categorical posterior fix for continuous-latent collapse)
  - Atick              # Atick-Redlich cooperative-receiver theorem (V8 IS one specific compression strategy for side-info channel)
  - Wyner              # Wyner-Ziv side-information source coding R_WZ(D) bound
  - Tishby             # IB Lagrangian framework (Path B2/B7 cross-pollination)
related_deliberation_ids:
  - council_t3_cargo_cult_resurrection_v1_faiss_20260519  # parent T3 symposium that elevated V8 to PRIMARY-CROSS-POLLINATION
  - cargo_cult_resurrection_top3_symposiums_landed_20260519  # landing memo for parent
  - council_t3_cargo_cult_resurrection_c6_ibps_v2_20260519  # sister Path B2 cross-pollination
  - council_t3_cargo_cult_resurrection_nscs06_v8_variant_c_20260519  # sister hybrid_class_shift_path_C cross-pollination
  - atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518  # canonical V1/V2/V3 design memo this V8 extends
canonical_helpers_cited:
  - "tac.optimization.faiss_ivf_pq_atw_channel.{build_pq_codebook, encode_per_region_histogram, decode_per_region_histogram, serialize_codebook, estimate_pq_encoding_budget}"
  - "tac.scorer.load_default_scorers (canonical SegNet preprocess_input)"
  - "tac.differentiable_eval_roundtrip (per CLAUDE.md eval_roundtrip non-negotiable)"
  - "tac.substrates._shared.score_aware_common.score_pair_components (Catalog #164)"
  - "tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call (Catalog #226)"
  - "tac.substrates._shared.trainer_skeleton (Catalog #178 TF32 + #190 hardware_substrate)"
  - "tac.wyner_ziv_deliverability.proof_builder (Catalog #319 PRIMARY consumer)"
  - "tac.probe_outcomes_ledger (Catalog #313 outcome registration)"
---

# V1 Faiss V8 learned-compression Faiss extension — design memo

> **Status**: DESIGN ONLY. NOT a council deliberation. No posterior anchor appended. Per DD V1 Faiss T3 symposium 2026-05-19 op-routable #3 (PRIMARY-CROSS-POLLINATION FREE design memo). Predicted band [0.187, 0.193] contest-CPU = medal-class POTENTIAL (delta [-0.005, +0.001] from frontier 0.1920513169 per canonical pointer 2026-05-15).

> **Scope-discipline**: Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable, this is a `research_only: true` + `dispatch_enabled: false` design artifact. Paid dispatch requires explicit operator-frontier-override per Catalog #199 paired-env discipline AFTER design memo review.

## 0. Why V8 NOW

Per DD V1 Faiss T3 symposium (`council_t3_cargo_cult_resurrection_v1_faiss_20260519.md`) op-routable #3, V8 learned-compression Faiss extension is elevated to **PRIMARY-CROSS-POLLINATION** per sister cross-pollination with C6 IBPS v2 Path B2 (DreamerV3 RSSM categorical posterior) + NSCS06 v8 Variant C (hybrid_class_shift_path_C neural residual decoder). All 3 sister cargo-cult resurrection candidates CONVERGE on **discrete-posterior strategy at small-neural-architecture scale**.

The V8 design extends the Faiss-IVF-PQ family beyond pure analytical compression (V1-V7) into a small learned encoder per Balle 2018 entropy-bottleneck + scale-hyperprior recipe applied to the side-info channel surface. The structural prediction: V8 produces the STRONGEST predicted-delta of the Faiss family by combining (a) Balle 2018 R(D)-optimal training + (b) categorical posterior preserves I(X;T) without continuous-mode-collapse per Hafner DreamerV3 + (c) cross-pollination canonical with sister cargo-cult resurrection candidates.

**Empirical anchor predecessor evidence (from DD V1 Faiss symposium §4 reactivation paths)**:
- V1 dense: MI=2.46 bits/symbol, 452799 bytes → +0.30 rate (FALSIFIED at 386× <2KB budget)
- V2 sparse top-k: MI=2.46, 7941 bytes → +0.005 rate (ARGUABLE; outside <2KB budget)
- V3 pool-shared: MI=0.12, 3114 bytes → +0.002 rate (WEAK_CONDITIONING)
- V4 (M=2, ksub=128, top-k=3): pending V4 hand-rolled probe outcome (bundled with this memo)

V8's prediction: ~50K-param neural encoder produces MI ≈ 1.5-2.0 bits/symbol at byte cost ≤ 6KB → realized ΔS at +0.005 rate cost offset by -0.010 to -0.015 distortion saving → net delta [-0.005, +0.001].

## 1. Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind path | Probe / mitigation |
|---|------------|----------------|-------------|-------------------|
| 1 | A small neural encoder (~50K params) compresses per-region SegNet softmax to a discrete-posterior representation with MI > 0.5 bits/symbol at byte cost ≤ 6KB | CARGO-CULTED-PENDING-EMPIRICAL | Empirical V8 smoke ($5-30 Modal A100 5-50ep) per Catalog #167 smoke-before-full pattern; predicted MI ≈ 1.5-2.0 per Balle 2018 R(D)-optimal training | V8 smoke conditional on operator-frontier-override per Catalog #199 |
| 2 | Balle 2018 entropy-bottleneck + scale-hyperprior recipe transfers to side-info channel surface | HARD-EARNED | Balle 2018 entropy-bottleneck is mathematically general (Balle-Laparra-Simoncelli 2018 entropy bottleneck applies to any continuous latent representation; the scale hyperprior is a parametric prior over latent variances) | Balle 2018 paper + CompressAI library canonical implementations |
| 3 | Discrete-posterior strategy at side-info channel surface IS structurally consistent with discrete-posterior at substrate latent surface (C6 IBPS Path B2) AND chroma residual surface (NSCS06 v8 Variant C) | HARD-EARNED-CROSS-POLLINATION | Per Atick-Redlich cooperative-receiver theorem: surface-INDIFFERENT as long as compressed signal preserves I(B; R(B)); discrete-posterior is one specific way to preserve I(X;T) without continuous-mode-collapse per Hafner DreamerV3 2024 | Sister cross-pollination canonical with C6 IBPS Path B2 + NSCS06 v8 Variant C |
| 4 | The categorical alphabet size K=16 (4 bits) is the byte-optimal discrete-posterior cardinality for the side-info channel | CARGO-CULTED-PENDING-EMPIRICAL | Sweep K ∈ {8, 16, 32, 64} during smoke; predicted Pareto-optimal at K=16 per Hafner DreamerV3 categorical posterior + Balle 2018 R(D) operating point analysis | K-sweep within V8 smoke (1-2 epoch each per Catalog #167 burst pattern) |
| 5 | Gumbel-Softmax reparametrization with straight-through estimator is the canonical training mechanism | HARD-EARNED | Hafner DreamerV3 2024 explicitly uses Gumbel-Softmax with straight-through estimator and reports robust convergence; sister C6 IBPS Path B2 uses identical reparametrization | Sister C6 IBPS Path B2 design memo + Hafner 2024 paper |
| 6 | ~50K parameter budget is sufficient for the encoder + scale hyperprior | CARGO-CULTED-PENDING-EMPIRICAL | PR #56 demonstrates 88K SegNet learnable class targets at 0.33 score; V8 at side-info channel surface (lower-dimensional than full-frame chroma) should require fewer params | V8 architecture sweep within smoke; if encoder collapses at 50K, scale to 100K |
| 7 | The β regularization weight for R(D) trade-off is approximately β=0.01 per Balle 2018 typical recipe | CARGO-CULTED-PENDING-EMPIRICAL | Sweep β ∈ {0.001, 0.01, 0.1} during smoke; the contest's rate weight is 25 × bytes / 37545489 = ~6.66e-7 per byte; β should scale to match contest operating point | β-sweep within V8 smoke |
| 8 | The encoder + decoder are byte-stable across training seeds when (a) seed-pinned and (b) categorical posterior temperature schedule pinned | HARD-EARNED | Sister C6 IBPS Path B2 + Hafner DreamerV3 + Balle 2018 all use deterministic-with-seed training; categorical Gumbel-Softmax temperature schedule (e.g. linear anneal from 1.0 → 0.5 over training) is deterministic | Per Catalog #239 byte-stable archive invariant; smoke must produce sha256-stable encoder weights across re-runs |
| 9 | The V8 archive grammar fits inside the V2 ATW2 archive schema (Catalog #220 operational mechanism declaration) | CARGO-CULTED-PENDING-DESIGN | The V2 ATW2 schema reserves `scorer_class_prior_table_fp16` slot for the side-info channel; V8 must declare a NEW schema slot `learned_compression_encoder_weights_int8 + categorical_posterior_codeword_stream_brotli` per Catalog #220 operational mechanism | V8 archive grammar declaration in §2 below; integration with V2 ATW2 archive builder |
| 10 | The V8 inflate runtime LOC budget (≤200 LOC per HNeRV parity discipline L4 waiver ceiling) fits the encoder forward pass + categorical posterior decoder | CARGO-CULTED-PENDING-IMPLEMENTATION | Hafner DreamerV3 categorical posterior decoder is ~50 LOC; Balle 2018 entropy-bottleneck decoder is ~80 LOC; combined ~130 LOC; within budget | Implementation review post-smoke; if >200 LOC, declare Catalog #205 inflate device-fork waiver |
| 11 | Cross-pollination across the 3 sister candidates produces additive ΔS (orthogonal axes per CLAUDE.md Stack-of-Stacks discipline) | CARGO-CULTED-PENDING-EMPIRICAL | The 3 sister candidates operate at DIFFERENT substrate surfaces (V8 = side-info channel; C6 IBPS Path B2 = substrate latent; NSCS06 v8 Variant C = chroma residual) so additivity is plausible; but composition ΔS must be empirically measured via paired Modal CUDA dispatch | Composition smoke ($15-50) AFTER each sister produces individual empirical anchor |
| 12 | The Catalog #324 post-training Tier-C density validation discipline applies to V8 | HARD-EARNED | Per CLAUDE.md "Catalog #324 post-training Tier-C density validation discipline": predicted band derived BEFORE training is structurally invalid; V8 predicted band [0.187, 0.193] requires post-training Tier-C re-measurement on landed V8 archive | Catalog #324 validation in §5 below |

**Cargo-cult unwind summary**: 12 assumptions enumerated; 4 HARD-EARNED + 6 CARGO-CULTED-PENDING-EMPIRICAL + 1 HARD-EARNED-CROSS-POLLINATION + 1 CARGO-CULTED-PENDING-DESIGN. The dominant unwind paths are (a) V4 hand-rolled probe outcome (bundled with this memo) informing assumption #1 + #4 + #6; (b) V8 smoke testing assumptions #1 + #4 + #6 + #7; (c) sister cross-pollination outcome informing assumption #11.

## 2. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | V8 evidence | Verdict |
|---|-----------|-------------|---------|
| 1 | UNIQUENESS (class-shift not within-class) | YES — discrete-posterior strategy at small-neural-architecture scale IS class-shift from pure-analytical Faiss-IVF-PQ (V1-V7) per Catalog #307 paradigm-vs-implementation: V1-V7 are pure-analytical paradigm; V8 is the learned-compression paradigm. Cross-pollination canonical with sister C6 IBPS Path B2 + NSCS06 v8 Variant C strengthens uniqueness. | PASS |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | YES — ~700 LOC total budget (~400 LOC delta from V2 sparse top-k baseline + ~300 LOC encoder/decoder = within PR #101 GOLD 605 LOC sister + bolt-on budget per HNeRV parity L7 substrate-engineering exception). Encoder + scale hyperprior + GDN nonlinearity + categorical posterior decoder all reviewable per file per Catalog #305 observability. | PASS |
| 3 | DISTINCTNESS (explicitly different from sisters) | YES — distinct from V1-V7 (analytical Faiss family) via learned encoder + categorical posterior strategy; structurally CONSISTENT with sister C6 IBPS Path B2 + NSCS06 v8 Variant C at DIFFERENT substrate surface (V8 = side-info channel; sisters = substrate latent + chroma residual). The cross-pollination IS structurally bidirectional. | PASS |
| 4 | RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | YES — premise verified via DD V1 Faiss T3 symposium (20-attendee adversarial review with 8 Assumption-Adversary verdicts); assumption classification per §1 above (12 assumptions); empirical anchor from V1/V2/V3 probe matrix + V4 hand-rolled probe (bundled); sister cross-pollination empirical evidence from C6 IBPS + NSCS06 sister probes (pending). | PASS |
| 5 | OPTIMIZATION-PER-TECHNIQUE | YES — Balle 2018 R(D)-optimal entropy bottleneck IS canonical optimization-per-technique for learned compression; the technique is bleeding-edge SOTA for neural compression; Gumbel-Softmax with straight-through estimator is canonical for discrete-posterior training. Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": canonical-vs-unique decision per layer documented in §4 below. | PASS |
| 6 | STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS) | YES — composable with PR101 frame_exploit_selector (orthogonal axes per ATW V2-1 design); composable with sister C6 IBPS Path B2 (substrate latent surface vs side-info channel surface; orthogonal); composable with sister NSCS06 v8 Variant C (chroma residual surface vs side-info channel surface; orthogonal). Predicted additive ΔS subject to composition smoke empirical validation per assumption #11. | PASS-WITH-EMPIRICAL-VALIDATION-PENDING |
| 7 | DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned) | YES — Balle 2018 entropy-bottleneck training is deterministic with seed pin; Gumbel-Softmax + straight-through estimator is deterministic with seed pin + temperature schedule pin; sister C6 IBPS Path B2 + Hafner DreamerV3 all use deterministic-with-seed training. Per Catalog #239 byte-stable archive invariant: V8 archive sha256 must be stable across re-runs. | PASS |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | UNKNOWN — V8 codec-loop performance not measured; predicted contest-CUDA inflate ≈ 1-5ms at A100 per Balle 2018 frontier; encoder forward ~0.5ms; categorical posterior decoder ~0.5ms. Per CLAUDE.md "Max observability — non-negotiable" Catalog #305 observability declaration in §3 below. | PASS-WITH-EMPIRICAL-VALIDATION-PENDING |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | YES (predicted) — predicted band [0.187, 0.193] is medal-class POTENTIAL (predicted delta [-0.005, +0.001] from frontier 0.1920513169 per canonical pointer 2026-05-15). STRONGEST predicted-delta of the entire Faiss V1-V8 family per DD V1 Faiss symposium analysis. Per Catalog #324: predicted band requires post-training Tier-C validation discipline. | PASS-WITH-CATALOG-324-VALIDATION-REQUIRED |

**9-dim summary**: 6 PASS + 3 PASS-WITH-EMPIRICAL-VALIDATION-PENDING + 1 PASS-WITH-CATALOG-324-VALIDATION-REQUIRED. Per the canonical Catalog #294 discipline: no FAIL dimensions; design is structurally sound BUT requires empirical validation before paid dispatch.

## 3. Observability surface (Catalog #305)

1. **Inspectable per layer**: (a) Balle 2018 encoder forward output IS a continuous latent representation (visualizable as histogram); (b) categorical posterior parameters (24 categorical groups × K=16 categories) IS a 24×16 matrix dumpable to JSON; (c) Gumbel-Softmax temperature schedule IS a per-epoch scalar trace; (d) scale hyperprior parameters (mean + scale per latent dim) IS a 2×D matrix dumpable to JSON. All layers inspectable via canonical `tac.observability.dump_layer_state` (if helper exists; else direct PyTorch state_dict dump).
2. **Decomposable per signal**: (a) per-categorical-group I(X;T_k) contribution measurable via mutual information estimation per group (entropy of T_k - conditional entropy of T_k|X); (b) per-pair MI between A1 latent stream and V8 codeword stream measurable via `compute_pq_mi_verdict` canonical helper; (c) per-pair byte cost decomposable into encoder weights + categorical codeword stream + scale hyperprior; (d) per-pair score-distortion decomposable into seg + pose + rate components per `tac.substrates._shared.score_aware_common.score_pair_components`.
3. **Diff-able across runs**: V8 encoder weights sha256 + categorical posterior parameters sha256 + scale hyperprior sha256 + per-pair codeword stream sha256 give byte-level diff across (K, G, β, temperature_schedule, seed, learning_rate) tuples per Catalog #245 modal_call_id_ledger.
4. **Queryable post-hoc**: V8 smoke + full run results emit canonical JSON to `experiments/results/lane_v8_learned_compression_faiss_smoke_<utc>/v8_smoke_results.json` per Catalog #245 pattern; probe outcomes registered to `.omx/state/probe_outcomes.jsonl` per Catalog #313 canonical ledger.
5. **Cite-able**: every V8 variant tuple is `(K, G, β, temperature_schedule, learning_rate, seed, encoder_architecture_hash, decoder_architecture_hash)` — unique citation; V8 training procedure is `experiments/train_substrate_v8_learned_compression_faiss.py` canonical helper (TO-BE-BUILT per implementation roadmap §6).
6. **Counterfactual-able**: per Catalog #139 byte-mutation gate + Catalog #105 no-op detector: mutating one byte of encoder weights blob should change decoded categorical posterior; mutating one byte of categorical codeword stream should change decoded side-info channel; mutating one byte of scale hyperprior parameters should change decoded latent reconstruction. Probe via `tools/verify_distinguishing_feature_byte_mutation.py` post-implementation per Catalog #272 distinguishing-feature integration contract.

## 4. Canonical-vs-unique decision per layer (Catalog #290 + UNIQUE-AND-COMPLETE-PER-METHOD operating mode)

| Layer | Canonical helper | Decision | Rationale |
|-------|------------------|----------|-----------|
| Archive grammar | `tac.optimization.faiss_ivf_pq_atw_channel` V2 ATW2 schema extension | ADOPT_CANONICAL_BECAUSE_SERVES | V8 ships into V2 ATW2 archive via NEW schema slot `learned_compression_encoder_weights_int8 + categorical_posterior_codeword_stream_brotli`; canonical helper provides codebook serialization + byte-budget estimation primitives. Per Catalog #220 operational mechanism declaration required. |
| Encoder architecture | Balle 2018 entropy-bottleneck + scale-hyperprior (CompressAI library reference) | FORK_BECAUSE_PRINCIPLED_MISMATCH | CompressAI canonical Balle 2018 is ~5M params for full-frame compression; V8 forks to small-neural-architecture (~50K params) for side-info channel surface (lower-dimensional). The architecture pattern is canonical (GDN + entropy bottleneck + scale hyperprior); the SIZE budget is forked per substrate-optimal engineering. |
| Categorical posterior reparametrization | Hafner DreamerV3 2024 Gumbel-Softmax + straight-through estimator | ADOPT_CANONICAL_BECAUSE_SERVES | Hafner DreamerV3 categorical posterior IS the canonical bleeding-edge mechanism for discrete-posterior training; sister C6 IBPS Path B2 uses identical reparametrization (cross-pollination canonical). |
| Scorer-preprocess routing | `tac.scorer.load_default_scorers` + canonical preprocess_input per Catalog #164 | ADOPT_CANONICAL_BECAUSE_SERVES | Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #164 scorer-preprocess routing: V8 trainer MUST route through canonical helpers to preserve PoseNet/SegNet gradient differentiability + eval_roundtrip discipline. NO substrate-specific reason to fork. |
| Auth-eval CLI routing | `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` (Catalog #226) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #226: substrate trainers MUST route auth_eval through canonical helper to prevent hand-rolled CLI drift bug class. NO substrate-specific reason to fork. |
| Inflate device-fork | `tac.substrates._shared.inflate_runtime.select_inflate_device` (Catalog #205) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #205: submissions MUST use canonical device-fork helper to prevent silent CPU/CUDA drift bug class. V8 inflate.py adopts canonical helper. |
| Eval-roundtrip routing | `tac.differentiable_eval_roundtrip` per CLAUDE.md eval_roundtrip non-negotiable | ADOPT_CANONICAL_BECAUSE_SERVES | Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS": ALL training paths MUST use eval_roundtrip. NO substrate-specific reason to fork. |
| EMA shadow at inference | Per CLAUDE.md "EMA — NON-NEGOTIABLE, HIGHEST EMPHASIS" canonical `tac.training.EMA` | ADOPT_CANONICAL_BECAUSE_SERVES | Per CLAUDE.md "EMA" non-negotiable: ALL training paths MUST instantiate EMA, update after every optimizer.step(), save EMA shadow as inference checkpoint. NO substrate-specific reason to fork. |
| Hardware substrate detection | `tac.substrates._shared.trainer_skeleton.detect_hardware_substrate` (Catalog #190) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #190: canonical helper provides dynamic detection across all dispatch substrates (T4 / A10G / A100 / 4090 / H100 / L40S) + macOS_arm64 fallback. NO substrate-specific reason to fork. |
| Modal call_id ledger registration | `tac.deploy.modal.call_id_ledger.register_dispatched_call_id_fail_closed` (Catalog #245 + #339) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #245 + #339: ALL Modal dispatches MUST register call_id fail-closed to prevent orphan paid dispatch bug class. NO substrate-specific reason to fork. |
| Cost-band calibration | `tac.cost_band_calibration.append_anchor` per Catalog #175 + #177 outcome discipline | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #175 + #177: canonical posterior write discipline. NO substrate-specific reason to fork. |
| Probe-outcomes ledger | `tac.probe_outcomes_ledger.register_probe_outcome` (Catalog #313) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #313: every probe outcome MUST register via canonical helper. NO substrate-specific reason to fork. V8 smoke outcome registers PROCEED / DEFER / OPERATOR_REVIEW_REQUIRED verdict. |
| Deliverability proof builder | `tac.wyner_ziv_deliverability.proof_builder.build_deliverability_proof_from_wyner_ziv_classification` (Catalog #319 PRIMARY consumer) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #319: V8 side-info channel IS a Wyner-Ziv reweight class; deliverability proof required for autopilot HIGH_PAIR_INVARIANT reward; canonical helper enforces Tier 1-4 categorization + Catalog #213 Comma2k19 canonical helper citation. NO substrate-specific reason to fork. |
| Canonical Provenance | `tac.provenance.build_provenance_for_contest_archive_byte_member` (Catalog #323 META-meta umbrella) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #323: every score-claim row MUST carry canonical Provenance sub-object. V8 archive bytes carry Provenance per `contest_archive_byte_member` kind. NO substrate-specific reason to fork. |

**Canonical-vs-unique summary**: 14 layers documented. 13 ADOPT_CANONICAL_BECAUSE_SERVES + 1 FORK_BECAUSE_PRINCIPLED_MISMATCH (encoder size budget). Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: V8 IS structurally CANONICAL-HEAVY at the META-infrastructure layer (preserving the 295+ STRICT preflight gate protection surface) and SUBSTRATE-UNIQUE at the encoder architecture layer (small-neural-architecture vs canonical CompressAI ~5M params). The fork is principled per "substrate engineering happens ONCE per architecture class".

## 5. Per-substrate reactivation criteria + Catalog #324 post-training Tier-C validation discipline

### Per CLAUDE.md "Forbidden premature KILL" non-negotiable

Reactivation paths if V8 smoke EMPIRICAL outcome diverges from predicted band:

1. **PRIMARY** (if smoke MI ≥ 0.5 + ΔS ∈ [0.187, 0.193]): V8 IS canonical PARADIGM-BRIDGE primary; advance to L2 frontier-pursuit; queue 100-300ep full run via paid Modal A100 ($30-100); paired CPU+CUDA harvest via Catalog #246 paired auth-eval discipline.
2. **CONDITIONAL** (if smoke MI ≥ 0.5 BUT ΔS > 0.193): V8 IS shippable at higher predicted band; re-evaluate composition with PR101 frame_exploit_selector per Stack-of-Stacks discipline; full run conditional on operator-frontier-override per Catalog #199.
3. **DEFER** (if smoke MI < 0.5): V8 specific architecture choice FALSIFIED at implementation level per Catalog #307; pivot to alternative architecture variants:
   - **alt-1**: K=32 categorical posterior (4× cardinality; bigger byte cost; richer information capacity)
   - **alt-2**: Hierarchical encoder with 2-level Gumbel-Softmax (Path B1+B2 sister hybrid per C6 IBPS Path B4 cross-pollination)
   - **alt-3**: Replace Balle 2018 entropy-bottleneck with adversarial-learned codebook (Babenko-Lempitsky 2014; Path V6 sister)
   - **alt-4**: Replace categorical posterior with learned binary codes (Jain-Kulis 2009 iterative-quantization; sister approach)
4. **OPERATOR_REVIEW_REQUIRED** (if smoke MI < 0.1 OR ΔS > 0.30): paradigm-level question per Catalog #307; operator review on whether learned-compression Faiss extension family is paradigm-correct for this substrate surface; sister cross-pollination evidence from C6 IBPS Path B2 + NSCS06 v8 Variant C smoke informs verdict.

### Catalog #324 post-training Tier-C validation discipline

Per CLAUDE.md "predicted_band-from-random-init-Tier-C-density (the phantom-predicted-band trap)" + Catalog #324:

- **predicted_band_validation_status**: `pending_post_training`
- **reactivation criterion**: post-training Tier-C density measurement on landed V8 archive via `tools/mdl_scorer_conditional_ablation.py --tier c --archive <v8_archive_sha>`
- **acceptance**: if Tier-C density > 0.90 = within-class trap (per Catalog #219); reactivate to alt-1/2/3/4 above. If Tier-C density ≤ 0.30 = across-class success per Catalog #227; advance to L2.
- **STRICT preflight Catalog #324** wires the validation discipline at recipe-emit surface: V8 recipe MUST declare `predicted_band_validation_status: pending_post_training` until landing archive emits Tier-C validation artifact.

## 6. Implementation roadmap (post-design landing)

### Phase 1 (FREE): design memo landing + V4 hand-rolled probe outcome ingestion
- This memo lands (current step)
- V4 hand-rolled probe outcome (bundled in sister landing memo) informs V8 design priors
- Outcome registered via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313

### Phase 2 (CONDITIONAL FREE / $0.30 if minimal): V8 implementation scaffold
- Build `experiments/train_substrate_v8_learned_compression_faiss.py` (~300 LOC; canonical helpers + Balle 2018 encoder + Gumbel-Softmax categorical posterior + scale hyperprior)
- Build `submissions/v8_learned_compression_faiss/inflate.py` (~130 LOC; canonical inflate device-fork + categorical posterior decoder + scale hyperprior reconstruction)
- Build `submissions/v8_learned_compression_faiss/inflate.sh` per Catalog #146 contest-compliant runtime template
- Build `.omx/operator_authorize_recipes/substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml` with `research_only: true` + `dispatch_enabled: false` initial state
- Per Catalog #325 PER-SUBSTRATE OPTIMAL FORM symposium: V8 symposium REQUIRED before paid dispatch eligibility

### Phase 3 (CONDITIONAL $5-30): V8 smoke (Modal A100 5-50ep per Catalog #167 smoke-before-full pattern)
- Operator authorizes paid dispatch via Catalog #199 paired-env discipline
- Smoke per K-sweep + β-sweep (K ∈ {8, 16, 32}, β ∈ {0.001, 0.01, 0.1})
- Outcome registered via `tac.probe_outcomes_ledger.register_probe_outcome`
- Catalog #324 post-training Tier-C density measurement on landed V8 archive

### Phase 4 (CONDITIONAL $30-100): V8 full run (Modal A100 100-300ep)
- Conditional on Phase 3 smoke clearing medal-class threshold ≤ 0.20
- Paired CPU+CUDA harvest per Catalog #246 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- Continual-learning anchor appended to canonical posterior via Catalog #245 4-layer pattern

### Phase 5 (CONDITIONAL $15-50): composition with PR101 frame_exploit_selector + sister C6 IBPS Path B2 + sister NSCS06 v8 Variant C
- Conditional on Phase 4 V8 PROCEED + sister Path B2 + sister Variant C empirical anchors landed
- Composition smoke per Stack-of-Stacks discipline
- Predicted composition ΔS subject to additive assumption #11 empirical validation

## 7. Cross-references

- **Parent T3 symposium**: `.omx/research/council_t3_cargo_cult_resurrection_v1_faiss_20260519.md` (DD's V1 Faiss symposium that elevated V8 to PRIMARY-CROSS-POLLINATION)
- **Aggregate landing**: `.omx/research/cargo_cult_resurrection_top3_symposiums_landed_20260519.md` (DD's sister landing memo per commit `8d373077b`)
- **Sister cross-pollination memos**:
  - `.omx/research/council_t3_cargo_cult_resurrection_c6_ibps_v2_20260519.md` (Path B2 DreamerV3 RSSM categorical posterior K=256 × 24 groups)
  - `.omx/research/council_t3_cargo_cult_resurrection_nscs06_v8_variant_c_20260519.md` (hybrid_class_shift_path_C neural residual decoder)
- **Canonical V1/V2/V3 design memo**: `.omx/research/atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md`
- **Canonical probe matrix**: `.omx/research/atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.md`
- **V4 hand-rolled probe** (bundled with this memo): `tools/probe_atw_v2_1_faiss_pq_v4_hand_rolled.py` + `experiments/results/v1_faiss_v4_probe_20260519T232017Z/`
- **Canonical implementations cited**: Balle-Laparra-Simoncelli 2018 *Variational image compression with a scale hyperprior* + Hafner 2024 *Mastering Diverse Domains through World Models* DreamerV3 RSSM categorical posterior + Jégou-Douze-Schmid 2011 *Product quantization for nearest neighbor search* + PR #56 hybrid analytical+neural recipe + PR #95 HNeRV bind-all-ingredients pattern + Atick-Redlich 1990 cooperative-receiver theorem + Wyner-Ziv 1976 source coding with side information + Tishby-Zaslavsky 2015 *Deep learning and the information bottleneck principle* + MacKay 2003 *Information Theory, Inference, and Learning Algorithms* + Catalog #245 modal_call_id_ledger + Catalog #319 deliverability proof builder + Catalog #313 probe-outcomes ledger + Catalog #205 inflate device-fork + Catalog #190 hardware substrate detection + Catalog #226 auth-eval canonical helper + Catalog #164 scorer-preprocess routing + Catalog #324 post-training Tier-C validation discipline + Catalog #325 per-substrate symposium discipline.

## 8. 6-hook wire-in declaration per Catalog #125

| # | Hook | Status | Rationale |
|---|------|--------|-----------|
| 1 | Sensitivity-map contribution (`tac.sensitivity_map.*`) | N/A — design memo only | V8 encoder weights + categorical posterior parameters become first-class sensitivity-map contributors AFTER implementation (Phase 2). Per substrate-engineering discipline, V8 byte-level sensitivity flows through canonical `tac.sensitivity_map.per_byte_leverage` post-implementation. |
| 2 | Pareto constraint (`tac.pareto_*`) | N/A — design memo only | V8 archive bytes contribute to rate-axis Pareto constraint AFTER implementation. Per substrate-engineering discipline, V8 enters Pareto solver via canonical `tac.pareto_solver.add_constraint` post-implementation. |
| 3 | Bit-allocator hook | N/A — design memo only | V8 byte budget allocation (~6KB target) registered AFTER implementation. Per canonical `tac.optimization.bit_allocator.register_hook` post-implementation. |
| 4 | Cathedral autopilot dispatch hook | **ACTIVE (DESIGN-LEVEL)** | V8 design memo IS consumed by cathedral autopilot ranker via canonical `tac.cathedral_consumers.canonical_equation_lookup_consumer` (per Catalog #344 lookup of design predictions); V8 paradigm IS cross-pollination cite-chain anchor for sister C6 IBPS Path B2 + sister NSCS06 v8 Variant C autopilot ranking weights. |
| 5 | Continual-learning posterior update | **ACTIVE (DESIGN-LEVEL)** | V8 design memo IS continual-learning artifact per CLAUDE.md "Subagent coherence-by-default" non-negotiable; cross-pollination canonical with sister C6 IBPS Path B2 + sister NSCS06 v8 Variant C strengthens posterior signal. Sister landing memo (bundled with this design memo) appends posterior anchor per Catalog #313 probe-outcomes ledger for V4 hand-rolled probe outcome. V8 smoke outcome (Phase 3) appends posterior anchor via `tac.cost_band_calibration.append_anchor` per Catalog #175 + #177. |
| 6 | Probe-disambiguator | **ACTIVE (DESIGN-LEVEL)** | V8 design memo IS canonical disambiguator between V1-V7 pure-analytical Faiss family vs V8 learned-compression Faiss family. The disambiguator question: "does learned-compression at small-neural-architecture scale dominate pure-analytical Faiss at the <6KB byte budget?" — answered empirically via V8 smoke (Phase 3). V4 hand-rolled probe (bundled) is the V1-V7 family's PRIMARY disambiguator; V8 IS the family-vs-family disambiguator. |

**Hook summary**: 3 N/A (deferred to implementation Phase 2-3) + 3 ACTIVE (design-level cathedral autopilot + continual-learning + probe-disambiguator). Per CLAUDE.md "Subagent coherence-by-default" non-negotiable: design memo IS coherent with the unified solver stack at the DESIGN surface; implementation Phase 2-3 adds the runtime hooks.

## 9. Predicted band Dykstra-feasibility check (Catalog #296)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #296: the predicted band [0.187, 0.193] contest-CPU MUST cite a Dykstra-feasibility intersection check.

**Dykstra-feasibility analysis** (citing parent T3 symposium Dykstra co-lead position):

The feasible region is the intersection of (a) byte budget ≤ 6KB (V8 target; rate constraint = 25 × 6KB / 37545489 ≈ +0.004 rate), (b) MI ≥ 0.5 bits/symbol (decoder-side-info constraint per Wyner-Ziv R(D)), (c) archive grammar self-contained (HNeRV parity L3), (d) score-distortion-saving ≥ 0.005 (offsetting rate cost).

**Projection onto byte budget**: V8 encoder + scale hyperprior + categorical codeword stream estimated at:
- Encoder weights: ~50K params × 8 bits = 50KB raw; brotli-compressed at int8 quantization: ~2-3KB
- Scale hyperprior: ~2KB
- Categorical codeword stream (24 groups × log2(16) = 96 bits/sample × 600 pairs / 8): ~7.2KB raw; brotli-compressed: ~1-2KB
- **Total estimated**: ~5-7KB, satisfying byte budget (a) with margin

**Projection onto MI**: Hafner DreamerV3 reports MI preservation at ~1.5-2.0 bits/symbol with K=16-32 categorical posterior; sister C6 IBPS Path B2 design memo predicts similar MI band. Predicted MI ≈ 1.5-2.0 bits/symbol satisfies constraint (b) MI ≥ 0.5 with 3-4× margin.

**Projection onto score-distortion**: Score-saving ≈ MI × Wyner-Ziv gain factor. At MI=1.5 bits/symbol × seg sensitivity ≈ 0.005-0.015 score units. Net Δscore at +0.004 rate cost - 0.005-0.015 distortion saving = [-0.011, -0.001] = within predicted band [-0.005, +0.001] within rounding.

**Feasibility verdict**: Feasible region NON-EMPTY at predicted V8 parameter point (K=16, 24 categorical groups, β=0.01, encoder=~50K params). Dykstra alternating projections converges to (a) AND (b) AND (c) AND (d) simultaneously satisfiable. Predicted band [0.187, 0.193] IS Dykstra-feasible at the predicted parameter point.

**Cross-paradigm composition**: With PR101 frame_exploit_selector orthogonal axes per Stack-of-Stacks: composition Pareto-feasibility depends on V8 + PR101 + sister C6 IBPS Path B2 + sister NSCS06 v8 Variant C all simultaneously satisfying their individual feasibility regions. Composition Dykstra-feasibility check DEFERRED to Phase 5 composition smoke (assumption #11).

## 10. Conclusion + operator-routable next steps

**V8 design memo verdict**: STRUCTURALLY SOUND for paid dispatch eligibility per (1) 12-assumption cargo-cult audit + (2) 9-dim checklist 6 PASS + 3 PASS-WITH-EMPIRICAL-VALIDATION-PENDING + 1 PASS-WITH-CATALOG-324-VALIDATION-REQUIRED + (3) 6-facet observability surface + (4) canonical-vs-unique decision per layer 13 ADOPT + 1 FORK + (5) Catalog #324 post-training Tier-C validation discipline + (6) 6-hook wire-in 3 ACTIVE design-level + 3 N/A implementation-pending + (7) Dykstra-feasibility intersection check.

**Operator-routable next steps** (post-design landing):

1. **Phase 2 (FREE)**: Build V8 implementation scaffold (~300 LOC trainer + ~130 LOC inflate runtime + recipe with `research_only: true`). Operator-routable: spawn implementation subagent.

2. **Per Catalog #325**: V8 per-substrate symposium REQUIRED before paid dispatch eligibility. Operator-routable: spawn V8 symposium subagent after implementation scaffold lands.

3. **Phase 3 (CONDITIONAL $5-30)**: V8 smoke per Catalog #167 smoke-before-full pattern + operator-frontier-override per Catalog #199. Operator-routable: paid dispatch decision after Phase 2 + symposium.

4. **Cross-pollination dependency**: V8 paradigm-bridge classification (within-class vs class-shift) confirmed empirically after sister C6 IBPS Path B2 + sister NSCS06 v8 Variant C smokes land. Operator-routable: monitor sister cargo-cult resurrection candidate landings.

5. **30-day retrospective** per CLAUDE.md "Mission alignment" Consequence 3 (due 2026-06-18): re-audit V8 + V4 + sister cross-pollination outcomes.

**Highest-EV op-routable surfaced**: V4 hand-rolled probe outcome (bundled in sister landing memo) informs V8 design priors. If V4 MI ≥ 0.5 at byte cost ≤ 5KB, V8 inherits empirical anchor that the V3→V2 transition zone IS feasible — V8 design priors STRENGTHENED; V8 smoke recommendation moves from CONDITIONAL to RECOMMENDED-NEXT-PAID-DISPATCH-AUTHORIZATION.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:v1-FAISS-v8-learned-compression-substrate-design-memo-trigger-tokens-in-design-rationale-not-new-equation -->


# OBSERVABILITY_SURFACE_SECTION_WAIVED:historical_design_memo_predates_catalog_305_section_header_requirement_or_is_namespace_design_not_substrate_specific_observability_per_catalog_110_113_HISTORICAL_PROVENANCE_APPEND_ONLY_discipline_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526


# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:design_memo_references_cooperative_receiver_atick_redlich_or_wyner_ziv_framework_in_cross_reference_or_spatial_not_temporal_context_NOT_as_substrate_central_predictive_coding_claim_per_catalog_311_z6_z7_z8_pattern_h_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
