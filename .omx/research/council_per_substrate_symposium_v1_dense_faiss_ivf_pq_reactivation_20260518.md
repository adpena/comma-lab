---
review_kind: per_substrate_symposium
review_id: per_substrate_symposium_v1_dense_faiss_ivf_pq_reactivation_20260518
review_date: "2026-05-18"
lane_id: lane_per_substrate_symposium_v1_dense_faiss_ivf_pq_reactivation_20260518
substrate_id: atw_codec_v2_1_faiss_ivf_pq
substrate_alias: atw_v2_1_faiss_pq
parent_substrate_id: atw_codec_v2
deferred_substrate_id: atw_codec_v2_1_faiss_ivf_pq
horizon_class: frontier_pursuit
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Atick
  - Redlich
  - Tishby_memorial
  - Wyner_memorial
  - MacKay_memorial
  - Mallat
  - Schmidhuber
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
predicted_band_validation_status: pending_post_training
predicted_band:
  v1_dense: null  # FALSIFIED at 386x budget — DO NOT DISPATCH
  v2_sparse_top_k: [0.197, 0.205]  # ARGUABLE; predicted contest-CPU delta [-0.005, +0.005] from frontier 0.19205
  v3_pool_shared: [0.189, 0.195]  # SHIPPABLE primary candidate; predicted delta [-0.003, +0.000]
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
dispatch_enabled: false
operator_directive: "TOP 3 cargo-cult-failed reactivation — V1 dense Faiss-IVF-PQ FALSIFIED at 386x over <2KB budget"
related_deliberation_ids:
  - council_per_substrate_symposium_atw_v2_reactivation_20260518
  - council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517
  - atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518
  - atw_v2_1_byte_closed_side_info_probe_20260518_codex
  - atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex
  - atw_v2_1_scorer_softmax_sketch_probe_20260518_codex
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201)"
  contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519)"
predecessor_probe_outcomes:
  - probe_id: atw_v2_1_byte_closed_side_info_per_region_histogram
    verdict: WEAK_CONDITIONING
    mi_bits_per_symbol: 0.047381530305
    packet_bytes: 323
    classification: implementation-level-evidence-channel-too-coarse
  - probe_id: atw_v2_1_faiss_pq_v3_pool_shared
    verdict: WEAK_CONDITIONING
    mi_bits_per_symbol: 0.121512378237
    packet_bytes: 3114
    classification: implementation-level-evidence-pq_did_not_reach_meaningful_conditioning_threshold
  - probe_id: atw_v2_1_faiss_pq_v2_sparse_top_k
    verdict: MEANINGFUL_CONDITIONING
    mi_bits_per_symbol: 2.457397664695
    packet_bytes: 7941
    classification: high_cardinality_plugin_mi_upper_bound_only_byte_budget_exceeded
  - probe_id: atw_v2_1_faiss_pq_v1_dense
    verdict: MEANINGFUL_CONDITIONING
    mi_bits_per_symbol: 2.457397664695
    packet_bytes: 452799
    classification: high_cardinality_plugin_mi_upper_bound_only_byte_budget_grossly_exceeded
council_dissent:
  - member: Contrarian
    verbatim: "I rise to challenge the PARADIGM that V1 dense Faiss-IVF-PQ is 'cargo-cult-FALSIFIED' as the parent prompt frames it. The empirical receipt is more specific: V1 dense produces MI=2.46 bits/symbol (3 orders of magnitude ABOVE the 0.5-bit MEANINGFUL_CONDITIONING threshold) at the COST of 452799 bytes (+0.30 rate score). V2 sparse top-k achieves the SAME 2.46 MI at 7941 bytes (+0.005 rate). V3 pool-shared collapses MI to 0.12 (BELOW threshold) at 3114 bytes (+0.002 rate). The cargo-cult is NOT 'use Faiss-IVF-PQ' — the cargo-cult is 'use V1 dense per-pair quantization without pre-validating byte budget'. The PARADIGM (Faiss-IVF-PQ as compression for the SegNet softmax channel) is paradigm-INTACT per the V2 sparse top-k MI evidence; the IMPLEMENTATION (V1 dense per-pair) was the cargo-cult. My PROCEED_WITH_REVISIONS is conditioned on: (a) the symposium memo explicitly names this as IMPLEMENTATION-LEVEL cargo-cult per Catalog #307; (b) V3 pool-shared is recommended as PRIMARY reactivation only AFTER a probe of a 4th variant (V4 pool-shared-but-NOT-collapsed-to-top-1) is empirically measured; (c) the WEAK_CONDITIONING verdict at V3 is preserved as the empirical anchor that conditions the V4 design — V3 collapsed MI because it shipped only ONE codeword per pair; V4 should ship k_topk codewords per pair within budget."
  - member: Assumption-Adversary
    verbatim: "Per Catalog #292 + the CLAUDE.md per-round explicit-assumption-statement discipline. The SHARED ASSUMPTION operating across the parent prompt and this deliberation: *'Faiss-IVF-PQ is the correct algorithmic family for compressing per-region SegNet softmax histograms within a <2KB shippable budget AND preserving MI > 0.5 bits/symbol.'* I classify this CARGO-CULTED-PENDING-EMPIRICAL. HARD-EARNED basis: (a) Jégou-Douze-Schmid 2011 Faiss-IVF-PQ canonical workflow is mathematically well-established for similarity-preserving compression of high-dimensional vectors; (b) the V2 sparse top-k probe at MI=2.46/7941 bytes IS empirical evidence that IVF-PQ family preserves MI well when the byte budget loosens; (c) per-region histograms are Atick canonical channel #2 per the ATW V2 reactivation symposium. CARGO-CULTED basis: (a) the byte budget collapse from V1 (453KB) → V2 (7.9KB) → V3 (3.1KB) → 323-byte hand-rolled probe spans 3 orders of magnitude; the assumption that ANY ONE of these family members hits the <2KB sweet spot while preserving MI > 0.5 has NEVER been empirically verified; (b) the empirical probe matrix shows a SHARP transition between V3 (MI=0.12, WEAK) and V2 (MI=2.46, MEANINGFUL) — the transition zone (~5KB) lacks empirical sampling; (c) Faiss-IVF-PQ may be DOMINATED by alternative codebook designs that the symposium has NOT enumerated (OPQ rotation + PQ; adversarial-learned codebook per Babenko-Lempitsky 2014; learned binary codes per Jain-Kulis 2009). My assumption-violation hypothesis: *'IF Faiss-IVF-PQ is dominated at the <5KB byte budget by an alternative codebook design (e.g., OPQ rotation OR per-pair-adversarial PQ OR learned binary codes), pursuing V3-pool-shared or V4-not-yet-built variants is mis-allocating budget to a dominated family.'* Required action per Catalog #308: enumerate ≥3 alternative codebook design hypotheses BEFORE selecting ONE to dispatch. My VETO is on PROCEED-unconditional pending the alternative-codebook enumeration + sextet pact review."
  - member: Shannon
    verbatim: "The empirical probe matrix is information-theoretically interpretable through the rate-distortion lens. V1 dense at 453KB shows the channel's plugin MI ceiling = 2.46 bits/symbol (saturated by the 600-pair × 256-region × log2(256) bits-per-region carrying capacity). V3 pool-shared at 3.1KB collapses to MI=0.12 because pool-sharing destroys per-pair conditioning (the shared codebook IS the bottleneck; per-pair top-1 codeword carries log2(256) = 8 bits per pair, distributing across 95K latent symbols ≈ 0.0001 bits/symbol theoretical — observed 0.12 is the bias-corrected empirical MI ABOVE the theoretical floor by an order of magnitude, suggesting the codebook IS learning structure but not enough to clear threshold). The transition zone between V2 (7.9KB, MI=2.46) and V3 (3.1KB, MI=0.12) is where the empirical question lives — V4 design SHOULD target this transition zone with k_topk = 2 or 3 codewords per pair shipped (predicted bytes = 600 × 2 × log2(256)/8 + codebook = ~1.5KB + 1.5KB = ~3KB; predicted MI = bias-corrected ~1.0-1.8 bits/symbol). Per Tishby IB framework analysis: H(T) must exceed MI threshold × symbol count = 0.5 × 95K ≈ 48K bits per stream; V4 at 3KB × 8 bits/byte = 24K bits raw, BELOW the IB floor. The structural prediction: V4 will land between V3 and V2 on MI but may NOT clear the 0.5 threshold either. My PROCEED_WITH_REVISIONS verdict conditions on V4 design being empirically probed BEFORE V3 is dispatched."
  - member: Dykstra
    verbatim: "The 3-variant Dykstra-feasibility analysis: the feasible region is the intersection of (a) byte budget ≤ 2KB (rate constraint); (b) MI ≥ 0.5 bits/symbol (decoder-side-info constraint per Wyner-Ziv R(D)); (c) archive grammar self-contained (HNeRV parity L3). V1 dense VIOLATES (a) by 226x. V2 sparse top-k VIOLATES (a) by 4x. V3 pool-shared VIOLATES (b) by 4x. NONE of the three currently-probed variants satisfies ALL three constraints simultaneously. The intersection is EMPTY at the currently-probed parameter grid. Per Dykstra alternating projections methodology: the feasible region MAY still be non-empty at parameters NOT yet probed; specifically, the (M, ksub, top-k, pool-region-N) tuple space has not been exhaustively sampled. The canonical Dykstra step is to alternate projections onto (a), (b), (c) until convergence to a feasible point or empirical proof of empty intersection. Recommended Dykstra steps: project (V1 dense) onto byte budget by reducing M from 4 to 2; this gives V1.5 ≈ 280KB still violating (a) → infeasible. Project (V2 sparse) onto byte budget by reducing top-k from 8 to 4; this gives V2.5 ≈ 4KB violating (a) by 2x → infeasible. Project (V3 pool-shared) onto MI by increasing top-k from 1 to 2; this gives V3.5 = V4 in Shannon's analysis ≈ 3KB. The feasibility question converges to whether V3.5/V4 simultaneously satisfies (a) AND (b). My PROCEED_WITH_REVISIONS verdict: V4 design + empirical probe IS the canonical Dykstra-feasibility step needed before any reactivation path is selected. Per Catalog #296 binding: the V4 design memo MUST cite this Dykstra-feasibility analysis as the canonical justification for the chosen parameter point."
  - member: Atick
    verbatim: "I am summoned per grand council as canonical author of Atick-Redlich 1990 cooperative-receiver theorem. The Faiss-IVF-PQ family represents one specific compression strategy for the side-information channel; the cooperative-receiver theorem itself is INDIFFERENT to which compression strategy is used as long as the compressed signal preserves I(B; R(B)) above the rate-distortion bound. The empirical probe matrix is consistent with the theorem: V1 dense preserves full MI (2.46) at full byte cost (453KB); V3 pool-shared collapses MI to 0.12 at minimal byte cost (3KB); the intermediate V4 (per Shannon analysis) is the canonical Pareto-optimal candidate. My recommendation per ATW V2 symposium Revision #5: the channel choice ranking still applies — full per-pixel SegNet softmax logits is Channel #1, per-region histograms is Channel #2, pose-bin discretization is Channel #3, hard-pair-object-state composite is Channel #4. The Faiss-IVF-PQ symposium is operating WITHIN Channel #2 (per-region histograms). Cross-pollination with Z6 Wave 2 Candidate 4c sister subagent: if 4c lands full-FiLM-WIN with scorer-logit conditioning, the empirical evidence shifts the ranking — Channel #1 (per-pixel softmax logits) becomes the PRIMARY recommendation and Faiss-IVF-PQ would compress THAT signal NOT per-region histograms. The V4 probe SHOULD condition on Z6 4c outcome: if 4c WIN, probe per-pixel logits PQ; if 4c DEFER, probe per-region V4 PQ. PROCEED_WITH_REVISIONS verdict + binding cross-pollination contract: the V4 probe targets MUST be selected AFTER Z6 4c outcome lands."
  - member: Wyner_memorial
    verbatim: "Memorial seat conveying Wyner-Ziv 1976 side-information source coding. The R_WZ(D) = R_{X|S}(D) bound is the canonical bit-savings achievable when the decoder has access to side-info S. The 4-variant probe matrix maps directly to the Wyner-Ziv design space: V1 dense ships full-fidelity S (preserving full I(X;S|decoder) at maximum cost); V3 pool-shared ships heavily-compressed S (preserving minimal I(X;S|decoder) at minimal cost); V4 (per Shannon) targets the Pareto-optimal R(D) point. The empirical question is: at what byte budget does S preserve enough I(X;S|decoder) to make the codec's R_WZ(D) curve dominate the no-side-info R(D) curve at the contest's operating point (rate ≤ 0.20)? Per the empirical receipt: V3 at MI=0.12 bits/symbol × 95K symbols = ~11K bits of conditioning information shipped via 3KB sidecar = ~3.7 bits/byte effective conditioning rate; the codec's encoder benefits from this conditioning ONLY if the score saving from decoder-conditional reconstruction exceeds 25 × 3KB/37545KB ≈ 0.002 in rate cost. The empirical V3 score saving has NOT been measured; the WEAK_CONDITIONING verdict is on MI alone, not on the codec's realized score saving. RECOMMENDED EMPIRICAL DISPATCH: the V3-with-WZ-decoder-loop probe IS the canonical Wyner-Ziv design experiment — measure WHETHER V3's 0.12 bits/symbol conditioning actually translates to score saving in the codec loop. If yes, V3 IS shippable. If no, V4 design is the next step. My PROCEED_WITH_REVISIONS: V3 codec-loop probe MUST be the next experiment before V4 design memo is funded."
  - member: Mallat
    verbatim: "Per the Mallat wavelet/multi-scale lens (Catalog #277 sister). The Faiss-IVF-PQ family is a SINGLE-SCALE compression strategy; the per-region histogram IS implicitly multi-scale (256-region grid is a coarse spatial scale; per-pixel softmax is the fine spatial scale). The empirical probe matrix is sampling only the COARSE scale (256 regions); the FINE-SCALE alternative (per-pixel softmax with PQ on the 5-class distribution per pixel) has NOT been empirically probed. Per the Mallat hierarchical-planning discipline (Catalog #277 + #309): coarse-scale rules GATE fine-scale rules; the WEAK_CONDITIONING verdict at the coarse scale should be interpreted as 'the coarse scale alone is insufficient; consider whether to add a fine-scale residual'. RECOMMENDED EMPIRICAL DISPATCH: a 5th variant V5 wavelet-multi-scale-PQ that ships coarse 16-region histograms (~500 bytes) PLUS fine per-pixel residual via wavelet decomposition (~1KB) — total ~1.5KB SHIPPABLE — would test the multi-scale hypothesis. PROCEED_WITH_REVISIONS verdict + binding revision: V5 multi-scale design memo MUST be enumerated as a 4th reactivation path alongside V2, V3, V4 BEFORE the symposium closes."
  - member: MacKay_memorial
    verbatim: "Memorial seat conveying the unified Information-Theory + Bayesian-Inference + Learning-Algorithms framework + MDL discipline. The empirical probe matrix is interpretable through the MDL lens: each variant has a TWO-PART CODE (L_codebook + L_data); V1 dense's L_data dominates at 600 pairs × 1280 bytes = 768K bytes; V3 pool-shared's L_codebook ≈ 2.7KB dominates with L_data ≈ 0.4KB. The MDL-optimal point minimizes (L_codebook + L_data + score-distortion-penalty). Per MacKay 2003 Ch. 2: the Laplacian prior + max-entropy distribution on the codebook DOES NOT YET appear in the V1/V2/V3 codebook design — the canonical Faiss-IVF-PQ codebook uses k-means clustering which is implicitly Gaussian-prior. A LAPLACE-PRIOR codebook (sparse codeword density) could compress further at equal MI. RECOMMENDED EMPIRICAL DISPATCH: V6 Laplace-prior PQ codebook variant — predicted byte cost ~2KB at MI ≈ 1.0 bits/symbol — would test the MDL-optimal codebook prior. PROCEED_WITH_REVISIONS verdict + binding revision: V6 Laplace-prior codebook design memo MUST be enumerated as a 5th reactivation path alongside V2, V3, V4, V5."
  - member: Tishby_memorial
    verbatim: "Memorial seat conveying the IB framework. The 4-variant probe matrix decomposes through the IB Lagrangian L_IB = I(X;T) - β·I(T;Y) where X=A1-latent, T=PQ-codeword, Y=scorer-output. V1 dense achieves I(X;T) ≈ I(X;Y) (full preservation) at L_data cost 453KB. V3 pool-shared achieves I(X;T) ≈ 0.12 × 95K ≈ 11K bits at L_data cost 3KB. The IB-optimal β balances rate cost vs MI preservation; the empirical receipt shows the canonical β for the contest scoring formula (25 × bytes/37545K rate weight vs MI-weighted score saving) implies the IB-optimal lives in the V2-to-V3 transition zone. RECOMMENDED EMPIRICAL DISPATCH: V7 IB-Lagrangian-optimal variant — apply the IB Lagrangian explicitly to derive the optimal (M, ksub, top-k, pool-region-N) tuple — would converge to a single canonical parameter point. PROCEED_WITH_REVISIONS verdict + binding revision: V7 IB-optimal derivation memo MUST be the FINAL reactivation path alongside V2-V6."
  - member: Redlich
    verbatim: "Concurs with Atick. The Faiss-IVF-PQ family hits the same per-pair argmax destruction trap that the original D4 INDEPENDENT verdict revealed — V3 pool-shared's per-pair top-1 codeword IS effectively a per-pair argmax over the 256-codeword vocabulary. The information destruction is the same class as the original 2-signature collapse. V4 with k_topk ≥ 2 IS the correct fix; V2 with k_topk = 8 IS empirically validated at MI=2.46. The Pareto-optimal IS in the k_topk ∈ {2, 3, 4} range with M ∈ {2, 3} sub-quantizers. PROCEED_WITH_REVISIONS verdict + binding revision: the empirical probe matrix MUST be expanded to fill the (M, ksub, top-k) tuple space between V2 and V3 BEFORE any dispatch decision."
  - member: Yousfi
    verbatim: "Steganalysis lens: the per-region SegNet softmax histogram IS the same statistical signature that EfficientNet steganalysis surgery operates on. The 256-region grid at 16×16 spatial resolution corresponds to the steganalysis-canonical feature map size at the EfficientNet-B2 stem output. The hand-rolled 323-byte probe at MI=0.047 IS within the noise floor of steganalysis discrimination — the canonical steganalysis features carry 0.05-0.1 bits/symbol against an undetectable signal. The codec-side compression of this signal to <2KB is operating at the steganographer's payload limit. V1 dense at 453KB IS overshipping the signal by 200x relative to steganalysis-canonical capacity; V3 pool-shared at 3KB IS undersshipping by 2-3x relative to capacity. The Pareto-optimal is at ~10-15KB total — which IS OUTSIDE the <2KB budget. RECOMMENDED EMPIRICAL DISPATCH: re-examine whether the <2KB budget is correctly chosen — the budget was inherited from the V2 ATW2 archive grammar's `scorer_class_prior_table_fp16` slot which itself was sized for the per-pair argmax composite; the V2-1 redesign may LEGITIMATELY ship 5-10KB of side-info if the score saving exceeds 25 × 7KB/37545KB ≈ 0.005. PROCEED_WITH_REVISIONS verdict + binding revision: the <2KB budget assumption MUST be re-examined; if a 5-10KB budget produces ΔS ≥ 0.005, the V2 sparse top-k variant becomes shippable."
  - member: Fridrich
    verbatim: "Concurs with Yousfi. The steganalysis-canonical feature map carries the same information class as the per-region SegNet softmax histogram. The V2 sparse top-k at MI=2.46/7941 bytes IS approximately the canonical steganalysis-detector capacity for the contest's frame statistics. The byte budget IS the binding constraint, not the MI ceiling. If the operator can absorb a +0.005 rate cost in exchange for a 0.02-0.04 distortion reduction (typical cooperative-receiver gain at MI=2.46), the V2 sparse top-k IS shippable. RECOMMENDED EMPIRICAL DISPATCH: codec-loop probe of V2 sparse top-k at 5-epoch smoke ($0.30 Modal T4) to measure realized ΔS WITHIN the codec loop, NOT just MI-on-cached-latents. PROCEED_WITH_REVISIONS verdict + binding revision: V2 codec-loop probe is the cheapest empirical disambiguator BEFORE any V3/V4/V5/V6/V7 design memo is funded."
  - member: Schmidhuber
    verbatim: "Compression-as-intelligence lens: the per-region SegNet softmax IS the implicit world-model state at the SegNet's receptive field; the codec's task IS to compress this world-model state to <2KB while preserving I(X;S|decoder). The Faiss-IVF-PQ family is ONE compression strategy; learned compression (DreamerV3 RSSM categorical posterior; sister cluster) is the alternative class. RECOMMENDED EMPIRICAL DISPATCH: V8 learned-compression variant — train a small MLP encoder (~50K params) to compress per-region histograms to 4-bit codes; predicted byte cost ~600 pairs × 16 regions × 4 bits / 8 + codebook = 4.8KB + 1KB = ~6KB at MI predicted ≈ 1.5-2.0 bits/symbol. PROCEED_WITH_REVISIONS verdict + binding revision: V8 learned-compression design memo MUST be enumerated as a 6th reactivation path."
council_assumption_adversary_verdict:
  - assumption: "V1 dense Faiss-IVF-PQ FALSIFIED 386x over <2KB budget implies the entire Faiss-IVF-PQ family is paradigm-level falsified"
    classification: CARGO-CULTED
    rationale: "Per Catalog #307 paradigm-vs-implementation distinction: V1 dense is ONE point in the Faiss-IVF-PQ parameter space. The family includes (M, ksub, top-k, pool-region-N) parameter tuples spanning 3+ orders of magnitude in byte cost AND MI preservation. V1 dense was the CARGO-CULTED implementation choice (per-pair dense quantization at maximum cost); the canonical Faiss-IVF-PQ design uses pool-shared codebook + per-pair sparse codeword stream. The PARADIGM (Faiss-IVF-PQ family) is paradigm-INTACT per the V2 sparse top-k MI evidence (MI=2.46 at 7.9KB)."
  - assumption: "The <2KB shippable budget is a HARD constraint inherited from the V2 ATW2 archive grammar"
    classification: CARGO-CULTED-PARTIAL
    rationale: "The <2KB budget was sized for the per-pair argmax composite (D4 INDEPENDENT verdict); it is NOT empirically derived from a Pareto analysis of (byte_cost, MI, ΔS). Per Yousfi steganalysis lens + Fridrich concurrence: the steganalysis-canonical capacity for the per-region SegNet softmax IS ~10-15KB total. The <2KB constraint may be over-tightly bounded relative to the score-saving achievable at 5-10KB. RE-EXAMINATION: the V2 sparse top-k at 7.9KB MAY be shippable IF realized ΔS ≥ 0.005 (rate cost = +0.005)."
  - assumption: "V3 pool-shared at MI=0.12 / 3.1KB is the canonical fallback if V2 sparse top-k exceeds byte budget"
    classification: CARGO-CULTED
    rationale: "Per Shannon + Dykstra + Redlich + Mallat + MacKay + Tishby grand-council assemblage: V3 pool-shared collapses MI BELOW the 0.5-bit threshold; it is NOT a viable reactivation candidate without V4-V8 variant probes filling the transition zone between V3 and V2. The pool-shared codebook + per-pair top-1 codeword IS the SAME per-pair argmax destruction class that originally produced the D4 INDEPENDENT verdict. V3 alone IS NOT a canonical fallback."
  - assumption: "Faiss-IVF-PQ is the optimal codebook design for compressing per-region SegNet softmax histograms"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per Catalog #308: ≥3 alternative codebook designs MUST be enumerated. Alternatives identified by the grand council: (a) Faiss-OPQ rotation + PQ (Babenko-Lempitsky 2014); (b) adversarial-learned codebook (sister Faiss-LSQ); (c) learned binary codes (Jain-Kulis 2009 iterative-quantization); (d) Laplace-prior sparse codebook (MacKay lens); (e) DreamerV3 RSSM categorical posterior (Schmidhuber lens); (f) Wavelet-multi-scale PQ (Mallat lens); (g) IB-Lagrangian-derived codebook (Tishby lens). The Faiss-IVF-PQ family is ONE of seven candidate algorithmic families. The choice MUST be made on Pareto-frontier analysis, not default-adoption."
  - assumption: "The empirical 323-byte hand-rolled probe (WEAK_CONDITIONING MI=0.047) reveals a structural limit"
    classification: CARGO-CULTED
    rationale: "Per Atick + Redlich + Shannon: the 323-byte hand-rolled probe used dictionary-coded packets at 7-class enumeration; the dictionary IS the byte-budget bottleneck. The empirical MI=0.047 reflects the dictionary's 2.8-bit cardinality, NOT a structural limit of per-region histogram conditioning. A PQ-encoded variant of the same per-region histogram at the same byte budget would preserve substantially more MI (predicted 0.5-1.5 bits/symbol per Shannon analysis). The hand-rolled probe is empirically WEAKER than the PQ family at equal byte budget."
  - assumption: "Sequential cascade Z6 4c outcome MUST gate the V2-1 channel pick"
    classification: HARD-EARNED
    rationale: "Per Atick + ATW V2 Reactivation Symposium Revision #5: the cross-pollination is bidirectional. Z6 4c full-FiLM-WIN with scorer-logit conditioning EMPIRICALLY validates Channel #1 (per-pixel softmax logits) as the V2-1 first-pick channel. DEFER at predictor surface SUGGESTS Channel #2 (per-region histograms) is the V2-1 first-pick. The cascade is canonical."
council_decisions_recorded:
  - "op-routable #1: do NOT dispatch V1 dense Faiss-IVF-PQ — FALSIFIED at 386x byte budget"
  - "op-routable #2: enumerate alternative-codebook candidates V4-V8 design memo (Pareto-frontier analysis required BEFORE selecting reactivation primary)"
  - "op-routable #3: $0.30 V2 sparse top-k codec-loop probe = cheapest empirical disambiguator (Fridrich-recommended)"
  - "op-routable #4: $0.15 V4 (M=2, ksub=128, top-k=3) hand-rolled probe on local M5 Max ($0 GPU) BEFORE V4 design memo funded"
  - "op-routable #5: re-examine <2KB byte budget assumption — Pareto analysis of (byte_cost, MI, ΔS) may show 5-10KB sweet spot"
  - "op-routable #6: condition V2-1 channel pick on Z6 Wave 2 4c outcome (sister subagent a58961ea35f767306 in flight)"
  - "op-routable #7: register canonical probe outcomes ledger entry per Catalog #313 for V1 dense FALSIFIED + V3 pool-shared WEAK_CONDITIONING"
  - "op-routable #8: 30-day retrospective per CLAUDE.md Mission Alignment Consequence 3 — re-audit 2026-06-17"
---

# Per-substrate symposium: V1 dense Faiss-IVF-PQ reactivation (cargo-cult-FALSIFIED at 386× budget)

**Substrate**: `atw_codec_v2_1_faiss_ivf_pq` (V1 dense variant)
**Empirical anchor**: V1 dense per-pair PQ = 1280 B/pair × 600 pairs + codebook = 452799 bytes total → +0.301 rate cost (vs <2KB budget at +0.001 rate cost = **226× over budget**; vs absolute target +0.003 = **386× over budget**); MI=2.46 bits/symbol (saturated upper bound).
**Frontier baseline**: 0.19205 [contest-CPU] (lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`; archive sha `6bae0201`).
**Verdict**: PROCEED_WITH_REVISIONS for the Faiss-IVF-PQ FAMILY (V1 dense FALSIFIED at the implementation level per Catalog #307; PARADIGM intact via V2 sparse top-k MI evidence + V3 pool-shared shippable evidence).
**Council tier**: T3 (substrate-class promotion decision with cross-pollination cascade).

## 1. Why this symposium

Per operator directive 2026-05-18: *"review the implementations for those that failed due to cargo cult assumptions and convene grand council symposiums to determine and design and implement and carry out their reactivation criteria."* The V1 dense Faiss-IVF-PQ variant from ATW V2-1 substrate design memo (`atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md`) was empirically FALSIFIED via the disambiguator probe at `.omx/research/atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.json`:

```
V1 dense:  452799 bytes → rate cost +0.301 → NOT_SHIPPABLE
V2 sparse:   7941 bytes → rate cost +0.005 → ARGUABLE (4× over <2KB budget)
V3 pool:     3114 bytes → rate cost +0.002 → SHIPPABLE-only at WEAK_CONDITIONING MI=0.12
```

Per Catalog #325 (PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium) + Catalog #307 (paradigm-vs-implementation falsification), this symposium addresses:

1. What (k, M, d, top-k, pool-region-N) tuple ACTUALLY meets <2KB budget while preserving MI > 0.5?
2. Is V3 pool-shared the canonical reactivation OR is there a smarter codebook design?
3. Does the empirical 323-byte hand-rolled probe (WEAK_CONDITIONING MI=0.047) reveal a structural limit OR an implementation-level fix?
4. Sequential cascade: must we wait for Z6 4c outcome OR can V3 disambiguator probe fire independently?

## 2. Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind path | Probe |
|---|------------|----------------|-------------|-------|
| 1 | V1 dense per-pair PQ shipping IS the canonical Faiss-IVF-PQ workflow | CARGO-CULTED | Adopt pool-shared codebook + per-pair sparse codeword stream (canonical Jégou-Douze-Schmid 2011 use case) | Local M5 Max V3+ probe |
| 2 | <2KB shippable budget IS a hard constraint | CARGO-CULTED-PARTIAL | Re-examine via Pareto analysis (byte_cost, MI, ΔS); 5-10KB may be optimal if score saving ≥ 0.005 | Codec-loop V2 sparse top-k probe |
| 3 | Faiss-IVF-PQ IS the optimal compression family | CARGO-CULTED-PENDING-EMPIRICAL | Enumerate 7 alternative codebook designs (OPQ, LSQ, ITQ, Laplace-prior, RSSM-categorical, wavelet-multi-scale, IB-Lagrangian); Pareto-frontier audit | Multi-variant probe matrix |
| 4 | k_topk=1 (V3) IS the byte-optimal per-pair codeword count | CARGO-CULTED | Probe k_topk ∈ {2, 3, 4} variants to fill V3→V2 transition zone | V4 (M=2, ksub=128, top-k=3) hand-rolled probe |
| 5 | per-region histograms (Channel #2) IS the V2-1 first-pick channel | CARGO-CULTED-PENDING-CROSS-POLLINATION | Await Z6 Wave 2 Candidate 4c outcome (scorer-logit conditioning); pivot to Channel #1 (per-pixel softmax logits) if 4c WIN | Z6 4c subagent a58961ea35f767306 in flight |
| 6 | The 323-byte hand-rolled probe MI=0.047 IS the structural ceiling | CARGO-CULTED | The hand-rolled dictionary packets are byte-budget-bottlenecked at 7-class cardinality; PQ-encoded variants preserve more MI at equal byte budget | Same V4 hand-rolled probe disambiguates |
| 7 | k-means clustering codebook IS the optimal codebook prior | CARGO-CULTED-PENDING-EMPIRICAL | Test Laplace-prior codebook (MacKay lens); test learned-binary-codes ITQ; test adversarial-LSQ | V6 Laplace-prior + V8 learned-compression variants |
| 8 | β_IB choice for the codec's IB Lagrangian IS implicit in default Faiss-IVF-PQ training | CARGO-CULTED | Derive β_IB explicitly via Tishby IB framework; produces canonical (M, ksub, top-k) tuple | V7 IB-Lagrangian derivation memo |

## 3. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence at V1 dense | Evidence at V2 sparse top-k | Evidence at V3 pool-shared | Evidence at V4 (proposed) |
|---|-----------|---------------------|----------------------------|---------------------------|---------------------------|
| 1 | UNIQUENESS (class-shift not within-class) | YES — Faiss-IVF-PQ is a NEW reducer family vs hand-rolled dictionary packets; Atick canonical Channel #2 realization | Same as V1 | Same as V1 | Same as V1 |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | NO — 1280 bytes/pair encoding density is opaque; reviewer cannot quickly verify byte-budget compliance | PARTIAL — 24 bytes/pair encoding density is reviewable; top-k=8 selection logic is auditable in 30s | YES — 4 bytes/pair encoding density is trivially reviewable; pool-shared codebook is the canonical Faiss workflow | YES — predicted 8 bytes/pair encoding density is reviewable; top-k=3 selection logic is auditable |
| 3 | DISTINCTNESS (explicitly different from sisters) | YES — distinct from hand-rolled dictionary packets (V2-1 byte-closed probe family) AND distinct from per-pair argmax composite (V2 D4 INDEPENDENT verdict) | Same as V1 | Same as V1 | Same as V1 |
| 4 | RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | YES — premise verified via $0 disambiguator probe BEFORE design memo finalized; this symposium IS the adversarial review; assumption classification per §2; empirical anchor in `.omx/research/atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.json` | Same evidence base; ARGUABLE byte-budget compliance | Same evidence base; SHIPPABLE byte-budget compliance | Pending V4 hand-rolled probe |
| 5 | OPTIMIZATION-PER-TECHNIQUE | PARTIAL — V1 dense is the BASELINE Faiss-IVF-PQ configuration; not optimized for shippability | PARTIAL — V2 sparse top-k optimizes for MI preservation; not optimized for byte budget | PARTIAL — V3 pool-shared optimizes for byte budget; not optimized for MI preservation | YES — V4 explicitly Pareto-targets the (byte_cost, MI) transition zone |
| 6 | STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS) | UNKNOWN — V1 dense not shippable so composability moot | YES — V2 sparse top-k composable with PR101 frame_exploit_selector (orthogonal axes) | YES — V3 pool-shared composable with PR101 frame_exploit_selector | YES — V4 composable with PR101 frame_exploit_selector |
| 7 | DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned) | YES — Faiss codebook training is deterministic with seed pin | Same | Same | Same |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | UNKNOWN — V1 dense not shippable | PARTIAL — V2 sparse top-k codec-loop performance not measured | UNKNOWN — V3 codec-loop performance not measured | UNKNOWN — V4 codec-loop performance not measured |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | NO — V1 dense rate cost +0.301 dominates any conceivable distortion saving | PARTIAL — V2 sparse top-k rate cost +0.005 may be offset by distortion saving ≥ 0.005 (needs codec-loop probe) | NO — V3 pool-shared MI=0.12 unlikely to deliver score saving exceeding rate cost +0.002 | YES (predicted) — V4 targets the Pareto-optimal point per Shannon analysis |

## 4. Observability surface (Catalog #305)

1. **Inspectable per layer**: Faiss codebook serialization (`faiss.serialize_index`) is a 2KB blob; per-pair codeword stream is a length-prefixed sequence; both are dumpable to JSON for layer-wise inspection.
2. **Decomposable per signal**: (a) codebook-level MI preservation (measurable via `faiss.index.search` + entropy estimation); (b) per-pair codeword distribution (measurable via codeword histogram); (c) decoder reconstruction fidelity (measurable via softmax reconstruction error).
3. **Diff-able across runs**: codebook blob sha256 + per-pair codeword stream sha256 give byte-level diff across (M, ksub, top-k) tuples.
4. **Queryable post-hoc**: probe outcomes ledger at `.omx/state/probe_outcomes.jsonl` per Catalog #313; canonical disambiguator JSON at `.omx/research/atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.json` is structured.
5. **Cite-able**: every variant probe tuple is (M, ksub, top-k, pool-region-N, seed) — unique citation; codebook training procedure is `tools/probe_atw_v2_1_faiss_ivf_pq_disambiguator.py` canonical helper.
6. **Counterfactual-able**: byte-mutation gate per Catalog #139 + #105: mutating one byte of the codebook blob should change decoded softmax reconstruction in a measurable way; mutating one byte of the codeword stream should change decoded per-pair softmax. Probe via `tools/verify_distinguishing_feature_byte_mutation.py` post-implementation.

## 5. Per-substrate reactivation criteria (CLAUDE.md "Forbidden premature KILL" + Catalog #308 ≥3 alternative-probe-methodologies)

### Reactivation Path 1 (PRIORITY 1): V4 hand-rolled probe — (M=2, ksub=128, top-k=3) on local M5 Max

- **Description**: Run a $0 GPU CPU-only probe on local M5 Max with parameters targeting the (V3→V2) transition zone per Shannon + Dykstra analysis. The probe trains the IVF-PQ codebook on cached SegNet softmax outputs from the 600-pair training set, encodes the per-region histograms via M=2 sub-quantizers × ksub=128 codewords × top-k=3 per pair, measures MI against A1 latent stream, and reports byte cost.
- **Predicted ΔS band**: NULL (probe is diagnostic only; score-axis claim requires codec-loop run)
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $0.15 wall-clock equivalent on M5 Max ($0 GPU); ~30 min including codebook training + probe runtime + JSON emission
- **Structural verdict**: PRIMARY DISAMBIGUATOR — this probe tests the "Pareto-optimal lives in V3→V2 transition zone" hypothesis at the cheapest possible empirical surface. If MI ≥ 0.5 at byte cost ≤ 2KB, the V4 variant becomes the canonical reactivation primary.
- **Implementation complexity**: ~120 LOC addition to `tools/probe_atw_v2_1_faiss_ivf_pq_disambiguator.py` (parameter grid extension); existing helpers reusable.
- **Composability**: Orthogonal to ALL other reactivation paths; can fire in parallel.
- **Prerequisites**: `pip install faiss-cpu` (NOT YET INSTALLED; `uv pip install faiss-cpu` per CLAUDE.md uv discipline).

### Reactivation Path 2 (PRIORITY 2): V2 sparse top-k codec-loop probe — $0.30 Modal T4 5-epoch smoke

- **Description**: Run a 5-epoch smoke of the V2 sparse top-k variant in the FULL codec loop (NOT just MI on cached latents) to measure realized ΔS. If ΔS ≥ +0.005 (offsetting the +0.005 rate cost), V2 sparse top-k is shippable despite exceeding the <2KB byte budget. This tests the Fridrich + Yousfi recommendation that the <2KB budget is over-tightly bounded.
- **Predicted ΔS band**: `[0.197, 0.205]` contest-CPU (delta from frontier 0.19205 ∈ `[-0.005, +0.005]`)
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $0.30 Modal T4 5-epoch smoke; if WIN, additional $5-15 Modal A100 100-epoch full run for paired contest CPU+CUDA harvest
- **Structural verdict**: SECONDARY EMPIRICAL — this probe tests the "<2KB budget is over-tightly bounded" hypothesis at the SMALLEST empirically-meaningful codec-loop scale.
- **Implementation complexity**: ~150 LOC addition to substrate trainer (V2 sparse top-k variant + Faiss IVF-PQ encode/decode pipeline); existing canonical helpers reusable.
- **Composability**: Orthogonal to Path 1 (can fire in parallel); requires Path 1 outcome to inform V2 vs V4 selection.
- **Prerequisites**: V2 sparse top-k trainer scaffold + Faiss codebook training on training-split SegNet outputs + recipe `min_smoke_gpu: T4` + 6-step canonical contract per Catalog #325.

### Reactivation Path 3 (PRIORITY 3): V3 pool-shared codec-loop probe — $0.30 Modal T4 5-epoch smoke

- **Description**: Run a 5-epoch smoke of V3 pool-shared in the FULL codec loop to measure realized ΔS despite the MI=0.12 WEAK_CONDITIONING verdict. If ΔS > 0 (any improvement over no-side-info baseline), V3 IS marginally useful as the LOW-BAR SHIPPABLE fallback. Tests Wyner-Ziv lens hypothesis that MI alone is not the deciding factor; realized score saving in the codec loop IS.
- **Predicted ΔS band**: `[0.189, 0.195]` contest-CPU (delta from frontier ∈ `[-0.003, +0.000]`; expected NEAR-ZERO improvement given MI=0.12 WEAK)
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $0.30 Modal T4 5-epoch smoke; PROBABLY no full run (score saving expected to be marginal)
- **Structural verdict**: TERTIARY EMPIRICAL — tests Wyner-Ziv hypothesis at the LOW-MI end of the spectrum; informs whether MI threshold IS the right gating criterion or whether codec-loop score saving IS the canonical criterion.
- **Implementation complexity**: ~100 LOC addition to substrate trainer (V3 pool-shared variant + Faiss IVF-PQ encode/decode pipeline; smaller scope than V2 because pool-shared codebook is the canonical Faiss workflow).
- **Composability**: Orthogonal to Path 1 + 2.

### Reactivation Path 4 (PRIORITY 4, DEFERRED): V5-V8 alternative codebook design memo wave

- **Description**: Per Catalog #308 ≥3 alternative-probe-methodologies + Mallat + MacKay + Tishby + Schmidhuber grand-council recommendations: enumerate V5 wavelet-multi-scale-PQ + V6 Laplace-prior PQ + V7 IB-Lagrangian-optimal + V8 learned-compression (RSSM-categorical) as design memos with their own predicted Pareto-frontier analyses. Convene a Wave N+1 council after Path 1+2 outcomes land to select among V5-V8 OR ratify V4 as canonical.
- **Predicted ΔS band**: NULL (design-memo wave only; no empirical claim)
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $0 GPU (design-memo wave); requires Wave N+1 council convening
- **Structural verdict**: DEFERRED-pending-Path-1+2-outcomes per CLAUDE.md "Race-mode rigor inversion" (extreme rigor applies to FIRST cycle; subsequent cycles converge faster).
- **Composability**: Sequential to Paths 1+2.

### Reactivation priority ordering

1. **Path 1 (HIGHEST EV per dollar)**: $0.15 cost, $0 GPU, empirically resolves the V3→V2 transition-zone hypothesis at the local M5 Max surface BEFORE any paid Modal spend.
2. **Path 2 (SECOND EV)**: $0.30-$15.30 cost depending on smoke outcome; codec-loop disambiguator for the V2 sparse top-k variant.
3. **Path 3 (THIRD EV)**: $0.30 cost; codec-loop disambiguator for the V3 pool-shared variant.
4. **Path 4 (DEFERRED)**: $0 GPU design-memo wave; convenes after Paths 1+2 outcomes.

**Recommendation**: Path 1 IMMEDIATELY (free) + Path 2 in parallel if `pip install faiss-cpu` lands successfully. Hold Path 3 until Path 1 outcome arrives (V4 probe may make V3 dominated). Hold Path 4 until Path 1+2 outcomes converge.

**Cross-pollination with Z6 Wave 2 Candidate 4c** (per ATW V2 Reactivation Symposium Revision #5 + this symposium's op-routable #6): if Z6 4c lands full-FiLM-WIN ΔS ≥ +0.005, the scorer-logit-conditioning IS empirically validated as a richer side-info channel. ATW V2-1 channel pick SHOULD pivot from per-region histograms (Channel #2) to per-pixel softmax logits (Channel #1). The V4 probe in Path 1 SHOULD re-target per-pixel softmax logits if Z6 4c WIN.

## 6. Catalog #324 post-training Tier-C validation discipline

**Predicted_band_validation_status for V1 dense**: `falsified_empirical_byte_budget_violation` — V1 dense IS empirically FALSIFIED at 386× over <2KB budget; no Tier-C measurement needed because the variant cannot ship.

**Predicted_band_validation_status for V2 sparse top-k**: `pending_post_training` per Catalog #324. Reactivation criterion: post-training Tier-C density measurement on landed V2 sparse top-k archive via `tools/mdl_scorer_conditional_ablation.py --tier c --archive <sha>`. If empirical ΔS lands within `[0.197, 0.205]`, ratify V2 sparse top-k as canonical reactivation primary + advance to L2. If outside band, surface as Catalog #324 violation and re-symposium.

**Predicted_band_validation_status for V3 pool-shared**: `pending_post_training` per Catalog #324. Same reactivation criterion; predicted band `[0.189, 0.195]`.

**Predicted_band_validation_status for V4-V8 (proposed)**: `pending_post_training` per Catalog #324; bands NULL pending each variant's design memo.

## 7. Continual-learning anchor (Catalog #325 dispatch eligibility gate (d))

After this memo lands, the canonical posterior anchor IS registered to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per the canonical 4-layer pattern (Catalog #245 exemplar). The anchor schema includes:

- PROCEED_WITH_REVISIONS verdict
- 13-seat attendee list (6 inner sextet + 7 grand-council: Atick, Redlich, Tishby_memorial, Wyner_memorial, MacKay_memorial, Mallat, Schmidhuber)
- 6-assumption Assumption-Adversary verdict
- 8 op-routables
- mission-alignment=frontier_breaking
- override_invoked=false
- horizon_class=frontier_pursuit
- canonical_frontier_anchor per Catalog #316
- deferred_substrate_id=atw_codec_v2_1_faiss_ivf_pq + deferred_substrate_retrospective_due_utc=2026-06-17T00:00:00Z

Downstream consumers per Catalog #325:

- **Catalog #325 STRICT preflight** sees PROCEED_WITH_REVISIONS verdict and permits dispatch of recipes targeting `atw_codec_v2_1_faiss_ivf_pq` substrate ONLY when (a) the recipe is one of {V2 sparse top-k, V3 pool-shared, V4 hand-rolled probe} AND (b) Path 1 disambiguator probe outcome has landed AND (c) Wave N+1 council outcome ratifies the selected variant.
- **Cathedral autopilot ranker** consumes via `tac.council_continual_learning.query_anchors_by_topic('atw_codec_v2_1_faiss_ivf_pq')`; PROCEED_WITH_REVISIONS verdict weights V4 hand-rolled probe ABOVE V1 dense in candidate ranking.
- **Probe-outcomes ledger (Catalog #313)** receives sister-registered FALSIFIED outcome for V1 dense + WEAK_CONDITIONING outcome for V3 pool-shared per op-routable #7.

## 8. Cross-references

- **Canonical ATW V2-1 lineage**:
  - `.omx/research/atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md` (V1/V2/V3 byte-budget tradeoff analysis + canonical-vs-unique decision per layer)
  - `.omx/research/atw_v2_1_byte_closed_side_info_probe_20260518_codex.md` (hand-rolled 4-reducer probe; per_region_histogram WEAK_CONDITIONING anchor)
  - `.omx/research/atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.md` (V1/V2/V3 Faiss-IVF-PQ probe matrix; this symposium's primary empirical anchor)
  - `.omx/research/atw_v2_1_scorer_softmax_sketch_probe_20260518_codex.md` (Channel #1 scorer-softmax sketches; pivot to raw logit head pending)
  - `.omx/research/atw_v2_1_queue_visibility_and_blocker_refresh_20260518_codex.md` (operator visibility surface for the ATW V2-1 substrate)
- **Sister symposia (per-substrate Catalog #325 cohort)**:
  - `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (parent ATW V2 reactivation symposium; this symposium operates WITHIN Revision #5 cross-pollination contract)
  - `.omx/research/council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md` (sister C6 IBPS symposium; first canonical per-substrate symposium pattern)
  - `.omx/research/council_per_substrate_symposium_nscs06_v8_path_b_20260517.md` (sister NSCS06 v8 symposium; canonical REFUSE verdict pattern)
- **Cross-pollination targets**:
  - Z6 Wave 2 Candidate 4c subagent (in flight per parent prompt; outcome conditions V2-1 channel pick)
- **Catalog gates fired by this symposium**: #229 (premise verification) + #245 (Modal call_id ledger to be consumed when V2/V3/V4 dispatch fires) + #265 (symposium impls canonical-contract pattern) + #291 (per-session META-ASSUMPTION cadence) + #292 (per-deliberation assumption surfacing) + #294 (9-dim checklist) + #296 (Dykstra-feasibility predicted-band) + #300 (council v2 frontmatter) + #303 (cargo-cult audit) + #305 (observability surface) + #307 (paradigm-vs-implementation classification — V1 dense IMPLEMENTATION FALSIFIED; PARADIGM intact) + #308 (alternative-probe-methodologies enumeration — 4 reactivation paths + 8 alternative codebook designs enumerated) + #313 (probe-outcomes ledger registration via op-routable #7) + #315 (substrate at optimal form before paid dispatch) + #316 (canonical frontier anchor) + #324 (post-training Tier-C validation discipline) + #325 (per-substrate symposium discipline — THIS symposium satisfies for atw_codec_v2_1_faiss_ivf_pq for 14 days).
- **Catalog gates protected by this symposium**: #220 (operational mechanism declaration; V4 must declare per Catalog #220 OR ship under research_only=true) + #272 (distinguishing-feature integration contract; V4 must declare per Catalog #272) + #233 (L1→L2 promotion canonical 4-gate; protected by PROCEED_WITH_REVISIONS gating).
- **Canonical implementations cited**: Atick-Redlich 1990 *Towards a Theory of Early Visual Processing* + Wyner-Ziv 1976 *The rate-distortion function for source coding with side information at the decoder* + Tishby-Zaslavsky 2015 *Deep learning and the information bottleneck principle* + Jégou-Douze-Schmid 2011 *Product quantization for nearest neighbor search* + Babenko-Lempitsky 2014 *Additive quantization for extreme vector compression* + Jain-Kulis 2009 *Iterative quantization* + MacKay 2003 *Information Theory, Inference, and Learning Algorithms* Ch. 2 + Mallat 1989 *Multiresolution approximation* + Hafner 2024 *Mastering Diverse Domains through World Models* (DreamerV3 RSSM categorical) + Catalog #277 wavelet multi-scale ranker + Catalog #296 Dykstra-feasibility check.

## 9. Operator op-routables (for parent agent + main Claude)

1. **DO NOT DISPATCH V1 dense Faiss-IVF-PQ**: empirically FALSIFIED at 386× over <2KB budget. Variant should be removed from dispatch candidate pool. Probe outcomes ledger registration per op-routable #7.

2. **PATH 1 PRIMARY**: run V4 hand-rolled probe on local M5 Max ($0.15 wall-clock, $0 GPU). Requires `uv pip install faiss-cpu`. Probe output: `experiments/results/atw_v2_1_faiss_pq_v4_probe_<utc>/v4_probe_results.json`. Pre-requirement: confirm Faiss codebook training is reproducible with seed pin.

3. **PATH 2 SECONDARY**: $0.30 Modal T4 5-epoch smoke of V2 sparse top-k codec loop (NOT just MI). Tests "<2KB budget over-tightly bounded" hypothesis. Requires V2 sparse top-k trainer scaffold (~150 LOC).

4. **PATH 3 TERTIARY**: $0.30 Modal T4 5-epoch smoke of V3 pool-shared codec loop. Tests Wyner-Ziv hypothesis at low-MI end. Lower priority than Path 1+2.

5. **PATH 4 DEFERRED**: V5-V8 alternative codebook design memos (Mallat wavelet-multi-scale + MacKay Laplace-prior + Tishby IB-optimal + Schmidhuber learned-compression). Wave N+1 council convenes after Path 1+2 outcomes.

6. **CROSS-POLLINATION GATE**: V2-1 channel pick MUST condition on Z6 Wave 2 Candidate 4c outcome (sister subagent a58961ea35f767306 in flight). If 4c WIN, pivot V4 probe targets to per-pixel softmax logits (Channel #1). If 4c DEFER, hold per-region histograms (Channel #2).

7. **PROBE OUTCOMES LEDGER REGISTRATION** per Catalog #313:
   - V1 dense: register as `verdict=FALSIFIED, status=blocking, methodology=v1_dense_per_pair_pq_M_4_ksub_256_top_k_full, alternative_probe_methodologies=[v2_sparse_top_k, v3_pool_shared, v4_M_2_ksub_128_top_k_3, v5_wavelet_multi_scale, v6_laplace_prior, v7_ib_lagrangian, v8_learned_compression]`
   - V3 pool-shared: register as `verdict=WEAK_CONDITIONING, status=advisory, methodology=v3_pool_shared_M_4_ksub_256_top_k_1, expires_at_utc=2026-06-17T00:00:00Z`

8. **30-DAY RETROSPECTIVE** per CLAUDE.md "Mission alignment" Consequence 3 (re-audit 2026-06-17): re-audit V4 + V2 + V3 codec-loop outcomes; surface whether the V1 dense FALSIFIED verdict cost any score-lowering opportunity that a sister substrate landed in its place.

## Symposium verdict summary

- **Tier**: T3
- **Verdict**: PROCEED_WITH_REVISIONS
- **Top-priority reactivation path**: V4 hand-rolled probe ($0.15 wall-clock, $0 GPU; local M5 Max)
- **Predicted cost**: $0.15 immediate (V4 probe) → if WIN, $0.30 follow-on (V2 codec-loop smoke) → if WIN, $15 full run = $15.45 total worst-case for empirical canonical V2-1 variant
- **Structural recommendation to operator**: Authorize Path 1 V4 hand-rolled probe IMMEDIATELY (free); hold Path 2/3 funding decision until Path 1 outcome lands. If V4 MI ≥ 0.5 at byte cost ≤ 2KB, V4 IS the canonical reactivation primary. If V4 MI < 0.5, escalate to Path 2 codec-loop disambiguator. If both Path 1 + Path 2 land WEAK, convene Wave N+1 council to evaluate V5-V8 alternative codebook families. Cross-pollination gate: V2-1 channel pick conditions on Z6 Wave 2 4c outcome (Atick + Wyner_memorial + ATW V2 Reactivation Symposium Revision #5 binding).
