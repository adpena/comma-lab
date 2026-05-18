# Ultimate stacking — grand-council symposium + repo audit + eureka moments — 2026-05-17

**Status:** $0 RESEARCH SYNTHESIS — operator directive 2026-05-17 *"scouring the repo and codebase and grand council symposium on any other stacking and composing and stack of stacks and stack of and vstack techniques we can use for extreme optimization; also doing any research and follow up research and eureka moments and shower thoughts on ultimate stacking; we love these $0 optimizations"*
**Lane:** `lane_ultimate_stacking_research_eureka_moments_20260517` (pre-register pending)
**Sister of:** master-gradient symposium memo `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md` — that memo names 8 master-gradient uses; this memo names the 12 stacking primitives that compose under those uses

## §0 — The unified stacking framework (Venn diagram + master gradient)

Per the symposium §2.2 Venn diagram analysis, every archive byte falls into one of FOUR regions:

| region | flows into | confidence of additive stacking | example bytes |
|---|---|---|---|
| **SegNet-only** | frame_1 below stride-2 stem blindspot | HIGH (orthogonal to PoseNet by construction) | SABOR boundary pixels; per-class equivalence index |
| **PoseNet-only** | frame_0 modifications (SegNet doesn't see frame_0) | HIGH (orthogonal to SegNet by archive grammar) | fec6 selector.bin (107 bytes), fec3-7 sister overlays |
| **Joint** | both frames, both scorers | LOW (cross-terms via operating-point shift) | decoder.bin (91% of archive), latents.bin, poses.bin |
| **Rate-only** | encoding bytes (not decoded values) | HIGH (no scorer effect) | magic codec selection; L5 Wyner-Ziv pose deltas; arithmetic-recode |

**The frontier-breaking lever per Venn analysis:** carve NEW disjoint regions OUT OF the joint region by archive-grammar redesign. Every byte you move from JOINT → orthogonal (SegNet-only / PoseNet-only / rate-only) becomes additive-stackable with bounded confidence.

## §1 — Existing stacking primitives in the codebase (15 enumerated)

Audit of `tac.packet_compiler.*` + `submissions/*` + `feedback_*_landed_*` memos + lane registry:

| # | name | byte region | byte cost | empirical ΔS | code anchor |
|---|---|---|---|---|---|
| 1 | **fec3/fec4/fec5/fec6 selector overlays** | PoseNet-only | 107-150 bytes | fec6 anchored at -0.001 vs PR101 GOLD on CPU | `tools/build_pr101_frame_exploit_selector_packet.py` |
| 2 | **PR106 format0a/b/c/d additive corrections** | Joint (latents) | 361-884 bytes | -0.024 CUDA vs PR101 GOLD; CPU-axis sister untested | `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575` |
| 3 | ***Magic codec*** per-stream auto-selector | Rate-only | 0 bytes (replaces existing encoding) | -2 to -7% rate per stream empirically on PR106 | `tac.packet_compiler.*` |
| 4 | **L5 Wyner-Ziv pose deltas** | Rate-only (replaces poses.bin) | -2,800 bytes vs qpose14 | predicted -0.008 to -0.015 | `src/tac/optimization/l5_*.py` |
| 5 | **SABOR boundary classification** | SegNet-only (new section) | +1-2 KB | predicted -0.005 to -0.010 | `tac.symposium_impls.sabor_renderer_atick_redlich` |
| 6 | **Wave-Ω-V2 stack** (water-fill + arithmetic + ADMM) | Joint (decoder.bin) | varies | -0.078 empirical on Lane Ω archive | `tac.optimization.water_fill_bit_budget` |
| 7 | **B1/B2/B3/B4 composition cells** | Joint (substrate × codec cross-product) | varies | composition_alpha-dependent | `tools/build_admm_x_lossy_coarsening_path_b_step6.py` etc. |
| 8 | **U-DIE-KL substrate-wide loss** | Operating-point shift (not pure stacking) | 0 bytes (loss-function change) | predicted -0.005 to -0.020 | `src/tac/losses/u_die_kl.py` |
| 9 | **Quantizr 5-stage staircase** | Training-time only (not archive bytes) | 0 bytes | enables higher-quality decoder | `tac.training.quantizr_5_stage_staircase` (subagent C in flight) |
| 10 | **PR101 grammar 0x02 paired runtime** | Joint (decoder.bin) | varies | ~$0.10-0.50 dispatched | Lane J landed per `feedback_J_PR101_grammar_0x02_landed` |
| 11 | **owv3_0120_orthogonal_stack** | historical SUB-1.000 CUDA champion (replaced) | varies | 0.9974 [contest-CUDA] | superseded |
| 12 | **NeRV mask codec (Lane 12 v2 NeRV-as-renderer)** | Joint (renderer slot replacement) | varies | DEFERRED-pending-renderer-rescope per HNeRV parity L5 | `lane_12_v2_nerv_as_renderer` |
| 13 | **Lane Ω Joint-ADMM coordinator** | Joint (cross-stream) | varies | Boyd-style alternating projections | `tac.lane_omega_joint_admm` |
| 14 | **Substrate composition matrix** (Catalog #227) | Meta-layer over all cells | 0 bytes | composition_alpha-aware ranking | `.omx/state/substrate_composition_matrix.json` |
| 15 | **Hessian-block-FP × magic codec** (B1 Cell 4) | Joint (decoder.bin codec) | varies | landed inflate adapter | `submissions/magic_codec_pr106_r2/` |

## §2 — EUREKA + shower-thought ideas for NEW $0 stacking primitives (12 enumerated)

**Operator: each is $0 build (deterministic, no training risk), byte-disjoint where possible, and stackable per the Venn framework.**

### A. Recursive selector stacking (fec6 + fec7)
fec6 is a K=16 selector overlay on PR101's frame_0. **Insight:** apply ANOTHER per-pair selector on top of fec6's OUTPUT frames. fec7 = K=8 (or K=16) secondary selector picking from {none, fec6_mode_amplify_2x, fec6_mode_amplify_4x, frame0_blue_chroma_shift_post_fec6, frame0_red_chroma_shift_post_fec6, frame0_combined_post_fec6, ...}. Byte cost: another ~100 bytes via fixed-Huffman or magic codec. Byte region: PoseNet-only (still frame_0). Predicted: -0.0005 to -0.002 (diminishing returns; the per-pair pose-distortion floor is finite).
**Cost:** $0 build + $0.30 paired auth eval.

### B. Mask-layer × renderer-layer dual selector
PR101 has masks.mkv (AV1) + renderer's `frame_1` output. **Insight:** apply selector overlays at BOTH layers. Mask-layer selector picks from {default, edge_smooth, edge_dilate, class_K_amplify} per frame (~50 bytes); renderer-layer is current fec6. Byte-disjoint. Mask-layer selector is SegNet-orthogonal IF only frame_0 mask is modified (per Quantizr's "only odd-frame masks encoded; even-frame masks warped" pattern). Predicted: -0.001 to -0.003.
**Cost:** $0 build + $0.30 paired auth eval. **Requires:** read of PR101 mask encoder path to confirm SegNet sees only frame_1 mask.

### C. Per-pixel-class stack (boundary + interior + correction)
SABOR's natural per-pixel-class split (boundary 3-5% vs interior 95-97%). **Insight:** stack THREE layers within the SegNet-only Venn region: (1) boundary pixels at full RGB (1.5 KB), (2) interior equivalence-class index (0.3 KB), (3) per-pair correction stream for pixels that flip class at the boundary between (1) and (2). Stack-of-stacks within SegNet-only.
**Cost:** $0 build; couples to op-routable #2 (SABOR landing) — could be subagent C-V2 (deferred).

### D. Per-tensor-class magic codec (cross-product expansion)
Current magic codec selects ONE codec per STREAM. **Insight:** evaluate codec PER PARAMETER GROUP within decoder.bin: Conv1 weights (LZMA-best), BN gamma (single-tensor AC-best), FiLM bias (Brotli-best), depthwise separable conv (block-FP-best). Per-tensor-class magic codec is a strict superset of per-stream magic codec. Empirically PR106 sister analyses showed 2-7% additional rate savings via this cross-product.
**Cost:** $0 build extending `tac.packet_compiler.*` selector to per-tensor granularity; $0.30 paired auth eval.
**This is the most likely highest-EV pure-$0 win beyond op-routable #9.**

### E. Hierarchical Wyner-Ziv pose stacking
L5 Wyner-Ziv encodes pair-N pose as delta against ego-motion predictor from pair-(N-1). **Insight:** make it HIERARCHICAL — pair-N delta against pair-(N-1) which is itself delta against pair-(N-2). Compounding side-info reduction per Rao-Ballard predictive-coding hierarchy. Predicted savings: another 30-50% on top of L5's already-reduced poses.bin.
**Cost:** $0 build (extends L5 implementation); couples to op-routable #6 (L5 landing).

### F. Magic codec on selector.bin itself
fec6's 107-byte selector overlay is currently compacted via fixed-Huffman (static codebook fit to empirical mode distribution). **Insight:** run the magic codec on the selector bytes themselves. Sub-byte stack-of-stacks. The selector mode distribution may be better compressed by arithmetic coding with adaptive context model. Predicted: -5 to -20 bytes (-3 to -13 ppm = -3e-8 to -1.3e-7 ΔS).
**Cost:** $0 build; trivial 1-line change to selector packet builder. **Land in-context if context permits.**

### G. Codec composition DAG (multi-level)
Build a DAG of codec applications: PR101 → fec6 selector → magic codec on decoder → L5 Wyner-Ziv on poses → SABOR boundary stream → magic codec on FEC6 stream. Each node is byte-disjoint from siblings. Total predicted: -0.015 to -0.025 ΔS from baseline 0.19205 = **predicted 0.167-0.177 [contest-CPU, council-consensus]**. The DAG IS the campaign plan §2-§4.
**Cost:** the full campaign ($245-490 envelope per the revised budget).

### H. Cross-axis opportunistic stack (probably non-compliant; document for completeness)
fec6 is CPU-fitted; format0d is CUDA-fitted. **Shower thought:** ship BOTH archives in submission with metadata flag (axis_target). Contest scorer evaluates exactly one. If host bot generates only ONE of {CPU, CUDA} score per submission, this is ineffective. If it generates both and the lower wins, this is the cleanest CPU+CUDA frontier capture. **Verdict per contest compliance audit:** contest packet schema is single-archive-per-submission per PR101 GOLD canonical contract; this is structurally non-compliant. **Surface for record only; don't implement.**

### I. Inflate-time score-aware deletion stack
At inflate time, the inflate.py loads decoder.bin and renders 1200 frames. **Insight:** identify which decoder weights have `|G[byte_i, :]·coeffs| < 1e-7` via the master gradient (use #1) and SKIP them during forward pass (e.g., zero them out or skip the entire conv block they belong to). Saves wall-clock without affecting score. Pure rate-axis $0 optimization — but the rate axis IS the byte count of decoder.bin which is unchanged by inflate-time skipping. **Insight refined:** this saves INFLATE-TIME wall-clock (operator efficiency) but doesn't save score. Useful for the 30-min inflate budget gate, not the score directly.
**Verdict:** not a score-stacking primitive; valuable as inflate-time optimization for cost-band feasibility.

### J. Per-pair-feature-class tuple selector
fec6 picks ONE mode per pair from K=16. **Insight:** each pair gets a 2D tuple `(mode_for_segnet_subset, mode_for_posenet_subset)` — effectively K=16² = 256 modes per pair, but transmitted as 2 separate bytes via cross-product Huffman. The two coordinates are nearly independent (because SegNet and PoseNet have disjoint scoring paths per Venn §2.2), so the joint Huffman dictionary is ~2× the marginal. Predicted: -10 to -30 bytes additional savings vs single-mode (the cross-product captures interaction the single mode misses), with -0.001 to -0.003 ΔS pose improvement.
**Cost:** $0 build extending selector enumeration; sister of B (mask × renderer dual selector).

### K. Composition_alpha-aware autopilot ranker
Catalog #227's `substrate_composition_matrix.json` already records per-pair-substrate composition_alpha. **Insight:** the autopilot's rerank_candidates_via_master_gradient (just landed) should consume composition_alpha to predict the cross-term magnitude BEFORE empirical anchoring. Falls under the master-gradient lens already; wire-in lift is ~50 LOC.
**Cost:** $0 build; couples to op-routable #3 (Phase-7 lens) extension.

### L. Magic codec score-aware mode (use #8 of master gradient)
Already documented in symposium §3.6 + campaign §1.7. Replaces magic codec's rate-only selection with rate × master-gradient-weighted selection. Predicted: -0.0005 to -0.001 incremental beyond pure-rate magic codec.
**Cost:** $0 build; needs op-routable #1 (master gradient anchor) first.

## §3 — Ranked $0-or-near-$0 stacking opportunities

By predicted ΔS per $0 (since all are $0 build + $0.30-15 measure):

| rank | primitive | predicted ΔS on top of fec6 (0.19205) | confidence | gating |
|---|---|---|---|---|
| 1 | **D. Per-tensor-class magic codec** | -0.003 to -0.007 | HIGH | independent of op-routable #1; lands cleanly |
| 2 | **op-routable #9** (PR101 + magic codec + fec6) | -0.003 to -0.005 | HIGH | subagent A in flight |
| 3 | **J. Per-pair-feature-class tuple selector** | -0.001 to -0.003 + -10 to -30 bytes | MEDIUM-HIGH | independent; sister of fec6 build |
| 4 | **B. Mask × renderer dual selector** | -0.001 to -0.003 | MEDIUM | requires PR101 mask path audit |
| 5 | **A. Recursive selector stacking (fec7)** | -0.0005 to -0.002 | MEDIUM | independent; trivial extension |
| 6 | **F. Magic codec on selector.bin** | -3e-8 to -1.3e-7 | HIGH | trivial; 1-line change |
| 7 | **C. Per-pixel-class stack within SABOR** | -0.002 to -0.005 incremental over SABOR | MEDIUM | couples to SABOR landing (op-routable #2) |
| 8 | **E. Hierarchical Wyner-Ziv pose stacking** | -0.003 to -0.008 incremental over L5 | MEDIUM | couples to L5 landing (op-routable #6) |
| 9 | **L. Magic codec score-aware mode** | -0.0005 to -0.001 incremental | LOW-MEDIUM | needs op-routable #1 anchor first |
| 10 | **K. Composition_alpha-aware autopilot ranker wire-in** | $0 score gain, but reduces wasted dispatch by ranking better | HIGH | independent autopilot extension |
| 11 | **G. Full codec composition DAG** | -0.015 to -0.025 (sum of above) | LOW (cross-terms unmeasured) | IS the campaign plan §2-§4 |
| 12 | **I. Inflate-time score-aware deletion** | $0 score; saves wall-clock | HIGH | useful for 30-min inflate gate, not score |

**Total predicted score floor from STACKING ONLY (no substrate-class shift):** sum of byte-disjoint additive primitives = -0.012 to -0.026 = **0.166-0.180 [contest-CPU, council-consensus]**. Beats PR101 GOLD by 0.013-0.027, our current 0.19205 by 0.012-0.026.

## §4 — Top-3 $0 quick wins to surface to operator (orthogonal to in-flight subagents)

These three are byte-disjoint, $0 build, and INDEPENDENT of the 3 subagents already running:

1. **Primitive D (Per-tensor-class magic codec)** — extends `tac.packet_compiler.*` selector to per-tensor granularity. Could be a 4th subagent OR in-context next turn. Predicted: -0.003 to -0.007. Single highest-EV new primitive of this audit.
2. **Primitive J (Per-pair tuple selector)** — extends fec6 build pipeline; trivial enumerator expansion. Predicted: -0.001 to -0.003 + small rate savings. Sister of subagent A's work but doesn't overlap (different selector axis).
3. **Primitive F (Magic codec on selector.bin)** — trivial 1-line build extension. Predicted: tiny but byte-disjoint pure $0 win.

**Recommendation:** when context permits OR when one of the 3 in-flight subagents completes (freeing a slot), spawn subagent D for Primitive D — that's the highest-EV $0 lever in the entire stacking inventory.

## §5 — PR-preparation handoff plan (top operator priority)

Per operator directive "op-routable #9 and the PR preparation should be our top priorities", PR prep splits into two phases:

**Phase 1 (this turn, $0):** the contest PR body lives at `docs/pr_writeups/cpu_frontier_fec6_20260517.md` (self-contained submission narrative pointing at fec6 archive `6bae0201`). Stub draft `PR_BODY_fec6_cpu_frontier_20260517.md` deleted 2026-05-17 per operator "Always delete and clean up clutter and cruft" standing directive — canonical writeup IS the PR body per its own header.

**Phase 2 (after subagent A completes):**
1. Run subagent A's dispatch ($0.30 paired auth eval)
2. If new archive beats 0.19205: update PR body to point at new archive sha; re-run Catalog #316 frontier scan; verify reports/latest.md FRONTIER section reflects new canonical best
3. Run pre-submission compliance check `scripts/pre_submission_compliance_check.py --contest-final`
4. Open fork PR via `tools/create_fork_pr_for_submission.py`

**Critical pre-PR gates per CLAUDE.md:**
- 5-turn consecutive clean-pass adversarial review (per CLAUDE.md "Submission PR gate — non-negotiable") — currently at Round 1/3 of recursive review on master-gradient bundle, NOT yet on the PR archive bundle itself; need separate review cycle on the FINAL submission packet
- Both CPU AND CUDA paired auth eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" — already in hand for `6bae0201` (CPU 0.19205, CUDA 0.22621); need re-measure for any new archive
- 1:1 contest-compliant hardware per CLAUDE.md — Modal Linux x86_64 CPU is within ±2e-7 of GHA per PR107 calibration; acceptable as long as final verification on GHA happens before merge

## §6 — Cross-references

- `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md` §2.2 (Venn diagram) + §3.6 (8 master-gradient uses)
- `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md` §1.7 (8 master-gradient uses ownership) + §4.1 (Wave 4 cross-paradigm stack)
- `.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md` (format0d two-layer attribution)
- `.omx/research/full_problem_space_reverse_engineering_cpu_gpu_both_20260517.md` (full problem-space; §11 ranked open questions)
- `.omx/state/substrate_composition_matrix.json` (Catalog #227 composition_alpha posterior)
- `feedback_owv3_0120_orthogonal_stack_sub_1_landed` (historical sub-1.000 stack precedent — superseded by current frontier)
- `feedback_council_22_22_prescription_complete` (the 22/22 council verdict that produced QZS3 + QP1 pose codec — a sister stacking lineage)

## §7 — Reactivation criteria

When operator approves Primitive D (per-tensor-class magic codec) and dispatches the $0.30 paired auth eval, update this memo with:
- Empirical Δrate per tensor class
- Empirical ΔS on fec6+per-tensor stack
- Update ranking table §3 with measured values

Similar update cycle for each of primitives A/B/C/E/F/J as they land.
