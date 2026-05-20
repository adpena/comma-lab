<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #344 canonical equation cross-ref: empirical anchors in this memo align with `tac.canonical_equations` registry equations `per_byte_leverage_cross_hardware_aware_v2` + `cross_codec_super_additive_orthogonality_predictor_v1` (commit `80484241f`). Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — no mutation to landing-memo body. -->
---
council_tier: T1
council_attendees:
  - Shannon
  - Carmack
  - Hotz
  - PR95Author
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_assumption_adversary_verdict:
  - assumption: "FEC6 fixed-Huffman k=16 wastes fractional bits per selector"
    classification: HARD-EARNED
    rationale: "CROSS-CANDIDATE finding #1 empirical anchor: +259 bytes / +0.00333 [contest-CPU]"
  - assumption: "Rice-Golomb k=2 fits geometric-decay selector distribution"
    classification: CARGO-CULTED
    rationale: "L0 design-time assumption; L1 must sweep k empirically"
  - assumption: "Per-class boundary signal matches SegNet 5-class structure"
    classification: HARD-EARNED
    rationale: "NSCS06 v6→v7 44% improvement empirical anchor"
council_decisions_recorded:
  - "land 5 G3 SELECTOR-EXTENSIONS variants at L0 SCAFFOLD"
  - "all dispatch_enabled=false research_only=true pending STAIRCASE Step 11-15 operator approval"
---

# PACT-NERV G3 SELECTOR-EXTENSIONS L0 SCAFFOLD DESIGN (5 variants)

## Context

WAVE-3-PACT-NERV-G3-SELECTOR-EXTENSIONS-L0-BUILD 2026-05-20 lands the 5
G3 (SELECTOR-PARADIGM-EXTENSIONS) variants from PACT-NERV-ULTIMATE
(commit `e3ad4243a`) as L0 SCAFFOLDs per CLAUDE.md "Substrate scaffolds
MUST be COMPLETE or RESEARCH-ONLY" non-negotiable. Direct empirical
extension of CROSS-CANDIDATE finding #1 (+259 bytes / +0.00333 ratio
from FEC6 fixed-Huffman k=16 selector).

## 5 variants

| # | Variant | Literature | LOC | Predicted ΔS |
|---|---|---|---|---|
| 11 | Pact-NeRV-SELECTOR-V2 | Witten 1987 arithmetic coding | ~150 | [-0.003, 0.000] |
| 12 | Pact-NeRV-SELECTOR-V3 | Golomb 1966 + Rice 1971 | ~300 | [-0.004, +0.001] |
| 13 | Pact-NeRV-SELECTOR-V4 | Robinson-Cherry 1967 RLE | ~150 | [-0.005, +0.001] |
| 14 | Pact-NeRV-IA3-Multi | Liu 2022 IA3 + FILM 8.6 multi | ~250 | [-0.003, +0.001] |
| 15 | Pact-NeRV-AsymmetricBoundary | NSCS06 v7 per-class chroma | ~250 | [-0.004, +0.001] |

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable:

| Layer | Decision | Rationale |
|---|---|---|
| HNeRV-class base decoder | ADOPT_CANONICAL_BECAUSE_SERVES | PR101 GOLD baseline = HARD-EARNED |
| Score-aware loss helper | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.substrates.score_aware_common.score_pair_components_dispatch` per Catalog #164 + #222 |
| Differentiable eval roundtrip | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally` per Catalog #6 MANDATORY |
| Device-or-die helper | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.substrates._shared.trainer_skeleton.device_or_die` per CLAUDE.md MPS-NOISE |
| Substrate trainer skeleton | ADOPT_CANONICAL_BECAUSE_SERVES | `pin_seeds + git_head_sha + detect_hardware_substrate` |
| Arithmetic coder (V2) | FORK_BECAUSE_SUPPRESSES | NEW primitive; canonical helpers don't exist; ~150 LOC bolt-on |
| Rice-Golomb coder (V3) | FORK_BECAUSE_SUPPRESSES | NEW primitive; ~50 LOC core class |
| RLE coder (V4) | FORK_BECAUSE_SUPPRESSES | NEW primitive; varint round-trip ~80 LOC |
| Multi-block IA3 (IA3-Multi) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Sister pact_nerv_ia3 is single-block; multi-block is the distinguishing primitive |
| Asymmetric boundary FiLM | FORK_BECAUSE_PRINCIPLED_MISMATCH | NSCS06 v7 chroma pattern; 5-class SegNet-aligned signal |
| Archive grammar (PSV2/PSV3/PSV4/PIM1/PAB1) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Each variant declares its own monolithic 0.bin grammar with distinctive magic |
| Substrate META layer SubstrateContract | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #241/#242 canonical contract |

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS** — 5 distinct primitives (arithmetic/RG/RLE/multi-IA3/asym-boundary); each FORK_BECAUSE in §canonical decision; each has distinctive archive magic.
2. **BEAUTY + ELEGANCE** — Each substrate package <= 8 files (~30-second-reviewable per HNeRV parity L12). Coder primitives <= ~120 LOC standalone.
3. **DISTINCTNESS** — Each variant's hypothesis is empirically distinguishable: V2 = fractional-bit precision; V3 = geometric-decay; V4 = temporal-coherence; IA3-Multi = multi-layer; Asym-Boundary = per-class.
4. **RIGOR** — All 5 variants pass canonical contract (Catalog #124 archive_grammar + 7 sister fields); 74/74 tests pass; archive byte-mutation no-op proofs.
5. **OPTIMIZATION PER TECHNIQUE** — Each coder optimized for its assumed distribution: arithmetic for non-power-of-2 probs / Rice-Golomb for geometric / RLE for temporal runs.
6. **STACK-OF-STACKS-COMPOSABILITY** — Per PACT-NERV-ULTIMATE Section 4: V2+V3 ADD (orthogonal side-info); V2+V4 ADD; IA3-Multi+V2 ADD; Asym-Boundary+IA3-Multi ADD.
7. **DETERMINISTIC REPRODUCIBILITY** — All packers use `sort_keys=True` JSON + brotli q=9; byte-stable round-trip verified in tests.
8. **EXTREME OPTIMIZATION** — `--enable-autocast-fp16 / --enable-torch-compile / --enable-gt-scorer-cache` flags reserved per Catalog #270; eval_roundtrip MANDATORY per Catalog #6.
9. **OPTIMAL MINIMAL CONTEST SCORE** — L0 SCAFFOLD posture defers absolute-score claims to Stage 1 dispatch operator-gated per Catalog #325.

## Cargo-cult audit per assumption

Per Catalog #303:

| Variant | Assumption | Classification | Unwind path |
|---|---|---|---|
| V2 | Static uniform cum_freq | CARGO-CULTED | L1 dispatch: fit cum_freq to actual selector distribution |
| V2 | Per-symbol encoding | CARGO-CULTED | L1: per-pair grouped coding |
| V3 | Fixed k=2 | CARGO-CULTED | L1: sweep k ∈ {1, 2, 3, 4}; adaptive-k per stream |
| V3 | Symbol-by-symbol | CARGO-CULTED | L1: RLE+Huffman hybrid |
| V4 | (value, varint) representation | HARD-EARNED-LITERATURE | Standard RLE format |
| V4 | No value-distribution context | CARGO-CULTED | L1: RLE+Huffman hybrid for value field |
| IA3-Multi | Sister-conditioning fusion (pose + difficulty) | CARGO-CULTED | L1: ablate pose-only vs difficulty-only vs combined |
| IA3-Multi | Multi-block at EVERY upsample | HARD-EARNED-EMPIRICALLY-SUPERIOR | FILM-FAMILY-RESEARCH §8.6 |
| Asym-Boundary | 5-class boundary matches SegNet | HARD-EARNED-EMPIRICAL | Upstream SegNet 5-class output |
| Asym-Boundary | FiLM γ+β vs IA3 γ-only | CARGO-CULTED | L1: ablate γ-only variant |
| Asym-Boundary | Final-block-only modulation | CARGO-CULTED | L1: sweep per-block vs final-block |
| All | FEC6 k=16 palette inherited | HARD-EARNED-EMPIRICAL | CROSS-CANDIDATE finding #1 anchor |
| All | HNeRV-class base decoder | HARD-EARNED | PR101 GOLD baseline |

## Observability surface

Per Catalog #305:

- **Inspectable per layer** — substrate trainer provenance.json captures: substrate_tag / lane_id / n_params / device / hardware_substrate_detected / per-variant primitive params (e.g. selector_palette_size + selector_bytes_encoded).
- **Decomposable per signal** — score_aware_loss returns dict with rate_term / seg_term / pose_term / loss_total.
- **Diff-able across runs** — git_head + started_at_utc + score_axis_tag enable run-to-run comparison.
- **Queryable post-hoc** — JSON provenance is grep-friendly; lane_registry.json + canonical state dirs queryable.
- **Cite-able** — every substrate cites Catalog # gates (#124 + #240 + #220 + others) + PACT-NERV-ULTIMATE Variant # + literature anchor.
- **Counterfactual-able** — byte-mutation no-op proofs in tests prove every distinguishing primitive empirically affects archive bytes.

## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility:

- L0 SCAFFOLD posture: predicted_band declared as `unknown_l0_scaffold_pending_stage_1_dispatch` per Catalog #324 (avoids the C6 IBPS 22× miss empirical-falsification pattern of declaring random-init predicted bands before training).
- Per PACT-NERV-ULTIMATE Section 5 Finding #1: each variant predicted at [-0.005, +0.001] when adopted as bolt-on to FEC6 frontier 0.19205. Cumulative if ADDITIVE [-0.019, +0.005].
- Dykstra-feasibility intersection check DEFERRED to Stage 1 dispatch when actual empirical anchors land.

## horizon-class declaration

Per Catalog #309: `horizon_class: plateau_adjacent` (all 5 variants target the [0.180, 0.200] plateau bands; not asymptotic-pursuit).

## Sister-files-recently-landed verdict

PRE-WRITE step 0 (Catalog #314/#340 self-protection): EXIT 0 PROCEED — no sister commits touched the 10 target files within 12-hour lookback.

## Reactivation criteria

Each variant's `_full_main` raises NotImplementedError per Catalog #240; reactivation requires:
1. PACT-NERV-ULTIMATE STAIRCASE Step (11/12/13/14/15) dispatch operator-gated per Catalog #325.
2. Cargo-cult audit per Catalog #303 — empirically validate the CARGO-CULTED assumptions OR reclassify as HARD-EARNED.
3. Score-aware Lagrangian wire-in + canonical auth-eval helper per Catalog #226.
4. Recipe `research_only` flip to `false`; `dispatch_enabled` flip to `true`; `predicted_band` declared with `validation_status` per Catalog #324.
5. Operator-frontier-override per Catalog #300 Mission alignment Consequence 1 OR Stage 1 approval.

## 6-hook wire-in declaration per Catalog #125

At L0 SCAFFOLD posture (all 5 variants):
- hook #1 sensitivity-map: N/A (no sensitivity signal until Stage 1 dispatch)
- hook #2 Pareto constraint: rate_distortion_v1
- hook #3 bit-allocator: N/A (no per-tensor bit allocator at scaffold posture)
- hook #4 cathedral autopilot dispatch: N/A (research_only)
- hook #5 continual-learning posterior: N/A (no anchor until [contest-CUDA] measurement)
- hook #6 probe-disambiguator: N/A (variant-specific hypothesis IS Stage 1 dispatch's empirical purpose)

## Cross-references

- PACT-NERV-ULTIMATE: `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md` (commit `e3ad4243a`)
- Sister Pact-NeRV-IA3: commit `9cf9bdb16`
- Sister NERV-LITERATURE-L0-RESCOPED (boost_nerv + nirvana + coin_plus_plus): commit `d9aaf7c13`
- FILM-FAMILY-RESEARCH: commit `9a95d1daf`
- FEC6 frontier anchor: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/`
- NSCS06 v7 44% improvement: `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md`
