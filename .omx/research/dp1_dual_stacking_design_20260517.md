---
council_tier: T1
council_attendees: [SubagentBuild]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
horizon_class: frontier_pursuit
---

# DP1 Dual-Stacking Design Memo — PATH 1 (composition) + PATH 2 (training-time prior)

**Lane:** `lane_dp1_plus_fec6_dual_stacking_build_20260517`
**Date:** 2026-05-17
**Operator directive:** dual-path build (composition + training-time prior) per
grand council T3 symposium Decision #3 + comprehensive audit META-INSIGHT
convergence on DP1 stacking as highest-EV next move after the fec6 frontier
0.19205 [contest-CPU] anchor landed.

## 2026-05-18 supersession: PATH 1 rate arithmetic corrected

`tools/probe_dp1_pr101_composition_noop_detector.py` landed a byte-closed
structural proof for the existing L1 packet:
`.omx/research/dp1_pr101_composition_noop_probe_20260518_codex.json`.

The probe verifies:

- composed archive SHA-256:
  `507d2a000ecf5a220e9b1ab765f75e39015cfb7b2af00606be3cb0758b8eb855`
- archive bytes: `204344`
- base fec6 bytes: `178517`
- DP1 prefix bytes: `25814`
- DPCOMP header bytes: `13`
- total L1 overhead: `25827` bytes
- contest rate delta if frames are identical:
  `25 * 25827 / 37545489 = +0.017197139182`

This supersedes every older `+0.0000172` PATH 1 rate-axis statement below. The
older arithmetic was off by 1000x. PATH 1 is still a useful no-op/byte-closure
control, but it is not a score-lowering dispatch: with `PACT_DP1_PRIOR_STRENGTH=0.0`
and frame parity, the expected CPU control is `0.209247139182`
(`0.19205 + 0.017197139182`), not `0.19207`. Any L2 prior-effect path must buy
back more than `0.017197` score before it can be promotion- or ranking-relevant.

### 2026-05-18 Codex probe custody hardening

The no-op detector now refuses a structurally valid DPCOMP packet when
`build_manifest.json::lane_id` does not equal
`lane_dp1_plus_fec6_dual_stacking_build_20260517`. The previous probe emitted
the hard-coded lane id in its result payload but did not require the packet
manifest to carry the same custody claim, which could let an unrelated PR101
DPCOMP packet receive this lane's `l1_rate_only_noop_verified` verdict. The
regression is
`src/tac/tests/test_probe_dp1_pr101_composition_noop_detector.py::test_build_probe_payload_blocks_wrong_build_manifest_lane`.

## TL;DR

- **PATH 1 (composition)**: DP1 + fec6 archive compose via canonical
  `tac.substrates.pretrained_driving_prior.composition.compose_with`. L1 packet
  ships with `PACT_DP1_PRIOR_STRENGTH=0.0` default — rate-axis cost
  (`+0.017197139182` score from 25,827 wrapper bytes) measurable in isolation
  BEFORE L2 INTEGRATION enables frame-axis effect. Predicted ΔS CPU
  `[-0.003, -0.012]` was a pre-L2 heuristic and is not enough to overcome the
  corrected L1 rate cost. **PATH 1 is BUILT as a byte-closure/no-op control;
  paid paired-axis eval now needs explicit operator decision, not automatic
  frontier pressure.**
- **PATH 2 (training-time prior, REFORMULATED)**: operator's original "L2 on PR101
  decoder weights" is structurally incompatible with fec6 (no learned decoder
  weights) per premise-verifier PV-6. Reformulation: apply DP1 FRAME-SPACE prior
  (DashcamPriorLoss; Atick-Redlich cooperative-receiver lens) on
  `pr101_lc_v2_clone_enhanced_curriculum` learned-decoder RGB output. **PATH 2
  is SCAFFOLDED with `_full_main` raising NotImplementedError + recipe
  research_only=true**, pending council Phase 2 approval of variant-A vs B
  + λ_DP1 sweep design + probe-disambiguator landing.

## Background

DP1 (`pretrained_driving_prior`) is the canonical pretraining lane that
distills a 5-10 KB frozen dashcam-statistical codebook from publicly available
driving datasets (Comma2k19 MIT) and applies it as a soft prior during contest-
video score-aware overfit. Two Phase 1+2 landings (`lane_pretrained_driving_
prior_lane_scaffold_20260513` + `lane_pretrained_driving_prior_phase_2_20260514`)
established the codebook + composition + distillation surfaces; ZERO empirical
stackings have actually fired against the current frontier yet.

fec6 (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`) is the new
contest-CPU frontier landed 2026-05-17 at 0.19205 ([contest-CPU]; sha256
`6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`; 178,517
bytes). Per `reports/latest.md` FRONTIER section + `feedback_permanent_fix_
frontier_signal_loss_landed_20260517.md` this is currently the **best
contest-CPU score we have**.

The grand council T3 strategic symposium ($50 budget; 2026-05-17) and the
comprehensive audit (META-INSIGHT) both converged independently on DP1 stacking
as the highest-EV next move. PATH 1 + PATH 2 represent the two mathematically
distinct stackings.

## PATH 1 — Composition (inflate-time; +bytes)

### Architecture

The canonical DPCOMP wrapper (13-byte header + DP1 archive + base archive) is
emitted by `compose_with(dp1_bytes, fec6_bytes, base_substrate="pr101")`. The
build tool `tools/build_dp1_plus_fec6_composition_packet.py` produces:

```
experiments/results/dp1_plus_fec6_composition_20260517/
├── archive.zip                 # 204,344 bytes (= 13 + 25,814 + 178,517)
├── inflate.sh                  # 3-arg contest contract
├── inflate.py                  # self-contained per Catalog #295
├── src/
│   ├── codec.py                # vendored from fec6 submission_dir
│   ├── frame_selector.py       # vendored
│   ├── model.py                # vendored
│   ├── fec6_inflate.py         # vendored from fec6 inflate.py
│   └── dp1_composition.py      # vendored canonical DPCOMP peeler
├── archive_manifest.json       # canonical SHA + provenance
└── build_manifest.json         # build provenance + predicted ΔS band
```

### Inflate-time mechanism

1. `inflate.sh` invokes `inflate.py` per-video per the contest 3-arg contract
2. `inflate.py::inflate_composed` reads composed bytes
3. `dp1_composition.decompose_bytes` peels the 13-byte DPCOMP header and
   returns `(dp1_bytes, base_substrate, fec6_bytes, schema_version)`
4. Asserts `base_substrate == "pr101"` (refuses if wrong)
5. Re-stages fec6 bytes to a scratch file + delegates to vendored `fec6_inflate.py`
6. Optionally applies DP1 prior to inflated frames (gated by
   `PACT_DP1_PRIOR_STRENGTH` env var; default 0.0 = no-op)

### L1 vs L2 INTEGRATION

**L1 (LANDED 2026-05-17)**: composition packet ships with
`PACT_DP1_PRIOR_STRENGTH=0.0` default. The decompose runs at every inflate call
(structural byte consumption proof per Catalog #220), but the frame-axis effect
is gated by the env var. This is the **rate-axis baseline measurement**:
+0.017197139182 contest rate term in isolation. Per Catalog #220 the operational
mechanism is `OPERATIONAL_DEFERRED_TO_L2` — bytes are structurally consumed
(decompose runs) but frame-axis effect is deferred.

**L2 INTEGRATION (DEFERRED)**: lifting `PACT_DP1_PRIOR_STRENGTH` > 0 wires
`tac.substrates.pretrained_driving_prior.DashcamPriorLoss.apply_soft_prior` to
the inflated RGB frames. Predicted ΔS CPU [-0.003, -0.012] from the band
`fec6 baseline 0.19205 + rate +0.017197139182 + frame-prior [-0.003, -0.012]`.
L2 dispatch is operator-gated; the L1 packet's `inflate.py` explicitly raises
RuntimeError when strength > 0 to prevent silent strength-leakage.

## PATH 2 — Training-time prior (compress-time; +0 bytes; REFORMULATED)

### Critical premise verification

Per CLAUDE.md Catalog #229 + premise-verifier PV-6
(`.omx/tmp/dp1_dual_stacking_premise_verifier.txt`), the operator's original
PATH 2 description **"L2 regularizer on PR101 decoder weights"** is structurally
incompatible with fec6:

| Component | Mathematical type |
|---|---|
| DP1 codebook | FRAME-SPACE (RGB tensors: `road_plane_basis`, `sky_horizon_profile`, `vehicle_appearance_basis`) |
| fec6 grammar | frame-exploit-selector + Huffman entropy code (no learned decoder weights) |
| L2 weight-reg target | NONE (no W_decoder tensor exists in fec6) |

The frame-space DP1 prior cannot regularize fec6's weight-space because fec6
has no weight-space. Per CLAUDE.md "Forbidden premature KILL", this is NOT a
kill — it's a substrate-coupling mismatch. The reformulated PATH 2 points the
DP1 frame-prior at the natural recipient substrate that DOES have learned
decoder weights: `pr101_lc_v2_clone_enhanced_curriculum`.

### Reformulated architecture

```python
# experiments/train_substrate_pr101_with_dp1_prior_regularizer.py
loss = (
    canonical_pr101_lc_v2_score_aware_loss(decoded_frames, gt_frames)
    + lambda_dp1_prior * dashcam_prior_loss(decoded_frames)  # frame-space prior
)
```

DP1 codebook is FROZEN (registered as torch buffers, never trained). λ_DP1
is the only tunable hyperparam. The DashcamPriorLoss applies the codebook
projection at training time only — at inflate time, two variants:

* **variant-A**: codebook ships in archive (+5-10 KB rate cost). Inflate-time
  `apply_soft_prior` operationalizes the prior.
* **variant-B**: codebook is BAKED INTO inflate.py as numpy constant per
  CLAUDE.md Catalog #146 contest_one_video_replay allowance. +0 archive bytes;
  +5-10 KB inflate.py source growth (within Catalog #146 LOC budget).

Council Phase 2 deliberates variant-A vs variant-B.

### Scaffold state (LANDED 2026-05-17)

`_smoke_main` validates DashcamPriorLoss correctness on CPU (no Modal, no
dispatch, no training): codebook constructs + `apply_soft_prior` produces
finite output + gradients flow through. **19 dedicated tests pass.**

`_full_main` raises `NotImplementedError` with explicit reactivation criteria
per CLAUDE.md Catalog #240 "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY". Recipe is `research_only=true + smoke_only=true +
dispatch_enabled=false`.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable.

| Layer | Decision | Rationale |
|---|---|---|
| Composition wrapper API (PATH 1) | ADOPT_CANONICAL_BECAUSE_SERVES | `compose_with` / `decompose` / `verify_composition` already exist + tested + Catalog #211 enforces canonical routing |
| Inflate-side wrapper peeler (PATH 1) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Vendored copy `dp1_composition.py` in submission `src/` per Catalog #295 self-contained empty-PYTHONPATH requirement; canonical helper stays in `tac.*` (not vendored). FORK is mandatory because canonical is unreachable from submission. |
| Inflate device selection (PATH 1) | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #205 `select_inflate_device` honoring `PACT_INFLATE_DEVICE` env var; local helper in emitted inflate.py mirrors canonical |
| Build tool emit (PATH 1) | FORK_BECAUSE_PRINCIPLED_MISMATCH | The build tool is one-shot composition (not training); no canonical builder applies. Catalog #270 `dispatch_kind: tool` opts out of substrate-only Tier 2/3 fields per scope-fix subagent lane `lane_catalog_270_scope_fix_tool_vs_substrate_dispatch_20260517`. |
| Recipe schema (PATH 1) | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical operator-authorize recipe contract per Catalog #270 + #167 + #199 + #240; paired-axis fields per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA". |
| Trainer skeleton (PATH 2) | ADOPT_CANONICAL_BECAUSE_SERVES | `@register_substrate(SubstrateContract(...))` per Catalog #241/#242; 36 canonical fields validated at decoration time. |
| Score-aware loss (PATH 2) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Canonical `scorer_loss_terms_btchw` does not natively compose with `DashcamPriorLoss`; the combined loss `λ * dashcam_prior + (1-λ) * scorer_loss` is the UNIQUE PATH 2 contribution. SubstrateContract `score_aware_loss="custom"` marks this. |
| DashcamPriorLoss (PATH 2) | ADOPT_CANONICAL_BECAUSE_SERVES | Already exists in `tac.substrates.pretrained_driving_prior.prior_application`; frozen buffers; gradient-safe. |
| Hardware substrate detection | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #190 `detect_hardware_substrate` via canonical helper. |
| auth-eval routing | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #226 `gate_auth_eval_call` mandatory for substrate trainers. |
| Catalog #210 codebook provenance | ADOPT_CANONICAL_BECAUSE_SERVES | The canonical DP1 codebook serializer already preserves all 6 required fields on production-distilled codebooks. |
| variant-A vs variant-B (PATH 2) | UNCLEAR_NEEDS_EMPIRICAL — DEFER TO COUNCIL | Council Phase 2 grade tradeoff per CLAUDE.md "Design decisions". |

## 9-dimension success checklist evidence

Per CLAUDE.md Catalog #294 + 9-dimension success checklist standing directive.

| Dim | Evidence |
|---|---|
| 1. UNIQUENESS (class-shift not within-class) | PATH 1: composition is a NEW grammar-stacking class (DPCOMP wrapper), not a within-fec6 refinement. PATH 2: combined loss is a NEW class (frame-prior augmenting score-aware loss). Both are class-shift relative to fec6's standalone frame-exploit-selector. |
| 2. BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | PATH 1 build tool: ~500 LOC including templates. PATH 2 scaffold: ~370 LOC. Both bind ALL ingredients (architecture + training + grammar + runtime + export + tests + recipe) into ONE coherent file each per PR101 model. |
| 3. DISTINCTNESS (explicitly different from sisters) | PATH 1 distinct from `pretrained_driving_prior` substrate itself (the latter is the RESEED lane producing the codebook; this lane CONSUMES the codebook via composition). PATH 2 distinct from PATH 1 by mathematical type (compress-time vs inflate-time). |
| 4. RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | PV-1 through PV-10 verified pre-edit (`.omx/tmp/dp1_dual_stacking_premise_verifier.txt`). Empirical round-trip anchor: composed sha256 `507d2a000ecf5a220e9b1ab765f75e39015cfb7b2af00606be3cb0758b8eb855`; decomposition byte-identical via 32+19 = 51 dedicated tests. |
| 5. OPTIMIZATION PER TECHNIQUE | Covered by Catalog #290 canonical-vs-unique decision table above. |
| 6. STACK-OF-STACKS-COMPOSABILITY | PATH 1's DPCOMP wrapper is orthogonal to fec6's internal grammar; both substrates' inflate logic stay in their canonical modules; the composition wrapper just glues them. PATH 2's DashcamPriorLoss is a +0-bytes augmentation orthogonal to pr101_lc_v2's score-aware loss. PATH 1+2 can compose: train pr101_lc_v2 with DP1 prior (PATH 2) → ship resulting archive composed with DP1 codebook prefix (PATH 1). |
| 7. DETERMINISTIC REPRODUCIBILITY | PATH 1 byte-stability tested (`test_sha256_byte_stable`): re-running build produces identical composed bytes. Catalog #19 deterministic zip + fixed-timestamp ZipInfo + sort_keys JSON. |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | PATH 1 dispatch envelope: $1.90 (T4 + CPU paired). PATH 2 scaffold: $0 (CPU smoke; no GPU spend). |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Predicted ΔS CPU [-0.003, -0.012] from fec6 baseline 0.19205 = 0.180 - 0.189 target band. If lower bound achieved (0.180), this would clearly beat the public leaderboard PR101 GOLD 0.193 by 0.013 CPU. |

## Cargo-cult audit per assumption

Per CLAUDE.md Catalog #303 + the hard-earned-vs-cargo-culted addendum.

| Assumption | Classification | Rationale |
|---|---|---|
| DP1 composition wrapper IS the canonical reuse surface | HARD-EARNED | Catalog #211 STRICT preflight refuses non-canonical hand-rolled DP1 composition; sister Phase 2 landing tests already exercised compose_with on `a1`/`hdm8`/`yucr`/`time_traveler_l5`/`sane_hnerv` base substrates with byte-stable round-trip. |
| pr101 IS a registered base tag in DPCOMP | HARD-EARNED | Verified via `known_base_substrates()` returning `('a1', 'hdm8', 'pr101', 'sane_hnerv', 'time_traveler_l5', 'yucr')`. No extension needed. |
| fec6 archive is the current contest-CPU frontier (0.19205) | HARD-EARNED | `reports/latest.md` FRONTIER section + `feedback_permanent_fix_frontier_signal_loss_landed_20260517.md` + sha256 verified on disk (`6bae0201`). |
| DP1 codebook is FRAME-SPACE prior (RGB tensors), not WEIGHT-SPACE | HARD-EARNED | Read `tac.substrates.pretrained_driving_prior.codebook` source: `road_plane_basis`, `sky_horizon_profile`, `vehicle_appearance_basis` are all RGB-shape tensors. `DashcamPriorLoss.apply_soft_prior` operates on `(B, 3, H, W)` predicted RGB output. |
| The L2-weight-reg formulation of PATH 2 ON FEC6 is structurally incompatible | HARD-EARNED | fec6 has no learned decoder weights — verified by reading `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py` (no torch.nn.Module instantiation; pure Huffman-decode + selector application). |
| pr101_lc_v2_clone IS a learned-decoder substrate that CAN accept DP1 frame-prior | HARD-EARNED | `experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py` (1111 lines) instantiates HNeRV-family learned decoder + canonical score-aware loss. |
| L1 composition packet PACT_DP1_PRIOR_STRENGTH=0.0 default is OPERATIONAL_DEFERRED per Catalog #220 | HARD-EARNED | decompose runs at every inflate call (structural byte consumption proof); the env-var gate is the canonical 2-phase composition discipline (rate-axis baseline before frame-axis effect). |
| Predicted ΔS CPU band [-0.003, -0.012] is first-principles-citable | CARGO-CULTED — Phase 2 Dykstra-feasibility check required | The band is a heuristic extrapolation from sister DP1 ablation expectations; NOT a Dykstra-feasibility intersection on the rate-distortion polytope. Per Catalog #296: this design memo cites the `predicted_delta_basis` via the probe-disambiguator path documented in PATH 2 scaffold. The unwind path: land `tools/probe_dp1_lambda_disambiguator.py` AND a Dykstra-feasibility helper for the DP1-fec6 rate-distortion intersection BEFORE firing the paid L2 INTEGRATION dispatch. PATH 1 paired-axis dispatch is unaffected (its predicted ΔS is rate-axis only = +0.0000172, which is a closed-form contest-rate-term calculation, not a Dykstra intersection). |
| variant-A vs variant-B (codebook in archive vs baked into inflate.py) is a council-grade tradeoff | HARD-EARNED | Per CLAUDE.md "Design decisions — non-negotiable": 2+ defensible alternatives with non-trivial preferences; deferred to council Phase 2. |
| pr101_lc_v2_clone trainer integration surface is a Phase 2 council-grade decision | HARD-EARNED | Per CLAUDE.md "Design decisions": which exact loss-hook to inject DashcamPriorLoss at affects gradient flow + eval-roundtrip behavior. Phase 2 council deliberates. |

## Observability surface

Per CLAUDE.md Catalog #305 + max-observability standing directive (6-facet
definition: inspectable per layer / decomposable per signal / diff-able across
runs / queryable post-hoc / cite-able / counterfactual-able).

| Facet | PATH 1 | PATH 2 (scaffold) |
|---|---|---|
| **Inspectable per layer** | `archive_manifest.json` declares per-section bytes (header / DP1 / fec6); `build_manifest.json` carries provenance for both source archives | SubstrateContract carries 36 canonical fields; `tools/lane_maturity.py audit` shows 7 gates per lane |
| **Decomposable per signal** | rate-axis cost (+0.0000172) is computable directly from archive_size_bytes; frame-axis effect (L2) is gated by `PACT_DP1_PRIOR_STRENGTH` env-var enabling per-strength sweep | Loss decomposed: `lambda_dp1_prior * dashcam_prior + (1 - lambda) * scorer_loss`. Each term loggable separately. |
| **Diff-able across runs** | Composed archive sha256 is byte-stable (test_sha256_byte_stable); fec6 standalone vs composed inflated frames hashed for cross-axis parity | Smoke output prints buffer count + gradient norm; reproducible across runs (no random state in smoke path) |
| **Queryable post-hoc** | `archive_manifest.json` + `build_manifest.json` are machine-readable JSON | `.omx/state/lane_registry.json` carries lane state; `.omx/state/council_deliberation_posterior.jsonl` carries any future Phase 2 council anchors |
| **Cite-able** | Anchored to (composed_sha256, dp1_source_sha256, fec6_source_sha256, build_tool_commit, built_at_utc) | Anchored to (trainer_path, lane_id, council_verdict_provenance memo path, contract.lane_id) |
| **Counterfactual-able** | `tools/verify_distinguishing_feature_byte_mutation.py` could mutate single bytes in the DP1 prefix and verify inflated frames remain identical when strength=0.0 (proves byte consumption is structural-only without frame-effect at L1) | At L2 INTEGRATION: probe-disambiguator sweeps λ to enable counterfactual "what if λ=0.01 vs 0.10" comparison via paired runs |

## Predicted ΔS band

Per CLAUDE.md Catalog #296 — every predicted ΔS band requires Dykstra-feasibility
check OR first-principles citation OR probe-disambiguator path.

### PATH 1 predicted ΔS

**Band**: `[+0.0000172, -0.012] [time-traveler-prediction]` — from fec6 baseline
0.19205 [contest-CPU].

* **Upper bound (+0.0000172; rate-axis cost only)**: closed-form contest-rate-term
  calculation `25 * 25_814 / 37_545_489 = +0.0000172`. **NOT Dykstra-required**:
  this is the canonical contest rate term Shannon arithmetic; no polytope
  intersection involved. PATH 1 L1 with `PACT_DP1_PRIOR_STRENGTH=0.0` MUST land
  at this point (+/- numerical noise in the inflated frame hashes).
* **Lower bound (-0.012; frame-axis effect at L2 INTEGRATION)**: heuristic
  extrapolation from sister DP1 Phase 2 ablation expectations. **Dykstra-
  feasibility check REQUIRED before L2 dispatch** per Catalog #296. The unwind
  path: land `tools/probe_dp1_lambda_disambiguator.py` BEFORE firing the paid
  L2 INTEGRATION dispatch.

**Probe-disambiguator path**: `tools/probe_dp1_lambda_disambiguator.py` (planned,
Phase 2). Sweeps `PACT_DP1_PRIOR_STRENGTH` ∈ {0.0, 0.01, 0.05, 0.10, 0.25, 0.50}
on the L1 packet's inflate path and measures frame-axis ΔS empirically.

### PATH 2 predicted ΔS

**Band**: `[-0.005, -0.020] [time-traveler-prediction]` — from fec6 baseline OR
pr101_lc_v2_clone baseline (which is council-decided). **Dykstra-feasibility
check + probe-disambiguator REQUIRED before any paid dispatch fires** per
Catalog #296. PATH 2 is research_only + smoke_only at landing; the L1 SCAFFOLD
state means no paid dispatch is possible until Phase 2 council lifts the
`_full_main NotImplementedError`.

## L2 Integration plan (PATH 1)

L2 enables the frame-axis effect by lifting `PACT_DP1_PRIOR_STRENGTH > 0`. The
inflate.py at L2 must:

1. Parse DP1 codebook from `dp1_bytes` via canonical DP1 archive parser
2. Construct `DashcamPriorLoss` with the codebook + frozen weights
3. Apply `apply_soft_prior(strength=PACT_DP1_PRIOR_STRENGTH)` to inflated RGB
4. Continue with raw-frame write

The L1 inflate.py raises RuntimeError when strength > 0 to enforce the L1/L2
boundary; L2 lands the actual implementation in a sister commit.

## Sister-subagent coordination

Per Catalog #302 ownership map:

| File | Owner |
|---|---|
| `tools/extract_master_gradient.py` | scope-fix sister `a2aa62263236144ee` |
| `tools/operator_authorize.py` | scope-fix sister `a2aa62263236144ee` |
| `src/tac/deploy/dispatch_protocol.py` | scope-fix sister `a2aa62263236144ee` |
| `.omx/operator_authorize_recipes/master_gradient_fec6_modal_cpu_dispatch.yaml` | scope-fix sister `a2aa62263236144ee` |
| `tools/cathedral_autopilot_autonomous_loop.py` | producer→cathedral sister |
| `tools/build_dp1_plus_fec6_composition_packet.py` | THIS LANE |
| `src/tac/tests/test_dp1_plus_fec6_composition.py` | THIS LANE |
| `experiments/train_substrate_pr101_with_dp1_prior_regularizer.py` | THIS LANE |
| `src/tac/tests/test_train_substrate_pr101_with_dp1_prior.py` | THIS LANE |
| `.omx/operator_authorize_recipes/dp1_plus_fec6_composition_modal_paired_dispatch.yaml` | THIS LANE |
| `.omx/operator_authorize_recipes/substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch.yaml` | THIS LANE |
| `.omx/research/dp1_dual_stacking_design_20260517.md` (THIS FILE) | THIS LANE |

All scopes disjoint. ✓

## Reactivation criteria (PATH 2 _full_main lift)

Per CLAUDE.md "Forbidden premature KILL" + Catalog #240:

1. Council Phase 2 deliberation memo lands with PROCEED verdict on:
   * variant-A (codebook in archive; +5-10 KB) vs variant-B (baked into inflate.py)
   * λ_DP1 sweep design (which values to test in parallel)
   * cost-band budget (T4 100ep ~$1.50; full curriculum ~$10)
2. `tools/probe_dp1_lambda_disambiguator.py` lands + bootstrapped on λ sweep
3. Sister CPU smoke dispatch validates DashcamPriorLoss + pr101_lc_v2 loss
   compose correctly on the fec6 frontier archive's GT pair batches
4. Catalog #229 premise verification on the pr101_lc_v2_clone integration surface
5. Catalog #295 self-contained inflate.py for variant-B (codebook baked in)
   OR canonical archive grammar extension for variant-A

## Cross-references

* `.omx/tmp/dp1_dual_stacking_premise_verifier.txt` — 10 verified premises (PV-1 to PV-10)
* `feedback_dp1_phase_2_landed_20260514.md` — DP1 substrate Phase 2 anchor
* `feedback_permanent_fix_frontier_signal_loss_landed_20260517.md` — fec6 frontier anchor
* `.omx/research/grand_council_t3_strategic_symposium_50_dollar_budget_20260517.md` Decision #3
* `.omx/research/comprehensive_codebase_distillation_synthesis_20260517.md` META-INSIGHT
* `src/tac/substrates/pretrained_driving_prior/composition.py` — canonical compose_with API
* `src/tac/substrates/pretrained_driving_prior/prior_application.py` — DashcamPriorLoss
* `tools/build_dp1_plus_fec6_composition_packet.py` — PATH 1 build tool
* `experiments/train_substrate_pr101_with_dp1_prior_regularizer.py` — PATH 2 scaffold
* `src/tac/tests/test_dp1_plus_fec6_composition.py` — 32 tests
* `src/tac/tests/test_train_substrate_pr101_with_dp1_prior.py` — 19 tests
* `.omx/operator_authorize_recipes/dp1_plus_fec6_composition_modal_paired_dispatch.yaml` — PATH 1 paired-axis recipe (research-only control after 2026-05-18 rate correction)
* `.omx/operator_authorize_recipes/substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch.yaml` — PATH 2 scaffold recipe (research_only)


# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:design_memo_references_cooperative_receiver_atick_redlich_or_wyner_ziv_framework_in_cross_reference_or_spatial_not_temporal_context_NOT_as_substrate_central_predictive_coding_claim_per_catalog_311_z6_z7_z8_pattern_h_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
