# PR95-HNeRV inverse-steganalysis carrier — LANDED (2026-06-01)

- **Lane:** `lane_pr95_hnerv_linf_carrier_20260601` (L1)
- **Date:** 2026-06-01
- **horizon_class:** asymptotic_pursuit (Phase-1 RESOLVED carrier of the inverse-steganalysis full stack)
- **axis_tag:** rate term `[exact, directly measured from real carrier bytes]`; carrier render `[macOS-MLX research-signal]`; d_seg/d_pose `[macOS-CPU advisory]`. **NON-PROMOTABLE** (`score_claim=False`, `promotable=False`, `promotion_eligible=False`, `ready_for_exact_eval_dispatch=False`) per Catalog #341/#192/#127/#323. No score claim; paired CPU+CUDA (Catalog #246) reserved for operator authorization.
- **$0 macOS-CPU/MLX-local only.** NO paid dispatch, NO cloud GPU, NO PR, NO Modal/Vast/Lightning.
- **Anchor:** `.omx/research/pr95_hnerv_linf_carrier_head_to_head_20260601T210402Z.json`

## What landed

The THIN wiring that binds the §7-proven **L∞ margin-budget OBJECTIVE** (commit `34fd190f0`, GREEN: L∞ beats L2 56.9% at equal rate, pose-Fisher-dominated) onto the **RESOLVED PR95-HNeRV CARRIER** (operator clarification 2026-06-01: *"consider PR95 for HNeRV's role"*). This completes the co-equal keystone for the PR95-HNeRV carrier choice: { score-exact oracle objective (§7 GREEN), a cheap-by-construction carrier (PR95-HNeRV) }.

- `src/tac/analysis/pr95_hnerv_linf_carrier.py` — the carrier+allocator wiring (CONSUMES the canonical primitives; no carrier rebuilt):
  - `carrier_rate_term` — parse the REAL PR95-HNeRV public archive → cheap-by-construction rate term `25·bytes/N`.
  - `load_carrier_decoder` + `render_carrier_pair_bcthw` — load the REAL trained carrier (canonical `load_pytorch_state_dict_into_mlx`, NOT default-init) + render real per-pair content.
  - `push_pixel_saliency_to_latent` — the **Fisher-pullback** `s_latent_k = Σ_i (∂frame_i/∂z_k)² s_pixel_i` (diagonal of `Jᵀ diag(s_pixel) J`) computed EXACTLY by central finite differences through the SHIPPED carrier decoder (decoder-agnostic; the carrier's 28-d latent Jacobian).
  - `allocate_latent_linf_vs_l2` — the §7-proven L∞-vs-L2 allocation in the carrier's 28-d latent domain (reuses canonical `allocate_linf_margin_budget` + `margin_budget_from_saliency`; `disadvantage_linf` anti-gaming guard forces L∞ ≥ L2 rate).
  - `measure_carrier_distortion` — advisory carrier d_seg/d_pose vs REAL `0.mkv` gt via the bit-exact CPU mirror `measure_pair_d_seg_d_pose`.
  - `z8_falsification` + `build_head_to_head_row` — the deliverable.
- `tools/run_pr95_hnerv_linf_carrier.py` — the CLI runner; emits the advisory head-to-head JSON.
- `src/tac/tests/test_pr95_hnerv_linf_carrier.py` — **23 NO-FAKE tests** (Slot EEE 5-class: would FAIL if any body were `return canonical_markers`; run on the REAL carrier + REAL `0.mkv`).

## The head-to-head row (advisory)

| axis | PR95-HNeRV carrier | Z8 (emergent-from-fidelity) |
|---|---|---|
| archive bytes | **178,321 B** (`--modelsize` budget, cheap-by-construction) | **28,406,255 B** (wavelet detail = 99.894% of `0.bin`) |
| rate term (`25·B/N`) | **0.1187** | **18.91** |
| advisory d_seg `[macOS-CPU advisory]` | **0.0013** | n/a (different carrier) |
| advisory d_pose `[macOS-CPU advisory]` | **3.84e-4** | n/a |
| advisory distortion-only score | **0.189** | — |
| advisory full-score estimate (rate + distortion) | **0.307** | 104.94 (600-pair byte-closed, 546× from frontier) |
| L∞-vs-L2 latent objective (§7, 28-d domain) | L∞ 112.1 bits vs L2 112.0 bits, **allocations DIFFER** | — |

(4 strided real `0.mkv` pairs, carrier latents 1:1 mapped; render `[macOS-MLX research-signal]`.)

## Z8-falsification verdict — CONFIRMED

**PR95-HNeRV rate (0.1187, 178,321 B) ≪ Z8 rate (18.91, 28,406,255 B) — Z8 is 159× heavier.** The carrier's *entire rate term* (0.119) is below the Z8 full byte-closed score (104.94). The carrier reaches a competitive advisory full-score estimate (0.307; the frontier carrier is 0.192) at 159× fewer bytes than Z8. This is the **cheap-by-construction** keystone half made concrete: PR95-HNeRV bytes are a budget (amortized INT8 decoder + 28-d per-pair latents); Z8 bytes are emergent-from-fidelity (raw-float wavelet detail). Z8's optimal joint-P18/P19 dead-zone allocation could not move its score because the carrier R(D) is the bottleneck — exactly the co-equal-necessity the T3 council established. The PR95-HNeRV carrier resolution is empirically vindicated at $0.

## Honest scope / nuance

- The carrier render uses the MLX decoder whose PyTorch forward parity is a known blocker (`PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER`); the render is `[macOS-MLX research-signal]` and the d_seg/d_pose are `[macOS-CPU advisory]` (frozen weights but Apple-Silicon CPU, NOT contest GHA-Linux-x86_64). No score claim.
- The L∞ latent allocation at this operating point spends ≈ equal bits as L2 with a DIFFERENT step map (genuine detector-aiming). Per the §7 finding + the HiNeRV sister finding (`e96ef0435`), the carrier byte budget is the dominant lever; the latent-domain L∞ is the proven objective in the carrier's coefficient domain (its marginal leverage on a cheap carrier is second-order, consistent with the §7 pose-Fisher-dominated result and the operating-point analysis).
- Sister-DISJOINT from `lane_hinerv_inverse_steganalysis_carrier_20260601` (HiNeRV, dense decoder-VJP) and `lane_snerv_inverse_steganalysis_carrier_20260601` (SNeRV) per Catalog #340 — different carrier, different module.

## Remaining before a score claim (paid; operator-gated)

Wire the proven L∞ latent allocation into the carrier's archive export → byte-closed `archive.zip` → paired CPU+CUDA on exact bytes (Catalog #246). The build is $0; the paired eval crosses into paid GPU → operator authorization required.

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map — ACTIVE.** `push_pixel_saliency_to_latent` pushes the oracle pixel saliency (`tac.analysis.score_exact_saliency` P18 `s_seg` + P19 `s_pose`) into the carrier's 28-d latent Fisher saliency via the exact decoder Jacobian.
2. **Pareto constraint — ACTIVE.** `allocate_latent_linf_vs_l2` projects onto {latent-rate ≤ R, per-coeff margin budget} via the canonical §7 `allocate_linf_margin_budget` reverse-water-fill (Dykstra-feasibility inherited; Catalog #296 satisfied natively).
3. **Bit-allocator — ACTIVE.** The L∞ allocation IS the per-latent-coeff bit allocator at cost = oracle latent saliency `ρ_k = 1/(s_latent_k + ε)`.
4. **Cathedral autopilot — ACTIVE.** The head-to-head row + Z8-falsification feed candidate ranking (the carrier rate term + advisory full-score estimate are queryable cathedral-consumer signals; NON-PROMOTABLE markers prevent promotion).
5. **Continual-learning posterior — ACTIVE.** The advisory anchor JSON recalibrates the carrier-cheapness + savings model; the §7 GREEN + this carrier wiring close the objective→carrier loop.
6. **Probe-disambiguator — ACTIVE.** The L∞-vs-L2 latent allocation (allocations-differ + oracle-vs-random control in tests) disambiguates the objective in the carrier's coefficient domain; the Z8-falsification disambiguates cheap-by-construction vs emergent-from-fidelity.

## Canonical-vs-unique decision per layer

- Oracle (L0): ADOPT_CANONICAL (`score_exact_saliency`).
- Objective (L1): ADOPT_CANONICAL (§7 `inverse_steganalysis_linf_vs_l2_gate`, GREEN).
- Carrier (L2): ADOPT_CANONICAL (PR95-HNeRV `pr95_hnerv_mlx`, RESOLVED carrier).
- Allocation (L3): ADOPT_CANONICAL (`allocate_linf_margin_budget` + `margin_budget_from_saliency`) + FORK to the carrier's latent domain via the Fisher-pullback.
- The only NEW code is the PR95-HNeRV-specific carrier loader + Fisher-pullback through the carrier decoder + the head-to-head row assembly.

## Cargo-cult audit per assumption

- "the carrier-reconstruction d_seg/d_pose is a contest score" — CARGO-CULTED; unwound: it is `[macOS-CPU advisory]` (frozen weights, Apple-Silicon CPU, MLX render with a parity blocker). NO score claim.
- "the L∞ latent allocation is the dominant lever on this carrier" — CARGO-CULTED at this operating point; HARD-EARNED reframe: the carrier byte budget is the dominant lever (§7 + HiNeRV sister); L∞ is the proven objective whose latent-domain marginal is second-order on an already-cheap carrier.
- "Z8 could match PR95-HNeRV with better allocation" — FALSIFIED; Z8 at 28.4 MB with optimal joint-P18/P19 dead-zone still sits 546× from frontier (the carrier R(D) is the bottleneck).

## Predicted ΔS band

Direction NEGATIVE (score-lowering vs Z8 carrier). Magnitude: the rate-axis alone is decisive (PR95-HNeRV 0.119 vs Z8 18.91); the distortion-axis advisory full-score estimate (0.307) is within ~1.6× of the frontier (0.192) at $0-local on a non-export-closed carrier. No numeric contest ΔS asserted pre-paired-eval. `# PREDICTED_BAND_VIBES_OK:rate-axis-falsification-is-dispositive-159x-byte-ratio-distortion-axis-deferred-to-operator-gated-paired-eval-per-catalog-246`

## Canonical equation reference

`# FORMALIZATION_PENDING: pr95_hnerv_carrier_cheap_by_construction_vs_z8_emergent_rate — register in tac.canonical_equations after the operator-gated paired CPU+CUDA anchor lands; the rate term 25·B/N + the byte ratio are verified-constant/directly-measured, the achievable full-score model is pending the paired empirical.`

## Observability surface

Inspectable per layer (carrier rate term; per-latent `s_latent`; per-coeff L∞/L2 step). Decomposable (rate / d_seg / d_pose / latent-allocation separable). Diff-able (carrier render run-to-run; oracle-vs-random allocation control). Queryable (the head-to-head JSON anchor). Cite-able (carrier archive sha256 + bytes + n_pairs + latent_dim). Counterfactual-able (the wrong-pairing distortion test + the concentrated-vs-uniform pullback test + the oracle-vs-random allocation test).

## Tests / verification

- 23 NO-FAKE tests pass (`src/tac/tests/test_pr95_hnerv_linf_carrier.py`).
- 24 sister HiNeRV tests pass (no regressions).
- ruff clean on all 3 files.
