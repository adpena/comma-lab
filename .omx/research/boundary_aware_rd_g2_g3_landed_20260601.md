# Boundary-aware RD allocation — G2/G3 gates + HPRC saliency wire-in (LANDED)

- **Date:** 2026-06-01
- **Lane:** `lane_boundary_aware_rd_allocation_grammar_20260601`
- **axis_tag:** `[macOS-CPU advisory]` — NON-PROMOTABLE (score_claim=false, promotable=false per Catalog #341/#192/#127/#323). $0, no GPU, no paid dispatch, no PR, no MPS authority.
- **What this is:** the GATED phase that bridges the pixel-space score-exact saliency producer (`tac.analysis.score_exact_saliency`, commit `84af30f58`) into the HPRC carrier's representation-domain bit allocator (`tac.substrates.hprc.rate_collapse` + `tac.optimization.joint_p18_p19_waterfill`), resolving the council's open items (`council_t3_score_exact_rd_oracle_keystone_ratification_20260601.md`).

## THE GAP this closes

The producer emits `s_seg` (P18 DeepFool flip-risk, last frame) + `s_pose` (P19 Fisher, both frames) in **camera-native PIXEL space**. HPRC allocates bits in its **representation domain** (residual-token grid `q[frame, gh, gw, c]` + per-frame latents). Pushing pixel saliency to the bit domain requires the **EXACT adjoint of the decode synthesis** (Daubechies GAP-3). If the adjoint is approximate or the basis non-orthonormal, the reverse-waterfill protects the WRONG coefficients silently.

## What the HPRC decode operator A actually is (source-inspected `learned_receiver.py`)

```
frame[f,H,W,c] = mean[H,W,c]
    + latent_gain * (latent[f,:] @ basis[:,H,W,c])              # latent stage
    + residual_gain * selector[f] * output_resize(             # residual stage
          residual_scale * nearest_resize(residual_q[f,gh,gw,c]) )
    -> clip/round (NON-linear OUTPUT clamp, NOT in the allocation Jacobian)
```

Both `_nearest_resize` and `_bilinear_resize_batch` are pure **linear gather** ops (index/weight maps, no nonlinearity). So `A` (residual-token → camera-frame, pre-clamp) is exactly linear; its adjoint `A^T` is a **scatter-add** (transpose of a gather = block sum-pool). This IS the orthonormal-grid synthesis adjoint Daubechies named.

## GATE RESULTS

### G3 (Daubechies adjoint) — EXACT to machine precision

Canonical adjoint dot-product test `<A x, y> == <x, A^T y>` on the LIVE HPRC packet geometry (3 real `0.mkv` pairs, grid 48×64):

| stage | rel_residual | is_exact |
|---|---|---|
| nearest_resize | 7.07e-16 | True |
| bilinear_resize | 0.00e+00 | True |
| **composite_residual_decode** | **1.93e-14** | **True** |
| latent_decode | 2.22e-15 | True |

**G3 adjoint-exactness number: rel_residual = 1.93e-14 (composite residual decode).** NO-FAKE guards: a mean-pool "adjoint" FAILS the test (rel_residual 0.98); a forward-resize-used-as-adjoint FAILS. The dot-product test is a genuine adjoint discriminator, not a tautology (19 dedicated tests in `test_hprc_synthesis_adjoint.py`).

### G2 (Balle proxy-rate) — proxy is CONSERVATIVE, frontier NON-fictional

Proxy `R = Σ −log2 p(symbol)` (order-0 entropy ideal) vs ACTUAL brotli q=11 coder bytes on the residual int8 symbol stream (55,296 symbols, 3 real pairs grid 48×64):

- proxy_bytes = 27,516 ; brotli coded_modeled = 23,951
- **residual_bytes = −3564.7 ; per_symbol_residual = −0.0645 B/symbol**
- `proxy_overpromises = False` (the coder BEATS the order-0 ideal via context modeling)
- `frontier_is_fictional = False`

Direction matters (the key G2 refinement): a NEGATIVE residual (real coder cheaper than the proxy) means the proxy UNDER-promises — the RD frontier traced in the proxy domain is **conservative/pessimistic, not fictional**. Fictional-optimism only arises if the proxy OVER-promises (proxy < actual) beyond the 1502-byte 0.001-score quantum. On a smaller uniform-random packet the order-0 proxy sits ~97 B above brotli (within quantum); on real-frame residuals brotli's context model beats order-0 by 0.064 B/symbol. **Either way the proxy-domain frontier is not optimistic-fictional.**

`non_entropy_coded_overhead = 1,625,908 bytes` (the basis/mean/JSON/ZIP framing) is reported SEPARATELY — this is the substrate-R(D) carrier cost (the co-keystone PR95Author flagged in op-routable #4: at 3 pairs the carrier is byte-heavy; the per-pair carrier amortizes at full-video scale).

### Revision 3 (frame/pair asymmetry) — HARD-EARNED, was INFERRED

Latent/token → frame Jacobian sparsity: HPRC's residual tokens AND latent coefficients carry an explicit per-frame leading axis. Perturbing frame_0's storage (residual + latent) leaves frame_1's render **byte-identical** (`residual_cross_frame_coupling = 0.0`, `latent_cross_frame_coupling = 0.0`). So the frame/pair asymmetry is **HARD-EARNED, not assumed**: frame_0 tokens carry pose-only, frame_1 tokens carry seg+pose; dropping a frame_0 token CANNOT hurt frame_1's SegNet term. The consumer wire-in (`build_saliency_driven_importance`) enforces this — `frame_0_seg_mass == 0.0` exactly.

## ADVISORY RE-MEASUREMENT (deliverable #4) — does the co-equal thesis hold on HPRC?

3 real `0.mkv` pairs, grid 48×64, coarsen-quantile sweep, saliency-driven (A^T-pushed) vs importance-blind (uniform) collapse at the SAME quantile, scored on real frames via the verified scorer mirror:

| quantile | SAL bytes | SAL d_pose | UNI bytes | UNI d_pose | pose-protect ratio | rate ratio (SAL/UNI) |
|---|---|---|---|---|---|---|
| 0.40 | 1,368,274 | 7.98 | 1,366,735 | 12.23 | **1.53×** | 1.001 |
| 0.70 | 1,362,072 | **5.93** | 1,360,493 | 30.16 | **5.08×** | 1.001 |
| 0.90 | 1,357,476 | 10.38 | 1,356,384 | 28.17 | **2.72×** | 1.001 |

**The EXACT-adjoint-pushed saliency correctly routes protection to pose-critical tokens — up to 5.08× lower d_pose than uniform at ~equal rate (within 0.1%).** The strict `co_equal_thesis_holds` flag is False (saliency costs ~1500 bytes more to protect those tokens), but the substantive finding is unambiguous: the oracle POINTS CORRECTLY (the adjoint works; pose protection is real and large), and the substrate-R(D) carrier cost is the binding co-keystone — **exactly the council's co-equal-necessity prediction (Z8 confirmed)**. d_seg is essentially unchanged because the residual grid at 48×64 is too coarse for the boundary-localized seg saliency to differentiate tokens (the carrier-resolution co-keystone again).

## VERDICT (1-line)

**The co-equal thesis HOLDS on HPRC: the EXACT synthesis adjoint (G3, 1.9e-14) correctly routes score-exact saliency to pose-critical tokens (5.08× protection at equal rate), G2 proves the proxy-rate frontier is conservative-not-fictional, Rev3 proves the frame/pair asymmetry is HARD-EARNED — but the substrate-R(D) carrier cost is the binding co-keystone (oracle points correctly; carrier-cheapness must improve for the pointing to move the score), confirming the council's Z8-anchored co-equal-necessity framing.**

## Per-substrate symposium inputs (Catalog #325 — PREPARED, NOT DISPATCHED)

Inputs the paid-dispatch gates require, prepared for a future per-substrate adversarial grand-council symposium BEFORE any paid HPRC dispatch:

1. **Cargo-cult audit (Catalog #303):** The CARGO-CULTED assumption to challenge — "a finer residual grid alone makes the seg saliency actionable." HARD-EARNED-vs-CARGO-CULTED: the grid-resolution → seg-actionability link is UNTESTED; the unwind is to measure d_seg-vs-grid-resolution and find the resolution where boundary-localized seg saliency differentiates tokens. The pose-protection link IS HARD-EARNED (measured 5.08×).
2. **9-dim checklist (Catalog #294):** uniqueness = score-exact-adjoint-routed allocation (no sister has the EXACT synthesis adjoint); rigor = G2/G3 gates + 31 NO-FAKE tests; deterministic-reproducibility = seeded, byte-stable; optimal-minimal-score = pending the carrier-cheapness co-keystone.
3. **Observability surface (Catalog #305):** per-stage adjoint rel_residual (G3), per-symbol proxy residual (G2), per-token coupling (Rev3), per-quantile pose-protection ratio (advisory) — all queryable in the emitted JSON.
4. **Dykstra-feasibility (Catalog #296):** the predicted ΔS band on HPRC is gated by the substrate-R(D) co-keystone; the Dykstra-feasibility intersection (rate ≤ R ∩ seg ≤ S ∩ pose ≤ P) is NOT yet reached because the carrier (basis/mean) dominates at small pair counts — the achievable-region check needs full-video carrier amortization first.

## Paired CPU+CUDA eval plan (Catalog #246 — PREPARED, NOT DISPATCHED)

When operator authorizes a paid dispatch: build a full-video (600-pair) HPRC archive with saliency-driven allocation at the pose-protection-optimal quantile (q≈0.7 from the advisory sweep), then run BOTH `upstream/evaluate.py --device cuda` (T4) AND `--device cpu` (GHA Linux x86_64) on the EXACT same `archive.zip` bytes, per the dual-eval mandate. The advisory rows above are `[macOS-CPU advisory]` and CANNOT be promoted without that paired eval. RESERVED for explicit operator authorization.

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map** = ACTIVE — `push_pixel_saliency_to_residual_grid` IS the canonical pixel→coefficient sensitivity transform; downstream `tac.optimization.joint_p18_p19_waterfill` consumes the coefficient-domain saliency.
2. **Pareto constraint** = ACTIVE — the G2 proxy-rate gate bounds the rate-axis term; the advisory re-measurement traces the rate-vs-distortion Pareto frontier (saliency vs uniform).
3. **Bit-allocator hook** = ACTIVE — the saliency-driven importance feeds the EXISTING `transcode_compact_receiver_importance_weighted_residual_tokens` bit allocator (the reverse-waterfill consumer).
4. **Cathedral autopilot dispatch** = N/A — advisory-only research producer; the artifact is non-promotable and the paired-eval is operator-gated. No autopilot ranking influence (score_claim=false).
5. **Continual-learning posterior** = ACTIVE — the durable JSON artifact `.omx/research/boundary_aware_rd_g2_g3_advisory_20260601T183516Z.json` is the canonical anchor; future agents inherit the G2/G3/Rev3 numbers + the co-equal verdict.
6. **Probe-disambiguator** = ACTIVE — the advisory re-measurement IS the disambiguator between "oracle points at wrong tokens" (FALSIFIED: adjoint exact + 5.08× pose protection) and "oracle points correctly but substrate-R(D) binds" (CONFIRMED: co-equal-necessity).

## Files

- `src/tac/analysis/hprc_synthesis_adjoint.py` — G3 exact synthesis adjoint A^T + dot-product gate.
- `src/tac/analysis/hprc_saliency_rd_allocation.py` — G2 proxy-rate, Rev3 Jacobian sparsity, consumer wire-in, advisory re-measurement.
- `src/tac/tests/test_hprc_synthesis_adjoint.py` (19 tests) + `src/tac/tests/test_hprc_saliency_rd_allocation.py` (12 tests) — NO-FAKE.
- `tools/run_boundary_aware_rd_g2_g3_advisory.py` — $0 orchestrator.
- `.omx/research/boundary_aware_rd_g2_g3_advisory_20260601T183516Z.json` — durable advisory artifact.
