---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "The grad-share shifts are single-digit-percent relative. This is a SECONDARY multiplier on a SECONDARY lever. Do not let the elegance of the Kronecker derivation inflate the expected ΔS — the only number that counts is the byte-closed n600 A/B, and I predict it will be small. Build banked, activation stays gated."
council_assumption_adversary_verdict:
  - assumption: "The cached per-pair sR (through-R + SegNet fragility-weighted Jacobian at the GT target) is the right S_R object, vs the pure R-chain static map."
    classification: HARD-EARNED
    empirical_verification_status: VERIFIED_VIA_EMPIRICAL_ANCHOR
    rationale: "MEASURED 2026-07-05: the exact R-chain-only column-norm map (pair-independent, ripple ±6.6%) correlates with cached sR at Pearson 0.043±0.003 (n24) ≈ chance — the SegNet content term dominates. The pair-dependent cache design is confirmed correct; the static map is NOT a substitute."
  - assumption: "The uint8-STE identity-gradient convention makes the linear-chain column norm the exact Jacobian surrogate."
    classification: HARD-EARNED
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
    rationale: "tools/precompute_sR_reachability.py::_R_torch L108 (q = clipped + (rounded-clipped).detach()) and the MLX fused-R use the same STE; clamp[0,255] treated as identity is MEASURED to affect only 0.88% of camera pixels (mean, n24 sample)."
  - assumption: "The reachability multiplier LOWERS the exact n600 d_seg."
    classification: ASSUMED_AWAITING_VERIFICATION
    empirical_verification_status: ASSUMED_AWAITING_VERIFICATION
    rationale: "Grad-geometry shifts are directionally right (below) but the payoff claim needs the byte-closed n600 A/B arm (operator GO). Verdict-status for the payoff: PROVISIONAL-PENDING-VERIFICATION."
council_decisions_recorded:
  - "op-routable #1: the #205-class isolated A/B arm (--margin-saliency-weight >0 --margin-saliency-reachability, gt_n600 sidecar ready) awaits operator GO — no autonomous heavy GPU."
  - "op-routable #2: append the 2026-07-05 anchors (S_geo-chance, grad-share table, bitwise proofs) as a third EmpiricalAnchor on margin_saliency_reachability_replaces_texture_proxy_v1 in the registry."
related_deliberation_ids: [council_grand_symposium_levelset_loss_geometry_20260705]
---

# #268 S_R reachability weight — rigor-completion build (2026-07-05)

**Axis: [macOS-CPU/MLX advisory] NON-PROMOTABLE throughout. Pointer 0.19110 UNMOVED (all of this is MEANS).**

## Proactive-recall state (what was ALREADY landed — not rebuilt)

`f99a3863a` (2026-07-03) landed the core of #268: `tools/precompute_sR_reachability.py`
(per-pair through-R fragility-weighted margin-Jacobian S_R, cached as `sR`), the
default-OFF trainer flag `--margin-saliency-reachability` (LEVER-4 `sal *= sR[pi]`,
replaces the MEASURED-inert texture proxy), the DSL gauge leg
(`MarginSaliencyGauge.THROUGH_R_REACHABILITY`), the equations leg
(`margin_saliency_reachability_replaces_texture_proxy_v1`, registered, 2 anchors), and
DAG FEED 2026-07-03p. This build COMPLETES the rigor contract around that landing —
nothing was duplicated.

## What this build added (commits `807c5b3da`, `5b333b709`)

1. **The static-map derivation (the open design question) — ANSWERED WITH MEASUREMENT.**
   `src/tac/boundary_math/sr_rchain_gain.py`: under the STE-identity convention the linear
   R-chain Jacobian is separable, and the L1 column norm of the Kronecker product
   factorizes EXACTLY: `||col_(r,c)||_1 = a_v[r]·a_h[c]` — ONE static (384,512) map,
   pair/θ/video-independent (the beautiful simplification holds for the R-CHAIN FACTOR).
   Torch parity: 1-D matrices ~1e-14, VJP-of-ones 1.3e-14, delta abs-column 1.8e-15.
   **BUT measured decomposition:** S_geo ripple is only ±6.6% (range 0.933–1.272, mean
   1.072) and correlates with the cached content-dependent sR at **Pearson 0.043±0.003
   (n24) ≈ chance** — the SegNet content term CARRIES the reachability signal. Verdict:
   the landed per-pair cached design is CORRECT; the static map is the exact numpy
   reference for the linear chain, not a weight substitute. Clamp-identity caveat
   measured: 0.88% of camera pixels saturated (mean, n24).

2. **Validation vs the memo's empirical measurement + proxy baselines (same 24 frames):**
   | signal vs cached sR (n24) | Pearson |
   |---|---|
   | texture proxy 1/(1+4·tex) | **−0.044 ± 0.030 (chance — replicates memo −0.033)** |
   | static R-chain map S_geo | **+0.043 ± 0.003 (chance-adjacent)** |
   | fragility w=exp(−margin/τ) | +0.200 (replicates memo 0.205) |
   | cached sR vs the memo's measured through-R reachability | **identical BY CONSTRUCTION** (the tool lifts `_compute_SR` verbatim from the $0 probe) |
   | sR band concentration (margin<p10 / margin>p50) | **3.02×** (replicates FEED-03p 3.0×) |
   The premise holds: exact S_R carries the reachability signal; both cheap proxies are at
   chance. "Beats the texture proxy by a large margin" = chance (−0.04) vs identity (1.0).

3. **Grad-share calibration probe** (`experiments/probe_sr_reachability_calibration.py`,
   focal-harness pattern; ep100 EMA `bd_calib_20260705/snap/ema_BEST_ep100.npz`, 12
   gt_n24 pairs; witness-alone surface; grads composed EXACTLY via linearity
   ∇(base+w·msal)=∇base+w·∇msal in float64; artifact
   `experiments/results/sr_calib_20260705/sr_calib_ep100.json`):
   | variant | island | bulk_boundary | bulk_interior |
   |---|---|---|---|
   | base_only | 3.40% | 10.09% | 86.51% |
   | msal term-only plain | 4.25% | 10.86% | 84.89% |
   | msal term-only tex | 4.25% | 10.86% | 84.89% |
   | msal term-only **reach** | **4.74%** | 10.69% | 84.57% |
   | plain_w2 (total) | 4.10% | 10.61% | 85.28% |
   | tex_w2 (total) | 4.10% | 10.62% | 85.29% |
   | **reach_w2 (total)** | **4.48%** | 10.51% | 85.01% |
   Readings: (a) **tex ≡ plain to 3 decimals at the GRADIENT level** — a NEW independent
   confirmation that the texture multiplier is inert (now on the ep100 total-gradient
   surface, not just the map-correlation surface); (b) reach shifts term-only island grad
   share **+11.5% relative** over plain/tex and dominates at every w ∈ {0.5,1,2}; (c) the
   shifts are MODEST — consistent with the memo's "SECONDARY multiplier, MODEST
   refinement, NOT a step change" honest scope. sal-mass distributions: plain 26.1/69.3/4.6%,
   reach 25.7/67.3/7.0% (island/boundary/interior).

4. **Bitwise byte-identity proof at OFF (the owed established A/B; n6, 2ep, CPU, seed 0):**
   A (baseline) ≡ B (flag present, msal_w=0) ≡ C (repeat): **every checkpoint leaf
   (49 EMA + 90 resume-state) bitwise identical + full JSON telemetry identical** (timing
   fields stripped). D (msal_w=0.5) ≠ E (msal_w=0.5 + reachability): 14 leaves differ =
   genuinely active. Post-sidecar-edit: A2 ≡ A and F(sidecar) ≡ E(main-cache) on every
   leaf except the `__cfg_git_sha` provenance scalar (commits landed between runs —
   correct provenance behavior).

5. **gt_n600 sR readiness (the FEED-03p OWED item) — WITHOUT touching the live cache:**
   trainer loader gained a SIDECAR FALLBACK (main-cache `sR` > `<stem>_sR.npz` >
   fail-closed; inside the flag gate ⇒ OFF path untouched, proven bitwise) +
   `gt_n600_sR.npz` sidecar BUILT (sha `d218d07be92c…`, (600,384,512), mean 0.13975,
   sharded CPU ~0.8 s/frame). The A/B arm can now launch with ZERO rewrite of the live
   gt_n600 cache. gt_n24 also carries `sR` inplace (probe dependency).

6. **Tests:** `src/tac/tests/test_sr_reachability_weight.py` — **20 tests** (1-D partition
   of unity ×2, torch matrix parity ×2, dense-Kronecker closed-form identity, VJP parity,
   delta abs-column parity, static-map structure/determinism/pinned-band ×3, contest-dims
   pin, tool normalization ×3, sidecar/inplace write contracts ×2, /tmp refusal, trainer
   flag + fail-closed + sidecar-fallback source contracts ×2, probe argparse contract).

## Honest schema decisions (documented, not hidden)

* **No new `--sr-reachability-weight` flag.** The strength IS the existing
  `--margin-saliency-weight` (the flag toggles the sal FLAVOR); a second weight would be
  redundant surface. Telemetry stays on the `margin_saliency` loss_terms key + the
  activation line now reports `sR_source` (main-cache vs sidecar). Normalization: the
  term is already scale-free (`sum(hmap)/sum(sal)` — a sal-weighted mean), so mean-1
  renormalization of sal would be a no-op by construction.
* **No `sr_reachability_exact_v1` equation registered** — the law is ALREADY registered as
  `margin_saliency_reachability_replaces_texture_proxy_v1` (2 anchors); a near-duplicate
  would violate consolidation discipline. The new anchors are op-routable #2.

## v5-rider assessment (asked; answered plainly)

`--margin-saliency-reachability` is **NOT a rider**: it is loss-shaping ⇒
trajectory-affecting ⇒ it changes the training path from the first msal-active step. It
is **A/B-owed** as an ISOLATED #205-class arm (the DAG already lists it un-stacked for
clean attribution) and MUST NOT be folded into a live run or a rider batch. Activation
requires operator GO; the n600 sidecar + sidecar-fallback make the arm launch-ready.

## Triality legs

* **DAG**: FEED row appended (sub015 DAG, 2026-07-05) — closes the FEED-03p OWED item.
* **DSL**: gauge leg unchanged and correct (`THROUGH_R_REACHABILITY` maps to the flag; no
  new flag added, so no gauge drift).
* **equations**: memo references the registered
  `margin_saliency_reachability_replaces_texture_proxy_v1`; new-anchor append is
  op-routable #2. <!-- Catalog #344: canonical equation referenced -->

**MEANS, not ends: the pointer moves only through a byte-closed `upstream/evaluate.py`
n600 exact row. Pointer 0.19110 UNMOVED.**
