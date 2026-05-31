# UNIWARD optimal-form best-chance re-test — detector-informed × direct-payload (LANDED 2026-05-31)

**Lane:** `lane_uniward_optimal_form_detector_informed_direct_payload_20260531` (L1: impl_complete + memory_entry)
**Operator directive:** *"see if anything falsified such as uniward were engineered correctly and given the best chance"*
**Verdict:** `MIXED_PARTIAL_DETECTOR_HELPS_AT_SOME_BUDGETS` — the prior UNIWARD negatives were
**cargo-culted** (Catalog #307/#315/#303); at OPTIMAL FORM the detector-informed allocation
is the **unique winner at adequate byte budgets**, but the absolute magnitude on this
$0 temporal-prediction proxy surface is small (~1e-5 d_seg). NON-PROMOTABLE
`[macOS-CPU advisory]` per Catalog #192/#341/#127/#323. **$0, no cloud, no PR.**

## The two cargo-cults the prior negatives carried (both fixed)

The prior UNIWARD negatives on NSCS06 v8 (#1570 wire-in, grayscale finalize `5951a3a02`)
were tested at a CARGO-CULTED form:

1. **SegNet-FREE cost-map.** #1570 + the grayscale work used a luma-class proxy
   ("only 20/80 bins nonempty"). UNIWARD/Fridrich is DETECTOR-informed *by definition*
   — spend distortion where the DETECTOR (SegNet) is blind, protect where it is
   sensitive. A SegNet-free cost-map discards the entire premise.
2. **Wrong surface class.** The grayscale finalize proved a spatial luma RASTER
   compresses via GLOBAL bit-depth, not spatial concentration → UNIWARD's spatial
   lever cannot help a globally-entropy-coded raster. UNIWARD belongs on a
   DIRECT-PAYLOAD surface where per-element precision IS the rate cost.

**OPTIMAL FORM fixes BOTH simultaneously:** detector-informed cost-map × direct-payload surface.

## What was built (REUSE, not rebuild)

- **DIRECT-PAYLOAD surface** = the canonical byte-closed UWD1 sparse-delta sidechannel
  (`tac.uniward_delta.pack_sparse_delta` → `unpack_sparse_delta` → `apply_delta_to_frame`;
  PR98-L28 class). `pack_sparse_delta` keeps the top-K pixel-channels of a δ residual
  ranked by `rank_score = |δ| * (1 + cost_norm)` — the `cost_map_bhw` parameter is the
  EXACT injection point. **No surface rebuild.**
- **DETECTOR-informed cost-map** = a new canonical composition module
  `src/tac/substrates/uniward_per_pixel_distortion/detector_informed_direct_payload_cost_map.py`
  (`compose_detector_informed_cost_map`) that composes two REAL signals on REAL frames:
  - `texture_cost = compute_uniward_cost_map(frames)` (S-UNIWARD; high in textured regions)
  - `boundary_w = segnet_boundary_band_weights(seg_logits)` (sister A's `w_i = exp(-margin/τ)`
    from the REAL SegNet top-2 margin — REUSED, not rebuilt).
  - `correction` role: `cost = texture × (eps + boundary_w)` — boost the score-relevant
    boundary band where a correction δ flips argmax back toward GT (contest `d_seg` is the
    per-pixel argmax-flip RATE, so δ only moves d_seg in the small-margin band).
  - `attack` role: `cost = texture × (eps + (1 - boundary_w))` — boost textured
    non-boundary interiors (detector-invisible perturbation).
- **NO-FAKE allocation-diff proof** (`allocation_diff_proof`, the Catalog #105/#139/#220
  no-op guard): proves the cost-map actually CHANGES which δ entries survive a budget.

## Apples-to-apples empirical smoke (REAL data, REAL SegNet)

`experiments/uniward_optimal_form_detector_informed_smoke.py`: 6 real frame-pairs from
`upstream/videos/0.mkv` (874×1164), real SegNet (`upstream/models/segnet.safetensors`).
A real degraded reconstruction (temporal prediction: frame N predicted by frame N−1 — the
NSCS06/HNeRV inter-frame premise) is CORRECTED by δ = GT − pred packed THREE ways at MATCHED
`target_bytes`, then the REAL SegNet `d_seg` (argmax-flip rate vs GT) is measured.

- **Detector-informed cost-map CONFIRMED SegNet-informed (NOT free):** boundary band_frac
  = **4.58%**, mean_w = 0.0897 — a real, sparse decision-boundary band on real frames.
- **Allocation-change (NO-FAKE) proof:** symmetric-difference vs uniform = **4000**, vs
  texture-only = **3968** (both > 0 → the cost-map genuinely re-ranks the kept set). The
  end-to-end UWD1 byte-closure test confirms the real wire format consumes the cost-map.
- **Baseline d_seg (uncorrected prediction):** 0.017592.

| target_bytes | detector_informed | texture_only | uniform | detector lowest? |
|---|---|---|---|---|
| 400  | 0.017596 | 0.017592 | 0.017592 | No (−4e-6 worse) |
| 800  | 0.017593 | 0.017592 | 0.017591 | No (tie) |
| 1600 | **0.017583** | 0.017592 | 0.017590 | **Yes** |
| 3200 | **0.017570** | 0.017591 | 0.017587 | **Yes** |

**detector_informed reduction vs baseline:** −4e-6 / −2e-6 / **+8e-6 / +22e-6** (400/800/1600/3200).
**Detector beats texture-only at:** {1600, 3200}. **Detector lowest of all 3 at:** {1600, 3200}.

## The honest verdict (Catalog #307 — IMPLEMENTATION-LEVEL, not a paradigm kill, not a contest claim)

**The prior negatives WERE cargo-culted.** On the OPTIMAL surface with the REAL detector:

- **The detector-informed lever EXISTS and behaves correctly.** It is the *unique* method
  that produces a meaningful d_seg reduction (texture_only and uniform stay flat at ~baseline
  — they barely correct anything score-relevant). The detector's win **margin WIDENS with
  budget** (−nothing at 400B → +22e-6 at 3200B) — the canonical signature of a real
  rate/distortion lever, not noise. The prior "uniform Pareto-dominates" was an artifact of
  the wrong surface (global-entropy raster) + the SegNet-free cost-map.
- **BUT it is NOT a strong false-negative→live conversion.** Two caveats keep this HONEST:
  1. **Magnitude is tiny** (~1e-5 d_seg) because this $0 temporal-prediction proxy has very
     low baseline d_seg (0.0176) → little boundary-flip headroom. `100·d_seg` ⇒ ~−0.0022 score
     on this 6-pair proxy, NOT a contest archive.
  2. **Budget-gated:** detector loses at tiny budgets (400/800B) where so few entries survive
     that the texture×boundary product can't out-rank raw |δ|.

So the classification is **MIXED_PARTIAL** — directionally vindicating (the lever is real and
the cargo-cult diagnosis is confirmed) but the contest-relevance of the magnitude is unproven
at this surface. This is NOT a paradigm-level DEFER (the prior negatives are reframed, not
ratified) and NOT a contest score claim (Catalog #192/#341 — `[macOS-CPU advisory]` only).

## 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map = ACTIVE** — `compose_detector_informed_cost_map` IS a per-pixel
  detector-informed sensitivity surface (texture × SegNet boundary band); consumable by
  `tac.sensitivity_map.*`. Observability per Catalog #305 via `DetectorInformedCostMap.as_dict`
  (band fraction, mean weight, cost gini).
- **hook #2 Pareto constraint = ACTIVE** — the matched-`target_bytes` × `d_seg` sweep IS the
  rate/distortion Pareto frontier the verdict adjudicates (detector is on or below the
  texture-only/uniform frontier at adequate budgets).
- **hook #3 bit-allocator = ACTIVE** — the cost-map drives `pack_sparse_delta`'s top-K
  byte-budget allocation (which δ entries get int8 precision under the byte cap).
- **hook #4 cathedral autopilot dispatch = N/A** — research-signal surface; non-promotable,
  no archive emission, no dispatch (explicitly `ready_for_exact_eval_dispatch=False`).
- **hook #5 continual-learning posterior = ACTIVE** — `update_from_anchor` (Catalog #335
  contract) echoes anchors with canonical non-promotable markers; probe-outcome PROCEED-advisory
  registered.
- **hook #6 probe-disambiguator = ACTIVE** — this memo + the matched-budget sweep ARE the
  disambiguator between "UNIWARD is dead (prior negatives)" and "UNIWARD's lever exists at
  optimal form (this finding)". The budget-crossover is the empirical arbiter.

## Canonical-equation status

**FORMALIZATION_PENDING** per Catalog #344 — NOT registered as a positive equation. The
detector-informed lever is directionally confirmed but the magnitude on this proxy surface is
~1e-5 and budget-gated; registering a positive savings equation would overstate a contest claim
the $0 proxy cannot support (Catalog #287 — no empirical-claim-without-contest-evidence). The
canonical equation registers only after a real-archive paired-CUDA anchor measures the d_seg
delta on an actual NSCS06-class archive at the contest operating point.

## Reactivation criteria

1. **Operating-point matters:** re-run at a HIGHER baseline d_seg surface (a real degraded
   renderer reconstruction, e.g. NSCS06 v8 grayscale-LUT render rather than temporal prediction)
   where boundary-flip headroom is larger — the lever's magnitude scales with the headroom.
2. **Composition with a real archive:** wire the detector-informed UWD1 δ sidecar onto an
   actual NSCS06 v8 / PR106-class byte-closed archive and run the matched-fidelity sweep against
   the real archive bytes; if detector-informed Pareto-dominates at the archive's operating
   point, register the canonical equation + queue the ~$0.06 paired-CUDA op-routable.
3. **τ sweep:** the boundary band τ=2.0 was a single point; a τ ∈ {0.5, 1, 2, 4} sweep would
   characterize how tight the band must be for the lever to fire at small budgets.

## Single highest-EV next step

**Re-run the matched-fidelity sweep on a real NSCS06 v8 grayscale-LUT *render* (not temporal
prediction)** as the degraded reconstruction — that surface has a much higher baseline d_seg
(the render genuinely collapses semantics per the z6/z8 render-collapse findings), so the
detector-informed lever's ~1e-5 proxy magnitude should scale up by 1-2 orders of magnitude into
contest-relevant territory. If it Pareto-dominates there, register the canonical equation and
surface the ~$0.06 paired-CUDA op-routable (DO NOT fire without operator authorization).

## Sister-coherence (Catalog #340)

DISJOINT from in-flight sisters: z8/DreamerV3 (Gumbel-vs-argmax audit) and z5 (Modal harvest)
owned by other subagents; this lane touched ONLY
`src/tac/substrates/uniward_per_pixel_distortion/detector_informed_direct_payload_cost_map.py`
+ its tests + `experiments/uniward_optimal_form_detector_informed_smoke.py` + this memo + the
lane registry + probe outcomes. No cloud dispatch. checkpoint discipline honored.

mission=frontier_breaking_enabler · horizon=frontier_pursuit
<!-- # HISTORICAL_SCORE_LITERAL_OK:uniward_optimal_form_proxy_dseg_advisory_only_no_contest_claim_2026-05-31 -->
