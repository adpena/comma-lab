# B1-R3 descent-proof smoke — the refined diagnosis (recon was UNDER-WEIGHTED, not absent)

UTC 2026-06-09 · claude · launched per the T4 grand council verdict
(`feedback_grand_council_symposium_all_results_roadmap_20260609.md`). [macOS-MLX research-signal].
Run: `b1_r3_descent_smoke_20260609T152706Z` (detached) + ep250 harvester (detached, exact eval).

## Strict-scrutiny correction to the earlier diagnosis
Earlier I claimed "B1 had NO RGB anchor." Reading the adapter falsified that: the trainer ALREADY
computes a source-RGB reconstruction loss — `adapter.py:5733  recon = mean((rgb_0-gt_0)^2) +
mean((rgb_1-gt_1)^2)` against `target_rgb_0/1` (the decoded source frames), with an optional
`recon_pixel_weight` margin hook, blended into the total at `:5920  + recon_stage_weight * recon`.
So the anchor EXISTS. The bug was its WEIGHT.

## The R1 / R2 / R3 synthesis (the real root cause)
- **R1** (`b1_229k_pilot_...055851Z`): `--pr95-stage-source-weight-amplification` ON (recon driven)
  but NO grad-clip → DIVERGED in stage-1 CE (loss 18→400).
- **R2** (`b1_229k_clean_...085348Z`): grad-clip ON (stable, nan_inf=0) but DROPPED amplification
  (mislabeled "kitchen-sink") → recon UNDER-WEIGHTED → the per-pair latents went DEAD → the renderer
  emitted 2 fixed latent-independent frames (inspection confirmed) → d_seg=0.50 FLAT over 3000ep.
  Line 5400 even flags `hinerv_full_missing_pr95_source_weight_amplification` — amplification is part
  of the PR95-faithful recipe; R2 relaxed it.
- **R3** (this smoke): amplification ON **+** grad-clip ON = the untested synthesis. Cheap: 600 epochs
  (not 3000), 600 pairs (so the exact 600-pair eval works), R2's full stabilizer set.

## The gate (operator hardening B: exact eval, no proxy-only verdict)
- Early signal: the proxy d_seg trajectory in `telemetry.jsonl` — does it DESCEND (vs R2's flat ~0.50)?
- Authoritative: the ep250 harvester runs the B2 bridge (600-pair `evaluate.py --device cpu`) on the
  backend-only archive → exact d_seg/d_pose → `hi_nerv_backend_only_ep250_exact_eval.json`.
- DECISION: d_seg << 0.50 (descent) → launch the full staged atlas-weighted B1-R3 (then ep250/ep3000
  exact eval vs 0.19199). FLAT → carrier fork (Carmack PR95-faithful config OR SNeRV dense-Y carrier).

## Caveats carried forward (council + operator hardenings)
- This R3 is NOT yet the full "atlas-weighted" trainer — it is the minimal descent-proof (recon-dominant
  + stabilized). The atlas refinements (margin-weighted seg via recon_pixel_weight + Y-dominant pose +
  the staged Stage-0 Y-anchor) layer on top IF R3 descends. Prior session found dense margin-weighted
  recon gave ~0 d_seg gain, so the descent must come from the recon WEIGHT (amplification), not the
  per-pixel margin weighting.
- "dense amortized carrier is the pragmatic bet," NOT "neural decoder theorem" (Assumption-Adversary).
- op-routable #3 still open: pose-output trajectory entropy + inverse-conditioning (cheaper-carrier
  question) — near-full-rank INPUT Jacobian ≠ high-entropy pose OUTPUT.
- The 21 dB single-pair overfit ceiling remains a capacity/topology flag (Carmack) — if R3 descends but
  plateaus high, the PR95-faithful decoder config fork is next.

## Next invocation: read the gate
1. `tail telemetry.jsonl` — proxy d_seg descending?
2. `hi_nerv_backend_only_ep250_exact_eval.json` — exact d_seg/d_pose; run apply_campaign_decision vs 0.19199.
3. descend → staged atlas-weighted B1-R3; flat → carrier fork.
