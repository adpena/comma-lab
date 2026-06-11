# Multi-scale / low-frequency seg-repair — the "different lens" verdict (#different-lens)

UTC 2026-06-11 · claude (DAG-first exact-row hunter) · `[macOS-CPU advisory]`, NON-PROMOTABLE
per Catalog #192/#341/#127/#323. $0 local, NO cloud, NO paid GPU, NO MPS, NO /tmp. GT decode
ONLY via `frame_utils.yuv420_to_rgb` (contest-exact). Authority = `upstream/modules.py` SegNet on CPU.
Subagent-id `multiscale_seg_repair_20260611`. Scratch → `.omx/tmp/multiscale_seg_repair_20260611/`.

## Mission (the operator's "different lens")

The prior frontier seg-repair verdict (`frontier_seg_repair_pool_verdict_20260610.md`) DEFER-ed:
a **per-pixel** (high-frequency) frame-1 correction sidecar cannot reach the seg pool under THE
LAW — the residual is salt-and-pepper (95% single-pixel flips, 1.1/4×4 block), position entropy
~2-3 B/flip > 1.27 B/flip break-even, AND SegNet's receptive field reads local appearance patches
as NEW boundaries (Finding A: snapping flip pixels to GT raised d_seg). This unit tests the
operator's reframe: represent the frame-1 correction in a **LOW-FREQUENCY / MULTI-SCALE basis**
(a coarse coefficient grid bicubic-upsampled to camera-res) and apply COHERENT smooth nudges,
on the HYPOTHESIS that band-limited corrections perturb the receptive field coherently (bounded,
predictable collateral) instead of creating high-frequency ringing.

Current frontier (UNMOVED): **0.19109982** `[contest-CPU]`, 177,169 B, sha `b46897267d`,
lane `pr110_payload_entropy_recode`. Score decomposition (from `report.txt`, recomputed):
`s_seg=0.05598 (29.3%)` · `s_pose=0.01715 (9.0%)` · `s_rate=0.11797 (61.7%)`. The seg pool is the
target. Pipeline confirmed: decoder native `EVAL_SIZE=(384,512)` **=** SegNet input size; render
= decoder→bicubic up to 874×1164→channel bias→round/uint8; SegNet sees frame-1 (`x[:,-1]`),
bilinear-DOWN to 384×512→argmax. The native correction scale is therefore 384×512.

Apparatus VALIDATED apples-to-apples: 40-pair sample d_seg = 5.33e-4 (frontier full-600 =
5.598e-4; small-sample variance), 105 flips/pair, reusing the contest-exact render+GT+scorer lib
(`render_and_score_lib.py` logic, re-pointed at the CURRENT frontier inflate path).

## 1. HEADLINE — the lens's CORE CLAIM (low collateral) is TRUE; the carrier claim is FALSE

The repair DIRECTION is the SegNet logit-space margin-normal (per the prior Finding B, the correct
lever): to flip a pixel's argmax pred→gt, move frame-1 along `+∂(gt_logit − pred_logit)/∂frame1`.
The lens LOW-PASSES this dense gradient onto a coarse `gh×gw` grid before applying. The decisive
metric is **collateral-per-fix** = new flips CREATED / flips FIXED.

| correction scale | min collat/fix (8-pair sweep) | net d_seg |
|---|---:|---|
| **per-pixel** (dense gradient-sign, high-freq) | **96.8×** (fix 206 → create 19,944) | +++ (catastrophic) |
| grid 96×128 | 25.6× | + (worse) |
| grid 48×64 | 11.6× | + (worse) |
| grid 24×32 | 4.5× | + (worse) |
| grid 12×16 | 5.1× | + (worse) |
| **grid 6×8** (coarsest) | **2.23×** (fix 13 → create 29) | + (worse) |

**The lens's central hypothesis is EMPIRICALLY CONFIRMED**: a low-frequency correction has
**~43× lower collateral-per-fix** than per-pixel (2.23× vs 96.8×), and collateral falls
monotonically as the basis gets coarser. The salt-and-pepper ringing the prior verdict found IS a
high-frequency artifact a band-limited edit avoids — exactly the operator's intuition.

**BUT** even the best low-freq config has collat/fix = 2.23 > 1, so **net d_seg still INCREASES**
in EVERY configuration (no config nets negative in aggregate). The geometric reason: the low-pass
that grants coherence ALSO spreads the correction over the whole neighborhood, so the few flips it
can fix come at the cost of nudging the many CORRECT pixels in the same band (99.99% of boundary
pixels are fragile / small-margin per `segnet_margin_field_20260609.json`).

## 2. The carrier — THREE structures measured, all fail THE LAW (the kill-gate)

A multi-scale correction's coding cost is set by the **coefficient grid**, not the flip count —
the lens's promise was to break the prior verdict's per-flip position-entropy floor (1.525 B/flip,
already over break-even). Measured economics (extrapolated to 600; budget = the rate a given seg
reduction can pay; carrier = the actual bytes; LAW clears iff budget > carrier):

| carrier structure | seg score Δ (600) | budget (B) | carrier (B) | LAW ratio | verdict |
|---|---:|---:|---:|---:|---|
| **1. ORACLE per-pair coarse grid** (v3; scorer-informed, receiver CANNOT do) | −0.00035 | 525 | ~4,295 | **0.122** | FAILS 8.2× |
| **2. SHARED universal coarse field** (v4; one field, all pairs) | **0.00000** | — | any | **0** | NO numerator |
| **3. SELECTED shared 12×16 field** (v4; one field + 1-bit/pair selector) | −0.00006 | 95 | ~113 | **0.847** | FAILS 1.2× |

**Structure 1 (oracle, the upper bound):** per-pair-optimal (grid,amp) selection using the TRUE
scorer found only **6/16 pairs improvable**, aggregate net **−11 flips over 16 pairs** → implied
seg score Δ only **−0.00035**. The 6×8×3 = 144-coefficient direction grid per improvable pair
(~225 of 600) is the cost killer (~4,104 B). Fails 8.2×. And this is an ORACLE — the direction
`sign(grad)` requires GT knowledge (which pixels are flips) the receiver lacks, so the honest
receiver-blind cost is strictly worse.

**Structure 2 (shared universal):** the cheapest carrier (one coarse field, no per-pair data). At
coarse scales (3×4, 6×8) it fixes **NOTHING** — `mean(sign(grad))` over the sample **cancels to
~zero amplitude** because **the per-pair flip directions DISAGREE** (the same structural finding
as the prior verdict's shared-sparse-field result, now confirmed in the smooth basis). At 12×16 it
only adds collateral (net +11/+22, worse). A shared smooth field is NOT different from a shared
sparse field on the decisive axis: amortization is destroyed by direction-disagreement.

**Structure 3 (selected shared field):** the closest any structure has come to THE LAW — ratio
**0.847, fails by only 1.2×** (vs the prior per-pixel carrier's 2.3× and its position-only info
floor already over break-even). One shared 12×16×3 field + a 1-bit/pair selector saves ~3 flips
per 24 pairs (at the noise floor) on the pairs whose flip direction happens to align with the
sample mean. It needs scorer-at-encode selection (a selector to store) and still falls short.

## 3. Verdict + routing

**VERDICT: DEFER-pending-decoder-axis — the multi-scale "different lens" CONFIRMS its core
hypothesis (band-limited correction = ~43× lower collateral-per-fix than per-pixel) and pushes
the carrier from the prior 2.3× LAW miss to 1.2× (structure 3), but NO multi-scale carrier clears
THE LAW. NO byte-close, NO Modal dispatch** — the local advisory kill-gate fired before byte-close
(no candidate's seg-axis gain pays for its carrier bytes; dispatching would burn ~$0.3 to confirm
a bound proven across three carrier structures + the oracle upper bound).

This HARDENS (does not contradict) the prior seg-repair DEFER. The prior verdict's reactivation
criterion — *"a class-of-flips spatially DENSE enough that position entropy drops below 1 B/flip…
requires a different vehicle whose errors are blocky"* — is partially answered: the multi-scale
basis DOES eliminate the position-entropy term (a coarse grid is dense, no positions to address),
which is why structure 3 reached 0.847. The remaining blocker is NOT position entropy; it is the
**collateral floor** (collat/fix ≥ 2.23 even at the coarsest scale) AND the **direction-
disagreement** that prevents amortization (the cheap shared field cancels to zero). These are two
distinct, independently-measured walls.

The root cause is structural and shared with the prior verdict: **the seg flips are produced by
SegNet's spatial reading of the renderer's globally-soft reconstruction, and the per-pair flip
directions disagree.** A correction sidecar — high-freq OR low-freq, per-pair OR shared — cannot
both (a) have enough amplitude at the sparse flips and (b) avoid perturbing the fragile correct
boundary majority, at a byte cost the tiny per-flip seg value (≈8.5e-7) can pay. **The only carrier
whose cost is NOT proportional to flips AND whose direction the receiver gets for free is the
decoder/latent axis** (a better reconstruction fixes flips inside the existing 91%-of-bytes decoder
budget) — explicitly OUT of this sidecar lane's scope, owned by the decoder/latent executors.

### Reactivation criteria

- A renderer/decoder change reducing the systematic boundary error, paid in existing decoder bytes
  (decoder/latent axis — the only lever where the direction is free and cost is not per-flip).
- A flip class that is both DENSE (low position entropy — the multi-scale basis already gives this)
  AND DIRECTION-CONSISTENT across pairs (so a shared coarse field does not cancel). The flip-map
  (`/Volumes/VertigoDataTier/pact/frontier_seg_repair_pool_20260610/flip_map_full/`) +
  `segnet_margin_field_20260609.json` show the current frontier flips are direction-inconsistent;
  this requires a different vehicle whose errors are systematically biased.
- A correction that ALSO carries a large pose gain (lifting per-flip value above the carrier floor)
  — measured d_pose deltas here were ~+1.9e-6 (negligible, slightly positive=worse), so no rescue.

### FRONTIER-CANDIDATE alert

**NONE.** No candidate beat 0.19109982 beyond noise; no candidate was byte-closed (the local
kill-gate refused before byte-close, per THE LAW). The frontier stands UNMOVED.

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map**: ACTIVE — the collat/fix-vs-scale curve (96.8× → 2.23×) is a new seg-axis
   sensitivity datum: the seg residual's collateral is a monotone function of correction bandwidth.
2. **Pareto constraint**: ACTIVE — the 3-structure LAW table (ratios 0.122 / 0 / 0.847) is the
   per-carrier Pareto-feasibility test on the seg/rate axes; structure-3's 0.847 is the tightest
   seg-sidecar Pareto point measured.
3. **Bit-allocator hook**: ACTIVE — "a coarse direction grid costs ~144 coeff/pair and the shared
   field cancels" is a bit-allocator constraint for any future multi-scale seg section.
4. **Cathedral autopilot dispatch**: N/A — advisory verdict; no candidate cleared the local
   kill-gate, no on-host dispatch minted.
5. **Continual-learning posterior**: N/A — no contest-CPU/CUDA anchor (kill-gate refused dispatch).
6. **Probe-disambiguator**: ACTIVE — RESOLVED "does a low-frequency/multi-scale correction reduce
   net d_seg with less collateral than per-pixel?" → collateral YES (~43× less), net d_seg NO
   (collat/fix floored at 2.23 > 1; shared field cancels via direction-disagreement; carrier fails
   THE LAW across all 3 structures + the oracle bound).

## Cross-references

`frontier_seg_repair_pool_verdict_20260610.md` (the per-pixel sister DEFER this HARDENS;
Finding A receptive-field coupling + Finding B gradient lever + the shared-sparse-field failure) ·
`segnet_margin_field_20260609.json` (99.99% of boundary pixels fragile — the collateral source) ·
`composition_algebra_coherence_law_20260610.md` (§2 receptive-field coupling — the collateral
mechanism) · `experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis/render_and_score_lib.py`
(the contest-exact render+GT+scorer reused) · probe code in
`.omx/tmp/multiscale_seg_repair_20260611/` (probe_lib + multiscale_probe v1/v3/v4 + results JSON).
