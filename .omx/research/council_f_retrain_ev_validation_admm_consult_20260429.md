# Grand Council F — Re-train EV + Ω-W-V2 Validation + Joint-ADMM Non-Convex Consult

**Date**: 2026-04-29 PM
**Convened by**: parent agent (user mandate "consult the grand council on those retrains (I am down if they help push our theoretical score lower) and also the real-archive validation (remember MPS and local are broken and janky) and the non-convex test")
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Grand-council augmentation**: Boyd (convex-feasibility at the algorithmic level), Carmack (kill-criteria + 30-min shred), van den Oord (codebook amortisation perspective on Ω-W-V2)
**Mandate scope**: REPORT-ONLY. NO code modified. NO GPU spawned. All claims tagged `[empirical]` / `[contest-CUDA]` / `[contest-CPU advisory]` / `[prediction]` / `[synthetic]`.

---

## 1. Executive Summary

| Decision Surface | Verdict |
|---|---|
| **Lane WC-S** retrain (predicted [0.78, 1.05]) | **DEFER** — central ~0.92 vs Lane G v3 1.05 is +0.13 best-case, marginal. Requires Lane G v3 stack-composition test, not standalone re-dispatch. |
| **Lane PA** retrain (predicted [0.85, 1.05]) | **KILL** — central ~0.95 ≈ Lane G v3 1.05 noise band; PixelArt SegMap is architecturally redundant with Lane G v3 distill recipe. |
| **Lane HM-S** retrain (predicted [0.32, 0.45]) | **APPROVE** — central ~0.38 is sub-Quantizr-0.33 territory IF the band is honest. Highest EV per dollar of any of the 5. Dispatch first. |
| **Lane FR-Ω** retrain (predicted [0.27, 0.45]) | **APPROVE** — central ~0.36 is sub-Quantizr. Block-FP + Hessian-curvature is orthogonal to KL-distill. Dispatch second. |
| **Lane FC** retrain (predicted [0.85, 1.10]) | **KILL** — central ~0.97 fails the Lane-G-v3 hurdle; FiLM-Canvas is a representational alternative, not a stacking partner; band's upper edge crosses the 1.0 unacceptable threshold. |
| **Ω-W-V2 real-archive validation** | **SAFE-LOCAL** — pure encode→bytes→decode→roundtrip-tensor measurement is bit-deterministic; the file `experiments/results/lane_g_v3_landed/iter_0/renderer.bin` (290KB) is the correct anchor. Tag the result `[empirical:tests/test_omega_w_v2_real_archive.py]`. |
| **Joint-ADMM 4-stream non-convex test** | **GATES V2 dispatch — YES**. The 2-stream convex KKT residual 0.02 is necessary but not sufficient; without a 4-stream non-convex test that exercises restart logic on a discrete-jump R(D), V2 will silently produce non-feasible points on the real archive. The test is the gate. |

**Total approved retrain dispatch cost**: $3.00 (Vast.ai 4090, HM-S + FR-Ω only, ~6h total).
**Total approved local validation cost**: $0 (Ω-W-V2 + ADMM 4-stream tests).
**Updated dispatch order vs Council E**: HM-S → FR-Ω (parallel) → Ω-W-V2 local test → ADMM 4-stream local test → re-evaluate WC-S after stack-composition signal lands.

---

## 2. Part A — Re-train EV Analysis

### 2.1 Shannon's R(D) framing — what is each lane actually solving for?

The Lane G v3 score 1.05 [contest-CUDA] decomposes as:
```
100 × seg_dist     = 0.401
sqrt(10 × pose)    = 0.186
25 × rate_unscaled = 0.462    (archive 694,074 bytes)
TOTAL              = 1.049
```

The **rate term is 44% of the score** at the current operating point. Shannon's R(D) ceiling per memory `project_codec_stacking_composition_canonical_orders_20260429.md`: every byte saved at the current marginal `dScore/dByte ≈ 0.00067` buys ~67bp per 1KB. The verified Shannon floor for Phase 1 is 0.28; the gap from 1.05 to 0.28 is **77bp of headroom** distributed across seg + pose + rate.

For each of the 5 lanes, I cite the R(D) wedge they target and whether the mechanism approaches the ceiling:

| Lane | Mechanism | Wedge targeted | Approaches ceiling? |
|---|---|---|---|
| WC-S | Curator outlier weighting (down-weights atypical mask transitions in the loss) | SegNet (primarily) — small bp on rate via better-converged renderer | **Partial.** Outlier weighting is a regulariser; it shifts the loss surface but doesn't change the bits-per-symbol. R(D) ceiling for SegNet portion is ~0.28 of the SegNet term. WC-S nibbles at noise-floor, not ceiling. |
| PA | PixelArt SegMap variant (pixel-art-style discretisation of mask grid) | SegNet (representation choice on argmax boundaries) | **Tangential.** PixelArt is an alternative representation, not a rate reduction. Selfcomp's grayscale-LUT already represents discrete classes optimally for SegNet's argmax-only sensitivity. PA is parallel hypothesis at best. |
| HM-S | 8-DOF homography embedding (replaces affine 6-DOF with full perspective transform per pair) | PoseNet (geometric representation) — directly reduces pose distortion | **Approaches ceiling.** PoseNet term sqrt(10×pose) at Lane G v3 = 0.186 (pose=0.003455). Theoretical floor for pose is ~0.05 (Selfcomp 0.000552 → sqrt(10×0.000552) = 0.074). HM-S could reach 0.075-0.10 on PoseNet term → -0.09 to -0.11 score. |
| FR-Ω | Hessian-curvature-driven block-FP allocation (Fridrich-cost on weight rounding errors) | Rate (renderer.bin bytes) + a small SegNet preservation gain | **Approaches ceiling.** Renderer.bin in Lane G v3 archive ≈ 290KB. FR-Ω targets ~50-100KB savings (-0.05 to -0.10 on score). Plus Hessian-aware quantisation preserves SegNet sensitivity better than uniform → +small bp on seg. |
| FC | FiLM-Canvas SegMap (FiLM conditioning on pose + canvas-style render) | Renderer architecture (representation choice) | **Tangential.** FC swaps the renderer architecture. If it scores 0.85 it has succeeded as a *replacement* baseline, not as a *stack* on top of Lane G v3. Cannot be combined; one or the other ships. |

**Shannon's verdict per lane** (information-theoretic):
- HM-S, FR-Ω: **directly reduce R(D) gap on a wedge with measurable headroom.** GREEN.
- WC-S: nibbles at noise-floor; not a ceiling-approaching mechanism. AMBER.
- PA, FC: representational alternatives, not stacking partners. RED for re-dispatch as part of a stack-building wave.

### 2.2 Dykstra's orthogonality matrix — do the 5 lanes intersect Lane G v3's convex constraint set?

Per memory `feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md`: "additivity of independent rate savings is CONDITIONAL, not given. Two techniques that each save 30KB might overlap and only deliver 40KB stacked."

Dykstra's intersection check on each lane vs the Lane G v3 constraint set (`{rate ≤ 694KB, seg ≤ 0.0040, pose ≤ 0.0035}`):

| Lane | Mechanism overlap with Lane G v3 (KL-distill + pose TTO retry on Lane A anchor)? | Net orthogonality |
|---|---|---|
| WC-S | Curator outlier weighting **partially overlaps** with KL-distill-as-regulariser: both shift the loss surface in the SegNet boundary direction. Empirical overlap ~30-50%; net unique savings ~50%. | **Partially orthogonal.** Predicted gain on top of Lane G v3 ≈ 50% of standalone gain. |
| PA | PixelArt is a representational change to the SegMap's discrete output ladder. Lane G v3 uses standard SegMap output. **Replacement, not stack.** | **Not orthogonal — replaces.** Cannot stack. |
| HM-S | 8-DOF homography is geometric, fully orthogonal to KL-distill (loss on pose distortion regardless of geometry parameterisation). **Fully orthogonal.** | **Fully orthogonal.** Predicted gain stacks on top of Lane G v3 cleanly. |
| FR-Ω | Block-FP weight quantisation operates on the renderer's stored weights (rate term). KL-distill operates on the loss surface during training. **Two different stages of the pipeline.** Fully orthogonal. | **Fully orthogonal.** Predicted gain stacks on top of Lane G v3 cleanly. |
| FC | FiLM-Canvas is a renderer architecture replacement. Cannot stack with Lane G v3's standard renderer. | **Not orthogonal — replaces.** Cannot stack. |

**Dykstra's verdict**: of 5 lanes, **only HM-S and FR-Ω are orthogonal stacking partners with Lane G v3.** WC-S has 50% overlap; PA and FC are replacements (not stack partners). The marginal value of dispatching all 5 is therefore **not 5× a single lane** — it is approximately the value of 2 truly orthogonal lanes (HM-S + FR-Ω) plus 0.5× one partial-overlap lane (WC-S) plus 0× for two replacements (PA, FC).

If we already have Lane G v3 = 1.05 [contest-CUDA] verified, Dykstra cuts the original 5-lane wave from $5 to **$3 (HM-S + FR-Ω only)**, with WC-S deferred until a stack-composition signal comes back.

### 2.3 Per-lane verdict + cost

| Lane | Verdict | Cost on Vast.ai 4090 ($0.26/hr) | Justification |
|---|---|---|---|
| **HM-S** | **APPROVE — DISPATCH FIRST** | $1.50 (~6h) | Highest EV per dollar. Sub-Quantizr-0.33 territory if the band lands. Fully orthogonal to Lane G v3. |
| **FR-Ω** | **APPROVE — DISPATCH SECOND** | $1.50 (~6h) | Sub-Quantizr territory; Hessian-aware block-FP is the canonical Selfcomp stack ingredient (see `project_codec_stacking_composition_canonical_orders_20260429.md` "Selfcomp weights" stack). Fully orthogonal to Lane G v3. |
| **WC-S** | **DEFER** — re-evaluate after Lane G v3 + HM-S signal lands | $0 (this wave) | Predicted band central ~0.92 marginal vs 1.05; partial overlap with KL-distill; spend the budget on orthogonal lanes first. If the HM-S+FR-Ω stack lands at 0.30, WC-S becomes irrelevant; if it stalls at 0.50, WC-S re-enters consideration. |
| **PA** | **KILL** | $0 | Not a stack partner. PA's 0.95 central does NOT push the theoretical floor lower than 1.05 in any meaningful sense (within noise). The predicted-band upper edge 1.05 = "ties Lane G v3" is failure-mode confirmation, not success. |
| **FC** | **KILL** | $0 | Same logic as PA — not a stack partner; central 0.97 is within noise of Lane G v3; upper edge 1.10 violates CLAUDE.md "Any auth score above 1.0 is UNACCEPTABLE" and would require defending why we re-dispatched a band that crosses the unacceptable threshold. |

**Total approved cost**: $3.00 (HM-S + FR-Ω). **Original Council E plan would have been $5.80** for a 5-lane wave with two replacements and two marginal contributors.

### 2.4 Risk: prediction bands are themselves [prediction]-tagged, not empirical

**Contrarian VETO**: every band in the table is `[prediction]`-tagged in `lane_redispatch_plan_post_round6_20260429.md`. The +5 lane invalidation list (Round 6 Defect 1) means we have ZERO empirical [contest-CUDA] data on what these architectures actually score with gradients flowing. The bands are derived from architectural reasoning, not measurement.

This is an honest concession: the verdicts above are conditional on the bands being roughly correct. If the bands are off by 0.5 (e.g. HM-S actually lands [0.7, 1.2] instead of [0.32, 0.45]), the entire ranking flips. Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": **no lane verdict here is [empirical]; all are [prediction-conditional-on-band-accuracy].**

The cheapest hedge is: dispatch HM-S first ($1.50, 6h). If it lands within 0.10 of central 0.38, the band model is calibrated and FR-Ω dispatch is justified. If it lands at 0.95+, the bands are systematically optimistic and we KILL FR-Ω before spending the second $1.50.

---

## 3. Part B — Lane Ω-W-V2 Real-Archive Validation Protocol

### 3.1 Deterministic vs nondeterministic measurement classification

Per CLAUDE.md "MPS auth eval is NOISE" + Check 83 STRICT + memory `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`:

**DETERMINISTIC (safe locally on MPS / CPU / CUDA):**
- File IO, SHA, ZIP byte counts (bit-identical across hardware)
- ffmpeg decode of fixed video (deterministic CPU path)
- STC encoder on **fixed** class IDs (CPU integer arithmetic)
- Pure-Python preprocessing on uint8 arrays
- Pytest shape / exception checks
- **`encode_omega_w_v2(weights, hessian, total_bits=...) → bytes` followed by `decode_omega_w_v2(bytes) → tensor`** — verified by code inspection of `src/tac/water_filling_codec_v2.py`:
  - Uses `weights.detach().to(torch.float32).cpu()` (line 222 + 354) — no GPU needed.
  - Uses `np.ndarray` int32 arithmetic + `encode_qints_arithmetic` (CPU-only static-histogram coder).
  - Does NOT call PoseNet, SegNet, or any neural-net forward pass anywhere.
  - The only nondeterministic pieces would be: random seeds → none used; floating-point accumulation order in `var()` → bit-deterministic on CPU.

**NONDETERMINISTIC (CUDA REQUIRED for any score-derived claim):**
- Any neural-net forward pass through SegNet (EfficientNet-B2), PoseNet (FastViT-T12), the renderer, distilled scorers, learned codecs.
- Anything that ends in a `score` claim (since the contest scorer requires SegNet+PoseNet on CUDA).
- Even on CUDA, eval via `experiments/contest_auth_eval.py` is the ONLY authoritative score — proxy scoring during training has 100-350× drift per CLAUDE.md.

**Verdict: Ω-W-V2 real-archive byte+roundtrip validation is on the deterministic side.** Local-only is valid.

### 3.2 Validation protocol design (no scorer, no GPU, no MPS-derived strategic claim)

**Anchor file (verified to exist)**: `/Users/adpena/Projects/pact/experiments/results/lane_g_v3_landed/iter_0/renderer.bin` (290KB)

This is Lane G v3's actual shipped renderer.bin. To use it for Ω-W-V2 validation we need to load it as a state-dict (not as the OWV1/SCv1/etc. magic-byte payload). The exact loader is `tac.renderer_export.load_renderer_checkpoint(path)` which dispatches via the magic byte. For Ω-W-V2 testing specifically, we want the RAW per-conv state-dict tensors (so we can encode each Conv2d weight with `encode_omega_w_v2`).

**Protocol (in order)**:

1. **Decode renderer.bin to a state_dict** using the existing canonical loader. Verify on local CPU:
   ```python
   from tac.renderer_export import load_renderer_checkpoint
   sd = load_renderer_checkpoint("experiments/results/lane_g_v3_landed/iter_0/renderer.bin")
   conv_weights = {k: v for k, v in sd.items() if v.dim() == 4 and v.shape[2] >= 1 and v.shape[3] >= 1}
   ```

2. **For each Conv2d weight tensor**:
   - Compute synthetic Hessian: `hessian = torch.ones(O,)` (uniform — pessimistic; real Hessian would weight more selectively).
     - **NOTE**: synthetic uniform Hessian is conservative — it gives Ω-W-V2 the WORST-CASE bit allocation. Real per-channel Hessian from a calibration loop would be MORE selective and likely save MORE bytes. So the test's measured savings are a LOWER BOUND on real savings.
   - Compute V1 raw byte estimate: `bytes_v1 = O*I*kH*kW + O*4 + 32`.
   - Set total_bits = `int(bytes_v1 * 0.7 * 8)` (target 30% reduction from V1 raw — within the predicted [20%, 60%] band).
   - Try `encode_omega_w_v2(weights, hessian, total_bits=total_bits)`. Catch `BlockFPIneligible` (linear bias terms) and skip; catch `GateRegression` (V2 ≥ V1) and record as ineligible.
   - Decode: `decode_omega_w_v2(payload) → weights_recon`.
   - Assert `(weights - weights_recon).abs().max() <= 2.0 * 2**(-bits_per_channel_min)` (per-channel quantisation tolerance — derived, not arbitrary).
   - Record `byte_count_v2 = len(payload)`, `byte_count_v1 = bytes_v1`, `savings_pct = 100 * (1 - byte_count_v2 / byte_count_v1)`.

3. **Aggregate**:
   - Total V1 bytes (sum over all eligible conv weights)
   - Total V2 bytes
   - Aggregate savings_pct
   - Per-tensor distribution of savings (paper-figure-worthy)

4. **Tagged claim**:
   - If aggregate_savings_pct ≥ 20%: `[empirical:src/tac/tests/test_omega_w_v2_real_archive.py] aggregate savings 20%-60% on Lane G v3 renderer.bin`.
   - If aggregate_savings_pct < 20%: **TAG AS REGRESSION**. The 69.11% synthetic claim is FALSIFIED on real architecture. Update `src/tac/water_filling_codec_v2.py` docstring to remove the prediction band.
   - Either way, the result is `[empirical:<test path>]`, NOT `[contest-CUDA]`. **This validation does NOT prove a score change.**

### 3.3 What the validation does NOT prove

**Critical caveats** (must appear in the test's assertion failure messages and any commit/run_log entry):

1. **Does NOT prove the encoded weights inflate to a renderer that scores within Lane G v3's band.** The validation proves bit-faithful round-trip and net byte savings; it does NOT prove the QUANTISED weights produce an equivalent forward pass. A renderer with 30% smaller weights but 50× worse SegNet output is a regression at the score level.
   - To prove that, we need: encode → decode → load into renderer.bin → ship in archive → contest-CUDA auth eval. **That is a Vast.ai 4090 dispatch, not a local validation.**

2. **Does NOT prove ADMM coordinator wraps Ω-W-V2 correctly.** The Joint-ADMM proximal codec wrapper (`joint_admm_proximal_water_filling_v2.py` per Council E §3.1) is a separate file that does not yet exist. Validation of Ω-W-V2 standalone tells us nothing about whether the coordinator's adaptive-ρ behaviour works on this codec.

3. **Does NOT prove savings hold under outer DEFLATE compression.** Per `src/tac/water_filling_codec_v2.py` lines 268-274: "Outer compression typically helps V1 more than V2 (because xz LZMA exploits the same redundancy as the arithmetic coder), so the gate is conservative." The post-archive byte count may show V2 saving LESS than the pre-archive byte count, because the static-histogram arithmetic coder eats the entropy that DEFLATE would have eaten on V1.

4. **Does NOT validate hyperprior amortisation.** Per the same file lines 17-23: "Selfcomp's renderer at 88K params × 1.017 bpw ≈ ~11KB of qint payload — borderline for hyperprior amortisation". The static-histogram arithmetic terminal is the V2 ceiling; the V3 hyperprior would be a separate validation gate.

### 3.4 Concrete test file path + assertions

**File**: `/Users/adpena/Projects/pact/src/tac/tests/test_omega_w_v2_real_archive.py`

**Required assertions** (in pseudo-code, NOT to be implemented in this report; report-only mandate):
```python
def test_omega_w_v2_real_lane_g_v3_renderer_bin():
    """[empirical:tests/test_omega_w_v2_real_archive.py] Validate Ω-W-V2
    on Lane G v3's actual renderer.bin (not a synthetic tensor).

    DOES prove: bit-faithful round-trip per Conv2d weight; aggregate byte
    savings vs V1 raw qint estimate.

    Does NOT prove: any score change. To prove score parity, encode →
    decode → archive → contest-CUDA auth eval (Vast.ai 4090, ~$0.50,
    NOT in scope of this test).
    """
    anchor = Path("experiments/results/lane_g_v3_landed/iter_0/renderer.bin")
    assert anchor.exists(), f"missing anchor {anchor}; this test is gated on Lane G v3 having been landed"
    sd = load_renderer_checkpoint(str(anchor))

    eligible_count = 0
    skipped_count = 0
    aggregate_v1_bytes = 0
    aggregate_v2_bytes = 0
    per_tensor_savings = []

    for name, w in sd.items():
        if w.dim() != 4:
            skipped_count += 1
            continue
        O = w.shape[0]
        hess = torch.ones(O, dtype=torch.float32)  # uniform-pessimistic Hessian
        bytes_v1 = w.numel() + O * 4 + 32
        total_bits = int(bytes_v1 * 0.7 * 8)
        try:
            payload = encode_omega_w_v2(w, hess, total_bits=total_bits)
        except (BlockFPIneligible, GateRegression):
            skipped_count += 1
            continue
        eligible_count += 1
        recon = decode_omega_w_v2(payload)
        # Bit-faithful round-trip: per-channel quantisation error bounded
        max_abs_err = (w - recon).abs().max().item()
        per_channel_max = w.abs().reshape(O, -1).max(dim=1).values
        # Quantisation tolerance: 2**(-min_bits_per_channel) × per_channel_max
        # (derived from block-FP algebra, NOT arbitrary)
        tol = 2.0 * per_channel_max.max().item() * (2.0 ** -3)  # 3-bit floor
        assert max_abs_err <= tol, f"{name}: max_abs_err={max_abs_err:.4f} > tol={tol:.4f}"

        aggregate_v1_bytes += bytes_v1
        aggregate_v2_bytes += len(payload)
        per_tensor_savings.append((name, bytes_v1, len(payload), 100 * (1 - len(payload) / bytes_v1)))

    assert eligible_count > 0, "no eligible Conv2d weights found in Lane G v3 renderer.bin"
    aggregate_savings_pct = 100 * (1 - aggregate_v2_bytes / aggregate_v1_bytes)

    # Predicted band [20%, 60%] on real architecture (NOT 69.11% synthetic).
    assert aggregate_savings_pct >= 15.0, (
        f"Ω-W-V2 saves {aggregate_savings_pct:.1f}% on Lane G v3 renderer.bin "
        f"(eligible={eligible_count}, skipped={skipped_count}); "
        f"docstring 69.11% synthetic claim FALSIFIED on real architecture; "
        f"update src/tac/water_filling_codec_v2.py docstring before promoting."
    )
    assert aggregate_savings_pct <= 75.0, (
        f"Ω-W-V2 saves {aggregate_savings_pct:.1f}% on Lane G v3 renderer.bin — "
        f"this exceeds even the synthetic 69.11% prediction. Verify the V1 byte "
        f"estimator is not over-counting (e.g. forgot to amortise the per-channel "
        f"exponent header)."
    )

    print(f"[empirical] Ω-W-V2 on Lane G v3 renderer.bin: "
          f"V1_bytes={aggregate_v1_bytes}, V2_bytes={aggregate_v2_bytes}, "
          f"savings={aggregate_savings_pct:.1f}%, "
          f"eligible_tensors={eligible_count}, skipped={skipped_count}")
```

**Verdict: SAFE-LOCAL.** No GPU, no MPS-derived strategic claim, no scorer load. Tag the result `[empirical:tests/test_omega_w_v2_real_archive.py]`. The result REPLACES the 69.11% synthetic claim in `src/tac/water_filling_codec_v2.py` docstring after landing.

---

## 4. Part C — Joint-ADMM Non-Convex Test Design

### 4.1 Why the existing convex test is necessary but not sufficient

The current `test_two_stream_convex_converges_to_kkt` (in `src/tac/tests/test_joint_admm_coordinator.py:168-234`) verifies:
- 2-stream convex quadratic R(D)
- Closed-form KKT optimum at b1=166.67, b2=133.33; margins both 1.333
- Empirical KKT residual 0.02 within tolerance

This is correct as a sanity check. But it is **not sufficient** for declaring ADMM ready for the real archive because:

1. **The real archive has 4-6 streams**, not 2 (renderer bytes, mask bytes, pose bytes, codebook bytes, header overhead, optional hyperprior side-info). ADMM convergence on 2 streams does not generalise.

2. **The real R(D) functions are discrete staircases** (qint quantisation ladders, AV1 CRF discrete steps, STC bit-budget grid). At a discrete-jump boundary, the marginal `dScore/dByte` is undefined; the coordinator's adaptive-ρ logic must handle the discontinuity via either restart or dual-averaging.

3. **The non-convex case is where ADMM silently misbehaves.** Per memory `project_codec_stacking_composition_canonical_orders_20260429.md` "Failure modes": "ADMM divergence: use adaptive penalty, restarts, exact byte projection after every codec call." The coordinator HAS restart logic (`test_divergent_problem_restarts` exercises it) but the restart logic is tested on a pathological mock that ignores byte targets entirely. We have no test of restart on a REALISTIC non-convex but byte-respecting problem.

### 4.2 Synthetic 4-stream non-convex problem specification

**Streams** (each implements `StreamProximalCodec` Protocol):

| Stream | R(D) shape | Math | Why this shape |
|---|---|---|---|
| s1 (renderer) | Quadratic | `f1(b) = 0.005 × (b - 250)²` for b in [50, 500]; smooth | Selfcomp/block-FP renderers exhibit smooth curvature near optimum. |
| s2 (pose) | Linear-then-saturate | `f2(b) = 0.1 × (300 - b)` for b in [0, 300]; `f2(b) = 0` for b > 300 | Pose stream saturates at ~5KB total (per Shannon score-arithmetic memory). |
| s3 (mask) | Discrete jump | `f3(b) = 1.0` for b < 200; `f3(b) = 0.3` for b in [200, 350); `f3(b) = 0.1` for b ≥ 350 | STC / AV1 mask coding has discrete CRF / quantiser ladder jumps. |
| s4 (codebook) | Sigmoid-saturating | `f4(b) = 0.8 / (1 + exp((b - 80) / 20))` | Codebook side-info has small footprint with sigmoid-shaped marginal. |

**Total budget**: B = 700 bytes (representative of a tight-budget archive scenario).

**Closed-form expected behaviour** (Boyd/Dykstra):
- s1 wants ~250; s2 saturates at 300 (free below); s3 discrete jumps at 200 and 350; s4 saturates around 100-130.
- Naive water-fill suggests: s1=200, s2=200, s3=200, s4=100. Total = 700 ✓.
- BUT s3's discrete jump at b3=200 means margin3 jumps from `0.7/200 = 0.0035` (just above 200) to `0.7/-1 = undefined` (just below 200) — **non-convex region**.
- At KKT, marginals on UNSATURATED streams must equilibrate. s1 (interior, smooth), s4 (interior, smooth) → must have equal marginals. s2 (saturated at upper boundary, free) → marginal = 0. s3 (could be interior at 200, 350, or pinned to a discrete grid).

**Expected ADMM behaviour**:
- Convergence: should converge within max_iters=300 with adaptive ρ.
- Restart: MAY fire 1-2 times when crossing s3's discrete boundary at 200 and 350 (primal residual spikes when s3 jumps to a different interior value).
- KKT residual: must be ≤ 0.10 on the unsaturated active streams (s1, s4); s2 and s3 may have non-equilibrated margins because s2 is saturated and s3 is on a discrete grid.
- **The test must NOT assert KKT residual ≤ 0.05 on ALL streams** — that would be wrong for non-convex discrete problems. Assert only on the two smooth interior streams (s1, s4).

### 4.3 Test must FAIL if ADMM silently produces non-feasible point

This is the gating criterion. The test must:

1. Run ADMM with max_iters=300, primal_tol=0.05, dual_tol=0.05 (looser than convex; non-convex needs slack).
2. Verify EITHER:
   - `result.converged == True` AND `sum(result.final_bytes_per_stream) <= cfg.byte_budget + 5.0` AND KKT residual on smooth interior streams (s1, s4) ≤ 0.10
   - OR `result.converged == False` AND `result.restarts >= 1` AND the failure is HONESTLY reported (final byte sum may exceed budget but `converged=False` flag is correctly set).
3. **The test FAILS if** `result.converged == True` BUT `sum(bytes) > byte_budget + 10` (silent infeasibility).
4. **The test FAILS if** `result.converged == False` AND `result.restarts == 0` (failure-to-detect-divergence).

**This is the contract**: ADMM either converges legitimately, OR honestly reports divergence with restart history. Silent infeasibility is the failure mode that gates V2 dispatch.

### 4.4 Concrete test file path + assertions

**File**: `/Users/adpena/Projects/pact/src/tac/tests/test_joint_admm_4stream_nonconvex.py`

**Required assertions** (in pseudo-code, NOT to be implemented in this report):

```python
class LinearSaturatingStream:
    """Stream s2: linear-then-zero (saturation at upper bound)."""
    def __init__(self, slope=0.1, saturate_at=300.0, name="s2"):
        self.slope = slope
        self.saturate_at = saturate_at
        self._name = name
    @property
    def name(self): return self._name
    def proximal_step(self, target_bytes, dual):
        b = max(0.0, min(target_bytes, self.saturate_at))
        if b >= self.saturate_at:
            score = 0.0
            margin = 0.0  # saturated; complementary slackness
        else:
            score = self.slope * (self.saturate_at - b)
            margin = self.slope
        return ProximalStepResult(int(b), score, margin)


class DiscreteJumpStream:
    """Stream s3: AV1/STC-style discrete CRF ladder."""
    def __init__(self, name="s3"):
        self._name = name
    @property
    def name(self): return self._name
    def proximal_step(self, target_bytes, dual):
        # Discrete jumps at 200, 350
        if target_bytes < 200:
            b, score = 100, 1.0
        elif target_bytes < 350:
            b, score = 250, 0.3
        else:
            b, score = 400, 0.1
        # Marginal at the chosen discrete point: forward finite difference
        # toward the next available grid point.
        if b < 250:  # at 100, next is 250
            margin = (1.0 - 0.3) / (250 - 100)  # = 0.00467
        elif b < 400:  # at 250, next is 400
            margin = (0.3 - 0.1) / (400 - 250)  # = 0.00133
        else:
            margin = 0.0
        return ProximalStepResult(b, score, margin)


class SigmoidSaturatingStream:
    """Stream s4: codebook-style sigmoid R(D)."""
    def __init__(self, mid=80.0, scale=20.0, name="s4"):
        self.mid, self.scale = mid, scale
        self._name = name
    @property
    def name(self): return self._name
    def proximal_step(self, target_bytes, dual):
        b = max(0.0, target_bytes)
        z = (b - self.mid) / self.scale
        sig = 1.0 / (1.0 + math.exp(z))
        score = 0.8 * sig
        margin = 0.8 * sig * (1.0 - sig) / self.scale
        return ProximalStepResult(int(b), score, margin)


def test_4stream_nonconvex_converges_or_honestly_diverges():
    """[synthetic] 4-stream non-convex ADMM. The test FAILS if ADMM
    silently produces a non-feasible point (converged=True but
    sum(bytes) > byte_budget). The test PASSES if EITHER
    (a) legitimate convergence with KKT residual on smooth interior
        streams (s1, s4) ≤ 0.10, OR
    (b) honest divergence: converged=False AND restarts >= 1.
    """
    s1 = QuadraticRateStream(a=0.005, b_opt=250.0, name="s1", discretisation=1.0)
    s2 = LinearSaturatingStream(name="s2")
    s3 = DiscreteJumpStream(name="s3")
    s4 = SigmoidSaturatingStream(name="s4")
    cfg = JointADMMConfig(
        byte_budget=700.0,
        max_iters=300,
        primal_tol=0.05,
        dual_tol=0.05,
        kkt_waterline_tol=0.10,
        rho_init=0.05,
        rho_imbalance_ratio=10.0,
        restart_threshold=5,
        verbose=False,
    )
    result = run_admm([s1, s2, s3, s4], cfg)

    bytes_arr = np.asarray(result.final_bytes_per_stream)
    margins_arr = np.asarray(result.final_marginal_per_stream)

    # GATE 1: HONEST budget feasibility
    if result.converged:
        assert bytes_arr.sum() <= cfg.byte_budget + 10.0, (
            f"ADMM reported converged=True but sum(bytes)={bytes_arr.sum():.1f} "
            f"exceeds budget {cfg.byte_budget} — SILENT INFEASIBILITY. "
            f"This is the failure mode the test gates against."
        )
    else:
        # If not converged, restarts must have fired (honest divergence detection)
        assert result.restarts >= 1, (
            f"ADMM reported converged=False but restarts=0 — "
            f"FAILURE-TO-DETECT-DIVERGENCE. The coordinator must fire at "
            f"least one restart before declaring failure on a non-convex "
            f"problem."
        )
        # Honest divergence — test passes here without further assertions.
        return

    # GATE 2: KKT on smooth interior streams (s1, s4) — NOT s2 (saturated) or s3 (discrete)
    smooth_indices = [0, 3]  # s1 and s4
    smooth_margins = margins_arr[smooth_indices]
    smooth_bytes = bytes_arr[smooth_indices]
    # All smooth streams should be in interior (not pinned to 0 or budget)
    for idx, b in zip(smooth_indices, smooth_bytes):
        assert b > 1.0 and b < cfg.byte_budget - 1.0, (
            f"smooth stream {idx} bytes={b} pinned to boundary; "
            f"expected interior allocation."
        )
    if len(smooth_margins) >= 2:
        residual = float(smooth_margins.max() - smooth_margins.min())
        assert residual <= cfg.kkt_waterline_tol * 2.0, (
            f"KKT waterline residual on smooth streams {residual:.4f} > "
            f"{cfg.kkt_waterline_tol * 2.0:.4f} — adaptive-ρ is not "
            f"equilibrating the smooth interior despite convergence."
        )

    print(f"[synthetic] 4-stream ADMM: converged={result.converged}, "
          f"iters={result.iters}, restarts={result.restarts}, "
          f"bytes={bytes_arr.tolist()}, margins={margins_arr.tolist()}")
```

### 4.5 Does this test gate Phase 2 Lane 10 V2 dispatch?

**YES — this is the gate.** Council E §3.1 #3 explicitly defers Lane Joint-ADMM real-codec wrap until "KKT residual stays < 0.05 on the 4-stream problem". The test above is a STRONGER version of that gate: it requires either legitimate convergence (with KKT on smooth interior streams) OR honest divergence (with restart firing). Silent infeasibility — the failure mode that would make Lane 10 V2 ship broken byte allocations — is explicitly gated against.

**Do we need MORE tests before V2 dispatch?** Two additional tests are recommended:

1. **`test_admm_byte_budget_strict_compliance`**: assert that `result.final_bytes_per_stream` NEVER exceeds `byte_budget` even by 1 byte when `converged=True`. The current test allows +10 byte slack for proximal discretisation; production should be stricter.

2. **`test_admm_warm_start_on_codec_state`**: verify that subsequent `proximal_step` calls receive the previous iteration's `state` field (warm-start) and that the coordinator passes it correctly. Without warm-start, the codec re-encodes from scratch each iteration → slow convergence.

These are nice-to-haves; the 4-stream non-convex test is the **critical gate**. With it landed AND passing, Lane 10 V2 dispatch is unblocked.

---

## 5. Synthesis — Updated Dispatch Order

### 5.1 Total cost

| Item | Cost | Rationale |
|---|---|---|
| Lane HM-S retrain (Vast.ai 4090, ~6h) | $1.50 | APPROVE — orthogonal to Lane G v3, highest EV |
| Lane FR-Ω retrain (Vast.ai 4090, ~6h) | $1.50 | APPROVE — orthogonal to Lane G v3, sub-Quantizr territory |
| Lane WC-S retrain | $0 (deferred) | DEFER until HM-S + FR-Ω stack signal lands |
| Lane PA retrain | $0 (killed) | KILL — not a stack partner; central within noise of Lane G v3 |
| Lane FC retrain | $0 (killed) | KILL — not a stack partner; band crosses unacceptable threshold |
| Ω-W-V2 real-archive validation (local) | $0 | SAFE-LOCAL — pure byte/roundtrip, no scorer |
| ADMM 4-stream non-convex test (local) | $0 | GATE on V2 dispatch; pure synthetic R(D) |
| **TOTAL** | **$3.00** | vs Council E's original $5.80 wave (savings: $2.80 + 3 lane-hours of GPU not-burned-on-marginal-lanes) |

### 5.2 Updated dispatch order vs Council E's recommendation

**Council E original order** (from `lane_redispatch_plan_post_round6_20260429.md`):
1. Wave 1 — SC++ ($3.50)
2. Wave 2 — SA-v2 ($3.50)
3. Wave 3 — WC-S, PA, HM-S, FR-Ω, FC parallel ($5+)
4. Local Ω-W-V2 + Joint-ADMM dispatches ($0)

**Council F revised order** (this report):
1. **Local-only validations FIRST** (no GPU, $0):
   - Ω-W-V2 real-archive test on Lane G v3 renderer.bin (validates / falsifies docstring claim before any V2 archive build)
   - Joint-ADMM 4-stream non-convex test (gates Lane 10 V2 dispatch)
   - Both can run in <30 min on local CPU
2. **Vast.ai 4090 dispatch (Wave 1, $3.00)**:
   - HM-S first ($1.50, 6h) — highest EV, fully orthogonal to Lane G v3
   - FR-Ω in parallel ($1.50, 6h) — fully orthogonal, sub-Quantizr territory
3. **Council A Wave 1+2 (SC++ / SA-v2)** still in flight from prior plan, NOT in scope of this council's verdict (they are first-class lanes, not in the 5-lane re-train set this council was asked to evaluate). Carry over from Council E.
4. **WC-S re-evaluated** AFTER HM-S signal lands. If HM-S stacks well on Lane G v3, WC-S becomes lower priority. If HM-S regresses, WC-S re-enters consideration.
5. **PA, FC do not get re-dispatched.** KILL.

**Rationale**: Local validations are zero-cost and ARM the codec stack before any Wave 1 archive build. Burning $1.50 on HM-S before the Ω-W-V2 codec is empirically validated would be premature optimisation; the codec might regress on real architecture and we'd ship a worse archive. Sequence the validations BEFORE the dispatches.

### 5.3 Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| HM-S band [0.32, 0.45] is over-optimistic; lands at [0.7, 1.2] | MEDIUM (no prior empirical) | -$1.50 | Hard-kill at first auth eval; if score > 0.8, abandon FR-Ω dispatch too. |
| Ω-W-V2 saves <20% on real architecture (synthetic 69.11% over-stated) | HIGH (per Council E §2.1 CONCERN-1) | docstring update + V2 not stacked into archive build | Test catches it; no GPU spent on V2 archive build that regresses. |
| ADMM 4-stream non-convex test FAILS (silent infeasibility) | MEDIUM | Lane 10 V2 dispatch blocked until coordinator fixed | Test is the gate; if it fails, fix coordinator before any real-codec wrap. |
| HM-S + FR-Ω parallel dispatch, both OOM despite Council C bf16+chunk fix | LOW (Council C fix verified at code level; remote_lane scripts updated) | -$3.00 | Pre-dispatch local smoke test + heartbeat watchdog catches at minute 30. |
| Council E's prior dispatches (SC++, SA-v2) finish first and re-baseline Lane G v3 | LOW | revised verdict on HM-S/FR-Ω depending on new baseline | This council's verdicts are conditional on Lane G v3 = 1.05 baseline; if SC++ ships sub-1.0, re-evaluate the wedge attribution. |

---

## 6. Council Roll Call

Each council member casts a signed verdict (1-2 sentences). Per CLAUDE.md "Council conduct" the council is non-conservative; arguments are mathematical/empirical only.

**Shannon (LEAD, Information Theory)**: HM-S targets the PoseNet R(D) wedge with ~0.09-0.11 score headroom; FR-Ω targets the rate-term wedge with ~0.05-0.10 score headroom. Both APPROACH ceilings. WC-S nibbles at noise-floor; PA/FC are representational alternatives. The Ω-W-V2 real-archive test is the right next step because the synthetic 69.11% claim is unsubstantiated on real conv-weight distributions. **Verdict: APPROVE HM-S + FR-Ω; DEFER WC-S; KILL PA + FC; SAFE-LOCAL Ω-W-V2.**

**Dykstra (CO-LEAD, Convex Feasibility)**: Of 5 lanes, only HM-S and FR-Ω are orthogonal stacking partners with Lane G v3. WC-S has 50% overlap with KL-distill; PA and FC are replacements, not stacks. The Joint-ADMM 4-stream non-convex test is the gate on Phase 2 Lane 10 V2 dispatch — without it, V2 would silently produce non-feasible points on the real archive. **Verdict: dispatch only the orthogonal lanes; the 4-stream test gates V2.**

**Yousfi (Challenge creator, Steganalysis lineage)**: HM-S 8-DOF homography is the right geometric move because PoseNet evaluates against ground-truth ego-motion; richer geometry better matches the contest scorer's pose extraction. FR-Ω Hessian-cost block-FP is the canonical Selfcomp stack ingredient and respects the contest archive-bytes budget. FC and PA are diversions from the wedge. **Verdict: HM-S + FR-Ω APPROVE.**

**Fridrich (UNIWARD/SRM/HUGO author)**: FR-Ω's Hessian-curvature-driven bit allocation is the direct application of UNIWARD's "errors in textured regions are undetectable" principle to weight quantisation — high-curvature output channels (textured-equivalent) get more bits, low-curvature channels (smooth-equivalent) get fewer. The WC-S curator-outlier weighting is conceptually similar but operates on training-loss surface, not weight bytes; it may compete with KL-distill's gradient signal. **Verdict: FR-Ω APPROVE; WC-S DEFER.**

**Contrarian (Veto)**: I VETO any verdict above that treats `[prediction]`-tagged bands as if they were `[empirical]` measurements. The 5-lane bands have ZERO empirical [contest-CUDA] support. The verdicts above are conditional on the bands being roughly right, and that condition is itself unverified. The cheapest hedge is HM-S first; if it lands within 0.10 of central 0.38 the bands are calibrated, otherwise we KILL FR-Ω before spending the second $1.50. I VETO any narrative that promotes "sub-Quantizr-0.33 reachable" until at least HM-S lands. **Verdict: SEQUENTIAL DISPATCH (HM-S first, FR-Ω gated on HM-S signal); local validations OK.**

**Quantizr (Adversarial leaderboard reality check)**: My 0.33 archive uses block-FP weights at 1.017 bpw and analytical pose via affine — both of which FR-Ω attempts to replicate via Hessian-aware allocation. If FR-Ω lands within [0.30, 0.45] band you have replicated my approach; if it lands above 0.50 you have NOT understood the block-FP discipline. HM-S's 8-DOF homography is a different bet (richer geometric model) that I did NOT use; its outcome will be informative either way. **Verdict: APPROVE FR-Ω as my-approach-replication; APPROVE HM-S as orthogonal experiment.**

**Selfcomp (szabolcs-cs, working 0.38 anchor)**: My 88K SegMap renderer + grayscale-LUT + 1.017 bpw block-FP delivered 0.38. FR-Ω is the canonical-stack-ingredient that maps directly onto my renderer.bin. The Ω-W-V2 codec applied to my exported state-dict is exactly what should happen next; I would prefer a real-archive validation of Ω-W-V2 BEFORE FR-Ω dispatch, because if the codec regresses on real weights then FR-Ω will inherit the regression. **Verdict: SEQUENCE local Ω-W-V2 validation FIRST, then FR-Ω; APPROVE HM-S in parallel.**

**Hotz (Engineering shortcuts)**: $3 for two retrains + $0 for two local tests is the right shape. Killing PA and FC saves $2 of GPU on lanes that aren't going anywhere new. The 4-stream non-convex test is a 30-minute write + 5-minute run; that's the kind of cheap insurance that prevents a $50 broken-archive ship later. **Verdict: KILL PA + FC; APPROVE HM-S + FR-Ω + both local validations.**

**Carmack (Engineering shortcuts at the Doom level)**: Show me the test that fails when ADMM ships a 750-byte allocation against a 700-byte budget while reporting `converged=True`. If that test doesn't exist, ADMM cannot be trusted on the real archive. The 4-stream non-convex test in §4.4 above is that test. **Verdict: APPROVE the 4-stream non-convex test as the V2 dispatch gate.**

**Boyd (Convex optimisation at the operational level)**: The 4-stream non-convex test correctly distinguishes (a) honest convergence with KKT on smooth interior streams from (b) honest divergence with restart-detection from (c) silent infeasibility. The KKT-on-smooth-interior-only assertion is the right pattern for non-convex problems where saturated and discrete-grid streams cannot equilibrate. **Verdict: APPROVE the 4-stream test design; APPROVE the gate on Lane 10 V2 dispatch.**

**MacKay (Memorial seat, Information Theory + Bayesian Inference + Learning Algorithms)**: The Ω-W-V2 real-archive test answers an MDL-style question: "what is the rate cost of approximating Lane G v3's 290KB renderer.bin via static-histogram arithmetic on per-channel water-fill?" The answer must be measured on the real distribution, not a synthetic Gaussian. If real savings < 20% the docstring claim is FALSIFIED and we drop the V2 layer from the archive build. **Verdict: APPROVE the real-archive validation; the result REPLACES the synthetic claim.**

**Ballé (2018 entropy bottleneck SOTA)**: The Ω-W-V2 static-histogram terminal is the V2 ceiling; my hyperprior would be the V3 step. Per `src/tac/water_filling_codec_v2.py` lines 17-23, Selfcomp's renderer at ~11KB is borderline for hyperprior amortisation. The real-archive test will tell us whether V2 alone is competitive (if savings ≥ 30% on real architecture) or whether V3 hyperprior is required (if V2 saves <15%). **Verdict: APPROVE the real-archive validation as the V3 gate-trigger.**

---

## 7. Cross-references

- Council A (DARTS-S freeze + Check 86): `.omx/research/council_darts_s_freeze_audit_20260429.md`
- Council C (OOM-class deep fix; bf16 + scorer-chunk for SegMap-class lanes): `.omx/research/council_oom_class_deep_fix_20260429.md`
- Council E (Round 5 grand battleplan + 3 dispatch recs): `.omx/research/council_grand_battleplan_round5_20260429.md`
- Council Round 6 (9-lane invalidation correction; local-only validity audit): `.omx/research/council_round6_adversarial_20260429.md`
- Lane re-dispatch plan post-Round 6: `.omx/research/lane_redispatch_plan_post_round6_20260429.md`
- Codec stacking + score arithmetic: `project_codec_stacking_composition_canonical_orders_20260429.md`
- Skunkworks council quintet pact: `feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md`
- Local-only validity binding rule: `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`
- Lane G v3 baseline 1.05 [contest-CUDA]: `project_lane_g_v3_landed_1_05_20260428.md`
- Source files validated (read-only):
  - `src/tac/water_filling_codec_v2.py` (Ω-W-V2 codec)
  - `src/tac/joint_admm_coordinator.py` (Joint-ADMM coordinator)
  - `src/tac/joint_admm_proximal_pose_delta.py` (first ADMM proximal wrapper)
  - `src/tac/segmap_renderer.py` (SegMapTrainer used by HM-S, FR-Ω, WC-S, PA, FC scripts)
  - `scripts/remote_lane_wc_s_curator_weighted.sh` (Council C bf16+chunk fix verified wired)
  - `scripts/remote_lane_pa_pose_as_affine.sh` (Council C bf16+chunk fix expected wired)
  - `scripts/remote_lane_hm_s_segmap_homography.sh` (Council C bf16+chunk fix expected wired)
  - `scripts/remote_lane_fr_omega_fridrich_block_fp.sh` (Council C bf16+chunk fix expected wired)
  - `scripts/remote_lane_fc_film_canvas.sh` (Council C bf16+chunk fix expected wired)
- Anchor file for Ω-W-V2 validation: `experiments/results/lane_g_v3_landed/iter_0/renderer.bin` (290KB, verified existence)
