# In-curriculum levers design — floor-chasing co-design with the base_ch=20 HNeRV substrate (2026-06-12)

**Author:** in-curriculum-lever DESIGN subagent (Layer 2 of the operator's three-layer stack: substrate / in-curriculum levers / post-hoc bolt-ons).
**Status:** DESIGN/SPEC ONLY. No GPU runs launched (Phase 1 base_ch=20 basin is live; a GPU run here would contend). Every lever is **gated on Phase 1 confirming a frontier** before Phase-2 deployment.
**Frontier (pointer, NOT hardcoded):** `.omx/state/canonical_frontier_pointer.json` → contest-CPU `0.19109982`, archive `177169 B`, lane `pr110_payload_entropy_recode`. **Frontier UNMOVED.** Target `T_floor = 0.11797` (rate-dominated, per the GOAL section; this memo treats it as a PREDICTION/derivation anchor, not a measured result).
**NO FAKE:** every quantified effect below is a **PREDICTION** with its first-principles basis named. A designed lever is a design, not a measured row. Nothing here asserts a score; deploy-gating + the dual CPU/CUDA exact-eval gate stand.

> Per CLAUDE.md "score-domain Lagrangian not weight-domain proxies" (HNeRV parity L6) + `eval_roundtrip` + EMA non-negotiables. The torch_vehicle loss loop ALREADY honors eval_roundtrip (bicubic↑874 → bilinear↓384 → uint8-STE round, `driver.py:389-395`) and EMA-after-every-step (`driver.py:445`); levers must NOT break those.

---

## 0. The decomposition that ranks every lever (the binding-constraint map)

The contest score is `S = 100·d_seg + sqrt(10·d_pose) + 25·|archive.zip|/37_545_489`.

At the **current frontier** (`0.19109982`, `177169 B`):
- **rate term** `= 25 · 177169 / 37_545_489 = 0.11796` — this is **61.7 %** of S and is **numerically equal to T_floor** (`0.11797`). That is not a coincidence: **T_floor is rate-dominated**; the measured floor is essentially "rate term at the best byte count we have, with d_seg and d_pose driven to their architectural floors."
- residual `S − rate = 0.19110 − 0.11796 = 0.07314` is the **distortion budget** still on the table: it splits into `100·d_seg` and `sqrt(10·d_pose)`.

**Consequence for ranking (the means/ends firewall applied to levers):**
1. **The rate term is the dominant lever AND the hardest floor.** Closing `0.19110 → 0.11797` requires the **distortion residual `0.07314` → ~0** *without paying it back in bytes*. So the highest-EV levers are the ones that **drive d_seg and d_pose to zero at constant-or-lower byte cost** — i.e. score-domain training (so we stop wasting bytes on reconstruction the scorer ignores) + a differentiable rate term (so the optimizer can trade reconstruction fidelity the scorer ignores for fewer bytes).
2. **Pose is structurally collapsible at near-zero byte (the Quantizr lever).** `sqrt(10·d_pose)` is the nonlinear term; at the frontier d_pose is already small but the **marginal value of pose near zero is high** (CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent": below `pose_avg ≈ 2.5e-4` the pose marginal *exceeds* SegNet's, derivative `5/sqrt(10·pose_avg) → ∞`). Pose-FiLM stores ~6 scalars/pair and **removes pose from the learning problem entirely**, freeing the whole decoder capacity for d_seg + rate.
3. **d_seg is argmax-flip, not L2.** The contest d_seg is the per-pixel **SegNet argmax-disagreement rate** (`upstream/modules.py`; `score_pair_components` confirms it). Training the decoder against an L2/reconstruction proxy spends bytes on pixels the scorer's argmax ignores. A **differentiable d_seg surrogate** + a **score-aware QAT** that protects the argmax boundary (not the L2 reconstruction) is how d_seg → its floor at lower byte.

So the levers are NOT independent knobs; they are a **co-designed system** that re-routes the entire byte/distortion budget toward the three score terms. The ranking in §6 follows directly from this map.

### What the substrate ALREADY has (do not reinvent — SEARCH-FIRST result)

The torch_vehicle PR95 8-stage curriculum (`src/tac/torch_vehicle/{driver,curriculum}.py`, vendored from PR95 — `build_curriculum` reads the live `StageConfig`) already carries:
- **C1a coder-aware entropy reg** = `cat_entropy_v2` (`src/tac/losses/cat_entropy_v2.py`), wired at `driver.py:409-415,424-429`. It penalizes the **per-weight Shannon entropy of the INT8-quantized weight histogram** (soft-histogram over `{-127..127}`, σ-annealed). **This is a per-tensor weight-entropy proxy, NOT the actual brotli/LZMA byte cost** — that is the gap lever #1 attacks.
- **σ noise schedule** (`cat_sigma`) + **QAT** (`spec.use_qat` → `self.v.apply_qat`/`restore_qat`, `driver.py:382-386`) — but QAT here is the **vendored fake-quant**, NOT score-aware.
- **Muon final-stage** + per-stage `seg_loss_fn` (the seg surrogate, swappable per stage) + EMA + eval_roundtrip.
- **split-by-head backward** (`driver.py:459-513`) — already routes SegNet grad on train-device (MPS, validated bit-identical on d_seg) and PoseNet grad on the **CPU authority** (zero MPS pose drift). This is the gradient-reachable score path the levers plug into.

The actual archive codec (the byte cost we must surrogate): vendored `codec.py` — **decoder = per-tensor symmetric INT8 → zigzag → brotli q=11** (one blob); **latents = per-dim minmax→uint8 → 1st-order temporal delta → zigzag uint16 → lo/hi byte-split → brotli**. `build_archive`/`parse_archive` = `self.v.*`, byte count = `len(archive)` (`driver.py:612-622`).

---

## Lever 1 — Differentiable rate/entropy term IN the loss (the brotli-byte surrogate)

**Attacks:** rate (61.7 % of S, the dominant term). **EV: HIGHEST.**

### Math
C1a's `cat_entropy_v2` computes `H_w = Σ_t numel(t)·H_t / Σ_t numel(t)` — the **per-weight Shannon entropy** of the soft-INT8-histogram. That is the *Shannon lower bound on a memoryless arithmetic coder of the quantized weights*. But the archive codec is **brotli q=11 over the zigzag-INT8 byte stream**, which exploits **inter-symbol context (LZ matches + order-N modeling)** that the memoryless `H_w` ignores. The gap between `H_w·numel/8` and the real brotli byte count is exactly the redundancy brotli captures and `cat_entropy_v2` cannot see. Empirically (vendored codec note) the decoder blob dominates the 177169 B archive.

Two differentiable surrogates, composed:

**(1a) Conditional (order-1) weight entropy.** Replace the marginal `H(W)` with the **conditional entropy along the brotli scan order** `H(W_i | W_{i-1})` using a soft 2-D joint histogram of adjacent zigzag symbols:
```
sa_i[b]  = softmax_b( -((w_n[i]-b)/σ)² / 2 )           # soft INT8 bin assignment (as now)
J[a,b]   = Σ_i sa_{i-1}[a] · sa_i[b] / (N-1)           # soft joint of (prev,cur) along scan order
H(cur|prev) = Σ_a J[a,·]·( -Σ_b (J[a,b]/J[a,·]) log2 (J[a,b]/J[a,·]) )
```
This is a differentiable **first-order Markov entropy** — a far tighter brotli proxy than the memoryless `H_w`, because brotli's order-N context modeling is bounded below by the order-1 conditional entropy and approaches it on smooth INT8 streams. Penalizing `H(cur|prev)` directly biases the decoder toward **scan-order-smooth INT8 weights** (long zigzag runs → long brotli LZ matches).

**(1b) Latent rate term (currently UNPENALIZED).** The latents path (per-dim delta + zigzag + lo/hi split) has **no training-time rate term at all** — only the decoder weights see C1a. Add a differentiable **temporal-delta entropy** on the latents `z ∈ (n_pairs, latent_dim)`:
```
dz[p] = z[p] - z[p-1]                                   # the quantity the codec actually delta-codes
R_lat = Σ_dim  H_soft( quantize_to_uint8(dz[:,dim]) )   # soft-hist entropy of the delta stream, per dim
```
This rewards **temporally-smooth latents** (small deltas → mostly-zero hi-byte stream → tiny brotli). Dashcam video is temporally redundant (per L25), so this is a real, currently-unexploited byte lever.

### Exact wire-in point
`driver.py:_train_one_epoch`, alongside the existing `cat_entropy_v2` block (lines 409-415 split path / 424-429 non-split). The conditional-entropy variant **replaces or augments** the `ent = self.v.cat_entropy_v2(...)` call with a new `tac.losses.rate_surrogate.brotli_rate_surrogate(decoder, latents[idx-context], cfg)` returning `(H_cond_weights, R_lat)`; add `loss += spec.rate_lambda_w · H_cond + spec.rate_lambda_lat · R_lat`. The latent term needs the *temporal neighbor* of each sampled pair — pass the full `latents` tensor (already in scope as `rt.latents`) rather than the batch slice, and compute `R_lat` once per epoch (it is global, like C1a which reads all weights). New `StageSpec` fields: `rate_lambda_w`, `rate_lambda_lat` (default 0.0 → byte-identical to today; opt-in per stage, schedule them up in stages 5-8 like C1a's `cat_lambda` 0.01→0.02).

### Predicted effect + basis
**PREDICTION (basis: information theory, Shannon order-1 ≤ order-N ≤ marginal bound).** The marginal `H_w` over-estimates the brotli cost (brotli captures the redundancy `H_w − H(cur|prev)`); training against the *conditional* entropy lets the optimizer find weight configurations whose brotli cost is lower at equal distortion. Order-of-magnitude: the decoder blob is the majority of 177169 B; a 5-15 % decoder-blob reduction at constant distortion is `ΔS ≈ −25·(0.07·177169)/37.5e6 ≈ −0.0083` (rate axis). The latent term (1b) is additive and currently zero-exploited; **PREDICTED** small but free (the latents are a few KB; even 30 % off is `ΔS ≈ −0.001`). These are derivations, not measurements; the falsification gate is a paired A/B (lever-on vs lever-off, same epoch budget) measuring `archive_bytes` at equal `d_seg/d_pose`.

### Risk
- **Over-regularization → distortion payback.** Pushing weight conditional-entropy too hard flattens the decoder → d_seg/d_pose rise → net `ΔS > 0`. Mitigation: schedule `rate_lambda` up only in late stages (5-8), and the score-aware loss (Lever 2) provides the counter-pressure — the Lagrangian arbitrates.
- **σ-bin / brotli mismatch.** The soft-histogram bins must match the codec's INT8 grid (they do — `cat_entropy_v2` already uses `{-127..127}`; reuse `CatEntropyV2Config`).
- **Scan-order assumption.** brotli's LZ window is not strictly the state_dict iteration order; the order-1 proxy is a *lower bound regardless of order*, so it is conservative (never claims more savings than achievable). HARD-EARNED, not cargo-culted: the proxy is a true bound.

### Deploy plan (gated)
Land `tac/losses/rate_surrogate.py` (new) + `StageSpec` fields default-0 (byte-identical). Phase-2 deploy ONLY after Phase 1 confirms a frontier: enable `rate_lambda_w` in stages 6-8 first (decoder dominates), A/B vs the live base_ch=20 best, accept only if `archive_bytes` drops at non-worse `d_seg/d_pose` (advisory local gate → paired CPU/CUDA exact-eval verdict). **Do not** assert a score from the proxy alone (Catalog #304 closed-form-rate-without-empirical-bit-spend-proof: the A/B IS the empirical bit-spend proof).

---

## Lever 2 — Full score-domain Lagrangian (stop training a reconstruction proxy)

**Attacks:** d_seg + d_pose + (indirectly) rate. **EV: HIGHEST (this is HNeRV parity L6 made primary).**

### Math
The canonical contest Lagrangian (`tac.substrates.score_aware_common`, `CONTEST_SEG_WEIGHT=100`, `CONTEST_POSE_SQRT_WEIGHT=sqrt(10)`):
```
L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ))
  = rate_lambda·R̂(θ)  +  100·d_seg_surrogate(θ)  +  sqrt(10·d_pose(θ) + ε)
```
The torch_vehicle loss is ALREADY `spec.seg_weight·seg_l + spec.pose_weight·pose_l` with `pose_l = sqrt(10·pose_mse)` (`driver.py:421-423`) — **so the pose term is already score-domain.** The gap is the **seg term**: `seg_l = spec.seg_loss_fn(seg_out, seg_targets_hard)` is the vendored per-stage seg loss (cross-entropy variants), NOT the differentiable **argmax-flip d_seg surrogate**. d_seg is `(argmax(out1) != argmax(out2)).mean()` — a 0/1 boundary quantity. CE optimizes log-likelihood of the *hard target everywhere*, spending capacity on confident-interior pixels the argmax already gets right.

Swap `seg_loss_fn` for the canonical differentiable d_seg surrogate (already built — `tac.losses.core.segnet_surrogate_per_pixel`, choices: `soft_cosine` (default), `fisher_rao`, `sinkhorn`). The score-optimal seg loss is **boundary-concentrated**: temperature-annealed soft-cosine `1 − Σ softmax(p/T)·softmax(q/T)` with `T: 1.0 → 0.05` over stages (sharpens toward hard argmax), OR Fisher-Rao (information-geometry distance, stronger gradients near agreement). Add α·R̂ from Lever 1. All three terms now live in score units; the loss IS S (up to the per-pair vs full-video aggregation).

### Exact wire-in point
`driver.py:_train_one_epoch` (both split and non-split paths) + `curriculum.py:StageSpec`. Two coordinated changes:
1. **seg term:** route `seg_l` through `tac.losses.core.segnet_surrogate_per_pixel(seg_out_pred, seg_out_gt, surrogate=spec.seg_surrogate, temperature=spec.seg_temperature)` instead of `spec.seg_loss_fn(seg_out, seg_targets_hard)`. NOTE: the surrogate needs **GT seg logits/probs** (two-frame argmax-disagreement), while the current path uses `seg_targets_hard` (precomputed hard argmax in `scorer_context.py`). Either (a) precompute GT seg *logits* in `scorer_context` (one extra cached tensor) for the soft surrogate, or (b) keep the hard target but use the boundary-weighted STE loss (`focal_segnet_ste_loss` / `boundary_aware_loss`, forward=hard-disagree, backward=boundary-weighted CE) — **(b) is the lower-risk first step** because it keeps the existing target cache and only changes the gradient shaping.
2. **rate term:** add Lever-1's `α·R̂`. New `StageSpec` fields: `seg_surrogate` (str, default keeps vendored `seg_loss_fn`), `seg_temperature` (float), `rate_lambda_*` (from Lever 1). Defaults preserve today's behavior exactly.

### Predicted effect + basis
**PREDICTION (basis: HNeRV parity L6 + the argmax-flip structure of d_seg + the 0.07314 distortion residual).** CE-on-hard-targets is a *surrogate of a surrogate*; the boundary-concentrated d_seg surrogate puts gradient where the argmax actually flips, so equal capacity buys lower d_seg → the `100·d_seg` slice of the 0.07314 residual shrinks. The council recommendation anchors (in `tac.losses.core` docstrings) estimate `−0.06 to −0.14` SegNet improvement for boundary/focal/fisher variants *on their measured substrates* — **those are other-substrate priors, not a base_ch=20 measurement**; for THIS substrate the effect is a PREDICTION pending the A/B. The pose term is already score-domain so no change there. Falsification gate: paired A/B at equal epoch budget, accept only if `d_seg` drops at non-worse `d_pose/rate`.

### Risk
- **GT-logits cache cost.** Option (a) doubles a cached tensor in `scorer_context`; option (b) avoids it. Start with (b).
- **Temperature annealing instability.** Too-fast `T→0.05` makes the loss near-discontinuous (vanishing gradient on the boundary). Anneal slowly, gate on the per-stage eval cadence already present.
- **Seg/pose coupling (L10).** Changing the seg loss can shift d_pose via shared decoder capacity; the Lagrangian's pose term keeps it honest, but record decoded-mask SHAs + pose regen per the mask-coupling gate before any byte claim.
- **NOT cargo-culted:** the surrogates are pre-built and tested in `tac.losses.core`; this is a *routing* change, the optimal-engineering choice per UNIQUE-AND-COMPLETE-PER-METHOD (the canonical helper SERVES here — adopt).

### Deploy plan (gated)
Add `StageSpec` fields default-preserving. Phase-2: enable option (b) boundary-weighted STE seg loss in stages 5-8 first (where C1a/QAT already concentrate), A/B vs base_ch=20 best, then layer option (a) soft-cosine + Lever-1 rate term if (b) confirms a frontier. Dual CPU/CUDA exact-eval before any score claim.

---

## Lever 3 — Pose-FiLM store (the Quantizr lever: d_pose → ~0 at ~1 KB)

**Attacks:** d_pose (the `sqrt(10·d_pose)` nonlinear term; high marginal value near zero). **EV: HIGH (architectural + curriculum).**

### Math
Instead of asking the decoder to **recover** pose from pixels (current design: PoseNet runs on the rendered frame, `pose_mse = ||pose_pred − pose_gt||²`, decoder must learn to produce frames whose PoseNet readout matches GT — a hard inverse problem, the binding d_pose constraint per the capstone audit), **store the 6 GT pose scalars per pair** and **FiLM-condition the decoder** on them. The decoder no longer learns pose; it is *told* the pose and modulates its features:
```
γ, β = PoseFiLM(pose6)          # small MLP: pose6 → (per-channel γ, β); identity at init
h'    = (1 + γ) · h + β          # feature-wise linear modulation at decoder layer(s)
```
The pose enters as **side information** (Wyner-Ziv: decoder has the pose, so it does not pay to encode it in the weights). d_pose collapses toward the **quantization floor of the stored pose** (uint8/fp16 of 6 scalars), not the decoder's learning floor.

**Byte cost:** `n_pairs · 6 · bytes_per_scalar`. At 600 pairs × 6 × 2 (fp16) = **7.2 KB raw**; with per-dim delta + brotli (pose is temporally smooth) → **~1-3 KB**. The FiLM MLP itself is ~6 K params (it ships in the decoder blob, ~3 KB @ FP4 / brotli). Net added bytes ≈ **3-7 KB**.

### Reuse (SEARCH-FIRST — do NOT reinvent)
A **complete, tested torch implementation exists**: `src/tac/residual_basis/cool_chic_carrier.py::_PoseFiLM` + `CoolChicPairCarrier` (stored_pose buffer `(n_pairs,6)`, `sin(fc1)→fc2(zero-init)→(γ,β)`, γ∈(0,2) via `1+tanh`, identity at init; tests in `test_cool_chic_pose_film.py` verify FiLM-identity-at-init, trained-FiLM-pose-dependence, numpy-inflate parity, byte-closed pose-dependence). The MLX capstone version is `src/tac/mlx_pr95_port/pose_film.py` (`PoseFiLMDecoderMLX`, `StoredPoseBundleMLX`, `stored_pose_bytes(quant_step=1e-3)` byte-cost helper) — **this is the task-#84 CAPSTONE ACCELERATOR**, MLX-only. For torch_vehicle, **port `_PoseFiLM` from cool_chic_carrier** (torch, CPU-authority-compatible).

### Exact wire-in point
The torch_vehicle decoder is the **vendored `HNeRVDecoder`** (`driver.py:286-290`, `latent_dim=28`, `base_channels=20`, `eval_size=(384,512)`). FiLM injection requires the decoder forward to accept pose:
1. **Wrap, don't fork the vendored decoder** (it is pristine-source, must stay byte-pristine). Add a thin `PoseFiLMHNeRVWrapper(nn.Module)` in `tac.torch_vehicle.pose_film` holding the vendored `HNeRVDecoder` + a `_PoseFiLM` (ported from cool_chic) + a `stored_pose` buffer `(n_pairs, 6)` set from `scorer.pose_targets`. Inject FiLM at the **stem** (after latent→linear projection, before the upsample cascade — the capstone injection point; earliest = maximal effect) OR per-block.
2. `driver._new_decoder` returns the wrapper when `cfg.pose_film_enabled`. `_train_one_epoch`'s `decoder(latents[idx])` becomes `decoder(latents[idx], stored_pose[idx])`.
3. **build_archive must serialize stored_pose** — extend the meta/codec to add a `pose` section (per-dim delta + brotli, mirroring the latent codec). This is the ONE place the vendored codec needs an additive section; declare the grammar BEFORE training (export-first, HNeRV parity L2). New `cfg.pose_film_enabled` (default False → byte-identical, no pose section).

### Predicted effect + basis
**PREDICTION (basis: Wyner-Ziv side-information + the Quantizr 0.33 archive empirical + the operating-point marginal-value flip).** Removing pose from the learning problem collapses d_pose to the stored-pose quant floor; the `sqrt(10·d_pose)` term shrinks toward `sqrt(10·d_pose_quant)`. Because the marginal value of pose near zero is high (CLAUDE.md operating-point section), even a small absolute d_pose reduction is a meaningful `ΔS`. **Net is a TRADE:** `−Δ(sqrt(10·d_pose))` vs `+Δrate` (the 3-7 KB pose section, `+25·5000/37.5e6 ≈ +0.0033`). The lever WINS only if the pose-term reduction exceeds the byte cost. PREDICTED net negative `ΔS` at the frontier operating point because (a) the byte cost is tiny and (b) it **frees the whole decoder capacity** previously spent learning pose, which then improves d_seg AND lets Lever-1 cut decoder bytes — a *compounding* effect with Levers 1+2. This compounding is the design rationale; the magnitude is a PREDICTION pending A/B.

### Risk
- **Net-positive ΔS if pose was already near-floor.** If base_ch=20 already drives d_pose to ~quant floor, the stored-pose bytes are pure cost. Mitigation: measure base_ch=20 d_pose first (Phase 1 telemetry); deploy Lever 3 ONLY if d_pose > the stored-pose quant floor (the disambiguator).
- **Codec section = new archive grammar.** Additive section must round-trip through `parse_archive` + numpy-portable inflate (HNeRV parity L4, inflate ≤ 100 LOC). The cool_chic tests already prove byte-closed pose-dependence; port that test.
- **Vendored-decoder pristineness.** Wrap, never edit the intake clone (Forbidden in-place edits to public PR intake clones). The wrapper is ours.
- **FiLM identity-at-init is mandatory** (else it perturbs the live basin). cool_chic's `fc2` zero-init + `γ=1+tanh(0)=1` gives exact identity — preserve it.

### Deploy plan (gated)
Land `tac.torch_vehicle.pose_film` (wrapper + ported `_PoseFiLM`) + the additive `pose` codec section + the byte-closed parity test, all default-OFF (byte-identical). Phase-2: enable ONLY if Phase-1 d_pose telemetry shows headroom above the stored-pose quant floor; A/B vs base_ch=20 best measuring the `sqrt(10·d_pose)` reduction NET of the pose-section bytes; dual CPU/CUDA exact-eval before any score claim.

---

## Lever 4 — Score-aware QAT (quantize where the argmax-boundary tolerates, protect where it flips)

**Attacks:** rate (the INT8 grid) WITHOUT paying d_seg/d_pose. **EV: MEDIUM-HIGH (sharpens Lever 1's byte win without distortion payback).**

### Math
The vendored QAT (`losses.py:apply_qat`) is **uniform symmetric INT8 fake-quant over every Conv2d/Linear weight** (`fake_quantize(w)` in a forward, restore after — `driver.py:382-386`). It is **L2-reconstruction-blind to which weights matter for d_seg/d_pose.** Quantization error in a weight that drives a SegNet argmax-boundary pixel costs d_seg; the same error in a weight the argmax ignores is free. Score-aware QAT makes the **per-tensor (or per-channel) quantization step a function of score sensitivity**, not uniform.

Two coupled mechanisms:
**(4a) Sensitivity-weighted fake-quant.** Compute per-tensor **score sensitivity** `s_t = ||∂S/∂w_t||` (the gradient of the *score-domain* loss w.r.t. each weight tensor — already available once Lever 2 routes the score-domain loss). Allocate **finer INT8 effective resolution to high-sensitivity tensors** (smaller relative step, more brotli bytes there) and **coarser to low-sensitivity tensors** (larger step → more zeros → fewer brotli bytes). This is the water-filling of the bit budget against score-sensitivity — the bit-allocator hook (CLAUDE.md 6-hook #3). The total bytes can DROP at equal d_seg because we stop spending grid resolution on score-irrelevant weights.
**(4b) Boundary-protective straight-through QAT.** Run the QAT fake-quant forward through the **d_seg surrogate** (Lever 2's boundary-weighted seg loss), so the QAT-induced argmax flips are *directly penalized in training*. The decoder learns weights that are robust to their OWN INT8 quantization at the boundary — the quantize-then-decompress roundtrip is simulated in the proxy (eval_roundtrip sister discipline, applied to the weight grid not the pixel grid).

### Exact wire-in point
`driver.py:_train_one_epoch`, the existing `if spec.use_qat:` block (lines 382-386). Today it calls the uniform `self.v.apply_qat(decoder)`. Replace with `tac.torch_vehicle.score_aware_qat.apply_score_aware_qat(decoder, sensitivity=rt.tensor_sensitivity, cfg)`:
- (4a) needs per-tensor sensitivity `s_t` — accumulate `||w_t.grad||` over the epoch (it is already computed; the score-domain loss backward populates `.grad`). Maintain an EMA of `s_t` in `_StageRuntime`. Use it to set per-tensor `n_quant` (the INT8 level count) or step.
- (4b) is automatic once the QAT forward flows through Lever-2's seg surrogate (it already does — QAT wraps the same `decoder(latents[idx])` forward whose output feeds the loss). The only change is ensuring the QAT block is active in the **score-aware** stages, not just the vendored QAT stage.
New `StageSpec` fields: `score_aware_qat` (bool, default False → uniform vendored QAT), `qat_sensitivity_ema` (float).

### Predicted effect + basis
**PREDICTION (basis: rate-distortion bit-allocation / water-filling + the argmax-flip structure).** Uniform quantization is provably sub-optimal when the distortion sensitivity is non-uniform (reverse water-filling, Cover & Thomas Ch.10). Allocating the INT8 budget against score-sensitivity moves bytes from score-irrelevant to score-relevant weights, so at equal d_seg the total brotli bytes fall (or at equal bytes, d_seg falls). Magnitude is a PREDICTION pending A/B; it is **second-order to Levers 1-2** (it sharpens their win rather than opening a new axis) — hence MEDIUM-HIGH not HIGHEST. Falsification: A/B uniform-QAT vs score-aware-QAT at equal epoch budget, measure `archive_bytes` at equal `d_seg/d_pose`.

### Risk
- **Sensitivity estimate noise.** `||∂S/∂w_t||` is noisy early in training; EMA it and only activate score-aware QAT in late stages (5-8) where the score-domain loss has stabilized.
- **Per-channel complexity.** Per-tensor is the safe first step (matches the codec's per-tensor scale); per-channel is a later refinement.
- **Codec coupling.** Variable per-tensor `n_quant` must round-trip through the codec's per-tensor `scale` (it already stores one scale per tensor — `codec.py:38`); no new grammar needed if we vary the *step* within the existing per-tensor-scale format. HARD-EARNED: the codec already supports per-tensor scale, so 4a is grammar-compatible.

### Deploy plan (gated)
Land `tac.torch_vehicle.score_aware_qat` + `StageSpec` fields default-OFF (uniform vendored QAT preserved → byte-identical). Deploy AFTER Levers 1+2 confirm a frontier (4 sharpens their byte win; deploying it first has nothing to sharpen). A/B in stages 6-8; accept only on `archive_bytes` drop at non-worse distortion; dual exact-eval.

---

## Lever 5 — Bolt-ons PROMOTED to in-curriculum (close the train/deploy mismatch)

**Attacks:** rate + d_seg, by removing the post-hoc-vs-trained gap. **EV: MEDIUM (depends on which bolt-on; coordinate conceptually with the Layer-3 inventory sibling — do NOT duplicate its catalog).**

### Principle
A post-hoc bolt-on (applied to a *frozen* trained archive) is dominated by the same transform applied *in the training loop*, because the decoder can then **co-adapt** to it. Two concrete promotions (the frontier lane `pr110_payload_entropy_recode` IS a post-hoc entropy recode — its in-curriculum form is Lever 1; here are two MORE):

**(5a) Entropy-recode → differentiable + trained-against (this is Lever 1's mandate, named here for the sibling).** The frontier bolt-on re-codes the payload entropy *after* training. Lever 1 makes that recode's objective (brotli-byte cost) differentiable and trains against it, so the decoder *produces* recode-friendly weights instead of being recoded after the fact. The post-hoc recode then has less to do (the bytes are already low) — but more importantly the decoder spent its capacity knowing the byte cost. **This is the canonical "promote the bolt-on" example; it is subsumed by Lever 1.**

**(5b) Margin-conditional residual → training-time d_seg-aware loss.** A post-hoc *margin-conditional residual* bolt-on (grep the sibling's shortlist; e.g. `margin`/`residual` sidecars that nudge frame pixels where the SegNet top-2 logit margin is small, to flip a boundary pixel back) is dominated by **training the decoder against the same margin signal**. The boundary margin map already exists (`tac.substrates.d1_segnet_margin_polytope.margin_map.compute_logit_margin_map` → `top1−top2` logit per pixel = Newton-step distance to the SegNet decision boundary; also `tac.analysis.segnet_boundary_marginals.logit_margin`). Fold it into the seg loss as a **margin-weighted d_seg surrogate**: weight the per-pixel seg-surrogate by `exp(−margin/τ)` (boundary-prone pixels get more gradient; interior pixels ~0). This is **exactly the boundary-aware variant of Lever 2** — the decoder learns to get the small-margin pixels right at training time, so the post-hoc residual sidecar (which costs sidecar bytes) is unnecessary or smaller.

### Exact wire-in point
- (5a) = Lever 1's wire-in (no separate point).
- (5b) = Lever 2's seg term, with an added margin weight: in `driver.py:_train_one_epoch`, compute `margin = compute_logit_margin_map(seg_scorer, decoded_bhwc)` (detached proxy weight, NOT a score claim), then `seg_l = (segnet_surrogate_per_pixel(...) · exp(−margin/τ)).mean()`. New `StageSpec` field `margin_weight_tau` (default None → unweighted = Lever 2 baseline).

### Predicted effect + basis
**PREDICTION (basis: co-adaptation dominance + the argmax-flip concentration at small-margin pixels).** A trained-against transform dominates its frozen post-hoc form because the decoder co-adapts (the same reason eval_roundtrip-in-the-loop beats post-hoc rounding). The margin-weighting concentrates gradient where d_seg actually flips (small top-2 margin), so equal capacity buys lower d_seg AND shrinks/eliminates the post-hoc residual-sidecar bytes. Magnitude is a PREDICTION; **net win requires the eliminated sidecar bytes + d_seg gain to exceed any training-time distortion cost** — measured by A/B against the bolt-on-on-frozen baseline. MEDIUM EV because it overlaps heavily with Levers 1-2 (it is their boundary-concentrated refinement), and the absolute sidecar bytes it removes are typically small.

### Risk
- **Double-counting with the Layer-3 sibling.** The sibling inventories bolt-ons; this lever PROMOTES specific ones. Coordinate: the sibling's "promotable" verdict is the input; do not re-catalog. If a bolt-on is NOT dominated by its in-curriculum form (e.g. a pure-byte ZIP repack with no decoder coupling), it stays a Layer-3 bolt-on — say so.
- **Margin map cost.** `compute_logit_margin_map` runs the SegNet forward (already run for the seg loss — reuse the `seg_out`, don't re-forward). Detached weight only (it is a proxy, Catalog #341 Tier A non-promotable).

### Deploy plan (gated)
(5a) ships with Lever 1. (5b) ships as a `margin_weight_tau` field on Lever 2, default-OFF. Deploy after Lever 2 confirms a frontier; A/B margin-weighted vs unweighted seg loss; accept on d_seg drop + sidecar-byte elimination; dual exact-eval.

---

## 6. RANKED priority order for Phase-2 deployment

Ranking by `(predicted |ΔS| toward T_floor) × (independence of a new axis) / (deploy risk)`, with the binding-constraint map (§0) as the prior. **Deploy in this order; each gated on Phase 1 confirming a frontier and on the prior lever's A/B clearing.**

| Rank | Lever | Axis attacked | Why this order | Deploy gate |
|------|-------|---------------|----------------|-------------|
| **1** | **Lever 2 — score-domain Lagrangian (seg surrogate)** | d_seg (+ frees capacity) | Lowest-risk highest-EV: a *routing* change over pre-built tested surrogates; makes the loss = S; everything else compounds on it. Start option (b) boundary-STE (no new cache). | A/B seg-surrogate vs vendored CE at equal budget; d_seg drops at non-worse d_pose/rate. |
| **2** | **Lever 1 — differentiable rate surrogate** | rate (61.7% of S, the dominant term) | THE rate lever; new code but conservative (order-1 entropy is a true lower bound). Compounds with Lever 2's freed capacity. Decoder-weight conditional-entropy first, latent term second. | A/B rate-on vs off; archive_bytes drops at non-worse d_seg/d_pose. |
| **3** | **Lever 3 — pose-FiLM store** | d_pose (high marginal value near 0) | Highest *structural* win (removes pose from learning) but it's a TRADE (pose-section bytes) and a new archive section. Gate on Phase-1 d_pose telemetry showing headroom above the stored-pose quant floor. Reuse cool_chic `_PoseFiLM`. | Phase-1 d_pose > stored-pose quant floor; A/B sqrt(10·d_pose) reduction NET of pose-section bytes. |
| **4** | **Lever 4 — score-aware QAT** | rate (sharpens Lever 1) | Second-order: sharpens Levers 1's byte win via sensitivity-weighted bit allocation. Needs the score-domain loss (Lever 2) + sensitivity EMA first. | A/B uniform vs score-aware QAT; archive_bytes drops at equal d_seg/d_pose. |
| **5** | **Lever 5 — bolt-on promotion (margin-weighted seg)** | d_seg (boundary refinement) | Overlaps Levers 1-2; promotes specific Layer-3 bolt-ons. Deploy as refinements once 1-2 land. Coordinate with the sibling inventory. | A/B margin-weighted vs unweighted; d_seg drop + sidecar-byte elimination. |

**Compounding note (the co-design thesis):** these are NOT additive independent knobs. Lever 2 frees decoder capacity (pose no longer fought via pixels once Lever 3 lands; d_seg gradient concentrated). Lever 1 then cuts the freed-up decoder's bytes. Lever 4 sharpens Lever 1's allocation. Lever 5 refines Lever 2's boundary. The **joint deployment** (Dykstra-feasible intersection of the rate/seg/pose constraints) is where T_floor lives — but per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check", the joint `ΔS` is a feasibility question, NOT an additive sum; deploy incrementally and let the paired exact-eval arbitrate each step (no compound `ΔS` claim until measured).

---

## 7. The path to T_floor = 0.11797 (which lever attacks what, and the residual gap)

**T_floor = 0.11797 is rate-dominated** (§0: the rate term at the frontier byte count IS ≈ T_floor). The frontier `0.19110` overshoots T_floor by `0.07314` — the **distortion residual** (`100·d_seg + sqrt(10·d_pose)` above their floors). The path is therefore: **drive the distortion residual to ~0 WITHOUT paying it back in bytes, then cut bytes below 177169.**

```
S = 0.19110  (frontier)
  = rate 0.11796  +  100·d_seg_resid  +  sqrt(10·d_pose_resid)   [= 0.07314 distortion residual]

Step A (Levers 2 + 5):  100·d_seg_resid → ~0   (score-domain seg surrogate + margin weighting)
Step B (Lever 3):       sqrt(10·d_pose_resid) → sqrt(10·d_pose_quant)  (pose stored as side-info)
                        ── after A+B, S ≈ rate + ε_distortion ≈ 0.118 + ε  ──  T_floor REACHED if ε→0
Step C (Levers 1 + 4):  cut rate below 0.11796   (differentiable brotli surrogate + score-aware QAT
                        free decoder capacity once A+B stop spending it on distortion)
                        ── this pushes BELOW T_floor's frontier-byte anchor toward the true rate floor ──
```

**Residual-gap honesty (NO FAKE):**
- T_floor = 0.11797 is the rate term *at the current 177169 B*. It is **not a hard floor** — it is "the rate you pay for the bytes you currently need." Lever 1 + Lever 4 attack the byte count itself, so the *true* floor is below 0.11797 (set by the genuine information content of the SegNet-argmax + pose witness, the S_floor=0.11797 rate-dominated derivation in the GOAL section is the *headroom proof*, not the terminus).
- The distortion residual `0.07314` is the **immediately reachable headroom**: Levers 2+5 (d_seg) + Lever 3 (d_pose) attack it directly, at constant-or-lower byte. If they drive it to ~0, `S → ~0.118` — **crossing the sub-0.15 target T_3** and landing essentially AT the rate-dominated floor.
- The gap from `~0.118 → below` requires Levers 1+4 to cut the 177169 B. That is the harder, second-phase descent (rate floor is the genuine information-theoretic limit; below it is impossible without losing witness fidelity).

**The binding constraint at each phase:** Phase-2a (Levers 2,5,3) is **distortion-bound** → the levers are d_seg/d_pose surrogates. Phase-2b (Levers 1,4) is **rate-bound** → the levers are byte surrogates. Deploy in that order; the binding constraint tells you which lever is on the critical path.

---

## 8. Wire-in hooks (CLAUDE.md 6-hook declaration per Catalog #125) + provenance

This is a **DESIGN memo** (`research_only` in the sense that no code/score lands here — the specs gate Phase-2 code). Hook status for the DESIGNED levers (each ACTIVE when the lever's code lands in Phase 2):
1. **Sensitivity-map** — ACTIVE for Lever 4 (per-tensor `||∂S/∂w_t||` IS a sensitivity map; feeds the bit-allocator). Lever 5's margin map is a per-pixel sensitivity map.
2. **Pareto constraint** — ACTIVE: the score-domain Lagrangian (Lever 2) IS the Pareto objective; Levers attack distinct constraints (rate/seg/pose) → the joint is a Dykstra-feasible intersection (§6 note).
3. **Bit-allocator hook** — ACTIVE for Levers 1+4 (rate surrogate + sensitivity-weighted QAT ARE bit-allocator primitives).
4. **Cathedral autopilot dispatch** — N/A at design time (no archive-deployable artifact yet; ACTIVE when Phase-2 code lands a dispatchable trainer flag).
5. **Continual-learning posterior** — DESIGN: each lever's A/B is a falsifiable empirical anchor; the A/B verdicts reseed the planner (the predicted-vs-measured `ΔS` per lever becomes a canonical-equation anchor candidate per Catalog #344).
6. **Probe-disambiguator** — ACTIVE: Lever 3's deploy gate (Phase-1 d_pose vs stored-pose quant floor) IS a probe-disambiguator; Lever 4's sensitivity-noise gate likewise.

**Mission contribution:** `frontier_breaking_enabler` (DESIGN that gates Phase-2 frontier-breaking code; the END is a lower exact score, this is the MEANS — stated plainly per the means/ends firewall). **Frontier UNMOVED 0.19109982.** No score asserted. No GPU launched.

**NO FAKE final check:** every quantified effect above is tagged PREDICTION with a named first-principles basis (Shannon order-1 bound / Wyner-Ziv side-info / reverse water-filling / co-adaptation dominance / argmax-flip structure). No measurement is claimed. Every lever's deploy plan ends at the **dual CPU/CUDA exact-eval gate** before any score claim, and each is **gated on Phase 1 confirming a frontier** (no contention with the live basin).


