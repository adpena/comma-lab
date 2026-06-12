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

