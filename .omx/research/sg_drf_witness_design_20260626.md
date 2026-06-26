# Score-Guided Deterministic Rectified-Flow Witness (SG-DRF) — design memo

**Date:** 2026-06-26 · **Status:** research+design ($0; NO training/GPU/dispatch) · **Evidence:**
all rows below are DESIGN/DERIVATION or `[macOS-MLX research-signal]` plans; nothing here is a
score claim. **Authority:** MLX/CPU train-gradient + CPU/CUDA-only score authority (MPS never).

## 0. One-paragraph thesis

The witness capstone needs a TRAINED continuous-texture generator that amortizes the
SegNet-argmax partition + its keying texture across the 600 seg-frames. The in-flight backbone
is a **single-forward coordinate-INR** (`ScoreNativeSegGenerator`, `lever_b_generator.py`) with
two MEASURED root failures: (a) **spectral bias** — the coordinate-MLP lays low frequencies
first, so the argmax step carries Gibbs rings that FLIP through the eval R; (b) the
**`params^-0.71` capacity power law** (manifold_topology_dseg_deep_synthesis_20260623.md) — a
fixed-depth single forward buys d_seg only as `params^-0.71`. This memo designs **SG-DRF**: the
witness frame is produced by a small **rectified-flow velocity field** `v_θ(x,t|z)` integrated by
a **deterministic ODE** (free generic compute in `inflate.py`) from a fixed noise seed,
conditioned on a tiny per-frame latent `z` (counted). It is **score-guided** — the frozen
SegNet/PoseNet steer the flow toward argmax/pose-correctness at COMPRESS TIME ONLY (scorer-free
at inflate, per the strict-scorer rule), trained THROUGH the exact R. The bet: (1) iterative
integration buys EFFECTIVE DEPTH free of the param budget → beats the `^-0.71` law; (2) a
generative noise→texture transport SAMPLES in-distribution interior texture cheaply instead of
MEMORIZING it against spectral bias. Both directly attack the two measured root failures.

## 1. The frozen eval contract (R) this witness must satisfy (grounded)

From `upstream/frame_utils.py` + `upstream/modules.py`:
- `camera_size = (W=1164, H=874)`; `segnet_model_input_size = (W=512, H=384)`; `seq_len = 2`
  (600 NON-overlapping pairs from 1200 frames).
- **d_seg path (R):** witness frame at camera-res `(874,1164)` uint8 → `SegNet.preprocess_input`
  takes **`x[:,-1,...]` (the LAST/odd frame of the pair ONLY)** → `F.interpolate(..., (384,512),
  mode='bilinear')` → SegNet → `argmax`; `d_seg = mean(argmax(witness) != argmax(GT))`
  (`modules.py:112`). **No bicubic-up.** Consequence: only ONE frame per pair (the odd frame)
  carries d_seg; the witness's seg job is 600 odd-frame argmax maps.
- **d_pose path:** `MSE(PoseNet(pair)[:6], PoseNet(GT pair)[:6])` over the two-frame YUV6 input.
  SOLVED by the stored-target sidecar (`scorer_targets.py`): 600×6×fp16 zlib `<5KB`; NOT an
  SG-DRF controllable. The witness's sole binding controllable is **d_seg**.
- **Score:** `S = 100·d_seg + sqrt(10·d_pose) + 25·|archive.zip|/37_545_489`. Pointer UNMOVED
  0.19110. Rate axis at entropy floor → **only d_seg moves S**; rate is a HARD BUDGET the witness
  weights+latent must fit inside (sub-0.15 ⇒ ~bc20-class rate ≈ 0.059 ≈ 88 KB counted).

## 2. LITERATURE synthesis + rate-model fit

The 6 load-bearing findings (papers read in full; URLs cited):

1. **Rectified Flow + Reflow** (Liu, Gong, Liu 2022, [arxiv 2209.03003](https://arxiv.org/abs/2209.03003);
   [cs.utexas.edu/~lqiang/rectflow](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html);
   code [github.com/gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow)). ODE
   `dZ_t=v(Z_t,t)dt`; linear interpolant `X_t=(1-t)X_0+tX_1`; LS objective
   `min_v ∫E‖(X_1-X_0)-v(X_t,t)‖²`, optimum `v=E[X_1-X_0|X_t]`. **Reflow** retrains on the model's
   own (noise→sample) pairs; straightness decays `O(1/K)`; perfectly straight ⇒ `Z_t=Z_0+t·v(Z_0,0)`
   = exact **1 Euler step**. "Sufficient to reflow once." Numbers: 2-reflow+distill = 1-step FID 4.85
   CIFAR10. **Load-bearing for SG-DRF:** straightening makes few-step deterministic sampling exact,
   so the FREE integrator does the work and bytes go only to weights+latent.
2. **FlowMAC** ([arxiv 2409.17635](https://arxiv.org/html/2409.17635v2), ICASSP 2025). Conditional
   flow-matching **decoder** for low-bitrate (3 kbps) audio coding — the exact latent→flow→recon
   pattern. Quantized latent (residual-VQ, 256 codebook × 8 stages) **concatenated to the Gaussian
   noise input** of the velocity net; CFM loss `‖v_t(x)-(x_1-(1-σ_min)x_0)‖²`. Inference **32 Euler
   steps, tunable down to 1 step** (FlowMAC-LC, CPU real-time). Stored bytes = RVQ indices ONLY;
   mel + waveform are COMPUTED at decode. **Load-bearing:** the conditioning-latent-is-the-payload,
   network-is-shared template + the "tune #steps freely at decode" property maps onto our free-integrator.
3. **Score/classifier-guided RF.** **RectifID** ([arxiv 2405.14677](https://arxiv.org/html/2405.14677v4),
   NeurIPS 2024): guided velocity `v̂=v+s·∇log p(c|z_t)` (Helmholtz potential) lets a **FROZEN
   off-the-shelf classifier** steer the ODE; **anchored** guidance applies the gradient at the
   trajectory ENDPOINT damped by the Jacobian, making the fixed-point a **contraction (linear
   convergence) when `s<1/(L_1 L_2)`** — the stability key for our long through-R path. **CFG-Zero***
   ([arxiv 2503.18886](https://arxiv.org/abs/2503.18886)): early ODE steps have inaccurate velocity
   so guidance pushes onto WRONG trajectories; fixes = per-step optimized scale + zero-init first
   steps. **Load-bearing:** anchored guidance + zero-init directly mitigate the witness's prior
   optimizer-divergence collapse.
4. **Deterministic sampler complexity** ([arxiv 2508.08735](https://arxiv.org/html/2508.08735v1)).
   First polynomial discretization bound for RF deterministic samplers: one-step
   `K∝L_f/ε_{W2}⁴` (uniform schedule). Straighter (reflowed) ⇒ smaller Lipschitz `L_f` ⇒ fewer
   steps. **Load-bearing:** few-step decode is provably poly(d,1/ε); reflow lowers the constant.
5. **Flow Matching Guide & Code** (Lipman et al, [arxiv 2412.06264](https://arxiv.org/abs/2412.06264)).
   Canonical CFM `L=E‖u_t^θ(X_t)-(X_1-X_0)‖²` on the linear/OT path (regression target = constant
   `X_1-X_0`); sample = Euler `x←x+h·u_θ`. **Load-bearing:** the minimal `path/scheduler/solver/loss`
   impl pattern we mirror in MLX.
6. **Flows on the Neural Manifold** ([arxiv 2507.10623](https://arxiv.org/html/2507.10623v2)). Flow
   matching over NN WEIGHTS (source = Kaiming init, VAE-compressed weight manifold); scope `<10⁶`
   params; generated nets work (98.5% MNIST). **Load-bearing (secondary):** a future path to
   compress `v_θ` ITSELF on a learned weight manifold if the weight-rate risk (§8.2) binds.

**OSS to port:** [lucidrains/rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch)
(**MIT**, tiny `RectifiedFlow(model)` + reflow utils — the port target) ·
[bamler-lab/constriction](https://github.com/bamler-lab/constriction) (entropy coder, ANS +
Range/queue, Rust+Python binary-compatible, MIT/Apache/BSD — Range coder for the AR latent) ·
⚠️ [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) is
**CC-BY-NC (non-commercial)** — reference only, do NOT vendor into a submission.

### 2.x Rate-model fit (the inflate.py free-interpreter boundary, per CLAUDE.md)

`upstream/evaluate.py:63` sizes ONLY `archive.zip`; `inflate.py/inflate.sh` are FREE (untimed
except the 30-min budget; `evaluate.py:92` has no time term). Mapping SG-DRF onto the
free/counted boundary:

| SG-DRF component | FREE in inflate.py | COUNTED in archive.zip |
|---|---|---|
| ODE integrator (Euler/Heun steps, fixed t-schedule) | ✅ generic deterministic algorithm | — |
| `v_θ` forward-pass CODE | ✅ generic algorithm | — |
| initial noise `x_0` | ✅ seeded Gaussian, reconstructed from a fixed integer seed | — |
| **`v_θ` weights (LEARNED)** | — | ✅ rule-118 large artifact (the dominant cost) |
| **per-frame latent `z` (VIDEO-DERIVED)** | — | ✅ counted (tiny) |
| stored-pose sidecar | — | ✅ `<5 KB` (already built) |

NO-FAKE boundary (binding): the integrator/forward-code/noise-seed are generic and free; the
weights+`z` are learned video-derived payload and ARE counted. Smuggling a per-frame table into
"code" to dodge the rate term is the hide-data-in-code fake (NO-FAKE #6/#7) — FORBIDDEN.

## 3. SG-DRF ARCHITECTURE (all ingredients bound)

**Target:** one camera-res odd seg-frame per pair (3-channel uint8 at `(874,1164)`), 600 frames,
ONE shared `v_θ` + 600 tiny `z`. (Pose rides the sidecar; the even frame can be a cheap
warp/copy or a second tiny `z` — not d_seg-binding.)

1. **Conditioning latent `z` (per frame, COUNTED).** `z ∈ R^{d_z}`, `d_z ≈ 16–32` (matches the
   coord-INR `mod_dim=32`). Encodes the LOW-dimensional argmax-relevant content: the partition
   geometry + which texture realization. NOT the texture pixels themselves (those come from the
   noise→data transport). Quantized int8 + temporal-delta + raw-LZMA/AR entropy coding
   (PR95 L24/L25 lineage). Estimate: `600 × 24 dims × ~1 B (post-AR) ≈ 5–15 KB` total.
2. **Velocity field `v_θ(x, t | z)` (shared, COUNTED).** A SMALL **convolutional** U-shaped net
   on the pixel grid (NOT a coordinate-MLP — this is the key anti-spectral-bias choice; conv
   represents sharp local steps without coordinate-MLP low-freq bias). Resolution-progressive
   (work at reduced res then upsample to camera-res to keep FLOPs/params low). Conditioning:
   **FiLM** of `z` (and a sinusoidal `t`-embedding) into each block's GN/scale-shift — reuses the
   FiLM-per-pair mechanism already proven in the coord-INR. Optional thin cross-attn only if FiLM
   under-fits. **Budget:** `~150–300 K params` → at FP4/int4 + brotli ≈ `60–110 KB`. This is the
   binding rate term; the whole sub-0.15 bet is that a flow hits adequate d_seg at this weight
   size where a single-forward INR cannot (§5).
3. **Deterministic sampling (FREE).** Rectified-flow straight-line transport `x_t=(1-t)x_0+t x_1`;
   **reflow** the trained couplings 1–2× to straighten trajectories so few Euler steps suffice.
   Inflate: `N` Euler/Heun steps (N≈4–16; tune for argmax-sharpness vs the 30-min budget — bytes
   are unaffected by N). Fixed t-grid, fixed seed `x_0`. Deterministic + portable (numpy/MLX
   reference + parity test per the packet-compiler discipline).
4. **Score-guidance (TRAIN/COMPRESS-time only; §4).** The novel piece.
5. **Amortization (§5).** `v_θ` carries the texture STATISTICS once for all 600 frames; `z` picks
   per-frame partition + realization.

## 4. SCORE-GUIDANCE mechanism (the novel contribution) + NO-FAKE check

Two coupled mechanisms, both scorer-free at inflate:

**(A) Compress-time `z`-optimization through the exact R (load-bearing, NO-FAKE-clean).** After
`v_θ` is trained, freeze it and OPTIMIZE each `z` to minimize a through-R d_seg surrogate:
`L(z) = CE( softargmax( SegNet( bilinear_down( STE_uint8( ODE_solve(v_θ, x_0, z) )))), argmax(GT) )`.
This is supervised TTO on `z` against the FROZEN SegNet (compress time), identical in spirit to
the stored-pose sidecar's supervised TTO — the scorer is in the COMPRESS loop, never the inflate
loop. The optimized `z` is quantized + stored (counted). At inflate: deterministic ODE from
`(x_0, z)`, NO SegNet. **NO-FAKE-clean:** `z` is counted video-derived payload optimized against
the REAL frozen oracle; the deployed witness loads no scorer.

**(B) Train-time score-guided velocity shaping (optional, shapes weights only).** During `v_θ`
training, use anchored classifier guidance (RectifID-style) so the sampler is pulled toward the
SegNet-argmax-correct manifold: at each step add `+λ·∇_x [softargmax-CE vs GT]`, ANCHORED to the
straight-line target to keep the trajectory on-manifold (the anchor is what stabilizes guidance —
the witness's prior optimizer-divergence collapse warns us here). The velocity↔score identity of
flow matching makes this a principled drift correction. This guidance is COMPRESS-TIME ONLY; it
biases the learned `v_θ` weights (counted) but is absent at inflate. **NO-FAKE-clean** for the
same reason.

Both are trained THROUGH the exact R (uint8 STE + bilinear-down + soft-argmax surrogate of the
non-differentiable argmax). The novelty vs the literature: **score-guided rectified flow as a
COMPRESSION decoder trained against a frozen task model through the exact eval operator** —
FlowMAC conditions on a quantized latent but optimizes reconstruction MSE; we optimize a frozen
classifier's argmax through a fixed degradation R. No prior work targets indirect (remote) RD /
coding-for-machines with a guided deterministic flow.

## 5. AMORTIZATION ARGUMENT — why SG-DRF beats the coord-INR `^-0.71` law for the texture crux

1. **Iterative depth defeats the single-forward power law.** The coord-INR is one forward pass;
   its d_seg falls only as `params^-0.71`. SG-DRF re-applies the SAME `v_θ` for `N` ODE steps →
   effective depth `≈ N × depth(v_θ)` at NO extra counted bytes (steps are free integrator
   compute). Effective expressivity-per-byte scales with N, so the byte-budget→d_seg curve is
   strictly better than a same-param single forward whenever the target needs depth (sharp,
   nested argmax boundaries do). This converts FREE inflate compute into capacity the INR must
   buy in counted params.
2. **Generative texture sampling beats spectral-bias memorization.** The binding wall is INTERIOR
   TEXTURE-KEYING: SegNet keys off interior texture STATISTICS in its null space (flat-paint
   penalty +0.00562 ⇒ cheap in-distribution texture suffices). A coord-INR must MEMORIZE texture
   as a deterministic function of coordinates and fights spectral bias (rings on the step). A
   flow LEARNS A NOISE→TEXTURE TRANSPORT: the texture statistics live in `v_θ` weights (amortized
   once across 600 frames), the per-frame `z` only selects the partition + realization. Matching
   a DISTRIBUTION (what SegNet's null space accepts) is exactly what a generative transport does —
   and far cheaper than pixel-exact regression. This is the direct match between the measured
   crux (statistical, null-space, in-distribution texture) and the tool (a sampler).
3. **Conv velocity field has no coordinate-MLP spectral bias.** Sharp argmax steps are local; a
   convolutional `v_θ` represents them with local kernels, killing the Gibbs-ring flip source.

Net: SG-DRF spends the SAME (or fewer) counted bytes on the scorer-relevant manifold while using
FREE iterative compute for the depth/texture the INR pays for in parameters — the capacity-vs-rate
trilemma resolution the capstone requires.

## 6. BYTE-CLOSE plan (composition into the archive / L13)

- **Counted:** `v_θ` weights `~60–110 KB` (FP4/int4 + per-tensor byte-maps + split brotli, PR95
  L21–L23/L29) + `z` blob `~5–15 KB` (int8 + temporal-delta + raw-LZMA, L24/L25) + stored-pose
  sidecar `<5 KB`. **Total estimate `~70–130 KB`** ⇒ rate `25·bytes/37.5M ≈ 0.047–0.087`. Sub-0.15
  needs d_seg `≲ 0.0009` at the low end of that band. The weight size is the binding uncertainty.
- **Free (inflate.py):** the ODE integrator, `v_θ` forward code, fixed t-grid, seeded `x_0`, the
  uint8/resize R-side ops, the pose-sidecar reader. Monolithic single-file archive grammar (PR95
  L20) with length-prefixed sections `(v_θ, z, pose-sidecar)`; numpy-portable reference +
  byte-parity test (packet-compiler discipline); fail-closed on no-op.
- **Composition:** witness produces the 600 odd seg-frames (d_seg); even frames cheap; pose from
  sidecar; integrates as a new backbone behind the existing L13 task-space format.

## 7. Determinism + camera-res sub-pixel top-up (#149) as a 0-rate bolt-on

The ODE is deterministic (fixed seed, fixed schedule) ⇒ reproducible bytes. The banked
**camera-res sub-pixel boundary placement** top-up (#149) is a 0-rate, train/compress-time
deterministic post-pass on the camera-res frame BEFORE R: nudge boundary pixels sub-pixel so that
after bilinear-down the argmax lands correct — composes additively after the ODE output, adds 0
counted bytes, scorer-free at inflate (the nudge field, if any, is a deterministic function or a
tiny counted residual). Stacks with SG-DRF.

## 8. HONEST RISKS (top 3 failure modes)

1. **Guidance / through-R optimizer instability.** The witness ALREADY suffered an
   optimizer-divergence collapse (Muon + weak pose term). The SG-DRF gradient path is LONG
   (ODE-unroll × `v_θ` × uint8-STE × bilinear × soft-argmax) and guidance can blow up. Mitigation:
   anchored guidance (RectifID), EMA, conservative LR, decouple (A) `z`-TTO from (B) weight
   guidance, gradient checkpointing. RISK that it never trains stably through R.
2. **Weight rate blows the budget (dominated → just PR95).** The entire bet is that a flow models
   texture more byte-efficiently than a coord-INR. If `v_θ` must be large (bc36-class ~118 KB
   rate) to model texture, SG-DRF lands at PR95's 0.19 and is DOMINATED. Untested for this target.
3. **Few-step quality vs argmax sharpness.** Reflow may not straighten enough; few Euler steps may
   leave the argmax step soft → flips through R. More steps are byte-free but cost the 30-min
   budget; even unlimited steps may not hit `d_seg ≲ 0.0009` if `z`/`v_θ` underfit the partition.

## 9. $0 FEASIBILITY PROBE (smallest faithful de-risk before ANY training arm)

**Question (de-risks the core bet directly):** can a tiny few-step rectified flow conditioned on a
tiny latent reconstruct ONE seg-frame's argmax through the EXACT R at all — and beat the
single-forward coord-INR at equal/lower params, ring-free?

**Setup (CPU/MLX, single frame, $0):**
- Pick ONE odd seg-frame + its GT SegNet argmax (the real `upstream/videos/0.mkv`, real frozen
  SegNet, exact R: camera-res uint8 → bilinear-down (384,512) → SegNet → argmax). MLX SegNet =
  train-gradient device; CPU SegNet = d_seg readout authority (`[macOS-CPU advisory]`). NO MPS.
- Build a tiny conv `v_θ` (~50–100 K params) + `d_z=16` in MLX; OVERFIT this one frame: noise→frame
  with `L = softargmax-CE(through-R) vs GT argmax`; reflow 1×; sample with N=4 Euler steps.
- Control: the existing single-forward `ScoreNativeSegGenerator` at MATCHED params on the same one
  frame.

**PASS gate (build SG-DRF):** the N=4 flow drives this one frame's d_seg BELOW the matched-param
coord-INR's single-frame floor (coord-INR best-measured ≈ 0.004445; n600 baseline 0.00826)
THROUGH the exact R, AND its argmax step is visibly ring-free where the INR rings (spectral-bias
win confirmed by overlay). **FAIL gate (do NOT build):** with full overfit the tiny flow cannot
beat the matched-param INR single-frame floor through R ⇒ the iterative-depth/texture-sampling
advantage does not materialize for this target ⇒ SG-DRF weaker than coord-INR; record the
negative and stay on the coord-INR gate.

This is $0 (one frame, MLX/CPU, no GPU/dispatch), faithful (exact R + real SegNet + real GT), and
falsifiable (d_seg threshold + ring overlay). It also yields a sister diagnostic: whether `z`-TTO
(mechanism A) alone, or A+train-guidance (B), is needed.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale (falling-rule) |
|---|---|---|
| Velocity field `v_θ` (conv U-net) | **FORK_PRINCIPLED** (unique) | coord-MLP backbone has MEASURED spectral-bias ring flips; a flow needs a grid-native conv field. New substrate-engineering, not a bolt-on. |
| FiLM-per-frame conditioning | **ADOPT_CANONICAL** | the FiLM-per-pair-mod mechanism is proven in `lever_b_generator.py`; reuse. |
| Deterministic Fourier/`t`-embed table | **ADOPT_CANONICAL** | `deterministic_fourier_B` free-table pattern reused for the `t`-embedding (0 bytes). |
| Stored-pose sidecar | **ADOPT_CANONICAL** | pose SOLVED (`scorer_targets.py`); do not re-treat. |
| Latent entropy coding (`z`) | **ADOPT_CANONICAL** | PR95 L24/L25 temporal-delta + raw-LZMA. |
| Weight coding (`v_θ`) | **ADOPT_CANONICAL** | PR95 L21–L23/L29 FP4 + byte-maps + split brotli. |
| Archive grammar | **ADOPT_CANONICAL** | PR95 L20 monolithic length-prefixed sections. |
| Score-guidance through R | **FORK_PRINCIPLED** (unique, novel) | indirect-RD coding-for-machines target; no canonical helper; the contribution. |
| Sampler/integrator | **FORK_PRINCIPLED** (unique) | rectified-flow + reflow ODE is new to the repo; numpy-portable + parity test. |

## Observability surface

1. **Inspectable per layer:** dump `x_t` at every ODE step (the trajectory), per-block FiLM
   scales, the through-R soft-argmax map vs hard argmax, the flip mask (witness argmax != GT).
2. **Decomposable per signal:** d_seg per pair / per class / per boundary-annulus; rate split
   `(v_θ bytes, z bytes, pose bytes)`; loss split `(through-R CE, guidance term, anchor term)`.
3. **Diff-able across runs:** fixed seed ⇒ byte-level + d_seg + trajectory diff between
   (steps N, reflow count, d_z, guidance λ) configs.
4. **Queryable post-hoc:** persist per-frame `z`, `v_θ` npz, per-step trajectories, d_seg rows as
   `.npz/.jsonl` on the SSD tier (NO /tmp), readable without re-running.
5. **Cite-able:** every row tagged `(commit, config, seed, upstream_snapshot_sha, device)`.
6. **Counterfactual-able:** byte-mutation no-op detector (mutate `z`/`v_θ` bytes → confirm the
   inflate output + d_seg change), reflow-count ablation, step-count ablation, guidance-off
   ablation — all without retraining `v_θ` (z-TTO + sampling are cheap).

## Cross-references

- `lever_b_generator.py` (the coord-INR backbone this would replace/augment) ·
  `scorer_targets.py` (stored-pose sidecar) · `upstream/modules.py` + `upstream/frame_utils.py`
  (the frozen R) · `.omx/research/manifold_topology_dseg_deep_synthesis_20260623.md` (the
  `α≈0.71` power law) · `.omx/research/witness_capstone_deepmath_levers_20260625.md` (the
  all-class directional-basis lever, a sister 0-byte prior that composes) · CLAUDE.md
  §"THE CURRENT FRONTIER ... WITNESS CAPSTONE" + §"inflate.py is a FREE interpreter".
