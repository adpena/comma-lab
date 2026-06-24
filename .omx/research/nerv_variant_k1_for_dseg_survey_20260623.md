# NeRV-variant k1-for-d_seg survey — is a non-HNeRV decoder lower-k1 for the frozen-contest boundary?

- **Subagent:** `nerv-variant-k1-survey-20260623`
- **Date:** 2026-06-23 (UTC 2026-06-24)
- **Scope:** $0 deep-math survey (Part A) + queued MPS k1-screen plan (Part B). Advisory / NON-PROMOTABLE. Pointer-only (frontier 0.19110 UNMOVED).
- **Authority:** all score math via `from tac.contest_score import ...`; no proxy/PSNR ranking; no exact row produced in this unit.
- **Queue discipline:** the concentrated-taper screen (`launch_split_by_head_basin`, PID 77644, `--train-device mps`) is ACTIVELY using MPS. Part B is a **queued plan**, not run, to avoid contention (verified `pgrep` before deciding).

---

## 0. Honest reframe of the framing (NO-FAKE / means-vs-ends)

The prompt's closed form (`S(C)=100·k1·C^-α + pose + 25·B/N`, "−30% k1 → S*≈0.165") is a **derivation at an assumed lower-k1 operating point**, not the measured one. At the **actual measured** small-basis bc20 operating point the arithmetic is much harsher — and this must be stated plainly so the architecture axis is judged against the real number:

| term | value (bc20, d_seg≈0.00279, d_pose≈0.000342, B≈73 KB) |
|---|---|
| `seg_term`  | **0.27900** (← the score is **d_seg-DOMINATED**) |
| `pose_term` | 0.05848 |
| `rate_term` | 0.04861 |
| **S** | **0.38609** |

`break_even_d_seg(0.15, …)` at this pose+rate = **0.000429** → sub-0.15 needs a **6.5× d_seg reduction**, not −30%.

k1-lever sensitivity (fixed bytes, fixed pose), via `tac.contest_score`:

| k1 multiplier | d_seg | S |
|---|---|---|
| ×1.00 | 0.00279 | 0.38609 |
| ×0.85 (−15%) | 0.00237 | 0.34424 |
| ×0.70 (−30%) | 0.00195 | 0.30239 |
| ×0.50 (−50%) | 0.00139 | 0.24659 |
| ×0.154 | 0.00043 | **0.15000** |
| ×0.70 **+ L13 witness coding (−59% bytes)** | 0.00195 | **0.27371** |

**Verdict on the lever, up front:** a lower-k1 decoder is a **real, monotone** score lever (−30% k1 = −0.084 S; −50% = −0.140 S), and it **stacks multiplicatively** with the byte axis (L13). But **no single architecture swap reaches sub-0.15** from here — sub-0.15 needs ~6.5× on d_seg, which is an *architecture × taper × byte-coding × long-train* compound. The architecture axis is worth pursuing **as the highest-leverage single factor on the dominant term**, not as a one-shot win. This is consistent with the standing finding that d_seg is the binding constraint (`feedback_small_basis_rate_headroom_is_the_sub015_asset...`).

---

## Part A — deep-math ranked survey

### A.1 The d_seg signal, precisely

`d_seg = (argmax(SegNet(f1)) != argmax(SegNet(f2))).mean()` — the **per-pixel argmax-flip rate** between the two reconstructed frames vs GT (verified `upstream/modules.py`). Geometrically this is a **codimension-1 set**: the contest video's d_seg is concentrated on the **moving horizon line + fuzzy class-transition bands** (sky/road/structure boundaries). It is a **high-frequency, sharp, sparse** feature: most pixels are deep-interior (margin large, never flip); the signal lives in a thin boundary tube where the SegNet decision margin is small.

Two math facts pin the architecture screen:

1. **Eval round-trip / Nyquist.** SegNet's input is a bilinear resize to `(512,384)`; the decoder synthesizes at `(384,512)` — same scale (axis transpose), so there is **no extra downsample band-loss** between render-res and scorer-res. The full render-resolution boundary band feeds d_seg → the decoder must put **high frequency exactly at the boundary band that exists at 384×512**, not beyond (no Nyquist waste) and not below (low-freq bias starves the edge).
2. **Boundary = piecewise-smooth-with-discontinuities.** The argmax map is piecewise-constant (each region = one class) with sharp jumps. The sparsest basis for piecewise-smooth-with-discontinuities is the **wavelet basis** (Daubechies / Mallat): O(1/N) coefficients to represent a 1-D discontinuity vs O(N) for Fourier. This is the single strongest a-priori reason an **edge-adapted basis** should be lower-k1 than a fixed-sine coordinate network.

### A.2 The k1 cost of the HNeRV baseline (the thing to beat)

The live decoder (`src/tac/torch_vehicle/configurable_taper_decoder.py`, a faithful generalization of vendored PR95 `hnerv_muon`):

```
Linear stem(latent→c0·48) → 6×[ Conv3×3(c_i, c_{i+1}·4) + PixelShuffle(2)
                                 + sin(·)  + bilinear-skip ] → refine → rgb_0/rgb_1
```

- **Activation = `torch.sin(x)` with FIXED ω=1** (line 177/185). This is the canonical NeRV sine — it has the **coordinate-network low-frequency spectral bias** (Rahaman 2019; Tancik 2020): a fixed-ω sine MLP/ConvNet learns low frequencies first and **under-represents sharp edges** unless ω is large, which it isn't here. → **high k1** on a sharp boundary.
- **Conv params = `in·out·k²`** — resolution-independent. This is why the taper reallocation (a5825bc) is byte-neutral and works: move channels to the high-res mid/late stages where the boundary is decided. **The taper attacks WHERE capacity sits; the activation/basis attacks WHAT each parameter can represent.** They are **orthogonal k1 levers** — this is the core reason the architecture axis is a real edge *beyond* the taper.
- PixelShuffle upsampling + bilinear-skip is bandwidth-efficient but does not change the spectral bias of the sine nonlinearity.

### A.3 The candidates, ranked by predicted-k1-vs-HNeRV (the high-freq mechanism)

All score math is in `tac.contest_score`. "build-status" from repo grep. "predicted k1" is a **spectral-bias / sparsity argument**, NOT a measurement (per the existence-proof rule, the claim needs Part B's measured d_seg before it is a verdict).

| rank | variant | high-freq mechanism (the math) | predicted k1 vs HNeRV | build status | byte/param efficiency | the math reason |
|---|---|---|---|---|---|---|
| **1** | **WIRE** (Gabor activation) | `sin(ωx)·exp(-½(s·x)²)` — a **Gabor wavelet**: jointly localized in space AND frequency (optimal Heisenberg product). | **LOWEST** (predicted −25–45%) | **BUILT, byte-closed** activation mode in `siren/activation_family.py` (`apply_activation_family(..., "wire")`); SIREN coord-MLP arch is L0 sketch | high — Gabor atoms are the sparsest representation of a *localized oscillating edge*; few atoms cover the boundary tube | A sharp class boundary is a **spatially-localized high-frequency event**. Gabor/wavelet atoms are *designed* for exactly this (Mallat). Fixed-sine spreads energy globally; Gabor concentrates it on the edge → fewer params per unit edge-fidelity → lower k1. |
| **2** | **Wavelet decoder** (fixed DB4 IDWT + synthesis MLP) | Stores per-pair LH/HL/HH **detail subbands**; IDWT synthesis. The detail subbands ARE the codimension-1 edge basis. | **LOW** (predicted −20–40%) | **BUILT** `substrates/wavelet/` (arch+archive+inflate+loss), L0 sketch, never anchored on frozen scorer | very high on edges — O(1) coeffs per discontinuity; **the rate budget IS the subbands** (already RD-aligned, cf. Z8 detail-coeff work) | Daubechies basis = sparsest for piecewise-smooth-with-jumps (= the argmax partition). The detail subbands put representational capacity exactly at the boundary band. Sister of Z8 hierarchical-PC (pose-solved, **seg still the ceiling** — wavelet at the *decoder* not just the codec is the untested move). |
| **3** | **FINER** (variable-frequency sine) | `sin(ω·(\|x\|+1)·x)` — the effective frequency **scales with \|x\|**, so high-activation regions (near edges) get higher local frequency *automatically*. | **MEDIUM-LOW** (predicted −10–25%) | **BUILT, byte-closed** activation mode (`"finer"`); drop-in on any sine decoder | same params as HNeRV (pure activation swap → **byte-IDENTICAL**) | Directly attacks the fixed-ω spectral bias: FINER's variable period lets the SAME network express higher frequency where the signal is large (the boundary), without adding parameters. **Cheapest possible k1 lever** — zero byte cost, drop-in on the live decoder's `sin`. |
| **4** | **HiNeRV** (explicit multi-scale hierarchy) | Separate per-stage decoders + per-stage RGB heads + multi-scale supervision → coarse captured low, fine stages forced to high-freq. | MEDIUM (−10–20%, but **byte-COST**) | `hi_nerv/` + `hinerv_as_renderer.py` (legacy research-only); L0/sketch | **lower** — per-stage decoders add params (Hu 2024 claims ~1.4× *param* efficiency but at more total params) | The multi-scale supervision is a strong high-freq inductive bias, but the separate per-stage decoders raise the byte budget; k1 = d_seg per *byte*, so the param cost partly cancels the d_seg gain. Dominated by FINER/WIRE on the byte axis. |
| **5** | **FF-NeRV** (band-limited DCT coeffs) | Predicts low-N DCT coefficient grids; IDCT2 synthesis. **Band-limited by construction** (HF coeffs = 0). | **HIGHER** (predicted **+k1**) | `ff_nerv/` BUILT, L0 sketch | high on smooth content, **terrible on edges** | DCT/Fourier is the **WORST** basis for a sharp discontinuity (Gibbs ringing; O(N) coeffs per edge). Band-limiting *removes* exactly the HF the boundary needs. Predicted to *raise* k1. Listed to be ruled out, not pursued. |
| **6** | **DS-NeRV** (depthwise-separable) | Same sine activation; factorizes conv `C²K²→C·K²+C²`. | NEUTRAL on k1; better byte/param | `ds_nerv/` BUILT, L0 sketch | **best param efficiency** (−40% params/conv) | Does NOT change the spectral bias (same `sin`) → same d_seg-per-param-*representation*; but cheaper params → could be a **byte-axis multiplier stacked under WIRE/FINER**, not a standalone k1 win. |

**Not ranked (orthogonal, not decoder-backbone k1 levers):** E-NeRV (compress-time encoder — changes latent quality, not decoder spectral bias), VQ-NeRV / cool_chic / C3 (codebook/entropy — byte axis), coord_mlp_residual_sidecar / boost_nerv (residual bolt-ons on top of a base decoder).

### A.4 The two-tier conclusion of Part A

- **Tier-1 (drop-in, byte-neutral, zero-risk):** **FINER** activation on the LIVE HNeRV decoder. It is the single cheapest k1 lever — a one-line swap of `torch.sin(x)` → `sin((|x|+1)·x)` in `configurable_taper_decoder.py`, byte-identical, and it directly attacks the fixed-ω low-frequency spectral bias that is the *mechanistic root* of high k1. Already implemented in `siren/activation_family.py`.
- **Tier-1.5 (drop-in, byte-neutral):** **WIRE** Gabor activation — same drop-in surface, predicted lowest k1, slightly higher risk (the Gaussian window can suppress signal if `wire_scale` is mistuned; needs the scale swept).
- **Tier-2 (new backbone, byte-cost, highest ceiling):** **Wavelet decoder** — the basis-matched move; the sparsest representation of the argmax partition; sister to the Z8 detail-coeff RD work but applied at the *decoder* not the *codec*. Higher build cost; reserve for after Tier-1 measures.

---

## Part B — k1 SCREEN (QUEUED, not run — MPS busy)

**Why queued:** `launch_split_by_head_basin` (taper screen, PID 77644) holds MPS (`--train-device mps`). Per the no-contention directive, Part B is a plan. Per the existence-proof rule, the Part-A ranking is a *prediction* until these measured d_seg rows exist.

**The screen (run when MPS frees — verify `pgrep -f launch_split_by_head_basin` is empty first):**

1. **Baseline anchor (re-use, don't rebuild):** the n96 HNeRV bc20 row at ~73 KB from the live driver, seed 0. (May already exist from a5825bc; re-use its d_seg.)
2. **Arm A — FINER drop-in:** same `launch_split_by_head_basin` invocation, same `--base-channels 20 --latent-dim 28 --seed 0`, SAME byte budget, with a `--activation finer` flag wired into `configurable_taper_decoder.py`'s `sin` call (the activation is already in `tac.substrates.siren.activation_family.apply_activation_family`). Train MPS-gradient, **eval CPU** (authority).
3. **Arm B — WIRE drop-in:** same, `--activation wire`, sweep `wire_scale ∈ {0.5, 1.0, 2.0}` (the window scale is the one tunable; mistuned scale kills it).
4. **Arm C (vs the taper champion):** FINER **on top of** the concentrated-taper decoder from a5825bc (the two levers are orthogonal — test the stack).
5. **Metric:** exact CPU `d_seg` at fixed bytes via `tac.contest_score` (lowest d_seg = lowest k1). n96-short to rank, confirm the winner at n600-short.
6. **Falsification threshold:** a variant is "meaningfully lower-k1" only if measured `d_seg` ≤ 0.85 × baseline at equal bytes (≥ −15%, beyond seed noise). Below that bar → Part-A spectral-bias prediction is implementation-falsified for the short-budget regime (paradigm intact; re-test at long-train).

**Build delta needed for the screen:** wire an `--activation {siren,finer,wire}` flag from `launch_split_by_head_basin` → the decoder's activation call, delegating to the existing `apply_activation_family`. ~10 LOC, byte-neutral, no new architecture. (Not done in this $0 unit — it touches the live driver that the running screen uses; queue it behind the screen to avoid editing a file under an active run.)

---

## 5-lens joint review

- **Math/geometry:** d_seg is a codimension-1 high-freq sparse boundary; wavelet/Gabor bases are provably sparsest for it; fixed-ω sine has the documented low-freq bias. Ranking is grounded, not vibes. ✓
- **Existence-proof:** Part-A is explicitly a *prediction*; the verdict is gated on Part-B measured d_seg. No k1 claim is asserted as fact. ✓
- **NO-FAKE / means-vs-ends:** corrected the rosy framing — at the real operating point S=0.386 (d_seg-dominated), sub-0.15 needs 6.5× not −30%; the architecture axis is the best *single factor* on the dominant term, not a one-shot win. ✓
- **No-premature-kill:** the L0-sketch variants were killed on proxy/PSNR historically; this survey re-opens them ranked by k1-for-d_seg on the frozen scorer (paradigm intact, implementation-falsified). ✓
- **Pointer/authority:** advisory, NON-PROMOTABLE, pointer 0.19110 UNMOVED, $0, MPS not contended. ✓

## Wire-in hooks (Catalog #125)

1. Sensitivity-map: N/A (no new per-byte map; consumes existing d_seg map). 2. Pareto: ranking feeds the d_seg-vs-byte constraint (architecture is a new k1 axis). 3. Bit-allocator: N/A. 4. Cathedral autopilot: the ranked table is a candidate-priority input. 5. Continual-learning: this memo + Part-B rows (when measured). 6. Probe-disambiguator: Part-B IS the FINER-vs-WIRE-vs-baseline disambiguator.

## Verdict

**The architecture axis is a real k1 edge orthogonal to the taper** (taper = WHERE capacity sits; activation/basis = WHAT each param represents). The math says a fixed-ω sine decoder under-represents the sharp boundary, and there are **already-built, byte-closed** lower-k1 mechanisms in the repo (FINER, WIRE in `siren/activation_family.py`; wavelet decoder in `substrates/wavelet/`). But the corrected arithmetic shows it is a *factor*, not a finish — sub-0.15 needs the architecture × taper × byte × long-train compound.

**Single recommended next step:** wire the `--activation {siren,finer,wire}` flag into the live decoder and run Part-B Arm A/B/C as a byte-neutral k1 screen **the moment the taper screen (PID 77644) releases MPS** — FINER first (zero byte cost, attacks the spectral-bias root, ~10 LOC). It is the cheapest, lowest-risk test of the whole architecture-axis hypothesis.
