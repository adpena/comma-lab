# Next-gen INR activations: survey + 3 custom designs for the argmax-EDGE d_seg objective

**Date:** 2026-06-24 · **Authority:** `[analysis]` / `[advisory]` NON-PROMOTABLE · pointer UNMOVED 0.19110 (NO score move claimed) · **Budget:** $0 research + build + stage only (NO training run launched; the n600 MPS confirm owns the GPU).

## TL;DR

Screened the next-gen INR-activation literature for the ONE objective that actually scores us: the frozen-SegNet **argmax STEP at lane boundaries**, scored on **argmax-flip d_seg through the uint8 round-trip** — NOT generic-image PSNR. Implemented **8 new activations** (5 fixed-form literature + 1 custom hybrid fixed-form + 2 custom learnable) into `activation_family.py` with the SAME NO-FAKE parity discipline (real math; siren bit-identical; tests fail on a no-op swap; every hyperparameter sensitivity-asserted). Threaded all 8 through `--activation` + `TorchVehicleConfig` + the decoder. **38 new NO-FAKE tests, all green; ruff clean.** Staged (NOT run) the apples-to-apples d_seg screen ranked by expected edge-fit.

**Ranked hypothesis (expected d_seg-fit for the argmax STEP):**

| rank | activation | why it might beat FINER on the ARGMAX EDGE | byte cost |
|---|---|---|---|
| **1** | **hosc** `tanh(β·sin(x))` | β→∞ turns the sine into a **square-wave / step train** — the native piecewise-constant carrier for an argmax partition; suppresses the Gibbs off-edge tail that flips distant cells | **byte-neutral** |
| **2** | **step_basis** (custom #2) `Σ aₖ·tanh(gₖ(x-cₖ))` | natively a **sum of soft Heaviside STEPS** = the exact shape of the SegNet argmax partition; flat between steps (NO ringing); learns edge locations | +3K params (~12 B) |
| **3** | **finer_gauss** (custom #1) `sin(ω(|x|+1)x)·exp(-½(s·x)²)` | keeps FINER's CONFIRMED −18.7 % variable-frequency edge win, **adds** a spatial envelope to damp the off-edge tail → composes the win with localization (the thing WIRE failed to add) | byte-neutral |
| **4** | **fkan** (custom #3) `Σ[aₖ sin(kωx)+bₖ cos(kωx)]` | "learn our own nonlinearity": optimizer concentrates harmonics in the exact lane-edge band; inits to the SIREN basis (inherits the working prior) | +2K params (~10 B) |
| 5 | **gauss** `exp(-(s·x)²)` | smooth bump, localized in BOTH space and frequency, **no Gibbs ringing at all** — a step is a sum of shifted bumps; but a single bump is low-pass (may under-fit the sharp edge) | byte-neutral |
| 6 | **rcgauss** (FLAIR) `sinc·RC·gauss` | band-localized under the time-frequency uncertainty bound; designed for sharp band cutoffs; heavier/oscillatory → spend last | byte-neutral |
| 7 | **sinc** `sin(x)/x` | ideal band-limited interpolation kernel; but oscillatory tails → expect ringing like SIREN | byte-neutral |
| — | wire (CONTROL, already screened) | **NULL** (0.98×). Space-localization ALONE didn't help → see existence-proof cross-check | byte-neutral |

---

## The frame (do NOT chase PSNR-SOTA)

Confirmed prior screen (bc20/n100/3000ep, byte-closed CPU authority): **SIREN(control) d_seg 0.001692; FINER `sin((|x|+1)x)` 0.001376 = −18.7 % (REAL lower-floor win, both plateaued, distinct asymptotes); WIRE(Gabor) 0.98× = NULL.** Source: `.omx/research/finer_wire_architecture_kone_screen_20260623.md`, `manifold_topology_dseg_deep_synthesis_20260623.md`, `island_representation_level_intrinsic_dim_20260624.md`.

The d_seg-binding source is a ~8-dim NONLINEAR manifold of high-frequency LANE-MARKING edges. The contest scorer derives masks from frames, slices the LAST frame, resizes to 512×384 with a **stride-2 EfficientNet-B2 stem** (loses half resolution immediately), and computes **argmax disagreement**. Two consequences for activation choice:

1. **Only the argmax FLIP matters.** A tiny logit perturbation at a class boundary is the ENTIRE signal; smooth PSNR gains far from boundaries are invisible. The ideal representation is **piecewise-constant with sharp transitions exactly at the lane edges** — i.e. a STEP function, not a smooth sinusoid.
2. **It must survive the uint8→argmax round-trip.** A representation that rings (Gibbs) near the step deposits oscillation into NEIGHBORING cells; after uint8 quantization those can flip argmax → spurious d_seg. Sinusoids ring on steps. **An edge/step-matched nonlinearity may beat FINER.**

This is why the ranking is **step-matched first** (hosc, step_basis), **localized-FINER second** (finer_gauss), and the pure oscillatory band-limited kernels (sinc, rcgauss) last.

---

## Candidate table — exact formulas + init (verbatim from the literature)

| activation | exact form φ(x) | key hyperparams | init | source |
|---|---|---|---|---|
| SIREN (control) | `sin(ω x)` | ω | Sitzmann uniform `U(−√(6/n)/ω, +)` (downstream), `U(−1/n,1/n)` (first) | Sitzmann 2020 |
| FINER (win) | `sin(ω(|x|+1)x)` | ω, bias `b∼U(−k,k)` k large | SIREN init + large-range bias | Liu 2024, arXiv 2312.02434 |
| WIRE (null) | `exp(−(s x)²+iωx)` (real: `sin(ωx)·exp(−½(s x)²)`) | s (width), ω | complex Gabor | Saragadam 2023 |
| **Gauss** | `exp(−(s x)²)` | s (width) | Ramasinghe init (sensitive) | Ramasinghe-Lucey 2022, arXiv 2111.15135 |
| **HOSC** | `tanh(β·sin(ω x))` | β (saturation/sharpness); Ada-Hosc learns β; β increases across layers | SIREN-style; β schedule | Serrano 2024 ("Hyperbolic Oscillating", arXiv 2601.07870) |
| **Sinc** | `sinc(ω x)=sin(ωx)/(ωx)` | ω | sampling-theory motivated | Saratchandran 2024 |
| **RC-GAUSS** (FLAIR) | `sinc(ωx)·rc(ωx;β)·exp(−½σ⁻²x²)`; rc = raised-cosine (roll-off β=0.05 fixed), σ learnable | ω (center freq), σ (Gaussian width), β=0.05 | band-localized; learnable σ | Lee 2025, FLAIR arXiv 2508.13544 (the "Band-Localized Activation" / RC-GAUSS) |
| FKAN (basis) | `Σ_{k=1..K}[aₖ sin(kx)+bₖ cos(kx)]` (1st-layer learnable Fourier series; K=270 in paper) | K (grid), aₖ,bₖ learnable | uniform; hidden ω₀=30 | FKAN arXiv 2409.09323 |
| SineKAN (basis) | `Σ A·sin(ωx+φ)` learnable A,ω,φ | per-edge A,ω,φ learnable | re-weighted sine grids | Reinhardt-Gleyzer 2024, arXiv 2407.04149 |

**Not implemented as drop-in activations (encoding-level, not a per-neuron nonlinearity):** CAFE / Fourier-Chebyshev content-aware (arXiv 2603.01028 — parallel-Fourier-features + Hadamard + Chebyshev *input encoding*, not an activation φ(x)); OptiINR / "globally optimal configuration" (arXiv 2509.23139 — a hyperparameter-configuration meta-method, not a new φ). These are deferred: they change the input-feature map, not the `_act` nonlinearity, so they don't fit the byte-neutral `--activation` drop-in screen. Flag for a separate input-encoding lane if the activation screen plateaus.

---

## The 3 custom designs (math + rationale)

### Custom #1 — `finer_gauss` (FINER × Gaussian envelope) [fixed-form, byte-neutral]
```
φ(x) = sin(ω(|x|+1)x) · exp(−½(s·x)²)
```
**Why:** FINER's CONFIRMED −18.7 % win comes from the variable local frequency `ω(|x|+1)` that rises near large |x| (edges). But pure FINER is a *global* sinusoid → its high-frequency lobes extend across the whole field, depositing off-edge oscillation that can flip distant argmax cells through uint8. Multiplying by a Gaussian envelope `exp(−½(s·x)²)` keeps FINER's edge-adaptivity where |x| is moderate while damping the far tail — **the localization WIRE tried to add, but applied to the variable-frequency carrier that actually wins** (WIRE wrapped a FIXED-frequency sine, which was already a null). This is the single most direct "compose the FINER win with localization" hypothesis.

### Custom #2 — `step_basis` (learnable soft-step sum) [learnable, +3K params]
```
φ(x) = Σ_{k=1..K} aₖ · tanh(gₖ·(x − cₖ))     (K=4 default)
```
Learnable: amplitudes `aₖ`, gains/sharpness `gₖ`, shifts/edge-locations `cₖ` (3K params total, shared across the decoder's 3 activation sites). **Why:** the SegNet argmax label field IS a piecewise-constant partition — locally flat class regions separated by sharp boundaries. A sum of `tanh` soft-Heaviside steps is **natively that shape**: each term saturates to ±aₖ away from its center cₖ (flat plateau, derivative→0, **NO Gibbs ringing**) with a sharp transition of width ~1/gₖ exactly at cₖ. The optimizer learns where the edges are (cₖ) and how sharp (gₖ). This is the most architecturally-matched-to-the-objective candidate: it represents a STEP without paying the off-edge oscillation tax that every sinusoid pays. (Init: amps 1/K, gains 1, shifts spread over [−1.5,1.5] — a gentle, well-conditioned non-trivial step-sum from epoch 0; the NO-FAKE tests assert it differs from siren and each of a/g/c moves the output.)

### Custom #3 — `fkan` (learnable Fourier-series activation) [learnable, +2K params]
```
φ(x) = Σ_{k=1..K} [aₖ·sin(kωx) + bₖ·cos(kωx)]     (K=5 default, ω=1)
```
Learnable per-harmonic `aₖ, bₖ` (2K params, shared). **Why:** "learn our own nonlinearity" — instead of SIREN's single fixed frequency, let the optimizer concentrate energy in exactly the harmonic band the lane-marking edges occupy (FKAN arXiv 2409.09323 / SineKAN arXiv 2407.04149). **Init a₁=1, rest 0 ⇒ φ(x)=sin(ωx) = the SIREN basis at epoch 0** — so it inherits SIREN's proven spectral prior and well-conditioned start, then learns to add the higher harmonics a step needs (a square wave is `Σ sin((2k−1)ωx)/(2k−1)` — exactly this basis). Lower-ranked than step_basis because the Fourier basis still rings on a step (finite harmonics → Gibbs), but it is the most flexible and can in principle discover the optimal edge spectrum.

---

## Implemented + parity-tested

**Module:** `src/tac/substrates/siren/activation_family.py` (extended).
- Fixed-form (stateless, byte-neutral, in `apply_activation_family`): **gauss, hosc, sinc, rcgauss, finer_gauss** + the legacy siren/finer/wire/bacon. New kwargs `hosc_beta`, `rcgauss_rolloff` (defaults leave legacy paths bit-identical).
- Learnable (real `nn.Module`): **`LearnableStepBasis`** (3K params), **`FourierKANActivation`** (2K params); built via `make_learnable_activation(...)`; `is_learnable_activation(...)` gate.
- Aliases added (gaussian→gauss, flair/rc-gauss→rcgauss, cardinal_sine→sinc, ada_hosc→hosc, sinekan→fkan, soft-step→step_basis, finer-gauss→finer_gauss).

**Decoder wire-in:** `src/tac/torch_vehicle/configurable_taper_decoder.py` — `__init__` gained `hosc_beta`/`step_basis_k`/`fkan_k`; one SHARED learnable sub-activation (`self._learnable_act`) applied at all 3 `_act` sites; fixed-form route through `apply_activation_family` unchanged. **siren stays bit-identical to vendored** (verified by the existing parity test, still green).

**Driver + launcher:** `TorchVehicleConfig` gained `hosc_beta`/`step_basis_k`/`fkan_k` (threaded into `_new_vendored_decoder`); `experiments/launch_split_by_head_basin.py` `--activation` choices expanded to all 11 + new `--hosc-beta`/`--step-basis-k`/`--fkan-k` flags.

**Measured param counts (bc20, ld28, vendored taper) — confirms byte budget:**
```
siren/gauss/hosc/sinc/rcgauss/finer_gauss = 83356 (byte-IDENTICAL = byte-neutral screen)
step_basis(K=4) = 83368  (+12 params ≈ +12 int8 bytes, negligible vs ~83KB)
fkan(K=5)       = 83366  (+10 params ≈ +10 int8 bytes, negligible)
```

**NO-FAKE tests (38 new, all green; ruff clean):**
- `src/tac/substrates/siren/tests/test_nextgen_activations.py` (NEW, 26 tests): exact-form verification (gauss==exp(−(sx)²), hosc==tanh(β sin x), sinc==sin/x, finer_gauss==FINER·gauss); each non-siren differs from siren; wire_scale & hosc_beta sensitivity; step_basis quasi-constant-between-steps gradient test (the no-Gibbs property); every learnable param moves the output (no dead params); fkan inits to sin(x); grad-flow; fail-closed on invalid sizes/families; aliases.
- `src/tac/torch_vehicle/tests/test_configurable_taper_decoder.py` (extended, +9 tests): each new fixed-form family changes the DECODER output vs siren at same weights & is byte-neutral & in [0,255]; hosc_beta modulates the decoder; learnable families add EXACTLY 3K/2K params & their params are in the state_dict & change output; fkan decoder inits ≈ siren; driver builds learnable-activation decoders.
- `src/tac/substrates/siren/tests/test_activation_family.py` (updated): registry/partition assertions for the expanded id set; SIREN coordinate-MLP fans over fixed-form families only.

---

## Staged d_seg screen (DO NOT RUN until the MPS frees — post n600 confirm)

Apples-to-apples with the FINER/SIREN n100 baseline. **Falsification bar:** a real win = `d_seg ≤ 0.85× SIREN`; to be worth adopting OVER FINER it must beat FINER's `0.813× SIREN` (i.e. `d_seg ≤ ~0.001376`, better still `< 0.001`). Run ranked top-down; stop early on a clear plateau-beater.

Common args (mirror the confirmed baseline): `--no-split-by-head --train-device mps --device cpu --base-channels 20 --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 --muon-lr-floor-fix --eval-every 50`

```bash
# rank 1 — HOSC step-carrier (sweep β; β=8 is the sharpest/step-iest)
.venv/bin/python experiments/launch_split_by_head_basin.py \
  --no-split-by-head --train-device mps --device cpu --base-channels 20 \
  --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 \
  --muon-lr-floor-fix --eval-every 50 \
  --activation hosc --hosc-beta 8.0 --out-dir experiments/results/act_screen_hosc_b8_n100

# rank 2 — learnable step-basis (native argmax partition; sweep K∈{4,8})
.venv/bin/python experiments/launch_split_by_head_basin.py \
  --no-split-by-head --train-device mps --device cpu --base-channels 20 \
  --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 \
  --muon-lr-floor-fix --eval-every 50 \
  --activation step_basis --step-basis-k 4 --out-dir experiments/results/act_screen_stepbasis_k4_n100

# rank 3 — finer_gauss (compose the FINER win with localization; sweep s via --wire-scale)
.venv/bin/python experiments/launch_split_by_head_basin.py \
  --no-split-by-head --train-device mps --device cpu --base-channels 20 \
  --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 \
  --muon-lr-floor-fix --eval-every 50 \
  --activation finer_gauss --wire-scale 1.0 --out-dir experiments/results/act_screen_finergauss_s1_n100

# rank 4 — fkan (learn the edge spectrum; sweep K∈{5,9})
.venv/bin/python experiments/launch_split_by_head_basin.py \
  --no-split-by-head --train-device mps --device cpu --base-channels 20 \
  --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 \
  --muon-lr-floor-fix --eval-every 50 \
  --activation fkan --fkan-k 5 --out-dir experiments/results/act_screen_fkan_k5_n100

# rank 5/6/7 — gauss / rcgauss / sinc (byte-neutral; run if 1-4 plateau)
.venv/bin/python experiments/launch_split_by_head_basin.py --no-split-by-head --train-device mps --device cpu --base-channels 20 --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 --muon-lr-floor-fix --eval-every 50 --activation gauss   --wire-scale 1.0 --out-dir experiments/results/act_screen_gauss_s1_n100
.venv/bin/python experiments/launch_split_by_head_basin.py --no-split-by-head --train-device mps --device cpu --base-channels 20 --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 --muon-lr-floor-fix --eval-every 50 --activation rcgauss --wire-scale 1.0 --out-dir experiments/results/act_screen_rcgauss_s1_n100
.venv/bin/python experiments/launch_split_by_head_basin.py --no-split-by-head --train-device mps --device cpu --base-channels 20 --latent-dim 28 --n-pairs 100 --total-epoch-budget 3000 --seed 0 --muon-lr-floor-fix --eval-every 50 --activation sinc                    --out-dir experiments/results/act_screen_sinc_n100
```

**CODEC NOTE (learnable families):** step_basis/fkan add `_learnable_act.*` keys to the decoder state_dict. The d_seg *training/eval* screen runs as-is (the driver builds the decoder directly). A byte-CLOSED archive for the learnable families needs the vendored codec to serialize the extra ~10-12 params (a trivial sidecar). For the SCREEN this is a non-blocker (the d_seg signal comes from the trained-decoder forward, +12 bytes is < 0.0001 on the rate axis). Flag for a codec extension only IF a learnable family wins the screen.

---

## 5-lens deep-math review (esp. NTK/spectral — which spectrum matches the lane edge)

**Lens 1 — NTK / spectral bias.** SIREN's NTK has a band-pass kernel centered at ω; it learns frequencies near ω fast and others slowly (spectral bias). A lane EDGE is a STEP, whose Fourier transform is `~1/f` (energy at ALL frequencies, heavy in the high band). So the ideal activation's effective kernel should be BROADBAND with a high-frequency emphasis. Ranking by this lens: **hosc** (a saturated sine → its harmonics are the odd-harmonic square-wave series `Σ1/(2k−1)` = the literal step spectrum) and **step_basis** (a soft step = the step spectrum by construction) have the right kernel; **fkan** can *learn* it (broadens the NTK band per-harmonic); **gauss** is LOW-PASS (NTK ~ Gaussian in frequency → under-fits the edge's high band — explains why a single bump may underperform); **sinc** is an ideal LOW-PASS brick-wall (band-limited → cannot represent the step's content above ω → predicted weak, consistent with its rank). This lens predicts hosc/step_basis > fkan > finer_gauss > gauss/sinc — matching the table.

**Lens 2 — Gibbs / uint8-survival.** A step approximated by a band-limited oscillatory basis (sinc, SIREN, FINER, fkan) overshoots ~9 % at the discontinuity and rings into neighbors. After uint8 quantization + argmax, ring lobes near a class boundary FLIP cells → spurious d_seg. **tanh-saturating bases (hosc, step_basis) have NO overshoot** (monotone step, derivative→0 in the tail) → the cleanest argmax survival. `finer_gauss` partially mitigates (envelope damps the ring tail). This is the lens that most strongly favors hosc/step_basis and most clearly explains the FINER ceiling (FINER still rings).

**Lens 3 — capacity at fixed bytes.** Fixed-form families are byte-IDENTICAL → a pure architecture A/B (the cleanest disambiguation of spectral-bias vs raw-capacity). Learnable families add ≤12 params (negligible) → still essentially a fixed-byte test. No capacity confound. The screen therefore isolates "does a step-matched nonlinearity lower d_seg at the SAME budget" — exactly the open question from the prior screen.

**Lens 4 — composition with FINER (the confirmed win).** FINER's win is variable-frequency. `finer_gauss` is the literal composition (FINER carrier × localization). `hosc` and `step_basis` are ALTERNATIVE mechanisms (step-matching, not frequency-adaptivity) — if one of them ALSO beats FINER, the next move is to compose them (e.g. a FINER carrier inside a HOSC saturation, or a step_basis whose centers are FINER-modulated) — a 2nd-order lever the screen's results will direct.

**Lens 5 — optimization / conditioning.** HOSC at large β has saturating gradients (tanh) → can stall; mitigation = β SCHEDULE (small→large across training/layers, per the paper) — our fixed β is a first cut, β-anneal is the follow-up if hosc shows promise-then-stall. step_basis with large gₖ also saturates → init gₖ=1 (gentle) and let it sharpen. fkan inits to sin(x) (SIREN conditioning) → safest start of the learnable pair. gauss is init-sensitive (Ramasinghe) → the wire_scale knob is the conditioning control. These conditioning notes are staged into the sweep (β∈{2,4,8}, K∈{4,8}/{5,9}, s via --wire-scale).

---

## Existence-proof cross-check (per "terminal-conclusion-needs-existence-proof" + "WIRE was null")

**The WIRE null is the key prior negative to reconcile.** WIRE = `sin(ωx)·exp(−½(sx)²)` adds SPACE-localization to a FIXED-frequency sine and was NULL (0.98×). Naive reading: "localization doesn't help." **Correct reading (the cross-check):** WIRE localized the WRONG carrier. The win (FINER) is in the *variable-frequency* `ω(|x|+1)x`, NOT the fixed `ωx` WIRE wraps. So:

- **What's different about finer_gauss vs WIRE:** it localizes the WINNING carrier (FINER), not the null one (fixed sine). If finer_gauss > FINER, that PROVES localization helps *when applied to the right carrier* and dissolves the WIRE null as a carrier-choice artifact. If finer_gauss ≈ FINER, localization is genuinely neutral and the FINER win is purely the frequency profile (a clean result either way).
- **What's different about hosc/step_basis vs WIRE:** they are NOT localization at all — they are STEP-MATCHING (changing the function's *shape* from oscillatory to piecewise-constant). WIRE's null says nothing about step-matching; this is an UNTESTED axis. The argmax objective's piecewise-constant target is the existence-proof argument that step-matching *should* help where space-localization didn't.
- **Existence-proof guard against a false floor:** before concluding "FINER is the activation floor," note that no STEP-shaped (vs oscillatory) activation has been screened yet. hosc/step_basis are exactly that missing class. A floor claim is premature until they're measured — this is the gap the staged screen fills.

---

## Wire-in / probe-outcome

- **Probe-outcome:** STAGED (not measured). The 8 activations are implemented, parity-tested, and CLI-threaded; the d_seg screen is ready to fire when the MPS frees (post n600 confirm). Falsification bar: `d_seg ≤ 0.85× SIREN` (win) / `≤ 0.813× SIREN` (beats FINER).
- **6-hook status:** (1) sensitivity-map — N/A (activation is architectural, not a per-byte axis); (2) Pareto — the byte-neutral families add 0 rate; learnable add ≤12 B (a Pareto-free lever); (3) bit-allocator — N/A; (4) cathedral autopilot — the `--activation` screen feeds the d_seg-lever ranker via the screen results; (5) continual-learning — this memo + probe row; (6) probe-disambiguator — the screen IS the disambiguator (spectral-bias/edge-fit vs raw-capacity), the falsification bar is the verdict rule.
- **Authority:** `[analysis]`/`[advisory]` NON-PROMOTABLE. Pointer UNMOVED 0.19110. NO score move claimed.

## Sources
- INR survey: https://arxiv.org/html/2411.03688v1 (exact formulas for siren/finer/wire/gauss/sinc/hosc/trident)
- FINER: https://arxiv.org/pdf/2312.02434 · HOSC: https://arxiv.org/html/2601.07870v1 · Gauss (Ramasinghe-Lucey): arXiv 2111.15135
- FLAIR (RC-GAUSS / band-localized): https://arxiv.org/abs/2508.13544 · Fourier-Chebyshev CAFE: https://arxiv.org/abs/2603.01028
- FKAN: https://arxiv.org/abs/2409.09323 · SineKAN: https://arxiv.org/html/2407.04149v1 · OptiINR/global-config: https://arxiv.org/html/2509.23139v1
