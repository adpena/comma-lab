# Rate-attack + L3 finishing-kit + D1 latent-dedup verification (P1b)

**Authority:** `[contest-CPU advisory]` — all numbers are REAL **byte** measurements at the
POST-int8-brotli archive level on the converged base_ch=20 solved-taper basin. **NO score
claim** (score requires the dual CPU/CUDA 600-pair exact eval, G3). Pointer 0.19110 UNMOVED.
$0, CPU/code only. **No `driver.py` edit** (P1a owns it; export wiring deferred to P2).

Basin: `experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best`
(decoder 28 tensors, latents `(600, 28)`, meta archive_bytes 89,136; my full vendored
re-pack = 89,248 B → decoder blob 73,469 B (82%) + latents blob 15,574 B (17%) + ~205 B
meta/zip overhead).

Harness: `experiments/verify_rate_attack_l3_d1.py` (+ `test_verify_rate_attack_l3_d1.py`,
4 tests). JSON: `reports/rate_attack_l3_d1_verification.json`. All 61 codec/kit/harness
tests green; ruff clean.

---

## Part D — rate-attack (variable-level codec) — REAL byte deltas, MEASURED WIN

`tac.losses.variable_level_codec` is already built (carrier-agnostic core + default-preserving
adapter). Verified on the REAL solved-taper decoder state dict:

| metric | value |
|---|---|
| vendored uniform-127 decoder blob | **73,469 B** |
| variable-level (sensitivity-driven, ratio band 0.5–1.0) | **65,534 B** |
| **decoder-blob delta** | **−7,935 B (−10.8%)** |
| projected FULL-archive (R4: post-int8-brotli, not param-count) | 89,248 → **81,313 B (−8.9%)** |
| variable decoder round-trip exact (dequant weights) | ✅ max_abs_err 0.0 |
| default-preserving (all-uniform == vendored, byte-identical) | ✅ True |
| level span (28 tensors) | 64 … 127 |

**The byte-neutrality check is done at the ARCHIVE level (review R4), not param-count.** The
taper sets channels; the rate-attack sets per-tensor LEVELS; together they set the final
post-brotli archive bytes. The −7,935 B is a real deployed reduction in the largest section.

### CRITICAL honest caveat — this is a rate↔DISTORTION trade, NOT a free win
The variable-level codec coarsens low-sensitivity tensors' grids, which introduces weight
quant distortion. Measured weight-RMSE vs the uniform-127 baseline:

| allocation | decoder bytes | delta | weight-RMSE |
|---|---|---|---|
| uniform-127 (baseline) | 73,469 | — | 1.249e-3 |
| **conservative (0.75–1.0)** | 69,794 | **−3,675 B** | 1.541e-3 (+23%) |
| **aggressive (0.5–1.0)** | 65,534 | **−7,935 B** | 2.033e-3 (+63%) |

The SCORE effect of that weight distortion (how it maps to d_seg / d_pose) is UNMEASURED here
(needs the scorer → G3). **For P2:** the rate-attack must either (a) be **co-trained with QAT**
so the decoder is robust at the deployed coarse grid (the synergy the spec calls out:
taper ↔ QAT/Lever-4 ↔ rate-codec all allocate by the SAME sensitivity), or (b) be tuned to a
**score-optimal operating point** on the rate↔distortion curve — applying it blind on a
non-QAT'd decoder will pay bytes but may cost d_seg. Two operating points (conservative /
aggressive) are banked above as the P2 sweep endpoints.

### Allocation-spine caveat
This harness drove the allocation with a code-only **RMS-energy proxy** ($0, no scorer). The
production shared spine is the **gate-2 d_seg-sensitivity map** (Δd_seg/param, scorer-measured).
The BYTE mechanism is real and proxy-independent; the per-tensor ALLOCATION must be re-driven
by the real sensitivity map in P2 (the synergistic-core "one shared sensitivity input" wiring).

---

## Part E — L3 finishing-kit — byte cost confirmed, deterministic, POST-round

`tac.torch_vehicle.distortion_finishing_kit` primitives verified (module-level; NOT driver-wired):

| primitive | section bytes | determinism / round-trip | apply |
|---|---|---|---|
| disabled (default-OFF) | **0 B** | parses back to disabled | no-op (same object) |
| PR98 residual (converged ep2120) | **54 B** fixed | byte-identical serialize→parse→serialize | POST-round, uint8 in-range ✅ |
| T10 affine (scale≠1) | **54 B** fixed | scale+bias round-trip exact | POST-round, uint8 in-range ✅ |
| S12 certification | **+0 B over PR98** (1-bit flag) | flag round-trips | render base → certification-only |

All transforms apply `round(clip(scale·x − bias, 0, 255))` per (frame-parity, channel) on the
uint8 frames AFTER the eval round — 1:1 with the scorer's cast point. PR98 is the only kit that
survived the converged under-power audit (the shipped default fixture is residual-only PR98;
T10 stays a refit capability). **Byte cost: ≤54 B for the whole kit; 0 B when disabled.** The
FIT constants are `[contest-CPU advisory]` until a byte-closed exact-eval row lands. Ready for
P2 to wire into the export (apply POST-round + serialize the 54-B section).

---

## Part F — D1 cross-pair latent dedup — BUILT (already), candidate space EXHAUSTED on these latents

`tac.losses.cross_pair_latent_codec` is already built (measure-and-select: framed-delta /
exact-dedup / VQ-codebook candidates + default-preserving adapter, all round-trip the quantized
codes bit-exact). Verified on the REAL `(600, 28)` basin latents:

| metric | value |
|---|---|
| vendored latents blob (temporal-delta + lo/hi + brotli) | **15,574 B** |
| adapter emitted | 15,574 B (vendored — no candidate won) |
| **delta** | **0 B** (default-preserving, byte-identical) |
| deployed latents round-trip exact | ✅ max_abs_err 0.0 |
| exact-dedup: unique quantized rows | **600 / 600** (0 dups) |
| 1st-order-delta symbol-entropy floor | 15,243 B |
| **vendored above entropy floor** | **+2.17%** |
| SVD ranks for 99.9% energy | **28 / 28 (FULL-RANK)** |

### I measured the FULL D1 candidate space on the real latents (the spec's options):
| candidate | bytes vs vendored | verdict |
|---|---|---|
| exact dedup | n/a (0 dups) | dead |
| VQ codebook (K=64/128/256) | +overhead | loses (already in module) |
| **low-rank / PCA factorization** | full-rank (28/28 for 99.9%) | **no lossless win possible** |
| 2nd-order delta | +1,857 B | loses (latents not temporally smooth) |
| **AR(1) per-dim predictor** | −36 B (= brotli block-ALIGNMENT noise, not structural) | **NOT a real win** |

**VERDICT: D1's cross-pair-redundancy space is EXHAUSTED on these latents** — they sit +2.17%
above the symbol-entropy floor, are FULL-RANK, have zero exact dups, and are not temporally
smooth. No lossless cross-pair candidate (dedup / codebook / low-rank / AR) beats vendored. The
vendored 1st-order-delta + lo/hi + brotli is already near-optimal.

### NO-FAKE decision (recorded): I did NOT ship an AR(1) candidate
I prototyped + measured an AR(1) per-dim integer-predictor candidate. On the real latents it was
−36 B (pure brotli block-alignment noise, smaller than its own coefficient-table cost) and even
on GENUINELY AR(1)-structured synthetic data (a=0.5 random walk) it was **+3 B** — brotli already
adapts to the delta stream, so the explicit coefficient table costs as much as the predictor
saves. Shipping it as a "structural win" would be a FAKE win (alignment-noise masquerading as
cross-pair-redundancy exploitation, exactly the R8 instrument-bug the module guards against) and
it broke the module's honest-negative regression tests. **Reverted** — the existing 3-candidate
module is correct and complete; the honest negative is the real result. The harness now durably
records the structural-headroom verdict (entropy floor + SVD rank + dedup ratio) so the negative
is a measured regression guard, not prose.

D1 stays the biggest LATENT-term recode, but the latent term is only 17% of the archive and is
near-floor — **the rate lever with real headroom is the DECODER blob (Part D), 82% of the
archive, where the variable-level codec banks −7,935 B.**

---

## Export hooks for P2 (what's wireable into the production export)
1. **Rate-attack:** `variable_level_codec.build_decoder_blob_variable_or_vendored(sd, levels)`
   where `levels = levels_from_sensitivity_for_codec(<gate-2 d_seg sensitivity>, names)`.
   Persist the 1-bit format flag in archive meta; inflate dispatches
   `codec.decode_decoder` vs `decode_decoder_variable` (numpy-portable). **Co-train with QAT or
   tune the operating point — it's a rate↔distortion trade.** Endpoints: conservative −3,675 B
   (+23% RMSE) / aggressive −7,935 B (+63% RMSE).
2. **L3 kit:** `apply_distortion_kit_to_raw_frames(frames, cfg)` POST-round on uint8 +
   `serialize_distortion_section(cfg)` (≤54 B; 0 B disabled). Shipped default = converged
   residual PR98 (`DistortionKitConfig.from_converged_residual_pr98()`).
3. **D1 latents:** `build_latent_blob_dedup_or_vendored(latents)` — byte-identical on these
   latents (the measured negative); wire it anyway (costs nothing, flips automatically on Track-B
   latents with real cross-pair structure). Persist its 1-bit format flag for the inflate
   dispatch (`decode_latent_blob`).

All three are default-preserving (byte-identical until a non-trivial config/allocation is
supplied), so wiring them into the export changes NOTHING until P2 supplies the sensitivity
allocation / fit constants.

## 6-hook wire-in (Catalog #125)
- #1 sensitivity-map: ACTIVE (rate-attack consumes the per-tensor sensitivity spine; L3 the
  per-(frame,channel) color-bias fit).
- #2 Pareto: ACTIVE (rate-attack moves the rate vertex; the distortion cost is the constraint).
- #3 bit-allocator: ACTIVE (variable-level codec IS the per-tensor bit/level allocator).
- #4 cathedral autopilot: N/A (export-time codec, not a dispatch surface).
- #5 continual-learning: ACTIVE (the −7,935 B decoder win + the D1 EXHAUSTED verdict reseed the
  rate-lever priors: decoder blob is the headroom, latents are near-floor).
- #6 probe-disambiguator: ACTIVE (the harness IS the disambiguator: rate-attack pays −8.9%
  archive but at a distortion cost; D1 does not pay losslessly on these latents).
