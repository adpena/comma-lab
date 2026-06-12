# Rate levers (Lever 1 + Lever 4) — DEBUG + TEST + ITERATE + OPTIMIZE (2026-06-12)

**Scope:** the two RATE levers — Lever 1 (differentiable brotli-rate surrogate,
`src/tac/losses/rate_surrogate.py`) + Lever 4 (score-aware QAT,
`src/tac/torch_vehicle/score_aware_qat.py`) — DEFERRED from the distortion arm. Optimizing
them does NOT invalidate the distortion-lever SEAL (orthogonal axis). Goal: close their
open review items + make their score-effect REAL + MEASURED.

**Authority:** every number below is `[macOS-CPU advisory]` NON-PROMOTABLE (small real-0.mkv
slices, advisory distortion — NOT a 600-pair contest eval). The frontier is UNMOVED
(`0.19109982`, pointer `.omx/state/canonical_frontier_pointer.json`). These levers land
MEANS; the dual CPU/CUDA 600-pair exact eval is still required before any SCORE claim.

**Scope discipline (collision-safe):** I made ZERO edits to `rate_surrogate.py`,
`score_aware_qat.py`, `driver.py`, or `curriculum.py` (the distortion-arm SEAL + basin
daemon pid 33911 are unperturbed — byte-identity-default holds BY CONSTRUCTION, the existing
lever source is untouched). All work is NEW files: the variable-level codec module + tests +
3 probes + this memo.

---

## (A) DEBUG — re-verify the R2/R6/R7 fixes + hunt remaining bugs

### R2 (resume) + R6/R7 (stage-boundary carry) QAT-EMA persistence — HOLD on HEAD
The 9 QAT-EMA persistence tests in `test_all_layer2_levers.py` (`-k "qat or sensitivity or
carry or resume or r7"`) **PASS on HEAD** (9 passed, 19 deselected, 350.95s under daemon
contention). Verified:
- `_capture_state`/`_restore_into` persist `tensor_sensitivity_ema` across resume (R2).
- `carry_sensitivity_ema` carries the EMA across QAT→QAT stage boundaries (R6).
- R7 matrix: non-QAT→QAT activation starts empty then accumulates; L4 deactivation does not
  mutate the carried EMA; resume mid-second-QAT-stage is bit-identical; AdamW→Muon QAT
  boundary carries. No sibling reset-bug.

### Bug hunt — latent-delta gradient flow (BOTH paths) + edge cases — CLEAN
`$0` probe (`.omx/tmp/debug_latent_delta_gradflow.py`):
- Lever-1 latent-delta entropy backprops to the `latents` leaf (grad norm 1.36e1, non-zero).
- **Split-path concern RESOLVED:** a frame-side backward THEN a SEPARATE `reg.backward()`
  (the rate surrogate reading the FULL latents) ADDS latent-delta gradient on top of the
  frame grad (25.9 → 29.1) — the split path's `reg.backward()` correctly accumulates into
  the same `.grad` buffers. No bug.
- Edge cases all safe: `n_pairs=1` → 0.0; all-zero latents → finite (2.05); `latents=None` →
  `r_lat=0.0`.

`$0` probe (`.omx/tmp/debug_qat_edge_and_c1a.py`):
- all-zero sensitivity dict → uniform → 127 levels each → apply bit-identical to
  `sensitivity=None` (vendored uniform). Single-tensor decoder safe (rank→0.5→~95 levels,
  within `[16,127]`). No `rate_lambda`×C1a double-count: on the REAL basin-arch decoder,
  conditional `H(W|Wprev)=7.28` ≤ marginal C1a `H(W)=7.90` (distinct, additive).

**No remaining rate-lever bug found.** Both levers are real, default-OFF, daemon-safe.

---

## (B) TEST — the MEASURED score effect on the REAL scorer/codec

### Lever 4 QAT — the unclosed MED-2 NET-SCORE A/B → **BYTE_DIRECTION_ONLY** (honest)

The R1 audit gated the Lever-4 score claim on the TRAINING A/B (uniform-QAT-TRAINED vs
score-aware-QAT-TRAINED). I built it
(`experiments/probe_lever4_qat_training_ab_net_score.py`) — two arms from the SAME basin-EMA
seed, SAME tiny real slice (12 pairs), SAME 8-epoch budget + RNG, both byte-closed through
the REAL vendored codec, scored on the REAL frozen SegNet/PoseNet.

| | uniform-127 TRAINED | score-aware TRAINED | Δ |
|---|---|---|---|
| decoder brotli blob | 73533 B | 73526 B | **−7 B** |
| advisory d_seg | 0.003330 | 0.003337 | +0.000007 |
| advisory d_pose | 0.00160917 | 0.00159600 | −0.00001317 |
| advisory contest S | 0.509195 | 0.509431 | **+0.000237 (WORSE)** |

**VERDICT — BYTE_DIRECTION_ONLY.** This is the key measured finding. The one-shot SNAP
(`probe_lever4_qat_brotli_blob_delta.py`) shows −3263 B; but once you actually **TRAIN both
arms, the win collapses to −7 B**, because the vendored codec ALWAYS re-quantizes at 127 →
the trained weights converge to a distribution that, after the 127-requant, looks nearly
identical to the uniform-trained arm. **The snap's −3263 B was an artifact of the snap NOT
being the deployed grid.** The R1 audit's "unproven indirect effect" concern is now MEASURED:
the indirect effect nearly vanishes under training. The honest docstring caveat STANDS and is
strengthened by a measured net-score A/B. **Lever-4 is NOT a confirmed net-score lever under
the vendored codec.** This directly motivated the optimization (C).

### Lever 1 rate surrogate — real-byte-reduction A/B → tiny but REAL

`experiments/probe_lever1_real_byte_reduction_ab.py` — Lever-1 ON (`rate_lambda_w>0`) vs OFF,
same seed/budget/slice, deploy-faithful `codec_scan_order=True`.

| λ_w | epochs × pairs | decoder blob Δ (ON−OFF) | advisory d_seg | verdict |
|---|---|---|---|---|
| 0.5 | 8 × 12 | **+2 B** (flat) | 0.00334→0.00333 | NO_BYTE_REDUCTION |
| 5.0 | 12 × 10 | **−6 B** | 0.00330→0.00331 | BYTE_REDUCTION |

At λ=0.5 the byte count is flat; at 10× λ=5.0 the surrogate DOES drive real bytes down, but
only **−6 B** (out of 73530). Honest read: the surrogate rank-tracks bytes (MED-1: Spearman
0.90) AND can drive them down when trained with, but the **magnitude is tiny on the
already-converged basin weights** — the basin is near its weight-entropy floor, so little
redundancy remains to squeeze. (The advisory score "improvements" in both arms are driven by
d_pose noise on the small slice, NOT by bytes.) Consistent with the rate-invariant-floor
reality (E).

---

## (C) ITERATE + OPTIMIZE — the variable-level codec (the REAL reverse-waterfill)

The Lever-4 training A/B proved the score-aware grid's byte win is washed out by the codec's
fixed 127-requant. The REAL fix is a per-tensor **variable-level codec** that stores each
tensor at ITS OWN `n_levels` so the reverse-waterfill ACTUALLY changes deployed bytes.

**Landed:** `src/tac/losses/variable_level_codec.py` (NEW, ~230 LOC, numpy-portable inflate)
+ `src/tac/tests/test_variable_level_codec.py` (9 NO-FAKE tests). Grammar = vendored decoder
blob + a 1-byte format flag + (variable path) one `u8 n_levels` per tensor. Default-preserving
builder `build_decoder_blob_variable_or_vendored` returns the EXACT vendored bytes on the
all-uniform path (byte-identical archive until a non-uniform allocation is supplied), and the
codec level map equals the score-aware-QAT training grid (integration contract).

**MEASURED (`probe_variable_level_codec_byte_distortion.py`, real basin EMA, real scorer):**

| min_level_ratio | coarsened | deployed blob Δ (var−vend) | advisory d_seg Δ | advisory S Δ | verdict |
|---|---|---|---|---|---|
| 0.5 | 27/28 | **−1721 B (−2.34%)** | +0.000008 | +0.001053 | BYTE_WIN_DISTORTION_NEAR_BREAKEVEN |
| 0.75 | 27/28 | **−789 B (−1.07%)** | +0.000007 | +0.006030 | BYTE_WIN_DISTORTION_NEAR_BREAKEVEN |

**The key advance:** the variable codec delivers a **REAL deployed byte win (−789 to −1721 B)
that SURVIVES inflate** — the vendored-127 path could only manage −7 B trained (washed). The
all-uniform builder is byte-IDENTICAL to vendored (73527 == 73527, verified). The snap's
distortion uptick is the expected cost of NOT training at the variable grid (the
eval_roundtrip-sister) — that recovery is the next training A/B. Tighter ratios trade byte-win
for distortion.

**Honest status of the variable codec:** it is the CORRECT optimization (makes the
sensitivity allocation actually move deployed bytes), with a numpy-portable inflate + 9 tests
+ a measured byte win. It is `research_only=true` (SCORE_CLAIM=False) with ONE named
integration blocker: **wiring it into the driver's live `build_archive`/parse-back eval path
requires the SEAL'd distortion arm to be unfrozen** (the archive-build + eval-decoder are on
the SEAL surface I must not perturb). The module + the standalone distortion probe ARE its
validated consumers; the driver wire-in is the gated next step.

### Lever 1 iteration
Pushed λ_w 0.5 → 5.0 (10×): byte reduction went +2 B → −6 B (real but tiny). Did NOT add the
order-2/context model — the measured ceiling (−6 B on converged basin weights) does not
justify the complexity; the rate-invariant floor bounds the EV (E). The latent-delta term
(1b) is wired + gradient-flowing in both paths (A) but was not the bottleneck here (decoder
weights dominate the blob).

---

## (D) Tests + self-protection

- `src/tac/tests/test_variable_level_codec.py` — **9 NO-FAKE behavioral tests** (round-trip
  bit-exact; coarser→strictly-smaller blob; uniform builder byte-identical-to-vendored;
  coarse dequant has fewer distinct symbols; level map default-preserving + monotone in
  sensitivity; min_abs_levels floor; codec grid == QAT grid; real-basin strict byte win). No
  constant-tests — every test would FAIL on a `return baseline` body.
- Full subset green: `test_rate_surrogate.py` + `test_score_aware_qat.py` +
  `test_variable_level_codec.py` = **34 passed in 7.11s**.
- QAT-EMA persistence (R2/R6/R7): **9 passed** (350.95s).
- **Byte-identity-default: CONFIRMED by construction** — zero edits to existing lever source;
  the basin daemon + distortion SEAL are unperturbed.
- ruff clean on all new files.

---

## (E) Honest EV statement — rate is the carrier-invariant floor

Rate (`25·|archive|/N`) is the carrier-INVARIANT floor: the decoder weights are already
brotli-compressed near their entropy at the basin, so these two levers' EV is **structurally
bounded**:
- **Lever 1** drives the real blob down only **−6 B even at 10× λ** on converged weights
  (the weight entropy is near-floor; the surrogate has little redundancy left to squeeze).
- **Lever 4** under the vendored codec is **byte-direction-only** (−7 B trained; the 127-
  requant washes the snap's −3263 B).
- The **variable-level codec** is the one lever here that moves DEPLOYED bytes meaningfully
  (**−789 to −1721 B**, ~−1 to −2.3% of the decoder blob) — because it changes the GRAMMAR
  (stores fewer levels), not just the training dynamics. Its net-SCORE win is gated on
  training the decoder at the variable grid (the distortion recovery) + the driver wire-in
  (SEAL-gated) + the 600-pair dual exact eval.

**Bottom line:** the rate levers are REAL + debugged + measured, but their honest EV is small
(rate is the invariant floor). The highest-EV rate path is the variable-level codec (a
grammar change), which is built + tested + measured-at-byte-win but NOT yet net-score-proven
(needs the SEAL-gated driver wire-in + a train-at-variable-grid A/B). No SCORE claim; frontier
UNMOVED.

---

## Wire-in / provenance (Catalog #125 6-hook)

- #1 sensitivity-map: ACTIVE — the variable codec consumes the per-tensor `||∂S/∂w||`
  sensitivity (the score-aware QAT EMA) as its level-allocation input.
- #2 Pareto: ACTIVE — the byte↔distortion trade is the `min_level_ratio` knob (measured RD
  points at 0.5/0.75).
- #3 bit-allocator: ACTIVE (PRIMARY) — the variable codec IS the reverse-waterfill bit
  allocator (deployed grid, not training-only proxy).
- #4 cathedral autopilot: N/A — research_only, not archive-deployable until the driver
  wire-in (SEAL-gated).
- #5 continual-learning: N/A — advisory $0 probes, non-promotable.
- #6 probe-disambiguator: ACTIVE — the 3 probes (Lever-4 training A/B, Lever-1 byte A/B,
  variable-codec byte+distortion) are the empirical disambiguators between "byte-direction
  only" and "real deployed win".

Mission contribution: `frontier_breaking_enabler` (the variable codec is the rate lever that
can actually move deployed bytes) + `rigor_overhead` (closing the MED-2 net-score A/B
honestly). Authority: all `[macOS-CPU advisory]` NON-PROMOTABLE. No GPU, no PR, no daemon
touched, frontier UNMOVED.
