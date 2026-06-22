# Optimal pose-carrier (saliency-confined) — RESULT: saliency confines cheaper AT CONVERGENCE (1.16–2.44×) but converges slower; pose wall closed ~900–2200×; #57 overflow REFUTED (2026-06-21)

**Task #163 (action 2):** build + MEASURE the math-optimal witness pose-carrier and decide the decisive
question — does the PoseNet-Jacobian-saliency-weighted objective confine carrier capacity to the pose-relevant
support CHEAPER than the dense free-INR? **Answer (HONEST, mixed-but-net-positive): YES at convergence (PTNC
beats dense 2.44× @ N=6 e45, 1.16× @ N=12 e72), but the saliency objective converges SLOWER so it LOSES at an
underconverged budget (N=12 e36, dense 2.13× better), and the advantage SHRINKS as N grows (2.44× → 1.16×).
Both modes close the pose wall ~900–2200× vs the pose-blind palette (d_pose 12.658 → ~0.006).**

**Authority:** `[contest-CPU advisory]`, NON-PROMOTABLE. **Pointer UNMOVED 0.19110. NO score claim.** This is a
MEANS — a mechanism de-risk of witness gap #3 (pose closure), NOT a pointer-mover. The advisory S is NOT a
winner (L13's seg-term dominates); presenting it as progress would be a means-as-ends fake. The ONLY claim:
the saliency confinement closes the pose axis at lower byte cost than the dense INR, isolating d_seg (the live
Muon-stage-8 run's job) as the sole remaining witness wall.

---

## 1. What was built (REUSE, not rewrite — operator search-and-familiarize honored)
The grounding memo's §4 optimal build (`#57 amortized coordinate-INR × #61 measured saliency-weighted objective,
round-trip-threaded, clean float64 forward + clean parity, real byte-close`) was **ALREADY fully implemented** in
`tools/ptnc_train_pose_carrier.py` (#61 PTNC trainer) over the two canonical modules:
- `src/tac/boundary_math/amortized_luma_carrier.py` (#57): the coordinate-INR arch (shared Fourier basis Φ +
  per-pair FiLM modulation), `numpy_reference_forward` (float64), `carrier_frame`, `quantize_params` +
  `measure_carrier_bytes` (brotli-q11 real byte-close), `CarrierByteAccount`.
- `src/tac/boundary_math/posenet_jacobian_saliency.py` (#61): `compute_posenet_pixel_saliency` (exact backprop
  Jacobian-norm of the FROZEN CPU PoseNet through the differentiable yuv6 patch), `saliency_to_weight_map`,
  `identity_weight_map` (the uniform control).

`ptnc_train_pose_carrier.py` already wires ALL of §4: `--anchor-mode {ptnc,dense,identity}` (saliency vs
free-INR, the falsifiable comparison), `_eval_roundtrip` (uint8-round STE, round-trip-threaded), the
differentiable yuv6 patch active during pose backprop + saliency measurement, `_verify_parity`
(torch-oracle-vs-clean-numpy within 1 LSB — the CORRECT clean-reference parity invariant), `measure_carrier_bytes`
(real brotli byte-close), and `_exact_measure` (exact PoseNet d_pose on the numpy-decoded frame). **No new
trainer was needed.** This subagent's contribution = RUN the decisive comparison + verify the alleged #57 defect
+ record the verdict. (Two `.py` files were created only as thin analysis helpers; the optimal build is the
existing tool.)

## 2. THE DECISIVE COMPARISON (saliency-confined vs free-INR at equal bytes, swept across budget)
frame0 pose-carrier (SegNet-invisible slot), same arch (n_fourier=24, hidden=64, n_hidden=3, mod=24, 8-bit;
~25.5K params, ~22.5–23.8 KB brotli-q11), same warm anchor→pose schedule. **ONLY the input-domain anchor weight
differs** (the non-rename guarantee — the pose objective is identical exact-PoseNet MSE across modes; only WHERE
recon capacity is spent changes: dense = uniform, ptnc = measured Jacobian saliency). Bytes are config-determined
(same arch) so the comparison is genuinely equal-budget. d_pose is the EXACT frozen-CPU-PoseNet MSE on the
numpy-decoded uint8 frame, through the round-trip.

| budget | mode | exact mean d_pose | carrier bytes | winner | ratio |
|---|---|---|---|---|---|
| N=6  e45 | PTNC (saliency) | **0.005751** | 23,646 | **PTNC** | dense/ptnc **2.44×** |
| N=6  e45 | DENSE (free-INR) | 0.014058 | 23,799 | | |
| N=12 e36 | PTNC (saliency) | 0.023610 | 23,499 | DENSE | ptnc/dense 2.13× (PTNC **underconverged**) |
| N=12 e36 | DENSE (free-INR) | 0.011059 | 23,394 | | |
| N=12 e72 | PTNC (saliency) | **0.006118** | 22,582 | **PTNC** | dense/ptnc **1.16×** |
| N=12 e72 | DENSE (free-INR) | 0.007094 | 22,910 | | |
| — | L13 palette (pose-blind) | 12.658 | — | — | — |

**Headline (honest, mixed-but-net-positive):** saliency confinement wins **AT CONVERGENCE** (2.44× @ N=6 e45;
1.16× @ N=12 e72) but the saliency-weighted objective **converges SLOWER** — at the underconverged N=12 e36 it
LOSES (dense 2.13× better) because it front-loads capacity on the sparse pose tube while the bulk frame is still
mismatched; doubling epochs (e36→e72) flips it back to a win (0.0236→0.0061). The advantage also **shrinks as N
grows** (2.44× at N=6 → 1.16× at N=12) — at higher pair counts the fixed-size shared basis is the binding
constraint, not where capacity is spent. **Net verdict: saliency DOES confine cheaper, but the win is modest and
budget-/N-dependent, not the universal 2.44× a single point would suggest.** Both modes vastly close the pose
wall: ~900–2200× better than the pose-blind palette (Jacobian saliency field is real + concentrated: median
1.9e-6, max 1.6e-3, ~870× spread, nonzero_fraction 0.77 → ~25× weight concentration at floor 0.02).

## 3. The #57 OVERFLOW / PARITY claim — REFUTED (honest correction of the grounding memo §3)
The grounding memo §3 alleged #57's numpy forward overflows (matmul OVERFLOW → NaN/inf) and the manifest's
`all_match=True` was a parse-back-vs-same-overflow false parity (Catalog #304/#307 defect). **This does NOT
reproduce.** `numpy_reference_forward` is ALREADY float64-guarded (`amortized_luma_carrier.py:126-141`:
`coords/fourier_B/mod/params` all cast to float64 under `np.errstate(over='ignore')`). The parity gate
`_verify_parity` is the CORRECT clean invariant: **clean float64 numpy `carrier_frame` vs the torch training
oracle**, NOT parse-back-vs-same-forward. Measured: `rgb_within_1lsb_frac_min = 1.0` on EVERY run (smoke N=4,
PTNC N=6, DENSE N=6) — 100% of pixels agree within 1 LSB after rounding. No overflow, no NaN/inf, clean parity.
**Correction (append-only):** the §3 overflow defect is FALSIFIED at the current module state — the float64 guard
the memo §4(a) prescribed is already present, and the parity is already the clean-reference invariant §4(b)
demanded. The action-2-probe overflow (if it ever occurred) was on a pre-guard fp32 path no longer in the module.

## 4. Round-trip + byte-close (the NO-FAKE realizability)
- **Round-trip-threaded:** the carrier is fit + scored THROUGH `_eval_roundtrip` (uint8-round STE) then exact
  PoseNet (resize→yuv6 half-res-luma→6-dim). d_pose is the exact frozen-CPU-PoseNet MSE on the numpy-decoded
  uint8 frame — the carrier luma survives the round-trip to land the pose. NO MPS.
- **Real byte-close (Catalog #304):** bytes = brotli-q11 of the ACTUAL quantized weights (23,472 B) + per-pair
  mod (148 B) + fp16 dequant scales (26 B) = 23,646 B. The numpy `carrier_frame` decodes from the dequantized
  params (the inflate-time path), bit-consistent with the parity gate. No phantom bytes.
- **Saliency gradient intact:** measured under `patch_upstream_yuv6_globally()` (#61 fails-closed if the field is
  all-zero = severed gradient); the field is non-degenerate (nonzero_fraction 0.77, max 1.6e-3) → gradient live.

## 5. Honest framing (NO-FAKE — means/ends firewall)
- **Pointer UNMOVED 0.19110. NO score claim.** `[contest-CPU advisory]`, non-promotable.
- The advisory S is NOT a winner: this carrier closes ONLY the pose axis. L13's seg-term (100·d_seg ≈ 0.68 at
  d_seg 0.0068) dominates; a frame0-pose-carrier does not touch d_seg (frame0 is SegNet-invisible). Reporting a
  "good S" would be a means-as-ends fake.
- **What this IS:** a mechanism de-risk. Witness gap #3 (pose closure, palette → d_pose 12.658) is closed to
  d_pose ~0.006 at ~22.5 KB carrier (~1800–2200× better than palette), AND the saliency-confined objective is
  shown to confine cheaper at convergence (1.16–2.44×, budget-dependent). This isolates d_seg (the generator
  power-law / Muon-stage-8 campaign on the live run) as the sole remaining witness wall — exactly the CAPSTONE §9
  convergent verdict.
- **Caveat (honest scope, the real verdict):** the 2.44× saliency win is NOT universal — it is a CONVERGENCE
  property that shrinks with N (2.44× @ N=6 → 1.16× @ N=12) and INVERTS at an underconverged budget (N=12 e36,
  dense 2.13× better) because the saliency objective converges slower. The robust claim is the WEAK one:
  saliency confinement is a small, real, convergence-dependent improvement, not a dramatic class-shift on the
  pose axis. The DRAMATIC win is the same for both modes: the amortized coordinate-INR carrier (saliency or not)
  closes the palette pose wall by ~3 orders of magnitude — that mechanism, not the saliency refinement, is the
  load-bearing pose-closure result. N≤12 (advisory, ≤48 OK); a full 600-pair / Muon-stage run would lower BOTH
  modes further and is the natural next probe to see if the saliency edge holds, vanishes, or grows at scale.

## 6. 6-hook wire-in declaration
- **#1 sensitivity-map:** ACTIVE — the measured PoseNet pixel-Jacobian saliency field IS a per-pixel pose
  sensitivity map (`compute_posenet_pixel_saliency`); the result confirms it is load-bearing but modestly
  (1.16–2.44× gain vs uniform, at convergence; converges slower).
- **#2 Pareto:** ACTIVE — d_pose-per-byte is the carrier's RD point; saliency dominates dense on the (bytes,
  d_pose) Pareto front AT CONVERGENCE (lower d_pose at equal/lower bytes for N=6 e45 + N=12 e72), but not at
  underconverged budgets.
- **#3 bit-allocator:** ACTIVE (prior) — the saliency field is the canonical pose-axis allocation prior (paint
  pose-relevant support, free the pose-null).
- **#4 cathedral autopilot dispatch:** N/A — advisory non-promotable; no archive-deployable contest row (the
  carrier is byte-closed but not integrated into a full L13 contest archive yet).
- **#5 continual-learning posterior:** N/A — `[contest-CPU advisory]` non-promotable; no exact-eval anchor.
- **#6 probe-disambiguator:** ACTIVE — the `--anchor-mode {ptnc,dense,identity}` switch IS the built-in
  disambiguator between the two interpretations (saliency-confines vs dense-suffices); this run resolves it to
  "saliency-confines-cheaper-at-convergence (1.16–2.44×) but-converges-slower" — a budget-conditional verdict,
  not a clean win.

## NO-FAKE ledger
- MEASURED (this turn, 6 byte-closed runs): saliency PTNC vs free-INR dense, equal arch (~25.5K params,
  ~22.5–23.8 KB), exact frozen-CPU-PoseNet d_pose, clean parity 1.0 every run:
  N=6 e45 ptnc 0.005751 < dense 0.014058 (2.44×, ptnc wins);
  N=12 e36 ptnc 0.023610 > dense 0.011059 (2.13×, dense wins — ptnc underconverged);
  N=12 e72 ptnc 0.006118 < dense 0.007094 (1.16×, ptnc wins). #57 numpy forward does NOT overflow; parity is
  the clean-reference-vs-oracle invariant (rgb_within_1lsb_frac_min=1.0) → FALSIFIES the grounding-memo §3
  overflow claim at current module state.
- DERIVED: the rank-≤6 sparse-saliency pose channel → saliency-weighted objective confines capacity to the
  pose-relevant support → lower d_pose-per-byte at convergence. Confirmed empirically but the effect is small
  (≤2.44×) and slower-converging; it is NOT the dramatic mechanism — the amortized-INR carrier itself (either
  mode) closing the palette pose wall ~900–2200× is the load-bearing pose-closure result.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; this is the pose-MECHANISM closure, not a contest
  archive; the saliency edge is NOT proven robust at full-N/Muon scale; the binding witness wall is d_seg
  (separate live-run campaign).

## Artifacts
- `experiments/results/witness_L13_optimal_pose_carrier_20260621/ptnc_n6_e45/ptnc_train_result.json`
- `experiments/results/witness_L13_optimal_pose_carrier_20260621/dense_n6_e45/ptnc_train_result.json`
- `experiments/results/witness_L13_optimal_pose_carrier_20260621/comparison_summary.json` (this memo's table)

## Cross-references
- `optimal_pose_carrier_deep_math_grounding_20260621.md` (§4 the spec this realizes; §3 overflow claim REFUTED here).
- `CAPSTONE_witness_taskspace_roundtrip_byte_floor_formulation_20260621.md` (§9 — pose is OFF the critical path
  once the vehicle-matched mechanism byte-closes; THIS is that byte-close + the saliency-cheaper proof).
- `src/tac/boundary_math/amortized_luma_carrier.py` (#57) + `posenet_jacobian_saliency.py` (#61) + `tools/ptnc_train_pose_carrier.py`.
