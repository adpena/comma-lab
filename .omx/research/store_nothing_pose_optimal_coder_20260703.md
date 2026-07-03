# #257 — the truly-optimal store-nothing pose carrier serializer (derive-H + ξ entropy coder)

**Date:** 2026-07-03
**Author:** serializer/byte-close subagent (build only; #205 training UNTOUCHED — pgrep-checked live, n24/n600 measurements are numpy-only + a tiny-synthetic real inflate, all memory-safe)
**Pointer:** 0.19110 UNMOVED (this is a byte-close-side rate-recovery fix, not an exact-eval row)
**Follows:** `finding1_store_nothing_pose_rate_resolution_20260703.md` (#256) — which proved the shipped store-nothing pose section was 52,135 B, 83% of it a redundant fp64 H derivable free from ξ.
**Status:** DONE (real derive-H + real arithmetic coder + measured), wired + integrated + tested.

---

## 1. What changed (the 4 deliverables, all real + measured)

1. **DROP the redundant per-pair fp64 H block (43,200 B / 600 pairs).** store_nothing v2 stores ONLY
   the ego twist ξ; H is DERIVED FREE at decode (`_xip.homographies_from_xi`: `exp_se3(ξ)` → plane
   homography `K(R − t nᵀ/d)K⁻¹`). Rule-118: a generic decode-time algorithm on fixed camera
   calibration is FREE; only the video-derived ξ is COUNTED.
2. **DROP the `kf_of_pair=[0]*600` junk list** (1,201 B) + the whole warp_real_luma header cruft — the
   v2 store-nothing header carries only what derive-H needs (n_pairs, native_hw, pitch, s_t/s_r,
   xi_coder, xi_q_levels, 6 per-channel scales).
3. **Derive H(ξ) at DECODE.** Inlined VERBATIM into the shipped `inflate.py` (`_ar_decode` +
   `_xip_parse` + `_xip_H_from_xi`), op-for-op mirrors of `tac.boundary_math.xi_pose_coder` +
   `tac.lossless.range_coder.decode_static_symbols` — **bit-exact-gate proven** (shipped inflate ==
   numpy oracle, both coders).
4. **The ξ entropy coder.** Per-channel int16 fixed-point quantization → first-order temporal delta →
   transmitted-PMF static **arithmetic coder** (`tac.lossless.range_coder`, pure-stdlib, inlinable).
   A `--no-xi-coder` (raw int16) path is the guaranteed-today fallback + strict-parity reference.

New canonical module: **`src/tac/boundary_math/xi_pose_coder.py`** (reusable `tac.*` codec primitive:
`quantize_xi` / `dequantize_xi` / `homographies_from_xi` / `serialize_xi_payload` / `parse_xi_payload` /
`decode_xi_payload` / `xi_payload_rate_report`).

## 2. MEASURED recovered rate (the REAL byte-close path, `build_pose_carrier_section`)

n600, live #205 calibration (`--pc-s-t 0.044 --pc-s-r 0.0 --pc-pitch 0.0`):

| variant | pose section bytes | rate `25·B/37.5M` | vs pre-#257 |
|---|---:|---:|---:|
| **pre-#257 as-shipped** (fp64 H + fp16 ξ + kf_of_pair) | 52,135 | **0.03471** | 1.0× |
| **#257 raw ξ** (`--no-xi-coder`, guaranteed today) | 7,790 | **0.00519** | 6.7× smaller |
| **#257 coded ξ** (DEFAULT `delta_ar`) | 3,762 | **0.00250** | **13.9× smaller** |

H_bytes = **0** (derived free) for both v2 variants. The redundant-H + junk-list drop alone recovers
**0.0295** of rate (guaranteed, no coding); the coder recovers a further **0.0027**.

**Honest note on the ~0.0007 aspiration:** the coder LANDS and is real, but the measured floor is
~0.0025, NOT the finding's aspirational 0.0007. Diagnosis (measured): with `s_r=0` only the 3
translation channels carry bits, and the lateral/vertical channels are near-zero **ego-jitter** — the
jitter IS the entropy (consistent with the "lane-band rate crux corrected: jitter not swaps" memo).
2nd-order delta does not help (1944 B > 1875 B est). The residual ~0.0025 is a genuine ego-trajectory
content floor at fp16-scale precision. The flagged OPEN lever toward ~0.0007 is a **d_pose-budget-aware
per-channel step allocation** (coarsen the low-d_pose-impact jitter channels via a measured sensitivity
pass) — NOT applied/faked here.

## 3. d_pose-INVARIANCE (MEASURED, the "(b) UNCHANGED" gate)

n24, live #205 calib, through the ACTUAL v2 serialize→parse→derive-H→warp path (frozen CPU-torch
PoseNet authority, NEVER MPS):

| path | d_pose | score-term `√(10·d_pose)` |
|---|---:|---:|
| OLD pre-#257 (stored fp64 H) | 11.084379 | 10.528238 |
| **NEW #257 (coded ξ, DERIVED H, default q=4096)** | 11.084576 | 10.528331 |
| **Δ** | **+1.97e-04** | **+9.35e-05** |

**d_pose is INVARIANT** — the score-term moves by ~9e-5 (the ξ fp16-scale quantization perturbs only a
handful of vanishing-point-straddle horizon pixels; PoseNet's coarse ego readout is unmoved). q_levels
sweep (n24): 4096 → Δd_pose +2.0e-4 (tightest, chosen default); 2048 → +1.8e-3; 512 → +1.1e-2. All
score-negligible; 4096 is the sweet spot (fp16+ precision, ~0.0025 n600 rate).

**Strict internal consistency (bit-exact):** deriving H from the stored ξ reproduces EXACTLY the fp64 H
one would store from that same ξ (`homographies_from_xi` == per-pair `homography_from_xi_numpy`,
max|Δ|=0.0) → the realized uint8 render is bit-identical to a store-H-from-same-ξ path. The +1.97e-4 vs
the pre-#257 packet is purely the ξ quantization (fp64 ξ → fp16-scale ξ), measured + bounded above.

## 4. Wired + integrated + tested

**Wire-in points:**
- `tools/levelset_byte_close_and_eval.py`:
  - `serialize_pose_carrier_store_nothing()` — v2 serializer (no H, no kf_of_pair).
  - `parse_pose_carrier()` — discriminates `pcar_store_nothing_v==2` → decode ξ + DERIVE H.
  - `build_pose_carrier_section()` store_nothing branch → v2 serializer + report (H_bytes=0, xi_coder).
  - `_cap_pose_carrier()` — v2 re-serialize (bit-exact-gate capping).
  - `_INFLATE_PY` shipped template — inline `_ar_decode` + `_xip_parse` + `_xip_H_from_xi` + v2 `_pcar_parse`.
  - CLI: `--pc-xi-coder {delta_ar,none}` / `--no-xi-coder` / `--pc-xi-qlevels` → `pose_carrier_cfg`.
- `src/tac/boundary_math/xi_pose_coder.py` — the reusable authority (reuses `tac.lossless.range_coder`
  + `warp_real_luma_frame0` geometry).
- warp_real_luma is **BYTE-IDENTICAL** (only the store_nothing branch changed; A/B path preserved).

**Tests (65 pass):**
- `src/tac/tests/test_xi_pose_coder.py` (25): quantize round-trip, coder losslessness + strict-parity
  (P∈{1,2,5,48,128} × q∈{512,4096,32767}), zero-crossing / size-1 alphabet, derive-H == module
  authority bit-for-bit, render-unchanged, coder-beats-raw, rate accounting, NO-MPS-by-construction,
  fail-closed.
- `src/tac/tests/test_levelset_pose_carrier_byte_close.py` (+7): v2 serialize/parse derive-H, both-coder
  strict-parity identical frames, **shipped-inflate-string == tool oracle** (both coders), v2 cap, v2
  build-section rate accounting on real gt_n6, warp_real_luma byte-identity regression.
- `experiments/tests/test_levelset_byte_close_and_eval.py` (+2): **END-TO-END bit-exact gate** — the
  REAL shipped `inflate.py` subprocess (inline ξ decoder + derive-H) == numpy oracle, both coders
  (also fixed 2 pre-existing stale `_read_blob_bytes` 5-tuple unpackings from the #205 6th-block landing).

Preflight: `ruff --select F821` (the commit-ready authority) clean on all changed files.

## 5. Ready for #205's final byte-close?

**YES.** When #205's trained witness + rank-6 dξ residual land, byte-close with
`--pose-carrier --pose-carrier-mode store_nothing` (defaults: `--pc-xi-coder delta_ar --pc-xi-qlevels 4096`).
The pose section will be ~3,762 B (rate 0.00250) instead of the pre-#257 52,135 B (0.03471) — a **0.032
rate-term recovery with zero d_pose cost and zero re-training**, keeping the pose carrier at ~1.7% of
the 0.15 budget instead of 23%. The `--no-xi-coder` raw path (0.00519) is the guaranteed fallback if a
strict raw-ξ reference is wanted. d_pose is measured through the REAL decode at byte-close (the derived
H reproduces the fp64-H warp; the trained residual's actual d_pose is #205's own measurement — no
borrowed number).

**Council lens (all agree):** se(3)/geometry — H carries no per-pair info beyond ξ (derivation exact,
not approximate); Shannon — the true payload is the entropy of the smooth 6-dof ego trajectory (coded to
~0.0025, residual = lateral-jitter content floor); NO-FAKE/rule-118 — derived-H is FREE, the 43,200 B
was counted redundancy now gone.
