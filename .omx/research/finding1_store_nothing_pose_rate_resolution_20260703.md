# FINDING-1 (#256) RESOLUTION — the store_nothing_205 pose carrier true rate

**Date:** 2026-07-03
**Author:** research subagent (investigation only; no heavy job launched — pgrep-checked, memory healthy)
**Pointer:** 0.19110 UNMOVED (this is a rate-accounting correction + a byte-close-side fix, not an exact row)
**Verdict in one line:** the 52,135 B is REAL and CORRECT for what the serializer *currently ships*; **83 % of it (43,200 B) is a redundant per-pair fp64 homography H that is a deterministic FREE function of the already-stored twist ξ.** The claimed ~0.0007 pose rate is **RECOVERABLE** but was never actually byte-closed — it was the code's own idealized "byte-optimal" note, not the shipped path. **Redundant-H = YES.**

---

## 1. What is actually IN the 52,135 B section (MEASURED, not asserted)

The store-nothing pose carrier is serialized by `serialize_pose_carrier`
(`tools/levelset_byte_close_and_eval.py:468`). Layout:

```
PCAR1\x00                          6 B    magic
u32 hdr_len                        4 B
hdr_json                       1,721 B    (of which kf_of_pair=[0]*600 JSON list = 1,201 B — USELESS in store_nothing)
H  (P × 9 × fp64)              43,200 B    600 × 72  ← THE REDUNDANT PART (82.9 %)
xi (P × 6 × fp16)              7,200 B    600 × 12  ← the actual video-derived pose payload (13.8 %)
u32 n_kf (= 0)                     4 B    NO keyframes stored (store_nothing)
[keyframe blobs]                   0 B
                              ─────────
TOTAL                         52,135 B    rate = 25 × 52,135 / 37,545,489 = 0.03471
```

**This is not a guess.** I reconstructed the exact header JSON (matching
`serialize_pose_carrier` L480-487 + the `hdr_extra` at L646-658) with the live
job's params (`--pc-s-t 0.044 --pc-s-r 0.0 --pc-pitch 0.0`, stride 1) and summed
the section arithmetically — it reproduces **52,135 B to the byte**. The report
block in the tool itself already emits `H_bytes = P*9*8 = 43,200` and
`xi_bytes = P*6*2 = 7,200` (L681-682).

Per-pair: H = **72 B/pair**, xi = **12 B/pair**, header ≈ 2.9 B/pair.

---

## 2. Is H stored, or derived? → **STORED (redundantly).**

Traced in `build_pose_carrier_section` (`levelset_byte_close_and_eval.py:576`):

```python
xi = _wrl.xi_from_pose_calibration(gt_poses[p], s_t, s_r, pitch)   # fp64
xi_stack[p]  = xi                                                  # stored fp16 (L492)
H_stack[p]   = _wrl.homography_from_xi_numpy(xi, geom)             # stored fp64 (L491)
```

**Both** H and ξ are written to the section. The decode
(`pose_carrier_frame0_from_source`, L550) warps the witness's own render with
`pc["H"][pi]` — the STORED fp64 H, never re-deriving it. The code author was
explicit about this being redundant (L418-423, L707-718):

> "H is a deterministic function of xi (**byte-optimal design stores ONLY xi and
> derives H FREE**; H stored here is the decode-simplicity choice, ~72 B/pair,
> negligible vs the keyframe payload)."

That "negligible vs the keyframe payload" is the trap: it is true for
`warp_real_luma` (keyframe = tens of KB dominates). In **`store_nothing` the
keyframe payload is ZERO**, so the "negligible" 43,200 B becomes **82.9 % of the
whole section.** The redundancy assumption silently inverted when the keyframe
went to zero — nobody re-derived the byte budget for the store-nothing branch.

---

## 3. Is H derivable-free? → **YES, EXACTLY (confirmed against the real code + comma geometry).**

The deep-math hypothesis in the task is **CONFIRMED, not merely plausible.** The
implementation is the plane-induced (Hartley–Zisserman) homography:

`homography_from_xi_numpy(xi, geom)` (`warp_real_luma_frame0.py:199`):
1. `T = exp_se3(xi)` — SE(3) group element from the twist (generic Lie map, FREE).
2. `R = rotation_of(T)`, `t = translation_of(T)`.
3. `H = K (R − t·nᵀ/d) K⁻¹` (`homography_from_Rt`, L170-173).

Every non-ξ input is **FIXED and known at decode**:
- **K** = pinned EON intrinsics `fx=fy=910`, principal point from native 1164×874, scaled by `intrinsics_at` (L105). Camera calibration constant.
- **n** = `plane_normal(pitch) = [0, −cos p, −sin p]` (L120-125) — a function of the scalar `pitch` (stored in the 6-byte-ish header, not per-pair).
- **d** = `CAMERA_HEIGHT_M = 1.22` (L91) — openpilot camera height, constant.
- `geom = GroundHomographyGeom.eon(pitch=pitch)` (L151) — rebuilt at decode from the stored scalar `pitch` alone.

So `H[p] = f(ξ[p]; K, n(pitch), d)` with `K, d` global constants and `pitch` a
single stored scalar. **There is NO per-pair information in H beyond ξ.** H is a
generic decode-time algorithm on the counted ξ → **rule-118 FREE, 0 archive
bytes.** Storing it is pure redundancy.

**Rule-118 firewall (NO-FAKE):**
- COUNTED (archive.zip): the per-pair twist ξ (video-derived ego trajectory — genuinely irreducible) + the scalar `pitch/s_t/s_r`.
- FREE (inflate.py): `exp_se3` + `H = K(R−tnᵀ/d)K⁻¹` + inverse-warp bilinear + R. All generic algorithm on fixed camera calibration.

---

## 4. The TRUE store-nothing pose rate (measured / derived)

| Scenario | Section bytes | Rate `25·B/37.5M` | Notes |
|---|---:|---:|---|
| **AS-SHIPPED (the honest current number)** | **52,135** | **0.03471** | raw fp64 H + raw fp16 ξ + kf_of_pair waste; the ~0.0007 claim was FALSE for this packet |
| Derive-H, drop useless kf list, fp16 ξ | 7,736 | 0.00515 | **6.7× smaller, guaranteed rule-118 FREE, no coding, no re-train** |
| Derive-H + fp32 ξ (bit-exact-safe) | 14,936 | 0.00995 | if strict fp64-H parity is wanted (see §5) |
| Derive-H + coded ξ (temporal-Δ + AR) | ~1,049 | ~0.00070 | the CLAIMED number — achievable, needs the ξ entropy coder to land |

**The 50× gap decomposes as:** redundant H (43,200 B, saves **0.02877**) +
uncoded fp16 ξ (7,200 → ~1,049, saves ~0.00409) + the useless `kf_of_pair=[0]*600`
header list (1,201 B, saves ~0.0008). `52,135 / 1,049 = 49.7× ≈ 50×`.

**Reconciliation with R1 (#239 / #248):** R1's "store-nothing viable @ d_pose
0.0011, contribution 0.105" (commit `148636537`, #248) is the **DISTORTION**
axis (`√(10·0.0011)=0.105`), a *separate* result. The **~0.0007 / ~1,049 B RATE**
was the code's own *idealized* `byte_optimal_note` ("store ONLY xi... 12 B/pair"
→ coded ≈ 1 KB), **asserted from the design intent, never byte-closed through
`serialize_pose_carrier` (which always writes H).** FINDING-1's error was
assuming store_nothing_205's byte-close used that idealized path; it used the
redundant-H serializer → 0.03471. Neither number is a lie in isolation; they
were conflated. **The redundant H is real; 0.035 is NOT a genuine floor.**

---

## 5. The fix — BYTE-CLOSE-SIDE + inflate template (NO re-training)

**Change:** in `store_nothing` mode, do NOT serialize `H_stack`; derive
`H[p] = homography_from_xi_numpy(ξ[p], geom)` at decode from the stored ξ + the
scalar `pitch`. Also drop the `kf_of_pair` list from the header (it is unused
when `n_keyframes = 0`). Surfaces:
- `serialize_pose_carrier` / `parse_pose_carrier` / `build_pose_carrier_section` (`levelset_byte_close_and_eval.py`) — gate the H block off when `store_nothing`.
- `pose_carrier_frame0_from_source` (L550) — compute H from ξ instead of reading `pc["H"]`.
- the **shipped inflate template's** inlined warp — add `exp_se3` + `homography_from_Rt` (both generic; **0 archive bytes**; currently L420 boasts "no exp_se3 needed" — that is exactly the redundancy to remove).

This is **entirely post-training / eval-side.** #205 is a *training* run producing
witness weights + the trained rank-6 dξ residual; the pose-carrier serialization
is applied at byte-close, so training is untouched. In fact derive-H is the
**natural** design for the #205 residual path — the residual already operates in
ξ-space, so storing the (trained) ξ and deriving H is the aligned choice; the
fp64 H is a `warp_real_luma` vestige.

**d_pose-invariance check (NO-FAKE — the one real subtlety):** the *current*
packet warps with `H(fp64 ξ)` while storing ξ at fp16. Deriving H from the
stored fp16 ξ yields `H(fp16 ξ) ≠ H(fp64 ξ)` (fp16 gives ~1e-3 relative ξ
precision → sub-pixel warp shift → a handful of edge pixels flip after uint8).
So derive-H-from-fp16-ξ is **NOT bit-identical** to the current packet. Two clean
resolutions:
- **(a) byte-optimal (recommended):** make the fp16 ξ the *source of truth* at BOTH compress and decode — round ξ → fp16 first, derive H from it, warp with that H, and measure d_pose there. Compress and decode then use the identical fp16-ξ→H→warp → **d_pose-invariant by construction** and bit-exact run-to-run. Because store_nothing d_pose is currently in flux (4.97 pre-residual / 0.0011 post-R1-residual, and #205's trained residual re-measures it), the exact fp64-vs-fp16 pre-residual value is not a frozen target — a light **n48** re-measure at eval suffices (no heavy n600, route any launch through the governor).
- **(b) strict parity:** store ξ at fp32 (14,936 B, rate 0.00995) so `H(fp32 ξ) ≈ H(fp64 ξ)` to ~1e-7 → warp identical after uint8 → bit-exact d_pose, still 3.5× smaller than as-shipped.

Either way **d_pose is preserved / re-measured honestly; the fix is a pure rate
win.** MLX-GPU is NOT bit-identical cross-process, so any bit-exact proof stays
CPU-locked (per the standing discipline) — the warp/derive-H authority here is
numpy-fp64 CPU, which is fine.

---

## 6. Corrected sub-0.15 rate budget

- **pose-carrier rate = 0.0007** (coded ξ, the intended endpoint; needs the ξ temporal-Δ+AR coder) — RECOVERABLE, not free-as-shipped.
- **conservative guaranteed pose-carrier rate = 0.0051** (derive-H, fp16 ξ, no coder, pure rule-118, lands today with the serialization change alone).
- **as-shipped (buggy) = 0.0347** — would consume **23 % of the entire 0.15 budget** on the pose carrier alone. **UNACCEPTABLE to byte-close #205 as-is.**

The previously-CLAIMED "pose rate ≈ 0" was optimistic-by-one-fix: it is
**recoverable to ~0.0007–0.005**, but only after the derive-H (and, for the full
0.0007, the ξ-coder) change. Until then the honest line item is **0.0347**.

---

## 7. Recommendation for #205

**Launch #205 as-is** (training is independent of the pose-carrier serialization;
the trained ξ / dξ residual is what training produces). **But do NOT byte-close /
claim the rate off the current serializer** — it will land at 0.0347 and blow the
rate budget. Apply the **derive-H byte-close fix** (drop the 43,200 B H + the
1,201 B kf_of_pair list; derive H from stored ξ; make fp16 ξ the source of truth;
inline `exp_se3`+`homography_from_Rt` in the shipped inflate) *before* the
byte-close/rate claim. That recovers the pose rate to **~0.0051 guaranteed**
(→ ~0.0007 once the ξ coder lands), consistent with the sub-0.15 budget. This is
a ~0.03 rate-term win with zero d_pose cost (§5) and zero re-training.

**Council lens:** se(3)/geometry (Chasles + plane-induced homography — the
derivation is *exact*, not approximate) says H carries no per-pair info beyond ξ;
Shannon says the true payload is the entropy of a smooth 6-dof ego trajectory
(→ coded ξ ≈ 1 KB); the NO-FAKE/rule-118 firewall says derived-H is FREE and
stored-H is a counted redundancy. All three agree: **the 43,200 B is waste.**
