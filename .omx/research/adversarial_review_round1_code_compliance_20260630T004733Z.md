# Adversarial Review Round 1 — LENS C (CODE/STACK + COMPLIANCE)

- **Subject:** the v2 witness program — deterministic byte-close arm (agent commits `f1ead4fd5` / `f94c304f0` / `fd8330c60` / `c4d2e1102`).
- **Lens:** code/stack bugs, missing wiring, byte-close/inflate contract, contest compliance.
- **Mode:** READ-ONLY ($0, no GPU, no MPS authority). FIND + PROPOSE; no edits.
- **Posture:** operator is "almost positive there are bugs / things missing." I assumed there are and hunted.
- **Pointer:** UNMOVED at contest-CPU 0.19110. None of this is a frontier claim. means ≠ ends.
- **Load-bearing file:** `experiments/v2_witness_byteclose_smoke.py` (366 LOC; builder + embedded `INFLATE_PY`).

---

## TL;DR (DAG-FEED)

**FEED (Lens C, round 1):** The d_pose=190 catastrophe has a precise, MEASURED root cause: the
"per-class pose warp" feeds PoseNet-scale pose scalars (dim0≈33.6, dims1-5≈1e-3) through an arbitrary
`×2.0` linear map + `clip(±6)`, which **saturates to a CONSTANT `dy=6, dx=0` for 100% of the 24 pairs** —
the warp is pose-INDEPENDENT (a fixed 6px circular roll of a vertical road band), carries zero per-pair
ego-motion → PoseNet distortion explodes. **[CRITICAL C1]** Separately, the "v2 6-section codec"
(shared-canonical + SDF carrier + entropy coder + pose codec + movables) is **NOT BUILT** — the byte-close
is a PHASE-1 de-risk that stores PER-PAIR content; the SDF/level-set/carrier modules are wired into
**nothing** (grep empty) **[MEDIUM M1]**, and the "solved" pose sidecar (`scorer_targets.py`) is unused
**[MEDIUM M2]**. Compliance is CLEAN: inflate.py ships no scorer weights, all video-derived bytes live in
`archive.zip`, byte counting matches `evaluate.py:63`. Apparatus is GENUINELY validated (store_raw → 0/0).
Parity holds (builder/decoder warps identical). Disk-hygiene non-negotiable is violated **[MEDIUM M3]**;
inflate.sh has no dep-closure **[MEDIUM M4]**.

---

## CRITICAL

### C1 — d_pose=190: the "pose warp" is pose-INDEPENDENT (units mismatch → 100% clip/round saturation)
**File:** `experiments/v2_witness_byteclose_smoke.py:91-98` (builder `warp_frame0_to_pred1`), duplicated
verbatim in the generated decoder at `:228-234`.

```python
fwd = float(pose6[0])           # :91   pose dim 0
lat = float(pose6[4])           # :92   pose dim 4
dy = int(round(np.clip(fwd * 2.0, -6, 6)))   # :93
dx = int(round(np.clip(lat * 2.0, -6, 6)))   # :94
road = f0[sky_end:hood_start]                # :95
shifted = np.roll(np.roll(road, dy, axis=0), dx, axis=1)   # :97
```

**MEASURED root cause** (`gt_n24.npz['gt_poses']`, float64, n=24):
- pose dim0 (used as `fwd`): mean **33.63**, std 0.60, range [32.07, 34.68] — a velocity-scale quantity,
  **not a pixel displacement**. `clip(round(33.63 × 2.0), -6, 6) = clip(67, ±6) = 6` → **`dy = 6` for ALL 24 pairs** (verified: `np.unique(dy) == [6]`).
- pose dim4 (used as `lat`): mean **-0.0008**, std 0.005. `round(-0.0008 × 2.0) = 0` → **`dx = 0` for ALL 24 pairs** (verified: `np.unique(dx) == [-0.]`).
- Pairs receiving the **identical** warp: **24 of 24**.

So the "per-class pose warp" is a **constant 6-pixel vertical circular roll of the road band, applied
identically to every pair**, with the pose's real per-pair signal (the 0.6 std in dim0 and the ~1e-3 rate
signals in dims 1-5) entirely destroyed by the ceiling/rounding. It cannot reproduce ego-motion; PoseNet
reads a frame-pair with no coherent motion → `d_pose = 190` (frontier ~3.4e-5). The `×2.0` scale and the
`±6` clip are magic numbers calibrated against **nothing** — they assume the stored pose is a small pixel
shift, but the stored pose is PoseNet's metric/normalized output (dim0 ~33.6, dims1-5 ~1e-3). There is a
**fundamental units mismatch**: metric pose → pixel flow requires camera intrinsics (focal, principal
point) + a depth/ground-plane model, none of which is present.

**Why d_pose is correctly isolated:** `store_raw` (exact f1) → d_pose=0; `store_jpeg q40` (lossy but
faithful frames) → d_pose=0.013. So faithful frames score fine — the warp is the specific failure, exactly
as the RESULT json's own `what_the_corner_actually_requires` and the grok-test (FEED-ja, stratified
per-class warp) state.

**Parity note (not a bug):** builder (`np.clip`) and decoder (`min/max`) warps are mathematically
identical, so inflate output byte-equals the oracle. The bug is in warp **semantics**, not determinism —
the wrong frame is reproduced *faithfully*.

**Proposed fix (build-gated, per the design):** replace the linear-roll warp with a real ground-plane
homography: `pixel_flow = K · [R(pose) | t(pose)] · K⁻¹` on the road stratum using comma2k19 intrinsics
(`K` = focal + principal point), with the correct per-stratum model from FEED-ja (Road = ground
homography, hood = identity, sky = rotation-only), and use edge-replicate (not circular `np.roll`) fill.
Verify the pose→dim semantics first (see L1). **Until a units-correct homography exists, the deterministic
warp arm cannot carry d_pose and is not shippable** — and the in-repo grep finds **no** real SE(3)/
homography warp to call (the `screw_warp_through_R` / `clean_canonical_warp` result dirs exist but no
generating warp function is present in `src/tac` or `experiments/`).

**Confidence:** HIGH (measured: 24/24 pairs get identical `dy=6,dx=0`; matches the RESULT json's own
diagnosis and the d_pose=190 datum).

---

## MEDIUM

### M1 — The "v2 6-section codec" is DESIGN, not BUILT; per-pair storage is the rate-dead cause (not a law)
**Files:** `experiments/v2_witness_byteclose_smoke.py` (only modes: `store_raw`, `v2_det`, `v2_warp`,
`store_jpeg`); SDF/carrier modules `src/tac/boundary_math/{lane_sdf_component,hood_static_component,
lever_b_levelset_generator,amortized_luma_carrier,context_partition_codec,contour_codec}.py`.

The MEMORY "v2 vehicle" is *store-canonical (one/few SHARED keyframes) + per-class-pose-warp + SDF carrier
+ entropy coder + pose codec + movables + integer-decode*. The built byte-close implements **none of the
shared-canonical / SDF / coder / movables pieces**. `grep -rln` for any of the six carrier/SDF/coder
modules across `experiments/`, `tools/`, `submissions/`, `src/` (excluding tests/`__init__`) returns
**empty** — they are wired into **no** inflate or byte-close path.

Concretely, "store-canonical" is implemented as **per-pair lossless `f0` storage** (`:120`, `:141`,
`:160` — each pair zlibs its own full `f0`). That is *why* rate is 150–450× over budget. The RESULT json
generalizes this to "per-pair pixel storage in ANY form cannot get within 100× of budget" (`verdict.
clears_0p19_at_n600`) — but that conclusion is drawn from a build that **stores a fresh keyframe per
pair**, which is the opposite of the designed shared-canonical. The rate-dead result is a property of the
**unbuilt-vehicle scaffold**, not evidence about the designed vehicle. The honest framing (which the json's
`honest_scope` does give, to its credit) is: PHASE-1 de-risked the apparatus; the shared-canonical vehicle
is UNBUILT and UNMEASURED.

**Proposed fix:** label the byte-close explicitly as an apparatus-validation scaffold (it largely does),
and do not let "per-pair storage is rate-dead" stand as a finding about the shared-canonical design. The
next build must (a) store ONE/few shared canonical keyframes, (b) wire an actual carrier/coder module, (c)
add the pose codec + movables. **Confidence:** HIGH (grep empty; modes enumerate the gap).

### M2 — The "solved" pose sidecar (`scorer_targets.py`, #140) is unused; v2_warp re-introduces pose collapse
**Files:** `src/tac/scorer_targets.py` (extract/save/load PoseNet 6-scalar targets, 7,200 B raw);
`experiments/v2_witness_byteclose_smoke.py` (stores `gt_poses` raw float32 at `:141`/`:161`, never the
PoseNet targets).

CLAUDE.md "Pose is SOLVED" says: store the 6 PoseNet OUTPUT scalars and supervised-condition the render to
hit them. The byte-close instead stores the *input* pose and tries to **geometrically reconstruct** the
pose-readable frame (v2_warp) — the exact "amortized-luma-CARRIER pose collapse" anti-pattern the design
warned against, and it collapses identically (d_pose=190). The actually-solved path (`scorer_targets.py`)
is imported by nothing in this arm.

**Compliance caveat (latent, not active):** `scorer_targets.py:12-13` + module docstring describe using the
targets "at inflate time [to] optimize the postfilter to match these exact targets." Optimizing a
postfilter to match PoseNet targets *at inflate* implies running PoseNet at decode → would violate the
no-scorer-at-inflate rule (README:118). It is currently **unused**, so this is not an active violation, but
if it is ever wired into inflate the supervised conditioning MUST happen at COMPRESS time only. Flag for the
next build. **Confidence:** HIGH (unused, by grep); MEDIUM on the latent-compliance reading of the docstring.

### M3 — Disk-hygiene non-negotiable violated: render oracle written with no certify-or-block / auto-clean
**File:** `experiments/v2_witness_byteclose_smoke.py:341-342` (`np.save(sub / f"render_oracle_n{n}.npy",
render)`).

CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup" is a HIGHEST-EMPHASIS non-negotiable: every new tool that
can create large artifacts MUST include an automatic disk-hygiene path (certify-or-block / move-to-SSD /
context-managed temp). The oracle is `2n·874·1164·3` bytes uint8 — **~146 MB at n=24, ~3.66 GB at n=600** —
written into `submission_dir` with no cleanup hook, no SSD-spill, no manifest. It is also written into the
SAME directory as `archive.zip` (it is correctly NOT inside the zip, so not shipped/counted — see compliance
✓ below — but it bloats local disk per run, and a careless "zip the submission_dir" would leak it).

**Proposed fix:** write the oracle to a context-managed temp / SSD tier (`/Volumes/VertigoDataTier/pact`
per the waterfall) OR gate it behind `--write-oracle`, and add a success-only auto-clean. At minimum keep
it OUT of `submission_dir`. **Confidence:** HIGH (explicit non-negotiable; sizes computed).

### M4 — inflate.sh has no dependency closure (numpy required; PIL for store_jpeg); JPEG cross-host determinism risk
**Files:** `INFLATE_SH` (`:305-310`), `INFLATE_PY` (`import numpy`; conditional `from PIL import Image` at
`:276`).

`inflate.sh` is bare `python3 inflate.py "$@"` with no `uv`/pip bootstrap. inflate.py hard-requires `numpy`
(all modes) and `PIL` (store_jpeg). HNeRV parity lesson 9 (runtime closure): a missing dep is a runtime
blocker, not a method result. The contest CPU runtime (README: 4×CPU/16GB) likely has numpy but PIL is not
guaranteed. For the shippable corner (v2_det) only numpy is needed, but the store_jpeg probe would fail a
clean-env replay.

**Additional, sharper risk — deterministic-decode non-negotiable:** store_jpeg's parity oracle is
`decode(encode(frame))` computed at BUILD time on macOS libjpeg; inflate re-decodes the SAME JPEG bytes at
eval time. JPEG IDCT can differ across libjpeg versions/platforms (macOS build vs Linux contest), so
inflate output may NOT be bit-identical to the oracle across hosts — violating CLAUDE.md deterministic-decode
#5 ("same archive.zip → bit-identical inflate output every run/host"). (This does not affect the score on
the eval host, but it breaks host-portable reproducibility and the oracle-parity guarantee.)

**Proposed fix:** declare/verify the dep set in inflate.sh (or document the contest runtime provides numpy);
avoid JPEG (or any platform-variant codec) in any shippable path; for lossy coding use a deterministic
integer codec (range/ANS) whose decode is bit-exact across hosts. **Confidence:** MEDIUM (numpy likely
present; PIL + JPEG-IDCT portability is a genuine, under-acknowledged risk).

---

## LOW

- **L1 — unverified pose-dim semantics.** `:91-92` assume `pose[0]=forward`, `pose[4]=lateral`. Measured
  dim0≈33.6 (velocity-scale), dims1-5≈1e-3. The index→DOF mapping is never verified against the PoseNet
  output convention; a real homography must first confirm which dims are rotation vs translation. Folds
  into the C1 fix. Confidence: MEDIUM.
- **L2 — dead code in inflate.py store_jpeg.** `:285` computes `f0=out[0::2]; f1=out[1::2]` which are never
  used (the `:288 if mode != "store_jpeg"` rebuild is skipped, and `out` is already correct from `:284`).
  Harmless, confusing. Confidence: HIGH.
- **L3 — circular-roll seam artifact.** `:97` uses `np.roll` (CIRCULAR) on the road band — content rolling
  off the band wraps to the opposite edge, creating a discontinuity at the band seams (rows 174 / 681).
  Even with a correct shift magnitude this is wrong; use edge-replicate/fill. Subsumed by the C1 homography
  rewrite. Confidence: HIGH (it is circular by definition of np.roll).
- **L4 — misleading rate labels in build_meta.** `:345` `rate_smalln = zsize / RATE_DENOM` divides the
  small-n archive by the FULL n600 denominator; `:357-359` project n600 bytes by linear per-pair scaling.
  Both are reasonable but the `rate_smalln`/`S_smalln` fields read like scores; they are advisory small-n
  numbers. The RESULT json flags this correctly; the in-file meta does not. Confidence: HIGH.

---

## COMPLIANCE VERDICT (the hard checks)

1. **NO scorer weights at inflate — PASS.** The generated `INFLATE_PY` imports only
   `argparse, io, json, struct, zlib, numpy` (+ conditional `PIL`). No `segnet`/`posenet`/`.safetensors`/
   `DistortionNet`/`load_state_dict`. No 37/53 MB model ship. The only files in `archive.zip` are
   `witness.bin` (`:335`). ✓
2. **rule-118 (no video-derived data smuggled as "code") — PASS.** All video-derived bytes (`f0`, `poses`,
   `resid`, `jpegs`) live in `witness.bin` INSIDE `archive.zip` and ARE counted; `inflate.py` is generic
   algorithm (parse + warp). No per-frame table is hidden in code. ✓ (Latent caveat: M2 `scorer_targets`
   if ever wired.)
3. **byte counting — PASS.** Builder reports `archive.zip.stat().st_size` (`:344`); `evaluate.py:63` counts
   exactly `(submission_dir/'archive.zip').stat().st_size`. The oracle `.npy` is NOT in the zip → not
   counted (but see M3 for disk hygiene). ✓
4. **FEED-kw independence lines — HOLDING (no PR95/HNeRV-reskin drift).**
   - geometric-warp-NOT-learned-decoder: ✓ (warp is np.roll integer ops; zero learned weights).
   - task-space-objective-NOT-RGB-PSNR: N/A in this scaffold (it is lossless storage / geometric warp, no
     training objective at all) — no drift, but also not yet a task-space witness.
   - tiny-residual-only-NOT-full-frame-HNeRV: **drifting in spirit** — `v2_det` ships a FULL-FRAME residual
     `(n,H,W,3)` (`:131`), not a tiny one. This is the rate-dead cause, acknowledged as build-gated, and it
     is NOT an HNeRV (no neural net), so no NO-FAKE #7 reskin — but the "tiny residual" promise is unmet
     until the shared-canonical + lane-survival-only residual is built.

**Apparatus validation — GENUINE (independently confirmed).** `store_raw` → d_seg=0, d_pose=0 proves the
full `archive.zip → inflate.sh → inflate.py → 0.raw → evaluate.py` path, the GT pair-ordering, and the
small-n `--batch-size n` truncation harness (`zip(dl_gt, dl_comp)` stops after the single comp batch →
scores the first n pairs). I verified the shape contract independently: inflate writes `(2n,H,W,C)` uint8;
`TensorVideoDataset` (frame_utils.py:218-231) memmaps to `(N,H,W,C)` and batches to `(n,2,H,W,3)`, matching
`evaluate.py:77` assert `[seq_len, 874, 1164, 3]`. The RESULT json correctly tags the SCORE axis
`[macOS-CPU advisory]` (Apple-silicon CPU, NOT contest Linux x86_64); promotion still requires Linux
x86_64 [contest-CPU] / [contest-CUDA].

---

## Wire-in / next-build asks (for the coordinator)
- C1 is the single load-bearing code bug in the deterministic arm and blocks the whole geometric vehicle:
  no units-correct homography exists in-repo → the warp arm is not shippable until one is built + verified
  against the pose-dim convention (L1).
- M1/M2 say the v2 vehicle is mostly UNBUILT: shared-canonical, SDF carrier, entropy coder, pose codec
  (#140), and movables are all design-only and wired into nothing. The byte-close is an apparatus-validation
  scaffold — valuable, but its rate-dead verdict is about the scaffold, not the designed vehicle.
- M3 (disk hygiene) and M4 (dep closure + JPEG portability) are landing-quality gates for the next build.
