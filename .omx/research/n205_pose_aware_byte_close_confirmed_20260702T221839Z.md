# #205 pose-aware byte-close — ENGINEERED + CONFIRMED (warp-real-luma frame0)

**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE`. No score/frontier/promotion claim. Pointer 0.19110 UNMOVED.
**UTC:** 2026-07-02T22:18:39Z · **git HEAD (pre-commit):** `18927a1ae` · **tool:** `tools/levelset_byte_close_and_eval.py`

## Why this exists (the launch-gate AXIS 9 requirement)

The #205 SEAL cited the **ancestor-RGB d_pose 3.4e-5**, never validated on the SDF witness, and the byte-close
path could **not even reproduce the pose decode** — so the true d_pose + rate were never MEASURED through the
REAL decode (`review_seals_borrowed_numbers_and_unrun_configs_measure_at_real_config` +
`project_pose_solved_screw_twist_dual_use_film_conditioned_sidecar` ⛔ leading correction). CLAUDE.md
"Recursive adversarial review protocol" **AXIS 9** now makes a launch SEAL INVALID until it measures
d_seg+d_pose+rate through the real byte-closed decode. This landing supplies the missing byte-close half so a
warp-real-luma pose-carrier checkpoint can be byte-closed and its pose MEASURED (not borrowed).

## The gap (found by a sister agent — CONFIRMED, then FIXED)

`tools/levelset_byte_close_and_eval.py` + its level-set inflate string rendered **frame0 from the INR**, so a
`--pose-carrier` checkpoint byte-closed today would be POSE-BLIND (d_pose garbage) AND under-count rate (the
keyframe payload was never in the archive). CONFIRMED: the pre-edit tool had **zero** `pose_carrier` / `warp` /
`keyframe` handling.

## What was engineered (all in `tools/levelset_byte_close_and_eval.py`; DEFAULT-OFF, byte-identical when off)

1. **Reproduce the warp-real-luma frame0 decode.** New optional archive block (`PCAR1`, the 6th block, gated by
   a manifest `pose_carrier` flag) stores per-pair homography `H` (fp64) + the dual-use twist `xi` (fp16,
   provenance) + the real keyframe luma (brotli, per `--pc-keyframe-stride`/`--pc-keyframe-downscale`). The
   inflate decodes **frame0 = warp(stored keyframe, stored H) at native res** (SEG-free: SegNet reads only
   frame1, `upstream/modules.py:108`); frame1 stays the witness INR render. The inline inflate warp is a
   VERBATIM copy of the tool-side `_pcar_warp_frame0_from_H`, which bit-matches the module authority
   `tac.boundary_math.warp_real_luma_frame0.warp_frame0_uint8_numpy` (proven 0-diff). No `exp_se3` in the
   decode path (H stored) → shipped inflate == numpy oracle **by construction**.
2. **Count the keyframe payload in the rate.** The `PCAR1` block is inside `0.bin` → inside `archive.zip` → the
   MEASURED `st_size` already includes it (rule-118: keyframe luma + H COUNTED; inverse-warp + R FREE). The
   canonical `keyframe_payload_accounting` (REUSED from `tools/compose_witness_archive.py`, f7c6abdea) is wired
   for the honest line item, fed the MEASURED keyframe bytes and **cross-checked against the real section
   `st_size`** (`section_bytes_match_report: true`).
3. **Carrier-general.** The block is decode-general (any per-pair H + keyframe schedule); the residual/FiLM
   pose paths (Quantizr-style store-nothing-but-ξ) reuse the same block by supplying a different H/keyframe
   plan. `xi` is stored (dual-use with the pose sidecar) so the byte-optimal design (store xi, derive H FREE)
   drops in without a grammar change.
4. **Determinism spine kept:** numpy-fp32 authority + the existing bit-exact round-trip gate now covers frame0
   (warp) too (carried through the capped repack + oracle). New tests
   `src/tac/tests/test_levelset_pose_carrier_byte_close.py` (5, incl. inflate-string == tool-oracle copy
   faithfulness). `warp_real_luma_frame0` tests still 22/22.

## CONFIRMED through the REAL byte-closed decode (t1 smoke witness, n=6, gt_n6, frozen CPU-torch; NEVER MPS)

Checkpoint `experiments/results/levelset_pose_smoke_20260627T070546Z/t1` (n_pairs=6 — a REAL 6-pair set, the
natural size for this witness; NOT a synthetic 4-pair toy). Two variants:

| variant | keyframes | keyframe bytes | archive.zip | rate_term | **frame0 decode bit-exact** | d_pose carrier | d_pose ceiling |
|---|---|---|---|---|---|---|---|
| native lossless (`--pc-keyframe-downscale 1`) | 6 @ 874×1164 | 9,507,931 B | 9,584,494 B | 6.3819 | **True (max_abs=0)** | 172.22 | 21.80 |
| downscaled (`--pc-keyframe-downscale 4`) | 6 @ 218×291 | 696,931 B | 770,817 B | 0.5133 | **True (max_abs=0)** | 172.66 | 36.50 |

- **(a) DECODE IS BIT-EXACT / DETERMINISTIC.** `frame0_max_abs_uint8_diff = 0` — the shipped inflate warp ==
  the numpy authority bit-for-bit (native == `warp_frame0_uint8_numpy`); the general bit-exact gate PASSES over
  frame0+frame1. Reports: `reports/n205_pose_carrier_byte_close_confirm_t1_n6_{native,ds4}_20260702T221839Z.json`.
- **(b) TRAINING-SIDE ↔ REAL-DECODE d_pose PARITY = IDENTICAL BY CONSTRUCTION.** raw frame0 == numpy-authority
  warp frame0 (bit-exact) AND raw frame1 == witness render (bit-exact gate) ⇒ the training-side warp d_pose
  EQUALS the real-decode d_pose — **no surrogate gap** (the AXIS-9 requirement).
- **(c) RATE COUNTS THE KEYFRAMES**, cross-checked against the real archive section (match=true).
- **Honest S (advisory, NON-PROMOTABLE):** native S ≈ 99.65 = 100·0.5177(d_seg) + √(10·172.22)(41.5) + 6.38.
  Dominated by the SMOKE witness's garbage d_seg (0.5177 — t1 is barely trained) and the un-lowered pose. This
  is an **infrastructure confirmation on a smoke witness**, NOT a promotable row.

## The measured d_pose is a FINDING (confirms the review memo — NO borrowed number)

The warp-real-luma carrier, in the **contest-legal witness composition** (warp-frame0 + witness-frame1),
**does NOT lower d_pose** on this witness: carrier **172.22** vs unwarped-real-f0 168.22 vs pose-blind null
189.62. The warp physics DO work on the reference **(real-f0, warped-real-f1)** pair — the CEILING d_pose is
**21.80** (native; the reference `tools/measure_warp_dpose_through_R.py` measures ~10.53 at its FITTED s_t; my
fixed s_t=0.16 + smoke witness account for the rest). So:

- The low d_pose the design needs is **OPEN + UNMEASURED on the witness**: it requires a GOOD witness frame1
  (not the t1 smoke's garbage render) AND the **trained `dxi` residual (w_pose>0)** — both gated on a real
  #205 run. The **ancestor-RGB 3.4e-5 is NOT reproduced and NOT borrowed.**
- Per-pair native keyframes are **expensive** (rate 6.38 @ n=6). Keyframe SPARSITY (stride>1 + kf→pair relative
  warp) + a lossy keyframe codec are the OPEN rate levers — MEASURED-of-what-is-stored here, the memo's
  ~0.0304 sparse figure is NOT borrowed.

## Default-off guarantee + regression

`--pose-carrier` OFF ⇒ archive.zip **byte-identical** to pre-edit (72,683 B on t1/n=6, verified). Trailing
optional blocks (lane 5th, pose-carrier 6th) are now **manifest-flag-driven** (never a bare `off<len`), so a
lone pose-carrier block is never misread as lane. Tests: pose-carrier 5/5, warp_real_luma 22/22, ruff F821 clean.

## Next (to move the pointer, gated on #205)

A REAL n600 witness (good frame1) + a w_pose>0 trained `dxi` residual (or the Quantizr-FiLM store-nothing-ξ
render) → byte-close with `--pose-carrier` → measure the REAL witness-composition d_pose through this decode →
if it clears, the FIRST pose-aware level-set exact-eval row. This landing removes the byte-close blocker for
that measurement.

## Files
- `tools/levelset_byte_close_and_eval.py` — pose-carrier serialize/parse/warp + inflate decode + rate counting
  + `pose_carrier_confirm` + CLI (`--pose-carrier`, `--pc-{s-t,s-r,pitch,keyframe-stride,keyframe-downscale}`).
- `src/tac/tests/test_levelset_pose_carrier_byte_close.py` — 5 tests (warp==authority, round-trip, default-off
  byte-identity, capped slice, inflate-string==tool-oracle).
- `reports/n205_pose_carrier_byte_close_confirm_t1_n6_{native,ds4}_20260702T221839Z.json`.
