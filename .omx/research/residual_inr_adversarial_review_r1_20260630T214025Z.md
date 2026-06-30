# RESIDUAL-INR HYBRID — ADVERSARIAL REVIEW R1 (round 1 of 3-clean-pass) — 2026-06-30T214025Z

**Reviewer role:** ADVERSARIAL senior reviewer (CPU-only; NO GPU; live n600 baseline pid 38641 + siblings
UNTOUCHED). **Scope:** the just-landed residual-INR hybrid (commit `a2460b14e` residual-mode trainer +
inflate compose; commit `707f9be68` PHASE-A composition layer; `src/tac/v2_compose/*`;
`tools/compose_witness_archive.py`). **Authority:** every number cited is `[macOS-CPU advisory]
NON-PROMOTABLE`; the frontier pointer is UNMOVED **0.19110** and this review does not move it.

## VERDICT: **ISSUES** (2 HIGH, 3 MED, 2 LOW; 0 CRITICAL)

The CORE rate-bearing mechanism is REAL and the faithfulness mirror is bit-exact (verified, see "What
holds"). But the **runnable end-to-end pipeline is not wired to the new mechanism** (HIGH-1) and the
**shipped composition mask is not proven to cover the residual it must fix** (HIGH-2). Both must close
before the HELD GPU run, or the "single decisive measurement" risks failing-on-load or returning a
false-negative INR-capacity verdict. NOT a NO-FAKE score violation — the pointer is honestly UNMOVED and
all artifacts are advisory.

---

## What HOLDS (attacked, verified GOOD — do not re-litigate in R2)

- **Rate win is REAL (attack #1 PASS).** `_bulk_rgb_mx`/`_resid_mask_mx` are CLOSURE-scope locals in
  `run_train` (`train_levelset_witness_realized_through_R_mlx.py:1091-1094`), never model attributes.
  EMA (`ema.shadow`), the quantized ship blob (`quantize_levelset_blob({... ema.shadow ...})` :1737), and
  every `_atomic_savez` checkpoint serialize ONLY `ema.shadow` / model state — the bulk is never in any of
  them. The bulk does not ship. Confirmed by reading the save path (no `_bulk`/`_resid` key in any savez).
- **Inflate ↔ oracle bit-exact (attack #4 PASS).** `test_inflate_residual_compose_bit_exact_parity`
  runs the self-contained `inflate.py` AS A SUBPROCESS (contest path) vs `residual_inflate_reference`,
  `np.array_equal` == True, WITH a real k=1 per-class warp + a non-empty residual INR. The floor test
  confirms empty-residual ⇒ frame0==frame1. **I ran the 9 non-MLX CPU tests: 9 passed.**
- **`contour_codec` is dense-LZMA (misnomer), inflate mirror matches.** My initial CRITICAL hypothesis
  (store encodes contour codes, inflate decodes raw LZMA) is REFUTED: `encode_partition` does
  `lzma.compress(uint8.tobytes(), FORMAT_RAW, _LZMA_FILTERS)` with the IDENTICAL filter chain the inflate
  uses — bit-compatible.
- **Baseline byte-unchanged (attack #5 PASS).** `render_through_R_mlx(compose_fn=None)` and
  `make_loss_fn(render_fn=None)` default to the exact pre-residual object; tests
  `test_render_through_R_compose_none_is_byte_identical` + `..._render_fn_default_is_identical` assert it.
- **Pairing correct (attack #2/#3).** Training loop calls `total_loss_fn(model,_cf_mx(pi),2*pi+0,2*pi+1,…)`
  (`:2024`); `_compose_mx` does `pair = code_idx//2` ⇒ both frames of pair p use bulk[p]/mask[p], matching
  the inflate's `r_code[2*p+0]`/`[2*p+1]` with shared bulk. Consistent.
- **Verdict R == inflate R.** verdict `_torch_R_to_camera_uint8` (round-then-clamp, fp32) vs inflate
  `_bicubic_up` (clamp-then-round, fp32) are provably identical for integer clamp bounds [0,255]; both
  bicubic-up render→camera. Compose op is `where`/`bulk*(1-m)+inr*m` with m∈{0,1} ⇒ exact selection.
- **Fail-closed guards real.** `--residual-mode` requires `--residual-target-npz`; incompatible with
  `--structured-init`/`--lane-prior-phi1`/`--freeze-decoder-fit-codes`; lone `--residual-target-npz`
  fails. 4 guard tests pass.

---

## HIGH-1 — the residual mechanism is NOT wired into the runnable pipeline; the "ready-to-fire" launch command would fail on load

**Files:** `tools/compose_witness_archive.py:116-118,149,253-258` · `src/tac/v2_compose/launch_command.py:228`
· `src/tac/v2_compose/residual_target.py:143` vs `residual_compose.py:201` · commit `a2460b14e` (did NOT
touch `compose_witness_archive.py`).

The rate-bearing fix + inflate compose are BUILT and unit-tested **in the library only**. The runnable
entry tool is un-rewired, and three links are missing for the HELD command to run:

1. **No producer for the trainer's input bundle.** The trainer's `--residual-target-npz` is consumed
   ONLY by `load_residual_training_bundle` (`residual_compose.py:201`), which reads keys
   `{bulk_rgb_render_res, composition_mask, composition_mask_shape, learn_classes, dilate, render_h,
   render_w, n_pairs}`. The producers (`generate_bulk_render_and_labels` → `build_residual_training_bundle`
   → `save_residual_training_bundle`) are referenced by NOTHING outside tests + `__init__` re-export +
   one trainer docstring. No CLI/tool builds the bundle.
2. **PHASE-A writes an INCOMPATIBLE npz of the same name.** `compose_witness_archive.py:116-118` writes
   `residual_target.npz` via `save_residual_target` (schema `{bulk_argmax_through_R, gt_lstars,
   residual_mask, bulk_dseg, n_pairs}`). Pointing the trainer at it ⇒ `KeyError: 'bulk_rgb_render_res'`.
   Verified key diff: MISSING = `bulk_rgb_render_res, composition_mask(+shape), learn_classes, dilate,
   render_h, render_w`.
3. **PHASE-A still emits the SUPERSEDED command; PHASE-B is still a stub.** `:149` calls
   `build_residual_inr_command` (emits `--structured-init` + `--lane-prior-phi1` = bake-into-weights =
   the NON-rate-shrinking gap-#1 mechanism). The corrected `build_residual_only_command` is **orphaned
   (no caller) and untested** (no reference in `src/tac/tests/`). `phase_b` `:253-258` still
   `raise SystemExit("…residual-INR section … not yet wired…")` — the runnable assembler cannot produce a
   residual-bearing archive even though `build_residual_blob` + the inflate compose exist + parity-pass.

**Why it matters:** the memo presents "the corrected residual-INR launch command (HOLD — emitted,
flag-validated)" + "one GPU run is the decisive measurement." In reality a HOLD→GO hits: no bundle
producer, PHASE-A writes the wrong schema, PHASE-B can't assemble the residual archive, and the corrected
command is uncalled/untested. The decisive run is not one-command-away as implied (means-as-ends
readiness overstatement; orphaned-signal / wire-in discipline).

**Fix (before GPU GO):** (a) add a CLI step (PHASE-A or new) that calls `generate_bulk_render_and_labels`
→ `build_residual_training_bundle` → `save_residual_training_bundle` to emit the BUNDLE schema the trainer
reads, under a DISTINCT filename (e.g. `residual_bundle.npz`) so it can't be confused with
`residual_target.npz`; (b) repoint `compose_witness_archive.py:149` to `build_residual_only_command`; (c)
wire `phase_b` to call `build_residual_blob` from the trained INR weights (remove the SystemExit path);
(d) add a launch_command test for `build_residual_only_command` (flag-validity + presence of
`--residual-mode`/`--residual-target-npz`/absence of `--structured-init`).

## HIGH-2 — the SHIPPED composition mask is not proven to COVER the residual; risks a false INR-capacity wall

**Files:** `residual_compose.py:70-83` (shipped mask = `isin(warped_bulk_label, {1,3})`) vs
`residual_target.py:104` (true residual = `bulk_argmax_through_R != gt_lstars`).

The composition the INR ships is `where(isin(bulk_label,{1,3}), INR, bulk)` — the INR can ONLY repaint
cells where the **bulk predicts Lane/Movable** (+dilate). But the d_seg residual the loss must close is
the set where the **bulk≠GT**. These are DIFFERENT sets. Structurally unreachable by the INR:
- bulk predicts Road/Undriv/MyCar but GT is Lane/Movable (lane false-negatives beyond the dilate radius);
- bulk Road↔Undriv↔sky confusion (neither in {1,3}) — and FACT-2 flip mass is ~50% Road / 13% Undriv,
  much of it on the road/undriv(sky) boundary, which the composition can NEVER touch.

Concrete demo (3-cell synthetic): residual=3 cells, reachable by the shipped mask=1, **UNREACHABLE=2**.
The loss floor is bounded below by the unreachable-residual fraction regardless of INR capacity. The memo
conflates the two masks ("the residual IS the Lane+Movable annulus"); `compute_residual_target` measures
residual by **GT class**, never the composition mask's coverage of it.

**Why it matters:** if unreachable-residual > the sub-0.15 d_seg budget (~6e-4…1.4e-3), the decisive GPU
run fails for a GEOMETRY reason, but would be misread as "the small INR lacks capacity" (a paradigm-level
mis-kill of the vehicle from an implementation-level cause).

**Fix (cheap, $0 CPU, BEFORE the GPU run):** once the bulk argmax stack exists, measure
`coverage = (residual_mask & composition_mask).sum() / residual_mask.sum()` and
`unreachable_dseg = (residual_mask & ~composition_mask).mean()` on the real n96/n600 cache (+ sweep
`dilate`). Gate the GPU GO on `unreachable_dseg` being comfortably below budget. Add this as a producer
output + a fail-closed warning in the compose tool.

---

## MED-1 — d_pose is a SECOND open axis, framed as "solved"; static-bulk pairs make PoseNet see ~no motion

`_compose_mx`/inflate share ONE bulk image across f0 and f1 of a pair (`pair=code_idx//2`;
`_bulk_rgb_mx[pair]` for both). The floor archive literally writes `frame0==frame1`. PoseNet maps the
(f0,f1) pair to 6-DOF; with ~95% of pixels identical (static bulk), it reads ≈zero motion ⇒ high d_pose
vs real GT motion. The residual INR differs f0/f1 only on the few-% Lane+Movable mask — closing 6-DOF
d_pose from that is dubious and **unmeasured**. The byte table lists "Pose sidecar … ~0.9KB MEASURED" and
the memo says "Pose is SOLVED," but the v2 `pose_blob` (PNTG) ships the **warp poses** (used to GENERATE
the bulk), NOT a PoseNet-output-target sidecar; no pose-target supervision is realized in the v2 inflate.
The "0.9KB MEASURED" is a BYTE measurement, true — but d_pose efficacy of the composed witness is
unestablished and should be listed alongside d_seg as OPEN, not implied solved.
**Fix:** add d_pose of the composed witness to the OPEN-quantity section; if the warp-pose dual-use is the
plan, state + measure how PoseNet recovers motion from a static-bulk pair (advisory CPU verdict already
computes it — report it for the floor + a tiny residual).

## MED-2 — the VERDICT compose path is not directly parity-tested vs the inflate in THIS build

The inflate↔oracle parity (HIGH confidence) uses `levelset_rgb_forward_numpy`/`_levelset_forward`. The
trainer's advisory VERDICT (the d_seg that decides the run) uses a DIFFERENT numpy forward `_fwd_numpy` +
`_compose_np` + `_torch_R_to_camera_uint8`. This build tests compose-equivalence and the base render seam,
but not an end-to-end "verdict d_seg == inflate-frames d_seg" on a tiny case. It relies on the
pre-existing (non-residual) verdict↔inflate forward parity. Not a regression, but the chain is assumed,
not re-proven for the composed path.
**Fix:** add a tiny test: compose+verdict frames vs the inflate subprocess frames for the same INR weights,
assert bit-identical (or d_seg-identical).

## LOW-1 — stored `warp_type_codes` are shipped but IGNORED by the inflate (dead payload + silent-divergence risk)

`build_store_blob` ships per-class warp-type codes (`compose_witness_archive.py:235` `[0,3,2,3,1]`), and
`_parse_store` reads them, but `_composite_warped` HARDCODES the routing (`fg=isin(cg,[0,1,3])` etc.) and
never consults `warp_codes`. Trivial bytes, but: (a) dead payload, and (b) if a future clip's self-detected
per-class routing differs from the hardcoded default, the inflate silently uses the wrong routing
(train≠inflate). **Fix:** either consume `warp_codes` in `_composite_warped` (drive the per-class regime
from the stored mask) or drop them from the store and document the routing as fixed-generic.

## LOW-2 — inflate runtime deps = numpy + torch + brotli (3 non-stdlib)

Runtime-closure note (HNeRV-parity L4 ≤2 deps; PR-family precedent allows brotli/torch). Verify the
contest inflate runtime tree includes `brotli` (top-level `import brotli` in the residual section) AND
`torch` (bicubic R) before the GPU run, per the "Runtime closure" non-negotiable. Pre-existing for the
witness; flagged for the byte-close packet smoke.

---

## R2 entry criteria (clean-pass counter currently 0/3)

Close HIGH-1 (wire the bundle producer + PHASE-A/B + repoint to `build_residual_only_command` + add its
test) and HIGH-2 (measure composition coverage of the residual on the real cache, gate the GO). Address
MED-1/MED-2 (open-quantity honesty + verdict↔inflate parity test). Then re-run R2. Per the Recursive
adversarial review protocol, any new finding resets the counter.

Cross-refs: `residual_only_trainer_mode_landed_20260630T212107Z` · `v2_compose_composition_layer_landed_20260630`
· `[[gr-unified-action-full-witness-architecture-20260629]]` · CLAUDE.md NO-FAKE #6/#7/#8 + rule-118 +
"no orphaned signals / wire-in" + "Forbidden premature KILL" (the false-capacity-wall risk).
