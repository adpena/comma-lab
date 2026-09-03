# ddm_gf2_static_dynamic_generator_form — the SECOND contender for the one open inequality: a STATIC/DYNAMIC-FACTORIZED generator (one shared scene/GOP static field + small per-pair dynamic codes for what moves) raced scorer-free at n600 against "packet ≤ 71,404.5 B with ≤ 46,804 mismatches", in parallel with gc1's capacity-knob race on the existing form (experiment-design rule 8: multiple contenders → multiple paths)

## MANDATE

Operator standing GO; "multiple contenders → multiple paths" is a CLAUDE.md non-negotiable. The only route with
arithmetic room to sub-0.12 is generator ACCURACY at ≈50–71 KB (`ddm_gs3_gestalt_after_submission_20260903.md`;
`ddm_rn1_n600_reopen_sweep_20260903.md`: 1.5× GF1 = 71,404.5 B packet must leave ≤ 46,804 mismatches, a 28.31×
reduction from GF1's 1,325,033; 1.8× is closed by bytes). Arm gc1 (LIVE) is racing an explicit capacity control on
the EXISTING form (HG1/GF1: `experiments/ddm_gf1_generator_form_on_lb1_field.py` 5b884ec957). ol1 (`ddm_ol1_online_signal_scan_20260903.md`
0e21272da1) found the one outside form that prices into range only as a changed generator: NerVast-style multi-chunk /
shared-scene structure (18,994 B projection) — and named the mechanism: "most dashcam structure is stable while
changes concentrate near hard boundaries". Our own measurements agree: MyCar 25.4% area is STATIC (IoU 0.994),
Undrivable 49.5% (IoU 0.995), Road (0.955); the bits live at Lane (IoU 0.263) and Movable (0.903). A generator that
spends its capacity ONCE on the stable 98% and per-pair only on the moving 2% is a different capacity allocation
from GF1's per-pair fit. This arm builds and races that form. Distortion is NOT inherited by a generator (bz2d's
1.157× amplification): this arm prices FIELD mismatches exactly; the scorer row is MAIN's later.

## THE OBJECT

- **Static term S:** one shared scene field (or a small number of GOP-shared fields — declare the count as a
  counted parameter) over the 512×384 lattice in the ego frame, represented by the same generator family class as
  GF1 (SDF/level-set or the born-small renderer's palette form — read `experiments/ddm_rb1_born_small_renderer_build.py`
  42fd8b1e55 and the GF1 runner at source; reuse their integer export + coder, never a float parameter count).
- **Dynamic term D_t:** per-pair codes for what moves: a low-dim ego-motion/horizon adjustment of S (warp
  parameters), plus a sparse per-pair set for Lane dashes and Movable objects (parametric dash/box events or a
  small per-pair residual generator). Everything counted after the real coder.
- **Composition:** field_t = argmax over classes of S warped by D_t's geometry, overwritten by D_t's dynamic
  objects. Deterministic integer decode. The residual (corrections to the exact field) priced with a
  domain-matched coder AND the generic 0.2909 B/site — report both.

## SCOPE

1. **Ceiling first (closed-form, /bin/zsh):** on the retained exact AFR1 field (JBP1 null field custody), measure the
   STATIC ceiling: the best single static field (per-site modal class over all 600 pairs after the best rigid
   alignment) — its mismatch count per class and the exact conditional-entropy of the residual given it. If the
   static-only mismatch count already exceeds what a ≤ 71 KB dynamic term could plausibly repair (state the
   arithmetic), refuse at the ceiling: typed CEILING-REFUSED with numbers.
2. **Build + fit** S and D at ≥ 3 total-packet sizes spanning 47–71 KB (never > 71,404.5 B), same fit budget per
   point (SCOPE reduction, declared), CPU torch/numpy only; retain every packet, decoded field, residual.
3. **Curves:** packet bytes → mismatches (per class, total); residual price (both coders); the crossing (if any)
   of packet + residual ≤ 85,020 B.
4. **Decision rule (pre-registered):** any point with packet ≤ 71,404.5 B AND (mismatches ≤ 46,804 OR
   packet + domain-matched residual ≤ 85,020 B) → typed CANDIDATE with a receiver-closed archive plan + a
   scorer fire order for MAIN (realized d_seg/d_pose must be measured; pose absolute budget ≤ 1.25e-4, memory
   m110). Otherwise typed FORM-CLOSED with the fitted curve and its extrapolated crossing as a number.
5. **Rule 8 leg:** report the static/dynamic split of bytes and of mismatches — is the form's failure (if it
   fails) in the static term (scene fit) or the dynamic term (Lane/Movable)? That decides the next form.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY; `submissions/semantic_joint_ctxmix/` READ-ONLY. NO scorer, NO Modal, NO Metal/MPS
  (the QBR1 burn owns the device; fpc3's memory preflight measured 116.6 GB projected system use vs a 96.2 GB
  ceiling while QBR1 is resident — a second heavy process is REFUSED by that arithmetic too): CPU only, and
  keep peak RSS ≤ 20 GB (measure it; refuse above). DETACHED >30-MIN COMPUTE ONLY via
  `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt <name> -- <cmd...>`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD under `/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/` (AP is
  reserved for the burn). Do not touch gc1's files/store (`experiments/ddm_gc1_*`, its Vertigo store) — cite
  its results if they land first.
- CLOSED-FORM-FIRST: the static ceiling and the byte inequality are exact arithmetic on the retained field.
- Rule 118: generator weights/params are counted; nothing video-derived in code.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_gf1_generator_form_capacity_verdict_20260830.md` — 5.09× at 47,603 B on the per-pair form; target-
  independent mechanism.
- `ddm_rn1_n600_reopen_sweep_20260903.md` — the 1.5×/1.8× inequalities; "calling a GF1 repeat 1.5× is false".
- `ddm_ol1_online_signal_scan_20260903.md` — NerVast prices in range only as a changed multi-chunk generator;
  contour/chain forms stay 77,864 B above the token section: do not build a contour coder.
- `ddm_bz2d_distortion_verdict_20260830.md` — 1.157× token→argmax amplification; pose 152× worse on the fork.
- `ddm_dds1_ceiling_readjudication_20260901.md` — born geometry as CONTEXT ~613 B: standalone accuracy is the route.
- `ddm_ltg1_lane_topology_generator_floor_20260831.md` (233 KB) / `ddm_blp1_born_lane_predictor_20260831.md`
  (60 KB weights) — Lane-only generators are far over; the dynamic term must be SPARSE (events), not a Lane model.
- `ddm_mc1_motion_compensated_previous_plane_20260903.md` — constant-velocity warps of the decoded field worsen
  alignment: the dynamic geometry must be FITTED per pair (counted), not extrapolated.
- memory `box-retired-min-s-target-warp-family-closed-1273-bytes-per-error` — warp as a FLIP predictor is
  closed (distortion axis); here warp is a counted generator parameter (rate axis) — different object.

## OPTIMAL FORM

- Family exemplar: GF1's measured generator form, reference `experiments/ddm_gf1_generator_form_on_lb1_field.py` (commit 5b884ec957;
  verdict `.omx/research/ddm_gf1_generator_form_capacity_verdict_20260830.md`), and the born-small renderer
  build `experiments/ddm_rb1_born_small_renderer_build.py` (commit 42fd8b1e55); the parallel contender is arm gc1
  (charter `.omx/research/charters/ddm_gc1_generator_capacity_control_20260903.md`).
- SCOPE reductions declared per row (fit budget per point; ≥3 points). MECHANISM reductions FORBIDDEN: no float
  parameter counts as bytes; no generic-only residual price; no prefix subsets; no static term fitted on a
  subset of pairs.
- **PRIOR-LAW PREDICTION (falsifiable):** the per-class IoU table predicts a static term captures ≥ 97% of sites
  at ≤ 20 KB, leaving the dynamic term to repair Lane+Movable (~1.8% of area) — and gf1's mechanism predicts
  that repair still costs > 46,804 mismatches at ≤ 51 KB dynamic bytes. FALSIFIER: any point meeting the
  decision rule — count it plainly and fire the scorer order.

## DELIVERABLE

`.omx/research/ddm_gf2_static_dynamic_generator_form_20260903.md` — the static ceiling table, the packet→
mismatch and residual curves (per class), the static/dynamic split, the typed decision, RECALL EVIDENCE,
NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer. Cite
`docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
