# ddm_js2b — edge-conditioned seg solve on custody bases, RELATIVE gauge + T4-gated acceptance

Successor of ddm_js2 (F1 fired; instrument question RESOLVED by MAIN's custody
control 2026-08-12). Consumes the corrected routing in the js1 skeleton annex
"DECODE EXONERATED; THE 44% IS THE FORWARD INSTRUMENT".

## MISSION

Produce the first admissible SEG candidates on the cp135 base: implicit
edge-conditioned corrections (fd135 edge-table row 5 + falsifier 2 — the F26
existence proof: d_seg −1.017e-7 via two FiLM codes + jointly compensated
carrier coords) that reduce flips on the LOCAL instrument by a margin large
enough to survive axis transfer, at counted-byte prices that keep the joint
ΔS negative. The seg leg must eventually supply ≥ −0.004 S of the −0.011955
gap to sub-0.15 (waterfall receipt: cp135_composed_floor_waterfall_20260810).

## WHY NOW (the resolved instrument state — binding facts)

- MEASURED 2026-08-12 (INSTRUMENT_VALIDATION_CUDA_CUSTODY.json): the CPU-torch
  SegNet on the REAL T4 scorer-input planes gives 50,389 flips vs the promoted
  34,968 — the CPU-vs-CUDA forward gap is 0.00013072 d_seg = 0.0131 S at this
  operating point. The local decode was scorer-equivalent within 6 flips.
- Therefore: the old ≤1% absolute-agreement gate is RETIRED (wrong-object).
  Local absolute seg numbers are NEVER quoted as progress on this family.
- The instrument gap lives in TIE pixels (argmax near-ties relabeled by FP
  order/device). DESIGN LAW: target margin-robust flips — corrections that
  flip a pixel with post-correction margin ≥ δ (calibrate δ from the custody
  data itself: the margin distribution of pixels where local and promoted
  instruments disagree). Margin-robust flips transfer across instruments;
  tie-nudges do not and are FORBIDDEN as claimed progress.

## CUSTODY BASES (all sha-receipted; payload law holds)

- Scorer-input planes (the exact arrays the T4 lane scored):
  /Volumes/VertigoDataTier/pact/ddm_js2_20260812/instrument_validation_cuda/scorer_input_cache_tensors/
  (segnet_last_rgb.npy 600x3x384x512 fp32 · posenet_yuv6_pair.npy · manifest.json)
- Local argmax field on custody inputs: lstars_local_on_custody.npy (+sha in
  INSTRUMENT_VALIDATION_CUDA_CUSTODY.json).
- GT: experiments/results/mlx_fleet_gt_cache/gt_n600.npz (lstars).
- Base archive: /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/
  (archive.zip sha 6eb1a3b7…, 186,252 B).
- fd135 receipts: .omx/research/ddm_fd135_fractal_decomposition_20260810.md
  (edge-table row 5, F26 existence proof, retained book path).

## BINDING LAWS

1. **RELATIVE GAUGE ONLY.** Rank proposals by local Δflips vs the local
   baseline 50,389 on the SAME instrument (same batch=16, same thread count —
   et4: batch shape is part of the instrument). Absolute d_seg claims carry
   the label [macOS-CPU advisory, instrument floor 0.0131 S] and are
   non-promotable.
2. **MARGIN-ROBUST DESIGN.** Every proposal reports the margin distribution of
   its flipped pixels; flips with margin below the calibrated δ are counted
   separately and NOT claimed.
3. **T4 ACCEPTANCE IS MAIN-FIRED.** Arms cannot dispatch Modal. When a
   proposal stack projects local Δflips ≤ −2,000 (≥ −0.0017 S local-relative)
   at ≤ +1,000 counted bytes, emit a QUEUED-WITH-A-FIRE-ORDER row for MAIN:
   one paired T4 row on the composed archive (~$0.08 vs #381).
4. **Payload law (P0 DEF CON 1000):** every materialized candidate archive +
   argmax field persisted with sha256+bytes receipts to the SSD tier.
5. **Resumable + serializer:** state.json on disk; commits via
   tools/subagent_commit_serializer.py --no-co-author with post-edit shas;
   tags [no-triality] [p0-ledger-ok]. No REVIEW_GATE_OVERRIDE on .py.
6. **Pose survival:** every seg proposal measures local d_pose on the custody
   pose planes as a guard (relative to the base); any pose regression ≥ 2e-6
   local disqualifies the proposal (the pose axis is knife-edge per the
   x15.2/x36.9 adjudication).

## OPTIMAL FORM

- Reference form: fd135 row-5 implicit conditioning per the F26 existence
  proof (two FiLM codes + jointly compensated carrier coords on the HPAC/CPR1
  receiver), receipts pinned at
  .omx/research/ddm_fd135_fractal_decomposition_20260810.md (path+content).
- SCOPE reductions (legal): propose/rank on pair subsets first, but any
  QUEUED fire-order row must be n600-projected on the local instrument.
- MECHANISM reductions: NONE authorized. A flat-paint or additive-overlay
  variant is TOY-BRACKETED (cannot produce a family verdict — the explicit
  overlay family is already dead per #941/fd135).
- Provenance pins: cp135 archive sha 6eb1a3b79cb167e03372339e07e93cae13b6ba31
  14a9eb917288bb038622edb6; custody manifest shas in
  instrument_validation_cuda/scorer_input_cache_tensors/manifest.json.

## DELIVERABLES

1. δ calibration: the margin distribution of local-vs-promoted disagreement
   pixels (from lstars_local_on_custody vs the promoted scalar arithmetic +
   margin readback on custody planes) → the measured margin-robustness bar.
2. The implicit-conditioning solve: FiLM-code/carrier-coord proposals through
   the real receiver → local Δflips + margin profile + counted-byte price per
   proposal (real coder, not entropy estimate).
3. Ranked proposal table with the rule-3 fire-order rows for MAIN.
4. Skeleton annex queue line (MAIN applies; skeleton edits reserved to MAIN).
5. Honest negative if the family prices out: verdict_scope labeled, with the
   measured price wall named.

## FALSIFIERS

- F1: δ calibration shows ≥ 80% of the reachable flip population is
  tie-fragile (below any workable δ) → margin-robust implicit conditioning is
  FORMULATION-dead on this base; report the measured margin histogram.
- F2: best proposal stack prices > 3 B per margin-robust flip (worse than the
  waterfall exchange rate −1,494 B = −0.001 S ≈ 0.85 B/flip equivalent) →
  edge-conditioning is rate-dominated; route residual to the training leg.
- F3: pose guard disqualifies every stack → seg/pose coupling wall; report
  the coupling map.
