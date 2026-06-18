# Path-B recalibration + re-solve audit — what changes when the vehicle is the frontier, not bc20 (2026-06-18)

**Operator: "there may be other things that need to be recalibrated and solved as well."** Correct — the
desk-calc redirect (bc20 small-basis build → QAT-shrink the EXISTING 0.19110 frontier) is a VEHICLE change,
and most session calibrations are vehicle-relative. This audit enumerates the cascade so nothing silently
mis-applies. All `[advisory]`; pointer UNMOVED 0.19110. Disposition: RECALIBRATE / MOOT / VALID + fold point.

## Calibrated-for-bc20 → Path-B disposition
| item | calibrated for bc20 | Path-B disposition | fold point |
|---|---|---|---|
| **margin-saliency map (#141)** | computed on the bc20 basin decoder | **RECALIBRATE** — recompute |∂margin/∂input| on the FRONTIER decoder (its d_seg-critical weights/stages differ) | the QAT build (the bit-alloc cost) |
| **byte-shrink model (#136)** | int8→int4 ratios on bc20 | **RECALIBRATE** — frontier is FP11+CTXR; FP11→int4 is a bigger drop + the CTXR re-encode differs | the PTQ-on-frontier gate (RUNNING — measures it) |
| **taper / channel-alloc (#121)** | re-solve channels at new capacity | **MOOT for Path B** (frontier architecture is FIXED) → translates to per-stage QAT BIT-alloc | already re-pointed (bit axis) |
| **d_seg levers (margin-hinge, #137, #138)** | LOWER bc20's d_seg (0.0026→…) | **RECALIBRATE the ROLE** — frontier d_seg already 0.00056; QAT must HOLD it, + #137 boundary sidecar TOPS-UP any QAT spill (cheap) | QAT build + #137 as top-up |
| **pose codec (#140) / pose treatment** | bc20 FiLM-STORE pose (d_pose 0.00034) | **RECALIBRATE/likely MOOT** — frontier d_pose ~0.00003 already (pose term 0.017); QAT must hold it, not improve it | QAT build (pose-hold check) |
| **byte-close pipeline (G3)** | torch_vehicle packet/inflate | **RECALIBRATE** — frontier = PR101 grammar (reconstruct_raw_sections → decode_decoder_compact → re-pack) | the PTQ gate already uses the PR101 path |
| **stretched-exp d_seg model** | bc20 50k anneal trajectory | **MOOT for Path B** — no long anneal; QAT-finetune is short, different dynamics | n/a (QAT has its own dynamics) |
| **label-noise floor / capacity reasoning** | bc20 capacity ceiling | **MOOT** — frontier already near the d_seg floor (capacity-bought); the question is rate, not capacity | resolved |
| **score-aware QAT (Lever 4)** | bc20 trainer | **RECALIBRATE** — re-tune sensitivity-EMA + per-tensor grids for the frontier decoder + the CTXR/int4 interaction | the QAT build |
| **canonical config / reconciliation memo** | bc20 bind-all | **RECALIBRATE** — the canonical "config" is now the frontier + the QAT-shrink recipe | post-QAT (new canonical memo) |
| **VALID (unchanged)** | the master gradient (∂S calculus), the contest score fn, the apples-to-apples/NO-FAKE discipline, the G3 finding (local≈contest-CPU), the margin-saliency PRINCIPLE | **VALID** — vehicle-independent invariants | — |

## What must be SOLVED (new, for Path B)
1. **The core solve: QAT holds the frontier's FP11→int4 distortion.** PTQ collapses (the gate measures the
   floor); QAT-finetune must recover d_seg within ~+0.0003. The decisive build.
2. **The CTXR/entropy re-encode at int4.** The frontier's decoder section is FP11+CTXR (range-coded). Shrinking
   to int4 means RE-DERIVING the entropy coding for int4 weights (the int4 histogram/context model), or the
   "−47.7% bytes" won't materialize. The byte-close must re-encode, not just re-quantize. (The PTQ gate's
   re-pack tests this.)
3. **margin-saliency on the frontier decoder** (recompute the cost map) → the per-stage/per-tensor bit-alloc.
4. **d_seg top-up:** if QAT spills d_seg above the frontier's 0.00056, the #137 boundary sidecar (0.78 B/flip)
   tops it up cheaply — re-pointed at the frontier's residual flips.
5. **Submission/originality (a GATE, not a blocker):** the frontier may be public-PR-derived (lane pr110). For
   the LOCAL exact pointer + proving the technique, QAT-shrink is legitimate (our original score-aware QAT on a
   borrowed substrate = defensive bank). For a SUBMISSION, the borrowed_substrate_accounting (NO-FAKE class 7)
   must separate ours-original (the score-aware QAT) from borrowed (the frontier decoder). Do NOT skip this if
   it ever PRs.

## Sequence (folds into the running Path-B pipeline)
PTQ-on-frontier floor (RUNNING — recalibrates the byte model + tests the CTXR re-encode + measures the QAT
floor) → [if collapses → ] score-aware QAT-finetune (recalibrate margin-saliency + Lever-4 to the frontier) →
byte-close (PR101 grammar) → CPU exact_eval → if beats 0.19110 byte-closed → paired dual CPU/CUDA exact eval
(the pointer-mover) + borrowed-substrate accounting if it PRs. Cross-refs: capacity_rd_score_aware_qat_pivot_*,
campaign_math_review_*, yousfi_council_checkin_*, score_aware_taper_channel_alloc_*, tac.margin_saliency_map,
tac.post_hoc_weight_shrink, tac.torch_vehicle.score_aware_qat, #136/#137/#141.
