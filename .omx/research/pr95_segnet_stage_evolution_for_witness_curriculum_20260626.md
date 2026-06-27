# PR95 d_seg TREATMENT (NOT the vehicle, NOT the rate machinery) -> the SHORT witness curriculum

Operator 2026-06-26: "learn from PR95's treatment of segnet and the order and how things evolve over the 30k
epochs" + SHARPEN "not hnerv though don't get sucked back into pr95 land just learn from the dseg" +
"shouldn't need 30k epochs because we don't need all of those rate steps." LOOP-TRAP GUARDRAIL: the witness
is a coord-INR, NOT HNeRV; learn the d_seg LOSS-FORM SEQUENCE only. $0 read-only. NO-FAKE: cites file:line.
PR95's OWN per-stage d_seg is unrecorded (intake L28); trajectory = our MLX-port `[macOS-MLX research-signal]`.

## WITNESS CURRICULUM RECOMMENDATION (HEADLINE — short, d_seg-only, NON-HNeRV)

Coord-INR witness (~83K, ~61s/ep), scorer-only, trained THROUGH R with frozen SegNet. NO HNeRV decoder, NO
QAT-for-weight-rate, NO C1a, NO lambda, NO weight-noise-rate. Minimal sequence (coarse -> margin -> drop):

- A. CE warm-up — `ce_seg_loss` full CE (mlx_losses.py:70-83), high LR, EMA off/low. DOES: coarse argmax
  partition over ALL pixels. ~200-400 ep.
- B. tau_softplus(0.3) margin-sharpen — `mean(tau*softplus(-margin/tau))` (L86-98), mid LR, EMA 0.997. DOES:
  the BIGGEST measured d_seg drop (CE 0.0104->0.0064 then softplus ->0.0040). ~400-700 ep. **Treat B as a NEW
  stage (re-treat the transition: own LR/EMA/spike-guard) — our just-failed witness destabilized because the
  margin stage inherited base-stage treatment** (feedback_different_stages_need_different_treatment_20260626).
- C. l7_softplus hard-pixel refine = **WHERE THE MARGIN-WEIGHT ALLOCATION LEVER FITS** — `l7_softplus`
  (L110-129): 5× weight (`1+l7_mult=4`, tunable) on small-margin pixels `margin<1.0`, renorm mean-1, weight
  under stop_gradient. DOES: concentrate gradient on the binding all-class edge band. KEEP THE LOSS, DROP the
  C1a co-passenger PR95 bundled here. low LR. ~300-600 ep.
- D. Muon finetune finisher — Muon on the INR weights, very low LR. DOES: the conditioning drop ("Muon is THE
  drop", our capstone obs). ~200-400 ep.
- ADAPT/DROP smooth: PR95 stage3 `sigmoid(-margin/tau)` (L101-107) RAISES d_seg in our trace (+6.8%,
  transient) — DROP unless it directly minimizes the realized post-R d_seg. KEEP from sigma ONLY the
  eval-roundtrip / R-survival robustness (uint8/resize noise in-loop), NOT the rate weight-noise schedule.

Estimate: ~1100-2100 total ep vs PR95's 29650 (~14-27× shorter) because the ~14500 rate-machinery epochs +
the smooth stage are gone. Absolute counts only set duration (curriculum.py:191); the ORDER is the lever.

## PARTITION OF PR95's 8 STAGES (keep vs drop) — epoch accounting

Spec `curriculum.py:116-158` (= profile L32-39). Total = 3000+5650+1500+500+9000+2000+3000+5000 = **29650**.
seg_weight=100, pose_weight=1 (curriculum.py:92-95); pose `sqrt(10*MSE)` CONSTANT all stages (L171-175); KEEP.

(a) d_seg-CONDITIONING = LEARN: stage1 CE 3000 · stage2 tau_softplus 5650 · the l7 LOSS FORM inside stage5 ·
    stage8 Muon mechanism 5000. = the coarse->margin->hard-pixel->drop sequence.
(b) RATE MACHINERY = DROPPABLE (entropy-codes the HNeRV DECODER WEIGHTS — a rate story the witness does NOT
    share): stage4 QAT 500, stage5 C1a entropy term (the 9000ep was long for C1a ANNEALING, not for d_seg —
    biggest single block), stage6 lambda_sweep 2000 (pure rate λ0.01->0.02), stage7 sigma rate-portion 3000.
    **Rate-dominated epoch blocks (stages 4+5+6+7) = 14500 of 29650 ≈ 49% droppable** for the witness; stage8
    also carries QAT+C1a co-passengers that drop (keep only Muon). The witness rate game is the L13 non-RGB
    format + ~8-dim task-space coords -> hundreds of bytes (int8+brotli of the small INR), so weight-histogram
    pressure is irrelevant.

## PER-STAGE LOSS FORMS + ORDER RATIONALE (cited)

Margin core `_target_margin_mlx` (L49-67) = `target_logit - max_competing_logit` = the EXACT sign-quantity
that decides argmax. Guard `pr95_distortion_practices_guard.py:276`: "CE -> margins -> QAT -> C1a -> sigma ->
Muon polish." Rationale: CE = coarse argmax (distribution match) -> softplus = switch to MARGIN (the only
thing argmax cares about) -> smooth = surrogate the disagreement RATE -> [QAT/sigma = survive INT8+brotli ->
C1a/λ = coder-friendly weights — RATE, dropped] -> Muon = conditioning polish. No `kl_on_logits`/T=2.0 in
PR95 (grep empty) — KL T=2.0 is QUANTIZR, not PR95.

## d_seg EVOLUTION (our MLX-port n600; DAG sub015...:532,538,637)

CE 0.01045->0.00643 (DROP) · softplus ->0.00396 (DROP, min) · smooth 0.00396->0.00423 (RISES, transient) ·
c1a_l7 ->0.00385->0.00369 (recovers, new min, slow) · Muon = finisher. Capstone obs (CLAUDE.md frontier §):
"CE+softplus LOWER, smooth RAISES, c1a->neutral, λ+sigma neutral, Muon is THE drop." End-state existence
proof: bc36 d_seg 6.02e-4 (MEMORY pr95_prune). Lever-B all-class-dir+cap hit 0.004445 ep100 descending.

## SEGNET KNOBS (cited) + FROZEN-SCORER EXPLOITATION

tau=0.3 (L87); l7_threshold=1.0, l7_mult=4.0->5× (L113-114); EMA 0.997 (curriculum.py:101); grad_clip 1.0;
muon_lr 2e-4 (L155). [DROP for witness: C1a λ0.01->0.02 cat_sigma0.2->0.1 (L182), sigma weight-noise
0.2->0.1 (L139-155), INT8 QAT (L243-256).] Frozen-scorer: SegNet scores ONLY last-frame hard argmax (guard
L107) -> only sign(margin) matters -> ALL seg losses are argmax surrogates; smooth=sigmoid(-margin/tau) IS
the relaxed score; L7 5× attacks the boundary band; no scorer bytes in archive.

## COMMIT

Committed via subagent_commit_serializer. Pointer UNMOVED 0.19110; $0 read-only.
