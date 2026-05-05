---
name: Round 6 council corrections — Lane MM v2 NOT invalidated; 5 more lanes ARE
description: 2026-04-29 PM. Council Round 6 (.omx/research/council_round6_adversarial_20260429.md) found 3 defects in my Round 5/Council A reporting. CORRECTING: Lane MM v2 is BUILD-only (no SegMapTrainer) so its 2.63 FALSIFICATION verdict STANDS. ALSO: Council A undercounted invalidated lanes by 5 — Lane WC-S, PA, HM-S, FR-Ω, FC ALL use SegMapTrainer and ARE invalidated by the .round() zero-gradient bug.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Defect 1 (commit attribution): Lane MM v2 verdict CORRECTION

**My earlier claim** (in commit body for Check 86 fix + memory project_lane_uniward_v8_NO_OP_finding_20260429.md):
> "Lane MM v2 ('FALSIFIED 2.63') verdict needs revisit — the lane likely never trained."

**Round 6 finding** (verified by Council Round 6):
> "Lane MM v2 is a BUILD-only path (`experiments/build_lane_mm_archive.py` re-encodes Lane A's masks with no training; verified zero references to `SegMapTrainer`/`train_segmap`). MM v2's 2.63 FALSIFICATION verdict STANDS unaffected."

**Status**: Lane MM v2 = 2.63 FALSIFIED IS THE FINAL VERDICT. Lane MM hard-argmax grayscale on 3ch-trained renderer is broken. Confirms council EUREKA: need SGD-optimized soft grayscale (Lane AL) OR joint-trained renderer (Lane SC++) — bolt-on doesn't work.

## Defect 1 (continued): 5 MORE lanes invalidated by Council A's bug

Council A (DARTS-S freeze) listed 4 invalidated lanes: SC++, SA-v2, SO, MM v2. Round 6 found Council A UNDERCOUNTED — 5 more lanes ALSO use SegMapTrainer and ARE invalidated:

- **Lane WC-S** (Curator outlier weighting via SegMap) — invalidated
- **Lane PA** (PixelArt SegMap variant) — invalidated
- **Lane HM-S** (Homography 8-DOF SegMap) — invalidated
- **Lane FR-Ω** (Fridrich-cost Block-FP SegMap variant) — invalidated
- **Lane FC** (FiLM-Canvas SegMap) — invalidated

(Round 6 verifies via grep: each of these has scripts/remote_lane_*.sh that invokes train_segmap.py with --variant kl_distill or hessian_quant, both of which use SegMapTrainer.train_epoch and hit the .round() zero-gradient path.)

**Total invalidated lanes**: 4 (Council A) + 5 (Round 6 correction) = **9 lanes** all silently produced never-trained checkpoints. Their reported scores are EITHER:
- The initial random-init renderer's score (if archive build succeeded)
- N/A (if the rc=1 crash terminated before archive build)

**Re-validation needed**: each of these 9 lanes must be re-run AFTER the Council A .round() fix lands AND with the Council C OOM-class deep fix (bf16 + scorer-chunk) for SegMap-class lanes that hit OOM.

**Lane G v3 STILL UNAFFECTED**: uses train_distill.py (which uses Uint8STE correctly via simulate_eval_roundtrip in renderer.py:1884 — the manual STE pattern). 1.05 [contest-CUDA] verdict STANDS.

## Defect 2 (whitelist foot-gun): scorer.py compute_proxy_score

`src/tac/scorer.py:compute_proxy_score` is whitelisted in Check 86 with comment "read-only" but the function body has bare `.round()` at lines 342 + 350 OUTSIDE the `with torch.no_grad():` block (which only starts at line 357). Today no caller passes requires_grad=True so the bug doesn't bite, but a future TTO loop using compute_proxy_score for gradient signal would silently get zero gradients — exactly the bug Check 86 exists to prevent.

**Fix landed (this loop)**: replaced both `.round().clamp(0, 255)` calls with `Uint8STE.apply(...)`. Forward unchanged (clamp+round); backward gradient flows correctly via the STE.

**Optional follow-up**: refine Check 86 to NOT whitelist a file just because it's read-only-by-design — the bug class survives even in "measurement" code if a caller accidentally enables grad. Better: scan for `.round()` outside `with torch.no_grad():` blocks specifically, regardless of file role.

## Defect 3 (test vacuity): test_segmap_trainer_train_epoch_loss_finite

`test_segmap_trainer_train_epoch_loss_finite` asserts `pre_param != post_param` which passes vacuously via AdamW weight-decay shrinkage (~0.99996/step) even when ALL gradients are zero. The test would NOT have caught the .round() bug — it took 5h on Vast.ai 4090 to surface.

**Fix needed (next loop)**: add `assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters() if p.requires_grad)` AFTER first `optimizer.step()`. This guarantees the gradient chain is intact.

**Generalize**: every training-loop test in src/tac/tests/ that asserts "params changed" should ALSO assert "grads are nonzero" — the latter is the actual contract.

## Cross-refs

- Council A report: .omx/research/council_darts_s_freeze_audit_20260429.md (UNDERCOUNTED invalidation list — by 5 lanes)
- Council Round 6 report: .omx/research/council_round6_adversarial_20260429.md (correction source)
- Memory project_lane_uniward_v8_NO_OP_finding_20260429.md (corrected here for Lane MM v2 claim)
- Memory feedback_check_86_eval_roundtrip_round_zero_gradient_20260429 (forthcoming)
- Lane G v3 unaffected baseline: project_lane_g_v3_landed_1_05_20260428.md
