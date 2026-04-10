# Current Focus — 2026-04-10 12:00 CDT

## Floor
- **Official score**: 1.52 (CPU), 1.49 canonical proxy
- **Leaderboard #1** by 0.37 (next: neural_inflate 1.89)

## Active training
- Local MPS: h=64 standard v5, ep ~100, scorer ~1.467
- Colab T4: h=64 standard, first epoch running
- Modal A10G: h=96 standard, ep 607+

## Ready to deploy
- GCP T4: configured, waiting for user to confirm free trial credits
- KL distill loss: implemented, reviewed, validated (Hinton SegNet attack)
- Pair-aware 6ch: implemented, tested
- Nuclear H100: modal_nuclear_deploy.py ready
- Precomputed fast-load: eliminates 10 min decode

## tac v0.8.0
- 61 tests, ruff clean, pydantic models, zero hardcoded paths
- Canonical scorer with uint8 compliance
- Contest mode (eval_holdout=0) vs production mode (0.25)

## Next milestones
1. Confirm GCP free trial credits → deploy T4
2. Let training lanes converge (ep 500+)
3. Canonical score best checkpoints
4. Submit PR as "adpena" when confident
5. Nuclear run (h=96 + kl_distill on H100) in week 3
