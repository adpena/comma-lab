# Current Focus — 2026-04-10 22:00 CDT

## Floor
- **Official score**: 1.33 (authoritative scorer, dilated h=64 from Modal A10G)
- **Previous floor**: 1.51 (standard h=64)
- **Leaderboard #1** by 0.55

## Active training fleet (all free, 5 lanes)
| Lane | GPU | Epoch | Scorer | Notes |
|------|-----|-------|--------|-------|
| Local MPS | Apple Silicon | ~190 | 1.452 | Standard h=64 v5 |
| bat00 RTX 2070 | CUDA 8GB | ~0 | — | Just launched, autocast fp16 |
| Modal A10G | NVIDIA A10 | 759 | 0.940 | h=96 standard |
| Modal Dilated | NVIDIA A10 | — | — | Dilated h=64 fallback |
| Lightning T4 | NVIDIA T4 | ~53 | 1.455 | Standard h=64, 79h/mo free |

## tac v0.8.0 — battle hardened
- 4 bug sweep rounds: 15+ bugs found and fixed
- 61 tests, ruff clean, pydantic models
- uint8 compliance, atomic saves, signal handlers
- Portable: works on MPS, CUDA, Modal, Colab, Lightning
- Optimized bundle: 243MB vs 6GB clone

## Ready to deploy
- KL distill loss (Hinton SegNet attack)
- Pair-aware 6ch architecture
- Nuclear H100 deploy
- Video comparison tool for writeup

## This session
- PROMOTED dilated h=64 from Modal: 1.51 → 1.33 (0.18 improvement)
- Compliance re-run confirmed: bit-identical 1.33 on clean dir
- tac v0.9.0: 8 council findings addressed, 66 tests
- Inflate optimized: batched inference, timing, redundant uv removed
- Writeup updated to 1.33, tone pass (no boastful language)
- Mac-mini x86_64 compliance eval set up (torch version issue on Intel Mac)
