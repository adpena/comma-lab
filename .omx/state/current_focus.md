# Current Focus — 2026-04-10 13:00 CDT

## Floor
- **Official score**: 1.52 (CPU)
- **Leaderboard #1** by 0.37

## Active training fleet (all free)
| Lane | GPU | Epoch | Scorer | Notes |
|------|-----|-------|--------|-------|
| Local MPS | Apple Silicon | 119 | 1.459 | Standard h=64 |
| Lightning T4 | NVIDIA T4 | 1 | 1.478 | Standard h=64, 79h/mo free |
| Modal A10G | NVIDIA A10 | 741 | 0.940 | h=96 standard |
| Colab T4 | NVIDIA T4 | ~0 | — | Standard h=64, 12h/wk free |

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
