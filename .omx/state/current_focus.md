# Current Focus — 2026-04-09 16:20 CDT

## Floor
- **Promoted**: 1.727 (h=64 standard, 45.6KB int8)
- **Leaderboard #1**: 1.89 (neural_inflate). Our lead: 0.163.

## Active training
- Standard h=64 long-2500: ep 143, scorer 3.928
- SegNet boundary h=64: ep 4, scorer 4.46 (council's highest-leverage)
- bat00 h=96: installing PyTorch on RTX 2070 Super

## Key pivot
- Killed PSD/PixelShuffle (proxy reject 1.99, hurts PoseNet)
- Boundary attack targets 2.39% of pixels where SegNet can flip
- Tao: seg 0.006→0.003 alone saves 0.276 points

## Next
- Monitor training convergence
- Proxy-score first candidate under 3.55
- Deploy site to Cloudflare Pages
- Submit PR for current 1.727 floor
