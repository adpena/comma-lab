# Current Focus — 2026-04-09 14:00 CDT

## Floor
- **Promoted**: 1.727 (h=64 long-1000 QAT+EMA, 45.6KB int8)
- **Leaderboard #1**: 1.89 (neural_inflate). Our lead: 0.163.
- **Target**: 1.45. Days remaining: 24.

## Active training
- PSD h=64 long-1000 v2 — launched, council #1 pick architecture
- Dilated h=64 long-1000 v2 — launched, strongest non-promoted challenger

## Site status
- Pushed to adpena/comma-lab (private)
- CSP fixed (was blocking ALL JS in production!)
- Hero updated with scoring formula
- Gradients stripped (user: "nothing that looks AI-designed")
- Sticky nav + conversation timeline pending integration
- Need Cloudflare Pages deployment

## Infrastructure
- Kaggle API connected (legacy key)
- Modal needs `modal setup` completion
- auto_commit.sh ready
- Conversation timeline extracted (1,360 events, 865KB)

## Next
- Deploy to Cloudflare Pages
- Polish site design (whitespace, symmetry, no AI gimmicks)
- Cross-browser test
- Wire conversation timeline into site
- Proxy-score PSD/dilated when they converge
- Pixelshuffle proxy resolved to `1.99`; keep as a non-promoted alternate reference
