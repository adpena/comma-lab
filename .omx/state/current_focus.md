# Current Focus — 2026-04-09 22:25 CDT

## Floor
- **Promoted**: 1.727 (h=64 standard, ep 918, scorer 3.547)
- **Saliency-fixed run**: ep 1090, scorer 3.5011 — BELOW promoted floor
- **Estimated new score**: ~1.70 (needs proxy confirmation)
- **Leaderboard #1**: 1.89 (neural_inflate). Our lead: 0.163+

## What changed this session
- Fixed saliency bug (H1): reconstruction loss now frame 1 only (SegNet's frame)
- Fixed 10 total bugs across 8 review rounds (3 CRITICAL, 7 HIGH)
- tac library v0.6.0 shipped (7 modules, 12 architectures, 7 SegNet interventions)
- Council identified saliency recon as SegNet suppressor (98.4% headroom blocked)
- Writeup draft completed, paper outline done, portfolio plan done

## Training overnight
- h=64 standard saliency-fixed: ep 1090+, grinding to 2500
- Modal h=96: ~3.5h on A10G, results save at completion
- Experiment runner monitoring with auto-restart
- Fleet Monitor sending persistent notifications

## Tomorrow morning priorities
1. Check h=64 best checkpoint scorer
2. Proxy-score it (on local Mac or tertiary)
3. If proxy confirms → new promoted floor
4. Check Modal h=96 results (download from volume)
5. Launch sal_lambda=0 as bounded side lane
6. Start writeup polishing
