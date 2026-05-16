# L5-v2 autopilot ranker test sync - 2026-05-16

Scope: `src/tac/tests/test_autopilot_dispatch_ranking.py`.

Finding: the ranker tests still expected TT5L `sideinfo_consumed=false`, but the
current L5-v2 state has a local parser/inflate consumption proof. Rank reward is
still correctly suppressed because paired baseline and empirical anchors are
missing; the stale assertion blurred consumption proof with rank authority.

Change: tests now assert `sideinfo_consumed=true` while keeping
`prediction_band_rank_reward_suppressed` and exact-anchor blockers as the
authority boundary. The source-scope assertion also checks the AC-state
demotion phrase so paper-fidelity regressions are caught in the ranker surface.

Verification: `test_autopilot_dispatch_ranking.py` passes locally.
