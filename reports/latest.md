# Latest Report — 2026-04-30 PFP16 custody update (3 days to deadline)

## Current Floor (the ONLY scores we report)

**Best contest-CUDA artifact: PFP16 A++ = 1.04 [contest-CUDA A++]** (exact Tesla T4 auth eval on 2026-04-30).

- Evidence: `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json`
- Custody: `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/custody/custody_manifest.json`
- Archive: `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip` (686,635 bytes, SHA `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`)
- SegNet: 0.00400656 / PoseNet: 0.00346442 / Rate: 0.01828808
- Recomputed score: `1.043987524793892`; `contest_auth_eval.json` is authoritative over build provenance and logs.
- Recipe: Lane G v3 renderer + deterministic PFP16 pose payload; old Lane G v3 `1.05` is the historical predecessor.
- Beats Lane A (1.15) by 0.11 and old Lane G v3 by ~0.0049.

**Fallback floor: Lane G v3 pre-PFP16 = 1.05 [contest-CUDA]** (`experiments/results/lane_g_v3_landed/contest_auth_eval.json`).

## Live leaderboard (fetched 2026-04-29 ~10am)

| Rank | Score | Entry | PR | Notes |
|------|-------|-------|----|-------|
| 1 | 0.33 | Quantizr | #55 | FiLM CNN 88K + KL-T2 + AV1 |
| 2 | **0.38** | **Selfcomp** | **#56** | self-compression ~1.017 bpw + analytical-pose affine |
| 3 | **0.60** | **Mask2mask** | **#53** | "slightly different arch" (obfuscated) |
| 4 | 1.89 | neural_inflate | #49 | |
| 5 | 1.91 | svtav1_dilated_renderer | #58 | |
| ours | 1.04 | PFP16 A++ | not submitted | would rank ~4th against this 2026-04-29 snapshot |

User-set NON-NEGOTIABLE goal: **sub-0.30**. Deadline May 3.

## Modal pipeline TRUSTED (canonical for >2h training)

PFP16 A++ supersedes the old Lane G v3 Modal/Vast reproduction for frontier wording. For training jobs >2h, Modal remains the preferred queue. Vast.ai 4090 NVDEC roulette has been ~85% bad-host rate this week; ~$5 burned across 5 dispatch rounds for 0 trained lanes on the bad nights. Modal's slightly higher per-hour cost is dominated by reliability.

## Active portfolio (as of 2026-04-29 PM)

In-flight on Modal (re-dispatched after Round 1 council fixes):
- q_faithful_v3 (true Quantizr 1:1 replica)
- sz_phase2_v2 (dilated moonshot)
- mae_v_v2 (mask-augment)
- lane_w_v2 (hard-pair self-compress)
- Lane MM (grayscale-LUT mask encoding)
- Lane SA (94K-param SegMap clone — Selfcomp paradigm)
- Lane SC++ (SA + KL distill T=2.0 — sub-Quantizr stack candidate)
- Lane SO (SC++ + Hessian-aware block-FP)

In implementation (5 sweep + 5 EUREKA lanes):
- FR-Ω, HM-S, DARTS-S, WC-S, FR-MM (sweep)
- PA, FC, SH, TR, PD (EUREKA bench: Shannon, Ballé, Karpathy, Schmidhuber, Carmack, Toderici, Olah)

## Negative / blocked evidence (recent)

- Lane M-V2 (radial-zoom rank-1 hypothesis): 1.84 [contest-CUDA], regression vs Lane A 1.15. Train/inference pose-pad asymmetry confirmed (Check 42).
- Lane H CRF56: 3.20 [contest-CUDA].
- Lane GP v3 (Gaussian-process pose fit): `89.67` [Modal-T4-CPU diagnostic; invalid for score claims]. Treat as a Runge-condition debugging signal only; it does not disprove the off-manifold hypothesis or retire the polynomial path without lane-local CUDA archive custody.
- Lane UNIWARD v8: `1.14` [Modal-T4-CPU diagnostic; invalid for score claims]. Treat as a suspected no-op/SLI1-decoder debugging signal only; no standalone kill or retirement without exact CUDA archive custody and scoped council review.
- Lane V (Quantizr halfframe joint-from-epoch-0): crashed at ~7.6h on channel mismatch.
- 2026-04-29 Modal first-wave failures: MAE-V missing pydantic; Omega Hessian CUDA assert; UNIWARD missing baseline; pose tensor shape (600,6) vs (N,1) on one inflate.

## Catastrophic-failure protections landed (past 7 days)

| Bug class | Detection | First incident |
|-----------|-----------|----------------|
| 48x64 mask resolution → score 53.61 | Check 76 STRICT (anchor mask resolution) | Lane UNIWARD v7 |
| Wrong archive bytes used in eval | `submission_archive.require_valid_archive()` | Multi-week regression |
| Overlapping pose pairs vs 600 non-overlap | Diff against upstream evaluate.py | Multi-week |
| eval_roundtrip defaulted False | CLAUDE.md non-negotiable, all paths True | TTO/training |
| Auto-bundle by file existence | All archive contents now require explicit flags | Compress.sh |
| Lane GP Runge polynomial blow-up | Surface fit-quality RMSE in RESULT_JSON | Lane GP v2/v3 |
| Vast.ai NVDEC roulette | Pre-DALI NVDEC probe (Stage 0.5) + Modal pivot | Multiple lanes |

Total STRICT preflight checks: **78** (was 36 a week ago). Catalog in CLAUDE.md.

## Verification notes

- `comma-lab doctor`: required local tools present.
- `canonical_local_auth_eval_smoke.py --lane g_v3_corrected_kl_weight --quiet`: PASS (10 stages, 0.02s).
- Focused test slice: 34 passed in 2.00s.
- Check 64 E2E smoke proof scan: 0 violations.

## Caveats

- Upstream snapshot is stale/ambiguous: `comma-lab status` reports snapshot `ec82c291...` from 2026-04-03 while live workspace upstream is `cd64c68...`; root `upstream/` is `11ad728...` with local modifications. Deliberate rebootstrap pending.
- Older "1.04 vs 1.05" Modal-T4 vs Vast-4090 drift is predecessor context only; the PFP16 exact T4 `contest_auth_eval.json` is the current score authority.
- Internal "predicted X.XX" stack projections and Modal/local diagnostics are advisory unless backed by lane-local CUDA `contest_auth_eval.json`, archive SHA/bytes, component recomputation, and custody. PFP16 A++ is the only current frontier anchor.

## Next queue

1. Land the Selfcomp-paradigm portfolio (MM → SA → SC++ → SO).
2. Sub-0.30 frontier: stack SC++ + FR-Ω + DARTS-S; predicted ~0.25 if additivity holds.
3. Submission PR gate: 5-pass clean adversarial review (stricter than the standard 3-pass) before any May 3 push.
4. Strategic-secrecy audit on writeup/site files before any public surface gets the PFP16 A++ details.
