# Latest Report — 2026-04-29 endgame status (4 days to deadline)

## Current Floor (the ONLY scores we report)

**Best contest-CUDA artifact: Lane G v3 = 1.05 [contest-CUDA]** (2026-04-28 verified, reproduced 1.04 on Modal T4 2026-04-29).

- Evidence: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`
- Modal reproduction: `experiments/results/modal_auth_eval_9b20bdfca246.json` (1.04, drift 0.01 on PoseNet)
- Archive: `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip` (694,074 bytes)
- SegNet: 0.00400846 / PoseNet: 0.00345458 / Rate: 0.01848622
- Recipe: KL distill weight=0.002 + pose TTO retry on Lane A anchor
- Beats Lane A (1.15) by 0.10 at the same archive size

**Fallback floor: Lane A = 1.15 [contest-CUDA]** (`experiments/results/lane_a_landed/contest_auth_eval.json`).

## Live leaderboard (fetched 2026-04-29 ~10am)

| Rank | Score | Entry | PR | Notes |
|------|-------|-------|----|-------|
| 1 | 0.33 | Quantizr | #55 | FiLM CNN 88K + KL-T2 + AV1 |
| 2 | **0.38** | **Selfcomp** | **#56** | self-compression ~1.017 bpw + analytical-pose affine |
| 3 | **0.60** | **Mask2mask** | **#53** | "slightly different arch" (obfuscated) |
| 4 | 1.89 | neural_inflate | #49 | |
| 5 | 1.91 | svtav1_dilated_renderer | #58 | |
| ours | 1.04 | Lane G v3 | not submitted | would rank ~4th if we shipped today |

User-set NON-NEGOTIABLE goal: **sub-0.30**. Deadline May 3.

## Modal pipeline TRUSTED (canonical for >2h training)

Modal T4 reproduced the Vast.ai Lane G v3 score within the noise floor (1.04 vs 1.05). For training jobs >2h, Modal is now canonical. Vast.ai 4090 NVDEC roulette has been ~85% bad-host rate this week; ~$5 burned across 5 dispatch rounds for 0 trained lanes on the bad nights. Modal's slightly higher per-hour cost is dominated by reliability — Modal wins on expected $ per successful lane.

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
- Lane GP v3 (Gaussian-process pose fit): 89.67 [Modal-T4-CPU]. Runge phenomenon at degree-10 polynomial; off-manifold hypothesis disproved. Lane GP polynomial path is dead — DCT or B-spline if revived.
- Lane UNIWARD v8: 1.14 [Modal-T4-CPU], identical to Lane A noise floor. Encoder pipeline is no-op on the bitstream without an SLI1 inflate-time decoder. Council 5/5 KILLED standalone (2026-04-29).
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
- ALL "1.04 vs 1.05" Modal-T4 vs Vast-4090 drift is within noise; both are contest-equivalent.
- Internal "predicted X.XX" stack projections in this report are advisory only — only [contest-CUDA] or [Modal-T4-CUDA] tagged scores are real.

## Next queue

1. Land the Selfcomp-paradigm portfolio (MM → SA → SC++ → SO).
2. Sub-0.30 frontier: stack SC++ + FR-Ω + DARTS-S; predicted ~0.25 if additivity holds.
3. Submission PR gate: 5-pass clean adversarial review (stricter than the standard 3-pass) before any May 3 push.
4. Strategic-secrecy audit on writeup/site files before any public surface gets the Lane G v3 details.
