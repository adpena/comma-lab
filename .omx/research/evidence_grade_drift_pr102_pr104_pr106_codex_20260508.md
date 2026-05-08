# Evidence-grade drift audit after PR102/PR104/PR106 replays - Codex - 2026-05-08

Scope: dated `.omx/research/` ledgers and small report/status surfaces only.
No score artifacts, generated experiment directories, dispatch claims, or
archive/runtime files were changed. This ledger supersedes stale wording; it
does not rewrite historical ledgers in place.

Evidence grade of this audit: `evidence_audit_no_score`.
Score claim: false. Dispatch performed: false.

## Current exact anchors

| Subject | Current evidence | Correct use |
| --- | --- | --- |
| Active local HNeRV rate anchor | PR103-on-PR106 exact T4 CUDA, score `0.20898105277982337` (`0.2089810755823297` strict rounded-component snapshot), archive `185578` bytes, SHA-256 `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`, artifact `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json` | Current local A++ HNeRV anchor. Future rate-only HNeRV candidates must compare here unless they explicitly target a narrower PR106-control question. |
| PR102 hardened replay | Exact T4 CUDA, score `0.22839372989108092`, archive `178981` bytes, SHA-256 `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`, artifact `experiments/results/lightning_batch/pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z/contest_auth_eval.adjudicated.json` | A++ replay/custody evidence. Confirms public CPU/leaderboard drift; not a new archive-byte win and not a local frontier. |
| PR104 hardened replay | Exact T4 CUDA, score `0.23113446620399658`, archive `178637` bytes, SHA-256 `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8`, artifact `experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/contest_auth_eval.adjudicated.json` | A++ replay/custody evidence. Closes the previous PR104 local replay hole; not a local frontier. |
| PR106 UNIWARD-Lagrangian rms=0.05 | Exact T4 CUDA, score `0.3371617511972341`, archive `150511` bytes, SHA-256 `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`, artifact `experiments/results/lightning_batch/pr106-uniward-rms005-exact-20260508T083555Z/contest_auth_eval.adjudicated.json` | `A-negative scoped forensic` for the measured runtime packet/config only. No promotion, no rank-frontier use, no family kill. |

## Statements superseded

| File reviewed | Exact stale statement or field | Supersession |
| --- | --- | --- |
| `.omx/research/public_frontier_drift_adversarial_review_20260508_codex.md` | `PR104 remains an evidence hole.` and `Downgrade "PR104 drift resolved" to "unresolved; public CUDA comment only"` | PR104 is now harvested as exact T4 CUDA at `0.23113446620399658` with archive SHA `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8`. The old note remains valid only for the pre-harvest snapshot. |
| `.omx/research/frontier_roadmap_evidence_correction_20260508_worker_a.md` | `PR102 has no exact CUDA artifact in that ledger` and `PR106 UNIWARD 150460 B ... Real CPU-prep byte anchor, not a score or deployable contest candidate.` | PR102 now has a structured exact replay JSON at `0.22839372989108092`. The PR106 UNIWARD CPU-prep row became a byte-closed `150511` byte runtime packet and then exact A-negative at `0.3371617511972341`; the CPU-prep row is still non-score, but the measured packet is now an exact negative for that config. |
| `.omx/research/pr104_exact_replay_dispatch_status_20260508_codex.md` | Initial `Status: QUEUED`, `NO_SCORE_CLAIM`, plus `ARTIFACT_NOT_READY` evidence boundary | Superseded by the same ledger's `2026-05-08T11:48Z Harvested Exact CUDA Result` section: PR104 exact T4 CUDA score `0.23113446620399658`, promotion-review/rank-candidate allowed for that exact replay artifact, non-frontier versus PR103-on-PR106. |
| `.omx/research/autopilot_post_session_refresh_planning_memo_20260508.md` | `Operator state: PR106 frontier (d_seg=6.7e-4, d_pose=3.4e-5, B=178,873 -> score 0.20454)` | Treat `0.20454` / `178873` as an unanchored formula projection, not contest-CUDA score truth. Current local HNeRV score truth is PR103-on-PR106 at `0.20898105277982337` and `185578` bytes. |
| `reports/autopilot_plan_post_session_refresh_20260508.json` | `current_score_baseline` / `current_score` fields equal `0.20454327743640793` | Generated planning output is stale for score anchoring. Do not use this JSON to promote, rank, or dispatch without regenerating from the PR103-on-PR106 anchor and exact evidence semantics. |
| `.omx/research/tier_a_cuda_dispatch_packets_prestaged_20260508.md` | PR106 UNIWARD dispatch snippets use `--baseline-score 0.20454` and describe `[CPU-build]` pending exact CUDA | The exact CUDA result has landed and is A-negative. The stale `0.20454` baseline was too optimistic; the packet is a regression versus both PR106 exact replay (`0.20945673680571203`) and active PR103-on-PR106 (`0.20898105277982337`). |
| `.omx/research/pr106_uniward_runtime_packet_dispatch_repair_20260508_codex.md` | `Adjudication baseline: PR106 0.20454` in the running-job section | Superseded by the ledger's exact-result section and by `.omx/research/pr106_uniward_lagrangian_exact_cuda_regression_20260508_codex.md`; use `A-negative scoped forensic`, not baseline-predicted promotion semantics. |
| `.omx/research/recursive_adversarial_greenup_20260508_worker_b.md` | PR106 UNIWARD `remains a CPU-build/proxy artifact until exact CUDA auth eval lands` | Exact CUDA landed. The correct current state is exact A-negative scoped forensic for rms=0.05, still non-promotable and non-killing. |
| `reports/latest.md` | `The PR100 adapter replay is the current score champion.` | Superseded in the status report by PR103-on-PR106 as active local HNeRV anchor. PR100 is historical/submission-packet context. |
| `reports/latest.md` | Report pipeline `frontier_summary` says `current default is the PR100 adapter replay` | Superseded in the status report by PR103-on-PR106 as current default exact HNeRV anchor; PR102/PR104 are exact replay context rows, not frontier rows. |
| `reports/latest.md` | Predicted bands labeled as `[contest-CUDA]`, including `Lane Ω-W-V3 ... predicted band [0.194, 0.204] [contest-CUDA]` and the intN predicted-band header | Superseded to `[prediction, NOT contest-CUDA]` in the status report. Prediction, CPU, MPS, proxy, and byte-only rows cannot promote, rank, kill, or dispatch without the exact packet gates. |

## Guardrail decisions

1. Exact CUDA positive A++ replay evidence can rank only the exact archive and
   runtime it evaluated. It does not make public leaderboard CPU rows local
   CUDA truth and does not create a stack atom without decode/re-encode or
   compression reproduction.
2. Exact CUDA negative evidence can retire the measured implementation/config.
   It does not kill UNIWARD, Lagrangian allocation, lossy coarsening,
   score-aware weighting, or future scorer/Jacobian-weighted variants.
3. `evidence_grade: "A++"` on raw exact-eval machinery is not enough for
   promotion. Downstream consumers must honor `promotion_eligible`,
   `score_claim_valid`, `paper_claim_grade`, `allowed_use`, `lane_status`, and
   `rank_or_kill_eligible`.
4. `0.20454`, `0.193`, `0.195`, CPU-prep byte anchors, MPS signals, and proxy
   predicted bands are planning/context signals only unless an exact CUDA
   `archive.zip -> inflate.sh -> upstream/evaluate.py` artifact with custody is
   cited.

## Files reviewed

- `.omx/research/pr102_hardened_exact_replay_result_20260508_codex.json`
- `.omx/research/pr104_exact_replay_dispatch_status_20260508_codex.md`
- `.omx/research/pr106_uniward_lagrangian_exact_cuda_regression_20260508_codex.md`
- `.omx/research/public_frontier_drift_adversarial_review_20260508_codex.md`
- `.omx/research/frontier_roadmap_evidence_correction_20260508_worker_a.md`
- `.omx/research/roadmap_state_reconciliation_20260508_codex.md`
- `.omx/research/autopilot_post_session_refresh_planning_memo_20260508.md`
- `.omx/research/tier_a_cuda_dispatch_packets_prestaged_20260508.md`
- `.omx/research/pr106_uniward_runtime_packet_dispatch_repair_20260508_codex.md`
- `.omx/research/recursive_adversarial_greenup_20260508_worker_b.md`
- `.omx/research/adjudicated_json_promotion_stamp_hardening_20260508_codex.md`
- `.omx/research/lossy_coarsening_exact_cuda_adversarial_review_20260508_worker_b.md`
- `.omx/research/lossy_coarsening_exact_cuda_result_review_20260508_codex.json`
- `reports/latest.md`
- `reports/autopilot_plan_post_session_refresh_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_019_reconciled_20260508.json`
- `reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_019_reconciled_20260508.json`

## Edit disposition

Only this ledger and `reports/latest.md` were edited. Historical ledgers remain
append-only evidence records; this file is the dated supersession pointer.
