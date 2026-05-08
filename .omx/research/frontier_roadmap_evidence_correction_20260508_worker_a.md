# Frontier Roadmap Evidence Correction - Worker A - 2026-05-08

Scope: adversarial review of a roadmap claiming PR106 contest frontier
`0.20454`, public medal band `0.193`/`0.195`, CPU-prep anchors at
`150460`, `148378`, `148494`, `137469`, `153671` bytes, in-flight
`arch_shrink` Lightning, and Tier-A CUDA dispatches.

Evidence grade of this review: `evidence_audit_no_score`. Score claim: false.
Dispatch performed: false. No GPU job was launched or claimed.

## Verification Basis

- Current branch observed with `git status --short --branch`: `main`.
- Exact score artifact inspected:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`.
- Public PR reproduction ledger inspected:
  `experiments/results/pr100_107_reproduction_ledger_20260507_codex/ledger.json`.
- CPU-prep manifests inspected:
  - `reports/raw/pr106_lagrangian_per_tensor_allocation_20260508T071433Z/manifest.json`
  - `experiments/results/unified_winners_stack_20260508T071803Z/build_manifest.json`
  - `reports/raw/pr101_cross_paradigm_hstack_vstack_corrected_20260508/manifest.json`
  - `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/build_manifest.json`
- Monolithic layout reverified with:
  `.venv/bin/python tools/pr106_archive_decomposition.py --summary-text --output-json /tmp/worker_a_frontier_layout_check_20260508.json`
- Layout regression reverified with:
  `.venv/bin/python -m pytest src/tac/tests/test_frontier_archive_layout.py -q`
  (`3 passed`).

## Corrected Evidence Matrix

| Roadmap claim | Current artifact evidence | Correct status | Risk correction |
| --- | --- | --- | --- |
| PR106 contest frontier is `0.20454` | The `0.20454327743640793` value appears in `reports/autopilot_plan_post_session_refresh_20260508.json` and `.omx/research/autopilot_post_session_refresh_planning_memo_20260508.md` as an operator-state formula projection using `B=178873`, `d_seg=0.00067`, `d_pose=0.000034`. I found no matching exact CUDA auth-eval JSON for that score. | Stale/unsupported as a contest frontier. | Replace with the current exact local HNeRV A++ anchor: PR103-on-PR106 at canonical auth-eval score `0.20898105277982337`, archive `185578` bytes, SHA `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`, T4 CUDA, `600` samples. The tracked strict-formula snapshot reports the same anchor as `0.2089810755823297`; both round to `0.20898`, not `0.20454`. |
| Public medal band is `0.193`/`0.195` | `experiments/results/pr100_107_reproduction_ledger_20260507_codex/ledger.json` records public leaderboard scores: PR101 `0.193`, PR100/102/103 `0.195`. The same ledger records local CUDA replay drift for public artifacts: PR101 exact `0.22635331443973267`, PR100 exact `0.22826947142244708`, PR103 public artifact exact `0.2277649714224471`; PR102 has no exact CUDA artifact in that ledger. | Supported only as `external` public-pressure context. | Do not use `0.193`/`0.195` as local exact score truth. It can motivate urgency, but promotion/ranking inside this repo still uses exact CUDA auth eval on exact archive/runtime custody. |
| PR106 UNIWARD `150460` B at about 5 percent rel_err | `reports/raw/pr106_lagrangian_per_tensor_allocation_20260508T071433Z/manifest.json` has `lagrangian_uniward.archive_bytes=150460` at `rms_target=0.05`, `rel_err=0.04663926433466547`. Its metadata says `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`. | Real CPU-prep byte anchor, not a score or deployable contest candidate. | Blockers remain: `byte_rel_err_proxy_only_no_score_test`, no runtime dequantize path for the modified decoder, missing exact CUDA auth eval, fixed scales during K sweep, no iterative primal-dual ADMM consensus, and lossless latent/sidecar assumptions. |
| PR101 unified Stage 1+2 `148378` B and Stage 1+2+3 `148494` B | `experiments/results/unified_winners_stack_20260508T071803Z/build_manifest.json` records Stage 1+2 `148378` B, SHA `fc539f935641e049f5cae443af930fbdbbec703103439abc45b2abf3a602ed13`, and Stage 1+2+3 `148494` B, SHA `45dd64d41ede9ec6dd74c82572996228face37f6184dd3ab5aa96aad7405ec06`. It also records `score_claim=false` and `ready_for_exact_eval_dispatch=false`. | Real CPU-build byte-closed anchors, not contest-CUDA evidence. Stage 1+2+3 is larger than Stage 1+2, so it is not a monotonic byte win. | Treat as a packet-construction lead only. Blockers include CPU byte proxy, no harvested exact CUDA JSON, UNIWARD variance proxy substituting for wavelet residual, no iterative ADMM consensus, and no score-aware per-tensor distortion loop. |
| Cross-paradigm decoder-only `137469` B | `reports/raw/pr101_cross_paradigm_hstack_vstack_corrected_20260508/manifest.json` headline/dominant stack records `Path_B_step6_ADMM_x_continuous_K_then_Op1` at `137469` B, `achieved_rel_err=0.0415393353487541`, with `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`. | Real substrate-corrected CPU-prep byte anchor, not a deployable archive or exact score. | Do not call it a contest archive until a full monolithic packet is rebuilt, the runtime consumes the changed section bytes, old/new archive and section SHAs are recorded, and exact CUDA auth eval lands. The manifest's own blockers include `byte_proxy_only_no_score_test`, `no_real_archive_substrate_for_cuda_replay`, and `missing_exact_cuda_auth_eval`. |
| ADMM no-dead-K `153671` B | `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/build_manifest.json` records archive `153671` B, SHA `b7b09089e852872bd67b4b8aa04c1b4d46168bb89343acff81796c5551d63d05`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`, `dispatch_attempted=false`. | Real CPU-build byte-closed anchor, not a score claim. | It is useful as an implementation ingredient, but cannot rank, promote, kill, or justify dispatch without exact CUDA and component review. |
| In-flight `arch_shrink` Lightning | `.omx/state/active_lane_dispatch_claims.md` has an active claim for `arch_shrink_x0.4_lightning` / `arch-shrink-x0-4-lightning-20260508T024304Z`. `.omx/research/arch_shrink_x0_4_lightning_review_20260508_worker_a.md` and `.omx/research/arch_shrink_lightning_live_strategy_20260508_worker_l3.md` report SDK status `Running`, current heartbeat/training, and no terminal `archive.zip`, `contest_auth_eval.json`, or `auth_eval.log` as of their polls. | In flight, not harvested. | Do not duplicate dispatch. Treat as live training/checkpoint signal until terminal exact artifacts land. If it lands, perform the full result-review packet before any score use. |
| Tier-A CUDA dispatches | The roadmap's Tier-A language mixes already-landed score evidence, exact-negative/reactivation work, and CPU-prep candidates. Current manifests for the listed CPU-prep candidates all say `ready_for_exact_eval_dispatch=false`. | Unsafe if interpreted as immediate new GPU dispatch. | The corrected Tier A is: harvest/adjudicate the active `arch_shrink` job; keep already-landed exact CUDA anchors as anchors; do not dispatch cross-paradigm/ADMM/UNIWARD CPU rows until monolithic packet closure, runtime-consumption proof, Level-2 claim, and static compliance are satisfied. |

## Monolithic Packet Correction

Fresh layout verification confirms PR101 and PR106 are single-member HNeRV
packets at the ZIP layer:

- PR101 archive:
  `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`,
  `178258` bytes, SHA
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`,
  member `x`, parser sections `decoder_blob` `162164` B,
  `latent_blob` `15387` B, and `sidecar_blob` `607` B.
- PR106 archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`,
  `186239` bytes, SHA
  `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`,
  member `0.bin`, parser sections `ff_header` `4` B,
  `decoder_packed_brotli` `170278` B, and
  `latents_and_sidecar_brotli` `15849` B.

Therefore, roadmap language that assumes separate PR101/PR106 ZIP members for
masks, poses, renderer, or other component budgets is invalid. Future stack
work must either mutate parser-proven internal sections with section SHA custody
or create a new charged sidecar/runtime packet and prove runtime consumption.

## Actionable Corrections

1. Baseline all current HNeRV local floor calculations on `0.20898105277982337`
   canonical auth-eval score (`0.2089810755823297` strict formula snapshot),
   `185578` charged bytes, and archive SHA
   `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`.
2. Label `0.20454` as a stale formula projection unless a matching exact CUDA
   auth-eval JSON with archive/runtime custody is produced.
3. Preserve public `0.193`/`0.195` as `external` leaderboard context only; do
   not merge it into exact local frontier rank.
4. Keep all listed CPU-prep anchors in the validation queue with
   `score_claim=false` and `ready_for_exact_eval_dispatch=false` until they
   become full monolithic packets with runtime-consumption proof and exact CUDA.
5. Do not dispatch new GPU jobs from this roadmap as written. The only active
   GPU lane found here is the existing `arch_shrink_x0.4_lightning` claim, which
   should be harvested and adversarially reviewed when terminal artifacts exist.
