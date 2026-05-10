# Current score-lowering roadmap (2026-05-10)

## Evidence anchors

- A1 `[contest-CPU GHA Linux x86_64]`: `0.19284757743677347`,
  archive bytes `178262`, archive SHA-256
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`.
- A1 paired `[contest-CUDA T4]`: `0.2263520234784395`.
- Do not describe the A1 CPU anchor as CUDA-frontier, gold-equivalent, or
  submission-ready until paired CUDA policy is satisfied.
- Current theoretical-floor planning anchor remains approximate:
  `0.140 ± 0.012`; lower movement requires byte-closed substrate/training work,
  not MPS or macOS advisory promotion.

## Dispatch state snapshot

- `tools/claim_lane_dispatch.py summary --format json` at
  `2026-05-10T07:30:49Z`: `active_count=0`,
  `stale_nonterminal_count=0`.
- Recent `track1_phase_a1_score_gradient` rows from `2026-05-10T07:28Z`-
  `07:30Z` are terminal dry-run or failed-before-remote-submission rows. They
  do not authorize CUDA work and do not carry score evidence.
- Any new remote/GHA/CUDA work below needs a fresh active claim row before
  launch and a terminal row on success, failure, stop, or refusal.

## Score-lowering action matrix

| Workstream | Status | Exact next action | Evidence and claim gate |
|---|---|---|---|
| A1 CUDA readiness | READY as evaluator/custody baseline; BLOCKED as a new score-lowering config | Use the paired A1 CPU/CUDA anchor to sanity-check future archive candidates. Do not spend on another baseline replay unless it validates infrastructure drift. | For any A1-derived candidate: changed archive bytes/SHA, runtime-tree SHA, exact `inflate.sh` closure, `contest_auth_eval.json` with recomputed component score and `n_samples=600`, hardware tag, logs, and terminal dispatch claim. |
| PR101 archive-in-loop A1 training | READY-TO-CLAIM after current WIP/preflight is the code under test | Claim `track1_phase_a1_score_gradient`, run the PR101 source-backed remote driver, build `best_proxy`/`final_ema` archives in-loop, then exact CUDA-eval only byte-closed candidates. | Required artifacts: `archive_builds_manifest.json`, selected archive/inflate path, archive bytes/SHA, PR101 source/archive custody, `canonical_score_source=score_recomputed_from_components`, `avg_posenet_dist`, `avg_segnet_dist`, `rate_unscaled`, `archive_size_bytes`, `n_samples=600`, logs, active then terminal claim. Checkpoint-only non-smoke runs are refused. |
| T1 Phase 1 Ballé end-to-end | READY for claimed training-only CUDA; BLOCKED for score promotion | If GPU is used, claim lane `t1_balle_128k_endtoend`, copy the claim ledger to remote, set `T1_ALLOW_SCORE_DOMAIN_TRAINING=1`, `LOCAL_CUDA_WORKER=1`, and run `scripts/remote_lane_t1_balle_endtoend.sh`. | Score promotion remains blocked until auth-eval custody is wired, packet-local runtime/export closure is re-proved, rate-tight state-dict format is selected, exact CUDA auth eval exists, and the claim lifecycle closes terminally. Training artifacts alone are `score_claim=false`. |
| HNeRV / PR95 / PR101 parity | READY as a gate; BLOCKED for lanes missing the gate | Apply the 13 HNeRV lessons before any representation dispatch: score-aware substrate, export-first archive grammar, eval-roundtrip/YUV6 differentiability, runtime closure, no-op proof, and exact CUDA anchor. | Every lane must declare the 8 archive-grammar fields, prove consumed bytes changed, and carry lane-specific dispatch claims. PR95/PR101 reproduction is useful only when it produces a packet, not just a checkpoint or proxy curve. |
| CMA-ES / Optuna PR101 byte search | READY for no-score local materialization; BLOCKED for exact dispatch | Stop broad optimizer churn. First rerun the best known PR101 CodecOp params with `--materialized-payload-output-dir` and a parser-proven payload contract, then substitute only if emitted bytes bind to a PR101/PR106 section. | Existing reports are CPU-prep only: no materialized payload paths, `score_claim=false`, `ready_for_exact_eval_dispatch=false`. Dispatch requires materialized payload path/SHA/contract, byte-closed archive, runtime parity/no-op proof, exact CUDA auth eval, and a fresh lane claim. |
| Xray / mechanism work | READY diagnostic; BLOCKED for score claims | Use `tools/xray_archive_section_entropy_heatmap.py` and the loader-drift discriminator to choose small, reviewable A1/PR101/PR106 section edits and to separate decode/input drift from forward/kernel drift. | Diagnostics require heatmap/probe JSON, section offsets/SHA, and explicit non-promotable labels. Any candidate generated from this work must graduate through archive bytes/SHA, consumed-byte proof, exact eval, and dispatch claim gates. |
| Preflight timing | DEFERRED as score-lowering work | Keep the current strict guard surface; revisit only if developer preflight exceeds the sub-30s budget or a repeated failure class appears. | Timing JSON is DX/custody evidence, not score evidence. Do not trade away strict dispatch/auth-eval guards for speed. |

## P0: local score-lowering and custody

1. Retire the measured A1 per-pair latent sidecar `proxy_mse` packet.
   - Archive candidate: `178316` bytes,
     `c7f3d88e1ad23bf8cda987583e702ac57e293b64bc7bfea77902e835d19cea10`.
   - Local packet proof completed: 600/600 scalar-equivalent records, exact
     `inflate.sh` smoke, strict pre-submission compliance, and live claim
     binding.
   - Exact `[contest-CPU GHA Linux x86_64]` dispatch result:
     `0.20962552129271272`, worse than A1 baseline
     `0.19284757743677347`; measured configuration retired.
   - Reactivate only with score-domain or joint SegNet/PoseNet sidecar search,
     not `proxy_mse` pair selection.
2. Keep A1 bias-coordinate work bounded.
   - Existing broad variants regressed or failed to beat V1.
   - Reopen only with a small reviewed candidate set and CPU-positive evidence.
3. Keep AVVideoDataset CUDA-path discriminator closed as CPU-only unresolved
   unless a fresh CUDA-capable claim is filed.

## P1: substrate recovery and HNeRV parity

1. Reproduce PR95/PR101/PR103 mechanics with exact custody:
   eval-roundtrip-in-training, differentiable YUV path, runtime constants,
   EMA/export discipline, and archive build in the loop. PR101/A1 now exports
   `best_proxy` and `final_ema` archive candidates during non-smoke training;
   next score-lowering step is claimed exact CUDA eval on selected candidates.
2. Reactivate Track4 only through score-gradient/cliff-aware saliency or a
   stronger criterion. The old UNIWARD/STC/Hessian measured configs remain
   negative and are not exact-eval candidates.
3. Rebuild A5 only around score-domain channel allocation or q-bit noise during
   training. Scalar/global splits are exhausted.

## P2: high-upside architecture lanes

1. Phase1/T1 stays local until auth-eval custody is wired. It now emits a
   packet-local three-member runtime with stricter no-op proof; remaining
   blockers are exact CUDA evidence, dispatch-claim lifecycle, and rate-tight
   state-dict wire format. No blind GPU dispatch.
2. Lane12-v2 stays local until it has hermetic runtime, scorer-preprocess
   gradcheck, PR95/PR100 parity or deviation record, packet builder, and dual
   exact-eval readiness.
3. Phase2/T15/T17/T18/T9 stay deferred until Phase1 or a single-axis substrate
   produces a validated exact anchor.

## Status corrections

- MPS is advisory only for sweeps, curves, configuration discovery, and training
  starts; never for auth eval promotion.
- `[contest-CPU]`, `[contest-CUDA]`, local macOS CPU, and MPS proxy evidence
  must remain separate.
- A1 sidecar `proxy_mse` packet passed local custody but regressed on exact
  `[contest-CPU]`; do not redispatch this measured packet.
- No active dispatch claims remain after closing the AV discriminator CPU-only
  run as unresolved for CUDA and closing the A1 sidecar regression claim.
- A1/PR101 training-time work is now export-first: checkpoint-only non-smoke
  runs are refused unless a PR101 source tree is available to emit archive
  candidates before score-bearing follow-up.

## 2026-05-10T08:03Z supersession addendum

- The dispatch snapshot above is stale. Current
  `tools/claim_lane_dispatch.py summary --format json` shows one active claim:
  `track1_phase_a1_score_gradient_modal_20260510T0738Z_codex` on Modal, call id
  `fc-01KR8D302GXGKGT49ETYMA0BZC`, predicted ETA `2026-05-10T11:37:03Z`.
  Latest recover poll returned `NOT READY`; do not duplicate this A1 lane.
- A1 Modal recover is now fail-closed for score claims: harvested results print
  `[contest-CUDA]` only after `tac.auth_eval_schema` reports zero blockers, and
  `harvest_summary.json` preserves claim blockers, sample count, score axis,
  evidence semantics, and promotion flags.
- `scripts/kaggle_check.py` now has both status and log timeouts. Kaggle remains
  a private proxy/config-search substrate only; it is not exact-eval evidence.
- Worker A landed a private Kaggle PR101 proxy-sweep builder. It generates
  `experiments/kaggle_kernels/pr101_proxy_sweep/` and an operator-controlled
  `uv run --with kaggle kaggle kernels push -p ...` command, but it does not
  push, launch, emit archives, or claim scores.
- Worker C landed a T1 Modal actuator in
  `experiments/modal_t1_balle_endtoend.py`. It is plan-only by default, requires
  `modal run ... --execute` for spend, opens a lane claim before `.spawn()`, and
  accepts score evidence only through strict 600-sample contest-CUDA auth-eval
  schema.
- Provider readiness cache at `2026-05-10T07:55:53Z`: Modal ready for exact
  CUDA but currently occupied by the active A1 claim; Kaggle ready as proxy
  only; AWS blocked by expired auth; GCP blocked by disabled billing; Azure
  blocked by auth.
- Red-team follow-up fixed three dispatch blockers before any new spend:
  future A1 Modal dispatches now use a 6h Modal function timeout with stage
  budget validation, A1 recover returns nonzero for non-claimable `rc=0`
  harvests, and T1 Modal cost planning is tied to the actual 24h Modal function
  timeout rather than a misleading requested timeout.
