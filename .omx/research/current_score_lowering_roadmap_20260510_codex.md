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
| PR103 global-combo histogram packet | BYTE-CLOSED LOCAL ARTIFACT; BLOCKED for exact dispatch | Use the `-12B` global-combo packet as the current PR103 byte target, then build same-runtime source/candidate frame or auth-eval parity before CUDA spend. | Candidate archive `578c8f4e86eafc9dc04eefe61cc0e7f3f3f43e134ef4447cf9ef26fd23a23551`, `178211` bytes, runtime tree `8b81480b74919295c37707ac5124934571314f30d3bfe0164cbe7b456e589936`; blockers: `full_frame_inflate_output_parity_missing`, lane claim, exact CUDA. |
| PR101 archive-in-loop A1 training | READY-TO-CLAIM after current WIP/preflight is the code under test | Claim `track1_phase_a1_score_gradient`, run the PR101 source-backed remote driver, build `best_proxy`/`final_ema` archives in-loop, then exact CUDA-eval only byte-closed candidates. | Required artifacts: `archive_builds_manifest.json`, selected archive/inflate path, archive bytes/SHA, PR101 source/archive custody, `canonical_score_source=score_recomputed_from_components`, `avg_posenet_dist`, `avg_segnet_dist`, `rate_unscaled`, `archive_size_bytes`, `n_samples=600`, logs, active then terminal claim. Checkpoint-only non-smoke runs are refused. |
| T1 Phase 1 Ballé end-to-end | ACTIVE Modal dispatch; BLOCKED for duplicate launch and score promotion | Monitor Modal call `fc-01KR955JSYQAVTTYZA48VAV7WJ` for lane `t1_balle_128k_endtoend`; do not launch a duplicate. Harvest artifacts when it reaches terminal state. | Score promotion remains blocked until auth-eval custody is wired, packet-local runtime/export closure is re-proved, rate-tight state-dict format is selected, exact CUDA auth eval exists, and the claim lifecycle closes terminally. Training artifacts alone are `score_claim=false`. |
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
3. Promote PR103 global-combo only through same-runtime source/candidate
   parity first.
   - Current candidate: `178211` bytes (`-12B` versus PR103 source).
   - Dispatch blocker is engineering correctness, not lack of byte signal:
     full-frame inflate output parity or same-runtime source replay is missing.
   - Next implementation slice is a reusable same-runtime comparator, not a
     blind CUDA rerun.
4. Keep AVVideoDataset CUDA-path discriminator closed as CPU-only unresolved
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

1. Phase1/T1 is active on Modal as of this update. Treat it as one outstanding
   harvest, not an invitation to launch another provider copy. When terminal,
   classify artifacts by archive SHA/runtime tree/component fields before any
   status promotion.
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

## 2026-05-10T09:10Z supersession addendum

- The `2026-05-10T08:03Z` active A1 claim is now terminal. Recovered exact
  Modal T4 `[contest-CUDA]` evidence for
  `track1_phase_a1_score_gradient_modal_20260510T0738Z_codex` is a measured
  regression, not a score-lowering candidate:
  score `0.5447505505333358`, seg `0.00336345`, pose `0.00050645`,
  archive bytes `206110`, archive SHA-256
  `f5d04f22d46bc1c4b863e9e2989c25f9b04e07cb21d54980b5effb654edc127a`,
  runtime tree SHA-256
  `fae77695921cd2a6c948cbd85d0e720b9a08d3a6e64c85f4a79f44dd579e6fa2`.
  See `a1_modal_score_gradient_regression_20260510_codex.md`.
- Current active claim is now only `t1_balle_128k_endtoend` on Modal,
  instance/job id `t1_balle_modal_guard_a3311268_20260510T0831Z`, call id
  `fc-01KR8GACB3NCW5TNG1E9YFPXHM`. Recover with:
  `.venv/bin/python experiments/modal_t1_balle_endtoend.py recover --label t1_balle_modal_guard_a3311268_20260510T0831Z`.
- T1 guard is intentionally bounded: `50` epochs, batch size `8`, max target
  pairs `64`, `score_claim=false`, `promotion_eligible=false`. It may update
  the Phase 1 trust region, but it cannot promote without exact CUDA auth-eval
  schema blockers at zero, paired CPU-axis policy, lane-registry promotion
  clearance, and operator submission policy clearance.
- Immediate score-lowering priority while T1 runs is not another duplicate GPU
  dispatch. The next unblocked work is local/proxy candidate generation with
  explicit boundaries:
  1. harvest T1 when ready and close the active claim terminally;
  2. run Kaggle/Optuna/CMA-ES only as proxy configuration search;
  3. materialize any proxy winner into a byte-closed archive/runtime packet;
  4. open a fresh lane claim before exact CUDA auth eval;
  5. promote only from archive/runtime custody, never from MPS/Kaggle/macOS
     advisory metrics.
- The A1 score-gradient family is not killed. Reactivation requires
  exact-eval-in-loop or frequent exact anchors, archive-byte growth caps, pose
  drift caps, or packet-compiler integration that proves runtime consumption
  before spend. Re-running the exact regressed configuration is blocked.

## 2026-05-10T09:35Z supersession addendum

- The T1 Modal guard is now terminal, not active. Recover closed the dispatch
  as `failed_t1_modal_recovered_no_score_claim`; current claim summary is
  `active=0`, `stale_nonterminal=0`.
- T1 did not train and produced no archive or score. It failed at Stage 5
  because `/workspace/pact/experiments/results/A1_canonical` was absent inside
  the Modal worker:
  `FrozenA1EncoderError: canonical A1 directory/symlink not found`.
- This is an actuator/export-custody bug, not a T1 model result. Before any
  T1 rerun, the Modal actuator must either fail locally before dispatch when
  the canonical A1 payload cannot be mounted, or explicitly materialize and
  mount the canonical A1 payload and record its SHA custody in metadata.
- Immediate score-lowering queue after this result:
  1. harden raw auth-eval and dispatch-claim overclaim traps from the red-team
     review;
  2. fix T1 canonical-A1 payload mounting/designation before re-claiming T1;
  3. keep Kaggle/Optuna/CMA-ES as proxy-only candidate search;
  4. promote only byte-closed archives through fresh exact CUDA claims.

## 2026-05-10T17:45Z supersession addendum

- Current active remote work is T1 Ballé Phase 1 on Modal, call id
  `fc-01KR955JSYQAVTTYZA48VAV7WJ`, lane `t1_balle_128k_endtoend`; latest poll
  returned `pending`. Do not launch duplicate T1 work while this call is live.
- PR103 arithmetic work produced a better local byte target:
  `global_combo_candidate` is `178211` bytes (`-12B` versus PR103 source,
  `-4B` versus the greedy packet), archive SHA-256
  `578c8f4e86eafc9dc04eefe61cc0e7f3f3f43e134ef4447cf9ef26fd23a23551`, packet
  runtime tree SHA-256
  `8b81480b74919295c37707ac5124934571314f30d3bfe0164cbe7b456e589936`.
- The next PR103 implementation blocker is not another optimizer sweep; it is
  same-runtime source/candidate frame or auth-eval parity. Without that, tiny
  CUDA component deltas can be harness/runtime artifacts.
- Score-lowering priority order now: harvest T1; build PR103 same-runtime
  comparator; only then decide whether the `-12B` global-combo packet deserves
  a fresh claimed exact CUDA run.

## 2026-05-10T09:55Z supersession addendum

- Red-team overclaim traps are now fixed in code. Raw `contest_auth_eval.py`
  CUDA/T4 JSON is exact-eval evidence but no longer claims promotion or
  rank/kill eligibility by itself; `auth_eval_schema` recomputes the contest
  formula; pre-submission compliance consumes schema blockers; stale claims
  require terminal stale closure; T1 recovery respects promotion blockers; and
  unsupported GPU-tier cost estimates fail closed.
- T1 canonical-A1 payload mounting is also fixed in the Modal actuator. Plan
  metadata now records archive/checkpoint/latent/memo SHA custody and refuses a
  missing canonical payload before spend.
- Claim table after the failed T1 guard is clean: `active=0`,
  `stale_nonterminal=0`.
- Next score-lowering action is a fresh, short T1 Modal guard rerun only after
  reviewing this combined hardening patch. It must open a new claim and remains
  `score_claim=false` unless exact CUDA auth-eval schema blockers are zero.

## 2026-05-10T08:58Z Codex active-dispatch addendum

- Commit `740778ba` landed the red-team score-evidence hardening and the T1
  canonical-A1 payload mount fix. Commit `9dd26850` recorded the relaunch
  custody note.
- Current active claim is `t1_balle_128k_endtoend` on Modal:
  `t1_balle_modal_guard_740778ba_20260510T1000Z`, call id
  `fc-01KR8HFZPJRQHWTXAX7TT72D04`, Modal app
  `ap-DFruX1atbAs8UnQYECRrOx`.
- The mounted code snapshot is clean at
  `740778bab317aa93c60fa1208685f1bcbd4383dc`; mounted worktree and index patch
  files are zero bytes. The canonical A1 payload is recorded in the Modal
  metadata with archive/checkpoint/latents/designation-memo SHA-256 custody.
- This run is not a submission or promotion claim. Recovery currently returns
  `NOT READY`, and the active claim must remain open until:
  `.venv/bin/python experiments/modal_t1_balle_endtoend.py recover --label t1_balle_modal_guard_740778ba_20260510T1000Z`
  returns a terminal result and writes a terminal claim row.
- Do not launch another T1 or duplicate exact CUDA run while this active claim
  exists. Parallel work should stay local/proxy: PR95/PR101 packetization,
  Kaggle/Optuna/CMA-ES advisory sweeps that materialize byte-closed candidates,
  xray candidate selection, and preflight/DX hardening.

## 2026-05-10T14:43Z Codex swarm-continuation addendum

- Current active dispatch claim is `t1_balle_128k_endtoend` on Modal:
  `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`, call id
  `fc-01KR955JSYQAVTTYZA48VAV7WJ`. Immediate status poll returned
  `result_state=pending`; do not launch any duplicate T1 or same-lane exact
  CUDA job until recovery writes a terminal claim row.
- Mounted T1 snapshot for the active run is clean at
  `ab2d0f6ec1cf7aed05b8424a0b5f5d79b42698bf` with zero-byte worktree/index
  patches, `epochs=3000`, `batch_size=1`, full target set, and
  `score_claim=false` until recovered auth-eval schema blockers are zero.
- Regenerated the optimizer candidate queue from current constrained-coordinate,
  M5 Max, Optuna/CMA, and codec-op sources:
  `experiments/results/optimizer_candidate_queue_20260510_codex/next_candidate_queue.json`.
  The queue has `n_candidates=124`, `top_k_count=30`, and
  `dispatch_ready_count=0`. This is intentional: every row remains planning-only
  until a separate exact-readiness gate proves archive/runtime custody.
- Top queue rows are still A1/PR101 inflate-time bias coordinate variants with
  predicted GHA CPU rank around `0.19286-0.19370`. These do not beat the A1
  baseline enough to justify blind spend, and they are not CUDA evidence.
- PR101 proxy promotion blocker was re-run with `--allow-blocked`. Current
  verdict remains `BLOCKED_PROXY_ONLY_NOT_PROMOTABLE` for `proxy_cmaes_0037`,
  with blockers `proxy_substrate_not_contest_exact_eval` and
  `no_candidate_contest_cuda_auth_eval`. Local no-scorer execution now proves
  the packet `inflate()` body consumes supported bias params, but that remains
  runtime-consumption evidence only, not scorer-backed exact eval.
- Fresh parallel agents were spawned for:
  1. PR101 proxy runtime-consumption proof hardening without remote jobs;
  2. preflight/DX speed and scan-architecture hardening without weakening checks;
  3. read-only score-lowering roadmap audit under the active T1 lock.
- Immediate local work while T1 runs is therefore:
  harvest T1 when Modal becomes ready; keep PR101/Kaggle proxy work local until
  runtime consumption and exact-readiness are real; continue preflight speed
  improvements only where strict signal is preserved; and keep MPS/macOS/Kaggle
  outputs advisory rather than promotion evidence.
- Refreshed local-only HNeRV frontier scorecard into
  `experiments/results/hnerv_frontier_scorecard_refresh_20260510_codex/`.
  It contains 8 exact CUDA rows; best refreshed row is `PR103-ac-repack` at
  score `0.20898105277982337`, followed by `PR106x-lowlevel` at
  `0.20935073680571203` with `total_byte_delta=-151`.
- `tools/audit_hnerv_frontier_scorecard.py` correctly fails closed on the
  refreshed scorecard: `ready_for_hidden_gem_routing=false` because
  `PR102`, `PR103-ac-repack`, `PR104`, and `PR106x-lowlevel` are missing
  section manifests. This refresh is therefore planning/xray input only, not a
  dispatch-ready hidden-gem queue.
- Preflight/DX hardening in this tranche added a real per-`SourceIndex` build
  lock around `_meta_python_shared_scan`, so parallel dev-preflight callers
  share one fused scan rather than racing duplicate MPS / `eval_roundtrip=False`
  / `--no-eval-roundtrip` passes. The all-lanes surface remains below the 30s
  crash budget (`2.60s` observed wall-clock).
- Kaggle proxy lane `kaggle_pr101_proxy_sweep` is also active as of
  `2026-05-10T09:00:22Z`: kernel `adpena/pr101-proxy-sweep`, version `1`,
  URL `https://www.kaggle.com/code/adpena/pr101-proxy-sweep`, status
  `KernelWorkerStatus.RUNNING`. This is explicitly `score_claim=false`,
  `proxy_only=true`, and cannot affect lane status until a harvested candidate
  is promoted into a byte-closed archive/runtime and evaluated through a
  separate exact-CUDA claim.

## 2026-05-10T09:08Z Codex terminal-state and runtime-architecture addendum

- T1 Modal relaunch `t1_balle_modal_guard_740778ba_20260510T1000Z` is now
  terminal: `failed_t1_modal_recovered_no_score_claim`. It reached Stage 5
  score-domain training but failed importing `segmentation_models_pytorch` from
  `upstream/modules.py`. This is a Modal scorer-runtime dependency closure bug,
  not a T1 model result and not score evidence.
- T1 dependency closure was fixed architecturally, not by adding another
  lane-local package list. Shared Modal contest-CUDA runtime now lives in
  `src/tac/deploy/modal/runtime.py`; T1 uses that helper, mounts
  `tools/tool_bootstrap.py`, and runs a remote scorer import probe before GPU
  training.
- `AGENTS.md` now records Provider Runtime Architecture as a non-negotiable:
  provider/runtime contracts go under `src/tac/deploy/<provider>/`; experiment
  files stay thin lane adapters; deterministic reproducibility and claim/custody
  metadata are mandatory.
- Kaggle proxy sweep `adpena/pr101-proxy-sweep` completed and was closed as
  `completed_proxy_no_score_claim`. Best proxy candidate:
  `experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/pr101_proxy_sweep/best_proxy_candidate.json`.
  Candidate `proxy_cmaes_0037` is proxy-only with objective
  `0.19287550335547282`; it cannot affect lane status until materialized into a
  byte-closed archive/runtime and exact-CUDA evaluated under a fresh claim.
- Current claim summary after recovery/harvest: `active=0`,
  `stale_nonterminal=0`, `terminal_latest=568`.
- Dev preflight remains inside the 30s budget after direct-call timeout
  hardening and Modal runtime refactor:
  `wall_elapsed_s=9.804019`, `serial_elapsed_s=5.035127`,
  `timeout_s=30.0`.
- Next exact score-lowering step is not a duplicate full T1 launch. Relaunch T1
  only as a true bounded guard (`--epochs 50 --batch-size 8
  --max-target-pairs 64 --train-timeout-hours 2`) to validate dependency
  closure, training entry, packet compile, and auth-eval path before any full
  run. In parallel, materialize the Kaggle proxy candidate only as a byte-closed
  candidate artifact with no score claim.

## 2026-05-10T09:30Z Codex guard relaunch and local materialization addendum

- Commit `c4100de4` passed review and was used as the clean mounted code
  snapshot for a fresh bounded T1 Modal guard:
  `t1_balle_modal_guard_c4100de4_20260510T0915Z`, call id
  `fc-01KR8JMK531A9PECP0CV513KQM`, Modal URL
  `https://modal.com/apps/adpena/main/ap-4EJPBdeQqxzaG7rE2KEriI`.
- Current active claim remains exactly one row:
  `lane_id=t1_balle_128k_endtoend`,
  `job=t1_balle_modal_guard_c4100de4_20260510T0915Z`,
  `status=active_dispatching`. Recovery still returns `NOT READY`.
- The run is bounded and non-promotional: `epochs=50`, `batch_size=8`,
  `max_target_pairs=64`, `train_timeout_hours=2`, `score_claim=false`,
  `promotion_eligible=false`.
- The completed Kaggle proxy candidate has been materialized only as local
  handoff JSON, not as an archive/runtime packet. Handoff SHA-256:
  `5e3ee3974ece1011790e3604a402811649865563f10b87e2ae87716c18f39251`.
  Manifest SHA-256:
  `dc709594374c915ffa9c825b33cccff97a9d1e98cc9bc4a991eea2268f71b804`.
  It remains `score_claim=false`, `ready_for_exact_eval_dispatch=false`,
  `archive_zip_emitted=false`, and `inflate_runtime_emitted=false`.
- PR101 A1 Modal score-gradient actuator now uses the shared Modal
  contest-CUDA runtime helper in `src/tac/deploy/modal/runtime.py` and runs a
  fail-closed scorer import probe before training, preserving the provider
  runtime architecture split.
- Parallel audit verdict: do not jump to full T1 until the bounded guard
  packet path clears. If it clears, the next PR101-family packet-producing
  action is certified A2 sensitivity-weighted PR101 packet work; A2 byte
  savings remain blocked from exact score dispatch until stub/proxy sensitivity
  is replaced with certified sensitivity.

## 2026-05-10T09:40Z Codex T1 guard OOM classification addendum

- T1 guard `t1_balle_modal_guard_c4100de4_20260510T0915Z` is now terminal:
  `failed_t1_modal_recovered_no_score_claim`, claim summary `active=0`.
- The Modal scorer runtime dependency closure is now proven for this guard:
  the remote import probe passed, and NVDEC was exposed on Tesla T4.
- The guard failed in Stage 5 score-domain training with CUDA OOM inside
  `src/tac/losses.py::sinkhorn_w2_mask_distortion_per_pixel`.
  Classification: T8 Sinkhorn surrogate memory implementation / guard sizing
  bug, not a T1 model negative and not score evidence.
- Fix applied: Sinkhorn-W2 now chunks flattened spatial rows by default instead
  of allocating one full `(N, C, C)` tensor for all SegNet pixels, preserving
  unchunked numeric output and gradients in tests. Shared Modal runtimes also
  set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Next T1 action after this commit is another bounded guard rerun from clean
  code. Do not escalate to full T1 until the guard proves training entry,
  packet compile, inflate closure, and exact-CUDA auth-eval schema closure.
- That rerun is now active from clean commit `c0ea27df`:
  `t1_balle_modal_guard_c0ea27df_20260510T0927Z`, call id
  `fc-01KR8KCXEGTVFGSZ75HAK9S2QX`, Modal URL
  `https://modal.com/apps/adpena/main/ap-g6JJhRr82ENgaEaOcfduIu`.
  Immediate recovery returned `NOT READY`; no score claim exists.

## 2026-05-10T10:20Z Codex score-lowering queue after c0 guard harvest

Current custody state:

- T1 guard `t1_balle_modal_guard_c0ea27df_20260510T0927Z` harvested terminal:
  `failed_t1_modal_recovered_no_score_claim`.
- Claim summary after harvest is `active=0`; no duplicate T1 claim remains.
- Classification is runtime/trainer guard OOM, not score evidence and not a
  T1 model-family negative.

Near-term score-lowering order:

1. Relaunch T1 only as a strict bounded guard from the next clean commit:
   `epochs=50`, `batch_size=1`, `max_target_pairs=8`,
   `sinkhorn_max_positions_per_chunk=2048`, `train_timeout_hours=2`. The goal
   is training entry + packet compile + exact-CUDA auth-eval schema closure,
   not a score claim unless the full archive/runtime path actually passes.
2. Materialize the PR101 Kaggle proxy candidate with
   `tools/build_pr101_kaggle_proxy_runtime_packet.py`. This is still
   fail-closed and `score_claim=false`: it copies PR101 runtime + archive,
   patches only the proven per-channel bias lines, records unsupported proxy
   params as blockers, and emits runtime custody. It is a bridge to a future
   exact-eval packet, not evidence by itself.
3. Promote certified A2 sensitivity only after the now-fixed manifest builder
   is fed a real certified sensitivity map. A2 remains a stacking component,
   not a standalone sub-0.17 path.
4. Start the preflight wall-clock tranche: migrate the remaining slow checks to
   the existing source-index snapshot and keep all strict checks under the
   operator's 30s crash threshold, ideally warm full preflight under 5s.

Provider/runtime architecture remains separated: Modal logic lives under
`src/tac/deploy/modal/` plus thin experiment adapters, Vast/AWS/Azure/GCP/Kaggle
work should follow the same provider-neutral runtime contract, and no
experiment script should become the provider abstraction.

## 2026-05-10T12:40Z strict T1 guard active

Commit `80b139c9` launched the strict bounded T1 Modal guard:

- lane id: `t1_balle_128k_endtoend`
- instance job id: `t1_balle_modal_guard_80b139c9_20260510T1240Z`
- Modal call id: `fc-01KR8YCW8F12KACG5TKWSNZ7A7`
- Modal URL:
  `https://modal.com/apps/adpena/main/ap-QR661mpdFN68qaOLXXv2JC`
- exact limits: `epochs=50`, `batch_size=1`, `max_target_pairs=8`,
  `sinkhorn_max_positions_per_chunk=2048`, `train_timeout_hours=2`
- immediate recover result: queued/running, not ready

Current score-lowering queue while this is active:

1. Harvest this T1 guard. If it clears training entry + packet compile +
   exact-CUDA auth-eval schema, promote to a full T1 dispatch decision; if it
   fails, classify the exact stage and add the next fail-closed guard.
2. In parallel, keep PR101 proxy packet work local-only: build packet manifests
   and runtime-consumption proofs, but do not dispatch until unsupported proxy
   params are either compiled into runtime-consumed bytes or explicitly
   removed from the candidate.
3. Begin preflight source-index migration using the timing profile from
   `experiments/results/preflight_timings_20260510T1237_poststage_codex.json`:
   top slow steps are Modal image build order, A2 packet ladder closure,
   untracked source inventory, PR91/HPM1, and tooling consolidation.

## 2026-05-10T12:45Z PR101 proxy runtime packet materialized locally

Local-only PR101 Kaggle proxy runtime packet artifact has been built:

- packet dir:
  `experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/pr101_proxy_sweep/proxy_runtime_packet`
- manifest:
  `experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/pr101_proxy_sweep/proxy_runtime_packet/runtime_packet_manifest.json`
- manifest SHA-256:
  `3b4993724e91ba112be6e02184d020b3485e5458eb759208be4bf78bd2395251`
- runtime tree SHA-256:
  `0d24791f96d614cdb2eb36baafb9412d3d8bc9b6bffd9acec1ad75e6f8cbe628`
- unchanged archive SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`

Authority boundary remains fail-closed:

- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- blockers include unsupported proxy params not runtime-consumed:
  `delta_scale`, `latent_delta_scale`, `smooth_weight`
- no Level 2 dispatch claim was opened for this local materialization

Next bridge work is a runtime-consumption proof and either compiling the
unsupported params into real runtime-consumed bytes or shrinking the candidate
contract to only the three bias params before any exact eval dispatch.

## 2026-05-10T12:55Z PR101 proxy runtime-consumption proof

Static local runtime-consumption proof was emitted for the PR101 proxy packet:

- proof:
  `experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/pr101_proxy_sweep/proxy_runtime_packet/runtime_consumption_proof.json`
- proof SHA-256 excluding self:
  `29dec2c1db61bb8bf93cc0229af8f4e8801d8fbe6e8c0dc245ce7428db321cfa`
- supported runtime-consumed params:
  `bias_r -> up[:, 0, 0]`, `bias_b -> up[:, 0, 2]`,
  `bias_g -> up[:, 1, 1]`
- old PR101 `sub_(1.0)` bias lines absent from `inflate.py`
- archive unchanged SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`

Authority boundary remains unchanged:

- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- no scorer/inflate/eval was run
- unsupported `delta_scale`, `latent_delta_scale`, and `smooth_weight` remain
  blockers and are explicitly not runtime-consumed

Verification:

```bash
.venv/bin/python -m pytest \
  tests/test_prove_pr101_kaggle_proxy_runtime_consumption.py \
  tests/test_build_pr101_kaggle_proxy_runtime_packet.py -q
# 10 passed
.venv/bin/python -m py_compile \
  tools/prove_pr101_kaggle_proxy_runtime_consumption.py \
  tests/test_prove_pr101_kaggle_proxy_runtime_consumption.py
```

Next PR101 proxy step: either compile the three unsupported params into
runtime-consumed bytes with tests, or shrink the candidate schema to the
three proven bias params before any Level-2 exact-eval claim can be opened.

## 2026-05-10T13:15Z adversarial review fixes + T1 guard harvest

Red-team review found two HIGH issues and one MEDIUM issue; fixes landed in
the current tranche:

- Modal `.spawn()` ambiguity no longer terminal-closes the lane claim when the
  SDK raises after the submission boundary. The claim is left nonterminal as
  `ambiguous_modal_spawn_submission_recovery_required` with a local recovery
  record for dashboard/API reconciliation.
- Remote T1 mounted-code git probes normalize no-git / detached / multiline
  `HEAD\nunknown` output to `unknown` while preserving fail-closed behavior for
  real mismatching SHAs.
- The PR101 proxy proof no longer claims full runtime consumption. It now
  proves static patched bias lines plus a no-scorer `inflate.sh -> inflate.py`
  wrapper route; `runtime_consumption_proven_for_supported_bias_params=false`
  until the real inflate body and scorer-backed output path are exercised.

Updated PR101 proxy proof:

- proof SHA-256 excluding self:
  `27f3239abc2aad6e791418f7944ae931d648951c9949c4b659a2b000e87f591e`
- `inflate_sh_routes_to_packet_inflate_py=true`
- `supported_bias_params_static_patch_proven=true`
- `runtime_consumption_proven_for_supported_bias_params=false`
- `score_claim=false`, `ready_for_exact_eval_dispatch=false`,
  `dispatch_attempted=false`

Strict T1 Modal guard `t1_balle_modal_guard_80b139c9_20260510T1240Z`
completed as a measured infrastructure/custody negative, not a model result:

- score-domain training entered and completed 50 epochs on T4.
- packet compiler emitted archive SHA-256
  `4b8073665aec2193a6f86407663da13f9f83540d5fd770aa65b8f91875d44d53`
  at 481,704 bytes.
- exact CUDA auth eval failed before scoring because the remote script passed
  the packet compiler's runtime-tree hash
  `133d32db974234bb499182772b5443a353aad61d3e3ce762f4eb44fdcd427a82`
  as `--expected-runtime-tree-sha256`, while `contest_auth_eval` correctly
  hashed the final runtime tree as
  `921a38ada4d82f5263207c230ce239eb6a275485343a62c6c76ed1551d6b9930`.
- dispatch claim was terminally closed as
  `failed_t1_modal_recovered_no_score_claim`.

Fix landed: T1 remote dispatch now computes the expected runtime-tree hash
using `experiments.contest_auth_eval._runtime_dependency_manifest` on the final
packet runtime immediately before auth eval, not the packet compiler's
different byte-closure hash. The packet compiler now withholds the
self-referential final `build_manifest.runtime_tree_sha256` and records its
pre-manifest byte-closure hash as `pre_manifest_runtime_tree_sha256`.

Preflight/DX status:

- source-index equivalence tests pass for comment-only contracts,
  bare-round eval-roundtrip, and profile-key resolver scans.
- full all-lanes preflight remains under the 30s DX budget: 5.07s wall,
  16.66s serial sum, 3.29x estimated speedup, all 29 checks passed after
  refreshing the raw `experiments/results/` runtime-source baseline for the
  newly harvested T1/proxy artifacts.

## 2026-05-10T13:01Z strict T1 guard relaunched from fixed commit

Commit `1aac11aa` relaunched the bounded T1 Modal guard after the runtime-tree
custody fix:

- lane id: `t1_balle_128k_endtoend`
- instance job id: `t1_balle_modal_guard_1aac11aa_20260510T1301Z`
- Modal call id: `fc-01KR8ZNESYP42EP7928ZK94ZQB`
- Modal URL:
  `https://modal.com/apps/adpena/main/ap-RNwmDecklm1TEfPvNrgYFo`
- exact limits: `epochs=50`, `batch_size=1`, `max_target_pairs=8`,
  `sinkhorn_max_positions_per_chunk=2048`, `train_timeout_hours=2`
- estimated cost in claim: `$14.16`
- immediate recover result: queued/running, not ready

Next harvest command:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_guard_1aac11aa_20260510T1301Z
```

Authority boundary: this is still a guard dispatch. It may become useful
contest-CUDA evidence only if recover verifies exact archive/runtime custody,
600 samples, CUDA/T4 hardware, component recomputation, zero auth-eval
blockers, and `score_claim=true` in the adjudication artifact.

## 2026-05-10T14:33Z T1 e7845 harvest + PR101 proxy packet refresh

T1 Modal full-path smoke
`t1_balle_modal_fullpath_smoke_e7845e4c_20260510T1410Z` is terminal. Claim
summary after recovery is `active=0`, `stale_nonterminal=0`.

Result classification:

- exact CUDA auth eval executed on `Tesla T4` with `n_samples=600`;
- measured diagnostic score was `56.06364706567909`, with `seg_avg=0.50482631`,
  `pose_avg=2.75759292`, and `archive_size_bytes=495206`;
- `score_claim=false`;
- hard blocker:
  `t1_mounted_code_missing_extracted_archive_runtime_hardening`;
- this is a full-path diagnostic negative from a pre-hardening code snapshot,
  not a promotion result and not a T1/Ballé model-family kill.

The positive signal is path coverage: the post-EMA-fix remote path reached
archive export, packet compile, no-op proof, and contest-CUDA auth eval. The
negative signal is that one epoch of the current T1 setup is catastrophically
untrained and that pre-hardening runtime custody is not acceptable evidence.

PR101 Kaggle proxy packet was regenerated under the current bias-only contract:

- runtime packet manifest SHA-256:
  `c3b20ed70442b0b5128692d02fa7d097edc0807495b495a85cd9c565ed2ce48b`
- runtime-consumption proof SHA-256:
  `108b7ab532a8e4ad17d511e17635fdbf4e4584cfba5dc8b1e01d454304b09ecf`
- routed candidate params: `bias_b`, `bias_g`, `bias_r`
- legacy proxy-search params `delta_scale`, `latent_delta_scale`, and
  `smooth_weight` are explicitly ignored rather than treated as candidate
  bytes.

Promotion blocker verdict remains correct:

```text
BLOCKED_PROXY_ONLY_NOT_PROMOTABLE
blockers = [
  "full_runtime_consumption_not_proven",
  "no_candidate_contest_cuda_auth_eval",
]
```

Next score-lowering order:

1. If relaunching T1, do it only from the hardened head with extracted-archive
   runtime custody and a bounded guard/full-run choice recorded in the claim.
2. For PR101 proxy, either run a no-scorer full inflate-body smoke that proves
   supported bias params execute through the real runtime, or explicitly keep
   the packet as local static-patch evidence until a fresh exact-CUDA claim is
   opened.
3. Keep Kaggle/CMA-ES/Optuna as candidate generators only; exact movement still
   requires byte-closed archive/runtime custody and contest-CUDA auth eval.

## 2026-05-10T14:38Z full T1 Phase 1 Modal dispatch active

After the pre-hardening e7845 smoke was harvested and the active claim closed,
commit `ab2d0f6e` launched a real full-path T1 Phase 1 Modal T4 run:

- lane id: `t1_balle_128k_endtoend`
- instance job id: `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- Modal call id: `fc-01KR955JSYQAVTTYZA48VAV7WJ`
- Modal run URL:
  `https://modal.com/apps/adpena/main/ap-1fCuVHqShCT1puDuPs7SHY`
- label commit: `ab2d0f6ec1cf7aed05b8424a0b5f5d79b42698bf`
- mounted code status: clean, worktree patch bytes `0`, index patch bytes `0`
- plan artifact:
  `experiments/results/t1_balle_modal_phase1_ab2d0f6_20260510T1437Z_plan.json`
- plan SHA-256:
  `7003baabb61c0545ff9177a5c3759d050f5a9e5d502004d4994b5d3eafac1d35`
- metadata artifact:
  `experiments/results/t1_balle_modal_phase1_ab2d0f6_20260510T1437Z/modal_metadata.json`
- metadata SHA-256:
  `e3cfc8dc088c42822edb3cf1b035612057b4d16da17ffbc6b4e1bc28104cce09`

Training/eval parameters:

- `epochs=3000`
- `batch_size=1`
- full target set (`max_target_pairs=null`)
- `sinkhorn_max_positions_per_chunk=2048`
- `train_timeout_hours=22.5`
- `timeout_hours=24`
- `cost_cap_usd=80`
- estimated cost in plan: `$14.16`
- contest-CUDA auth eval requested for the exported packet.

Immediate status poll returned `pending`; active-claim summary is now one row:

```text
ACTIVE lane_id=t1_balle_128k_endtoend
job=t1_balle_modal_phase1_ab2d0f6_20260510T1437Z
platform=modal
status=active_dispatching
```

Recover with:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_phase1_ab2d0f6_20260510T1437Z
```

Do not launch another T1 job while this claim is active. This run is the first
current-head T1 attempt aligned with the operator's actual score-lowering
goal: full training plus packet compile plus exact contest-CUDA auth eval.
Promotion still requires recovery to verify archive/runtime custody, 600
samples, CUDA/T4 hardware, component recomputation, zero auth-eval blockers,
and `score_claim=true` in adjudication.

## 2026-05-10T15:08Z swarm harvest addendum

Current active remote state is unchanged: `t1_balle_128k_endtoend` remains the
only active claim, job `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`, call id
`fc-01KR955JSYQAVTTYZA48VAV7WJ`. Recovery still returns `NOT READY`, so no
duplicate T1 or same-lane exact-CUDA dispatch is allowed.

Three local-only swarm results advanced the queue without score overclaim:

1. HNeRV frontier hidden-gem routing is unblocked. The refreshed local
   scorecard at
   `experiments/results/hnerv_frontier_scorecard_refresh_20260510_codex/`
   now has section profiles for `PR102`, `PR103-ac-repack`, `PR104`, and
   `PR106x-lowlevel`. `tools/audit_hnerv_frontier_scorecard.py --format json`
   reports `ready_for_hidden_gem_routing=true`, zero blockers, 8 canonical
   labels, 24 follow-up targets, and 8 payload section manifests. The next
   routing target is still not a score claim: `PR103-ac-repack` /
   `merged_range_coded_weights_and_hi_latents` at `153856` bytes. Required
   next gate is a byte-different candidate with old/new section SHA-256,
   charged-byte proof, runtime-consumption proof, lane claim, then exact CUDA
   auth eval.
2. PR106 sidechannel triage found the shortest static-packet path is not the
   three-sister scaffold. It is the already byte-closed WR01 PR106x packet:
   `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/archive.zip`
   at `186222` bytes, archive SHA-256
   `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628`.
   It remains `score_claim=false` and dispatch-blocked until exact CUDA auth
   eval, adjudication, terminal dispatch claim, and operator score review.
   The three PR106 sidechannel smoke manifests now record local archive paths,
   archive SHA-256s, `dispatch_attempted=false`, and exact dispatch blockers.
3. A2 sensitivity-weighted packetization now fails closed with structured
   blocker evidence instead of a generic certification error. Current blocker
   artifact:
   `reports/a2_certified_sensitivity_blocker_20260510_codex.json`. Stable
   blockers are `a2_certified_sensitivity_binding_invalid`,
   `a2_component_sensitivity_manifest_reference_missing`,
   `a2_sensitivity_artifact_diagnostic_allowed`, and
   `a2_sensitivity_artifact_metadata_blockers_present`. The raw combined map
   found locally is not enough: promotion still requires a promotion-grade
   `component_sensitivity_v1` manifest binding the map, not a synthesized or
   diagnostic stub.

Immediate score-lowering order after this harvest:

1. Continue polling and recover the active T1 Modal run; close the claim
   terminally before any same-lane rerun or score promotion.
2. Use the unblocked HNeRV scorecard to build the next byte-different hidden
   gem candidate locally, starting with `PR103-ac-repack` section-level
   transforms; do not dispatch until old/new section hashes and runtime closure
   are recorded.
3. Keep WR01 PR106x as a static exact-eval candidate for the next available
   non-conflicting CUDA slot, but do not call it a result before auth eval.
4. Build a real promotion-grade A2 component-sensitivity manifest or keep A2
   blocked. Do not downgrade the certification gate.
5. Keep preflight under the 30s crash budget and continue only speedups that
   preserve the strict guard surface.
