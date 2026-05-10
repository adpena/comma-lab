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
