# Shannon-Floor Next-Wave Telemetry And Research Ledger - 2026-04-30

Adjacent source documents:

- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- `.omx/research/council_paradigm_shift_round1_20260430.md`
- `.omx/research/council_paradigm_shift_round2_20260430.md`
- `.omx/research/council_paradigm_shift_round3_20260430.md`
- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`

This ledger records the next wave after the OWV3 R6 exact CUDA/T4 negative.
It is not a score ledger and does not create promotion evidence.

## Provider Telemetry

Live checks from local credentials:

- MCP helpers: killed again; no `chrome-devtools-mcp` or `rbx-studio-mcp`
  processes remained after the kill pass.
- Vast: `.venv/bin/vastai show instances --raw` returned `[]`.
- Modal: `.venv/bin/modal app list` showed zero tasks across listed apps.
- Lightning: bulk SDK status refresh covered all non-dry-run local records and
  skipped dry-runs by default. Output saved to
  `.omx/state/lightning_batch_jobs_refresh_20260430_codex_nextwave.json`.

Lightning refresh summary:

```text
refreshed_count = 9
skipped_count = 13
failure_count = 0
latest completed = owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1
```

No new harvestable provider result surfaced beyond the already harvested R6
exact CUDA/T4 result.

Worker C independently recorded the same provider posture in
`.omx/research/provider_telemetry_canonical_harvest_audit_20260430_worker_c.md`:

- Vast live instances: none.
- Modal live tasks: none.
- Lightning live running records: none in local state.
- Lightning recorded total across non-dry local records: `$1.307211119`.
- Historic Vast tracker rows are stale and not live spend truth.
- No provider kill action is recommended; keep Lane 19 and Lane 20 holds in
  place.

## DX Hardening Landed

`scripts/launch_lightning_batch_job.py refresh-status` now supports:

- `--all`: refresh every latest non-dry-run local Lightning record using
  `lightning_sdk.Job` attributes only.
- `--include-dry-runs`: opt-in for dry-run records.
- `--fail-on-error`: fail the bulk command if any individual refresh fails.
- Per-job `--job-name` behavior remains supported.

This closes the operator footgun where a stale or unsupported `--all` command
could leave telemetry unknown. It also preserves the security rule: never call
the compromised-risk `lightning` console script for status.

Verification:

```bash
.venv/bin/python -m py_compile \
  scripts/launch_lightning_batch_job.py \
  src/tac/tests/test_lightning_batch_jobs.py

.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q
# 33 passed

.venv/bin/python scripts/scan_lightning_supply_chain.py \
  --json-out .omx/state/lightning_supply_chain_scan_20260430_codex_nextwave.json \
  --strict
# status OK, violation_count 0
```

## Research-Agent Consensus

### arXiv:2604.26919v1

`Causal Learning with Neural Assemblies` is not a compression, entropy-model,
video, neural-codec, or rate-distortion paper. Direct contest score impact is
`0.000`.

Useful transfer:

- dual readout for sensitivity claims: structural/Fisher channel rankings must
  agree with held-out finite-difference component movement;
- warm ramps for scorer-sensitive loss/quantization pressure;
- sparse Top-K protection as a byte-allocation heuristic;
- explicit causal audit graph from mask payload through scorer components.

Non-use:

- no runtime/archive lane;
- no new dependency;
- no promotion evidence without exact CUDA archive evaluation.

Primary source: https://arxiv.org/abs/2604.26919

### PufferLib / RL / LM Studio / Visual Primitives

PufferLib is useful only after a cheap, validated surrogate environment exists.
Exact archive eval is too expensive and sparse for direct PPO-style rollout
search.

Admissible near-term order:

1. bandit / BO / CMA-ES over bounded codec knobs;
2. surrogate correlation gate with exact CUDA anchors;
3. PufferLib or Protein-style search only if the surrogate passes correlation;
4. LM Studio only for read-only custody triage with strict JSON schemas;
5. DeepSeek visual primitives only as a methodology for Alpha geometry
   diagnostics, not as a runtime dependency.

Hard rule: none of these methods can rank, promote, kill, dispatch, or parse
scores from logs.

### Training-Free GRPO / GEPA / BOHB

Training-Free GRPO is useful as memory-backed agentic lane search, not direct
codec optimization. It can generate grouped proposals, score them with
diagnostics, extract lessons, and update an experience library.

More directly applicable:

- Hyperband / BOHB for budgeted training and candidate pruning;
- GEPA/TextGrad-style prompt refinement for operator prompts and runbooks;
- AlphaEvolve-lite only inside tiny, test-fenced allocator code such as OWV3
  byte-plan selection.

Guardrail: semantic memory can propose configs, but promotion truth remains
`archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda`, archive SHA,
bytes, component recomputation, adjudication JSON, and adversarial review.

Primary source: https://arxiv.org/abs/2510.08191

## Current Scientific State

- PFP16 final-deploy A++ remains the controlling promotion-grade anchor. The
  clean paired PFP16 r3 run is same-run forensic calibration for OWV deltas
  only; it fired the final-deploy SegNet component gate and must not be cited
  as a new promotion packet.
- OWV3 R6 is exact CUDA/T4 forensic negative for its measured config:
  byte savings of `104` bytes were overwhelmed by PoseNet drift.
- Vast and Modal have no visible live jobs from local credentials.
- Lightning has no newly surfaced harvestable result after the bulk refresh.
- Component-sensitivity and OWV3 next work should use dual readout:
  Fisher/channel structure plus held-out finite-difference component response.
- OWV3 scalar-threshold R7 is blocked on the current byte-plan grid. Worker B
  landed `r7-pose-balanced` selection in
  `experiments/sweep_owv3_byte_plan.py`; the existing R5 sweep candidates
  produce `candidate_count=0` when using R6
  `owv3_0076_bbr0p65_protect0p0013_aggr1em05` as reference.
- Canonical official component-response producer is now present:
  `experiments/profile_component_sensitivity_official.py`. It consumes
  baseline and perturbation archives, evaluates them through
  `experiments/contest_auth_eval.py` or validates existing exact
  `contest_auth_eval.json` custody, and emits PoseNet/SegNet/combined official
  response curves for `build_component_sensitivity_manifest.py`. It does not
  generate perturbation archives, component maps, or stability JSON.

## Next Admissible Actions

1. Generate perturbation archives for the official response producer, then run
   `profile_component_sensitivity_official.py --device cuda --require-passed`
   on Lightning or another CUDA provider.
2. Assemble `component_sensitivity_v1` only when official response curves,
   component maps, stability JSON, sample plan, archive SHA/bytes, and exact
   `contest_auth_eval.json` custody all pass.
3. Do not exact-eval another OWV3 scalar-threshold candidate until the
   PoseNet-drift failure mode
   is addressed by component sensitivity or a materially new action rule.
4. Build a `causal_audit_v1` / experience-memory schema only as non-promotable
   proposal hygiene; it must carry evidence grade, source paths, archive SHA
   when applicable, failure mode, and forbidden-generalization notes.
5. Use BOHB/bandit search before PufferLib; require exact-CUDA correlation
   anchors before any RL environment is trusted.
6. Keep provider telemetry checks on pinned tools:
   `.venv/bin/vastai`, `.venv/bin/modal`, and
   `scripts/launch_lightning_batch_job.py refresh-status --all`.

Verification for the official-response producer:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_profile_component_sensitivity_official.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  -q
# 48 passed
```

## 2026-05-01T00:07Z Next Wave 2 Kickoff

Swarm roles:

- Perturbation archive / official response-plan producer.
- Lightning official-response queue readiness.
- Alpha visual-primitive geometry diagnostics.
- NWCS corpus/sensitivity fail-closed readiness.
- Claim ledger adversarial audit.
- DX/preflight self-protection.

Control-plane telemetry before implementation:

- MCP helpers killed and verified absent.
- Vast: `.venv/bin/vastai show instances --raw` returned `[]`.
- Modal: `.venv/bin/modal app list` showed zero tasks.
- Lightning SDK bulk refresh saved to
  `.omx/state/lightning_batch_jobs_refresh_20260501_codex_nextwave2.json`:
  `refreshed_count=9`, `skipped_count=13`, `failure_count=0`.
- Lightning supply-chain scan saved to
  `.omx/state/lightning_supply_chain_scan_20260501_codex_nextwave2.json`:
  `status=OK`, `violation_count=0`, no PyPI `lightning` or
  `pytorch-lightning`, `lightning-sdk==2026.4.10`.

Interpretation:

- No provider kill or harvest action is available from current local
  credentials.
- Next deployable work remains blocked on official perturbation archives and a
  passing `component_sensitivity_v1` packet.

## 2026-05-01T00:22Z Next Wave 2 Integration

Landed implementation and hardening:

- Deterministic perturbation archive/plan producer:
  `experiments/build_component_response_perturbation_plan.py` and
  `src/tac/tests/test_build_component_response_perturbation_plan.py`.
  It emits `official_component_response_plan_v1`, writes
  `perturbation_basis_v1.json`, builds bounded ZIP archive variants, rejects
  duplicate/hidden/unsafe members, blocks renderer-magic mutation by default,
  and records `auth_eval_required=cuda`. It makes no score claim.
- Lightning official component-response queue readiness:
  `src/tac/deploy/lightning/batch_jobs.py`,
  `scripts/launch_lightning_batch_job.py`,
  `src/tac/tests/test_lightning_batch_jobs.py`, and
  `docs/runbooks/lightning_official_component_response.md`. The new
  `component-response` job role is CUDA-only, supply-chain scanned, DALI
  hash-pinned, staged-manifest gated for non-dry-run submit, and harvests a
  compact official-response evidence set.
- Official-response harness regression coverage now verifies the
  `contest_auth_eval.py` subprocess command contains one `--inflate-timeout`
  and one `--evaluate-timeout` flag, preventing argparse breakage before exact
  eval.
- NWCS fail-closed greenup tightened fake/proxy/uniform/diagnostic sensitivity
  rejection in builders, manifests, and promotable remote scripts. Promotion
  remains blocked until a real `component_sensitivity_v1` exists.
- Alpha-Geo-0 now emits a CPU-only residual-region ranking artifact for Lane
  12 `jsonfix40` versus Lane G v3/base masks:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_residual_regions_20260430.json`.
  Artifact SHA: `99a5747ff2ceca745845a1316df8acf906969a7109d0f1a4400e29bc653f8a9c`.
  This is `promotion_eligible=false` and `score_claim_eligible=false`.
- Claim hygiene backfilled failed-gate OWV3/PFP16 calibration provenance with
  `promotion_eligible=false`, `paper_claim_grade="A-negative scoped forensic"`,
  and `allowed_use=["forensic","no_rank_frontier","no_promotion"]` where
  appropriate. `evidence_grade` is now treated as hardware/custody grade, not
  paper-promotion status.
- Modal CPU auth-eval wording and recovery paths now label non-CUDA scores as
  advisory/non-promotable and prefer structured
  `score_recomputed_from_components`.

Telemetry and security:

- Final provider refresh saved to
  `.omx/state/lightning_batch_jobs_refresh_20260501_codex_nextwave2_final.json`:
  `refreshed_count=9`, `skipped_count=13`, `failure_count=0`; status counts
  are `Completed=1`, `Failed=7`, `Stopped=1`, with no running jobs. The only
  completed item is already-harvested R6.
- Vast live instances remain `[]`.
- Modal app list shows zero tasks across all visible apps.
- MCP process probes returned no persistent MCP helper after cleanup; later
  single-PID hits were transient search processes.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_component_response_perturbation_plan.py \
  src/tac/tests/test_profile_component_sensitivity_official.py \
  src/tac/tests/test_lightning_batch_jobs.py \
  -q
# 52 passed
```

Additional focused checks in this wave:

- Alpha diagnostics: `11 passed`.
- Remote auth/Modal CPU advisory guard: `33 passed`.
- NWCS/remote auth combined slice: `87 passed`.
- Python compile and shell syntax checks passed for touched Python/shell files.

Ongoing research swarm spawned at xhigh after closing the first six workers:

- arXiv:2604.26919v1 and related Shannon-floor methods: completed.
- PufferLib/RL, local LM Studio tooling, and DeepSeek visual primitives:
  completed.
- Tencent training-free GRPO and related GRPO/RLVR/black-box optimization:
  completed.
- KL-distill hardening architecture and all KL variant code paths: ongoing.

Next canonical deployment path:

1. Choose a bounded perturbation basis against the PFP16 final-deploy A++
   archive and generate the official response plan/archives.
2. Stage baseline archive, baseline `contest_auth_eval.json`, plan, and every
   plan-point archive through `scripts/lightning_repro_workspace.py`.
3. Submit `scripts/launch_lightning_batch_job.py component-response` without
   `--dry-run`, with `--source-manifest`, `--local-perturbation-plan`, and
   `--require-passed`.
4. Harvest with `harvest-component-response-ssh`, then assemble
   `component_sensitivity_v1` only if response curves, maps, stability, sample
   plan, archive custody, and exact CUDA JSON all pass.
5. Resume OWV3/NWCS/Alpha promotion decisions only from that canonical
   sensitivity packet or from a materially new pre-reviewed action rule.

## 2026-05-01T00:34Z Research Swarm Intake / MCP Hardening

External-method intake:

- arXiv:2604.26919v1, `Causal Learning with Neural Assemblies`, is not a
  compression mechanism and must not spawn a codec lane. Its useful transfer is
  methodological: sparse top-k selection, warm-ramped directed updates, and
  dual readout. Apply this as a requirement that Fisher/channel structure must
  be paired with held-out functional component response before sensitivity or
  byte-allocation claims.
- Related paper scan priorities:
  - CI-ICM/channel-importance compression is directly relevant to
    PoseNet/SegNet-separated Beta/OWV3 protection.
  - S2-CoT is a warning that feature/codec adaptation and entropy modeling
    must be co-tuned; NWCS/hyperprior lanes must measure coded bytes after
    side-info accounting.
  - TinyNeRV supports Alpha redesign, but only against decoded baseline masks
    and exact CUDA eval.
  - Feedback-driven rate control is useful as a Gamma controller pattern, not
    a runtime dependency.
- DeepSeek visual-primitives methodology is admissible as Alpha diagnostic
  design input over decoded baseline masks: connected components, boundary
  polylines, lane/road/vehicle primitives, temporal tracks, and worst
  primitive failures. It is CPU/diagnostic only until tied to exact archive
  custody and CUDA eval.
- PufferLib/RL is deferred until a cheap surrogate reward is rank-correlated
  against exact CUDA anchors. Near-term search should use deterministic
  bandit/BO/CMA-ES/Optuna/Nevergrad/BoTorch-style loops over bounded knobs.
- LM Studio/local models may be used only for read-only structured triage and
  lane-card generation behind JSON schemas and deterministic validators.
- Tencent Training-Free GRPO is useful as a hashed read-only experience
  library for grouped agent/lane proposals. It is not compression evidence and
  cannot dispatch, promote, kill, or change scorer/archive code.

MCP hardening:

- Killed live `rbx-studio-mcp`, `chrome-devtools-mcp`, and child helper
  processes spawned by the outer Codex supervisor.
- Disabled future-session Codex curated plugins in
  `/Users/adpena/.codex/config.toml`:
  `game-studio@openai-curated` and `cloudflare@openai-curated`.
- Removed transient MCP/runtime artifacts:
  `.playwright-mcp`,
  `/Users/adpena/.codex/.tmp/plugins/plugins/cloudflare/.mcp.json`, and
  `/Users/adpena/.codex/.tmp/plugins/plugins/build-ios-apps/.mcp.json`.
- Verified Claude MCP config remains empty:
  `/Users/adpena/.claude/mcp.json` has `mcpServers={}` and Claude
  `chrome-devtools-mcp@claude-plugins-official` is disabled.
- Final process probe with bracketed patterns returned no MCP helpers.

No-claim constraints:

- No external paper, RL, GRPO, LM Studio, visual-primitive, Fisher, CPU/MPS,
  proxy, or byte-only output can promote, rank, kill, or anchor paper claims.
- These methods can prioritize exact-eval spend only after deterministic
  artifacts and exact CUDA calibration show predictive value.

## 2026-05-01T00:41Z KL Audit Intake / Micro-Hardening

KL Grand Council audit outcome:

- Primary scorer KL remains mostly fenced: `primary_scorer` requires explicit
  waiver and non-promotable status; legacy `segnet_kl` is forced into
  forensic/non-promotable use.
- Remaining blocker is architectural, not a single flag: KL/JBL/distillation
  needs one typed policy surface and one provenance serializer used by
  `TrainConfig`, SegMap, `train_renderer`, `optimize_poses`, remote scripts,
  and adjudication.
- Required policy fields: family, scope, weight, temperature, class weights,
  teacher/student roundtrip contract, promotion eligibility, forensic reason,
  banned-primary opt-in, and optional controller/SNR telemetry.
- Promotion of any KL/JBL/distill-active archive must require exact CUDA
  archive eval, archive SHA/bytes, full sample count, component gates,
  canonical eval path, and non-collapse evidence.

Landed low-risk hardening:

- `src/tac/losses.py`: `kl_distill_scorer_loss` and
  `kl_distill_segnet_only` now validate finite positive temperatures before
  dividing logits.
- `src/tac/losses_jbl.py`: removed unsafe wording that JBL “cannot induce”
  PoseNet collapse; JBL is now documented as distillation-family and gated by
  exact CUDA component evidence before promotion.
- Tests:
  - `src/tac/tests/test_losses.py` covers invalid auxiliary KL temperatures.
  - `src/tac/tests/test_training.py` covers invalid full KL-scorer
    temperatures.

Verification:

```bash
.venv/bin/python -m py_compile \
  src/tac/losses.py src/tac/losses_jbl.py \
  src/tac/tests/test_losses.py src/tac/tests/test_training.py

.venv/bin/pytest \
  src/tac/tests/test_losses.py \
  src/tac/tests/test_training.py \
  src/tac/tests/test_kl_distill_weight_plumbed.py \
  src/tac/tests/test_train_renderer_auth_eval_wiring.py \
  -q
# 70 passed
```

Next KL implementation unit:

1. Add `src/tac/kl_config.py` typed policy and compatibility normalization.
2. Serialize standard KL policy/provenance into train summaries, archive
   metadata, remote provenance, and adjudication provenance.
3. Add preflight checks for missing KL policy, teacher/student roundtrip
   contract, high-weight scale review, and unsafe JBL/PoseNet-safety wording.
4. Add adjudication blockers for KL/JBL/distill-active archives without exact
   CUDA non-collapse evidence.

## 2026-05-01T01:02Z Nextwave 3 Execution State

Landed artifacts:

- `experiments/select_renderer_blob_perturbation_basis.py` and
  `src/tac/tests/test_select_renderer_blob_perturbation_basis.py`.
- PFP16 A++ official response plan directory:
  `experiments/results/official_component_response_pfp16_a_plus_plus_20260501_codex_r1/`.
  The plan is symmetric over eps `[-2,-1,0,+1,+2]`, mutates five ASYM
  renderer quantized-weight payload bytes only, and records
  `auth_eval_required=cuda`.
- `src/tac/kl_config.py` and `src/tac/tests/test_kl_config.py`.
- Alpha visual-primitives packet and bounded CLI controls in
  `experiments/diagnose_nerv_geometry.py`.

Official response queue state:

- Dry-run queue spec exists for
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r1`.
- Source-manifest closure exists locally and includes every plan-listed
  archive. The queue command is ready to submit once Lightning SSH auth works.
- Non-dry-run is blocked by Lightning public-key rejection after setup script
  regeneration. This is an infrastructure auth blocker, not a missing plan or
  repo-artifact blocker.

Lane 12 / Alpha state:

- Existing exact CUDA Lane 12 jsonfix40 remains rejected:
  score `26.03719330455429`, archive `296478` bytes, PoseNet
  `49.7784996`, SegNet `0.03528685`, SHA
  `864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97`.
- Existing full Alpha scalar artifact versus Lane G/PFP16-equivalent masks has
  global mask disagreement `0.012303928799099393`, 25 residual regions, and
  exploratory pass `false`.
- New visual-primitives full-sequence run was CPU-cut off twice; the tool now
  supports bounded visual frame stride and should be run on predecoded masks or
  a longer worker budget. This is diagnostic only.

Provider telemetry:

- Vast structured live snapshots show zero live instances.
- Lightning structured state has terminal exact-eval records plus one
  official-response dry-run; no non-dry-run official response submitted.
- Modal local state is stale, with six historical `not_ready` call IDs; no
  live provider API evidence was collected in this pass.
- No MCP helper process is running.

Next wall-clock-critical path:

1. Fix Lightning SSH public-key acceptance, then run
   `scripts/lightning_repro_workspace.py` without `--dry-run` for the official
   response manifest.
2. Submit `scripts/launch_lightning_batch_job.py component-response` without
   `--dry-run` and with `--require-passed`.
3. Harvest official response curves and assemble `component_sensitivity_v1`
   only if response curves, maps, stability, sample plan, SHA/bytes, and exact
   CUDA custody all pass.
4. Wire `src/tac/kl_config.py` into `TrainConfig`, remote provenance,
   adjudication, and preflight so KL-family lanes cannot evade the policy
   object.
5. Use official component response to drive OWV3/NWCS/Alpha decisions; do not
   spend exact CUDA on blind scalar OWV3 grids or scorer-sensitive retraining.

## 2026-05-01T01:36Z Nextwave 3 KL Runtime Hardening Delta

Additional landed work:

- `distillation_policy_v1` is no longer a standalone schema only. It is forced
  through `TrainConfig`, generic `Trainer`, and `SegMapTrainer` construction.
- Active KL/JBL/distillation profile normalization is now a strict preflight
  path via `check_distillation_policy_schema_clean`.
- Training artifacts now include a canonical policy hash wherever this pass
  touched the save path, reducing sidecar/provenance drift risk.
- Training proxy outputs are stamped as non-score/non-promotion evidence and
  require exact CUDA auth eval before claims.

Remaining KL custody work:

1. Add strict harvested-artifact provenance validation for remote
   `provenance.json`/adjudication JSON, including policy format, schema
   version, policy hash, CUDA device, exact archive SHA/bytes, and component
   non-collapse gates.
2. Extend remote script scans so any KL/JBL/distill flags require explicit
   policy provenance and exact CUDA adjudication before promotion.
3. Wire `optimize_poses.py` KL/Pose-TTO provenance through the same schema and
   hash contract.

## 2026-05-01T01:30Z Continuation Telemetry / Research Intake

Live telemetry and blockers:

- Vast: no live instances from the latest structured/local audit.
- Modal: task count remains zero from the latest ops audit; historical
  `not_ready` rows are stale and not spend truth.
- Lightning: no running jobs; non-dry-run official component-response submit
  is still blocked by SSH public-key rejection, not by repo artifact
  readiness.
- MCP helpers respawned under the outer runtime during this pass and were
  killed by exact PID. User-level Claude/Cursor MCP JSON files contain empty
  `mcpServers`; repo-owned MCP config probe is clean.

Security:

- Confirmed public advisory scope: PyPI `lightning` 2.6.2/2.6.3, reported on
  April 30, 2026, import-time Mini Shai-Hulud credential-stealer behavior.
- Active local environment: `lightning_sdk==2026.4.10`, no `lightning`, no
  `pytorch-lightning`, strict scan OK.
- Hardened scanner to include the newer reported 2.6.3 `start.py`, malicious
  `lightning/__init__.py`, wheel hashes, and pip/uv cache artifact detection.

Research intake:

- arXiv:2604.26919v1 and Tencent Training-Free GRPO are useful as
  control-plane discipline only: dual readouts, sparse/winner-take allocation,
  warm-ramp pressure, and grouped proposal review.
- PufferLib is deferred. It is not wall-clock efficient against exact CUDA
  archive eval until a cheap surrogate has rank-correlation anchors.
- DeepSeek visual primitives should inform Alpha diagnostics: boxes,
  centroids, polylines, temporal tracks, component failures, and perturbation
  atoms tied to official component-response curves.

Next admissible order:

1. Fix Lightning SSH auth or switch exact-response microbatch to another
   trustworthy CUDA runner with identical custody.
2. Submit the official component-response microbatch with `--require-passed`.
3. Build `component_sensitivity_v1` only after exact CUDA response curves,
   stability, sample plan, maps, SHA/bytes, and custody pass.
4. Use that packet to pick OWV3/NWCS/Alpha exact-eval finalists.
5. Run an offline deterministic bandit replay on historical exact-eval cards
   before adding any RL/PufferLib dependency.

Alpha diagnostic closeout:

- The previously timing-out visual-primitives run is now feasible through
  bounded streaming/cache controls. Artifact:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_pfp16_visual_primitives_bounded_20260501.json`.
- The artifact is empirical CPU diagnostic only and explicitly records
  no-promotion/no-score/no-exact-eval-claim metadata under
  `visual_primitives`.
- It does not clear Lane 12 L2. Recommendation remains
  `repair_or_retrain_before_exact_eval_spend`; exact-eval spend gate failed on
  global disagreement, 2px-boundary disagreement, pair-transition
  disagreement, critical missing rate, and critical missing area rate.

## 2026-05-01T01:58Z Live Telemetry: Lightning Jobs Active, Studio CPU-Only

Provider state:

- Lightning interactive Studio SSH is reachable through `scratch-studio-devbox`
  and uses the expected alias policy, but the current shell has no GPU:
  `torch_cuda_available=false`, `torch_cuda_device_count=0`, `nvidia_smi=null`.
  This is a Studio-runtime state issue, not evidence that Batch Jobs are CPU.
- Lightning Batch Jobs now have three official component-response attempts:
  r1 T4 `Running`, r2 T4_SMALL `Running`, r3 T4 no-gate `Pending`.
  Jobs must be judged from their terminal artifacts, especially
  `lightning_runner_preflight.json`; do not infer job GPU state from the
  interactive Studio shell.
- MCP cleanup remains clean: `scripts/kill_orphaned_mcp_processes.py --strict`
  reports zero matched and zero remaining MCP helper processes.

Critical implementation deltas:

- Added a remote runtime CUDA probe to `scripts/lightning_repro_workspace.py`.
  `--ssh-check-only --require-cuda` now fails closed when the Studio shell is
  CPU-only and writes structured diagnostics.
- Integrated Lightning SSH static policy hardening from the provider-auth
  worker: preflight catches disabled host-key checking, `/dev/null`
  known-hosts, and bare provider-host usage.
- Submitted `official_component_response_pfp16_a_plus_plus_20260501_codex_r3_t4_no_gate`
  without `--require-passed` because the response plan has no nonzero
  `predicted_delta` fields. This preserves diagnostic official CUDA curves
  even when promotion gates are incomplete.

Next live actions:

1. Refresh r1/r2/r3 until terminal.
2. Harvest successful terminal artifacts with validation. For r1/r2 failures
   caused only by gate failure after curves exist, harvest without
   `--require-passed` and label as diagnostic official-response evidence.
3. Build `component_sensitivity_v1` only after CUDA response curves, maps,
   stability, sample plan, and exact custody are all present.
4. Keep interactive Lightning CUDA work blocked until
   `scripts/lightning_repro_workspace.py --ssh-check-only --require-cuda`
   succeeds.

## 2026-05-01T02:15Z Live Telemetry: Stale Lightning Jobs Retired, r4 Clean Queue

- r1 forensic result: fail-closed supply-chain gate on a stale snapshot before
  `lightning_runner_preflight.json`, DALI bootstrap, input preflight, official
  response profiler, or response curves. Classification: harness/config
  failure only; no lane-performance evidence.
- r2 is `Failed` and r3 is `Stopped`; both are stale-snapshot jobs with old
  `tools/lightning_*` wrappers. Do not harvest them for promotion. Inspect
  only if a future billing/provenance audit needs exact failure artifacts.
- r4 clean job is queued through the deterministic path:
  manifest SHA `80d44b40b4048ee1d2c7ba850e1e98e45025eda65b248b12a494d6e1fdf1928e`,
  command SHA `d9eec67b70b20b938dc76b66b34e0f498cc7d92e5307348c8798c0aa072a63c0`,
  job name `official_component_response_pfp16_a_plus_plus_20260501_codex_r4_clean_t4_stateful`.
- Next telemetry action: refresh r4 until terminal. If completed, run
  `scripts/launch_lightning_batch_job.py harvest-component-response-ssh
  --state-path .omx/state/lightning_batch_jobs.json --job-name
  official_component_response_pfp16_a_plus_plus_20260501_codex_r4_clean_t4_stateful
  --ssh-target scratch-studio-devbox --expected-baseline-archive-sha256
  0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
  --expected-baseline-archive-size-bytes 686635` without `--require-passed`
  for diagnostic curves. Add `--require-passed` only after a promotion-grade
  plan includes nonzero predicted-delta gates.

## 2026-05-01T02:38Z Live Telemetry: r4 Failed Cleanly, r5 Queued

- r4 is `Failed`, but it is not lane evidence. Remote artifacts prove the job
  passed strict supply-chain, hash-pinned DALI bootstrap, and CUDA T4 runner
  preflight. Failure occurred at component-response input preflight because the
  response plan carried a host-local `/Users/adpena/.../contest_auth_eval.json`
  path. No official response curves were emitted.
- Fixed and retested the bug class before requeue:
  explicit baseline eval JSON now overrides stale plan metadata; future plans
  emit repo-internal portable paths; submit closure rejects absolute point
  archives/per-point JSON; Lightning staging and harvest copy operations now
  use noninteractive SSH options end to end.
- r5 is the active job:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r5_explicit_baseline_t4`.
  It uses portable plan
  `experiments/results/official_component_response_pfp16_a_plus_plus_20260501_codex_r5_portable_plan/official_component_response_plan.json`,
  manifest SHA `ae3028935151c8e8e8f57315fa2a4f54edbfaebf3e6fd6c56064824e36f7e7e4`,
  command SHA `182c287d986a4fce61dbf12871b1e985bf01c4715b8e897e980e44d7e9c6ffa7`.
- Latest r5 status at `2026-05-01T02:38:30Z`: `Pending`, cost `0.0`.
  Next action is refresh until terminal, then run state-derived
  `harvest-component-response-ssh` without `--require-passed` to preserve
  diagnostic curves. Add a promotion pass only after the plan has valid
  prediction gates.

## 2026-05-01T02:55Z Live Telemetry: r5 Running, r6 Pending, More Closure Hardened

- Lightning live queue:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r5_explicit_baseline_t4`
  refreshed to `Running` at `2026-05-01T02:48:23Z`; r6 T4_SMALL race
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r6_t4_small_race`
  refreshed to `Pending` at `2026-05-01T02:48:24Z`.
- r6 staging passed remote manifest verification and clean environment scan.
  Manifest SHA:
  `91cd1e8011a7045a3068b0a2a4a74b0f842be6b8be4232da91473e6445780684`.
  Submit command SHA:
  `8cd111eb0b3448c1f9143929a96b0fd990afd361e6488c66a2c7c2ed0086deec`.
- Interactive SSH checks of the Studio path did not show live job output dirs
  for r5/r6. Treat this as provider artifact isolation; do not infer failure
  or CPU fallback from the interactive filesystem. Continue stateful refresh
  and harvest only terminal SDK artifacts.
- Provider telemetry audit: Vast live inventory is empty despite stale local
  trackers; Modal apps have `Tasks=0`; no active Modal/Vast compute needs
  killing. Modal stale call IDs remain forensic/poll-only and are not
  promotable unless rerun through canonical CUDA auth eval.
- Sensitivity audit: no promotable local `component_sensitivity_v1.json`
  exists. Current OWV3 Fisher map is CUDA-authored but proxy/Fisher evidence,
  not official component-response evidence. Do not spend another OWV3 exact
  eval until official CUDA component-response curves, maps, stability, and
  custody can be assembled into a valid component-sensitivity artifact.
- Additional permanent DX hardening landed:
  source manifests for Lightning exact-eval/component-response now reject
  traversal/absolute/hidden/resource-fork/duplicate paths; direct library SSH
  harvest rejects bare `ssh.lightning.ai`; exact-eval submit validates staged
  queue-metadata baseline JSON custody.
- J-NWC/NWCS readiness audit: fake/proxy sensitivity is blocked for promotion,
  but CUDA promotion still needs a prebuilt corpus-manifest + replay-root path
  so remote scripts cannot regenerate a divergent corpus. A worker is assigned
  to implement that fail-closed custody path.

Latest refresh at `2026-05-01T02:54Z`: r5 remains `Running` on T4 with
cost `0.11215556`; r6 is also `Running` on T4_SMALL with cost `0.007388889`.
Modal recovery polls for `lane_sa_v4`, `lane_sc_plus_plus_v4`, `mae_v_v2`,
`q_faithful_v3`, `stc_cuda`, and `sz_phase2_v2` all returned
`STILL RUNNING`, while `modal app list` showed zero live tasks. Treat these
as unresolved Modal-control-plane records only; no artifacts were harvested.

## 2026-05-01T03:10Z Live Telemetry: r5 Harvested, r6 Still Running

- r5 reached `Completed` and was harvested through state-derived SSH into
  `experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_codex_r5_explicit_baseline_t4`.
  Artifact validation is local and canonical; baseline archive SHA/bytes match
  the PFP16 A++ anchor.
- r5 interpretation: official CUDA component-response diagnostic evidence,
  not promotion evidence. Coverage, finite values, signal, and zero repro pass.
  Promotion fails because prediction deltas are absent from the diagnostic
  perturbation plan, so map calibration gates were not implemented. This is
  expected and blocks `component_sensitivity_v1` assembly until real component
  maps plus predicted deltas/stability/sample-plan evidence exist.
- Observed official response signal from r5:
  PoseNet max absolute delta `0.0003012800000000001`, SegNet
  `1.3420000000000099e-05`, combined `0.006991338976567674`.
- r6 remains `Running` on T4_SMALL at `2026-05-01T03:07:35Z`, cost
  `0.048555557`. It is scientifically redundant unless we want cross-machine
  reproducibility of the same diagnostic no-prediction plan. If it completes,
  harvest without `--require-passed`; do not promote.
- r6 completed and was harvested without `--require-passed`. It matches r5 as
  a cross-machine diagnostic replicate, not a promotion packet. R6 has the
  same promotion blockers (`missing_prediction_deltas`,
  `prediction_error_gate_failed`) and near-identical deltas: max r6-r5
  absolute difference across the epsilon ladder is approximately `4.0e-7`
  PoseNet, `6.0e-8` SegNet, and `8.3e-6` combined.
- Fixed r5 harvest blocker: read-only copied validation JSON can no longer
  break local mirror validation; artifact validation writes now use atomic
  replace and chmod.
- J-NWC/NWCS custody patch is landed and parent-verified (`42 passed` focused):
  remote scripts can consume exact prebuilt corpus manifests and replay roots,
  and NWCS promotion fails closed on mismatched/missing corpus-manifest custody.
- Lane 12 Alpha-Geo current Lane G rerun confirms the existing block:
  `overall_pass=false`, global disagreement `0.012303928799099393`, 2px
  boundary disagreement `0.14883144511692872`, missing-component rate
  `0.4611606740560512`. No Lane 12 retraining or exact eval should run until
  a successor archive passes Alpha-Geo plus pose-regeneration provenance.

## 2026-05-01T05:11Z No-MCP Closure And Lightning Repo Intake

- MCP noise cleanup is now complete locally: known config files, disabled
  backups, plugin caches, tool-output caches, OAuth/state files, and
  `.playwright-mcp` artifacts were deleted from local tool homes. Final MCP
  filesystem scans returned empty, and
  `scripts/kill_orphaned_mcp_processes.py --strict --json` returned no live
  matches.
- The cleanup tooling was hardened after it matched audit `find` processes
  containing `*model.context*` in the command line. The matcher now detects
  actual helper launches and ignores search/audit commands. Regression
  coverage passed: `6 passed`.
- r5/r6 interpretation is tightened: both remain useful cross-machine
  diagnostic response packets, but non-promotable. Their zero point reused an
  external baseline JSON rather than a same-run zero archive; promotion now
  requires same-run eps=0 under `--require-passed`.
- Official component-response planning now has a pre-response prediction
  artifact path:
  `experiments/build_component_response_prediction_deltas.py` ->
  `official_component_response_prediction_deltas_v1` ->
  fail-closed perturbation plan ingestion. This blocks arbitrary/post-hoc
  predicted-delta JSON and observed-response leakage.
- Lightning ecosystem repositories are in active research intake:
  LitModels, lightning-thunder, and utilities. Initial primary-source verdict:
  adapt patterns only; do not add broad dependencies or install any path that
  pulls the PyPI `lightning` package. Detailed ledger:
  `.omx/research/lightning_ecosystem_repo_intake_20260501_codex.md`.

Next execution order:

1. Produce real CUDA component maps, response sample plan, and stability
   packet.
2. Build map-projected prediction deltas from those maps and the byte
   perturbation basis.
3. Rebuild official component-response plan with structured predictions and
   run Lightning CUDA response with `--require-passed` so same-run zero is
   mandatory.
4. Assemble `component_sensitivity_v1` only after all explicit gates are true.
5. Resume OWV3 exact eval only if the sensitivity artifact is promotable and
   the byte frontier is justified against the PFP16 A++ anchor.

## 2026-05-01T06:01Z Active Queue And Fastest-Wall-Clock Order

- Active CUDA queue: two fixed diagnostic component-sensitivity jobs are
  submitted on Lightning Batch Jobs, T4 and L40S in parallel:
  `component_sensitivity_pfp16_a_plus_plus_cuda_fisher_20260501_r2` and
  `component_sensitivity_pfp16_a_plus_plus_cuda_fisher_l40s_20260501_r2`.
  Latest refresh shows both `Pending` with zero cost.
- The r1 sensitivity failures are closed as a deterministic harness bug:
  negative epsilon ladders must be emitted as `--response-epsilons=-...`.
  The fix is in the job generator and covered by tests.
- Immediate harvest plan when either job becomes terminal:
  `scripts/launch_lightning_batch_job.py harvest-component-sensitivity-ssh`
  using `.omx/state/lightning_batch_jobs.json`, the job name, and
  `scratch-studio-devbox`. Do not `scp -r`; copy only compact validated
  artifacts.
- After the first valid CUDA diagnostic maps arrive, build map-projected
  prediction deltas, rebuild the official component-response plan with
  structured predictions, and submit official CUDA response with
  `--require-passed`. The second machine result should be used as a
  reproducibility/adversarial cross-check, not as an independent score claim.
- Lane 12/Alpha remains off the compute-critical path until geometry is fixed:
  current `jsonfix40` Alpha-Geo fails badly and the L2 clearance packet is
  absent. Spending exact eval on that archive has negative EV.
- J-NWC/NWCS remains staged behind sensitivity evidence: corpus/replay custody
  is in place, but Conv2d-only diagnostic maps are insufficient for promotion.
  The next useful J-NWC/NWCS work is exact corpus-manifest rehearsal and
  all-parameter sensitivity coverage once CUDA maps exist.
- Resource state: Vast has no live instances; Modal has zero live tasks and a
  stale harvest backlog only. Keep Lightning GPU queue primary for exact CUDA
  signal, Modal for cheap build/smoke/forensics, and Vast idle until a lane has
  a clean dispatch packet.

Next execution order:

1. Refresh both Lightning r2 jobs until one reaches terminal state.
2. Harvest terminal diagnostic sensitivity artifacts through the state-derived
   SSH wrapper and validate archive custody.
3. Convert harvested CUDA maps into prediction deltas and an official response
   plan with absolute-magnitude error semantics.
4. Submit official component-response Batch Job with same-run eps=0 required.
5. Only then assemble `component_sensitivity_v1`, unlock OWV3/NWCS exact eval,
   and resume stack experiments against the PFP16 A++ byte frontier.

## 2026-05-01T06:17Z Live Queue Update

- L40S diagnostic sensitivity completed and was harvested:
  `experiments/results/lightning_batch/component_sensitivity_pfp16_a_plus_plus_cuda_fisher_l40s_20260501_r2`.
  It is CUDA/L40S, 600-pair diagnostic Fisher evidence with exact baseline
  archive custody, but remains non-promotable.
- A fresh-basis official response packet was built from the L40S maps:
  `experiments/results/official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex`.
  This avoids reusing r5/r6 stale response plans and preserves pre-response
  prediction provenance.
- Submitted official CUDA component-response calibration job:
  `official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex_lightning_l40s`.
  It is `Pending` on L40S with `--require-passed`; failure is useful gate
  evidence, not a lane kill.
- T4 sensitivity remains `Running`. T4 is still the preferred promotion-source
  diagnostic map; L40S is the fastest calibration signal and cross-machine
  check.

Next execution order:

1. Refresh `resp_l40s` until terminal; harvest with
   `harvest-component-response-ssh`. If it fails `--require-passed`, harvest
   without `--require-passed` only for forensic diagnosis and keep blockers.
2. Refresh and harvest T4 sensitivity when terminal.
3. If L40S response passes, repeat or validate on T4 before any promotion
   assembly; if it fails, inspect component gate/prediction errors and update
   the signed/magnitude prediction model before another exact response spend.
4. Do not assemble promotable `component_sensitivity_v1` from diagnostic
   Fisher maps without a reviewed promotable map artifact path; current maps
   are for prediction/planning and official-response calibration.

## 2026-05-01T06:37Z Telemetry Update And Next-Wave Decision

- T4 diagnostic sensitivity r2 completed and was harvested. It is CUDA/T4,
  full 600-pair diagnostic Fisher/proxy evidence with exact baseline archive
  custody, but remains non-promotable. Use it for comparison/calibration only
  until a direct finite-difference map source exists.
- L40S official response r7 failed. The useful signal is forensic:
  coverage, finite values, same-run zero, and signal were present, but
  prediction error failed catastrophically. Additionally, same-run L40S
  baseline components drifted from the supplied PFP16 A++ T4 baseline JSON,
  so L40S response curves cannot certify the PFP16 A++ sensitivity map path.
- Hardening landed after this failure: future official-response jobs now gate
  external-baseline agreement through `external_baseline_repro` whenever an
  external baseline JSON is supplied. This closes the "same-run-only zero hides
  runner drift" meta-bug.
- T4 official response r7 remains running. Because it was queued before the
  new hardening, harvest it when terminal and manually compare its eps=0
  components against
  `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json`
  before using the curves. Passing old-code gates are not sufficient if
  external-baseline drift is present.

Next execution order:

1. Refresh and harvest the T4 official response when terminal.
2. If T4 response passes prediction and external-baseline agreement, use it
   only as official response evidence; map certification still waits for
   direct finite-difference CUDA maps.
3. If T4 response fails prediction, fit R8 signed/quadratic response from the
   harvested official curves and rebuild fresh prediction deltas; do not reuse
   the L40S r7 magnitude-only calibration as promotable evidence.
4. Implement or dispatch direct finite-difference CUDA map generation with
   full parameter coverage for OWV3/NWCS, then certify maps through
   `experiments/certify_component_sensitivity_maps.py`.
5. Use the Alpha primitive contract to unblock decoded-baseline Lane 12
   retraining design, still build-only/no exact eval until geometry and L2
   clearance pass.

## 2026-05-01T06:55Z T4 R7 Response Closed And R8 Direction

- T4 official response r7 is terminal `Failed` and harvested:
  `experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex_lightning_t4_parallel`.
  The packet is CUDA/T4 forensic evidence only. It cannot certify maps or
  support stack decisions.
- Failure mode is twofold:
  prediction-error gates failed on all components, and the same archive's
  eps=0 baseline drifted from the PFP16 A++ T4 anchor. The old-code r7 packet
  lacks the new explicit external-baseline gate, but the manual comparison is
  decisive: treat this as runner/scorer calibration plus prediction-model
  failure, not method retirement.
- Fastest-wall-clock next action is not another Fisher-map response with the
  same R7 model. Build R8 around direct finite-difference CUDA maps or, if
  using the harvested r7 curves only for model fitting, fit a signed/quadratic
  response model with explicit runner identity and external-baseline repro
  gates before spending another official response job.
- Lane 12 Alpha should proceed as decoded-baseline build-only retraining using
  the `alpha_geo_primitive_contract_v1` contract consumer now landed. Exact
  eval remains blocked until Alpha-Geo exploratory gates and L2 clearance pass.

Next execution order:

1. Run or queue direct finite-difference CUDA component-sensitivity map
   generation with full 600-pair coverage; Fisher/proxy maps stay planning
   only.
2. Generate archive-byte perturbation basis and pre-response prediction deltas
   from those direct-FD maps, then run official response with external-baseline
   repro required.
3. Certify maps only with prediction-deltas SHA, perturbation-basis SHA,
   official curves, baseline custody, stability, sample coverage, and three
   clean review passes.
4. Use the Alpha contract consumer for build-only NeRV retraining and run
   Alpha-Geo diagnostics before any CUDA exact eval spend.
5. Keep Lightning status anomaly records under review; status alone never
   promotes or kills a lane.

## 2026-05-01T07:10Z Direct-FD Packet Ready, Sharding Is Next Bottleneck

- Direct renderer CUDA finite-difference component-sensitivity is now a
  first-class Lightning job mode and harvest artifact type. It is still
  diagnostic/no-score, but the local and remote validators now preserve it as a
  certification handoff candidate instead of rejecting it as non-Fisher.
- Validation now checks `.pt` map metadata, not only JSON summaries and
  response curves. Any score claim, promotion claim, official-response claim,
  canonical-scorer claim, or source mismatch fails closed before a map can be
  used downstream.
- Lane 12 Alpha remote dispatch now requires and forwards
  `ALPHA_PRIMITIVE_CONTRACT`; build-only decoded-baseline retraining cannot
  silently run without the Alpha primitive contract.
- Grand Council research note from Gauss: the unsharded direct-FD packet is
  valid but wall-clock suboptimal. Current PFP16 candidate has hundreds of
  eligible channels, so direct-FD should be sharded by deterministic
  layer/channel ranges with a merge validator before spending serious T4 time.

Immediate execution order:

1. Implement deterministic direct-FD layer/channel sharding and merge
   validation for `experiments/profile_component_sensitivity.py` artifacts.
2. Restage Lightning source after the current hardening; do not reuse stale
   r2 Fisher manifests.
3. Submit direct-FD shards on T4/equivalent with full 600-pair coverage,
   `--promotion-finite-difference`, external baseline custody, and state-derived
   harvest.
4. Build prediction deltas and archive-byte perturbation basis from merged
   direct-FD maps, then run official component response with
   external-baseline repro gates.
5. Certify maps only after official response gates, stability gates, custody
   SHA checks, and three clean adversarial passes.

Telemetry snapshot:

- Lightning local state currently has no running job records; recent scientific
  jobs are terminal/harvested/failed. Two historical records are `Stopped` and
  require no interpretation as active evidence.
- Vast reconciliation reports `live_count=0`. Local Vast trackers are stale
  relative to live API and must not be used as live evidence; active-dispatch
  rows for old instance IDs are missing live counterparts.

## 2026-05-01T07:55Z Queue-Diverse Direct-FD Execution And Alpha Repair Signal

Deadline context: user confirmed the contest deadline is 2026-05-03 00:00
local time, with the working time near 2026-05-01 02:25 local. The governing
policy is therefore highest-EV real signal first, with harness changes only
where they remove immediate signal-loss risk.

Direct finite-difference component-sensitivity execution state:

- Lightning reproducible staging succeeded at
  `.omx/state/lightning_direct_fd_sensitivity_20260501T0226_codex_manifest.json`.
  Manifest SHA-256:
  `c2f414c52fcda142fcfbfd7ff237f9ae0434249a41129b4d48cc42573b5fd705`.
- Baseline archive for every shard:
  `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip`,
  SHA-256 `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  bytes `686635`.
- Submitted Lightning queue-diverse duplicate waves, all using the same
  deterministic 16-shard topology and non-score diagnostic metadata:
  L40S `g6e.4xlarge`, T4 `g4dn.2xlarge`, and RTX PRO `g7e.4xlarge`.
- Latest refresh
  `.omx/state/lightning_refresh_direct_fd_allwaves_20260501T074759Z.jsonl`
  shows all 48 Lightning shard jobs still `Pending`; cost was still zero at
  refresh. Status is telemetry only, not evidence.
- Added bounded Lightning SSH transport hardening after a transient
  `kex_exchange_identification: read: Connection reset by peer` during rapid
  pre-submit remote supply-chain preflights. The patch retries known
  transport resets but keeps auth and supply-chain failures fail-closed.

Modal queue-diversity fallback:

- Added `experiments/modal_component_sensitivity_shards.py`, a lightweight
  A10G/T4 direct-FD shard launcher that uploads the 686KB PFP16 archive bytes,
  mounts source plus upstream assets, zip-slip-safely extracts
  `renderer.bin`, `masks.mkv`, and `optimized_poses.bin`, and runs the same
  `profile_component_sensitivity.py --promotion-finite-difference` shard
  command.
- Dispatched Modal A10G label
  `pfp16_direct_fd_modal_a10g_20260501`, shards `0..15`, app
  `comma-component-sensitivity`. Call IDs are recorded in
  `experiments/results/modal_component_sensitivity/pfp16_direct_fd_modal_a10g_20260501/modal_call_ids.json`.
- Initial recovery reports all 16 Modal shards still pending, with no failed
  shards and no harvested artifacts yet. Modal output remains
  `diagnostic_cuda_modal_direct_renderer_finite_difference`, `score_claim=false`,
  and `promotion_eligible=false`.

Alpha/Lane 12 repair signal:

- Ran the admissible CPU empirical Alpha-Geo residual materialization:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_pfp16_residual_regions_20260501T073504Z.json`.
- This packet is `device=cpu`, `score_evidence_grade=empirical`; it cannot
  promote, rank, kill, or justify exact-eval spend.
- Operational finding: the dominant repair target is lower-field lane-marking
  collapse and temporal boundary underfit. The top 200 residual regions total
  87,652 pixels with 80,305 critical-class pixels across 165 frames; dense
  clusters include frames 267-268, 304-305, 321-323, 583-584, 1034-1046,
  1054-1059, and 1191-1198.
- Next Alpha artifact, still pre-L2 and non-score, is a larger
  `alpha_geo_1` primitive-contract packet with 1000 residual regions and
  component-track diagnostics. Lane 12 retraining and exact eval remain
  blocked until valid L2 clearance exists.

Immediate next execution order:

1. Poll Modal and Lightning direct-FD shards every few minutes; harvest
   terminal shards state-derived only.
2. Merge the first complete 16-shard source with
   `experiments/merge_component_sensitivity_shards.py`; incomplete merges are
   planning-only and non-handoff.
3. Build prediction deltas and archive-byte perturbation basis from merged
   direct-FD maps.
4. Run official component response with external-baseline repro gates.
5. Only then certify `component_sensitivity_v1` and route OWV3/NWCS exact-eval
   candidates.

## 2026-05-01T08:15Z Lightning SSH Permanence Diagnosis And Hardening

Root cause / operating model:

- SSH reachability and GPU attachment are separate Lightning states. Current
  probe via hardened alias `scratch-studio-devbox` reaches the Studio shell,
  but the interactive Studio has no `nvidia-smi` and no default `python` in the
  shell path; therefore it is CPU-only at this moment.
- This is not an SSH-key failure and not a repo staging failure. It is
  Lightning machine-policy behavior: a Studio can remain reachable while the
  GPU machine has switched back to CPU after inactivity. Promotion-grade CUDA
  must therefore use Batch Jobs with explicit machine selection and per-job
  runner preflight, or an interactive Studio that has just passed
  `--require-cuda`.
- Online/current-source review found that `lightning-sdk` exposes explicit
  Studio machine switching and Batch Jobs with machine selection. OpenSSH
  documentation supports the exact controls now used here:
  `ConnectTimeout`, `ConnectionAttempts`, `ServerAliveInterval`,
  `ServerAliveCountMax`, and `ControlMaster`/`ControlPersist`.

Permanent hardening landed:

- Added `scripts/configure_lightning_ssh.py`, a reproducible alias installer
  that writes a managed SSH config block with public-key-only auth,
  `BatchMode yes`, bounded connect attempts, client keepalives,
  opportunistic multiplexing, `StrictHostKeyChecking accept-new`, and a
  persistent known-hosts file. This replaces the Lightning UI one-liner for
  contest custody because the UI helper may disable host-key checking and send
  host keys to `/dev/null`.
- Hardened legacy Lightning shell wrappers:
  `scripts/lightning_auth_eval.sh`,
  `scripts/lightning_deploy_asymmetric.sh`,
  `scripts/lightning_auth_eval_renderer.sh`,
  `scripts/pfp16_a_plus_plus_exact_t4_eval.sh`,
  `src/tac/deploy/lightning/deploy.sh`,
  `tools/lightning_run.sh`, and `tools/lightning_monitor.sh`.
- Hardened legacy Python dispatcher
  `src/tac/deploy/lightning/lightning_dispatch.py`: public-key-only
  noninteractive SSH/SCP, keepalives, bounded attempts, and rejection of bare
  `ssh.lightning.ai` as a target.
- Extended `src/tac/preflight.py` so Lightning static SSH policy scans
  `src/tac/deploy/lightning/` in addition to scripts/tools/runbooks/AGENTS.
- Added `cloud` optional dependency extra in `pyproject.toml` and `uv.lock`
  for reproducible operator tooling: `lightning-sdk`, `modal`, and `vastai`.
  Local environment now has `lightning-sdk==2026.4.23`, `modal==1.4.2`,
  `vastai==1.0.8`, and no installed `lightning` or `pytorch-lightning`.
- Local `~/.ssh/config` was updated with the same hardened
  `scratch-studio-devbox` options. This is operator-machine state, not a repo
  custody artifact; the reproducible source of the policy is the new script.

Current queue telemetry:

- Lightning direct-FD refresh
  `.omx/state/lightning_refresh_direct_fd_allwaves_20260501T081337Z.jsonl`
  shows 6 L40S shards running (`00,01,03,04,05,07`) and 42 pending
  (10 remaining L40S, all 16 T4, all 16 RTX PRO). No direct-FD Lightning shard
  has completed/harvested yet.
- Modal A10G r1 and r2 exposed dependency-closure bugs (`timm`, then
  `segmentation_models_pytorch`). r3 image includes both dependencies and
  dispatched all 16 A10G shards under label
  `pfp16_direct_fd_modal_a10g_20260501_r3`; latest recovery still shows all
  16 pending, with no failed shards and no artifacts yet.
- Alpha-Geo-1 primitive-contract diagnostic completed:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.json`
  SHA-256 `fcc7fcf9e22518cd95a5af4cb36aff189249c6248b5298f377ecc8ca66991a3e`
  and
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.primitive_contract.json`
  SHA-256 `e5da815b680ba5c02bf653dae8c77b4f6d12500461e45b06d0cfb0881be5c16e`.
  It remains CPU empirical/non-score.

Verification:

```bash
.venv/bin/python -m py_compile \
  scripts/configure_lightning_ssh.py \
  scripts/launch_lightning_batch_job.py \
  scripts/lightning_repro_workspace.py \
  src/tac/deploy/lightning/batch_jobs.py \
  src/tac/deploy/lightning/lightning_dispatch.py \
  src/tac/preflight.py \
  experiments/modal_component_sensitivity_shards.py \
  experiments/merge_component_sensitivity_shards.py

bash -n \
  scripts/lightning_auth_eval.sh \
  scripts/lightning_deploy_asymmetric.sh \
  scripts/lightning_auth_eval_renderer.sh \
  scripts/pfp16_a_plus_plus_exact_t4_eval.sh \
  src/tac/deploy/lightning/deploy.sh \
  tools/lightning_run.sh \
  tools/lightning_monitor.sh

.venv/bin/python -m pytest \
  src/tac/tests/test_configure_lightning_ssh.py \
  src/tac/tests/test_lightning_dispatch.py \
  src/tac/tests/test_preflight_meta_bugs.py \
  src/tac/tests/test_lightning_batch_jobs.py \
  src/tac/tests/test_lightning_repro_workspace.py \
  src/tac/tests/test_merge_component_sensitivity_shards.py -q

.venv/bin/python scripts/scan_lightning_supply_chain.py --strict
git diff --check
```

Observed result: `367 passed`; supply-chain scanner OK; Lightning static SSH
preflight OK; MCP process preflight OK; diff whitespace clean.
