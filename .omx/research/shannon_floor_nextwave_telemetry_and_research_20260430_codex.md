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

## 2026-05-01T09:40Z Wall-Clock Frontier Update

Work landed:

- Completed full direct-FD sensitivity coverage by merging:
  - mixed Lightning/Modal full merge:
    `experiments/results/lightning_batch/component_sensitivity_pfp16_direct_fd_mixed_l40s_rtxpro_modal_20260501_complete_merge_20260501T091302Z`
  - Modal A10G full merge:
    `experiments/results/modal_component_sensitivity/pfp16_direct_fd_modal_a10g_20260501_r3_complete_merge_20260501T0929Z`
- Added sensitivity-ranked perturbation basis selection. The selector now
  consumes `tac_score_sensitivity_map_v1` maps, ranks channels
  deterministically, records map custody, records atom sensitivity score/source,
  and still decodes every epsilon renderer variant.
- Added merged-shard response planning with explicit fail-closed custody:
  complete exactly-once merge required, source-shard baseline custody checked,
  summary/validation merge payloads cross-checked, and basis source archive
  plus atom original bytes verified against the baseline archive.
- Added a hard mathematical guard against the unit bug: direct-FD channel
  weight-space maps cannot generate archive-byte `predicted_delta` artifacts
  unless future byte-basis calibration explicitly certifies the map metadata.

Current clean next-dispatch packets:

- Mixed response-only official-response packet:
  `experiments/results/official_component_response_pfp16_direct_fd_mixed_complete_response_only_20260501`
  with plan SHA-256
  `d0026cf7ec5f22ff3de4e8402d3f3a660767eb64e890d779cb466933ed60c23f`.
- Modal A10G response-only official-response packet:
  `experiments/results/official_component_response_pfp16_direct_fd_modal_a10g_complete_response_only_20260501`
  with plan SHA-256
  `a460009d891a38287378164989ca8a417780a11f857ae998f5d724a5eaa310bf`.
- These are response-calibration jobs, not promotion-passed map certification
  jobs, because nonzero prediction deltas are intentionally omitted.

Operational blocker:

- Lightning SSH now fails after offering the configured key:
  `Permission denied (publickey)`. Fingerprint offered:
  `SHA256:af6xKc8r7y0WYc4FL6lGrvhHDT0qyo2gSruKhrk/c5Y`.
- This blocks reproducible Studio staging and remote supply-chain preflight.
  The repo config has been hardened and duplicate alias stanzas are pruned;
  remaining action is Lightning UI/account reauthorization of
  `~/.ssh/lightning_rsa.pub`.

Next fastest path:

1. User reauthorizes the Lightning SSH public key; then run
   `scripts/lightning_repro_workspace.py` staging for the response-only packet
   and submit `component-response` with explicit
   `--baseline-contest-auth-eval-json`, same-run zero baseline, and no
   `--require-passed`.
2. Harvest official response curves and use them to fit a byte-basis
   calibration contract. Only after that can `archive_byte_prediction_eligible`
   maps be generated.
3. Run `certify_component_sensitivity_maps.py` only after official response
   curves, stability gates, external-baseline reproduction, and three clean
   adversarial passes exist.
4. In parallel, keep Alpha/Lane 12 read-only diagnostics moving; retraining is
   still blocked by missing L2 clearance.

Verification addendum:

- Focused compile passed for the direct-FD planning, perturbation, prediction,
  and Lightning SSH config files touched in this loop.
- Focused tests passed:
  `src/tac/tests/test_build_component_response_perturbation_plan.py`,
  `src/tac/tests/test_select_renderer_blob_perturbation_basis.py`, and
  `src/tac/tests/test_configure_lightning_ssh.py` (`27 passed`).
- `scripts/kill_orphaned_mcp_processes.py --strict` returned clean.
- `scripts/scan_lightning_supply_chain.py --strict --quiet --json-out .omx/state/lightning_supply_chain_local_codex_20260501T0939Z.json`
  returned clean: no `lightning` / `pytorch-lightning` package present,
  `lightning-sdk==2026.4.23`, zero violations.
- `git diff --check` returned clean.

## 2026-05-01T09:50Z Lightning Response-Only CUDA Calibration Dispatch

Purpose:

- Convert the direct-FD sensitivity geometry into official archive-byte
  response curves. This is calibration evidence, not a score-improvement claim
  and not promotion-grade map certification yet.

Preflight and staging:

- SSH alias `scratch-studio-devbox` is now reachable again; plain Studio shell
  remains CPU-only, so interactive CUDA is still invalid for scoring.
- Lightning doctor passed with local supply-chain scan, SSH auth, remote
  supply-chain scan, and AWS T4 machine inventory.
- Full `upstream/` staging initially failed closed because hidden upstream
  metadata (`upstream/.devcontainer/...`) is forbidden in source manifests.
  Restaged only the scorer/runtime files and public video needed by the eval
  path, plus the response plan and baseline artifacts.
- Clean staged manifest:
  `.omx/state/component_response_direct_fd_mixed_response_only_20260501T0950Z_manifest.json`
  with SHA-256 `ca98d9a707ea5be6da5a69c045e89563a0bcd18917eb88242048dc918aebd907`,
  `1146` files, `179556072` bytes, and remote manifest verification OK.
- Component-response dry-run passed against this manifest and plan.

Dispatched jobs:

- `component_response_pfp16_direct_fd_mixed_response_only_t4_aws_20260501T0950Z`
  on `g4dn.2xlarge`, SDK-normalized job
  `component-response-pfp16-direct-fd-mixed-response-only-t4-aws-20260501t0950z`.
  Status after first refresh: `Pending`.
- `component_response_pfp16_direct_fd_mixed_response_only_t4_aws_small_20260501T0950Z`
  on `g4dn.xlarge`, SDK-normalized job
  `component-response-pfp16-direct-fd-mixed-response-only-t4-aws-small-20260501t0950z`.
  Status after first refresh: `Pending`.
- A GCP `n1-standard-8` duplicate was attempted but rejected before dispatch:
  Lightning reported that accelerator `n1-standard-8` was not found for the
  AWS cluster. No GCP job was created.

Harvest plan:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name component_response_pfp16_direct_fd_mixed_response_only_t4_aws_20260501T0950Z

.venv/bin/python scripts/launch_lightning_batch_job.py harvest-component-response-ssh \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name component_response_pfp16_direct_fd_mixed_response_only_t4_aws_20260501T0950Z \
  --ssh-target scratch-studio-devbox \
  --expected-baseline-archive-sha256 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f \
  --expected-baseline-archive-size-bytes 686635
```

Repeat the same commands for
`component_response_pfp16_direct_fd_mixed_response_only_t4_aws_small_20260501T0950Z`.

## 2026-05-01T10:00Z Large-Move Refocus, Alpha-Geo-0 Dispatch, Modal CUDA Hardening

User correction: the required path is a large move toward the Shannon floor,
not repeated millipoint shaving. Treat the `1.0368667951088641` perturbation
archive as useful telemetry/custody signal only; it is not the strategic
breakthrough.

Exact lower-score perturbation custody bundle preserved:

- Bundle:
  `experiments/results/frontier_candidate_pfp16_r7_eps_p2_20260501/`.
- Archive SHA-256:
  `c00393dda0736edb2b2a25e3108624b7571042c63e6c5b138b5168c1a28f1193`.
- Exact CUDA/T4 score:
  `1.0368667951088641`, archive `686636` bytes, PoseNet
  `0.00316061`, SegNet `0.00401883`, `600` samples.
- Status: component-gate review required, not clean promotion. It improves
  total score and PoseNet versus PFP16 A++, but SegNet is slightly above the
  strict internal component gate. Independent Lightning exact re-eval queued
  as `exact_eval_r7_p2_canonical_t4_20260501T1020Z` on `g4dn.xlarge`
  / T4_SMALL; first status `Pending`, cost `0.0`.

Large-move Alpha/Lane 12 action:

- Added `scripts/remote_lane_12_alpha_geo0_pose_regen.sh`.
- Purpose: stale-pose isolation for exact `jsonfix40` `masks.nrv`. The script
  does not retrain the mask codec. It decodes the exact candidate mask stream,
  regenerates `optimized_poses.bin` against those masks, rebuilds a
  deterministic archive with the same `renderer.bin` and `masks.nrv`, then
  runs canonical CUDA auth eval and JSON adjudication.
- Dispatch started through Vast retry wrapper with label
  `lane_12_nerv_alpha_geo0_pose_regen`; the label intentionally uses the Lane
  12 prefix so the no-new-retraining gate recognizes this as the Alpha
  causal experiment, not unrelated retraining.
- Interpretation rule: if PoseNet recovers, stale poses were the load-bearing
  failure and Alpha mask-payload collapse remains a high-EV route. If PoseNet
  remains collapsed, the exact `jsonfix40` mask geometry is incompatible and
  Alpha must move to decoded-baseline retraining plus sparse residual repair,
  still behind L2 clearance.

Modal exact-eval hardening:

- `experiments/modal_auth_eval.py` now fails closed around the canonical
  `experiments/contest_auth_eval.py --device cuda` path. It no longer uses
  direct `inflate_renderer.py` or CPU evaluation as an auth score.
- Modal result metadata remains non-promotable until adjudication validates
  archive custody, CUDA device, DALI/CUDA preflight, sample count, SHA, bytes,
  and component gates.
- Focused Modal tests passed (`9 passed`) and Alpha mask-codec probe tests
  passed (`35 passed`).

Alpha probe hardening:

- `experiments/paradigm_alpha_real_archive_eval.py` now defaults to the PFP16
  A++ archive, loads only a named safe mask member, records archive/member
  SHA custody, rejects zip-slip/hidden members, and labels outputs
  empirical/no-score.
- Runbook added:
  `docs/runbooks/alpha_lane12_large_move_next_actions.md`.

Immediate active queue:

- Vast: `lane_12_nerv_alpha_geo0_pose_regen` dispatch in progress.
- Lightning: `exact_eval_r7_p2_canonical_t4_20260501T1020Z` pending;
  response-only direct-FD calibration jobs still pending at zero cost.
- Modal: CUDA auth-eval wrapper ready; use it for independent exact-eval
  fallback only if Lightning remains queued or Vast Alpha-Geo needs
  independent reproduction.

## 2026-05-01T10:30Z Frontier Candidate Reproduction And Live-Lane Cleanup

Large-score focus remains Alpha mask-payload replacement and exact CUDA
frontier reproduction. No score claim below is promotable unless it has exact
CUDA custody, archive SHA/bytes, full sample count, and adjudication.

OWV3/Fisher frontier candidate:

- Local exact CUDA artifact discovered in
  `experiments/results/lane_g_v3_owv3_fisher_beta_20260501_LANDED/lane_g_v3_owv3_fisher_stack_results/`.
- Archive:
  `archive_lane_g_v3_owv3.zip`, SHA-256
  `57abe0fdf786d95b38325334b568e7a947143afe097ba189f214f2208492cb8f`,
  `638165` bytes.
- Local CUDA score evidence: `1.0160176664836693`, PoseNet
  `0.00360019`, SegNet `0.00401348`, `600` samples, RTX 4090.
- Local adjudication: A-grade score custody, promotion eligible under the
  internal component gates used for PFP16 comparison, not A++ until T4 or
  contest-equivalent reproduction.
- Lightning T4 reproduction queued as
  `exact_eval_owv3_fisher_beta_t4_20260501T1025Z`; current status `Running`.
  This is the current highest-priority reproduction because it could replace
  PFP16 as the frontier anchor if T4 custody agrees.
- Modal independent rerun inflated successfully but failed during
  `upstream/evaluate.py` with DALI/NVML internal-driver error before
  `contest_auth_eval.json` was produced. Classification: infrastructure
  failure, no score claim, no evidence against the archive.

R7 perturbation custody:

- Lightning T4 job `exact_eval_r7_p2_canonical_t4_20260501T1020Z` completed
  and was harvested via state-derived SSH artifact mirror.
- A++ T4 result: score `1.036860107631869`, archive `686636` bytes, SHA-256
  `c00393dda0736edb2b2a25e3108624b7571042c63e6c5b138b5168c1a28f1193`,
  PoseNet `0.00316055`, SegNet `0.00401878`, `600` samples.
- Adjudication: `promotion_eligible=true`, evidence grade `A++ contest T4`,
  component gates passed. This is a custody/telemetry improvement, not the
  strategic seven-tenths move.

Alpha-Geo-0:

- Vast retry attempts for `lane_12_nerv_alpha_geo0_pose_regen` failed on
  NVDEC/SCP infrastructure before producing valid artifacts. Classification:
  run abort, not evidence against NeRV or Alpha.
- Added and dispatched Modal T4 wrapper
  `experiments/modal_alpha_geo0_pose_regen.py` with label
  `lane12_alpha_geo0_pose_regen_modal_t4_20260501T1026Z`.
- The Modal wrapper keeps exact `masks.nrv`, regenerates poses, builds a
  deterministic archive, then runs canonical CUDA auth eval and adjudication.
  It is the active replacement path for the failed Vast Alpha-Geo-0 attempts.
- First Modal Alpha-Geo-0 dispatch used the pre-pin image and failed closed at
  CUDA/DALI/NVDEC preflight with `nvml error (999)` before score work. It is
  diagnostic only.
- Modal exact-eval/training wrappers now pin `nvidia-dali-cuda120==1.52.0`
  with the NVIDIA index, matching the audited Lightning DALI line. Pinned
  Alpha-Geo-0 dispatch:
  `lane12_alpha_geo0_pose_regen_modal_t4_20260501T1030Z_pinned_dali`,
  call `fc-01KQHHFXYWRY8184K37992EPXZ`; recovery failed closed at the same
  CUDA/DALI/NVDEC preflight with DALI `1.52.0` and `nvml error (999)`.
  Classification: Modal T4 infrastructure abort, no score claim, no evidence
  against Alpha-Geo-0.

Vast live cleanup and forensic notes:

- `scripts/reconcile_vast_dispatch_state.py --json --max-items 1000` now
  reports `live_count=0` after cleanup.
- Harvested and destroyed stale live instances for OWV3/Fisher, Lane 19
  snapshot, MAE-V, and Lane 17/J-IMP preflight stalls under
  `experiments/results/vast_live_harvest/`.
- Lane 19 snapshot archive was one-member `renderer.bin` only and is invalid
  for scoring until redesigned with full payload closure.
- MAE-V had two concurrent trainers writing the same output directory; custody
  is invalid. `AGENTS.md` now records the durable single-flight rule for
  Vast/output lanes.

Alpha empirical probe:

- `experiments/paradigm_alpha_real_archive_eval.py --candidates alpha4`
  completed as empirical/non-score evidence only.
- Baseline AV1 mask bytes: `421483`; Alpha4 grayscale-LUT bytes:
  `859664`; class agreement `0.9983855486`; rate delta
  `+0.291766741938`.
- Classification: negative for this Alpha4 config only. Not a mask-family
  kill; no broad conclusion without exact archive and adversarial review.

## 2026-05-01T10:45Z Lightning Alpha Reroute And OWV3 T4 Harvest

Backend-neutral Alpha-Geo-0 implementation:

- Added `experiments/alpha_geo0_pose_regen.py` as a reusable, non-Modal
  runner for the Lane 12 stale-pose isolation experiment. It preserves exact
  `masks.nrv` and `renderer.bin`, decodes candidate masks, regenerates
  `optimized_poses.bin`, rebuilds deterministic `archive.zip`, and runs
  canonical CUDA auth eval plus adjudication.
- Added `scripts/launch_lightning_alpha_geo0_pose_regen.py` to stage/queue the
  same experiment on Lightning Batch with source/artifact manifest custody,
  remote supply-chain preflight, hash-pinned DALI bootstrap, CUDA runner
  preflight, and state-derived harvest compatibility.
- Hardened `src/tac/deploy/lightning/batch_jobs.py` with
  `alpha_geo0_exact_eval` as a fail-closed role. This closes the bug class
  where an exact-score-producing custom Lightning job could be tracked as a
  generic non-fail-closed job.

Alpha-Geo-0 Lightning dispatch:

- Manifest staging completed:
  `.omx/state/alpha_geo0_pose_regen_lightning_t4_20260501T103952Z_manifest.json`,
  `1053` files, `113177404` bytes, SHA-256
  `ba3a27cb08fb58f6a3c9fe83ee298fe78fade6d31a1f4d43170abaf3726cc496`.
- Local and remote supply-chain scans passed with no `lightning`,
  `pytorch-lightning`, or Mini Shai-Hulud indicators.
- Initial Lightning submit with symbolic `T4` failed API validation
  (`accelerator T4 not found for this AWS cluster`). Resubmitted with concrete
  `g4dn.xlarge`, matching prior successful exact-eval jobs.
- Job queued:
  `alpha_geo0_pose_regen_lightning_t4_20260501T103952Z`, SDK job
  `alpha-geo0-pose-regen-lightning-t4-20260501t103952z`, status `Pending` at
  `2026-05-01T10:40:53Z`, machine reported by Lightning as `T4_SMALL`.
- Inputs locked in queue metadata: candidate archive SHA-256
  `864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97`,
  `296478` bytes; PFP16 baseline archive SHA-256
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  `686635` bytes; warm poses SHA-256
  `cb8517f7a7e3c9382e952ff278dc3f8de44ba066db07746f16354c1dbe2cbca4`;
  GT pose targets SHA-256
  `9b8d6c63e44be105f4d81c17b344b272ca4585337c767af3e0af0aaeffbb83c5`.

OWV3/Fisher T4 harvest:

- Lightning T4 job `exact_eval_owv3_fisher_beta_t4_20260501T1025Z`
  completed and was harvested through state-derived SSH artifact mirror.
- Exact T4 score evidence: score `1.0093151488173409`, archive `638165`
  bytes, SHA-256
  `57abe0fdf786d95b38325334b568e7a947143afe097ba189f214f2208492cb8f`,
  PoseNet `0.00331601`, SegNet `0.00402288`, `600` samples, Tesla T4.
- Adjudication: evidence grade `A++ contest T4`, but
  `promotion_eligible=false` because the SegNet relative component gate was
  exceeded by a tiny margin: observed/reference
  `0.00402288 / 0.00400656 = 1.0040733197556009` versus the configured
  `1.004` gate. Lane status is `COMPONENT_GATE_REVIEW_REQUIRED`.
- Scientific classification: exact lower-score signal, not a promotable
  frontier anchor until component-gate policy and/or a repaired archive pass
  adversarial review. The measured implementation is not killed.

## 2026-05-01T11:08Z Exact T4 Frontier Candidate From Direct-FD Response Points

New exact frontier packet:

- Harvested Lightning component-response jobs:
  `component_response_pfp16_direct_fd_mixed_response_only_t4_aws_20260501T0950Z`
  and duplicate
  `component_response_pfp16_direct_fd_mixed_response_only_t4_aws_small_20260501T0950Z`.
- Both jobs ran on Tesla T4 with CUDA, full 600-sample canonical
  `archive.zip -> inflate.sh -> upstream/evaluate.py` custody. The response
  packet itself remains `promotion_eligible=false` because it is
  response-only and has no prediction-delta semantics.
- Individual point archives inside the packet were then adjudicated directly
  from their exact T4 `contest_auth_eval.json` artifacts and original local
  archive bytes.
- Best point: `point_000_eps_m2`, archive copied to
  `experiments/results/frontier_candidate_direct_fd_m2_pfp16_20260501/archive/archive.zip`.
- Exact score: `1.035635453539817`, archive bytes `686632`, archive SHA-256
  `d561b8bf1367619e6c6e1d9b9d09213da6bdfdfd894be518866ef5119cba2927`,
  PoseNet `0.00311743`, SegNet `0.00401873`, `600` samples, Tesla T4.
- Adjudication: `promotion_eligible=true`, evidence grade `A++ contest T4`,
  lane status `IN_PREDICTED_BAND`. Component gates passed against same-run
  PFP16 T4 reference: PoseNet relative `0.8998418205644811`, SegNet relative
  `1.0030375184697096` with max SegNet relative gate `1.004`.
- Duplicate T4_SMALL job independently reproduced the same archive SHA with
  score `1.0356379048481845`, PoseNet `0.00311734`, SegNet `0.00401878`,
  and passing gates.
- Custody packet:
  `experiments/results/frontier_candidate_direct_fd_m2_pfp16_20260501/manifest.json`,
  `archive/archive.zip`, `eval/contest_auth_eval.json`,
  `eval/contest_auth_eval.adjudicated.json`, `eval/provenance.json`,
  `eval/report.txt`.

Scientific interpretation:

- This is a real A++ exact-score improvement of `-0.008352071254075` versus
  the PFP16 A++ anchor. It is not the seven-tenths strategic move, but it is a
  clean new frontier candidate and a high-value sensitivity calibration signal.
- The response curve showed all tested nonzero perturbation points improved
  combined scorer component, but `point_004_eps_p2` exceeded the strict SegNet
  relative gate and remains non-promotable forensic evidence.
- Protocol lesson: a non-promotable response-only packet can still contain
  separately rankable exact point archives, if those point archives have exact
  archive custody, canonical CUDA eval JSON, provenance, and independent
  adjudication.

Alpha reroute update:

- Initial Lightning Alpha-Geo-0 job
  `alpha_geo0_pose_regen_lightning_t4_20260501T103952Z` failed before lane
  evidence at `decode_candidate_masks`: remote `upstream/ffmpeg-new` could
  not load `libSvtAv1Enc.so.2`.
- Hardened `src/tac/mask_codec.py` so `TAC_FFMPEG=ffmpeg` path-name
  overrides are honored, upstream `ffmpeg-new` candidates must pass
  `ffmpeg -version`, and explicit bad overrides fail closed.
- Added `experiments/alpha_geo0_pose_regen.py` fallback env wiring to prefer
  system `ffmpeg` when available.
- Replacement Lightning job queued:
  `alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z`, status
  `Pending` at `2026-05-01T11:06:10Z`, zero cost so far, source manifest
  `.omx/state/alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z_manifest.json`.

Vast cleanup update:

- Live Vast instance `35955469` / `owv3_0134_eval_repro` was harvested and
  destroyed. Current live Vast count is `0`.
- Harvested RTX 4090 OWV3 evals under
  `experiments/results/vast_live_harvest/35955469_owv3_0134_eval_repro_20260501/`.
- Best harvested Vast score: `owv3_0001_r7`, score `1.0134396099014253`,
  bytes `631473`, SHA-256
  `5c11013539755c6470fb9f55e4d7f2ab6ec1edb2b951a468513d4ed7550f66ef`,
  A-grade CUDA evidence only, not A++ contest-equivalent T4 evidence.

## 2026-05-01T11:20Z Live Queue, Alpha Screen, And Wave3 State

Lightning queue:

- `alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z` is
  `Running` on T4_SMALL at `2026-05-01T11:18:13Z`, cost `0.026758334`.
  This is the repaired Alpha-Geo-0 stale-pose isolation run after the
  ffmpeg resolver fix.
- `exact_eval_owv3_5c110_r7_t4_20260501T1112Z` is `Running` on T4_SMALL at
  `2026-05-01T11:18:27Z`. It is the T4 reproduction/adjudication of the
  RTX4090 OWV3 archive with bytes `631473` and SHA-256
  `5c11013539755c6470fb9f55e4d7f2ab6ec1edb2b951a468513d4ed7550f66ef`.
- `exact_eval_direct_fd_m2_frontier_t4_20260501T1110Z` remains `Pending` at
  `2026-05-01T11:18:20Z`. It is the standalone T4 reproduction of the current
  direct-FD frontier archive SHA-256
  `d561b8bf1367619e6c6e1d9b9d09213da6bdfdfd894be518866ef5119cba2927`.

Alpha local empirical screen:

- Added and verified `experiments/alpha_frontier_candidate_screen.py` as a
  deterministic, non-promotable screen. It records `score_claim=false`,
  `promotion_eligible=false`, and `evidence_grade=empirical`; exact CUDA auth
  eval remains mandatory for any candidate.
- Hardened the screen's Alpha4 ffmpeg resolver to match `src/tac/mask_codec.py`:
  explicit `TAC_FFMPEG` overrides must be usable, broken upstream
  `ffmpeg-new` is skipped, and system `ffmpeg` is validated before use.
- Verification: `py_compile` passed for the screen and tests; pytest
  `src/tac/tests/test_alpha_frontier_candidate_screen.py -q` passed
  (`7 passed`).
- Capped empirical screen on the PFP16 A++ archive:
  `experiments/results/alpha_frontier_candidate_screen_pfp16_20260501/report_max16.json`.
  It used the first `16` of `1200` frames, archive SHA-256
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  archive bytes `686635`, and `masks.mkv` bytes `421483`.
- Non-score candidate bytes on the first 16 frames:
  AV1 baseline `421483` bytes, Alpha2 wavelet `325906` bytes at agreement
  `1.0`, Alpha3 VQ `54819` bytes at agreement `0.999315579732`, Alpha4
  grayscale-LUT AV1 `11989` bytes at agreement `0.998299280802`.
  Interpretation: Alpha3/Alpha4 are large-byte-move candidates, but current
  disagreement requires geometry/score-preserving repair before exact eval.

Vast:

- Reconciliation found new live instance `35956905`, label
  `owv3_wave3_chain_eval`, RTX 4090, `$0.26982407407407405/hr`, host
  `ssh3.vast.ai:36904`. Status was `loading` at first local inspection and
  `running` by `2026-05-01T11:20:27Z`.
- Local Wave3 selection source:
  `experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/wave3_chain_selection.json`.
  The six selected archives are `owv3_0043`, `owv3_0032`, `owv3_0076`,
  `owv3_0065`, `owv3_0120`, and `owv3_0119`, with byte counts
  `624419`, `624996`, `621914`, `622407`, `617410`, and `618443`.
- Instance inspection showed the repo and archives present but no eval process
  running. Patched the local and remote Wave3 summary-line formatting bug.
- First restart exposed missing `uv`; installed `uv 0.11.8` and restarted.
  Second restart exposed missing/old ffmpeg parity: Ubuntu ffmpeg `4.4.2`
  lacks the `scale` filter `in_primaries`/`in_transfer` options required by
  `submissions/robust_current/inflate.sh`.
- Downloaded BtbN master ffmpeg build
  `ffmpeg-master-latest-linux64-gpl.tar.xz`, SHA-256
  `e75caec4d65d9baa84c063e54746d2e08f1bdcd719b187967f34330f7c1486fb`,
  reporting `ffmpeg version N-124278-gcc3ca17127-20260430`, and verified the
  required `scale` options exist.
- Killed duplicate Wave3 drivers/partial evals, cleared partial result dirs,
  and restarted exactly one chain at `2026-05-01T11:27:18Z` with
  `FFMPEG_BIN=/workspace/ffmpeg-btbn/bin/ffmpeg`; remote PID `4494`.
- At `2026-05-01T11:27:37Z`, candidate `owv3_0043` had reached
  `contest_auth_eval.py`, with `uv run` inflating the archive. No result JSON
  had been harvested yet.
- Vast Wave3 evidence remains advisory until harvested from canonical JSON and
  locally adjudicated; do not promote or rank from the remote driver logs.

Alpha primitive diagnostics:

- Added `experiments/alpha_primitive_mask_diagnostics.py` and
  `src/tac/tests/test_alpha_primitive_mask_diagnostics.py`.
- The diagnostic safely reads `masks.mkv`, rejects zip-slip/hidden/duplicate
  archive hazards, decodes masks with the existing mask codec, and emits
  connected-component/bounding-box/centroid/area/boundary/temporal-change
  summaries. It is explicitly non-promotable:
  `score_claim=false`, `promotion_eligible=false`, `evidence_grade=empirical`.
- Hardened the CLI default to a bounded `64` frames; full-corpus primitive
  diagnostics require explicit `--all-frames`.
- Verification: `py_compile` passed; pytest
  `src/tac/tests/test_alpha_primitive_mask_diagnostics.py -q` passed
  (`6 passed`).
- Capped PFP16 diagnostic artifact:
  `experiments/results/alpha_primitive_mask_diagnostics_pfp16_20260501/report_max8.json`.
  First-8-frame summary showed class-1 fragmentation (`727` total components,
  max `116` in one frame), total temporal changed pixels `10598` across
  seven adjacent frame pairs, and mean temporal changed fraction
  `0.007700602214`. This is diagnostic geometry signal for Alpha repair, not
  score evidence.

Latest remote status at `2026-05-01T11:29Z`:

- Lightning Alpha-Geo-0 ffmpegfix
  `alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z` remains
  `Running`, cost `0.058425`.
- Lightning OWV3 5c110 T4 exact eval
  `exact_eval_owv3_5c110_r7_t4_20260501T1112Z` remains `Running`, cost
  `0.029608333`.
- Lightning direct-FD frontier standalone repro
  `exact_eval_direct_fd_m2_frontier_t4_20260501T1110Z` is now `Running`,
  cost `0.028605556`.
- Vast Wave3 is running exactly one active chain with candidate `owv3_0043`
  in `contest_auth_eval.py`/inflate under parity ffmpeg. No canonical JSON has
  been harvested from Wave3 yet.

## Codex continuation update — 2026-05-01T11:45Z

Lightning exact eval custody:

- Harvested and locally validated
  `experiments/results/lightning_batch/exact_eval_direct_fd_m2_frontier_t4_20260501T1110Z/`.
  Result: score `1.0356355862798443`, PoseNet `0.00311747`, SegNet
  `0.00401872`, archive bytes `686632`, archive SHA-256
  `d561b8bf1367619e6c6e1d9b9d09213da6bdfdfd894be518866ef5119cba2927`,
  `600` samples, CUDA, Tesla T4. Adjudication: `IN_PREDICTED_BAND`,
  component gates pass, `promotion_eligible=true`, evidence grade
  `A++ contest T4`.
- Harvested and locally validated
  `experiments/results/lightning_batch/exact_eval_owv3_5c110_r7_t4_20260501T1112Z/`.
  Result: score `1.0077865870356524`, PoseNet `0.00340903`, SegNet
  `0.00402679`, archive bytes `631473`, archive SHA-256
  `5c11013539755c6470fb9f55e4d7f2ab6ec1edb2b951a468513d4ed7550f66ef`,
  `600` samples, CUDA, Tesla T4. Adjudication: component gate review required;
  SegNet relative ratio `1.0050492192803802` exceeds strict cap `1.004`.
  Evidence is exact and valuable, but `promotion_eligible=false`; use only as
  scoped forensic/re-design input until repaired and re-evaluated.
- The Lightning SDK briefly regressed both exact-eval job statuses from
  Running to Pending, producing `REMOTE_STATUS_RECONCILIATION_REQUIRED`.
  Direct SSH artifact validation found canonical JSON/archive/adjudication
  complete, and later SDK refresh reconciled both jobs to Completed. This is a
  control-plane anomaly, not a scientific result.

Alpha mask candidate builder:

- Added and verified `experiments/alpha_mask_candidate_builder.py` and
  `src/tac/tests/test_alpha_mask_candidate_builder.py`. The tool is explicitly
  empirical/non-promotable by default, rejects unsafe archives and bad ffmpeg
  overrides, records SHA/bytes/provenance, and refuses partial repair unless
  explicitly allowed.
- Full PFP16 run:
  `experiments/results/alpha_mask_candidate_builder_pfp16_20260501_full/alpha_mask_candidate_manifest.json`.
  Source archive SHA-256
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  `masks.mkv` member bytes `421483`, decoded frames `1200`.
- First full run failed closed at `max_repair_runs=250000`; rerun with
  `max_repair_runs=1000000`, `max_repair_pixels=2000000` completed with
  `repair_full=true`.
- Candidate artifacts: `grayscale.mkv` `859664` bytes, SHA-256
  `d9c1a1e75897be9b662e6b84a2a2b447e56b92bd73e07a14b9ca566725700ba8`;
  `alpha4_residual_repair.amr1` `3657451` bytes, SHA-256
  `f795adfdb035c87e039b85637b31c51749297e93e41a04fface81bd527abd4ec`.
  Before repair: `380889` differing pixels, agreement `0.99838558197`.
  After repair: exact class-id agreement `1.0`.
- Interpretation: this is a useful exact-reconstruction diagnostic and runtime
  integration target, but not a byte frontier candidate as emitted. Do not
  spend T4 exact eval on the current full-repair artifact; next Alpha work
  should target residual compression/selection or a different mask
  representation that can beat the original `421483` mask bytes.

Vast Wave3:

- New live instance `35957332`, label `owv3_wave3_chain_eval_v2`, RTX 4090,
  host `ssh9.vast.ai:37332`, driver `550.120`.
- The first attempt revealed the archive-only eval path still had a floating
  `uv run --with torch` dependency. On driver `550.120`, uv resolved a CUDA 13
  Torch and inflate fell back to CPU rendering before the CUDA scorer. The
  partial run was stopped before producing a JSON to avoid ambiguous advisory
  evidence.
- Hardened `submissions/robust_current/inflate.sh` to use deterministic
  dependency specs (`INFLATE_BROTLI_SPEC`, `INFLATE_AV_SPEC`,
  `INFLATE_TORCH_SPEC`, `INFLATE_NUMPY_SPEC`) and hardened
  `scripts/remote_archive_only_eval.sh` to choose driver-compatible
  `INFLATE_TORCH_SPEC=torch==2.5.1` on older Vast drivers while preserving
  `torch==2.11.0` as the default matching current Lightning exact-eval
  lock-era behavior.
- Re-staged the patched `inflate.sh`, `remote_archive_only_eval.sh`, and
  Wave3 driver to Vast, restarted Wave3 at `2026-05-01T11:44:46Z`, and
  confirmed the first candidate starts with `INFLATE_TORCH_SPEC=torch==2.5.1`,
  CUDA available, NVDEC OK, and BtbN ffmpeg parity. No Wave3 JSON has been
  harvested yet.

Verification in this continuation:

- `src/tac/tests/test_alpha_mask_candidate_builder.py`
  + `src/tac/tests/test_remote_archive_only_eval_script.py`: `11 passed`.
- `src/tac/tests/test_inflate_sh_brotli_clean_env.py`
  + `src/tac/tests/test_remote_archive_only_eval_script.py`: `13 passed`.
- `bash -n` passed for `submissions/robust_current/inflate.sh`,
  `scripts/remote_archive_only_eval.sh`, and the Wave3 driver.

## Codex continuation update — 2026-05-01T13:09Z

Coordination and live queue:

- Partner Claude has four orthogonal lanes in flight on Vast instance
  `35959478`: an owv3_0120 orthogonal stack composer, NeRV/HNeRV mask codec
  training, Joint-ADMM cross-stream coordination, and PoseNet-aware sparse
  overfit encoding. Treat those as partner-owned write/eval scopes until their
  artifacts are handed off for independent custody review or Lightning T4
  promotion. Codex remains owner for Lightning T4 promotion, Alpha primitive
  response, component-response optimizer, custom decoder planning, and durable
  DX fixes.
- Lightning T4 status at this checkpoint: Alpha CRF63 grayscale exact eval is
  `Running`; OWV3 0119 exact eval is `Running`; OWV3 0120 exact eval is
  `Pending`; Alpha primitive component-response
  `component_response_alpha_primitive_pfp16_crf63_t4_20260501T130822Z` is
  `Pending`.
- 0065/0032 Wave3 jobs were not durable local queue records despite earlier
  terminal output. Do not rely on them as queued until explicitly resubmitted
  and present in `.omx/state/lightning_batch_jobs.json`.

Local implementation and planning evidence:

- Landed durable Lightning SDK job-name normalization fix:
  `lightning_sdk_job_name()` now lowercases after underscore normalization.
  This matches observed SDK artifact paths such as
  `/teamspace/jobs/exact-eval-...t130313z/artifacts` and prevents manual
  path drift during refresh/harvest.
- Alpha primitive response plan rerun completed after the first attempt
  correctly rejected a nonportable `..` baseline JSON path. Output:
  `experiments/results/alpha_mask_primitive_response_plan_pfp16_20260501_r1/`
  with eight nonzero perturbation archives and a non-promotable CUDA
  component-response plan. The plan was staged to Lightning and queued as
  `component_response_alpha_primitive_pfp16_crf63_t4_20260501T130822Z`.
- Full PFP16 Alpha mask codec candidate matrix completed:
  `experiments/results/alpha_mask_codec_candidate_matrix_pfp16_20260501_full/`.
  Exact pure-symbol candidates were larger than the current `masks.mkv`
  (`421483` raw bytes, `412169` compressed bytes): COCO-style foreground RLE
  `902161` bytes, transition endpoints `997410` bytes, component boundary
  delta `1476543` bytes. Interpretation: exact naive RLE/transition/component
  packets are not the breakthrough path. Continue with lossy geometry plus
  charged sparse repair, neural INR/NeRV, or entropy-coded temporal grammar
  gated by CUDA component response.
- Custom decoder scaffold landed:
  `experiments/custom_mask_codec_probe.py`,
  `src/tac/tests/test_custom_mask_codec_probe.py`, and
  `docs/runbooks/custom_decoder_overfit_codec_plan_20260501.md`. The current
  `CMCP_RLE1` probe is empirical only and intentionally non-promotable.
- Component-response stack optimizer landed:
  `experiments/optimize_component_response_stack.py` and
  `src/tac/tests/test_optimize_component_response_stack.py`. It rejects
  non-CUDA/noncanonical/prediction-only inputs for promotable recommendation
  mode and records Dykstra-style constraints without claiming stack
  composability.
- Alpha lossy-repair budget planning is delegated to Codex worker Turing,
  focused on turning matrix/primitive evidence into repair-byte/spec
  candidates without scorer calls or promotion claims.

Verification at this checkpoint:

- `py_compile` passed for
  `experiments/alpha_mask_codec_candidate_matrix.py`,
  `experiments/custom_mask_codec_probe.py`,
  `experiments/optimize_component_response_stack.py`, and
  `src/tac/deploy/lightning/batch_jobs.py`.
- Focused pytest passed: Alpha matrix, custom mask codec probe, stack
  optimizer, and the Lightning SDK normalization test: `15 passed`.
- MCP cleanup remained clean: no orphaned MCP helper processes matched.
- MCP cleanup remained clean.

## Codex continuation update - 2026-05-01T13:23Z

Sequential mode is active per operator instruction: no new subagents; cloud
jobs may still run in parallel when they are already queued or are high-EV
custody duplicates.

Live evidence updates:

- Harvested `exact_eval_alpha_crf63_grayscale_t4_20260501T125258Z` with exact
  T4 CUDA custody. Archive SHA-256
  `76bc850551269cad8bc32315959521cbd6f02c2e29f6c16e38f4c68ecd3f0eea`, bytes
  `458341`, score `4.926778674301541`, PoseNet `1.34352684`, SegNet
  `0.00956173`. This is a scoped forensic negative for plain grayscale CRF63
  mask replacement: the byte cut is real, but PoseNet/SegNet geometry collapse
  blocks promotion. This is not an Alpha-family kill.
- Harvested `exact_eval_direct_fd_m2_frontier_t4_20260501T1110Z`: exact T4 CUDA
  score `1.0356355862798443`, bytes `686632`, PoseNet `0.00311747`, SegNet
  `0.00401872`, promotion-eligible under its recorded component gates.
- Submitted one fast duplicate for OWV3 0120 on L40S:
  `exact_eval_owv3_0120_wave3_l40s_20260501T1322Z`. It uses archive SHA-256
  `06af57f770342cde494c37839200fdda79bdadd29826009e5e107ab296b4057a`, bytes
  `617410`, and the same adjudication references as the T4 run. Treat it as
  fast CUDA custody signal; T4 remains the promotion-grade confirmation run.
- At last refresh, OWV3 0119 T4, OWV3 0120 T4, Alpha primitive response T4, and
  the L40S duplicate jobs were still active or pending.
- Subsequent harvest of `exact_eval_owv3_0119_wave3_t4_20260501T125143Z`
  produced exact T4 CUDA score `1.0025871157494655`, bytes `618443`, PoseNet
  `0.00357964`, SegNet `0.00401592`, archive SHA-256
  `75fc6c5eee02845f09296cda4854158d6663bb7533c2bf5f3c7a4a5b0638e802`.
  The total score is strong, but strict adjudication marks it
  `COMPONENT_GATE_REVIEW_REQUIRED` because PoseNet is `1.050046494164029`
  relative to the OWV3 R7 T4 reference under a `1.002` gate. Treat as exact
  scoped forensic evidence and a pose-protection target, not promotion.
- `exact_eval_owv3_0120_wave3_t4_20260501T130313Z` showed a Lightning status
  regression from Running back to Pending while total cost increased. This is a
  remote status reconciliation issue, not score evidence. A fast RTX PRO
  duplicate `exact_eval_owv3_0120_wave3_rtxpro_20260501T1326Z` was submitted
  after the L40S duplicate remained pending.

Orthogonal/frozen-stream optimization rule:

- Treat mask/video geometry, renderer weights, and pose stream as separate
  constrained streams. Optimize one stream while freezing the others as scorer
  anchors, then build a new archive and run exact CUDA eval before composing
  deltas.
- Alpha mask compression cannot be judged by bytes alone. The CRF63 result
  demonstrates that large mask-byte cuts can destroy PoseNet. Alpha successors
  must include one of: PoseNet-aware sparse repair, component-response-selected
  protected atoms, NeRV/INR geometry preservation, or pose regeneration, with
  all repair bits charged in the archive.
- OWV3/direct-FD renderer optimization should keep the mask stream frozen until
  an exact stacked archive proves the interaction. Direct-FD is currently a
  component-protection anchor; Alpha mask work should use it as a reference,
  not assume additive composition.
- Stacks are their own archives. No additive score claim is allowed from
  separate component deltas until exact CUDA auth eval on the stacked archive.

Planning artifacts advanced:

- Turing's `experiments/alpha_lossy_repair_budget_planner.py` landed with
  focused tests. A real planning run wrote
  `experiments/results/alpha_lossy_repair_budget_planner_20260501_r1/` with
  `91` empirical budget records and `12` candidate archive-build specs. These
  specs are non-promotable and wait on official CUDA component-response before
  any archive build.

## Codex continuation update - 2026-05-01T13:40Z

Operational changes:

- Per operator request, spawned exactly one read-only xhigh reverse-engineering
  research agent (`Schrodinger`) to audit top-leaderboard strategy signal,
  archive anatomy, likely stream allocation, and highest-EV implications for
  the next 48 hours. No file edits assigned.
- Stopped redundant OWV3 0120 exact-eval duplicates after the T4 result was
  harvested: `exact_eval_owv3_0120_wave3_l40s_20260501T1322Z` and
  `exact_eval_owv3_0120_wave3_rtxpro_20260501T1326Z` both now report
  `Stopped`.
- Stopped the slower Alpha component-response T4 backup after the L40S
  component-response evidence was harvested and locally validated.

New exact evidence:

- Harvested `exact_eval_owv3_0120_wave3_t4_20260501T130313Z` as exact T4 CUDA
  promotion-eligible evidence. Archive SHA-256
  `06af57f770342cde494c37839200fdda79bdadd29826009e5e107ab296b4057a`, bytes
  `617410`, score `1.0021175309471926`, PoseNet `0.00356094`, SegNet
  `0.00402305`, adjudication `IN_PREDICTED_BAND`, promotion eligible. This is
  the current exact T4 promotion-grade frontier in this ledger.
- Harvested
  `component_response_alpha_primitive_pfp16_crf63_l40s_20260501T131540Z`.
  Local validation copied official response curves, per-point
  `contest_auth_eval.json` files, supply-chain scans, preflight, and artifact
  validation. Baseline archive SHA-256
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`, bytes
  `686635`; L40S CUDA preflight was true on NVIDIA L40S, driver `580.126.09`,
  torch `2.11.0+cu130`.

Alpha response interpretation:

- The CRF63 primitive response is a scoped forensic negative for the measured
  plan shape, not an Alpha-family kill. Every nonzero point collapsed to
  combined component response around `12.33` to `12.35`, with PoseNet around
  `8.35` to `8.39` and SegNet around `0.0319`.
- The response is confounded by the lossy CRF63 base: nonzero points use
  ~`108.7KB` `masks.mkv` variants and are compared against the PFP16 baseline,
  with `same_run_zero_baseline=false`. Therefore it must not be used as a
  marginal sparse-repair selector. The correct next question is the lossiness
  threshold and geometry-preserving representation, not which of the eight
  CRF63 primitive atoms is best.

New threshold probes:

- Built deterministic grayscale replacement archives:
  - CRF60: `experiments/results/alpha_mask_replacement_pfp16_crf60_grayscale_only/archive.zip`,
    bytes `645623`, SHA-256
    `f83e17f136afd651e866ca4b564ad28681c3db36f15fd911c0a07e45fe9ac8ae`,
    argmax agreement `0.995597229004`.
  - CRF62: `experiments/results/alpha_mask_replacement_pfp16_crf62_grayscale_only/archive.zip`,
    bytes `531222`, SHA-256
    `90423f438273f4d5cae324023bb14fdc568f91950a6ddedf371067b8548b1dd8`,
    argmax agreement `0.994034733242`.
- Staged both archives and their manifests through
  `scripts/lightning_repro_workspace.py` under
  `.omx/state/alpha_grayscale_crf60_62_exact_sweep_20260501T1336Z_manifest.json`;
  remote manifest verification passed with `1167` files and `20799478` bytes.
- Submitted L40S exact CUDA threshold probes:
  - `exact_eval_alpha_crf60_grayscale_l40s_20260501T1339Z`
  - `exact_eval_alpha_crf62_grayscale_l40s_20260501T1339Z`
  These are fast CUDA screening jobs. If either preserves components near the
  PFP16 gate, rerun the exact same archive bytes on T4/equivalent before any
  promotion claim.

Verification:

- Alpha focused compile/test surface passed:
  `68 passed in 1.51s` across the Alpha builder, planner, replacement archive,
  Geo0, INR readiness, and related tests.
- MCP cleanup remained clean before Lightning status work.

## Codex grayscale reverse-engineering update - 2026-05-01T14:40Z

Scope:

- Operator requested reverse engineering of how the #2-style grayscale path can
  work without the PoseNet explosion observed in our CRF-only grayscale
  replacements.
- This update records source-level external/reference findings, exact local
  CUDA evidence, code parity fixes, and the next experiment order. It is not a
  score claim.

Exact evidence motivating the pivot:

- CRF60 grayscale replacement exact L40S CUDA score recomputed to
  `3.3965224624990418`, bytes `645623`, archive SHA-256
  `f83e17f136afd651e866ca4b564ad28681c3db36f15fd911c0a07e45fe9ac8ae`,
  PoseNet `0.57232893`, SegNet `0.00574289`. This is non-promotable
  component-collapse evidence.
- CRF62 grayscale replacement exact L40S CUDA score recomputed to
  `4.358051471129711`, bytes `531222`, archive SHA-256
  `90423f438273f4d5cae324023bb14fdc568f91950a6ddedf371067b8548b1dd8`,
  PoseNet `1.06793761`, SegNet `0.00736401`. This is non-promotable
  component-collapse evidence.
- Interpretation: these are scoped negatives for plain CRF grayscale
  substitution into the existing renderer path. They do not kill grayscale,
  learned mask, Selfcomp, Quantizr, or Alpha learned-topology families.

Public/reference reverse-engineering:

- Selfcomp reference inspected from public clone
  `szabolcs-cs/comma_video_compression_challenge` at commit
  `2b2d76de6f5aa34c76352c2cc02b03ed44a03a26`,
  `submissions/selfcomp/inflate.py`.
- Key Selfcomp mechanism:
  `CLASS_TARGETS = [0, 255, 64, 192, 128]`, `LUT_SIGMA = 15.0`, and the LUT
  computes `softmax(exp(-(gray - target)^2 / (2 sigma^2)))` across classes.
  The renderer receives the resulting soft probability map directly. It does
  not first nearest-neighbour decode to hard one-hot classes.
- Selfcomp also expands one decoded grayscale frame into a two-frame pair by
  feeding the same probability map twice with distinct frame indices. The
  trained renderer absorbs the frame-pair duality; this is the likely reason a
  small grayscale stream can preserve scorer geometry instead of acting like a
  destructive post-hoc mask transcode.
- Quantizr reference inspected from public clone
  `Quantizr/comma_video_compression_challenge` branch `quantizr` at commit
  `e0b643b0a7c21f62cc93b5d920bcf3fc0d5a33d9`.
- Key Quantizr mechanism: odd-frame-only hard class masks are scaled by `63`,
  encoded as raw-gray AV1 OBU, brotli-compressed, and decoded by rounding
  `gray / 63`. The pose-conditioned joint frame generator emits both pair
  frames from one charged mask plus pose vector. It is trained through the
  decoded representation and scorer losses, not retrofitted after training.

Root-cause conclusion:

- The missing trick in our failing grayscale probes was train/inflate parity.
  Our code had a sharper log-distance LUT (`softmax(-d^2 / 2 sigma^2)`) and
  runtime SegMap inflates that nearest-neighbour decoded grayscale to one-hot.
  That is not the Selfcomp #2 analog path and explains why byte savings came
  with PoseNet collapse.
- Correct testable hypothesis: train the renderer and any analog canvas
  optimizer against the exact soft bell-LUT distribution that inflate will
  feed, then exact-eval the archive. Plain CRF grayscale substitution should
  stop receiving wall-clock budget except as forensic threshold evidence.

Code changes landed:

- `src/tac/mask_grayscale_lut.py`: canonical LUT now matches the public
  Selfcomp bell-softmax formula and exposes
  `grayscale_to_probability_map(...)` as the train/inflate parity helper.
- `experiments/train_segmap.py` and
  `experiments/train_segmap_film_canvas.py`: pair tensors now use
  `encode_masks_grayscale(...) -> grayscale_to_probability_map(...)` instead
  of hard one-hot masks.
- `submissions/robust_current/inflate_segmap.py` and
  `submissions/robust_current/inflate_segmap_film_canvas.py`: default
  `SEGMAP_GRAYSCALE_MODE=soft_lut` feeds the soft LUT map to the renderer;
  `hard_onehot` remains as a forensic compatibility mode only. Film-canvas
  inflate also now uses bicubic upscaling for trainer parity.
- `scripts/remote_lane_sa_segmap_clone.sh` and
  `scripts/remote_lane_fc_film_canvas.sh`: remote provenance and generated
  `config.env` now explicitly record `SEGMAP_GRAYSCALE_MODE=soft_lut` so the
  corrected path is reproducible instead of relying on an implicit default.
- `src/tac/segmap_renderer.py` and
  `src/tac/optimize_grayscale_canvas.py`: differentiable LCT/analog-canvas LUT
  math now uses the same bell-softmax formula.
- Tests added/updated to enforce formula equality, helper tensor shapes,
  non-one-hot exact target rows, and SegMap/FilmCanvas pair-tensor parity.

Verification:

- `scripts/kill_orphaned_mcp_processes.py --strict`: clean.
- `py_compile` passed for all touched Python modules.
- `bash -n` passed for `scripts/remote_lane_sa_segmap_clone.sh`,
  `scripts/remote_lane_fc_film_canvas.sh`, and
  `scripts/remote_lane_q_faithful_jointgen.sh`.
- Focused tests passed:
  `26 passed` for grayscale LUT plus analog canvas tests.
- Broader focused SegMap/KL/LCT tests passed:
  `29 passed, 1 skipped`.
- `git diff --check` passed.

Next experiment order:

1. Relaunch or resume SA/FilmCanvas/Q-FAITHFUL-class learned representation
   training on fast CUDA with corrected soft-LUT parity.
2. Build deterministic archives that charge all decoder weights, grayscale
   streams, poses, and config.
3. Run fast-chip exact CUDA screening, then exact same archive bytes on
   T4/equivalent for any frontier/promotion candidate.
4. Keep CRF60/62/63 as scoped forensic negatives for plain post-hoc grayscale
   transcode; do not use them to retire the learned grayscale family.

## Codex exact-eval promotion and compute hygiene update - 2026-05-01T15:10Z

Scope:

- Operator requested continued sequential execution with no signal loss.
- This update records the custody chain for the new `owv3_0120_stack` frontier
  candidate, the Lightning T4 promotion job, and compute cleanup. It is not yet
  an A++ promotion claim; the T4 job is still pending at the time of writing.

Current exact evidence:

- Vast RTX 4090 instance `35959478` produced and was harvested locally at
  `experiments/results/vast_harvest/owv3_0120_stack_rtx4090_20260501T1501Z/`.
- Candidate archive:
  `experiments/results/vast_harvest/owv3_0120_stack_rtx4090_20260501T1501Z/owv3_0120_stack_archive.zip`.
- Archive bytes: `609963`.
- Archive SHA-256:
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`.
- RTX 4090 CUDA auth eval recomputed score:
  `0.997430122363832`, PoseNet `0.00356167`, SegNet `0.00402557`, full
  `600` samples.
- Evidence grade before T4 confirmation: `A` score-grade, because the source
  eval is exact CUDA but not T4/equivalent contest-promotion hardware.

T4 promotion job:

- Queued via `scripts/lightning_exact_eval_repro.py` with local and remote
  supply-chain scans clean, remote source manifest verification clean, and
  exact archive SHA/byte preflight embedded in the job spec.
- Job name: `exact_eval_owv3_0120_stack_t4_20260501T150652Z`.
- Lightning SDK job name:
  `exact-eval-owv3-0120-stack-t4-20260501t150652z`.
- Lightning artifact path:
  `/teamspace/jobs/exact-eval-owv3-0120-stack-t4-20260501t150652z/artifacts`.
- Local planned artifact directory:
  `experiments/results/lightning_batch/exact_eval_owv3_0120_stack_t4_20260501T150652Z/`.
- State files:
  `.omx/state/exact_eval_owv3_0120_stack_t4_20260501T150652Z_lightning_exact_eval_repro_plan.json`,
  `.omx/state/exact_eval_owv3_0120_stack_t4_20260501T150652Z_lightning_batch_record.json`,
  `.omx/state/exact_eval_owv3_0120_stack_t4_20260501T150652Z_manifest.json`,
  `.omx/state/exact_eval_owv3_0120_stack_t4_20260501T150652Z_local_lightning_supply_chain_scan.json`.
- Baseline/reference for adjudication:
  `experiments/results/lightning_batch/exact_eval_owv3_0120_wave3_t4_20260501T130313Z/contest_auth_eval.json`,
  recomputed score `1.0021175309471926`, bytes `617410`, PoseNet
  `0.00356094`, SegNet `0.00402305`.
- Initial Lightning refresh status: `Pending` on machine class `T4_SMALL`.
- Wall-clock hedge job queued after repeated pending refreshes:
  `exact_eval_owv3_0120_stack_t4aws_20260501T151050Z`, SDK job
  `exact-eval-owv3-0120-stack-t4aws-20260501t151050z`, machine class `T4`
  via `g4dn.2xlarge`, same archive bytes/SHA, same baseline/adjudication
  gates, and `duplicate_of=exact_eval_owv3_0120_stack_t4_20260501T150652Z`
  in queue metadata. This is a redundant promotion attempt to reduce
  wall-clock latency, not a separate lane or independent score claim.

Compute hygiene:

- MCP cleanup remained clean before this work.
- Vast instance `35961748` (`arith_coding_h100`, H100 SXM) was confirmed idle:
  GPU utilization `0.0`, no `/workspace/pact`, no tmux session, no active
  payload, and effectively empty disk. It was destroyed to stop nonproductive
  spend; audit record:
  `.omx/state/vast_destroy_35961748_20260501T1507Z.json`.
- Vast RTX 4090 instance `35959478` remains available until the T4 promotion
  packet is harvested and reconciled.

Next required actions:

1. Refresh Lightning status for
   `exact_eval_owv3_0120_stack_t4_20260501T150652Z` until completion.
2. Harvest with the state-derived wrapper, validate archive/adjudication JSON,
   and compare T4 recomputed score against the RTX 4090 source result and the
   current T4 frontier.
3. If T4 confirms, promote the candidate as the current exact frontier and
   update the claim matrix with evidence grade and custody links.
4. If T4 does not confirm, preserve all artifacts and classify the mismatch
   before drawing any lane conclusion.

## Codex Alpha custody and sparse-repair planning update - 2026-05-01T15:18Z

Lightning T4 promotion status:

- `exact_eval_owv3_0120_stack_t4_20260501T150652Z`: `Running` on `T4_SMALL`
  as of `2026-05-01T15:15:54Z`, cost `0.010555555`.
- `exact_eval_owv3_0120_stack_t4aws_20260501T151050Z`: `Running` on `T4`
  as of `2026-05-01T15:15:54Z`, cost `0.0`.
- Both jobs evaluate the identical archive bytes/SHA. The second job is a
  wall-clock hedge only, not independent lane evidence.

Alpha Geo0 failure custody:

- Mirrored small canonical failure artifacts from the failed Lightning job
  `alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z` into:
  `experiments/results/lightning_batch/alpha_geo0_pose_regen_lightning_t4_ffmpegfix_20260501T110126Z/`.
- Manual SSH mirror was used because the state-derived harvest wrapper expected
  `lightning_queue_metadata.json`, which this custom Alpha diagnostic job did
  not emit. The mirror copied only small JSON/log artifacts, not the full
  `mask_cache`.
- T4 preflight was clean: `cuda_available=true`, device `Tesla T4`,
  `gpu_t4_match=true`.
- NVDEC probe passed.
- The run failed at `decode_candidate_masks` with
  `RuntimeError: diagnose_nerv_geometry failed`.
- Geometry evidence: global mask disagreement `0.012303911844889322`
  (`2,902,857 / 235,929,600` pixels). The job records
  `score_claim=false`, `promotion_eligible=false`, and `passed=false`.
- Scientific classification: scoped measured-implementation geometry failure
  for this Alpha Geo0/jsonfix40 candidate. This is not a family kill and not a
  score claim. It is redesign input for NeRV/HNeRV geometry preservation,
  sparse repair, and component-response atom selection.

Alpha lossy sparse-repair planner:

- Materialized the planner output under
  `experiments/results/alpha_lossy_repair_budget_planner_20260501/`.
- Command:
  `.venv/bin/python experiments/alpha_lossy_repair_budget_planner.py --output-dir experiments/results/alpha_lossy_repair_budget_planner_20260501 --lossy-base-bytes 60000,80000,100000,150000,250000 --max-specs 24 --force`.
- Output report:
  `experiments/results/alpha_lossy_repair_budget_planner_20260501/alpha_lossy_repair_budget_plan.json`.
- Output candidate specs:
  `experiments/results/alpha_lossy_repair_budget_planner_20260501/candidate_archive_specs/`.
- Scope/evidence: `empirical`, `score_claim=false`,
  `promotion_eligible=false`, `archives_built=false`,
  `requires_exact_cuda_auth_eval=true`.
- Planner summary: `104` budget records and `24` candidate archive specs. The
  no-repair hypotheses around `60KB` and `80KB` mask payloads show the expected
  large byte headroom against the current `421,483` byte mask stream, but remain
  planning estimates until real lossy-base artifacts plus official CUDA
  component-response determine which repair atoms are allowed.

Verification:

- `.venv/bin/python -m py_compile experiments/alpha_lossy_repair_budget_planner.py src/tac/tests/test_alpha_lossy_repair_budget_planner.py`
- `.venv/bin/python -m pytest src/tac/tests/test_alpha_lossy_repair_budget_planner.py -q`
- Result: `5 passed in 0.08s`.

Next required actions:

1. Continue monitoring both T4 exact eval jobs; harvest the first completed
   packet with the state-derived wrapper and stop the duplicate only after one
   valid adjudicated result is local.
2. Preserve Alpha Geo0 as a scoped diagnostic failure and use it to constrain
   NeRV/HNeRV retraining or sparse repair, not to kill the family.
3. Route the `alpha_lossy_repair_budget_planner_20260501` candidate specs into
   the official CUDA component-response queue once the component-response path
   has validated archive custody and sample coverage.

## Codex rapid 2026 research intake while T4 jobs run - 2026-05-01T15:22Z

Scope:

- This is external motivation only, not contest evidence.
- Intake target was current compression/task-aware/NeRV/RL material relevant to
  fastest path below `0.3`.

External references checked:

- WACV 2026, `How to Design and Train Your Implicit Neural Representation for
  Video Compression`: confirms NeRV-family design remains active and that
  implementation choices/training time materially affect the rate-quality
  point. The paper's RNeRV/hyper-network discussion is useful for Alpha mask
  codec design, but our deadline favors overfit-per-video training and sparse
  repair rather than a general encoder.
- arXiv `2601.19293`, `Reinforced Rate Control for Neural Video Compression via
  Inter-Frame Rate-Distortion Awareness`: supports rate-control/Bayesian or RL
  allocation framing. For this contest, use the principle as a low-step-count
  bandit/CMA-ES allocator over codec knobs and component-response data, not a
  full PPO system before per-step cost drops.
- arXiv `2512.12936`, `Content Adaptive based Motion Alignment Framework for
  Learned Video Compression`: reinforces content-specific motion/quality
  adaptation. Operational read: Alpha mask/video replacements should condition
  on motion magnitude and hard zones, and component-response should weight
  high-motion/hard-pair strata explicitly.
- arXiv `2601.22189`, `SCENE`: semantic-aware codec preprocessing is relevant
  as a pattern for task-aware pre/post-processing while keeping a standard
  codec path. Operational read: use scorer/component semantics as the
  saliency/controller signal, not human-perceptual saliency.
- `cshw2021/Learned-Image-Video-Compression` and
  `Xinjie-Q/Awesome-Neural-Compression`: current curated lists include DCC
  2026/WACV 2026/AAAI 2026 low-bitrate, task-aware, INR, rate-control, and
  neural-video-codec references. Use them as intake indexes for implementation
  patterns only.

Council operational synthesis:

- These references do not change the immediate ordering: confirm the
  `owv3_0120_stack` T4 packet, then use official component-response plus Alpha
  sparse-repair specs to decide the next exact archive. The external research
  reinforces the same high-EV path: content/task-aware allocation over
  importing a full general-purpose neural codec before the deadline.
- For sub-`0.3`, the only currently credible large move remains the Alpha
  mask-payload collapse plus score-aware protection/repair. RL/rate-control
  becomes useful as a cheap search controller once the eval step is reduced to
  component-response or proxy-screen cost; it should not displace Alpha
  geometry and exact CUDA auth eval.

## Codex component-response optimizer replay - 2026-05-01T15:24Z

Purpose:

- Use measured CUDA component-response packets to rank next exact-eval
  candidates without making composability claims.

Commands:

- `.venv/bin/python experiments/optimize_component_response_stack.py experiments/results/lightning_batch/component_response_pfp16_direct_fd_mixed_response_only_t4_aws_20260501T0950Z/official_component_response_summary.json --allow-calibration-inputs --top-k 20 --output-json experiments/results/component_response_stack_optimizer_20260501/optimizer_directfd_t4.json`
- `.venv/bin/python experiments/optimize_component_response_stack.py experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_r7_predicted_from_r2_l40s_codex_lightning_t4_parallel/official_component_response_summary.json --allow-calibration-inputs --top-k 20 --output-json experiments/results/component_response_stack_optimizer_20260501/optimizer_r7_t4_parallel.json`

Results:

- Both runs completed and emitted non-promotable planner JSON.
- `optimizer_directfd_t4.json`: baseline score `1.043987587070934`, `4`
  actions considered, `0` feasible under the default Dykstra gates, `4`
  infeasible. Best infeasible exact point remains `eps_m2`, projected score
  `1.0356355182399997`, score delta `-0.008352068830934423`, but it violates
  the strict SegNet gate by `1.2169999999999716e-05` absolute.
- `optimizer_r7_t4_parallel.json`: baseline score `1.037044888879275`, `4`
  actions considered, `0` feasible under default gates, `4` infeasible.
- Attempted two-source composition between the direct-FD and r7 packets failed
  closed because their baselines differ. This is correct: additive stack
  planning is invalid across different baseline states unless rebased.

Interpretation:

- No new exact-eval candidate should be dispatched from these optimizer
  outputs without either relaxed/adversarially reviewed component gates or a
  rebase onto the active `owv3_0120_stack` archive.
- The optimizer confirms the current near-term frontier path is the pending
  `owv3_0120_stack` T4 confirmation, not another PFP16 direct-FD microstep.

## Codex OWV3 0120 stack T4 promotion landed - 2026-05-01T15:27Z

Primary T4 packet:

- Job: `exact_eval_owv3_0120_stack_t4_20260501T150652Z`.
- Local artifact directory:
  `experiments/results/lightning_batch/exact_eval_owv3_0120_stack_t4_20260501T150652Z/`.
- Archive bytes/SHA: `609963`,
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`.
- Exact CUDA/T4 recomputed score: `0.9975405870574277`.
- Components: PoseNet `0.00357302`, SegNet `0.00402367`, `600` samples.
- Hardware: `Tesla T4`, `gpu_t4_match=true`.
- Adjudication: evidence grade `A++ contest T4`,
  `promotion_eligible=true`, `lane_status=IN_PREDICTED_BAND`,
  `component_gate_triggered=false`.
- Delta versus `owv3_0120_wave3_t4_frontier`: `-0.004576943889764928`.

Wall-clock hedge reproduction:

- Job: `exact_eval_owv3_0120_stack_t4aws_20260501T151050Z`.
- Local artifact directory:
  `experiments/results/lightning_batch/exact_eval_owv3_0120_stack_t4aws_20260501T151050Z/`.
- Same archive bytes/SHA: `609963`,
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`.
- Exact CUDA/T4 recomputed score: `0.9975385870574276`.
- Components: PoseNet `0.00357302`, SegNet `0.00402365`, `600` samples.
- Hardware: `Tesla T4`, `gpu_t4_match=true`.
- Adjudication: evidence grade `A++ contest T4`,
  `promotion_eligible=true`, `lane_status=IN_PREDICTED_BAND`,
  `component_gate_triggered=false`.
- Delta versus `owv3_0120_wave3_t4_frontier`: `-0.004578943889764986`.

Queue cleanup:

- The hedge job reached `Completed` and was harvested.
- The primary SDK status regressed to `Pending` after artifacts were already
  present and harvested. It was stopped through the Lightning SDK to prevent
  further spend. State refresh now records `Stopped` at cost `0.04222222`.
- Hedge final cost: `0.17503889`.

Claim matrix:

- Added C-044 to
  `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`.
- C-044 is the active A++ frontier row as of this update. It supersedes the
  direct-FD PFP16 row C-043 for current-rank purposes while preserving C-043 as
  sensitivity calibration history.

Scientific interpretation:

- This is a clean orthogonal-stack byte win: OWV3 0120 renderer/mask stack plus
  PFP16 pose representation, same exact archive bytes confirmed twice on T4.
- It is still a microstep, not the Shannon-floor move. The next high-EV work
  remains Alpha mask-payload collapse plus scorer-aware sparse repair and
  component-response gating against this new active frontier.

## Codex Alpha sparse-repair runtime and C-044 exact-eval queue - 2026-05-01T15:41Z

Purpose:

- Turn the existing Alpha4 grayscale/residual planning artifacts into actual
  contest-evaluable archives against the active C-044 frontier, without
  trusting proxy scores.

Implementation landed:

- `submissions/robust_current/inflate_renderer_grayscale.py` now supports an
  optional AMR1 residual repair payload after decoding `grayscale.mkv` and
  before re-encoding the legacy `masks.mkv` handoff. Supported members:
  `alpha4_residual_repair.amr1`, `.amr1.xz`, `.amr1.zlib`, and `.amr1.br`.
- Runtime repair checks are fail-closed: AMR1 magic/schema/record struct,
  shape, bounds, class ids, decoded-candidate SHA, and full-repair source SHA
  are verified before generation.
- `experiments/build_alpha_mask_replacement_archive.py` can now build
  `repair_policy=full` or `class_prefix_<ids>` archives with raw/zlib/xz/brotli
  repair members and records policy, selected runs/pixels, raw/compressed SHA,
  and anchor mask SHA matching.
- Added focused tests:
  `src/tac/tests/test_inflate_renderer_grayscale_repair.py` and expanded
  `src/tac/tests/test_build_alpha_mask_replacement_archive.py`.
- Added durable protocol text to `AGENTS.md` for Alpha sparse-repair archive
  custody.

Verification:

- `.venv/bin/python -m py_compile submissions/robust_current/inflate_renderer_grayscale.py experiments/build_alpha_mask_replacement_archive.py src/tac/tests/test_build_alpha_mask_replacement_archive.py src/tac/tests/test_inflate_renderer_grayscale_repair.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_alpha_mask_replacement_archive.py src/tac/tests/test_inflate_renderer_grayscale_repair.py -q`
- `bash -n submissions/robust_current/inflate.sh`
- Result: focused tests `6 passed in 0.42s`; shell syntax and py-compile clean.

Concrete archives built on C-044:

- `experiments/results/alpha_mask_repair_c044_crf63_class2_lzma_20260501/archive.zip`
  - bytes/SHA: `549148`,
    `fc2721050dd3cca77c84ddb0604d91701a7c371de079bbd29ede61d7299dae03`
  - byte delta vs C-044: `-60815`, formula rate delta `-0.04049421223412485`
  - repair policy: `class_prefix_2`, selected `177142` pixels,
    `82201` runs, `0.098386914921` residual coverage, lzma member
    `167288` bytes.
  - score claim: none until CUDA auth eval returns.
- `experiments/results/alpha_mask_repair_c044_crf62_class2_lzma_20260501/archive.zip`
  - bytes/SHA: `606572`,
    `3cd592c53056585944dcf270fcc39532c01750ebe7b0ded24ffe695715d907b0`
  - byte delta vs C-044: `-3391`, formula rate delta `-0.002257927710037283`
  - repair policy: `class_prefix_2`, selected `137331` pixels,
    `75747` runs, `0.097578981699` residual coverage, lzma member
    `151836` bytes.
  - score claim: none until CUDA auth eval returns.
- Also built `alpha_mask_repair_c044_crf63_no_repair_20260501` at `381669`
  bytes for custody/design comparison only. It was not dispatched because the
  earlier grayscale-only exact evals already showed severe collapse for this
  family of measured configs.

Lightning staging and queue:

- Local/remote supply-chain scans were clean. Doctor passed SSH, remote scan,
  and machine inventory.
- Staged source/artifacts with
  `.omx/state/alpha_repair_exact_20260501T1540Z_manifest.json`; remote manifest
  verification reported `1165` files and `20790441` bytes.
- Submitted two exact CUDA/T4 jobs:
  - `exact_eval_alpha_repair_c044_crf63_class2_t4_20260501T1540Z`
  - `exact_eval_alpha_repair_c044_crf62_class2_t4_20260501T1540Z`
- Both jobs use C-044 as baseline: score `0.9975385870574276`, bytes `609963`,
  PoseNet `0.00357302`, SegNet `0.00402365`; component gates are forensic
  fail-open with `max_*_relative=1.25` so collapses preserve evidence.

Follow-up top-frame-group policy:

- Added `top_residual_frame_groups_<count>_of_<total>` support to
  `experiments/build_alpha_mask_replacement_archive.py`, matching the residual
  planner's frame-group ranking: groups are ranked by selected residual pixels,
  then lower group id.
- Verification after this patch:
  - `.venv/bin/python -m py_compile experiments/build_alpha_mask_replacement_archive.py`
  - `.venv/bin/python -m pytest src/tac/tests/test_build_alpha_mask_replacement_archive.py -q`
  - Result: `3 passed in 0.39s`.
- Built
  `experiments/results/alpha_mask_repair_c044_crf63_topgroup1_lzma_20260501/archive.zip`
  at `467747` bytes, SHA
  `e2d0548ee63d0df4c6ab3c9e3a2ce9c23a8062cdf1274bc3ea29a3179dbab6d9`.
  It is `142216` bytes under C-044, formula-only rate delta
  `-0.09469579687722272`, with one top residual frame group selected
  (`frames 1000-1049`, `93336` pixels, `43074` runs).
- Restaged source/artifacts with
  `.omx/state/alpha_repair_topgroup_exact_20260501T1546Z_manifest.json`
  (`1166` files, `21261921` bytes, remote verify OK).
- Submitted third exact CUDA/T4 job:
  `exact_eval_alpha_repair_c044_crf63_topgroup1_t4_20260501T1546Z`.
  Direct SDK status at submission+poll: all three Alpha sparse-repair jobs
  were still `Pending`, cost `0.0`, machine `T4`.

Next required action:

- Harvest the three Alpha sparse-repair jobs through
  `scripts/launch_lightning_batch_job.py harvest-ssh --require-adjudication`
  as soon as SDK status reaches completed or artifacts appear. A frontier claim
  requires exact CUDA JSON, T4 preflight, archive SHA/bytes, and adjudication.

## Codex Alpha sparse-repair follow-up grid - 2026-05-01T15:53Z

Purpose:

- Expand the Alpha sparse-repair candidate set while the first exact CUDA jobs
  run. This is byte-custody planning only until each archive returns exact
  CUDA auth eval and adjudication.

Additional deterministic archives built on C-044:

- `alpha_mask_repair_c044_crf63_topgroup2_lzma_20260501/archive.zip`
  - bytes/SHA: `547416`,
    `c41be612114b05189038d782f9795469f9eb75f0a7380ac1939b27c746986fea`
  - byte delta vs C-044: `-62547`, formula-only rate delta
    `-0.04164747994093245`
  - repair policy: `top_residual_frame_groups_2_of_24`, selected `179314`
    pixels, `0.099593271286` residual coverage.
- `alpha_mask_repair_c044_crf62_topgroup1_lzma_20260501/archive.zip`
  - bytes/SHA: `533251`,
    `98766ecc68e10ed446ce9fe9987baef9c293c06f68569ba0a4aa01730cf74b96`
  - byte delta vs C-044: `-76712`, formula-only rate delta
    `-0.05107937201190801`
  - repair policy: `top_residual_frame_groups_1_of_24`, selected `74590`
    pixels, `0.05299907701` residual coverage.

Byte-regressive local archives retained for design comparison only:

- `alpha_mask_repair_c044_crf63_topgroup4_lzma_20260501`: `709113` bytes,
  `+0.0660199152020633` formula rate delta vs C-044.
- `alpha_mask_repair_c044_crf63_topgroup8_lzma_20260501`: `1024569` bytes,
  `+0.27606911711817095` formula rate delta vs C-044.
- `alpha_mask_repair_c044_crf63_class2_1_lzma_20260501`: `1186991` bytes,
  `+0.38421926000218026` formula rate delta vs C-044.
- `alpha_mask_repair_c044_crf62_topgroup4_lzma_20260501`: `752961` bytes,
  `+0.09521649857856426` formula rate delta vs C-044.
- `alpha_mask_repair_c044_crf60_topgroup1_lzma_20260501`: `636544` bytes,
  `+0.017699196832940436` formula rate delta vs C-044.

Lightning staging and dispatch:

- Staged the two viable follow-up archives through
  `scripts/lightning_repro_workspace.py` with manifest
  `.omx/state/alpha_repair_next_exact_20260501T1600Z_manifest.json`.
  Remote manifest verification passed with `1165` files and `20719121` bytes.
- Submitted two additional exact CUDA/T4 jobs:
  - `exact_eval_alpha_repair_c044_crf63_topgroup2_t4_20260501T1600Z`
  - `exact_eval_alpha_repair_c044_crf62_topgroup1_t4_20260501T1600Z`
- Direct SDK status after submission:
  - CRF63 class2: `Running`, T4, cost `0.102355555`
  - CRF62 class2: `Running`, T4, cost `0.08765556`
  - CRF63 topgroup1: `Running`, T4, cost `0.0539`
  - CRF63 topgroup2: `Pending`, T4
  - CRF62 topgroup1: `Pending`, T4

Scientific caution:

- None of these Alpha sparse-repair archives has score evidence yet. The byte
  wins are necessary but not sufficient; the exact question is whether partial
  temporal/class repair protects PoseNet and SegNet enough to convert rate
  savings into a real frontier move.

## Codex AMR1 auth-eval allowlist fix and rerun - 2026-05-01T15:59Z

Failure classification:

- The first two Alpha sparse-repair Lightning jobs failed before scoring:
  `contest_auth_eval.py` extracted the archive, then rejected
  `alpha4_residual_repair.amr1.xz` as an unknown member suffix.
- This is a harness allowlist bug, not Alpha lane performance evidence.
  No `contest_auth_eval.json` was produced, so there is no score claim,
  no component signal, and no method conclusion.
- Failure logs are preserved under
  `.omx/state/lightning_alpha_repair_amr1_allowlist_failures_20260501T1600Z/`;
  both saved summaries mark `contains_amr1_allowlist_failure=true`.

Permanent fix landed:

- `experiments/contest_auth_eval.py` now admits `.amrc` plus AMR1 sparse-repair
  payload suffixes: `.amr1`, `.amr1.xz`, `.amr1.zlib`, `.amr1.br`.
- `experiments/canonical_local_auth_eval_smoke.py` whitelist was updated for
  parity.
- Tests added/updated:
  - `src/tac/tests/test_runtime_guards_pass_3.py`
  - `src/tac/tests/test_canonical_local_e2e_smoke.py`
- Verification:
  - `.venv/bin/python -m py_compile experiments/contest_auth_eval.py experiments/canonical_local_auth_eval_smoke.py src/tac/tests/test_runtime_guards_pass_3.py src/tac/tests/test_canonical_local_e2e_smoke.py`
  - `.venv/bin/python -m pytest src/tac/tests/test_runtime_guards_pass_3.py src/tac/tests/test_canonical_local_e2e_smoke.py::test_smoke_whitelist_parity_with_contest_auth_eval -q`
  - Result: `11 passed in 0.07s`.
- Durable protocol note added to `AGENTS.md`: AMR1 archive suffixes must remain
  in exact auth-eval and local smoke whitelists together.

Stale job handling:

- Stopped or let fail the pre-fix job set because they were submitted from the
  old source snapshot and could not produce valid Alpha sparse-repair score
  evidence.
- Restaged patched source and all five candidate archives through
  `.omx/state/alpha_repair_amr1_allowlist_fix_20260501T1605Z_manifest.json`.
  Remote manifest verification passed: `1168` files, `22343446` bytes.
- Relaunched exact CUDA/T4 jobs with `fix1` names:
  - `exact_eval_alpha_repair_c044_crf63_class2_fix1_t4_20260501T1605Z`
  - `exact_eval_alpha_repair_c044_crf62_class2_fix1_t4_20260501T1605Z`
  - `exact_eval_alpha_repair_c044_crf63_topgroup1_fix1_t4_20260501T1605Z`
  - `exact_eval_alpha_repair_c044_crf63_topgroup2_fix1_t4_20260501T1605Z`
  - `exact_eval_alpha_repair_c044_crf62_topgroup1_fix1_t4_20260501T1605Z`
- Submit transcripts are stored under
  `.omx/state/alpha_repair_amr1_allowlist_fix_submit_20260501T1605Z/`.
- Because T4 jobs were pending at zero cost, launched L40S diagnostic hedges for
  the two highest-EV byte candidates:
  - `exact_eval_alpha_repair_c044_crf63_topgroup1_fix1_l40sdiag_20260501T1605Z`
  - `exact_eval_alpha_repair_c044_crf62_topgroup1_fix1_l40sdiag_20260501T1605Z`
- L40S diagnostics are CUDA evidence for triage only; they cannot supersede
  T4/equivalent promotion evidence.

## Codex Alpha hard-pair repair selector - 2026-05-01T16:06Z

Rationale:

- The old grayscale-only exact results show severe PoseNet collapse:
  CRF63 grayscale-only T4 score `4.926778674301541`, PoseNet `1.34352684`;
  CRF62 grayscale-only L40S score `4.358051471129711`, PoseNet `1.06793761`.
  Sparse repair must protect pose-critical frames, not only large residual
  regions.
- Existing Lane W hard-pair metadata provides 30 absolute hardest pair indices:
  `119,124,143,210,222,292,329,349,370,372,379,388,401,404,409,417,430,440,444,446,449,454,456,517,522,552,559,579,584,591`.

Implementation landed:

- Added `pair_indices_<ids>` repair policy to
  `experiments/build_alpha_mask_replacement_archive.py`. It selects AMR1 runs
  where `frame_index // 2` is in the provided absolute contest pair set, i.e.
  frames `2*i` and `2*i+1` for each pair index `i`.
- Added test coverage in
  `src/tac/tests/test_build_alpha_mask_replacement_archive.py`.
- Verification:
  - `.venv/bin/python -m py_compile experiments/build_alpha_mask_replacement_archive.py src/tac/tests/test_build_alpha_mask_replacement_archive.py`
  - `.venv/bin/python -m pytest src/tac/tests/test_build_alpha_mask_replacement_archive.py -q`
  - Result: `4 passed in 0.41s`.

Hard-pair archives built on C-044:

- `alpha_mask_repair_c044_crf63_hardpairs30_lzma_20260501/archive.zip`
  - bytes/SHA: `473571`,
    `96e233deda811c34b8db44e3fbe4db250aa22c02df6f02b891859528362e5ff2`
  - byte delta vs C-044: `-136392`, formula-only rate delta
    `-0.09081783433423919`
  - selected `94934` pixels, `45379` runs, `60` frames from `30` pairs,
    `0.052727548414` residual coverage.
- `alpha_mask_repair_c044_crf62_hardpairs30_lzma_20260501/archive.zip`
  - bytes/SHA: `537520`,
    `d845c934f5cdbd2e4fca5968b50d972a4139200a8c83783aa56922ca7be0da55`
  - byte delta vs C-044: `-72443`, formula-only rate delta
    `-0.04823682014102946`
  - selected `73985` pixels, `41205` runs, `60` frames from `30` pairs,
    `0.052569201134` residual coverage.

Staging and queue:

- Staged patched source and both hard-pair archives with
  `.omx/state/alpha_repair_hardpairs_exact_20260501T1615Z_manifest.json`
  (`1165` files, `20654581` bytes, remote verify OK).
- Submitted T4 exact jobs plus L40S diagnostic hedges:
  - `exact_eval_alpha_repair_c044_crf63_hardpairs30_fix1_t4_20260501T1615Z`
  - `exact_eval_alpha_repair_c044_crf63_hardpairs30_fix1_l40sdiag_20260501T1615Z`
  - `exact_eval_alpha_repair_c044_crf62_hardpairs30_fix1_t4_20260501T1615Z`
  - `exact_eval_alpha_repair_c044_crf62_hardpairs30_fix1_l40sdiag_20260501T1615Z`
- Submit transcripts are stored under
  `.omx/state/alpha_repair_hardpairs_submit_20260501T1615Z/`.

## Codex Alpha AMR1 sparse-repair exact outcome - 2026-05-01T16:47Z

Exact CUDA outcome:

- All harvested sparse-repair archives are scoped forensic negatives. They are
  useful evidence, but none can rank or promote against C-044.
- T4 and L40S agree on the failure class: archive bytes drop materially, but
  PoseNet leaves the C-044 geometry basin by roughly two to three orders of
  magnitude.
- Best T4 sparse-repair result was still bad:
  `exact_eval_alpha_repair_c044_crf62_class2_fix1_t4_20260501T1605Z`
  scored `4.0751452447605425` at `606572` bytes, SHA
  `3cd592c53056585944dcf270fcc39532c01750ebe7b0ded24ffe695715d907b0`,
  SegNet `0.00674468`, PoseNet `0.89807248`.
- Best byte candidate was also bad:
  `exact_eval_alpha_repair_c044_crf63_pairatom_top10_t4_20260501T1620Z`
  scored `4.86920683032904` at `410939` bytes, SHA
  `56c4011ad368ea9f4f603d70f6930ab25c30066ce1febf3db30ea42f6e715aa0`,
  SegNet `0.00929418`, PoseNet `1.34407389`.

T4 packet matrix:

| job | bytes | score | seg | pose | status |
| --- | ---: | ---: | ---: | ---: | --- |
| `crf62_class2_fix1_t4` | `606572` | `4.0751452447605425` | `0.00674468` | `0.89807248` | `A-negative scoped forensic` |
| `crf62_topgroup1_fix1_t4` | `533251` | `4.198229443231151` | `0.00709643` | `0.98189253` | `A-negative scoped forensic` |
| `crf62_hardpairs30_fix1_t4` | `537520` | `4.236762988266674` | `0.00725173` | `0.99456817` | `A-negative scoped forensic` |
| `crf62_pairatom_top20_t4` | `506890` | `4.261563683070625` | `0.00723193` | `1.02454627` | `A-negative scoped forensic` |
| `crf62_pairatom_top10_t4` | `480308` | `4.306079773235291` | `0.00725631` | `1.06317163` | `A-negative scoped forensic` |
| `crf63_topgroup2_fix1_t4` | `547416` | `4.710287093193034` | `0.00888891` | `1.19501185` | `A-negative scoped forensic` |
| `crf63_class2_fix1_t4` | `549148` | `4.734057947482995` | `0.00857007` | `1.23299015` | `A-negative scoped forensic` |
| `crf63_topgroup1_fix1_t4` | `467747` | `4.761320183455963` | `0.00914294` | `1.25002742` | `A-negative scoped forensic` |
| `crf63_pairatom_top10_t4` | `410939` | `4.86920683032904` | `0.00929418` | `1.34407389` | `A-negative scoped forensic` |
| `crf63_pairatom_top20_t4` | `440201` | `4.93659541719088` | `0.00922456` | `1.38460469` | `A-negative scoped forensic` |
| `crf63_hardpairs30_fix1_t4` | `473571` | `4.978544093908586` | `0.00900789` | `1.41558313` | `A-negative scoped forensic` |

Interpretation:

- Do not call this an Alpha family kill. It narrowly retires the measured
  configuration family: post-hoc CRF62/CRF63 grayscale base plus AMR1 residual
  repairs selected by class prefix, residual frame group, hard-pair prior, or
  pair-atom byte prior.
- The failed assumption was separability. Once the base mask stream exits the
  PoseNet geometry basin, local residual pixels do not have stable marginal
  benefits. This invalidates direct water-filling over those AMR1 atoms.
- Water-filling/Lagrangian allocation remains the right control language, but
  the atom type must move up one level. Candidate atoms now need to be
  geometry-preserving units: pair-consistent latent decoder changes, temporal
  endpoint repairs, class-confusion transforms, pose-conditioned generator
  parameters, or learned soft-LUT/SegMap/Q-FAITHFUL latent packets.

External theory note:

- The relevant outside pattern is standard rate-distortion optimized coding:
  compare coding decisions by a Lagrangian cost such as `D + lambda R`, then
  adapt quantization/feature choices to equalize marginal slope.
- Task-aware compression papers add downstream task loss to the distortion
  term. For this contest, the distortion term is exactly the scored
  SegNet/PoseNet component response, not visual reconstruction.
- Therefore the next allocator should rank atoms by exact or calibrated
  `score_saved_per_byte`, with a feasibility gate that rejects any base whose
  PoseNet/SegNet components already leave the local basin.

Next redesign:

1. Freeze the AMR1 sparse-repair matrix as A-negative scoped forensic evidence.
2. Stop spending exact eval on additional AMR1 residual selectors for CRF62/63
   unless the base representation changes.
3. Build a `repair_atom_response_v2` planner around geometry-preserving atoms:
   pair, frame, class-confusion, temporal endpoint, boundary band, and learned
   latent correction, each with charged bytes and component benefit.
4. Make the allocator solve a Lagrangian/knapsack objective only after atoms
   have a non-collapsed baseline:
   `benefit = 100*dseg_saved + sqrt(10*pose_before)-sqrt(10*pose_after)`,
   `rate_cost = 25*bytes/37545489`.
5. Highest-EV execution path remains corrected soft-LUT SA/FilmCanvas,
   Q-FAITHFUL/Quantizr-like pair-conditioned generation, or NeRV/INR/HNeRV
   geometry-preserving learned topology. Plain CRF grayscale plus AMR1 residual
   is now a forensic control.

Tooling follow-up:

- `experiments/alpha_repair_atom_planner.py` now accepts optional
  `--baseline-contest-json` and `--candidate-contest-json` inputs and performs
  an exact component geometry-basin check before emitting water-filling archive
  policies.
- If candidate PoseNet/SegNet exceed the baseline relative limits, the planner
  still emits atom byte/prior data but sets `water_filling_allowed=false` and
  leaves `recommended_archive_policies=[]`.
- Focused verification: `src/tac/tests/test_alpha_repair_atom_planner.py`
  reports `4 passed`.
- Real blocked artifacts:
  - `experiments/results/alpha_repair_atom_plan_c044_crf63_pair_lzma_20260501_basin_blocked/alpha_repair_atom_plan.json`
    with PoseNet ratio `376.173066481576`, SegNet ratio `2.309887788451`.
  - `experiments/results/alpha_repair_atom_plan_c044_crf62_pair_lzma_20260501_basin_blocked/alpha_repair_atom_plan.json`
    with PoseNet ratio `251.348293600372`, SegNet ratio `1.676259117965`.

## Codex component-manifold probe artifact - 2026-05-01T17:22Z

Implementation landed:

- Added `experiments/build_component_manifold_probe_plan.py`.
- Added focused tests in
  `src/tac/tests/test_component_manifold_probe_plan.py`.
- The artifact schema is `component_manifold_probe_plan_v1`.
- It consumes exact CUDA `contest_auth_eval*.json` points and emits:
  `axis_id`, `family`, `epsilon`, archive SHA/bytes, SegNet/PoseNet/score
  deltas, geometry-basin status, slope, curvature, continuation candidates,
  blocked points, and optional synergy/antagonism records.
- It is deliberately non-promotable: `score_claim=false`,
  `promotion_eligible=false`, `evidence_grade=derivation`.

Focused verification:

- `.venv/bin/python -m py_compile experiments/build_component_manifold_probe_plan.py src/tac/tests/test_component_manifold_probe_plan.py`
- `.venv/bin/python -m pytest src/tac/tests/test_component_manifold_probe_plan.py -q`
- Result: `3 passed in 0.08s`.

Real artifacts:

- `experiments/results/component_manifold_probe_c044_macro_20260501/component_manifold_probe_plan.json`
  - SHA-256:
    `a7253cce788f8e5b973bf7f0008c33b61bdab91a4c4ff2f06037223f47bed822`
  - Baseline: C-044, score `0.9975385870574276`, bytes `609963`.
  - Continuation candidates: none.
  - Inside-basin but not better: `direct_fd_m2_frontier_t4`, score delta
    `+0.03809699922241671`, bytes delta `+76669`, PoseNet ratio
    `0.872502812747`, SegNet ratio `0.998774744324`.
  - Blocked collapsed points:
    - `lane12_nerv_jsonfix40`: score delta `+25.039654717496862`,
      bytes delta `-313485`, PoseNet ratio `13931.771890445618`,
      SegNet ratio `8.769860698619`.
    - `alpha_crf62_class2_fix1_t4`: score delta `+3.077606657703115`,
      bytes delta `-3391`, PoseNet ratio `251.348293600372`,
      SegNet ratio `1.676259117965`.
    - `alpha_crf63_pairatom_top10_t4`: score delta
      `+3.8716682432716127`, bytes delta `-199024`, PoseNet ratio
      `376.173066481576`, SegNet ratio `2.309887788451`.
- `experiments/results/component_manifold_probe_direct_fd_response_20260501/component_manifold_probe_plan.json`
  - SHA-256:
    `c24574a3a83a51cfac6a6d01fff235e2a512f3be4397bcef7d2e20a5d975c4c4`
  - Baseline: direct-FD response point `eps=0`, score
    `1.043987524793892`, bytes `686635`.
  - All four `eps={-2,-1,+1,+2}` points stay inside the component basin and
    improve versus that local baseline.
  - Central score slope around `eps=±1`: `0.0003277501494941548`.
  - Score curvature around `eps=±1`: `-0.014399711023609596`.
  - Central score slope around `eps=±2`: `0.0005555742777479944`.
  - Score curvature around `eps=±2`: `-0.0036204613492895055`.

Interpretation:

- The C-044 macro chart says the current post-hoc CRF/AMR1 and old NeRV points
  are outside the scorer geometry basin; they update risk priors only.
- The direct-FD local chart shows a safe, curved, measurable tangent direction
  near its own PFP16 response baseline. That is the kind of local chart the
  next learned soft-LUT/SegMap/Q-FAITHFUL/INR axes should produce before any
  water-filling policy emits exact-eval archives.
- This converts the manifold/differential-equation plan into an executable
  control artifact without pretending that the artifact itself is evidence of
  a better archive.

Next use:

1. Build similar local charts for corrected soft-LUT SegMap/FilmCanvas,
   Q-FAITHFUL export variants, and any NeRV/INR/HNeRV build-only archives that
   do not require new retraining.
2. Let heterogeneous resources enter through atoms in the same coordinate
   system: Python learned decoders, Rust/Zig/C deterministic decoders,
   arithmetic/ANS/range streams, temporal grammars, symbolic constraints,
   Bayesian/bandit selectors, control-theoretic lambda updates, and
   differential-equation continuation paths.
3. Dispatch exact archives only from points that remain inside the component
   basin or from explicit cliff-probing diagnostics marked non-promotable.

## Codex contest docs and source-video reverse engineering - 2026-05-01T17:45Z

Added `.omx/research/contest_docs_video_reverse_engineering_20260501_codex.md`.

Implementation landed:

- `experiments/reverse_engineer_contest_video.py`
- `src/tac/tests/test_reverse_engineer_contest_video.py`
- Focused verification: `3 passed in 0.42s`.

Current official/public-doc signal:

- Contest repo confirms one fixed 1-minute `0.mkv`, semantic SegNet
  disagreement, temporal PoseNet MSE over two consecutive frames, archive-rate
  term, public PR submissions, and T4 GPU evaluation for GPU inflate.
- Current public leaderboard top rows now include `qpose14` `0.32`,
  `unified_brotli` `0.33`, `quantizr` `0.33`, `fp4_mask_gen` `0.37`, and
  `selfcomp` `0.38`.
- Public PR descriptions reinforce the same architecture target:
  Quantizr-style neural renderer, full-res 5-class masks, compact quantized
  pose/velocity side-channel, FP4/tiny renderer weights, and cross-stream
  brotli packing.

Local source-video anatomy:

- Exact video SHA:
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`.
- HEVC `1164x874`, `20 fps`, `1200` frames, `600` pairs,
  `37,545,489` bytes.
- Full-res mask proxy artifact:
  `experiments/results/contest_video_reverse_engineering_20260501/contest_video_reverse_engineering_fullres_masks.json`
  with SHA
  `b852daeb64f2a950f961a3cb3d09cea3717a20cfab398f87444ae6087e749bd1`.
- Low-res proxy artifact:
  `experiments/results/contest_video_reverse_engineering_20260501/contest_video_reverse_engineering.json`
  decodes `submissions/robust_current/masks.mkv` to `1200x48x64`; use it as
  forensic metadata only, not foveation/ego-motion selector evidence.
- Full-res lane-mark log-zoom has std `0.14425052012530618`, matching the
  existing lane-mark pose coupling assumptions.
- High luma-motion pairs cluster around `514-523`; high lane-log-zoom pairs
  include `208`, `218`, `94`, `528`, `544`, `212`, `91`, `598`, `157`,
  `513`.

Immediate decision:

- Add hardware/ego/foveation atoms to the component-manifold planner rather
  than launching more collapsed AMR1 selectors.
- Build the next candidate around the public-leaderboard lesson:
  full-res semantic controls + tiny charged renderer + quantized/delta-coded
  velocity or log-zoom pose channel + unified packing.
- Hyperbolic/Telescope foveation is viable as a controlled exact probe if its
  parameter side-channel is charged and starts from identity/FoE-safe
  trust-region values; it is not a standalone score claim.

## Codex live leaderboard reverse-engineering refresh - 2026-05-01T17:14Z

The current public lowest scores on the active lossy video compression
leaderboard are:

| name | score | bytes | PoseNet | SegNet |
| --- | ---: | ---: | ---: | ---: |
| `qpose14` | `0.32` | `287573` | `0.00052154` | `0.00061261` |
| `unified_brotli` | `0.33` | `287165` | `0.00061622` | `0.00061261` |
| `quantizr` | `0.33` | `299970` | `0.00051328` | `0.00061261` |
| `fp4_mask_gen` | `0.37` | `249624` | `0.00076576` | `0.00121106` |
| `selfcomp` | `0.38` | `279036` | `0.00055221` | `0.00122167` |

Temporary public-branch clones were inspected under
`/tmp/pact_topsubs_dpEPDD/` for qpose14, unified_brotli, quantizr,
fp4_mask_gen, and selfcomp. Key technical signal:

- qpose/unified/quantizr/fp4 are all variants of a small depthwise-separable
  neural generator driven by full-res 5-class mask controls and a compact
  pose/velocity side-channel.
- qpose/unified make the rate side almost surgical: one zip member, FP4 model,
  AV1/brotli mask, quantized or delta-coded pose, and sometimes dropping
  rotation entirely.
- selfcomp is the correct grayscale lesson: Gaussian soft-LUT probability map
  into a learned SegMap plus affine latent motion, not hard-class CRF
  grayscale.
- The qpose-class non-rate target is already around `0.133-0.134` score
  points. At C-044, non-rate is around `0.591`, so post-hoc byte repair cannot
  bridge the gap.

Next action stays strict:

1. Quantizr/qpose-style export closure and scorer-path training become the
   highest-EV sub-0.3 lane once retraining is unblocked.
2. Until the Lane 12 clearance file exists, continue build-only closure:
   archive contracts, FP4 export, qpose-style pose packing, unified payload
   packer, and identity-safe foveation probes.
3. Use public leaderboard code only as external design signal. Local claims
   still require exact CUDA auth eval on our archive bytes.

## Codex C-044 lossless unified-payload pack - 2026-05-01T17:26Z

Lane 12 retraining remains gated, so the current no-retraining frontier action
is lossless archive packing against exact C-044 bytes.

Built artifacts:

- Source A++ archive:
  `experiments/results/lightning_batch/exact_eval_owv3_0120_stack_t4_20260501T150652Z/archive.zip`
  with bytes `609963`, SHA
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`,
  score `0.9975405870574277`, PoseNet `0.00357302`, SegNet `0.00402367`.
- Per-member Brotli diet-pack screen:
  `experiments/results/renderer_dietpack_c044_20260501/archive.zip`,
  bytes `594746`, byte-exact decoded members, formula-only score delta
  `-0.010132375689660082`.
- Unified renderer payload screen:
  `experiments/results/renderer_packed_payload_c044_20260501/archive.zip`,
  bytes `594634`, SHA
  `ff01e11f525514cebd27325d8207c06351df437cd25d61ee82af15b6e0ddcae6`,
  formula-only score delta `-0.010206951892409765`.

Implementation landed:

- `submissions/robust_current/unpack_renderer_payload.py` expands
  `renderer_payload.bin[.br]` into the existing renderer members with strict
  magic/header/member/SHA/overwrite checks.
- `submissions/robust_current/inflate.sh` now expands the unified payload
  after Stage 0 Brotli decompression and before renderer dispatch.
- `experiments/build_renderer_packed_payload_archive.py` builds deterministic
  single-member payload archives with `score_claim=false`.
- `src/tac/submission_archive.py` now detects both unified payload archives
  and legacy per-member `.br` renderer archives.

Verification:

- `py_compile` passed on the builder, unpacker, archive helper, and focused
  tests.
- `bash -n submissions/robust_current/inflate.sh` passed.
- Focused tests passed: `21 passed`.
- The real C-044 unified payload unpacked locally to byte-identical
  `renderer.bin`, `masks.mkv`, and `optimized_poses.bin`.
- `experiments/contest_auth_eval.py` archive-member whitelist accepts
  `renderer_payload.bin.br` and `renderer_payload.bin`.

Exact eval queue:

- Lightning doctor:
  `.omx/state/lightning_doctor_renderer_payload_c044_t4_20260501T172533Z.json`
  status `OK`.
- Source manifest:
  `.omx/state/exact_eval_renderer_payload_c044_t4_20260501T1726Z_manifest.json`
  verified remotely with `1174` files and `20341851` bytes.
- T4 exact job:
  `exact_eval_renderer_payload_c044_t4_20260501T1726Z`.
- Expected result if lossless CUDA components match C-044:
  approximately `0.987333635165018`.

This is still a pending exact-eval candidate. Do not promote until
`contest_auth_eval.adjudicated.json` is harvested and component gates pass.

Follow-up during the same queue wait:

- Added lossless `pose_fp16_col_delta_v1` inside the same payload header.
  Runtime expands it to exact `optimized_poses.bin` bytes before the renderer
  loads poses; the header carries both encoded SHA and decoded pose SHA.
- Focused pose-codec tests pass: `7 passed`.
- Pose-codec candidate:
  `experiments/results/renderer_packed_payload_c044_posecd_20260501/archive.zip`,
  bytes `594456`, SHA
  `f8b13737ea226524869b40132a31ca77ffdaa887ca8571e51e408861e27ecb54`,
  formula-only score delta `-0.01032547478606551`.
- T4 exact job:
  `exact_eval_renderer_payload_posecd_c044_t4_20260501T1731Z`.
- Expected result if lossless CUDA components match C-044:
  approximately `0.9872151122713622`.

The pose-codec job supersedes the raw unified-payload job as the preferred
pending pack candidate, but both remain non-claims until exact CUDA artifacts
land.

## Codex renderer payload exact frontier and atom system - 2026-05-01T18:12Z

Live public leaderboard was refreshed from `https://comma.ai/leaderboard`:
`qpose14` remains `0.32`, `unified_brotli` and `quantizr` remain `0.33`,
`fp4_mask_gen` is `0.37`, and `selfcomp` is `0.38`.

Exact local frontier update:

- `exact_eval_renderer_payload_posecd_c044_t4fix1_20260501T1754Z`
  harvested as A++ contest T4.
- Archive bytes/SHA: `594456`,
  `f8b13737ea226524869b40132a31ca77ffdaa887ca8571e51e408861e27ecb54`.
- Exact score: `0.9872158806043723`.
- Component gates passed: PoseNet `0.00357305` versus C-044 reference
  `0.00357302`, SegNet `0.00402367` versus reference `0.00402367`.
- Score delta versus prior C-044 T4 frontier:
  `-0.010324706453055388`.

Pending better pack:

- `exact_eval_renderer_payload_posecd_palias_c044_t4_20260501T1806Z`
  is running on T4 with exact archive bytes `594412` and SHA
  `831f643f5c523dd4cf524cec18faf65e5d6e572a7e347ebd9ef712087fce2c09`.
  If components match, it should supersede the non-p posecd result by
  another `44` bytes of rate.

Additional top-submission probe:

- Implemented `pose_qpose14_col_delta_v1`, an all-channel qpose-style
  quantized pose codec with max local pose-value error `0.000244140625` on
  dims 1-5 and zero error on dim0 for C-044.
- Focused renderer payload tests pass: `11 passed`.
- Built empirical archive:
  `experiments/results/renderer_packed_payload_c044_qpose14_palias_20260501/archive.zip`,
  bytes `594458`, SHA
  `33cf2d63647bca34302789aa5bbf779b2453b38bfb812054c65599cd7999971d`.
- It is byte-dominated by lossless posecd p (`594412` bytes), so it should
  only receive exact eval if we explicitly test whether qpose-grid smoothing
  improves PoseNet enough to pay for the `46` byte rate loss.

Alpha repair wave accounting:

- Harvested Alpha sparse-repair CRF62/CRF63 exact T4 packets are all
  A-negative scoped forensic results, with scores roughly `4.07` to `4.98`.
- Conclusion is narrow: sparse AMR1 repair on top of the measured CRF
  grayscale base does not escape the PoseNet geometry collapse. It does not
  kill learned soft-LUT/qpose/Quantizr-style representations.

Mathematical planning note:

- Added `.omx/research/atom_lagrangian_waterfill_sub03_system_20260501_codex.md`.
- It formalizes the next allocator as atomized rate-distortion water filling:
  every pair, frame, pose dimension, mask region, foveation transform, decoder
  weight block, and archive-layout byte is an atom with charged bytes,
  component benefit, confidence, and interaction risk.

## Codex RP2 frontier and qpose14/QZS3 reverse engineering - 2026-05-01T18:40Z

Exact frontier update:

- Harvested
  `exact_eval_renderer_payload_posecd_rp2_palias_allowp_c044_t4_20260501T1823Z`
  as A++ contest T4.
- Archive bytes/SHA: `594111`,
  `090541612003c4f153e448925cc9152787b0842338ae08ec0f1420e997c7ed11`.
- Exact score: `0.986986909633818`.
- Component gates passed versus C-049: PoseNet `0.00357307` versus
  `0.00357305`, SegNet `0.00402367` versus `0.00402367`.
- Score delta versus C-049:
  `-0.0002289709705542986`.
- Score delta versus C-044:
  approximately `-0.010553677423609687`.
- Added C-050 to
  `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`; C-050 is now
  the active exact frontier.

Follow-up exact-eval harvest:

- `exact_eval_renderer_payload_qpose14_rp2_palias_allowp_c044_t4_20260501T1823Z`
  initially showed a nonterminal `Running -> Pending` SDK status regression
  after accruing cost, so it was marked status-reconciliation-required.
  Terminal artifacts later landed and were harvested through the state-derived
  SSH path; terminal artifact custody supersedes the nonterminal telemetry
  anomaly.
- Archive bytes/SHA: `594047`,
  `dc855b10b69353f1046aeb25d2eba17f43f48039ea0ef2f2d95f5c2a2bef782f`.
- Exact score: `0.9867772369277311`.
- Component gates passed versus C-049: PoseNet `0.00356884` versus
  `0.00357305`, SegNet `0.00402312` versus `0.00402367`.
- Score delta versus C-049:
  `-0.00043864367664114834`.
- Score delta versus C-050:
  `-0.0002096727060869`.
- Added C-051 to
  `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`; C-051 is now
  the active exact frontier.

Top-submission reverse-engineering facts:

- Public leaderboard refresh from `https://comma.ai/leaderboard` still renders
  `qpose14` at `0.32`, `unified_brotli` and `quantizr` at `0.33`,
  `fp4_mask_gen` at `0.37`, and `selfcomp` at `0.38`. GitHub PR #67 title
  says `qpose14_qzs3_filmq9g_slsb1_r55 (0.31)`, so archive/code artifacts are
  the stronger source for engineering decisions.
- PR #67 archive:
  `reports/raw/top_submission_prs` equivalent tmp clone at
  `/tmp/pact_pr_reverse_20260501T182548Z/pr67/submissions/qpose14_qzs3_filmq9g_slsb1_r55/archive.zip`,
  bytes `276564`, archive SHA
  `a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765`,
  single member `p` size `276464`.
- PR #67 `p` decomposes into `mask_obu_br` `219472` bytes, `model_qzs3_br`
  `56093` bytes, and `pose_qp1_br` `899` bytes. Decompressed raw segments are
  mask `223385` bytes, QZS3 model `59288` bytes, and QP1 pose `1140` bytes.
- PR #65 archive:
  bytes `284425`, archive SHA
  `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68`,
  single member `x` size `284325`, with first ten 24-bit segment lengths
  `[219472, 57074, 1487, 1400, 226, 106, 149, 154, 223, 273]`.
- Both PRs use the same `JointFrameGenerator` architecture shape as
  `src/tac/quantizr_faithful_renderer.py`: `87836` params, `111` state-dict
  keys, same key order and tensor shapes.
- The exact comparison artifact is
  `experiments/results/top_submission_reverse_roundtrip_20260501/archive_anatomy.json`.

Implementation landed:

- Added `experiments/reverse_engineer_top_submissions.py`, a canonical
  top-submission reproducer that pins PR #67/#65 commits, enforces expected
  archive SHA/bytes, emits deterministic JSON with no host paths or timestamps,
  and records `score_claim=false`.
- Added `src/tac/quantizr_qzs3_codec.py`, a PR #67-style grouped FP4/QV codec
  for Quantizr-faithful `JointFrameGenerator` state dicts.
- Verified the decoder against the actual PR #67 decompressed QZS3 model blob:
  it reconstructs all `111` tensors.
- Added QZS3 dispatch in `submissions/robust_current/inflate_renderer.py` with
  an inflate shim matching the QFAI pair-output contract.
- Added `QZS3` to `experiments/canonical_local_auth_eval_smoke.py` renderer
  magic recognition.
- Added `experiments/repack_quantizr_faithful_qzs3_archive.py`, a build-only
  deterministic repacker that refuses non-`QFAI`/`QZS3` renderer payloads so it
  cannot silently relabel OWV3/ASYM archives.

Verification:

- `py_compile` passed on QZS3 codec, inflate renderer, smoke tool, repacker,
  and top-submission reproducer.
- Focused tests passed: `20 passed` for top-submission specs,
  Q-FAITHFUL/QZS3, and smoke QZS3 recognition.
- `git diff --check` passed for touched QZS3/repacker/smoke/test files.

Strategic consequence:

- C-051 is still an OWV3 archive, so QZS3 cannot shrink the active frontier
  directly. The #1/#2 trick is a high-value accelerator for the Q-FAITHFUL /
  learned-JointFrameGenerator lane: train or recover a good JFG candidate, pack
  it with QZS3/QP1/single-blob layout, then exact-eval. The active no-retraining
  work continues on RP2/pose/payload atoms; the sub-0.3 break requires the
  learned JFG/SegMap/soft-LUT representation family plus this archive packing.

## Codex Q-FAITHFUL packer greenup and early snapshot eval - 2026-05-01T19:55Z

Live public leaderboard refresh:

- `https://comma.ai/leaderboard` renders `qpose14` at `0.32`, then
  `unified_brotli` and `quantizr` at `0.33`.
- PR #63 qpose14 reports CUDA `600` sample components:
  PoseNet `0.00052154`, SegNet `0.00061261`, bytes `287573`.
- PR #64 unified_brotli reports CUDA `600` sample components:
  PoseNet `0.00061622`, SegNet `0.00061261`, bytes `287165`.

Exact local frontier remains C-051:

- Archive bytes/SHA: `594047`,
  `dc855b10b69353f1046aeb25d2eba17f43f48039ea0ef2f2d95f5c2a2bef782f`.
- Exact T4 score: `0.9867772369277311`.
- Components: PoseNet `0.00356884`, SegNet `0.00402312`.

QP1 diagnostic exact eval:

- Job:
  `exact_eval_renderer_payload_qp1_rp2_palias_c044_l40s_fix1_20260501T1935Z`.
- Artifact:
  `experiments/results/lightning_batch/exact_eval_renderer_payload_qp1_rp2_palias_c044_l40s_fix1_20260501T1935Z/`.
- Archive bytes/SHA: `588562`,
  `2a080314233011b0f82d20cec304d4931eb9d105d8063cb7c56f1b0a1b11b8b9`.
- Exact L40S CUDA score: `2.2175001665333225`.
- Components: PoseNet `0.2441081`, SegNet `0.00263205`.
- Component gate: PoseNet is `68.3198x` the C-044 reference; SegNet improves.
- Classification: A-negative scoped forensic. This does not kill QP1 for
  learned qpose/Q-FAITHFUL renderers; it only says velocity-only pose is not
  transferable to the current OWV3 geometry basin.

Q-FAITHFUL live training:

- Vast instance `35959478`, RTX 4090, remains healthy.
- PID `71346` is training
  `src/tac/experiments/train_renderer.py --profile q_faithful_dilated_88k`.
- Latest checked epoch: about `280/3000`, phase P1, loss around `0.0204`,
  GPU utilization around `92-100%`.
- Half-frame mask artifact is stable:
  bytes `223738`, SHA
  `568439a26dbb9e240cf2c385ff350ddc2cb81f5bc514577498446d6043b21178`.

Postprocess implementation fixes:

- `scripts/remote_q_faithful_postprocess_fixed.sh` now supports crash-recovery
  checkpoints with weights under `model`, not only `model_state_dict` or
  `state_dict`.
- The same script now serializes `ArchiveValidationResult` through
  `dataclasses.asdict` with a stringified `archive_path`; the previous
  `to_dict()` assumption was invalid for the current repo API.
- Focused test:
  `src/tac/tests/test_remote_q_faithful_postprocess_fixed_script.py`.

Early Q-FAITHFUL snapshot, build-only:

- Snapshot checkpoint:
  `/workspace/pact/lane_q_faithful_results/postprocess_snapshots/training_state_snapshot_20260501T1947Z.pt`,
  SHA `71a13572ca17e81279c0dd9daeec9f7fe93d6b048ae3332bbf07be41a3c6c6dd`.
- Local mirror:
  `experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/postprocess_fixed_snapshot_20260501T1947Z_fix3/`.
- Raw QFAI archive: `563206` bytes, SHA
  `8ad1b494e03fba010a663d18f5d9b5b530e3fdcbd5fccf7633b0db295f4ad33e`.
- QZS3 archive: `284486` bytes, SHA
  `751f4f9105a3479548685b2a70041b222ddae35ec25bae601b33284e76e48bf3`.
- RP2 qpose14 archive: `276651` bytes, SHA
  `1d5d5e86cb0902f306f1e0968385033172c97ab2b882866beffbf9b26a2acca4`.
- RP2 QP1 archive: `273048` bytes, SHA
  `e70bb3fbda789e4fb8880e80b672a608e318bc0084415958c52fc6631a6a20f4`.
- All early snapshot artifacts remain `score_claim=false` until CUDA auth eval.

New diagnostic exact eval queued:

- Job:
  `exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_l40s_20260501T1952Z`.
- State:
  `.omx/state/qfaithful_snapshot_qzs3_rp2_qpose14_l40s_batch_jobs_20260501T1952Z.json`.
- Source manifest:
  `.omx/state/exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_l40s_20260501T1952Z_manifest.json`.
- Archive under test: early Q-FAITHFUL RP2 qpose14, bytes `276651`, SHA
  `1d5d5e86cb0902f306f1e0968385033172c97ab2b882866beffbf9b26a2acca4`.
- Machine: L40S (`g6e.4xlarge`), advisory CUDA score-grade if it completes.
- Purpose: validate the full top-submission-shaped runtime path without
  stealing the 4090 trainer.

Operational incident:

- A mistaken remote `rsync` command briefly overwrote the Vast `.venv/bin/python`
  symlink with a host-local path while copying extra root files.
- The training process survived because its interpreter was already open.
- The symlink was repaired to `/opt/conda/bin/python3.11`, and CUDA/Torch import
  was reverified. Do not broad-clean the remote checkout while training runs;
  only precise file copies are safe.

## Codex Q-FAITHFUL current-floor transfer update - 2026-05-01T20:06Z

Evidence grade: mixed `external_plus_empirical_byte_anatomy` and queued exact
CUDA work. Score claim: `false` for all Q-FAITHFUL snapshot archives until
auth eval lands.

Current public-floor check:

- `qpose14` remains listed at `0.32`.
- `unified_brotli` and `quantizr` remain listed at `0.33`.
- Source links: comma leaderboard plus PR #63/#64 reports.

Canonical current-floor reverse-engineering output:

```text
experiments/results/top_submission_reverse_roundtrip_20260501/archive_anatomy.json
experiments/results/top_submission_current_floor_20260501/current_floor_archive_anatomy.json
```

Key external finding:

- PR #63/#64 use the same `JointFrameGenerator` parameter count (`87836`), but
  store the renderer as a Torch quantized payload (`40` FP4-packed modules,
  `3` dense FP16 weight modules, `46` dense FP16 entries), not the older PR #67
  QZS3 payload.
- PR #64's byte edge over PR #63 is a single Brotli stream over raw
  mask/model/pose plus velocity-only delta pose coding.
- Our early Q-FAITHFUL QZS3/RP2 archives are smaller than PR #63/#64, but the
  exact question is whether that extra renderer compression remains inside the
  scorer basin.

Queued exact evals:

- Running L40S diagnostic:
  `exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_l40s_20260501T1952Z`.
  Archive `276651` bytes, SHA
  `1d5d5e86cb0902f306f1e0968385033172c97ab2b882866beffbf9b26a2acca4`.
- Running T4 hedge:
  `exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_t4_20260501T1957Z`.
  Same archive bytes/SHA. If it lands first with component gates, it can be
  promotion-grade for that early snapshot.
- Queued L40S sibling:
  `exact_eval_qfaithful_snapshot_qzs3_rp2_qp1_l40s_20260501T2002Z`.
  Archive `273048` bytes, SHA
  `e70bb3fbda789e4fb8880e80b672a608e318bc0084415958c52fc6631a6a20f4`.

Branch rule:

- If qpose14 lands near public-basin quality, immediately harvest, record the
  new frontier if applicable, stop redundant duplicates, and test QP1/PR64-like
  velocity-only packing as the next byte atom.
- If qpose14 runtime passes but score collapses, preserve it as scoped
  A-negative and build a PR63-style Torch quantized model payload from the same
  checkpoint before judging the checkpoint or renderer family.
- If the runtime fails, classify as harness/runtime only, patch closure, and
  relaunch; do not draw scorer conclusions.

## Codex Q-FAITHFUL A-negative harvest and PR63 fallback closure - 2026-05-01T20:20Z

Evidence grade: `A-negative` for the early snapshot exact CUDA diagnostic;
`empirical_byte_only_until_cuda_auth_eval` for the new Torch-FP4 fallback
archives. Score claim: `false` for fallback archives.

Harvested exact diagnostic:

- Job:
  `exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_l40s_20260501T1952Z`.
- Artifact:
  `experiments/results/lightning_batch/exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_l40s_20260501T1952Z/contest_auth_eval.adjudicated.json`.
- Archive: `276651` bytes, SHA
  `1d5d5e86cb0902f306f1e0968385033172c97ab2b882866beffbf9b26a2acca4`.
- Exact CUDA score: `8.420915711675079`.
- Components: PoseNet `4.47971487`, SegNet `0.01543638`, full `600` samples
  on L40S.
- Classification: runtime/custody valid, scorer basin failed. This is a scoped
  early-checkpoint negative, not a Q-FAITHFUL/QZS3 family kill.

Stopped redundant diagnostics:

- `exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_t4_20260501T1957Z` stopped
  after the L40S result showed catastrophic PoseNet collapse for identical
  archive bytes.
- `exact_eval_qfaithful_snapshot_qzs3_rp2_qp1_l40s_20260501T2002Z` stopped at
  `2026-05-01T20:12:13Z`; the all-channel qpose14 sibling already proved the
  same early checkpoint is out of basin.

Implemented current-floor fallback:

- Added deterministic PR63-style Torch-FP4 JointFrameGenerator codec and
  runtime loader:
  `src/tac/quantizr_torch_fp4_codec.py` and
  `submissions/robust_current/inflate_renderer.py`.
- Extended `experiments/repack_quantizr_faithful_qzs3_archive.py` with
  `--renderer-codec torch_fp4`.
- Extended `scripts/remote_q_faithful_postprocess_fixed.sh` so every future
  Q-FAITHFUL checkpoint emits both aggressive QZS3 and conservative
  Torch-FP4 fallback archives.

Existing early-snapshot byte screen with the new fallback:

- `torchfp4_half_posebin`: `302294` bytes, SHA
  `036740fc447314a08475ee338ce6df6f0d01f5a3c5fcf68c27b764f631c6cf02`.
- `torchfp4_rp2_qpose14`: `288889` bytes, SHA
  `9e31e23cb6a66d5c070df73e3e0cb261efea8878706ee76be89bc4bbdc695ba6`.
- `torchfp4_rp2_qp1`: `285346` bytes, SHA
  `1c07e522235014a743314d3f853619083f54f1b4f8166946ab693c47f33c6bc2`.

Interpretation:

- QZS3 remains the higher-rate-EV codec if a later checkpoint enters the
  scorer basin.
- Torch-FP4 is now the conservative current-floor-compatible fallback if QZS3
  is too lossy on a mature checkpoint.
- Do not exact-eval these early fallback archives now; the checkpoint already
  failed basin entry. Use the fallback immediately on the next later checkpoint
  snapshot.

PR64 length-table container follow-up:

- Added `pr64_len_table` to the payload builder and unpacker. It is a
  single-Brotli raw length-table container (`<III> + renderer + masks + pose`)
  matching the public `unified_brotli` family more closely than RP2.
- Early-snapshot byte screen:
  - `qzs3_pr64_qpose14`: `276610` bytes, SHA
    `dcaedb62a3b3f57fdd3808efbc229fb55ec87aac861cb8f83457459b10d18fd8`
    (`41` bytes below RP2 qpose14).
  - `qzs3_pr64_qp1`: `273057` bytes, SHA
    `043166e87d685acf817513ef28eae170aa045f4ebbe00fb8f7550863718e903b`
    (`9` bytes above RP2 QP1).
  - `torchfp4_pr64_qpose14`: `289042` bytes, SHA
    `b98b9bb84b1709ee9b3bea1bc4dc3ddade1aa761b4871c68ca87e6b43e0926ef`
    (`153` bytes above Torch-FP4 RP2 qpose14).
  - `torchfp4_pr64_qp1`: `285482` bytes, SHA
    `cb6563fe399a99fe61debce2a9ea6896ceb294c85ce56350873f0d7f505061f5`
    (`136` bytes above Torch-FP4 RP2 QP1).
- Dispatch rule for later snapshots: use `qzs3_pr64_qpose14` only if the
  qpose14 scorer basin is healthy and every byte matters; otherwise RP2 remains
  the cleaner default for QP1/Torch-FP4.

## Exact Component Trace Tooling and C-051 Dispatch - 2026-05-01T21:16Z

Evidence grade: `diagnostic_component_trace_pending_cuda_artifact`.
Score claim: `false`.

New tooling:

- Added `experiments/contest_component_trace.py`, a repo-owned diagnostic
  companion to `experiments/contest_auth_eval.py`.
- It does not modify pinned `upstream/evaluate.py`. It mirrors the exact
  DistortionNet loop and preserves the per-sample tensors that upstream sums
  into average PoseNet and SegNet distances.
- Output schema is explicitly non-promotable:
  `score_claim=false`, `evidence_grade=diagnostic_component_trace`.
- It supports:
  - `--archive` mode for exact archive-byte custody plus inflate.
  - `--submission-dir` mode for an existing contest-shaped
    `archive.zip + inflated/` work directory.
  - `--contest-auth-eval-json` cross-check; the trace is accepted only when
    averages and recomputed score match canonical CUDA auth eval.
  - `--baseline-trace-json` for candidate-minus-baseline hard-pair excess
    ranking.

Lightning integration:

- Extended the exact-eval Batch Job path with opt-in `--component-trace`.
- The trace runs only after canonical CUDA auth eval has produced
  `contest_auth_eval.json`.
- The job fails closed unless `component_trace.json` cross-checks against that
  JSON.
- SSH harvest treats `component_trace.json` and `component_trace.log` as
  optional exact-eval artifacts and validates the trace if present.

Verification:

- `py_compile` passed for the trace script, Lightning launcher, Lightning
  batch helper, and focused tests.
- Focused tests passed: `88 passed in 1.18s`.
- Lightning doctor passed after restoring the canonical `lightning-pact` SSH
  alias from the hardened existing `scratch-studio-devbox` settings:
  SSH OK, remote supply-chain scan OK, L40S inventory OK.

Queued diagnostic:

- Job:
  `exact_eval_component_trace_c051_l40s_20260501T2116Z`.
- Archive:
  `experiments/results/lightning_batch/exact_eval_renderer_payload_qpose14_rp2_palias_allowp_c044_t4_20260501T1823Z/archive.zip`.
- Archive bytes/SHA:
  `594047`,
  `dc855b10b69353f1046aeb25d2eba17f43f48039ea0ef2f2d95f5c2a2bef782f`.
- State:
  `.omx/state/component_trace_c051_l40s_batch_jobs_20260501T2116Z.json`.
- Source/artifact manifest:
  `.omx/state/component_trace_c051_20260501T2115Z_manifest.json`.
- Initial status: `Pending`, zero cost at `2026-05-01T21:13:59Z`.

Decision rule after harvest:

- If trace cross-check passes, the top combined/pose/seg pair lists become the
  canonical hard-pair prior for repair-atom selection.
- The next allocator must optimize marginal benefit per charged byte, not raw
  pair error:

```text
include atom a only when
  E[Delta score saved by a | current archive] / charged_bytes(a)
    > 25 / 37,545,489
after confidence and interaction penalties.
```

- Candidate repair atoms should then be split in this order:
  pair -> frame -> class/confusion -> connected component -> boundary band ->
  learned latent correction.

## Component Trace to Atom Planner Bridge - 2026-05-01T21:21Z

Evidence grade: `empirical_tooling_plus_external_refresh`.
Score claim: `false`.

Implementation landed:

- Extended `experiments/alpha_repair_atom_planner.py` so
  `--pair-weights-meta` accepts a CUDA-cross-checked
  `component_trace.json` directly.
- The planner now fails closed on trace input unless:
  - `score_claim=false`;
  - `evidence_grade=diagnostic_component_trace`;
  - `n_samples=600`;
  - all 600 `pair_index` values are present exactly once;
  - `contest_auth_eval_cross_check.all_match=true`.
- For trace-derived inputs, atom ranking uses
  `score_combined_contribution_first_order` before legacy
  pose/seg formula priors, so pair policies are selected by scorer-gradient
  signal per charged byte rather than raw distortion.
- `--pair-signal-top-k` controls how many trace-ranked pairs become the
  hard-pair set; default is `100`.

Verification:

- `py_compile` passed for the planner and tests.
- Focused Alpha atom planner tests passed: `6 passed in 0.47s`.
- Focused component-trace/Lightning tests remain green:
  `88 passed in 1.17s`.
- `git diff --check` passed for the touched trace/planner/Lightning files.

Live status:

- `exact_eval_component_trace_c051_l40s_20260501T2116Z` advanced to
  `Running` on L40S at `2026-05-01T21:19:30Z`; cost was `0.048166666` at the
  `2026-05-01T21:20:00Z` refresh.
- Q-FAITHFUL training remains live on Vast instance `35959478`:
  RTX 4090 at `92%` GPU, process runtime about `02:51:14` as of
  `2026-05-01T21:20:53Z`; no duplicate dispatch was started.

Public floor refresh:

- Official comma leaderboard still lists `qpose14` at `0.32`,
  `unified_brotli` at `0.33`, and `quantizr` at `0.33`.
- PR #63/#64 archive copies were rechecked locally:
  - `qpose14`: `287573` bytes,
    SHA `e012ebeffcc1e1655f4d674d0a779c1bf4cd41cfa82746de2ff6f73692e82a66`.
  - `unified_brotli`: `287165` bytes,
    SHA `7e48da0be75f915d6a4cf76a4679f8c1fbe689f82d69c7559f6ba1c2cb1e981d`.
- Archive anatomy confirms the current public floor is still a
  JointFrameGenerator-family renderer plus extreme packing/pose allocation:
  PR #64 uses one Brotli stream over raw mask/model/pose with a length table
  and velocity-only pose deltas; this remains the packer target for future
  in-basin Q-FAITHFUL checkpoints.

Immediate branch on C-051 trace harvest:

1. Validate `component_trace.json` and harvest custody through the
   state-derived Lightning path.
2. Run Alpha atom planner with the harvested trace as `--pair-weights-meta`.
3. Build the top pair-index repair archives selected by
   score-signal-per-byte, not by legacy Lane-W priors.
4. Dispatch only byte-efficient policies to L40S diagnostics first, then T4
   promotion if exact components remain in basin.

## C-051 Trace Harvest, Alpha Sparse-Repair Collapse, Q Snapshot Dispatch - 2026-05-01T21:39Z

Evidence grades:

- `C-051 component trace`: `A score-grade CUDA diagnostic`, score claim still
  anchored by the prior T4 C-051 run.
- `Alpha sparse repair`: `A-negative scoped forensic` for the measured
  grayscale-plus-AMR1 implementations only.
- `Q-FAITHFUL 21:31Z snapshot`: `empirical byte/custody plus queued CUDA`.

C-051 component trace harvested:

```text
job=exact_eval_component_trace_c051_l40s_20260501T2116Z
archive_bytes=594047
archive_sha256=dc855b10b69353f1046aeb25d2eba17f43f48039ea0ef2f2d95f5c2a2bef782f
gpu=NVIDIA L40S
score=0.9867713723638321
pose=0.00356167
seg=0.00402496
component_trace_sha256=dabf29a76ac390c19a83cf77d2487b7ebc39c280b00fdba8c4984ea148fe98b6
cross_check_all_match=true
state_status_note=SDK name-only status regressed Running->Pending; state-derived artifact validation supersedes telemetry.
```

Trace top combined pair indices, first 30:

```text
127, 75, 109, 133, 514, 125, 517, 522, 177, 111,
45, 584, 521, 516, 130, 289, 90, 210, 592, 512,
179, 325, 69, 549, 519, 356, 64, 150, 478, 513
```

Alpha repair exact evidence:

- All previously queued CRF62/CRF63 sparse-repair candidates completed with
  exact CUDA custody and component-gate failure.
- Best total among them was still bad:
  `exact_eval_alpha_repair_c044_crf62_class2_fix1_t4_20260501T1605Z`,
  score `4.0751452447605425`, `606572` bytes, PoseNet `0.89807248`,
  SegNet `0.00674468`.
- The failure is now classified as geometry/pose-basin incompatibility rather
  than insufficient pair selection. Trace-ranked sparse repair is therefore a
  diagnostic probe only unless a lossy-candidate trace shows localized repair
  opportunity.

Trace-ranked Alpha byte screen:

```text
crf63_trace_top5_lzma:  399375 bytes, sha256=5e3ea813fbc11836842d473a9f4d75721ace5813778addc2913aef563dbaad0a
crf63_trace_top10_lzma: 415024 bytes, sha256=37eb13f64fd2a7b78df4571328e62c91a5ab1d3d295be582778e9f4b19a77050
crf62_trace_top10_lzma: 483992 bytes, sha256=cc669f8206ecd8ff83f3eeaf32ef38ae738c26434ba1d60927b9ec934faa34f9
crf62_trace_top15_lzma: 498749 bytes, sha256=6787c3b1c7e4201f4e87e148f7adb1e002ef27071e9f8b4f485299ec473a2e36
```

No Alpha trace-ranked archive has score evidence yet. Because the prior exact
repair grid was globally out of pose basin, the next Alpha CUDA spend should be
either a lossy-candidate component trace or a pose/geometry regeneration test,
not a broad repair sweep.

Q-FAITHFUL snapshot dispatch:

```text
trainer_instance=Vast 35959478
checkpoint_snapshot=lane_q_faithful_results/postprocess_snapshots/training_state_snapshot_20260501T2131Z.pt
checkpoint_sha256=2d98d481b21cdf4f188e27bf73c133e7b64274ce62f854e345c89b22dcd065b1
postprocess_summary=experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/postprocess_fixed_snapshot_20260501T2131Z_fix1/summary.json
best_byte_candidate=qzs3_pr64_qp1, 272995 bytes, sha256=1a35d4d8899afa47602137efd36a427cdadc6e3a25615c4c54ec51c1ac73374f
safer_eval_candidate=qzs3_pr64_qpose14, 276542 bytes, sha256=70ac01f7446db7766577829a7ec0fc7ab633676408ad8c3ffdd662f2e66a0f3d
queued_job=exact_eval_qfaithful_snapshot_2131_pr64_qpose14_l40s_20260501T2138Z
state=.omx/state/qfaithful_snapshot_2131_pr64_qpose14_l40s_batch_jobs_20260501T2138Z.json
manifest=.omx/state/qfaithful_snapshot_2131_pr64_qpose14_narrow_20260501T2137Z_manifest.json
manifest_sha256=744fd1d08a8a9c331cc0b0b9fe17600cc11435c1737a3a66c22ec187b2f5eb9c
```

The failed full-tree staging attempt tried to transfer about `5.45GB` because
the shared results tree has grown. It was interrupted before completion and
replaced by a narrow `13.1MB` manifest against the already-staged Lightning
repo plus the exact archive artifact.

## Public Floor Trace Harvest and C-051 Gap Decomposition - 2026-05-01T22:09Z

Work landed:

- Harvested public PR63 and PR64 exact CUDA trace jobs through the state-derived
  Lightning path with explicit local mirrors.
- Added deterministic trace comparator:
  `experiments/compare_component_traces.py`.
- Added focused coverage:
  `src/tac/tests/test_compare_component_traces.py`.
- Generated:
  `experiments/results/component_trace_comparison_c051_vs_public_floor_20260501/trace_comparison.json`.

Current public lowest scores were refreshed from `https://comma.ai/leaderboard`:

```text
0.32 qpose14          PR #63
0.33 unified_brotli  PR #64
0.33 quantizr        PR #55
0.37 fp4_mask_gen    PR #62
0.38 selfcomp        PR #56
```

Local exact public-floor traces:

```text
PR63 qpose14:
  score=0.32518843312932477
  bytes=287573
  sha256=e012ebeffcc1e1655f4d674d0a779c1bf4cd41cfa82746de2ff6f73692e82a66
  pose=0.00052823
  seg=0.00061026
  gpu=NVIDIA L40S
  trace_sha256=99c7cb20d20b18ec798dc59c9be075d4ba5ecfdc24ea8d95cbdafb32bfc0c53c

PR64 unified_brotli:
  score=0.33137914516864686
  bytes=287165
  sha256=7e48da0be75f915d6a4cf76a4679f8c1fbe689f82d69c7559f6ba1c2cb1e981d
  pose=0.00062634
  seg=0.00061026
  gpu=NVIDIA L40S
  trace_sha256=81dc854f749f52bcea6ba2bf4426dbb8040405343adb7737beb5f0f905e5fd78
```

C-051 versus PR63:

```text
total_gap=0.6615832118515242
seg_gap=0.3414705165511501
rate_gap=0.20406845679916435
pose_gap=0.11604423850120972
archive_delta_bytes=306474
```

Hard-pair implication:

- `127` and `75` dominate PoseNet excess.
- `133`, `517`, `109`, `522`, and `177` dominate SegNet excess.
- The trace proves that pair-level repair is not enough by itself. To reach
  sub-0.3, the next major score drop must come from public-floor-class global
  mask/renderer geometry and byte packing, with pair atoms used as the
  allocation measure.

Q-FAITHFUL live branch:

- `exact_eval_qfaithful_snapshot_2146_pr64_qp1_l40s_20260501T2201Z` advanced
  from Pending to Running at `2026-05-01T22:08:20Z`.
- This tests whether the 21:46Z Q checkpoint improves when forced onto the
  public PR64 velocity-only pose basis. If it remains out of basin, the trainer
  should be evidence-preserved and redesigned around the one-scalar pose
  manifold before further training spend.

## Mission Charter and Q-FAITHFUL Stop Decision - 2026-05-01T23:16Z

Mission control document:

```text
.omx/research/shannon_floor_mission_charter_20260501_codex.md
deadline=2026-05-03T12:00:00-05:00 America/Chicago
decision_rule=highest expected score reduction per wall-clock minute under exact-evidence and compliance constraints
```

Q-FAITHFUL public-pose-basis exact diagnostic completed:

```text
job=exact_eval_qfaithful_snapshot_2146_pr64_qp1_l40s_20260501T2201Z
artifact_dir=experiments/results/lightning_batch/exact_eval_qfaithful_snapshot_2146_pr64_qp1_l40s_20260501T2201Z
archive_bytes=273103
archive_sha256=a34f493b77e3a2ccba7e059134127e9b3cb6e774a41862143d369fa3f5fc81af
gpu=NVIDIA L40S
n_samples=600
score=22.065520725118258
pose=46.18100739
seg=0.00393906
component_trace_sha256=383a383aa75fce26a5c8962ed8adcfa6128b16fb8e6cf69fe21a9b4dbdbcd7e8
component_trace_cross_check_all_match=true
```

Classification:

- `A-negative scoped forensic` for this measured Q-FAITHFUL profile/snapshot
  and its public PR64 velocity-only/QP1 pose basis export.
- This is not a family kill for JointFrameGenerator or public-floor packing;
  public PR63/PR64 prove that family is in-basin when trained/exported under
  the right geometry.
- The failure is renderer/PoseNet basin mismatch. The public one-scalar pose
  basis does not rescue this checkpoint; it worsens PoseNet relative to the
  already-catastrophic qpose14 snapshot.

Trainer state preserved and stopped:

```text
remote_instance=Vast 35959478
remote_process=71346
last_remote_epoch_seen=799
last_remote_fp4_scorer=16.051653
best_remote_epoch=649
best_remote_fp4_scorer=14.808107
local_mirror=experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/stopped_out_of_basin_20260501T2316Z
renderer_q_faithful_modal_best_fp4.pt_sha256=9717e160e83856175142cca491eef81f939ff66900444865fe2fcc97d9e5ca9d
renderer_q_faithful_modal_best_fp32.pt_sha256=b25249670fc7bd8deaa769a05db65a57641b42a9b0d3c92f9870af4a068428a3
training_state_q_faithful_modal.pt_sha256=871dfe65ddbd11e19ba5cb723e9a2dffb8bc35f3f82927d45a0c9f135dd6924b
renderer_q_faithful_modal_best_meta.json_sha256=88053c9f935fee2a44cfc7af9c6de21ac52bd5c699b63704760bea447ffe4bfa
q_faithful_modal_telemetry.jsonl_sha256=07a21967c318d1974c05646da2b50bf10ed0d760499f0ce74966ed5246ea047d
post_stop_gpu=NVIDIA GeForce RTX 4090, 0% utilization, 1 MiB VRAM
```

Decision:

- Stop spending on this Q profile as parameterized.
- Do not launch another training run until the retraining gate permits it or
  the mission controller records an explicit unblock.
- The next Q design must train directly into the public-floor basin:
  one-scalar pose manifold, public mask geometry, scorer-path uint8/resize,
  then QZS3/QP1/PR64 packing only after the checkpoint is in basin.

Q trace comparison:

```text
comparison_json=experiments/results/component_trace_comparison_qfaithful_2146_qp1_vs_public_floor_20260501/trace_comparison.json
score_delta_vs_pr63=21.74033131907131
pose_delta_vs_pr63=21.417086479367764
seg_delta_vs_pr63=0.3328798187552214
rate_delta_vs_pr63=-0.009634979051677844
archive_delta_bytes_vs_pr63=-14470
top_excess_pairs=72,75,89,71,77,67,73,443,78,76
```

This is global out-of-manifold behavior. The byte win is real but irrelevant
against the PoseNet collapse.

## Public-Floor PVL1 Fast-Chip Dispatch - 2026-05-01T23:36Z

Mission-control update:

- The active path is now public-floor-basin exact archive optimization, not
  incremental OWV3/Alpha repair. Public qpose14/unified_brotli are the measured
  feasible basin and their decoded metadata are optimization constraints.
- Per the fast-chip directive, every non-promotion score-affecting experiment
  must use the fastest available verified CUDA hardware. T4 remains reserved
  for A++ promotion on identical bytes.

Candidate under test:

```text
archive=experiments/results/public_floor_contract_variants_20260501/pr63_pr64_pvl1/archive.zip
archive_bytes=286960
archive_sha256=4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
source=PR63 decoded geometry + PR64 length-table single-blob + PVL1 pose repack
decoded_pose_sha256=cc99e99c28b2ea686439b226ee504ba3a0d82fd8eb8550f4fed05d35ece5dc40
pose_error=zero_vs_source_decoded_pose
masks_sha256=a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb
renderer_sha256=d97849d15859ae013ec983de8c1e2f638e63f3876fef658a8b7781bcfaa16a5f
score_claim=false_until_cuda_auth_eval
```

Expected math if exact decoded components remain identical to PR63:

```text
baseline=public_pr63_qpose14_l40s
baseline_score=0.32518843312932477
baseline_bytes=287573
candidate_bytes=286960
delta_bytes=-613
rate_delta=-0.0004081811801578958
expected_score=0.3247802519491669
```

Queued/running evidence paths:

```text
Lightning L40S diagnostic:
  job=exact_eval_public_floor_pr63_pr64_pvl1_l40s_20260501T2332Z
  state=.omx/state/public_floor_pvl1_l40s_batch_jobs_20260501T2332Z.json
  status_at_2026-05-01T23:33:24Z=Pending cost=0

Lightning T4 promotion:
  job=exact_eval_public_floor_pr63_pr64_pvl1_t4_20260501T2332Z
  state=.omx/state/public_floor_pvl1_t4_batch_jobs_20260501T2332Z.json
  status_at_2026-05-01T23:33:24Z=Pending cost=0

Vast H100 NVL fast diagnostic:
  instance=35985850
  label=public_floor_pvl1_h100diag_20260501
  remote_archive=/workspace/pact/iter_0/archive.zip
  remote_archive_sha256=4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
  remote_archive_bytes=286960
  remote_log_dir=/workspace/pact/experiments/results/vast_h100_public_floor_pvl1_20260501
  status_at_2026-05-01T23:36Z=launched, bootstrap running
```

Immediate decision rule:

- Harvest whichever exact CUDA result lands first.
- If H100 confirms the expected PR63 component parity, keep T4 promotion
  running and use H100 immediately for the next PR63/PR64-basin byte/atom
  variant.
- If H100 breaks parity, classify the exact failure before spending more T4
  time on this container/pose contract.

## Permanent Fragility Fixes And Public-Floor Sweep - 2026-05-01T23:59Z

The PVL1 H100 diagnostic succeeded and the fixed T4 promotion job is running.
The earlier Lightning PVL1 L40S/T4 jobs are classified as stale/non-promotable
because their staged source manifest omitted
`submissions/robust_current/config.env`.

Bug classes closed in code/preflight:

```text
1. Lightning exact-eval source manifests now fail closed unless they include:
   - archive.zip
   - submissions/robust_current/inflate.sh
   - submissions/robust_current/config.env

2. Lightning T4/g4dn exact-eval submits now fail closed unless the job env
   explicitly pins inflate-side Torch. The current fixed T4 job uses:
   INFLATE_TORCH_SPEC=torch==2.5.1+cu124
   UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
   UV_INDEX_STRATEGY=unsafe-best-match

3. `preflight_all()` now runs an executable regression for both checks via
   `check_lightning_exact_eval_manifest_runtime_closure`.

4. `scripts/remote_archive_only_eval.sh` now bootstraps scorer runtime deps
   (`timm`, `einops`, `segmentation-models-pytorch`, `safetensors`, `av`,
   `tqdm`) and cleans multi-GB inflated eval work by default after preserving
   canonical JSON/provenance/report files.
```

Verification:

```text
py_compile:
  scripts/launch_lightning_batch_job.py
  src/tac/deploy/lightning/batch_jobs.py
  src/tac/preflight.py

bash -n:
  scripts/remote_archive_only_eval.sh

pytest:
  src/tac/tests/test_remote_archive_only_eval_script.py
  exact-eval manifest/runtime closure focused tests
  result: 9 passed plus remote-archive-only 5 passed after cleanup patch

preflight:
  check_lightning_exact_eval_manifest_runtime_closure(strict=True) OK
```

Fast-chip exact evidence:

```text
PVL1 H100:
  score=0.3246902093443082
  trace_score=0.3246906061663023
  archive_sha256=4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
  harvest=experiments/results/vast_harvest/public_floor_pvl1_h100diag_fix2_20260501/

Variant H100 sweep:
  harvest=experiments/results/vast_harvest/h100_public_floor_variant_sweep_20260501/
  best_non_pvl1=pr63_pr64_pvr1_top256, score=0.3248499593443082
  no variant beat PVL1
```

Immediate next decision:

- Harvest fixed T4 job `exact_eval_public_floor_pr63_pr64_pvl1_t4_fix1_20260501T2353Z`.
- If it matches H100 and adjudication passes, PVL1 becomes the A++ public-floor
  submission candidate.
- Keep H100 for the next architecture/packer iteration only if it can test a
  candidate with expected score below `0.32469`; otherwise avoid spending on
  variants that are byte-regressive before scorer effects.

## PVL1 A++ Promotion And Metabug Fixes - 2026-05-02T00:15Z

Fixed-manifest Lightning T4 promotion landed:

```text
job=exact_eval_public_floor_pr63_pr64_pvl1_t4_fix1_20260501T2353Z
artifact_dir=experiments/results/lightning_batch/exact_eval_public_floor_pr63_pr64_pvl1_t4_fix1_20260501T2353Z
archive_sha256=4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
archive_bytes=286960
hardware=Tesla T4
n_samples=600
score_recomputed_from_components=0.3247176275031171
score_reported_rounded=0.32
avg_posenet_dist=0.00052391
avg_segnet_dist=0.00061261
component_trace_score=0.3247177087654878
evidence_grade=A++ contest T4
promotion_eligible=true
component_gates=passed
```

This supersedes the previous C-051 frontier and is the current deploy/submit
candidate.

Metabugs fixed during the promotion cycle:

```text
1. scripts/adjudicate_contest_auth_eval.py now interprets
   --regression-threshold as allowed positive score delta versus baseline.
   Deprecated --hard-kill-above keeps the old absolute-score ceiling.

2. scripts/remote_archive_only_eval.sh now mirrors the exact evaluated
   archive.zip into the result directory and writes archive_custody.json before
   scorer launch, preventing overwritten-path score attribution drift.

3. tools/claim_lane_dispatch.py now provides an atomic cross-agent claim helper
   for paid remote dispatches; AGENTS.md requires it and preflight checks for it.

4. submissions/robust_current/unpack_renderer_payload.py now decodes public
   PR64 bare velocity deltas with int32 cumulative sums, matching the public
   unified_brotli implementation rather than uint16 wraparound.
```

Focused verification:

```text
py_compile:
  scripts/adjudicate_contest_auth_eval.py
  scripts/launch_lightning_batch_job.py
  src/tac/deploy/lightning/batch_jobs.py
  src/tac/preflight.py
  tools/claim_lane_dispatch.py
  submissions/robust_current/unpack_renderer_payload.py
  experiments/build_renderer_packed_payload_archive.py

pytest:
  src/tac/tests/test_remote_auth_eval_hardening.py::test_adjudicator_regression_threshold_is_delta_vs_baseline
  src/tac/tests/test_claim_lane_dispatch.py
  src/tac/tests/test_remote_archive_only_eval_script.py
  src/tac/tests/test_renderer_packed_payload.py::test_public_pr64_velocity_delta_decode_uses_int32_cumsum_not_uint16_wrap
  result: 10 passed

preflight:
  check_dispatch_claim_helper_present(strict=True) OK
  check_remote_archive_only_eval_custody_closure(strict=True) OK
  check_lightning_exact_eval_manifest_runtime_closure(strict=True) OK
```

Next dispatch rule: do not spend exact-eval time on any public-floor packer
variant unless it either beats `286960` archive bytes with identical decoded
members, improves components, or implements the PR67/QZS3/QP1 byte opportunity.

## QZS3/QP1 Frontier And Dispatch-Claim Hardening - 2026-05-02T01:08Z

New exact frontier:

```text
claim=C-053
job=exact_eval_public_floor_qzs3_qp1_t4_20260502T0036Z
archive_sha256=c5260473c26c4d4537d99d4a6a18b8ff0d9d1a901f6db17cd2208559e1010362
archive_bytes=276296
score=0.3243472585872431
hardware=Tesla T4
samples=600
promotion_eligible=true
```

First line-search T4 promotion:

```text
claim=C-054
h100_score=0.32114254758178584
h100_archive_sha256=8c9000f67eb21f366299fe033e3e6031ab63992e8067758600e43d0091c9a9fa
h100_archive_bytes=276427
t4_job=exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z
t4_score=0.3218613619571356
t4_pose=0.00058608
t4_seg=0.00061244
t4_evidence=A++ contest T4
t4_status_note=SDK Running->Pending telemetry regression; state-derived artifact mirror complete and locally validated
```

Active H100 continuation:

```text
instance=35985850
session=pact_ls_qzs3_continue_0100
remote_dir=/workspace/pact/experiments/results/line_search_qzs3_qp1_fixedslice_continue_r3_20260502T0100Z
best_logged_objective_at_2026-05-02T01:08Z=0.253897808
latest_checkpoint=archive.accepted_latest.zip
```

Bug/metabug class closed:

```text
class=paid-dispatch without active cross-agent claim
fix=scripts/launch_lightning_batch_job.py now requires an active
    .omx/state/active_lane_dispatch_claims.md row for non-dry-run
    Studio-backed exact-eval, component-response, and component-sensitivity
    submissions, unless --allow-missing-dispatch-claim-reason is supplied.
preflight=check_dispatch_claim_helper_present now checks both helper and
    Lightning launcher guard strings.
verification=py_compile plus focused pytest:
    src/tac/tests/test_claim_lane_dispatch.py
    src/tac/tests/test_lightning_batch_jobs.py::test_non_dry_run_studio_submit_requires_active_dispatch_claim
    result=6 passed
```

## Pose Line-Search Frontier And Learned Proposal Extension - 2026-05-02T01:23Z

C-054 is now the active exact frontier:

```text
archive=experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z/archive.zip
archive_sha256=8c9000f67eb21f366299fe033e3e6031ab63992e8067758600e43d0091c9a9fa
archive_bytes=276427
score=0.3218613619571356
hardware=Tesla T4
samples=600
promotion_eligible=true
score_delta_vs_C053=-0.002485896630107509
```

The stronger r8 continuation remains a queued/running T4 promotion candidate:

```text
job=exact_eval_line_search_qzs3_qp1_r8_t4_20260502T0110Z
h100_score=0.3152653422017416
archive_sha256=c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1
archive_bytes=276426
status=running as of 2026-05-02T01:18Z
```

The active H100 continuation from the r8 checkpoint reached:

```text
remote_session=pact_ls_qzs3_continue_r13_0112
radius13_pass1_obj=0.253779157
radius13_pass2_obj=0.253772163
checkpoint_bytes=276423
```

Proposal-search hardening landed locally:

```text
files=experiments/line_search_pose_refinement.py, src/tac/tests/test_line_search_pose.py
new_modes=--delta-sets, --gradient-delta-sets, --gradient-backtrack-deltas
meaning=sparse/asymmetric and differentiable gradient-guided integer pose proposals
acceptance=same rounded archive objective; no score claim without exact CUDA
verification=py_compile; 5 targeted tests passed; git diff --check passed
```

This directly addresses the non-arbitrary search requirement: use the
differentiable scorer-aligned path to propose deltas, but charge the resulting
QP1/Brotli pose bytes and validate complete archives through exact CUDA.

## QZS3/QP1 r8 T4 Frontier And r13/Gradient Continuation - 2026-05-02T01:34Z

The r8 continuation T4 promotion landed and supersedes C-054:

```text
claim=C-056
job=exact_eval_line_search_qzs3_qp1_r8_t4_20260502T0110Z
archive=experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_r8_t4_20260502T0110Z/archive.zip
archive_sha256=c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1
archive_bytes=276426
score=0.3159064496962538
pose=0.00049846
seg=0.00061244
hardware=Tesla T4
samples=600
promotion_eligible=true
score_delta_vs_C054=-0.0059549122608818
```

The r13 scalar continuation is still diagnostic only until its own T4 job
lands:

```text
h100_archive=experiments/results/vast_harvest/archive_eval_line_search_qzs3_qp1_fixedslice_continue_r13_20260502T0123Z/archive.zip
h100_archive_sha256=d3f3300531886d9dcb3553baffdd201567e3adaf7b746a7f405b15ad6c23b148
h100_archive_bytes=276423
h100_score=0.31514356926681697
h100_pose=0.00049102
h100_seg=0.00061012
t4_job=exact_eval_line_search_qzs3_qp1_r13_t4_20260502T0128Z
t4_status=pending at 2026-05-02T01:30Z
```

The H100 successor run is now the anisotropic/directional proposal stage:

```text
instance=35985850
session=pact_ls_qzs3_gradient_r13_0129
remote_dir=/workspace/pact/experiments/results/line_search_qzs3_qp1_fixedslice_gradient_20260502T0129Z
proposal=--gradient-delta-sets "1,2,3,5,8,13;1,2,3,5" --gradient-backtrack-deltas "1,2"
score_claim=false until exact CUDA archive eval
```

Operational hardening during this slice: the H100 had only `4.3G` free because
completed auth-eval directories retained reproducible `uv_project_env`
directories. Those env dirs were removed after archive/JSON/provenance custody
was harvested, restoring the instance to `59G` free without deleting scientific
artifacts. The bug class is now hardened in the canonical wrapper:

```text
file=scripts/remote_archive_only_eval.sh
fix=remove per-job UV_PROJECT_ENVIRONMENT after successful custody copy when path is under LOG_DIR
test=src/tac/tests/test_remote_archive_only_eval_script.py
verification=bash -n scripts/remote_archive_only_eval.sh; pytest result 5 passed
```

A second launch-wrapper bug class was caught and corrected live: an accidental
nohup relaunch briefly overlapped with the claimed H100 gradient run. The lower
priority duplicate process was terminated, the duplicate claim was cancelled,
and AGENTS now requires single-GPU process/output-dir preflight plus first-byte
logging before any warm Vast/H100 ad-hoc launch.

The next pose-manifold implementation is ready if the active gradient pass
flattens:

```text
file=experiments/line_search_pose_refinement.py
new_cli=--basis-delta-sets "dct:1,2,3;pair_window:1,2"
new_cli=--basis-modes "0,1,2,3,5,8,13,21"
new_cli=--basis-pair-indices "<absolute contest pair ids>"
interpretation=vector atoms over QP1 col0, including smooth DCT motion modes and hard-pair windows
score_claim=false until exact CUDA archive eval
verification=py_compile plus focused pytest result 5 passed
```

## Closed-Loop Feedback Protocol And Component Trace Guard - 2026-05-02T02:25Z

The positive-feedback framing is now formalized as a compiler-style control
loop, not just a discussion pattern:

```text
representation -> prediction -> quantization -> hyperprior/arithmetic -> pack
        ^               ^                 ^                         |
        |               |                 |                         v
profile feedback <- component traces <- exact CUDA eval <- archive custody
```

Operational meaning:

- Each stage emits typed profile facts: bytes/SHA/member layout, legality,
  runtime, scorer component deltas, hard-pair opportunity density, selected and
  rejected atoms, active subspace basis, and Lagrangian/water-fill constraints.
- Exact CUDA archive eval is still the only promotion truth. Component traces,
  public-archive anatomy, Hessian/Fisher maps, learned selectors, ego-motion/
  camera priors, and low-dimensional subspaces are proposal/profile feedback.
- Positive feedback is allowed and desired: a representation that makes poses
  cheaper, a packer that makes repair atoms worthwhile, a pose update that
  shifts SegNet hard pairs, or a public-submission anatomy clue that narrows the
  active manifold. Negative feedback is equally valuable when it exposes an
  antagonism before a T4 promotion run.

Hardening landed so the feedback loop cannot ingest runtime-drifted traces:

```text
file=src/tac/preflight.py
new_check=check_contest_component_trace_runtime_parity
strict=true in preflight_all
guards=parity FFMPEG_BIN, explicit scale color options, job-local UV_PROJECT_ENVIRONMENT,
  UV_LINK_MODE=copy, component_trace_runtime_env.json, diagnostic/non-promotable status,
  contest_auth_eval cross-check support
verification=py_compile; focused pytest 16 passed; git diff --check clean
```

## QZS3/QP1 basis T4 confirmation and PR67 pair feedback - 2026-05-02T02:27Z

The anisotropic basis archive from the H100 diagnostic path landed on T4 and
supersedes the prior r13/r8 promotion frontier:

```text
claim=C-057
job=exact_eval_line_search_qzs3_qp1_basis_r13_t4_20260502T0200Z
archive=experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_basis_r13_t4_20260502T0200Z/archive.zip
archive_sha256=63e6213ae154b5b5ce164829c15e675ad6d7819a9bdb1e8c9b2f099374fa7009
archive_bytes=276423
score=0.3157562807844823
pose=0.00049637
seg=0.00061244
hardware=Tesla T4
samples=600
promotion_eligible=true
score_delta_vs_r13_t4=-0.000010644510485824377
```

The same job emitted a component trace and cross-checked it against the exact
auth JSON:

```text
component_trace=experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_basis_r13_t4_20260502T0200Z/component_trace.json
component_trace_cross_check=all_match
trace_score=0.31575601237045303
trace_score_abs_diff_vs_auth=2.684140292807946e-07
```

PR67 comparison feedback is now concrete:

```text
compare_json=experiments/results/top_submission_reverse_engineering_20260502T0206Z/basis_vs_pr67_component_delta.json
archive_delta_bytes_vs_pr67=-141
score_delta_rate_exact=-0.00009388611239022349
score_delta_pose_exact=+0.00000983656708697378
score_delta_seg_exact=+0.000029669621047417882
score_delta_total=-0.000054379924255831824
```

Interpretation:

- C-057 beats public PR67 in the H100 comparison because it saves 141 bytes.
- PR67 is still slightly better on geometry, by about `0.0000395` score.
- The next positive-feedback loop should target that geometry gap only, not
  re-open broad arbitrary scalar sweeps. Top excess pairs from the PR67
  comparison include `105, 87, 546, 24, 295, 313, 155, 197, 46, 590` by
  combined excess and `87, 295, 257, 309, 105, 63, 313, 72, 36, 65` by
  PoseNet excess. These are active-subspace basis inputs, not score claims.

Next dispatch should be a PR67-informed low-dimensional pose/search pass on
the warm H100: pair-window and local DCT/jerk basis modes over the excess-pair
set, accepted only by complete archive objective, followed by H100 exact
diagnostic and T4 confirmation if the archive beats C-057 by enough to matter.

## Shannon/Tao/Dykstra council and active-subspace hardening - 2026-05-02T02:50Z

Read-only council `019de686-2985-7fc2-b4ab-ecc337dd2342` independently
verified the same control decision:

```text
active_frontier=C-057
score=0.3157562807844823
archive_sha256=63e6213ae154b5b5ce164829c15e675ad6d7819a9bdb1e8c9b2f099374fa7009
archive_bytes=276423
hardware=Tesla T4
top_next_lane=PR67-informed low-dimensional QP1 active-subspace search
evidence_rule=H100/L40S/A100 are compiler-profile feedback; T4/equivalent is promotion truth
```

The mathematical framing is now explicitly compiler-style profile-guided
rate-distortion optimization:

```text
S(A) = 100 * D_seg(A) + sqrt(10 * D_pose(A)) + 25 * bytes(A) / 37,545,489

accept atom a iff
  Delta S_archive(a | selected_so_far) < 0
after charged bytes, complete archive rebuild, and exact CUDA evaluation.
```

At C-057, holding distortion fixed would need roughly `23.7k` archive bytes
removed to break `0.30`. Holding bytes and SegNet fixed would need PoseNet
distance around `0.000299` versus the current `0.00049637`. Therefore the
highest-EV local direction is not arbitrary repacking; it is pose-geometry
water-filling in a low-dimensional active subspace while preserving the QZS3/
QP1 byte basin.

Trace-recut artifacts:

```text
cross_device_recut=experiments/results/top_submission_reverse_engineering_20260502T0206Z/c057_vs_pr67_recut_top120.json
status=diagnostic_only_cross_device_noise_for_fine_deltas

same_device_recut=experiments/results/top_submission_reverse_engineering_20260502T0206Z/basis_h100_vs_pr67_recut_top120.json
status=preferred PR67 pair-feedback artifact
score_delta_total_vs_pr67=-5.4379924255831824e-05
```

Active H100 continuation:

```text
claim=2026-05-02T02:36:12Z lane_line_search_pose_refinement
remote=35985850:pact_ls_qzs3_pr67_active_subspace_c057_fix2_20260502T0240Z
output_dir=experiments/results/line_search_qzs3_qp1_pr67_active_subspace_c057_fix2_20260502T0240Z
status=training/diagnostic
first_checkpoint=archive.accepted_latest.zip
first_checkpoint_bytes=276422
first_checkpoint_obj=0.253770695
score_claim=false
```

Permanent bug classes closed during this tranche:

- `experiments/line_search_pose_refinement.py` now preflights scorer-runtime
  imports (`timm`, `einops`, `segmentation_models_pytorch`, `safetensors`)
  before importing `upstream/modules.py`. Missing dependencies are
  `failed_scorer_runtime_deps`, not lane evidence.
- The same tool now preflights `nvidia.dali` before GT-video target extraction
  through `upstream/frame_utils.DaliVideoDataset`. Missing DALI is
  `failed_dali_runtime_deps`, not lane evidence.
- Refined archive metadata now records the actual output archive path,
  archive bytes, archive SHA-256, pose payload SHA, source archive path, and
  source archive SHA. This prevents stale source metadata from masquerading as
  checkpoint custody.

Verification:

```text
py_compile=experiments/line_search_pose_refinement.py passed
pytest=src/tac/tests/test_line_search_pose.py 30 passed, 1 skipped
diff_check=touched files clean
AGENTS.md=durable scorer/DALI runtime preflight rule appended
```

## C-059 Weighted-Pair Pose Promotion Wave - 2026-05-02T05:16Z

The PR67-informed weighted-pair top32 pose continuation produced a real H100
diagnostic improvement over C-059 and is now queued for T4/equivalent
promotion.

H100 diagnostic artifact:

```text
artifact=experiments/results/vast_harvest/archive_eval_ls_c059_weighted_pairs_top32_h100_20260502/contest_auth_eval.json
archive=experiments/results/vast_harvest/archive_eval_ls_c059_weighted_pairs_top32_h100_20260502/archive.zip
score_h100=0.3151364334691563
archive_bytes=276423
archive_sha256=877fc5ac13e9fbd5c4158a9c7fa9dec3354057522b086004a4a28c6822456fe8
posenet=0.00049092
segnet=0.00061012
hardware=NVIDIA H100 NVL
samples=600
```

This is not an A++ claim because `gpu_t4_match=false`, but it is strong enough
to justify immediate T4 spend. The local line-search metadata originally
reported SHA `cda95e70440e9ef295985a042fda2d74715ef6a0e665a1c37871cddd051cd908`;
that metadata is superseded by the harvested archive hash and
`contest_auth_eval.json` provenance above.

T4 promotion dispatch:

```text
job=exact_eval_ls_c059_weighted_pairs_top32_t4_20260502T0509Z
state=.omx/state/ls_c059_weighted_pairs_top32_t4_batch_jobs_20260502T0509Z.json
manifest=.omx/state/exact_eval_ls_c059_weighted_pairs_top32_t4_20260502T0509Z_manifest.json
manifest_sha256=987e77f725206fb1cee1faa9d523c29b8a1e0551316906602576db151df98162
machine=g4dn.2xlarge
expected_archive_sha256=877fc5ac13e9fbd5c4158a9c7fa9dec3354057522b086004a4a28c6822456fe8
expected_archive_size_bytes=276423
inflate_torch=torch==2.5.1+cu124
status=Pending at submit, zero cost
```

Two permanent guard outcomes came from this dispatch:

- `scripts/lightning_repro_workspace.py` now excludes hidden and macOS
  resource-fork source paths even when staging from `--source .`; the exact-eval
  submitter correctly refused the earlier manifest containing `.DS_Store`.
- The T4 exact-eval submitter correctly refused unpinned g4dn/T4 inflate
  dependencies until `INFLATE_TORCH_SPEC=torch==2.5.1+cu124` and the PyTorch
  cu124 index were recorded.

Next action is state-derived harvest as soon as the T4 job becomes terminal.
If T4 lands inside the predicted band and component gates pass, it supersedes
C-059. If it misses, preserve component trace and feed the pair deltas into the
next water-fill selector rather than repeating scalar pose churn.

## C-059 Top32 T4 Miss, Top48 Water-Fill, and Lossless Repack - 2026-05-02T05:38Z

The weighted-pair top32 archive received valid T4/equivalent exact evidence but
did not move the frontier:

```text
job=exact_eval_ls_c059_weighted_pairs_top32_t4_20260502T0509Z
artifact_dir=experiments/results/lightning_batch/exact_eval_ls_c059_weighted_pairs_top32_t4_20260502T0509Z
archive_sha256=877fc5ac13e9fbd5c4158a9c7fa9dec3354057522b086004a4a28c6822456fe8
archive_bytes=276423
score_t4=0.3157569904685481
posenet=0.00049638
segnet=0.00061244
gpu=Tesla T4
n_samples=600
score_delta_vs_c059=+0.00005145968406577994
component_trace_cross_check=true
adjudication_promotion_eligible=true
lane_status=OUT_OF_PREDICTED_BAND
```

Interpretation: the H100 diagnostic improvement was not stable at T4 precision/
runtime. This is valid A++ custody but not a new frontier. Use the component
trace as calibration signal; do not repeat the same top32 T4 promotion.

The next H100 continuation is already running under an explicit claim:

```text
claim=c059_pose_manifold_top48_h100
remote=35985850:H100 NVL
session=pact_c059_pose_top48_r2_0529_fix1
output=experiments/results/line_search_c059_pose_atoms_top48_r2_20260502T0529Z
source_archive=experiments/results/line_search_c059_weighted_pairs_top32_20260502T0410Z/archive.zip
policy=c_059_pose_atoms_top048
pair_window_radius=2
basis_delta_sets=pair_window:1,2,3,5,8;pair_window:1,2,3;pair_window:1
status=running; exact H100 archive eval chained after search completes
score_claim=false until archive CUDA eval lands
```

This is the higher-EV path: broader scorer-weighted pose-manifold water-fill
from an existing accepted archive, not another tiny scalar continuation.

A separate near-free byte-only candidate was also built and queued for T4:

```text
tool=experiments/repack_single_payload_brotli.py
source=C-059 archive.zip
output=experiments/results/lossless_repack_c059_brotli_q11m2w18_20260502/archive.zip
manifest=experiments/results/lossless_repack_c059_brotli_q11m2w18_20260502/manifest.json
brotli_params=quality=11,mode=font,lgwin=18,lgblock=0
archive_sha256=83615afd130afa08e972e4a02476612397bffea53327caf3591891f8317aa52d
archive_bytes=276223
archive_delta_bytes=-124
formula_only_rate_score_delta=-0.00008256651018714925
payload_roundtrip=identical
t4_job=exact_eval_lossless_repack_c059_brotli_t4_20260502T0537Z
state=.omx/state/lossless_repack_c059_t4_batch_jobs_20260502T0537Z.json
status=Pending at submit
```

This is not allowed to distract the main effort: it is a deterministic lossless
repack and one T4 confirmation only. The same transform correctly refused the
top32 archive because that archive member is not a Brotli-compressed payload;
top32 remains a pose-search archive, not a Brotli-container repack target.

## PR67 Direct-Slice Hybrid Screen - 2026-05-02T05:47Z

The next higher-EV action is a full archive-grammar/mask-basin screen rather
than another scalar pose shave. A deterministic PR67 fixed-slice archive was
built locally using the charged public PR67 mask slice and the C-059 QZS3 model
and QP1 pose bytes:

```text
output=experiments/results/pr67_direct_publicmask_c059_modelpose_20260502/archive.zip
manifest=experiments/results/pr67_direct_publicmask_c059_modelpose_20260502/build_manifest.json
archive_sha256=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
archive_bytes=276214
payload_bytes=276114
segments=pr67_mask_br:219472,c059_model_br:55965,c059_pose_br:677
source_c059_sha256=cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab
source_pr67_sha256=a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765
score_claim=false
```

Rationale:

- The sub-0.30 gap is too large for current pose scalar polishing alone.
- The mask/model public-floor anatomy still exposes a larger stream-level
  opportunity than 1-byte/100-byte rate shaves.
- The candidate is exact-parseable through the hardened
  `unpack_renderer_payload.py` PR67 branch and charges all source bytes inside
  `archive.zip`.

H100 diagnostic dispatch:

```text
claim=lane_qzs3_qp1_packer
remote=35995649:H100 SXM
session=pact_eval_pr67_direct_publicmask_c059_modelpose
log_dir=experiments/results/archive_eval_pr67_direct_publicmask_c059_modelpose_h100_20260502
source_sha_preflight=passed for unpack_renderer_payload.py, inflate.sh,
    inflate_renderer.py, remote_archive_only_eval.sh, contest_auth_eval.py
status=running
```

This is diagnostic only. If it lands materially below C-059 on H100, promote
identical bytes on T4/equivalent. If it misses, preserve the component trace as
mask-basin evidence and pivot strict CMG1/component-boundary work toward
learned atom selection rather than copying public archive mechanics.

Result update:

```text
harvest=experiments/results/vast_harvest/archive_eval_pr67_direct_publicmask_c059_modelpose_h100_20260502
archive_sha256=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
archive_bytes=276214
gpu=NVIDIA H100 80GB HBM3
gpu_t4_match=false
n_samples=600
score_recomputed_from_components=0.3629464459892586
avg_posenet_dist=0.00102119
avg_segnet_dist=0.00077973
classification=A-negative H100 diagnostic
t4_promotion=false
```

Interpretation: direct public PR67 mask slicing is antagonistic with the C-059
model/pose slices. This is not a public-basin family kill; it specifically
rules out wholesale PR67-mask + C-059-model/pose splicing as a promotion path.
The next mask-basin experiment needs learned/scorer-aware charged atoms,
strict mask grammar, or full public-basin co-optimization rather than a direct
stream transplant.

## C-063 Lossless Repack Frontier - 2026-05-02T05:53Z

The C-059 single-member Brotli repack landed as exact T4 A++ frontier evidence:

```text
artifact_dir=experiments/results/lightning_batch/exact_eval_lossless_repack_c059_brotli_t4_20260502T0537Z
archive_sha256=83615afd130afa08e972e4a02476612397bffea53327caf3591891f8317aa52d
archive_bytes=276223
score_recomputed_from_components=0.3156230307844823
avg_posenet_dist=0.00049637
avg_segnet_dist=0.00061244
gpu=Tesla T4
gpu_t4_match=true
n_samples=600
promotion_eligible=true
score_delta_vs_c059_t4=-0.00008250000000004087
archive_delta_vs_c059=-124
classification=A++ contest T4 frontier
```

This is pure charged rate improvement: the transform preserves the decompressed
payload exactly and changes only the Brotli container parameters of C-059's
single `p` member. It supersedes C-059 as the exact internal frontier. It is
also a warning: remaining sub-0.30 movement cannot come from lossless repack
polish alone; the required gap is still tens of kilobytes or a real
PoseNet/SegNet component improvement.

## Q-FAITHFUL Existing Snapshot H100 Exact Wave - 2026-05-02T06:00Z

Halley implemented a standalone non-training snapshot loop scaffold:

```text
script=scripts/q_faithful_snapshot_loop.py
tests=src/tac/tests/test_q_faithful_snapshot_loop.py
verification=py_compile passed; pytest 5 passed in 0.39s
score_claim=false
role=checkpoint scanner/export/repack/H100 exact-screen scaffold
```

No new Q-FAITHFUL training was dispatched because the Lane 12/Alpha retraining
gate remains in force and no live stable Q-FAITHFUL checkpoint was found.
Instead, the idle H100 instance `35995649` is screening existing already-built
Q-FAITHFUL snapshot archives through the hardened archive-only CUDA wrapper:

```text
claim=q_faithful_snapshot_exact_wave
remote=35995649:H100 SXM
session=pact_qfaithful_snapshot_h100_wave_0558
driver=.omx/state/qfaithful_snapshot_h100_wave_20260502T055846Z_driver.sh
log_root=experiments/results/qfaithful_snapshot_h100_wave_20260502T055846Z
candidate_1=qfaithful_2131_pr64_qp1 bytes=272995 sha=1a35d4d8899afa47602137efd36a427cdadc6e3a25615c4c54ec51c1ac73374f
candidate_2=qfaithful_2146_rp2_qp1 bytes=272986 sha=d90a937da2127086f28b66f7df58a027c8c565488eb8e765e468808361602128
candidate_3=qfaithful_2131_pr64_qpose14 bytes=276542 sha=70ac01f7446db7766577829a7ec0fc7ab633676408ad8c3ffdd662f2e66a0f3d
source_sha_preflight=enabled
status=running first candidate as of 2026-05-02T06:00Z
```

This is the correct aggressive use of existing work: it tests a larger learned
architecture/packing family without launching forbidden retraining and without
spending T4 until H100 exact evidence shows a real candidate.

Result update:

```text
qfaithful_2131_pr64_qp1 score=20.958219620496074 PoseNet=41.15432739
qfaithful_2146_rp2_qp1 score=22.146784937370022 PoseNet=46.54520035
qfaithful_2131_pr64_qpose14 score=20.957476577718626 PoseNet=41.14173126
classification=A-negative H100 diagnostic
common_failure=half-frame masks without charged zoom/warp geometry
t4_promotion=false
```

Halley's follow-up guard makes this failure class fail closed:
`scripts/q_faithful_snapshot_loop.py` now records a
`qfaithful_snapshot_runtime_contract_v2` and refuses `--eval-mode run` for
half-frame snapshots unless the charged zoom/warp geometry is preserved by the
archive/repack/runtime contract. Q-FAITHFUL remains high-EV only after that
geometry is charged and closed, or after a full-frame snapshot is exported.

## Public PR65/PR67 External Replay And CMG Constraint - 2026-05-02T06:55Z

External leaderboard replay is now locally measured as reverse-engineering
signal:

```text
public_trace_dir=experiments/results/vast_harvest/public_leaderboard_external_trace_20260502T0630Z
pr67_archive_sha256=a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765
pr67_archive_bytes=276564
pr67_h100_score=0.3630208381950889
pr67_pose=0.00101833
pr67_seg=0.00077956
pr65_faithful_status=failed_torch_boolean_indexing_in_public_inflate
pr65_compat_trace_dir=experiments/results/vast_harvest/public_pr65_torch25_compat_trace_20260502T0640Z
pr65_compat_archive_sha256=b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68
pr65_compat_archive_bytes=284425
pr65_compat_h100_score=0.3599586412428538
pr65_compat_pose=0.00087405
pr65_compat_seg=0.00077081
classification=external reverse-engineering signal only
```

The PR65 compatibility replay changes one public-inflate PyTorch boolean
assignment so it can run under the current stack. It is useful for anatomy and
component geometry, but not exact public-submission faithfulness evidence.

Strict CMG planning also landed a decoded PR67 mask contract:

```text
cmg_dir=experiments/results/charged_mask_grammar_atoms_pr67_decoded_20260502_codex
decoded_mask_sha256=13f81851ae206f6142cd3348db5537eda4ea832665052824816f4d1bf432de0c
decoded_shape=600x384x512
public_pr67_mask_stream_bytes=219472
best_naive_probe=raw_uint8_bz2_9 bytes=340315
row_rle_lzma9_bytes=385896
score_claim=false
```

Decision: direct public-mask splicing and naive class-mask grammars are both
below EV threshold now. The high-EV mask path is a learned/predictive/foveated
charged grammar or a Q-FAITHFUL full-frame/zoom-warp-closed export, not more
raw class-RLE or whole-stream transplants.

## C-063 vs Public PR67 Atom Ledger - 2026-05-02T06:52Z

The C-063 component trace was compared against the existing PR67 public-basin
component trace and compiled into a water-fill atom ledger:

```text
trace_comparison=experiments/results/component_trace_comparison_c063_vs_public_pr67_20260502/trace_comparison.json
atom_ledger=experiments/results/frontier_atom_ledger_c063_pr67_20260502/atom_ledger.json
candidate=C-063
reference=public_pr67_trace
score_delta_candidate_minus_reference=0.00043105090413975145
rate_delta_candidate_minus_reference=-0.00022705790301466577
pose_delta_candidate_minus_reference=0.00039701390094226685
seg_delta_candidate_minus_reference=0.00026109490621214343
ledger_active_frontier=C-063
```

The derived ledger ranks pose pair atoms as positive EV under the public PR67
prior, with top pair indices `153,156,87,37,111,46,81,382,280,184`. It ranks
mask/postprocess atoms as non-positive EV under the crude per-pair byte model.
This is not score evidence; it is the next proposal table for a charged
pair-pose residual or low-dimensional pose-manifold experiment. T4 should not
be spent until those atoms become a concrete archive and beat C-063 on H100 by
a material margin.

## Same-Hardware C-063/PR67 Trace And Q-FAITHFUL Contract Result - 2026-05-02T07:25Z

Allocator-grade trace comparison was refreshed on the same H100 NVL class:

```text
c063_trace=experiments/results/vast_harvest/c063_same_h100_component_trace_20260502T0700Z/component_trace.json
pr67_trace=experiments/results/vast_harvest/pr67_same_h100_component_trace_20260502T0712Z/component_trace.json
comparison=experiments/results/component_trace_comparison_c063_h100nvl_vs_pr67_h100nvl_20260502/trace_comparison.json
atom_ledger=experiments/results/frontier_atom_ledger_c063_h100nvl_pr67_h100nvl_20260502/atom_ledger.json
c063_h100_score=0.31500175622241344
pr67_h100_score=0.3151919111815026
delta_c063_minus_pr67=-0.0001894896514510469
allocator_use_allowed=true
```

The refreshed ledger finds 48 positive pose atoms and no positive mask or
postprocess atoms under the current byte model. This supports pose-atom
sidecar experiments only as micro-EV work unless a stackable low-dimensional
pose grammar can charge materially fewer bytes than the current QP1/PVL
grammar.

The Q-FAITHFUL geometry-closed direct candidate was exact-screened on H100:

```text
harvest_dir=experiments/results/vast_harvest/qfaithful_geometry_closed_h100_20260502T0700Z
archive_sha256=f64dcb3d12db394efa9b0e0f924bb62b6b24f096d66baf9ed83447077d4f9b61
archive_bytes=274257
contest_auth_score=22.147631187370024
component_trace_score=22.147632116385466
classification=A-negative diagnostic contract failure
```

Failure mode: `zoom_scalars.bin` was preserved and charged, but the renderer
did not expose `use_zoom_flow`, so inflate duplicated half-frame masks instead
of consuming the zoom geometry. This retires only the measured
"add charged zoom to non-zoom QZS3 JointFrameGenerator" implementation.
Q-FAITHFUL remains high-EV only if the exporter produces a real
`use_zoom_flow=True` runtime contract or a full-frame architecture.
