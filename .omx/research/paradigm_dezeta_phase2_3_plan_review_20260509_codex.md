# PARADIGM-dezeta Phase 2/3 Plan Review - 2026-05-09

Generated: `2026-05-09T09:08:47Z`
Owner: `codex`
Scope: adversarial design and implementation review only
Score claim: `false`
Promotion eligible: `false`
Ready for exact eval dispatch: `false`
Remote dispatch performed: `false`
Files intentionally not touched: A5 q-bit wire files and current runtime packet files

## Inputs inspected

- `AGENTS.md`
- `.omx/state/paradigm_dezeta_phase2_3_plan_20260509.md`
- `.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md`
- `.omx/research/deltaepszeta_phase1_targets_and_smoke_20260509_codex.md`
- `.omx/research/phase_4_optimal_stack_predicted_band_20260508.md`
- `.omx/research/phase4_optimal_stack_design_20260508_claude.md`
- `.omx/research/council_phase_a_complete_extreme_rigor_review_20260508.md`
- `.omx/research/phase_a_pareto_solver_integration_20260508_codex.md`
- `reports/phase_a_pareto_20260508.md`
- `.omx/research/phase_a1_best_proxy_checkpoint_selection_20260509_codex.md`
- `.omx/research/phase_a1_best_proxy_modal_dispatch_20260509_codex.md`
- `.omx/research/phase_a1_best_proxy_modal_harvest_20260509_codex.md`
- `.omx/research/true_strategy_cpu_gpu_loader_drift_optimal_floor_20260508_xhigh.md`
- `.omx/research/loader_drift_xray_supersession_20260508_codex.md`
- `.omx/research/grand_council_a1_post_cpu_anchor_strategy_20260509.md`
- `.omx/research/grand_council_fields_medal_theoretical_floor_20260509.md`
- Source council memo under `.claude/.../feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md`
- Current implementation/help surfaces:
  - `tools/build_deltaepszeta_training_targets.py --help`
  - `tools/run_deltaepszeta_training.py --help`
  - `experiments/build_deltaepszeta_pr106_candidate.py --help`
  - `src/tac/codec_pipeline_deltaepszeta_callback.py`
  - `src/tac/joint_scorer_aware_training.py`
  - `src/tac/learnable_entropy_model.py`
  - `src/tac/self_compress_full_renderer.py`

## Executive verdict

The Phase 2/3 plan is a valid high-upside research direction, but it is not an
implementation plan that can be dispatched as written. Keep the family live.
Retire only measured configurations. Convert the plan into a staged compiler
contract: score-geometry surrogate -> byte-closed payload -> runtime-consumed
archive -> paired exact CPU/CUDA custody.

The current state of evidence does not support the plan's floor language as a
score claim. A1 now has exact CUDA evidence at `0.2263520234784395`, `178262`
bytes, archive SHA `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`,
but the best-proxy refire duplicated the existing packet. The paired CPU anchor
is public-axis positive at `0.19284757743677347`; the CUDA axis is not a new
frontier. This validates the plumbing and the device-gap model for this
HNeRV-class packet. It does not validate sub-0.155, sub-0.140, or a new
architecture class.

The most important correction: Phase 2/3 should not start with another blind
GPU run. The first implementation work should make the existing dezeta
scaffolding emit a byte-different, runtime-consumed candidate packet with
fail-closed provenance. Only then does exact eval spend have a score-lowering
path instead of proxy leakage.

## Key adversarial findings

1. The `.omx/state` plan conflates prediction, conjecture, and action. Keep
   `S_floor = 0.132 +/- 0.014` as a planning posterior only. It lacks a
   reproducible likelihood model and is not calibrated against a byte-closed
   Phase 2/3 archive.

2. The plan's T7 claim, "CPU-only 50 LOC in `tac.codec_pipeline_deltaepszeta_callback.py`,"
   is under-specified. Fisher-Rao distance is meaningful on SegNet softmax
   distributions, not on codec byte telemetry by itself. If T7 uses stored
   logits, it is a profile prior. If it loads scorers, it is compress-time
   scorer work and must live in the joint scorer-aware training path, not in a
   pure codec callback. Either way it cannot produce score movement until the
   trained payload is consumed by `inflate.sh`.

3. The source council memo repeats stale CUDA/CPU mechanism language. The
   loader-drift supersession explicitly corrected the FastViT attention/TF32
   story: PoseNet is FastViT-T12 with RepMixer/conv-style blocks, and T4 is
   not an Ampere TF32 device. Treat DALI/PyAV input-byte drift and CPU/CUDA
   forward-kernel drift as open competing hypotheses.

4. The Phase 4 predicted band already moved down in confidence after Phase A.
   Phase A posterior says several byte/proxy lanes underperformed council
   priors. Current solver target for sub-0.17 under CPU-floor assumptions is
   `137103` bytes, requiring `41041` bytes saved versus the PR101 brotli
   anchor. No current dezeta artifact has produced that as a runtime-consumed
   archive.

5. Existing dezeta code is more than memo-only, but still not shippable.
   `JointScorerAwareLoss.forward` is implemented for scorer tensors, the
   entropy model has CPU smoke encode/decode plus `LEPR` build/parse, and
   zeta has conv-swap/export/load parsing. The training driver is still a
   CPU state-dict H0 loop, and the candidate builder is fail-closed planning
   only. There is no integrated candidate compiler that turns a trained
   dezeta payload into the scored runtime path.

6. T10 auxiliary-scorer work is useful only as compress-time gradient
   densification. It must never replace the contest scorer for evidence, and
   no auxiliary scorer bits or weights can be hidden outside the archive if
   they affect inflate output. The first T10 artifact should be a surrogate
   calibration report, not a dispatch.

7. T8 full Sinkhorn W2 over frame pixels is likely too expensive naively.
   A feasible first version should operate on SegNet logits/tiles, boundary
   bands, sliced-W2 projections, or a downsampled cost grid, with a manifest
   recording approximation error and runtime.

8. Kill-as-last-resort is still the right policy. A1 best-proxy duplicate,
   A2/A3 weight proxies, A4 hand-parametric ChARM, and A6 selfcomp hyperprior
   retire measured configs only. Reactivation criteria remain score-domain
   boundary validation, learned/co-designed priors, runtime-consumed packet
   changes, or exact component evidence.

## Recommended Phase 2/3 decomposition

### Phase 2A - geometry and surrogate profile layer

Goal: replace MSE/H0-only training signals with score-geometry priors that are
explicitly non-scoring until they produce a changed archive.

Required typed interfaces:

- `SegGeometryProfile`
  - `archive_sha256`
  - `runtime_tree_sha256`
  - `source_device_axis`
  - `logit_source`
  - `distance_name`: `fisher_rao`, `hellinger`, `sliced_w2`, or `boundary_margin`
  - `clamp_epsilon`
  - `reduction`
  - `score_claim=false`
  - `promotion_eligible=false`

- `PoseBasisProfile`
  - `basis_name`
  - `fit_archive_sha256`
  - `pose_dims_used=6`
  - `explained_variance`
  - `component_axis`: `contest_cuda`, `contest_cpu`, or advisory tag
  - `score_claim=false`

Implementation note: put reusable math in `src/tac/analysis/` or a small
`src/tac/paradigm_delta_epsilon_zeta/` package, then thread it into
`JointScorerAwareLoss`. Do not put scorer loads into the codec byte callback.

### Phase 2B - byte-closed candidate bridge

Goal: use the existing target table and training driver to create a
byte-different payload and a candidate archive that contains that payload.

Existing verified CLI surfaces:

```bash
.venv/bin/python tools/build_deltaepszeta_training_targets.py \
  --shannon-json <one-or-more paths/globs> \
  --output-dir <results-dir> \
  --started-at-utc <iso8601-utc>

.venv/bin/python tools/run_deltaepszeta_training.py \
  --state-dict <checkpoint.pt> \
  --targets-json <targets.json> \
  --n-epochs <n> \
  --steps-per-epoch <n> \
  --learning-rate <float> \
  --lambda-init <float> \
  --lambda-step <float> \
  --rate-budget-bits <float> \
  --log-dir <results-dir>/run \
  --run-label <label> \
  --seed <int>

.venv/bin/python experiments/build_deltaepszeta_pr106_candidate.py \
  --targets-json <targets.json> \
  --renderer-input <baseline payload> \
  --trained-payload <new payload> \
  --candidate-archive <archive.zip> \
  --top-k <n> \
  --output-dir <results-dir> \
  --created-at-utc <iso8601-utc>
```

The candidate builder already fails closed on missing payload/archive, unsafe
ZIP members, duplicate members, unchanged payload SHA, and archive not
containing the trained payload SHA. The missing implementation task is the
packet compiler that consumes `final_state_dict.pt` or `ZETA`/`LEPR` sections
and produces an `archive.zip` whose `inflate.sh` path actually reads those
bytes.

### Phase 2C - score-aware warm-start run

Goal: train from the A1 checkpoint or PR106 entropy-target substrate with
SegNet/PoseNet gradients at compress time only.

Required gates before any future dispatch:

- exact input checkpoint bytes/SHA and selected tensor list;
- score-axis tag of all priors;
- `eval_roundtrip=True`;
- EMA shadow specified as the only export source;
- no MPS or CPU fallback in CUDA training path;
- no scorer imports in inflate/runtime files;
- byte-different candidate archive built and locally unpacked through the
   same runtime loader that exact eval will use.

### Phase 3A - epsilon learned entropy prior

Goal: prove `LEPR + encoded payload` beats a static baseline on the same
quantized tensor set after prior overhead.

Break-even manifest:

- baseline payload bytes/SHA;
- `LEPR` section bytes/SHA and config;
- encoded payload bytes/SHA;
- total `LEPR + payload + container` bytes;
- exact old/new byte delta;
- roundtrip tensor equality or explicitly bounded quantization error;
- `score_claim=false` until a runtime archive consumes it.

The `MAX_LEPR_BYTES=5000` cap is sane as a default. Raising it should require
an Occam break-even row showing the prior saves more bytes than it costs.

### Phase 3B - zeta full-renderer self-compression

Goal: prove a `ZETA` section can reconstruct a usable renderer without scorer
or sidecar dependencies.

Required conformance vectors:

- tiny renderer with one protected FiLM-like layer and one compressible conv;
- export bytes/SHA;
- parsed layer names, shapes, kept channel indices, scales, bit depths;
- zero-prune negative vector;
- malformed header/truncated body negative vectors;
- proof that protected layers remain protected;
- load path returns tensors that can be inserted into the parent renderer.

Do not dispatch zeta until there is an archive builder that consumes the parsed
weights in the scored inflate path. Export/load parsing alone is not runtime
closure.

### Phase 3C - joint composition

Goal: merge delta scorer-aware training, epsilon prior, and zeta renderer
compression into one candidate compiler.

The compiler target should emit:

- `candidate.archive_zip`
- `candidate.inflate_runtime_manifest.runtime_tree_sha256`
- `candidate.changed_sections[]` with old/new bytes and SHA-256
- `candidate.score_affecting_payload_changed=true`
- `candidate.target_modes=["contest_exact_eval"]`
- `candidate.deployment_target="t4_contest_runtime"`
- `candidate.no_scorer_at_inflate=true`
- `candidate.ready_for_exact_eval_dispatch=false` until compliance,
  preflight, dispatch claim, and review gates are complete.

## Exact gates

Gate 0 - evidence tagging:
Every Phase 2/3 artifact must set `score_claim=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`, and
`ready_for_exact_eval_dispatch=false` unless it is an exact eval artifact.

Gate 1 - mathematical grounding:
Every new constant or threshold must cite one of: contest formula derivative,
Fisher/Hessian/Jacobian map, entropy/MDL calculation, component-response data,
or an ablation manifest. Otherwise it is `planning_only`.

Gate 2 - byte-change proof:
The trained payload must differ from the input payload by SHA-256, and the
candidate archive must contain the trained payload or parsed section bytes.

Gate 3 - runtime consumption:
`inflate.sh` must consume the changed bytes in the scored path. A target table,
checkpoint, `LEPR` blob, or `ZETA` blob outside the archive is not score
evidence.

Gate 4 - scorer boundary:
Scorers may be loaded at compress/training time only. No scorer or auxiliary
scorer load is allowed at inflate time. Auxiliary scorer outputs are priors,
not evidence.

Gate 5 - archive compliance:
Before any future eval dispatch, run the repo's visible preflight/compliance
surfaces and preserve the resulting JSON/log artifacts. ZIP member safety,
member uniqueness, local/central header parity, hidden/resource file exclusion,
and packed-payload multiplicity all apply.

Gate 6 - dispatch coordination:
Before any future remote training/eval/GPU job, claim the lane through the
dispatch-claim helper and close the claim with a terminal row after completion.
This review intentionally did not dispatch or claim.

Gate 7 - dual-axis custody:
For a scored archive, preserve both axes for the same archive/runtime when the
goal is submission/frontier status:

- `[contest-CUDA]`: exact archive/runtime CUDA custody, full sample count,
  component recomputation, hardware, logs, runtime tree SHA.
- `[contest-CPU]`: Linux x86_64 exact archive/runtime custody for the public
  leaderboard axis.

Do not transfer the HNeRV CPU/CUDA gap to a Ballé, selfcomp, W2, or auxiliary
scorer architecture without paired anchors.

Gate 8 - adversarial result review:
Every positive or negative result must record custody, recomputation,
classification, engineering/math/scorer/compliance review, and reactivation
criteria. Bad configs update trust regions; they do not kill the method family.

## First implementation tasks

1. Build a `SegGeometryProfile` module and tests for Fisher-Rao/Hellinger on
   categorical logits. Use clamped probabilities, explicit reduction, gradient
   finite checks, and no score-claim fields.

2. Thread the geometry profile into `JointScorerAwareLoss` as an opt-in
   compress-time loss component. Keep default behavior backward-compatible.
   Do not put scorer loading in `CodecPipelineAwareTrainingCallback`.

3. Add a small dezeta packet compiler that turns a trained state dict or
   `ZETA`/`LEPR` sections into a byte-closed `archive.zip`, then immediately
   runs `experiments/build_deltaepszeta_pr106_candidate.py` against that
   archive to prove changed payload custody.

4. Add `LEPR` break-even smoke on the real A1/PR106 tensor set: compare
   baseline bytes against `LEPR + encoded payload + container`, record old/new
   SHA-256, and keep the row non-dispatchable unless the archive consumes it.

5. Add `ZETA` conformance vectors for export/load and FiLM protection before
   any full-renderer QAT spend. The loader must produce tensors insertable into
   the parent renderer; parsing metadata is not enough.

6. Replace global CPU/CUDA ratio assumptions in dezeta planners with per-class
   calibration fields. New architecture classes must start with wide drift
   bands until paired exact artifacts exist.

7. Update the local state plan or add a supersession note only after the packet
   compiler exists. Until then, this review is the durable control ledger for
   Phase 2/3 execution.

## Blockers

- No runtime-consumed dezeta archive compiler exists yet.
- The current training driver optimizes a state-dict H0 proxy with MSE-vs-
  reference by default; that is implementation progress, not score-domain
  evidence.
- T7/T8/T10 are currently design concepts, not calibrated typed artifacts.
- Loader/kernel drift attribution is still diagnostic-only and incomplete for
  new architecture classes.
- Phase 2/3 predicted floors are not backed by a byte-closed exact CUDA or
  paired CPU/CUDA archive.

## Reactivation policy

- A1 `lr=2e-6`, `kl=0.2`, `pixel_l1=0.01`, `40x8`, best-proxy selection:
  do not relaunch as-is; reactivate only if it emits a byte-different archive
  with score-domain or SegNet-boundary validation.
- A2/A3 weight-domain proxies: measured-config retired; reactivate with
  score-domain Hessian/Fisher/Jacobian or byte-domain compression-hardness
  priors.
- A4 ChARM hand-parametric real-PR101 probe: measured-config retired; reactivate
  with learned/co-designed prior plus runtime archive consumption.
- A6 selfcomp hyperprior compose: measured-config negative; reactivate on a
  better substrate, learned hyper-decoder, cross-tensor grouping, or
  compose-after-lossy-coarsening evidence.
- Phase 2/3 family: explicitly live, pending byte-closed compiler, typed
  geometry priors, and exact custody gates.
