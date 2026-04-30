# Component Sensitivity Producer Grand Council Review - 2026-04-30

Agent: Codex

Scope reviewed:

- `src/tac/component_sensitivity_artifact.py`
- `experiments/build_component_sensitivity_manifest.py`
- `experiments/profile_hessian_per_weight.py`
- `src/tac/sensitivity_map.py`
- `upstream/evaluate.py`, `upstream/modules.py`, `upstream/frame_utils.py`
- `src/tac/scorer.py`
- Focused tests for component sensitivity artifacts, manifest assembly,
  sensitivity maps, Hessian/Fisher profiling, contest auth eval, and adjudication.

This document is a producer-design review and operating contract. It is not a
score ledger. No sensitivity, Fisher, Hessian, finite-difference, byte-only,
proxy, CPU, or MPS artifact below is promotion, ranking, kill, or
Shannon-floor evidence. CUDA exact auth eval on exact archive bytes remains the
only score truth:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

## Executive Verdict

The current code has a useful custody shell, not a promotion-grade component
sensitivity producer.

What is sound:

- `component_sensitivity_v1` validation is fail-closed for promotion: CUDA
  device fields, no debug/proxy/random/smoke markers, required input custody,
  sample plan, PoseNet/SegNet/combined map entries, stability entries, response
  curves, exact eval custody, and `n_samples == 600`.
- `build_component_sensitivity_manifest.py` is a deterministic assembler. It
  rejects non-CUDA exact eval JSON, wrong sample counts, missing tensor payloads,
  and response-curve files without holdout error.
- `sensitivity_map.py` gives OWV3 a small per-Conv2d-channel map contract with
  CUDA authority checks, shape checks, finite/nonnegative checks, and CV-distance
  support.
- `profile_hessian_per_weight.py` is useful for empirical CUDA Fisher-style
  renderer-weight ranking and now rejects MPS, supports protected Conv2d
  coverage, remaps encoded grayscale masks to class ids, and records useful
  metadata.

What is not yet sound:

- No producer currently emits complete `component_sensitivity_v1` artifacts end
  to end.
- `profile_hessian_per_weight.py` computes squared gradients of a differentiable
  surrogate. It is not an exact Hessian and not an exact SegNet score gradient.
- The existing manifest schema validates metadata presence, not the scientific
  truth of maps. A producer can still attach a tensor with the right shape but
  the wrong scorer target unless finite-difference and stability gates are
  enforced before assembly.
- `build_component_sensitivity_manifest.py` reads torch payloads to discover
  tensor metadata. That is acceptable for trusted local/staged artifacts, but
  remote artifacts must be hash-custodied before they are loaded.

Grand Council decision: implement a new strict producer, preferably
`experiments/profile_component_sensitivity.py`, or extend
`profile_hessian_per_weight.py` only if it can emit the complete contract below.
Until then, current Fisher/channel maps are empirical producer inputs only.

## Required Producer Algorithm

### 1. Anchor And Custody

The producer starts from one exact anchor. The anchor is either an evaluated
archive or a deterministic renderer/mask/pose/video bundle that can build one.

Required preflight:

- Record `archive.zip` bytes and SHA-256 when archive-derived.
- Run or cite exact CUDA `contest_auth_eval.py` on the anchor archive if the
  artifact will be used for promotion decisions.
- Record exact `contest_auth_eval.json`, `provenance.json`, eval command,
  hardware, upstream hash/manifest, source/staged-tree manifest, and logs.
- Hash every score-affecting input: renderer/checkpoint, masks, poses, video,
  pair weights, upstream scorer tree, video-name list, and any decoded/inflated
  frame materialization.
- Fail closed unless every promotion-capable device field is `cuda` or
  `cuda:<index>`.

If the anchor lacks exact archive custody, the producer output may be
`diagnostic_cuda` but not promotion-capable.

### 2. Deterministic Perturbation Basis

Before computing any map, materialize `perturbation_basis_v1`.

Required basis fields:

- `basis_id`: SHA-256 over canonical basis JSON plus input hashes.
- `atom_family`: one of `renderer_channel`, `renderer_block`, `pixel_block`,
  `mask_geometry`, `pose`, or `archive_action`.
- Deterministic atom ordering: module path, tensor name, channel/block/spatial
  index, pair index, and action id as applicable.
- Normalization: exact atom norm, epsilon units, clamp/domain rule, quantize or
  uint8 round rule, scorer-resize rule, and sign convention.
- Pair plan: full 600 pair universe, calibration pairs, holdout pairs, split
  seed, split hash, pair weights, and exclusion reasons.
- Roundtrip policy: whether perturbations pass through renderer output,
  384->874->uint8->384 scorer roundtrip, or exact inflate/raw frame material.

Default split for corpus-wide maps is 480 calibration pairs and 120 holdout
pairs with a stored deterministic split. Top-k hard-pair profiling may be a
diagnostic acceleration path, but it cannot replace the full split for a
promotion-capable component map unless the artifact records the narrowed scope
and promotion blockers.

### 3. Component Readouts

For each atom, compute and store three component maps:

- `posenet`: CUDA PoseNet pose MSE response using the official PoseNet value
  path from `upstream/modules.py`.
- `segnet`: CUDA SegNet argmax-disagreement response using the official SegNet
  value path from `upstream/modules.py`.
- `combined`: component recomputation from measured components, not a separate
  proxy loss.

The combined finite-difference delta must be computed as:

```text
DeltaCombined(eps)
  = 100 * (seg_eps - seg_0)
  + sqrt(10 * pose_eps) - sqrt(10 * pose_0)
```

Do not replace this with `d_score/d_pose * DeltaPose` except in explicitly
tagged local-linear diagnostics. The square-root PoseNet term is global over
mean pose distortion, so pairwise or atomwise additive deltas are only
approximations until validated.

A differentiable CE/entropy/softmax SegNet proxy may be stored as auxiliary
data. It must not be the `segnet` component map unless the response-curve gate
below validates it against official argmax disagreement on CUDA holdout atoms.

### 4. Map Estimation

For renderer-weight maps, the producer may use Fisher/diagonal Gauss-Newton as
the first estimator:

```text
I_c(w) = mean_over_pairs pair_weight[p] * (d L_c,p / d w)^2
```

where `c` is `posenet`, `segnet_proxy`, or `combined_proxy`. The output must be
named Fisher or diagonal Gauss-Newton proxy, not Hessian, unless actual
second-order estimation is performed.

For the promotion-capable component map, each atom receives a nonnegative
expected damage value derived from response curves:

```text
s_c(atom) = max(0, robust_slope_or_curvature_c(atom))
```

The producer must record how `robust_slope_or_curvature` is defined. For
zero-mean quantization-noise models, use symmetric curvature from central
differences. For one-sided codec actions, use the measured one-sided action
delta and mark it as directional. Mixing those two meanings in one tensor is
forbidden unless each atom records its `sensitivity_kind`.

### 5. Stability And Calibration

Compute maps independently on calibration and holdout splits where feasible.
At minimum, compute calibration maps and validate them on a deterministic
holdout atom subset with finite differences.

Required stability fields per component:

- normalized L1 CV distance between calibration and holdout maps,
- Spearman and Pearson rank correlation where defined,
- top-k overlap for at least top 1%, 5%, and 10% of atoms,
- per-layer/per-family summaries for renderer atoms,
- count of zero, nonfinite, negative, clipped, excluded, and proxy-only atoms.

The producer must pre-register pass/fail thresholds in the artifact. Missing
thresholds make the artifact diagnostic only.

## Finite-Difference Response Curve Gates

Every promotion-capable component map requires linked response curves. Response
curves are not decorative metadata; they are the gate that distinguishes
measured scorer sensitivity from shaped tensors.

### Curve Construction

For each component and deterministic validation subset:

- Evaluate `eps = 0`.
- Evaluate symmetric central points `-eps`, `+eps`.
- Evaluate at least one larger symmetric magnitude, normally `-2eps`, `+2eps`.
- Include the actual codec action point when the atom represents a discrete
  archive action.
- Record raw component values, component deltas, predicted deltas, wall-clock
  runtime, pair ids, atom ids, and all exceptions.

The epsilon ladder must be in the atom's natural units and recorded in
`perturbation_basis_v1`. It must not be inferred from logs.

### Required Gates

A response-curve packet passes only if all gates below pass for each component
used in promotion:

1. `finite_values`: every baseline, perturbed component, prediction, and error
   statistic is finite.
2. `zero_repro`: the `eps=0` recomputation matches the stored baseline from the
   same local CUDA scorer path within the artifact's recorded tolerance.
3. `component_separation`: PoseNet, SegNet, and combined curves are stored
   separately. A combined-only curve cannot validate component-specific maps.
4. `segnet_argmax_readout`: any SegNet proxy map must include holdout curves
   using official argmax disagreement. CE-only SegNet validation is
   non-promotable.
5. `calibration_holdout_rank`: predicted sensitivity and observed holdout damage
   must have positive rank correlation and pass the artifact's pre-registered
   Spearman/top-k thresholds.
6. `top_k_stability`: top-k overlap across calibration and holdout must pass
   the artifact's thresholds for the atom subset that the downstream codec will
   protect or spend bytes on.
7. `asymmetry_flagging`: sign-asymmetric, nonmonotone, or curvature-changing
   curves must be flagged per atom. If flagged atoms are used for automatic
   byte allocation, the artifact is diagnostic unless the allocation explicitly
   handles directional/nonconvex response.
8. `holdout_error_bound`: response-curve JSON must include `holdout_error` or
   `max_holdout_error`, the threshold used, and `passed=true`.
9. `exact_archive_followup`: any archive built from the map must run exact CUDA
   auth eval with component gates before it can promote, rank, or retire
   anything.

Recommended first-pass thresholds, to be tightened per lane after data lands:

- Spearman rank correlation on holdout: `>= 0.30` minimum, `>= 0.50` target.
- Top-decile overlap: `>= 0.50` minimum.
- Component-specific sign accuracy for directional action deltas: `>= 0.70`.
- Normalized holdout error for protected/top-k atoms: `<= 0.35`.
- Calibration/holdout normalized L1 CV distance: `<= 0.35` for map-wide use.

If any threshold is missed, preserve the artifact with `promotion_eligible=false`
and a mathematical `promotion_blockers` entry explaining the failed gate.

## Mathematical Caveats

### Fisher Is Not Hessian

The current profiler accumulates squared gradients. That is empirical Fisher or
diagonal Gauss-Newton style evidence. It is not a Hessian unless the producer
uses second differences, Hessian-vector products, or another reviewed
second-order estimator.

Consequences:

- Fisher can rank likely fragile channels.
- Fisher cannot by itself prove second-order distortion under quantization.
- Fisher cannot validate SegNet argmax discontinuities without finite
  differences.
- A bad Fisher-driven archive result retires that measured configuration only,
  not OWV3, Fisher, or sensitivity-aware allocation as a family.

### SegNet Is Discontinuous

Official SegNet distortion is argmax disagreement. Differentiable CE against
GT argmax labels is an optimization proxy, not the official component. A
producer may use CE to choose candidate atoms, but must validate against
official argmax response curves before writing a promotable SegNet map.

### PoseNet Square Root Is Global

The contest score uses `sqrt(10 * mean_pose_dist)`, not a per-pair linear pose
term. Local derivatives can be useful near an operating point, but additive
PoseNet deltas across atoms are approximations. Stacks need their own exact
archive eval.

### Local CUDA Component Curves Are Still Diagnostic

Even CUDA finite differences through the scorer path are not exact archive
scores. They are promotion-candidate inputs. A codec archive constructed from a
map must be evaluated as its own exact archive on CUDA.

### Byte Accounting Is Part Of The Math

Sensitivity without charged side-information bytes is incomplete. Any map,
basis, codebook, threshold table, protected-channel list, or learned allocator
state needed by inflate must be inside `archive.zip` or fixed contest code.
The optimizer's objective is:

```text
100 * seg_dist + sqrt(10 * pose_dist)
  + 25 * archive_bytes / 37,545,489
```

## Metadata Requirements

A complete producer directory must contain:

- `component_sensitivity_v1.json`: assembled by
  `experiments/build_component_sensitivity_manifest.py`, not hand-edited for
  promotion.
- `perturbation_basis_v1.json`: canonical atom basis and basis hash.
- `posenet_map.pt`, `segnet_map.pt`, `combined_map.pt`: tensor payloads with
  dtype, shape, numel, finite/nonnegative stats, atom count, component target,
  sensitivity kind, normalization, and file SHA-256.
- `posenet_response_curve.json`, `segnet_response_curve.json`,
  `combined_response_curve.json`: curve points, gate thresholds, gate results,
  holdout errors, rank metrics, top-k overlaps, asymmetry/nonmonotonic flags,
  and exact/proxy labels.
- `stability.json`: CV distance, rank correlations, top-k overlaps, per-layer
  summaries, excluded atom reasons, and threshold pass/fail.
- `command.txt`: exact CLI and environment variables.
- `environment.json`: Python, PyTorch, CUDA, DALI/AV, ffmpeg, package versions,
  deterministic flags, and seed settings.
- `hardware.json`: GPU model, driver, CUDA runtime, visible device ids,
  `nvidia-smi` output, and whether hardware is contest-equivalent.
- `source_manifest.json`: repo file count, bytes, SHA-256, git status, source
  versus artifact role.
- `upstream_manifest.json`: scorer tree hash, model weight hashes, and
  evaluator hash.
- `archive_manifest.json` when archive-derived: deterministic member order,
  member bytes, compressed bytes, CRC, permissions, timestamps, SHA-256, and
  side-info classification.
- `contest_auth_eval.json` and `provenance.json` when exact archive evidence is
  attached.

Promotion JSON must explicitly include:

- `device: "cuda"` or `cuda:<index>`.
- `promotion_eligible: true` only when every gate passes.
- `evidence_grade: "A"` or `"A++"` only when exact archive custody supports it.
- Empty `promotion_blockers` for promotion artifacts.
- Nonempty `promotion_blockers` with mathematical explanations for diagnostic,
  partial, proxy, CPU/MPS, smoke, or failed-gate artifacts.

## Code-Specific Findings

### `src/tac/component_sensitivity_artifact.py`

Strengths:

- Strong promotion-vs-diagnostic split.
- Recursive rejection of non-CUDA device fields and debug/proxy/smoke/random
  markers.
- Required PoseNet, SegNet, and combined map slots.
- Required sample plan, stability, response curves, and exact eval custody.
- Deterministic JSON emission and directory custody hashing.

Risks:

- It validates metadata and finite numeric stability trees, not tensor values.
  The producer must validate tensor finiteness, nonnegativity, component target,
  and atom count before assembly.
- It requires response-curve count and holdout error but does not enforce
  scientific thresholds. Producer-side `gate_spec` and `passed` fields must be
  required by policy until the validator grows these checks.
- `evidence_grade` can conflate exact archive score custody with sensitivity
  map authority. Treat component maps as promotion-candidate inputs even when
  attached to an A/A++ anchor archive.

### `experiments/build_component_sensitivity_manifest.py`

Strengths:

- Good deterministic assembler and CUDA exact-eval precheck.
- Fails on wrong sample count and non-CUDA eval provenance.
- Extracts tensor metadata and response-curve holdout error.

Risks:

- It is intentionally not a producer.
- It loads `.pt` files with `torch.load`; use only after artifact custody and
  trusted-run provenance are established.
- It does not verify response-curve gate pass/fail, exact argmax SegNet
  validation, or calibration/holdout thresholds.

### `experiments/profile_hessian_per_weight.py`

Strengths:

- Useful empirical CUDA Fisher profiler.
- MPS forbidden, CUDA default, CPU explicitly advisory.
- Protected Conv2d inclusion supports OWV3 coverage.
- Encoded mask luma is remapped to class ids.

Risks:

- The name "Hessian" is mathematically too strong for the current estimator.
- SegNet term is CE proxy, not official argmax disagreement.
- Combined loss uses local differentiable component estimates and does not
  produce PoseNet-only, SegNet-only, and combined artifact maps.
- No calibration/holdout split, finite-difference response curves, or exact
  archive linkage are emitted.

### `src/tac/sensitivity_map.py`

Strengths:

- Good compact per-channel contract for codec consumers.
- Shape, nonfinite, negative, and missing Conv2d checks are useful.
- `sensitivity_cv_distance` is the right primitive for stability reporting.

Risks:

- The map is a consumer contract, not a scientific artifact.
- It does not encode component target, atom basis, epsilon ladder, response
  curves, or exact eval custody by itself.

### Scorer/Eval Path

`upstream/evaluate.py` is the final score path. It computes mean PoseNet and
SegNet distortions over 600 samples, then:

```text
score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37,545,489
```

`upstream/modules.py` defines the component behavior:

- PoseNet preprocesses two-frame RGB sequences into YUV6 and computes MSE over
  the first half of the pose head.
- SegNet uses the last frame and computes argmax disagreement.

`src/tac/scorer.py` is appropriate for differentiable diagnostics, but
`compute_proxy_score()` and CE-style objectives remain proxy paths unless they
are tied back to exact local argmax curves and then to exact archive eval.

## Required Test Plan

### Unit Tests

- `component_sensitivity_artifact` rejects:
  - CPU/MPS nested device fields,
  - proxy/debug/random/smoke markers,
  - missing PoseNet/SegNet/combined maps,
  - missing response curves,
  - missing exact eval custody,
  - nonempty promotion blockers in promotion mode,
  - empty/missing mathematical blockers in diagnostic mode.
- Manifest assembler rejects:
  - non-CUDA `contest_auth_eval.json`,
  - `n_samples != 600`,
  - response curves with no holdout error,
  - map files with no tensors,
  - mismatched custody bytes/SHA,
  - response-curve JSON missing `gate_spec` or `passed=true` once producer
    output schema is tightened.
- Sensitivity map consumer rejects:
  - missing Conv2d keys under promotion/missing-policy-error,
  - NaN/Inf/negative values,
  - wrong channel counts,
  - non-CUDA authoritative metadata.
- Perturbation basis builder rejects:
  - nondeterministic atom order,
  - duplicate atom ids,
  - missing normalization,
  - missing epsilon ladder,
  - missing pair split hash,
  - uncharged side-info atoms.

### Producer Tests

- Synthetic differentiable toy model where the analytic gradient, Fisher, and
  finite-difference curvature agree within a tight tolerance.
- Synthetic discontinuous SegNet-like argmax target where CE proxy ranking
  intentionally disagrees; producer must mark proxy-only/non-promotable unless
  argmax validation passes.
- Calibration/holdout split determinism: same inputs and seed produce identical
  pair lists, split hash, basis id, and manifest bytes.
- Response-curve gate fixtures for:
  - clean monotone curve,
  - nonfinite curve,
  - asymmetric curve,
  - nonmonotone curve,
  - low rank correlation,
  - failed top-k overlap,
  - missing exact argmax SegNet readout.
- Tensor payload tests: map atom count and tensor shape match
  `perturbation_basis_v1`.

### Integration Tests

- Build a small trusted fixture producer directory, run
  `build_component_sensitivity_manifest.py`, and validate deterministic output.
- Run the profiler in `--allow-diagnostic-cpu` or CPU fixture mode only for
  smoke, assert output has `promotion_eligible=false` and blockers.
- On CUDA CI/runner, run a tiny CUDA producer smoke with real scorer modules and
  assert all promotion-precondition fields are present, but do not make score
  claims.
- Build an archive from a sensitivity map and require exact CUDA
  `contest_auth_eval.py` plus `scripts/adjudicate_contest_auth_eval.py` with
  PoseNet/SegNet component gates before any result can update rankings.

### Verification Commands

For doc/schema changes:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_sensitivity_map.py \
  src/tac/tests/test_profile_hessian_per_weight.py \
  -q
git diff --check
```

For producer code changes, add:

```bash
.venv/bin/python -m py_compile \
  experiments/profile_component_sensitivity.py \
  experiments/build_component_sensitivity_manifest.py \
  src/tac/component_sensitivity_artifact.py \
  src/tac/sensitivity_map.py
```

For any score-affecting archive:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <candidate archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence dir>
```

Then adjudicate only from JSON:

```bash
.venv/bin/python scripts/adjudicate_contest_auth_eval.py \
  --contest-json <evidence dir>/contest_auth_eval.json \
  --archive <candidate archive.zip> \
  --provenance <candidate provenance.json> \
  --baseline-score <baseline recomputed score> \
  --predicted-band <low> <high> \
  --regression-threshold <scoped threshold> \
  --max-posenet-dist <gate> \
  --max-segnet-dist <gate>
```

## Implementation Order

1. Add `perturbation_basis_v1` writer and tests.
2. Add `profile_component_sensitivity.py` that emits PoseNet, SegNet, and
   combined maps plus response curves and stability JSON.
3. Add response-curve gate parsing to the manifest assembler, or keep the gate
   in the producer and require `gate_spec` plus `passed=true` fields by test.
4. Rename or relabel Hessian outputs as Fisher/diagonal Gauss-Newton unless a
   real Hessian/HVP path is added.
5. Wire OWV3/NWCS consumers to require `component_sensitivity_v1` custody for
   promotion, while allowing legacy `tac_score_sensitivity_map_v1` only as
   empirical or smoke input.
6. Run exact archive CUDA eval only after byte accounting is plausible and the
   sensitivity artifact passes its own gates.

## Final Gate

A component sensitivity artifact can authorize building a candidate archive. It
cannot authorize a score claim. The candidate archive must still pass exact CUDA
auth eval on its own exact bytes, with PoseNet and SegNet component gates, before
promotion, ranking, retirement, or paper claims.
