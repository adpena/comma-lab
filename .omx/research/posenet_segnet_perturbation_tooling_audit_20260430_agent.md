# PoseNet/SegNet Perturbation Tooling Audit - 2026-04-30

Agent: Codex
Scope: scorer/eval, Fisher/sensitivity, perturbation, Alpha-Geo, component
sensitivity, and Hessian profiling tooling in `/Users/adpena/Projects/pact`.

This is an audit and roadmap document, not a score ledger. No score, ranking,
promotion, or retirement claim below is stronger than the evidence class named
with it.

## Non-Negotiable Evidence Boundary

The only promotion, kill, or ranking truth remains exact CUDA auth evaluation
on exact archive bytes through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

Prefer:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <candidate archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence dir>
```

CPU, MPS, local proxy scorer, byte-only, smoke, gradient, Fisher, Hessian,
finite-difference, mask-geometry, renderer-only, and sidecar-only evidence can
guide engineering. They cannot promote, rank, kill, anchor stack math, or
support Shannon-floor claims.

## Executive Finding

The repo already has the right skeleton for contest-grade sensitivity work:

- `experiments/contest_auth_eval.py` is a strong canonical custody wrapper for
  exact CUDA archive evaluation.
- `src/tac/component_sensitivity_artifact.py` defines a promotion-aware
  `component_sensitivity_v1` schema with CUDA, custody, sample-plan, stability,
  response-curve, and exact-eval gates.
- `experiments/profile_hessian_per_weight.py`,
  `experiments/convert_fisher_to_owv3_sensitivity_map.py`, and
  `src/tac/sensitivity_map.py` form a useful CUDA-authored Fisher-to-channel
  map pipeline.
- Alpha-Geo diagnostics provide useful preflight geometry checks before
  spending exact eval budget.

The missing piece is a producer and validator that connects those pieces under
one deterministic, component-separated perturbation protocol. Current Fisher
artifacts are useful empirical CUDA proxy artifacts, but they do not yet prove
component sensitivity, stack composability, or exact archive score improvement.

## Tool Inventory

### Exact scorer/eval path

- `experiments/contest_auth_eval.py`
  - Good: validates archive SHA/bytes, safe extraction, allowed members,
    inflate output counts and sizes, upstream eval path, CUDA/device
    provenance, JSON artifact, `n_samples == 600`, finite components, and
    recomputed contest formula.
  - Good: treats structured `contest_auth_eval.json` as canonical artifact.
  - Caution: the wrapper currently forces `PYTHON_INFLATE=renderer` in the eval
    environment even though the docstring describes a generic submission eval
    wrapper. That is fine for current robust renderer lanes, but it should be
    made explicit or generalized before using the wrapper for non-renderer
    submissions.

- `upstream/evaluate.py`, `upstream/modules.py`, `upstream/frame_utils.py`
  - Good: official evaluator defines the score formula and component metrics.
  - Important: PoseNet consumes YUV6 sequence pairs and computes MSE over the
    first six pose outputs. SegNet consumes the last frame and computes argmax
    disagreement. Differentiable proxies must not be mislabeled as exact
    SegNet or exact score deltas.
  - Important: the official evaluator prints text; the repo wrapper must remain
    responsible for converting that output into guarded structured JSON.

### Differentiable scorer/proxy support

- `src/tac/scorer.py`
  - Good: `load_differentiable_scorers()` patches the scorer path for gradient
    work by replacing non-differentiable normalization helpers.
  - Caution: `detect_device()` auto-selects CUDA, then MPS, then CPU. Any
    promotion-capable profiler must override this behavior and fail closed on
    non-CUDA unless explicitly marked diagnostic.
  - Caution: `compute_proxy_score()` is useful for dev checks but wraps scorer
    forwards in `torch.no_grad()` and is therefore not a gradient/Fisher tool.

### Fisher/Hessian profiling

- `experiments/profile_hessian_per_weight.py`
  - Good: defaults to CUDA, rejects MPS, supports protected Conv2d inclusion,
    fixes grayscale mask decoding to class ids, records useful metadata, and
    accumulates per-weight squared gradients for eligible Conv2d/Linear
    weights.
  - Good: the current CUDA artifact has protected Conv2d coverage and feeds
    OWV3 conversion with `--missing-policy error`.
  - Caution: the script name says Hessian, but the computed quantity is
    accumulated squared gradients, closer to empirical Fisher/diagonal
    Gauss-Newton proxy than a validated Hessian.
  - Caution: SegNet contribution uses differentiable softmax cross entropy
    against GT class ids, not official argmax disagreement.
  - Caution: metadata lacks component-separated maps, calibration/holdout
    splits, response curves, input SHA/bytes for all scorer inputs, exact
    archive linkage, and finite-difference validation.
  - Evidence grade today: empirical CUDA proxy, not promotion-grade component
    sensitivity.

- `experiments/convert_fisher_to_owv3_sensitivity_map.py`
  - Good: requires CUDA-authored source metadata by default, supports strict
    missing policies, records checkpoint and Fisher hashes, and can fail closed
    when Conv2d sensitivity is missing.
  - Caution: output is a single per-channel map, not PoseNet, SegNet, and
    combined component maps.

- `src/tac/sensitivity_map.py`
  - Good: defines a small, testable sensitivity map contract and CUDA
    authority checks.
  - Good: includes a train/holdout CV distance helper.
  - Missing: no current producer emits a complete CV/stability artifact using
    that helper.

### Component sensitivity artifact

- `src/tac/component_sensitivity_artifact.py`
  - Good: defines and validates `component_sensitivity_v1`.
  - Good: promotion validation requires CUDA, non-smoke/non-proxy markers,
    input custody, calibration/holdout plan, PoseNet/SegNet/combined maps,
    stability metrics, response curves, and exact eval custody.
  - Missing: no current profiling pipeline emits this artifact end to end.

### Pair and pixel sensitivity diagnostics

- `experiments/profile_pair_sensitivity.py`
  - Good: computes per-pair PoseNet, SegNet, and contribution values using the
    scorer path and can identify high-impact frame pairs.
  - Caution: it accepts CPU/MPS if requested and does not force a
    non-promotable label in the sidecar. It should be fail-closed or explicitly
    diagnostic on non-CUDA.
  - Caution: custody is incomplete for promotion: input hashes, exact archive
    linkage, split hash, and full response curves are missing.

- `experiments/profile_scorer_saliency.py` and
  `src/tac/saliency_inversion.py`
  - Good: CUDA-only saliency entry point and useful pixel-gradient diagnostics.
  - Caution: SegNet saliency is entropy/softmax-gradient proxy behavior, not
    exact argmax disagreement. No finite-difference response validation is
    attached.

### Legacy sensitivity scripts

- `experiments/profile_fp4_layer_sensitivity.py`
  - Caution: allows CPU if explicitly requested. It appears to decode grayscale
    mask video frames as class ids without the luma-to-class remap now present
    in `profile_hessian_per_weight.py`. If used with Lane G/Lane A-style mask
    videos, this can create wrong renderer inputs or failures.

- `experiments/scorer_sensitivity_sweep.py`,
  `experiments/sensitivity_sweep.py`,
  `experiments/analysis/posenet_sensitivity.py`, and
  `experiments/analysis/posenet_sensitivity_map.py`
  - Caution: these are legacy diagnostic/proxy tools. Defaults or accepted
    devices include MPS/CPU in several paths. They should be quarantined from
    promotion workflows or rewritten to emit explicit diagnostic evidence
    labels.

### Alpha-Geo tooling

- `experiments/diagnose_nerv_geometry.py`
  - Good: safe archive member loading, deterministic mask-geometry metrics,
    boundary-band checks, temporal checks, speckles, transition F1, and
    centroid jumps.
  - Good: correctly labels output as empirical CPU geometry diagnostics.
  - Role: exact-eval budget preflight only. It can reject obviously bad alpha
    candidates from dispatch, but it cannot promote or kill.

- `experiments/paradigm_alpha_real_archive_eval.py`
  - Caution: empirical tooling still uses direct ZIP extraction. It should use
    the same zip-slip-safe extraction pattern as contest auth eval before it is
    reused.

- `experiments/train_nerv_mask.py`
  - Caution: current target generation supports fresh SegNet labels by default.
    The Alpha-Geo redesign requires a decoded-baseline target mode so the model
    preserves the exact geometry of the current archive instead of chasing a
    fresh scorer-side target.

### OWV3 sensitivity-weighted codec path

- `src/tac/owv3_sensitivity_weighted.py`
  - Good: decode path is scorer-independent, diagnostic fp16 fallback is marked
    non-promotable, and byte planning is explicit.
  - Good: byte budget enforcement rejects byte regressions unless a reviewed
    distortion justification is supplied.
  - Caution: `aggressive_threshold` is validated and stored, but current channel
    classification only uses the protection threshold. Either implement the
    aggressive action tier or remove the dead knob.
  - Caution: Conv biases remain fp16 side payloads for OWV3 Conv blocks and
    should be reviewed if byte-blocked.

- `experiments/build_lane_g_v3_owv3_stack.py`
  - Good: deterministic archive rebuild check, manifest, byte accounting, and
    default comparison to the current PFP16 frontier.
  - Caution: the latest known OWV3 r2 artifact reduced renderer bytes but was
    still larger than PFP16 at archive level because of the surrounding stack
    payload. That is byte evidence only and must not be evaluated as a ranking
    claim.
  - Caution: display math contains a hardcoded Lane G v3 baseline score for
    rate-only prediction. Keep exact JSON recomputation authoritative.

## Deterministic Perturbation Basis Roadmap

Create a canonical `perturbation_basis_v1` artifact that names every atom that
can be perturbed, its normalization, its input tensors, and its deterministic
order. The basis must be independent of wall-clock order, filesystem traversal
order, and GPU nondeterminism where practical.

Required basis metadata:

- `schema_version`: `perturbation_basis_v1`
- `basis_id`: SHA-256 over canonical JSON metadata plus input hashes
- `source_archive_sha256` and `source_archive_bytes` when archive-derived
- renderer/checkpoint hashes, mask/video/pose hashes, upstream hash
- ordered sample plan: pair ids, calibration pairs, holdout pairs, split seed,
  split hash, and exclusion reasons
- device policy: CUDA required for promotable artifacts
- normalization policy: per-atom norm, epsilon ladder, clamp/domain behavior,
  sign convention, and aggregation rule
- deterministic ordering: module path, tensor name, channel, block, spatial
  index, pair index

Recommended atom families:

- Pair atoms: one scalar coefficient per evaluator pair for pair-weight and
  high-leverage sample analysis.
- Pixel atoms: YUV/RGB patch atoms on the scorer input grid with fixed patch
  size, stride, channel grouping, and clamp behavior.
- Mask geometry atoms: boundary-band flips, connected-component preserving
  edits, temporal transition edits, and lane/vehicle critical-region atoms.
- Renderer atoms: Conv2d output-channel groups, block groups, quant-action
  groups, bias groups, and protected channel groups.
- Pose atoms: per-pair pose target perturbations and pose-output dimensions for
  PoseNet sensitivity audits.
- Archive action atoms: byte-accounted codec decisions such as protected fp16,
  ASYM, OWV2/OWV3 quant action, entropy payload choice, and pose payload
  variant.

All random sampling must be replaced by seeded, materialized sample plans. If
randomness is unavoidable in a diagnostic sweep, the artifact must be marked
non-promotable and record the PRNG, seed, and sampled atom ids.

## CUDA-Only Exact Evidence Rules

Profilers and perturbation tools need two separate evidence labels:

- `diagnostic_cuda`: CUDA scorer/proxy/finite-difference evidence without exact
  archive eval custody.
- `promotion_candidate_input`: CUDA-authored component artifact that is complete
  enough to justify building an archive, but still not a score claim.
- `exact_archive_cuda`: exact archive evidence from `contest_auth_eval.py` with
  archive SHA/bytes, manifest, logs, hardware, `n_samples == 600`, and
  recomputed score.

Promotion-capable sensitivity tooling must fail closed unless:

- `device == cuda`
- scorer inputs and model inputs have SHA-256 and byte custody
- sample plan is materialized and split into calibration and holdout
- PoseNet, SegNet, and combined maps are stored separately
- response curves and stability metrics are present
- output tensors are finite, nonnegative where required, and shape-checked
- provenance records exact command, git status, environment, hardware, PyTorch,
  CUDA, and upstream hashes

Any CPU/MPS/proxy/smoke/debug/dummy/random result must carry an explicit
non-promotable marker in JSON. Human-readable logs are not enough.

## Finite-Difference, Hessian, and Fisher Validation

Current per-weight Fisher gives a useful ranking hypothesis. It needs a
validation harness before it can drive exact-eval spending with confidence.

Recommended validation protocol:

1. Choose a fixed, materialized subset of perturbation atoms from calibration
   and holdout splits.
2. For each atom, run symmetric central differences with an epsilon ladder:
   `-eps`, `0`, `+eps`, and optionally `+-2eps`.
3. Record component deltas separately:
   - `Delta PoseNet`: official-style pose MSE on CUDA scorer path.
   - `Delta SegNet proxy`: differentiable CE/softmax proxy when used.
   - `Delta SegNet exact-local`: argmax disagreement on the local CUDA scorer
     path when available.
   - `Delta combined`: contest-formula local recomputation from component
     measurements, labeled diagnostic unless it came from exact archive eval.
4. Compare predicted versus observed deltas:
   - sign accuracy
   - Spearman/Pearson rank correlation
   - top-k overlap
   - calibration/holdout CV distance
   - slope and curvature fit over the epsilon ladder
   - non-monotonic or sign-asymmetric response flags
5. Store response curves in `component_sensitivity_v1` rather than a separate
   unlinked sidecar.

Hessian language should be reserved for actual second-order estimation:

- Use Hessian-vector products or block/group second differences on a small
  deterministic atom subset.
- Do not attempt a dense Hessian for renderer weights.
- Label squared-gradient accumulators as empirical Fisher or diagonal
  Gauss-Newton proxy unless a Hessian/HVP calculation is performed.
- Treat nonconvex response curves as first-class evidence against naive
  additive stacking assumptions.

SegNet requires special care:

- Differentiable CE/entropy proxies are useful for optimization but are not the
  official argmax disagreement metric.
- Every SegNet proxy map needs an argmax-response validation subset.
- Component collapse gates must be checked by exact archive eval before any
  score-affecting dispatch is promoted.

## Profiler Hooks

The next profiler should emit one canonical artifact instead of many loosely
related sidecars.

Recommended new entry point:

```text
experiments/profile_component_sensitivity.py
```

or a strict extension of:

```text
experiments/profile_hessian_per_weight.py
```

Required outputs:

- `component_sensitivity_v1.json`
- `perturbation_basis_v1.json`
- PoseNet map tensor plus hash
- SegNet map tensor plus hash
- combined map tensor plus hash
- response curves JSONL or JSON with tensor hashes
- calibration/holdout stability JSON
- command/environment/hardware/source manifest
- optional exact archive custody section when an archive has already been
  evaluated

Profiler performance hooks are useful but must be separated from scientific
claims:

- CUDA memory peak and runtime per stage
- scorer forward/backward timing
- renderer forward/backward timing
- number of pairs, atoms, and gradients processed
- failure/retry count
- deterministic flags and CUDA library settings

These hooks help decide wall-clock routing. They are not score evidence.

## Reproducibility Artifact Requirements

Every perturbation or sensitivity run should write a directory with:

- `command.txt`
- `environment.json`
- `hardware.json`
- `git_status.txt`
- source/staged-tree manifest with SHA-256, bytes, and role
- upstream manifest/hash
- archive manifest when archive-derived
- scorer model source hashes when available
- input hashes for renderer, masks, poses, videos, pair weights, and GT data
- sample plan with split seed and split hash
- perturbation basis artifact
- component sensitivity artifact
- tensor files with SHA-256 and dtype/shape metadata
- response curves and stability metrics
- exact eval JSON/logs only when produced through canonical auth eval

For learned or archive-impacting lanes, include deterministic archive rebuild
proof: member order, timestamps, permissions, compression settings, manifest,
archive SHA-256, archive bytes, and deterministic rebuild comparison.

## Minimal Code Changes Needed Next

These are roadmap items, not edits made in this audit.

1. Add a `component_sensitivity_v1` producer.
   - Best location: extend `experiments/profile_hessian_per_weight.py` or add
     `experiments/profile_component_sensitivity.py`.
   - It must emit PoseNet, SegNet, and combined maps with calibration/holdout
     splits and response curves.

2. Add a deterministic perturbation basis module.
   - Suggested path: `src/tac/perturbation_basis.py`.
   - Include canonical JSON materialization, stable atom ordering, split hashes,
     and tests.

3. Add finite-difference response validation.
   - Suggested path: `experiments/validate_component_sensitivity_response.py`.
   - Use central differences on deterministic atom subsets and write response
     curves into the component artifact.

4. Fail closed or explicitly label legacy profilers.
   - `experiments/profile_pair_sensitivity.py` should reject non-CUDA unless an
     `--allow-non-cuda-diagnostic` flag is set, and JSON must mark such output
     non-promotable.
   - `experiments/scorer_sensitivity_sweep.py`,
     `experiments/sensitivity_sweep.py`, and the two PoseNet analysis scripts
     should be quarantined as diagnostic or upgraded to CUDA-only evidence
     labels.

5. Port the luma-to-class mask decode fix into
   `experiments/profile_fp4_layer_sensitivity.py`.
   - This is a high-confidence correctness issue if that script is reused with
     grayscale class-coded mask videos.

6. Add decoded-baseline mask target mode for Alpha-NeRV/Alpha-Geo training.
   - `experiments/train_nerv_mask.py` should support targets from the decoded
     baseline archive masks, not only fresh SegNet labels.

7. Replace direct ZIP extraction in Alpha empirical tooling.
   - `experiments/paradigm_alpha_real_archive_eval.py` should use the
     zip-slip-safe extraction helper pattern from contest auth eval.

8. Resolve OWV3 archive byte blockers before exact-eval spending.
   - Build against the current PFP16 anchor or reuse the PFP16 pose payload
     where contest-compatible.
   - Implement or remove `aggressive_threshold` so byte-plan knobs correspond
     to actual codec actions.
   - Review Conv bias payload format for byte savings.

9. Clarify `contest_auth_eval.py` renderer-only behavior.
   - Either document that the current wrapper is intentionally renderer-lane
     specific, or add a reviewed submission-mode option that does not force
     `PYTHON_INFLATE=renderer`.

## Recommended Execution Order

1. Land artifact-schema producer and deterministic perturbation basis tests.
2. Re-run CUDA Fisher/component sensitivity on the known OWV3 calibration plan.
3. Validate top-k and random-holdout atoms with finite differences and response
   curves.
4. Only then build byte-feasible archives from the validated action map.
5. Spend exact CUDA auth eval only on byte-feasible or explicitly reviewed
   distortion-justified archives.
6. Promote or retire only exact archives, scoped to the measured implementation
   and config.

## Current Evidence Classification

- Exact CUDA archive truth remains in `contest_auth_eval.json` artifacts.
- Current OWV3 Fisher-derived maps are empirical CUDA proxy sensitivity
  artifacts.
- Current Alpha-Geo diagnostics are empirical geometry preflight artifacts.
- Current local saliency, pair sensitivity, FP4 layer sensitivity, and legacy
  sweeps are diagnostic unless upgraded to the component artifact protocol.
- No new score claim, rank claim, promotion, kill, or retirement is made by
  this audit.

## Files Changed By This Audit

- `.omx/research/posenet_segnet_perturbation_tooling_audit_20260430_agent.md`

No code files were modified.
