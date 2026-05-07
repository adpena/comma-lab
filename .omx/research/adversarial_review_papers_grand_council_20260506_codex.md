# Adversarial Review — Papers, Grand Council, And Score-Lowering Surfaces — 2026-05-06

This ledger records the current senior-engineer review pass over the active
paradigm implementations. It is not a score ledger and makes no score claim.
All findings below are implementation, custody, math, or research-alignment
items that must be resolved before a lane can be promoted toward exact CUDA
auth eval.

## External Reference Anchors

- LA-Pose project page, `LA-Pose: Latent Action Pretraining Meets Pose
  Estimation`, CVPR 2026. The paper describes self-supervised latent-action
  pretraining from unlabeled driving video, then a pose head that predicts
  translation, rotation/quaternion, FOV, and metric scale from latent actions.
  Source: https://la-pose.github.io/
- Park et al., `Semantic Image Synthesis with Spatially-Adaptive
  Normalization`, CVPR 2019 / arXiv 1903.07291. SPADE is semantic-layout
  conditioning through spatially adaptive normalization, so any runtime use of
  labels/semantic layouts must be archive-charged and label-order canonical.
  Source: https://arxiv.org/abs/1903.07291
- Tan et al., `Efficient Semantic Image Synthesis via Class-Adaptive
  Normalization`, arXiv 2012.04644. CLADE trades spatially adaptive per-pixel
  modulation for class-adaptive normalization plus optional intra-class
  positional encoding; this reinforces the need for canonical class order and
  label-contract tests. Source: https://arxiv.org/abs/2012.04644
- Yousfi, Dworetzky, and Fridrich, `Detector-Informed Batch Steganography and
  Pooled Steganalysis`, IH-MMSec 2022. The useful analogy is detector-informed
  allocation under nonlinear detector feedback, not free uncharged labels or
  broad additive assumptions. Source: https://doi.org/10.1145/3531536.3532951
- Holub, Fridrich, and Denemark, `Universal Distortion Function for
  Steganography`, EURASIP JIS. UNIWARD uses directional wavelet relative
  distortion and the paper's additive approximation; 2-tap gradient proxies must
  be labelled as approximations and cannot support broad Fridrich claims.
  Source: https://ws.binghamton.edu/fridrich/research/uniward-eurasip-final.pdf

## Findings

### F1 — CRITICAL — β Sensitivity Producer Has No Promotable Compute Path

Evidence:
- `experiments/build_sensitivity_map_pr106.py:109-119` defines
  `_iter_contest_pairs()` as an unconditional `NotImplementedError`.
- `experiments/build_sensitivity_map_pr106.py:181-206` enters the supposed
  CUDA path with `posenet = None`, `segnet = None`, then fails closed when the
  iterator raises.
- `scripts/remote_lane_omega_w_v3_pr106.sh` still calls this producer directly
  in CUDA mode, so a remote launch would fail instead of producing a certified
  artifact.

Impact:
- The new validator correctly blocks stub sensitivity maps, but Ω-W-V3/NWCS
  cannot produce a certified sensitivity artifact because the certified
  producer cannot compute one.
- This blocks the adjoint/Frechet derivative program for `d(score)/d(atom)`
  on β lanes.

Required fix:
- Implement a real CUDA scorer/pair iterator that consumes the canonical
  `archive.zip -> inflate.sh -> upstream/evaluate.py` frame-pair contract or
  an exactly reviewed equivalent.
- Split the computation into an importable function with tests that mock only
  the pair/scorer interfaces, while promotion still requires real CUDA metadata
  and certification hashes.
- Until that lands, mark the remote wrapper historical or blocked so operators
  do not believe Stage 2 is launch-ready.

### F1b — HIGH — β Certification Is Not Bound Tightly Enough To The Actual Archive And Model

Evidence:
- Reviewer B found that `experiments/pipeline.py` validates the sensitivity
  artifact without passing expected `source_archive_sha256`,
  `source_archive_bytes`, or component scope into the validator.
- `src/tac/sensitivity_map.py` checks for SHA-shaped metadata and bad
  stub/proxy markers, but does not yet prove the map belongs to the actual
  candidate archive/model/eval contract.

Research alignment:
- OBD/HAWQ-style sensitivity is local to the objective, weights, and model
  state being quantized. It is not a transferable metadata label.

Required fix:
- Extend `validate_real_sensitivity_artifact()` with optional expected
  archive bytes/SHA, checkpoint SHA, component scope, and certification-summary
  hash checks.
- Require `component="combined"` for score-lowering candidates unless the
  manifest explicitly says the row is component-only planning evidence.

### F1c — HIGH — Ω-W Repack Can Omit Sensitivity Coverage And Still Look Ready

Evidence:
- Reviewer B found `experiments/repack_pr106_with_water_filling.py` budgets
  only tensors present in the sensitivity map, while missing layers fall back
  to PR106 int8/brotli behavior.
- The output labelling can still imply readiness when a subset of water-fillable
  tensors was never covered by sensitivity evidence.

Research alignment:
- OBD saliency and HAWQ bit allocation are per-perturbation/per-parameter
  analyses. Uncovered tensors invalidate the allocation claim unless their
  exclusions are explicit and reviewed.

Required fix:
- In non-stub/non-design mode, require every water-fillable Conv2d tensor to
  have canonical sensitivity, or require an explicit reviewed exclusion list.
- If any fallback occurs, mark the manifest non-promotable until a coverage
  report explains it.

### F1d — MEDIUM — Current β "Water Filling" Is Proportional Allocation, Not Reverse Water Filling

Evidence:
- Reviewer B found `experiments/repack_pr106_with_water_filling.py` allocates
  bytes proportional to sensitivity sums.
- The repo already has a canonical `water_fill_bit_budget` implementation in
  `src/tac/water_filling_codec.py` using variance/count-aware allocation.

Research alignment:
- Reverse water filling allocates rate by source variances/eigenvalues under a
  common water level. Sensitivity can weight distortion, but the allocation
  still needs a marginal rate-distortion objective.

Required fix:
- Build a global channel table over eligible tensors with variance, count, and
  sensitivity. Feed that table to the canonical allocator, then lower the
  channel-level qint plan into OWV/OWV3.

### F1e — MEDIUM — OWV3 Sensitivity Threshold Is Absolute And `aggressive_threshold` Is Dead

Evidence:
- Reviewer B found `src/tac/owv3_sensitivity_weighted.py` classifies channels
  with an absolute sensitivity threshold and records `aggressive_threshold`
  without using it for allocation.

Research alignment:
- HAWQ/HAWQ-V2 use relative second-order sensitivity and Pareto/bit selection,
  not unnormalized absolute thresholds.

Required fix:
- Normalize sensitivity units per artifact, switch to quantile/top-k or
  marginal rate-distortion allocation, and either implement the aggressive tier
  or remove the parameter from public config.

### F2 — HIGH — LA-Pose/Foveation Is Structurally Byte-Closed But Not Paper-Aligned Yet

Evidence:
- `src/tac/lapose_foveation_runtime_skeleton.py:151-219` always returns
  `passed: False` and always appends
  `lapose_foveation_scorer_visible_output_parity_not_proven`.
- `src/tac/lapose_foveation_payload_candidate.py:790-861` correctly validates
  the fail-closed bridge report.
- Reviewer D found `src/tac/hyperbolic_foveation.py` lets caller-supplied
  `image_size` override HFV1 header dimensions, so readiness can miss a
  payload written for the wrong frame lattice.
- Reviewer D found `src/tac/foveation_readiness.py` proves runtime consumption
  with substring checks; a comment containing `load_foveation_params` and
  `foveation_params.bin` can remove the consumer blocker.

Paper alignment:
- LA-Pose is not a tuple sidechannel by itself. The paper's mechanism is
  latent-action pretraining from video and a pose estimator over latent action
  features. Our LFV1 tuple payload is only a structural bridge until it drives
  scorer-visible RGB/mask/pose output.
- Telescope/foveated rendering papers treat foveation as grid-dependent
  resampling. Payload dimensions must be bound to the actual scorer-visible
  image lattice; they cannot be silently coerced by the checker.

Required fix:
- Add a real `optimize_poses.py --init-poses` or equivalent pose-initialization
  path that consumes archive-contained LA-pose/RAFT/radial outputs.
- Add a mutation control proving LFV1 changes scorer-visible frames, masks, or
  poses, plus an identity control proving no-op parity.
- Make HFV1 load/readiness validate header dimensions against expected image
  size instead of replacing header dimensions.
- Replace substring runtime-consumption proof with AST or runtime proof of an
  actual call path from archive member load to foveation application.

### F2b — MEDIUM — RAFT/Radial Pose Math Is A Contest Proxy, Not Calibrated Ego-Motion

Evidence:
- Reviewer D found `src/tac/raft_radial_pose.py` describes its 6-DoF flow basis
  as Longuet-Higgins-style ego-motion, but the rotational fields omit calibrated
  normalized-coordinate terms and translation lacks depth/intrinsics.
- Existing tests synthesize from the same proxy basis, so they cannot falsify
  the physical model.

Required fix:
- Either rename the implementation and docs as a contest proxy, or implement a
  calibrated normalized-coordinate flow basis with explicit depth/proxy-depth
  assumptions and analytic tests.

### F2c — MEDIUM — Radial-Zoom Pose Optimization Ignores Its Own Masked Loss Helper

Evidence:
- Reviewer D found `experiments/optimize_poses.py` defines `_posenet_mse_loss`
  for masked radial-zoom loss but later computes full 6D PoseNet MSE directly.

Required fix:
- Route radial-zoom mode through the masked helper with dims `(0,)`; use all
  available dims for full pose mode. Add a gradient/loss regression proving
  target changes in dims 1-5 do not affect radial-zoom loss.

### F3 — HIGH — HNeRV HDC2 Entropy Work Product Is Byte-Equivalent But Not Score-Lowering

Evidence:
- Current HDC2 work product records source decoder section bytes around
  `170278` and candidate stream bytes around `221381`.
- `src/tac/hnerv_entropy_candidate_packet.py:500-507` still lists the correct
  blockers: model-overhead reduction, context-table diff, candidate archive
  manifest, strict compliance JSON, meta-Lagrangian atom, runtime parity,
  lane claim, and exact CUDA.

Impact:
- The work product is useful forensic entropy evidence but should be treated as
  a negative compression/control result until it beats the source section after
  all model overhead is charged.

Required fix:
- Produce an entropy-rate decomposition table with actual bits/symbol,
  static-context overhead, model-overhead bytes, and entropy-floor gap.
- Add a negative-result row to the meta-Lagrangian/field planner so this stream
  is not selected until byte delta is truly negative under charged runtime.

### F3b — MEDIUM — HDC2 CLI Treats Runtime-Tree Custody As Optional Despite Validation Requiring It

Evidence:
- Reviewer A found the HDC2 CLI help labels source exact-eval/runtime custody
  JSON as optional while the packet validator requires `runtime_tree_sha256` in
  the source archive manifest.

Impact:
- This is fail-closed, not a bypass, but it wastes operator time by allowing
  knowingly invalid packet artifacts.

Required fix:
- Make the exact-eval/runtime custody JSON required whenever building an HDC2
  stream work product, or change the help text and tests to say the resulting
  packet is intentionally invalid until runtime custody is supplied.

### F3c — HIGH — WMC1/VQM1 Mask Decoders Can Accept Truncated Payloads

Evidence:
- Reviewer A reproduced one-byte truncation of VQ and wavelet mask payloads
  decoding to plausible wrong masks instead of failing closed.
- The implicated surfaces are `src/tac/wavelet_mask_codec.py` and
  `src/tac/vqvae_mask_codec.py`, where entropy payload slices and arithmetic
  bit readers do not prove exact encoded-length consumption.

Research alignment:
- VQ-VAE and wavelet coders are valid representational tools only if the
  runtime grammar is deterministic and uniquely decodable. EOF zero-padding
  violates archive custody because corrupted payloads can silently map to
  high-agreement but wrong outputs.

Required fix:
- Check `pos + payload_size <= len(blob)`, reject trailing or short payloads
  unless explicitly versioned, make EOF in arithmetic readers raise, and add
  `blob[:-1]` regression tests for both codecs.

### F4 — HIGH — Categorical/Openpilot Contract Still Needs A Canonical Label Map Artifact

Evidence:
- `src/tac/categorical_openpilot_mask_prior_contract.py:18` hardcodes
  `contest_zero_based_comma10k_order`.
- `src/tac/categorical_openpilot_mask_prior_contract.py:194-260` validates that
  runtime-consumed priors are charged and SHA-linked, which is the right
  custody direction.

Paper/source alignment:
- SPADE/CLADE require semantic layout consistency. Openpilot/supercombo outputs
  road/path/lane/lead signals, not automatically the same semantic class map as
  contest masks. A string label-contract name is not enough scientific custody.

Required fix:
- Commit a canonical label-contract JSON with class names, integer ids, source
  provenance, source SHA/URL, and permutation controls.
- Add tests that a label permutation or openpilot-lane/path reorder fails the
  readiness gate.

### F4b — HIGH — Categorical Readiness Can Self-Attest Dispatch Readiness

Evidence:
- Reviewer A found `categorical_candidate_readiness.py` accepts
  manifest-declared no-op controls and parity reports, then can set
  `ready_for_exact_eval_dispatch = len(blockers) == 0`.
- Existing tests include dummy payload bytes plus declared decoded-mask SHA and
  assert empty blockers/dispatch readiness.

Impact:
- This is too strong for a label-payload lane. The audit should prove archive
  readiness, but dispatch readiness requires independent decode/reencode proof,
  runtime-loader execution proof, lane claim, and exact CUDA auth eval.

Required fix:
- Downgrade the audit output to archive-readiness only, or require independent
  proof artifacts with path/SHA and rerunnable verification before dispatch
  readiness can be true.

### F5 — MEDIUM — Meta-Lagrangian Selector Is Safer, But Its Penalties Are Not KKT-Derived

Evidence:
- `tools/build_field_meta_dispatch_selection.py:51-62` defines additive
  penalty magnitudes such as `10000`, `1000`, and `500`.
- `tools/build_field_meta_dispatch_selection.py:1142-1151` adds these to
  expected score deltas to produce `field_selection_score`.

Impact:
- The current selector is appropriate as a planning and fail-closed ordering
  tool. It should not be described as a rigorous KKT/Pareto optimizer because
  the penalty units are arbitrary and not dimensionally tied to the action
  objective.

Required fix:
- Replace or supplement the additive score with an explicit lexicographic
  feasibility tuple:
  `exact_ready, static_ready, byte_closed, runtime_closed, clean, pareto, kkt,
  expected_score_delta, expected_information_gain`.
- Keep arbitrary penalties only as display diagnostics.

### F6 — MEDIUM — Fridrich Renderer Language Still Overclaims Relative To Evidence

Evidence:
- `experiments/train_renderer_fridrich.py:2-21` calls the lane "the path to
  sub-0.50" and includes projected scores in the module docstring.
- `experiments/train_renderer_fridrich.py:1433-1445` implements a conventional
  augmented Lagrangian constraint term, not specifically UNIWARD or
  detector-informed batch steganography.

Paper alignment:
- UNIWARD is directional wavelet relative distortion. Detector-informed
  steganography is about nonlinear detector feedback and allocation. The
  training loop may be useful, but the file should not imply Fridrich-grade
  correctness or score projection without exact evidence.

Required fix:
- Rewrite the docstring as an experimental constrained-renderer training path
  with evidence grade `prediction` until exact CUDA artifacts exist.
- If UNIWARD is claimed, wire actual Daubechies/wavelet relative distortion
  and label it separately from the augmented-Lagrangian scorer constraint.

## Grand Council Synthesis

The current implementations are moving in the right direction because they now
fail closed instead of silently emitting promotable stubs. The next tranche must
convert fail-closed scaffolding into one byte-closed candidate that can enter
the dispatch claim path.

Current exact local anchor from the reviewed state is PR106x-lowlevel-brotli at
score `0.20935073680571203`, `186080` bytes, exact T4 evidence. If SegNet and
PoseNet components remain unchanged, sub-0.15 by rate alone would require an
archive around `96946` bytes, about `89134` fewer bytes than that anchor. This
means small repacks are custody controls and incremental wins; sub-0.15 needs a
representation/distortion/pose breakthrough or a large byte-mass collapse.

Priority order:

1. β real sensitivity producer: highest cross-paradigm unlock, because it
   provides the adjoint/Frechet derivative substrate for byte allocation.
2. Categorical label-contract artifact and permutation controls: highest
   hidden-gem unlock for QMA9/CLADE/SPADE/openpilot work.
3. HNeRV entropy negative/positive byte accounting: fastest exact-byte
   decision on whether HDC2 can lower the current PR106x frontier.
4. LA-pose scorer-visible bridge: required before telescopic foveation or
   latent-action pose priors can affect score.
5. Meta selector lexicographic/KKT cleanup: prevents arbitrary planning weights
   from being mistaken for math.

No remote GPU work should be launched from this review alone. The next valid
score-lowering artifact is a byte-closed archive or certified sensitivity map
plus lane claim, then exact CUDA auth eval.

Additional custody/action notes from synthesis:

- Refresh `reports/latest.md` and paper result tables so PR106x-lowlevel is
  clearly the local analysis anchor, while PR100 remains only a submitted packet
  context where applicable.
- Record PR108 as an external non-frontier intake row: it is newer, but its PR
  body reports CPU score `3.59`, so it is not a score threat.
- Before dispatch or release claims, write a dirty-worktree custody note that
  separates owned dirty changes from partner/public-intake gitlinks and raw
  Kaggle checkout state.

## Takeover Fix Tranche 2026-05-06

This section records the resumed worker output after the beta worker failed at
model-capacity and its partial edits had to be taken over locally. No remote
GPU work, exact-eval dispatch, score claim, or lane claim was performed in this
tranche.

Implemented and reviewed fixes:

- `ATBH` arithmetic terminal now rejects truncated headers, invalid block
  sizes/counts, truncated payloads, and trailing bytes. Static and hyperprior
  encoders verify decode roundtrip before returning bytes.
- `HFV1` hyperbolic foveation payload loading now binds the on-wire image size
  and refuses caller-side dimension overrides. Runtime readiness proof now uses
  AST inspection and rejects comment-only mentions.
- `WMC1` and `VQM1` mask payload decoders now reject declared-payload overruns
  and trailing bytes instead of accepting ambiguous bytestreams.
- `HDC2` HNeRV entropy candidate packets now require an exact-eval source JSON
  when stream work products are supplied. The current HDC2 work product remains
  byte-equivalent and non-dispatchable, not a score claim.
- Beta sensitivity work now fails closed unless a real CUDA sensitivity
  artifact is bound to the source archive and model/checkpoint digests.
  Stub/proxy maps require explicit local design mode and cannot promote.
- PR106 water-fill repack now validates real sensitivity custody, requires
  explicit reviewed exclusions for uncovered water-fillable tensors, records
  fallback/promotability blockers, and refuses stub sensitivity unless local
  design mode is explicitly requested.
- RAFT/radial pose now uses normalized-flow calibration, focal-normalized
  proxy-depth translation terms, and calibrated least-squares projection to the
  contest pose basis. Radial-zoom TTO now constrains the PoseNet loss to the
  optimized one-dimensional pose component instead of scoring unrelated frozen
  dimensions.
- Categorical/openpilot readiness now requires independent decode/re-encode
  proof, independent runtime execution proof, active lane-claim representation,
  and exact CUDA auth-eval requirements before it can report ready for exact
  dispatch.
- Meta-Lagrangian/KKT selection now separates diagnostic penalty units from
  score deltas and uses lexicographic feasibility ordering. KKT-ready status
  requires a structured KKT proof or converged ADMM residual evidence.
- `submissions.apogee_v2` is now an importable runtime package so package-entry
  smoke tests exercise the same runtime surface expected by dispatch tooling.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_arithmetic_terminal.py \
  src/tac/tests/test_hnerv_entropy_candidate_packet.py \
  src/tac/tests/test_hyperbolic_foveation.py \
  src/tac/tests/test_foveation_readiness.py \
  src/tac/tests/test_wavelet_mask_codec.py \
  src/tac/tests/test_vqvae_mask_codec.py \
  src/tac/tests/test_raft_radial_pose.py \
  src/tac/tests/test_lane_m_v3_clean_train_inference_parity.py \
  src/tac/tests/test_optimize_poses_radial_zoom_save_shape.py \
  src/tac/tests/test_optimize_poses_lane_m_n_wiring.py \
  src/tac/tests/test_optimize_poses_kl_distill_wiring.py \
  src/tac/tests/test_sensitivity_map.py \
  src/tac/tests/test_build_sensitivity_map_pr106.py \
  src/tac/tests/test_repack_pr106_sensitivity_gate.py \
  src/tac/tests/test_dispatch_dryrun_omega_w_v3.py \
  src/tac/tests/test_pipeline_beta_dispatch.py \
  src/tac/tests/test_lane_omega_w_v3_local_smoke.py \
  src/tac/tests/test_build_categorical_candidate_fixture.py \
  src/tac/tests/test_build_categorical_candidate_payload.py \
  src/tac/tests/test_categorical_candidate_readiness.py \
  src/tac/tests/test_build_field_meta_dispatch_selection.py \
  src/tac/tests/test_build_frontier_roadmap_status.py \
  src/tac/tests/test_meta_lagrangian_allocator.py \
  src/tac/tests/test_field_equation_planner.py \
  -q
```

Result: `251 passed, 1 warning`. The warning is the intentional duplicate-ZIP
negative test in categorical readiness.

Additional checks:

```bash
.venv/bin/ruff check --select E9,F821,F823 <focused Python files>
.venv/bin/python -m py_compile <focused Python files>
git diff --check
```

Result: all passed.
