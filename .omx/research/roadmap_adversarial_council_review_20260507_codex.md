---
title: Roadmap adversarial council review for Wave-Omega and PARADIGM-dezeta
date: 2026-05-07
author: codex research/adversarial council
status: REVIEW LEDGER - no code changes promoted
evidence_grade: code-inspection + literature-constraint + readiness-audit
score_claim: false
dispatch_attempted: false
branch_constraint: main only
reviewed_control_ledgers:
  - .omx/research/wave_omega_stack_composition_blueprint_20260507_claude.md
  - .omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md
---

# Scope

This is an adversarial review of the current path toward lower-score roadmaps.
It intentionally makes no score claim. All numerical bands in the reviewed
blueprints remain `prediction` unless a later ledger binds exact archive bytes,
CUDA auth eval JSON, runtime-tree custody, and component recomputation.

Primary sources consulted:

- Balle et al., "Variational image compression with a scale hyperprior",
  ICLR 2018 / arXiv: <https://arxiv.org/abs/1802.01436>
- Chen et al., "NeRV: Neural Representations for Videos", NeurIPS 2021:
  <https://arxiv.org/abs/2110.13903> and
  <https://papers.nips.cc/paper/2021/hash/b44182379bf9fae976e6ae5996e13cd8-Abstract.html>
- Ewen et al., "Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range
  Object Detection", arXiv:2604.06332: <https://arxiv.org/abs/2604.06332>
- Holub, Fridrich, Denemark, "Universal distortion function for steganography
  in an arbitrary domain": <https://link.springer.com/article/10.1186/1687-417X-2014-1>
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network":
  <https://arxiv.org/abs/1503.02531>
- Park et al., "Semantic Image Synthesis with Spatially-Adaptive Normalization":
  <https://arxiv.org/abs/1903.07291>
- Tan et al., "Efficient Semantic Image Synthesis via Class-Adaptive
  Normalization": <https://arxiv.org/abs/2012.04644>
- comma/openpilot public docs: <https://docs.comma.ai/>

# Findings Ordered By Impact

## 1. Omega-1 is not dispatch-ready: the remote packer call is self-contradictory

Evidence grade: `code-inspection`, no score claim.

The Wave-Omega blueprint says the remaining Omega-1 blocker is only the three
SJ-KL helper stubs plus real scorer wiring. That is incomplete. The current
remote wrapper calls:

- `scripts/remote_lane_sjkl_c067.sh:384-388`:
  `experiments/build_sjkl_c067_archive.py ... --sjkl-member-name p`

But the builder's safe layout is explicitly "preserve source `p`; add
`sjkl.bin` as a top-level sibling":

- `experiments/build_sjkl_c067_archive.py:8-14`
- `experiments/build_sjkl_c067_archive.py:72-79`
- `experiments/build_sjkl_c067_archive.py:94-98`
- `experiments/build_sjkl_c067_archive.py:184-185`

A C067-style source archive already contains member `p`, so asking the builder
to add the SJ-KL payload under member name `p` should fail closed before exact
eval. The adjacent test is stale in the other direction: it asserts an old
`--payload-member-name p` token that does not match the live parser
(`src/tac/tests/test_remote_lane_sjkl_c067_script.py:55-61`).

Correction:

1. Change the wrapper to omit `--sjkl-member-name` or pass `sjkl.bin`.
2. Update the remote-script test to assert the live builder parser and the
   `sjkl.bin` sibling contract.
3. Add a no-op synthetic source archive test that fails if the wrapper ever
   targets `p` again.
4. Only after that, treat Omega-1 implementation work as the scorer/Fisher
   problem below.

## 2. SJ-KL still lacks the mathematical object it claims to approximate

Evidence grade: `code-inspection + derivation-risk`, no score claim.

Current code has two different objects under the "Fisher" label:

- `src/tac/sjkl_basis.py:328-335` computes a Hessian-vector product of a scalar
  `score_fn` through double backward.
- `src/tac/sjkl_basis.py:676-698` leaves `fisher_matvec`, `lanczos_topk`, and
  `effective_rank` as `NotImplementedError` stubs.
- `experiments/build_sjkl_residual.py:178-183` raises on the CUDA real-scorer
  path.

The official evaluator makes PoseNet differentiable but makes SegNet distortion
a hard argmax-disagreement metric:

- `upstream/modules.py:82-84` pose distortion is MSE on the first half of the
  pose head.
- `upstream/modules.py:111-113` segmentation distortion is hard
  `argmax != argmax` mean.

Therefore the roadmap needs an explicit surrogate contract before saying
"Fisher residual":

```text
F_local =
  lambda_pose * J_pose(x)^T J_pose(x)
  + lambda_seg  * J_seg_surrogate(x)^T W_seg J_seg_surrogate(x)
```

where `J_seg_surrogate` must specify the soft target, KL direction, temperature,
and an argmax trust-region proof. Hinton distillation supports soft targets and
temperature as a training signal; it does not make the hard SegNet contest
metric differentiable evidence by itself. The exact-eval gate remains the only
promotion route.

Correction:

1. Write the Fisher/GGN equation into `src/tac/sjkl_basis.py` docstrings before
   implementation.
2. Implement `fisher_matvec` as a vector-Jacobian/Jacobian-vector product
   against frozen scorer outputs, not a generic scalar Hessian unless the scalar
   surrogate is explicitly recorded.
3. Add a local rank-measurement manifest that records `surrogate_kind`,
   `kl_direction`, `temperature`, `pose_lambda`, `seg_lambda`, device, and
   frame geometry.
4. Gate the first charged `sjkl.bin` archive on `SJKL_REQUIRE_APPLIED=1` plus
   a perturbation/no-op control that proves selected pairs actually change
   scorer-visible frames.

## 3. Omega-2 NeRV status in the Wave-Omega ledger is stale

Evidence grade: `code-inspection`, no score claim.

The Wave-Omega blueprint says NeRV mask inflate wiring, trainer, encoder CLI,
and dispatch script are missing (`wave_omega_stack_composition_blueprint...:
88-96`, `210-213`). Current main contradicts that:

- `submissions/robust_current/inflate_renderer.py:1780-1814` loads `masks.nrv`
  with `decode_nerv_codec` and `render_mask_argmax`.
- `submissions/robust_current/inflate_renderer.py:2972-2990` dispatches by
  `.nrv` suffix or `NRV1` magic.
- `experiments/train_nerv_mask.py:1-38` is a CUDA-default NRV2 trainer and
  output writer.
- `scripts/remote_lane_nerv.sh:307-397` trains and rebuilds an archive with
  `masks.nrv`.

The real Omega-2 blockers are different:

1. The current NeRV implementation is a coordinate MLP over `(t, y, x)` to
   class logits. The NeRV paper's core claim is an image-wise implicit neural
   representation for video frames; transfer to categorical masks is plausible
   but not equivalent.
2. `scripts/remote_lane_nerv.sh:3`, `:40`, and `:127` still embed predicted
   score-band/provenance literals. These should be renamed to
   `prediction_only_*` fields or removed from dispatch provenance.
3. `scripts/wave_omega_2_nerv_full_cuda.sh:18-20`, `:121`, and `:155-156`
   also hard-code score-band/baseline literals in a pre-staged dispatcher.

Correction:

1. Replace the stale blueprint tasks with "prove archive replacement and
   runtime consumption on current `remote_lane_nerv.sh`".
2. Preserve the existing L2 clearance gates, but strip or clearly demote
   prediction literals before any real run.
3. Add an exact `masks.nrv -> inflate -> class tensor` parity/control report
   before pairing NeRV with Wave-Omega stack composition.

## 4. PARADIGM-dezeta entropy claims outrun the byte-producing code

Evidence grade: `code-inspection + literature-constraint`, no score claim.

Balle's scale-hyperprior source motivates side information and a jointly
optimized entropy model over latents. The current repo has useful scaffolds but
not the claimed epsilon codec:

- `src/tac/mdl_bayesian_codec.py:1-13` and `:19-22` explicitly say it is a
  codec-selection framework, not a byte-producing codec.
- `src/tac/mdl_bayesian_codec.py:104-150` ranks already measured
  `model_bits + residual_bits`; it refuses to manufacture bytes.
- `src/tac/balle_hyperprior_renderer.py:43-47` marks real-stream training,
  variational inference, and block-FP integration out of scope.
- `src/tac/balle_hyperprior_renderer.py:225-234` emits side-info MLP weights
  and states that the arithmetic coder driven by per-element sigma is out of
  scope.

The PARADIGM-dezeta blueprint's 1D-channel hyper-encoder and mixture-Gaussian
decoder may be good hypotheses, but they are not direct Balle guarantees.
Balle supports the general side-information accounting; it does not prove that
renderer-weight channel correlations will amortize a learned prior after zeta
without measured `L(M)+L(D|M)` on the exact qint stream.

Correction:

1. Implement a lossless byte codec first: quantized qint stream, quantized CDF
   tables or coder state, learned-prior bytes, and deterministic decode.
2. Compare against static histogram/range coder on the same renderer qints with
   `L(M)`, `L(D|M)`, and archive overhead separated.
3. Do not build a Phase-4 stack around epsilon until the byte-producing codec
   wins locally against the static coder under the same decode contract.

## 5. Zeta full-renderer self-compression needs layer inventory before a new module

Evidence grade: `code-inspection + implementation-risk`, no score claim.

The blueprint proposes a new `self_compress_full_renderer.py`, but the current
`src/tac/self_compress.py` already contains the safer primitive:

- `src/tac/self_compress.py:1094-1184` swaps eligible `nn.Conv2d` layers in
  place and preserves a configurable protected-pattern list.
- `src/tac/self_compress.py:1275-1298` implements the differentiable rate
  penalty over self-compress layers.

Risk: a new full-renderer module can duplicate or bypass existing protection
logic. It also cannot assume every score-sensitive weight is a swappable
`Conv2d`; FiLM, linear conditioning, output heads, transposed convs, and small
class-affine parameters need explicit inventory and protection.

Correction:

1. Add an inventory tool/report first: layer name, type, parameter count,
   protected reason, current compressed bytes, proposed zeta treatment.
2. Reuse `swap_renderer_convs_with_self_compress` and `get_protected_patterns`
   unless a measured gap requires a new abstraction.
3. Export can be new, but must round-trip the existing swapped model before any
   from-scratch self-compress architecture is considered.

## 6. Telescopic foveation is a geometry hypothesis, not a score path yet

Evidence grade: `literature-constraint + local-readiness`, no score claim.

The Telescope paper is relevant because it uses learnable hyperbolic foveation,
but its stated domain is ultra-long-range object detection. That domain mismatch
matters: Pact's contest scorer sees short-range lane/road/movable structure and
PoseNet output, not the Telescope detector target distribution.

Local code already encodes this caution:

- `.omx/research/foveation_archive_runtime_proof_gate_20260506_codex.md` says
  foveation is useful only if scored inflate consumes charged foveation bytes.
- `src/tac/lapose_foveation_runtime_skeleton.py` is a fail-closed proof
  skeleton, not a contest decoder.
- `tools/audit_hyperbolic_foveation_readiness.py` exposes the charged-member
  and runtime-consumer checks.

Correction:

1. Treat foveation as a charged geometry residual candidate.
2. First prove inverse stability and no-op controls on frame geometry; then
   prove the runtime consumer changes scorer-visible frames.
3. Only after that should it enter LA-pose/TTO training as a compress-time
   prior or typed archive member.

## 7. Categorical/openpilot label priors are blocked on decode parity, not theory

Evidence grade: `code-inspection + source-contract`, no score claim.

SPADE/CLADE support semantic-label conditioning as an architecture idea, and
CLADE specifically argues that class adaptiveness can be more parameter
efficient than spatially adaptive normalization. That supports the class-prior
direction but does not replace contest label custody.

The current repo has the right local boundary:

- `.omx/research/categorical_label_contract_20260506_codex.md` distinguishes
  contest SegNet class IDs from Selfcomp grayscale LUT targets.
- `.omx/research/categorical_byte_closed_payload_candidate_20260506_codex.md:51-67`
  blocks dispatch on decode/re-encode and runtime parity.
- The same ledger's addendum keeps `ready_for_exact_eval_dispatch=false`
  after adding a charged label-prior manifest.

Correction:

1. Keep semantic label mapping centralized; do not infer contest class names
   from public openpilot prose when local evaluator code is available.
2. Recover full HPM1 decode and byte-exact re-encode before touching scorer
   spend.
3. Add a label-permutation fail-closed control; a semantic-prior runtime that
   still produces plausible frames after permuted labels is not consuming the
   charged prior in a contest-faithful way.

# Highest-EV Implementation Queue

1. Fix the SJ-KL remote packer invocation and stale remote-script test. This is
   a cheap blocking bug with direct dispatch impact.
2. Implement the real SJ-KL scorer/Fisher contract only after writing the
   surrogate equation and trust-region controls.
3. Refresh the Wave-Omega ledger/task list to reflect current NeRV code and move
   Omega-2 work to archive/runtime parity, not missing files.
4. Strip or demote prediction literals in NeRV/Wave-Omega remote provenance so
   no wrapper emits score-looking fields before exact CUDA evidence.
5. Build epsilon as a byte-producing entropy coder on real renderer qints before
   adding mixture priors or Phase-4 stack wiring.
6. Produce a zeta layer-inventory artifact from the existing self-compress
   primitives before creating a parallel full-renderer implementation.
7. Keep foveation and categorical/openpilot lanes behind their existing
   charged-runtime/no-op/decode-parity gates.

# Dispatch Boundary

No lane should be dispatched from this review alone. The next dispatchable
artifact must come from a code patch plus focused tests, then a normal
`tools/claim_lane_dispatch.py claim ...` row before any remote GPU work.
