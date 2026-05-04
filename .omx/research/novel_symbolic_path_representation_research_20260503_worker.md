# Novel Symbolic Path Representation Research - 2026-05-03 Worker

Scope: fast research/planning memo for intrinsically smaller data
representations, symbolic/path/logic compression, and arXiv:2603.21852.
Remote jobs launched: false. Dispatch claims created: false. Score claim:
false. Evidence grade: external literature + design review + repo-local
planning.

## Paper Summary: EML Single-Operator Symbolic Basis

Primary paper: Andrzej Odrzywolek, "All elementary functions from a single
binary operator", arXiv:2603.21852v2, submitted 2026-03-23 and revised
2026-04-04. URL: https://arxiv.org/abs/2603.21852

Core result: the paper constructs a single binary operator
`eml(x, y) = exp(x) - ln(y)` plus constant `1` and shows constructively that
standard scientific-calculator elementary functions can be generated from
that basis. The important representation idea is not "use exp/log for video";
it is the grammar collapse:

```text
S -> 1 | eml(S, S)
```

Every expression becomes a homogeneous binary tree. That gives a tiny program
grammar, a uniform tree serialization, and a clean search space for symbolic
regression. The paper also reports PyTorch complex128 EML-tree training with
Adam and a hardening/snap phase to exact binary symbolic choices; blind
recovery worked reliably at shallow depth but degraded quickly by depth 5-6.
The author explicitly notes overflow, NaN, complex-domain, clamping, and basin
search issues. Code/repro material is linked from the paper and mirrored at:
https://github.com/VA00/SymbolicRegressionPackage

## Relevance And Non-Relevance To Contest Archives

Relevant:

- It supports a "small charged DSL" mindset: replace repeated numeric/state
  payloads with a compact grammar plus charged coefficients/atoms.
- It gives a search template: fixed grammar, differentiable soft choices,
  then harden/snap to a deterministic wire program.
- It aligns with current Pact field-equation thinking: atom libraries,
  marginal benefit per byte, and exact CUDA as the oracle.
- The homogeneous-tree idea maps cleanly to mask boundary programs, temporal
  path programs, row-run dictionaries, and renderer tensor micro-libraries.

Not directly relevant:

- EML itself is a poor inflate primitive for this contest. `exp`, `ln`,
  complex intermediates, clamping, and NaN avoidance are runtime and
  reproducibility hazards on a T4-equivalent target.
- The paper is about elementary functions and symbolic regression, not about
  discrete segmentation masks, quantized renderer tensors, or PoseNet basins.
- The current pose/action streams are already tiny; a symbolic formula that
  saves tens of pose bytes is low EV unless it also improves PoseNet.
- Any task-specific constants, tables, learned programs, or formulas must be
  charged inside `archive.zip`; generic source code may only implement the DSL
  interpreter.

## Current Pact Fit

Current frontier anatomy from the active ledgers:

- C-067 A++ anchor: `276214` archive bytes, C067 source archive SHA
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
- Charged PR67/PR75-style mask segment: `219472` bytes. This is the dominant
  rate lever.
- QZS3/JointFrameGenerator renderer segment: about `56K` packed bytes, or
  `59288` logical C067 renderer member bytes in the self-compression ledger.
- QP1 pose segment: hundreds of charged bytes. PR75 action streams: hundreds
  of bytes. These are component-sensitive but not large rate levers.
- Pure-rate score movement is `25 / 37545489 = 6.65859e-7` per byte, so
  `1024` bytes is about `0.00068184` score, prediction-only.

Consequence: prioritize symbolic/path compression for masks first, renderer
tensor grammar second, and pose/action DSLs only when they improve components
or unlock a safer stack.

## Adjacent SOTA Signals

- AI Feynman shows why symbolic search can work when functions have
  separability, symmetry, and compositional structure, but also why blind
  generic expression search is hard. URL: https://arxiv.org/abs/1905.11481
- AI Feynman 2.0 adds Pareto accuracy/complexity search and graph-modularity
  discovery, which is closer to our atom-library objective. URL:
  https://arxiv.org/abs/2006.10782
- Cranmer et al. distill symbolic expressions from sparsity-biased deep
  models; this suggests training a renderer or mask predictor with sparse
  latent gates, then extracting formulas/programs. URL:
  https://arxiv.org/abs/2006.11287
- PySR/SymbolicRegression.jl is the practical mature tool family for Pareto
  symbolic regression if we need quick formula searches outside the inflate
  runtime. URL: https://arxiv.org/abs/2305.01582
- DreamCoder/LILO support the library-learning view: repeatedly solve small
  reconstruction tasks, compress common subprograms into reusable abstractions,
  and guide later search. URLs: https://arxiv.org/abs/2006.08381 and
  https://arxiv.org/abs/2310.19791
- Ladderpath is directly relevant to mask/action streams: it builds nested
  reusable components over symbolic sequences and emits a JSON hierarchy for
  compression/motif analysis. URL:
  https://www.sciencedirect.com/science/article/pii/S2590005625002905
- Compressed path databases with wildcards/redundant symbols are relevant to
  path/action tables and "don't-care" regions in mask rows. URL:
  https://research.ibm.com/publications/cutting-the-size-of-compressed-path-databases-with-wildcards-and-redundant-symbols
- HNeRV/HiNeRV show the broader neural-representation direction, but the repo
  already has NeRV/Q-FAITHFUL/self-compression paths; the new contribution
  here should be symbolic payloads that plug into them. URLs:
  https://arxiv.org/abs/2304.02633 and https://arxiv.org/abs/2306.09818

## Concrete Experiments

### 1. SPG1 Symbolic Path Grammar Mask Overlay

Design: add a charged `SPG1` mask overlay/program format that decodes to mask
repairs after the existing C067/PR75 mask stream or after a deliberately shaved
mask stream. Start with low-order paths, not EML: line segments, quadratic or
cubic Bezier boundaries, scanline spans, affine temporal tracks, connected
component births/deaths, and small residual tiles. Program atoms are charged;
the interpreter is generic source.

Plug-in path within hours:

- Planner emits symbolic boundary/path atoms from
  `experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/decoded_mask_array.npy`
  or current C067 decoded masks.
- Builder consumes them as a `masks.cdo1`/new `masks.spg1` overlay beside the
  existing QZS3/QP1 payload.
- Runtime applies the overlay before rendering frames.

Likely files touched:

- `experiments/plan_symbolic_mask_path_grammar.py` (new)
- `experiments/build_symbolic_mask_path_candidate.py` (new)
- `experiments/build_charged_mask_grammar_candidate.py`
- `submissions/robust_current/unpack_renderer_payload.py`
- `submissions/robust_current/inflate_renderer_grayscale.py` or
  `submissions/robust_current/inflate_renderer.py`
- tests under `src/tac/tests/test_symbolic_mask_path_grammar.py`

Expected score movement: prediction only. Full mask replacement has huge
rate upside but is not hours-safe because CMG3/CMG3A same-family byte wins
have PoseNet-collapse history. A safer overlay paired with a 5-12 KB AV1 mask
shave could net 3-10 KB if it keeps components near C067, or about
`0.0020-0.0068` pure-rate score. Component regressions can easily dominate.

### 2. Ladderpath/Re-Pair Row-Run Dictionary For Mask Programs

Design: treat each frame row or component boundary as a symbolic sequence of
class runs and deltas. Mine repeated subsequences ("ladderons") and emit a
charged dictionary plus references. This is the literal bridge from
Ladderpath/grammar compression to the existing CMG3 row-run work.

Plug-in path within hours:

- Emit a deterministic policy JSON from row-run motifs.
- Feed it to `experiments/build_cmg3_adaptive_runs_candidate.py` or a narrow
  successor, preserving the existing field-policy/base-run guards.
- Use it first as a repair/overlay dictionary, not as an immediate full-mask
  replacement.

Likely files touched:

- `experiments/plan_ladderpath_mask_runs.py` (new)
- `experiments/build_cmg3_adaptive_runs_candidate.py`
- `experiments/plan_yousfi_fridrich_field_equations.py`
- `src/tac/tests/test_build_cmg3_adaptive_runs_candidate.py`

Expected score movement: prediction only. A pure full-mask grammar can screen
as tens of KB smaller, but current exact-negative geometry makes that low
priority without a geometry escape proof. A trust-region row-run dictionary
should target 1-5 KB net rate improvement first, roughly `0.0007-0.0034`
pure-rate score, while producing negative/positive training signal for later
larger replacements.

### 3. QZS3 Renderer Tensor Grammar / Micro-Library

Design: do not use EML at inflate time. Instead, apply the single-grammar
lesson to QZS3/QBF1 renderer tensors: mine repeated quantized nibble/scale
blocks, affine-tile transforms, per-channel templates, and wildcard blocks.
Encode a small charged dictionary plus block references. Decode to the same
JointFrameGenerator state dict, then reuse the current loader path.

Plug-in path within hours:

- First pass is a byte-only local planner over `renderer.bin`, fail-closed if
  decoded state differs or if raw renderer outputs drift.
- If a new wire format is needed, implement a small `QRG1` loader branch with
  strict magic and manifest fields.
- Always run raw-output parity preflight before any exact eval consideration.

Likely files touched:

- `experiments/plan_c067_renderer_self_compression_v2.py`
- `src/tac/quantizr_qzs3_codec.py`
- `src/tac/qbf1_renderer_codec.py` or new `src/tac/qrg1_renderer_codec.py`
- `submissions/robust_current/inflate_renderer.py`
- `experiments/preflight_renderer_transplant_pose_safety.py`
- `src/tac/tests/test_plan_c067_renderer_self_compression_v2.py`

Expected score movement: prediction only. Local QZS3 recoding already found
only an 87-byte safe-looking win, and prior QBF1 exact CUDA collapsed PoseNet.
A real tensor grammar might save 0.5-4 KB if it preserves outputs, or
`0.0003-0.0027` pure-rate score. It becomes high-value only if it can clear
the C088 sub-0.314 pure-rate gap or combine with mask/action improvements.

### 4. Symbolic Pose Path Residuals

Design: fit QP1 velocity as piecewise low-degree polynomial/sinusoid/Bezier
paths over pair index, then charge sparse residuals for hard pairs. Avoid EML
runtime; use fixed-point polynomial/path evaluation. This is a component
experiment more than a rate experiment.

Likely files touched:

- `src/tac/qp1_pose_codec.py`
- `experiments/plan_pose_manifold_waterfill_candidates.py`
- `experiments/build_fixedslice_segment_mix_candidates.py`
- `submissions/robust_current/unpack_renderer_payload.py`
- `src/tac/tests/test_qp1_pose_codec.py`

Expected score movement: prediction only. QP1 pose bytes are already too small
for large pure-rate wins. Expect at most 50-200 bytes saved
(`<0.00014` pure-rate score), or a small component win if residuals target
PoseNet-hard pairs. This is not a first-choice byte lane.

### 5. PR75 Action Logic Compression / Wildcard Rules

Design: represent PR75 SegNet tile-action records as a tiny rule program:
ordered pair ranges, wildcard tile/action classes, and dictionary entries for
exceptions. The CPD wildcard/redundant-symbol idea is directly relevant here.

Likely files touched:

- `experiments/build_pr75_tile_action_subset_candidates.py`
- `experiments/build_pr75_lossless_repack_candidates.py`
- `submissions/robust_current/unpack_renderer_payload.py`
- `submissions/robust_current/apply_qzs3_postprocess.py`
- `src/tac/tests/test_build_pr75_tile_action_subset_candidates.py`

Expected score movement: prediction only. Action-stream rate is tiny, so
expect `<100` bytes pure-rate improvement unless the rule program enables a
better action subset/component effect. Useful as a precision tool, not a main
rate lever.

## Exact-Eval Route

No exact eval or remote dispatch should happen from this memo. Candidate path:

1. Build local deterministic archive with one strict payload member and full
   manifest: member bytes, SHA-256, source archive SHA, decoded mask/pose
   hashes, renderer magic, and no sidecars.
2. Run local parser/roundtrip/preflight tests. For renderer changes, require
   `experiments/preflight_renderer_transplant_pose_safety.py` to report
   `safe_for_exact_eval_dispatch=true` against the exact source archive.
3. Only after local gates pass, claim a lane with
   `tools/claim_lane_dispatch.py claim ...` before any non-dry-run GPU
   training/eval/remote submit.
4. Promotion evidence must be the canonical exact CUDA path:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

preferably via:

```text
experiments/contest_auth_eval.py --device cuda
```

Use structured `contest_auth_eval.json` and recompute the score from
components. Human logs are not authoritative.

## Risks And Guardrails

- EML runtime risk: exp/log/complex/clamp/NaN handling is not a good contest
  inflate dependency. Use EML as search-space inspiration only.
- Uncharged-state risk: learned grammars, constants, selectors, tables, and
  discovered formulas must be charged payload bytes.
- No-op risk: every builder must prove the targeted stream changed, or record
  source-preserving/no-op status in the manifest.
- Geometry cliff risk: current CMG3/QZS byte wins have exact-negative PoseNet
  history. Full replacement candidates need a new geometry escape proof.
- Loader risk: any new wire format needs content detection and fail-closed
  magic before `torch.load()` or decoder use.
- Parser/compliance risk: strict ZIP names, single packed payload container,
  deterministic timestamps/order/permissions, no hidden files, no central/local
  mismatch.
- Evidence risk: CPU/MPS/proxy and byte screens are planning only. Exact CUDA
  archive evidence is the score truth.

## Top 3 Immediate Tests

1. `test_symbolic_mask_path_roundtrip`: build `SPG1` from a small C067 decoded
   mask slice, decode it locally, assert exact or bounded disagreement,
   deterministic bytes/SHA, no unsafe members, and no host sidecars.
2. `test_ladderpath_runs_trust_region`: mine row-run motifs from decoded masks,
   emit field-policy JSON, build a local CMG3/overlay archive, assert base-run
   semantics, unique atoms, decoded disagreement threshold, and manifest
   `promotion_eligible=false`.
3. `test_renderer_qgram_pose_safety`: apply a renderer tensor grammar candidate
   to QZS3/QBF1 bytes, decode to JointFrameGenerator state, run raw-output
   parity through `preflight_renderer_transplant_pose_safety.py`, and fail
   closed unless bytes improve and output parity passes.

## Recommendation

Fastest useful implementation is not literal EML. Build `SPG1`/Ladderpath-style
mask path overlays that plug into current QZS3/QP1/PR75 archives and use the
existing CMG3/Yousfi field-policy machinery for atom selection. In parallel,
run a local-only QZS3 renderer tensor grammar screen guarded by output parity.
Pose formulas and action logic are secondary precision tools because their
charged streams are already too small for the main rate breakthrough.
