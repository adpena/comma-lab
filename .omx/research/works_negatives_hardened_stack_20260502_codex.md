# What Works, What Fails, And How The Stack Hardens - 2026-05-02 Codex

## Purpose

This ledger records durable scientific and engineering signal from the
contest-faithful Shannon-floor push. It is intentionally not a score ledger;
score claims remain in `shannon_floor_claim_matrix_20260430_codex.md`.

The working principle is that successful archives, negative exact evidence,
bug classes, preflight blocks, and composable software abstractions are all
part of the research result. A system that only remembers winners will repeat
expensive failures.

## Current Positive Evidence

- C-059 is the active internal A++ frontier: exact T4 score
  `0.3157055307844823`, archive `276347` bytes, SHA
  `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab`.
- The PR67/QZS3/QP1 public-basin route works as the dominant contest-faithful
  family so far: tiny JointFrameGenerator-style renderer, charged mask stream,
  lossy QP1 pose manifold, QZS3 grouped renderer packing, deterministic
  single-blob layout, exact CUDA validation.
- The anisotropic/active-subspace pose search worked until it reached a very
  flat local basin: C-054 -> C-056 -> C-057 -> C-058 moved PoseNet down and
  then squeezed bytes.
- QZS3 global block size `32` with mask-first single-blob layout works for the
  C-058/C-059 anchor and yields a real `75` byte T4-promoted win.
- The infrastructure loop works when followed strictly: active dispatch claim,
  source/artifact manifest, supply-chain scan, CUDA preflight, exact auth eval,
  state-derived harvest, adjudication, claim-matrix update, report update.

## Current Negative Evidence

- Global QZS3 block changes are sharply nonmonotone. Around C-058/C-059,
  b16, b24, b48, b64, and QZS4/block128 all collapse PoseNet or regress badly.
  This narrows the next renderer work to mixed/local block allocation or
  learned quantization, not another global scalar sweep.
- Raw residual-magnitude PVR1 pose atoms did not help at top32/top64: PoseNet
  and SegNet stayed unchanged while bytes increased. Pose side information is
  still open, but the selector must be scorer-informed, not residual-only.
- PR65-style postprocess atoms can improve SegNet locally but often collapse
  PoseNet. These atoms remain candidate repair tools only when combined with
  hard-pair/component traces and exact stacked archive eval.
- Plain CRF grayscale replacement and post-hoc AMR1 repair on a collapsed base
  failed catastrophically. That retires the measured implementation only; it
  does not kill grayscale, soft-LUT, SegMap, NeRV/INR, or Q-FAITHFUL.
- PR70/PR69-style uncharged script-payload or malformed-container behavior is
  useful compliance forensics but invalid for our contest-faithful route.

## Bug Classes Converted Into Guardrails

- Remote CUDA/Torch mismatch: T4/g4dn evals must pin inflate-side Torch to the
  CUDA-12 compatible wheel when the runner needs it. The exact-eval submit
  guard blocked C-059 until `INFLATE_TORCH_SPEC=torch==2.5.1+cu124` and related
  env vars were explicit.
- Same-lane parallel dispatch ambiguity: `tools/claim_lane_dispatch.py` now
  supports controlled child claims with `--allow-parallel`, `--child-of`, and
  `--parallel-reason`, and strict preflight checks for that surface.
- Repacker no-op controls: QZS repacking now records whether it reused,
  decoded/re-encoded, or transformed a source payload. Existing QZS3 sources
  with a changed block size must actually decode and re-encode.
- Packed-frontier source ingestion: repackers now handle already deployed
  single-blob archives (`p`/`renderer_payload.bin*`) through the runtime
  unpacker instead of requiring ad hoc manual extraction.
- Direct-script import fragility: experiment scripts that operators call from
  runbooks must self-root the repo path, not depend on implicit `PYTHONPATH`.
- Lightning SDK status regressions: nonterminal status regressions block
  status-only promotion, but terminal state-derived artifacts can still promote
  after local archive/JSON/trace validation.

## Software Stack Pattern

The stack is strongest when it acts like a tinygrad-inspired compiler for
archives:

- Small composable primitives: renderer codecs, pose codecs, mask codecs,
  repair atoms, postprocess atoms, archive packers, manifest writers, eval
  wrappers, and adjudicators each do one thing with typed byte contracts.
- Explicit lowering passes: representation -> prediction -> quantization ->
  hyperprior/residual -> arithmetic/range/Brotli -> deterministic archive pack.
- Profile-guided optimization: component traces and exact CUDA deltas feed
  atom planners, water-fill tables, and active-subspace search.
- Guarded execution: every low-level trick has a preflight or provenance check
  so bug classes become permanent friction rather than recurring failures.
- Reproducible final artifacts: archive bytes, SHA-256, exact JSON, component
  trace, manifest, hardware, logs, and failure classification travel together.

## Immediate Research Implication

The next meaningful score drop is unlikely to come from another scalar global
knob. The high-value direction is an atom compiler:

1. Rank pose/mask/postprocess/renderer atoms by marginal expected score benefit
   per charged byte.
2. Use exact component traces and public PR67 deltas as priors, but keep all
   outputs non-promotable until a complete archive is exact-evaluated.
3. Search low-dimensional subspaces inside the high-dimensional archive space:
   hard-pair windows, pose temporal bases, class/boundary atoms, mixed block
   radii, and foveated ego-motion charts.
4. Promote only stacked, deterministic archives that pass exact CUDA auth eval
   on T4/equivalent hardware.

## 2026-05-02T04:10Z Concrete Atom-Compiler Step

- Added `experiments/plan_scorer_weighted_pose_atoms.py` and focused tests as
  the first policy compiler for scorer-weighted pair atoms.
- Generated
  `experiments/results/pose_atom_plan_c059_20260502/pose_atom_policies.json`
  for C-059. It produced 256 ranked atoms and four non-promotable policies.
- The first H100 diagnostic uses the top32 policy as pair-window QP1 velocity
  search, not raw PVR1 residuals, because the current QP1 decoded non-velocity
  columns are zero. This is a concrete example of the stack protecting itself:
  the mathematical atom plan is mapped onto the actuator that actually changes
  the contest archive.

## 2026-05-02T04:25Z Council Return And Negative-Signal Reuse

The latest council/swarm returns sharpened the same compiler discipline:

- `charged_mask_grammar_ego_foveation_greenup_20260502_codex.md` defines the
  high-EV mask-grammar/ego-foveation program, but keeps it behind strict
  charged-payload accounting. PR70-style source constants and malformed ZIP
  behavior remain external forensics only.
- `atom_lagrangian_waterfill_sub03_system_20260501_codex.md` now has concrete
  atom allocation tooling for pose, mask, postprocess, renderer, pack, runtime,
  and selection-policy atoms. This is the mathematical control plane; exact
  complete-archive CUDA eval remains the truth.
- QZS4/global block-search is a scoped negative on the C-058/C-059 basin:
  the `qzs4_maskfirst_qp1` archive saved `3100` bytes versus C-059, but exact
  H100 diagnostic score was `1.5244097988910252` with PoseNet `0.156837`.
  The conclusion is not "QZS4 family killed"; it is "global byte-only block
  selection is outside the scorer basin unless mixed/local allocation is guided
  by component response."
- The active line of attack is therefore low-dimensional, scorer-weighted
  atom search: pair/window QP1 velocity atoms now, then mixed/local renderer
  block atoms and charged mask-boundary/foveation atoms once each has a
  complete archive builder.
- Long-running search opacity was promoted to a harness bug class:
  `line_search_pose_refinement.py` now emits basis-candidate progress telemetry
  before expensive objective calls, so future fast-chip searches can be
  monitored without guessing whether they are stuck.

This is the stack's self-protection loop: negative exact evidence does not
slow the system down; it becomes a constraint in the next compiler pass.

## 2026-05-02T04:19Z Dispatch-Custody Guardrail

Positive infrastructure signal:

- The Vast launcher now has an opt-in `--prefer-fast-chip` path for H100/H200/
  A100-first dispatch and a fail-closed explicit-anchor tarball path.
- This matters for the current score loop because C-059 follow-up line searches
  require exact source archives, policy JSON, and evidence files on the remote
  box. Recording those paths in metadata without shipping them would create a
  false reproducibility trail.

Negative/metabug retired:

- The old anchor hook attempted `tar -rzf` against a gzip tarball and then
  continued on failure. That was not safe for contest custody. It has been
  removed; anchors are now included in the deterministic positive file list or
  launch fails before remote spend.

Verification:

```text
py_compile: scripts/launch_lane_on_vastai.py, scripts/probe_fastest_chip.py,
  src/tac/tests/test_check_loop_session_extinction_pcc5_pcc8.py
focused pytest: 3 passed for disk floor, fast-chip flag, explicit anchor
  inclusion
git diff --check: clean for the touched launcher/test files
```
