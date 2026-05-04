# Nostradamus Future Frontier - 2026-05-04 Worker

Scope: anticipatory public-frontier research/design/build pass for PR92-PR100
style moves. No GPU work was dispatched. No runtime/inflate paths were edited.
This ledger is planning/evidence routing only unless it cites an exact CUDA
`contest_auth_eval` artifact.

## Current Exact Anchor

The local contest-faithful anchor is now the STBM1BR lossless PR85 mask recode:

- exact artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.adjudicated.json`
- score: `0.25369011029397787`
- archive bytes: `229756`
- archive SHA-256:
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- SegNet: `0.00057185`
- PoseNet: `0.0001894`
- samples: `600`
- hardware: Tesla T4 / CUDA, evidence grade `A++`

This supersedes raw PR85 as the internal score target, but PR85 remains the
semantic/runtime basin for static anatomy and atom-transfer work.

## Fresh Public PR Surface

Read-only GitHub API probe in this turn:

```bash
curl -L -s 'https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls?state=all&per_page=20&sort=created&direction=desc'
```

Observed latest PRs:

- PR94 `optimization_qpose_josema`, open, head
  `ae765553e9dc4bb12a93f4150a41f73ca3d9af16`, created
  `2026-05-04T07:13:19Z`.
- PR93 `flatpup`, open, head
  `887cccaccb376629829982660a1cf1ed06945bfc`, created
  `2026-05-04T06:23:10Z`.
- PR92 `qzs3_range_joint_r258 (0.26)`, open, head
  `95c711e8ec7a55e6cb066a0b4e20090391ccd2c2`, created
  `2026-05-04T06:23:05Z`.
- PR91 `Hpac coder hybrid`, open, head
  `77f958d24e55980d95e01e3e9767b5a94320ed43`.
- PR90 `add qrepro submission(0.28)`, open, head
  `cce857392701e73861ad513d34906faba523f719`.

PR95-PR100 were not present in the latest API page at the probe time.

Planner smoke using the new tool:

```bash
.venv/bin/python experiments/plan_nostradamus_future_frontier.py \
  --github-pr 90 --github-pr 91 --github-pr 92 --github-pr 93 --github-pr 94 \
  --json-out /tmp/nostradamus_pr90_pr94_plan.json
```

Important recomputed public-report math versus the STBM anchor:

| PR | Report class | Reported bytes | Recomputed score from body | Delta vs STBM anchor | Custody verdict |
| --- | --- | ---: | ---: | ---: | --- |
| PR91 | CUDA text, HPM1/HPAC | `222404` | `0.24879480490416128` | `-0.004895305389816584` | external only; local replay fails before score |
| PR92 | CUDA text, QZS3/range/joint | `236516` | `0.2587078229986317` | `+0.005017712704653843` | external and currently worse than STBM |
| PR90 | qrepro/STBM/QRGB text | `218080` | `0.2788721801656914` | `+0.02518206987171351` | external, architecture signal only |
| PR93 | CUDA text, flatpup/Quantizr | `284396` | `0.3204744681076375` | `+0.06678435781365966` | external, low frontier priority |
| PR94 | MPS text, qpose | `277087` | `0.33425141289817706` | `+0.08056130260419919` | MPS invalid for promotion |

## Top 5 Anticipated Competitor Moves

### 1. HPM1/HPAC Mask Entropy Replacement

Prediction: PR95-PR100 attempts will keep PR85/STBM non-mask segments and
attack the semantic-mask byte term with PR86-style HPAC/HPM1 token entropy
coding.

Exact evidence for why:

- PR91 claims PR85-equivalent components with `222404` bytes and score
  `0.24879480490416128`.
- Local PR91 anatomy shows only the mask segment changes versus PR85:
  PR91 `HPM1` mask is `145087` bytes; PR85 QMA9 mask was `159011` bytes,
  a `13924` byte mask opportunity versus PR85.
- Local exact replay of PR91 failed before any score JSON on both T4 hedge and
  L40S diagnostic with:
  `AssertionError: Tried to decode from compressed data that is invalid for the employed entropy model.`
- PR86 is merged and public, proving HPAC is a serious competitor family even
  though our local PR86 replay remains invalid before score.

Counter-design:

- Recover the exact HPM1 probability/token-generation contract first. Do not
  chase PR91 score text.
- Build an Apogee-owned HPAC/HPM1-like mask entropy coder over PR85/STBM token
  fields only after full decode and byte-exact reencode parity exist.
- Target neutral-component bytes below the STBM anchor. PR91's claimed `222404`
  bytes would be worth about `-0.0048954` score if SegNet/PoseNet remain flat.

Existing files/tools to use:

- `runtime-rs/crates/hpac-codec/`
- `experiments/profile_pr85_residual_sufficient_program.py`
- `experiments/plan_pr85_full_stack_opportunity_matrix.py`
- `experiments/preflight_pr85_fixed_runtime_readiness.py`
- `experiments/contest_auth_eval.py`

Readiness:

- Blocked. Required unblocker is full HPM1 decode plus byte-exact reencode
  parity under pinned entropy dependencies. Exact eval dispatch is forbidden
  until that blocker is removed and a Level-2 dispatch claim exists.

### 2. Semantic Geometry Mask Recode

Prediction: competitors will split semantic masks into road/topband/geometry
priors plus sparse residuals, because direct semantic masks are now the largest
byte lever.

Exact evidence for why:

- PR90 qrepro stores a compact `p` payload with a `152431` byte STBM1BR-style
  mask body, `56385` byte QFQ4 model body, and `4106` bytes of QRGB residual
  controls. Its reported score is worse than STBM, but the representation
  is a clear mask-geometry signal.
- Our corrected STBM1BR PR85 mask recode is exact positive: `229756` bytes,
  unchanged SegNet/PoseNet at JSON precision, score `0.25369011029397787`.
- The residual sufficient-program profile shows naive residual bitmaps are
  not enough: event-location cost dominates despite high predictor agreement.

Counter-design:

- Extend STBM only through parity-preserving geometry-aware mask coding.
- Use residual-density and row/span fields as training/profile signals for a
  learned entropy coder; do not dispatch direct residual maps from entropy
  fantasies.

Existing files/tools to use:

- `src/tac/stbm1br_mask_codec.py`
- `experiments/profile_pr85_residual_sufficient_program.py`
- `experiments/plan_pr85_full_stack_opportunity_matrix.py`
- `scripts/pre_submission_compliance_check.py`

Readiness:

- Ready for local planning and parity screens. Blocked for dispatch until a
  concrete archive changes charged mask bytes, passes decode/runtime parity,
  and has exact custody metadata.

### 3. PR85/QZS3 Range-Joint Near-Neighbor Search

Prediction: competitors will keep the PR85 QZS3/range-mask basin and search
tiny side-channel or joint-frame variants around it, because PR85 remains
component-stable.

Exact evidence for why:

- PR92 is a live open PR titled `qzs3_range_joint_r258 (0.26)`.
- PR92 body reports CUDA, PoseNet `0.00018963`, SegNet `0.00057675`, and
  `236516` bytes. Recomputed score is `0.2587078229986317`, worse than the
  STBM anchor by `+0.005017712704653843`.
- PR92 changed only `inflate.py`, `inflate.sh`, and `range_mask_codec.cpp`;
  the range codec blob SHA from the API matches the known PR85 range-mask
  codec lineage (`5a8f7a11...`).

Counter-design:

- Intake PR92 archive by SHA, diff raw wire segments against PR85/STBM, and
  mine only the changed side-channel semantics.
- Do not spend exact-eval budget on a PR92 clone; its own public report is
  already behind the STBM exact anchor.

Existing files/tools to use:

- `src/tac/pr85_bundle.py`
- `experiments/plan_frontier_stack_reconstruction.py`
- `experiments/preflight_pr85_fixed_runtime_readiness.py`
- `scripts/pre_submission_compliance_check.py`

Readiness:

- Ready for static intake after archive download. Blocked for candidate
  dispatch until raw segment diffs show a byte-closed, non-noop atom with
  break-even math against `0.25369011029397787`.

### 4. QPose / Tile-Action Pose Manifold Optimizers

Prediction: PR95-PR100 may revive qpose/tile-action optimization around the
PR75/PR79 lineage, possibly stacked onto newer mask streams.

Exact evidence for why:

- PR94 is a live qpose/tile-action style PR and includes an archive in the PR.
- PR94's report is MPS, not CUDA: PoseNet `0.00061985`, SegNet `0.00071020`,
  `277087` bytes, recomputed score `0.33425141289817706`.
- AGENTS.md explicitly treats public-floor QZS3/QP1/JFG pose search as an
  anisotropic manifold problem after scalar-radius gains flatten.

Counter-design:

- Keep pose search as an explicit charged action manifold: sparse delta sets,
  basis deltas, hard-pair temporal windows, and qpose residual atoms only.
- Use MPS/CPU reports as development hints, never promotion evidence.

Existing files/tools to use:

- `experiments/optimize_poses.py`
- `experiments/build_qp1_pose_active_subspace_candidates.py`
- `experiments/build_pr85_pair_action_candidates.py`
- `experiments/contest_auth_eval.py`

Readiness:

- Blocked. Existing pair-action lowering still needs grounded stream/value
  action evidence and local non-noop archive-changing paths before exact eval.

### 5. Tiny Bias/QRGB/Post Side Channels

Prediction: competitors will keep adding tiny RGB, bias, post, randmulti, or
region controls because the byte cost is small and public PRs show these
surfaces are easy to package.

Exact evidence for why:

- PR89 added a 300 byte `fb` final-bias side channel to PR85 but was withdrawn
  after public-master CUDA mismatch.
- PR90 uses about `4106` compressed bytes of QRGB residual controls.
- Our PR85 QRGB singleton wave produced exact negative evidence:
  PR85 QRGB candidates for pairs `0060`, `0164`, `0197`, and later randglobal
  `0192` all regressed on T4.

Counter-design:

- Preserve QRGB/final-bias tooling as signed atom-training and transfer
  machinery, not as an immediate PR85 stack.
- Reopen only if a new target basin, likely HPM1/STBM-derived, proves that
  the same tiny controls have different component sign.

Existing files/tools to use:

- `experiments/build_pr85_qrgb_pair_atom_archive_candidates.py`
- `experiments/build_pr85_qrgb_pair_atom_combo_candidates.py`
- `experiments/build_pr85_final_bias_stack_candidates.py`
- `experiments/plan_pr85_full_stack_opportunity_matrix.py`

Readiness:

- Measured negative on PR85 singletons. Blocked for combos until exact positive
  singleton evidence exists in a specific basin.

## Compliance And Custody Guardrails

- PR reports, PR titles, README scores, MPS reports, L40S diagnostics, CPU
  smokes, and public GitHub comments are external or diagnostic until a local
  exact CUDA `contest_auth_eval.adjudicated.json` exists for exact archive
  bytes and runtime tree.
- Source-embedded payload or dummy tiny-archive patterns remain forensic only.
  They are useful to harden guards, not to beat contest-faithfully.
- Any future GPU dispatch must first use `tools/claim_lane_dispatch.py claim`
  and must close the claim terminally after completion or failure.
- Do not edit scorer files or runtime/inflate paths for this forecast lane.

## Tool Built

Built:

- `experiments/plan_nostradamus_future_frontier.py`
- `src/tac/tests/test_plan_nostradamus_future_frontier.py`

Purpose:

- Convert live or saved public PR metadata into recomputed score math, likely
  innovation families, exact-custody blockers, and counter-design rows.
- Default behavior is stdout only. `--json-out` is opt-in and was used only
  under `/tmp` during this pass.

Highest-EV concrete tool/spec:

- Use `experiments/plan_nostradamus_future_frontier.py` as the first responder
  for every new PR92-PR100 style public submission. It is lower risk and more
  valuable than another archive builder right now because the main frontier
  blocker is classification/parity/custody, not lack of random candidate
  generation.

Verification:

```bash
.venv/bin/python -m py_compile experiments/plan_nostradamus_future_frontier.py src/tac/tests/test_plan_nostradamus_future_frontier.py
.venv/bin/python -m pytest src/tac/tests/test_plan_nostradamus_future_frontier.py -q
```

Result:

- `py_compile`: passed
- focused pytest: `4 passed in 0.10s`

