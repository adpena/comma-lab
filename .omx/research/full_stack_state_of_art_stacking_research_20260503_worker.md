# Full-Stack State-Of-Art Stacking Research - 2026-05-03 Worker

Scope: focused research and local synthesis. No remote GPU jobs were dispatched,
no lane claim was opened, and no new score is claimed here. Existing exact JSONs
and ledgers are cited only as prior local evidence.

## Sources

Public challenge and leaderboard:

- comma leaderboard, refreshed 2026-05-03:
  https://comma.ai/leaderboard
- comma video compression challenge README and rules:
  https://github.com/commaai/comma_video_compression_challenge
- PR67 `qpose14_r55_segactions_minp`:
  https://github.com/commaai/comma_video_compression_challenge/pull/67
- PR75 `qpose14_r55_segactions_minp` follow-up:
  https://github.com/commaai/comma_video_compression_challenge/pull/75
- PR65 `henosis_qz_n3z_r25_clean`:
  https://github.com/commaai/comma_video_compression_challenge/pull/65
- PR70 `mask_decoder` exploit-forensics signal:
  https://github.com/commaai/comma_video_compression_challenge/pull/70

External research:

- Ballé et al., variational image compression with scale hyperprior:
  https://arxiv.org/abs/1802.01436
- CompressAI, learned compression library and model/eval platform:
  https://github.com/InterDigitalInc/CompressAI
- Lu et al., DVC end-to-end deep video compression:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.html
- Li et al., DCVC deep contextual video compression:
  https://arxiv.org/abs/2109.15047
- Chen et al., NeRV neural representations for videos:
  https://arxiv.org/abs/2110.13903
- Chen et al., HNeRV hybrid neural representation for videos:
  https://arxiv.org/abs/2304.02633
- Liu et al., task-oriented image semantic communication:
  https://arxiv.org/abs/2201.10929
- Duda, asymmetric numeral systems:
  https://arxiv.org/abs/1311.2540

Local control-plane and evidence:

- `AGENTS.md`
- `.omx/research/contest_faithful_swarm_execution_20260502_codex.md`
- `.omx/research/c067_byte_self_compression_opportunity_review_20260503_codex.md`
- `.omx/research/renderer_shrink_pr75_c088_20260503_worker.md`
- `.omx/research/pr75_qp1_nextwave_sub314_20260503_worker.md`
- `.omx/research/sjkl_pr75_qp1_sub314_20260503_worker.md`
- `.omx/research/yousfi_fridrich_field_equations_20260502_codex.md`
- `.omx/research/works_negatives_hardened_stack_20260502_codex.md`
- `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/contest_auth_eval.adjudicated.json`
- `experiments/results/lightning_batch/exact_eval_c082_p6_stream_resweep_276333_t4_20260503T0705Z/contest_auth_eval.adjudicated.json`
- `experiments/results/lightning_batch/exact_eval_pr75_qp1_public_replay_t4_20260503T0608Z/contest_auth_eval.adjudicated.json`

## Public Frontier Read

The official leaderboard currently rounds the top public entries as:

| rank read | public entry | official rounded score | PR |
| --- | --- | ---: | --- |
| 1 | `qpose14_qzs3_filmq9g_slsb1_r55` | `0.32` | `#67` |
| 2 | `henosis_qz_n3z_r25_clean` | `0.32` | `#65` |
| 3 | `qpose14` | `0.32` | `#63` |
| 4 | `unified_brotli` | `0.33` | `#64` |
| 5 | `quantizr` | `0.33` | `#55` |

PR bodies expose more useful component data than the rounded leaderboard:

- PR67/PR75 report the same archive SHA
  `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`,
  bytes `276741`, PoseNet `0.00048653`, SegNet `0.00060686`, rounded score
  `0.31`. PR75 is an open follow-up with embedded tile-action corrections;
  PR67 is merged.
- PR65 initially reported bytes `284425`, PoseNet `0.00035283`, SegNet
  `0.00070896`; the later GitHub eval-bot block reports PoseNet `0.00049291`,
  SegNet `0.00060138`, same bytes, rounded score `0.32`. Treat it as public
  anatomy and basin signal, not a directly transferable stream.
- PR70 reports `0.19` at `57329` bytes, but its author states the payload
  moved bytes into `inflate.py`. This is compliance forensics, not a contest-
  faithful target.

The challenge README confirms the score contract:

```text
score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489
```

and the submission contract is `archive.zip` plus `inflate.sh`, with all needed
assets charged and a 30 minute-ish T4-class evaluation budget.

## Local Exact Anchor And Gap Math

The strongest checked local exact artifact in this pass is prior evidence:

- Candidate: `c067_pr75_qp1_top40_p6`
- Exact JSON:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/contest_auth_eval.adjudicated.json`
- Canonical score recomputed from components: `0.3154707273953505`
- Archive bytes: `276342`
- PoseNet: `0.00049601`
- SegNet: `0.00061038`
- Samples: `600`
- Archive SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- Runtime tree SHA-256:
  `1a9d08972804a57bcce81ab41baa2972b04e949d767c88c7c0310ad26abd2c89`

Two nearby exact facts matter for experiment selection:

- `c082_p6_stream_resweep_276333` is smaller at `276333` bytes but scores
  `0.3154874419767294`, so byte-only micro-wins can lose when component
  terms shift.
- Local PR75 replay scored `0.31550098128455195` at `276741` bytes with
  PoseNet `0.00049378`, SegNet `0.00060961`, and a different runtime tree.
  Same archive bytes with different runtime custody should not be interpreted
  as a pure archive comparison.

Rate value is `25 / 37545489 = 6.658589531e-7` score per byte. From
`0.3154707273953505`, unchanged-component byte savings needed are:

- Sub-`0.314`: about `2209` bytes.
- Sub-`0.300`: about `23234` bytes.

Therefore action-stream and packer work can plausibly cross sub-`0.314`, but
sub-`0.300` needs either a renderer-stream step change or material PoseNet/
SegNet component movement.

## Top-Submission Anatomy

PR75/PR67 are the relevant public anatomy. The downloaded PR75 archive is a
single ZIP member `p`:

| slice | charged bytes | decoded bytes | decoded SHA-256 |
| --- | ---: | ---: | --- |
| mask Brotli | `219472` | `223385` | `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb` |
| renderer/model Brotli | `56034` | `59288` | `2333284a73446c3b323948fb883ade0f677baf9ad5d9d06aa1da7bec337bd9c9` |
| SegNet tile actions | `236` | `268` | `bfd46b2b481a5064cc1f64b7b1288640c51b89ad6aeb5598408150f7945eac15` |
| QP1 pose Brotli | `899` | `1140` | `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc` |

Stack lessons:

- The winning public family is mask stream + tiny JointFrameGenerator/QZS3-ish
  renderer + QP1 pose stream + single-blob deterministic packing + score-aware
  pose/tile-action corrections.
- The mask stream dominates charged bytes, but exact-lossless replacement is
  not currently available; generic nested recompression and naive lossless mask
  recoding were byte-regressive locally.
- Renderer/model bytes are the only large non-mask stream. Public Selfcomp and
  NeRV/HNeRV-style work make this the highest-EV compression target.
- PR75 tile actions transfer as a real charged atom class, but current P6
  variants are in the `1e-5` improvement regime after exact T4 replay.
- Direct PR65 qpost/side-stream transfer and SJ-KL residual stacking are not
  immediate sub-`0.314` paths on current evidence.

## SOTA Synthesis For This Contest

Learned image/video compression: hyperpriors, context models, DCVC conditional
coding, and CompressAI's rate-distortion tooling all point to the same local
rule: encode side information only when its charged bytes reduce the exact task
score. A general learned codec is too heavy for the final inflate path unless
the decoder is already present or tiny; the useful transplant is the objective
shape, not a large off-the-shelf model.

Neural video representation and self-compressing networks: NeRV/HNeRV validate
storing a fixed video as weights and then compressing the network. The contest
already uses this pattern with a small renderer. The open opportunity is not
"use a bigger INR"; it is trained self-compression, slim student distillation,
or sparse/low-bit export that keeps the scorer basin.

Task-aware and semantic compression: this challenge is explicitly compression
for machines. Pixels only matter through fixed SegNet/PoseNet. Local byte
allocation should optimize the task components directly, with exact CUDA as
the final oracle. Human visual quality and proxy CPU/MPS similarity are only
development filters.

Low-rank, active-subspace, and Lagrangian water-fill: the local planner is
aligned with SOTA dimension-reduction practice. Search should operate on
charged atoms with marginal benefit per byte: pair windows, temporal pose
bases, boundary bands, tile actions, renderer tensor blocks, and foveated/
ego-motion charts. The exact archive is the optimizer checkpoint.

Entropy coding: ANS/range coding can beat Brotli for structured small streams,
but only if decoder bytes, tables, determinism, and runtime are charged. For
near-term contest work, custom varint/P6/Brotli variants are lower risk unless
an ANS decoder is already amortized across multiple payloads.

## Local Stack Opportunities

High EV:

- Renderer self-compression: learned bit depth, trained sparse recovery, and
  slim/factorized teacher-student JFG. This is the only no-mask-geometry lever
  with enough bytes for sub-`0.300` rate movement.
- Renderer parity shrink: the current naive QZS3 reblock screen saves up to
  `2971` bytes but fails local pose-safety parity. The builder/preflight path
  is valuable; the unconditioned candidates are not dispatchable.

Medium EV:

- PR75 tile-action compiler v2: exact T4 shows the action stream transfers,
  but ranking noise is high and expected movement is micro unless a learned
  dictionary or second-order interaction model changes the action basis.
- QP1 pose active-subspace refinement: PoseNet remains flat but not exhausted;
  scorer-weighted pair windows and DCT/spline bases are more defensible than
  scalar sweeps.

Blocked but important:

- Mask geometry / Yousfi-Fridrich field policies have large rate headroom, but
  CMG3A body200 exact evidence saved `18538` bytes while PoseNet collapsed.
  No remote dispatch should happen without a new geometry-escape proof.

Lower priority:

- SJ-KL residuals are byte-additive in the PR75/QP1 stack and need implausibly
  large component gain relative to measured C067 response.
- PR65 side streams remain useful public anatomy, not a direct transplant plan.

## Top 5 Contest-Faithful Next Experiments

### 1. RENDERER-SC-LBD - learned bit-depth renderer self-compression

Expected movement: if it saves only `3000` to `8000` bytes at unchanged
components, score moves about `-0.0020` to `-0.0053`, enough for sub-`0.314`.
If the public Selfcomp-like density is reachable and the renderer stream drops
by `32000` to `40000` bytes, rate movement is about `-0.0213` to `-0.0266`,
large enough to matter for sub-`0.300`.

Wall-clock: `1-3h` if a trained burn artifact can be harvested and exported;
`6-12h` if a new short training pass is needed.

Needed implementation: consume the fixed-renderer burn or train per-tensor/per-
channel bit-depth with explicit rate loss. Preserve `masks.mkv`, QP1 poses, and
tile actions byte-for-byte. Export a deterministic, charged, pickle-free or
reviewed-safe runtime payload. Run transplant byte closure and
`preflight_renderer_transplant_pose_safety.py`.

Exact eval route: after a future lane claim, run the canonical
`archive.zip -> inflate.sh -> upstream/evaluate.py` path through
`experiments/contest_auth_eval.py --device cuda`, then adjudicate
`contest_auth_eval.adjudicated.json`.

Adversarial risks: compressed weights can leave the PoseNet basin while looking
visually close; decoder metadata can erase byte savings; any scorer load,
sidecar, or unsafe torch-load fallback is a compliance failure.

### 2. RENDERER-STUDENT-JFG - slim/factorized teacher-student renderer

Expected movement: a `30KB` to `34KB` renderer payload would save roughly
`22KB` to `26KB` against the current `~56KB` renderer stream, for rate movement
around `-0.0147` to `-0.0173` if components survive. That is near the
sub-`0.300` byte target but component survival is the hard part.

Wall-clock: `6-18h` training/export cycle plus `30-60m` local custody and
parity validation.

Needed implementation: distill a narrower/factorized JFG from the current
exact renderer on the contest window with fixed masks, poses, and actions.
Record architecture, parameter count, seeds, losses, exported bytes, and
runtime hashes. Prefer factorized or grouped layers only where deterministic
export already exists.

Exact eval route: same contest auth eval route after lane claim and local
preflight. Include runtime tree SHA in any comparison because architecture
changes affect inflate custody.

Adversarial risks: visible reconstruction parity is not scorer parity; small
geometry changes can collapse PoseNet; extra runtime code may consume the
saved bytes; half-frame/full-frame indexing must stay guarded.

### 3. PR75-ACTION-DICT-V2 - second-order tile-action dictionary

Expected movement: `-0.00005` to `-0.00035` is realistic from current T4
signals. This can improve the frontier or help cross sub-`0.314` when combined
with renderer shrink, but it is not a standalone sub-`0.300` path.

Wall-clock: `1-3h` local candidate generation and trace replay; exact T4
promotion later only for the best byte-closed candidates.

Needed implementation: extend the existing tile-action subset builder into a
small action dictionary or signed-amplitude basis selected by exact T4 traces,
with no-op/source-preserving policies penalized in the manifest. Keep P6/P7
wire contracts deterministic and cheap.

Exact eval route: exact CUDA auth eval on the complete archive only; use
diagnostic traces for queue ordering, not claims.

Adversarial risks: T4 reorders RTX/L40S trace ranking at this scale; pair/tile
interactions are nonadditive; action changes can trade SegNet for PoseNet in a
way local proxies miss.

### 4. QP1-POSE-ACTIVE-SUBSPACE - scorer-weighted pose basis search

Expected movement: `-0.0002` to `-0.0009` if hard-pair windows or DCT/spline
pose bases can move PoseNet from about `0.000496` toward the better public
PR75 pose basin without adding more than a few hundred bytes.

Wall-clock: `1-4h` local/diagnostic search for a small candidate set; exact T4
later for byte-closed finalists.

Needed implementation: use scorer-weighted pair/window priors, temporal basis
functions, and QP1-preserving float32 decode semantics. Emit policies with
changed-pair counts, byte cost, expected component movement, and failure class.

Exact eval route: complete archive exact CUDA only. Pose proxy or CPU/MPS
output cannot promote or retire a candidate.

Adversarial risks: current basin is flat; QP1 precision and runtime custody can
dominate small changes; pose improvements may regress SegNet or inflate bytes.

### 5. MASK-GEOMETRY-ESCAPE-PROOF - local proof before CMG/YF dispatch

Expected movement: if a geometry policy saves `2209+` bytes with component
parity, it crosses sub-`0.314`; `18500+` byte saves show the rate headroom is
real. Current exact evidence, however, shows PoseNet collapse, so expected
score movement is option value until a strict local proof passes.

Wall-clock: `4-8h` local proof and builder work, no remote dispatch.

Needed implementation: build a geometry-escape proof around connected
components, boundary-DCT atoms, pose-conditioned foveation, and local parity
thresholds. Enforce duplicate-atom, unmatched-atom, base-run, decompression,
magic, and body-search manifest checks.

Exact eval route: no exact eval until the local proof reports safe. After a
future claim, run exact CUDA auth eval on one deterministic archive and classify
as scoped evidence, not a broad family conclusion.

Adversarial risks: CMG/YF body screens are nonmonotone and already have exact
negative evidence; large byte wins can be pure PoseNet cliff maps; base-run
mismatch or nonmonotone body selection can create misleading dispatch targets.

## Recommendation

Spend minimum wall-clock in this order:

1. Harvest/export any trained renderer self-compression burn and run the
   transplant pose-safety gate.
2. If no usable burn exists, start a short learned-bit-depth or slim-student
   renderer pass with fixed PR75/C088 masks, QP1 poses, and tile actions.
3. In parallel locally, prepare a tiny PR75 action-dictionary v2 candidate set
   to pair with any renderer-shrink byte win.
4. Run QP1 pose active-subspace search only as a bounded component-side hedge.
5. Keep CMG/YF mask geometry local-only until a new geometry-escape proof
   passes; do not dispatch body-byte screens.

These recommendations preserve contest compliance, paper/OSS usefulness, and
production-style deterministic archive contracts while targeting the few
remaining levers with enough byte or component headroom.
