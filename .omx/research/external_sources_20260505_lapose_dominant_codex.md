# External Sources Design Ledger - LA-Pose Dominant Review - 2026-05-05

Author: Codex
Scope: research/design only, plus static scaffold in `src/tac/external_sources_20260505.py`.
Dispatch: none. Remote provider work: none. Upstream scorer patches: none.
Evidence status: design/scaffold only. No contest score claim is made here.

## Source Intake

### Primary: LA-Pose

Sources:
- Project page: https://la-pose.github.io
- Paper: https://arxiv.org/abs/2604.27448 and https://arxiv.org/pdf/2604.27448
- Wayve blog: https://wayve.ai/thinking/la-pose/
- OpenDV-YouTube dataset referenced by the project page: https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv

Read top to bottom. LA-Pose is directly relevant because it treats consecutive
driving frames as the self-supervised signal for ego-motion. The method uses a
Genie-style inverse dynamics encoder and forward dynamics objective to learn
latent actions from unlabeled driving clips, then discards the forward model and
attaches a lightweight pose head. The paper's architecture is especially useful
for this repository because the latent action is explicitly motion-centric,
feed-forward, and separable from the decoder used for pretraining.

Mechanically important facts:
- Input clips are 16 consecutive front-camera frames with randomized frame rate
  between 1 fps and 4 fps.
- The image tokenizer emits 15 x 7 x 1536 tokens per frame.
- The inverse dynamics path creates 15 latent action tokens, one per transition.
- A 1536 -> 50 -> 1536 MLP bottleneck is used during pretraining; the paper
  reports that a smaller 50-dimensional bottleneck improves metric-scale pose
  consistency versus a larger representation, despite worse reconstruction loss.
- Pose post-training predicts relative translation, quaternion rotation,
  field-of-view, and metric scale. The backbone-frozen version generalizes
  better than fine-tuning on the unseen PandaSet setting.
- Limitations include reverse motion and medium-curvature regimes, which need
  explicit hard-pair guards before any contest dispatch.
- Pretraining used an internal corpus of 10.2 million unlabeled driving snippets;
  code/model weights were not found on the project page, arXiv page, or source
  search during this review.

Off-the-shelf feasibility: not currently actionable as a vendored model. The
project page and paper are enough for an architecture clone or adapter design,
but no public LA-Pose implementation/checkpoint was found. OpenDV-YouTube is
useful as an in-the-wild driving-video prior, but its README says the processed
data is not directly distributed due to YouTube licensing; any use here would
need local reconstruction and a separate licensing/compliance review.

Contest coupling:
- PoseNet target path: train a LA-Pose-style inverse dynamics encoder on contest
  frames or public driving frames, freeze it, and fit a small head to frozen
  PoseNet pair targets extracted offline. The only archive payload may be
  charged model bytes, charged latent bytes, or a charged pose-prior sidechannel.
  Inflate must not load PoseNet or any scorer.
- HNeRV latent path: use the 50-dimensional latent action bottleneck as a
  compact conditioning stream for HNeRV/NeRV residual or renderer latents. The
  stream must have no-op controls: zeroed, shuffled, and original streams must
  produce different decoded-frame SHA evidence before any exact eval.
- Camera geometry/openpilot path: LA-Pose predicts relative pose, field-of-view,
  and metric scale, which maps cleanly to openpilot-style front-camera geometry
  constraints. This should be treated as a deterministic pose-basis prior, not as
  a learned hidden side channel.
- Sidechannel path: latent actions are only contest-faithful if their exact bytes
  are inside `archive.zip`, entropy-coded deterministically, and recorded in the
  archive manifest. A sidechannel outside the archive is invalid.

### Secondary Sources

MA-GIG:
- Paper: https://arxiv.org/pdf/2605.02167
- Code: https://github.com/leekwoon/ma-gig

MA-GIG adapts guided integrated gradients into latent space of a pretrained VAE.
The main transferable idea is not its image-classification benchmark, but its
failure model: input-space attribution paths can leave the data manifold, while
decoder-Jacobian latent paths better follow plausible data directions. For Pact,
this is a useful guard against brittle PoseNet/SegNet sensitivity maps and HNeRV
latent perturbation maps. Its MIT code is not a codec dependency; use the idea
for diagnostics only until exact archive evidence exists.

Graph Lottery Ticket:
- Paper: https://arxiv.org/pdf/2312.04762

The graph lottery ticket work argues that sparse graph backbones, often at
average degree around 2 to 5, can preserve graph-learning utility. The useful
contest translation is a deterministic sparse graph over frame pairs, latent
actions, or repair atoms. Seeded random spanning-tree tickets can preserve
connectivity without adding dense charged metadata. Any selected graph remains
planning evidence until component traces and exact archive eval close the loop.

Manifold learning survey:
- Paper: https://arxiv.org/pdf/2311.03757

The survey is a guardrail source. Neighborhood scale, intrinsic dimension,
tangent-space consistency, and sampling-density bias matter for interpreting
LA-Pose latent clouds. A t-SNE plot or nearest-neighbor graph is not evidence by
itself; it must be paired with seeded graph construction, local reconstruction or
target-prediction diagnostics, and exact archive evidence.

FloWM:
- Code: https://github.com/hlillemark/flowm
- Paper: https://arxiv.org/abs/2601.01075

FloWM frames self-motion and object motion as Lie-group flows for stable memory
under partial observation. It supports the LA-Pose conclusion that motion
structure should be first-class, but its released MIT code targets synthetic
world-model benchmarks and large checkpoint workflows, not contest inflate. Use
as a design prior for geometry-aware latent conditioning, not as drop-in runtime.

Goodfire VPD:
- Article: https://www.goodfire.ai/research/interpreting-lm-parameters
- Code: https://github.com/goodfire-ai/param-decomp

VPD decomposes model parameters into simple subcomponents and checks
mechanistic faithfulness with adversarial ablations. The useful contest pattern
is no-op resistance: renderer or latent sidechannel decompositions should prove
that selected bytes are causally used under adversarial ablations. The code is
MIT but LLM-oriented and not appropriate as an inflate dependency.

Architecture warm-up:
- Paper: https://openreview.net/pdf?id=DuNf2vPTTK

This source gives a concrete stability recipe for transformer pretraining:
online curvature tracking and progressively unlocking zero-initialized depth.
For any local LA-Pose clone, this is a training-stability option, not contest
runtime logic. It is especially relevant if a latent-action transformer is
trained from scratch and early loss spikes would otherwise waste compute.

Multiplicative Gaussian input masking:
- Blog: https://akyrillidis.github.io/aiowls/multiplicative_gaussian_input.html
- Paper: https://arxiv.org/abs/2602.17423

This source analyzes input-level multiplicative Gaussian masking and motivates
it as implicit regularization. The contest-safe application is narrow:
camera/frame-rate/input-channel noise regularization during offline latent-action
pretraining, with the noise scale recorded in the manifest. It does not validate
any contest score.

CauchyNet:
- Paper: https://arxiv.org/pdf/2510.10195

CauchyNet proposes compact holomorphic/inversion-based activations for
data-scarce function approximation and missing-data settings. This is a
low-priority inspiration for tiny pose residual functions or smooth sidechannel
decoders. Code is described as forthcoming, and complex-valued runtime would
need a small audited implementation before any inflate use.

## Top Implementation Recommendations

1. LA-Pose-style latent-action PoseNet target distillation

Implementation design:
- Build an offline inverse-dynamics encoder over contest frame pairs/windows.
- Use LA-Pose's 16-frame, 15-transition structure initially; support 2-frame
  compatibility for the scorer pair structure by packing local windows around
  pair indices.
- Freeze the motion encoder before fitting a small target head.
- Train the head against frozen PoseNet outputs extracted offline from GT frame
  pairs, never inside inflate.
- Export a charged model or charged pose-prior stream. Candidate archive members
  must be deterministic and manifest-recorded.

Why this is first:
- It attacks PoseNet directly without patching the scorer.
- It uses LA-Pose's strongest claim: compact self-supervised motion features
  transfer to pose estimation with little labeled data.
- It can start as a local diagnostic without GPU dispatch.

Required gates:
- Offline target manifest with frame sources, pair indices, seeds, target
  extraction command, tensor hashes, and frame-stride policy.
- Local smoke proving deterministic output from exact archive bytes and no
  scorer load at inflate time.
- Exact CUDA archive eval on `archive.zip -> inflate.sh -> upstream/evaluate.py`
  before any score or promotion claim.

2. LA-Pose bottleneck conditioning for HNeRV latent/residual streams

Implementation design:
- Encode each pair/window into a compact latent-action stream.
- Quantize and entropy-code the stream as charged archive bytes.
- Condition HNeRV/NeRV residual or renderer latents on the stream.
- Build no-op controls: original, zeroed, shuffled, and pair-permuted streams
  must change decoded frames and produce byte/semantic diffs before dispatch.

Why this is second:
- Current memory says HNeRV decoder weights and latent payloads dominate the
  archive mass; a motion bottleneck may reduce or better allocate that mass.
- The design composes with public frontier deconstruction and sidechannel
  arithmetic coding.

Required gates:
- Byte anatomy for raw latent bytes, coded latent bytes, histogram/model
  overhead, decoder bytes, and member SHA-256s.
- Decode-diff no-op guard with frame SHA evidence.
- Exact CUDA archive eval before claim.

3. Motion-manifold sparse atom planner

Implementation design:
- Build a deterministic graph over pair windows using LA-Pose-like latent-action
  distances, openpilot/camera geometry priors, and hard-pair component traces.
- Use seeded graph-lottery-ticket style sparse backbones to select connected
  motion atoms or repair windows.
- Use MA-GIG/manifold diagnostics to flag attribution paths that are likely
  off-manifold.
- Use Goodfire-style adversarial ablation framing to prove selected atoms are
  causally used, not metadata/no-op artifacts.

Why this is third:
- It is lower risk than adding a new runtime model, and it may improve existing
  alpha/pose-sensitive repair planners.
- It is scaffoldable as deterministic manifests and tests before any archive
  mutation.

Required gates:
- Seeded sparse graph manifest with node count, edge count, average degree, RNG
  seed, source pair IDs, and input hashes.
- CUDA component-trace cross-check against exact eval components.
- Exact source/candidate archive SHA matching and exact CUDA archive eval before
  dispatch or promotion.

## Prototype Scaffold Added

Files:
- `src/tac/external_sources_20260505.py`
- `src/tac/tests/test_external_sources_20260505.py`

The scaffold is a static registry of sources and ranked implementation lanes.
It intentionally does not import torch, touch scorers, build archives, or launch
jobs. Its validation enforces:
- LA-Pose is the dominant source.
- No lane may claim a score.
- No lane may allow remote dispatch.
- Forbidden actions include GPU dispatch, remote provider work, upstream scorer
  patching, uncharged sidecars, and score claims.
- Every lane must include an exact CUDA archive-eval gate.

## Recursive Adversarial Review And Greenup

Pass 1 - off-the-shelf reuse:
- Finding: LA-Pose is high impact but not drop-in. No public code or weights
  were found; original pretraining depends on internal Wayve-scale data.
- Greenup: scaffold records LA-Pose as design target only and blocks vendored
  code assumptions.
- Remaining blocker: re-check official code/model release before implementation.

Pass 2 - contest compliance:
- Finding: a latent-action sidechannel can silently become an invalid uncharged
  dependency if stored outside the archive or inferred from scorer outputs at
  inflate time.
- Greenup: top lanes require charged payload manifests and no scorer loads at
  inflate time.
- Remaining blocker: archive builder/inflate implementation does not exist yet.

Pass 3 - no-op risk:
- Finding: HNeRV conditioning and sparse atom planners can produce metadata
  churn without changing decoded frames.
- Greenup: required zero/shuffle/original latent no-op controls and
  adversarial-ablation framing.
- Remaining blocker: future implementation must add actual decode-diff tests
  once a candidate stream exists.

Pass 4 - evidence grade:
- Finding: all source claims are external or diagnostic; none can rank or promote
  a Pact contest archive.
- Greenup: static tests verify `score_claim=false`,
  `remote_dispatch_allowed=false`, and exact CUDA gates on all lanes.
- Remaining blocker: exact CUDA archive evidence is still required for any future
  claim.

Pass 5 - collision/worktree safety:
- Finding: shared worktree is heavily dirty.
- Greenup: only new files with `external_sources_20260505` or the dated ledger
  name were created.
- Remaining blocker: future integration into existing renderer/archive code
  should start with a fresh dirty-state check.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_external_sources_20260505.py -q
.venv/bin/python -m ruff check --fix src/tac/external_sources_20260505.py src/tac/tests/test_external_sources_20260505.py
.venv/bin/python -m ruff check src/tac/external_sources_20260505.py src/tac/tests/test_external_sources_20260505.py
```

Result:
- `5 passed`
- `ruff check`: all checks passed after import-order fix

## Next Concrete Work

Do not dispatch GPU jobs from this ledger. The next low-risk local patch should
be a manifest-only LA-Pose target-extraction design test:
- fixture frame-pair IDs
- deterministic window spec
- target tensor hash placeholder
- charged payload manifest schema
- explicit `score_claim=false`
- exact CUDA gate pending

## 2026-05-05 Follow-Up Research Agent Verdict

An xhigh LA-POSE research pass rechecked the project page, arXiv entry, author
page, GitHub/Hugging Face searches, and the Wayve blog. Verdict:

- No public LA-POSE code release or checkpoint was found.
- LA-POSE remains useful as a motion-prior and atom-allocation design, not as a
  drop-in runtime or checkpoint-backed lane.
- The reported full recipe is infeasible for contest-velocity replication:
  roughly dozens of H100s over multi-day pretraining/post-training, with
  private/internal driving data.
- The practical implementation should be LA-POSE-lite at compression time:
  deterministic per-pair/window latent-action features from available contest
  signals such as openpilot-style priors, RAFT/flow, HNeRV latent deltas, mask
  motion, and PoseNet target deltas.
- LA-POSE-lite must feed the meta-Lagrangian allocator and hard-pair routing,
  not rank or promote scores directly.
- No scorer, LA-POSE, Openpilot, LAPA, Waymo, YouTube-derived model, or hidden
  sidecar may load at inflate time.

Immediate integration decision:

- Continue the planning-only route already started in
  `src/tac/lapose_motion_atoms.py`.
- Add an evidence-ingestion bridge from CUDA component-response artifacts to
  LA-POSE motion records, preserving source archive SHA, evidence artifact SHA,
  device, and dispatch blockers.
- Keep all LA-POSE output `score_claim=false` until a charged archive builder
  consumes selected atoms and exact CUDA auth eval scores the resulting bytes.
