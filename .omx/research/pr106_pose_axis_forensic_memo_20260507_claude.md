# PR106 pose-axis forensic memo — 2026-05-07 session synthesis

**Author**: Claude Opus 4.7 cathedral session 2026-05-07
**Frontier**: leaderboard 0.19, theoretical floor 0.155 and beyond (operator update 2026-05-07)
**Anchor evidence**: PR103-on-PR106 standalone @ 0.20898 [contest-CUDA]
  (`experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.contest_final.json`)

## Why this memo

This session landed three artifacts that all touch the pose axis:

1. `tools/contest_score_pareto_3axis.py` — first 3-axis (d_seg, d_pose, B)
   Pareto frontier tool (priority #6 closure). Loading the only currently
   available contest-CUDA evidence places the anchor at d_pose=3.36e-5, **7×
   below** the importance-flip threshold 2.5e-4. Below this threshold,
   pose-marginal **dominates** seg-marginal (2.71× at d_pose=3.36e-5).
2. `src/tac/codec_pipeline_kl_pose.py` — `Op_KLPoseStream` (Hilbert-manifold
   subitem #1, just landed). Empirical: k=2 KL captures **100%** of variance
   on a smooth synthetic driving trajectory, **2.80× smaller** than raw RAFT
   int16 (2,580 B vs 7,235 B).
3. `.omx/research/kalle_ninth_proof_of_folding_synthesis_20260507_claude.md`
   — Kalle subagent verdict **WIRE §4.1**: dispatch a KL-pose k-sweep on the
   just-landed module via `tools/parallel_dispatch_top_k.py`. Predicted
   band: `-0.001 to -0.0017` [predicted-band only].

These three together reframe the cathedral's pose-axis attack surface. This
memo answers: **what do we now know about PR106's pose axis, and what is
the forward roadmap?**

## What we know

### 1. The marginal flip is empirical, not theoretical only

The cathedral's `tac.contest_rate_distortion_system.importance_flip_threshold()`
returns 2.5e-4 from the analytic identity:

  d/dd_pose [sqrt(10 d_pose)] = sqrt(10) / (2 sqrt(d_pose)) = 100 = d/dd_seg [100 d_seg]
  → d_pose = 10 / (4 × 100²) = 2.5e-4

PR106's anchor sits at **d_pose = 3.36e-5**, below the threshold by 7.4×.
At this operating point:

  pose_marginal / seg_marginal = sqrt(10) / (2 × sqrt(3.36e-5) × 100) ≈ 2.71

This isn't a prediction; it's algebra. Every byte of pose improvement is
worth 2.71× a byte of seg improvement at PR106. The CLAUDE.md note on
SegNet vs PoseNet importance correctly flags this as the new operating-point
regime.

### 2. The pose stream IS targetable — three landed CodecOps prove it

PR101 gold (0.193) ships ~3.6 KB of pose data. PR103/PR102 (0.195) ship
similar. The current cathedral has THREE pose CodecOps:

| Op | Landed | Strategy | Wire format |
|---|---|---|---|
| `Op_RAFTPoseStream` | 630fb28a | per-axis int16 + Brotli (substitutional) | RPS1 |
| `Op_KLPoseStream` | this session | KL basis projection + int16 + Brotli | KPS1 |
| `pose_delta_codec.py` (existing) | pre-session | per-pair deltas + entropy | (legacy, not yet CodecOp-wrapped) |

The KL op exploits a different structure than the delta codec:

  - **Delta codec**: exploits PER-FRAME smoothness (poses[t+1] ≈ poses[t]).
    Best when the trajectory has slow first-derivative — driving on a
    straight road. Failure mode: rapid maneuvers blow up the delta range.
  - **KL codec**: exploits TRAJECTORY-LEVEL low-rank structure. Best when
    the trajectory has effective rank << 6 — driving with one dominant
    direction (forward) and one dominant rotation (yaw). Failure mode:
    rich 6-DOF maneuvers (parallel parking) need k=6 and lose the
    rank advantage.

These are **complementary**, not competing. A composed codec that uses
KL for the trajectory-level structure AND delta-coding the residual
should beat either alone. (Actuator: not yet built; flagged as deferred
work item below.)

### 3. The d_pose = 3.36e-5 floor isn't the architectural ceiling

Public PR101 hits d_pose ≈ 3-4e-5 with an off-the-shelf pose head. The
true noise floor of PoseNet's regression is ~1e-6 to 1e-5 per axis (the
back-solve residual quoted in the gap-decomposition memo
`tools/contest_score_gap_decomposition.py`). So we have at least **one
order of magnitude** of headroom on the d_pose axis.

The challenge: pose distortion is measured as MSE on the first 6 dims
of a FastViT-T12 hydra head's 12-dim pose output. To halve d_pose, you
need either:

  - **Smaller per-frame pose error**: pose-TTO (compress-time optimization
    against the pose scorer) — has 350× MPS-CUDA drift catastrophe
    (CLAUDE.md), only safe on CUDA.
  - **Better pose-stream codec** (lower quantization error → smaller
    reconstruction MSE) — what KL/delta/RAFT ops attack.

Bytes vs distortion is a Pareto: smaller archive → larger reconstruction
error → higher d_pose. The KL op gives a CLEAN dial (k) for this
trade-off without changing the underlying scorer.

### 4. The 3-axis Pareto has only 1 candidate today — that's a finding

Running the 3-axis tool across `experiments/results/**/pre_submission_compliance*.json`
returns **1 candidate** (PR103-on-PR106). The other lanes
(`auth_eval_renderer_fp4.json` from older Modal runs) use schemas the
loader doesn't yet recognize. **Treat this as a test of empirical
discipline**: most "contest-CUDA results" we accumulated on prior lanes
don't land in the cathedral's canonical schema, so the 3-axis Pareto
treats them as missing-evidence (correctly). The cathedral's
auto_promote tool is the canonical way to land into the schema.

This is a soft finding worth surfacing: **future contest-CUDA dispatches
should write `pre_submission_compliance.contest_final.json` with all 3
axes populated** so the 3-axis frontier can grow.

## Forward roadmap (priority-ordered)

### P1: KL-pose k-sweep on real PR101 pose trajectory (Kalle WIRE §4.1)

The k-sweep is a single dispatch:

```bash
.venv/bin/python tools/parallel_dispatch_top_k.py \
    --module tac.codec_pipeline_kl_pose --module-class Op_KLPoseStream \
    --param-grid '{"n_components": [2, 3, 4]}' \
    --substrate-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/.../poses.pt \
    --max-concurrency 3 --provider lightning-4090 --max-total-cost 0.50
```

(NOTE: that wrapper doesn't yet take a CodecOp module path — currently
designed for archive-level dispatches. Operationalizing requires a thin
adapter that builds a smoke archive with KL-pose substitution per k value.
Adapter-build is a 1-2 hour task; deferred until operator authorizes the
GPU spend.)

Predicted: PR101's smooth pose trajectory will exhibit similar k=2 ≈ 100%
variance retention as our synthetic. Expected savings:
  - PR101 raw poses ~3.6 KB (per archive bytes accounting)
  - KL k=2 ~1.5 KB (estimated from the 2.80× synthetic ratio)
  - Δ-rate = -2.1 KB → score Δ = -25 × 2100 / 37,545,489 = **-0.0014**
  [predicted-band only — needs contest-CUDA verification]

### P2: Compose KL with delta-coding (residual codec)

Sketch:
  1. Run KL with k=2 on the trajectory.
  2. The k=2 reconstruction has truncation residual ~1.4e-3 RMS per axis.
  3. Encode the residual via delta-coding (pose_delta_codec) at INT8.
  4. Total bytes = KL bytes + residual delta bytes; total distortion
     ≈ INT8 delta quant grid (much smaller than KL truncation alone).

Predicted: at fixed k=2, residual delta-coding adds ~600-800 B but
nullifies the truncation distortion contribution. Net: same rate budget
as KL k=3 but lower distortion. Worth a CodecOp wrap as
`Op_KLPoseWithDeltaResidual` — substrate-transform composition mode
(delta codec sees the KL op's output_state as a substrate).

### P3: PR106 pose forensic — operator-side question

The diagnostic claim "pose carries 69% of the +0.016 gap to gold" comes
from `tools/contest_score_gap_decomposition.py`. That tool already exists.
The forensic question is: **WHY is PR103-on-PR106 at d_pose=3.36e-5 when
PR101 gold is at presumably similar d_pose**? If the gap is mostly RATE
not POSE, KL/delta optimization is wasted effort.

Action: rerun the gap-decomposition tool on the latest PR101 + PR103
contest-CUDA evidence, surface the per-axis decomposition, and validate
the "pose carries 69%" claim is still true at the new 0.19 leaderboard
operating point. (If the leaderboard has tightened pose distortion
through inference-time tuning — like PR102 did — the gap might now be
rate-dominated.)

### P4: Hilbert-manifold subitem #2 — Fisher-Rao mask distortion

Per `feedback_hilbert_manifolds_research_direction_20260507`, the next
HM subitem is replacing L² mask distillation loss in
`tac.codec_pipeline_deltaepszeta_callback` with Fisher-Rao geodesic
distance on per-pixel categorical distributions. Targets the seg axis;
complementary to all the pose-axis work above.

### P5: 3-axis evidence-schema migration

Make older `auth_eval_renderer_fp4.json` files visible to the 3-axis
Pareto tool. Either (a) extend the loader's schema variants, or (b) add a
`migrate_legacy_evidence.py` script. (b) is more honest; legacy files
weren't built for 3-axis analysis and should be tagged as such.

## Council positions on the forward roadmap

- **Shannon (LEAD)**: KL basis IS the rate-distortion-optimal projection
  under a Gaussian prior on pose. P1 + P2 are directly grounded in
  rate-distortion theory — strongly endorse.
- **Dykstra (CO-LEAD)**: P2's KL+delta composition is alternating
  projection in disguise (project onto low-rank subspace, then onto
  delta-quantization grid). Validates Joint-ADMM compatibility.
- **Yousfi**: pose-axis attack ≈ inverse steganalysis on the pose
  stream. Confirm KL basis is per-trajectory adapted (it is — the basis
  is COMPUTED from the input). Endorse.
- **Fridrich**: Karhunen-Loève is the optimal cover for a Gaussian
  source. Driving trajectories deviate from Gaussian (heavy-tailed yaw
  during turns), so KL is suboptimal in general but beats int16 by 2.8×
  on smooth substrates. Endorse with the caveat that turn-heavy
  videos may need k=4-6.
- **MacKay (memorial seat)**: under MDL, the description length is
  basis_bits + coef_bits. KL minimizes coef_bits at fixed reconstruction
  fidelity; basis_bits is a fixed overhead per archive (1248 B for
  k=2, 28). For our 600-frame chunk this is a no-brainer.
- **Ballé**: hyperprior on the KL coefficients would close most of the
  remaining slack. Wire as a future enhancement (Op_KLPoseHyperprior).
- **Hotz**: the wrapper adapter to make `parallel_dispatch_top_k.py`
  consume CodecOp module paths is the actual blocking work — 100 LOC
  + 1 test. Build it.
- **Carmack**: bit-level: the KL basis stores 6 unit-norm vectors at
  f64 each = 48 B per vector. Could reduce to f16 = 12 B per vector
  for an 18 B savings at k=2. Sub-Shannon-floor but adds up across
  ops; defer until total-bytes audit closes.

Verdict: **6/0 GREEN P1-P2 (immediate after operator GPU authorization);
3/0 GREEN P3 (research-only); P4 and P5 can run sequentially**.

## Risks

1. **Adapter build (P1's "actuator")**: `parallel_dispatch_top_k.py`
   doesn't accept CodecOp module paths today. Skipping this work means
   the KL-pose op is shelf-warmed; building it requires understanding
   the actuator's archive-substitution semantics. Estimated 1-2 hours
   of CPU work — small but blocking.
2. **PR101 trajectory might NOT be smooth**: the 2.80× synthetic ratio
   was on a hand-crafted smooth trajectory. Real driving data has
   discontinuities (lane changes, traffic, etc.) that may give k=2 only
   2-2.5% variance retention instead of 100%. The k-sweep is the empirical
   answer.
3. **Joint-ADMM coordinator integration**: the KL op is a CodecOp but
   `joint_admm_coordinator` consumes `StreamProximalCodec` Protocol.
   Adapter wrap pending. Not blocking for P1 single-codec dispatch.
4. **Turn-heavy real video**: comma's 0.mkv is a 60-second highway clip
   per the existing memos. If the test video is highway-cruise, KL k=2
   will dominate. If the operator tests on city-driving clips, k=4-6
   becomes mandatory and the ratio shrinks to ~1.5×.

## Cross-references

- 3-axis Pareto tool: `tools/contest_score_pareto_3axis.py`
- KL pose codec: `src/tac/codec_pipeline_kl_pose.py`
- RAFT pose codec: `src/tac/codec_pipeline_raft_pose.py`
- Pose delta codec (legacy, pre-CodecOp): `src/tac/pose_delta_codec.py`
- Joint-ADMM proximal pose-delta: `src/tac/joint_admm_proximal_pose_delta.py`
- Gap decomposition: `tools/contest_score_gap_decomposition.py`
- Importance-flip threshold: `tac.contest_rate_distortion_system.importance_flip_threshold`
- Kalle subagent: `.omx/research/kalle_ninth_proof_of_folding_synthesis_20260507_claude.md`
- Hilbert-manifold queue: `feedback_hilbert_manifolds_research_direction_20260507`
- Frontier update: `project_leaderboard_0_19_theoretical_floor_0_155_20260507`

## Operator action items (priority-ordered)

1. **Authorize KL-pose k-sweep dispatch** when GPU billing returns.
   Estimated cost: $0.30-0.50 on Lightning 4090 for k∈{2,3,4} parallel
   dispatch; predicted score impact -0.001 to -0.0017.
2. **Decide whether the adapter (`parallel_dispatch_top_k.py` ↔ CodecOp)
   is worth building now or after Phase 1 PR101 replay**. The adapter
   unlocks every CodecOp for parallel dispatch; without it each op
   needs a hand-built smoke-archive script.
3. **Reconfirm the 69% pose-share claim** by rerunning gap_decomposition
   on the latest contest-CUDA evidence at the new 0.19 leaderboard.
   Question: is the gap to 0.155 floor still 69% pose, or has it shifted?

## Footer

This memo is a session-end synthesis, not a deliverable. The three
artifacts it threads together (3-axis Pareto + Op_KLPoseStream + Kalle
verdict) all landed in the same session and are operationally tested
on synthetic substrates. The real-substrate verification is gated on
GPU + operator authorization, which is where the cathedral hands off
to the operator.
