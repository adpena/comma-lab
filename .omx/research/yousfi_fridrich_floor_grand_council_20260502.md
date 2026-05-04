# Yousfi-Fridrich Floor Grand Council - 2026-05-02

Scope: read-heavy/write-scoped council memo for the contest-faithful comma
video compression challenge push.

Write scope honored: this file only.

Score claim: `false`.

Evidence grade: `derivation` + `external` + `empirical/design_review`.

Council role labels below are analytic personas, not claims that any named
outside person participated. The names are shorthand for review disciplines:
Shannon/information theory, Fridrich/steganalysis and payload accounting,
Yousfi/contest compliance, Tao-style mathematical adversarial review,
Dykstra/Boyd projection and allocation, compiler engineering, and contrarian
failure analysis.

## Current Anchor

Active exact internal frontier:

```text
label:          C-067
score:          0.31561703078448233
archive bytes:  276214
archive sha256: 226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
PoseNet:        0.00049637
SegNet:         0.00061244
samples:        600
hardware:       Tesla T4 CUDA
artifact:       experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json
```

C-067 is a charged fixed-slice archive using the PR67 mask segment plus C-059
model/pose bytes. The score is local exact T4 evidence for our archive bytes;
the PR67 mask source remains externally attributed.

Current decomposition, using the contest formula and rounded component fields:

```text
lambda_rate_per_byte = 25 / 37545489 = 6.658589531221714e-7
rate contribution    = 0.18391956487768746
SegNet contribution  = 0.061244
PoseNet contribution = 0.07045353078448233
distortion total     = 0.13169753078448233
```

At C-067-like distortion, score targets imply:

| target score | max bytes at same distortion | required byte cut from C-067 |
|---:|---:|---:|
| `0.315` | `275287` | `927` |
| `0.310` | `267778` | `8436` |
| `0.300` | `252760` | `23454` |
| `0.250` | `177669` | `98545` |
| `0.200` | `102578` | `173636` |

Immediate implication: `0.30` is reachable by a moderate rate win if the PR67
geometry basin survives. `0.25` or lower requires a real mask/decoder/semantic
sufficient-statistic change, not another scalar pose or container polish.

## External Sources Used

- PR67 `qpose14_qzs3_filmq9g_slsb1_r55`: reports CUDA 600-sample score `0.31`,
  PoseNet `0.00048597`, SegNet `0.00061000`, `276564` bytes, and describes
  QZS3 grouped quantization, delta+VLQ pose coding, and single-blob packing:
  https://github.com/commaai/comma_video_compression_challenge/pull/67
- PR65 `henosis_qz_n3z_r25_clean`: reports PoseNet `0.00035283`, SegNet
  `0.00070896`, `284425` bytes, rounded score `0.32`, and author-tracked
  exact local score around `0.31968005`:
  https://github.com/commaai/comma_video_compression_challenge/pull/65
- PR55 Quantizr: public `0.33` JointFrameGenerator/Quantizr basin source:
  https://github.com/commaai/comma_video_compression_challenge/pull/55
- PR56 Selfcomp: records soft-LUT/SegNet fitting, affine learned-image pose,
  and self-compressed weights:
  https://github.com/commaai/comma_video_compression_challenge/pull/56
- PR68 loophole proof: explicitly moves payload bytes into script side and asks
  not to be leaderboarded; useful only as compliance hardening:
  https://github.com/commaai/comma_video_compression_challenge/pull/68
- PR70 mask decoder: reports `0.19` but explains that bytes were moved from
  archive into `inflate.py`; useful only as exploit forensics:
  https://github.com/commaai/comma_video_compression_challenge/pull/70
- NeRV: neural video as overfit network, motivating charged learned decoders:
  https://arxiv.org/abs/2110.13903
- HNeRV: content-adaptive embeddings and faster convergence for neural video
  representations:
  https://arxiv.org/abs/2304.02633
- Scale hyperprior learned compression: side information and entropy modeling
  as trainable compression variables:
  https://arxiv.org/abs/1802.01436
- Joint autoregressive/hierarchical priors: complementary entropy models:
  https://arxiv.org/abs/1809.02736
- Learned compression for machine perception and task-aware compression:
  https://arxiv.org/abs/2111.02249
  https://arxiv.org/abs/2206.05650
  https://arxiv.org/abs/2409.19184

## Mathematical Objective

The strict objective is not human-video Shannon coding. It is the
Yousfi-Fridrich floor:

```text
minimize over charged programs p:

  S(p) = 100 * seg(D_fixed(p))
       + sqrt(10 * pose(D_fixed(p)))
       + lambda * |archive_zip(p)|_bytes

where lambda = 25 / 37545489,
      D_fixed is fixed contest inflate code plus charged archive payload,
      and exact CUDA auth eval is the only score oracle.
```

This can be lower than generic perceptual-video intuition because the decoder
only needs enough sufficient statistics to satisfy fixed SegNet/PoseNet on this
fixed video. It is not lower because of uncharged payload tricks. Every
per-video table, learned weight, postfilter parameter, mask grammar atom, pose
stream, codebook, entropy table, or side-channel must be charged inside
`archive.zip` unless it is fixed contest code.

Useful derived forms:

```text
distortion_budget(target, bytes) = target - lambda * bytes
rate_saving_score_delta(delta_bytes) = lambda * delta_bytes
break_even_atom(a) = expected_component_gain(a) > lambda * charged_bytes(a)
```

The current C-067 distortion total is about `0.13169753`. To reach `0.30`
without changing distortion, the archive must be about `252760` bytes or less.
To reach `0.25` without changing distortion, it must be about `177669` bytes or
less. Therefore:

- 0.30 path: any geometry-preserving mask/packer cut of `24KB` is enough.
- 0.25 path: needs `98KB` at same distortion, or coupled distortion gains plus
  a smaller byte cut.
- 0.20 path: needs a true sufficient-statistic decoder or exploit-like rate,
  but strict compliance forbids uncharged script payload relocation.

## Lagrangian And Water-Filling Formulation

Define candidate atoms over mask, pose, renderer, packer, postfilter, runtime,
and learned-decoder state:

```text
a = {
  stream,
  scope,
  charged_bytes,
  expected_delta_seg,
  expected_delta_pose,
  expected_delta_rate,
  uncertainty,
  interaction_keys,
  evidence_grade_required
}
```

For a current archive `A` and selected atoms `X`, minimize:

```text
L(X; mu) =
    100 * seg(A + X)
  + sqrt(10 * pose(A + X))
  + lambda * bytes(A + X)
  + mu_compliance * I_noncompliant
  + mu_runtime * max(0, inflate_time - 30min)
  + mu_repro * I_unreproducible
```

In planning, use the conservative marginal estimate:

```text
accept atom a if:

E[-Delta_component_score(a | X)] >
  lambda * charged_bytes(a | X)
  + uncertainty_penalty(a)
  + interaction_penalty(a | X)
```

This is water-filling: spend bytes first on the atoms with highest expected
component benefit per charged byte. A hard pair is therefore not "large error";
it is high opportunity density:

```text
hardness_i =
  E[score saved by repairing pair i] / E[charged bytes to repair pair i]
```

The Dykstra-style operating loop is:

```text
A_{k+1} =
  P_exact_eval P_runtime P_compliance P_pose P_seg P_rate (A_k + selected_atoms)
```

This is a discipline, not a proof. The feasible sets are nonconvex because the
renderer, PoseNet, SegNet, quantizers, and entropy coder interact. A stacked
archive is a new hypothesis and needs its own exact CUDA eval.

## Low-Dimensional Subspaces Of High-Dimensional Atom Manifolds

The high-dimensional mask tensor and renderer weights should not be optimized
as arbitrary pixels/parameters. The high-EV route is to search low-dimensional
charts whose coordinates are physically, temporally, or scorer-aligned:

- Pose charts: scalar radius, anisotropic per-axis radius, pair-window
  velocities, DCT/spline temporal bases, jerk-limited updates, log-zoom/ego
  flow modes, hard-pair local windows.
- Mask charts: row-run depth, global row-run atom budget, connected
  components, class-boundary splines, horizon band, lane-like elongated
  components, foveal ellipses centered near the scorer-space vanishing point.
- Renderer charts: mixed QZS3 block size per tensor/group, grouped bit depth,
  per-layer FP4 codebook, residual low-rank adapters, tiny FiLM side channels,
  EMA-vs-live export selection.
- Postfilter charts: pair/class/region-gated residuals, deterministic
  convolution kernels, learned LUTs, foveated correction maps, identity outside
  selected atoms.
- Entropy charts: run-length versus delta coding, length-table coding,
  arithmetic/range coding, order/model selection, single-blob member layout.

Each chart must emit a deterministic archive candidate and a machine-readable
atom ledger. Any learned selector is development signal until a selected
archive passes exact CUDA auth eval.

## Positive Feedback Compiler-Optimization Loop

The best mental model is an optimizing compiler:

```text
source video / public anatomy / component traces
  -> representation IR
  -> prediction / topology / pose IR
  -> quantization IR
  -> entropy / pack IR
  -> deterministic archive.zip
  -> exact CUDA profile
  -> next pass
```

Required feedback artifacts:

- `archive_profile`: member sizes, SHA-256, runtime tree hash, decoder schema,
  charged payload inventory, zip integrity.
- `scorer_profile`: exact SegNet/PoseNet total and per-pair traces, component
  gates, hard-pair opportunity density, failure class.
- `optimizer_profile`: atom set, rejected atoms, estimated benefit/byte,
  Lagrangian multipliers, active subspace basis, interaction assumptions.
- `negative_profile`: exact regression, no-op proof, harness bug, stale queue,
  sidecar/closure violation, reactivation criteria.

The feedback loop is allowed to be aggressive:

- Public PR67/PR65 anatomy informs proposal priors.
- C-067 exact component trace informs hard-pair weights.
- CMG2 exact negatives define forbidden block-mode trust regions.
- CMG3 top1/top2 exact traces should immediately train/update CMG3A atom
  weights if they land.
- Q-FAITHFUL snapshot collapse updates geometry preflight requirements.

But feedback is not score truth. Promotion requires exact CUDA on the final
archive bytes.

## Contrarian Failure Attacks

The council's adversarial attack list:

1. **Uncharged payload attack.** Any PR70/PR68-like move of score-affecting
   bytes into `inflate.py`, source constants, generated literals, hidden files,
   host paths, or sidecars invalidates the claim. Preflight should scan source
   blobs and archive members for suspicious task-specific payload relocation.
2. **Runtime-custody attack.** Same `archive.zip` SHA can score differently
   after runtime code changes. Exact eval must record runtime tree SHA and
   compare it in result tables.
3. **No-op packer attack.** A repacker can claim a transform while reusing the
   same payload. Every packer must record source member SHA, transformed member
   SHA, and transform class: reuse, decode/re-encode, true transform.
4. **Proxy-overfit attack.** Local mask disagreement, CPU/MPS proxy, H100-only
   diagnostic, or visual fidelity can rank proposals but cannot promote or
   kill.
5. **PoseNet cliff attack.** Global QZS block and CMG2 downsample results show
   very sharp PoseNet cliffs. Any byte win with larger than tiny geometry drift
   needs fast exact diagnostic before T4.
6. **Additivity attack.** PR65 qpost and pose atoms can be individually
   plausible but stacked destructive. Additive deltas are not composable until
   the stack is exact-evaluated.
7. **Stale queue attack.** Active/Pending/Running cloud state without artifact
   harvest is not evidence. Queue records need freshness, manifest linkage, and
   terminal harvest validation.
8. **Learned-lane dead-pose attack.** Q-FAITHFUL or JointFrameGenerator
   training with `pose_dim>0` but zero/missing pose tensors is invalid as a
   faithful successor. Fail closed before remote spend.

## Top 3 Immediate Experiments

### 1. CMG3A adaptive global row-run allocation

Rationale: CMG3 top1/top2 has the largest current material rate surface:
top1 saves `148186` bytes but has `0.03548` pixel disagreement; top2 saves
`47018` bytes with `0.01117` disagreement. Fixed uniform top-K wastes bytes on
easy rows and starves hard rows. CMG3A selects row-run atoms from a global
priority queue, which is the correct water-fill move.

Concrete local builds:

```bash
FRONTIER=experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip
MASKS=experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/decoded_mask_array.npy

.venv/bin/python experiments/build_cmg3_adaptive_runs_candidate.py \
  --frontier-archive "$FRONTIER" \
  --decoded-mask-array "$MASKS" \
  --output-dir experiments/results/c067_cmg3a_adaptive_body140k \
  --target-body-bytes 140000 \
  --adaptive-max-runs-per-row 8 \
  --compressor auto

.venv/bin/python experiments/build_cmg3_adaptive_runs_candidate.py \
  --frontier-archive "$FRONTIER" \
  --decoded-mask-array "$MASKS" \
  --output-dir experiments/results/c067_cmg3a_adaptive_body166k \
  --target-body-bytes 166000 \
  --adaptive-max-runs-per-row 8 \
  --compressor auto
```

Hard gates:

- Do not T4-dispatch until byte screen is below C-067 by at least `20KB` and
  reconstructed-mask SHA/runtime smoke pass.
- If CMG3 top1/top2 exact T4 both collapse by PoseNet `>10x` C-067, CMG3A must
  add component-trace/hard-pair weighting or move to learned postfilter before
  more exact eval.
- Evidence needed for claim: A++ exact T4 CUDA, archive SHA/bytes, component
  trace, runtime tree SHA, CMG3 reconstructed-mask SHA.

### 2. PR67/QZS3 mixed local quantization and packer atoms

Rationale: global QZS3 block size b32 is a narrow safe point; b16/b24/b48/b64
and QZS4/block128 collapsed or regressed. This means the next renderer byte
surface is mixed/local allocation, not another global block sweep.

Experiment:

- Build a tensor/group-level mixed QZS3 block allocator.
- Keep sensitive tensors at b32 or smaller only if exact H100 says they help.
- Allow byte-efficient insensitive tensors to use b48/b64/QZS4-like local
  blocks.
- Solve group selection as a knapsack using exact or diagnostic component
  response, with no promotion from byte-only output.

Hard gates:

- Local byte screen must save at least `1KB`; H100/L40S diagnostic must keep
  PoseNet within a tight trust region before T4.
- T4 only if predicted total score beats C-067 by at least `0.0005`.
- Evidence needed for claim: A++ exact T4 CUDA on the stacked archive.

### 3. Learned charged residual/postfilter atoms over C-067/CMG3

Rationale: We know exact mask residual pixels and exact per-pair component
traces. Hand-picked AMR1 over CMG2 failed because the base destroyed geometry.
A learned, charged, identity-default postfilter over CMG3 or C-067 can search
pair/class/row-run corrections in a low-dimensional differentiable family.

Experiment:

- Use `plan_cmg3_pixel_lagrangian_atoms.py` after CMG3 top1/top2 results to
  build pair/frame/class/row-run atom ledgers.
- Train or solve a tiny deterministic correction program: decision tree,
  LUT, small convolution, spline/range-coded residuals, or arithmetic-coded
  selected atoms.
- Ensure every learned parameter and selected atom is in the archive.

Hard gates:

- No dispatch if the postfilter is global and non-identity on easy pairs.
- No scorer access or sidecar at inflate time.
- First exact diagnostic can be H100/L40S; T4 only if a full archive is within
  `<=0.31` predicted band or has material byte headroom with safe components.
- Evidence needed for claim: A++ exact T4 CUDA; H100-only is proposal ranking.

## Top 5 Next-After Experiments

1. **Q-FAITHFUL successor geometry v1.** Patch/verify nonzero pose training
   and export custody, then run five-stage QAT++ with EMA, exact deployed pose
   stream, scorer-roundtrip simulation, and snapshot export every useful
   checkpoint. Use H100/H200/A100 for training/export diagnostics, T4 only for
   a closed archive near C-067 components. Evidence needed: deterministic
   archive plus exact CUDA; current collapsed snapshots are A-negative only.

2. **PR65 pose manifold transfer without SegNet tax.** PR65's PoseNet is much
   better than C-067 but SegNet worse. Mine its pose/length-table/postprocess
   structure as low-dimensional pose atoms, not wholesale output qpost. Gate
   every transfer on C-067 SegNet trust region. Evidence needed: exact CUDA
   stacked archive; public fields are external only.

3. **CMG4 connected-component and boundary grammar.** Move from row-run
   preservation to tracked components, boundary splines, birth/death events,
   horizon/lane priors, and residual tiles. This is the mask analog of a
   semantic codec. Evidence needed: local closed archive smoke, then fast CUDA
   exact diagnostic, then T4 promotion if score plausible.

4. **Entropy/range coding for CMG3A/pose atoms.** Once CMG3A selected atom
   distributions stabilize, replace generic bz2/lzma bodies with a small
   charged range/ANS model or enumerative coder. Evidence needed: byte-identical
   decoded mask SHA plus exact eval if any runtime changes.

5. **Stacked atom compiler pass.** Combine only exact-evidenced safe atoms:
   C-067 base, CMG3A if component-safe, mixed-QZS local blocks if safe, pose
   hard-pair atoms if safe, and learned postfilter if safe. Evidence needed:
   exact eval of the stacked archive; standalone deltas do not compose.

## Stacking Plan

Stack only after each component has at least one exact or high-confidence
diagnostic artifact:

```text
C-067 base
  + one mask representation replacement or repair path
  + one renderer local quantization path
  + one pose atom path
  + one postfilter/residual path
  + one entropy/container layout path
  -> complete archive.zip
  -> exact CUDA auth eval
```

Stack priority:

1. Rate-dominant mask grammar if it preserves components.
2. Renderer mixed local QZS only where component-safe.
3. Pose atoms only if they improve PoseNet enough to pay for bytes.
4. Postfilter atoms only if they are pair/region gated and identity elsewhere.
5. Entropy/layout last, unless it is decoded-mask/decoded-renderer identical.

Stop conditions:

- If a stack changes runtime tree, record runtime tree SHA and treat it as a
  new runtime-custody comparison.
- If score worsens but components identify a single bad atom, retire that atom
  config, not the whole family.
- If a stack is byte-positive but component-neutral, promote only if exact T4
  score beats C-067 after recomputation.

## Exact-Eval And Promotion Plan

Use hardware by role:

- H100/H200/A100: training, search, export, fast exact diagnostics, large atom
  sweeps, QAT snapshots.
- L40S/RTX 4090: diagnostics only when fast chips unavailable.
- Lightning T4/equivalent: promotion-grade A++ confirmation only.

Promotion steps for any candidate:

1. Claim dispatch with `tools/claim_lane_dispatch.py claim ...` before remote
   work.
2. Build deterministic archive with fixed member ordering, timestamps,
   permissions, hidden-file exclusion, zip-slip safety, payload closure, and
   manifest.
3. Record archive bytes/SHA, source archive bytes/SHA, runtime tree files,
   decoded-mask/pose/renderer payload SHAs, and `score_claim=false`.
4. Run fast CUDA diagnostic unless the job is already T4 and high-EV.
5. T4 exact eval through `archive.zip -> inflate.sh -> upstream/evaluate.py`,
   preferably via `experiments/contest_auth_eval.py --device cuda`.
6. Harvest state-derived artifacts; do not parse human logs when JSON exists.
7. Recompute score from components; compare runtime tree SHA; adjudicate
   component gates; update claim matrix/report only after validation.

## Paper And Data-Viz Outputs

Required figures/tables for the paper and site:

- Frontier ladder: C-053 through C-067 exact T4 rows with score decomposition.
- Rate-distortion budget curve: bytes versus allowed distortion for targets
  `0.33`, `0.315`, `0.30`, `0.25`, `0.20`.
- Archive anatomy Sankey: mask/model/pose/postfilter/container bytes for
  PR67, PR65, C-067, CMG3 candidates, Q-FAITHFUL candidates.
- Hard-pair heatmap: per-pair PoseNet/SegNet contribution and repair
  opportunity density.
- Atom waterfall: accepted/rejected atoms with charged bytes, expected benefit,
  exact result, and failure class.
- Negative cliff plots: QZS3 block-size cliff, CMG2 downsample/repair collapse,
  Q-FAITHFUL snapshot collapse, PR65 qpost collapse.
- Compiler feedback graph: source profiles -> atom planner -> archive builder
  -> exact eval -> next profile.
- Floor analysis figure: Shannon human-video floor versus Yousfi-Fridrich
  contest sufficient-statistic floor, with explicit compliance boundary.

Every plotted point must carry an evidence grade. External public PR points
remain external unless reproduced through our exact archive-custody path.

## Permanent Preflight And Metabug Hardening

Recommended permanent guards:

1. Source-payload scanner for large base85/base64/brotli/zlib literals or
   suspicious task-specific constants in `inflate.py`/runtime code.
2. Runtime tree SHA required in every score-bearing JSON and report row.
3. Decoder-member consumption proof: if archive contains `masks.cmg3`,
   `alpha4_residual_repair.*`, `qpost.bin`, `zoom_scalars.bin`, or a new
   payload member, exact inflate logs and/or runtime manifest must prove it was
   consumed or fail as no-op.
4. Packer no-op guard: changed flags must cause changed targeted payload SHA
   unless explicitly recorded as reuse/control.
5. CMG decoded-mask SHA guard for every grammar/reconstruction path, not only
   compressed-body SHA.
6. Pose-stream preflight: `pose_dim>0` training/export rejects zero or missing
   pose tensors unless explicitly forensic/non-promotable.
7. Cloud dispatch claim closure: terminal rows required for completed/failed
   jobs to prevent phantom active claims.
8. T4/g4dn Torch pinning and no CPU fallback: exact eval must fail if CUDA
   preflight or CUDA sample path is not true CUDA.
9. Zip strictness: central/local filename match, no duplicate names, no hidden
   files/resource forks, no absolute paths, no parent traversal.
10. CLI surface discipline: scripts and runbooks must invoke real argparse
    surfaces; never invent flags.

## Senior Adversarial Review Of This Memo

Pass 1 - compliance: All proposed score-affecting changes require charged
archive payloads or fixed contest code. PR68/70 are quarantined as invalid
forensics. No proposal relies on scorer patches, sidecars, CPU/MPS, or hidden
host files.

Pass 2 - evidence: Every lane names the evidence grade needed before a claim.
H100/L40S diagnostics are not promoted. Public PRs are external until locally
rebuilt and exact-evaluated.

Pass 3 - wall-clock: CMG3A is first because the builder exists and can produce
archives immediately. Mixed QZS local allocation is second because it reuses
the current public-floor basin. Learned postfilter/Q-FAITHFUL are higher upside
but require more training/export closure.

Pass 4 - contrarian objection: The memo may still overestimate CMG3A because
pixel disagreement is a weak proxy for PoseNet geometry. Mitigation: use exact
CMG3 top1/top2 traces before adding more T4 spend, and move to learned/predictive
grammar if PoseNet collapses.

Pass 5 - no-meat-left criterion: The memo preserves small byte polish only as
secondary. The main EV is concentrated on mask grammar, learned geometry,
mixed quantization, and stacked exact archives.

## Ranked Dispatch Table

| rank | lane | expected score EV | wall-clock path | hardware | required archive bytes | risk | exact eval gate |
|---:|---|---:|---|---|---:|---|---|
| 1 | CMG3A adaptive row-run grammar | `-0.005` to `-0.040` if geometry survives; high variance | local byte screen now; fast diagnostic if non-T4; T4 promotion only after top1/top2 trace branch | local + H100/L40S diag + T4 promotion | preferably `<250000`; strong if `<230000` with safe components | PoseNet cliff from hard-mask geometry | A++ T4 CUDA, decoded-mask SHA, component trace |
| 2 | PR67/QZS3 mixed local quantization | `-0.0005` to `-0.003` | implement mixed block/group allocator; H100 sweep; T4 only for safe candidate | H100/H200/A100 diag, T4 final | C-067 minus at least `1KB`; stronger if `>5KB` | local block can silently hit global QZS cliff | A++ T4 stacked archive |
| 3 | Learned charged postfilter/residual atoms | `-0.002` to `-0.020` | plan atoms from CMG3/C-067 traces; train/solve tiny identity-default corrector; exact screen | H100/A100 train/diag, T4 final | depends on atom payload; must beat lambda after bytes | qpost-style PoseNet/SegNet regression | A++ T4; H100 only non-promotable |
| 4 | Q-FAITHFUL successor geometry v1 | `-0.010` to `-0.080` if geometry closure works | patch pose/EMA/export gates; long fast-chip QAT++ snapshots; archive every checkpoint | H100/H200/A100 | near C-067 bytes unless distortion improves materially | prior snapshots catastrophic; pose/geometry contract fragile | deterministic archive plus exact CUDA, T4 promotion for rank |
| 5 | PR65 pose manifold transfer | `-0.001` to `-0.008` | extract low-dim pose/length atoms; preserve C-067 SegNet; no global qpost | local/H100 diag, T4 final | small side channel, ideally `<2KB` | PR65 improves PoseNet but pays SegNet/rate | A++ T4 stacked archive |
| 6 | CMG4 component/boundary grammar | `-0.020` to `-0.080` | build component/boundary decoder; local byte screen; fast exact diagnostic | local + H100/H200, T4 final | target `<180000` with C-067-like components | implementation time; boundary mistakes hurt PoseNet | exact CUDA after closed archive |
| 7 | Entropy/range coding for stable atoms | `-0.001` to `-0.010` | after atom distribution stabilizes, replace generic compression | local first, T4 if runtime changes | decoded output identical or better; any byte cut helps | premature coder work before stable atom distribution | byte-identical smoke; A++ if runtime/eval changes |

## Final Council Verdict

The next score break is most likely from a charged mask sufficient-statistic
program that preserves PR67 geometry while deleting tens of kilobytes. CMG3A is
the immediate executable move; learned/postfilter/Q-FAITHFUL successors are the
larger floor moves. Packer/pose polish still matters, but only as stacked atoms
or if it is essentially free. The correct shape is not one codec. It is a
closed, contest-faithful atom compiler driven by exact CUDA feedback.
