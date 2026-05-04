# Grand Council Shannon-Floor C-059 Next Wave - 2026-05-02

Evidence stance: contest-faithful, adversarial, and archive-metered.
Score claim: `false` except for exact artifacts explicitly cited below.
Runtime code edits in this session: none.

This memo synthesizes the current C-059 frontier, public PR65/PR67 signals,
Lagrangian atom planning, hard-pair and low-dimensional pose subspaces,
ego-motion/foveation priors, Quantizr five-stage QAT clues, stacking, and
exact-eval constraints into a top-5 experiment roadmap.

## Source Boundary

Internal exact anchor:

- C-059 exact T4 auth eval:
  `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.json`
- C-059 adjudicated JSON:
  `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.adjudicated.json`
- C-059 component trace:
  `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/component_trace.json`
- C-059 submission packet:
  `experiments/results/submission_packet_c059_20260502/submission_packet_manifest.json`

Public/external signals:

- PR67 `qpose14_qzs3_filmq9g_slsb1_r55`:
  https://github.com/commaai/comma_video_compression_challenge/pull/67
- PR65 `henosis_qz_n3z_r25_clean`:
  https://github.com/commaai/comma_video_compression_challenge/pull/65
- PR64 `unified_brotli`:
  https://github.com/commaai/comma_video_compression_challenge/pull/64
- PR55 `quantizr`:
  https://github.com/commaai/comma_video_compression_challenge/pull/55
- Local structured PR summary:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/public_pr_summary.json`
- Local PR anatomy:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/archive_anatomy.json`

Planning and strategy sources:

- `experiments/results/pose_atom_plan_c059_20260502/pose_atom_policies.json`
- `experiments/results/public_pose_manifold_transfer_20260502/pose_manifold_transfer_policy.json`
- `.omx/research/top_submission_reverse_engineering_refresh_20260502_codex.md`
- `.omx/research/atom_lagrangian_waterfill_sub03_system_20260501_codex.md`
- `.omx/research/public_pose_manifold_transfer_20260502_codex.md`
- `.omx/research/qfaithful_five_stage_qatpp_execution_20260502_codex.md`
- `.omx/research/charged_mask_grammar_ego_foveation_greenup_20260502_codex.md`
- `.omx/research/mixed_local_qzs_block_allocation_20260502_codex.md`
- `.omx/research/works_negatives_hardened_stack_20260502_codex.md`

## Current Exact Anchor

| Label | Evidence | Score | PoseNet | SegNet | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| C-059 QZS3/QP1 b32 mask-first packed layout | A++ exact T4 CUDA | `0.3157055307844823` | `0.00049637` | `0.00061244` | `276347` | `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab` |

JSON component terms:

```text
pose contribution = 0.07045353078448233
seg contribution  = 0.061244
rate contribution = 0.184008
n_samples         = 600
device            = cuda
gpu               = Tesla T4
gpu_t4_match      = true
```

The recomputation from rounded public formula inputs gives
`0.315705654902935`; the authoritative C-059 value is the structured JSON
field `score_recomputed_from_components = 0.3157055307844823`.

## Public Frontier Contrast

| Source | Evidence Grade | Public/Local Status | Score Signal | PoseNet | SegNet | Bytes | Use |
|---|---:|---|---:|---:|---:|---:|---|
| PR67 | external plus local anatomy | open public PR | public rounded `0.31`; rounded-field formula `0.314864164052394` | `0.00048597` | `0.00061000` | `276564` | contest-faithful component target |
| PR65 | external plus local anatomy | open public PR | public rounded `0.32`; author approx `0.31968005`; rounded-field formula `0.319682427689121` | `0.00035283` | `0.00070896` | `284425` | pose-manifold clue with SegNet danger |
| PR64 | external historical | closed public PR | live body reports `0.34`; official bot later reports `0.33` with different PoseNet/SegNet fields | `0.00052868` in body, `0.00061622` in bot eval | `0.00072205` in body, `0.00061261` in bot eval | `287165` | single-Brotli and velocity-delta packing clue |
| PR55 Quantizr | external historical | merged public PR | rounded `0.33` | `0.00051010` | `0.00061113` | `299970` | five-stage QAT and scorer-contract training clue |

Local PR67 anatomy reports one member `p`, payload `276464` bytes, with fixed
slices:

```text
mask_obu_br  = 219472
model_qzs3_br = 56093
pose_qp1_br = 899
QZS3 params  = 87836 finite parameters, 111 state-dict keys
```

PR67 beats C-059 by public rounded fields despite being `217` bytes larger:

```text
net PR67 - C059 score delta = -0.0008413667320887885
pose contribution delta     = -0.0007419822433689938
seg contribution delta      = -0.00024400000000000376
rate contribution delta     = +0.00014449139282751117
```

Therefore C-059 already has the rate edge. The next score drop should not be a
generic byte diet unless the component terms remain fixed. The active target
is charged component quality inside the C-059 b32 mask-first archive shape.

## Dialectic

Thesis: C-059 is the only current internal A++ anchor. It is already
contest-faithful, T4-equivalent, exact-CUDA, full-sample, and byte-stable.

Antithesis: PR67 shows the current public basin is still ahead on components,
and PR65 shows much lower PoseNet is possible, but both are external signals.
PR65 also demonstrates the central danger: pose improvement can be repaid by
SegNet loss. Negative evidence says global scalar QZS block sweeps, raw
residual top-K, postprocess atoms without pair gates, and byte-only wins are
not enough.

Synthesis: the next wave should be an atom compiler, not another lane zoo. Each
candidate atom must have:

- a typed stream contract: pose, mask, renderer, postprocess, pack, or runtime;
- charged byte cost inside `archive.zip`;
- expected score utility under current C-059 component traces;
- an interaction penalty;
- deterministic archive construction;
- exact CUDA auth eval before promotion.

Negation of the negation: failed atoms are not discarded as anecdotes. They
become constraints on the next atom family. QZS4 global block collapse becomes
a mixed/local quantization guard. PR65 SegNet loss becomes a SegNet trust-region
gate. PR70/PR69-style payload movement becomes an archive validator rule.

## Score Geometry

The contest formula is:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

Useful reference calculations:

| Scenario | Evidence Grade | Pose Term | Seg Term | Rate Term | Score |
|---|---:|---:|---:|---:|---:|
| C-059 exact JSON anchor | A++ | `0.07045353` | `0.061244` | `0.184008` | `0.31570553` |
| PR67 public rounded fields | external | `0.06971155` | `0.061000` | `0.18415262` | `0.31486416` |
| PR67 components at C-059 bytes | derivation from external fields | `0.06971155` | `0.061000` | `0.18400812` | `0.31471967` |
| PR65 pose with C-059 SegNet/rate | derivation from external fields | `0.05939949` | `0.061244` | `0.18400812` | `0.30465162` |
| PR65 pose plus PR67 SegNet at C-059 bytes | derivation from external fields | `0.05939949` | `0.061000` | `0.18400812` | `0.30440762` |
| C-059 weighted top32 diagnostic if SegNet stable | empirical planning only | `0.06971106` | `0.061244` | `0.18405873` | `0.31501379` |

Sub-0.30 with C-059 SegNet and bytes requires approximately:

```text
pose_dist <= 0.000299733
```

That is below PR65's public PoseNet `0.00035283`, so sub-0.30 probably needs
both stronger pose geometry and at least one rate or SegNet improvement. This
is why the roadmap is stacked only after component archives are measured.

## Top-5 Experiment Roadmap

### 1. C059-PAIRQP1-TOP32: scorer-weighted hard-pair QP1 line search

Evidence grade now: `diagnostic_planning_non_promotable` plus one H100
diagnostic accepted archive with `score_claim=false`.

Concrete basis:

- C-059 pose atom planner:
  `experiments/results/pose_atom_plan_c059_20260502/pose_atom_policies.json`
- Active diagnostic:
  `experiments/results/vast_harvest/line_search_c059_weighted_pairs_top32_20260502T0410Z/archive.accepted_latest.json`
- Dispatch provenance:
  `experiments/results/vast_harvest/line_search_c059_weighted_pairs_top32_20260502T0410Z/dispatch_provenance.json`

The current diagnostic accepted archive records:

```text
archive_bytes       = 276423
archive_sha256      = cda95e70440e9ef295985a042fda2d74715ef6a0e665a1c37871cddd051cd908
mask_br_bytes       = 219472
model_br_bytes      = 55965
pose_br_bytes       = 886
best_pose_mse       = 0.00048596322381248076
score_claim         = false
```

Expected term movement if SegNet remains C-059-stable:

```text
pose term  0.07045353 -> 0.06971106
seg term   0.06124400 -> 0.06124400
rate term  0.18400812 -> 0.18405873
score      0.31570553 -> about 0.31501379
net gain   about 0.00069
```

Dispatch recommendation:

1. Claim lane before any GPU job:
   `tools/claim_lane_dispatch.py claim C059-PAIRQP1-TOP32 ...`
2. Continue H100/H200 iteration only from exact C-059 SHA and the policy JSON.
3. Repack accepted candidates through the exact C-059 mask-first layout.
4. Run fast CUDA exact diagnostic immediately on each accepted archive.
5. Queue T4/equivalent promotion only when H100 diagnostic recomputed score is
   below C-059 by at least `0.0003` and SegNet does not regress.

Required preflights:

- Source archive SHA equals C-059 SHA.
- Pair list and selected frames are recorded in archive provenance.
- Accepted archive is copied into the result directory before eval.
- `contest_auth_eval.py --device cuda` uses the exact archive bytes.
- No scorer imports, no sidecars, no host-local pose files.

Failure modes:

- Pose improvement disappears under exact archive inflate.
- SegNet drifts even though only pose was modified.
- Brotli/VLQ byte overhead consumes pose gain.
- H100/T4 component mismatch reverses the diagnostic result.
- Pair-window search overfits one exact hardware path or stale source archive.

Adversarial verdict: highest immediate EV. It is already close to PR67 pose
without changing the mask/model streams, and it has the smallest implementation
surface.

### 2. PR65/PR67 pose-manifold atom transfer under C-059 SegNet gates

Evidence grade now: `external_public_summary_plus_local_anatomy` and
`diagnostic_planning_non_promotable`.

Concrete basis:

- Policy:
  `experiments/results/public_pose_manifold_transfer_20260502/pose_manifold_transfer_policy.json`
- PR65 summary:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/public_pr_summary.json`
- PR65 anatomy:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/archive_anatomy.json`
- GitHub PR65:
  https://github.com/commaai/comma_video_compression_challenge/pull/65
- GitHub PR67:
  https://github.com/commaai/comma_video_compression_challenge/pull/67

The top transfer policy is formula-only and non-promotable, but it gives a
specific hard-pair list:

```text
policy_name = public_pose_manifold_transfer_top016
pairs       = 164,64,130,112,97,153,70,198,420,289,166,435,78,156,418,87
charged_bytes_estimate = 32
net_expected_utility_after_rate_and_risk = 0.00027099915892772895
requires_exact_cuda_stack_eval = true
```

Strategic use:

- PR67 is the safer pair-transfer source because it is close to C-059 and
  improves both public PoseNet and SegNet.
- PR65 is the stronger pose manifold clue because public PoseNet is
  `0.00035283`, but its public SegNet `0.00070896` is far outside the C-059
  trust region.
- PR65-derived atoms should be admitted only behind hard SegNet gates and
  preferably as local pose residuals, not global postprocess shifts.

Expected term movement:

- Near-term realistic target: recover PR67 pose/SegNet at C-059 bytes,
  approximately `0.31471967`.
- Aggressive planning target: approach PR65 pose while retaining C-059 SegNet,
  approximately `0.30465162`.
- The PR65 target is not a score prediction. It is an upper bound on pose
  opportunity if the SegNet penalty can be prevented.

Dispatch recommendation:

1. Claim a separate lane: `C059-PR65PR67-MANIFOLD-TOP16`.
2. Build three closed archive candidates: PR67-pair-only, PR65-pose-only, and
   PR67-plus-PR65 guarded hybrid.
3. Keep PR65 qpost disabled in the first pass unless the candidate records
   exact predicted SegNet repair and charged qpost bytes.
4. Evaluate on H100/H200 for triage, then T4 only if score beats C-059 and
   component gates pass.

Required preflights:

- Every PR-derived atom must have a source URL, archive SHA, segment bytes, and
  transform provenance.
- C-059 SegNet trust gate: no promotion if SegNet exceeds C-059 by more than
  a tiny predeclared tolerance.
- No global learned selector at inflate unless its weights are charged and
  justified by byte utility.

Failure modes:

- PR65 pose advantage is global and does not decompose into pair-local atoms.
- PR65-style side information repairs PoseNet but damages SegNet.
- External PR rounded fields do not reproduce under local exact archive
  custody.
- Small charged residuals interact badly with QP1 byte layout.

Adversarial verdict: second immediate EV. The upside is larger than roadmap 1,
but the interaction risk is also larger.

### 3. CMG1 strict mask grammar, then component-boundary water-fill

Evidence grade now: `design_review`, `external`, `empirical byte anatomy`,
`score_claim=false`.

Concrete basis:

- Mask grammar plan:
  `.omx/research/charged_mask_grammar_ego_foveation_greenup_20260502_codex.md`
- PR67 anatomy:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/archive_anatomy.json`
- Exact anchor:
  C-059 archive and component trace above.

The largest charged stream in the PR67/C-059 basin is the mask segment:

```text
PR67 mask_obu_br bytes = 219472
rate term of 219472 bytes = about 0.1461 score
```

Roadmap:

1. `CMG1_STRICT_PR67_MASK_REPRO`: replace the current mask segment with a
   charged deterministic grammar that reproduces identical OBU bytes or
   identical class tensors.
2. If strict reproduction beats the current mask segment, exact-eval the full
   archive even though component risk should be zero.
3. Only after strict accounting works, run `CMG1_COMPONENT_BOUNDARY_WATERFILL`
   with connected components, boundaries, scanline spans, temporal tracks,
   residual tiles, hard pairs, luma motion, lane/horizon bands, and
   openpilot-like ego priors as compress-time proposal features.

Expected term movement:

- Strict bit-identical grammar changes only the rate term if the decoder is
  correct.
- Every saved `1000` charged bytes is worth about `0.00066586` score.
- A `5000` byte strict mask win is worth about `0.003329` score with no
  component movement.
- Lossy boundary water-fill is potentially larger, but not predictable until
  exact component traces exist.

Dispatch recommendation:

1. Build locally first; no GPU dispatch until decoded mask identity and archive
   validator checks pass.
2. Claim `CMG1-STRICT-C059` before remote exact eval.
3. Use H100 exact eval only after payload closure is proven.
4. T4-promote only if the strict candidate materially beats C-059 bytes or if
   a lossy grammar beats C-059 total score with component gates.

Required preflights:

- Decoded mask SHA or class tensor SHA matches reference for strict mode.
- Every grammar table, frequency table, residual, checksum, and selector bit is
  charged in `archive.zip`.
- New archive suffixes, if any, are admitted in both the auth-eval archive
  validator and local smoke whitelist.
- Inflate runtime stays within contest budget.

Failure modes:

- Grammar decoder plus charged tables is larger than Brotli OBU.
- Direct class tensor path differs subtly from scorer mask input.
- Lossy connected-component simplification collapses PoseNet or SegNet.
- Foveation or ego priors become uncharged side information.
- Decoder complexity creates hidden nondeterminism or review risk.

Adversarial verdict: best structural upside. It should start with strict
reproduction because that separates accounting bugs from scientific risk.

### 4. Mixed/local QZS allocation and protected learned quantization

Evidence grade now: `empirical byte-screen`; current mixed/local archives are
not exact-evaluable until a charged decoder exists.

Concrete basis:

- Mixed/local ledger:
  `.omx/research/mixed_local_qzs_block_allocation_20260502_codex.md`
- C-059 source archive:
  `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/archive.zip`
- Byte-screen root:
  `experiments/results/mixed_local_qzs_block_allocation_20260502/`
- Negative ledger:
  `.omx/research/works_negatives_hardened_stack_20260502_codex.md`

Observed byte-screen:

```text
global:32 source = 276347 bytes, exact-evaluable existing QZS3 runtime
mixed:48:frame1_head=32,pose_mlp=32 = 276162 bytes, -185 bytes, MQZ1 decoder absent
mixed:64:frame1_head=32,frame2_head=32,pose_mlp=32 = 276075 bytes, -272 bytes, MQZ1 decoder absent
```

Expected term movement if made exact-evaluable and component-stable:

```text
-185 bytes -> about -0.000123 score
-272 bytes -> about -0.000181 score
```

This is not enough alone to beat PR67, but it is valuable as a rate-side atom
stacked with pose/mask improvements.

Dispatch recommendation:

1. Do not dispatch GPU eval for MQZ1 until a charged deterministic runtime
   decoder exists and local smoke passes.
2. Use H100 only for exact-evaluable candidates.
3. Keep global QZS4/block sweeps deprioritized: prior exact H100 diagnostic
   saved bytes but collapsed PoseNet badly.
4. Prefer protected per-tensor choices driven by component response and
   hard-pair sensitivity.

Required preflights:

- Repacker proves transform vs reuse/no-op.
- QZS decode emits finite tensors and exact expected parameter count.
- Mixed block metadata is charged and deterministic.
- Component-response guards reject known PoseNet-sensitive tensors.
- Archive remains compatible with `archive.zip -> inflate.sh ->
  upstream/evaluate.py`.

Failure modes:

- Decoder overhead erases all byte wins.
- Local tensor block changes perturb output enough to collapse PoseNet.
- Existing QZS3 runtime cannot safely express the proposed allocation.
- Byte-screen archive is mistaken for score evidence.

Adversarial verdict: medium immediate EV, high stacking utility. It is a
supporting atom family, not the main frontier mover.

### 5. Q-FAITHFUL five-stage QAT++ snapshot export, not blind training wait

Evidence grade now: `external` for Quantizr PR55; `empirical` for local
Q-FAITHFUL training until an archive snapshot gets exact CUDA auth eval.

Concrete basis:

- QAT++ ledger:
  `.omx/research/qfaithful_five_stage_qatpp_execution_20260502_codex.md`
- Quantizr PR55:
  https://github.com/commaai/comma_video_compression_challenge/pull/55
- Local public-basin code surfaces:
  `src/tac/quantizr_qzs3_codec.py`,
  `src/tac/qp1_pose_codec.py`,
  `submissions/robust_current/inflate_renderer.py`

Quantizr's durable clues:

- odd-frame/half-rate mask stream;
- FiLM-on-pose JointFrameGenerator;
- depthwise-separable parameter efficiency around 88k params;
- training through upsample, clamp/round, and downsample scorer contract;
- five-stage-like anchor, finetune, joint, QAT, final process;
- EMA, hard examples, quantization-aware export, and exact packed archive
  accounting.

Expected term movement:

- Reproducing PR55 at C-059 bytes is not enough: Quantizr-like components at
  C-059 rate are still around `0.3165`.
- A useful snapshot must beat C-059 components, not just look like Quantizr.
- Minimum harvest threshold: deterministic snapshot exact diagnostic at or
  below about `0.3150`.
- Larger upside target: approach PR65 pose without PR65 SegNet loss, which
  implies the `0.304x` derivation band if bytes remain near C-059.

Dispatch recommendation:

1. Keep active long training isolated; do not mutate its artifact tree.
2. Harvest snapshots into separate output directories with explicit claims and
   source manifests.
3. Export QZS3/QP1/QPose variants as deterministic archives, not loose
   checkpoints.
4. Run fast CUDA diagnostics on H100/A100; T4-promote only a byte-identical
   archive that approaches or beats C-059.

Required preflights:

- `eval_roundtrip=True`.
- Exact deployed pose stream is nonzero and recorded; zero-pose fallback is a
  hard failure.
- EMA/live export choice is recorded.
- QAT scales, bit depths, codebooks, and all learned side information are
  charged in the archive.
- Snapshot export path writes archive, manifest, SHA, bytes, and component
  trace.
- No duplicate trainer writes the same output tree.

Failure modes:

- Training loss improves but exact CUDA score does not.
- QAT fake-quant differs from packed archive behavior.
- Snapshot export silently uses wrong pose stream or live/EMA weights.
- Full schedule consumes wall-clock while immediate atom loops move faster.
- Runtime size or inflate time violates contest budget.

Adversarial verdict: high upside but not critical-path. It should run in
parallel as snapshot-export/eval, not as a wait-for-training strategy.

## Dispatch Queue Order

Recommended order if operators have clean claims and fast chips:

1. `C059-PAIRQP1-TOP32`: continue/repack/eval the accepted weighted pair-window
   QP1 candidate; T4-promote if H100 diagnostic beats C-059 with stable SegNet.
2. `C059-PR65PR67-MANIFOLD-TOP16`: build PR67-safe and PR65-guarded transfer
   archives; hard SegNet gate.
3. `CMG1-STRICT-C059`: build bit-identical charged mask grammar locally, then
   exact-eval only if byte-positive.
4. `MQZ1-PROTECTED-C059`: make mixed/local QZS allocation exact-evaluable;
   screen as a stackable rate atom.
5. `QFAITHFUL-SNAPSHOT-C059`: harvest deterministic snapshots from existing
   training without interfering with the training tree.

Fast-chip policy:

- H100/H200/A100: iteration and diagnostics.
- T4/equivalent: promotion only.
- L40S/H100 diagnostic results may reorder candidate priority but cannot
  promote claims.

## Required Preflight For Every GPU Dispatch

Before any training, eval, or remote-GPU job:

```bash
tools/claim_lane_dispatch.py claim <lane_id> ...
```

Then verify:

1. No active same-lane conflict in `.omx/state/active_lane_dispatch_claims.md`.
2. Exact source archive SHA, bytes, and evidence path recorded.
3. Candidate archive is deterministic and copied into the result directory.
4. Archive has no hidden files, resource forks, zip-slip paths, duplicate
   names, central/local header mismatch, or score-affecting sidecars.
5. All runtime artifacts needed by inflate are inside `archive.zip` or fixed
   contest code.
6. Eval path is literal:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

preferably through:

```bash
experiments/contest_auth_eval.py --device cuda
```

7. JSON artifacts are treated as authority; human logs are support evidence.
8. Score is recomputed from component fields before any claim.
9. Component trace is generated for any candidate that changes the frontier.
10. Dispatch row status is updated success or fail.

## Recursive Adversarial Review

Question 1: What if the public PR fields are stale or non-reproducible?

Answer: classify them as external only. Use them to propose atoms, never to
promote. The exact C-059 JSON remains score truth until a repo-built archive
gets exact CUDA eval on identical bytes.

Question 2: What if a candidate saves bytes but damages PoseNet?

Answer: preserve it as scoped negative evidence. Do not broad-kill the family.
Convert the failure into a sensitivity guard or atom interaction penalty.

Question 3: What if a candidate improves local/H100 but not T4?

Answer: keep it diagnostic. T4/equivalent exact eval is the promotion surface.
The H100 result can still guide the next proposal if archive custody is clean.

Question 4: What if PR65 pose transfer damages SegNet?

Answer: this is expected risk, not surprise. Admit PR65 atoms only under
predeclared SegNet gates and pair-local attribution.

Question 5: What if CMG1 cannot beat Brotli strictly?

Answer: stop copying public exploit mechanics. Pivot to component-boundary
water-fill and learned selector atoms where byte savings are explained by
geometry and are stackable.

Question 6: What if QAT++ is too slow?

Answer: do not wait. Harvest snapshots as deterministic archives and exact
screen them. The critical path remains C-059 atom loops.

## Final Council Verdict

The frontier is no longer bottlenecked by discovering a brand-new high-level
representation. The measured basin is already known:

```text
half-rate mask stream
+ small JointFrameGenerator
+ QZS3 packed renderer
+ QP1/qpose-style pose stream
+ single-blob deterministic archive
+ exact CUDA feedback
```

The next frontier movement should come from a Lagrangian atom compiler:

```text
score_drop(atom | current archive)
  > charged_byte_cost(atom)
  + uncertainty_penalty(atom)
  + interaction_penalty(atom | selected set)
```

The first atom family to execute is scorer-weighted pair-window QP1 velocity
search. The largest structural upside is charged mask grammar and
component-boundary water-fill. PR65 and QAT++ are pose-opportunity engines, but
both must be kept inside C-059 SegNet and payload-closure gates. Mixed/local
QZS allocation is a supporting rate atom once exact-evaluable.

No Shannon-floor attainment claim is warranted. The correct claim is narrower:
C-059 is the current internal A++ anchor, PR67 is the public component target,
and the next contest-faithful work should be complete-archive exact-eval of
charged atoms, not proxy wins or external-field extrapolation.
