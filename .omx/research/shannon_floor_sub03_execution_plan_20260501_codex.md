# Shannon-Floor Sub-0.3 Execution Plan - 2026-05-01

## Objective

Drive from the current exact T4 promotion-grade frontier near `1.002` to a
contest-compliant sub-`0.3` score as fast as possible, without relaxing
custody, reproducibility, CUDA auth eval, or component gates.

This is an execution plan, not a result ledger. Exact results remain in the
dated experiment artifact directories and progress ledgers.

## Current Score Geometry

Current exact T4 frontier:

- Archive: `owv3_0120`
- Score: `1.0021175309471926`
- Bytes: `617410`
- SHA-256:
  `06af57f770342cde494c37839200fdda79bdadd29826009e5e107ab296b4057a`
- Components: PoseNet `0.00356094`, SegNet `0.00402305`

Approximate score contributions:

- Rate: `~0.411`
- SegNet: `~0.402`
- PoseNet: `~0.189`

Therefore sub-`0.3` cannot come from renderer byte polish alone. It requires a
new scorer-aligned representation class that cuts mask/geometry payload and
does not collapse PoseNet or SegNet.

## Reverse-Engineering Synthesis

Top public leaderboard evidence is external motivation only, but it converges
with local analysis:

- Top approaches appear to use scorer-aligned learned representations and tiny
  charged deterministic decoders, not generic video compression.
- The likely winning contract is: one compact geometry/mask stream, a small
  pose-conditioned renderer/decoder, and aggressively compressed weights.
- Quantizr/Selfcomp-like designs imply the full archive contract matters. A
  standalone half-frame, CRF, or generic RLE substitution is not equivalent.
- The target is not "better AV1"; it is "encode only what SegNet/PoseNet care
  about, with all side information charged."

## Highest-EV Lanes

### Lane A - Alpha Geometry Replacement

Priority: highest.

Why: the mask stream is the largest remaining rate target and the only near-term
path with enough score delta.

Immediate plan:

1. Harvest CRF60/CRF62 exact CUDA threshold probes.
2. If either preserves components, promote same bytes to T4 and build sparse
   repair only where exact component response says it is safe.
3. If both collapse, stop plain grayscale-LUT as a promotion path and pivot to
   geometry-preserving Alpha: NeRV/INR/SegMap/self-compressing mask decoder.

Gate:

- No Alpha archive can promote unless exact CUDA eval shows PoseNet/SegNet near
  baseline and all repair/model bits are inside `archive.zip`.

### Lane B - Learned Topology Decoder

Priority: co-equal with Alpha once CRF threshold is known.

Candidate families:

- Q-FAITHFUL / Quantizr-like pose-conditioned compact renderer.
- Selfcomp/SegMap-like grayscale or latent mask self-compression.
- HNeRV/SNeRV/TinyNeRV mask codec trained against decoded baseline masks.

Why: this is the most plausible sub-`0.3` class by reverse-engineering and
score geometry.

Immediate plan:

1. Fix export/runtime/archive closure before spending large GPU time.
2. Run smoke inflate locally, then fast CUDA exact eval on L40S/RTX/H100/A100.
3. Promote only exact same archive bytes on T4/equivalent if it clears frontier
   and component gates.

Gate:

- No sidecars, no uncharged weights, no scorer patches, no CPU/MPS ranking.

### Lane C - PoseNet-Aware Sparse Repair

Priority: high, dependent on safe base representation.

Use hard pairs, hard zones, classes, pose intervals, component response, and
inverse-steg style allocation as optimization variables. This is not a
constraint on exploration; it is the targeting system.

The archive must contain the sparse repair payload and deterministic decoder.
The repair objective is not visual fidelity; it is SegNet/PoseNet component
preservation at minimum charged bytes.

Gate:

- Response-selected atoms must be validated by exact archive eval. Do not rank
  atoms from a collapsed lossy base.

### Lane D - OWV3 / Direct-FD Renderer Polish

Priority: filler/stack component, not the main sub-`0.3` path.

Why: it can shave `~0.005` to `~0.03`, but cannot bridge the missing `~0.7`.

Plan:

- Keep only cheap/parallel exact candidates that preserve components.
- Stack with Alpha/learned-geometry only as complete archives.

### Lane E - Full-Stack Coordinator

Priority: after component archives exist.

Use ADMM/Dykstra/water-filling/entropy coding as coordination tools, not proof
of additive composition. The stack is its own archive.

Gate:

- Exact CUDA eval on the stacked archive is mandatory before any claim.

## Dispatch Order

1. Harvest CRF60/CRF62 threshold exact evals.
2. If either is safe, immediately run T4 confirmation and build minimal charged
   sparse repair candidates.
3. If both fail, pivot Alpha effort to NeRV/INR/SegMap/Q-FAITHFUL-class learned
   representation and stop spending on plain CRF grayscale.
4. In parallel, harden Q-FAITHFUL export/inflate/archive custody and launch one
   fast-chip exact eval as soon as closure is proven.
5. Keep OWV3/direct-FD exact polish only when it does not block Alpha/learned
   representation work.
6. Stack only exact-evidenced components, then exact eval the full archive.

## Resource Policy

- Use L40S/RTX PRO/H100/A100 for fast screening when queue wait is low.
- Use T4/equivalent for promotion-grade confirmation.
- Stop duplicate jobs once equivalent evidence exists.
- Modal/Vast are acceptable for build, training, smoke, and cheap CUDA signal,
  but exact promotion must preserve archive custody and canonical eval.

## Non-Negotiable Failure Discipline

- Bad results are suspected engineering/config/math/geometry failures until
  reviewed.
- Do not kill a family from one failed implementation.
- Do not call sub-`0.3` or Shannon-floor attainment without exact archive
  evidence.
- Every negative result must produce a redesign or a narrowly scoped
  retirement.

## Next Immediate Actions

1. Refresh and harvest `exact_eval_alpha_crf60_grayscale_l40s_20260501T1339Z`.
2. Refresh and harvest `exact_eval_alpha_crf62_grayscale_l40s_20260501T1339Z`.
3. Based on those exact components, choose:
   - component-safe threshold -> T4 promotion + sparse repair build;
   - collapse -> learned topology decoder/Q-FAITHFUL relaunch.
4. Keep the reverse-engineered leaderboard hypothesis active as design pressure:
   compact charged learned decoder, scorer-aligned geometry, tiny archive,
   exact deterministic inflate.

## Supersession - Grayscale Path Correction - 2026-05-01T14:40Z

CRF60 and CRF62 exact L40S CUDA evidence collapsed PoseNet, so the dispatch
order above is superseded as follows:

1. Stop spending primary wall-clock on plain post-hoc CRF grayscale replacement.
   Treat CRF60/62/63 as scoped forensic negatives for that measured
   implementation shape only.
2. Prioritize learned grayscale/topology routes that train through the exact
   decode representation. The immediate highest-EV implementation families are:
   corrected SA/SegMap soft-LUT parity, FilmCanvas soft-LUT parity,
   Q-FAITHFUL/Quantizr-like one-mask-per-pair pose-conditioned generation, and
   NeRV/INR/HNeRV mask-topology preservation.
3. For Selfcomp-style soft-LUT work, the required invariant is now explicit:
   `encode_masks_grayscale -> softmax(exp(-d^2 / 2 sigma^2)) -> renderer`
   during both training and inflate. Hard one-hot grayscale decode is
   forensic-only unless a separate exact review reopens it.
4. Fastest next CUDA work: train the corrected soft-LUT SA/FilmCanvas path on
   a fast chip, export a closed deterministic archive, run exact CUDA
   screening, and T4-confirm any frontier bytes. In parallel only if resources
   are free, repair Q-FAITHFUL export/runtime closure and run one exact fast
   CUDA eval.
5. Sub-`0.3` still requires a representation-class jump, not thousandth-place
   renderer polish. OWV3/direct-FD stays useful for stacking and protection,
   but the primary wall-clock path is learned charged decoder geometry plus
   pair-frame/pose-conditioned generation.

## Supersession - AMR1 Sparse-Repair Outcome - 2026-05-01T16:47Z

The AMR1 residual-selector branch has exact CUDA evidence and is not the
current sub-`0.3` path.

- CRF62/CRF63 grayscale plus AMR1 sparse repair was evaluated across class
  prefix, top residual frame group, Lane W hard-pair, and pair-atom prior
  selectors.
- T4 and L40S agree: rate drops, but PoseNet collapses. Best T4 repair score
  was `4.0751452447605425` (`crf62_class2_fix1`, `606572` bytes, PoseNet
  `0.89807248`), far worse than C-044.
- This is not an Alpha/grayscale family kill. It is a scoped retirement of
  post-hoc CRF grayscale base plus AMR1 residual selector configs.

Updated highest-EV plan:

1. Treat water-filling/Lagrangian allocation as the global optimizer, but only
   after the atom family remains inside the PoseNet/SegNet basin.
2. Replace pixel-residual AMR1 selectors with geometry-preserving atoms:
   learned soft-LUT decoder weights, pair-conditioned latent packets,
   temporal-endpoint corrections, class-confusion transforms, boundary-band
   latent repairs, or NeRV/INR/HNeRV topology atoms.
3. Use the contest Lagrange threshold exactly:
   include atom `a` only when expected
   `100*dseg_saved + sqrt(10*pose_before)-sqrt(10*pose_after)` exceeds
   `25*bytes_a/37545489`, with confidence penalties and exact CUDA checkpoints.
4. Keep C-044 as the frontier anchor. No further T4 spend on CRF62/63 AMR1
   residual selectors unless a new base representation has first passed a
   component-safe exact screen.
5. Next aggressive tranche should be corrected soft-LUT SA/FilmCanvas or
   Q-FAITHFUL/Quantizr-like pair-conditioned generation, with deterministic
   export/inflate/archive closure and fast CUDA screen before T4 promotion.

## Multidimensional Atom System - 2026-05-01T16:58Z

The search should now be expressed as a constrained atom system, not as
single-lane intuition.

Let each candidate change be an atom `a_i` with decision variable
`x_i in {0,1}` or relaxed `x_i in [0,1]` during planning. Atoms may be:

- representation atoms: soft-LUT parameters, learned decoder weights,
  NeRV/INR/HNeRV latent packets, SegMap/Q-FAITHFUL control signals;
- temporal atoms: pair, frame, transition endpoint, odd/even pair generator;
- semantic atoms: class, class-confusion transform, boundary band, connected
  component, hard-zone selector;
- rate atoms: entropy model, quantizer, arithmetic/range coding block,
  compressed side-information stream;
- runtime atoms: decoder branch, lookup table, pose-conditioned FiLM block,
  deterministic postprocessor.

The score objective is:

```text
score(A) =
  100 * seg_dist(A)
  + sqrt(10 * pose_dist(A))
  + 25 * bytes(A) / 37,545,489
```

The Lagrangian planning objective for a candidate atom set `S` around a valid
base `B` is:

```text
maximize  U(S | B)
        = sum_i x_i * b_i
        + sum_{i<j} x_i*x_j * s_ij
        + sum_{i<j<k} x_i*x_j*x_k * t_ijk
        - lambda * sum_i x_i * c_i
        - rho * risk(S)
```

where:

- `b_i` is measured or calibrated component benefit:
  `100*dseg_saved_i + sqrt(10*pose_before)-sqrt(10*pose_after_i)`;
- `c_i = bytes_i` or the measured archive byte delta of the concrete stack;
- `s_ij > 0` is synergy: two atoms together save more score than their
  independent deltas;
- `s_ij < 0` is antagonism: two atoms interfere, duplicate bytes, or break
  scorer geometry;
- `t_ijk` captures rare higher-order interactions, used only after exact
  factorial evidence or a strong mechanistic reason;
- `risk(S)` penalizes calibration uncertainty, component-gate proximity,
  inflate-time risk, dependency closure risk, and custody gaps.

The KKT/water-filling condition for relaxed planning is:

```text
include atom i while:

  dU/dx_i =
    b_i
    + sum_j x_j*s_ij
    + sum_{j<k} x_j*x_k*t_ijk
    - lambda*c_i
    - rho*d risk/dx_i
  > 0
```

For the contest formula, the byte shadow price in score units is:

```text
lambda_rate = 25 / 37,545,489 ~= 6.6586e-7 score per byte
```

Any atom must beat this rate cost after confidence penalties. However, byte
optimality is valid only inside the geometry basin:

```text
pose_dist(candidate) <= gamma_pose * pose_dist(C-044)
seg_dist(candidate)  <= gamma_seg  * seg_dist(C-044)
```

with current screening defaults `gamma_pose = gamma_seg = 1.25`. If this gate
fails, local residual atoms do not have trustworthy marginal benefits. The
new `alpha_repair_atom_planner.py` basin guard records this and refuses to emit
water-filling archive policies.

Concrete blocked artifacts:

- `experiments/results/alpha_repair_atom_plan_c044_crf63_pair_lzma_20260501_basin_blocked/alpha_repair_atom_plan.json`
  - `water_filling_allowed=false`
  - PoseNet ratio `376.173066481576`
  - SegNet ratio `2.309887788451`
- `experiments/results/alpha_repair_atom_plan_c044_crf62_pair_lzma_20260501_basin_blocked/alpha_repair_atom_plan.json`
  - `water_filling_allowed=false`
  - PoseNet ratio `251.348293600372`
  - SegNet ratio `1.676259117965`

Measurement design:

1. Estimate `b_i` with exact finite-difference archive eval when possible.
2. Estimate cheap priors from hard pairs, component sensitivity, residual byte
   cost, and scorer-aligned geometry, but label them prediction/empirical until
   exact eval.
3. Estimate pairwise `s_ij` only for high-EV combinations using a small
   factorial design: `i`, `j`, and `i+j` exact archives.
4. Treat `s_ij = score(i+j)-score(base)-delta_i-delta_j`; positive values are
   stack synergy, negative values are antagonism.
5. Promote a stack only as its own exact archive, never by summing deltas.

Hard-pair identification:

```text
hardness(pair p, atom family F) =
  E[score_saved(p,F)] / E[bytes_to_repair(p,F)]
```

This is not baseline error alone. A pair is operationally hard when it has high
repair opportunity density under a specific atom family. The next hard-pair
table should record:

- pair index, frames `2p` and `2p+1`;
- PoseNet delta, SegNet delta, and combined score delta;
- class-confusion counts and boundary/transition indicators;
- byte cost by repair family;
- calibrated confidence interval;
- known synergies or antagonisms with other atoms.

Repair design:

- First fix the base representation so it remains in the component basin.
- Then allocate bytes with Lagrangian water-filling over geometry-preserving
  atoms, not over arbitrary residual pixels.
- For learned decoders, expose latent dimensions/channels as atoms and measure
  channel or group slopes, analogous to rate-distortion optimized learned
  encoding.
- For pair-conditioned generation, treat each pair latent or FiLM coefficient
  group as an atom and measure score-per-byte after deterministic archive
  closure.

Research anchors used as design motivation, not evidence:

- TACTIC frames compression as rate-distortion-accuracy optimization for a
  downstream task:
  `https://arxiv.org/abs/2109.10658`.
- Task-oriented semantic communication formulates joint pixel/task distortion
  under a rate constraint:
  `https://arxiv.org/abs/2201.10929`.
- Rate-distortion optimized learned image coding highlights Lagrangian
  `d + lambda R`, signal-dependent encoding decisions, and near-optimal bit
  allocation across learned feature channels:
  `https://refubium.fu-berlin.de/bitstream/handle/fub188/33392/Rate-Distortion_Optimized_Encoding_for_Deep_Image_Compression.pdf`.

These references support the mathematical framing only. Contest claims still
require exact CUDA archive evidence.

## Manifold, Flow, And Self-Learning Layer - 2026-05-01T17:04Z

The atom system can be lifted onto a manifold view to avoid the failure we just
measured: CRF residual atoms moved along a byte-saving direction that was far
off the scorer manifold.

State:

```text
z = (r, theta, q, h, e, p)
```

where:

- `r` is representation state: masks, grayscale controls, latents, topology;
- `theta` is decoder/runtime parameter state;
- `q` is quantization/allocation state;
- `h` is hyperprior/entropy model state;
- `e` is exact-eval evidence state;
- `p` is policy/search state: priors, lambdas, active hypotheses, risk.

Feasible contest manifold:

```text
M_valid = {
  z :
    archive_closed(z)
    and inflate_deterministic(z)
    and bytes_charged(z)
    and device_cuda_eval(z)
    and component_basin(z)
    and runtime_budget(z)
}
```

The best search direction is a projected natural-gradient or mirror-descent
step on this feasible manifold:

```text
dz/dt = Proj_Tz(M_valid)(
          - G(z)^-1 * grad_z score(z)
          + exploration(z)
          - risk_barrier(z)
        )
```

Interpretation:

- `G(z)` is the local geometry/Fisher/Hessian metric of the scorer and decoder
  response. It should weight directions by how sensitive PoseNet/SegNet are to
  perturbations, not by pixel MSE.
- `Proj_Tz(M_valid)` removes directions that violate archive closure, inflate
  determinism, side-information accounting, CUDA-only evidence, or component
  gates.
- `risk_barrier(z)` grows sharply near known cliffs: PoseNet collapse, SegNet
  class-boundary collapse, non-deterministic runtime, venv/tooling drift, and
  uncharged sidecars.
- `exploration(z)` is allowed, but only as bounded, manifest-recorded probes
  that become evidence through exact archive eval.

Differential equation for lambda/water-fill adaptation:

```text
d lambda_i / dt =
  eta * (observed_component_loss_i - target_loss_i)
  - kappa * lambda_i
  + coupling_i(lambda, S)
```

This matches the existing dual-weight pattern in `src/tac/learnable_pair_weights.py`
but generalizes from pair weights to atoms. `lambda_i` rises for atoms or zones
whose measured component loss exceeds target, then decays when the repair is no
longer needed. `coupling_i` captures synergy/antagonism between neighboring
pairs, classes, temporal endpoints, or latent channels.

Self-learning policy:

```text
p_{t+1} = UpdatePolicy(
  p_t,
  evidence_t,
  posterior(atom_benefits, synergies, risks),
  constraints_t
)
```

Policy updates must be evidence-typed:

- exact CUDA archive evidence updates promotion/ranking priors;
- L40S/fast CUDA diagnostic evidence updates triage priors only;
- byte/proxy/local/MPS/CPU evidence updates engineering priors only;
- external papers and leaderboard reverse engineering update hypothesis priors
  only.

Manifold diagnostics to add:

1. Local tangent basis: perturb representation/latent/quantizer axes and
   measure exact component slope where affordable.
2. Curvature/cliff detection: run small positive/negative perturbations and
   detect nonlinearity in PoseNet/SegNet response.
3. Basin certificate: require the base representation to pass component
   relative gates before local atom allocation.
4. Synergy matrix: exact small factorial archives for high-EV atom pairs.
5. Antagonism matrix: flag atoms whose combination violates component gates or
   duplicates charged bytes.
6. Continuation path: move from C-044 toward learned topology in small
   component-safe steps rather than jumping directly to a collapsed base.

Practical next implementation:

- Add a `component_manifold_probe_v1` artifact schema:
  `axis_id`, `atom_family`, `epsilon`, `archive_sha256`, `bytes`, `seg`,
  `pose`, `score`, `slope`, `curvature`, `basin_passed`, `risk_flags`.
- Build probes around learned soft-LUT/SegMap/Q-FAITHFUL latent axes first,
  not CRF AMR1 residual pixels.
- Use the probe table to update atom priors and the Lagrangian allocator.
- Every continuation step is still a concrete archive; no manifold claim is a
  score claim without exact CUDA evaluation.

## Component-Manifold Probe V1 Landed - 2026-05-01T17:22Z

Implemented:

- `experiments/build_component_manifold_probe_plan.py`
- `src/tac/tests/test_component_manifold_probe_plan.py`

The tool emits `component_manifold_probe_plan_v1` from exact CUDA eval JSONs.
It is a derivation/planning artifact only: `score_claim=false`,
`promotion_eligible=false`. It records local chart coordinates, component
deltas, rate deltas, slopes, curvature, basin violations, continuation
candidates, synergy/antagonism residuals, and resource-diversity hooks.

Generated charts:

- `experiments/results/component_manifold_probe_c044_macro_20260501/component_manifold_probe_plan.json`
  - C-044-centered macro chart.
  - No continuation candidates.
  - Direct-FD M2 is inside the component basin but rate-regressive versus
    C-044.
  - Lane12 NeRV jsonfix40 and Alpha CRF62/CRF63 AMR1 repair points are
    component-basin failures and cannot seed water-filling policies.
- `experiments/results/component_manifold_probe_direct_fd_response_20260501/component_manifold_probe_plan.json`
  - Local direct-FD response chart with `eps={-2,-1,+1,+2}`.
  - All points stay inside the local component basin.
  - Curvature is measurable and nonzero, so future response charts should not
    assume globally linear archive perturbation behavior.

Mathematical consequence:

The optimizer now has an evidence-preserving chart hierarchy:

```text
macro family chart -> local tangent chart -> curvature/cliff chart
                   -> synergy matrix -> archive dispatch
```

Only local charts inside the component basin can feed Lagrangian water-filling.
Collapsed charts update priors and barriers in the ODE/self-learning policy:

```text
barrier_alpha += f(PoseNet_ratio, SegNet_ratio, archive_family)
lambda_safe_axis += eta * exact_inside_basin_benefit_per_byte
```

This keeps the resource search broad without letting invalid local assumptions
turn byte savings into exact-eval churn. The next high-EV chart families are:

1. Corrected soft-LUT SegMap/FilmCanvas build/eval variants.
2. Q-FAITHFUL/Quantizr-like pose-conditioned export closure.
3. NeRV/INR/HNeRV build-only or stale-pose-isolation variants that do not
   violate the Lane 12 retraining gate.
4. Deterministic external-language decoder atoms: Rust/Zig/C range/ANS/RLE
   payloads, temporal grammars, and tiny charged learned decoders.

## Contest Docs And Video-Overfit Update - 2026-05-01T17:45Z

The public target moved from abstract "sub-0.4 is possible" to concrete byte
geometry:

- qpose14: `0.32`, `287,573` bytes.
- unified_brotli: `0.33`, `287,165` bytes.
- quantizr: `0.33`, `299,970` bytes.
- fp4_mask_gen: `0.37`, `249,624` bytes.

Common pattern: full-res 5-class semantic controls, tiny charged neural
renderer, pose/velocity side-channel, and aggressive stream packing. This is
the architecture family to clone/adapt, not plain CRF grayscale or local AMR1
repair on a collapsed base.

The local video reverse-engineering artifact is:

- `experiments/results/contest_video_reverse_engineering_20260501/contest_video_reverse_engineering_fullres_masks.json`

It records the exact fixed clip SHA, HEVC metadata, luma-motion hard pairs,
full-res lane-mark log-zoom, and camera/FoE geometry. Next allocator state
should include these coordinates:

```text
atom = {
  pair_index,
  luma_motion,
  horizon_motion,
  lane_log_zoom,
  fullres_mask_class_mix,
  pose_velocity_delta,
  foveation_param_delta,
  charged_bytes,
  exact_component_delta
}
```

Hardware/openpilot constraints:

- Use EON/openpilot-style camera assumptions only as priors:
  native `1164x874`, focal length `910`, principal point approximately
  `(582,437)`, FoE/scorer VP `(256,174)`.
- Openpilot/supercombo is compress-time only unless weights are charged in the
  archive; its best role is teacher/prior for pose/foveation/latent atoms.
- Telescope/hyperbolic foveation should be exact-probed as a trust-region
  identity-centered transform around the FoE, not as an unconstrained warp.

Next concrete build/eval wave:

1. Build unified-packing prototypes over existing C-044 payload components:
   brotli/raw concatenation, delta-coded pose or velocity/log-zoom stream, and
   single-member archive overhead minimization.
2. Build one identity-safe hyperbolic-foveation side-channel probe with tiny
   charged params and exact CUDA eval to see whether PoseNet accepts the
   transform family at all.
3. Reopen Q-FAITHFUL/Quantizr-style export closure with full-res mask controls
   and pose velocity side-channel. This is highest-EV for sub-0.3.

## Live Leaderboard Reverse-Engineering Update - 2026-05-01T17:14Z

Official/public lowest scores now inspected live:

- `qpose14`: `0.32`, `287573` bytes, PoseNet `0.00052154`,
  SegNet `0.00061261`.
- `unified_brotli`: `0.33`, `287165` bytes, PoseNet `0.00061622`,
  SegNet `0.00061261`.
- `quantizr`: `0.33`, `299970` bytes, PoseNet `0.00051328`,
  SegNet `0.00061261`.
- `fp4_mask_gen`: `0.37`, `249624` bytes, PoseNet `0.00076576`,
  SegNet `0.00121106`.
- `selfcomp`: `0.38`, `279036` bytes, PoseNet `0.00055221`,
  SegNet `0.00122167`.

Branch-code reverse engineering was performed on temporary clones under
`/tmp/pact_topsubs_dpEPDD/`. Main implementation lesson:

```text
top submissions = semantic mask controls + tiny charged renderer
                + pose/velocity/log-zoom side-channel
                + exact scorer-path training
                + unified packing
```

Aggressive reorder:

1. Promote Quantizr/qpose-style export closure above any more OWV3 polish.
   The public frontier carries only about `0.19-0.20` rate points and about
   `0.13-0.14` non-rate points; C-044 is still about `0.406` rate and
   `0.591` non-rate. Both sides must change.
2. Keep no-retraining work on packing, pose-channel quantization, and
   foveation probes moving immediately. These can execute under the Lane 12
   gate.
3. Treat soft-LUT grayscale as valid only when trained and inflated through
   the same Gaussian probability map plus learned renderer. Hard onehot CRF
   grayscale is now a measured bad implementation, not the leaderboard trick.
4. Add hardware/ego-motion atoms to the allocator:
   `velocity`, `lane_log_zoom`, `FoE-centered foveation`, `horizon luma
   motion`, `pair index`, and `full-res mask class mix`.
5. Once the retraining gate clears, run qpose-class training on the fastest
   available H100/A100/4090-class chip, not T4. T4 is reserved for exact
   promotion on locked archive bytes.

## No-Retraining Frontier Pack Closure - 2026-05-01T17:26Z

While Lane 12 retraining is blocked, the best immediate score EV is lossless
rate reduction plus qpose/unified-style archive closure.

Resulting candidate:

- `renderer_payload_c044`: one `renderer_payload.bin.br` member that expands
  to the exact C-044 `renderer.bin`, `masks.mkv`, and `optimized_poses.bin`.
- Candidate archive:
  `experiments/results/renderer_packed_payload_c044_20260501/archive.zip`.
- Bytes/SHA: `594634`,
  `ff01e11f525514cebd27325d8207c06351df437cd25d61ee82af15b6e0ddcae6`.
- Formula-only delta versus C-044: `-0.010206951892409765`.
- Predicted exact score if byte-identical components hold:
  about `0.987333635165018`.
- T4 exact eval queued:
  `exact_eval_renderer_payload_c044_t4_20260501T1726Z`.

Strategic value:

1. Immediate possible frontier drop from pure charged rate reduction.
2. Validates a single-member payload contract matching the public qpose and
   unified_brotli design pattern.
3. Creates the container needed for next no-retraining pose-stream work:
   delta-coded velocity/log-zoom, qpose-style integer pose, and cross-stream
   entropy packing.

Next after harvest:

- If exact eval lands in-band, promote as the new C-048 frontier and use it as
  the source archive for all subsequent no-retraining packs.
- If unpack/runtime fails, treat as harness/contract bug only; source member
  bytes are already verified byte-identical locally.
- In parallel, implement qpose-style pose-stream packing behind the same
  payload header, but do not score-claim until exact CUDA auth eval.

Update: lossless pose-column-delta packing is now implemented and queued.

- Preferred pending archive:
  `experiments/results/renderer_packed_payload_c044_posecd_20260501/archive.zip`.
- Bytes/SHA: `594456`,
  `f8b13737ea226524869b40132a31ca77ffdaa887ca8571e51e408861e27ecb54`.
- Pose codec: `pose_fp16_col_delta_v1`, decoded pose SHA
  `4b3d8cd58965971ad518a970441f5e9cc53ce361a62d51e95c1cdd2056be4449`.
- Formula-only delta versus C-044: `-0.01032547478606551`.
- Preferred T4 exact eval:
  `exact_eval_renderer_payload_posecd_c044_t4_20260501T1731Z`.

If both raw-payload and posecd exact evals complete, promote only the lower
adjudicated score with matching component gates; the raw job becomes a
reproducibility/control packet.

## Frontier Update - 2026-05-01T18:12Z

The first lossless pose-column-delta unified-payload T4 eval landed:

- Job: `exact_eval_renderer_payload_posecd_c044_t4fix1_20260501T1754Z`
- Evidence: A++ contest T4, promotion eligible.
- Score: `0.9872158806043723`.
- Archive: `594456` bytes,
  `f8b13737ea226524869b40132a31ca77ffdaa887ca8571e51e408861e27ecb54`.
- Components: PoseNet `0.00357305`, SegNet `0.00402367`.
- Delta versus C-044: `-0.010324706453055388`.

This proves the qpose/unified-style single-payload contract is useful even
before retraining: pure charged archive layout can move the frontier.

The next pending exact packet is stronger by construction:

- Job: `exact_eval_renderer_payload_posecd_palias_c044_t4_20260501T1806Z`
- Archive: `594412` bytes,
  `831f643f5c523dd4cf524cec18faf65e5d6e572a7e347ebd9ef712087fce2c09`.
- Expected if components match: about `0.9871866`.

The qpose14 all-channel pose codec was implemented and tested as a
top-submission probe, but its first archive is `594458` bytes, so it is
byte-dominated by the lossless posecd p archive. Keep it as an adversarial
pose-smoothing atom only; do not make it the main T4 promotion path unless an
exact diagnostic shows component-score improvement.

The mathematical control plane is now formalized in
`.omx/research/atom_lagrangian_waterfill_sub03_system_20260501_codex.md`.
Use that document for hard-pair, foveation, ego-motion, learned-renderer, and
archive-layout atom allocation.

## Current-Floor Transfer Update - 2026-05-01T20:20Z

The public floor remains `0.32` (`qpose14`) with `0.33` (`unified_brotli` and
`quantizr`) behind it. The current-floor transfer path is now split into two
renderer-codec atoms for the same Q-FAITHFUL `JointFrameGenerator` topology:

- Aggressive atom: QZS3 + RP2 qpose14/QP1. Best bytes, but the early phase-1
  checkpoint scored `8.420915711675079` on L40S because PoseNet collapsed.
- Conservative atom: PR63-style Torch-FP4 + RP2 qpose14/QP1. Slightly larger
  but closer to the current public floor payload and now deterministic in repo
  custody.

Operational rule:

- Do not spend more exact eval on the early snapshot; it is out of basin.
- Keep the 4090 Q-FAITHFUL trainer running.
- For the next later checkpoint snapshot, build both QZS3 and Torch-FP4
  variants, exact-eval the QZS3 qpose14 candidate first on a fast diagnostic
  GPU, and immediately fall back to Torch-FP4 qpose14 if QZS3 is still
  PoseNet-unstable.

## Control-Plane Update - 2026-05-01T21:16Z

The fastest sub-0.3 route now has two synchronized loops:

1. Learned-renderer loop:
   keep the live Q-FAITHFUL trainer running, snapshot later checkpoints, build
   QZS3 and Torch-FP4 current-floor packer variants, and exact-eval the first
   mature qpose14 candidate on a fast diagnostic GPU.
2. Exact atom-allocation loop:
   run `component_trace.json` on the C-051 exact archive and then on any lossy
   candidate that enters the basin. Use pair/frame/class/component atoms ranked
   by score-saved-per-byte, not by raw distortion.

New queued atom signal:

```text
exact_eval_component_trace_c051_l40s_20260501T2116Z
component_trace=true
top_k=100
archive_sha256=dc855b10b69353f1046aeb25d2eba17f43f48039ea0ef2f2d95f5c2a2bef782f
archive_bytes=594047
```

This is not a frontier score attempt. It is the missing exact per-pair
coordinate system for water-fill repair, foveated protection, ego-motion
protection, and learned selector calibration.
