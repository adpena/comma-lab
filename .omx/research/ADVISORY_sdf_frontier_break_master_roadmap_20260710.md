# ADVISORY — SDF level-set witness frontier-break master roadmap — 2026-07-10

`research_only=true`

Authority: architecture and execution advisory only. This memo does not move a frontier pointer,
ratify a score, authorize a launch, change a DSL/config, dispatch compute, or mutate a run. All score
claims retain their explicit evidence axis. The live dirty tree, v7.5.2 dry-start, owed16v2 work, and
click-polish work are separately owned and were preserved.

Companion advisories:

- `ADVISORY_sdf_pose_inverse_carrier_20260710.md`
- `ADVISORY_sdf_scorer_waterfill_20260710.md`
- `ADVISORY_sdf_v753_v8_receiver_disambiguators_20260710.md`
- `ADVISORY_evaluator_video_geometry_20260710.md`
- `ADVISORY_pr128_hnerv_reverse_engineering_sdf_transfer_v753_v8_20260710.md`
- `t5_crucible2/ADVISORY_v753_texture_trunk_fresh_eyes_20260710.md`
- `t5_crucible3/ADVISORY_v8_fresh_eyes_20260710.md`

## Answer first

The highest-upside path is still the original task-space SDF/level-set witness, not a smaller
HiNeRV archive. PR #128 is a useful public HNeRV-family control and an apparatus lesson, but its
byte-level techniques do not repair the central SDF debts: class birth and topology through the
frozen SegNet cell, a sufficiently expressive legal frame-0 Pose witness, exact receiver closure,
and evaluator-aware allocation of counted bytes.

The recursively optimized stack should be treated as one legal witness compiler with four homes:

\[
W=(G,T,P,C).
\]

- `G`: frame-1 class geometry as globally consistent SDF/class potentials;
- `T`: frame-1 texture placed only where it buys evaluator debt and, where claimed, projected through
  an exact Pose-preprocess null construction;
- `P`: a pose-only frame-0 generator plus pair motion/coefficients, distilled offline with PoseNet but
  decoded without PoseNet or any scorer weight;
- `C`: the counted archive compiler, quantizer, entropy coder, manifest, receiver, and exact custody
  surface.

v7.5.3 should be the smallest receiver-closed single-trunk experiment that separates these homes.
v8 should make geometry ownership class-disjoint while retaining a globally defined potential
decoder. Separate edge fields are a later disambiguator, not the default, because arbitrary edge
preferences are not automatically integrable.

The immediate order is therefore:

1. allow the current governed bounded dry-start to finish naturally and preserve its receipt;
2. repair one manifest-aware receiver used by MLX, NumPy, byte-close, and inflate;
3. build a legal decoder-native frame-0 pose carrier;
4. prove exact-D texture survival through uint8/resize/fresh raw;
5. run matched, receiver-closed n600 A/Bs;
6. only then select an exact archive and evaluate separate contest-CPU and contest-CUDA axes.

No scored v7.5.3 or v8 launch is presently justified.

## 1. Authority snapshot

Snapshot basis: `main` read at `91dbd820f`; all values below are observations, not hard-coded future
pointers.

### 1.1 Local and public score surfaces

- Canonical local pointer: `[contest-CPU] 0.19109982419209975`, archive 177,169 bytes, SHA-256
  prefix `b4689726`. Its public-source snapshot is stale and must not be used to describe the current
  public frontier.
- Separate local CUDA pointer: `[contest-CUDA] 0.20533002902019143`. It is not inferred from CPU.
- Public PR #112: merged; official maintainer CPU result `0.191126`, 177,136 bytes.
- Public PR #125: open; `0.190946` is author-claimed, not maintainer-ratified.
- Public PR #127: open; `0.190503` x86 is author-claimed, not maintainer-ratified.
- Public PR #128: open HNeRV-family submission; PR body, release asset, and report disagree on score
  and archive identity. `0.187992` remains external, unratified evidence and does not move the local
  pointer.

Public sources:

- <https://github.com/commaai/comma_video_compression_challenge/pull/112>
- <https://github.com/commaai/comma_video_compression_challenge/pull/125>
- <https://github.com/commaai/comma_video_compression_challenge/pull/127>
- <https://github.com/commaai/comma_video_compression_challenge/pull/128>

### 1.2 v7.5.2 process truth

Two directories must not be conflated:

- `levelset_v752_pilot_20260710T153914Z` was refused before training because its 300-epoch budget
  placed the Muon event at epoch 726.
- `levelset_v752_pilot_20260710T154100Z` reached epoch 1 and then exited with
  `SigmaMinPlateauDetector.should_ship_banked_r1` missing. `safe_run` recorded exit 1 after 1501.28
  seconds and peak RSS 32,668 MiB. No checkpoint, EMA, or resume artifact exists.

The crash fix and explicit AMBER controls landed at `41db09638`. The live successor observed during
this audit is only a governed bounded dry-start under
`experiments/results/__v752_drystart_final__/`; it is not a pilot relaunch. The parent requested
`--dry-start 2`, but deliberately executes the exact real 3,000-epoch schedule under a 2,500-second
wall-clock timeout per pass; it is not literally capped at two epochs. Pass 1 must boot, step, and
checkpoint, then the launcher may perform a second bounded resume round-trip. Literal disposition:

- failed pilot: `ABORTED-DEFECT`, preserve;
- live dry-start: `PRESERVE / MONITOR`, do not signal;
- full relaunch: `NOT YET LAUNCHED`;
- inference about resumability or score trajectory: `REFUSED — NO RECEIPT`.

## 2. The target optimization problem

Let `D_A` be the deterministic receiver encoded by archive `A`, `R` the exact resize/round/raw
operator, `F_seg` and `F_pose` the frozen evaluator cells, and `B(A)` the exact archive length. The
only outer objective is

\[
S(A)=100d_{seg}(F_{seg}(R(D_A)))+
      \sqrt{10d_{pose}(F_{pose}(R(D_A)))}+
      25\frac{B(A)}{37{,}545{,}489}.
\]

The legal design problem is

\[
\min_{A\in\mathcal A_{legal}} S(A)
\quad\text{subject to deterministic decode, exact cardinality, runtime, and custody constraints.}
\]

This is not ordinary RGB rate-distortion. Human fidelity is useful only when it moves one of these
frozen evaluator cells. A differentiable inner energy may propose updates, but only a fresh decoded
archive scored through `R` may accept them.

### 2.1 SDF-native inner energy

A coherent inner energy for geometry `phi`, texture `T`, pose carrier `P`, and quantized code `q` is

\[
\begin{aligned}
E ={}& \lambda_{seg}\,\widetilde d_{seg}
     +\lambda_{pose}\,\widetilde d_{pose}
     +\lambda_R\,\widehat\ell(q) \\
   &+\lambda_{eik}\int (\lVert\nabla\phi\rVert-1)^2
     +\lambda_{len}\int \delta_\epsilon(\phi)\lVert\nabla\phi\rVert \\
   &+\lambda_{curv}\int \delta_\epsilon(\phi)\kappa^2
     +\sum_c \mu_c(A_c(\phi)-A_c^*) \\
   &+\lambda_{screw}E_{transport}(\phi_t,\phi_{t+1},\xi_t)
     +\lambda_{birth}E_{Morse/topology}
     +\lambda_D E_{exact-D}.
\end{aligned}
\]

The area multiplier, temporal screw consistency, topology/birth term, and exact-D term are not
decorations. They are the forces demanded by the task geometry that a generic INR/HNeRV objective
omits. Loss weights change only at typed stage boundaries. They must not become per-step adaptive
knobs that erase causal attribution.

## 3. Recursive/fractal optimization map

“Fractal optimization” should mean the same score-per-counted-bit law is applied at every nested
scale, with each scale exposing a proposal, a receiver survival check, an interaction check, and an
exact accept/reject receipt.

| Scale | Object | Correct question | Required receipt |
|---|---|---|---|
| representation | `G/T/P/C` | Is this the right witness grammar? | complete archive vs complete archive |
| vehicle | v7.5.3/v8 | single trunk or class-owned fields? | matched config, compute, seed, checkpoint |
| frame | frame0/frame1 | which evaluator constrains it? | frame-specific ablation through `R` |
| class | five potentials | which class owns debt and bits? | per-class d_seg, birth/island counts |
| edge/junction | pairwise margins | where does argmax flip? | margin annuli, holonomy, tie flicker |
| 2x2 footprint | RGB/YUV6 atoms | what survives Pose resize? | exact first-six Pose inputs after raw |
| pair/time | `xi_t,c_t` | what motion/appearance state is irreducible? | innovation entropy and hard-pair debt |
| coefficient | quantized symbol | what is exact score value per byte? | parse-back mutation and delta score |
| entropy stream | contexts/headers | does context save net bytes? | final ZIP bytes, not payload estimate |
| apparatus | manifest/custody | can the candidate be reproduced? | source, archive, raw, scorer, hardware hashes |

At any scale, “proxy improved” is not an admission rule. The admission statistic is the exact
decoded `Delta S`, including interactions and changed archive bytes.

## 4. Optimal v7.5.3 design hypothesis

v7.5.3 should remain an optimal single-trunk control, but “single trunk” must not mean one
undifferentiated output head. The best current hypothesis is:

### 4.1 Geometry home `G`

- shared coordinate/temporal feature stem;
- five class potentials `phi_c` with explicit gauge fixing;
- SDF/eikonal regularity and topology-aware birth forces;
- margins, not RGB texture, own the class partition;
- lane and movable birth schedules remain event-driven and resume-persisted;
- class-level diagnostics include d_seg, island birth/death, anchor drift, and rate.

### 4.2 Texture home `T`

- frame1 only;
- compose final RGB before projecting a texture update into the six local Pose-preprocess-null atoms;
- use a bounded camera-grid preimage/lift so the null property is measured after bilinear resize,
  uint8 rounding, raw serialization, and fresh parsing;
- remove palette/texture gauge duplication;
- choose exactly one A/B arm: A1 linear control, A2 generic capacity, or A3 exact-D texture;
- count only receiver-consumed trainable coefficients; deterministic analytic banks are regenerated,
  not serialized as video-derived payload.

### 4.3 Pose home `P`

- frame0 has no Seg obligation, so do not render a ceremonial frame0 partition;
- start with the frame1 warp/motion carrier `xi_t`;
- add a small decoder-native shared basis/generator with per-pair coefficients only when its exact
  pose-score gain beats its archive cost;
- use PoseNet/Jacobians only offline to generate distillation targets;
- never place PoseNet, scorer weights, per-pair rasters, or GT-output tables in the archive.

The preferred formulation generates local quotient coordinates
`q0=(Y00,Y10,Y01,Y11,U,V)` per 2x2 Pose footprint from SDF margins, normals, curvature,
polyphase coordinates, `xi_t`, and a small pair code. A deterministic inverse-YUV and bounded integer
camera-grid lift produces frame0 raw. This is a Morse-Smale pose fiber over the decoded SDF state; it
does not assume “six Pose outputs implies six physical motion parameters are sufficient.” Learned
global basis/generator weights are counted once, pair codes are counted per pair, and scorer-side
Jacobians remain encoder-only.

### 4.4 Compiler/receiver home `C`

- one manifest-aware forward shared by MLX export, canonical NumPy, byte-close oracle, and inflate;
- reject every serialized tensor not consumed by that forward;
- reject every live trainable tensor absent from the archive grammar;
- regenerate fixed banks deterministically and hash their construction law;
- parse back the final ZIP before selection;
- exact cardinality: 1,200 frames and 3,662,409,600 raw bytes;
- preserve stage checkpoints and all selection/rejection receipts.

## 5. Optimal v8 design hypothesis

The default v8 increment should be `K` globally defined class potentials with class-owned trainable
state:

\[
\phi_c(x,t)=f_c(\gamma(x,t);\theta_c,z_{t,c}),
\qquad \hat y(x,t)=\arg\max_c\phi_c(x,t).
\]

`gamma` may be a frozen deterministic coordinate dictionary. Every trainable coefficient and pair
latent affecting class `c` must be owned by `c`; otherwise a class-c update can still steal another
class through a shared code. “Edge-centric” should initially describe the loss and rate ledger on
`phi_i-phi_j`, not a separate nonintegrable decoder.

The isolation invariant is

\[
\frac{\partial\phi_c}{\partial\theta_{c'}}=0,
\qquad
\frac{\partial\phi_c}{\partial z_{t,c'}}=0,
\quad c\ne c'.
\]

It must be tested after a real one-class optimizer step over all trainable state, not only by
perturbing one output row.

### 5.1 Optional E-edge disambiguator

If pairwise edge fields `g_ij` are tested, require:

\[
g_{ij}=-g_{ji},
\qquad g_{ij}=\phi_i-\phi_j
\]

or explicitly project them to the nearest integrable potential system under a fixed gauge. Report
cycle holonomy, reconstruction residual, and post-quantization partition drift. A projected edge arm
that is always `B^T phi` is a reparameterization of K potentials, not a distinct codec. A genuinely
cyclic edge system needs a separately specified graph-labeling decoder and stays held until that
decoder is byte-closed.

### 5.2 v8 texture and pose

v8 does not gain permission to duplicate texture and pose state per class. Use the same `T` and `P`
homes as v7.5.3 unless exact interaction measurements justify a class-owned exception. Junctions and
edge annuli may receive more geometry bits, but only under the exact water-filling rule.

## 6. Frontier waterlines

For a target `S*`, archive bytes `B`, and candidate `d_seg`, the maximum admissible pose debt is

\[
d_{pose}^{max}=\frac{\max(S^*-100d_{seg}-25B/37{,}545{,}489,0)^2}{10}.
\]

Using the current local CPU pointer only as a design waterline:

| B | d_seg | maximum d_pose to beat 0.1910998242 |
|---:|---:|---:|
| 75,000 | 0.0012 | 4.4776e-5 |
| 75,000 | 0.0010 | 1.6942e-4 |
| 75,000 | 0.0007 | 5.0638e-4 |
| 90,000 | 0.0012 | 1.2483e-5 |
| 90,000 | 0.0010 | 9.7173e-5 |
| 90,000 | 0.0007 | 3.7421e-4 |
| 90,000 | 0.0005 | 6.5890e-4 |
| 100,000 | 0.0010 | 6.0093e-5 |
| 100,000 | 0.0007 | 2.9718e-4 |
| 120,000 | 0.0010 | 1.2537e-5 |

The existing R1 advisory artifact (`d_seg=0.004549`, `d_pose=0.001610`, `B=89,772`) implies
`S=0.64156` on its advisory axis. It is not a near-frontier candidate. At roughly 90 kB and
`d_seg=0.001`, pose must fall below about `9.72e-5`, 16.6 times below R1. At `d_seg=0.0007`, the
pose requirement is about `3.74e-4`, 4.3 times below R1.

This also corrects an inherited arithmetic error: `d_pose=0.018` contributes
`sqrt(10*0.018)=0.424264`, not approximately `0.02`.

For the long-term sub-0.15 objective at 90 kB:

| target | B | d_seg | maximum d_pose |
|---:|---:|---:|---:|
| 0.15 | 90,000 | 0.0010 | infeasible even at zero pose debt |
| 0.15 | 90,000 | 0.0007 | 4.029e-5 |
| 0.15 | 90,000 | 0.0005 | 1.606e-4 |

Thus sub-0.15 is not “pose polishing.” At 90 kB it requires a materially smaller Seg partition as
well as a new pose carrier. R1's `d_pose=0.001610` is about 40 times above the sub-0.15 pose waterline
at `d_seg=0.0007`.

## 7. Proof DAG before any long or scored run

### Gate G0 — custody and current public authority

- refresh the correct challenge repository and leaderboard surface;
- distinguish official, author-claimed, local advisory, contest-CPU, and contest-CUDA evidence;
- pin source, inputs, runtime, evaluator, archive, and report hashes.

### Gate G1 — receiver identity

- one forward law across MLX, NumPy, byte-close, and inflate;
- tensor-consumption manifest;
- deliberate nonzero perturbation for every optional section survives parse-back and changes raw;
- unknown/unconsumed sections refuse.

### Gate G2 — deterministic cardinality/runtime

- two fresh inflates produce identical raw SHA-256;
- exactly 1,200 frames / 3,662,409,600 bytes;
- receiver runtime and peak memory below contest limits;
- durable SSD evidence and cleanup manifest.

### Gate G3 — exact-D texture

- frame0 texture contribution exactly zero;
- six-atom construction applied after final RGB composition;
- bounded camera-grid lift;
- fresh raw has identical first-six Pose inputs within the pre-registered exact tolerance;
- nonzero effect remains on the intended Seg annulus.

### Gate G4 — pose carrier

- no evaluator/scorer weight or output table in receiver;
- global generator/basis counted once, pair coefficients counted and parse-backed;
- complete-artifact A/B against `xi`-only, not a grafted residual comparison;
- exact pose gain exceeds exact archive cost at the operating point.

### Gate G5 — SDF/class isolation

- one-class loss/optimizer step across all trainable state;
- off-class potential movement below registered tolerance;
- per-class d_seg, anchors, island births, and topology receipts;
- edge-cycle receipt if an E-edge formulation is used.

### Gate G6 — matched n600 experiment

- same seed set, source, curriculum, stage boundaries, optimizer budget, and receiver;
- matched actual archive bytes, not raw parameter counts;
- separate A1/A2/A3 arms;
- decoded partition and raw hashes;
- exact component score and interaction matrix.

### Gate G7 — archive selection and exact axes

- accepted/rejected candidate ledger and selection replay;
- final ZIP identity, member table, raw identity, exact timing;
- exact contest-CPU and contest-CUDA evaluated separately;
- pointer move only through the canonical promotion gate.

## 8. Ranked advisory roadmap

The following is deliberately expansive. `GO` means useful next advisory/build work after a new
owner claims the relevant lane; it is not launch authority.

### P0 — eliminate false experiments

1. `GO`: preserve the failed-pilot crash receipt and classify it `ABORTED-DEFECT`.
2. `GO`: harvest the live dry-start only after it exits naturally; verify checkpoint/EMA/config and
   exact process exit receipt.
3. `GO`: refresh the public watcher against the correct challenge repository.
4. `GO`: build receiver tensor-consumption and unknown-section refusal manifests.
5. `GO`: make MLX/NumPy/inflate call one manifest-aware forward law.
6. `GO`: make archive accounting refuse fixed-bank and receiver-blind bytes.
7. `GO`: add exact frame/raw cardinality refusal before any score is accepted.
8. `HOLD`: all long training, scored v7.5.3/v8 A/B, and exact dispatch until G0-G2 close.

### P1 — pose floor escape

9. `GO`: measure the `xi`-only complete-artifact floor at n600 after receiver repair.
10. `GO`: generate offline full-image/minimum-norm frame0 inverse targets with scorer weights retained
    only on the encoder side.
11. `GO`: fit analytic low-frequency/polyphase bases under the local Pose metric.
12. `GO`: fit a counted shared decoder-native SDF-conditioned basis/generator plus small pair codes.
13. `GO`: sweep coefficient dimension, quantizer, entropy context, and hard-pair sidecar under actual
    ZIP bytes.
14. `GO`: compare linear basis, coordinate MLP, wavelet dictionary, and low-rank hypernetwork only as
    complete receiver-closed archives.
15. `GO`: partition easy/hard pairs by exact residual value per byte, not RGB error.
16. `HOLD`: per-pair raster tables, scorer weights, GT-output tables, or hidden data in code.

### P1 — SDF geometry and topology

17. `GO`: recover/measure the SegNet same-cell, adjacent-cell, and remote-cell interaction matrix.
18. `GO`: build an evaluator footprint graph for boundary annuli and high-order junctions.
19. `GO`: measure class birth/death, island persistence, anchor drift, and tie flicker per stage.
20. `GO`: derive class-specific area multipliers from measured area debt.
21. `GO`: derive temporal screw/advection forces from pair motion rather than hand-set global weights.
22. `GO`: measure curvature-scale schedules against Seg footprint scale.
23. `GO`: test junction-specific carriers only after global integrability is guaranteed.
24. `GO`: allocate SDF basis bandwidth by edge debt and entropy, not uniformly by class.

### P1 — exact-D texture

25. `GO`: implement the six-atom post-composition projection in a future claimed build lane.
26. `GO`: solve and bound the 2x2 camera-grid preimage under bilinear resize.
27. `GO`: verify null survival through uint8, raw serialization, and a fresh evaluator process.
28. `GO`: remove the texture-bias/palette gauge duplicate.
29. `GO`: make A1/A2/A3 mutually exclusive in the typed compiler.
30. `GO`: add ChromaRung as a named v7.5.3 arm only after the receiver consumes it.
31. `HOLD`: the statement “high-frequency chroma is Pose-null” without the fresh-raw proof.

### P2 — v8 structural isolation

32. `GO`: give every class its own pair latent and all other trainable field state.
33. `GO`: keep coordinate features frozen/deterministic in the first isolation experiment.
34. `GO`: replace narrow output-row perturbation tests with one-class optimizer-step tests.
35. `GO`: correct the config/harness kill-predicate semantic mismatch from one stable predicate ID.
36. `GO`: use K potentials plus edge-weighted losses as the first v8 decoder.
37. `GO`: build the E-edge Hodge/integrability disambiguator separately.
38. `GO`: register holonomy, reconstruction residual, post-quantization partition drift, and gauge.
39. `HOLD`: any “theft impossible” claim while shared trainable code exists.
40. `HOLD`: any E-edge scored arm without a globally defined decoder.

### P2 — exact bit allocation and compiler

41. `GO`: enumerate atomic receiver mutations with exact byte and component deltas.
42. `GO`: estimate pairwise interactions and build a conflict graph.
43. `GO`: solve budgeted set packing/knapsack on nonconflicting proposals, then re-score jointly.
44. `GO`: include ZIP headers, entropy contexts, alignment, and decoder code in every byte delta.
45. `GO`: estimate global-generator bytes and pair-code bytes separately.
46. `GO`: allocate a reserve for sparse hard-pair and high-order-junction corrections.
47. `GO`: stop a component when its best exact value per byte falls below the global waterline.
48. `GO`: retain all rejected proposals as continual-learning signal with verdict scope.

### P3 — apparatus and scientific closure

49. `GO`: define a candidate passport schema binding source, LFS inputs, runtime, config, checkpoints,
    archive, raw, scores, axis, timing, storage, and cleanup.
50. `GO`: make deterministic re-inflate and raw-hash identity a selection prerequisite.
51. `GO`: record complete accepted/rejected search receipts and resume state.
52. `GO`: protect partner-owned run/state surfaces with explicit ownership checks.
53. `GO`: validate SSD waterfall; current observation found approximately 771 GiB free on
    `/Volumes/VertigoDataTier/pact` and no second tier.
54. `GO`: keep exact CPU/CUDA and macOS advisory results as separate non-inferable rows.
55. `GO`: conduct a fresh licensing/custody review for every imported OSS component.
56. `GO`: make no-license repositories paper/behavior references only; do not copy their code.

## 9. Stop/continue laws

Continue a formulation only when at least one of these is true:

- it closes a receiver/custody blocker;
- it produces a complete candidate with a favorable exact `Delta S`;
- it reduces uncertainty enough to change the next allocation decision;
- it supplies a reusable negative result with a scoped verdict and regression guard.

Stop or reformulate when:

- an optional tensor is counted but not receiver-consumed;
- a supposed nullspace fails fresh-raw survival;
- the best exact marginal score gain is smaller than its exact byte cost;
- class isolation fails through any shared trainable state;
- edge holonomy is nonzero without a defined global decoder;
- a complete pose carrier cannot cross the waterline at its measured bytes;
- runtime/cardinality/custody cannot be certified.

A stopped formulation does not kill the SDF family. Record `verdict_scope` precisely: coefficient,
basis, receiver, carrier, vehicle, or paradigm.

## 10. Literal dispositions

| Action | Disposition | Exact reason |
|---|---|---|
| signal or terminate live dry-start | `FORBIDDEN` | separate live ownership; preserve process |
| resume failed pilot | `REFUSED` | no checkpoint/EMA/resume state exists |
| call dry-start a relaunch | `REFUSED` | two-epoch smoke is not a governed pilot |
| fresh governed v7.5.2 relaunch | `HOLD` | dry-start receipt and owner action still owed |
| v7.5.3 receiver/exact-D repair | `GO — BUILD ONLY` | high-EV prerequisite; requires new claimed lane |
| v7.5.3 scored A/B or training | `HOLD` | receiver/exact-D/matched-arm receipts absent |
| pose-carrier design/probe build | `GO — BUILD ONLY` | legal architecture defined; no scored authority yet |
| v8 K-potential ownership repair | `GO — BUILD ONLY` | first coherent isolation formulation |
| v8 training event | `HOLD` | shared code, receiver, predicate, and receipt gaps |
| independent E-edge carrier | `HOLD` | integrability/global decoder unresolved |
| PR #128 pointer move | `REFUSED` | external unratified and custody-incoherent |
| exact CPU/CUDA evaluation | `HOLD` | no receiver-closed selected candidate |

## 11. Exact remaining blockers

1. The live dry-start has not yet emitted a completion/checkpoint/resume-round-trip receipt.
2. No replacement governed v7.5.2 pilot exists.
3. v7.5.3 texture and v8 decoupled tensors are not consumed by the canonical NumPy/inflate receiver.
4. Fixed texture-bank bytes can be serialized despite being intended as deterministic/free.
5. No post-composition exact-D construction has survived fresh raw.
6. No legal decoder-native frame0 pose generator has crossed the frontier waterline.
7. v7.5.3 A1/A2/A3 arms are not mutually exclusive or matched in actual archive bytes.
8. ChromaRung is registered but not emitted as a named v7.5.3 compiler arm.
9. v8 shares trainable pair code across fields, so full class isolation is false.
10. v8 config and harness do not share one semantically stable kill predicate.
11. E-edge vs K-potential semantics and integrability are unresolved.
12. SegNet global dependency/interactions are not yet measured adequately for independent waterfill.
13. The public watcher and generated report surfaces are stale or target the wrong repository.
14. PR #128 score/archive/release custody and licensing/authorship remain incoherent.
15. No receiver-closed selected archive has separate exact contest-CPU and contest-CUDA receipts.

## 12. Triality and future wire-in

This advisory changes no triality leg. Future implementation units must land all three:

- DSL: typed `PoseInverseCarrier`, `ExactDTexture`, `ClassOwnedPotential`, and optional
  `IntegrableEdgeCarrier` levers with mutually exclusive arms and resume state;
- DAG: G0-G7 gates, candidate passports, matched A/Bs, and literal stop/continue edges;
- equations: receiver identity, pose MDL waterline, exact-D survival, class-isolation Jacobian, edge
  holonomy, and interaction-aware score-per-byte allocation.

Future empirical anchors must update the sensitivity map, Pareto constraint set, bit allocator,
cathedral/autopilot consumer, continual-learning posterior, and a disambiguator whenever two
formulations remain defensible. This memo does not write those shared surfaces because its authority
is advisory-only.

## 13. Stores and research consulted

Local authoritative surfaces:

- `CLAUDE.md`, `AGENTS.md`, top-10 Claude memory, current lane/subagent/directive/canonical surfaces;
- `ADVISORY_RESTART_HANDOFF_v752_v753_v8_20260710.md`;
- the evaluator, PR #128, v7.5.3, v8, ChromaRung, R1, output-space inverse, frame0 inverse, Kalman,
  and full-stack synthesis memos linked above;
- `reports/latest.md`, canonical frontier pointer, lane registry, dispatch ledger, and live run logs;
- current trainer, byte-close receiver, texture-trunk, decoupled-field, and DSL sources.

Primary research used as context, not as score authority:

- task-aware image coding: <https://arxiv.org/abs/2108.09993>
- task-oriented lossy compression: <https://arxiv.org/abs/2405.04144>
- offline inversion/distillation precedent (DeepInversion):
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html>
- implicit representation controls: SIREN <https://arxiv.org/abs/2006.09661>, WIRE
  <https://arxiv.org/abs/2301.05187>, BACON <https://arxiv.org/abs/2112.04645>.

These papers support possible basis/generator choices. None establishes contest legality,
receiver closure, or a favorable score for this task.
