# ADVISORY — SDF epoch-31 evidence delta: modulation geometry, hybrid costates, and exact exits — 2026-07-10

```yaml
schema: advisory_sdf_moddim_hybrid_costate_delta_v1
observed_at_utc: 2026-07-10T21:10:19Z
final_observed_at_utc: 2026-07-10T21:19:04Z
lane_id: lane_advisory_codex_v752_v753_v8_fresh_eyes_20260710
lane_scope: research_only
parent_advisory: .omx/research/ADVISORY_sdf_postclosure_delta_watcher_ep25_curriculum_costate_20260710.md
parent_advisory_sha256: 34abfc358638ce237cc2f593de9ef04f336fbff1fb8f7f0c1e1f442a6c3a98ae
delta_status: ADVISORY_EVIDENCE_DELTA_CLOSED
engineering_status: OPEN_BLOCKED
engineering_gates_passed: []
pointer_delta: 0
execution_authority: none
launches_by_this_unit: 0
evals_by_this_unit: 0
inflations_by_this_unit: 0
dispatches_by_this_unit: 0
harvests_by_this_unit: 0
signals_by_this_unit: 0
processes_stopped_by_this_unit: 0
owned_output: .omx/research/ADVISORY_sdf_ep31_moddim_hybrid_costate_delta_20260710.md
```

## 0. Result

This append-only delta records the first live modulation-coordinate intervention map from the current
v7.5.2 run, the post-advisory apparatus repair at `f41d54769`, a fresh PR128 authority check, and a
more rigorous costate/curriculum architecture for the original task-space SDF witness program.

The conclusions are deliberately narrow:

1. The epoch-25 `mod_dim_ablation` row is a useful **probe-order signal**, not evidence of an
   eight- or nine-dimensional intrinsic witness, not a safe pruning mask, and not a bit-allocation
   verdict. It is basis-dependent, Seg-only, 32-pair, pre-receiver attribution.
2. The delayed epoch-25 n600 advisory verdict arrived while this document was under validation. It
   is worse than the epoch-2 row by `+5.3202` full score, and its new `EMA_BEST` pointer selects only
   realized `d_seg`, not complete score. It is a retained deploy candidate, not a score champion.
3. The lane-band event fired at epoch 31 from epoch-25 nucleus evidence with an explicit six-epoch
   sensor lag. This is direct evidence that curriculum transitions are hybrid, delayed-data events;
   it does not supply a reset/saltation costate or a stage-boundary checkpoint.
4. A coordinate costate is not well-defined across the witness stack's hybrid events without typing
   each event. Scheduled resets need a discrete reset VJP; transverse state-triggered guards need a
   saltation update; uint8, argmax, topology and coder-length faces need exact matched-edge score
   differences rather than a fabricated smooth derivative.
5. The apparatus repair is real but does not close debt. It fixes dashboard schema drift and aligns a
   stale supersampling test with the measured disqualification, while ratcheting historical HOSC debt
   from 8 to 9 and DSL-authoring debt from 4 to 5. Those increases are evidence that migration debt
   grew, not passes.
6. PR128 remains an open, unreviewed, unratified **HNeRV-family payload-polish child**. It is a
   valuable control and technique donor; it is not a new representation family and does not redirect
   the primary program away from task-space SDF/level-set witnesses.
7. No R0–R8 gate passes. No v7.5.3 or v8 launch becomes admissible. The current v7.5.2 process is
   live and must remain undisturbed.

```text
ADVISORY DELTA: CLOSED
ENGINEERING R0-R8 PASSES ADDED: NONE
NEW AUTHORITATIVE SDF SCORE ROWS: NONE
VEHICLE PROMOTIONS: NONE
PR128 FAMILY RECLASSIFICATION: NONE — HNERV FAMILY
POINTER DELTA: ZERO
LIVE RUN DISPOSITION: LIVE / OBSERVE-ONLY / DO NOT SIGNAL
```

## 1. Authority and custody snapshot

### 1.1 Repository, pointer, and owned surface

| surface | observed value | authority reading |
|---|---|---|
| branch | `main` | sole source of truth |
| `HEAD` | `f41d54769f3aed0c2eef241379e7b56c256ea6da` | concurrent apparatus-fix landing |
| `origin/main` | `f41d54769f3aed0c2eef241379e7b56c256ea6da` | matched at observation |
| `CLAUDE.md` | SHA-256 `52405bac18c6227df1d99b597a2f55987614f035e501eb74a9d08be48e1dbdd7` | full campaign preflight authority |
| `AGENTS.md` | SHA-256 `d2bdceb42d394d78bac4f9ddcaa9e0b3758d0be206fea2920784ccdc6f2ec495` | full campaign preflight authority |
| restart handoff | SHA-256 `48c837ded5858c7cbfa6a9ffff1d5afd8ff02e30f7b9523e2e7a15c5fec6856d` | durable continuation contract |
| closure matrix | SHA-256 `acd002dda2e07666000ce56ac445855b823873e8cff076465ba2bc44cab4909b` | R0–R8 disposition parent |
| canonical pointer file | SHA-256 `6111c56e68fc51c914bda6cad7b20b499087dc74e9fd41922e7f47fdf572bc90` | unchanged by this unit |
| CPU pointer | `0.19108282419209976 [contest-CPU]`, 177,169 B, archive SHA `ad02b012...d079c` | PR110 click-polish lineage, not SDF |
| CUDA pointer | `0.20533002902019143 [contest-CUDA]`, archive SHA `9cb989ce...7cf4` | different archive and lineage |

The shared dirty set was five `.omx/state/` ledgers plus untracked `paper/__marimo__/`. During final
validation, concurrent run-artifact migration WIP also appeared in
`src/tac/witness_control/dynamics_analyzer.py`, `trace_probes.py`, and
`src/tac/witness_dsl/campaign.py`. It replaces hardcoded filenames with canonical constants but is
uncommitted and carries no gate authority here. All of this state belongs to other lanes. This unit
neither stages nor edits it. The only owned mutation is this advisory file.

### 1.2 Live v7.5.2 run

| surface | value | disposition |
|---|---|---|
| launcher | PID `88029`, alive | do not signal |
| trainer | PID `88030`, alive | do not signal |
| run directory | `experiments/results/levelset_v752_baseline_20260710T185913Z` | shared owner custody |
| progress at final snapshot | loss telemetry through epoch 31 | training telemetry only |
| latest verdict row | epoch 25, async n600 CPU telemetry; explicit `axis` field absent | `AXIS_UNSTAMPED / NON-PROMOTABLE` |
| latest rolling checkpoint | epoch 25 | partial crash-resume evidence only |
| latest preserved deploy candidate | epoch-25 `levelset_witness_ema_BEST.npz` | d_seg-only selection, not full-score best |
| latest pose-conditioning row | epoch 30, `DEGENERATE_GUARD_TRIPPED` | advisory selector signal only |
| latest curriculum event | lane render-band engaged at epoch 31 from epoch-25 sensor data | exact hybrid event; no authority upgrade |

The run directory now also contains `levelset_best.json` and `levelset_witness_ema_BEST.npz`. It
still contains no `archive.zip`, complete `.raw`, LVLS1 center, immutable run manifest,
stage-encoded checkpoint, or contest-axis receipt.

The click-polish Modal call `fc-01KX6DZWCHNPQ6KN59V2MZ845J` remains sister-owned and recorded as
active. This unit did not poll, harvest, cancel, duplicate or mutate it.

## 2. Exact live evidence after the parent advisory

### 2.1 Checkpoint identity remains unchanged

| file | bytes | SHA-256 | narrow status |
|---|---:|---|---|
| `levelset_resume_state.npz` | 1,902,570 | `651b84e503323d96430694609292c611a86583537c3bf031d7b6c3bb0d366f3c` | rolling epoch-25 resume state |
| `levelset_witness_ema_mlx.npz` | 482,472 | `076712589886fb7b038d4b45550f3191c719af5d8c07a6e3e4ab066c45461214` | rolling epoch-25 EMA |
| `levelset_witness_ema_BEST.npz` | 482,112 | `440b7e7b8fcd4003b0ae8f333da3932e172442068fc251d51f9dfa424dcd9bfe` | preserved epoch-25 d_seg-selected deploy candidate |
| `levelset_best.json` | 125 | `87f5dee655fc163d22e56c2f76dbc425a6b89486b2e9da3ca514f0b092a0c934` | mutable pointer to that deploy candidate |

The parent advisory's partial resumability disposition therefore stands. `EMA_BEST` is preserved
across ordinary non-improving verdicts, but it is deploy-only and selected on d_seg. No immutable
stage-named resume copy, dirty-file manifest, exact continuous-versus-resume equality receipt, or
six-leaf R8 branch packet has appeared.

### 2.2 Epoch-26 pose gate

The live row is:

```text
epoch=26
classification=DEGENERATE_GUARD_TRIPPED
fired=false
should_ship_banked_r1=true
actuated=true
axis=[macOS-MLX advisory] NON-PROMOTABLE
```

`actuated=true` means the controller path is enabled. It does **not** mean that a banked-R1 artifact
was selected, byte-closed, grafted, compared, or rolled back. The source now refuses to let the
epoch-726 backstop override a degenerate banked-R1 decision, which is a useful source-level repair.
There is still no banked endpoint pair or joint Seg/Pose/full-byte receipt.

The run remains attribution-confounded relative to the sealed clean-rung intent: it includes chroma,
explicit `--epochs 3000`, Muon and Pose backstops at epoch 726, and the temporal-screw force. Startup
reports `preset: "none"`; this is not a named amber-admission receipt. None of these facts authorizes
stopping a live operator-approved process. They constrain what may later be claimed from it.

### 2.3 Delayed epoch-25 verdict and d_seg-only `BEST`

The asynchronous n600 CPU verdict completed at `2026-07-10T21:17:41Z`, 2,194.3 seconds after
scheduling:

| row | `d_seg` | `d_pose` | blob bytes | implied `S` | explicit axis field |
|---|---:|---:|---:|---:|---|
| epoch 2 `baseline_v0` | `0.041123` | `6.500110` | 91,397 | `12.2355` | `[macOS-CPU advisory] NON-PROMOTABLE` |
| epoch 25 `unify_tau` | `0.042281` | `17.611089` | 85,467 | `17.5557` | absent |

The component changes from epoch 2 to epoch 25 are:

```text
Seg score contribution:  +0.115800
Pose score contribution: +5.208352
rate contribution:       -0.003949  (5,930 fewer bytes)
complete Delta S:        +5.320203  (worse)
```

The epoch-25 row records `verdict_device="cpu"` on this macOS host, but it omits the mandatory
`axis` tag. Its narrow status is `AXIS_UNSTAMPED_CPU_TELEMETRY / NON-PROMOTABLE`, not a contest-CPU
score.

The trainer then wrote `levelset_witness_ema_BEST.npz` because `_best` had no prior candidate and
`_is_new_best()` compares only finite realized `d_seg`. `levelset_best.json` stores only `d_seg`,
epoch, path and timestamp. It omits `d_pose`, bytes, full score, axis, receiver identity and archive
identity. The source comment accurately calls this a best realized-d_seg deploy pointer. Downstream
surfaces must not shorten that to “best score,” especially because this first stored candidate is
worse than the epoch-2 row in d_seg and full score.

This is a new apparatus defect, not a corrupt artifact: the bytes are useful and retained, but the
selection order is partial. A future full-score selector must compare byte-closed endpoint tuples and
keep the d_seg-best label distinct.

### 2.4 Epoch-31 lane-band event

The first live curriculum actuation in this delta is:

```text
transition=lane_band
epoch=31
sensor=lane_nucleus
sensor_data_epoch=25
sensor_lag_epochs=6
fired_by=event
cap=500 (not reached)
```

The event then emitted `lane_render_band_engage` and cleared recent losses. The epoch-25 handoff row
had `nucleus_all_ok=true` but `plateau_ok=false` and `ready=false`; the lane-band guard consumes its
narrow lane-nucleus predicate rather than the whole handoff predicate. The event telemetry is a real
positive: it records the deciding sensor epoch and lag. It is also the exact case that invalidates a
single smooth costate. At the final snapshot there was no event-boundary stage checkpoint, reset-VJP,
saltation receipt, matched no-event branch, or full-score consequence. The transition is therefore
`ACTUATED_TRAINING_EVENT / ATTRIBUTION_OPEN`, not an R8 pass.

## 3. Epoch-25 modulation map: what is measured

### 3.1 Exact row summary

The epoch-25 observer zeroed each of 32 modulation coordinates on a code copy and measured Seg
disagreement over `k_sample=32` pairs. It reported:

| field | value |
|---|---:|
| baseline `d_seg` | `0.027531` |
| positive `delta_d_seg` count | 21 |
| negative `delta_d_seg` count | 11 |
| sum of positive coordinate deltas | `0.015336` |
| sum of negative coordinate deltas | `-0.001143` |
| top-eight normalized hint mass | about `0.66000` |
| authority | `[macOS-numpy advisory] NON-PROMOTABLE` |

The largest normalized hints are:

| rank | coordinate | zero-ablation `delta_d_seg` | normalized hint |
|---:|---:|---:|---:|
| 1 | 9 | `0.001703` | `0.10024` |
| 2 | 23 | `0.001528` | `0.09643` |
| 3 | 27 | `0.001501` | `0.09126` |
| 4 | 21 | `0.001372` | `0.08454` |
| 5 | 25 | `0.001302` | `0.07910` |
| 6 | 26 | `0.001254` | `0.07540` |
| 7 | 29 | `0.001136` | `0.06767` |
| 8 | 13 | `0.001092` | `0.06536` |

Coordinate 6 has the largest negative delta, `-0.000522`. Removing it helped the 32-pair Seg
observer in this one conditional intervention. That is a hypothesis about destructive interaction or
misallocation, not permission to remove the coordinate.

### 3.2 Co-temporal spectrum and intervention signals remain different objects

The epoch-2 `mod_dim_dynamics` observer reported:

```text
participation-ratio effective_rank = 8.6773
k90 = 23
k99 = 31
D = 32
```

The delayed epoch-25 verdict later emitted a co-temporal dynamics row:

```text
participation-ratio effective_rank = 14.5650
k90 = 23
k99 = 31
top-eight energy fraction = 0.59182
```

The epoch-25 spectrum and epoch-25 ablation may now be paired as two observer signals from the same
checkpoint epoch. They are still not one additive decomposition. In particular:

- neither `8.6773` at epoch 2 nor `14.5650` at epoch 25 proves an intrinsic dimension;
- the changing participation ratio shows that the chart spectrum evolves during training;
- `k90=23` and `k99=31` exhibit a spiked spectrum with a consequential tail;
- the top-eight hint mass is 66% of the observer's normalized
  `FiLM-column-norm × |delta_d_seg|` heuristic, not 66% of score, mutual information, rate, or
  recoverable archive bytes; and
- the source's `19,424 B` epoch-2 and `21,808 B` epoch-25 estimates for 23/32 reductions are linear
  arithmetic, not compiled archive measurements.

The same-epoch signals may guide probe ordering. They cannot be multiplied together or used to
declare an intrinsic dimension.

## 4. Why raw-coordinate ablation is not geometric authority

Let decoder input be `W z`. For any invertible matrix `A`, the unquantized change of chart

```text
z' = A z
W' = W A^{-1}
```

preserves `W z`. Coordinate-zeroing, FiLM column norms and coordinate rankings generally change.
Participation ratio is invariant under orthogonal changes of basis, not arbitrary invertible
reparameterizations. Quantization and coding break this gauge by selecting a physical lattice basis,
but that makes the basis a **codec design variable**, not an intrinsic manifold coordinate system.

The measured intervention is

```text
Delta_i = d_seg(z with coordinate i set to zero) - d_seg(z).
```

It is a finite, off-manifold conditional effect. It is neither a derivative nor an additive ANOVA
component. Therefore `sum_i Delta_i` has no score meaning. A negative value can arise from a harmful
coordinate, cancellation with another coordinate, a chart artifact, quantization mismatch, or subset
noise.

This is exactly the setting in which interaction-aware attribution matters. Shapley effects were
developed to apportion shared effects under dependent inputs, while ordinary first-order Sobol-style
interpretations become ambiguous. That literature motivates a probe design; it does not make a
Shapley estimate contest authority. See [Iooss and Prieur, Shapley effects with correlated inputs](https://arxiv.org/abs/1707.01334).

## 5. Receiver/rate generalized geometry

### 5.1 Proposal metric

Replace raw coordinate importance with a pair of local quadratic forms:

```text
G_task = E[J_z^T M_task J_z]
G_rate = declared covariance / quantizer / entropy metric, positive definite
G_task v_i = lambda_i G_rate v_i.
```

`M_task` must include both Seg margin geometry and Pose output geometry on both frames. For hard
argmax and integer receiver faces, it is only a proposal metric. Under a congruent chart change, the
generalized spectrum is invariant when both forms transform together.

If `T` contains tangent generators of a known parameter gauge, a horizontal projector can be defined
in a positive task/rate metric `M`:

```text
P_H = I - T (T^T M T)^dagger T^T M.
```

Attribution should act on receiver-compilable horizontal modes `P_H v`, not arbitrary raw
coordinates. This does not eliminate the need to compile and score endpoints.

### 5.2 Receiver-closed attribution protocol

For each proposed mode `v_i` and each signed quantized action `q`:

1. start from one immutable archive/receiver parent;
2. compile `+q v_i`, `-q v_i`, and unchanged controls into legal complete archives;
3. parse every archive through the frozen receiver to EOF;
4. inflate all 1,200 frames and record raw/support hashes;
5. measure per-class Seg, real per-pair Pose from both frames, and exact archive bytes;
6. compute the complete nonlinear score difference;
7. repeat fresh enough times to derive `epsilon_det(receiver)`;
8. retain the endpoints and exact inverse/rollback;
9. repeat on a preregistered held-out partition and later the retained n600 center; and
10. admit only if the upper confidence bound clears the preregistered negative threshold.

The exact verdict remains:

```text
Delta S_i(q) = S(archive_i(q)) - S(parent_archive).
admit iff UCB95(Delta S_i(q)) < -delta_admit.
```

No proxy coordinate is pruned merely because its 32-pair Seg-only `Delta_i` is negative.

### 5.3 Interaction closure

Before diagonal waterfilling, measure pairwise interaction residuals on the prioritized modes:

```text
I_ij = Delta S(i,j) - Delta S(i) - Delta S(j).
```

If material residuals remain, retain an interaction graph or hypergraph and select compatible sets;
do not force an additive allocator. For dependent latent groups, define coalition states through a
receiver-legal conditional reconstruction, not independent zero fill. Bootstrap or repeat-derived
confidence intervals must include evaluator determinism and subset variation.

Only after interaction closure may a bit allocator consume a score-unit-per-byte value. Since the
full contest score already contains the exact byte term, the safest acceptor is complete `Delta S`,
not a separately tuned rate proxy.

## 6. Hybrid costates for the curriculum/controller

### 6.1 Why one smooth adjoint is false

The complete witness program is hybrid. Its state includes at least:

```text
x = {live parameters, EMA, optimizer moments, RNG/data order, latent codes,
     Lagrange multipliers, topology state, controller counters, curriculum mode,
     receiver fingerprint, archive grammar and selected endpoint}.
```

Its objective has continuous training regions separated by scheduled optimizer/head/tau switches,
state-triggered conditioning/birth events, and receiver-stratum jumps at uint8 bins, argmax ties,
topology changes and coder-length boundaries. The exact score is locally constant or discontinuous
on many continuous parameter directions. Calling its ordinary gradient an exact contest costate
would therefore be false.

### 6.2 Three event types

#### A. Scheduled reset

Muon switches, fixed-epoch head changes and scheduled tau changes have fixed event time. If the
composed step is `F_e = R_e o F_q`, use the discrete reset VJP:

```text
p_k = (D F_e)^T p_{k+1} + grad l_k.
```

There is no saltation denominator for a fixed-time event.

#### B. State-triggered guard

For a guard `h(x,t)=0` with reset `R`, saltation is admissible only when the crossing is bracketed
and transverse:

```text
Xi = D_x R
     + ((f_plus - D_x R f_minus - partial_t R) n^T)
       / (n^T f_minus + partial_t h),
n = grad_x h,
p_minus = Xi^T p_plus + grad c_event.
```

If the denominator lies inside a measured numerical floor, the event sensitivity is
`UNIDENTIFIABLE`. Plateau and conditioning gates cannot silently manufacture a finite costate at a
tangent or degenerate crossing. Saltation matrices are the standard sensitivity update for hybrid
jumps; see [Kong et al., Saltation Matrices](https://arxiv.org/abs/2306.06862).

#### C. Receiver-stratum event

At a uint8-bin crossing, argmax tie, ZIP-length change, birth/death/merge or other discrete receiver
face, use an exact matched-edge record:

```text
{parent archive, child archive, inverse, Delta d_seg by class,
 Delta d_pose by pair, Delta bytes, Delta S, support/topology hashes}.
```

Do not report a smooth costate for this edge. It is a typed critical-face atom.

### 6.3 Event collisions and reverse order

Colliding events compose in their recorded forward order; adjoints compose in reverse order. A
Muon/tau/pose/topology collision without an immutable order stamp is not identifiable. The R8
six-leaf design must therefore name the common ancestor, ordered event word, reset hashes and
receiver identity for every leaf.

### 6.4 Current costate apparatus remains blocked

The current `CostateEstimate` defines `.status`, while `record_run_costates()` reads `.tier`. Real
objects are skipped by the posterior writer. A test fake carrying `.tier` masks the production
boundary. `PARTIAL` can also reach the record path without a proved link to full score, and posterior
readout discards evidence class.

Consequently the curriculum pool's costate digest is **SENSE-only**. It cannot actuate a vehicle.
Before any controller consumes a posterior, the engineering owner must:

1. make the schema one-fact/one-field (`status` or a versioned replacement);
2. fail closed on `PARTIAL` and `UNIDENTIFIABLE` for actuation;
3. retain method, units, event type, parent/child hashes and receiver identity;
4. include the curriculum candidate's semantic flags in checkpoint provenance and the F2 resume
   divergence guard;
5. reject an additive-margin head whose positive margin-field weight is absent;
6. preserve candidate-pool schema validation on read as strongly as on write; and
7. require held-out sign calibration before converting advice into an action.

## 7. Topology and group structure in the same controller

The topology losses already present are proposal forces. Soft-clDice is designed to preserve
connectivity in tubular structures, and differentiable persistent-homology losses can encode Betti
priors. These results justify useful training surrogates; they do not prove through-R argmax topology
or contest-score benefit. See [clDice](https://arxiv.org/abs/2003.07311) and
[persistent-homology segmentation loss](https://arxiv.org/abs/1910.01877).

For the SDF witness stack, every proposed mode or curriculum transition should separate four
supports:

```text
bulk class interior
pairwise zero-set/interface annulus
triple-or-higher junction complex
Morse/topology critical face.
```

For v8, pairwise class differences must remain integrable. If `psi_cd = u_c - u_d`, then oriented
cycle sums vanish at class junctions. Quantized edge carriers should be projected back to a global
class-potential chart or rejected when cocycle residual survives parse-back. Group-equivariant
camera transport is only a hypothesis because resize, uint8 and the frozen scorers break ideal
symmetry; record the equivariance residual and let exact endpoint score arbitrate.

Topology events belong to event type C above. A birth that improves a soft persistence loss but fails
to survive uint8/resize/argmax is not a birth in the authority complex.

## 8. Post-advisory apparatus commit `f41d54769`

| change | narrow positive | remaining defect / gate effect |
|---|---|---|
| dashboard `_TRAJ_KEYS` and `_slim` share one schema | prevents declared/actual telemetry drift and internal-field leakage | observability only; no R gate pass |
| supersample test now records disqualification | aligns the test with measured train/decode and observation mismatch | does not prove IPE or another AA mode beneficial |
| HOSC ratchet `8 -> 9` | prevents silent growth beyond the new baseline | confirms another historical artifact entered debt; exit remains zero future violations plus documented immutable history |
| DSL-authoring ratchet `4 -> 5` | keeps the larger migration queue visible | debt grew; typed-DSL closure is farther away, not passed |

This commit fixes three test failures. It does not repair the costate field mismatch, create a run
manifest, close receiver endpoints, produce an SDF center, or authorize a launch.

## 9. PR128 fresh authority check

Fresh read-only GitHub evidence for [PR #128](https://github.com/commaai/comma_video_compression_challenge/pull/128):

| field | observed value |
|---|---|
| state | open, not draft, unmerged |
| title | `rhnerv_latent_polish (0.187991)` |
| head | `3eb39cac8261075888b1c562e9d9c2a7f1c7aebf` |
| claimed archive | 176,531 B, SHA `cfd941de10e5c27a5c855f97b0c84e39f6171f23c53c150e4afd90915f41e395` |
| reviews/checks | none |
| maintainer evidence | bot notice only; maintainer must trigger evaluation |
| release tag target | stale commit `ea478f64f230111e20f78f736673933c15b8ca49` |

Two existing SSD archive copies match the claimed size and SHA; no download was needed in this
delta. The tag still does not identify the current head that corresponds to the served asset.

Family classification remains exact: frozen PR95/PR101 HNeRV decoder lineage, PR110 selector,
PR112 coder/container, plus native uint8 latent polish and sidecar folding. It is
`EXTERNAL_UNRATIFIED_HNERV_FAMILY_SIGNAL`, not a new representation family. Transferable techniques
remain exact-gated discrete endpoint search, deterministic packing, receiver/runtime hardening,
sidecar folding and explicit custody. The target representation remains the SDF witness stack.

## 10. R0–R8 delta and executable falsifiers

| gate | new evidence | immediate falsifier / required next artifact | disposition |
|---|---|---|---|
| R0 | rolling checkpoint unchanged | any parent/config/RNG/receiver mismatch; still need retained n600 archive and three deterministic replays | open |
| R1 | none at archive authority | untracked live state or non-consumed compiled mode; need signed actual-archive controls | open |
| R2 | epoch-25 observer is Seg-only | any admission without both frames, real per-pair Pose, per-class Seg and actual bytes | `ACTUATION_REFUSE` |
| R3 | no new center | epoch-2 spectrum cannot be paired with epoch-25 ablation as one center; need retained Linux contest-CPU n600 center | open/blocking |
| R4 | no five-state cell | unequal destinations after inverse imply `NONCOMMUTING_NO_2_CELL` | missing/not authorized |
| R5 | no 24-state RQTD | raw coordinates without rank/condition/inverse/gauge stress are not signed modes | missing/not authorized |
| R6 | no held-out predictor | hybrid predictor must beat no-jump ablation and avoid material wrong-sign edges | missing/not authorized |
| R7 | topology surrogates only | through-R extinction or unregistered collateral topology falsifies that atom only | missing/blocking |
| R8 | epoch-25 rolling ancestor candidate; epoch-30 degenerate gate; epoch-31 lane-band event | no boundary checkpoint/reset or matched no-event branch; wrong event calculus or no held-out sign calibration blocks attribution | open/not authorized |

No row is an engineering pass.

## 11. Exact new exit predicates

### 11.1 `MOD_DIM_CLOSED`

This predicate passes only when all are true:

1. the parent is one retained, R0/R1-closed n600 SDF archive;
2. modes are defined in a declared receiver/rate metric, with gauge stress recorded;
3. signed actions compile to complete legal archives with actual coder bytes;
4. all counted state is consumed and no live learned state escapes the archive;
5. three fresh decodes are byte-identical;
6. full per-class Seg and real per-pair two-frame Pose close to their aggregates;
7. interaction residuals are below a replay-derived materiality floor or explicitly modeled;
8. held-out and n600 results agree in sign within preregistered uncertainty;
9. the exact inverse restores archive/raw/support/topology identity; and
10. `UCB95(Delta S) < -delta_admit`, with `delta_admit` derived from repeatability rather than guessed.

Raw coordinate rankings, participation ratio, linear byte estimates and 32-pair Seg-only ablations
can never satisfy this predicate by themselves.

### 11.2 `HYBRID_COSTATE_IDENTIFIED`

This predicate passes only when:

1. every curriculum transition is typed as scheduled reset, transverse guard, or receiver face;
2. the complete state and receiver identity are checkpointed on both sides;
3. scheduled reset VJPs match finite differences above a derived noise floor;
4. each guard crossing is bracketed, transverse and saltation-predicted, otherwise explicitly
   `UNIDENTIFIABLE`;
5. receiver faces carry exact matched-edge score/support/topology receipts;
6. event collisions carry immutable forward order and reverse adjoint order;
7. the `.status`/`.tier` schema contradiction is extinct with real-object regression coverage;
8. no `PARTIAL` or `UNIDENTIFIABLE` estimate can actuate;
9. held-out material edges have correct predicted sign; and
10. the hybrid predictor beats its no-jump ablation:

```text
UCB95(error_hybrid - error_no_jump) < -epsilon_pred,
```

where `epsilon_pred` comes from replay and finite-difference noise.

### 11.3 `CONTROLLER_ACTUATION_ADMISSIBLE`

In addition to `HYBRID_COSTATE_IDENTIFIED`, controller actuation requires:

- a retained common ancestor and rollback packet;
- candidate semantics in DSL and resume guards;
- across-seed evidence or a registered variance model;
- no collision ambiguity;
- complete full-score endpoint measurement; and
- `UCB95(Delta S) < -delta_admit` on the authority axis.

The current controller satisfies none of these complete conjunctions.

## 12. Prioritized roadmap from this delta

The order is dependency-driven:

1. **Preserve, do not disturb, the current run.** Observe later verdicts and checkpoints only under
   the owning lane; this advisory creates no watcher, signal or harvest.
2. **Close R0 custody.** Retain one complete n600 SDF archive, receiver, config, dirty manifest,
   checkpoints, legality scan and triple decode.
3. **Close R1 actual consumption.** Compile negative and signed positive controls for every counted
   optional group through the exact receiver.
4. **Repair R2 before optimizing coordinates.** Eliminate scalar-expanded Pose, evaluate both
   frames, carry exact bytes and accept only complete nonlinear score.
5. **Establish R3.** Obtain one retained Linux contest-CPU n600 center; keep CPU/CUDA axes separate.
6. **Replace coordinate hints with generalized modes.** Freeze one checkpoint/receiver, derive the
   task/rate pencil, run signed receiver-closed probes, then interaction closure.
7. **Repair costate persistence and event typing.** Extinguish `.tier`/`.status`, register all semantic
   resume keys, and validate reset/saltation/critical-face calculus in shadow mode.
8. **Run R4/R5 only after R3.** Use one generalized mode and one topology/codec action before
   scaling the atlas.
9. **Promote topology atoms only through R.** clDice/persistence/Morse signals nominate atoms;
   exact argmax topology and score decide them.
10. **Use v7.5.3 as an exact-D carrier build, not a launch.** The frame1 luma-null home law,
    camera-grid preimage, byte-close consumption and Pose equality remain prerequisites.
11. **Use v8 only after v7.5 closure.** Enforce global-potential integrability, class isolation,
    receiver sections, full rate law and public-apparatus defects before training.
12. **Treat PR128 as a control.** Reuse licensed/independently implemented techniques only with
    explicit provenance; do not transplant its representation as the primary program.

The existing century plan remains the long-horizon authority. This delta sharpens the immediate
measurement and controller exits; it does not replace that plan.

## 13. Literal launch and vehicle dispositions

| surface | literal disposition |
|---|---|
| current v7.5.2 PID 88029/88030 | `LIVE / OBSERVE-ONLY / DO NOT SIGNAL OR RESTART` |
| current v7.5.2 promotion | `HOLD` — d_seg-selected deploy candidate is not archive/full-score/axis authority |
| another v7.5.2 launch | `HOLD` — attribution/config/custody chain not closed |
| v7.5.3 | `DESIGN/BUILD-ONLY`; no training EVENT |
| v8 | `HOLD TRAINING EVENT`; v7.5-first gates remain binding |
| SDF finisher | `ACTUATION_REFUSE` until R2 full-score contract is repaired |
| modulation pruning/bit allocation | `PROBE-ORDER ONLY`; no action from epoch-25 row |
| costate controller | `SENSE-ONLY / NO ADVISORY-PROMOTION ACTUATION`; live lane-band training event remains owner-controlled |
| PR128 | `EXTERNAL_UNRATIFIED_HNERV_FAMILY_SIGNAL` |
| click-polish Modal claim | sister-owned; no poll/harvest/cancel by this unit |
| pointer | unchanged |

## 14. Exact remaining blockers

1. no retained authoritative n600 SDF archive, immutable manifest or triple decode;
2. no actual-archive signed receiver controls for all counted groups;
3. finisher still omits true joint nonlinear Seg/Pose endpoint authority and can expand aggregate
   Pose into a fabricated vector;
4. no Linux contest-CPU n600 LVLS1 center;
5. no executed R4/R5 atlas, inverse packet or identifiability receipt;
6. no versioned complex, Hodge solve or held-out hybrid predictor;
7. no typed reversible topology atom surviving through-R;
8. no collision-stamped six-leaf R8 experiment from one immutable ancestor;
9. epoch-25 dimension evidence is basis-dependent, 32-pair, Seg-only and pre-archive;
10. costate persistence reads `.tier` while production estimates expose `.status`;
11. curriculum candidate semantic flags are not fully protected by checkpoint/F2 resume guards;
12. typed-DSL debt increased from 4 to 5 and historical HOSC debt from 8 to 9;
13. the live run remains attribution-confounded; epoch-25 full score is `+5.3202` worse than epoch 2,
    its `BEST` pointer is d_seg-only and axis-unstamped, and the epoch-31 lane-band event lacks a
    boundary/counterfactual attribution packet;
14. v7.5.3 exact-D, Pose-null and byte-close laws remain unproved;
15. v8 integrability, isolation, receiver grammar and rate closure remain unproved; and
16. PR128 has no maintainer evaluation/review and its release tag still targets stale source.

These are engineering blockers, not evidence that the SDF family is dead.

## 15. Triality and stores consulted

### DSL leg

Future generalized modes and curriculum event semantics must enter the typed DSL, checkpoint
provenance and F2 resume-divergence guard together. The epoch-25 observer remains read-only and must
not be silently converted into a flag.

### DAG leg

```text
R0 custody
  -> R1 receiver bijection
  -> R2 joint full-score finisher
  -> R3 authoritative center
  -> generalized-mode signed probes
  -> interaction closure
  -> R4/R5 atlas
  -> topology/hybrid predictor R6/R7
  -> collision-stamped R8 controller
  -> v7.5.3 exact-D and v8 per-class descendants.
```

### Equation leg

The durable equations introduced or sharpened here are:

1. receiver/rate generalized eigenproblem `G_task v = lambda G_rate v`;
2. metric horizontal projection `P_H`;
3. receiver-closed full-score `Delta S` admission;
4. pairwise interaction residual `I_ij`;
5. discrete reset adjoint;
6. transverse saltation adjoint; and
7. exact critical-face endpoint certificate.

### Stores consulted

- full `CLAUDE.md` and `AGENTS.md` campaign preflight;
- top-10 Pact Claude memory entries;
- current lane registry, subagent progress, recent directives, dispatch claims and canonical pointer;
- the restart handoff and all preceding SDF/vehicle/PR128/curriculum/costate advisories;
- v7.5 and v8 normative SPECs;
- live run log, launch script and rolling checkpoint metadata;
- `src/tac/boundary_math/mod_dim_dynamics.py`;
- `src/tac/witness_control/costate_estimator.py` and `costate_posterior.py`;
- `src/tac/through_r/mc_finisher.py`;
- `src/tac/tests/test_confound_gates.py` and commit `f41d54769`;
- frozen evaluator/video/scorer custody surfaces;
- fresh read-only GitHub PR128/release metadata and existing SSD intake bytes; and
- the four primary research papers linked above.

No external source code was copied. The research sources informed the mathematical advisory only;
any future OSS reuse still requires file-level license and provenance review.

## 16. Pointer-delta honesty

```text
canonical CPU pointer before this unit: 0.19108282419209976 [contest-CPU]
canonical CPU pointer after this unit:  0.19108282419209976 [contest-CPU]
canonical CUDA pointer before/after:     0.20533002902019143 [contest-CUDA]
SDF authoritative score rows added:     0
R0-R8 engineering passes added:         0
launches/dispatches/evals/signals:       0 / 0 / 0 / 0
pointer delta:                           0
```

The advisory delta is closed. The engineering campaign remains open and blocked on the exact
artifacts enumerated above.
