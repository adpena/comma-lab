# Codex findings — task-space semantic/realization decoupling

Date: 2026-07-26

Status: `research_only=true`; `[macOS-CPU frozen-scorer advisory]`; no n600,
contest-axis, candidate, promotion, score, or pointer-mutation claim.

## Verdict

The current exact-target semantic `G` is a diagnostic control, not a live
codec prerequisite.  It closes an internal label table that is not the frozen
evaluator's realized SegNet argmax surface.  Treating that closure as mandatory
conflates two different coordinates:

1. topology: which task-space regions the receiver intends to render; and
2. realization: which uint8 RGB witness survives `R` and the frozen scorers.

The real n2 evidence falsifies the implication

```text
internal semantic debt decreases  =>  realized evaluator debt decreases.
```

It does not close task-space topology, V9 primitives, G8 realization, A3, or
the original codec family.  It changes their composition order and admission
authority.

## Exact evidence

Primary receipt:
`.omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_n2_taskspace_stage_ablation_macos_cpu_advisory.json`

- receipt bytes: `10,027`
- receipt SHA-256:
  `2af9b50d70f342224aa438e95b4d53a05be3f253709c1fa1835da089f37e0f61`
- axis: `[macOS-CPU frozen-scorer advisory]`
- pair count: `2`; realized Seg sites: `2*384*512 = 393,216`
- target cache SHA-256:
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
- dynamic competitive-target snapshot: `0.172`, pointer SHA-256
  `928ff5f28147cfdc59576c191d7d709c2a0194a09cf89e26efaac1566cd3ddca`

| realized chronology | d_seg | realized Seg errors | d_pose | distortion score without rate |
|---|---:|---:|---:|---:|
| P only | 0.0033213298302143812 | 1,306 | 158.092041015625 | 40.0929228413 |
| P + exact-semantic G overlay | 0.0202229805290699 | 7,952 | 121.45529174804688 | 36.8727344559 |
| P0 + exact target Y1 | 0 | 0 | 10.575891494750977 | 10.2839153510 |
| exact target Y0 + G Y1 | 0.0202229805290699 | 7,952 | 15.52652359008789 | 14.4828452279 |
| exact target Y0 + predictor Y1 | 0.0033213298302143812 | 1,306 | 57.28706359863281 | 24.2668491081 |
| exact target Y0/Y1 control | 0 | 0 | 8.488579856182781e-16 | approximately 0 |

The internal semantic compiler reports `60,217 -> 0` debt cells.  Before G,
that internal debt is `15.313975%` of the n2 grid, while the actual P-only
SegNet disagreement is only `0.332133%`.  After exact internal closure, actual
SegNet disagreements increase by `6,646` cells.

This does **not** mean the semantic overlay is pure harm.  Before rate, it
improves the joint distortion term from `40.0929228413` (P only) to
`36.8727344559` (P + exact-semantic G), a gain of `3.2201883854` score units,
because the Pose improvement from `158.0920` to `121.4553` outweighs the Seg
regression.  Reading only the Seg component would suppress real joint signal;
making exact internal closure a prerequisite would suppress a different signal.
The allocator must preserve both facts and let the coupled finite differential
decide.

The counted exact-target G packet is `341,316` raw bytes.  A same-profile raw
DEFLATE section prefilter is `106,373` bytes for only two pairs.  The complete
P/G/PASS-A outer object is `191,838` bytes.  At the current `0.172` target,
even a zero-distortion n600 archive has a derived maximum of
`258,312.96432` bytes.  Exact-label row-span closure therefore cannot be the
production representation.

## What the evidence says geometrically

The scorer objective observes only the realized quotient:

```text
Q(P,G,A) = (argmax SegNet(R(Y1)), PoseNet(R(Y0),R(Y1)), exact archive bytes).
```

The internal semantic partition is a decoder coordinate, not an authority
coordinate.  A topology event has value only through its finite change in
`Q`; a same-class RGB realization can have value even when topology is
unchanged.  Conversely, exact intended labels can be evaluator-worse when the
palette, boundary support, resize numerator, or contextual SegNet response does
not realize those labels.

This yields a lossless four-way acquisition partition at each scorer cell.  Let
`T` be the target frozen label, `Z` the receiver's current semantic/topology
label, and `H=argmax SegNet(R(Y1))` the realized evaluator label:

| cell state | interpretation | default actuator |
|---|---|---|
| `Z=T`, `H=T` | closed | PASS/preserve |
| `Z=T`, `H!=T` | realization debt | same-class G8 RGB coordinate |
| `Z!=T`, `H!=T` | topology debt | sparse topology event, then realization |
| `Z!=T`, `H=T` | fortunate semantic mismatch | preserve unless an exact joint proposal proves value |

The last row is the orphaned signal that exact-semantic closure can erase:
an internally "wrong" coordinate may already lie in the correct frozen scorer
cell.  The acquisition ledger must count all four sets before and after each
proposal, plus off-support contextual spill where changing one support changes
`H` elsewhere.  No proposal is admitted from its addressed-cell gain alone.

The n2 chronology also orders the two frame roles.  Y1 realization is the
first large lever: substituting exact target Y1 while retaining P0 reaches
`d_pose=10.5759` and zero Seg debt.  Y0/A3 remains necessary, but it is a
conditional finisher bound to the post-G8 Y1, not an independently budgeted
stream.

## Required codec composition

The production search must preserve four explicit, allocator-visible modes:

1. `PASS_SEMANTIC_G`: use P topology and P Y1 exactly; a distinct nonempty,
   source-bound packet or envelope, never an empty/fake legacy G.
2. `REALIZATION_ONLY_G8`: keep the chosen topology but change Y1 RGB through a
   compact receiver-closed program.
3. `SPARSE_TOPOLOGY_G`: admit V9 boundary/event/island/worldsheet changes only
   by exact realized joint-score improvement, never by internal debt closure.
4. `TOPOLOGY_PLUS_REALIZATION_G8`: compose the two when their interaction earns
   a lower whole-object score.

For every G mode, compile A only after decoding the exact post-G8 Y1.  Whole
archive STORE and DEFLATE sizes, receiver output, `d_seg`, and pooled `d_pose`
must be remeasured for every accepted prefix.  No independent segment, pose,
or rate caps are authoritative.

## Atomic G8/A admission contract

The G8 realization packet and its conditional A packet are one proposal, not
two independently composable streams.  A legacy `TACA3P1` packet is bound to
the pre-G8 Y1 surface and is invalid beside a `TACG8S1` composite G.  Every G8
branch must therefore carry a versioned post-G8 `TACA8P1`, beginning with a
real `PASS_P0`, and the receiver must validate the exact post-G8 Y1 binding.

Alternative G8 families and acquisition orders are mutually exclusive
branches.  They must each be measured as a singleton against the same baseline;
they cannot be placed in one order-dependent greedy chain.  Greedy allocation
is valid only for genuinely additive prefixes inside one frozen branch.  For
each retained branch, copy-mode A rows are reranked from that branch's realized
Y1.  Pre-G8 copy rankings are controls, not an admissible production ordering.

A changes Y0 only.  All A variants under one exact G8 branch must therefore
have identical Y1 bytes, Y1 SHA-256, candidate Seg labels, and `d_seg`.  Any
difference is a receiver, cache-association, or scorer-measurement bug and
blocks the experiment.

The only finite admission inequality is the whole-object score differential:

```text
delta_S = 100*delta_d_seg
        + sqrt(10*d_pose_new) - sqrt(10*d_pose_old)
        + 25*delta_zip_bytes/37545489
```

For a distortion-improving proposal, its exact recompressed byte ceiling is:

```text
delta_B_max = -(37545489/25)
              * (100*delta_d_seg
                 + sqrt(10*d_pose_new) - sqrt(10*d_pose_old))
```

Raw packet estimates are useful prefilters only.  The G8 composite raw delta is
`314 + 13*runs` bytes; post-G8 A relative to PASS is `9*cells` bytes for
constant RGB and `6*cells` for Y1-copy.  Header and source-binding changes can
alter DEFLATE, so exact STORE and DEFLATE archive prices remain authoritative.

The first bounded measurement should screen each G8 family/order at geometric
prefixes `{1,4,16,64,256,1024,4096,7952}` with post-G8 PASS A.  Retain each
family/order optimum and its immediate neighbors, then test A prefixes
`{0,1,4,16,64}` for constant RGB and post-G8-reranked Y1-copy.  Report the
finite interaction
`I=S(G8,A)-S(G8,PASS)-S(G0,A)+S(G0,PASS)` so G8 value, A value, and coupling
are not conflated.

The first proposal ladder should be paired and geometric:

- P/PASS-G/PASS-A control;
- P/PASS-G plus class-shared and target-medoid realization controls;
- P/PASS-G plus bounded spatial/temporal realization bases;
- sparse topology alone;
- topology plus each realization prefix;
- each admitted post-G8 state with PASS-A, counted XIP2 A3, and then a compact
  generic residual coordinate.

The encoder may use frozen scorer outputs, target RGB, target labels, VJPs, and
inverse solves to acquire parameters.  The receiver and archive may contain
only the counted sufficient statistics and generic deterministic decoder.

## Guardrail and pending implementation

Until the distinct PASS-G path exists, any runner that makes exact-semantic G
mandatory must label it `exact_semantic_G_control` and cannot be called the
production allocator.  G8 compile-time same-class checks based on internal
labels are structural diagnostics; actual admission belongs to the frozen
through-R scorer callback.  The exact semantic control remains valuable as an
oracle/falsifier and must not be deleted.

Immediate implementation order:

1. land strict PASS-semantic-G plus G8 receiver composition;
2. bind A to exact post-G8 Y1;
3. run n2 paired controls through the nonlinear whole-archive allocator;
4. carry only measured-positive representations to n24 and then n600;
5. package the generic runtime and perform exact contest CPU/CUDA replay only
   after the n600 byte/distortion box closes.

## Stores consulted

- `CLAUDE.md`
- `AGENTS.md`
- `reports/latest.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- the primary n2 ablation receipt above
- `ep725_n2_causal_pga_control_receipt.json`
- `v9_target_partition_grammar_census_n600.json`
- `v10_ratecrush_phase1_20260719.md`
- `v10_frozen_space_surprises_20260719_codex.md`
- `ddm_pc2_pose_descent_smoke_result_20260725.json`

Pointer delta: none.

## Recovered inverse-solve signal and the A representation hierarchy

A cross-ledger reopen exposes the missing bridge between the new counted
P/G/A receiver and months of pose-inverse work.  The settled evidence does not
support either extreme -- neither "pose is already solved" nor "store an
inverse-solved frame0" is a production conclusion.  It supports a specific
successive-refinement hierarchy for A:

```text
post-G8 Y1
  -> zero-byte/protocol base choice
  -> counted quantized XIP2 trajectory, searched through the actual receiver
  -> only then a compact shared nonlinear residual chart
  -> never a per-pair dense inverse frame or scorer-derived basis table
```

The relevant prior measurements are:

- `joint_seg_pose_inverse_solve_receipt_n24_20260719.json` proves output-space
  reachability on 24 real pairs: `d_seg=0` and
  `d_pose=5.351929655623205e-10`.  Its actual range-coordinate description is
  `2,337,608.42 bytes/pair`, so it is an encoder-side hard-oracle teacher, not
  a candidate representation.
- `pose_frame0_inverse_solve_probe_20260703T0810Z.md` proves that the six PoseNet
  outputs are locally reachable through frame0 and that an STE uint8 solve can
  survive R.  The same ledger retracts its apparent cheap 96x128 image-space
  win: the n3 rate was scaled as n3 instead of n600.  The corrected n600 rate
  is `8.56` score units.  Dense or coarse per-pair inverse frames remain
  forbidden as a rate-dead/eval-hack formulation.
- The same probe measured a negative affine fit (`R^2=-0.215`) from physical
  xi to PoseNet-6 on that render.  Therefore an affine pose-to-xi conversion is
  not an admissible production assumption.  The quantized xi codes must be
  optimized directly against actual pooled PoseNet loss after G/Y1 decode.
- `xi_pose_coder.py` and `xi_spline_residual_coder.py` already provide the
  correct information carrier: six quantized values per pair, exact parse-back,
  and temporal delta/residual coding.  The existing measured real-n600
  trajectory needed roughly `2.7 KB` in the best lossless delta-residual mode.
  This signal belongs inside counted A, not in a parallel pose sidecar.
- `pose_carrier_arms_measured_20260708.md` measured the then-current generated
  store-nothing surface at `d_pose=1.995` on n8.  The older `0.0011` number is
  not transferable to the new ep725 PASS/G8 Y1 surface.  Every G branch needs
  its own direct quantized XIP2 solve and realized measurement.
- `ddm_p1_frame0_pose_quotient_carrier_receipt_20260725.json` falsifies one
  shared low-rank, parent-additive, quantized actuator formulation at n600:
  ranks 1--6 produced `d_pose=19.89..48.15` while its matched rank-6 control
  produced `20.32`.  This is a formulation-scoped negative, not a kill of
  nonlinear or mixture-of-chart residual carriers.
- The older amortized pose-carrier work reached approximately `d_pose=0.006`
  at about `22.5--23.8 KB` on n6/n12.  It does not beat the analytic XIP2
  hierarchy by itself, but it identifies the only defensible training role:
  learn a small shared residual chart after direct XIP2 has removed the
  representable six-dimensional component, never relearn the P/G surface or
  the full frame0.

This changes the codec engineering decision.  A is not an RGB correction list
with a single arbitrary row budget.  It is a nested conditional code whose
base mode, quantization level, trajectory entropy coder, and optional residual
chart are jointly priced with G and the exact archive.  The direct search
variable should be the integer XIP2 code table itself (plus coder and
quantization choice), because optimizing a float xi and quantizing afterward
can cross the nonlinear PoseNet and uint8 trust regions.  The receiver-decoded
integer table is the state carried between stages and across resume.

The first production-grade pose experiment should therefore compare, on every
retained G branch and its exact Y1:

1. PASS P0;
2. global `Y0 := Y1` protocol action;
3. XIP2 with each canonical interpretation, quantization grid, and lossless
   trajectory coder, optimized directly through the frozen scorer;
4. XIP2 plus a small shared residual chart only if the remaining pose score can
   pay for the chart's exact counted bytes.

All four are mutually exclusive A modes in one versioned archive envelope.
The hard-oracle frame and its Jacobian/VJP are acquisition evidence only.  A
trained residual is admitted only when its deletion control, matched-byte
analytic controls, receiver parse-back, Y1/Seg invariance, and whole-object
nonlinear score delta all pass.

This is also the concrete anti-rediscovery rule for pose: reopen the exact
artifacts above, but do not repeat their formulations.  The live unknown is
the quantized, temporally coded, post-G8 conditional control trajectory on the
new receiver surface.
