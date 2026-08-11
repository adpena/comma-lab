# ddm_hr1 — realization engineering for the C1 × PR135 hybrid

**Date:** 2026-08-11  
**Status:** DESIGN COMPLETE; `execution_allowed=false` until the fire gates below pass  
**Axis:** scorer-free precedent derivation; no new score axis  
**Score claim:** false  
**Pointer movement:** none

## Outcome

The realization stage remains necessary and is now specified as a full-n600, four-arm race. The
collapse falsifier does not fire: the retained PR135 semantic renderer disagrees with its own
semantic token plane at `d_seg = 0.00029639352578669786`, so its derived fidelity is only
**99.9703606474%**, below the charter's 99.99% collapse threshold. This is a derivation from the
retained renderer/token semantics and the measured F26 row, not a new scorer measurement.

The race must begin with a v14-style causal ladder on the terminal ps135 base. It must then compare
one common receiver across four treatments: frozen renderer, full renderer fine-tune, counted
low-rank adapters, and joint token-plus-renderer descent. Every learned treatment sees the frozen
scorers only after the camera-resolution uint8 cliff and the complete resize path. Every changed
weight, adapter, entropy model, token stream, and other video-derived value is counted in the
complete archive. Admission is by a retained whole candidate, never by a pre-round-trip field,
surrogate loss, component-only delta, or section-only byte count.

HY1's measured F26 calibration remains the planning prior: the exact C1 token plane is 114,717 B
under frozen F26 HPAC plus RC64, only 11 B above the shipped 114,706 B stream, with exact independent
decode and 100% grammar representability. At that proxy, sub-0.15 requires 82.8236457% of the C1
Seg gain. Using the retained F26 numbers, that corresponds to the derived diagnostic ceiling

```text
d_seg <= 0.0001767688985132100
```

for unchanged pose and the +11 B proxy. It is not a candidate gate after ps135 lands. The stage must
recompute the required `d_seg`, survival fraction, and full score from the terminal base, actual
candidate pose, and actual final archive bytes.

The effective frontier remains cp135 at **S = 0.16195513827824176 @ 186,252 B**
`[contest-CUDA T4, n600]`, archive SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`. The own-vehicle
frontier remains LC2 at **S = 0.16959899569230852 @ 187,226 B**
`[contest-CUDA T4, adjudicated, n600]`, archive SHA-256
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.

## Pinned inputs and fire boundary

The typed compiler must bind these objects by content, not only by path:

| object | binding |
|---|---|
| HY1 memo | commit `3a3825ad56`; `.omx/research/ddm_hy1_capstone_hybrid_20260811.md` |
| C1 solved tokens | 117,964,800 bytes; SHA-256 `2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5` |
| F26 calibration stream | 114,717 bytes; SHA-256 `9def0a4ba849757d473ba2a23cb0fd5370f2566355e5a5cfd398f847349636e8` |
| current joint-solve charter | `.omx/research/charters/ddm_js1_global_joint_solve_charter_20260810.md`, including Amendment 2 |
| terminal base | ps135's landed archive, renderer, carrier, coefficients, probability object, convergence receipt, and sensitivity map; exact hashes unknown until landing |
| scorer/runtime authority | exact frozen scorer weights, preprocessing source, receiver source, batch shape, thread/device configuration, and public evaluator hash captured at fire time |

The stage cannot fire from the F26 calibration alone. It waits for ps135 to land a terminal archive
and safe-run receipt, then reseals every base-dependent number. The live scorer lane remains owned by
ps135; ddm_hr1 used none of it.

## Precedent verdicts and what they change

### DW1 and QA75/KD-#74

DW1 is a negative precedent with a narrow scope, not a learned-prior family kill. On the TR1 E2
endpoint, its six-form mini-race selected attack-weighted Hinton KD (`T=2`, weight 100,
`attack_temp=1.0`). It initially moved toward the solve-frame teacher: the 12-epoch winner reached
`d_seg = 0.0050507`, and the long window transiently reached `0.004995`. The longer matched window
then reversed. The plain control ended at `0.0051147` with slope `-6.80e-6/epoch`; the distillation
arm was refused after 29 epochs at `0.0054967` with slope `+1.37e-5/epoch`. Its deficit versus the
control was `3.82e-4`, 12.8 times the measured `2.99e-5` noise floor. Head-range relaxation did not
rescue it.

The honest verdict is FORMULATION-scoped: solve-field KD, margin-field distillation, and argmax
distillation are closed as finishing-stage levers on that converged TR1 endpoint. The early transient
also proves that a short mini-race can select the wrong objective. QA74's earlier n8 result remains a
directional existence proof only; RV1 correctly reopened it as promising rather than a capacity wall,
but it is not a full-population decision row.

Consequences for HR1:

- The PR135 initialization is the learned realization prior; pure KD is not the primary objective.
- Full-population CE to the C1 plane followed by a one-sided scorer-margin stage is the default.
- If a teacher term is ever reintroduced, it is an explicitly separate from-birth treatment with
  attack weighting, a long-window reversal guard, and its own preregistered case. It is not smuggled
  into the four required arms.
- No TR1 `d_seg` magnitude transfers to PR135. Only the failure mechanism and guard design transfer.

### V14 realization ladder

V14 localized an older realization failure by separating exact semantic geometry from paint,
camera placement, amplitude/uint8, resize, and final argmax. Its exact target grid reached
`d_seg = 0.000282948812`, while the repaired realized output remained about `0.0274`; the G4
horizon realized only 5.2867% of the forecast cell gain. The binding lesson is that semantic
representability does not imply learned paint survival. HR1 therefore runs the causal ladder before
changing weights.

### JD-line joint descent

JD1/JD3/JD4 supply the mechanics, not transferable endpoint numbers: pose computation must be
explicitly armed; the Seg floor is latched in realized space; pose engagement is terminal; every
breach rolls back; optimizer, EMA, RNG, gate, and accepted-candidate state survive resume. JD1's
loss-space hold allowed live `d_seg` to worsen from about 0.00357 to 0.00599 while pose improved,
which closes loss-only holds. JD3's stage-scoped EMA and realized rollback are adopted. The active
EMA must be re-anchored whenever the event-run geometry changes; carrying a smoke-window decay into
a longer continuation is refused.

### PR135 renderer and HPAC

The borrowed PR135/F26 renderer is width 96 with token and 600-frame embeddings, coordinate mixing,
four depthwise/pointwise/GroupNorm/FiLM residual blocks, and a three-channel sigmoid head. The source
training family used CE, softplus margin, expected-flip polishing, quantization-aware rendering, and
the real token plane. The semantic token plane is the GT SegNet argmax cache, so its measured
renderer `d_seg` is exactly renderer-argmax versus token fidelity. That supplies the 99.9703606474%
collapse check above.

HB1 did not have a trained HPAC row on our labels. HB2 later repaired the self-compress deploy-bound
mismatch and closed tq1c stage 3/4: 14,116 B packed model plus 97,928 B exact token stream, 112,044 B
total, with `max_logit_diff=0.0` and exact n600 decode. HY1 then measured the directly relevant C1
plane under frozen F26 at 114,717 B. HR1 reuses the HPAC training, self-compress, packing, checkpoint,
RC64, and independent-decode machinery; none of those byte counts may be transferred to the terminal
ps135 probability object without rebuilding it.

## Binding receiver and round-trip contract

For every learned arm, the forward graph is:

```text
hard semantic tokens
  -> learned renderer at 384x512 float RGB
  -> bicubic resize to 874x1164 camera RGB
  -> clamp + round-to-uint8 in the forward pass, STE in the backward pass
  -> bilinear scorer resize to 384x512
  -> frozen SegNet preprocessing and logits
  -> differentiable RGB-to-YUV6 and frozen PoseNet wherever pose is armed
```

The saved public receiver must perform the same forward operation without STE. A candidate is
invalid unless the training hard-forward camera bytes equal the public receiver's camera bytes on
real frames, and the receiver's ordered argmax vector equals the CPU-Torch reference under the
pinned batch shape.

There is a P0 implementation gap to close before fire. The current
`tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` performs bicubic up,
bilinear down, then `Uint8STE`. The retained PR130 public renderer performs its resize to camera
resolution, rounds to uint8, and only then lets the frozen scorer downsample. PR130 also currently
uses bilinear, not bicubic, for the semantic camera lift. The operator's binding HR1 order is
bicubic-up, camera-uint8, bilinear-down. Therefore the existing helper cannot be called exact for
this stage without a typed camera-uint8 mode and a real-frame positive control against the candidate
runtime. The four arms share the corrected bicubic receiver; the unchanged shipping receiver is
retained as the incumbent comparator so the zero-counted interpolation change cannot hide inside a
training claim.

Three placements are independently enforced:

1. **R in loop:** both resize operations are inside the autograd graph, with camera quantization
   between them.
2. **uint8 in loop:** the forward is exact clamp/round and the backward uses the registered STE;
   float-only candidates never reach an admission gate.
3. **YUV6 in loop:** when pose is armed, `differentiable_rgb_to_yuv6` feeds the real PoseNet graph;
   no `no_grad` or in-place clamp can sever the renderer gradient.

The #855/#903 controls are binding. Parity uses retained real frames, never random tensors. It
records forward max error, the ordered per-pair argmax SHA vector, and batch shape. Training ports
also record per-tensor gradient max-relative error against CPU Torch. A matching scalar loss is not
accepted: #903 measured identical loss while 40 of 41 arrays diverged through the upsample VJP and
Adam sign amplification. MLX GPU may be used only with the atomics-free deterministic R path or a
new equally strong receipt; default MLX resize backward is refused.

## Stage 0 — causal realization ladder

Stage 0 runs once on the terminal base and once on the frozen C1 head before any optimizer step.
Every scorer call is full n600 in chunks of at most 120, after the sole scorer lane is claimed.
Intermediate RGB fields and all final scorer outputs are retained.

| rung | cumulative surface | retained diagnostic | purpose |
|---|---|---|---|
| V0 | terminal shipping archive through its unchanged receiver | camera RGB, scorer RGB, Seg argmax/logits, Pose6, components, archive bytes | same-axis base |
| V1 | C1 hard token plane, grammar/HPAC parse-back only | exact token equality, coder state, probability object, stream and archive repeats | reprove semantic and wire closure on the terminal object |
| V2 | frozen renderer's native 384x512 float paint | native RGB plus Seg argmax/logits on the native diagnostic surface | isolate token-to-paint loss; diagnostic only, never a candidate score |
| V3 | V2 plus bicubic camera lift, still float | camera float, range/clip statistics, native-versus-camera deltas | isolate interpolation and camera placement |
| V4 | V3 plus exact camera clamp/round | exact camera uint8 bytes, rounding/clipping histogram, byte equality with public runtime | isolate amplitude/uint8 loss |
| V5 | V4 plus scorer bilinear down and preprocessing | final Seg/Pose outputs, per-pair/per-edge errors, C1-event survival map | exact direct-realization row |

The ladder reports both token-plane mismatch and source-GT mismatch. Only V0 and V5 are complete
candidate surfaces. V2-V4 are causal diagnostics and must not be described as official `d_seg` or a
candidate score.

Let `d_base` be V0's measured `d_seg`, `d_target` the scorer-hash-compatible C1 target value, and
`d_cand` V5's value. The diagnostic survival is

```text
rho = (d_base - d_cand) / (d_base - d_target).
```

If `d_base <= d_target`, the denominator is non-positive and the stage refuses the survival metric;
only complete-score comparison remains. Otherwise the stage recomputes the sub-0.15 requirement from
the actual V5 pose and archive bytes. No 82.8236457% literal enters the launch config.

## Exact training and admission objective

For target class `y` from the hard C1 plane and frozen SegNet logits `z` after the full round trip,
define

```text
m(z,y) = z[y] - max(z[c] for c != y)
L_CE    = mean(cross_entropy(z, y))
L_margin = mean(relu(m_C1 - m(z,y)))
```

`m_C1` is the nonnegative target margin from the exact C1 solve-frame teacher under the same scorer
hash; it is not a guessed constant. Missing or hash-incompatible teacher logits block the margin
stage rather than falling back silently. Sampling may oversample the retained 27,351
C1-versus-shipped sites, but inverse-probability weights must keep both losses unbiased over all
117,964,800 pixels. Focusing may change variance, not the objective.

The event sequence uses CE to enter the correct class cells, then one-sided margin to build
camera-uint8 robustness. This consumes the #63 lesson without importing a PR95 stage skeleton.
Attack-weighted KD is absent by default because DW1's full-window result reversed. The joint-pose
event adds the exact differentiable component

```text
L_pose = sqrt(10 * mean((Pose6(candidate) - Pose6(source))**2)).
```

The rate surrogate for joint token moves is the exact HPAC/RC64 symbol NLL priced at
`25/(8*37,545,489)` score units per bit. It proposes moves only. At every admission event the hard
source of truth is

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*len(final_archive_zip)/37,545,489,
```

recomputed from the exact receiver output and final archive bytes. Model-blob changes, ZIP effects,
and coder framing are never inferred from NLL.

## The full four-arm race

All arms start from the same terminal ps135 object, use the same C1 plane, corrected bicubic
camera-uint8 receiver, scorer hashes, pair order, seed, event graph, full-population verdicts, and
archive compiler. The unchanged shipping base V0 remains an external comparator. Each arm retains
its own probability objects, streams, weights, checkpoints, camera RGB, scorer outputs, and archive
repeats.

### Arm A — frozen decode

- **Trainable state:** none. No optimizer or EMA is fabricated.
- **Action:** replace the semantic plane and rebuild the terminal probability object/archive while
  freezing renderer, carrier, coefficients, and learned priors.
- **Purpose:** direct-realization baseline and v14 localization.
- **Admission:** exact parse-back, deterministic double decode, V5 whole score, and freshly derived
  survival threshold. If it reaches sub-0.15, Arms B-D fold without firing.

### Arm B — full renderer fine-tune

- **Trainable state:** all PR135 semantic renderer tensors, initialized from the terminal W4 state;
  tokens fixed to C1 during the Seg events. Carrier/pose state remains frozen until the terminal
  joint-pose event.
- **Objective:** CE event, then target-margin event, always after camera-uint8 R. No finishing KD.
- **Optimizer:** source-lineage AdamW semantics and parameter grouping are imported from the exact
  terminal training receipt. The initial learning rate is calibrated by a deterministic
  current-vehicle update-norm line search; the old `2e-7` tail value is an ancestor anchor, not an
  automatic value. Gradient clipping is inherited only if its exact source receipt and consumer are
  bound.
- **Bytes:** the complete requantized renderer/model blob is packed and priced at every admission
  event. Changed weights are learned video-derived content and counted.

### Arm C — counted low-rank adapter

- **Trainable state:** residual low-rank updates on the four FiLM projections and RGB head. The
  borrowed renderer is frozen. Factors are initialized to make the first hard forward byte-identical
  to Arm A.
- **Rank selection:** begins at the smallest representable rank and grows only at event boundaries
  when the measured same-object score marginal beats the exact rate price. A safety cap is not a
  rank-selection rule. Each rank is a preserved stage checkpoint and complete archive.
- **Objective/optimizer:** the same CE-to-margin graph and source-lineage AdamW semantics as Arm B.
- **Bytes:** factors, scales, schema, and any changed quantizer values are counted. Nothing learned
  is embedded into free runtime code.

### Arm D — joint token plus renderer descent

- **Trainable state:** hard token proposals plus Arm B renderer state; the terminal event may also
  use js1's joined `int12 × basis × FiLM` pose/realization coordinates. The public forward always
  consumes hard grammar-valid tokens. A straight-through categorical relaxation exists only in the
  backward proposal path.
- **Objective:** exact CE/margin plus the HPAC NLL rate proposal. At the pose event, use the complete
  Seg/Pose objective and jointly compensate carrier state. Every hard event re-encodes the tokens and
  rebuilds the full archive.
- **Optimizer:** AdamW for renderer groups; mirror/categorical descent for token logits; distinct
  state and step counters, both checkpointed. No dense full-n600 five-logit tensor is kept in memory;
  token relaxations are pair-chunk local and the durable state is the hard token plane plus sparse
  proposal state.
- **Guard:** a joint token may depart from C1 only when the complete receiver-realized candidate
  improves the exact objective. C1 is a teacher/initialization, not an untouchable label oracle.

## Seg hold, pose guard, and candidate selection

The following hard rules apply to Arms B-D:

1. The best accepted full-n600 realized `d_seg` is a ratcheting floor. A later event may not increase
   it on the same scorer/device/batch instrument. No smooth-loss allowance substitutes for this
   check.
2. Before pose engagement, every candidate must also improve the complete
   `100*d_seg + sqrt(10*d_pose)` distortion term versus its accepted parent. This prevents a semantic
   win from silently spending more pose than it earns.
3. Pose can engage only after the arm has reached the freshly recomputed realization requirement.
   The pre-pose checkpoint and Seg floor are latched. Any Seg breach or full-score regression rolls
   back to that checkpoint.
4. Archive bytes are included at every hard admission. A smaller semantic section cannot offset an
   unmeasured model-blob, carrier, or ZIP interaction.
5. Select among live weights and EMA shadow only by paired complete byte-closed n600 rows. The shadow
   is not automatically authoritative.

## Optimizer, EMA, event schedule, and verdict cadence

The schedule uses `DDMEventContinuationV1`: causal events and accepted receiver states move the
stage; update counts and wall time are safety caps only.

| event | entry condition | exit condition | checkpoint |
|---|---|---|---|
| `receiver_bound` | terminal hashes, corrected receiver, scorer hashes, charged/free audit, storage and lane gates pass | V0/V1 parse-back and deterministic repeat pass | `receiver_bound` |
| `v14_ladder_complete` | Stage 0 owns scorer lane | V0-V5 receipts complete with failure localization | every ladder rung |
| `ce_cell_entry` | arm hard-forward equals Arm A at initialization | realized hard-error slope stops improving above measured gate noise, or the arm reaches the survival requirement | periodic plus exit |
| `margin_robustness` | CE event complete and C1 teacher-margin hash matches | realized `d_seg`/survival stops improving for the registered slope watcher, or target is met | periodic plus best/exit |
| `rate_hardening` | a candidate passes Seg/Pose holds | exact pack/encode/decode and complete archive improve S | every complete candidate |
| `pose_terminal` | realization requirement met and ps135 pose target/state bound | exact distortion and full S stop improving without Seg-floor breach | pre-engage, every accepted step, exit |
| `n600_admitted` | all prior gates pass | exact final tuple, repeat archive, double inflate, and manifests exist | terminal |

The convergence detector uses the registered slope/noise machinery; a rising loss cannot satisfy a
knee predicate. A cap emits `safety_bound_REPORTED`, never convergence. Each hard verdict covers all
600 pairs in deterministic chunks of at most 120 and flushes per-pair rows before the next chunk.
Full n600 verdicts occur at every event boundary, every best-candidate replacement, and before/after
pose engagement. Cheaper batch-local signals may guide proposals but never bank a result.

Every trained arm uses `EmaDecayCalibrated` with LawRef `ema_decay_run_geometry_v1` after the actual
optimizer-update horizon `U` is known. The current DSL default `target_seed_fraction=0.01` is an
explicit inherited policy input; the decay is then `0.01**(1/U)`. It is never a literal 0.997, never
clamped without a recorded override, and is re-derived/re-anchored if event geometry changes. If the
stage instead supplies a warmup fraction, it must record the scientific source of that fraction;
the LawRef derives the arithmetic, not the input policy.

Optimizer moments are continuation state. Same-geometry resume restores them byte-for-byte; it does
not cold-start Adam. Geometry-changing resume must either transform/re-warm state under the registered
beta2 law or refuse. #903 additionally requires deterministic R-gradient parity before any first
Adam step is trusted.

## Memory and storage preflight at the real configuration

The closest current-vehicle receipt is M1's real semantic-renderer n120 memory probe at
microbatch 4, verdict batch 32: measured RSS 2.021515 GiB plus MLX reported peak 8.493787 GiB,
10.515302 GiB combined; its sealed projection was 16.0 GiB after a 1.5 safety factor. That probe did
not include HR1's corrected camera-uint8 placement, full joint token state, or terminal pose event,
so it is an ancestor lower bound, not launch clearance. JD1's joint-pose batch-4 projection was
84.95 GiB, while batch 8 was refused at 108.07 GiB; those are a conservative ancestor envelope for
the terminal joint event, not PR135 measurements.

| arm/event | provisional reservation | basis | fire rule |
|---|---:|---|---|
| Arm A full decode + n600 scorer | 24 GiB | existing PR135 scorer-wrapper reservation | fresh full-config load/decode probe required |
| Arm B Seg training | 16 GiB lower bound | M1 real n120 renderer receipt | fresh camera-uint8/R/scorer-in-loop microbatch-4 probe; project `ceil(1.5×max(RSS+device peak))` |
| Arm C Seg training | 16 GiB lower bound | same scorer-dominated ancestor | independent adapter-config probe; no inheritance merely because it has fewer trainable weights |
| Arm D joint token/renderer | 84.95 GiB conservative envelope | JD1 batch-4 joint-pose projection | fresh pair-chunk-local token + renderer + scorer probe; batch 8 starts REFUSED |
| any terminal pose event | 84.95 GiB conservative envelope | JD1 batch-4 | fresh complete pose-gradient probe and governor clearance |

The governed launcher consumes the measured peak and active fleet growth, not these prose values.
Any missing, stale, config-mismatched, or lower-scope receipt is a hard refusal. Storage preflight
reserves the complete retained tree before launch. Every token plane, probability object, coded
stream, model blob, adapter blob, checkpoint, archive, decoded camera field, scorer output, and
determinism repeat is retained under the SSD consumer root. No scalar-only measurement is permitted.

## Resumability and payload custody

Every event and periodic checkpoint is written atomically and preserved under a distinct
stage/event/step name. A checkpoint is byte-close-loadable only if it contains:

- live renderer, adapter, token proposal, HPAC, carrier, basis, FiLM, and quantizer state applicable
  to the arm;
- EMA shadow plus its LawRef inputs, resolved decay, update count, warmup state, and reanchor events;
- every optimizer/mirror-descent moment and group step counter;
- Python, NumPy, Torch/MLX/device RNG states and the single root seed;
- event-node identity, slope/noise history, Seg floor, pose gate, rollback parent, best-candidate
  identity, and accepted/rejected hard-oracle chain;
- typed config, final argv, source/runtime/scorer/input hashes, batch shape, threads, device, and
  charged/free manifest;
- hard token plane, probability/coder state, exact archive compiler state, and hashes/paths of every
  retained payload.

A forced resume from every event boundary must reproduce the next hard archive and scorer receipt.
The maximum recovery loss is one declared intra-event interval. Success-only scratch may be cleaned
only after a machine-readable certify-or-block record; material payloads remain.

## DSL-compiled design stubs

These are schema obligations, not claimed implementations. Until each has a real consumer, positive
control, default-off byte-identity test, resume-registry entry, and expected-active manifest,
`execution_allowed=false`.

| typed lever/program | required consumer effect |
|---|---|
| `Hr1CameraUint8RoundTrip` | selects bicubic camera lift, camera-resolution STE, scorer bilinear down, differentiable YUV6, and deterministic backward |
| `Hr1V14RealizationLadder` | emits V0-V5 retained surfaces and refuses candidate claims on V2-V4 |
| `Hr1FrozenDecodeProgram` | compiles Arm A and proves it has no trainable state/EMA |
| `Hr1FullRendererFinetuneProgram` | owns the complete renderer parameter set and source-lineage optimizer |
| `Hr1LowRankAdapterProgram` | owns adapter placement, rank-event state, counted factor schema, and exact baseline initialization |
| `Hr1JointTokenRendererProgram` | owns hard/relaxed token state, renderer groups, HPAC rate proposal, and joint carrier/pose terminal event |
| `Hr1SegPoseHold` | consumes full-n600 realized components and enforces ratcheting Seg plus complete-distortion guards |
| `Hr1CoordinatePriorEqualParameterAB` | compiles the conditional fallback treatment/control with exact parameter and schema equality |
| `EmaDecayCalibrated` | resolves `ema_decay_run_geometry_v1` from actual `U`; no literal fallback |
| `DDMEventContinuationV1` | persists event graph; budgets remain safety caps |

Raw flags may not be appended to make any stub appear active. The final compiler attaches custody
last, then proves typed-config-to-argv parity and single ownership of every emitted field.

## Second learned-prior reading — conditional coordinate entropy

HY1 direction C is conditional and fires only if Stage 0 shows the failed C1 events are localized
enough to define a stable dense-token/level-set split. The treatment is a learned causal entropy
prior over quantized generator/level-set coordinates, conditioned on state the decoder already has:
previous coordinates, decoded semantic groups, decoder-derived edge topology, and already-carried
pose/carrier state. It may also use that state to allocate fixed model capacity. No edge mask or
contour stream is transmitted.

SR1 closes the post-hoc rate interpretation at FORMULATION scope: additive causal-edge calibration
saved only 2 charged bytes on the 114,706 B F26 stream, and scalar pose conditioning cost 43 B. The
second prior is admissible only if conditioning participates in the representation or capacity
allocation; another probability-calibration table is folded.

The matched control has exactly the same parameter tensors, quantization, model schema, coordinate
alphabet, symbol order, optimizer steps, seed, and counted model bytes. Its conditioning projection
receives a fixed zero/null input while retaining all parameters; the treatment receives the
decoder-derived state. Both use the HB1/HB2 HPAC stage machinery, including deploy-bound-safe
self-compress, stage-3 pack, stage-4 encode/decode, terminal empty coder state, independent exact
decode, and deterministic repeat. Both payloads and models are retained. Selection uses complete
archive bytes at equal receiver-realized Seg/Pose, not NLL alone. If the treatment cannot beat the
equal-parameter control as a complete object, the conditional-prior formulation closes without
reopening SR1's additive table family.

## Pre-staging assessment

The following work is safe before ps135 terminates because it is scorer-free, payload-light, and can
run under ordinary process limits. The RSS values are conservative design projections, not measured
HR1 peaks; implementation must record actual safe-run RSS and refuse a cap breach.

| item | disposition | projected peak RSS | boundary |
|---|---|---:|---|
| typed schemas/program factories plus compile/no-consumer tests | SAFE-TO-PREPARE | 0.5 GiB | no model/scorer import; no raw flags |
| camera-uint8 round-trip pixel positive control on retained real RGB, with every output retained | SAFE-TO-PREPARE | 1.0 GiB | pixel equality only; no SegNet/PoseNet conclusion |
| checkpoint/resume/payload-manifest schemas and atomic-write tests | SAFE-TO-PREPARE | 0.5 GiB | synthetic state tests are apparatus tests, never mechanism evidence |
| content-hash/path binder for HY1, terminal-base placeholders, and HPAC sources | SAFE-TO-PREPARE | 0.5 GiB | streaming reads; public intake remains read-only |
| shape-only per-arm memory configuration compiler | SAFE-TO-PREPARE | 0.5 GiB | emits REFUSE until a fresh real-config memory probe exists |

The following work must wait: any SegNet/PoseNet forward, V0-V5 n600 ladder, full terminal archive
decode/rebuild, real memory probe that loads the learned renderer/scorers, HPAC retraining, any of
Arms B-D, or any Modal/contest dispatch. They are nontrivial beside the live solve and require the
ps135 terminal hashes, a freshly claimed scorer lane where applicable, storage clearance, and the
governed launcher. This arm deliberately did not pre-stage them by running smaller toy jobs.

## Borrowed and original work

Borrowed: PR135/F26 token grammar, HPAC/RC64, renderer architecture and weights, carrier and pose
state, public receiver, source training recipe, and ps135/cp135 composition. HR1 claims no originality
for those objects or their measured rows.

Ours-original in this unit: the PR135-specific realization race, the v14 causal ladder placement,
the camera-uint8 ordering audit, full-score Seg/Pose hold contract, event/EMA/resume integration,
adapter and joint-token treatment definitions, equal-parameter conditional-coordinate prior control,
pre-stage refusal boundary, and the typed consumer/fire specification. The C1 plane, HY1 wire probes,
and js1 solver arsenal are earlier Pact work and are cited rather than reclaimed.

## RECALL EVIDENCE

The recall searched beyond the charter seeds before choosing the objective, receiver, memory plan,
or negative scopes. Commands/queries included:

```text
.venv/bin/python tools/list_canonical_equations.py --json

rg -n -i 'ema|event|margin|distill|round.trip|realiz|seg.rate|pose|score' <equation listing>

rg -n -i 'implicit conditioning|learned entropy prior|equal.param|HPAC|self.compress|renderer|token fidelity|realization ladder|camera uint8|upsample.VJP|76 argmax' \
  .omx/research .omx/state src/tac tools experiments

rg -n '#63|#686|#855|#903|event.driven|margin_hinge|kl_t2' \
  .omx/research .omx/state src/tac

rg -n -i 'realization|PR135|HPAC|round.trip|EMA|joint descent' \
  .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_* \
  .omx/state/canonical_task_status.jsonl .omx/research/harness_tasklist_bridge_20260803.jsonl
```

Directly read sources included the governing files/live board; HY1; DW1; RV1 and the QA74/75
lineage; v14; JD1/JD3/JD4 and TP1; fd135/eh1/pi135 plus the PR130 full-stack intake; the retained
PR130 `semantic_renderer_oracle.py`, `train_semantic_full.py`, and `inflate.py`; HB1/HB2; SR1
implicit-edge conditioning; js1 Amendment 2; `tac.differentiable_eval_roundtrip`; the event
continuation engine; the EMA law/evaluator/DSL; the #63 margin-hinge receipts; and the #855/#903
parity/determinism receipts.

Findings beyond the named seed list changed the plan in five ways:

1. The current helper's uint8 placement differs from the public camera-byte path, so a typed
   camera-uint8 mode and real-frame positive control became a P0 fire gate.
2. The retained PR135 fidelity is 99.9703606474%, so the four-arm race cannot honestly collapse.
3. #855/#903 require ordered real-frame argmax and gradient parity plus deterministic R; loss-scalar
   parity is insufficient.
4. SR1 already closed additive edge/pose probability calibration, so the second learned prior is
   representation-level and equal-parameter, not another table.
5. M1 supplies a 16 GiB ancestor projection while JD1 supplies an 84.95 GiB joint-pose envelope;
   neither is current-config clearance, so every arm now fails closed on a fresh memory probe.

The bounded search did not find a retained full-n600 PR135 C1 renderer-adaptation row or an exact
terminal-base renderer-versus-C1 receipt. Those are Stage 0 obligations, not presumed absences in all
possible stores.

## Measured and unmeasured boundaries

- Measured by prior receipts and used here: HY1 C1 event count/rate/decode/representability; PR135
  F26 renderer `d_seg`; DW1 trajectories/noise/bytes; HB2 exact model/token pack; SR1 conditional
  coder rows; M1/JD1 memory anchors.
- Derived here: 99.9703606474% renderer fidelity, the planning `d_seg` ceiling
  `0.0001767688985132100`, and the race/config consequences.
- Not measured here: direct C1 survival through the terminal PR135 receiver, any new `d_seg`,
  `d_pose`, archive bytes, full score, CPU/CUDA gap, adapter effect, fine-tune effect, joint-descent
  effect, coordinate-prior effect, or new memory peak.
- No scorer, trainer, Modal, MPS, CUDA, exact evaluator, HPAC encode, archive build, or payload-
  materializing run was launched. Public intake clones and `upstream/` remained read-only.
- This design did not move the effective or own-vehicle frontier. It is means, not the sub-0.15 end.

## Landing custody

The required serializer was invoked with the post-edit SHA, `base-content-sha256=new`, both commit
message tags, and `--no-co-author`. It failed before staging because the managed worktree could not
write a Git object: `unable to create temporary file: Operation not permitted` followed by
`failed to insert into database`. The shared index remained empty and this memo remains an
uncommitted worktree artifact. No direct `git commit`, alternate index, stash, or hidden clone was
used to bypass the repository custody boundary.

## NEXT_IF_RESUMED

- `hr1_spec_landing` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: MAIN repository custodian; consumer store: the Pact `main` Git history and js1 Amendment-2 intake; fire trigger: a session with Git-object write permission verifies this exact memo's SHA and reruns `tools/subagent_commit_serializer.py` with `base-content-sha256=new`, the post-edit SHA, `[no-triality] [p0-ledger-ok]`, and `--no-co-author`.
- `hr1_roundtrip_and_dsl_preflight` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: js1 realization-apparatus successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/`; fire trigger: this spec is landed and the work remains scorer-free; build the typed camera-uint8 receiver mode, real-frame pixel positive control, four arm programs, resume schemas, and fail-closed memory stubs without loading scorers.
- `hr1_stage0_direct_realization` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: js1/#995 scorer successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/stage0_v14/`; fire trigger: ps135 terminal archive/safe-run receipt exists, the whole-container C1 probability object parses and independently decodes, storage/governor pass, and the sole n600 scorer lane is freshly claimed.
- `hr1_four_arm_realization_race` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: js1 realization successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/joint_realization/`; fire trigger: Stage 0 direct realization misses the freshly recomputed sub-0.15 requirement, all four typed consumers and real-config memory receipts pass, and no heavier live job conflicts.
- `hr1_conditional_coordinate_prior` — disposition: **CONDITIONAL-QUEUED**; owner: HY1 conditional-representation successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/conditional_levelset/`; fire trigger: retained Stage 0/arm failures localize enough to define a deterministic dense-token/level-set split and an equal-parameter coordinate-prior control; otherwise fold it.
- `hr1_exact_promotion` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: MAIN only; consumer store: MAIN exact-evaluation and canonical-pointer stores; fire trigger: one deterministic complete n600 candidate strictly improves the local full score, retains every payload and repeat, passes compliance/custody, and has separate contest-CPU/CUDA execution clearance.

## LIVE-HYPOTHESES

- Full PR135 renderer adaptation may retain at least the required C1 gain because all token changes
  are grammar-valid and the starting renderer is already 99.9704% faithful, while the remaining
  errors are exactly the cells the full round-trip objective targets.
- A counted FiLM/head adapter may dominate full fine-tuning in score per byte because the F26-to-F26
  semantic movement was concentrated in FiLM state, but only a complete packed archive can decide.
- Joint token-plus-renderer descent may beat fixed-C1 fine-tuning because it can choose a nearby hard
  token preimage that the frozen renderer realizes more reliably while preserving the source
  SegNet cells.
- Camera-bicubic training may improve boundary survival relative to the shipping bilinear lift, but
  it is a zero-counted receiver treatment whose pose and Seg effects remain unmeasured.
- A representation-level edge-conditioned coordinate prior may help the localized fallback even
  though additive edge calibration is dead, because it can change capacity allocation rather than
  merely retune an already-saturated probability lattice.

## DEAD-ENDS

- Finishing-stage solve-frame KD/margin/argmax distillation on the converged TR1 endpoint is closed
  at FORMULATION scope: DW1's early gain reversed and ended 12.8 noise floors worse than control.
- Treating a short distillation mini-race as a verdict is closed: DW1's 12-epoch winner was the arm
  that later reversed.
- Collapsing HR1 to baseline verification is closed by the retained 99.9703606474% fidelity, below
  the explicit 99.99% threshold.
- Calling the current round-trip helper exact for HR1 without changing the uint8 placement is
  closed: its default rounds after downsampling, unlike the public camera-byte path and the binding
  stage order.
- Pre-round-trip, float-only, loss-only, or scalar-parity optimization is closed by v14, JD1,
  #855, and #903; only receiver-realized hard candidates can be admitted.
- A literal EMA decay such as 0.997 is closed; the decay must resolve from the actual event-run
  update geometry and re-anchor when that geometry changes.
- Standalone additive causal-edge or scalar-pose probability calibration is closed at FORMULATION
  scope by SR1's -2 B and +43 B rows; another table is not the second learned prior.
- Explicit transmitted edge masks/contours are closed on this vehicle; the conditional route may
  use only decoder-derived state unless a new complete-object representation earns its bytes.
- Generic dense Brotli is closed at INSTANCE scope for the C1 token plane: HY1 measured 429,383 B
  versus 114,717 B for frozen F26 HPAC plus RC64.
