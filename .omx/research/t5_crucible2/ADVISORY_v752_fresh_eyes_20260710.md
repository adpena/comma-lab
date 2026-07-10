# ADVISORY — v7.5.2 fresh-eyes launch audit — 2026-07-10

`research_only=true`

STORES CONSULTED: .omx/research (SPEC_v75_optimal_single_trunk_20260708 · t5_crucible2/SPEC_v752_20260709 · t5_crucible2/SYNTHESIS_v3_v752_20260709 · DUAL_CHAIN_BRIEF_385_20260710 · r1_dxi_shippability_byteclose_20260708 · ADVISORY_evaluator_video_geometry_20260710) · DSL (src/tac/witness_dsl/curriculum_dsl.py) · run-artifacts (experiments/results/owed16v2_rebalanced_ON_20260710T114759Z/safe_run.out). Not consulted: memories index, canonical equations, task list.

**Disposition:** **HOLD the full v7.5.2 launch.** The current pilot is also not a faithful
test of the sealed v7.5.2 program unless the config/spec differences below are explicitly
accepted, relabeled, and protected by an exact compiled-argv contract. This is an advisory
means artifact. It does not move the canonical frontier pointer and it did not authorize,
start, stop, or mutate any run.

**Authority labels:** source/code claims are `SOURCE-VERIFIED`; local MLX/CPU observations are
`[macOS advisory] NON-PROMOTABLE`; the only score authority remains exact contest CPU/CUDA replay
on the exact archive bytes.

## Answer first

The v7.5.2 idea remains coherent: one geometric trunk, a formed-partition curriculum, late pose
conditioning, and explicit fallback custody are the right shape. The instantiated launch object is
not yet that vehicle. Five launch-contract gaps are load-bearing:

1. the repaired conditioning gate exists but is not composed;
2. epoch 726 bypasses the gate even when the detector says to ship the banked arm;
3. “banked R1” is a coupled complete checkpoint/archive, not a proven dxi component that can be
   grafted onto a fresh v7.5.2 render;
4. chroma is inherited even though the sealed launch keeps it for a later attribution rung; and
5. the launch-blocking amber realization remains unresolved.

The current green tests and dry-start checks do not detect these semantic mismatches. They prove
parser/boot/checkpoint mechanics, not that the intended vehicle was instantiated.

## P0 findings

### P0-1 — #383 is built but absent from the launcher-facing program

`PoseFinishConditioningGate` emits `--pose-finish-engage-on sigma_min_plateau` in
`src/tac/witness_dsl/curriculum_dsl.py:1662-1689`. The exact launcher-facing v7.5.2 object is built
by `src/tac/witness_autoconfig.py:3049-3094` and selected by
`tools/launch_witness_run.py:728-736`. Its compiled argv contains
`--pose-finish-start-epoch 726` but no `--pose-finish-engage-on`; its lever tuple omits
`pose_finish_conditioning_gate`. The trainer therefore takes the parser default `muon` at
`experiments/train_levelset_witness_realized_through_R_mlx.py:10134-10147`.

The regression test is stale in the dangerous direction: it explicitly requires the flag to be
absent at `src/tac/tests/test_crucible2_v752_dsl_wirein.py:105-120`. The focused config/resume suite
passes while preserving the wrong active-lever set.

**Required gate:** one expected-active-lever manifest derived from the sealed program, checked at
compile time, dry-start startup, resume, and exporter entry. A lever being implemented elsewhere in
the tree must never satisfy this gate.

### P0-2 — the epoch backstop contradicts the banked-fallback contract

The sealed fallback says a never-fired, degenerate, or canary-failed conditioning detector leaves
pose finish disengaged and selects the banked R1 artifact; a regressing finish rolls back to the
pre-finish artifact (`SPEC_v752_20260709.md:183-187,236-240`). Runtime instead computes:

```text
sigma mode: pose_finish_on = condition_fired OR epoch >= pose_finish_start
muon mode : pose_finish_on = muon_fired      OR epoch >= pose_finish_start
```

at `experiments/train_levelset_witness_realized_through_R_mlx.py:8720-8737`. At epoch 726, joint
pose descent therefore starts even if the detector has classified the signal as degenerate. The
end-of-run disengaged alarm at `:8750-8762` is unreachable in a 3000-epoch run after the backstop
has forced pose on.

This is no longer hypothetical. The live, preserved owed16v2 arm reported
`DEGENERATE_GUARD_TRIPPED`, `should_ship_banked_r1=true`, and `actuated=false` at epochs 669 and 673
in `experiments/results/owed16v2_rebalanced_ON_20260710T114759Z/safe_run.out`. No selector consumed
that decision. The process and its checkpoints were left untouched.

**Required gate:** replace the boolean/epoch OR with a typed terminal state machine:

```text
ARMED -> ENGAGED -> ACCEPTED
   |         |-> REGRESSED_ROLLBACK
   |-> BANKED_COMPLETE_ARTIFACT
```

The epoch cap may force a decision, but it may not silently reinterpret “banked fallback” as “start
joint descent anyway.”

### P0-3 — R1 pose is shippable, but not proven composable with a v7.5.2 Seg trunk

The #238 result is real and should not be reopened: on the R1 checkpoint, the counted
`xi_stored + dxi` payload survives the byte-closed receiver and measures d_pose 0.001610 at n600.
Its own custody record is explicit that the twist comes from the **same resolved checkpoint** as the
INR weights (`.omx/research/r1_dxi_shippability_byteclose_20260708.md:22-26`). On that co-adapted R1
render, `xi_stored` already gives d_pose 0.02197 and dxi refines it to 0.001127 at n24 (`:44-61`).

The byte-close connector likewise reads `pose_carrier.xi_stored` and `pose_carrier.dxi` from the
selected checkpoint at `tools/levelset_byte_close_and_eval.py:2962-2993`; it does not import an
external R1 render/twist bank into a different checkpoint. A fresh carrier initializes dxi to zero,
and no current v7.5.2 trainer/launcher flag performs an R1 f0+xi graft.

Therefore the settled claim is:

- **R1 complete checkpoint/archive fallback:** measured and shippable.
- **R1 dxi graft onto an arbitrary v7.5.2 Seg EMA:** unmeasured and currently unimplemented.
- **v7.5.2 pose floor = 0.127:** not established until one of those complete artifact branches is
  selected and remeasured on the exact bytes.

**Required gate:** every bank descriptor must bind
`{checkpoint_sha, render_sha, xi_sha, archive_grammar, measured d_seg, measured d_pose, bytes,
hardware_axis, compatibility_key}`. A telemetry boolean cannot imply that an artifact selector ran.

### P0-4 — chroma is silently inherited despite being a later A/B rung

The sealed launch headline holds chroma out and assigns it to its own add-back rung
(`SPEC_v752_20260709.md:48-49,253-259`). The inherited v6/v7 base emits the chroma boundary weight,
band, start epoch, and event (`src/tac/witness_autoconfig.py:984-989,1505-1511`). v7.5.2 merges the
entire v7 base and removes only `--length-sigma-matrix` at `:2962-2974`.

The exact current launch compile consequently includes:

```text
--seg-chroma-boundary-weight 0.1
--seg-chroma-boundary-margin-band 1.0
--seg-chroma-boundary-start-epoch 450
--seg-chroma-boundary-start-event annulus_plateau
```

That is a Class-B loss-form change inside the alleged clean launch-1 trunk, so the pilot cannot
attribute movement to the advertised launch-1 delta.

**Required gate:** remove chroma from launch-1, or amend/reseal the launch as a composed treatment
with its own prior measured add-back receipt. Do not inherit it accidentally.

### P0-5 — amber remains a declared but unrealized admission precondition

The spec requires amber at admission and records OI-5 as launch-blocking
(`SPEC_v752_20260709.md:41,75-89,167,450-470`). A naive `--stability-preset amber` would be defeated
by the inherited explicit `--grad-clip 1.0`, which the config itself documents at
`src/tac/witness_autoconfig.py:2839-2848`. The live launch still emits no stability preset and keeps
the inherited clip.

**Required gate:** either realize the intended values explicitly and verify startup telemetry, or
record an authority-bearing waiver/amendment. “Known open item” is not the same as “launch program
instantiated.”

## P1 findings

### P1-1 — schedule endpoints differ between the sealed document and the program

The spec requires HOSC beta 1→4 during the tau stage (`SPEC_v752_20260709.md:156-170`). The current
program emits 1→3.177 through the inherited v7 event endpoint
(`src/tac/witness_autoconfig.py:1847-1871`). The program also retains the hardcoded-with-waiver
`--curriculum-min-stage-epochs 250` placeholder (`:1423,1495-1500`) while the spec asks for a fitted
post-transition clamp.

These may be defensible amendments; they are not the same schedule. Resolve them in one direction
and add value-identity tests.

### P1-2 — the GO'd self-orient-OFF treatment still carries an unisolated taper

The amendment removes the learned directional input but explicitly retains d_seg-aware taper
(`src/tac/witness_autoconfig.py:2991-2998`). The runtime applies taper to the always-on curvelet bank
independently of self-orient (`experiments/train_levelset_witness_realized_through_R_mlx.py:3097-3136`).
That makes the new OFF+taper vehicle a distinct treatment from the old “taper riding the directional
basis” evidence. The spec already grades taper as INSTANCE/estimated and owes a fresh n600 isolate.

**Required proof:** matched self-orient-OFF arms from the same initialization, one without taper and
one with taper. Until then, either remove taper from launch-1 or label the launch as a combined
unisolated treatment.

### P1-3 — the pose engage mode is not restart-protected

Checkpoint/deploy provenance records `__cfg_pose_finish_start_epoch` but not
`pose_finish_engage_on` (`experiments/train_levelset_witness_realized_through_R_mlx.py:557-558,
603-605`). `_resume_lever_divergences` has no pose-mode check (`:827-962`). A sigma-gated run resumed
without the token silently defaults to muon. The conditioning detector is registered only when it is
actuating (`:6403-6405`), and the current registry restoration does not make an absent current
controller fail closed against a controller present in the sidecar
(`src/tac/witness_control/resume_registry.py:248-310`).

**Required proof:** continuous vs crash/resume on both sides of detector fire, with identical mode,
detector history, engaged epoch, selected terminal artifact, and final hashes.

### P1-4 — there are two public defaults for one program name

`derive_crucible_v752_config` and `compile_crucible_v752_config` default to the sealed
self-orient-ON artifact (`src/tac/witness_autoconfig.py:2929-3039`), while
`compile_crucible_v752_launch_config` defaults OFF (`:3049-3079`). They share the name
`crucible_v752` while producing different argv and typed hashes. Direct callers can validate the
wrong vehicle without any failure.

Use immutable program IDs such as `crucible_v752_sealed_on` and `crucible_v752_launch_off`, or make
the public default the current launch and require an explicit legacy selector.

### P1-5 — dry-start proves mechanics, not semantic completeness

The dry-start proves parser boot, a step, checkpoint production, and resume. It has no expected
active-gate manifest, bank-selector branch test, or sealed-config value-identity check. That is why
all of P0-1 through P0-5 can coexist with a green dry-start.

## Smallest convincing proof matrix

1. **Compiled active-set contract:** require self-orient OFF; sigma-prime absent; conditioning mode
   `sigma_min_plateau`; chroma absent; resolved amber values present; beta endpoint reconciled.
2. **Four detector outcomes:** fires, never fires, degenerate, and canary fails. Prove no epoch cap
   overrides a banked terminal decision.
3. **Artifact selector:** assert the exact complete EMA+xi branch and hashes selected for each outcome.
4. **R1 compatibility:** compare the complete R1 artifact with an explicit R1-xi graft into a v7.5.2
   EMA at n24, then n600, through the actual receiver. Only the latter can prove component composability.
5. **Resume:** continuous vs interrupted just before/after detector fire and stage boundaries.
6. **Attribution:** self-orient-OFF no-taper/+taper and launch-1 no-chroma/+chroma, with paired seeds,
   per-class d_seg, island birth, d_pose, and exact bytes.
7. **Only then:** solo dry-start, from-scratch pilot, byte close, and exact contest-axis replay.

## Scorer-derived additions — frozen evaluator/video pass

The companion evaluator audit at
`.omx/research/ADVISORY_evaluator_video_geometry_20260710.md` adds four constraints without changing
the HOLD disposition.

1. **The fallback selection unit is the complete decoded function.** The scorer consumes exact
   uint8 `.raw` frames, not a checkpoint label or a `dxi` scalar. A banked R1 selection receipt must
   bind the full EMA, xi/per-pair state, decoder/runtime/config hashes, archive bytes, parse-back, and
   decoded output. The evaluator offers no composability assumption under which R1 `dxi` can be
   grafted onto an arbitrary v7.5.2 EMA.
2. **Pose evidence narrows the actuator but does not close it.** A recovered six-pair summary puts
   95.97% of local energy in luma and splits frame energy 54.37%/45.63% between frame0/frame1, but
   restart validation found no raw Jacobian receipt or reproduction command. It motivates a
   luma-dominated empirical control after reproduction; it does not prove chroma structurally null,
   remove the need for both frames, or justify inheriting chroma in the clean launch-1 rung.
3. **The Seg response is not local enough to infer additivity.** Individual Seg margin gradients
   were nonzero across the full 384x512 input. Taper and curvelet claims therefore require the
   decoded full-frame composite, per-class/topology guardrails, and remote interaction checks—not a
   sum of isolated local gains.
4. **Score arithmetic is exact and operating-point dependent.** One corrected Seg cell is worth
   `8.4771050347e-7`; one byte costs `6.6585895312e-7`; one cell buys 1.273108 bytes absent Pose.
   `sqrt(10*0.018)=0.424264`, not approximately 0.02, and `0.022/0.00161=13.6646`; the 0.022 row is
   n24, so it cannot support an n600 matched ratio. These corrections narrow two follow-on sentences
   in `r1_dxi_shippability_byteclose_20260708.md`; they do not revoke #238 complete-artifact
   shippability.
   At run-1 `d_pose=1.79`, a local `1e-6` reduction buys only 1.77485 bytes; at banked R1
   `d_pose=0.001610`, it buys 59.17998 bytes. Use the exact square-root difference for finite moves.
5. **The camera resize has a large exact kernel, but it is a future treatment.** Only 786,432 of
   1,017,336 camera pixels enter either scorer resize; 230,904 per frame are exact unsampled
   coordinates. A future receiver can fill them generically and target the disjoint sampled
   footprints. Do not fold this new representation change into the v7.5.2 launch-1 attribution rung
   or mutate the proven R1 artifact. Any later use still requires exact 1,200-frame raw cardinality.

**Scorer-derived launch disposition:** **HOLD unchanged.** The missing composed #383 mode, epoch-726
bypass, selector/compatibility receipt, clean attribution rung, and amber realization remain exact
blockers.

## Already settled — do not reopen

- Self-orient's realized benefit was approximately zero in the measured owed16 treatment and its
  memory tax was large; OFF is a legitimate operator-selected direction.
- #238 proved the R1 complete checkpoint's trained xi payload is shippable through byte close.
- Stage and intra-stage checkpoint machinery exists and has focused regression coverage.
- The live owed16v2 process is separate evidence; do not stop, rename, or mutate it for this advisory.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/t5_crucible2/SPEC_v752_20260709.md`
- `.omx/research/t5_crucible2/SYNTHESIS_v3_v752_20260709.md`
- `.omx/research/DUAL_CHAIN_BRIEF_385_20260710.md`
- `.omx/research/r1_dxi_shippability_byteclose_20260708.md`
- `src/tac/witness_autoconfig.py`, `src/tac/witness_dsl/curriculum_dsl.py`
- `experiments/train_levelset_witness_realized_through_R_mlx.py`
- `tools/launch_witness_run.py`, `tools/levelset_byte_close_and_eval.py`
- `.omx/research/ADVISORY_evaluator_video_geometry_20260710.md`
- focused v7.5.2, pose-gate, and resume tests
- preserved owed16/owed16v2 run logs and checkpoints (read-only)

**Pointer delta:** none. `0.19109982419209975 [contest-CPU]` remains the local canonical pointer.
