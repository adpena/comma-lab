# TC4 incremental map execution: implementation identity controls

MEASURED [scorer-free macOS-CPU implementation controls]: all 34 real frames
passed (seeded 32 without replacement, plus explicit frames 0 and 599), every
one of 190 groups and all five context maps. The complete-map oracle and
`FastContextMixer.coding/observe` agreed exactly on context codes, source
integer frequencies, and mixed float32 probability rows. Complete snapshots
restored into a fresh instance identically; a fresh process reloaded every
retained completed state and exited zero. No scorer, byte-price experiment,
public receiver, or exact score ran in this subtask.

Source: `experiments/ddm_tc4_fast.py`, SHA256
`6b2ecc39ae32590ef44a638197f6d7d4201316d88dd947b98b4b3787e5fd5197`.
No other implementation file was edited. The existing maps, price and validator
remain unchanged and retain their prior reviewed hashes.

## Mechanism and own source review

Row run lengths update only after actual symbols are observed and reset at
64-column boundaries. Six uint64 bitsets per column hold known unequal vertical
neighbour pairs. An observation can complete a pair with an already-known
neighbour above or below; queries mask out all transition endpoints at or below
the queried row. Highest-set-bit detection uses six integer shifts, never a
floating logarithm. Prior-frame neighbourhood computation remains once per
frame; above-row predictor bins and horizon/run state retain their original
definitions. Scratch resets on begin_frame, so inherited completed-frame
snapshot/restore needs no extra persisted scratch.

Own review traced group guards before scratch mutation, image and 64-cell block
boundaries, zero-bit sentinels, alphabet ranges, integer shift width, class-major
weight layout, zero-feature no-op, and complete-frame resume. No material
finding remained in this source. Parent owns independent review before use.
For receiver integration the experiment import of `ddm_tc4_maps` must become
the corresponding package-relative runtime import. The validation CLI is not
a receiver input or a substitute for public-entrypoint proof.

## Control scope and evidence

Store: `/Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/fast_controls/`.
`INPUTS.json` pins producer, imported map/pricing dependencies, source archive,
field, source preparation manifest, selected frames and counted control
configuration. `frame_*.json` and `state_*.npz` preserve each completed API
control. `RESULT.json` records the full success and fresh-process resume.
Both launch directories retain argv, logs, process and safe-run receipts.

The source is move41 archive SHA
`299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936`
and field SHA
`b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5`.
The control reconstructs each sampled frame's canonical old TC3 state from
the pre-frame preparation checkpoint. TC3's accumulated counts do not depend
on its additional five lane weights; its configuration is explicitly rebound
to the shipped old40, and actual old frequencies are checked against the
move41 source control. TC4 test counts accumulate across selected control
frames, with retained nonzero extra control weights. This exercises nonzero
map correction but is not the production n600 KT history and is not priced.

Measured coding/observe/assertion/end-frame wall time was 0.477053–0.526326 s
per sampled frame, mean 0.509623 s and total 17.327186 s over 34 frames.
This excludes input loading, the complete-map oracle, and full setup; it is
not a before/after speedup or an n600/public-runtime guarantee. The enclosing
first safe-run completed in 30.562 s with exit 0. The fresh-process completed
resume took 0.51 s and exit 0; the detached launcher returned its known
early-exit-during-verification warning because the child finished successfully
inside three seconds. Its actual child exit and safe-run receipt agree on zero.

Both launches used nice=10 with `--nice-best-effort`, projected 1 GiB and a
2 GiB safe-run RSS cap. Those values are planning estimates/caps. Reported RSS
zero is unavailable sandbox visibility, not measured zero memory or a verified
memory bound. All writes remain under the arm's shared 6 GiB accounting cap;
all payload/config/checkpoint bytes are retained, with no deletion or movement.

## Assumption challenge

This changes execution cost only. It retains the same fixed bins, same causal
observations, same frame-frozen counts and same fitted-weight interpretation.
The real five-map/composite prices and literal public-output identity remain
parent ddm_tc4's consumers. Passing sampled identity cannot establish an exact
score or replace the public receiver proof.

LIVE-HYPOTHESES: The incremental implementation may reduce public decode cost
if any TC4 map pays; measured per-frame API work is small, but full public
runtime has not been measured here.

DEAD-ENDS: Full-plane scanning per group is unnecessary for these exact maps;
the incremental API matched the declared complete-map outputs in the tested
34-frame control scope. No compression formulation was closed.

verdict_scope: formulation — tc4's five declared causal context maps and their greedy composition on the move-41 tail (maps 3/4 under the 20 B bar; composition −32 B vs the standalone sum); the mixer's context axis as a family is NOT closed by these measurements.
