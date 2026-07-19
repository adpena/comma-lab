# M1 / STEP 3 — C2 banded-around-source generator n600 run: LAUNCH BLOCKED (missing integration glue), recovery spec

**UTC:** 2026-07-19
**Arm:** M1 launch respawn (task #575) — predecessor DIED (Fable credits) mid-glue-build; worktree auto-cleaned; the run/solve GLUE was LOST, the component TOOLING survived on `main`.
**Verdict:** **NOT FIREABLE from `main`. HARD STOP.** The C2 banded-generator n600 run is gated on integration CODE that does not exist on `main` (it was in the lost worktree), not on a config that can be reconstructed-and-launched. No launch fired. No dispatch spent.
**Pointer delta:** **NONE.** `0.1910828242 [contest-CPU Linux x86_64]` UNMOVED. This unit is MEANS (signal-recovery), not an exact row.
**Verdict scope:** the fireability of M1 from current `main` HEAD (`34d4948721`). Not a verdict on the C2 vehicle, the banded-generator design, or the SPEC roadmap — those are intact and correct; only the integration is unbuilt.

## Stores consulted
- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`.
- `.omx/research/SPEC_v10_integer_plane_vehicle_20260719.md` §ADDENDUM 2026-07-19-B (M1/M2/M3 ladder + STEP 0–6 roadmap; STEP 3 = this arm).
- `.omx/research/generator_description_crux_synthesis_20260719.md` §0 (budget box + 4-order hole), §1 family 1 (the blocker), §2.1 (the base), §3.1 (the run recipe + glue estimate).
- `.omx/research/c2_integer_plane_emitter_build_20260719_codex.md` (BUILD COMPLETE; blocker #3 "Rate/receiver custody owed … the current smoke is not byte-closed").
- `.omx/research/c2_integer_plane_emitter_implementation_spec_20260719.md` (Unit A/B/C; "The lever is argv-inert until a later governed trainer integration").
- Built modules: `src/tac/boundary_math/integer_plane_emitter.py` · `src/tac/witness_dsl/integer_plane_emitter_policy.py` · `src/tac/witness_dsl/curriculum_dsl.py` (Lever factory) · `tools/measure_c2_integer_plane_emitter.py` · `tools/levelset_byte_close_and_eval.py` · `src/tac/optimization/joint_seg_pose_rate.py` (margin-band law) · `src/tac/boundary_math/horizon_weighted_margin.py`.
- Dead run dir `experiments/results/levelset_n600_witness_20260717T113932Z/` (STOP_LEDGER 2026-07-18; the `c2_surgical_warm` LEVELSET-witness run — a DIFFERENT vehicle).

## What M1 requires (from SPEC + crux, first-hand)
Per crux §2.1 + §3.1 + family-1 blocker, the M1 measurement is a **counted ŷ-generator solved within source-centered margin bands**: the C2 integer-plane emitter's quotient-residual **trained** to sit inside per-pixel seg margin bands *around the source planes* (paying only residual-vs-free-predictor; pose falls out via `pose_plane_proximity_corollary_v1`), then **byte-closed** through the production receiver + hard oracle at n600. Target zone **60–85 KB total at d_seg ≤ ~1e-3**. Gate: a byte-closed (bytes, d_seg, d_pose) row.

## What is BUILT on `main` (survived) — component tooling
1. **Emitter forward** (`integer_plane_emitter.py`, 228 tests): NumPy/Torch/lazy-MLX two-plane emitter, U4 head basis, exact factor-2 lattice bridge, train-least deletion executable, `CapacitySignature`.
2. **Margin-band law**: `joint_seg_pose_rate.py` + `horizon_weighted_margin.py` (the band/margin loss primitives).
3. **DSL Lever**: `IntegerPlaneEmitter(...)` factory in `curriculum_dsl.py` — default-OFF, registry/activation-visible.
4. **Advisory measure harness** (`measure_c2_integer_plane_emitter.py`): `fixture` + `advisory` modes only.
5. **A byte-close/eval tool** (`levelset_byte_close_and_eval.py`) — for the `lever_b_levelset_generator` vehicle.

## What is MISSING (the lost glue) — verified absent on `main`
This is the "trainer config + band loss (~300–600 LOC)" the crux §3.1 itself names, plus a byte-close adapter. Verified by direct grep at HEAD `34d4948721`:

1. **No trainer consumes the emitter.** Only `curriculum_dsl.py` references `IntegerPlaneEmitter` (the factory definition); NO trainer/launcher/`witness_autoconfig.py` reads the policy. The impl spec confirms the Lever "is argv-inert until a later governed trainer integration" and "compilation must be byte-identical to baseline" — i.e. turning the flag on today changes NO training. There is no governed-launch config that references the emitter. **A heavy launch cannot be reconstructed because the launch target does not exist.**
2. **No band-training loop.** The margin-band law exists as a loss primitive but is NOT wired into any training loop that solves the emitter's quotient residual within source-centered bands. `measure_c2_integer_plane_emitter.py advisory` runs a *fresh-seeded, UNTRAINED* residual (`[macOS-CPU advisory, untrained]`); it does not train.
3. **No byte-close for the emitter.** `measure_c2_integer_plane_emitter.py` has only `fixture`/`advisory` — no archive build, no rate accounting. `levelset_byte_close_and_eval.py` has **0** references to the emitter (it byte-closes `lever_b_levelset_generator` from a `--ckpt-dir`). Build-memo blocker #3 states plainly: *"the current smoke is not byte-closed"* and the advisory's base is *"a full source-derived exact-projection base that is not serialized or rate-counted in C2."* Citing the advisory as a "byte-closed row" would be a rate FAKE.

## Why the two tempting shortcuts are FORBIDDEN (NO-FAKE)
- **Fire the levelset witness trainer instead** (it IS governed-launchable): WRONG VEHICLE. That is the coord-INR `train_levelset_witness_realized_through_R_mlx.py` path; its margin lever (`HorizonWeightedMargin`) was measured WEAK (#141/#169, dS ceiling 0.012–0.024) and explicitly DEFERRED in the last run's `launch.sh`. It does NOT produce a "counted ŷ-generator solved within source-centered bands." Firing it and calling it M1 = fake M1.
- **Cite the advisory smoke's low d_seg (~1.18e-4 mean n6) as the M1 row**: its base is source-derived and un-rate-counted; there is no archive, no bytes. That is the "surrogate-optimized-but-not-exact-authority-verified" + rate-fake class.

## Recovery spec — what a build-arm must land to make M1 fireable (rebuilds the lost glue)
Estimated ~300–600 LOC (crux §3.1), all inside the mutation frontier, resumable + per-stage-checkpoint P0:
1. **Band-training loop** that optimizes the emitter's quotient residual against the margin-band law (`joint_seg_pose_rate.py`) with the band centered on the exact source scorer planes (per §2.1 reframe). Realized-through-R (the exact factor-2 lattice), EMA shadow saved, per-stage checkpoints, `--resume-from`.
2. **DSL wiring**: make `IntegerPlaneEmitterPolicy` argv-EFFECTIVE — the trainer must actually consume it (today it is inert). New/changed levers land as `Lever` factories in `curriculum_dsl.py` (already registered) + a DAG FEED line (triality), never a hand-added flag.
3. **Byte-close adapter** for the emitter: emit a counted archive (base generator description + pair codes + band-slack repair) whose inflate bit-identically reproduces the emitter forward, then run it + the hard CPU-Torch oracle at n600. Either extend `levelset_byte_close_and_eval.py` to the emitter vehicle or add a sibling `measure_c2 … byte-close` mode. This closes build-memo blocker #3.
4. **Governed-launch config** (`tac-config-family` + DSL compile hash + memory preflight rc=4 + config-freshness rc=6) referencing the new trainer.
5. **Runnability-first**: short local smoke AT n600 scale exercising the real memory/throughput path (record peak RSS) BEFORE the full fire; every scored quantity measured through the real byte-closed decode.

## Triality legs
- **DSL leg:** `IntegerPlaneEmitter` factory exists but is INERT; the recovery makes it effective (owed).
- **DAG leg:** FEED-m1-launch-blocked (this memo) — edge in: crux-synthesis M1 edge-out; edge out: build-arm for the band-training + byte-close glue.
- **Equation leg:** no new equation; `seg_secant_rd_curve` break-even (150.18 B/1e-6) + score law price the target, unchanged.
- **Pointer leg:** UNMOVED (0.19108). This unit is MEANS.

## Honest position
Task #575 as scoped ("fire the M1 run through the governed launcher") is **not completable from current `main`** — not because of a governor/memory REFUSE, but because the launch TARGET (band-training trainer + emitter byte-close) is unbuilt code that was lost with the predecessor's worktree. No signal is lost: this memo reconstructs exactly what the glue must be so a build-arm can rebuild it. Recommended next action: authorize a **build-arm** to land items 1–4 above (with its own 2-pass review + n600 runnability smoke gates), then re-attempt the governed launch. Pointer 0.19108 UNMOVED.
