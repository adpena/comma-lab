# Inverse-depth Riemannian compander build — 2026-07-13

**Lane:** `compander_build`  
**Outcome:** `BUILT-NEVER-FIRED`; `$0` local; uncommitted; no launch  
**Lane maturity:** `L1` (`impl_complete` only; real receiver A/B and contest axes remain open)  
**Pointer delta:** `NONE`  
**Lever:** `MarginCompandedGroundChart` (default OFF)

## Outcome

The S1 softened-inverse-depth profile is implemented as a deterministic coordinate transform and
composed *after* the existing projective `GroundFrameChart`. It does not replace perspective-aware
chart #185/#194. The implementation changes where a fixed coordinate field spends capacity; it
does not change feature width, parameter count, optimizer steps, cache shape, or matched archive
budget.

The exact measured constants are:

```text
v_h = 174.0 rows
delta = 32.5257801441824 rows
seed = 0 (DERIVED analytic-family identity; the current map consumes no RNG)
```

Value provenance is
`.omx/research/manifold_geometry_slots_probe_s1_s2_20260713.json`. The fitted profile is
video-derived and therefore must be counted in treatment archive bytes; the generic transform does
not launder the fitted payload into free decoder code.

## Implementation

- `src/tac/boundary_math/inverse_depth_compander.py` provides validated profile custody, analytic
  fp32 forward/inverse row and normalized-coordinate maps, the MLX twin, and
  `MarginCompandedGroundChart` composition/resume state.
- `src/tac/witness_dsl/curriculum_dsl.py` provides the nilary/defaulted factory
  `MarginCompandedGroundChart`. It emits only flags present in the real trainer parser.
- `experiments/train_levelset_witness_realized_through_R_mlx.py` adds the default-OFF typed parser
  surface, refuses compander-without-projective-chart, wraps the base chart after it is built, keeps
  the OFF/base-only path unchanged, persists deploy/resume identity, and guards divergence.
- `src/tac/witness_control/resume_registry.py` canonically names the direct controller
  `margin_compander`; the trainer registers `__mcc_` state only for the treatment arm.
- `tools/build_compander_ground_class_pair_ledger.py` builds a fingerprint-resumable atomic ledger
  from cached argmax arrays only.
- `tools/probe_compander_receiver_close_ab.py` is the future strict receipt comparator. It runs no
  model or archive operation itself.

Resume state includes enabled/version, grid height, exact horizon/delta, and seed. A mismatched
resume identity fails closed. The feature and per-pair cache shapes remain identical; therefore the
`launch_prego_worklist` sibling needs no memory-envelope recompile adjustment from this lever.

## Expected mechanism and honesty boundary

The local metric density is `sqrt(g_vv) proportional to (v-v_h+delta)^-2`. Its cumulative chart
spends more coordinate resolution in the upper ground support, including the measured dash-erasure
band, and less in the lower tail. This is a capacity-placement hypothesis, not new capacity and not
a score claim. A proxy-density win cannot establish that training uses the placement effectively.

The future A/B must hold seed, optimizer steps, total archive bytes, authority axis, receiver runtime,
and parse-back custody fixed. It must report treatment-minus-control `d_seg` for all five classes,
with **Lane primary**, and must show non-worsening `d_pose`; rate is matched exactly. The treatment
chart payload must have explicit bytes/SHA custody inside the counted archive.

## Ground-class-pair n600 ledger

Artifact:
`.omx/research/compander_ground_class_pair_ledger_n600_20260713.json`

`MEASURED [cached-argmax local-CPU analysis], score_authority=false`:

- seven source files have bytes and SHA-256 custody;
- exactly 600 unique pair indices cover `0..599` without gaps or duplicates;
- all 20 directed and 10 undirected class pairs remain visible;
- strict planar-ground classification is only Road<->Lane (no global planar claim for the mixed
  Undrivable class);
- total cached flips: `785058`, disagreement rate `0.006655019124348958`;
- Road->Lane: `102877`; Lane->Road: `344743`; combined: `447620`, or
  `0.5701744329718313` of all flips;
- for combined Road<->Lane on rows `v>174`, fixed S1 softened-inverse-depth JS is
  `0.13905534495630045`, versus uniform `0.23730967324348934` and unshifted log-depth
  `0.25990849245359443`.

The per-class ledger confirms the fixed S1 density is the best of these three profiles for the
Road<->Lane cached-flip mechanism. It does **not** close promotion: no chart treatment was trained,
receiver-closed, scored, or byte-counted.

## Verification

Focused verification completed without trainer/scorer/evaluator execution:

```text
module tests:                         10 passed, 2 environment-skipped
DSL/resume/lever structural tests:   22 passed
ledger synthetic tests:               3 passed
A/B receipt synthetic tests:          8 passed
```

The two skips are the MLX CPU/default-device assertions. `mlx.core` imports, but this managed session
raises `No Metal device available` when it initializes any MLX array, including the CPU stream. The
test still requires bit-exact NumPy/MLX CPU output wherever the device is executable; no parity value
is fabricated here. NumPy forward/inverse, monotonicity, density placement, projective composition,
and resume tests pass.

The broader focused suite reports `142 passed, 4 deselected`; the deselections are the two new and
two existing MLX-device tests deliberately excluded from the sandbox-blocked aggregate run.

All new Python modules/tests are ruff-clean. The shared trainer and resume-registry files have
pre-existing broad lint debt; this lane verifies them with syntax compilation plus focused fatal/
undefined-name checks instead of rewriting unrelated hot-file regions.

## Triality and apparatus wire-in

- **DSL:** `MarginCompandedGroundChart`; real-parser validated, default OFF, lever-registry visible.
- **DAG:** `.omx/research/compander_build_DAG_FEED_20260713.md`.
- **Equation:** existing `flip_density_chart_metric_v1`; no duplicate equation.
- **Sensitivity map:** cached row/class density is a mechanism prior, not an effect anchor; future
  receiver A/B owns the empirical Lane/all-class update.
- **Pareto:** matched rate/steps, Lane/all-class d_seg, non-worsening pose.
- **Bit allocator:** unchanged total capacity; chart payload counted, no byte saving claimed.
- **Cathedral/autopilot:** curriculum pool duty queue, not a launch or dispatch mutation.
- **Continual learning:** custody ledger + `built-never-fired` pool row.
- **Probe disambiguator:** strict future receipt comparator separates proxy-density fit from realized
  receiver effect.

## Storage, execution, and remaining gate

The build produces only source, tests, a 335 KB JSON ledger, and small Markdown receipts. Atomic
temporary files are success-cleaned; no bulky output, `/tmp` evidence, archive copy, raw-frame tree,
or checkpoint is created. Existing cached inputs are read-only and unchanged.

No training, evaluator, scorer, exact replay, archive mutation, provider/GPU dispatch, live-run
mutation, launch-ticket edit, or commit occurred. The remaining gate is the governed, receiver-closed,
counted, matched-byte/matched-step n600 control/treatment A/B described above.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; v7.5 §8 operating contract; v8 spec; top project-memory
entries; current lane/subagent registries; last-24-hour directives; latest sister findings/session,
council, and design memos; `reports/latest.md` plus canonical frontier scan; S1 source memo/receipt;
canonical equations registry; `GroundFrameChart` source/tests; trainer DSL/parser/resume paths;
lever/activation registries; curriculum candidate pool; cached n600 GT and six witness chunks.
