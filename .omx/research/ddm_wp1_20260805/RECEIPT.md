# ddm_wp1 receipt - TR1 Muon finisher port plus vh1 cached reads

Axis: mixed source-inspection and cached-receipt reads only. No launch, no
scorer forward, no evaluator, no writes to the live jd6 run directory.
Score claim: false. Pointer moved: false.

Own-vehicle frontier unchanged:

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`

## Unit 1 - Muon finisher port

Verdict: BUILT, default-off, not fired.

Edit sites:

- `experiments/train_tr1_partition_renderer_mlx.py`
  - Added `--jd1-finisher {off,muon}`, default `off`, args-only.
  - Added TR1 renderer-param routing for Muon on renderer weight tensors
    (`w_conv`, `w_up`, `w_head`, `s_conv`, `s_up`, `s_head`); tokens, bias,
    gates, gains, and 1-D/non-renderer leaves stay on Adam.
  - Added a real `mlx.optimizers.Muon` + Adam `optim.MultiOptimizer` builder,
    Adam first-moment to Muon momentum seeding, MultiOptimizer optimizer-state
    path mapping, checkpoint/metadata persistence, and finisher switch telemetry.
  - Runtime refuses inert or mid-window use: Muon requires active JD1 pose finish,
    `--resume-from`, and `--jd1-pose-finish-engage-on start_epoch`.
- `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`
  - Added `lever_jd1_muon_finisher()` as an ON-only lever factory. The lever
    emits only `--jd1-finisher muon` and documents that the runtime requires a
    JD1 start-epoch resumed boundary.
- `src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`
  - Added AST/validation coverage for the composed JD1 Muon lever.
- `src/tac/tests/test_ddm_wp1_tr1_muon_finisher.py`
  - Added pure tests for filter routing, split counts, state-path mapping,
    beta1-derived momentum, args-only default-off behavior, and inert refusal.
- `src/tac/tests/test_ddm_bp1_boundary_reset_race.py`
  - Updated the existing optimizer-wiring AST guard to allow the dedicated
    WP1 Muon-finisher Adam fallback while still requiring exactly one
    non-finisher Adam construction wired to the boundary-reset bias-correction
    selector.

Muon derivation values and sources:

- Structure source: canonical `muon_finisher_schedule_warmstart_and_lr_anneal_v1`
  plus witness MLX implementation in
  `experiments/train_witness_realized_through_R_mlx.py` and reusable MLX helper
  `src/tac/optimization/muon_finisher_mlx.py`.
- TR1 value source: TR1 optimizer geometry, not witness constants.
- `muon_lr = cfg.lr`; live TR1 tickets use `cfg.lr = 0.002`.
- Adam fallback LR for non-Muon leaves is also `cfg.lr = 0.002`.
- `muon_momentum = RESET_ADAM_BETAS[0] = 0.9`, because warm-started Muon `v`
  consumes the outgoing Adam first moment `m` and should preserve the TR1
  first-moment time constant.
- `muon_ns_steps = 5`, matching MLX Muon's Newton-Schulz default and the existing
  repo Muon helper contract.
- `muon_weight_decay = 0.0`, preserving TR1's current no-decoupled-decay regime.
- LR floor source: existing `ddm_la1_jd1_lr_anneal.v1` tail law. If an explicit
  `--jd1-lr-final-frac` is absent, the final fraction is
  `tail_sd / (tail_sd + tail_half_range)`, falling back to `1.0` for a flat tail.
  If the fraction is below `1.0`, the Muon group cosine-decays across remaining
  JD1 updates; otherwise it stays scalar.

Tests:

- `python3 -m py_compile experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/spec_tr1_renderer_20260728.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py src/tac/tests/test_ddm_wp1_tr1_muon_finisher.py`
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_wp1_tr1_muon_finisher.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py -q`
  - Result: `21 passed in 0.70s`.
- `git diff --check -- experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/spec_tr1_renderer_20260728.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py src/tac/tests/test_ddm_wp1_tr1_muon_finisher.py`
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_trainer_actually_wires_the_arm_into_its_optimizer -q`
  - Result: `1 passed`.
- Review tracker policy checks after two mark-file passes:
  - `experiments/train_tr1_partition_renderer_mlx.py`: 123 compliant, 0 violations.
  - `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`: 46 compliant, 0 violations.
  - `src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`: 15 compliant, 0 violations.
  - `src/tac/tests/test_ddm_wp1_tr1_muon_finisher.py`: 8 compliant, 0 violations.
  - `src/tac/tests/test_ddm_bp1_boundary_reset_race.py`: 50 compliant, 0 violations.

Note: importing MLX optimizers for a live optimizer-step smoke is not possible in
this sandbox because MLX fails at device load with `No Metal device available`.
The landed tests therefore verify source-level routing and validation only.

## Unit 2 - vh1 row 3 along-tangent spectral read

Verdict: QUEUED-WITH-A-FIRE-ORDER.

Transfer yes/no: UNKNOWN, not measured on TR1.

Searched scope:

- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805`
- live read-only run
  `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1646`
- `jd4_endpoint_n600_both_bases.json`
- `jd5_endpoint_n600_both_bases.json`
- `.omx/research/ddm_vh1_v8v9v10_harvest_20260730.md`

Finding: cached TR1 endpoint receipts contain per-pair aggregate `d_seg` and
`d_pose`, and the live run directory contains checkpoints and telemetry only. I
did not find a cached TR1 residual field, realized argmax tensor, rendered frame
cache, or per-pixel error atlas in that scope. Measuring along-boundary tangent
vs across-boundary spectral energy would require a fresh render/scorer-derived
residual surface, which this charter forbids.

Fire order:

1. After jd6 has a terminal endpoint or the one-scorer fleet slot is explicitly
   assigned, materialize a TR1 endpoint residual atlas to SSD: pair id, rendered
   frame hash, realized argmax/error mask, GT argmax, boundary tangent/normal
   fields, and scorer batch provenance.
2. Run the row-3 spectral analyzer against that cached atlas only.
3. Record along/across energy ratio, denominator pixels, pair selection, and
   axis. Do not promote from the historical 3.2x witness law without this TR1
   atlas.

## Unit 3 - vh1 row 4 g3 hard-pair transfer

Verdict: MEASURED from cached receipts.

Artifact:

- `.omx/research/ddm_wp1_20260805/row4_g3_transfer_tr1_endpoint_delta.json`

Inputs:

- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd4_endpoint_n600_both_bases.json`
- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd5_endpoint_n600_both_bases.json`
- `.omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/hard_pair_registry.json`

Delta definition: `jd5 d_seg_per_pair - jd4 d_seg_per_pair`; positive means JD5
is worse than JD4. Denominator: n600 per basis. Axis:
`[macOS-CPU advisory cached-receipt read]`.

Results:

| basis | full mean delta | top64 r(delta) | top64 r(abs delta) | top64 mean delta | top64 abs/full |
| --- | ---: | ---: | ---: | ---: | ---: |
| ema | +0.0002075280 | +0.2672903589 | +0.2348742495 | +0.0006207625 | 1.5301133912 |
| live | +0.0003772227 | -0.1450982096 | -0.0398190339 | +0.0000843207 | 0.9090700192 |

Transfer yes/no: NO as a cross-basis/default decision rule; PARTIAL yes for the
EMA basis only. EMA top64 catches larger deterioration than the full mean, but
live is weak/inverted. This does not license using g3 subsets as a sole TR1
n600 surrogate.

## Unit 4 - vh1 row 9 flicker-phase coherence

Verdict: QUEUED-WITH-A-FIRE-ORDER.

Transfer yes/no: UNKNOWN, not measured on TR1.

Recall source:

- `/Volumes/VertigoDataTier/pact/ddm_of1_20260729/flicker_phase_coherence_receipt.json`
  measured OF1 W1-COH on a prior atlas: `total_flips=458738`,
  `flicker_flip_frac=0.4954200437`, area-weighted phase agreement
  `0.8692128676`, falsifier not fired.
- `/Volumes/VertigoDataTier/pact/ddm_of1_20260729/of1_coherence_probes.py`
  requires an atlas with per-flip pair/y/x/gt/realized class and GT flicker.

Finding: I did not find a TR1 endpoint flip atlas or per-flip realized class
stream in the jd4/jd5/jd6 cached scope. The endpoint JSON is per-pair aggregate,
which is insufficient for W1-COH connected-component phase coherence. Reusing
OF1 arithmetic without TR1 flip sites would be a false transfer.

Fire order:

1. Materialize a TR1 endpoint flip atlas from an assigned endpoint scorer run,
   chunked under the fleet limit, and persist it on SSD with hashes.
2. Run `of1_coherence_probes.py --probe flicker` against the TR1 atlas and
   `gt_n600.npz`.
3. Record phase agreement, flicker flip fraction, support-byte assumption, and
   break-even against the current waterline. Keep OF1 historical values as a
   prior only until this is done.

## Unit 5 - vh1 row 13 solve_project rendered-init verification

Verdict: QUEUED-WITH-A-FIRE-ORDER.

Transfer yes/no: NO measured proof on the live path this turn.

Finding:

- The sealed TP1 ticket and current TR1 configs declare
  `token_init_mode = solve_project`.
- The live JD6 lineage is a resume chain from earlier checkpoints:
  TP1 `full_birth_lane_on_w4m` -> JD1 -> JD3 -> JD4 -> JD6.
- Source inspection shows `solve_project` runs only when
  `args.resume_from is None`; all searched live-chain runs are resume launches.
- I did not find `solve_init_projected`, `solve_init_targets_ready`,
  `stage_solve_init_pretrain.npz`, or a rendered-init class-mass/sky-detector
  receipt in the live-chain scopes searched.

This is not evidence that solve_project is defective. It is bounded absence of a
rendered-init verification artifact in the live resume chain.

Fire order:

1. On the next non-resume solve_project launch, or on a separately assigned
   read-only verification pass over the exact initial checkpoint, render the
   initialized head before scorer training.
2. Persist a rendered-init receipt with per-class mass, Lane/Movable protected
   channel checks, sky-detector class leakage checks, image/hash provenance, and
   no score claim.
3. Add a regression only if the rendered-init receipt shows a defect.

## RECALL EVIDENCE

Governing files read:

- `PROGRAM.md`
- `CLAUDE.md` / `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`
- WP1 charter prompt and common arm contract under the repo-local codex-runs
  charter directory.

Queries and what changed:

- `rg -n "muon|newton_schulz|warm-start|lr.*anneal"` across experiments, src,
  docs, and research. Found the existing MLX Muon helper, witness warm-start
  code, canonical Muon schedule law, and torch v9 optimizer adapter. This
  changed the plan from a new local optimizer to a `mlx.optimizers.Muon`
  MultiOptimizer port with TR1-specific routing and value derivation.
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  Muon/JD1/finisher terms. Relevant hits included
  `muon_finisher_schedule_warmstart_and_lr_anneal_v1`,
  `muon_switch_conditioning_criterion_v1`,
  `anisotropic_basis_along_tangent_frequency_deficit_v1`, and
  `jd1_plateau_tail_average_ema_v1`. This kept the Muon port warm-started,
  annealed, and JD1-boundary-only.
- `rg -n "solve_project|solve_init_projected|rendered.*init"` across source,
  research, and SSD run scopes. Found the solve_project code path and the
  resume gate, but no live-chain rendered-init artifact. This changed row 13 to
  queued-with-fire-order.
- `find` and structured JSON reads under
  `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805` and
  `/Volumes/VertigoDataTier/pact/ddm_of1_20260729`. Found per-pair endpoint
  arrays for row 4, OF1 historical W1-COH receipt, and no TR1 per-flip or
  residual atlas. This made row 4 measured and rows 3/9 queued.
- `CANONICAL_RESEARCH_INDEX_20260629.md` search for Muon/g3/spectral/flicker.
  Found witness Muon LR conflict notes and vh1/g3 references. It did not change
  the TR1-specific value derivation because the charter forbids importing
  witness constants directly.

Protected files were not edited:

- `.omx/research/ddm_cr1_composition_row_827_20260801.md`
- `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`
- `src/tac/optimization/direct_description_carrier_compose.py`

## Boundary

No exact eval was run. No scorer job was launched. No live jd6 file was modified.
The own-vehicle frontier remains:

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`
