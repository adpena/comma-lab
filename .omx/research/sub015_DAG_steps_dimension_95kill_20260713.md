# sub015 DAG feed — steps-dimension 95-kill activation audit — 2026-07-13

**Status:** RESEARCH-ONLY / ticket accounting. No trainer, scorer, provider, GPU, or paid job was launched. `score_claim=false`; `pointer_moved=false`.

**STORES CONSULTED:** `steps_dimension_95kill_20260713_SPEC.md`; `src/tac/witness_dsl/curriculum_dsl.py`; `experiments/train_levelset_witness_realized_through_R_mlx.py`; `src/tac/witness_init/{fixed_quality.py,fresh_trainer_contract.py}`; `tools/measure_witness_fixed_quality.py`; `experiments/results/fresh_init_n8_fixed_quality_20260712/measurement_blocker.json`; `experiments/results/v9_cgauge_432_coherent_arm_20260711/{run.log,launch.sh,levelset_ckpt_stageOctave1_ep251.npz}`; `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`; `reports/basin_finisher_probe_20260707.json`; `src/tac/canonical_equations/quadratic_head_chart_subset_solve_gap_20260707.py`; `src/tac/canonical_equations/segnet_exact_forward_cpu_thread_law_20260713.py`; `experiments/results/segnet_exact_forward_20260713T020000Z/receipt.json`; `experiments/results/cheapen_real95_tilehalo_fp16_20260713/tile_halo_receipt.json` (SHA-256 `b9f264166fea40224966c1902065eebd3fb34949750f87d7fd020e963bb99465`, 10,615 bytes); `experiments/results/cheapen_real95_tilehalo_fp16_20260713/current_wall_receipt.json` (SHA-256 `c9ec6b2d7154a69b98dddd5c8a6a47455187fcdd3c0f4ea6afbff28554ac3614`, 5,673 bytes).

```text
source audit
  ├─ FreSh cold-init source path, default OFF, same-arm resume restoration
  │    └─ cold n600 matched receipt (first crossing + candidate/training timing
  │         + identical speed-config custody + NumPy/R/frozen-CPU authority custody)
  ├─ hardness order construction / unconsumed-extra repair gate
  │    └─ equal-update n600 weighted-vs-uniform matched receipt
  └─ TerminalSolve display-only / no-argv full-P build gate
       └─ cloned full-P n600 accept-or-rollback receipt
             ↓
      epochs/update accounting law
             ↓
      measured sequential time/step composition
             ↓
      fire / rollback decision
```

## Source-audit dispositions

| Lever | Source state | Receipt gate | Current delta |
|---|---|---|---|
| `fresh_frequency_shift_init` | Existing default-OFF `FreShInitControl` / `FreshFrequencyShift` / fixed-quality DSL and runtime path. Candidate selection is cold-start-only; matching FreSh checkpoints restore selected frequency/bias and persisted state. Threshold custody: `run.log` `3860bcf20a341f562e1dd402e281a3298a347f60fa94928cb592ee5dcee480e8`, `launch.sh` `bd760505c445d51dc51d0b31eadd5a4d2628261220ffa46e2474ca83f358c601`, ep251 checkpoint `c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758`; **[macOS-CPU advisory verdict from macOS-MLX training; NON-PROMOTABLE]**. | Cold n600, `eval_every=1`, `ckpt_every=1`, deterministic epoch-0 verdict then derive harness `threshold_factor=0.040763/control_epoch0_d_seg` (must be in `(0,1)`), first crossing, 50 epochs (ASSUMED ceiling). | **UNMEASURED / A/B-TICKET**; `None` |
| `hardness_oversample_lever5` | Existing DSL factory, but trainer builds then shuffles `P+n_extra` order and consumes only `P` draws: some base pairs can be omitted and declared extras are not all consumed. | Repair consumes all visits, asserts `P` base plus `round(P*oversample)` extras, preserves RNG/resume, then equal-update n600 A/B. `oversample=0.5` is an existing DSL default / ASSUMED policy, not a measured optimum; 25 epochs is ASSUMED. | **WIRING_NEEDED**; `None` |
| `TerminalSolve` | Existing `ScheduleDisplay` validates n600 but `flags()` returns `{}`; no in-trainer full-P HVP/CG stage. Frozen A/B premise start: `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz` SHA `6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca`; #341 record `reports/basin_finisher_probe_20260707.json` SHA `7515cfe7495526e0dcae656477dc2718180d71f77447e69c23159250ca1afbb2`; still build-gated. | Typed default-OFF full-P GN/CG build, atomic checkpoints, solver resume state, mutation ledger, exact n600 accept/rollback. Its `0.98` is an ASSUMED preregistration policy; only the numeric threshold computed from measured `d_seg_start` is DERIVED. | **WIRING_NEEDED**; `None` |

## Scoped negatives and reactivation queues

- **Hardness false-actuator — verdict scope:** the current additive/full-base-coverage implementation constructs `P+n_extra`, shuffles it, then consumes only `P` draws. It can alter the seen-pair distribution, but omits some base pairs and cannot support the declared extra-update accounting. This is not a verdict on repaired equal-update allocation or an explicitly pre-registered fixed-budget replacement-resampling formulation. **Reactivation:** consume `len(order)`, assert counts, preserve RNG/resume, then measure weighted versus uniform extras; separately pre-register replacement resampling if that is the intended formulation.
- **TerminalSolve K=8 NO-GO — verdict scope:** post-run subset solve with `K<P`. This is not a verdict on a full-P in-trainer solve. **Reactivation:** build full-P HVP/CG with typed compiler, atomic state, and n600 accept/rollback ledger.
- **FreSh cold-init custody — verdict scope:** candidate selection is skipped on resume, so a non-FreSh checkpoint cannot seed this cold initialization A/B. This does not forbid continuation from the same persisted FreSh arm: matching checkpoints restore selected frequency/bias and FreSh state. **Reactivation:** governed cold-start n600 matched receipt.

## Composition custody

All per-lever epoch savings, update savings, wall fractions, and composed steps saved are **UNKNOWN / None**. A completed but uncrossed receipt is `MEASURED_CENSORED`: it preserves update/timing custody yet still cannot yield an exact delta. Epoch-zero crossings and zero update counts are valid, especially for initialization; `step_fraction_saved` returns `None` when control updates are zero because there is no denominator. A zero-update arm must record `seconds_per_update=None`, never an invented timing. `UNMEASURED` cannot admit wall composition, negative counts/costs are rejected even on the direct elapsed path, and fallback refuses a zero control total. Wall accounting prefers measured direct elapsed-to-crossing. Its fallback is only `U*t_update + E*t_recurring_nonupdate + one_time + terminal_critical_path` when every recurring critical-path term is allocated; async service is recorded separately and excluded unless a measured wait enters the critical path. `solver_hvp_steps` remain separate from optimizer updates so a TerminalSolve cannot appear to have done zero work; their wall cost travels in critical-path time. An independent product is only an `ASSUMED_INDEPENDENT_SYMBOLIC_SCENARIO`; it cannot be a composed measurement. The tile-halo receipt supplies a **DERIVED ideal speedup upper bound of 1.0** only for the frozen-B2 finite input-crop tile-with-halo formulation, with measured n600 boundary coverage. It is neither a whole-step wall split nor a factor to compose here.

## Receipt authority and exact-speed gate

Every A/B ticket has typed `speed_configuration_rule` and `measurement_authority_rule`; a receipt-backed row additionally requires nonempty `speed_configuration_custody` and `measurement_authority_custody`. The speed custody must machine-readably show identical exact speed configuration in both arms, every currently admitted/requested neutral fleet speed lever ON, and `all_requested_speed_levers_on=true`; otherwise record a blocker and do not call the window compliant. Authority custody must show deterministic NumPy-fp32 realization through actual `R` plus the frozen CPU-torch scorer on all 600 states. MLX training is advisory and provides no score authority. The current-wall receipt has `all_requested_speed_levers_on=false`, so it cannot supply either a compliant speed gate or requested wall composition.

The measured `2.995x` CPU one-thread result remains a scorer-forward subcomponent factor, not a measured whole-step multiplier. The sibling current-wall receipt **DERIVES** 295.352 seconds/epoch from measured n600 log timestamps on the observed critical path and records zero measured async verdict-service wait, but declares `composition_admissible=false`, `all_requested_speed_levers_on=false`, and a fully unallocated training residual. It therefore forces an explicit wall-composition refusal: `wall_fraction=None` and `wall_fraction_saved=None`; its residual is not folded into update time.

**Drift-check boundary:** `tools/triality_drift_detector.py` has no narrow-path interface and inspects committed windows while updating shared marker state. These four files are uncommitted and this lane is forbidden from mutating shared state, so there is no applicable global detector verdict for them yet; no unrelated sibling drift is attributed to this feed. Main must run the global committed-window detector at integration.
