# Steps-dimension 95-kill activation audit and epochs-to-target law — 2026-07-13

**Verdict:** FreSh delta is **UNMEASURED / A/B-TICKET**; hardness delta is **WIRING_NEEDED**; TerminalSolve delta is **WIRING_NEEDED**. Each is `None`; composed steps saved is **UNKNOWN**, not zero. No score claim and no frontier-pointer movement are made.

**STORES CONSULTED:** `.omx/research/steps_dimension_95kill_20260713_SPEC.md`; `src/tac/witness_dsl/curriculum_dsl.py`; `experiments/train_levelset_witness_realized_through_R_mlx.py`; `src/tac/witness_init/{fixed_quality.py,fresh_trainer_contract.py}`; `tools/measure_witness_fixed_quality.py`; `experiments/results/fresh_init_n8_fixed_quality_20260712/measurement_blocker.json`; `experiments/results/v9_cgauge_432_coherent_arm_20260711/{run.log,launch.sh,levelset_ckpt_stageOctave1_ep251.npz}`; `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`; `reports/basin_finisher_probe_20260707.json`; `src/tac/canonical_equations/{fresh_frequency_shift_init_20260712.py,quadratic_head_chart_subset_solve_gap_20260707.py,segnet_exact_forward_cpu_thread_law_20260713.py}`; `experiments/results/segnet_exact_forward_20260713T020000Z/receipt.json`; `experiments/results/cheapen_real95_tilehalo_fp16_20260713/tile_halo_receipt.json` (SHA-256 `b9f264166fea40224966c1902065eebd3fb34949750f87d7fd020e963bb99465`, 10,615 bytes); `experiments/results/cheapen_real95_tilehalo_fp16_20260713/current_wall_receipt.json` (SHA-256 `c9ec6b2d7154a69b98dddd5c8a6a47455187fcdd3c0f4ea6afbff28554ac3614`, 5,673 bytes).

## Activation truth

| Lever | Existing surface | Exact disposition | Delta custody |
|---|---|---|---|
| FreSh | Default-OFF `FreShInitControl`, `FreshFrequencyShift`, and `FreShFixedQualitySlice`; init/receipt/resume surfaces exist. | **Runtime/DSL fire-ready only.** It is not n600-measurement-ready: the candidate sweep is cold-start-only, so a non-FreSh checkpoint cannot seed the init A/B; matching FreSh checkpoints may continue bit-faithfully because they restore frequency/bias and persisted state. | `epochs_saved=None`, `step_fraction_saved=None`, `wall_fraction_saved=None` |
| Hardness oversample | Default-OFF `HardnessOversample` and candidate-pool row exist. | **False actuator relative to its declared additive/full-base-coverage semantics / WIRING_NEEDED.** `order` contains and shuffles `P+n_extra`, but the loop consumes only `P` draws: it can omit base pairs and cannot support promised extra-update or equal-update accounting. | `epochs_saved=None`, `step_fraction_saved=None`, `wall_fraction_saved=None` |
| TerminalSolve | Existing `TerminalSolve` is a display/validation object. | **Designed, not built / WIRING_NEEDED.** It returns no argv and no full-P in-trainer GN/CG stage exists. K=8 post-run subset NO-GO does not close the full-P family. | `epochs_saved=None`, `step_fraction_saved=None`, `wall_fraction_saved=None` |

FreSh is consequently the only one of the three that is runtime/DSL fire-ready. That is deliberately separate from a matched n600 result: a fire-ready default-OFF mechanism cannot yield an epochs-to-target delta until its cold matched receipt records both first crossings and all timing overhead.

## Accounting law

For receipt-backed matched arms only:

\[
E_{saved}=E_c-E_t,\qquad f_{step,saved}=1-U_t/U_c
\]

\[
W=U\,t_{update}+E\,t_{recurring\ nonupdate}+t_{one\ time}+t_{terminal\ critical\ path}.
\]

`f_wall = elapsed_treatment/elapsed_control` is preferred when both direct elapsed-to-crossing receipts exist. Otherwise it is `W_t/W_c` only if every recurring critical-path term is allocated; async service is recorded separately and excluded unless a measured wait proves it critical-path.

`U` is exact optimizer updates. Therefore a repaired hardness arm that visits additional pairs cannot present a nominal-epoch decrease as a step saving. Epoch-zero crossings and zero optimizer-update counts are valid (especially for an initialization arm). `epochs_saved` accepts zero; `step_fraction_saved` returns `None` when `U_control=0`, because a denominator-free rate is not a saving. A zero-update arm must record `seconds_per_update=None`; no timing is invented. Negative counts/costs are rejected even when direct elapsed exists, and fallback wall division rejects a zero control total. `UNMEASURED` rows cannot admit wall composition; a receipt helper defaults composition to false with a nonempty refusal until it is explicitly admitted. Non-n600 rows, and any claimed `MEASURED` row missing a crossing, update count, solver-HVP count, critical-path timing, speed-config custody, or authority custody are rejected. `solver_hvp_steps` are explicit and separate from optimizer updates; direct solve work is not silently normalized away, and its elapsed cost stays on the wall-critical path. A completed window with a missing crossing is instead `MEASURED_CENSORED`: it retains update/timing custody while its exact delta remains unavailable. A missing crossing stays `None`.

Sequential step/time composition is allowed only after a measured sequential receipt supplies the relevant updates and timing. Multiplying independently observed savings is an **ASSUMED** symbolic scenario, never a measured composed saving.

## Frozen tickets

- **FreSh cold n600 fixed-quality A/B:** control is `FreShInitControl`; treatment is `FreshFrequencyShift`; both compose `FreShFixedQualitySlice(eval_every=1, ckpt_every=1)`. Start cold only. The existing receipt harness accepts `threshold_factor`, not an absolute d_seg CLI: after the deterministic **MEASURED** control epoch-0 verdict, derive `threshold_factor=0.040763/control_epoch0_d_seg` and refuse unless it lies strictly in `(0,1)`. This makes the existing harness measure the preregistered absolute target. `0.040763` is the **MEASURED advisory** reference; 50 epochs is an **ASSUMED** ticket ceiling anchored to that reference. No interpolation; epoch-zero crossing is allowed; right-censor at 50. Account for candidate-sweep scorer calls/seconds plus training calls. Status: `AB_TICKET_ONLY`.

  Threshold custody is `run.log` SHA-256 `3860bcf20a341f562e1dd402e281a3298a347f60fa94928cb592ee5dcee480e8`, `launch.sh` SHA-256 `bd760505c445d51dc51d0b31eadd5a4d2628261220ffa46e2474ca83f358c601`, and `levelset_ckpt_stageOctave1_ep251.npz` SHA-256 `c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758`. Its axis is **[macOS-CPU advisory verdict from macOS-MLX training; NON-PROMOTABLE]**. The absent n8 control/treatment logs are explicitly named by `experiments/results/fresh_init_n8_fixed_quality_20260712/measurement_blocker.json`; their absence is a blocker, not substitute evidence.
- **Hardness repaired equal-update A/B:** precondition is consuming every `len(order)` visit and asserting one base visit per pair plus exactly `round(P*oversample)` extras with RNG/resume persistence. Uniform extras (`weighted=False`) are control and hardness extras (`weighted=True`) treatment, both `oversample=0.5` (**existing DSL default / ASSUMED policy, not a measured optimum**), `source=realized`, same seed and exact updates. First emitted `d_seg <= 0.040915` is a **MEASURED advisory** reference; 25 nominal epochs is an **ASSUMED** bounded-ticket ceiling. Right-censor at window. Status: `WIRING_NEEDED`. The current negative is narrow: it does not rule out either this repaired formulation or a separately pre-registered fixed-budget replacement-resampling formulation.

  The epoch-251 weights-only start is custodied by `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_ckpt_stageOctave1_ep251.npz`; the same coherent `run.log` and review-time SHA above custody the reference, not an achieved hardness result.
- **TerminalSolve full-P build A/B:** its frozen A/B premise start is `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`, SHA-256 `6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca`; the #341 probe record is `reports/basin_finisher_probe_20260707.json`, SHA-256 `7515cfe7495526e0dcae656477dc2718180d71f77447e69c23159250ca1afbb2`. This is a premise/start only and remains build-gated. Treatment needs typed default-OFF in-trainer full-P HVP/CG, atomic pre/post checkpoints, solver/resume state, and accept/rollback ledger. Each arm records `solver_hvp_steps` separately from optimizer updates; the treatment permits one damped-GN/CG LM proposal, at most 16 CG steps (ceiling inherited from the **MEASURED** #341 probe, not a promised optimum), then exact n600 acceptance. `0.98` is an **ASSUMED** preregistration policy constant; after a common n600 realized-through-R `d_seg_start` is measured, the numeric target `0.98*d_seg_start` is **DERIVED**. The across-seed/noise floor is **UNMEASURED**. Control's 250 epochs is an **ASSUMED** ticket ceiling. Status: `WIRING_NEEDED`.

## Receipt authority and speed equality gate

Every A/B ticket requires two typed, nonempty receipt-custody rows before it can be `MEASURED` or `MEASURED_CENSORED`: `speed_configuration_custody` and `measurement_authority_custody`. The first proves both arms used an identical exact, machine-readable speed configuration with every currently admitted/requested neutral fleet speed lever ON and `all_requested_speed_levers_on=true`; any absent, unadmitted, or OFF lever is a blocker, not a compliant window. The second proves deterministic NumPy-fp32 realization through actual `R` and frozen CPU-torch scorer execution on all 600 states. MLX training remains advisory and carries no score authority. The sibling current-wall receipt has `all_requested_speed_levers_on=false`, so it cannot supply this gate or an admitted requested composition.

## Time-factor boundary

The measured CPU one-thread `2.995x` result is a scorer-forward subcomponent factor, not a measured whole-step multiplier. The reviewed tile-halo receipt has measured n600 boundary coverage but a **DERIVED ideal exact speedup upper bound of 1.0** scoped to the frozen B2 U-Net finite input-crop tile-with-halo formulation. It provides no whole-step wall split and is not composed into any ticket. The current-wall receipt **DERIVES** 295.352 seconds/epoch from measured n600 log timestamps on the observed training critical path and records zero measured async verdict wait, but explicitly says `composition_admissible=false`, `all_requested_speed_levers_on=false`, and leaves the full training residual unallocated. Thus `wall_fraction=None` and `wall_fraction_saved=None`: no residual is assigned to `t_update`, and the whole-step factor remains **UNKNOWN**.

## Triality and next gate

- **DSL:** existing surfaces are audited only; no duplicate DSL class or invented argv was created.
- **DAG:** `.omx/research/sub015_DAG_steps_dimension_95kill_20260713.md` holds the source-audit → receipt → accounting → decision chain.
- **Equations:** `tac.canonical_equations.steps_dimension_epochs_to_target_20260713` provides an unanchored, `ASSUMED_AWAITING_VERIFICATION` accounting law. **Registry integration is BLOCKED**: no population helper exists in this lane, and a future canonical importer must verify durable receipt schema, n600 cohort, epoch-0 history, matched config/source/checkpoint hashes, target/censor/init receipt hashes, and wiring-closure evidence before registration.

`tools/triality_drift_detector.py` has no narrow-path mode and operates on committed windows while updating shared marker state. This new-file-only lane leaves shared state untouched, so main must run its global committed-window check during integration; no unrelated sibling drift is assigned here.

The next action is a governed cold n600 FreSh A/B only after the normal lane/launch authority is supplied. Hardness and TerminalSolve must first close their listed wiring gates. Local MLX/CPU rows remain advisory and cannot move the score pointer.
