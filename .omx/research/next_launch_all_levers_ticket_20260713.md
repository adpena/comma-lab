# Next-launch all-compatible-levers witness ticket — 2026-07-13

**STATUS: HELD / NO LAUNCH / OPERATOR-GO REQUIRED / POINTER DELTA ZERO.**

Ticket: `experiments/results/next_launch_all_levers_ticket_20260713/`  
Typed program: `next_launch_all_levers_20260713`  
Composition: **28 IN / 7 EXCLUDED**, of which **21 are active typed-DSL levers**.  
Full-stack projected wall-clock/epoch: **REFUSED / UNKNOWN (`null`)**. The only current whole-epoch
anchor is **MEASURED 295.352 s/epoch** on the audited micro-batch-OFF run; its component allocation is
100% unmeasured and therefore cannot be multiplied by isolated speedups.  
Preflight verdict: typed compile/validation, real-argparse audit, schedule provenance, safe-region
fingerprint, per-run memory projection, and readiness gate pass; the canonical B=2 memory waterfill,
system governor, storage placement, and five dependency blockers refuse. No trainer process was spawned.

The eventual admitted run is designed to measure three previously missing surfaces in one trajectory:

1. **D-A** — exact teacher forward/backward, witness forward/backward, realized-through-R, verdict,
   checkpoint-I/O, and epoch-total wall time (`witness_component_wallclock.v1`).
2. **D-B** — SPS gradient-role cosine/norm/conflict rows at temporal-screw engagement (event, ep450
   fail-safe cap) and phase-advection engagement (ep726), using the math in
   `tools/probe_sps_gradient_role_conflict.py` (`sps_gradient_role_conflict_engagement.v1`).
3. **Causal rows** — the sibling's actual `pact.causal_manifest.v1` append-only
   `causal_manifest.jsonl` stream. The shared trainer contains a default-ON, read-only writer with no
   new launch flag, and the late-arriving sibling DAG FEED explicitly confirms that this apparatus is
   `N/A-with-reason` for the DSL. That dependency is therefore consumed without inventing a flag.

## Verdict and scope

This is the strongest **currently compatible typed composition**, not a claim that every requested
lever is admissible today. The generated `launch.sh` is a held preview. The governed launcher refuses a
real spawn while `dsl_program_manifest.json::launch_blockers` is non-empty; there is no override route.

`verdict_scope = launch-ticket instance at current worktree and current host pressure.` The refusal does
not kill MicroBatch, component timing, SPS conflict observation, causal manifests, or the V9 family. It
says this exact B=2/all-debt ticket lacks the required n600 memory measurement and exact in-run D-A/D-B
producers and has not passed storage/system admission.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`
- `PROGRAM.md`, `docs/vehicle_operating_system.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/t5_crucible2/SPEC_v752_20260709.md`
- `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`
- `.omx/research/sub015_DAG_sps_gradient_separation_20260713.md`
- `.omx/research/sub015_DAG_cheapen_real95_tilehalo_fp16_20260713.md`
- `experiments/results/cheapen_real95_tilehalo_fp16_20260713/current_wall_receipt.json`
- `src/tac/witness_dsl/spec_v9_cgauge.py`, `src/tac/witness_autoconfig.py`,
  `src/tac/witness_dsl/curriculum_dsl.py`, `src/tac/witness_dsl/lever_registry.py`
- `experiments/train_levelset_witness_realized_through_R_mlx.py` real argparse and current observer paths
- `tools/probe_sps_gradient_role_conflict.py`, `tools/memory_waterfill_config.py`,
  `tools/plan_experiment_storage.py`, `tools/witness_launch_readiness_gate.py`,
  `tools/launch_witness_run.py`
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`,
  `.omx/state/lever_relative_significance.jsonl`, `.omx/state/probe_outcomes.jsonl`
- `src/tac/causal_manifest.py`, its live shared-trainer import/writer surface, and the late-arriving
  `.omx/research/causal_manifest_DAG_FEED_20260713.md`. The separately named
  `.omx/research/causal_manifest_build_20260713.md` was not present, but the sibling DAG FEED explicitly
  closes the launch-composer question: default-ON score-neutral apparatus, no new trainer/DSL flag.

Repo HEAD during composition: `2dce691facb75e8540a91981f24bf8c3800d2a4f`. The worktree contains
sibling/user-owned uncommitted work; this lane leaves its own ticket uncommitted as directed and does not
absorb or revert sibling edits.

## Sealed lineage and typed composition

The compiler calls `compile_v9_cgauge_ideal_mod19_launch_config(...)`; that compiler owns the V9·CGauge
#432 → v7.5.2 → V9 ideal-mod19 lineage and sealed schedule derivation. The ticket then removes the one
positive-weight tie-locus treatment that the real trainer refuses under B>1 and composes six typed
additions: `MicroBatch(2)`, `FusedRKernel()`, `CacheGtSkeleton()`,
`SafeCompileRegions("hosc_activation")`, `FreshFrequencyShift()`, and the existing real-parser observer
bundle. No argv was hand-authored. `WitnessProgram.compile_trainer_argv()` emitted all trainer arguments.

The compiled ticket has 21 active typed levers and 231 emitted trainer flags. The launcher independently
matched all 21 expected lever names and all 231 flag names against the real parser. Required performance
environment is structurally declared as:

```text
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1
TAC_MLX_CUSTOM_PERSISTENCE_POOL=1
```

One-thread standard behavior remains trainer-owned, not an invented flag.

## Per-lever disposition

The machine-readable authority is `include_exclude_table.json`. This table is the compact human audit.

| Lever/surface | Disposition | Provenance or reason |
|---|---:|---|
| `seg_form_unify_tau` | IN | V9 ideal typed lineage; continuous tau form. |
| `tail_k_warm_restart` | IN | V9 ideal typed lineage; rung-persisted tail cycles. |
| `n323_ladder_island_homotopy` | IN | Typed, realized-through-R island-birth continuation. |
| `R7_polyak_finisher` | IN | Typed terminal candidate; does not replace EMA. |
| `v75_area_constraint_birth` | IN | v7.5 area-Lagrange counter-force. |
| `v75_birth_completion_event` | IN | v7.5 Morse-Smale birth-completion event. |
| `n287_dash_comb` | IN | Typed ego-phase homogenization corrector. |
| `temporal_screw_consistency` | IN | Typed Force-1; engagement event with ep450 cap. |
| `pose_finish_conditioning_gate` | IN | Typed sigma-min gate with banked fallback. |
| `phase_advection_consistency` | IN | Typed T1 phase force; ep726 static anchor in this lineage. |
| `unified_tau_eikonal_hold` | IN | Typed rung-coupled retention. |
| `n292_closed_loop_eikonal_control` | IN | Typed bounded d_seg-trend controller. |
| `R7_beta2_window_rewarmup` | IN | Typed derived 14-epoch beta2 memory window. |
| `FEED_08a_length_sigma` | IN | Typed class-pair interface length weighting. |
| `margin_band_satisficing` | IN | Typed one-sided realized-margin hinge. |
| `micro_batch_pairs` | IN, BLOCKED | B=2 typed; operator training-only drift waiver exists, but n600 full-step RSS/functional receipt is still owed. |
| `n252_fused_r_kernel` | IN | #348/L70 bit-exact forward and approximately 1-ULP VJP; score-neutral speed surface. |
| `cache_gt_skeleton` | IN | Bit-identical constant-GT skeleton cache. |
| `n252_safe_compile_regions` | IN | Only fingerprint-certified `hosc_activation`; no whole-step compile claim. |
| `fresh_frequency_shift_init` | IN | #448 cold-start treatment; 94 scorer-pair-equivalent selection budget, not a claimed epoch speedup. |
| `next_launch_observer_telemetry` | IN, PARTIAL | Existing coarse timing, loss-term, and generic K=4 gradient rows; exact D-A/D-B hooks remain blockers. |
| one-thread standard | IN | MEASURED 2.995x frozen CPU SegNet-forward subcomponent; not promoted to whole-step. |
| grouped backward | IN | Required perf env; fixed-order custom VJP path. |
| persistence pool | IN | Required perf env; real-n600 full-loss parity. |
| async verdict | IN | Existing sealed flag; prior measured critical-path wait was zero. |
| `DsegAwareTaper` | IN, TELEMETRY | Duty rank 1; record trajectory, but this run is not an isolated taper A/B. |
| `latent_table_truncate_d18_k90` | IN, TERMINAL A/B | Exact stop-time byte-close slot, not a training flag. |
| `mod32_neutrality_19_ab` | IN, TERMINAL A/B | Exact matched terminal bytes, not a training flag. |
| `tie_locus_displacement` | EXCLUDED | Proven real-trainer incompatibility: positive subpixel-boundary weight is refused with B>1. |
| whole-step megakernel | EXCLUDED | MEASURED formulation NO-GO from floating-point reordering; only certified HOSC region is admitted. |
| hardness oversampling | EXCLUDED | Repair not landed/certified: current enlarged order is truncated to P consumed visits. |
| `HorizonWeightedMargin` | EXCLUDED | Must be an isolated exact-V9 A/B; no matched treatment custody. |
| `StepNativeActivation` | EXCLUDED | Must be isolated; stacking confounds the FreSh cold-start/activation basin. |
| Muon round-2 FiLM polar-chart SPEL | EXCLUDED | Sibling owns wiring; default-OFF until governed micro-A/B and operator GO. |
| FreSh fixed-quality slice | EXCLUDED | n8/n64 treatment protocol, not a per-epoch verdict/checkpoint policy for the n600 production window. |

### Registry completeness cross-check

Current `lever_registry.completeness()` is **385 trainer flags / 291 DSL-referenced / 285 mapped /
100 unmapped / 0 stale**, mapped coverage **0.7402597403**. Completeness is an inventory check, not
permission to invent the 100 missing routes. Every flag emitted by this ticket independently exists in
the real trainer parser.

## Costate duty-to-measure top five

| Rank | Duty row | Current duty | Evidence/DeltaS | This run |
|---:|---|---:|---|---|
| 1 | `DsegAwareTaper` | 78.9% | ESTIMATED 0.03; fired-unmeasured | **IN as trajectory telemetry, not a causal A/B.** Its global-taper negative is formulation-scoped; do not call this run a taper treatment. |
| 2 | `HorizonWeightedMargin` | 47.3% | MEASURED oracle ceiling 0.018; never fired | **EXCLUDED.** Needs an isolated exact-V9 warm-start A/B with support custody. |
| 3 | `StepNativeActivation` | 34.2% | MEASURED screen 0.013; never fired | **EXCLUDED.** Activation-basin treatment would confound FreSh. |
| 4 | `latent_table_truncate_d18_k90` | 2.6% | ESTIMATED 0.001 | **IN as terminal exact byte-close A/B**, not a training flag. |
| 5 | `mod32_neutrality_19_ab` | 1.3% | ESTIMATED 0.0005 | **IN as terminal matched exact byte A/B**, not a training flag. |

This satisfies the curriculum-pool discipline: every current top-five duty row is either given a named
measurement surface or an explicit isolation reason. It does not relabel ESTIMATED values as MEASURED.

## Telemetry contract

### D-A — exact component wall clock

Required default-ON, score-neutral schema: `witness_component_wallclock.v1`.

```text
teacher_forward_s
teacher_backward_s
witness_forward_s
witness_backward_s
realized_R_s
verdict_s
checkpoint_io_s
epoch_total_s
```

The existing `--profile-timing` route is included because it already yields useful fused
step/verdict/overhead rows and an isolated-R microbench. It **does not** yield this decomposition. Until an
owned trainer hook produces the exact fields without changing the update path, D-A remains a launch
blocker. The historical “95/5” is not inserted into any field.

### D-B — SPS engagement conflict

Required default-ON, score-neutral schema: `sps_gradient_role_conflict_engagement.v1`.
At each named boundary, select the same cheap four-stratum sample count (`K=4`) and emit per-role gradient
norms, pairwise cosine, and the conflict predicate from `tools/probe_sps_gradient_role_conflict.py`:

- `temporal_screw_engaged`: actual event boundary, with ep450 only as the governed fail-safe cap.
- `phase_advection_engaged`: ep726 static terminal-band anchor for this compiled lineage.

The existing `--grad-interaction-telemetry --grad-interaction-k-pairs 4
--grad-interaction-every 0` is included. It emits the generic loss-term matrix at seg-form stage
boundaries; screw and phase engagement are not seg-form boundaries. Exact engagement callback wiring is
therefore still a blocker, not silently approximated.

### Causal transition rows and standing telemetry

The actual sibling schema is `pact.causal_manifest.v1`, output `causal_manifest.jsonl`. Shared-trainer
inspection finds it default-ON and read-only, with run-manifest, boundary, and ordered transition rows.
The sibling DAG FEED confirms that adding a flag would create an orphanable off-state and parallel config
vocabulary. Accordingly there is no causal launch flag to add: the dependency slot is **READY** through
the default-on writer. Its aggregate transition rows intentionally do not claim pair-level HCM or FORE
identification. Existing liveness, per-epoch loss-term, verdict, and stage/checkpoint rows remain on.

## Zero-dollar preflight chain

| Gate | Result | Evidence |
|---|---|---|
| Typed DSL compile | **PASS** | 21 active levers; `WitnessProgram.validate()` 0 violations. |
| Real argparse audit | **PASS** | 231/231 emitted flags exist in the current trainer parser. |
| Expected-active manifest | **PASS** | 21/21 names match. |
| Schedule/value provenance | **PASS** | lane/chroma/screw caps ep500/450/450; phase/Muon/pose ep726; Polyak ep2546; event companions present. |
| Perf-env structural gate | **PASS** | grouped-backward and persistence-pool values exactly `1`. |
| Safe-region fingerprint | **PASS** | `hosc_activation` manifest/fingerprint valid on this host. |
| Readiness/config freshness | **PROCEED** | Horizon and StepNative are explicitly deferred with non-empty reasons; two fire-now rungs included. |
| Per-run memory projection | **PASS, NON-ADMISSION** | projected 71.54 GiB; adjusted 77.84 GiB below 89.6 GiB safe ceiling (70% of 128 GiB). |
| Canonical memory waterfill | **REFUSE B=2** | selects B=1 / verdict batch 16; target-n600 B=2 RSS remains unmeasured. |
| System governor | **REFUSE** | final observed composition projected 119.4 GiB system use versus 97.7 GiB adaptive ceiling; 21.7 GiB over at check time (47.8 GiB current + 71.54 GiB new projection). |
| Storage waterfall | **REFUSE** | requested 1,026,048,000 B; Vertigo has approximately 827 GB free but the authorized workload root is missing; APDataStore absent; local disabled. |
| Ticket dependency gate | **REFUSE rc=11** | four typed blockers listed below; launcher stopped before spawn. |
| Launch | **NOT ATTEMPTED** | dry-run/materialization only; no daemon, GPU job, paid dispatch, or run-dir mutation. |

Four current typed blockers:

1. `D_A_EXACT_COMPONENT_TIMERS_MISSING`
2. `D_B_EXACT_ENGAGEMENT_HOOK_MISSING`
3. `MEMORY_WATERFILL_B2_UNMEASURED_N600`
4. `SSD_WORKLOAD_ROOT_MISSING`

The causal dependency changed during composition. The trainer writer appeared first; the sibling DAG
FEED then landed and confirmed `pact.causal_manifest.v1`, `causal_manifest.jsonl`, default-ON read-only
logging, and no DSL flag. The final recompile consumes that contract and removes the former causal blocker.

## Wall-clock projection: deliberately null

`projected_sec_per_epoch_full_stack = null`.

The audited receipt has `total_training_critical_path_s_per_epoch = 295.352`,
`unallocated_training_critical_path_s_per_epoch = 295.352`, `unallocated_fraction = 1.0`, and
`composition_admissible = false`. Its micro-batch was OFF. The async CPU-torch verdict service measured
2238.3 s/call and 89.532 s/epoch amortized, but measured critical wait was zero; that is not a license to
subtract 89.532 from the epoch. Isolated K=8 scorer anchors of 1.56x GPU / 1.75x CPU are not whole-step
speedups. FreSh is an initialization treatment and has no measured epoch multiplier. Any numeric
full-stack projection would therefore be a guessed product, forbidden by the value-provenance ladder.

The future run itself pays D-A; only then may `teacher + witness + R + verdict + checkpoint_IO` be
reported as a MEASURED wall allocation and used to project later launches.

## Operator GO brief

**Do not execute the current `launch.sh`.** Before requesting GO:

1. Land and independently verify the exact D-A component-timer producer and schema.
2. Land and verify the exact D-B screw/phase engagement callback using the probe's existing math.
3. Run the cheapest governed real-Metal B=1/B=2 full-step n600 calibration that records functional
   parity, median step time, and peak RSS; feed its receipt to `memory_waterfill_config.py`.
4. Keep the consumed causal-manifest default-on contract fresh; if the separately named build memo later
   changes the surface, read it and recompile rather than inventing a flag. Its focused tests currently pass.
5. Create/authorize the Vertigo workload root through the storage-waterfall path and rerun the storage
   preflight. Do not fall back to local disk.
6. Recompile the typed ticket, require `launch_blockers=[]`, rerun DSL validation, argparse audit,
   memory waterfill, system governor, readiness/config-freshness gate, and storage preflight.
7. Obtain explicit operator GO. Only the governed launcher may spawn.

When admitted, runtime output belongs at
`/Volumes/VertigoDataTier/pact/next_launch_all_levers_20260713`, not the local ticket directory.
Resumption is from the typed `--resume-from` surface and complete atomic checkpoints. The compiled config
keeps stage checkpoints ON, periodic checkpoints at the sealed cadence, distinct stage-encoded files,
EMA custody, optimizer/stage/epoch state, and prior stage files preserved. A restart must lose at most one
periodic interval and must keep the causal treatment/data-order hashes consistent.

## Triality and apparatus delta

- **DSL leg:** `src/tac/witness_dsl/spec_next_launch_all_levers_20260713.py` and the compiled
  `typed_witness_config.json`, `witness_program.json`, `compiled_trainer_argv.json`.
- **DAG leg:** `.omx/research/DAG_next_launch_all_levers_ticket_20260713.md`.
- **Equations leg:** sealed v7.5/V9 constant manifests plus named D-A sum and SPS cosine dependency slots;
  no new equation is minted from an unmeasured 95/5 premise.
- **Consumer leg:** governed launcher recognizes the named typed config and refuses a non-dry spawn while
  dependencies are unresolved.

Pointer delta: **0**. Score claim: **none**. Measurement axes: preflight/config only plus cited
`[macOS-MLX research-signal]` / `[macOS-CPU advisory]` anchors; no contest-CPU/CUDA inference.

## Artifacts

- `experiments/results/next_launch_all_levers_ticket_20260713/launch.sh`
- `experiments/results/next_launch_all_levers_ticket_20260713/typed_witness_config.json`
- `experiments/results/next_launch_all_levers_ticket_20260713/witness_program.json`
- `experiments/results/next_launch_all_levers_ticket_20260713/compiled_trainer_argv.json`
- `experiments/results/next_launch_all_levers_ticket_20260713/constants_manifest.json`
- `experiments/results/next_launch_all_levers_ticket_20260713/dsl_program_manifest.json`
- `experiments/results/next_launch_all_levers_ticket_20260713/include_exclude_table.json`
- `experiments/results/next_launch_all_levers_ticket_20260713/ticket_compile_receipt.json`
- `experiments/results/next_launch_all_levers_ticket_20260713/storage_preflight.json`
- `experiments/results/next_launch_all_levers_ticket_20260713/preflight_summary.json`

All are intentionally uncommitted. No launch occurred.
