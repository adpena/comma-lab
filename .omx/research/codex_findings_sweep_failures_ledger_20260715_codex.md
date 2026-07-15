# Sweep Arm C — failure, deferral, and duty-to-measure recursive drain

`research_only=true` · `$0` · `NO_TRAINING` · `NO_DISPATCH` · `NO_SCORE_CLAIM`

Pointer `0.1910828242 [contest-CPU Linux x86_64]` and borrowed defensive bank `0.1880443980`
are UNMOVED. The current vehicle is V9 CGauge. Negative findings below are implementation/instance scoped;
no formulation family is killed.

## Stores consulted

- `docs/operating_manual_craft_handoff.md`, relevant NO-FAKE/serializer/forbidden-premature-KILL
  contracts in `CLAUDE.md`, and the worktree `AGENTS.md` contract.
- `.omx/state/harness_failure_ledger.jsonl` through all 37 pre-audit raw rows and the canonical
  `tac.harness_failure_ledger` writer/loader.
- `.omx/state/deferral_ledger.md` all 57 row identities (duplicate D21-D26 disambiguated as `a/b`).
- `.omx/research/default_off_decision_table_20260710.jsonl` and
  `.omx/research/witness_train_sweep_spec_20260714.md` (SHA-256
  `e2362d353426d4ef5cb2a77adf2a4aafaac75fa330050d27d2c829c1eed1976b`).
- Current `lever_registry.completeness()`: `399` trainer flags, `303` DSL-referenced/mapped,
  `96` unmapped, `stale=[]`.
- Live inbox/broadcast through `2026-07-14T20:32:37Z`; binding updates consumed: V9 is the sole
  vehicle, provenance-owned DSL surfaces are exclusive, and negative verdicts stay narrowly scoped.

## Harness failure ledger — six-row drain

| Failure | Terminal disposition | Real verification / route |
|---|---|---|
| `false_dead_diagnosis_incomplete_process_tree_walk` | **FIXED-WITH-REPRO / gate-landed** | `tools/process_tree_liveness.py` now walks the root-custodied PPID tree, labels token-only fallback as non-root custody, and never converts one absent sample into `DEAD`. Live self-tree repro returned `TREE_PRESENT`; regression passed. Commit `279d801b09`. |
| `review_gate_override_on_py_commit_20260711` | **FIXED-WITH-REPRO / gate-landed** | Serializer refuses Python under `REVIEW_GATE_OVERRIDE=1`, including `--no-stage` staged-index targets. Explicit CLI repro returned `rc=12` before staging; normal reviewed Python serializer commit then succeeded. Commit `279d801b09`. |
| `codex_workspace_write_sandbox_blocks_git_objects_20260712` | **RESOLVED-CITED / gate-landed** | Dispatcher commit `e0af3ebd7e` makes isolated danger-full-access worktrees the default; this arm's serializer commit `279d801b09` proves git-object custody. Legacy harvest fallback remains `401f26002a`. Explicitly forced workspace-write remains externally constrained by design. |
| `daemon_5min_harness_long_call_sweep_kill` | **STILL-OPEN-ROUTED / worked-around** | Commit `91b4b5db1` records the external lifetime wall. Owner governed-launch operator; terminal action is foreground chunking with disk checkpoint per chunk, or unsandboxed launch plus `ppid==1` proof. No daemon launched here. |
| `zsh_nomatch_glob_aborts_monitor_scripts` | **STILL-OPEN-ROUTED / worked-around** | Current tracked monitor scan found no cited vulnerable `*stage*.npz` shell pattern; Python monitors use empty-safe `Path.glob`. New tracked watchers must use `find(1)` or `setopt null_glob`; owner must add a shell-lint refusal fixture for future scripts. Ad-hoc interactive zsh cannot be statically intercepted. |
| `codex_probe_token_limit_death_incomplete_wip_20260712` | **STILL-OPEN-ROUTED / worked-around** | Original token-limit diagnosis canonically falsified; measured cause is service capacity beyond the bounded retry tail. `tools/tests/test_codex_delegate_retry.py` passed `3/3`. Owner dispatcher/service recovery; re-run original ticket from checkpoint only after capacity returns, never salvage broken WIP. |

Original-six counts: `FIXED-with-repro=2`, `RESOLVED-cited=1`, `STILL-OPEN-routed=3`,
`NEEDS-INVESTIGATION=0`. Three malformed noncanonical raw rows remain historical bytes but are now
superseded by valid canonical events; no valid failure id `?` exists. Current canonical states for the six
are `gate-landed=3`, `worked-around=3`, `open=0`.

### Surfaced bug class

The mandatory review-index rebuild itself failed on two top-level functions with the same name in one
module. The old self-test only warned, so the empty tracker could not review anything. Commit `279d801b09`
adds deterministic `@L<line>` qualified-name discriminators only for repeated definitions and a regression.
A clean rebuild then completed: `128,397` entities / `8,403` files. This is a class fix plus self-protect,
not merely a workaround.

## Deferral ledger — 57-row drain

The complete per-row overlay is appended to `.omx/state/deferral_ledger.md`; no row is silently dropped.

- `13` are closed/verified/cited or explicitly superseded for the old vehicle.
- `7` have triggers met now and are owner-routed with complete build or measurement gates.
- `37` are not-met/run-gated/identity-blocked and retain exact named triggers and owners.
- `8` stale triggers were explicitly corrected to current V9 custody.
- Total: `13 + 7 + 37 = 57`.

Hot-row results:

| Row | Result |
|---|---|
| D1 | Not met: governed n600 GPU/CPU probe has no data. Exact resumable command preserved in its memo. |
| D2 | Re-pointed to first clean converged V9 C0 checkpoint; no C0 exists. |
| D18 | Not met: final V9 checkpoint/k90 telemetry absent; byte-close PCA machinery exists. |
| D21a | Trigger met, route receiver owner: blind-coordinate module/proof exists but the canonical levelset byte-close receiver does not consume it. |
| D21b | Closed measured: ep700 ON `+3.2e-05 d_seg`, marginally worse; bounded-warm-start scope only. |
| D22a | Closed verified: exact raw cardinality refusal exists before scoring in both byte-close tools. |
| D24a | Trigger met, scorer-geometry owner: required radius/block-Jacobian tail receipt is still absent. |
| D25a | Closed for V9: current V9 compile imports/emits AMBER stability. Historical-pilot semantics are not rewritten. |
| D26b | Closed by `279d801b09` process-tree gate. |
| D27b | Not met: no custodied `d27b_ready=true`; exact audit command and slope/Muon predicate retained. |
| D41 | Latest directive supersedes headroom-only status: native-width apparatus is built/reviewed but V9 integration and Metal receipt are owed to the exclusive provenance owner. |

## Duty-to-measure drain and fire tickets

The authoritative V9 campaign spec reports `82` known levers, `80` never-fired, and `81` owed at its
2026-07-14 snapshot. It already gives a per-row owner/gate for the full queue and a ranked seven-treatment
campaign. This pass does not duplicate that design; it verifies whether the top treatments are executable
through the current sole DSL path.

### Top-three tickets

#### TAPER-ISO — rank 1, 78.9%

- Exact scientific delta: fresh `C0` versus `C0 - DsegAwareTaper`; C0 keeps
  `DsegAwareTaper(strength=1.0, scale=0.0, floor=0.05)`.
- Measurement: matched n600 seed/order/steps, terminal EMA/live/Polyak byte-close; accept only lower exact
  total `S`, with n600 d_seg as a facet rather than authority.
- Current blocker: the factory/trainer flags are mapped, but removing a base lever fails the V9
  expected-active-lever/provenance manifest. Owner: Arm B / exclusive V9 provenance owner.
- Complete build ticket: add canonical config id `v9_cgauge_432_taper_off`, remove exactly the taper Lever
  and its LawRefs from the expected manifest, preserve every other emitted token, and add a one-delta
  compile test. No raw `--dseg-aware-*` edits.
- Fire command after owner receipt and operator GO:
  `.venv/bin/python tools/launch_witness_run.py --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --epochs 3000 --config v9_cgauge_432_taper_off --out-dir /Volumes/VertigoDataTier/pact/v9_cgauge_432_taper_off --label v9_cgauge_432_taper_off --purpose 'matched fresh V9 TAPER-ISO treatment' --accept-wall-clock 10.71`

#### HORIZON-ISO — rank 2, 47.3%

- Exact Lever: `HorizonWeightedMargin(weight=LawRef(HWM_V9_STAGE_SHARE), target=0.5,
  margin_lo=0.3, margin_hi=0.5, row_lo=96, row_hi=288, start_epoch=<typed stage boundary>)`.
- Law: at the frozen eligible C0 boundary, resolve and freeze
  `w_h=(0.15/0.85)*L_o/max(L_h,eps)`; absent measurement keeps the arm HELD.
- Measurement: n600 exact total `S`; surviving flips must shift toward higher GT margin inside rows 96-288.
- Current blocker: seven scientific LawRefs plus the measured weight consumer are absent from the V9
  provenance bijection. `--dsl-lever HorizonWeightedMargin` would silently emit default weight `0.0` and is
  therefore forbidden. Owner: Arm B / exclusive V9 provenance owner.
- Complete build ticket: add `v9_cgauge_432_horizon_iso`, the boundary measurement receipt schema,
  all seven LawRefs, trainer consumer, expected-lever manifest row, and a refusal test for weight `0` or a
  missing receipt.
- Fire command after owner receipt and operator GO:
  `.venv/bin/python tools/launch_witness_run.py --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --epochs 3000 --config v9_cgauge_432_horizon_iso --out-dir /Volumes/VertigoDataTier/pact/v9_cgauge_432_horizon_iso --label v9_cgauge_432_horizon_iso --purpose 'matched V9 HORIZON-ISO treatment' --accept-wall-clock 10.71`

#### STEP-ISO — rank 3, 34.2%

- Exact Lever: `StepNativeActivation(beta_start=1.0, beta_end=8.0, anneal="linear",
  basis="annealed_hosc", omega=1.0, finer_bias_init=False)`; fixed beta and FreSh stacking are forbidden.
- Measurement: fresh matched n600 activation basin; exact total `S`, ring/edge survival, saturation and
  dead-gradient telemetry; any negative is this V9 formulation only.
- Current blocker: factory/trainer flags are mapped, but beta-end `8.0` conflicts with sealed V9 `3.177`.
  Owner: Arm B / exclusive V9 provenance owner.
- Complete build ticket: add `v9_cgauge_432_step_iso` as a distinct scientific declaration with matching
  LawRefs/manifest/consumer receipt and a test proving exactly one activation treatment delta.
- Fire command after owner receipt and operator GO:
  `.venv/bin/python tools/launch_witness_run.py --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --epochs 3000 --config v9_cgauge_432_step_iso --out-dir /Volumes/VertigoDataTier/pact/v9_cgauge_432_step_iso --label v9_cgauge_432_step_iso --purpose 'matched fresh V9 STEP-ISO treatment' --accept-wall-clock 10.71`

These config ids intentionally do not exist yet: their absence is the measured integration blocker, and
the build tickets specify the sole legal creation path. Each real launch must first claim its lane with
`tools/claim_lane_dispatch.py claim`, pass SSD storage/governor/timing/resume/provenance gates, and receive
explicit operator GO. This arm received no GO and launched nothing.

### Remaining ranked high-value tickets

The existing V9 spec routes `AA-SUPER2`, `ETF-HEAD`, `POLAR-FINISH`, and conditional `HORIZONxSTEP` with
exact pseudo-DSL, measurement/falsifier gates, and envelopes. They remain held because V9 scientific
bindings are missing; `HORIZONxSTEP` additionally requires both isolated arms to have exact `B>0`.
`HardnessOversample` is correctly excluded: the trainer truncates its enlarged order to the original P
visits, a live wiring gap owned by the trainer/Arm-B integration lane.

Queue-wide wiring verdict:

- Factory-to-trainer map: `303/399`, `96` unmapped, `0` stale.
- The top three are trainer-mapped but **not V9-fireable** due provenance/variant gaps above.
- The other `78` owed rows retain disposition/owner/gate in the machine-readable decision table; none is
  promoted merely because it is registered. Alias-false “never-fired” rows remain attribution work, not
  new training tickets.

## Verification receipt

- Harness/review apparatus: `30 passed`; explicit override repro `rc=12`; live process-tree
  `TREE_PRESENT`; clean review scan `128,397` entities / `8,403` files.
- Top-lever factories/readiness: `82 passed` across taper, Horizon, Step, and launch-readiness tests.
- Codex capacity retry: `3 passed`.
- Ruff on all changed Python: clean under the repository's pre-existing serializer exceptions.
- `git diff --check`: clean.
- No trainer, scorer, evaluator, daemon, provider, GPU, or paid dispatch executed.

## Main merge review

Review commit `279d801b09` for the two original harness fixes plus the surfaced review-index class fix;
review the canonical append events in `harness_failure_ledger.jsonl`; and apply the ignored operational
`deferral_ledger.md` overlay from this worktree. The V9 provenance owner, not this arm, owns all named
`src/tac/witness_dsl` follow-ons.
