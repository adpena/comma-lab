# ddm_rv8 2026-08-06 Receipt

Verdict: `BLOCKED_AXIS9_ENVIRONMENT`; counter state `0/3`.

This round does not advance to clean-pass 2/3. Static review axes found no code/ticket
blocker in the la1 A/B surface, but the required measured-runnability smoke could not
execute in this headless Codex sandbox because MLX fails at `import mlx.nn` with no
Metal device available.

Own-vehicle frontier line remains:
`S = 0.7537933983374265 @ 357,837 B [macOS-CPU advisory]`; borrowed contest pointer
`0.19108 [contest-CPU]` unmoved.

## RECALL EVIDENCE

Searches run before verdict:

- Charter/common/governance: `.omx/tmp/codex_runs/rv8_prompt.md`,
  `.omx/tmp/codex_runs/_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`,
  `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Memory quick pass: `rg -n "rv8|v8|perclass|per-class|common contract|restart|rv"`
  in `/Users/adpena/.codex/memories/MEMORY.md`.
- Full-corpus recall query over `.omx/research`, `.omx/state`, docs, configs, src,
  tools, experiments for `la1|jd1-lr-anneal|derived_tail|bd1|realized_gate_dpose|
  emit-error-atlas|wp1|jd1-finisher|jd6|jd6_endpoint|jd7on|jd7off`.
- Canonical equations registry: `.venv/bin/python tools/list_canonical_equations.py --json`.
- Arm final messages: `la1_20260806T020517Z.md`, `bd1_20260806T035657Z.md`,
  `wp1_20260806T023856Z.md`, `ddm_jd4_20260805T203409Z.md`.

Found beyond charter seeds:

- Canonical equation `jd1_plateau_tail_average_ema_v1`: confirms JD1 tail-average EMA
  is registered, but this A/B checkpoint is still geometric EMA; no plan change.
- `bd1_20260806T035657Z.md` already reported its real gate36 timing attempt blocked
  by unavailable Metal in this sandbox. This changed the rv8 plan by treating the
  smoke as a likely environment gate and by preserving the exact import failure.
- `main_hot_state.md` had rebased the own-vehicle frontier from `d5e814d5` to
  `75df9cc3 @ 357,837 B`; final frontier line uses the live board, not the common
  contract's older 357,836 B line.

## Findings

| Axis | Severity | File/line | Result |
|---|---:|---|---|
| 9 measured-runnability | CRITICAL | `experiments/train_tr1_partition_renderer_mlx.py:4175` import path | Required smoke did not start: `RuntimeError [metal::load_device] No Metal device available` at `import mlx.nn as nn`. No anneal telemetry, no first `a1_gate`, no dpose row, no finite-loss row, and no valid RSS sample were measured. Fire remains blocked until the same six-epoch ON smoke runs on a Metal-capable host. |

No CRITICAL/MED/LOW implementation finding was found in axes 1-8 within the reviewed scope.

## Axis 1 - A/B Ticket

Tickets:

- ON: `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd7on_ticket_cont_ep1766_la1on.json`,
  ticket hash `ff17df2fad09989683339d9aa357e079fb45ab19128c592d5befd626e2cdc077`.
- OFF: `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd7off_ticket_cont_ep1766_la1off.json`,
  ticket hash `01f9eee8773251bb4c5df6c14cd66527d1958133807aa4d6ef2aed91fa8d91f4`.

Measured/derived checks:

- Argv diff is only `--out-dir` and ON-only `--jd1-lr-anneal derived_tail`.
- Both resume from the same checkpoint:
  `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1646/checkpoints/stage_joint_pose_finish_final.npz`.
- Resume checkpoint exists, `29,931,437` bytes, `meta::epoch = 1766`,
  `telemetry_tail` last epoch `1765`, stage `joint_pose_finish`.
- Ticket epoch geometry: `1886 - 1766 = 120` window epochs; `600 // 4 = 150`
  steps/epoch; `ceil(55 s/epoch * 120 / 60 * 1.5) = 165` wall minutes.
- Ticket metadata diffs beyond argv are expected derived identities only:
  child dir, next resume template under child dir, `regenerated_from.lr_anneal_arm`,
  and ticket hash.
- Regenerator OFF branch refuses inherited anneal flags at
  `experiments/ddm_jd1_ticket_regenerate.py:344-350`.

EMA inheritance:

- Ticket argv carries `--ema-decay 0.999960019990005` from the parent config.
- The resumed JD1 active EMA is preserved from checkpoint metadata as
  `active_ema_decay = 0.9997777777777778`; code restores saved JD1 EMA metadata at
  `experiments/train_tr1_partition_renderer_mlx.py:4771-4775`.
- The regenerator records `derived_stage_ema_decay = 0.9997777777777778`,
  `parent_stage_ema_u = new_window_u = 18000`, and `force_ema_reanchor_on_resume = false`;
  no hidden EMA reanchor is introduced between arms.

Current-boundary la1 derivation from actual ep1646 parent telemetry:

- `active_ema_decay = 0.9997777777777778`, `beta2 = 0.999`, `steps_per_epoch = 150`.
- `beta2_memory_epochs_c2 = 14`; `active_ema_memory_epochs_c2 = 60`; `tail_epochs = 60`;
  `onset_epoch = 1826` for the ep1766->1886 window.
- Signal source: `epoch.ep_loss[jd1_pose_finish_active]` from
  `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1646/telemetry.jsonl`.
- Oscillation: `n=60`, mean `0.8585571848816342`, sd `0.011725983822340868`,
  half_range `0.023408392667770395`, sign changes `36/58`.
- Derived final fraction `0.3337467458869293`; final LR `0.0006674934917738586`.
- Code path matches the receipt rule: `derive_jd1_lr_tail_schedule()` computes the
  max beta2/EMA tail window and `sd/(sd+half_range)` at
  `experiments/train_tr1_partition_renderer_mlx.py:3735-3815`; `_resolve_jd1_lr_anneal_schedule()`
  logs the typed row only for `derived_tail` at lines `5390-5414`.

## Axes 2-4 - Landed Mechanisms

bd1 `c478dd1712`:

- `realized_gate_dposes()` renders through `_apply_R`, calls `pose_adapter.posenet`,
  materializes pose, and compares first-6 PoseNet outputs to `gt_poses`; no RNG, no
  optimizer, no value/grad call, no `.update(` appears in the helper. Source lines:
  `experiments/train_tr1_partition_renderer_mlx.py:1944-1963`.
- Dpose telemetry fields are included in `BS3_TELEMETRY_ONLY_KEYS` and stripped before
  `telemetry_tail`; relevant source lines `1758-1779`, `1783-1795`, `5981`, `6133`.
- Default-on can affect telemetry timing but not optimizer/train state by the reviewed
  source path. Runtime gate36 wall-clock delta remains unmeasured because axis 9 did not start.

bd1 `b7056a8ca7`:

- `--emit-error-atlas` parser default is false at
  `experiments/ddm_jd4_endpoint_n600_both_bases.py:97-99`.
- The receipt only attaches `error_atlas_manifest` under the flag at lines `233-243`,
  `278-286`, `310-311`.
- Current `jd6_endpoint_n600_both_bases.json` has no `error_atlas_manifest`, confirming
  absent-flag schema stability in the live endpoint artifact.

wp1 `098b98e11c`:

- `--jd1-finisher {off,muon}` default-off parser lines:
  `experiments/train_tr1_partition_renderer_mlx.py:3207-3212`.
- Muon refuses inert/off-pose mode and non-resumed/mid-window use at lines `4059-4068`,
  `4114-4120`, `5656-5662`.
- A/B tickets contain no `--jd1-finisher`; wp1 cannot engage in either la1 A/B arm.
- Real Muon path uses MLX `optim.Muon` plus `optim.MultiOptimizer` at lines `3888-3936`;
  no Muon launch was attempted in rv8.

Regenerator `43607879d1`:

- `--anneal {off,derived_tail}` and `--out-dir-suffix` are present at
  `experiments/ddm_jd1_ticket_regenerate.py:405-411`.
- OFF arm refuses inherited `--jd1-lr-anneal`, so a control ticket cannot silently carry
  the treatment flag.
- `finalize_ticket()` rebuilds lever overrides from final argv and validates declared
  overrides against argv at lines `272-288`.

## Axis 5 - jd6 Arithmetic

Source: `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd6_endpoint_n600_both_bases.json`,
baseline ep1646 constants in `experiments/ddm_jd4_endpoint_n600_both_bases.py:69-79`.

Recomputed from components:

- EMA d_seg delta `-0.0004660458034939233` => seg term `-0.04660458034939233`.
- EMA d_pose delta `-0.025723172606435994` => pose term
  `sqrt(10*0.010094668682475555) - sqrt(10*0.03581784128891155) =
  -0.2807590022435698`.
- EMA net no-rate delta `-0.32736358259296217 S/window`, matching the hot-state claim.
- Live d_seg delta `-0.0004537031385633671` => seg term `-0.04537031385633671`.
- Live d_pose delta `+0.07454459555637416` => pose term `+0.28175860103130934`;
  live net no-rate delta `+0.23638828717497262`.
- Endpoint live-vs-EMA d_pose divergence:
  `0.21424890481926206 - 0.010094668682475555 = 0.20415423613678652`,
  ratio `21.22396599218757`.

## Axis 8 - Assumption Challenge

Shared assumption of this wave:

1. The launch decision assumes terminal live/EMA divergence is plausibly caused by LR-scale
   oscillation, so a tail-derived LR damping A/B is a high-value discriminator.
2. The review protocol assumes local Codex can execute enough of the MLX trainer to measure
   runnability before fire.

Would violating it unlock breakthrough?

- If (1) is false, la1 should fail by its preregistered falsifier and the next controller
  should route to Case-B/optimizer-family or pose-coupling mechanisms instead of LR damping.
- Violating (2) does not unlock score directly, but it changes operations: axis-9 smoke must
  run on a Metal-capable local host or governed launcher surface before the clean-pass counter
  can advance.

## Axis 9 Smoke Attempt

Attempted smoke:

- Base: ON ticket argv.
- Edits: `--epochs 1772`; `--out-dir /Volumes/VertigoDataTier/pact/ddm_rv8_20260806/smoke_la1on`.
- Intended scope: six-epoch ON smoke from ep1766 boundary; not the ticket out-dir.
- Parent SSD free space before attempt: `97 GiB`.

Result:

- Trainer process rc `1`.
- Failure excerpt: `RuntimeError: [metal::load_device] No Metal device available` while importing
  `mlx.nn` from `experiments/train_tr1_partition_renderer_mlx.py:4175`.
- Minimal reproducer after `mx.set_default_device(mx.cpu)` also fails on `import mlx.nn` with
  the same error. `MLX_METAL_GPU_ARCH=applegpu_g15` does not change the failure.
- RSS sampler using `ps` was denied by the sandbox (`operation not permitted`), so no valid
  peak RSS was measured.

Certified smoke evidence (kept on SSD; not `/tmp`):

- `/Volumes/VertigoDataTier/pact/ddm_rv8_20260806/smoke_la1on.stdout.log`
  - bytes `1290`
  - sha256 `15c990a9b44b9fce626ca3519638c3ece506c47f4c82f5a7d57fd13c3b2e3296`
  - reason kept: durable failed-smoke stderr evidence; small text artifact.
- `/Volumes/VertigoDataTier/pact/ddm_rv8_20260806/smoke_la1on.rss.tsv`
  - bytes `24`
  - sha256 `29c68d9dbd796a8e6412a8a293ff70845a04b3333a6b25be478d782646a6dd1c`
  - reason kept: records that RSS sampling was attempted but not usable; small text artifact.
- `/Volumes/VertigoDataTier/pact/ddm_rv8_20260806/smoke_la1on/`
  - empty directory; no checkpoint, telemetry, or bulky scratch was created.

## Verification

- Focused pytest:
  `.venv/bin/python -m pytest -q src/tac/tests/test_ddm_bs3_gate_projection_kernel.py::test_bd1_realized_gate_dpose_fields_match_gate_naming_and_label src/tac/tests/test_ddm_bs3_gate_projection_kernel.py::test_bd1_realized_gate_dpose_pass_is_structurally_read_only src/tac/tests/test_ddm_bd1_endpoint_error_atlas.py src/tac/tests/test_ddm_wp1_tr1_muon_finisher.py src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_la1_jd1_lr_anneal_flags_fail_closed_when_inert_or_unresumable src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_la1_jd1_lr_anneal_derives_tail_from_parent_telemetry src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_lr_anneal_lever_composes_with_joint_pose_finish_and_defaults_derive src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_lr_anneal_lever_flags_are_declared_by_the_live_trainer src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_lr_anneal_factory_refuses_inert_or_invalid_shapes src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_muon_finisher_lever_composes_with_case_b_start_boundary src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_muon_finisher_lever_flags_are_declared_by_the_live_trainer src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_muon_finisher_validation_refuses_mid_window_or_unresumed_shapes`
  => `19 passed in 0.58s`; known MLX atexit `No Metal device available` warning after pytest exit.

No source fix was landed. No scorer slot, full launch, archive build, exact eval, or pointer move.

## Fire Order

Do not fire the jd7 ON arm from this review. Required next action:

Run the same bounded ON smoke on a Metal-capable host, with:

- same ON ticket argv,
- `--epochs 1772`,
- owned smoke out-dir under `/Volumes/VertigoDataTier/pact/ddm_rv8_20260806/`,
- capture `jd1_lr_anneal_config`,
- capture first `a1_gate` including `realized_gate_dpose_*`,
- report pose-channel wall-clock delta,
- record valid peak RSS/VRAM inside the tq1c co-tenancy envelope.

Only after that measured smoke passes can the recursive review counter advance.
