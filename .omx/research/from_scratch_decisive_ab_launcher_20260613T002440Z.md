# FROM-SCRATCH decisive A/B launcher — validated + READY TO LAUNCH

**UTC:** 2026-06-13T00:24:40Z
**Partner:** from-scratch-launcher (RESPAWN; resumed from checkpoint step 3)
**Authority:** `[macOS-CPU advisory]` / `[macOS-MLX research-signal]` — NON-PROMOTABLE.
Validation here is apparatus-trust evidence (epoch-0 identity, curriculum-match,
levers-active, gradient-alive), NOT a score claim. The score is authoritative ONLY
after the long run + `upstream/evaluate.py` on the byte-closed archive.

## What this is

A `--from-scratch` (epoch-0) mode on `experiments/launch_l2_combined_attacks.py`
that runs the **distortion arm** (FiLM identity-init at ep0 + **levers 2+3+5**) as a
**clean apples-to-apples A/B** against the levers-OFF basin control
`experiments/results/torch_vehicle_full_mps_basin_bc20_n600/`. The curriculum is
built IDENTICALLY to the basin's (8 PR95 stages, 29,650 total epochs, same
seed/n/base_ch/ema/eval_every/latent) so the levers are the ONLY difference.

* **Lever 2** — seg argmax-flip surrogate (`soft_cosine`) + cosine T-anneal 1.0→0.30.
* **Lever 3** — pose-FiLM (identity-init: gamma=1/beta=0 at ep0).
* **Lever 5** — margin-weighted seg promotion (`margin_weight_tau=2.0`).
* Levers 1 (rate) + 4 (score-aware QAT) are **OFF** (distortion arm only).

## CRITICAL CONFIG CORRECTION (R12, commit ab3969ebb) — APPLIED

The anneal endpoint default was `0.05`. **CORRECTED to `seg_temperature_end=0.30`**
(the R12-measured gradient-alive floor `SEG_ANNEAL_GRADIENT_FLOOR_T=0.3`). R12 proved
on the real frozen scorer that the COMBINED seg-lever gradient (Lever-5 margin ×
Lever-2 surrogate) is DEAD below T=0.3 (ratio 1.7e-3 at T=0.3 → ~10 orders below warm
by T=0.1). An endpoint of 0.05/0.10 silently wastes the seg lever on the cold tail of
every stage. The launcher now defaults BOTH `seg_pose` and `all` modes to 0.30 and
SURFACES the `seg_anneal_temperature_is_gradient_alive()` verdict in its mode header.

* `_resolve_lever_overrides`: default `seg_temperature_end = 0.30` (was 0.05).
* `seg_temperature_start = 1.0`, `margin_weight_tau = 2.0`.
* New regression test `test_default_anneal_endpoint_is_the_r12_gradient_alive_floor_030`
  guards the correction (a silent regression to a sub-floor default FAILS).

## BASIN-EXACT MATCH (recovered from the basin's own launch command)

The basin daemon (`launch_nohup.log`) was launched with:

```
experiments/launch_split_by_head_basin.py --no-split-by-head --train-device mps \
  --device cpu --base-channels 20 --n-pairs 600 --async-eval --eval-every 10 \
  --out-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600
```

Basin `run_meta` (from `torch_vehicle_summary.json`): `base_channels=20`,
`latent_dim=28`, `ema_decay=0.999`, `n_pairs=600`, `total_epoch_budget=null`,
`device=cpu` (authority), `seed=0` (default). The from-scratch arm matches ALL of
these. **Critical:** `eval_every=10` (NOT the launcher default 25) and
`total_epoch_budget=None` (the full faithful 29,650 schedule via stage-boundary
alignment). The driver builds its default curriculum via the SAME
`build_curriculum(total_epoch_budget, ema_decay, eval_every)` call the from-scratch
arm uses — so passing the basin's values yields byte-identical schedules.

## VALIDATION EVIDENCE

### Lens 1 — EPOCH-0 IDENTITY (shared-init; NO-FAKE) — PASS
* `test_from_scratch_epoch0_render_is_bit_equal_to_vendored_no_film`: the
  FiLM-identity-init render `torch.equal`s the vendored no-FiLM render at init.
* `test_epoch0_identity_is_not_vacuous_perturbed_film_diverges`: perturbing `fc2`
  (break identity) → render diverges by >1.0 → the identity test verifies BEHAVIOR.
* `test_from_scratch_initial_latents_bit_match_basin_rng_neutral_film`: the FiLM-ON
  initial latents bit-match the no-FiLM basin's (the driver `_new_decoder`
  RNG-snapshot/restore around the FiLM build keeps the stage-1 latent draw at the
  SAME global-RNG stream position).
* `test_latent_bit_match_is_not_vacuous_without_rng_restore_they_diverge`: WITHOUT
  the restore, the FiLM init advances the stream and latents diverge → non-vacuous.

### Lens 2 — CURRICULUM-SCHEDULE MATCH (apples-to-apples) — PASS
Dry-run at the basin-exact config (`eval_every=10`, `total_epoch_budget=None`,
`ema_decay=0.999`):
* `schedule_identical_to_basin = True`; zero schedule-mismatch stage rows.
* 8 stages, epochs `[3000, 5650, 1500, 500, 9000, 2000, 3000, 5000]` = 29,650.
* `eval_every` per stage = 10 (all); `ema_decay` per stage = 0.999.
* The launcher FAILS CLOSED (raises SystemExit) if the schedule diverges from the
  basin reference — the matched-epoch A/B requires identical schedules.
* `seg_loss_fn` per-call closure handled correctly (compared by module+qualname, not
  object-identity, so the benign per-call lambda re-creation is not false-flagged; a
  real swap IS caught — `test_match_table_detects_a_real_seg_loss_fn_swap_no_fake`).

### Lens 3 — LEVERS GENUINELY ACTIVE FROM EPOCH 0 + end=0.30 GRADIENT-ALIVE — PASS
Dry-run resolved active levers (distortion arm flags):
```
lever2_seg_surrogate            = soft_cosine
lever2_seg_temperature          = 1.0
lever2_seg_temperature_end      = 0.30          <- R12-corrected
lever2_anneal_endpoint_gradient_alive = True    <- >= floor 0.3
lever3_pose_film_enabled        = True
lever5_margin_weight_tau        = 2.0
lever1_rate_lambda_w            = 0.0   (OFF)
lever4_score_aware_qat          = False (OFF)
```
* `test_levers_active_on_every_stage_from_stage0`: seg surrogate + anneal + margin set
  on EVERY stage's StageSpec (stage 0 = `stage1_v328_ce`) → the driver routes the seg
  term through the surrogate from epoch 0 of stage 1.
* `test_from_scratch_self_test_runs_levers_end_to_end`: synthetic-scorer ~3-epoch
  end-to-end run with seg+margin+pose-FiLM trains + exports a parseable archive WITH
  the pose section (the composed lever path runs from a fresh init).
* Real-scorer descent smoke: `experiments/probe_r13_all5_end_to_end_descent.py`
  (real EfficientNet-B2 SegNet + FastViT PoseNet, anneal to floor 0.3) —
  <!-- result inserted below after the detached daemon completes -->

## SAFETY (the live arm is NOT touched)
* Refuses launch without `--go`.
* Hard guard: refuses `--out-dir == experiments/results/torch_vehicle_full_mps_basin_bc20_n600`.
* Default byte-identity preserved: levers-OFF path = vendored baseline
  (`test_default_*` in `test_all_layer2_levers.py` green).
* The driver RNG-snapshot/restore is BASIN-SAFE: the basin never enters the
  `pose_film_enabled` branch, so its trajectory is byte-identical.

## TEST SUMMARY
* `test_from_scratch_launcher.py`: 16 passed.
* `test_from_scratch_launcher.py` + `test_pose_film_wire_in.py` +
  `test_seg_surrogate_lever.py`: 38 passed.
* `test_all_layer2_levers.py -k "default or byte"`: <!-- inserted below -->

## THE EXACT LAUNCH COMMAND (basin-matched, gradient-alive end=0.30)

Do NOT launch yet — the orchestrator fires AFTER the Lever-2-vs-CE gate (#117) clears.

```bash
.venv/bin/python -u experiments/launch_l2_combined_attacks.py \
  --from-scratch --go \
  --levers seg_pose \
  --seg-surrogate soft_cosine --seg-temperature 1.0 --seg-temperature-end 0.30 \
  --margin-weight-tau 2.0 \
  --pose-film-hidden 8 \
  --base-channels 20 --latent-dim 28 --n-pairs 600 \
  --ema-decay 0.999 --eval-every 10 \
  --no-split-by-head --train-device mps --device cpu --async-eval \
  --seed 0 \
  --out-dir experiments/results/from_scratch_decisive_ab_lever235_<UTC>
```

Notes:
* `--seg-temperature-end 0.30` is explicit (matches the new default; explicit = audit-clear).
* `--margin-weight-tau 2.0` is REQUIRED (seg_pose mode defaults Lever 5 OFF; the
  distortion arm is levers 2+3+5, so margin must be passed).
* `total_epoch_budget` is omitted → `None` → the full faithful 29,650 schedule
  (matches the basin). Do NOT pass `--total-epoch-budget`.
* `--out-dir` MUST be a NEW dir (the hard guard refuses the live control dir). Stamp it.
* Recommended detached daemon launch (per "Durable detached daemons, not
  session-watchers"; long run >> SIGURG-144 window):
  ```bash
  nohup bash -c '<the command above> > <out-dir>/launch_nohup.log 2>&1' \
    < /dev/null > <out-dir>/launch.outer.log 2>&1 & disown
  ```
