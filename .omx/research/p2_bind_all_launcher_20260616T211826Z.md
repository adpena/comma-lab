# P2 — bind-all integrated A/B launcher + new P1a flag wiring

**Date:** 2026-06-16T21:18:26Z · **Authority:** `[contest-CPU advisory]` — NO score claim
(this is the LAUNCHER actuator; the exact d_seg/d_pose that pick BEST still run the full
scorer on the CPU authority on the byte-closed archive; pointer 0.19110 UNMOVED).
**Spend:** $0, CPU/code only. No GPU touched; no training launched; no running job disturbed.
**Scope:** launchers ONLY (`experiments/launch_*.py` + tests + this memo). The driver
(`src/tac/torch_vehicle/driver.py`) was NOT edited — a sister subagent (P2-export) owns it.

This lands P2 of the bind-all production build-out
(`.omx/research/production_readiness_bind_all_ingredients_20260616.md`): the composed
**bind-all integrated A/B launcher** that emits the two production run commands, plus the
thin CLI wiring of P1a's new driver Config fields (the "small follow-on" P1a left for the
launcher, per `kd_warm_start_actuator_20260616T210540Z.md`).

## New file
- `experiments/launch_bind_all_taper_ab.py` — the bind-all A/B launcher (NEW, not an
  extension: `launch_taper_ab.py` is the plain-curriculum taper A/B and stays as-is; this
  binds the FULL lever stack and is its own focused actuator per the bolt-on/substrate
  split — extending `launch_taper_ab.py` would have forced the plain A/B to carry the
  bind-all combo it is deliberately NOT testing).
- `experiments/tests/test_launch_bind_all_taper_ab.py` — 21 NO-FAKE tests (all green;
  existing `test_launch_split_by_head_basin_flag.py` still green = no regression).

## Wired flags → driver `TorchVehicleConfig` field (each verified against driver.py)

| launcher flag | `TorchVehicleConfig` field | type / default | source |
|---|---|---|---|
| `--kd-warm-start-dir` | `kd_warm_start_dir` | `Path \| None` / None | P1a |
| `--kd-warm-epochs` | `kd_warm_epochs` | `int` / 300 | P1a |
| `--kd-warm-lr` | `kd_warm_lr` | `float` / 1e-3 | P1a |
| `--kd-warm-train-latents` / `--no-kd-warm-train-latents` | `kd_warm_train_latents` | `bool` / True | P1a |
| `--pose-film-rgb0-pose-trainable` | `pose_film_rgb0_pose_trainable` | `bool` / False | P1a |
| `--pose-film-v2` | `pose_film_enabled=True` + `pose_film_version=2` | — | existing |
| `--pose-film-trunk-stopgrad` | `pose_film_trunk_stopgrad` | `bool` / False | existing |
| `--pose-grad-every-k` | `pose_grad_every_k` | `int` / 1 | existing |
| `--pose-grad-resume-threshold` / `--pose-grad-adaptive` / `--pose-grad-floor-tol` / `--pose-grad-k-max` / `--pose-grad-trend-window` | the APGC fields (same names) | — | existing |
| `--ema-warmup` / `--no-ema-warmup` | `ema_warmup` | `bool` / **True** (launcher default) | existing |
| `--muon-lr-floor-fix` / `--no-muon-lr-floor-fix` | `muon_lr_floor_fix` | `bool` / **True** (launcher default) | existing |
| `--rate-attack` | `lever4_variable_level_export_enabled` | `bool` / False | existing |
| `--taper-channels` (implicit via `--arm`) | `taper_channels` | `list[int]` / None | existing |

Every field name was confirmed to exist in `TorchVehicleConfig` (driver.py L430-664) and
the full bind-all combo BUILDS (passes every `__post_init__` guard — the relaxed
taper+FiLM-v2 constraint P1a landed makes taper [22,16,15,14,15,14,10] + FiLM-v2 +
trunk-stopgrad + rgb_0-pose-trainable + KD-warm + rate-attack co-exist legally). NO flag
was invented; `test_*` exercise the LIVE Config build, not a mock, so a misnamed field
would raise.

## The composed A/B contract

Two arms, **IDENTICAL in every Config field EXCEPT `taper_channels`**
(`test_arms_differ_only_in_taper_channels` diffs the full `dataclasses.asdict` and asserts
the symmetric difference is exactly `{"taper_channels"}`):

- **arm_a (control):** vendored taper `[20,20,20,15,11,10,10]` (83,356 decoder params).
- **arm_b (treatment):** solved byte-neutral taper `[22,16,15,14,15,14,10]` (82,899 params;
  **−0.55% bytes**, feeds the d_seg-critical mid-late HIGH band; gate-2 reallocation).

Both arms pass the vendored schedule EXPLICITLY (arm_a is NOT `None`) so BOTH route through
the SAME `ConfigurableTaperHNeRVDecoder` carrier — the A/B isolates the channel SCHEDULE,
not the decoder class. Both bind, identically:
- **KD-warm-start** from the converged vendored-taper basin (latents load direct; decoder
  frame-MSE distilled) — the wall-clock resolution.
- **oomph** sharp soft_cosine seg (T 0.3→0.05 + tight τ=0.5 + margin RENORM + seg_weight
  1.5×) on the **refinement stages 4-7 only** (coarse stages 0-3 stay vendored CE — the
  KD-warm/basin carries the coarse structure). Verified by
  `test_oomph_overlays_refinement_stages_only`.
- **FiLM-v2** + **trunk-stopgrad** + **rgb_0-pose-trainable** = the COMPLETE pose
  decoupling (∂d_seg/∂(pose-objective)=0 EXACTLY; pose trains {FiLM path + rgb_0}).
- **pose every epoch (k=1)** — per "score > training time" the throttle is a time-proxy;
  APGC is the optional safety net (`--pose-grad-adaptive`).
- **eval_roundtrip** (always-on in the driver) + **EMA** + **EMA-warmup** (launcher
  default ON so the post-KD fine-tune shadow tracks).
- **Muon + score-aware QAT + C1a + σ** inherited from the faithful PR95 refinement stages.
- **Rate attack** (`--rate-attack`) = Lever-4 variable-level export from the online
  score-sensitivity EMA (the SAME shared spine as the QAT).

**Convergence, not a fixed cap:** the default `--total-epoch-budget None` = the full PR95
29,650-epoch schedule; with KD-warm the run converges fast from the distilled basin. Run to
convergence (score is the arbiter, not wall-clock); resumable per-epoch.

**Same seed/budget across arms** (`--seed 0`). A bit-identical init is impossible across
DIFFERENT architectures (the two tapers); same-seed-same-distribution is the cleanest
apples-to-apples for an architecture A/B (as `launch_taper_ab.py` documents). The verdict is
cross-arm: arm_b best contest score `100·d_seg + √(10·d_pose) + 25·B/N` vs arm_a, at matched
bytes.

## The EXACT two run commands (integrated local-MPS run)

Basin teacher confirmed present:
`experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best/`
(holds `best_ema_decoder.pt` + `best_ema_latents.pt` + `best_meta.json`).

Preview each arm's resolved config first ($0, no scorer load, no training):
```bash
.venv/bin/python experiments/launch_bind_all_taper_ab.py --arm arm_a \
  --kd-warm-start-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best \
  --out-dir experiments/results/bind_all_ab_20260616/arm_a \
  --pose-film-v2 --pose-film-trunk-stopgrad --pose-film-rgb0-pose-trainable \
  --train-device mps --split-by-head --n-pairs 600 --pose-grad-every-k 1 --rate-attack \
  --print-plan
```
(swap `--arm arm_b` + `arm_b` out-dir for the treatment preview.)

**arm_a (control — vendored taper):**
```bash
nohup bash -c '.venv/bin/python experiments/launch_bind_all_taper_ab.py --arm arm_a \
  --kd-warm-start-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best \
  --out-dir experiments/results/bind_all_ab_20260616/arm_a \
  --pose-film-v2 --pose-film-trunk-stopgrad --pose-film-rgb0-pose-trainable \
  --train-device mps --split-by-head --n-pairs 600 --pose-grad-every-k 1 \
  --rate-attack --go' \
  </dev/null >experiments/results/bind_all_ab_20260616/arm_a.outer.log 2>&1 & disown
```

**arm_b (treatment — solved taper):**
```bash
nohup bash -c '.venv/bin/python experiments/launch_bind_all_taper_ab.py --arm arm_b \
  --kd-warm-start-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best \
  --out-dir experiments/results/bind_all_ab_20260616/arm_b \
  --pose-film-v2 --pose-film-trunk-stopgrad --pose-film-rgb0-pose-trainable \
  --train-device mps --split-by-head --n-pairs 600 --pose-grad-every-k 1 \
  --rate-attack --go' \
  </dev/null >experiments/results/bind_all_ab_20260616/arm_b.outer.log 2>&1 & disown
```

(`--total-epoch-budget` omitted ⇒ full PR95 schedule to convergence. The arms differ ONLY
in `--arm`/out-dir; everything else is identical. Sequence them so the second starts after
the first frees the MPS GPU, OR run on distinct devices; do NOT contend the Metal GPU.)

## Test confirmation
- `experiments/tests/test_launch_bind_all_taper_ab.py`: **21 passed** — flag→Config-field
  mappings (each via the live Config build), arms-differ-only-in-taper, full-combo-builds,
  `--print-plan` $0 dry-run (rc=0, no scorer), oomph-refinement-only overlay, --go guard.
- `experiments/tests/test_launch_split_by_head_basin_flag.py`: still **5 passed** (no
  regression in the existing launcher tests).
- `ruff check` on both new files: **All checks passed!**

## Wire-in hooks (per Catalog #125; this is a LAUNCHER actuator, no score claim)
1. sensitivity-map — N/A (the launcher consumes the solved taper from the gate-2 map; it
   does not produce a new map. The shared-spine integration — gate-2 map → codec levels +
   QAT bits — is P2-export/driver scope, not the launcher).
2. Pareto — N/A (no byte/score claim from the launcher; the byte/score come from the
   byte-closed exact eval P3 runs on the trained archive).
3. bit-allocator — N/A (the launcher ENABLES the Lever-4 variable-level export via
   `--rate-attack`; the allocation itself lives in the driver export).
4. cathedral autopilot — N/A (a launch actuator, not an archive-deployable lane).
5. continual-learning posterior — N/A (no empirical score anchor; advisory, $0).
6. probe-disambiguator — the launcher IS the A/B actuator (arm_a vs arm_b resolves the
   taper-vs-vendored question empirically); the cross-arm score verdict is the arbitration.
