# latest report

## current state - 2026-04-09

Track B's promoted honest floor is now **`1.73`** after the `h64` long-horizon QAT+EMA promotion. Track A remains transparency-only and must stay runnable, but it is not part of the honest frontier.

## private ops

- Report history viewer: `reports/graphs/report_history.html`
- Scheduler status: `comma-lab sched status`
- Scheduler results: `comma-lab sched results`
- Scheduler budget: `comma-lab sched budget`
- Default platform registry: `configs/platforms.json`
- Kaggle bundle builder: `experiments/build_kaggle_kernels.py`

## authoritative promoted floor

- Track: `robust_current`
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / long1000 QAT+EMA learned int8 post-filter (alpha=20, h=64)`
- Current-workflow score: **`1.73`** at `864,167` bytes
- Rule-faithful estimate: `1.7947470454539947` at `966,071` bytes
- Distortions: PoseNet `0.03317023`, SegNet `0.00575544`
- Evidence:
  - `reports/raw/robust_current-current_workflow-cpu-summary.json`
  - `reports/raw/robust_current-current_workflow-cpu-report.txt`
  - `reports/raw/robust_current-smoke-current.json`
  - `reports/raw/2026-04-09-long1000-h64-authoritative/`

## verified timeline

- `2026-04-08 09:47:50 -0500` — saliency `alpha=20` scorer-backed `2.01` floor
- `2026-04-08 14:57:15 -0500` — long-500 QAT+EMA scorer-backed `1.99` floor
- `2026-04-08 18:08:27 -0500` — long-500 h32 scorer-backed `1.95` floor
- `2026-04-08 19:12:26 -0500` — long1000 h16 scorer-backed `1.92` floor
- `2026-04-08 21:36:54 -0500` — long1000 h32 scorer-backed `1.85` floor
- `2026-04-09 06:37:51 -0500` — ensemble scorer-backed `1.84` floor
- `2026-04-09 09:43:19 -0500` — h64 smoke confirmation
- `2026-04-09 10:07:00 -0500` — h64 authoritative scorer-backed `1.73` floor

## current frontier

- `1.73` - long1000 QAT+EMA learned int8 post-filter (`alpha=20 h64`), promoted
- `1.84` - weighted ensemble learned int8 post-filter (`long1000 h32 + MC refine1`, `75/25`), prior promoted floor
- `1.84` - SegNet fixed h32 faithful proxy, strongest resolved SegNet-family alternate
- `1.85` - long1000 QAT+EMA learned int8 post-filter (`alpha=20 h32`), older promoted floor
- `1.85` - PSD h64 faithful proxy, resolved non-promoted alternate
- `1.86` - Monte Carlo / layer-scale refinements, strongest non-promoted alternate family
- `1.90` - SegNet-native learned int8 post-filter, close miss

## current outlook

- The strongest scorer-backed move so far is now width scaling to `h64`, not further ensemble tuning.
- The local ensemble micro-tuning lane is closed; the three-way `h32 + MC + Kalman` blend only managed `1.89`.
- The local `h64` training run is complete, and the repo-side official-path proxy on its saved-best artifact independently matched the authoritative `1.73` scorer result.
- The completed `bat00` WSL quantization-parity rerun is now just a reference artifact: its best saved checkpoint was epoch `199` at local scorer `3.9258`, and the trainer was no longer alive when last checked.
- The local MPS host is still the real side-lane factory. The strongest packaged local artifacts are now `dilated h64` at `3.5754`, `psd_h64` at `3.6042`, `pixelshuffle_h64` at `3.6049`, `h96` at `3.8017`, and `alpha30 h32` at `3.8023`; the latest printed checkpoints are `380`, `300`, `380`, `310`, and `660` respectively.
- The refreshed machine-readable fleet snapshot at `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json` now reflects those newer bests and the latest visible SegNet rows: `segnet_attack_fixed_v2` through epoch `1000 / 1.0671` and `segnet_attack_h64` through epoch `480 / 1.0544`.
- The current live `dilated h64` artifact still has one operational caveat: its `/private/tmp` saved meta reports `variant: "saliency_weighted"`, so it should stay observation-only until it is relaunched with the repo-side deploy-correct wrapper. The refreshed proxy gate now shows the raw local gap is only `0.0281`, but deployability still blocks it.
- `pixelshuffle_h64_long1000` was the strongest deploy-ready packaged side lane, and its faithful proxy has now resolved as a clean reject at **`1.99`**. That is a real transfer, but it is nowhere near the promoted `1.73` floor.
- `psd_h64_long1000` has now also resolved honestly at **`1.85`** with PoseNet `0.05271273` and SegNet `0.00551752`. That is a real near-miss, but still a reject for promotion.
- The first real saved SegNet-family artifact, `segnet_attack_fixed_ste_h32`, has now also resolved honestly at **`1.84`** with PoseNet `0.05168364` and SegNet `0.00543626`. That makes it the strongest resolved SegNet-family alternate so far, but it still does not beat the promoted `1.73` floor.
- The SegNet trainer itself is now less bug-prone for future reruns: `experiments/train_postfilter_segnet_attack.py` now always writes a durable `*_final_meta.json` and backstops `*_best_meta.json` when a best-checkpoint payload exists.
- The next honest promotion path is no longer “proxy PSD/pixelshuffle again”; it is to make `dilated_h64_long1000` deploy-correct and to get fresh SegNet lanes that actually emit `best_*` artifacts.
- The SegNet/research side lanes remain non-promoted for now: `segnet_attack_fixed_v2` has printed through epoch `1000 / 1.0671` and did finally write a real fp32/int8 pair, but still no proper `best_meta` record; `segnet_attack_h64` is through epoch `480 / 1.0544` and still has no rankable saved artifact.
- Sidecar tooling is now on disk:
  - `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json`
  - `reports/raw/2026-04-09-sidecar-analysis/proxy_gate_triage.json`
  - `reports/raw/2026-04-09-sidecar-analysis/quantization_drift_audit.json`
  - The refreshed triage output marks `h48 best` as already proxied, keeps `dilated h64` as the strongest unproxied packaged lane, shows `alpha30 h32` ahead of the old h32 rerun, and now blocks `dilated h64` from `proxy_ready` until its deploy metadata is corrected.
  - The quantization audit still points away from “lower fp32→int8 drift” as the main reason the promoted `h64` line wins.
- The waiting-time build swarm also landed three real foundations:
  - `src/comma_lab/task_codec/` for scorer/architecture/quantization/evaluation records
  - `comma-lab sched ...` read-only scheduler reporting
  - `reports/graphs/report_history.html` + `report_history.json` for git-backed markdown/time-machine browsing
- Free-tier operator plumbing is now grounded in repo state rather than chat memory:
  - `configs/platforms.json` now defines `local`, `bat00`, `kaggle`, `modal`, and `coiled`
  - `configs/run_manifests/` now contains neutral Kaggle/Modal/Coiled manifest templates plus a shared status template
  - scheduler compatibility now tolerates legacy manifests without `run_id` and counts `running_managed_session` as active
- Kaggle is now doing real GPU work, not just sitting in the plan:
  - `adpena/comma-lab-dilated-h64-long1000` is running on Kaggle GPU
  - `adpena/comma-lab-segnet-attack-fixed-h32` is running on Kaggle GPU
  - `pairaware_smoke` is built and ready, but currently blocked by Kaggle's free-tier maximum of two batch GPU sessions
- The Kaggle launch surface itself is now hardened:
  - `experiments/kaggle_kernel_builder.py`
  - `experiments/build_kaggle_kernels.py`
  - the version-2 Kaggle kernels install missing Python deps plus `git-lfs` before cloning upstream
- Future-facing experiment code is now on disk too:
  - `experiments/train_postfilter_dilated_h64.py`
  - `experiments/train_postfilter_pixelshuffle_dilated.py`
  - `experiments/train_postfilter_pairaware.py`
  - all three have local tests and are ready for relaunch when a slot is deliberately reassigned
- Latest official leaderboard check at `2026-04-09 06:47:39 -0500` still puts the public targets at `1.89`, `1.94`, and `1.95`. The promoted `1.73` floor is now `0.16` better than public first place.
