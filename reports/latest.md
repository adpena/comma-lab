# latest report

## current state - 2026-04-09

Track B's promoted honest floor is now **`1.73`** after the `h64` long-horizon QAT+EMA promotion. Track A remains transparency-only and must stay runnable, but it is not part of the honest frontier.

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
- `1.85` - long1000 QAT+EMA learned int8 post-filter (`alpha=20 h32`), older promoted floor
- `1.86` - Monte Carlo / layer-scale refinements, strongest non-promoted alternate family
- `1.90` - SegNet-native learned int8 post-filter, close miss

## current outlook

- The strongest scorer-backed move so far is now width scaling to `h64`, not further ensemble tuning.
- The local ensemble micro-tuning lane is closed; the three-way `h32 + MC + Kalman` blend only managed `1.89`.
- The local `h64` training run is complete, and the repo-side official-path proxy on its saved-best artifact independently matched the authoritative `1.73` scorer result.
- The completed `bat00` WSL quantization-parity rerun is now just a reference artifact: its best saved checkpoint was epoch `199` at local scorer `3.9258`, and the trainer was no longer alive when last checked.
- The local MPS host is now running the real active side-lane fleet. The strongest newly packaged local artifacts are `dilated h64` at `3.7601`, `pixelshuffle_h64` at `3.8106`, `alpha30 h32` at `3.8315`, and `h96` at `3.8945`; the latest printed checkpoints are now `250`, `210`, `480`, and `220` respectively, but none is yet close enough to the promoted h64 local regime to justify proxy time.
- A new machine-readable fleet snapshot is now on disk at `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json`. It corrected the earlier hand-tail undercounts further and now merges special-case log slugs back into their canonical best-artifact lanes: `pixelshuffle_h64` improved to saved best `3.8106`, `psd_h64` to `3.8837`, `alpha30 h32` had printed through `480`, `dilated h64` through `250`, and the two live SegNet lanes through `890` and `340`.
- The current live `dilated h64` artifact also has one operational caveat: its `/private/tmp` saved meta still reports `variant: "saliency_weighted"`, so it should stay observation-only until it is relaunched with the repo-side deploy-correct wrapper. Even after the new best at `3.7601`, the refreshed proxy gate still keeps it out because it is both deploy-blocked and still `0.2128` above the promoted h64 local reference.
- `pixelshuffle_h64_long1000` is now the strongest deploy-ready packaged side lane at local best `3.7977`, and a faithful proxy run on that artifact is now in flight after fixing the inflate-time loader mismatch.
- The SegNet/research side lanes remain non-promoted for now: `segnet_attack_fixed_v2` has reached epoch `890` with best observed local scorer `0.9819` at epoch `590`, `segnet_attack_h64` is through epoch `340` with latest printed scorer `1.1381`, `pixelshuffle_h64` has improved to a still-weak saved best at `3.8106`, and `psd_h64` has improved to `3.8837`; the live SegNet trainers still have no saved best artifact, and the active SegNet PIDs predate the trainer file now on disk, so the missing saves are best explained by stale running processes rather than a fresh code-path failure.
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
- Future-facing experiment code is now on disk too:
  - `experiments/train_postfilter_dilated_h64.py`
  - `experiments/train_postfilter_pixelshuffle_dilated.py`
  - `experiments/train_postfilter_pairaware.py`
  - all three have local tests and are ready for relaunch when a slot is deliberately reassigned
- Latest official leaderboard check at `2026-04-09 06:47:39 -0500` still puts the public targets at `1.89`, `1.94`, and `1.95`. The promoted `1.73` floor is now `0.16` better than public first place.
