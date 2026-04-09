# Current Focus — 2026-04-09 16:54 CDT

## Floor
- **Promoted honest floor**: `1.73` from `robust_current-long1000-h64-promoted-cpu-2026-04-09`
- **Public target to beat**: last verified public first was `1.89`, but re-check before citing again

## Resolved proxy lanes
- `pixelshuffle_h64_long1000`
  - faithful proxy: `1.99`
  - decision: reject for promotion
- `psd_h64_long1000`
  - faithful proxy: `1.85`
  - PoseNet `0.05271273`
  - SegNet `0.00551752`
  - bytes `864,167`
  - decision: real transfer, still reject for promotion

## Best unresolved local lane
- `dilated_h64_long1000`
  - best local scorer: `3.5753838920593264`
  - proxy-gap delta vs promoted h64 best: `0.0281`
  - blocker: not deploy-ready because saved meta still advertises `variant: "saliency_weighted"`

## Infra focus
- `configs/platforms.json` now exists and makes `local`, `bat00`, `kaggle`, `modal`, and `coiled` first-class scheduler platforms
- scheduler compatibility is hardened:
  - legacy manifests/status files may omit `run_id`; loader now falls back to `slug`
  - `launching` and `running_managed_session` now count as active states
- operator templates now exist under `configs/run_manifests/`
- private ops surfaces stay in-repo:
  - `reports/graphs/report_history.html`
  - `comma-lab sched status`
  - `comma-lab sched results`
  - `comma-lab sched budget`

## Next real moves
1. Relaunch `dilated_h64_long1000` with deploy-correct metadata on Kaggle or Modal.
2. Relaunch a fresh SegNet lane that actually emits `best_*` artifacts.
3. Use Kaggle/Modal for GPU training lanes and Coiled for CPU-side audits only.
