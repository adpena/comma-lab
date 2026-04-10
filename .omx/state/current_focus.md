# Current Focus — 2026-04-09 19:15 CDT

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
- `segnet_attack_fixed_ste_h32`
  - faithful proxy: `1.84`
  - PoseNet `0.05168364`
  - SegNet `0.00543626`
  - bytes `864,167`
  - decision: strongest resolved SegNet-family alternate so far, still reject for promotion

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
- SegNet trainer hardening landed:
  - `experiments/train_postfilter_segnet_attack.py` now writes a durable `postfilter_<tag>_final_meta.json`
  - if a best checkpoint payload exists, it also backstops the adjacent `postfilter_<tag>_best_meta.json`
- Kaggle integration is now real but still blocked:
  - version 3 of `adpena/comma-lab-dilated-h64-long1000` errored
  - version 3 of `adpena/comma-lab-segnet-attack-fixed-h32` errored
  - root cause is no longer the read-only workspace path; it is that Kaggle effectively executes only the main code file, so helper modules are unavailable at runtime
  - `adpena/comma-lab-pairaware-smoke` is still ready but blocked by Kaggle's maximum batch GPU session count of `2`
- operator templates now exist under `configs/run_manifests/`
- private ops surfaces stay in-repo:
  - `reports/graphs/report_history.html`
  - `comma-lab sched status`
  - `comma-lab sched results`
  - `comma-lab sched budget`

## Next real moves
1. Replace the broken multi-file Kaggle launch path with truly self-contained single-script kernels for dilated and SegNet.
2. Push `pairaware_smoke` only after a GPU slot frees up or move it to Modal.
3. Use Coiled only for CPU-side audits/reporting while Kaggle is being repaired.
