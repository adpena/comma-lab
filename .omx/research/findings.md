# findings

## 2026-04-09 PSD proxy resolution and free-tier platform hardening

### new measured result

- `psd_h64_long1000` faithfully proxied to **`1.85`**
- Distortions: PoseNet `0.05271273`, SegNet `0.00551752`
- Current-workflow bytes: `864,167`
- Evidence:
  - `reports/raw/2026-04-09-psd-h64-best/psd_h64_long1000_proxy_summary.json`
  - `reports/raw/2026-04-09-psd-h64-best/proxy_psd_h64_long1000_best.log`

### verdict

- The PSD family is a real transfer, not a loader mirage.
- It is still a reject for promotion because it does not beat the promoted `1.73` floor.
- `pixelshuffle_h64_long1000` (`1.99`) and `psd_h64_long1000` (`1.85`) should now be treated as resolved non-promoted alternates, not active promotion lanes.

### platform/scheduler findings

1. `configs/platforms.json` now makes `local`, `bat00`, `kaggle`, `modal`, and `coiled` first-class scheduler platforms instead of chat-only intentions.
2. The scheduler had a real compatibility bug against repo history: several legacy `.omx/logs/remote_jobs/*.json` manifests omit `run_id`. The loader now falls back to `slug`, which keeps historical operator state readable instead of crashing `comma-lab sched ...`.
3. The scheduler also had a vocabulary bug: repo-local live states like `launching` and `running_managed_session` were real active runs but were not counted as active. That is now fixed.
4. Kaggle/Modal/Coiled integration is now grounded in operator templates, not just plans:
   - `configs/run_manifests/kaggle_run_manifest.template.json`
   - `configs/run_manifests/modal_run_manifest.template.json`
   - `configs/run_manifests/coiled_run_manifest.template.json`
   - `configs/run_manifests/run_status.template.json`
   - `docs/operator_run_manifest_templates.md`
5. Under a free-tier-first strategy, Kaggle is the primary GPU training surface, Modal is the secondary GPU fallback, and Coiled is best treated as CPU-side fan-out for audits/reporting rather than the main training path.
6. Kaggle is now not just “integrated” on paper; it is actually being exercised. The helper-file bundle strategy failed, but the direct-code-file pivot is now live. At the moment `adpena/comma-lab-segnet-attack-fixed-h32` is the live Kaggle run, `adpena/comma-lab-dilated-h64-long1000` has fallen to `CANCEL_ACKNOWLEDGED` and is queued for repush, and `pairaware_smoke` remains blocked by Kaggle's maximum batch GPU session count of `2`.
7. The first Kaggle launch attempt exposed a real bootstrap bug: the generic runner assumed the image already had our Python/video dependencies and `git-lfs`. That is now hardened in `experiments/kaggle_kernel_builder.py`.
8. The next Kaggle failure exposed the deeper integration truth: Kaggle effectively executes only the main code file, so helper modules from the uploaded bundle are not reliably importable at runtime. The evidence is on disk under `reports/raw/2026-04-09-kaggle-launch-debug/`, where both version-3 logs fail on `ModuleNotFoundError` for the helper trainer modules.
9. The right Kaggle execution model is therefore simpler than the original bundle plan: the kernel code file itself must be the self-contained trainer. The repo now has that path for both the deploy-correct dilated lane and the SegNet fixed h32 lane.
10. Kaggle's Tesla P100 is still a real constraint. The self-contained cloud trainers now fall back to CPU when they detect unsupported sm_60 CUDA instead of crashing immediately, which keeps the run alive long enough to produce checkpoints and logs.
11. The baseline archive is now staged through a private Kaggle dataset, `adpena/comma-lab-private-assets`, so future kernel attempts no longer need to depend on bundle-side data files for that asset.

## 2026-04-09 SegNet fixed faithful proxy resolution

### new measured result

- `segnet_attack_fixed_ste_h32` faithfully proxied to **`1.84`**
- Distortions: PoseNet `0.05168364`, SegNet `0.00543626`
- Current-workflow bytes: `864,167`
- Evidence:
  - `reports/raw/2026-04-09-segnet-attack-fixed-best/segnet_attack_fixed_ste_h32_proxy_summary.json`
  - `reports/raw/2026-04-09-segnet-attack-fixed-best/proxy_segnet_attack_fixed_ste_h32.log`

### verdict

- This is the strongest resolved SegNet-family alternate so far.
- It transfers honestly and ties the old ensemble floor numerically at `1.84`.
- It is still not a promotion because it does not beat the promoted `1.73` h64 floor.
- The main remaining operational weakness is packaging metadata: the trainer wrote fp32/int8 weights, but not a proper `best_meta` record.
- That metadata weakness is now fixed for future reruns: `experiments/train_postfilter_segnet_attack.py` writes a durable `*_final_meta.json` and backstops `*_best_meta.json` when a best-checkpoint payload exists.

## 2026-04-09 h64 authoritative promotion

### new authoritative floor

- Track B now has a promoted honest floor at **`1.73`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / long1000 QAT+EMA learned int8 post-filter (alpha=20, h=64)`
- Current-workflow bytes: `864,167`
- Rule-faithful estimate: `1.7947470454539947` at `966,071` bytes
- Distortions: PoseNet `0.03317023`, SegNet `0.00575544`

### authoritative evidence

- scorer summary: `reports/raw/robust_current-current_workflow-cpu-summary.json`
- scorer report: `reports/raw/robust_current-current_workflow-cpu-report.txt`
- smoke: `reports/raw/robust_current-smoke-current.json`
- cycle evidence:
  - `reports/raw/2026-04-09-long1000-h64-authoritative/robust_current-long1000-h64-current_workflow-cpu-summary.json`
  - `reports/raw/2026-04-09-long1000-h64-authoritative/robust_current-long1000-h64-current_workflow-cpu-report.txt`
  - `reports/raw/2026-04-09-long1000-h64-authoritative/robust_current-long1000-h64-smoke.json`

### verified timestamps

- `2026-04-09 09:25:04 -0500` — repo-side official-path proxy on the saved-best h64 artifact resolved at the same `1.73` current_workflow score
- `2026-04-09 09:43:19 -0500` — h64 smoke report landed
- `2026-04-09 10:07:00 -0500` — h64 authoritative scorer report landed
- `2026-04-09 10:07:00 -0500` — canonical summary and weight copy refreshed
- `2026-04-09 06:47:39 -0500` — official leaderboard rechecked at `https://comma.ai/leaderboard`

### cycle verdicts

- **Promoted:** long1000 QAT+EMA learned int8 post-filter (`alpha=20 h64`) at `1.73`
- **Prior floor:** weighted ensemble learned int8 post-filter (`long1000 h32 + MC refine1`, `75/25`) at `1.84`
- **Older floor:** long1000 QAT+EMA learned int8 post-filter (`alpha=20 h32`) at `1.85`
- **Strongest non-promoted alternate:** bounded Monte Carlo / layer-scale family at `1.86`
- **Close miss:** SegNet-native h32 at `1.90`

### main findings

1. Width scaling inside the same shipped learned post-filter family beat the ensemble branch decisively. The h64 line, not further checkpoint mixing, is now the best scorer-backed result in the repo.
2. The promoted `1.73` floor came from the same honest payload interpretation as the prior winners. The gain was not a packaging trick; it was a large PoseNet reduction at the same current-workflow byte regime.
3. SegNet leverage remains the main mathematical lesson. At the promoted `1.73` operating point, the score is about **`11.5x`** more sensitive to SegNet than PoseNet.
4. Latest official leaderboard check at `2026-04-09 06:47:39 -0500` puts the promoted floor `0.16` ahead of first (`1.89`), `0.21` ahead of second (`1.94`), and `0.22` ahead of third (`1.95`).
5. The bounded ensemble family transferred honestly but is now closed as a prior-floor branch. The exact two-model sweep peaked near `70/30`, and the three-way follow-on regressed to `1.89`.
6. The “headroom paradox” review surfaced a real code-path mismatch. The repo already had per-channel runtime quantization support in `inflate_postfilter.py` and quantized checkpoint selection in `train_postfilter_v2.py`, but the winning QAT+EMA recipe had not yet inherited either mechanism. `train_postfilter_qat_ema.py` now has explicit `--checkpoint-select-int8`, `--per-channel-int8`, and `--checkpoint-eval-every` controls so that hypothesis can be tested directly.
7. The bat00 WSL quantization-parity rerun produced a clean saved reference artifact but did not earn promotion. Its final useful state was a saved best checkpoint at epoch `199` with local scorer `3.9258260917663574` and a `16,781`-byte int8 payload; the trainer was no longer alive when checked at `2026-04-09 10:59:38 -0500`.
8. The local `h64` long-run produced a real transfer, not just a proxy mirage. Its saved best checkpoint on disk is epoch `918` with local scorer `3.5472697671254476` and a `45,587`-byte int8 payload, and both the authoritative scorer path and the repo-side official-path proxy landed the same rounded `1.73`.
9. The post-promotion local fleet widened materially, but still has not beaten the promoted line. The strongest packaged side lanes are now `dilated h64` at local scorer `3.5753838920593264`, `psd_h64` at `3.604202709197998`, `pixelshuffle_h64` at `3.6048873551686604`, `h96` at `3.8016996637980145`, and `alpha30 h32` at `3.802276372909546`, all still weaker than the promoted h64 best at `3.5472697671254476`.
10. The SegNet side lane still has attractive theoretical headroom, but it remains only partially operationalized. `segnet_attack_fixed_v2` printed through epoch `1000`, wrote a real fp32/int8 pair, and faithfully proxied to `1.84`, but it still did not emit a proper `best_meta` record; `segnet_attack_h64` has printed through epoch `480` with latest visible scorer `1.0544` and still has no rankable saved artifact.
11. A machine-readable fleet snapshot at `2026-04-09 13:13:08 -0500` confirmed that the widened local host still has no honest proxy candidate. `dilated h64` improved its saved best further to epoch `253`, `pixelshuffle_h64` to epoch `211`, `psd_h64` to epoch `115`, `alpha30 h32` held at epoch `423`, and `h96` held at epoch `217`; the saved packaged standings still remained too weak to justify proxy time.
12. Sidecar automation is now in-tree and tested:
   - `experiments/proxy_gate_triage.py`
   - `experiments/quantization_drift_audit.py`
   - tests: `experiments/test_proxy_gate_triage.py`, `experiments/test_quantization_drift_audit.py`
13. The refreshed proxy-gate triage output at `reports/raw/2026-04-09-sidecar-analysis/proxy_gate_triage.json` kept the same operational conclusion while syncing the newer side-lane values: `postfilter_long1000_h48_best` is already resolved because a proxy log already exists for it, `postfilter_dilated_h64_long1000` remains the strongest *unproxied* packaged lane, `postfilter_long1000_h32_a30` remains ahead of the old h32 rerun, and the newer `psd_h64_long1000` lane is visible on the board but still weak.
14. The proxy-gate triage is now deployability-aware. It preserves the ranking and proxy-log logic, but it now blocks `postfilter_dilated_h64_long1000` from ever appearing `proxy_ready` while its saved meta still reports the wrong variant for that special lane.
15. The quantization drift audit at `reports/raw/2026-04-09-sidecar-analysis/quantization_drift_audit.json` suggests the promoted `h64` line is not winning because it has the lowest fp32→int8 drift. Across the audited packaged lanes, aggregate drift is actually slightly lower on `dilated_h64` and `h96` than on the promoted `h64`, which pushes the explanation back toward architecture or optimization rather than quantization alone.
16. The refreshed `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json` keeps proving the same operational lesson: hand-tail polling was understating several lanes. The machine-readable snapshot now shows `dilated_h64_long1000` at epoch `386 / 3.5753838920593264`, `pixelshuffle_h64_long1000` at `383 / 3.6048873551686604`, `psd_h64_long1000` at `296 / 3.604202709197998`, `segnet_attack_fixed_v2` printed through epoch `1000`, and `segnet_attack_h64` through epoch `480`.
17. The snapshot tool needed one more hardening pass beyond simple log/meta parsing: special-case log slugs and best-meta slugs were initially being treated as separate lanes. After merging `dilated_h64` -> `dilated_h64_long1000`, `pixelshuffle_h64` -> `pixelshuffle_h64_long1000`, and `psd_h64` -> `psd_h64_long1000`, the fleet view became honest enough to use as the default polling surface.
18. Even after correcting the polling method, the decision still did not change. The strongest non-promoted packaged lanes are now `dilated h64` (`3.5754`), `psd_h64` (`3.6042`), `pixelshuffle_h64` (`3.6049`), `h96` (`3.8017`), and `alpha30 h32` (`3.8023`), all still weaker than the promoted h64 best at `3.5472697671254476`.
19. The new best `dilated h64` artifact is now even closer to the promoted h64 local regime than before, but the proxy gate still keeps it out for two concrete reasons: it is still deploy-blocked by the wrong saved variant metadata, and its local gap is still `0.2128`, just above the current `0.20` threshold.
20. Three previously chat-only architecture ideas are now concrete repo scaffolds:
   - `experiments/train_postfilter_dilated_h64.py`
   - `experiments/train_postfilter_pixelshuffle_dilated.py`
   - `experiments/train_postfilter_pairaware.py`
   plus tests:
   - `experiments/test_train_postfilter_dilated_h64.py`
   - `experiments/test_train_postfilter_pixelshuffle_dilated.py`
   - `experiments/test_train_postfilter_pairaware.py`
   They are not scored yet, but the next relaunch no longer needs fresh implementation work.
21. The current `/private/tmp` `dilated h64` artifact is not yet an honest deploy-path candidate even if its local score keeps improving, because its saved meta still reports `variant: "saliency_weighted"` instead of a dilated-specific variant. The repo-side `experiments/train_postfilter_dilated_h64.py` wrapper exists specifically to fix that on the next clean relaunch.
22. The two live SegNet research jobs are most likely stale processes rather than evidence that the current trainer still fails to save checkpoints. Their PIDs started at `2026-04-09 07:49:46 -0500` and `2026-04-09 09:29:13 -0500`, but the current `/private/tmp/pact-mine/experiments/train_postfilter_segnet_attack.py` file was updated later at `2026-04-09 11:50:48 -0500`. That matches the logs, which show epoch rows only and none of the current trainer's `eval score=` / `best checkpoint -> ...` lines.
23. The waiting-time build swarm produced three usable foundations, not just plans:
   - `src/comma_lab/task_codec/` now exposes metadata-first scorer, architecture, quantization, and evaluation/proxy abstractions that fit the repo’s existing artifact formats.
   - `src/comma_lab/scheduler/` plus `comma-lab sched status/results/budget` now provide read-only scheduler reporting from existing manifests, ledgers, and `reports/results.jsonl`.
   - `reports/graphs/build_report_history.py` now emits `report_history.json`, and `reports/graphs/report_history.html` renders a git-backed history/time-machine view that exports through the static site pipeline.
24. The scheduler foundation is intentionally conservative. It is cross-platform and stdlib-first, but it is reporting-only for now: no fake remote execution, no unsafe shell templating, and `sched budget` requires an explicit or default `configs/platforms.json` registry.
25. The report-history viewer is already usable as a standalone static page, but it is not yet linked from the main dashboard. That is a product/navigation choice, not a missing build step.
26. Modal / Coiled / Kaggle integration work is still in progress. Keep those operational notes private-facing for now and avoid exposing any sensitive launch details in report surfaces.
27. `pixelshuffle_h64_long1000` is now the first post-h64 packaged lane to earn an honest faithful proxy attempt after the runtime-path fixes. The earlier proxy failure was a real loader bug, not a bad candidate. The loader now supports the pixelshuffle-dilated architecture and can infer it from the artifact state layout even when the saved metadata is wrong.
28. That faithful proxy has now resolved cleanly, and it is not close. `pixelshuffle_h64_long1000` landed at **`1.99`** with PoseNet `0.07282460`, SegNet `0.00562080`, and `864,167` bytes. That is a real transfer, but it is nowhere near the promoted `1.73` floor.
29. With pixelshuffle resolved, `psd_h64_long1000` briefly became the strongest packaged lane that was both deploy-ready and unresolved. That proxy has now landed at `1.85`, so the family moves from “active decision lane” to “resolved non-promoted alternate.”

## 2026-04-08 prior long1000 h32 QAT+EMA post-filter promotion

### new authoritative floor

- Track B now has a promoted honest floor at **`1.85`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / long1000 QAT+EMA learned int8 post-filter (alpha=20, h=32)`
- Current-workflow bytes: `864,167`
- Rule-faithful estimate: `1.8925757653476154` at `935,166` bytes
- Distortions: PoseNet `0.04809216`, SegNet `0.00576402`

### authoritative evidence

- scorer summary: `reports/raw/robust_current-current_workflow-cpu-summary.json`
- scorer report: `reports/raw/robust_current-current_workflow-cpu-report.txt`
- cycle evidence:
  - `reports/raw/2026-04-08-long1000-h32-authoritative/robust_current-long1000-h32-current_workflow-cpu-summary.json`
  - `reports/raw/2026-04-08-long1000-h32-authoritative/robust_current-long1000-h32-current_workflow-cpu-report.txt`
  - `reports/raw/2026-04-08-long1000-h32-authoritative/robust_current-long1000-h32-smoke.json`

### verified timestamps

- `2026-04-08 09:08:58 -0500` — prior `2.01` saliency smoke pass
- `2026-04-08 09:47:50 -0500` — prior `2.01` saliency scorer report
- `2026-04-08 13:02:24 -0500` — ROI + post-filter reject evidence landed
- `2026-04-08 14:57:15 -0500` — long-500 QAT+EMA smoke and scorer report landed
- `2026-04-08 17:55:44 -0500` — long-500 h32 smoke and scorer prep landed
- `2026-04-08 18:08:27 -0500` — long-500 h32 scorer report landed
- `2026-04-08 19:12:26 -0500` — long1000 h16 scorer report landed
- `2026-04-08 19:15:00 -0500` — long1000 h16 smoke report landed
- `2026-04-08 21:36:54 -0500` — long1000 h32 scorer report landed
- `2026-04-08 21:39:21 -0500` — long1000 h32 smoke report landed
- `2026-04-08 22:37:31 -0500` — Jacobian SVD diagnostic landed
- `2026-04-08 22:45:41 -0500` — CNN residual / Karpathy diagnostic landed
- `2026-04-08 22:58:04 -0500` — trust-region sweep landed
- `2026-04-08 22:58:37 -0500` — SegNet-attack side-lane advanced to epoch `340`
- `2026-04-08 22:59:12 -0500` — Kalman side-lane advanced to epoch `190`
- `2026-04-08 22:59:18 -0500` — uint8-STE side-lane advanced to epoch `200`

These were backfilled from evidence file mtimes so the chronology is recoverable from disk.

### cycle verdicts

- **Promoted:** long1000 QAT+EMA learned int8 post-filter (`alpha=20 h32`) at `1.85`
- **Prior floor:** long1000 QAT+EMA learned int8 post-filter (`alpha=20 h16`) at `1.92`
- **Older floor:** long-500 QAT+EMA learned int8 post-filter (`alpha=20 h32`) at `1.95`
- **Older floor:** long-500 QAT+EMA learned int8 post-filter (`alpha=20 h16`) at `1.99`
- **Older floor:** saliency-weighted learned int8 post-filter (`alpha=20`) at `2.01`
- **Close miss:** saliency-weighted learned int8 post-filter (`alpha=10 h32`) at `2.03`
- **Close miss:** saliency-weighted learned int8 post-filter (`alpha=20 h16 + weight-only QAT`) at `2.03`
- **Close miss:** saliency-weighted learned int8 post-filter (`alpha=20 h32`) at `2.04`
- **Close miss:** saliency-weighted learned int8 post-filter (`alpha=10`) at `2.04`
- **Rejected:** ROI-trained post-filter rerun on the current learned-postfilter path at `2.10`
- **Rejected:** generic width sweep variants (`hidden=24` -> `2.07`, `hidden=8` -> `2.06`)

### main findings

1. Epoch budget and stable checkpointing mattered more than the newer loss family. The long-horizon QAT+EMA branch outperformed both the `2.01` promoted saliency baseline and the scorer-faithful `v2` family.
2. The h32 long1000 branch pushed the floor further to `1.85` on the official local CPU path and opened a real gap over the public `1.95` lead.
3. SegNet leverage is the main mathematical lesson. At the promoted `1.85` operating point, the score is about `13.9x` more sensitive to SegNet than PoseNet.
4. Generic architecture changes without scorer-aware objective shaping were not enough. Depthwise and luma-only proxy gains were weak.
5. Width on the winning saliency objective did not buy a win. `alpha=20 h32` kept SegNet healthy at `0.00571195` but PoseNet regressed to `0.08001617`, landing at `2.04`.
6. Train/deploy parity work helped, but not enough by itself. Weight-only QAT tightened SegNet to `0.00571970` but gave back too much PoseNet (`0.07874124`) and landed at `2.03`.
7. Correct path parity matters. One saliency rerun falsely reproduced the old `2.08` floor because it used the wrong decode config; the valid rerun required the `2.08` archive with the learned-postfilter decode path.
8. The scorer/trainer fidelity gap is now clearer. PoseNet training is effectively faithful at batch size 1, but SegNet is not: the scorer measures hard `argmax` disagreement while training currently uses a soft overlap surrogate for differentiability.
9. The next training-fidelity branch after the live ROI scorer should test a harder SegNet surrogate or scorer-faithful model-selection loop rather than more width.
10. The ROI proxy story also needs stricter archive discipline. A strong proxy read on `decode_base_archive.zip` did not transfer to the live packaged submission archive, where the same ROI-trained post-filter scored `2.10` with a large SegNet regression (`0.00628219`).
11. The local scorer-fidelity pass found two concrete process bugs:
   - the faithful proxy needed an explicit/live archive resolver instead of silently defaulting to the old base archive
   - `train_postfilter_v2.py` needed full-eval EMA checkpoint selection instead of minibatch training-loss checkpoint selection
12. After those fixes, the first scorer-faithful h16 candidate still only proxied to `2.04` on the official upstream evaluator. That keeps the family alive, but not yet promotion-worthy.
13. Latest official leaderboard check now puts the new floor `0.04` off first (`1.95`) and `0.01` off second (`1.98`), which materially changes the next-step risk/reward calculus.
14. The scorer-faithful h32 follow-up also only proxied to `2.04` on the official upstream evaluator. That strongly suggests the v2 loss family, as currently implemented, is still not the shortest path below `1.99`.
15. Per-channel quantization is not a free floor win. The promoted artifact became smaller (`8018` bytes vs `8519`), but the official upstream proxy stayed at `1.99` with slightly worse distortions, so this is a packaging/rule-faithful improvement at best, not a current-workflow promotion.
16. `bat00` is now a real authenticated CUDA box, but the first long1000 h16 training lane still exits after decode. The remaining blocker is no longer auth or missing inputs; it is a training-path failure that needs direct debugging.
17. Official challenge hardware matters for compliance interpretation. The upstream README says official evaluation has a 30-minute limit, uses a CPU instance with `CPU: 4, RAM: 16GB` when no GPU is required, and a `T4` GPU instance with `VRAM: 16GB, RAM: 26GB` when the inflate path requires a GPU.
18. The late-night Jacobian diagnostics materially changed the theory story. The per-pair PoseNet Jacobian is effectively rank-1 at the operating point that matters: mean singular values `[0.03603, 0.00080, 0.00055, 0.00028, 0.00018, 0.00009]`, entropy-based effective rank `1.008 / 6`, and condition number `398.8`.
19. The trust-region sweep killed test-time Newton or one-shot Jacobian correction more strongly than expected. The measured linear knee is `~0.0001` pixels RMS, and the median relative linearization error is already above `1.0` there.
20. The winning CNN residual is dense and spectral, not sparse or adversarial. On the shipped `1.85` winner it moves `56.6%` of pixels by more than `0.5` LSB and places `90.3%` of luma energy in the mid-frequency band. The failed Jacobian delta moves only `0.0024%` of pixels past `0.5` LSB.
21. Those diagnostics make three next side lanes evidence-backed rather than decorative:
   - an explicit DCT-basis post-filter
   - a low-dimensional Monte Carlo / evolution-strategy search around the incumbent winner
   - an empirical rate-distortion floor analysis lane for cleaner target-setting
22. Current side-lane readings are interesting but still non-promoted:
   - SegNet-attack epoch `340`: local score `1.2348`, PoseNet `0.059356`, SegNet `0.005916`
   - Kalman epoch `190`: local score `3.9578`, PoseNet `0.048824`, SegNet `0.034373`
   - uint8-STE epoch `200`: local score `4.0883`, PoseNet `0.063304`, SegNet `0.034937`
23. The patched h32 rerun solved the checkpoint-preservation problem but not the score problem. Its saved best checkpoint at epoch `191` looked strong locally (`3.8740`) but only official-proxied to `1.99`, so the branch should be treated as a non-transfering local mirage rather than a promotion candidate.
24. The first explicit DCT-basis smoke was a clean non-starter. `dct_midband_alpha20_b8_smoke` stayed exactly at the baseline `4.5182` through epoch `30`, so the spectral prior may still be right in principle, but this first parameterization is not.
25. The first bounded Monte Carlo / layer-scale search did transfer. Its cheap 20-pair screen improved from `4.1098` to `3.8557`, and the saved best artifact official-proxied to `1.93`. That is still weaker than the `1.85` floor, but it is strong enough to keep the family alive for one deeper bounded search.
26. The first seeded Monte Carlo refinement transferred further. `mc_layer_scale_refine1` official-proxied to `1.86` with PoseNet `0.04709668` and SegNet `0.00601479`, which makes this the strongest non-promoted alternate family in the repo so far.
27. The SegNet-native h32 branch also transferred, but it is weaker than Monte Carlo. `segnet_attack_long1000_h32` official-proxied to `1.90` with PoseNet `0.05670501` and SegNet `0.00575334`.
28. The tighter `sigma=0.05` Monte Carlo refinement did not materially improve on the first refinement. `mc_layer_scale_refine2` official-proxied to `1.86` again, which suggests this six-dimensional layer-scale family may already be near its local ceiling.

## 2026-04-07 learned post-filter promotion and state reconciliation

### new authoritative floor

- Track B now has a promoted honest floor at **`2.05`**
- Config: `522x392 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / learned int8 post-filter`
- Current-workflow bytes: `861,986`
- Rule-faithful estimate: `2.0778631822069484` at `896,432` bytes
- Distortions: PoseNet `0.07996829`, SegNet `0.00586716`

### authoritative evidence

- scorer summary: `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-current_workflow-cpu-summary.json`
- scorer report: `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-current_workflow-cpu-report.txt`
- smoke: `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-smoke.json`
- canonical live summary/report pair:
  - `reports/raw/robust_current-current_workflow-cpu-summary.json`
  - `reports/raw/robust_current-current_workflow-cpu-report.txt`

### cycle verdicts

- **Promoted:** learned int8 post-filter at `2.05`
- **Rejected:** grain-mask recovery lane at `2.30` (`716,797` bytes, PoseNet `0.15428504`, SegNet `0.00577725`)
- **Rejected:** first post-filter variant at `2.35` because it was trained on the wrong archive distribution and overcorrected the canonical fg22 floor

### main findings

1. PoseNet remains the real bottleneck. The promoted lane won by improving PoseNet materially while keeping SegNet and bytes within tolerance.
2. Broad preprocessing stays rejected. The repo should not burn another cycle there without a genuinely new causal hypothesis.
3. Film grain remains structurally important evaluator-facing signal.
4. Tiny task-aware decode correction is now the strongest active family in the lab.
5. The post-filter training distribution must match the deployment distribution. The `2.35` miss established that distribution shift is fatal here.

### next-cycle constraints

- Cap the next experiment cycle at three scored candidates:
  1. slightly larger post-filter
  2. smaller or luma-only post-filter
  3. cheaper CPU-friendly post-filter architecture
- BAT00 remains non-authoritative for promotion decisions.
- If a summary surface omits shipped post-filter files from the rule-faithful payload path list, refresh it before citing the path list.

## 2026-04-06 AV1+ROI lane activation and research

### leaderboard intelligence

Public leaderboard as of 2026-04-06:
- #31: 1.95 (ROI + Lanczos + unsharp 0.40)
- #30: 1.98 (hand-authored ROI masks + chroma collapse, 4 temporal segments)
- #24: 2.05 (consensus SVT-AV1 params)
- Our floor: 2.12

The gap is 0.17 points. ROI preprocessing is the dominant technique.

Consensus codec settings across all top entries: SVT-AV1, preset 0, CRF 33, film-grain=22, keyint=180, scd=0, hqdn3d=1.5:0:0:0.

### bleeding-edge research findings

1. **AV1 grain synthesis as a free texture channel**: grain parameters stored in bitstream header (tens of bytes) but synthesized at decode time. Encode at lower bitrate, let grain model inject mid-frequency texture SegNet interprets as edge detail. Potentially the highest-leverage unexploited technique.

2. **Wavelet pre-filtering**: Haar/db2 decomposition → zero out HH/HL sub-bands in unimportant regions → reconstruct. Published results: 10-20% bitrate savings, <0.5% mIoU drop.

3. **Mid-frequency band is what matters**: SegNet/PoseNet rely on 4-32 cycles/frame (edges, texture gradients). Frequencies above ~64 cycles/frame can be safely suppressed in non-ROI regions.

4. **Task-aware QP mapping**: spatially vary QP per-CTU based on importance mask. Road edges/pedestrians/lanes tolerate no quality loss; sky can take QP +15-20 with negligible mIoU impact.

5. **Neural post-processing**: Tiny models (~50-80KB) exist but well-tuned unsharp mask recovers 70-80% of the benefit at zero byte cost. Only worth it at very low bitrates.

### scored experiment results (2026-04-06)

| Exp | Score | Archive | Key change | Verdict |
|-----|-------|---------|-----------|---------|
| Baseline | 2.12 | 864KB | flat AV1 CRF 34 | — |
| sharpness=1 | **2.08** | 864KB | sharpness=1, CRF 34 | **NEW FLOOR** |
| A2 | 2.27 | 667KB | tune=0+qm+fgd, CRF 38 | REJECTED (CRF too aggressive) |
| J | 2.52 | 785KB | sharpness=1+preprocess | REJECTED (blur destroys PoseNet) |
| I | 2.14 | 825KB | sharpness=1+VQ+QM+fgd | Near miss (VQ helps PoseNet, hurts SegNet) |
| K | 2.47 | 841KB | gentle preprocess (σ=0.8, blend=0.25) | REJECTED (even gentle blur kills PoseNet) |
| L | 2.14 | 858KB | sharpness=1+scd0+hqdn3d | REJECTED (denoise hurts slightly) |
| P | 2.16 | 921KB | CRF 33+sharpness=1 only | REJECTED (more bytes, marginal SegNet gain) |
| Q | 2.51 | 838KB | chroma-only + static corridor | REJECTED (chroma degradation ALSO kills PoseNet!) |
| R | 2.48 | 835KB | Falcon ML masks + chroma-only | REJECTED (ML masks don't fix PoseNet issue) |
| O | **2.09** | 864KB | sharpness=2 | Tied with floor (PoseNet -5.7%) |
| Python bicubic USM 0.40 | **2.08** | 864KB | Python inflate (CRF 34) | **TIES FLOOR** |
| **ROI map CRF34 sky+10** | **2.09** | 887KB | SvtAv1EncApp + Falcon QP map | **BEST PoseNet** (0.08475) |
| ROI map CRF34 sky+20 | 2.14 | 882KB | More aggressive sky penalty | Worse (sky penalty too strong) |
| ROI map CRF35 sky+20 | 2.16 | 825KB | CRF 35 + ROI map | Worse (CRF too aggressive with ROI) |
| CRF 35 | 2.09 | 808KB | Lower CRF | Close |
| sharpness=2 | 2.09 | 864KB | Stronger deblocking | Close |
| FG18 | 2.10 | 862KB | Film-grain=18 | Best SegNet with ffmpeg |
| Bicubic downscale | 2.10 | 852KB | Different downscale | Close |
| FG18+Python | 2.10 | 862KB | fg18 + Python inflate | Close |
| CRF35+Python | 2.10 | 808KB | CRF 35 + Python inflate | Close |
| FG18+CRF35 | 2.13 | 806KB | Combined rate savings | Close |
| P | 2.16 | 921KB | CRF 33 + sharpness=1 | REJECTED (more bytes) |
| Python bicubic USM 0.40 | **2.08** | 864KB | Python inflate (CRF 34) | **TIES FLOOR** (PoseNet -9%, SegNet +1%) |
| Python bicubic USM 0.85 | 2.09 | 864KB | Python inflate stronger USM | Marginally worse |
| CRF33+Python USM 0.85 | 2.16 | 921KB | Match PR #24 config | REJECTED (CRF 33 hurts) |
| **T2: CRF 33** | 2.15 | 895KB | CRF 33, canonical inflate | REJECTED (PoseNet 0.096, larger file) |
| **T2: No color tags** | 2.16 | 841KB | Remove -colorspace/-color_primaries/-color_trc from encode | REJECTED (PoseNet 0.101, SegNet 0.00579) |
| **T2: keyint=120** | 2.16 | 840KB | keyint=120 (vs 180 baseline) | REJECTED (PoseNet 0.101, SegNet 0.00581) |

Key learnings:
1. The scoring function heavily penalizes distortion. PoseNet is extremely sensitive to blur outside the corridor (J: +104% PoseNet).
2. VQ mode (tune=0) + QM improve PoseNet by 10% but hurt SegNet by 17.6%. Since 100*seg is the biggest term, SegNet matters more.
3. Rate savings from byte reduction are dwarfed by distortion penalties. Rate term contributes ~0.6 pts; seg contributes ~0.6 pts; pose contributes ~1.0 pts. Marginal rate gains are tiny vs marginal distortion costs.
4. The winning approach must improve BOTH SegNet AND PoseNet, or at least not hurt either.
5. ANY Gaussian blur outside the corridor kills PoseNet — even sigma=0.8, blend=0.25 caused +90% PoseNet degradation. PoseNet uses the ENTIRE frame for pose estimation, including distant features. Only sky-only degradation or chroma-only degradation may be safe.
6. The preprocessing approach needs ML-grade masks that protect everything PoseNet cares about (essentially everything except sky).
7. **CRITICAL**: Even CHROMA-ONLY degradation outside the corridor kills PoseNet by 105%. PoseNet uses color information for pose estimation, not just luminance. ANY pixel modification outside the protected region hurts PoseNet.
8. The preprocessing lane may be fundamentally limited for this scorer. Only encoder-side parameter tuning (like sharpness=1) has proven safe.

### innovation edges identified

1. `tune=0` (VQ mode) — nobody on leaderboard uses it. Untested at CRF 34 (Exp I in progress).
2. `enable-qm=1` (quantization matrices) — untested publicly. Untested at CRF 34.
3. `film-grain-denoise=1` — free byte savings via encode denoise + decode resynthesize
4. ML-grade masking (SAM 3 + Falcon Perception) — needed to fix preprocessing (naive masks fail)
5. Gentle preprocessing (sigma=0.8, blend=0.25) — Exp K in progress

## 2026-04-06 colorspace-hardening AV1 promotion

### new authoritative floor

- Track B now has a new promoted honest floor: **`2.12`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp=0.35 / explicit bt709/tv encode tags / explicit rgb24(pc) decode`
- Current-workflow bytes: `864486`
- Rule-faithful estimate: `2.1418040615200598` at `897745` bytes

### hypothesis and result

- Hypothesis: explicit colorspace/range handling would reduce evaluator mismatch on the flat AV1 path.
- Result: hypothesis held.
- Byte delta vs prior floor: `+31` (`+0.0036%`)
- Score delta vs prior floor: `-0.0600`
- Pose delta: `-0.01272625`
- Seg delta: `+0.00005696`

### interpretation

- This is a production-hardening win, not just a tuning win.
- The score improved materially even though bytes barely changed, which means the evaluator cared about the explicit color conversion contract.
- PoseNet appears much more sensitive to this conversion path than SegNet at the current operating point.

## local frontier shape

The current AV1 frontier now has:
- a compression-side loss (`crf35`)
- a softer-reconstruction loss (`unsharp 0.30`)
- a synthesis-removal loss (`film-grain 0`)
- a geometry loss (`522x392`)
- an upscale-kernel win (`lanczos`)

That is excellent writeup material.

## 2026-04-06 comprehensive bug / rigor pass

### fixed execution-contract bugs

- `robust_current` packaging now honors the requested upstream root
- `--package` without sync is now rejected because it would package bytes different from the bytes under test
- evaluation now clears stale `inflated/` raws before scorer runs
- rule-faithful accounting now charges the installed runtime payload under test

### fixed ROI-path bugs

- AV1 + ROI now fails fast instead of silently drifting into x265-only logic
- ROI metadata analysis now honors both `FFMPEG_BIN` and `FFPROBE_BIN`
- `ROI_X_FRAC` now actually changes static ROI placement
- `INFLATE_POSTFILTER` now applies on ROI inflate paths
- source dimensions are now explicit config values rather than scattered literals

### verified evidence

- AV1 ROI guard returns a deliberate failure
- metadata ROI wrapper log shows both `ffprobe` and `ffmpeg`
- `ROI_X_FRAC=0.05` and `ROI_X_FRAC=0.45` produce different `roi.mkv` artifacts
- ROI inflate output changes under `INFLATE_POSTFILTER=hflip`
- encoded AV1 stream now probes as `tv / bt709 / bt709 / bt709`
- inflated raw path now probes as `rgb24(pc, gbr/bt709/bt709, progressive)`

### repo hygiene

- root `.gitignore` now covers caches and scratch artifacts
- transient `archive/` scratch is no longer left in `submissions/robust_current`
- git history cleanup was not possible because this workspace is not a git repository

## speculative next lane recorded

If AV1 + ROI is revisited, the required implementation plan is now explicitly recorded as:

1. codec-agnostic ROI encode abstraction
2. AV1 params for base/ROI/ROI2 streams
3. matching AV1-aware metadata ROI path
4. matching inflate/smoke/scorer parity checks
5. fresh scorer-backed evidence that it actually helps

This remains speculative until those steps are complete and measured.

## 2026-04-06 writeup system / frontend pass

### what changed

- Added a generated experiment manifest for durable reuse.
- Added generated code callouts tied to measured findings.
- Added reproducibility commands in `justfile` plus `docs/repro_checklist.md`.
- Added browser-preview comparison media with synced full-frame and crop-zoom playback controls.
- Added top-of-page contest context, repo identity, GitHub link, and localized last-updated metadata.
- Added poster images for the comparison videos and mobile-safe horizontal scrolling for the local-frontier table.

### interpretation

- The writeup is now easier to audit because the evidence surfaces and reproduction path are generated, not hand-maintained.
- The landing page is now closer to a technical brief than a generic dashboard.
- Remaining frontend work is refinement, not missing infrastructure.

## 2026-04-06 player / scatter coherence pass

### player findings

- Root cause: the comparison UI only reset the newly active mode and did not pause the hidden pair, so a full-to-zoom switch left the hidden videos playing in the background.
- Fix: pause all videos on mode changes, sync all four players to the shared playhead, then resume only the active pair if playback was already running.
- Result: full/zoom switching now preserves context instead of resetting the comparison.

### chart / layout findings

- The prior `Why 2.12 beat 2.18` SVG metric rows were brittle under real browser layout and could overlap.
- Fix: replaced the brittle SVG metric rows with semantic HTML cards and table layout.
- The search-path failure branch now hangs downward so the temporary regression reads visually as a detour, not an improvement.
- The scatter plot now includes a focused operating-range view, smaller markers, and explicit lower-left-is-better guidance.

## 2026-04-06 final frontend closeout

### closeout status

- No remaining high-confidence frontend blockers were found in the final desktop/mobile audit.
- The landing page now behaves coherently as a static technical brief: clear context, correct directional semantics, stable comparison media, and lighter but still evidence-dense charts.
- Further work from here would be optional refinement, not issue-driven repair.

## 2026-04-06 semantic-rigor refresh

### fresh evidence

- `robust_current` smoke gate re-ran successfully at the live 2.12 floor:
  - evidence: `reports/raw/2026-04-06-semantic-rigor/robust_current-smoke.json`
  - file count: pass
  - exact frame count / geometry-derived bytes: pass
  - sampled RGB semantic sanity: pass
- sampled semantic metrics on frames `0`, `600`, `1199`:
  - MAE mean: `5.536483740103783`
  - MAE max: `6.332820556171543`
  - per-channel mean absolute diff: `[6.930292777738459, 3.1149420316067324, 6.564216410966157]`

### interpretation

- The smoke gate is now strong enough to catch the bug classes that previously mattered most in practice: missing raws, wrong geometry, wrong frame counts, and evaluator-facing RGB byte/semantic drift on representative frames.
- The remaining rigor work should now focus on documentation accuracy and branch-specific parity, not on a missing flat-path smoke layer.

## 2026-04-06 submission policy update

- The lab should **not** submit the current 2.12 floor yet.
- New submission gate:
  1. authoritative measured score below `2.1`
  2. another full low-hanging-fruit exploration round completed first
- Rationale: the current score is competitive but not clearly leading; the lab should extract one more cheap exploration round before spending reputation on a public submission.

## 2026-04-06 pre-submit exploration round

### authoritative win so far

- `sharpness=1` improved the live flat AV1 floor from `2.12` to **`2.08`**.
- Current-workflow bytes: `864168`
- Rule-faithful estimate: `2.1235784276618737` at `922416` bytes

### interrupted candidate

- `scd=0` passed smoke but the scorer process was terminated before completion (`exit 143`).
- This is not valid score evidence and must not be treated as a rejection or a win.

### interpretation

- `sharpness=1` is now the strongest cheap validated lever in the public-AV1 family.
- `scd=0` remains unresolved.
- The next decision is whether to finish the public-family coupled probe (`sharpness=1 + scd=0`) or jump to the first serious speculative byte lane (`Exp A2`).


## 2026-04-06T20:49:00-05:00 — BAT00 saturation / queue-hardening findings

### queue bug found and fixed

- The first BAT00 smoke batch was invalid because the remote worker called `smoke-submission` without `--package` and packaged from a shared mutable source tree.
- Symptom: different configs produced suspiciously identical bytes / semantic metrics.
- Fix applied:
  1. remote worker now creates a per-job copy of the local repo subset
  2. remote worker snapshots the requested config into that job workspace
  3. remote worker runs `smoke-submission --package` against a per-job isolated upstream root
  4. per-job `manifest.json`, `status.json`, and remote JSONL ledger are written under `~/bat00-runs`

### valid BAT00 smoke results

- `exp_j_sharpness1_preprocess`
  - archive bytes: `787244`
  - semantic MAE mean/max: `5.793853969801739` / `6.706579078429677`
  - interpretation: strongest byte reduction of the batch; worth authoritative scoring despite somewhat worse semantic proxy
- `exp_l_sharpness1_scd0_denoise`
  - archive bytes: `858827`
  - semantic MAE mean/max: `5.7153861763577725` / `6.523910815436919`
  - interpretation: modest byte win, modest semantic loss; mixed
- `exp_m_sharpness1_fg24`
  - archive bytes: `863994`
  - semantic MAE mean/max: `5.81114171391425` / `6.582112825392332`
  - interpretation: larger and worse than the current 2.08 lane; weak
- `exp_n_gop240_sharpness1`
  - archive bytes: `860551`
  - semantic MAE mean/max: `5.743901392132655` / `6.528453070896275`
  - interpretation: public-PR-inspired knob looks plausible but not clearly stronger than the ROI-preprocess lane

### authoritative local result

- `exp_h_sharpness1_consensus` full scorer result: **2.13** at `909307` bytes
- verdict: reject
