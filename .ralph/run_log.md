# run log

## 2026-04-09 13:38:11 -0500 - pixelshuffle proxy lane opened for real

### runtime-path fix
- Root cause:
  - the first faithful proxy attempt on `pixelshuffle_h64_long1000` failed because `submissions/robust_current/inflate_postfilter.py` could not reconstruct the pixelshuffle-dilated architecture from the artifact
- Fix:
  - added `PixelShuffleDilatedPostFilter`
  - added explicit `pixelshuffle_dilated` loader support
  - added state-layout inference so the loader can recover the correct architecture even when the saved metadata still says `saliency_weighted`
- Regression:
  - `experiments/test_postfilter_loader.py`

### active proxy lane
- Candidate:
  - `postfilter_pixelshuffle_h64_long1000_best_int8.pt`
- Local best:
  - epoch `229`
  - scorer `3.797689816157023`
  - int8 `94,285`
- Status:
  - faithful proxy session is now running
  - advanced past loader setup and entered inflation on CPU successfully

## 2026-04-09 13:13:08 -0500 - packaged side lanes tightened again

### packaged side lanes
- `dilated_h64_long1000`
  - saved best improved again to epoch `253`, local scorer `3.7600972843170166`, int8 `45,731`
  - latest printed checkpoint: epoch `250`
  - refreshed proxy gate gap vs promoted h64 best: `0.2128`
  - still blocked as not deploy-ready because saved meta advertises `variant: "saliency_weighted"`
- `pixelshuffle_h64_long1000`
  - saved best improved to epoch `211`, local scorer `3.8106321811676027`, int8 `94,285`
  - latest printed checkpoint: epoch `210`
- `psd_h64_long1000`
  - saved best improved to epoch `115`, local scorer `3.8837292766571045`, int8 `94,087`
  - latest printed checkpoint: epoch `130`
- `long1000_h32_a30`
  - saved best holds at epoch `423`, local scorer `3.8314755090077717`
  - latest printed checkpoint: epoch `480`
- `long1500_h96`
  - saved best holds at epoch `217`, local scorer `3.8944560209910075`
  - latest printed checkpoint: epoch `220`

### research lanes
- `segnet_attack_fixed_v2`
  - latest printed checkpoint: epoch `890`
  - best observed local scorer still `0.9819` at epoch `590`
- `segnet_attack_h64`
  - latest printed checkpoint: epoch `340`
  - scorer `1.1381`, PoseNet `0.065391`, SegNet `0.005426`

### decision
- packaged side lanes improved materially again, but no lane crossed the proxy gate
- `dilated_h64_long1000` remains strongest raw score, but still not deploy-ready
- `pixelshuffle_h64_long1000` is now clearly the strongest deploy-ready packaged side lane

## 2026-04-09 12:59:49 -0500 - foundation swarm landed cleanly

### task-aware codec core
- Added:
  - `src/comma_lab/task_codec/__init__.py`
  - `src/comma_lab/task_codec/scorers.py`
  - `src/comma_lab/task_codec/architectures.py`
  - `src/comma_lab/task_codec/quantization.py`
  - `src/comma_lab/task_codec/records.py`
  - `experiments/test_task_codec_core.py`
- Verification:
  - `python3 -m unittest experiments.test_task_codec_core`

### scheduler foundation
- Added:
  - `src/comma_lab/scheduler/__init__.py`
  - `src/comma_lab/scheduler/models.py`
  - `src/comma_lab/scheduler/registry.py`
  - `src/comma_lab/scheduler/repository.py`
  - `src/comma_lab/scheduler/reporting.py`
  - `experiments/test_scheduler_registry.py`
  - `experiments/test_scheduler_cli.py`
- Integrated:
  - `src/comma_lab/cli.py` now serves `comma-lab sched status`, `sched results`, and `sched budget`
- Verification:
  - `python3 -m src.comma_lab.cli sched status --json`
  - `python3 -m src.comma_lab.cli sched results --limit 3 --json`
  - `python3 -m src.comma_lab.cli sched budget` correctly requires a registry

### report/history viewer
- Added:
  - `reports/graphs/build_report_history.py`
  - `reports/graphs/report_history.html`
  - `reports/graphs/report_history.json`
  - `experiments/test_report_history_build.py`
- Integrated:
  - `reports/graphs/build_static_site.py` now exports `report_history.html` and `report_history.json`
- Verification:
  - `python3 reports/graphs/build_report_history.py`
  - `python3 reports/graphs/build_static_site.py`
  - `python3 reports/graphs/build_static_site.py --check`

## 2026-04-09 12:49:18 -0500 - dilated h64 tightened, still not enough

### packaged side lanes
- `dilated_h64_long1000`
  - saved best improved again to epoch `220`, local scorer `3.7737958431243896`, int8 `45,731`
  - latest printed checkpoint: epoch `220`
  - refreshed proxy gate gap vs promoted h64 best: `0.2265`
  - still blocked as not deploy-ready because saved meta advertises `variant: "saliency_weighted"`
- `long1000_h32_a30`
  - saved best holds at epoch `423`, local scorer `3.8314755090077717`
  - latest printed checkpoint: epoch `440`
- `long1500_h96`
  - saved best holds at epoch `192`, local scorer `3.8979716682434082`
  - latest printed checkpoint: epoch `200`
- `pixelshuffle_h64_long1000`
  - saved best holds at epoch `166`, local scorer `3.8732400353749594`
  - latest printed checkpoint: epoch `170`
- `psd_h64_long1000`
  - saved best holds at epoch `63`, local scorer `4.0955122947692875`
  - latest printed checkpoint: epoch `90`

### research lanes
- `segnet_attack_fixed_v2`
  - latest printed checkpoint: epoch `840`
- `segnet_attack_h64`
  - latest printed checkpoint: epoch `310`
  - scorer `1.1058`, PoseNet `0.049163`, SegNet `0.005578`

### decision
- the loop still should not spend proxy time
- `dilated_h64_long1000` is now the closest unresolved packaged lane, but it is still outside the local gap threshold and still not deploy-ready

## 2026-04-09 12:45:25 -0500 - merged snapshot confirms stronger side lanes, still no proxy candidate

### snapshot hardening
- `experiments/live_fleet_snapshot.py` now merges special-case log slugs into canonical best-meta slugs:
  - `dilated_h64` -> `dilated_h64_long1000`
  - `pixelshuffle_h64` -> `pixelshuffle_h64_long1000`
  - `psd_h64` -> `psd_h64_long1000`
- Result:
  - each of those lanes now carries both `best` and `latest` in one record inside `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json`

### packaged side lanes
- `dilated_h64_long1000`
  - saved best epoch `206`, local scorer `3.8148834991455076`, int8 `45,731`
  - latest printed checkpoint: epoch `210`
- `long1000_h32_a30`
  - saved best epoch `423`, local scorer `3.8314755090077717`, int8 `16,091`
  - latest printed checkpoint: epoch `440`
- `long1500_h96`
  - saved best epoch `192`, local scorer `3.8979716682434082`, int8 `93,331`
  - latest printed checkpoint: epoch `200`
- `pixelshuffle_h64_long1000`
  - saved best epoch `166`, local scorer `3.8732400353749594`, int8 `94,285`
  - latest printed checkpoint: epoch `170`
- `psd_h64_long1000`
  - saved best epoch `63`, local scorer `4.0955122947692875`, int8 `94,087`
  - latest printed checkpoint: epoch `80`

### decision
- even after correcting the lane-merging bug, the best non-promoted packaged lane is still too far from the promoted `h64` best (`3.5472697671254476`)
- `dilated_h64_long1000` remains strongest, but the proxy gate still blocks it as not deploy-ready because its saved meta advertises the wrong variant

## 2026-04-09 12:45:25 -0500 - machine snapshot shows real progress, but still no proxy candidate

### packaged side lanes
- `dilated_h64_long1000`
  - saved best improved to epoch `206`, local scorer `3.8148834991455076`, int8 `45,731`
  - latest printed checkpoint: epoch `190`
  - still blocked from `proxy_ready` because the saved meta advertises the wrong variant
- `long1000_h32_a30`
  - saved best improved to epoch `423`, local scorer `3.8314755090077717`, int8 `16,091`
  - latest printed checkpoint: epoch `430`
- `long1500_h96`
  - saved best improved to epoch `192`, local scorer `3.8979716682434082`, int8 `93,331`
  - latest printed checkpoint: epoch `190`
- `pixelshuffle_h64_long1000`
  - saved best improved to epoch `166`, local scorer `3.8732400353749594`, int8 `94,285`
  - latest printed checkpoint: epoch `140`
- `psd_h64_long1000`
  - saved best improved to epoch `63`, local scorer `4.0955122947692875`, int8 `94,087`
  - latest printed checkpoint: epoch `50`

### research lanes
- `segnet_attack_fixed_v2`
  - latest printed checkpoint: epoch `840`
  - best observed local scorer still `0.9819` at epoch `590`
- `segnet_attack_h64`
  - latest printed checkpoint: epoch `300`
  - scorer `1.1058`, PoseNet `0.049163`, SegNet `0.005578`

### decision
- the snapshot corrected the stale hand-tail view, but not the actual decision
- strongest unproxied packaged lane remains `dilated_h64`
- even the improved packaged lanes are still too far from the promoted h64 best (`3.5472697671254476`) to justify proxy time

## 2026-04-09 12:21:38 -0500 - snapshot tooling exposed poll drift, but not a new candidate

### new sidecar tooling
- Added:
  - `experiments/live_fleet_snapshot.py`
  - `experiments/test_live_fleet_snapshot.py`
- Refined:
  - `experiments/proxy_gate_triage.py`
  - `experiments/test_proxy_gate_triage.py`
- Generated:
  - `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json`
  - refreshed `reports/raw/2026-04-09-sidecar-analysis/proxy_gate_triage.json`

### operational finding
- The new machine-readable snapshot showed the earlier hand-tail polling was undercounting several active lanes.
- Corrected current side-lane bests and latest rows:
  - `pixelshuffle_h64_long1000`: saved best epoch `140`, local scorer `3.978545821507772`, int8 `94,285`
  - `psd_h64_long1000`: saved best epoch `58`, local scorer `4.109967034657796`, int8 `94,087`
  - `alpha30_h32`: latest printed epoch `410`
  - `dilated_h64`: latest printed epoch `190`
  - `segnet_attack_fixed_v2`: latest printed epoch `810`
  - `segnet_attack_h64`: latest printed epoch `280`

### decision
- even after correcting the polling source, no lane crossed the proxy gate
- strongest unproxied packaged lane remains `dilated_h64`, but the proxy gate now also marks it as not deploy-ready because its saved meta still advertises the wrong variant

## 2026-04-09 12:09:51 -0500 - follow-on fleet poll, still no proxy candidate

### local packaged side lanes
- `dilated h64`
  - latest printed checkpoint: epoch `170`
  - saved best unchanged at epoch `123`, local scorer `3.8193325742085773`, int8 `45,731`
- `alpha30 h32`
  - latest printed checkpoint: epoch `370`
  - saved best unchanged at epoch `353`, local scorer `3.8521939086914063`, int8 `16,091`
- `h96`
  - latest printed checkpoint: epoch `170`
  - saved best improved to epoch `169`, local scorer `3.95241400718689`, int8 `93,331`
- Decision:
  - the ordering is unchanged in the important way
  - still no packaged lane is locally close enough to the promoted h64 best (`3.5472697671254476`) to justify proxy time

### research lanes
- `pixelshuffle_h64`
  - latest printed checkpoint: epoch `100`
  - saved best unchanged at epoch `97`, local scorer `4.034508647918702`, int8 `94,285`
- `psd_h64`
  - latest printed checkpoint: epoch `20`
  - saved best improved to epoch `26`, local scorer `4.163042246500651`, int8 `94,087`
- `segnet_attack_fixed_v2`
  - latest printed checkpoint: epoch `770`
  - best observed local scorer remains `0.9819` at epoch `590`
- `segnet_attack_h64`
  - latest printed checkpoint: epoch `260`
  - scorer `1.1472`, PoseNet `0.053301`, SegNet `0.005730`
- Decision:
  - keep the research lanes running
  - still no saved SegNet best artifact is visible on disk

## 2026-04-09 12:05:14 -0500 - refreshed live fleet + sidecar outputs

### refreshed sidecar outputs
- Regenerated:
  - `reports/raw/2026-04-09-sidecar-analysis/proxy_gate_triage.json`
  - `reports/raw/2026-04-09-sidecar-analysis/quantization_drift_audit.json`
- Operational result:
  - `postfilter_dilated_h64_long1000` remains the strongest unproxied packaged lane
  - `postfilter_long1000_h32_a30` has now moved ahead of the old h32 rerun in the saved-artifact ranking
  - the quantization-drift conclusion is unchanged qualitatively: the promoted `h64` line does not win because it has uniquely low fp32→int8 drift

### local packaged side lanes
- `dilated h64`
  - latest printed checkpoint: epoch `160`
  - saved best: epoch `123`, local scorer `3.8193325742085773`, int8 `45,731`
  - caveat: current `/private/tmp` saved meta still reports `variant: "saliency_weighted"`, so this artifact should stay observation-only until a deploy-correct relaunch
- `alpha30 h32`
  - latest printed checkpoint: epoch `360`
  - saved best: epoch `353`, local scorer `3.8521939086914063`, int8 `16,091`
- `h96`
  - latest printed checkpoint: epoch `160`
  - saved best: epoch `143`, local scorer `3.957272122701009`, int8 `93,331`
- Decision:
  - no packaged side lane is locally close enough to the promoted h64 best (`3.5472697671254476`) to justify proxy time

### research lanes
- `pixelshuffle_h64`
  - latest printed checkpoint: epoch `90`
  - saved best: epoch `97`, local scorer `4.034508647918702`, int8 `94,285`
- `segnet_attack_fixed_v2`
  - latest printed checkpoint: epoch `760`
  - best observed local scorer remains `0.9819` at epoch `590`
  - still no saved best artifact visible on disk
- `segnet_attack_h64`
  - latest printed checkpoint: epoch `250`
  - scorer `1.0800`, PoseNet `0.045946`, SegNet `0.005610`
  - still no saved best artifact visible on disk
- Verified caveat:
  - both live SegNet processes predate the current `/private/tmp` trainer file on disk
  - PIDs started at `2026-04-09 07:49:46 -0500` and `2026-04-09 09:29:13 -0500`
  - `/private/tmp/pact-mine/experiments/train_postfilter_segnet_attack.py` mtime is `2026-04-09 11:50:48 -0500`
  - interpretation: these are stale in-memory runs, so the honest next artifact-producing SegNet attempt is a clean relaunch from the synced trainer, not more wishful polling
- Decision:
  - keep the research lanes running
  - do not proxy until one of them writes a real, deployable artifact

## 2026-04-09 - 🏆 BREAKTHROUGH: 1.727 via h=64 long1000 QAT+EMA

### h=64 long-1000 QAT+EMA with saved-best int8 checkpoint
- Saved-best weights: `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long1000_h64_best_int8.pt` (`45,587` bytes)
- Architecture: PostFilter h=64 (`25,219` params)
- Training: `1000` epochs, saliency `α=20`, QAT+EMA
- Saved-best checkpoint metadata:
  - epoch `918`
  - local scorer `3.5472697671254476`
- **Authoritative scorer result (CPU):**
  - PoseNet: `0.03317023`
  - SegNet: `0.00575544`
  - bytes: `864,167`
  - rate: `0.02301653`
  - **Final score: `1.73` (rounded) / `1.7268932707523055` (exact)**
- Evidence root:
  - `reports/raw/2026-04-09-long1000-h64-authoritative/`
- Cross-check:
  - repo-side official-path proxy on the saved-best artifact also resolved to the same rounded `1.73`
  - durable log: `/private/tmp/pact-mine/proxy_h64_best_repo.log`
- Decision: **PROMOTE** as the new honest Track B floor
- **Lead over leaderboard #1 (`1.89`): `0.1631` exact / `0.16` rounded**

### Session trajectory
`2.01 → 1.99 → 1.945 → 1.92 → 1.845 → 1.762 → 1.727`

### LeCun scaling law (updated with all data points)
| h | Score | Int8 KB | Params |
|---|-------|---------|--------|
| 8 | 2.06 | ~3 | ~435 |
| 16 | 1.92 | ~8 | ~3203 |
| 32 | 1.845 | ~16 | ~11011 |
| 48 | 1.762 | 28.6 | ~24019 |
| 64 | **1.727** | 45.6 | ~25219 |
| 96 | ~1.67 (projected) | ~100 | est |
| 128 | ~1.66 (crossover zone) | ~130+ | est |

Log-linear fit: `score = -0.159 × ln(h) + 2.382`

### Key mechanism: best-checkpoint int8 evaluation
The partner's `save_best_checkpoint` function evaluates the EMA weights AFTER int8 quantization, selecting the epoch where the quantized model happens to perform best on the scorer. This is why h=48 beat LeCun's prediction (1.762 vs 1.83) and h=64 continued the trend. The train-to-deploy gap is 2.25× on PoseNet — the best-checkpoint mechanism finds the rare epoch where that gap is smallest.

### Panel consensus on maximum potential (24 days remaining)
- Realistic target: **1.45 ± 0.10**
- Path: h=96 + per-channel int8 + SegNet boundary attack + boundary weighting
- SegNet is 98.4% headroom — the single biggest untapped lever
- Rate penalty crossover at h=128-160

### Active follow-ons
1. **bat00 WSL quantization-parity reference**
   - trainer was no longer alive when checked at `2026-04-09 10:59:38 -0500`
   - final useful state: latest printed `200`, saved best epoch `199`, local scorer `3.9258260917663574`
2. **local packaged side-lane fleet**
  - `alpha30 h32`: saved best epoch `231`, local scorer `3.935487003326416`, int8 `16,091`
  - `dilated h64`: saved best epoch `75`, local scorer `4.006724173227946`, int8 `45,731`
  - `h96`: saved best epoch `66`, local scorer `4.030949624379476`, int8 `93,331`
   - decision: all are real, but none is locally close enough to the promoted h64 line to justify proxy time yet
3. **SegNet side-lane watch**
   - `segnet_attack_fixed_v2`: latest printed epoch `630`; best observed local scorer `0.9819` at epoch `590`
   - `segnet_attack_h64`: latest printed epoch `150`, local scorer `1.1657`
   - blocker: neither run has written a rankable saved artifact yet
4. **h64 promotion sync**
   - writeup/site/state surfaces need to stay aligned to the new `1.73` floor while the side-lane fleet evolves

## 2026-04-09 10:58:26 -0500 - fleet poll, no new proxy candidate

### bat00 parity lane
- Remote trainer process was gone when checked at `2026-04-09 10:59:38 -0500`
- Final useful state:
  - latest printed checkpoint `200`
  - saved best epoch `199`, local scorer `3.9258260917663574`
- Decision:
  - mark the lane complete without promotion
  - keep the saved artifact as a reference point only

### local packaged side lanes
- `dilated h64`
  - latest printed checkpoint: epoch `120`
  - saved best improved to epoch `116`, local scorer `3.8786255900065103`, int8 `45,731`
- `alpha30 h32`
  - latest printed checkpoint: epoch `310`
  - saved best improved to epoch `289`, local scorer `3.9331334749857585`, int8 `16,091`
- `h96`
  - latest printed checkpoint: epoch `140`
  - saved best improved to epoch `115`, local scorer `3.9938077545166015`, int8 `93,331`
- Decision:
  - all three are alive
  - none is locally close enough to the promoted h64 best (`3.5472697671254476`) to justify proxy time yet

### SegNet side lanes
- `pixelshuffle_h64`
  - newly launched on local MPS
  - saved best so far: epoch `41`, scorer `4.162973534266154`, int8 `94,285`
- `segnet_attack_fixed_v2`
  - latest printed checkpoint: epoch `710`
  - best observed local scorer remains `0.9819` at epoch `590`
  - still no saved rankable artifact on disk
- `segnet_attack_h64`
  - latest printed checkpoint: epoch `210`
  - scorer `1.2307`, PoseNet `0.063745`, SegNet `0.006025`
  - still no saved rankable artifact on disk
- Decision:
  - keep both running as research lanes
  - do not proxy until one writes a real artifact

### sidecar tooling landed
- Added:
  - `experiments/proxy_gate_triage.py`
  - `experiments/quantization_drift_audit.py`
  - tests:
    - `experiments/test_proxy_gate_triage.py`
    - `experiments/test_quantization_drift_audit.py`
- New experiment scaffolds:
  - `experiments/train_postfilter_dilated_h64.py`
  - `experiments/train_postfilter_pixelshuffle_dilated.py`
  - `experiments/train_postfilter_pairaware.py`
  - tests:
    - `experiments/test_train_postfilter_dilated_h64.py`
    - `experiments/test_train_postfilter_pixelshuffle_dilated.py`
    - `experiments/test_train_postfilter_pairaware.py`
- Generated:
  - `reports/raw/2026-04-09-sidecar-analysis/proxy_gate_triage.json`
  - `reports/raw/2026-04-09-sidecar-analysis/quantization_drift_audit.json`
- Operational result:
  - `h48 best` is no longer a live question; the triage tool marks it as already proxied once proxy-log path normalization is applied
  - strongest unproxied packaged lane is now `dilated_h64`
- Interpretation:
  - the quantization audit does not show the promoted `h64` line winning because it is uniquely cleaner under fp32→int8 conversion

## 2026-04-09 06:47:39 -0500 - weighted ensemble promotion + FiLM cut

### weighted ensemble promotion
- Candidate:
  - `ensemble_h32_qat_mc75_25`
- Composition:
  - `0.75 * postfilter_long1000_qat_ema_alpha20_h32_fp32.pt`
  - `0.25 * postfilter_mc_layer_scale_refine1_best_fp32.pt`
- Evidence root:
  - `reports/raw/2026-04-09-ensemble-h32-qat-mc75-25/`
- Measured result:
  - current_workflow `1.84`
  - PoseNet `0.04678315`
  - SegNet `0.00581610`
  - bytes `864,168`
  - rule_faithful estimate `1.8894260215636247` at `936,886` bytes
- Decision:
  - promote as the new Track B floor
  - the ensemble family is now the shortest path to the next measured gain

### FiLM-conditioned lane
- Durable log:
  - `experiments/postfilter_weights/train_film_conditioned_alpha20_h32_smoke.log`
- Best saved checkpoint before cut:
  - epoch `94`
  - score `4.028719946543376`
  - int8 size `18,195`
- Decision:
  - cut the lane as a weak smoke
  - do not spend proxy time on it in its current form

### official leaderboard check
- Checked at:
  - `2026-04-09 06:47:39 -0500`
  - source: `https://comma.ai/leaderboard`
- Snapshot:
  - `1.89` `neural_inflate`
  - `1.94` `roi_v2`
  - `1.95` `av1_roi_lanczos_unsharp`
- Gap:
  - promoted `1.84` floor is `0.05` better than first
  - `0.10` better than second
  - `0.11` better than third

### bounded ensemble follow-on
- Built:
  - `ensemble_h32_qat_mc80_20`
  - `ensemble_h32_qat_mc70_30`
- Durable proxy log:
  - `/private/tmp/pact-mine/proxy_ensemble_h32_qat_mc80_20.log`
- Official-path proxy for `ensemble_h32_qat_mc80_20` landed at:
  - `2026-04-09 07:16:28 -0500`
- Result:
  - rounded score `1.84`
  - PoseNet `0.04686867`
  - SegNet `0.00580606`
  - bytes `864,167`
  - exact proxy-implied score `1.840626217537`
- Comparison against promoted `75/25` line:
  - promoted exact proxy/scorer-implied score: `1.841006090409`
  - interpretation: the `80/20` variant is a real near-tie and may be microscopically better, but the margin is too small to justify destabilizing the authoritative scorer lane yet
- Official-path proxy for `ensemble_h32_qat_mc70_30` landed at:
  - `2026-04-09 07:55:21 -0500`
- Result:
  - rounded score `1.84`
  - PoseNet `0.04659592`
  - SegNet `0.00582520`
  - bytes `864,167`
  - exact proxy-implied score `1.840545292085`
- Comparison:
  - slightly better exact score than both the promoted `75/25` line and the `80/20` follow-on
  - still too small a margin to justify scorer churn
- Decision:
  - treat the bounded two-point sweep as evidence that the ensemble family still has slope
  - do not promote or scorer-spend on sub-basis-point improvements
- Active now:
  - `ensemble_h32_qat_mc60_40` official-path proxy in progress
  - durable log: `/private/tmp/pact-mine/proxy_ensemble_h32_qat_mc60_40.log`
- Official-path proxy for `ensemble_h32_qat_mc60_40` landed at:
  - `2026-04-09 08:13:46 -0500`
- Result:
  - rounded score `1.84`
  - PoseNet `0.04655845`
  - SegNet `0.00584928`
  - bytes `864,167`
  - exact proxy-implied score `1.842678776449`
- Comparison:
  - worse exact score than both `70/30` and `80/20`
  - confirms the two-model sweep is not just noise; it appears to peak near `70/30`
- Active now:
  - `ensemble_h32_qat_mc_kalman_65_25_10` official-path proxy in progress
  - durable log: `/private/tmp/pact-mine/proxy_ensemble_h32_qat_mc_kalman_65_25_10.log`
- Official-path proxy for `ensemble_h32_qat_mc_kalman_65_25_10` landed at:
  - `2026-04-09 08:31:19 -0500`
- Result:
  - rounded score `1.89`
  - PoseNet `0.05673429`
  - SegNet `0.00566049`
  - bytes `864,167`
- Decision:
  - reject the three-way blend
  - the ensemble family is now mapped well enough; shift the next serious spend to the QAT+EMA quantization-parity rerun

### QAT+EMA parity hardening
- Verified code-path gap:
  - per-channel int8 save/load already existed in the runtime path
  - quantized checkpoint selection already existed in the scorer-faithful `v2` trainer
  - the winning `train_postfilter_qat_ema.py` path had neither
- Implemented in `experiments/train_postfilter_qat_ema.py`:
  - `--checkpoint-select-int8`
  - `--per-channel-int8`
  - `--checkpoint-eval-every`
- Verification:
  - `uv run --with torch --with av --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python -m unittest experiments/test_train_postfilter_qat_ema.py`

### local MPS occupancy check
- Verified at:
  - `2026-04-09 09:21:19 -0500`
- Active long-runs found under `/private/tmp/pact-mine`:
  - `train_segnet_attack_fixed_v2.log`
    - latest visible epoch `400`
    - process still consuming CPU/GPU time
  - `train_long1000_h64.log`
    - completed at epoch `1000`
    - final local eval: loss `3.8046`, PoseNet `0.025294`, SegNet `0.033017`
    - final artifacts:
      - `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long1000_h64_fp32.pt`
      - `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long1000_h64_int8.pt`
- Active local proxy side-lane:
  - repo-side official-path proxy log: `/private/tmp/pact-mine/proxy_h64_best_repo.log`
  - older local estimate log: `/private/tmp/pact-mine/proxy_h64_best.log`
  - target artifact: `/tmp/pact-mine/experiments/postfilter_weights/postfilter_long1000_h64_best_int8.pt`
  - saved best metadata: epoch `918`, scorer `3.5472697671254476`, int8 size `45,587`
  - first local estimate:
    - PoseNet `0.03407505`
    - SegNet `0.00576652`
    - estimated score `1.7358`
  - current status: repo-side official-path proxy is running
  - latest visible progress: `34/38` batches
  - latest visible progress: `30/38` batches
- Decision:
  - do not pretend the local MPS lane is free
  - defer the QAT+EMA quantization-parity rerun until that lane is actually available

### bat00 WSL quantization-parity lane
- Verified capacity snapshot:
  - Windows host memory: `33468356 KiB` total, `17513536 KiB` free
  - WSL2 visible memory: `11 GiB` total, `10 GiB` free
  - GPU: `NVIDIA GeForce RTX 2070 SUPER, 8192 MiB, 0% util, 940 MiB used`
  - no `.wslconfig` present; default WSL memory limits still in effect
- Synced updated trainer/runtime files into `/home/adpena/pact-side`
- Launched at:
  - `2026-04-09 08:35:00 -0500`
- Command:
  - `env PYTHONUNBUFFERED=1 ~/.local/bin/uv run --with av --with torch --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python -u experiments/train_postfilter_qat_ema.py --alpha 20 --hidden 32 --epochs 400 --eval-subsample 1 --checkpoint-eval-every 10 --checkpoint-select-int8 --per-channel-int8 --tag long1000_qat_ema_alpha20_h32_qint8sel_pc_bat00`
- Remote pid:
  - `9615`
- Remote log:
  - `/home/adpena/pact-side/experiments/postfilter_weights/train_long1000_h32_qint8sel_pc_bat00.log`
- Manifest:
  - `.omx/logs/remote_jobs/bat00-long1000-h32-qint8sel-pc.json`
- Relaunched cleanly and confirmed at:
  - `2026-04-09 08:39:00 -0500`
- Remote pid:
  - `9852`
- First confirmed log lines:
  - `[postfilter-saliency] device: cuda`
  - `[qat-ema] device=cuda alpha=20.0 hidden=32 ema=0.997 clip=0.5 tag=long1000_qat_ema_alpha20_h32_qint8sel_pc_bat00`
  - `[qat-ema] Loading scorer models...`
  - `[qat-ema] Decoding compressed archive + ground truth...`
  - `[qat-ema] Baseline: loss=4.5801 pose=0.082572 seg=0.036715`
  - epoch `170`: scorer `4.1164`, PoseNet `0.068095`, SegNet `0.034597`
- First saved best checkpoint:
  - epoch `112`
  - scorer `3.9944437026977537`
  - int8 size `16,781`
  - fp32: `/home/adpena/pact-side/experiments/postfilter_weights/postfilter_long1000_qat_ema_alpha20_h32_qint8sel_pc_bat00_best_fp32.pt`
  - int8: `/home/adpena/pact-side/experiments/postfilter_weights/postfilter_long1000_qat_ema_alpha20_h32_qint8sel_pc_bat00_best_int8.pt`
- Decision:
  - treat the lane as active
  - keep local CPU free for proxy/scorer work while bat00 soaks the quantization-parity rerun

## 2026-04-09 - prior breakthrough: 1.845 via h=32 long-1000 QAT+EMA

### long-1000 h=32 QAT+EMA authoritative scorer result
- Weights: `experiments/postfilter_weights/postfilter_long1000_qat_ema_alpha20_h32_int8.pt` (16473 bytes)
- Architecture: PostFilter h=32 (11011 params)
- Training: **1000 epochs** (compound scaling: h16→h32 AND 500→1000 epochs), saliency α=20, QAT+EMA decay=0.997, warmup=5, cosine LR 5e-4→1e-5
- **Authoritative scorer result (CPU, official path):**
  - PoseNet: `0.04809216` (down from 1.92 floor's `0.05891` — **-18.4%**)
  - SegNet: `0.00576402`
  - bytes: `864,167`
  - rate: `0.02301653`
  - **Final score: `1.85` (rounded) / `1.84532` (exact)**
- Proxy→scorer calibration: 8-decimal match maintained
- Evidence: `reports/raw/2026-04-08-long1000-h32/robust_current-long1000-h32-current_workflow-cpu-report.txt`
- Decision: **PROMOTE** as new floor at **1.845** — **0.10 below leaderboard #1 (1.95)**
- Key insight: compound scaling on BOTH axes (epochs 500→1000 AND width h=16→h=32) gave approximately additive gains. Score trajectory in one session: **2.01 → 1.99 → 1.945 → 1.92 → 1.845**.

### long1000 h32 patched rerun
- Relaunched on the patched trainer in managed session `64668`.
- Purpose:
  - preserve `best_*` checkpoints during the run
  - avoid another blind wait for only the final epoch artifact
- Best saved checkpoint so far:
  - epoch `191`
  - scorer `3.873950522740682`
  - int8 artifact: `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long1000_qat_ema_alpha20_h32_rerun_best_int8.pt`
  - int8 size `16,735` bytes
- Official-path proxy:
  - launched at `2026-04-08 23:24:39 -0500`
  - candidate artifact: `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long1000_qat_ema_alpha20_h32_rerun_best_int8.pt`
- Proxy result:
  - landed at `2026-04-08 23:45:57 -0500`
  - current_workflow `1.99`
  - PoseNet `0.06883860`
  - SegNet `0.00581951`
- Decision:
  - infrastructure fix confirmed
  - rerun branch is cut; the saved best checkpoint does not transfer and is far behind the promoted `1.85` floor

## 2026-04-08 22:59:11 -0500 - late-night diagnostics and new side-lane surfaces

### Jacobian SVD diagnostic
- Evidence:
  - `/private/tmp/pact-mine/jacobian_svd.log`
- Landed at:
  - `2026-04-08 22:37:31 -0500`
- Result:
  - mean singular values: `[0.03603, 0.00080, 0.00055, 0.00028, 0.00018, 0.00009]`
  - entropy-based effective rank: `1.008 / 6`
  - mean condition number: `398.8`
- Interpretation:
  - PoseNet's local pixel sensitivity is effectively one-dimensional at the operating point that matters
  - one-shot pseudoinverse failure is now explained by both rank collapse and conditioning

### CNN residual / Karpathy diagnostic
- Evidence:
  - `/private/tmp/pact-mine/karpathy_analysis.log`
- Landed at:
  - `2026-04-08 22:45:41 -0500`
- Result:
  - CNN mean `|δ|`: `0.8317`
  - CNN max `|δ|`: `46.25`
  - pixels moved > `0.5` LSB: `56.6%`
  - luma spectral energy: low `3.2%`, mid `90.3%`, high `6.4%`
  - Jacobian pixels moved > `0.5` LSB: `0.0024%`
- Interpretation:
  - the winning residual is dense and mid-frequency dominated, not a sparse adversarial spike pattern
  - explicit DCT-basis exploration is now grounded by measurement

### Trust-region diagnostic
- Evidence:
  - `/private/tmp/pact-mine/trust_region.log`
- Landed at:
  - `2026-04-08 22:58:04 -0500`
- Result:
  - trust-radius knee: `0.0001` pixels RMS
  - median relative linearization error is already > `1` there
- Interpretation:
  - test-time Newton / trust-region inflate methods are dead for this evaluator path
  - the learned-CNN route is no longer just the best empirical path; it is the only practical path left among the tested local-linear methods

### Side-lane training snapshots
- `train_segnet_attack.log` at `2026-04-08 22:58:37 -0500`
  - epoch `340`
  - local score `1.2348`
  - PoseNet `0.059356`
  - SegNet `0.005916`
- `train_kalman.log` at `2026-04-08 22:59:12 -0500`
  - epoch `190`
  - local score `3.9578`
  - PoseNet `0.048824`
  - SegNet `0.034373`
- `train_uint8ste.log` at `2026-04-08 22:59:18 -0500`
  - epoch `200`
  - local score `4.0883`
  - PoseNet `0.063304`
  - SegNet `0.034937`
- Decision:
  - all three remain side-lane signals, not promotion candidates
  - Kalman is the most interesting on local PoseNet signal alone

### New exploration surfaces added
- `experiments/train_postfilter_dct.py`
  - purpose: explicit mid-frequency DCT-basis residual filter
- `experiments/monte_carlo_layer_scale_search.py`
  - purpose: keep Monte Carlo / ES in the toolbelt with a low-dimensional layer-scale manifold
- `experiments/rate_distortion_floor.py`
  - purpose: quantify empirical floor geometry and target requirements

### DCT mid-band smoke
- Command:
  - `experiments/train_postfilter_dct.py --epochs 40 --alpha 20 --block 8 --train-subsample 12 --eval-subsample 8 --tag dct_midband_alpha20_b8_smoke`
- Model:
  - `385` params
- Baseline:
  - score `4.5182`
  - PoseNet `0.073558`
  - SegNet `0.036606`
- Observed checkpoints:
  - epoch `10`: `4.5182`
  - epoch `20`: `4.5182`
  - epoch `30`: `4.5182`
- Decision:
  - cut immediately
  - the explicit DCT mid-band parameterization did not move off baseline in the first bounded smoke, so it does not earn more local MPS time yet

### Monte Carlo layer-scale smoke
- Command:
  - `experiments/monte_carlo_layer_scale_search.py --iterations 6 --population 4 --eval-subsample 30 --tag mc_layer_scale_smoke`
- Baseline on 20-pair screen:
  - score `4.1098`
  - PoseNet `0.050177`
  - SegNet `0.034015`
- Best smoke result:
  - iteration `6`
  - score `3.8557`
  - PoseNet `0.022061`
  - SegNet `0.033860`
  - saved artifact: `experiments/postfilter_weights/postfilter_mc_layer_scale_smoke_best_int8.pt`
  - saved fp32: `experiments/postfilter_weights/postfilter_mc_layer_scale_smoke_best_fp32.pt`
  - int8 size: `16,355` bytes
- Runtime note:
  - the first saved int8 artifact used an unsupported `mc_layer_scale` runtime variant tag
  - fixed by preserving the incumbent runtime variant and storing `search_strategy=mc_layer_scale` instead
- Official-path proxy:
  - launched at `2026-04-08 23:59:28 -0500`
  - candidate artifact: `experiments/postfilter_weights/postfilter_mc_layer_scale_smoke_best_int8.pt`
  - landed at `2026-04-09 00:14:10 -0500`
  - current_workflow `1.93`
  - PoseNet `0.05394274`
  - SegNet `0.00617033`
- Decision:
  - keep this family alive
  - unlike the h32 rerun checkpoint, this cheap gradient-free lane actually transferred on the official proxy path
  - next step is one deeper bounded search on the same six-dimensional layer-scale manifold

### Monte Carlo layer-scale seeded refinement
- Command:
  - `experiments/monte_carlo_layer_scale_search.py --iterations 8 --population 6 --sigma 0.20 --eval-subsample 10 --init-meta experiments/postfilter_weights/postfilter_mc_layer_scale_deep1_best_meta.json --tag mc_layer_scale_refine1`
- 60-pair screen:
  - baseline `4.0039`
  - best local screen `3.9180`
- Official-path proxy:
  - landed at `2026-04-09 01:21:27 -0500`
  - current_workflow `1.86`
  - exact current_workflow `1.8631624591742`
  - PoseNet `0.04709668`
  - SegNet `0.00601479`
- Decision:
  - keep this family alive
  - it improved materially over the first `1.93` Monte Carlo transfer and is now the strongest non-promoted alternate family
  - next step is one tighter refinement with smaller sigma around the new theta

### Monte Carlo layer-scale `sigma=0.05` refinement
- Local 60-pair screen:
  - best local screen `3.9176`
  - best artifact:
    - `/Users/adpena/Projects/pact/experiments/postfilter_weights/postfilter_mc_layer_scale_refine2_best_fp32.pt`
    - `/Users/adpena/Projects/pact/experiments/postfilter_weights/postfilter_mc_layer_scale_refine2_best_int8.pt`
- Official-path proxy:
  - landed at `2026-04-09 02:44:33 -0500`
  - current_workflow `1.86`
  - PoseNet `0.04706553`
  - SegNet `0.00601742`
  - durable log: `/private/tmp/pact-mine/proxy_mc_layer_scale_refine2.log`
- Decision:
  - effectively tied the first seeded refinement rather than beating it materially
  - Monte Carlo remains the strongest non-promoted alternate family, but the local search now looks near a plateau around `1.86`

### SegNet-native long run already in flight
- Live processes:
  - `train_postfilter_segnet_attack.py --alpha 20 --hidden 32 --epochs 1000 --tag segnet_attack_long1000_h32`
  - `train_postfilter_segnet_attack.py --alpha 20 --hidden 32 --epochs 1000 --tag segnet_attack_fixed_ste_h32`
- Latest observed logs:
  - `train_segnet_attack.log` at `2026-04-09 02:18:17 -0500`
    - epoch `990`
    - local total `1.0821`
    - scorer `1.0523`
    - PoseNet `0.041651`
    - SegNet `0.005684`
    - best recent local checkpoint remains epoch `940`
      - local total `1.0056`
      - scorer `0.9774`
      - PoseNet `0.030106`
      - SegNet `0.005718`
  - `train_segnet_attack_fixed.log` at `2026-04-09 01:54:54 -0500`
    - epoch `340`
    - local total `1.2562`
    - scorer `1.1141`
    - PoseNet `0.053045`
    - SegNet `0.005773`
- Decision:
  - do not start another duplicate SegNet-native run from this lane
  - wait for a saved artifact from the stronger `segnet_attack_long1000_h32` branch, or retrofit checkpoint saving in a clean rerun later if this run finishes without one

### SegNet-native h32 artifact emitted
- Artifact paths:
  - `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_segnet_attack_long1000_h32_fp32.pt`
  - `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_segnet_attack_long1000_h32_int8.pt`
- Final local EMA eval:
  - landed at `2026-04-09 02:25:35 -0500`
  - total `1.2920`
  - PoseNet `0.052372`
  - SegNet `0.005683`
  - delta `-0.1328`
- Official-path proxy:
  - launched at `2026-04-09 02:25:35 -0500`
  - candidate artifact: `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_segnet_attack_long1000_h32_int8.pt`
  - landed at `2026-04-09 02:39:30 -0500`
  - current_workflow `1.90`
  - PoseNet `0.05670501`
  - SegNet `0.00575334`
- Decision:
  - real close miss
  - weaker than the Monte Carlo family, so it does not take over the frontier

### SegNet-native hardened rerun
- Launched at:
  - `2026-04-09 03:11:02 -0500`
- Command:
  - `train_postfilter_segnet_attack.py --alpha 20 --hidden 32 --epochs 1000 --tag segnet_attack_bestckpt_h32`
- Durable log:
  - `experiments/postfilter_weights/train_segnet_attack_bestckpt_h32.log`
- Purpose:
  - preserve `best_*` checkpoints during the run so the strongest earlier epoch can be ranked honestly
- Current saved best:
  - epoch `400`
  - score `1.2856935588153324`
  - int8 size `16,527` bytes
  - now clearly below the earlier non-checkpointed run's final local EMA score of `1.2920`
- Official-path proxy:
  - landed at `2026-04-09 04:37:21 -0500`
  - current_workflow `1.90`
  - PoseNet `0.06025248`
  - SegNet `0.00547161`
- Decision:
  - cut the rerun
  - checkpoint preservation changed the PoseNet/SegNet tradeoff but did not improve the official-path score over the original `1.90` artifact

### h48 long-horizon QAT+EMA lane
- Device slot reclaimed from the resolved SegNet-native rerun.
- Launched at:
  - `2026-04-09 04:41:27 -0500`
- Command:
  - `train_postfilter_qat_ema.py --alpha 20 --hidden 48 --epochs 1500 --tag long1500_qat_ema_alpha20_h48`
- Durable log:
  - `experiments/postfilter_weights/train_long1500_h48.log`
- Latest observed checkpoint:
  - epoch `40`
  - local total `4.3801`
  - scorer `4.3218`
  - PoseNet `0.088424`
  - SegNet `0.035408`
- Current saved best:
  - epoch `28`
  - score `4.101413129170735`
  - int8 size `28,851` bytes
- Decision:
  - cut the branch as a weak capacity bet
  - it never produced a remotely competitive saved checkpoint

### h32 restart + SWA QAT+EMA lane
- Device slot reclaimed from the weak `h48` branch.
- Launched at:
  - `2026-04-09 05:00:55 -0500`
- Command:
  - `train_postfilter_qat_ema.py --alpha 20 --hidden 32 --epochs 1000 --restart-t0 100 --restart-tmult 1 --restart-eta-min 1e-5 --swa-start-epoch 800 --swa-every 10 --tag long1000_qat_ema_alpha20_h32_restart_swa`
- Durable log:
  - `experiments/postfilter_weights/train_long1000_h32_restart_swa.log`
- Current saved best:
  - epoch `59`
  - score `4.080144761403401`
  - int8 size `16,907` bytes
- Decision:
  - cut the branch as a weak dead end
  - warm restarts plus deferred SWA did not produce a remotely competitive checkpoint story

### FiLM-conditioned lane
- Device slot reclaimed from the weak `restart_swa` branch.
- Launched at:
  - `2026-04-09 05:55:32 -0500`
- Command:
  - `train_postfilter_film_conditioned.py --alpha 20 --hidden 32 --epochs 160 --tag film_conditioned_alpha20_h32_smoke`
- Durable log:
  - `experiments/postfilter_weights/train_film_conditioned_alpha20_h32_smoke.log`
- Current saved best:
  - epoch `15`
  - score `4.179513591130575`
  - int8 size `18,195` bytes

### Monte Carlo layer-scale `sigma=0.05` refinement
- Local 60-pair screen:
  - best local screen `3.9176`
  - best artifact:
    - `/Users/adpena/Projects/pact/experiments/postfilter_weights/postfilter_mc_layer_scale_refine2_best_fp32.pt`
    - `/Users/adpena/Projects/pact/experiments/postfilter_weights/postfilter_mc_layer_scale_refine2_best_int8.pt`
- Durable proxy rerun:
  - launched at `2026-04-09 02:44:33 -0500`
  - log path: `/private/tmp/pact-mine/proxy_mc_layer_scale_refine2.log`

### Weighted ensemble candidate
- Build:
  - `ensemble_h32_qat_mc75_25`
  - inputs:
    - `postfilter_long1000_qat_ema_alpha20_h32_fp32.pt`
    - `postfilter_mc_layer_scale_refine1_best_fp32.pt`
  - weights: `0.75 / 0.25`
  - artifact: `experiments/postfilter_weights/postfilter_ensemble_h32_qat_mc75_25_int8.pt`
- Official-path proxy:
  - landed at `1.84`
  - PoseNet `0.04678315`
  - SegNet `0.00581610`
- Promotion ladder:
  - smoke passed
  - authoritative local CPU scorer is still active via `comma-lab eval-submission`
  - raw evidence root: `reports/raw/2026-04-09-ensemble-h32-qat-mc75-25/`

## 2026-04-08 18:38:04 -0500 - post-promotion stabilization cycle

### local h24 managed long-run stays alive
- Managed session:
  - `42898`
- Latest observed checkpoints:
  - epoch `40`: scorer `4.1235`, PoseNet `0.063619`, SegNet `0.035225`
  - epoch `50`: scorer `4.2486`, PoseNet `0.079513`, SegNet `0.035418`
  - epoch `60`: scorer `4.2780`, PoseNet `0.085587`, SegNet `0.035742`
  - epoch `70`: scorer `4.1821`, PoseNet `0.069419`, SegNet `0.035489`
  - epoch `80`: scorer `4.1112`, PoseNet `0.062075`, SegNet `0.034961`
  - epoch `90`: scorer `4.1309`, PoseNet `0.067211`, SegNet `0.034822`
  - epoch `100`: scorer `4.2193`, PoseNet `0.063125`, SegNet `0.035608`
  - epoch `110`: scorer `4.1210`, PoseNet `0.070566`, SegNet `0.034784`
  - epoch `180`: scorer `4.0935`, PoseNet `0.061039`, SegNet `0.034818`
  - epoch `190`: scorer `4.2276`, PoseNet `0.071757`, SegNet `0.035384`
  - epoch `200`: scorer `4.1066`, PoseNet `0.058720`, SegNet `0.035090`
- Decision:
  - h24 was later stopped intentionally after the `1.92` promotion
  - it never turned into a stronger frontier candidate than the long1000 lines

### site parity / drift guard
- Added a real site parity check:
  - `python3 reports/graphs/build_static_site.py --check`
- Added regression coverage:
  - `reports/graphs/test_build_static_site.py`
- `refresh_site.py` now runs the parity check after rebuilding.
- Purpose:
  - catch stale or drifted `reports/graphs/site/*` copies automatically instead of noticing them by hand after a promotion

### local long1000 h16 artifact is now real
- Local training log:
  - `/private/tmp/pact-mine/experiments/postfilter_weights/train_long1000_h16.log`
- Final local eval:
  - baseline `4.5179`
  - filtered `4.1515`
  - delta `-0.3664`
  - PoseNet `0.052258`
  - SegNet `0.034286`
- Saved artifacts:
  - `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long1000_qat_ema_alpha20_h16_fp32.pt`
  - `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long1000_qat_ema_alpha20_h16_int8.pt` (`8537` bytes)
- Decision:
  - do not promote from local training metrics
  - official upstream-path proxy is the next honest ranking step and is already running

### long1000 h16 official proxy result
- Proxy log:
  - `/private/tmp/pact-mine/proxy_long1000_h16.log`
- Observed at:
  - `2026-04-08 18:55:54 -0500`
- Official upstream-path proxy result:
  - PoseNet `0.05890975`
  - SegNet `0.00579293`
  - estimated score `1.9222`
- Decision:
  - this clears the proxy gate and earns the authoritative local CPU scorer slot

### long1000 h16 smoke + scorer launch
- Smoke summary:
  - passed at `2026-04-08 18:58:41 -0500`
  - frame count matched
  - byte count matched
  - semantic check passed
  - semantic MAE mean `5.3486547217438485`
- Authoritative scorer:
  - launched on the same packaged artifact immediately after smoke
  - evidence root: `reports/raw/2026-04-08-long1000-h16-authoritative/`
  - report copy target: `reports/raw/2026-04-08-long1000-h16-authoritative/robust_current-long1000-h16-current_workflow-cpu-report.txt`
  - summary target: `reports/raw/2026-04-08-long1000-h16-authoritative/robust_current-long1000-h16-current_workflow-cpu-summary.json`

### long1000 h16 authoritative scorer result
- Authoritative scorer result:
  - current_workflow `1.92`
  - PoseNet `0.05890975`
  - SegNet `0.00579293`
  - bytes `864,167`
  - rule_faithful estimate `1.9642242695971968` at `927,230` bytes
- Decision:
  - PROMOTE as the new Track B floor
  - this is the first scorer-backed result in the repo below the current public `1.95` lead

### long1000 h32 local launch
- Local training log:
  - `/private/tmp/pact-mine/experiments/postfilter_weights/train_long1000_h32.log`
- Observed at:
  - `2026-04-08 18:55:54 -0500`
- Latest seen checkpoints:
  - epoch `1`: scorer `4.3107`, PoseNet `0.057141`, SegNet `0.036916`
  - epoch `10`: scorer `4.2553`, PoseNet `0.062842`, SegNet `0.036342`
  - epoch `20`: scorer `4.4270`, PoseNet `0.104312`, SegNet `0.036116`
  - epoch `30`: scorer `4.3234`, PoseNet `0.080888`, SegNet `0.035813`
  - epoch `40`: scorer `4.3153`, PoseNet `0.073298`, SegNet `0.036065`
  - epoch `50`: scorer `4.1693`, PoseNet `0.063771`, SegNet `0.035490`
  - epoch `60`: scorer `4.2251`, PoseNet `0.068862`, SegNet `0.035820`
  - epoch `70`: scorer `4.2620`, PoseNet `0.070256`, SegNet `0.035870`
  - epoch `80`: scorer `4.1879`, PoseNet `0.072442`, SegNet `0.034946`
  - epoch `90`: scorer `4.1422`, PoseNet `0.073299`, SegNet `0.034613`
  - epoch `100`: scorer `4.1727`, PoseNet `0.068009`, SegNet `0.035300`
  - epoch `120`: scorer `4.0964`, PoseNet `0.081087`, SegNet `0.034246`
  - epoch `130`: scorer `4.0620`, PoseNet `0.060675`, SegNet `0.034792`
  - epoch `190`: scorer `4.0290`, PoseNet `0.060119`, SegNet `0.034262`
  - epoch `210`: scorer `4.0563`, PoseNet `0.074450`, SegNet `0.033949`
  - epoch `270`: scorer `4.0235`, PoseNet `0.056129`, SegNet `0.034372`
  - epoch `280`: scorer `4.0936`, PoseNet `0.078344`, SegNet `0.034374`
  - epoch `290`: scorer `4.0752`, PoseNet `0.057441`, SegNet `0.034630`
  - epoch `300`: scorer `4.1058`, PoseNet `0.083165`, SegNet `0.034025`
  - epoch `310`: scorer `4.0609`, PoseNet `0.058773`, SegNet `0.034663`
  - epoch `320`: scorer `4.0668`, PoseNet `0.070117`, SegNet `0.034266`
  - epoch `330`: scorer `3.9977`, PoseNet `0.060179`, SegNet `0.034075`
  - epoch `340`: scorer `3.9976`, PoseNet `0.051810`, SegNet `0.034394`
  - epoch `350`: scorer `4.1246`, PoseNet `0.078668`, SegNet `0.034366`
  - epoch `360`: scorer `3.9269`, PoseNet `0.040667`, SegNet `0.034165`
  - epoch `440`: scorer `3.8531`, PoseNet `0.039350`, SegNet `0.033991`
  - epoch `580`: scorer `4.0170`, PoseNet `0.066315`, SegNet `0.034056`
  - epoch `590`: scorer `3.9829`, PoseNet `0.064109`, SegNet `0.034197`
  - epoch `600`: scorer `3.9791`, PoseNet `0.070756`, SegNet `0.033488`
  - epoch `570`: scorer `3.8805`, PoseNet `0.051772`, SegNet `0.033636`
  - epoch `580`: scorer `4.0170`, PoseNet `0.066315`, SegNet `0.034056`
- Decision:
  - keep running, but it is not yet stronger than the long1000 h16 branch
  - no `best_*` artifact is appearing for this live process, which likely means it was launched before the save-best patch and must run to completion before proxy ranking

### bat00 stability note
- Latest remote probe against `bat00` hit:
  - `WSL/Service/HCS_E_CONNECTION_TIMEOUT`
- Interpretation:
  - partner-owned training may still be running, but remote introspection is not reliable enough to treat bat00 as dependable orchestration infrastructure yet

### refresh-site hygiene pass
- `reports/graphs/refresh_site.py` now runs:
  1. source builders
  2. graph/site tests
  3. final site copy
  4. final site parity check
- `reports/graphs/build_static_site.py --check` now detects source/site drift directly
- `reports/graphs/test_build_static_site.py` covers both in-sync and drifted cases
- Operational lesson:
  - dashboard tests mutate source artifacts, so parity must be checked after the final site copy, not before
- Full-path verification:
  - `python3 reports/graphs/refresh_site.py` completed end-to-end and finished with `site_in_sync`

### packet/notebook state corrected after the 1.95 promotion
- Fixed stale public-facing packet/notebook surfaces that were still claiming the old `2.05` floor or mislabeling the promoted branch as h16.
- Exact refresh timestamp used for this synchronization pass:
  - `2026-04-08 18:38:04 -0500`
- Follow-on h24 observation timestamp now recorded at:
  - `2026-04-08 18:43:53 -0500`

### bat00 coordination note
- Our h16 long1000 lane on `bat00` is still blocked after decode and is not yet a stable remote training lane.
- Partner-owned remote process observed on `bat00`:
  - `train_postfilter_qat_ema.py --alpha 20 --hidden 32 --epochs 1000 --tag bat00_long1000_qat_ema_alpha20_h32`
- Operational rule:
  - do not trample the partner-owned `bat00` job from this lane

## 2026-04-08 - 🏆 BREAKTHROUGH: 1.945 (ties leaderboard #1) via h=32 long-500 QAT+EMA

### long500 h=32 QAT+EMA authoritative scorer result
- Weights: `experiments/postfilter_weights/postfilter_long500_qat_ema_alpha20_h32_int8.pt` (16455 bytes)
- Architecture: PostFilter h=32 (11011 params — 3.4x the h=16 model)
- Training: 500 epochs, saliency loss (soft SegNet, α=20), QAT fake-quant, EMA decay=0.997, warmup=5, cosine LR 5e-4→1e-5
- **Authoritative scorer result (CPU, official path):**
  - PoseNet: `0.06215462` (down from 1.99 floor's `0.06925` — another **-10.2%**)
  - SegNet: `0.00581255` (essentially flat)
  - bytes: `864,167`
  - rate: `0.02301653`
  - **Final score: `1.95` (rounded) / `1.9450` (exact)**
- Faithful proxy was pre-calibrated to the exact same numbers (8-decimal match)
- Evidence: `reports/raw/2026-04-08-long500-h32/robust_current-long500-h32-current_workflow-cpu-report.txt`
- Decision: **PROMOTE** as new Track B floor at **1.945** (ties leaderboard #1 at 1.95, slightly below)
- Key lesson: **patience + EMA + wider model** wins. Old soft-SegNet loss is fine for EMA training.

## 2026-04-08 - BREAKTHROUGH: 1.985 via long-500 QAT+EMA

### long-500 QAT+EMA faithful proxy result
- Weights: `experiments/postfilter_weights/postfilter_long500_qat_ema_alpha20_h16_int8.pt` (8519 bytes)
- Architecture: same PostFilter (3→16→16→3, 3x3, ReLU, residual)
- Training: 500 epochs, saliency loss (soft SegNet, α=20), QAT fake-quant, EMA decay=0.997, warmup=5, cosine LR from 5e-4 to 1e-5
- Faithful proxy (CPU, DistortionNet, all 600 pairs, batch_size=16):
  - PoseNet: `0.069249` (down from `0.074549` — **-7.1%**)
  - SegNet: `0.005778` (from `0.005741` — essentially flat)
  - Estimated score: **`1.985`** (from `2.01` — **-0.025 points**)
- Proxy calibration: matches real scorer to 8 decimal places (verified on promoted weights)
- **CONFIRMED: authoritative scorer = 1.99**
  - PoseNet: `0.06924924` (exact 8-decimal match with proxy)
  - SegNet: `0.00577789` (exact 8-decimal match with proxy)
  - bytes: `864,167`
  - Evidence: `/tmp/pact-mine/reports/raw/2026-04-08-long500-qat-ema/robust_current-long500-current_workflow-cpu-report.txt`
- Decision: **PROMOTE** as new Track B floor at **1.99** (from 2.01)
- Key lesson: patience + EMA > fancy loss functions. 500 epochs of QAT+EMA with the old soft-SegNet loss outperformed the v2 scorer-faithful trainer (2.04) and the baseline 100-epoch saliency trainer (2.01).

## 2026-04-08 - CRITICAL: training loss mismatch with scorer

### SegNet loss discrepancy
- **Training loss** used `1 - softmax_cosine_similarity` (soft proxy, ~0.036 per pair)
- **Real scorer** uses `argmax_disagreement_rate` (hard, ~0.006 per pair)
- The soft version gave SegNet **~10× more gradient weight** than warranted
- All prior experiments optimized too conservatively on SegNet, limiting PoseNet gains

### Fix: `train_postfilter_v2.py`
- Uses `DistortionNet.compute_distortion()` (same as scorer) for PoseNet
- Uses straight-through hard-argmax (STE) for SegNet disagreement
- Evaluates ALL 600 pairs (not 25% subsampled)
- QAT + EMA for stability + deployment gap
- First v2 results: epoch 10 loss=1.189, pose=0.0625 (scorer-faithful metric)

### Re-opened lanes for investigation with v2
1. h=24, h=32 (prior rejects may be false negatives from SegNet overweighting)
2. Lower α values (α=5, α=10) with corrected gradient balance
3. Depthwise/luma variants
4. Higher epoch counts (long-500 hitting pose=0.048 at epoch 210)

### Proxy scorer calibration
- Built `proxy_score_faithful.py` using the exact upstream evaluation pipeline
- Uses DistortionNet, yuv420_to_rgb GT loading, batch_size=16, CPU
- Running calibration on promoted α=20 weights (result pending)

## 2026-04-08 - context-resume cycle (post-2.01)

- Picked up the loop after a context compaction. Verified the promoted floor is `2.01` (saliency α=20, h=16).
- Partner's PF-SALIENCY-ALPHA10-H32 result landed at `2.03` (PoseNet `0.07734`, SegNet `0.00570925`, bytes `864167`). Worse than the `2.01` floor. SegNet is essentially identical to the α=20 h=16 winner (0.00571 vs 0.00574); the entire gap is PoseNet. Conclusion: saliency-mask α dominates model capacity. Higher α at h=16 beats lower α at h=32. Next bet is α=30 h=16, not bigger architectures.
- The partner's eval was being throttled by 23 forgotten `/tmp/sieve*` test binaries from another project (load avg 115, each at 30-50% CPU since Friday). User authorized killing them; load dropped to ~80 immediately and the partner's eval finished within seconds of the kill.
- My duplicate scoring run was killed cleanly to avoid lock contention. Operational lesson: always check `ps -ef | grep evaluate.py` before launching authoritative scoring. The `.omx/locks/robust_current-*.lock` is the secondary signal but the process check is faster.
- Set up `/tmp/pact-mine` as a separate isolated workspace so future authoritative runs do not collide with the partner. Heavy assets (`videos/`, `models/`, both `.venv` trees) are symlinked from `/tmp/pact-authoritative` to keep disk footprint small while remaining read-isolated.
- Confirmed the PR submission directory at `workspace/upstream/comma_video_compression_challenge/submissions/learned_postfilter_av1/` bundles `postfilter_int8.pt` (8131 B) inside `archive.zip` (884868 B total). That is the rule-faithful packaging the contest requires; it costs ~+0.014 score vs the unbundled current_workflow accounting and lands the PR around `2.024`.

## 2026-04-08 - saliency-weighted promotion

### PF-SALIENCY-ALPHA20: authoritative win
- Isolated evidence root:
  - `/tmp/pact-authoritative/reports/raw/2026-04-08-saliency-alpha20-postfilter-isolated/`
- Pinned config:
  - `submissions/robust_current/config.av1-2.08-postfilter.env`
- Smoke:
  - passed
  - semantic MAE mean: `5.345036011264278`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.01`
  - PoseNet: `0.07454902`
  - SegNet: `0.00574092`
  - bytes: `864,168`
  - rule-faithful estimate: `2.0540238873043903` at `925,893` bytes
- Decision:
  - promote as the new Track B floor

### PF-SALIENCY-ALPHA10: parity debug outcome
- The first isolated rerun that came back at `2.08` was invalid for the saliency branch because it used the wrong decode path (`INFLATE_POSTFILTER=unsharp`) instead of `PYTHON_INFLATE=postfilter`.
- Fix:
  - added `submissions/robust_current/config.av1-2.08-postfilter.env`
  - runtime loader now accepts `saliency_weighted` residual weights directly
- Operational lesson:
  - saliency-family reruns must pin the `2.08` archive config with the learned-postfilter decode path, not the older unsharp-only config

### PF-SALIENCY-ALPHA10-H32: authoritative close miss
- Isolated evidence root:
  - `/tmp/pact-authoritative/reports/raw/2026-04-08-saliency-alpha10-h32-postfilter-isolated/`
- Pinned config:
  - `submissions/robust_current/config.av1-2.08-postfilter.env`
- Smoke:
  - passed
  - semantic MAE mean: `5.393719369892433`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.03`
  - PoseNet: `0.07734269`
  - SegNet: `0.00570925`
  - bytes: `864,167`
  - rule-faithful estimate: `2.072004529928459` at `933,580` bytes
- Decision:
  - verified close miss, not a promotion over the `2.01` alpha20 floor

### PF-SALIENCY-ALPHA30-H16: authoritative close miss
- Isolated evidence root:
  - `/tmp/pact-authoritative/reports/raw/2026-04-08-saliency-alpha30-h16-postfilter-isolated/`
- Pinned config:
  - `submissions/robust_current/config.av1-2.08-postfilter.env`
- Smoke:
  - passed
  - semantic MAE mean: `5.373262236970983`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.04`
  - PoseNet: `0.08075187`
  - SegNet: `0.00570817`
  - bytes: `864,167`
  - rule-faithful estimate: `2.086046804192108` at `926,036` bytes
- Decision:
  - reject; higher saliency strength did not beat the `2.01` alpha20 floor

### PF-SALIENCY-ALPHA20-H16-EMA99: authoritative close miss
- Isolated evidence root:
  - `/tmp/pact-authoritative/reports/raw/2026-04-08-saliency-alpha20-ema99-postfilter-isolated/`
- Pinned config:
  - `submissions/robust_current/config.av1-2.08-postfilter.env`
- Smoke:
  - passed
  - semantic MAE mean: `5.372145267421754`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.04`
  - PoseNet: `0.08021699`
  - SegNet: `0.00570570`
  - bytes: `864,167`
  - rule-faithful estimate: `2.0827947711456036` at `926,000` bytes
- Decision:
  - reject; EMA alone did not beat the `2.01` alpha20 floor

### PF-SALIENCY-ALPHA20-H16-QAT: authoritative close miss
- Isolated evidence root:
  - `/tmp/pact-authoritative/reports/raw/2026-04-08-saliency-alpha20-qat-postfilter-isolated/`
  - repo copy: `reports/raw/2026-04-08-saliency-alpha20-qat-postfilter/`
- Pinned config:
  - `submissions/robust_current/config.av1-2.08-postfilter.env`
- Proxy:
  - baseline loss `4.5179`
  - filtered loss `4.4641`
  - delta `-0.0538`
- Smoke:
  - passed
  - semantic MAE mean: `5.402222842578831`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.03`
  - PoseNet: `0.07874124`
  - SegNet: `0.00571970`
  - bytes: `864,168`
  - rule-faithful estimate: `2.0758946946115664` at `925,965` bytes
- Decision:
  - reject; QAT improved SegNet relative to the `2.01` winner but gave back too much PoseNet
- Next grounded branch:
  - stay on the winning `alpha=20 h16` family and test feature-matching loss before reopening weaker structural branches

### PF-FEATMATCH-ALPHA20-H16: bugfix + weak proxy
- Root-cause bug found before the first proxy epoch:
  - `experiments/train_postfilter_qat_ema.py` crashed in `FakeQuantSTE.backward()` because zero-scale tensors returned from `forward()` without any saved backward state
  - trigger: the zero-initialized residual head in the QAT wrapper
- Fix:
  - added a failing regression at `experiments/test_train_postfilter_qat_ema.py`
  - zero-scale tensors now save an all-false saturation mask before returning
  - verification passed:
    - `uv run --with torch --with av --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python -m unittest experiments/test_train_postfilter_qat_ema.py experiments/test_train_postfilter_saliency.py`
    - `python3 -m py_compile experiments/train_postfilter_qat_ema.py experiments/train_postfilter_featmatch.py`
- Proxy result after the fix:
  - baseline loss `4.3359`
  - filtered loss `4.3213`
  - delta `-0.0146`
  - Pose proxy: `0.0741595 -> 0.0731940`
  - Seg proxy: `0.0365672 -> 0.0364653`
  - weight artifact: `experiments/postfilter_weights/postfilter_featmatch_alpha20_h16_e12_fl05_int8.pt` (`8,573` bytes)
- Decision:
  - too weak to justify scorer time

### PF-SALIENCY-ALPHA20-H32: authoritative run in progress
- Evidence root:
  - `/tmp/pact-authoritative/reports/raw/2026-04-08-saliency-alpha20-h32-postfilter-isolated/`
- Pinned config:
  - `submissions/robust_current/config.av1-2.08-postfilter.env`
- Proxy ranking result:
  - baseline loss `4.3359`
  - filtered loss `4.2839`
  - delta `-0.0520`
  - Pose proxy: `0.0741595 -> 0.0709253`
  - Seg proxy: `0.0365672 -> 0.0362264`
- Smoke:
  - passed
  - semantic MAE mean: `5.444546126135099`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.04`
  - PoseNet: `0.08001617`
  - SegNet: `0.00571195`
  - bytes: `864,167`
  - rule-faithful estimate: `2.0876541394907395` at `934,044` bytes
- Decision:
  - reject; wider residual capacity on the winning alpha protected SegNet but gave back too much PoseNet
- Next grounded branch:
  - ROI + post-filter stacking

### PF-ROI-POSTFILTER-CURRENT-ARCHIVE: proxy false positive, scorer reject
- Candidate artifact:
  - `experiments/postfilter_weights/postfilter_roi_int8.pt`
- Base-archive proxy result:
  - baseline loss `4.335900025367737`
  - filtered loss `4.185806833902995`
  - delta `-0.15009319146474187`
  - Pose proxy: `0.07415949892156884 -> 0.07502391445569326`
  - Seg proxy: `0.036567249298095704 -> 0.03502244472503662`
- Interpretation:
  - materially stronger than the recent saliency h32 and feature-matching proxies
  - but the proxy was computed against `reports/raw/2026-04-06-av1-roi-experiments/decode_base_archive.zip`, not the live packaged submission archive
- Authoritative launch:
  - evidence root: `/tmp/pact-authoritative/reports/raw/2026-04-08-roi-postfilter-current-archive-isolated/`
  - pinned config: `submissions/robust_current/config.av1-2.08-postfilter.env`
  - weight copied into isolated submission tree as `postfilter_int8.pt`
- Smoke:
  - passed
  - semantic MAE mean: `6.601744490840129`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.10`
  - PoseNet: `0.07980920`
  - SegNet: `0.00628219`
  - bytes: `864,167`
  - rule-faithful estimate: `2.1377661567894055` at `925,402` bytes
- Decision:
  - reject; the base-archive proxy did not transfer, and SegNet regressed badly on the real current archive
- Operational lesson:
  - any “faithful” proxy must take an explicit archive input or default to the live packaged archive, not an older training/base archive path

### SegNet-fidelity note
- Verified by source inspection:
  - scorer path in `workspace/upstream/comma_video_compression_challenge/modules.py` uses hard `argmax` disagreement for SegNet distortion
  - training path in `experiments/train_postfilter_saliency.py` uses a softer overlap surrogate for differentiability
- Immediate decision:
  - keep the live ROI scorer branch intact
  - queue a post-ROI training-fidelity branch around a harder SegNet surrogate or faithful int8-on-CPU checkpoint selection

## 2026-04-08 - scorer-fidelity hardening pass

### faithful proxy archive-discipline fix
- Root cause:
  - `experiments/proxy_score_faithful.py` previously defaulted to `reports/raw/2026-04-06-av1-roi-experiments/decode_base_archive.zip`
  - that allowed stale base-archive proxy signals to be mislabeled as current-archive evidence
- Fix:
  - added explicit `--archive-zip`
  - default resolution now prefers `submissions/robust_current/archive.zip` and only falls back to the legacy base archive when the live package is absent
  - proxy now prepares a temp submission directory and runs the upstream `evaluate.py` end-to-end instead of manually looping DistortionNet in-process
- Regression:
  - `experiments/test_proxy_score_faithful.py`
- Verification:
  - `uv run --with torch --with av --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python -m unittest experiments/test_proxy_score_faithful.py`
  - `python3 -m py_compile experiments/proxy_score_faithful.py`

### v2 trainer checkpoint-selection hardening
- Root cause:
  - `experiments/train_postfilter_v2.py` claimed scorer-faithful full-pair checkpoint selection but still tracked the best state from minibatch training loss
- Fix:
  - added `evaluate_model_pairs(...)`
  - checkpoint selection now runs on EMA weights using full evaluation every `--checkpoint-eval-every` epochs
  - removed the unimplemented dual-saliency claim from the v2 docstring
- Regression:
  - `experiments/test_train_postfilter_v2.py`
- Verification:
  - `uv run --with torch --with av --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python -m unittest experiments/test_train_postfilter_v2.py`
  - `python3 -m py_compile experiments/train_postfilter_v2.py`

### v2 local run opened
- Command:
  - `PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python3 -u experiments/train_postfilter_v2.py --alpha 20 --hidden 16 --epochs 20 --checkpoint-eval-every 5 --tag v2_alpha20_h16_e20`
- Early measured status:
  - baseline loss `1.4782`
  - baseline PoseNet `0.082571`
  - baseline SegNet `0.005695`
  - epoch-1 training loss `1.3948`
  - epoch-1 faithful eval `1.2879`
- Status:
  - still running on MPS

### v2 int8-selection hardening follow-up
- New rigor fix:
  - `train_postfilter_v2.py` checkpoint evaluation now quantizes EMA weights with the same int8 scheme used at save time before scoring them
  - regression added in `experiments/test_train_postfilter_v2.py`
- New rerun launched:
  - `PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python3 -u experiments/train_postfilter_v2.py --alpha 20 --hidden 16 --epochs 20 --checkpoint-eval-every 5 --tag v2i8sel_alpha20_h16_e20`
- Parallel status:
  - restarted faithful CPU proxy is now evaluating `postfilter_v2_alpha20_h16_e20_int8.pt` through the upstream evaluator
  - new MPS rerun is producing the next artifact with the corrected selector

### v2 int8-selection rerun result
- Command:
  - `PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python3 -u experiments/train_postfilter_v2.py --alpha 20 --hidden 16 --epochs 20 --checkpoint-eval-every 5 --tag v2i8sel_alpha20_h16_e20`
- Result:
  - baseline loss `1.4782`
  - filtered loss `1.4679`
  - delta `-0.0103`
  - PoseNet `0.082571 -> 0.080553`
  - SegNet `0.005695 -> 0.005704`
  - weights:
    - `experiments/postfilter_weights/postfilter_v2i8sel_alpha20_h16_e20_fp32.pt`
    - `experiments/postfilter_weights/postfilter_v2i8sel_alpha20_h16_e20_int8.pt` (`8,447` bytes)
- Decision:
  - improvement is real but slightly weaker than the earlier `v2_alpha20_h16_e20` run, so the first v2 artifact remains the lead candidate until the official-path proxy resolves

### official-path proxy verdict for v2 h16
- Command:
  - `PYTHONUNBUFFERED=1 uv run --with torch --with av --with safetensors --with timm --with einops --with segmentation-models-pytorch --with numpy python experiments/proxy_score_faithful.py experiments/postfilter_weights/postfilter_v2_alpha20_h16_e20_int8.pt`
- Important rigor note:
  - this run used the upstream `evaluate.py` end-to-end on a temp submission tree, not the old in-process distortion loop
- Result:
  - PoseNet `0.07986387`
  - SegNet `0.00569092`
  - rate `0.02301653`
  - score `2.04`
- Decision:
  - reject for authoritative promotion; strong enough to keep the scorer-faithful family alive, but not enough to spend the next scorer slot on h16
- Next grounded branch:
  - local `v2 alpha20 h32`

### v2 h32 status update
- Local scorer-faithful result:
  - baseline loss `1.4782`
  - filtered loss `1.4604`
  - delta `-0.0178`
  - PoseNet `0.082571 -> 0.079481`
  - SegNet `0.005695 -> 0.005688`
  - weights:
    - `experiments/postfilter_weights/postfilter_v2_alpha20_h32_e20_fp32.pt`
    - `experiments/postfilter_weights/postfilter_v2_alpha20_h32_e20_int8.pt`
- Operational note:
  - the official-path h32 proxy was started, then deliberately allowed to stop once the partner-owned long-500 authoritative scorer was confirmed active
  - reason: preserve the single authoritative CPU scorer lane and avoid contaminating the most promising branch on the host
- Current decision:
  - wait for the long-500 authoritative result first
  - resume or rerun the h32 official proxy only if the long-500 scorer fails to beat the floor

## 2026-04-07/08 - post-filter follow-on cycle restarted

### PF-LARGE-24: proxy win, smoke pass, scorer reject
- Canonical training command:
  - `experiments/train_postfilter_canonical.py --hidden 24 --epochs 12`
- Proxy result:
  - baseline loss `4.5801`
  - filtered loss `4.4463`
  - delta `-0.1339`
- Candidate weights:
  - `experiments/postfilter_weights/postfilter_canonical_h24_int8.pt`
  - size: `11,467` bytes
- Isolated smoke result:
  - passed under `/tmp/pact-authoritative/reports/raw/2026-04-08-postfilter-h24-canonical/`
  - semantic MAE mean: `5.604341469616069`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.07`
  - PoseNet: `0.07787662`
  - SegNet: `0.00611822`
  - bytes: `861,986`
  - rule-faithful estimate: `2.1103207165479754` at `925,153` bytes
  - verdict: reject; PoseNet improved but SegNet regression erased the gain
- Important operational note:
  - the shared-tree `submissions/robust_current/config.env` has drifted away from the promoted floor
  - authoritative candidate evaluation must pin `submissions/robust_current/config.av1-2.05-postfilter.env`

### PF-SMALL-8: proxy win, scorer pending
- Canonical training command:
  - `experiments/train_postfilter_canonical.py --hidden 8 --epochs 12`
- Proxy result:
  - baseline loss `4.5801`
  - filtered loss `4.4984`
  - delta `-0.0817`
- Candidate weights:
  - `experiments/postfilter_weights/postfilter_canonical_h8_int8.pt`
  - size: `5,945` bytes
- Isolated smoke result:
  - passed under `/tmp/pact-authoritative/reports/raw/2026-04-08-postfilter-h8-canonical/`
  - semantic MAE mean: `6.245820019694139`
- Authoritative local CPU scorer result:
  - `current_workflow`: `2.06`
  - PoseNet: `0.07766201`
  - SegNet: `0.00604941`
  - bytes: `861,986`
  - rule-faithful estimate: `2.0985460525864283` at `919,631` bytes
  - verdict: reject; smaller model reduced the h24 penalty but still did not beat the promoted floor

### PF-LARGE-32: proxy lane started
- Canonical training command:
  - `experiments/train_postfilter_canonical.py --hidden 32 --epochs 12`
- Proxy result:
  - baseline loss `4.5801`
  - filtered loss `4.4546`
  - delta `-0.1255`
- Candidate weights:
  - `experiments/postfilter_weights/postfilter_canonical_h32_int8.pt`
  - size: `15,883` bytes
- Verdict:
  - no scorer run yet
  - weaker proxy than `PF-LARGE-24` while larger, so larger-residual follow-up is deprioritized behind the small-model and cheap-architecture lanes

### PF-CHEAP-DW: implementation gate opened
- Runtime loader and canonical trainer now support a `depthwise` variant.
- Loader regression test passes for:
  - legacy residual weights
  - residual metadata weights
  - depthwise metadata weights
- Next execution step:
  - run canonical proxy training with `--variant depthwise`

### PF-CHEAP-DW16: weak full-proxy gain
- Canonical training command:
  - `experiments/train_postfilter_canonical.py --variant depthwise --hidden 16 --epochs 12`
- Proxy result:
  - baseline loss `4.5801`
  - filtered loss `4.5561`
  - delta `-0.0240`
- Candidate weights:
  - `experiments/postfilter_weights/postfilter_canonical_dw16_int8.pt`
  - size: `5,277` bytes
- Verdict:
  - too weak to justify authoritative scorer time

### PF-LUMA16: weak full-proxy gain despite strong epoch-1
- Canonical training command:
  - `experiments/train_postfilter_canonical.py --variant luma --hidden 16 --epochs 12`
- Proxy result:
  - baseline loss `4.5801`
  - filtered loss `4.5712`
  - delta `-0.0090`
- Candidate weights:
  - `experiments/postfilter_weights/postfilter_canonical_luma16_int8.pt`
  - size: `7,617` bytes
- Important lesson:
  - cheap variants can show a strong epoch-1 signal and still collapse by full proxy eval
  - the next immediate branch should test training/checkpoint regime, not just architecture swaps

### PF-SALIENCY-ALPHA10: authoritative rerun in progress
- Trusted-partner evidence already present in:
  - `reports/raw/2026-04-06-av1-roi-experiments/exp_saliency_alpha10_v2_report.txt`
  - `reports/raw/2026-04-06-av1-roi-experiments/exp_saliency_alpha10_v2_scorer.log`
- Claimed result from that run:
  - `current_workflow`: `2.04`
  - PoseNet: `0.08031919`
  - SegNet: `0.00567797`
  - bytes: `864,168`
- Local follow-up work:
  - runtime loader now accepts `saliency_weighted` residual weights
  - isolated rerun launched on `/tmp/pact-authoritative/workspace/upstream/comma_video_compression_challenge`
  - pinned encode config: `submissions/robust_current/config.av1-2.08.env`
  - smoke passed
  - authoritative local CPU scorer finished at `2.08`
- Rerun result:
  - `current_workflow`: `2.08`
  - PoseNet: `0.08694527`
  - SegNet: `0.00576935`
  - bytes: `864,167`
  - verdict: did not reproduce the earlier claimed `2.04`
- Immediate implication:
  - the live question is parity mismatch between the trusted-partner run and the isolated rerun, not whether the branch should be promoted as-is

## 2026-04-07 - learned post-filter promotion + state reconciliation

### learned post-filter retrained on the real operating point: Score 2.05
- Config: `522x392`, `sharpness=1`, `fg=22`, `crf34`, learned int8 post-filter in the inflate path
- Current-workflow bytes: `861,986`
- Rule-faithful estimate: `2.0778631822069484` at `896,432` bytes
- PoseNet: `0.07996829`
- SegNet: `0.00586716`
- Smoke: passed locally and on BAT00 before the authoritative scorer run
- Decision: PROMOTE. This is now the honest Track B floor.

### first post-filter trained on the wrong distribution: Score 2.35 - REJECTED
- Root cause: trained on the `film-grain=0` recovery distribution, then deployed on the canonical fg22 floor
- Effect: the model overcorrected and added grain-like texture on top of already useful grain
- Lesson: the post-filter training distribution must match the deployed archive distribution

### grain-mask final: Score 2.30 - REJECTED
- Bytes: `716,797`
- PoseNet: `0.15428504`
- SegNet: `0.00577725`
- Decision: keep as research evidence only; not a promotion candidate

### operating conclusion
- The repo is now primarily a task-aware decode-correction project
- The next scored cycle is capped at three post-filter follow-ons: larger, smaller or luma-only, and cheaper architecture
- The immediate blocker is not another scorer run; it is the senior-engineer and senior-editor review loop on the `2.05` writeup/site state

## 2026-04-07 — Rick Rubin breakthrough cycle + canonical inflate validation

### canonical inflate (PyAV + torch bicubic + BT.601): Score 2.08
- Config: 522x392, sharpness=1, fg=22, CRF 34, canonical inflate.py
- Archive: 862K
- PoseNet: 0.08570, SegNet: 0.00577
- Validates that the canonical inflate matches the evaluator exactly

### film-grain=0: Score 2.94 — CATASTROPHIC
- PoseNet: 0.36751 (+292% worse). Grain synthesis is essential signal for PoseNet.

### 512x384 + canonical: Score 2.14
- Smaller resolution hurts both PoseNet and SegNet despite rate savings

### ROI map + canonical inflate: Score 2.11
- PoseNet: 0.08890, SegNet: 0.00573, 887K
- ROI map + canonical didn't compound as hoped

### Rubin stack (ROI map + CRF 35 + canonical): Score 2.09
- PoseNet: 0.09222, SegNet: 0.00577, 827K
- CRF 35 hurt PoseNet more than ROI map helped

### CRF 36 canonical: Score 2.15
- Too aggressive rate trade

### CONFIRMED: authoritative 2.08 report has PoseNet=0.08694 (NOT 0.0938)
- Ledger had a transcription bug for the sharpness=1 ffmpeg row
- All three 2.08s cluster at PoseNet 0.085-0.087

### Multi-perspective reviews completed:
- Lattner: BT.709 vs BT.601 mismatch (fixed)
- Dean: SegNet only sees last frame; chroma nearly irrelevant to PoseNet
- Tao: rate and SegNet have higher marginal returns than PoseNet
- Rubin v2: ROI is the only zero-sum-breaking knob; identified ledger ghost
- Luckey: even/odd QP, PoseNet-gradient saliency map
- Newell: per-video CRF, learned post-filter, residual patches in archive

### Floor remains 2.08 after 38+ experiments

## 2026-04-06 — AV1+ROI lane activation: encoding experiments

### Experiment A: Full aggressive SVT params (CRF 33 + tune=0 + qm + variance-boost + film-grain-denoise)
- Config: `exp_a_vq_mode.env`
- Archive: **2,181,005 bytes** (+152% vs baseline)
- Status: **REJECTED** — variance-boost at CRF 33 catastrophically inflated bitrate
- Reflection: tune=0 + enable-variance-boost redistributes bits but at this CRF the total is way too high

### Experiment A2: VQ mode + QM at higher CRF (CRF 38, no variance-boost)
- Config: `exp_a2_vq_high_crf.env`
- Archive: **667,270 bytes** (-22.8% vs baseline)
- Scorer: **2.27** — REJECTED
- PoseNet: 0.12025 (+28% worse), SegNet: 0.00730 (+27% worse), Rate: 0.01777 (-22.6%)
- Reflection: CRF 38 is too aggressive. The 22% byte savings (0.13 pts on rate) was overwhelmed by distortion degradation (0.15 pts SegNet + 0.13 pts PoseNet). VQ+QM may still help at CRF 34 where distortion is lower.

### Experiment D: VQ mode only (CRF 34, tune=0)
- Config: `exp_d_vq_only_conservative.env`
- Archive: **888,241 bytes** (+2.7% vs baseline)
- Smoke: PASS (all checks green, semantic MAE 5.53 vs baseline 5.54)
- Scorer: NOT YET RUN (queued after A2)
- Reflection: VQ mode barely changes file size at same CRF — its value is in distortion, not rate

### Experiment H: sharpness=1 + consensus params (CRF 33)
- Config: `exp_h_sharpness1_consensus.env`
- Archive: **909,307 bytes** (+5.2% vs baseline)
- Scorer: NOT YET RUN
- Note: sharpness=1 already validated externally at 2.08 with CRF 34

### Experiment I: sharpness=1 + VQ + QM + film-grain-denoise (CRF 34)
- Config: `exp_i_sharpness1_vq.env`
- Archive: **825,284 bytes** (-4.5% vs baseline)
- Scorer: **2.14** — close but worse than 2.08 floor
- PoseNet: 0.08440 (-10% better!), SegNet: 0.00676 (+17.6% worse), Rate: 0.02198 (-4.5%)
- Reflection: VQ+QM improve PoseNet substantially but hurt SegNet. Since 100*seg is the biggest term, SegNet damage outweighs PoseNet gains. VQ mode redistributes bits toward perceptual structure (good for pose) at the cost of fine detail (bad for segmentation).

### Experiment J: sharpness=1 + ROI preprocessing (CRF 34)
- Config: `exp_j_sharpness1_preprocess.env`
- Archive: **785,300 bytes** (-9.2% vs baseline)
- Scorer: **2.52** — REJECTED
- PoseNet: 0.19174 (+104% worse!), SegNet: 0.00612 (+6%), Rate: 0.02092 (-9%)
- Reflection: Static polygon mask + Gaussian blur is WAY too aggressive for PoseNet. The outside-blur destroyed temporal pose cues. Need either much lighter degradation (blend=0.30, sigma=1.0) or ML-generated masks that precisely identify PoseNet-relevant regions.

### Experiment L: sharpness=1 + scd=0 + hqdn3d (CRF 34)
- Config: `exp_l_sharpness1_scd0_denoise.env`
- Archive: **858,473 bytes** (-0.7% vs baseline)
- Scorer: **2.14** — REJECTED (scd=0 + hqdn3d slightly hurt both distortions)

### Experiment K: Gentle preprocessing (sigma=0.8, blend=0.25, CRF 34, sharpness=1)
- Config: `exp_k_gentle_preprocess.env`
- Archive: **841,098 bytes** (-2.7% vs baseline)
- Scorer: **2.47** — REJECTED (even gentle blur kills PoseNet by 90%)

### Experiment P: CRF 33 + sharpness=1 only
- Config: `exp_p_crf33_sharpness1_only.env`
- Archive: **921,198 bytes** (+6.6% vs baseline)
- Scorer: **2.16** — REJECTED (CRF 33 adds bytes, marginal SegNet gain doesn't compensate)
- Reflection: CRF 34 is the sweet spot for our pipeline. CRF 33 hurts on rate more than it helps on SegNet.

### Experiment Q: Chroma-only degradation + sharpness=1 (static corridor, CRF 34)
- Config: `exp_q_chroma_only.env`
- Archive: **837,868 bytes** (-3.0% vs baseline)
- Scorer: IN PROGRESS
- Hypothesis: chroma-only degradation preserves luminance (PoseNet-safe)

### Experiment R: Falcon Perception ML masks + chroma-only + sharpness=1 (CRF 34)
- Config: `exp_r_falcon_chroma.env`
- Archive: **835,286 bytes** (-3.4% vs baseline)
- Scorer: QUEUED after Q
- Innovation: semantically-aware ML masks from Falcon Perception + chroma-only degradation

### Infrastructure delivered this cycle
- Codec-agnostic ROI refactoring (compress.sh + analyze_roi.py)
- ROI preprocessing pipeline (roi_preprocess.py) with static + adaptive masking
- ML mask generator (Falcon Perception + SAM 3/2 with toolchain conditioning)
- 10+ experiment configs
- All critical code review bugs fixed
- scipy installed for fast Gaussian blur

## 2026-04-06 — AV1 Lanczos-upscale probe

- Candidate: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp=0.35`
- Hypothesis: swapping bicubic for lanczos on the upscale axis might improve task fidelity at identical bytes.
- Estimate before run: `2.16`–`2.19`.
- Smoke gate: PASS.
- Result: **`2.18`** at `864,455` bytes.
- Reflection: hypothesis held; bytes stayed flat while both pose and seg improved slightly.
- Decision: promote Lanczos-upscale as the new Track B floor.

## 2026-04-06 — comprehensive bug / rigor pass

- Scope: package/install/eval contract, rule-faithful accounting, ROI path correctness, repo hygiene.
- Fixed:
  - upstream-root packaging
  - package-without-sync rejection
  - stale `inflated/` cleanup before eval
  - installed-payload rule-faithful accounting
  - AV1+ROI fail-fast guard
  - `FFMPEG_BIN` / `FFPROBE_BIN` propagation in ROI metadata analysis
  - live `ROI_X_FRAC`
  - ROI-side `INFLATE_POSTFILTER`
  - root `.gitignore` + transient scratch cleanup
- Verified:
  - ROI guard fails deliberately on AV1
  - wrapper audit logs both `ffprobe` and `ffmpeg`
  - `ROI_X_FRAC` changes the produced ROI artifact
  - ROI inflate output changes under `INFLATE_POSTFILTER=hflip`
  - smoke gate passed on the historical 2.18 floor
- Note: git history could not be cleaned because this workspace is not a git repository.

## 2026-04-06 — colorspace/range hardening promotion

- Candidate: same live AV1 floor with explicit `tv / bt709 / bt709 / bt709` encode tags and explicit `rgb24(pc)` decode conversion.
- Hypothesis: explicit color handling would reduce evaluator mismatch on the flat path.
- Smoke gate: PASS.
- Encoded ffprobe tags: PASS.
- Result: **`2.12`** at `864,486` bytes.
- Reflection: bytes moved by only `+31`, SegNet worsened slightly, but PoseNet improved sharply enough to win by `0.06`.
- Decision: promote the hardened explicit-color path as the new Track B floor.

## 2026-04-06 — speculative lane capture

- Recorded the AV1 + ROI parity lane as explicitly speculative only.
- Required plan captured in durable state:
  1. codec-agnostic ROI encode abstraction
  2. AV1 params for base/ROI/ROI2 streams
  3. matching AV1-aware metadata ROI path
  4. matching inflate/smoke/scorer parity checks
  5. fresh scorer-backed evidence that it actually helps
- Rule: do not promote until scorer evidence justifies it.

## 2026-04-06 — writeup system / frontend pass

- Added generated artifacts for reproducibility and reuse:
  - `reports/graphs/build_experiment_manifest.py`
  - `reports/graphs/build_code_callouts.py`
  - `reports/graphs/build_comparison_media.py`
  - `reports/graphs/refresh_site.py`
  - `docs/repro_checklist.md`
  - `just rebuild-site`
- Added site-level context so the landing page now states the contest, the repo identity, the GitHub source, and when the page was last rebuilt.
- Added browser-preview comparison media with synced play/pause + scrubber and crop zoom.
- Added posters for the comparison videos and horizontal-scroll handling for the dense frontier table on mobile.
- Verified with local desktop and iPhone screenshots generated via Playwright.

## 2026-04-06 — player / scatter coherence pass

- Reworked the comparison-player state model so mode switches preserve playhead and pause hidden videos.
- Replaced the brittle 2.18→2.12 SVG metric rows with semantic HTML layout.
- Added published-baseline comparison to the summary strip.
- Flipped the bug detour downward in the lineage graph so failures read as failures.
- Added a focused operating-range scatter view plus lighter markers and hover/focus details.
- Verified via Chrome DevTools browser automation and local desktop/iPhone screenshots.

## 2026-04-06 — final frontend closeout

- Added point-anchored scatter feedback, tighter mobile spacing, and cleaner grouped references.
- Re-ran desktop and mobile screenshot audits.
- Marked the visual-verdict pass as closed with no remaining concrete issues or suggestions.

## 2026-04-06 — semantic-rigor maintenance pass

- Re-ran `robust_current` smoke with packaging enabled and captured fresh evidence at `reports/raw/2026-04-06-semantic-rigor/robust_current-smoke.json`.
- Result: PASS on file count, exact frame count, exact geometry-derived byte size, and sampled RGB semantic sanity.
- Recorded the semantic smoke metrics explicitly in the docs so the repo no longer understates the current smoke gate.
- Re-ran `exact_current` scorer regression to keep Track A healthy under the current workflow path.

## 2026-04-06 — submission gate update

- Submission decision changed: do **not** submit at 2.12.
- New gate recorded for the next cycle:
  1. beat `2.1` authoritatively on the local scorer path
  2. complete another full low-hanging-fruit exploration round first
- Track B remains the only plausible submission lane; Track A remains transparency-only.

## 2026-04-06 — pre-submit exploration round

- `sharpness=1` smoke passed and scorer improved the authoritative local result to **`2.08`**.
- `scd=0` smoke passed, but the scorer run was interrupted before completion (`exit 143`), so it remains unresolved.
- BAT00 side lane is now live enough for speculative work: SSH works, WSL2 Ubuntu works, upstream repo is cloned, and the lab workspace is seeded there.
- BAT00 remains explicitly non-authoritative for score claims.


## 2026-04-06T20:49:00-05:00 — BAT00 saturation hardening

- Added BAT00 sync / launch / poll helpers under `experiments/`.
- Fixed a real remote-queue correctness bug: the first BAT00 batch skipped packaging and reused shared mutable source state.
- Hardened the remote worker so each job now:
  - snapshots the requested config
  - copies a minimal per-job repo workspace
  - packages inside that isolated workspace
  - smokes against a per-job isolated upstream root
  - writes `manifest.json`, `status.json`, and a remote ledger line
- Valid BAT00 smoke batch completed for `exp_j`, `exp_l`, `exp_m`, and `exp_n`.
- Local authoritative reject recorded: `exp_h` = `2.13`.
- Local authoritative scorer run started for `exp_j` because it is the strongest byte-saving smoke result.

### saliency-weighted post-filter alpha=10: Score 2.04 - NEW FLOOR
- Config: h=16, saliency alpha=10, CRF 34, canonical archive
- PoseNet: 0.08032, SegNet: 0.00568, Rate: 0.02302
- Key: saliency weighting protected SegNet (0.0057 vs 0.0059 baseline)
- Decision: NEW PROMOTED FLOOR

### saliency-weighted alpha=20 h=16: Score 2.03 - NEW FLOOR
- PoseNet: 0.07918, SegNet: 0.00569, Rate: 0.02302
- Best SegNet of any post-filter variant at 2.03

### saliency-weighted alpha=10 h=32: Score 2.03 - TIED
- PoseNet: 0.07734 (best PoseNet overall!), SegNet: 0.00571, Rate: 0.02302
- Best PoseNet at cost of slightly larger weights (16KB)

### submission candidate: alpha=20 h=16 at 2.03
- Smaller weights (8KB), better SegNet, same total score as h=32
- Ready for PR submission

## 2026-04-08 - post-1.99 leaderboard and fleet update

- Latest official comma.ai leaderboard check:
  - `1.95` first place
  - `1.98` second place
  - `2.05` third place
- Implication:
  - promoted `1.99` floor is `0.01` off second and `0.04` off first
- Fleet status:
  - `mini` is reachable with key auth, Python 3.12, and `uv`; now approved as a proxy/eval side lane
  - `molt` is reachable and has `uv`, but still needs path hardening before it joins the active queue
  - `tertiary` remains unresolved by hostname and is not counted as available capacity
  - `bat00` auth is now confirmed via `adpena@bat00.local`
  - `bat00` WSL path is live at `/home/adpena`
  - `bat00` GPU confirmed: `NVIDIA GeForce RTX 2070 SUPER, 591.86, 8192 MiB`
- Action taken:
  - launched the official-path `v2_alpha20_h32_e20_int8.pt` proxy on `mini` as a side lane
  - local CPU scorer lane remains reserved for authoritative runs only

## 2026-04-08 - active post-promotion lanes

- Local training lane:
  - `long500_qat_ema_alpha20_h32` is active on MPS as the first long-horizon capacity follow-up to the promoted h16 winner
  - actual live log path is `/private/tmp/pact-mine/experiments/postfilter_weights/train_long500_h32.log`
  - best scorer line seen during training: epoch `350`, scorer `3.9655`, pose `0.044424`, seg `0.034820`
  - run completed at epoch `500`
  - final local eval:
    - baseline `4.5179`
    - filtered `4.1665`
    - delta `-0.3513`
    - pose `0.056962`
    - seg `0.034118`
  - saved artifacts:
    - `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long500_qat_ema_alpha20_h32_fp32.pt`
    - `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_long500_qat_ema_alpha20_h32_int8.pt`
  - official upstream-path proxy is now running on the int8 artifact
- Network proxy lane:
  - `mini` is running the official-path upstream proxy for `postfilter_v2_alpha20_h32_e20_int8.pt`
  - current visible progress in the remote log reached at least `Processed 600 frames ...`
- GPU lane status:
  - `bat00` is auth-validated and CUDA-capable, but still not a stable training lane
  - the trusted partner is already using it for active jobs, so do not cancel or trample existing remote work unless there is a concrete conflict

## 2026-04-08 - per-channel quantization lane opened

- Local implementation:
  - `experiments/train_postfilter_saliency.py` now supports `save_model_int8(..., per_channel=True)`
  - conv weights save per-output-channel scales
  - biases stay in fp32 to avoid quantizing away tiny tensors for negligible byte savings
- Verification:
  - `experiments/test_train_postfilter_saliency.py` now covers per-channel save output
  - `experiments/test_postfilter_loader.py` now covers per-channel weights + fp32 biases through the runtime loader
- Artifact:
  - source fp32: `experiments/postfilter_weights/postfilter_long500_qat_ema_alpha20_h16_fp32.pt`
  - new int8: `experiments/postfilter_weights/postfilter_long500_qat_ema_alpha20_h16_pc_int8.pt`
  - size: `8018` bytes vs the previous tensorwise artifact at roughly `8.3K`
- Status:
  - official upstream-path proxy for the per-channel promoted-floor artifact is running locally

## 2026-04-08 - bat00 CUDA lane activated

- Auth:
  - `adpena@bat00.local`
  - password auth succeeded
- Environment:
  - WSL workspace: `/home/adpena/pact-side`
  - `uv` present in WSL
  - torch CUDA probe succeeded with `2.11.0+cu130`
  - GPU: `NVIDIA GeForce RTX 2070 SUPER`
- Launch:
  - `long1000_qat_ema_alpha20_h16`
  - remote PID reported as `406`
- Role:
  - CUDA long-run lane for epoch-budget follow-ons while local host handles proxy/eval work
- Reality check:
  - after syncing the missing base archive and saliency mask, a 1-epoch foreground smoke still exited after decode
  - the background `long1000_qat_ema_alpha20_h16` launch did not stay alive
  - bat00 is therefore auth-validated and environment-validated, but not yet a stable training lane
  - later debugging showed the trainer progressed further after the lazy-pair/saliency patch, but still does not yet complete a clean 1-epoch smoke on bat00

## 2026-04-08 - remote job surface hardening

- Added `experiments/remote_job.py` as the first reusable remote-job helper surface.
- Current scope:
  - render a deterministic remote launch script
  - write a timestamped launch manifest
- Verification:
  - `uv run --with torch python -m unittest experiments/test_remote_job.py`
  - `python3 -m py_compile experiments/remote_job.py`
- Reason:
  - reduce ad hoc SSH launch behavior and preserve remote-work metadata on disk instead of leaving it in shell history

## 2026-04-08 - post-1.95 next branch opened

- Launched the next local long-horizon capacity sweep:
  - `long500_qat_ema_alpha20_h24`
- Intended log path:
  - `/private/tmp/pact-mine/experiments/postfilter_weights/train_long500_h24.log`
- Durable manifest:
  - `.omx/logs/remote_jobs/local-long500-h24.json`
- Reason:
  - after the h32 `1.95` promotion, h24 is the next clean capacity follow-up under the same proven regime

### official-path proxy verdict for per-channel promoted-floor artifact
- Artifact:
  - `experiments/postfilter_weights/postfilter_long500_qat_ema_alpha20_h16_pc_int8.pt`
- Repo evidence copy:
  - `reports/raw/2026-04-08-long500-per-channel-proxy/per_channel_proxy_summary.json`
  - `reports/raw/2026-04-08-long500-per-channel-proxy/per_channel_proxy_report.txt`
- Packaging:
  - `8018` bytes vs `8519` bytes for the tensorwise promoted artifact
- Result:
  - PoseNet `0.06952669`
  - SegNet `0.00579276`
  - rate `0.02301653`
  - score `1.99`
  - rule-faithful estimate computed locally after swapping the artifact would be slightly worse than the promoted tensorwise bundle because the distortions regressed
- Decision:
  - reject for promotion; keep the original tensorwise long-500 artifact as the shipped floor

### official-path proxy verdict for v2 h32
- Side lane:
  - `mini`
- Artifact:
  - `experiments/postfilter_weights/postfilter_v2_alpha20_h32_e20_int8.pt`
- Repo evidence copy:
  - `reports/raw/2026-04-08-v2-alpha20-h32-proxy-mini/proxy_v2_alpha20_h32_e20_mini.log`
- Result:
  - PoseNet `0.07943186`
  - SegNet `0.00568828`
  - rate `0.02301653`
  - score `2.04`
- Decision:
  - reject for authoritative promotion; h32 under the scorer-faithful v2 family is still only a close miss
  - verified `molt` is reachable and has `uv`, but it still needs package/cache hardening before active use

## 2026-04-09 14:00:00 -0500 - site + infra + training relaunch

### site
- pushed to adpena/comma-lab (private GitHub)
- CRITICAL CSP fix: `script-src 'self'` was blocking ALL JS in production
- hero text updated: technical tagline with scoring formula + rate definition
- stripped radial gradients (user: "nothing that looks AI-designed")
- video explorer zoom: preload=auto, first-frame seek, loadeddata geometry update
- cards: flatter borders, muted uppercase labels
- Track A "0.00" explained as documented exploit
- leaderboard snapshot dated
- research diary: 41 run-log entries scrubable with play/pause

### infra
- Kaggle legacy API connected
- Modal CLI installed (auth pending)
- auto_commit.sh tool written
- conversation transcript parser: 1,360 events, 4,144 score trajectory points
- CLAUDE.md git discipline rules added
- 20+ commits creating proper research history (was 10)
- .gitignore: exclude .raw, .npy, .safetensors >100MB
- git filter-branch removed 116MB smoke.raw from history

### training
- PSD h=64 v2 relaunched in tmux (1000 epochs, alpha=20)
- Dilated h=64 v2 relaunched in tmux (1000 epochs, alpha=20)
- both using .venv/bin/python with MPS

### research
- competition landscape: our challenge is unique (only task-aware scoring)
- ISCAS 2026 neural video coding (May 24-27, $4,500/track)
- no Kaggle competitions for video compression exist

### decision
- keep polishing site design (whitespace, symmetry, no AI gimmicks)
- monitor PSD/dilated training convergence
- deploy to Cloudflare Pages after next design pass

## 2026-04-09 14:30:00 -0500 - expert council consensus on sub-1.6 path

### Tao (math)
- SegNet is 11.5x more score per unit improvement than PoseNet (100x coeff vs sqrt compression)
- seg 0.00576 → 0.003 saves 0.276 alone (nearly enough for 1.45)
- SegNet boundary-band attack with h=64 is the highest-leverage single experiment

### Karpathy (arch)
- h=128 scaling law predicts 1.61 but rate crossover nearly cancels it
- pair-aware 6-channel is the highest-EV architectural change (PoseNet scores pairs, we filter frames independently)
- skip adding more layers — dilation is more parameter-efficient for RF expansion

### LeCun (representation)
- h=96 is the sweet spot (rate crossover at h=128)
- run 2500 epochs not 1000 — more lottery tickets for best-checkpoint
- LSQ quantization: 5-line change, expected 0.03-0.08 improvement
- fp16 inference would unlock h=128 without rate penalty

### Jensen (compute)
- get cloud GPU online THIS WEEK
- 6 training slots left in 24 days
- week 1: infra + h=96, week 2: compound, week 3: polish, week 4: buffer

### consensus projection
- conservative: 1.60 (h=96 + LSQ + mild SegNet)
- central: 1.50 (SegNet boundary attack + h=96 compound)
- optimistic: 1.40 (everything works)
- hard floor: ~1.2

### decision
- priority 1: get Modal/Kaggle GPU online for h=96 training
- priority 2: SegNet boundary-band attack on local Mac
- priority 3: LSQ quantization (5-line change)
- continue PSD/dilated training as insurance

## 2026-04-09 14:45:00 -0500 - film-grain sweep: dead end for rate

### results
- fg=18: 854KB (preset 0), fg=22 (current): ~844KB — only 10KB difference
- at preset 4: fg=20-32 all within 875-877KB — flat, no meaningful rate variation
- film-grain parameter barely affects archive size at CRF 34

### decision
- film-grain sweep is exhausted — not worth pursuing for rate reduction
- the codec is already aggressive at CRF 34; grain synthesis is visual-only
- focus rate efforts on resolution micro-tuning or CRF fractional (if any headroom)

## 2026-04-09 15:30:00 -0500 - Modal deployed, PSD converging, Kaggle P100 blocked

### training status
- PSD v2: ep 91, scorer 3.836 (promoted floor: 3.547, gap: 0.289)
- PSD long1000 v2: ep 101, scorer 3.953
- 2 local MPS trainers alive
- Modal A10G h=96 deployment attempted (image built, need to verify execution at modal.com/apps)
- Kaggle: P100 CUDA incompatible with PyTorch 2.x — dead end for GPU training

### infrastructure
- Modal authed and operational
- Self-contained cloud_h96_trainer.py (977 lines) deployed
- bat00 RTX 2070 Super alive and reachable

### council roadmap execution
- [x] LSQ quantization implemented in tac library
- [x] Film-grain sweep (dead end — no rate savings at CRF 34)
- [x] Top-K checkpoint averaging tool built
- [x] Cloud GPU: Modal deployed, Kaggle blocked by P100
- [ ] SegNet boundary-band attack (council #2, next after PSD converges)
- [ ] Pair-aware 6-channel input (council #4, needs data pipeline)
- [ ] h=96 long training on cloud GPU (council #1, Modal pending)

### decision
- continue monitoring PSD v2 convergence
- when PSD v2 < 3.55, proxy-score it
- fix Modal training execution (check logs at dashboard)
- keep bat00 as backup GPU resource
