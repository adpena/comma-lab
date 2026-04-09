# next experiments

## 2026-04-09 active queue after the h64 `1.73` promotion

The promoted honest floor is now `long1000_h64` at **`1.73`** on the current published scorer path. The next cycle should not reopen closed ensemble tuning or stale proxy work. The completed bat00 parity lane is now a reference artifact, and the active queue is the still-running local MPS fleet.

## cycle budget

1. Prefer at most **3** serious lanes in flight.
2. Do not spend the authoritative scorer on anything that has not already packaged, smoked, and looked promising on an official-path proxy.
3. Do not reopen closed lanes just because compute is free.

## current priority order

1. **PF-DILATED-H64 / PF-ALPHA30-H32 / PF-H96**
   - Why first: these are now the strongest still-live packaged side lanes.
   - Status: **ACTIVE** on local MPS
   - Current packaged standings:
     - `dilated h64`: best epoch `253`, scorer `3.7600972843170166`, int8 `45,731` bytes
     - `alpha30 h32`: best epoch `423`, scorer `3.8314755090077717`, int8 `16,091` bytes
     - `h96`: best epoch `192`, scorer `3.8979716682434082`, int8 `93,331` bytes
   - Latest printed checkpoints:
     - `dilated h64`: epoch `250`
     - `alpha30 h32`: epoch `440`
     - `h96`: epoch `200`
   - Decision:
      - keep the lanes alive
      - do not proxy yet; all three are still materially weaker locally than the promoted h64 best (`3.5473`)
      - sidecar triage now confirms `h48 best` was already proxied, so `dilated h64` is the strongest unproxied packaged lane
      - treat the live `dilated h64` artifact as observation-only until it is relaunched with the repo-side deploy-correct wrapper, because the current `/private/tmp` saved meta still says `variant: "saliency_weighted"`
      - even after the new best at `3.7601`, the refreshed proxy gate still keeps it out: it is deploy-blocked and still `0.2128` above the promoted h64 local reference

2. **PF-SEGNET-FIXED-V2 / PF-SEGNET-H64**
   - Why second: the SegNet lane still has the best remaining theoretical headroom, but its local numbers are on a different training metric and the runs still are not packaging cleanly.
   - Status: **ACTIVE** on local MPS
   - Current observations:
     - `pixelshuffle_h64`: latest printed epoch `230`; saved best epoch `229`, scorer `3.797689816157023`, int8 `94,285` bytes; improved materially and is now the strongest deploy-ready packaged side lane, but still too weak
     - `psd_h64`: latest printed epoch `140`; saved best epoch `115`, scorer `3.8837292766571045`, int8 `94,087` bytes; improved, but still weaker than pixelshuffle
     - `segnet_attack_fixed_v2`: latest printed epoch `900`; best observed local scorer `0.9819` at epoch `590`; no saved best artifact visible
     - `segnet_attack_h64`: latest printed epoch `350`, scorer `1.0854`, PoseNet `0.047744`, SegNet `0.005653`; no saved best artifact visible
   - Decision:
      - keep all three lanes alive as research-only lanes
      - do not spend proxy time until one of them writes a real artifact
      - if a slot is deliberately reassigned, relaunch the SegNet family from the synced repo-side trainer instead of trusting the already-running stale local processes

3. **PF-QAT-EMA-INT8-SELECTION**
   - Why third: it directly tested the verified train-to-deploy gap with post-quant checkpoint selection and per-channel int8 enabled.
   - Status: **COMPLETED_NO_PROMOTION** on `bat00` WSL
   - Remote manifest: `.omx/logs/remote_jobs/bat00-long1000-h32-qint8sel-pc.json`
   - Remote pid at launch: `9852`
   - Remote log: `/home/adpena/pact-side/experiments/postfilter_weights/train_long1000_h32_qint8sel_pc_bat00.log`
   - Latest printed checkpoint: epoch `200`, scorer `4.1889`, PoseNet `0.095425`, SegNet `0.034340`
   - Saved best checkpoint: epoch `199`, scorer `3.9258260917663574`, int8 size `16,781`
   - Decision:
      - keep the saved artifact as a resolved reference point
      - do not proxy; it never closed enough toward the promoted h64 local regime

4. **PF-PIXELSHUFFLE-H64**
   - Why fourth:
      - this is now the strongest packaged lane that is both deploy-ready and inside the local proxy gate
   - Status: **PROXY_RUNNING**
   - Current local best:
      - epoch `229`
      - scorer `3.797689816157023`
      - int8 `94,285` bytes
   - Proxy lane:
      - `experiments/proxy_score_faithful.py`
      - weights: `/private/tmp/pact-mine/experiments/postfilter_weights/postfilter_pixelshuffle_h64_long1000_best_int8.pt`
   - Important blocker resolved:
      - `submissions/robust_current/inflate_postfilter.py` now supports the pixelshuffle-dilated runtime path and can infer it from the artifact state layout
   - Decision:
      - wait for the faithful proxy result
      - if it misses, keep the lane as a non-promoted alternate; if it hits, it earns scorer consideration

5. **READY-TO-RELAUNCH SCAFFOLDS**
   - `experiments/train_postfilter_dilated_h64.py`
     - repo-side deploy-correct wrapper for the dilated h64 lane with `variant: "dilated"` metadata
   - `experiments/train_postfilter_pixelshuffle_dilated.py`
     - hybrid PixelShuffle + dilated h64 scaffold, tested locally
   - `experiments/train_postfilter_pairaware.py`
     - pair-aware 6-channel scaffold with dry-run CLI, tested locally
   - Decision:
     - these are code-complete sidecars, not scored results
     - use them for the next clean relaunch once a local slot is deliberately reassigned

6. **FOUNDATION-BUILD FOLLOW-THROUGHS**
   - `src/comma_lab/task_codec/`
     - landed scorer, architecture, quantization, and evaluation/proxy record abstractions
     - next step: wire these into higher-level tooling instead of duplicating artifact parsing ad hoc
   - `comma-lab sched`
     - `sched status`, `sched results`, and `sched budget` now exist as read-only/reporting surfaces
     - next step: add a real platform registry under `configs/` before any launch semantics are attempted
   - `reports/graphs/report_history.html`
     - git-backed history/time-machine viewer now exports through the static site pipeline
     - next step: decide whether to link it from the main dashboard/navigation once the history scope is approved

## sidecar analysis outputs

- `reports/raw/2026-04-09-sidecar-analysis/proxy_gate_triage.json`
  - marks `postfilter_long1000_h48_best` as `proxy_already_run`
  - refreshed from disk at `2026-04-09 12:05:14 -0500`
  - strongest unproxied packaged lane is currently `postfilter_dilated_h64_long1000`
  - `postfilter_long1000_h32_a30` has now moved ahead of the old h32 rerun in the saved-artifact ranking
  - deployability gate now marks `postfilter_dilated_h64_long1000` as observation-only until its variant metadata is fixed
- `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json`
  - machine-readable lane snapshot generated directly from best-meta files and training logs
  - use this instead of hand-tail polling when checking whether a lane materially changed
  - latest snapshot now merges special log slugs back into canonical best-artifact lanes, so `dilated_h64_long1000`, `pixelshuffle_h64_long1000`, and `psd_h64_long1000` each carry both best and latest fields in one record
  - latest snapshot at `2026-04-09 13:13:08 -0500` shows `dilated_h64_long1000` improved to `3.7600972843170166` and keeps `pixelshuffle_h64_long1000` as the strongest deploy-ready packaged side lane
- `reports/raw/2026-04-09-sidecar-analysis/quantization_drift_audit.json`
  - audited `h64`, `dilated_h64`, `alpha30 h32`, and `h96`
  - drift is not lowest on the promoted `h64` line, so the current edge does not look like “quantization cleanliness” alone

## resolved reference points

1. **PF-H64-LONG1000**
   - Status: **PROMOTED**
   - Authoritative result: `1.73` at `864,167` bytes
   - Distortions: PoseNet `0.03317023`, SegNet `0.00575544`
   - Evidence root: `reports/raw/2026-04-09-long1000-h64-authoritative/`
   - Repo-side official-path proxy on the saved-best artifact is also resolved and matched the authoritative scorer path

2. **PF-ENSEMBLE-WEIGHT-SWEEP**
   - Status: **CLOSED**
   - Best exact two-model point: `70/30` proxy-implied `1.840545292085`
   - Promoted scorer-backed point: `75/25` at `1.84`
   - Three-way follow-on (`h32 + MC + Kalman`) regressed to `1.89`

3. **PF-MONTE-CARLO-LAYER-SCALE**
   - Status: **NON-PROMOTED REFERENCE**
   - Best transferred family score: `1.86`

4. **PF-SEGNET-NATIVE-H32**
   - Status: **NON-PROMOTED REFERENCE**
   - Best scorer-backed result: `1.90`

## queue hygiene

1. Promote weights, not narratives.
2. Keep `current_workflow` and `rule_faithful` explicit on every promotion surface.
3. Preserve exact timestamps in state, reports, and evidence roots.
4. Any public leaderboard comparison must include the exact check time and source URL.
5. A saved local artifact that is still far from the promoted h64 local regime is not a proxy candidate yet.
6. Leave the next agent a truthful queue, not a hope queue.
