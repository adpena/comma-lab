# current focus

## authoritative state after the 2026-04-09 h64 promotion

- Track B authoritative floor: **`1.73`** on the local CPU scorer path
- Current-workflow bytes: `864,167`
- Rule-faithful estimate: `1.7947470454539947` at `966,071` bytes
- Distortions: PoseNet `0.03317023`, SegNet `0.00575544`
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / long1000 QAT+EMA learned int8 post-filter (alpha=20, h=64)`
- Evidence root: `reports/raw/2026-04-09-long1000-h64-authoritative/`
- Track A / `exact_current` remains transparency-only. Keep it runnable, but do not treat it as the honest frontier.

## verified milestone timestamps

- `2026-04-08 09:47:50 -0500` — saliency `alpha=20` scorer-backed `2.01` floor landed
- `2026-04-08 14:57:15 -0500` — long-500 QAT+EMA scorer-backed `1.99` floor landed
- `2026-04-08 18:08:27 -0500` — long-500 h32 scorer-backed `1.95` floor landed
- `2026-04-08 19:12:26 -0500` — long1000 h16 scorer-backed `1.92` floor landed
- `2026-04-08 21:36:54 -0500` — long1000 h32 scorer-backed `1.85` floor landed
- `2026-04-09 06:37:51 -0500` — ensemble `75/25` scorer-backed `1.84` floor landed
- `2026-04-09 09:43:19 -0500` — h64 smoke gate passed
- `2026-04-09 10:07:00 -0500` — h64 authoritative local CPU scorer landed at `1.73`

## what is now known

1. The strongest path in the lab is still the tiny shipped learned post-filter family.
2. Width scaling kept paying after h32 and after the ensemble dead-ended. The h64 branch broke the `1.84` floor decisively.
3. SegNet leverage is still the main mathematical lesson, but at the `1.73` operating point it is now about `11.5x` stronger than PoseNet locally, not `13.7x`.
4. The local ensemble micro-tuning lane is closed. The best two-model point stayed near `70/30`, and the three-way blend regressed to `1.89`.
5. The quantization-parity hypothesis produced a clean saved artifact but did not yet produce a promotion candidate. The completed bat00 rerun topped out at local scorer `3.9258`, still far from the promoted h64 regime.
6. The local MPS host now has a real post-promotion fleet, and several packaged side lanes improved again under the new machine-readable snapshot. The strongest new saved non-promoted artifacts are now `dilated h64` at `3.7601`, `pixelshuffle_h64` at `3.8106`, `alpha30 h32` at `3.8315`, and `h96` at `3.8945`.
7. The sidecar analysis tooling is now materially better than the earlier hand-poll loop. `proxy_gate_triage` now carries deployability checks and marks the live `dilated h64` artifact as not deploy-ready because its saved meta still says `variant: "saliency_weighted"`. The `live_fleet_snapshot` tool now reads best-meta files and log rows directly from disk and merges special-case log slugs back into their canonical best-artifact lanes, which caught that the earlier manual tail polling was understating several active lanes.
8. Three future-facing experiment scaffolds are now in-repo and locally verified: a deploy-correct `dilated_h64` trainer wrapper, a `pixelshuffle_dilated` hybrid trainer scaffold, and a `pairaware` 6-channel scaffold. That deploy-correct wrapper now matters operationally, because the live `/private/tmp` `dilated h64` artifact still writes `meta.variant: "saliency_weighted"` and should be treated as observation-only until it is relaunched honestly.
9. The live SegNet research lanes are most likely stale processes, not current-code checkpoint-save failures. The active PIDs started at `2026-04-09 07:49:46 -0500` and `2026-04-09 09:29:13 -0500`, while `/private/tmp/pact-mine/experiments/train_postfilter_segnet_attack.py` was updated later at `2026-04-09 11:50:48 -0500`. That matches the logs: they print epoch rows only, without the current trainer's `eval score=` and `best checkpoint -> ...` lines.
10. The waiting-time build swarm has now landed three foundation surfaces cleanly in-tree:
   - `src/comma_lab/task_codec/` for scorer, architecture, quantization, and evaluation record abstractions
   - `src/comma_lab/scheduler/` plus `comma-lab sched ...` read-only reporting commands
   - `reports/graphs/report_history.html` + `report_history.json` for git-backed markdown/history exploration

## current honest frontier

| Score | Technique | Note |
|-------|-----------|------|
| **1.73** | long1000 QAT+EMA learned int8 post-filter (`alpha=20 h64`) | promoted floor |
| 1.84 | weighted ensemble learned int8 post-filter (`long1000 h32 + MC refine1`, `75/25`) | prior promoted floor |
| 1.85 | long1000 QAT+EMA learned int8 post-filter (`alpha=20 h32`) | older promoted floor |
| 1.86 | Monte Carlo / layer-scale refinement (`refine1`, `refine2`) | strongest non-promoted alternate family |
| 1.89 | public leaderboard leader `neural_inflate` | external target |
| 1.90 | SegNet-native learned int8 post-filter (`long1000 h32`) | scorer-backed close miss |
| 1.94 | public leaderboard second place `roi_v2` | external target |
| 1.95 | public leaderboard third place `av1_roi_lanczos_unsharp` | external target |

## immediate owner priorities

1. Keep the new `1.73` artifact and canonical report pair stable while follow-on branches continue in isolated workspaces.
2. Keep the writeup and generated site in sync with the new `1.73` floor.
3. Keep the local MPS host in observation mode: `alpha30 h32`, `h96`, `dilated h64`, `pixelshuffle_h64`, `segnet_attack_h64`, and `segnet_attack_fixed_v2` are all live, but none has earned proxy time yet.
4. Treat `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json` as the current machine-readable source of truth for local lane status instead of relying on manual tail polls.
5. Treat the completed bat00 parity artifact as a resolved reference point, not an active lane.
6. Do not spend fresh proxy/scorer time on anything that has not already written a real artifact to disk and closed materially toward the promoted h64 local regime.
7. Keep the newly landed foundation surfaces healthy:
   - `task_codec` should stay metadata-first and scorer-agnostic
   - `comma-lab sched` should stay read-only/reporting until launch semantics are trustworthy
   - `report_history.html` should stay git-backed and static-site-exportable

## next measured bets

1. **Completed quantization-parity reference**
   - Status: RESOLVED on `bat00` WSL
   - Remote manifest: `.omx/logs/remote_jobs/bat00-long1000-h32-qint8sel-pc.json`
   - Trainer process was gone when checked at `2026-04-09 10:59:38 -0500`
   - Latest printed checkpoint: epoch `200`, scorer `4.1889`, PoseNet `0.095425`, SegNet `0.034340`
   - Saved best checkpoint: epoch `199`, scorer `3.9258260917663574`, int8 size `16,781`
   - Decision: keep as a resolved reference artifact, not as an active promotion lane

2. **Local packaged side-lane fleet**
   - Status: ACTIVE on local MPS
   - `dilated h64`
      - log: `/private/tmp/pact-mine/experiments/postfilter_weights/train_dilated_h64.log`
      - latest printed checkpoint: epoch `250`
      - saved best: epoch `253`, scorer `3.7600972843170166`, int8 size `45,731`
      - caveat: live saved meta still reports `variant: "saliency_weighted"`, so this artifact is not yet an honest deploy-correct proxy candidate; refreshed proxy gate still keeps it out because it is deploy-blocked and still `0.2128` above the promoted h64 local reference
   - `alpha30 h32`
      - log: `/private/tmp/pact-mine/experiments/postfilter_weights/train_long1000_h32_a30.log`
      - latest printed checkpoint: epoch `500`
      - saved best: epoch `423`, scorer `3.8314755090077717`, int8 size `16,091`
   - `h96`
      - log: `/private/tmp/pact-mine/experiments/postfilter_weights/train_long1500_h96.log`
      - latest printed checkpoint: epoch `220`
      - saved best: epoch `217`, scorer `3.8944560209910075`, int8 size `93,331`
   - Decision: keep training, but none is close enough to h64 to justify proxy time yet

3. **Research-only side-lane watch**
   - Status: ACTIVE on local MPS
   - `pixelshuffle_h64`
      - log: `/private/tmp/pact-mine/experiments/postfilter_weights/train_pixelshuffle_h64.log`
      - latest printed checkpoint: epoch `230`, scorer `3.8357`, PoseNet `0.061960`, SegNet `0.032494`
      - saved best: epoch `229`, scorer `3.797689816157023`, int8 size `94,285`
      - status: still weak, but now clearly the strongest deploy-ready packaged side lane
   - `psd_h64`
      - log: `/private/tmp/pact-mine/experiments/postfilter_weights/train_psd_h64.log`
      - latest printed checkpoint: epoch `140`, scorer `3.9387`, PoseNet `0.059022`, SegNet `0.032986`
      - saved best: epoch `115`, scorer `3.8837292766571045`, int8 size `94,087`
      - status: active but still very weak
   - `segnet_attack_fixed_v2`
      - log: `/private/tmp/pact-mine/experiments/postfilter_weights/train_segnet_attack_fixed_v2.log`
      - latest printed checkpoint: epoch `900`
      - best observed local scorer so far: epoch `590`, scorer `0.9819`, PoseNet `0.035829`, SegNet `0.005334`
      - blocker: no saved best fp32/int8 artifact on disk yet; live process most likely predates the synced checkpoint-saving trainer
   - `segnet_attack_h64`
      - log: `/private/tmp/pact-mine/experiments/postfilter_weights/train_segnet_attack_h64.log`
      - latest printed checkpoint: epoch `350`, scorer `1.0854`, PoseNet `0.047744`, SegNet `0.005653`
      - blocker: no saved artifact visible yet; live process most likely predates the synced checkpoint-saving trainer

## leaderboard context

- Official leaderboard check at `2026-04-09 06:47:39 -0500` from `https://comma.ai/leaderboard`:
  - `1.89` first place: `neural_inflate`
  - `1.94` second place: `roi_v2`
  - `1.95` third place: `av1_roi_lanczos_unsharp`
- Current gap:
  - `0.16` better than current public first place
  - `0.21` better than current public second place
  - `0.22` better than current public third place
