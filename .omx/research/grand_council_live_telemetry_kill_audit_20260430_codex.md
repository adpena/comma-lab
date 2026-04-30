# Grand Council Live Telemetry Kill Audit - 2026-04-30

Checked at: 2026-04-30T21:12:21Z
Author: Codex

Scope: user-supplied live Vast/Modal/Lightning telemetry, local lane scripts,
local harvested snapshots under `experiments/results/live_*`, and contest-grade
evidence rules. No remote instances were stopped or modified in this pass.

This is not a score ledger. No current live/harvested lane in this audit has a
lane-local archive plus CUDA `contest_auth_eval.json`, so none can promote,
rank, kill a family, or anchor Shannon-floor math.

## Global Verdict

- No lane-family or method kill is justified from the current telemetry.
- HM-S, SA, and H-V3 are current-run engineering failures or run aborts only.
- Lane 19 is still an active diagnostic/training run; its proxy telemetry is
  poor, but proxy telemetry is non-promotable and non-killing.
- Modal apps with `Tasks=0` require no live kill action. Existing Modal CPU
  artifacts remain advisory unless rerun through CUDA exact eval.
- Lightning r4 pending is not evidence either way; wait for its canonical
  adjudication packet.

## Vast Instance Recommendations

| Instance | Lane | Evidence observed | Recommendation |
|---|---|---|---|
| `35885106` | HM-S SegMap homography KL variant | Local harvest has training logs, `segmap_inference.pt`, and `segmap_weights.tar.xz`; no lane archive, no `auth_eval.log`, no `contest_auth_eval.json`. Local replay of Stage 3 guard failed: `verify_roundtrip: layer_in.weight MSE 0.000296662 > tol 1e-06`. | Harvest diagnostics and retire the current run/instance after custody is complete. Classify as current-run engineering failure at Stage 3 pack, not homography-family failure and not KL-family proof. |
| `35906669` | SA SegMap clone plain variant | Local harvest has training logs, `segmap_inference.pt`, and `segmap_weights.tar.xz`; no lane archive, no `auth_eval.log`, no `contest_auth_eval.json`. Local replay of Stage 3 guard failed: `verify_roundtrip: layer_in.weight MSE 0.000322065 > tol 1e-06`. | Harvest diagnostics and retire the current run/instance after custody is complete. Classify as current-run engineering failure at Stage 3 pack, not SegMap-family failure. |
| `35907873` | H-V3 joint half-frame | Local harvest has checkpoints, telemetry, `zoom_scalars.pt`, and logs; no archive/eval JSON. Training crashed at Phase 3 in `segnet_uncertainty_weighted_loss` with SegNet receiving a 1-channel tensor. Local code shows `_hwc_to_chw(... )[:, :, 0]` selects channel 0 while leaving the singleton time dimension, producing `(B,1,H,W)` instead of `(B,3,H,W)`. | Harvest diagnostics and retire the current run/instance after custody is complete. Classify as current-run engineering failure in the uncertainty-loss helper. Do not kill half-frame revival from this. |
| `35899850` | Lane 19 logit-margin | Active training snapshot only; no archive/eval JSON. Logs show profile validation with `loss_mode=logit_margin`, `logit_margin_weight=10.0`, `kl_distill_weight=0.0`, and Check 93 passing. Proxy/best around 40 with PoseNet around 150 is bad but non-canonical. | Keep only if budget tolerates waiting for exact archive/eval. If stopped, call it a cost/proxy run abort, not a lane kill. Do not launch duplicate Lane 19 attempts until this run either completes or is explicitly aborted. |

## Code/Script Implications

- HM-S and SA both reached full CUDA training completion, wrote inference
  checkpoints, and wrote block-FP payloads. The absence of a Stage 4 log line
  is consistent with the strict `verify_roundtrip(..., tol=1e-6)` guard failing
  after payload write. This is a packaging/codec-contract problem, not contest
  evidence.
- The SegMap block-FP packer appears intentionally lossy at the observed
  tolerance scale. Before rerunning, decide whether the payload must be exact
  enough for a strict tensor round-trip or whether the lane should permit lossy
  payloads only after archive-level CUDA exact eval. Either way, capture Stage
  3 stdout/stderr in `run.log`.
- H-V3 needs a unit test and code fix for `segnet_uncertainty_weighted_loss`:
  the helper should feed SegNet a 3-channel BCHW frame. The current bug is a
  mechanical tensor-indexing failure.
- Lane 19 can become Grade A only if its final `archive_lane_19.zip`,
  `eval_work/contest_auth_eval.json`, provenance, logs, and SHA manifest are
  harvested together. The script currently lacks the JSON adjudicator used by
  newer lanes, so downstream adjudication should be run locally after harvest.

## Kill Discipline

- HM-S: current-run engineering kill only. Homography, SegMap, and KL-like
  ideas remain unproven, with known confounds.
- SA: current-run engineering kill only. Plain SegMap clone remains untested
  at archive/eval level for this attempt.
- H-V3: current-run engineering kill only. Half-frame family is not killed.
- Lane 19: keep/monitor or cost-abort. No scientific kill without exact CUDA
  archive evidence.

## Next Actions

1. Finish diagnostic harvest for `35885106`, `35906669`, and `35907873`, then
   stop/destroy those idle or crashed Vast instances to avoid spend.
2. Patch and test `segnet_uncertainty_weighted_loss` before any H-V3 rerun.
3. Fix the SegMap packaging contract or relax it explicitly as a lossy codec
   gate, then rerun SA before drawing any SegMap conclusion.
4. Let Lane 19 finish only if budget allows. Otherwise stop it as a run abort
   due to bad proxy telemetry and missing exact archive, preserving all logs.
5. Treat Modal `Tasks=0` as no live action. Treat Lightning r4 as pending.

## Codex Execution Delta - 2026-04-30T21:26Z

Actions taken after the independent audit:

- Harvested diagnostic/custody artifacts and destroyed Vast instances
  `35885106`, `35906669`, `35907873`, and `35899850`.
- Detected and destroyed duplicate/orphan Vast relaunches
  `35925274`, `35925374`, `35925475`, `35925801`, `35925825`, and
  self-test escape `35925916`.
- Final Vast live inventory after cleanup:
  `.omx/state/vastai_show_instances_live_final_20260430.json` = `[]`.
- Modal app list still shows `Tasks=0` for all apps; no Modal kill action
  needed.
- MCP helper processes were killed again. Post-kill process sweep matched only
  the checking `rg`; helpers may respawn from the outer app, not repo config.

Current-run classifications:

- HM-S, SA, H-V3: current-run engineering failures only; no lane-family kill.
- Lane 19: current-run cost/proxy abort only; no lane-family kill and no score
  evidence.
- Lane 20 q2: duplicate spend killed; Lane G-v3-anchor Ballé remains narrowly
  retired/no-op until a real non-static byte win exists before GPU eval.

Lightning r4 update:

- SDK status became `Failed`, but failure occurred after exact CUDA eval.
- Harvested focused artifacts to
  `experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4/`
  with `SHA256SUMS`.
- CUDA/T4 result packet:
  `score_recomputed_from_components=1.0378905176070103`,
  `final_score=1.04`, `avg_posenet_dist=0.00319052`,
  `avg_segnet_dist=0.00402120`, `archive_size_bytes=686557`,
  archive SHA
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`.
- Adjudication failed the SegNet relative component gate:
  `0.00402120 / 0.00400656 = 1.003654`, above the `1.002` cap.
- Verdict: promising exact score evidence, but not promotable under the
  current gate and not a regression/family kill. Requires Grand Council review
  of component gate policy and/or byte-plan/sensitivity mitigation before any
  claim.

Bug class fixed:

- `scripts/launch_lane_with_retry.py` now uses `logical_lane_key()` for both
  advisory lock files and live Vast duplicate detection. Timestamped labels
  such as `_q1_20260430T...`, `_q1c_20260430T...`, and non-numeric queue tags
  now collapse to one logical lane key, preventing the duplicate-dispatch
  class observed for Lane 19/20.
- Added `.omx/state/dispatch_holds.json` and a launcher-level
  `FATAL_DISPATCH_HOLD` guard. Lane 19 and Lane 20 are held fail-closed until
  Grand Council records clearance.
- Verification: `py_compile` passed for the launcher/preflight/test files and
  `src/tac/tests/test_remote_auth_eval_hardening.py` passed:
  `22 passed in 1.46s`.

## Codex Swarm Integration Delta - 2026-04-30T21:49Z

Follow-up repairs and classifications:

- H-V3 current-run failure is repaired at the tensor boundary:
  `segnet_uncertainty_weighted_loss` now preserves 3-channel BCHW before
  SegNet preprocessing. This clears the mechanical rerun blocker but does not
  retroactively promote or kill the prior run.
- HM-S/SA SegMap current-run failures are repaired at the pack-contract layer:
  block-FP is explicitly lossy with a `1e-3` MSE gate, lossy metadata, and
  `segmap_pack_roundtrip.json`. Exact CUDA archive eval remains mandatory.
- Lane 19/20 hold guard is stronger than the JSON file: the launcher checks
  lane-specific clearance requirements and blocks if they are unmet even when
  the hold entry is absent or marked cleared.
- Lightning supply-chain status remains clean locally:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_current.json`,
  `status=OK`, `violation_count=0`.
- MCP helper processes were killed and all discovered MCP config server maps
  were emptied; strict config scan passes.

No telemetry reviewed in this document justifies a broad family kill. Prior
negative/aborted runs remain scoped engineering failures or cost/proxy aborts
until exact CUDA archive evidence proves otherwise.
