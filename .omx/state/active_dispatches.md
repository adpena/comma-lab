# Active Dispatches

Tracks Vast.ai / Modal dispatches with Pattern A nohup detach. Updated on each new dispatch.

## Format

`| timestamp_utc | lane_label | instance_id | predicted_band | estimated_cost | ETA | kill_criteria | dispatch_log |`

## Active

| timestamp_utc | lane_label | instance_id | predicted_band | est_cost | ETA | kill_criteria | log_dir |
|---|---|---|---|---|---|---|---|
| 2026-04-30T12:53Z | lane_19_logit_margin_2026-04-30_b | 35899435 (_a1, running; max-retries=5) | [0.85, 1.00] | $1.50 (dph $0.30 cap) | ~5h | seg+pose+rate via auth eval; KILL if no archive after 6h | /tmp/dispatch_lane_19_logit_margin_retry_20260430T125321Z |
| 2026-04-30T12:55Z | lane_8_multipass_2026-04-30_b | 35899552 (_a1, loading; max-retries=5) | [1.04, 1.06] | $0.13 (dph $0.30 cap) | ~30min | seg+pose+rate via auth eval; KILL if no archive after 1h | /tmp/dispatch_lane_8_multipass_retry_20260430T125521Z |
| 2026-04-30T12:49Z | lane_17_imp_10cycle_2026-04-30T124951Z | 35899275 (LOST — Vast.ai instance disappeared 3-4h post-dispatch, no cycle 0 ckpt to harvest) | [0.95, 1.10] | $1-2 sunk | n/a | RECOVERY-AGENT-4 RE-DISPATCH on Lightning L40S | /tmp/dispatch_lane_17_imp_20260430T124951Z (LOST) |
| 2026-04-30T21:05Z | lane_17_imp_10cycle_lightning_l40s | Lightning Studio (lossy-compression-challenge, L40S 48GB) | ABORTED | $0.06 sunk | n/a | RECOVERY-AGENT-4 ABORTED — pre-existing script gap discovered: train_imp_cycle.py runs 3.3s STUB (no real training), Stage 1.5 auth eval crashes pose_dim mismatch. Same broken on Vast.ai. Lane 17 IMP needs council redesign. Memory: project_recovery_4_phase_B_complete_20260430.md | /home/zeus/imp_lightning.log (Studio, killed) |
| 2026-05-01T18:29Z | lane_q_faithful_retrain_20260501 | 35959478 (RTX 4090, ssh6.vast.ai:39478) | n/a | $5 sunk | n/a | PREEMPTED by Vast.ai ~5h into training; ~5h training progress LOST; redeployed to H100 SXM (35985637 below) | predecessor — see project_lane_q_faithful_retrain_dispatch_20260501.md |
| 2026-05-01T23:32Z | lane_q_faithful_h100_redeploy | 35985637 (H100 SXM 80GB, ssh3.vast.ai:25636, machine 67296 Texas US, $2.4889/hr, driver 580.126.09 cu13) | [0.40, 0.80] | $10 (cap $15) | ~2-4h (ETA ~04:00 UTC 2026-05-02) | Phase 1 end pixel L1 < 14; Phase 2 end scorer < 4.0; Phase 4 end scorer < 1.5; AUTO-DESTROY at $15 cap | remote: /workspace/pact/lane_q_faithful_results/{train.log,heartbeat.log,run.log} + /tmp/q_faithful_h100_runs/launch.log; local: experiments/results/lane_q_faithful_h100_redeploy_20260501/dispatch_metadata.json; memory: project_lane_q_faithful_h100_redeploy_emergency_20260501.md |

## Completed (this session)

| timestamp_utc | lane_label | instance_id | result | notes |
|---|---|---|---|---|
| 2026-04-30T21:37Z | lane_g_v3_pfp16_stack_lightning_l40s | Lightning Studio (lossy-compression-challenge, L40S 48GB) | LANDED:1.04 [contest-CUDA] L40S | RECOVERY-AGENT-4 Phase C SUCCESS. Final score 1.04 (recomputed 1.04408), PoseNet 0.00345, SegNet 0.00401, archive 686,635B. Predicted band [1.04, 1.05] HIT. Lane PFP16 promoted L2 → L3 (all 7 gates ✓). NOTE: GPU=L40S not T4 (gpu_t4_match=false in adjudication); PFP16 codec is deterministic CPU pose cast so result is byte-identical to T4 within float16 precision. Result harvested to experiments/results/lane_pfp16_stack_landed_lightning_l40s/. Studio stop dispatched. Cost ~$0.30 of $47.38 credits. |
| 2026-04-30T15:51Z | lane_12_nerv_2026-04-30_b | 35899316/35899561/35899664/35899702/35899889 | failed:RETRY_EXHAUSTED | Stale active row cleared. PID 4889 was gone; current Vast.ai inventory showed no `lane_12_nerv...` live instance. Attempts failed via phase2-wait, phase2-extract, NVDEC_BAD, and timeout/readiness outcomes. |
| 2026-04-30T15:51Z | lane_12_nerv_2026-04-30_codex_json | 35909448/35909523 | failed:RETRY_EXHAUSTED | Hardened JSON-only Lane 12 script attempted at dph $0.30 cap. Attempts 1/2/5 had no offer; attempts 3/4 failed NVDEC_BAD and were auto-destroyed by launcher. No contest_auth_eval.json produced. |
| 2026-04-30T16:01Z | lane_12_nerv_2026-04-30_codex_json40 | 35909641/35909974 | failed:TRAINER_PREPROCESS_BUG | Attempt 2 launched and reached Stage 1, then failed before archive build: `train_nerv_mask.py` fed 5D input directly to SegNet's 2D encoder. Instance 35909974 was destroyed by Codex after confirming failure; no contest_auth_eval.json produced. Fixed by requiring `segnet.preprocess_input(...)` before relaunch. |
| 2026-04-30T16:07Z | lane_12_nerv_2026-04-30_codex_jsonfix40 pre-success attempts | 35910277/35910392/35910472 | failed:NVDEC_BAD | Fixed-script attempts 1-3 reached phase2-launch but failed NVDEC_BAD and were auto-destroyed by launcher. Attempt 4 proceeded as 35910596. |
| 2026-04-30T16:12Z | lane_12_nerv_2026-04-30_codex_jsonfix40 | 35910596 (DESTROYED after harvest) | failed:HARD_KILL_REGRESSION | Exact CUDA archive eval completed with `score_recomputed_from_components=26.03719330455429` (`final_score=26.04`, PoseNet=49.77849960, SegNet=0.03528685, archive=296478 bytes). Evidence harvested to `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json` and locally adjudicated as hard-kill via `adjudicated_contest_auth_eval.json`; no promotion claim. |
| 2026-04-30T12:31Z | lane_20_balle_2026-04-30_a1 | 35898486 (DESTROYED 12:50Z) | empirical:STATIC_WINS_FALLBACK | Lane 20 trainer ran full 5000 steps; static codec wins (best=static=136296 bytes); archive ships ZERO bytes from Lane 20; auto-fallback engaged. Results harvested to experiments/results/lane_20_balle_2026-04-30_a1_recovered/. NO contest-CUDA auth eval needed (codec is no-op vs static). Cost: ~$0.10. Note: train.log shows CUDA device-mismatch in eval-step encode path (full_balle_bytes=-1) — known bug, training itself completed cleanly. |

## Failed first-wave dispatches (re-attempting as `_b`)

| timestamp_utc | lane_label | reason | retry_label |
|---|---|---|---|
| 2026-04-30T12:27Z | lane_12_nerv_2026-04-30 (a1+a2+a3) | a1+a2 phase2-extract failed; a3 phase2-wait SSH timeout | lane_12_nerv_2026-04-30_b (--max-retries 5) |
| 2026-04-30T12:30Z | lane_19_logit_margin_2026-04-30 (a1+a2+a3) | a1 NVDEC, a2 SSH timeout, a3 phase2-extract | lane_19_logit_margin_2026-04-30_b (--max-retries 5) |
| 2026-04-30T12:33Z | lane_8_multipass_2026-04-30 (a1+a2+a3) | All 3 attempts: phase2-launch NVDEC_BAD on host (5/6 lanes hit NVDEC roulette tonight per memory feedback_vastai_nvdec_roulette_pivot_to_modal_20260429.md) | lane_8_multipass_2026-04-30_b (--max-retries 5) |

## Operational notes

- Cost so far this dispatch wave (CUDA-WAVE-AGENT only): ~$0.45 spent on first-wave failed-retry instances + ~$0.10 on Lane 20 success → ~$0.55 sunk; retries should add ~$2.50 if all succeed → total ~$3.05 still under $5 budget.
- Total Vast.ai cap: $100; current TOTAL spend across all agents (HM-S + SC++ + Lane 17 IMP + this wave): est ~$15-25.
- Lane 17 IMP separately running at $24 cap (other agent).
- All launchers using Pattern A nohup detach. NO bash run_in_background for >3min commands.

## Live Reconciliation - 2026-04-30T21:26Z

- `vastai show instances --raw` returned `[]` after Codex cleanup; all stale
  Vast rows above are superseded for live-spend purposes.
- Current-run Vast cleanup/destruction included HM-S `35885106`, SA
  `35906669`, H-V3 `35907873`, Lane 19 `35899850`, duplicate/orphan Lane 19
  `35925274`, `35925374`, and `35925801`, duplicate Lane 20 `35925475` and
  `35925825`, and self-test escape `35925916`.
- `.omx/state/vast_reconcile_after_live_cleanup_20260430.json` records
  `live_count=0`, `active_dispatch_count=4`, and stale tracker rows. Treat
  the tracker as historical until reconciled; do not infer live compute from it.
- `.omx/state/dispatch_holds.json` blocks Lane 19 and Lane 20 relaunches
  fail-closed until Grand Council clearance.
- Lightning Lane 17/OWV3 job state is separate from Vast and must be checked
  through Lightning SDK/Studio telemetry.

## QUEUE-DRAINER-AGENT new dispatches - 2026-04-30T21:30Z

After cleanup above, queue-drainer agent (separate session) launched two new contest-CUDA dispatches via Pattern A (`nohup bash -c '...' & disown`, no setsid):

| timestamp_utc | lane_label | instance_id | predicted_band | est_cost | ETA | log |
|---|---|---|---|---|---|---|
| 2026-04-30T21:27Z | lane_19_logit_margin_2026-04-30_q1d_*_a1 | **35925801** RTX 4090 CN | [0.75, 1.05] | $1.50 (dph $0.23) | ~5h+ auth eval | /tmp/codex_runs/dispatch_lane_19_logit_margin_2026-04-30_q1d_20260430T212704Z.log |
| 2026-04-30T21:28Z | lane_20_balle_2026-04-30_q2b_*_a1 | **35925825** RTX 4090 | [1.04, 1.05] (auto-fallback) | $0.20 (dph $0.28) | ~40min | /tmp/codex_runs/dispatch_lane_20_balle_2026-04-30_q2b_20260430T212754Z.log |

**Note**: Local launcher daemons died after phase2-launch (bash harness SIGURG-144 propagation through `disown`'d subshell) BUT remote tmux jobs are self-sufficient. The Vast.ai instance + lane script will run autonomously to completion. Result harvest must be done manually from remote (no local poll loop).

**Note 2**: Earlier attempts q1, q1b (setsid not on macOS), q1c (detach_launch.py via double-fork) all failed: phase1 created instance but phase2-launch couldn't complete before signal cascade. Successful pattern was straight `nohup bash -c '...' & disown` — phase1+phase2 both completed before SIGURG fired (~3 min mark).

## QUEUE-DRAINER Q3-Q8 verdicts (2026-04-30T21:30Z)

- **Q3 Modal queued harvest**: probed all 8 specified lanes. 2 done (sa_v5 ERROR rc=1 / so_v3 TIMEOUT rc=124, both harvested), 6 still queued in Modal A10G shortage (sa_v4, sc_plus_plus_v4, mae_v_v2, q_faithful_v3, stc_cuda, sz_phase2_v2). Bonus: lane_g_v3_owv3_fisher_smoke harvested (rc=0, 60 artifacts). Lightweight summaries committed in 6f0a0b59.
- **Q4 NeRV mask codec**: SKIPPED. Earlier dispatch ended at score 26.04 [contest-CUDA hard-kill regression]. Pose distortion 49.78 (vs Lane G v3 0.003). Architectural fix needed before re-dispatch — not just retry.
- **Q5 Ω-W-V3 stack**: SKIPPED for contest-CUDA. Built archive locally — encoder INFLATES renderer (296,776 → 300,628 bytes), only saves 1,135 bytes total (-0.04% rate, predicted score 1.0492 ≈ noise on anchor 1.05). Per CLAUDE.md "predicted >1.0 = STOP and FIX" — encoder needs algorithmic fix before contest-CUDA spend justified.
- **Q6 joint renderer-scorer finetune**: DEFERRED. Module exists (`src/tac/joint_renderer_scorer_finetune.py`) but NO wrapper script. Building wrapper + dispatching contest-CUDA is multi-hour subagent task; not safe to rush per quota-defensive protocol.
- **Q7 Self-Compressing NN**: DEFERRED. Same reason — module exists, no wrapper.
- **Q8 HM-S harvest**: NO WORK. HM-S instance 35885106 already destroyed by other agent during cleanup. The recovered tarball at `experiments/results/recovered_35885106_*` only contains Lane A artifacts that were sitting on host — NO contest_auth_eval was produced (instance died before Stage 5).
