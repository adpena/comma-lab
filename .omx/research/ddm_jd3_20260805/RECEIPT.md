# ddm_jd3 #366 Reroute Receipt

Date: 2026-08-05

Status: BLOCKED-BEFORE-SMOKE by local MLX/Metal access, after build + ticket reseal.

## RECALL EVIDENCE

- Read `.omx/tmp/codex_runs/jd3_prompt.md` and `_common_contract.md`: jd3 is build+reseal only; bounded re-smoke must use `tools/launch_detached_process.py`; full FIRE stays MAIN.
- Refreshed `.omx/state/main_hot_state.md`: live task is ddm_jd3 #366 reroute from jd1 v2, with own-vehicle pointer `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
- Read `.omx/research/ddm_tp1_boundary_receipt_20260805.md` and the jd1 v2 receipt/telemetry on SSD: v2 failed through live realized seg erosion while loss-space hold stayed below floor; EMA basis was parent-horizon scoped.
- Read `.omx/research/ddm_gc19_20260805/OP_ROUTABLES.md`, `.omx/research/operator_directive_per_edge_optimality_criteria_20260805.md`, and `.omx/research/ddm_sl2_20260805/SL2_RECEIPT.md`: seg must be held in the realized surface, pose is only admissible after conditioned seg, and MAIN owns full FIRE/exact promotion.
- Read `.omx/research/ddm_jd3_20260805/MAIN_ADDENDUM_OPTIMAL_CONVERSION.md`: do not silently stack extra treatments into v3; queue SL2 solved-frame teacher and PE3 conditioning as separate v4 riders, and use MAIN's both-bases sweep if it lands before resume-start adjudication.
- Memory recall checked `MEMORY.md` for #366/frontier dispatch context before edits; no full exact/contest score claim is made here.

## Build

Implemented jd3 v3 controls in `experiments/train_tr1_partition_renderer_mlx.py`:

- `--jd1-seg-hold-space realized`: latch first post-engagement realized-gate `d_seg` as floor; derive margin from `sd(per-pair d_seg)/sqrt(n_gate)` when margin flag is `0.0`.
- Real rollback path: when a later realized gate exceeds floor+margin, restore the previous accepted gate snapshot, retreat effective pose pressure, and replay from the next epoch.
- `--jd1-ema-stage-scope window`: preserve the parent EMA shadow under `jd1_parent_ema::` arrays in `stage_joint_pose_finish_entry.npz`, re-anchor active EMA to live weights, and derive active decay from the remaining joint-pose stage window.
- `--jd1-live-gate-telemetry on`: log separate `jd1_live_basis_gate` rows while leaving the normal A1 EMA gate/checkpoint tail intact.

Extended `experiments/ddm_jd1_ticket_regenerate.py` with `--v3` ticket generation for both jd3 resume candidates.

Added jd3 derivation/validation coverage to `src/tac/tests/test_ddm_bp1_boundary_reset_race.py`.

## Tickets

| candidate | ticket | ticket_hash | resume checkpoint | smoke window |
|---|---|---:|---|---:|
| entry_ep1336 | `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_entry_ep1336.json` | `81383a2cddb6190a97626d9a2c4de2c9dac8385e499907deb4f7fad21a33e774` | `stage_joint_pose_finish_entry.npz` epoch 1336 | 8 epochs |
| refuse_final_ep1354 | `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_refuse_final_ep1354.json` | `c62894e0bf72921d82bcb9d8d61d3e5bbf23da03ce10c2fcb4c9d27db5b03ee6` | `stage_joint_pose_finish_final.npz` epoch 1355 | 8 epochs |

Full confirm was deliberately dropped from both tickets; MAIN keeps full FIRE/exact replay ownership.

## Bounded Smoke Attempts

Both candidates were launched through `tools/launch_detached_process.py`, one at a time.

| candidate | launcher dir | done receipt | rc | outcome |
|---|---|---|---:|---|
| entry_ep1336 | `.omx/research/ddm_jd3_20260805/launch_entry_ep1336` | `.omx/tmp/codex_runs/ddm_jd3_entry_ep1336.done` | 1 | blocked before trainer startup |
| refuse_final_ep1354 | `.omx/research/ddm_jd3_20260805/launch_refuse_final_ep1354` | `.omx/tmp/codex_runs/ddm_jd3_refuse_final_ep1354.done` | 1 | blocked before trainer startup |

Both logs stop at the same import boundary:

`RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.`

This is not a jd3 negative result. No epoch trained, no realized gate ran, no controller floor was latched, and no candidate was selected by smoke evidence.

## Verification

- `py_compile`: PASS for trainer, ticket generator, and test file.
- `pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py -q -k jd3`: PASS, 4 passed / 35 deselected. MLX emitted the same no-Metal atexit warning.
- Full `pytest src/tac/tests/test_ddm_bp1_boundary_reset_race.py -q`: BLOCKED in this sandbox, 9 failures from `mlx.nn`/`mlx.optimizers` importing without a Metal device; 30 tests passed before the environment-blocked failures.
- Detached smokes: both launched; both rc=1 at the same no-Metal import boundary before training.

## SHA-256

| artifact | sha256 |
|---|---:|
| `experiments/train_tr1_partition_renderer_mlx.py` | `bf7190ceaecf0ad168bce6c4c843015b123b362ea48158fc645c089e8d065e75` |
| `experiments/ddm_jd1_ticket_regenerate.py` | `2e94af05690fb56466fa5d5aa0a592b10f3a07c6a71d0f942e82e4d9e4959926` |
| `src/tac/tests/test_ddm_bp1_boundary_reset_race.py` | `c78a9e56319d243d5a2172fc321b52f58dd0fbad5fa042f09a66cdb046e23333` |
| `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_entry_ep1336.json` | `cdbf2338c56082a56e1bd4d79d0f0b9e7f2be0d57e12539a71ab17666c35b1b2` |
| `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_refuse_final_ep1354.json` | `71dc8c75f7c6c6123ddd28a36a619ddc8036deedaf1324e1b514db1762eee376` |
| `.omx/research/ddm_jd3_20260805/launch_entry_ep1336/launch_manifest.json` | `971a3cf22ddf8c7ab8874a6cf9b781aa2229d8499ab94700e80f221c63bf036f` |
| `.omx/research/ddm_jd3_20260805/launch_entry_ep1336/run.log` | `7611b58abb03132d8e30e77a9ccf1b3998e142bff3c211939b6c8b3cb0acc2e4` |
| `.omx/research/ddm_jd3_20260805/launch_refuse_final_ep1354/launch_manifest.json` | `5b63dd059b0a99ca6adc1fc88edb18ae070ab278cef675d0b52ecc9bd9872471` |
| `.omx/research/ddm_jd3_20260805/launch_refuse_final_ep1354/run.log` | `2af02fe9e919b84aa50e6acc497351d2644d60924359a0953145726df71b303d` |

## Follow-ons

QUEUED-WITH-FIRE-ORDER:

1. On a Metal-access host, re-run the entry ticket with `tools/launch_detached_process.py`; require at least one post-engagement realized gate and a latched jd3 floor.
2. Then re-run the refused-final ticket with the same launcher; do not run concurrently.
3. If `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/chain_both_bases_sweep.json` exists before adjudication, include it in resume-start choice; otherwise proceed with the two charter candidates.
4. Compare realized-gate trajectories, rollback history, effective pose weight, live-vs-EMA basis rows, and final receipts. Pick only after both smokes produce real gates.
5. If a candidate preserves realized seg while improving pose, hand MAIN the selected ticket for full FIRE/exact replay. If both fail with real gate evidence, fold jd3 v3 and route the next #366 controller.
6. Keep SL2 solved-frame teacher distill, PE3 conditioning, and EN1 margin-weight as separately-flagged v4 riders. Do not silently stack them into v3.

Final line: S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
