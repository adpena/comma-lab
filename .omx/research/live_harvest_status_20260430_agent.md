# Live Harvest Status Audit - 2026-04-30

Author: Codex agent
Checked at: 2026-04-30T19:28:57Z to 2026-04-30T19:30Z

Scope: live Vast status, `.omx/state/vastai_active_instances.json`,
`.omx/state/active_dispatches.md`, recent recovered and harvested artifacts,
and the canonical harvest provenance audit. This is not a new score ledger.
Score claims remain limited to exact CUDA auth eval on exact archive bytes via
`archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Commands Run

- `date -u`
- `.venv/bin/vastai show instances --raw`
- `.venv/bin/python scripts/reconcile_vast_dispatch_state.py`
- Read-only SSH probes of live Vast instances: `find`, `tail`, and `ps`.
- Local scans of `experiments/results`, recovered `recovery_metadata.json`,
  harvested Modal artifacts, `contest_auth_eval.json`, archive SHA-256 values,
  `.omx/state/active_dispatches.md`, `.omx/state/vastai_active_instances.json`,
  and `.omx/research/canonical_harvest_provenance_audit_vast_modal_lightning_20260430_worker_d.md`.

No artifacts were copied from Vast and no instances were destroyed.

## Current Live Vast Truth

`vastai show instances --raw` returned four live instances. The local tracker
has 204 rows and reconciliation found 200 tracker rows missing from live Vast.
`active_dispatches.md` has three stale active rows by instance id:
`35899435`, `35899552`, and `35899275`.

| Instance | Label | Live status | Remote evidence | Harvest decision |
|---|---|---|---|---|
| `35885106` | `lane_hm_s_2026-04-30_b_a2` | running, but GPU 0% at inventory | Training completed at 18:11Z and Stage 3 packing started. `ps` shows no lane, Python, pack, or eval process. Files present: `segmap_weights.tar.xz`, train checkpoint, logs, provenance. No `archive*.zip` and no `contest_auth_eval.json`. | Diagnostic-harvestable only. Score-blocked/run-abort until packing/eval is forensically resolved. |
| `35899850` | `lane_19_logit_margin_2026-04-30_b_a4` | running, GPU active | Active `train_renderer.py`; latest inspected log around ep `1100/1980`, ETA about 5.8h. Checkpoints/proxy telemetry exist. No archive or auth JSON. | Not harvestable for score. Monitor until archive plus lane-local exact CUDA JSON exists. |
| `35906669` | `lane_sa_segmap_clone_2026-04-30_codex_a2` | running, GPU active | Active `train_segmap.py`; latest inspected log around epoch `468/600`. Logs/provenance only. No archive or auth JSON. | Not harvestable for score. Monitor. |
| `35907873` | `lane_h_v3_joint_halfframe_2026-04-30_codex_a4` | running, GPU active | Active `train_renderer.py`; latest inspected log around ep `1150/1980`, ETA about 3.2h. Checkpoints/proxy telemetry exist. No archive or auth JSON. | Not harvestable for score. Monitor. |

Live rows missing from `active_dispatches.md`: HM-S, SA clone, and H-V3.
Lane 19 is live by normalized label, but the table still points at stale
attempt `35899435` instead of live attempt `35899850`.

## Harvestable Or Already Harvested

| Artifact | Status | Evidence |
|---|---|---|
| `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/` | Deploy-grade A++ packet remains the only clean promotion packet in this audit. | `archive/archive.zip` is 686635 bytes, SHA-256 `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`; exact CUDA T4 eval score recomputes to `1.043987524793892`, 600 samples, `gpu_t4_match=true`. |
| `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/` | Already harvested A-negative measured implementation/config regression. | Archive is 296478 bytes, SHA-256 `864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97`; exact CUDA eval recomputes to `26.03719330455429`, 600 samples. This supports only scoped NeRV implementation/config retirement, not a family kill. |
| Live HM-S `35885106` | Diagnostic harvestable, not score harvestable. | Training weights and logs exist remotely, but no lane archive or JSON exists and no lane process is alive. |

## Need Exact Eval Rerun

These have archive bytes or near-archive artifacts but lack promotable exact
CUDA custody.

| Artifact | Reason | Required action |
|---|---|---|
| `experiments/results/recovered_35793092_lane_sz_phase2_c/.../lane_sz_phase2_results/archive_lane_sz.zip` | Archive exists, 3388 bytes, SHA-256 `6c99778a407522a3f591842cdd7ea56a96d3bbc313f69cdedd5800aef2ac32a0`. CUDA eval provenance exists, but no `contest_auth_eval.json`; run appears to have stopped after Stage 4 began. | Rerun canonical exact CUDA eval on this exact archive before any score or failure claim. |
| `experiments/results/lane_uniward_v8_modal/lane_uniward_results/archive_lane_uniward.zip` | Modal CPU JSON only: score recompute `1.1350939985261173`, device `cpu`, archive SHA `74bc09803a7cbca6a6220f3d302c5e6b8cb5ca2af37812e77bfef5d0a21d4080`. | Rerun on CUDA exact path if still scientifically useful. |
| `experiments/results/lane_lane_mm_v2_modal/lane_mm_grayscale_lut_results/archive_lane_mm.zip` | Modal CPU JSON only: score recompute `2.6326019733574295`, device `cpu`, archive SHA `4cb2d97e7ce2ffdfbcfce0718c039bcf729e5f153d03af974268db27584f6f2c`. | Rerun on CUDA exact path before ranking. |
| `experiments/results/lane_lane_gp_v3_modal/lane_gp_results/archive_lane_gp.zip` | Modal CPU JSON only and poor CPU result: score recompute `89.66694444427259`, device `cpu`, archive SHA `f731285c50cfd02d16de152010b83daebd747f7ec343e362b5f2647d55afa968`. | CUDA rerun only if forensics need it; do not rank from CPU. |
| `experiments/results/lane_uniward_v7_modal/lane_uniward_results/archive_lane_uniward.zip` | Modal CPU JSON only: score recompute `53.60783559031081`, device `cpu`, archive SHA `6e5d4c8c9eb6b58d3ccdd7a320261a065827533ad4cb1cf988322a717763439d`. | CUDA rerun only if needed for diagnostics. |

## Blocked Or Invalid For Score

| Lane/artifact | Classification | Rationale |
|---|---|---|
| HM-S live `35885106` | Blocked/run-abort until reviewed | Finished training, entered pack, then left no lane process, no archive, no JSON. KL-active (`variant=kl_distill`), so any future result also needs component-gate review. |
| Lane 19 live `35899850` | Blocked in-progress | Training proxy logs and checkpoints are not evidence. Wait for final archive and exact CUDA JSON. |
| SA clone live `35906669` | Blocked in-progress | Training only. No archive/eval artifact. |
| H-V3 live `35907873` | Blocked in-progress | Training only. No archive/eval artifact. |
| `active_dispatches.md` rows for Lane 8 and Lane 17 | Stale/blocked | Reconciliation found no live matching instances. Local results only show broad recovery snapshots or the J-IMP cycle-0 abort. |
| `experiments/results/lane_j_imp_crashed_cycle0/` | Run abort | Cycle-0 eval hit checkpoint shape mismatch (`motion.head` shape mismatch). No score result and no method kill. |
| `experiments/results/lane_20_balle_2026-04-30_a1_recovered/` | Empirical/no-op fallback | Static codec won; lane ships zero new bytes and skipped exact eval by design. Useful byte diagnostic only. |
| `experiments/results/lane_g_v3_omega_w_v2_stack_landed/` | Custody-blocked CUDA diagnostic | JSON says CUDA score recomputes to `1.0701007499356794`, archive SHA `eba8e4360e6366ca10905b58e9ec1d12b3480c78b041548daa21da0c46c31625`, but that exact archive is not present locally. Recover exact bytes or rebuild as a new archive and rerun. |
| OWV3/Fisher r2 and smoke Modal OWV3 artifacts | Build/sensitivity only | Archives exist, but no canonical CUDA score eval; current audit and byte-plan docs classify them as byte-blocked or smoke. |
| Recovered `recovered_*` snapshots | Recovery-only unless lane-local archive and JSON exist | Most `recovery_metadata.json` files point `archive_zip` at Lane A or Lane G anchor archives due broad workspace salvage. Do not attribute those archives to recovered lanes. |
| Modal harvested artifacts generally | Invalid for score | `experiments/modal_train_lane.py` forces `AUTH_EVAL_DEVICE=cpu`; logs may show T4 hardware, but `contest_auth_eval.json` provenance device is `cpu`. |
| `submissions/robust_current` archives inside harvested/recovered trees | Invalid for lane attribution | These are stale sidecars or anchor/baseline files, not lane-local completion artifacts. |

## Next Actions

1. Harvest HM-S diagnostics only if needed: logs, `segmap_weights.tar.xz`,
   `segmap_inference.pt`, `segmap_train.pt`, and provenance. Do not call it a
   scored lane.
2. Keep monitoring Lane 19, SA clone, and H-V3 until each emits a lane-local
   archive plus `contest_auth_eval.json`.
3. Rerun exact CUDA eval for SZ phase2 if the 3388-byte archive is still worth
   adjudicating.
4. Route only selected Modal CPU archives to Lightning/Vast exact CUDA. UNIWARD
   v8 and MM v2 are the only Modal CPU artifacts here with enough apparent
   upside to prioritize.
5. Recover exact Omega-W-V2 archive SHA
   `eba8e4360e6366ca10905b58e9ec1d12b3480c78b041548daa21da0c46c31625`, or
   treat any rebuild as a new archive requiring its own exact eval.
6. Before the next dispatch/harvest wave, clean or supersede stale
   `active_dispatches.md` and tracker rows so stale instance IDs do not drive
   duplicate recovery or false status conclusions.
