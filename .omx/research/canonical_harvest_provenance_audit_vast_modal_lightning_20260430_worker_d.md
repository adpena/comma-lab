# Canonical Harvest / Provenance Audit - Vast, Modal, Lightning

Date: 2026-04-30
Worker: D
Scope: `experiments/results`, `scripts/remote_*`, Modal/Vast/Lightning
launch and harvest helpers, and current progress ledgers.

This is a custody and provenance audit, not a new score ledger. Existing
`contest_auth_eval.json` files are treated as score authority only when they
come from the canonical archive path and pass the device/sample/archive custody
gates. CPU, MPS, Modal-CPU, local proxy, byte-only, smoke, stale-log, and
memory-only outputs remain non-promotable.

## Commands / Inputs Audited

- Read source/progress ledgers:
  - `.omx/research/contest_grade_all_lane_results_audit_20260430.md`
  - `.omx/research/contest_grade_all_lane_results_audit_20260430_codex_progress.md`
  - `.omx/research/active_dispatch_harvest_status_20260430.md`
  - `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`
  - `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`
  - `.omx/research/lightning_pypi_compromise_security_review_20260430_codex.md`
  - `.omx/research/owv3_fisher_byte_aware_redesign_spec_20260430_codex.md`
- Scanned `experiments/results` for `contest_auth_eval.json`,
  `auth_eval_*`, archives, manifests, provenance, Modal harvest summaries, and
  recovered Vast snapshots.
- Reviewed representative remote and cloud helpers:
  - `scripts/remote_lane_pfp16_stack.sh`
  - `scripts/remote_lane_nerv.sh`
  - `scripts/remote_lane_g_v3_owv3_fisher_stack.sh`
  - `scripts/remote_lane_8_multipass.sh`
  - `scripts/remote_lane_omega_w_v2_stack.sh`
  - `scripts/remote_lane_sa_segmap_clone.sh`
  - `scripts/remote_lane_h_v3_jointly_trained_halfframe.sh`
  - `scripts/remote_lane_19_logit_margin.sh`
  - `scripts/remote_lane_20_balle.sh`
  - J-NWC/NWCS remote scripts
  - `scripts/launch_lane_on_vastai.py`
  - `scripts/launch_lane_with_retry.py`
  - `scripts/reconcile_vast_dispatch_state.py`
  - `scripts/verify_vast_instances.py`
  - `experiments/modal_train_lane.py`
  - `experiments/modal_recover_lane.py`
  - `experiments/modal_auth_eval.py`
  - `scripts/lightning_repro_workspace.py`
  - `scripts/launch_lightning_batch_job.py`
  - `src/tac/deploy/lightning/batch_jobs.py`
  - `scripts/pfp16_a_plus_plus_exact_t4_eval.sh`
- Ran read-only Vast reconciliation and SSH probes of the four live Vast
  instances. No artifacts were copied and no instances were destroyed.

## Claim-Capable Custody

These artifacts have the canonical pieces needed for contest claims, subject to
the scope stated here.

| Artifact | Grade / use | Custody finding |
|---|---|---|
| `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/` | A++ deploy/paper score custody | Contains `archive/archive.zip`, `archive/archive_manifest.json`, `eval/contest_auth_eval.json`, `eval/eval_provenance.json`, T4 GPU proof, inflate timing, run command, custody manifest, upstream commit, and quarantine of stale parser fields. This is the authoritative PFP16 packet. |
| `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json` | A++ source eval JSON | Same exact archive SHA/bytes as the final bundle. The raw eval dir does not colocate the archive, but the final bundle does. |
| `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z/` | Exact CUDA corroboration only | Archive and JSON are colocated, but hardware is RTX 4090 rather than T4/equivalent. Superseded by the T4 packet for contest-identity wording. |
| `experiments/results/lane_g_v3_landed/` | Historical exact CUDA artifact | Preserved archive and `contest_auth_eval.json` match by SHA and sample/device fields. Historical predecessor, not current frontier. |
| `experiments/results/lane_a_landed/` | Historical exact CUDA artifact | Preserved archive and `contest_auth_eval.json` match by SHA and sample/device fields. Valid baseline/history only. |
| `experiments/results/lane_m_v2_landed/` | Exact CUDA regression artifact | Preserved archive and JSON match by SHA and sample/device fields. Supports scoped implementation lessons only. |
| `experiments/results/lane_h_crf56/` | Exact CUDA regression artifact | Archive and auth-eval JSON are present; the auth dir includes `eval_work/archive.zip`. Supports scoped regression lessons only. |
| `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/` | A-negative measured implementation/config | Contains `archive_lane_12_nerv.zip`, `contest_auth_eval.json`, adjudication provenance, dispatch provenance, logs, and matching archive SHA. Retires this NeRV implementation/config only; it is not a NeRV/INR/mask-compression family kill. |

Do not promote duplicate copies of the same archives under recovered Vast
workspaces as independent evidence. They are custody duplicates or recovery
snapshots, not separate lane results.

## Diagnostic / Incomplete Custody

| Artifact / lane | Current classification | Required action before ranking/promotion/retirement |
|---|---|---|
| `experiments/results/lane_g_v3_omega_w_v2_stack_landed/contest_auth_eval.json` | CUDA diagnostic only | Exact archive SHA in JSON is not present locally. Recover or deterministically rebuild that exact archive before using the result in a Grade A table. |
| `experiments/results/lane_g_v3_owv3_fisher_lightning_20260430_codex_r2/` | CUDA/T4 Fisher/build artifact, no score claim | Has CUDA Fisher-derived sensitivity map, deterministic archive build provenance, and an archive, but `run_contest_eval=false`; no exact eval was run. It is also byte-blocked against the active comparator. |
| `experiments/results/lane_lane_g_v3_owv3_fisher_smoke_20260430_codex_modal/` | Modal smoke only | Build packet and archive exist; no canonical CUDA exact eval. Modal sidecars in `submissions/robust_current` are stale/non-lane score artifacts. |
| `experiments/results/lane_20_balle_2026-04-30_a1_recovered/` | Empirical codec measurement | Baseline measurement records static-wins/fallback behavior; no lane archive exact CUDA eval. |
| `experiments/results/lane_j_imp_crashed_cycle0/` | Run abort / engineering failure | Cycle 0 failed with checkpoint shape mismatch before usable exact eval. Not a method kill and not a score result. |
| `experiments/results/lane_stc_clean*` and `lane_stc_av1_residual_smoke` | MPS/local empirical only | Manifests show local/MPS or byte-only smoke. No score claim. |
| `experiments/results/recovered_*` | Recovery snapshots | Most recovered directories contain generic repo artifacts, stale anchor archives, old baseline auth JSON, or `submissions/robust_current` sidecars. Treat as artifact salvage only unless a lane-local `contest_auth_eval.json` and matching lane archive are present. |

## Modal Findings

Modal training harvests are useful for cheap smoke, build, and ablation work,
but not for score claims in their current form.

- `experiments/modal_train_lane.py` deliberately sets
  `AUTH_EVAL_DEVICE=cpu` in Modal and stubs the NVDEC probe. Any
  `contest_auth_eval.json` emitted by those jobs is Modal-CPU advisory even
  when the hardware field says T4.
- Modal CPU `contest_auth_eval.json` files found under GP v2/v3, MM v2, and
  UNIWARD v7/v8 have matching archive custody, but `provenance.device=cpu`.
  They cannot rank, promote, retire, or kill lanes.
- Many Modal result dirs contain `submissions/robust_current/auth_eval_renderer_fp4.json`
  sidecars. These are stale MPS/local robust-current artifacts and must not be
  attributed to the harvested lane.
- `experiments/modal_recover_lane.py` prints `AUTH SCORE` from any JSON/log
  with a `score` or `final_score` field. That console text is not an evidence
  grade; readers must inspect `contest_auth_eval.json` provenance and device.
- `experiments/modal_auth_eval.py` is not canonical despite its header: it
  runs `inflate_renderer.py` directly and invokes upstream evaluation with
  `--device cpu`. JSONs from this path lack the full canonical provenance/sha
  contract and are advisory only.

## Vast Findings

Read-only reconciliation on 2026-04-30 found:

- Live Vast instances: 4.
- Local tracker rows: 204, mostly historical/stale.
- `active_dispatches.md` rows: 3, all pointing at non-live instance ids.
- Live instances missing from the active-dispatch table: HM-S, SA clone, and
  H-V3. Lane 19 matches by normalized label but not by the stale instance id.

Current live probes:

| Live instance | Lane label | Read-only probe result | Harvest status |
|---|---|---|---|
| `35885106` | `lane_hm_s_2026-04-30_b_a2` | Training completed and Stage 3 pack started. Lane dir has heartbeat, provenance, run/train logs, training weights, and `segmap_weights.tar.xz`; no lane-local archive/eval JSON yet. Uses `variant=kl_distill`, so forensic/component gates apply. | Do not harvest as score yet. Watch for lane-local archive plus `contest_auth_eval.json`. |
| `35899850` | `lane_19_logit_margin_2026-04-30_b_a4` | Still training, no lane-local archive/eval JSON. Training logs show high scorer/PoseNet proxy values; those are not exact evidence. | Monitor only. |
| `35906669` | `lane_sa_segmap_clone_2026-04-30_codex_a2` | Still training, no lane-local archive/eval JSON. | Monitor only. |
| `35907873` | `lane_h_v3_joint_halfframe_2026-04-30_codex_a4` | Still training, no lane-local archive/eval JSON. Training logs show high proxy scorer values; not exact evidence. | Monitor only. |

Recovery folders from destroyed/old Vast instances are not enough for claims.
The recovery metadata often points `archive_zip` at the Lane G v3 anchor or
baseline archive because the recovery tool salvages broad workspace files. For
claim harvest, target only the lane result directory and require:

- lane-local final archive,
- `eval_work/contest_auth_eval.json` copied to the lane result root,
- `eval_work/provenance.json`,
- `auth_eval.log`,
- report,
- deterministic archive manifest/member hashes,
- source/staged-tree provenance,
- hardware and command provenance.

## Lightning Findings

Lightning is currently the cleanest promotion path when used through the strict
helpers.

- `scripts/pfp16_a_plus_plus_exact_t4_eval.sh` correctly uploads fixed archive
  bytes, verifies SHA/bytes locally and remotely, requires a T4 GPU, runs
  `experiments/contest_auth_eval.py --device cuda`, and fetches JSON/logs.
- The PFP16 final deploy bundle wraps the raw Lightning eval with archive
  manifest, custody manifest, GPU proof, timing, upstream commit, and stale
  parser quarantine.
- `scripts/lightning_repro_workspace.py` now provides manifest-based source
  and artifact staging. Future promotion runs should use it instead of ad hoc
  `rsync`.
- `src/tac/deploy/lightning/batch_jobs.py` and
  `scripts/launch_lightning_batch_job.py` are stricter than the older SSH/tmux
  path: exact-eval jobs require `--device cuda`, preserve
  `contest_auth_eval.json`, validate archive identity, and validate harvested
  artifact dirs without parsing human logs.
- Known PFP16 bundle gap: the raw Lightning staged tree was non-git, so the
  final bundle records local git/diff state and explicitly marks the missing
  remote staged-tree manifest. This is documented and does not invalidate the
  exact archive/eval custody, but future runs should avoid the gap.

## Remote Script Harness Risks

The modern scripts are moving toward structured JSON adjudication, but legacy
patterns remain.

Good patterns observed:

- PFP16, NeRV, SA, H-V3, OWV3, FL, GP, J-NWC, J-NWCS, and related newer
  scripts copy `eval_work/contest_auth_eval.json` or run
  `scripts/adjudicate_contest_auth_eval.py`.
- J-NWC/J-NWCS promotion scripts fail closed when `AUTH_EVAL_DEVICE` is not
  `cuda`.
- OWV3 fails closed for non-CUDA promotion unless explicit smoke override is
  set.
- Lightning Batch Jobs validate JSON/archive identity and reject non-CUDA.

Risk patterns still present:

- Several `scripts/remote_*.sh` files only check for `RESULT_JSON` in
  `auth_eval.log` and do not copy/adjudicate `contest_auth_eval.json`.
  Examples include current Lane 19 and Lane 20 scripts.
- Some sweep/older scripts still parse `RESULT_JSON` from log text for score
  selection. That may be acceptable for internal sweep control, but it cannot
  be a promotion or paper claim source.
- Modal recovery console output can label CPU/MPS values as auth scores.
- Recovered Vast snapshots include broad workspace files, so readers can
  accidentally attribute stale anchor/baseline artifacts to the recovered lane.

## What Must Not Be Ranked / Killed

- Any Modal CPU result, even if run on T4/A10G hardware.
- Any `auth_eval_renderer_fp4.json` under Modal/recovered `submissions/robust_current`.
- Any MPS/local proxy result or `true_auth_eval_round1.json`.
- GP v2/v3, MM v2, UNIWARD v7/v8 Modal outputs until exact CUDA rerun on the
  exact archive bytes exists.
- OWV3/Fisher r2 as a score result; it is a CUDA sensitivity/build artifact
  only.
- Omega-W-V2 as Grade A until the exact archive is recovered/rebuilt.
- Lane 12 NeRV as a family kill; only the measured `jsonfix40` implementation
  is retired.
- Lane J-IMP cycle0 as scientific failure evidence; it is a shape-mismatch run
  abort.
- Training proxy logs from live Vast lanes.

## Prioritized Harvest Actions

1. **Keep PFP16 A++ as the only deploy-grade packet.** Use
   `final_deploy_bundle_20260430` for paper/deploy references, not raw logs or
   legacy remote provenance.
2. **Harvest HM-S only after Stage 4 exact eval exists.** It has completed
   training and entered packing, so it is the nearest live lane to completion.
   Because it is KL-active, require component gates and scoped forensic wording
   before any claim.
3. **Continue watching Lane 19, SA, and H-V3, but do not harvest/rank until
   lane-local `contest_auth_eval.json` and archive exist.** Current training
   logs are proxy only.
4. **Recover or rebuild the exact Omega-W-V2 archive SHA named in its JSON.**
   Without that archive, keep it Grade B diagnostic.
5. **Route interesting Modal archives through Lightning exact CUDA.** Start
   with any Modal artifact that still seems scientifically useful after
   archive inspection; the existing Modal CPU JSONs are not enough.
6. **OWV3/Fisher should stay build-only until byte-plausible.** If a candidate
   survives byte gating, run Lightning exact CUDA with archive manifest and
   adjudication.
7. **Do a tracker hygiene pass.** Reconcile or annotate stale
   `.omx/state/vastai_active_instances.json` and `active_dispatches.md` rows
   before the next harvest wave so old instance ids do not drive duplicate
   dispatch or false recovery conclusions.
8. **Harden legacy result handling after the current audit.** Retire log-score
   scraping for promotion scripts, make Modal recovery print advisory labels
   when device is not CUDA, and make recovered Vast artifact selection
   lane-local by default.

