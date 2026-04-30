# Codex Progress Ledger - Shannon Floor Execution Readiness

Adjacent source of truth:
`shannon_floor_execution_readiness_20260430.md`

Date: 2026-04-30

## Scope

This ledger records current execution state, parallel lanes, blockers, and the
next-turn command plan for fastest wall-clock progress.

## Landed

1. OWV3 implementation slice:
   - Sensitivity-map contract added.
   - Mixed-channel OWV3 archive added.
   - `OWV3` magic registered.
   - Contest inflate dispatch added.
   - Synthetic and inflate-path tests passing.

2. Lane 12 packaging slice:
   - `.nrv` accepted by auth archive validator.
   - `masks.nrv` discoverable by inflate resolver even with default
     `masks.mkv` setting.

3. Documentation state:
   - Readiness, contest audit, and external research intake docs updated with
     implementation-readiness status and strict evidence labels.

## Ongoing Work

- OWV3 still needs:
  - stack archive builder,
  - sensitivity artifact generator or converter,
  - train/holdout sensitivity CV report,
  - exact archive eval.
- Lane 12 still needs:
  - full CUDA training,
  - clean contest dependency closure for `tac.nerv_mask_codec`,
  - exact `.nrv` archive,
  - exact contest eval.
- PFP16 exact CUDA eval is harvested. It now needs the remote
  provenance/adjudication parser fix and, if promoted to a submission
  candidate, T4/equivalent A++ evidence.
- Lane 17 IMP needs harvest or exact eval once remote result is ready.
- Lane 19, Lane 8, and hidden-gem recovery lanes need dispatch triage.

## Fastest-Wall-Clock Roadmap

1. Parent agent implements OWV3 stack builder and per-weight-to-per-channel
   sensitivity conversion.
2. Swarm audits exact eval dispatch, Lane 12 clean inflate closure, sensitivity
   artifact availability, and hidden-gem recovery ordering in parallel.
3. Fix the PFP16 remote provenance/adjudication parser bug class and preserve
   the harvested `contest_auth_eval.json` as the authoritative score source.
4. Dispatch OWV3 exact eval after builder plus sensitivity artifact exist.
5. Dispatch Lane 12 full CUDA `.nrv` eval after dependency closure is proven.
6. Run IMP and hidden-gem lanes in parallel where GPU slots are available.
7. Run gamma MDL/ADMM/static entropy only after measured component streams
   exist.

## Next Turn Checklist

- Spawn focused workers for:
  - PFP16 parser/adjudication fix and A++ readiness,
  - Lane 12 clean contest dependency closure,
  - sensitivity artifact inventory/conversion,
  - hidden-gem dispatch order.
- Implement:
  - `experiments/build_lane_g_v3_owv3_stack.py`,
  - per-weight Fisher to per-channel sensitivity-map conversion utility.
- Verify:
  - targeted pytest,
  - deterministic zip manifest,
  - no scorer import in OWV3 decode,
  - `.nrv` resolver avoids SegNet fallback.

---

## Update - 2026-04-30 Later

Swarm results incorporated:

1. **PFP16 harvest:** exact CUDA eval has landed and is Grade A score-grade.
   A++ remains blocked by RTX 4090 hardware (`gpu_t4_match=false`) and missing
   contest-budget/T4 evidence.
2. **Hidden-gem ordering:** harvest active Lane 8, Lane 19, Lane 17 first; next
   parallel dispatch wave should be SegMap clone, H-V3 half-frame, and
   Q-FAITHFUL. FL waits for RAFT chunking. MAE-V runs if spare 4090/A10G exists.
3. **Sensitivity inventory:** no usable `hessian_per_weight.pt` exists. Prior
   run crashed because mask grayscale values were used as class IDs. That bug
   is now fixed.
4. **Reproducibility audit:** central archive builder had nondeterministic set
   ordering and source mtimes. That is now patched for deterministic writes.
5. **Paper rigor:** writeup blueprint now exists and must be used for all
   claims and ablations.

Implementation landed after swarm:

- `experiments/build_lane_g_v3_owv3_stack.py`
- `experiments/convert_fisher_to_owv3_sensitivity_map.py`
- `src/tac/tests/test_owv3_sensitivity_conversion.py`
- `src/tac/tests/test_profile_hessian_mask_decode.py`
- `src/tac/submission_archive.py` `masks.nrv` manifest support and deterministic
  archive writes.

Verification:

```text
62 passed:
  profile_hessian_mask_decode
  lane12_nerv_dependency_closure
  owv3_sensitivity_conversion
  sensitivity_map
  owv3_sensitivity_weighted
  runtime_guards_pass_3
  contest_auth_eval
  pfp16_codec

17 passed:
  integration_boundaries
  engineered_corrections
```

Immediate fastest-wall-clock commands:

1. Fix the PFP16 remote provenance/adjudication parser class:
   - `contest_auth_eval.json` is authoritative.
   - `remote_provenance.json` currently says `contest_cuda_score=100.0`,
     `hard_kill_triggered=true`, and `lane_status=HARD_KILL_REGRESSION`; those
     fields are invalid and superseded until the script fix lands.
   - The harvested local evidence directory is
     `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z`.

2. Rerun CUDA Fisher after mask decode fix:

```bash
PYTHONPATH=src:upstream .venv/bin/python experiments/profile_hessian_per_weight.py \
  --checkpoint experiments/results/lane_a_landed/iter_0/renderer.bin \
  --video upstream/videos/0.mkv \
  --masks-mkv experiments/results/lane_a_landed/iter_0/masks.mkv \
  --poses experiments/results/lane_a_landed/iter_0/optimized_poses.pt \
  --upstream upstream \
  --pair-weights experiments/results/lane_lane_w_modal/harvested_artifacts/lane_w_results/pair_weights.pt \
  --top-k 30 \
  --device cuda \
  --pair-batch 4 \
  --output experiments/results/owv3_sensitivity/hessian_per_weight.pt
```

3. Convert Fisher and build OWV3:

```bash
PYTHONPATH=src .venv/bin/python experiments/convert_fisher_to_owv3_sensitivity_map.py \
  --checkpoint experiments/results/lane_g_v3_landed/iter_0/renderer.bin \
  --fisher experiments/results/owv3_sensitivity/hessian_per_weight.pt \
  --output experiments/results/owv3_sensitivity/sensitivity_map.pt \
  --metadata-json experiments/results/owv3_sensitivity/sensitivity_map.meta.json

PYTHONPATH=src .venv/bin/python experiments/build_lane_g_v3_owv3_stack.py \
  --sensitivity-map experiments/results/owv3_sensitivity/sensitivity_map.pt \
  --output experiments/results/lane_g_v3_owv3/archive_lane_g_v3_owv3.zip \
  --provenance-json experiments/results/lane_g_v3_owv3/provenance.json
```

4. Dispatch hidden-gem wave as capacity allows:

```bash
scripts/remote_lane_sa_segmap_clone.sh
scripts/remote_lane_h_v3_jointly_trained_halfframe.sh
scripts/remote_lane_q_faithful_jointgen.sh
```

Q-FAITHFUL remains high-risk because its current script uses KL-distill-like
machinery; it must be labeled prediction/high-risk until exact CUDA proves no
PoseNet collapse.

---

## Update - 2026-04-30 PFP16 Harvest

PFP16 evidence now in hand:

- Directory:
  `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z`.
- `contest_auth_eval.json`: `final_score=1.04`,
  `score_recomputed_from_components=1.0440481283330025`,
  `avg_posenet_dist=0.0034602`, `avg_segnet_dist=0.0040083`,
  `archive_size_bytes=686635`, `n_samples=600`.
- Archive SHA-256:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Hardware: NVIDIA GeForce RTX 4090 CUDA, `gpu_t4_match=false`.

Readiness impact:

- PFP16 is Grade A score-grade and becomes the current verified frontier.
- It is not A++ until the same archive SHA has T4/equivalent hardware,
  contest-budget inflate, and full 1:1 provenance evidence.
- Current targeted bug-class fix is remote score parser/adjudication: invalid
  `contest_cuda_score=100.0` and `hard_kill_triggered=true` fields must be
  recomputed from, or superseded by, `contest_auth_eval.json`.

---

## Update - 2026-04-30 Remote Harness Hardening

Remote score parser/adjudication is now fixed and guarded:

- New canonical helper: `scripts/adjudicate_contest_auth_eval.py`.
- PFP16, Ω-W-V2, and Lane 8 stack scripts now adjudicate from
  `eval_work/contest_auth_eval.json`, not `auth_eval.log` text.
- PFP16 remote script now asserts the exact deterministic archive SHA before
  spending GPU time.
- Adjacent NWC/GP/FL/WC/sweep scripts now require `contest_auth_eval.json` and
  refuse last-object log scraping.
- `scripts/launch_lane_with_retry.py` now uses timeouts compatible with the
  underlying launcher poll windows and returns `UNKNOWN_REMOTE_STATE` on
  phase2-launch timeout to prevent duplicate dispatches.
- Strict preflight check:
  `check_remote_lane_auth_eval_json_adjudication`.
- Test coverage:
  `src/tac/tests/test_remote_auth_eval_hardening.py`.

Verification:

- `6 passed` for remote auth-eval hardening tests.
- `bash -n` clean across all modified lane scripts.
- `py_compile` clean across adjudicator, retry launcher, and preflight.

Readiness impact:

- PFP16 Grade A score-grade evidence remains valid and is now protected from
  false hard-kill provenance in future dispatches.
- Next wall-clock priority returns to CUDA Fisher sensitivity generation,
  OWV3 candidate build/eval, Lane 12 `.nrv` CUDA eval, and hidden-gem harvest.

---

## Update - 2026-04-30 Dispatch Readiness / DX Hardening

Dispatch orchestration is now treated as a correctness surface:

- `scripts/launch_lane_with_retry.py` now fails closed on same-label live Vast
  state through `live_instances_with_label_prefix`.
- A per-label advisory lock under `.omx/state/launch_locks/` prevents two local
  retry wrappers from launching the same logical lane simultaneously.
- Child launcher phases run in their own process group and are terminated on
  timeout/SIGINT/SIGTERM, preventing orphaned phase2 children.
- `phase2-launch` timeout remains `UNKNOWN_REMOTE_STATE`, never blind retry.
- Strict preflight now includes
  `check_launch_retry_wrapper_singleflight_and_signal_safe`.

Remote wave status:

- SegMap clone is live on Vast as instance `35906669`,
  label `lane_sa_segmap_clone_2026-04-30_codex_a2`, RTX 4090, `$0.2539/hr`,
  SSH `root@ssh2.vast.ai:26668`.
- Remote proof at launch:
  - `/workspace/setup.log` contains `SETUP_COMPLETE`.
  - `/workspace/pact/lane_sa_segmap_clone_results/heartbeat.log` exists.
  - `run.log` reached Stage 2 training after NVDEC probe and anchor checks.
- Existing active lanes at the same checkpoint:
  - `35885106` HM-S.
  - `35899850` Lane 19 logit margin.
- H-V3 dispatch added after the SA checkpoint:
  - Live instance `35907873`,
    label `lane_h_v3_joint_halfframe_2026-04-30_codex_a4`, RTX 4090,
    `$0.2731/hr`, SSH `root@ssh5.vast.ai:27872`.
  - The host passed lightweight NVDEC pre-probe.
  - At 2026-04-30T15:21Z it was still in setup Stage 3 installing
    `nvidia-dali-cuda120`; lane training had not started yet.

Still blocked / not promoted:

- PFP16 A++: exact T4/equivalent rerun required.
- OWV3: CUDA Fisher artifact, sensitivity conversion, archive build, and exact
  CUDA eval required.
- Lane 12: full CUDA `.nrv` archive eval required.
- H-V3: dispatched/setup-running only; exact archive CUDA eval still required.
- Q-FAITHFUL: gated because of KL-distill-like risk.

Verification:

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 9 passed.
- `py_compile`: launcher wrapper, preflight, adjudicator.
- Preflight scanners:
  - `launch_self_protect_violations 0`
  - `remote_auth_json_violations 0`
- `git diff --check`: clean.

---

## Update - 2026-04-30T16:16Z Routing And Readiness Delta

Promotion-grade compute routing:

- Lightning AI endpoint recorded for exact CUDA reruns:
  `ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`.
- A xhigh worker is validating Lightning access/hardware and will run
  `scripts/pfp16_a_plus_plus_exact_t4_eval.sh --run` only if the machine
  satisfies the T4/equivalent A++ gate.
- Modal credits remain in play for non-promotion smoke, ablation, build-only
  codec work, and Fisher/sensitivity generation.
- Vast should be used conservatively because state tracking and NVDEC
  reliability remain weak; current live lanes are monitored but not trusted
  until they emit lane-local `contest_auth_eval.json`.

Readiness changes:

- Lane 12 NeRV exact CUDA eval completed and hard-killed:
  `26.03719330455429` recomputed, PoseNet `49.77849960`, SegNet
  `0.03528685`, archive `296478` bytes.
- The NeRV lane is no longer launch-blocked, but the current technique is not
  frontier-useful without a geometry-preserving target.
- PFP16 remains the cleanest near-term promotion candidate: Grade A
  score-grade evidence exists, A++ is blocked only on exact T4/equivalent run.
- OWV3 remains blocked on a real Fisher artifact and archive eval; Vast NVDEC
  failures make Lightning/Modal the next correct route.
- H-V3 has progressed from setup to Stage 1 training and remains worth
  monitoring because it attacks a real train/inflate distribution mismatch.

Immediate execution priority:

1. Use Lightning for PFP16 A++ exact rerun as soon as hardware is confirmed.
2. Use Modal/Lightning to produce Fisher/sensitivity artifacts for OWV3.
3. Continue harvesting HM-S, Lane 19, SA, and H-V3 only through exact
   `contest_auth_eval.json`.
4. Treat Lane 12 NeRV as failed evidence and redesign alpha around
   pose-preserving masks before more spend.

---

## Update - 2026-04-30T16:25Z Lightning Exact-Eval Home Validated

Lightning AI successfully produced a promotion-grade PFP16 exact eval:

- SSH endpoint used:
  `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`.
- Remote repo for this run was staged separately at
  `/home/zeus/content/pact_pfp16_exact_20260430T1625Z` because the default
  `/home/zeus/content/pact` tree was stale/non-git and missing required eval
  files.
- Remote upstream tree: `/home/zeus/content/upstream`, git `c5e1274`.
- Hardware: Tesla T4; `gpu_t4_match=true`.
- Evidence:
  `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`.

Readiness implication:

- Use Lightning as the exact-eval home for PFP16, OWV3 candidates, and any
  stack that must be promoted.
- Use separate staged Lightning project/worktree directories for hermetic runs
  instead of mutating a possibly stale default workspace.
- Modal remains the fastest supplementary path for Fisher/build-only smoke and
  ablations, but current Modal wrappers are advisory because they force CPU eval.
