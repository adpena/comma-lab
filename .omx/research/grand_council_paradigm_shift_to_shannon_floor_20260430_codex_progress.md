# Codex Progress Ledger - Grand Council Shannon Floor

Adjacent source of truth:
`grand_council_paradigm_shift_to_shannon_floor_20260430.md`

Date: 2026-04-30

## Scope

This ledger records implementation progress against the Grand Council paradigm
shift plan. It does not replace the source document. It records what has
landed, what remains blocked, and what must happen next to keep the fastest
wall-clock path toward the theoretical floor.

## Landed

1. Beta foundation implementation started:
   - `src/tac/sensitivity_map.py` defines the per-Conv2d-channel sensitivity
     artifact contract, validation, save/load, CUDA-authoritative gate, and
     train/holdout CV-distance helper.
   - `src/tac/owv3_sensitivity_weighted.py` implements the OWV3 mixed-channel
     renderer archive: protected high-sensitivity Conv2d output channels stay
     FP16, lower-sensitivity channels use OWV2.
   - `OWV3` is registered in `src/tac/codec_magic_registry.py`.
   - `submissions/robust_current/inflate_renderer.py` dispatches `OWV3` with
     no scorer import on decode.

2. Alpha Lane 12 compliance blocker partially removed:
   - `experiments/contest_auth_eval.py` accepts `.nrv` archive members.
   - `inflate_renderer.py` resolves `masks.nrv` when callers still pass the
     legacy `masks.mkv` default.

3. Tests landed:
   - Sensitivity-map artifact contract.
   - OWV3 magic registry, mixed-channel encode/decode, protected FP16 channel
     fidelity, callable forward pass, and contest inflate dispatch.
   - `.nrv` archive whitelist and resolver discovery.

4. Memory persisted:
   - `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/project_codex_shannon_floor_orchestration_20260430.md`

## Evidence Status

- PFP16 exact CUDA archive eval has been harvested locally:
  `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z`.
- PFP16 is now Grade A score-grade evidence: `final_score=1.04`,
  `score_recomputed_from_components=1.0440481283330025`,
  `avg_posenet_dist=0.0034602`, `avg_segnet_dist=0.0040083`,
  `archive_size_bytes=686635`, archive SHA-256
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- PFP16 is not Grade A++: eval provenance records
  `gpu_model=NVIDIA GeForce RTX 4090` and `gpu_t4_match=false`.
- The PFP16 `remote_provenance.json` fields `contest_cuda_score=100.0`,
  `hard_kill_triggered=true`, and `lane_status=HARD_KILL_REGRESSION` are
  invalid parser/adjudication outputs. Treat them as superseded by
  `contest_auth_eval.json` until the script fix lands.
- Current verified frontier is Lane G v3 PFP16 `1.0440481283330025`
  score-grade.
- OWV3 is implementation-smoke evidence only.
- Lane 12 NeRV remains empirical-only until full CUDA training, clean archive,
  dependency closure, SHA custody, and exact contest eval exist.

## Next Wall-Clock Actions

1. Build the OWV3 stack archive builder and provenance writer.
2. Convert per-weight Fisher artifacts into per-channel sensitivity maps.
3. Target the remote provenance/adjudication parser bug class that misread the
   PFP16 report as `contest_cuda_score=100.0` and hard-killed a valid
   `contest_auth_eval.json`.
4. Attach Grade A++ evidence for PFP16 on T4/equivalent hardware if it becomes
   the submission candidate.
5. Run Lane 12 full CUDA NeRV with exact `.nrv` archive eval once dependency
   closure is proven.
6. Harvest Lane 17 IMP and run hidden-gem recovery lanes in parallel.
7. Defer full gamma coordinator work until at least one alpha and one beta or
   renderer component has exact archive score evidence.

## Non-Negotiables

- KL distill remains dead.
- Adaptive rebalance remains retired.
- No scorer patching.
- No external sidecar artifacts at inflate.
- No promotion or kill from predictions, smoke tests, CPU/MPS, or byte-only
  reports.

---

## Update - 2026-04-30 Later

Additional implementation landed:

1. **OWV3 archive builder added.**
   - `experiments/build_lane_g_v3_owv3_stack.py`
   - Deterministic zip output.
   - Provenance JSON records anchor hashes, sensitivity map hash/metadata,
     OWV3 thresholds, byte deltas, and score-validation requirement.

2. **Fisher-to-channel converter added.**
   - `experiments/convert_fisher_to_owv3_sensitivity_map.py`
   - Uses per-output-channel Fisher sum as the default aggregation:
     `sensitivity_c = sum_i Fisher[c, i]`.
   - Missing Conv2d layers default to protected sensitivity, not zero.

3. **Fisher profiler correctness fix.**
   - `experiments/profile_hessian_per_weight.py` now decodes grayscale mask
     video pixels back to class IDs before renderer embedding lookup.
   - This fixes the CUDA device-side assert from treating `63/126/189/252` as
     class indices.

4. **Lane 12 canonical archive support improved.**
   - `src/tac/submission_archive.py` now includes `masks_nrv` manifests,
     `masks.nrv` validation, and deterministic archive writing.
   - This removes the prior canonical validation blocker for `masks.nrv`.

5. **Paper rigor blueprint added.**
   - `.omx/research/shannon_floor_paper_rigor_writeup_blueprint_20260430.md`
   - It defines claim grades, formulas, reproducibility packs, ablation tables,
     and adversarial review gates.

Verification:

- Targeted OWV3/Lane12/PFP16/conversion suite: `62 passed`.
- Neighboring archive/integration checks: `17 passed`.
- OWV3 builder smoke with synthetic CPU all-protect sensitivity map succeeded,
  but produced a larger archive. This is intentional smoke evidence only and
  proves the builder path, not codec value.

Current beta blocker:

- No real `hessian_per_weight.pt` exists yet. Rerun the CUDA Fisher profiler
  after this mask-decode fix, then convert to OWV3 sensitivity map and build a
  real candidate archive.

---

## Update - 2026-04-30 PFP16 Harvest

PFP16 moved from prediction/readiness to harvested Grade A score-grade:

- Local evidence directory:
  `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z`.
- Authoritative report: `contest_auth_eval.json`.
- Score facts: `final_score=1.04`,
  `score_recomputed_from_components=1.0440481283330025`,
  `avg_posenet_dist=0.0034602`, `avg_segnet_dist=0.0040083`,
  `archive_size_bytes=686635`.
- Archive SHA-256:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Hardware: NVIDIA GeForce RTX 4090 CUDA, `gpu_t4_match=false`, so this is
  Grade A score-grade only, not A++.
- Current targeted fix: repair the remote lane provenance/adjudication parser
  so the invalid `contest_cuda_score=100.0` and `hard_kill_triggered=true`
  fields are derived from `contest_auth_eval.json` components or suppressed.

---

## Update - 2026-04-30 Auth-Eval Hardening

The PFP16 parser/adjudication bug class is now structurally fixed:

- Added `scripts/adjudicate_contest_auth_eval.py`, a JSON-only adjudicator that
  validates `contest_auth_eval.json`, archive byte count, archive SHA-256,
  `n_samples=600`, and `provenance.device="cuda"`.
- Refactored `scripts/remote_lane_pfp16_stack.sh`,
  `scripts/remote_lane_omega_w_v2_stack.sh`, and
  `scripts/remote_lane_8_multipass.sh` to use JSON adjudication instead of
  regex parsing of `auth_eval.log`.
- Added deterministic PFP16 SHA guard:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Removed remaining remote-lane fallbacks that scraped the last JSON-looking
  object from auth logs.
- Hardened `scripts/launch_lane_with_retry.py` so `phase2-launch` timeout is
  longer than the launcher's 240s poll, and timeout now returns
  `UNKNOWN_REMOTE_STATE` instead of launching duplicate retries.
- Added strict preflight check
  `check_remote_lane_auth_eval_json_adjudication`.
- Added tests in `src/tac/tests/test_remote_auth_eval_hardening.py`.

Verification:

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 6 passed.
- `bash -n` clean for all modified remote lane scripts.
- `py_compile` clean for adjudicator, launcher wrapper, and preflight.
- Static search found zero remaining banned remote auth-eval parser patterns.

---

## Update - 2026-04-30 Swarm + DX Self-Protection

Six-item execution state:

1. Active harvest monitoring completed:
   - `35885106` HM-S and `35899850` Lane 19 remain live/training.
   - No active lane had lane-local `contest_auth_eval.json` at the monitor
     checkpoint, so no new Grade A result was harvested.
2. OWV3/Fisher path prepared but not promoted:
   - `scripts/remote_lane_g_v3_owv3_fisher_stack.sh` and
     `docs/owv3_fisher_runbook.md` exist.
   - Local CUDA is unavailable; no Fisher artifact or OWV3 archive eval has
     landed.
3. PFP16 A++ path prepared but not promoted:
   - `.omx/research/pfp16_a_plus_plus_exact_t4_eval_runbook_20260430.md`
     gives the exact T4/equivalent rerun path for archive SHA
     `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
   - Current evidence remains Grade A score-grade, not A++.
4. Lane 12 `.nrv` dependency closure hardened:
   - Added dependency-closure tests for NeRV codec imports, `.nrv` inflate,
     archive acceptance, auth whitelist, and remote script exact-eval wiring.
   - Verification from worker: 40 passed plus shell syntax and py_compile.
   - Full CUDA `.nrv` archive eval is still outstanding.
5. Hidden-gem wave:
   - `scripts/remote_lane_sa_segmap_clone.sh` and
     `scripts/remote_lane_h_v3_jointly_trained_halfframe.sh` now use JSON-only
     contest-auth adjudication.
   - SegMap clone dispatched cleanly after hardening. Instance `35906669`
     (`lane_sa_segmap_clone_2026-04-30_codex_a2`) passed setup/NVDEC, wrote a
     fresh heartbeat, and entered Stage 2 training at 2026-04-30T14:59Z.
   - H-V3 dispatched through the hardened wrapper. Attempts 1/2 hit slow SSH
     readiness and were retired; attempt 3 failed NVDEC and auto-destroyed;
     attempt 4 `35907873`
     (`lane_h_v3_joint_halfframe_2026-04-30_codex_a4`) launched. At the
     checkpoint it was still in remote setup installing DALI, not yet training.
   - Q-FAITHFUL remains high-risk/gated because its current path contains
     KL-distill-like machinery; it must not be treated as trusted until exact
     CUDA evidence proves no PoseNet collapse.
6. DX/harness self-protection:
   - `scripts/launch_lane_with_retry.py` now has a per-label single-flight
     lock, live Vast label-prefix guard, signal-safe process-group cleanup,
     and `UNKNOWN_EXISTING_LABEL_PREFIX` fail-closed behavior.
   - New strict preflight:
     `check_launch_retry_wrapper_singleflight_and_signal_safe`.
   - `src/tac/tests/test_remote_auth_eval_hardening.py` expanded from 6 to
     9 tests covering duplicate-prefix refusal and timeout cleanup.

Operational incident resolved:

- A previous interrupted SA dispatch had produced duplicate/partial Vast state.
  Empty duplicate `35905846` was destroyed; staged `35905118` failed NVDEC and
  auto-destroyed; the hardened retry then produced one live SA instance
  `35906669`.

Verification:

- `src/tac/tests/test_remote_auth_eval_hardening.py`: 9 passed.
- `py_compile` clean for `scripts/launch_lane_with_retry.py`,
  `src/tac/preflight.py`, and `scripts/adjudicate_contest_auth_eval.py`.
- `check_launch_retry_wrapper_singleflight_and_signal_safe`: 0 violations.
- `check_remote_lane_auth_eval_json_adjudication`: 0 violations.
- `git diff --check`: clean.

---

## Update - 2026-04-30T16:16Z XHigh Swarm Evidence Pass

Compute routing is now explicit:

- Lightning AI is the preferred promotion-grade exact-eval path once SSH/env is
  verified: `ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`.
- Modal credits remain available and should be used for cheap build/smoke,
  Fisher-map generation, and ablations whose outputs are not claimed as
  contest-grade until rerun through exact CUDA archive eval.
- Vast remains active only for currently running lanes and opportunistic cheap
  training; no new high-trust evidence should depend on Vast process state.

Paradigm-alpha status changed materially:

- Lane 12 NeRV was unblocked, trained/evaluated, harvested, and destroyed.
- Exact CUDA archive eval exists at
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json`.
- Result: `score_recomputed_from_components=26.03719330455429`,
  rounded `26.04`, PoseNet `49.77849960`, SegNet `0.03528685`, archive
  `296478` bytes, 600 samples, CUDA on RTX 4090.
- Verdict: hard-kill regression. This is not a promotion result. The current
  NeRV mask replacement destroys PoseNet geometry even though rate is small.
  Future alpha work must preserve pose geometry or pivot away from this form.

Paradigm-beta/KL hardening:

- Primary `loss_mode="kl_distill"` is now an explicit forensic-only path with
  promotion-ineligible guards.
- SegNet-only KL auxiliary use is explicitly scoped as `segnet_aux`; ambiguous
  primary KL configs are rejected.
- The collapse root cause remains a units bug: spatial KL with `batchmean` on
  `[B,C,H,W]` effectively multiplied the intended auxiliary pressure by the
  image area, producing the historic ~254x overweight failure.
- Grade policy: KL-like lanes remain gated unless exact archive CUDA evidence
  proves no PoseNet collapse.

OWV3/Fisher:

- Two hardened Vast attempts for `lane_g_v3_owv3_fisher_20260430_codex_a1`
  failed NVDEC and were auto-destroyed.
- No Fisher artifact, sensitivity map, OWV3 archive, or exact eval landed.
- Next route: run Fisher/build-only on Modal or Lightning; use Lightning for
  any promotion-grade exact eval.

Current live remote training state:

- HM-S `35885106`: running on RTX 4090, heartbeat fresh at 2026-04-30T16:14Z,
  `variant=kl_distill`; high risk and promotion-gated.
- Lane 19 `35899850`: running on RTX 4090, heartbeat fresh at
  2026-04-30T16:16Z, logit-margin profile, no contest eval yet.
- SegMap clone `35906669`: running on RTX 4090, heartbeat fresh at
  2026-04-30T16:14Z, Stage 2 training, no contest eval yet.
- H-V3 `35907873`: running on RTX 4090, heartbeat fresh at 2026-04-30T16:16Z,
  Stage 1 joint half-frame training, no contest eval yet.

No new Grade A/A++ result landed in this pass.

---

## Update - 2026-04-30T16:25Z PFP16 A++ Evidence Landed

PFP16 now has exact T4/equivalent promotion-grade evidence:

- Evidence directory:
  `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/`.
- Eval chain:
  `experiments/contest_auth_eval.py --device cuda -> inflate.sh -> upstream/evaluate.py`.
- Hardware: Lightning AI `Tesla T4`, driver `580.126.09`.
- Contest gate: `gpu_t4_match=true`, `n_samples=600`.
- Exact archive SHA:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Archive bytes: `686635`.
- Score: rounded `1.04`, recomputed `1.043987524793892`.
- Components: PoseNet `0.00346442`, SegNet `0.00400656`, rate
  `0.01828808`.

Impact:

- PFP16 is now the highest-rigor deployable baseline from this wave.
- It is not a Shannon-floor breakthrough by itself, but it establishes the
  exact-eval floor that all stacked candidates must beat under identical
  archive/provenance discipline.
- Lightning is validated as a usable exact-eval home; no separate Lightning
  project is required for this path unless isolation of long-running jobs
  becomes operationally helpful.
