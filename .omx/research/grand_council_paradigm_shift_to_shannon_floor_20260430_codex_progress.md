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
  invalid legacy parser/adjudication outputs. Treat them as superseded by
  `contest_auth_eval.json`; the adjudicator now emits scoped
  `regression_triggered` / `REGRESSION_REVIEW_REQUIRED` language for future
  runs.
- Current verified frontier is Lane G v3 PFP16 `1.0440481283330025`
  score-grade.
- OWV3 is implementation-smoke evidence only.
- Lane 12 NeRV remains empirical-only until full CUDA training, clean archive,
  dependency closure, SHA custody, and exact contest eval exist.

## Next Wall-Clock Actions

1. Build the OWV3 stack archive builder and provenance writer.
2. Convert per-weight Fisher artifacts into per-channel sensitivity maps.
3. Target the remote provenance/adjudication parser bug class that misread the
   PFP16 report as `contest_cuda_score=100.0` and misclassified a valid
   `contest_auth_eval.json`.
4. Attach Grade A++ evidence for PFP16 on T4/equivalent hardware if it becomes
   the submission candidate.
5. Run Lane 12 full CUDA NeRV with exact `.nrv` archive eval once dependency
   closure is proven.
6. Harvest Lane 17 IMP and run hidden-gem recovery lanes in parallel.
7. Defer full gamma coordinator work until at least one alpha and one beta or
   renderer component has exact archive score evidence.

## Non-Negotiables

- Legacy primary KL-distill remains promotion-ineligible; scoped auxiliary KL
  remains forensic-gated.
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
- Verdict: exact-CUDA regression retires the current measured implementation
  and config only. This is not a promotion result and not a broad NeRV/alpha
  kill. The current NeRV mask replacement destroys PoseNet geometry even
  though rate is small. Future alpha work must preserve pose geometry or pivot
  away from this form.

Paradigm-beta/KL hardening:

- Primary `loss_mode="kl_distill"` is now an explicit forensic-only path with
  promotion-ineligible guards.
- SegNet-only KL auxiliary use is explicitly scoped as `segnet_aux`; ambiguous
  primary KL configs are rejected.
- A confirmed KL scale bug is one documented contributor/risk factor:
  spatial KL with `batchmean` on `[B,C,H,W]` effectively multiplied the
  intended auxiliary pressure by the image area, producing the historic
  ~254x overweight failure. Do not present it as the sole collapse root cause
  without matched post-fix exact CUDA ablations.
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

---

## Update - 2026-04-30T16:45Z Grand Council Reconciliation

The non-progress Grand Council source doc was patched to prevent outdated
"necessary and sufficient" language from being read as a proved architecture.
After Lane 12 exact collapse and OWV3/Fisher size-regression smoke, α/β/γ are
still prioritized directions, but sufficiency requires exact standalone and
stacked archive evidence.

Current controlling facts:

- PFP16 A++ is the deploy baseline:
  `1.043987524793892`, SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  bytes `686635`, Lightning AI Tesla T4, `gpu_t4_match=true`.
- Lane 12 NeRV `jsonfix40` is retired by exact-CUDA regression for that
  implementation/config only:
  score `26.03719330455429`, PoseNet `49.77849960`, archive `296478`.
- OWV3/Fisher Modal smoke produced artifacts but a larger archive:
  `912971` bytes, `+218897` vs Lane G v3, SHA
  `710cba0c7c490b13db8b0aee897dd0f33cb8b66a6ed229466bf0d1aea392f5a3`.
  Treat as suspicious negative smoke and fix encoder overhead/config before
  another promotion run.

The Dykstra ceiling is now explicit in the paper blueprint: sub-`0.30` requires
`archive_bytes <= 450545` even with zero distortion. PFP16 is contest-grade and
deployable, but not a Shannon-floor architecture by itself.

---

## Update - 2026-04-30T17:35Z Six-Item Implementation Resumption

After the urgent Lightning PyPI supply-chain audit, execution resumed on the
six Grand Council wall-clock items.

Landed this pass:

- Security/DX hardening:
  - `lightning_sdk==2026.4.10` audited against the Mini Shai-Hulud indicators.
    No payload evidence found.
  - `src/tac/preflight.py` now blocks PyPI `lightning`, bad pins, unsafe
    `lightning --version` probes, planted repo paths, hidden `_runtime`, and
    known IOC hashes.
  - `src/tac/deploy/cloud_deploy.py` no longer executes `lightning --version`;
    it checks `lightning-sdk` metadata instead.
  - `src/tac/deploy/lightning/batch_jobs.py` disables the SDK version check
    before SDK import.
  - Security review ledger:
    `.omx/research/lightning_pypi_compromise_security_review_20260430_codex.md`.
- PFP16 bundle/provenance:
  - Final deploy bundle exists at
    `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/`.
  - The bundle co-locates archive, SHA, manifest, exact T4 eval JSON, logs,
    inflate timing, build provenance, source diff, upstream hash, council
    reviews, and runbook.
- Claim discipline:
  - `.omx/research/shannon_floor_claim_matrix_20260430_codex.md` is the
    current machine-readable claim nucleus.
- OWV3/Fisher redesign:
  - New spec:
    `.omx/research/owv3_fisher_byte_aware_redesign_spec_20260430_codex.md`.
  - It makes ASYM-preserving fallback and post-ZIP byte accounting mandatory
    before any exact eval.
- Alpha/Lane 12 redesign:
  - New spec:
    `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`.
  - It orders Alpha-Geo-0 stale-pose isolation before larger retraining.

Swarm state:

- Xhigh workers/explorers are running for Lightning Batch Jobs hardening,
  active Vast harvest, OWV3/Fisher implementation audit, Alpha/Lane 12 rescue,
  and PFP16/paper bundle audit.
- The sixth KL/DX audit slot was unavailable at spawn time; Codex is covering
  that path locally after the first five return.

Current controlling gates:

- PFP16 A++ is the deploy baseline until a stacked archive beats it under the
  same exact-eval discipline.
- Live lanes remain unscored until lane-local canonical
  `contest_auth_eval.json` appears.
- OWV3/Fisher cannot spend promotion eval budget until byte-plausible archive
  accounting exists.
- Alpha cannot spend large retraining budget until stale-pose and decoded-mask
  target confounds are isolated.

---

## Update - 2026-04-30T17:43Z Swarm Return And Guard Tightening

Xhigh swarm returns have been folded into the control plane:

- Active Vast harvest audit:
  - Four live lanes remain in progress: HM-S (`35885106`), Lane 19
    (`35899850`), SA (`35906669`), H-V3 (`35907873`).
  - No lane-local `contest_auth_eval.json`, adjudication JSON, lane-local ZIP,
    or auth-eval log exists yet. They remain watch-only.
- PFP16/paper audit:
  - Score evidence remains strong A++:
    `1.043987524793892`, `686635` bytes, SHA
    `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
  - Paper/deploy bundle blockers remain: missing remote staged-tree manifest,
    stale legacy `hard_kill_triggered` provenance fields, broad dirty source
    snapshot, and stale public docs.
- OWV3/Fisher audit:
  - Current implementation is smoke-only because protected/fallback layers are
    encoded as FP16 instead of ASYM-preserving bytes. Modal smoke increased
    archive size and cannot promote.
  - The byte-aware ASYM-preserving redesign spec is now the implementation
    gate.
- Alpha/Lane 12 audit:
  - Lane 12 `jsonfix40` failure is scoped exact negative evidence for that
    measured mask replacement only. `renderer.bin` and `optimized_poses.bin`
    were byte-identical to the base archive; only `masks.nrv` changed.
  - Alpha must proceed through geometry diagnostics and pose-rescue gates
    before larger retraining.
- Lightning Batch Jobs worker:
  - Added expected archive SHA/byte validation, queue metadata, command hash,
    JSON-preserving exact CUDA command, local artifact validation/mirroring,
    harvest attachment, status refresh from SDK attributes, and CLI
    `validate-artifacts` / `harvest-local` modes.
- KL hardening:
  - New ledger:
    `.omx/research/kl_distill_hardening_status_20260430_codex.md`.
  - `src/tac/segmap_renderer.py` now rejects `loss_mode="kl_distill"` unless
    `kl_distill_scope=="segnet_aux"`, because SegMapTrainer implements only
    standard scorer loss plus SegNet-only KL auxiliary.
  - Primary scorer KL remains forensic-only. SegNet-aux KL remains an
    experimental auxiliary under exact PoseNet/component gates.

Claim matrix updates:

- C-011: KL scope and lane-interpretation policy.
- C-012: OWV3/Fisher byte-preserving promotion block.
- C-013: Alpha/NeRV remains open via geometry-preserving redesign.

Open gates before deploy/paper promotion:

- Verify the Lightning Batch Jobs and KL hardening tests locally.
- Wait for KL xhigh council sidecar and fold additional findings.
- Regenerate stale public writeup files around PFP16 A++ only after source
  custody/provenance contradictions are quarantined.

---

## Update - 2026-04-30T17:50Z KL Council Closure And OWV3 Fail-Closed Gate

The xhigh KL council sidecar returned. Confirmed and fixed issues:

- `SegMapTrainer` primary/scope confusion:
  - Fixed by rejecting `loss_mode="kl_distill"` unless
    `kl_distill_scope=="segnet_aux"`.
  - Regression test added.
- KL SNR-controller-only no-op in `optimize_poses.py`:
  - Fixed by defining `kl_distill_active` as static weight positive OR SNR
    controller present.
  - GT frame pairs now materialize for controller-only KL, per-step logging
    keys off effective controller weight, and Lane PS warnings no longer
    falsely call controller-active runs no-ops.
- KL provenance:
  - Generic Trainer checkpoint metadata now records `kl_distill_scope`,
    `kl_distill_weight`, `kl_distill_temperature`,
    `allow_banned_primary_kl_distill`, and `promotion_eligible`.
- KL preflight coverage:
  - Roundtripped-KL scanner now includes `src/tac/segmap_renderer.py`.

OWV3/Fisher fail-closed gate:

- `experiments/convert_fisher_to_owv3_sensitivity_map.py` now defaults to
  `missing_policy="error"` instead of `protect`.
- `scripts/remote_lane_g_v3_owv3_fisher_stack.sh` now passes
  `--missing-policy error`.
- `protect` remains available for smoke/debug only and is not promotion-safe.

Verification:

- `py_compile` passed for touched Python files.
- `bash -n scripts/remote_lane_g_v3_owv3_fisher_stack.sh` passed.
- Focused pytest suite passed: `291 passed in 23.89s`.
- `git diff --check` passed for touched repo files.
- Lightning Batch Jobs dry run against PFP16 archive identity succeeded and
  recorded command hash
  `895eae34fc47a2d3211511f9bea4a3cdbab97a66876cf6dc8f9055d426c8630d`.

Remaining KL council work:

- Add high-weight KL profile/script waiver gates before promoting any
  `kl_distill_weight >= 1.0` lane.
- Patch stale public docs/comments that use broad "KL dead" or old
  `kl_distill` naming without scope.
- Add exact CUDA component gates in adjudication for KL-active lanes.

---

## Update - 2026-04-30T18:08Z Recursive Review Greenup Wave

Grand Council implementation status against the six immediate actions:

- **PFP16:** custody and paper packet repaired without touching archive bytes.
  `contest_auth_eval.json` is now the only active score authority in the final
  deploy bundle; legacy `hard_kill_triggered` / `HARD_KILL_REGRESSION` parser
  fields are quarantined as invalid superseded output.
- **Sensitivity foundation:** OWV3 no longer silently expands protected layers
  to FP16 in promotion mode. The byte plan now explicitly distinguishes
  `keep_asym` from `diagnostic_fp16`, and promotion archives are gated against
  the PFP16 A++ byte frontier.
- **OWV3 design:** build it, but only through the new gate: CUDA sensitivity
  map, ASYM-preserving fallback, post-ZIP byte viability, deterministic ZIP
  rebuild proof, member manifest, and exact CUDA eval after byte plausibility.
- **Lane 12 / Alpha:** do not spawn blind retraining. Alpha-Geo-0 diagnostics
  now exist to localize NeRV geometry defects before pose rescue or retraining.
- **Retraining gate:** still active. No new scorer-sensitive retraining should
  promote without component gates and exact archive custody.
- **J-NWC corpus codec:** delegated and ongoing. Required result is
  deterministic corpus manifests and fail-closed sensitivity handling; dummy
  random sensitivity is forbidden in promotable scripts.

Additional bug-class hardening:

- Exact adjudicator now supports absolute and relative PoseNet/SegNet component
  gates, closing the KL-style component-collapse loophole where total score
  alone could look acceptable.
- Alpha diagnostic ZIP member loading now reads controlled member bytes instead
  of extracting arbitrary paths.
- MCP helper processes were killed again after respawn; project-level MCP
  configs remain empty/disabled.

Verification:

- Focused cross-lane suite: `77 passed in 3.56s`.
- Alpha diagnostic suite after zip-slip hardening: `8 passed in 1.02s`.
- `py_compile`, `bash -n`, PFP16 custody `jq`, archive SHA/byte checks, and
  `git diff --check` all passed for the integrated slices.

Current adversarial stance:

- PFP16 A++ is the deploy baseline and paper score authority.
- OWV3 is implementation-green for byte-planned builds but not scored.
- Alpha/NeRV remains open; Lane 12 is scoped A-negative evidence only.
- Live Vast lanes remain unscored.
- J-NWC remains implementation-pending until corpus manifest and fake-signal
  gates return and pass tests.

---

## Update - 2026-04-30T18:42Z Lightning Repro Contract And CUDA Fisher R2

Additional completed work:

- **MCP/DX:** killed the live `chrome-devtools-mcp` and `rbx-studio-mcp`
  helper processes again and disabled the remaining OpenCode MCP config by
  setting `mcpServers` to `{}`. Codex and Claude MCP configs are already empty
  for active servers. If helpers respawn, that is coming from the outer app
  runtime, not project config.
- **Lightning reproducibility:** added and verified
  `scripts/lightning_repro_workspace.py` as the OSS/reproducible replacement
  for ad hoc Lightning `rsync`.
  - Fixed the source manifest contract so generated archives, `.pt`, `.mkv`,
    `.raw`, `.zip`, and large side outputs are excluded unless explicitly
    passed with `--artifact`.
  - Added fallback environment recording for `--no-install` trees where
    `.venv/bin/python` does not yet exist.
  - Real Lightning sync verified:
    `.omx/state/owv3_repro_contract_20260430_r1_manifest.json`,
    `1074` files, `18645277` bytes, remote SHA verification OK.
  - Environment record intentionally used system Python and reported no torch
    because this pass was `--no-install`; locked runtime install is the next
    exact-eval setup step.
- **OWV3 CUDA Fisher r2:** generated and harvested the first strict CUDA/T4
  Fisher-to-sensitivity artifact with protected Conv2d coverage.
  - Output directory:
    `experiments/results/lane_g_v3_owv3_fisher_lightning_20260430_codex_r2/`.
  - Sensitivity map SHA:
    `ed69bec3c9c530e4d574d82d3b6764399a6feca0289f2114899fa09689fabeba`.
  - `19` Conv2d layers, `717` channels, no missing protected/nonprotected
    Conv2d keys.
  - Candidate archive:
    `archive_lane_g_v3_owv3.zip`, `689342` bytes, SHA
    `29a02b2af2c37371eec80ca3e278c4ce368703ba0a0a2121e2b32f570106a84c`.
  - Promotion blocked before exact eval because archive is `+2707` bytes versus
    PFP16 A++. No OWV3 r2 score exists.
- **Alpha-Geo-0:** ran Lane 12 `jsonfix40` versus Lane G v3 masks.
  - Global disagreement `0.012303928799099393`.
  - Temporal transition F1 `0.095099661402374`.
  - Lane-marking class F1 `0.3197888`.
  - Component centroid jump p95 `155.69px`.
  - Interpretation: mask global Hamming is not the main issue; component and
    temporal geometry damage is severe and plausibly PoseNet-dominant.
- **J-NWC:** deterministic corpus manifest / no-fake-sensitivity patch passed
  focused tests: `23 passed in 1.77s`. Independent xhigh adversarial audit is
  still pending before dispatch promotion.
- **Sensitivity roadmap:** Russell audit completed. The missing gate is now
  specified as `component_sensitivity_v1`: CUDA-only PoseNet/SegNet/combined
  maps, calibration/holdout stability, response curves, exact custody linkage,
  and fail-closed promotion validation.

Verification for this update:

- `py_compile scripts/lightning_repro_workspace.py` passed.
- `pytest src/tac/tests/test_lightning_repro_workspace.py -q` passed:
  `4 passed in 0.07s`.
- Lightning real manifest sync and remote SHA verification passed.

Current gates:

- OWV3 exact eval remains blocked until the archive beats PFP16 bytes or earns
  a reviewed exact distortion-justification override.
- Lightning main tree now has manifest custody but still needs locked runtime
  install before it is an exact-eval home.
- Component sensitivity implementation should be the next beta foundation slice
  before any scorer-sensitive allocation is promoted.

---

## Update - 2026-04-30T19:08Z J-NWC/NWCS Fail-Closed Greenup

Heisenberg's xhigh J-NWC/NWCS audit found eight blockers. Integrated fixes:

- **Deterministic training seed:** `experiments/train_neural_weight_codec.py`
  now seeds torch before `WeightCodec` construction, so random codebook
  initialization is inside the advertised seed contract. NWCS inline remote
  snippets seed before `SensitivityAwareWeightCodec` construction.
- **Relocatable corpus replay:** `build_corpus_from_manifest(...,
  replay_root=...)` can replay a manifest after staging to a new workspace,
  while still checking size, SHA-256, shape, dtype, block count, and block
  ordering.
- **Zip-slip-safe anchor extraction:** J-NWC, J-NWCS, and J-NWCS-EC remote
  scripts no longer use `extractall`; they reject absolute paths, traversal,
  hidden sidecars, unexpected members, duplicates, and `__MACOSX`.
- **CUDA-only promotion gate:** the same three remote scripts now fail closed
  unless `AUTH_EVAL_DEVICE=cuda`; no CPU/MPS override can emit a misleading
  `[contest-CUDA]` result.
- **Sensitivity provenance gate:** promotable NWCS anchor/corpus sensitivities
  must include metadata for anchor archive SHA-256, anchor renderer SHA-256,
  corpus manifest SHA-256, block size, parameter names, shapes, block counts,
  and nonnegative finite values. Raw shape-only dicts are debug-only.
- **Custody metadata:** provenance now records SHA-256 and byte count for
  anchor archive, extracted payloads, corpus manifest, codec checkpoints,
  sensitivity artifacts, candidate archive, and result JSON.
- **NWCS renderer format:** added `NWCS1` magic container support with
  deterministic JSON header, embedded codec checkpoint bytes, tensor metadata,
  signed length-prefixed blobs, strict parser validation, renderer export
  dispatch, and inflate-side loader dispatch.
- **Component sensitivity validator:** added `component_sensitivity_v1`
  validator for CUDA-only PoseNet/SegNet/combined maps with calibration,
  holdout, response curves, exact custody linkage, and fail-closed promotion
  markers.

Verification:

- `bash -n` passed for all three J-NWC/NWCS remote scripts.
- `py_compile` passed for touched Python modules and tests.
- Focused J-NWC/NWCS/component/Lightning suite passed: `64 passed in 2.88s`.
- Targeted NWCS container/static hardening tests passed: `14 passed in 0.44s`.
- Lightning source/artifact sync refreshed:
  `.omx/state/shannon_greenup_20260430_jnwcs_r1_manifest.json`, `1078` files,
  `18724610` bytes, remote SHA verification OK. Environment record is still
  `--no-install` system Python with no torch, so exact eval requires runtime
  install or explicit CUDA `PYBIN`.

Current status:

- J-NWC/NWCS are no longer blocked by the known container/determinism/
  zip-custody/CUDA-downgrade bug classes.
- They still have no score claim. Next valid promotion step is a reviewed CUDA
  exact archive run with validated sensitivity provenance and full artifact
  custody.

---

## Update - 2026-04-30T19:14Z Byte-Feasible OWV3, Lightning Security, And Swarm Continuation

Additional implementation and evidence landed:

- **OWV3 byte sweep:** added a deterministic byte-plan sweep over the CUDA/T4
  Fisher sensitivity map.
  - Script: `experiments/sweep_owv3_byte_plan.py`.
  - Tests: `src/tac/tests/test_sweep_owv3_byte_plan.py`.
  - Sweep output:
    `experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/`.
  - Best byte-feasible candidate:
    `best_byte_feasible/archive_lane_g_v3_owv3.zip`, `686557` bytes, SHA-256
    `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`.
  - Delta versus PFP16 A++ frontier `686635` bytes: `-78` bytes.
  - Knobs: `bit_budget_ratio=0.69`, `protect_threshold=0.0014`,
    `aggressive_threshold=1e-5`, `fallback_action=keep_asym`.
  - Evidence grade: byte-only/empirical. No score claim exists until exact CUDA
    auth eval on this exact archive bytes.

- **Lightning reproducibility fix:** hardened
  `scripts/lightning_repro_workspace.py` after the real OWV3 staging attempt
  exposed a generated remote-Python `null` literal bug for
  `python_bin_requested=None`.
  - Fixed generated source to emit valid Python `None`.
  - Added regression assertions in
    `src/tac/tests/test_lightning_repro_workspace.py`.
  - OWV3 byte-feasible candidate staged successfully to Lightning:
    `.omx/state/owv3_byte_feasible_repro_20260430_r1_manifest.json`,
    `1081` files, `17674947` bytes, manifest SHA-256
    `5fde235b76d19c991d489ce603aa640b391bb46b235ef866d0b7095230c0790e`.

- **Lightning Batch Jobs dry-run:** generated a dry-run queue entry for the
  OWV3 byte-feasible archive.
  - Run id: `owv3_byte_feasible_exact_cuda_20260430_codex_dryrun`.
  - Command hash:
    `e8551610ddb813ae6d0ee4857c3f110a22affa201ce64333624709bbeee15e89`.
  - Expected archive SHA/bytes match the byte-feasible candidate above.
  - Baseline comparator: PFP16 T4 score `1.043987524793892`, bytes `686635`,
    PoseNet `0.00346442`, SegNet `0.00400656`.
  - This remains a dry-run because the current Lightning SSH shell has no
    `nvidia-smi` and no CUDA-visible torch.

- **Lightning supply-chain scan utility:** added
  `scripts/scan_lightning_supply_chain.py` and
  `src/tac/tests/test_lightning_supply_chain_scan.py`.
  - Local strict JSON scan output:
    `.omx/state/lightning_supply_chain_scan_20260430_codex.json`.
  - Status: OK, zero violations. Local `.venv` has `lightning-sdk==2026.4.10`
    and no PyPI `lightning` / `pytorch-lightning` install.
  - Remote read-only SSH IOC scan found no `lightning-2.6.2`,
    `lightning-2.6.3`, hidden `lightning/_runtime`, or known planted repo
    files in the checked Studio roots.

- **Alpha-Geo-0 completion:** ran Lane 12 `jsonfix40` against both Lane G v3
  and Lane A/base mask targets.
  - Outputs:
    `alpha_geo_0_vs_lane_g_v3.json` and `alpha_geo_0_vs_lane_a_base.json`.
  - Both targets produce the same diagnostic numbers: global disagreement
    `0.012303928799099393`, temporal transition disagreement
    `0.009507171571470149`, transition F1 `0.095099661402374`.
  - Interpretation remains diagnostic only: global mask distance is modest, but
    temporal/component geometry is poor and must guide Alpha-Geo-1.

- **MCP/DX:** post-kill process sweep was clean for live MCP helpers; the only
  match was the `rg` command performing the check.

Verification:

- Focused cross-slice pytest suite passed: `126 passed in 3.64s`.
- `py_compile` passed for the touched Lightning, supply-chain, OWV3 sweep, and
  component-sensitivity files/tests.
- `bash -n` passed for J-NWC/NWCS/OWV3/active remote lane scripts.

New xhigh swarm wave started:

- `Averroes`: KL distill hardening grand-council review.
- `Feynman`: arXiv `2604.26919v1` and related Shannon-floor research intake.
- `Volta`: PufferLib/RL/local-model tooling and DeepSeek visual-primitives
  applicability.
- `Meitner`: PoseNet/SegNet perturbation and profiling tooling audit.

Immediate next gates:

1. Do not exact-eval OWV3 until a CUDA worker is visible; when visible, submit
   only the byte-feasible archive with expected SHA/byte and component gates.
2. Treat the OWV3 byte candidate as a rate probe, not a score claim, until
   CUDA exact eval lands.
3. Produce real `component_sensitivity_v1` maps and response curves on CUDA
   before promoting sensitivity-aware lanes beyond engineering readiness.
4. Keep harvesting Vast/Modal/Lightning only from canonical
   JSON/archive/provenance artifacts.

---

## Update - 2026-04-30T19:22Z Sensitivity Producer, KL Fence, And Audit Integration

New hardening landed:

- **Component sensitivity manifest assembler:** added
  `experiments/build_component_sensitivity_manifest.py` and
  `src/tac/tests/test_build_component_sensitivity_manifest.py`.
  - Builds deterministic `component_sensitivity_v1` manifests from explicit
    PoseNet/SegNet/combined maps, response curves, stability JSON, exact CUDA
    `contest_auth_eval.json`, archive, checkpoint, video, and upstream tree.
  - Materializes SHA-256/byte custody through the validator rather than
    trusting hand-written JSON.
  - Generates deterministic calibration/holdout pair splits when no sample
    plan is supplied and records a `split_hash`.
  - Fails closed on non-CUDA contest eval, wrong sample count, missing tensor
    payloads, and response curves without holdout error.

- **FP4 sensitivity profiler bug class fixed:** hardened
  `experiments/profile_fp4_layer_sensitivity.py`.
  - Decodes grayscale mask luma values through the same class-id remap used by
    the Fisher profiler, preventing raw `63/126/189/252` class IDs from
    corrupting renderer inputs.
  - CPU now requires explicit `--allow-diagnostic-cpu` and records
    `promotion_eligible=False` / `evidence_grade=diagnostic_cpu`.
  - CUDA profiler metadata records `mask_decode` and evidence class.

- **KL Grand Council gap closed:** `loss_mode="segnet_kl"` is now fenced in
  `src/tac/training.py`.
  - It must set `kl_distill_scope="segnet_aux"`.
  - It must set `promotion_eligible=False`.
  - `SEGNET_KL_SMOKE` and `SEGNET_KL_FULL` in `src/tac/profiles.py` are now
    explicitly forensic/non-promotable.
  - Regression tests landed in `src/tac/tests/test_kl_distill_weight_plumbed.py`.

- **Swarm research/audit reports completed and integrated into the queue:**
  - `external_research_arxiv_2604_26919_shannon_floor_20260430_agent.md`:
    no direct codec path; useful patterns are adaptive warm-ramp, sparse top-k
    allocation, and dual-readout validation for sensitivity/Alpha diagnostics.
  - `pufferlib_rl_visual_primitives_shannon_floor_20260430_agent.md`:
    use bandit/BO and local-model triage first; defer direct PPO/PufferLib
    exact-eval control until a cheap correlated surrogate exists; visual
    primitives map well to Alpha geometry preservation.
  - `posenet_segnet_perturbation_tooling_audit_20260430_agent.md`:
    confirms the missing production bridge was deterministic component
    sensitivity assembly plus perturbation/response-curve custody; flags legacy
    FP4 mask decode and CPU diagnostic risks now fixed.
  - `kl_distill_hardening_grand_council_review_20260430_agent.md`:
    confirms primary KL forensic-only policy and identified the now-closed
    `segnet_kl` promotion-fence gap.

Current Lightning status:

- Local supply-chain scan rerun:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_r2.json`, status OK,
  zero violations.
- Lightning SSH still has no visible CUDA in the current shell:
  `nvidia-smi` absent and system Python reports no torch CUDA. Exact eval
  remains intentionally held.

Verification:

- Expanded focused suite passed: `172 passed in 4.78s`.
- `py_compile` passed for touched Python scripts/modules/tests.
- `bash -n` passed for touched/critical remote lane scripts.
- `git diff --check` passed for touched repo files.

Immediate next gates:

1. Use the new assembler as the required bridge when CUDA runs emit
   PoseNet/SegNet/combined maps and response curves.
2. Implement or dispatch the actual CUDA map/response-curve producer next; the
   assembler is custody plumbing, not the scorer perturbation computation.
3. Recheck Lightning GPU after workspace/runtime changes; submit OWV3
   byte-feasible exact eval only when CUDA is visible and supply-chain scan is
   clean.
4. Fold visual-primitives diagnostics into Alpha-Geo-1 before retraining or
   exact-eval spend on Alpha.

---

## Update - 2026-04-30T19:34Z Component Producer Landed And Next Swarm Started

New implementation landed:

- **CUDA component sensitivity producer:** added
  `experiments/profile_component_sensitivity.py` and
  `src/tac/tests/test_profile_component_sensitivity.py`.
  - Produces PoseNet, SegNet, and combined per-channel sensitivity maps using
    component-separated empirical Fisher proxies.
  - Emits response curves using deterministic signed RMS perturbations of top-k
    sensitive Conv2d channels on the holdout split.
  - Emits `sample_plan.json`, `stability.json`, component map `.pt` files,
    response-curve JSON files, and a profile summary.
  - Includes optional handoff to `component_sensitivity_v1` manifest assembly
    when exact archive and CUDA `contest_auth_eval.json` are supplied.
  - Uses CUDA by default. CPU requires `--allow-diagnostic-cpu` and remains
    explicitly non-promotable.

Swarm artifacts integrated:

- `component_sensitivity_producer_grand_council_review_20260430_agent.md`
  confirmed the algorithmic requirements and caveats for a promotion-grade
  producer: empirical Fisher is a proxy, SegNet argmax needs finite-difference
  response gates, custody must link to exact eval and archive bytes, and the
  manifest shell alone is insufficient.
- `alpha_geo_1_visual_primitives_design_20260430_agent.md` defined the
  `alpha_geo_visual_primitives_v1` diagnostic packet with boxes, centroids,
  lane/boundary polylines, temporal tracks, pose-sensitive primitives, and
  rejection-only CPU gates before retraining/exact eval spend.

Runtime checks:

- Lightning SSH still has no visible CUDA:
  `nvidia-smi` absent; system Python has no torch.
- Local Lightning supply-chain scan:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_r3.json`, status OK,
  zero violations.
- MCP helpers respawned from the outer runtime and were killed again by exact
  process pattern.

Verification:

- Expanded focused suite passed: `203 passed in 5.14s`.
- `py_compile` passed for touched producer, manifest, security, training, and
  test files.
- `bash -n` passed for critical remote scripts.
- `git diff --check` passed for touched repo files.

Ongoing xhigh agents:

- `Kierkegaard`: exact-eval queue ops and CUDA runtime blocker report.
- `Herschel`: NWCS1 build-only smoke and sensitivity provenance plan.
- `Hubble`: live harvest/status audit.

Immediate next gates:

1. Run `experiments/profile_component_sensitivity.py` on a CUDA runner to
   create real component maps and response curves.
2. Assemble the produced artifacts with
   `experiments/build_component_sensitivity_manifest.py` after exact CUDA eval
   JSON exists for the candidate archive.
3. Bring up CUDA-visible Lightning/Vast/Modal and submit the OWV3 byte-feasible
   exact eval with SHA/byte and component gates.
4. Implement Alpha-Geo-1 diagnostics against Lane 12 and future Alpha
   candidates before retraining spend.

---

## Update - 2026-04-30T19:40Z Swarm Closed, NWCS Blocker Fixed, Lightning Dry-Run Corrected

Swarm outputs closed and integrated:

- `exact_eval_queue_ops_20260430_agent.md`: OWV3 byte-feasible archive is
  present locally and on Lightning with expected SHA/bytes, but no exact CUDA
  score exists. Fastest safe queue is Lightning Batch Jobs on T4.
- `live_harvest_status_20260430_agent.md`: live Vast lanes are not
  score-harvestable; HM-S is diagnostic-stalled, Lane 19/SA/H-V3 have proxy
  logs only, Modal harvests require exact CUDA reruns, SZ needs exact rerun,
  and Omega-W-V2 lacks exact archive custody.
- `nwcs1_build_smoke_and_sensitivity_plan_20260430_agent.md`: NWCS1
  build-only smoke succeeded, but real promotion was blocked by a Stage 5
  `_infer_asymmetric_config` import gap in both NWCS remote scripts.

New fixes landed:

- Patched `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh` and
  `scripts/remote_lane_j_nwcs_ec_stack.sh` so the Stage 5 export heredoc
  imports `_infer_asymmetric_config`; the prior fallback to `{"tensor_only":
  True}` can no longer hide missing architecture metadata in that path.
- Strengthened `src/tac/tests/test_remote_lane_j_nwc_hardening.py` so it checks
  the exact export heredoc, not merely any earlier import in the file.
- Hardened `experiments/profile_component_sensitivity.py`: when top-k
  `--pair-weights` selects a subset, `sample_plan.json` now records absolute
  dataset pair IDs, not relative subset offsets. Added regression coverage in
  `src/tac/tests/test_profile_component_sensitivity.py`.

Lightning exact-eval queue state:

- Created corrected dry-run
  `owv3_byte_feasible_exact_cuda_20260430_codex_studio_pact_dryrun`.
- Recorded `studio: "pact"`, expected archive SHA
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
  expected bytes `686557`, command SHA
  `45456318dccbd437e02c4446f7339ad66aaa4e79668c0e69d4707b39c506358f`,
  and local artifact target
  `experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_studio_pact`.
- This remains a dry-run only. No score, ranking, promotion, or kill claim is
  permitted until real T4 CUDA artifacts are harvested and adjudicated.

Security/DX state:

- Fresh strict local supply-chain scan:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_r4.json`, status OK,
  zero violations, installed `lightning-sdk==2026.4.10`, no PyPI `lightning`
  or `pytorch-lightning`.
- MCP process sweep matched only the checking `rg`; no live MCP helper process
  was present.

Verification:

- Sensitivity/NWCS/Lightning batch suite: `60 passed in 1.45s`.
- Preflight/supply-chain/repro/config suite: `250 passed in 23.53s`.
- `py_compile` passed for touched Python files.
- `bash -n` passed for J-NWC/J-NWCS/NWCS-EC scripts.
- `git diff --check` passed for touched code/state files.

Immediate next gates:

1. Submit the corrected Lightning Batch Jobs exact eval only from a
   CUDA-visible T4 runtime with a fresh strict supply-chain scan.
2. Run the component sensitivity producer on CUDA against the exact candidate
   archive/eval pair, then assemble `component_sensitivity_v1`.
3. Apply the NWCS Stage 5 import fix to any copied/staged remote scripts before
   promotable NWCS dispatch.
4. Keep live Vast/Modal harvest restricted to canonical archive plus
   lane-local `contest_auth_eval.json` plus custody/adjudication artifacts.

---

## Update - 2026-04-30T19:55Z Real Lightning Queue, Fail-Closed Sensitivity, And MCP Cleanup

Swarm outputs integrated this loop:

- `lightning_exact_eval_ops_readiness_20260430_codex.md`: Lightning ops path
  was superseded by a real Batch Jobs submission after runner preflight
  hardening.
- `component_sensitivity_owv3_nwcs_execution_plan_20260430_codex.md`:
  current producer is structural/diagnostic Fisher proxy only; it cannot be
  promotion evidence until official component response validation lands.
- `j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md`:
  corpus replay and NWCS debug/fake-sensitivity paths now fail closed.
- `live_harvest_canonical_artifact_triage_20260430_codex.md`: no live Vast or
  Modal lane is newly score-harvestable.
- `alpha_geo_diagnostics_lane12_readiness_20260430_codex.md`: Alpha-Geo-0 2px
  boundary diagnostics fail exploratory geometry gates for Lane 12 `jsonfix40`.
- `shannon_floor_paper_claim_hygiene_20260430_codex.md`: GP v3 and UNIWARD v8
  Modal/local rows are quarantined from score/writeup claims.

New fixes landed:

- Hardened `src/tac/deploy/lightning/batch_jobs.py`: exact CUDA eval command
  now runs `scripts/scan_lightning_supply_chain.py --quiet --strict` inside
  the runner, writes `lightning_supply_chain_scan.json`, performs a CUDA/T4
  runner preflight, writes `lightning_runner_preflight.json`, and prints
  `LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK` before `contest_auth_eval.py`.
- `LightningBatchJobSpec.validate()` now rejects exact-eval commands that lack
  either the supply-chain scan or the runner CUDA preflight marker.
- Added Lightning CLI ops:
  `scripts/launch_lightning_batch_job.py refresh-status` for SDK status
  refresh without log parsing, and `list-machines` for provider machine
  discovery.
- Submitted real OWV3 byte-feasible exact eval:
  `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x` on
  `g4dn.2xlarge` in teamspace `comma-lab`, Studio
  `lossy-compression-challenge`.
  - SDK job:
    `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x`.
  - Link:
    `https://lightning.ai/adpena/comma-lab/studios/lossy-compression-challenge/app?app_id=jobs&job_name=owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x`.
  - Artifact path:
    `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x/artifacts`.
  - Expected archive SHA:
    `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`;
    expected bytes `686557`; command SHA
    `8333f867324ff9a1fef521d28418b7faad9a3097cf722ce0e581d1d1678ed0e6`.
  - Last SDK refresh at `2026-04-30T19:55:00Z`: `Running`, total cost
    reported `0.03048889`.
- Hardened `experiments/profile_component_sensitivity.py` to mark all current
  outputs `promotion_eligible=false` with diagnostic Fisher-proxy blockers and
  to reject `--manifest-output` assembly from those outputs. Negative epsilons
  are accepted for future symmetric/directional response-curve probes.
- Updated `AGENTS.md` so current component-sensitivity profiler output is
  explicitly diagnostic/proxy-only even on CUDA until official response
  validation and exact custody are implemented.
- Added Lightning `list-machines --help` CLI coverage.
- Killed live Roblox/Chrome DevTools MCP helper processes. Active Codex config
  had no `[mcp_servers]`; removed the stale MCP backup and Cloudflare plugin
  `.mcp.json` cache so config search now has zero MCP server definitions.

Verification:

- Focused sensitivity/NWCS/Lightning/Alpha/preflight suite:
  `312 passed in 26.28s`.
- `py_compile` passed for touched Python files.
- `bash -n` passed for J-NWC/J-NWCS/NWCS-EC scripts.
- `jq empty` passed for Lightning queue state and Alpha-Geo-0 artifact JSON.
- `git diff --check` passed for touched code/state/report files.
- MCP process sweep after cleanup matched only the checking `rg`.

Score/claim status:

- No new score claim exists. OWV3 byte-feasible exact eval is active but
  unharvested. Promotion remains blocked until `archive.zip`,
  `contest_auth_eval.json`, `contest_auth_eval.adjudicated.json`,
  `adjudication_provenance.json`, runner preflight, supply-chain scan, logs,
  and custody metadata validate.
- Current component-sensitivity outputs are diagnostic only; do not use them
  for OWV3/NWCS promotion or paper claims.

Immediate next wall-clock gates:

1. Poll the Lightning job with `refresh-status`; when complete, harvest and
   validate exact artifacts from the recorded artifact path with expected
   SHA/bytes and `--require-adjudication`.
2. If adjudication passes, update the claim matrix and paper evidence; if it
   fails, treat the result as an engineering/config/math investigation first,
   not a family kill.
3. Build the official component-response producer: finite-difference PoseNet
   and SegNet response validation, symmetric/directional curves, calibration
   and holdout stability thresholds, exact CUDA custody, then
   `component_sensitivity_v1` assembly.
4. Rerun NWCS build-only smoke under the new fake-sensitivity/build-only
   guards, then exact CUDA eval only with validated sensitivity provenance.

### 2026-04-30T20:00Z Lightning Queue Wrapper Repair

- The first real Lightning job
  `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x` failed
  before eval. SDK logs show `mkdir: cannot create directory
  '/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x':
  Permission denied`.
- Root cause: Lightning SDK materializes job artifact paths with hyphenated
  job names, while our default command output directory used the local
  underscore job name. This is a queue-wrapper bug only; no archive was
  evaluated and no lane evidence exists.
- Fixed in `src/tac/deploy/lightning/batch_jobs.py` with
  `lightning_sdk_job_name()` and default output directory
  `/teamspace/jobs/<sdk-hyphen-name>/artifacts`.
- Updated `scripts/launch_lightning_batch_job.py refresh-status` to use the
  same helper and added regression coverage:
  `test_exact_eval_default_output_dir_matches_sdk_job_artifact_path`.
- Resubmitted as
  `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r2`.
  - SDK job:
    `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2`.
  - Artifact path:
    `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2/artifacts`.
  - Expected archive SHA/bytes unchanged:
    `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
    `686557`.
  - New command SHA:
    `cf1ab6a1c81b0fa69273007ec4c3efcd54f25bdce1f8f3856bb2bb3bddc21e08`.
  - Last refresh at `2026-04-30T20:00:46Z`: `Pending`.
- Verification for the repair:
  `src/tac/tests/test_lightning_batch_jobs.py` `19 passed in 0.18s`;
  `py_compile` passed for Lightning Batch Jobs files; `git diff --check`
  passed for the repaired code/state slice.
- Full focused rerun after the repair:
  `313 passed in 24.19s`.
- R2 SDK status refresh at `2026-04-30T20:04:33Z`: `Running`.

### 2026-04-30T20:15Z Swarm Round And Lightning Writable Output Repair

Subagent returns integrated:

- `Lovelace`: no lane beyond PFP16 A++ and active OWV3 Lightning watch is
  newly promotable. Omega-W-V2 remains custody-blocked; Modal rows are CPU or
  missing exact custody; Lane 12 NeRV remains scoped negative evidence only.
- `Bacon`: Alpha-Geo-1 pre-retraining patch landed. `train_nerv_mask.py` now
  supports `--gt-masks-source decoded-baseline` with ZIP-safe baseline
  `masks.mkv` custody and class/scorer-geometry checks. `remote_lane_nerv.sh`
  exposes `GT_MASKS_SOURCE=decoded-baseline`,
  `DECODED_BASELINE_PATH=<archive.zip>`, and
  `DECODED_BASELINE_MEMBER=masks.mkv`. Verification: `9 passed in 0.82s`;
  no CUDA eval or retraining.
- `Locke`: diagnostic sensitivity sources are now rejected during direct
  `component_sensitivity_v1` manifest assembly, and strict preflight has a
  repo-owned MCP server config check. Verification: `12 passed`; no shell
  scripts touched.
- `Avicenna`: component sensitivity promotion validation now requires official
  component response metadata, `passed=true`, finite gate specs, official
  readouts, symmetric/directional coverage, and stability thresholds.
  Verification: `48 passed in 1.50s`; no CUDA eval.
- `Boole`: NWCS build-only/provenance hardening landed. Corpus manifests
  exclude hidden/macOS sidecars, NWCS block sensitivities reject NaN/Inf and
  negative values, promotion export fails closed without real architecture
  config, and exact-path provenance boolean expansion is fixed. CPU-only
  build smoke produced archive SHA
  `9339fed08deffb25b73803b2e311ec34a93508256e5aff993758d23ec0e9c6fd`,
  `3895` bytes, `auth_eval_skipped=true`, `promotion_eligible=false`,
  `score_claim=false`. Verification: `36 passed in 1.48s`; no auth eval.

Lightning r2 outcome:

- R2 failed before exact eval with
  `OSError: [Errno 30] Read-only file system` writing
  `lightning_queue_metadata.json` under
  `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2/artifacts`.
- Root cause: the SDK artifact path is a read-only collection view inside the
  running Studio Batch Job. This is infrastructure-only; no archive eval,
  score, promotion, or kill evidence exists.

R3 repair:

- `src/tac/deploy/lightning/batch_jobs.py` now distinguishes:
  - `lightning_sdk_artifact_path(name)`: SDK-reported read-only artifact view.
  - `default_exact_eval_output_dir(repo_dir, job_name)`: writable Studio
    workspace path under
    `<repo>/experiments/results/lightning_batch/<job_name>`.
  - `_validate_writable_output_dir()`: rejects `/teamspace/jobs/...` as an
    exact-eval output target.
- `LightningBatchJobSpec` records `remote_output_dir`; queue records also keep
  `sdk_artifact_path` for harvest lookup.
- Submitted r3:
  `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3`.
  - SDK job:
    `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r3`.
  - Remote output dir:
    `/teamspace/studios/this_studio/pact/experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3`.
  - SDK artifact path:
    `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r3/artifacts`.
  - Expected archive SHA/bytes unchanged:
    `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
    `686557`.
  - Command SHA:
    `23bba87e21b56791278cd7b6c8686a8718004f67fbabf72fd07d2a90bc813467`.
  - Last refresh at `2026-04-30T20:18:59Z`: `Running`; no artifacts yet.

Verification after worker patches and Lightning writable-output repair:

- Component/Lightning/MCP focused suite: `72 passed in 1.64s`.
- `py_compile` passed for Lightning, component, and preflight touched files.
- `git diff --check` passed for touched slices.

Score/claim status remains unchanged:

- PFP16 A++ is still the only claim-capable score anchor.
- OWV3 byte-feasible remains active exact-eval watch only. No score claim
  exists until CUDA JSON, adjudication, runner preflight, supply-chain scan,
  archive, logs, and custody artifacts validate.
- MCP helper processes respawned from an external parent after cleanup; killed
  exact `rbx-studio-mcp` and `chrome-devtools-mcp` helpers again. Active
  config search still shows no MCP server definitions; post-kill process sweep
  matched only the checking `rg`.

### 2026-04-30T21:05Z Preflight Metabug Hardening and Lightning r4

R3 outcome was infrastructure-only: it reached the CUDA/archive/inflate setup
path, then upstream `evaluate.py` failed because `nvidia.dali` was absent. No
`contest_auth_eval.json` landed, so R3 creates no score, promotion, regression,
or kill evidence.

Permanent hardening landed in the exact-eval runner and preflight layer:

- Lightning exact CUDA eval now requires expected archive SHA-256 and byte
  count before job construction; omitted identity is a hard error.
- Lightning exact CUDA eval now requires adjudication provenance. The CLI
  rejects `exact-eval` without `--adjudicate`.
- The runner deletes stale output artifacts before execution, records separate
  pre/post Lightning supply-chain scans, writes hash-pinned DALI requirements,
  validates DALI bootstrap contents, validates CUDA/DALI runner preflight
  contents, and validates adjudicated JSON equality/custody on harvest.
- The PyPI Lightning compromise guard now rejects any installed bare
  `lightning` distribution, not just known-bad `2.6.2/2.6.3`.
- Remote lane contest-CUDA eval calls were made literal `--device cuda`;
  preflight now catches reintroduction of unguarded `AUTH_EVAL_DEVICE` under
  `[contest-CUDA]` and requires `--keep-work-dir` plus `--work-dir` custody.
- All live MCP helper processes were killed again; the final sweep matched only
  the checking `rg`.

Verification:

- Focused guardrail suite:
  `270 passed in 24.80s`.
- Broader lane/Lightning/component/NWCS/Alpha focused suite:
  `374 passed in 27.14s`.
- `py_compile` passed for touched Python files.
- `git diff --check` passed.
- Strict Lightning supply-chain scan:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_preflight_metabugs.json`,
  `status=OK`, `violation_count=0`, `lightning=null`,
  `lightning-sdk=2026.4.10`.

R4 exact-eval submission:

- Local job/spec:
  `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4`.
- SDK job:
  `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r4`.
- Link:
  `https://lightning.ai/adpena/comma-lab/studios/lossy-compression-challenge/app?app_id=jobs&job_name=owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r4`.
- Expected archive SHA/bytes:
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`,
  `686557`.
- Command SHA:
  `79bf98e80faa762456f5f0d35845a0326bee79d8285e89b871ded8a7d837ca60`.
- Last refresh `2026-04-30T21:01:31Z`: `Pending`.

Claim discipline: still no OWV3 score claim. R4 must produce and pass
validation for `contest_auth_eval.json`, `contest_auth_eval.adjudicated.json`,
`adjudication_provenance.json`, `archive.zip`, `lightning_dali_bootstrap.json`,
`lightning_dali_requirements.txt`, `lightning_runner_preflight.json`,
`lightning_supply_chain_scan_pre.json`, and `lightning_supply_chain_scan.json`
before the result can enter the claim matrix.

### 2026-04-30T21:26Z Live Telemetry Cleanup, R4 Exact Score Packet, Duplicate Guard

Provider telemetry:

- Vast.ai live inventory is now empty after cleanup:
  `.omx/state/vastai_show_instances_live_final_20260430.json` = `[]`.
- Destroyed current-run/duplicate Vast instances after harvest/snapshot:
  `35885106` HM-S, `35906669` SA, `35907873` H-V3, `35899850` Lane 19,
  duplicate/orphan Lane 19 `35925274`, `35925374`, and `35925801`, duplicate
  Lane 20 `35925475` and `35925825`, and self-test escape `35925916`.
- Modal still has `Tasks=0` across apps; no Modal work is running.
- MCP helper processes were killed again; post-kill sweep matched only the
  checking `rg`.

Scientific classifications:

- No broad lane-family kill is justified from the cleaned Vast telemetry.
- HM-S/SA are SegMap packaging-contract failures with no archive/eval JSON.
- H-V3 is a tensor-channel engineering failure in the uncertainty-loss path.
- Lane 19 is a cost/proxy abort only; no exact archive/eval evidence exists.
- Lane 20 on the Lane G v3 anchor remains a no-op/static-fallback result; do
  not rerun on that anchor unless byte precheck first beats static.

Lightning r4:

- `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4`
  reached exact CUDA eval on Tesla T4 and produced a complete non-adjudicated
  score packet.
- Harvested focused artifacts to
  `experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4/`
  plus `SHA256SUMS`.
- Result:
  `score_recomputed_from_components=1.0378905176070103`,
  `final_score=1.04`, `avg_posenet_dist=0.00319052`,
  `avg_segnet_dist=0.00402120`, `archive_size_bytes=686557`, SHA
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`.
- Adjudication failed, not because of score, but because the SegNet relative
  component gate was exceeded:
  `0.00402120 / 0.00400656 = 1.003654` against cap `1.002`.
- Claim status: NOT promotable yet. Treat as high-value diagnostic exact CUDA
  evidence. Next move is Grand Council review on the component-gate policy,
  score/component tradeoff, and sensitivity-aware mitigation plan.

DX hardening landed:

- `scripts/launch_lane_with_retry.py` now uses `logical_lane_key()` for
  advisory locks and live Vast duplicate detection. Timestamped queue labels
  like `_q1_20260430T...`, `_q1c_20260430T...`, and non-numeric queue tags now
  block each other unless `--allow-existing-label-prefix` is explicitly used.
- `.omx/state/dispatch_holds.json` now holds Lane 19 and Lane 20 fail-closed;
  the launcher exits with `FATAL_DISPATCH_HOLD` before creating a Vast
  instance unless `--override-dispatch-hold` is explicitly used after recorded
  Grand Council clearance.
- `src/tac/preflight.py` Check 100 now requires the logical lane duplicate key.
- Regression tests added to
  `src/tac/tests/test_remote_auth_eval_hardening.py`.
- Verification: `py_compile` passed for touched Python files;
  `22 passed in 1.46s` for the focused hardening suite.

Immediate next order:

1. Grand Council adjudicate OWV3 r4: component-gate policy, whether the SegNet
   +0.3654% relative regression is acceptable under contest/writeup standards,
   and what sensitivity/byte-plan mitigation can recover the component margin.
2. Patch H-V3 tensor-channel bug before any half-frame rerun.
3. Fix SegMap pack/roundtrip contract before HM-S/SA reruns.
4. Do not relaunch Lane 19 until deterministic archive/adjudicator/current
   frontier gates are fixed.
5. Keep Vast empty unless the lane has a preflighted exact evidence path and a
   non-duplicate logical key.

### 2026-04-30T21:49Z Swarm Integration Delta

Landed implementation and protocol changes:

- H-V3 channel-shape bug is fixed in `segnet_uncertainty_weighted_loss`; the
  SegNet path now preserves RGB BCHW and has a regression test.
- HM-S/SA SegMap pack contract is repaired as explicit lossy block-FP:
  `segmap_block_fp_per_channel_lossy_v1`, `segmap_pack_roundtrip.json`, and
  exact CUDA archive-eval gate before any score claim.
- Lane 19/20 forensic holds are now conditional fail-closed launch gates.
  Cleared or missing hold JSON cannot relaunch a lane while script-level
  clearance markers are unmet.
- Lightning supply-chain scan is clean locally and the Lightning batch CLI now
  disables `lightning_sdk` import-time PyPI version checks before SDK import.
- Renderer KL/JBL auxiliaries now require explicit
  `kl_distill_scope="segnet_aux"`; primary/full-scorer KL is blocked in
  `train_renderer`, and current positive-KL profiles declare scope.
- MCP helper processes were killed and discovered user/project MCP config
  server maps are empty.
- Paper blueprint trailing whitespace warning was fixed.

OWV3 r4 Grand Council verdict:

- Keep the predeclared SegNet component gate as a hard blocker.
- r4 exact CUDA/T4 packet remains valuable evidence:
  `score_recomputed_from_components=1.0378905176070103`,
  PoseNet `0.00319052`, SegNet `0.00402120`, bytes `686557`, archive SHA
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`.
- It is not promotable because `0.00402120 / 0.00400656 = 1.003654`, above
  the `1.002` SegNet cap.
- Next admissible path: paired same-run PFP16 calibration on the r4 runner and
  SegNet-conservative OWV3 R5 candidates. No retroactive relaxation.

Verification:

- `107 passed in 2.20s` for Lightning batch/supply-chain, Lane 19/20 holds,
  remote auth hardening, SegMap pack, H-V3 loss, and renderer KL scope tests.
- `py_compile` passed for touched Python files.
- `bash -n` passed for touched remote shell scripts.
- Strict preflights passed for renderer KL explicit scope, HM-S/SA lossy pack,
  and disabled MCP config.

New adjacent delta ledger:

- `.omx/research/shannon_floor_swarm_execution_delta_20260430_codex.md`

### 2026-04-30T22:05Z Six-Item Follow-Up Integration

Swarm findings and patches:

- Lane PFP16 required no new implementation. It is already the current A++
  frontier with exact T4 CUDA score evidence, archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  `686635` bytes, and recomputed score `1.043987524793892`.
- Lane 12 NeRV/Alpha is now fail-closed for exact eval:
  `scripts/remote_lane_nerv.sh` defaults to decoded-baseline target custody and
  build-only output. Exact eval requires pose-regeneration provenance, and the
  launcher blocks unrelated retraining dispatches until an explicit Lane 12 L2
  clearance packet exists.
- Sensitivity/OWV3 R5 readiness advanced: diagnostic profiler outputs now
  include deterministic perturbation-basis custody and response-prediction
  calibration diagnostics. The profiler remains non-promotable until CUDA
  finite-difference component response and `component_sensitivity_v1` custody
  exist.
- OWV3 R5 byte-sweep support now ranks SegNet-conservative neighbors around
  r4. The first byte-only R5 candidate reported by the worker is
  `owv3_0047_bbr0p67_protect0p00135_aggr1em05`, `686468` bytes, `-167`
  versus PFP16, with fewer low-bit channels than r4. It has no score claim.
- J-NWC now has an explicit build-only non-promotable path; J-NWC/NWCS/NWCS-EC
  exact paths now run JSON adjudication with PFP16 A++ score/component gates.
- Claim matrix and the Grand Council source doc were corrected: OWV3 r4 is
  exact diagnostic/non-promotable due SegNet gate failure, and the paradigm
  shifts are necessary but not yet proven sufficient.
- Live MCP process absence is now enforced in preflight, not just by config
  cleanup.

Verification:

- Integrated focused suite: `69 passed in 2.66s`.
- Worker suites: PFP16 `34 passed`; Sensitivity/OWV3 `83 passed`; Lane 12
  `11 passed`; J-NWC/NWCS `29 passed`.
- Shell syntax and Python compile checks passed for touched scripts/modules.
- Vast and Modal telemetry are currently empty/idle by `uv run --no-sync`
  provider commands.

Open blockers:

- No promotable `component_sensitivity_v1` exists.
- OWV3 R5 still needs exact CUDA/T4 eval and component-gate adjudication.
- Lane 12 exact eval remains blocked by geometry diagnostics and pose
  regeneration provenance.
- PFP16 A++ source-custody bundle still carries the known non-git Lightning
  staged-tree caveat, but the archive/score authority is intact.

### 2026-04-30T22:24Z Grand Council Closeout Delta

Contest-grade status after the six-worker swarm:

- No new score is promoted. PFP16 remains the only A++ promotion-grade anchor.
- OWV3 R5 has a deterministic rank-1 queue candidate:
  `owv3_0047_bbr0p67_protect0p00135_aggr1em05`,
  `686468` bytes,
  `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`.
  It is not a result until exact CUDA/T4 adjudication passes against paired
  PFP16 calibration.
- OWV3 r4 exact CUDA/T4 evidence remains diagnostic only. The adjudicator now
  writes forensic artifacts even when predeclared component gates fire, and the
  validator marks those artifacts `promotion_eligible=false`.
- Lightning Batch Jobs are operational through SDK discovery and dry-run queue
  generation, but SSH staging is not operational yet (`Permission denied
  (publickey)`). Do not spend a job on an archive path unless remote custody is
  proven.
- Lane 12 NeRV/Alpha has direct geometry evidence against the current
  jsonfix40 output and failed Alpha-Geo-0 gates. This supports engineering
  redesign, not exact eval.
- J-NWC/NWCS lanes now have stronger provenance and adjudication custody, but
  remain blocked from exact eval until real component-sensitivity artifacts
  exist.

Rigor implications:

- Bad or disappointing lane outputs remain suspected implementation/config
  evidence until exact CUDA custody, component gates, and adversarial review
  close. No family-level KILL is justified by these artifacts.
- The Grand Council docs remain the source of truth for claims; this progress
  file records operational deltas and explicitly distinguishes queued,
  diagnostic, blocked, and promotion-grade evidence.
- The paper/writeup should label OWV3 r4 as "exact diagnostic,
  non-promotable due SegNet gate" and label OWV3 R5 as "queued candidate, no
  score claim."

Verification added in this closeout:

- `137 passed in 3.27s` focused regression slice.
- `py_compile`, shell syntax, targeted `git diff --check`, and strict MCP /
  remote-auth / launcher preflights passed.

### 2026-04-30T22:30Z Lightning Exact-Eval Queue Delta

- Lightning SSH is now operational after rerunning the setup script.
- Reproducible staging is operational after hardening the remote `uv sync`
  path to use copy-mode installs on Lightning filesystems.
- Submitted two T4 Batch Jobs:
  `pfp16_paired_calibration_20260430_codex_lightning_t4_r2` and
  `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4`.
- Both were `Pending` at the first SDK refresh. They are active queue work, not
  results.
- R5 remains non-promotable until the paired PFP16 calibration result is
  harvested and used for re-adjudication.

### 2026-04-30T22:48Z Harness Failure Classification

- The first Lightning PFP16/R5 exact-eval attempts are classified as harness
  failures only. They do not measure PFP16 or R5.
- Root cause: inflate-side `uv run` recreated the shared scorer `.venv`, then
  `upstream/evaluate.py` failed due missing `tqdm`.
- Permanent exact-eval isolation and bootstrap locking landed before clean
  reruns were submitted.

### 2026-04-30T22:53Z Grand Council Claim-State Closeout

- Clean isolated Lightning jobs remain `Pending`; there is no new score, no new
  rank claim, and no R5 promotion.
- PFP16 A++ remains the only promotion-grade anchor.
- OWV3 R5 rank-1 remains a queued candidate with a predeclared paired
  calibration requirement. It must be readjudicated against the PFP16
  calibration artifact before any score-band, component, or stack claim is
  admissible.
- The harness bug class is fixed at command-generation level, with regression
  coverage and a green `177 passed in 5.04s` focused suite.
- Scientific posture: queued, failed-harness, diagnostic, blocked, and
  promotion-grade evidence remain separated in the writeup. No disappointing
  lane result is treated as a family-level KILL without three clean adversarial
  review passes and contest-grade artifact custody.

### 2026-04-30T22:55Z Running Eval Claim-State Update

- Both clean isolated Lightning jobs are now `Running`.
- No Grand Council claim state changes: running jobs are not evidence.
- The next scientifically admissible branch is:
  terminal status -> artifact harvest -> archive identity validation -> CUDA/T4
  provenance check -> adjudication JSON validation -> paired R5/PFP16 review ->
  Grand Council adversarial promotion or forensic failure classification.

### 2026-04-30T23:10Z Grand Council Exact-Evidence Update

- The branch completed: terminal status -> SDK artifact harvest -> local
  validation -> paired R5/PFP16 review.
- PFP16 paired calibration scored `1.037045485927815` on exact CUDA/T4 but
  fired the strict SegNet component gate versus the reference component
  baseline. Treat as calibration/forensic evidence, not a new promotion packet.
- OWV3 R5 scored `1.0373951773937642`, fired the same SegNet component gate,
  and was worse than paired PFP16 by `0.00034969146594909795`.
- Grand Council verdict for this branch: R5 is not promoted and not a family
  KILL. It is a precise negative for the current R5 byte-plan/sensitivity
  configuration. The next admissible build must target SegNet-conservative
  perturbations or use an official finite-difference `component_sensitivity_v1`
  artifact.
- The writeup may cite these values only with the labels "exact CUDA/T4
  forensic, non-promotable due predeclared component gate" and "paired R5 worse
  than paired PFP16"; it must not cite R5 as a frontier result.

### 2026-04-30T23:30Z Grand Council Queue/Gate Update

- R6 is now the active OWV3 branch, not a result:
  `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1` is `Pending` on
  Lightning as of `2026-04-30T23:29:12Z`.
- R6 candidate:
  `owv3_0076_bbr0p65_protect0p0013_aggr1em05`, `686531` bytes, SHA
  `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`,
  `-104` bytes versus paired PFP16, low-bit channels reduced from R5's `62`
  to `58`.
- The R6 selector is now codified in `experiments/sweep_owv3_byte_plan.py`:
  after failed exact R5, a candidate must be byte-feasible, promotion-eligible
  by byte-plan metadata, fallback `keep_asym`, no diagnostic fp16 layers, and
  strictly lower OWV2-low-bit channels than failed R5.
- Exact-eval DX now distinguishes operational failure from scientific
  component-gate failure: Lightning jobs can complete with valid forensic
  artifacts while remaining `promotion_eligible=false`.
- Sensitivity promotion was hardened against zero-signal finite-difference
  curves, NaN/Inf maps, contest JSON/archive custody mismatch, and sample-plan
  hash rewriting.
- Grand Council claim state is unchanged: PFP16 A++ remains the only rankable
  frontier artifact; R6 is pending queue evidence only.

### 2026-04-30T23:44Z Swarm Hardening / R6 Running Update

- R6 exact eval is now `Running` on Lightning as of `2026-04-30T23:42:42Z`,
  cost `0.0882`, not terminal and not harvestable. It remains queue evidence
  only until canonical artifact harvest and adjudication.
- Fixed Lightning status-refresh DX: `refresh-status` now infers SDK job name,
  teamspace, org, and user from `.omx/state/lightning_batch_jobs.json`, so
  operators do not have to retype launch context during harvest monitoring.
- Extended the Lightning/PyPI supply-chain guard after the 2026-04-30
  `lightning==2.6.2/2.6.3` compromise reports. Strict scan now covers `tools/`
  and blocks stale `.venv/bin/lightning`, bare `lightning <subcommand>`, and
  `$LIGHTNING` console-script wrappers. `tools/lightning_run.sh` and
  `tools/lightning_monitor.sh` now use SSH only.
- Local exposure remains clean: strict scan recorded at
  `.omx/state/lightning_supply_chain_scan_20260430_codex_tools_hardened.json`
  reports `status=OK`, no PyPI `lightning` or `pytorch-lightning`, and
  `lightning-sdk==2026.4.10`.
- Component-sensitivity finite differences are now explicitly diagnostic:
  `promotion_eligible=false`, `official_component_response=false`, no
  `--manifest-output`, exact `1200`-frame diagnostic guard, and
  `not_canonical_inflate_eval_path` blocker until archive -> `inflate.sh` ->
  `upstream/evaluate.py` custody exists.
- Lane 12 remains a no-go. Worker E found missing L2 clearance and exact CUDA
  negative `jsonfix40` evidence (`score=26.03719330455429`,
  PoseNet `49.7784996`, SegNet `0.03528685`); do not dispatch Lane 12 before a
  real `.omx/state/lane12_nerv_l2_clearance.json` packet and three clean
  reviews.
- KL hardening landed: high-weight `kl_distill_weight >= 0.1` in
  `train_renderer` now requires explicit forensic opt-in; legacy 1.0-weight KL
  profiles are non-promotable/forensic; FilmCanvas KL is scoped to `segnet_aux`.
  Corrected Lane D-V3 remains promotion-capable only pending exact gates.
- NWCS sensitivity input builder landed at
  `experiments/build_nwcs_sensitivity_inputs.py`. It consumes only promotable
  `component_sensitivity_v1` and rejects fake/proxy/uniform/stale or
  incomplete coverage maps. It does not create sensitivity evidence; it blocks
  fabrication until canonical component sensitivity exists.
- MCP helper processes were killed again after respawn. The repo/config
  preflight may be clean, but the host still has an external respawn source
  that must be removed outside this repo if it appears again.

### 2026-04-30T23:48Z Grand Council R6 Exact Negative

- R6 terminal harvest completed with valid exact CUDA/T4 custody:
  `score_recomputed_from_components=1.0393166493980681`, `686531` bytes, SHA
  `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`,
  `avg_posenet_dist=0.00323147`, `avg_segnet_dist=0.00402421`, `n_samples=600`,
  GPU `Tesla T4`.
- It regressed versus paired PFP16 by `+0.0022711634702530237` score while
  saving `104` bytes.
- Component gate outcome changed from the R5 failure mode: PoseNet failed
  (`1.0213113614240024` relative > `1.002`), SegNet passed
  (`1.0011319365319455` relative <= `1.002`).
- Strict final-deploy adjudication returned exit code `2`. R6 is
  `promotion_eligible=false` forensic evidence, not a deploy candidate.
- Grand Council interpretation: scoped negative for this R6 byte-plan/config,
  not an OWV3 family KILL. The next OWV branch must solve PoseNet drift while
  preserving the SegNet improvement, or switch to canonical sensitivity
  evidence before spending more exact evals.

### 2026-04-30T23:58Z Next-Wave Telemetry / Research / DX Update

- New adjacent ledger:
  `.omx/research/shannon_floor_nextwave_telemetry_and_research_20260430_codex.md`.
- MCP helper processes were killed again and verified absent.
- Live provider posture:
  - Vast: `.venv/bin/vastai show instances --raw` returned `[]`.
  - Modal: `.venv/bin/modal app list` showed zero tasks.
  - Lightning: SDK bulk refresh updated 9 non-dry-run local records, skipped
    13 dry-runs, and had zero failures. No new harvestable result surfaced
    beyond already harvested R6.
- DX hardening landed:
  `scripts/launch_lightning_batch_job.py refresh-status --all` now performs a
  state-driven bulk SDK refresh without using the Lightning console script.
  Verification: `src/tac/tests/test_lightning_batch_jobs.py` passed
  `33 passed`; Python compile passed.
- Supply-chain scan:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_nextwave.json`
  reports `status=OK`, no PyPI `lightning`/`pytorch-lightning`, and zero
  violations.
- Research-agent consensus:
  arXiv:2604.26919v1, PufferLib/RL/LM Studio/visual primitives, and
  Training-Free GRPO are useful for proposal hygiene, dual-readout audits,
  bandit/BOHB scheduling, and experience memory only. They are not score
  evidence and must not add runtime dependencies or weaken exact CUDA gates.
- Provider audit worker wrote
  `.omx/research/provider_telemetry_canonical_harvest_audit_20260430_worker_c.md`
  and independently confirmed no live Vast/Modal/Lightning running work from
  local state; historic Vast tracker rows are stale and not live spend truth.
- OWV3 R7 guardrail landed in `experiments/sweep_owv3_byte_plan.py` with
  coverage in `src/tac/tests/test_sweep_owv3_byte_plan.py`.
  `r7-pose-balanced` excludes exact failed R5/R6 candidates, requires byte
  feasibility versus paired PFP16, keeps `fallback_action=keep_asym`, rejects
  diagnostic FP16 layers and non-promotable rows, requires OWV2-low-bit
  channels `<=58`, and requires `bit_budget_ratio>=0.65`. On the existing
  byte-plan rows it returns zero candidates, so blind scalar OWV3 thresholding
  is stopped pending component-balanced PoseNet/SegNet sensitivity evidence.
  Verification: `13 passed` for `test_sweep_owv3_byte_plan.py`, plus compile
  and whitespace checks.
- Official component-response producer landed at
  `experiments/profile_component_sensitivity_official.py` with tests in
  `src/tac/tests/test_profile_component_sensitivity_official.py`. It evaluates
  baseline/perturbation archives through `contest_auth_eval.py` or validates
  existing exact `contest_auth_eval.json` custody, then emits PoseNet, SegNet,
  and combined official response curves for
  `build_component_sensitivity_manifest.py`. It does not generate perturbation
  archives, component maps, or stability JSON. Codex added a fail-closed guard
  rejecting the baseline archive at nonzero epsilon. Verification:
  official-response + manifest/schema suite `48 passed`, plus compile and
  whitespace checks.

### 2026-05-01T00:22Z Next Wave 2 Integration / Official Response Queue Ready

- Integrated the xhigh swarm outputs for perturbation archive generation,
  Lightning official-response queue readiness, Alpha diagnostics, NWCS
  fail-closed sensitivity use, claim hygiene, and Modal CPU advisory guard.
- New deterministic perturbation plan producer:
  `experiments/build_component_response_perturbation_plan.py` emits
  `official_component_response_plan_v1` plus deterministic archive variants
  with custody metadata. It is explicitly a plan/archive producer, not score
  evidence; CUDA exact eval remains mandatory.
- Lightning Batch Jobs now have an `official_component_response` role with
  CUDA runner preflight, strict Lightning supply-chain scans, DALI
  hash-pinned/no-deps bootstrap, staged-manifest gating for non-dry-run submit,
  compact cleanup, local validation, SSH harvest validation, and a runbook at
  `docs/runbooks/lightning_official_component_response.md`.
- Parent integration added regression coverage for the official-response
  subprocess command so timeout flags cannot be duplicated/misparsed before
  `contest_auth_eval.py` runs.
- OWV3/PFP16 calibration claim hygiene is stricter: failed-gate exact CUDA/T4
  records carry `promotion_eligible=false`,
  `paper_claim_grade="A-negative scoped forensic"`, and
  `allowed_use=["forensic","no_rank_frontier","no_promotion"]`; `evidence_grade`
  is hardware/custody grade only.
- Alpha/Lane 12 remains blocked from retraining or exact-eval spend without L2
  clearance. The new residual-region ranking artifact is diagnostic only and
  confirms the current `jsonfix40` measured config remains rejected, not
  family-killed.
- Provider telemetry at this checkpoint: Vast has no live instances, Modal has
  zero tasks, Lightning has no running jobs, and the only completed refreshed
  job is the already-harvested R6 exact negative.
- Verification in parent:
  - official response / perturbation plan / Lightning queue suite:
    `52 passed`;
  - Alpha diagnostics: `11 passed`;
  - remote auth + NWCS guard slice: `87 passed`;
  - Modal CPU advisory guard: `33 passed`;
  - Python compile and touched shell syntax checks passed.
- Open Grand Council research swarm, spawned xhigh after closing the first six
  workers: arXiv:2604.26919v1, PufferLib/RL + visual primitives, Tencent
  training-free GRPO, and KL-distill architecture audit.

Next admissible production sequence:

1. Generate official perturbation archives from a reviewed byte/semantic basis.
2. Stage all inputs through the Lightning reproducibility workspace manifest.
3. Submit the official component-response Batch Job with `--require-passed`.
4. Harvest and validate compact artifacts; then assemble
   `component_sensitivity_v1` only with exact CUDA response curves, component
   maps, stability metrics, sample plan, and custody.
5. Use that packet to unlock component-balanced OWV3/NWCS/Alpha choices; do
   not exact-eval another scalar-threshold OWV3 grid point blind.

### 2026-05-01T00:34Z External Research Intake / MCP Config Hardened

- Research agents completed intake on arXiv:2604.26919v1, DeepSeek visual
  primitives / PufferLib / LM Studio, and Tencent Training-Free GRPO. Consensus:
  these are proposal, diagnostics, and meta-optimization methods only. They are
  not score evidence and must not weaken exact CUDA archive custody.
- Adopt immediately:
  - Alpha visual-primitives diagnostic packet over decoded baseline masks.
  - Beta/OWV3 dual-readout rule: structural channel/Fisher signal plus
    held-out finite-difference component response.
  - Hashed read-only experience library for grouped agent/lane proposals,
    scored only from structured artifacts.
- Defer:
  - PufferLib/PPO-style search until a cheap surrogate reward has exact CUDA
    rank-correlation anchors.
  - External learned-codec stacks until a measured component proves byte value
    under contest inflate and archive side-info budgets.
- Related-method notes:
  CI-ICM/channel-importance is a strong Beta/OWV3 design input; S2-CoT warns
  that feature adaptation and entropy modeling must be co-tuned; TinyNeRV is
  relevant to Alpha but only with decoded-mask training/eval discipline.
- MCP configuration and process cleanup:
  - killed live Roblox/Chrome MCP helpers;
  - disabled `game-studio@openai-curated` and `cloudflare@openai-curated` in
    `/Users/adpena/.codex/config.toml`;
  - removed `.playwright-mcp` and transient `.codex/.tmp/.../.mcp.json` files;
  - confirmed `/Users/adpena/.claude/mcp.json` is empty and no MCP helper
    process remains.
- KL-distill architecture audit remains the only active research subagent from
  this wave.

### 2026-05-01T00:41Z KL Audit Integrated

- KL Grand Council audit completed. Primary scorer KL remains mostly fenced,
  but promotion-grade use still requires a shared typed policy/provenance
  architecture across `TrainConfig`, SegMap, `train_renderer`,
  `optimize_poses`, remote runners, and adjudication.
- Low-risk hardening landed immediately:
  - `kl_distill_scorer_loss` and `kl_distill_segnet_only` validate finite
    positive temperature before any logit division;
  - JBL documentation no longer claims it cannot induce PoseNet collapse, and
    now states exact CUDA component-gate proof is required for promotion.
- Verification:
  KL/training focused suite `70 passed`; py_compile passed for touched
  KL/loss/test files.
- Remaining KL work is a first-class next implementation lane:
  add `src/tac/kl_config.py`, normalize legacy CLI/profile flags into one
  policy object, serialize KL policy in all provenance, extend preflight and
  adjudication blockers, and require teacher/student roundtrip contracts plus
  exact CUDA non-collapse evidence before any KL-family claim.

### 2026-05-01T01:02Z Official Response Plan, KL Policy Schema, Alpha Bounded Packet

- Closed the xhigh swarm after integrating all six reports. Scout confirmed
  PFP16 A++ as the exact CUDA archive anchor:
  `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip`,
  `686635` bytes, SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`;
  baseline eval JSON SHA
  `b84e7352165cf0d2be0631a177d8404c3a1e5f27633d82d48990552b9ef382ab`.
- Added ASYM-aware perturbation-basis selector:
  `experiments/select_renderer_blob_perturbation_basis.py`. It parses the
  renderer container, avoids magic/header/length/scale/bias bytes, selects
  deterministic quantized-weight payload bytes, and CPU-decode-verifies every
  epsilon renderer before handoff.
- Generated first reviewed official component-response plan artifacts in
  `experiments/results/official_component_response_pfp16_a_plus_plus_20260501_codex_r1/`:
  `official_component_response_plan.json` SHA
  `cc264c432bcdda0748ae3a8bf945f678ac5db26a4247c4531ce4e09ffd74999c`,
  `perturbation_basis_v1.json` SHA
  `7f3e85c60b363cca039dd8ff003a84529d59f0470886dccda58c614e51f3cdbf`,
  and four nonzero archive variants for eps `-2,-1,+1,+2`, each `686632`
  bytes. This is staging input only, not score evidence.
- Produced a Lightning component-response dry-run queue record:
  one `official_component_response` `DRY_RUN` for
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r1`.
  Manifest closure exists at
  `.omx/state/official_component_response_pfp16_a_plus_plus_20260501_codex_r1_manifest.json`
  and includes the baseline archive/eval JSON, plan, both basis files,
  archive-variant manifest, and all four nonzero archives.
- Lightning non-dry-run submit is blocked by SSH auth, not repo readiness:
  `ssh scratch-studio-devbox` and direct
  `ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai` both return
  `Permission denied (publickey)` after running the user-provided setup script.
- Added typed KL/distillation policy surface:
  `src/tac/kl_config.py` with tests in `src/tac/tests/test_kl_config.py`.
  It normalizes legacy flags into frozen `distillation_policy_v1` provenance,
  bans primary scorer KL unless explicitly forensic/non-promotable, fences
  legacy `segnet_kl` and JBL, and requires eval-roundtrip contracts for
  promotion-capable SegNet-aux KL.
- Tightened paper/ledger claim hygiene in `docs/paper/ara/logic/claims.md`,
  `docs/writeup_draft.md`, `docs/archive/killed_techniques/README.md`, and
  the Grand Council source doc.
- Integrated Alpha visual-primitives diagnostics into
  `experiments/diagnose_nerv_geometry.py` with strict no-claim metadata and a
  bounded `--visual-frame-stride` CLI path. Full-sequence Lane 12 vs PFP16
  visual extraction was stopped after CPU cutoff; existing full scalar Alpha
  JSON remains the full-sequence evidence.
- Telemetry: read-only provider audit found no live Vast instances and no
  Lightning running/pending jobs in structured state. Modal has stale local
  `not_ready` entries only. MCP process probes are clean at close.
- Verification in parent:
  official response / plan / Alpha / KL focused suite `31 passed`;
  KL + J-NWC/NWCS hardening suite `105 passed`; Alpha stride suite `13 passed`;
  py_compile and J-NWC/NWCS shell syntax passed; scoped diff checks passed.

### 2026-05-01T01:36Z KL Policy Runtime Greenup

- Ran an xhigh read-only KL hardening audit after landing the first policy
  schema. The audit identified two immediate high-severity gaps: policy
  validation was not forced before trainer startup, and policy custody lived
  mostly in sidecar JSON rather than movable checkpoint artifacts.
- Hardened `distillation_policy_v1` runtime integration:
  - `TrainConfig` now exposes `forensic_reason`, validates the frozen
    distillation policy during config construction, and exposes a canonical
    policy SHA-256 helper;
  - `Trainer` and `SegMapTrainer` now normalize/store the frozen policy at
    construction time before optimizer/scorer training work can proceed;
  - active SegNet-aux KL now fails before trainer construction if
    `kl_distill_temperature < 2.0`;
  - forensic primary KL and legacy SegNet-KL must carry an explicit reason.
- Embedded policy custody in training artifacts:
  generic training state, int8 checkpoint metadata, renderer training state,
  renderer FP4/FP32 `__meta__`, best-checkpoint meta JSON, and JSONL telemetry
  now carry `distillation_policy` and `distillation_policy_sha256` where those
  paths are available.
- Tightened profile/preflight surface:
  `check_distillation_policy_schema_clean(strict=True)` is wired into
  `preflight_all`, validates live profiles through `src/tac/kl_config.py`, and
  catches active KL/JBL schema drift. Live profile scan is clean.
- Verification:
  KL/config/training/loss suite `99 passed`; train-renderer/preflight adjacent
  suite `270 passed`; py_compile and scoped diff-check passed; MCP helper probe
  was clean after killing respawned `rbx-studio-mcp` and
  `chrome-devtools-mcp` helpers.

### 2026-05-01T01:30Z Continuation Swarm / Supply-Chain And Promotion Hardening

- Spawned/closed xhigh agents for KL/distillation promotion hardening,
  J-NWC/NWCS custody, six-item ops audit, Lightning security advisory review,
  arXiv/Tencent research intake, and PufferLib/visual-primitives research.
  Alpha visual-primitives runtime unblock also completed before closeout.
- Landed fail-closed KL/JBL/distillation promotion adjudication:
  `scripts/adjudicate_contest_auth_eval.py` now reports component distances,
  component gates, `promotion_eligible`, `paper_claim_grade`, `allowed_use`,
  and distillation-policy gate violations. Distillation-active artifacts must
  carry `distillation_policy_v1`, matching policy SHA, exact CUDA device,
  archive SHA/bytes, and passed PoseNet/SegNet component gates before
  promotion.
- Extended remote preflight in `src/tac/preflight.py` so remote scripts with
  KL/JBL/distillation promotion paths must include policy provenance and
  component-gate adjudication. Regression tests landed in
  `src/tac/tests/test_remote_auth_eval_hardening.py`.
- Landed NWCS corpus-manifest custody rechecks in
  `experiments/build_nwcs_sensitivity_inputs.py`: selected checkpoint
  size/SHA, tensor shape, dtype, block count, and used-block count are
  revalidated before sensitivity projection.
- Hardened the Lightning official component-response dry-run path:
  `scripts/launch_lightning_batch_job.py component-response` now fail-closes
  if only one of `--source-manifest` or `--local-perturbation-plan` is passed,
  and validates that every plan-listed perturbation archive is present in the
  staged manifest even on `--dry-run`.
- Re-ran the prepared official component-response closure guard against
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r1`; dry-run
  validation passed with command SHA
  `1c6231a6c7e6528f5ab4a6587a5d3aad79602834eaa5a963b0cf0dfd818e0dab`.
  No non-dry-run submission occurred because Lightning SSH still returns
  `Permission denied (publickey)`.
- Urgent Lightning security review result:
  the active venv has no PyPI `lightning` or `pytorch-lightning`, has
  `lightning_sdk==2026.4.10`, and strict scan output
  `.omx/state/lightning_supply_chain_scan_20260501_codex_ioc_expanded.json`
  reports `status=OK`, `violation_count=0`.
- Expanded the Lightning compromise preflight IOC set to include the reported
  2.6.3 `_runtime/start.py`, malicious `lightning/__init__.py`, and
  malicious `lightning` 2.6.2/2.6.3 wheel hashes, plus pip/uv cache scans for
  cached 2.6.2/2.6.3 artifacts.
- Research consensus from the new paper/OSS swarm:
  arXiv:2604.26919v1, Tencent Training-Free GRPO, PufferLib, and DeepSeek
  visual primitives are admissible as control-plane methods only:
  dual-readout validation, warm-ramp scheduling, byte-aware sparse winner
  allocation, deterministic bandit/BO trial control, and visual-primitive
  Alpha diagnostics. They are not score evidence, runtime dependencies, kill
  evidence, or promotion evidence.
- Ops audit repeated: Vast live `[]`, Modal tasks `0`, Lightning no running
  jobs. Existing R4/R5/R6 OWV3/Fisher negatives remain scoped forensic
  component-gate failures, not family kills. Lane 12 jsonfix40 remains
  A-negative diagnostic only and lacks L2 retraining clearance.
- Verification in parent:
  supply-chain/preflight focused `27 passed`; combined preflight, Lightning
  batch, KL/adjudication, and J-NWC/NWCS regression slice `347 passed`;
  py_compile and scoped diff-check passed for touched implementation/test
  files.

### 2026-05-01T01:38Z Lane 12 Alpha Diagnostic Runtime Unblocked

- Integrated the Alpha runtime worker output. `experiments/diagnose_nerv_geometry.py`
  now supports bounded `.nrv` CPU streaming decode, deterministic decoded-mask
  cache, bounded visual boundary sampling, and explicit no-claim residual/track
  skip controls. Tests landed in
  `src/tac/tests/test_lane12_nerv_geometry_diagnostics.py`.
- Real bounded diagnostic artifact:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_pfp16_visual_primitives_bounded_20260501.json`.
  It covers 1200 scalar frames and 1200 visual frames using the decoded-mask
  cache at
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/predecoded_mask_cache`
  (`450M`).
- No-claim status is intact:
  `visual_primitives.promotion_eligible=false`,
  `score_claim_eligible=false`, `exact_eval_claim=false`, evidence grade
  `empirical`, device `cpu`.
- Diagnostic conclusion: no L2 unblock. The exact-eval spend gate fails with
  blockers `global_disagreement`, `boundary_2px_disagreement`,
  `pair_transition_disagreement`, `critical_missing_rate`, and
  `critical_missing_area_rate`. Observed core metrics:
  global disagreement `0.012303928799099393`, 2px-boundary disagreement
  `0.14883144511692872`, pair-transition disagreement
  `0.009507171571470149`, critical missing rate `0.6452857808237408`,
  critical missing area rate `0.004038536840025019`.
- Parent verification:
  py_compile plus Alpha diagnostic focused tests `18 passed`; scoped
  diff-check clean.

### 2026-05-01T01:58Z Lightning Exact-Response Queue And Runtime Hardening

- Lightning SSH is now operational through the alias `scratch-studio-devbox`,
  but the interactive Studio shell is currently CPU-only:
  `torch.cuda.is_available=false`, `device_count=0`, `nvidia-smi=None`.
  Diagnostic evidence:
  `.omx/state/lightning_ssh_runtime_cuda_preflight_20260501_cpu_only.json`.
- Landed a fail-fast runtime guard in `scripts/lightning_repro_workspace.py`:
  `--ssh-check-only --require-cuda` now probes the remote Python/Torch runtime
  and fails on CPU-only Studios. Plain SSH success is no longer allowed to
  masquerade as GPU readiness for interactive CUDA work.
- Integrated the Lightning SSH/provider-auth hardening worker output:
  static preflight now rejects unsafe Lightning SSH policy such as disabled
  host-key checking, `/dev/null` known-hosts, and bare `ssh.lightning.ai`
  provider-host usage in Lightning scripts/runbooks.
- `AGENTS.md` now records the permanent rule: interactive Lightning CUDA work
  requires the runtime CUDA probe; Batch Jobs remain governed by their own
  `lightning_runner_preflight.json` artifacts.
- Official component-response jobs in flight:
  - `official_component_response_pfp16_a_plus_plus_20260501_codex_r1`: T4,
    `Running`, `--require-passed`, command SHA
    `772395f8e71bf67b095f2e36dd56479d52f82b25fab613b0e2dd61ccd71c0c45`.
  - `official_component_response_pfp16_a_plus_plus_20260501_codex_r2_t4_small_race`:
    T4_SMALL, `Running`, `--require-passed`, command SHA
    `c7cc181f924f50df1ba65c10b30c78adef1f5bdb9e4615b75d3c655feb7432fe`.
  - `official_component_response_pfp16_a_plus_plus_20260501_codex_r3_t4_no_gate`:
    T4, `Pending`, no `--require-passed`, command SHA
    `da87a91dc26a68a451a1326b33d234e7a4f77160c3f3cd521efecccaa6f23b5f`.
- Rationale for the r3 no-gate race: adversarial review found the current
  official response plan lacks nonzero `predicted_delta` entries, so
  `--require-passed` may fail promotion gates after producing scientifically
  useful official CUDA response curves. R3 is diagnostic CUDA evidence only and
  is not promotable `component_sensitivity_v1`.
- Latest known Lightning status at `2026-05-01T01:57:09Z`: r1 `Running`,
  r2 `Running`, r3 `Pending`. Harvest terminal jobs immediately. If r1/r2
  fail after writing curves, harvest without `--require-passed` for forensic
  official-response evidence; do not discard curves because the prediction gate
  model was incomplete.
- Verification: merged Lightning/MCP hardening slice passed py_compile,
  `bash -n` on touched Lightning shell tools, MCP cleanup, and focused tests:
  `38 passed`.

### 2026-05-01T02:15Z Deterministic Lightning Component-Response Harness

- Classified r1/r2/r3 component-response jobs as harness/snapshot failures,
  not lane-performance evidence. r1 failed before CUDA/DALI/input preflight
  because its stale snapshot failed `scripts/scan_lightning_supply_chain.py
  --strict` on old `tools/lightning_*` wrappers invoking the PyPI `lightning`
  console script. r2 failed from the same stale snapshot class. r3 was stopped
  to avoid spend on known-bad provenance.
- Landed reusable production harness fixes:
  `scripts/launch_lightning_batch_job.py component-response` and `exact-eval`
  now support `--remote-preflight-ssh-target`, which runs the strict remote
  supply-chain scan before non-dry-run SDK submission. Component-response SSH
  harvest now supports `--job-name --state-path`, deriving the persisted SDK
  artifact mirror from recorded `remote_output_dir` and `sdk_artifact_path`.
- Added `LightningBatchJobsClient.harvest_ssh_component_response_artifacts`
  and regression coverage for state-derived component-response harvest and
  remote pre-submit supply-chain preflight. Focused verification:
  `src/tac/tests/test_lightning_batch_jobs.py` = 45 passed; py_compile clean;
  scoped `git diff --check` clean; MCP cleanup matched 0 processes.
- Updated `AGENTS.md` and Lightning runbooks: `comma-lab` teamspace,
  `lossy-compression-challenge` Studio, state-derived harvest, and remote
  preflight are now the documented path. Manual `/teamspace/jobs/...` path
  composition is explicitly non-promotable.
- Staged r4 through `scripts/lightning_repro_workspace.py` with explicit
  artifacts and remote byte verification. Manifest:
  `.omx/state/official_component_response_pfp16_a_plus_plus_20260501_codex_r4_clean_t4_stateful_manifest.json`;
  file_count 1114; total_bytes 21307573; manifest SHA
  `80d44b40b4048ee1d2c7ba850e1e98e45025eda65b248b12a494d6e1fdf1928e`.
- Submitted clean r4 diagnostic no-gate T4 Batch Job:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r4_clean_t4_stateful`;
  command SHA `d9eec67b70b20b938dc76b66b34e0f498cc7d92e5307348c8798c0aa072a63c0`;
  link
  `https://lightning.ai/adpena/comma-lab/studios/lossy-compression-challenge/app?app_id=jobs&job_name=official-component-response-pfp16-a-plus-plus-20260501-codex-r4-clean-t4-stateful`.
  Latest status at submit refresh: `Pending`. Harvest with state-derived
  component-response SSH once terminal.

### 2026-05-01T02:38Z r4 Forensic Root Cause, r5 Portable Plan Queue

- Refreshed r4:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r4_clean_t4_stateful`
  reached strict remote supply-chain OK, hash-pinned DALI OK, and
  `lightning_runner_preflight.json` with `cuda_available=true`,
  `device_name=Tesla T4`, then failed before official profiler output.
  SDK logs identify the root cause as a non-portable plan path:
  `baseline_contest_auth_eval_json` resolved to `/Users/adpena/...` inside
  the job. Classification remains harness/input-plan failure only; no
  lane-performance evidence.
- Landed permanent fix class:
  `experiments/profile_component_sensitivity_official.py` and the generated
  Lightning component-response input preflight now let explicit
  `--baseline-contest-auth-eval-json` override stale plan metadata. The plan
  builder now emits repo-internal paths relative to the plan file instead of
  host-local absolutes when possible.
- Hardened submit closure:
  `scripts/launch_lightning_batch_job.py component-response` now rejects
  absolute plan point archives and per-point eval JSON unless a top-level
  baseline JSON is explicitly supplied as the authority. Existing r1 plan also
  had a zero-epsilon absolute archive path, so a portable r5 plan was produced
  at
  `experiments/results/official_component_response_pfp16_a_plus_plus_20260501_codex_r5_portable_plan/official_component_response_plan.json`.
- Hardened SSH transfer DX:
  `scripts/lightning_repro_workspace.py` and Lightning SSH harvest helpers now
  reuse noninteractive SSH policy for actual `ssh`/`scp`/`rsync` operations:
  `BatchMode=yes`, password and keyboard-interactive auth disabled, and
  explicit `ConnectTimeout`.
- Verification: py_compile clean and focused tests passed:
  `src/tac/tests/test_lightning_batch_jobs.py`,
  `src/tac/tests/test_lightning_repro_workspace.py`,
  `src/tac/tests/test_profile_component_sensitivity_official.py`, and
  `src/tac/tests/test_build_component_response_perturbation_plan.py`
  (`85 passed`).
- Pre-submit doctor passed:
  `.omx/state/lightning_doctor_20260501_r5_pre_submit.json` with local and
  remote supply-chain `status=OK`, SSH auth OK, and T4 inventory OK.
- r5 staged with remote byte verification. Manifest:
  `.omx/state/official_component_response_pfp16_a_plus_plus_20260501_codex_r5_explicit_baseline_t4_manifest.json`,
  file_count `1115`, total_bytes `21367564`, manifest SHA
  `ae3028935151c8e8e8f57315fa2a4f54edbfaebf3e6fd6c56064824e36f7e7e4`.
- Submitted r5 diagnostic no-gate T4 Batch Job:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r5_explicit_baseline_t4`,
  command SHA `182c287d986a4fce61dbf12871b1e985bf01c4715b8e897e980e44d7e9c6ffa7`.
  Latest refresh at `2026-05-01T02:38:30Z`: `Pending`. Harvest
  state-derived component-response artifacts when terminal; do not classify
  response quality until canonical curves and validation JSON are local.

### 2026-05-01T02:55Z r6 Race Queue And Submit-Closure Hardening

- r6 was staged to race r5 wall-clock queue latency on T4_SMALL:
  `official_component_response_pfp16_a_plus_plus_20260501_codex_r6_t4_small_race`.
  Remote manifest verification passed with file_count `1115`, total_bytes
  `21370080`, manifest SHA
  `91cd1e8011a7045a3068b0a2a4a74b0f842be6b8be4232da91473e6445780684`.
  Remote environment/supply-chain preflight stayed clean: compromised
  `lightning` package absent; `lightning-sdk` absent from the remote venv at
  staging time; no runtime-security findings.
- Submitted r6 diagnostic no-gate component-response Batch Job:
  command SHA `8cd111eb0b3448c1f9143929a96b0fd990afd361e6488c66a2c7c2ed0086deec`,
  expected baseline archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  size `686635`.
- Latest Lightning status refresh:
  r5 `Running` at `2026-05-01T02:48:23Z`; r6 `Pending` at
  `2026-05-01T02:48:24Z`. Neither had harvestable local artifacts yet.
  The interactive Studio SSH filesystem did not show the remote output dirs;
  continue judging only from SDK status and terminal harvested artifacts.
- Grand Council adversarial review found an additional closure bug class:
  raw manifest path strings could include repo-relative traversal such as
  `../archive.zip`. Permanent fix landed in
  `scripts/launch_lightning_batch_job.py`: exact-eval and component-response
  source manifests now reject absolute paths, traversal, empty/unstable
  separators, backslashes, control characters, duplicate entries, and hidden
  or resource-fork components. `_remote_repo_rel` now also validates the
  derived repo-relative path.
- Direct SSH artifact harvest is hardened in
  `src/tac/deploy/lightning/batch_jobs.py`: library calls reject the bare
  `ssh.lightning.ai` provider host and whitespace/control-character targets,
  aligning lower-level helpers with CLI SSH custody policy.
- Exact-eval submit closure now proves queue-metadata `baseline_json` or
  `baseline_contest_auth_eval_json` is present in the staged source manifest
  when provided by wrappers such as `scripts/lightning_exact_eval_repro.py`.
- Verification: py_compile clean for touched Python; focused Lightning batch
  suite passed with `57 passed`.
- Parallel worker `Sagan the 2nd` is implementing the next J-NWC/NWCS custody
  fix: prebuilt corpus-manifest plus replay-root support in remote scripts so
  CUDA exact eval consumes the exact manifest that produced
  `CORPUS_SENSITIVITY_PT`, failing closed on SHA/custody mismatch.
- Latest refresh at `2026-05-01T02:54Z`: r5 remains `Running` on T4
  (`total_cost=0.11215556`); r6 is now also `Running` on T4_SMALL
  (`total_cost=0.007388889`). No terminal artifact exists yet. Vast live
  inventory is empty. Modal stale call polls returned `STILL RUNNING` for
  six labels despite app list showing zero live tasks; no Modal artifacts were
  harvested or classified.

### 2026-05-01T03:10Z r5 Harvested, J-NWC Custody Landed, Alpha Gate Rechecked

- r5 component-response completed and was harvested statefully:
  `experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_codex_r5_explicit_baseline_t4`.
  Validation confirms CUDA evidence, expected baseline archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  size `686635`, point_count `5`, nonzero_point_count `4`, and copied
  canonical per-point eval/provenance artifacts. It is diagnostic only:
  `promotion_eligible=false`.
- r5 official response curves have finite signal, zero baseline repro error,
  and coverage passed, but fail promotion because the diagnostic plan has no
  map-predicted component deltas. Observed max absolute official deltas:
  PoseNet `0.0003012800000000001`, SegNet `1.3420000000000099e-05`,
  combined `0.006991338976567674`. Blockers for all curves:
  `missing_prediction_deltas`, `prediction_error_gate_failed`.
- r5 harvest exposed a local mirror bug class: copied remote validation JSON
  can be read-only, blocking overwrite during local validation. Fixed in
  `src/tac/deploy/lightning/batch_jobs.py` with atomic JSON replace and
  chmod `0644`; regression added to
  `src/tac/tests/test_lightning_batch_jobs.py`. Verification:
  `57 passed`.
- J-NWC/NWCS corpus-manifest custody landed:
  `experiments/train_neural_weight_codec.py` now supports
  `--corpus-manifest`, `--manifest-out`, and `--corpus-replay-root`,
  preserving exact prebuilt manifest bytes and recording manifest SHA/replay
  root in codec payloads. Remote J-NWC/NWCS scripts accept
  `PREBUILT_CORPUS_MANIFEST` + `CORPUS_REPLAY_ROOT`; NWCS promotion requires a
  matching prebuilt manifest when `CORPUS_SENSITIVITY_PT` is used. Parent
  verification: shell syntax clean, py_compile clean, and focused J-NWC/NWCS
  tests `42 passed`.
- Alpha-Geo current Lane G v3 rerun produced:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_codex_current_20260501.json`,
  SHA `6b17b004d238ada62180077aa072f02594954ef02f5a5c610bc70e65619fa80d`.
  It reproduces the known `jsonfix40` failure: `overall_pass=false`,
  global disagreement `0.012303928799099393`, 2px boundary disagreement
  `0.14883144511692872`, missing-component rate `0.4611606740560512`.
  The duplicate Lane A rerun was stopped after the Lane G packet completed;
  existing Lane A/base diagnostic remains the reference.
- r6 is still live as a redundant T4_SMALL diagnostic race at
  `2026-05-01T03:07:35Z`: `Running`, total_cost `0.048555557`.
  Leave or harvest terminal artifacts only if useful for cross-machine
  reproducibility; r5 already supplies the official diagnostic curve packet.
- r6 then completed and was harvested:
  `experiments/results/lightning_batch/official_component_response_pfp16_a_plus_plus_20260501_codex_r6_t4_small_race`.
  It reproduces the r5 diagnostic shape on T4_SMALL with the same baseline
  archive SHA, CUDA device evidence, coverage, signal, and zero repro; it is
  also non-promotable for the same missing-prediction-delta reason. R6 minus
  r5 delta differences are small: PoseNet within `4.0e-7`, SegNet within
  `6.0e-8`, combined within `8.3e-6` over the epsilon ladder.
