# Tier-A CUDA dispatch packets — PRE-STAGED 2026-05-08

**Author**: Subagent CUDA-PRESTAGE
**Status**: PRE-STAGED, AWAITING OPERATOR AUTHORIZATION
**Date**: 2026-05-08
**Repo HEAD at prestage**: `5064fc3b` (post fix codex HIGH#1 `c83eff00` + post monolithic-closure-gate `2f339175`)

This packet pre-stages three Tier-A `[contest-CUDA]` dispatch candidates so the
operator can fire them with a single AUTH toggle. **No GPU jobs were launched.**
Each candidate has a `pending_authorization` row in
`.omx/state/active_lane_dispatch_claims.md`; flipping to `active_dispatching`
requires explicit operator approval per the
`forbidden_CPU_MPS_derived_dispatch_readiness_flag` rule.

---

## Operator authorization checklist (preview)

| # | Lane | Archive bytes | SHA-256 (truncated) | AUTH | Blockers cleared? |
|---|---|---:|---|:---:|:---:|
| 1 | `pr101_admm_step6_no_dead_k` | 153,671 | `b7b09089…` | `[ ]` | inflate.sh 3-arg ✓; verify-tool ✗ (covers sibling); sanity gates ✗ (anchors=0/3, distortion proxy unset) |
| 2 | `pr106_lagrangian_per_tensor_uniward` | 150,511 | `0641b8ac…` | `[ ]` | runtime packet BUILT (Subagent BUILD-PR106-UNIWARD-RUNTIME); 1200-frame smoke roundtrip ✓; B3 verifier ✓; sanity gates ✗ (anchors=0; no PR106 Lagrangian calibration) |
| 3 | `apogee_int6_contest_cuda_anchor` | 170,450 | `0176a269…` | `[ ]` | basin-parity PASSED; sanity gates ✗ (predicted < rate-distortion floor); 2026-05-07 prior REFUSED at Lightning AWS T4 capacity |

**Authorization semantics:** Toggling AUTH = `[x]` is the gate that lets the
operator (or a downstream subagent acting under operator approval) flip the
matching `pending_authorization` row to `active_dispatching` via
`tools/claim_lane_dispatch.py claim --force --status active_dispatching --notes
"<operator approval ref>"` and then run the dispatcher invocation listed below.

---

## Candidate 1 — PR101 ADMM step 6 no-dead-K @ 153,671 B

### Identity

| Field | Value |
|---|---|
| Lane id | `pr101_admm_step6_no_dead_k` |
| Source commit | `0b24e5d1` (C2 free win: ADMM step 6 no-dead-K variant) |
| Archive path | `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/archive.zip` |
| Archive bytes | 153,671 |
| Archive SHA-256 | `b7b09089e852872bd67b4b8aa04c1b4d46168bb89343acff81796c5551d63d05` |
| Build manifest | `…/build_manifest.json` (`schema_version: admm_x_lossy_coarsening_path_b_step6_no_dead_k_build.v1`) |
| Submission dir | `…/submission_dir/` (inflate.sh + inflate.py + src/codec.py + src/model.py) |
| Predicted band | `[0.18, 0.22]` (per manifest `predicted_band`, `[CPU-build]` evidence_grade) |
| rel_err (int8 vs lossless fp32) | 4.15% (proxy) / 3.62% (fp32 smoke) |

### Verification gates

- [x] **byte-closure**: manifest `archive_sha256` = on-disk SHA = `b7b09089…` ✓.
- [x] **inflate.sh 3-arg signature**: confirmed `submission_dir/inflate.sh`
      reads `${1}` data dir + `${2}` output dir + `${3}` file list,
      mirrors `submissions/robust_current/inflate.sh` post `c83eff00`. ✓.
- [x] **retired-config check (BUGCLASSES B6)**: SHA `b7b09089…` is
      DISTINCT from the retired T0312 config SHA
      `ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28`
      (per `.omx/research/lossy_coarsening_T0312_retired_config_do_not_redispatch_20260508_claude.md`).
      Rel-err 4.15% is also distinct from the retired K_budget=0.05 path. ✓.
- [ ] **`tools/verify_admm_step6_archive_sha256.py`**: confirmed runs against
      the **original** step6 sibling (`23c662d6…`, 153,699 B), NOT the
      no-dead-K variant. The verifier successfully rebuilds the original
      from `tools/build_admm_x_lossy_coarsening_path_b_step6.py` byte-identically;
      the no-dead-K variant has its own dedicated build tool
      `tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py` and 8/8
      passing tests. **Recommendation:** add a sibling verify tool BEFORE
      operator AUTH, or accept the manifest+build-tool determinism as evidence.
- [ ] **predispatch_sanity.py (lane_class=`pr101_admm_lossy_coarsening`)**:
      REFUSED, gate `anchors_sufficient` = 0/3 calibration anchors at
      `.omx/calibration/anchors_pr101_admm_lossy_coarsening.json`. Gate
      `distortion_model_gate` = `--distortion-proxy-ran` not set; rel_err
      4.15% > 1.0% threshold. **Resolution:** either run
      `experiments/distortion_proxy_local.py` first, or supply
      `--override-reason "<≥40 char operator approval>"` at dispatch time
      (override is logged to `.omx/state/predispatch_overrides.log`).
- [x] **apogee_int6 dispatch_blocker**: manifest line 117 explicitly lists
      `apogee_int6_contest_cuda_anchor_required_first`. **This blocker
      should remain set until candidate 3 lands a `[contest-CUDA]` anchor
      score**, per the C3 inheritance from REVIEW-ENG.

### Dispatcher invocation snippet (Lightning T4 — preferred)

```bash
JOB="admm-no-dead-k-$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="experiments/results/lightning_batch/${JOB}"
mkdir -p "${OUTPUT_DIR}"

.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
    --job-name "${JOB}" \
    --archive experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/archive.zip \
    --repo-dir /teamspace/studios/this_studio/pact \
    --upstream-dir /teamspace/studios/this_studio/pact/upstream \
    --output-dir "/teamspace/studios/this_studio/pact/${OUTPUT_DIR}" \
    --inflate-sh experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/submission_dir/inflate.sh \
    --machine T4 \
    --infer-expected-archive \
    --predicted-band 0.18 0.22 \
    --baseline-score 0.20935 \
    --baseline-archive-bytes 178873 \
    --max-runtime 5400 \
    --queue-metadata "lane_id=pr101_admm_step6_no_dead_k" \
    --queue-metadata "subagent=CUDA-PRESTAGE-AUTHORIZED-OPERATOR"
```

**Cost estimate (Lightning T4 g4dn.2xlarge):** ~$0.30–0.60 (~30–60 min wall-clock for inflate + auth eval).

### Vast.ai 4090 fallback (if Lightning T4 capacity refuses)

`scripts/launch_lane_on_vastai.py` requires a `--lane-script` (a
`scripts/remote_lane_<id>.sh`) — none exists for this lane yet. Building one
is out-of-scope for this prestage; if Vast.ai is the chosen path, follow the
canonical pattern from
`reference_lightning_studio_canonical_dispatch_recipe_20260505.md` /
`scripts/remote_archive_only_eval.sh` (canonical `bootstrap_runtime_deps()`
per CLAUDE.md `forbidden_remote_bootstrap_inline`).

### Cross-ref

- Dispatch claim row: `.omx/state/active_lane_dispatch_claims.md`,
  `lane_id=pr101_admm_step6_no_dead_k`, status=`pending_authorization`,
  job=`PRESTAGE:admm-no-dead-k-20260508-PLACEHOLDER`.
- Smallest-credible-bolt-on per May 4 race postmortem rule (Rule 2 inversion):
  this is the smallest credible byte-saving bolt-on (-28 B vs sibling) on top
  of the sibling step 6 ADMM stack.

---

## Candidate 2 — PR106 UNIWARD-Lagrangian @ rms=0.05 = 150,511 B (BUILT)

**Status update 2026-05-08** (Subagent BUILD-PR106-UNIWARD-RUNTIME): runtime
packet BUILT. The original "no byte-closed archive/runtime" blocker is lifted.
This is still `[CPU-build]`, not score evidence; exact CUDA dispatch requires
fresh claim plus explicit operator AUTH/override because sanity gates remain.

### Identity

| Field | Value |
|---|---|
| Lane id | `pr106_uniward_lagrangian_runtime_packet` |
| Source commit | `ff92b954` (empirical sweep) + Subagent BUILD-PR106-UNIWARD-RUNTIME (build tool) |
| Build tool | `tools/build_pr106_uniward_runtime_packet.py` |
| Verifier (B3 custody) | `tools/verify_pr106_uniward_runtime_packet_sha256.py` |
| Empirical manifest | `reports/raw/pr106_lagrangian_per_tensor_allocation_20260508T071433Z/manifest.json` |
| Build manifest | `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/build_manifest.json` |
| Archive path | `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip` |
| Archive bytes | 150,511 (-35,728 vs PR106 published 186,239) |
| Archive SHA-256 | `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b` |
| Submission dir | `…/submission_dir/` (inflate.sh + inflate.py + src/codec.py + src/model.py — PR106-byte-identical decoder) |
| rms_target | 0.05 |
| Achieved rel_err (joint L1) | 4.66% (matches empirical sweep ±51 B drift due to bisect convergence) |
| Per-tensor Ks | `[2,5,5,5,4,6,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]` |
| UNIWARD λ | 6.111e+08 |
| Smoke roundtrip n_frames implied | 1200 (600 latent pairs × 2 frames) |
| Smoke weight-identity rel_err | 1.85e-08 (fp32 ULP — wire-format roundtrip clean) |
| Predicted band | `[0.18, 0.22]` (default; manifest preserves a band for dispatch metadata only) |
| Evidence grade | `[CPU-build]` (not score evidence — operator must AUTH for [contest-CUDA] anchor) |

### Verification gates

- [x] **byte-closure**: archive on disk SHA matches manifest SHA matches B3
      verifier output: `0641b8ac…`. ✓.
- [x] **B3 custody**: rebuild-and-SHA-assert smoke
      (`tools/verify_pr106_uniward_runtime_packet_sha256.py`) confirms
      deterministic reproduction from committed source archive +
      committed build tool. ✓.
- [x] **inflate.sh 3-arg signature**: PR106's source `inflate.sh` already
      implements `${1}` data dir + `${2}` output dir + `${3}` file list
      (FIX-CODEX-HIGH commit `c83eff00` contract). The build tool stages
      it verbatim. ✓.
- [x] **smoke roundtrip**: archive parses via PR106's vendored
      `parse_archive`; decoder yields 28 tensors and 600 latent pairs;
      meta dict matches PR106 (latent_dim=28, base=36, eval=384×512); per-
      tensor weight identity rel_err 1.85e-08 (essentially fp32 ULP). ✓.
- [x] **wire-format identity**: 0xff + 3-byte LE decoder length +
      decoder_packed_brotli + latents_and_sidecar_brotli (verbatim from
      PR106 source). PR106's published inflate path consumes our archive
      with ZERO code changes. ✓.
- [x] **dispatch_blockers cleared (build-side, per manifest)**: ZERO of
      the original six manifest blockers remain. The build manifest
      replaces them with operator-side blockers:
      `cpu_build_rel_err_proxy_not_score_evidence`,
      `exact_cuda_auth_eval_not_yet_harvested`,
      `requires_contest_auth_eval_json_before_score_promotion_rank_or_kill`,
      `no_pr106_lagrangian_calibration_anchor_yet`.
- [ ] **predispatch_sanity.py (lane_class=`pr106_lagrangian_per_tensor`)**:
      anchors_sufficient = 0/3 (no PR106 Lagrangian calibration anchors
      yet at `.omx/calibration/anchors_pr106_lagrangian_per_tensor.json`).
      The byte-savings prediction band 0.18–0.22 sits below PR106's
      lossless 0.20454; rate_delta ≈ -0.00951; distortion-proxy unset.
      **Resolution at AUTH time:** operator must supply
      `--override-reason "<≥40 char approval>"` to dispatch.
- [ ] **first PR106 cross-substrate Lagrangian dispatch**: this is the
      FIRST attempt to land a `[contest-CUDA]` score on PR106 substrate
      with per-tensor K coarsening. ANY [contest-CUDA] outcome (GREEN
      0.18–0.22, REGRESSED, or KILLED) immediately becomes the
      calibration anchor that satisfies the missing
      `no_pr106_lagrangian_calibration_anchor_yet` blocker for follow-on
      runs at other rms_targets (0.02, 0.10).

### Dispatcher invocation snippet (Lightning T4 — preferred)

```bash
JOB="pr106-uniward-rms005-$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="experiments/results/lightning_batch/${JOB}"
mkdir -p "${OUTPUT_DIR}"

.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
    --job-name "${JOB}" \
    --archive experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip \
    --repo-dir /teamspace/studios/this_studio/pact \
    --upstream-dir /teamspace/studios/this_studio/pact/upstream \
    --output-dir "/teamspace/studios/this_studio/pact/${OUTPUT_DIR}" \
    --inflate-sh experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/submission_dir/inflate.sh \
    --machine T4 \
    --infer-expected-archive \
    --predicted-band 0.18 0.22 \
    --baseline-score 0.20454 \
    --baseline-archive-bytes 186239 \
    --max-runtime 5400 \
    --queue-metadata "lane_id=pr106_uniward_lagrangian_runtime_packet" \
    --queue-metadata "subagent=BUILD-PR106-UNIWARD-RUNTIME-AUTHORIZED-OPERATOR" \
    --queue-metadata "rms_target=0.05" \
    --queue-metadata "expected_rel_err=0.0466"
```

**Cost estimate (Lightning T4 g4dn.2xlarge):** ~$0.30–0.60 (~30–60 min wall-clock for inflate + auth eval).

### Cost estimate

~$0.30–0.60 Lightning T4. Build phase complete; no further CPU prep needed
before dispatch.

### Cross-ref

- Dispatch claim row: `.omx/state/active_lane_dispatch_claims.md`,
  `lane_id=pr106_uniward_lagrangian_runtime_packet`, status=`pending_authorization`.
- Council recommendation: this candidate is now CPU-build ready for an
  operator-authorized exact-eval attempt. It reuses PR106's published decoder
  verbatim (no new code at inflate time); only the encoder produces
  K-coarsened symbols. Per CLAUDE.md "Auth eval EVERYWHERE" and "Auth eval
  measurement" rules, the byte-closed runtime + B3 verifier + 1200-frame smoke
  roundtrip satisfy the build-side precondition. Operator AUTH and a fresh
  dispatch claim are still required before CUDA spend.
- Memory: `feedback_pr106_cross_substrate_lagrangian_uniward_20260508.md`,
  `feedback_pr106_uniward_runtime_packet_landed_20260508.md`.

---

## Candidate 3 — apogee_int6 [contest-CUDA] anchor

### Identity

| Field | Value |
|---|---|
| Lane id | `apogee_int6_contest_cuda_anchor` |
| Repack metadata | `experiments/results/apogee_int6_repack_20260504_claude/repack_metadata.json` |
| Archive path | `experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip` |
| Archive bytes | 170,450 |
| Archive SHA-256 | `0176a2691a4daf5991170404d30a304ae30389621c0fc54914628414aef39ff1` |
| Source PR106 SHA | `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58` |
| Predicted band | `[0.190, 0.204]` (manifest `predicted_score_band`; `[design-validation]` tag) |
| rel_err per weight | 1.55% |
| Magic byte | `0xA6` |
| Block size | 128 |
| Prediction status (per metadata) | `forensic_byte_only_invalidated_by_int4_exact_negative` |

### Verification gates

- [x] **byte-closure**: repack metadata `candidate_archive_sha256` =
      on-disk SHA = `0176a269…`. ✓.
- [x] **basin-parity PASSED [scorer-basin-parity:CPU]**:
      `experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json`.
      `basin_parity_passed: true`, pose dist delta +1.08e-4 (30× below
      threshold), seg dist delta +9.62e-4 (5× below threshold), Hessian
      log10 ratio +0.02 (within ±1.00 tolerance). Latents match exact
      between candidate and lossless source PR106 archive. ✓.
- [x] **retired-config check (BUGCLASSES B6)**: apogee_int6 is unrelated
      to the lossy_coarsening T0312-noproject retired config (different
      paradigm — fixed-block int6 quantization, not analytical lossy
      coarsening). ✓.
- [ ] **predispatch_sanity.py REFUSED on `sanity_lossy_vs_lossless`**:
      predicted_high=0.2040 is below the SHA-tied rate-distortion floor
      0.3067 (lossless=0.2095, rate_delta=-0.010513,
      component_penalty=0.107727). The gate's reasoning: "Parity/readiness
      evidence is not score-lowering evidence." This is the
      forensic-byte-only invalidation flag from the int4 negative result
      (1.4287 [contest-CUDA T4]) — the int6 prediction band would need
      empirical anchoring to break this floor.
- [ ] **dispatch_blockers (per repack_metadata, lines 9–13)**:
      `missing_contest_faithful_distortion_model`,
      `missing_scorer_basin_parity_gate` (now SATISFIED via the 2026-05-07
      basin-parity evidence — could mark as cleared in a follow-up
      metadata update), `byte_only_prediction_not_score_evidence`.
- [ ] **2026-05-07 prior dispatch attempt outcome**:
      `dispatch_outcome.md` records REFUSED at Lightning SDK
      `ApiException(400) "accelerator T4 not found for this AWS cluster"`.
      Capacity may be back; verify before retry. Override log records the
      operator-supplied 40-char reason; dispatch packet is staged.

### Dispatcher invocation snippet (Lightning T4 — preferred)

```bash
JOB="apogee-int6-cuda-anchor-$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="experiments/results/lightning_batch/${JOB}"
mkdir -p "${OUTPUT_DIR}"

.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
    --job-name "${JOB}" \
    --archive experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip \
    --repo-dir /teamspace/studios/this_studio/pact \
    --upstream-dir /teamspace/studios/this_studio/pact/upstream \
    --output-dir "/teamspace/studios/this_studio/pact/${OUTPUT_DIR}" \
    --machine T4 \
    --infer-expected-archive \
    --predicted-band 0.190 0.204 \
    --baseline-score 0.20935 \
    --baseline-archive-bytes 178873 \
    --max-runtime 5400 \
    --queue-metadata "lane_id=apogee_int6_contest_cuda_anchor" \
    --queue-metadata "subagent=CUDA-PRESTAGE-AUTHORIZED-OPERATOR" \
    --queue-metadata "basin_parity_evidence=experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json"
```

**Cost estimate (Lightning T4 g4dn.2xlarge):** ~$0.30–0.60 (~30–60 min
wall-clock for inflate + auth eval).

### Note on sanity refusal vs. operator override

The `sanity_lossy_vs_lossless` gate fails because the SHA-tied lossless
baseline is 0.2095 and the predicted_high (0.2040) sits below the
rate-distortion floor (0.3067) computed with the int4 component-penalty
proxy. The basin-parity gate (PASSED) is now an EXPLICIT
non-proxy evidence input — `predispatch_sanity.py` already accepts it via
`--readiness-evidence-json experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json`,
which made `apogee_evidence_semantics` PASS. The remaining failure is
strictly about the prediction band not exceeding the rate-distortion floor.
Operator should supply `--override-reason "..."` (≥40 chars) when invoking
the dispatcher; the override is logged to
`.omx/state/predispatch_overrides.log`.

### Cross-ref

- Dispatch claim row: `.omx/state/active_lane_dispatch_claims.md`,
  `lane_id=apogee_int6_contest_cuda_anchor`, status=`pending_authorization`.
- Basin parity: `experiments/results/apogee_int6_basin_parity_20260507_claude/`.
- Prior dispatch outcome:
  `experiments/results/lightning_batch/claude_apogee_int6_override_20260507_101520Z/dispatch_outcome.md`.
- Lane stack motivation: candidate 1 (ADMM no-dead-K) has manifest
  blocker `apogee_int6_contest_cuda_anchor_required_first`, so candidate 3
  unblocks candidate 1 immediately on landing a `[contest-CUDA]` score.

---

## Recommended dispatch order (operator advisory only)

1. **Candidate 3 (apogee_int6)** — **fire FIRST**. It is the precondition
   blocker on candidate 1, has independent CPU-build closure plus
   basin-parity evidence, and is the highest-information per dollar. ~$0.30–0.60.
2. **Candidate 1 (ADMM no-dead-K)** — fire SECOND, after candidate 3
   lands a `[contest-CUDA]` score (regardless of magnitude). The ADMM
   manifest's `apogee_int6_contest_cuda_anchor_required_first` blocker
   is satisfied by *any* contest-CUDA result on apogee_int6, not by a
   particular score band. ~$0.30–0.60.
3. **Candidate 2 (PR106 UNIWARD)** — **CPU-BUILD READY, AUTH-LOCKED** (post
   Subagent BUILD-PR106-UNIWARD-RUNTIME landing 2026-05-08). 150,511 B archive
   on disk at SHA `0641b8ac...`; 1200-frame smoke roundtrip clean; B3 verifier
   reproduces deterministically. Sanity gate refuses on
   `anchors_sufficient`/`distortion_proxy_gate` since this is the FIRST PR106
   Lagrangian calibration anchor. Operator may override with a fresh claim and
   `--override-reason "<at least 40 char approval>"`. Best fired AFTER candidate 3
   (so apogee_int6 contest-CUDA result calibrates the rate-distortion proxy for
   the operator's override decision); independent exact-eval is technically
   supported because the PR106 substrate decouples from the PR101 ADMM stack.
   ~$0.30–0.60.

Total Tier-A spend if 1+3 fire: ~$0.60–1.20.

---

## Per-CLAUDE.md compliance checklist

- [x] `forbidden_CPU_MPS_derived_dispatch_readiness_flag`: ZERO `ready_for_exact_eval_dispatch=True` flips were performed by this prestage. All three lanes remain `pending_authorization`.
- [x] `forbidden_premature_kill_without_research_exhaustion`: no KILL/FALSIFIED verdict introduced.
- [x] `Auth eval EVERYWHERE`: every dispatcher invocation routes through `scripts/launch_lightning_batch_job.py exact-eval`, which terminates with `upstream/evaluate.py` on the inflated archive bytes.
- [x] `Auth eval measurement`: `--infer-expected-archive` ensures the dispatcher locks the SHA-256 + bytes in the queue metadata before staging — any drift raises before launch.
- [x] `Strategic Secrecy Rule`: this memo is in `.omx/research/` (private; not for public disclosure).
- [x] `Race-mode rigor inversion + parallel-dispatch first`: the May 4 postmortem rule is *advisory* here — we are pre-staging, not racing. The actuator path remains `tools/parallel_dispatch_top_k.py` for K≥2 fan-out, NOT this single-shot prestage.
- [x] `Subagent commits MUST use serializer`: artifacts committed via `tools/subagent_commit_serializer.py`.
- [x] `KILL/FALSIFIED memory verdicts`: not invoked.
- [x] BUGCLASSES B6 retired-config check: explicit per-lane verification above.

---

## Cross-references

- `.omx/state/active_lane_dispatch_claims.md` (3 new `pending_authorization` rows)
- `.omx/research/lossy_coarsening_T0312_retired_config_do_not_redispatch_20260508_claude.md`
- `.omx/research/apogee_int6_predispatch_current_gate_20260507_codex.md`
- `.omx/research/apogee_int6_scorer_basin_parity_20260507_codex.md`
- `~/.claude/projects/.../memory/feedback_pr106_archive_is_monolithic_single_file_20260508.md`
- `~/.claude/projects/.../memory/feedback_pr106_cross_substrate_lagrangian_uniward_20260508.md`
- `~/.claude/projects/.../memory/feedback_may_4_hnerv_race_postmortem_20260505.md`
- `~/.claude/projects/.../memory/feedback_tier_a_cuda_dispatch_packets_prestaged_20260508.md` (this prestage)
