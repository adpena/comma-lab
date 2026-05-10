# A1 sidecar exact-smoke + preflight speed hardening (2026-05-09)

## Scope

Codex converted recursive adversarial review findings into durable guards:

- A1 sidecar runtime readiness now has an explicit exact `inflate.sh <archive_dir> <output_dir> <file_list>` smoke producer, separate from bounded import-smoke.
- Exact smoke evidence now requires a real output digest and empty evidence-local blockers; a zero-exit `inflate.sh` that writes no raw output fails readiness.
- Red-team follow-up: exact smoke evidence now also requires the expected full
  raw byte count (`600 * 2 * 874 * 1164 * 3` by default for A1). Truncated or
  wrong-size raw output fails readiness even with exit code 0.
- Red-team follow-up: dispatch custody now binds embedded claim records to the
  live `.omx/state/active_lane_dispatch_claims.md` row by lane, job, platform,
  status, timestamp, and freshness. Terminal/latest-stale rows block readiness.
- Red-team follow-up: full-preflight clean-cache fingerprints now include the
  narrow result artifacts guarded by status/manifest/rebuild checks, and the
  MPS fallback scanner catches formatted `torch.cuda \ .is_available()` through
  the SourceIndex path.
- Developer/full clean-cache fingerprints are split: full preflight includes
  guarded result status/manifest/rebuild artifacts, while developer preflight
  ignores those full-only surfaces to preserve local edit velocity.
- A1 sidecar dispatch blockers are recomputed from structured facts instead of preserving stale provisional text blockers forever.
- Recheck telemetry distinguishes planned unproven rechecks from pairs actually rechecked this run, with a remaining-unproven counter.
- Developer/full preflight clean-cache hits run before filesystem/source-index setup, so clean repeat runs do not pay source-index construction.
- `check_no_mps_fallback_default` now caches zero-violation candidate scans with a strong stat fingerprint.

## Pre-dispatch A1 sidecar classification

Before the claim row was bound, the local sidecar archive was **locally
packet-ready but not dispatch-ready**: all packet, no-op, exact-smoke, and
strict pre-submission checks passed, and the only remaining readiness blocker
was the required live dispatch claim row.

Current byte-different archive:

- path: `experiments/results/a1_sidecar_resumable_codex_20260509T_local/submission_dir/archive.zip`
- bytes: `178316`
- sha256: `c7f3d88e1ad23bf8cda987583e702ac57e293b64bc7bfea77902e835d19cea10`
- pair records: `600/600` scalar-equivalent records present; `0` missing,
  `0` unsafe
- exact `inflate.sh` smoke: passed in `35.704488s`, wrote
  `3,662,409,600` bytes, output SHA-256
  `4a4da61a6f34075c836dc56d3f106934de4b2776237d2a714f18fa9a924aac12`
- strict pre-submission compliance preflight: passed, report SHA-256
  `eba59004ffeac1944536fe73376bf44b413aa3a1a41ef6677520a67dc90f3e91`
- runtime-tree SHA-256:
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`
- current blockers: `dispatch_claim_record_missing`

No MPS result is promoted; MPS remains advisory only for sweeps and
configuration discovery.

## Exact CPU dispatch result

After the dispatch-claim gate was satisfied, the packet was evaluated via the
fork GHA CPU path:

- workflow run:
  `https://github.com/adpena/comma_video_compression_challenge/actions/runs/25620918686`
- fork PR supplying runtime files:
  `https://github.com/adpena/comma_video_compression_challenge/pull/25`
- result path:
  `experiments/results/a1_sidecar_resampled_proxy_mse_20260510T053453Z_cpu_eval_gha/contest_auth_eval.adjudicated.json`
- result SHA-256:
  `2274dd4b5725667c8a23b8eb4707bae7cdd15b928d4c5ae7328d37a0e0bd55ca`
- report SHA-256:
  `4c39903505b9d4ae170a4e6d4bc11080c198bb948a20163115a5cd598dc440e3`
- exact `[contest-CPU GHA Linux x86_64]` score:
  `0.20962552129271272`
- components: `seg=0.00071063`, `pose=0.00003932`, `rate=0.00474933`,
  `n_samples=600`
- formula recomputation:
  `100*0.00071063 + sqrt(10*0.00003932) + 25*0.00474933 =
  0.20962552129271272`
- classification: measured implementation regression relative to A1 baseline
  `0.19284757743677347 [contest-CPU GHA Linux x86_64]`
- claim closure:
  `completed_contest_cpu_regression` in
  `.omx/state/active_lane_dispatch_claims.md`

Adversarial classification: this retires the measured
`proxy_mse`-selected per-pair latent sidecar configuration only. It does not
kill the latent-sidecar family; reactivation requires a score-domain sidecar
search objective or joint SegNet/PoseNet delta resampling that is proven against
the exact packet before dispatch.

## Verification

- `python -m pytest src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q`
  - 32 passed
- `python -m pytest src/tac/tests/test_preflight_all_clean_cache.py src/tac/tests/test_preflight_meta_bugs.py::TestNoMpsFallbackDefault src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q`
  - 52 passed
- Red-team follow-up focused tests:
  - `python -m pytest src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py src/tac/tests/test_preflight_all_clean_cache.py src/tac/tests/test_preflight_meta_bugs.py::TestNoMpsFallbackDefault -q`
  - 57 passed
  - `python -m pytest src/tac/tests/test_source_index.py src/tac/tests/test_profile_preflight_latency.py src/tac/tests/test_build_phase1_packet_compiler.py -q`
  - 99 passed
- Cache split follow-up:
  - `python -m pytest src/tac/tests/test_preflight_all_clean_cache.py src/tac/tests/test_preflight_meta_bugs.py::TestNoMpsFallbackDefault -q`
  - 23 passed
  - `python -m pytest src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py src/tac/tests/test_source_index.py src/tac/tests/test_profile_preflight_latency.py -q`
  - 61 passed
  - `python tools/profile_preflight_latency.py --surface preflight-dev-cli --top 20 --fail-on-surface-failure`
  - populate run: 9.799s
  - clean-cache repeat: 2.035s
- `python tools/profile_preflight_latency.py --surface preflight-dev-cli --top 20 --fail-on-surface-failure`
  - passed, 10.556s while the worktree was dirty
- Post-commit repeat measurement:
  - populate run: 9.826s
  - clean-cache repeat: 1.970s
- Dispatch-readiness custody exclusion follow-up:
  - `python -m pytest src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q`
  - 36 passed

## Remaining gates

- Do not redispatch this exact `proxy_mse` sidecar packet; it is a measured
  `[contest-CPU]` regression.
- Reactivate sidecar work only through a score-domain/joint SegNet+PoseNet
  sidecar search objective with no-op proof and exact packet custody.
- If a future sidecar packet is CPU-positive, run paired CUDA/T4-equivalent
  eval before any submission readiness claim.
- Keep Phase 1/T1 and Lane 12-v2 local until they emit runtime-consumed packets with no-op proof and exact custody.
