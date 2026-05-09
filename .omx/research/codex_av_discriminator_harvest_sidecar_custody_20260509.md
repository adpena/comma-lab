# AV discriminator harvest + A1 sidecar custody check

<!-- generated_at: 2026-05-09T18:58:00Z -->
<!-- evidence_grade: partial_harvest_review; no score promotion; no remote dispatch -->

## Scope

Reviewed the active `lane_avvideodataset_cuda_path_mechanism_discriminator`
claim and the highest-EV A1 same-archive sidecar artifact before any new eval
dispatch. This is a custody/adversarial-review ledger only.

## AVVideoDataset discriminator harvest

- Active claim remains:
  `lane_avvideodataset_cuda_path_mechanism_discriminator` /
  `discriminator-sweep-20260509T110211Z`, platform
  `github_actions+lightning`, status `eval`.
- GHA run harvested:
  `https://github.com/adpena/comma_video_compression_challenge/actions/runs/25599944911`
  (`workflowName=eval`, `status=completed`, `conclusion=success`,
  `headSha=d0013db5a97066414217667e82673254eff2347d`).
- Artifact downloaded locally under ignored custody storage:
  `experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/gha_dispatch/`.
- Downloaded `archive.zip` byte-matches the local baseline archive:
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`,
  `178262` bytes.
- GHA report fields:
  - device: `cpu`
  - samples: `600`
  - PoseNet distortion: `0.00003286`
  - SegNet distortion: `0.00056023`
  - archive bytes: `178262`
  - rounded score: `0.19`
- Recomputed score from report-rounded component fields:
  `0.19284767613823797`. The stronger previously recorded exact CPU anchor
  for the same archive remains `0.19284757743677347`; the delta here is
  explained by report truncation, not score movement.

## Classification

This harvest confirms the discriminator baseline CPU control reproduces the A1
CPU anchor archive. It does **not** resolve the CUDA/CPU mechanism by itself:
the active claim covers the 4-variant CPU/CUDA discriminator family, so the
claim should stay open until the remaining variant/CUDA evidence is harvested
or explicitly terminal-classified.

No new dispatch was launched.

## A1 sidecar custody blocker

Artifact checked:
`experiments/results/a1_per_pair_latent_sidecar_resampled_20260509T103155Z/`.

Files present:

- `sidecar_manifest.json`
- `sidecar_search.log`

Files missing:

- `submission_dir/archive.zip`

Manifest hazards:

- `ready_for_exact_eval_dispatch=true` conflicts with the missing archive.
- `lane_id` is `lane_a1_inflate_time_bias_correction_sweep`, stale for the
  sidecar-resample lane.
- `score_claim=false`, `evidence_grade=predicted/proxy`,
  `search_signal=proxy_mse`, `smoke_only=true`, `n_pairs_searched=10`.

Classification: **custody bug / not dispatchable**. Do not run exact eval from
this manifest. Reactivation requires rebuilding a real byte-different
`submission_dir/archive.zip`, proving old/new archive SHA and runtime-consumed
sidecar bytes, assigning the correct lane id, and rerunning local runtime smoke
before any GHA/CUDA claim.

## Follow-up smoke sanity

Ran the current sidecar builder in smoke mode to check whether the present
working-tree tool still reproduces the stale missing-archive bug:

```bash
.venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --smoke \
  --output-dir experiments/results/a1_sidecar_codex_smoke_20260509T190000Z
```

Result:

- materialized archive:
  `experiments/results/a1_sidecar_codex_smoke_20260509T190000Z/submission_dir/archive.zip`
- new archive SHA:
  `6cca6972e3d768789f332c5bfa1c465d45a7f2787860b1aa157021faa9de6583`
- new archive bytes: `178316` (`+54` vs A1)
- no-op detector: sidecar SHA changed
- manifest remained `ready_for_exact_eval_dispatch=false`
- runtime: `174.74s` wall for 10 proxy-MSE pairs on local macOS CPU

Interpretation: the stale artifact is not dispatchable, but the current builder
does materialize a byte-different archive. The builder now has an explicit
fail-closed manifest-readiness guard so future outputs cannot claim exact-eval
readiness unless the archive exists and matches manifest SHA/size, the run is
not smoke-only, runtime smoke has passed, and the no-op detector proves sidecar
bytes changed.

## .gitignore check

The harvested GHA artifacts and `experiments/results/**/__pycache__` remain
ignored via the existing `experiments/results/` rule. `.omx/cache/` is also
ignored. No `.gitignore` change was required for this harvest.

## Next custody-safe actions

1. Harvest the remaining AV discriminator variants/CUDA artifacts and run the
   verdict analyzer only after the baseline plus all isolation rows exist.
2. Repair or explicitly quarantine the A1 sidecar builder so it cannot stamp
   `ready_for_exact_eval_dispatch=true` without a materialized archive.
3. Keep A1 as split-axis evidence: strong `[contest-CPU]` anchor, regressed
   `[contest-CUDA]` anchor, not a CUDA-ready score promotion.
4. Do not duplicate active discriminator dispatches; use
   `tools/claim_lane_dispatch.py summary` before any new remote/eval action.

## AV discriminator claim hygiene addendum

<!-- generated_at: 2026-05-09T21:53:06Z -->
<!-- evidence_grade: harvest_hygiene; no terminal claim row; no remote dispatch -->

Reviewed the active dispatch claim again. Current status:

- `tools/claim_lane_dispatch.py summary` reports exactly one active claim:
  `lane_avvideodataset_cuda_path_mechanism_discriminator` /
  `discriminator-sweep-20260509T110211Z`, platform
  `github_actions+lightning`, status `eval`.
- `gh pr list` filtered for discriminator/A1 CUDA-CPU branches returns only
  PR #24, the baseline CPU eval PR:
  `add submissions/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/ for GHA CPU eval`.
- `gh run view 25599944911` is terminal-success and belongs to the baseline
  CPU control. The report records device `cpu`, 600 samples, PoseNet
  `0.00003286`, SegNet `0.00056023`, archive bytes `178262`, rounded score
  `0.19`, artifact ID `6895314985`, and local archive SHA-256
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`.
- The nearby 2026-05-09 11:13 UTC GHA runs
  `25599734336`, `25599734618`, `25599735583`, and `25599735947` are NOT
  this lane; their logs identify `a1_segnet_boundary_smoothing_*` submissions.
- Local discriminator artifact dirs exist for all four variants, but only
  `v_baseline` has a non-empty `gha_dispatch/` artifact directory. The
  `v_loader_isolated`, `v_conv_isolated`, and `v_hydra_isolated`
  `gha_dispatch/` directories are empty.
- All four variant `archive.zip` files are byte-identical to A1:
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`.
- Local runtime-tree SHA-256s computed with
  `experiments.contest_auth_eval._runtime_dependency_manifest`:
  - `v_baseline`: `880d786594051958935826f49e7d393243bae358573a507a4ab4960c4e4e9b1a`
  - `v_loader_isolated`: `9d51d1f3f778d35bcf563dc06438c1a2532c264916de6a3390c0c0e7aecbd8d1`
  - `v_conv_isolated`: `d5cdf6e1ff5ffd1cdc04f0d09897b4f134d87148603b6a3cf6653fc5a3643d67`
  - `v_hydra_isolated`: `55da3c3453deea9c5caaaffb657b38731fdc0b595ffc5bfa48d02841de26cc23`
- No Lightning CLI binary is installed (`lightning`, `lightning-cloud`, and
  `lit` are absent). The venv can import `lightning_sdk`, but local Lightning
  state is the only safe non-dispatch source inspected here.
- `.omx/state/lightning_active_jobs.json` and
  `.omx/state/lightning_batch_jobs.json` contain no discriminator/A1 CUDA-CPU
  rows. There is no
  `experiments/results/a1_cuda_cpu_drift_discriminator_run_20260509T110211Z/`
  run-record directory.
- The checked-in runbook defaults `SKIP_CUDA=1`; its CUDA path prints
  operator-decision text rather than selecting a Lightning wrapper. There is
  no local evidence that the CUDA half of the claimed sweep was launched.

Classification: **not terminal / not closable**. The baseline CPU GHA artifact
is terminal and harvested, but it is only 1 of the 8 intended paired CPU/CUDA
cells. The combined dispatch claim must not receive a terminal claim row until
the missing variant CPU rows and CUDA rows are harvested, or the operator
explicitly chooses to close the partially executed sweep as abandoned.

Next safe non-dispatch status command:

```bash
tools/claim_lane_dispatch.py summary
gh pr view 24 --repo adpena/comma_video_compression_challenge \
  --json number,title,state,isDraft,comments,updatedAt,url
gh run view 25599944911 --repo adpena/comma_video_compression_challenge \
  --json databaseId,status,conclusion,jobs,url,updatedAt
jq 'map(select((tostring|test("discriminator|a1_cuda_cpu"; "i"))))' \
  .omx/state/lightning_active_jobs.json \
  .omx/state/lightning_batch_jobs.json
```

## AV discriminator revalidation addendum

<!-- generated_at: 2026-05-09T23:03:56Z -->
<!-- evidence_grade: harvest_revalidation; no terminal claim row; no remote dispatch -->

Revalidated the active AVVideoDataset discriminator claim from read-only
surfaces only. No code was modified and no dispatch/eval command was launched.

Exact status commands run:

```bash
.venv/bin/python tools/claim_lane_dispatch.py summary
rg -n "discriminator-sweep-20260509T110211Z|lane_avvideodataset_cuda_path_mechanism_discriminator|subagent_avvideodataset_cuda_path_mechanism_discriminator" .omx/state .omx/research experiments reports scripts tools src --glob '!experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/**'
find .omx/state .omx/research experiments reports -iname '*discriminator*' -o -iname '*avvideo*' | sort
gh pr list --repo adpena/comma_video_compression_challenge --state all --limit 100 --search 'discriminator' --json number,title,state,isDraft,updatedAt,url,headRefName
gh pr view 24 --repo adpena/comma_video_compression_challenge --json number,title,state,isDraft,comments,updatedAt,url,headRefName,headRepositoryOwner,headRefOid
gh run view 25599944911 --repo adpena/comma_video_compression_challenge --json databaseId,status,conclusion,jobs,url,updatedAt,workflowName,headSha,headBranch,event
gh run list --repo adpena/comma_video_compression_challenge --workflow eval --limit 50
gh run view 25599734336 --repo adpena/comma_video_compression_challenge --log
gh run view 25599734618 --repo adpena/comma_video_compression_challenge --log
gh run view 25599735583 --repo adpena/comma_video_compression_challenge --log
gh run view 25599735947 --repo adpena/comma_video_compression_challenge --log
find experiments/results/a1_cuda_cpu_drift_discriminator_v_*_20260509T110211Z -path '*/gha_dispatch/*' -maxdepth 4 -type f -print | sort
find experiments/results/a1_cuda_cpu_drift_discriminator_v_*_20260509T110211Z -path '*/cuda_dispatch/*' -maxdepth 4 -type f -print | sort
rg -n "discriminator|a1_cuda_cpu|AVVideoDataset|20260509T110211Z" .omx/state/lightning_active_jobs.json .omx/state/lightning_batch_jobs.json
.venv/bin/python - <<'PY'
from pathlib import Path
from experiments.contest_auth_eval import _runtime_dependency_manifest
root = Path('/Users/adpena/Projects/pact')
upstream = root / 'upstream'
for d in sorted((root/'experiments/results').glob('a1_cuda_cpu_drift_discriminator_v_*_20260509T110211Z')):
    manifest = _runtime_dependency_manifest(d/'submission_dir/inflate.sh', upstream, repo_root=root)
    archive = d/'submission_dir/archive.zip'
    import hashlib
    print(d.name, hashlib.sha256(archive.read_bytes()).hexdigest(), archive.stat().st_size, manifest['runtime_tree_sha256'])
PY
```

Claim state:

- `tools/claim_lane_dispatch.py summary` still reports exactly one active
  claim: `lane_avvideodataset_cuda_path_mechanism_discriminator` /
  `discriminator-sweep-20260509T110211Z`, platform
  `github_actions+lightning`, status `eval`.
- No terminal row for this lane/job appears in
  `.omx/state/active_lane_dispatch_claims.md`.

Rows found:

- One GHA CPU baseline control:
  - PR: `adpena/comma_video_compression_challenge#24` (open draft)
  - run: `25599944911`, workflow `eval`, status `completed`, conclusion
    `success`, event `workflow_dispatch`
  - variant:
    `a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z`
  - device: `cpu`, samples: `600`
  - PoseNet: `0.00003286`
  - SegNet: `0.00056023`
  - archive bytes: `178262`
  - archive SHA-256:
    `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
  - displayed score: `0.19`; same-archive canonical prior GHA adjudication
    recomputes to `0.19284757743677347`

Rows checked and rejected as non-discriminator:

- GHA runs `25599734336`, `25599734618`, `25599735583`, and `25599735947`
  are `a1_segnet_boundary_smoothing_*` submissions, not AV discriminator rows.
- `gh pr list --search 'discriminator'` returns only PR #24 for this lane.

Rows/artifacts missing:

- CPU GHA rows are missing for:
  - `v_loader_isolated`
  - `v_conv_isolated`
  - `v_hydra_isolated`
- CUDA rows are missing for all four variants:
  - `v_baseline`
  - `v_loader_isolated`
  - `v_conv_isolated`
  - `v_hydra_isolated`
- `cuda_dispatch/` files are absent for all variant directories.
- `experiments/results/a1_cuda_cpu_drift_discriminator_run_20260509T110211Z/`
  is absent.
- `.omx/state/lightning_active_jobs.json` and
  `.omx/state/lightning_batch_jobs.json` contain no
  `discriminator`, `a1_cuda_cpu`, `AVVideoDataset`, or
  `20260509T110211Z` rows.

Archive/runtime custody rechecked:

| Variant | archive SHA-256 | bytes | runtime tree SHA-256 |
|---|---:|---:|---|
| `v_baseline` | `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` | `178262` | `880d786594051958935826f49e7d393243bae358573a507a4ab4960c4e4e9b1a` |
| `v_loader_isolated` | `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` | `178262` | `9d51d1f3f778d35bcf563dc06438c1a2532c264916de6a3390c0c0e7aecbd8d1` |
| `v_conv_isolated` | `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` | `178262` | `d5cdf6e1ff5ffd1cdc04f0d09897b4f134d87148603b6a3cf6653fc5a3643d67` |
| `v_hydra_isolated` | `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` | `178262` | `55da3c3453deea9c5caaaffb657b38731fdc0b595ffc5bfa48d02841de26cc23` |

Classification: **not terminal / active claim cannot be closed**. The only
terminal evidence is the baseline CPU control. The claimed sweep intended 4
variants across CPU and CUDA, so the current evidence covers 1 of 8 cells and
cannot support a discriminator verdict.

Blockers:

1. Three CPU GHA isolation rows are missing.
2. All four CUDA rows are missing.
3. No run-record directory or local Lightning state row proves that the CUDA
   half was ever launched.
4. `tools/analyze_a1_cuda_cpu_drift_discriminator_verdict.py` must not be used
   for a final verdict until paired CPU/CUDA rows exist for the baseline and
   the isolation variants, or until a deliberate partial-abandonment terminal
   decision is recorded.

Reactivation / closure criteria:

- To close as completed: harvest all missing CPU/CUDA rows, recompute
  per-variant CPU/CUDA ratios, run the verdict analyzer, record the verdict
  and registry-update decision, then append a terminal claim row with the same
  `lane_id` and `instance/job_id`.
- To close as abandoned/stopped: operator must explicitly choose partial
  closure, and the terminal row should state that only the baseline CPU control
  was harvested while the three CPU isolation rows plus all CUDA rows were
  never recovered.
