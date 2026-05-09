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
