# A1 sidecar custody fail-closed fix (2026-05-09)

<!-- generated_at: 2026-05-09T22:06:00Z -->
<!-- research_only=true -->

## Classification

This landing repairs the A1 per-pair latent sidecar builder's manifest
custody contract. It is not a score claim and did not dispatch remote, GPU, or
exact-eval work.

## Fixed guard

`tools/build_a1_per_pair_latent_correction_sidecar.py` now refuses
`ready_for_exact_eval_dispatch=true` unless the manifest has all of:

- sidecar-specific `lane_id=lane_a1_per_pair_latent_sidecar_resampled`;
- canonical archive path aliases pointing at the materialized archive;
- archive bytes and SHA-256 matching the file on disk;
- `local_runtime_custody` / `runtime_manifest` with `runtime_tree_sha256`;
- matching local runtime file SHA-256s for `inflate.py`, `inflate.sh`,
  `src/codec.py`, and `src/model.py`;
- executable `inflate.sh`;
- non-smoke status, runtime smoke checked, and sidecar no-op proof.

Focused tests cover the old failure mode: missing archive materialization,
wrong inherited lane id, missing archive path, complete custody success, and
runtime-tree drift.

## Regenerated local artifact

Command:

```bash
.venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
  --smoke \
  --output-dir experiments/results/a1_per_pair_latent_sidecar_resampled_20260509T103155Z
```

Result:

- Manifest:
  `experiments/results/a1_per_pair_latent_sidecar_resampled_20260509T103155Z/sidecar_manifest.json`
- Archive:
  `experiments/results/a1_per_pair_latent_sidecar_resampled_20260509T103155Z/submission_dir/archive.zip`
- Archive bytes: `178316`
- Archive SHA-256:
  `6cca6972e3d768789f332c5bfa1c465d45a7f2787860b1aa157021faa9de6583`
- Runtime-tree SHA-256:
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`
- Old sidecar bytes/SHA-256:
  `607` /
  `6c2946e323bbbc6f8d906ef6c68989e8acbd8d60332c87da8fe8147f1ea7b12f`
- New sidecar bytes/SHA-256:
  `661` /
  `e9eb2aa828ca3168e3162f7d07ffcf2715cc6373c89f6a5ac74a7ca023f41fec`
- `ready_for_exact_eval_dispatch=false`
- Current blockers:
  `claim lane before any GHA/remote eval dispatch`,
  `run exact-eval dispatcher preflight against submission_dir`,
  `record runtime tree SHA and terminal dispatch claim row`,
  `smoke_only_not_exact_eval_ready`,
  `runtime_smoke_not_checked`

## Reactivation Criteria

Promote only after all of the following are true:

1. Rebuild a non-smoke sidecar candidate or explicitly record operator approval
   that the 10-pair smoke artifact should be tested as a custody probe.
2. Run local runtime smoke on the exact `submission_dir` and stamp
   `runtime_smoke_checked=true` with an evidence path.
3. Preserve the archive path, bytes, SHA-256, sidecar old/new SHA-256s,
   local runtime custody, and runtime-tree SHA-256.
4. Run the exact-eval dispatcher preflight against the submission directory.
5. Claim `lane_a1_per_pair_latent_sidecar_resampled` before any GHA, remote,
   GPU, or exact-eval job.
6. Evaluate CPU GHA first; only consider CUDA dispatch after CPU evidence is
   positive and the dispatch claim is fresh.

## Recursive hardening addendum

<!-- generated_at: 2026-05-09T22:42:00Z -->

Fresh-eyes adversarial review found three residual false-green risks in the
initial custody repair. They are now guarded in code and tests:

- Runtime custody now recursively manifests the full runtime tree, excluding
  only `archive.zip`, `__pycache__`, `.pyc`, and `.pyo`. Extra runtime files
  such as `src/helper.py` or `weights.pt` are included in the runtime-tree
  SHA-256, and stale/missing/extra file records block readiness.
- Sidecar no-op proof now requires valid unequal
  `old_inner_sidecar_sha256` and `new_inner_sidecar_sha256`, not just
  `sidecar_changed=true`.
- `runtime_smoke_checked=true` is no longer sufficient alone. A smoke evidence
  object must bind command, exit code, archive SHA-256, runtime-tree SHA-256,
  and optional output digest.

Focused verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q
```

Result: `18 passed`.

## Local runtime smoke addendum

<!-- generated_at: 2026-05-09T22:58:00Z -->

Ran an exact `inflate.sh <archive_dir> <output_dir> <file_list>` runtime smoke
against the rebuilt smoke artifact without scorer/eval/GPU dispatch:

```bash
SUB=experiments/results/a1_per_pair_latent_sidecar_resampled_20260509T103155Z/submission_dir
TMP=$(mktemp -d /tmp/a1_sidecar_smoke.XXXXXX)
mkdir -p "$TMP/data" "$TMP/out"
unzip -p "$SUB/archive.zip" x > "$TMP/data/x"
printf '0.mkv\n' > "$TMP/file_list.txt"
PYTHON=.venv/bin/python "$SUB/inflate.sh" "$TMP/data" "$TMP/out" "$TMP/file_list.txt"
```

Evidence:

- Archive SHA-256:
  `6cca6972e3d768789f332c5bfa1c465d45a7f2787860b1aa157021faa9de6583`
- Archive bytes: `178316`
- Runtime-tree SHA-256:
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`
- Output: `1200` frames, `3662409600` bytes
- Output SHA-256:
  `6783b143af4d1f95a68409add46e5b1cebe7dfbe494f3a48dbfdbfef8a2d15cb`
- Temporary raw output directory was removed after hashing.

Classification remains **not exact-eval dispatch-ready** because the artifact
is still a 10-pair smoke candidate and the committed manifest remains in
ignored custody storage, not a non-smoke full 600-pair candidate. The runtime
closure itself is now proven for this smoke packet.

## Solver Wire-In

- Sensitivity-map contribution: N/A — no empirical exact-eval anchor landed.
- Pareto constraint: N/A — smoke-only local custody artifact, no score claim.
- Bit-allocator hook: N/A — sidecar bytes changed but no exact component
  response yet.
- Cathedral autopilot dispatch hook: blocked by `ready_for_exact_eval_dispatch=false`.
- Continual-learning posterior update: not run; no authoritative tag.
- Probe-disambiguator: N/A — no competing defensible interpretation in this
  fix; the manifest either matches file/runtime custody or fails closed.
