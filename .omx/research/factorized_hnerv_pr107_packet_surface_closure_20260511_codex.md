# Factorized HNeRV PR107 packet surface closure (2026-05-11)

## Scope

Operator goal: keep pushing score-lowering work while avoiding local minima,
axis conflation, and non-dispatchable research artifacts. This pass targeted
the sub-0.17 HNeRV factorization path because it is a concrete implementation
bridge from the HNeRV byte-stack roadmap into a contest packet surface.

No remote job was launched. No lane claim was opened. This is a local
byte-closed packet/materialization and custody-hardening pass only.

## Code fix

`tools/build_factorized_hnerv_archive.py` now closes the submission surface it
builds:

- copies the generated `archive.zip` into `submission_dir/archive.zip`;
- writes `submission_dir/archive_manifest.json`;
- writes `submission_dir/report.txt`;
- makes `archive_sha256` unambiguously mean the ZIP archive SHA-256;
- records the inner `0.bin` hash separately as `archive_payload_sha256`;
- preserves `score_claim=false` and `ready_for_exact_eval_dispatch=false`.

Regression test added:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_build_factorized_hnerv_archive.py \
  src/tac/tests/test_factorized_hnerv_codec.py
```

Result: `23 passed`.

## Real PR107 build artifact

Command:

```bash
.venv/bin/python tools/build_factorized_hnerv_archive.py \
  --substrate-archive experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/archive.zip \
  --output-dir experiments/results/sub017_factorized_hnerv_pr107_codex_20260511T0310Z
```

Artifact:

- submission dir:
  `experiments/results/sub017_factorized_hnerv_pr107_codex_20260511T0310Z/submission_dir`
- archive ZIP:
  `experiments/results/sub017_factorized_hnerv_pr107_codex_20260511T0310Z/submission_dir/archive.zip`
- archive ZIP SHA-256:
  `25005d3807e9efc1a63ef6514339a2f9c6681baed4d636bd719e67c6e762aba6`
- inner `0.bin` payload SHA-256:
  `b9c4b6d0d47708cd622d6150eefd57c3f709d8666da4d308ad21a393db9a3673`
- archive ZIP bytes: `158751`
- archive payload bytes: `158643`
- PR103-on-PR106 active-floor archive bytes for scale: `185578`
- rate-only delta if distortion held exactly:
  `-0.01786299813540849`

Non-final compliance:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --strict \
  --submission-dir experiments/results/sub017_factorized_hnerv_pr107_codex_20260511T0310Z/submission_dir \
  --archive experiments/results/sub017_factorized_hnerv_pr107_codex_20260511T0310Z/submission_dir/archive.zip \
  --expected-archive-sha256 25005d3807e9efc1a63ef6514339a2f9c6681baed4d636bd719e67c6e762aba6 \
  --expected-archive-size-bytes 158751 \
  --expect-single-member 0.bin \
  --json-out experiments/results/sub017_factorized_hnerv_pr107_codex_20260511T0310Z/pre_submission_compliance.nonfinal.json
```

Result: `passed=true`, with only `auth_eval_optional_missing` warning. This is
not a score claim and not an exact-ready dispatch row.

## Adversarial classification

The implementation now closes the prior export/surface bug, but the measured
configuration is still not dispatch-ready for score lowering:

- per-tensor RMS relative errors are high:
  `0.4699488021762209`, `0.504270987179695`,
  `0.49509991200402337`, `0.39598184111862805`;
- dispatch blockers remain:
  `no_cpu_eval_yet`, `no_cuda_eval_yet`,
  `rel_err_per_tensor_above_baseline_relerr_envelope`;
- exact auth eval would very likely measure a distortion regression unless a
  low-rank-trained substrate or much safer rank/error schedule is used.

Verdict: **DEFERRED-pending-low-rank-trained-substrate-or-relerr-safe-schedule**.
The lane is not killed. The useful score-lowering signal is that the packet
compiler and runtime are now contest-surface compliant enough for future
low-rank/QAT variants to materialize without manual glue.

## Follow-up manifest hardening

Codex follow-up on 2026-05-11 restored top-level charged-byte fields in the
builder output:

- `archive_bytes`
- `archive_size_bytes`
- `archive_member_manifest`

This removes an avoidable ambiguity where downstream custody tools had to infer
the charged ZIP byte count from `archive_zip_bytes` even though AGENTS.md score
math and exact-eval ledgers standardize on `archive_bytes`.

## Next score-lowering move from this lane

Do not exact-dispatch this high-error build as a primary score-lowering job.
Use the closed packet compiler to drive one of these:

1. train or fine-tune a low-rank-structured HNeRV/Ballé substrate so SVD
   factorization is no longer post-hoc destructive;
2. add a rank/error-sweep planner that selects the highest byte saving under a
   strict per-tensor rel-err cap, then materializes only rel-err-safe packets;
3. integrate factor streams with continuous-K / lossy coarsening only after
   rank selection proves reconstruction error is within the score-domain trust
   region;
4. if a future candidate clears the rel-err gate, open a fresh dispatch claim
   and run exact CPU/CUDA auth eval with raw-output custody.
