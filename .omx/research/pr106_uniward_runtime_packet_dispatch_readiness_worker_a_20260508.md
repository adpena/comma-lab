# PR106 UNIWARD Runtime Packet Dispatch Readiness - Worker A - 2026-05-08

## Verdict

NO-GO for GPU dispatch of
`experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip`
as currently staged.

The archive byte custody is strong enough for a future exact-CUDA attempt after
runtime-path repair, but the current `submission_dir/inflate.sh` fails before
score under the canonical `experiments/contest_auth_eval.py` surface. The
predispatch sanity ladder also refuses normal dispatch, and a newer active
claim now exists for this lane.

No GPU job was launched by Worker A.

## Packet Custody Positives

- Repo context observed at start: `main` at
  `0cf90d2b469820e78902696c36ac335bcd2551d3`; unrelated dirty/untracked
  state already existed and was not modified.
- Archive:
  `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip`
- Bytes: `150511`
- SHA-256:
  `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`
- ZIP structure: one stored member, `0.bin`, inner payload `150403` bytes.
- Deterministic verifier:
  `.venv/bin/python tools/verify_pr106_uniward_runtime_packet_sha256.py`
  rebuilt the archive byte-identically from the committed PR106 inputs and
  build tool.

These are CPU-build/custody facts only, not score evidence.

## Blocking Findings

### 1. Staged inflate runtime is not runnable from its current path

`submission_dir/inflate.sh` is copied verbatim from the PR106
`submissions/<name>/` layout:

```bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"
...
cd "$ROOT"
python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
```

But the current packet stages it under:

```text
experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/submission_dir
```

So `ROOT` becomes `experiments/results`, `SUB_NAME` becomes
`submission_dir`, and the module target becomes
`submissions.submission_dir.inflate`. That module does not exist in the staged
runtime layout.

Canonical auth-eval proof, no GPU:

```bash
tmp=$(mktemp -d); .venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip \
  --inflate-sh experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir "$tmp" \
  --inflate-timeout 10 \
  --evaluate-timeout 10; rc=$?; rm -rf "$tmp"; exit $rc
```

Result:

```text
[contest_auth_eval] archive sha256: 0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b
[contest_auth_eval] extracted 1 member(s): ['0.bin']
[contest_auth_eval] archive members validated against whitelist
[inflate] cmd: bash .../submission_dir/inflate.sh .../extracted .../inflated .../upstream/public_test_video_names.txt
Inflating 0.mkv ... .../submission_dir/inflate.sh: line 27: python: command not found
[inflate] returncode=127 elapsed=0.0s
RuntimeError: [inflate] FAILED with returncode=127
```

Even if `.venv/bin` is prepended to `PATH`, the same canonical runner fails
before score:

```text
Inflating 0.mkv ... .venv/bin/python: Error while finding module specification for
'submissions.submission_dir.inflate' (ModuleNotFoundError: No module named 'submissions')
[inflate] returncode=1 elapsed=0.0s
RuntimeError: [inflate] FAILED with returncode=1
```

This is a hard dispatch blocker. A GPU exact-eval would spend only to fail
before scoring unless the runtime is staged under a real `submissions/<name>/`
package or `inflate.sh` is adapted to be self-contained for the
`experiments/results/.../submission_dir` layout.

### 2. Predispatch sanity refuses normal dispatch

Command:

```bash
.venv/bin/python tools/predispatch_sanity.py \
  --archive experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip \
  --predicted-low 0.18 \
  --predicted-high 0.22 \
  --rel-err-pct 4.6567025 \
  --lane-class pr106_lagrangian_per_tensor \
  --json
```

Exit code: `64`.

Refusal reasons:

```text
anchors_sufficient: only 0 calibration anchors at
.omx/calibration/anchors_pr106_lagrangian_per_tensor.json (need >=3).

distortion_model_gate: rel_err 4.66% > 1.0% but --distortion-proxy-ran not set.
Run experiments/distortion_proxy_local.py against the archive first; then attach
non-proxy distortion/parity evidence for the exact candidate bytes.
```

Passing gates in the same run:

```text
hazard_scan: 0 dispatch_local_path_leak / remote_script_local_pythonpath_leak hazards
lane_registry_consistent: lane_maturity validate clean
apogee_evidence_semantics: not applicable
```

This may be overrideable later as a first calibration dispatch, but it is not a
normal unblocked GO.

### 3. Current claim ledger has an active same-lane claim

After Worker A's initial dry-run claim probe, another agent wrote:

```text
2026-05-08T08:24:50Z | codex:gpt-5 | pr106_uniward_lagrangian_runtime_packet |
lightning | pr106-uniward-rms005-exact-20260508T082438Z | ... |
active_staging
```

Worker A did not write that row and did not edit `.omx/state`.

Read-only conflict probe:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id pr106_uniward_lagrangian_runtime_packet \
  --platform lightning \
  --instance-job-id DRYRUN:pr106-uniward-rms005-conflict-check \
  --agent codex:gpt-5.5-worker-a \
  --status active_exact_eval_prepare \
  --notes 'dry-run conflict probe only; no GPU launch' \
  --dry-run
```

Exit code: `3`.

Output:

```text
REFUSING_DISPATCH: active claim(s) already exist for lane_id=pr106_uniward_lagrangian_runtime_packet
  2026-05-08T08:24:50Z codex:gpt-5 job=pr106-uniward-rms005-exact-20260508T082438Z status=active_staging
```

No additional dispatch is allowed until that claim is terminal or explicitly
coordinated.

### 4. Strict pre-submission packet check is not release-clean

Command:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/submission_dir \
  --archive experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip \
  --expect-single-member 0.bin \
  --expected-archive-sha256 0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b \
  --expected-archive-size-bytes 150511 \
  --strict
```

Exit code: `1`.

Relevant failures:

```text
required_file_present:archive.zip failed inside submission_dir
required_file_present:report.txt failed
report_exists failed
```

Relevant passes:

```text
inflate_sh_executable passed
archive_exists passed
zip_member_safe:0.bin passed
zip_local_header_matches:0.bin passed
zip_no_duplicate_members passed
zip_expected_single_member passed
expected_archive_sha256_matches passed
expected_archive_size_bytes_matches passed
submission_runtime_manifest_computable passed
runtime_tree_sha256=667c54d19b5d8bd591cbdefeb4a7d24eb305039650aa2284213cde1240492be8
```

This check is stricter than an exact-eval launcher invocation because it expects
a release-style `submission_dir` with `archive.zip` and `report.txt`, but the
failure is still relevant: this is not a contest-final packet surface.

### 5. Verified launcher surface requires more than the existing snippets

Actual argparse/help surfaces were checked. For `scripts/launch_lightning_batch_job.py exact-eval`:

- `--adjudicate` is mandatory; the command exits if absent.
- Non-dry-run Studio exact-eval requires a `--source-manifest` from
  `scripts/lightning_repro_workspace.py`.
- Non-dry-run Studio exact-eval requires a matching active dispatch claim via
  `--dispatch-lane-id` or `--queue-metadata lane_id=...`.
- Studio submit requires explicit `--studio`, `--teamspace`, and `--user` or
  `--org`.
- Studio submit requires `--remote-preflight-ssh-target` unless a break-glass
  skip reason is supplied.
- T4/g4dn exact-eval requires an inflate-side torch pin through
  `--env INFLATE_TORCH_SPEC=...`; if using a cu124 pin it also requires
  `--env UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124` and
  `--env UV_INDEX_STRATEGY=unsafe-best-match`.

No `source-manifest` for this packet was present under
`experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/`.

## Commands Verified From Help/Argparse

```bash
.venv/bin/python tools/claim_lane_dispatch.py --help
.venv/bin/python tools/claim_lane_dispatch.py claim --help
.venv/bin/python scripts/launch_lightning_batch_job.py --help
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval --help
.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts --help
.venv/bin/python scripts/launch_lightning_batch_job.py harvest-ssh --help
.venv/bin/python scripts/lightning_repro_workspace.py --help
.venv/bin/python experiments/contest_auth_eval.py --help
.venv/bin/python scripts/pre_submission_compliance_check.py --help
.venv/bin/python tools/predispatch_sanity.py --help
```

## Terminal Claim Closure Commands

Worker A did not edit `.omx/state/active_lane_dispatch_claims.md`. If the
current `active_staging` row has not submitted a GPU job, close it before any
future relaunch with a terminal row using the same lane id and job id:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id pr106_uniward_lagrangian_runtime_packet \
  --platform lightning \
  --instance-job-id pr106-uniward-rms005-exact-20260508T082438Z \
  --agent codex:gpt-5.5-worker-a \
  --status refused_dispatch_runtime_contract_no_gpu_launch \
  --notes "NO-GO Worker A: staged inflate.sh fails canonical auth-eval before score; no GPU launch should proceed until runtime path and sanity blockers are fixed" \
  --force
```

If a GPU job was already submitted under that claim, use a factual terminal
status instead, for example:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id pr106_uniward_lagrangian_runtime_packet \
  --platform lightning \
  --instance-job-id pr106-uniward-rms005-exact-20260508T082438Z \
  --agent <agent> \
  --status stopped_runtime_contract_no_score \
  --notes "Stopped after Worker A NO-GO: staged inflate.sh fails before score; preserve artifacts/logs if any" \
  --force
```

For a completed future fixed job, close with the actual outcome:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id pr106_uniward_lagrangian_runtime_packet \
  --platform lightning \
  --instance-job-id <exact-job-name> \
  --agent <agent> \
  --status completed_score_<score_token> \
  --notes "contest_auth_eval.adjudicated.json=<path> archive_sha256=<sha> bytes=<bytes>" \
  --force
```

or:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id pr106_uniward_lagrangian_runtime_packet \
  --platform lightning \
  --instance-job-id <exact-job-name> \
  --agent <agent> \
  --status failed_<precise_failure_class> \
  --notes "failure_class=<class> artifact=<path> no score claim" \
  --force
```

## Required Remediation Before GO

1. Repair the runtime path:
   - either stage the packet under a real repo package such as
     `submissions/pr106_uniward_lagrangian_runtime_packet/`, with
     `archive.zip` and `report.txt` if preparing a release surface;
   - or adapt `inflate.sh` so it runs from
     `experiments/results/.../submission_dir` without relying on
     `submissions.${SUB_NAME}` or bare `python`.
2. Re-run canonical local no-GPU auth-eval to prove Stage 2 starts from the
   same `--inflate-sh` path intended for Lightning. It may still time out on
   CPU later, but it must not fail immediately on interpreter or module
   resolution.
3. Re-run `tools/verify_pr106_uniward_runtime_packet_sha256.py` if archive bytes
   are rebuilt or moved.
4. Re-run `tools/predispatch_sanity.py`. Normal GO requires passing gates; a
   first-calibration dispatch requires an explicit operator override reason and
   should be labeled as calibration, not promotion.
5. Generate a `scripts/lightning_repro_workspace.py` source manifest that
   includes the exact archive and inflate runtime closure.
6. Create or reuse exactly one active claim with the exact future job name.
7. Use the actual `launch_lightning_batch_job.py exact-eval` surface with
   `--adjudicate`, source manifest, dispatch lane id, remote preflight, explicit
   Lightning identity, and T4 torch env pin.

