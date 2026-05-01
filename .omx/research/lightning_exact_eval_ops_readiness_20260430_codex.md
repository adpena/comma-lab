# Lightning Exact-Eval Ops Readiness - OWV3 Byte-Feasible - 2026-04-30

Agent: Codex
Scope: Lightning exact-eval ops stream for the corrected OWV3 byte-feasible
archive. This report contains no score claims and records no new score
evidence. No spendful Lightning job was submitted. No runtime code was edited.

## Files And Ledgers Inspected

- `scripts/launch_lightning_batch_job.py`
- `src/tac/deploy/lightning/batch_jobs.py`
- `.omx/state/lightning_batch_jobs.json`
- `docs/runbooks/lightning_exact_eval.md`
- `.omx/research/exact_eval_queue_ops_20260430_agent.md`
- `.omx/research/lightning_pypi_compromise_security_review_20260430_codex.md`
- `.omx/state/owv3_byte_feasible_repro_20260430_r1_manifest.json`

## Target Custody

Target archive:

```text
experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip
```

Local identity verified on 2026-04-30:

```text
sha256=e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec
bytes=686557
```

Read-only Lightning SSH check verified the same remote archive identity at the
same relative path under `/teamspace/studios/this_studio/pact`.

## Lightning State

The latest Lightning Batch Jobs entry is a corrected dry-run, not a submitted
job:

```text
recorded_at_utc=2026-04-30T19:38:21Z
status=DRY_RUN
job_name=owv3_byte_feasible_exact_cuda_20260430_codex_studio_pact_dryrun
machine=T4
studio=pact
expected_archive_sha256=e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec
expected_archive_size_bytes=686557
command_sha256=45456318dccbd437e02c4446f7339ad66aaa4e79668c0e69d4707b39c506358f
local_artifact_dir=experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_studio_pact
```

This supersedes the older dry-run that had `spec.studio=null`. There is still
no submitted Lightning job id, no SDK artifact path, and no harvested artifact
directory for this OWV3 exact eval.

## Current Preflight Results

Local strict Lightning supply-chain scan passed:

```text
recorded_at_utc=2026-04-30T19:45:05Z
status=OK
violation_count=0
python=/Users/adpena/Projects/pact/.venv/bin/python
lightning=null
pytorch-lightning=null
lightning-sdk=2026.4.10
```

Read-only Lightning SSH strict scan passed:

```text
recorded_at_utc=2026-04-30T19:45:31Z
status=OK
violation_count=0
python=/teamspace/studios/this_studio/pact_pfp16_exact_20260430T1625Z/.venv/bin/python
lightning=null
pytorch-lightning=null
lightning-sdk=null
```

The current interactive Lightning SSH shell is still not an exact-eval runner:

```text
nvidia-smi=missing
torch_version=2.11.0+cu130
cuda_available=false
cuda_device_count=0
```

Therefore the safe execution target is a Lightning Batch Job T4 worker, not the
current SSH shell.

## Launcher Assessment

The current Batch Jobs helper has strong archive/eval custody controls:

- exact-eval specs are non-interruptible and `reuse_snapshot=false`;
- the generated command uses `set -euo pipefail`;
- it copies the archive into the job artifact directory as `archive.zip`;
- it verifies archive SHA-256 and byte count before eval;
- it runs `experiments/contest_auth_eval.py --device cuda`;
- it asserts `n_samples=600`, `provenance.device=cuda`, finite recomputed score,
  and expected archive SHA/bytes from `contest_auth_eval.json`;
- it runs JSON-based adjudication and component gates when `--adjudicate` is
  supplied.

Missing runner preflights block a strict "safe to submit now" answer:

- the generated Batch Job command does not run
  `scripts/scan_lightning_supply_chain.py --strict --quiet` inside the actual
  Batch worker before importing the eval path;
- it does not emit a runner preflight JSON artifact for `nvidia-smi`, Torch CUDA
  availability, CUDA device count, or T4 identity before spending eval time;
- artifact validation records `gpu_t4_match`, but does not fail closed when a
  T4 evidence path produces `gpu_t4_match != true`.

Conclusion: do not submit this OWV3 exact eval as "safely submitted now" under
the strict Lightning runner-preflight standard. The corrected dry-run is close,
but the actual Batch worker remains unverified until the job command performs
its own supply-chain and CUDA/T4 preflight or an operator explicitly accepts
that residual risk.

## Exact Patch Recommendations

Do not edit runtime code in this turn. Recommended hardening patch:

1. In `src/tac/deploy/lightning/batch_jobs.py`, add artifact constants:

```python
ARTIFACT_SUPPLY_CHAIN_SCAN = "lightning_supply_chain_scan.json"
ARTIFACT_RUNNER_PREFLIGHT = "lightning_runner_preflight.json"
```

2. Add `_runner_preflight_command(python_bin, output_dir, require_t4)` and call
   it in `exact_cuda_eval_command` immediately after `mkdir -p {out}` and before
   metadata/archive copy. The command should:

```bash
<python> scripts/scan_lightning_supply_chain.py --strict --quiet \
  | tee <out>/lightning_supply_chain_scan.json
command -v nvidia-smi
<python> - <out>/lightning_runner_preflight.json <require_t4> <<'PY'
import json, subprocess, sys, torch, time
out = sys.argv[1]
require_t4 = sys.argv[2] == "1"
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    text=True,
).strip().splitlines()[0]
payload = {
    "schema_version": 1,
    "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "gpu_name": gpu_name,
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "require_t4": require_t4,
}
if not payload["cuda_available"] or payload["cuda_device_count"] < 1:
    raise SystemExit("FATAL: Lightning Batch runner has no Torch CUDA device")
if require_t4 and "T4" not in gpu_name:
    raise SystemExit(f"FATAL: Lightning Batch runner is not T4: {gpu_name}")
open(out, "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
```

3. Thread a `require_t4` boolean through `exact_cuda_eval_command` and
   `make_exact_eval_spec`, defaulting to true when `machine.upper() == "T4"`.
   Include it in queue metadata and `LightningBatchJobSpec.asdict()`.

4. In the post-eval JSON assertion block, fail closed when `require_t4` is true
   and `prov.get("gpu_t4_match") is not True`.

5. In `validate_local_artifact_dir`, `mirror_local_artifact_dir`, and
   `harvest_local_artifacts`, add `require_t4: bool = False`. When true, require:

```python
if provenance.get("gpu_t4_match") is not True:
    raise ValueError("contest_auth_eval provenance.gpu_t4_match is not true")
```

6. Add CLI flags to `validate-artifacts` and `harvest-local`:

```text
--require-t4
```

7. Add tests in `src/tac/tests/test_lightning_batch_jobs.py` covering:

- exact-eval command contains strict supply-chain scan and runner preflight
  artifact names;
- T4 machine specs set `require_t4=true`;
- validation with `--require-t4` rejects `gpu_t4_match=false`;
- validation with `--require-t4` accepts `gpu_t4_match=true`.

## Verification Run This Turn

Passed:

```bash
.venv/bin/python -m py_compile \
  scripts/launch_lightning_batch_job.py \
  src/tac/deploy/lightning/batch_jobs.py \
  scripts/scan_lightning_supply_chain.py \
  scripts/adjudicate_contest_auth_eval.py

.venv/bin/python -m pytest \
  src/tac/tests/test_lightning_batch_jobs.py \
  src/tac/tests/test_lightning_supply_chain_scan.py \
  -q
# 18 passed

git diff --check -- \
  scripts/launch_lightning_batch_job.py \
  src/tac/deploy/lightning/batch_jobs.py \
  scripts/scan_lightning_supply_chain.py \
  scripts/adjudicate_contest_auth_eval.py \
  docs/runbooks/lightning_exact_eval.md \
  .omx/state/lightning_batch_jobs.json
```

## Next Safe Action

Land the runner-preflight hardening above, run a fresh corrected dry-run, then
submit the same spec without `--dry-run` only after local and remote strict
supply-chain scans still pass and the dry-run command includes the in-job
preflight artifacts.

## 2026-04-30T23:17Z Worker E Component-Gate Exit Semantics Audit

Scope: read-only review of `src/tac/deploy/lightning/batch_jobs.py` and
`scripts/launch_lightning_batch_job.py` after `harvest-ssh` landed. No runtime
code was edited in this audit.

Current behavior:

- `exact_cuda_eval_command()` runs under `set -euo pipefail`.
- `_adjudication_command()` pipes `scripts/adjudicate_contest_auth_eval.py`
  through `tee`, so the adjudicator exit code controls the SDK job result.
- `scripts/adjudicate_contest_auth_eval.py` writes
  `adjudication_provenance.json` and `contest_auth_eval.adjudicated.json`, then
  returns `2` only when `component_gate_triggered=true`.
- `validate_local_artifact_dir()` already treats this as valid forensic
  evidence: it returns `promotion_eligible=false`,
  `adjudication_lane_status=COMPONENT_GATE_REVIEW_REQUIRED`, and preserves the
  component-gate violation details.
- `harvest_ssh_artifacts()` can mirror failed SDK jobs successfully by deriving
  the SDK-persisted Studio artifact path and copying only canonical artifacts.

Recommendation:

Component-gate-only outcomes should not keep Lightning SDK jobs `Failed` once
the CUDA eval, archive identity checks, runner preflights, adjudication JSON,
and artifact copies completed. They should be successful Batch Jobs that
produce forensic, non-promotable evidence. SDK `Failed` should mean an
operational or custody failure: missing CUDA, wrong sample count, archive
identity mismatch, missing JSON, malformed adjudication, missing artifacts,
supply-chain/preflight failure, timeout, or crash.

Rationale:

- A component gate is a scientific/adjudication result, not an execution
  failure. The current behavior conflates "measured implementation failed a
  predeclared component constraint" with "Batch Job failed to produce valid
  evidence".
- The current adjudicator already returns success for score-regression review
  when no component gate fires; component gates should use the same evidence
  semantics while still marking promotion ineligible.
- `harvest-ssh` plus artifact validation now gives enough custody to preserve
  non-promotable failures without depending on a red SDK job state.
- Keeping gate-only jobs failed encourages duplicate retries and stale queue
  alarms even when the result is complete and scientifically useful.

Concrete patch plan:

1. Add an explicit mode to `scripts/adjudicate_contest_auth_eval.py`, for
   example `--allow-component-gate-forensic-success`.
   - Default behavior stays fail-closed and returns `2` on component gates, so
     existing remote promotion scripts do not silently change semantics.
   - With the new flag, component-gate violations still write provenance,
     print the violation summary, set `component_gate_triggered=true`, and
     return `0`.

2. Thread the new adjudicator flag through Lightning exact-eval command
   generation in `src/tac/deploy/lightning/batch_jobs.py`.
   - Add a `LightningAdjudicationSpec` field such as
     `allow_component_gate_forensic_success: bool = True` for Lightning Batch
     exact-eval specs.
   - Include the field in `asdict()`/metadata so the artifact records the exit
     policy.
   - Have `_adjudication_command()` append
     `--allow-component-gate-forensic-success` when the field is true.
   - Have `LightningBatchJobSpec.validate()` require that exact CUDA eval
     commands contain the flag, or explicitly document any opt-out.

3. Keep artifact validation promotion semantics unchanged.
   - `validate_local_artifact_dir()` should continue accepting adjudicated
     component-gate artifacts and returning `promotion_eligible=false`.
   - Do not weaken checks for CUDA device, sample count, archive SHA/bytes,
     supply-chain scans, DALI/bootstrap, runner preflight, or adjudicated JSON
     equality.

4. Add focused tests.
   - New adjudicator CLI test: component gate plus default mode returns `2` and
     writes provenance/result-copy artifacts.
   - New adjudicator CLI test: component gate plus forensic-success flag
     returns `0`, writes identical artifacts, and records
     `component_gate_triggered=true`.
   - Lightning command-generation test asserts
     `--allow-component-gate-forensic-success` is present in exact-eval jobs.
   - Existing validation test
     `test_validate_local_artifact_dir_reports_failed_component_gate_as_non_promotable`
     already covers the harvest-side non-promotable result and should remain.

Patch risk:

- Low if the default adjudicator behavior remains unchanged and only Lightning
  Batch exact-eval opts into forensic-success exit semantics.
- Medium if the adjudicator default changes globally, because remote lane
  scripts using strict shell mode may currently rely on exit `2` to stop
  promotion flows.
