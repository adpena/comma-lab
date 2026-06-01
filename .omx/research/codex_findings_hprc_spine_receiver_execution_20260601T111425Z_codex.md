# Codex Findings: HPRC Spine Receiver Execution

Generated: 2026-06-01T11:14:25Z
Author: Codex

## Summary

Executed the selected compact-representation spine rows instead of adding more
planning surface. The receiver-proof path is now real for the HPRC spine, and
the first official-raw-output PACT repair produced a custody-valid archive under
the hard byte ceilings. The result is a useful negative: rate can be made tiny,
but the repaired PACT row is distortion-dominated and must be routed toward
decoder/base fidelity, residual value pricing, or demotion.

Implementation commit already pushed:

- `df8f33f50 Execute HPRC spine receiver proofs`

## Code Surface Landed

- `src/tac/substrates/hprc/spine_receiver_execution.py`
- `tools/execute_hprc_spine_receiver_rows.py`
- `src/tac/substrates/hprc/runtime_closure_repair.py`
- `tools/repair_embedded_runtime_zip_closure.py`
- `src/tac/substrates/hprc/spine_bounded_runner.py`

Core behavior:

- consumes `hprc_spine_bounded_runner_plan.v1`;
- dedupes selected rows by family and projection manifest;
- runs public `inflate.sh`;
- hashes receiver output;
- cleans success-only scratch;
- emits `hprc_spine_receiver_execution_report.v1` and per-row
  `hprc_spine_receiver_proof.v1`;
- keeps all local replay and receiver rows `score_claim=false`,
  `promotable=false`, and `ready_for_exact_eval_dispatch=false` until exact
  contest authority signs them.

## Receiver-Proof Results

### HNeRV / PR95

Report:

- `/Volumes/VertigoDataTier/pact/hprc_spine_receiver_execution_live_20260601T104807Z/hprc_spine_receiver_execution_report.json`

Rows:

| family | archive bytes | archive sha256 | receiver output | result |
| --- | ---: | --- | ---: | --- |
| `hnerv_packed` | 178258 | `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` | 3662409600 | receiver proof passed |
| `pr95_hnerv` | 178417 | `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a` | 3662409600 | receiver proof passed |

Both rows decode to the official raw-output size and have no receiver blockers.

### PACT-NeRV Full-Coverage Selector Rows

Report:

- `/Volumes/VertigoDataTier/pact/hprc_spine_pact_receiver_execution_live_20260601T105454Z/hprc_spine_receiver_execution_report.json`

Rows:

| archive bytes | archive sha256 | receiver output | result |
| ---: | --- | ---: | --- |
| 77546 | `986ef525c84990f661750f53b74ef22ed3c489e980a0124ee802390a208f5798` | 0 | blocked: missing runtime closure |
| 98292 | `f0bf2aae9a4bd1ec70f485959984482ba28ec0b40c814f74053a86f759c10d7b` | 157544081 | PNG-tree receiver proof passed |
| 137895 | `2b46611f35a7fa729882193bbf32b0019337f4b156ff85e884e25274695bd6e2` | 51444300 | PNG-tree receiver proof passed |

Important: PNG-tree proof is a parser/adapter proof, not contest eval
compatibility. Local replay against the official evaluator requires `0.raw`.

### Runtime-Closed Raw PACT Repair

The 77546-byte PACT row was repaired into an official raw-output adapter with
embedded runtime closure.

Artifacts:

- root:
  `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_heterogeneous_runtime_closure_repair_v4_raw_20260601Tlocal`
- archive:
  `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_heterogeneous_runtime_closure_repair_v4_raw_20260601Tlocal/archive.runtime_closed_raw.zip`
- receiver proof:
  `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_heterogeneous_runtime_closure_repair_v4_raw_20260601Tlocal/receiver_proof_raw/hprc_spine_receiver_execution_report.json`
- local replay:
  `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_heterogeneous_runtime_closure_repair_v4_raw_20260601Tlocal/local_cpu_replay/local_submission_replay_summary.json`

Archive custody:

- bytes: 106419
- sha256:
  `12e939c9942832841e4b14d3ea0fa31e0c528a04e317d08d6c72070ff2638ef9`
- receiver output: 3662409600 raw bytes
- receiver proof: passed
- scratch cleanup: success-only inflated output deleted after hashing/replay

Local replay result:

- axis: `[macOS-CPU advisory]`
- evaluation passed: true
- local score estimate: 90.86404366344534
- SegNet distortion: 0.5048244
- PoseNet distortion: 162.49560547
- rate: 0.0028344
- score authority: false
- promotion eligible: false
- exact-ready: false

## Verdict

The byte goal is not the blocker for this repaired PACT row: 106419 archive
bytes gives a tiny rate term. The blocker is distortion. This row should not
receive exact spend as-is. It should enter posterior learning as
`rate_success_distortion_failure` and drive the next compact-base work toward
measured decoder fidelity, section value pricing, and residual token admission.

The most actionable next steps are:

1. Apply the raw-output adapter to the 98292-byte int8 PACT row and run the same
   receiver proof plus local replay. It already has better PNG-tree custody and
   may be the cheapest official-format compact baseline.
2. Fill MLX full-video section-value rows for the raw-output PACT candidates:
   decoder bytes, latent bytes, selector bytes, codebook bytes, and optional
   residual tokens.
3. Admit HPRC/Z8/VQ residual tokens only when
   `delta_nonrate + rate_cost < 0` against full-video replay.
4. Keep exact CPU/CUDA spend blocked until a receiver-proven candidate clears
   local replay with credible non-rate terms.

## Verification

- `ruff check` on touched HPRC files passed.
- `pytest src/tac/substrates/hprc/tests -q` passed: 88 tests.
- Review tracker gate passed for the touched code/test surfaces.
- Commit `df8f33f50` pushed to `origin/main`.

## Dirty-Tree Discipline

The shared worktree still contains substantial unrelated partner WIP in `.omx`,
`reports`, scheduler queue files, HPRC training files, and scorer-region tests.
Those files were not staged or absorbed.
