# PR106 Sidechannel Stack Readiness - 2026-05-07

This records a local-only stack-planning artifact for PR106 sidechannels. It is
not a score claim, not dispatch approval, and not a lane claim.

## Artifact

- stack readiness JSON:
  `experiments/results/pr106_sidechannel_stack_readiness_20260507_codex/stack_readiness.json`
- sha256:
  `b7fa48e2c4239e69aae630fde391c2bbe6142fb18a2cb33af0d4760f7e050480`
- schema: `pr106_sidechannel_stack_readiness.v1`
- `score_claim=false`
- `dispatch_attempted=false`
- `remote_gpu_run=false`
- `ready_for_local_stack_planning=true`
- `ready_for_exact_eval_dispatch=false`

## Atom Ledger Summary

The artifact converts existing latent/yshift/LRL1/wavelet/WR01 surfaces into a
single meta-Lagrangian atom ledger with six rows:

| atom | byte delta | expected total score delta | stack-review ready |
| --- | ---: | ---: | --- |
| `pr106_sidechannel_stack:wr01_exact_eval_packet` | `-9` | `-0.000005992731` | yes |
| `pr106_sidechannel_stack:latent` | `+23` | `+0.000015314756` | no |
| `pr106_sidechannel_stack:yshift` | `+44` | `+0.000029297794` | no |
| `pr106_sidechannel_stack:lrl1` | `+50` | `+0.000033292948` | no |
| `pr106_sidechannel_stack:three_sister_stack` | `+109` | `+0.000072578626` | no |
| `pr106_sidechannel_stack:wavelet_noop_stack` | `+387` | `+0.000257687415` | no |

WR01 is the only row marked stack-review-ready because it has a byte-closed
release-surface archive manifest. It remains not dispatchable because it lacks
candidate-specific exact CUDA auth eval and KKT/component-response proof. The
other rows are proxy/planning rows and explicitly require byte-closed manifests
plus exact CUDA evidence before stack promotion.

## Current Blockers

- `stack_readiness_is_local_planning_only`
- `requires_exact_cuda_auth_eval_before_score_claim`
- `requires_no_dispatch_lane_claim_before_remote_submit`
- `latent_exact_cuda_artifact_missing`
- `yshift_exact_cuda_artifact_missing`
- `lrl1_exact_cuda_artifact_missing`
- `wr01_exact_eval_packet_exact_cuda_artifact_missing`
- `wavelet_gate:requires_reviewed_wr01_apply_transform`
- `wavelet_gate:requires_component_benefit_evidence_over_break_even`
- `wavelet_gate:requires_archive_manifest_preflight`
- `wavelet_gate:requires_exact_cuda_auth_eval`
- `wavelet_gate:wr01_rate_penalty_must_be_recovered_by_distortion`
- `wavelet_gate:wr01_runtime_mode_is_explicit_noop`

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_pr106_sidechannel_stack_readiness.py -q
.venv/bin/ruff check src/tac/pr106_sidechannel_stack_readiness.py tools/build_pr106_sidechannel_stack_readiness.py src/tac/tests/test_pr106_sidechannel_stack_readiness.py
.venv/bin/python tools/build_pr106_sidechannel_stack_readiness.py --json-out experiments/results/pr106_sidechannel_stack_readiness_20260507_codex/stack_readiness.json
.venv/bin/python tools/build_pr106_sidechannel_stack_readiness.py --fail-if-dispatch-ready --json-out /tmp/pr106_stack_readiness_verify.json
.venv/bin/python tools/dispatch_dryrun_pr106_sidechannels.py --json --skip-help-subprocess >/tmp/pr106_sidechannel_dryrun_after_stack_readiness.json
```

Results:

- pytest: `3 passed`
- ruff: `All checks passed`
- builder: wrote the stack readiness JSON without dispatching or claiming a lane
- fail-if-dispatch-ready verifier: exited `0` with
  `ready_for_exact_eval_dispatch=false`, `atom_count=6`, `blocker_count=17`
- PR106 sidechannel dry-run: `ok=true`, `ready_for_local_readiness=true`,
  `blockers=[]`, `failed_check_count=0`, `check_count=25`

## Next Action

Keep this artifact as the local stack scheduler input. Before any exact-eval
dispatch exists, the next local improvement should attach byte-closed manifests
for the latent/yshift/LRL1 sister archives or promote a reviewed WR01 apply
transform with component-response evidence over the existing break-even gate.
