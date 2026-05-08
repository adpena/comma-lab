# Lightning Blocker And A2 Packet-Parity Closure - 2026-05-08

Owner: `codex`
Branch: `main`
Dispatch performed: `false`
Score claim: `false`
Promotion/rank/kill claim: `false`

## Executive State

Two concrete outcomes landed in this pass:

1. Lightning exact-CUDA dispatch is currently blocked from this machine before
   job submission. The canonical flags and lane candidate are known, but the
   Studio shell path is unavailable and the account balance is insufficient to
   start the stopped Studio.
2. A2 packet-ladder local inflate parity now exercises the real contest
   3-argument `inflate.sh <data_dir> <output_dir> <file_list>` contract and
   passes on the latest packet artifact. This clears the packet-local
   inflate-parity integration blocker for this artifact only. It does not
   clear the sensitivity/proxy/eval blockers.

## Lightning Dispatch Blocker

Candidate intended for exact CUDA:

- Lane: `cross_paradigm_admm_continuous_k_plus_op1_finalizer`
- Archive:
  `experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/archive.zip`
- Archive bytes: `153513`
- Archive SHA-256:
  `7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897`
- Dispatch status: not submitted; previous claim closed as
  `failed_presubmit_ssh_publickey`

Observed blockers:

- `lightning-pact` resolves to the expected `ssh.lightning.ai` endpoint and
  local key files exist, but SSH returns `Permission denied (publickey)`.
- `scripts/launch_lightning_batch_job.py doctor` passed local supply-chain and
  machine inventory, but failed `ssh_auth` and `remote_supply_chain`.
- `lightning_sdk.Studio('lossy-compression-challenge', teamspace='comma-lab',
  user='adpena', create_ok=False)` reported `Stopped`.
- Attempting to start the Studio through the SDK failed before a shell became
  available:
  `insufficient balance to start the cloud space, top up and try again`.

Conclusion:

This is not a CLI-flag problem and not a packet-compliance problem. The current
blocking class is provider/account state: Lightning shell authorization plus
account balance. Do not relaunch the lane until both checks pass:

```bash
ssh -o BatchMode=yes lightning-pact true
LIGHTNING_DISABLE_VERSION_CHECK=1 .venv/bin/python - <<'PY'
from lightning_sdk import Studio
s = Studio('lossy-compression-challenge', teamspace='comma-lab', user='adpena', create_ok=False)
print({'status': str(s.status), 'machine': str(s.machine), 'cloud_account': s.cloud_account})
PY
```

After those pass, create a fresh Level-2 claim and dispatch through
`scripts/lightning_exact_eval_repro.py` with `--remote lightning-pact`,
`--stage-workspace`, `--submit`, `--studio lossy-compression-challenge`,
`--teamspace comma-lab`, `--sdk-user adpena`, and `--machine g4dn.2xlarge`.

## A2 Packet-Parity Closure

Tool hardened:

- `tools/build_a2_sensitivity_weighted_pr101_packet.py`

Bug fixed:

- The local parity verifier previously invoked packet-local `inflate.sh` with
  zero arguments. That tested neither the contest contract nor the packet's
  real runtime path and produced false parity failures.
- The verifier now extracts source and candidate archives into per-run
  `data/` directories, writes a `file_list.txt`, invokes
  `inflate.sh <data_dir> <output_dir> <file_list>`, and compares the output
  directory contract.
- The verifier also prepends the active `.venv/bin` path so packet runtimes
  that call `python` see the same dependency environment used by local tooling.

Latest packet artifact:

- Manifest:
  `experiments/results/track1_phase_a2_packet_ladder_codex_fixed_20260508T203813Z/a2_packet_ladder_manifest.json`
- Variant:
  `weighted_k_00_rms_0p0386`
- Archive bytes: `159491`
- Archive SHA-256:
  `bfb912ff7dbbd843b3bf6e5d12ff876eeab359e38113204d0ccae4277fd35d27`
- Member `x` bytes: `159391`
- Member `x` SHA-256:
  `3906da037c6e6604669a863cacd6be88efb3453ea1acbf1b885bf22bb5771a78`
- Inflate parity: `passed`
- Output contract: one source output path and one candidate output path,
  nonempty, paths match
- Output bytes identical: `false` as expected for a score-affecting packet

Remaining blockers propagated by the manifest:

- `cpu_local_allocator_proxy_only`
- `diagnostic_or_stub_sensitivity_map_not_score_authority`
- `is_stub=true`
- `no_active_level2_lane_dispatch_claim`
- `no_contest_cpu_auth_eval`
- `no_exact_cuda_auth_eval`
- `operator_score_claim_review_not_done`
- `score_sensitivity_artifact_must_be_certified_before_promotion`
- `tag contains 'stub'`

Conclusion:

A2 is still not a score candidate. The important progress is that the packet
builder now proves changed archive bytes are consumed by the runtime under the
real inflate contract. The next A2 work should replace the stub sensitivity
map with a certified scorer/component sensitivity artifact, then rebuild this
same packet ladder and run paired exact CPU/CUDA eval only after a fresh lane
claim.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest tests/test_build_a2_sensitivity_weighted_pr101_packet.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_a2_sensitivity_weighted_pr101_packet.py \
  --a2-manifest experiments/results/track1_phase_a2_sensitivity_quant_20260508T154125Z/A2_result.json \
  --output-dir experiments/results/track1_phase_a2_packet_ladder_codex_fixed_20260508T203813Z \
  --variant-limit 1 \
  --run-inflate-parity \
  --inflate-parity-timeout 600

.venv/bin/python tools/audit_a2_packet_ladder_closure.py \
  --repo-root . \
  --strict \
  --json-out /tmp/a2_audit_fixed.json
```

Results:

- A2 focused tests: `13 passed`
- A2 packet ladder: `inflate-parity ladder: PASS`
- A2 closure audit: `passed=true`, `violations=[]`

