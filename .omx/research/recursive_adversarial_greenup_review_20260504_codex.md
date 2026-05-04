# Recursive Adversarial Greenup Review - 2026-05-04 Codex

Scope: current PR85 frontier pipeline and 2026-05-04 worker/subagent outputs.
No remote or GPU jobs were dispatched.  This is a guardrail ledger, not a score
ledger.

## Findings Ordered By Severity

### 1. Manifest readiness vocabulary can drift across builders

Severity: high pre-dispatch burn risk.

The Lightning exact-eval submit path already blocks manifests with an unsafe
`exact_eval_dispatch_gate`, but current builders and worker outputs also use
`dispatch_gate`, `dispatch_unlocked`, `ready_for_exact_eval_after_lane_claim`,
`ready_for_exact_eval_dispatch_claim`, `ready_for_fixed_runtime_exact_eval`,
`exact_eval_runtime_contract`, and nested fixed-runtime preflight sections.

Observed examples:

- STBM1BR + QRGB stack manifest records
  `dispatch_gate=blocked_local_only_until_standalone_exact_positives_and_lane_claim`
  and `dispatch_unlocked=false`.
- QMA9 run grammar candidate summary records a structured dispatch gate with
  `dispatch_unlocked=false` and `safe_for_remote_dispatch=false`.
- Fixed-runtime bridge and QRGB pair-atom manifests can be eligible only after a
  Level-2 lane claim.

Permanent protection implemented:

- Added `experiments/preflight_candidate_manifest_dispatch_readiness.py`.
- Added focused tests in
  `src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py`.

This guard fails closed on builder-specific blocked/planning-only/local-only
dispatch gates, false readiness booleans, nested exact-runtime/fixed-runtime
blockers, unsafe `exact_eval_dispatch_gate`, exact-negative markers, and reports
lane-claim warnings for otherwise eligible manifests.

Recommended promotion command:

```bash
.venv/bin/python experiments/preflight_candidate_manifest_dispatch_readiness.py \
  --manifest <candidate-dir>/manifest.json \
  --fail-if-not-ready
```

### 2. Exact runtime contract drift remains the biggest current harness class

Severity: high correctness risk.

The STBM1BR mask-recode incident proved that local builder/runtime support is
not sufficient when the submitted `inflate.sh` is a different public replay
runtime.  Current STBM1BR builder output now records explicit exact replay
runtime contract state, and the new manifest preflight catches remaining
blocked/local-only stack manifests before claim or queue.

Promotion blocker:

- Do not exact-eval a runtime-changing or public-replay candidate unless its
  manifest preflight is ready and the exact submitted runtime path/tree is
  recorded.

### 3. Public replay parity gaps must remain external signal

Severity: high evidence-risk, medium dispatch-risk.

PR86/PR91/HPAC/HPM1 worker outputs still report decode/parity failures or
prefix-only recovery.  These remain external design signals until full
decode/reencode parity and canonical exact CUDA replay exist.  Self-reported
public scores must not replace our PR85 T4 exact anchor.

Permanent checks already present:

- Source-embedded payload runtime guard in `scripts/launch_lightning_batch_job.py`.
- Public external inflate closure tests.
- PR86/PR91 parity diagnostics and fail-closed ledgers.

### 4. Negative stack reuse needs manifest-level friction

Severity: medium-high dispatch-quality risk.

QRGB singleton/stack waves have exact-negative or directionally negative
evidence.  Stack builders may remain useful infrastructure, but exact eval must
require a fresh positive premise, non-noop archive mutation, fixed/runtime
preflight, and lane claim.  The new manifest preflight blocks current
local-only stack manifests and allows eligible pair-atom manifests only with a
lane-claim warning.

### 5. Dispatch-claim hygiene is covered but must stay on every queue path

Severity: medium coordination risk.

Focused verification confirms the claim helper and Lightning submit guard are
still present.  No dispatch was attempted in this review.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py -q
# 5 passed

.venv/bin/python -m pytest \
  src/tac/tests/test_lightning_batch_jobs.py::test_exact_eval_manifest_dispatch_gate_blocks_renderer_stack_without_pose_safety \
  src/tac/tests/test_claim_lane_dispatch.py::test_dispatch_claim_preflight_covers_helper_and_lightning_launcher -q
# 2 passed
```

Representative manifest probes:

```bash
.venv/bin/python experiments/preflight_candidate_manifest_dispatch_readiness.py \
  --manifest experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/pr85_stbm1br_plus_qrgb_f2_randglobal_pair_0192/manifest.json \
  --fail-if-not-ready
# exits 2; blocks local-only STBM1BR + QRGB stack

.venv/bin/python experiments/preflight_candidate_manifest_dispatch_readiness.py \
  --manifest experiments/results/pr85_qma9_run_grammar_candidates_20260504_worker/candidate_summary.json \
  --fail-if-not-ready
# exits 2; blocks unsafe QMA9 run grammar summary

.venv/bin/python experiments/preflight_candidate_manifest_dispatch_readiness.py \
  --manifest experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/pr85_qrgb_f1_bias_pair_0164/manifest.json \
  --fail-if-not-ready
# exits 0 with lane_claim_still_required warning

.venv/bin/python experiments/preflight_candidate_manifest_dispatch_readiness.py \
  --manifest experiments/results/pr85_fixed_runtime_bridge_candidates_20260504_codex/expanded_qpost_qrm1_posefp16/manifest.json \
  --fail-if-not-ready
# exits 0 with lane_claim_still_required warning
```

## Promotion Blockers

- Any exact-eval candidate with blocked/planning/local-only manifest readiness
  must remain local.
- Any public replay or runtime-changing candidate needs exact submitted
  `inflate.sh` runtime support, runtime tree/hash custody, and manifest
  readiness.
- PR86/PR91/HPAC/HPM1 remain blocked until full decode/reencode parity.
- QRGB-derived stacks need fresh positive evidence before reusing exact-negative
  atoms.
- A Level-2 dispatch claim is still mandatory before any remote/GPU job.
