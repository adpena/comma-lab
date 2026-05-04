# PR86 HPAC To PR85 Mask Contract Port - 2026-05-04 Codex

Scope: PR86/HPAC-to-PR85 mask contract path only. Local planning and parity
guards; no CUDA eval, no remote GPU work, no dispatch claim, and no score
claim.

## Inputs

- PR86 archive anatomy:
  `experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_token_anatomy_forensics.json`
- PR86 full decode/reencode gate:
  `experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_full_decode_reencode_gate_20260504_codex.json`
- PR86 probability contract matrix:
  `experiments/results/pr86_hpac_probability_contract_20260504_worker/pr86_hpac_probability_contract_variants.json`
- PR85 bundle profile:
  `experiments/results/public_pr85_intake_20260503_codex/profile_pr85_bundle.json`
- PR85 QMA9 token-source profile:
  `experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_token_source_profile.json`
- PR85 HPAC parity probe:
  `experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_pr85_qma9_parity_probe.json`

## Artifact Emitted

Planner artifact:

```text
experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_pr85_contract_port_plan.json
```

File SHA-256:

```text
d8aae2bcfe2dafc892c3c29604a197ce30aa88c485ed41978ff060bd49b7d6fe
```

Stable plan digest inside JSON:

```text
58b964de20b7a885d4983ea116d3968bdf12615897a5b37dc9bf1571d98e374a
```

The plan records `blocker.id=pr86_hpac_pr85_mask_contract_port`,
`blocker.status=blocked_fail_closed`, `dispatchable=false`, and
`candidate_spec.status=not_emitted_fail_closed`.

## Contract State

PR85 baseline token extraction is locally proved as a raw-token source profile:

- shape: `[600, 512, 384]`
- dtype: `uint8`
- token SHA-256:
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- source QMA9 mask segment SHA-256:
  `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`

PR86 remains blocked before a reusable HPAC contract can be ported:

- source-contract variant `source_float64_perfect_false` fails at frame `0`,
  group `10`, symbol `191`, after `5951` decoded symbols.
- `source_float32_perfect_false` fails at frame `0`, group `24`, symbol `561`.
- `source_float64_perfect_true` fails at frame `0`, group `15`, symbol `1534`.
- `source_float32_perfect_true` fails at frame `0`, group `15`, symbol `191`.
- no variant produced full submitted-token decode or byte-exact reencode.

The PR85 HPAC parity probe is also blocked:

- status: `blocked_entropy_decode_assertion`
- failure class: `pr86_hpac_decode_contract_or_dependency_mismatch`
- observed error: constriction invalid entropy-model assertion

## Guard Added

`experiments/plan_pr86_hpac_pr85_contract_port.py` now consumes both the PR86
probability-matrix artifact and the PR85 HPAC parity probe. It emits a first-
class blocker object and a candidate-spec gate object. Candidate specs are not
emitted unless all local build gates pass.

The PR85 HPAC candidate parity gate now requires all of the following before a
candidate can be treated as non-blocking:

- `mask_replacement_kind=hpac_pr85_mask_replacement`
- decoded PR85 token SHA and shape match the baseline token-source profile
- byte-exact token parity is true
- runtime output parity is true
- archive byte closure is true
- `score_claim=false` and `dispatch_performed=false`
- the replacement is explicitly non-no-op and the candidate mask SHA differs
  from the source QMA9 mask SHA

This prevents source-preserving, no-op, stale, or non-parity artifacts from
unlocking exact-eval dispatch.

## Verification

```text
.venv/bin/python -m py_compile experiments/plan_pr86_hpac_pr85_contract_port.py src/tac/pr86_hpac_codec.py
.venv/bin/python -m pytest src/tac/tests/test_plan_pr86_hpac_pr85_contract_port.py -q
.venv/bin/python -m pytest src/tac/tests/test_plan_pr86_hpac_pr85_contract_port.py src/tac/tests/test_pr86_hpac_codec.py src/tac/tests/test_pr86_hpac_replay_parity.py -q
.venv/bin/python experiments/plan_pr86_hpac_pr85_contract_port.py --request-dispatchable
```

Results:

- planner tests: `11 passed`
- PR86 HPAC suite: `30 passed, 1 warning`
- planner artifact: fail-closed, no score claim, no dispatch, no candidate
  paths

## Blocker

Remaining blocker:
`pr86_hpac_pr85_mask_contract_port` remains blocked fail-closed.

Removal criteria:

1. PR86 full submitted `tokens.bin` decode succeeds under a named probability
   contract.
2. PR86 decode/reencode emits byte-identical `tokens.bin`.
3. A PR85 HPAC candidate decodes to the PR85 baseline token SHA and shape.
4. The candidate is byte-closed, runtime-output parity checked, and non-no-op.

No PR85-compatible HPAC candidate or candidate spec was built in this pass,
because the required PR86 full decode/reencode parity and PR85 HPAC token
parity gates are still red.
