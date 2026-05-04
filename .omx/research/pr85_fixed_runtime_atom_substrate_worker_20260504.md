# PR85 Fixed-Runtime Atom Substrate Worker - 2026-05-04

## Scope

Worker A hardened the PR85 fixed-runtime readiness surface only. No remote GPU
dispatch was performed and no dispatch claim was created.

## Artifact Status

- Current readiness artifact:
  `experiments/results/public_pr85_intake_20260503_codex/pr85_fixed_runtime_readiness_preflight.json`
- Source archive:
  `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- Source archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- Source PR85 member SHA-256:
  `53bc78effa78cc7850d08a9ddc5488665b93136e9843549d917c17df729a1c50`
- Preflight evidence grade:
  `planning_only/no_score_claim`

The refreshed fixed-runtime substrate preflight is `ready` for the bridge
substrate itself. The new atom-substrate checks all pass:

- source payload SHA matches the archive member bytes
- expanded runtime member manifest matches actual `masks.qma9`,
  `renderer.bin`, `optimized_poses.bin`, and `qpost.bin` bytes
- `qpost.bin` QPS1 stream lengths and SHAs match the PR85 source segments
- robust_current `PR82_QRM1_RANDMULTI_SPECS` exactly matches the PR85 72-group
  schedule
- transcoded randmulti QRM1 stream parses under that runtime schedule
- qpost apply has a JSON records summary surface in stdout

## Dispatch-Unlocked Versus Blocked

Unlocked:

- The minimal fixed-runtime bridge substrate remains eligible for a later exact
  CUDA eval dispatch after the normal lane-claim rule. This is only a runtime
  substrate readiness result, not a score claim.

Blocked:

- No PR85 atom-edit candidate is unlocked by this ledger alone.
- Atom-edit exact eval candidates must now pass the opt-in guard:
  `--atom-source-archive <source-pr85-archive>`.
- Source-preserving candidates fail closed with
  `pr85_atom_edit_noop_source_preserving`.
- Stale/mismatched dispatch targets can fail closed with
  `--expected-archive-sha256` and/or `--expected-member-sha256`.

## Permanent Guard Added

`experiments/preflight_pr85_fixed_runtime_readiness.py` now emits:

- `fixed_runtime_bridge.atom_substrate`
- `custody_expectations`
- `atom_edit_guard`

These checks prevent a readiness report from going green when charged PR85 atom
bytes are not tied to the runtime members that robust_current will actually
consume, when the QRM1 runtime schedule diverges, when qpost application lacks
an apply summary surface, when expected archive/member SHAs mismatch, or when a
candidate preserves every source PR85 segment under the atom-edit guard.

## Commands

```bash
.venv/bin/python -m pytest src/tac/tests/test_preflight_pr85_fixed_runtime_readiness.py
```

Result: `7 passed in 0.24s`.

```bash
.venv/bin/python experiments/preflight_pr85_fixed_runtime_readiness.py --fail-if-not-ready
```

Result: exit `0`; refreshed
`experiments/results/public_pr85_intake_20260503_codex/pr85_fixed_runtime_readiness_preflight.json`
with `ready_for_fixed_runtime_exact_eval=true` and no blockers.

```bash
.venv/bin/python experiments/preflight_pr85_fixed_runtime_readiness.py \
  --archive experiments/results/public_pr85_intake_20260503_codex/archive.zip \
  --robust-current-dir submissions/robust_current \
  --atom-source-archive experiments/results/public_pr85_intake_20260503_codex/archive.zip \
  --json-out /tmp/pr85_same_source_atom_guard.json \
  --fail-if-not-ready
```

Result: exit `2`; blocker
`pr85_atom_edit_noop_source_preserving` proves the same-source/no-op atom-edit
path is fail-closed.

## Next Action

For any PR85 atom candidate, rerun this preflight with
`--atom-source-archive experiments/results/public_pr85_intake_20260503_codex/archive.zip`
and the candidate archive/member expected SHA values before creating a dispatch
claim. Dispatch only if the refreshed report has no blockers and the normal
Level 2 lane claim is active.
