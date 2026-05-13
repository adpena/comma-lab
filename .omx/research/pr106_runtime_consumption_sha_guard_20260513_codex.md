# PR106/R2 Runtime-Consumption SHA Guard

Date: 2026-05-13
Author: codex

## Scope

This landing hardens the PR106/R2 runtime-consumption proof path. The identity
tool already accepted `--expected-archive-sha256`; the runtime-consumption
tool did not, which allowed a stale or swapped archive to produce a green
`runtime_sidecar_decode_consumption_claim=true` manifest.

## Code Change

- `src/tac/packet_compiler/pr106_runtime_consumption.py`
  - Adds `expected_archive_sha256`.
  - Emits an `archive` custody block with path, bytes, SHA-256, member name,
    member payload bytes, member payload SHA-256, and expected-SHA match state.
  - Fails closed with blocker `expected_archive_sha256_mismatch` before runtime
    decode/apply when the expected SHA does not match.
  - Adds explicit `contest_axis_claim=false` to the runtime-consumption proof.
- `tools/prove_pr106_sidecar_runtime_consumption.py`
  - Adds `--expected-archive-sha256`.
  - Returns rc=2 on blockers.
- `src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py`
  - Covers function-level and tool-level SHA mismatch failure.

## Refreshed Artifact

Ignored experiment artifact refreshed locally:

- `experiments/results/pr106_r2_pr101_runtime_decode_consumption_current.json`

Artifact facts after refresh:

- archive SHA-256: `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- expected SHA match: `true`
- blockers: `[]`
- runtime sidecar decode/apply proof: `true`
- full-frame inflate parity claim: `false`
- contest axis claim: `false`
- score claim: `false`
- promotion eligible: `false`
- ready for exact eval dispatch: `false`

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py

.venv/bin/python -m ruff check \
  tools/prove_pr106_sidecar_runtime_consumption.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py
```

Observed: 38 focused tests passed; ruff passed on the touched Python files.

Follow-up integrated verification after the preflight hook was wired:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py

.venv/bin/python -m ruff check \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  tools/prove_pr106_sidecar_runtime_consumption.py \
  tools/all_lanes_preflight.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py
```

Observed: 47 focused tests passed; ruff passed on the touched Python/preflight
surface.

```bash
.venv/bin/python tools/all_lanes_preflight.py
```

Observed: Gate #26 `PR106 sidecar runtime consumption` passed and Gate #27
`active dispatch claims closed` passed. Full all-lanes preflight still failed
on three broader custody gates:

- Gate #10 `untracked source inventory`: the new research ledger was untracked
  before staging; the existing `experiments/results/` runtime-source baseline
  also differed from the recorded generated-custody baseline.
- Gate #13 `recovery custody snapshots`: bytecode custody is incomplete
  (`14/97` pyc files present) in the historical orphan recovery snapshot.
- Gate #19 `PR91 HPM1 fail-closed custody`: the HPAC runtime has an ambient /
  contradictory device contract and still lacks sidecar-free HPM1 runtime
  consumption proof.

These are non-PR106 score-claim blockers. They remain fail-closed and no
dispatch was attempted.

## Remaining Exact-Eval Boundary

This proof is still not a score claim. It proves runtime parser/decode/apply
consumption of the sidecar bytes only. Promotion still requires full-frame
same-runtime parity or same-runtime auth eval, plus explicit `[contest-CUDA]`
or `[contest-CPU]` custody.

## 2026-05-13 Adversarial Review Hardening Addendum

The recursive review found two silent-failure modes beyond the original
archive-SHA guard:

- Runtime-swapping: a proof could bind the candidate archive SHA while silently
  importing a different `inflate.py` / `src/*.py` runtime tree.
- Partial score-affecting-section proof: format `0x02` carries
  `framing_meta` as a score-affecting PacketIR section, but the old runtime
  proof only mutated `sidecar_payload`.

Hardening landed in the dirty checkout:

- `src/tac/packet_compiler/pr106_runtime_consumption.py`
  - Adds deterministic `pr106_runtime_source_manifest()` over `inflate.py`,
    `src/codec.py`, `src/model.py`, and `src/pr101_grammar.py`.
  - Adds `expected_runtime_source_tree_sha256` fail-closed checking.
  - Adds `runtime_framing_meta_consumption_probe()` for PR101 grammar packets.
  - Emits `runtime_consumed_score_affecting_sections` and
    `runtime_all_score_affecting_sections_consumed`.
- `tools/prove_pr106_sidecar_runtime_consumption.py`
  - Adds `--expected-runtime-source-tree-sha256`.
- `tools/all_lanes_preflight.py`
  - Gate #26 now threads expected archive and runtime-tree hashes for both R2
    grammars and refuses missing `pr106_payload`, `sidecar_payload`, or
    `framing_meta` runtime-consumption proof.
- Refreshed runtime proof artifacts include the hardened schema for:
  - `experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/original_r2/runtime_consumption.json`
  - `experiments/results/pr106_r2_packetir_identity_recode_consumption_20260513T182944Z/pr101_grammar/runtime_consumption.json`
  - `experiments/results/pr106_r2_pr101_grammar_runtime_consumption_proof.json`
  - `experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/runtime_consumption.json`
  - `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/runtime_decode_consumption.json`

Latest focused verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_adjudicate_contest_auth_eval_policy.py -q

.venv/bin/python -m ruff check \
  tools/prove_pr106_packetir_identity.py \
  tools/prove_pr106_sidecar_runtime_consumption.py \
  tools/all_lanes_preflight.py \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  scripts/adjudicate_contest_auth_eval.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_adjudicate_contest_auth_eval_policy.py
```

Observed: 44 focused tests passed; ruff passed; Gate #26 passed with expected
archive/runtime SHA custody and non-score labels.
