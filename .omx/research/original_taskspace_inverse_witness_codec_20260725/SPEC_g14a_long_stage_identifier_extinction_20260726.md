# G14a long stage-identifier bug-class extinction

Date: 2026-07-26
Parent lane: `lane_g14_taskspace_g8_a3_n2_allocator_20260726`
Status: implemented and locally verified; no relaunch authorized; `research_only=true`

## Failure evidence

The reviewed full G14 run in
`.omx/runs/taskspace_g8_a3_n2_allocator/20260726T135256Z_full_p2_4_8_16`
completed and preserved 64 G8 rows, 29 G8+A rows, and 28 matched G0+A
controls before stopping at:

```text
TaskspaceG8A3N2AllocatorError: stage identifier must be bounded safe ASCII
```

The failing call was the `stage_310.g0_a_control` token for a lawful retained
G8 proposal plus the longer `POST_G8_Y1_SUPPORT_COPY_V1` program identity and
its SHA-256.  `_stage_token()` hashes and truncates its source identity to a
bounded filename, but first called the general `_require_id()` helper whose
192-character wire/object-ID bound rejected the longer composition.  This is
an implementation failure after a successful G8+A checkpoint, not a scorer,
receiver, representation, or optimization-family negative.

Durable evidence:

- blocker receipt:
  `.omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_n2_g8_a3_allocation_stage_id_blocker_20260726.json`;
- blocker receipt SHA-256:
  `18c7e437e3c0b3caca9b6272c319b753b37f4dea04c6222c1931cfb201ff79e0`;
- failed-run manifest SHA-256:
  `cdf0e3c9db8e73e8bffa23ed9d9ec01e6e699cfbceece3379a30c0f346f43ec4`;
- historical runner SHA-256:
  `4fbaa88ac67451f932584fa9109d48e13bc9c606ec2d1c773efb060f2261c3a1`;
- canonical probe outcome:
  `g14_full_n2_stage_id_blocker_20260726T143433Z`.

## Derived bound and required change

Do not weaken the general 192-character ID contract.  Add one stage-token
source-identity validator whose maximum is derived from the closed fields used
by G14:

```text
max proposal_id  = 192
separator         =   1
max A program_id  = 192
separator         =   1
program SHA-256   =  64
--------------------------------
maximum           = 450 ASCII characters
```

The validator accepts only the existing safe alphabet
`[A-Za-z0-9_.:-]`, requires the first character to be alphanumeric, and
accepts lengths `1..450` exactly.  `_stage_token()` must validate its internal
prefix with the existing `_require_id()`, validate the source identity with
the new 450-character contract, normalize the safe punctuation exactly as
before, retain at most the existing 96-character human-readable prefix, and
suffix the existing first 12 hexadecimal characters of SHA-256 over the
**entire untruncated source identity**.

Short lawful identifiers must produce byte-identical stage tokens to the
historical implementation.  Two long identities that differ only after the
visible 96-character prefix must produce distinct tokens.  Character 450 is
accepted; character 451, non-ASCII, whitespace, path separators, empty input,
and unsafe punctuation are rejected.

## Regression and execution contract

Focused tests must reproduce the actual composition class:

1. a maximum-length proposal/program/SHA source identity is accepted;
2. the concrete long post-G8-copy matched-control identity that stopped the
   first run is accepted;
3. suffix-only differences remain collision-resistant through the digest;
4. the 451-character and unsafe-input boundaries fail closed; and
5. every pre-existing short-token expectation remains unchanged.

Run Ruff, formatting, py_compile, the focused G14 suite, and the composed
G10/G12/G14 suite.  Freeze new runner/test/amendment hashes before execution.

The old run directory and manifest are immutable evidence.  Do not monkeypatch
the process, relabel the old implementation hash, or silently import its
checkpoints under new custody.  Relaunch the exact reviewed full lattice in a
fresh durable run directory under the new transitive implementation manifest.
The old checkpoints remain useful research evidence and a reproducibility
oracle; they are not deleted.

After the new run passes the formerly failing branch, update the canonical
probe outcome from blocking infrastructure failure to a resolved/advisory
status with the new run evidence.  Representation-family verdicts remain
unchanged until the full receipt closes.

## Triality and pointer honesty

- DSL: no packet, receiver, scorer, or archive grammar changes.
- DAG: only durable stage-name derivation changes before checkpoint publish.
- Equation: no score, delta, byte-ceiling, interaction, or selection arithmetic
  changes.

Pointer delta: none.  This amendment authorizes no score claim, candidate,
promotion, or exact-eval dispatch.

## Implementation closure receipt

Closed locally on 2026-07-26 without source decode, scorer load, experiment
launch, checkpoint migration, or mutation of the failed run.  The general
`_require_id()` 192-character contract remains byte-for-byte unchanged.  A
stage-specific derived `192 + 1 + 192 + 1 + 64 = 450` safe-ASCII contract now
guards source identities, while the existing general contract guards the stage
prefix.  The token still exposes at most 96 normalized source characters and
uses the first 12 hex characters of SHA-256 over the complete untruncated
source identity.

Frozen implementation hashes after the fix:

- runner SHA-256:
  `47d3601285600cea2d9ec03069b6910be33074bd526358b243e71cf5b5383400`;
- focused-test SHA-256:
  `308c6278078c8b0d8303b5e05af973046a17918aed0453026c55dd76d9beddd9`.

Regression-first evidence on the historical runner: the new boundary suite
returned `11 failed, 25 passed`, including the exact 194-character stopped
post-G8-copy matched-control identity.  After the fix:

- Ruff lint: clean;
- Ruff format check: both files already formatted;
- `py_compile`: clean;
- focused G14 suite: `36 passed`;
- composed G10/G12/G14 suite: `55 passed`;
- `git diff --no-index --check` on each owned untracked file: no whitespace
  diagnostics (return code 1 is expected for an added file).

Pointer delta remains none.  No score, formulation, receiver, representation,
or family verdict changed, and this receipt does not authorize a run.
