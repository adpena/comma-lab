# Codex findings — Z8 entropy-delta full artifact, cleanup, and rate-floor gate

- **Date:** 2026-06-01T18:11:00Z
- **Agent:** Codex
- **Lane:** `z8_joint_p18_p19_entropy_delta_ratefloor`
- **Axis:** `[macOS-CPU advisory]`
- **Score authority:** false
- **Promotion authority:** false

## What Landed

The Z8 detail-coefficient entropy transcode proof was regenerated as a single
durable artifact with:

- byte-closed `archive.zip`
- receiver proof through the emitted `inflate.sh`
- integrated full receiver-runtime benchmark
- manifest-then-delete cleanup for raw benchmark output
- factual archive-changedness fields preserved in archive-bound candidate rows
- a machine-readable contest rate-floor gate

Durable artifact:

`/Volumes/VertigoDataTier/pact/z8_entropy_delta_full_ratefloor_integrated_20260601T180520Z/materialized`

Key files:

- `z8_joint_p18_p19_deadzone_manifest.json`
  - sha256 `aca1e09c9bb351fdeab11c0455195e94ca037995a832a9b74a6ae1fe2421c992`
- `archive_bound_candidate_adapter_package.json`
  - sha256 `c3092bd10a3b29e577866d50bab933a74b881671acb06bd3279eeff6538d52d4`
- `z8_hpc1_receiver_proof.json`
  - sha256 `d66cbe47222ebe596083225be453babce8a95bc8cfc2e190daeef080a36583fa`
- `z8_joint_p18_p19_deadzone_inflate_runtime_benchmark.json`
  - sha256 `f5a9209cf52f456ae676335e209f9877583ae8c7339106365e3232e1a6e1959d`

## Byte Verdict

The entropy-coded detail transcode is a real rate-axis collapse for Z8:

- source archive bytes: `152,069,787`
- candidate archive.zip bytes: `28,338,722`
- candidate `0.bin` bytes: `28,239,010`
- archive byte delta: `-123,830,777`
- archive rate ratio: `0.18569770206885342`
- wavelet blob bytes: `152,040,265 -> 28,208,322`
- wavelet blob delta: `-123,831,943`

This proves the raw-float detail blob diagnosis, but it also closes the
score-promotion question for this Z8 representation: the candidate archive
still has a contest rate-only score floor of `18.869591763740246`.

That is now encoded in manifest field:

`rate_floor_report.schema = z8_contest_rate_floor_gate.v1`

## Receiver And Runtime

Receiver proof:

- `runtime_consumption_proof_passed: true`
- `receiver_contract_satisfied: true`
- `receiver_output_bytes: 3,662,409,600`
- receiver raw retained: `false`

Runtime benchmark:

- `inflate_seconds_best: 94.8361566659878`
- auth-window fraction: `0.05268675370332655`
- output retention policy: `manifest_then_delete`
- large artifact cleanup default: `true`
- benchmark output directory size after run: `0B`

No `.raw`, `.mkv`, or `.mp4` artifacts were retained under the run root.

## Bug Fixed

`build_archive_bound_candidate_runtime_package(...)` used the shared
false-authority contract after setting factual changedness fields. That
overwrote:

- `score_affecting_payload_changed`
- `charged_bits_changed`

Archive-bound package rows now expand false-authority first, then set factual
changedness fields. Regression test:

`test_archive_bound_runtime_package_preserves_changedness_without_score_authority`

The corrected package row preserves:

- `semantic_payload_changed: true`
- `score_affecting_payload_changed: true`
- `exact_axis_score_affecting_adjudication_required: true`
- `charged_bits_changed: true`

while keeping:

- `score_claim: false`
- `score_claim_valid: false`
- `promotion_eligible: false`
- `ready_for_exact_eval_dispatch: false`

## Engineering Hardening

`benchmark_z8_submission_inflate_runtime(...)` now defaults to
manifest-then-delete for inflated raw outputs. Operators can opt into retention
with `--retain-output`, but default benchmark execution is disk-safe.

The Z8 one-shot and relinearized materializers now emit `rate_floor_report`
from the canonical contest denominator via `tac.archive_byte_profile`.

## Tests

Commands:

```bash
uv run pytest \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_joint_coefficient_waterfill.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_inflate_runtime_benchmark.py \
  src/tac/tests/test_archive_bound_candidate_runtime_bridge.py -q

uv run ruff check \
  src/tac/substrates/z8_hierarchical_predictive_coding/joint_coefficient_waterfill.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/inflate_runtime_benchmark.py \
  tools/benchmark_z8_submission_inflate_runtime.py \
  tools/materialize_z8_joint_p18_p19_deadzone_candidate.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_joint_coefficient_waterfill.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_inflate_runtime_benchmark.py \
  src/tac/optimization/archive_bound_candidate_runtime_bridge.py \
  src/tac/tests/test_archive_bound_candidate_runtime_bridge.py
```

Result:

- `32 passed`
- ruff clean

## Verdict

Do not dispatch this Z8 wavelet-detail entropy-transcode candidate to exact
CPU/CUDA auth eval as a score-lowering candidate. The receiver/runtime and
cleanup blockers are burned down; the remaining blocker is mathematical:
transmitting wavelet video is still the wrong rate scale for the contest.

Keep the implementation as reusable infrastructure and posterior evidence.
The next score-lowering tranche should spend effort on the score-exact oracle
stack: HPRC/PR95-style compact receiver training, full-video P18/P19 saliency,
latent/coeff adjoints, native rate pressure, and archive-byte gates before
exact auth.
