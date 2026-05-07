# HNeRV q10 adversarial score-lowering review - 2026-05-07

Scope: adversarial dispatch-readiness review for the operator-approved
`pr106_q10_151byte_brotli` exact-eval target, compared against the HNeRV
lgblock16 rate-only candidate and Apogee int6. No lane claim, GPU dispatch, or
score claim was made.

Primary artifacts:

- q10 packet:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json`
- field/meta selection:
  `experiments/results/field_meta_dispatch_selection_20260507_codex/selection.json`
- q10 archive:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/pr106_hnerv_brotli_repack_candidate.zip`
- apogee int6 metadata:
  `experiments/results/apogee_int6_repack_20260504_claude/repack_metadata.json`

## Current selection

`pr106_q10_151byte_brotli` is the selected operator-approved exact-eval target
for this review. Its local packet is static-ready and byte-closed:

- archive SHA-256:
  `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
- archive bytes: `186088`
- source PR106 bytes: `186239`
- byte delta: `-151`
- official rate-only score delta: `-0.000100544702`
- runtime tree SHA-256:
  `f402908b2490718c4f7b76987335ec1a496cb12ab71c27e1e1aea4024d5712cb`
- score claim: `false`
- dispatch attempted: `false`

The selector records operator approval, but also records that approval does not
unlock dispatch. Remaining field-selection blockers are:

- missing active Level-2 lane claim for
  `pr106_q10_151byte_brotli` /
  `exact_eval_pr106_q10_151byte_brotli_20260507`
- missing KKT proof or converged ADMM waterline result
- missing Lightning environment in this local shell:
  `LIGHTNING_SSH_TARGET`, `LIGHTNING_REMOTE_PACT`, `LIGHTNING_UPSTREAM_DIR`,
  `LIGHTNING_TEAMSPACE`, `LIGHTNING_STUDIO`, `LIGHTNING_SDK_USER`

## Comparison

| candidate | byte delta | rate/proxy score delta | exact-CUDA readiness | dispatch route |
| --- | ---: | ---: | --- | --- |
| `pr106_q10_151byte_brotli` | `-151` | `-0.000100544702` | static-ready, runtime-closed, not dispatch-ready | claim + env + KKT/ADMM proof, then exact CUDA |
| `pr106x_lgblock16_1byte_brotli` | `-1` | `-0.000000665859` | static-ready, runtime-closed, Pareto-dominated by q10 | do not spend before q10 unless q10 invalidates |
| `apogee_int6` | `-15789` | `-0.010513247011` rate component only | not exact-dispatch-ready | blocked by missing distortion gate and runtime proof |

Apogee int6 is not a rate-only archive-equivalence candidate. It changes
score-affecting payload semantics, so the large byte delta is only forensic
until a contest-faithful distortion model, scorer-basin parity report, or exact
positive CUDA evidence exists. Its readiness audit correctly fails closed.

## Fastest safe path

1. Export the required Lightning environment variables and rerun the packet's
   `verify_lightning_env` command.
2. Add a narrow KKT/ADMM proof for this discrete rate-only selection, or record
   an explicit operator/adversarial override that KKT is over-conservative for
   raw-equivalent single-archive byte repacks. Do not silently bypass it.
3. Claim the lane with the packet's copy-safe `claim_lane_no_dispatch` command.
4. Refresh the packet with `--operator-approved-exact-cuda`.
5. Submit exact CUDA only after the active claim is present and no same-lane
   conflict appears.
6. Harvest with adjudication required before any score claim.

## Failure modes

- Stale candidate naming can route the wrong lane; fixed q10 claim notes now
  derive from the actual lane, byte delta, source label, member name, SHA, and
  packet path.
- Stale packet manifests can lose runtime custody; fixed q10 manifests now
  carry the public replay runtime tree SHA into both runtime-contract sections.
- Field selector metadata drift can hide byte closure; fixed Apogee intN
  metadata normalization now reads `candidate_archive_sha256` with
  `archive_path` and `archive_size_bytes`.
- False readiness remains blocked: operator approval, byte closure, and
  predicted rate deltas do not satisfy lane claim, environment, exact CUDA, or
  adversarial score-review gates.
- Dirty WR01/private state is intentionally excluded from the committed
  selection artifact for this q10 review.

## Over-conservatism check

The KKT/ADMM requirement is the only gate that appears potentially
over-conservative for `pr106_q10_151byte_brotli`: q10 is a raw-equivalent,
single-archive, rate-only repack with an official byte-rate delta proof. It is
still a useful field-planning guard, but it can cost wall-clock if environment
and lane claim are otherwise ready. The safe reduction is to add a narrow,
auditable rate-only KKT proof or explicit adversarial override for this class,
not to dispatch around the gate.
