# SNeRV Runtime Closure Audit - 2026-06-04

## Scope

This ledger records the read-only runtime closure audit for the SNSA2 SNeRV
package. It does not prune runtime code, change inflate semantics, claim score,
or make the package launchable.

## Source Artifact

- Audit JSON:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/snerv_runtime_closure_audit.json`
- Archive:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/archive.zip`
- Runtime package:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/runtime_package`
- Archive bytes: `142134`
- Archive SHA-256:
  `aebe1b9884ca5a1f06cde05bdc6f4d208ef5d910341fb9e48fabba8b59050fbe`

## Byte Crux

- `0.bin` compressed member bytes: `51586`
- Runtime member compressed bytes: `87364`
- Runtime Python compressed bytes: `87038`
- Runtime-over-payload compressed ratio: `1.693560268289846`
- Static whole-file unreachable runtime bytes: `0`
- Upstream-shaped data-only `archive.zip` bytes: `51694`
- Data-only delta versus self-contained archive: `-90440`

The next byte lever is not a whole vendored-file drop. The live package imports
all vendored Python modules from the generated `inflate.py` entrypoint. The
largest compressed runtime members are:

- `src/tac/substrates/snerv_inverse_steg_carrier/archive.py`: `25014`
- `src/tac/substrates/snerv_inverse_steg_carrier/carrier.py`: `13193`
- `src/tac/analysis/snerv_step_map_coder.py`: `11195`
- `src/tac/substrates/snerv_inverse_steg_carrier/official_mfu.py`: `8617`
- `src/tac/substrates/snerv_inverse_steg_carrier/lf_payload_codec.py`: `6017`
- `src/tac/substrates/snerv_inverse_steg_carrier/official_tub.py`: `5821`
- `src/tac/substrates/snerv_inverse_steg_carrier/dwt.py`: `5221`

## Upstream Bundle Convention

Checked upstream surfaces:

- `upstream/README.md`
- `upstream/evaluate.py`
- `upstream/evaluate.sh`
- `upstream/.github/workflows/eval.yml`
- `src/comma_lab/evaluate.py`
- `src/comma_lab/install.py`

The upstream score rate uses `submission_dir/archive.zip` size. `evaluate.sh`
unzips that archive, then runs `submission_dir/inflate.sh` from the submission
directory. That means runtime source in the PR checkout is outside the upstream
rate formula, while runtime bytes embedded inside `archive.zip` are charged.

Materialized upstream-shaped bundle:

- JSON:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_submission_bundle.json`
- Submission dir:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/submission`
- Data-only archive:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/submission/archive.zip`
- Archive bytes: `51694`
- Archive SHA-256:
  `2f57653c2e21834b731cb102a77d1fa603198f7c9ab13c0b82d94f3ad1f42ee2`
- Receiver proof passed: `True`
- Receiver output SHA-256:
  `5b3b136caf05678e61620773d622523908fd547d090241808614f2684a278970`
- Receiver proof wall seconds: `193.123362`
- Generated audit:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_submission_runtime_audit.json`
- Automated upstream eval gate:
  `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_eval_gate.json`
- Upstream CPU gate result: return code `0`, score `90.61`, PoseNet
  `162.09104919`, SegNet `0.50314105`, rate `0.00137684`, archive bytes
  `51694`.
- Inflated output was hashed before cleanup:
  `c02b88fa5e431ba853d854bcaba109fc0c03913b441cf4184107ab7cd9f9eb2e`.
- Inflated output retained: `False`;
  cleanup status: `deleted_after_success_with_manifest_certificate`.

Internal conservative accounting remains separate: external runtime source
bytes are `392080`; archive plus external runtime source bytes are `443774`.
If a future compliance ruling treats runtime source as charged bytes, the next
path is minified/identifier-stripped external runtime plus receiver replay.

Source minification estimate on the self-contained archive:

- Runtime Python compressed bytes: `87038`
- AST docstring/comment stripped estimate: `66816`
- Estimated saved ZIP bytes: `20222`
- Identifier renaming: not implemented
- Minified runtime receiver replay: missing

## Closure Findings

- Import smoke on packaged `inflate` passed.
- Existing receiver proof is present and passed.
- Receiver output was not retained; proof records output bytes and SHA-256.
- Static import graph still sees dormant/runtime helper imports for `mlx`,
  `pywt`, `brotli`, and unvendored
  `tac.local_acceleration.mlx_scorer_adapters`.
- The audit keeps `snerv_runtime_static_import_closure_missing_members` as a
  blocker even though the current receiver replay passed, because branch-level
  pruning/specialization has not been materialized and reproven.

## Runnable Command

```bash
uv run python tools/audit_snerv_runtime_closure.py \
  --archive-zip /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/archive.zip \
  --runtime-package-dir /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/runtime_package \
  --output-json /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/snerv_runtime_closure_audit.json \
  --scratch-dir /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex
```

Upstream-shaped bundle materialization:

```bash
uv run python tools/materialize_snerv_upstream_submission_bundle.py \
  --source-submission-dir /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/runtime_package/submission \
  --source-archive-zip /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/archive.zip \
  --output-submission-dir /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/submission \
  --output-json /Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_submission_bundle.json \
  --run-receiver-proof \
  --receiver-proof-timeout-seconds 1800
```

## Blockers Preserved

- `minimal_snerv_runtime_closure_not_materialized`
- `contest_inflate_dependency_closure_not_proven_for_pruned_runtime`
- `full_video_scorer_replay_missing`
- `paired_contest_cpu_cuda_auth_eval_missing`
- `snerv_runtime_static_import_closure_missing_members`

## Next Action

Build a candidate-specific minimal inflate/runtime materializer that removes
dormant train/MLX/PyWavelets/Brotli branches and specializes the receiver path
to the actual SNSA2/SNAR2 payload schema. The pruned package must emit its own
byte-closed `archive.zip`, run the same full-video receiver replay, and keep all
score/launch authority false until paired contest eval exists.

If using the upstream PR submission convention, the data-only submission now
passes the automated upstream CPU eval gate, but the measured scorer result is
bad. The next promotion gate remains paired contest CPU/CUDA auth eval only if a
future representation change makes the component score frontier-relevant.

The upstream eval gate is also harvested into
`nerv_candidate_feedback_row.v1` at
`/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/upstream_eval_gate_20260604Tcodex/snerv_upstream_eval_candidate_feedback_row.json`.
The long-training planner consumes it through `--auto-candidate-feedback-root`
and now disables the affected SNeRV queue row with
`snerv_upstream_eval_gate_score_bad`, preventing the current data-only packet
from masquerading as a launchable long-training candidate.
