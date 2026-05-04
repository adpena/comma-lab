# PR82 QRM1 Mask-Dependent Runtime Guard Addendum - 2026-05-03

Scope: local runtime/preflight only. No remote dispatch, no GPU eval, and no
score claim.

The public PR82 replay has nine replay-special randmulti groups in the QRM1
contract. Two are safe in `submissions/robust_current/apply_qzs3_postprocess.py`
because they operate directly on raw frame 1:

- group 61, `(222, 222, 4)`: global RGB/channel bias.
- group 71, `(224, 222, 4)`: 2x2 tile channel bias.

The remaining active special groups require the decoded PR82 source mask tensor
to construct class or boundary masks. The raw-frame-only postprocess helper
cannot apply them faithfully without a wider runtime contract:

- group 62, `(222, 223, 4)`: class-conditioned all-channel bias.
- group 63, `(223, 222, 2)`: class-conditioned channel bias.
- group 64, `(223, 223, 4)`: boundary all-channel bias.
- group 65, `(223, 221, 4)`: class-conditioned channel bias.
- group 66, `(223, 224, 4)`: boundary/class-boundary channel bias.
- group 67, `(223, 221, 4)`: class-conditioned channel bias.
- group 68, `(223, 219, 4)`: width-2 boundary channel/class bias.
- group 70, `(223, 218, 4)`: width-3 boundary channel/class bias.

Behavior added:

- `classify_qrm1_randmulti_stream()` and `classify_archive_qrm1_support()`
  produce a candidate-specific fail-closed support report.
- Active unsupported groups block QRM1 dispatch. Unsupported groups with all
  zero rows are semantically inert and no longer fail the runtime decoder.
- Duplicate `qpost.bin` members fail closed in the archive classifier.

Candidate-specific proof on the generated PR81+PR82 stack artifacts:

- `pr81_qma9_pr82_qps1_controls_all600`: no randmulti stream.
- `pr81_qma9_pr82_qps1_nm2_generic_randmulti`: not QRM1; no unsupported QRM1
  group ids.
- `pr81_qma9_pr82_qps1_qrm1_all072_randmulti`: active unsupported group ids
  `[62, 63, 64, 65, 66, 67, 68, 70]`.
- `pr81_qma9_pr82_qps1_controls_qrm1_all072`: active unsupported group ids
  `[62, 63, 64, 65, 66, 67, 68, 70]`.

Reactivation criterion: feed the decoded mask tensor into the robust-current
postprocess path, port the public PR82 class/boundary semantics byte-for-byte,
and attach a local raw-output parity/delta proof for the exact archive SHA
before any exact CUDA dispatch.

## Supported-Subset Candidate Addendum

No remote GPU dispatch was performed.  No scorer was invoked.

Implemented a deterministic PR81+PR82 QRM1 subset/exclusion policy in
`experiments/build_pr81_pr82_henosis_stack_candidate.py`: the builder first
classifies the all-72 QRM1 stream with the robust-current runtime classifier,
then drops only active unsupported group ids before emitting the
exact-evaluable supported-subset candidates.  For the current PR82 source this
excludes `[62, 63, 64, 65, 66, 67, 68, 70]` and preserves supported specials
61 `(222,222,4)`, 71 `(224,222,4)`, and generic group 69.

New deterministic archive artifacts:

- `pr81_qma9_pr82_qps1_qrm1_supported_subset_randmulti/archive.zip`:
  `230247` bytes, SHA-256
  `7d16b7e6ecd1cea7aa31b24b5aad1a22d3f50ae5424ff49d691293c62d7041d0`,
  `qpost.bin` `14193` bytes, qpost SHA-256
  `d3b4569d2c37eb86169cfd7468ff255350e2262be08b43c95b54805d6b322908`,
  synthetic raw-output delta proof changed `4323678` values.
- `pr81_qma9_pr82_qps1_controls_qrm1_supported_subset/archive.zip`:
  `232778` bytes, SHA-256
  `cd7e1ad2f4aa2056d07e056bb0266f4b7cfd7dc9a5ad8ec0c02ea085187bb883`,
  `qpost.bin` `16724` bytes, qpost SHA-256
  `ff55043fa75117d5d9613453eae9b13f05f4b54d091594338fe9ce1ebbd8f930`,
  synthetic raw-output delta proof changed `5070768` values.

Both supported-subset manifests report
`dispatch_ready_now=true`, `no_remote_dispatch=true`,
`remote_dispatch_performed=false`, `score_claim=false`, and
`lane_claim_required_before_any_exact_eval=true`.
This means exact-evaluable after a future lane claim; it is not score evidence.

Verification:

- `.venv/bin/python experiments/build_pr81_pr82_henosis_stack_candidate.py`
  rebuilt six local candidates and selected
  `pr81_qma9_pr82_qps1_controls_qrm1_supported_subset` as the highest-EV
  QRM1-compatible candidate.
- `.venv/bin/python -m pytest src/tac/tests/test_build_pr81_pr82_henosis_stack_candidate.py src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py -q`
  -> `12 passed, 1 warning`.

## Manifest Identity Hardening Addendum

Follow-up guard added after dispatch preparation: supported-subset manifests
now expose archive identity at the top level as `archive_bytes`,
`archive_path`, and `archive_sha256`, mirroring `output_archive`.  This avoids
future automation relying on nested proof fields or candidate-summary rows to
recover exact archive custody.

The rebuild preserved all queued archive bytes and SHA-256s:

- controls: `9fe02888361924c247039e9c8ec99fcb83a85f7556e89b045b978a761f444609`
- NM2: `435e8b1695955b4dc3ced62070afb8d6f0282ed7b280bd1d36a287c1636c722e`
- controls+QRM1 supported subset:
  `cd7e1ad2f4aa2056d07e056bb0266f4b7cfd7dc9a5ad8ec0c02ea085187bb883`

Regression guard:

- `src/tac/tests/test_build_pr81_pr82_henosis_stack_candidate.py` asserts
  top-level manifest archive identity matches `output_archive`.
- `.venv/bin/python -m pytest src/tac/tests/test_build_pr81_pr82_henosis_stack_candidate.py src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py -q`
  -> `12 passed, 1 warning`.

## PR81 Reordered QZS3 Restore Bug Addendum

The first exact-eval wave failed before scoring, so it is harness evidence
only.  All failures shared the same inflate traceback:
`ValueError: PR81 chunk length mismatch: consumed 0, got 38032` in
`submissions/robust_current/inflate_renderer.py`.

Failure logs preserved locally:

- `experiments/results/lightning_batch/exact_eval_pr81_pr82_controls_all600_g4dn_t4_20260503T210600Z/auth_eval.failed_restore.log`
- `experiments/results/lightning_batch/exact_eval_pr81_pr82_nm2_randmulti_g4dn_t4_20260503T210600Z/auth_eval.failed_restore.log`
- `experiments/results/lightning_batch/exact_eval_pr81_pr82_controls_qrm1_supported_g4dn_t4_20260503T210116Z/auth_eval.failed_restore.log`
- `experiments/results/lightning_batch/exact_eval_pr81_pr82_controls_qrm1_supported_l40sdiag_20260503T210116Z/auth_eval.failed_restore.log`

Root cause: robust-current restored PR81 reordered QZS3 chunks from runtime
module attributes, but the faithful renderer intentionally replaces public
`QConv2d`/`QEmbedding` with regular PyTorch modules.  That made packed/scales
chunk lists empty.  Fix: derive restore chunks from the exact QZS3 state-dict
packing predicates used by `tac.quantizr_qzs3_codec`.

Regression guard:

- `src/tac/tests/test_unpack_renderer_payload_fixedslice.py::test_actual_pr81_reordered_qzs3_model_payload_restores_to_qzs3`
  restores the actual PR81 public model payload and asserts QZS3 output.
- `.venv/bin/python -m pytest src/tac/tests/test_unpack_renderer_payload_fixedslice.py::test_actual_pr81_reordered_qzs3_model_payload_restores_to_qzs3 src/tac/tests/test_build_pr81_pr82_henosis_stack_candidate.py src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py -q`
  -> `13 passed, 1 warning`.

Relaunch wave after fix:

- `exact_eval_pr81_pr82_controls_all600_g4dn_t4_fixrestore_20260503T211855Z`
- `exact_eval_pr81_pr82_nm2_randmulti_g4dn_t4_fixrestore_20260503T211855Z`
- `exact_eval_pr81_pr82_controls_qrm1_supported_g4dn_t4_fixrestore_20260503T211855Z`

## Builder-Level PR81 Restore Preflight Addendum

The PR81 restore bug class is now guarded at candidate-build time, not only by
the runtime unit test. `experiments/build_pr81_pr82_henosis_stack_candidate.py`
loads `submissions/robust_current/inflate_renderer.py`, restores the actual
PR81 reordered model payload with
`_restore_pr81_reordered_qzs3_model_payload`, and records
`source_pr81_runtime_restore_preflight` in every candidate manifest. Candidate
generation fails closed if the restore does not emit valid `QZS3` bytes with a
positive block size.

Current manifest proof for
`pr81_qma9_pr82_qps1_controls_qrm1_supported_subset`:

- input model payload bytes: `55725`
- input model payload SHA-256:
  `b649b0dacb1dcc93fd7da2e7f5c6d398fa933d2fe3087520359612afc8e4832d`
- restored model bytes: `59288`
- restored model SHA-256:
  `2333284a73446c3b323948fb883ade0f677baf9ad5d9d06aa1da7bec337bd9c9`
- restored block size: `32`
- runtime file:
  `submissions/robust_current/inflate_renderer.py`
- runtime file SHA-256:
  `5465cd34a2d86e8f708d458c416c0e579c4747edc1fba9c9ba3208b6f13d1291`

Verification:

- `.venv/bin/python -m py_compile experiments/build_pr81_pr82_henosis_stack_candidate.py src/tac/tests/test_build_pr81_pr82_henosis_stack_candidate.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_pr81_pr82_henosis_stack_candidate.py src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py src/tac/tests/test_unpack_renderer_payload_fixedslice.py::test_actual_pr81_reordered_qzs3_model_payload_restores_to_qzs3 -q`
  -> `13 passed, 1 warning`.
- `.venv/bin/python experiments/build_pr81_pr82_henosis_stack_candidate.py --no-pr81-sha-check --no-pr82-sha-check`
  rebuilt six candidates and preserved queued archive SHA-256s.

## Full QRM1 Mask-Dependent Runtime Port Addendum

No remote GPU dispatch was performed. No scorer was invoked.

Implemented the PR82 replay-special QRM1 randmulti branches in
`submissions/robust_current/apply_qzs3_postprocess.py`:

- group 61 `(222,222,4)`: global RGB/channel frame-1 bias.
- group 62 `(222,223,4)`: class-conditioned all-channel frame-1 bias.
- group 63 `(223,222,2)`: class-conditioned channel frame-1 bias.
- group 64 `(223,223,4)`: boundary all-channel frame-1 bias.
- groups 65 and 67 `(223,221,4)`: class-conditioned channel frame-1 bias.
- group 66 `(223,224,4)`: boundary/class-boundary channel frame-1 bias.
- group 68 `(223,219,4)`: width-2 boundary channel/class frame-1 bias.
- group 70 `(223,218,4)`: width-3 boundary channel/class frame-1 bias.
- group 71 `(224,222,4)`: 2x2 tile channel frame-1 bias.

Runtime boundary: mask-conditioned branches now parse as supported but require
charged source masks at apply time. `apply_qzs3_postprocess.py` loads those
masks from the archive directory via the same robust-current mask loader used
by `inflate_renderer.py`; for PR81/QMA9 this resolves the unpacked
`masks.qma9` member. If active mask-conditioned QRM1 groups are present and no
source mask tensor is available, apply fails closed. QRM1 duplicate group ids,
out-of-spec group ids, trailing bytes, and out-of-range replay-special choices
still fail closed.

Candidate reactivation status after local rebuild:

- `pr81_qma9_pr82_qps1_qrm1_all072_randmulti/archive.zip`:
  `232580` bytes, SHA-256
  `05f26b0c47414661ed0c3049f6ef54ae35553b3a759824266d577db802e76fe3`,
  `dispatch_ready_now=true`, synthetic raw-output delta changed `4920856`
  values.
- `pr81_qma9_pr82_qps1_controls_qrm1_all072/archive.zip`:
  `235111` bytes, SHA-256
  `c0f030e2d2efdff53e25e0ad89e053dec0cc1ed2af221434b0145f09c3e46a12`,
  `dispatch_ready_now=true`, synthetic raw-output delta changed `5295342`
  values.

These are exact-eval dispatchable only after a lane claim. They remain
`score_claim=false` and empirical local runtime evidence until exact CUDA auth
eval completes.

Verification:

- `.venv/bin/python -m py_compile submissions/robust_current/apply_qzs3_postprocess.py experiments/build_pr81_pr82_henosis_stack_candidate.py src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py src/tac/tests/test_build_pr81_pr82_henosis_stack_candidate.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_pr81_pr82_henosis_stack_candidate.py src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py -q`
  -> `14 passed, 1 warning`.
- `.venv/bin/python experiments/build_pr81_pr82_henosis_stack_candidate.py --no-pr81-sha-check --no-pr82-sha-check`
  rebuilt the six local PR81/PR82 candidates with all-72 QRM1 parse/apply
  proofs.
