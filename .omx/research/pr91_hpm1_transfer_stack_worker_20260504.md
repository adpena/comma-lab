# PR91 HPM1 Transfer Stack Worker - 2026-05-04

Scope: PR91/HPM1 transfer and beat-PR91 candidate design only. No remote GPU
dispatch, no training, and no exact eval submission were performed.

## Source Artifacts

- PR91 intake archive:
  `experiments/results/public_pr91_intake_20260504_worker/archive.zip`
- PR91 archive bytes: `222404`
- PR91 archive SHA-256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- PR91 member `x` bytes: `222304`
- PR91 member `x` SHA-256:
  `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- PR91 anatomy evidence:
  `experiments/results/public_pr91_intake_20260504_worker/pr91_archive_anatomy.json`
- PR91/PR85 segment identity evidence:
  `experiments/results/public_pr91_intake_20260504_worker/pr91_vs_pr85_segment_diff.json`
- PR91 transfer blockers:
  `experiments/results/public_pr91_intake_20260504_worker/pr91_transfer_decisions.json`
- PR85 QRGB action source:
  `experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/action_spec_bias_region.json`

## HPM1 Typed Contract

`src/tac/pr85_bundle.py` now parses PR91 as the existing PR85-family
single-member `x` bundle and types the `mask` segment as `HPM1`.

- bundle format: `pr85_v5_micro_24bit_lengths_fixed_bias_region`
- mask bytes: `145087`
- mask SHA-256:
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- `HPM1` header bytes: `48`
- dimensions: `N=600`, `H=384`, `W=512`
- params: `P=32`, `delta=2`, `ch=64`, `use_spm=1`, `hpac_d_film=8`,
  `ppmd_order=4`
- token stream bytes: `116796`
- token stream SHA-256:
  `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- HPAC model bytes: `28243`
- HPAC model SHA-256:
  `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`
- tail bytes: `0`

The parser also accepts explicit-30 headers when the first segment is `HPM1`,
which is required for local PR91+QRGB side-channel candidates whose
`bias`/`region` Brotli lengths move outside the public v5 fixed-size slots.

## PR91+QRGB Local Candidates

Existing PR85 QRGB builders cannot be reused against PR91 as-is without either
failing source provenance or fabricating PR91 scorer-gradient evidence. The new
local builder avoids both: it consumes explicit PR85 QRGB actions only after
the touched PR91 side-channel stream is byte-identical to PR85 and the action
`source_value` matches the decoded PR91 stream.

- builder:
  `experiments/build_pr91_qrgb_pair_atom_candidates.py`
- planning artifact:
  `experiments/results/pr91_qrgb_pair_atom_candidates_20260504_codex/planning.json`
- score claim: `false`
- dispatch performed: `false`
- remote jobs dispatched: `false`
- scorer gradient claim: `false`
- PR91 scorer gradient consumed: `false`
- dispatch unlocked: `false`

Built local candidates:

| candidate | archive bytes | archive SHA-256 | changed segment | delta vs PR91 |
| --- | ---: | --- | --- | ---: |
| `pr91_hpm1_qrgb_f1_bias_pair_0060` | `222412` | `1086a22e825f0e4fa2a1c777b9be66e92c6db5c7e1501e2e6f0a026e1a1f38e5` | `bias` | `+8` |
| `pr91_hpm1_qrgb_f1_bias_pair_0164` | `222411` | `dbf5cd22e08bc1cbcb69df987689c0561a7495105f1df437a4e76e520831ea6f` | `bias` | `+7` |
| `pr91_hpm1_qrgb_f1_region_pair_0197` | `222411` | `186c9645eb67dc3b2dec9c59c429ad4380bc272204c6852084982359e50fd567` | `region` | `+7` |

Each candidate preserves the PR91 `HPM1` mask bytes and records a non-noop
decoded side-channel semantic change in its manifest.

## Blockers

- PR91 HPM1 local prefix decode currently fails with the recorded constriction
  assertion in
  `experiments/results/public_pr91_intake_20260504_worker/pr91_hpm1_frame0_decode_smoke.json`.
- PR91 has no submitted HPAC compressor/build recipe for reproducing HPM1
  tokens from PR85 masks.
- PR91 runtime fallback is not fail-closed for HPM1 entropy decode failure.
- PR91 range codec does not yet reproduce the recorded PR85 QMA9 token-source
  SHA and must not replace the existing PR85 QMA9 fallback path.
- The PR91+QRGB candidates use explicit-30 headers; they need reviewed PR91
  HPM1 inflate support and local runtime parity before exact eval dispatch.

## Next Dispatch Recommendation

Do not dispatch these PR91+QRGB candidates yet. Next local work should add a
reviewed HPM1 inflate/preflight path, prove explicit-30 HPM1 candidate runtime
parity, and preserve the `source_value`/segment-identity gates. Only after that
should an operator claim a lane with `tools/claim_lane_dispatch.py` and run
exact CUDA auth eval.

## Focused Verification

- `.venv/bin/python -m pytest src/tac/tests/test_pr85_bundle.py src/tac/tests/test_build_pr91_qrgb_pair_atom_candidates.py`
- `.venv/bin/python experiments/build_pr91_qrgb_pair_atom_candidates.py --stdout`
