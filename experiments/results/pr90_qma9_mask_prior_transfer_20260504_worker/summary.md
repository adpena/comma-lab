# PR90 QMA9 Mask-Prior Transfer Analysis

Local-only artifact. No training, scorer run, remote dispatch, or score claim.

## Decision

- Status: `ready_for_local_archive_builder_after_runtime_port`
- Implementable next archive builder: `True`
- Fixed PR85 runtime preserved: `False`
- Reason: PR90 topband stream decodes to the exact PR85 mask tensor; a builder can replace only the mask segment after adding an explicit STBM decoder path.

## Exact Mask Parity

- PR90/PR85 decoded mask equal: `True`
- Diff pixels: `0`
- PR90 render-order SHA-256: `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`
- PR85 render-order SHA-256: `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`

## Top Candidate

- Policy: `pr90_stbm1br_lossless_pr85_mask_recode`
- Candidate mask bytes: `152439`
- Delta mask bytes: `-6572`
- Estimated archive bytes: `229756`
- Rate score delta if components unchanged: `-0.004376025039918911`

## Blocker

The fixed PR85 runtime cannot consume STBM1BR. The next builder must include a reviewed runtime decoder port and parity gates.
