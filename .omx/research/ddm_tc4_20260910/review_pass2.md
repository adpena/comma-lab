# TC4 independent review pass 2

Disposition: FINDING in the new validator; all four pass-1 findings are fixed
by source inspection. This is not a clean pass. No reviewer experiments or
scorers ran.

Reviewed hashes:
- maps: `669b852c6a350e8fd4b8b0befe06d5316ff38af2d55983c4b4ce4ad6603be8d7`
- price: `002fcc660f0313a4debef31c061de443a988a8958573cc446343d32f13533193`
- validate_maps: `9894057303ba34db8d374e81104f1c3712c6ae9314a5bb6250e45a668c34cfaa`

## Finding

MEDIUM — `ddm_tc4_validate_maps.py::run` uses only `price.pin("map_controls")`
as the binding. That binding pins price, maps and their dependencies, but not
the validator source itself. Its per-frame resume branch skips prior work when
that binding matches. Editing the future-token mutation or equivalence-check
logic can therefore reuse old receipts and emit `all_passed` without executing
the edited control. Include the validator's own file fact in a stage-local
binding persisted in every frame receipt and RESULT. Parent owns this fix and
the two following clean review passes.

## Verified fixes and remaining proof boundary

Source and inherited stage bindings now cover geometry, runtime files, source
manifests, and control/prepare/fit linkage. Source trace frames verify their
producer binding and actual token equality against the pinned full field.
Requested partial boundaries reject backwards resume, LATEST agrees with the
checkpoint, and arithmetic/objective native builds are bound into saved state.
Encode verifies selection-mask columns and retained fitted weight bytes.
`maps.mix_rows` supplies the shared literal zero-feature no-op to receiver and
offline pricing. The new generation root preserves earlier runs separately.

The validator compares all 190 prefix groups per selected real frame and mutates
unavailable current symbols and saved predictor bins for seven declared groups.
These are implementation controls; neither script claims the controls measure
byte savings or a score. Full n600 twin encode, composite admission, receiver
runtime closure and public raw-output identity remain separate empirical gates.

## Assumption challenge

The validator checks formulas using retained TC3 predictor-bin values, not a
new end-to-end public decoder. This is appropriate for map equivalence but
cannot replace public-entrypoint output identity. The five maps still define a
fixed-bin, old40-frozen formulation; a negative cannot close every causal map.
