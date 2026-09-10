# TC4 independent review pass 4

Disposition: CLEAN by source inspection, consecutive clean counter 2/2.
No new material finding. The source hashes are unchanged from pass 3:
- maps: `669b852c6a350e8fd4b8b0befe06d5316ff38af2d55983c4b4ce4ad6603be8d7`
- price: `0e9d0e377bb8492675540a8eda29223b1dee6963800f7821656612dfad113d0d`
- validate_maps: `e832ebe688f7c0ce763e150c2d6da4f6c5e7f797b62b45da475dde5ee94f6851`

This pass changed lens to categorical dimensions, numerical scaling, wire
layout, map edge cases, and crash recovery. The five map alphabets fit their
declared table levels, including unknown sentinels; 64-column and 64-row
boundaries exclude unavailable neighbours. The complete and prefix maps use
the same saved above-row event bin rather than a future-refitted boundary.
Per-class mixture weights are serialized in the same class-major column order
used by `mix_rows`. The C objective's feature scale ln(2)/1024 agrees with the
full continuous-price formula; signed-int8 weights use the inherited /32
receiver scale. The projected box gap remains a KEEP-only continuous statement.

Counts and expected counts are computed from the unchanged source frequencies
and decoded symbols, with frame-frozen tables. Integer accumulation has ample
int64 range at 117,964,800 symbols. A zero feature row returns its original
float32 probabilities in both paths. Final arithmetic envelopes, raw streams,
weighted riders and archives are all persisted for both encoders, and actual
archive member parse-back is checked. A source edit, native build change,
inherited stage mismatch or inconsistent resume frame refuses reuse.

Assumption challenge: the public receiver will recompute the maps from its own
decoded prefix. The offline feature cache is an encoder convenience and cannot
become free runtime side information. These three files do not yet wire or
prove that public runtime; only the separately required public-entrypoint
identity can close that remaining boundary. Two clean source reviews do not
assert measured n600 savings, measured memory/runtime, a seal, or a score.

Reviewer wrote only review_pass1.md through review_pass4.md. Parent ddm_tc4
owns review-tracker marking, final-generation empirical controls, serializer
landing, actual five-map/composite pricing and any receiver seal.
