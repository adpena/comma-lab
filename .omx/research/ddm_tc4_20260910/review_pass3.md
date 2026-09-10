# TC4 independent review pass 3

Disposition: CLEAN by source inspection, consecutive clean counter 1/2.
No new material finding in the three reviewed files. No experiments or scorers
were run by this reviewer.

Reviewed hashes:
- maps: `669b852c6a350e8fd4b8b0befe06d5316ff38af2d55983c4b4ce4ad6603be8d7`
- price: `0e9d0e377bb8492675540a8eda29223b1dee6963800f7821656612dfad113d0d`
- validate_maps: `e832ebe688f7c0ce763e150c2d6da4f6c5e7f797b62b45da475dde5ee94f6851`

The validator now includes its own producer fact in every per-frame and final
binding, and verifies both completion flags before reusing a frame. This fixes
the pass-2 finding. The storage cap includes the whole parent arm store, so
earlier retained generations count toward 6 GiB. The four pass-1 fixes remain
present: inherited producer/input binding, partial-boundary checks, native
checkpoint custody, and shared zero-feature literal identity.

This pass traced the source-field/control/prepare/fit/encode chain and its
resume boundaries. Pricing reads real retained move41 symbols and reconstructs
the shipped TC3 A source row; source RC64 identity gates preparation. The KT
counts use unchanged source frequencies and real symbols, independent of fitted
extra weights, so features can be prepared once and shared across standalone
and composite fits. Encoder and receiver both apply `mix_rows` to the selected
feature columns and counted signed-int8 coefficients. All candidate payloads
are retained; full n600 pricing is distinct from the KEEP-only fitting gap.

Assumption challenge: fixed generic bins, frozen old40, and frame-frozen KT
counts define a formulation rather than a universal causal-map family. The
charter's real standalone/composite encode decides this formulation. No score,
public receiver identity, runtime budget, measured RSS, or full slate result is
established by this source review.
