# Native source boundary

The only vendored native source is `submissions/mrs2/range_decoder.c`. Its 100 physical lines implement five-symbol probability-to-frequency conversion, interval selection, arithmetic renormalization and bit reads. The learned prior, online context statistics, renderer, carrier and all learned values stay in their original Python/archive locations.

The recurrence is adapted from the sealed move-53 `runtime/entropy/rc64_backend.c`, whose arithmetic has PR135 lineage. It is not claimed as original compression research. This implementation replaces its wide-integer division with an exactly equivalent cumulative-interval comparison and splits multiplication at bit 31, so it uses only standard C11 integer types. It contains no encoder, model, learned table, video-specific selector, OpenMP, host intrinsics, environment access, network access or external file reads. Python supplies the payload and probability rows and retains ownership of every buffer.

For width W and cumulative count C, with T=2^31, `floor(W*C/T) = (W >> 31)*C + (((W & (T-1))*C) >> 31)`. W<=2^63 and C<=2^31 bound both intermediate products within uint64. The original symbol test `floor(((D+1)*T-1)/W) < C` is exactly `D < floor(W*C/T)`, where D=code-low. This is an integer equivalence, not a floating-point approximation. Probability conversion preserves float32 input, sequential float64 sum, truncation, positive-frequency clamp and first-maximum tie handling.

The library accepts a four-integer state and commits it only after successful group decoding. A native error raises; it does not silently retry a partially consumed stream. An absent/unloadable library emits one clear fallback line and uses the original Python recurrence. The shell removes stale compiler output before attempting its single plain compile line. Only the generated library is a disposable build product; no archive payload is removed.

`SOURCE_SCOPE_AUDIT.json`, the 48-pair retained comparisons, and the cold public raw proof are separate evidence surfaces. Target T4 behavior remains unmeasured here.

<!-- # FORMALIZATION_PENDING: exact integer implementation equivalence and scoped native audit; no score law -->
