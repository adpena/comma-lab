# Exact integer geometry port

The new native component is a source-derived port of sealed `rlc1_geometry.c`
(sha256 414cc13dfdf597ca...); it is not a new compression method. The state layout,
row/slot iteration, moment updates, ambiguity checks, Q16 arithmetic, floor
division, integer square root and nearest-even distance bins are unchanged.
The only arithmetic representation change replaces the compiler-specific signed
128-bit integer with two unsigned 64-bit limbs, including explicit signed
comparison, negation, multiplication and floor division. No floating arithmetic
or video-derived constant was introduced. All 17 configuration values are parsed
from the counted 19-byte archive section and checked in Python.

For the supported domain, the weighted window count is at most 5,120. Its y
variance is at most 15^2/4, giving determinant <= 1,474,560,000. The x range in a
slot is at most 63, bounding the covariance residual by
38,355,131,695,104,000,000 (66 bits). Multiplication by 255 and 2^32 needs at most
106 bits; accepted residuals give a square-root argument below 2^48. Divisors
are positive because the valid branch requires positive count and determinant.
These bounds exclude signed overflow in the original 128-bit expressions and
in the portable implementation. Narrow int64 residual arithmetic would not be
an equivalent implementation.

The Python class preserves the complete causal-group and symbol validation,
keeps its output plane for end-frame consistency, and owns/frees one C handle.
Missing or unloadable native code selects the existing Python class with a
stderr diagnostic. Live allocation failure raises rather than silently using a
partially advanced state. The public shell removes all three stale libraries
before compiling with the declared C11 and floating-point flags.

Two independent source-review passes found no defects in this delta or the
proof/bootstrap/smoke helper compatibility. Reviewer `geometry_review` ran
strict C11 syntax and Python AST checks, not the heavy proofs. Root separately
reviewed the delta and runs the byte proofs. These roles are not represented as
a fresh-reader public review: MAIN still owns that review.

Compile verification: all three C files pass `cc -std=c11 -pedantic-errors
-Wall -Wextra -Werror -fsyntax-only`. Geometry is 145 physical lines. Public
source hashes live in SOURCE_MANIFEST.json; the geometry hash is
ac99ab49efbc9fce9bab177b70a62b8e32b576caea16a03d78326b61bf9a0e63.

<!-- # FORMALIZATION_PENDING: implementation arithmetic bounds and source audit -->
