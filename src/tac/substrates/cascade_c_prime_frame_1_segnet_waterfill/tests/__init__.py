# SPDX-License-Identifier: MIT
"""Tests for the cascade_c_prime_frame_1_segnet_waterfill substrate scaffold.

Per Catalog #287/#323 canonical Provenance: tests assert structural invariants
of the scaffold (substrate contract validation + per-pair routing argmin +
archive roundtrip + byte-mutation smoke per Catalog #139).

The MLX-local smoke at .omx/research/cascade_c_prime_artifacts_20260526/ is
the operator-facing reproducer; this tests/ subdir asserts the substrate
package's structural invariants for the pytest suite.
"""
