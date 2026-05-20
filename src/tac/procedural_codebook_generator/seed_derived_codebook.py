# SPDX-License-Identifier: MIT
"""tac.procedural_codebook_generator.seed_derived_codebook — 3-PRNG-kind canonical helper.

Per operator NON-NEGOTIABLE 2026-05-20 follow-on per
``.omx/research/procedural_codebook_generator_null_exploit_design_20260520.md``
Top-3 op-routable #1 (Q5 follow-on of the canonical investigation
``.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
Q4 STRUCTURALLY COMPLIANT verdict).

Operationalizes the operator's null-exploit framing on the 16,292 empirical
null-byte candidates landed by the NULL-EXPLOIT PROBE wave (commit
``0b4f370659c5d6fa``; 9.131% of the fec6 frontier 178,417 inner-member
bytes; 100% null on the grammar headers + 249-byte FEC6 selector + 9.00%
null on PR101 lc_v2 source_payload).

This module is the **3-PRNG-kind canonical API** sister of the
already-landed :mod:`tac.procedural_codebook_generator.hash_seed_codebook_generator`
(``emit_seed`` + ``expand_seed_to_codebook`` for the ``uniform_int8``
distribution via numpy stdlib ``PCG64``). Where the sister module
provides a single canonical PCG64 backend with a fixed ``uniform_int8``
distribution, this module provides:

* 3 generator-kind taxonomy (``xorshift`` / ``lcg`` / ``pcg64``) per
  memo §3 design surface (LOC-budget tradeoff per HNeRV parity discipline
  lesson 4: ≤100 LOC inflate.py ideal, ≤200 with waiver)
* Arbitrary numpy ``dtype`` + ``output_shape`` (not just ``uniform_int8``
  on a flat array)
* Byte-stable algorithms implemented in pure Python (no numpy stdlib
  PCG64 dependency for ``xorshift`` / ``lcg``; the ``pcg64`` kind is the
  canonical reference implementation per O'Neill 2014 paper, independent
  of numpy's implementation for cross-implementation parity)
* Inflate-time validator :func:`verify_codebook_from_seed` (sister of
  ``verify_generator_seed_mutation_smoke`` but on the derive-and-compare
  surface rather than the mutate-and-detect-change surface)

Compliance citation chain (per memo Q4 STRUCTURALLY COMPLIANT verdict):

1. **`upstream/evaluate.py` line 63**:
   ``compressed_size = (args.submission_dir / 'archive.zip').stat().st_size``
   — only ``archive.zip`` bytes are charged to the rate term. Seed bytes
   live INSIDE archive.zip + generator code lives in inflate.py
   (legitimate "external library / tool" per upstream README).
2. **Structural distinction from rejected loophole pattern**
   (PR #36 / #38 / #68 / #69 / #78 / #87): the rejected pattern relocates
   score-relevant bytes OUTSIDE archive.zip (base85 literals in inflate.py
   referencing payload data; ``inflate.sh`` reading source video direct);
   this canonical helper KEEPS BOTH the seed bytes AND the derivation
   routine inside the canonical submission package. This is
   *sensitivity-aware compression*, not *payload smuggling*.
3. **Catalog #213** ``check_comma2k19_downloads_route_through_canonical_cache``:
   sister pattern for OOD-derived constants — same compliance posture
   (seed bytes inside archive; canonical helper inside inflate runtime).
4. **Catalog #272** ``check_substrate_distinguishing_feature_integration_contract``:
   the seed bytes MUST produce frame-level changes (mutate seed →
   re-inflate → frames change). Empty seed slots fail this check; the
   sister ``verify_generator_seed_mutation_smoke`` is the structural
   byte-mutation surface; this module's :func:`verify_codebook_from_seed`
   is the inflate-time derive-and-compare surface.
5. **Catalog #318** master-gradient raw-byte-authority guard: the typed
   ``CandidateModificationSpec`` discipline is preserved — this helper
   is the *forward* (build) direction; the null-byte probe is the
   *inverse* (identify) direction. The two together close the
   procedural-codebook class on both compose-time and inflate-time
   surfaces.

Sister of:

* :mod:`tac.cathedral_consumers.procedural_codebook_generator_consumer`
  (Catalog #335 auto-discovered consumer; surfaces seed-derived codebook
  candidate routing as Tier A observability metadata per Catalog #341)
* :mod:`tac.cathedral_consumers.null_byte_codebook_candidate_consumer`
  (sister identifier — null-byte probe identifies WHERE; this helper
  replaces those bytes WITH seed-derived bytes)
* :mod:`tac.canonical_equations.null_space_byte_fraction` (Catalog #344
  canonical equation ``master_gradient_null_space_byte_fraction_v1``)
* :mod:`tac.canonical_equations.procedural_codebook_savings` (sister
  canonical equation ``procedural_codebook_from_seed_compression_savings_v1``
  registering predicted ΔS for procedural-codebook replacement candidates)

Public API:

* :func:`derive_codebook_from_seed` — forward direction (build bytes from
  seed at trainer / compose-time)
* :func:`verify_codebook_from_seed` — inflate-time validator (derive bytes
  from seed + compare byte-for-byte to expected)
* :data:`SUPPORTED_GENERATOR_KINDS` — frozenset of canonical generator
  kinds (xorshift / lcg / pcg64)

Generator-kind taxonomy:

* ``"xorshift"`` (Marsaglia 2003 xorshift64*): ~30 LOC; fastest;
  deterministic across platforms; suitable for general-purpose
  codebook derivation
* ``"lcg"`` (linear congruential): ~25 LOC; smallest implementation;
  well-studied; lower entropy but adequate for codebook-derivation use
  cases where downstream training has absorbed the seed's bias
* ``"pcg64"`` (O'Neill 2014 PCG XSL-RR 128/64): ~80 LOC; highest entropy;
  statistically rigorous; canonical reference implementation pinned to
  the original paper

All three implement ``(seed_bytes, n_output_bytes) -> bytes`` with
byte-stable deterministic decode (same seed + same n_bytes ALWAYS produces
identical bytes across runs / threads / Python versions on the same
architecture; cross-architecture stability is endian-stable via explicit
little-endian byte ordering).

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #341 Tier
A: this helper is **observability-neutral** for the cathedral consumer
surface (the consumer per
:mod:`tac.cathedral_consumers.procedural_codebook_generator_consumer`
emits ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[predicted]"``). The actual score-mutating mechanism is the
per-substrate substrate-trainer + inflate-runtime integration, which
remains gated by Catalog #325 per-substrate symposium + Catalog #324
post-training Tier-C validation per CLAUDE.md "Substrate scaffolds MUST
be COMPLETE or RESEARCH-ONLY" non-negotiable.
"""
from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np

__all__ = [
    "SUPPORTED_GENERATOR_KINDS",
    "DEFAULT_GENERATOR_KIND",
    "MAX_OUTPUT_BYTES",
    "ProceduralCodebookGeneratorError",
    "derive_codebook_from_seed",
    "verify_codebook_from_seed",
]


SUPPORTED_GENERATOR_KINDS = frozenset({"xorshift", "lcg", "pcg64"})
"""Canonical generator-kind taxonomy. Frozen for byte-stability guarantee."""

DEFAULT_GENERATOR_KIND: Literal["xorshift", "lcg", "pcg64"] = "pcg64"
"""Default generator kind. PCG64 has highest entropy + statistical rigor
(O'Neill 2014); xorshift / lcg are smaller LOC alternatives for inflate
runtime LOC budget per HNeRV parity discipline lesson 4 (≤100 LOC ideal,
≤200 LOC waivered)."""

MAX_OUTPUT_BYTES = 64 * 1024 * 1024
"""Defensive ceiling — refuse codebook derivations larger than 64 MiB.
Sister of CLAUDE.md "Apples-to-apples evidence discipline" — protects
against accidental tensor-shape errors that would otherwise materialize
multi-GB byte buffers."""


class ProceduralCodebookGeneratorError(ValueError):
    """Raised on invalid inputs to the canonical helper.

    Sister of :class:`tac.canonical_equations.InvalidEquationError`,
    :class:`tac.wyner_ziv_deliverability.contract.DeliverabilityProofValidationError`.
    """


def _validate_inputs(
    seed_bytes: bytes,
    output_shape: tuple[int, ...],
    dtype: np.dtype,
    generator_kind: str,
) -> int:
    """Validate inputs; return total byte count required.

    Per Catalog #287 placeholder-rationale rejection sister discipline +
    CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" — every
    invalid input fails fast with a substantive error message naming the
    offending field.
    """
    if not isinstance(seed_bytes, (bytes, bytearray, memoryview)):
        raise ProceduralCodebookGeneratorError(
            f"seed_bytes must be bytes-like; got {type(seed_bytes).__name__}"
        )
    seed_view = bytes(seed_bytes)
    if len(seed_view) == 0:
        raise ProceduralCodebookGeneratorError(
            "seed_bytes must be non-empty (empty seed = zero entropy)"
        )
    if not isinstance(output_shape, tuple):
        raise ProceduralCodebookGeneratorError(
            f"output_shape must be tuple[int, ...]; got {type(output_shape).__name__}"
        )
    if len(output_shape) == 0:
        raise ProceduralCodebookGeneratorError(
            "output_shape must be non-empty tuple (scalar codebooks not supported)"
        )
    for axis, dim in enumerate(output_shape):
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ProceduralCodebookGeneratorError(
                f"output_shape[{axis}] must be int; got {type(dim).__name__}"
            )
        if dim <= 0:
            raise ProceduralCodebookGeneratorError(
                f"output_shape[{axis}] must be positive; got {dim}"
            )

    np_dtype = np.dtype(dtype)
    n_elements = 1
    for dim in output_shape:
        n_elements *= dim
    n_bytes = n_elements * np_dtype.itemsize
    if n_bytes > MAX_OUTPUT_BYTES:
        raise ProceduralCodebookGeneratorError(
            f"output requires {n_bytes} bytes which exceeds MAX_OUTPUT_BYTES "
            f"({MAX_OUTPUT_BYTES}); refusing to materialize multi-MiB buffer "
            "without explicit operator opt-in (raise MAX_OUTPUT_BYTES if intentional)"
        )

    if generator_kind not in SUPPORTED_GENERATOR_KINDS:
        raise ProceduralCodebookGeneratorError(
            f"generator_kind {generator_kind!r} not in canonical set "
            f"{sorted(SUPPORTED_GENERATOR_KINDS)}; see "
            "tac.procedural_codebook_generator.seed_derived_codebook.SUPPORTED_GENERATOR_KINDS"
        )

    return n_bytes


# ---------------------------------------------------------------------------
# Generator kind: xorshift (Marsaglia 2003 xorshift64*)
# ---------------------------------------------------------------------------
#
# Algorithm reference: Marsaglia, G. (2003), "Xorshift RNGs", Journal of
# Statistical Software 8(14). The xorshift64* variant multiplies the raw
# xorshift64 output by 0x2545F4914F6CDD1D for improved low-bit statistics.
#
# State: single uint64. Period: 2^64 - 1 (all states except 0).
# Output: little-endian uint64 → 8 bytes per step.
#
# Seed derivation: take first 8 bytes of seed_bytes (zero-padded if
# shorter); reject all-zero seed by setting initial state to a deterministic
# nonzero salt derived from sha256(seed_bytes) first 8 bytes.

_XORSHIFT64_STAR_MULTIPLIER = 0x2545F4914F6CDD1D
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF


def _xorshift_seed_to_state(seed_bytes: bytes) -> int:
    """Derive 64-bit non-zero initial state from arbitrary-length seed.

    For seeds longer than 8 bytes, we hash via sha256 to preserve entropy
    from the full seed (otherwise two seeds differing only in trailing
    bytes would map to the same xorshift state and produce identical
    output).
    """
    if len(seed_bytes) == 8:
        state = int.from_bytes(seed_bytes, byteorder="little", signed=False)
    elif len(seed_bytes) < 8:
        padded = seed_bytes + bytes(8 - len(seed_bytes))
        state = int.from_bytes(padded, byteorder="little", signed=False)
    else:
        # len(seed_bytes) > 8 — sha256 to absorb full entropy
        digest = hashlib.sha256(seed_bytes).digest()
        state = int.from_bytes(digest[:8], byteorder="little", signed=False)
    if state == 0:
        # Salt with sha256 to guarantee non-zero state (xorshift period
        # excludes 0). Deterministic given seed_bytes.
        digest = hashlib.sha256(seed_bytes).digest()
        state = int.from_bytes(digest[:8], byteorder="little", signed=False)
        if state == 0:  # defense in depth (sha256 collision to zero)
            state = 0x1
    return state


def _xorshift_generate(seed_bytes: bytes, n_bytes: int) -> bytes:
    """Marsaglia 2003 xorshift64* — deterministic byte-stable PRNG."""
    state = _xorshift_seed_to_state(seed_bytes)
    output = bytearray(n_bytes)
    offset = 0
    while offset < n_bytes:
        state ^= (state << 13) & _UINT64_MASK
        state ^= (state >> 7) & _UINT64_MASK
        state ^= (state << 17) & _UINT64_MASK
        out = (state * _XORSHIFT64_STAR_MULTIPLIER) & _UINT64_MASK
        chunk = out.to_bytes(8, byteorder="little", signed=False)
        take = min(8, n_bytes - offset)
        output[offset : offset + take] = chunk[:take]
        offset += take
    return bytes(output)


# ---------------------------------------------------------------------------
# Generator kind: lcg (linear congruential)
# ---------------------------------------------------------------------------
#
# Algorithm reference: Knuth TAOCP Vol 2 Chapter 3.2.1 + Numerical Recipes
# 3rd Ed §7.1.1 "Quick and Dirty Generators". We use the MMIX constants
# (Knuth 1998): a = 6364136223846793005, c = 1442695040888963407.
# Modulus: 2^64 (implicit via uint64 mask).
#
# State: single uint64. Period: 2^64.
# Output: high 32 bits of state per step (low bits of LCG are
# statistically poor — Knuth §3.2.1.3 Theorem A); 4 bytes per step.
#
# Seed derivation: identical to xorshift (first 8 bytes of seed, sha256
# salt if all-zero).

_MMIX_MULTIPLIER = 6364136223846793005
_MMIX_INCREMENT = 1442695040888963407


def _lcg_seed_to_state(seed_bytes: bytes) -> int:
    """Derive 64-bit initial state from arbitrary-length seed (zero OK).

    For seeds longer than 8 bytes, we hash via sha256 to preserve entropy
    from the full seed (otherwise two seeds differing only in trailing
    bytes would map to the same LCG state and produce identical output).
    """
    if len(seed_bytes) == 8:
        return int.from_bytes(seed_bytes, byteorder="little", signed=False)
    if len(seed_bytes) < 8:
        padded = seed_bytes + bytes(8 - len(seed_bytes))
        return int.from_bytes(padded, byteorder="little", signed=False)
    # len(seed_bytes) > 8 — sha256 to absorb full entropy
    digest = hashlib.sha256(seed_bytes).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _lcg_generate(seed_bytes: bytes, n_bytes: int) -> bytes:
    """Knuth MMIX LCG — deterministic byte-stable PRNG (high 32 bits per step)."""
    state = _lcg_seed_to_state(seed_bytes)
    output = bytearray(n_bytes)
    offset = 0
    while offset < n_bytes:
        state = (state * _MMIX_MULTIPLIER + _MMIX_INCREMENT) & _UINT64_MASK
        # Take high 32 bits (low bits of LCG have short period for low-order bits)
        high32 = (state >> 32) & 0xFFFFFFFF
        chunk = high32.to_bytes(4, byteorder="little", signed=False)
        take = min(4, n_bytes - offset)
        output[offset : offset + take] = chunk[:take]
        offset += take
    return bytes(output)


# ---------------------------------------------------------------------------
# Generator kind: pcg64 (O'Neill 2014 PCG XSL-RR 128/64)
# ---------------------------------------------------------------------------
#
# Algorithm reference: O'Neill, M.E. (2014), "PCG: A Family of Simple Fast
# Space-Efficient Statistically Good Algorithms for Random Number
# Generation", Harvey Mudd College Technical Report HMC-CS-2014-0905.
# Reference C implementation: https://www.pcg-random.org/download.html
# (pcg_basic.c). We implement the PCG-XSL-RR variant with 128-bit state
# and 64-bit output, matching the canonical pcg64 reference.
#
# State: (state: uint128, inc: uint128). Period: 2^128.
# Output: 64-bit per step via XSL-RR permutation.
#
# Seed derivation: first 32 bytes of sha256(seed_bytes) provides 128-bit
# state + 128-bit inc; inc is forced odd per pcg_setseq_128_step requirement.
#
# NOTE: This implementation is INDEPENDENT of numpy's PCG64 (which numpy
# wraps in its own RNG state machine). We implement the canonical PCG-XSL-RR
# 128/64 directly from the paper for cross-implementation byte-stability
# (the sister tac.procedural_codebook_generator.hash_seed_codebook_generator
# uses numpy's PCG64 wrapper; this module uses the raw algorithm).

_PCG_DEFAULT_MULTIPLIER_128 = (2549297995355413924 << 64) | 4865540595714422341
_UINT128_MASK = (1 << 128) - 1


def _pcg64_seed_to_state_and_inc(seed_bytes: bytes) -> tuple[int, int]:
    """Derive 128-bit state + 128-bit (odd) inc from arbitrary-length seed.

    Per pcg_basic.c reference: state and inc are derived from a 256-bit
    seed; inc is forced to be odd (LSB = 1) per PCG's structural
    requirement for full-period guarantees.
    """
    # sha256 = 32 bytes; first 16 → state, next 16 → inc
    digest = hashlib.sha256(seed_bytes).digest()
    state = int.from_bytes(digest[:16], byteorder="little", signed=False)
    inc = int.from_bytes(digest[16:32], byteorder="little", signed=False)
    inc |= 1  # PCG requires odd inc for full period
    return state, inc


def _pcg64_step(state: int, inc: int) -> tuple[int, int]:
    """Single LCG step: state = state * multiplier + inc (mod 2^128)."""
    state = (state * _PCG_DEFAULT_MULTIPLIER_128 + inc) & _UINT128_MASK
    return state, inc


def _pcg64_output(state: int) -> int:
    """PCG-XSL-RR 128/64 output permutation.

    Reference: pcg_basic.c pcg_output_xsl_rr_128_64 from
    https://www.pcg-random.org/download.html. The XSL-RR transformation
    XORs the high 64 bits with the low 64 bits (XSL = xorshift-low), then
    rotates the result by the top 6 bits of the state.
    """
    rot = (state >> 122) & 0x3F  # top 6 bits = rotation amount
    high = (state >> 64) & _UINT64_MASK
    low = state & _UINT64_MASK
    xsl = (high ^ low) & _UINT64_MASK
    # Rotate-right by `rot` bits (canonical PCG output permutation)
    rotated = ((xsl >> rot) | (xsl << ((-rot) & 63))) & _UINT64_MASK
    return rotated


def _pcg64_generate(seed_bytes: bytes, n_bytes: int) -> bytes:
    """O'Neill 2014 PCG-XSL-RR 128/64 — deterministic byte-stable PRNG."""
    state, inc = _pcg64_seed_to_state_and_inc(seed_bytes)
    output = bytearray(n_bytes)
    offset = 0
    while offset < n_bytes:
        # PCG advances state THEN outputs the *new* state's permutation
        # per the canonical reference (pcg_basic.c). We follow the same
        # convention: step first, then permute the new state.
        state, inc = _pcg64_step(state, inc)
        out64 = _pcg64_output(state)
        chunk = out64.to_bytes(8, byteorder="little", signed=False)
        take = min(8, n_bytes - offset)
        output[offset : offset + take] = chunk[:take]
        offset += take
    return bytes(output)


# ---------------------------------------------------------------------------
# Generator dispatch
# ---------------------------------------------------------------------------

_GENERATORS = {
    "xorshift": _xorshift_generate,
    "lcg": _lcg_generate,
    "pcg64": _pcg64_generate,
}


def derive_codebook_from_seed(
    seed_bytes: bytes,
    output_shape: tuple[int, ...],
    dtype: np.dtype | type,
    generator_kind: Literal["xorshift", "lcg", "pcg64"] = DEFAULT_GENERATOR_KIND,
) -> np.ndarray:
    """Deterministically derive codebook ndarray from seed bytes via stateless PRNG.

    Per memo Q4 STRUCTURALLY COMPLIANT verdict (canonical investigation memo
    2026-05-18): seed bytes live INSIDE archive.zip; codebook bytes derived
    at inflate-time. Distinguishable from rejected loophole-class pattern
    (PR #68 base85 literal in inflate.py / out-of-archive bytes).

    Compliance citation chain:

    * PR #68 maintainer rejection precedent: loophole-class out-of-archive
      bytes REJECTED
    * ``upstream/evaluate.py:63`` rate-charging: bytes-inside-archive ARE
      charged
    * Catalog #213 Comma2k19 canonical helper pattern: seeded derivation
      from in-archive bytes
    * Catalog #272 distinguishing-feature integration contract: seed bytes
      MUST produce frame-level changes (mutate seed → inflate → frames
      change); the byte-mutation smoke is the operator-facing proof

    Args:
        seed_bytes: small in-archive byte slice (typically 16-64 bytes).
            Must be non-empty bytes-like.
        output_shape: shape of derived tensor. Must be non-empty tuple of
            positive ints. Byte length = ``prod(shape) * dtype.itemsize``.
        dtype: numpy dtype for output (e.g. ``np.uint8``, ``np.int8``,
            ``np.float16``, ``np.float32``).
        generator_kind: PRNG family — ``"xorshift"`` (fastest, 30 LOC),
            ``"lcg"`` (smallest, 25 LOC), ``"pcg64"`` (highest entropy,
            80 LOC; default per :data:`DEFAULT_GENERATOR_KIND`).

    Returns:
        Codebook ndarray of shape ``output_shape`` and dtype ``dtype``
        whose underlying bytes (``.tobytes()``) are reproducibly derived
        from ``seed_bytes`` via the specified PRNG. Same (seed, shape,
        dtype, kind) ALWAYS produces identical bytes across runs /
        threads / Python versions on the same architecture; cross-
        architecture stability is endian-stable via explicit little-
        endian byte ordering in every generator.

    Raises:
        :class:`ProceduralCodebookGeneratorError`: invalid inputs (empty
            seed / non-tuple shape / zero or negative dim / unsupported
            generator_kind / output exceeds :data:`MAX_OUTPUT_BYTES`).

    Example:
        >>> import numpy as np
        >>> seed = bytes(range(32))
        >>> lut = derive_codebook_from_seed(
        ...     seed_bytes=seed,
        ...     output_shape=(1024, 4),
        ...     dtype=np.uint8,
        ...     generator_kind="pcg64",
        ... )
        >>> lut.shape
        (1024, 4)
        >>> str(lut.dtype)
        'uint8'
        >>> lut2 = derive_codebook_from_seed(
        ...     seed_bytes=seed,
        ...     output_shape=(1024, 4),
        ...     dtype=np.uint8,
        ...     generator_kind="pcg64",
        ... )
        >>> bool((lut == lut2).all())
        True
    """
    # Validate seed type first so non-bytes inputs (e.g. str) produce a
    # canonical ProceduralCodebookGeneratorError rather than a raw
    # TypeError from bytes(...) coercion.
    if not isinstance(seed_bytes, (bytes, bytearray, memoryview)):
        raise ProceduralCodebookGeneratorError(
            f"seed_bytes must be bytes-like; got {type(seed_bytes).__name__}"
        )
    seed_blob = bytes(seed_bytes)
    np_dtype = np.dtype(dtype)
    n_bytes = _validate_inputs(
        seed_bytes=seed_blob,
        output_shape=output_shape,
        dtype=np_dtype,
        generator_kind=generator_kind,
    )
    generator = _GENERATORS[generator_kind]
    raw = generator(seed_blob, n_bytes)
    array = np.frombuffer(raw, dtype=np_dtype)
    # Copy so caller can write to the returned array (np.frombuffer view is RO)
    return array.reshape(output_shape).copy()


def verify_codebook_from_seed(
    seed_bytes: bytes,
    expected_codebook_bytes: bytes,
    output_shape: tuple[int, ...],
    dtype: np.dtype | type,
    generator_kind: Literal["xorshift", "lcg", "pcg64"] = DEFAULT_GENERATOR_KIND,
) -> bool:
    """Inflate-time validation: derive codebook from seed + compare byte-for-byte.

    Returns ``True`` iff the derived bytes (per
    :func:`derive_codebook_from_seed` with identical args) equal
    ``expected_codebook_bytes`` byte-for-byte.

    Used by inflate.py to verify procedurally-derived codebooks at runtime
    when the substrate ships BOTH the seed + the trained-codebook bytes
    (transitional / belt-and-suspenders deployments) — eventually the
    trained-codebook bytes are removed from archive.zip once the
    Catalog #272 byte-mutation smoke confirms seed-derived bytes preserve
    score.

    Args:
        seed_bytes: same seed used to derive expected_codebook_bytes.
        expected_codebook_bytes: the canonical reference bytes (typically
            from the trained substrate before seed-search). Must be length
            ``prod(output_shape) * dtype.itemsize``.
        output_shape: shape used during derivation.
        dtype: dtype used during derivation.
        generator_kind: generator kind used during derivation.

    Returns:
        ``True`` iff bytes match; ``False`` otherwise. Does NOT raise on
        mismatch (callers gate on the boolean return; inflate runtime is
        expected to log mismatches as INFO, not crash).

    Raises:
        :class:`ProceduralCodebookGeneratorError`: invalid inputs to
            :func:`derive_codebook_from_seed` (propagates from validation).
    """
    derived = derive_codebook_from_seed(
        seed_bytes=seed_bytes,
        output_shape=output_shape,
        dtype=dtype,
        generator_kind=generator_kind,
    )
    derived_bytes = derived.tobytes()
    return derived_bytes == expected_codebook_bytes
