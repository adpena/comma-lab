# SPDX-License-Identifier: MIT
"""Tests for tac.training_curriculum package.

Per CLAUDE.md "Production-hardened dispatch optimization protocol" + Catalog
#229 (premise-verification-before-edit) + Catalog #305 (Observability surface),
every helper in :mod:`tac.training_curriculum` MUST have a sister test that
exercises (a) happy-path construction + invocation, (b) every validation
invariant in ``__post_init__`` / argument checks, (c) edge cases that the
helper documents in its cargo-cult audit section.

The test surface IS part of the helper's observability contract — readers can
inspect these tests to understand the helper's intended usage WITHOUT reading
the implementation source.
"""
