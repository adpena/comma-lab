# SPDX-License-Identifier: MIT
"""A1 substrate runtime adapter.

This package exposes the committed A1 submission runtime through the canonical
``tac.substrates.<id>.inflate`` interface so composed substrates can delegate
without inventing a second A1 implementation.
"""

__all__ = ["inflate"]
