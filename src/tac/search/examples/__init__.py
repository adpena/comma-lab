# SPDX-License-Identifier: MIT
"""Canonical example search strategies for the tac.search namespace.

Importing this package registers five example strategies (one per builder)
into the namespace registry so downstream consumers can validate the
decorator + composition surface without needing external libraries
installed. The example bodies are intentionally TOY (synthetic objectives
+ fixed bytes_added) so the strategies run in <1ms each — real consumers
replace the body with substrate-specific logic.

Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — every claim in the
docstrings is backed by an executable body.
"""

from __future__ import annotations

from tac.search.examples.example_searches import (  # noqa: F401
    random_search_baseline_example,
    register_example_searches,
)

__all__ = [
    "random_search_baseline_example",
    "register_example_searches",
]
