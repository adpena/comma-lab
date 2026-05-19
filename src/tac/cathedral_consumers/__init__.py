# SPDX-License-Identifier: MIT
"""Auto-discovered cathedral autopilot consumer packages.

Per operator directive 2026-05-19 + Catalog #335 self-protection: every
package in this directory is auto-discovered by
``tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers``
at loop start.

Each package MUST satisfy :class:`tac.cathedral.consumer_contract.CathedralConsumerContract`
OR carry ``# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>`` waiver in its
``__init__.py`` first 30 lines.

THE PARADIGM SHIFT: convention-over-configuration extincts the
orphan-signal-at-cathedral-autopilot bug class permanently. NEW consumers
land here + expose the canonical contract + auto-discovery loop ingests
them WITHOUT manual ranker-cascade edits.

See ``src/tac/cathedral_consumers/README.md`` for the canonical migration
pattern.
"""
