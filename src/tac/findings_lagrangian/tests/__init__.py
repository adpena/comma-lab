# SPDX-License-Identifier: MIT
"""Phase 1.A canonical tests for `tac.findings_lagrangian` (TRACK A).

Per Carmack ULTRA-MVP discipline (slot 20-second-supplemental Q9 AMEND
verbatim *"Phase 1.A = ONE equation only"*): the Phase 1.A test suite
covers the canonical Phase 1.A equation — closed-form Gaussian conjugate
posterior update per `tac.findings_lagrangian.posterior`. Sister modules
(info_gain / weights / partition / interpretability / lagrangian /
action_selector / unified) are covered by Phase 1.B + Phase 1.C tests
landed in successor subagents.
"""
