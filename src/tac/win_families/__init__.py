# SPDX-License-Identifier: MIT
"""Canonical engines for the five measured WIN FAMILIES.

Each module here is the standardised form of a pattern that several arms
independently rebuilt.  The inventory, the canonical-vs-unique decision per family,
and the adoption rows live in
``.omx/research/ddm_cw1_win_family_canonicalization_20260819.md``.

======  ====================================  ======================================
family  module                                the pattern it standardises
======  ====================================  ======================================
F1      :mod:`~tac.win_families.realized_acceptance`   propose -> realize -> accept on
                                              the REAL decode only
F2      :mod:`~tac.win_families.terminal_compile`      the downstream-refit pipeline
                                              after an upstream edit lands
F3      :mod:`~tac.win_families.container_optimizer`   declared-deterministic container
                                              search at the ARCHIVE layer
F4      :mod:`~tac.win_families.model_axis`            probability-model recoding with
                                              deflated-reservoir accounting
F5      :mod:`tac.local_contest_instruments`           local DALI-lineage verdicts on
                                              either scored axis
======  ====================================  ======================================

F5 deliberately sits at ``tac.local_contest_instruments`` rather than inside this
package: it composes :mod:`tac.gt_lineage` and :mod:`tac.contest_score`, is needed by
arms that use none of the other four engines, and belongs beside the modules it
composes.
"""

from __future__ import annotations

__all__ = ["container_optimizer", "model_axis", "realized_acceptance", "terminal_compile"]
