"""Lab-facing reverse-engineering helpers.

The reusable curation logic lives in :mod:`tac.reverse_engineering_curation`
so ``tac`` does not depend on the comma-lab operations package. This module is
the stable operator-facing import surface for comma-lab tools and reports.
"""

from __future__ import annotations

from tac.reverse_engineering_curation import *  # noqa: F403
