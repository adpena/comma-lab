"""PR130 semantic renderer lift for ddm_mx1.

The files under :mod:`tac.pr130_lift.lifted` are borrowed PR130 source with
per-file accounting headers.  The MLX port in this package is ours only as a
substrate adaptation; the renderer recipe remains attributed to PR130.
"""

from __future__ import annotations

__all__ = [
    "SOURCE_REPO_HEAD",
    "SOURCE_REPO_ROOT",
]

SOURCE_REPO_ROOT = "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo"
SOURCE_REPO_HEAD = "2f94596bb0136d342254022a5c9584756eae0468"
