"""PR130 semantic renderer lift for ddm_mx1.

The files under :mod:`tac.pr130_lift.lifted` are borrowed PR130 source with
per-file accounting headers.  The MLX port in this package is ours only as a
substrate adaptation; the renderer recipe remains attributed to PR130.
"""

from __future__ import annotations

__all__ = [
    "LIFTED_AT_HEAD",
    "SOURCE_REPO_HEAD",
    "SOURCE_REPO_ROOT",
]

SOURCE_REPO_ROOT = "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo"

# TWO DIFFERENT QUANTITIES — do not collapse them again.
#
# LIFTED_AT_HEAD is the commit at which the files under `lifted/` were copied.
# Each header separately names SOURCE_REPO_HEAD as the commit whose original
# bytes its source_sha256 authenticates, plus an explicit adaptation id.  The
# custody test reconstructs every current body from those pinned original bytes
# and refuses any edit outside the enumerated adaptations.  LIFTED_AT_HEAD is a
# fact about when OUR copies were made and only changes when we re-lift.
#
# SOURCE_REPO_HEAD is the CURRENT state of the actual PR130 repo, which we also
# execute directly for the files we did NOT lift (hpac_self_compress.py,
# integer_model_io.py, pack_hpac_self_compress.py). It advances whenever that
# repo does. A single constant served both roles until 2026-08-08, when hb2's
# HPAC deploy-bounds fix (e34f31bc) advanced the repo and the pin still read
# 2f94596 — the record described our copies while being read as the executed
# source. Same value, two meanings, one of them silently wrong.
LIFTED_AT_HEAD = "2f94596bb0136d342254022a5c9584756eae0468"
SOURCE_REPO_HEAD = "e34f31bc4969042c0051ac81aa3c56884419a231"
