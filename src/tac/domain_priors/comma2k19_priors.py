# SPDX-License-Identifier: MIT
"""Comma2k19 dashcam priors — wraps Comma2k19LocalCache + dataset metadata.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.1: this wrapper exposes the
Comma2k19 dataset's OOD dashcam priors via the canonical
``tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache``
helper per Catalog #213 (forbids raw URL fetches outside the canonical
cache).

The Comma2k19 dataset is the canonical OOD dashcam corpus for our domain.
Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L1
+ Catalog #209 ``check_no_contest_video_leakage_in_distillation_callers``:
domain priors derived from Comma2k19 MUST NOT leak into a path that
contaminates the contest video (``upstream/videos/0.mkv``). The priors
returned by this wrapper are EXPLICITLY tagged as OOD-derived priors
(NOT contest-video signal), so consumers can route them through the
correct gradient path.

Cross-references:
  * CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4
  * Catalog #209 contest-video-leakage non-negotiable
  * Catalog #210 DP1 codebook provenance pattern (this module inherits the
    license_spdx="MIT" + dataset_provenance="commaai/comma2k19" propagation)
  * Catalog #213 Comma2k19 canonical local-chunk cache
  * ``tac.substrates.pretrained_driving_prior.local_chunk_cache``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tac.provenance.contract import Provenance


@dataclass(frozen=True)
class Comma2k19DashcamPriors:
    """Comma2k19 dashcam priors — canonical wrapper output.

    Surfaces OOD dashcam statistics derived from cached Comma2k19 chunks.
    Per Catalog #209: this is ALWAYS OOD-tagged so consumers don't leak
    it into contest-video gradient paths.

    Fields:
        cached_chunk_ids: tuple of chunk_ids the cache currently holds.
        total_cached_bytes: sum of cached chunk sizes.
        dataset_license_spdx: ``"MIT"`` (Comma2k19's canonical license).
        dataset_provenance: ``"commaai/comma2k19"`` (canonical repo).
        is_ood_relative_to_contest_video: ALWAYS True per Catalog #209.
        canonical_cache_helper_invocation: dotted path to the helper
            (sister of Catalog #213 enforcement at the call site).
        provenance: canonical Provenance per Catalog #323.
    """

    cached_chunk_ids: tuple[str, ...]
    total_cached_bytes: int
    dataset_license_spdx: str
    dataset_provenance: str
    is_ood_relative_to_contest_video: bool
    canonical_cache_helper_invocation: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.cached_chunk_ids, tuple):
            raise TypeError("cached_chunk_ids must be tuple")
        for i, cid in enumerate(self.cached_chunk_ids):
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(
                    f"cached_chunk_ids[{i}]={cid!r} must be non-empty string"
                )
        if not isinstance(self.total_cached_bytes, int) or self.total_cached_bytes < 0:
            raise ValueError(
                f"total_cached_bytes={self.total_cached_bytes!r} must be non-negative int"
            )
        if not isinstance(self.dataset_license_spdx, str) or not self.dataset_license_spdx:
            raise ValueError("dataset_license_spdx must be non-empty")
        # Per Catalog #213 canonical contract: Comma2k19 is MIT.
        if self.dataset_license_spdx != "MIT":
            raise ValueError(
                f"dataset_license_spdx={self.dataset_license_spdx!r} must be 'MIT' per "
                "Comma2k19 canonical license"
            )
        if not isinstance(self.dataset_provenance, str) or not self.dataset_provenance:
            raise ValueError("dataset_provenance must be non-empty")
        # Per Catalog #213: canonical repo is commaai/comma2k19.
        if "comma2k19" not in self.dataset_provenance.lower():
            raise ValueError(
                f"dataset_provenance={self.dataset_provenance!r} must reference "
                "the canonical 'comma2k19' dataset"
            )
        if not isinstance(self.is_ood_relative_to_contest_video, bool):
            raise TypeError(
                "is_ood_relative_to_contest_video must be bool"
            )
        # Per Catalog #209: this MUST always be True. Comma2k19 is OOD
        # relative to the contest video by canonical construction.
        if self.is_ood_relative_to_contest_video is not True:
            raise ValueError(
                "is_ood_relative_to_contest_video MUST be True per Catalog #209 "
                "contest-video-leakage non-negotiable"
            )
        if (
            not isinstance(self.canonical_cache_helper_invocation, str)
            or not self.canonical_cache_helper_invocation
        ):
            raise ValueError(
                "canonical_cache_helper_invocation must be non-empty string"
            )
        # Per Catalog #213: must cite the canonical helper.
        EXPECTED_HELPERS = (
            "tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache",
            "Comma2k19LocalCache.fetch_chunk",
            "Comma2k19LocalCache.fetch_chunks",
        )
        if not any(h in self.canonical_cache_helper_invocation for h in EXPECTED_HELPERS):
            raise ValueError(
                f"canonical_cache_helper_invocation={self.canonical_cache_helper_invocation!r} "
                f"must reference one of {EXPECTED_HELPERS} per Catalog #213"
            )
        if not isinstance(self.provenance, Provenance):
            raise TypeError(
                f"provenance must be Provenance, got {type(self.provenance).__name__}"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "schema": "comma2k19_dashcam_priors_v1",
            "cached_chunk_ids": list(self.cached_chunk_ids),
            "total_cached_bytes": self.total_cached_bytes,
            "dataset_license_spdx": self.dataset_license_spdx,
            "dataset_provenance": self.dataset_provenance,
            "is_ood_relative_to_contest_video": self.is_ood_relative_to_contest_video,
            "canonical_cache_helper_invocation": self.canonical_cache_helper_invocation,
            "provenance": provenance_to_dict(self.provenance),
        }


def build_comma2k19_dashcam_priors_from_cache(
    cache: Any,
    *,
    provenance: Provenance,
) -> Comma2k19DashcamPriors:
    """Build :class:`Comma2k19DashcamPriors` from a canonical
    :class:`Comma2k19LocalCache` instance.

    Per Catalog #213: every Comma2k19 fetch routes through the canonical
    helper. This wrapper introspects the cache state (cached chunk IDs,
    total bytes) and returns a typed atlas without hitting the network.

    Args:
        cache: a ``tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache``
            instance (the canonical helper).
        provenance: canonical Provenance per Catalog #323.

    Returns:
        Frozen :class:`Comma2k19DashcamPriors` with OOD-tagged metadata.

    Raises:
        TypeError if ``cache`` is not a Comma2k19LocalCache (or sister
        with the required ``list_cached_chunks`` / ``CANONICAL_SOURCE_URL``
        / ``DATASET_LICENSE`` attrs).
    """
    # Duck-typing on the canonical surface; the real type lives in
    # tac.substrates.pretrained_driving_prior which we don't want to
    # eagerly import here (avoid circular-import risk + keep module
    # importable in minimal envs).
    required_attrs = ("list_cached_chunks", "DATASET_LICENSE", "CANONICAL_SOURCE_URL")
    for attr in required_attrs:
        if not hasattr(cache, attr):
            raise TypeError(
                f"cache argument missing required attribute {attr!r}; must be a "
                "Comma2k19LocalCache (or sister implementing the canonical surface)"
            )
    cached_ids_raw = cache.list_cached_chunks()
    if not isinstance(cached_ids_raw, list):
        raise TypeError(
            f"cache.list_cached_chunks() must return list; got {type(cached_ids_raw).__name__}"
        )
    cached_ids = tuple(sorted(str(c) for c in cached_ids_raw))
    # Compute total bytes from manifest entries for cached chunks. Skip
    # any chunk_id not in the manifest (defensive). Manifest field name
    # is `chunk_manifest` per the canonical helper.
    total_bytes = 0
    manifest = getattr(cache, "chunk_manifest", {}) or {}
    for chunk_id in cached_ids:
        entry = manifest.get(chunk_id)
        if entry is not None and hasattr(entry, "size_bytes"):
            total_bytes += int(entry.size_bytes)
    return Comma2k19DashcamPriors(
        cached_chunk_ids=cached_ids,
        total_cached_bytes=total_bytes,
        dataset_license_spdx=str(cache.DATASET_LICENSE),
        dataset_provenance=str(cache.CANONICAL_SOURCE_URL),
        is_ood_relative_to_contest_video=True,
        canonical_cache_helper_invocation=(
            "tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache.list_cached_chunks"
        ),
        provenance=provenance,
    )


__all__ = [
    "Comma2k19DashcamPriors",
    "build_comma2k19_dashcam_priors_from_cache",
]
