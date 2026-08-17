# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.gt_lineage` — the GT decode-lineage registry and fail-closed guard.

These tests are HERMETIC: they build synthetic registries over temp files rather than depending
on retained run directories, so they keep working when external volumes are unmounted.  A small
number of integration tests are guarded on the real registry being present.

The tests are written to FAIL if the guard is replaced by a no-op.  Per the craft manual: a test
that still passes when the code is broken is verifying constants, not behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.gt_lineage import (
    AUTHORITY_LINEAGE,
    DALI_NVDEC,
    PYAV_YUV420_TO_RGB,
    REGISTRY_PATH,
    UNKNOWN_AMBIGUOUS,
    GtArtifactLineage,
    GtLineageMismatch,
    GtLineageSplit,
    GtLineageUnknown,
    GtSource,
    assert_gt_lineage,
    assert_single_lineage,
    basename_lineage_collisions,
    is_known_lineage,
    lineage_of_file,
    lineage_of_source,
    load_registry,
    population_split_report,
    runtime_decode_lineage,
    sha256_file,
)


@pytest.fixture
def dali_file(tmp_path: Path) -> Path:
    p = tmp_path / "gt_argmax_n600.npy"
    p.write_bytes(b"pretend-this-is-a-dali-lineage-argmax-field")
    return p


@pytest.fixture
def av_file(tmp_path: Path) -> Path:
    # Deliberately the SAME basename as the DALI fixture, in a different directory.  This is the
    # real situation ddm_gl1 measured: seven files named gt_argmax_n600.npy, three distinct
    # sha256, spanning both lineages.
    d = tmp_path / "other_run"
    d.mkdir()
    p = d / "gt_argmax_n600.npy"
    p.write_bytes(b"pretend-this-is-a-pyav-lineage-argmax-field")
    return p


@pytest.fixture
def registry(dali_file: Path, av_file: Path) -> dict[str, GtArtifactLineage]:
    return {
        sha256_file(dali_file): GtArtifactLineage(
            sha256=sha256_file(dali_file),
            bytes=dali_file.stat().st_size,
            basename=dali_file.name,
            lineage=DALI_NVDEC,
            evidence="EMPIRICAL_EXACT_MATCH",
            measurement="seg: 3 differing sites vs DALI ruler, 20,672 vs AV ruler",
            claim_boundary="synthetic fixture",
            known_paths=(str(dali_file),),
            known_basenames=(dali_file.name,),
        ),
        sha256_file(av_file): GtArtifactLineage(
            sha256=sha256_file(av_file),
            bytes=av_file.stat().st_size,
            basename=av_file.name,
            lineage=PYAV_YUV420_TO_RGB,
            evidence="EMPIRICAL_EXACT_MATCH",
            measurement="seg: 20,671 differing sites vs DALI ruler, 2 vs AV ruler",
            claim_boundary="synthetic fixture",
            known_paths=(str(av_file),),
            known_basenames=(av_file.name,),
        ),
    }


# --- identity is content, not name ---------------------------------------------------------


def test_same_basename_resolves_to_different_lineages(dali_file, av_file, registry):
    """The defect that makes a name-keyed rule unsafe, asserted directly."""
    assert dali_file.name == av_file.name
    assert lineage_of_file(dali_file, registry=registry).lineage == DALI_NVDEC
    assert lineage_of_file(av_file, registry=registry).lineage == PYAV_YUV420_TO_RGB


def test_basename_collisions_are_reported(registry):
    collisions = basename_lineage_collisions(registry=registry)
    assert "gt_argmax_n600.npy" in collisions
    assert set(collisions["gt_argmax_n600.npy"]) == {DALI_NVDEC, PYAV_YUV420_TO_RGB}


def test_identical_content_at_two_paths_is_one_entry(tmp_path, registry, dali_file):
    """A copy of the same bytes must resolve identically, wherever it lives."""
    copy = tmp_path / "elsewhere.npy"
    copy.write_bytes(dali_file.read_bytes())
    assert lineage_of_file(copy, registry=registry).lineage == DALI_NVDEC


# --- the per-read guard --------------------------------------------------------------------


def test_assert_passes_matching_lineage(dali_file, registry):
    entry = assert_gt_lineage(dali_file, required=DALI_NVDEC, registry=registry)
    assert entry.lineage == DALI_NVDEC
    assert entry.sha256 == sha256_file(dali_file)


def test_assert_refuses_mismatched_lineage(dali_file, registry):
    with pytest.raises(GtLineageMismatch) as exc:
        assert_gt_lineage(dali_file, required=PYAV_YUV420_TO_RGB, registry=registry)
    assert "DALI_NVDEC" in str(exc.value)


def test_assert_refuses_av_when_authority_required(av_file, registry):
    """The direction that silently costs score: authority-tracking read of an AV cache."""
    with pytest.raises(GtLineageMismatch):
        assert_gt_lineage(av_file, required=AUTHORITY_LINEAGE, registry=registry)


def test_assert_refuses_unregistered_file(tmp_path, registry):
    stranger = tmp_path / "unregistered.npy"
    stranger.write_bytes(b"nobody has ever classified these bytes")
    with pytest.raises(GtLineageUnknown):
        assert_gt_lineage(stranger, required=DALI_NVDEC, registry=registry)


def test_assert_refuses_missing_file(tmp_path, registry):
    with pytest.raises(GtLineageUnknown):
        assert_gt_lineage(tmp_path / "does_not_exist.npy", required=DALI_NVDEC, registry=registry)


def test_assert_refuses_unresolved_lineage(tmp_path):
    p = tmp_path / "ambiguous.npy"
    p.write_bytes(b"measured but indecisive")
    reg = {
        sha256_file(p): GtArtifactLineage(
            sha256=sha256_file(p),
            bytes=p.stat().st_size,
            basename=p.name,
            lineage=UNKNOWN_AMBIGUOUS,
            evidence="EMPIRICAL_INDECISIVE",
            measurement="no leg reached a decisive margin",
            claim_boundary="lineage NOT established",
        )
    }
    with pytest.raises(GtLineageUnknown):
        assert_gt_lineage(p, required=DALI_NVDEC, registry=reg)


def test_required_must_be_a_real_lineage(dali_file, registry):
    with pytest.raises(ValueError):
        assert_gt_lineage(dali_file, required="SOMETHING_ELSE", registry=registry)


def test_is_known_lineage_does_not_raise(tmp_path, dali_file, registry, monkeypatch):
    monkeypatch.setattr("tac.gt_lineage.load_registry", lambda **_: registry)
    assert is_known_lineage(dali_file) is True
    stranger = tmp_path / "nope.npy"
    stranger.write_bytes(b"x")
    assert is_known_lineage(stranger) is False


# --- runtime decode sources ----------------------------------------------------------------


def test_runtime_decode_lineages():
    assert runtime_decode_lineage("frame_utils.yuv420_to_rgb") == PYAV_YUV420_TO_RGB
    assert runtime_decode_lineage("DaliVideoDataset") == DALI_NVDEC


def test_unknown_runtime_decoder_refuses():
    with pytest.raises(GtLineageUnknown):
        runtime_decode_lineage("pyav_rgb24")  # the forbidden third path


def test_lineage_of_source_handles_both_kinds(dali_file, registry):
    assert lineage_of_source(GtSource.file(dali_file), registry=registry) == DALI_NVDEC
    assert (
        lineage_of_source(GtSource.runtime_decode("frame_utils.yuv420_to_rgb"))
        == PYAV_YUV420_TO_RGB
    )


def test_bad_source_kind_refuses(registry):
    with pytest.raises(GtLineageUnknown):
        lineage_of_source(GtSource(kind="telepathy", ref="x"), registry=registry)


# --- the span detector: the ddm_pi2 defect as a predicate ----------------------------------


def test_span_detector_fires_on_the_pi2_configuration(dali_file, registry):
    """DALI seg cache + fresh PyAV decode — exactly what shipped, and it must refuse."""
    with pytest.raises(GtLineageSplit) as exc:
        assert_single_lineage(
            [GtSource.file(dali_file), GtSource.runtime_decode("frame_utils.yuv420_to_rgb")],
            instrument="reconstructed_pi2_instrument",
            registry=registry,
        )
    msg = str(exc.value)
    assert "DALI_NVDEC" in msg and "PYAV_YUV420_TO_RGB" in msg


def test_span_detector_fires_on_two_files_of_different_lineage(dali_file, av_file, registry):
    with pytest.raises(GtLineageSplit):
        assert_single_lineage(
            [GtSource.file(dali_file), GtSource.file(av_file)],
            instrument="two_caches",
            registry=registry,
        )


def test_span_detector_passes_a_coherent_instrument(dali_file, registry):
    lin = assert_single_lineage(
        [GtSource.file(dali_file), GtSource.runtime_decode("DaliVideoDataset")],
        instrument="coherent",
        registry=registry,
    )
    assert lin == DALI_NVDEC


def test_span_detector_refuses_an_empty_declaration(registry):
    """Declaring no sources must not certify anything."""
    with pytest.raises(GtLineageUnknown):
        assert_single_lineage([], instrument="declares_nothing", registry=registry)


def test_span_detector_propagates_unknown_source(tmp_path, dali_file, registry):
    stranger = tmp_path / "unknown.npy"
    stranger.write_bytes(b"unclassified")
    with pytest.raises(GtLineageUnknown):
        assert_single_lineage(
            [GtSource.file(dali_file), GtSource.file(stranger)],
            instrument="partly_unknown",
            registry=registry,
        )


def test_split_exception_carries_structured_detail(dali_file, registry):
    """Consumers must never have to scrape the message text to learn what split."""
    with pytest.raises(GtLineageSplit) as exc:
        assert_single_lineage(
            [GtSource.file(dali_file), GtSource.runtime_decode("frame_utils.yuv420_to_rgb")],
            instrument="structured",
            registry=registry,
        )
    assert exc.value.lineages == (DALI_NVDEC, PYAV_YUV420_TO_RGB)
    assert set(exc.value.resolved.values()) == {DALI_NVDEC, PYAV_YUV420_TO_RGB}


def test_digest_cache_is_invalidated_when_content_changes(tmp_path):
    """The stat-keyed digest memo must not serve a stale lineage after an in-place edit.

    Both payloads are the SAME LENGTH on purpose.  With differing sizes this test would pass on
    the size component alone and never exercise mtime invalidation -- it would be a test that
    still passes when the code is broken.
    """
    a_bytes = b"A" * 64
    b_bytes = b"B" * 64
    p = tmp_path / "mutating.npy"
    p.write_bytes(a_bytes)
    sha_a, sha_b = sha256_file(p), None
    p.write_bytes(b_bytes)
    sha_b = sha256_file(p)
    p.write_bytes(a_bytes)

    reg = {
        sha_a: GtArtifactLineage(
            sha256=sha_a, bytes=64, basename=p.name, lineage=DALI_NVDEC,
            evidence="EMPIRICAL_EXACT_MATCH", measurement="fixture", claim_boundary="fixture",
        ),
        sha_b: GtArtifactLineage(
            sha256=sha_b, bytes=64, basename=p.name, lineage=PYAV_YUV420_TO_RGB,
            evidence="EMPIRICAL_EXACT_MATCH", measurement="fixture", claim_boundary="fixture",
        ),
    }
    assert lineage_of_file(p, registry=reg).lineage == DALI_NVDEC
    p.write_bytes(b_bytes)  # same length, different content
    assert lineage_of_file(p, registry=reg).lineage == PYAV_YUV420_TO_RGB


# --- the gauge must not zero out on its own cure -------------------------------------------


def test_population_gauge_is_unchanged_by_adding_declarations(dali_file, av_file, registry):
    """The sister-law check: applying the cure must not move the detector.

    ``population_split_report`` measures the artifact population and what instruments read.
    Asserting lineage a thousand times adds instrumentation, not single-lineage-ness, so the
    gauge must read identically before and after.
    """
    before = population_split_report(registry=registry)
    for _ in range(25):
        assert_gt_lineage(dali_file, required=DALI_NVDEC, registry=registry)
        assert_gt_lineage(av_file, required=PYAV_YUV420_TO_RGB, registry=registry)
    after = population_split_report(registry=registry)
    assert before == after
    assert after["distinct_resolved_lineage_count"] == 2
    assert after["population_is_single_lineage"] is False


def test_population_gauge_reports_single_lineage_only_when_true(dali_file, registry):
    single = {k: v for k, v in registry.items() if v.lineage == DALI_NVDEC}
    report = population_split_report(registry=single)
    assert report["population_is_single_lineage"] is True
    assert report["distinct_resolved_lineages_present"] == [DALI_NVDEC]


def test_population_gauge_lists_split_instruments(dali_file, registry):
    report = population_split_report(
        {
            "split_one": [
                GtSource.file(dali_file),
                GtSource.runtime_decode("frame_utils.yuv420_to_rgb"),
            ],
            "coherent_one": [GtSource.file(dali_file)],
        },
        registry=registry,
    )
    assert "split_one" in report["split_instruments"]
    assert "coherent_one" not in report["split_instruments"]


# --- the shipped registry ------------------------------------------------------------------


@pytest.mark.skipif(not REGISTRY_PATH.exists(), reason="shipped registry not present")
def test_shipped_registry_is_wellformed():
    raw = json.loads(REGISTRY_PATH.read_text())
    assert raw["schema"] == "ddm_gl1_gt_lineage_registry_v1"
    assert raw["keying"].startswith("sha256")
    assert raw["score_claim"] is False
    reg = load_registry(refresh=True)
    assert reg, "shipped registry must not be empty"
    for sha, entry in reg.items():
        assert len(sha) == 64, f"registry key {sha!r} is not a sha256"
        assert entry.sha256 == sha
        assert entry.lineage, "every entry must carry a lineage, even if UNKNOWN"
        assert entry.claim_boundary, "every entry must state how far its label may be pushed"


@pytest.mark.skipif(not REGISTRY_PATH.exists(), reason="shipped registry not present")
def test_shipped_registry_records_both_lineages():
    """If this ever reports one lineage, the population really did converge — check before editing."""
    report = population_split_report()
    assert report["distinct_resolved_lineage_count"] >= 1
    assert set(report["distinct_resolved_lineages_present"]) <= {DALI_NVDEC, PYAV_YUV420_TO_RGB}
