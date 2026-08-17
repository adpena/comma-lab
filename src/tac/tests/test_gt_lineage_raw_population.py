# SPDX-License-Identifier: MIT
"""Tests for the ddm_gl2 raw-GT population cure: extension-agnostic discovery + the refusing gate.

THE BUG THESE PIN
-----------------
``ddm_gl1`` built the content-addressed lineage registry but discovered artifacts through an
EXTENSION ALLOW-LIST (``npy|npz|pt|pth``) in both of its legs.  A second ground-truth population --
headerless raw memmaps (``gt_segnet_argmax.u8``, ``gt_segnet_margin.f16``,
``gt_segnet_logits.f16``) with 15 live readers -- was therefore structurally invisible, and
``ddm_gl2`` measured that its ``.u8`` is PyAV lineage while the ``.npy`` most instruments compare
against is DALI lineage: **20,673 of 117,964,800 sites = 0.017525 S units apart.**

The cure is a DENY-list: unknown suffixes now fail CLOSED into the must-be-registered population.
These tests exist to prove the gate can actually FAIL -- a guard that has never been observed
refusing is indistinguishable from a guard that cannot refuse.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from tac.gt_lineage import (
    GtArtifactLineage,
    GtLineageUnregisteredPopulation,
    assert_gt_population_registered,
    enumerate_gt_artifact_literals,
    resolve_gt_literal,
    unregistered_gt_artifacts,
)
from tac.measurement_integrity import (
    GT_ARGMAX_TOKENS,
    NON_FIELD_SUFFIXES,
    find_gt_artifact_literals,
)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fake_repo(tmp_path, literal: str, payload: bytes = b"\x00\x01\x02\x03"):
    """A minimal repo whose one instrument names ``literal``, plus a search root holding it."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tools" / "reader.py").write_text(
        f'GT = TARGETS / "{literal}"\nd = load(GT)\n', encoding="utf-8"
    )
    root = tmp_path / "targets"
    root.mkdir()
    artifact = root / literal
    artifact.write_bytes(payload)
    return repo, root, artifact


# --- the literal finder: deny-list polarity ------------------------------------------------


def test_find_literals_accepts_a_container_nobody_has_defined():
    """The whole cure: an UNSEEN suffix must be DISCOVERED, not silently dropped.

    Under gl1's allow-list this returns nothing, which reads identically to "there is nothing
    here" -- the failure mode that hid a 117,964,800-byte GT field with 15 readers.
    """
    found = find_gt_artifact_literals('p = "gt_segnet_argmax.qq7"')
    assert found == {"gt_segnet_argmax.qq7"}


def test_find_literals_accepts_the_raw_containers_gl1_missed():
    text = 'a = "gt_segnet_argmax.u8"\nb = "gt_segnet_margin.f16"\nc = "gt_poses.f32"'
    assert find_gt_artifact_literals(text) == {
        "gt_segnet_argmax.u8",
        "gt_segnet_margin.f16",
        "gt_poses.f32",
    }


def test_find_literals_still_accepts_the_gl1_containers():
    text = 'a = "gt_argmax_n600.npy"\nb = "gt_n96.npz"\nc = "gt_cache_600.pt"'
    assert find_gt_artifact_literals(text) == {
        "gt_argmax_n600.npy",
        "gt_n96.npz",
        "gt_cache_600.pt",
    }


@pytest.mark.parametrize("suffix", sorted(NON_FIELD_SUFFIXES))
def test_find_literals_drops_non_field_suffixes(suffix):
    """A receipt, a script, or a config is not a tensor field and must not enter the population."""
    assert find_gt_artifact_literals(f'p = "gt_thing{suffix}"') == set()


def test_find_literals_keeps_directory_qualified_literals():
    text = 'p = "targets_n600/gt_segnet_argmax.u8"'
    assert find_gt_artifact_literals(text) == {"targets_n600/gt_segnet_argmax.u8"}


# --- resolution: one basename, several distinct byte-blobs -----------------------------------


def test_resolve_returns_every_root_not_the_first(tmp_path):
    """``targets_n16`` and ``targets_n600`` both hold a ``gt_segnet_argmax.u8`` and they are
    DIFFERENT bytes.  Collapsing them to a first hit is the name-as-identity error again."""
    r1 = tmp_path / "n16"
    r2 = tmp_path / "n600"
    r1.mkdir()
    r2.mkdir()
    (r1 / "gt_segnet_argmax.u8").write_bytes(b"a")
    (r2 / "gt_segnet_argmax.u8").write_bytes(b"bb")
    hits = resolve_gt_literal(
        "gt_segnet_argmax.u8", repo_root=tmp_path, search_roots=(r1, r2)
    )
    assert len(hits) == 2
    assert {p.read_bytes() for p in hits} == {b"a", b"bb"}


def test_resolve_returns_empty_for_a_missing_artifact(tmp_path):
    assert resolve_gt_literal("gt_absent.u8", repo_root=tmp_path, search_roots=()) == []


# --- the gate: it must be observed FAILING ----------------------------------------------------


def test_positive_control_gate_refuses_an_unregistered_artifact(tmp_path):
    """POSITIVE CONTROL: the gate raises on a reachable GT artifact absent from the registry."""
    repo, root, artifact = _fake_repo(tmp_path, "gt_segnet_argmax.u8")
    with pytest.raises(GtLineageUnregisteredPopulation) as exc:
        assert_gt_population_registered(
            repo_root=repo, search_roots=(root,), registry={}
        )
    assert len(exc.value.artifacts) == 1
    got = exc.value.artifacts[0]
    assert got["sha256"] == _sha256_bytes(artifact.read_bytes())
    assert got["bytes"] == artifact.stat().st_size
    assert "tools/reader.py" in got["readers"]
    # Structured detail, not scraped text: a message reword must not break a caller.
    assert str(artifact) == got["path"]


def test_positive_control_fires_on_an_unknown_container(tmp_path):
    """The control that proves the cure is not cosmetic: a suffix nobody has defined still fires."""
    repo, root, _ = _fake_repo(tmp_path, "gt_future_field.zz9")
    with pytest.raises(GtLineageUnregisteredPopulation):
        assert_gt_population_registered(repo_root=repo, search_roots=(root,), registry={})


def test_negative_control_gate_passes_once_registered(tmp_path):
    """NEGATIVE CONTROL: registering the DIGEST clears it, and the call returns an empty list."""
    repo, root, artifact = _fake_repo(tmp_path, "gt_segnet_argmax.u8")
    digest = _sha256_bytes(artifact.read_bytes())
    registry = {
        digest: GtArtifactLineage(
            sha256=digest,
            bytes=artifact.stat().st_size,
            basename=artifact.name,
            lineage="PYAV_YUV420_TO_RGB",
            evidence="EMPIRICAL_EXACT_MATCH",
            measurement="test fixture",
            claim_boundary="test fixture",
        )
    }
    assert (
        assert_gt_population_registered(
            repo_root=repo, search_roots=(root,), registry=registry
        )
        == []
    )


def test_registering_a_DIFFERENT_file_does_not_clear_the_gate(tmp_path):
    """A verified file's reputation must NOT launder an unverified one -- the whole reason the
    registry is keyed by content.  Same basename, same byte count, different bytes."""
    repo, root, artifact = _fake_repo(tmp_path, "gt_segnet_argmax.u8", payload=b"AAAA")
    other = _sha256_bytes(b"BBBB")  # same length, different content
    registry = {
        other: GtArtifactLineage(
            sha256=other,
            bytes=4,
            basename="gt_segnet_argmax.u8",
            lineage="DALI_NVDEC",
            evidence="EMPIRICAL_EXACT_MATCH",
            measurement="test fixture",
            claim_boundary="test fixture",
        )
    }
    with pytest.raises(GtLineageUnregisteredPopulation) as exc:
        assert_gt_population_registered(
            repo_root=repo, search_roots=(root,), registry=registry
        )
    assert exc.value.artifacts[0]["sha256"] == _sha256_bytes(artifact.read_bytes())


def test_allow_list_is_keyed_by_digest_not_by_name(tmp_path):
    """The escape hatch must be content-addressed too, or it reintroduces the defect it excuses."""
    repo, root, artifact = _fake_repo(tmp_path, "gt_segnet_argmax.u8")
    digest = _sha256_bytes(artifact.read_bytes())
    assert (
        assert_gt_population_registered(
            repo_root=repo, search_roots=(root,), registry={}, allow_sha256=(digest,)
        )
        == []
    )
    with pytest.raises(GtLineageUnregisteredPopulation):
        assert_gt_population_registered(
            repo_root=repo,
            search_roots=(root,),
            registry={},
            allow_sha256=("gt_segnet_argmax.u8",),  # a NAME cannot excuse anything
        )


def test_gate_ignores_test_fixtures(tmp_path):
    """A fixture that invents a GT name in tmp_path names a file no instrument loads."""
    repo = tmp_path / "repo"
    (repo / "src" / "tests").mkdir(parents=True)
    (repo / "src" / "tests" / "test_thing.py").write_text(
        'GT = "gt_fixture.u8"\n', encoding="utf-8"
    )
    root = tmp_path / "targets"
    root.mkdir()
    (root / "gt_fixture.u8").write_bytes(b"\x00")
    assert (
        assert_gt_population_registered(repo_root=repo, search_roots=(root,), registry={})
        == []
    )


def test_unregistered_list_carries_literal_and_readers(tmp_path):
    repo, root, _ = _fake_repo(tmp_path, "gt_segnet_argmax.u8")
    rows = unregistered_gt_artifacts(repo_root=repo, search_roots=(root,), registry={})
    assert len(rows) == 1
    assert rows[0]["literal"] == "gt_segnet_argmax.u8"
    assert rows[0]["readers"] == ["tools/reader.py"]


def test_enumerate_maps_literals_to_the_files_naming_them(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tools" / "a.py").write_text('p = "gt_x.u8"\n', encoding="utf-8")
    (repo / "tools" / "b.py").write_text('p = "gt_x.u8"\nq = "gt_y.npy"\n', encoding="utf-8")
    lits = enumerate_gt_artifact_literals(repo_root=repo)
    assert lits["gt_x.u8"] == ["tools/a.py", "tools/b.py"]
    assert lits["gt_y.npy"] == ["tools/b.py"]


# --- the live repo: the coverage the cure actually delivered ----------------------------------


def test_the_raw_u8_population_is_now_registered():
    """The five raw artifacts ddm_gl1's allow-list could not see are recorded, by sha256."""
    from tac.gt_lineage import REGISTRY_PATH

    reg = json.loads(REGISTRY_PATH.read_text())
    by_sha = {a["sha256"]: a for a in reg["artifacts"]}
    # The n600 argmax field: PyAV lineage, producer receipt bound to these exact bytes.
    n600 = by_sha.get("36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68")
    assert n600 is not None, "the n600 gt_segnet_argmax.u8 must be registered by sha256"
    assert n600["bytes"] == 117_964_800
    assert n600["lineage"] == "PYAV_YUV420_TO_RGB"
    assert "PRODUCER_DECLARED" in n600["lineage_evidence"]
    # The n16 field is a DIFFERENT digest and gets its OWN row despite the identical basename.
    n16 = by_sha.get("7646794a39c8e3283c5458916185f60d1a2628e244f90e8681eaea5722449730")
    assert n16 is not None
    assert n16["bytes"] == 3_145_728
    assert n16["lineage"] == "PYAV_YUV420_TO_RGB"
    assert n600["sha256"] != n16["sha256"]
    assert n600["basename"] == n16["basename"] == "gt_segnet_argmax.u8"


def test_unresolved_rows_are_recorded_but_not_usable():
    """Recording an UNKNOWN buys VISIBILITY, never USABILITY: assert_gt_lineage still refuses."""
    from tac.gt_lineage import REGISTRY_PATH, load_registry

    reg = json.loads(REGISTRY_PATH.read_text())
    by_sha = {a["sha256"]: a for a in reg["artifacts"]}
    margin = by_sha.get(
        "177d22f0ef16e31f9de0229606f72e69d22dd550b7ff55342f82d01ebe6f228d"
    )
    assert margin is not None, "the n600 gt_segnet_margin.f16 must be recorded, not hidden"
    assert margin["lineage"] == "UNKNOWN_UNCOMPARABLE"
    entry = load_registry()[margin["sha256"]]
    assert not entry.is_resolved
    assert "Lineage NOT established" in entry.claim_boundary


def test_gt_argmax_tokens_are_documented_as_detection_only():
    """The name-keyed hole: these tokens must never read as a lineage authorization."""
    import tac.measurement_integrity as mi

    assert "gt_segnet_argmax" in GT_ARGMAX_TOKENS
    src = mi.__doc__ or ""
    module_src = (
        __import__("pathlib").Path(mi.__file__).read_text(encoding="utf-8")
    )
    assert "A NAME IS NOT AN IDENTITY" in module_src
    assert "tac.gt_lineage" in module_src
    assert isinstance(src, str)
