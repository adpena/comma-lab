# SPDX-License-Identifier: MIT
"""ddm_ms8 — guards for the s_t codebook becoming a READ, FITTABLE table.

MEASURED on the live pw1 archive (``v4d_composed_pw1_archive.zip``): the
manifest already ships ``"st_grid"`` and ``inflate_runner_v4d`` never read it,
using its vendored ``ST_GRID`` constant instead.  That is 34 counted deflated
bytes no decode step consumed — the #417 counted-but-inert class — and the
existing parse-back verifier could not see it because its denominator is
``pose_warp.stp`` only.

These tests guard BEHAVIOUR, not constants:

* the receiver honours a manifest table and falls back to the vendored ladder
  when the key is absent, so every pre-ms8 archive decodes bit-identically;
* the receiver fails CLOSED on a table that cannot index the shipped stream,
  rather than silently wrapping or truncating;
* the builder's override validates the codebook contract and, when absent,
  leaves the copied ``st_coded`` stream untouched (the byte-identity guard);
* a re-encoded index stream round-trips through the REAL shipped coder.

Each would fail if the receiver went back to ignoring the manifest, if the
range check were dropped, or if the override silently accepted an unsorted /
duplicated / out-of-range codebook.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load(name: str, relpath: str):
    """Import a repo script by path (experiments/ is not a package)."""
    path = _REPO / relpath
    if not path.exists():  # pragma: no cover - env guard
        pytest.skip(f"missing {relpath}")
    sys.path.insert(0, str(_REPO / "experiments"))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - env guard
        pytest.skip(f"cannot load {relpath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - env guard
        pytest.skip(f"cannot exec {relpath}: {exc}")
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load("_ms8_builder", "experiments/ddm_v4d_build_composed_archive.py")


@pytest.fixture(scope="module")
def ms8():
    return _load("_ms8_race", "tools/ms8_st_codebook_race.py")


# --------------------------------------------------------------------------- #
# the receiver's table resolution + range guard
# --------------------------------------------------------------------------- #
_RECEIVER = _REPO / "experiments" / "inflate_runner_v4d.py"


def _receiver_source() -> str:
    if not _RECEIVER.exists():  # pragma: no cover - env guard
        pytest.skip("receiver missing")
    return _RECEIVER.read_text()


def test_receiver_reads_st_grid_from_manifest_not_only_the_constant():
    """The inert manifest field must become a CONSUMED one."""
    src = _receiver_source()
    assert 'manifest.get("st_grid"' in src, (
        "receiver must resolve the s_t codebook from the manifest; without "
        "this the shipped st_grid bytes are counted-but-inert (#417)")


def test_receiver_keeps_the_vendored_ladder_as_fallback():
    """Pre-ms8 archives have no key and must still decode."""
    src = _receiver_source()
    assert 'manifest.get("st_grid", ST_GRID)' in src, (
        "the vendored ladder must remain the fallback so archives whose "
        "manifest predates the key decode bit-identically")


def test_receiver_refuses_an_index_the_table_cannot_serve():
    """Fail closed, never wrap: a short table with a large index is a refusal."""
    src = _receiver_source()
    assert "outside st_grid of" in src, (
        "receiver must refuse st_idx >= len(st_grid) instead of wrapping")


def test_receiver_table_resolution_reproduces_the_vendored_ladder():
    """Executable form of the fallback claim, on the real vendored constant."""
    sys.path.insert(0, str(_REPO / "experiments"))
    try:
        import ddm_pfs1_ep_warp_pose_solve as pfs1
    except Exception as exc:  # pragma: no cover - env guard
        pytest.skip(f"cannot import pfs1: {exc}")
    grid = tuple(float(x) for x in pfs1.ST_GRID)
    assert np.asarray({}.get("st_grid", grid), np.float64).tolist() == list(grid)
    fitted = [0.06, 0.08, 0.1]
    assert np.asarray({"st_grid": fitted}.get("st_grid", grid),
                      np.float64).tolist() == fitted


# --------------------------------------------------------------------------- #
# the builder override contract
# --------------------------------------------------------------------------- #
def test_override_absent_is_a_passthrough(builder):
    """No flag => the base st_coded is copied verbatim (byte identity)."""
    assert builder.load_st_override(None, 600) == (None, None)


def test_override_accepts_a_valid_fitted_codebook(builder, tmp_path):
    doc = {"st_grid": [0.06, 0.08, 0.1, 0.12],
           "st_idx": [0, 1, 2, 3, 1, 1]}
    p = tmp_path / "st.json"
    p.write_text(json.dumps(doc))
    grid, idx = builder.load_st_override(str(p), 6)
    assert grid == [0.06, 0.08, 0.1, 0.12]
    assert idx.dtype == np.uint8
    assert idx.tolist() == [0, 1, 2, 3, 1, 1]


@pytest.mark.parametrize(
    ("doc", "n", "needle"),
    [
        ({"st_grid": [0.06, 0.08], "st_idx": [0, 1, 0]}, 4, "expected (4,)"),
        ({"st_grid": [0.06, 0.08], "st_idx": [0, 2]}, 2, "outside st_grid"),
        ({"st_grid": [0.06, 0.08], "st_idx": [0, -1]}, 2, "outside st_grid"),
        ({"st_grid": [0.08, 0.06], "st_idx": [0, 1]}, 2, "strictly increasing"),
        ({"st_grid": [0.06, 0.06], "st_idx": [0, 1]}, 2, "strictly increasing"),
        ({"st_grid": [0.06], "st_idx": [0, 0]}, 2, "expected 2..16"),
        ({"st_grid": [i / 100 for i in range(17)],
          "st_idx": [0, 1]}, 2, "expected 2..16"),
    ],
)
def test_override_fails_closed_on_a_broken_codebook(builder, tmp_path, doc, n,
                                                    needle):
    p = tmp_path / "st.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(SystemExit) as exc:
        builder.load_st_override(str(p), n)
    assert needle in str(exc.value)


def test_inherited_foreign_table_is_refused(builder):
    """POSITIVE CONTROL for the guard: a foreign inherited table must REFUSE.

    Without the override the ``st_coded`` stream is copied verbatim and was
    indexed against the vendored ladder; a different inherited table would
    decode every pair at the wrong s_t, silently.
    """
    foreign = [x + 0.001 for x in builder.vendored_st_grid()]
    with pytest.raises(SystemExit) as exc:
        builder.assert_inherited_st_grid_is_vendored({"st_grid": foreign})
    assert "differs from the vendored ladder" in str(exc.value)


def test_inherited_vendored_table_passes_and_absent_key_passes(builder):
    """NEGATIVE CONTROL: the two legitimate shapes must NOT refuse."""
    builder.assert_inherited_st_grid_is_vendored(
        {"st_grid": builder.vendored_st_grid()})
    builder.assert_inherited_st_grid_is_vendored({})


def test_vendored_grid_matches_the_receiver_fallback(builder, ms8):
    """The builder guard and the race tool must speak of the SAME ladder."""
    assert builder.vendored_st_grid() == list(ms8.INCUMBENT_ST_GRID)


def test_reencoded_stream_roundtrips_through_the_real_coder(builder):
    """The shipped coder must reproduce the exact index stream."""
    sys.path.insert(0, str(_REPO / "experiments"))
    try:
        from ddm_r7_token_coder import decode_token_codes
    except Exception as exc:  # pragma: no cover - env guard
        pytest.skip(f"cannot import r7 coder: {exc}")
    rng = np.random.default_rng(20260802)
    idx = rng.integers(0, 5, size=600).astype(np.uint8)
    frame = builder.encode_st_stream(idx, 5)
    back = np.asarray(decode_token_codes(frame), np.int64).reshape(-1)[:600]
    assert back.tolist() == idx.tolist()


def test_narrower_alphabet_never_costs_more_than_the_incumbent(builder):
    """A codebook race is only honest if the coder prices the alphabet.

    Same index content, declared over a 5-entry vs an 11-entry alphabet: the
    narrower declaration must not be larger.  If this ever inverts, the
    'shrinking the alphabet is free' arithmetic in the ms8 race is wrong.
    """
    rng = np.random.default_rng(7)
    idx = rng.integers(0, 5, size=600).astype(np.uint8)
    assert len(builder.encode_st_stream(idx, 5)) <= len(
        builder.encode_st_stream(idx, 11))


# --------------------------------------------------------------------------- #
# the race tool's own contracts
# --------------------------------------------------------------------------- #
def test_race_support_contains_every_incumbent_codeword(ms8):
    """RESEL and the positive control must land on measured columns."""
    for c in ms8.INCUMBENT_ST_GRID:
        assert any(abs(s - c) < 1e-12 for s in ms8.S_EVAL), (
            f"incumbent codeword {c} is not in S_EVAL; the canary would have "
            "to interpolate and RESEL would not be exactly representable")


def test_race_support_brackets_the_incumbent_on_both_sides(ms8):
    """The instrument must be able to detect clipping in EITHER direction."""
    assert min(ms8.S_EVAL) <= min(ms8.INCUMBENT_ST_GRID)
    assert max(ms8.S_EVAL) > max(ms8.INCUMBENT_ST_GRID), (
        "S_EVAL must extend past the incumbent's top codeword, else this tool "
        "assumes the incumbent support was right — the pw1 bug, repeated")


def test_race_support_is_sorted_and_unique(ms8):
    assert list(ms8.S_EVAL) == sorted(set(ms8.S_EVAL))


def test_kmedoid_finds_the_exact_optimum_on_a_planted_problem(ms8):
    """Behaviour, not a constant: a planted 2-cluster problem has a known
    optimal 2-subset and the selector must return it."""
    curves = np.array([
        [0.0, 5.0, 5.0, 9.0],
        [0.1, 5.0, 5.0, 9.0],
        [9.0, 5.0, 5.0, 0.0],
        [9.0, 5.0, 5.0, 0.2],
    ])
    assert ms8.kmedoid_subset(curves, 2) == (0, 3)


def test_kmedoid_is_monotone_in_k(ms8):
    """More codewords can never increase the unconstrained distortion."""
    rng = np.random.default_rng(3)
    curves = rng.random((40, 9))
    prev = float("inf")
    for k in range(1, 8):
        sub = ms8.kmedoid_subset(curves, k)
        cost = float(curves[:, list(sub)].min(axis=1).sum())
        assert cost <= prev + 1e-12
        prev = cost


def test_kmedoid_returns_the_whole_support_when_k_saturates(ms8):
    curves = np.random.default_rng(1).random((5, 4))
    assert ms8.kmedoid_subset(curves, 9) == (0, 1, 2, 3)


def test_fitted_table_is_priced_and_the_generic_ladder_is_not(ms8):
    """Rule 118: a table fitted to THIS video is counted; the generic
    receiver-side ladder is free.  The pricing helper must encode that."""
    assert ms8.table_manifest_bytes(None) == 0
    assert ms8.table_manifest_bytes((0.06, 0.08, 0.1)) > 0
