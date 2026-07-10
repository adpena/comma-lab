# SPDX-License-Identifier: MIT
"""Tests for the exact-score-gated latent click-polish search (task #399).

Fast tests (codec / round-trip / fold / resume / accounting) need no torch or GT.
Render tests (locality / batch-equivalence / monotone) load the frozen scorers +
a small GT cache and render a couple pairs on CPU; they are slower (tens of
seconds) but exercise the load-bearing correctness claims. All are skipped
cleanly if the frontier archive or GT cache is absent from the checkout.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tac import click_polish as cp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ARCHIVE = os.path.join(ROOT, cp.DEFAULT_ARCHIVE)
SUBDIR = os.path.join(ROOT, cp.DEFAULT_SUBMISSION_DIR)
UPSTREAM = os.path.join(ROOT, cp.DEFAULT_UPSTREAM)
GT_CACHE = os.path.join(ROOT, cp.DEFAULT_GT_CACHE)

_have_archive = os.path.exists(ARCHIVE) and os.path.isdir(SUBDIR)
_have_gt = os.path.isdir(GT_CACHE) and os.path.exists(os.path.join(GT_CACHE, "gt_n6.npz"))
_have_scorers = os.path.exists(os.path.join(UPSTREAM, "models", "segnet.safetensors"))

needs_archive = pytest.mark.skipif(not _have_archive, reason="frontier archive absent")
needs_render = pytest.mark.skipif(
    not (_have_archive and _have_gt and _have_scorers),
    reason="scorers/GT/archive absent",
)


@pytest.fixture(scope="module")
def packet():
    return cp.FrozenPacket.parse(ARCHIVE, SUBDIR)


# --------------------------------------------------------------------------- #
# fast codec / byte-custody tests (no torch)
# --------------------------------------------------------------------------- #
@needs_archive
def test_parse_shapes(packet):
    assert packet.Q0.shape == (cp.N_PAIRS, cp.LATENT_DIM)
    assert packet.Q0.dtype == np.uint8
    assert packet.mins.shape == (cp.LATENT_DIM,)
    assert packet.scales.shape == (cp.LATENT_DIM,)
    assert len(packet.sidecar) == 607  # our frontier carries the PR101 sidecar


@needs_archive
def test_latent_section_roundtrip_byte_exact(packet):
    # encode(decode(lat_sec)) == lat_sec  (the polished-section no-op detector)
    lat_sec = packet._original_lat_sec()
    latent_raw = packet.ns.codec_ctx.decode_latent_section(lat_sec)
    assert packet.ns.codec_ctx.encode_latent_section(latent_raw) == lat_sec


@needs_archive
def test_Q_latent_raw_roundtrip_byte_exact(packet):
    lat_sec = packet._original_lat_sec()
    latent_raw = packet.ns.codec_ctx.decode_latent_section(lat_sec)
    assert packet.latent_raw_from_Q(packet.Q0) == latent_raw


@needs_archive
def test_member_roundtrip_byte_exact(packet):
    assert packet.repack_member(packet.Q0) == packet.original_member


@needs_archive
def test_archive_roundtrip_byte_exact_and_sha(packet):
    rt = packet.verify_roundtrip()
    assert rt["archive_byte_exact"] is True
    assert rt["archive_sha256"] == cp.FRONTIER_ARCHIVE_SHA256
    assert rt["archive_bytes"] == 177169


@needs_archive
def test_noop_detector_unchanged_Q_is_identity(packet):
    # the no-op detector: no click => byte-identical archive
    assert packet.repack_archive_bytes(packet.Q0) == packet.original_archive


@needs_archive
def test_single_click_changes_only_latent_section(packet):
    Q = packet.Q0.copy()
    Q[10, 3] = np.clip(int(Q[10, 3]) + 1, 0, 255)
    m0, m1 = packet.original_member, packet.repack_member(Q)
    assert m1 != m0
    # decoder section, sidecar, selector, dqs1 tail must be byte-identical
    assert packet.repack_member(Q)[-len(packet.dqs1_tail):] == packet.dqs1_tail
    assert packet.sel_bytes in m1 and packet.dec_sec in m1 and packet.sidecar in m1


@needs_archive
def test_fold_sidecar_custody(packet):
    fold = packet.fold_sidecar_custody()
    assert fold["delta_matches_sidecar"] is True
    assert fold["archive_byte_delta"] == -607
    assert fold["before_sha256"] == cp.FRONTIER_ARCHIVE_SHA256
    assert fold["after_archive_bytes"] == 177169 - 607


@needs_archive
def test_drop_sidecar_repack_size(packet):
    kept = len(packet.repack_archive_bytes(packet.Q0, drop_sidecar=False))
    dropped = len(packet.repack_archive_bytes(packet.Q0, drop_sidecar=True))
    assert kept - dropped == len(packet.sidecar) == 607


@needs_archive
def test_rate_accounting_matches_stat(packet, tmp_path):
    # re-encode rate accounting == stat() of the written bytes
    archive = packet.repack_archive_bytes(packet.Q0)
    p = tmp_path / "a.zip"
    p.write_bytes(archive)
    assert os.path.getsize(p) == len(archive) == 177169


@needs_archive
def test_score_uses_canonical_helper(packet):
    from tac.contest_score import compute_contest_score

    S = compute_contest_score(0.0005, 3e-5, 177169)
    manual = 100 * 0.0005 + (10 * 3e-5) ** 0.5 + 25 * 177169 / 37_545_489
    assert abs(S - manual) < 1e-12


@needs_archive
def test_borrowed_substrate_accounting_keys():
    acc = cp.borrowed_substrate_accounting()
    assert acc["score_claim"] is False and acc["promotable"] is False
    assert "PR128" in acc["mechanism"]
    assert "PR110" in acc["substrate"] or "pr110" in acc["substrate"].lower()


# --------------------------------------------------------------------------- #
# resume: ledger replay reconstructs Q (no render needed)
# --------------------------------------------------------------------------- #
@needs_archive
def test_resume_roundtrip(packet, tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    # write a fake accepted-clicks ledger, then resume and check Q matches replay
    clicks = [[5, 2, 1], [7, 0, -2], [5, 2, 1]]  # note: pair 5 dim 2 clicked twice
    with open(os.path.join(out, "accepted_clicks_ledger.jsonl"), "w") as f:
        f.write(json.dumps({"round": 0, "clicks": [clicks[0], clicks[1]]}) + "\n")
        f.write(json.dumps({"round": 1, "clicks": [clicks[2]]}) + "\n")

    class _Stub(cp.ClickPolishSearch):
        def __post_init__(self):  # skip render/scorer wiring
            self.n = 8
            self.pairs = list(range(8))
            self.Q = packet.Q0.copy()
            self.out_dir = out
            self.ledger_path = os.path.join(out, "accepted_clicks_ledger.jsonl")

    s = _Stub(packet=packet, renderer=None, scorer=None,
              gt_lstars=np.zeros((8, 4, 4)), gt_poses=np.zeros((8, 6)), out_dir=out)
    rounds = s.resume()
    assert rounds == 2
    expected = packet.Q0.copy()
    expected[5, 2] = np.clip(int(expected[5, 2]) + 2, 0, 255)  # two +1 clicks
    expected[7, 0] = np.clip(int(expected[7, 0]) - 2, 0, 255)
    assert np.array_equal(s.Q, expected)


# --------------------------------------------------------------------------- #
# render-based correctness (slower; skipped if scorers/GT absent)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def render_ctx(packet):
    rnd = cp.Renderer(packet, device="cpu")
    scorer = cp.Scorer(upstream_dir=UPSTREAM, device="cpu")
    lstars, poses, _ = cp.load_gt_targets(GT_CACHE, 6)
    return packet, rnd, scorer, lstars, poses


@needs_render
def test_pair_locality_no_cross_talk(render_ctx):
    packet, rnd, scorer, lstars, poses = render_ctx
    loc = cp.verify_pair_locality(packet, rnd)
    assert loc["pair_b_unchanged_by_pair_a_click"] is True
    assert loc["pair_a_changed_by_its_click"] is True
    assert loc["locality_holds"] is True


@needs_render
def test_diagonal_batch_equals_sequential(render_ctx):
    packet, rnd, scorer, lstars, poses = render_ctx
    beq = cp.verify_batch_equivalence(
        packet, rnd, scorer, lstars, poses, pairs=(0, 1, 2, 3)
    )
    assert beq["diagonal_equals_sequential_seg_exact"] is True
    assert beq["pose_within_batch_float_tol"] is True
    assert beq["equivalence_holds"] is True


@needs_render
def test_exact_components_score_consistent(render_ctx):
    packet, rnd, scorer, lstars, poses = render_ctx
    from tac.contest_score import compute_contest_score

    comp = cp.exact_components_for_Q(
        packet, rnd, scorer, packet.Q0, lstars, poses, pair_indices=[0, 1]
    )
    for k in ("d_seg", "d_pose", "archive_bytes", "S"):
        assert k in comp
    assert abs(
        comp["S"]
        - compute_contest_score(comp["d_seg"], comp["d_pose"], comp["archive_bytes"])
    ) < 1e-12


@needs_render
@pytest.mark.timeout(600)
def test_monotone_accept_never_regresses(render_ctx, tmp_path, monkeypatch):
    packet, rnd, scorer, lstars, poses = render_ctx
    # lean the sweep for the test: 2 pairs, +-1 clicks only (still exact-gated)
    monkeypatch.setattr(cp, "SWEEP_DELTAS", (1, -1))
    ls, po = lstars[:2], poses[:2]
    search = cp.ClickPolishSearch(
        packet=packet, renderer=rnd, scorer=scorer, gt_lstars=ls, gt_poses=po,
        out_dir=str(tmp_path / "run"), axis_tag="[test]", max_rounds=1,
    )
    base = search._score_Q(packet.Q0)
    result = search.run()
    # the banked S must be <= the baseline S (monotone by construction)
    assert result["best_row"]["S"] <= base["S"] + 1e-12
    # candidate archive is a valid, correctly-sized zip
    assert os.path.exists(result["candidate_archive_path"])
    assert result["candidate_archive_bytes"] == os.path.getsize(
        result["candidate_archive_path"]
    )
