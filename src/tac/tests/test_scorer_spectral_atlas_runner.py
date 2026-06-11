# SPDX-License-Identifier: MIT
"""Behavioral tests for the resumable, multi-tier scorer-atlas runner.

NO FAKE Class-2 discipline: these verify the RESUME CONTRACT behaves correctly,
not constants. The headline guards:

* ``test_resume_is_bit_identical_to_uninterrupted`` — runs a sweep, KILLS it
  (truncates the JSONL to simulate a crash mid-write), RESUMES, and asserts the
  resumed atlas's per-cell values are BIT-IDENTICAL to an uninterrupted run AND
  that the skip-set was correct. If resume re-seeded differently (the bug this
  whole effort prevents) the values would differ and this FAILS.
* ``test_cell_seed_is_order_invariant_and_key_derived`` — the seed depends ONLY
  on the cell key + global seed, not on grid ordering/subset. A relabel would
  FAIL.
* ``test_truncated_final_jsonl_line_is_skipped_and_remeasured`` — a half-written
  final line (the crash case) is ignored on read; the cell is re-measured.
* ``test_lowering_analysis_spends_high_sheds_low`` — the analysis ranks the
  high-sensitivity axis values as spend-here and the blind cells as shed-here.

The runner is exercised with a DETERMINISTIC STUB scorer whose response is a
pure function of the perturbed-pair bytes (so a resumed cell, drawing the same
seeded field, produces the same response) — this makes the bit-identity claim
meaningful without loading the 200ms+ frozen DistortionNet. A separate
torch-gated test runs the REAL scorer end-to-end through the resume path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.analysis import scorer_spectral_atlas_runner as runner
from tac.analysis import scorer_spectral_sensitivity_v2 as v2


class _DeterministicStubScorer:
    """Stub FrozenScorer: response is a pure deterministic function of the pair.

    Crucially, the response depends on the PERTURBED PAIR CONTENTS — so a cell
    that draws the same seeded perturbation field produces the same response.
    This is what lets the resume-bit-identity test actually test bit-identity
    (rather than testing that a constant equals a constant).
    """

    def __init__(self) -> None:
        self.device = "cpu"
        self.calls = 0

    def response_levels(self, source_pair, perturbed_pair, *, boundary_margin_thresh):
        self.calls += 1
        src = np.asarray(source_pair, dtype=np.float64)
        prt = np.asarray(perturbed_pair, dtype=np.float64)
        delta = prt - src
        # A few content-derived scalars (deterministic given the perturbation).
        l2 = float(np.sqrt((delta**2).sum()))
        mae = float(np.abs(delta).mean())
        # Make the responses depend on the realized delta so distinct cells differ.
        d_seg = mae * 1e-3
        d_pose = l2 * 1e-4
        return {
            "d_logit_margin_mean": mae * 1e-2,
            "d_logit_margin_p10": -mae * 1e-3,
            "logit_l2_delta": l2 * 1e-2,
            "d_seg": d_seg,
            "flip_count_total": mae * 1e-4,
            "flip_count_boundary": mae * 5e-5,
            "flip_count_interior": mae * 5e-5,
            "d_pose": d_pose,
            "score_nonrate": 100.0 * d_seg + (10.0 * d_pose) ** 0.5,
        }


def _tiny_grid(seed: int = 0) -> v2.AtlasGrid:
    return v2.AtlasGrid(
        n_pairs=2,
        n_bands=3,
        band_spacing="log",
        amplitudes_lsb=(2.0, 8.0),
        orientations=("isotropic", "horizontal"),
        frame_incidences=("frame1_only", "both_opposite"),
        channel_bases=("yuv",),
        rgb_channels=("all",),
        yuv_channels=("y",),
        n_phase_samples=1,
        seed=seed,
    )


def _source_pairs(n: int = 2, rng_seed: int = 0) -> np.ndarray:
    h, w = 24, 32  # small camera-ish frame; physics is resolution-agnostic
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, 256, size=(n, 2, h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Seed / key determinism (the resume foundation).
# ---------------------------------------------------------------------------


def test_cell_seed_is_order_invariant_and_key_derived() -> None:
    """The seed depends ONLY on the cell key + global seed — not grid ordering."""
    kw = {
        "global_seed": 7,
        "band_index": 2,
        "orientation": "horizontal",
        "amplitude_lsb": 8.0,
        "channel_basis": "yuv",
        "channel": "y",
        "frame_incidence": "both_opposite",
    }
    s1 = v2.cell_seed_for(**kw)
    s2 = v2.cell_seed_for(**kw)
    assert s1 == s2  # deterministic
    # different cell -> (almost surely) different seed
    assert v2.cell_seed_for(**{**kw, "band_index": 1}) != s1
    # different global seed -> different seed
    assert v2.cell_seed_for(**{**kw, "global_seed": 8}) != s1


def test_enumerate_cell_keys_matches_grid_total() -> None:
    grid = _tiny_grid()
    keys = v2.enumerate_cell_keys(grid)
    assert len(keys) == grid.total_cells()
    # keys are unique
    strs = {v2.cell_key_str(k) for k in keys}
    assert len(strs) == len(keys)


def test_cell_key_str_is_amplitude_precision_stable() -> None:
    a = v2.cell_key_str(
        v2.cell_key(
            band_index=0, orientation="isotropic", amplitude_lsb=2.0,
            channel_basis="yuv", channel="y", frame_incidence="frame1_only",
        )
    )
    b = v2.cell_key_str(
        v2.cell_key(
            band_index=0, orientation="isotropic", amplitude_lsb=2.0000000001,
            channel_basis="yuv", channel="y", frame_incidence="frame1_only",
        )
    )
    assert a == b  # float noise does not split a logical cell


# ---------------------------------------------------------------------------
# JSONL custody + resume skip-set.
# ---------------------------------------------------------------------------


def test_append_and_load_completed_cells_roundtrip(tmp_path: Path) -> None:
    jsonl = tmp_path / "atlas_cells.jsonl"
    cell = {"band_index": 0, "orientation": "isotropic", "H_seg": 0.1, "H_pose": 0.2}
    runner.append_cell_jsonl(jsonl, key="b0|o:isotropic", cell=cell, cell_index=1)
    keys, cells = runner.load_completed_cells(jsonl)
    assert keys == {"b0|o:isotropic"}
    assert cells == [cell]


def test_load_completed_cells_last_writer_wins(tmp_path: Path) -> None:
    jsonl = tmp_path / "atlas_cells.jsonl"
    runner.append_cell_jsonl(jsonl, key="k", cell={"v": 1}, cell_index=1)
    runner.append_cell_jsonl(jsonl, key="k", cell={"v": 2}, cell_index=2)
    keys, cells = runner.load_completed_cells(jsonl)
    assert keys == {"k"}
    assert cells == [{"v": 2}]  # most-recent record wins (no double-count)


def test_truncated_final_jsonl_line_is_skipped_and_remeasured(tmp_path: Path) -> None:
    """A crash mid-write leaves a half-line; it must be ignored on read."""
    jsonl = tmp_path / "atlas_cells.jsonl"
    runner.append_cell_jsonl(jsonl, key="k1", cell={"v": 1}, cell_index=1)
    # simulate a crash that left a truncated (invalid) final JSON line
    with jsonl.open("a") as fh:
        fh.write('{"key": "k2", "cell_in')  # truncated, no newline
    keys, cells = runner.load_completed_cells(jsonl)
    assert keys == {"k1"}  # k2's partial line skipped -> will be re-measured
    assert cells == [{"v": 1}]


# ---------------------------------------------------------------------------
# THE HEADLINE GUARD: resume == uninterrupted, bit-for-bit.
# ---------------------------------------------------------------------------


def test_resume_is_bit_identical_to_uninterrupted(tmp_path: Path) -> None:
    grid = _tiny_grid(seed=3)
    pairs = _source_pairs(n=grid.n_pairs, rng_seed=1)
    baseline = {"d_seg": 0.0, "d_pose": 0.0, "d_logit_margin_mean": 0.0}
    thresh = 1.0

    # --- A) uninterrupted full run ---
    work_full = tmp_path / "full"
    paths_full = runner.AtlasRunPaths.under(work_full)
    atlas_full = runner.run_resumable_atlas(
        pairs, grid, paths_full, tier="quick", device="cpu", progress_every=0,
        scorer=_DeterministicStubScorer(), baseline=baseline,
        boundary_margin_thresh=thresh,
    )
    full_cells = {v2.cell_key_str(_key_of(c)): c for c in atlas_full["cells"]}
    assert len(full_cells) == grid.total_cells()

    # --- B) interrupted run: keep only the first ~60% of cells, then RESUME ---
    work_res = tmp_path / "resumed"
    paths_res = runner.AtlasRunPaths.under(work_res)
    # First, produce the full JSONL via a run, then truncate it to simulate a kill.
    runner.run_resumable_atlas(
        pairs, grid, paths_res, tier="quick", device="cpu", progress_every=0,
        scorer=_DeterministicStubScorer(), baseline=baseline,
        boundary_margin_thresh=thresh, write_final_atlas=False,
    )
    lines = paths_res.cells_jsonl.read_text().splitlines()
    keep = max(1, int(0.6 * len(lines)))
    # append a truncated partial line too (the realistic crash signature)
    paths_res.cells_jsonl.write_text(
        "\n".join(lines[:keep]) + "\n" + lines[keep][:20]
    )
    keys_before, _ = runner.load_completed_cells(paths_res.cells_jsonl)
    assert len(keys_before) == keep  # the truncated line was NOT counted

    # RESUME: a fresh scorer instance, fresh process state.
    scorer_resume = _DeterministicStubScorer()
    atlas_res = runner.run_resumable_atlas(
        pairs, grid, paths_res, tier="quick", device="cpu", progress_every=0,
        scorer=scorer_resume, baseline=baseline, boundary_margin_thresh=thresh,
    )
    # the resume only recomputed the REMAINING cells (skip-set worked)
    assert scorer_resume.calls == (grid.total_cells() - keep) * grid.n_pairs * grid.n_phase_samples

    res_cells = {v2.cell_key_str(_key_of(c)): c for c in atlas_res["cells"]}
    assert set(res_cells) == set(full_cells)  # all cells present after resume

    # BIT-IDENTITY: every cell value matches the uninterrupted run exactly.
    for k, cell in full_cells.items():
        assert res_cells[k] == cell, f"cell {k} differs after resume (NOT bit-identical)"

    # headline peaks also identical
    assert atlas_res["seg_peak_cell"] == atlas_full["seg_peak_cell"]
    assert atlas_res["pose_peak_cell"] == atlas_full["pose_peak_cell"]


def test_resume_writes_done_marker_and_progress(tmp_path: Path) -> None:
    grid = _tiny_grid(seed=0)
    pairs = _source_pairs(n=grid.n_pairs)
    baseline = {"d_seg": 0.0, "d_pose": 0.0, "d_logit_margin_mean": 0.0}
    work = tmp_path / "run"
    paths = runner.AtlasRunPaths.under(work)
    runner.run_resumable_atlas(
        pairs, grid, paths, tier="quick", device="cpu", progress_every=1,
        scorer=_DeterministicStubScorer(), baseline=baseline,
        boundary_margin_thresh=1.0,
    )
    # progress sidecar reports complete + 100%
    prog = json.loads(paths.progress_json.read_text())
    assert prog["status"] == "complete"
    assert prog["completed_cells"] == grid.total_cells()
    assert prog["fraction_complete"] == pytest.approx(1.0)
    # atlas json written + re-aggregatable
    atlas = json.loads(paths.atlas_json.read_text())
    assert atlas["cells_measured"] == grid.total_cells()
    assert atlas["promotable"] is False
    assert atlas["authority_tier"] == "exact_cpu_advisory"


def _key_of(cell: dict) -> dict:
    return v2.cell_key(
        band_index=cell["band_index"],
        orientation=cell["orientation"],
        amplitude_lsb=cell["amplitude_lsb"],
        channel_basis=cell["channel_basis"],
        channel=cell["channel"],
        frame_incidence=cell["frame_incidence"],
    )


# ---------------------------------------------------------------------------
# Tier presets.
# ---------------------------------------------------------------------------


def test_tier_presets_increase_in_size() -> None:
    q = runner.grid_for_tier("quick").total_cells()
    m = runner.grid_for_tier("medium").total_cells()
    e = runner.grid_for_tier("exhaustive").total_cells()
    assert q < m < e
    # quick is "minutes" scale, exhaustive is the killed-run scale (~thousands)
    assert q <= 64
    assert e >= 1000


def test_grid_for_tier_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        runner.grid_for_tier("ludicrous")


def test_all_tier_presets_have_valid_axes() -> None:
    for name, preset in runner.TIER_PRESETS.items():
        grid = preset.to_grid(seed=0)
        # every orientation/basis/channel/incidence is in the canonical label space
        for o in grid.orientations:
            assert o in v2.ORIENTATIONS, name
        for b in grid.channel_bases:
            assert b in v2.CHANNEL_BASES, name
        for fi in grid.frame_incidences:
            assert fi in v2.FRAME_INCIDENCES, name
        # cells enumerate cleanly
        assert len(v2.enumerate_cell_keys(grid)) == grid.total_cells()


# ---------------------------------------------------------------------------
# Lowering-opportunity analysis (the score-LOWERING lever).
# ---------------------------------------------------------------------------


def _cell(band, orient, ch, fi, h_seg, h_pose, w_equiv=10.0):
    return {
        "band_index": band,
        "orientation": orient,
        "amplitude_lsb": 8.0,
        "channel_basis": "yuv",
        "channel": ch,
        "frame_incidence": fi,
        "r_center": 0.1 * (band + 1),
        "H_seg": h_seg,
        "H_pose": h_pose,
        "frequency_coordinates": {
            "siren_w_equivalent": w_equiv,
            "scorer_cycles_per_pixel": 0.05,
            "aliases_at_scorer": False,
        },
    }


def test_lowering_analysis_spends_high_sheds_low() -> None:
    cells = [
        _cell(0, "horizontal", "y", "both_opposite", 0.001, 0.5),  # high pose response
        _cell(1, "horizontal", "y", "frame1_only", 0.002, 0.1),
        _cell(5, "vertical", "v", "frame0_only", 0.0, 0.0),  # scorer BLIND -> shed
        _cell(6, "vertical", "v", "frame0_only", 0.0, 1e-7),  # scorer BLIND -> shed
    ]
    lo = runner.analyze_lowering_opportunities(cells, top_k=4)
    # (a) spend-here: the high-response orientation/channel ranked first
    orient_rank = lo["spend_here_freq_budget"]["per_axis_ranking"]["orientation"]
    assert orient_rank[0]["value"] == "horizontal"  # where pose reacts most
    # (b) shed-here: the blind cells are flagged as shed-bytes opportunities
    shed_keys = {c["band_index"] for c in lo["shed_here_low_sensitivity_cells"]}
    assert 5 in shed_keys and 6 in shed_keys
    assert lo["consumer"] == "bit_allocator_waterfiller"
    assert lo["promotable"] is False


def test_lowering_analysis_empty_cells() -> None:
    lo = runner.analyze_lowering_opportunities([])
    assert lo["shed_here_low_sensitivity_cells"] == []
    assert lo["consumer"] == "bit_allocator_waterfiller"


def test_lowering_analysis_reports_per_band_siren_w() -> None:
    cells = [
        _cell(0, "horizontal", "y", "both_opposite", 0.0, 0.5, w_equiv=3.0),
        _cell(7, "horizontal", "y", "both_opposite", 0.0, 0.01, w_equiv=40.0),
    ]
    lo = runner.analyze_lowering_opportunities(cells)
    band_budget = lo["spend_here_freq_budget"]["per_band_with_siren_w"]
    # the highest-sensitivity band (band0, w=3) is ranked first -> the carrier's omega
    assert band_budget[0]["band_index"] == 0
    assert band_budget[0]["siren_w_equivalent"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# End-to-end through the FROZEN scorer via the resume path (torch-gated).
# ---------------------------------------------------------------------------


def _scorer_available() -> bool:
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return False
        repo = Path(__file__).resolve().parents[3]
        seg = repo / "upstream" / "models" / "segnet.safetensors"
        pose = repo / "upstream" / "models" / "posenet.safetensors"
        return seg.exists() and pose.exists()
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.timeout(0)
@pytest.mark.skipif(not _scorer_available(), reason="frozen scorer weights / torch unavailable")
def test_resume_end_to_end_real_scorer_bit_identical(tmp_path: Path) -> None:
    """Resume through the REAL DistortionNet must be bit-identical (the authority
    surface of the NO-FAKE resume claim). Tiny grid (loads the scorer once)."""
    import sys

    repo = Path(__file__).resolve().parents[3]
    for p in (str(repo / "src"), str(repo / "upstream"), str(repo / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)

    H, W = v2.CAMERA_HW
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, 256, size=(1, 2, H, W, 3), dtype=np.uint8)
    grid = v2.AtlasGrid(
        n_pairs=1, n_bands=2, band_spacing="log", amplitudes_lsb=(8.0,),
        orientations=("isotropic",), frame_incidences=("frame1_only", "both_opposite"),
        channel_bases=("yuv",), rgb_channels=("all",), yuv_channels=("y",),
        n_phase_samples=1, seed=0,
    )

    # A) full run
    paths_full = runner.AtlasRunPaths.under(tmp_path / "full")
    atlas_full = runner.run_resumable_atlas(
        pairs, grid, paths_full, tier="quick", device="cpu", progress_every=0
    )
    # B) kill after 1 cell, resume
    paths_res = runner.AtlasRunPaths.under(tmp_path / "res")
    runner.run_resumable_atlas(
        pairs, grid, paths_res, tier="quick", device="cpu", progress_every=0,
        write_final_atlas=False,
    )
    lines = paths_res.cells_jsonl.read_text().splitlines()
    paths_res.cells_jsonl.write_text(lines[0] + "\n")  # keep only 1 cell
    atlas_res = runner.run_resumable_atlas(
        pairs, grid, paths_res, tier="quick", device="cpu", progress_every=0
    )
    full = {v2.cell_key_str(_key_of(c)): c for c in atlas_full["cells"]}
    res = {v2.cell_key_str(_key_of(c)): c for c in atlas_res["cells"]}
    assert set(full) == set(res)
    for k in full:
        assert res[k] == full[k], f"REAL-scorer cell {k} not bit-identical after resume"
