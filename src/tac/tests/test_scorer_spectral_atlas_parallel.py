# SPDX-License-Identifier: MIT
"""Behavioral tests for CELL-LEVEL PARALLELISM of the resumable scorer atlas.

NO FAKE Class-2 discipline: these verify the PARALLEL CONTRACT behaves
correctly, not constants. The headline guard:

* ``test_parallel_jsonl_is_bit_identical_to_serial`` — a parallel run
  (workers>1, real subprocesses) produces a JSONL whose per-cell values are
  BIT-IDENTICAL to a serial (workers=1) run for the same cells. If a worker
  re-seeded off its position-in-queue instead of the intrinsic cell key (the
  bug this prevents) the values would differ and this FAILS.

* ``test_real_scorer_parallel_bit_identical_to_serial`` (torch-gated) — the
  same bit-identity claim through the EXACT DistortionNet across real worker
  processes (the authority surface).

* ``test_parallel_writes_every_remaining_cell_exactly_once`` — no cell computed
  twice, none skipped (the partition correctness NO-FAKE guard).

* ``test_parallel_resume_then_parallel_is_bit_identical`` — kill a parallel run
  mid-way, resume in parallel, assert still bit-identical to uninterrupted.

* ``test_auto_worker_count_leaves_headroom`` — the auto policy reserves cores
  for the coexisting capstone daemon and caps the pool.

The multiprocess tests run a DETERMINISTIC STUB scorer (content-derived, NOT a
constant) injected via the ``TAC_ATLAS_WORKER_SCORER_FACTORY`` test seam so the
EXACT spawn/queue/single-writer machinery is exercised WITHOUT loading the
200ms+ torch DistortionNet. The seam changes nothing about the cell math or the
single-writer JSONL append (the real daemon never sets it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.analysis import scorer_spectral_atlas_parallel as parallel
from tac.analysis import scorer_spectral_atlas_runner as runner
from tac.analysis import scorer_spectral_sensitivity_v2 as v2

# ---------------------------------------------------------------------------
# A deterministic, content-derived stub scorer (importable for the spawn seam).
# ---------------------------------------------------------------------------


class _DeterministicStubScorer:
    """Response is a pure deterministic function of the perturbed-pair bytes.

    Because the response depends on the realized perturbation (which is drawn
    from the cell's intrinsic deterministic seed), a cell measured by ANY worker
    in ANY order produces the SAME response — which is exactly what the
    bit-identity test must be able to detect a violation of.
    """

    def __init__(self) -> None:
        self.device = "cpu"

    def response_levels(self, source_pair, perturbed_pair, *, boundary_margin_thresh):
        src = np.asarray(source_pair, dtype=np.float64)
        prt = np.asarray(perturbed_pair, dtype=np.float64)
        delta = prt - src
        l2 = float(np.sqrt((delta**2).sum()))
        mae = float(np.abs(delta).mean())
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


def make_stub_scorer() -> _DeterministicStubScorer:
    """Zero-arg factory referenced by the worker via the test seam env var."""
    return _DeterministicStubScorer()


_STUB_FACTORY = f"{__name__}:make_stub_scorer"
# Deterministic baseline + threshold (the stub has no torch to measure them).
_STUB_BASELINE = {"d_seg": 0.0, "d_pose": 0.0, "d_logit_margin_mean": 0.0}
_STUB_THRESH = 0.5


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


def _key_of(cell: dict) -> dict:
    return v2.cell_key(
        band_index=cell["band_index"],
        orientation=cell["orientation"],
        amplitude_lsb=cell["amplitude_lsb"],
        channel_basis=cell["channel_basis"],
        channel=cell["channel"],
        frame_incidence=cell["frame_incidence"],
    )


def _cells_by_key(jsonl: Path) -> dict[str, dict]:
    _keys, cells = runner.load_completed_cells(jsonl)
    return {v2.cell_key_str(_key_of(c)): c for c in cells}


# ---------------------------------------------------------------------------
# Unit-level bit-identity: measure_cell_by_key == the serial iter cell.
# ---------------------------------------------------------------------------


def test_measure_cell_by_key_matches_serial_iter_cell() -> None:
    """One parallel-path cell == the serial generator's cell, value-for-value.

    This is the per-cell proof that distributing cells across processes cannot
    change a value: both paths rebuild the SAME band, derive the SAME intrinsic
    seed, and call the SAME _aggregate_cell.
    """
    grid = _tiny_grid()
    pairs = _source_pairs(grid.n_pairs).astype(np.float64)
    stub = _DeterministicStubScorer()

    serial_cells = {
        ks: cell
        for cell, ks in v2.iter_atlas_cells(
            pairs,
            grid,
            scorer=stub,
            baseline=_STUB_BASELINE,
            boundary_margin_thresh=_STUB_THRESH,
        )
    }
    assert serial_cells, "serial generator produced no cells"

    for key in v2.enumerate_cell_keys(grid):
        ks = v2.cell_key_str(key)
        cell = parallel.measure_cell_by_key(
            key,
            scorer=stub,
            source_pairs=pairs,
            grid=grid,
            baseline=_STUB_BASELINE,
            boundary_margin_thresh=_STUB_THRESH,
        )
        assert cell == serial_cells[ks], f"cell {ks} differs from serial"


# ---------------------------------------------------------------------------
# THE HEADLINE GUARD: parallel JSONL == serial JSONL (multiprocess, stub).
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_seam(monkeypatch):
    monkeypatch.setenv(parallel.WORKER_SCORER_FACTORY_ENV, _STUB_FACTORY)
    yield


def _run_parallel_stub(tmp_path: Path, *, workers: int, grid, pairs) -> Path:
    work = tmp_path / f"w{workers}"
    work.mkdir(parents=True, exist_ok=True)
    paths = runner.AtlasRunPaths.under(work)
    raw = work / "source.raw"
    parallel.run_resumable_atlas_parallel(
        pairs,
        grid,
        paths,
        tier="quick",
        workers=workers,
        raw_path=raw,
        device="cpu",
        torch_threads_per_worker=1,
        progress_every=0,
        baseline=_STUB_BASELINE,
        boundary_margin_thresh=_STUB_THRESH,
        scorer=make_stub_scorer(),  # parent baseline (already provided) — unused for measure
    )
    return paths.cells_jsonl


def test_parallel_jsonl_is_bit_identical_to_serial(tmp_path: Path, _stub_seam) -> None:
    grid = _tiny_grid()
    pairs = _source_pairs(grid.n_pairs)

    serial_jsonl = _run_parallel_stub(tmp_path, workers=1, grid=grid, pairs=pairs)
    parallel_jsonl = _run_parallel_stub(tmp_path, workers=3, grid=grid, pairs=pairs)

    serial = _cells_by_key(serial_jsonl)
    par = _cells_by_key(parallel_jsonl)
    assert set(serial) == set(par), "parallel covered a different cell set than serial"
    assert serial, "no cells measured"
    for k in serial:
        assert par[k] == serial[k], f"parallel cell {k} not bit-identical to serial"


def test_parallel_writes_every_remaining_cell_exactly_once(
    tmp_path: Path, _stub_seam
) -> None:
    """No double-write, no skip: the JSONL has exactly total_cells unique keys."""
    grid = _tiny_grid()
    pairs = _source_pairs(grid.n_pairs)
    jsonl = _run_parallel_stub(tmp_path, workers=4, grid=grid, pairs=pairs)

    raw_lines = [ln for ln in jsonl.read_text().splitlines() if ln.strip()]
    import json as _json

    keys = [_json.loads(ln)["key"] for ln in raw_lines]
    assert len(keys) == grid.total_cells(), "wrong number of cells written"
    assert len(set(keys)) == len(keys), "a cell was written more than once"
    assert set(keys) == {v2.cell_key_str(k) for k in v2.enumerate_cell_keys(grid)}


def test_parallel_resume_then_parallel_is_bit_identical(
    tmp_path: Path, _stub_seam
) -> None:
    """Kill a parallel run mid-way, resume in parallel -> still bit-identical."""
    grid = _tiny_grid()
    pairs = _source_pairs(grid.n_pairs)

    # Reference: uninterrupted parallel run.
    ref_jsonl = _run_parallel_stub(tmp_path, workers=2, grid=grid, pairs=pairs)
    ref = _cells_by_key(ref_jsonl)

    # Resumed: write a partial JSONL (first ~40% of cells), then resume parallel.
    work = tmp_path / "resumed"
    work.mkdir(parents=True, exist_ok=True)
    paths = runner.AtlasRunPaths.under(work)
    full_lines = [ln for ln in ref_jsonl.read_text().splitlines() if ln.strip()]
    keep = max(1, int(0.4 * len(full_lines)))
    paths.cells_jsonl.write_text("\n".join(full_lines[:keep]) + "\n")
    completed_before, _ = runner.load_completed_cells(paths.cells_jsonl)
    assert completed_before, "resume skip-set should be non-empty"

    raw = work / "source.raw"
    parallel.run_resumable_atlas_parallel(
        pairs,
        grid,
        paths,
        tier="quick",
        workers=3,
        raw_path=raw,
        device="cpu",
        progress_every=0,
        baseline=_STUB_BASELINE,
        boundary_margin_thresh=_STUB_THRESH,
        scorer=make_stub_scorer(),
    )
    resumed = _cells_by_key(paths.cells_jsonl)
    assert set(resumed) == set(ref)
    for k in ref:
        assert resumed[k] == ref[k], f"resumed-parallel cell {k} not bit-identical"


def test_serial_path_when_workers_le_1_delegates(tmp_path: Path, _stub_seam) -> None:
    """workers<=1 transparently uses the serial runner (the bit-identity ref)."""
    grid = _tiny_grid()
    pairs = _source_pairs(grid.n_pairs)
    work = tmp_path / "serial"
    paths = runner.AtlasRunPaths.under(work)
    raw = work / "source.raw"
    atlas = parallel.run_resumable_atlas_parallel(
        pairs,
        grid,
        paths,
        tier="quick",
        workers=1,
        raw_path=raw,
        device="cpu",
        progress_every=0,
        scorer=make_stub_scorer(),
        baseline=_STUB_BASELINE,
        boundary_margin_thresh=_STUB_THRESH,
    )
    assert atlas["cells_measured"] == grid.total_cells()


# ---------------------------------------------------------------------------
# auto worker-count policy.
# ---------------------------------------------------------------------------


def test_auto_worker_count_leaves_headroom() -> None:
    # M5 Max: 16 physical -> min(12, 16-4) = 12.
    assert parallel.auto_worker_count(physical_cores=16) == 12
    # small machine: min(12, 8-4) = 4
    assert parallel.auto_worker_count(physical_cores=8) == 4
    # tiny machine never goes below 1
    assert parallel.auto_worker_count(physical_cores=2) == 1
    assert parallel.auto_worker_count(physical_cores=1) == 1
    # cap holds on a huge machine
    assert parallel.auto_worker_count(physical_cores=128) == parallel.MAX_AUTO_WORKERS


def test_auto_worker_count_default_uses_machine_cores() -> None:
    n = parallel.auto_worker_count()
    assert 1 <= n <= parallel.MAX_AUTO_WORKERS


# ---------------------------------------------------------------------------
# The raw-bytes contract (workers memmap the identical source pairs).
# ---------------------------------------------------------------------------


def test_ensure_raw_matches_writes_and_trusts(tmp_path: Path) -> None:
    pairs = _source_pairs(2).astype(np.uint8)
    shape = (2, 2, 24, 32, 3)
    raw = tmp_path / "source.raw"
    parallel._ensure_raw_matches(raw, pairs, shape)
    assert raw.exists()
    mm = np.memmap(raw, dtype=np.uint8, mode="r")
    got = np.asarray(mm[: int(np.prod(shape))]).reshape(shape)
    assert np.array_equal(got, pairs)
    # idempotent: a second call with matching bytes is a no-op (mtime may change
    # but content identical).
    parallel._ensure_raw_matches(raw, pairs, shape)
    mm2 = np.memmap(raw, dtype=np.uint8, mode="r")
    got2 = np.asarray(mm2[: int(np.prod(shape))]).reshape(shape)
    assert np.array_equal(got2, pairs)


def test_ensure_raw_matches_handles_superset_source_raw(tmp_path: Path) -> None:
    """A canonical source.raw decoded with MORE pairs is a valid superset; the
    leading N-pair bytes the worker slices must equal the measured pairs."""
    big = _source_pairs(4).astype(np.uint8)
    raw = tmp_path / "source.raw"
    big.tofile(raw)
    first_two = np.ascontiguousarray(big[:2])
    shape = (2, 2, 24, 32, 3)
    # _ensure_raw_matches should TRUST the existing superset (leading bytes match).
    parallel._ensure_raw_matches(raw, first_two, shape)
    assert raw.stat().st_size == big.nbytes  # not truncated/rewritten
    mm = np.memmap(raw, dtype=np.uint8, mode="r")
    head = np.asarray(mm[: int(np.prod(shape))]).reshape(shape)
    assert np.array_equal(head, first_two)


# ---------------------------------------------------------------------------
# End-to-end through the FROZEN scorer via the parallel path (torch-gated).
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
def test_real_scorer_parallel_bit_identical_to_serial(tmp_path: Path) -> None:
    """Parallel (workers>1, real subprocesses, EXACT DistortionNet) == serial.

    The authority-surface NO-FAKE guard: distributing cells across processes
    cannot change any cell's value. Tiny grid so each worker loads the scorer
    once and the test stays bounded.
    """
    import sys

    repo = Path(__file__).resolve().parents[3]
    for p in (str(repo / "src"), str(repo / "upstream"), str(repo / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Real camera-res pairs decoded from the contest source (the exact pipeline).
    import hi_nerv_renderer_sanity_ladder as ladder

    H, W, C = ladder._frame_dims()
    work_src = tmp_path / "src"
    work_src.mkdir(parents=True, exist_ok=True)
    src_raw = work_src / "source.raw"
    ladder.decode_source_to_raw(
        repo / "upstream" / "videos" / "0.mkv", src_raw, max_frames=2
    )
    frames = np.memmap(src_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    pairs = np.array(frames[:2]).reshape(1, 2, H, W, C)

    grid = v2.AtlasGrid(
        n_pairs=1,
        n_bands=2,
        band_spacing="log",
        amplitudes_lsb=(8.0,),
        orientations=("isotropic",),
        frame_incidences=("frame1_only", "both_opposite"),
        channel_bases=("yuv",),
        rgb_channels=("all",),
        yuv_channels=("y",),
        n_phase_samples=1,
        seed=0,
    )

    # Serial reference.
    paths_serial = runner.AtlasRunPaths.under(tmp_path / "serial")
    paths_serial.work_dir.mkdir(parents=True, exist_ok=True)
    raw_s = paths_serial.work_dir / "source.raw"
    np.asarray(pairs, dtype=np.uint8).tofile(raw_s)
    parallel.run_resumable_atlas_parallel(
        pairs, grid, paths_serial, tier="quick", workers=1, raw_path=raw_s,
        device="cpu", progress_every=0,
    )

    # Parallel (2 workers, real subprocesses).
    paths_par = runner.AtlasRunPaths.under(tmp_path / "par")
    paths_par.work_dir.mkdir(parents=True, exist_ok=True)
    raw_p = paths_par.work_dir / "source.raw"
    np.asarray(pairs, dtype=np.uint8).tofile(raw_p)
    parallel.run_resumable_atlas_parallel(
        pairs, grid, paths_par, tier="quick", workers=2, raw_path=raw_p,
        device="cpu", torch_threads_per_worker=1, progress_every=0,
    )

    serial = _cells_by_key(paths_serial.cells_jsonl)
    par = _cells_by_key(paths_par.cells_jsonl)
    assert set(serial) == set(par)
    assert serial, "no cells measured"
    for k in serial:
        assert par[k] == serial[k], f"REAL-scorer parallel cell {k} != serial"
