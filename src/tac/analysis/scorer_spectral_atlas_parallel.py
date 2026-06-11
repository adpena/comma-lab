# SPDX-License-Identifier: MIT
"""Cell-level PARALLELISM for the resumable scorer spectral-sensitivity atlas.

THE DEFECT THIS FIXES (operator directive 2026-06-11): the resumable atlas
(``scorer_spectral_atlas_runner.run_resumable_atlas``, commit ad07cb838) is
SINGLE-PROCESS. Each atlas cell scores a band-limited perturbation through the
EXACT torch ``DistortionNet`` on CPU (~1-2s/cell) and the cells are INDEPENDENT
(embarrassingly parallel), but the serial loop uses only ~2 of the M5 Max's 18
cores. The exhaustive 6400-cell sweep takes ~6 days serial; ~12 workers cut it
to ~1 day.

THE PARALLELIZATION (NO change to the scorer, the device, or any cell's value):
this module adds a worker POOL over cells. The PARENT process is the single
JSONL writer; WORKERS only compute. Concretely:

  * The parent reads the resume skip-set ONCE (``load_completed_cells``),
    enumerates the remaining cell keys, and feeds them to a shared task queue.
  * N worker PROCESSES (``multiprocessing`` ``spawn`` context — torch CPU + fork
    can deadlock; spawn is mandatory) each build their OWN ``FrozenScorer`` once,
    load the source pairs from the shared memmapped ``.raw`` (no 73MB pickle per
    worker), and pull cell keys, measuring each through the EXACT same
    ``_aggregate_cell`` + deterministic ``cell_seed_for`` codepath the serial
    runner uses.
  * Each worker caps its OWN torch thread count (``torch.set_num_threads`` +
    OMP/MKL env) so ``workers x threads`` does not oversubscribe the cores and
    starve the coexisting capstone daemon.
  * Workers push ``(key_str, cell_dict)`` back on a result queue; the PARENT
    appends each to the durable JSONL under the existing ``fcntl`` lock (single
    writer => no contention, no double-write) and refreshes the progress sidecar.

BIT-IDENTICAL TO SERIAL (the headline NO-FAKE guard): a cell's perturbation is
seeded from its INTRINSIC key (``cell_seed_for``), NOT from its position in the
sweep or which worker drew it. So a parallel run produces a JSONL whose per-cell
values are BIT-IDENTICAL to a serial (``--workers 1``) run for the same cells.
``src/tac/tests/test_scorer_spectral_atlas_parallel.py`` proves this on both the
stub scorer and (torch-gated) the EXACT ``DistortionNet``. Per CLAUDE.md
"Seeds pinned" + "NO FAKE IMPLEMENTATIONS".

Authority: ``[macOS-CPU advisory]`` / ``exact_pair_scorer`` ->
``mechanism_update_eligible`` ONLY (inherited from the v2 physics). NOT a score
row; does NOT update the score roadmap. NO MPS (CPU only, by design).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.analysis import scorer_spectral_atlas_runner as runner
from tac.analysis import scorer_spectral_sensitivity_v2 as v2

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as _np

__all__ = [
    "DEFAULT_TORCH_THREADS_PER_WORKER",
    "MAX_AUTO_WORKERS",
    "RESERVED_CORES",
    "WORKER_SCORER_FACTORY_ENV",
    "auto_worker_count",
    "measure_cell_by_key",
    "run_resumable_atlas_parallel",
]

# Test-only seam: a worker builds its scorer from this importable factory path
# ("module:attr") when the env var is set, instead of the real FrozenScorer.
# This lets the multiprocess bit-identity test run a DETERMINISTIC STUB scorer
# WITHOUT loading the 200ms+ torch DistortionNet, while exercising the EXACT
# spawn/queue/single-writer machinery. The real daemon never sets it -> always
# the exact FrozenScorer. (NO FAKE: the stub is content-derived, not constant —
# see the test; the seam changes nothing about the cell math or the writer.)
WORKER_SCORER_FACTORY_ENV = "TAC_ATLAS_WORKER_SCORER_FACTORY"


# ---------------------------------------------------------------------------
# Worker-count policy.
#
# auto = min(MAX_AUTO_WORKERS, max(1, physical_cores - RESERVED_CORES)).
# On the M5 Max (16 physical / 18 logical) this is min(12, max(1, 16-4)) = 12.
# RESERVED_CORES leaves headroom for the coexisting capstone daemon (~2.5 cores)
# + the OS. With DEFAULT_TORCH_THREADS_PER_WORKER == 1, total atlas thread
# pressure ~= workers (12), well under the 18 logical cores with the capstone.
# ---------------------------------------------------------------------------

MAX_AUTO_WORKERS = 12
RESERVED_CORES = 4
DEFAULT_TORCH_THREADS_PER_WORKER = 1


def _physical_core_count() -> int:
    """Best-effort physical core count (falls back to logical / 1)."""
    # os.cpu_count() is logical; prefer physical when available.
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        n = int(out.stdout.strip())
        if n > 0:
            return n
    except Exception:
        pass
    return os.cpu_count() or 1


def auto_worker_count(
    *,
    physical_cores: int | None = None,
    max_workers: int = MAX_AUTO_WORKERS,
    reserved_cores: int = RESERVED_CORES,
) -> int:
    """The auto worker count: ``min(max_workers, max(1, cores - reserved))``.

    Leaves ``reserved_cores`` free for the coexisting capstone daemon + the OS,
    and caps at ``max_workers`` so a future bigger machine does not spawn an
    unbounded pool.
    """
    cores = int(physical_cores) if physical_cores is not None else _physical_core_count()
    return int(min(int(max_workers), max(1, cores - int(reserved_cores))))


# ---------------------------------------------------------------------------
# The single-cell measurement (top-level, picklable for spawn).
#
# This reuses the EXACT serial physics: build_band_specs -> the same band ->
# cell_seed_for(intrinsic key) -> _aggregate_cell -> _cell_record. The ONLY
# input that varies per cell is the key; everything else (scorer, baseline,
# threshold, pairs) is fixed for the run. Because the seed is intrinsic to the
# key, the result is identical regardless of which process computes it.
# ---------------------------------------------------------------------------


def measure_cell_by_key(
    key: dict[str, Any],
    *,
    scorer: v2.FrozenScorer,
    source_pairs: _np.ndarray,
    grid: v2.AtlasGrid,
    baseline: dict[str, float],
    boundary_margin_thresh: float,
) -> dict[str, Any]:
    """Measure ONE atlas cell from its key, returning the canonical cell dict.

    Identical math to the body of :func:`scorer_spectral_sensitivity_v2.iter_atlas_cells`
    for a single cell: it rebuilds the cell's ``BandSpec`` (so the orientation
    wedge + radial annulus match exactly), derives the SAME deterministic seed
    from the intrinsic key, and runs the SAME ``_aggregate_cell``. This is what
    makes the parallel JSONL bit-identical to the serial one.
    """
    band_index = int(key["band_index"])
    orientation = str(key["orientation"])
    amplitude = float(key["amplitude_lsb"])
    basis = str(key["channel_basis"])
    channel = str(key["channel"])
    incidence = str(key["frame_incidence"])

    band_specs = v2.build_band_specs(
        grid.n_bands, orientation, spacing=grid.band_spacing
    )
    band = band_specs[band_index]
    coords = v2.frequency_coordinates_for_band(band)
    cell_seed = v2.cell_seed_for(
        global_seed=grid.seed,
        band_index=band_index,
        orientation=orientation,
        amplitude_lsb=amplitude,
        channel_basis=basis,
        channel=channel,
        frame_incidence=incidence,
    )
    result = v2._aggregate_cell(
        scorer,
        source_pairs,
        band,
        amplitude_lsb=amplitude,
        channel_basis=basis,
        channel=channel,
        incidence=incidence,
        n_phase_samples=grid.n_phase_samples,
        seed=cell_seed,
        boundary_margin_thresh=boundary_margin_thresh,
        baseline=baseline,
    )
    return v2._cell_record(
        band=band,
        orientation=orientation,
        amplitude=amplitude,
        basis=basis,
        channel=channel,
        incidence=incidence,
        coords=coords,
        result=result,
    )


# ---------------------------------------------------------------------------
# Worker process entry point (top-level for spawn-picklability).
# ---------------------------------------------------------------------------


def _pin_torch_threads(n: int) -> None:
    """Pin torch CPU threads to ``n`` (no-op if torch is unavailable / stub run).

    The bit-identity contract requires serial and every worker to run the
    float32 scorer reductions at the SAME thread count (reduction order is
    thread-count-dependent). The real daemon/parent + every worker call this
    with the same value (``torch_threads_per_worker``, default 1).
    """
    n = max(1, int(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass  # stub-only tests have no torch; nothing to pin.


def _load_scorer_from_factory(factory_path: str) -> Any:
    """Import + call a ``module:attr`` zero-arg scorer factory (test seam only)."""
    import importlib

    mod_name, _, attr = factory_path.partition(":")
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, attr)
    return factory()


def _cap_worker_threads(torch_threads: int) -> None:
    """Cap this worker's CPU thread oversubscription (torch + BLAS env)."""
    n = max(1, int(torch_threads))
    # These env vars must be set BEFORE the BLAS/OMP libs init; in a spawned
    # worker they are set before torch is imported in _worker_loop.
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))


def _worker_loop(
    worker_id: int,
    raw_path: str,
    raw_shape: tuple[int, int, int, int, int],
    grid: v2.AtlasGrid,
    baseline: dict[str, float],
    boundary_margin_thresh: float,
    device: str,
    torch_threads: int,
    task_q: mp.Queue[dict[str, Any] | None],
    result_q: mp.Queue[tuple[int, str, dict[str, Any]] | tuple[int, str, None]],
) -> None:
    """Pull cell keys off ``task_q``; push measured ``(worker_id, key_str, cell)``.

    Builds its OWN FrozenScorer once + memmaps the source pairs from the shared
    ``.raw`` (no 73MB pickle). On a ``None`` sentinel it exits. On a per-cell
    exception it pushes ``(worker_id, key_str, None)`` so the parent can record
    the failure without the whole pool dying (fail-loud at the parent).
    """
    import numpy as np

    _cap_worker_threads(torch_threads)

    factory_path = os.environ.get(WORKER_SCORER_FACTORY_ENV)
    if not factory_path:
        # The real path: cap torch threads then build the EXACT FrozenScorer.
        import torch

        torch.set_num_threads(max(1, int(torch_threads)))

    # Memmap the source pairs (read-only) — shared bytes, not a per-worker pickle.
    # The canonical source.raw may be a SUPERSET (decoded with more frames); slice
    # the leading raw_shape bytes so every worker sees the identical first-N pairs.
    expected = int(np.prod(raw_shape))
    mm = np.memmap(raw_path, dtype=np.uint8, mode="r")
    frames = np.asarray(mm[:expected]).reshape(raw_shape)
    source_pairs = np.array(frames)  # materialize a private read-only copy

    scorer = (
        _load_scorer_from_factory(factory_path)
        if factory_path
        else v2.FrozenScorer(device=device)
    )

    while True:
        task = task_q.get()
        if task is None:  # shutdown sentinel
            return
        key = task
        ks = v2.cell_key_str(key)
        try:
            cell = measure_cell_by_key(
                key,
                scorer=scorer,
                source_pairs=source_pairs,
                grid=grid,
                baseline=baseline,
                boundary_margin_thresh=boundary_margin_thresh,
            )
            result_q.put((worker_id, ks, cell))
        except Exception as exc:  # pragma: no cover - defensive; surfaced at parent
            result_q.put((worker_id, ks, None))
            # Also stash the error text on the queue payload via a second message
            # so the parent can fail loud; keep it simple: re-raise into stderr.
            import traceback

            print(
                f"[atlas.worker {worker_id}] cell {ks} FAILED: {exc!r}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# The parallel resumable run loop (parent process).
# ---------------------------------------------------------------------------


def run_resumable_atlas_parallel(
    source_pairs: _np.ndarray,
    grid: v2.AtlasGrid,
    paths: runner.AtlasRunPaths,
    *,
    tier: str,
    workers: int,
    raw_path: Path,
    device: str = "cpu",
    torch_threads_per_worker: int = DEFAULT_TORCH_THREADS_PER_WORKER,
    progress_every: int = 1,
    write_final_atlas: bool = True,
    scorer: v2.FrozenScorer | None = None,
    baseline: dict[str, float] | None = None,
    boundary_margin_thresh: float | None = None,
    mp_context: str = "spawn",
) -> dict[str, Any]:
    """Run (or RESUME) the atlas with a WORKER POOL over cells (parent = writer).

    Identical results to :func:`scorer_spectral_atlas_runner.run_resumable_atlas`
    (same cells, same values) but distributes the per-cell scorer forwards across
    ``workers`` processes. ``workers <= 1`` transparently delegates to the serial
    runner (so ``--workers 1`` IS the serial path, the bit-identity reference).

    * ``raw_path`` — the source ``.raw`` on the SSD that workers memmap (avoids a
      73MB pickle per worker). MUST hold ``source_pairs`` reshaped to
      ``(N, 2, H, W, 3)`` uint8.
    * The PARENT computes the baseline + boundary threshold ONCE (deterministic
      given the pairs) and ships the small dict + float to each worker, so the
      resume floor is identical to serial.
    * The PARENT is the single JSONL writer (reuses ``append_cell_jsonl`` under
      its fcntl lock), so there is no concurrent-writer contention and no cell is
      written twice. The remaining-key set (after the skip-set) partitions the
      work; each key is enqueued exactly once.
    """
    import numpy as np

    pairs = np.asarray(source_pairs)
    if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[-1] != 3:
        raise ValueError(
            f"source_pairs must be (N, 2, H, W, 3); got shape {pairs.shape}"
        )
    n_pairs = min(int(grid.n_pairs), pairs.shape[0])
    pairs = pairs[:n_pairs]

    workers = int(workers)
    # Pin torch CPU threads to the SAME value the workers use. This is REQUIRED
    # for bit-identity: torch's multi-threaded float32 reductions (e.g. the
    # kthvalue/sort behind logit_margin_drop_p10) are reduction-ORDER-dependent,
    # so an unpinned serial baseline and a 1-thread worker diverge at ~1e-7 (far
    # below contest precision, but it breaks the bit-identical NO-FAKE claim).
    # Fixing the thread count makes serial==parallel bit-identical. (Probed
    # 2026-06-11: same-thread runs are bit-identical; 1-vs-8 thread differ at the
    # 7th decimal.) NO MPS / CPU only.
    _pin_torch_threads(int(torch_threads_per_worker))
    if workers <= 1:
        # Serial path IS the bit-identity reference; delegate to the committed runner.
        return runner.run_resumable_atlas(
            pairs,
            grid,
            paths,
            tier=tier,
            device=device,
            progress_every=progress_every,
            write_final_atlas=write_final_atlas,
            scorer=scorer,
            baseline=baseline,
            boundary_margin_thresh=boundary_margin_thresh,
        )

    paths.work_dir.mkdir(parents=True, exist_ok=True)

    # Resume: which cells are already done?
    completed_keys, _completed_cells = runner.load_completed_cells(paths.cells_jsonl)
    total = grid.total_cells()
    started_utc = runner._utc_iso()

    # The parent measures the baseline + boundary threshold ONCE (deterministic),
    # then ships them to workers (so resume floor == serial floor). Building the
    # parent's own scorer also validates the model loads before fanning out.
    if scorer is None:
        factory_path = os.environ.get(WORKER_SCORER_FACTORY_ENV)
        scorer = (
            _load_scorer_from_factory(factory_path)
            if factory_path
            else v2.FrozenScorer(device=device)
        )
    if baseline is None or boundary_margin_thresh is None:
        baseline, boundary_margin_thresh = v2._measure_baseline_and_threshold(
            scorer, pairs
        )

    # Enumerate the REMAINING cell keys (skip the completed set). Each enqueued
    # exactly once => no double-compute, none skipped.
    all_keys = v2.enumerate_cell_keys(grid)
    remaining = [k for k in all_keys if v2.cell_key_str(k) not in completed_keys]

    done = len(completed_keys)
    print(
        f"[atlas.parallel] tier={tier} total_cells={total} "
        f"already_completed={len(completed_keys)} remaining={len(remaining)} "
        f"workers={workers} torch_threads/worker={torch_threads_per_worker} "
        f"device={device} work_dir={paths.work_dir}",
        flush=True,
    )

    runner.write_progress_sidecar(
        paths.progress_json,
        tier=tier,
        completed=done,
        total=total,
        started_utc=started_utc,
        last_cell_key=None,
        eta_seconds=None,
        status="in_progress",
    )

    if remaining:
        raw_shape = (int(pairs.shape[0]), 2, int(pairs.shape[2]), int(pairs.shape[3]), 3)
        # Persist the (possibly trimmed) source pairs to the raw the workers memmap.
        # The CLI passes the canonical source.raw; if a caller passes a different
        # raw_path, write the exact pairs there so workers see identical bytes.
        raw_path = Path(raw_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_raw_matches(raw_path, pairs, raw_shape)

        ctx = mp.get_context(mp_context)
        task_q: Any = ctx.Queue()
        result_q: Any = ctx.Queue()

        n_workers = min(workers, len(remaining))
        procs: list[Any] = []
        for wid in range(n_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(
                    wid,
                    str(raw_path),
                    raw_shape,
                    grid,
                    baseline,
                    boundary_margin_thresh,
                    device,
                    int(torch_threads_per_worker),
                    task_q,
                    result_q,
                ),
                daemon=True,
            )
            p.start()
            procs.append(p)

        # Feed all remaining keys, then one shutdown sentinel per worker.
        for key in remaining:
            task_q.put(key)
        for _ in range(n_workers):
            task_q.put(None)

        t0 = time.monotonic()
        new_this_run = 0
        last_key: str | None = None
        n_expected = len(remaining)
        n_received = 0
        failures: list[str] = []
        while n_received < n_expected:
            wid, ks, cell = result_q.get()
            n_received += 1
            if cell is None:
                failures.append(ks)
                continue
            done += 1
            new_this_run += 1
            last_key = ks
            # PARENT is the single writer (the existing fcntl-locked append).
            runner.append_cell_jsonl(
                paths.cells_jsonl, key=ks, cell=cell, cell_index=done
            )
            elapsed = time.monotonic() - t0
            eta = None
            if new_this_run > 0:
                per_cell = elapsed / new_this_run
                eta = per_cell * (total - done)
            if progress_every > 0 and (new_this_run % progress_every == 0 or done == total):
                runner.write_progress_sidecar(
                    paths.progress_json,
                    tier=tier,
                    completed=done,
                    total=total,
                    started_utc=started_utc,
                    last_cell_key=last_key,
                    eta_seconds=eta,
                    status="in_progress",
                )
                rate = (new_this_run / elapsed * 60.0) if elapsed > 0 else 0.0
                print(
                    f"[atlas.parallel] cell {done}/{total} (w{wid}) {ks} "
                    f"H_seg={cell['H_seg']:+.5f} H_pose={cell['H_pose']:+.4f} "
                    f"(new={new_this_run}, {rate:.1f} cells/min, eta~{(eta or 0)/3600:.2f}h)",
                    flush=True,
                )

        for p in procs:
            p.join(timeout=30)
            if p.is_alive():  # pragma: no cover - defensive
                p.terminate()

        if failures:
            # Fail loud: a cell that errored is NOT silently dropped. The JSONL is
            # still valid (it has every SUCCESSFUL cell); re-running RESUMES and
            # retries the failed keys (they are not in the skip-set).
            raise RuntimeError(
                f"[atlas.parallel] {len(failures)} cell(s) failed and were NOT "
                f"written; re-run to retry them (resume): {failures[:10]}"
            )

    # Re-aggregate the FINAL atlas from the FULL JSONL (skipped + new cells).
    _all_keys, all_cells = runner.load_completed_cells(paths.cells_jsonl)
    atlas = v2.aggregate_atlas_from_cells(
        all_cells,
        grid,
        baseline=baseline,
        boundary_margin_threshold=boundary_margin_thresh,
        n_pairs=n_pairs,
    )
    atlas["tier"] = tier
    atlas["utc"] = runner._utc_iso()
    atlas["resumed_from_jsonl"] = str(paths.cells_jsonl)
    atlas["cells_jsonl"] = str(paths.cells_jsonl)
    atlas["workers"] = workers
    atlas["lowering_opportunities"] = runner.analyze_lowering_opportunities(all_cells)

    if write_final_atlas:
        paths.atlas_json.parent.mkdir(parents=True, exist_ok=True)
        import json

        paths.atlas_json.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n")

    runner.write_progress_sidecar(
        paths.progress_json,
        tier=tier,
        completed=done,
        total=total,
        started_utc=started_utc,
        last_cell_key=None,
        eta_seconds=0.0,
        status="complete",
    )
    return atlas


def _ensure_raw_matches(
    raw_path: Path, pairs: _np.ndarray, raw_shape: tuple[int, int, int, int, int]
) -> None:
    """Make sure ``raw_path`` holds exactly ``pairs`` (uint8) for workers to memmap.

    If the file already exists with the exact expected byte length we trust it
    (the CLI's canonical source.raw is decoded deterministically). Otherwise we
    write the pairs' bytes so every worker memmaps identical source frames.
    """
    import numpy as np

    expected_bytes = int(np.prod(raw_shape))  # uint8 -> 1 byte each
    if raw_path.exists() and raw_path.stat().st_size >= expected_bytes:
        # Verify the leading bytes match the pairs we will measure against; if the
        # canonical source.raw was decoded with more frames it is a superset and
        # the worker memmap reshape uses raw_shape (the first N pairs) — identical.
        mm = np.memmap(raw_path, dtype=np.uint8, mode="r")
        head = np.array(mm[:expected_bytes]).reshape(raw_shape)
        if np.array_equal(head, np.asarray(pairs, dtype=np.uint8)):
            return
    np.asarray(pairs, dtype=np.uint8).tofile(raw_path)
