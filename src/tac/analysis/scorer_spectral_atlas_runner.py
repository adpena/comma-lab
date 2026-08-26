# SPDX-License-Identifier: MIT
"""Resumable, multi-tier driver for the scorer spectral-sensitivity atlas (v2).

THE DEFECT THIS FIXES (operator standing directive 2026-06-11, the killed 31h
runaway): the v2 atlas held every cell in memory and only wrote a final JSON at
the END, so a kill/crash/reboot lost ALL progress. This module makes the sweep
**resumable by construction**: each measured cell is appended to a durable JSONL
on the SSD tier AS it is computed; on startup the runner reads the JSONL, builds
the set of completed cell-keys, and SKIPS them (idempotent resume). A
kill/crash/reboot loses at most the in-flight cell.

THE BIT-IDENTICAL-RESUME CONTRACT (NO FAKE): every cell's perturbation is seeded
deterministically from its cell-key + the global seed
(``tac.analysis.scorer_spectral_sensitivity_v2.cell_seed_for``), so a resumed run
produces results bit-identical to an uninterrupted run for the same cells,
regardless of the order cells are visited. Per CLAUDE.md "Seeds pinned".

TIERS: ``quick`` (~30 cells, minutes — coarse peak), ``medium`` (~256 cells,
~1-2h — the right-sized actionable-bands recipe), ``exhaustive`` (the full
cross-product, ~days — full CI). Explicit grid flags override the preset.

This is the reusable orchestration (JSONL custody, resume, tier presets,
progress sidecar, DONE.marker, the lowering-opportunity analysis). The thin CLI
``tools/measure_scorer_spectral_sensitivity.py v2-resume`` delegates here.

Authority: ``[macOS-CPU advisory]`` / ``exact_pair_scorer`` ->
``mechanism_update_eligible`` ONLY (inherited from the v2 physics). NOT a score
row; does NOT update the score roadmap.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.analysis import scorer_spectral_sensitivity_v2 as v2

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as _np

__all__ = [
    "TIER_PRESETS",
    "AtlasRunPaths",
    "TierPreset",
    "analyze_lowering_opportunities",
    "append_cell_jsonl",
    "grid_for_tier",
    "load_completed_cells",
    "read_cells_jsonl",
    "run_resumable_atlas",
    "write_done_marker",
    "write_progress_sidecar",
]


# ---------------------------------------------------------------------------
# Tier presets — quick / medium / exhaustive.
#
# Each preset configures the v2 AtlasGrid cross-product size. The dimensions
# kept are the SIGNAL-BEARING ones per the partial-atlas finding (the killed
# run's cells 1-625): pose reacts to LOW-freq + HORIZONTAL; seg is broadly weak;
# the two amplitudes {2, 8} LSB and the two orientations {isotropic, horizontal}
# and two incidences {frame1_only, both_opposite} carry the signal. quick keeps
# the bare minimum to locate the coarse peak; medium is the right-sized
# actionable-bands recipe from the kill memo (~256 cells); exhaustive is the
# full cross-product for CI.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierPreset:
    """A named resolution tier -> the v2 grid axes it sweeps."""

    name: str
    n_pairs: int
    n_bands: int
    band_spacing: str
    amplitudes_lsb: tuple[float, ...]
    orientations: tuple[str, ...]
    frame_incidences: tuple[str, ...]
    channel_bases: tuple[str, ...]
    rgb_channels: tuple[str, ...]
    yuv_channels: tuple[str, ...]
    n_phase_samples: int
    note: str

    def to_grid(self, *, seed: int) -> v2.AtlasGrid:
        return v2.AtlasGrid(
            n_pairs=self.n_pairs,
            n_bands=self.n_bands,
            band_spacing=self.band_spacing,
            amplitudes_lsb=self.amplitudes_lsb,
            orientations=self.orientations,
            frame_incidences=self.frame_incidences,
            channel_bases=self.channel_bases,
            rgb_channels=self.rgb_channels,
            yuv_channels=self.yuv_channels,
            n_phase_samples=self.n_phase_samples,
            seed=seed,
        )


TIER_PRESETS: dict[str, TierPreset] = {
    # ~30 cells: 1 amplitude x 2 orientations x 1 incidence x 1 channel x ~8 bands
    # x ... -> coarse peak in minutes. (8 bands * 2 orient * 1 amp * 1 ch * 1 inc = 16;
    # add the second incidence to reach ~32.)
    "quick": TierPreset(
        name="quick",
        n_pairs=4,
        n_bands=8,
        band_spacing="log",
        amplitudes_lsb=(8.0,),
        orientations=("isotropic", "horizontal"),
        frame_incidences=("frame1_only", "both_opposite"),
        channel_bases=("yuv",),
        rgb_channels=("all",),
        yuv_channels=("y",),
        n_phase_samples=1,
        note="coarse peak (~32 cells, minutes); 1 amp x 2 orient x 2 incidence x Y x 8 bands",
    ),
    # ~256 cells: the right-sized recipe from the kill memo. 2 amp x 2 orient x
    # 2 incidence x (yuv:{y} + rgb:{all} = 2 ch) x 8 bands = 128; doubling pairs
    # and adding the U channel reaches ~actionable density in ~1-2h.
    "medium": TierPreset(
        name="medium",
        n_pairs=12,
        n_bands=8,
        band_spacing="log",
        amplitudes_lsb=(2.0, 8.0),
        orientations=("isotropic", "horizontal"),
        frame_incidences=("frame1_only", "both_opposite"),
        channel_bases=("yuv", "rgb"),
        rgb_channels=("all",),
        yuv_channels=("y", "u"),
        n_phase_samples=1,
        # cells = 8 bands * 2 orient * 2 amp * (yuv:2 + rgb:1 = 3 ch) * 2 inc = 192
        note="actionable bands (~192 cells, ~1-2h); 2 amp x 2 orient x 2 inc x {Y,U,RGB} x 8 log bands",
    ),
    # POST-MORTEM: the full cross-product used by the retired run. ~days on CPU.
    "exhaustive": TierPreset(
        name="exhaustive",
        n_pairs=12,
        n_bands=8,
        band_spacing="log",
        amplitudes_lsb=(0.5, 1.0, 2.0, 4.0, 8.0),
        orientations=("isotropic", "horizontal", "vertical", "diag_plus", "diag_minus"),
        frame_incidences=("frame0_only", "frame1_only", "both_same", "both_opposite"),
        channel_bases=("rgb", "yuv"),
        rgb_channels=("all", "r", "g", "b"),
        yuv_channels=("all", "y", "u", "v"),
        n_phase_samples=2,
        # cells = 8 * 5 orient * 5 amp * (rgb:4 + yuv:4 = 8 ch) * 4 inc = 6400
        note="full cross-product (~6400 cells, ~days, full CI); the killed run's scale, now checkpointed",
    ),
}


def grid_for_tier(tier: str, *, seed: int = 0) -> v2.AtlasGrid:
    """Return the v2 ``AtlasGrid`` for a named tier preset."""
    if tier not in TIER_PRESETS:
        raise ValueError(
            f"unknown tier {tier!r}; expected one of {sorted(TIER_PRESETS)}"
        )
    return TIER_PRESETS[tier].to_grid(seed=seed)


# ---------------------------------------------------------------------------
# Durable JSONL custody (the resume store) + sidecars.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasRunPaths:
    """The canonical durable file layout for one resumable atlas run."""

    work_dir: Path
    cells_jsonl: Path
    """Per-cell incremental store (one JSON line per completed cell)."""
    progress_json: Path
    """Heartbeat/progress sidecar (re-read cheaply by an external check)."""
    done_marker: Path
    """Written on exit with the exit code + completed-cell count."""
    atlas_json: Path
    """The aggregated final atlas (re-aggregatable from the JSONL any time)."""

    @classmethod
    def under(cls, work_dir: Path, *, atlas_out: Path | None = None) -> AtlasRunPaths:
        work = Path(work_dir)
        return cls(
            work_dir=work,
            cells_jsonl=work / "atlas_cells.jsonl",
            progress_json=work / "atlas_progress.json",
            done_marker=work / "DONE.marker",
            atlas_json=atlas_out if atlas_out is not None else work / "atlas.json",
        )


def read_cells_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every cell record from the JSONL (skips blank/corrupt trailing lines).

    A kill mid-write can leave a truncated final line; that line is skipped (the
    in-flight cell is simply re-measured on resume — idempotent).
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # truncated/partial final line from an interrupted write — skip.
                continue
            if isinstance(rec, dict) and "cell" in rec and "key" in rec:
                out.append(rec)
    return out


def load_completed_cells(
    path: Path,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Return ``(completed_key_strs, completed_cell_dicts)`` from the JSONL.

    Last-writer-wins per key (a duplicate key keeps the most recent record), so a
    re-run that re-measured a cell does not double-count it in aggregation.
    """
    records = read_cells_jsonl(path)
    by_key: dict[str, dict[str, Any]] = {}
    for rec in records:
        by_key[str(rec["key"])] = rec["cell"]
    return set(by_key), list(by_key.values())


def append_cell_jsonl(
    path: Path, *, key: str, cell: dict[str, Any], cell_index: int
) -> None:
    """Append one cell record to the JSONL under an exclusive fcntl lock.

    The record is ``{"key": <cell_key_str>, "cell": <cell_dict>, "cell_index":
    <int>, "ts_utc": <iso>}``. The lock + flush + fsync make the append atomic
    enough that a concurrent reader (the external progress check) never sees a
    half-line and a crash loses at most the line being written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": str(key),
        "cell_index": int(cell_index),
        "ts_utc": _utc_iso(),
        "cell": cell,
    }
    line = json.dumps(record, sort_keys=True) + "\n"
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            # If a prior crash left the final line without a trailing newline,
            # appending would MERGE the new line into that corrupt partial and
            # lose this cell. Repair by prepending a newline so the partial line
            # stays isolated (it is skipped by read_cells_jsonl as invalid JSON).
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(fh.tell() - 1)
                if fh.read(1) != "\n":
                    fh.write("\n")
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def write_progress_sidecar(
    path: Path,
    *,
    tier: str,
    completed: int,
    total: int,
    started_utc: str,
    last_cell_key: str | None,
    eta_seconds: float | None,
    status: str = "in_progress",
) -> None:
    """Write the heartbeat/progress sidecar (atomic replace).

    An external check can read this single small file to see %-complete + ETA
    without parsing the whole JSONL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "scorer_spectral_atlas_progress.v1",
        "tier": tier,
        "status": status,
        "completed_cells": int(completed),
        "total_cells": int(total),
        "fraction_complete": (float(completed) / total) if total else 0.0,
        "started_utc": started_utc,
        "updated_utc": _utc_iso(),
        "last_cell_key": last_cell_key,
        "eta_seconds": eta_seconds,
        "authority_tier": "exact_cpu_advisory",
        "promotable": False,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_done_marker(path: Path, *, exit_code: int, completed: int, total: int) -> None:
    """Write the marker-on-exit (exit code + completed-cell count)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"EXIT={int(exit_code)}\n"
        f"COMPLETED_CELLS={int(completed)}\n"
        f"TOTAL_CELLS={int(total)}\n"
        f"FRACTION_COMPLETE={(float(completed) / total) if total else 0.0:.6f}\n"
        f"UTC={_utc_iso()}\n"
    )


def _utc_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# The resumable run loop.
# ---------------------------------------------------------------------------


def run_resumable_atlas(
    source_pairs: _np.ndarray,
    grid: v2.AtlasGrid,
    paths: AtlasRunPaths,
    *,
    tier: str,
    device: str = "cpu",
    progress_every: int = 1,
    write_final_atlas: bool = True,
    scorer: v2.FrozenScorer | None = None,
    baseline: dict[str, float] | None = None,
    boundary_margin_thresh: float | None = None,
) -> dict[str, Any]:
    """Run (or RESUME) the atlas, streaming each cell to the durable JSONL.

    On startup, reads ``paths.cells_jsonl`` to build the completed-cell-key set
    and SKIPS those cells. Each newly measured cell is appended to the JSONL (so
    a kill loses at most the in-flight cell), the progress sidecar is refreshed,
    and at the end the final atlas is re-aggregated from the FULL JSONL (the
    skipped + the new cells) and written to ``paths.atlas_json``.

    Returns the aggregated atlas dict (over all cells in the JSONL).
    """
    import numpy as np

    pairs = np.asarray(source_pairs)
    if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[-1] != 3:
        raise ValueError(
            f"source_pairs must be (N, 2, H, W, 3); got shape {pairs.shape}"
        )
    n_pairs = min(int(grid.n_pairs), pairs.shape[0])
    pairs = pairs[:n_pairs]

    paths.work_dir.mkdir(parents=True, exist_ok=True)

    # Resume: which cells are already done?
    completed_keys, completed_cells = load_completed_cells(paths.cells_jsonl)
    total = grid.total_cells()
    started_utc = _utc_iso()
    print(
        f"[atlas.resume] tier={tier} total_cells={total} "
        f"already_completed={len(completed_keys)} remaining={total - len(completed_keys)} "
        f"device={device} work_dir={paths.work_dir}",
        flush=True,
    )

    # Load the frozen scorer + measure the (deterministic) baseline ONCE.
    if scorer is None:
        scorer = v2.FrozenScorer(device=device)
    if baseline is None or boundary_margin_thresh is None:
        baseline, boundary_thresh = v2._measure_baseline_and_threshold(scorer, pairs)
    else:
        boundary_thresh = boundary_margin_thresh

    done = len(completed_keys)
    t0 = time.monotonic()
    new_this_run = 0
    last_key: str | None = None

    write_progress_sidecar(
        paths.progress_json,
        tier=tier,
        completed=done,
        total=total,
        started_utc=started_utc,
        last_cell_key=None,
        eta_seconds=None,
        status="in_progress",
    )

    for cell, key_str in v2.iter_atlas_cells(
        pairs,
        grid,
        device=device,
        skip_cell_keys=completed_keys,
        scorer=scorer,
        baseline=baseline,
        boundary_margin_thresh=boundary_thresh,
    ):
        done += 1
        new_this_run += 1  # noqa: SIM113 - a distinct counter from `done` (resume-aware)
        last_key = key_str
        append_cell_jsonl(
            paths.cells_jsonl, key=key_str, cell=cell, cell_index=done
        )
        elapsed = time.monotonic() - t0
        eta = None
        if new_this_run > 0:
            per_cell = elapsed / new_this_run
            eta = per_cell * (total - done)
        if progress_every > 0 and (done % progress_every == 0 or done == total):
            write_progress_sidecar(
                paths.progress_json,
                tier=tier,
                completed=done,
                total=total,
                started_utc=started_utc,
                last_cell_key=last_key,
                eta_seconds=eta,
                status="in_progress",
            )
            print(
                f"[atlas.resume] cell {done}/{total} "
                f"{key_str} H_seg={cell['H_seg']:+.5f} H_pose={cell['H_pose']:+.4f} "
                f"(new_this_run={new_this_run}, eta~{(eta or 0)/3600:.2f}h)",
                flush=True,
            )

    # Re-aggregate the FINAL atlas from the FULL JSONL (skipped + new cells).
    _all_keys, all_cells = load_completed_cells(paths.cells_jsonl)
    atlas = v2.aggregate_atlas_from_cells(
        all_cells,
        grid,
        baseline=baseline,
        boundary_margin_threshold=boundary_thresh,
        n_pairs=n_pairs,
    )
    atlas["tier"] = tier
    atlas["utc"] = _utc_iso()
    atlas["resumed_from_jsonl"] = str(paths.cells_jsonl)
    atlas["cells_jsonl"] = str(paths.cells_jsonl)
    atlas["lowering_opportunities"] = analyze_lowering_opportunities(all_cells)

    if write_final_atlas:
        paths.atlas_json.parent.mkdir(parents=True, exist_ok=True)
        paths.atlas_json.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n")

    write_progress_sidecar(
        paths.progress_json,
        tier=tier,
        completed=done,
        total=total,
        started_utc=started_utc,
        last_cell_key=last_key,
        eta_seconds=0.0,
        status="complete",
    )
    return atlas


# ---------------------------------------------------------------------------
# Lowering-opportunity analysis (the GOAL_v3 invisible-subspace rate lever).
#
# The atlas is not only the carrier's FREQUENCY BUDGET (where to SPEND bytes —
# where the scorer reacts). Its complement is the SCORE-LOWERING lever: the cells
# where the scorer barely reacts (H ~ 0) are where the carrier can SHED bytes
# INVISIBLY (spend fewer bits where the scorer is blind). This is the
# GOAL_v3 80.67%-invisible-subspace / null-space rate-shedding direction. The
# consumer is the bit-allocator / waterfiller: spend bits where H is high, shed
# where H ~ 0.
# ---------------------------------------------------------------------------


def analyze_lowering_opportunities(
    cells: list[dict[str, Any]],
    *,
    top_k: int = 12,
    low_sensitivity_quantile: float = 0.25,
) -> dict[str, Any]:
    """Mine the atlas cells for (a) the freq-budget ranking + (b) shed-bytes cells.

    (a) **Per-axis sensitivity ranking** — for each axis (band, orientation,
        channel, frame-incidence) the axis VALUE with the highest mean combined
        sensitivity, where combined sensitivity is the contest-weighted response
        ``100*|H_seg| + sqrt(10*max(H_pose,0))`` (the score units the scorer
        actually moves). This is WHERE THE CARRIER SHOULD SPEND BYTES.

    (b) **Low-sensitivity (shed-bytes) cells** — the cells in the bottom
        ``low_sensitivity_quantile`` of combined sensitivity AND below the seg/pose
        response medians: where the scorer is effectively BLIND, so the carrier
        can shed bytes invisibly. This is the SCORE-LOWERING opportunity for the
        bit-allocator / waterfiller.

    Pure-python; no torch. Returns a JSON-serializable dict.
    """
    if not cells:
        return {
            "note": "no cells measured yet",
            "consumer": "bit_allocator_waterfiller",
            "spend_here_freq_budget": {},
            "shed_here_low_sensitivity_cells": [],
        }

    def _combined(cell: dict[str, Any]) -> float:
        h_seg = abs(float(cell.get("H_seg", 0.0)))
        h_pose = float(cell.get("H_pose", 0.0))
        return 100.0 * h_seg + (10.0 * max(0.0, h_pose)) ** 0.5

    import math as _math

    scored = [(c, _combined(c)) for c in cells]
    vals = sorted(s for _c, s in scored)
    n = len(vals)

    def _quantile(q: float) -> float:
        if n == 1:
            return vals[0]
        # ceil rounding so the bottom-q FRACTION is captured inclusively (a 4-cell
        # bottom-quartile = >=1 cell; a 200-cell one = ~50 cells). The blind-floor
        # below additionally captures any cell the scorer does not react to at all.
        idx = min(n - 1, max(0, _math.ceil(q * (n - 1))))
        return vals[idx]

    # (a) per-axis spend ranking — combined sensitivity grouped by axis value.
    axes = ("band_index", "orientation", "channel_basis", "channel", "frame_incidence")
    spend_ranking: dict[str, Any] = {}
    for axis in axes:
        groups: dict[str, list[float]] = {}
        for cell, s in scored:
            key = str(cell.get(axis))
            groups.setdefault(key, []).append(s)
        ranked = sorted(
            (
                {
                    "value": k,
                    "mean_combined_sensitivity": sum(v) / len(v),
                    "max_combined_sensitivity": max(v),
                    "n_cells": len(v),
                }
                for k, v in groups.items()
            ),
            key=lambda d: d["mean_combined_sensitivity"],
            reverse=True,
        )
        spend_ranking[axis] = ranked

    # Per-band freq-budget with the SIREN-w-equivalent (the carrier's omega).
    band_freq_budget = []
    seen_bands: dict[int, dict[str, Any]] = {}
    for cell, s in scored:
        bi = int(cell.get("band_index", -1))
        coords = cell.get("frequency_coordinates", {}) or {}
        cur = seen_bands.get(bi)
        if cur is None or s > cur["max_combined_sensitivity"]:
            seen_bands[bi] = {
                "band_index": bi,
                "r_center": cell.get("r_center"),
                "siren_w_equivalent": coords.get("siren_w_equivalent"),
                "scorer_cycles_per_pixel": coords.get("scorer_cycles_per_pixel"),
                "aliases_at_scorer": coords.get("aliases_at_scorer"),
                "max_combined_sensitivity": s,
            }
    band_freq_budget = sorted(
        seen_bands.values(), key=lambda d: d["max_combined_sensitivity"], reverse=True
    )

    # (b) low-sensitivity shed-bytes cells (the score-lowering lever).
    # A cell is a shed candidate if it is in the bottom-quantile combined
    # sensitivity AND below both response medians, OR it is below a near-zero
    # "blind floor" (the scorer does not react to it at all). The blind floor is
    # in CONTEST SCORE UNITS: 1e-4 ~ a d_seg of 1e-6 / a sqrt(10*d_pose) of 1e-4,
    # i.e. movements far below contest reporting precision.
    thresh = _quantile(low_sensitivity_quantile)
    blind_floor = 1e-4
    seg_med = _median([abs(float(c.get("H_seg", 0.0))) for c, _s in scored])
    pose_med = _median([max(0.0, float(c.get("H_pose", 0.0))) for c, _s in scored])

    def _is_shed(cell: dict[str, Any], s: float) -> bool:
        if s <= blind_floor:
            return True  # truly blind: scorer does not react -> always sheddable
        return (
            s <= thresh
            and abs(float(cell.get("H_seg", 0.0))) <= seg_med
            and max(0.0, float(cell.get("H_pose", 0.0))) <= pose_med
        )

    shed_cells = [
        {
            "cell_key": v2.cell_key_str(
                v2.cell_key(
                    band_index=cell["band_index"],
                    orientation=cell["orientation"],
                    amplitude_lsb=cell["amplitude_lsb"],
                    channel_basis=cell["channel_basis"],
                    channel=cell["channel"],
                    frame_incidence=cell["frame_incidence"],
                )
            ),
            "band_index": cell["band_index"],
            "orientation": cell["orientation"],
            "amplitude_lsb": cell["amplitude_lsb"],
            "channel_basis": cell["channel_basis"],
            "channel": cell["channel"],
            "frame_incidence": cell["frame_incidence"],
            "H_seg": cell.get("H_seg"),
            "H_pose": cell.get("H_pose"),
            "combined_sensitivity": s,
            "siren_w_equivalent": (cell.get("frequency_coordinates") or {}).get(
                "siren_w_equivalent"
            ),
        }
        for cell, s in scored
        if _is_shed(cell, s)
    ]
    shed_cells.sort(key=lambda d: d["combined_sensitivity"])

    return {
        "note": (
            "spend_here = the carrier's frequency budget (axis values where the "
            "scorer reacts most, in contest score units 100*|H_seg|+sqrt(10*H_pose)). "
            "shed_here = the SCORE-LOWERING lever: cells where the scorer is "
            "effectively blind (combined sensitivity below a near-zero blind floor, "
            "OR bottom-quantile combined sensitivity AND below the seg+pose response "
            "medians) -> the carrier can spend FEWER bytes there invisibly. "
            "Consumer = the bit-allocator / waterfiller."
        ),
        "consumer": "bit_allocator_waterfiller",
        "authority_tier": "exact_cpu_advisory",
        "promotable": False,
        "blind_floor_combined_sensitivity": blind_floor,
        "n_cells_analyzed": n,
        "low_sensitivity_quantile": low_sensitivity_quantile,
        "low_sensitivity_threshold_combined": thresh,
        "seg_response_median": seg_med,
        "pose_response_median": pose_med,
        "spend_here_freq_budget": {
            "per_axis_ranking": spend_ranking,
            "per_band_with_siren_w": band_freq_budget,
        },
        "shed_here_low_sensitivity_cells": shed_cells[:top_k],
        "n_shed_candidate_cells": len(shed_cells),
    }


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])
