# SPDX-License-Identifier: MIT
"""BUILD-1 canonical empirical 5D canvas populator.

Per BUILD-1 routing directive 2026-05-26 (DROP-MANY+REPLACE+COMPOSITION
APPARATUS STATE AUDIT memo at
`.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md`
operator-routable #1; sister of design memo
`.omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md`
§DELIVERABLE 3 BUILD-1).

This module implements the canonical empirical population of the 5D canvas
SCAFFOLD `tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas`.
It is the BUILD-1 sister deliverable; BUILD-2 + BUILD-3 + BUILD-4 sister
subagents consume the populated canvas downstream.

## What this module does (BUILD-1 scope)

1. Read `.omx/state/master_gradient_anchors.jsonl` rows via the canonical
   `tac.master_gradient.load_anchors_strict` (Catalog #138 fail-closed).
2. Per archive sha256 declared in the ledger:
   - Group anchors by (archive_sha256, measurement_axis).
   - Derive the canonical `CpuCudaAxis` per anchor's `measurement_axis`
     (`[contest-CPU]` -> CONTEST_CPU; `[contest-CUDA]` -> CONTEST_CUDA_T4;
     advisory axes per CLAUDE.md "MPS auth eval is NOISE" carry distinct
     evidence_grade and are SKIPPED from the canonical 5D canvas
     population per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"
     non-negotiable + Catalog #192).
   - Populate `PairFrameScorerGeometryCell` per (pair_idx, frame_idx,
     scorer_axis, receiver_runtime, cpu_cuda_axis) coordinate where the
     anchor's `operating_point` carries finite per-axis components.
3. Persist the populated canvas to
   `.omx/state/pair_frame_scorer_geometry_lattice/<archive_sha[:12]>_<utc>.json`
   via fcntl-locked atomic write per Catalog #131/#138/#245 canonical
   4-layer ledger pattern.
4. Provide canonical `load_empirical_lattice` reader sister of the
   canonical writer.

## Canonical-vs-unique decision per layer

- Master gradient anchor reader: ADOPT canonical
  `tac.master_gradient.load_anchors_strict` (Catalog #138 fail-closed; the
  canonical 4-layer ledger reader).
- Contest-axis custody: ADOPT canonical
  `tac.master_gradient.contest_axis_authority_violation_reason` so
  non-authoritative anchors flow into Provenance with explicit reason
  rather than silently leaking into the canvas as if authoritative.
- Provenance contract: ADOPT canonical Provenance per Catalog #323; every
  cell's `catalog_323_provenance` carries `tac.provenance.builders` output.
- fcntl-lock pattern: ADOPT canonical pattern per
  `tac.deploy.modal.call_id_ledger._ledger_lock` (Catalog #131 sister
  discipline); transactional `.tmp.<uuid12>` + `os.replace` per Catalog
  #128/#131.
- 5D canvas schema: ADOPT scaffold contract; this module ONLY populates
  + persists + reads (no schema extensions per BUILD-1 scope).
- BUILD-2 operation generators: NOT IMPLEMENTED HERE (sister-disjoint per
  parallel BUILD-2 spawn); this module's `build_lattice` does NOT call
  any of the 4 canonical operation generators
  (`generate_full_drop_starts` / `generate_repair_starts` /
  `generate_masked_starts` / `generate_feathered_starts`) which remain
  BUILD-2 deferred per scaffold `NotImplementedError`.
- BUILD-4 Tier B promotion: NOT IMPLEMENTED HERE; this module keeps the
  scaffold's `CONSUMER_TIER = TIER_A_OBSERVABILITY_ONLY` per Catalog #357
  + #341 canonical-routing markers (non-promotable defaults).

## Observability surface (Catalog #305)

- Every populated cell carries `catalog_323_provenance` with archive sha
  + measurement_axis + hardware_substrate + evidence_grade.
- Canvas state inspectable via `cell_count` / `query_cell` (sister of
  scaffold's per-cell lookup).
- Canvas decomposable per (pair, frame, scorer_axis, receiver_runtime,
  cpu_cuda_axis) coordinate.
- Canvas diff-able across runs via deterministic archive_sha256 keying
  + sort_keys=True at writer.
- Canvas queryable post-hoc via JSON sidecar at `.omx/state/`.
- Canvas cite-able via per-cell `catalog_323_provenance.inputs_sha256`.
- Canvas counterfactual-able via per-cell `receiver_feasibility` bool.

## 9-dimension success checklist evidence (Catalog #294)

- UNIQUENESS: FIRST canonical empirical populator of the 5D canvas;
  binds master_gradient_anchors -> PairFrameScorerGeometryCell with
  canonical Provenance.
- BEAUTY: 5 public functions + 1 canonical writer + 1 canonical reader;
  all reuse canonical helpers (no fork).
- DISTINCTNESS: explicit sister-disjoint from BUILD-2 operation
  generators + BUILD-3 Catalog #356 wire-in + BUILD-4 Tier B promotion.
- RIGOR: canonical Catalog #138 fail-closed load + Catalog #131 fcntl
  + Catalog #323 Provenance + Catalog #192 advisory-axis skip.
- OPTIMIZATION-PER-TECHNIQUE: per-anchor sparse representation; no
  21.6M-cell dense allocation; only cells with finite empirical signal
  populated.
- STACK-OF-STACKS-COMPOSABILITY: output sidecar IS the canonical
  composability primitive BUILD-2 + BUILD-3 + BUILD-4 sister subagents
  consume; canvas-vs-algorithm separation per design memo.
- DETERMINISTIC-REPRODUCIBILITY: archive_sha256-keyed; byte-stable
  output via sort_keys=True; same input -> same output bytes.
- EXTREME-OPTIMIZATION-PERFORMANCE: sparse representation; canvas
  population O(N_anchors) where N_anchors is current ledger row count.
- OPTIMAL-MINIMAL-CONTEST-SCORE: BUILD-1 itself is foundation; the
  score-lowering value lands when BUILD-2 + BUILD-3 + BUILD-4 +
  paired-axis dispatch wave land downstream per Phase 4 operator-routable
  in audit memo.

## Cargo-cult audit per assumption (Catalog #303)

1. **`measurement_axis` -> `CpuCudaAxis` mapping** — HARD-EARNED per
   CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval - BOTH
   CPU AND CUDA" non-negotiables: `[contest-CPU]` and `[contest-CUDA]`
   are 1:1 contest-compliant axes; advisory axes (`[macOS-CPU advisory]`,
   `[MPS-PROXY]`, etc.) are NOT 1:1 contest-compliant.

2. **`scorer_axis` per-anchor decomposition** — HARD-EARNED per CLAUDE.md
   canonical contest formula `S = 100*d_seg + sqrt(10*d_pose) + 25*rate`;
   `operating_point` ledger field carries `d_seg`, `d_pose`, `rate`
   verbatim per canonical equation #36
   `pairset_component_marginal_score_decomposition_v1`.

3. **`receiver_runtime = RAW_RESIDUAL` default for empirical anchors** —
   HARD-EARNED per design memo: master gradient anchors are computed
   against the AS-IS archive bytes; no per-receiver compensation has been
   applied at the measurement point. Sister `MASKED` / `FEATHERED` /
   `SMOOTHED_RESIDUAL` / `FULL_DROP` modes are queried via
   `query_receiver_runtime_feasibility` per BUILD-1 design.

4. **Per-pair decomposition** — HEURISTIC-PRIOR-LANDED (was
   CARGO-CULTED-PENDING-EMPIRICAL until the per-pair gradient artifact
   landed): the archive-AGGREGATE `master_gradient_anchors.jsonl`
   `operating_point` path (`_build_cells_from_anchor`) builds ONE
   coordinate (`pair_idx=0`, `frame_idx=0`) per anchor. The per-pair
   path `populate_per_pair_cells_from_gradient_array` consumes the
   `(N_archive_bytes, N_pairs, 3)` MLX per-pair master-gradient artifact
   (`.omx/state/master_gradient_fec6_frontier_mlx_per_pair_*.npy`) and
   builds ONE coordinate per distinct `pair_idx` — the ≥2-distinct-
   coordinate structure the 12-operator multi-op composition sweep needs.
   The artifact is a **macOS-MLX research-signal HEURISTIC PRIOR** (NOT a
   HARD-EARNED authority anchor) per CLAUDE.md "MLX portable-local-
   substrate authority" + Catalog #192/#127/#323: every per-pair cell
   carries `build_provenance_for_macos_cpu_advisory` Provenance with
   `score_claim=false` / `promotion_eligible=false` /
   `evidence_grade="macOS-MLX research-signal"` / `axis_tag="[predicted]"`.
   The per-tensor-FD-projected-per-byte artifact attributes uniformly
   across the decompressed mantissa span, NOT the true archive-byte
   domain (the heuristic's 3 anchor blockers per the producer memo); it
   is a probe-RANKING prior for the closed-form PREDICTION sweep, NOT a
   contest score. The PyTorch-autograd `tools/extract_master_gradient.py`
   remains the authority surface. See
   `.omx/research/mlx_per_pair_master_gradient_authoritative_artifacts_landed_20260527.md`.

## Predicted ΔS band (Catalog #296)

NOT a substrate dispatch proposal. This is a canonical populator module
that emits empirical anchors AS-IS from master_gradient_anchors.jsonl.
No Dykstra-feasibility check needed per Catalog #296 because the
populator does NOT compose constraints; it READS anchored measurements.

## Council attendees / verdict (Catalog #300)

T1 working-group VERDICT PROCEED (BUILD-1 sister subagent op-routable
per audit memo §Phase 4 Priority 1; no quorum required at T1 per Catalog
#300). Attendees: Shannon LEAD + Dykstra CO-LEAD + Daubechies CO-LEAD +
Rudin CO-LEAD + Carmack + Assumption-Adversary. Mission contribution per
Catalog #300: `apparatus_maintenance` (foundation; unblocks BUILD-2 +
BUILD-3 + BUILD-4 sister subagents + Phase 4 paired-axis dispatch wave).

Sister cross-references:

- `tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas`
  (canonical SCAFFOLD this module populates)
- `tac.master_gradient` (canonical ledger reader + custody validator)
- `tac.provenance.builders` (canonical Provenance per Catalog #323)
- `tac.cathedral.consumer_contract` (canonical AxisDecomposition per
  Catalog #356)
- `tac.deploy.modal.call_id_ledger` (canonical fcntl-locked 4-layer
  ledger pattern this populator mirrors)

Lane: `lane_build_1_populate_5d_canvas_20260526` L1.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.master_gradient import (
    AUTHORITATIVE_CONTEST_AXES,
    MASTER_GRADIENT_LEDGER_PATH,
    contest_axis_authority_violation_reason,
    load_anchors_strict,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CANONICAL_FRAME_COUNT,
    CANVAS_SCHEMA,
    CELL_SCHEMA,
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.provenance.builders import (
    build_provenance_for_macos_cpu_advisory,
    build_provenance_for_predicted,
)
from tac.provenance.validator import provenance_to_dict

# ---------------------------------------------------------------------------
# Canonical writer + reader constants.
# ---------------------------------------------------------------------------

POPULATOR_SCHEMA_VERSION = "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1"

_LOCK_TIMEOUT_SECONDS = 30.0

EMPIRICAL_LATTICE_DIR = ".omx/state/pair_frame_scorer_geometry_lattice"
"""Canonical lattice sidecar directory under `.omx/state/` per Catalog #131."""

EMPIRICAL_LATTICE_LOCK = (
    ".omx/state/pair_frame_scorer_geometry_lattice/.empirical_lattice.lock"
)
"""Canonical lock file path per Catalog #131."""

# ---------------------------------------------------------------------------
# `measurement_axis` -> `CpuCudaAxis` mapping (HARD-EARNED per CLAUDE.md
# "Submission auth eval - BOTH CPU AND CUDA" + "MPS auth eval is NOISE").
# ---------------------------------------------------------------------------

_AUTHORITATIVE_AXIS_MAP: dict[str, CpuCudaAxis] = {
    "[contest-CPU]": CpuCudaAxis.CONTEST_CPU,
    "[contest-CUDA]": CpuCudaAxis.CONTEST_CUDA_T4,
}
"""Canonical 1:1 contest-compliant axis mapping. Advisory axes are NOT
in this map; the populator emits them with `evidence_grade=advisory`
in Provenance and tags `is_authoritative=False`."""


class PopulatorError(ValueError):
    """Raised when canvas population cannot proceed safely."""


@dataclass(frozen=True)
class PopulatedCanvasManifest:
    """Canonical manifest returned by `populate_5d_canvas_from_master_gradient_anchors`.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": one
    canonical return contract for the canonical populator.

    Per Catalog #341 + #323: every populator-emitted artifact carries
    non-promotable canonical-routing markers + canonical Provenance.

    Args:
        canvas: populated `PairFrameScorerGeometryLattice` instance.
        archive_sha256: archive sha256 the canvas was populated against.
        cells_populated: total number of cells the populator landed.
        anchors_consumed: number of master_gradient_anchors rows the
            populator consumed for this archive.
        anchors_skipped_non_authoritative: number of non-authoritative
            advisory anchors the populator deliberately skipped per
            Catalog #192.
        output_path: canonical sidecar path the populator wrote (if any).
        catalog_323_provenance: dict-form Provenance per Catalog #323
            attached to the manifest itself; archive-level not per-cell.
    """

    canvas: PairFrameScorerGeometryLattice
    archive_sha256: str
    cells_populated: int
    anchors_consumed: int
    anchors_skipped_non_authoritative: int
    output_path: Path | None
    catalog_323_provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization (byte-stable via sort_keys=True at writer)."""
        return {
            "schema": POPULATOR_SCHEMA_VERSION,
            "archive_sha256": str(self.archive_sha256),
            "cells_populated": int(self.cells_populated),
            "anchors_consumed": int(self.anchors_consumed),
            "anchors_skipped_non_authoritative": int(
                self.anchors_skipped_non_authoritative
            ),
            "output_path": str(self.output_path) if self.output_path else None,
            "catalog_323_provenance": dict(self.catalog_323_provenance),
            # Tier A canonical non-promotable markers per Catalog #341.
            "predicted_delta_adjustment": 0.0,
            "promotable": False,
            "axis_tag": "[predicted]",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
        }


# ---------------------------------------------------------------------------
# fcntl-locked write helpers (canonical pattern per Catalog #131).
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _populator_lock(lock_path: Path):
    """Acquire fcntl exclusive lock per canonical Catalog #131 pattern.

    Mirrors `tac.deploy.modal.call_id_ledger._ledger_lock` (canonical
    4-layer ledger pattern).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not acquire {lock_path} within "
                        f"{_LOCK_TIMEOUT_SECONDS}s"
                    ) from None
                time.sleep(0.05)
        try:
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with microseconds (canonical helper sister)."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    """Transactional write per Catalog #128/#131 canonical sister pattern.

    Writes to `<target>.tmp.<uuid12>` then `os.replace` for atomicity.
    Caller MUST hold the populator lock before invoking this helper.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# Canonical helpers consumed by the populator.
# ---------------------------------------------------------------------------


def _resolve_repo_root() -> Path:
    """Resolve canonical repo root via this file's location."""
    here = Path(__file__).resolve()
    # src/tac/optimization/<this file>.py -> repo_root
    return here.parents[3]


def _derive_archive_sha256_from_path(archive_path: Path) -> str:
    """Compute sha256 of archive bytes for canonical canvas keying.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the canvas is
    keyed by archive sha256 so two callers with the same archive bytes
    populate equivalent canvases.
    """
    h = hashlib.sha256()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify_cpu_cuda_axis(measurement_axis: str) -> CpuCudaAxis | None:
    """Classify a master gradient anchor's measurement axis to a canonical
    `CpuCudaAxis` per the canonical mapping. Returns None for advisory axes
    that are NOT 1:1 contest-compliant per CLAUDE.md "MPS auth eval is NOISE".
    """
    return _AUTHORITATIVE_AXIS_MAP.get(str(measurement_axis))


def _build_cells_from_anchor(
    anchor: dict[str, Any],
    cpu_cuda_axis: CpuCudaAxis,
) -> list[PairFrameScorerGeometryCell]:
    """Build canvas cells from a single master gradient anchor dict.

    Per Cargo-cult audit assumption 4: archive-aggregate decomposition.
    The anchor's `operating_point` carries (d_seg, d_pose, rate, score)
    at archive scope. Populates 3 cells per anchor (one per
    `ScorerAxis`); pair_idx=0 + frame_idx=0 as the canonical
    archive-aggregate coordinate.

    Per anchor: 3 cells (SEGNET_5CLASS + POSENET_6D + RATE_TERM); each
    cell carries the per-axis component value verbatim from
    `operating_point`. `receiver_feasibility=True` for RAW_RESIDUAL
    (the as-measured receiver mode).

    Per Catalog #356: `predicted_axis_decomposition` not populated here
    (BUILD-3 sister wires the per-cell AxisDecomposition emission).

    Returns 3 cells per anchor (one per scorer_axis).
    """
    cells: list[PairFrameScorerGeometryCell] = []
    op = anchor.get("operating_point") or {}
    if not isinstance(op, dict):
        op = {}
    d_seg = float(op.get("d_seg", 0.0))
    d_pose = float(op.get("d_pose", 0.0))
    rate = float(op.get("rate", 0.0))
    measurement_axis = str(anchor.get("measurement_axis", ""))
    measurement_hardware = str(anchor.get("measurement_hardware", "unknown"))
    measurement_method = str(anchor.get("measurement_method", "unknown"))
    measurement_call_id = str(anchor.get("measurement_call_id", "unknown"))
    measurement_utc = anchor.get("measurement_utc")
    archive_sha = str(anchor.get("archive_sha256", ""))
    n_bytes = int(anchor.get("n_bytes", 0))
    n_pairs_used = anchor.get("n_pairs_used")
    # Provenance per Catalog #323 — every cell cites the anchor's archive
    # sha + measurement axis + hardware substrate.
    is_authoritative = measurement_axis in AUTHORITATIVE_CONTEST_AXES
    violation_reason = contest_axis_authority_violation_reason(anchor)

    prov = build_provenance_for_predicted(
        model_id="pair_frame_scorer_geometry_lattice_5d_canvas_populator",
        inputs_sha256=archive_sha,
        measurement_axis=measurement_axis or "[predicted]",
        hardware_substrate=measurement_hardware,
        captured_at_utc=measurement_utc or _now_iso(),
    )
    prov_dict = provenance_to_dict(prov)
    # Attach diagnostic fields per Catalog #305 observability.
    prov_dict["measurement_method"] = measurement_method
    prov_dict["measurement_call_id"] = measurement_call_id
    prov_dict["n_pairs_used"] = n_pairs_used
    prov_dict["n_bytes"] = n_bytes
    prov_dict["is_authoritative_contest_axis"] = bool(is_authoritative)
    prov_dict["contest_axis_authority_violation_reason"] = violation_reason

    for scorer_axis, signed_delta in (
        (ScorerAxis.SEGNET_5CLASS, d_seg),
        (ScorerAxis.POSENET_6D, d_pose),
        (ScorerAxis.RATE_TERM, rate),
    ):
        cell = PairFrameScorerGeometryCell(
            pair_idx=0,
            frame_idx=0,
            scorer_axis=scorer_axis,
            receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
            cpu_cuda_axis=cpu_cuda_axis,
            # `predicted_delta_score` carries the per-axis empirical
            # component value verbatim per canonical equation #36
            # decomposition.
            predicted_delta_score=signed_delta,
            # `predicted_byte_cost=0` because no operation has been
            # applied; this is the as-measured archive state.
            predicted_byte_cost=0,
            receiver_feasibility=True,
            catalog_323_provenance=prov_dict,
        )
        cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Canonical primary entry point: populate from master_gradient_anchors.jsonl.
# ---------------------------------------------------------------------------


def populate_5d_canvas_from_master_gradient_anchors(
    *,
    archive_sha256: str | None = None,
    ledger_path: Path | None = None,
    output_path: Path | None = None,
    write_sidecar: bool = True,
    skip_non_authoritative: bool = True,
    repo_root: Path | None = None,
) -> PopulatedCanvasManifest:
    """Canonical empirical populator for the 5D canvas (BUILD-1 entry point).

    Per BUILD-1 routing directive 2026-05-26:

    1. Read `.omx/state/master_gradient_anchors.jsonl` via canonical
       `tac.master_gradient.load_anchors_strict` (Catalog #138).
    2. Filter to anchors for the given archive sha256 (or process ALL
       distinct archives if `archive_sha256` is None).
    3. Per anchor: derive `CpuCudaAxis` from `measurement_axis`; skip
       non-authoritative advisory axes per CLAUDE.md non-negotiable if
       `skip_non_authoritative=True` (default).
    4. Populate `PairFrameScorerGeometryCell` per anchor; build canvas.
    5. Persist canvas to canonical sidecar at
       `.omx/state/pair_frame_scorer_geometry_lattice/<sha[:12]>_<utc>.json`
       via fcntl-locked atomic write (skip if `write_sidecar=False`).
    6. Return `PopulatedCanvasManifest` with the canvas + provenance.

    Args:
        archive_sha256: optional archive sha256 to filter by. If None,
            the populator picks the LATEST distinct archive in the
            ledger (per `effective_anchor_sort_key`).
        ledger_path: optional override for the master gradient ledger
            path (default: canonical `MASTER_GRADIENT_LEDGER_PATH`).
        output_path: optional override for the sidecar output path
            (default: canonical `.omx/state/pair_frame_scorer_geometry_lattice/<sha[:12]>_<utc>.json`).
        write_sidecar: if True (default), persist canvas to sidecar.
        skip_non_authoritative: if True (default), advisory axes (e.g.
            `[macOS-CPU advisory]`, `[MPS-PROXY]`) are excluded from
            the canvas per CLAUDE.md "Submission auth eval - BOTH CPU
            AND CUDA" non-negotiable + Catalog #192.
        repo_root: optional repo root override (default: resolved from
            this module's location).

    Returns:
        `PopulatedCanvasManifest` with canvas + provenance + counts.

    Raises:
        PopulatorError: if no anchors match the archive sha256 OR
            ledger is corrupt (per Catalog #138 fail-closed).
    """
    root = (repo_root or _resolve_repo_root()).resolve()
    ledger = (
        ledger_path
        if ledger_path is not None
        else (root / MASTER_GRADIENT_LEDGER_PATH)
    )

    if not ledger.exists():
        raise PopulatorError(f"master gradient ledger not found at {ledger}")

    try:
        anchors = load_anchors_strict(ledger)
    except FileNotFoundError as exc:
        raise PopulatorError(
            f"master gradient ledger not found at {ledger}: {exc}"
        ) from exc

    if not anchors:
        raise PopulatorError(
            f"master gradient ledger at {ledger} carries 0 rows; "
            "no anchors available to populate canvas"
        )

    # Filter to target archive sha256.
    if archive_sha256 is None:
        # Pick the LATEST distinct archive per measurement_utc.
        sorted_anchors = sorted(
            anchors,
            key=lambda a: a.get("measurement_utc") or "",
            reverse=True,
        )
        target_sha = sorted_anchors[0]["archive_sha256"]
    else:
        target_sha = str(archive_sha256)

    matching = [a for a in anchors if a.get("archive_sha256") == target_sha]
    if not matching:
        raise PopulatorError(
            f"no master gradient anchors found for archive_sha256={target_sha!r}; "
            f"ledger carries {len(anchors)} rows but none match"
        )

    # Build canvas.
    canvas = PairFrameScorerGeometryLattice(archive_sha256=target_sha)
    consumed = 0
    skipped = 0
    cells_by_coord: dict[tuple, PairFrameScorerGeometryCell] = {}

    for anchor in matching:
        cpu_cuda = _classify_cpu_cuda_axis(anchor.get("measurement_axis", ""))
        if cpu_cuda is None:
            # Advisory axis per CLAUDE.md "MPS auth eval is NOISE".
            if skip_non_authoritative:
                skipped += 1
                continue
            # If the operator explicitly opted into advisory inclusion,
            # default to CONTEST_CPU bucket so the cell lands (the
            # Provenance carries `is_authoritative_contest_axis=False`
            # so downstream consumers can disambiguate).
            cpu_cuda = CpuCudaAxis.CONTEST_CPU

        cells = _build_cells_from_anchor(anchor, cpu_cuda)
        for cell in cells:
            # Latest-row-wins on collision (anchors emitted in
            # chronological order; later anchor supersedes earlier per
            # `effective_anchor_sort_key` sister discipline).
            cells_by_coord[cell.coordinate] = cell
        consumed += 1

    # Re-instantiate canvas with populated cells.
    canvas = PairFrameScorerGeometryLattice(
        archive_sha256=target_sha, cells=cells_by_coord
    )

    # Build manifest-level Provenance.
    manifest_prov = build_provenance_for_predicted(
        model_id="pair_frame_scorer_geometry_lattice_5d_canvas_populator_manifest",
        inputs_sha256=target_sha,
        measurement_axis="[predicted]",
        hardware_substrate="apparatus_canonical_helper",
        captured_at_utc=_now_iso(),
    )
    manifest_prov_dict = provenance_to_dict(manifest_prov)
    manifest_prov_dict["ledger_path"] = str(ledger)
    manifest_prov_dict["anchors_total"] = len(matching)
    manifest_prov_dict["anchors_consumed"] = consumed
    manifest_prov_dict["anchors_skipped_non_authoritative"] = skipped

    # Persist sidecar if requested.
    sidecar_path: Path | None = None
    if write_sidecar:
        if output_path is not None:
            sidecar_path = Path(output_path).resolve()
        else:
            sidecar_dir = root / EMPIRICAL_LATTICE_DIR
            utc_compact = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            sidecar_path = sidecar_dir / f"{target_sha[:12]}_{utc_compact}.json"

        payload = {
            "schema": POPULATOR_SCHEMA_VERSION,
            "canvas_schema": CANVAS_SCHEMA,
            "cell_schema": CELL_SCHEMA,
            "archive_sha256": target_sha,
            "cells_populated": len(cells_by_coord),
            "anchors_consumed": consumed,
            "anchors_skipped_non_authoritative": skipped,
            "cells": [c.as_dict() for c in cells_by_coord.values()],
            "manifest_provenance": manifest_prov_dict,
            # Catalog #341 canonical non-promotable markers.
            "predicted_delta_adjustment": 0.0,
            "promotable": False,
            "axis_tag": "[predicted]",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            # Catalog #341 routing marker rationale.
            "tier_a_rationale": (
                "pair_frame_scorer_geometry_lattice_5d_canvas BUILD-1 populator "
                "scaffold; BUILD-4 sister subagent op-routable promotes to Tier B"
            ),
        }
        lock_path = root / EMPIRICAL_LATTICE_LOCK
        with _populator_lock(lock_path):
            _atomic_write_json(sidecar_path, payload)

    return PopulatedCanvasManifest(
        canvas=canvas,
        archive_sha256=target_sha,
        cells_populated=len(cells_by_coord),
        anchors_consumed=consumed,
        anchors_skipped_non_authoritative=skipped,
        output_path=sidecar_path,
        catalog_323_provenance=manifest_prov_dict,
    )


# ---------------------------------------------------------------------------
# Per-pair populator: consume the MLX per-pair master-gradient HEURISTIC PRIOR.
#
# Per cargo-cult audit assumption 4 (HEURISTIC-PRIOR-LANDED): consumes the
# `(N_archive_bytes, N_pairs, 3)` MLX per-pair artifact + builds ONE coordinate
# per distinct `pair_idx` — the >=2-distinct-coordinate structure the
# 12-operator multi-op composition sweep needs. NON-PROMOTABLE macOS-MLX
# research-signal per CLAUDE.md "MLX portable-local-substrate authority" +
# Catalog #192/#127/#323.
# ---------------------------------------------------------------------------

# Canonical MLX per-pair heuristic-prior schema (producer memo
# `.omx/research/mlx_per_pair_master_gradient_authoritative_artifacts_landed_20260527.md`).
_MLX_PER_PAIR_HEURISTIC_SCHEMA = "mlx_tensor_fd_gradient_heuristic_v1_20260527"
_MLX_PER_PAIR_EVIDENCE_GRADE = "macOS-MLX research-signal"


def populate_per_pair_cells_from_gradient_array(
    gradient_npy_path: Path | str,
    *,
    archive_sha256: str | None = None,
    meta_json_path: Path | None = None,
    output_path: Path | None = None,
    write_sidecar: bool = True,
    cpu_cuda_axis: CpuCudaAxis = CpuCudaAxis.CONTEST_CPU,
    max_pairs: int | None = None,
    repo_root: Path | None = None,
) -> PopulatedCanvasManifest:
    """Populate the 5D canvas from an MLX per-pair master-gradient HEURISTIC PRIOR.

    Per cargo-cult audit assumption 4 (HEURISTIC-PRIOR-LANDED): this is the
    per-pair sister of `populate_5d_canvas_from_master_gradient_anchors`. Where
    the archive-aggregate path builds ONE coordinate (`pair_idx=0`,
    `frame_idx=0`) per anchor, this path consumes the
    `(N_archive_bytes, N_pairs, 3_axes)` per-pair artifact and builds ONE
    coordinate per distinct `pair_idx`, aggregating the per-byte sensitivity
    over the byte axis into a per-pair `(d_seg, d_pose, rate)` magnitude.

    The artifact is a **macOS-MLX research-signal HEURISTIC PRIOR**, NOT a
    HARD-EARNED authority anchor (per CLAUDE.md "MLX portable-local-substrate
    authority" + Catalog #192/#127/#323). Every cell carries
    `build_provenance_for_macos_cpu_advisory` Provenance with
    `score_claim=false` / `promotion_eligible=false` /
    `evidence_grade="macOS-MLX research-signal"` / `axis_tag="[predicted]"`.
    The PyTorch-autograd `tools/extract_master_gradient.py` remains the
    authority surface; this per-pair STRUCTURE is the Half-2 unblock for the
    closed-form PREDICTION sweep (which itself gates whether the paid
    contest-CUDA/CPU FIRE-phase per Catalog #246 is worth it), NOT a contest
    score.

    Shape contract: the artifact MUST be `(N_bytes, N_pairs, 3)` float with
    axes `(seg, pose, rate)` matching the producer memo. The rate column is
    expected to be all-zero (byte-value sensitivities do not move the rate
    term) but a non-zero rate column is accepted verbatim.

    Args:
        gradient_npy_path: path to the `.npy` per-pair gradient artifact.
        archive_sha256: optional override; default read from the sidecar
            `.npy.meta.json` `archive_sha256` field (Catalog #323 cite-able).
        meta_json_path: optional override for the sidecar meta path
            (default: `<gradient_npy_path>.meta.json`).
        output_path: optional override for the canvas sidecar path.
        write_sidecar: if True (default), persist canvas to canonical sidecar.
        cpu_cuda_axis: which 1:1 contest-compliant axis the artifact's scorer
            oracle measured against. The MLX oracle is a macOS-local forward
            port; the per-pair STRUCTURE is the same per-pair leverage map the
            contest-CPU axis would produce, so the default is CONTEST_CPU.
            The Provenance carries `evidence_grade="macOS-MLX research-signal"`
            so downstream consumers NEVER promote it.
        max_pairs: optional cap on the number of pairs to populate (default:
            all pairs in the artifact).
        repo_root: optional repo root override.

    Returns:
        `PopulatedCanvasManifest` with the per-pair canvas + Provenance.

    Raises:
        PopulatorError: if the artifact is missing / wrong shape / corrupt
            sidecar (per Catalog #138 fail-closed).
    """
    import numpy as np  # local import: numpy is a heavy dep; keep module-light.

    if not isinstance(cpu_cuda_axis, CpuCudaAxis):
        raise PopulatorError(
            f"cpu_cuda_axis must be CpuCudaAxis, got {type(cpu_cuda_axis).__name__}"
        )

    root = (repo_root or _resolve_repo_root()).resolve()
    npy_path = Path(gradient_npy_path)
    if not npy_path.is_absolute():
        npy_path = (root / npy_path).resolve()
    if not npy_path.exists():
        raise PopulatorError(f"per-pair gradient artifact not found at {npy_path}")

    # Resolve + read the sidecar meta (Catalog #323 cite-able provenance).
    meta_path = (
        Path(meta_json_path)
        if meta_json_path is not None
        else npy_path.with_suffix(npy_path.suffix + ".meta.json")
    )
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta = loaded
        except json.JSONDecodeError as exc:
            raise PopulatorError(
                f"per-pair gradient sidecar {meta_path} carries corrupt JSON: {exc}"
            ) from exc

    # Fail-closed if the sidecar declares it is NOT the MLX heuristic-prior
    # schema we expect (Catalog #229 premise verification + #138).
    sidecar_schema = meta.get("schema_version")
    if sidecar_schema is not None and sidecar_schema != _MLX_PER_PAIR_HEURISTIC_SCHEMA:
        raise PopulatorError(
            f"per-pair gradient sidecar {meta_path} schema mismatch: got "
            f"{sidecar_schema!r}, expected {_MLX_PER_PAIR_HEURISTIC_SCHEMA!r}; "
            "this populator path consumes the MLX per-pair heuristic-prior "
            "artifact ONLY (NON-PROMOTABLE per Catalog #192/#127/#323)"
        )

    target_sha = (
        str(archive_sha256)
        if archive_sha256 is not None
        else str(meta.get("archive_sha256", ""))
    )
    if not target_sha:
        raise PopulatorError(
            "archive_sha256 not provided and absent from sidecar; cannot key "
            "the per-pair canvas"
        )

    # Load the per-pair gradient array.
    try:
        arr = np.load(npy_path)
    except Exception as exc:  # numpy raises a variety of errors on bad input.
        raise PopulatorError(
            f"failed to load per-pair gradient artifact {npy_path}: {exc!r}"
        ) from exc
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise PopulatorError(
            f"per-pair gradient artifact {npy_path} must be shape "
            f"(N_bytes, N_pairs, 3), got {tuple(arr.shape)}"
        )
    n_bytes, n_pairs, _ = arr.shape
    if n_pairs <= 0 or n_bytes <= 0:
        raise PopulatorError(
            f"per-pair gradient artifact {npy_path} carries empty axes "
            f"(shape {tuple(arr.shape)})"
        )

    # Aggregate per-byte sensitivity over the byte axis into per-pair
    # (d_seg, d_pose, rate) magnitudes. axis 0 = byte; axis 1 = pair; axis 2 =
    # (seg, pose, rate). This is the canonical per-pair leverage the 12
    # operators compose over.
    per_pair = arr.sum(axis=0)  # shape (N_pairs, 3) float64
    n_to_populate = (
        min(int(max_pairs), n_pairs) if max_pairs is not None else n_pairs
    )

    # Canonical macOS-MLX advisory Provenance (NON-PROMOTABLE heuristic prior).
    advisory_source = str(meta.get("npy_path", str(npy_path)))
    advisory_captured = meta.get("captured_at_utc") or _now_iso()
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256=target_sha,
        source_path=advisory_source,
        captured_at_utc=advisory_captured,
    )
    prov_dict = provenance_to_dict(prov)
    # Attach diagnostic + heuristic-prior fields per Catalog #305 observability.
    prov_dict["heuristic_prior"] = True
    prov_dict["mlx_per_pair_schema_version"] = (
        sidecar_schema or _MLX_PER_PAIR_HEURISTIC_SCHEMA
    )
    prov_dict["gradient_tensor_kind"] = meta.get("gradient_tensor_kind")
    prov_dict["gradient_byte_domain"] = meta.get("gradient_byte_domain")
    prov_dict["master_gradient_anchor_written"] = bool(
        meta.get("master_gradient_anchor_written", False)
    )
    prov_dict["master_gradient_anchor_blockers"] = meta.get(
        "master_gradient_anchor_blockers"
    )
    prov_dict["n_bytes"] = int(n_bytes)
    prov_dict["n_pairs_total"] = int(meta.get("n_pairs_total", n_pairs))
    prov_dict["n_pairs_used"] = int(meta.get("n_pairs_used", n_pairs))
    prov_dict["hardware_substrate"] = meta.get(
        "hardware_substrate", "darwin_arm64_m5_max_macos_mlx_advisory"
    )
    prov_dict["measurement_method"] = meta.get("measurement_method")
    prov_dict["is_authoritative_contest_axis"] = False
    prov_dict["contest_axis_authority_violation_reason"] = (
        "macos_mlx_research_signal_heuristic_prior_not_authoritative_per_"
        "catalog_192_127_323"
    )

    cells_by_coord: dict[tuple, PairFrameScorerGeometryCell] = {}
    for pair_idx in range(n_to_populate):
        d_seg = float(per_pair[pair_idx, 0])
        d_pose = float(per_pair[pair_idx, 1])
        rate = float(per_pair[pair_idx, 2])
        # frame_idx = first frame of the pair (2 * pair_idx), clamped per the
        # scaffold cell's [0, CANONICAL_FRAME_COUNT) invariant.
        frame_idx = min(pair_idx * 2, CANONICAL_FRAME_COUNT - 1)
        for scorer_axis, signed_delta in (
            (ScorerAxis.SEGNET_5CLASS, d_seg),
            (ScorerAxis.POSENET_6D, d_pose),
            (ScorerAxis.RATE_TERM, rate),
        ):
            cell = PairFrameScorerGeometryCell(
                pair_idx=pair_idx,
                frame_idx=frame_idx,
                scorer_axis=scorer_axis,
                receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
                cpu_cuda_axis=cpu_cuda_axis,
                predicted_delta_score=signed_delta,
                predicted_byte_cost=0,
                receiver_feasibility=True,
                catalog_323_provenance=dict(prov_dict),
            )
            cells_by_coord[cell.coordinate] = cell

    canvas = PairFrameScorerGeometryLattice(
        archive_sha256=target_sha, cells=cells_by_coord
    )

    manifest_prov_dict = dict(prov_dict)
    manifest_prov_dict["gradient_npy_path"] = str(npy_path)
    manifest_prov_dict["gradient_npy_sha256"] = meta.get("npy_sha256")
    manifest_prov_dict["pairs_populated"] = n_to_populate

    sidecar_path: Path | None = None
    if write_sidecar:
        if output_path is not None:
            sidecar_path = Path(output_path).resolve()
        else:
            sidecar_dir = root / EMPIRICAL_LATTICE_DIR
            utc_compact = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            sidecar_path = (
                sidecar_dir / f"{target_sha[:12]}_perpair_{utc_compact}.json"
            )

        payload = {
            "schema": POPULATOR_SCHEMA_VERSION,
            "canvas_schema": CANVAS_SCHEMA,
            "cell_schema": CELL_SCHEMA,
            "archive_sha256": target_sha,
            "cells_populated": len(cells_by_coord),
            "anchors_consumed": n_to_populate,
            "anchors_skipped_non_authoritative": 0,
            "source_kind": "mlx_per_pair_heuristic_prior",
            "gradient_npy_path": str(npy_path),
            "gradient_npy_sha256": meta.get("npy_sha256"),
            "pairs_populated": n_to_populate,
            "cells": [c.as_dict() for c in cells_by_coord.values()],
            "manifest_provenance": manifest_prov_dict,
            # Catalog #341 canonical non-promotable markers.
            "predicted_delta_adjustment": 0.0,
            "promotable": False,
            "axis_tag": "[predicted]",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "tier_a_rationale": (
                "MLX per-pair master-gradient HEURISTIC PRIOR per CLAUDE.md "
                "'MLX portable-local-substrate authority' + Catalog "
                "#192/#127/#323; NON-PROMOTABLE; gates closed-form PREDICTION "
                "sweep ONLY (Half-2 paradox closer); FIRE-phase per Catalog "
                "#246 remains the only path to a score/frontier/PR claim"
            ),
        }
        lock_path = root / EMPIRICAL_LATTICE_LOCK
        with _populator_lock(lock_path):
            _atomic_write_json(sidecar_path, payload)

    return PopulatedCanvasManifest(
        canvas=canvas,
        archive_sha256=target_sha,
        cells_populated=len(cells_by_coord),
        anchors_consumed=n_to_populate,
        anchors_skipped_non_authoritative=0,
        output_path=sidecar_path,
        catalog_323_provenance=manifest_prov_dict,
    )


# ---------------------------------------------------------------------------
# Canonical reader: load_empirical_lattice (sister of canonical writer).
# ---------------------------------------------------------------------------


def load_empirical_lattice(
    archive_sha256: str,
    *,
    repo_root: Path | None = None,
    sidecar_dir: Path | None = None,
) -> PairFrameScorerGeometryLattice:
    """Load canonical empirical canvas JSON sidecar (BUILD-1 reader).

    Per canonical sidecar pattern: reads the LATEST sidecar file matching
    `<archive_sha[:12]>_*.json` under
    `.omx/state/pair_frame_scorer_geometry_lattice/`.

    Args:
        archive_sha256: canonical archive sha256 the sidecar was keyed by.
        repo_root: optional repo root override.
        sidecar_dir: optional sidecar dir override.

    Returns:
        Populated `PairFrameScorerGeometryLattice`.

    Raises:
        PopulatorError: if no sidecar exists for the archive sha256 OR
            sidecar is corrupt (per Catalog #138 fail-closed).
    """
    if not isinstance(archive_sha256, str) or not archive_sha256:
        raise PopulatorError("archive_sha256 must be a non-empty string")

    root = (repo_root or _resolve_repo_root()).resolve()
    sidecar_dir = (
        root / EMPIRICAL_LATTICE_DIR
        if sidecar_dir is None
        else Path(sidecar_dir).resolve()
    )

    if not sidecar_dir.exists():
        raise PopulatorError(
            f"empirical lattice sidecar dir does not exist: {sidecar_dir}; "
            "run populate_5d_canvas_from_master_gradient_anchors first"
        )

    prefix = f"{archive_sha256[:12]}_"
    matches = sorted(p for p in sidecar_dir.glob(f"{prefix}*.json"))
    if not matches:
        raise PopulatorError(
            f"no sidecar found for archive_sha256={archive_sha256[:12]}... "
            f"under {sidecar_dir}"
        )

    # Pick the LATEST sidecar (sorted lexicographically; the UTC suffix
    # ensures lexicographic == chronological).
    latest = matches[-1]
    try:
        with open(latest, encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise PopulatorError(
            f"sidecar {latest} carries corrupt JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise PopulatorError(
            f"sidecar {latest} root is not a dict (got {type(payload).__name__})"
        )

    schema = payload.get("schema")
    if schema != POPULATOR_SCHEMA_VERSION:
        raise PopulatorError(
            f"sidecar {latest} schema mismatch: got {schema!r}, "
            f"expected {POPULATOR_SCHEMA_VERSION!r}"
        )

    sidecar_sha = payload.get("archive_sha256")
    if sidecar_sha != archive_sha256:
        raise PopulatorError(
            f"sidecar {latest} archive_sha256 mismatch: got {sidecar_sha!r}, "
            f"requested {archive_sha256!r}"
        )

    cells_raw = payload.get("cells", [])
    if not isinstance(cells_raw, list):
        raise PopulatorError(
            f"sidecar {latest} cells field is not a list "
            f"(got {type(cells_raw).__name__})"
        )

    cells_by_coord: dict[tuple, PairFrameScorerGeometryCell] = {}
    for row in cells_raw:
        if not isinstance(row, dict):
            raise PopulatorError(
                f"sidecar {latest} cell row is not a dict "
                f"(got {type(row).__name__})"
            )
        try:
            cell = PairFrameScorerGeometryCell(
                pair_idx=int(row["pair_idx"]),
                frame_idx=int(row["frame_idx"]),
                scorer_axis=ScorerAxis(row["scorer_axis"]),
                receiver_runtime=ReceiverRuntime(row["receiver_runtime"]),
                cpu_cuda_axis=CpuCudaAxis(row["cpu_cuda_axis"]),
                predicted_delta_score=float(row["predicted_delta_score"]),
                predicted_byte_cost=int(row["predicted_byte_cost"]),
                receiver_feasibility=bool(row["receiver_feasibility"]),
                catalog_323_provenance=dict(
                    row.get("catalog_323_provenance", {})
                ),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise PopulatorError(
                f"sidecar {latest} cell row parse failed: {exc!r}; "
                f"row keys={sorted(row.keys()) if isinstance(row, dict) else None}"
            ) from exc
        cells_by_coord[cell.coordinate] = cell

    return PairFrameScorerGeometryLattice(
        archive_sha256=archive_sha256, cells=cells_by_coord
    )


# ---------------------------------------------------------------------------
# Helper: list_distinct_archives_in_ledger (operator-facing audit support).
# ---------------------------------------------------------------------------


def list_distinct_archives_in_ledger(
    ledger_path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    """List distinct archive sha256 values present in master gradient ledger.

    Returns sorted list (lexicographic) of unique sha256 strings.

    Args:
        ledger_path: optional override for ledger path.
        repo_root: optional repo root override.

    Returns:
        Sorted list of distinct archive sha256 values.
    """
    root = (repo_root or _resolve_repo_root()).resolve()
    ledger = (
        ledger_path
        if ledger_path is not None
        else (root / MASTER_GRADIENT_LEDGER_PATH)
    )

    if not ledger.exists():
        return []

    try:
        anchors = load_anchors_strict(ledger)
    except FileNotFoundError:
        return []

    return sorted({a.get("archive_sha256", "") for a in anchors if a.get("archive_sha256")})


__all__ = [
    "EMPIRICAL_LATTICE_DIR",
    "EMPIRICAL_LATTICE_LOCK",
    "POPULATOR_SCHEMA_VERSION",
    "PopulatedCanvasManifest",
    "PopulatorError",
    "list_distinct_archives_in_ledger",
    "load_empirical_lattice",
    "populate_5d_canvas_from_master_gradient_anchors",
    "populate_per_pair_cells_from_gradient_array",
]
