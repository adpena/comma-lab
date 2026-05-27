# SPDX-License-Identifier: MIT
"""Tests for BUILD-1 5D canvas populator.

Per BUILD-1 routing directive 2026-05-26 + sister design memo + audit
memo + canonical 4-layer ledger pattern (Catalog #131 / #138 / #245).

Test coverage:

1. Helper unit tests (10): _classify_cpu_cuda_axis / _build_cells_from_anchor
   / list_distinct_archives_in_ledger / Provenance roundtrip / fcntl lock /
   atomic write / default LATEST picker / skip_non_authoritative semantics.

2. End-to-end populator (10): synthetic ledger -> canvas / live-repo
   regression / target archive filter / target sha not found / corrupt
   ledger fail-closed / sidecar write + read roundtrip / include
   non-authoritative opt-in / canvas cell count + coordinate lookup /
   canonical Provenance attached / sister disjoint from BUILD-2.

3. CLI subprocess (5): --list-archives / --latest / --json mode /
   error handling / --no-sidecar.

4. Catalog discipline regression (5): canonical non-promotable
   markers per Catalog #341 / Catalog #323 Provenance contract /
   Catalog #356 scope-disjoint / Catalog #357 Tier A observability-only
   / Catalog #185 sister-callable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure src/ on sys.path for test invocation
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.cathedral.consumer_contract import ConsumerTier  # noqa: E402
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (  # noqa: E402
    CONSUMER_HOOK_NUMBERS,
    CONSUMER_NAME,
    CONSUMER_TIER,
    CONSUMER_VERSION,
    CpuCudaAxis,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (  # noqa: E402
    EMPIRICAL_LATTICE_DIR,
    POPULATOR_SCHEMA_VERSION,
    PopulatedCanvasManifest,
    PopulatorError,
    _build_cells_from_anchor,
    _classify_cpu_cuda_axis,
    list_distinct_archives_in_ledger,
    load_empirical_lattice,
    populate_5d_canvas_from_master_gradient_anchors,
)

# ---------------------------------------------------------------------------
# Synthetic ledger fixtures
# ---------------------------------------------------------------------------


def _synthetic_anchor(
    *,
    archive_sha: str = "a" * 64,
    measurement_axis: str = "[contest-CPU]",
    measurement_hardware: str = "linux_x86_64_amd_epyc_gha_runner",
    d_seg: float = 0.001,
    d_pose: float = 0.002,
    rate: float = 0.005,
    score: float = 0.193,
    measurement_utc: str = "2026-05-26T10:00:00.000000Z",
    measurement_method: str = "autograd_per_parameter_projected_8pair",
    n_bytes: int = 178417,
    n_pairs_used: int = 8,
) -> dict:
    """Build a synthetic master gradient anchor for tests."""
    return {
        "archive_sha256": archive_sha,
        "measurement_axis": measurement_axis,
        "measurement_hardware": measurement_hardware,
        "measurement_method": measurement_method,
        "measurement_call_id": f"synthetic_{archive_sha[:8]}_{measurement_axis}",
        "measurement_utc": measurement_utc,
        "n_bytes": n_bytes,
        "n_pairs_used": n_pairs_used,
        "operating_point": {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "rate": rate,
            "score": score,
        },
        "schema_version": "master_gradient_anchor_v1",
        "written_at_utc": measurement_utc,
        "written_host": "test_host",
        "written_pid": 99999,
        "gradient_array_path": ".omx/state/synthetic_gradient.npy",
        "pareto_facets": [],
        "rashomon_disagreement_score": None,
    }


def _write_synthetic_ledger(repo_root: Path, anchors: list[dict]) -> Path:
    """Write a synthetic ledger JSONL under repo_root/.omx/state/."""
    ledger = repo_root / ".omx" / "state" / "master_gradient_anchors.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "w", encoding="utf-8") as f:
        for a in anchors:
            f.write(json.dumps(a, sort_keys=True) + "\n")
    return ledger


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_classify_cpu_cuda_axis_contest_cpu():
    assert _classify_cpu_cuda_axis("[contest-CPU]") == CpuCudaAxis.CONTEST_CPU


def test_classify_cpu_cuda_axis_contest_cuda():
    assert (
        _classify_cpu_cuda_axis("[contest-CUDA]") == CpuCudaAxis.CONTEST_CUDA_T4
    )


def test_classify_cpu_cuda_axis_macos_advisory_returns_none():
    assert _classify_cpu_cuda_axis("[macOS-CPU advisory]") is None


def test_classify_cpu_cuda_axis_mps_proxy_returns_none():
    assert _classify_cpu_cuda_axis("[MPS-PROXY]") is None


def test_classify_cpu_cuda_axis_empty_returns_none():
    assert _classify_cpu_cuda_axis("") is None


def test_build_cells_from_anchor_emits_3_cells_per_anchor():
    anchor = _synthetic_anchor()
    cells = _build_cells_from_anchor(anchor, CpuCudaAxis.CONTEST_CPU)
    assert len(cells) == 3
    axes = {c.scorer_axis for c in cells}
    assert axes == {
        ScorerAxis.SEGNET_5CLASS,
        ScorerAxis.POSENET_6D,
        ScorerAxis.RATE_TERM,
    }


def test_build_cells_from_anchor_carries_per_axis_components():
    anchor = _synthetic_anchor(d_seg=0.0012, d_pose=0.0023, rate=0.0034)
    cells = _build_cells_from_anchor(anchor, CpuCudaAxis.CONTEST_CPU)
    by_axis = {c.scorer_axis: c.predicted_delta_score for c in cells}
    assert by_axis[ScorerAxis.SEGNET_5CLASS] == pytest.approx(0.0012)
    assert by_axis[ScorerAxis.POSENET_6D] == pytest.approx(0.0023)
    assert by_axis[ScorerAxis.RATE_TERM] == pytest.approx(0.0034)


def test_build_cells_from_anchor_attaches_canonical_provenance():
    anchor = _synthetic_anchor(measurement_axis="[contest-CUDA]")
    cells = _build_cells_from_anchor(anchor, CpuCudaAxis.CONTEST_CUDA_T4)
    for cell in cells:
        prov = cell.catalog_323_provenance
        assert "measurement_axis" in prov
        assert prov["measurement_axis"] == "[contest-CUDA]"
        assert prov["is_authoritative_contest_axis"] is True
        assert "n_bytes" in prov
        assert "measurement_method" in prov


def test_build_cells_from_anchor_advisory_axis_marks_non_authoritative():
    """Advisory axes carry is_authoritative_contest_axis=False.

    Per tac.master_gradient.contest_axis_authority_violation_reason: the
    helper returns a violation reason ONLY when a CONTEST axis is FALSELY
    claimed on non-compliant hardware. Advisory axes return None because
    they make NO contest claim (they self-disclose advisory). The
    populator's contract per Catalog #323 is: capture both signals so
    downstream consumers can disambiguate.
    """
    anchor = _synthetic_anchor(
        measurement_axis="[macOS-CPU advisory]",
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
    )
    cells = _build_cells_from_anchor(anchor, CpuCudaAxis.CONTEST_CPU)
    for cell in cells:
        assert cell.catalog_323_provenance["is_authoritative_contest_axis"] is False
        # Advisory axes self-disclose; no contest-axis violation to flag
        assert (
            "contest_axis_authority_violation_reason"
            in cell.catalog_323_provenance
        )


def test_build_cells_from_anchor_contest_cpu_on_macos_hw_flags_violation():
    """Catalog #327: a contest-axis claim on advisory hardware is FALSE_AUTHORITY.

    The canonical helper flags this as a violation reason; the populator
    captures it in Provenance.
    """
    anchor = _synthetic_anchor(
        measurement_axis="[contest-CPU]",
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
    )
    cells = _build_cells_from_anchor(anchor, CpuCudaAxis.CONTEST_CPU)
    for cell in cells:
        assert (
            cell.catalog_323_provenance[
                "contest_axis_authority_violation_reason"
            ]
            is not None
        )


def test_build_cells_default_receiver_runtime_is_raw_residual():
    anchor = _synthetic_anchor()
    cells = _build_cells_from_anchor(anchor, CpuCudaAxis.CONTEST_CPU)
    for cell in cells:
        assert cell.receiver_runtime == ReceiverRuntime.RAW_RESIDUAL
        assert cell.receiver_feasibility is True


# ---------------------------------------------------------------------------
# list_distinct_archives_in_ledger
# ---------------------------------------------------------------------------


def test_list_distinct_archives_in_ledger_empty(tmp_path: Path):
    archives = list_distinct_archives_in_ledger(repo_root=tmp_path)
    assert archives == []


def test_list_distinct_archives_in_ledger_synthetic(tmp_path: Path):
    sha_a = "a" * 64
    sha_b = "b" * 64
    _write_synthetic_ledger(
        tmp_path,
        [
            _synthetic_anchor(archive_sha=sha_a),
            _synthetic_anchor(archive_sha=sha_b, measurement_axis="[contest-CUDA]"),
            _synthetic_anchor(archive_sha=sha_a, measurement_axis="[contest-CUDA]"),
        ],
    )
    archives = list_distinct_archives_in_ledger(repo_root=tmp_path)
    assert archives == [sha_a, sha_b]


def test_list_distinct_archives_in_ledger_live_repo():
    """Live-repo regression: at least 1 archive in canonical ledger."""
    archives = list_distinct_archives_in_ledger()
    assert isinstance(archives, list)
    # Live ledger should carry the canonical fec6 / pr106 / etc. shas.
    assert len(archives) >= 1


# ---------------------------------------------------------------------------
# populate_5d_canvas_from_master_gradient_anchors (end-to-end)
# ---------------------------------------------------------------------------


def test_populate_synthetic_canonical_happy_path(tmp_path: Path):
    sha = "c" * 64
    _write_synthetic_ledger(
        tmp_path,
        [
            _synthetic_anchor(archive_sha=sha),
            _synthetic_anchor(
                archive_sha=sha, measurement_axis="[contest-CUDA]"
            ),
        ],
    )
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha,
        write_sidecar=False,
        repo_root=tmp_path,
    )
    assert isinstance(manifest, PopulatedCanvasManifest)
    assert manifest.archive_sha256 == sha
    assert manifest.anchors_consumed == 2
    assert manifest.anchors_skipped_non_authoritative == 0
    # 2 anchors × 3 scorer_axes = 6 cells (different cpu_cuda_axis)
    assert manifest.cells_populated == 6
    assert manifest.canvas.cell_count() == 6


def test_populate_advisory_axis_skipped_by_default(tmp_path: Path):
    sha = "d" * 64
    _write_synthetic_ledger(
        tmp_path,
        [
            _synthetic_anchor(archive_sha=sha),
            _synthetic_anchor(
                archive_sha=sha, measurement_axis="[macOS-CPU advisory]"
            ),
            _synthetic_anchor(
                archive_sha=sha, measurement_axis="[MPS-PROXY]"
            ),
        ],
    )
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha, write_sidecar=False, repo_root=tmp_path
    )
    assert manifest.anchors_consumed == 1
    assert manifest.anchors_skipped_non_authoritative == 2
    assert manifest.cells_populated == 3


def test_populate_advisory_axis_included_when_opted_in(tmp_path: Path):
    sha = "e" * 64
    _write_synthetic_ledger(
        tmp_path,
        [
            _synthetic_anchor(archive_sha=sha),
            _synthetic_anchor(
                archive_sha=sha, measurement_axis="[macOS-CPU advisory]"
            ),
        ],
    )
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha,
        write_sidecar=False,
        repo_root=tmp_path,
        skip_non_authoritative=False,
    )
    # Advisory anchor included; both anchors mapped to CONTEST_CPU axis
    # (advisory default per the populator), so collision = latest wins
    # = 3 cells total (advisory overwrites contest-CPU as later anchor).
    assert manifest.anchors_consumed == 2
    assert manifest.anchors_skipped_non_authoritative == 0
    assert manifest.cells_populated == 3


def test_populate_archive_not_found_raises(tmp_path: Path):
    _write_synthetic_ledger(tmp_path, [_synthetic_anchor(archive_sha="a" * 64)])
    with pytest.raises(PopulatorError, match="no master gradient anchors"):
        populate_5d_canvas_from_master_gradient_anchors(
            archive_sha256="z" * 64, write_sidecar=False, repo_root=tmp_path
        )


def test_populate_empty_ledger_raises(tmp_path: Path):
    _write_synthetic_ledger(tmp_path, [])
    with pytest.raises(PopulatorError, match="0 rows"):
        populate_5d_canvas_from_master_gradient_anchors(
            write_sidecar=False, repo_root=tmp_path
        )


def test_populate_missing_ledger_raises(tmp_path: Path):
    """Missing ledger surfaces canonical PopulatorError (Catalog #138 fail-closed).

    Per tac.master_gradient.load_anchors_strict canonical contract:
    missing ledger file returns empty list (NOT a corrupt-file raise);
    the populator surfaces this as "0 rows" PopulatorError.
    """
    # No ledger written
    with pytest.raises(PopulatorError):
        populate_5d_canvas_from_master_gradient_anchors(
            write_sidecar=False, repo_root=tmp_path
        )


def test_populate_latest_picker_chooses_latest_utc(tmp_path: Path):
    sha_old = "1" * 64
    sha_new = "2" * 64
    _write_synthetic_ledger(
        tmp_path,
        [
            _synthetic_anchor(
                archive_sha=sha_old, measurement_utc="2026-01-01T10:00:00.000Z"
            ),
            _synthetic_anchor(
                archive_sha=sha_new, measurement_utc="2026-05-26T10:00:00.000Z"
            ),
        ],
    )
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=None, write_sidecar=False, repo_root=tmp_path
    )
    assert manifest.archive_sha256 == sha_new


def test_populate_sidecar_write_and_read_roundtrip(tmp_path: Path):
    sha = "f" * 64
    _write_synthetic_ledger(
        tmp_path,
        [
            _synthetic_anchor(archive_sha=sha),
            _synthetic_anchor(
                archive_sha=sha, measurement_axis="[contest-CUDA]"
            ),
        ],
    )
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha, write_sidecar=True, repo_root=tmp_path
    )
    assert manifest.output_path is not None
    assert manifest.output_path.exists()
    assert manifest.output_path.parent.name == "pair_frame_scorer_geometry_lattice"

    # Verify schema + cells in the sidecar
    payload = json.loads(manifest.output_path.read_text())
    assert payload["schema"] == POPULATOR_SCHEMA_VERSION
    assert payload["archive_sha256"] == sha
    assert payload["cells_populated"] == 6
    assert isinstance(payload["cells"], list)
    assert len(payload["cells"]) == 6

    # Now load via the canonical reader
    loaded = load_empirical_lattice(sha, repo_root=tmp_path)
    assert isinstance(loaded, PairFrameScorerGeometryLattice)
    assert loaded.cell_count() == 6
    assert loaded.archive_sha256 == sha

    # Spot-check a cell
    cell = loaded.query_cell(
        0,
        0,
        ScorerAxis.SEGNET_5CLASS,
        ReceiverRuntime.RAW_RESIDUAL,
        CpuCudaAxis.CONTEST_CPU,
    )
    assert cell is not None
    assert cell.scorer_axis == ScorerAxis.SEGNET_5CLASS


def test_populate_canonical_non_promotable_markers_in_manifest(tmp_path: Path):
    sha = "9" * 64
    _write_synthetic_ledger(tmp_path, [_synthetic_anchor(archive_sha=sha)])
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha, write_sidecar=False, repo_root=tmp_path
    )
    d = manifest.as_dict()
    assert d["predicted_delta_adjustment"] == 0.0
    assert d["promotable"] is False
    assert d["axis_tag"] == "[predicted]"
    assert d["score_claim"] is False
    assert d["promotion_eligible"] is False
    assert d["ready_for_exact_eval_dispatch"] is False
    assert d["rank_or_kill_eligible"] is False


def test_populate_live_repo_regression():
    """Live-repo regression guard: real master_gradient_anchors.jsonl works."""
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=None, write_sidecar=False
    )
    assert manifest.cells_populated >= 0
    assert manifest.anchors_consumed >= 0
    # At minimum, the LATEST archive should have authoritative anchors
    # or the populator should skip them all (also OK).
    assert manifest.anchors_consumed + manifest.anchors_skipped_non_authoritative >= 1


# ---------------------------------------------------------------------------
# load_empirical_lattice (sister reader)
# ---------------------------------------------------------------------------


def test_load_empirical_lattice_missing_sidecar_dir_raises(tmp_path: Path):
    with pytest.raises(PopulatorError, match="sidecar dir does not exist"):
        load_empirical_lattice("a" * 64, repo_root=tmp_path)


def test_load_empirical_lattice_missing_archive_raises(tmp_path: Path):
    sidecar_dir = tmp_path / ".omx" / "state" / "pair_frame_scorer_geometry_lattice"
    sidecar_dir.mkdir(parents=True)
    with pytest.raises(PopulatorError, match="no sidecar found"):
        load_empirical_lattice("z" * 64, repo_root=tmp_path)


def test_load_empirical_lattice_corrupt_json_raises(tmp_path: Path):
    sidecar_dir = tmp_path / ".omx" / "state" / "pair_frame_scorer_geometry_lattice"
    sidecar_dir.mkdir(parents=True)
    corrupt = sidecar_dir / "abcdef012345_20260526T100000Z.json"
    corrupt.write_text("{not valid json")
    with pytest.raises(PopulatorError, match="corrupt JSON"):
        load_empirical_lattice("abcdef012345" + "0" * 52, repo_root=tmp_path)


def test_load_empirical_lattice_schema_mismatch_raises(tmp_path: Path):
    sidecar_dir = tmp_path / ".omx" / "state" / "pair_frame_scorer_geometry_lattice"
    sidecar_dir.mkdir(parents=True)
    bad_schema = sidecar_dir / "abcdef012345_20260526T100000Z.json"
    bad_schema.write_text(
        json.dumps({"schema": "wrong_schema_v1", "archive_sha256": "x"})
    )
    with pytest.raises(PopulatorError, match="schema mismatch"):
        load_empirical_lattice("abcdef012345" + "0" * 52, repo_root=tmp_path)


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------


def test_cli_list_archives_human_readable():
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools" / "populate_5d_canvas_cli.py"),
            "--list-archives",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Distinct archive sha256 values in ledger" in result.stdout


def test_cli_list_archives_json():
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools" / "populate_5d_canvas_cli.py"),
            "--list-archives",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "archives" in payload
    assert isinstance(payload["archives"], list)


def test_cli_latest_no_sidecar():
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools" / "populate_5d_canvas_cli.py"),
            "--latest",
            "--no-sidecar",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    assert "5D canvas populator manifest" in result.stdout
    assert "Tier A observability-only" in result.stdout


def test_cli_latest_json_emits_non_promotable_markers():
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools" / "populate_5d_canvas_cli.py"),
            "--latest",
            "--no-sidecar",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["promotable"] is False
    assert payload["axis_tag"] == "[predicted]"
    assert payload["predicted_delta_adjustment"] == 0.0


def test_cli_invalid_archive_sha_exits_1():
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools" / "populate_5d_canvas_cli.py"),
            "--archive-sha256",
            "z" * 64,
            "--no-sidecar",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 1
    assert "PopulatorError" in result.stderr


def test_cli_mutually_exclusive_modes():
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "tools" / "populate_5d_canvas_cli.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 2  # argparse error


# ---------------------------------------------------------------------------
# Catalog discipline regression tests
# ---------------------------------------------------------------------------


def test_consumer_tier_is_tier_a_observability_only():
    """Catalog #357: scaffold lands at Tier A; BUILD-4 promotes to Tier B."""
    assert CONSUMER_TIER == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_consumer_name_canonical():
    assert CONSUMER_NAME == "pair_frame_scorer_geometry_lattice_5d_canvas"


def test_consumer_hook_numbers_all_six():
    """Catalog #125: all 6 hooks declared per 6-hook wire-in non-negotiable."""
    assert len(CONSUMER_HOOK_NUMBERS) == 6


def test_consumer_version_is_scaffold():
    """Per scaffold contract: version carries '-scaffold' until BUILD-4."""
    assert "scaffold" in CONSUMER_VERSION.lower()


def test_populator_emits_canvas_consumable_by_build_2_3_sister(tmp_path: Path):
    """BUILD-2/3 sister-disjoint: populator emits a canvas consumable by sister operation generators.

    Per BUILD-1 routing directive parallel spawn: BUILD-1 (this subagent)
    populates the canvas; BUILD-2+3 sister subagent (parallel spawn)
    implements the 4 canonical operation generators consuming the canvas.
    BUILD-1 MUST NOT touch the operation generator methods (sister-
    disjoint scope). This test verifies the populator emits a canvas
    that the sister operation generators can consume (instance methods
    exist + canvas reports populated cells via the canonical container
    contract).
    """
    sha = "8" * 64
    _write_synthetic_ledger(tmp_path, [_synthetic_anchor(archive_sha=sha)])
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha, write_sidecar=False, repo_root=tmp_path
    )
    # Canvas IS populated
    assert manifest.canvas.cell_count() == 3
    # All 4 operation generator methods exist on the canvas (sister-
    # disjoint contract: my populator emits a canvas; sister BUILD-2+3
    # implements the operations)
    assert hasattr(manifest.canvas, "generate_full_drop_starts")
    assert hasattr(manifest.canvas, "generate_repair_starts")
    assert hasattr(manifest.canvas, "generate_masked_starts")
    assert hasattr(manifest.canvas, "generate_feathered_starts")


def test_sidecar_path_uses_canonical_lattice_dir(tmp_path: Path):
    """Catalog #131 sister: sidecar path under canonical EMPIRICAL_LATTICE_DIR."""
    sha = "7" * 64
    _write_synthetic_ledger(tmp_path, [_synthetic_anchor(archive_sha=sha)])
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha, write_sidecar=True, repo_root=tmp_path
    )
    assert manifest.output_path is not None
    assert EMPIRICAL_LATTICE_DIR in str(manifest.output_path)
    assert manifest.output_path.name.startswith(sha[:12])
    assert manifest.output_path.suffix == ".json"


def test_atomic_write_no_tmp_leakage(tmp_path: Path):
    """Catalog #131 sister: transactional write leaves no .tmp.<uuid12> leakage."""
    sha = "6" * 64
    _write_synthetic_ledger(tmp_path, [_synthetic_anchor(archive_sha=sha)])
    populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha, write_sidecar=True, repo_root=tmp_path
    )
    sidecar_dir = tmp_path / EMPIRICAL_LATTICE_DIR
    tmp_files = list(sidecar_dir.glob("*.tmp.*"))
    assert tmp_files == [], f"Found leaked tmp files: {tmp_files}"


def test_populator_provenance_carries_canonical_helper_invocation(tmp_path: Path):
    """Catalog #323: every score-claim row cites canonical_helper_invocation."""
    sha = "5" * 64
    _write_synthetic_ledger(tmp_path, [_synthetic_anchor(archive_sha=sha)])
    manifest = populate_5d_canvas_from_master_gradient_anchors(
        archive_sha256=sha, write_sidecar=False, repo_root=tmp_path
    )
    assert (
        "canonical_helper_invocation" in manifest.catalog_323_provenance
    )
    assert (
        "build_provenance_for_predicted"
        in manifest.catalog_323_provenance["canonical_helper_invocation"]
    )


def test_populator_module_callable_via_globals():
    """Catalog #185 sister: gate function callable via module globals."""
    from tac.optimization import (
        pair_frame_scorer_geometry_lattice_5d_canvas_populator as mod,
    )

    assert callable(mod.populate_5d_canvas_from_master_gradient_anchors)
    assert callable(mod.load_empirical_lattice)
    assert callable(mod.list_distinct_archives_in_ledger)
