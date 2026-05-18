# SPDX-License-Identifier: MIT
"""Tests for ``tools/option_b_archive_member_pre_entropy_sweep.py``.

Per Catalog #321 invariant: every emitted savings_estimate row must carry
``validation_status="VALIDATED_CONTEST_MEMBER"`` + ``evidence_tag`` per
Catalog #287.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog #229
premise verifier: tests cover positive (compressible member produces
positive savings), negative (random-bytes member ≈ 0 savings), aggregation
correctness, ranking, schema, and CLI end-to-end.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOL_PATH = REPO_ROOT / "tools/option_b_archive_member_pre_entropy_sweep.py"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from option_b_archive_member_pre_entropy_sweep import (  # type: ignore[import-not-found]  # noqa: E402
    Q4_MIN_DELIVERABLE_SAVINGS_FOR_RETARGET,
    SCHEMA_VERSION,
    aggregate_per_substrate,
    build_q4_recommendation,
    load_validated_targets_from_corrected_artifact,
    main,
    rank_by_savings,
)


# ──────────────────────────────────────────────────────────────────── #
# Synthetic fixtures                                                   #
# ──────────────────────────────────────────────────────────────────── #


def _write_zip_with_member(
    path: Path, member_name: str, data: bytes
) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(member_name, data)


def _write_corrected_artifact(
    tmp_path: Path,
    targets: list[dict[str, str]],
) -> Path:
    """Write a minimal corrected-artifact JSON containing the canonical
    per_substrate_results schema with VALIDATED_CONTEST_MEMBER rows."""
    per_results = {}
    for tgt in targets:
        per_results[tgt["substrate_name"]] = {
            "archive_path": tgt["archive_zip_path"],
            "archive_sha256": "deadbeef" * 8,
            "validation_status": "VALIDATED_CONTEST_MEMBER",
            "substrate_class": tgt.get("substrate_class", "post_entropy_contest_archive"),
            "evidence_grade_per_row": "predicted",
            "member_breakdown": {tgt["member_name"]: {}},
        }
    artifact = {
        "schema_version": "pre_entropy_pivot_probe_v1",
        "per_substrate_results": per_results,
        "substrates_at_entropy_floor": [t["substrate_name"] for t in targets],
        "substrates_post_entropy": [],
        "substrates_with_pre_entropy_bytes": [],
    }
    out = tmp_path / "corrected.json"
    out.write_text(json.dumps(artifact, indent=2))
    return out


# ──────────────────────────────────────────────────────────────────── #
# 1. load_validated_targets_from_corrected_artifact                    #
# ──────────────────────────────────────────────────────────────────── #


def test_load_targets_extracts_only_validated_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "corrected.json"
    artifact.write_text(
        json.dumps(
            {
                "per_substrate_results": {
                    "good": {
                        "validation_status": "VALIDATED_CONTEST_MEMBER",
                        "archive_path": str(tmp_path / "good.zip"),
                        "substrate_class": "post_entropy_contest_archive",
                        "member_breakdown": {"x": {}},
                    },
                    "bad": {
                        "validation_status": "REJECTED_RESEARCH_SIDECAR",
                        "archive_path": str(tmp_path / "bad.pt"),
                        "substrate_class": "raw_float_weights",
                        "member_breakdown": {},
                    },
                }
            }
        )
    )
    targets = load_validated_targets_from_corrected_artifact(artifact)
    assert len(targets) == 1
    assert targets[0]["substrate_name"] == "good"
    assert targets[0]["member_name"] == "x"


def test_load_targets_handles_empty_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "corrected.json"
    artifact.write_text(json.dumps({"per_substrate_results": {}}))
    targets = load_validated_targets_from_corrected_artifact(artifact)
    assert targets == []


# ──────────────────────────────────────────────────────────────────── #
# 2. Synthetic compressible member → positive savings                  #
# ──────────────────────────────────────────────────────────────────── #


def test_compressible_member_yields_positive_savings(tmp_path: Path) -> None:
    """A 100KB member of all-zero bytes is HIGHLY compressible (ratio ~0).

    Synthetic archives can't be placed under the real repo's canonical
    ``experiments/results/`` root (the prober resolves repo_root from the
    tool's own __file__ for Catalog #321 safety), so we exercise the
    apples-to-apples compression math by calling
    ``probe_member_compression`` directly on synthetic bytes; the
    classification + ratio invariants are the same contract
    ``probe_substrate_archive_member`` enforces internally.
    """
    from tools.pre_entropy_substrate_pivot_prober import (
        estimate_deliverable_score_savings,
        probe_member_compression,
    )

    m = probe_member_compression("0.bin", b"\x00" * 100_000)
    # All-zero bytes compress to ~tens of bytes; ratio should be tiny
    # AND classification should be PRE_ENTROPY (< 0.99).
    assert m.classification == "PRE_ENTROPY"
    assert m.best_ratio < 0.05  # nearly all bytes saved
    delta = estimate_deliverable_score_savings(
        pre_entropy_bytes=m.raw_bytes,
        best_compression_ratio=m.best_ratio,
    )
    assert delta > 0.0


# ──────────────────────────────────────────────────────────────────── #
# 3. Synthetic random member ≈ 0 savings                               #
# ──────────────────────────────────────────────────────────────────── #


def test_random_bytes_member_yields_zero_savings(tmp_path: Path) -> None:
    """A 100KB random-bytes member is at entropy floor (ratio ≥ ~1).

    Like the compressible-member test, we exercise the canonical compression
    math directly (the validator's repo-root resolution prevents synthetic
    archives outside the real repo from being VALIDATED_CONTEST_MEMBER).
    """
    from tools.pre_entropy_substrate_pivot_prober import (
        estimate_deliverable_score_savings,
        probe_member_compression,
    )

    # Use os.urandom for cryptographically random bytes (incompressible).
    m = probe_member_compression("x", os.urandom(100_000))
    # Random bytes either AT_FLOOR or POST_ENTROPY; in both cases
    # apples-to-apples savings ≈ 0 by definition.
    assert m.classification in {"AT_FLOOR", "POST_ENTROPY"}
    delta = estimate_deliverable_score_savings(
        pre_entropy_bytes=m.raw_bytes if m.classification == "PRE_ENTROPY" else 0,
        best_compression_ratio=m.best_ratio,
    )
    assert delta < 0.0005


# ──────────────────────────────────────────────────────────────────── #
# 4. Aggregation correctness                                           #
# ──────────────────────────────────────────────────────────────────── #


def test_aggregate_per_substrate_sums_member_savings() -> None:
    probe_dict = {
        "sub_a": {
            "archive_zip_path": "/path/to/a.zip",
            "archive_sha256": "a" * 64,
            "substrate_class": "raw_float_weights",
            "members": [
                {"name": "x", "savings_estimate": 0.001, "pre_entropy_bytes": 1000,
                 "at_floor_bytes": 0, "post_entropy_bytes": 0},
                {"name": "y", "savings_estimate": 0.002, "pre_entropy_bytes": 2000,
                 "at_floor_bytes": 0, "post_entropy_bytes": 0},
            ],
        }
    }
    agg = aggregate_per_substrate(probe_dict)
    assert "sub_a" in agg
    assert abs(agg["sub_a"]["aggregate_savings_estimate"] - 0.003) < 1e-12
    assert agg["sub_a"]["aggregate_pre_entropy_bytes"] == 3000
    assert agg["sub_a"]["validation_status"] == "VALIDATED_CONTEST_MEMBER"


def test_aggregate_handles_empty_members() -> None:
    probe_dict = {
        "sub_empty": {
            "archive_zip_path": "/path/to/empty.zip",
            "archive_sha256": "b" * 64,
            "substrate_class": "post_entropy_contest_archive",
            "members": [],
        }
    }
    agg = aggregate_per_substrate(probe_dict)
    assert agg["sub_empty"]["aggregate_savings_estimate"] == 0.0


# ──────────────────────────────────────────────────────────────────── #
# 5. Ranking correctness                                               #
# ──────────────────────────────────────────────────────────────────── #


def test_rank_by_savings_orders_descending() -> None:
    agg = {
        "low": {"aggregate_savings_estimate": 0.001, "aggregate_pre_entropy_bytes": 100},
        "high": {"aggregate_savings_estimate": 0.010, "aggregate_pre_entropy_bytes": 1000},
        "mid": {"aggregate_savings_estimate": 0.005, "aggregate_pre_entropy_bytes": 500},
    }
    ranked = rank_by_savings(agg)
    assert [name for name, _ in ranked] == ["high", "mid", "low"]


def test_rank_breaks_ties_by_pre_entropy_bytes() -> None:
    agg = {
        "a": {"aggregate_savings_estimate": 0.001, "aggregate_pre_entropy_bytes": 500},
        "b": {"aggregate_savings_estimate": 0.001, "aggregate_pre_entropy_bytes": 1000},
    }
    ranked = rank_by_savings(agg)
    assert ranked[0][0] == "b"  # b has more pre-entropy bytes → ranked higher


# ──────────────────────────────────────────────────────────────────── #
# 6. Q4 recommendation                                                 #
# ──────────────────────────────────────────────────────────────────── #


def test_q4_defer_when_top_savings_below_threshold() -> None:
    ranked = [("sub_low", 0.0005)]
    agg = {
        "sub_low": {
            "archive_sha256": "c" * 64,
            "members": [{"name": "x", "savings_estimate": 0.0005}],
        }
    }
    rec = build_q4_recommendation(ranked, agg)
    assert rec["verdict"] == "DEFER_Q4"
    assert rec["recommended_q4_target_substrate"] is None
    assert len(rec["alternative_paths"]) >= 4


def test_q4_build_when_top_savings_meets_threshold() -> None:
    ranked = [("sub_high", 0.005)]
    agg = {
        "sub_high": {
            "archive_sha256": "d" * 64,
            "members": [{"name": "x", "savings_estimate": 0.005}],
        }
    }
    rec = build_q4_recommendation(ranked, agg)
    assert rec["verdict"] == "BUILD_Q4_VALIDATED_TARGET"
    assert rec["recommended_q4_target_substrate"] == "sub_high"
    assert rec["recommended_q4_target_member_name"] == "x"


def test_q4_defer_on_empty_ranked() -> None:
    rec = build_q4_recommendation([], {})
    assert rec["verdict"] == "DEFER_Q4"
    assert rec["recommended_q4_target_substrate"] is None


# ──────────────────────────────────────────────────────────────────── #
# 7. Schema invariant per Catalog #321 / #287                          #
# ──────────────────────────────────────────────────────────────────── #


def test_emitted_artifact_carries_validation_status_per_member(tmp_path: Path) -> None:
    """Verify every per-member emitted row in the LIVE Option B sweep
    artifact carries `validation_status="VALIDATED_CONTEST_MEMBER"`
    (Catalog #321) AND an `evidence_tag` per Catalog #287. Uses the real
    repo's small `dp1` contest archive (25KB) so the validator's repo-root
    resolution accepts it as a contest member.
    """
    real_corrected = (
        REPO_ROOT
        / ".omx/state/wyner_ziv_deliverability/"
        "pre_entropy_candidate_substrates_corrected_20260517T215345.json"
    )
    if not real_corrected.exists():
        pytest.skip("corrected artifact not present in this checkout")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        rc = main(
            [
                "--corrected-artifact",
                str(real_corrected),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        files = list(out_dir.glob("option_b_archive_member_sweep_*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text())
        # Schema invariant
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["catalog_321_compliant"] is True
        assert payload["phantom_score_class_extincted"] is True
        assert payload["score_claim"] is False
        assert payload["promotion_eligible"] is False
        # All 8 VALIDATED archives present
        assert payload["candidates_validated"] == 8
        # Per-member invariant
        for substrate_name, row in payload["per_substrate_results"].items():
            assert row["validation_status"] == "VALIDATED_CONTEST_MEMBER"
            for m in row["members"]:
                assert m["validation_status"] == "VALIDATED_CONTEST_MEMBER"
                assert "evidence_tag" in m
                assert m["evidence_tag"].startswith("[empirical:")
    finally:
        os.chdir(cwd)


# ──────────────────────────────────────────────────────────────────── #
# 8. CLI end-to-end                                                    #
# ──────────────────────────────────────────────────────────────────── #


def test_cli_end_to_end(tmp_path: Path) -> None:
    """Run the CLI as a subprocess against the LIVE corrected artifact in
    the real repo. Confirms CLI rc + JSON contract."""
    real_corrected = (
        REPO_ROOT
        / ".omx/state/wyner_ziv_deliverability/"
        "pre_entropy_candidate_substrates_corrected_20260517T215345.json"
    )
    if not real_corrected.exists():
        pytest.skip("corrected artifact not present in this checkout")
    out_dir = tmp_path / "out_cli"
    out_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + str(REPO_ROOT / "tools")
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--corrected-artifact",
            str(real_corrected),
            "--output-dir",
            str(out_dir),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["candidates_validated"] == 8
    assert payload["q4_verdict"] in {"BUILD_Q4_VALIDATED_TARGET", "DEFER_Q4"}
    # Empirical guard: this corrected artifact is known to produce
    # DEFER_Q4 (top aggregate savings ~0.000332 < 0.001 threshold).
    assert payload["q4_verdict"] == "DEFER_Q4"


# ──────────────────────────────────────────────────────────────────── #
# 9. Threshold constant pinned                                         #
# ──────────────────────────────────────────────────────────────────── #


def test_q4_min_threshold_pinned() -> None:
    """Catalog #321 + Catalog #287: the Q4 retarget threshold must be at the
    precision floor of the contest leaderboard (~0.001). Pin this constant
    so a future refactor cannot silently downgrade the verdict criterion."""
    assert Q4_MIN_DELIVERABLE_SAVINGS_FOR_RETARGET == 0.001


def test_schema_version_pinned() -> None:
    assert SCHEMA_VERSION == "option_b_archive_member_sweep_v1"
