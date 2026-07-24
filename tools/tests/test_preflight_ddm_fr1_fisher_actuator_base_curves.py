from __future__ import annotations

import hashlib
import json
from pathlib import Path

import brotli
import pytest

from tac.optimization.ddm_fr1_fisher_preflight import (
    FR1PreflightError,
    build_preflight_receipt,
)


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fixture(tmp_path: Path, *, ready: bool = False) -> dict[str, object]:
    columns = [
        "pair",
        "row",
        "col",
        "linear_index",
        "target_class",
        "realized_class",
        "necessity_edge_tier",
        "resize_support_taps",
        "top1_top2_margin",
        "fisher_trace",
        "target_realized_head_pair_norm",
        "flip_distance",
        "vjp_native_arrangement_match",
        "vjp_local_lipschitz",
        "vjp_unit_pullback_rgb",
    ]
    header = {
        "schema": "r1b5_fisher_ev_ordering_jsonl.v1",
        "candidate_count": 2,
        "columns": columns,
    }
    row1 = [22, 225, 45, 4440621, 1, 0, 0, 4, 0.000048, 0.5, 3.95, 0.000012, True, 0.89, [0.4, -0.1, -0.8]]
    row2 = [14, 189, 323, 1, 1, 0, 0, 4, 0.000097, 0.5, 3.95, 0.000024, True, 1.08, [-0.5, -0.8, -0.02]]
    ordering_raw = b"\n".join(_json_bytes(row) for row in [header, row1, row2]) + b"\n"
    ordering = _write(tmp_path / "ordering.jsonl.br", brotli.compress(ordering_raw))
    ordering["candidate_count"] = 2

    fisher = {
        "schema": "r1b5_fisher_ev_and_resize_coupling_audit.v1",
        "ordering_artifact": {"sha256": ordering["sha256"]},
        "ranking": {"candidate_count": 2},
        "blockers": [] if ready else ["PER_CANDIDATE_EXACT_PREFIX_BYTE_MARGINAL_ABSENT"],
    }
    fisher_binding = _write(tmp_path / "fisher.json", _json_bytes(fisher))

    inner = {
        "schema": "m1_band_inner_jacobian_secant_qp_status.v1",
        "first_order_vjp": "MEASURED_REAL_N600",
        "realized_backbone_secants": "MEASURED_RECEIVER_CLOSED" if ready else "ABSENT",
        "qp_receiver_closure": "MEASURED_RECEIVER_CLOSED" if ready else "ABSENT",
        "formalization": "FORMALIZED_EXECUTABLE" if ready else "FORMALIZATION_PENDING",
    }
    inner_binding = _write(tmp_path / "records" / "inner.json", _json_bytes(inner))
    band = {
        "custody": {
            "ev_selection": {
                "artifact_records": {
                    "inner_jacobian_secant_qp": {
                        "path": "records/inner.json",
                        "sha256": inner_binding["sha256"],
                    }
                }
            }
        }
    }
    band_binding = _write(tmp_path / "band.json", _json_bytes(band))

    archive = _write(tmp_path / "v19c.zip", b"v19c")
    menu_config = {
        "v19c_archive_sha256": archive["sha256"],
        "v19c_archive_bytes": archive["bytes"],
    }
    menu_config_binding = _write(tmp_path / "menu_config.json", _json_bytes(menu_config))
    v19c_row = {
        "candidate_id": "v19c_base",
        "archive_bytes": 137827,
        "errors": 2923991,
        "d_seg": 0.024786978827582466,
        "d_pose": 163.06121002915629,
    }
    v19c_binding = _write(
        tmp_path / "v19c_receipt.json",
        _json_bytes({"curve": [v19c_row, {"candidate_id": "joined", "errors": 8318787}]}),
    )
    ws1_row = {
        "candidate_id": "ws1",
        "archive_bytes": 138031,
        "errors": 2845843,
        "d_seg": 0.024124510023328993,
        "d_pose": 146.3649324958955,
    }
    ws1_binding = _write(
        tmp_path / "ws1.json",
        _json_bytes(
            {
                "warm_start_candidates": {"W_seg": ws1_row},
                "seg_lexicographic_rerank": {"receiver_recompile_status": {"archive_sha256": "base-only"}},
            }
        ),
    )
    runtime_binding = _write(
        tmp_path / "runtime.py",
        b"class DDMRuntimePerturbationV1:\n    pass\n",
    )
    return {
        "schema": "DDMFR1FisherActuatorBaseCurvesConfigV1",
        "run_id": "test",
        "lane_id": "lane_test",
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "ordering": ordering,
        "fisher_receipt": fisher_binding,
        "band_manifest": band_binding,
        "runtime_sensitivity": runtime_binding,
        "v19c": {
            "menu_config": menu_config_binding,
            "receipt": v19c_binding,
            "archive": archive,
            "candidate_id": "v19c_base",
            "expected_row": v19c_row,
        },
        "ws1": {"receipt": ws1_binding, "expected_row": ws1_row},
        "ws2_receipt_glob": "no-ws2/**/*.json",
    }


def test_blocked_receipt_names_rank1_and_never_emits_fake_deltas(tmp_path: Path) -> None:
    receipt = build_preflight_receipt(_fixture(tmp_path), repo_root=tmp_path)
    top = receipt["top_ranked_candidate"]
    assert top["rank"] == 1
    assert top["pair"] == 22
    assert top["target_class_name"] == "Lane"
    assert top["realized_class_name"] == "Road"
    assert receipt["execution_allowed"] is False
    assert "FR1_CORRECTED_INNER_JACOBIAN_REALIZED_SECANTS_ABSENT" in receipt["blockers"]
    assert receipt["measurement"]["rows"] == []
    assert receipt["measurement"]["joint_delta_S"] == "NOT_MEASURED"


def test_v19c_base_is_not_confused_with_joined_curve(tmp_path: Path) -> None:
    receipt = build_preflight_receipt(_fixture(tmp_path), repo_root=tmp_path)
    assert receipt["base_curves"]["v19c_endpoint"]["errors"] == 2923991


def test_ws1_endpoint_is_explicitly_not_a_materialized_state(tmp_path: Path) -> None:
    receipt = build_preflight_receipt(_fixture(tmp_path), repo_root=tmp_path)
    ws1 = receipt["base_curves"]["ws1_W_seg"]
    assert ws1["endpoint_is_state"] is False
    assert "does not bind a materialized W_seg archive" in ws1["endpoint_state_caveat"]


def test_sha_drift_refuses_before_interpretation(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    Path(config["ordering"]["path"]).write_bytes(b"drift")
    with pytest.raises(FR1PreflightError, match="sha256 drift"):
        build_preflight_receipt(config, repo_root=tmp_path)


def test_missing_binding_key_refuses_even_when_other_keys_exist(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["ordering"].pop("sha256")
    with pytest.raises(FR1PreflightError, match="lacks path or sha256"):
        build_preflight_receipt(config, repo_root=tmp_path)


def test_missing_inner_status_binding_refuses(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    band_path = Path(config["band_manifest"]["path"])
    band_path.write_bytes(_json_bytes({"custody": {}}))
    config["band_manifest"] = _write(band_path, band_path.read_bytes())
    with pytest.raises(FR1PreflightError, match="lacks inner-Jacobian"):
        build_preflight_receipt(config, repo_root=tmp_path)


def test_ready_inner_status_still_refuses_without_runtime_bridge(tmp_path: Path) -> None:
    receipt = build_preflight_receipt(
        _fixture(tmp_path, ready=True),
        repo_root=tmp_path,
    )
    assert receipt["blockers"] == ["FR1_RANK1_TO_DDM_RUNTIME_PERTURBATION_BRIDGE_ABSENT"]
    assert receipt["execution_allowed"] is False


def test_ws2_receipts_are_observed_but_not_substituted(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["ws2_receipt_glob"] = "ws2/**/*.json"
    _write(tmp_path / "ws2" / "receipt.json", b"{}")
    receipt = build_preflight_receipt(config, repo_root=tmp_path)
    assert receipt["base_curves"]["ws2"]["observed_receipt_count"] == 1
    assert receipt["base_curves"]["ws2"]["launchable_materialized_W_seg_count"] == 0
    assert receipt["base_curves"]["ws2"]["fallback"].startswith("WS1_ENDPOINT")


def test_ws2_launchable_requires_bound_archive_bytes(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["ws2_receipt_glob"] = "ws2/**/*.json"
    archive = _write(tmp_path / "ws2" / "wseg.zip", b"materialized")
    ws2 = {
        "execution_allowed": True,
        "score_claim": False,
        "warm_start_candidates": {
            "W_seg": {
                "archive_path": archive["path"],
                "archive_sha256": archive["sha256"],
                "archive_bytes": archive["bytes"],
            }
        },
    }
    _write(tmp_path / "ws2" / "receipt.json", _json_bytes(ws2))
    receipt = build_preflight_receipt(config, repo_root=tmp_path)
    assert receipt["base_curves"]["ws2"]["launchable_materialized_W_seg_count"] == 1
    assert receipt["base_curves"]["ws2"]["fallback"] == "MATERIALIZED_WS2_W_SEG"


def test_ws2_drifted_archive_is_not_launchable(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["ws2_receipt_glob"] = "ws2/**/*.json"
    archive = _write(tmp_path / "ws2" / "wseg.zip", b"materialized")
    ws2 = {
        "execution_allowed": True,
        "score_claim": False,
        "warm_start_candidates": {
            "W_seg": {
                "archive_path": archive["path"],
                "archive_sha256": "0" * 64,
                "archive_bytes": archive["bytes"],
            }
        },
    }
    _write(tmp_path / "ws2" / "receipt.json", _json_bytes(ws2))
    receipt = build_preflight_receipt(config, repo_root=tmp_path)
    row = receipt["base_curves"]["ws2"]["receipts"][0]
    assert row["launchable"] is False
    assert "sha256 drifted" in row["archive_validation_error"]
