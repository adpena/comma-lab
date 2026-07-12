"""Tests for the durable FreSh fixed-quality measurement CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools/measure_witness_fixed_quality.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("measure_witness_fixed_quality", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_arm(
    path: Path,
    rows: list[dict],
    *,
    mode: str,
    candidates: int,
    init_seconds: float,
    n_pairs: int = 8,
    epochs: int = 2,
    matched_config: dict | None = None,
) -> None:
    path.mkdir(parents=True)
    (path / "run.log").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    custody = {
        "git_sha": "a" * 40,
        "git_dirty": False,
        "upstream_snapshot_sha256": "b" * 64,
        "seed": 123,
    }
    normalized = matched_config or {
        "epochs": epochs,
        "max_pairs": n_pairs,
        "seed": 123,
    }
    config_sha = _canonical_sha(normalized)
    sampled = [0, n_pairs - 1] if n_pairs > 1 else [0]
    selection = {
        "schema": "tac.witness_init.fresh_runtime.v1",
        "claim_scope": "init_time_spectral_selection_not_contest_score",
        "provenance": {
            **custody,
            "axis": "macOS-MLX training-gradient init telemetry",
            "mlx_device": "gpu",
            "candidate_count": candidates,
            "mode": mode,
            "selection_surface": "thin_lane_boundary_residual",
            "matched_config": normalized,
            "matched_config_sha256": config_sha,
            "target_authority_sha256": "c" * 64,
            "config": {
                "freq_across": 32.0,
                "n_dir_freqs": 8,
                "reference_freq_along": 8.0,
                "tangent_deficit": 3.2,
                "bias_grid": [0.0] if mode == "control" else [0.0, 0.1],
                "render_aa": "none",
            },
        },
        "result": {
            "requested_sample_count": len(sampled),
            "total_target_pairs": n_pairs,
            "sampled_pair_indices": sampled,
            "initialization_draws": 1,
            "init_scorer_forward_calls": candidates,
            "init_scorer_pair_equivalents": candidates,
            "ordered_candidates": [
                {"freq_along": float(index + 1), "bias_k": 0.0}
                for index in range(candidates)
            ],
            "targets": [
                {
                    "pair_index": index,
                    "shape": [384, 512],
                    "label_sha256": f"{index + 1:064x}",
                    "boundary_sha256": f"{index + 11:064x}",
                    "spectral_weight_sha256": f"{index + 21:064x}",
                }
                for index in sampled
            ],
        },
    }
    selection_path = path / "fresh_init_receipt.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    committed = {
        "schema": "tac.witness_init.fresh_committed_state.v1",
        "claim_scope": "post_init_spectral_telemetry_not_contest_score",
        "selection_receipt_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "provenance": {
            **custody,
            "mode": mode,
            "matched_config": normalized,
            "matched_config_sha256": config_sha,
            "target_authority_sha256": "c" * 64,
            "total_init_seconds_to_epoch0": init_seconds,
        },
        "result": {
            "total_init_scorer_forward_calls": candidates + 1,
            "total_init_scorer_pair_equivalents": candidates + 1,
        },
    }
    (path / "fresh_init_post_structured_receipt.json").write_text(
        json.dumps(committed), encoding="utf-8"
    )
    (path / "result.json").write_text(
        json.dumps(
            {
                "utc": "2026-07-12T00:00:00Z",
                "n_pairs": n_pairs,
                "epochs": epochs,
                "final_epoch": epochs,
                "provenance": custody,
                "render_hw": [384, 512],
                "activation": "hosc",
                "fresh_init": {"control": mode == "control"},
                "history": [],
                "checkpoint": str(path / "checkpoint.npz"),
                "stage_checkpoints": [],
                "best": None,
            }
        ),
        encoding="utf-8",
    )


def test_init_overhead_can_erase_epoch_scorer_and_wall_benefit(tmp_path: Path) -> None:
    tool = _load_tool()
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    _write_arm(
        baseline,
        [
            {"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"},
            {"stage": "verdict", "epoch": 1, "d_seg": 0.095, "ts": "2026-07-12T00:00:04Z"},
            {"stage": "verdict", "epoch": 2, "d_seg": 0.089, "ts": "2026-07-12T00:00:08Z"},
        ],
        mode="control",
        candidates=1,
        init_seconds=0.25,
    )
    _write_arm(
        treatment,
        [
            {"stage": "verdict", "epoch": 0, "d_seg": 0.11, "ts": "2026-07-12T00:01:00Z"},
            {"stage": "verdict", "epoch": 1, "d_seg": 0.089, "ts": "2026-07-12T00:01:05Z"},
        ],
        mode="select",
        candidates=93,
        init_seconds=5.0,
    )

    payload = tool.build_measurement(
        baseline,
        treatment,
        fixed_epoch_budget=2,
        scorer_pairs_per_epoch=8,
    )

    comparison = payload["comparison"]
    assert comparison["epoch_reduction"] == 1
    assert comparison["total_scorer_pair_equivalent_reduction"] == -84
    assert comparison["wall_seconds_to_threshold_reduction"] == pytest.approx(-1.75)
    assert comparison["baseline"]["total_wall_seconds_to_threshold"] == pytest.approx(8.25)
    assert comparison["treatment"]["total_wall_seconds_to_threshold"] == pytest.approx(10.0)
    assert payload["treatment"]["init_scorer_forward_calls"] == 94
    assert payload["measurement_config"]["wall_clock_source"].startswith("verdict_row_ts")


def test_history_must_start_at_epoch_zero(tmp_path: Path) -> None:
    tool = _load_tool()
    log = tmp_path / "run.log"
    log.write_text(
        json.dumps(
            {"stage": "verdict", "epoch": 1, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must start at epoch 0"):
        tool.parse_verdict_history(log)


def test_verdict_epochs_require_exact_nonbool_integers(tmp_path: Path) -> None:
    tool = _load_tool()
    log = tmp_path / "run.log"
    log.write_text(
        json.dumps(
            {"stage": "verdict", "epoch": 0.0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid epoch"):
        tool.parse_verdict_history(log)


def test_noncanonical_or_unlinked_receipts_fail_closed(tmp_path: Path) -> None:
    tool = _load_tool()
    arm = tmp_path / "arm"
    _write_arm(
        arm,
        [{"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}],
        mode="control",
        candidates=1,
        init_seconds=0.1,
        epochs=0,
    )
    post = json.loads((arm / "fresh_init_post_structured_receipt.json").read_text())
    post["selection_receipt_sha256"] = "0" * 64
    (arm / "fresh_init_post_structured_receipt.json").write_text(json.dumps(post))

    with pytest.raises(ValueError, match="does not link"):
        tool.read_init_accounting(arm, expected_mode="control")


@pytest.mark.parametrize(
    ("receipt_name", "field", "value", "match"),
    [
        ("fresh_init_receipt.json", "schema", "wrong", "non-canonical schema"),
        ("fresh_init_receipt.json", "claim_scope", "wrong", "non-canonical claim_scope"),
        (
            "fresh_init_post_structured_receipt.json",
            "schema",
            "wrong",
            "non-canonical schema",
        ),
        (
            "fresh_init_post_structured_receipt.json",
            "claim_scope",
            "wrong",
            "non-canonical claim_scope",
        ),
    ],
)
def test_receipt_schema_and_scope_are_canonical(
    tmp_path: Path, receipt_name: str, field: str, value: str, match: str
) -> None:
    tool = _load_tool()
    arm = tmp_path / "arm"
    _write_arm(
        arm,
        [{"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}],
        mode="control",
        candidates=1,
        init_seconds=0.1,
        epochs=0,
    )
    receipt = arm / receipt_name
    payload = json.loads(receipt.read_text())
    payload[field] = value
    receipt.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        tool.read_init_accounting(arm, expected_mode="control")


def test_control_and_select_modes_are_not_interchangeable(tmp_path: Path) -> None:
    tool = _load_tool()
    arm = tmp_path / "arm"
    _write_arm(
        arm,
        [{"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}],
        mode="control",
        candidates=1,
        init_seconds=0.1,
        epochs=0,
    )

    with pytest.raises(ValueError, match="mode must be 'select'"):
        tool.read_init_accounting(arm, expected_mode="select")


@pytest.mark.parametrize("bad_count", [True, 1.0, "1"])
def test_receipt_counts_require_exact_nonbool_integers(
    tmp_path: Path, bad_count: object
) -> None:
    tool = _load_tool()
    arm = tmp_path / "arm"
    _write_arm(
        arm,
        [{"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}],
        mode="control",
        candidates=1,
        init_seconds=0.1,
        epochs=0,
    )
    selection_path = arm / "fresh_init_receipt.json"
    selection = json.loads(selection_path.read_text())
    selection["result"]["init_scorer_forward_calls"] = bad_count
    selection_path.write_text(json.dumps(selection))

    with pytest.raises(ValueError, match="exact non-negative integer"):
        tool.read_init_accounting(arm, expected_mode="control")


def test_post_structured_total_counts_are_exact_integers(tmp_path: Path) -> None:
    tool = _load_tool()
    arm = tmp_path / "arm"
    _write_arm(
        arm,
        [{"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}],
        mode="control",
        candidates=1,
        init_seconds=0.1,
        epochs=0,
    )
    receipt = arm / "fresh_init_post_structured_receipt.json"
    payload = json.loads(receipt.read_text())
    payload["result"]["total_init_scorer_pair_equivalents"] = 1.0
    receipt.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="exact non-negative integer"):
        tool.read_init_accounting(arm, expected_mode="control")


def test_cli_counts_must_match_result_authority(tmp_path: Path) -> None:
    tool = _load_tool()
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    rows = [
        {"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}
    ]
    _write_arm(baseline, rows, mode="control", candidates=1, init_seconds=0.1, epochs=0)
    _write_arm(treatment, rows, mode="select", candidates=2, init_seconds=0.2, epochs=0)

    with pytest.raises(ValueError, match="scorer_pairs_per_epoch disagrees"):
        tool.build_measurement(
            baseline,
            treatment,
            fixed_epoch_budget=0,
            scorer_pairs_per_epoch=7,
        )
    with pytest.raises(ValueError, match="fixed_epoch_budget disagrees"):
        tool.build_measurement(
            baseline,
            treatment,
            fixed_epoch_budget=1,
            scorer_pairs_per_epoch=8,
        )


def test_mismatched_full_config_custody_fails_closed(tmp_path: Path) -> None:
    tool = _load_tool()
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    rows = [
        {"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}
    ]
    _write_arm(baseline, rows, mode="control", candidates=1, init_seconds=0.1, epochs=0)
    _write_arm(
        treatment,
        rows,
        mode="select",
        candidates=2,
        init_seconds=0.2,
        epochs=0,
        matched_config={"epochs": 0, "max_pairs": 8, "seed": 999},
    )

    with pytest.raises(ValueError, match="matched_config"):
        tool.build_measurement(
            baseline,
            treatment,
            fixed_epoch_budget=0,
            scorer_pairs_per_epoch=8,
        )


def test_main_writes_exact_blocker_when_run_receipt_is_missing(tmp_path: Path) -> None:
    tool = _load_tool()
    output = tmp_path / "blocker.json"
    rc = tool.main(
        [
            "--baseline-run-dir",
            str(tmp_path / "missing_control"),
            "--treatment-run-dir",
            str(tmp_path / "missing_treatment"),
            "--fixed-epoch-budget",
            "50",
            "--scorer-pairs-per-epoch",
            "8",
            "--output",
            str(output),
        ]
    )

    assert rc == 2
    payload = json.loads(output.read_text())
    assert payload["schema"] == tool.BLOCKER_SCHEMA
    assert payload["claim_scope"] == "measurement_blocker_no_epochs_reduction_claim"
    assert "run log is missing" in payload["error"]


def test_conflicting_duplicate_epoch_is_refused(tmp_path: Path) -> None:
    tool = _load_tool()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            (
                json.dumps(
                    {"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}
                ),
                json.dumps(
                    {"stage": "verdict", "epoch": 0, "d_seg": 0.2, "ts": "2026-07-12T00:00:00Z"}
                ),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting verdict"):
        tool.parse_verdict_history(log)


def test_identical_duplicate_epoch_is_deduplicated(tmp_path: Path) -> None:
    tool = _load_tool()
    row = {"stage": "verdict", "epoch": 0, "d_seg": 0.1, "ts": "2026-07-12T00:00:00Z"}
    log = tmp_path / "run.log"
    log.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")

    assert tool.parse_verdict_history(log) == [
        {"epoch": 0, "d_seg": 0.1, "elapsed_seconds": 0.0}
    ]


def test_output_below_tmp_is_refused() -> None:
    tool = _load_tool()

    with pytest.raises(ValueError, match="must not be below /tmp"):
        tool._validate_durable_output(Path("/tmp/fresh_fixed_quality.json"))
