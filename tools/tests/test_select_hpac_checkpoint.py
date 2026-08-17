"""Controls for scorer-free HPAC checkpoint selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO / "tools/select_hpac_checkpoint.py"
    spec = importlib.util.spec_from_file_location("select_hpac_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SELECT = _load_module()


def _row(epoch: int, joint_bytes: int | float, top1_error: float) -> str:
    return json.dumps(
        {
            "epoch": epoch,
            "phase": "discrete_qat",
            "estimated_joint_bytes": joint_bytes,
            "top1_error": top1_error,
        }
    )


def test_selects_joint_proxy_argmin_not_latest_or_byte_minimum(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                _row(482, 130000, 0.0020),
                _row(484, 129000, 0.0022),
                _row(486, 131000, 0.0019),
                _row(488, 128000, 0.0023),
            ]
        )
        + "\n"
    )
    checkpoints = tmp_path / "periodic"
    checkpoints.mkdir()
    for epoch in (482, 484, 486, 488):
        (checkpoints / f"epoch_{epoch:04d}.pt").write_bytes(f"cp-{epoch}".encode())
    result = SELECT.select(log, checkpoints)
    assert result["selected"]["epoch"] == 486
    assert result["selected"]["epoch"] != max(row["epoch"] for row in result["candidates"])
    assert result["selected"]["estimated_joint_bytes"] != min(
        row["estimated_joint_bytes"] for row in result["candidates"]
    )
    assert result["selection_formula"]["pose_term_included"] is False
    assert result["score_claim"] is False
    assert result["selected_checkpoint"]["sha256"] == SELECT._sha256(
        checkpoints / "epoch_0486.pt"
    )


def test_reports_rate_only_optimum_and_seg_proxy_cost(tmp_path: Path) -> None:
    """ddm_oa2: the receipt must expose what the zero-score-effect seg proxy costs.

    MC36 token labels are fixed and the HPAC codec is lossless, so d_seg is
    decode-invariant across checkpoints and only rate moves. The seg-proxy term
    can therefore only ever cost bytes; the receipt must say how many.
    """
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                _row(482, 130000, 0.0020),
                _row(484, 129000, 0.0022),
                _row(486, 131000, 0.0019),
                _row(488, 128000, 0.0023),
            ]
        )
        + "\n"
    )
    checkpoints = tmp_path / "periodic"
    checkpoints.mkdir()
    for epoch in (482, 484, 486, 488):
        (checkpoints / f"epoch_{epoch:04d}.pt").write_bytes(f"cp-{epoch}".encode())
    result = SELECT.select(log, checkpoints)
    rate_only = result["rate_only_optimum"]
    # The selection itself is unchanged: this field is observability only.
    assert result["selected"]["epoch"] == 486
    assert rate_only["epoch"] == 488
    assert rate_only["estimated_joint_bytes"] == 128000
    assert rate_only["differs_from_selected"] is True
    assert rate_only["seg_proxy_cost_bytes"] == 131000 - 128000
    assert rate_only["seg_proxy_cost_score"] == pytest.approx(
        3000 * SELECT.RATE_NUMERATOR / SELECT.RATE_DENOMINATOR
    )


def test_rate_only_optimum_reports_zero_cost_when_criteria_agree(tmp_path: Path) -> None:
    """When the proxy and the rate axis agree, the reported cost must be zero."""
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join([_row(482, 130000, 0.0020), _row(484, 128000, 0.0019)]) + "\n"
    )
    checkpoints = tmp_path / "periodic"
    checkpoints.mkdir()
    for epoch in (482, 484):
        (checkpoints / f"epoch_{epoch:04d}.pt").write_bytes(f"cp-{epoch}".encode())
    result = SELECT.select(log, checkpoints)
    rate_only = result["rate_only_optimum"]
    assert result["selected"]["epoch"] == 484
    assert rate_only["epoch"] == 484
    assert rate_only["differs_from_selected"] is False
    assert rate_only["seg_proxy_cost_bytes"] == 0
    assert rate_only["seg_proxy_cost_score"] == 0.0


def test_refuses_duplicate_epoch_telemetry(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(_row(482, 130000, 0.002) + "\n" + _row(482, 129000, 0.002) + "\n")
    checkpoints = tmp_path / "periodic"
    checkpoints.mkdir()
    (checkpoints / "epoch_0482.pt").write_bytes(b"cp")
    with pytest.raises(SELECT.SelectionError, match="duplicate selection telemetry"):
        SELECT.select(log, checkpoints)


def test_refuses_when_no_retained_checkpoint_joins_telemetry(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(_row(482, 130000, 0.002) + "\n")
    checkpoints = tmp_path / "periodic"
    checkpoints.mkdir()
    (checkpoints / "epoch_0484.pt").write_bytes(b"cp")
    with pytest.raises(SELECT.SelectionError, match="both telemetry"):
        SELECT.select(log, checkpoints)


def test_refuses_empty_periodic_checkpoint(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(_row(482, 130000, 0.002) + "\n")
    checkpoints = tmp_path / "periodic"
    checkpoints.mkdir()
    (checkpoints / "epoch_0482.pt").touch()
    with pytest.raises(SELECT.SelectionError, match="checkpoint is empty"):
        SELECT.select(log, checkpoints)


def test_refuses_nonintegral_joint_byte_telemetry(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(_row(482, 130000.5, 0.002) + "\n")
    checkpoints = tmp_path / "periodic"
    checkpoints.mkdir()
    (checkpoints / "epoch_0482.pt").write_bytes(b"cp")
    with pytest.raises(SELECT.SelectionError, match="nonintegral"):
        SELECT.select(log, checkpoints)
