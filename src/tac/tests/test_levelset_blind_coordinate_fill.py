from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tac.boundary_math.compact_shearlet_frame import compact_shearlet_feats
from tac.witness_dsl.basis_control import genuine_frame_compact_shearlet_config

_REPO = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO / "tools" / "levelset_byte_close_and_eval.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("levelset_byte_close_d21a", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


def _valid_receipt() -> dict[str, object]:
    return {
        "schema": "blind_coordinate_proof.v1",
        "blind_fraction": {
            "schema": "blind_coordinate_fraction.v1",
            "n_blind_px": 230_904,
            "retained_subgrid_hw": [768, 1024],
        },
        "bit_identity_through_R": {
            "schema": "blind_coordinate_bit_identity.v1",
            "n_pairs": 600,
            "all_bit_identical": True,
            "max_abs_diff_pose": 0.0,
            "max_abs_diff_seg": 0.0,
            "n_failures": 0,
            "failing_pairs": [],
        },
    }


def test_n600_zero_delta_receipt_is_required_and_hashed(
    tool: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "blind_coordinate_proof.json"
    path.write_text(json.dumps(_valid_receipt()))
    binding = tool.validate_blind_coordinate_n600_receipt(path)
    assert binding["n_pairs"] == 600
    assert binding["n_blind_px_per_frame"] == 230_904
    assert binding["delta_d_seg"] == 0.0
    assert binding["delta_d_pose"] == 0.0
    assert len(binding["proof_receipt_sha256"]) == 64


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("bit_identity_through_R", "n_pairs", 599),
        ("bit_identity_through_R", "all_bit_identical", False),
        ("bit_identity_through_R", "max_abs_diff_seg", 1.0),
        ("blind_fraction", "n_blind_px", 230_903),
    ],
)
def test_receipt_gate_refuses_non_n600_or_nonzero_delta(
    tool: ModuleType,
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    receipt = _valid_receipt()
    receipt[section][key] = value  # type: ignore[index]
    path = tmp_path / f"bad_{section}_{key}.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="zero-delta receipt gate failed"):
        tool.validate_blind_coordinate_n600_receipt(path)


def test_run_refuses_before_checkpoint_or_candidate_selection(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def _unexpected_load(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("checkpoint load must not happen before D21a receipt validation")

    monkeypatch.setattr(tool, "_load_levelset_ckpt", _unexpected_load)
    with pytest.raises(FileNotFoundError, match="requires an existing n600 proof receipt"):
        tool.run(
            tmp_path / "checkpoint",
            npz_name=None,
            max_pairs=None,
            fold_pose_sidecar=False,
            pose_sidecar_path=None,
            gt_cache=None,
            keep_packet=False,
            packet_dir=None,
            skip_parity=True,
            so_overrides={},
            blind_coordinate_fill=True,
            blind_coordinate_receipt=tmp_path / "absent.json",
        )
    assert not called


def test_oracle_consumes_canonical_apply_blind_fill(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((874, 1164, 3), dtype=np.uint8)
    sentinel = np.full_like(frame, 17)
    calls: list[tuple[tuple[int, ...], object]] = []

    def _fake_apply(value: np.ndarray, fill: object) -> np.ndarray:
        calls.append((value.shape, fill))
        return sentinel

    monkeypatch.setattr(tool, "apply_blind_fill", _fake_apply)
    got = tool._blind_coordinate_reference(frame, {"blind_coordinate_fill": {}})
    assert got is sentinel
    assert calls == [((874, 1164, 3), None)]


def test_generated_receiver_derives_exact_blind_geometry(tool: ModuleType) -> None:
    namespace: dict[str, object] = {"__name__": "d21a_inline_test"}
    exec(tool._INFLATE_PY, namespace)
    retained_rows = namespace["_blind_retained_axis"](874, 384)
    retained_cols = namespace["_blind_retained_axis"](1164, 512)
    assert len(retained_rows) == 768
    assert len(retained_cols) == 1024
    assert 874 * 1164 - len(retained_rows) * len(retained_cols) == 230_904


def test_generated_receiver_fill_is_exact_canonical_op_twin(tool: ModuleType) -> None:
    namespace: dict[str, object] = {"__name__": "d21a_inline_fill_test"}
    exec(tool._INFLATE_PY, namespace)
    yy, xx = np.indices((874, 1164))
    frame = np.stack(
        ((xx + yy) % 256, (2 * xx + yy) % 256, (xx + 3 * yy) % 256), axis=-1
    ).astype(np.uint8)
    canonical = tool.apply_blind_fill(frame, fill=None)
    inline = namespace["_blind_coordinate_fill"](frame)
    assert np.array_equal(inline, canonical)


def test_generated_compact_shearlet_is_numpy_authority_op_twin(tool: ModuleType) -> None:
    namespace: dict[str, object] = {"__name__": "compact_shearlet_inline_test"}
    exec(tool._INFLATE_PY, namespace)
    coords = np.asarray(
        [(-1.0, -1.0), (-0.5, 0.25), (0.0, 0.0), (0.75, -0.2), (1.0, 1.0)],
        dtype=np.float32,
    )
    config = genuine_frame_compact_shearlet_config()
    expected = compact_shearlet_feats(coords, config)
    got = namespace["_compact_shearlet_feats"](coords, asdict(config))
    assert np.array_equal(got, expected)
