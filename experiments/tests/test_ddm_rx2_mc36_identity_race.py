from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from tac.payload_retention_gate import check_no_measure_and_discard_payload

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments/ddm_rx2_mc36_identity_race.py"


def _load():
    spec = importlib.util.spec_from_file_location("ddm_rx2_mc36_identity_race", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signed_int6_table_round_trips_through_shipping_parser() -> None:
    rx2 = _load()
    codes = np.arange(rx2.TABLE_STATES * rx2.CLASSES, dtype=np.int16).reshape(rx2.TABLE_STATES, rx2.CLASSES)
    codes = ((codes % 64) - 32).astype(np.int8)
    payload = rx2.serialize_table(codes, 0.125)
    assert len(payload) == rx2.TABLE_FULL_BYTES
    table = rx2.cp.load_runtime(rx2.BASE_RUNTIME)
    del table
    parsed = rx2.importlib.import_module("runtime.residual_archive")._decode_fixed_table(payload)
    assert np.array_equal(parsed.codes, codes)
    assert parsed.scale == 0.125


def test_stratified_development_frames_are_seeded_nonprefix() -> None:
    rx2 = _load()
    first = rx2._stratified_frames()
    second = rx2._stratified_frames()
    assert np.array_equal(first, second)
    assert len(first) == 120
    assert len(np.unique(first)) == 120
    assert first.tolist() != list(range(120))
    assert all(5 * index <= frame < 5 * (index + 1) for index, frame in enumerate(first))


def test_table_quantizer_uses_full_int6_domain_and_fp16_scale() -> None:
    rx2 = _load()
    ratio = np.linspace(-3.0, 4.0, rx2.TABLE_STATES * rx2.CLASSES).reshape(rx2.TABLE_STATES, rx2.CLASSES)
    codes, scale = rx2._candidate_table(ratio, shrink=1.25, clip_scale=0.75)
    assert codes.dtype == np.int8
    assert codes.shape == (25, 5)
    assert int(codes.min()) >= -32 and int(codes.max()) <= 31
    assert float(np.asarray([scale], dtype="<f2")[0]) == scale


def test_parser_exposes_only_checkpointed_race_stages() -> None:
    rx2 = _load()
    args = rx2.parser().parse_args(["materialize", "--variant", "neutral"])
    assert args.variant == "neutral"
    assert args.start_frame == 0 and args.end_frame == 600


def test_rx2_identity_race_passes_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=REPO,
        strict=False,
        roots=(
            "experiments/ddm_rx2_mc36_identity_race.py",
            "experiments/tests/test_ddm_rx2_mc36_identity_race.py",
        ),
    )
    assert findings == []
