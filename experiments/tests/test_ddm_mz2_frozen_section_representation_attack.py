from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from tac.payload_retention_gate import check_no_measure_and_discard_payload

mz2 = importlib.import_module("experiments.ddm_mz2_frozen_section_representation_attack")


def _synthetic_records():
    baseline, _, _, _ = mz2._book_modules()
    records = []
    for index, schema in enumerate(baseline.SEMANTIC_SCHEMA):
        if schema.is_fp16:
            raw = np.full(schema.shape, index / 32, dtype="<f2").tobytes()
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(schema.shape)
            records.append(baseline.TensorStorage(schema, "fp16", values, None, None, raw_fp16=raw))
            continue
        scales_f16 = np.full(schema.scale_count, 0.125 + index / 1024, dtype="<f2")
        scales = scales_f16.astype(np.float32)
        codes = np.zeros(schema.count, dtype=np.int8)
        codes[:: max(1, index + 1)] = (index % 7) + 1
        codes = codes.reshape(schema.shape)
        scale_shape = [1] * len(schema.shape)
        scale_shape[-1 if schema.name.endswith("embed.weight") else 0] = schema.scale_count
        values = codes.astype(np.float32) * scales.reshape(scale_shape)
        records.append(
            baseline.TensorStorage(
                schema,
                "w4",
                values,
                scales,
                codes,
                raw_scales=scales_f16.tobytes(),
            )
        )
    return tuple(records)


def test_exact_representations_restore_every_decoded_tensor():
    records = _synthetic_records()
    for strategy in ("dense", "hybrid_sparse", "hybrid_rowdict", "hybrid_all"):
        payload, rows = mz2.pack_exact(records, strategy)
        restored = mz2.unpack_exact(payload)
        assert mz2._same_state(records, restored)
        assert len(rows) == 16
        assert payload.startswith(mz2.MZ2E_MAGIC)


def test_unsigned_bit_packing_round_trip():
    for bits in range(1, 10):
        values = np.arange(min(1 << bits, 79), dtype=np.uint32)
        payload = mz2._pack_unsigned(values, bits)
        restored, remaining = mz2._unpack_unsigned(memoryview(payload), len(values), bits)
        assert np.array_equal(values, restored)
        assert len(remaining) == 0


def test_row_dictionary_wins_on_repeated_rows():
    records = _synthetic_records()
    record = next(item for item in records if item.schema.name == "coord_mix.weight")
    _, _, _, bits = mz2._book_modules()
    record.codes[:] = 0
    options = mz2._code_payloads(record, bits)
    assert len(options[mz2.MODE_ROWDICT]) < len(options[mz2.MODE_DENSE])


def test_mz2_python_files_pass_payload_retention_gate():
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_mz2_frozen_section_representation_attack.py",
            "experiments/tests/test_ddm_mz2_frozen_section_representation_attack.py",
        ),
    )
    assert findings == []
