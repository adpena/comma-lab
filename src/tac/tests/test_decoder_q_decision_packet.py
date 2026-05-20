from __future__ import annotations

from tools.summarize_decoder_q_advisory_batch import _fixed_length_custody


def test_fixed_length_custody_accepts_waterbucket_top_level_manifest() -> None:
    row = {
        "mutation_manifest": {
            "fixed_length_runtime_compatible": True,
            "source_decoder_len": 162164,
            "mutated_decoder_len": 162164,
            "length_delta": 0,
            "archive_zip_bytes": 178517,
            "archive_bin_bytes": 178417,
            "mutated_decoder_sha256": "a" * 64,
            "mutation_records": [
                {
                    "mutation": {
                        "tensor_name": "rgb_1.weight",
                        "q_offset": 4,
                        "delta": 1,
                    }
                }
            ],
        }
    }

    custody = _fixed_length_custody(row)

    assert custody["passed"] is True
    assert custody["fixed_length_runtime_compatible"] is True
    assert custody["length_delta"] == 0
