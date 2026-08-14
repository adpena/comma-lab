#!/usr/bin/env python3
"""Verify retained F26R optimized, repeat, and scalar-twin full-field evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_TOKEN_SHA256 = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
EXPECTED_TOKEN_BYTES = 117_964_800
EXPECTED_BIT_POSITION = 921_964
EXPECTED_LOGIT_SHA256 = "617e9fcfc967c200f1ecc8bea93dd45a22f7af2a050092f982169b5f5e5a3523"
EXPECTED_CDF_SHA256 = "ba0d529b7eaf6e16da1f62fc1cc7ca43ccc1b989356a68b8d37988088cb7c7ff"
DEFAULT_WORK_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814")
DEFAULT_RUNG = "direct_context_delta_v1"
PARENT_REFERENCE_DIR = Path(
    "/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814/retained/profiles"
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _verify_full_run(receipt: dict) -> dict:
    report = receipt["native_report"]
    payload = Path(report["decoded_token_path"])
    assert receipt["complete"] and all(receipt["gates"].values())
    assert payload.stat().st_size == EXPECTED_TOKEN_BYTES
    assert _sha256(payload) == EXPECTED_TOKEN_SHA256 == report["decoded_token_sha256"]
    assert report["decoder_bit_position"] == EXPECTED_BIT_POSITION
    assert report["corrected_quantized_logit_sha256"] == EXPECTED_LOGIT_SHA256
    assert report["corrected_cdf_input_sha256"] == EXPECTED_CDF_SHA256
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--rung", default=DEFAULT_RUNG)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    rung = root / "rungs" / args.rung
    receipts = rung / "receipts"

    primary = _read(receipts / "native_run_v13_n600.json")
    repeat = _read(receipts / "native_run_optimized_repeat_n600.json")
    scalar = _read(receipts / "native_run_scalar_twin_n600.json")
    reports = [_verify_full_run(item) for item in (primary, repeat, scalar)]
    for field in (
        "decoded_token_sha256",
        "corrected_quantized_logit_sha256",
        "corrected_cdf_input_sha256",
        "decoder_bit_position",
        "model_manifest_sha256",
    ):
        assert len({report[field] for report in reports}) == 1

    reference_prefix = PARENT_REFERENCE_DIR / "reference_prefix_n32.u8"
    prefix_paths = [
        rung / "retained/native_tokens_v13_n32.u8",
        rung / "retained/native_tokens_optimized_repeat_n32.u8",
        rung / "retained/native_tokens_scalar_twin_n32.u8",
    ]
    reference_bytes = reference_prefix.read_bytes()
    assert all(path.read_bytes() == reference_bytes for path in prefix_paths)

    optimized_build = _read(receipts / "native_build_v13.json")
    scalar_build = _read(receipts / "native_build_scalar_twin.json")
    for build in (optimized_build, scalar_build):
        assert build["deterministic_binary"]
        assert len({item["sha256"] for item in build["repeat_builds"]}) == 1

    result = _read(root / "receipts/result.json")
    fire_order = _read(root / "SEALED_FIRE_ORDER.json")
    assert result["fire_ready"] and result["decode_engineering_residual"] is None
    assert result["derived_modal_projection"]["projected_total_seconds"] <= 1600.0
    assert fire_order["disposition"] == "SEALED_FIRE_ORDER"
    assert fire_order["archive"]["sha256"] == (
        "f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de"
    )
    assert any(
        item.startswith("inflated_outputs_volume_manifest.json")
        for item in fire_order["returned_bundle_required"]
    )

    print(
        json.dumps(
            {
                "full_field_primary_parity": True,
                "full_field_repeat_determinism": True,
                "full_field_scalar_twin_parity": True,
                "prefix_python_oracle_parity": True,
                "deterministic_repeat_builds": True,
                "sealed_fire_order": True,
                "decoded_token_sha256": EXPECTED_TOKEN_SHA256,
                "decoder_bit_position": EXPECTED_BIT_POSITION,
                "derived_modal_total_seconds": result["derived_modal_projection"][
                    "projected_total_seconds"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
