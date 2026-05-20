# SPDX-License-Identifier: MIT
"""Tests for PR101 seeded-selector adapter feasibility."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.packet_compiler.pr101_fec6_packetir import FEC6_FIXED_K16_CODE_BITS
from tac.packet_compiler.pr101_seeded_selector_adapter import (
    build_seeded_selector_candidate,
    decode_residual_overrides,
    deterministic_seed,
    encode_residual_overrides,
    profile_seeded_selector_adapter,
    selector_order_entropy_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fec6_selector_payload(codes: tuple[int, ...]) -> bytes:
    bits = "".join(FEC6_FIXED_K16_CODE_BITS[code] for code in codes)
    payload = int(bits + "0" * ((8 - len(bits) % 8) % 8), 2).to_bytes(
        (len(bits) + 7) // 8,
        "big",
    )
    return b"FEC6" + len(codes).to_bytes(2, "little") + payload


def _fp11_payload(codes: tuple[int, ...]) -> bytes:
    source = b"source"
    selector = _fec6_selector_payload(codes)
    return (
        b"FP11"
        + len(source).to_bytes(4, "little")
        + source
        + len(selector).to_bytes(2, "little")
        + selector
    )


def _archive(path: Path, codes: tuple[int, ...]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, _fp11_payload(codes), compress_type=zipfile.ZIP_STORED)


def test_residual_overrides_roundtrip_with_bitmask() -> None:
    target = (0, 2, 2, 13, 0, 7, 7, 7)
    predicted = (0, 0, 2, 13, 1, 7, 0, 7)

    residual = encode_residual_overrides(target, predicted)
    decoded = decode_residual_overrides(
        predicted,
        residual.payload,
        encoding=residual.encoding,
        mismatch_count=residual.mismatch_count,
    )

    assert decoded == target
    assert residual.mismatch_count == 3
    assert residual.encoding == "bitmask_nibble_codes"


def test_seeded_selector_candidate_direct_seed_prior_is_tiny() -> None:
    seed = deterministic_seed(4, 0)
    target = build_seeded_selector_candidate(
        (0,) * 600,
        prediction_mode="seed_mod16",
        seed=seed,
        generator_kind="pcg64",
    ).predicted_codes

    candidate = build_seeded_selector_candidate(
        target,
        prediction_mode="seed_mod16",
        seed=seed,
        generator_kind="pcg64",
    )

    assert candidate.residual_encoding.mismatch_count == 0
    assert candidate.payload.startswith(b"F6SD")
    assert candidate.payload_bytes < len(_fec6_selector_payload(target))


def test_profile_is_fail_closed_when_seed_plus_residual_loses() -> None:
    codes = (0, 2, 13, 7) * 30
    profile = profile_seeded_selector_adapter(
        codes,
        fec6_selector_payload_bytes=len(_fec6_selector_payload(codes)),
        seed_lengths=(1, 2),
        search_seeds_per_length=4,
        target_saving_bytes=1,
    )

    assert profile["score_claim"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert "order_entropy" in profile
    assert profile["best_candidate"]["payload_bytes"] >= 0


def test_order_entropy_stats_surface_order_bounds() -> None:
    alternating = (0, 1) * 20
    stats = selector_order_entropy_stats(alternating, context_mods=(2,))

    assert stats["run_count"] == 40
    assert stats["pairmod_context_lower_bounds"][0]["context_mod"] == 2
    assert stats["pairmod_context_lower_bounds"][0][
        "zero_model_entropy_floor_bytes"
    ] == 0
    assert stats["first_order_transition_entropy_floor_bytes_plus_first_symbol"] <= stats[
        "global_entropy_floor_bytes"
    ] + 1


def test_probe_pr101_seeded_selector_adapter_cli(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.zip"
    output_json = tmp_path / "profile.json"
    output_md = tmp_path / "profile.md"
    _archive(archive_path, (0, 2, 13, 7) * 8)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "probe_pr101_seeded_selector_adapter.py"),
            "--archive",
            str(archive_path),
            "--seed-lengths",
            "1,2",
            "--search-seeds-per-length",
            "4",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    profile = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout["schema"] == "pr101_seeded_selector_adapter_profile_v1"
    assert profile["score_claim"] is False
    assert "PR101 Seeded Selector Adapter Profile" in output_md.read_text(
        encoding="utf-8"
    )
