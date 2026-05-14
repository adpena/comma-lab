# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "remote_lane_sjkl_c067.sh"
RUNBOOK = REPO_ROOT / ".omx/research/sjkl_c067_remote_dispatch_runbook_20260502_codex.md"


def _text() -> str:
    return SCRIPT.read_text()


def test_sjkl_c067_remote_script_uses_fail_closed_cuda_path() -> None:
    text = _text()
    assert "set -euo pipefail" in text
    assert "nvidia-smi" in text
    assert "torch.cuda.is_available()" in text
    assert "FATAL: torch.cuda.is_available() is false" in text
    assert "SJKL_FAST_CHIP_REGEX" in text
    assert "SJKL_MAX_BYTES" in text
    assert "sjkl.bin exceeds SJKL_MAX_BYTES cap" in text
    assert "--device cuda" in text
    assert "falling back to CPU" in text
    assert "ADVISORY only" in text
    assert "manifest device is not cuda" in text
    assert "--device cpu" not in text
    assert "--device mps" not in text


def test_sjkl_c067_remote_script_uses_real_sjkl_cli_flags() -> None:
    text = _text()
    for token in (
        "experiments/prepare_sjkl_pair_tensors.py",
        "SJKL_PREPARED_TENSOR_DIR",
        "renderer_target_slot_chw.pt",
        "gt_pairs_btchw.pt",
        "--renderer-output",
        "--target-frames",
        "--output-dir",
        "--anchor-pair-idx 0",
        "experiments/build_sjkl_residual.py",
        "--pair-tensor-manifest",
        "--rank",
        "--n-pairs",
        "--alpha-bits",
        "--basis-quant-bits",
        "--max-bytes",
    ):
        assert token in text
    assert "--output-json" not in text
    assert "--base" not in text


def test_sjkl_c067_remote_script_packs_charged_payload_deterministically() -> None:
    text = _text()
    assert "experiments/build_sjkl_c067_archive.py" in text
    assert "--source-archive" in text
    assert "--sjkl-bin" in text
    assert "--sjkl-member-name sjkl.bin" in text
    assert "--max-sjkl-bytes \"$MAX_SJKL_BYTES\"" in text
    assert "if int(sjkl.get(\"bytes\", -1)) > int(sjkl.get(\"max_bytes\", -1))" in text
    assert "sjkl_c067_archive_manifest.json" in text
    assert "output_logical_runtime_members" in text
    assert "sjkl.bin missing from output runtime members" in text
    assert "score_affecting_payload_charged_in_archive" in text
    assert "SJ-KL runtime-apply proof is absent" in text
    assert "zipfile.ZipFile" not in text
    assert 'members.append(("sjkl.bin", sjkl.read_bytes()))' not in text


def test_sjkl_c067_remote_script_delegates_exact_eval_to_canonical_wrapper() -> None:
    text = _text()
    assert "scripts/remote_archive_only_eval.sh" in text
    assert 'export ARCHIVE_PATH="$OUTPUT_ARCHIVE"' in text
    assert 'export REQUIRED_SOURCE_SHA256S="$SOURCE_SHA_LINES"' in text
    assert 'export SJKL_REQUIRE_APPLIED="${SJKL_REQUIRE_APPLIED:-1}"' in text
    assert 'SJKL_REQUIRE_APPLIED must remain 1 for SJ-KL eval dispatch' in text
    assert 'bash scripts/remote_archive_only_eval.sh' in text
    assert 'CONTEST_JSON="$LOG_DIR/contest_auth_eval.json"' in text
    assert "-u experiments/contest_auth_eval.py" not in text
    assert "--archive \"$OUTPUT_ARCHIVE\"" not in text


def test_sjkl_c067_runbook_records_claim_and_no_spend_boundary() -> None:
    text = RUNBOOK.read_text()
    assert "tools/claim_lane_dispatch.py claim" in text
    assert "--lane-id sjkl_c067" in text
    assert "Do not run the remote command until the main agent/operator approves spend" in text
    assert "score_claim=false" in text
    assert "contest_auth_eval.json" in text
    assert "scripts/lightning_repro_workspace.py" in text
    assert "bash scripts/remote_lane_sjkl_c067.sh" in text
