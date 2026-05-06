from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "pr86_hpac_replay_parity.py"
PR86_DIR = REPO / "experiments/results/public_pr86_intake_20260504_codex"


def _load_script():
    spec = importlib.util.spec_from_file_location("pr86_hpac_replay_parity_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_hpac_fixture(tmp_path: Path, *, token_blob: bytes | None = None) -> Path:
    inflate_py = tmp_path / "inflate.py"
    inflate_py.write_text(
        """
import torch
import torch.nn as nn
import constriction
import pyppmd


def _reconstruct_hpac_state_dict(packed_sd, device):
    return packed_sd


class HPACMini(nn.Module):
    def __init__(self, num_pairs=1, num_classes=5, P=1, delta=0, ch=1, d_film=1, use_spm=False):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, tokens, idx, prev_tokens):
        return torch.zeros(
            (tokens.shape[0], self.num_classes, tokens.shape[-2], tokens.shape[-1]),
            dtype=torch.float32,
            device=tokens.device,
        )
""",
        encoding="utf-8",
    )

    import constriction
    import numpy as np
    import pyppmd
    import torch

    if token_blob is None:
        probabilities = np.full(5, 0.2, dtype=np.float64)
        encoder = constriction.stream.queue.RangeEncoder()
        for symbol in [0, 1, 2, 3]:
            model = constriction.stream.model.Categorical(
                probabilities=probabilities,
                perfect=False,
            )
            encoder.encode(symbol, model)
        token_blob = encoder.get_compressed().tobytes()

    hpac_buf = io.BytesIO()
    torch.save({}, hpac_buf)
    hpac_ppmd = pyppmd.compress(hpac_buf.getvalue(), max_order=4, mem_size=16 << 20)
    meta_buf = io.BytesIO()
    torch.save(
        {
            "N": 1,
            "P": 1,
            "delta": 0,
            "ch": 1,
            "hpac_d_film": 1,
            "use_spm": False,
            "ppmd_max_order": 4,
        },
        meta_buf,
    )

    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("hpac.pt.ppmd", hpac_ppmd)
        zf.writestr("tokens.bin", token_blob)
        zf.writestr("meta.pt", meta_buf.getvalue())
    return archive


def test_static_contract_classifies_submitted_pr86_tokens_as_raw() -> None:
    script = _load_script()

    report = script.analyze_token_semantics(PR86_DIR)

    assert report["training_objective"] == "residual_tokens"
    assert report["archive_compute_gt_tokens_call_present"] is True
    assert report["archive_write_tokens_second_arg"] == "gt"
    assert report["inflate_reconstructs_residuals"] is False
    assert report["submitted_archive_token_encoding"] == "raw_tokens"


def test_inflate_device_contract_flags_cpu_comment_mismatch() -> None:
    script = _load_script()

    report = script.analyze_inflate_device_contract(PR86_DIR)

    assert report["comment_claims_hpac_cpu"] is True
    assert report["main_device_expression_selects_cuda_when_available"] is True
    assert report["decompress_tokens_hpac_device_arg"] == "device"
    assert report["comment_code_mismatch"] is True


def test_constriction_queue_self_test_documents_same_order_decode() -> None:
    script = _load_script()

    report = script.constriction_queue_self_test()

    assert report["available"] is True
    assert report["status"] == "passed"
    assert report["queue_api"] == "constriction.stream.queue.RangeEncoder/RangeDecoder"
    assert report["compressed_dtype"] == "uint32"
    assert report["compressed_word_count"] > 1
    assert report["same_order_roundtrip_ok"] is True
    assert report["decoded_symbols_prefix"] == report["encoded_symbols_prefix"]
    assert report["decoded_symbols_sha256"] == report["encoded_symbols_sha256"]


def test_decode_reencode_gate_passes_byte_exact_on_tiny_pinned_contract(tmp_path: Path) -> None:
    script = _load_script()
    archive = _write_tiny_hpac_fixture(tmp_path)

    report = script.decode_reencode_pr86_tokens(
        archive=archive,
        pr86_dir=tmp_path,
        device="cpu",
        height=2,
        width=2,
    )

    assert report["status"] == "passed"
    assert report["scope"] == "full_decode_reencode"
    assert report["score_claim"] is False
    assert report["planning_only"] is True
    assert report["full_stream"] is True
    assert report["byte_exact_reencode"] is True
    assert report["decoded_symbol_count"] == 4
    assert report["source_tokens_sha256"] == report["reencoded_tokens_sha256"]


def test_decode_reencode_gate_fails_closed_on_invalid_token_word_alignment(tmp_path: Path) -> None:
    script = _load_script()
    archive = _write_tiny_hpac_fixture(tmp_path, token_blob=b"abc")

    report = script.decode_reencode_pr86_tokens(
        archive=archive,
        pr86_dir=tmp_path,
        device="cpu",
        height=2,
        width=2,
    )

    assert report["status"] == "failed_closed"
    assert report["byte_exact_reencode"] is False
    assert report["score_claim"] is False
    assert report["planning_only"] is True
    assert report["failure_class"] == "invalid_tokens_bin_word_alignment"
    assert report["source_tokens_bytes"] == 3


def test_probability_contract_keeps_readme_claim_separate_from_code() -> None:
    script = _load_script()

    report = script.analyze_probability_contract(PR86_DIR)

    assert report["categorical_perfect_false_in_archive_code"] is True
    assert report["probability_clip_eps"] == "1e-7"
    assert report["readme_mentions_16384_grid"] is True
    assert report["explicit_16384_grid_in_archive_code"] is False


def test_skip_decode_diagnostic_is_non_promotable_and_lists_transfer_gates(tmp_path: Path) -> None:
    script = _load_script()
    out = tmp_path / "report.json"

    assert (
        script.main(
            [
                "--pr86-dir",
                str(PR86_DIR),
                "--archive",
                str(PR86_DIR / "archive.zip"),
                "--json-out",
                str(out),
                "--skip-decode",
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert payload["planning_only"] is True
    assert payload["promotable"] is False
    assert payload["conclusions"]["submitted_archive_token_encoding"] == "raw_tokens"
    gates = {row["gate"]: row for row in payload["required_gates_before_hpac_transfer_to_pr85"]}
    assert gates["pr86_own_stream_decode"]["required_before_pr85_transfer"] is True
    assert gates["byte_exact_reencode"]["status"] == "not_run"
    assert gates["pr85_transfer_parity"]["status"] == "not_run"
