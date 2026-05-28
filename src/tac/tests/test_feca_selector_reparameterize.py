from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tac.optimizer.materializer_chain_harvest import (
    adapt_materializer_manifest_to_candidate,
)
from tac.packet_compiler.feca_selector_reparameterize import (
    FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
    FecaSelectorReparameterizationError,
    build_feca_selector_reparameterized_candidate,
    join_fp11_member,
    split_fp11_member,
)


def _write_no_positive_feca_source_submission(tmp_path: Path) -> Path:
    source = tmp_path / "source_submission"
    encoder = source / "encoder"
    encoder.mkdir(parents=True)
    (source / "inflate.py").write_text("# inflate runtime placeholder\n", encoding="utf-8")
    (source / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py \"$@\"\n", encoding="utf-8")
    (encoder / "build_pr101_frame_exploit_selector_packet_fec10_hybrid.py").write_text(
        """
ALPHA_DEFAULT = 2

class _BlendContextModel:
    SCALE = 1 << 14

_PRIOR_MODEL = None
_CTX1_MODELS = None
_CTX_BLEND_MODELS = None
_CTX2_ROW_SUMS = None

def _build_priors():
    return None, None, None, None

def encode_fec10_hybrid_adaptive_blend(codes, n_pairs):
    return b"FECa" + bytes([len(codes)]) + bytes(codes) + b"not-smaller"

def decode_fec10_hybrid_selector(payload):
    assert payload[:4] == b"FECa"
    count = payload[4]
    return list(payload[5:5 + count])
""".lstrip(),
        encoding="utf-8",
    )
    member = join_fp11_member(
        source_payload=b"SRC",
        selector_payload=b"FECa\x02\x00\x01",
        dqs1_tail=b"DQS1",
    )
    info = zipfile.ZipInfo("x")
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(source / "archive.zip", "w") as archive:
        archive.writestr(info, member)
    return source


def test_fp11_join_split_preserves_single_dqs1_tail() -> None:
    member = join_fp11_member(
        source_payload=b"source",
        selector_payload=b"FECa\x00",
        dqs1_tail=b"tail",
    )

    parts = split_fp11_member(member)

    assert parts["source_payload"] == b"source"
    assert parts["selector_payload"] == b"FECa\x00"
    assert parts["dqs1_tail"] == b"tail"
    assert member.endswith(b"tail")
    assert not member.endswith(b"tailtail")


def test_feca_selector_recode_can_emit_zero_delta_manifest_without_queue_failure(
    tmp_path: Path,
) -> None:
    source = _write_no_positive_feca_source_submission(tmp_path)

    manifest = build_feca_selector_reparameterized_candidate(
        source_submission_dir=source,
        output_dir=tmp_path / "candidate",
        codec_families=("fec10_adaptive_blend",),
        scales=(64,),
        alphas=(1,),
        allow_nonpositive_candidate=True,
    )

    assert manifest["schema"] == FECA_REPARAMETERIZATION_MANIFEST_SCHEMA
    assert manifest["byte_closed_candidate_emitted"] is True
    assert manifest["candidate_runtime_patched"] is False
    assert manifest["selected_payload"]["status"] == "zero_delta"
    assert manifest["selected_payload"]["savings_realized"] is False
    assert manifest["serialized_archive_delta"]["status"] == "zero_delta"
    assert manifest["serialized_archive_delta"]["realized_saved_bytes"] == 0
    assert manifest["source_archive_sha256"] == manifest["candidate_archive_sha256"]
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_feca_selector_manifest_is_harvestable_materializer_signal(
    tmp_path: Path,
) -> None:
    source = _write_no_positive_feca_source_submission(tmp_path)
    output_dir = tmp_path / "candidate"
    manifest = build_feca_selector_reparameterized_candidate(
        source_submission_dir=source,
        output_dir=output_dir,
        codec_families=("fec10_adaptive_blend",),
        scales=(64,),
        alphas=(1,),
        allow_nonpositive_candidate=True,
    )
    manifest_path = output_dir / "feca_selector_reparameterization_manifest.json"

    row = adapt_materializer_manifest_to_candidate(
        manifest,
        source_path=manifest_path,
        repo_root=tmp_path,
    )

    assert row["schema"] == FECA_REPARAMETERIZATION_MANIFEST_SCHEMA
    assert row["candidate_family"] == "selector_stream_context_recode"
    assert row["receiver_contract_satisfied"] is True
    assert row["runtime_adapter_ready"] is True
    assert row["candidate_runtime_tree_sha256"] == manifest["candidate_runtime_tree_sha256"]
    assert row["expected_runtime_tree_sha256"] == manifest["expected_runtime_tree_sha256"]
    assert row["rate_positive"] is False
    assert row["realized_saved_bytes"] == 0
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_feca_selector_recode_strict_mode_still_fails_on_no_positive(
    tmp_path: Path,
) -> None:
    source = _write_no_positive_feca_source_submission(tmp_path)

    with pytest.raises(FecaSelectorReparameterizationError, match="no rate-positive"):
        build_feca_selector_reparameterized_candidate(
            source_submission_dir=source,
            output_dir=tmp_path / "candidate",
            codec_families=("fec10_adaptive_blend",),
            scales=(64,),
            alphas=(1,),
        )
