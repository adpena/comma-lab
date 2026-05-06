from __future__ import annotations

import io

import numpy as np
import pytest
import torch

pyppmd = pytest.importorskip("pyppmd")
constriction = pytest.importorskip("constriction")

from tac.pr86_hpac_codec import (
    HPACMini,
    PPMD_MAX_ORDER,
    PPMD_MEM_SIZE,
    Pr86HpacReplayError,
    _categorical_from_probs,
    _group_masks,
    _normalize_probability_row,
    analyze_pr86_hpac_entropy_contract,
    collect_dependency_report,
    decode_gzip_torch_member,
    decode_meta_member,
    decode_tokens_hpac,
    load_source_artifact_summaries,
    load_hpac_model_from_ppmd,
    read_pr86_archive,
    run_pr86_hpac_probability_variant_matrix,
    run_pr86_hpac_replay,
    sha256_bytes,
)
from tac.pr91_hpm1_codec import (
    DEFAULT_PR91_ARCHIVE,
    EXPECTED_PR91_HPM1_HPAC_SHA256,
    EXPECTED_PR91_HPM1_TOKENS_SHA256,
    extract_pr91_hpm1_payload,
    load_hpm1_hpac_model,
)


def _zero_uniform_model(*, N: int = 2, H: int = 4, W: int = 4, P: int = 2, delta: int = 1) -> HPACMini:
    model = HPACMini(num_pairs=N, P=P, delta=delta, ch=4, d_film=2, use_spm=False)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        for buffer in model.buffers():
            if buffer.dtype.is_floating_point:
                buffer.zero_()
    return model.cpu().eval()


def _encode_uniform_tokens(tokens: np.ndarray, *, P: int, delta: int) -> bytes:
    N, H, W = tokens.shape
    masks = _group_masks(H, W, P=P, delta=delta, device="cpu")
    cat = constriction.stream.model.Categorical(
        probabilities=np.asarray([0.2] * 5, dtype=np.float64),
        perfect=False,
    )
    enc = constriction.stream.queue.RangeEncoder()
    for frame in range(N):
        for mask in masks:
            if mask is None:
                continue
            selected = tokens[frame][mask.cpu().numpy()]
            for symbol in selected:
                enc.encode(int(symbol), cat)
    return enc.get_compressed().astype(np.uint32).tobytes()


def test_hpacmini_decodes_uniform_synthetic_stream() -> None:
    raw_tokens = np.asarray(
        [
            [[0, 1, 2, 3], [4, 0, 1, 2], [3, 4, 0, 1], [2, 3, 4, 0]],
            [[4, 3, 2, 1], [0, 4, 3, 2], [1, 0, 4, 3], [2, 1, 0, 4]],
        ],
        dtype=np.uint8,
    )
    blob = _encode_uniform_tokens(raw_tokens, P=2, delta=1)
    model = _zero_uniform_model(N=2, P=2, delta=1)

    decoded = decode_tokens_hpac(
        model,
        blob,
        2,
        4,
        4,
        2,
        1,
        device="cpu",
        probability_variant="source_float64_perfect_false",
    )

    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, raw_tokens)


def test_hpac_decode_respects_max_frames_prefix() -> None:
    raw_tokens = np.asarray(
        [
            [[0, 1, 2, 3], [4, 0, 1, 2], [3, 4, 0, 1], [2, 3, 4, 0]],
            [[4, 3, 2, 1], [0, 4, 3, 2], [1, 0, 4, 3], [2, 1, 0, 4]],
        ],
        dtype=np.uint8,
    )
    blob = _encode_uniform_tokens(raw_tokens, P=2, delta=1)
    model = _zero_uniform_model(N=2, P=2, delta=1)

    decoded = decode_tokens_hpac(
        model,
        blob,
        2,
        4,
        4,
        2,
        1,
        device="cpu",
        max_frames=1,
    )

    assert decoded.shape == (1, 4, 4)
    assert np.array_equal(decoded[0], raw_tokens[0])


def test_hpac_probability_variant_contracts() -> None:
    row64 = _normalize_probability_row(
        [0.0, 0.25, 0.25, 0.25, 0.25],
        variant="source_float64_perfect_false",
    )
    row32 = _normalize_probability_row(
        [0.0, 0.25, 0.25, 0.25, 0.25],
        variant="source_float32_perfect_true",
    )

    assert row64.dtype == np.float64
    assert row32.dtype == np.float32
    assert row64[0] > 0.0
    assert np.isclose(float(row64.sum()), 1.0)
    assert np.isclose(float(row32.sum()), 1.0)
    assert _categorical_from_probs(row64) is not None


def test_hpac_helpers_fail_closed_on_bad_contracts() -> None:
    with pytest.raises(Pr86HpacReplayError, match="expected_single_num_classes_row"):
        _normalize_probability_row([0.5, 0.5])
    with pytest.raises(Pr86HpacReplayError, match="geometry_not_divisible_by_patch"):
        _group_masks(5, 4, P=2, delta=1)
    with pytest.raises(Pr86HpacReplayError, match="tokens_bin_not_uint32_aligned"):
        decode_tokens_hpac(
            _zero_uniform_model(),
            b"abc",
            1,
            4,
            4,
            2,
            1,
        )
    with pytest.raises(Pr86HpacReplayError, match="pr86_hpac_replay_is_cpu_only"):
        decode_tokens_hpac(
            _zero_uniform_model(),
            b"\x00\x00\x00\x00",
            1,
            4,
            4,
            2,
            1,
            device="cuda",
        )


def test_load_hpac_model_from_ppmd_roundtrips_synthetic_state_dict() -> None:
    source = _zero_uniform_model(N=2, P=2, delta=1)
    buf = io.BytesIO()
    torch.save(source.state_dict(), buf)
    packed = pyppmd.compress(
        buf.getvalue(),
        max_order=PPMD_MAX_ORDER,
        mem_size=PPMD_MEM_SIZE,
    )

    loaded = load_hpac_model_from_ppmd(
        packed,
        num_pairs=2,
        P=2,
        delta=1,
        ch=4,
        d_film=2,
        use_spm=False,
        device="cpu",
        strict=True,
    )

    assert isinstance(loaded, HPACMini)
    assert loaded.P == 2
    assert loaded.delta == 1

    loaded_from_config = load_hpac_model_from_ppmd(
        packed,
        {"N": 2, "P": 2, "delta": 1, "ch": 4, "hpac_d_film": 2, "use_spm": False},
        device="cpu",
        strict=True,
    )
    assert isinstance(loaded_from_config, HPACMini)
    assert loaded_from_config.P == 2


@pytest.mark.skipif(
    not DEFAULT_PR91_ARCHIVE.parents[1].joinpath("public_pr86_intake_20260504_codex/archive.zip").is_file(),
    reason="public PR86 archive not present",
)
def test_real_pr86_archive_contract_and_members_decode() -> None:
    pr86_archive = DEFAULT_PR91_ARCHIVE.parents[1] / "public_pr86_intake_20260504_codex" / "archive.zip"
    bundle = read_pr86_archive(pr86_archive)

    assert bundle.extra["archive_bytes"] == 207579
    assert set(bundle.members) == {"master.pt.gz", "slave.pt.gz", "hpac.pt.ppmd", "tokens.bin", "meta.pt"}
    assert len(bundle.members["tokens.bin"]) == 113900

    meta = decode_meta_member(bundle.members["meta.pt"])
    master = decode_gzip_torch_member(bundle.members["master.pt.gz"])

    assert meta["mode"] == "hpac"
    assert meta["N"] == 600
    assert meta["P"] == 32
    assert meta["use_spm"] is True
    assert isinstance(master, dict)
    assert "frame_embed.weight" in master


def test_dependency_and_source_artifact_reports_are_structured() -> None:
    dep = collect_dependency_report(strict=False)
    assert dep["observed"]["numpy"]
    assert dep["strict"] is False

    source_report = load_source_artifact_summaries({"missing": DEFAULT_PR91_ARCHIVE.parent / "missing.json"})
    assert source_report["status"] == "passed_source_artifact_inventory"
    assert source_report["artifacts"]["missing"]["exists"] is False


@pytest.mark.skipif(
    not DEFAULT_PR91_ARCHIVE.parents[1].joinpath("public_pr86_intake_20260504_codex/archive.zip").is_file(),
    reason="public PR86 archive not present",
)
def test_real_pr86_replay_reports_known_entropy_mismatch() -> None:
    pr86_archive = DEFAULT_PR91_ARCHIVE.parents[1] / "public_pr86_intake_20260504_codex" / "archive.zip"

    report = run_pr86_hpac_replay(
        pr86_archive,
        source_dir=None,
        max_frames=1,
        attempt_reencode=False,
    )

    assert report["score_claim"] is False
    assert report["dispatch_unlocked"] is False
    assert report["tokens_bin"]["sha256_matches_expected"] is True
    assert report["status"] == "failed_closed"
    assert report["failure_stage"] == "submitted_tokens_decode"
    assert report["failure_reason"] == "hpac_entropy_decode_contract_mismatch"
    assert report["failure_context"]["frame"] == 0
    assert report["failure_context"]["group"] == 10
    assert report["failure_context"]["symbol_in_group"] == 191


@pytest.mark.skipif(
    not DEFAULT_PR91_ARCHIVE.parents[1].joinpath("public_pr86_intake_20260504_codex/archive.zip").is_file(),
    reason="public PR86 archive not present",
)
def test_real_pr86_probability_matrix_is_local_only() -> None:
    pr86_archive = DEFAULT_PR91_ARCHIVE.parents[1] / "public_pr86_intake_20260504_codex" / "archive.zip"

    report = run_pr86_hpac_probability_variant_matrix(
        pr86_archive,
        variants=("source_float64_perfect_false",),
        source_dir=None,
        max_frames=1,
        attempt_reencode=False,
    )

    assert report["status"] == "failed_closed"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["passed_variants"] == []
    assert report["variant_results"][0]["failure_reason"] == "hpac_entropy_decode_contract_mismatch"
    assert report["variant_results"][0]["failure_context"]["group"] == 10


@pytest.mark.skipif(
    not DEFAULT_PR91_ARCHIVE.parents[1].joinpath("public_pr86_intake_20260504_codex/archive.zip").is_file(),
    reason="public PR86 archive not present",
)
def test_real_pr86_entropy_contract_analysis_classifies_auth_eval_failure() -> None:
    pr86_archive = DEFAULT_PR91_ARCHIVE.parents[1] / "public_pr86_intake_20260504_codex" / "archive.zip"

    report = analyze_pr86_hpac_entropy_contract(
        pr86_archive,
        variants=("source_float64_perfect_false",),
    )

    assert report["status"] == "not_locally_contest_validated_entropy_contract_mismatch"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["classification"]["local_exact_validated"] is False
    assert report["classification"]["entropy_failure_variants"] == ["source_float64_perfect_false"]
    assert "hpac_entropy_decode_contract_mismatch" in report["classification"]["auth_failure_kinds"]
    assert report["classification"]["contest_compliance_position"] == (
        "external_leaderboard_claim_not_promotable_until_auth_eval_passes"
    )


def test_pr86_group_masks_match_public_pr91_failure_geometry() -> None:
    masks = _group_masks(384, 512, P=32, delta=2, device="cpu")
    counts = [0 if mask is None else int(mask.sum().item()) for mask in masks]

    assert len(masks) == 94
    assert sum(counts[:10]) == 5760
    assert counts[10] == 1152


@pytest.mark.skipif(not DEFAULT_PR91_ARCHIVE.is_file(), reason="public PR91 archive not present")
def test_real_pr91_hpm1_model_load_matches_embedded_contract() -> None:
    payload = extract_pr91_hpm1_payload(DEFAULT_PR91_ARCHIVE)
    model = load_hpm1_hpac_model(payload, device="cpu")

    assert payload.config() == {
        "n_frames": 600,
        "height": 384,
        "width": 512,
        "predictor_count": 32,
        "delta": 2,
        "channels": 64,
        "use_spm": 1,
        "hpac_d_film": 8,
        "tokens_len": 116796,
        "hpac_len": 28243,
        "ppmd_order": 4,
    }
    assert sha256_bytes(payload.tokens) == EXPECTED_PR91_HPM1_TOKENS_SHA256
    assert sha256_bytes(payload.hpac) == EXPECTED_PR91_HPM1_HPAC_SHA256
    assert isinstance(model, HPACMini)
    assert model.P == 32
    assert model.ch == 64
    assert model.use_spm is True


@pytest.mark.skipif(not DEFAULT_PR91_ARCHIVE.is_file(), reason="public PR91 archive not present")
def test_real_pr91_hpm1_prefix_decode_reproduces_known_range_mismatch() -> None:
    payload = extract_pr91_hpm1_payload(DEFAULT_PR91_ARCHIVE)
    model = load_hpm1_hpac_model(payload, device="cpu")

    with pytest.raises(Pr86HpacReplayError) as excinfo:
        decode_tokens_hpac(
            model,
            payload.tokens,
            payload.n_frames,
            payload.height,
            payload.width,
            payload.predictor_count,
            payload.delta,
            device="cpu",
            probability_variant="source_float64_perfect_false",
            max_frames=1,
        )

    exc = excinfo.value
    assert exc.contract == "submitted_tokens_decode"
    assert exc.code == "hpac_entropy_decode_contract_mismatch"
    assert exc.fields["frame"] == 0
    assert exc.fields["group"] == 10
    assert exc.fields["symbol_in_group"] == 191
    assert exc.fields["decoded_symbol_count_before_failure"] == 5951
