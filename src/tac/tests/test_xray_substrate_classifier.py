# SPDX-License-Identifier: MIT
"""Tests for ``tools/xray_substrate_classifier.py``.

Per CLAUDE.md "Recursive adversarial review protocol" + the substrate-classifier
contract this module covers:

* substrate-rule firing order (more-specific rules win);
* magic-byte signature detection at member offset 0;
* per-section sha256 + entropy estimate computation;
* deterministic-bytes guarantee (same archive → same JSON);
* refusal on corrupt archives + non-zip inputs + missing files;
* /tmp path refusal in --output-dir;
* CLI dry-run produces JSON to stdout without touching disk;
* CLI --refuse-unclassifiable exits 3 on unknown substrates;
* CLI --output-dir creates the manifest path.
"""

from __future__ import annotations

import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tac.packet_compiler.cooperative_receiver_grammars import (
    COOPERATIVE_RECEIVER_PACKET_GRAMMARS,
)
from tools.xray_substrate_classifier import (
    _SECTION_MAGIC_SIGNATURES,
    _SUBSTRATE_CLASSES,
    XraySubstrateClassifierError,
    _detect_magic,
    _shannon_entropy_bits_per_byte,
    _validate_output_dir,
    classify_archive,
    main,
    parse_args,
)

# ── Synthetic-archive helpers ───────────────────────────────────────────────


def _make_dense_streams_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("dense.bin", b"MDS1" + struct.pack("<BB", 1, 1) + b"\x00" * 32)
    return path


def _make_magic_codec_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "primitive.bin",
            b"MAGC" + struct.pack("<BB", 0xF0, 1) + b"\x00" * 16,
        )
    return path


def _make_single_magic_archive(path: Path, magic: bytes) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("0.bin", magic + b"\x01" + b"\x00" * 32)
    return path


def _make_pr106_r2_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("decoder.bin", b"\x00" * 256)
        zf.writestr("sidecar.bin", b"\xfe" + b"\x01" * 50)
    return path


def _make_unknown_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("random.dat", b"random random random " * 50)
    return path


# ── Smoke: magic detection ──────────────────────────────────────────────────


class TestMagicDetection:
    def test_detect_dense_streams_magic(self):
        assert _detect_magic(b"MDS1\x01\x00") == "magic_codec_dense_streams"

    def test_detect_magic_codec_envelope(self):
        assert _detect_magic(b"MAGC\xf0\x01") == "magic_codec_envelope"

    def test_detect_pr106_sidecar(self):
        assert _detect_magic(b"\xfe\x01\x00\x00") == "pr106_sidecar_v1"

    def test_detect_pr91_qm0(self):
        assert _detect_magic(b"QM0\x00other") == "pr91_qm0_grammar"

    def test_detect_pr92_rmc1(self):
        assert _detect_magic(b"RMC1other") == "pr92_joint_stream_rmc1"

    def test_detect_pr65_pq12(self):
        assert _detect_magic(b"PQ12other") == "pr65_pq12_pose"

    def test_detect_cooperative_receiver_packet_magics(self):
        assert _detect_magic(b"DFL1\x01") == "renderer_payload_dfl1_native_v1"
        assert _detect_magic(b"TT5L\x01") == "time_traveler_l5_v1"
        assert _detect_magic(b"SBO1\x01") == "sabor_boundary_only_renderer_v1"
        assert _detect_magic(b"S2SB" + b"S2S1") == "s2sbs_byte_stuffing_archive_v1"
        assert _detect_magic(b"CMLR\x01") == "coord_mlp_residual_sidecar_v1"
        assert _detect_magic(b"DPW1\x01") == "driving_prior_world_model_v1"

    def test_detect_returns_none_for_unknown(self):
        assert _detect_magic(b"XXXX") is None

    def test_detect_handles_short_input(self):
        assert _detect_magic(b"") is None
        assert _detect_magic(b"M") is None

    def test_signatures_table_nonempty(self):
        assert len(_SECTION_MAGIC_SIGNATURES) > 5


# ── Smoke: entropy estimate ─────────────────────────────────────────────────


class TestEntropy:
    def test_zero_entropy_for_all_zeros(self):
        assert _shannon_entropy_bits_per_byte(b"\x00" * 256) == 0.0

    def test_high_entropy_for_random_like(self):
        # 256 unique bytes — Shannon entropy is exactly 8.0.
        data = bytes(range(256))
        assert abs(_shannon_entropy_bits_per_byte(data) - 8.0) < 1e-9

    def test_empty_bytes_zero_entropy(self):
        assert _shannon_entropy_bits_per_byte(b"") == 0.0

    def test_two_byte_alphabet_entropy_at_most_1(self):
        data = b"\x00\xff" * 128
        e = _shannon_entropy_bits_per_byte(data)
        assert 0.99 < e <= 1.0


# ── Substrate classification ────────────────────────────────────────────────


class TestSubstrateClassification:
    def test_dense_streams_archive_classified(self, tmp_path):
        path = _make_dense_streams_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        assert result.substrate_class == "magic_codec_dense_streams_packet"
        assert result.archive_version == "dense_streams_v1"
        assert result.substrate_class_confidence in ("medium", "high")
        assert any(
            "magic_codec_dense_streams" in s for s in result.classification_signals
        )

    def test_magic_codec_archive_classified(self, tmp_path):
        path = _make_magic_codec_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        assert result.substrate_class == "magic_codec_packet"
        assert result.archive_version == "magic_codec_v1"

    def test_pr106_r2_archive_classified(self, tmp_path):
        path = _make_pr106_r2_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        assert result.substrate_class in (
            "pr106_r2_sidecar",
            "pr106_r1_sidecar",
        )

    def test_unknown_archive_marked_unclassifiable(self, tmp_path):
        path = _make_unknown_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        assert result.substrate_class == "unknown_substrate_unclassifiable"
        assert "no_substrate_rule_matched" in result.ambiguity_blockers

    def test_dense_streams_wins_over_pr106(self, tmp_path):
        # Bundle with both MDS1 + sidecar — more-specific rule (dense streams)
        # wins because it appears first in _SUBSTRATE_RULES.
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("dense.bin", b"MDS1\x01\x00")
            zf.writestr("sidecar.bin", b"\xfe" + b"\x00" * 32)
        result = classify_archive(path)
        assert result.substrate_class == "magic_codec_dense_streams_packet"

    def test_substrate_classes_includes_unknown_sentinel(self):
        assert "unknown_substrate_unclassifiable" in _SUBSTRATE_CLASSES

    @pytest.mark.parametrize("grammar", COOPERATIVE_RECEIVER_PACKET_GRAMMARS)
    def test_cooperative_receiver_packet_archive_classified(
        self,
        tmp_path,
        grammar,
    ):
        path = _make_single_magic_archive(tmp_path / "a.zip", grammar.magic)
        result = classify_archive(path)
        assert result.substrate_class == grammar.substrate_class
        assert result.archive_version == grammar.archive_version
        assert result.substrate_class_confidence == "medium"


# ── Per-section info + manifest ─────────────────────────────────────────────


class TestSectionInfo:
    def test_sections_carry_sha256_and_entropy(self, tmp_path):
        path = _make_pr106_r2_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        assert len(result.sections) == 2
        for s in result.sections:
            assert len(s.sha256) == 64
            assert s.size_bytes > 0
            assert 0.0 <= s.entropy_estimate_bits_per_byte <= 8.0

    def test_parser_section_manifest_layout(self, tmp_path):
        path = _make_pr106_r2_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        manifest = result.parser_section_manifest
        assert isinstance(manifest, dict)
        for key in (
            "offsets",
            "lengths",
            "section_names",
            "section_sha256s",
            "section_magics",
            "entropy_estimates_bits_per_byte",
        ):
            assert key in manifest, f"missing key {key}"
        n = len(result.sections)
        assert len(manifest["offsets"]) == n
        assert len(manifest["lengths"]) == n
        assert len(manifest["section_names"]) == n
        assert len(manifest["section_sha256s"]) == n

    def test_offsets_are_monotonic_and_match_cumulative_sum(self, tmp_path):
        path = _make_pr106_r2_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        offsets = result.parser_section_manifest["offsets"]
        lengths = result.parser_section_manifest["lengths"]
        assert offsets[0] == 0
        for i in range(1, len(offsets)):
            assert offsets[i] == offsets[i - 1] + lengths[i - 1]


# ── Refusal paths ───────────────────────────────────────────────────────────


class TestRefusalPaths:
    def test_missing_archive_rejected(self, tmp_path):
        missing = tmp_path / "does_not_exist.zip"
        with pytest.raises(
            XraySubstrateClassifierError, match="not found"
        ):
            classify_archive(missing)

    def test_corrupt_archive_rejected(self, tmp_path):
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        with pytest.raises(
            XraySubstrateClassifierError, match="corrupt"
        ):
            classify_archive(bad)

    def test_directory_input_rejected(self, tmp_path):
        with pytest.raises(
            XraySubstrateClassifierError, match="not a file"
        ):
            classify_archive(tmp_path)

    def test_validate_output_dir_refuses_tmp(self, tmp_path):
        with pytest.raises(
            XraySubstrateClassifierError, match="must not be under /tmp"
        ):
            _validate_output_dir(Path("/tmp/foo"))

    def test_validate_output_dir_refuses_var_tmp(self, tmp_path):
        with pytest.raises(
            XraySubstrateClassifierError, match="must not be under /tmp"
        ):
            _validate_output_dir(Path("/var/tmp/foo"))

    def test_validate_output_dir_accepts_experiments_results(self, tmp_path):
        ok = tmp_path / "experiments" / "results" / "ok"
        ok.mkdir(parents=True)
        # Should not raise.
        _validate_output_dir(ok)


# ── CLI ────────────────────────────────────────────────────────────────────


class TestCLI:
    def test_parse_args_requires_archive(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_parse_args_minimal(self):
        ns = parse_args(["--archive", "/foo.zip", "--dry-run"])
        assert ns.archive == Path("/foo.zip")
        assert ns.dry_run is True

    def test_cli_dry_run_emits_json(self, tmp_path, capsys):
        path = _make_dense_streams_archive(tmp_path / "a.zip")
        rc = main(["--archive", str(path), "--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["substrate_class"] == "magic_codec_dense_streams_packet"

    def test_cli_requires_output_dir_unless_dry_run(self, tmp_path, capsys):
        path = _make_dense_streams_archive(tmp_path / "a.zip")
        rc = main(["--archive", str(path)])
        assert rc == 2

    def test_cli_writes_manifest_to_output_dir(self, tmp_path):
        path = _make_dense_streams_archive(tmp_path / "a.zip")
        out = tmp_path / "out"
        rc = main(
            ["--archive", str(path), "--output-dir", str(out)]
        )
        assert rc == 0
        manifest = out / "xray_substrate_classifier_manifest.json"
        assert manifest.exists()
        payload = json.loads(manifest.read_text())
        assert payload["substrate_class"] == "magic_codec_dense_streams_packet"

    def test_cli_refuse_unclassifiable_exits_3_on_unknown(self, tmp_path):
        path = _make_unknown_archive(tmp_path / "a.zip")
        rc = main(
            [
                "--archive",
                str(path),
                "--dry-run",
                "--refuse-unclassifiable",
            ]
        )
        assert rc == 3

    def test_cli_refuse_unclassifiable_exits_0_on_known(self, tmp_path):
        path = _make_dense_streams_archive(tmp_path / "a.zip")
        rc = main(
            [
                "--archive",
                str(path),
                "--dry-run",
                "--refuse-unclassifiable",
            ]
        )
        assert rc == 0


# ── Determinism ─────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_archive_same_manifest(self, tmp_path):
        # Build the SAME bytes twice into separate paths and verify the
        # manifest body (excluding archive_path + timestamp) matches.
        for name in ("a.zip", "b.zip"):
            with zipfile.ZipFile(tmp_path / name, "w") as zf:
                zf.writestr("dense.bin", b"MDS1\x01\x01" + b"\xab" * 32)
        r1 = classify_archive(tmp_path / "a.zip")
        r2 = classify_archive(tmp_path / "b.zip")
        # Same content → same archive_sha256.
        assert r1.archive_sha256 == r2.archive_sha256
        # Sections identical apart from name (the only metadata that may drift
        # between two parallel zips IS the member content, which we held equal).
        assert (
            r1.parser_section_manifest["section_sha256s"]
            == r2.parser_section_manifest["section_sha256s"]
        )
        assert r1.substrate_class == r2.substrate_class

    def test_target_substrate_hint_default(self, tmp_path):
        path = _make_dense_streams_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        # Catalog #100 byte-closure: the classifier is byte-grammar plumbing
        # and the substrate-hint is "any packetized archive with dense
        # residual", NOT a saturated-base claim.
        assert (
            result.target_substrate_hint
            == "any_packetized_archive_with_dense_residual"
        )

    def test_no_score_claim_field_in_result(self, tmp_path):
        path = _make_dense_streams_archive(tmp_path / "a.zip")
        result = classify_archive(path)
        # The result MUST NOT have score_claim / promotion_eligible /
        # ready_for_exact_eval_dispatch — that's per task contract.
        # ClassificationResult is the public surface; check via fields.
        fields = set(result.__dataclass_fields__.keys())
        assert "score_claim" not in fields
        assert "promotion_eligible" not in fields
        assert "ready_for_exact_eval_dispatch" not in fields
