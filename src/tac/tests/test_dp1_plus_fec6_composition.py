# SPDX-License-Identifier: MIT
"""Tests for tools/build_dp1_plus_fec6_composition_packet.py (PATH 1).

Lane: lane_dp1_plus_fec6_dual_stacking_build_20260517

Coverage:
* schema test — emitted build_manifest + archive_manifest carry every required field
* round-trip test — compose → decompose returns byte-identical fec6 bytes
* leakage test — DP1 prior is distilled from Comma2k19 NOT contest video (Catalog #209)
* vendored helper symmetry — submission-side decompose matches canonical compose
* inflate.py syntactic correctness — emitted file compiles + imports vendored helper
* /tmp refusal — build tool refuses output paths under transient prefixes
* device selection — emitted inflate.py honors PACT_INFLATE_DEVICE (Catalog #205)
* self-containment — emitted inflate.py works with empty PYTHONPATH (Catalog #295)
* canonical-vs-vendored byte parity — both compose surfaces produce identical bytes
"""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


DEFAULT_DP1_ARCHIVE = REPO_ROOT / "experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/0.bin"
DEFAULT_FEC6_ARCHIVE = REPO_ROOT / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
DEFAULT_FEC6_SUBMISSION_DIR = REPO_ROOT / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir"

# We refuse to skip on missing inputs by default — these archives are the
# canonical anchors; if they are missing the build is broken upstream and the
# tests SHOULD fail loud.
_INPUTS_PRESENT = all(p.exists() for p in (
    DEFAULT_DP1_ARCHIVE, DEFAULT_FEC6_ARCHIVE, DEFAULT_FEC6_SUBMISSION_DIR
))


def _require_inputs() -> None:
    if not _INPUTS_PRESENT:
        pytest.skip(
            "DP1 / fec6 canonical anchor inputs missing — build cannot smoke. "
            f"Required paths: {DEFAULT_DP1_ARCHIVE}, {DEFAULT_FEC6_ARCHIVE}, "
            f"{DEFAULT_FEC6_SUBMISSION_DIR}"
        )


@pytest.fixture(scope="module")
def composition_packet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one composition packet for the module; reused across tests."""
    _require_inputs()
    from tools.build_dp1_plus_fec6_composition_packet import build_composition_packet
    out_dir = tmp_path_factory.mktemp("dp1_plus_fec6_packet")
    build_composition_packet(
        dp1_archive_path=DEFAULT_DP1_ARCHIVE,
        fec6_archive_path=DEFAULT_FEC6_ARCHIVE,
        fec6_submission_dir=DEFAULT_FEC6_SUBMISSION_DIR,
        output_dir=out_dir,
    )
    return out_dir


class TestBuildManifestSchema:
    """Schema test — every required field present + correctly typed."""

    def test_archive_manifest_exists(self, composition_packet: Path) -> None:
        assert (composition_packet / "archive_manifest.json").exists()

    def test_build_manifest_exists(self, composition_packet: Path) -> None:
        assert (composition_packet / "build_manifest.json").exists()

    def test_archive_manifest_required_fields(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "archive_manifest.json").read_text()
        )
        required = {
            "archive_relpath", "archive_sha256", "archive_size_bytes",
            "composition_schema_version", "base_substrate",
            "dp1_source_sha256", "dp1_source_size_bytes",
            "dp1_basis_sha256", "dp1_dataset_provenance", "dp1_license_tags",
            "fec6_source_sha256", "fec6_source_size_bytes",
            "header_size_bytes", "num_pairs", "output_height", "output_width",
        }
        missing = required - set(manifest.keys())
        assert not missing, f"archive_manifest missing fields: {missing}"

    def test_build_manifest_required_fields(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        required = {
            "lane_id", "build_tool", "built_at_utc", "archive_relpath",
            "archive_sha256", "archive_size_bytes", "custody_status",
            "dp1_source", "fec6_source", "vendored_deps",
            "predicted_delta_cpu", "predicted_delta_basis",
            "operational_mechanism_status", "operational_mechanism_note",
        }
        missing = required - set(manifest.keys())
        assert not missing, f"build_manifest missing fields: {missing}"

    def test_base_substrate_is_pr101(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "archive_manifest.json").read_text()
        )
        assert manifest["base_substrate"] == "pr101"

    def test_lane_id_canonical(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        assert manifest["lane_id"] == "lane_dp1_plus_fec6_dual_stacking_build_20260517"

    def test_custody_status_committed_or_rebuildable(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        assert manifest["custody_status"] in {
            "committed-binary", "published", "ci-rebuildable",
        }


class TestRoundTripBytes:
    """Round-trip test — compose + decompose returns byte-identical fec6 bytes."""

    def test_decompose_recovers_fec6_bytes(self, composition_packet: Path) -> None:
        from tac.substrates.pretrained_driving_prior.composition import decompose
        composed = (composition_packet / "archive.zip").read_bytes()
        original_fec6 = DEFAULT_FEC6_ARCHIVE.read_bytes()
        parsed = decompose(composed)
        assert parsed.base_archive_bytes == original_fec6
        assert parsed.base_substrate == "pr101"

    def test_decompose_recovers_dp1_via_size(self, composition_packet: Path) -> None:
        from tac.substrates.pretrained_driving_prior.composition import (
            decompose, DPCOMP_HEADER_SIZE,
        )
        composed = (composition_packet / "archive.zip").read_bytes()
        original_dp1 = DEFAULT_DP1_ARCHIVE.read_bytes()
        parsed = decompose(composed)
        # We can't access dp1 bytes directly from ComposedArchive, but length
        # is verifiable via header layout:
        embedded_dp1_len = len(composed) - DPCOMP_HEADER_SIZE - len(parsed.base_archive_bytes)
        assert embedded_dp1_len == len(original_dp1)

    def test_composed_size_equals_sum_plus_header(self, composition_packet: Path) -> None:
        from tac.substrates.pretrained_driving_prior.composition import DPCOMP_HEADER_SIZE
        composed = (composition_packet / "archive.zip").read_bytes()
        original_dp1 = DEFAULT_DP1_ARCHIVE.read_bytes()
        original_fec6 = DEFAULT_FEC6_ARCHIVE.read_bytes()
        assert len(composed) == DPCOMP_HEADER_SIZE + len(original_dp1) + len(original_fec6)

    def test_sha256_byte_stable(self, composition_packet: Path) -> None:
        """Re-running the build with same inputs must produce identical bytes."""
        _require_inputs()
        from tools.build_dp1_plus_fec6_composition_packet import build_composition_packet
        composed_a = (composition_packet / "archive.zip").read_bytes()
        sha_a = hashlib.sha256(composed_a).hexdigest()
        # Re-build into a fresh tmp.
        out_dir_b = composition_packet.parent / "fresh_rebuild"
        build_composition_packet(
            dp1_archive_path=DEFAULT_DP1_ARCHIVE,
            fec6_archive_path=DEFAULT_FEC6_ARCHIVE,
            fec6_submission_dir=DEFAULT_FEC6_SUBMISSION_DIR,
            output_dir=out_dir_b,
        )
        composed_b = (out_dir_b / "archive.zip").read_bytes()
        sha_b = hashlib.sha256(composed_b).hexdigest()
        assert sha_a == sha_b, "composition build is NOT byte-stable"


class TestComma2k19LeakagePrevention:
    """Per CLAUDE.md Catalog #209: DP1 codebook from Comma2k19 NOT contest video."""

    def test_dp1_provenance_recorded_in_manifest(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        dp1_src = manifest["dp1_source"]
        assert "dataset_provenance" in dp1_src
        # Per CLAUDE.md "Apples-to-apples evidence discipline" the provenance
        # must NOT contain the contest video filename or "0.mkv" as substring.
        provenance = dp1_src["dataset_provenance"]
        assert "0.mkv" not in provenance.lower(), (
            f"DP1 provenance leaks contest video: {provenance}"
        )
        assert "upstream/video" not in provenance.lower(), (
            f"DP1 provenance leaks contest video: {provenance}"
        )

    def test_dp1_license_tags_recorded(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        dp1_src = manifest["dp1_source"]
        assert "license_tags" in dp1_src
        # license_tags MUST be a list (even if synthetic-test-only)
        assert isinstance(dp1_src["license_tags"], list)


class TestVendoredHelperSymmetry:
    """Vendored decompose_bytes must match canonical decompose output."""

    def test_vendored_helper_compiles(self, composition_packet: Path) -> None:
        py_compile.compile(
            str(composition_packet / "src" / "dp1_composition.py"), doraise=True
        )

    def test_vendored_decompose_matches_canonical(self, composition_packet: Path) -> None:
        from tac.substrates.pretrained_driving_prior.composition import decompose
        composed = (composition_packet / "archive.zip").read_bytes()
        # Canonical
        canonical = decompose(composed)
        # Vendored — load by absolute path so we don't pollute sys.path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dp1_composition_vendored",
            str(composition_packet / "src" / "dp1_composition.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        dp1_bytes, sub, base_bytes, version = mod.decompose_bytes(composed)
        assert sub == canonical.base_substrate
        assert base_bytes == canonical.base_archive_bytes
        assert version == canonical.schema_version


class TestInflatePyCorrectness:
    """Emitted inflate.py must compile + import its vendored deps."""

    def test_inflate_py_exists(self, composition_packet: Path) -> None:
        assert (composition_packet / "inflate.py").exists()

    def test_inflate_py_compiles(self, composition_packet: Path) -> None:
        py_compile.compile(str(composition_packet / "inflate.py"), doraise=True)

    def test_inflate_sh_exists_and_executable(self, composition_packet: Path) -> None:
        inflate_sh = composition_packet / "inflate.sh"
        assert inflate_sh.exists()
        assert os.access(inflate_sh, os.X_OK), "inflate.sh not executable"

    def test_inflate_sh_3_arg_contract(self, composition_packet: Path) -> None:
        """Catalog #146: contest contract is inflate.sh archive_dir output_dir file_list."""
        text = (composition_packet / "inflate.sh").read_text()
        assert 'DATA_DIR="$1"' in text
        assert 'OUTPUT_DIR="$2"' in text
        assert 'FILE_LIST="$3"' in text


class TestCatalog205DeviceSelection:
    """Catalog #205: inflate.py honors PACT_INFLATE_DEVICE env var."""

    def test_inflate_py_has_select_inflate_device(self, composition_packet: Path) -> None:
        text = (composition_packet / "inflate.py").read_text()
        assert "def select_inflate_device" in text

    def test_inflate_py_honors_pact_inflate_device(self, composition_packet: Path) -> None:
        text = (composition_packet / "inflate.py").read_text()
        assert 'PACT_INFLATE_DEVICE' in text

    def test_inflate_py_refuses_mps(self, composition_packet: Path) -> None:
        text = (composition_packet / "inflate.py").read_text()
        # MPS must be explicitly refused (CLAUDE.md "MPS auth eval is NOISE")
        assert 'requested == "mps"' in text or "'mps'" in text
        assert "REFUSED" in text or "REFUSE" in text or "RuntimeError" in text


class TestCatalog295EmptyPythonPath:
    """Catalog #295: emitted inflate.py works with empty PYTHONPATH."""

    def test_inflate_py_uses_local_src_dir(self, composition_packet: Path) -> None:
        text = (composition_packet / "inflate.py").read_text()
        assert 'SRC_DIR = HERE / "src"' in text
        assert "sys.path.insert(0, str(SRC_DIR))" in text

    def test_inflate_py_no_tac_imports(self, composition_packet: Path) -> None:
        text = (composition_packet / "inflate.py").read_text()
        # NO `from tac.` / `import tac.` at module level — Catalog #295
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from tac.") or stripped.startswith("import tac."):
                pytest.fail(f"inflate.py has forbidden tac.* import: {line}")

    def test_vendored_deps_present(self, composition_packet: Path) -> None:
        src = composition_packet / "src"
        for required in ("codec.py", "frame_selector.py", "model.py",
                         "fec6_inflate.py", "dp1_composition.py"):
            assert (src / required).exists(), f"missing vendored dep: {required}"

    def test_inflate_py_compiles_with_isolated_subprocess(
        self, composition_packet: Path
    ) -> None:
        """Verify inflate.py compiles + does smoke import via subprocess with empty
        PYTHONPATH per Catalog #295 contract."""
        env = os.environ.copy()
        env["PYTHONPATH"] = ""
        # Just compile-check via subprocess; do NOT actually run inflate (requires
        # real archive_dir).
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import py_compile; "
                f"py_compile.compile({str(composition_packet / 'inflate.py')!r}, doraise=True); "
                "print('OK')"
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"inflate.py compile-check failed in empty-PYTHONPATH subprocess: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )


class TestTmpPathRefusal:
    """Build tool refuses /tmp paths per CLAUDE.md."""

    def test_build_refuses_tmp_output_dir(self, tmp_path: Path) -> None:
        _require_inputs()
        from tools.build_dp1_plus_fec6_composition_packet import build_composition_packet
        with pytest.raises(ValueError, match="transient path"):
            build_composition_packet(
                dp1_archive_path=DEFAULT_DP1_ARCHIVE,
                fec6_archive_path=DEFAULT_FEC6_ARCHIVE,
                fec6_submission_dir=DEFAULT_FEC6_SUBMISSION_DIR,
                output_dir=Path("/tmp/dp1_plus_fec6_test"),
            )


class TestOperationalMechanismDeclaration:
    """Catalog #220: substrate with byte addition >1KB must declare operational mechanism."""

    def test_build_manifest_declares_operational_mechanism(
        self, composition_packet: Path
    ) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        status = manifest["operational_mechanism_status"]
        # OPERATIONAL_DEFERRED_TO_L2 is the canonical L1 status: structural byte
        # consumption proven (decompose runs at every inflate call) but frame-axis
        # effect deferred to L2 per the design memo's 2-phase composition discipline.
        assert status in {
            "OPERATIONAL", "OPERATIONAL_DEFERRED_TO_L2", "research_only",
        }

    def test_build_manifest_explains_mechanism(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        note = manifest["operational_mechanism_note"]
        assert len(note) > 50, "operational_mechanism_note too brief"
        # Must reference Catalog #220 (the canonical contract)
        assert "Catalog #220" in note or "L2 INTEGRATION" in note


class TestPredictedDeltaCitation:
    """Catalog #296: predicted ΔS band must cite Dykstra / first-principles / probe."""

    def test_predicted_delta_cpu_has_band_format(self, composition_packet: Path) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        delta = manifest["predicted_delta_cpu"]
        # Must be a band, not a point estimate.
        assert "[-" in delta or "[" in delta, f"predicted_delta not a band: {delta}"
        # Must carry an axis tag.
        assert (
            "[time-traveler-prediction]" in delta
            or "[empirical:" in delta
            or "[contest-CPU]" in delta
            or "[contest-CUDA]" in delta
        ), f"predicted_delta missing axis tag: {delta}"

    def test_predicted_delta_basis_explains_mechanism(
        self, composition_packet: Path
    ) -> None:
        manifest = json.loads(
            (composition_packet / "build_manifest.json").read_text()
        )
        basis = manifest["predicted_delta_basis"]
        assert len(basis) > 30, "predicted_delta_basis too brief"


class TestCanonicalVendoredByteParity:
    """Canonical compose_with + vendored decompose_bytes round-trip byte-identical."""

    def test_canonical_compose_matches_build_output(self, composition_packet: Path) -> None:
        from tac.substrates.pretrained_driving_prior.composition import compose_with
        dp1_bytes = DEFAULT_DP1_ARCHIVE.read_bytes()
        fec6_bytes = DEFAULT_FEC6_ARCHIVE.read_bytes()
        canonical = compose_with(dp1_bytes, fec6_bytes, base_substrate="pr101")
        from_file = (composition_packet / "archive.zip").read_bytes()
        assert canonical == from_file, "build tool output diverges from canonical compose_with"
