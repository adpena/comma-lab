"""Tests for ``tools/build_pr106_uniward_runtime_packet.py``.

Subagent BUILD-PR106-UNIWARD-RUNTIME landed this build tool to close the
CUDA-PRESTAGE Candidate 2 deploy-readiness gap (Tier-A advisory in
``.omx/research/tier_a_cuda_dispatch_packets_prestaged_20260508.md``).

These tests verify the source declares the required CLAUDE.md flags, the
forked submission_dir uses the FIX-CODEX-HIGH 3-arg auth-eval contract, the
build manifest schema is correct, the wire-format is byte-identical to PR106,
and the BUGCLASSES B3 custody verifier exists alongside the build tool.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.build_pr106_uniward_runtime_packet import (
    EVIDENCE_GRADE,
    REPO_ROOT,
    _repo_resolve,
    cpu_build_proxy_guard_fields,
)

BUILD_TOOL = REPO_ROOT / "tools" / "build_pr106_uniward_runtime_packet.py"
VERIFY_TOOL = REPO_ROOT / "tools" / "verify_pr106_uniward_runtime_packet_sha256.py"
CANONICAL_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "pr106_uniward_runtime_packet_20260508_codex_smoke"
)
CANONICAL_MANIFEST = CANONICAL_OUTPUT_DIR / "build_manifest.json"


def _read_tool_source() -> str:
    return BUILD_TOOL.read_text()


# ---------------------------------------------------------------------------
# Existing source-import smoke (codex landing — preserved)
# ---------------------------------------------------------------------------


def test_repo_resolve_makes_relative_output_paths_repo_absolute() -> None:
    path = _repo_resolve(Path("experiments/results/pr106_uniward_runtime_packet_test"))
    assert path.is_absolute()
    assert path == (
        REPO_ROOT / "experiments/results/pr106_uniward_runtime_packet_test"
    ).resolve()


def test_cpu_build_guard_fields_are_fail_closed_for_score_claims() -> None:
    fields = cpu_build_proxy_guard_fields()
    assert fields["evidence_grade"] == EVIDENCE_GRADE
    assert fields["score_claim"] is False
    assert fields["promotion_eligible"] is False
    assert fields["rank_or_kill_eligible"] is False
    assert fields["ready_for_exact_eval_dispatch"] is False
    assert fields["dispatch_attempted"] is False
    assert fields["cuda_eval_worth_testing"] is True
    assert fields["custody_status"] == "transient-allowed"
    assert "auth eval" in fields["custody_status_reason"]
    assert "exact_cuda_auth_eval_not_yet_harvested" in fields["dispatch_blockers"]


# ---------------------------------------------------------------------------
# Source-level static checks (BUILD-PR106-UNIWARD-RUNTIME)
# ---------------------------------------------------------------------------


def test_build_tool_reuses_canonical_primitives() -> None:
    """Per CANONICALIZE-OSS: tool must NOT duplicate UniwardWeightedAllocator
    or per_tensor_codecs helpers. It must import them from tac.optimization
    and tac.codec.cost_curves."""
    src = _read_tool_source()
    assert "from tac.optimization.lagrangian_per_tensor_allocation" in src
    assert "UniwardWeightedAllocator" in src
    assert "from tac.codec.cost_curves" in src
    assert "DEFAULT_K_RANGE" in src
    assert "precompute_per_tensor_K_curves" in src
    assert "from tac.hnerv_lowlevel_packer" in src
    assert "parse_ff_packed_brotli_hnerv" in src
    assert "from tac.hnerv_decoder_recode" in src
    assert "PACKED_STATE_SCHEMA" in src
    # No new local copy of the allocator class — must be imported.
    assert "class UniwardWeightedAllocator" not in src


def test_build_tool_inflate_sh_uses_self_contained_3_arg_contract() -> None:
    """FIX-CODEX-HIGH commit c83eff00 mandates the contest auth-eval 3-arg
    contract. PR106's original wrapper is only location-correct when it lives
    under submissions/<name>, so the packet must synthesize a self-contained
    shell wrapper that calls the byte-identical PR106 inflate.py by path."""
    src = _read_tool_source()
    assert 'src_inflate_sh = pr106_source_dir / "inflate.sh"' in src
    assert "SELF_CONTAINED_INFLATE_SH" in src
    assert '"$PYTHON_BIN" "$HERE/inflate.py" "$SRC" "$DST"' in src
    assert "submissions.${SUB_NAME}.inflate" not in src
    assert "PYTHON_INFLATE" not in src
    assert "shutil.copy2(src_inflate_sh, dst_inflate_sh)" not in src


def test_build_tool_documents_cuda_prestage_advisory_closure() -> None:
    """The tool must explicitly cite the CUDA-PRESTAGE advisory it closes."""
    src = _read_tool_source()
    assert "CUDA-PRESTAGE" in src
    assert "tier_a_cuda_dispatch_packets_prestaged_20260508" in src


def test_build_tool_has_smoke_roundtrip_with_n_pairs_assertion() -> None:
    """Per BUGCLASSES B1: encoder must have a paired decoder roundtrip
    test. The tool must fail-loud if smoke n_latent_pairs != 600 (PR106
    contract) or if weight identity rel_err > 1e-5."""
    src = _read_tool_source()
    assert "_local_smoke_roundtrip" in src
    assert "PR106_FRAME_PAIRS = 600" in src
    assert "n_frames_implied" in src
    assert "weight_identity_rel_err_smoke" in src
    # Smoke must raise on mismatch (not warn).
    assert "raise SystemExit(" in src


def test_verify_tool_exists_for_b3_custody() -> None:
    """Per BUGCLASSES B3: the build_manifest.json archive_relpath must
    EITHER be committed OR have a rebuild-and-SHA-assert smoke. We chose
    the latter; the verifier MUST exist alongside the build tool."""
    assert VERIFY_TOOL.is_file(), (
        f"BUGCLASSES B3 custody verifier missing at {VERIFY_TOOL}"
    )
    verify_src = VERIFY_TOOL.read_text()
    assert "BUGCLASSES B3" in verify_src
    assert "build_pr106_uniward_runtime_packet" in verify_src
    assert "rebuilt archive SHA mismatch" in verify_src


# ---------------------------------------------------------------------------
# Manifest schema + canonical-output integrity (uses the local sample
# build artifact at experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/).
# ---------------------------------------------------------------------------


def test_canonical_build_manifest_schema_and_byte_targets() -> None:
    """Verify the canonical sample build artifact's manifest matches the
    schema this build tool produces. Validates the tool actually ran
    end-to-end and produced a wire-correct PR106 packet at rms=0.05."""
    if not CANONICAL_MANIFEST.is_file():
        # On a fresh checkout the canonical output directory will not exist;
        # the verifier handles that. Skip rather than fail.
        pytest.skip(
            f"canonical build manifest not on disk: {CANONICAL_MANIFEST}"
        )
    manifest = json.loads(CANONICAL_MANIFEST.read_text())
    assert manifest["schema_version"] == (
        "pr106_uniward_lagrangian_runtime_packet_build.v1"
    )
    assert manifest["lane_id"] == "pr106_uniward_lagrangian_runtime_packet"
    assert manifest["rms_target"] == 0.05
    assert manifest["brotli_quality"] == 11
    assert manifest["input_source_archive_bytes"] == 186239
    assert manifest["expected_source_archive_bytes"] == 186239
    assert manifest["input_source_archive_sha256"] == (
        "3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58"
    )
    assert manifest["expected_source_archive_sha256"] == (
        "3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58"
    )
    assert manifest["input_source_archive_member_name"] == "0.bin"
    assert manifest["expected_source_archive_member_name"] == "0.bin"
    assert manifest["input_source_archive_member_bytes"] == 186131
    assert manifest["expected_source_archive_member_bytes"] == 186131
    assert manifest["input_source_archive_member_sha256"] == (
        "7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7"
    )
    assert manifest["expected_source_archive_member_sha256"] == (
        "7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7"
    )
    assert manifest["n_tensors"] == 28
    assert manifest["n_symbols"] == 228958
    # Wire-format identity: 600 latent pairs decoded; meta matches PR106.
    assert manifest["smoke_n_latent_pairs_decoded"] == 600
    assert manifest["smoke_n_frames_implied"] == 1200
    assert manifest["smoke_meta_latent_dim"] == 28
    assert manifest["smoke_meta_base_channels"] == 36
    assert manifest["smoke_meta_eval_size"] == [384, 512]
    # Achieved rel_err must be at-or-below the rms_target ± fp32 noise.
    assert manifest["achieved_rel_err"] <= manifest["rms_target"] + 1e-3
    # weight-identity rel_err is a wire-format integrity check; must be
    # essentially zero (fp32 ULP).
    assert manifest["achieved_rel_err_smoke_weight_identity"] < 1e-5
    # Archive must be smaller than the PR106 published baseline.
    assert manifest["archive_bytes"] < manifest["pr106_archive_baseline_bytes"]
    # Custody compliance — manifest must declare ALL CLAUDE.md flags.
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["cuda_eval_worth_testing"] is True
    assert manifest["family_falsified"] is False
    assert (
        manifest["falsification_scope"] == "pr106_uniward_runtime_packet_only"
    )
    assert manifest["score_affecting_payload_changed"] is True
    assert manifest["charged_bits_changed"] is True
    assert manifest["wire_layout_identity_with_pr106_published"] is True
    assert manifest["wire_payload_byte_identity_with_pr106_published"] is False
    assert manifest["wire_format_identity_with_pr106_published"] is False
    assert manifest["evidence_grade"] == "[CPU-build]"
    # SHA + bytes must be present (BUGCLASSES B3).
    assert isinstance(manifest["archive_sha256"], str)
    assert len(manifest["archive_sha256"]) == 64
    assert int(manifest["archive_bytes"]) > 0


def test_canonical_archive_byte_closure_against_manifest() -> None:
    """Verify the on-disk archive matches the manifest's recorded SHA-256
    + byte count (BUGCLASSES B3 byte-closure)."""
    if not CANONICAL_MANIFEST.is_file():
        pytest.skip(
            f"canonical build manifest not on disk: {CANONICAL_MANIFEST}"
        )
    manifest = json.loads(CANONICAL_MANIFEST.read_text())
    archive_path = CANONICAL_OUTPUT_DIR / "archive.zip"
    assert archive_path.is_file(), (
        f"canonical archive missing: {archive_path}"
    )
    archive_bytes = archive_path.read_bytes()
    assert len(archive_bytes) == manifest["archive_bytes"]
    actual_sha = hashlib.sha256(archive_bytes).hexdigest()
    assert actual_sha == manifest["archive_sha256"]
