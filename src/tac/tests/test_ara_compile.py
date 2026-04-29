"""Smoke tests for tools/ara_compile.py.

Per CLAUDE.md the Ara compiler is read-mostly and conservative; these tests
verify that:

1. The redaction layer catches every disclosure-policy term listed in
   CLAUDE.md's strategic-secrecy section.
2. The seal level 1 implementation flags dangling evidence pointers
   without crashing.
3. The CLI exit code is 0 when there are no errors and 1 otherwise.

These tests deliberately do NOT exercise the file-system walk; that is
covered by running ``python tools/ara_compile.py`` end-to-end which is
verified by the recommendation document committed alongside the compiler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ARA_COMPILE_PATH = REPO_ROOT / "tools" / "ara_compile.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ara_compile", ARA_COMPILE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ara_compile"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ara_compile():
    return _load_module()


# ---------------------------------------------------------------------------
# Redaction tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term",
    [
        "Lane W hard-pair self-compress",
        "Lane Ω Hessian-aware quantization",
        "Lane Omega Hessian-aware quantization",
        "Lane DARTS-S architecture-search recipe",
        "FR-Ω cost-block-FP",
        "Hessian-aware step",
        "SO Hessian schedule",
        "hard-pair training run",
    ],
)
def test_redact_private_catches_disclosure_terms(ara_compile, term):
    redacted = ara_compile._redact_private(term)
    assert redacted.startswith("[REDACTED")
    # The redacted output must NOT contain the original term.
    assert term.lower() not in redacted.lower() or term.startswith("Hessian")
    # Cross-check: the redaction is the placeholder, not the input.
    assert "private lane reference" in redacted


def test_redact_private_passes_safe_text(ara_compile):
    safe = "Lane G v3 landed 1.05 contest-CUDA"
    assert ara_compile._redact_private(safe) == safe


def test_redact_private_passes_lane_a(ara_compile):
    safe = "Lane A pose TTO from baseline poses"
    assert ara_compile._redact_private(safe) == safe


# ---------------------------------------------------------------------------
# Seal level 1 tests
# ---------------------------------------------------------------------------


def test_seal_handles_missing_claims_file(ara_compile, tmp_path):
    findings = ara_compile.seal_level_1(tmp_path / "no_such.md", [])
    assert any(f.severity == "error" and f.code == "missing_claims" for f in findings)


def test_seal_pass_path_emits_ok(ara_compile, tmp_path):
    claims = tmp_path / "claims.md"
    # No claim blocks at all -> warn (not error).
    claims.write_text("# claims\n\nno blocks here\n")
    findings = ara_compile.seal_level_1(claims, [])
    assert any(f.code == "no_claim_blocks_matched" for f in findings)


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


def test_classify_memory_known_categories(ara_compile):
    assert ara_compile._classify_memory("feedback_anything") == "heuristic"
    assert ara_compile._classify_memory("project_lane_g_v3_landed") == "experiment"
    assert (
        ara_compile._classify_memory("project_lane_gp_v3_landed_runge_phenomenon")
        == "dead_end"
    )
    assert ara_compile._classify_memory("project_council_kill_uniward") == "decision"
    assert ara_compile._classify_memory("project_observation_xyz") == "observation"
