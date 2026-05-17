# SPDX-License-Identifier: MIT
"""Catalog #319 STRICT preflight gate tests.

Verifies ``check_substrate_wyner_ziv_reweight_has_deliverability_proof`` per
the Q2 deliverable of
``lane_q2_q3_batched_catalog_319_gate_plus_autopilot_reweight_v2_20260517``.

Bug class anchor: the autopilot's HIGH_PAIR_INVARIANT reward branch must
consult the per-archive ``DeliverabilityProof`` artifact emitted by
``tac.wyner_ziv_deliverability.proof_builder``. Venn classification alone is
a planning signal; it is NOT contest deliverability proof. The fec6 empirical
anchor (``probe_f174192aeadf_20260517T205208.json``) shows lzma/brotli/zlib
all INFLATE the candidate-shared-prior set, proving the previous flat 1.15×
HIGH_PAIR_INVARIANT reward was FAKE for fec6.

Acceptance for the gate (per the canonical helper docstring):

  (a) Wrapper consults ``load_deliverability_proof_for_archive(archive_sha256)``
      AND ``verify_deliverability_proof_contest_compliance`` AND
      ``_venn_deliverability_reward_factor_for_archive`` in the source text;
  (b) Same-line waiver
      ``# VENN_REWEIGHT_DELIVERABILITY_OK:<rationale>`` (placeholder
      ``<rationale>`` / ``<reason>`` literal rejected so the gate's docstring
      example cannot self-waive);
  (c) Blanket ``_VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR`` constants
      (if defined) MUST carry the waiver inline.

Sister of Catalog #1 (`check_no_mps_fallback_default`), Catalog #127
(custody routing), Catalog #205 (inflate device-fork), and Catalog #220
(substrate L1+ scaffold operational mechanism).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_wyner_ziv_reweight_has_deliverability_proof,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ──────────────────────────────────────────────────────────────────────────── #
# 1. Live-repo regression guard (the canonical clean state)                    #
# ──────────────────────────────────────────────────────────────────────────── #


def test_live_repo_zero_violations():
    """Catalog #319 live count MUST be 0 (Q2 lands strict-from-byte-one)."""
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert violations == [], (
        f"Catalog #319 expected 0 live violations; got {len(violations)}: "
        f"{violations[:3]}"
    )


def test_live_repo_strict_passes():
    """STRICT-mode call on the live repo MUST NOT raise."""
    # Should not raise PreflightError; should return empty list
    result = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=REPO_ROOT, strict=True, verbose=False
    )
    assert result == []


# ──────────────────────────────────────────────────────────────────────────── #
# 2. Synthetic positive cases (bug class regressions)                           #
# ──────────────────────────────────────────────────────────────────────────── #


def _write_fake_autopilot(tmp_path: Path, body: str) -> Path:
    """Stage a fake repo with a synthetic autopilot file."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(parents=True)
    target = tools_dir / "cathedral_autopilot_autonomous_loop.py"
    target.write_text(body, encoding="utf-8")
    return tmp_path


def test_synthetic_high_invariant_without_proof_consultation_flagged(tmp_path):
    """HIGH_PAIR_INVARIANT branch without DeliverabilityProof consultation = violation."""
    body = """
def adjust(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * 1.15
    return predicted_delta
"""
    repo = _write_fake_autopilot(tmp_path, body)
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=False
    )
    assert len(violations) >= 1
    assert any("Catalog #319" in v for v in violations)
    assert any(
        "missing" in v and "load_deliverability_proof_for_archive" in v
        for v in violations
    )


def test_synthetic_canonical_pattern_accepted(tmp_path):
    """Canonical pattern (3 required tokens) MUST pass."""
    body = """
# Canonical Q2+Q3 cascade
from tac.wyner_ziv_deliverability.proof_builder import (
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

def _venn_deliverability_reward_factor_for_archive(archive_sha256):
    return 1.0

def adjust(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * _venn_deliverability_reward_factor_for_archive(
            archive_sha256
        )
    return predicted_delta
"""
    repo = _write_fake_autopilot(tmp_path, body)
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=False
    )
    assert violations == [], f"unexpected violations: {violations}"


def test_synthetic_no_high_invariant_branch_skipped(tmp_path):
    """Source without HIGH_PAIR_INVARIANT branch = out of scope (no violation)."""
    body = """
def adjust(predicted_delta, archive_sha256):
    return predicted_delta
"""
    repo = _write_fake_autopilot(tmp_path, body)
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=False
    )
    assert violations == []


def test_synthetic_blanket_constant_flagged(tmp_path):
    """Bare _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR usage WITHOUT waiver = flagged."""
    body = """
# Canonical token + token-set present...
from tac.wyner_ziv_deliverability.proof_builder import (
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

def _venn_deliverability_reward_factor_for_archive(sha):
    return 1.0

_VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR = 1.15

def adjust(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR
    return predicted_delta
"""
    repo = _write_fake_autopilot(tmp_path, body)
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=False
    )
    # The constant definition + use should flag (no waiver on either line)
    assert any(
        "blanket" in v and "HIGH_PAIR_INVARIANT delta factor" in v
        for v in violations
    ), f"expected blanket-factor violation; got: {violations}"


def test_synthetic_blanket_constant_with_waiver_accepted(tmp_path):
    """Bare delta factor WITH same-line waiver MUST pass (canonical opt-out)."""
    body = """
from tac.wyner_ziv_deliverability.proof_builder import (
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

def _venn_deliverability_reward_factor_for_archive(sha):
    return 1.0

# This constant is the per-tier factor floor; canonical helper consults it post-proof.
_VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR = 1.15  # VENN_REWEIGHT_DELIVERABILITY_OK:per-tier-factor-floor consumed by helper

def adjust(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * _venn_deliverability_reward_factor_for_archive(archive_sha256)
    return predicted_delta
"""
    repo = _write_fake_autopilot(tmp_path, body)
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=False
    )
    assert violations == [], f"unexpected violations: {violations}"


# ──────────────────────────────────────────────────────────────────────────── #
# 3. Waiver semantics (placeholder rejection)                                   #
# ──────────────────────────────────────────────────────────────────────────── #


def test_placeholder_waiver_rejected(tmp_path):
    """Waiver with literal `<rationale>` MUST be rejected."""
    body = """
from tac.wyner_ziv_deliverability.proof_builder import (
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

def _venn_deliverability_reward_factor_for_archive(sha):
    return 1.0

_VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR = 1.15  # VENN_REWEIGHT_DELIVERABILITY_OK:<rationale>

def adjust(p, s):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return p * _venn_deliverability_reward_factor_for_archive(s)
    return p
"""
    repo = _write_fake_autopilot(tmp_path, body)
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=False
    )
    # Placeholder rejection should surface either via placeholder-waiver flag
    # OR via the blanket-factor violation (no valid waiver on the line)
    assert len(violations) >= 1


def test_placeholder_reason_waiver_rejected(tmp_path):
    """Waiver with literal `<reason>` MUST be rejected."""
    body = """
from tac.wyner_ziv_deliverability.proof_builder import (
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

def _venn_deliverability_reward_factor_for_archive(sha):
    return 1.0

_VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR = 1.15  # VENN_REWEIGHT_DELIVERABILITY_OK:<reason>

def adjust(p, s):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return p * _venn_deliverability_reward_factor_for_archive(s)
    return p
"""
    repo = _write_fake_autopilot(tmp_path, body)
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=False
    )
    assert len(violations) >= 1


def test_bare_waiver_marker_rejected(tmp_path):
    """Waiver marker with empty rationale MUST be rejected."""
    body = """
from tac.wyner_ziv_deliverability.proof_builder import (
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

def _venn_deliverability_reward_factor_for_archive(sha):
    return 1.0

_VENN_REWEIGHT_HIGH_PAIR_INVARIANT_DELTA_FACTOR = 1.15  # VENN_REWEIGHT_DELIVERABILITY_OK:

def adjust(p, s):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return p * _venn_deliverability_reward_factor_for_archive(s)
    return p
"""
    repo = _write_fake_autopilot(tmp_path, body)
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=False
    )
    assert len(violations) >= 1


# ──────────────────────────────────────────────────────────────────────────── #
# 4. Strict-mode semantics                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


def test_strict_mode_raises_on_violation(tmp_path):
    """strict=True MUST raise PreflightError on violation."""
    body = """
def adjust(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * 1.15
    return predicted_delta
"""
    repo = _write_fake_autopilot(tmp_path, body)
    with pytest.raises(PreflightError) as exc_info:
        check_substrate_wyner_ziv_reweight_has_deliverability_proof(
            repo_root=repo, strict=True
        )
    assert "check_substrate_wyner_ziv_reweight_has_deliverability_proof" in str(
        exc_info.value
    )


def test_strict_mode_silent_on_clean(tmp_path):
    """strict=True MUST NOT raise when no violations."""
    body = """
from tac.wyner_ziv_deliverability.proof_builder import (
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

def _venn_deliverability_reward_factor_for_archive(sha):
    return 1.0

def adjust(predicted_delta, archive_sha256):
    if pair_invariant_frac >= _VENN_REWEIGHT_HIGH_PAIR_INVARIANT_THRESHOLD:
        return predicted_delta * _venn_deliverability_reward_factor_for_archive(archive_sha256)
    return predicted_delta
"""
    repo = _write_fake_autopilot(tmp_path, body)
    result = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=repo, strict=True
    )
    assert result == []


# ──────────────────────────────────────────────────────────────────────────── #
# 5. Edge cases (missing repo, unreadable file, no autopilot)                   #
# ──────────────────────────────────────────────────────────────────────────── #


def test_missing_autopilot_returns_empty(tmp_path):
    """Repo without tools/cathedral_autopilot_autonomous_loop.py MUST return []."""
    # Empty repo with no tools/ subdir
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_string_repo_root_accepted(tmp_path):
    """repo_root may be a str (not just Path)."""
    body = """
def adjust(predicted_delta, archive_sha256):
    return predicted_delta
"""
    repo = _write_fake_autopilot(tmp_path, body)
    # Pass as a string instead of Path
    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof(
        repo_root=str(repo), strict=False
    )
    assert violations == []


# ──────────────────────────────────────────────────────────────────────────── #
# 6. Orchestrator wire-in regression guard                                      #
# ──────────────────────────────────────────────────────────────────────────── #


def test_orchestrator_callsite_strict_true():
    """Verify preflight_all() wires Catalog #319 with strict=True per Q2 strict-flip."""
    preflight_text = (REPO_ROOT / "src/tac/preflight.py").read_text(encoding="utf-8")
    # Find the orchestrator callsite for our gate. Note: preflight_all() lives
    # in the early part of the file (line ~2500); the gate's def lives in the
    # check-implementations block (line ~70200). Scan the orchestrator region
    # explicitly so the gate def doesn't satisfy this test.
    marker = "check_substrate_wyner_ziv_reweight_has_deliverability_proof("
    # First occurrence is the orchestrator wire-in; second is the gate def.
    idx = preflight_text.find(marker)
    assert idx != -1, "Catalog #319 orchestrator callsite not found"
    # Confirm we have the wire-in (not the def): the line should NOT start
    # with `def ` and SHOULD precede a kwargs invocation.
    line_start = preflight_text.rfind("\n", 0, idx) + 1
    line = preflight_text[line_start : preflight_text.find("\n", idx)]
    assert not line.lstrip().startswith("def "), (
        f"first match is the def, not the orchestrator callsite: {line!r}"
    )
    # Get the ~200 chars following the call (should contain strict=True)
    snippet = preflight_text[idx : idx + 200]
    assert "strict=True" in snippet, (
        f"Catalog #319 orchestrator callsite must be strict=True per Q2 strict-flip; "
        f"got snippet: {snippet[:150]!r}"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# 7. Catalog #185 sister regression (gate function callable via globals)        #
# ──────────────────────────────────────────────────────────────────────────── #


def test_gate_function_callable_via_globals():
    """Catalog #185 sister: gate function MUST be in tac.preflight globals."""
    import tac.preflight as preflight_module

    fn = getattr(
        preflight_module,
        "check_substrate_wyner_ziv_reweight_has_deliverability_proof",
        None,
    )
    assert fn is not None, (
        "Catalog #319 gate function NOT exposed via tac.preflight globals; "
        "Catalog #185 META-meta-meta gate would refuse the LIVE_COUNT_ZERO claim"
    )
    assert callable(fn)


def test_gate_signature_keyword_only():
    """Gate signature must accept (*, repo_root=, strict=, verbose=) keyword-only."""
    import inspect

    sig = inspect.signature(check_substrate_wyner_ziv_reweight_has_deliverability_proof)
    params = sig.parameters
    assert "repo_root" in params
    assert "strict" in params
    assert "verbose" in params
    # All params should be keyword-only (no positional)
    for name, p in params.items():
        if name in ("repo_root", "strict", "verbose"):
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"param {name} must be KEYWORD_ONLY; got kind={p.kind}"
            )
