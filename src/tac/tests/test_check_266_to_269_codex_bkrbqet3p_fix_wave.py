# SPDX-License-Identifier: MIT
"""Catalog #266 / #267 / #268 / #269 — codex review bkrbqet3p 4 fix-wave gates.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable. Memory:
``feedback_codex_fix_wave_bkrbqet3p_4_findings_LANDED_20260515.md``.

Codex review bkrbqet3p 2026-05-15 surfaced 4 verdict=needs-attention
findings:

* F1 (Catalog #266): G1 substrate trainer emits archive with empty
  hyperprior + class-index slots; auth-eval measures direct-residual
  control NOT the advertised G1 scorer-class gating.
* F2 (Catalog #267): G1 substrate trainer silently falls back to
  uniform class assignment on any scorer error; converts dependency /
  runtime / scorer failures into plausible-looking training run.
* F3 (Catalog #268): blahut_arimoto_theoretical_floor.contest_score_floor
  adds 25 * bits-per-unit directly to the contest score (dimensionally
  invalid against contest's normalized-archive-bytes rate term).
* F4 (Catalog #269): mackay_conditional_entropy_a1_archive presents
  position-partition-by-hash buckets as ``H(X | scorer_state_dict)``
  while never reading scorer state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only,
    check_scorer_conditional_entropy_actually_uses_scorer_state,
    check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in,
    check_theoretical_floor_rate_term_unit_calibrated,
)


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Catalog #266 (F1) tests
# ---------------------------------------------------------------------------


_G1_TRAINER_RELPATH = (
    "experiments/train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py"
)


def _g1_canonical_body() -> str:
    """Canonical body containing all 3 required validator surfaces."""
    return """\
# Codex bkrbqet3p F1 + F2 fix wave
import argparse


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--research-only-direct-residual", action="store_true")
    p.add_argument("--allow-uniform-class-fallback", action="store_true")
    return p


def _validate_g1_research_only_flags(args):
    if not args.research_only_direct_residual:
        raise SystemExit("ERROR: needs --research-only-direct-residual")


def _full_main(args):
    _g1_archive_consumes_hyperprior_bytes = False
    _g1_skip_auth_eval_reason = None
    if not _g1_archive_consumes_hyperprior_bytes:
        _g1_skip_auth_eval_reason = "catalog_266_direct_residual"
    return 0
"""


def test_check_266_flags_g1_trainer_missing_required_flag(tmp_path: Path) -> None:
    # Strip every textual reference to the required flag.
    body = _g1_canonical_body().replace(
        "--research-only-direct-residual", "--unrelated-flag-name"
    ).replace("research_only_direct_residual", "unrelated_attr")
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    violations = (
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert len(violations) >= 1
    assert any("--research-only-direct-residual" in v for v in violations)


def test_check_266_flags_g1_trainer_missing_validator_function(tmp_path: Path) -> None:
    body = _g1_canonical_body().replace(
        "_validate_g1_research_only_flags", "_unrelated_helper"
    )
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    violations = (
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert any("_validate_g1_research_only_flags" in v for v in violations)


def test_check_266_flags_g1_trainer_missing_archive_consumes_marker(
    tmp_path: Path,
) -> None:
    body = _g1_canonical_body().replace(
        "_g1_archive_consumes_hyperprior_bytes", "_unrelated_marker"
    )
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    violations = (
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert any("_g1_archive_consumes_hyperprior_bytes" in v for v in violations)


def test_check_266_accepts_canonical_g1_trainer(tmp_path: Path) -> None:
    _write(tmp_path, _G1_TRAINER_RELPATH, _g1_canonical_body())
    assert (
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_266_skips_when_target_file_absent(tmp_path: Path) -> None:
    assert (
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_266_strict_raises_on_violation(tmp_path: Path) -> None:
    _write(tmp_path, _G1_TRAINER_RELPATH, "# empty body")
    with pytest.raises(PreflightError, match="Catalog #266"):
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_266_file_level_waiver_with_rationale_accepts(tmp_path: Path) -> None:
    body = (
        "# G1_ARCHIVE_DIRECT_RESIDUAL_OK: archive consumer landed via X+Y+Z\n"
        + "# empty intentionally\n"
    )
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    assert (
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_266_placeholder_rationale_rejected(tmp_path: Path) -> None:
    body = "# G1_ARCHIVE_DIRECT_RESIDUAL_OK: <rationale>\n# empty\n"
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    violations = (
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert len(violations) >= 1


def test_check_266_live_repo_has_no_violations() -> None:
    assert (
        check_g1_substrate_archive_consumes_hyperprior_class_bytes_or_research_only(
            strict=False, verbose=False
        )
        == []
    )


# ---------------------------------------------------------------------------
# Catalog #267 (F2) tests
# ---------------------------------------------------------------------------


def _g1_f2_canonical_body() -> str:
    return """\
# Codex bkrbqet3p F2 fix wave
import argparse


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--allow-uniform-class-fallback", action="store_true")
    return p


def _full_main(args):
    try:
        x = _compute_class_indices()
    except Exception as exc:
        if not args.allow_uniform_class_fallback:
            raise SystemExit("ERROR: scorer derivation failed") from exc
        args._g1_uniform_class_fallback_active = True
        x = _uniform_fallback()
    return 0
"""


def test_check_267_flags_missing_flag(tmp_path: Path) -> None:
    body = _g1_f2_canonical_body().replace(
        "--allow-uniform-class-fallback", "--unrelated-flag"
    ).replace("allow_uniform_class_fallback", "unrelated_attr")
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    violations = (
        check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert any("--allow-uniform-class-fallback" in v for v in violations)


def test_check_267_flags_missing_guard_token(tmp_path: Path) -> None:
    body = _g1_f2_canonical_body().replace(
        "args.allow_uniform_class_fallback", "args.unrelated"
    )
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    violations = (
        check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert any("args.allow_uniform_class_fallback" in v for v in violations)


def test_check_267_flags_missing_pairing_marker(tmp_path: Path) -> None:
    body = _g1_f2_canonical_body().replace(
        "_g1_uniform_class_fallback_active", "_unrelated_marker"
    )
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    violations = (
        check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert any("_g1_uniform_class_fallback_active" in v for v in violations)


def test_check_267_accepts_canonical_body(tmp_path: Path) -> None:
    _write(tmp_path, _G1_TRAINER_RELPATH, _g1_f2_canonical_body())
    assert (
        check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_267_strict_raises_on_violation(tmp_path: Path) -> None:
    _write(tmp_path, _G1_TRAINER_RELPATH, "# empty\n")
    with pytest.raises(PreflightError, match="Catalog #267"):
        check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_267_file_level_waiver_accepts(tmp_path: Path) -> None:
    body = "# G1_UNIFORM_FALLBACK_OK: trainer no longer has any silent uniform path\n"
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    assert (
        check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_267_placeholder_rationale_rejected(tmp_path: Path) -> None:
    body = "# G1_UNIFORM_FALLBACK_OK: <reason>\n"
    _write(tmp_path, _G1_TRAINER_RELPATH, body)
    violations = (
        check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert len(violations) >= 1


def test_check_267_live_repo_has_no_violations() -> None:
    assert (
        check_substrate_scorer_class_derivation_fail_closed_or_explicit_opt_in(
            strict=False, verbose=False
        )
        == []
    )


# ---------------------------------------------------------------------------
# Catalog #268 (F3) tests
# ---------------------------------------------------------------------------


_F3_TARGET_RELPATH = "src/tac/symposium_impls/blahut_arimoto_theoretical_floor.py"


def test_check_268_flags_uncalibrated_rate_term(tmp_path: Path) -> None:
    body = """\
# bare bits-per-unit theoretical floor
def compute_floor(d_seg, d_pose):
    r_combined = 0.5
    return 100 * d_seg + 25.0 * r_combined
"""
    _write(tmp_path, _F3_TARGET_RELPATH, body)
    violations = check_theoretical_floor_rate_term_unit_calibrated(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "bits-per-unit" in violations[0]


def test_check_268_accepts_canonical_calibration(tmp_path: Path) -> None:
    body = """\
CONTEST_RATE_DENOM_BYTES = 37_545_489


def bits_per_unit_to_contest_rate_term(bits_per_unit, num_units):
    archive_bytes = (bits_per_unit * num_units) / 8
    return 25.0 * archive_bytes / CONTEST_RATE_DENOM_BYTES


def compute_floor(d_seg, d_pose, num_units):
    r_bits = 0.5
    rate_term = bits_per_unit_to_contest_rate_term(r_bits, num_units)
    return 100 * d_seg + rate_term
"""
    _write(tmp_path, _F3_TARGET_RELPATH, body)
    assert (
        check_theoretical_floor_rate_term_unit_calibrated(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_268_accepts_advisory_tag(tmp_path: Path) -> None:
    body = """\
# theoretical_floor_units_calibrated=false  (advisory tag)
def compute_floor(d_seg, d_pose):
    return 100 * d_seg + 25.0 * 0.5
"""
    _write(tmp_path, _F3_TARGET_RELPATH, body)
    assert (
        check_theoretical_floor_rate_term_unit_calibrated(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_268_skips_when_target_absent(tmp_path: Path) -> None:
    assert (
        check_theoretical_floor_rate_term_unit_calibrated(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_268_strict_raises_on_violation(tmp_path: Path) -> None:
    _write(tmp_path, _F3_TARGET_RELPATH, "x = 25.0 * r_combined\n")
    with pytest.raises(PreflightError, match="Catalog #268"):
        check_theoretical_floor_rate_term_unit_calibrated(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_268_file_level_waiver_accepts(tmp_path: Path) -> None:
    body = "# THEORETICAL_FLOOR_UNITS_OK: pre-calibration stub for v0.1.0-rc1\n"
    _write(tmp_path, _F3_TARGET_RELPATH, body)
    assert (
        check_theoretical_floor_rate_term_unit_calibrated(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


# ---------------------------------------------------------------------------
# Catalog #269 (F4) tests
# ---------------------------------------------------------------------------


_F4_TARGET_RELPATH = "src/tac/symposium_impls/mackay_conditional_entropy_a1_archive.py"


def test_check_269_flags_position_partition_without_disclosure(
    tmp_path: Path,
) -> None:
    body = '''\
"""Implements H(X | scorer_state_dict) — claim-only."""

import hashlib


def _scorer_prior_buckets(payload, n_buckets=8):
    buckets = [bytearray() for _ in range(n_buckets)]
    for pos, byte in enumerate(payload):
        h = hashlib.blake2b(pos.to_bytes(8, "little"), digest_size=4).digest()
        buckets[int.from_bytes(h, "little") % n_buckets].append(byte)
    return [bytes(b) for b in buckets]
'''
    _write(tmp_path, _F4_TARGET_RELPATH, body)
    violations = check_scorer_conditional_entropy_actually_uses_scorer_state(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_269_accepts_real_scorer_feature_binding(tmp_path: Path) -> None:
    body = """\
from tac.differentiable_eval_roundtrip import load_default_scorers


def estimate_conditional(payload):
    pose, seg = load_default_scorers()
    # ... real binding ...
    return 0.0
"""
    _write(tmp_path, _F4_TARGET_RELPATH, body)
    assert (
        check_scorer_conditional_entropy_actually_uses_scorer_state(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_269_accepts_self_disclosure(tmp_path: Path) -> None:
    body = '''\
"""Position-partition proxy for H(X | partition)."""

# true_scorer_conditional_entropy_claim=false
def estimate_position_partition(payload):
    return 0.0
'''
    _write(tmp_path, _F4_TARGET_RELPATH, body)
    assert (
        check_scorer_conditional_entropy_actually_uses_scorer_state(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_269_skips_when_target_absent(tmp_path: Path) -> None:
    assert (
        check_scorer_conditional_entropy_actually_uses_scorer_state(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_269_strict_raises_on_violation(tmp_path: Path) -> None:
    _write(tmp_path, _F4_TARGET_RELPATH, "# bare module no scorer access\n")
    with pytest.raises(PreflightError, match="Catalog #269"):
        check_scorer_conditional_entropy_actually_uses_scorer_state(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_269_file_level_waiver_accepts(tmp_path: Path) -> None:
    body = "# SCORER_CONDITIONAL_ENTROPY_OK: stub module pending feature binding\n"
    _write(tmp_path, _F4_TARGET_RELPATH, body)
    assert (
        check_scorer_conditional_entropy_actually_uses_scorer_state(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_check_269_docstring_mention_alone_does_NOT_satisfy(
    tmp_path: Path,
) -> None:
    """F4 anchor: docstring mentions of scorer_state_dict don't satisfy.

    Codex caught the live MacKay file claiming H(X | scorer_state_dict)
    in the module docstring while the implementation only does
    position-partition bucketing. Tokens must be CALL signatures.
    """
    body = '''\
"""Computes H(A1_archive | scorer_state_dict) via partition.

This module references scorer_state_dict in prose only.
"""


def buckets(payload, n=8):
    return [payload[i::n] for i in range(n)]
'''
    _write(tmp_path, _F4_TARGET_RELPATH, body)
    violations = check_scorer_conditional_entropy_actually_uses_scorer_state(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1, (
        "docstring 'scorer_state_dict' mention must NOT count as feature binding"
    )
