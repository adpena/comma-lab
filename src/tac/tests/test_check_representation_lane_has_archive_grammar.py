# SPDX-License-Identifier: MIT
# FAKE_LANE_OK_FILE: this test file constructs fake representation-lane
# fixtures (lane_test_nerv, lane_pose_dc3, lane_c30_xyz, lane_arc3_v2, etc.)
# to exercise Check #124's classification + field-discovery logic. Per Check #126
# file-level waiver semantics.
"""Tests for preflight Catalog #124:
``check_representation_lane_has_archive_grammar_at_design_time``.

The check refuses Level 1+ promotion of representation/codec lanes
(NeRV / HNeRV / Cool-Chic / C3 / wavelet / VQ-VAE / grayscale-LUT / SIREN /
hyperprior / etc.) without 8 design-time evidence fields:

  archive_grammar, parser_section_manifest, inflate_runtime_loc_budget,
  runtime_dep_closure, export_format, score_aware_loss, bolt_on_loc_budget,
  no_op_detector_planned

Two opt-outs (per HNeRV parity discipline lessons 2 + 7):
  - lane_class=substrate_engineering
  - research_only=true

This test set verifies:
  1. Each of the 8 fields missing → WARN (not RAISE) in non-strict
  2. All 8 fields present → no warning
  3. Opt-outs (lane_class=substrate_engineering, research_only=true) → no warning
  4. Non-representation lanes → no warning regardless
  5. STRICT mode → RAISE PreflightError with formatted message
  6. Level 0 lane → no warning (only level >= 1)
  7. Missing 'evidence' dict → WARN
  8. Field discovery via top-level / evidence dict / design_evidence dict /
     gate evidence string substring
  9. Lane name token detection (NeRV, HNeRV, Cool-Chic, wavelet, VQ-VAE, etc.)
 10. Lane description token detection (representation, learned codec)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    _REPRESENTATION_LANE_REQUIRED_FIELDS,
    PreflightError,
    _lane_has_field,
    _lane_is_representation_lane,
    check_representation_lane_has_archive_grammar_at_design_time,
)

# Lane registry path (constant from tools/lane_maturity.py)
_REGISTRY_REL = ".omx/state/lane_registry.json"

# Canonical gate set — matches tools/lane_maturity.GATES
_GATES = [
    "impl_complete",
    "real_archive_empirical",
    "contest_cuda",
    "contest_cpu",
    "strict_preflight",
    "three_clean_review",
    "memory_entry",
    "deploy_runbook",
]


def _make_repo(tmp: Path, lanes: list[dict]) -> Path:
    """Create a minimal fake repo at tmp with .omx/state/lane_registry.json
    seeded from `lanes`. Returns the repo root."""
    (tmp / ".omx" / "state").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "tac").mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": 1,
        "updated_at": "2026-05-09T00:00:00Z",
        "gate_definitions": {g: f"<def of {g}>" for g in _GATES},
        "lanes": lanes,
    }
    (tmp / _REGISTRY_REL).write_text(json.dumps(registry, indent=2))
    return tmp


def _empty_gates() -> dict:
    return {g: {"status": False, "evidence": ""} for g in _GATES}


def _level1_gates() -> dict:
    """One gate true → level 1."""
    g = _empty_gates()
    g["impl_complete"] = {"status": True, "evidence": "src/tac/fake.py"}
    return g


def _level2_gates() -> dict:
    """impl_complete + real_archive_empirical true → level 2."""
    g = _empty_gates()
    g["impl_complete"] = {"status": True, "evidence": "src/tac/fake.py"}
    g["real_archive_empirical"] = {
        "status": True, "evidence": "src/tac/fake.py",
    }
    return g


def _representation_lane(
    lid: str = "lane_test_nerv",
    name: str = "Test NeRV mask codec",
    level: int = 1,
    gates: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a fake representation lane (matched by 'nerv' token in id/name)."""
    lane = {
        "id": lid,
        "name": name,
        "phase": 1,
        "level": level,
        "gates": gates if gates is not None else (
            _level2_gates() if level == 2 else _level1_gates()
        ),
        "notes": "",
    }
    if extra:
        lane.update(extra)
    return lane


def _all_8_fields() -> dict:
    """Return a dict with all 8 required fields populated."""
    return {
        "archive_grammar": "monolithic_single_file_0_bin",
        "parser_section_manifest": "experiments/results/foo/parser.json",
        "inflate_runtime_loc_budget": 80,
        "runtime_dep_closure": ["brotli"],
        "export_format": "brotli_quantized_with_lzma_latents",
        "score_aware_loss": "score-domain (gradient-through-FastViT-T12 PoseNet)",
        "bolt_on_loc_budget": 350,
        "no_op_detector_planned": True,
    }


# ── Test 1-8: each field missing → WARN ──────────────────────────────────


@pytest.mark.parametrize("missing_field", list(_REPRESENTATION_LANE_REQUIRED_FIELDS))
def test_each_field_missing_individually_warns(
    tmp_path: Path, missing_field: str,
) -> None:
    """Drop one field at a time; check warns for each."""
    fields = _all_8_fields()
    del fields[missing_field]
    lane = _representation_lane(extra=fields)
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1, (
        f"expected 1 violation when {missing_field} missing, got {len(violations)}"
    )
    assert missing_field in violations[0]
    assert "[Check 124]" in violations[0]
    assert "lane_test_nerv" in violations[0]


# ── Test 9: all 8 fields present → no warning ────────────────────────────


def test_all_8_fields_present_no_warning(tmp_path: Path) -> None:
    lane = _representation_lane(extra=_all_8_fields())
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 10-11: opt-outs ─────────────────────────────────────────────────


def test_substrate_engineering_optout_no_warning_even_if_fields_missing(
    tmp_path: Path,
) -> None:
    lane = _representation_lane(extra={"lane_class": "substrate_engineering"})
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_research_only_optout_no_warning_even_if_fields_missing(
    tmp_path: Path,
) -> None:
    lane = _representation_lane(extra={"research_only": True})
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_research_only_false_does_not_optout(tmp_path: Path) -> None:
    """research_only=False is the default semantic; must not opt out."""
    lane = _representation_lane(extra={"research_only": False})
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_lane_class_other_than_substrate_engineering_does_not_optout(
    tmp_path: Path,
) -> None:
    lane = _representation_lane(extra={"lane_class": "production_codec"})
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1


# ── Test 12: non-representation-lane → no warning ────────────────────────


def test_non_representation_lane_no_warning(tmp_path: Path) -> None:
    lane = {
        "id": "lane_g_v3",
        "name": "Lane G v3 (renderer training, score-aware)",
        "phase": 1,
        "level": 3,
        "gates": _empty_gates(),
        "notes": "Renderer training with score-aware loss + EMA + roundtrip.",
    }
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_non_representation_lane_with_psd_in_name_no_warning(tmp_path: Path) -> None:
    """Lane PSD has no representation-lane token → ignored."""
    lane = {
        "id": "lane_psd",
        "name": "Lane PSD architecture",
        "phase": 1,
        "level": 1,
        "gates": _level1_gates(),
        "notes": "PixelShuffle-Downscale architecture; not a separate codec.",
    }
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 13: STRICT mode raises ──────────────────────────────────────────


def test_strict_mode_raises_with_formatted_message(tmp_path: Path) -> None:
    lane = _representation_lane()
    repo = _make_repo(tmp_path, [lane])
    with pytest.raises(PreflightError) as excinfo:
        check_representation_lane_has_archive_grammar_at_design_time(
            repo_root=repo, strict=True, verbose=False,
        )
    msg = str(excinfo.value)
    assert "check_representation_lane_has_archive_grammar_at_design_time" in msg
    assert "[Check 124]" in msg
    assert "lane_test_nerv" in msg
    assert "HNeRV parity discipline forbidden pattern #4" in msg


def test_strict_mode_no_violations_does_not_raise(tmp_path: Path) -> None:
    lane = _representation_lane(extra=_all_8_fields())
    repo = _make_repo(tmp_path, [lane])
    # Should not raise
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=True, verbose=False,
    )
    assert violations == []


# ── Test 14: level 0 lane → no warning ────────────────────────────────────


def test_level_0_lane_no_warning(tmp_path: Path) -> None:
    """Level 0 = SKETCH; the check only enforces from Level 1+."""
    lane = _representation_lane(level=0, gates=_empty_gates())
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 15: missing 'evidence' dict / 'design_evidence' dict — empty lane ──


def test_lane_with_no_evidence_dict_warns(tmp_path: Path) -> None:
    """Lane with no top-level fields and no evidence dict warns on all 8."""
    lane = _representation_lane(extra={})  # no fields at all
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1
    # All 8 fields should be in the violation message
    for f in _REPRESENTATION_LANE_REQUIRED_FIELDS:
        assert f in violations[0]


# ── Test 16-19: field discovery via different locations ──────────────────


def test_field_discovery_via_top_level(tmp_path: Path) -> None:
    fields = _all_8_fields()
    lane = _representation_lane(extra=fields)
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_field_discovery_via_evidence_dict(tmp_path: Path) -> None:
    fields = _all_8_fields()
    lane = _representation_lane(extra={"evidence": fields})
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_field_discovery_via_design_evidence_dict(tmp_path: Path) -> None:
    fields = _all_8_fields()
    lane = _representation_lane(extra={"design_evidence": fields})
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_field_discovery_via_gate_evidence_string(tmp_path: Path) -> None:
    """Inline declaration inside a gate's evidence string is acceptable."""
    gates = _level1_gates()
    inline_evidence = (
        "src/tac/fake.py — design notes: "
        "archive_grammar=monolithic_0_bin; "
        "parser_section_manifest=parsers/foo.json; "
        "inflate_runtime_loc_budget=80; "
        "runtime_dep_closure=[brotli]; "
        "export_format=fp4a_brotli; "
        "score_aware_loss=score-domain (gradient-through-PoseNet); "
        "bolt_on_loc_budget=350; "
        "no_op_detector_planned=true"
    )
    gates["impl_complete"]["evidence"] = inline_evidence
    lane = _representation_lane(gates=gates)
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_field_discovery_partial_via_gate_evidence_warns_on_missing(
    tmp_path: Path,
) -> None:
    """Partial inline declaration warns on missing fields only."""
    gates = _level1_gates()
    gates["impl_complete"]["evidence"] = (
        "src/tac/fake.py — archive_grammar=monolithic; export_format=fp4a"
    )
    lane = _representation_lane(gates=gates)
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1
    # archive_grammar AND export_format should NOT be in missing list
    assert "archive_grammar" not in violations[0]
    assert "export_format" not in violations[0]
    # The other 6 should be in the violation
    expected_missing = set(_REPRESENTATION_LANE_REQUIRED_FIELDS) - {
        "archive_grammar", "export_format"
    }
    for f in expected_missing:
        assert f in violations[0]


# ── Test 20-25: representation-lane name token detection ─────────────────


@pytest.mark.parametrize("token", [
    "nerv", "hnerv", "cool_chic", "c3", "wavelet", "vqvae", "vq_vae",
    "grayscale_lut", "siren", "coordinate_mlp", "hyperprior", "balle",
])
def test_representation_lane_classified_by_name_token(token: str) -> None:
    lane = {"id": f"lane_test_{token}_codec", "name": "Test", "notes": ""}
    assert _lane_is_representation_lane(lane), (
        f"lane id containing '{token}' should be classified as representation"
    )


def test_representation_lane_classified_by_lane_name_token() -> None:
    lane = {"id": "lane_xyz", "name": "Test HNeRV mask encoder", "notes": ""}
    assert _lane_is_representation_lane(lane)


def test_representation_lane_classified_by_description_token() -> None:
    lane = {
        "id": "lane_xyz",
        "name": "Some test lane",
        "notes": "This is a learned codec stack for the renderer.",
    }
    assert _lane_is_representation_lane(lane)


def test_renderer_lane_NOT_classified_as_representation() -> None:
    """A pure renderer training lane is NOT a representation/codec lane."""
    lane = {
        "id": "lane_g_v3",
        "name": "Lane G v3 renderer training",
        "notes": "Renderer training with eval roundtrip, EMA, score-aware loss.",
    }
    assert not _lane_is_representation_lane(lane)


def test_pose_lane_NOT_classified_as_representation() -> None:
    lane = {
        "id": "lane_pose_delta_pd_v2",
        "name": "Lane Pose-delta + PD-V2",
        "notes": "Pose stream byte packer.",
    }
    assert not _lane_is_representation_lane(lane)


def test_explicit_calibration_lane_class_overrides_hnerv_token() -> None:
    lane = {
        "id": "lane_non_hnerv_class_drift_calibration",
        "name": "Non-HNeRV cluster drift calibrations",
        "lane_class": "calibration_diagnostic",
        "notes": "Learns CPU/CUDA drift profiles; emits no representation packet.",
    }
    assert not _lane_is_representation_lane(lane)


def test_fix_wave_guard_lane_not_classified_by_representation_token() -> None:
    lane = {
        "id": "lane_fix_wave_12c_surgical_20260513",
        "name": "FIX-WAVE-12C-SURGICAL: siren_readiness blocklist",
        "notes": "",
    }
    assert not _lane_is_representation_lane(lane)


def test_calibration_lane_class_not_forced_to_archive_grammar(tmp_path: Path) -> None:
    lane = _representation_lane(
        lid="lane_non_hnerv_class_drift_calibration",
        name="Non-HNeRV cluster drift calibrations",
        extra={
            "lane_class": "calibration_diagnostic",
            "notes": (
                "Calibration/control-plane lane. Produces drift posterior rows, "
                "not an archive grammar or inflate runtime."
            ),
        },
    )
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_short_token_word_boundary_does_not_falsely_match_substring() -> None:
    """`c3` as a token must not match `c30` / `dc3` / `arc3` substrings.

    Round 1 adversarial review (Yousfi): the original substring matcher
    was over-permissive on short tokens. Word-boundary regex now enforces
    `[^A-Za-z0-9]` on either side OR start/end of string.
    """
    # False positives that the original substring matcher would have hit:
    assert not _lane_is_representation_lane(
        {"id": "lane_c30_xyz", "name": "version 30 codec", "notes": ""}
    )
    assert not _lane_is_representation_lane(
        {"id": "lane_pose_dc3", "name": "pose dc3 packer", "notes": ""}
    )
    assert not _lane_is_representation_lane(
        {"id": "lane_arc3_v2", "name": "arc3 lane", "notes": ""}
    )


def test_short_token_word_boundary_still_matches_real_token() -> None:
    """Real `c3` lane (with proper word boundary) must still match."""
    assert _lane_is_representation_lane(
        {"id": "lane_arch_c3_v2", "name": "C3 codec", "notes": ""}
    )
    assert _lane_is_representation_lane(
        {"id": "lane_xyz", "name": "Cool-Chic + C3 stack", "notes": ""}
    )
    assert _lane_is_representation_lane(
        {"id": "c3_lane", "name": "test", "notes": ""}
    )


# ── Test 26-28: helper unit tests ─────────────────────────────────────────


def test_lane_has_field_top_level_truthy_only() -> None:
    lane = {"id": "x", "archive_grammar": ""}
    assert not _lane_has_field(lane, "archive_grammar"), (
        "empty string should not satisfy field-presence"
    )
    lane2 = {"id": "x", "archive_grammar": None}
    assert not _lane_has_field(lane2, "archive_grammar")
    lane3 = {"id": "x", "archive_grammar": "monolithic"}
    assert _lane_has_field(lane3, "archive_grammar")


def test_lane_has_field_in_evidence_dict() -> None:
    lane = {"id": "x", "evidence": {"archive_grammar": "monolithic"}}
    assert _lane_has_field(lane, "archive_grammar")


def test_lane_has_field_inline_in_gate_evidence() -> None:
    lane = {
        "id": "x",
        "gates": {
            "impl_complete": {
                "status": True,
                "evidence": "src/foo.py - archive_grammar=monolithic_0_bin",
            },
        },
    }
    assert _lane_has_field(lane, "archive_grammar")


def test_lane_has_field_returns_false_when_absent() -> None:
    lane = {"id": "x", "gates": {}}
    assert not _lane_has_field(lane, "archive_grammar")


def test_lane_has_field_inline_in_notes_string() -> None:
    """Subagent C (Lane 12-v2) declares the 8 fields inline in notes."""
    lane = {
        "id": "x",
        "notes": (
            "Re-scoped per HNeRV retrospective. "
            "archive_grammar=monolithic_0_bin "
            "parser_section_manifest=ARCHIVE_GRAMMAR_constant_in_module "
            "inflate_runtime_loc_budget=100 "
            "runtime_dep_closure=torch+brotli "
            "export_format=monolithic_single_file_0_bin "
            "score_aware_loss=gradient_through_PoseNet+SegNet "
            "bolt_on_loc_budget=350 "
            "no_op_detector_planned=true_via_inflate_roundtrip"
        ),
    }
    for f in _REPRESENTATION_LANE_REQUIRED_FIELDS:
        assert _lane_has_field(lane, f), f"field {f} should be discovered in notes"


def test_field_discovery_via_notes_string_lane_passes(tmp_path: Path) -> None:
    lane = _representation_lane(extra={
        "notes": (
            "archive_grammar=monolithic_0_bin "
            "parser_section_manifest=parser.json "
            "inflate_runtime_loc_budget=100 "
            "runtime_dep_closure=brotli "
            "export_format=fp4a "
            "score_aware_loss=score-domain "
            "bolt_on_loc_budget=350 "
            "no_op_detector_planned=true"
        ),
    })
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 29-30: live-codebase invariant + verbose mode ───────────────────


def test_live_codebase_invariant_at_least_runs_without_crash() -> None:
    """The check runs against the actual live registry without crashing.

    The current live count is non-zero (warn-only initially); STRICT mode
    will raise but non-strict must succeed.
    """
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        strict=False, verbose=False,
    )
    # Do NOT assert violations == [] — the check is warn-only with known
    # in-flight lanes. This test only verifies the check doesn't crash.
    assert isinstance(violations, list)
    for v in violations:
        assert "[Check 124]" in v


def test_verbose_mode_does_not_crash(tmp_path: Path, capsys) -> None:
    lane = _representation_lane()
    repo = _make_repo(tmp_path, [lane])
    check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=True,
    )
    captured = capsys.readouterr()
    assert "representation-lane-archive-grammar" in captured.out


def test_verbose_mode_clean_lane_prints_ok(tmp_path: Path, capsys) -> None:
    lane = _representation_lane(extra=_all_8_fields())
    repo = _make_repo(tmp_path, [lane])
    check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=True,
    )
    captured = capsys.readouterr()
    assert "OK" in captured.out


# ── FIX-A 2026-05-12: ZZZZZ Medium fix — `<=`/`>=` operator acceptance ────
#
# The original `_lane_has_field` matcher only accepted `<field>=` /
# `<field>:` separators. Two in-flight substrates (`cool_chic`, `wavelet`)
# declare `inflate_runtime_loc_budget<=100 LOC` and would false-positive
# bomb the gate on L1 promotion. The hardened matcher now accepts
# `<=` / `>=` / `=` / `:`.


def test_le_operator_in_notes_counts_as_declared() -> None:
    """`<field><=value>` form (e.g. `inflate_runtime_loc_budget<=100`) is
    accepted as a declaration. Regression for the ZZZZZ Medium gap."""
    lane = {
        "id": "lane_test_nerv",
        "notes": "inflate_runtime_loc_budget<=100 LOC",
    }
    assert _lane_has_field(lane, "inflate_runtime_loc_budget"), (
        "`<=` operator in notes must count as a declaration"
    )


def test_ge_operator_in_notes_counts_as_declared() -> None:
    """`<field>>=<value>` form is also accepted."""
    lane = {
        "id": "lane_test_nerv",
        "notes": "bolt_on_loc_budget>=350 LOC",
    }
    assert _lane_has_field(lane, "bolt_on_loc_budget"), (
        "`>=` operator in notes must count as a declaration"
    )


def test_le_operator_in_gate_evidence_counts_as_declared() -> None:
    """`<field><=<value>` in a gate's evidence string is accepted."""
    lane = {
        "id": "lane_test_nerv",
        "gates": {
            "impl_complete": {
                "status": True,
                "evidence": "inflate_runtime_loc_budget<=100 LOC",
            },
        },
    }
    assert _lane_has_field(lane, "inflate_runtime_loc_budget"), (
        "`<=` operator in gate evidence must count as a declaration"
    )


def test_le_operator_full_lane_passes_check(tmp_path: Path) -> None:
    """End-to-end: representation lane declaring all 8 fields using
    `<=` operators throughout passes the gate (no violation)."""
    lane = _representation_lane(extra={
        "notes": (
            "archive_grammar=monolithic_0_bin "
            "parser_section_manifest=parser.json "
            "inflate_runtime_loc_budget<=100 "
            "runtime_dep_closure=brotli "
            "export_format=fp4a "
            "score_aware_loss=score-domain "
            "bolt_on_loc_budget<=350 "
            "no_op_detector_planned=true"
        ),
    })
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], (
        "Lane declaring all 8 fields with mix of `=` and `<=` separators "
        "must not violate Catalog #124."
    )


def test_cool_chic_wavelet_style_notes_no_false_positive(tmp_path: Path) -> None:
    """Cool-Chic / wavelet substrate-style notes (with `<=`) must pass
    when all 8 fields are declared. This is the exact ZZZZZ scenario:
    `inflate_runtime_loc_budget<=100 LOC` was previously rejected."""
    cool_chic_notes = (
        "research_only=true; substrate_engineering exception per HNeRV L7; "
        "archive_grammar=CCV1 monolithic single-file 0.bin fixed offsets; "
        "parser_section_manifest=parse_archive() returns 5 tuple; "
        "inflate_runtime_loc_budget<=100 LOC; "
        "runtime_dep_closure=torch+brotli; "
        "export_format=brotli(state_dicts)+int16(latents); "
        "score_aware_loss=alpha*B/N+beta*d_seg+gamma*sqrt(d_pose); "
        "bolt_on_loc_budget=~530 LOC (substrate_engineering tag); "
        "no_op_detector_planned=Catalog #139 byte-mutation smoke"
    )
    lane = _representation_lane(
        lid="lane_substrate_cool_chic_20260512",
        name="Cool-Chic substrate scaffold",
        extra={"notes": cool_chic_notes},
    )
    # Note: lane is research_only=true in the live registry which exempts
    # it via _NON_REPRESENTATION_LANE_CLASSES or research_only check. Here
    # we simulate the post-promotion case (research_only stripped) to
    # confirm the matcher accepts the `<=` declaration.
    repo = _make_repo(tmp_path, [lane])
    violations = check_representation_lane_has_archive_grammar_at_design_time(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == [], (
        "Cool-Chic style notes with `inflate_runtime_loc_budget<=100 LOC` "
        "must pass; the original matcher would have flagged this lane "
        "as missing `inflate_runtime_loc_budget` because it only matched "
        "`=` / `:` separators."
    )
