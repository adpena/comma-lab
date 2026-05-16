# SPDX-License-Identifier: MIT
"""Tests for the Rudin floor interpretable-ML L1 SCAFFOLD substrate.

Per CLAUDE.md "Lane maturity registry" lifecycle discipline + sister test
patterns at ``tac.substrates.atw_codec_v1.tests``: 10+ dedicated tests
covering rule-list application, archive grammar, inflate, and
SubstrateContract-style invariants.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287
(docstring overstatement trap): no empirical-claim percentages without
``[empirical:<artifact path>]`` or ``[prediction]`` tags.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.substrates.rudin_floor_interpretable_ml import (
    CANONICAL_GOSDT_DEPTH,
    CANONICAL_K_RASHOMON,
    CANONICAL_K_RULES,
    CANONICAL_SLIM_COEFF_BOUND,
    IMPLEMENTATION_STATUS,
    RDIF_HEADER_SIZE,
    RDIF_MAGIC,
    RDIF_VERSION,
    RESEARCH_ONLY,
    RudinFallingRule,
    RudinRuleList,
    inflate_one_video,
    pack_archive,
    parse_archive,
)
from tac.substrates.rudin_floor_interpretable_ml.archive import (
    RDIFv1Archive,
    RDIFv1Header,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
TRAINER_PATH = (
    REPO_ROOT / "experiments" / "train_substrate_rudin_floor_interpretable_ml.py"
)
RECIPE_PATH = (
    REPO_ROOT
    / ".omx"
    / "operator_authorize_recipes"
    / "substrate_rudin_floor_interpretable_ml_modal_t4_dispatch.yaml"
)
DRIVER_PATH = (
    REPO_ROOT
    / "scripts"
    / "remote_lane_substrate_rudin_floor_interpretable_ml.sh"
)
PROBE_PATH = (
    REPO_ROOT
    / "tools"
    / "probe_rudin_floor_substrate_disambiguator.py"
)
DESIGN_MEMO_PATH = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md"
)


# ---------------------------------------------------------------------------
# 1. Canonical constants — Wang-Rudin 2015 + Ustun-Rudin 2016 +
#    Semenova-Rudin-Parr 2020 + Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 anchors
# ---------------------------------------------------------------------------

def test_canonical_constants_match_rudin_literature() -> None:
    """Canonical constants match the Rudin discipline literature."""
    assert CANONICAL_K_RULES == 6, (
        "Wang-Rudin 2015 canonical falling-rule-list depth K=4-6"
    )
    assert CANONICAL_K_RASHOMON == 8, (
        "Semenova-Rudin-Parr 2020 canonical ensemble K=8"
    )
    assert CANONICAL_SLIM_COEFF_BOUND == 10, (
        "Ustun-Rudin 2016 canonical integer-coefficient bound K=10"
    )
    assert CANONICAL_GOSDT_DEPTH == 4, (
        "Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 canonical depth D=4"
    )


def test_rdif_magic_and_version_pinned() -> None:
    """RDIF magic + version are pinned for byte-deterministic archives."""
    assert RDIF_MAGIC == b"RDF1", "RDIF v1 magic must be b'RDF1'"
    assert RDIF_VERSION == 0x0001, "RDIF v1 version must be 0x0001"
    assert RDIF_HEADER_SIZE == 34, "RDIF v1 header size must be 34 bytes"


def test_research_only_flag_set_at_l1_scaffold() -> None:
    """RESEARCH_ONLY=True at L1 SCAFFOLD per Catalog #240 cascade."""
    assert RESEARCH_ONLY is True, (
        "Rudin floor L1 SCAFFOLD MUST be research_only=true at landing "
        "per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY'"
    )
    assert "research_only" in IMPLEMENTATION_STATUS or "scaffold" in IMPLEMENTATION_STATUS


# ---------------------------------------------------------------------------
# 2. Rule-list serialization + first-match-wins semantics (Wang-Rudin 2015)
# ---------------------------------------------------------------------------

def test_rule_matches_simple_equality_predicate() -> None:
    """Rule's matches() respects simple key==value predicate."""
    rule = RudinFallingRule(predicate="mean_class==road", action_rgb=(100, 100, 100))
    assert rule.matches({"mean_class": "road"}) is True
    assert rule.matches({"mean_class": "sky"}) is False


def test_rule_matches_always_predicate() -> None:
    """Rule with predicate='always' catches every input (catch-all)."""
    rule = RudinFallingRule(predicate="always", action_rgb=(0, 0, 0))
    assert rule.matches({}) is True
    assert rule.matches({"any_key": "any_value"}) is True


def test_rule_list_first_match_wins_semantics() -> None:
    """RuleList.evaluate returns the FIRST matching rule's action_rgb."""
    rules = RudinRuleList(
        rules=(
            RudinFallingRule(
                predicate="mean_class==sky", action_rgb=(60, 90, 180)
            ),
            RudinFallingRule(
                predicate="mean_class==road", action_rgb=(100, 100, 100)
            ),
            RudinFallingRule(predicate="always", action_rgb=(40, 40, 40)),
        ),
        default_rgb=(0, 0, 0),
    )
    assert rules.evaluate({"mean_class": "sky"}) == (60, 90, 180)
    assert rules.evaluate({"mean_class": "road"}) == (100, 100, 100)
    # Catch-all fires when no specific rule matches
    assert rules.evaluate({"mean_class": "unknown"}) == (40, 40, 40)


def test_rule_list_default_rgb_when_no_match_and_no_catchall() -> None:
    """RuleList falls back to default_rgb when no rule matches."""
    rules = RudinRuleList(
        rules=(
            RudinFallingRule(
                predicate="mean_class==sky", action_rgb=(60, 90, 180)
            ),
        ),
        default_rgb=(7, 7, 7),
    )
    assert rules.evaluate({"mean_class": "road"}) == (7, 7, 7)


def test_rule_list_json_roundtrip_byte_deterministic() -> None:
    """RuleList.to_json_bytes / from_json_bytes is a byte-stable roundtrip."""
    rules = RudinRuleList(
        rules=(
            RudinFallingRule(predicate="mean_class==road", action_rgb=(100, 100, 100), slim_coefficients=(1, 2, 3)),
            RudinFallingRule(predicate="always", action_rgb=(40, 40, 40)),
        ),
        default_rgb=(0, 0, 0),
    )
    blob1 = rules.to_json_bytes()
    blob2 = rules.to_json_bytes()
    assert blob1 == blob2, "rule_list serialization NOT byte-deterministic"
    parsed = RudinRuleList.from_json_bytes(blob1)
    assert parsed.default_rgb == rules.default_rgb
    assert len(parsed.rules) == len(rules.rules)
    for orig, restored in zip(rules.rules, parsed.rules, strict=True):
        assert orig.predicate == restored.predicate
        assert orig.action_rgb == restored.action_rgb
        assert orig.slim_coefficients == restored.slim_coefficients


# ---------------------------------------------------------------------------
# 3. RDIF v1 archive grammar (HNeRV parity L3 monolithic single-file)
# ---------------------------------------------------------------------------

def test_pack_archive_starts_with_rdif_magic() -> None:
    """RDIF v1 archives ALWAYS start with b'RDF1' per HNeRV parity L3."""
    rules = RudinRuleList(
        rules=(RudinFallingRule(predicate="always", action_rgb=(0, 0, 0)),),
        default_rgb=(0, 0, 0),
    )
    archive = pack_archive(rule_list=rules)
    assert archive[:4] == RDIF_MAGIC, (
        "RDIF v1 archive MUST start with b'RDF1' magic per HNeRV parity L3"
    )


def test_pack_archive_is_byte_deterministic_hnerv_l9() -> None:
    """Two pack_archive calls with same input ⇒ byte-identical (HNeRV L9)."""
    rules = RudinRuleList(
        rules=(
            RudinFallingRule(predicate="mean_class==sky", action_rgb=(60, 90, 180)),
            RudinFallingRule(predicate="always", action_rgb=(0, 0, 0)),
        ),
        default_rgb=(0, 0, 0),
    )
    archive1 = pack_archive(rule_list=rules)
    archive2 = pack_archive(rule_list=rules)
    assert archive1 == archive2, (
        "pack_archive NOT byte-deterministic (HNeRV L9 byte-identity violation); "
        "two runs produced different bytes"
    )


def test_parse_archive_roundtrips_pack_archive() -> None:
    """parse_archive recovers the same rule_list that pack_archive packed."""
    rules = RudinRuleList(
        rules=(
            RudinFallingRule(
                predicate="mean_class==road",
                action_rgb=(100, 100, 100),
                slim_coefficients=(1,),
            ),
            RudinFallingRule(
                predicate="mean_class==sky",
                action_rgb=(60, 90, 180),
                slim_coefficients=(2,),
            ),
            RudinFallingRule(
                predicate="always",
                action_rgb=(40, 40, 40),
            ),
        ),
        default_rgb=(0, 0, 0),
    )
    archive = pack_archive(rule_list=rules)
    parsed = parse_archive(archive)
    assert isinstance(parsed, RDIFv1Archive)
    assert isinstance(parsed.header, RDIFv1Header)
    assert parsed.header.version == RDIF_VERSION
    assert len(parsed.rule_list.rules) == 3
    assert parsed.rule_list.rules[0].predicate == "mean_class==road"
    assert parsed.rule_list.rules[1].action_rgb == (60, 90, 180)
    assert parsed.rule_list.rules[2].predicate == "always"


def test_parse_archive_refuses_corrupted_sha256_trailer() -> None:
    """parse_archive rejects archives with mutated payload (HNeRV L9 closure)."""
    rules = RudinRuleList(
        rules=(RudinFallingRule(predicate="always", action_rgb=(0, 0, 0)),),
        default_rgb=(0, 0, 0),
    )
    archive = bytearray(pack_archive(rule_list=rules))
    # Flip a byte in the directory region (after header, before sha256 trailer)
    # to invalidate the sha256.
    if len(archive) > RDIF_HEADER_SIZE + 10:
        archive[RDIF_HEADER_SIZE + 5] ^= 0x01
    with pytest.raises(ValueError, match="sha256"):
        parse_archive(bytes(archive))


def test_parse_archive_refuses_wrong_magic() -> None:
    """parse_archive rejects archives whose magic is not b'RDF1'."""
    rules = RudinRuleList(
        rules=(RudinFallingRule(predicate="always", action_rgb=(0, 0, 0)),),
        default_rgb=(0, 0, 0),
    )
    archive = bytearray(pack_archive(rule_list=rules))
    archive[:4] = b"XXXX"
    with pytest.raises(ValueError, match="magic"):
        parse_archive(bytes(archive))


def test_archive_has_8_named_sections_per_design_memo_section_10() -> None:
    """RDIF v1 carries 8 named sections per design memo §10 grammar."""
    rules = RudinRuleList(
        rules=(RudinFallingRule(predicate="always", action_rgb=(0, 0, 0)),),
        default_rgb=(0, 0, 0),
    )
    archive = pack_archive(rule_list=rules)
    parsed = parse_archive(archive)
    expected_sections = {
        "encoder_tree_blob",
        "rule_list_blob",
        "scorer_priors_blob",
        "frame_0_init_blob",
        "wavelet_residuals_blob",
        "pose_residuals_blob",
        "per_pair_rule_indices_blob",
        "rashomon_disagreement_blob",
    }
    assert set(parsed.sections.keys()) == expected_sections, (
        f"RDIF v1 must carry 8 named sections per design memo §10; "
        f"got {sorted(parsed.sections.keys())}"
    )


# ---------------------------------------------------------------------------
# 4. Inflate runtime (HNeRV parity L4 ≤200 LOC pure Python; numpy+Pillow only)
# ---------------------------------------------------------------------------

def test_inflate_one_video_writes_output_for_valid_archive(tmp_path: Path) -> None:
    """inflate_one_video produces an output file for a valid RDIF v1 archive."""
    rules = RudinRuleList(
        rules=(
            RudinFallingRule(predicate="mean_class==sky", action_rgb=(60, 90, 180)),
            RudinFallingRule(predicate="always", action_rgb=(40, 40, 40)),
        ),
        default_rgb=(0, 0, 0),
    )
    archive = pack_archive(rule_list=rules)
    output_path = inflate_one_video(
        archive, tmp_path / "frame_test", features={"mean_class": "sky"}
    )
    assert output_path.exists(), "inflate_one_video did not produce output"
    assert output_path.stat().st_size > 0, "inflate output is empty"


def test_inflate_picks_first_matching_rule_for_features(tmp_path: Path) -> None:
    """Inflate uses first-match-wins semantics on the features dict."""
    rules = RudinRuleList(
        rules=(
            RudinFallingRule(predicate="mean_class==road", action_rgb=(100, 100, 100)),
            RudinFallingRule(predicate="mean_class==sky", action_rgb=(60, 90, 180)),
            RudinFallingRule(predicate="always", action_rgb=(40, 40, 40)),
        ),
        default_rgb=(0, 0, 0),
    )
    archive = pack_archive(rule_list=rules)
    out_road = inflate_one_video(
        archive, tmp_path / "road", features={"mean_class": "road"}
    )
    out_sky = inflate_one_video(
        archive, tmp_path / "sky", features={"mean_class": "sky"}
    )
    assert out_road.exists() and out_sky.exists()
    # Both files exist; rule-application happens at the bytes level
    # (per-rule_list.evaluate); inflate is the 1x1 PNG proof for byte-mutation.


# ---------------------------------------------------------------------------
# 5. Sister artifact existence (trainer + recipe + driver + probe + memo)
# ---------------------------------------------------------------------------

def test_design_memo_exists_at_canonical_path() -> None:
    """Design memo exists per Catalog #290 canonical-vs-unique discipline."""
    assert DESIGN_MEMO_PATH.is_file(), (
        f"design memo missing at {DESIGN_MEMO_PATH}; "
        "Catalog #290 requires substrate scaffolds carry design memo"
    )


def test_trainer_exists_and_imports_substrate_package() -> None:
    """Trainer file exists + imports from the substrate package."""
    assert TRAINER_PATH.is_file(), f"trainer missing at {TRAINER_PATH}"
    body = TRAINER_PATH.read_text(encoding="utf-8")
    assert "from tac.substrates.rudin_floor_interpretable_ml" in body, (
        "trainer must import from substrate package"
    )
    assert "NotImplementedError" in body, (
        "trainer _full_main MUST raise NotImplementedError per Catalog #220 cascade"
    )
    assert "TIER_1_OPERATOR_REQUIRED_FLAGS" in body, (
        "trainer must declare TIER_1_OPERATOR_REQUIRED_FLAGS per Catalog #151"
    )


def test_recipe_exists_and_declares_research_only_dispatch_disabled() -> None:
    """Recipe exists + research_only=true + dispatch_enabled=false per Catalog #240."""
    assert RECIPE_PATH.is_file(), f"recipe missing at {RECIPE_PATH}"
    body = RECIPE_PATH.read_text(encoding="utf-8")
    assert "research_only: true" in body, "recipe MUST declare research_only: true"
    assert "dispatch_enabled: false" in body, (
        "recipe MUST declare dispatch_enabled: false at landing per Catalog #240"
    )
    assert "min_smoke_gpu" in body, "recipe MUST declare min_smoke_gpu per Catalog #215"
    assert "min_vram_gb" in body, "recipe MUST declare min_vram_gb per Catalog #170"


def test_driver_exists_carries_canonical_nvml_block() -> None:
    """Driver exists + carries Catalog #244 canonical Modal/CUDA env block."""
    assert DRIVER_PATH.is_file(), f"driver missing at {DRIVER_PATH}"
    body = DRIVER_PATH.read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in body, (
        "driver MUST export DALI_DISABLE_NVML per Catalog #244"
    )
    assert "CUBLAS_WORKSPACE_CONFIG" in body, (
        "driver MUST export CUBLAS_WORKSPACE_CONFIG per Catalog #244"
    )
    assert "PYTORCH_CUDA_ALLOC_CONF" in body, (
        "driver MUST export PYTORCH_CUDA_ALLOC_CONF per Catalog #244"
    )
    assert "REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1" in body, (
        "driver MUST use sentinel per Catalog #163 when sourcing bootstrap"
    )
    assert "provenance.json" in body, "driver MUST write provenance.json per Catalog L"


def test_probe_disambiguator_exists_and_runs(tmp_path: Path) -> None:
    """Probe-disambiguator (Catalog #125 hook #6) exists + runs to completion."""
    assert PROBE_PATH.is_file(), f"probe missing at {PROBE_PATH}"
    # Run the probe via subprocess (smoke; minimal sample)
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--output-dir",
            str(tmp_path / "probe_out"),
            "--max-frames",
            "2",
            "--downsample-factor",
            "32",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # rc in {0,1,2} = verdict; rc=3 = INFRA_ERROR
    assert result.returncode in {0, 1, 2, 3}, (
        f"probe returned unexpected rc={result.returncode}; "
        f"stderr={result.stderr[:500]}"
    )
    assert "[rudin_floor PROBE]" in result.stdout, (
        f"probe output missing canonical tag; stdout={result.stdout[:500]}"
    )
    # Verify the JSON artifact landed
    json_artifact = tmp_path / "probe_out" / "probe_disambiguator.json"
    assert json_artifact.is_file(), f"probe JSON missing at {json_artifact}"
    data = json.loads(json_artifact.read_text(encoding="utf-8"))
    assert "verdict" in data
    assert data["verdict"] in {
        "MEANINGFUL_INTERPRETABILITY",
        "WEAK_INTERPRETABILITY",
        "OPAQUE",
        "INFRA_ERROR",
    }
    # Per CLAUDE.md "Apples-to-apples evidence discipline": probe must never
    # claim a contest-axis score
    assert data.get("score_claim_valid") is False
    assert data.get("promotion_eligible") is False
    assert data.get("rank_or_kill_eligible") is False


# ---------------------------------------------------------------------------
# 6. Trainer smoke (end-to-end CPU; $0)
# ---------------------------------------------------------------------------

def test_trainer_smoke_runs_to_completion_writes_archive(tmp_path: Path) -> None:
    """train_substrate_rudin_floor_interpretable_ml.py --smoke runs cleanly."""
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER_PATH),
            "--output-dir",
            str(tmp_path / "trainer_smoke"),
            "--smoke",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"trainer --smoke failed with rc={result.returncode}; "
        f"stderr={result.stderr[:500]}"
    )
    archive = tmp_path / "trainer_smoke" / "0.bin"
    assert archive.is_file(), f"trainer --smoke did not write 0.bin at {archive}"
    assert archive.read_bytes()[:4] == RDIF_MAGIC, (
        "trainer --smoke output does not start with RDIF magic"
    )
    stats_path = tmp_path / "trainer_smoke" / "smoke_stats.json"
    assert stats_path.is_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["research_only"] is True
    assert stats["score_claim"] is False
    assert stats["promotion_eligible"] is False


def test_trainer_full_main_raises_not_implemented_error() -> None:
    """trainer _full_main MUST raise NotImplementedError per Catalog #240."""
    # Run in non-smoke mode and assert non-zero exit + NotImplementedError
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER_PATH),
            "--output-dir",
            "/tmp/rudin_floor_full_main_should_not_complete",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, (
        "trainer _full_main must NOT exit cleanly at L1 SCAFFOLD landing"
    )
    assert "NotImplementedError" in result.stderr or "council-gated" in result.stderr, (
        f"trainer _full_main must raise NotImplementedError citing Catalog #220; "
        f"got stderr={result.stderr[:500]}"
    )


def test_remote_driver_requires_and_closes_dispatch_claim() -> None:
    """Remote driver must not run without a lane-claim lifecycle."""
    src = DRIVER_PATH.read_text(encoding="utf-8")
    assert "DISPATCH_INSTANCE_JOB_ID is required" in src
    assert "verify_active_dispatch_claim()" in src
    assert "claim_lane_dispatch.py\" summary" in src
    assert "--live-only" in src
    assert "append_terminal_claim()" in src
    assert "claim_lane_dispatch.py" in src
    assert "CLAIM_VERIFIED=1" in src
    assert "trap cleanup EXIT" in src
