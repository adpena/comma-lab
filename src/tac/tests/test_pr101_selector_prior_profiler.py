# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "pr101_selector_prior_profiler.py"


def _load_tool():
    for path in (REPO_ROOT, REPO_ROOT / "tools", REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("pr101_selector_prior_profiler", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _encode_fec6_codes(tool, codes: list[int]) -> bytes:
    bits = "".join(tool.FEC6_FIXED_K16_CODE_BITS[code] for code in codes)
    out = bytearray((len(bits) + 7) // 8)
    for bit_pos, bit in enumerate(bits):
        if bit == "1":
            out[bit_pos // 8] |= 1 << (7 - (bit_pos % 8))
    return bytes(out)


def _synthetic_tables(tool, target_codes: list[int], *, penalty: float = 0.01):
    component = []
    pose = []
    seg = []
    missing = []
    for target in target_codes:
        component.append(
            tuple(0.05 if code == target else 0.05 + penalty for code in range(16))
        )
        pose.append(tuple(0.00001 for _ in range(16)))
        seg.append(tuple(0.0005 for _ in range(16)))
        missing.append(tuple(False for _ in range(16)))
    return tool.ComponentTables(
        component_score=tuple(component),
        pose=tuple(pose),
        seg=tuple(seg),
        missing_rows=tuple(missing),
        pair_rows_loaded=len(target_codes) * 16,
        pair_rows_paths=("synthetic.jsonl",),
    )


def test_candidate_with_missing_component_row_cannot_pass_proxy_gate() -> None:
    tool = _load_tool()
    target_codes = [0, 1, 2, 3]
    tables = _synthetic_tables(tool, target_codes, penalty=0.0)
    missing = [list(row) for row in tables.missing_rows]
    missing[0][0] = True
    tables = tool.ComponentTables(
        component_score=tables.component_score,
        pose=tables.pose,
        seg=tables.seg,
        missing_rows=tuple(tuple(row) for row in missing),
        pair_rows_loaded=tables.pair_rows_loaded - 1,
        pair_rows_paths=tables.pair_rows_paths,
    )
    candidate = tool.RuleCandidate(
        family="test",
        name="missing_row_candidate",
        predictions=tuple(target_codes),
        params={},
        source_literal_bytes_estimate=1,
        learned_source_values=1,
    )

    row = tool.evaluate_candidate(
        candidate,
        target_codes=target_codes,
        target_component=tool.summarize_predictions(target_codes, tables)[
            "component_score_no_rate_proxy"
        ],
        tables=tables,
        cpu_score=0.1920513168811056,
        threshold=0.192,
        fec6_archive_bytes=1_000,
        source_archive_bytes=700,
    )

    assert row["component_risk"]["missing_pair_mode_rows"] == 1
    assert row["score_estimate"]["component_rows_complete_for_prediction"] is False
    assert row["score_estimate"]["proxy_allows_sub0192_gate"] is False
    assert row["verdict"] == "blocked_by_missing_pair_mode_rows"


def test_build_profile_rejects_non_cpu_eval_axis(tmp_path: Path) -> None:
    tool = _load_tool()
    bad_eval = tmp_path / "contest_cuda.json"
    bad_eval.write_text(
        '{"score_axis": "contest_cuda", "canonical_score": 0.19}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[contest-CPU\]"):
        tool.build_profile(cpu_eval=bad_eval)


def test_fec6_huffman_decoder_round_trips_synthetic_codes() -> None:
    tool = _load_tool()
    codes = [0, 2, 7, 13, 15, 1, 0, 14]

    decoded, used_bits = tool.decode_fec6_fixed_huffman_codes(
        _encode_fec6_codes(tool, codes), n_pairs=len(codes)
    )

    assert decoded == codes
    assert used_bits == sum(len(tool.FEC6_FIXED_K16_CODE_BITS[code]) for code in codes)


def test_periodic_rule_family_can_pass_when_sequence_is_really_periodic() -> None:
    tool = _load_tool()
    target_codes = [0, 2, 7, 13] * 8
    tables = _synthetic_tables(tool, target_codes)
    target_component = tool.summarize_predictions(target_codes, tables)[
        "component_score_no_rate_proxy"
    ]
    candidates = tool.generate_rule_candidates(target_codes, max_period=8, max_buckets=8)
    periodic_p4 = next(candidate for candidate in candidates if candidate.name == "periodic_p4")

    row = tool.evaluate_candidate(
        periodic_p4,
        target_codes=target_codes,
        target_component=target_component,
        tables=tables,
        cpu_score=0.1920513168811056,
        threshold=0.192,
        fec6_archive_bytes=1_000,
        source_archive_bytes=700,
    )

    assert row["selector_match"]["exact_match"] is True
    assert row["score_estimate"]["archive_saved_bytes_vs_fec6"] == 300
    assert row["score_estimate"]["proxy_allows_sub0192_gate"] is True
    assert row["source_param_accounting"]["risk"]["level"] in {"low", "medium"}


def test_large_source_selector_table_is_not_compliance_clean() -> None:
    tool = _load_tool()

    risk = tool.source_embedding_risk(
        family="periodic_table",
        learned_source_values=600,
        n_pairs=600,
        exact_selector_match=True,
    )

    assert risk["level"] == "forbidden"
    assert "per-pair" in risk["reason"]


def test_real_pr101_fec6_selector_prior_profile_blocks_simple_rules() -> None:
    tool = _load_tool()
    required = [
        tool.DEFAULT_FEC6_ARCHIVE,
        tool.DEFAULT_FEC6_MANIFEST,
        tool.DEFAULT_CPU_EVAL,
        tool.DEFAULT_SOURCE_ARCHIVE,
    ]
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        pytest.skip("missing local PR101/FEC6 artifacts: " + ", ".join(map(str, missing)))

    profile = tool.build_profile(max_period=16, max_buckets=16, top_k=5)

    assert profile["exact_cpu_reference"][
        "required_same_component_saving_bytes_for_sub0192"
    ] == 78
    assert profile["source_rule_byte_accounting"]["archive_saved_bytes_vs_fec6"] == 259
    assert profile["conclusion"]["any_rule_plausibly_beats_sub0192"] is False
    assert profile["conclusion"]["best_estimated_cpu_score"] > 0.192
