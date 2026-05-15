# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "pr101_fec6_sub0192_cpu_component_selector.py"


def _load_tool():
    for path in (REPO_ROOT, REPO_ROOT / "tools"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("pr101_fec6_sub0192_cpu_component_selector", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reference():
    tool = _load_tool()
    return tool.AxisReference(
        threshold=0.192,
        cpu_score=0.1920513168811056,
        cpu_score_axis="contest_cpu",
        cuda_score=0.22621002169349796,
        cuda_score_axis="contest_cuda",
        archive_bytes=178_517,
        archive_sha256="fec6sha",
        avg_segnet_dist=0.00056029,
        avg_posenet_dist=0.00002943,
        score_rate_contribution=0.11886714273451067,
        effective_selector_policy_sha256="fec6policy",
        selector_payload_sha256="fec6payload",
        selector_payload_bytes=249,
        proxy_charged_score=0.19206196565708117,
        proxy_uncharged_score=0.19188950818822254,
    )


def test_required_saving_bytes_is_strict_for_pr101_fec6_cpu_gate() -> None:
    tool = _load_tool()

    needed = tool.required_saving_bytes_for_strict_gate(0.1920513168811056, 0.192)

    assert needed == 78
    assert tool.score_after_byte_delta(0.1920513168811056, -78) < 0.192
    assert tool.score_after_byte_delta(0.1920513168811056, -77) >= 0.192


def test_rate_only_same_selector_must_save_required_bytes() -> None:
    tool = _load_tool()
    manifest = {
        "archive": {
            "bytes": 178_477,
            "sha256": "rate-only-sha",
            "selector_pack_manifest": {
                "selector_payload_sha256": "other-payload",
                "selector_payload_bytes": 209,
                "selector_wire_format": "synthetic_rate_only",
            },
        },
        "selector": {
            "effective_selector_policy_sha256": "fec6policy",
            "selected_non_none_pairs": 466,
        },
        "proxy": {},
    }

    row = tool.classify_candidate_manifest(_reference(), Path("synthetic.json"), manifest)

    assert row["selector"]["policy_kind"] == "rate_only_same_decoded_selector_policy"
    assert row["archive"]["saved_bytes_vs_fec6"] == 40
    assert row["rate_only_gate"]["passes_sub0192_if_components_unchanged"] is False
    assert row["verdict"] == "rate_only_not_enough_bytes"


def test_component_moving_k4_shape_can_clear_rate_but_fail_proxy_allowance() -> None:
    tool = _load_tool()
    manifest = {
        "archive": {
            "bytes": 178_434,
            "sha256": "k4sha",
            "selector_pack_manifest": {
                "selector_payload_sha256": "k4payload",
                "selector_payload_bytes": 166,
                "selector_wire_format": "FEC3_archive_charged_static_or_dynamic_compact_palette",
            },
        },
        "selector": {
            "effective_selector_policy_sha256": "k4policy",
            "selected_non_none_pairs": 310,
            "histogram": {"none": 290},
        },
        "proxy": {
            "selector_score_proxy_charged_formula": 0.19214933798933803,
            "selector_score_proxy_uncharged_formula": 0.19203214681358854,
            "evidence_grade": "MPS-research-signal",
        },
    }

    row = tool.classify_candidate_manifest(_reference(), Path("k4.json"), manifest)

    assert row["selector"]["policy_kind"] == "component_moving_selector_policy"
    assert row["archive"]["saved_bytes_vs_fec6"] == 83
    assert row["rate_only_gate"]["passes_sub0192_if_components_unchanged"] is True
    assert row["component_moving_evidence"]["component_delta_within_rate_allowance"] is False
    assert row["component_moving_evidence"]["proxy_allows_sub0192_gate"] is False
    assert row["verdict"] == "component_moving_rate_feasible_proxy_blocks_gate"
