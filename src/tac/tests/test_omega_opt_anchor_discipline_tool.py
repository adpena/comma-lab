from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tac.codec_stack_planner import build_hstack_vstack_multipass_plan
from tac.omega_opt_claims import omega_opt_claim_rows

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "check_omega_opt_anchor_discipline.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("_omega_opt_anchor_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_repo(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    ledger = tmp_path / ".omx/research/omega_opt_anchor_discipline_20260508_codex.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        "\n".join(
            " | ".join(
                [
                    row["claim_id"],
                    "Next 1:1 test",
                    "score_claim=false",
                    "promotion_eligible=false",
                    "rank_or_kill_eligible=false",
                    "ready_for_exact_eval_dispatch=false",
                ]
            )
            for row in omega_opt_claim_rows()
        ),
        encoding="utf-8",
    )

    reports = tmp_path / "reports"
    reports.mkdir()
    evidence = reports / "cathedral_autopilot_evidence.jsonl"
    evidence.write_text("", encoding="utf-8")

    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    registry = state / "lane_registry.json"
    registry.write_text(json.dumps({"lanes": []}), encoding="utf-8")
    return tmp_path, ledger, evidence, registry


def test_scanner_accepts_fail_closed_proxy_omega_evidence(tmp_path: Path) -> None:
    mod = _load_tool()
    repo, ledger, evidence, registry = _write_minimal_repo(tmp_path)
    evidence.write_text(
        json.dumps({
            "technique": "omega_opt_linear_stack_post_hoc_composition",
            "evidence_grade": "[CPU-prep empirical Omega-OPT byte-anchor]",
            "evidence_semantics": "cpu_omega_opt_linear_stack_byte_composition_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "source": "reports/raw/omega/manifest.json",
        })
        + "\n",
        encoding="utf-8",
    )

    findings = mod.scan(
        repo_root=repo,
        ledger=ledger,
        evidence_jsonl=evidence,
        lane_registry=registry,
    )

    assert findings == []


def test_scanner_rejects_unanchored_true_score_flag(tmp_path: Path) -> None:
    mod = _load_tool()
    repo, ledger, evidence, registry = _write_minimal_repo(tmp_path)
    evidence.write_text(
        json.dumps({
            "technique": "omega_opt_bilevel_optimization",
            "evidence_grade": "prediction",
            "score_claim": True,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "source": "Omega-OPT predicted 0.100, no exact eval",
        })
        + "\n",
        encoding="utf-8",
    )

    findings = mod.scan(
        repo_root=repo,
        ledger=ledger,
        evidence_jsonl=evidence,
        lane_registry=registry,
    )

    assert [finding.reason for finding in findings] == [
        "score_claim_must_be_false_without_exact_1to1_anchor"
    ]


def test_scanner_validates_generated_codec_stack_claim_table(tmp_path: Path) -> None:
    mod = _load_tool()
    repo, ledger, evidence, registry = _write_minimal_repo(tmp_path)
    manifest_path = tmp_path / "reports/hstack_plan.json"
    manifest_path.write_text(
        json.dumps(build_hstack_vstack_multipass_plan().to_manifest()),
        encoding="utf-8",
    )

    findings = mod.scan(
        repo_root=repo,
        ledger=ledger,
        evidence_jsonl=evidence,
        lane_registry=registry,
        plan_manifest=manifest_path,
    )

    assert findings == []


def test_scanner_blocks_lane_registry_promotion_without_exact_anchor(tmp_path: Path) -> None:
    mod = _load_tool()
    repo, ledger, evidence, registry = _write_minimal_repo(tmp_path)
    registry.write_text(
        json.dumps({
            "lanes": [
                {
                    "id": "lane_omega_opt_joint_admm",
                    "level": 1,
                    "notes": "Predicted score 0.105 [predicted]",
                    "gates": {
                        "impl_complete": {
                            "status": True,
                            "evidence": "synthetic planning note",
                        },
                    },
                }
            ]
        }),
        encoding="utf-8",
    )

    findings = mod.scan(
        repo_root=repo,
        ledger=ledger,
        evidence_jsonl=evidence,
        lane_registry=registry,
    )

    assert [finding.reason for finding in findings] == [
        "lane_registry_gate_or_level_promoted_without_exact_1to1_anchor"
    ]
