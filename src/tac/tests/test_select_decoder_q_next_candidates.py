from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "select_decoder_q_next_candidates.py"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))


def _load_module():
    spec = importlib.util.spec_from_file_location("select_decoder_q_next_candidates", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _fixed_row(candidate_id: str, delta: int) -> dict:
    return {
        "mutation_id": candidate_id,
        "fixed_length_runtime_compatible": True,
        "mutation": {"tensor_name": "latent", "q_offset": 7, "delta": delta},
        "op3v3_target_evidence": {"score_impact_abs_sum": 0.01},
    }


def _advisory_row(candidate_id: str, delta: int, score: float, *, strict: bool) -> dict:
    authority = _false_authority() if strict else {}
    advisory_authority = _false_authority() if strict else {}
    return {
        "candidate_id": candidate_id,
        **authority,
        "mutation_manifest": {
            "archive_sha256": "a" * 64,
            "runtime_tree_sha256": "b" * 64,
            "mutation_row": _fixed_row(candidate_id, delta),
        },
        "advisory_eval": {
            **advisory_authority,
            "returncode": 0,
            "canonical_score": score,
            "score_axis": "macos_cpu_advisory",
            "evidence_grade": "[macOS-CPU advisory]",
            "archive_sha256": "a" * 64,
            "runtime_tree_sha256": "b" * 64,
            "n_samples": 600,
        },
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_decoder_q_selector_skips_custody_free_advisory_scores(tmp_path: Path) -> None:
    module = _load_module()
    feasibility = _write_json(
        tmp_path / "feasibility.json",
        {
            "fixed_length_runtime_compatible_rows": [
                _fixed_row("cand_neg", -1),
                _fixed_row("cand_pos", 1),
            ]
        },
    )
    advisory_summary = _write_json(
        tmp_path / "advisory.json",
        {
            "candidates": [
                _advisory_row("cand_neg", -1, 0.20, strict=False),
                _advisory_row("cand_pos", 1, 0.10, strict=False),
            ]
        },
    )

    selection = module.build_selection(
        argparse.Namespace(
            feasibility=feasibility,
            advisory_summary=advisory_summary,
            baseline_score=0.2,
            limit=8,
        )
    )

    assert selection["summary"]["advisory_candidate_count"] == 0
    assert selection["summary"]["skipped_advisory_candidate_count"] == 2
    assert selection["summary"]["signed_slope_model_count"] == 0
    assert all(row["reason"] == "unmeasured_fixed_length_candidate" for row in selection["queue"])


def test_decoder_q_selector_uses_strict_false_authority_advisory_rows(
    tmp_path: Path,
) -> None:
    module = _load_module()
    feasibility = _write_json(
        tmp_path / "feasibility.json",
        {
            "fixed_length_runtime_compatible_rows": [
                _fixed_row("cand_neg", -1),
                _fixed_row("cand_pos", 1),
                _fixed_row("cand_pos2", 2),
            ]
        },
    )
    advisory_summary = _write_json(
        tmp_path / "advisory.json",
        {
            "candidates": [
                _advisory_row("cand_neg", -1, 0.20, strict=True),
                _advisory_row("cand_pos", 1, 0.10, strict=True),
            ]
        },
    )

    selection = module.build_selection(
        argparse.Namespace(
            feasibility=feasibility,
            advisory_summary=advisory_summary,
            baseline_score=0.2,
            limit=8,
        )
    )

    assert selection["summary"]["advisory_candidate_count"] == 2
    assert selection["summary"]["skipped_advisory_candidate_count"] == 0
    assert selection["summary"]["signed_slope_model_count"] == 1
    assert selection["queue"][0]["candidate_id"] == "cand_pos2"
    assert selection["queue"][0]["reason"] == "signed_slope_preferred_direction"
