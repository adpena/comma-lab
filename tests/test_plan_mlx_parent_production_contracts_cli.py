from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "plan_mlx_parent_production_contracts.py"


def _hash(char: str) -> str:
    return char * 64


def _dataset() -> dict:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    cache = {
        "pair_indices": _hash("0"),
        "posenet_yuv6_pair": _hash("1"),
        "segnet_last_rgb": _hash("2"),
    }
    return {
        "schema": "scorer_response_dataset.v1",
        "producer": "test",
        **false_authority,
        "authority": dict(false_authority),
        "rows": [
            {
                "schema": "scorer_response_row.v1",
                "row_id": "mlx-row-1",
                "family": "mlx_scorer_response",
                "archive_sha256": _hash("a"),
                "raw_sha256": _hash("r"),
                "source_inflated_outputs_aggregate_sha256": _hash("d"),
                "source_batch_pairs": 1,
                "source_pair_window": [0, 1],
                "source_candidate_cache_array_sha256": cache,
                "source_reference_cache_array_sha256": cache,
                "authority_source_score_claim": False,
                **false_authority,
            }
        ],
    }


def _cache_auth_audit(*, archive_sha256: str = "a") -> dict:
    auth_hashes = {
        "pair_indices": _hash("0"),
        "posenet_yuv6_pair": _hash("3"),
        "segnet_last_rgb": _hash("4"),
    }
    return {
        "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
        "verdict": "FAIL_CACHE_AUTH_EVAL_IDENTITY",
        "passed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cache": {
            "archive_sha256": _hash(archive_sha256),
            "inflated_outputs_aggregate_sha256": _hash("d"),
            "raw_sha256": _hash("r"),
            "array_sha256": {
                "pair_indices": _hash("0"),
                "posenet_yuv6_pair": _hash("1"),
                "segnet_last_rgb": _hash("2"),
            },
        },
        "auth_eval": {
            "archive_sha256": _hash(archive_sha256),
            "inflated_outputs_aggregate_sha256": _hash("e"),
            "raw_file_sha256": _hash("s"),
            "n_samples": 600,
            "scorer_input_hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
            "scorer_input_array_sha256": auth_hashes,
        },
    }


def test_plan_mlx_parent_production_contracts_cli_auth_cache_audit_aliases(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    audit_a = tmp_path / "audit_a.json"
    audit_b = tmp_path / "audit_b.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset.write_text(json.dumps(_dataset()), encoding="utf-8")
    audit_a.write_text(json.dumps(_cache_auth_audit()), encoding="utf-8")
    audit_b.write_text(
        json.dumps(_cache_auth_audit(archive_sha256="b")),
        encoding="utf-8",
    )

    blocked = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            str(dataset),
            "--auth-cache-audit",
            str(audit_a),
            "--cache-auth-audit",
            str(audit_b),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode == 2
    assert json_out.is_file()
    assert md_out.is_file()

    allowed = subprocess.run(
        [
            *blocked.args,
            "--allow-blocked-output",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert allowed.returncode == 0
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["summary"]["cache_auth_audit_count"] == 2
    assert plan["summary"]["cache_auth_audit_mismatched_group_count"] == 1
    assert any(
        blocker.endswith(":segnet_last_rgb") for blocker in plan["blockers"]
    )
