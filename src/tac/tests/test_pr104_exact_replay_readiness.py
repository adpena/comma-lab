from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
READINESS = REPO / ".omx/research/pr104_exact_replay_readiness_20260508_codex.json"
ADAPTER = REPO / "experiments/public_runtime_adapters/pr104_qhnerv_ft_best_adapter/inflate.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readiness() -> dict:
    return json.loads(READINESS.read_text(encoding="utf-8"))


def test_pr104_readiness_pins_current_adapter_and_archive_identity() -> None:
    payload = _readiness()
    adapter = payload["adapter"]
    runtime = payload["runtime"]
    archive = payload["archive"]

    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["gpu_dispatch_attempted"] is False
    assert payload["ready_for_exact_eval_dispatch"] is True
    assert adapter["inflate_sh_sha256"] == _sha256(ADAPTER)
    assert runtime["inflate_sh"] == adapter["path"]
    assert runtime["inflate_sh_sha256"] == adapter["inflate_sh_sha256"]
    assert runtime["runtime_manifest"]["external_dependency_roots"]
    assert os.access(ADAPTER, os.X_OK)
    assert archive["bytes"] == 178637
    assert (
        archive["sha256"]
        == "6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8"
    )
    assert archive["members"] == [
        {
            "name": "0.bin",
            "file_size": 178529,
            "compress_size": 178529,
            "crc32": "d08e0c9e",
            "sha256": "0a0f2cac1961f3ab5128e70ff10c0287a66949d435b5ee7a4dcf2017917910a3",
        }
    ]


def test_pr104_adapter_is_source_sized_and_fail_closed() -> None:
    payload = _readiness()
    text = ADAPTER.read_text(encoding="utf-8")

    assert "PACT_RUNTIME_DEPENDENCY_ROOT=" in text
    assert "pip install" not in text
    assert "curl " not in text
    assert "wget " not in text
    assert "import score" not in text
    assert "score.py" not in text
    assert "ambiguous PR104 payload members" in text
    assert "neither ${BASE_BIN} nor ${X_MEMBER} exists" in text
    assert payload["compliance"]["adapter_contains_payload_bytes"] is False
    assert payload["compliance"]["no_scorer_loads_in_inflate"] is True
    assert payload["compliance"]["no_network_install_in_inflate"] is True
    assert payload["runtime_source"]["external_runtime_dependency_marked_by_pact_directive"] is True


def test_pr104_readiness_keeps_score_promotion_blocked_until_cuda_auth_eval() -> None:
    payload = _readiness()

    assert payload["blockers"] == []
    assert payload["score_promotion_blockers"] == [
        "exact_cuda_auth_eval_not_run_by_design_for_this_task"
    ]
    assert (
        payload["public_report"]["score_truth_note"]
        == "external public report only; CUDA auth eval through this checkout is required before promotion, ranking, or kill decisions"
    )
    assert payload["local_smokes"]["cpu_safe_archive_parse_and_model_load"]["status"] == "pass"
