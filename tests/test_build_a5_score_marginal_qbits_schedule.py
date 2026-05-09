from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tac.repo_io import json_text

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "build_a5_score_marginal_qbits_schedule.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_a5_score_marginal_qbits_schedule",
        TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path) -> Path:
    path.write_text(
        json_text(
            {
                "schema": "pr101_a5_per_pair_score_marginals.v1",
                "score_claim": False,
                "n_pairs": 4,
                "per_pair_score_marginals": [0.1, 0.9, 0.2, 0.8],
                "per_pair_q_bits": [2, 8, 3, 8],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builds_low_marginal_trust_region_schedule(tmp_path: Path) -> None:
    tool = _load_tool()
    source = _write_manifest(tmp_path / "score_marginals.json")

    payload = tool.build_schedule(
        score_marginal_manifest_path=source,
        base_q_bits=8,
        low_q_bits=6,
        low_fraction=0.5,
        latent_dim=4,
        repo_root=tmp_path,
    )

    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["low_pair_count"] == 2
    assert payload["per_pair_q_bits"] == [6, 8, 6, 8]
    assert payload["q_bits_summary"]["unique_counts"] == {"6": 2, "8": 2}
    assert payload["raw_latent_payload_bits"] == 112
    assert payload["alignment"]["q_bits_vs_score_marginal_pearson"] > 0.9


def test_cli_writes_tool_run_manifest(tmp_path: Path) -> None:
    tool = _load_tool()
    source = _write_manifest(tmp_path / "score_marginals.json")
    out = tmp_path / "schedule.json"

    assert (
        tool.main(
            [
                "--score-marginal-manifest",
                str(source),
                "--json-out",
                str(out),
                "--base-q-bits",
                "8",
                "--low-q-bits",
                "6",
                "--low-fraction",
                "0.5",
                "--latent-dim",
                "4",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tool_run_manifest"]["tool"] == (
        "tools/build_a5_score_marginal_qbits_schedule.py"
    )
    assert payload["per_pair_q_bits"] == [6, 8, 6, 8]
