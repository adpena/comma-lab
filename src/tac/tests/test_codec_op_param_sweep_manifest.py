"""Tests for ``tools.codec_op_param_sweep_manifest``.

The sweep manifest keeps rich CodecOp provenance, while the optional
meta-Lagrangian candidate output must stay in the strict schema accepted by
``MetaLagrangianSearch.evaluate_candidate(**candidate)``.
"""
from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "codec_op_param_sweep_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "codec_op_param_sweep_manifest", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_to_meta_lagrangian_candidates_emits_strict_evaluate_schema() -> None:
    mod = _load_tool_module()
    candidate = mod.SweepCandidate(
        candidate_id="kl_pose_k2",
        op_module="tac.codec_pipeline_kl_pose",
        op_class="Op_KLPoseStream",
        op_params={"n_components": 2},
        candidate_substream_bytes=128,
        bytes_in=1024,
        bytes_out=128,
        predicted_archive_bytes=185_120,
        predicted_score=0.2085,
        predicted_decomposition={"total": 0.2085},
        predicted_band=[0.2075, 0.2095],
        score_delta_vs_anchor=-0.0004,
        rate_delta_vs_anchor=-0.0004,
    )

    rows = mod.to_meta_lagrangian_candidates(
        [candidate], lane_class="kl_pose_stream"
    )

    assert rows == [
        {
            "candidate_id": "kl_pose_k2",
            "archive_bytes": 185_120,
            "rel_err_pct": 0.0,
            "n_layers": 0,
            "lane_class": "kl_pose_stream",
            "archive_path": None,
        }
    ]

    from tac.optimizer.meta_lagrangian import MetaLagrangianSearch

    accepted = set(inspect.signature(MetaLagrangianSearch.evaluate_candidate).parameters)
    accepted.discard("self")
    assert set(rows[0]).issubset(accepted)

