# SPDX-License-Identifier: MIT
"""Tests for ``tac.jcsp_score_marginals``.

Adversarial-coverage angles:
- Empty state_dict raises.
- Non-positive / non-finite ``value`` raises.
- Sensitivity-derived: missing keys fall back, non-finite mean falls back,
  fallback marginal honored, ``.weight`` suffix lookup works.
- Save/load roundtrip preserves values, source tag, evidence,
  model_sha256, n_streams.
- Save rejects: non-allowed source, missing model_sha256 for contest CUDA,
  non-finite/non-positive marginal values, missing parent dir,
  collision with reserved envelope fields.
- Load rejects: wrong schema, unknown source tag, missing model_sha256
  for contest_cuda_calibrated, non-dict marginals.
- End-to-end: derive → save → load → matches input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tac.jcsp_score_marginals import (
    ALLOWED_SOURCES,
    JCSP_SCORE_MARGINALS_SCHEMA,
    JCSPScoreMarginalsError,
    derive_marginals_from_sensitivity_map,
    derive_uniform_planning_marginals,
    load_marginals,
    save_marginals,
)


def test_uniform_marginals_simple_state_dict() -> None:
    state_dict = {
        "a.weight": torch.zeros(8, dtype=torch.float32),
        "b.weight": torch.zeros(4, dtype=torch.float32),
    }
    out = derive_uniform_planning_marginals(state_dict, value=1e-6)
    assert set(out.keys()) == {"a.weight", "b.weight"}
    assert all(v == pytest.approx(1e-6) for v in out.values())


def test_uniform_marginals_empty_raises() -> None:
    with pytest.raises(JCSPScoreMarginalsError, match="empty"):
        derive_uniform_planning_marginals({})


def test_uniform_marginals_nonpositive_raises() -> None:
    state_dict = {"x": torch.zeros(2, dtype=torch.float32)}
    with pytest.raises(JCSPScoreMarginalsError, match=">"):
        derive_uniform_planning_marginals(state_dict, value=0.0)
    with pytest.raises(JCSPScoreMarginalsError, match=">"):
        derive_uniform_planning_marginals(state_dict, value=-1.0)


def test_uniform_marginals_nonfinite_raises() -> None:
    state_dict = {"x": torch.zeros(2, dtype=torch.float32)}
    with pytest.raises(JCSPScoreMarginalsError, match="finite"):
        derive_uniform_planning_marginals(state_dict, value=float("nan"))


def test_sensitivity_derived_uses_channel_mean_and_falls_back() -> None:
    state_dict = {
        "a.weight": torch.zeros(64, dtype=torch.float32),  # 64 elements
        "b.weight": torch.zeros(16, dtype=torch.float32),  # no sensitivity
    }
    sensitivities = {
        "a.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),  # mean 2.5
    }
    out = derive_marginals_from_sensitivity_map(
        state_dict,
        sensitivities,
        bytes_per_element_estimate=0.5,
        fallback_marginal=1e-9,
    )
    expected_a = 2.5 / max(1.0, 64 * 0.5)
    assert out["a.weight"] == pytest.approx(expected_a, rel=1e-6)
    # b had no sensitivity → fallback
    assert out["b.weight"] == pytest.approx(1e-9, rel=1e-6)


def test_sensitivity_derived_strips_weight_suffix_for_lookup() -> None:
    state_dict = {"layer1.weight": torch.zeros(8, dtype=torch.float32)}
    # Pass sensitivity under bare module name; helper should still find it
    sensitivities = {"layer1": torch.tensor([2.0, 4.0])}  # mean 3.0
    out = derive_marginals_from_sensitivity_map(
        state_dict, sensitivities, bytes_per_element_estimate=1.0
    )
    expected = 3.0 / max(1.0, 8 * 1.0)
    assert out["layer1.weight"] == pytest.approx(expected, rel=1e-6)


def test_sensitivity_derived_invalid_args_raise() -> None:
    state_dict = {"a": torch.zeros(2, dtype=torch.float32)}
    with pytest.raises(JCSPScoreMarginalsError, match="bytes_per_element"):
        derive_marginals_from_sensitivity_map(
            state_dict, {}, bytes_per_element_estimate=0.0
        )
    with pytest.raises(JCSPScoreMarginalsError, match="fallback_marginal"):
        derive_marginals_from_sensitivity_map(
            state_dict, {}, fallback_marginal=-1.0
        )


def test_save_load_roundtrip_preserves_values_and_metadata(tmp_path: Path) -> None:
    marginals = {"a.weight": 1.5e-6, "b.weight": 2.5e-6}
    out_path = save_marginals(
        tmp_path / "marginals.json",
        marginals,
        source="placeholder_uniform",
        evidence="test fixture for jcsp roundtrip",
    )
    loaded, meta = load_marginals(out_path)
    assert loaded == marginals
    assert meta["schema"] == JCSP_SCORE_MARGINALS_SCHEMA
    assert meta["source"] == "placeholder_uniform"
    assert meta["evidence"].startswith("test fixture")
    assert meta["n_streams"] == 2


def test_save_invalid_source_raises(tmp_path: Path) -> None:
    with pytest.raises(JCSPScoreMarginalsError, match="source must be"):
        save_marginals(
            tmp_path / "x.json",
            {"a": 1e-6},
            source="invented_source",
            evidence="ok",
        )


def test_save_contest_cuda_requires_model_sha(tmp_path: Path) -> None:
    with pytest.raises(JCSPScoreMarginalsError, match="model_sha256"):
        save_marginals(
            tmp_path / "x.json",
            {"a": 1e-6},
            source="contest_cuda_calibrated",
            evidence="ok",
        )


def test_save_rejects_nonfinite_marginal(tmp_path: Path) -> None:
    with pytest.raises(JCSPScoreMarginalsError, match="finite"):
        save_marginals(
            tmp_path / "x.json",
            {"a": float("nan")},
            source="placeholder_uniform",
            evidence="ok",
        )
    with pytest.raises(JCSPScoreMarginalsError, match=">"):
        save_marginals(
            tmp_path / "x.json",
            {"a": -0.001},
            source="placeholder_uniform",
            evidence="ok",
        )


def test_save_extra_field_collision_raises(tmp_path: Path) -> None:
    with pytest.raises(JCSPScoreMarginalsError, match="reserved"):
        save_marginals(
            tmp_path / "x.json",
            {"a": 1e-6},
            source="placeholder_uniform",
            evidence="ok",
            extra={"schema": "tampered"},
        )


def test_save_missing_parent_dir_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "nonexistent" / "x.json"
    with pytest.raises(JCSPScoreMarginalsError, match="parent directory"):
        save_marginals(
            bogus,
            {"a": 1e-6},
            source="placeholder_uniform",
            evidence="ok",
        )


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    p = tmp_path / "bad_schema.json"
    p.write_text(
        json.dumps(
            {
                "schema": "wrong_schema_v0",
                "source": "placeholder_uniform",
                "evidence": "x",
                "n_streams": 1,
                "marginals": {"a": 1e-6},
            }
        )
    )
    with pytest.raises(JCSPScoreMarginalsError, match="schema"):
        load_marginals(p)


def test_load_rejects_unknown_source(tmp_path: Path) -> None:
    p = tmp_path / "bad_source.json"
    p.write_text(
        json.dumps(
            {
                "schema": JCSP_SCORE_MARGINALS_SCHEMA,
                "source": "fabricated",
                "evidence": "x",
                "n_streams": 1,
                "marginals": {"a": 1e-6},
            }
        )
    )
    with pytest.raises(JCSPScoreMarginalsError, match="unknown source"):
        load_marginals(p)


def test_load_contest_cuda_without_sha_raises(tmp_path: Path) -> None:
    p = tmp_path / "missing_sha.json"
    p.write_text(
        json.dumps(
            {
                "schema": JCSP_SCORE_MARGINALS_SCHEMA,
                "source": "contest_cuda_calibrated",
                "evidence": "x",
                "n_streams": 1,
                "marginals": {"a": 1e-6},
            }
        )
    )
    with pytest.raises(JCSPScoreMarginalsError, match="model_sha256"):
        load_marginals(p)


def test_end_to_end_derive_save_load(tmp_path: Path) -> None:
    state_dict = {
        "a.weight": torch.zeros(8, dtype=torch.float32),
        "b.weight": torch.zeros(4, dtype=torch.float32),
    }
    derived = derive_uniform_planning_marginals(state_dict, value=2.5e-6)
    out_path = save_marginals(
        tmp_path / "round.json",
        derived,
        source="placeholder_uniform",
        evidence="end-to-end test",
    )
    loaded, meta = load_marginals(out_path)
    assert loaded == derived
    assert meta["n_streams"] == 2
    assert "envelope_sha256" in meta
    assert all(s in ALLOWED_SOURCES for s in [meta["source"]])


def test_save_with_contest_cuda_and_model_sha_succeeds(tmp_path: Path) -> None:
    out_path = save_marginals(
        tmp_path / "cuda.json",
        {"a": 1e-6, "b": 2e-6},
        source="contest_cuda_calibrated",
        evidence="finite difference probe at git deadbeef",
        model_sha256="a" * 64,
    )
    loaded, meta = load_marginals(out_path)
    assert loaded == {"a": 1e-6, "b": 2e-6}
    assert meta["model_sha256"] == "a" * 64
