# SPDX-License-Identifier: MIT
"""Dedicated tests for the split-device MPS-vs-CUDA harvest contract.

Sister of ``test_mps_gap_experiment_tiny_renderer.py`` (which covers the
TinyRenderer model + the legacy single-machine ``compute_gap_components``).

This module pins the SPLIT-DEVICE architecture introduced per predecessor
verdict ``mps_phase_b_gap_experiment_verdict_20260519T053530Z`` Option A
reactivation:

* :func:`compute_local_mps_reference_components` writes
  ``local_mps_components.json`` + ``local_mps_forward_outputs.pt`` on the
  LOCAL Mac MPS hardware (or CPU for unit testing).
* :func:`compute_target_cuda_components` writes
  ``target_cuda_components.json`` + ``target_cuda_forward_outputs.pt`` on
  the REMOTE Modal A10G worker (or local CPU for unit testing).
* :func:`diff_components_and_classify_verdict` loads BOTH JSONs (never
  recomputes on a single device) + computes per-component gap + emits
  canonical ``gap_results.json``.

Catalog cross-refs: #229 (premise verification before edit),
#192 (macOS-CPU advisory non-promotion), #317 (local research-signal
evidence stamping), #324 (predicted band validation status),
#205 (canonical inflate device selector), #287 (axis-tag every score).

NOT a contest substrate suite — purely diagnostic infrastructure tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tac.mps_gap_experiment.harvest_and_verdict import (
    ComponentGap,
    GapManifest,
    classify_verdict,
    compute_local_mps_reference_components,
    compute_target_cuda_components,
    diff_components_and_classify_verdict,
)
from tac.mps_gap_experiment.tiny_renderer import build_tiny_renderer


def _seed_checkpoint_and_cache(tmp_path: Path) -> tuple[Path, Path]:
    """Helper: build a tiny renderer, save its state_dict + a tiny frame cache."""
    model = build_tiny_renderer(seed=42)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "checkpoint_ema.pt"
    torch.save(state, ckpt_path)
    cache = torch.zeros(2, 2, 3, 384, 512)
    cache_path = tmp_path / "frame_cache.pt"
    torch.save(cache, cache_path)
    return ckpt_path, cache_path


def test_compute_local_mps_reference_components_writes_canonical_pair(
    tmp_path: Path,
) -> None:
    """Local reference helper writes BOTH the components JSON + the outputs .pt."""
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    out_dir = tmp_path / "out"
    components_path = compute_local_mps_reference_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=out_dir,
        device="cpu",  # CPU here for hermetic unit test; harness uses "mps"
        include_scorer_components=False,
    )
    assert components_path == out_dir / "local_mps_components.json"
    assert components_path.exists()
    outputs_path = out_dir / "local_mps_forward_outputs.pt"
    assert outputs_path.exists()


def test_compute_local_mps_reference_components_axis_tag_and_evidence_grade(
    tmp_path: Path,
) -> None:
    """Local reference manifest carries [MPS-research-signal] axis tag."""
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    components_path = compute_local_mps_reference_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path,
        device="cpu",
    )
    manifest = json.loads(components_path.read_text())
    assert manifest["axis_tag"] == "[MPS-research-signal]"
    assert manifest["evidence_grade"] == "MPS-research-signal"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["device"] == "cpu"
    assert manifest["num_pairs"] == 2
    assert "components" in manifest
    assert "pixel_l1_mean" in manifest["components"]


def test_compute_local_mps_reference_components_forward_outputs_shape_matches_cache(
    tmp_path: Path,
) -> None:
    """The persisted forward-outputs .pt has the same num_pairs as the input batch."""
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    components_path = compute_local_mps_reference_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path,
        device="cpu",
    )
    outputs = torch.load(
        tmp_path / "local_mps_forward_outputs.pt", weights_only=True
    )
    assert outputs.shape == (2, 2, 3, 384, 512)


def test_compute_target_cuda_components_writes_canonical_pair(
    tmp_path: Path,
) -> None:
    """Target CUDA helper writes BOTH the components JSON + the outputs .pt."""
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    out_dir = tmp_path / "modal_out"
    components_path = compute_target_cuda_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=out_dir,
        device="cpu",  # CPU stub for hermetic unit test; Modal uses "cuda"
        include_scorer_components=False,
    )
    assert components_path == out_dir / "target_cuda_components.json"
    assert components_path.exists()
    outputs_path = out_dir / "target_cuda_forward_outputs.pt"
    assert outputs_path.exists()


def test_compute_target_cuda_components_axis_tag_and_evidence_grade(
    tmp_path: Path,
) -> None:
    """Target manifest carries [diagnostic-CUDA Modal A10G] axis tag."""
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    components_path = compute_target_cuda_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path,
        device="cpu",
    )
    manifest = json.loads(components_path.read_text())
    assert manifest["axis_tag"] == "[diagnostic-CUDA Modal A10G]"
    assert manifest["evidence_grade"] == "diagnostic-CUDA Modal A10G"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["num_pairs"] == 2


def test_compute_target_cuda_components_persists_modal_call_id_when_provided(
    tmp_path: Path,
) -> None:
    """When the caller threads a modal_call_id, it lands in the manifest."""
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    components_path = compute_target_cuda_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path,
        device="cpu",
        modal_call_id="fc-TEST-FAKE-ID-FOR-UNITTEST",
    )
    manifest = json.loads(components_path.read_text())
    assert manifest["modal_call_id"] == "fc-TEST-FAKE-ID-FOR-UNITTEST"


def test_compute_target_cuda_components_omits_modal_call_id_when_none(
    tmp_path: Path,
) -> None:
    """Without an explicit modal_call_id the manifest does NOT carry a fake one."""
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    components_path = compute_target_cuda_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path,
        device="cpu",
    )
    manifest = json.loads(components_path.read_text())
    assert "modal_call_id" not in manifest


def test_diff_components_and_classify_verdict_zero_gap_on_self_pair(
    tmp_path: Path,
) -> None:
    """Diffing two IDENTICAL component dicts produces gap_relative_aggregate = 0.0."""
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    local_path = compute_local_mps_reference_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path / "local",
        device="cpu",
    )
    # Re-use the same device for the target side to get a deterministic 0 gap;
    # this is structurally analogous to the Modal worker running the SAME
    # weights on a different device — the diff helper does NOT care which
    # device produced each side, only that both were produced.
    target_path = compute_target_cuda_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path / "target",
        device="cpu",
    )
    out_path = tmp_path / "gap_results.json"
    manifest = diff_components_and_classify_verdict(
        local_mps_components_path=local_path,
        target_cuda_components_path=target_path,
        output_path=out_path,
    )
    assert manifest.gap_relative_aggregate == 0.0
    assert manifest.verdict == "LOCAL_MPS_TRAIN_VIABLE"
    assert manifest.num_pairs == 2
    assert out_path.exists()


def test_diff_components_and_classify_verdict_emits_per_component_rows(
    tmp_path: Path,
) -> None:
    """The diff helper emits one ComponentGap row per shared component key."""
    local_dir = tmp_path / "local"
    target_dir = tmp_path / "target"
    local_dir.mkdir()
    target_dir.mkdir()
    local_dir.joinpath("local_mps_components.json").write_text(
        json.dumps(
            {
                "device": "mps",
                "num_pairs": 3,
                "components": {
                    "pixel_l1_mean": 0.10,
                    "segnet_mean_output": -0.50,
                    "posenet_mean_output": 0.20,
                },
            }
        )
    )
    target_dir.joinpath("target_cuda_components.json").write_text(
        json.dumps(
            {
                "device": "cuda",
                "num_pairs": 3,
                "components": {
                    "pixel_l1_mean": 0.11,
                    "segnet_mean_output": -0.49,
                    "posenet_mean_output": 0.18,
                },
            }
        )
    )
    out_path = tmp_path / "gap_results.json"
    manifest = diff_components_and_classify_verdict(
        local_mps_components_path=local_dir / "local_mps_components.json",
        target_cuda_components_path=target_dir / "target_cuda_components.json",
        output_path=out_path,
    )
    assert len(manifest.components) == 3
    names = sorted(c.name for c in manifest.components)
    assert names == [
        "pixel_l1_mean",
        "posenet_mean_output",
        "segnet_mean_output",
    ]
    pixel = next(c for c in manifest.components if c.name == "pixel_l1_mean")
    assert pixel.mps_value == pytest.approx(0.10)
    assert pixel.target_value == pytest.approx(0.11)
    assert pixel.absolute_diff == pytest.approx(0.01, abs=1e-9)
    assert pixel.relative_diff == pytest.approx(0.10, rel=1e-3)


def test_diff_components_and_classify_verdict_threshold_branches(
    tmp_path: Path,
) -> None:
    """Pin the three verdict bands via synthetic component pairs."""

    def _emit(tag: str, mps_val: float, target_val: float) -> Path:
        path = tmp_path / f"{tag}.json"
        path.write_text(
            json.dumps(
                {
                    "device": "mps" if "local" in tag else "cuda",
                    "num_pairs": 1,
                    "components": {"pixel_l1_mean": mps_val if "local" in tag else target_val},
                }
            )
        )
        return path

    # 1% relative gap -> VIABLE
    local_a = _emit("local_a", 0.10, 0.10)
    target_a = _emit("target_a", 0.10, 0.101)
    out_a = tmp_path / "gap_a.json"
    m_a = diff_components_and_classify_verdict(
        local_mps_components_path=local_a,
        target_cuda_components_path=target_a,
        output_path=out_a,
    )
    assert m_a.verdict == "LOCAL_MPS_TRAIN_VIABLE"

    # 10% relative gap -> ADVISORY
    local_b = _emit("local_b", 0.10, 0.10)
    target_b = _emit("target_b", 0.10, 0.11)
    out_b = tmp_path / "gap_b.json"
    m_b = diff_components_and_classify_verdict(
        local_mps_components_path=local_b,
        target_cuda_components_path=target_b,
        output_path=out_b,
    )
    assert m_b.verdict == "LOCAL_MPS_TRAIN_VIABLE_ADVISORY_ONLY"

    # 50% relative gap -> NOT_VIABLE
    local_c = _emit("local_c", 0.10, 0.10)
    target_c = _emit("target_c", 0.10, 0.15)
    out_c = tmp_path / "gap_c.json"
    m_c = diff_components_and_classify_verdict(
        local_mps_components_path=local_c,
        target_cuda_components_path=target_c,
        output_path=out_c,
    )
    assert m_c.verdict == "LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX"


def test_diff_components_and_classify_verdict_refuses_num_pairs_mismatch(
    tmp_path: Path,
) -> None:
    """num_pairs mismatch raises (the two sides MUST compare the same batch)."""
    local_dir = tmp_path / "local"
    target_dir = tmp_path / "target"
    local_dir.mkdir()
    target_dir.mkdir()
    local_dir.joinpath("local_mps_components.json").write_text(
        json.dumps({"device": "mps", "num_pairs": 10, "components": {"x": 0.5}})
    )
    target_dir.joinpath("target_cuda_components.json").write_text(
        json.dumps({"device": "cuda", "num_pairs": 11, "components": {"x": 0.5}})
    )
    with pytest.raises(ValueError, match="num_pairs mismatch"):
        diff_components_and_classify_verdict(
            local_mps_components_path=local_dir / "local_mps_components.json",
            target_cuda_components_path=target_dir / "target_cuda_components.json",
            output_path=tmp_path / "gap_results.json",
        )


def test_diff_components_and_classify_verdict_refuses_missing_local(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target_dir.joinpath("target_cuda_components.json").write_text(
        json.dumps({"device": "cuda", "num_pairs": 2, "components": {"x": 0.0}})
    )
    with pytest.raises(FileNotFoundError, match="local MPS components"):
        diff_components_and_classify_verdict(
            local_mps_components_path=tmp_path / "does_not_exist.json",
            target_cuda_components_path=target_dir / "target_cuda_components.json",
            output_path=tmp_path / "gap_results.json",
        )


def test_diff_components_and_classify_verdict_refuses_missing_target(
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    local_dir.joinpath("local_mps_components.json").write_text(
        json.dumps({"device": "mps", "num_pairs": 2, "components": {"x": 0.0}})
    )
    with pytest.raises(FileNotFoundError, match="target CUDA components"):
        diff_components_and_classify_verdict(
            local_mps_components_path=local_dir / "local_mps_components.json",
            target_cuda_components_path=tmp_path / "missing.json",
            output_path=tmp_path / "gap_results.json",
        )


def test_diff_components_and_classify_verdict_persists_device_fields_in_output(
    tmp_path: Path,
) -> None:
    """Output gap_results.json records both device labels from the input JSONs."""
    local_dir = tmp_path / "local"
    target_dir = tmp_path / "target"
    local_dir.mkdir()
    target_dir.mkdir()
    local_dir.joinpath("local_mps_components.json").write_text(
        json.dumps({"device": "mps", "num_pairs": 2, "components": {"x": 0.5}})
    )
    target_dir.joinpath("target_cuda_components.json").write_text(
        json.dumps({"device": "cuda", "num_pairs": 2, "components": {"x": 0.55}})
    )
    out_path = tmp_path / "gap_results.json"
    manifest = diff_components_and_classify_verdict(
        local_mps_components_path=local_dir / "local_mps_components.json",
        target_cuda_components_path=target_dir / "target_cuda_components.json",
        output_path=out_path,
    )
    assert manifest.mps_reference_device == "mps"
    assert manifest.target_device == "cuda"
    written = json.loads(out_path.read_text())
    assert written["mps_reference_device"] == "mps"
    assert written["target_device"] == "cuda"


def test_classify_verdict_handles_nan_aggregate() -> None:
    """NaN propagates to the NOT_VIABLE bucket (predecessor's NaN-fallback)."""
    assert (
        classify_verdict(float("nan"))
        == "LOCAL_MPS_TRAIN_NOT_VIABLE_PIVOT_MLX_OR_VTOOLBOX"
    )


def test_local_mps_components_json_is_loadable_by_diff_helper_end_to_end(
    tmp_path: Path,
) -> None:
    """End-to-end: capture LOCAL + TARGET via the canonical helpers, then diff.

    Mirrors the production split-device flow (just using CPU on both sides for
    hermetic test): train_on_mps writes local_mps_components.json -> Modal
    dispatch writes target_cuda_components.json -> harvest CLI diffs them.
    """
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    local_path = compute_local_mps_reference_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path / "local",
        device="cpu",
    )
    target_path = compute_target_cuda_components(
        checkpoint_path=ckpt_path,
        frame_cache_path=cache_path,
        output_dir=tmp_path / "target",
        device="cpu",
    )
    out_path = tmp_path / "gap_results.json"
    manifest = diff_components_and_classify_verdict(
        local_mps_components_path=local_path,
        target_cuda_components_path=target_path,
        output_path=out_path,
    )
    # CPU-vs-CPU on same weights + inputs is bit-stable; gap is 0
    assert manifest.gap_relative_aggregate == 0.0
    assert manifest.verdict == "LOCAL_MPS_TRAIN_VIABLE"
    # The end-to-end manifest is a canonical GapManifest dict on disk
    written = json.loads(out_path.read_text())
    assert "components" in written
    assert "verdict" in written
    assert "gap_relative_aggregate" in written


def test_modal_dispatch_entry_uses_compute_target_cuda_components_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Modal-side dispatch script MUST emit ONLY target CUDA components.

    Regression guard against re-introducing the single-device measurement
    artifact bug (predecessor verdict
    mps_phase_b_gap_experiment_verdict_20260519T053530Z). Running the dispatch
    entry point should write target_cuda_components.json (NOT gap_results.json).
    """
    ckpt_path, cache_path = _seed_checkpoint_and_cache(tmp_path)
    out_dir = tmp_path / "modal_out"
    # Import inside the test so pytest doesn't pull torch/scorer at collection
    from experiments.mps_gap_experiment_a10g_dispatch import main as dispatch_main

    rc = dispatch_main(
        [
            "--checkpoint-input",
            str(ckpt_path),
            "--frame-cache-input",
            str(cache_path),
            "--output-dir",
            str(out_dir),
            "--target-device",
            "cpu",  # CPU stub for hermetic test; Modal uses "cuda"
        ]
    )
    assert rc == 0
    assert (out_dir / "target_cuda_components.json").exists()
    assert (out_dir / "target_cuda_forward_outputs.pt").exists()
    # And the dispatch entry point MUST NOT pretend to emit the canonical
    # gap_results.json — that's the LOCAL diff helper's job
    assert not (out_dir / "gap_results.json").exists()


def test_harvest_cli_diff_subcommand_smoke(
    tmp_path: Path,
) -> None:
    """The CLI `diff` subcommand invocation matches the canonical helper."""
    from tac.mps_gap_experiment.harvest_and_verdict_cli import main as cli_main

    local_dir = tmp_path / "local"
    target_dir = tmp_path / "target"
    local_dir.mkdir()
    target_dir.mkdir()
    local_dir.joinpath("local_mps_components.json").write_text(
        json.dumps({"device": "mps", "num_pairs": 1, "components": {"x": 0.5}})
    )
    target_dir.joinpath("target_cuda_components.json").write_text(
        json.dumps({"device": "cuda", "num_pairs": 1, "components": {"x": 0.5}})
    )
    out_path = tmp_path / "gap_results.json"
    rc = cli_main(
        [
            "diff",
            "--local",
            str(local_dir / "local_mps_components.json"),
            "--target",
            str(target_dir / "target_cuda_components.json"),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    manifest = json.loads(out_path.read_text())
    assert manifest["verdict"] == "LOCAL_MPS_TRAIN_VIABLE"
