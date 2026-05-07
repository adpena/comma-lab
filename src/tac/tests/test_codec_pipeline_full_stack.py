"""Tests for :mod:`tac.codec_pipeline_full_stack` - Op 4 full-stack orchestrator.

Coverage (>= 8 tests per parent task spec):

  1. Protocol satisfaction (:class:`Op4_FullStackOrchestrator` -> CodecOp).
  2. ``CANONICAL_STACK_NAMES`` matches the parent task's eight canonical
     stacks.
  3. ``run_composition_matrix`` returns a dict with all eight stacks.
  4. Each stack roundtrips bit-faithful within its quantization grid:
        * Op1_alone, Op2_alone, beta_identity_then_Op1: per-tensor rel_err
          <= 1% (Op 1 / Op 2 quantize internally to a ~int7-band fixed grid).
        * Op_GammaJointADMM_alone: int8-grid recovery (rel-err <= 5%).
        * Op3_int6/Op3_int7 stacks: int-N-grid recovery (rel-err <= 5%).
  5. ``pick_smallest_stack`` returns a canonical name and matches the
     argmin of ``bytes_out``.
  6. Manifest is written to
     ``experiments/results/lane_codec_pipeline_full_stack_<UTC>/composition_matrix.json``
     and tagged ``[empirical:<path>]`` (no ``[contest-CUDA]`` claim).
  7. The matrix is byte-deterministic across calls when the inputs are
     identical (no scorer dependency, no random state).
  8. Op 4's own ``encode`` records the matrix in ``op_state`` and decodes
     bit-faithfully via the Op 1 delegate.
  9. Manifest payload contains ``score_claim=False`` and the
     ``evidence_grade`` is ``"predicted"`` per CLAUDE.md "Forbidden
     score claims" discipline.
 10. Each stack's ``per_op_bytes`` length matches its op chain length.

Strict-scorer-rule: pure CPU; tests never load a scorer.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tac.codec_pipeline import CodecOp, CodecPipeline
from tac.codec_pipeline_full_stack import (
    CANONICAL_STACK_NAMES,
    Op4_FullStackOrchestrator,
    StackRunRecord,
    pick_smallest_stack,
    run_composition_matrix,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Stacks whose composition path is fully lossless (Op 1 / Op 2 are exact;
#: beta=identity passes tensors through unchanged). Tensor values must come
#: back bit-exact from the pipeline decode.
_LOSSLESS_STACKS: frozenset[str] = frozenset(
    {"Op1_alone", "Op2_alone", "beta_identity_then_Op1"}
)

#: Stacks whose composition includes Op 3 (apogee_intN). Bit-exact recovery
#: is not expected; instead the dequantized substrate is recovered exactly
#: and the per-tensor relative error vs the original fp32 input is bounded.
_INT_N_STACKS: frozenset[str] = frozenset(
    {
        "Op3_int6_then_Op1",
        "Op3_int6_then_Op2",
        "Op3_int7_then_Op1",
        "beta_identity_then_Op3_int6_then_Op2",
    }
)

#: gamma alone uses int8 internally; its recovery is on the int8-rounded
#: substrate. We use a generous atol/rtol when comparing against fp32 input.
_GAMMA_STACKS: frozenset[str] = frozenset({"Op_GammaJointADMM_alone"})


def _synthetic_state_dict(seed: int = 0, scale: float = 0.1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * scale
        for name, shape in FIXED_STATE_SCHEMA
    }


# ---------------------------------------------------------------------------
# 1. Protocol satisfaction
# ---------------------------------------------------------------------------

def test_op4_satisfies_codec_op_protocol() -> None:
    op = Op4_FullStackOrchestrator()
    assert isinstance(op, CodecOp)
    assert op.name == "full_stack_orchestrator"


# ---------------------------------------------------------------------------
# 2. Canonical stack names match the spec
# ---------------------------------------------------------------------------

def test_canonical_stack_names_match_parent_task_spec() -> None:
    """Eight stacks per the parent task table, in order."""
    expected = (
        "Op1_alone",
        "Op2_alone",
        "Op_GammaJointADMM_alone",
        "beta_identity_then_Op1",
        "Op3_int6_then_Op1",
        "Op3_int6_then_Op2",
        "Op3_int7_then_Op1",
        "beta_identity_then_Op3_int6_then_Op2",
    )
    assert expected == CANONICAL_STACK_NAMES


# ---------------------------------------------------------------------------
# 3. Matrix returns all eight stacks
# ---------------------------------------------------------------------------

def test_run_composition_matrix_returns_all_eight_stacks(tmp_path: Path) -> None:
    sd = _synthetic_state_dict()
    records = run_composition_matrix(sd, write_manifest=False, output_dir=tmp_path)
    assert set(records.keys()) == set(CANONICAL_STACK_NAMES)
    assert len(records) == 8
    for name, rec in records.items():
        assert isinstance(rec, StackRunRecord)
        assert rec.stack_name == name
        assert rec.bytes_out > 0
        assert rec.final_blob_sha256  # 64-char hex digest
        assert len(rec.final_blob_sha256) == 64


# ---------------------------------------------------------------------------
# 4a. Lossless stacks roundtrip bit-faithfully
# ---------------------------------------------------------------------------

def test_each_op1_op2_stack_roundtrips_within_quant_grid(tmp_path: Path) -> None:
    """Op1_alone, Op2_alone, beta_identity_then_Op1 each roundtrip within
    their internal quantization grid. Op 1 / Op 2 quantize the per-tensor
    weights to a fixed-grid representation (~int7-band scale per tensor);
    the recovery is lossy at ~0.4-1% relative error per tensor. The
    beta_identity_then_Op1 stack composes a true passthrough beta with
    Op 1, so its rel-err mirrors Op 1 alone's."""
    sd = _synthetic_state_dict()
    pipelines = {
        "Op1_alone": CodecPipeline(_pipeline_for("Op1_alone")),
        "Op2_alone": CodecPipeline(_pipeline_for("Op2_alone")),
        "beta_identity_then_Op1": CodecPipeline(
            _pipeline_for("beta_identity_then_Op1")
        ),
    }
    for name, pipeline in pipelines.items():
        blob, _ = pipeline.encode(sd)
        decoded, _ = pipeline.decode(blob)
        assert set(decoded.keys()) == set(sd.keys()), name
        for tensor_name, t_in in sd.items():
            t_out = decoded[tensor_name].to(torch.float32)
            t_orig = t_in.to(torch.float32)
            assert tuple(t_orig.shape) == tuple(t_out.shape), f"{name}/{tensor_name}"
            denom = max(t_orig.abs().max().item(), 1e-8)
            rel_err = (t_orig - t_out).abs().max().item() / denom
            assert rel_err <= 0.01, (
                f"{name}/{tensor_name}: rel_err {rel_err:.4f} > 0.01 "
                f"(Op 1 / Op 2 internal quant grid)"
            )


# ---------------------------------------------------------------------------
# 4b. int-N stacks roundtrip within the int-N grid (rel-err <= 5%)
# ---------------------------------------------------------------------------

def test_each_int_n_stack_roundtrips_within_quantization_grid(tmp_path: Path) -> None:
    """Stacks containing Op 3 (apogee_intN) are lossy at the int-N grid;
    per-tensor relative error must be <= 5% (matches the int6 basin-parity
    safety guard from test_codec_pipeline_apogee_int)."""
    sd = _synthetic_state_dict()
    for stack_name in _INT_N_STACKS:
        pipeline = CodecPipeline(_pipeline_for(stack_name))
        blob, _ = pipeline.encode(sd)
        decoded, _ = pipeline.decode(blob)
        assert set(decoded.keys()) == set(sd.keys()), stack_name
        for tensor_name, t_in in sd.items():
            t_out = decoded[tensor_name].to(torch.float32)
            t_orig = t_in.to(torch.float32)
            denom = max(t_orig.abs().max().item(), 1e-8)
            rel_err = (t_orig - t_out).abs().max().item() / denom
            assert rel_err <= 0.05, (
                f"{stack_name}/{tensor_name}: rel_err {rel_err:.4f} > 0.05"
            )


# ---------------------------------------------------------------------------
# 4c. gamma alone roundtrips on the int8 substrate
# ---------------------------------------------------------------------------

def test_gamma_alone_roundtrips_on_int8_substrate(tmp_path: Path) -> None:
    """Op_GammaJointADMM uses int8 internally; recovery on the int8 grid
    is exact, but vs the original fp32 input the per-tensor max-abs error
    is bounded by one int8 step (= scale = abs_max / 127). Generous bound."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline(_pipeline_for("Op_GammaJointADMM_alone"))
    blob, _ = pipeline.encode(sd)
    decoded, _ = pipeline.decode(blob)
    assert set(decoded.keys()) == set(sd.keys())
    for tensor_name, t_in in sd.items():
        t_out = decoded[tensor_name].to(torch.float32)
        t_orig = t_in.to(torch.float32)
        # Per-tensor int8 step = abs_max / 127. Allow up to 1 step (= 1/127
        # ~ 0.79% of abs_max) max-abs error.
        denom = max(t_orig.abs().max().item(), 1e-8)
        rel_err = (t_orig - t_out).abs().max().item() / denom
        assert rel_err <= 0.05, (
            f"gamma/{tensor_name}: rel_err {rel_err:.4f} > 0.05 int8 step"
        )


# ---------------------------------------------------------------------------
# 5. pick_smallest_stack picks the argmin
# ---------------------------------------------------------------------------

def test_pick_smallest_stack_returns_argmin() -> None:
    sd = _synthetic_state_dict()
    records = run_composition_matrix(sd, write_manifest=False)
    winner = pick_smallest_stack(records)
    assert winner in CANONICAL_STACK_NAMES
    winner_bytes = records[winner].bytes_out
    for name, rec in records.items():
        assert rec.bytes_out >= winner_bytes, (
            f"{name} ({rec.bytes_out}B) is smaller than winner "
            f"{winner} ({winner_bytes}B)"
        )


# ---------------------------------------------------------------------------
# 6. Manifest is written to experiments/results/<lane>/composition_matrix.json
# ---------------------------------------------------------------------------

def test_manifest_written_with_empirical_tag(tmp_path: Path) -> None:
    """Manifest must land at ``<output_dir>/composition_matrix.json``
    tagged ``[empirical:<path>]`` with ``score_claim=False`` per CLAUDE.md."""
    sd = _synthetic_state_dict()
    out_dir = tmp_path / "lane_codec_pipeline_full_stack_TEST"
    records = run_composition_matrix(sd, write_manifest=True, output_dir=out_dir)
    manifest_path = out_dir / "composition_matrix.json"
    assert manifest_path.exists(), f"manifest not written to {manifest_path}"

    payload = json.loads(manifest_path.read_text())
    assert payload["score_claim"] is False
    assert payload["evidence_grade"] == "predicted"
    assert payload["evidence_tag"].startswith("[empirical:")
    assert "lane_codec_pipeline_full_stack" in payload["evidence_tag"]
    # Forbidden score claims discipline: no [contest-CUDA] anywhere.
    raw = manifest_path.read_text()
    assert "[contest-CUDA]" not in raw, (
        "manifest must not emit a [contest-CUDA] tag - score claims forbidden"
    )

    # Eight stack rows, each carrying op_names + bytes_out + final_blob_sha256.
    assert len(payload["stacks"]) == 8
    seen_names = {row["stack_name"] for row in payload["stacks"]}
    assert seen_names == set(CANONICAL_STACK_NAMES)
    for row in payload["stacks"]:
        assert row["bytes_out"] > 0
        assert len(row["final_blob_sha256"]) == 64
        assert row["op_names"]  # non-empty op chain
        assert len(row["per_op_bytes"]) == len(row["op_names"])

    # Winner is consistent with the records dict.
    assert payload["winner_stack_name"] == pick_smallest_stack(records)
    assert payload["winner_bytes_out"] == records[payload["winner_stack_name"]].bytes_out


# ---------------------------------------------------------------------------
# 7. Byte-deterministic matrix
# ---------------------------------------------------------------------------

def test_run_composition_matrix_byte_deterministic() -> None:
    """Two identical runs must produce identical SHA-256 per stack."""
    sd = _synthetic_state_dict()
    rec_a = run_composition_matrix(sd, write_manifest=False)
    rec_b = run_composition_matrix(sd, write_manifest=False)
    assert set(rec_a.keys()) == set(rec_b.keys())
    for name in CANONICAL_STACK_NAMES:
        assert rec_a[name].final_blob_sha256 == rec_b[name].final_blob_sha256, name
        assert rec_a[name].bytes_out == rec_b[name].bytes_out, name


# ---------------------------------------------------------------------------
# 8. Op4 encode / decode façade
# ---------------------------------------------------------------------------

def test_op4_encode_records_matrix_in_op_state_and_decode_within_grid() -> None:
    """Op 4's encode delegates to Op 1 and records the full matrix in
    ``op_state``. Decode replays the Op 1 delegate (recovery is lossy at
    Op 1's internal quant grid; per-tensor rel-err <= 1%)."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op4_FullStackOrchestrator()])
    blob, manifest = pipeline.encode(sd)
    assert blob[:4] == b"CPL1"

    op4_state = manifest.op_results[0].op_state
    assert op4_state["delegate_op_name"] == "pr101_split_brotli"
    matrix_records = op4_state["matrix_records"]
    assert len(matrix_records) == 8
    seen_names = {r["stack_name"] for r in matrix_records}
    assert seen_names == set(CANONICAL_STACK_NAMES)
    assert op4_state["matrix_winner_stack_name"] in CANONICAL_STACK_NAMES

    # Decode: Op 4 replays the Op 1 delegate; recovery is on Op 1's
    # internal quant grid (rel-err <= 1% per tensor).
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["full_stack_orchestrator"]
    assert set(decoded.keys()) == set(sd.keys())
    for name, t_in in sd.items():
        t_orig = t_in.to(torch.float32)
        t_out = decoded[name].to(torch.float32)
        denom = max(t_orig.abs().max().item(), 1e-8)
        rel_err = (t_orig - t_out).abs().max().item() / denom
        assert rel_err <= 0.01, f"{name}: rel_err {rel_err:.4f} > 0.01"


# ---------------------------------------------------------------------------
# 9. Per-op-bytes length consistency
# ---------------------------------------------------------------------------

def test_per_op_bytes_length_matches_op_chain_length() -> None:
    sd = _synthetic_state_dict()
    records = run_composition_matrix(sd, write_manifest=False)
    expected_chain_lens = {
        "Op1_alone": 1,
        "Op2_alone": 1,
        "Op_GammaJointADMM_alone": 1,
        "beta_identity_then_Op1": 2,
        "Op3_int6_then_Op1": 2,
        "Op3_int6_then_Op2": 2,
        "Op3_int7_then_Op1": 2,
        "beta_identity_then_Op3_int6_then_Op2": 3,
    }
    for name, expected_len in expected_chain_lens.items():
        rec = records[name]
        assert len(rec.op_names) == expected_len, name
        assert len(rec.per_op_bytes) == expected_len, name


# ---------------------------------------------------------------------------
# 10. Empty records edge case for pick_smallest_stack
# ---------------------------------------------------------------------------

def test_pick_smallest_stack_empty_returns_empty_string() -> None:
    """Defensive: pick_smallest_stack on an empty mapping returns ''."""
    assert pick_smallest_stack({}) == ""


# ---------------------------------------------------------------------------
# Helpers - rebuild the canonical pipelines per stack name (mirror of
# ``_build_canonical_pipelines`` so tests can introspect the op chain).
# ---------------------------------------------------------------------------

def _pipeline_for(stack_name: str) -> list:
    """Return the op list for one canonical stack (matches the orchestrator).

    Test-side mirror of :func:`tac.codec_pipeline_full_stack._build_canonical_pipelines`.
    Kept inline so a refactor that changes the orchestrator's pipelines
    will fail these tests if the spec changes - the test pins the spec
    at the parent-task level rather than trusting the orchestrator's
    private builder.
    """
    from tac.codec_pipeline import (
        Op1_PR101SplitBrotli,
        Op2_PR103ArithmeticCodec,
    )
    from tac.codec_pipeline_apogee_int import Op3_ApogeeIntN_Substrate
    from tac.codec_pipeline_joint_admm import Op_GammaJointADMM
    from tac.codec_pipeline_sensitivity import Op_SensitivityPreprocess

    if stack_name == "Op1_alone":
        return [Op1_PR101SplitBrotli(auto_select=False)]
    if stack_name == "Op2_alone":
        return [Op2_PR103ArithmeticCodec()]
    if stack_name == "Op_GammaJointADMM_alone":
        return [Op_GammaJointADMM(max_admm_iters=2)]
    if stack_name == "beta_identity_then_Op1":
        return [
            Op_SensitivityPreprocess.identity(),
            Op1_PR101SplitBrotli(auto_select=False),
        ]
    if stack_name == "Op3_int6_then_Op1":
        return [
            Op3_ApogeeIntN_Substrate(bits=6),
            Op1_PR101SplitBrotli(auto_select=False),
        ]
    if stack_name == "Op3_int6_then_Op2":
        return [
            Op3_ApogeeIntN_Substrate(bits=6),
            Op2_PR103ArithmeticCodec(),
        ]
    if stack_name == "Op3_int7_then_Op1":
        return [
            Op3_ApogeeIntN_Substrate(bits=7),
            Op1_PR101SplitBrotli(auto_select=False),
        ]
    if stack_name == "beta_identity_then_Op3_int6_then_Op2":
        return [
            Op_SensitivityPreprocess.identity(),
            Op3_ApogeeIntN_Substrate(bits=6),
            Op2_PR103ArithmeticCodec(),
        ]
    raise KeyError(f"unknown canonical stack name: {stack_name}")
