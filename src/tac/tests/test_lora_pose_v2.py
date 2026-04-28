"""Lock-in tests for Lane LR-V2 — LEARNABLE-rank LoRA pose adaptation.

V1 covered (test_lora_pose.py): rank=R is FROZEN at construction time. The
operator picks R offline (typically R=1 per project_posenet_rank1_discovery).

V2 (this file) adds: per-rank gates that the optimiser co-trains with U/V.
The effective rank is data-driven; ranks with final gate < prune_threshold
are dropped at serialisation. Tests:

  1. Forward correctness: ``base + (U * gate) @ V`` matches a hand-computed
     reference; gate values affect the output exactly as expected.
  2. Initial state: with U=0, forward() == base regardless of gate values
     (the warm-start identity holds).
  3. Gradient flow: gate logits + U + V all receive gradients (so the
     optimiser CAN drive a gate to zero).
  4. Pruning: kept_indices() respects the threshold; degenerate "all zero"
     case returns the strongest gate so we never serialise rank=0.
  5. Encode/decode round-trip: pruned dict materialises to the SAME pose
     tensor that the un-pruned forward produced (modulo the pruned ranks
     themselves contributing zero by construction).
  6. Loader integration: ``tac.submission_archive.load_optimized_poses``
     auto-detects the V2 sentinel and returns (N, 6) transparently.
  7. Schema validation: bad V2 dicts raise ValueError with diagnostics.
  8. CLI integration: ``--learnable-lora-max-rank`` exists in
     ``experiments/optimize_poses.py``'s argparse.
  9. Mutual exclusion: --lora-rank + --learnable-lora-max-rank both > 0
     is a hard error (the optimize_poses.py path raises SystemExit).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from tac.lora_pose_v2 import (
    DEFAULT_PRUNE_THRESHOLD,
    LORA_FORMAT_SENTINEL_V2,
    LearnableRankLoRAPose,
    decode_lora_v2_poses_dict,
    encode_lora_v2_poses_dict,
    is_lora_v2_poses_dict,
    save_lora_v2_poses,
)
from tac.submission_archive import load_optimized_poses

REPO = Path(__file__).resolve().parents[3]


# ── Forward correctness ───────────────────────────────────────────────────


def test_v2_forward_matches_gated_reference() -> None:
    """forward() == base + (U * sigmoid(logit_gate)) @ V (exact, fp32)."""
    torch.manual_seed(0)
    base = torch.randn(13, 6) * 0.01
    lora = LearnableRankLoRAPose(base=base, max_rank=4, init_gate_logit=0.0)
    with torch.no_grad():
        lora.U.fill_(0.5)
        lora.V.fill_(0.25)
        lora.logit_gate.copy_(torch.tensor([2.0, -3.0, 0.0, 1.0]))
    expected_gate = torch.sigmoid(lora.logit_gate)
    expected = base + (lora.U * expected_gate.unsqueeze(0)) @ lora.V
    out = lora()
    assert torch.allclose(out, expected, atol=1e-6)


def test_v2_initial_state_is_warm_start() -> None:
    """U is zero-initialised so forward() == base regardless of gate values."""
    torch.manual_seed(0)
    base = torch.randn(7, 6) * 0.01
    # Even with crazy gate logits, U=0 means U@V=0 and forward()==base.
    lora = LearnableRankLoRAPose(base=base, max_rank=3, init_gate_logit=5.0)
    out = lora()
    assert torch.allclose(out, base, atol=1e-6), (
        "warm-start identity must hold at U=0 regardless of gate values"
    )


def test_v2_shape_contracts() -> None:
    base = torch.zeros(11, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=4)
    assert lora.U.shape == (11, 4)
    assert lora.V.shape == (4, 6)
    assert lora.logit_gate.shape == (4,)
    assert lora().shape == (11, 6)
    assert lora.trainable_params == 11 * 4 + 4 * 6 + 4


def test_v2_invalid_max_rank_rejected() -> None:
    base = torch.zeros(10, 6)
    with pytest.raises(ValueError):
        LearnableRankLoRAPose(base=base, max_rank=0)
    with pytest.raises(ValueError):
        LearnableRankLoRAPose(base=base, max_rank=-1)


def test_v2_rejects_non_2d_base() -> None:
    bad = torch.zeros(10, 6, 1)
    with pytest.raises(ValueError):
        LearnableRankLoRAPose(base=bad, max_rank=2)


# ── Gradient flow ─────────────────────────────────────────────────────────


def test_v2_gradients_flow_to_U_V_and_gate() -> None:
    """All three trainable tensors must receive gradients in steady state."""
    torch.manual_seed(1)
    base = torch.randn(9, 6) * 0.01
    lora = LearnableRankLoRAPose(base=base, max_rank=3)
    with torch.no_grad():
        # Break U=0 so dL/dV and dL/dlogit_gate are non-zero (dL/dU is
        # non-zero anyway via dL/dout @ V.T as long as gate is non-zero).
        lora.U.normal_(0, 0.01)
    out = lora()
    target = torch.zeros_like(out)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    assert lora.U.grad is not None and lora.U.grad.abs().sum().item() > 0
    assert lora.V.grad is not None and lora.V.grad.abs().sum().item() > 0
    assert lora.logit_gate.grad is not None
    assert lora.logit_gate.grad.abs().sum().item() > 0, (
        "dL/dlogit_gate must be non-zero — otherwise the optimiser cannot "
        "prune ranks and the LEARNABLE-rank claim is hollow"
    )
    # base must remain a buffer, not a parameter.
    param_ids = {id(p) for p in lora.parameters()}
    assert id(lora.base) not in param_ids


def test_v2_gate_can_be_driven_toward_zero() -> None:
    """A few SGD steps with a gate-zero target must reduce the gate logit."""
    torch.manual_seed(2)
    base = torch.randn(20, 6) * 0.01
    lora = LearnableRankLoRAPose(base=base, max_rank=3, init_gate_logit=0.0)
    with torch.no_grad():
        lora.U.normal_(0, 0.05)
    optim = torch.optim.SGD(
        [lora.U, lora.V, lora.logit_gate], lr=0.5,
    )
    # Optimise toward base (so the optimiser wants U @ V = 0, which is
    # achieved by driving gate -> 0).
    initial_gate = lora.gate.detach().clone()
    for _ in range(20):
        optim.zero_grad()
        loss = ((lora() - base) ** 2).mean()
        loss.backward()
        optim.step()
    final_gate = lora.gate.detach().clone()
    # At least one gate must have moved (toward zero, since the optimiser
    # is trying to make U@V vanish and gate=0 achieves that with any U/V).
    assert (final_gate < initial_gate).any(), (
        f"gates failed to move toward zero; initial={initial_gate.tolist()} "
        f"final={final_gate.tolist()}"
    )


# ── Pruning ───────────────────────────────────────────────────────────────


def test_v2_kept_indices_respects_threshold() -> None:
    base = torch.zeros(5, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=4)
    with torch.no_grad():
        # gate values: 0.95, 0.05, 0.50, 0.01 (via inverse sigmoid)
        lora.logit_gate.copy_(torch.tensor([2.944, -2.944, 0.0, -4.595]))
    keep = lora.kept_indices(prune_threshold=0.1)
    assert keep == [0, 2], f"expected [0, 2], got {keep}"
    # Tighter threshold drops the 0.5 gate too:
    keep_strict = lora.kept_indices(prune_threshold=0.6)
    assert keep_strict == [0]


def test_v2_kept_indices_never_returns_empty() -> None:
    """Degenerate case: every gate < threshold. Must keep the strongest."""
    base = torch.zeros(5, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=4)
    with torch.no_grad():
        # All gates ~0.001
        lora.logit_gate.copy_(torch.tensor([-7.0, -8.0, -6.0, -9.0]))
    keep = lora.kept_indices(prune_threshold=0.1)
    assert len(keep) == 1, f"expected fallback to argmax, got {keep}"
    # The strongest of the 4 (least-negative logit) was index 2 (-6.0).
    assert keep == [2]


# ── Encode / decode round-trip ───────────────────────────────────────────


def test_v2_encode_dict_has_canonical_schema() -> None:
    base = torch.zeros(5, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=3)
    with torch.no_grad():
        # Force two ranks to survive (others get sigmoid(0)=0.5 = above 0.1).
        lora.logit_gate.copy_(torch.tensor([2.0, 2.0, -5.0]))
    obj = encode_lora_v2_poses_dict(lora, prune_threshold=0.1)
    assert obj["format"] == LORA_FORMAT_SENTINEL_V2
    assert obj["rank"] == 2
    assert obj["max_rank"] == 3
    assert obj["kept_indices"] == [0, 1]
    assert obj["n_pairs"] == 5
    assert obj["pose_dim"] == 6
    assert obj["base"].dtype == torch.float16
    assert obj["U"].dtype == torch.float16
    assert obj["V"].dtype == torch.float16
    assert obj["U"].shape == (5, 2)
    assert obj["V"].shape == (2, 6)
    assert obj["final_gate_values"].shape == (3,)
    assert is_lora_v2_poses_dict(obj)


def test_v2_decode_round_trip_matches_forward_full_rank() -> None:
    """When NO pruning happens, decoded poses == forward output (modulo fp16)."""
    torch.manual_seed(3)
    base = torch.randn(50, 6) * 0.01
    lora = LearnableRankLoRAPose(base=base, max_rank=2, init_gate_logit=4.0)
    # Initialise U/V to non-zero so forward != base.
    with torch.no_grad():
        lora.U.normal_(0, 0.05)
        lora.V.normal_(0, 0.05)
    # Both gates are sigmoid(4) ≈ 0.982 so both survive a 0.1 threshold.
    pre = lora().detach()
    obj = encode_lora_v2_poses_dict(lora, prune_threshold=0.1)
    assert obj["rank"] == 2  # both ranks kept
    decoded = decode_lora_v2_poses_dict(obj, pose_dim=6)
    assert decoded.shape == pre.shape
    assert decoded.dtype == torch.float32
    assert torch.allclose(decoded, pre, atol=1e-2)


def test_v2_decode_round_trip_matches_pruned_forward() -> None:
    """When pruning drops ranks, decoded poses == forward with those ranks
    forced to gate=0 (i.e. their U@V contribution removed)."""
    torch.manual_seed(4)
    base = torch.randn(20, 6) * 0.01
    lora = LearnableRankLoRAPose(base=base, max_rank=4)
    with torch.no_grad():
        lora.U.normal_(0, 0.1)
        lora.V.normal_(0, 0.1)
        # Two strong, two weak gates.
        lora.logit_gate.copy_(torch.tensor([3.0, -5.0, 2.5, -6.0]))
    obj = encode_lora_v2_poses_dict(lora, prune_threshold=0.1)
    assert obj["kept_indices"] == [0, 2]
    decoded = decode_lora_v2_poses_dict(obj, pose_dim=6)
    # Manual reference: gate the U columns by sigmoid(logit), then drop the
    # weak ones (their gated U is ~0 anyway, but we drop them entirely).
    with torch.no_grad():
        gate = lora.gate.detach().cpu()
        ref_U = (lora.U.detach().cpu() * gate.unsqueeze(0))[:, [0, 2]]
        ref_V = lora.V.detach().cpu()[[0, 2], :]
        ref_poses = base + ref_U @ ref_V
    assert torch.allclose(decoded, ref_poses, atol=1e-2)


def test_v2_save_load_round_trip_via_load_optimized_poses(
    tmp_path: Path,
) -> None:
    """End-to-end: save_lora_v2_poses + load_optimized_poses roundtrip."""
    torch.manual_seed(5)
    base = torch.randn(600, 6) * 0.01
    lora = LearnableRankLoRAPose(base=base, max_rank=6, init_gate_logit=3.0)
    with torch.no_grad():
        lora.U.normal_(0, 0.05)
        lora.V.normal_(0, 0.05)
    out = tmp_path / "optimized_poses.pt"
    n_bytes = save_lora_v2_poses(lora, out, prune_threshold=0.1)
    assert n_bytes > 0
    poses = load_optimized_poses(out, pose_dim=6, expected_n_pairs=600)
    assert poses.shape == (600, 6)
    assert poses.dtype == torch.float32
    # All gates start at sigmoid(3)≈0.953 → all 6 ranks kept; reconstruction
    # should match the un-pruned forward to within fp16 tolerance.
    pre = lora().detach()
    assert torch.allclose(poses, pre, atol=1e-2)


# ── Schema validation ────────────────────────────────────────────────────


def test_v2_decode_rejects_wrong_sentinel() -> None:
    bad = {
        "format": "lora_pose_v1",  # V1 sentinel must NOT be accepted by V2
        "rank": 1, "max_rank": 1, "kept_indices": [0],
        "n_pairs": 2, "pose_dim": 6,
        "base": torch.zeros(2, 6),
        "U": torch.zeros(2, 1), "V": torch.zeros(1, 6),
        "final_gate_values": torch.zeros(1),
    }
    with pytest.raises(ValueError, match="not a LoRA-V2"):
        decode_lora_v2_poses_dict(bad, pose_dim=6)


def test_v2_decode_rejects_missing_key() -> None:
    base = torch.zeros(3, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=2)
    obj = encode_lora_v2_poses_dict(lora)
    del obj["V"]
    with pytest.raises(ValueError, match="missing required key"):
        decode_lora_v2_poses_dict(obj, pose_dim=6)


def test_v2_decode_rejects_pose_dim_mismatch() -> None:
    base = torch.zeros(3, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=1)
    obj = encode_lora_v2_poses_dict(lora)
    with pytest.raises(ValueError, match="pose_dim"):
        decode_lora_v2_poses_dict(obj, pose_dim=4)


def test_v2_decode_rejects_kept_indices_length_mismatch() -> None:
    base = torch.zeros(3, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=2)
    obj = encode_lora_v2_poses_dict(lora)
    obj["kept_indices"] = [0]  # rank says 2 but indices says 1
    with pytest.raises(ValueError, match="kept_indices"):
        decode_lora_v2_poses_dict(obj, pose_dim=6)


def test_v2_decode_rejects_rank_above_max_rank() -> None:
    base = torch.zeros(3, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=2)
    obj = encode_lora_v2_poses_dict(lora)
    obj["rank"] = 99
    obj["kept_indices"] = list(range(99))
    with pytest.raises(ValueError, match=r"rank"):
        decode_lora_v2_poses_dict(obj, pose_dim=6)


def test_v2_decode_rejects_wrong_shapes() -> None:
    base = torch.zeros(3, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=2)
    obj = encode_lora_v2_poses_dict(lora)
    obj["U"] = torch.zeros(99, obj["rank"])  # wrong N
    with pytest.raises(ValueError, match="U shape"):
        decode_lora_v2_poses_dict(obj, pose_dim=6)


# ── Loader integration smoke ─────────────────────────────────────────────


def test_v2_load_optimized_poses_dispatches_v1_v2_and_tensor(
    tmp_path: Path,
) -> None:
    """The canonical loader handles all THREE on-disk formats."""
    from tac.lora_pose import LoRAPose, save_lora_poses

    base = torch.randn(20, 6) * 0.01
    # V1 path
    v1 = LoRAPose(base=base, rank=1)
    with torch.no_grad():
        v1.U.normal_(0, 0.05)
    v1_path = tmp_path / "v1.pt"
    save_lora_poses(v1, v1_path)
    p1 = load_optimized_poses(v1_path, pose_dim=6, expected_n_pairs=20)
    assert p1.shape == (20, 6)

    # V2 path
    v2 = LearnableRankLoRAPose(base=base, max_rank=2, init_gate_logit=4.0)
    with torch.no_grad():
        v2.U.normal_(0, 0.05)
    v2_path = tmp_path / "v2.pt"
    save_lora_v2_poses(v2, v2_path, prune_threshold=0.1)
    p2 = load_optimized_poses(v2_path, pose_dim=6, expected_n_pairs=20)
    assert p2.shape == (20, 6)

    # Vanilla tensor path
    plain = torch.randn(20, 6, dtype=torch.float32) * 0.01
    plain_path = tmp_path / "plain.pt"
    torch.save(plain, plain_path)
    p3 = load_optimized_poses(plain_path, pose_dim=6, expected_n_pairs=20)
    assert torch.allclose(p3, plain, atol=1e-6)


# ── CLI integration: --learnable-lora-max-rank exists in argparse ────────


def test_v2_cli_flag_registered_in_optimize_poses() -> None:
    """experiments/optimize_poses.py MUST register --learnable-lora-max-rank,
    --lora-prune-threshold, and --lora-init-gate-logit so the remote
    bootstrap script can pass them. Catches the dead-flag-wiring bug class
    (memory: feedback_dead_flag_wiring_pattern)."""
    src = (
        REPO / "experiments" / "optimize_poses.py"
    ).read_text()
    # Match `add_argument("--flag-name"`
    add_re = re.compile(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)")
    flags = set(add_re.findall(src))
    for required in (
        "learnable-lora-max-rank",
        "lora-prune-threshold",
        "lora-init-gate-logit",
    ):
        assert required in flags, (
            f"experiments/optimize_poses.py is missing --{required} in "
            f"argparse — Lane LR-V2 CLI wiring is broken"
        )


def test_v2_mutual_exclusion_documented_in_optimize_poses() -> None:
    """The mutual-exclusion gate (--lora-rank + --learnable-lora-max-rank
    both > 0 → SystemExit) must be present in the script. Source-grep the
    explicit FATAL message so a refactor that removes the gate trips the
    test."""
    src = (
        REPO / "experiments" / "optimize_poses.py"
    ).read_text()
    assert (
        "mutually" in src and "learnable_lora_max_rank" in src
    ), (
        "experiments/optimize_poses.py is missing the V1/V2 mutual-exclusion "
        "gate; the operator could pass both --lora-rank and "
        "--learnable-lora-max-rank and the V2 pass would silently supersede"
    )


# ── Rate budget audit ────────────────────────────────────────────────────


def test_v2_archive_bytes_decreases_with_pruning() -> None:
    """The headline claim: more pruning → smaller archive."""
    base = torch.zeros(600, 6)
    lora = LearnableRankLoRAPose(base=base, max_rank=6)
    full = lora.archive_bytes_fp16()
    pruned_to_2 = lora.archive_bytes_fp16(kept_indices=[0, 1])
    pruned_to_1 = lora.archive_bytes_fp16(kept_indices=[0])
    assert pruned_to_2 < full, (
        f"pruning to 2 ranks should reduce archive bytes "
        f"(full={full}, pruned={pruned_to_2})"
    )
    assert pruned_to_1 < pruned_to_2 < full


def test_v2_default_prune_threshold_is_documented() -> None:
    assert DEFAULT_PRUNE_THRESHOLD == 0.1, (
        "Lane LR-V2 default prune threshold is contractually 0.1; "
        "tests + docs assume this value"
    )
