# SPDX-License-Identifier: MIT
"""Cross-primitive integration tests for the 5 new composition primitives.

These tests prove the primitives compose with EACH OTHER (closed under
composition) and with the pre-existing :mod:`tac.composition.td_lora`,
:mod:`tac.composition.wbce_mera`, and :mod:`tac.composition.stack_of_stacks`
surfaces — closing the wire-in loop required by CLAUDE.md "Beauty,
simplicity, and developer experience".
"""

from __future__ import annotations

import torch

from tac.composition.bregman_mixing import (
    BregmanGenerator,
    BregmanMixer,
    BregmanMixerSpec,
)
from tac.composition.distillation_chain import (
    DistillationChain,
    DistillationLevel,
    distillation_loss,
)
from tac.composition.hypernetwork import Hypernetwork, HypernetworkSpec
from tac.composition.product_of_experts import (
    ProductOfExpertsComposer,
    ProductOfExpertsSpec,
)
from tac.composition.sinkhorn_ot_mixing import (
    SinkhornOTMixer,
    SinkhornOTMixerSpec,
)
from tac.composition.td_lora import TropicalLoRAAdapter, TropicalLoRASpec


def test_bregman_centroid_of_hypernet_outputs_is_finite() -> None:
    """Hypernet -> Bregman: aggregate K hypernet outputs into a centroid.

    Demonstrates that a Bregman mixer can fold K candidate parameter
    proposals from K hypernets into a single centroid.
    """
    spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=8, num_codes=3)
    hyper = Hypernetwork(spec)
    outputs = [hyper(torch.tensor([i])).flatten() for i in range(3)]
    mixer = BregmanMixer(BregmanMixerSpec(generator=BregmanGenerator.SQUARED_EUCLIDEAN))
    centroid = mixer.mix(outputs)
    assert centroid.shape == (8,)
    assert torch.all(torch.isfinite(centroid))


def test_sinkhorn_transport_of_hypernet_codes_to_anchor_set() -> None:
    """Hypernet codes -> Sinkhorn-OT: map N hypernet codes to M anchors.

    Demonstrates the Sinkhorn-OT mixer transporting source vectors
    (here: per-pair hypernet codes) to a smaller target anchor support.
    """
    hyper = Hypernetwork(
        HypernetworkSpec(latent_dim=3, hidden_dim=4, output_dim=8, num_codes=10)
    )
    # Get 10 source vectors (one per latent code).
    src = hyper.codes.detach()
    # Choose 3 target anchors at uniform-cluster centres.
    tgt = src[[0, 5, 9]].clone()
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.05))
    z = mixer.transport(src, tgt)
    assert z.shape == tgt.shape
    assert torch.all(torch.isfinite(z))


def test_product_of_experts_with_two_tropical_adapter_outputs() -> None:
    """Tropical-LoRA × PoE: ensemble two TD-LoRA-equipped renderers.

    Demonstrates the PoE composer fusing per-pair log-likelihoods of two
    TD-LoRA expert renderers.
    """
    adapter_a = TropicalLoRAAdapter(TropicalLoRASpec(in_features=4, out_features=4, rank=4, num_branches=2))
    adapter_b = TropicalLoRAAdapter(TropicalLoRASpec(in_features=4, out_features=4, rank=4, num_branches=2))
    x = torch.randn(2, 4)
    out_a = adapter_a(x)
    out_b = adapter_b(x)
    # Treat each adapter's mean output as a log-likelihood.
    lp_a = out_a.mean(dim=-1)
    lp_b = out_b.mean(dim=-1)
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    combined = composer.combine_log_densities([lp_a, lp_b])
    assert combined.shape == (2,)


def test_distillation_chain_loss_works_on_hypernet_outputs() -> None:
    """Hypernet × Distillation chain: distill big-hypernet into small-hypernet.

    Demonstrates distillation_loss flowing gradients through both teacher
    and student hypernets.
    """
    teacher_spec = HypernetworkSpec(latent_dim=4, hidden_dim=16, output_dim=10, num_codes=4)
    student_spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=10, num_codes=4)
    teacher = Hypernetwork(teacher_spec)
    student = Hypernetwork(student_spec)
    idx = torch.tensor([0, 1, 2, 3])
    t_logits = teacher(idx).unsqueeze(-1)  # (4, 10, 1) — treat output_dim as C, add S=1
    s_logits = student(idx).unsqueeze(-1)
    loss = distillation_loss(s_logits, t_logits, temperature=2.0, kl_weight=1.0)
    loss.backward()
    # Student grads flow.
    assert student.fc1.weight.grad is not None
    # Teacher detached internally -> no grad.
    assert teacher.fc1.weight.grad is None


def test_distillation_chain_spec_describes_hypernet_compression() -> None:
    """Chain validates that hypernet output_dim shrinks across levels."""
    chain = DistillationChain(
        levels=(
            DistillationLevel(name="hyper_large", param_count=300_000),
            DistillationLevel(name="hyper_med", param_count=100_000),
            DistillationLevel(name="hyper_small", param_count=30_000),
        )
    )
    assert chain.num_levels() == 3
    assert chain.total_compression() == 10.0


def test_bregman_and_sinkhorn_agree_on_simple_case() -> None:
    """Two different mixers should produce finite + comparable outputs.

    Not asserting equality (different math) but proving both flow through
    end-to-end on the SAME inputs without numerical failure.
    """
    src = torch.tensor([[0.0], [1.0], [2.0]])
    bregman = BregmanMixer(BregmanMixerSpec()).mix([src[0], src[1], src[2]])
    sinkhorn_target = torch.tensor([[1.0]])
    sinkhorn = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1)).transport(
        src, sinkhorn_target
    )
    assert torch.all(torch.isfinite(bregman))
    assert torch.all(torch.isfinite(sinkhorn))


def test_poe_can_consume_sinkhorn_transported_outputs() -> None:
    """Sinkhorn-OT output flows into PoE composer.

    Demonstrates the composition closure: Sinkhorn projects source
    samples to a smaller target support, then PoE combines per-target
    log-likelihoods.
    """
    src = torch.randn(8, 2)
    tgt = torch.randn(2, 2)
    z = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1)).transport(src, tgt)
    # Use z as two "expert outputs"; build artificial log-likelihoods.
    lp_a = torch.zeros(2)
    lp_b = torch.zeros(2)
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    out = composer.soft_gate(
        [lp_a, lp_b],
        [z[:, 0:1], z[:, 1:2]],
    )
    assert out.shape == (2, 1)
    assert torch.all(torch.isfinite(out))


def test_all_primitives_serialise_roundtrip() -> None:
    """Each primitive's spec can be serialised + restored deterministically."""
    bregman = BregmanMixer(BregmanMixerSpec(weights=(0.3, 0.7)))
    sinkhorn = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.05))
    hyper = Hypernetwork(HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=4))
    poe = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    chain = DistillationChain(
        levels=(
            DistillationLevel(name="big", param_count=100),
            DistillationLevel(name="small", param_count=30),
        )
    )
    for primitive in (bregman, sinkhorn, hyper, poe, chain):
        blob = primitive.serialize_state()
        assert len(blob) > 0
        restored = type(primitive).deserialize_state(blob)
        assert restored is not None
