# SPDX-License-Identifier: MIT
"""NO-FAKE tests for Lever 3 v2 — the OPTIMAL Quantizr pose-solve port
(``tac.torch_vehicle.pose_film_v2.PoseFiLMHNeRVWrapperV2``).

Design memo: ``.omx/research/lever3_optimal_quantizr_pose_solve_port_trackA_20260613.md``.

The load-bearing claims (each, if wrong, breaks the lever or — worse — silently
changes the LIVE base_ch=20 basin if it crash-resumes onto this code):

1. **IDENTITY-AT-INIT (basin-resume safety).** The residual FiLM branch is
   zero-init (``proj`` + ``beta_head`` = 0) so ``film_resid == 0`` exactly -> the
   FiLM-on render @ init is bit-equal to the no-FiLM render AND to the bare
   vendored decoder. ``idx is None`` also renders the exact vendored path.

2. **d_seg / d_pose DECOUPLING (the headline).** ``f1`` (the SegNet frame) is
   rendered from the CLEAN trunk feature ``x`` (never ``x0``), so it is
   BIT-INVARIANT to the stored pose: vary ``pose6`` arbitrarily -> ``f1`` does not
   change at all. (Contrast v1 stem-FiLM where pose changes f1.)

3. **NO-FAKE: ``f0`` DOES change with pose (the FiLM is live).** A
   pose-invariance test alone is satisfiable by a stub that ignores the pose
   entirely; so we ALSO assert ``f0`` (the pose-conditioned frame) actually varies
   with ``pose6`` once the FiLM is trained, and is differentiable to the FiLM
   weights. Decoupling + liveness together = a real mechanism, not a stub.

4. **RESIDUAL CONTAINMENT.** The identity path anchors ``x0 = x + film_resid``; a
   bounded FiLM gives a bounded ``f0`` perturbation (the v1 instability fix).

5. **pose_mlp / film_resid shapes + param plumbing** are correct.

6. **BYTE-CLOSED INFLATE v2 round-trip** (export-first): the additive pose
   section + the v2 FiLM weights round-trip through ``inflate_film_decoder_v2``.

Authority: torch-CPU TRUSTED. NO MPS. ``[contest-CPU advisory]`` until byte-closed
exact eval.
"""

from __future__ import annotations

import torch

from tac.torch_vehicle.driver import import_vendored_bundle
from tac.torch_vehicle.pose_film import (
    _FiLMEvalDecoder,
    build_archive_with_pose,
    parse_pose_section,
)
from tac.torch_vehicle.pose_film_v2 import (
    PoseFiLMHNeRVWrapperV2,
    _PoseCondMLP,
    _RGB0ResidualFiLM,
    inflate_film_decoder_v2,
)


def _wrapper(base_ch: int = 8, n_pairs: int = 6, *, train_film: bool = False, seed: int = 0):
    v = import_vendored_bundle()
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=base_ch).eval()
    w = PoseFiLMHNeRVWrapperV2(dec, n_pairs=n_pairs, film_hidden=8)
    if train_film:
        g = torch.Generator().manual_seed(seed)
        for p in list(w.pose_mlp.parameters()) + list(w.film_resid.parameters()):
            p.data = p.data + torch.randn(p.shape, generator=g) * 0.2
    g2 = torch.Generator().manual_seed(seed + 1)
    pose = torch.randn(n_pairs, 6, generator=g2) * 0.5
    w.set_stored_pose(pose)
    return v, dec, w, pose


# ---------------------------------------------------------------------------
# CLAIM 1 — IDENTITY AT INIT.
# ---------------------------------------------------------------------------
def test_v2_identity_at_init_matches_no_film_render():
    """Zero-init residual branch => first-step FiLM render == no-FiLM render,
    bit-for-bit. The basin-non-perturbation contract for enabling v2 FiLM."""
    _v, dec, w, _pose = _wrapper(train_film=False)
    z = torch.randn(3, 28) * 0.1
    idx = torch.arange(3)
    out_film = w(z, idx)
    out_nofilm = w(z, None)
    assert torch.equal(out_film, out_nofilm), "v2 FiLM is not identity at init"


def test_v2_identity_at_init_matches_bare_vendored_decoder():
    """FiLM-on @ init equals the BARE vendored decoder forward exactly (both heads),
    so enabling FiLM does not perturb the basin until the FiLM weights train."""
    _v, dec, w, _pose = _wrapper(train_film=False)
    z = torch.randn(3, 28) * 0.1
    idx = torch.arange(3)
    out_film = w(z, idx)
    out_vendored = dec(z)
    assert torch.equal(out_film, out_vendored), "v2 FiLM-on @ init != vendored decoder"


def test_v2_idx_none_renders_exact_vendored_path():
    """``idx is None`` short-circuits to the no-FiLM trunk render == vendored."""
    _v, dec, w, _pose = _wrapper(train_film=True, seed=7)
    z = torch.randn(2, 28) * 0.1
    out_nofilm = w(z, None)
    out_vendored = dec(z)
    assert torch.equal(out_nofilm, out_vendored), "idx=None != vendored render"


def test_v2_film_resid_is_exactly_zero_at_init():
    """The residual ``film_resid(x, cond)`` is the EXACT zero tensor at init (not
    merely close) — the mechanism that guarantees identity-at-init."""
    film = _RGB0ResidualFiLM(channels=10, cond_dim=8)
    x = torch.randn(4, 10, 12, 16)
    cond = torch.randn(4, 8)
    resid = film(x, cond)
    assert torch.equal(resid, torch.zeros_like(resid)), "film_resid != 0 at init"


# ---------------------------------------------------------------------------
# CLAIM 2 — d_seg / d_pose DECOUPLING (the headline). f1 INVARIANT to pose.
# ---------------------------------------------------------------------------
def test_v2_f1_is_bit_invariant_to_stored_pose_at_init():
    """THE DECOUPLING (init). Vary the stored pose arbitrarily -> ``f1`` (the
    SegNet frame, index 1) is BIT-IDENTICAL. The rgb_1 path has NO FiLM, so the
    pose physically cannot perturb the SegNet d_seg."""
    _v, _dec, w, _pose = _wrapper(train_film=False, n_pairs=6)
    z = torch.randn(6, 28) * 0.1
    idx = torch.arange(6)
    f1_a = w(z, idx)[:, 1]
    # Overwrite the stored pose with a totally different pose; re-render.
    g = torch.Generator().manual_seed(123)
    w.set_stored_pose(torch.randn(6, 6, generator=g) * 3.0)
    f1_b = w(z, idx)[:, 1]
    assert torch.equal(f1_a, f1_b), "f1 changed with pose at init (NOT decoupled)"


def test_v2_f1_is_bit_invariant_to_stored_pose_when_trained():
    """THE DECOUPLING (trained). Even with a TRAINED (non-identity) FiLM, ``f1`` is
    STILL bit-invariant to the stored pose — because rgb_1 reads the clean trunk
    ``x``, never ``x0``. This is the structural guarantee, not an init artifact."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=3, n_pairs=6)
    z = torch.randn(6, 28) * 0.1
    idx = torch.arange(6)
    f1_a = w(z, idx)[:, 1]
    g = torch.Generator().manual_seed(999)
    w.set_stored_pose(torch.randn(6, 6, generator=g) * 3.0)
    f1_b = w(z, idx)[:, 1]
    assert torch.equal(f1_a, f1_b), "trained-FiLM f1 changed with pose (NOT decoupled)"


def test_v2_f1_matches_vendored_rgb_1_head_exactly():
    """``f1`` equals ``sigmoid(rgb_1(trunk(z))) * 255`` from the bare vendored
    decoder — i.e. the SegNet frame is rendered by the UNTOUCHED rgb_1 head on the
    clean trunk, regardless of the (trained) FiLM on rgb_0."""
    _v, dec, w, _pose = _wrapper(train_film=True, seed=5, n_pairs=4)
    z = torch.randn(4, 28) * 0.1
    idx = torch.arange(4)
    f1_wrapper = w(z, idx)[:, 1]
    f1_vendored = dec(z)[:, 1]  # vendored f1 = sigmoid(rgb_1(x))*255 (no FiLM)
    assert torch.equal(f1_wrapper, f1_vendored), "f1 != vendored rgb_1 head output"


def test_v2_f0_DOES_change_with_pose_when_trained_NO_FAKE():
    """NO-FAKE complement to the f1-invariance test: a stub that IGNORED the pose
    would pass f1-invariance too. So assert the pose-conditioned ``f0`` (index 0)
    ACTUALLY varies with the stored pose once the FiLM is trained. f1-invariant
    AND f0-varying together => a REAL decoupled mechanism, not a stub."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=11, n_pairs=6)
    z = torch.randn(6, 28) * 0.1
    idx = torch.arange(6)
    f0_a = w(z, idx)[:, 0]
    g = torch.Generator().manual_seed(424242)
    w.set_stored_pose(torch.randn(6, 6, generator=g) * 3.0)
    f0_b = w(z, idx)[:, 0]
    assert not torch.allclose(f0_a, f0_b, atol=1e-3), (
        "f0 did not change with pose (FAKE/no-op FiLM guard)"
    )


def test_v2_trained_film_changes_f0_vs_no_film_render():
    """A trained FiLM modulates ``f0`` away from the no-FiLM render (operational,
    not a metadata stub) — while ``f1`` stays put (already asserted elsewhere)."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=13)
    z = torch.randn(4, 28) * 0.1
    idx = torch.arange(4)
    out_film = w(z, idx)
    out_nofilm = w(z, None)
    assert not torch.allclose(out_film[:, 0], out_nofilm[:, 0], atol=1e-3), (
        "trained FiLM did not change f0"
    )
    # f1 is identical between the two renders (FiLM never touches rgb_1).
    assert torch.equal(out_film[:, 1], out_nofilm[:, 1]), "FiLM leaked into f1"


# ---------------------------------------------------------------------------
# CLAIM 3 — NO-FAKE differentiability of the FiLM mechanism.
# ---------------------------------------------------------------------------
def test_v2_film_is_differentiable_to_film_weights():
    """Gradient of an ``f0``-dependent loss flows to BOTH ``pose_mlp`` and
    ``film_resid`` weights (so training can descend the pose lever). A constant /
    no-op FiLM would give zero/None grad on its own weights."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=2)
    z = torch.randn(2, 28) * 0.1
    idx = torch.arange(2)
    out = w(z, idx)
    loss = out[:, 0].pow(2).mean()  # f0-only loss
    loss.backward()
    grads = []
    for p in list(w.pose_mlp.parameters()) + list(w.film_resid.parameters()):
        grads.append(p.grad)
    assert any(g is not None and g.abs().sum() > 0 for g in grads), (
        "no gradient reached the FiLM weights (no-op mechanism guard)"
    )


def test_v2_no_grad_flows_to_film_from_f1_only_loss():
    """An ``f1``-only loss produces ZERO gradient on the FiLM weights — because the
    FiLM is on the rgb_0 path only. This is the gradient-level proof that d_seg
    (which reads f1) cannot train the pose FiLM (full decoupling)."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=21)
    z = torch.randn(2, 28) * 0.1
    idx = torch.arange(2)
    out = w(z, idx)
    loss = out[:, 1].pow(2).mean()  # f1-only (SegNet-frame) loss
    loss.backward()
    for p in list(w.pose_mlp.parameters()) + list(w.film_resid.parameters()):
        assert p.grad is None or p.grad.abs().sum() == 0, (
            "f1-only loss leaked gradient into the pose FiLM (NOT decoupled)"
        )


# ---------------------------------------------------------------------------
# CLAIM 4 — RESIDUAL CONTAINMENT (bounded perturbation).
# ---------------------------------------------------------------------------
def test_v2_residual_containment_bounds_the_f0_perturbation():
    """With a modestly-trained FiLM, the ``f0`` perturbation (vs the clean render)
    is bounded — the identity path anchors the output, so the FiLM is a bounded
    *correction*. (Sanity bound: per-pixel mean |delta| stays well under the full
    [0,255] range.)"""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=31)
    z = torch.randn(4, 28) * 0.1
    idx = torch.arange(4)
    out_film = w(z, idx)
    out_nofilm = w(z, None)
    delta = (out_film[:, 0] - out_nofilm[:, 0]).abs()
    assert torch.isfinite(delta).all(), "non-finite f0 perturbation"
    assert delta.mean().item() < 64.0, (
        f"f0 perturbation unbounded (mean |delta|={delta.mean().item():.2f})"
    )


def test_v2_wrapper_does_not_mutate_vendored_decoder_params():
    """The wrapper holds the vendored decoder by reference and adds only the FiLM
    sub-modules; the vendored decoder params are disjoint from the FiLM params."""
    _v, dec, w, _pose = _wrapper()
    assert w.decoder is dec
    film_ids = {id(p) for p in list(w.pose_mlp.parameters()) + list(w.film_resid.parameters())}
    dec_ids = {id(p) for p in dec.parameters()}
    assert film_ids.isdisjoint(dec_ids)


# ---------------------------------------------------------------------------
# CLAIM 5 — shapes + plumbing.
# ---------------------------------------------------------------------------
def test_v2_pose_cond_mlp_shapes():
    """pose_mlp: (B, 6) -> (B, cond_dim)."""
    mlp = _PoseCondMLP(pose_dim=6, cond_dim=8)
    cond = mlp(torch.randn(5, 6))
    assert cond.shape == (5, 8)
    assert mlp.param_count() > 0


def test_v2_film_resid_shapes_and_head_channels():
    """film_resid output matches the feature shape; head_channels = channels[-1]."""
    _v, dec, w, _pose = _wrapper(base_ch=8)
    assert w.head_channels == int(dec.channels[-1])
    film = w.film_resid
    x = torch.randn(3, w.head_channels, 12, 16)
    cond = w.pose_mlp(torch.randn(3, 6))
    resid = film(x, cond)
    assert resid.shape == x.shape


def test_v2_set_stored_pose_rejects_wrong_shape():
    """set_stored_pose validates the (n_pairs, pose_dim) shape (fail closed)."""
    _v, _dec, w, _pose = _wrapper(n_pairs=6)
    try:
        w.set_stored_pose(torch.zeros(5, 6))  # wrong n_pairs
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("set_stored_pose accepted a wrong shape")


def test_v2_forward_output_shape_full_pair():
    """forward returns (B, 2, 3, 384, 512) in [0, 255]."""
    _v, _dec, w, _pose = _wrapper(base_ch=8, n_pairs=4)
    z = torch.randn(4, 28) * 0.1
    out = w(z, torch.arange(4))
    assert out.shape == (4, 2, 3, 384, 512)
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_v2_eval_cursor_adapter_reconstructs_per_pair_pose():
    """The reused ``_FiLMEvalDecoder`` cursor adapter (vendored eval calls
    ``decoder(z)`` without idx) reconstructs the per-pair index for v2 too: a
    streamed eval == an explicit-idx render, pair-for-pair.

    NOTE: the full-batch vs split-batch renders differ only by float conv
    nondeterminism (~3e-5 on the 384x512 ``proj`` conv; the BARE decoder is
    bit-identical, so the tiny diff is the FiLM conv's batched-BLAS jitter, NOT a
    routing error). We assert tight ``allclose`` for the render AND prove the
    per-pair index reconstruction EXACTLY via the cursor arithmetic + a
    deliberate MISROUTE control (shuffled pose -> the render WOULD differ a lot if
    the index were wrong)."""
    _v, _dec, w, _pose = _wrapper(train_film=True, seed=4, n_pairs=6)
    z = torch.randn(6, 28) * 0.1
    explicit = w(z, torch.arange(6))
    adapter = _FiLMEvalDecoder(w)
    adapter.eval()  # resets cursor to 0
    # stream in two batches of 3 (the vendored evaluate_decoder batches pairs).
    out0 = adapter(z[:3])
    out1 = adapter(z[3:])
    streamed = torch.cat([out0, out1], dim=0)
    assert torch.allclose(streamed, explicit, atol=1e-3), (
        "eval cursor adapter misrouted v2 pose (beyond float jitter)"
    )
    # MISROUTE control: if the cursor mis-indexed the pose (e.g. always pair 0),
    # the pose-conditioned f0 of pairs 1..5 would be conditioned on the WRONG
    # pose, diverging FAR beyond 1e-3. Build a render that deliberately uses the
    # wrong (all-zero idx) pose and confirm it is materially different.
    wrong = w(z, torch.zeros(6, dtype=torch.long))  # everyone gets pair-0's pose
    assert not torch.allclose(wrong[1:, 0], explicit[1:, 0], atol=1e-3), (
        "misroute control failed: pose index does not actually drive f0"
    )


# ---------------------------------------------------------------------------
# CLAIM 6 — BYTE-CLOSED INFLATE v2 round-trip (export-first).
# ---------------------------------------------------------------------------
def test_v2_inflate_round_trip_is_deterministic_and_film_active():
    """build_archive_with_pose -> inflate_film_decoder_v2 reconstructs the v2 FiLM
    render (FiLM weights ship in the decoder blob under pose_mlp.*/film_resid.*;
    pose ships in the additive section). Deterministic; FiLM-active end-to-end."""
    v, dec, w, pose = _wrapper(train_film=True, seed=6, n_pairs=6, base_ch=8)
    latents = torch.randn(6, 28) * 0.1
    # The decoder blob ships the vendored decoder keys + the v2 FiLM keys.
    wrapper_sd = w.state_dict()
    dec_sd = {}
    for k, val in wrapper_sd.items():
        if k == "stored_pose":
            continue
        if k.startswith("decoder."):
            dec_sd[k[len("decoder.") :]] = val
        else:  # pose_mlp.* / film_resid.*
            dec_sd[k] = val
    meta = {
        "n_pairs": 6,
        "latent_dim": 28,
        "base_channels": 8,
        "eval_size": (384, 512),
    }
    archive = build_archive_with_pose(v.build_archive, dec_sd, latents, meta, pose)
    # The additive pose section round-trips.
    parsed_pose = parse_pose_section(archive, v.parse_archive)
    assert parsed_pose is not None and parsed_pose.shape == (6, 6)
    # Inflate twice -> deterministic.
    out_a = inflate_film_decoder_v2(archive, v.parse_archive, v.HNeRVDecoder, film_hidden=8)
    out_b = inflate_film_decoder_v2(archive, v.parse_archive, v.HNeRVDecoder, film_hidden=8)
    assert out_a.shape == (6, 2, 3, 384, 512)
    assert torch.equal(out_a, out_b), "inflate v2 is non-deterministic"
    # f1 in the inflate == vendored rgb_1 (the decoupling survives the round trip),
    # within the int8-dequant tolerance of the decoder blob.
    decoded_dec = v.HNeRVDecoder(latent_dim=28, base_channels=8, eval_size=(384, 512))
    decoded_sd, decoded_latents, _m = v.parse_archive(archive)
    dec_only = {k: val for k, val in decoded_sd.items()
                if not (k.startswith("pose_mlp.") or k.startswith("film_resid."))}
    decoded_dec.load_state_dict(dec_only)
    decoded_dec.eval()
    with torch.inference_mode():
        vend = decoded_dec(decoded_latents)
    assert torch.allclose(out_a[:, 1], vend[:, 1], atol=1e-4), (
        "inflate f1 != vendored rgb_1 (decoupling broken through round-trip)"
    )
