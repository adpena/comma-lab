# SPDX-License-Identifier: MIT
"""The 5 canonical NeRV-family vehicle fidelity manifests (claimed vs actual).

Populated 2026-06-09 from the two audit memos +  by reading each carrier's
source (file:line cited in every ``MechanismEvidence.evidence``):

* ``.omx/research/deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md``
* ``.omx/research/snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md``

Each manifest is a frozen :class:`VehicleFidelityManifest`. The module's
``emit_all(...)`` writes them as durable JSON under
``.omx/state/vehicle_fidelity/`` (the Catalog #384 gate + operator dashboards
consume them). Per CLAUDE.md "NO FAKE IMPLEMENTATIONS": where the pact_nerv_vq
full-stack audit is still in flight, uncertain test ids carry the honest
:data:`AUDIT_PENDING` status (not a fabricated value).

The 5 vehicles + their headline claimed-vs-actual:

* ``hi_nerv``    — claims "HiNeRV" but is a skip-free vanilla NeRV; grid PE +
  ConvNeXt + bilinear-skip are present-but-OFF (config flags ``False``) or
  absent. HONEST: it explicitly declares the absent mechanisms, so it verifies.
* ``snerv_inverse_steg_carrier`` — FAITHFUL architecture (real DWT + official
  MFU/HFR/TUB conv blocks) but the trained MLX path optimized recon-MSE-only
  (``observed_segnet_distillation_weight=None`` at ep22399).
* ``pact_nerv_vq`` — GENUINE VQ (STE + EMA + commitment) but a skip-free
  PixelShuffle+sin decoder. PyTorch loss has the right shape; MLX long-run lane
  audit AUDIT_PENDING.
* ``sane_hnerv`` — docstring advertises "bilinear-skip" the forward NEVER
  implements. This is the canonical name-laundering FAIL: ``verify()`` raises.
* ``ff_nerv`` — skip-free + band-limited 64x64 DCT grid (HF absent by
  construction). No false claims; verifies.
"""

from __future__ import annotations

from pathlib import Path

from tac.substrates._shared.vehicle_fidelity_manifest import (
    AUDIT_PENDING,
    MechanismEvidence,
    VehicleFidelityManifest,
    emit_manifest,
)

__all__ = [
    "CANONICAL_VEHICLE_MANIFESTS",
    "FF_NERV",
    "HI_NERV",
    "PACT_NERV_VQ",
    "SANE_HNERV",
    "SNERV",
    "emit_all",
]

_HINERV_MEMO = (
    ".omx/research/deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md"
)
_FLEET_MEMO = (
    ".omx/research/snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md"
)


# ---------------------------------------------------------------------------
# 1. hi_nerv — named "HiNeRV" but a skip-free vanilla NeRV
# ---------------------------------------------------------------------------
HI_NERV = VehicleFidelityManifest(
    vehicle_id="hi_nerv",
    claimed_family="HiNeRV (Hierarchical NeRV, arXiv 2306.09818)",
    actual_mechanisms_present=(
        # The bilinear-skip path EXISTS as an opt-in config flag (added per F1),
        # but is OFF by default. Recorded present-with-OFF-note + a behavior test.
        MechanismEvidence(
            mechanism="bilinear_skip",
            evidence="src/tac/substrates/hi_nerv/architecture.py:154-160",
            test_id=AUDIT_PENDING,
            notes="opt-in via HinervConfig.use_bilinear_skip; OFF by default",
        ),
    ),
    mechanisms_absent=(
        # HiNeRV's two DEFINING features are present-in-code-but-OFF-by-default,
        # so the DEFAULT carrier does NOT have them: recorded absent (honest).
        "grid_pe",
        "positional_encoding",
        "terminal_refine",
        # The trained MLX path optimizes recon-MSE with distillation weights
        # defaulting to 0.0 (the shared harness; see SNeRV manifest).
        "scorer_objective_weights_nonzero",
        "residual_hf_path",
    ),
    docstring_claims=(
        # The module docstring advertises "Hierarchical NeRV with multi-scale
        # latents" + a 3-scale latent pyramid. It does NOT claim grid-PE /
        # bilinear-skip as PRESENT in the default forward (the config flags are
        # explicitly False), so this is an honest carrier about its absences.
        "hi_nerv architecture - Hierarchical NeRV with multi-scale latents (L0 SKETCH)",
        "The substrate's distinctive prior is multi-resolution representation",
        "use_hierarchical_feature_grid: bool = False  (grid PE OFF by default)",
        "use_convnext_blocks: bool = False  (ConvNeXt OFF by default)",
    ),
    tests_proving_mechanism=(),
    summary=(
        "CLAIMED HiNeRV; ACTUAL skip-free vanilla NeRV (sin+PixelShuffle). The "
        "two HiNeRV-defining features (grid PE, ConvNeXt) are present-in-code "
        "but OFF by default; bilinear-skip is an opt-in flag OFF by default. "
        "Honest about absences -> verifies. Empirical d_seg=0.5075 "
        "[macOS-MLX research-signal]."
    ),
    source_memos=(_HINERV_MEMO, _FLEET_MEMO),
)


# ---------------------------------------------------------------------------
# 2. snerv_inverse_steg_carrier — faithful arch, recon-MSE-trained
# ---------------------------------------------------------------------------
SNERV = VehicleFidelityManifest(
    vehicle_id="snerv_inverse_steg_carrier",
    claimed_family="SNeRV (Spectra-preserving NeRV, arXiv 2501.01681)",
    actual_mechanisms_present=(
        MechanismEvidence(
            mechanism="mfu_hfr_tub",
            evidence="src/tac/substrates/snerv_inverse_steg_carrier/official_mfu.py:295-344",
            test_id=AUDIT_PENDING,
            notes=(
                "GENUINE conv MFU/HFR/TUB w/ LeakyReLU + skip-concat (path B); "
                "real orthonormal DWT adjoint rel-residual 0.0 (dwt.py:196-279)"
            ),
        ),
        MechanismEvidence(
            mechanism="residual_hf_path",
            evidence="src/tac/substrates/snerv_inverse_steg_carrier/mlx_renderer.py:1306-1319",
            test_id=AUDIT_PENDING,
            notes="HFR head conv1->LeakyReLU(0.1)->conv2 is a genuine nonlinear HF path (path B)",
        ),
    ),
    mechanisms_absent=(
        # Trained recon-MSE-only at ep22399: observed_segnet_distillation_weight=None.
        "scorer_objective_weights_nonzero",
        # SNeRV is not an HNeRV/PR95 decoder; bilinear-skip + terminal refine + grid
        # PE + codebook VQ are not its mechanisms (declared absent for completeness).
        "bilinear_skip",
        "terminal_refine",
        "grid_pe",
        "positional_encoding",
        "codebook_vq",
    ),
    docstring_claims=(
        # The __init__ docstring advertises MFU/HFR/TUB + store-LF/generate-HF.
        # These ARE present (path B), so the MFU/HFR/TUB claim verifies.
        "SNeRV stores the (coarse) orthonormal-DWT approximation and GENERATES "
        "the high-frequency detail (MFU/HFR/TUB)",
        "DWT synthesis adjoint (dwt.py, rel-residual 0.0)",
    ),
    tests_proving_mechanism=(),
    summary=(
        "CLAIMED SNeRV; ACTUAL faithful (real DWT + conv MFU/HFR/TUB w/ "
        "nonlinearity). Failure is OBJECTIVE-STARVATION not fakery: trained "
        "recon-MSE-only (observed_segnet_distillation_weight=None) at ep22399, "
        "d_seg=0.711 [macOS-MLX research-signal]. Fix is config-only."
    ),
    source_memos=(_FLEET_MEMO,),
)


# ---------------------------------------------------------------------------
# 3. pact_nerv_vq — genuine VQ, skip-free decoder
# ---------------------------------------------------------------------------
PACT_NERV_VQ = VehicleFidelityManifest(
    vehicle_id="pact_nerv_vq",
    claimed_family="VQ-VAE NeRV (van den Oord 1711.00937 codebook + HNeRV decoder)",
    actual_mechanisms_present=(
        MechanismEvidence(
            mechanism="codebook_vq",
            evidence="src/tac/substrates/pact_nerv_vq/architecture.py:141-166",
            test_id=AUDIT_PENDING,
            notes=(
                "GENUINE: argmin nearest-codebook (141), STE z_e+(z_q-z_e).detach() "
                "(147), EMA cluster_size + Laplace smoothing (156-166), commitment "
                "loss ||z_e-sg(z_q)||^2 (144)"
            ),
        ),
    ),
    mechanisms_absent=(
        # _DsUpBlock = PixelShuffle(sin30(DepthSepConv)) — NO skip/PE/refine.
        "bilinear_skip",
        "terminal_refine",
        "grid_pe",
        "positional_encoding",
        "mfu_hfr_tub",
        "residual_hf_path",
        # PyTorch score_aware_loss is the right shape (frozen-scorer, no recon
        # base); but the MLX long-run lane routes through the shared recon-MSE
        # harness. The scorer-objective status of the TRAINED MLX path is the
        # in-flight audit -> recorded absent for the default MLX route + summary
        # notes the PyTorch path is correct. (Conservative honest absence.)
        "scorer_objective_weights_nonzero",
    ),
    docstring_claims=(
        # The module docstring advertises the VQ codebook + STE + EMA +
        # commitment — all GENUINELY present (architecture.py:141-166), so the
        # codebook_vq claim verifies. It does NOT advertise a bilinear-skip.
        "HNeRV-class implicit renderer with vector-quantized per-pair latents",
        "nearest codebook entry z_q via straight-through estimator (Bengio 2013)",
        "codebook updates via EMA (van den Oord 3.2)",
        "commitment loss ||z_e - sg(z_q)||^2",
    ),
    tests_proving_mechanism=(),
    summary=(
        "CLAIMED VQ-VAE NeRV; ACTUAL genuine VQ (STE+EMA+commitment) + genuine "
        "index consumption at inflate, BUT skip-free PixelShuffle+sin decoder "
        "(will mean-field regardless of VQ). PyTorch loss shape is correct; the "
        "MLX long-run lane scorer-objective status is AUDIT_PENDING."
    ),
    source_memos=(_FLEET_MEMO,),
)


# ---------------------------------------------------------------------------
# 4. sane_hnerv — THE canonical name-laundering FAIL (docstring claims a skip)
# ---------------------------------------------------------------------------
SANE_HNERV = VehicleFidelityManifest(
    vehicle_id="sane_hnerv",
    claimed_family="HNeRV-LC-v2 (PR101/PR100 canonical HNeRV with bilinear-skip)",
    actual_mechanisms_present=(),
    mechanisms_absent=(
        # Docstring claims "bilinear-skip" (lines 5, 27) but _UpBlock.forward
        # (139) = shuffle(act(conv(x))) and main forward (236) is a bare
        # `for block in self.blocks: h = block(h)` loop with NO skip accumulation.
        "bilinear_skip",
        "terminal_refine",
        "grid_pe",
        "positional_encoding",
        "mfu_hfr_tub",
        "codebook_vq",
        "residual_hf_path",
        "scorer_objective_weights_nonzero",
    ),
    docstring_claims=(
        # These verbatim docstring phrases CLAIM a bilinear-skip the code lacks.
        "canonical HNeRV with per-pair latent + bilinear-skip + sin activation",
        "(with bilinear-skip from each prior block)",
        "One Conv -> sin -> PixelShuffle(2) block, with optional bilinear-skip",
    ),
    tests_proving_mechanism=(),
    summary=(
        "CLAIMED HNeRV-with-bilinear-skip; ACTUAL skip-free PixelShuffle+sin. "
        "Docstring (architecture.py:5,27,122) advertises a bilinear-skip the "
        "forward (139, 236-237) NEVER implements -> verify() RAISES (Catalog "
        "#307 documentation-fake; the canonical name-laundering case)."
    ),
    source_memos=(_FLEET_MEMO,),
)


# ---------------------------------------------------------------------------
# 5. ff_nerv — skip-free + band-limited DCT grid (HF absent by construction)
# ---------------------------------------------------------------------------
FF_NERV = VehicleFidelityManifest(
    vehicle_id="ff_nerv",
    claimed_family="FFNeRV-style band-limited DCT-grid NeRV",
    actual_mechanisms_present=(),
    mechanisms_absent=(
        # _UpBlock = Conv->sin->PixelShuffle; main forward (256) is a bare loop
        # then IDCT2 of a 64x64 band-limited grid -> HF absent by construction.
        "bilinear_skip",
        "terminal_refine",
        "grid_pe",
        "positional_encoding",
        "mfu_hfr_tub",
        "codebook_vq",
        "residual_hf_path",
        "scorer_objective_weights_nonzero",
    ),
    docstring_claims=(
        # Honest docstring: it advertises a band-limited 64x64 DCT grid and
        # makes NO false mechanism claim. Verifies.
        "predicts band-limited 2D DCT coefficient grids per frame-pair",
        "64x64 frequency grid stores the lowest 64 horizontal x 64 vertical DCT",
        "inflate-time IDCT2 -> 384x512 RGB (interpolated from 64x64 reconstruction)",
    ),
    tests_proving_mechanism=(),
    summary=(
        "CLAIMED band-limited DCT-grid NeRV; ACTUAL exactly that (honest). "
        "Skip-free Conv->sin->PixelShuffle decoder + 64x64 DCT grid CANNOT "
        "represent boundary HF by construction -> strictly-worse mean-field "
        "variant. No false claims -> verifies."
    ),
    source_memos=(_FLEET_MEMO,),
)


CANONICAL_VEHICLE_MANIFESTS: tuple[VehicleFidelityManifest, ...] = (
    HI_NERV,
    SNERV,
    PACT_NERV_VQ,
    SANE_HNERV,
    FF_NERV,
)


def emit_all(repo_root: str | Path | None = None) -> list[Path]:
    """Emit all 5 canonical manifests as durable JSON; return the paths.

    Manifests are emitted with ``verify=False`` so the deliberately-laundering
    ``sane_hnerv`` manifest IS written (the Catalog #384 gate + operator review
    consume it to SURFACE the documentation-fake). Verification is the gate's
    job, not a precondition for recording the honest claimed-vs-actual truth.
    """
    return [emit_manifest(m, repo_root=repo_root, verify=False) for m in CANONICAL_VEHICLE_MANIFESTS]
