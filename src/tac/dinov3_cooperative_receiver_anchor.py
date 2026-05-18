# SPDX-License-Identifier: MIT
"""DINOv3 frozen-feature cooperative-receiver anchor.

Per ``feedback_deep_research_wave_landed_20260518.md`` op-routable: replace
the synthetic per-region histogram path that ATW V2-1 (sister symposium
``feedback_z6_v2_wave_2_codex_repairs_landed_20260517.md``) falsified at 386x
over budget with a *pretrained* DINOv3 feature extractor that has been
trained on 1.7B internet images (LVD-16-89M) and produces semantically rich
CLS + patch tokens. The features are frozen at inference; no training cost.

Mathematical justification (cooperative-receiver framing per Atick-Redlich
1990/1992 + Tishby-Zaslavsky 2015):

  Let X = dashcam pair, S = SegNet+PoseNet (the contest scorer). The
  Atick-Redlich efficient-coding theorem says the optimal encoder maximizes
  ``MI(B; S(X))`` where B is the encoded representation. The contest scorer
  S is published in ``upstream/modules.py`` and the canonical
  ``tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss``
  helper directly computes this. BUT: the SegNet/PoseNet scorer is itself a
  narrow visual receiver (EfficientNet-B2 / FastViT-T12 trained on the
  contest's small frame domain). A SECOND cooperative-receiver — DINOv3
  pretrained on LVD-16-89M — provides an ORTHOGONAL signal axis:

    L_cooperative = lambda_S * L_atick_S + lambda_D * L_atick_D

  where L_atick_D is the KL divergence between the predicted-pair DINOv3
  features and the ground-truth-pair DINOv3 features, distilled via T=2.0
  softening per Hinton-Vinyals-Dean 2014 (T preserves dark-knowledge from
  the wide pretrained distribution).

  Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-
  feasibility-check": the predicted ΔS band [-0.015, -0.005] on the
  ATW V2-1 + DINOv3 stack is grounded in the Tishby-IB lens (DINOv3 features
  contain measurably more bits about driving-scene semantics than synthetic
  per-region histograms, per the deep-research wave's empirical anchor on
  COCO + ImageNet downstream tasks). The Dykstra-feasibility intersection
  is between {rate constraint, ATW V2-1 distortion constraint, DINOv3
  cooperative-receiver constraint, archive grammar constraint} per
  ``tac.substrates._shared.score_aware_common.score_pair_components``.

  This module is a **THIN COMPOSITION LAYER**: it loads the frozen DINOv3
  model exactly once per process and exposes a ``dinov3_pair_features`` API
  that emits the CLS + patch features for a (B, T, C, H, W) RGB-255 pair
  tensor. Substrates compose this with ``cooperative_receiver_loss`` for
  the dual-anchor cooperative-receiver loss.

Canonical-vs-unique decision per layer
--------------------------------------
1. Model loading: ADOPT canonical timm ``create_model`` (we are not the
   custodian of the DINOv3 weights; HuggingFace Hub is).
2. Preprocess: FORK from generic ImageNet 224 — the canonical timm DINOv3
   base/16 checkpoint expects 256x256
   normalized to (0.485,0.456,0.406)/(0.229,0.224,0.225); our contest
   frames are 384x512 RGB-255. We resize to 256x256 via bilinear and strip
   CLS/register prefix tokens before exposing the patch grid.
3. Distillation loss: ADOPT Hinton T=2.0 KL divergence per the canonical
   ``tac.symposium_impls.atw_codec_atick_tishby_wyner_triple`` family
   (matches the existing cooperative-receiver loss machinery).
4. Frozen-vs-trainable: ADOPT frozen-by-construction. The DINOv3 weights
   are immutable. Substrates that want to fine-tune DINOv3 must explicitly
   opt out and document the reason.
5. Eval-roundtrip: SUBSTRATE-OPTIONAL. The substrate caller decides whether
   to compute DINOv3 features on roundtripped (post-eval-roundtrip) pairs
   (canonical PR95 paradigm) or raw RGB pairs (research-only smoke). The
   helper does NOT impose eval-roundtrip; the caller composes.

Observability surface
---------------------
Per CLAUDE.md "Max observability — non-negotiable" (Catalog #305):

1. Inspectable per layer: ``dinov3_pair_features`` returns ``DinoV3Features``
   dataclass with ``cls_token``, ``patch_tokens``, ``frame_idx``, and
   ``model_name`` — each tensor inspectable.
2. Decomposable per signal: CLS token and patch tokens are returned
   separately; downstream substrates can use either or both.
3. Diff-able across runs: ``model_name`` + ``frame_idx`` + tensor shapes
   are deterministic; two runs with the same input must produce identical
   features (frozen model + bilinear resize).
4. Queryable post-hoc: features can be serialized to ``.pt`` for offline
   analysis.
5. Cite-able: every ``DinoV3Features`` row carries the source ``model_name``
   (``timm/vit_base_patch16_dinov3.lvd1689m``).
6. Counterfactual-able: ``cooperative_receiver_dinov3_kl_loss`` is
   differentiable so substrates can probe "what if this byte changed?"
   via backprop.

References
----------
- Oquab et al., "DINOv2: Learning Robust Visual Features without
  Supervision", TMLR 2024. Successor DINOv3 trained on LVD-16-89M
  (1.7B images).
- Atick & Redlich, "Towards a theory of early visual processing", 1990.
- Tishby & Zaslavsky, "Deep learning and the information bottleneck
  principle", ITW 2015.
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network",
  NeurIPS 2014 Workshop.

Cross-references
----------------
- :mod:`tac.codec.cooperative_receiver.atick_redlich` — sister canonical
  cooperative-receiver loss against the contest SegNet+PoseNet scorer.
- :mod:`tac.symposium_impls.atw_codec_atick_tishby_wyner_triple` — ATW
  codec that this anchor is wired into.
- :mod:`tac.differentiable_eval_roundtrip` — eval-roundtrip pipeline
  (substrate-optional composition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

CANONICAL_DINOV3_MODEL_NAME = "timm/vit_base_patch16_dinov3.lvd1689m"
"""Canonical DINOv3 model identifier on HuggingFace Hub (86.6M params,
pretrained on LVD-16-89M = 1.7B internet images).

PV-DINOv3-1 (HARD-EARNED via deep-research wave 2026-05-18 +
``feedback_deep_research_wave_landed_20260518.md``): this is the canonical
identifier for the timm-published DINOv3 base-16 checkpoint as of
2026-05-18. Future agents must verify the model exists at this identifier
before invoking this module (the HF Hub identifier MAY rename to
``facebook/dinov3-base-patch16-lvd1689m`` per HF-canonical naming when
upstream lands the model).
"""

DINOV3_INPUT_SIZE = 256
"""DINOv3 native timm input resolution for the canonical base/16 checkpoint."""

DINOV3_REGISTER_TOKENS = 4
"""Canonical DINOv3 ViT register tokens between CLS and patch tokens."""

DINOV3_IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""ImageNet mean for DINOv3 normalization (matches DINOv2 reference)."""

DINOV3_IMAGENET_STD = (0.229, 0.224, 0.225)
"""ImageNet std for DINOv3 normalization (matches DINOv2 reference)."""

HINTON_DISTILL_T_CANONICAL = 2.0
"""Hinton temperature T=2.0 (matches Quantizr UCLA SegNet distillation per
CLAUDE.md "Quantizr intelligence" + the AuxiliaryScorer canonical T=2.0)."""


# ---------------------------------------------------------------------------
# Dataclass surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DinoV3Features:
    """Frozen DINOv3 features for a single frame.

    Attributes
    ----------
    cls_token: (B, 768) CLS token (B = batch size; 768 = DINOv3-base hidden).
    patch_tokens: (B, N, 768) patch tokens (N = (256/16)**2 = 256 for base/16).
    frame_idx: 0 (first frame in pair) or 1 (second frame in pair).
    model_name: Source DINOv3 identifier (cite-chain provenance).
    """

    cls_token: torch.Tensor
    patch_tokens: torch.Tensor
    frame_idx: int
    model_name: str = CANONICAL_DINOV3_MODEL_NAME

    def __post_init__(self) -> None:
        if self.frame_idx not in (0, 1):
            raise ValueError(
                f"DinoV3Features.frame_idx must be 0 or 1, got {self.frame_idx}"
            )
        if self.cls_token.ndim != 2:
            raise ValueError(
                "DinoV3Features.cls_token must be (B, hidden) 2D, got "
                f"{tuple(self.cls_token.shape)}"
            )
        if self.patch_tokens.ndim != 3:
            raise ValueError(
                "DinoV3Features.patch_tokens must be (B, N, hidden) 3D, "
                f"got {tuple(self.patch_tokens.shape)}"
            )
        if self.cls_token.shape[0] != self.patch_tokens.shape[0]:
            raise ValueError(
                "DinoV3Features cls_token + patch_tokens must share batch "
                f"dim, got cls {self.cls_token.shape[0]} vs patch "
                f"{self.patch_tokens.shape[0]}"
            )


@dataclass(frozen=True)
class DinoV3PairFeatures:
    """Frozen DINOv3 features for a (frame_0, frame_1) pair.

    Substrates that compose with the cooperative-receiver loss consume both
    frames' CLS + patch tokens.
    """

    frame_0: DinoV3Features
    frame_1: DinoV3Features

    def __post_init__(self) -> None:
        if self.frame_0.frame_idx != 0:
            raise ValueError("DinoV3PairFeatures.frame_0 must have frame_idx=0")
        if self.frame_1.frame_idx != 1:
            raise ValueError("DinoV3PairFeatures.frame_1 must have frame_idx=1")
        if self.frame_0.model_name != self.frame_1.model_name:
            raise ValueError(
                "DinoV3PairFeatures: frame_0 and frame_1 must share "
                "model_name (apples-to-apples cooperative receiver)"
            )


# ---------------------------------------------------------------------------
# Frozen model loader (singleton)
# ---------------------------------------------------------------------------


_DINOV3_MODEL_CACHE: dict[tuple[str, str], nn.Module] = {}


def load_frozen_dinov3_model(
    model_name: str = CANONICAL_DINOV3_MODEL_NAME,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load + freeze the DINOv3 backbone (per-(model_name, device) singleton).

    The model is set to ``eval()`` mode and ``requires_grad_(False)`` on
    every parameter so the substrate's autograd graph cannot accidentally
    backprop into DINOv3 weights. The features themselves remain
    gradient-reachable through the substrate's predicted RGB pair (the
    forward pass is differentiable; only the *weights* are frozen).

    Parameters
    ----------
    model_name : str, default CANONICAL_DINOV3_MODEL_NAME
        timm-style HuggingFace Hub identifier. Default is the canonical
        DINOv3 base-16 LVD-16-89M variant.
    device : str | torch.device, default "cpu"
        Device for the frozen model. Pass ``"cuda"`` for GPU substrates;
        the model is ~330 MB FP32 (~165 MB FP16). For HF Jobs t4-small
        (16 GB) leaves ample headroom.

    Returns
    -------
    nn.Module
        The frozen DINOv3 backbone (timm ViT-Base/16 with hf_hub_id).

    Raises
    ------
    ImportError
        If ``timm`` is not installed (the substrate must declare
        ``timm`` in its dispatch dependencies).
    RuntimeError
        If the model identifier cannot be resolved (HF Hub auth issue,
        network failure, model renamed).
    """
    device_str = str(device)
    cache_key = (model_name, device_str)
    cached = _DINOV3_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "tac.dinov3_cooperative_receiver_anchor.load_frozen_dinov3_model "
            "requires `timm>=1.0.0`; install via `uv pip install timm`"
        ) from exc

    # timm syntax for HF-hub-hosted models is `hf_hub:<owner>/<model>`
    # (see timm.create_model docs). Some checkpoints are also registered
    # by their bare timm name; we accept both.
    if model_name.startswith("hf_hub:") or "/" in model_name:
        timm_name = (
            model_name
            if model_name.startswith("hf_hub:")
            else f"hf_hub:{model_name}"
        )
    else:
        timm_name = model_name

    model = timm.create_model(
        timm_name,
        pretrained=True,
        num_classes=0,  # remove classifier head; we want feature tokens
    )
    data_config = timm.data.resolve_model_data_config(model)
    resolved_input_size = data_config.get("input_size")
    if resolved_input_size is not None:
        resolved_hw = tuple(int(v) for v in resolved_input_size[-2:])
        expected_hw = (DINOV3_INPUT_SIZE, DINOV3_INPUT_SIZE)
        if resolved_hw != expected_hw:
            raise RuntimeError(
                "DINOv3 preprocessing contract drifted: resolved model "
                f"input size {resolved_hw}, expected {expected_hw}. Update "
                "DINOV3_INPUT_SIZE and the cooperative-receiver probe before "
                "using this anchor."
            )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)

    _DINOV3_MODEL_CACHE[cache_key] = model
    return model


def _normalize_for_dinov3(rgb_255_bchw: torch.Tensor) -> torch.Tensor:
    """Resize + normalize RGB-255 (B, 3, H, W) for DINOv3 inference.

    Per the canonical timm DINOv3 base/16 checkpoint:
    1. Bilinear-resize to 256x256.
    2. Divide by 255 to get [0, 1].
    3. Subtract ImageNet mean / divide by ImageNet std.
    """
    if rgb_255_bchw.ndim != 4 or rgb_255_bchw.shape[1] != 3:
        raise ValueError(
            "DINOv3 input must be (B, 3, H, W) RGB-255, got "
            f"{tuple(rgb_255_bchw.shape)}"
        )
    # Resize to the canonical timm DINOv3 base/16 input size.
    resized = F.interpolate(
        rgb_255_bchw,
        size=(DINOV3_INPUT_SIZE, DINOV3_INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    # Scale [0, 255] -> [0, 1]
    scaled = resized / 255.0
    # ImageNet normalization
    mean = torch.tensor(
        DINOV3_IMAGENET_MEAN,
        dtype=scaled.dtype,
        device=scaled.device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        DINOV3_IMAGENET_STD,
        dtype=scaled.dtype,
        device=scaled.device,
    ).view(1, 3, 1, 1)
    return (scaled - mean) / std


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _is_square(n: int) -> bool:
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def _infer_vit_prefix_token_count(tokens: torch.Tensor, model: nn.Module) -> int:
    """Infer how many leading ViT tokens are non-patch tokens.

    DINOv3 ViT checkpoints expose CLS + register tokens before the patch grid.
    timm usually publishes this as ``num_prefix_tokens``; the square-grid
    fallback keeps fake/test backbones and older timm variants usable.
    """
    explicit = getattr(model, "num_prefix_tokens", None)
    if explicit is not None:
        prefix_count = int(explicit)
        if prefix_count < 1 or prefix_count >= tokens.shape[1]:
            raise RuntimeError(
                "DINOv3 model num_prefix_tokens is invalid for forward_features "
                f"shape {tuple(tokens.shape)}: {prefix_count}"
            )
        return prefix_count

    token_count_after_cls = tokens.shape[1] - 1
    if _is_square(token_count_after_cls):
        return 1
    token_count_after_dinov3_prefix = token_count_after_cls - DINOV3_REGISTER_TOKENS
    if _is_square(token_count_after_dinov3_prefix):
        return 1 + DINOV3_REGISTER_TOKENS
    return 1


def dinov3_frame_features(
    rgb_255_bchw: torch.Tensor,
    frame_idx: int,
    model: Optional[nn.Module] = None,
    model_name: str = CANONICAL_DINOV3_MODEL_NAME,
) -> DinoV3Features:
    """Extract frozen DINOv3 CLS + patch tokens for a single frame.

    Parameters
    ----------
    rgb_255_bchw : (B, 3, H, W) RGB-255 tensor.
    frame_idx : 0 or 1 (which frame of the pair this is).
    model : optional pre-loaded DINOv3 model. If None, the canonical
        singleton is loaded on the input's device.
    model_name : DINOv3 identifier for the singleton loader. Ignored if
        ``model`` is explicitly passed.

    Returns
    -------
    DinoV3Features
    """
    if model is None:
        model = load_frozen_dinov3_model(
            model_name=model_name, device=rgb_255_bchw.device
        )

    normalized = _normalize_for_dinov3(rgb_255_bchw)

    # timm ViT models with num_classes=0 expose forward_features +
    # forward_head; the head is identity but we want the raw tokens not
    # the pooled vector. We use forward_features which returns
    # (B, prefix+N, hidden) including CLS token at position 0 and, for
    # DINOv3, register tokens before the patch grid.
    forward_features = getattr(model, "forward_features", None)
    if forward_features is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "DINOv3 model must expose `forward_features`; timm ViT "
            "backbones do by default. Check model identifier and version."
        )
    tokens = forward_features(normalized)
    if tokens.ndim != 3:  # pragma: no cover - defensive
        raise RuntimeError(
            "DINOv3 forward_features must return (B, 1+N, hidden), got "
            f"shape {tuple(tokens.shape)}"
        )

    prefix_token_count = _infer_vit_prefix_token_count(tokens, model)
    cls_token = tokens[:, 0, :]  # (B, hidden)
    patch_tokens = tokens[:, prefix_token_count:, :]  # (B, N, hidden)

    return DinoV3Features(
        cls_token=cls_token,
        patch_tokens=patch_tokens,
        frame_idx=frame_idx,
        model_name=model_name,
    )


def dinov3_pair_features(
    pair_btchw: torch.Tensor,
    model: Optional[nn.Module] = None,
    model_name: str = CANONICAL_DINOV3_MODEL_NAME,
) -> DinoV3PairFeatures:
    """Extract frozen DINOv3 features for both frames of a (B, T=2, C, H, W) pair.

    Parameters
    ----------
    pair_btchw : (B, 2, 3, H, W) RGB-255 tensor.

    Returns
    -------
    DinoV3PairFeatures with frame_0 + frame_1 DinoV3Features.
    """
    if pair_btchw.ndim != 5 or pair_btchw.shape[1] != 2 or pair_btchw.shape[2] != 3:
        raise ValueError(
            "dinov3_pair_features input must be (B, T=2, C=3, H, W) RGB-255, "
            f"got {tuple(pair_btchw.shape)}"
        )

    # Reuse model singleton across the pair (saves load cost)
    if model is None:
        model = load_frozen_dinov3_model(
            model_name=model_name, device=pair_btchw.device
        )

    frame_0_features = dinov3_frame_features(
        pair_btchw[:, 0, ...],
        frame_idx=0,
        model=model,
        model_name=model_name,
    )
    frame_1_features = dinov3_frame_features(
        pair_btchw[:, 1, ...],
        frame_idx=1,
        model=model,
        model_name=model_name,
    )
    return DinoV3PairFeatures(frame_0=frame_0_features, frame_1=frame_1_features)


# ---------------------------------------------------------------------------
# Cooperative-receiver KL distillation loss
# ---------------------------------------------------------------------------


def cooperative_receiver_dinov3_kl_loss(
    predicted_pair_btchw: torch.Tensor,
    ground_truth_pair_btchw: torch.Tensor,
    *,
    temperature: float = HINTON_DISTILL_T_CANONICAL,
    use_cls_token: bool = True,
    use_patch_tokens: bool = True,
    patch_weight: float = 0.5,
    model: Optional[nn.Module] = None,
    model_name: str = CANONICAL_DINOV3_MODEL_NAME,
) -> torch.Tensor:
    """KL divergence between predicted-pair and GT-pair DINOv3 features.

    Implements the Hinton-style distillation cooperative-receiver loss
    against the DINOv3 anchor:

        L_D = T^2 * KL( softmax(D(GT)/T) || softmax(D(pred)/T) )

    where D is the frozen DINOv3 forward, T is the softening temperature
    (canonical T=2.0 per Hinton-Vinyals-Dean 2014), and KL is computed
    separately over the CLS token and the patch token grid (with optional
    weight ``patch_weight`` on the patch term).

    The T^2 scaling preserves gradient magnitudes per the canonical
    distillation formulation (otherwise dividing logits by T shrinks
    gradients by T^2).

    Parameters
    ----------
    predicted_pair_btchw : (B, 2, 3, H, W) substrate-predicted RGB-255 pair.
        Must be gradient-reachable (the substrate's training graph).
    ground_truth_pair_btchw : (B, 2, 3, H, W) ground-truth RGB-255 pair.
        Detached / no-grad (the canonical teacher signal).
    temperature : Hinton T. Default T=2.0 matches Quantizr SegNet
        distillation per CLAUDE.md "Quantizr intelligence".
    use_cls_token : Include CLS-token KL term.
    use_patch_tokens : Include patch-token KL term.
    patch_weight : Weight on the patch term relative to CLS (default 0.5;
        patch grid has 256 tokens but CLS captures global semantics so we
        weight CLS more heavily).
    model : Optional pre-loaded DINOv3 singleton.
    model_name : DINOv3 identifier for singleton loader.

    Returns
    -------
    torch.Tensor : scalar KL loss (mean over batch + frame pair).

    Raises
    ------
    ValueError : if neither CLS nor patch tokens are requested, or if
        predicted and GT pairs have mismatched shapes.
    """
    if not use_cls_token and not use_patch_tokens:
        raise ValueError(
            "cooperative_receiver_dinov3_kl_loss requires at least one of "
            "use_cls_token / use_patch_tokens"
        )
    if predicted_pair_btchw.shape != ground_truth_pair_btchw.shape:
        raise ValueError(
            "predicted_pair / ground_truth_pair shape mismatch: "
            f"{tuple(predicted_pair_btchw.shape)} vs "
            f"{tuple(ground_truth_pair_btchw.shape)}"
        )

    # GT features are frozen (no grad); predicted features are
    # gradient-reachable into the substrate's training graph.
    with torch.no_grad():
        gt_features = dinov3_pair_features(
            ground_truth_pair_btchw, model=model, model_name=model_name
        )
    pred_features = dinov3_pair_features(
        predicted_pair_btchw, model=model, model_name=model_name
    )

    total_loss = predicted_pair_btchw.new_zeros(())
    n_terms = 0

    for pred_fr, gt_fr in (
        (pred_features.frame_0, gt_features.frame_0),
        (pred_features.frame_1, gt_features.frame_1),
    ):
        if use_cls_token:
            # KL on CLS token logits softened by T
            cls_loss = (
                F.kl_div(
                    F.log_softmax(pred_fr.cls_token / temperature, dim=-1),
                    F.softmax(gt_fr.cls_token / temperature, dim=-1),
                    reduction="batchmean",
                )
                * (temperature ** 2)
            )
            total_loss = total_loss + cls_loss
            n_terms += 1
        if use_patch_tokens:
            # Patch tokens: KL averaged over patches then over batch
            pred_patch = pred_fr.patch_tokens.reshape(
                -1, pred_fr.patch_tokens.shape[-1]
            )
            gt_patch = gt_fr.patch_tokens.reshape(-1, gt_fr.patch_tokens.shape[-1])
            pred_log = F.log_softmax(pred_patch / temperature, dim=-1)
            gt_p = F.softmax(gt_patch / temperature, dim=-1)
            patch_loss = (
                F.kl_div(pred_log, gt_p, reduction="batchmean")
                * (temperature ** 2)
                * patch_weight
            )
            total_loss = total_loss + patch_loss
            n_terms += 1

    if n_terms == 0:  # pragma: no cover - defensive (caught above)
        raise ValueError("cooperative_receiver_dinov3_kl_loss: no terms enabled")
    return total_loss / n_terms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CANONICAL_DINOV3_MODEL_NAME",
    "DINOV3_INPUT_SIZE",
    "DINOV3_REGISTER_TOKENS",
    "DINOV3_IMAGENET_MEAN",
    "DINOV3_IMAGENET_STD",
    "HINTON_DISTILL_T_CANONICAL",
    "DinoV3Features",
    "DinoV3PairFeatures",
    "load_frozen_dinov3_model",
    "dinov3_frame_features",
    "dinov3_pair_features",
    "cooperative_receiver_dinov3_kl_loss",
]
