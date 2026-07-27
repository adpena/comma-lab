# SPDX-License-Identifier: MIT
"""HOPE BN-capacity generator for the frozen SegNet (task #725, arm hb1).

Applies the HOPE (arXiv 2607.21366) per-neuron capacity functional

    ||f_i||_H = ||w_out,i||_2 * sqrt(K(i,i)),   K(i,i) = E_{x~P_X}[Psi(y_i)^2]

to OUR frozen SegNet (smp.Unet 'tu-efficientnet_b2', classes=5), with one
decisive substitution over the paper: **P_X is not their data-free max-entropy
Gaussian surrogate — it is the exact n600 input measure in custody** (the 600
cached last-frame inputs, bit-checked against the cached GT argmax). Every
kernel here is an exact empirical second moment through the real frozen
forward pass; the paper's closed-form ReLU kernel (their Eq. 79) is computed
ONLY as a comparison column for the 10 decoder ReLU units.

ReLU-family check (charter caveat, MEASURED at import of the frozen net):
the SegNet encoder activations are SiLU (timm ``BatchNormAct2d`` with SiLU)
plus sigmoid SE gates — NOT positively homogeneous, so HOPE's closed-form
kernels (their Eqs. 3/5/79/85) do NOT transfer to the encoder. No kernel
extension is needed because the exact empirical kernels used here never
invoke a closed form. The decoder's 10 units are plain BN+ReLU, where the
closed form applies and is reported as ``surrogate_K`` next to the exact K.

Outputs (all advisory, ``[macOS-CPU frozen-scorer advisory]``; no score
claims; rate columns are intentionally ABSENT — per the crosswalk caveat,
rate denominators must be measured coder bytes, never parameter counts, and
no coder bytes are measured by this module):

1. per-unit / per-channel capacity table over the whole SegNet
   (exact sqrt(K) everywhere; ||w_out|| only where the consumer graph is
   structurally unambiguous — unresolved consumers are labelled, never
   guessed);
2. per-channel x per-stratum kernel table at the 16-channel pre-head layer
   (strata = the 37 occupied pf2 buckets), composed with the exact rank-4
   head factorisation (canonical equation ``segnet_head_rank4_linear_
   flipdist_v1``) into per-class-pair capacities: the analytic weighting
   for the FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK coordinate family;
3. fine-band selections for the RG3 Fisher-margin codebook rows —
   parity mode (must reproduce the 17 hand-derived rows of
   ``ddm_rg3_residual_family_assignment.json``) and capacity-refined mode
   (site-local proposals for the 9 blocked rows; advisory only).

Nothing in this module ships in any archive; margins, kernels and scorer
state stay offline exactly as in the RG3 assignment builder.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.direct_description_minimizer import DirectDescriptionError

REPO: Final = Path(__file__).resolve().parents[3]

SCHEMA_TABLE: Final = "ddm_hb1_hope_bn_capacity_table.v1"
SCHEMA_STRATUM: Final = "ddm_hb1_hope_per_stratum_capacity_table.v1"
SCHEMA_AGREEMENT: Final = "ddm_hb1_hope_rg3_agreement_receipt.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"

# --- custody pins (all recalled from sealed receipts, never re-derived) ----
SEGNET_SAFETENSORS_SHA256: Final = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
MARGIN_F16_SHA256: Final = "177d22f0ef16e31f9de0229606f72e69d22dd550b7ff55342f82d01ebe6f228d"
PF2_EVENT_INDEX_SHA256: Final = "dd164a75d4c09dc9bad6bdad549477533d13ba56a8f5f777c91fc4bbddf3d1d1"
V19C_BASE_SHA256: Final = "dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9"
RG3_ASSIGNMENT_SHA256: Final = "40d4150eb3c1bee1197b4023a1e2986e498f429d85677e992a8464ac7acab82e"

# Canonical comma10k class order (CLAUDE.md, MEASURED 2026-06-27; never
# luma-sorted, never re-derived here — indices are consumed from the typed
# keys of the sealed RG3 assignment, and this table is used only to render
# human-readable names in receipts).
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

MARGIN_SHAPE: Final = (600, 384, 512)
ROW_BAND_HEIGHT: Final = 64
FINE_BAND_HEIGHT: Final = 16
FINE_BAND_COUNT: Final = 4
MARGIN_CLIP: Final = 40.0
FISHER_FAMILY: Final = "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK"

HEAD_IN_CHANNELS: Final = 16
N_CLASSES: Final = 5


# ---------------------------------------------------------------------------
# Fisher trace + fine-band selection (independent implementation; parity with
# ddm_rg1_receiver_grammar.derive_rg3_fisher_margin_band is asserted in tests
# and in the agreement receipt, not assumed).
# ---------------------------------------------------------------------------


def fisher_trace_map(margin: np.ndarray) -> np.ndarray:
    """Categorical Fisher trace surrogate 0.5*sech^2(m/2) of a margin map."""

    m = np.asarray(margin, dtype=np.float32)
    if m.ndim != 2 or not np.isfinite(m).all() or (m < 0).any():
        raise DirectDescriptionError("Fisher margin map must be finite nonnegative 2-D")
    clipped = np.minimum(m, np.float32(MARGIN_CLIP))
    return np.float32(0.5) / np.cosh(clipped * np.float32(0.5)) ** np.float32(2.0)


def select_fine_band(mass: np.ndarray, *, row_band: int) -> int:
    """Argmax 16-row subband (earliest wins ties) inside one 64-row band."""

    arr = np.asarray(mass, dtype=np.float64)
    if arr.shape[0] != 384:
        raise DirectDescriptionError("fine-band mass must live on the 384-row scorer grid")
    if not 0 <= int(row_band) < 384 // ROW_BAND_HEIGHT:
        raise DirectDescriptionError("row band escapes the scorer grid")
    start = int(row_band) * ROW_BAND_HEIGHT
    sums = [
        float(arr[start + k * FINE_BAND_HEIGHT : start + (k + 1) * FINE_BAND_HEIGHT].sum())
        for k in range(FINE_BAND_COUNT)
    ]
    if not any(s > 0.0 for s in sums):
        raise DirectDescriptionError("fine-band address has no support mass")
    return int(np.argmax(np.asarray(sums, dtype=np.float64)))


def fisher_fine_band(
    margin: np.ndarray,
    support: np.ndarray,
    *,
    row_band: int,
    site_weight: np.ndarray | None = None,
) -> int:
    """Fisher-trace fine-band selection, optionally site-locally reweighted.

    ``site_weight=None`` is exact parity with the sealed RG3 hand derivation.
    A nonnegative ``site_weight`` field multiplies the Fisher trace pointwise
    (the HOPE capacity-refined mode). The weight field is normalised to unit
    mean over the support so that a flat field reduces to parity exactly.
    """

    trace = fisher_trace_map(margin)
    sup = np.asarray(support, dtype=bool)
    if sup.shape != trace.shape:
        raise DirectDescriptionError("support and margin shapes disagree")
    mass = np.where(sup, trace, np.float32(0.0)).astype(np.float64)
    if site_weight is not None:
        w = np.asarray(site_weight, dtype=np.float64)
        if w.shape != trace.shape or not np.isfinite(w).all() or (w < 0).any():
            raise DirectDescriptionError("site weight must be finite nonnegative and margin-shaped")
        denom = float(w[sup].mean()) if sup.any() else 0.0
        if denom <= 0.0:
            raise DirectDescriptionError("site weight has zero mass on the support")
        mass = mass * (w / denom)
    return select_fine_band(mass, row_band=row_band)


# ---------------------------------------------------------------------------
# HOPE closed-form ReLU kernel (their Eq. 79) — surrogate comparison ONLY.
# ---------------------------------------------------------------------------


def relu_gaussian_self_kernel(gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """HOPE Eq. 79: K(i,i) for ReLU under the BN surrogate y ~ N(beta, gamma^2).

    This is the paper's data-free Gaussian-surrogate kernel. It is emitted
    ONLY as a comparison column against the exact empirical kernel for the
    decoder's BN+ReLU units; it is never used as an authority anywhere in
    this module (their surrogate is exactly what the exact n600 measure
    replaces).
    """

    g = np.abs(np.asarray(gamma, dtype=np.float64))
    b = np.asarray(beta, dtype=np.float64)
    if g.shape != b.shape:
        raise DirectDescriptionError("gamma/beta shapes disagree")
    # Guard the degenerate gamma -> 0 limit: K -> beta^2 * 1[beta > 0].
    tiny = g < 1e-12
    z = np.where(tiny, 0.0, b / np.where(tiny, 1.0, g))
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    from math import sqrt

    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / sqrt(2.0)))
    k = (g * g + b * b) * cdf + b * g * phi
    k_tiny = np.where(b > 0.0, b * b, 0.0)
    return np.where(tiny, k_tiny, k)


# ---------------------------------------------------------------------------
# Unit inventory over the frozen SegNet.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HopeUnit:
    """One (conv -> BN -> activation) HOPE unit of the frozen SegNet."""

    unit_id: str
    module_path: str  # module whose OUTPUT is Psi(y) (BatchNormAct2d / ReLU)
    bn_path: str
    n_channels: int
    activation: str  # 'silu' | 'relu' | 'identity'
    stage: str  # 'encoder' | 'decoder'
    consumer_status: str
    consumer_paths: tuple[str, ...] = ()
    consumer_note: str = ""


_RESOLVED: Final = "RESOLVED"
_RESOLVED_SE: Final = "RESOLVED_WITH_SE_GATE_UPPER_BOUND"
_UNRESOLVED: Final = "UNRESOLVED_CONSUMER_GRAPH_V1"


def _activation_name(module: Any) -> str:
    name = type(module).__name__.lower()
    if "silu" in name or "swish" in name:
        return "silu"
    if "relu" in name:
        return "relu"
    if "identity" in name:
        return "identity"
    return name


def enumerate_segnet_units(segnet: Any) -> tuple[HopeUnit, ...]:
    """Walk the frozen SegNet and enumerate every BN-normalised unit.

    Consumer resolution is deliberately conservative: a ``||w_out||`` is
    attached only where the consuming convolution(s) are structurally
    unambiguous. Block outputs that fan out through residual adds and
    U-Net skip concatenations are labelled ``UNRESOLVED_CONSUMER_GRAPH_V1``
    and contribute sqrt(K) only (per NO-FAKE: no guessed capacity).
    """

    import torch.nn as nn

    units: list[HopeUnit] = []
    modules = dict(segnet.named_modules())

    # --- encoder: timm efficientnet_b2 (BatchNormAct2d units) -------------
    for path, module in modules.items():
        if type(module).__name__ != "BatchNormAct2d":
            continue
        act = _activation_name(module.act)
        n_ch = int(module.num_features)
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        parent = modules.get(parent_path)
        leaf = path.rsplit(".", 1)[-1]
        consumer_status = _UNRESOLVED
        consumer_paths: tuple[str, ...] = ()
        note = ""
        ptype = type(parent).__name__ if parent is not None else ""
        if leaf == "bn1" and ptype == "InvertedResidual":
            consumer_paths = (f"{parent_path}.conv_dw",)
            consumer_status = _RESOLVED
        elif leaf == "bn2" and ptype == "InvertedResidual":
            consumer_paths = (f"{parent_path}.conv_pwl",)
            consumer_status = _RESOLVED_SE
            note = "SE sigmoid gate between unit output and conv_pwl; gate in (0,1) => capacity is an upper bound"
        elif leaf == "bn1" and ptype == "DepthwiseSeparableConv":
            consumer_paths = (f"{parent_path}.conv_pw",)
            consumer_status = _RESOLVED_SE
            note = "SE sigmoid gate between unit output and conv_pw; gate in (0,1) => capacity is an upper bound"
        else:
            note = "block-output/stem unit fans out via residual add / next block / U-Net skip concat; not priced in v1"
        units.append(
            HopeUnit(
                unit_id=f"enc.{path}",
                module_path=path,
                bn_path=path,
                n_channels=n_ch,
                activation=act,
                stage="encoder",
                consumer_status=consumer_status,
                consumer_paths=consumer_paths,
                consumer_note=note,
            )
        )

    # --- decoder: smp UnetDecoderBlock conv1/conv2 (Conv2d,BN,ReLU) -------
    n_blocks = len(segnet.decoder.blocks)
    out_ch = [int(b.conv2[0].out_channels) for b in segnet.decoder.blocks]
    for i, _block in enumerate(segnet.decoder.blocks):
        for conv_name in ("conv1", "conv2"):
            bn_path = f"decoder.blocks.{i}.{conv_name}.1"
            act_path = f"decoder.blocks.{i}.{conv_name}.2"
            bn = modules[bn_path]
            if not isinstance(bn, nn.BatchNorm2d):
                raise DirectDescriptionError(f"decoder unit {bn_path} is not BatchNorm2d")
            n_ch = int(bn.num_features)
            if conv_name == "conv1":
                consumer_paths = (f"decoder.blocks.{i}.conv2.0",)
                note = ""
            elif i + 1 < n_blocks:
                # concat order in smp UnetDecoderBlock.forward is
                # [upsampled_prev, skip] => prev channels occupy [0:out_ch].
                consumer_paths = (f"decoder.blocks.{i + 1}.conv1.0[:, :{out_ch[i]}]",)
                note = "consumer slice = first channels of next block conv1 (concat order verified)"
            else:
                consumer_paths = ("segmentation_head.0",)
                note = "final decoder features; consumer = rank-4 segmentation head"
            units.append(
                HopeUnit(
                    unit_id=f"dec.blocks.{i}.{conv_name}",
                    module_path=act_path,
                    bn_path=bn_path,
                    n_channels=n_ch,
                    activation="relu",
                    stage="decoder",
                    consumer_status=_RESOLVED,
                    consumer_paths=consumer_paths,
                    consumer_note=note,
                )
            )
    return tuple(units)


def relu_family_check(units: Sequence[HopeUnit]) -> dict[str, Any]:
    """The charter-mandated scorer activation-family census (MEASURED)."""

    census: dict[str, int] = {}
    for u in units:
        key = f"{u.stage}:{u.activation}"
        census[key] = census.get(key, 0) + 1
    pure_relu = all(u.activation in ("relu", "identity") for u in units)
    return {
        "census": dict(sorted(census.items())),
        "scorer_is_pure_relu_family": bool(pure_relu),
        "closed_form_kernel_scope": (
            "decoder BN+ReLU units only; encoder SiLU (+ sigmoid SE gates) is not "
            "positively homogeneous, so HOPE Eqs. 3/5/79/85 do not transfer there. "
            "No kernel extension is required: all authority kernels in this table "
            "are exact empirical second moments under the n600 measure."
        ),
    }


def consumer_weight_norms(segnet: Any, unit: HopeUnit) -> np.ndarray | None:
    """Per-channel ||w_out,i|| for a unit with a resolved consumer set.

    Frobenius norm over the consumer conv weight slice that reads channel i
    (HOPE App. B.1 conv adaptation); multiple consumers add in quadrature
    (HS norm additivity over the stacked consumer map). Returns ``None``
    for unresolved units — never a guessed number.
    """

    if unit.consumer_status not in (_RESOLVED, _RESOLVED_SE):
        return None
    modules = dict(segnet.named_modules())
    total_sq = np.zeros(unit.n_channels, dtype=np.float64)
    for spec in unit.consumer_paths:
        if "[" in spec:
            path, slice_part = spec.split("[", 1)
            limit = int(slice_part.rstrip("]").split(":")[-1])
        else:
            path, limit = spec, None
        conv = modules[path.strip()]
        w = conv.weight.detach().cpu().numpy().astype(np.float64)
        groups = int(getattr(conv, "groups", 1))
        if groups == w.shape[0] and w.shape[1] == 1:
            # depthwise: channel i is consumed by its own kernel w[i, 0]
            if w.shape[0] != unit.n_channels:
                raise DirectDescriptionError(f"depthwise consumer {path} channel mismatch for {unit.unit_id}")
            total_sq += (w[:, 0] ** 2).sum(axis=(1, 2))
        else:
            if limit is not None and limit != unit.n_channels:
                raise DirectDescriptionError(f"consumer slice width disagrees with unit width for {unit.unit_id}")
            if w.shape[1] < unit.n_channels:
                raise DirectDescriptionError(f"consumer {path} reads fewer channels than {unit.unit_id} emits")
            sl = w[:, : unit.n_channels]
            total_sq += (sl**2).sum(axis=(0, 2, 3))
    return np.sqrt(total_sq)


# ---------------------------------------------------------------------------
# Rank-4 head composition (canonical equation segnet_head_rank4_linear_
# flipdist_v1: the 5-logit head is an exact linear map of the 16 pre-head
# channels; class-pair flips live in the rank-4 difference space).
# ---------------------------------------------------------------------------


def head_class_pair_delta_norms(head_weight: np.ndarray) -> dict[str, np.ndarray]:
    """Per-channel Frobenius norms of Delta-w for every unordered class pair."""

    w = np.asarray(head_weight, dtype=np.float64)
    if w.shape[:2] != (N_CLASSES, HEAD_IN_CHANNELS):
        raise DirectDescriptionError("head weight must be (5, 16, kh, kw)")
    out: dict[str, np.ndarray] = {}
    for a in range(N_CLASSES):
        for b in range(a + 1, N_CLASSES):
            delta = w[a] - w[b]  # (16, kh, kw)
            out[f"{a}-{b}"] = np.sqrt((delta**2).sum(axis=(1, 2)))
    return out


def head_difference_rank(head_weight: np.ndarray, *, rtol: float = 1e-6) -> dict[str, Any]:
    """Numerical rank of the head logit-difference space (expected: 4)."""

    w = np.asarray(head_weight, dtype=np.float64).reshape(N_CLASSES, -1)
    diffs = w[1:] - w[0]
    s = np.linalg.svd(diffs, compute_uv=False)
    rank = int((s > s[0] * rtol).sum()) if s.size else 0
    return {"singular_values": [float(v) for v in s], "rank": rank, "rtol": rtol}


# ---------------------------------------------------------------------------
# Exact empirical kernel measurement over the n600 measure.
# ---------------------------------------------------------------------------


@dataclass
class KernelAccumulator:
    """Streaming fp64 second-moment accumulator for one unit."""

    n_channels: int
    sum_sq: np.ndarray = field(init=False)
    sum_val: np.ndarray = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        self.sum_sq = np.zeros(self.n_channels, dtype=np.float64)
        self.sum_val = np.zeros(self.n_channels, dtype=np.float64)

    def update(self, output: Any) -> None:
        import torch

        with torch.no_grad():
            o = output if not isinstance(output, tuple) else output[0]
            self.sum_sq += o.pow(2).sum(dim=(0, 2, 3), dtype=torch.float64).cpu().numpy()
            self.sum_val += o.sum(dim=(0, 2, 3), dtype=torch.float64).cpu().numpy()
            self.count += int(o.shape[0] * o.shape[2] * o.shape[3])

    def k_diag(self) -> np.ndarray:
        if self.count == 0:
            raise DirectDescriptionError("kernel accumulator saw no samples")
        return self.sum_sq / float(self.count)

    def mean(self) -> np.ndarray:
        if self.count == 0:
            raise DirectDescriptionError("kernel accumulator saw no samples")
        return self.sum_val / float(self.count)


def load_frozen_segnet(segnet_path: Path, *, expected_sha256: str = SEGNET_SAFETENSORS_SHA256) -> Any:
    """Load the frozen SegNet exactly as the contest scorer does (CPU fp32)."""

    from safetensors.torch import load_file

    import segmentation_models_pytorch as smp

    digest = hashlib.sha256(segnet_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise DirectDescriptionError(f"segnet.safetensors sha mismatch: {digest}")
    seg = smp.Unet("tu-efficientnet_b2", classes=5, activation=None, encoder_weights=None)
    seg.load_state_dict(load_file(str(segnet_path), device="cpu"))
    seg.eval()
    for p in seg.parameters():
        p.requires_grad_(False)
    return seg


def preprocess_last_frame(frames_u8: np.ndarray) -> Any:
    """Bit-parity replica of upstream SegNet.preprocess_input on gt_f1 frames."""

    import torch

    x = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float()
    return torch.nn.functional.interpolate(x, size=(384, 512), mode="bilinear")


def load_bucket_index(
    pf2_index_path: Path,
    bucket_name_map: Mapping[str, str],
    *,
    expected_sha256: str = PF2_EVENT_INDEX_SHA256,
) -> dict[str, dict[int, np.ndarray]]:
    """Load the 37 occupied pf2 bucket supports, grouped per pair.

    The sealed index stores flat uint32 indices over the (600, 384, 512)
    scorer grid; returns bucket_id -> {pair_id -> flat pixel indices in the
    (384*512) plane of that pair}. Custody is pinned to the sealed index
    SHA by default; tests must pass the SHA of their synthetic fixture.
    """

    digest = hashlib.sha256(pf2_index_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise DirectDescriptionError(f"pf2 event index sha mismatch: {digest}")
    plane = MARGIN_SHAPE[1] * MARGIN_SHAPE[2]
    limit = MARGIN_SHAPE[0] * plane
    z = np.load(pf2_index_path)
    out: dict[str, dict[int, np.ndarray]] = {}
    for bucket_id, array_key in bucket_name_map.items():
        flat = np.asarray(z[array_key], dtype=np.int64)
        if flat.size and (flat.min() < 0 or flat.max() >= limit):
            raise DirectDescriptionError(f"bucket {bucket_id} indices escape the n600 scorer grid")
        pairs = flat // plane
        local = flat % plane
        per_pair: dict[int, np.ndarray] = {}
        order = np.argsort(pairs, kind="stable")
        pairs_sorted = pairs[order]
        local_sorted = local[order]
        boundaries = np.flatnonzero(np.diff(pairs_sorted)) + 1
        for chunk_pairs, chunk_local in zip(
            np.split(pairs_sorted, boundaries), np.split(local_sorted, boundaries), strict=True
        ):
            if chunk_pairs.size:
                per_pair[int(chunk_pairs[0])] = chunk_local.astype(np.int64)
        out[bucket_id] = per_pair
    return out


@dataclass
class MeasurementResult:
    units: tuple[HopeUnit, ...]
    k_diag: dict[str, np.ndarray]
    k_mean: dict[str, np.ndarray]
    pixel_count: int
    head_bucket_sum_sq: dict[str, np.ndarray]
    head_bucket_count: dict[str, int]
    argmax_agreement: float
    target_pair_features: dict[int, np.ndarray]


def measure_exact_kernels(
    segnet: Any,
    frames_u8: np.ndarray,
    lstars: np.ndarray,
    bucket_index: Mapping[str, Mapping[int, np.ndarray]],
    *,
    target_pairs: Sequence[int] = (),
    batch_size: int = 4,
    progress: Callable[[str], None] | None = None,
) -> MeasurementResult:
    """One streaming n600 pass computing every exact kernel diagonal.

    Also verifies, pair by pair, that our forward argmax matches the cached
    frozen-scorer GT argmax (``lstars``) — the custody proof that the measure
    integrated over IS the exact n600 measure.
    """

    import torch

    units = enumerate_segnet_units(segnet)
    modules = dict(segnet.named_modules())
    accs = {u.unit_id: KernelAccumulator(u.n_channels) for u in units}
    head_unit = next(u for u in units if u.consumer_paths == ("segmentation_head.0",))

    plane = MARGIN_SHAPE[1] * MARGIN_SHAPE[2]
    bucket_sum_sq = {b: np.zeros(HEAD_IN_CHANNELS, dtype=np.float64) for b in bucket_index}
    bucket_count = {b: 0 for b in bucket_index}
    target_set = {int(p) for p in target_pairs}
    target_features: dict[int, np.ndarray] = {}
    captured: dict[str, Any] = {}

    hooks = []
    try:
        for u in units:
            mod = modules[u.module_path]

            def _hook(_m: Any, _inp: Any, out: Any, uid: str = u.unit_id) -> None:
                accs[uid].update(out)
                if uid == head_unit.unit_id:
                    captured["head"] = out.detach()

            hooks.append(mod.register_forward_hook(_hook))

        n = int(frames_u8.shape[0])
        agree_pixels = 0
        total_pixels = 0
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                x = preprocess_last_frame(frames_u8[start:stop])
                logits = segnet(x)
                pred = logits.argmax(dim=1).cpu().numpy()
                agree_pixels += int((pred == lstars[start:stop]).sum())
                total_pixels += int(pred.size)
                feats = captured.pop("head")  # (B, 16, 384, 512)
                flat = feats.double().reshape(feats.shape[0], HEAD_IN_CHANNELS, plane)
                for offset in range(stop - start):
                    pair = start + offset
                    for bucket_id, per_pair in bucket_index.items():
                        idx = per_pair.get(pair)
                        if idx is None or idx.size == 0:
                            continue
                        sel = flat[offset][:, torch.from_numpy(idx)]
                        bucket_sum_sq[bucket_id] += sel.pow(2).sum(dim=1).cpu().numpy()
                        bucket_count[bucket_id] += int(idx.size)
                    if pair in target_set:
                        target_features[pair] = feats[offset].float().cpu().numpy()
                if progress is not None:
                    progress(f"pairs {stop}/{n} argmax-agree {agree_pixels / max(total_pixels, 1):.6f}")
    finally:
        for h in hooks:
            h.remove()

    return MeasurementResult(
        units=units,
        k_diag={uid: acc.k_diag() for uid, acc in accs.items()},
        k_mean={uid: acc.mean() for uid, acc in accs.items()},
        pixel_count=next(iter(accs.values())).count if accs else 0,
        head_bucket_sum_sq=bucket_sum_sq,
        head_bucket_count=bucket_count,
        argmax_agreement=agree_pixels / max(total_pixels, 1),
        target_pair_features=target_features,
    )


# ---------------------------------------------------------------------------
# Table assembly.
# ---------------------------------------------------------------------------


def assemble_capacity_table(segnet: Any, result: MeasurementResult) -> dict[str, Any]:
    """Per-unit/per-channel HOPE capacity table (exact kernels, n600)."""

    import torch.nn as nn

    modules = dict(segnet.named_modules())
    unit_rows: list[dict[str, Any]] = []
    for u in result.units:
        k = result.k_diag[u.unit_id]
        sqrt_k = np.sqrt(k)
        row: dict[str, Any] = {
            "unit_id": u.unit_id,
            "module_path": u.module_path,
            "stage": u.stage,
            "activation": u.activation,
            "n_channels": u.n_channels,
            "consumer_status": u.consumer_status,
            "consumer_paths": list(u.consumer_paths),
            "consumer_note": u.consumer_note,
            "k_diag_exact": [float(v) for v in k],
            "sqrt_k_exact": [float(v) for v in sqrt_k],
            "dead_channels_k_lt_1e-12": int((k < 1e-12).sum()),
        }
        norms = consumer_weight_norms(segnet, u)
        if norms is not None:
            row["w_out_norm"] = [float(v) for v in norms]
            row["capacity"] = [float(v) for v in norms * sqrt_k]
        if u.stage == "decoder" and u.activation == "relu":
            bn = modules[u.bn_path]
            assert isinstance(bn, nn.BatchNorm2d)
            gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
            beta = bn.bias.detach().cpu().numpy().astype(np.float64)
            k_sur = relu_gaussian_self_kernel(gamma, beta)
            row["k_diag_bn_surrogate_hope_eq79"] = [float(v) for v in k_sur]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(k > 0, np.abs(k_sur - k) / k, np.inf)
            row["surrogate_rel_deviation_median"] = float(np.median(rel))
            row["surrogate_rel_deviation_max"] = float(np.max(rel))
        unit_rows.append(row)

    return {
        "schema": SCHEMA_TABLE,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "measure": "exact_n600_last_frame_inputs_not_gaussian_surrogate",
        "pixels_integrated_per_unit_note": "K(i,i)=E[Psi(y_i)^2]; expectation over all n600 frames x unit spatial grid",
        "argmax_agreement_with_cached_gt": result.argmax_agreement,
        "relu_family_check": relu_family_check(result.units),
        "rate_denominator_policy": (
            "NO rate column on purpose: per the HOPE crosswalk caveat, rate denominators must be "
            "measured coder bytes per action, never parameter counts. score_units_per_byte_status="
            "OWED_NOT_ADMITTED for every row."
        ),
        "units": unit_rows,
    }


def assemble_per_stratum_table(
    segnet: Any,
    result: MeasurementResult,
    *,
    bucket_class_pairs: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    """Per-channel x per-stratum capacity table at the pre-head layer.

    For each occupied pf2 bucket b and pre-head channel i:
      K_b(i)      = E[psi_i^2 | (pair, pixel) in b]   (exact, n600)
      cap_b^ab(i) = ||Delta-w_head^{ab}[i]||_F * sqrt(K_b(i))
    where (a, b) is the bucket's class pair through the exact rank-4 head.
    """

    head_w = segnet.segmentation_head[0].weight.detach().cpu().numpy()
    delta_norms = head_class_pair_delta_norms(head_w)
    head_unit_id = next(u.unit_id for u in result.units if u.consumer_paths == ("segmentation_head.0",))
    k_global = result.k_diag[head_unit_id]

    rows: list[dict[str, Any]] = []
    for bucket_id in sorted(result.head_bucket_sum_sq):
        count = result.head_bucket_count[bucket_id]
        if count == 0:
            continue
        k_b = result.head_bucket_sum_sq[bucket_id] / float(count)
        a, b = bucket_class_pairs[bucket_id]
        pair_key = f"{min(a, b)}-{max(a, b)}"
        dn = delta_norms[pair_key]
        cap = dn * np.sqrt(k_b)
        rows.append(
            {
                "bucket_id": bucket_id,
                "class_pair": pair_key,
                "class_pair_names": f"{CLASS_NAMES[min(a, b)]}--{CLASS_NAMES[max(a, b)]}",
                "support_pixel_count": count,
                "k_diag_bucket": [float(v) for v in k_b],
                "delta_w_head_norm": [float(v) for v in dn],
                "capacity_per_channel": [float(v) for v in cap],
                "capacity_share": [float(v) for v in (cap / cap.sum() if cap.sum() > 0 else cap)],
                "score_units_per_byte_status": "OWED_NOT_ADMITTED",
            }
        )
    return {
        "schema": SCHEMA_STRATUM,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "coordinate_family": "FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK",
        "head_rank_check": head_difference_rank(head_w),
        "k_diag_global_head_layer": [float(v) for v in k_global],
        "strata": rows,
    }


_CLASS_TOKEN_TO_ID: Final = {
    "road": 0,
    "lane": 1,
    "undrivable": 2,
    "movable": 3,
    "mycar": 4,
}


def bucket_class_pair(bucket_id: str) -> tuple[int, int]:
    """Parse the canonical class pair out of a pf2 bucket id.

    Bucket ids follow ``{tokA}_{tokB}__{stratum}__{temporal}`` with class
    tokens drawn from the canonical comma10k order (Road0, Lane1,
    Undrivable2, Movable3, MyCar4). The pair is parsed, never inferred
    from luma or hardcoded per-bucket.
    """

    head = bucket_id.split("__", 1)[0]
    tokens = head.split("_")
    if len(tokens) != 2 or any(t not in _CLASS_TOKEN_TO_ID for t in tokens):
        raise DirectDescriptionError(f"bucket id {bucket_id!r} does not parse to a canonical class pair")
    a, b = (_CLASS_TOKEN_TO_ID[tokens[0]], _CLASS_TOKEN_TO_ID[tokens[1]])
    if a == b:
        raise DirectDescriptionError(f"bucket id {bucket_id!r} names a degenerate class pair")
    return (min(a, b), max(a, b))


def site_local_capacity_field(
    features: np.ndarray,
    capacity_per_channel: np.ndarray,
) -> np.ndarray:
    """Site-local field W(x) = sum_i c_hat_i * psi_i(x)^2 (nonnegative, HxW)."""

    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 3 or f.shape[0] != HEAD_IN_CHANNELS:
        raise DirectDescriptionError("features must be (16, H, W) pre-head activations")
    c = np.asarray(capacity_per_channel, dtype=np.float64)
    if c.shape != (HEAD_IN_CHANNELS,) or (c < 0).any() or not np.isfinite(c).all():
        raise DirectDescriptionError("capacity vector must be nonnegative finite length-16")
    total = float(c.sum())
    if total <= 0.0:
        raise DirectDescriptionError("capacity vector has zero mass")
    return np.tensordot(c / total, f * f, axes=(0, 0))


__all__ = [
    "CLASS_NAMES",
    "FISHER_FAMILY",
    "HopeUnit",
    "KernelAccumulator",
    "MeasurementResult",
    "assemble_capacity_table",
    "assemble_per_stratum_table",
    "bucket_class_pair",
    "consumer_weight_norms",
    "enumerate_segnet_units",
    "fisher_fine_band",
    "fisher_trace_map",
    "head_class_pair_delta_norms",
    "head_difference_rank",
    "load_bucket_index",
    "load_frozen_segnet",
    "measure_exact_kernels",
    "preprocess_last_frame",
    "relu_family_check",
    "relu_gaussian_self_kernel",
    "select_fine_band",
    "site_local_capacity_field",
]
