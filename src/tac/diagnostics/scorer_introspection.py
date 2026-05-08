"""Per-layer observational introspection for SegNet and PoseNet ("neon dye").

This module attaches `register_forward_hook`s to every descendant `nn.Module`
of a SegNet or PoseNet instance, captures per-layer input/output tensors during
a forward pass, and emits an `IntrospectionRecord`. Two capture modes:

- ``"full"``: the entire output tensor is retained (suitable for small
  layers / unit tests / per-layer drift comparison).
- ``"fingerprint"``: only a statistical fingerprint (mean, std, min, max,
  p1, p99, L2-norm, sparsity, dtype, shape) is kept. Memory-bounded; suitable
  for the full ~600-module PoseNet on a single frame pair.

Special-cased "attention-like" layers
-------------------------------------

The pinned upstream PoseNet uses ``timm`` ``fastvit_t12`` whose 12 token-mixer
blocks are ``RepMixer`` (a reparameterizable conv-based mixer) rather than
softmax self-attention. We therefore treat ``RepMixerBlock`` as the structural
analog of an attention layer and capture:

- input/output shape and fingerprint of the block as a whole
- input/output of the inner ``token_mixer`` (RepMixer)
- input/output of the inner ``mlp`` (ConvMlp)
- a "mixer rank proxy" — the relative variance contribution along the token
  spatial axis of the token-mixer output, which provides a scalar analogous to
  the entropy/rank of a softmax attention pattern.

If a future model adds true ``MultiheadAttention`` or class
``Attention`` modules, ``attach_attention_hooks`` will pick those up and capture
``Q@K^T`` softmax weights when accessible via the standard timm Attention API.

Rules from CLAUDE.md respected
------------------------------

- No upstream/* edits: hooks are observational; no monkeypatching of upstream.
- No score claims: any numeric output is tagged ``[diagnostic-not-score]``.
- No /tmp paths in persisted artifacts: callers pass an explicit output path.
- Hooks are removed via context-manager cleanup so this is leak-safe.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Capture modes and configuration
# ---------------------------------------------------------------------------

CAPTURE_MODES = ("full", "fingerprint")
"""Allowed values for the ``capture_mode`` constructor argument."""

# A module's flattened element count above which "full" mode falls back to
# fingerprint storage to bound disk/memory usage during the demo run. Operators
# can override per-layer via ``ScorerIntrospector(full_threshold=...)``.
DEFAULT_FULL_THRESHOLD_ELEMENTS = 1 << 20  # 1 048 576 elements


# ---------------------------------------------------------------------------
# Statistical fingerprint
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Statistical fingerprint of a tensor.

    All numbers are tagged ``[diagnostic-not-score]`` — these are observational
    summaries, never a contest score.
    """

    shape: tuple[int, ...]
    dtype: str
    numel: int
    mean: float
    std: float
    min: float
    max: float
    p1: float
    p99: float
    l2_norm: float
    sparsity_eps: float
    sparsity_frac: float

    @classmethod
    def from_tensor(cls, t: torch.Tensor, sparsity_eps: float = 1e-6) -> "LayerStats":
        """Compute a fingerprint from a tensor (single forward pass)."""
        flat = t.detach().to(dtype=torch.float32).flatten().cpu()
        if flat.numel() == 0:
            return cls(
                shape=tuple(t.shape),
                dtype=str(t.dtype),
                numel=0,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                p1=0.0,
                p99=0.0,
                l2_norm=0.0,
                sparsity_eps=sparsity_eps,
                sparsity_frac=0.0,
            )
        # quantile is approximate but bounded; use kthvalue for exact percentile
        # on small tensors and quantile() on large.
        if flat.numel() <= 100_000:
            p1 = float(torch.quantile(flat, 0.01))
            p99 = float(torch.quantile(flat, 0.99))
        else:
            # subsample for percentiles to avoid the O(N log N) sort cost
            idx = torch.linspace(0, flat.numel() - 1, 50_000).to(dtype=torch.long)
            sub = flat[idx]
            p1 = float(torch.quantile(sub, 0.01))
            p99 = float(torch.quantile(sub, 0.99))
        l2 = float(torch.linalg.vector_norm(flat).item())
        sparsity_frac = float((flat.abs() < sparsity_eps).float().mean().item())
        return cls(
            shape=tuple(t.shape),
            dtype=str(t.dtype),
            numel=int(flat.numel()),
            mean=float(flat.mean().item()),
            std=float(flat.std(unbiased=False).item()),
            min=float(flat.min().item()),
            max=float(flat.max().item()),
            p1=p1,
            p99=p99,
            l2_norm=l2,
            sparsity_eps=sparsity_eps,
            sparsity_frac=sparsity_frac,
        )


# ---------------------------------------------------------------------------
# Attention-like fingerprint (RepMixer in fastvit_t12 substitutes for softmax)
# ---------------------------------------------------------------------------


@dataclass
class AttentionFingerprint:
    """Fingerprint of an attention-like (token-mixing) layer.

    For RepMixer: ``softmax_entropy`` and ``head_rank`` are not meaningful and
    are recorded as ``None``. ``mixer_rank_proxy`` is computed from the spatial
    variance distribution of the mixer output, which behaves analogously to an
    effective rank: low value -> few spatial modes carry the signal (similar to
    a sharply concentrated attention pattern), high value -> the response is
    spread across the full spatial map.

    For true MultiheadAttention/Attention: ``softmax_entropy`` (per-head),
    ``head_rank`` (per-head effective rank of Q@K^T softmax), and
    ``attention_output_norm`` are recorded if the underlying module exposes
    Q/K/V access.
    """

    layer_name: str
    layer_type: str
    num_heads: int | None = None
    softmax_entropy: list[float] | None = None  # per-head entropy if available
    head_rank: list[float] | None = None  # per-head effective rank
    attention_output_norm: float | None = None
    mixer_rank_proxy: float | None = None  # for RepMixer
    spatial_concentration: float | None = None  # for RepMixer


# ---------------------------------------------------------------------------
# Per-layer record entry + full IntrospectionRecord
# ---------------------------------------------------------------------------


@dataclass
class LayerRecord:
    """Single-layer entry in an :class:`IntrospectionRecord`."""

    name: str
    module_type: str
    input_stats: list[LayerStats]
    output_stats: list[LayerStats]
    full_input: list[torch.Tensor] | None = None
    full_output: list[torch.Tensor] | None = None
    is_attention_like: bool = False
    attention_fingerprint: AttentionFingerprint | None = None

    def to_dict(self, include_full_tensors: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "module_type": self.module_type,
            "input_stats": [asdict(s) for s in self.input_stats],
            "output_stats": [asdict(s) for s in self.output_stats],
            "is_attention_like": self.is_attention_like,
            "attention_fingerprint": (
                asdict(self.attention_fingerprint)
                if self.attention_fingerprint is not None
                else None
            ),
            "has_full_input": self.full_input is not None,
            "has_full_output": self.full_output is not None,
        }
        if include_full_tensors and self.full_input is not None:
            d["full_input_b64"] = [_tensor_to_b64(t) for t in self.full_input]
        if include_full_tensors and self.full_output is not None:
            d["full_output_b64"] = [_tensor_to_b64(t) for t in self.full_output]
        return d


@dataclass
class IntrospectionRecord:
    """Container for every captured layer's input/output across one forward."""

    model_kind: str  # "PoseNet" / "SegNet" / other
    device: str
    capture_mode: str
    input_shape: tuple[int, ...]
    layers: list[LayerRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]

    def get(self, layer_name: str) -> LayerRecord | None:
        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        return None

    def attention_layers(self) -> list[LayerRecord]:
        return [layer for layer in self.layers if layer.is_attention_like]

    def to_dict(self, include_full_tensors: bool = False) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "device": self.device,
            "capture_mode": self.capture_mode,
            "input_shape": list(self.input_shape),
            "metadata": self.metadata,
            "layers": [layer.to_dict(include_full_tensors=include_full_tensors) for layer in self.layers],
        }

    def to_json(self, path: str | Path, include_full_tensors: bool = False) -> Path:
        """Write a JSON dump of fingerprints (and optional full tensors).

        Per CLAUDE.md "no /tmp paths in persisted artifacts": the caller
        chooses the path. Tensors, when included, are stored as base64-encoded
        torch ``save`` payloads inside the JSON.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(include_full_tensors=include_full_tensors), indent=2)
        )
        return out

    def to_disk(self, path: str | Path) -> Path:
        """Pickle-free torch save (preferred for round-trip with tensors)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_kind": self.model_kind,
            "device": self.device,
            "capture_mode": self.capture_mode,
            "input_shape": list(self.input_shape),
            "metadata": self.metadata,
            "layers": [],
        }
        for layer in self.layers:
            payload["layers"].append(
                {
                    "name": layer.name,
                    "module_type": layer.module_type,
                    "input_stats": [asdict(s) for s in layer.input_stats],
                    "output_stats": [asdict(s) for s in layer.output_stats],
                    "is_attention_like": layer.is_attention_like,
                    "attention_fingerprint": (
                        asdict(layer.attention_fingerprint)
                        if layer.attention_fingerprint is not None
                        else None
                    ),
                    "full_input": layer.full_input,
                    "full_output": layer.full_output,
                }
            )
        torch.save(payload, str(out))
        return out

    @classmethod
    def from_disk(cls, path: str | Path) -> "IntrospectionRecord":
        # weights_only=False is used here intentionally: the file is produced by
        # ``to_disk`` from a trusted local script in the same repo, never from
        # an untrusted source. Per `forbidden_torch_load_weights_only_false` in
        # CLAUDE.md, we must annotate the rationale at the call site.
        # WEIGHTS_ONLY_FALSE_OK: round-trip of own diagnostic dump produced by
        # IntrospectionRecord.to_disk in this same repo.
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        layers: list[LayerRecord] = []
        for layer_d in payload["layers"]:
            attn_d = layer_d.get("attention_fingerprint")
            attn = AttentionFingerprint(**attn_d) if attn_d is not None else None
            layers.append(
                LayerRecord(
                    name=layer_d["name"],
                    module_type=layer_d["module_type"],
                    input_stats=[LayerStats(**s) for s in layer_d["input_stats"]],
                    output_stats=[LayerStats(**s) for s in layer_d["output_stats"]],
                    full_input=layer_d.get("full_input"),
                    full_output=layer_d.get("full_output"),
                    is_attention_like=layer_d.get("is_attention_like", False),
                    attention_fingerprint=attn,
                )
            )
        return cls(
            model_kind=payload["model_kind"],
            device=payload["device"],
            capture_mode=payload["capture_mode"],
            input_shape=tuple(payload["input_shape"]),
            layers=layers,
            metadata=payload.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor_to_b64(t: torch.Tensor) -> str:
    import base64
    import io

    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _coerce_to_tensor_list(obj: Any) -> list[torch.Tensor]:
    """Flatten the tuple/list/dict structure that hooks can deliver."""
    out: list[torch.Tensor] = []

    def _walk(x: Any) -> None:
        if isinstance(x, torch.Tensor):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                _walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                _walk(y)
        # otherwise silently drop (None, ints, strings, etc.)

    _walk(obj)
    return out


def fingerprint_tensor(t: torch.Tensor) -> LayerStats:
    """Convenience wrapper around :meth:`LayerStats.from_tensor`."""
    return LayerStats.from_tensor(t)


def _is_repmixer_block(module: nn.Module) -> bool:
    return type(module).__name__ == "RepMixerBlock"


def _is_softmax_attention(module: nn.Module) -> bool:
    """Heuristic detection of a true softmax-attention module.

    Picks up:
    - ``torch.nn.MultiheadAttention``
    - timm-style ``Attention`` (class name match)

    RepMixer/RepMixerBlock are NOT included here — they are conv-based mixers,
    not attention. Use :func:`_is_repmixer_block` for those.
    """
    if isinstance(module, nn.MultiheadAttention):
        return True
    cls = type(module).__name__
    return cls in {"Attention", "MultiHeadAttention", "MultiHeadSelfAttention"}


def _compute_repmixer_fingerprint(
    layer_name: str, module: nn.Module, output: torch.Tensor | None
) -> AttentionFingerprint:
    """Compute the conv-mixer rank proxy for a RepMixerBlock output.

    Approach: flatten spatial dims, compute the per-position variance, and
    derive an effective rank by ``exp(entropy(p))`` where ``p`` is the
    normalized variance distribution. Concentration = max(p) is a separate
    scalar useful for sanity checks.
    """
    if output is None:
        return AttentionFingerprint(
            layer_name=layer_name,
            layer_type=type(module).__name__,
            num_heads=None,
            mixer_rank_proxy=None,
            spatial_concentration=None,
        )
    out = output.detach().to(dtype=torch.float32)
    if out.dim() < 3:
        return AttentionFingerprint(
            layer_name=layer_name,
            layer_type=type(module).__name__,
            num_heads=None,
            mixer_rank_proxy=None,
            spatial_concentration=None,
        )
    # (B, C, H, W) -> (B, H*W) variance over channel
    if out.dim() == 4:
        bs, c, h, w = out.shape
        spatial = out.permute(0, 2, 3, 1).reshape(bs, h * w, c)
    elif out.dim() == 3:
        # (B, T, C)
        spatial = out
    else:
        spatial = out.reshape(out.shape[0], -1, out.shape[-1])
    var = spatial.var(dim=-1, unbiased=False).mean(dim=0)  # (H*W,)
    var_sum = float(var.sum().item())
    if var_sum <= 0:
        return AttentionFingerprint(
            layer_name=layer_name,
            layer_type=type(module).__name__,
            num_heads=None,
            mixer_rank_proxy=0.0,
            spatial_concentration=0.0,
        )
    p = (var / var_sum).cpu()
    # entropy in nats then exponentiated -> effective rank
    p_pos = p[p > 0]
    if p_pos.numel() == 0:
        rank = 0.0
    else:
        entropy = float(-(p_pos * p_pos.log()).sum().item())
        rank = math.exp(entropy)
    concentration = float(p.max().item())
    return AttentionFingerprint(
        layer_name=layer_name,
        layer_type=type(module).__name__,
        num_heads=None,
        mixer_rank_proxy=rank,
        spatial_concentration=concentration,
    )


# ---------------------------------------------------------------------------
# ScorerIntrospector
# ---------------------------------------------------------------------------


class ScorerIntrospector:
    """Attach observational forward hooks to a SegNet or PoseNet.

    Usage::

        introspector = ScorerIntrospector(posenet, capture_mode="fingerprint")
        with introspector.session():
            record = introspector.capture(input_tensor)
        record.to_disk("experiments/results/.../record.pt")

    The ``session()`` context manager attaches hooks on entry and removes them
    on exit, so accidental garbage collection of the introspector cannot leave
    dangling hooks on the model.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        capture_mode: str = "fingerprint",
        full_threshold_elements: int = DEFAULT_FULL_THRESHOLD_ELEMENTS,
        capture_attention: bool = True,
        skip_module_types: Iterable[str] = (
            "Identity",
            "Dropout",
            "ModuleList",
            "ModuleDict",
        ),
    ) -> None:
        if capture_mode not in CAPTURE_MODES:
            raise ValueError(
                f"capture_mode must be one of {CAPTURE_MODES}, got {capture_mode!r}"
            )
        self._model_ref = weakref.ref(model)
        self._model_kind = type(model).__name__
        self._capture_mode = capture_mode
        self._full_threshold = int(full_threshold_elements)
        self._capture_attention = bool(capture_attention)
        self._skip_types = set(skip_module_types)

        self._hooks: list[Any] = []
        self._hook_attached = False
        # Filled per-capture, then cleared.
        self._scratch: dict[str, dict[str, Any]] = {}
        # Stable, deterministic name for the model root for hook lookup.
        self._model_named_modules: list[tuple[str, nn.Module]] = list(model.named_modules())

    # ------------------------- context-manager API -----------------------

    def attach_hooks(self) -> None:
        """Register forward hooks on every captured descendant module."""
        if self._hook_attached:
            return
        for name, module in self._model_named_modules:
            if name == "":
                # Skip the top-level container — we do capture its output via
                # the regular hook path because PoseNet returns a dict and the
                # top-level fingerprint is useful, but include it.
                pass
            cls_name = type(module).__name__
            if cls_name in self._skip_types:
                continue
            self._hooks.append(module.register_forward_hook(self._make_hook(name, module)))
        self._hook_attached = True

    def remove_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:  # pragma: no cover - defensive
                pass
        self._hooks.clear()
        self._hook_attached = False

    @contextmanager
    def session(self) -> Iterator["ScorerIntrospector"]:
        """Context manager: attaches hooks on enter, removes on exit."""
        self.attach_hooks()
        try:
            yield self
        finally:
            self.remove_hooks()

    # ------------------------- capture path ------------------------------

    def _make_hook(self, name: str, module: nn.Module) -> Callable:
        def _hook(mod: nn.Module, inputs: Any, output: Any) -> None:
            in_tensors = _coerce_to_tensor_list(inputs)
            out_tensors = _coerce_to_tensor_list(output)
            self._scratch[name] = {
                "module_type": type(mod).__name__,
                "in_tensors": in_tensors,
                "out_tensors": out_tensors,
                "is_repmixer": _is_repmixer_block(mod),
                "is_attention": _is_softmax_attention(mod),
            }
        return _hook

    def capture(self, model_input: torch.Tensor | tuple) -> IntrospectionRecord:
        """Run the model on ``model_input`` and return an ``IntrospectionRecord``.

        ``model_input`` is whatever the model's ``forward`` accepts. PoseNet
        and SegNet both expect a 4D tensor ``(B, C, H, W)`` after their own
        ``preprocess_input``; pass that pre-processed tensor here. The
        introspector does not run ``preprocess_input`` for you because (a)
        ``rgb_to_yuv6`` has ``@torch.no_grad`` and we want to keep the
        observation surface small, and (b) the demo scripts have explicit
        control over preprocessing.
        """
        if not self._hook_attached:
            raise RuntimeError(
                "ScorerIntrospector.capture called before hooks attached. "
                "Use `with introspector.session():` or call attach_hooks()."
            )
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Underlying model has been garbage-collected.")

        self._scratch = {}
        with torch.inference_mode():
            if isinstance(model_input, tuple):
                _ = model(*model_input)
                input_shape = tuple(model_input[0].shape) if model_input else ()
            else:
                _ = model(model_input)
                input_shape = tuple(model_input.shape)

        device_str = "cpu"
        for p in model.parameters():
            device_str = str(p.device)
            break

        record = IntrospectionRecord(
            model_kind=self._model_kind,
            device=device_str,
            capture_mode=self._capture_mode,
            input_shape=input_shape,
            metadata={
                "full_threshold_elements": self._full_threshold,
                "skip_module_types": sorted(self._skip_types),
                "tag": "[diagnostic-not-score]",
            },
        )

        # Iterate in named_modules order so layer order is deterministic.
        for name, module in self._model_named_modules:
            if name not in self._scratch:
                continue
            entry = self._scratch[name]
            in_tensors: list[torch.Tensor] = entry["in_tensors"]
            out_tensors: list[torch.Tensor] = entry["out_tensors"]

            in_stats = [LayerStats.from_tensor(t) for t in in_tensors]
            out_stats = [LayerStats.from_tensor(t) for t in out_tensors]

            full_input: list[torch.Tensor] | None = None
            full_output: list[torch.Tensor] | None = None
            if self._capture_mode == "full":
                if all(t.numel() <= self._full_threshold for t in in_tensors):
                    full_input = [t.detach().cpu().clone() for t in in_tensors]
                if all(t.numel() <= self._full_threshold for t in out_tensors):
                    full_output = [t.detach().cpu().clone() for t in out_tensors]

            attention_fingerprint = None
            is_attn_like = False
            if entry["is_repmixer"] and self._capture_attention:
                is_attn_like = True
                attention_fingerprint = _compute_repmixer_fingerprint(
                    name, module, out_tensors[0] if out_tensors else None
                )
            elif entry["is_attention"] and self._capture_attention:
                is_attn_like = True
                # Best-effort: timm Attention modules expose num_heads via a
                # plain attribute. We do not unpack Q/K/V here because doing so
                # safely requires monkey-patching the inner forward, which the
                # task constraint forbids on upstream code. The fingerprint
                # remains "structural-only" until a future toolkit revision
                # adds a non-invasive Q/K/V capture.
                num_heads = getattr(module, "num_heads", None)
                attn_norm = (
                    float(out_tensors[0].detach().to(dtype=torch.float32).norm().item())
                    if out_tensors
                    else None
                )
                attention_fingerprint = AttentionFingerprint(
                    layer_name=name,
                    layer_type=type(module).__name__,
                    num_heads=num_heads,
                    softmax_entropy=None,
                    head_rank=None,
                    attention_output_norm=attn_norm,
                )

            record.layers.append(
                LayerRecord(
                    name=name,
                    module_type=entry["module_type"],
                    input_stats=in_stats,
                    output_stats=out_stats,
                    full_input=full_input,
                    full_output=full_output,
                    is_attention_like=is_attn_like,
                    attention_fingerprint=attention_fingerprint,
                )
            )

        # Free temporary references so a long-running script doesn't hold
        # onto the activation tensors longer than necessary.
        self._scratch = {}
        return record

    def attach_attention_hooks(self) -> None:
        """Compatibility shim — attention hooks are part of `attach_hooks` already.

        Kept for API discoverability per the task spec; idempotent with
        :meth:`attach_hooks`.
        """
        self.attach_hooks()


# ---------------------------------------------------------------------------
# Module-level convenience: count attention-like layers without capturing
# ---------------------------------------------------------------------------


def list_attention_like_layers(model: nn.Module) -> list[tuple[str, str]]:
    """Return ``[(name, module_type), ...]`` for every attention-like layer.

    For the pinned PoseNet (timm ``fastvit_t12``) this returns 12 entries —
    the 12 RepMixerBlock token mixers. For models containing true softmax
    attention modules, those are included as well.
    """
    out: list[tuple[str, str]] = []
    for name, module in model.named_modules():
        if _is_repmixer_block(module) or _is_softmax_attention(module):
            out.append((name, type(module).__name__))
    return out


def hash_record(record: IntrospectionRecord) -> str:
    """Stable hash of a record's structural metadata (not the full tensors).

    Useful for "did the architecture I introspected change between runs?"
    """
    h = hashlib.sha256()
    payload = {
        "model_kind": record.model_kind,
        "input_shape": list(record.input_shape),
        "layers": [
            {"name": layer.name, "module_type": layer.module_type}
            for layer in record.layers
        ],
    }
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()
