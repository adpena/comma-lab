# SPDX-License-Identifier: MIT
"""Strip torch.nn.utils.parametrize hooks from saved state_dicts.

When a model is saved while `torch.nn.utils.parametrize` hooks are attached
(self-compress, FakeQuant FP4, INT8 QAT, weight_norm, etc.), the state_dict
contains keys like:

    layer.parametrizations.weight.original   # the underlying float weight
    layer.parametrizations.weight.0.codebook # parametrize-internal state
    parametrizations.weight.original         # ROOT-level (no leading dot)
    layer.parametrizations.weight.original0  # weight_norm: magnitude
    layer.parametrizations.weight.original1  # weight_norm: direction

A fresh model (loader-side, no hooks attached) expects plain `layer.weight`.
Loading the state_dict directly fails with `missing keys=['layer.weight', ...]
unexpected keys=['layer.parametrizations.weight.original', ...]`.

This module provides the canonical strip helper. Replaces 4 inline impls:
  - `experiments/qat_finetune.py:218-238`
  - `experiments/train_joint_pair.py:980-982`
  - `experiments/train_distill.py:1288-1290`
  - inline Stage 3 in `scripts/remote_lane_i_coolchic_masks.sh`

## Edge cases handled (Round 11 codex adversarial review)

1. **Root-level parametrizations**: keys like `parametrizations.weight.original`
   (no leading dot) emitted by `register_parametrization` on a root nn.Module.
2. **Multi-original (weight_norm style)**: `original0` + `original1` where
   PyTorch's `parametrize.remove_parametrizations` would COMPUTE the combined
   weight at strip time. We CANNOT compute that statically without instantiating
   the parametrize class. We DROP these with a warning (the caller must use
   `parametrize.remove_parametrizations(model, attr)` on a live module instead).
3. **Nested parametrize chains**: `layer.parametrizations.weight.0.parametrizations.weight.original`
   uses the OUTERMOST parametrizations marker (canonical interpretation).

References:
  - `project_lane_i_crashed_parametrize_strip_20260428`
  - `feedback_parametrize_strip_helper_factored_20260428`
  - Codex Round 11 adversarial review (2026-04-28) — surfaced edge cases
"""
from __future__ import annotations

import warnings
from typing import Mapping


_DROPPED_MULTI_ORIG_WARNED: set[str] = set()


def _parse_param_key(key: str) -> tuple[str | None, str | None, str | None]:
    """Parse a state_dict key into (path_before, param_name, suffix).

    Returns (None, None, None) if the key is not a parametrize hook.

    Path components:
        <path>.parametrizations.<param_name>.<suffix...>

    The OUTERMOST `parametrizations` marker (FIRST occurrence) is used
    when nested parametrizations exist. Root-level layouts (key starts
    with `parametrizations.`) are handled by treating path_before as "".
    """
    parts = key.split(".")
    # Find the FIRST occurrence of `parametrizations` (outer-most when nested)
    try:
        i = parts.index("parametrizations")
    except ValueError:
        return None, None, None
    # Need at least: parametrizations.<name>.<suffix>
    if i + 2 >= len(parts):
        return None, None, None
    path_before = ".".join(parts[:i])  # may be "" for root-level
    param_name = parts[i + 1]
    # Suffix is everything after parametrizations.<name>.
    # For nested, this captures the full inner chain — caller decides.
    suffix = ".".join(parts[i + 2:])
    return path_before, param_name, suffix


def _is_canonical_original(suffix: str) -> bool:
    """True iff suffix is exactly 'original' (single-original parametrize).

    Single-original case is the SAFE strip — we know the underlying weight
    tensor is the parametrize input. PyTorch's standard parametrize uses
    `original`. self_compress / FakeQuantFP4 / INT8 QAT all use `original`.
    """
    return suffix == "original"


def _is_multi_original(suffix: str) -> bool:
    """True iff suffix is `originalN` for some non-negative integer N.

    Multi-original is used by `weight_norm` (original0=g, original1=v) and
    similar parametrizations that need multiple underlying tensors. We
    can't statically combine them — caller must use `parametrize.remove_parametrizations`
    on a live module instead. Strip drops with warning.
    """
    if not suffix.startswith("original"):
        return False
    rest = suffix[len("original"):]
    return rest.isdigit() and len(rest) > 0


def strip_parametrize_hooks(
    state: Mapping[str, "object"],
    *,
    drop_internal: bool = True,
    warn_multi_original: bool = True,
) -> dict[str, "object"]:
    """Return a copy of ``state`` with parametrize hook keys normalized.

    Mapping rules:
      - ``[<path>.]parametrizations.<name>.original`` → ``[<path>.]<name>``
        (single-original — safe rename)
      - ``[<path>.]parametrizations.<name>.original<N>`` → DROPPED with warning
        (multi-original e.g. weight_norm — can't statically combine)
      - Other ``[<path>.]parametrizations.<name>.<internal>`` keys (codebook,
        _buffers, nested chains, …) are DROPPED when ``drop_internal=True``
        (default).
      - All other keys (no parametrize marker) pass through unchanged.

    Args:
        state: The state_dict to strip.
        drop_internal: When True (default), parametrize-internal keys
            (codebook, _buffers, etc.) are dropped. When False, all
            parametrize keys (except the renamed `original`) pass through.
        warn_multi_original: When True (default), emit a UserWarning the
            FIRST time a multi-original parametrize key is encountered for
            each unique (path, name) pair. Helps caller know to use
            `parametrize.remove_parametrizations(model, attr)` on a live
            module to combine the originals correctly.

    Examples
    --------
    >>> import torch
    >>> state = {
    ...     "renderer.head.weight": torch.zeros(1),
    ...     "renderer.conv1.parametrizations.weight.original": torch.ones(1),
    ...     "renderer.conv1.parametrizations.weight.0.codebook": torch.full((1,), 7.0),
    ...     "parametrizations.bias.original": torch.tensor([0.5]),  # ROOT-level
    ... }
    >>> out = strip_parametrize_hooks(state)
    >>> sorted(out.keys())
    ['bias', 'renderer.conv1.weight', 'renderer.head.weight']
    """
    if not has_parametrize_keys(state):
        return dict(state)
    out: dict[str, "object"] = {}
    for k, v in state.items():
        path, name, suffix = _parse_param_key(k)
        if path is None:
            # Not a parametrize key — pass through unchanged
            out[k] = v
            continue
        if _is_canonical_original(suffix):
            # Standard rename: parametrizations.<name>.original → <name>
            target_key = f"{path}.{name}" if path else name
            out[target_key] = v
            continue
        if _is_multi_original(suffix):
            # weight_norm-style: drop, can't statically combine
            if warn_multi_original:
                token = f"{path}.{name}" if path else name
                if token not in _DROPPED_MULTI_ORIG_WARNED:
                    _DROPPED_MULTI_ORIG_WARNED.add(token)
                    warnings.warn(
                        f"strip_parametrize_hooks: dropped multi-original "
                        f"parametrize key for '{token}' (suffix={suffix!r}). "
                        f"weight_norm and similar multi-original parametrizes "
                        f"cannot be combined statically — load the checkpoint "
                        f"INTO a model with the same parametrize registered, "
                        f"then call torch.nn.utils.parametrize."
                        f"remove_parametrizations(model, '{name}'). "
                        f"Result of this strip will have MISSING '{token}' key.",
                        UserWarning, stacklevel=2,
                    )
            continue
        # Other internals (codebook, _buffers, nested chains, etc.)
        if drop_internal:
            continue
        out[k] = v
    return out


def has_parametrize_keys(state: Mapping[str, "object"]) -> bool:
    """True iff `state` contains any parametrize-hook keys.

    Detects both nested layer keys (`layer.parametrizations.…`) and ROOT-level
    keys (`parametrizations.…` with no leading path).
    """
    for k in state:
        parts = k.split(".")
        if "parametrizations" in parts:
            return True
    return False


def reset_warning_cache() -> None:
    """Reset the multi-original warning cache. Useful for tests."""
    _DROPPED_MULTI_ORIG_WARNED.clear()


__all__ = [
    "strip_parametrize_hooks",
    "has_parametrize_keys",
    "reset_warning_cache",
]
