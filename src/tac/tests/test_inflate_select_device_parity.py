# SPDX-License-Identifier: MIT
"""Parity tests: substrate inflate.py select_inflate_device == canonical.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog #205
(``check_inflate_py_uses_canonical_select_inflate_device``): every
``submissions/*/inflate.py`` must contain a ``select_inflate_device``
helper. Per A1 council Round 1 finding F1/F11, all 5 affected substrates
were drifting in body shape — some returned ``torch.device``, some
returned ``str``, all used slightly different reject ordering for
unknown values.

This module verifies that the body shape across the 5 affected substrates
(a1, anr_substrate, categorical_substrate, hnerv_lc_ac, scpp_substrate)
is BYTE-EQUIVALENT in behavior to the canonical
``tac.substrates._shared.inflate_runtime.select_inflate_device``.
Contest runtime closure forbids inflate.py importing from ``tac`` at
inflate time, so the canonical helper is duplicated literally in each
inflate.py — this parity test is the structural protection against
drift.

Lane: ``lane_canonicalize_inflate_and_smoke_auth_eval_20260514``.
Memory: ``feedback_canonicalize_inflate_and_smoke_auth_eval_landed_20260514.md``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


# Each entry: (slug, returns_torch_device_bool)
# The five substrates the BUG-FIX-WAVE subagent identified as inline-fork
# offenders + the canonical helper itself (which returns ``str``).
SUBSTRATES = [
    ("a1", True),
    ("anr_substrate", False),
    ("categorical_substrate", False),
    ("hnerv_lc_ac", True),
    ("scpp_substrate", True),
]


def _load_substrate_module(slug: str):
    """Import submissions/<slug>/inflate.py as a module without importing tac."""

    inflate_path = REPO_ROOT / "submissions" / slug / "inflate.py"
    assert inflate_path.is_file(), f"missing inflate.py at {inflate_path}"
    mod_name = f"_test_inflate_{slug}"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, inflate_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    # Each inflate.py runs sys.path.insert for its own ``src/``; allow that.
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _canonical_str(env_value: str | None, cuda_available: bool) -> str:
    """Mirror of tac.substrates._shared.inflate_runtime.select_inflate_device.

    This function is intentionally re-implemented INLINE (not imported from
    tac) so the test verifies the shape — if the canonical body changes in
    tac, this test mirror must be updated in lockstep, surfacing the drift.
    """

    value = (env_value or "auto").strip().lower()
    if value == "auto":
        return "cuda" if cuda_available else "cpu"
    if value == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "PACT_INFLATE_DEVICE=cuda but torch.cuda is not available"
            )
        return "cuda"
    if value == "cpu":
        return "cpu"
    raise RuntimeError(
        f"unsupported PACT_INFLATE_DEVICE={value!r}; expected auto/cpu/cuda"
    )


# Cross-product of env-var settings × cuda-available booleans we expect to
# yield IDENTICAL behavior across all 5 substrates and the canonical.
ENV_VALUES = [None, "", "auto", "AUTO", "cuda", "CUDA", "cpu", "CPU"]


def _normalize_return(val) -> str:
    """Collapse torch.device('cuda') | 'cuda' -> 'cuda' (string)."""

    if isinstance(val, torch.device):
        return val.type
    if isinstance(val, str):
        return val
    raise TypeError(f"unexpected return type {type(val)!r} from select_inflate_device")


@pytest.mark.parametrize("slug,returns_device", SUBSTRATES)
@pytest.mark.parametrize("env_value", ENV_VALUES)
@pytest.mark.parametrize("cuda_available", [False, True])
def test_select_inflate_device_parity(
    slug: str,
    returns_device: bool,
    env_value: str | None,
    cuda_available: bool,
) -> None:
    """Each substrate's helper must agree with the canonical (modulo wrap)."""

    mod = _load_substrate_module(slug)
    env_patch = {} if env_value is None else {"PACT_INFLATE_DEVICE": env_value}

    def _call_substrate():
        with mock.patch.dict(os.environ, env_patch, clear=False), mock.patch(
            "torch.cuda.is_available", return_value=cuda_available
        ):
            if env_value is None:
                os.environ.pop("PACT_INFLATE_DEVICE", None)
            return mod.select_inflate_device()

    canonical_raised: BaseException | None = None
    canonical_result: str | None = None
    try:
        canonical_result = _canonical_str(env_value, cuda_available)
    except BaseException as exc:  # noqa: BLE001
        canonical_raised = exc

    if canonical_raised is not None:
        # Substrate must raise too (parity on the failure mode)
        with pytest.raises(type(canonical_raised)):
            _call_substrate()
        return

    got = _call_substrate()
    got_str = _normalize_return(got)
    assert got_str == canonical_result, (
        f"{slug}.select_inflate_device(env={env_value!r}, "
        f"cuda_available={cuda_available}) returned {got!r} "
        f"({got_str}); canonical: {canonical_result}"
    )


@pytest.mark.parametrize("slug,returns_device", SUBSTRATES)
def test_unknown_env_value_raises(slug: str, returns_device: bool) -> None:
    mod = _load_substrate_module(slug)
    with mock.patch.dict(os.environ, {"PACT_INFLATE_DEVICE": "mps"}, clear=False):
        with pytest.raises(RuntimeError, match="unsupported PACT_INFLATE_DEVICE"):
            mod.select_inflate_device()


@pytest.mark.parametrize("slug,returns_device", SUBSTRATES)
def test_metal_env_value_raises(slug: str, returns_device: bool) -> None:
    mod = _load_substrate_module(slug)
    with mock.patch.dict(os.environ, {"PACT_INFLATE_DEVICE": "metal"}, clear=False):
        with pytest.raises(RuntimeError, match="unsupported PACT_INFLATE_DEVICE"):
            mod.select_inflate_device()


@pytest.mark.parametrize("slug,returns_device", SUBSTRATES)
def test_cuda_no_cuda_available_raises(slug: str, returns_device: bool) -> None:
    mod = _load_substrate_module(slug)
    with mock.patch.dict(
        os.environ, {"PACT_INFLATE_DEVICE": "cuda"}, clear=False
    ), mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="cuda is not available"):
            mod.select_inflate_device()


@pytest.mark.parametrize("slug,returns_device", SUBSTRATES)
def test_return_type_signature(slug: str, returns_device: bool) -> None:
    """Each substrate preserves its documented return-type signature."""

    mod = _load_substrate_module(slug)
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PACT_INFLATE_DEVICE", None)
        with mock.patch("torch.cuda.is_available", return_value=False):
            got = mod.select_inflate_device()
    if returns_device:
        assert isinstance(got, torch.device), (
            f"{slug} must return torch.device per its signature; got {type(got)!r}"
        )
    else:
        assert isinstance(got, str), (
            f"{slug} must return str per its signature; got {type(got)!r}"
        )


def test_canonical_helper_returns_string() -> None:
    """The canonical helper itself returns str (per its docstring)."""

    from tac.substrates._shared.inflate_runtime import select_inflate_device

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PACT_INFLATE_DEVICE", None)
        with mock.patch("torch.cuda.is_available", return_value=False):
            assert select_inflate_device() == "cpu"
        with mock.patch("torch.cuda.is_available", return_value=True):
            assert select_inflate_device() == "cuda"


def test_all_five_substrates_documented() -> None:
    """Catalog #205 + the BUG-FIX-WAVE subagent identified exactly these 5."""

    slugs = {slug for slug, _ in SUBSTRATES}
    assert slugs == {
        "a1",
        "anr_substrate",
        "categorical_substrate",
        "hnerv_lc_ac",
        "scpp_substrate",
    }
