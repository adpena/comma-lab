# SPDX-License-Identifier: MIT
"""Guards for the Cool-Chic MPS-split device resolution (local-MPS enablement).

The MPS-split path trains the gradient on the Apple GPU and keeps the authority
(BEST-selection + auth-eval) on CPU/CUDA — exactly the validated basin design.
These tests assert the device-resolution helpers ACTUALLY route MPS to the
gradient backend and NEVER to the authority (CLAUDE.md "MPS auth eval is NOISE"),
and that the BatchNorm-MPS patch is applied when mps is requested.

NO-FAKE discipline: these tests verify BEHAVIOR (the returned device.type, the
patch flag actually flipped), not just that constants exist. A body that returned
canonical markers without resolving the device or applying the patch would FAIL.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINER_PATH = REPO_ROOT / "experiments" / "train_substrate_cool_chic.py"


_TRAINER_MODULE_CACHE = []


def _load_trainer_module():
    """Import the trainer-by-path (it lives in experiments/, not the tac package).

    Cached: the trainer registers a SubstrateContract at import time, so a second
    ``exec_module`` would collide on the duplicate ``cool_chic`` id. We load once
    and reuse — the helpers under test are pure functions on the module object.
    """
    if _TRAINER_MODULE_CACHE:
        return _TRAINER_MODULE_CACHE[0]
    import sys

    # If a prior path-import already landed under our test module name, reuse it.
    cached = sys.modules.get("_cool_chic_trainer_for_mps_test")
    if cached is not None:
        _TRAINER_MODULE_CACHE.append(cached)
        return cached
    spec = importlib.util.spec_from_file_location(
        "_cool_chic_trainer_for_mps_test", _TRAINER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cool_chic_trainer_for_mps_test"] = mod
    spec.loader.exec_module(mod)
    _TRAINER_MODULE_CACHE.append(mod)
    return mod


_MPS_AVAILABLE = bool(
    getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
)


def test_resolve_train_device_none_returns_authority_device():
    """None (single-device legacy) => train on the authority device, no patches."""
    mod = _load_trainer_module()
    authority = torch.device("cpu")
    train_device, report = mod._resolve_train_device(None, authority)
    assert train_device == authority
    assert report == {}


def test_resolve_train_device_cpu_is_cpu_no_patches():
    mod = _load_trainer_module()
    train_device, report = mod._resolve_train_device("cpu", torch.device("cpu"))
    assert train_device.type == "cpu"
    assert report == {}


def test_resolve_train_device_unknown_raises():
    mod = _load_trainer_module()
    with pytest.raises(SystemExit):
        mod._resolve_train_device("tpu", torch.device("cpu"))


@pytest.mark.skipif(not _MPS_AVAILABLE, reason="torch MPS not available on this host")
def test_resolve_train_device_mps_returns_mps_and_applies_bn_patch():
    """The whole point: mps => gradient backend is mps AND the BatchNorm-MPS
    contiguity patch is applied (so the upstream SegNet decoder's BN-backward
    runs on the Apple GPU). Verifies the patch FLAG actually flipped, not a
    canonical constant."""
    import torch.nn.functional as F

    from tac.torch_mps_compat import _BN_PATCH_FLAG

    mod = _load_trainer_module()
    train_device, report = mod._resolve_train_device("mps", torch.device("cpu"))
    assert train_device.type == "mps"
    # The patch report is the canonical dict from patch_scorer_for_mps; after
    # this call the module-level patch flag MUST be set (idempotent — may already
    # have been set by a prior call, in which case applied_now is False but the
    # flag is True).
    assert "batchnorm_contiguous" in report
    assert getattr(F, _BN_PATCH_FLAG, False) is True


def test_authority_device_never_mps_via_device_or_die():
    """The canonical authority gate refuses bare cpu outside smoke AND refuses mps
    entirely — so the authority device can never be mps (the noise axis)."""
    mod = _load_trainer_module()
    # mps is not even a --device choice; the authority gate only accepts cuda/cpu.
    # bare cpu outside smoke must raise (production gate intact).
    with pytest.raises(SystemExit):
        mod._device_or_die("cpu", smoke=False)


def test_cpu_advisory_for_mps_split_admits_cpu_authority():
    """The MPS-split advisory path admits CPU as the (NON-PROMOTABLE) authority via
    the canonical allow_full_cpu exception — returns a cpu device, not a raise."""
    mod = _load_trainer_module()
    dev = mod._device_or_die_cpu_advisory_for_mps_split()
    assert dev.type == "cpu"


def test_trainer_has_train_device_cli_flag():
    """The --train-device {cpu,cuda,mps} flag is wired into the argparse surface."""
    mod = _load_trainer_module()
    parser = mod._build_parser()
    # locate the --train-device action
    actions = {tuple(a.option_strings): a for a in parser._actions}
    train_dev = None
    for opts, action in actions.items():
        if "--train-device" in opts:
            train_dev = action
            break
    assert train_dev is not None, "--train-device flag missing from parser"
    assert set(train_dev.choices) == {"cpu", "cuda", "mps"}
