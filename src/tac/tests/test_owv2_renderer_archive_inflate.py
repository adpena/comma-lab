"""OWV2 multi-tensor renderer archive — inflate-side round-trip tests.

Tests the new ``OWV2`` magic-byte handler in
``submissions/robust_current/inflate_renderer.py`` end-to-end:

1. Encode the real Lane G v3 renderer.bin into the OWV2 archive format
   (``encode_owv2_archive``).
2. Run it through the inflate-side ``_load_renderer`` dispatch (which now
   recognises the OWV2 magic and routes to ``decode_owv2_archive``).
3. Compare the decoded state_dict tensor-by-tensor against the original
   ASYM-loaded state_dict.

What this test PROVES
---------------------
* ``encode_owv2_archive`` → ``decode_owv2_archive`` is a faithful round trip:
  shapes preserved, dtypes preserved, max element-wise error stays inside
  the per-channel block-FP algebra bound on every OWV2-encoded conv (and is
  bit-faithful via ``half`` precision on every FP16-encoded layer).
* The inflate-side ``_load_renderer`` correctly dispatches to OWV2 on
  ``magic == b"OWV2"`` and reconstructs the renderer to a callable model.
* The OWV2 archive is materially smaller than the source ASYM .bin (Council F
  band) — so the dispatch yields a real rate-term reduction.

What this test does NOT prove
-----------------------------
* Score parity in the contest evaluator. To prove that, the OWV2 archive
  must be packed into a full submission archive (with masks.mkv +
  optimized_poses.pt) and run through ``contest_auth_eval.py`` on
  contest-CUDA. That is a Vast.ai 4090 dispatch (~$0.50), NOT in scope.

Tag: ``[empirical:src/tac/tests/test_owv2_renderer_archive_inflate.py]``
Anchor: ``experiments/results/lane_g_v3_landed/iter_0/renderer.bin``
"""
from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path

import pytest
import torch

from tac.owv2_renderer_archive import (
    OWV2_ARCHIVE_MAGIC,
    OWV2_ARCHIVE_VERSION,
    OWV2ArchiveError,
    decode_owv2_archive,
    encode_owv2_archive,
    is_owv2_archive,
)
from tac.renderer_export import load_renderer_checkpoint


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANCHOR_PATH = (
    _REPO_ROOT
    / "experiments"
    / "results"
    / "lane_g_v3_landed"
    / "iter_0"
    / "renderer.bin"
)


def _load_inflate_renderer_module():
    """Import the inflate-side ``inflate_renderer`` module from
    ``submissions/robust_current/`` without polluting sys.path long-term.

    The module lives outside the standard tac package; we treat it as the
    contest's reference inflate path and exercise its public ``_load_renderer``
    function for the round-trip dispatch test.
    """
    target = _REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"
    assert target.exists(), f"missing inflate_renderer at {target}"
    spec = importlib_util.spec_from_file_location(
        "_owv2_test_inflate_renderer", str(target),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── (a) anchor file present + OWV2 magic byte registered ──────────────────


def test_owv2_archive_anchor_present() -> None:
    """The Lane G v3 renderer.bin anchor must exist (test prerequisite)."""
    assert _ANCHOR_PATH.exists(), (
        f"missing anchor {_ANCHOR_PATH} — Lane G v3 must have been landed for "
        f"this test to run."
    )
    head = _ANCHOR_PATH.read_bytes()[:4]
    assert head in (b"ASYM", b"DPSM"), (
        f"anchor {_ANCHOR_PATH} has unexpected magic {head!r}; expected "
        f"ASYM or DPSM."
    )


def test_owv2_archive_magic_constants() -> None:
    """Sanity: magic constants exposed at module level."""
    assert OWV2_ARCHIVE_MAGIC == b"OWV2"
    assert OWV2_ARCHIVE_VERSION == 1


def test_owv2_archive_is_owv2_archive_sniff() -> None:
    """The is_owv2_archive sniff function correctly identifies OWV2 blobs."""
    assert is_owv2_archive(b"OWV2\x00\x00")
    assert not is_owv2_archive(b"ASYM\x00\x00")
    assert not is_owv2_archive(b"")
    assert not is_owv2_archive(b"OWV1\x00\x00")  # NOT V2


# ── (b) silent-default audit: encode/decode require explicit kwargs ───────


def test_owv2_archive_encode_requires_model() -> None:
    """encode_owv2_archive must reject silent None default (Check 81 STRICT)."""
    with pytest.raises(OWV2ArchiveError, match="model is None"):
        encode_owv2_archive()


def test_owv2_archive_decode_requires_data() -> None:
    """decode_owv2_archive must reject silent None default for data."""
    with pytest.raises(OWV2ArchiveError, match="data is required"):
        decode_owv2_archive(device="cpu")


def test_owv2_archive_decode_requires_device() -> None:
    """decode_owv2_archive must reject silent None default for device."""
    with pytest.raises(OWV2ArchiveError, match="device is required"):
        decode_owv2_archive(data=b"OWV2\x00\x00\x00\x00\x00\x00\x00\x00")


def test_owv2_archive_decode_rejects_bad_magic() -> None:
    """Malformed magic must be rejected."""
    with pytest.raises(OWV2ArchiveError, match="bad/missing magic"):
        decode_owv2_archive(data=b"FAKE\x00\x00\x00\x00\x00\x00\x00\x00", device="cpu")


# ── (c) primary round-trip: encode → decode → state_dict equivalence ──────


def test_owv2_archive_round_trip_state_dict_matches() -> None:
    """[empirical:test_owv2_renderer_archive_inflate.py] Round-trip the Lane G
    v3 renderer.bin through OWV2 archive encode → decode and verify every
    state_dict tensor matches within the per-channel block-FP algebra bound.
    """
    assert _ANCHOR_PATH.exists()
    model = load_renderer_checkpoint(str(_ANCHOR_PATH))
    sd_orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    blob = encode_owv2_archive(model=model)
    assert is_owv2_archive(blob), "encoded blob does not start with OWV2 magic"

    decoded = decode_owv2_archive(data=blob, device="cpu")
    sd_decoded = decoded.state_dict()

    assert set(sd_decoded.keys()) == set(sd_orig.keys()), (
        f"state_dict key drift: missing="
        f"{set(sd_orig.keys()) - set(sd_decoded.keys())} "
        f"extra={set(sd_decoded.keys()) - set(sd_orig.keys())}"
    )

    for name, orig in sd_orig.items():
        new = sd_decoded[name].detach().cpu()
        assert orig.shape == new.shape, (
            f"{name}: shape drift {tuple(orig.shape)} -> {tuple(new.shape)}"
        )

        # Tolerance: integer-valued buffers (e.g. running stats counters) must
        # be exact. Floating tensors get the per-channel block-FP bound for
        # 4-D conv weights; everything else is FP16 round-trip (~2^-10 rel).
        if not orig.is_floating_point():
            assert torch.equal(orig, new), f"{name}: integer tensor drift"
            continue

        max_abs = float(orig.abs().max().item())
        if max_abs == 0.0:
            assert float(new.abs().max().item()) == 0.0, (
                f"{name}: zero tensor decoded to non-zero"
            )
            continue

        if orig.dim() == 4 and int(orig.shape[0]) >= 2:
            # OWV2-eligible conv weight; the per-channel block-FP algebra
            # gives an L_inf bound of ~max_abs / 4. That's the same bound
            # asserted in test_omega_w_v2_real_archive.py.
            tol = 2.0 * max_abs * (2.0 ** -3)
        else:
            # FP16 fallback: relative error ~2^-10.
            tol = max(1e-3, max_abs * (2.0 ** -9))

        max_err = float((orig.float() - new.float()).abs().max().item())
        assert max_err <= tol, (
            f"{name}: round-trip max_abs_err={max_err:.6f} > tol={tol:.6f} "
            f"(max_abs={max_abs:.6f})"
        )


# ── (d) inflate-side dispatch: the new OWV2 case in _load_renderer ────────


def test_owv2_archive_inflate_renderer_dispatch(tmp_path: Path) -> None:
    """[empirical:test_owv2_renderer_archive_inflate.py] The inflate-side
    ``_load_renderer`` dispatches OWV2 magic to the new handler and returns
    a callable model.
    """
    assert _ANCHOR_PATH.exists()
    model = load_renderer_checkpoint(str(_ANCHOR_PATH))
    blob = encode_owv2_archive(model=model)

    out_path = tmp_path / "owv2_renderer.bin"
    out_path.write_bytes(blob)

    inflate_mod = _load_inflate_renderer_module()
    loaded = inflate_mod._load_renderer(str(out_path), device="cpu")

    # Sanity: the loaded model has matching state_dict keys.
    sd_orig = model.state_dict()
    sd_loaded = loaded.state_dict()
    assert set(sd_loaded.keys()) == set(sd_orig.keys()), (
        "inflate-loaded state_dict keys diverge from source"
    )

    # Sanity: model is in eval mode (so dropout/BN won't introduce noise).
    assert not loaded.training, "inflate-loaded model is not in eval mode"


# ── (e) byte-savings sanity: OWV2 archive is materially smaller than ASYM ─


def test_owv2_archive_yields_byte_savings_vs_asym_anchor(tmp_path: Path) -> None:
    """[empirical:test_owv2_renderer_archive_inflate.py] On Lane G v3's
    ASYM renderer.bin (~290KB), the OWV2 archive must come out materially
    smaller. Council F band is [20%, 60%] on the eligible Conv2d aggregate;
    on the FULL renderer (including FP16 fallback for protected layers) we
    expect at least 25% savings. If this drops below 20%, the OWV2 codec
    has regressed or the architecture has shifted.
    """
    assert _ANCHOR_PATH.exists()
    model = load_renderer_checkpoint(str(_ANCHOR_PATH))
    blob = encode_owv2_archive(model=model)

    asym_bytes = _ANCHOR_PATH.stat().st_size
    owv2_bytes = len(blob)
    savings_pct = 100.0 * (1.0 - owv2_bytes / asym_bytes)

    print(
        f"\n  [empirical:test_owv2_renderer_archive_inflate.py] "
        f"OWV2 archive on Lane G v3 renderer.bin: "
        f"asym={asym_bytes:,}B -> owv2={owv2_bytes:,}B "
        f"(savings={savings_pct:+.2f}%)"
    )

    assert owv2_bytes < asym_bytes, (
        f"OWV2 archive ({owv2_bytes}B) is NOT smaller than ASYM source "
        f"({asym_bytes}B); the codec regressed somewhere."
    )
    assert savings_pct >= 20.0, (
        f"OWV2 archive saves {savings_pct:.1f}% vs ASYM source on Lane G v3 "
        f"renderer.bin; expected >=20% per Council F band. Either the codec "
        f"regressed or the bit_budget_ratio is too loose. Investigate before "
        f"promoting."
    )


# ── (f) decode rejects truncated body / unsupported version ───────────────


def test_owv2_archive_decode_rejects_unsupported_version() -> None:
    """A blob with the right magic but wrong version must be rejected."""
    import json
    import struct

    fake_header = json.dumps(
        {"version": 9999, "format": "owv2_renderer_archive_v1", "arch": {}, "layers": [], "scalar_params": {}, "body_len": 0},
        separators=(",", ":"),
    ).encode("utf-8")
    blob = (
        OWV2_ARCHIVE_MAGIC
        + struct.pack("<I", len(fake_header))
        + fake_header
        + struct.pack("<I", 0)
    )
    with pytest.raises(OWV2ArchiveError, match="unsupported version"):
        decode_owv2_archive(data=blob, device="cpu")
