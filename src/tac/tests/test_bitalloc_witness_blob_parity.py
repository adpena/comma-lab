# SPDX-License-Identifier: MIT
"""Round-2 review F1 — blob-grammar parity pin for tools/apply_sensitivity_bitalloc_witness.py.

The #336 tool's ``_int8_realize`` re-states the #202 grammar (per-tensor ``_int8_symmetric``
int8 concat -> brotli-11) instead of calling ``build_levelset_blob``. MEASURED on the real
mod32cap snapshot (``snapshot_ema_BEST.npz``, ep425, 2026-07-07 one-shot probe): byte-IDENTICAL —
base int8 concat 72,695 B sha256-equal, code int8 38,400 B sha256-equal, brotli-11 streams
byte-equal (61,953 / 21,145 B), tensor order identical, delta = 0 bytes. This suite pins that
identity STRUCTURALLY on synthetic params (both real streams compared through the canonical
``read_levelset_blob`` reader), so any drift in either surface — tensor order, ``_int8_symmetric``,
brotli quality, blob layout — fails loudly. Also pins the render-authority coupling signatures the
#336 tool imports from tools/measure_contour_string_flip_coding.py (the F1(b) half).

means != ends: a parity pin, NOT a score; pointer 0.19110 moves only via a byte-closed exact row.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def ab():
    """The #336 bit-alloc apply tool (module-level imports are numpy+stdlib only)."""
    return _load(_REPO / "tools" / "apply_sensitivity_bitalloc_witness.py", "ab_parity_f1")


@pytest.fixture(scope="module")
def bc():
    """The canonical #202 byte-close tool (heavy import: pulls the trainer/MLX once per module)."""
    return _load(_REPO / "tools" / "levelset_byte_close_and_eval.py", "bc_parity_f1")


@pytest.fixture(scope="module")
def mc():
    """The #307 contour tool — the render authority the #336 tool reuses."""
    return _load(_REPO / "tools" / "measure_contour_string_flip_coding.py", "mc_parity_f1")


def _synthetic_params() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    # 'code' deliberately NOT last: pins that BOTH surfaces preserve the same relative base
    # order (build_levelset_blob skips code in place; the #336 tool pops it — same order).
    return {
        "in_proj.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "hidden.0.weight": rng.standard_normal((16, 16)).astype(np.float32),
        "code": (rng.standard_normal((12, 6)) * 3.0).astype(np.float32),
        "film.bias": rng.standard_normal((32,)).astype(np.float32),
        "out_sdf.weight": rng.standard_normal((5, 16)).astype(np.float32),
        "palette": rng.standard_normal((5, 3)).astype(np.float32),
    }


_CFG = {
    "n_pairs": 12, "n_classes": 5, "hidden_dim": 16, "n_hidden": 1, "mod_dim": 6,
    "activation": "step_basis", "softmax_temp": 1.0, "chroma": True,
    "wire_w0": 1.0, "wire_s0": 1.0, "hosc_beta": 1.0, "hosc_omega": 1.0,
    "bank_n_scales": 2, "bank_n_orient0": 4, "bank_f0": 1.5, "bank_base": 2.0,
    "bank_n_iso": 2, "max_bank_freq": None, "render_h": 24, "render_w": 32,
}
_SO = {"self_orient": False}


def test_int8_realize_baseline_byte_identical_to_build_levelset_blob(ab, bc, tmp_path):
    """The measured identity, pinned: same raw int8 payloads, same brotli-11 streams, same
    tensor order, and dequant parity through the canonical shipped reader."""
    import brotli

    from tac.local_acceleration.torch_levelset_inflate import dequant_params, read_levelset_blob

    params = _synthetic_params()
    params_fp = {k: v for k, v in params.items() if k != "code"}
    code_fp = params["code"]

    dq, code_dq, base_b, code_b = ab._int8_realize(params_fp, code_fp)
    blob, breakdown = bc.build_levelset_blob(params, _CFG, _SO, None)

    p = tmp_path / "blob.bin"
    p.write_bytes(blob)
    manifest, base_br, code_br, pose = read_levelset_blob(p)

    # raw int8 payloads byte-identical (the parity the F1 probe measured on the real snapshot)
    assert brotli.decompress(base_br) == base_b
    assert brotli.decompress(code_br) == code_b
    # brotli-11 rate accounting byte-identical too (the bytes_before/after the tool reports)
    assert brotli.compress(base_b, quality=11) == base_br
    assert brotli.compress(code_b, quality=11) == code_br
    assert breakdown["base_int8_brotli_bytes"] == len(base_br)
    assert breakdown["code_int8_brotli_bytes"] == len(code_br)
    # same base tensor ORDER despite 'code' sitting mid-dict on entry
    assert manifest["base_param_order"] == list(params_fp.keys())
    assert pose == b""
    # dequant parity: the tool's dq values == the canonical reader's dequant of the blob
    canon_dq = dequant_params(
        brotli.decompress(base_br), manifest["base_param_order"],
        manifest["base_shapes"], manifest["base_scales"])
    for k in params_fp:
        np.testing.assert_array_equal(np.asarray(dq[k], np.float32).reshape(canon_dq[k].shape),
                                      canon_dq[k])
    code_canon = (np.frombuffer(brotli.decompress(code_br), np.int8).astype(np.float32)
                  * float(manifest["code_scale"])).reshape(manifest["code_shape"])
    np.testing.assert_array_equal(code_dq, code_canon)


def test_realize_alloc_empty_allocation_equals_baseline(ab):
    """_realize_alloc({}) IS the shipped int8-only path (the tool's baseline claim)."""
    params = _synthetic_params()
    params_fp = {k: v for k, v in params.items() if k != "code"}
    code_fp = params["code"]
    _, _, base_b0, code_b0 = ab._int8_realize(params_fp, code_fp)
    _, _, base_b1, code_b1, distinct = ab._realize_alloc(params_fp, code_fp, {})
    assert base_b1 == base_b0
    assert code_b1 == code_b0
    assert set(distinct) == {"base_stream", "code_stream"}


def test_render_authority_coupling_signatures(bc, mc, ab):
    """F1(b): the #336 tool imports render_frame1_both/segnet_argmax/build_render_ctx from the
    contour tool and calls bc._torch_R_reference / bc.detect_self_orient positionally — pin the
    exact parameter lists so a signature change in either tool fails here, not mid-daemon."""
    assert list(inspect.signature(mc.build_render_ctx).parameters) == \
        ["bc", "params_dq", "code_dq", "manifest", "lane_pairs"]
    assert list(inspect.signature(mc.render_frame1_both).parameters) == ["bc", "ctx", "pi"]
    assert len(inspect.signature(mc.segnet_argmax).parameters) == 2
    assert list(inspect.signature(bc._torch_R_reference).parameters) == \
        ["rgb", "rh", "rw", "ch", "cw"]
    assert list(inspect.signature(bc.detect_self_orient).parameters) == ["cfg", "so_overrides"]
    # the #336 tool's measure_d_seg surface: the 7 positional args its callers use + the
    # keyword-only chunked-foreground resume trio (2026-07-07 daemon-kill workaround: per-pair
    # cache + wall-clock deadline + persist callback; all optional, defaults preserve the
    # original 7-arg behavior value-identically).
    sig = inspect.signature(ab.measure_d_seg)
    assert list(sig.parameters) == \
        ["bc", "ctx", "seg_cpu", "lstars_all", "pair_ids", "params_dq", "code_dq",
         "pair_cache", "deadline", "persist"]
    for kw in ("pair_cache", "deadline", "persist"):
        p = sig.parameters[kw]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is None
