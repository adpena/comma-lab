# SPDX-License-Identifier: MIT
"""Tests for the canonical numpy-portable inflate bridge.

Coverage:
- state_dict pack/unpack round-trip (byte-stable, fp16 + fp32 + int8 dtypes)
- round-trip parity vs a torch state_dict (the migration target)
- each decode primitive vs its torch reference within FD tolerance
- pixel_shuffle / GRU cell exact-vs-torch byte stability
- the AST portability verifier (positive + negative)
- the runtime emitter produces a torch-FREE tree + self-verifies

torch is imported ONLY in the test (the bridge module itself is framework-free;
the test uses torch as the parity oracle per CLAUDE.md "Python remains the
oracle until native implementations pass the same vectors").
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from tac.substrates._shared import numpy_portable_inflate as bridge
from tac.substrates._shared.numpy_portable_inflate import (
    FORBIDDEN_INFLATE_FRAMEWORKS,
    InflateNotNumpyPortableError,
    NumpyPortableStateDictError,
    assert_inflate_is_numpy_portable,
    bilinear_resize_nhwc,
    conv2d_numpy,
    film_modulate_numpy,
    find_forbidden_framework_imports,
    gru_cell_numpy,
    linear,
    pack_state_dict_numpy,
    pixel_shuffle_2x_nhwc,
    sigmoid,
    tanh,
    unpack_state_dict_numpy,
    write_numpy_portable_contest_runtime,
)

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]


# --------------------------------------------------------------------------- #
# state_dict pack / unpack
# --------------------------------------------------------------------------- #


def test_pack_unpack_numpy_roundtrip_fp16():
    sd = {
        "layer0.weight": np.random.randn(8, 16).astype(np.float32),
        "layer0.bias": np.random.randn(8).astype(np.float32),
        "scalar": np.array(3.5, dtype=np.float32),
    }
    blob = pack_state_dict_numpy(sd, dtype="fp16")
    out = unpack_state_dict_numpy(blob)
    assert set(out) == set(sd)
    for k in sd:
        # fp16 storage: values match within fp16 rounding
        assert out[k].dtype == np.float16
        np.testing.assert_allclose(
            out[k].astype(np.float32), sd[k].astype(np.float16).astype(np.float32), rtol=0, atol=0
        )
        assert out[k].shape == sd[k].shape


def test_pack_unpack_byte_stable_deterministic():
    sd = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(4, dtype=np.float32)}
    b1 = pack_state_dict_numpy(sd, dtype="fp32")
    b2 = pack_state_dict_numpy(sd, dtype="fp32")
    assert b1 == b2  # byte-stable for fixed input


def test_pack_unpack_fp32_exact():
    sd = {"w": np.array([[1.25, -2.5], [3.75, 4.0]], dtype=np.float32)}
    out = unpack_state_dict_numpy(pack_state_dict_numpy(sd, dtype="fp32"))
    assert out["w"].dtype == np.float32
    np.testing.assert_array_equal(out["w"], sd["w"])


def test_pack_unpack_int8_exact():
    sd = {"q": np.array([-128, 0, 127, 42], dtype=np.int8)}
    out = unpack_state_dict_numpy(pack_state_dict_numpy(sd, dtype="int8"))
    assert out["q"].dtype == np.int8
    np.testing.assert_array_equal(out["q"], sd["q"])


def test_pack_preserves_insertion_order():
    sd = {f"k{i}": np.zeros(1, dtype=np.float32) for i in range(20)}
    out = unpack_state_dict_numpy(pack_state_dict_numpy(sd))
    assert list(out.keys()) == list(sd.keys())


def test_pack_unpack_scalar_zero_dim():
    sd = {"s": np.array(7.0, dtype=np.float32)}
    out = unpack_state_dict_numpy(pack_state_dict_numpy(sd, dtype="fp32"))
    assert out["s"].shape == ()
    assert float(out["s"]) == pytest.approx(7.0)


def test_pack_unpack_roundtrip_vs_torch_state_dict():
    """The migration-target parity: pack a torch state_dict, unpack to numpy."""

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 6)
            self.fc2 = torch.nn.Linear(6, 3)

    m = TinyMLP().eval()
    sd_torch = m.state_dict()
    blob = pack_state_dict_numpy(sd_torch, dtype="fp16")
    out = unpack_state_dict_numpy(blob)
    assert set(out) == set(sd_torch.keys())
    for k, v in sd_torch.items():
        ref = v.detach().cpu().to(torch.float16).numpy()
        np.testing.assert_array_equal(out[k], ref)
        assert out[k].shape == tuple(v.shape)


def test_pack_accepts_mlx_like_array_interface():
    """A non-torch, non-numpy array exposing __array__ packs via np.asarray."""

    class FakeMlxArray:
        def __init__(self, data):
            self._d = np.asarray(data, dtype=np.float32)

        def __array__(self, dtype=None):
            return self._d if dtype is None else self._d.astype(dtype)

    sd = {"w": FakeMlxArray([[1.0, 2.0]])}
    out = unpack_state_dict_numpy(pack_state_dict_numpy(sd, dtype="fp32"))
    np.testing.assert_array_equal(out["w"], np.array([[1.0, 2.0]], dtype=np.float32))


# --------------------------------------------------------------------------- #
# fail-closed on malformed blobs
# --------------------------------------------------------------------------- #


def test_unpack_rejects_short_blob():
    with pytest.raises(NumpyPortableStateDictError, match="too short"):
        unpack_state_dict_numpy(b"NP")


def test_unpack_rejects_bad_magic():
    bad = struct.pack(bridge._NPSD_HEADER_FMT, b"XXXX", 1, 0, 0)
    with pytest.raises(NumpyPortableStateDictError, match="bad magic"):
        unpack_state_dict_numpy(bad)


def test_unpack_rejects_bad_version():
    bad = struct.pack(bridge._NPSD_HEADER_FMT, bridge.NPSD_MAGIC, 99, 0, 0)
    with pytest.raises(NumpyPortableStateDictError, match="unsupported schema version"):
        unpack_state_dict_numpy(bad)


def test_unpack_rejects_oversized_num_entries():
    bad = struct.pack(bridge._NPSD_HEADER_FMT, bridge.NPSD_MAGIC, 1, 0, 0xFFFFFFFF)
    with pytest.raises(NumpyPortableStateDictError, match="exceeds cap"):
        unpack_state_dict_numpy(bad)


def test_unpack_rejects_truncated_data():
    sd = {"w": np.ones((4, 4), dtype=np.float32)}
    blob = pack_state_dict_numpy(sd, dtype="fp32")
    with pytest.raises(NumpyPortableStateDictError, match=r"truncated|trailing"):
        unpack_state_dict_numpy(blob[:-8])


def test_unpack_rejects_trailing_bytes():
    sd = {"w": np.ones(2, dtype=np.float32)}
    blob = pack_state_dict_numpy(sd, dtype="fp32")
    with pytest.raises(NumpyPortableStateDictError, match="trailing"):
        unpack_state_dict_numpy(blob + b"\x00\x00")


def test_pack_rejects_unknown_dtype():
    with pytest.raises(NumpyPortableStateDictError, match="unknown dtype"):
        pack_state_dict_numpy({"a": np.zeros(1)}, dtype="bf16")


def test_pack_rejects_non_str_key():
    with pytest.raises(NumpyPortableStateDictError, match="keys must be str"):
        pack_state_dict_numpy({1: np.zeros(1, dtype=np.float32)})


def test_unpack_rejects_non_bytes():
    with pytest.raises(NumpyPortableStateDictError, match="bytes-like"):
        unpack_state_dict_numpy("not bytes")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# decode primitives vs torch reference
# --------------------------------------------------------------------------- #


def test_linear_vs_torch():
    x = np.random.randn(5, 4).astype(np.float32)
    w = np.random.randn(7, 4).astype(np.float32)
    b = np.random.randn(7).astype(np.float32)
    got = linear(x, w, b)
    ref = F.linear(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b)).numpy()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


def test_conv2d_numpy_vs_torch():
    # NHWC numpy vs NCHW torch (transpose at the boundary)
    x = np.random.randn(1, 6, 6, 3).astype(np.float32)
    w = np.random.randn(5, 3, 3, 3).astype(np.float32)  # (C_out, kH, kW, C_in)
    b = np.random.randn(5).astype(np.float32)
    got = conv2d_numpy(x, w, b, padding=1)  # (1, 6, 6, 5)
    x_nchw = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())
    w_nchw = torch.from_numpy(np.transpose(w, (0, 3, 1, 2)).copy())
    ref = F.conv2d(x_nchw, w_nchw, torch.from_numpy(b), padding=1)
    ref_nhwc = np.transpose(ref.numpy(), (0, 2, 3, 1))
    np.testing.assert_allclose(got, ref_nhwc, rtol=0, atol=1e-4)


def test_bilinear_resize_nhwc_vs_torch_align_false():
    x = np.random.randn(1, 4, 5, 3).astype(np.float32)
    got = bilinear_resize_nhwc(x, target_h=9, target_w=11, align_corners=False)
    x_nchw = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())
    ref = F.interpolate(x_nchw, size=(9, 11), mode="bilinear", align_corners=False)
    ref_nhwc = np.transpose(ref.numpy(), (0, 2, 3, 1))
    np.testing.assert_allclose(got, ref_nhwc, rtol=0, atol=1e-5)


def test_bilinear_resize_nhwc_vs_torch_align_true():
    x = np.random.randn(1, 4, 5, 2).astype(np.float32)
    got = bilinear_resize_nhwc(x, target_h=8, target_w=10, align_corners=True)
    x_nchw = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)).copy())
    ref = F.interpolate(x_nchw, size=(8, 10), mode="bilinear", align_corners=True)
    ref_nhwc = np.transpose(ref.numpy(), (0, 2, 3, 1))
    np.testing.assert_allclose(got, ref_nhwc, rtol=0, atol=1e-5)


def test_pixel_shuffle_2x_nhwc_vs_torch():
    # torch PixelShuffle is NCHW: (N, C*4, H, W) -> (N, C, 2H, 2W)
    n, c_out, h, w = 1, 3, 4, 5
    x_nchw = torch.randn(n, c_out * 4, h, w)
    ref_nchw = F.pixel_shuffle(x_nchw, 2)  # (1, 3, 8, 10)
    ref_nhwc = np.transpose(ref_nchw.numpy(), (0, 2, 3, 1))
    # our NHWC input is the NCHW input transposed channels-last
    x_nhwc = np.transpose(x_nchw.numpy(), (0, 2, 3, 1)).copy()  # (1, 4, 5, 12)
    got = pixel_shuffle_2x_nhwc(x_nhwc)  # (1, 8, 10, 3)
    np.testing.assert_allclose(got, ref_nhwc, rtol=0, atol=1e-6)


def test_pixel_shuffle_rejects_bad_channels():
    with pytest.raises(ValueError, match="not divisible"):
        pixel_shuffle_2x_nhwc(np.zeros((1, 2, 2, 6), dtype=np.float32))


def test_sigmoid_vs_torch():
    x = np.linspace(-40, 40, 81).astype(np.float32)
    np.testing.assert_allclose(sigmoid(x), torch.sigmoid(torch.from_numpy(x)).numpy(), atol=1e-6)


def test_tanh_vs_torch():
    x = np.linspace(-10, 10, 51).astype(np.float32)
    np.testing.assert_allclose(tanh(x), torch.tanh(torch.from_numpy(x)).numpy(), atol=1e-6)


def test_film_modulate_numpy():
    h = np.random.randn(20, 8).astype(np.float32)
    g = np.random.randn(8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    got = film_modulate_numpy(h, g, b)
    np.testing.assert_allclose(got, g[None, :] * h + b[None, :], atol=1e-6)


def test_gru_cell_numpy_vs_torch():
    input_size, hidden_size = 5, 7
    cell = torch.nn.GRUCell(input_size, hidden_size)
    cell.eval()
    x = torch.randn(3, input_size)
    h0 = torch.randn(3, hidden_size)
    with torch.no_grad():
        ref = cell(x, h0).numpy()
    sd = cell.state_dict()
    got = gru_cell_numpy(
        x.numpy(),
        h0.numpy(),
        weight_ih=sd["weight_ih"].numpy(),
        weight_hh=sd["weight_hh"].numpy(),
        bias_ih=sd["bias_ih"].numpy(),
        bias_hh=sd["bias_hh"].numpy(),
    )
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


def test_gru_cell_rejects_width_mismatch():
    with pytest.raises(ValueError, match="gate width"):
        gru_cell_numpy(
            np.zeros((1, 5), dtype=np.float32),
            np.zeros((1, 7), dtype=np.float32),
            weight_ih=np.zeros((10, 5), dtype=np.float32),  # wrong: should be 21
            weight_hh=np.zeros((21, 7), dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# numpy raw-output writer
# --------------------------------------------------------------------------- #


def test_write_rgb_pair_to_raw_numpy(tmp_path):
    rgb_0 = np.random.rand(1, bridge.CAMERA_HW[0], bridge.CAMERA_HW[1], 3).astype(np.float32)
    rgb_1 = np.random.rand(1, bridge.CAMERA_HW[0], bridge.CAMERA_HW[1], 3).astype(np.float32)
    out = tmp_path / "0.raw"
    with out.open("wb") as fh:
        n = bridge.write_rgb_pair_to_raw_numpy(fh, rgb_0, rgb_1, input_range="unit")
    assert n == 2
    expected = 2 * bridge.CAMERA_HW[0] * bridge.CAMERA_HW[1] * 3
    assert out.stat().st_size == expected


def test_write_rgb_pair_resizes_to_camera_hw(tmp_path):
    rgb_0 = np.random.rand(1, 20, 30, 3).astype(np.float32)
    rgb_1 = np.random.rand(1, 20, 30, 3).astype(np.float32)
    out = tmp_path / "0.raw"
    with out.open("wb") as fh:
        bridge.write_rgb_pair_to_raw_numpy(fh, rgb_0, rgb_1)
    assert out.stat().st_size == 2 * bridge.CAMERA_HW[0] * bridge.CAMERA_HW[1] * 3


def test_write_rgb_pair_rejects_bad_shape(tmp_path):
    out = tmp_path / "0.raw"
    with out.open("wb") as fh, pytest.raises(ValueError, match="NHWC"):
        bridge.write_rgb_pair_to_raw_numpy(fh, np.zeros((1, 3, 4, 4)), np.zeros((1, 3, 4, 4)))


# --------------------------------------------------------------------------- #
# AST portability verifier
# --------------------------------------------------------------------------- #


def test_find_forbidden_imports_detects_torch():
    src = "import torch\nimport numpy as np\n"
    hits = find_forbidden_framework_imports(src)
    assert hits == [(1, "torch")]


def test_find_forbidden_imports_detects_from_import():
    src = "from mlx import core\n"
    hits = find_forbidden_framework_imports(src)
    assert hits == [(1, "mlx")]


def test_find_forbidden_imports_detects_submodule():
    src = "import torch.nn.functional as F\n"
    assert find_forbidden_framework_imports(src) == [(1, "torch")]


def test_find_forbidden_imports_ignores_relative():
    src = "from . import archive\nfrom .archive import parse\n"
    assert find_forbidden_framework_imports(src) == []


def test_find_forbidden_imports_clean_numpy_source():
    src = "import numpy as np\nimport brotli\nfrom pathlib import Path\n"
    assert find_forbidden_framework_imports(src) == []


def test_find_forbidden_imports_does_not_false_positive_on_substring():
    # 'torchvision_helper' is not 'torch'; only the root package matters
    src = "import torchvision_helper\n"
    # 'torchvision_helper'.split('.')[0] == 'torchvision_helper' which is NOT in
    # the forbidden set -> no hit
    assert find_forbidden_framework_imports(src) == []


def test_assert_portable_passes_on_clean_file(tmp_path):
    f = tmp_path / "inflate.py"
    f.write_text("import numpy as np\nfrom PIL import Image\n", encoding="utf-8")
    assert_inflate_is_numpy_portable(f)  # no raise


def test_assert_portable_raises_on_torch(tmp_path):
    f = tmp_path / "inflate.py"
    f.write_text("import torch\nimport numpy as np\n", encoding="utf-8")
    with pytest.raises(InflateNotNumpyPortableError, match="forbidden framework"):
        assert_inflate_is_numpy_portable(f)


def test_assert_portable_message_names_line_and_framework(tmp_path):
    f = tmp_path / "inflate.py"
    f.write_text("import numpy\nimport torch\n", encoding="utf-8")
    with pytest.raises(InflateNotNumpyPortableError, match=r"line 2.*torch"):
        assert_inflate_is_numpy_portable(f)


def test_forbidden_frameworks_set():
    assert "torch" in FORBIDDEN_INFLATE_FRAMEWORKS
    assert "mlx" in FORBIDDEN_INFLATE_FRAMEWORKS


# --------------------------------------------------------------------------- #
# the bridge module itself is numpy-portable
# --------------------------------------------------------------------------- #


def test_bridge_module_is_framework_free():
    bridge_path = REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "numpy_portable_inflate.py"
    # The bridge imports from pr95_hnerv_numpy_reference (numpy-only) and stdlib;
    # it must not import torch/mlx itself.
    assert_inflate_is_numpy_portable(bridge_path)


def test_numpy_reference_is_framework_free():
    ref_path = REPO_ROOT / "src" / "tac" / "local_acceleration" / "pr95_hnerv_numpy_reference.py"
    assert_inflate_is_numpy_portable(ref_path)


def test_coin_plus_plus_inflate_is_framework_free():
    """Regression: the proof-of-pattern inflate stays numpy-portable."""
    p = REPO_ROOT / "src" / "tac" / "substrates" / "coin_plus_plus" / "inflate.py"
    assert_inflate_is_numpy_portable(p)


# --------------------------------------------------------------------------- #
# runtime emitter produces a torch-free tree
# --------------------------------------------------------------------------- #


def _make_portable_fixture_substrate(repo_root: Path, pkg_name: str) -> None:
    """Create a minimal fully-numpy-portable fixture substrate under repo_root.

    Both archive.py and inflate.py are torch-free (the migration TARGET state).
    Used to test the emitter's torch-free tree contract end to end. (The live
    coin_plus_plus archive.py still carries the torch-side parser mid-migration,
    so it is not yet a fully-portable tree — its inflate.py IS portable and is
    regression-guarded separately.)
    """
    sub_src = repo_root / "src" / "tac" / "substrates" / pkg_name
    sub_src.mkdir(parents=True, exist_ok=True)
    (sub_src / "archive.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        "import numpy as np\n"
        "from tac.substrates._shared.numpy_portable_inflate import unpack_state_dict_numpy\n"
        "def parse_archive_numpy(blob):\n"
        "    return unpack_state_dict_numpy(blob)\n",
        encoding="utf-8",
    )
    (sub_src / "inflate.py").write_text(
        "# SPDX-License-Identifier: MIT\n"
        "import numpy as np\n"
        "from pathlib import Path\n"
        "def inflate_one_video(archive_bytes, output_dir):\n"
        "    Path(output_dir).mkdir(parents=True, exist_ok=True)\n",
        encoding="utf-8",
    )


def test_write_numpy_portable_contest_runtime_torch_free_tree(tmp_path):
    """The emitter vendors a numpy-portable tree and self-verifies it."""
    fake_repo = tmp_path / "repo"
    # vendor the canonical bridge + primitives into the fake repo
    import shutil

    for rel in (
        "src/tac/substrates/_shared/numpy_portable_inflate.py",
        "src/tac/local_acceleration/pr95_hnerv_numpy_reference.py",
    ):
        dst = fake_repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)
    _make_portable_fixture_substrate(fake_repo, "portable_fixture_sub")

    sub = tmp_path / "submission"
    write_numpy_portable_contest_runtime(
        sub,
        substrate_pkg_name="portable_fixture_sub",
        repo_root=fake_repo,
    )
    # inflate.sh + inflate.py present
    assert (sub / "inflate.sh").is_file()
    assert (sub / "inflate.py").is_file()
    # vendored substrate modules
    pkg = sub / "src" / "tac" / "substrates" / "portable_fixture_sub"
    assert (pkg / "archive.py").is_file()
    assert (pkg / "inflate.py").is_file()
    # vendored bridge + primitives
    assert (sub / "src" / "tac" / "substrates" / "_shared" / "numpy_portable_inflate.py").is_file()
    assert (sub / "src" / "tac" / "local_acceleration" / "pr95_hnerv_numpy_reference.py").is_file()
    # NO file in the emitted tree imports a forbidden framework
    for py in sub.rglob("*.py"):
        assert_inflate_is_numpy_portable(py)
    # inflate.sh has the 3-arg contract + set -euo pipefail
    sh = (sub / "inflate.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in sh
    assert '"$1"' in sh and '"$2"' in sh and '"$3"' in sh


def test_emitter_self_verify_raises_on_torch_substrate(tmp_path, monkeypatch):
    """If a vendored substrate inflate carries torch, verify_portable raises."""
    # Create a fake substrate with a torch-importing inflate.py under a tmp repo.
    fake_repo = tmp_path / "repo"
    sub_src = fake_repo / "src" / "tac" / "substrates" / "fake_torch_sub"
    sub_src.mkdir(parents=True)
    (sub_src / "archive.py").write_text("import numpy as np\n", encoding="utf-8")
    (sub_src / "inflate.py").write_text(
        "import torch\n\ndef inflate_one_video(b, d):\n    pass\n", encoding="utf-8"
    )
    # Need the bridge + primitives present in the fake repo for the emitter to vendor.
    shared = fake_repo / "src" / "tac" / "substrates" / "_shared"
    accel = fake_repo / "src" / "tac" / "local_acceleration"
    shared.mkdir(parents=True)
    accel.mkdir(parents=True)
    import shutil

    shutil.copy2(
        REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "numpy_portable_inflate.py",
        shared / "numpy_portable_inflate.py",
    )
    shutil.copy2(
        REPO_ROOT / "src" / "tac" / "local_acceleration" / "pr95_hnerv_numpy_reference.py",
        accel / "pr95_hnerv_numpy_reference.py",
    )
    with pytest.raises(InflateNotNumpyPortableError):
        write_numpy_portable_contest_runtime(
            tmp_path / "out",
            substrate_pkg_name="fake_torch_sub",
            repo_root=fake_repo,
        )


def test_emitter_raises_on_missing_module(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_numpy_portable_contest_runtime(
            tmp_path / "out",
            substrate_pkg_name="does_not_exist_substrate_xyz",
            repo_root=REPO_ROOT,
        )


# --------------------------------------------------------------------------- #
# end-to-end: a tiny FiLM coord-MLP decode round-trips through the bridge
# --------------------------------------------------------------------------- #


def test_end_to_end_film_coord_mlp_parity_vs_torch():
    """Pack a tiny torch FiLM-MLP, unpack numpy, decode via bridge primitives,
    compare to the torch forward within fp16-roundtrip tolerance."""
    torch.manual_seed(0)
    in_dim, hidden, mod_dim = 3, 8, 4
    W1 = torch.randn(hidden, in_dim)
    b1 = torch.randn(hidden)
    g_proj = torch.randn(hidden, mod_dim)
    g_bias = torch.randn(hidden)
    Wo = torch.randn(3, hidden)
    bo = torch.randn(3)
    sd = {
        "fc.weight": W1,
        "fc.bias": b1,
        "gamma.weight": g_proj,
        "gamma.bias": g_bias,
        "out.weight": Wo,
        "out.bias": bo,
    }
    coords = torch.randn(50, in_dim)
    mod = torch.randn(mod_dim)

    # torch forward (fp32 oracle)
    with torch.no_grad():
        gamma = F.linear(mod, g_proj, g_bias)
        h = torch.sin(gamma[None, :] * F.linear(coords, W1, b1))
        ref = torch.sigmoid(F.linear(h, Wo, bo)).numpy()

    # bridge forward (fp16-stored weights -> numpy decode)
    blob = pack_state_dict_numpy(sd, dtype="fp16")
    nsd = unpack_state_dict_numpy(blob)
    g = linear(mod.numpy(), nsd["gamma.weight"], nsd["gamma.bias"])
    hh = np.sin(film_modulate_numpy(linear(coords.numpy(), nsd["fc.weight"], nsd["fc.bias"]), g, 0.0))
    got = sigmoid(linear(hh, nsd["out.weight"], nsd["out.bias"]))

    # fp16 storage -> ~1e-2 tolerance (the parity bound the audit documents)
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-2)


def test_decode_primitives_registry_complete():
    expected = {
        "to_float32", "linear", "conv2d_nhwc", "conv2d_numpy",
        "bilinear_upsample_2x_nhwc", "bilinear_resize_nhwc",
        "pixel_shuffle_2x_nhwc", "film_modulate_numpy", "gru_cell_numpy",
        "sigmoid", "sin", "tanh", "relu", "gelu", "mean", "kahan_mean",
    }
    assert set(bridge.DECODE_PRIMITIVES) == expected
    for name, fn in bridge.DECODE_PRIMITIVES.items():
        assert callable(fn), name
