"""ddm_b2b QA75/QA80 SegNet field-pass harness tests — plumbing only, STUB scorer.

NO real SegNet, NO Metal, NO n600 pass (the scorer pass is POST-BURN). Validates the
loader -> derived chunking -> field compute -> SSD manifest+sha plumbing with the
deterministic stub so the harness is READY-TO-RUN when the real scorer is injected post-burn.
Pointer 0.1910828242 [contest-CPU] UNMOVED; advisory fields, score_claim=False.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ddm_b2b_segnet_field_pass",
    str(Path(__file__).resolve().parents[3] / "tools" / "ddm_b2b_segnet_field_pass.py"))
H = importlib.util.module_from_spec(_SPEC)
import sys  # noqa: E402

sys.modules[_SPEC.name] = H  # register BEFORE exec (dataclass/typing resolution)
_SPEC.loader.exec_module(H)  # type: ignore[union-attr]


def test_derive_chunk_size_honors_120_cap():
    assert H.derive_chunk_size(2, None) == 2
    assert H.derive_chunk_size(600, None) == 120           # capped at the charter law
    assert H.derive_chunk_size(600, 200) == 120            # request above cap -> cap
    assert H.derive_chunk_size(50, 200) == 50              # fewer pairs than cap -> n_pairs
    assert H.derive_chunk_size(0, None) == 1               # never zero


def _fixture_source(n=2):
    frames = np.random.default_rng(0).integers(
        0, 256, (n, H.SEG_H, H.SEG_W, 3), dtype=np.uint8)
    return H.FramePairSource("fixture", n, lambda i: frames[i])


def test_field_pass_both_shapes_and_manifest(tmp_path):
    src = _fixture_source(2)
    man = H.run_field_pass(src, tmp_path, H.stub_segnet_field_fn(), field_kind="both", chunk=1)
    assert man["field_kind"] == "both" and man["pair_count"] == 2 and man["chunk_size"] == 1
    assert man["score_claim"] is False
    assert len(man["pairs"]) == 2
    d = np.load(tmp_path / "pair-000000.npz")
    # QA75 distill targets
    assert d["distill_logits"].shape == (H.N_CLASSES, H.SEG_H, H.SEG_W)
    assert d["distill_logits"].dtype == np.float16
    assert d["distill_margin"].shape == (H.SEG_H, H.SEG_W)
    # QA80 exact flip-distance
    assert d["exact_flip_distance"].shape == (H.SEG_H, H.SEG_W)
    assert np.all(d["winner"] != d["runner"])              # exact field needs winner != runner
    assert np.all(np.isfinite(d["exact_flip_distance"]))


def test_field_pass_deterministic_sha(tmp_path):
    src = _fixture_source(2)
    a = H.run_field_pass(src, tmp_path / "a", H.stub_segnet_field_fn(), field_kind="both")
    b = H.run_field_pass(src, tmp_path / "b", H.stub_segnet_field_fn(), field_kind="both")
    assert [p["sha256"] for p in a["pairs"]] == [p["sha256"] for p in b["pairs"]]


def test_field_pass_kinds_emit_expected_keys(tmp_path):
    src = _fixture_source(1)
    H.run_field_pass(src, tmp_path / "d", H.stub_segnet_field_fn(), field_kind="distill_logit_margin")
    keys_d = set(np.load(tmp_path / "d" / "pair-000000.npz").files)
    assert keys_d == {"distill_logits", "distill_margin", "argmax"}
    H.run_field_pass(src, tmp_path / "e", H.stub_segnet_field_fn(), field_kind="exact_flip_distance")
    keys_e = set(np.load(tmp_path / "e" / "pair-000000.npz").files)
    assert keys_e == {"exact_flip_distance", "winner", "runner"}


def test_field_pass_multi_chunk_covers_all_pairs(tmp_path):
    src = _fixture_source(5)
    man = H.run_field_pass(src, tmp_path, H.stub_segnet_field_fn(), field_kind="exact_flip_distance",
                           chunk=2)
    assert man["chunk_size"] == 2 and man["pair_count"] == 5
    assert {p["pair_id"] for p in man["pairs"]} == set(range(5))
    for pid in range(5):
        assert (tmp_path / f"pair-{pid:06d}.npz").is_file()


def test_frame_source_rejects_bad_shape():
    src = H.FramePairSource("bad", 1, lambda i: np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        src.frame1(0)


def test_real_segnet_field_fn_refuses_mps():
    with pytest.raises(ValueError):
        H.real_segnet_field_fn(device="mps")  # authority is cpu|cuda, NEVER mps
