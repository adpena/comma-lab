# SPDX-License-Identifier: MIT
"""Tests for the ddm_dr1 receipt/retention additions to the delta_R tool.

The tool's MEASUREMENT is ALREADY-SETTLED (SPEC_v75 §8B) and must not change.
These tests pin exactly that: the new ``--receipts-out`` / ``--retain-dir`` /
``--threads`` flags are additive, the ``--out`` JSON is identical whether or not
they are passed, and every receipt row is a read-only view of the same arrays.

The strongest invariant here is the PAYLOAD-COMPLETENESS one: delta_R must be
recomputable, bit-for-bit, from the retained m0/m1 arrays alone (ALWAYS KEEP THE
PAYLOAD — a retained payload that cannot reproduce the headline is not a payload).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "measure_delta_R_noise_floor.py"

CAM_HW = (874, 1164)
SEG_HW = (384, 512)


def _load_tool():
    spec = importlib.util.spec_from_file_location("_dr1_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


class _FakeSegNet:
    """Deterministic stand-in for the frozen SegNet.

    Real SegNet is far too heavy for a unit test, and loading it would make the
    test an integration test of upstream weights rather than of these flags.
    The only property the receipts code depends on is that logits are a
    deterministic function of the input, so uint8 rounding perturbs the margin.
    """

    def __init__(self):
        import torch

        g = torch.Generator().manual_seed(20260904)
        self.w = torch.randn(5, 3, generator=g)
        self.b = torch.linspace(-0.5, 0.5, 5)

    def __call__(self, x):  # x: (1,3,H,W)
        import torch

        out = torch.einsum("cr,brhw->bchw", self.w, x / 255.0)
        return out + self.b.view(1, 5, 1, 1) + torch.sin(out * 3.0)


def _write_synthetic_npz(path: Path, n: int = 2) -> None:
    rng = np.random.default_rng(20260904)
    gt_f1 = rng.integers(0, 256, size=(n, *CAM_HW, 3), dtype=np.uint8)
    # margins spread across the band edge so the annulus is a proper subset
    margins = rng.uniform(-2.0, 2.0, size=(n, *SEG_HW)).astype(np.float32)
    lstars = rng.integers(0, 5, size=(n, *SEG_HW)).astype(np.int64)
    np.savez(
        path,
        n_pairs=np.int64(n),
        gt_f0=gt_f1,
        gt_f1=gt_f1,
        lstars=lstars,
        margins=margins,
        gt_poses=np.zeros((n, 6), dtype=np.float64),
    )


@pytest.fixture(scope="module")
def synthetic_npz(tmp_path_factory):
    path = tmp_path_factory.mktemp("dr1_gt") / "gt_synth.npz"
    _write_synthetic_npz(path)
    return path


def _run(tool, npz, out_dir: Path, *, receipts=False, retain=False):
    argv = [
        "--gt-npz", str(npz),
        "--n", "2",
        "--band", "1.0",
        "--threads", "1",
        "--out", str(out_dir / "out.json"),
    ]
    if receipts:
        argv += ["--receipts-out", str(out_dir / "receipts.json")]
    if retain:
        argv += ["--retain-dir", str(out_dir / "retain")]
    assert tool.main(argv) == 0
    return json.loads((out_dir / "out.json").read_text())


# ---------------------------------------------------------------- pure helpers


def test_quantile_summary_matches_numpy(tool):
    a = np.linspace(0.0, 1.0, 1001)
    got = tool._quantile_summary(a)
    assert got["mean"] == float(np.mean(a))
    assert got["p50"] == float(np.quantile(a, 0.50))
    assert got["p90"] == float(np.quantile(a, 0.90))
    assert got["p95"] == float(np.quantile(a, 0.95))
    assert got["p99"] == float(np.quantile(a, 0.99))
    assert got["max"] == float(np.max(a))


def test_sha256_file_matches_hashlib(tool, tmp_path):
    import hashlib

    p = tmp_path / "blob.bin"
    payload = b"delta_R" * 5000
    p.write_bytes(payload)
    assert tool._sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_class_names_are_the_canonical_comma10k_order(tool):
    # MEASURED 2026-06-27 (CLAUDE.md). The luma-sorted order is FORBIDDEN and
    # has bitten this repo three times; pin the canonical one.
    assert tool.SEG_CLASS_NAMES == (
        "Road", "Lane", "Undrivable", "Movable", "MyCar"
    )


def test_sub_band_divisors_include_the_identity(tool):
    # 1.0 must be present: it is what makes band_<band> equal delta_R exactly,
    # which is the receipts' self-consistency check.
    assert 1.0 in tool.RECEIPT_SUB_BAND_DIVISORS
    assert all(d >= 1.0 for d in tool.RECEIPT_SUB_BAND_DIVISORS)


# ------------------------------------------------------------- end-to-end runs


def test_out_json_identical_with_and_without_new_flags(tool, synthetic_npz, tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "_load_segnet_cpu", lambda upstream: _FakeSegNet())
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    rich_dir = tmp_path / "rich"
    rich_dir.mkdir()

    plain = _run(tool, synthetic_npz, plain_dir)
    rich = _run(tool, synthetic_npz, rich_dir, receipts=True, retain=True)

    assert plain == rich, "receipt/retention flags must not perturb the measurement"
    assert not (plain_dir / "receipts.json").exists()
    assert not (plain_dir / "retain").exists()


def test_receipts_band_identity_equals_delta_R(tool, synthetic_npz, tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "_load_segnet_cpu", lambda upstream: _FakeSegNet())
    out_dir = tmp_path / "r"
    out_dir.mkdir()
    result = _run(tool, synthetic_npz, out_dir, receipts=True)
    receipts = json.loads((out_dir / "receipts.json").read_text())

    identity = receipts["sub_band_sensitivity"]["band_1"]
    assert identity["p95"] == result["delta_R"]
    assert receipts["delta_R"] == result["delta_R"]
    assert receipts["torch_num_threads"] == 1

    # nested sub-bands are strict subsets -> monotone non-increasing pixel counts
    counts = [
        receipts["sub_band_sensitivity"][f"band_{1.0 / d:g}"]["n_px"]
        for d in tool.RECEIPT_SUB_BAND_DIVISORS
    ]
    assert counts == sorted(counts, reverse=True)


def test_per_class_annulus_partitions_the_annulus(tool, synthetic_npz, tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "_load_segnet_cpu", lambda upstream: _FakeSegNet())
    out_dir = tmp_path / "c"
    out_dir.mkdir()
    _run(tool, synthetic_npz, out_dir, receipts=True)
    receipts = json.loads((out_dir / "receipts.json").read_text())

    pooled = receipts["per_class_annulus_pooled"]
    assert set(pooled) == set(tool.SEG_CLASS_NAMES)
    class_total = sum(v["n_px"] for v in pooled.values())
    frame_total = sum(row["n_annulus_px"] for row in receipts["per_frame"])
    identity_total = receipts["sub_band_sensitivity"]["band_1"]["n_px"]
    assert class_total == frame_total == identity_total

    assert len(receipts["per_frame"]) == receipts["n_frames"] == 2
    for row in receipts["per_frame"]:
        assert set(row["per_class_annulus"]) == set(tool.SEG_CLASS_NAMES)


def test_retained_payload_reproduces_delta_R_exactly(tool, synthetic_npz, tmp_path, monkeypatch):
    """ALWAYS KEEP THE PAYLOAD: the retained arrays must regenerate the headline."""
    monkeypatch.setattr(tool, "_load_segnet_cpu", lambda upstream: _FakeSegNet())
    out_dir = tmp_path / "p"
    out_dir.mkdir()
    result = _run(tool, synthetic_npz, out_dir, receipts=True, retain=True)
    receipts = json.loads((out_dir / "receipts.json").read_text())

    retained = receipts["retained_payloads"]
    assert set(retained) == {"m0_no_uint8.npy", "m1_with_uint8.npy"}
    for name, meta in retained.items():
        path = out_dir / "retain" / name
        assert meta["bytes"] == path.stat().st_size
        assert meta["sha256"] == tool._sha256_file(path)

    m0 = np.load(out_dir / "retain" / "m0_no_uint8.npy")
    m1 = np.load(out_dir / "retain" / "m1_with_uint8.npy")
    z = np.load(synthetic_npz)
    annulus = np.abs(z["margins"][: m0.shape[0]]) < 1.0
    replayed = float(np.quantile(np.abs(m1 - m0)[annulus], 0.95))
    assert replayed == result["delta_R"]


def test_receipts_omit_per_class_when_cache_lacks_lstars(tool, tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "_load_segnet_cpu", lambda upstream: _FakeSegNet())
    rng = np.random.default_rng(7)
    npz = tmp_path / "no_lstars.npz"
    np.savez(
        npz,
        gt_f1=rng.integers(0, 256, size=(1, *CAM_HW, 3), dtype=np.uint8),
        margins=rng.uniform(-2.0, 2.0, size=(1, *SEG_HW)).astype(np.float32),
    )
    out_dir = tmp_path / "n"
    out_dir.mkdir()
    assert tool.main([
        "--gt-npz", str(npz), "--n", "1", "--band", "1.0",
        "--out", str(out_dir / "out.json"),
        "--receipts-out", str(out_dir / "receipts.json"),
    ]) == 0
    receipts = json.loads((out_dir / "receipts.json").read_text())
    assert receipts["per_class_annulus_pooled"] is None
    assert receipts["per_frame"][0]["per_class_annulus"] is None
    assert receipts["retained_payloads"] is None


# --- ddm_bh1: the PRODUCER's defaults are the population, never the retired prefix ----------
#
# This tool produced the retired delta_R = 0.019590163230895963 on the n96 prefix; ddm_dr1
# measured 0.021881818771362305 at n600 (the prefix was 11.70% LOW) and ddm_ql2/ql3 found live
# harnesses still deciding R-safety with the retired value.  The consumers were cured; the
# PRODUCER's defaults were not, so a flagless re-run silently regenerated the retired constant.
# These pin the cure at the producer.


class _ParserCaptured(Exception):
    def __init__(self, parser):
        super().__init__("parser captured")
        self.parser = parser


def _defaults(tool, monkeypatch) -> dict:
    """Capture the tool's real argparse defaults without running the measurement."""

    import argparse

    def _fake_parse_args(self, argv=None, namespace=None):
        raise _ParserCaptured(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _fake_parse_args)
    try:
        tool.main([])
    except _ParserCaptured as exc:
        return {
            action.option_strings[0]: action.default
            for action in exc.parser._actions
            if action.option_strings
        }
    raise AssertionError("the tool did not build an argparse parser")


def test_default_frame_count_is_the_n600_population(tool, monkeypatch):
    assert _defaults(tool, monkeypatch)["--n"] == 600


def test_default_gt_cache_is_the_n600_population(tool, monkeypatch):
    default = _defaults(tool, monkeypatch)["--gt-npz"]
    assert default.endswith("gt_n600.npz"), default
    assert "n96" not in default


def test_no_retired_delta_r_literal_in_the_producer(tool):
    source = Path(tool.__file__).read_text(encoding="utf-8")
    for retired in ("0.019590163230895963", "0.039180326461791926"):
        for line in source.splitlines():
            if retired in line:
                # only admissible inside the prose that explains WHY it is retired
                assert "retired" in line.lower(), f"live retired literal: {line.strip()}"
