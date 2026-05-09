"""Tests for the A1 CUDA-CPU drift discriminator variant builder."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_a1_cuda_cpu_drift_discriminator_variants.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "a1_cuda_cpu_drift_discriminator_tool", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Minimal inflate.py template that contains every anchor line the mutators
# look for. Keeps the test independent from whether the A1 forensic clone is
# present in this checkout.
_TEMPLATE = "\n".join(
    [
        "def inflate(src_bin: str, dst_raw: str):",
        "    archive_bytes = open(src_bin, 'rb').read()",
        "    decoder_sd, latents = parse_a1_finetuned_archive(archive_bytes)",
        "",
        '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
        "    decoder = HNeRVDecoder(",
        "        latent_dim=LATENT_DIM,",
        "        base_channels=BASE_CHANNELS,",
        "        eval_size=(EVAL_H, EVAL_W),",
        "    ).to(device)",
        "    decoder.load_state_dict(decoder_sd)",
        "    decoder.eval()",
        "",
        "    latents = latents.to(device)",
        '    with torch.inference_mode(), open(dst_raw, "wb") as fout:',
        "        for i in range(0, N_PAIRS, 16):",
        "            j = min(i + 16, N_PAIRS)",
        "            batch = j - i",
        "            decoded = decoder(latents[i:j])",
        "            flat = decoded.reshape(batch * 2, 3, EVAL_H, EVAL_W)",
        "            up = F.interpolate(",
        "                flat, size=(CAMERA_H, CAMERA_W),",
        '                mode="bicubic", align_corners=False,',
        "            )",
        "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)",
        "            up[:, 0, 0].sub_(1.0)",
        "            up[:, 0, 2].sub_(1.0)",
        "            up[:, 1, 1].sub_(1.0)",
        "            frames = (",
        "                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)",
        "                .clamp(0, 255)",
        "                .permute(0, 2, 3, 1)",
        "                .round()",
        "                .to(torch.uint8)",
        "                .cpu()",
        "                .numpy()",
        "            )",
        "            fout.write(frames.tobytes())",
        "",
    ]
)


# ---------------------------------------------------------------------------
# Variant table sanity.
# ---------------------------------------------------------------------------


def test_variant_table_has_exactly_four_variants() -> None:
    tool = load_tool()
    ids = [v["variant_id"] for v in tool.VARIANTS]
    assert ids == [
        "v_baseline",
        "v_loader_isolated",
        "v_conv_isolated",
        "v_hydra_isolated",
    ]


def test_each_variant_declares_a_mechanism_hypothesis() -> None:
    tool = load_tool()
    expected = {
        "v_baseline": "control",
        "v_loader_isolated": "loader_byte_drift",
        "v_conv_isolated": "conv_kernel_accumulation_drift",
        "v_hydra_isolated": "hydra_head_numerical_sensitivity",
    }
    by_id = {v["variant_id"]: v for v in tool.VARIANTS}
    for vid, hypothesis in expected.items():
        assert by_id[vid]["mechanism_hypothesis"] == hypothesis


def test_every_mutator_handle_is_registered() -> None:
    tool = load_tool()
    for v in tool.VARIANTS:
        assert v["inflate_mutator"] in tool._MUTATORS, (
            f"variant {v['variant_id']!r} declares mutator "
            f"{v['inflate_mutator']!r} which is not in _MUTATORS"
        )


# ---------------------------------------------------------------------------
# Mutator-level tests.
# ---------------------------------------------------------------------------


def test_mutate_baseline_returns_template_unchanged() -> None:
    tool = load_tool()
    out = tool._mutate_baseline(_TEMPLATE)
    assert out == _TEMPLATE


def test_mutate_loader_isolated_replaces_device_line_with_force_cpu() -> None:
    tool = load_tool()
    out = tool._mutate_loader_isolated(_TEMPLATE)
    assert 'device = torch.device("cpu")  # DISCRIMINATOR' in out
    # The original cuda-fallback line must be GONE.
    assert (
        'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in out
    )


def test_mutate_conv_isolated_inserts_deterministic_flags_above_device() -> None:
    tool = load_tool()
    out = tool._mutate_conv_isolated(_TEMPLATE)
    assert "torch.use_deterministic_algorithms(True, warn_only=True)" in out
    assert "torch.backends.cudnn.deterministic = True" in out
    assert "torch.backends.cudnn.benchmark = False" in out
    # The device line must STILL be present (we don't replace, we insert).
    assert tool.ANCHOR_DEVICE in out
    # Insertion must occur ABOVE the device line.
    deterministic_idx = out.index(
        "torch.use_deterministic_algorithms(True, warn_only=True)"
    )
    device_idx = out.index(tool.ANCHOR_DEVICE)
    assert deterministic_idx < device_idx


def test_mutate_hydra_isolated_replaces_round_with_coarse_grid() -> None:
    tool = load_tool()
    out = tool._mutate_hydra_isolated(_TEMPLATE)
    # New chain replaces the bare .round() with the coarse-grid op.
    assert ".div(2.0).round().mul(2.0).clamp(0, 255)" in out
    # The bare .round() line on its own should no longer exist exactly as before.
    bare_round_lines = [
        line for line in out.splitlines() if line.strip() == ".round()"
    ]
    assert bare_round_lines == [], (
        "expected the bare '.round()' line to be replaced; found: "
        f"{bare_round_lines!r}"
    )


def test_mutator_raises_when_anchor_missing_for_loader() -> None:
    tool = load_tool()
    with pytest.raises(RuntimeError, match="ANCHOR_DEVICE"):
        tool._mutate_loader_isolated("def inflate(): pass\n")


def test_mutator_raises_when_anchor_missing_for_conv() -> None:
    tool = load_tool()
    with pytest.raises(RuntimeError, match="ANCHOR_DEVICE"):
        tool._mutate_conv_isolated("def inflate(): pass\n")


def test_mutator_raises_when_anchor_missing_for_hydra() -> None:
    tool = load_tool()
    with pytest.raises(RuntimeError, match="ANCHOR_REROUND_TAIL"):
        tool._mutate_hydra_isolated("def inflate(): pass\n")


# ---------------------------------------------------------------------------
# build_inflate_py dispatcher.
# ---------------------------------------------------------------------------


def test_build_inflate_py_dispatches_via_mutator_name() -> None:
    tool = load_tool()
    out = tool.build_inflate_py(_TEMPLATE, "_mutate_baseline")
    assert out == _TEMPLATE
    out2 = tool.build_inflate_py(_TEMPLATE, "_mutate_loader_isolated")
    assert "DISCRIMINATOR: force inflate-side CPU" in out2


def test_build_inflate_py_unknown_mutator_raises_keyerror() -> None:
    tool = load_tool()
    with pytest.raises(KeyError):
        tool.build_inflate_py(_TEMPLATE, "_mutate_does_not_exist")


# ---------------------------------------------------------------------------
# write_variant and manifest contract.
# ---------------------------------------------------------------------------


def test_manifest_path_relativizes_repo_paths_and_keeps_external_absolute(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    assert tool.manifest_path(REPO_ROOT / "tools" / "x.py") == "tools/x.py"
    assert tool.manifest_path(tmp_path / "x.py") == str(tmp_path / "x.py")


def test_write_variant_baseline_produces_sha_identical_inflate(tmp_path: Path) -> None:
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 forensic archive not present in this checkout")
    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    baseline_variant = next(v for v in tool.VARIANTS if v["variant_id"] == "v_baseline")
    manifest = tool.write_variant(
        baseline_variant,
        tmp_path,
        "TEST",
        template,
    )
    assert manifest["inflate_py_sha256_old"] == manifest["inflate_py_sha256_new"]
    assert manifest["mechanism_hypothesis"] == "control"
    assert manifest["archive_sha256"] == tool.A1_EXPECTED_ARCHIVE_SHA
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_blockers"]


def test_write_variant_loader_isolated_produces_distinct_inflate(tmp_path: Path) -> None:
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 forensic archive not present in this checkout")
    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    loader_variant = next(
        v for v in tool.VARIANTS if v["variant_id"] == "v_loader_isolated"
    )
    manifest = tool.write_variant(
        loader_variant,
        tmp_path,
        "TEST",
        template,
    )
    assert manifest["inflate_py_sha256_old"] != manifest["inflate_py_sha256_new"]
    assert manifest["mechanism_hypothesis"] == "loader_byte_drift"
    inflate_text = (
        tmp_path
        / "a1_cuda_cpu_drift_discriminator_v_loader_isolated_TEST"
        / "submission_dir"
        / "inflate.py"
    ).read_text()
    assert 'device = torch.device("cpu")' in inflate_text


def test_write_variant_refuses_when_baseline_actually_mutates(tmp_path: Path) -> None:
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 forensic archive not present in this checkout")
    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    bogus = {
        "variant_id": "v_baseline",
        "name": "bogus",
        "mechanism_hypothesis": "control",
        "isolation_spec": {"kind": "control", "modifications": []},
        "inflate_mutator": "_mutate_loader_isolated",  # WRONG mutator for v_baseline
        "rationale": "test bogus",
    }
    with pytest.raises(RuntimeError, match="byte-identical"):
        tool.write_variant(bogus, tmp_path, "BOGUS", template)


def test_write_variant_refuses_when_isolated_variant_does_not_mutate(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 forensic archive not present in this checkout")
    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    bogus = {
        "variant_id": "v_loader_isolated",
        "name": "bogus",
        "mechanism_hypothesis": "loader_byte_drift",
        "isolation_spec": {"kind": "force_inflate_cpu", "modifications": []},
        "inflate_mutator": "_mutate_baseline",  # WRONG: no mutation
        "rationale": "test bogus",
    }
    with pytest.raises(RuntimeError, match="IDENTICAL"):
        tool.write_variant(bogus, tmp_path, "BOGUS", template)


def test_write_variant_emits_dual_eval_dispatch_blockers(tmp_path: Path) -> None:
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 forensic archive not present in this checkout")
    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    variant = next(
        v for v in tool.VARIANTS if v["variant_id"] == "v_hydra_isolated"
    )
    manifest = tool.write_variant(variant, tmp_path, "TEST", template)
    blockers = manifest["dispatch_blockers"]
    assert any("CPU" in b and "CUDA" in b for b in blockers), (
        f"expected dual-eval mandate to be in dispatch_blockers; got {blockers}"
    )


def test_write_variant_records_drift_decomposition_decision_rules(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 forensic archive not present in this checkout")
    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    manifest = tool.write_variant(
        tool.VARIANTS[1], tmp_path, "TEST", template
    )
    rules = manifest["drift_decomposition_decision_rules"]
    for key in (
        "primary_mechanism_threshold",
        "multi_mechanism_threshold",
        "fourth_mechanism_threshold",
        "no_op_detector",
    ):
        assert key in rules


def test_write_variant_round_trip_serialises_manifest_as_json(tmp_path: Path) -> None:
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 forensic archive not present in this checkout")
    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    manifest = tool.write_variant(
        tool.VARIANTS[0], tmp_path, "TEST", template
    )
    out_path = (
        tmp_path
        / "a1_cuda_cpu_drift_discriminator_v_baseline_TEST"
        / "discriminator_manifest.json"
    )
    on_disk = json.loads(out_path.read_text())
    assert on_disk["lane_id"] == "lane_avvideodataset_cuda_path_mechanism_discriminator"
    assert on_disk["variant_id"] == "v_baseline"
    assert on_disk["archive_sha256"] == manifest["archive_sha256"]
    assert on_disk["a1_canonical_cpu_score_baseline"]["value"] == pytest.approx(
        0.19284757743677347
    )
