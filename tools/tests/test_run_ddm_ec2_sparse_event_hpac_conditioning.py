from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/run_ddm_ec2_sparse_event_hpac_conditioning.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ddm_ec2_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coordinate_coders_roundtrip_exact_and_brotli_wins_fixture() -> None:
    module = load_module()
    indices = np.array([7, 8, 512, 384 * 512 + 19, module.PIXELS - 1], dtype=np.int64)
    canonical = module.encode_coordinate_canonical(indices)
    np.testing.assert_array_equal(module.decode_coordinate_canonical(canonical), indices)
    rows = {}
    for coder in ("raw", "brotli_q11", "lzma_xz"):
        payload = module.frame_coordinate_payload(canonical, coder)
        rows[coder] = payload
        np.testing.assert_array_equal(module.unframe_coordinate_payload(payload), indices)
        assert payload == module.frame_coordinate_payload(canonical, coder)
    assert len(rows["brotli_q11"]) < len(rows["lzma_xz"])


def test_coordinate_codec_rejects_duplicate_and_trailing_sites() -> None:
    module = load_module()
    with pytest.raises(module.EC2Error, match="strictly increasing"):
        module.encode_coordinate_canonical(np.array([5, 5], dtype=np.int64))
    canonical = module.encode_coordinate_canonical(np.array([5, 9], dtype=np.int64))
    with pytest.raises(module.EC2Error, match="trailing"):
        module.decode_coordinate_canonical(canonical + b"x")


def test_complete_container_counts_framing_and_has_strict_falsifier() -> None:
    module = load_module()
    model = b"model"
    tokens = b"tokens"
    coordinates = b"coordinates"
    package = module.build_package(model, tokens, coordinates)
    assert len(package) == module.PACKAGE_HEADER.size + len(model) + len(tokens) + len(coordinates)
    assert module.parse_package(package) == (model, tokens, coordinates)
    assert module.package_admission_passes(116_715) is True
    assert module.package_admission_passes(116_716) is False


def test_sparse_event_channel_is_local_consumed_and_self_compressed() -> None:
    module = load_module()
    xi1 = module.load_xi1()
    integer, compression, packer, _ = xi1.configure_hpac()
    model = module.make_model(
        integer,
        compression,
        torch.device("cpu"),
        self_compressed=True,
        initialize=False,
    ).eval()
    with torch.no_grad():
        model.conv_event.weight.zero_()
        model.conv_event.weight[:, 0, 1, 1] = 8
        model.conv_event.bias.zero_()
        model.conv_event.exponent.fill_(-3)
    compression.set_deployed_bit_depths(model, True)
    previous = torch.zeros((1, 64, 64), dtype=torch.long)
    event_off = torch.zeros((1, 64, 64), dtype=torch.uint8)
    event_on = event_off.clone()
    event_on[0, 17, 23] = 1
    idx = torch.tensor([0])
    past_off = model.prepare_frame_context(idx, previous, event_off)[1]
    past_on = model.prepare_frame_context(idx, previous, event_on)[1]
    assert torch.count_nonzero(past_on - past_off).item() > 0

    raw = packer.serialize_self_compressed(model)
    restored = module.make_model(
        integer,
        compression,
        torch.device("cpu"),
        self_compressed=False,
        initialize=False,
    ).eval()
    packer.deserialize_self_compressed(restored, raw)
    current = torch.zeros_like(previous)
    expected = model(current, idx, previous, event_on)
    actual = restored(current, idx, previous, event_on)
    assert torch.equal(expected, actual)


def test_fire_command_uses_measured_peak_margin_and_resume_auto() -> None:
    module = load_module()
    command = module.pinned_fire_command()
    assert "tools/safe_run.py --rss-mb 6144 --projected-gib 6 --timeout 7200" in command
    assert "--leg all --resume-from auto" in command
    assert str(module.OUTPUT / "run/main.safe_run.json") in command


def test_cleanup_certifies_keep_without_deleting_payloads(tmp_path: Path) -> None:
    module = load_module()
    module.RETAINED = tmp_path
    payload = tmp_path / "serialized/terminal.package.ec2pkg"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"retained-real-payload")
    result = module.cleanup_stage()
    assert result["status"] == "CERTIFIED_KEEP_NO_SIGNAL_LOSS"
    assert result["deleted"] == []
    assert payload.read_bytes() == b"retained-real-payload"
    assert result["files"][0]["action"] == "KEEP"
