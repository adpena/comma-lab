from __future__ import annotations

import math
import random
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import ddm_rj2_joint_renderer_object_change as rj2


def test_read_stored_npz_pair_reads_only_requested_row(tmp_path: Path) -> None:
    values = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    archive = tmp_path / "rows.npz"
    np.savez(archive, values=values)

    observed = rj2.read_stored_npz_pair(
        archive,
        "values",
        2,
        expected_shape=values.shape,
        expected_dtype=values.dtype,
    )

    assert np.array_equal(observed, values[2])


def test_read_stored_npz_pair_rejects_compressed_member(tmp_path: Path) -> None:
    values = np.arange(8, dtype=np.int64).reshape(4, 2)
    archive = tmp_path / "compressed.npz"
    np.savez_compressed(archive, values=values)

    with pytest.raises(rj2.RJ2Error, match="not ZIP_STORED"):
        rj2.read_stored_npz_pair(
            archive,
            "values",
            0,
            expected_shape=values.shape,
            expected_dtype=values.dtype,
        )

    with zipfile.ZipFile(archive) as bundle:
        assert bundle.getinfo("values.npy").compress_type != zipfile.ZIP_STORED


def test_contest_arithmetic_matches_closed_form() -> None:
    observed = rj2.contest_arithmetic(
        d_seg=0.001,
        d_pose=0.0004,
        archive_bytes=150_000,
    )

    assert observed["seg_s"] == pytest.approx(0.1)
    assert observed["pose_s"] == pytest.approx(math.sqrt(0.004))
    assert observed["rate_s"] == pytest.approx(25.0 * 150_000 / rj2.RATE_DENOMINATOR)
    assert observed["s"] == pytest.approx(observed["seg_s"] + observed["pose_s"] + observed["rate_s"])


def test_ema_update_uses_shadow_and_rejects_key_drift() -> None:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(4.0)
        model.bias.fill_(2.0)
    shadow = {
        "weight": torch.zeros_like(model.weight),
        "bias": torch.zeros_like(model.bias),
    }

    observed = rj2.ema_update(shadow, model, 0.75)

    assert torch.equal(observed["weight"], torch.ones_like(model.weight))
    assert torch.equal(observed["bias"], torch.full_like(model.bias, 0.5))
    with pytest.raises(rj2.RJ2Error, match="keys differ"):
        rj2.ema_update({"weight": shadow["weight"]}, model, 0.75)


def test_atomic_torch_once_preserves_immutable_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "stage_10.pt"
    payload = {"step": 1, "weights": torch.tensor([1.0, 2.0])}

    first = rj2.atomic_torch_once(path, payload)
    repeat = rj2.atomic_torch_once(path, payload)

    assert first == repeat
    with pytest.raises(rj2.RJ2Error, match="refusing to overwrite differing retained payload"):
        rj2.atomic_torch_once(
            path,
            {"step": 2, "weights": torch.tensor([1.0, 2.0])},
        )


def test_validate_retained_records_rejects_escape_and_drift(tmp_path: Path) -> None:
    custody = tmp_path / "custody"
    retained = rj2.atomic_bytes(custody / "payload.bin", b"payload")

    assert rj2.validate_retained_records({"nested": [retained]}, allowed_root=custody) == 1
    with pytest.raises(rj2.RJ2Error, match="escaped custody root"):
        rj2.validate_retained_records(retained, allowed_root=tmp_path / "other")

    (custody / "payload.bin").write_bytes(b"changed")
    with pytest.raises(rj2.RJ2Error, match="drifted"):
        rj2.validate_retained_records(retained, allowed_root=custody)


def test_object_repricing_is_complete_and_typed() -> None:
    rows = {row["leg"]: row for row in rj2.object_repricing_rows()}

    assert set(rows) == {
        "QS2",
        "RE1",
        "EC1",
        "LD1",
        "AE1",
        "OE1",
        "HPAC sharp-optimum rows",
    }
    assert {rows[name]["disposition"] for name in ("QS2", "RE1", "EC1", "LD1")} == {"RE-PRICED"}
    assert {rows[name]["disposition"] for name in ("AE1", "OE1", "HPAC sharp-optimum rows")} == {"UNCHANGED"}


def test_restore_rng_state_replays_all_three_generators() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    checkpoint = {
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
    }
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    rj2.restore_rng_state(checkpoint)

    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == expected


def test_prepare_runtime_copy_excludes_source_archive_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "runtime.py").write_text("value = 1\n", encoding="utf-8")
    (source / "archive.zip").write_bytes(b"source")
    destination = tmp_path / "candidate"

    first = rj2.prepare_runtime_copy(source, destination)
    assert first["resumed"] is False
    assert (destination / "runtime.py").is_file()
    assert not (destination / "archive.zip").exists()

    (destination / "archive.zip").write_bytes(b"candidate")
    second = rj2.prepare_runtime_copy(source, destination)
    assert second["resumed"] is True
    assert second["archive_present_before_patch"] is True


def test_clean_trivial_runtime_residue_is_narrow(tmp_path: Path) -> None:
    cache = tmp_path / "runtime/__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.pyc").write_bytes(b"cache")
    (tmp_path / "runtime/._module.py").write_bytes(b"metadata")
    source = tmp_path / "runtime/module.py"
    source.write_text("value = 1\n", encoding="utf-8")

    rj2.clean_trivial_runtime_residue(tmp_path / "runtime")

    assert source.is_file()
    assert not cache.exists()
    assert not (tmp_path / "runtime/._module.py").exists()
