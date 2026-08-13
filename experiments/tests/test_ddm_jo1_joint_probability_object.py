from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from experiments import ddm_jo1_joint_probability_object as jo1
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_deterministic_zip_is_stored_and_repeatable() -> None:
    first = jo1.deterministic_zip(b"joint-probability-object")
    second = jo1.deterministic_zip(b"joint-probability-object")
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["p"]
        assert archive.getinfo("p").compress_type == zipfile.ZIP_STORED
        assert archive.read("p") == b"joint-probability-object"


def test_contour_bending_energy_is_finite_and_nonnegative() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    energy = jo1.contour_bending_energy(mask)
    assert np.isfinite(energy)
    assert energy > 0.0


def test_bending_delta_enforces_source_and_retains_class_breakdown() -> None:
    frame = np.zeros((jo1.HEIGHT, jo1.WIDTH), dtype=np.uint8)
    frame[100:120, 100:120] = 1
    index = np.asarray([110 * jo1.WIDTH + 119], dtype=np.uint32)
    result = jo1.bending_delta(frame, 1, 0, index)
    assert np.isfinite(result["delta"])
    assert set(result["by_class"]) == {"0", "1"}


def test_live_input_crosswalk_has_required_disjoint_terminal_counts() -> None:
    store, rows, compose, trials = jo1.load_inputs()
    assert len(store.rows) == len(rows) == 200
    assert len(compose["selected_ids"]) == 44
    assert sum(row["reason"] == "POSE_GATE" for row in trials) == 20
    assert sum(row["reason"] == "NO_MARGINAL_ROBUST_GAIN" for row in trials) == 1


def test_jo1_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_jo1_joint_probability_object.py",
            "experiments/tests/test_ddm_jo1_joint_probability_object.py",
        ),
    )
    assert findings == []
