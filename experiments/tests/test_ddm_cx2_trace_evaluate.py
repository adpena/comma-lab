"""Tests for full-precision capture around the immutable evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "experiments" / "ddm_cx2_trace_evaluate.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("ddm_cx2_trace_test", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import CX2 evaluator wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trace_wrapper_captures_values_without_reimplementing_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wrapper = _load_wrapper()
    submission = tmp_path / "submission"
    (submission / "inflated").mkdir(parents=True)
    (submission / "archive.zip").write_bytes(b"exact-archive")
    raw = submission / "inflated" / "0.raw"
    raw.write_bytes(b"exact-raw")
    originals = tmp_path / "originals"
    originals.mkdir()
    original = originals / "0.mkv"
    original.write_bytes(b"exact-original")
    names = tmp_path / "names.txt"
    names.write_text("0.mkv\n")
    models = tmp_path / "models"
    models.mkdir()
    evaluator_dependencies = {
        "frame_utils.py": tmp_path / "frame_utils.py",
        "modules.py": tmp_path / "modules.py",
        "models/posenet.safetensors": models / "posenet.safetensors",
        "models/segnet.safetensors": models / "segnet.safetensors",
    }
    evaluator_dependencies["modules.py"].write_text("# exact modules\n")
    evaluator_dependencies["models/posenet.safetensors"].write_bytes(b"pose")
    evaluator_dependencies["models/segnet.safetensors"].write_bytes(b"seg")
    evaluator = tmp_path / "evaluate.py"
    evaluator_dependencies["frame_utils.py"].write_text(
        "SIBLING_IMPORT_OK = True\n"
    )
    evaluator.write_text(
        """
import sys
import torch
from frame_utils import SIBLING_IMPORT_OK

class AVVideoDataset:
    pass

def main():
    assert SIBLING_IMPORT_OK
    submission_dir = sys.argv[sys.argv.index('--submission-dir') + 1]
    posenet_dist = 0.00001987654321
    segnet_dist = 0.0002987654321
    compressed_size = 13
    uncompressed_size = 14
    rate = compressed_size / uncompressed_size
    batch_sizes = torch.tensor(600.0)
    device = torch.device('cpu')
    DefaultDatasetClass = AVVideoDataset
    score = 100 * segnet_dist + (10 * posenet_dist) ** 0.5 + 25 * rate
    printed_results = [f'{submission_dir}: {score:.2f}']
    print(printed_results[0])

if __name__ == '__main__':
    main()
""".lstrip()
    )
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(wrapper, "EXPECTED_EVALUATE_SHA256", _sha256(evaluator))
    monkeypatch.setattr(wrapper, "EXPECTED_VIDEO_NAMES_SHA256", _sha256(names))
    monkeypatch.setattr(wrapper, "EXPECTED_ORIGINAL_BYTES", len(b"exact-original"))
    monkeypatch.setattr(wrapper, "EXPECTED_ORIGINAL_SHA256", _sha256(original))
    monkeypatch.setattr(wrapper, "EXPECTED_RAW_BYTES", len(b"exact-raw"))
    monkeypatch.setattr(
        wrapper,
        "EXPECTED_EVALUATOR_DEPENDENCIES",
        {
            name: _sha256(path)
            for name, path in evaluator_dependencies.items()
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(WRAPPER),
            "--evaluate-py",
            str(evaluator),
            "--receipt",
            str(receipt),
            "--",
            "--submission-dir",
            str(submission),
            "--uncompressed-dir",
            str(originals),
            "--video-names-file",
            str(names),
            "--device",
            "cpu",
            "--batch-size",
            "16",
            "--seed",
            "1234",
            "--num-threads",
            "2",
            "--prefetch-queue-depth",
            "4",
            "--report",
            str(tmp_path / "report.txt"),
        ],
    )
    wrapper.main()
    payload = json.loads(receipt.read_text())
    assert payload["average_posenet_distortion"] == 0.00001987654321
    assert payload["average_segnet_distortion"] == 0.0002987654321
    assert payload["archive_bytes"] == len(b"exact-archive")
    assert payload["sample_count"] == 600
    assert payload["resolved_device"] == "cpu"
    assert payload["ground_truth_dataset"] == "AVVideoDataset"
    assert payload["capture_mechanism"].startswith("line trace of immutable")
