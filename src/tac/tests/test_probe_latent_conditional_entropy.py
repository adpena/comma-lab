# SPDX-License-Identifier: MIT
"""Sanity tests for the H(latent | scorer_class) probe.

Per the HIGH-RISK substrate cargo-cult unwind audit 2026-05-16 D4
operator-approved decision: ATW V1 / D4 Wyner-Ziv / Z4 cooperative-receiver
all predict that conditioning the codec on SegNet class labels reduces the
latent's bit rate. This shared probe measures empirical conditional entropy
+ emits a typed verdict so the predicted Wyner-Ziv gain ceiling can be
audited against the measured value.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    target = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "tools"
        / "probe_latent_conditional_entropy_h_latent_given_scorer_class.py"
    )
    module_name = "probe_latent_conditional_entropy_test"
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_helper_module_loads():
    mod = _load_module()
    assert hasattr(mod, "compute_h_latent_given_scorer_class")
    assert hasattr(mod, "HLatentGivenScorerClassVerdict")
    assert mod.DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS == 0.5


def test_meaningful_conditioning_when_latent_equals_class():
    """If latent == class then H(latent|class) = 0 and MI = H(latent)."""
    mod = _load_module()
    # 4 classes uniformly distributed, latent == class -> H(latent) = 2 bits,
    # H(latent|class) = 0, MI = 2.
    classes = [0, 1, 2, 3] * 250
    latents = list(classes)
    verdict = mod.compute_h_latent_given_scorer_class(
        substrate_id="test_meaningful",
        latent_stream=latents,
        class_stream=classes,
    )
    assert verdict.verdict == "MEANINGFUL_CONDITIONING"
    assert verdict.mutual_information_bits > 1.9
    assert verdict.h_latent_given_scorer_class_bits_per_symbol < 0.01
    assert verdict.evidence_grade == "diagnostic_cpu"
    assert verdict.score_claim is False
    assert "diagnostic-CPU" in verdict.axis_label


def test_independent_when_latent_uncorrelated_with_class():
    """Independent uniform streams -> MI ~= 0."""
    mod = _load_module()
    # Constant class, uniform latent -> H(latent|class) = H(latent), MI = 0.
    classes = [0] * 1000
    latents = [i % 4 for i in range(1000)]
    verdict = mod.compute_h_latent_given_scorer_class(
        substrate_id="test_independent",
        latent_stream=latents,
        class_stream=classes,
    )
    assert verdict.verdict == "INDEPENDENT"
    assert verdict.mutual_information_bits < 0.01
    assert verdict.score_claim is False


def test_weak_conditioning_in_between():
    """Partial correlation -> MI in the WEAK_CONDITIONING band."""
    mod = _load_module()
    # 2 classes; latent equals class half the time, random otherwise -> MI
    # around 0.3 bits which falls below the default 0.5 threshold.
    classes = []
    latents = []
    for i in range(1000):
        cls = i % 2
        classes.append(cls)
        if i % 4 < 3:  # 75% correlated
            latents.append(cls)
        else:
            latents.append(1 - cls)
    verdict = mod.compute_h_latent_given_scorer_class(
        substrate_id="test_weak",
        latent_stream=latents,
        class_stream=classes,
        # Tighten the meaningful threshold so 75% correlation falls in WEAK.
        meaningful_mi_threshold_bits=0.9,
    )
    assert verdict.verdict == "WEAK_CONDITIONING"
    assert 0.01 < verdict.mutual_information_bits < 0.9


def test_invalid_inputs_raise():
    mod = _load_module()
    # Mismatched lengths
    try:
        mod.compute_h_latent_given_scorer_class(
            substrate_id="x",
            latent_stream=[0, 1],
            class_stream=[0],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched lengths")
    # Empty latent
    try:
        mod.compute_h_latent_given_scorer_class(
            substrate_id="x",
            latent_stream=[],
            class_stream=[],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty latent")
    # Empty substrate_id
    try:
        mod.compute_h_latent_given_scorer_class(
            substrate_id="",
            latent_stream=[0, 1],
            class_stream=[0, 1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty substrate_id")


def test_cli_writes_json_with_evidence_grade(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tool = (
        repo_root
        / "tools"
        / "probe_latent_conditional_entropy_h_latent_given_scorer_class.py"
    )
    latent_path = tmp_path / "latent.bin"
    class_path = tmp_path / "classes.bin"
    # latent == class -> meaningful conditioning
    payload = bytes(i % 4 for i in range(1000))
    latent_path.write_bytes(payload)
    class_path.write_bytes(payload)
    out_path = tmp_path / "verdict.json"
    proc = subprocess.run(
        [
            sys.executable, str(tool),
            "--substrate-id", "test_cli",
            "--latent-bytes", str(latent_path),
            "--scorer-classes", str(class_path),
            "--output-json", str(out_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["verdict"] == "MEANINGFUL_CONDITIONING"
    assert data["evidence_grade"] == "diagnostic_cpu"
    assert data["score_claim"] is False
    assert "diagnostic-CPU" in data["axis_label"]
