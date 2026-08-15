from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/train_ddm_cl1_hpac_capacity_mps.py"
SPEC = importlib.util.spec_from_file_location("train_ddm_cl1_hpac_capacity_mps", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
port = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(port)


def test_reference_hash_is_pinned_and_current() -> None:
    assert port._sha256_file(port.REFERENCE_TRAINER) == port.REFERENCE_TRAINER_SHA256


@pytest.mark.parametrize(
    ("mode", "device", "epochs"),
    (
        ("parity-cpu", "cpu", 6),
        ("parity-mps", "mps", 6),
        ("full-mps", "mps", 60),
    ),
)
def test_mode_envelopes(mode: str, device: str, epochs: int) -> None:
    reference = port._load_reference()
    args = SimpleNamespace(
        profile="rx2_mc36",
        device=device,
        epochs=epochs,
        qat_fraction=0.5,
        eval_every=2,
        stop_after_epoch=None,
    )
    reference._build_parser = lambda: SimpleNamespace(parse_args=lambda argv: args)
    observed = port._validate_mode_args(reference, mode, ())
    assert observed is args


def test_parity_envelope_refuses_non_qat_scaled_run() -> None:
    reference = port._load_reference()
    args = SimpleNamespace(
        profile="rx2_mc36",
        device="mps",
        epochs=6,
        qat_fraction=0.5,
        eval_every=5,
        stop_after_epoch=None,
    )
    reference._build_parser = lambda: SimpleNamespace(parse_args=lambda argv: args)
    with pytest.raises(port.MPSPortError, match="sealed envelope"):
        port._validate_mode_args(reference, "parity-mps", ())


def test_configure_reference_preserves_cpu_reference_and_labels_mps() -> None:
    reference = port._load_reference()
    original = dict(reference.RX2_PREREGISTERED_CONFIG)
    intake = {"hpac_integer.py": "a" * 64}
    local = {"src/tac/training.py": "b" * 64}
    reference._run_identity = lambda *args, **kwargs: {
        "trainer_sha256": port.REFERENCE_TRAINER_SHA256,
        "intake_source_sha256": intake,
        "local_causal_source_sha256": local,
    }
    port._configure_reference(
        reference,
        port_mode="parity-mps",
        raw_launch_argv=["python", str(MODULE_PATH)],
    )
    assert reference.RX2_PREREGISTERED_CONFIG["device"] == "mps"
    assert reference.RX2_PREREGISTERED_CONFIG["epochs"] == 6
    assert {
        key: value for key, value in reference.RX2_PREREGISTERED_CONFIG.items() if key not in {"device", "epochs"}
    } == {key: value for key, value in original.items() if key not in {"device", "epochs"}}
    identity = reference._run_identity()
    assert identity["mps_trained"] is True
    assert identity["reference_trainer_sha256"] == port.REFERENCE_TRAINER_SHA256
    assert identity["training_reproducibility"]["run_to_run_bit_reproducible"] is False
    assert identity["training_reproducibility"]["serialization_authority"] == "cpu_pack_path"


def test_split_port_args_keeps_reference_arguments() -> None:
    mode, remaining = port._split_port_args(["--profile", "rx2_mc36", "--port-mode", "full-mps", "--device", "mps"])
    assert mode == "full-mps"
    assert remaining == ["--profile", "rx2_mc36", "--device", "mps"]
