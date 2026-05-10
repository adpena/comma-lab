from __future__ import annotations

import importlib.util
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "profile_pr95_hnerv_muon_intake.py"


def load_module():
    spec = importlib.util.spec_from_file_location("profile_pr95_hnerv_muon_intake", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _decoder_blob() -> bytes:
    out = io.BytesIO()
    records = [
        ("stem.weight", (2, 2), 0.125, b"\x01\x02\x03\x04"),
        ("block.weight", (2, 3), 0.25, b"\x05\x06\x07\x08\x09\x0a"),
        ("rgb.bias", (2,), 0.5, b"\x0b\x0c"),
    ]
    out.write(struct.pack("<I", len(records)))
    for name, shape, scale, qbytes in records:
        name_b = name.encode("utf-8")
        out.write(struct.pack("<I", len(name_b)))
        out.write(name_b)
        out.write(struct.pack("<I", len(shape)))
        for dim in shape:
            out.write(struct.pack("<I", dim))
        out.write(struct.pack("<f", scale))
        out.write(struct.pack("<I", len(qbytes)))
        out.write(qbytes)
    return out.getvalue()


def _latent_payload() -> bytes:
    n_pairs = 3
    latent_dim = 2
    return (
        struct.pack("<II", n_pairs, latent_dim)
        + b"\x00<\x00="
        + b"\x00>\x00?"
        + b"\x01\x02\x03\x04\x05\x06"
        + b"\x00\x00\x00\x00\x01\x00"
    )


def _top_blob() -> bytes:
    meta_raw = json.dumps({"latent_dim": 2, "n_pairs": 3}, sort_keys=True).encode()
    sections = [
        brotli.compress(meta_raw, quality=5),
        brotli.compress(_decoder_blob(), quality=5),
        brotli.compress(_latent_payload(), quality=5),
    ]
    out = io.BytesIO()
    for section in sections:
        out.write(struct.pack("<I", len(section)))
        out.write(section)
    return out.getvalue()


def _write_archive(path: Path) -> Path:
    info = zipfile.ZipInfo("0.bin", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, _top_blob())
    return path


def _write_source_tree(root: Path) -> Path:
    (root / "src" / "stages").mkdir(parents=True)
    (root / "src" / "__pycache__").mkdir(parents=True)
    (root / "README.md").write_text(
        "# HNeRV Muon\n\n"
        "A tiny archive for tests.\n\n"
        "The pipeline is an eight stage curriculum.\n\n"
        "```bash\npython src/train.py\n```\n\n"
        "~1 hours on test hardware\n"
        "Full writeup: https://example.invalid/writeup\n",
        encoding="utf-8",
    )
    (root / "src" / "model.py").write_text(
        "class HNeRV:\n"
        "    def __init__(self, latent_dim=2, base_channels=4, eval_size=(64, 128)):\n"
        "        self.base_h, self.base_w = 6, 8\n",
        encoding="utf-8",
    )
    (root / "src" / "optim.py").write_text(
        "def muon(lr=0.01, momentum=0.9, weight_decay=0.01, ns_steps=5):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (root / "src" / "codec.py").write_text(
        "# hybrid AC was ~217 bytes smaller in notes\n",
        encoding="utf-8",
    )
    (root / "src" / "score.py").write_text(
        '"""score helper\n\nscore = 100 * seg + sqrt(10 * pose) + rate\n"""\n',
        encoding="utf-8",
    )
    (root / "src" / "train.py").write_text("# train\n", encoding="utf-8")
    (root / "src" / "__pycache__" / "model.cpython-312.pyc").write_bytes(b"bytecode")
    stage_defs = [
        ("stage1_v328_ce", 3000, "ce_seg_loss()", "1e-3", "0.0", "0.2", False, False, "2e-4", "0.0"),
        ("stage2_v331_softplus", 5650, "tau_softplus_seg_loss()", "1e-3", "0.0", "0.2", False, False, "2e-4", "0.0"),
        ("stage3_v332_smooth", 1500, "smooth_disagreement_seg_loss()", "1e-4", "0.0", "0.2", False, False, "2e-4", "0.0"),
        ("stage4_v332_qat", 500, "smooth_disagreement_seg_loss()", "1e-4", "0.0", "0.2", True, False, "2e-4", "0.0"),
        ("stage5_c1a_l7", 9000, "l7_softplus_seg_loss()", "3e-5", "0.01", "0.2", True, False, "2e-4", "0.0"),
        ("stage6_lambda_sweep", 2000, "l7_softplus_seg_loss()", "3e-5", "0.02", "0.2", True, False, "2e-4", "0.0"),
        ("stage7_sigma_sweep", 3000, "l7_softplus_seg_loss()", "3e-5", "0.02", "0.1", True, False, "2e-4", "0.0"),
        ("stage8_muon_finetune", 5000, "l7_softplus_seg_loss()", "1e-5", "0.02", "0.1", True, True, "2e-4", "5e-4"),
    ]
    for index, (
        name,
        epochs,
        loss_call,
        adamw_lr,
        cat_lambda,
        cat_sigma,
        use_qat,
        use_muon,
        muon_lr,
        muon_weight_decay,
    ) in enumerate(stage_defs, start=1):
        (root / "src" / "stages" / f"stage{index}.py").write_text(
            f'"""stage {index} doc"""\n'
            f'name="{name}"\n'
            f"def run(epochs: int = {epochs}, muon_weight_decay: float = {muon_weight_decay}):\n"
            f"    train(adamw_lr={adamw_lr}, muon_lr={muon_lr}, "
            f"cat_lambda={cat_lambda}, cat_sigma={cat_sigma}, "
            f"use_qat={use_qat}, use_muon={use_muon})\n"
            f"    {loss_call}\n",
            encoding="utf-8",
        )
    return root


def test_profile_pr95_hnerv_muon_intake_static_archive_and_source(tmp_path: Path) -> None:
    module = load_module()
    archive = _write_archive(tmp_path / "archive.zip")
    source_dir = _write_source_tree(tmp_path / "source")
    static_intake = tmp_path / "static.json"
    static_intake.write_text(
        json.dumps(
            {
                "score_claim": False,
                "claimed_body_score_inputs": {
                    "seg": 0.001,
                    "pose": 0.004,
                    "archive_bytes": archive.stat().st_size,
                    "recomputed_score": module.compute_score_terms(
                        0.001,
                        0.004,
                        archive.stat().st_size,
                    )["recomputed_score"],
                },
            }
        ),
        encoding="utf-8",
    )

    profile = module.build_profile(archive, source_dir, static_intake)

    assert profile["schema"] == "pr95_hnerv_muon_static_intake_profile_v1"
    assert profile["evidence_grade"] == "external_static_intake_only"
    assert profile["score_term_math"]["score_claim"] is False
    assert profile["dispatch_readiness"]["ready_for_dispatch"] is False
    assert profile["hnerv_muon_blob"]["decoder"]["tensor_count"] == 3
    assert profile["hnerv_muon_blob"]["decoder"]["muon_partition_params"] == 6
    assert profile["hnerv_muon_blob"]["latents"]["n_frame_pairs"] == 3
    assert profile["source_intake"]["source_file_count"] == 14
    assert profile["source_intake"]["model_defaults"]["latent_dim"] == 2
    assert profile["source_intake"]["training_stages"][0]["name"] == "stage1_v328_ce"
    assert profile["source_intake"]["training_stages"][-1]["uses_muon"] is True
    parity = profile["trainer_parity_contract"]
    assert parity["schema"] == "pr95_hnerv_muon_t1_trainer_parity_v1"
    assert parity["preflight_contract"]["local_trainer_parity_preflight_passed"] is True
    assert parity["preflight_contract"]["ready_for_score_bearing_t1_hnerv_parity_dispatch"] is False
    assert parity["t1_trainer_config"]["required_flags"]["--enable-scorer-domain-loss"] is True
    assert parity["t1_trainer_config"]["required_flags"]["--yuv6-mode"] == "monkey_patch_global"
    assert profile["immediate_improvement_hypotheses"][0]["hook"] == "RAFT/ego-motion/foveation latent bases"


def test_default_pr95_source_dir_uses_release_view_source_not_pyc_recovery_tree() -> None:
    module = load_module()

    assert "public_pr_archive_release_view" in module.DEFAULT_SOURCE_DIR.as_posix()
    assert module.DEFAULT_SOURCE_DIR.as_posix().endswith(
        "source/submissions/hnerv_muon"
    )
