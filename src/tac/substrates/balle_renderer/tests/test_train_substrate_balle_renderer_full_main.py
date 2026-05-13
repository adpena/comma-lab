"""OD-SUBSTRATE-3 follow-up Option A wired ``_full_main`` test suite for β.

These tests verify the wired contract WITHOUT dispatching GPU. They cover:

* argparse correctness (every TIER_1 flag has an argparse entry; help OK)
* device gating (cuda/cpu/--smoke interactions)
* TIER_1 manifest schema (per Catalog #151 — β-specific flags included)
* helper utilities (seed pin, sha256, archive bytes proxy, decoder)
* runtime emission (Catalog #146 contract; deterministic zip)
* smoke training path runs to completion on CPU
* β-specific Lagrangian computation on a small CPU batch (hyperprior term)
* NaN watchdog wiring (``_full_main`` source carries the watchdog clause)
* EMA shadow save semantics (smoke fallback for state_dict pattern)
* State-dict-split helper round-trips the substrate's hyper/decoder/hp keys
* β's hyperprior rate term is wired into the loss + backprop path
* Source-level introspection of ``_full_main`` covers all CLAUDE.md tokens

Test discipline: no GPU, no scorer load, no real video decode at test scope.
Heavy paths (scorer + pyav decode) are tested by integration smoke at
operator-gated dispatch time per the council's "minimum-viable-integration-loop
first" rule.
"""

from __future__ import annotations

import importlib
import io
import math
import subprocess
import sys
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
"""Repo root, navigated from ``src/tac/substrates/balle_renderer/tests/``."""


@pytest.fixture(scope="module")
def trainer_module():
    """Import ``experiments.train_substrate_balle_renderer`` once per module."""
    return importlib.import_module(
        "experiments.train_substrate_balle_renderer"
    )


# ---------------------------------------------------------------------------
# 1. TIER_1 manifest schema (Catalog #151)
# ---------------------------------------------------------------------------

def test_tier_1_operator_required_flags_is_nonempty_dict(trainer_module):
    """The manifest must be a non-empty dict per Catalog #151."""
    m = trainer_module.TIER_1_OPERATOR_REQUIRED_FLAGS
    assert isinstance(m, dict)
    assert m, "TIER_1 manifest must declare ≥1 required flag"


def test_tier_1_flags_have_required_keys(trainer_module):
    """Every TIER_1 entry must carry ``env`` + ``rationale`` strings."""
    m = trainer_module.TIER_1_OPERATOR_REQUIRED_FLAGS
    for flag, meta in m.items():
        assert isinstance(meta, dict), f"{flag} value must be dict"
        assert "env" in meta and isinstance(meta["env"], str) and meta["env"], (
            f"{flag} missing non-empty 'env' key"
        )
        assert (
            "rationale" in meta
            and isinstance(meta["rationale"], str)
            and meta["rationale"]
        ), f"{flag} missing non-empty 'rationale' key"


def test_tier_1_includes_beta_specific_flags(trainer_module):
    """β substrate adds ``--balle-hyperprior-channels``, ``--gdn-eps`` and
    ``--lambda-hyperprior`` per the council §2.10 / §4.2 design."""
    m = trainer_module.TIER_1_OPERATOR_REQUIRED_FLAGS
    for required in (
        "--balle-hyperprior-channels", "--gdn-eps", "--lambda-hyperprior",
    ):
        assert required in m, f"TIER_1 missing β-specific flag {required}"


def test_tier_1_required_input_file_default_resolves_under_repo(trainer_module):
    """Catalog #152 contract: required_input_file=True defaults must resolve
    to a real file in the pinned repo (contest video).
    """
    m = trainer_module.TIER_1_OPERATOR_REQUIRED_FLAGS
    for flag, meta in m.items():
        if not meta.get("required_input_file"):
            continue
        default = meta.get("default")
        assert isinstance(default, str) and default.strip(), (
            f"{flag} declares required_input_file=True without a string default"
        )
        path = (REPO_ROOT / default).resolve()
        assert path.is_file(), (
            f"{flag}'s required_input_file default {path} does not exist on disk"
        )


def test_tier_1_flags_match_argparse_subset(trainer_module):
    """Every flag in TIER_1 must also appear in argparse (sister of Catalog
    #12 dead-flag detector)."""
    parser = trainer_module._build_parser()
    actions = {a.option_strings[0] for a in parser._actions if a.option_strings}
    m = trainer_module.TIER_1_OPERATOR_REQUIRED_FLAGS
    for flag in m:
        assert flag in actions, (
            f"TIER_1 flag {flag} not declared in argparse — Catalog #151 "
            "+ #12 wire-up will refuse this trainer at preflight."
        )


def test_tier_1_env_namespace_balle_renderer(trainer_module):
    """All env var names must use the BALLE_RENDERER_* namespace so a wrapper
    can confidently scope env-gating without colliding with α's namespace."""
    m = trainer_module.TIER_1_OPERATOR_REQUIRED_FLAGS
    for flag, meta in m.items():
        env = meta["env"]
        assert env.startswith("BALLE_RENDERER_"), (
            f"{flag}.env={env} must use BALLE_RENDERER_* namespace"
        )


# ---------------------------------------------------------------------------
# 2. Argparse correctness
# ---------------------------------------------------------------------------

def test_argparse_help_runs(trainer_module):
    """``--help`` must print and exit 0 (no SystemExit traceback)."""
    parser = trainer_module._build_parser()
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    out = buf.getvalue()
    assert "--video-path" in out
    assert "--output-dir" in out
    assert "--epochs" in out
    assert "--balle-hyperprior-channels" in out
    assert "--lambda-hyperprior" in out


def test_argparse_smoke_mode_required_flags_minimal(trainer_module, tmp_path):
    """Smoke mode parses with the minimal flag set."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--video-path", str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
        "--output-dir", str(tmp_path),
        "--epochs", "3",
        "--device", "cpu",
        "--smoke",
    ])
    assert args.smoke is True
    assert args.device == "cpu"
    assert args.epochs == 3
    assert args.ema_decay == 0.997  # CLAUDE.md non-negotiable default
    assert args.alpha_rate == 25.0  # contest evaluate.py default
    assert args.beta_seg == 100.0
    assert args.gamma_pose == math.sqrt(10.0)
    assert args.pose_weight_scale == 1.0
    # β-specific defaults
    assert args.balle_hyperprior_channels == 8
    assert args.gdn_eps == 1e-12
    assert args.lambda_hyperprior == 0.5


def test_argparse_rejects_mps_device(trainer_module, tmp_path):
    """--device mps must be rejected at argparse level (CLAUDE.md MPS rule)."""
    parser = trainer_module._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--video-path", str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
            "--output-dir", str(tmp_path),
            "--epochs", "1",
            "--device", "mps",
        ])


# ---------------------------------------------------------------------------
# 3. Device gating
# ---------------------------------------------------------------------------

def test_device_or_die_refuses_cpu_without_smoke(trainer_module):
    """`_device_or_die` must refuse cpu in non-smoke mode."""
    with pytest.raises(SystemExit) as exc:
        trainer_module._device_or_die("cpu", smoke=False)
    msg = str(exc.value)
    assert "cpu" in msg.lower() or "smoke" in msg.lower()


def test_device_or_die_accepts_cpu_with_smoke(trainer_module):
    """`_device_or_die` accepts cpu when smoke=True (CPU smoke is allowed)."""
    device = trainer_module._device_or_die("cpu", smoke=True)
    assert device.type == "cpu"


def test_device_or_die_refuses_unknown_device(trainer_module):
    """Unknown device name is refused with a clear error."""
    with pytest.raises(SystemExit):
        trainer_module._device_or_die("tpu", smoke=True)


# ---------------------------------------------------------------------------
# 4. Helper utilities
# ---------------------------------------------------------------------------

def test_pin_seeds_is_deterministic(trainer_module):
    """Calling _pin_seeds with the same seed produces same random draws."""
    import random as _r

    import torch

    trainer_module._pin_seeds(42)
    a_py = _r.random()
    a_torch = torch.rand(3).tolist()
    trainer_module._pin_seeds(42)
    b_py = _r.random()
    b_torch = torch.rand(3).tolist()
    assert a_py == b_py
    assert a_torch == b_torch


def test_sha256_bytes_matches_hashlib(trainer_module):
    import hashlib
    data = b"balle_renderer test bytes"
    expect = hashlib.sha256(data).hexdigest()
    assert trainer_module._sha256_bytes(data) == expect


def test_utc_now_iso_format(trainer_module):
    s = trainer_module._utc_now_iso()
    assert len(s) == 20
    assert s.endswith("Z")
    assert s[10] == "T"


def test_archive_bytes_proxy_is_positive(trainer_module):
    """Closed-form archive-bytes proxy is positive and lives on model device.

    β: the proxy must include BOTH the main-latent and hyper-latent int16
    byte cost (sister of α's main-latent-only proxy).
    """
    import torch

    from tac.substrates.balle_renderer.architecture import (
        BalleRendererConfig,
        BalleRendererSubstrate,
    )

    cfg = BalleRendererConfig(
        latent_dim=8, hyper_latent_dim=4,
        embed_dim=24, initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        hyper_mlp_channels=(8, 8),
        num_pairs=4, output_height=24, output_width=32,
        num_upsample_blocks=3,
    )
    model = BalleRendererSubstrate(cfg)
    proxy = trainer_module._archive_bytes_proxy_closed_form(model)
    assert float(proxy.item()) > 0
    assert proxy.dtype == torch.float32
    # β-specific: must be strictly larger than the same model's main-latents-only
    # naive proxy, since the β proxy ALSO counts the hyper-latent int16 stream.
    main_only = float(
        sum(p.numel() for n, p in model.named_parameters() if n != "latents") * 2
        + model.latents.numel() * 2
    )
    assert float(proxy.item()) > main_only, (
        "β proxy must include hyper-latent bytes; otherwise the rate-term "
        "underestimates archive bytes vs the actual BRV1 grammar."
    )


def test_split_state_dict_partitions_keys_correctly(trainer_module):
    """``_split_state_dict_for_archive`` must place every substrate key in
    exactly one of (encoder, decoder, hyperprior) + latents.
    """
    from tac.substrates.balle_renderer.architecture import (
        BalleRendererConfig,
        BalleRendererSubstrate,
    )

    cfg = BalleRendererConfig(
        latent_dim=4, hyper_latent_dim=4,
        embed_dim=16, initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(8, 6, 4, 4, 4, 4, 4),
        hyper_mlp_channels=(4, 4),
        num_pairs=2, output_height=24, output_width=32,
        num_upsample_blocks=3,
    )
    model = BalleRendererSubstrate(cfg).eval()
    sd = model.state_dict()
    enc, dec, hp, latents = trainer_module._split_state_dict_for_archive(sd)
    # Every non-latents key must land in exactly one bucket.
    all_keys_in_buckets = set()
    # Encoder keys are de-prefixed; we re-prefix to compare.
    for k in enc:
        all_keys_in_buckets.add("hyper_analysis." + k)
    all_keys_in_buckets.update(dec.keys())
    all_keys_in_buckets.update(hp.keys())
    # latents is separate
    state_minus_latents = {k for k in sd if k != "latents"}
    assert all_keys_in_buckets == state_minus_latents, (
        "non-latents state_dict keys must partition cleanly across "
        "(encoder, decoder, hyperprior); missing="
        f"{state_minus_latents - all_keys_in_buckets} "
        f"extra={all_keys_in_buckets - state_minus_latents}"
    )
    # Latents shape preserved
    assert latents.shape == sd["latents"].shape


def test_balle_renderer_config_wires_gdn_eps_and_quant_noise():
    """Operator flags must change constructed substrate behavior."""
    from tac.substrates.balle_renderer.architecture import (
        BalleRendererConfig,
        BalleRendererSubstrate,
    )

    cfg = BalleRendererConfig(
        latent_dim=4,
        hyper_latent_dim=4,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(8, 6, 4, 4, 4, 4, 4),
        hyper_mlp_channels=(4, 4),
        num_pairs=2,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
        quantize_noise_std=0.125,
        gdn_eps=1e-6,
    )
    model = BalleRendererSubstrate(cfg)
    assert model.cfg.quantize_noise_std == 0.125
    assert model.blocks[0].igdn.eps == 1e-6


def test_balle_renderer_config_rejects_invalid_numeric_floors():
    from tac.substrates.balle_renderer.architecture import BalleRendererConfig

    with pytest.raises(ValueError, match="gdn_eps"):
        BalleRendererConfig(gdn_eps=0.0)
    with pytest.raises(ValueError, match="quantize_noise_std"):
        BalleRendererConfig(quantize_noise_std=-0.1)


# ---------------------------------------------------------------------------
# 5. Runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

def test_write_runtime_emits_three_positional_args_in_inflate_sh(
    trainer_module, tmp_path
):
    """inflate.sh template must carry $1/$2/$3 (or named equivalents)."""
    trainer_module._write_runtime(tmp_path)
    sh = (tmp_path / "inflate.sh").read_text()
    assert "$1" in sh
    assert "$2" in sh
    assert "$3" in sh


def test_write_runtime_inflate_sh_uses_set_e(trainer_module, tmp_path):
    """inflate.sh must include `set -e` or `set -euo pipefail`."""
    trainer_module._write_runtime(tmp_path)
    sh = (tmp_path / "inflate.sh").read_text()
    assert "set -euo pipefail" in sh or "set -e " in sh or "set -e\n" in sh


def test_write_runtime_inflate_sh_no_single_arg_passthrough(
    trainer_module, tmp_path
):
    """inflate.sh must NOT use the forbidden `inflate.py "$@"` scaffold pattern."""
    trainer_module._write_runtime(tmp_path)
    sh = (tmp_path / "inflate.sh").read_text()
    assert 'inflate.py" "$@"' not in sh


def test_write_runtime_inflate_py_no_scorer_imports(trainer_module, tmp_path):
    """inflate.py template must NOT import scorer code per strict-scorer-rule."""
    trainer_module._write_runtime(tmp_path)
    py = (tmp_path / "inflate.py").read_text()
    forbidden = ("PoseNet", "SegNet", "rgb_to_yuv6", "EfficientNet", "FastViT")
    for tok in forbidden:
        assert tok not in py, (
            f"inflate.py contains forbidden scorer token {tok}"
        )


def test_write_runtime_inflate_py_has_per_video_loop(trainer_module, tmp_path):
    """inflate.py must iterate file_list (per Catalog #146)."""
    trainer_module._write_runtime(tmp_path)
    py = (tmp_path / "inflate.py").read_text()
    assert "splitlines()" in py
    assert "file_list_path" in py or "file_list.read_text" in py


def test_write_runtime_inflate_py_references_balle_renderer_substrate(
    trainer_module, tmp_path
):
    """inflate.py must reference the β-specific package (not α's sane_hnerv).

    This guards against copy-paste regression where the emitted runtime
    accidentally points at α's substrate.
    """
    trainer_module._write_runtime(tmp_path)
    py = (tmp_path / "inflate.py").read_text()
    assert "tac.substrates.balle_renderer.inflate" in py
    assert "tac.substrates.sane_hnerv.inflate" not in py, (
        "beta inflate.py must NOT import alpha's sane_hnerv inflate"
    )


def test_write_runtime_inflate_py_under_size_budget(trainer_module, tmp_path):
    """inflate.py wrapper itself should be small; substrate's inflate has its
    own ≤ 200 LOC waiver budget audited separately."""
    trainer_module._write_runtime(tmp_path)
    py = (tmp_path / "inflate.py").read_text()
    nonblank = [
        ln for ln in py.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert len(nonblank) <= 40, (
        f"inflate.py wrapper too large: {len(nonblank)} non-blank/non-comment "
        f"lines (budget: 40)"
    )


def test_build_archive_zip_is_deterministic(trainer_module, tmp_path):
    """Archive zip with same bytes + same submission_dir must yield identical bytes."""
    sub = tmp_path / "sub"
    trainer_module._write_runtime(sub)
    bin_bytes = b"BRV1\x01" + b"\x00" * 100
    (sub / "0.bin").write_bytes(bin_bytes)
    zip_a = tmp_path / "a.zip"
    zip_b = tmp_path / "b.zip"
    trainer_module._build_archive_zip(
        zip_a, bin_bytes=bin_bytes, submission_dir=sub
    )
    trainer_module._build_archive_zip(
        zip_b, bin_bytes=bin_bytes, submission_dir=sub
    )
    assert zip_a.read_bytes() == zip_b.read_bytes()
    with zipfile.ZipFile(zip_a, "r") as zf:
        names = set(zf.namelist())
    assert "0.bin" in names
    assert "inflate.sh" in names
    assert "inflate.py" in names


def test_full_main_custody_uses_archive_zip_sha_not_0bin():
    """Auth-eval custody must validate the scored archive.zip object.

    Catalog #190-style substrate trainers must not pass the internal ``0.bin``
    payload SHA to auth-eval claim validation, because contest_auth_eval scores
    the archive.zip supplied via ``--archive``.
    """
    src_path = REPO_ROOT / "experiments" / "train_substrate_balle_renderer.py"
    src = src_path.read_text(encoding="utf-8")
    idx = src.find("def _full_main")
    assert idx >= 0, "_full_main definition not found"
    body = src[idx:]
    assert "payload_0bin_sha" in body
    assert "payload_0bin_bytes" in body
    assert "archive_bytes = archive_zip_path.stat().st_size" in body
    assert "archive_sha = _sha256_bytes(archive_zip_path.read_bytes())" in body
    assert "archive_sha256=archive_sha" in body
    assert '"payload_0bin_sha256": payload_0bin_sha' in body
    assert '"payload_0bin_bytes": payload_0bin_bytes' in body
    assert "auth_eval_alias_path = args.output_dir / \"auth_eval.json\"" in body
    assert "shutil.copy2(auth_eval_result_path, auth_eval_alias_path)" in body
    assert '"exact_eval_packet": {' in body
    assert '"archive_path": (' in body
    assert "str(archive_zip_path) if archive_zip_path.is_file() else None" in body
    assert '"inflate_sh_path": (' in body


# ---------------------------------------------------------------------------
# 6. Smoke training (CPU end-to-end)
# ---------------------------------------------------------------------------

def test_smoke_main_runs_to_completion_cpu(trainer_module, tmp_path):
    """``_smoke_main`` must run to completion on CPU + write smoke checkpoint."""
    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--video-path", str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
        "--output-dir", str(tmp_path / "smoke"),
        "--epochs", "2",
        "--device", "cpu",
        "--smoke",
    ])
    rc = trainer_module._smoke_main(args)
    assert rc == 0
    ckpt = tmp_path / "smoke" / "smoke_checkpoint.pt"
    assert ckpt.is_file(), "smoke must write smoke_checkpoint.pt"


def test_smoke_checkpoint_carries_config_and_state(trainer_module, tmp_path):
    """Smoke checkpoint must contain ``state_dict`` + ``config`` keys."""
    import torch

    parser = trainer_module._build_parser()
    args = parser.parse_args([
        "--video-path", str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
        "--output-dir", str(tmp_path / "smoke"),
        "--epochs", "1",
        "--device", "cpu",
        "--smoke",
    ])
    trainer_module._smoke_main(args)
    ckpt = torch.load(
        tmp_path / "smoke" / "smoke_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "state_dict" in ckpt
    assert "config" in ckpt
    assert ckpt.get("smoke") is True
    # β-specific: the hyper_latent_dim must be reflected in the config
    assert "hyper_latent_dim" in ckpt["config"]
    assert ckpt["config"]["hyper_latent_dim"] > 0


# ---------------------------------------------------------------------------
# 7. Lagrangian smoke (no scorer; just verify the β loss module's contract)
# ---------------------------------------------------------------------------

def test_score_aware_loss_runs_on_dummy_scorers():
    """β Lagrangian computes a scalar on a tiny dummy-scorer setup,
    including the hyperprior rate term contribution."""
    import torch
    import torch.nn as nn

    from tac.substrates.balle_renderer.score_aware_loss import (
        BalleRendererScoreAwareLoss,
        BalleScoreAwareLossWeights,
    )

    class DummySeg(nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            # (B, T, C, H, W) -> (B, C, H, W), matching upstream SegNet.
            return x_btchw[:, -1, ...]

        def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
            # x: (B, 3, H, W) -> (B, 5, h, w)
            b, _c, h, w = x_bchw.shape
            return (
                torch.zeros(b, 5, h // 4, w // 4, dtype=x_bchw.dtype)
                + x_bchw.mean()
            )

    class DummyPose(nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            # (B, T, C, H, W) -> (B, T*6, H/2, W/2), a tiny yuv6 analogue.
            b, t, _c, h, w = x_btchw.shape
            flat = x_btchw.reshape(b * t, 3, h, w).mean(dim=1, keepdim=True)
            flat6 = flat.expand(-1, 6, -1, -1)
            flat6_sub = flat6.reshape(b * t, 6, h // 2, 2, w // 2, 2).mean(dim=(3, 5))
            return flat6_sub.reshape(b, t * 6, h // 2, w // 2)

        def forward(self, x_bc: torch.Tensor) -> torch.Tensor:
            # x: (B, 12, H, W) -> {"pose": (B, 12)}
            return {"pose": x_bc.flatten(2).mean(dim=2)}

    weights = BalleScoreAwareLossWeights(
        alpha_rate=25.0,
        beta_seg=100.0,
        gamma_pose=math.sqrt(10.0),
        pose_weight_scale=1.0,
        lambda_hyperprior=0.5,
        contest_normalizer=37545489.0,
    )
    loss_fn = BalleRendererScoreAwareLoss(
        seg_scorer=DummySeg(), pose_scorer=DummyPose(), weights=weights,
    )
    b = 2
    rgb_0 = torch.rand(b, 3, 16, 24, requires_grad=True) * 255.0
    rgb_1 = torch.rand(b, 3, 16, 24, requires_grad=True) * 255.0
    gt_0 = torch.rand(b, 3, 16, 24) * 255.0
    gt_1 = torch.rand(b, 3, 16, 24) * 255.0
    bytes_proxy = torch.tensor(100_000.0)
    rate_components = {
        "hyper_rate": torch.tensor(0.5, requires_grad=True),
        "main_rate": torch.tensor(1.0, requires_grad=True),
        "total_rate": torch.tensor(1.5, requires_grad=True),
    }
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, rate_components,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # Parts must include rate/seg/pose + β's hyperprior-rate component
    for key in (
        "rate_term", "seg_term", "pose_term",
        "hyperprior_rate_term", "loss_total",
    ):
        assert key in parts, f"loss parts missing {key!r}"


def test_score_aware_loss_refuses_eval_roundtrip_false():
    """Catalog #5: ``apply_eval_roundtrip=False`` is forbidden in production."""
    import torch
    import torch.nn as nn

    from tac.substrates.balle_renderer.score_aware_loss import (
        BalleRendererScoreAwareLoss,
        BalleScoreAwareLossWeights,
    )

    class _Dummy(nn.Module):
        def forward(self, x):
            return x.flatten(1).mean(dim=1, keepdim=True).expand(-1, 12)

    loss_fn = BalleRendererScoreAwareLoss(
        seg_scorer=_Dummy(), pose_scorer=_Dummy(),
        weights=BalleScoreAwareLossWeights(),
    )
    rc = {
        "hyper_rate": torch.tensor(0.0),
        "main_rate": torch.tensor(0.0),
        "total_rate": torch.tensor(0.0),
    }
    with pytest.raises(ValueError):
        loss_fn(
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.tensor(1.0),
            rc,
            apply_eval_roundtrip=False,
        )


def test_score_aware_loss_refuses_negative_noise_std():
    import torch
    import torch.nn as nn

    from tac.substrates.balle_renderer.score_aware_loss import (
        BalleRendererScoreAwareLoss,
        BalleScoreAwareLossWeights,
    )

    class _Dummy(nn.Module):
        def forward(self, x):
            return x.flatten(1).mean(dim=1, keepdim=True).expand(-1, 12)

    loss_fn = BalleRendererScoreAwareLoss(
        seg_scorer=_Dummy(),
        pose_scorer=_Dummy(),
        weights=BalleScoreAwareLossWeights(),
    )
    rc = {
        "hyper_rate": torch.tensor(0.0),
        "main_rate": torch.tensor(0.0),
        "total_rate": torch.tensor(0.0),
    }
    with pytest.raises(ValueError, match="noise_std"):
        loss_fn(
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.tensor(1.0),
            rc,
            apply_eval_roundtrip=True,
            noise_std=-0.1,
        )


def test_score_aware_loss_hyperprior_term_contributes_to_loss():
    """β-specific: the loss must change when the hyperprior rate changes
    (i.e. the term is wired into the total, not silently dropped)."""
    import torch
    import torch.nn as nn

    from tac.substrates.balle_renderer.score_aware_loss import (
        BalleRendererScoreAwareLoss,
        BalleScoreAwareLossWeights,
    )

    class DummySeg(nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            return x_btchw[:, -1, ...]

        def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
            b, _c, h, w = x_bchw.shape
            return torch.zeros(b, 5, h, w, dtype=x_bchw.dtype) + x_bchw.mean()

    class DummyPose(nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            if x_btchw.dim() != 5:
                raise AssertionError(f"expected 5D scorer input, got {x_btchw.dim()}D")
            b, t, _c, h, w = x_btchw.shape
            flat = x_btchw.reshape(b * t, 3, h, w).mean(dim=1, keepdim=True)
            flat6 = flat.expand(-1, 6, -1, -1)
            flat6_sub = flat6.reshape(b * t, 6, h // 2, 2, w // 2, 2).mean(dim=(3, 5))
            return flat6_sub.reshape(b, t * 6, h // 2, w // 2)

        def forward(self, x):
            return {"pose": x.flatten(2).mean(dim=2)}

    weights = BalleScoreAwareLossWeights(
        alpha_rate=25.0,
        beta_seg=100.0,
        gamma_pose=math.sqrt(10.0),
        pose_weight_scale=1.0,
        lambda_hyperprior=2.0,  # bigger to amplify
        contest_normalizer=37545489.0,
    )
    loss_fn = BalleRendererScoreAwareLoss(
        seg_scorer=DummySeg(), pose_scorer=DummyPose(), weights=weights,
    )
    rgb_0 = torch.rand(1, 3, 8, 8, requires_grad=True) * 255.0
    rgb_1 = torch.rand(1, 3, 8, 8, requires_grad=True) * 255.0
    gt_0 = torch.rand(1, 3, 8, 8) * 255.0
    gt_1 = torch.rand(1, 3, 8, 8) * 255.0
    bytes_proxy = torch.tensor(100_000.0)
    rc_low = {
        "hyper_rate": torch.tensor(0.0),
        "main_rate": torch.tensor(0.0),
        "total_rate": torch.tensor(0.0),
    }
    rc_high = {
        "hyper_rate": torch.tensor(2.0),
        "main_rate": torch.tensor(3.0),
        "total_rate": torch.tensor(5.0),
    }
    loss_low, _ = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, rc_low,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    loss_high, _ = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, rc_high,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    # With lambda_hyperprior=2.0 and total_rate difference of 5.0,
    # the loss must increase by ~10.0.
    assert float((loss_high - loss_low).item()) > 5.0, (
        "β hyperprior rate term must contribute to the total Lagrangian; "
        f"got delta = {float((loss_high - loss_low).item())}"
    )


# ---------------------------------------------------------------------------
# 8. _full_main wiring proof (source-level introspection)
# ---------------------------------------------------------------------------

_FULL_MAIN_REQUIRED_TOKENS = (
    # CLAUDE.md non-negotiables
    "patch_upstream_yuv6_globally",
    "unpatch_upstream_yuv6",
    "load_differentiable_scorers",
    "apply_eval_roundtrip=True",
    "EMA",
    "_pin_seeds",
    "AdamW",
    "CosineAnnealingLR",
    "ema.update",
    "ema.apply",
    "torch.nn.utils.clip_grad_norm_",
    # NaN watchdog (Council D)
    "nan_strike",
    "isfinite",
    # Best-ckpt EMA shadow save
    "ema.state_dict()",
    # Auth-eval invocation (CLAUDE.md "Auth eval EVERYWHERE")
    "contest_auth_eval",
    "--device",
    # Continual-learning posterior (Catalog #128)
    "posterior_update_locked",
    # Provenance manifest
    "provenance.json",
    # Real-video decode (NOT synthetic)
    "_decode_real_pairs",
    # β archive bytes
    "pack_archive",
    # Contest-compliant runtime
    "_write_runtime",
    "_build_archive_zip",
    # β-specific: hyperprior rate term + scales must be wired
    "rate_components",
    "scales",
    "hyper_analysis",
    "lambda_hyperprior",
)


def test_full_main_source_carries_all_required_tokens():
    """``_full_main`` source must reference all non-negotiable building blocks.

    This guards against accidental regression: if a refactor accidentally
    drops e.g. ``patch_upstream_yuv6_globally`` or ``ema.update``, this test
    fires before dispatch wastes GPU.
    """
    src_path = REPO_ROOT / "experiments" / "train_substrate_balle_renderer.py"
    src = src_path.read_text(encoding="utf-8")
    idx = src.find("def _full_main")
    assert idx >= 0, "_full_main definition not found"
    body = src[idx:]
    missing = [tok for tok in _FULL_MAIN_REQUIRED_TOKENS if tok not in body]
    assert not missing, (
        "_full_main missing required non-negotiable tokens: "
        + ", ".join(missing)
    )


def test_full_main_uses_ema_decay_0_997_default():
    """The argparse default for ``--ema-decay`` must be 0.997 per CLAUDE.md."""
    src_path = REPO_ROOT / "experiments" / "train_substrate_balle_renderer.py"
    src = src_path.read_text(encoding="utf-8")
    assert "default=0.997" in src or "0.997" in src, (
        "EMA decay 0.997 not visible in source"
    )


def test_full_main_refuses_eval_roundtrip_disabled():
    """No place in _full_main may pass ``apply_eval_roundtrip=False``."""
    src_path = REPO_ROOT / "experiments" / "train_substrate_balle_renderer.py"
    src = src_path.read_text(encoding="utf-8")
    assert "apply_eval_roundtrip=False" not in src


def test_full_main_does_not_call_make_synthetic_outside_smoke():
    """Per Catalog #114: ``make_synthetic_*`` may only appear inside smoke."""
    src_path = REPO_ROOT / "experiments" / "train_substrate_balle_renderer.py"
    src = src_path.read_text(encoding="utf-8")
    assert "make_synthetic_pair_batch" not in src
    full_main_start = src.find("def _full_main")
    full_main_end = src.find("\ndef ", full_main_start + 1)
    full_main_body = src[full_main_start:full_main_end]
    assert "make_synthetic" not in full_main_body


def test_full_main_references_beta_substrate_not_alpha():
    """The β trainer must import from ``tac.substrates.balle_renderer.*``,
    not from α's ``tac.substrates.sane_hnerv.*``. Guards against copy-paste
    regression where the trainer accidentally lands α's substrate twice.
    """
    src_path = REPO_ROOT / "experiments" / "train_substrate_balle_renderer.py"
    src = src_path.read_text(encoding="utf-8")
    assert "tac.substrates.balle_renderer.architecture" in src
    assert "tac.substrates.balle_renderer.archive" in src
    assert "tac.substrates.balle_renderer.score_aware_loss" in src
    # Alpha's modules must NOT be imported anywhere in this trainer.
    assert "from tac.substrates.sane_hnerv" not in src, (
        "beta trainer must not import from alpha's sane_hnerv package"
    )


# ---------------------------------------------------------------------------
# 9. CLI smoke (subprocess; full process surface)
# ---------------------------------------------------------------------------

def test_cli_smoke_subprocess_runs_to_completion(tmp_path):
    """Running the trainer as a subprocess in --smoke mode succeeds (<60s)."""
    py = sys.executable
    cmd = [
        py,
        str(REPO_ROOT / "experiments" / "train_substrate_balle_renderer.py"),
        "--video-path", str(REPO_ROOT / "upstream" / "videos" / "0.mkv"),
        "--output-dir", str(tmp_path / "cli_smoke"),
        "--epochs", "2",
        "--device", "cpu",
        "--smoke",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, (
        f"smoke subprocess rc={res.returncode}; stderr=\n{res.stderr[:2000]}"
    )
    assert (tmp_path / "cli_smoke" / "smoke_checkpoint.pt").is_file()
