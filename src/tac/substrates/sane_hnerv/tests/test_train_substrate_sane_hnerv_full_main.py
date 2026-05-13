"""OD-SUBSTRATE-4 wired ``_full_main`` test suite.

These tests verify the wired contract WITHOUT dispatching GPU. They cover:

* argparse correctness (every TIER_1 flag has an argparse entry; help OK)
* device gating (cuda/cpu/--smoke interactions)
* TIER_1 manifest schema (per Catalog #151)
* helper utilities (seed pin, sha256, archive bytes proxy, decoder)
* runtime emission (Catalog #146 contract; deterministic zip)
* smoke training path runs to completion on CPU
* Lagrangian computation on a small CPU batch
* NaN watchdog wiring (`_full_main` source carries the watchdog clause)
* EMA shadow save semantics (smoke fallback for state_dict pattern)

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

# Lazy import (importing the trainer triggers torch import via the smoke path,
# which is fine but expensive — keep the import in a fixture so tests that only
# need argparse don't pay the cost twice).

REPO_ROOT = Path(__file__).resolve().parents[5]
"""Repo root, navigated from ``src/tac/substrates/sane_hnerv/tests/``."""


@pytest.fixture(scope="module")
def trainer_module():
    """Import ``experiments.train_substrate_sane_hnerv`` once per module."""
    return importlib.import_module(
        "experiments.train_substrate_sane_hnerv"
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
    data = b"sane_hnerv test bytes"
    expect = hashlib.sha256(data).hexdigest()
    assert trainer_module._sha256_bytes(data) == expect


def test_utc_now_iso_format(trainer_module):
    s = trainer_module._utc_now_iso()
    # YYYY-MM-DDTHH:MM:SSZ
    assert len(s) == 20
    assert s.endswith("Z")
    assert s[10] == "T"


def test_archive_bytes_proxy_is_positive(trainer_module):
    """Closed-form archive-bytes proxy is positive and lives on model device."""
    import torch

    from tac.substrates.sane_hnerv.architecture import (
        SaneHnervConfig,
        SaneHnervSubstrate,
    )

    cfg = SaneHnervConfig(
        latent_dim=8, embed_dim=32, initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4), num_pairs=4,
        output_height=24, output_width=32, num_upsample_blocks=3,
    )
    model = SaneHnervSubstrate(cfg)
    proxy = trainer_module._archive_bytes_proxy_closed_form(model)
    assert float(proxy.item()) > 0
    assert proxy.dtype == torch.float32


# ---------------------------------------------------------------------------
# 5. Runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

def test_write_runtime_emits_three_positional_args_in_inflate_sh(
    trainer_module, tmp_path
):
    """inflate.sh template must carry $1/$2/$3 (or named equivalents)."""
    trainer_module._write_runtime(tmp_path)
    sh = (tmp_path / "inflate.sh").read_text()
    # 3-arg contract per Catalog #146
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
        assert tok not in py, f"inflate.py contains forbidden scorer token {tok}"


def test_write_runtime_inflate_py_has_per_video_loop(trainer_module, tmp_path):
    """inflate.py must iterate file_list (per Catalog #146)."""
    trainer_module._write_runtime(tmp_path)
    py = (tmp_path / "inflate.py").read_text()
    assert "splitlines()" in py
    assert "file_list_path" in py or "file_list.read_text" in py


def test_write_runtime_inflate_py_under_size_budget(trainer_module, tmp_path):
    """inflate.py wrapper itself should be small; substrate's inflate has its
    own 100-LOC budget audited separately."""
    trainer_module._write_runtime(tmp_path)
    py = (tmp_path / "inflate.py").read_text()
    nonblank = [
        ln for ln in py.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    # Aspirational budget: wrapper itself ≤ 40 LOC of non-blank/non-comment
    assert len(nonblank) <= 40, (
        f"inflate.py wrapper too large: {len(nonblank)} non-blank/non-comment "
        f"lines (budget: 40)"
    )


def test_build_archive_zip_is_deterministic(trainer_module, tmp_path):
    """Archive zip with same bytes + same submission_dir must yield identical bytes."""
    sub = tmp_path / "sub"
    trainer_module._write_runtime(sub)
    bin_bytes = b"SHV1\x01" + b"\x00" * 100  # bogus but plausible bin
    (sub / "0.bin").write_bytes(bin_bytes)
    zip_a = tmp_path / "a.zip"
    zip_b = tmp_path / "b.zip"
    trainer_module._build_archive_zip(zip_a, bin_bytes=bin_bytes, submission_dir=sub)
    trainer_module._build_archive_zip(zip_b, bin_bytes=bin_bytes, submission_dir=sub)
    assert zip_a.read_bytes() == zip_b.read_bytes()
    # Round-trip: members are present
    with zipfile.ZipFile(zip_a, "r") as zf:
        names = set(zf.namelist())
    assert "0.bin" in names
    assert "inflate.sh" in names
    assert "inflate.py" in names


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


# ---------------------------------------------------------------------------
# 7. Lagrangian smoke (no scorer; just verify the loss module's contract)
# ---------------------------------------------------------------------------

def test_score_aware_loss_runs_on_dummy_scorers():
    """Lagrangian computes a scalar on a tiny dummy-scorer setup.

    The dummies expose the upstream contract: ``preprocess_input`` (5D→4D)
    + ``forward`` (4D→logits/pose). Without ``preprocess_input``, the loss
    would crash at the scorer's stem (WWW4 dispatch bug fixed 2026-05-12).
    """
    import torch
    import torch.nn as nn

    from tac.substrates.sane_hnerv.score_aware_loss import (
        SaneHnervScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    class DummySeg(nn.Module):
        """Mimics upstream SegNet: 5D in, slice last frame, return 4D logits."""

        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            # (B, T, C, H, W) -> (B, C, H, W) using last frame, per upstream
            return x_btchw[:, -1, ...]

        def forward(self, x_bchw: torch.Tensor) -> dict[str, torch.Tensor]:
            # (B, C, H, W) -> (B, 5, h, w)
            b, _c, h, w = x_bchw.shape
            # Connect to the input so autograd has a path through the dummy
            return (
                torch.zeros(b, 5, h // 2, w // 2, dtype=x_bchw.dtype)
                + x_bchw.mean()
            )

    class DummyPose(nn.Module):
        """Mimics upstream PoseNet: 5D in, return 4D 12-channel after yuv6."""

        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            # (B, T, C, H, W) -> (B, T*6, H/2, W/2). Tiny analogue of upstream
            # rgb_to_yuv6 4:2:0 chroma subsample (mean-pool to halve spatial),
            # 6 channels per frame.
            b, t, _c, h, w = x_btchw.shape
            flat = x_btchw.reshape(b * t, 3, h, w).mean(dim=1, keepdim=True)
            # 6 channels per frame: repeat the mean across 6 (pretend yuv6)
            flat6 = flat.expand(-1, 6, -1, -1)
            # 4:2:0: avg_pool kernel=2
            flat6_sub = flat6.reshape(b * t, 6, h // 2, 2, w // 2, 2).mean(dim=(3, 5))
            return flat6_sub.reshape(b, t * 6, h // 2, w // 2)

        def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
            # (B, 12, H, W) -> (B, 12)
            return {"pose": x_bchw.flatten(2).mean(dim=2)}

    weights = ScoreAwareLossWeights(
        alpha_rate=25.0,
        beta_seg=100.0,
        gamma_pose=math.sqrt(10.0),
        pose_weight_scale=1.0,
        contest_normalizer=37545489.0,
    )
    loss_fn = SaneHnervScoreAwareLoss(
        seg_scorer=DummySeg(), pose_scorer=DummyPose(), weights=weights,
    )
    b = 2
    rgb_0 = torch.rand(b, 3, 16, 24, requires_grad=True) * 255.0
    rgb_1 = torch.rand(b, 3, 16, 24, requires_grad=True) * 255.0
    gt_0 = torch.rand(b, 3, 16, 24) * 255.0
    gt_1 = torch.rand(b, 3, 16, 24) * 255.0
    bytes_proxy = torch.tensor(100_000.0)
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # Parts must include rate/seg/pose components
    for key in ("rate_term", "seg_term", "pose_term", "loss_total"):
        assert key in parts


def test_score_aware_loss_refuses_eval_roundtrip_false():
    """Catalog #5: ``apply_eval_roundtrip=False`` is forbidden in production."""
    import torch
    import torch.nn as nn

    from tac.substrates.sane_hnerv.score_aware_loss import (
        SaneHnervScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    class _Dummy(nn.Module):
        def preprocess_input(self, x):
            # (B, T, C, H, W) -> (B, C, H, W) — slice last frame like SegNet
            return x[:, -1, ...]

        def forward(self, x):
            return x.flatten(1).mean(dim=1, keepdim=True).expand(-1, 12)

    loss_fn = SaneHnervScoreAwareLoss(
        seg_scorer=_Dummy(), pose_scorer=_Dummy(),
        weights=ScoreAwareLossWeights(),
    )
    with pytest.raises(ValueError):
        loss_fn(
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.tensor(1.0),
            apply_eval_roundtrip=False,
        )


def test_score_aware_loss_refuses_negative_noise_std():
    import torch
    import torch.nn as nn

    from tac.substrates.sane_hnerv.score_aware_loss import (
        SaneHnervScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    class _Dummy(nn.Module):
        def preprocess_input(self, x):
            # (B, T, C, H, W) -> (B, C, H, W) — slice last frame like SegNet
            return x[:, -1, ...]

        def forward(self, x):
            return x.flatten(1).mean(dim=1, keepdim=True).expand(-1, 12)

    loss_fn = SaneHnervScoreAwareLoss(
        seg_scorer=_Dummy(),
        pose_scorer=_Dummy(),
        weights=ScoreAwareLossWeights(),
    )
    with pytest.raises(ValueError, match="noise_std"):
        loss_fn(
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.rand(1, 3, 8, 8) * 255.0,
            torch.tensor(1.0),
            apply_eval_roundtrip=True,
            noise_std=-0.1,
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
    "--device", # contest_auth_eval --device cuda
    # Continual-learning posterior (Catalog #128)
    "posterior_update_locked",
    # Provenance manifest
    "provenance.json",
    # Real-video decode (NOT synthetic)
    "_decode_real_pairs",
    # 0.bin archive
    "pack_archive",
    # Contest-compliant runtime
    "_write_runtime",
    "_build_archive_zip",
)


def test_full_main_source_carries_all_required_tokens():
    """``_full_main`` source must reference all non-negotiable building blocks.

    This guards against accidental regression: if a refactor accidentally
    drops e.g. ``patch_upstream_yuv6_globally`` or ``ema.update``, this test
    fires before dispatch wastes GPU.
    """
    src_path = REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"
    src = src_path.read_text(encoding="utf-8")
    # Restrict to the _full_main body for token discipline
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
    src_path = REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"
    src = src_path.read_text(encoding="utf-8")
    assert "default=0.997" in src or 'default": "0.997"' in src or (
        "0.997" in src
    ), "EMA decay 0.997 not visible in source"


def test_full_main_refuses_eval_roundtrip_disabled():
    """No place in _full_main may pass ``apply_eval_roundtrip=False``."""
    src_path = REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"
    src = src_path.read_text(encoding="utf-8")
    # The score-aware loss's ValueError already enforces this at runtime; this
    # guards against the source itself smuggling False.
    assert "apply_eval_roundtrip=False" not in src


def test_full_main_does_not_call_make_synthetic_outside_smoke():
    """Per Catalog #114: ``make_synthetic_*`` may only appear inside smoke."""
    src_path = REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"
    src = src_path.read_text(encoding="utf-8")
    # We never call make_synthetic_pair_batch at all.
    assert "make_synthetic_pair_batch" not in src
    # _smoke_main may write synthetic targets (it doesn't use video); _full_main
    # decodes real video.
    full_main_start = src.find("def _full_main")
    full_main_end = src.find("\ndef ", full_main_start + 1)
    full_main_body = src[full_main_start:full_main_end]
    assert "make_synthetic" not in full_main_body


# ---------------------------------------------------------------------------
# 9. CLI smoke (subprocess; full process surface)
# ---------------------------------------------------------------------------

def test_cli_smoke_subprocess_runs_to_completion(tmp_path):
    """Running the trainer as a subprocess in --smoke mode succeeds (<10s)."""
    py = sys.executable
    cmd = [
        py,
        str(REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"),
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


def test_cli_help_subprocess_exits_zero():
    """``--help`` must succeed via subprocess (catches import-time errors)."""
    py = sys.executable
    cmd = [
        py,
        str(REPO_ROOT / "experiments" / "train_substrate_sane_hnerv.py"),
        "--help",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0
    assert "--video-path" in res.stdout
