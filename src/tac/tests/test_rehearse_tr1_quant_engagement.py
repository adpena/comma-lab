# SPDX-License-Identifier: MIT
"""ddm_gh1 #828 — the R6/E1 TR1 rehearsal gate must not emit a FALSE ``BLOCKED`` verdict.

Bug class (false refusal): ``_quant_engaged`` is a PLAIN BOOL on the TR1 module — deliberately
not an ``mx.array``, so it never enters the MLX trainable traversal and is therefore ABSENT from
the checkpoint. A freshly-built module starts at ``cfg.token_quant_anneal != "at_knee"``. The
trainer re-engages it on resume; ``tools/rehearse_ddm_tr1_runtime.py::_mlx_reference`` did not.
So on any ``at_knee`` checkpoint the reference rendered UNQUANTIZED float tokens while the
receiver (``compile_archive_from_checkpoint`` -> ``_token_codes``) quantizes UNCONDITIONALLY, the
token gate measured exactly one half lattice step, and the tool returned
``BLOCKED_DEPLOYMENT_BACKEND_PARITY`` on a candidate that was fine.

MEASURED on a real post-knee ``at_knee`` checkpoint (``ddm_b4s_20260731/rejected/
window_01_preamendment_r6only_retired/checkpoints/intra_seg_trunk_tau_ep00644.npz``, levels=16):
``token_max_abs`` 0.06666672 (= 1.0000008 x the 1/15 half-step) -> **0.0 exactly**, on BOTH kept
and dropped cells. 230 checkpoints on the fleet carried ``at_knee`` + a post-knee stage at the
time of the fix. Advisory axis; no score claim.

These tests are hermetic (a tiny synthetic checkpoint under ``tmp_path``); they never touch the
VertigoDataTier custody paths, never take a scorer slot, and never render n600.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

TR = pytest.importorskip("experiments.train_tr1_partition_renderer_mlx")
mx = pytest.importorskip("mlx.core")

rehearse_mod = pytest.importorskip("tools.rehearse_ddm_tr1_runtime")


def _cfg(quant_anneal: str):
    """A tiny lotto TR1 config (D=16 => 24x32 grid, width 8) — cheap to build and render."""
    return TR.TR1Config(
        variant="lotto", num_pairs=2, grid_downsample=16, code_width=2, renderer_width=8,
        token_quant_levels=16, seed=0, lotto_seed=118, lotto_mask_density_init=0.5,
        seg_form_start="ce", w_seg=100.0, lr=2e-3, batch_pairs=1, epochs=2, gate_every=1,
        ema_decay=0.99, ema_decay_provenance="test", token_temporal_mode="shared_base",
        token_ste="round", class_weight_lane=1.0, margin_target=1.0,
        token_quant_anneal=quant_anneal)


def _write_checkpoint(tmp_path, *, quant_anneal: str, stage: str):
    """Write a synthetic checkpoint whose EMA token field is deliberately OFF-lattice."""
    cfg = _cfg(quant_anneal)
    model = TR.build_module(cfg)
    ema = {k: mx.array(np.asarray(v)) for k, v in _flat(model)}
    rng = np.random.default_rng(11)
    for key in ("tokens_base", "tokens_delta"):
        if key in ema:
            shape = tuple(int(d) for d in ema[key].shape)
            # Off-lattice by construction: uniform in (-0.4, 0.4) is a.s. not a k/15 grid point.
            ema[key] = mx.array((rng.random(shape) * 0.8 - 0.4).astype(np.float32))
    path = tmp_path / "ckpt.npz"
    payload = {f"ema::{k}": np.asarray(v) for k, v in ema.items()}
    payload["meta::epoch"] = np.array([7], dtype=np.int64)
    meta = json.dumps({"stage": stage, "cfg": _asdict(cfg), "config_hash": cfg.config_hash(),
                       "telemetry_tail": []}).encode()
    payload["meta::json"] = np.frombuffer(meta, dtype=np.uint8)
    np.savez(path, **payload)
    return path, cfg


def _flat(model):
    from mlx.utils import tree_flatten

    return tree_flatten(model.trainable_parameters())


def _asdict(cfg):
    from dataclasses import asdict

    return asdict(cfg)


def _on_lattice(values: np.ndarray, levels: int) -> np.ndarray:
    """True where a value sits exactly on the shipped k/(levels-1) lattice in [-1, 1]."""
    scale = float(levels - 1)
    codes = np.rint((values + 1.0) * 0.5 * scale)
    return np.abs(((codes / scale) * 2.0 - 1.0) - values) <= 1e-6


@pytest.mark.parametrize("stage", ["seg_trunk_tau", "seg_trunk_ce"])
def test_reference_tokens_sit_on_the_shipped_lattice_for_at_knee(tmp_path, stage):
    """THE REGRESSION. Pre-fix the reference returned raw float tokens for an ``at_knee``
    checkpoint (``_quant_engaged`` False at build) and this assertion fails."""
    path, cfg = _write_checkpoint(tmp_path, quant_anneal="at_knee", stage=stage)
    tokens, _src, _dep, _sha, engagement = rehearse_mod._mlx_reference(path, [0])
    assert engagement["reference_quant_engaged"] is True
    assert engagement["build_default_quant_engaged"] is False  # the latent defect's precondition
    assert np.all(_on_lattice(tokens[0], cfg.token_quant_levels)), (
        "reference tokens are OFF the shipped lattice => the gate compares an unquantized "
        "reference to a quantized receiver and emits a FALSE BLOCKED verdict (#828)"
    )


def test_reference_matches_the_receiver_exactly_for_at_knee(tmp_path):
    """End-to-end: the receiver's decoded grid and the MLX reference must agree EXACTLY.

    This is the exact quantity the gate thresholds (``token_max_abs == 0.0``). Pre-fix it equals
    one half lattice step (1/(levels-1)); post-fix it is 0.0.
    """
    runtime = pytest.importorskip("tac.optimization.ddm_tr1_runtime")
    path, cfg = _write_checkpoint(tmp_path, quant_anneal="at_knee", stage="seg_trunk_tau")
    parsed = runtime.parse_archive(runtime.compile_archive_from_checkpoint(path).archive_bytes)
    decoded = runtime.decode_token_grid(parsed.packet, 0)
    tokens, _src, _dep, _sha, _eng = rehearse_mod._mlx_reference(path, [0])
    token_max_abs = float(np.abs(decoded - tokens[0]).max())
    assert token_max_abs == 0.0, (
        f"token_max_abs={token_max_abs} (half lattice step is "
        f"{1.0 / (cfg.token_quant_levels - 1)}) => FALSE BLOCKED_DEPLOYMENT_BACKEND_PARITY"
    )


def test_off_mode_checkpoint_is_unaffected(tmp_path):
    """No regression for the ``off`` (STE-from-birth) lineage: it already engaged at build, and
    the receipt records that the trainer's own resume rule agrees with the reference."""
    path, cfg = _write_checkpoint(tmp_path, quant_anneal="off", stage="seg_trunk_tau")
    tokens, _src, _dep, _sha, engagement = rehearse_mod._mlx_reference(path, [0])
    assert engagement["build_default_quant_engaged"] is True
    assert engagement["trainer_rule_agrees_with_reference"] is True
    assert np.all(_on_lattice(tokens[0], cfg.token_quant_levels))


def test_engagement_provenance_is_emitted_not_silent(tmp_path):
    """A silently-corrected instrument is the sister bug. The receipt must carry the basis so a
    reader can see WHY the reference is on the lattice."""
    path, _cfg_unused = _write_checkpoint(tmp_path, quant_anneal="at_knee", stage="seg_trunk_ce")
    _t, _s, _d, _sha, engagement = rehearse_mod._mlx_reference(path, [0])
    assert engagement["basis"] == "receiver_contract_compile_archive_always_quantizes"
    assert engagement["checkpoint_stage"] == "seg_trunk_ce"
    assert engagement["token_quant_anneal"] == "at_knee"
    # Pre-knee is the ONE case where the trainer rule and the receiver contract disagree; the
    # receipt must say so out loud rather than silently comparing mismatched states.
    assert engagement["trainer_resume_rule_would_engage"] is False
    assert engagement["trainer_rule_agrees_with_reference"] is False
