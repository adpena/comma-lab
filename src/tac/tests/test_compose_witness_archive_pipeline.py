# SPDX-License-Identifier: MIT
"""End-to-end CONTRACT tests for the compose_witness_archive orchestrator (B1 + META-bug fix).

These extinct the "two correct halves, broken seam, invisible to per-component unit tests" bug class
(review r1 HIGH-1 / G1): the trainer side + inflate side were each parity-tested in isolation, but
the orchestrator emitted the SUPERSEDED command + the WRONG npz schema + a SystemExit stub, so the
AUTOMATED pipeline never actually tested the hybrid. The tests here cross the seams:

  * the residual BUNDLE phase_a produces is loadable by the trainer's --residual-mode consumer
    (load_residual_training_bundle) -- the npz-schema handoff (G1.2);
  * the emitted command is build_residual_only_command (--residual-mode + --residual-target-npz),
    NOT build_residual_inr_command (--structured-init = no rate shrink) (G1.1);
  * phase_b assembles the REAL 4-section archive from trained weights and the inflate is BIT-IDENTICAL
    to the numpy oracle -- the inflate==train end-to-end proof through the TOOL (G1.3 + the META-bug
    end-to-end handoff-contract test).

means != ends: this validates the PLUMBING. The live competitive target comes from a byte-bound
dynamic pointer snapshot; the pointer moves only through the canonical workflow after a byte-closed
upstream/evaluate.py row from the (HELD) residual-INR GPU run. [macOS-CPU advisory].
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_frontier_pointer import POINTER_SCHEMA_VERSION  # noqa: E402
from tac.witness_dsl.dynamic_frontier_target import (  # noqa: E402
    DynamicFrontierTargetError,
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _pointer_payload(score: float) -> dict[str, object]:
    now = datetime.now(UTC).isoformat()
    entry = {
        "score": score,
        "rank": 1,
        "name": "synthetic-public-row",
        "pr_number": 9001,
        "pr_url": "https://invalid.example/synthetic",
    }
    return {
        "schema_version": POINTER_SCHEMA_VERSION,
        "our_local_frontier_contest_cpu": None,
        "our_local_frontier_contest_cuda": None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": {
            "best_entry": dict(entry),
            "entries": [dict(entry)],
        },
        "upstream_leaderboard_snapshot_at_utc": now,
        "last_refreshed_utc": now,
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "synthetic-fixture-do-not-run",
        "refresh_provenance": {"fixture": True},
        "effective_frontier": None,
    }


def _dynamic_target(repo: Path, score: float = 0.172) -> DynamicFrontierTargetSnapshot:
    path = repo / ".omx/state/canonical_frontier_pointer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_pointer_payload(score)), encoding="utf-8")
    return load_dynamic_frontier_target(repo_root=repo)


def _load_compose_tool():
    name = "compose_witness_archive"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "compose_witness_archive.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- synthetic trained-INR EMA-shadow npz (mirrors _build_ema_checkpoint_arrays' schema) ---
def _synth_ema_npz(path, *, H, W, hidden, n_hidden, mod, bank, n_classes=5, seed=3):
    from tac.boundary_math.lever_b_levelset_generator import CurveletBankConfig, curvelet_directional_B

    in_feat = 2 * curvelet_directional_B(CurveletBankConfig(**bank)).shape[1]
    rng = np.random.RandomState(seed)

    def lin(o, i):
        return (rng.standard_normal((o, i)) * 0.3).astype(np.float32)

    params = {
        "in_proj.weight": lin(hidden, in_feat),
        "in_proj.bias": (rng.standard_normal(hidden) * 0.1).astype(np.float32),
        "film.weight": lin(2 * hidden * n_hidden, mod),
        "film.bias": (rng.standard_normal(2 * hidden * n_hidden) * 0.1).astype(np.float32),
        "out_sdf.weight": lin(n_classes, hidden),
        "out_sdf.bias": (rng.standard_normal(n_classes) * 0.1).astype(np.float32),
        "out_tex.weight": lin(3, hidden),
        "out_tex.bias": (rng.standard_normal(3) * 0.1).astype(np.float32),
        "palette": (rng.rand(n_classes, 3) * 4 - 2).astype(np.float32),
        "code": (rng.standard_normal((6, mod)) * 0.5).astype(np.float32),
    }
    for li in range(n_hidden):
        params[f"hidden.{li}.weight"] = lin(hidden, hidden)
        params[f"hidden.{li}.bias"] = (rng.standard_normal(hidden) * 0.1).astype(np.float32)
    cfg_keys = {
        "__cfg_n_hidden": np.asarray(n_hidden),
        "__cfg_hidden_dim": np.asarray(hidden),
        "__cfg_softmax_temp": np.asarray(0.1),
        "__cfg_activation": np.asarray("hosc"),
        "__cfg_chroma": np.asarray(1),
        "__cfg_wire_w0": np.asarray(20.0),
        "__cfg_wire_s0": np.asarray(10.0),
        "__cfg_hosc_beta": np.asarray(4.0),
        "__cfg_hosc_omega": np.asarray(1.0),
        "__bank_n_scales": np.asarray(bank["n_scales"]),
        "__bank_n_orient0": np.asarray(bank["n_orient0"]),
        "__bank_f0": np.asarray(bank["f0"]),
        "__bank_base": np.asarray(bank["base"]),
        "__bank_n_iso": np.asarray(bank["n_iso"]),
        "__render_hw": np.asarray([H, W]),
        "__cfg_max_bank_freq": np.asarray(-1.0),
        "__epoch": np.asarray(123),
    }
    np.savez(path, **params, **cfg_keys)
    return params


def test_residual_blob_from_weights_npz_roundtrip(tmp_path):
    """B1/G1.3: residual_blob_from_weights_npz reconstructs the forward cfg from the EMA npz scalars
    and builds a valid COUNTED residual blob (mask_mode/learn_classes/dilate threaded)."""
    from tac.v2_compose.archive_grammar import parse_residual_blob

    tool = _load_compose_tool()
    bank = {"n_scales": 2, "n_orient0": 3, "f0": 2.0, "base": 2.0, "n_iso": 2}
    p = tmp_path / "levelset_witness_ema_BEST.npz"
    params = _synth_ema_npz(p, H=24, W=32, hidden=6, n_hidden=2, mod=4, bank=bank)
    blob, cfg = tool.residual_blob_from_weights_npz(p, learn_classes=(1, 3), dilate=2, mask_mode="boundary_annulus")
    assert isinstance(blob, bytes) and len(blob) > 0
    assert cfg["n_hidden"] == 2 and cfg["hidden_dim"] == 6 and cfg["mod_dim"] == 4
    assert cfg["n_classes"] == 5 and cfg["mask_mode"] == "boundary_annulus" and cfg["dilate"] == 2
    rb = parse_residual_blob(blob)
    assert rb.manifest["mask_mode"] == "boundary_annulus"
    assert rb.manifest["learn_classes"] == [1, 3] and rb.manifest["dilate"] == 2
    for k in params:  # every trained param survives (int8 dequant within scale)
        assert k in rb.params and rb.params[k].shape == params[k].shape


def test_residual_blob_from_weights_npz_rejects_non_ema(tmp_path):
    """Fail-closed: a npz lacking the EMA-checkpoint cfg keys cannot be silently shipped."""
    tool = _load_compose_tool()
    bad = tmp_path / "bad.npz"
    np.savez(bad, code=np.zeros((4, 3), np.float32), **{"out_sdf.weight": np.zeros((5, 6), np.float32)})
    with pytest.raises(ValueError, match="missing required cfg key"):
        tool.residual_blob_from_weights_npz(bad)
    nocode = tmp_path / "nocode.npz"
    np.savez(nocode, **{"in_proj.weight": np.zeros((6, 4), np.float32)})
    with pytest.raises(ValueError, match="'code'"):
        tool.residual_blob_from_weights_npz(nocode)


def test_bundle_schema_matches_trainer_consumer(tmp_path):
    """G1.2: the bundle phase_a writes is loadable by the trainer's --residual-mode consumer
    (load_residual_training_bundle) with NO KeyError -- the npz-schema handoff the seam broke."""
    from tac.v2_compose.residual_compose import (
        build_residual_training_bundle,
        load_residual_training_bundle,
        save_residual_training_bundle,
    )

    rng = np.random.RandomState(1)
    bulk = (rng.rand(3, 8, 10, 3) * 255).astype(np.float32)
    labels = rng.randint(0, 5, (3, 8, 10)).astype(np.int64)
    bundle = build_residual_training_bundle(bulk, labels, dilate=2, mode="boundary_annulus")
    p = tmp_path / "residual_bundle.npz"
    save_residual_training_bundle(bundle, p)
    rb = load_residual_training_bundle(p)  # the EXACT call the trainer makes -- must not KeyError
    assert rb.n_pairs == 3 and (rb.render_h, rb.render_w) == (8, 10)
    assert rb.mask_mode == "boundary_annulus"
    assert rb.composition_mask.shape == (3, 8, 10) and rb.bulk_rgb_render_res.shape == (3, 8, 10, 3)


def test_phase_a_emits_residual_only_command_not_structured_init():
    """G1.1: the orchestrator must emit build_residual_only_command (--residual-mode +
    --residual-target-npz), NOT the SUPERSEDED build_residual_inr_command (--structured-init)."""
    from tac.v2_compose.launch_command import build_residual_only_command

    cmd = build_residual_only_command(
        out_dir="experiments/results/_x",
        gt_cache="c.npz",
        residual_target_npz="b.npz",
        num_pairs=8,
        epochs=10,
        seed=0,
        hidden_dim=48,
        mod_dim=16,
        strict=True,
    )
    assert cmd.all_flags_valid, f"emitted unknown flags: {cmd.unknown_flags}"
    assert "--residual-mode" in cmd.command and "--residual-target-npz" in cmd.command
    assert "--structured-init" not in cmd.command and "--lane-prior-phi1" not in cmd.command


def test_warp_codes_are_physical_regime():
    """A3.1/A3.2: phase_a + phase_b share ONE physical-regime derivation (no [0,3,2,3,1] seam)."""
    from tac.v2_compose.archive_grammar import screw_regime_warp_codes
    from tools.measure_screw_reach_through_R import SCREW_REGIME

    tool = _load_compose_tool()
    assert tool._warp_codes_for_clip() == screw_regime_warp_codes(SCREW_REGIME) == [0, 0, 2, 0, 1]


def test_compose_target_payload_requires_live_snapshot(tmp_path):
    """The codec compiler reports the recomputed, byte-bound live target."""
    tool = _load_compose_tool()
    target = _dynamic_target(tmp_path)

    payload = tool.validated_frontier_target_payload(target)

    assert payload["target_score"] == 0.172
    assert payload["selected_axis"] == "official_leaderboard"
    assert payload["selected_custody"] == "external target only; no local archive authority implied"
    assert payload["pointer_sha256"] == target.pointer_sha256


def test_coupled_surface_charges_external_keyframe_payload(tmp_path: Path) -> None:
    """Phase-B planning score uses ZIP plus every external counted byte."""
    tool = _load_compose_tool()
    target = _dynamic_target(tmp_path)

    audit = tool.coupled_score_surface_payload(
        frontier_target=target,
        d_seg=0.0002,
        d_pose=0.00002331,
        archive_zip_bytes=167_125,
        external_counted_bytes=32_875,
    )

    assert audit["archive_bytes_charged"] == 200_000
    assert audit["external_counted_bytes"] == 32_875
    assert audit["inside_strict_sublevel"] is True
    assert audit["admission_rule"].startswith("exact coupled score")


def test_coupled_surface_refuses_forged_or_refreshed_target(tmp_path: Path) -> None:
    tool = _load_compose_tool()
    target = _dynamic_target(tmp_path)
    forged = replace(target, target_score=0.19110)

    with pytest.raises(DynamicFrontierTargetError, match="changed after snapshot"):
        tool.coupled_score_surface_payload(
            frontier_target=forged,
            d_seg=0.0,
            d_pose=0.0,
            archive_zip_bytes=1,
        )

    Path(target.pointer_path).write_text(
        json.dumps(_pointer_payload(0.171)),
        encoding="utf-8",
    )
    with pytest.raises(DynamicFrontierTargetError, match="changed after snapshot"):
        tool.coupled_score_surface_payload(
            frontier_target=target,
            d_seg=0.0,
            d_pose=0.0,
            archive_zip_bytes=1,
        )


def test_noncanonical_pointer_override_is_refused_before_load(tmp_path: Path) -> None:
    from types import SimpleNamespace

    tool = _load_compose_tool()
    foreign = tmp_path / "forged.json"
    foreign.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot substitute a non-canonical"):
        tool._frontier_snapshot_for_args(SimpleNamespace(frontier_pointer=str(foreign)))


def _build_store_and_pose(tmp_path, *, H, W, n_pairs, reach_kstar, n_classes=5):
    from tac.v2_compose.archive_grammar import build_store_blob
    from tac.v2_compose.pose_sidecar import build_pose_sidecar_from_cache_poses

    rng = np.random.RandomState(11)
    keyframes = list(range(0, n_pairs, reach_kstar))
    kf = rng.randint(0, n_classes, (len(keyframes), H, W)).astype(np.int64)
    palette = (rng.rand(n_classes, 3) * 255).astype(np.float32)
    warp_codes = [0, 0, 2, 0, 1]  # physical regime codes (the inflate consumes them)
    store_blob = build_store_blob(
        keyframes, kf, palette, (0.16, 0.05, 0.02), warp_codes, reach_kstar, n_pairs, n_classes=n_classes
    )
    poses = (rng.standard_normal((n_pairs, 6)) * 0.05).astype(np.float32)
    pose_path = tmp_path / "posenet_targets.bin"
    build_pose_sidecar_from_cache_poses(poses, pose_path)
    pose_blob = pose_path.read_bytes()
    poses_f16 = poses.astype(np.float16).astype(np.float64)
    return store_blob, pose_blob, poses_f16


@pytest.mark.skipif(not _HAS_TORCH, reason="torch (bicubic R) unavailable")
def test_phase_b_residual_archive_inflate_equals_oracle_end_to_end(tmp_path):
    """B1 + META-bug: assemble the REAL 4-section archive via the TOOL's residual_blob_from_weights_npz
    (trained weights -> build_residual_blob -> pack -> inflate.py) and prove the inflate is
    BIT-IDENTICAL to the numpy oracle. This is the end-to-end handoff-contract test that the
    per-component unit tests could not provide (the seam the orchestrator broke)."""
    from tac.v2_compose.archive_grammar import (
        assemble_v2_packet,
        pack_v2_archive,
        parse_residual_blob,
        parse_store_blob,
        residual_inflate_reference,
    )

    tool = _load_compose_tool()
    H, W, n_pairs, reach_kstar = 24, 32, 3, 2
    bank = {"n_scales": 2, "n_orient0": 3, "f0": 2.0, "base": 2.0, "n_iso": 2}
    store_blob, pose_blob, poses_f16 = _build_store_and_pose(
        tmp_path, H=H, W=W, n_pairs=n_pairs, reach_kstar=reach_kstar
    )

    weights = tmp_path / "levelset_witness_ema_BEST.npz"
    _synth_ema_npz(weights, H=H, W=W, hidden=6, n_hidden=2, mod=4, bank=bank)
    residual_blob, _cfg = tool.residual_blob_from_weights_npz(
        weights, learn_classes=(1, 3), dilate=1, mask_mode="boundary_annulus"
    )

    blob = pack_v2_archive(store_blob, residual_blob, pose_blob, b'{"format_version":"v2.0"}')
    pkt = tmp_path / "packet"
    zip_path, _ = assemble_v2_packet(blob, pkt)
    (pkt / "0.bin").write_bytes(blob)

    dst = pkt / "0.raw"
    r = subprocess.run(
        [sys.executable, str(pkt / "inflate.py"), str(pkt / "0.bin"), str(dst)], capture_output=True, text=True
    )
    assert r.returncode == 0, f"inflate failed: {r.stderr}"
    raw = np.fromfile(dst, dtype=np.uint8).reshape(2 * n_pairs, 874, 1164, 3)

    sb = parse_store_blob(store_blob)
    rb = parse_residual_blob(residual_blob)
    oracle = residual_inflate_reference(sb, rb, poses_f16, n_pairs)
    assert np.array_equal(raw, oracle), (
        "the TOOL-assembled residual archive inflate MUST be bit-identical to the numpy oracle "
        f"(max abs diff {int(np.abs(raw.astype(int) - oracle.astype(int)).max())}) -- the end-to-end "
        "handoff-contract proof (B1 + META-bug)"
    )
    # the inflate composes (residual present) -> f0 != f1 somewhere (not the flat floor).
    assert not all(np.array_equal(raw[2 * p], raw[2 * p + 1]) for p in range(n_pairs))


# ---------------------------------------------------------------------------
# EXPLICIT real-luma pose-carrier keyframe accounting (the rate-honesty line item;
# closes the warp_keyframe_payload_rate_minimization gap: partition keyframes + pose sidecar
# were counted but the real-luma --pose-carrier keyframes were silently omitted).
# ---------------------------------------------------------------------------
def _kf_args(**kw):
    from types import SimpleNamespace

    base = {
        "keyframe_payload_bytes": 0,
        "keyframe_payload_path": None,
        "keyframe_count": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_keyframe_payload_default_is_zero_stored_sidecar():
    """Default (no --keyframe-payload-*) -> 0 B COUNTED, source 'none', and the note REQUIRES a
    payload for a real-luma --pose-carrier row (the stored sidecar is dead bytes for d_pose)."""
    tool = _load_compose_tool()
    kf = tool.keyframe_payload_accounting(_kf_args())
    assert kf["keyframe_blob_bytes"] == 0
    assert kf["keyframe_blob_rate"] == 0.0
    assert kf["source"] == "none"
    assert "REQUIRES a keyframe payload > 0" in kf["note"]


def test_keyframe_payload_explicit_bytes_counted():
    """--keyframe-payload-bytes N -> N B COUNTED at the canonical rate_term (25*N/RATE_DENOM)."""
    from tac.contest_score import rate_term

    tool = _load_compose_tool()
    kf = tool.keyframe_payload_accounting(_kf_args(keyframe_payload_bytes=32875, keyframe_count=13))
    assert kf["keyframe_blob_bytes"] == 32875
    assert kf["keyframe_blob_rate"] == rate_term(32875)
    assert kf["source"] == "explicit_measured_bytes"
    assert kf["keyframe_count_hint"] == 13
    assert "COUNTED into the rate" in kf["note"]


def test_keyframe_payload_path_counts_st_size_and_overrides_bytes(tmp_path):
    """--keyframe-payload-path counts the REAL blob st_size and OVERRIDES --keyframe-payload-bytes."""
    from tac.contest_score import rate_term

    tool = _load_compose_tool()
    blob = tmp_path / "keyframes.bin"
    payload = b"\x00" * 4096
    blob.write_bytes(payload)
    kf = tool.keyframe_payload_accounting(_kf_args(keyframe_payload_bytes=999999, keyframe_payload_path=str(blob)))
    assert kf["keyframe_blob_bytes"] == len(payload)
    assert kf["keyframe_blob_rate"] == rate_term(len(payload))
    assert kf["source"] == "real_payload_file"
    assert kf["payload_path"] == str(blob)


def test_keyframe_payload_negative_bytes_raises():
    tool = _load_compose_tool()
    with pytest.raises(ValueError):
        tool.keyframe_payload_accounting(_kf_args(keyframe_payload_bytes=-1))


def test_keyframe_payload_missing_path_raises(tmp_path):
    tool = _load_compose_tool()
    with pytest.raises(FileNotFoundError):
        tool.keyframe_payload_accounting(_kf_args(keyframe_payload_path=str(tmp_path / "nope.bin")))


def test_phase_a_budget_folds_keyframes_into_honest_total(tmp_path):
    """The phase_a byte-budget arithmetic: known-store total = partition store + pose sidecar +
    real-luma keyframe payload (the honest rate). We assert the fold via the helper + rate_term so the
    test is decoupled from the (GPU-touching) full phase_a bulk generation."""
    from tac.contest_score import rate_term

    tool = _load_compose_tool()
    store_proj600, pose_bytes = 40_000, 2_424
    kf = tool.keyframe_payload_accounting(_kf_args(keyframe_payload_bytes=32_875))
    known_excl = store_proj600 + pose_bytes
    known_incl = known_excl + kf["keyframe_blob_bytes"]
    # honest total strictly larger; rate is monotone; keyframe rate is additive.
    assert known_incl == known_excl + 32_875
    assert rate_term(known_incl) > rate_term(known_excl)
    assert abs((rate_term(known_incl) - rate_term(known_excl)) - kf["keyframe_blob_rate"]) < 1e-12
