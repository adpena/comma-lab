# SPDX-License-Identifier: MIT
"""Tests for the #205 witness-launch memory preflight (tools/witness_memory_preflight.py).

Guards the OOM self-protection: the projection must REFUSE the #205-original OOM config (n600,
unchunked verdict) and PASS the fixed config (n600, chunked verdict) on a 128 GiB box, and the
launch.sh parser must extract the memory-relevant flags from the \\-continued multiline body.
Constants are calibrated from the 2026-07-02 ledger's isolated micro-probes (unchunked N=600 ->
+66 GiB verdict spike; chunked vbatch 32 -> +5.6 GiB).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import witness_memory_preflight as wmp  # noqa: E402

RAM = 128.0


def test_unchunked_n600_refuses_on_128gib():
    """The #205-original OOM config (verdict-batch 0 => 600-wide verdict spike) must REFUSE."""
    p = wmp.project_peak_rss_gib(num_pairs=600, verdict_batch=0, total_ram_gib=RAM)
    assert not p.safe
    assert p.projected_peak_gib > p.safe_ceiling_gib
    # the verdict spike dominates (~0.11/pair * 600)
    assert p.verdict_transient_gib > 50.0


def test_chunked_n600_passes_on_128gib():
    """The FIX (verdict-batch 32) must PASS on a 128 GiB box (peak ~68 GiB < 89.6 ceiling)."""
    p = wmp.project_peak_rss_gib(num_pairs=600, verdict_batch=32, total_ram_gib=RAM)
    assert p.safe
    assert p.projected_peak_gib < p.safe_ceiling_gib
    assert p.verdict_transient_gib <= wmp.VERDICT_FLOOR_GIB + 0.01  # bounded to the chunked floor


def test_chunking_is_the_decisive_lever_at_n600():
    """Same config, verdict-batch 0 vs 32: the ONLY difference is the verdict spike -> flips safety."""
    unchunked = wmp.project_peak_rss_gib(num_pairs=600, verdict_batch=0, total_ram_gib=RAM)
    chunked = wmp.project_peak_rss_gib(num_pairs=600, verdict_batch=32, total_ram_gib=RAM)
    assert unchunked.cf_cache_gib == chunked.cf_cache_gib  # cf_mx_cache identical (resident floor)
    assert unchunked.gt_gib == chunked.gt_gib
    assert unchunked.projected_peak_gib - chunked.projected_peak_gib > 50.0  # ~60 GiB delta
    assert not unchunked.safe and chunked.safe


def test_verdict_batch_capped_at_num_pairs():
    """vbatch >= num_pairs behaves like the single-batch path (no cap-induced under-estimate)."""
    p = wmp.project_peak_rss_gib(num_pairs=16, verdict_batch=64, total_ram_gib=RAM)
    # eff_batch = min(64, 16) = 16 -> verdict = max(floor, 0.11*16) = floor
    assert p.verdict_transient_gib == wmp.VERDICT_FLOOR_GIB


def test_self_orient_off_shrinks_cf_cache():
    """Without self-orient the cf feats are a single shared tensor, not per-pair -> tiny cache."""
    on = wmp.project_peak_rss_gib(num_pairs=600, self_orient=True, verdict_batch=32, total_ram_gib=RAM)
    off = wmp.project_peak_rss_gib(num_pairs=600, self_orient=False, verdict_batch=32, total_ram_gib=RAM)
    assert off.cf_cache_gib < 1.0
    assert on.cf_cache_gib > 40.0
    assert off.projected_peak_gib < on.projected_peak_gib


def test_peak_scales_with_num_pairs():
    small = wmp.project_peak_rss_gib(num_pairs=64, verdict_batch=32, total_ram_gib=RAM)
    big = wmp.project_peak_rss_gib(num_pairs=600, verdict_batch=32, total_ram_gib=RAM)
    assert big.cf_cache_gib > small.cf_cache_gib
    assert big.projected_peak_gib > small.projected_peak_gib


def test_render_resolution_scales_cf_cache():
    """cf_mx_cache scales with pixel count (per-pair feats live at render resolution)."""
    base = wmp.project_peak_rss_gib(num_pairs=300, render_h=384, render_w=512, total_ram_gib=RAM)
    big = wmp.project_peak_rss_gib(num_pairs=300, render_h=768, render_w=1024, total_ram_gib=RAM)
    assert big.cf_cache_gib > base.cf_cache_gib * 3.5  # 4x pixels


def test_parse_launch_flags_multiline():
    body = (
        "#!/bin/bash\nset -euo pipefail\ncd /repo\n"
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python trainer.py \\\n"
        "  --num-pairs 600 \\\n"
        "  --render-h 384 \\\n"
        "  --render-w 512 \\\n"
        "  --self-orient \\\n"
        "  --verdict-batch 32 \\\n"
        "  --mod-dim 32 \\\n"
    )
    flags = wmp.parse_launch_flags(body)
    assert flags["num_pairs"] == 600
    assert flags["render_h"] == 384
    assert flags["render_w"] == 512
    assert flags["verdict_batch"] == 32
    assert flags["self_orient"] is True


def test_parse_launch_flags_defaults_verdict_batch_when_absent():
    """A launch.sh with NO --verdict-batch => project_from_launch_sh uses the trainer default (32)."""
    body = "python trainer.py \\\n  --num-pairs 600 \\\n  --self-orient \\\n"
    flags = wmp.parse_launch_flags(body)
    assert "verdict_batch" not in flags


def test_project_from_launch_sh_uses_default_verdict_batch(tmp_path):
    """The failed #205 launch.sh had no --verdict-batch; with the fix (default 32) it projects SAFE."""
    p = tmp_path / "launch.sh"
    p.write_text("python trainer.py \\\n  --num-pairs 600 \\\n  --self-orient \\\n")
    proj = wmp.project_from_launch_sh(p, total_ram_gib=RAM)
    assert proj.verdict_batch == wmp.DEFAULT_VERDICT_BATCH == 32
    assert proj.safe  # chunked-by-default => safe


def test_cli_strict_returns_3_on_unsafe(capsys):
    rc = wmp.main(["--num-pairs", "600", "--verdict-batch", "0", "--total-ram-gib", "128", "--strict"])
    assert rc == 3
    assert "REFUSE" in capsys.readouterr().out


def test_cli_safe_returns_0(capsys):
    rc = wmp.main(["--num-pairs", "600", "--verdict-batch", "32", "--total-ram-gib", "128", "--strict"])
    assert rc == 0
    assert "SAFE" in capsys.readouterr().out


def test_measured_constants_match_ledger():
    """Lock the calibration to the ledger's measured anchors so a silent drift is caught."""
    # unchunked N=600 verdict spike measured at +66.2 GiB -> ~0.11/pair
    assert abs(wmp.VERDICT_PER_PAIR_GIB * 600 - 66.0) < 6.0
    # chunked floor measured ~5.6 GiB
    assert 5.0 <= wmp.VERDICT_FLOOR_GIB <= 7.0
    # cf_mx_cache measured 0.070-0.072 GiB/pair @ 384x512
    assert 0.065 <= wmp.CF_PER_PAIR_GIB_REF <= 0.075


# ── C4 fix (SEAL review 2026-07-04): the --launch-sh path derives in_feat from the bank flags ────
# The FALSE-SAFE this extincts: bank-6 + self-orient => in_feat 176 => cf_mx_cache 86.4 GiB =>
# true peak 110.81 GiB REFUSE, but the in_feat-BLIND parser projected "SAFE 67.6".

_BANK4_LAUNCH_BODY = (
    "#!/bin/bash\nset -euo pipefail\ncd /repo\n"
    "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \\\n"
    "  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \\\n"
    "  --num-pairs 600 \\\n  --self-orient \\\n  --n-dir-freqs 2 \\\n  --freq-across 32 \\\n"
    "  --freq-along 4 \\\n  --max-bank-freq 64 \\\n  --verdict-batch 32 \\\n"
    "  --render-h 384 \\\n  --render-w 512 \\\n"
)
_BANK6_LAUNCH_BODY = _BANK4_LAUNCH_BODY.replace(
    "--max-bank-freq 64", "--bank-n-scales 6 --max-bank-freq 64")


def test_derive_in_feat_bank4_default_matches_ref_88():
    """#205-style flags (default bank-4, self-orient, n-dir-freqs 2) -> the measured REF 88."""
    flags = wmp.parse_launch_flags(_BANK4_LAUNCH_BODY)
    flags["self_orient"] = True
    assert wmp.derive_in_feat_from_flags(flags) == wmp.REF_IN_FEAT == 88


def test_derive_in_feat_bank6_capped_is_176():
    """bank-6 @ max-bank-freq 64 -> cols 84 (float32 cap drops 4 of the f=64 atoms) -> in_feat
    2*84 + 4*2 = 176 — the review-measured number the blind path missed."""
    flags = wmp.parse_launch_flags(_BANK6_LAUNCH_BODY)
    flags["self_orient"] = True
    assert wmp.derive_in_feat_from_flags(flags) == 176


def test_derive_in_feat_self_orient_off_drops_dir_feats():
    flags = wmp.parse_launch_flags(_BANK4_LAUNCH_BODY)
    flags["self_orient"] = False
    assert wmp.derive_in_feat_from_flags(flags) == 80  # 2*40 curvelet cols only


def test_derive_in_feat_uses_trainer_default_n_dir_freqs_when_absent():
    """self-orient with NO --n-dir-freqs -> trainer argparse default 6 -> dir_w 24."""
    assert wmp.derive_in_feat_from_flags({"self_orient": True, "max_bank_freq": 64.0}) == 80 + 24


def test_parse_launch_flags_extracts_bank_flags():
    flags = wmp.parse_launch_flags(_BANK6_LAUNCH_BODY)
    assert flags["bank_n_scales"] == 6
    assert flags["max_bank_freq"] == 64.0
    assert flags["n_dir_freqs"] == 2


def test_launch_sh_bank6_refuses_at_true_in_feat(tmp_path):
    """The C4 FALSE-SAFE regression: the bank-6 launch.sh must now project the TRUE peak
    (110.81 GiB per the review's executed number) and REFUSE on a 128 GiB box."""
    p = tmp_path / "launch.sh"
    p.write_text(_BANK6_LAUNCH_BODY)
    proj = wmp.project_from_launch_sh(p, total_ram_gib=RAM)
    assert proj.in_feat == 176
    assert proj.projected_peak_gib == 110.81
    assert not proj.safe


def test_launch_sh_bank4_safe_at_67_61(tmp_path):
    p = tmp_path / "launch.sh"
    p.write_text(_BANK4_LAUNCH_BODY)
    proj = wmp.project_from_launch_sh(p, total_ram_gib=RAM)
    assert proj.in_feat == 88
    assert proj.projected_peak_gib == 67.61
    assert proj.safe


def test_launch_sh_in_feat_override_honored(tmp_path):
    """--in-feat alongside --launch-sh must be honored (it was silently ignored pre-fix)."""
    p = tmp_path / "launch.sh"
    p.write_text(_BANK4_LAUNCH_BODY)
    proj = wmp.project_from_launch_sh(p, total_ram_gib=RAM, in_feat_override=176)
    assert proj.in_feat == 176
    assert not proj.safe


def test_cli_launch_sh_bank6_strict_returns_3(tmp_path, capsys):
    p = tmp_path / "launch.sh"
    p.write_text(_BANK6_LAUNCH_BODY)
    rc = wmp.main(["--launch-sh", str(p), "--total-ram-gib", "128", "--strict"])
    assert rc == 3
    out = capsys.readouterr().out
    assert "REFUSE" in out and "in_feat=176" in out


def test_cli_launch_sh_honors_in_feat_override(tmp_path, capsys):
    p = tmp_path / "launch.sh"
    p.write_text(_BANK4_LAUNCH_BODY)
    rc = wmp.main(["--launch-sh", str(p), "--in-feat", "176", "--total-ram-gib", "128", "--strict"])
    assert rc == 3
    assert "in_feat=176" in capsys.readouterr().out


def test_real_205_launch_sh_still_projects_safe():
    """The #205 real launch.sh (bank-4 default, verdict-batch 32) must still project SAFE ~67.6."""
    real = _REPO / "experiments/results/levelset_n600_witness_20260703T120444Z/launch.sh"
    if not real.exists():
        import pytest

        pytest.skip("#205 launch.sh not present")
    proj = wmp.project_from_launch_sh(real, total_ram_gib=RAM)
    assert proj.in_feat == 88
    assert proj.projected_peak_gib == 67.61
    assert proj.safe


# ── SYSTEM-aware admission composition (the P0 SUM-over-RAM gate) ───────────────────────────────
import system_memory_governor as gov  # noqa: E402


def _fixed_snapshot(used_gib):
    return gov.SystemMemorySnapshot(
        total_gib=128.0, available_gib=128.0 - used_gib, used_gib=used_gib, free_gib=128.0 - used_gib,
        wired_gib=0.0, compressor_gib=0.0, swap_used_gib=0.0, pressure_level=1, load1=0.0, load5=0.0,
        load15=0.0, available_primary_gib=128.0 - used_gib, closure_gib=1.0, closure_ok=True,
        cross_validated=True, discrepancy_gib=0.0, fail_safe=False)


def test_system_aware_admission_refuses_second_concurrent_run(monkeypatch):
    """R1 (~67 GB) already running; a 2nd ~67 GB run must be REFUSED (the crash)."""
    r1 = gov.TrackedJob(label="r1", pid=1, pgid=1, cmd="python train_witness.py", priority=100,
                        projected_peak_gib=67.0, current_rss_gib=67.0, paused=False,
                        throttle_eligible=True, own_group_leader=True)
    monkeypatch.setattr(gov, "read_system_memory_snapshot", lambda: _fixed_snapshot(70.0))
    monkeypatch.setattr(gov, "list_tracked_jobs", lambda: [r1])
    ctx = wmp.system_aware_admission(67.0)
    assert not ctx.decision.admit  # 70 + 0 growth + 67 = 137 > ceiling 117.76


def test_system_aware_admission_admits_lone_run(monkeypatch):
    monkeypatch.setattr(gov, "read_system_memory_snapshot", lambda: _fixed_snapshot(18.0))
    monkeypatch.setattr(gov, "list_tracked_jobs", lambda: [])
    ctx = wmp.system_aware_admission(67.0)
    assert ctx.decision.admit  # 18 + 0 + 67 = 85 < ceiling 117.76
