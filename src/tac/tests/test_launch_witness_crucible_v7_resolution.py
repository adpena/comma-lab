"""crucible_v7 LAUNCH-PATH RESOLUTION — the seam the per-builder tests could not see.

The v7 builder/compile suite (``test_crucible_v7_config.py``) proves the TypedWitnessConfig compiles
and its manifests are shaped right. It NEVER exercised ``tools/launch_witness_run.py`` — so the
authored config was UNLAUNCHABLE (``--config crucible_v7`` rejected by argparse; ``derive_named_config``
fell through to a proven_base ``WitnessConfig`` that ``config_family`` MISLABELED proven_base). This
harness closes that gap (seal v7 r1 BLOCKER #1 + MAJOR #2 + MINOR #3):

  * ``derive_named_config('crucible_v7', ...)`` resolves to the v7 launch adapter (NOT proven_base);
  * an UNKNOWN config name RAISES (no silent fall-through — the bug class that hid #1);
  * the adapter satisfies the WHOLE launcher cfg protocol (emit adapters + BOTH provenance manifests);
  * the full dry-run gate chain PASSES on it: b-perf present, b0.5 0-naked, b0.6 manifest VERIFIED
    (for real, not WARN-no-op), wall-clock budget DERIVED & positive;
  * ``config_family`` labels it crucible_v7 (stamped into launch.sh);
  * MINOR #3: the phantom ``--tau-octave-max-dwell`` CAP row is gone — every governance KEY is emitted.

means != ends: gates a MEANS. Only a byte-closed n600 exact row < 0.19110 moves the pointer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The launcher's internal gate imports (`import schedule_provenance_gate`, `witness_memory_preflight`)
# are BARE — they assume tools/ is on sys.path (the operator invokes `python tools/launch_witness_run.py`,
# so tools/ IS sys.path[0]). Mirror that so the gate chain RUNS (not fail-opens) under pytest.
_TOOLS = str(Path(__file__).resolve().parents[3] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import launch_witness_run as L  # noqa: E402  (after the sys.path shim, like the operator's invocation)
import schedule_provenance_gate as gate  # noqa: E402
from tac.witness_autoconfig import CrucibleV7LaunchConfig  # noqa: E402
from tac.witness_dsl.curriculum_dsl import TRAINER_REL  # noqa: E402
from tac.witness_dsl.typed_config import verify_launch_manifest  # noqa: E402

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


@pytest.fixture(scope="module")
def v7_cfg():
    # small num_pairs keeps the derive fast + machine-independent; the launch PATH is what we test.
    return L.derive_named_config("crucible_v7", _GT, num_pairs=8, epochs=3000, overfit=True)


# ── (a) BLOCKER #1: name-resolution reaches v7, not the proven_base fall-through ──────
def test_derive_named_config_resolves_crucible_v7(v7_cfg):
    assert isinstance(v7_cfg, CrucibleV7LaunchConfig)
    assert v7_cfg.name == "crucible_v7"


def test_config_family_labels_crucible_v7_not_proven_base(v7_cfg):
    # the exact mislabel the fall-through produced: a proven_base WitnessConfig stamped proven_base.
    assert L.config_family(v7_cfg) == "crucible_v7"


def test_unknown_config_name_raises_not_silent_fallthrough():
    """The fail-LOUD default: an unmapped name RAISES (the silent fall-through that hid BLOCKER #1)."""
    with pytest.raises(ValueError, match="unknown config name"):
        L.derive_named_config("bogus_not_a_config", _GT, num_pairs=8, epochs=8, overfit=True)


def test_proven_base_and_all_levers_still_fall_through():
    """Regression: the two names that LEGITIMATELY use derive_config still resolve (no over-tighten)."""
    pb = L.derive_named_config("proven_base", _GT, num_pairs=8, epochs=8, overfit=True)
    al = L.derive_named_config("all_levers", _GT, num_pairs=8, epochs=8, overfit=True)
    assert L.config_family(pb) == "proven_base"
    assert L.config_family(al) == "all_levers"


def test_real_argparse_rejects_unknown_config():
    """The REAL launcher argparse enforces --config choices (SystemExit on an unknown token); v7's
    ACCEPTANCE by the same parser is proven end-to-end by test_full_dry_run_gate_chain_passes."""
    with pytest.raises(SystemExit):
        L.main(["--gt-cache", _GT, "--num-pairs", "8", "--config", "not_a_real_config", "--dry-run"])


# ── (b) MAJOR #2: the adapter satisfies the WHOLE launcher cfg protocol ───────────────
def test_adapter_exposes_full_cfg_protocol(v7_cfg):
    # emit adapters (delegated to the typed config)
    assert v7_cfg.to_command("OUT", perf_env=True).count("TAC_MLX_CUSTOM_GROUPED_BACKWARD") == 1
    assert any(f == "--num-pairs" for f, _ in v7_cfg.to_trainer_flags("OUT"))
    # the three provenance manifests the gate chain reads
    assert v7_cfg.dsl_program_manifest and v7_cfg.constants_manifest
    assert all(isinstance(v, dict) for v in v7_cfg.schedule_governance.values())
    # emit metadata
    assert v7_cfg.purpose and "crucible v7" in v7_cfg.purpose.lower()
    # 4-lever set as of the basis integration (793631e00: FEED_07a joined the sealed three).
    assert v7_cfg.dsl_levers == (
        "seg_form_unify_tau", "tail_k_warm_restart", "n323_ladder_island_homotopy",
        "FEED_07a_directional_basis_rebalance")


def test_b06_manifest_verifies_for_real(v7_cfg):
    """b0.6 DSL-authored-config gate VERIFIES (present + argv fingerprint matches) — NOT a WARN-no-op.
    This is the acceptance test for MAJOR #2: the launched object's manifest matches its emitted argv."""
    emitted = list(L._emitted_flag_names(v7_cfg, "OUT"))
    ok, detail = verify_launch_manifest(v7_cfg.dsl_program_manifest, emitted)
    assert ok, detail
    # and the gate's pure decision returns "ok" (proceed), not "warn"/"refuse".
    action, _ = L.dsl_config_gate_action(
        ok=ok, detail=detail, manifest_absent=not v7_cfg.dsl_program_manifest,
        config="crucible_v7", dry_run=False, skip=False, enforce=True, allow_rationale=None)
    assert action == "ok"


def test_b05_zero_naked_on_the_adapter(v7_cfg):
    """b0.5 schedule-provenance gate classifies the adapter's emitted argv with 0 NAKED triggers."""
    trainer_text = Path(TRAINER_REL).read_text()
    pairs = list(v7_cfg.to_trainer_flags("OUT"))
    verdicts = gate.classify_launch(
        pairs, registry=gate.schedule_when_flags(trainer_text),
        manifest_keys=set(v7_cfg.constants_manifest), governance=v7_cfg.schedule_governance,
        event_registry=gate.event_start_flags(trainer_text))
    ok, violations, table = gate.gate_report(verdicts)
    assert ok, f"NAKED triggers on the launched adapter:\n{table}"
    assert violations == []


def test_wall_clock_budget_derived_and_positive(v7_cfg):
    """The L45 wall-clock gate has a DERIVED, positive budget to check (wall-clock within budget)."""
    days = L._config_wall_clock_budget_days(v7_cfg)
    assert days is not None and days > 0.0


# ── (c) MINOR #3: the governance table describes launch.sh reality ────────────────────
def test_tau_octave_max_dwell_not_a_governance_key(v7_cfg):
    """The phantom CAP row is gone (it is TRAINER-DERIVED-INTERNAL, never emitted)."""
    assert "--tau-octave-max-dwell" not in v7_cfg.schedule_governance


def test_every_governance_key_is_an_emitted_flag(v7_cfg):
    """Truthfulness: every schedule_governance KEY is actually an emitted launch token."""
    emitted = {f for f, _ in v7_cfg.to_trainer_flags("OUT")}
    for key in v7_cfg.schedule_governance:
        assert key in emitted, f"governance declares {key} but launch.sh never emits it"


# ── (d) the full launcher dry-run gate chain end-to-end (the test the builders never wrote) ──
def test_full_dry_run_gate_chain_passes(tmp_path, capsys):
    rc = L.main([
        "--gt-cache", _GT, "--num-pairs", "8", "--epochs", "3000",
        "--config", "crucible_v7", "--out-dir", str(tmp_path),
        "--dry-run", "--skip-mem-preflight",  # skip mem so the chain is machine-RAM-independent
    ])
    out = capsys.readouterr()
    combined = out.out + out.err
    assert rc == 0, combined
    # b-perf present, b0.6 VERIFIED, dry-run terminal — the reachable dry-run chain, all green.
    assert "perf-env guard: launch.sh carries" in combined
    assert "dsl-config gate: OK" in combined and "crucible_v7" in combined
    assert "DRY-RUN" in combined
    # no gate REFUSED, and b0.5 surfaced no naked-epoch trigger.
    assert "REFUSING to launch" not in combined
    assert "NAKED primary-epoch schedule trigger" not in combined
    # launch.sh was written with the v7 identity + the ~17x perf-env prefix.
    launch_sh = (tmp_path / "launch.sh").read_text()
    assert "# tac-config-family: crucible_v7" in launch_sh
    assert "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in launch_sh
