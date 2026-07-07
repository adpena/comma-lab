"""Tests for the confound-immune-system L2 STRICT preflight gates
(``tac.confound_gates``, Catalog #397-#402).

Coverage per gate: positive (catches the anti-pattern) / negative (allows the
clean form) / waiver-respect / placeholder-waiver-rejected / edge cases / strict
raises PreflightError / warn-only returns. Plus a repo-smoke test that bounds each
gate's live-count against the real tree so a future regression that ADDS a
violation is caught.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac import confound_gates as cg
from tac.preflight import PreflightError

_LEVELSET = "experiments/train_levelset_witness_realized_through_R_mlx.py"
_BASE = "experiments/train_witness_realized_through_R_mlx.py"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _mk(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _trainer(root: Path, body: str, *, base: bool = False) -> Path:
    return _mk(root, _BASE if base else _LEVELSET, body)


def _launch(root: Path, name: str, body: str) -> Path:
    return _mk(root, f"experiments/results/{name}/launch.sh", body)


# ===========================================================================
# helper unit tests
# ===========================================================================


class TestHelpers:
    def test_rationale_ok_real(self):
        assert cg._rationale_ok("byte-identity A/B baseline")

    def test_rationale_ok_rejects_placeholder(self):
        assert not cg._rationale_ok("<rationale>")
        assert not cg._rationale_ok("<reason>")
        assert not cg._rationale_ok("< RATIONALE >")

    def test_rationale_ok_rejects_too_short(self):
        assert not cg._rationale_ok("ab")
        assert not cg._rationale_ok("   ")

    def test_waiver_present_real(self):
        assert cg._waiver_present("x = 1  # FOO_OK:a genuine reason here", "FOO_OK")

    def test_waiver_present_placeholder_rejected(self):
        assert not cg._waiver_present("x = 1  # FOO_OK:<rationale>", "FOO_OK")

    def test_waiver_present_absent(self):
        assert not cg._waiver_present("x = 1  # NOPE:reason", "FOO_OK")

    def test_repo_root_points_at_repo(self):
        assert (cg.REPO_ROOT / "src" / "tac" / "confound_gates.py").is_file()


# ===========================================================================
# Catalog #397 — spike-guard default deadlock mode
# ===========================================================================

_SPIKE_LEGACY = (
    'import argparse\n'
    'def build():\n'
    '    ap = argparse.ArgumentParser()\n'
    '    ap.add_argument("--spike-guard-mode", type=str, default="legacy",\n'
    '                    choices=("legacy", "rollback"))\n'
)
_SPIKE_ROLLBACK = _SPIKE_LEGACY.replace('default="legacy"', 'default="rollback"')


class TestSpikeGuardDefault:
    def test_positive_legacy_default(self, tmp_path):
        _trainer(tmp_path, _SPIKE_LEGACY)
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1
        assert "deadlock/degrade mode" in v[0]

    def test_negative_rollback_default(self, tmp_path):
        _trainer(tmp_path, _SPIKE_ROLLBACK)
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_positive_skip_default(self, tmp_path):
        _trainer(tmp_path, _SPIKE_LEGACY.replace('"legacy"', '"skip"'))
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_waiver_respected(self, tmp_path):
        body = _SPIKE_LEGACY.replace(
            'choices=("legacy", "rollback"))',
            'choices=("legacy", "rollback"))  # SPIKE_GUARD_DEFAULT_OK:byte-identity A/B baseline; autoconfig injects rollback',
        )
        _trainer(tmp_path, body)
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        body = _SPIKE_LEGACY.replace(
            'choices=("legacy", "rollback"))',
            'choices=("legacy", "rollback"))  # SPIKE_GUARD_DEFAULT_OK:<rationale>',
        )
        _trainer(tmp_path, body)
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_no_default_no_violation(self, tmp_path):
        body = (
            'import argparse\n'
            'def build():\n'
            '    ap = argparse.ArgumentParser()\n'
            '    ap.add_argument("--spike-guard-mode", type=str)\n'
        )
        _trainer(tmp_path, body)
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_unrelated_flag_ignored(self, tmp_path):
        body = (
            'import argparse\n'
            'def build():\n'
            '    ap = argparse.ArgumentParser()\n'
            '    ap.add_argument("--activation", type=str, default="legacy")\n'
        )
        _trainer(tmp_path, body)
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_strict_raises(self, tmp_path):
        _trainer(tmp_path, _SPIKE_LEGACY)
        with pytest.raises(PreflightError):
            cg.check_no_spike_guard_defaults_to_deadlock_mode(
                repo_root=tmp_path, strict=True, verbose=False
            )

    def test_missing_trainer_no_crash(self, tmp_path):
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=True, verbose=False
        )
        assert v == []

    def test_syntax_error_skipped(self, tmp_path):
        _trainer(tmp_path, "def broken(:\n")
        v = cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []


# ===========================================================================
# Catalog #398 — reject-filter accepted-only append needs re-arm (structural)
# ===========================================================================

_REJECT_NO_REARM = (
    "def run_train():\n"
    "    recent = []\n"
    "    for b in batches:\n"
    "        skip = b > 5 * median\n"
    "        if skip:\n"
    "            n_skips += 1\n"
    "        else:\n"
    "            opt.update()\n"
    "            recent.append(b)\n"
)
_REJECT_WITH_CLEAR = _REJECT_NO_REARM + (
    "        if n_skips > 20:\n"
    "            recent.clear()  # deadlock-free re-arm\n"
)
_REJECT_WITH_ROLLBACK = _REJECT_NO_REARM.replace(
    "            n_skips += 1\n",
    "            n_skips += 1\n            rollback_to_last_good()\n",
)


class TestRejectFilterRearm:
    def test_positive_accepted_only_no_rearm(self, tmp_path):
        _trainer(tmp_path, _REJECT_NO_REARM)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1
        assert "accepted-only" in v[0]

    def test_negative_has_clear_rearm(self, tmp_path):
        _trainer(tmp_path, _REJECT_WITH_CLEAR)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_negative_has_rollback_rearm(self, tmp_path):
        _trainer(tmp_path, _REJECT_WITH_ROLLBACK)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_generic_token_far_from_append_does_not_clear(self, tmp_path):
        # A generic re-arm token (.clear()) on an UNRELATED list, far (>proximity) from the
        # accepted-only append, must NOT falsely clear the gate (the tightened miss surface).
        pad = "".join(f"    _x{i} = {i}\n" for i in range(cg._REARM_PROXIMITY_LINES + 5))
        body = _REJECT_NO_REARM + pad + "    unrelated.clear()\n"
        _trainer(tmp_path, body)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1, v

    def test_generic_token_near_append_clears(self, tmp_path):
        # The SAME generic token, but within proximity of the accepted-only append, DOES clear it.
        body = _REJECT_NO_REARM + (
            "        if n_skips > 20:\n"
            "            recent.clear()\n"  # near the append -> genuine re-arm
        )
        _trainer(tmp_path, body)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_specific_token_far_still_clears(self, tmp_path):
        # A SPECIFIC token (reanchor) far from the append still clears (unambiguous re-arm intent).
        pad = "".join(f"    _x{i} = {i}\n" for i in range(cg._REARM_PROXIMITY_LINES + 5))
        body = _REJECT_NO_REARM + pad + "    reanchor_median()\n"
        _trainer(tmp_path, body)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_negation_shape_accepted_only(self, tmp_path):
        body = (
            "def run_train():\n"
            "    recent_losses = []\n"
            "    for b in batches:\n"
            "        if not (spiked or nonfinite):\n"
            "            recent_losses.append(b)\n"
        )
        _trainer(tmp_path, body)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_unconditional_append_not_flagged(self, tmp_path):
        # Append not guarded by any spike/skip test -> not accepted-only.
        body = (
            "def run_train():\n"
            "    recent = []\n"
            "    for b in batches:\n"
            "        recent.append(b)\n"
        )
        _trainer(tmp_path, body)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_non_reference_window_name_ignored(self, tmp_path):
        # ep_gnorms is not a reference-window name.
        body = (
            "def run_train():\n"
            "    ep_gnorms = []\n"
            "    for b in batches:\n"
            "        if not skip:\n"
            "            ep_gnorms.append(b)\n"
        )
        _trainer(tmp_path, body)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_waiver_respected(self, tmp_path):
        body = _REJECT_NO_REARM.replace(
            "            recent.append(b)\n",
            "            recent.append(b)  # REJECT_FILTER_REARM_OK:bounded window, EoS self-stabilizes\n",
        )
        _trainer(tmp_path, body)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        body = _REJECT_NO_REARM.replace(
            "            recent.append(b)\n",
            "            recent.append(b)  # REJECT_FILTER_REARM_OK:<reason>\n",
        )
        _trainer(tmp_path, body)
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_strict_raises(self, tmp_path):
        _trainer(tmp_path, _REJECT_NO_REARM)
        with pytest.raises(PreflightError):
            cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
                repo_root=tmp_path, strict=True, verbose=False
            )

    def test_real_repo_both_trainers_have_cure(self):
        # Both live trainers already carry a re-arm cure (levelset rollback,
        # base recent.clear()); the gate must be at live-count 0.
        v = cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            strict=False, verbose=False
        )
        assert v == [], f"unexpected live violations: {v}"


# ===========================================================================
# Catalog #399 — duplicate long flags in launch.sh
# ===========================================================================

_LAUNCH_DUP = (
    "#!/bin/bash\n"
    "python train.py \\\n"
    "  --epochs 1000 \\\n"
    "  --lr 0.01 \\\n"
    "  --tau-start 300 \\\n"
    "  --tau-start 400\n"
)
_LAUNCH_CLEAN = (
    "#!/bin/bash\n"
    "python train.py \\\n"
    "  --epochs 1000 \\\n"
    "  --lr 0.01 \\\n"
    "  --tau-start 300\n"
)


class TestDuplicateLongFlags:
    def test_positive_duplicate(self, tmp_path):
        _launch(tmp_path, "run1", _LAUNCH_DUP)
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1
        assert "--tau-start" in v[0]

    def test_negative_clean(self, tmp_path):
        _launch(tmp_path, "run1", _LAUNCH_CLEAN)
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_negation_flag_distinct(self, tmp_path):
        # --foo and --no-foo are distinct tokens; not a duplicate.
        _launch(tmp_path, "run1", "python t.py --foo --no-foo\n")
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_equals_form_duplicate(self, tmp_path):
        _launch(tmp_path, "run1", "python t.py --lr=0.01 --lr=0.02\n")
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1
        assert "--lr" in v[0]

    def test_comment_example_not_counted(self, tmp_path):
        body = "#!/bin/bash\n# example: --lr 0.01 --lr 0.02\npython t.py --lr 0.01\n"
        _launch(tmp_path, "run1", body)
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_waiver_respected(self, tmp_path):
        body = _LAUNCH_DUP + "# DUP_FLAG_OK:action=append flag intentionally repeated\n"
        _launch(tmp_path, "run1", body)
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        body = _LAUNCH_DUP + "# DUP_FLAG_OK:<rationale>\n"
        _launch(tmp_path, "run1", body)
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_multiple_launches(self, tmp_path):
        _launch(tmp_path, "run1", _LAUNCH_DUP)
        _launch(tmp_path, "run2", _LAUNCH_CLEAN)
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_strict_raises(self, tmp_path):
        _launch(tmp_path, "run1", _LAUNCH_DUP)
        with pytest.raises(PreflightError):
            cg.check_no_duplicate_long_flags_in_launch(
                repo_root=tmp_path, strict=True, verbose=False
            )

    def test_no_results_dir_no_crash(self, tmp_path):
        v = cg.check_no_duplicate_long_flags_in_launch(
            repo_root=tmp_path, strict=True, verbose=False
        )
        assert v == []


# ===========================================================================
# Catalog #400 — resume palliative flags imply warm-start-weights-only
# ===========================================================================

_LAUNCH_PALLIATIVE_NO_WS = (
    "#!/bin/bash\n"
    "python train.py \\\n"
    "  --resume-from ckpt.npz \\\n"
    "  --resume-clear-spike-guard \\\n"
    "  --resume-allow-lever-drift\n"
)
_LAUNCH_PALLIATIVE_WS = _LAUNCH_PALLIATIVE_NO_WS + "  --warm-start-weights-only\n"


class TestResumePalliativeWarmStart:
    def test_positive_palliative_without_ws(self, tmp_path):
        _launch(tmp_path, "run1", _LAUNCH_PALLIATIVE_NO_WS)
        v = cg.check_resume_palliative_flags_imply_warm_start(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1
        assert "warm-start-weights-only" in v[0]

    def test_negative_with_ws(self, tmp_path):
        _launch(tmp_path, "run1", _LAUNCH_PALLIATIVE_WS)
        v = cg.check_resume_palliative_flags_imply_warm_start(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_negative_no_palliative(self, tmp_path):
        _launch(tmp_path, "run1", "python train.py --resume-from ckpt.npz\n")
        v = cg.check_resume_palliative_flags_imply_warm_start(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_negative_palliative_but_no_resume(self, tmp_path):
        # Palliative flag but no resume-restore -> no stale moments to poison.
        _launch(tmp_path, "run1", "python train.py --resume-allow-lever-drift\n")
        v = cg.check_resume_palliative_flags_imply_warm_start(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_only_clear_spike_guard(self, tmp_path):
        body = "python train.py --resume-from c.npz --resume-clear-spike-guard\n"
        _launch(tmp_path, "run1", body)
        v = cg.check_resume_palliative_flags_imply_warm_start(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_waiver_respected(self, tmp_path):
        body = _LAUNCH_PALLIATIVE_NO_WS + "# RESUME_PALLIATIVE_OK:deliberate opt-moment carry, verified geometry unchanged\n"
        _launch(tmp_path, "run1", body)
        v = cg.check_resume_palliative_flags_imply_warm_start(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        body = _LAUNCH_PALLIATIVE_NO_WS + "# RESUME_PALLIATIVE_OK:<reason>\n"
        _launch(tmp_path, "run1", body)
        v = cg.check_resume_palliative_flags_imply_warm_start(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_strict_raises(self, tmp_path):
        _launch(tmp_path, "run1", _LAUNCH_PALLIATIVE_NO_WS)
        with pytest.raises(PreflightError):
            cg.check_resume_palliative_flags_imply_warm_start(
                repo_root=tmp_path, strict=True, verbose=False
            )

    def test_no_results_dir_no_crash(self, tmp_path):
        v = cg.check_resume_palliative_flags_imply_warm_start(
            repo_root=tmp_path, strict=True, verbose=False
        )
        assert v == []


# ===========================================================================
# Catalog #401 — verdict-pairs default is n600 (== 0)
# ===========================================================================

_VP_24 = (
    'import argparse\n'
    'def build():\n'
    '    ap = argparse.ArgumentParser()\n'
    '    ap.add_argument("--verdict-pairs", type=int, default=24)\n'
)
_VP_0 = _VP_24.replace("default=24", "default=0")


class TestVerdictPairsDefault:
    def test_positive_subset_default(self, tmp_path):
        _trainer(tmp_path, _VP_24)
        v = cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1
        assert "non-n600 subset" in v[0]

    def test_negative_zero_default(self, tmp_path):
        _trainer(tmp_path, _VP_0)
        v = cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_other_subset_value(self, tmp_path):
        _trainer(tmp_path, _VP_24.replace("default=24", "default=120"))
        v = cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_no_default_ignored(self, tmp_path):
        body = _VP_24.replace(", default=24", "")
        _trainer(tmp_path, body)
        v = cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_waiver_respected(self, tmp_path):
        body = _VP_24.replace(
            "default=24)",
            "default=24)  # VERDICT_PAIRS_DEFAULT_OK:fast dev-loop default; launcher forces 0 at n600",
        )
        _trainer(tmp_path, body)
        v = cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        body = _VP_24.replace(
            "default=24)", "default=24)  # VERDICT_PAIRS_DEFAULT_OK:<rationale>"
        )
        _trainer(tmp_path, body)
        v = cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_strict_raises(self, tmp_path):
        _trainer(tmp_path, _VP_24)
        with pytest.raises(PreflightError):
            cg.check_verdict_pairs_default_is_n600(
                repo_root=tmp_path, strict=True, verbose=False
            )

    def test_unrelated_flag_ignored(self, tmp_path):
        body = _VP_24.replace('"--verdict-pairs"', '"--num-pairs"')
        _trainer(tmp_path, body)
        v = cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_missing_trainer_no_crash(self, tmp_path):
        v = cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=True, verbose=False
        )
        assert v == []


# ===========================================================================
# Catalog #402 — telemetry verdict/loss_terms rows carry liveness
# ===========================================================================

_EMIT_NO_LIVENESS = (
    'import json\n'
    'def emit(ep, d_seg):\n'
    '    print(json.dumps({"stage": "verdict", "epoch": ep,\n'
    '                      "d_seg": d_seg}))\n'
)
_EMIT_WITH_LIVENESS = (
    'import json\n'
    'def emit(ep, d_seg, accepted_frac):\n'
    '    print(json.dumps({"stage": "verdict", "epoch": ep,\n'
    '                      "d_seg": d_seg, "accepted_frac": accepted_frac}))\n'
)


class TestTelemetryLiveness:
    def test_positive_no_liveness(self, tmp_path):
        _trainer(tmp_path, _EMIT_NO_LIVENESS)
        v = cg.check_telemetry_verdict_rows_carry_liveness(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1
        assert "no liveness field" in v[0]

    def test_negative_with_accepted_frac(self, tmp_path):
        _trainer(tmp_path, _EMIT_WITH_LIVENESS)
        v = cg.check_telemetry_verdict_rows_carry_liveness(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_loss_terms_with_spike_skipped_ok(self, tmp_path):
        body = (
            'import json\n'
            'def emit(ep, skipped):\n'
            '    row = {"stage": "loss_terms", "ep": ep}\n'
            '    row["spike_skipped"] = bool(skipped)\n'
            '    print(json.dumps(row))\n'
        )
        _trainer(tmp_path, body)
        v = cg.check_telemetry_verdict_rows_carry_liveness(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_frozen_epoch_token_ok(self, tmp_path):
        body = _EMIT_NO_LIVENESS.replace(
            '"d_seg": d_seg}))', '"d_seg": d_seg, "frozen_epoch": frz}))'
        )
        _trainer(tmp_path, body)
        v = cg.check_telemetry_verdict_rows_carry_liveness(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_non_stateful_stage_ignored(self, tmp_path):
        body = 'import json\ndef e():\n    print(json.dumps({"stage": "spike_skip", "ep": 1}))\n'
        _trainer(tmp_path, body)
        v = cg.check_telemetry_verdict_rows_carry_liveness(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_waiver_respected(self, tmp_path):
        body = _EMIT_NO_LIVENESS.replace(
            '"d_seg": d_seg}))',
            '"d_seg": d_seg}))  # TELEMETRY_LIVENESS_OK:v0 baseline emitter, always live',
        )
        _trainer(tmp_path, body)
        v = cg.check_telemetry_verdict_rows_carry_liveness(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert v == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        body = _EMIT_NO_LIVENESS.replace(
            '"d_seg": d_seg}))',
            '"d_seg": d_seg}))  # TELEMETRY_LIVENESS_OK:<rationale>',
        )
        _trainer(tmp_path, body)
        v = cg.check_telemetry_verdict_rows_carry_liveness(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(v) == 1

    def test_strict_raises(self, tmp_path):
        _trainer(tmp_path, _EMIT_NO_LIVENESS)
        with pytest.raises(PreflightError):
            cg.check_telemetry_verdict_rows_carry_liveness(
                repo_root=tmp_path, strict=True, verbose=False
            )

    def test_missing_trainer_no_crash(self, tmp_path):
        v = cg.check_telemetry_verdict_rows_carry_liveness(
            repo_root=tmp_path, strict=True, verbose=False
        )
        assert v == []


# ===========================================================================
# module + real-repo smoke
# ===========================================================================


class TestModule:
    def test_all_gates_registered(self):
        assert len(cg.CONFOUND_GATES) == 7
        names = {fn.__name__ for fn in cg.CONFOUND_GATES}
        assert names == {
            "check_no_spike_guard_defaults_to_deadlock_mode",
            "check_reject_filter_updates_reference_from_accepted_only_has_rearm",
            "check_no_duplicate_long_flags_in_launch",
            "check_resume_palliative_flags_imply_warm_start",
            "check_verdict_pairs_default_is_n600",
            "check_telemetry_verdict_rows_carry_liveness",
            "check_levelset_hosc_requires_beta_end",
        }

    @pytest.mark.parametrize("fn", cg.CONFOUND_GATES, ids=lambda f: f.__name__)
    def test_real_repo_warn_only_never_raises(self, fn):
        # Warn-only default (strict=False) must never raise, whatever the live
        # tree looks like.
        out = fn(strict=False, verbose=False)
        assert isinstance(out, list)

    @pytest.mark.parametrize("fn", cg.CONFOUND_GATES, ids=lambda f: f.__name__)
    def test_real_repo_live_count_bounded(self, fn):
        # Bound each gate's live-count so a regression that ADDS a violation is
        # caught. Bounds reflect the sibling-owned trainer/launcher fix state at
        # landing (warn-only); tighten as strict-flips land.
        bounds = {
            # #397 / #401 STRICT-FLIPPED 2026-07-06: live-count 0 (trainer defaults fixed).
            "check_no_spike_guard_defaults_to_deadlock_mode": 0,
            "check_reject_filter_updates_reference_from_accepted_only_has_rearm": 0,
            "check_no_duplicate_long_flags_in_launch": 40,
            "check_resume_palliative_flags_imply_warm_start": 40,
            "check_verdict_pairs_default_is_n600": 0,
            "check_telemetry_verdict_rows_carry_liveness": 12,
            "check_levelset_hosc_requires_beta_end": 8,  # historical launch.sh artifacts (append-only)
        }
        v = fn(strict=False, verbose=False)
        assert len(v) <= bounds[fn.__name__], f"{fn.__name__} live-count grew: {v[:3]}"
