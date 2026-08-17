"""Tests for the confound-immune-system L2 STRICT preflight gates
(``tac.confound_gates``, Catalog #397-#402).

Coverage per gate: positive (catches the anti-pattern) / negative (allows the
clean form) / waiver-respect / placeholder-waiver-rejected / edge cases / strict
raises PreflightError / warn-only returns. Plus a repo-smoke test that bounds each
gate's live-count against the real tree so a future regression that ADDS a
violation is caught.
"""

from __future__ import annotations

import json
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
# 2026-07-15 follow-on H1/H2/H3 alarm/canary/warmup strict gates
# ===========================================================================

_PARTIAL_FREEZE_CLEAN = '''
def alarm(accepted_frac):
    if accepted_frac <= 0.02:
        return "frozen"
    elif accepted_frac < 0.5:
        return {"stage": "confound_alarm", "alarm": "partial_freeze",
                "accepted_frac": accepted_frac}
'''


class TestPartialFreezeFollowon:
    @pytest.mark.parametrize("base", (False, True))
    def test_clean_exact_band(self, tmp_path, base):
        _trainer(tmp_path, _PARTIAL_FREEZE_CLEAN, base=base)
        assert cg.check_witness_trainers_emit_partial_freeze_alarm(
            repo_root=tmp_path, strict=True, verbose=False
        ) == []

    def test_missing_alarm_refused(self, tmp_path):
        _trainer(tmp_path, "def alarm(accepted_frac):\n    return accepted_frac <= 0.02\n")
        out = cg.check_witness_trainers_emit_partial_freeze_alarm(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert len(out) == 1 and "missing typed partial_freeze" in out[0]
        with pytest.raises(PreflightError):
            cg.check_witness_trainers_emit_partial_freeze_alarm(
                repo_root=tmp_path, strict=True, verbose=False
            )

    def test_wrong_band_refused(self, tmp_path):
        _trainer(tmp_path, _PARTIAL_FREEZE_CLEAN.replace("0.5", "0.9"))
        assert cg.check_witness_trainers_emit_partial_freeze_alarm(
            repo_root=tmp_path, strict=False, verbose=False
        )


_DSEG_CANARY_CLEAN = '''
def canary_suite():
    return None
def verdict_clearance():
    return True
def _dseg_canary_telemetry_fields():
    return {"dseg_descent_canary_passed": True,
            "dseg_descent_positive_control_registered": True,
            "dseg_verdict_clearance": verdict_clearance()}
setup = {"stage": "dseg_descent_canary_setup", **_dseg_canary_telemetry_fields()}
baseline = {"stage": "verdict", **_dseg_canary_telemetry_fields()}
async_row = {"stage": "verdict", **_dseg_canary_telemetry_fields()}
sync_row = {"stage": "verdict", **_dseg_canary_telemetry_fields()}
'''


class TestDsegCanaryFollowon:
    def test_clean_all_paths(self, tmp_path):
        _trainer(tmp_path, _DSEG_CANARY_CLEAN)
        assert cg.check_witness_verdict_rows_carry_dseg_descent_canary(
            repo_root=tmp_path, strict=True, verbose=False
        ) == []

    def test_missing_sync_stamp_refused(self, tmp_path):
        _trainer(tmp_path, _DSEG_CANARY_CLEAN.replace(
            'sync_row = {"stage": "verdict", **_dseg_canary_telemetry_fields()}\n', ""
        ))
        out = cg.check_witness_verdict_rows_carry_dseg_descent_canary(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert any("baseline, async, and sync" in row for row in out)
        with pytest.raises(PreflightError):
            cg.check_witness_verdict_rows_carry_dseg_descent_canary(
                repo_root=tmp_path, strict=True, verbose=False
            )


_LIVE_GAP_CLEAN = '''
import argparse
VERDICT_LIVE_GAP_AUTO_WARMUP = -1
def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict-live-gap-every", type=int, default=-1)
def verdict_live_gap_due():
    return ema_warmup_updates(0.997)
def _verdict_live_gap_is_due(ep):
    return verdict_live_gap_due()
_live = {"ema_updates": 0}
_live["ema_updates"] += 1
async_due = _verdict_live_gap_is_due(ep)
sync_due = _verdict_live_gap_is_due(ep)
'''


class TestLiveGapWarmupFollowon:
    def test_clean_auto_warmup(self, tmp_path):
        _trainer(tmp_path, _LIVE_GAP_CLEAN)
        assert cg.check_verdict_live_gap_defaults_on_during_ema_warmup(
            repo_root=tmp_path, strict=True, verbose=False
        ) == []

    def test_default_off_refused(self, tmp_path):
        _trainer(tmp_path, _LIVE_GAP_CLEAN.replace("default=-1", "default=0"))
        out = cg.check_verdict_live_gap_defaults_on_during_ema_warmup(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert any("must default to -1" in row for row in out)
        with pytest.raises(PreflightError):
            cg.check_verdict_live_gap_defaults_on_during_ema_warmup(
                repo_root=tmp_path, strict=True, verbose=False
            )

    def test_missing_sync_predicate_refused(self, tmp_path):
        _trainer(tmp_path, _LIVE_GAP_CLEAN.replace(
            "sync_due = _verdict_live_gap_is_due(ep)\n", ""
        ))
        out = cg.check_verdict_live_gap_defaults_on_during_ema_warmup(
            repo_root=tmp_path, strict=False, verbose=False
        )
        assert any("async and sync" in row for row in out)


# ===========================================================================
# module + real-repo smoke
# ===========================================================================


class TestModule:
    def test_all_gates_registered(self):
        # 17 -> 20 on 2026-07-18: registers the two 2026-07-17 bug-class-sweep
        # gates (raw-vm basis + observer-flag exclusion; added to the tuple by
        # a sibling arm without updating this test — pre-existing failure,
        # fixed forward here) + the NAME-ANCHORED-SEARCH duplicate-SoT gate.
        # 20 -> 22 on 2026-07-31: the same fix-forward, twice more — ddm_sb2
        # (#819) added check_no_stub_lever_factories without updating this
        # count (a pre-existing failure ddm_rg5 confirmed by git-stash A/B),
        # and ddm_rg5 (#825) adds its consumer-side sister.
        # 22 -> 23 on 2026-07-31: ddm_gh1 registers the CLASS GUARD
        # (check_refusal_gates_have_live_positive_control), which consumes
        # CONFOUND_GATES to discover the refuse-capable set and therefore
        # self-registers after the catalog tuple.
        # 23 -> 25 on 2026-08-01: the same fix-forward pattern, twice more.
        # check_upstream_pin_no_content_drift was appended to the tuple by a
        # sibling arm (#836) without updating this test, so the assert was
        # ALREADY failing at 24 before ddm_vc1 touched it — DERIVED from the
        # tuple's tail on main, where that gate is the last append and this
        # name set does not list it. ddm_vc1 (#842) adds the second:
        # check_verdict_surfaces_report_examined_count.
        # 25 -> 26 on 2026-08-03: ddm_op2 (OP2-1) appends
        # check_checkpoint_saves_do_not_silently_drop_optimizer_state, the second
        # landing for the silently-dropped-optimizer-state class (all six trainer
        # save_checkpoint callsites passed the bare `opt_state_flat={}` literal, so
        # every resume was a full Adam moment reset -- #824 arm B, MEASURED at
        # 16.167 epochs of re-convergence per boundary). It lands STRICT at
        # live-count 0 WITH a registered positive control, so the uncovered
        # ceiling does not move.
        # 26 -> 28 on 2026-08-03: ddm_lr2 appends the two landings for the
        # instruments-point-at-a-retired-vehicle class —
        # check_lever_module_declares_its_trainer (a DESIGNED-STUB verdict may not
        # rest on a trainer binding nobody declared; the mechanism that filed 8
        # TR1-targeted factories under the retired vehicle) and
        # check_no_asserted_packet_ir_readiness_fields (a readiness field DECLARED
        # instead of DECIDED, whose value is summed into a total readers take as
        # checked). Both land STRICT at live-count 0 WITH registered positive
        # controls, so the uncovered ceiling does not move.
        # 28 -> 29 on 2026-08-15: ddm_gb1 (#1073) appends
        # check_throttle_rearms_and_admission_reconciles — ONE gate for the memory
        # governor's two stale-reference anti-patterns (throttle resume gated solely
        # on the sticky OS pressure level; admission counting an unreconciled
        # registry). Warn-only for one cycle at MEASURED live-count 0, with two
        # executed positive controls, so the uncovered ceiling does not move.
        # 29 -> 31 on 2026-08-16: the same fix-forward pattern, twice more.
        # check_no_row_contract_error_quarantines_the_ledger (#1081) was appended
        # to the tuple by a sibling arm without updating this test, so the assert
        # was ALREADY failing at 30 before ddm_pl1 touched it -- and its absence
        # from the `bounds` map below made that parametrised case a KeyError,
        # which masks a real live-count regression rather than reporting one.
        # ddm_pl1 adds the second: check_no_bulk_write_strands_the_ready_record,
        # the two-landing gate for the ddm_lr1/A2 payload loss (a ready `result`
        # stranded behind a fragile checkpoint save). It lands WARN-ONLY at a
        # MEASURED live count of 10 over 11,016 modules, WITH a registered
        # positive control, so the uncovered ceiling does not move.
        assert len(cg.CONFOUND_GATES) == 31
        names = {fn.__name__ for fn in cg.CONFOUND_GATES}
        assert names == {
            "check_no_spike_guard_defaults_to_deadlock_mode",
            "check_reject_filter_updates_reference_from_accepted_only_has_rearm",
            "check_no_duplicate_long_flags_in_launch",
            "check_resume_palliative_flags_imply_warm_start",
            "check_verdict_pairs_default_is_n600",
            "check_telemetry_verdict_rows_carry_liveness",
            "check_levelset_hosc_requires_beta_end",
            "check_launch_config_authored_in_dsl",
            "check_no_unjustified_magnitude_dismissal",
            "check_no_inert_additive_margin_composition",
            "check_codex_retry_preserves_original_sandbox_authority",
            "check_codex_nonisolated_writer_cap",
            "check_codex_drain_timeout_uses_liveness",
            "check_consolidation_debt_monitor_observability_and_cadence",
            "check_witness_trainers_emit_partial_freeze_alarm",
            "check_witness_verdict_rows_carry_dseg_descent_canary",
            "check_verdict_live_gap_defaults_on_during_ema_warmup",
            "check_no_raw_virtual_memory_safety_basis",
            "check_process_guard_excludes_observer_flag_values",
            "check_no_duplicate_canonical_spec_across_refs",
            "check_no_stub_lever_factories",
            "check_no_legacy_single_module_lever_surface_consumers",
            # 2026-07-31 ddm_gh1 CLASS GUARD: refuse-capable gates must carry an
            # EXECUTED positive control + a declared denominator.
            "check_refusal_gates_have_live_positive_control",
            # 2026-07-31 ddm_gh1 (#836): the parent repo is structurally blind to
            # the nested upstream/ git repo.
            "check_upstream_pin_no_content_drift",
            # 2026-08-01 ddm_vc1 (#842): a verdict emitted over an ENUMERATED
            # scope must be able to report how many items it examined —
            # "vacuity is indistinguishable from PASS".
            "check_verdict_surfaces_report_examined_count",
            "check_checkpoint_saves_do_not_silently_drop_optimizer_state",
            "check_lever_module_declares_its_trainer",
            "check_no_asserted_packet_ir_readiness_fields",
            # 2026-08-15 ddm_gb1 (#1073): the memory governor's two stale-reference
            # anti-patterns — a SIGSTOP-throttle resume gated on the STICKY macOS
            # pressure level (five jobs frozen 75+ min at 40.4 GiB available) and an
            # admission path that counts the durable-daemon registry without
            # reconciling it (three dead rows = 100.0 GiB phantom growth, two
            # refused launches). One gate, two legs, per the #299 consolidation
            # discipline; lands warn-only WITH two executed positive controls.
            "check_throttle_rearms_and_admission_reconciles",
            "check_no_row_contract_error_quarantines_the_ledger",
            "check_no_bulk_write_strands_the_ready_record",
        }

    def test_followon_gates_are_strict_flipped_in_preflight_all(self):
        source = (cg.REPO_ROOT / "src" / "tac" / "preflight.py").read_text(encoding="utf-8")
        strict_block = source.split("_CONFOUND_STRICT = {", 1)[1].split("}", 1)[0]
        for name in (
            "check_witness_trainers_emit_partial_freeze_alarm",
            "check_witness_verdict_rows_carry_dseg_descent_canary",
            "check_verdict_live_gap_defaults_on_during_ema_warmup",
        ):
            assert name in strict_block

    def test_consolidation_debt_gate_is_wired_warn_only_in_preflight_all(self):
        source = (cg.REPO_ROOT / "src" / "tac" / "preflight.py").read_text(encoding="utf-8")
        assert "from tac.confound_gates import CONFOUND_GATES as _CONFOUND_GATES" in source
        strict_block = source.split("_CONFOUND_STRICT = {", 1)[1].split("}", 1)[0]
        assert "check_consolidation_debt_monitor_observability_and_cadence" not in strict_block

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
            # historical launch.sh artifacts (append-only, immutable run records); the LIVE
            # v7.5.2 config correctly uses annealed --hosc-beta-end 3.177. Re-baselined 8→9
            # (a 2026-07-10 run.sh accreted); cannot edit history, only block further growth.
            "check_levelset_hosc_requires_beta_end": 9,
            # #403: the un-migrated derive_*_config (sealed/store_nothing/fresh_seeded/…) =
            # the documented DSL-migration queue; crucible routes through the typed layer.
            # Re-baselined 4→5 (queue depth grew by one); strict-flip to 0 when it drains (req V, #353).
            "check_launch_config_authored_in_dsl": 5,
            # #404 MAGNITUDE-DISMISSAL: warn-only, max_report-capped at 15. The
            # .omx/research corpus predates the discipline (historical hits); strict-flip
            # to 0 after the memory-point-3 re-audit sweep drains them.
            "check_no_unjustified_magnitude_dismissal": 15,
            # #405 additive-margin inert composition: live-count 0 (no launch.sh ships an
            # inert AM arm); strict-flip-eligible once the DSL fail-closed + trainer L1 land.
            "check_no_inert_additive_margin_composition": 0,
            # 2026-07-14 apparatus fix+gate atomic strict landings.
            "check_codex_retry_preserves_original_sandbox_authority": 0,
            "check_codex_nonisolated_writer_cap": 0,
            "check_codex_drain_timeout_uses_liveness": 0,
            "check_consolidation_debt_monitor_observability_and_cadence": 0,
            "check_witness_trainers_emit_partial_freeze_alarm": 0,
            "check_witness_verdict_rows_carry_dseg_descent_canary": 0,
            "check_verdict_live_gap_defaults_on_during_ema_warmup": 0,
            # 2026-07-17 bug-class-sweep gates (bounds backfilled 2026-07-18;
            # both MEASURED at live-count 0 on the real tree).
            "check_no_raw_virtual_memory_safety_basis": 0,
            "check_process_guard_excludes_observer_flag_values": 0,
            # 2026-07-18 NAME-ANCHORED-SEARCH duplicate-SoT gate (live 0 at
            # landing: all working-tree specs registered at their own paths).
            "check_no_duplicate_canonical_spec_across_refs": 0,
            # 2026-07-31 ddm_sb2 (#819) DESIGNED-STUB gate, landed into
            # CONFOUND_GATES with NO bounds entry — which makes this test raise
            # KeyError rather than assert, i.e. a gate registered without an
            # owner for its own live count. Backfilled by ddm_rg5 (#825) at the
            # MEASURED value; drains to 0 as #819 builds the stubs (that gate's
            # own named strict-flip condition), and until then this bound stops
            # the debt GROWING.
            "check_no_stub_lever_factories": 10,
            # 2026-07-31 ddm_rg5 (#825) sister of the above: refuses a CONSUMER
            # binding orphan accounting to the single-module lever surface.
            # STRICT in preflight at live-count 0, so the bound is 0 and must
            # stay 0 — any growth here means a new consumer re-created the
            # blindness that hid 9 of 10 designed-stubs from the duty queue.
            "check_no_legacy_single_module_lever_surface_consumers": 0,
            # 2026-07-31 ddm_gh1 CLASS GUARD. STRICT in preflight at live-count 0:
            # any growth means a registered positive control stopped firing (a
            # detector was narrowed/gutted) or control coverage regressed below
            # the ratchet floor.
            "check_refusal_gates_have_live_positive_control": 0,
            # 2026-07-31 ddm_gh1 (#836). PRE-EXISTING KeyError: the gate was
            # registered without a bound, so this parametrised case had been
            # erroring (not passing) since it landed. Fixed forward by ddm_vc1
            # at its MEASURED live count of 0 — leaving a known-red test on main
            # is the exact failure that produced the law this batch cures.
            "check_upstream_pin_no_content_drift": 0,
            # 2026-08-01 ddm_vc1 (#842). STRICT in preflight at live-count 0
            # (MEASURED across 2404 files / 1595 scope-enumerating functions,
            # after the two same-batch fixes: the preflight CLI ScopeLedger
            # wire-in and review_tracker's cmd_selftest verdict). Any growth
            # means a NEW verdict landed that cannot report its denominator —
            # fix it or waive it with a real rationale; do NOT raise this bound.
            "check_verdict_surfaces_report_examined_count": 0,
            # ddm_op2 (OP2-1) STRICT from byte one: all six trainer callsites now
            # pass the run's resolver or `no_opt_state("<reason>")`, and the three
            # test-fixture callsites carry a stated OPT_STATE_DROP_OK waiver. A
            # nonzero count here means a checkpoint write started silently dropping
            # optimizer state again -- fix the callsite; do NOT raise this bound.
            "check_checkpoint_saves_do_not_silently_drop_optimizer_state": 0,
            # ddm_lr2 2026-08-03, both STRICT from byte one at live count 0.
            # A nonzero count here means a lever module started deciding a
            # DESIGNED-STUB verdict from an inherited trainer binding again --
            # declare TRAINER_RELPATH on the module; do NOT raise this bound.
            "check_lever_module_declares_its_trainer": 0,
            # A nonzero count means a packet-IR readiness field is being ASSERTED
            # (bare `True` / bare `len(...)`) instead of computed, and is being
            # summed into packet_ir_byte_closed_operation_count as if checked.
            # Compute it like byte_shaving_campaign's sibling does; do NOT raise.
            "check_no_asserted_packet_ir_readiness_fields": 0,
            # ddm_gb1 2026-08-15, warn-only for one cycle at MEASURED live count 0
            # (pre-fix it measured 5: decide_governor_action + gov.main +
            # mbb._govern_tick on Leg A, safe_run + governor CLI on Leg B). A
            # nonzero count means a throttle resume went back to trusting the
            # sticky OS pressure level alone, or an admission path started
            # counting the durable-daemon registry without reconciling it -- wire
            # the re-arm / the reconcile; do NOT raise this bound.
            "check_throttle_rearms_and_admission_reconciles": 0,
            # #1081, STRICT at a MEASURED live count of 0 over 7,532 modules.
            # Was absent from this map entirely, which made its parametrised case
            # raise KeyError -- a gate registered without a bound reports as a
            # test ERROR, never as the live-count regression the bound exists to
            # catch. Fixed forward by ddm_pl1; do NOT raise it.
            "check_no_row_contract_error_quarantines_the_ledger": 0,
            # ddm_pl1: WARN-ONLY at a MEASURED live count of 10 over 11,016
            # modules (8 runner-context, 2 test-fixture). The bound is the
            # measured value, not a round number, so ANY new stranded record
            # fails here. Lower it as the ten sites are cured; never raise it.
            "check_no_bulk_write_strands_the_ready_record": 10,
        }
        v = fn(strict=False, verbose=False)
        assert len(v) <= bounds[fn.__name__], f"{fn.__name__} live-count grew: {v[:3]}"


# ===========================================================================
# consolidation-debt observability + cadence self-protect
# ===========================================================================

_CONSOLIDATION_CLEAN = """\
import subprocess

def read_status():
    return subprocess.run([\"git\", \"status\"], capture_output=True, text=True).stdout
"""


def _consolidation_fixture(
    root: Path,
    source: str = _CONSOLIDATION_CLEAN,
    *,
    session_start: bool = True,
    stop: bool = True,
) -> None:
    monitor = root / "tools" / "consolidation_debt.py"
    monitor.parent.mkdir(parents=True)
    monitor.write_text(source, encoding="utf-8")
    command = cg._CONSOLIDATION_HOOK_COMMAND
    settings = {
        "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": command}]}]
            if session_start
            else [],
            "Stop": [{"hooks": [{"type": "command", "command": command}]}]
            if stop
            else [],
        }
    }
    settings_path = root / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps(settings), encoding="utf-8")


def test_consolidation_debt_guard_accepts_clean_read_only_monitor(tmp_path):
    _consolidation_fixture(tmp_path)
    assert cg.check_consolidation_debt_monitor_observability_and_cadence(
        repo_root=tmp_path, strict=True, verbose=False
    ) == []


@pytest.mark.parametrize(
    "source, expected",
    [
        ("def bad(p):\n    p.write_text('x')\n", "filesystem write"),
        (
            "import subprocess\ndef bad():\n    subprocess.run(['git', 'commit', '-m', 'x'])\n",
            "mutating git subprocess",
        ),
        (
            "import subprocess\ndef bad():\n    subprocess.run('git commit -m x', shell=True)\n",
            "mutating git subprocess",
        ),
        (
            "import subprocess\ndef bad():\n    subprocess.run(['python', 'tools/launch_job.py'])\n",
            "launch/dispatch subprocess",
        ),
        ("def bad():\n    open('state.json', 'w')\n", "writable open"),
        ("import os\ndef bad():\n    os.remove('state.json')\n", "filesystem write"),
    ],
)
def test_consolidation_debt_guard_negates_side_effect_mutants(tmp_path, source, expected):
    _consolidation_fixture(tmp_path, source)
    violations = cg.check_consolidation_debt_monitor_observability_and_cadence(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any(expected in violation for violation in violations)
    with pytest.raises(PreflightError, match=expected):
        cg.check_consolidation_debt_monitor_observability_and_cadence(
            repo_root=tmp_path, strict=True, verbose=False
        )


@pytest.mark.parametrize("event", ["SessionStart", "Stop"])
def test_consolidation_debt_guard_negates_lost_hook_wiring(tmp_path, event):
    _consolidation_fixture(
        tmp_path,
        session_start=event != "SessionStart",
        stop=event != "Stop",
    )
    violations = cg.check_consolidation_debt_monitor_observability_and_cadence(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any(event in violation and "lost exact" in violation for violation in violations)


def test_consolidation_debt_guard_respects_substantive_same_call_waiver(tmp_path):
    source = (
        "def reviewed_false_positive(p):\n"
        "    p.write_text('x')  # CONSOLIDATION_DEBT_SIDE_EFFECT_OK:test fixture only\n"
    )
    _consolidation_fixture(tmp_path, source)
    assert cg.check_consolidation_debt_monitor_observability_and_cadence(
        repo_root=tmp_path, strict=True, verbose=False
    ) == []


def test_consolidation_debt_guard_rejects_placeholder_waiver(tmp_path):
    source = (
        "def bad(p):\n"
        "    p.write_text('x')  # CONSOLIDATION_DEBT_SIDE_EFFECT_OK:<rationale>\n"
    )
    _consolidation_fixture(tmp_path, source)
    violations = cg.check_consolidation_debt_monitor_observability_and_cadence(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("filesystem write" in violation for violation in violations)


# ===========================================================================
# Catalog #405 — additive-margin inert composition (#404 binding-vs-inert).
# ===========================================================================

# Minimal launch.sh bodies — the gate scans the --head/--additive-margin/
# --margin-field-head-weight flag tokens (trainer path is irrelevant to the scan).
_AM_INERT_NO_MFH = "#!/bin/bash\nrun --head additive-margin --additive-margin 0.3 --epochs 10\n"
_AM_INERT_WRONG_HEAD = (
    "#!/bin/bash\nrun --head softmax --additive-margin 0.3 "
    "--margin-field-head-weight 1.0 --epochs 10\n")
_AM_ACTIVE = (
    "#!/bin/bash\nrun --head additive-margin --additive-margin 0.3 "
    "--margin-field-head-weight 1.0 --epochs 10\n")
_AM_ETF_CLEAN = "#!/bin/bash\nrun --head etf --epochs 10\n"


class TestAdditiveMarginEngagementClassifier:
    """The pure classifier SoT (#404) — the canary must be VISIBLE (fires on inert)
    and QUIET on a genuinely-active / clean composition (positive control)."""

    def test_inert_head_am_no_mfh(self):
        e = cg.additive_margin_engagement("additive-margin", 0.3, 0.0)
        assert e["nominally_set"] and e["inert"] and not e["engaged"]

    def test_inert_am_value_set_wrong_head(self):
        e = cg.additive_margin_engagement("softmax", 0.3, 1.0)
        assert e["nominally_set"] and e["inert"] and not e["engaged"]

    def test_inert_head_am_zero_margin(self):
        # head=additive-margin but the AM value is 0 -> zero hinge base -> inert.
        e = cg.additive_margin_engagement("additive-margin", 0.0, 1.0)
        assert e["inert"] and not e["engaged"]

    def test_engaged_full_triple(self):
        e = cg.additive_margin_engagement("additive-margin", 0.3, 1.0)
        assert e["engaged"] and not e["inert"] and e["nominally_set"]

    def test_clean_etf_not_nominally_set(self):
        e = cg.additive_margin_engagement("etf", 0.0, 0.0)
        assert not e["nominally_set"] and not e["inert"] and not e["engaged"]

    def test_clean_softmax_default(self):
        e = cg.additive_margin_engagement("softmax", 0.0, 0.0)
        assert not e["nominally_set"] and not e["inert"]


class TestNoInertAdditiveMarginComposition:
    def test_positive_inert_no_mfh(self, tmp_path):
        _launch(tmp_path, "am_inert", _AM_INERT_NO_MFH)
        v = cg.check_no_inert_additive_margin_composition(
            repo_root=tmp_path, strict=False, verbose=False)
        assert len(v) == 1 and "INERT" in v[0]

    def test_positive_inert_wrong_head(self, tmp_path):
        _launch(tmp_path, "am_wrong_head", _AM_INERT_WRONG_HEAD)
        v = cg.check_no_inert_additive_margin_composition(
            repo_root=tmp_path, strict=False, verbose=False)
        assert len(v) == 1 and "IGNORED" in v[0]

    def test_negative_active_composition(self, tmp_path):
        _launch(tmp_path, "am_active", _AM_ACTIVE)
        v = cg.check_no_inert_additive_margin_composition(
            repo_root=tmp_path, strict=False, verbose=False)
        assert v == []

    def test_negative_etf_clean(self, tmp_path):
        _launch(tmp_path, "am_etf", _AM_ETF_CLEAN)
        v = cg.check_no_inert_additive_margin_composition(
            repo_root=tmp_path, strict=False, verbose=False)
        assert v == []

    def test_waiver_respected(self, tmp_path):
        body = _AM_INERT_NO_MFH + "# ADDITIVE_MARGIN_INERT_OK: intentional off A/B baseline arm\n"
        _launch(tmp_path, "am_waived", body)
        v = cg.check_no_inert_additive_margin_composition(
            repo_root=tmp_path, strict=False, verbose=False)
        assert v == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        body = _AM_INERT_NO_MFH + "# ADDITIVE_MARGIN_INERT_OK:<rationale>\n"
        _launch(tmp_path, "am_ph", body)
        v = cg.check_no_inert_additive_margin_composition(
            repo_root=tmp_path, strict=False, verbose=False)
        assert len(v) == 1

    def test_strict_raises(self, tmp_path):
        _launch(tmp_path, "am_inert2", _AM_INERT_NO_MFH)
        with pytest.raises(PreflightError):
            cg.check_no_inert_additive_margin_composition(
                repo_root=tmp_path, strict=True, verbose=False)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS for the three positive controls landed 2026-08-01 (#831).
#
# A control that only FIRES is half a control. If the paired clean fixture is
# never asserted, a detector that flags EVERYTHING passes its positive control
# perfectly while being useless — the positive leg proves sensitivity, the
# negative leg proves specificity, and a refusal gate needs both before it can
# be called proven. Each clean case below is the exact CURE the gate names, run
# through the same scan surface as its planted twin.
# ---------------------------------------------------------------------------


class TestPositiveControlNegativeTwins831:
    def test_spike_guard_rollback_default_is_silent(self, tmp_path):
        """#397's cure: default='rollback' (what the live trainer ships)."""
        _trainer(
            tmp_path,
            "import argparse\n"
            "def build():\n"
            "    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--spike-guard-mode', default='rollback')\n"
            "    return p\n",
            base=True,
        )
        assert cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False) == []
        # NON-VACUITY: prove the clean silence came from the CURE, not from an
        # unscanned path. Same file, defect restored -> must fire.
        _trainer(
            tmp_path,
            "import argparse\n"
            "def build():\n"
            "    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--spike-guard-mode', default='legacy')\n"
            "    return p\n",
            base=True,
        )
        assert cg.check_no_spike_guard_defaults_to_deadlock_mode(
            repo_root=tmp_path, strict=False, verbose=False) != []

    def test_verdict_pairs_zero_default_is_silent(self, tmp_path):
        """#401's cure: default=0 == all pairs == n600."""
        _trainer(
            tmp_path,
            "import argparse\n"
            "def build():\n"
            "    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--verdict-pairs', type=int, default=0)\n"
            "    return p\n",
            base=True,
        )
        assert cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False) == []
        # NON-VACUITY: same file, defect restored -> must fire.
        _trainer(
            tmp_path,
            "import argparse\n"
            "def build():\n"
            "    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--verdict-pairs', type=int, default=24)\n"
            "    return p\n",
            base=True,
        )
        assert cg.check_verdict_pairs_default_is_n600(
            repo_root=tmp_path, strict=False, verbose=False) != []

    def test_rearm_present_is_silent(self, tmp_path):
        """#398's cure: an explicit re-anchor next to the accepted-only append."""
        _trainer(
            tmp_path,
            "def train_step(loss, median_window, stall):\n"
            "    ref = sorted(median_window)[len(median_window) // 2]\n"
            "    spiked = loss > 3.0 * ref\n"
            "    if spiked:\n"
            "        stall += 1\n"
            "        if stall > 50:\n"
            "            median_window.clear()\n"
            "        return True\n"
            "    else:\n"
            "        median_window.append(loss)\n"
            "    return False\n",
            base=True,
        )
        assert cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False) == []
        # NON-VACUITY: same file, re-arm removed -> must fire.
        _trainer(
            tmp_path,
            "def train_step(loss, median_window):\n"
            "    ref = sorted(median_window)[len(median_window) // 2]\n"
            "    spiked = loss > 3.0 * ref\n"
            "    if spiked:\n"
            "        return True\n"
            "    else:\n"
            "        median_window.append(loss)\n"
            "    return False\n",
            base=True,
        )
        assert cg.check_reject_filter_updates_reference_from_accepted_only_has_rearm(
            repo_root=tmp_path, strict=False, verbose=False) != []

    def test_control_registry_and_ratchets_are_consistent(self):
        """The two ratchets must describe the SAME live coverage they gate on.

        A floor or ceiling that drifts from the measured coverage is a silent
        instrument: the meta-gate keeps passing while the number it reports has
        stopped meaning anything.
        """
        coverage = cg.positive_control_coverage()
        assert coverage["covered"] >= cg.MIN_POSITIVE_CONTROL_COVERAGE
        assert len(coverage["uncovered_gates"]) <= cg.MAX_UNCOVERED_REFUSE_GATES
        # The ceiling is the debt QUEUE length; it must not have been raised to
        # admit a bare gate.
        assert cg.MAX_UNCOVERED_REFUSE_GATES <= 17
        assert cg.MIN_POSITIVE_CONTROL_COVERAGE >= 8
