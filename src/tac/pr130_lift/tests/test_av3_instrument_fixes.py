"""Self-protection for the four ddm_av3 instrument defects.

Each is a two-landing item: the fix, plus a gate that refuses its re-entry.

* **F2a** ``--save`` / ``--out`` pointing at an existing DIRECTORY.  ddm_lr1/A2
  fired that shape, ran 379 s of Metal, computed its whole result, and lost it to
  ``IsADirectoryError`` at the final ``os.replace``.
* **F2b** the write ORDER: the cheap irreplaceable ``result.json`` was written
  AFTER the expensive rebuildable checkpoint, so the checkpoint's failure took
  the result with it.
* **F3**  ``result.json`` headlining the argmin INCLUDING step 0, so a run that
  only degrades reports its own INIT with ``verdict: PASS`` -- which is what all
  four ddm_lr1 arms did.
* **F7**  the LawRef-derived EMA decay recorded as governing a run whose warmup
  ramp never reaches it.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.pr130_lift import train_semantic_quantized_resumable as trainer

REPO_ROOT = Path(__file__).resolve().parents[4]


def _base_argv(tmp_path: Path) -> list[str]:
    return [
        "--challenge-root", "upstream",
        "--cache", "/dev/null",
        "--init", "/dev/null",
        "--bits", "4",
        "--steps", "600",
        "--save", str(tmp_path / "ckpt"),
        "--out", str(tmp_path / "result.json"),
    ]


# ---------------------------------------------------------------------------
# F2a -- fail fast instead of computing-then-discarding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["--save", "--out"])
def test_directory_at_an_output_path_refuses_at_parse_time(tmp_path, flag):
    collide = tmp_path / "checkpoints"
    collide.mkdir()
    argv = _base_argv(tmp_path)
    argv[argv.index(flag) + 1] = str(collide)
    with pytest.raises(SystemExit) as excinfo:
        trainer.parse_args(argv)
    assert excinfo.value.code == 2


def test_a_plain_file_basename_is_accepted(tmp_path):
    args = trainer.parse_args(_base_argv(tmp_path))
    assert args.save.name == "ckpt"
    assert args.out.name == "result.json"


def test_an_existing_FILE_at_the_output_path_is_still_allowed(tmp_path):
    """Only a DIRECTORY is fatal -- os.replace over a file is the normal case."""

    (tmp_path / "ckpt").write_text("stale")
    (tmp_path / "result.json").write_text("{}")
    assert trainer.parse_args(_base_argv(tmp_path)).save.name == "ckpt"


# ---------------------------------------------------------------------------
# F2b -- ALWAYS KEEP THE PAYLOAD: cheap+irreplaceable is written first
# ---------------------------------------------------------------------------
def test_result_json_is_written_before_the_final_checkpoint():
    """The ordering IS the fix; this gate refuses a future re-inversion."""

    source = inspect.getsource(trainer.run)
    write_result = source.index("_atomic_write_json(result, args.out)")
    save_final = source.index("_atomic_torch_save(final_payload, args.save)")
    assert write_result < save_final, (
        "the result JSON must be written BEFORE the final checkpoint: it is the "
        "cheap irreplaceable artifact, the checkpoint is the expensive "
        "rebuildable one (ddm_av3 F2)"
    )


def test_the_parity_refusal_path_also_writes_its_json_before_raising():
    source = inspect.getsource(trainer.run)
    assert source.index("_atomic_write_json(blocked, args.out)") < source.index(
        "EMA shadow refused"
    )


# ---------------------------------------------------------------------------
# F3 -- the result must say whether it improved on its own input
# ---------------------------------------------------------------------------
def test_result_declares_best_step_and_improved_over_init():
    source = inspect.getsource(trainer.run)
    for field in ('"best_step"', '"improved_over_init"', '"init_quantized_exact_seg"'):
        assert field in source, f"result dict lost {field} (ddm_av3 F3)"


def test_checkpoint_payload_requires_best_step_and_init_seg():
    """Never defaulted: a silent default would reintroduce the F3 defect."""

    parameters = inspect.signature(trainer._checkpoint_payload).parameters
    for name in ("best_step", "init_seg"):
        assert name in parameters, name
        assert parameters[name].default is inspect.Parameter.empty, (
            f"{name} must stay REQUIRED so a caller cannot silently omit it"
        )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"init_seg": 0.5, "best_seg": 0.1, "history": []}, 0.5),
        ({"best_seg": 0.1, "history": [{"quantized_exact_seg": 0.7}]}, 0.7),
        ({"best_seg": 0.1, "history": [{"step": 2}]}, 0.1),
        ({"best_seg": 0.1}, 0.1),
    ],
)
def test_restored_init_seg_is_legacy_tolerant_and_lazy(state, expected):
    """A pre-fix checkpoint has no init_seg; the fallback must not raise."""

    assert trainer._restored_init_seg(state) == pytest.approx(expected)


def test_restored_init_seg_prefers_the_recorded_value_over_history():
    """dict.get(k, default) evaluates eagerly -- this guards the lazy path."""

    state = {"init_seg": 0.25, "best_seg": 0.9, "history": [{"step": 2}]}
    assert trainer._restored_init_seg(state) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# F7 -- the EMA manifest must not over-claim
# ---------------------------------------------------------------------------
def test_short_run_reports_that_warmup_dominates_the_target_decay():
    policy = trainer.resolve_ema_policy(600)
    assert policy["warmup_dominates_target"] is True
    # min(decay, (1+t)/(10+t)) reaches the target only at this many updates.
    assert policy["warmup_target_crossover_updates"] == pytest.approx(1167.0, abs=2.0)
    # The declared intent, and what the run actually realises, differ by ~17
    # orders of magnitude.  Both are recorded so neither can be quoted alone.
    assert policy["declared_target_seed_fraction"] == pytest.approx(0.01, rel=1e-6)
    assert policy["realized_seed_retention_under_warmup"] < 1e-15


@pytest.mark.parametrize("updates", [100, 600, 3000, 20000, 1000000])
def test_warmup_dominance_is_STRUCTURAL_not_a_short_run_artifact(updates):
    """ddm_av3 F7 read this as "short runs"; it holds at EVERY run length.

    The LawRef derives decay = f**(1/N) from the geometry, so the warmup
    crossover ~ 9N/ln(1/f) ~ 1.954 N at the canonical f = 0.01.  The ratio is
    N-independent, so the target decay never governs.
    """

    policy = trainer.resolve_ema_policy(updates)
    assert policy["warmup_dominates_target"] is True
    assert policy["governing_policy"] == "warmup_ramp"
    assert policy["warmup_crossover_over_updates"] == pytest.approx(1.954, abs=0.06)


def test_the_dominance_threshold_is_the_derived_closed_form():
    """Dominance iff target_seed_fraction > exp(-9); assert the boundary bites."""

    assert pytest.approx(
        2.718281828459045**-9
    ) == trainer.EMA_WARMUP_DOMINANCE_SEED_FRACTION_THRESHOLD
    loose = trainer.resolve_ema_policy(
        600, target_seed_fraction=trainer.EMA_WARMUP_DOMINANCE_SEED_FRACTION_THRESHOLD * 10
    )
    tight = trainer.resolve_ema_policy(
        600, target_seed_fraction=trainer.EMA_WARMUP_DOMINANCE_SEED_FRACTION_THRESHOLD / 10
    )
    assert loose["warmup_dominates_target"] is True
    assert tight["warmup_dominates_target"] is False


def test_derived_ema_fields_are_excluded_from_the_resume_equality_check():
    """Additive in BOTH directions: old checkpoint vs new reader, and back."""

    policy = trainer.resolve_ema_policy(600)
    legacy = {
        key: value
        for key, value in policy.items()
        if key not in trainer.EMA_POLICY_DERIVED_OBSERVABILITY_KEYS
    }
    assert trainer._causal_ema_policy(policy) == trainer._causal_ema_policy(legacy)
    # ... but a REAL policy change must still refuse.
    changed = dict(policy)
    changed["decay"] = 0.5
    assert trainer._causal_ema_policy(changed) != trainer._causal_ema_policy(policy)


def test_derived_ema_keys_are_all_functions_of_causal_inputs():
    """The non-weakening argument, asserted: nothing causal is excluded."""

    assert trainer.EMA_POLICY_DERIVED_OBSERVABILITY_KEYS.isdisjoint(
        {"decay", "equation_id", "warmup", "fallback_used", "resolved_manifest"}
    )


# ---------------------------------------------------------------------------
# the band objective is additive on the resume path
# ---------------------------------------------------------------------------
def test_pre_band_checkpoint_resumes_into_an_inert_band_run():
    current = {"steps": 600, "band_objective_weight": 0.0, "band_weight_table_sha256": None}
    prior = {"steps": 600}
    reconciled_prior, reconciled_current = trainer._reconcile_additive_resume_config(
        prior, current
    )
    assert reconciled_prior == reconciled_current


def test_pre_band_checkpoint_refuses_an_ACTIVE_band_run():
    """An active term genuinely differs from its parent; the guard must fire."""

    current = {"steps": 600, "band_objective_weight": 1.0, "band_weight_table_sha256": "ab" * 32}
    prior = {"steps": 600}
    reconciled_prior, reconciled_current = trainer._reconcile_additive_resume_config(
        prior, current
    )
    assert reconciled_prior != reconciled_current


# ---------------------------------------------------------------------------
# safe_run: a crashed child must not read as a success
# ---------------------------------------------------------------------------
def test_safe_run_receipt_flags_a_nonzero_child_exit(tmp_path):
    receipt = tmp_path / "status.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "safe_run.py"),
            "--rss-mb", "2048",
            "--timeout", "60",
            "--status-receipt", str(receipt),
            "--skip-admission-gate",
            "--quiet",
            "--",
            sys.executable, "-c", "raise SystemExit(1)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    # The exit-code passthrough contract is UNCHANGED.
    assert completed.returncode == 1
    payload = json.loads(receipt.read_text())
    assert payload["exit"] == 1
    assert payload["child_exit_nonzero"] is True
    # This is the ddm_lr1/A2 receipt signature, now legible.
    assert payload["receipt_status_disagrees_with_exit"] is True


def test_safe_run_receipt_is_clean_for_a_successful_child(tmp_path):
    receipt = tmp_path / "status.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "safe_run.py"),
            "--rss-mb", "2048",
            "--timeout", "60",
            "--status-receipt", str(receipt),
            "--skip-admission-gate",
            "--quiet",
            "--",
            sys.executable, "-c", "pass",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "ok"
    assert payload["exit"] == 0
    assert payload["child_exit_nonzero"] is False
    assert payload["receipt_status_disagrees_with_exit"] is False
