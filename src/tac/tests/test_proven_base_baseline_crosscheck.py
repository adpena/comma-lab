"""Fail-closed cross-check: ``witness_autoconfig._proven_base()`` <-> ``curriculum_dsl.BASELINE``.

Audit #1 (``.omx/research/hardcode_duplication_audit_witness_stack_20260710.md``) flagged that
``_proven_base()`` is a hand-maintained parallel encoding of the shared baseline knobs that
``tac.witness_dsl.curriculum_dsl.BASELINE`` also encodes, WITH NO CROSS-CHECK TEST — the
``config MUST be DSL-defined`` (#353) discipline being violated inside the DSL's own actuator module.

The two are INDEPENDENTLY-FROZEN historical configs (``_proven_base`` = the 0.003698 muon arm;
``BASELINE`` = the completed n200 CE->tau->l7 run), so we do NOT couple one to the other by derivation
(that would falsely imply a dependency and could break the frozen §7 SEALED byte-identity gate when the
n200 baseline is tuned). Instead this test FAIL-CLOSED asserts their EMPIRICAL agreement on every shared
knob, with an EXPLICIT documented exception allow-list. A future silent divergence in either encoding
fires loud here -> the "no silent drift / the apparatus holds the memory, not the operator" intent.

Second guard (``test_proven_base_sealed_knobs_pinned_to_section7_not_v752``): PINS the specific knobs an
audit reader might mistake for "stale" (``grad_clip``/``softmax_temp_end``/``hosc_beta``/
``lane_prior_phi1_mode``) to their SEALED §7 #205 values, because they differ from the LATER live v7.5.2
crucible launch (a DIFFERENT config LINEAGE via ``derive_crucible_v752_config``) — NOT from drift.
Reconciling them toward v752 would BREAK ``test_sealed_205_argv_byte_identical_to_sealed_section7``.

means != ends: this guards a MEANS (the launch apparatus). Only a byte-closed n600 exact row < 0.19108
from ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer.
"""
from __future__ import annotations

from tac import witness_autoconfig as wac
from tac.witness_dsl.curriculum_dsl import BASELINE

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _baseline_shared_knobs() -> dict:
    """snake_case view of BASELINE's shared baseline knobs: base dict (``--flag`` -> ``snake``) +
    the temp Anneal (softmax_temp_start/end) + the regularizers (eikonal/length weight) + preserve
    (ckpt_every). This is the DSL-side encoding of the values ``_proven_base()`` also carries."""
    out: dict = {}
    for flag, val in BASELINE.base.items():
        out[flag.lstrip("-").replace("-", "_")] = val
    out["softmax_temp_start"] = BASELINE.temp.start
    out["softmax_temp_end"] = BASELINE.temp.end
    for reg in BASELINE.regularizers:
        out[reg.flag.lstrip("-").replace("-", "_")] = reg.weight
    out["ckpt_every"] = BASELINE.preserve.ckpt_every
    return out


# The ONE documented, intentional shared-knob divergence. Any NEW divergence must be added here WITH a
# rationale (a deliberate, reviewed decision) — that is exactly the drift-catching mechanism: a silent
# change to either encoding lands as an UNEXPECTED disagreement and fails the test below.
_INTENTIONAL_EXCEPTIONS = {
    # _proven_base keeps w_pose=0 (pose-BLIND base); _sealed_205_deltas() flips it to 1.0 so the SEALED
    # delta owns pose engagement (means/ends firewall). BASELINE bakes --w-pose 1.0 into its base dict
    # directly. Both intentional; the emitted sealed argv still ends at --w-pose 1.0.
    "w_pose": ("proven_base=0 (sealed delta owns pose); BASELINE bakes 1.0 into base", 0, 1.0),
}


def test_proven_base_agrees_with_baseline_on_shared_knobs():
    """FAIL-CLOSED cross-check: every knob shared by ``_proven_base()`` and ``BASELINE`` agrees,
    except the explicitly-documented ``_INTENTIONAL_EXCEPTIONS``. Catches the audit-#1 silent-drift
    class the moment either encoding changes without the other."""
    pb = wac._proven_base()
    bl = _baseline_shared_knobs()
    shared = set(pb) & set(bl)
    assert shared, "no shared knobs found — the key-translation broke (BASELINE refactor?)"

    unexpected = {
        k: (pb[k], bl[k])
        for k in shared
        if str(pb[k]) != str(bl[k]) and k not in _INTENTIONAL_EXCEPTIONS
    }
    assert not unexpected, (
        "proven_base <-> BASELINE drifted on shared knob(s) (audit-#1 silent-drift class). "
        f"Disagreements {{knob: (proven_base, BASELINE)}} = {unexpected}. Either propagate the tune "
        "to BOTH encodings, or add the knob to _INTENTIONAL_EXCEPTIONS with a reviewed rationale."
    )

    # The documented exceptions must still hold their KNOWN pair — so a change to an INTENTIONAL
    # divergence ALSO fires (forces re-review of the exception, not a silent slide).
    for knob, (why, pb_expected, bl_expected) in _INTENTIONAL_EXCEPTIONS.items():
        assert knob in shared, f"documented exception {knob!r} is no longer shared: {why}"
        assert str(pb[knob]) == str(pb_expected), (
            f"intentional-exception {knob!r} proven_base value changed "
            f"({pb[knob]!r} != documented {pb_expected!r}); re-review {why}")
        assert str(bl[knob]) == str(bl_expected), (
            f"intentional-exception {knob!r} BASELINE value changed "
            f"({bl[knob]!r} != documented {bl_expected!r}); re-review {why}")


def test_proven_base_sealed_knobs_pinned_to_section7_not_v752():
    """Pin the knobs an audit reader might mistake for "stale" to their SEALED §7 #205 values.

    These differ from the LATER live v7.5.2 crucible launch (grad-clip 0.5 / softmax-temp-end 0.31 /
    annealed hosc-end 3.177 / lane-prior-phi1-mode paint) — but that is a DIFFERENT config LINEAGE
    (``derive_crucible_v752_config``, typed DSL, never reads _proven_base), NOT drift. Reconciling
    _proven_base toward v752 would break the frozen §7 byte-identity gate. This test makes that
    mis-fix fail LOUD."""
    pb = wac._proven_base()
    assert pb["grad_clip"] == 1.0, "SEALED §7 grad-clip is 1.0 (v7.5.2's 0.5 is a different lineage)"
    assert pb["softmax_temp_end"] == 0.05, "SEALED §7 softmax-temp-end is 0.05 (v7.5.2's 0.31 differs)"
    assert pb["hosc_beta"] == 4.0, "proven_base plain-baseline hosc_beta (attribution-clean path)"
    assert pb["lane_prior_phi1_mode"] == "replace", (
        "SEALED §7 lane-prior-phi1-mode is 'replace' (v7.5.2's 'paint' is a different lineage)")


def test_derive_sealed_205_emits_the_pinned_section7_values():
    """Tie the pinned _proven_base knobs to the ACTUAL emitted sealed argv: derive_sealed_205_config
    must emit exactly the §7 values (grad-clip 1.0 / softmax-temp-end 0.05 / hosc-beta-end 4.0 /
    lane-prior-phi1-mode replace). This is the byte-identity-to-ground-truth check for the CORRECT
    ground truth (the §7 SEALED oracle), complementing test_sealed_205_canonical_config."""
    cfg = wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)
    fmap = {f: v for f, v in cfg.to_trainer_flags("/OUT")}
    assert str(fmap["--grad-clip"]) == "1.0"
    assert str(fmap["--softmax-temp-end"]) == "0.05"
    assert str(fmap["--hosc-beta-end"]) == "4.0"       # from all_levers_base (annealed 1.0 -> 4.0)
    assert fmap["--lane-prior-phi1-mode"] == "replace"
    # and the sealed config carries NONE of the v7.5.2-only crucible flags (proves lineage separation)
    fnames = {f for f, _ in cfg.to_trainer_flags("/OUT")}
    for v752_only in ("--ladder-island-homotopy", "--birth-completion-event",
                      "--seg-temporal-screw-weight", "--dseg-aware-taper"):
        assert v752_only not in fnames, (
            f"{v752_only} leaked into sealed_205 — it is a v7.5.2 crucible-only flag")
