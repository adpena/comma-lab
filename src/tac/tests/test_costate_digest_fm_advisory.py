"""Tests for tools/costate_digest.py section_fm_advisory — the #522 FM advisory wire-in.

Protects the CONTRACT: the section is PRESENT ONLY when the fmtools venv exists (⇒ the
digest is byte-identical without it); it consumes already-computed digest data (no rework);
regime disagreement surfaces a diagnostic line but is NEVER an override; the whole thing is
fail-open (a broken FM never breaks a session). Advisory · NON-PROMOTABLE.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import costate_digest as cd  # noqa: E402

from tac import fm_advisory as fa  # noqa: E402


# ─────────────────────── venv-absent ⇒ no lines (byte-identical digest) ───────────────────────
def test_section_absent_when_no_venv(monkeypatch) -> None:
    monkeypatch.setattr(fa, "fm_python", lambda: None)
    lines, data = cd.section_fm_advisory(None, {})
    assert lines == []
    assert data == {"available": False}


def test_build_digest_byte_identical_without_fm(monkeypatch) -> None:
    """With the venv absent, section_fm_advisory contributes zero lines — the digest is the
    same as if the section did not exist."""
    monkeypatch.setattr(fa, "fm_python", lambda: None)
    lines, data = cd.build_digest()
    assert data["fm_advisory"] == {"available": False}
    assert not any(ln.startswith("fm-advisory") for ln in lines)


# ─────────────────────── venv-present ⇒ compact section rendered ───────────────────────
def _data_with_regime() -> dict:
    return {
        "shadow": {"classification": {"classification": "DIVERGING_ERASING", "phase_regime": None,
                                      "reason": "sustained erosion"}},
        "annulus": {"annulus": {"per_class_annulus_flip_frac": {"0": 0.1, "1": 0.8, "3": 0.05}}},
        "duty_to_measure": {"ranked_top": [{"lever": "thin_lane", "activation_state": "never-fired",
                                            "why": "lane band"}]},
        "failure_ledger": None,
        "costate_organ": {"dispatch": {"regime": "transient"}},
    }


def test_section_renders_regime_and_agreement(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        # regime call → lane-erosion (matches numeric hint); duty → high
        lab = "lane-erosion" if "lane-erosion" in job["labels"] else job["labels"][0]
        return {"ok": True, "results": [{"id": it["id"], "label": lab, "rationale": "cited",
                                         "classifier": "apple-fm-on-device"} for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    monkeypatch.setattr(fa, "capability_report", lambda **_k: {
        "backend": "AppleFMBackend",
        "sdk_version": "0.2.1",
        "model_available": True,
        "supports_guided_generation": True,
        "supports_tools": True,
        "supports_streaming": True,
        "supports_generation_options": True,
    })
    lines, data = cd.section_fm_advisory(None, _data_with_regime())
    assert lines[0].startswith("fm-advisory (on-device FM")
    assert data["available"] is True
    assert data["capability_report"]["sdk_version"] == "0.2.1"
    assert any("capability: sdk=0.2.1" in ln and "guided=Y" in ln for ln in lines)
    assert any("regime: fm=lane-erosion" in ln and "[AGREE]" in ln for ln in lines)
    # duty-relevance secondary hint present, P8 order note included
    assert any("duty-relevance (secondary hint; P8 order unchanged)" in ln for ln in lines)


def test_section_surfaces_disagreement_line(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        # regime call → mixed-Lane-Road while numeric hint = lane-erosion ⇒ DISAGREE
        lab = "mixed-Lane-Road" if "lane-erosion" in job["labels"] else job["labels"][0]
        return {"ok": True, "results": [{"id": it["id"], "label": lab, "rationale": "both move",
                                         "classifier": "apple-fm-on-device"} for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    lines, _ = cd.section_fm_advisory(None, _data_with_regime())
    assert any("[DISAGREE]" in ln for ln in lines)
    assert any("regime DISAGREEMENT (advisory)" in ln and "never an override" in ln for ln in lines)


def test_section_present_but_no_inputs(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")
    monkeypatch.setattr(fa, "_run_job", lambda *_a, **_k: {"ok": True, "results": []})
    lines, data = cd.section_fm_advisory(None, {})
    assert data["available"] is True
    assert any("no classifiable inputs this cycle" in ln for ln in lines)


def test_section_fail_open_on_exception(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def boom(*_a, **_k):
        raise RuntimeError("regime exploded")
    monkeypatch.setattr(fa, "regime_supplement", boom)
    # must not raise; section still returns a header + honest unavailable sub-line
    lines, data = cd.section_fm_advisory(None, _data_with_regime())
    assert lines[0].startswith("fm-advisory")
    assert data["available"] is True


def test_build_digest_never_raises_with_fm_wired() -> None:
    # include_fm=False keeps the test fast + deterministic (no live FM subprocess); the section
    # key is still present as the recorded cost-gate state.
    lines, data = cd.build_digest(include_fm=False)
    assert lines and "fm_advisory" in data
    assert data["fm_advisory"]["enabled"] is False


# ─────────────────────── compute-cost gate (fast session-start path) ───────────────────────
def test_fm_gate_default_on_for_explicit_off_for_session_start(monkeypatch) -> None:
    monkeypatch.delenv("COSTATE_FM_ADVISORY", raising=False)
    assert cd._fm_advisory_enabled(session_start=False) is True   # explicit call → on
    assert cd._fm_advisory_enabled(session_start=True) is False   # session-start → off (<5s budget)


def test_fm_gate_env_override_both_directions(monkeypatch) -> None:
    monkeypatch.setenv("COSTATE_FM_ADVISORY", "1")
    assert cd._fm_advisory_enabled(session_start=True) is True    # forced on even at session-start
    monkeypatch.setenv("COSTATE_FM_ADVISORY", "0")
    assert cd._fm_advisory_enabled(session_start=False) is False  # forced off even for explicit
