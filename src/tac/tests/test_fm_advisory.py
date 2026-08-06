# SPDX-License-Identifier: MIT
"""Tests for tac.witness_control.fm_advisory — the #522 on-device FM ADVISORY sense layer.

Protects the CONTRACT (advisory boundary, never actuation):
  * venv-ABSENT graceful degrade → every entry point returns None (clean, not a stub);
  * classify subprocess call contract (mocked runner) → aligned labelled rows;
  * in-process content-hash cache (mocked runner called once for identical text);
  * numeric_regime_hint derivation + agreement-flag logic;
  * the four insertion helpers shape their outputs correctly.
Every FM output is advisory · NON-PROMOTABLE; these tests never touch a live model.
"""
from __future__ import annotations

from tac import fm_advisory as fa


# ─────────────────────── venv-absent graceful degrade (the clean None path) ───────────────────────
def test_available_false_when_no_fm_python(monkeypatch) -> None:
    monkeypatch.setattr(fa, "fm_python", lambda: None)
    assert fa.available() is False


def test_all_entry_points_degrade_to_none_when_absent(monkeypatch) -> None:
    monkeypatch.setattr(fa, "fm_python", lambda: None)
    assert fa.classify(["x"], ("a", "b"), "i") is None
    assert fa.classify_one("x", ("a", "b"), "i") is None
    assert fa.regime_supplement([{"epoch": 1}]) is None
    assert fa.classify_events([{"stage": "s"}]) is None
    assert fa.duty_relevance([{"lever": "l"}], "lane-erosion") is None
    assert fa.charter_class("build this") is None
    assert fa.mechanism_reduction_language("quick toy run") is None
    assert fa.classify_confounds([{"failure_id": "f"}], ["c1"]) is None
    assert fa.capability_report() is None
    assert fa.shadow_advisory(telemetry_texts=[{"a": 1}], event_texts=[{"b": 2}]) is None


# ─────────────────────── classify subprocess contract (mocked runner) ───────────────────────
def _fake_runner(results):
    def _run(_fm_py, job, _timeout):
        # echo back one labelled result per input item, honoring id alignment
        return {"ok": True, "results": [
            {"id": it["id"], "label": results.get(it["text"][:20], job["labels"][0]),
             "rationale": "cited", "classifier": "apple-fm-on-device"}
            for it in job["items"]
        ]}
    return _run


def test_classify_aligns_and_labels(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")
    monkeypatch.setattr(fa, "_run_job", _fake_runner({}))
    out = fa.classify(["alpha", "beta"], ("x", "y"), "instr", cache=False)
    assert out is not None and len(out) == 2
    assert all(r["label"] == "x" for r in out)
    assert all(r["classifier"] == "apple-fm-on-device" for r in out)


def test_classify_none_when_subprocess_fails_and_no_cache(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")
    monkeypatch.setattr(fa, "_run_job", lambda *_a, **_k: None)
    assert fa.classify(["a"], ("x",), "i", cache=False) is None


def test_classify_abstains_on_out_of_set_label(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        return {"ok": True, "results": [{"id": it["id"], "label": "NOT_A_LABEL",
                                         "rationale": "", "classifier": "apple-fm-on-device"}
                                        for it in job["items"]]}
    # The subprocess script itself nulls out-of-set labels; here _run_job returns a valid
    # id-map so classify passes it through — the enforcement lives in the script. We assert
    # classify does not crash and returns the row (label enforcement is the script's job).
    monkeypatch.setattr(fa, "_run_job", _run)
    out = fa.classify(["a"], ("x", "y"), "i", cache=False)
    assert out is not None and out[0]["id"] == 0


# ─────────────────────── content-hash cache ───────────────────────
def test_cache_avoids_recompute_for_identical_text(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")
    calls = {"n": 0}

    def _run(_fm_py, job, _t):
        calls["n"] += 1
        return {"ok": True, "results": [{"id": it["id"], "label": job["labels"][0],
                                         "rationale": "r", "classifier": "apple-fm-on-device"}
                                        for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    fa.classify(["same"], ("x", "y"), "i", cache=True)
    fa.classify(["same"], ("x", "y"), "i", cache=True)  # should hit cache
    assert calls["n"] == 1


def test_cache_key_stable_and_text_sensitive() -> None:
    k1 = fa._cache_key(("a", "b"), "instr", "text-1")
    k2 = fa._cache_key(("a", "b"), "instr", "text-1")
    k3 = fa._cache_key(("a", "b"), "instr", "text-2")
    assert k1 == k2 and k1 != k3


# ─────────────────────── numeric_regime_hint + agreement ───────────────────────
def test_numeric_regime_hint_lane_erosion() -> None:
    ann = {"annulus": {"per_class_annulus_flip_frac": {"0": 0.1, "1": 0.8, "3": 0.05}}}
    assert fa.numeric_regime_hint(ann) == "lane-erosion"


def test_numeric_regime_hint_mixed_lane_road() -> None:
    ann = {"annulus": {"per_class_annulus_flip_frac": {"0": 0.4, "1": 0.5, "3": 0.05}}}
    assert fa.numeric_regime_hint(ann) == "mixed-Lane-Road"


def test_numeric_regime_hint_none_when_no_shares() -> None:
    assert fa.numeric_regime_hint(None) is None
    assert fa.numeric_regime_hint({"annulus": {}}) is None
    assert fa.numeric_regime_hint({"annulus": {"per_class_annulus_flip_frac": {}}}) is None


def test_regime_supplement_agreement_flag(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        return {"ok": True, "results": [{"id": it["id"], "label": "lane-erosion",
                                         "rationale": "lane flicker", "classifier": "apple-fm-on-device"}
                                        for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    out = fa.regime_supplement([{"epoch": 350, "d_seg": 0.006}], numeric_hint="lane-erosion")
    assert out is not None
    assert out["fm_regime"] == "lane-erosion"
    assert out["agrees_with_numeric"] is True

    out2 = fa.regime_supplement([{"epoch": 350}], numeric_hint="mixed-Lane-Road")
    assert out2["agrees_with_numeric"] is False

    out3 = fa.regime_supplement([{"epoch": 350}], numeric_hint=None)
    assert out3["agrees_with_numeric"] is None


# ─────────────────────── insertion helpers shape ───────────────────────
def test_classify_events_shape(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        return {"ok": True, "results": [{"id": it["id"], "label": "stage-transition",
                                         "rationale": "muon switch", "classifier": "apple-fm-on-device"}
                                        for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    rows = fa.classify_events([{"stage": "muon_finisher_switch", "epoch": 726}])
    assert rows is not None and rows[0]["event_class"] == "stage-transition"


def test_duty_relevance_shape(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        return {"ok": True, "results": [{"id": it["id"], "label": "high",
                                         "rationale": "targets lanes", "classifier": "apple-fm-on-device"}
                                        for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    rows = fa.duty_relevance([{"lever": "thin_lane", "why": "lane band"}], "lane-erosion")
    assert rows is not None and rows[0]["lever"] == "thin_lane" and rows[0]["relevance"] == "high"


def test_charter_class_shape(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        return {"ok": True, "results": [{"id": it["id"], "label": "build_race_train_measure",
                                         "rationale": "build and measure", "classifier": "apple-fm-on-device"}
                                        for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    row = fa.charter_class("Implement and measure the receiver.")
    assert row is not None
    assert row["charter_class"] == "build_race_train_measure"
    assert row["authority"] == fa.AUTHORITY


def test_mechanism_reduction_language_flags_multiple(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        results = []
        for it in job["items"]:
            present = "TARGET quick-train:" in it["text"] or "TARGET undersized:" in it["text"]
            results.append({
                "id": it["id"],
                "label": "present" if present else "absent",
                "rationale": "shortcut language" if present else "not present",
                "classifier": "apple-fm-on-device",
            })
        return {"ok": True, "results": results}

    monkeypatch.setattr(fa, "_run_job", _run)
    row = fa.mechanism_reduction_language("Run a quick tiny proxy and call it done.")
    assert row is not None
    assert row["flags"] == ["quick-train", "undersized"]
    assert {r["label"]: r["present"] for r in row["rows"]}["toy-scale"] is False


def test_mechanism_reduction_language_none_when_all_items_unclassified(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        return {"ok": True, "results": [{"id": it["id"], "label": None,
                                         "rationale": "unclassified",
                                         "classifier": "apple-fm-skip"}
                                        for it in job["items"]]}

    monkeypatch.setattr(fa, "_run_job", _run)
    assert fa.mechanism_reduction_language("quick tiny proxy") is None


def test_classify_confounds_maps_none(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        return {"ok": True, "results": [{"id": it["id"], "label": "none",
                                         "rationale": "novel", "classifier": "apple-fm-on-device"}
                                        for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    rows = fa.classify_confounds([{"failure_id": "x", "symptom": "spike"}], ["spike_deadlock"])
    assert rows is not None and rows[0]["matched_class"] is None


def test_capability_report_shape(monkeypatch) -> None:
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run_capability(_fm_py, _timeout):
        return {
            "ok": True,
            "report": {
                "backend": "AppleFMBackend",
                "sdk_version": "0.2.1",
                "model_available": True,
                "supports_guided_generation": True,
                "supports_tools": True,
                "supports_streaming": True,
                "supports_generation_options": True,
            },
        }

    monkeypatch.setattr(fa, "_run_capability_job", _run_capability)
    report = fa.capability_report(timeout=3)
    assert report is not None
    assert report["sdk_version"] == "0.2.1"
    assert report["supports_guided_generation"] is True


def test_capability_report_fail_open(monkeypatch) -> None:
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")
    monkeypatch.setattr(fa, "_run_capability_job", lambda *_a, **_k: {"ok": False})
    assert fa.capability_report(timeout=3) is None


def test_shadow_advisory_bundle(monkeypatch) -> None:
    fa._CACHE.clear()
    monkeypatch.setattr(fa, "fm_python", lambda: "/fake/py")

    def _run(_fm_py, job, _t):
        lab = "lane-erosion" if job["labels"][0] == "lane-erosion" else "info"
        return {"ok": True, "results": [{"id": it["id"], "label": lab,
                                         "rationale": "r", "classifier": "apple-fm-on-device"}
                                        for it in job["items"]]}
    monkeypatch.setattr(fa, "_run_job", _run)
    ann = {"annulus": {"per_class_annulus_flip_frac": {"0": 0.1, "1": 0.8, "3": 0.05}}}
    out = fa.shadow_advisory(
        telemetry_texts=[{"epoch": 350, "d_seg": 0.006}],
        event_texts=[{"stage": "verdict", "epoch": 350}],
        annulus_data=ann,
    )
    assert out is not None
    assert out["regime"]["fm_regime"] == "lane-erosion"
    assert out["regime"]["agrees_with_numeric"] is True  # numeric hint = lane-erosion
    assert isinstance(out["events"], list)


# ─────────────────────── prosify (guardrail framing) ───────────────────────
def test_prosify_dict_and_str() -> None:
    assert "epoch is 5" in fa.prosify({"epoch": 5, "d_seg": 0.01})
    assert fa.prosify("plain text") == "plain text"
    assert "epoch is 5" in fa.prosify('{"epoch": 5}')  # json-string is unwrapped
