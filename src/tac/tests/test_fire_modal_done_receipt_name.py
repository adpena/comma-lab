"""The fire tool composes its watcher's --done-receipt from --instance-job-id.

On 2026-08-18 that compose produced `modal:ddm_sa3_cuda_t4_20260818_harvest`, which
launch_detached_process.py refuses (':' is outside its receipt-name charset). The arm
call went LIVE on Modal with NO watcher and the tool still exited 0. These tests bind
the sanitizer to the launcher's own pattern and pin the real failing input.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fire():
    return _load("_fire_modal_auth_eval", "tools/fire_modal_auth_eval.py")


def _launcher_receipt_pattern() -> str:
    """Read the launcher's _RECEIPT_NAME pattern WITHOUT importing it.

    Importing the launcher would run its module body; we only need the contract.
    """
    src = (REPO / "tools" / "launch_detached_process.py").read_text(encoding="utf-8")
    m = re.search(r"_RECEIPT_NAME\s*=\s*re\.compile\(\s*r?[\"'](?P<pat>[^\"']+)[\"']", src)
    assert m, "launch_detached_process.py no longer defines _RECEIPT_NAME as a literal regex"
    return m.group("pat")


def test_our_allowed_pattern_matches_the_launchers_contract() -> None:
    """Anti-drift: if the launcher tightens its charset, this test fails LOUD."""
    assert _fire()._DONE_RECEIPT_ALLOWED.pattern == _launcher_receipt_pattern()


def test_the_real_2026_08_18_failure_now_produces_a_legal_name() -> None:
    """POSITIVE CONTROL — the exact input that armed no watcher."""
    fire = _fire()
    bad = "modal:ddm_sa3_cuda_t4_20260818"
    # The pre-fix compose is genuinely illegal — proving the control is not vacuous.
    assert not fire._DONE_RECEIPT_ALLOWED.fullmatch(f"{bad}_harvest")
    receipt = fire._done_receipt_name(bad)
    assert fire._DONE_RECEIPT_ALLOWED.fullmatch(receipt)
    assert receipt == "modal_ddm_sa3_cuda_t4_20260818_harvest"


def test_an_already_legal_job_id_is_only_suffixed() -> None:
    """NEGATIVE CONTROL — colon-free ids (e.g. iv1's) must not be rewritten."""
    fire = _fire()
    assert fire._done_receipt_name("iv1_repinned_r2") == "iv1_repinned_r2_harvest"


def test_hostile_inputs_still_yield_a_legal_name() -> None:
    fire = _fire()
    for hostile in ("::::", "  ", "/a/b/c", "-leading-dash", "..dots", "x" * 400):
        receipt = fire._done_receipt_name(hostile)
        assert fire._DONE_RECEIPT_ALLOWED.fullmatch(receipt), hostile
        assert len(receipt) <= 128
