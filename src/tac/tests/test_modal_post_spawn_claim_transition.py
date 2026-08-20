"""The POST-spawn claim transition must survive the dispatcher's OWN live ledger row.

THE DEFECT (measured 2026-08-20 by ``ddm_cpu1``; present in ``ddm_rr7``'s CUDA
fire log too, where nobody noticed).

A detached Modal auth-eval dispatch runs this sequence:

1. pre-claim  ``..._spawning``
2. ``assert_modal_single_flight``           <- the real guard, correctly placed
3. ``.spawn()``
4. ``register_dispatched_call_id_fail_closed``  <- registers THIS call_id live
5. claim transition to ``..._spawned``

Step 5 called :func:`claim_modal_auth_eval_dispatch`, which re-ran the step-2
guard. By then step 4 had made the dispatcher's own call_id a live
(non-terminal) ledger row, and the guard's LEDGER leg excludes nothing by
design — *"Live LEDGER rows are never excluded: a non-terminal call_id on the
same lane is exactly the un-harvested duplicate-breeder this guard exists for."*
So step 5 always raised. The claim stayed stranded at ``..._spawning`` and the
operator-facing ``DISPATCHED DETACHED`` / ``Recover:`` banner never printed.

WHY THE SUITE MISSED IT. ``test_modal_auth_eval.py``'s autouse
``_hermetic_single_flight`` fixture stubs the guard so unit tests do not couple
to live session state. That isolation is right, and it also removed the only
surface that could observe this self-conflict. These tests therefore exercise the
REAL guard against a SYNTHETIC on-disk ledger: hermetic AND real at once.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
from pathlib import Path

import pytest

from tac.deploy.modal import auth_eval as auth_eval_mod
from tac.deploy.modal.auth_eval import ClaimSpec, claim_modal_auth_eval_dispatch
from tac.deploy.modal.single_flight import ModalSingleFlightRefusal

REPO = Path(__file__).resolve().parents[3]
_LANE = "lane_unit_post_spawn_transition"  # FAKE_LANE_OK:test-fixture lane_id
_JOB = "modal:unit_post_spawn_transition"
_CALL_ID = "fc-unit-post-spawn-transition"


@pytest.fixture
def synthetic_repo(tmp_path: Path, monkeypatch) -> Path:
    """A repo root whose ledger holds the dispatcher's OWN live call_id."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    now = _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "") + "Z"
    (state / "modal_call_id_ledger.jsonl").write_text(
        json.dumps(
            {
                "call_id": _CALL_ID,
                "status": "dispatched",
                "platform": "modal",
                "label": "modal_auth_eval_cpu",
                "lane_id": _LANE,
                "written_at_utc": now,
                "dispatched_at_utc": now,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (state / "active_lane_dispatch_claims.md").write_text(
        "# Active lane dispatch claims\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | eta | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )
    # The cloud leg shells out to the modal CLI; keep the test hermetic.
    monkeypatch.setenv("TAC_MODAL_SINGLE_FLIGHT_SKIP_CLOUD", "1")
    return tmp_path


@pytest.fixture
def recorded_claims(monkeypatch) -> list[dict]:
    """Capture claim writes without shelling out to the claim tool."""
    calls: list[dict] = []
    monkeypatch.setattr(auth_eval_mod, "record_dispatch_claim", lambda **kw: calls.append(kw))
    return calls


def _spec() -> ClaimSpec:
    return ClaimSpec(lane_id=_LANE, instance_job_id=_JOB, agent="unit", force=True)


def test_pre_spawn_guard_still_refuses_on_a_live_ledger_row(synthetic_repo, recorded_claims):
    """The guard is NOT weakened: default behaviour still refuses. Regression floor."""
    with pytest.raises(ModalSingleFlightRefusal):
        claim_modal_auth_eval_dispatch(
            repo_root=synthetic_repo, spec=_spec(), status="active_modal_auth_eval_spawning"
        )
    assert recorded_claims == [], "a refused guard must never write a claim"


def test_post_spawn_transition_survives_the_dispatchers_own_live_row(synthetic_repo, recorded_claims):
    """THE FIX: the post-spawn transition records despite the dispatcher's own live call_id."""
    claim_modal_auth_eval_dispatch(
        repo_root=synthetic_repo,
        spec=_spec(),
        status="active_modal_auth_eval_spawned",
        pre_spawn_guard=False,
    )
    assert [c["status"] for c in recorded_claims] == ["active_modal_auth_eval_spawned"]


def test_post_spawn_transition_still_requires_lane_and_job(synthetic_repo, recorded_claims):
    """Skipping the guard must not skip the identity precondition."""
    with pytest.raises(SystemExit):
        claim_modal_auth_eval_dispatch(
            repo_root=synthetic_repo,
            spec=ClaimSpec(lane_id="", instance_job_id="", agent="unit"),
            status="active_modal_auth_eval_spawned",
            pre_spawn_guard=False,
        )
    assert recorded_claims == []


@pytest.mark.parametrize(
    "dispatcher",
    ["experiments/modal_auth_eval.py", "experiments/modal_auth_eval_cpu.py"],
)
def test_production_dispatchers_skip_the_guard_only_after_spawn(dispatcher):
    """STRUCTURAL: every ``..._spawned`` transition passes ``pre_spawn_guard=False``.

    This is the anti-reintroduction gate. A future edit that drops the flag puts
    the dispatcher straight back into the unreachable-transition state, and the
    dispatch would still succeed, so only a source-level assertion catches it.
    """
    path = REPO / dispatcher
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    spawned_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
        if name != "claim_modal_auth_eval_dispatch":
            continue
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        status = kwargs.get("status")
        if isinstance(status, ast.Constant) and str(status.value).endswith("_spawned"):
            spawned_calls.append((node.lineno, kwargs))

    assert spawned_calls, f"{dispatcher}: no post-spawn '..._spawned' claim transition found"
    for lineno, kwargs in spawned_calls:
        guard = kwargs.get("pre_spawn_guard")
        assert isinstance(guard, ast.Constant) and guard.value is False, (
            f"{dispatcher}:{lineno} advances a claim to '..._spawned' without "
            "pre_spawn_guard=False. The dispatcher's own call_id is already live in "
            "the ledger by then, so the guard refuses and the transition is "
            "unreachable (ddm_cpu1 2026-08-20)."
        )


def test_pre_spawn_guard_call_site_is_still_present_in_both_dispatchers():
    """The cure must not have removed the guard that actually prevents a second spawn."""
    for dispatcher in ("experiments/modal_auth_eval.py", "experiments/modal_auth_eval_cpu.py"):
        src = (REPO / dispatcher).read_text(encoding="utf-8")
        assert "assert_modal_single_flight(" in src, (
            f"{dispatcher}: the pre-spawn single-flight guard call site is GONE. "
            "pre_spawn_guard=False is a post-spawn affordance, never a licence to "
            "delete the guard."
        )
