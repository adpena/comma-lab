"""Scorer-free contract fixtures; these do NOT prove RLC2's real producer/harvest door.

Only Git history and the 3.6GB raw-file transport are fixture substitutes. Archive,
member, runtime, normalized receiver, small retained evidence and legacy T4 validation
are real local byte reads. No decoder, timing sampler, scorer, or provider is invoked.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import tac.candidate_seal as cs
from tac.decode_wall_clock import build_t4_direct_leg, measure_receiver_digest
from tac.tests.test_candidate_seal import DEFAULT_PAYLOAD, _public_smoke, _stage_candidate, _write_pointer
from tac.tests.test_decode_wall_clock_t4_direct import _report, _t4_receipt

COMMIT = "a1" * 20


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2))
    return cs.prefire_file_reference(path)


def sign(intent):
    intent["intent_sha256"] = cs.prefire_digest(intent, "intent_sha256")
    return intent


def load_tool(name):
    path = Path(__file__).resolve().parents[3] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root, archive = _stage_candidate(tmp_path / "candidate")
    base, base_archive = _stage_candidate(tmp_path / "base", bytes(range(256)) * 10)
    runtime = cs.measure_runtime_digest(root)
    receiver = measure_receiver_digest(root)
    pointer = _write_pointer(tmp_path, score=0.2, sha=cs.sha256_file(base_archive))
    smoke = _public_smoke(root, archive)
    base_smoke = _public_smoke(base, base_archive)
    for group in ("public_path_probes", "inflate_sh_smokes"):
        smoke[group]["frontier"] = base_smoke[group]["frontier"]
    endpoints = {"archive_sha256": cs.sha256_file(archive), "runtime_sha256": runtime.sha256,
                 "receiver_sha256": receiver}
    rows = [{"relative_path": p, "bytes": n, "sha256": h} for p, (n, h) in runtime.file_map().items()]
    store = tmp_path / "evidence"
    manifest = write(store / "manifest.json", {**endpoints, "files": rows, "producer_source_commit": COMMIT,
        "production_started_at_utc": "2026-09-10T00:00:00Z"})
    verification = write(store / "verified.json", {**endpoints, "manifest": manifest,
        "all_hashes_passed": True, "all_runtime_dependencies_listed": True})
    payloads = []
    for index in range(2):
        path = store / f"payload{index}.bin"
        path.write_bytes(DEFAULT_PAYLOAD)
        payloads.append(cs.prefire_file_reference(path))
    executions = [write(store / f"encode_{i}.json", {"execution_id": f"fixture-encode-{i}",
        "n_samples": 600, "completed": True, "payload": payload, "command": ["fixture-encoder", str(i)],
        "producer_source_commit": COMMIT}) for i, payload in enumerate(payloads)]
    twin = write(store / "twin.json", {**endpoints, "n_samples": 600, "payloads": payloads, "executions": executions})
    member_sha, member_bytes = cs.read_archive_member_identity(archive, "0.bin")
    parseback = write(store / "parseback.json", {**endpoints,
        "member": {"name": "0.bin", "sha256": member_sha, "bytes": member_bytes}})
    raw_refs = []
    for name in ("candidate.raw", "pointer.raw"):
        path = store / name
        path.write_bytes(b"fixture raw bytes; no full decode claim")
        ref = cs.prefire_file_reference(path)
        ref["bytes"] = 3662409600
        raw_refs.append(ref)
    original_ref = cs._pf_ref
    def fixture_ref(ref, code, *, parse=True):
        if isinstance(ref, dict) and ref.get("path") in {r["path"] for r in raw_refs}:
            assert not parse
            assert ref["sha256"] == cs.sha256_file(Path(ref["path"]))
            return Path(ref["path"])
        return original_ref(ref, code, parse=parse)
    monkeypatch.setattr(cs, "_pf_ref", fixture_ref)
    (store / "raw_public.log").write_text(_report())
    raw = write(store / "raw_identity.json", {**endpoints, "n_samples": 600, "pair_count": 600,
        "entrypoint": "inflate.sh", "checkpoint_resume": False, "token_cache_status": "DISABLED",
        "candidate_public_stdout": cs.prefire_file_reference(store / "raw_public.log"),
        "command": ["bash", str(root / "inflate.sh")],
        "candidate_raw": raw_refs[0], "pointer_raw": raw_refs[1], "pointer_archive_sha256": cs.sha256_file(base_archive)})
    census = write(store / "census.json", {**endpoints, "rule": 118, "verdict": "CLEAR", "complete": True, "files": rows})
    retention = write(store / "retention.json", {"payloads": [cs.prefire_file_reference(archive), *payloads, *raw_refs, parseback]})
    receipt = _t4_receipt(base, store / "source_t4.json", seconds=900)
    leg = build_t4_direct_leg(t4_receipt_path=receipt, runtime_dir=base, archive_path=base_archive)
    leg_ref = write(store / "source_leg.json", leg)
    def diagnostic(name, runtime_dir, wall):
        ref = write(store / name, {"wall_seconds": wall, "receiver_sha256": measure_receiver_digest(runtime_dir),
            "score_claim": False, "actual_verdict": "REFUSED", "cold_start": True, "checkpoint_resume": False, "frames": list(range(600))})
        return {**ref, "wall_seconds": wall, "authority": False, "actual_verdict": "REFUSED"}
    base_diagnostic = diagnostic("base_local.json", base, 100)
    candidate_diagnostic = {**diagnostic("candidate_local.json", root, 110), "cold": True, "n_samples": 600}
    smap = {r[0]: list(r[1:]) for r in cs.prefire_receiver_rows(base)}
    cmap = {r[0]: list(r[1:]) for r in cs.prefire_receiver_rows(root)}
    delta = write(store / "delta.json", {"source_receiver_sha256": measure_receiver_digest(base),
        "candidate_receiver_sha256": receiver, "files": [{"relative_path": p, "source": smap.get(p),
        "candidate": cmap.get(p)} for p in sorted(smap.keys() | cmap.keys())]})
    fraction = 110 / 100 - 1
    risk = {"schema": cs.PREFIRE_RISK_SCHEMA, "mode": "completed_t4_receiver_delta", "authority": False,
        "timing_clearance": False, "source_t4_leg": leg_ref, "source_receiver": {"sha256": leg["receiver_sha256"]},
        "candidate_receiver": {"sha256": receiver}, "diagnostic_reference_receiver": {"path": str(root), "sha256": receiver},
        "receiver_delta_manifest": delta, "base_local_diagnostic": base_diagnostic,
        "candidate_local_diagnostics": [candidate_diagnostic], "score_claim": False,
        "calculation": {"candidate_local_ceiling_seconds": 110, "local_cost_fraction_upper": fraction,
            "source_t4_seconds": 900, "t4_risk_ceiling_seconds": 900 * (1 + fraction),
            "policy_limit_seconds": 1260.0, "hard_timeout_seconds": 1800.0, "passed": True}}
    risk["risk_sha256"] = cs.prefire_digest(risk, "risk_sha256")
    risk_ref = write(store / "risk.json", risk)
    implementation_rows = []
    for name in cs.PREFIRE_IMPLEMENTATION_PATHS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic committed source fixture\n")
        implementation_rows.append({"path": name, "sha256": cs.sha256_file(path)})
    implementation = write(store / "implementation.json", sorted(implementation_rows, key=lambda r: r["path"]))
    memo_path = tmp_path / cs.PREFIRE_MEMO
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_bytes((Path(__file__).resolve().parents[3] / cs.PREFIRE_MEMO).read_bytes())
    def git(repo, *args):
        if args[:2] == ("show", "-s"):
            return b"2026-09-09T00:00:00+00:00"
        if args[0] == "show":
            return (repo / args[1].split(":", 1)[1]).read_bytes()
        if args[:2] == ("rev-parse", "HEAD"):
            return COMMIT.encode()
        return b""
    monkeypatch.setattr(cs, "_pf_git", git)
    intent = {"schema": cs.PREFIRE_INTENT_SCHEMA, "state": "PREFIRE_FIRST_MEASUREMENT_ONLY", "candidate_id": "fixture_candidate",
        "created_at_utc": "2026-09-10T01:00:00Z", "created_by": "fixture-producer", "producer_source_commit": COMMIT,
        "score_claim": False, "promotion_eligible": False, "timing_clearance": False,
        "contract": {"adjudication_memo": cs.prefire_file_reference(memo_path), "implementation_commit": COMMIT,
            "implementation_manifest": implementation, "implementation_manifest_sha256": implementation["sha256"]},
        "candidate": {"archive": cs.prefire_file_reference(archive), "runtime": {"path": str(root), **runtime.to_dict()},
            "normalized_receiver": {"digest_definition": "tac.decode_wall_clock.measure_receiver_digest", "sha256": receiver},
            "receiver_pins": [row for row in rows if row["relative_path"] in {"inflate.py", "inflate.sh"}], "archive_member": None},
        "admit_bar": {"rule": "net dS = dS_rate + 100*(d_seg_new - d_seg_base) + (sqrt(10*d_pose_new) - sqrt(10*d_pose_base)) < threshold", "net_dS_threshold": -1e-8,
            "pointer_axis": "contest_cuda", "pointer_score_at_intent": 0.2,
            "pointer_archive_sha256_at_intent": cs.sha256_file(base_archive), "pointer_tolerance_abs": 0.0,
            "require_pointer_archive_identity": True, "rate_only_precheck": {"raw_identity_required": True,
                "normalizer_bytes": 37545489, "derived_net_dS": 25 * (archive.stat().st_size - base_archive.stat().st_size) / 37545489, "passed": True}},
        "public_entrypoint_smoke": smoke, "evidence": {"candidate_manifest": manifest, "manifest_validation": verification,
            "twin_encode": twin, "archive_parseback": parseback, "raw_identity_n600": raw, "literal_census": census,
            "retention_manifest": retention, "timing_risk": risk_ref},
        "dispatch_policy": dict(cs.PREFIRE_DISPATCH_POLICY), "retained_payload_paths": [str(archive)],
        "falsifiers": ["exact candidate net-dS misses frozen bar"]}
    path = tmp_path / "intent.json"
    write(path, sign(intent))
    return {"intent": intent, "path": path, "repo": tmp_path, "pointer": pointer, "store": store, "root": root}


def validate(f):
    return cs.validate_prefire_intent(f["path"], repo=f["repo"], pointer_path=f["pointer"])


def test_fixture_intent_validates_but_is_not_a_seal(fixture):
    assert validate(fixture)["schema"] == cs.PREFIRE_INTENT_SCHEMA
    assert cs.validate_seal(fixture["path"]).verdict == "PREFIRE_INTENT_SCHEMA_REFUSED"


@pytest.mark.parametrize("mutation", [lambda d: d.update(extra=1), lambda d: d.pop("candidate_id"),
    lambda d: d.update(candidate_id="TBD"), lambda d: d.update(created_at_utc="yesterday"),
    lambda d: d.update(state="SEALED"), lambda d: d["dispatch_policy"].update(max_paid_dispatches=True)])
def test_schema_refusals(fixture, mutation):
    mutation(fixture["intent"])
    write(fixture["path"], sign(fixture["intent"]))
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_INTENT_SCHEMA_REFUSED"):
        validate(fixture)


@pytest.mark.parametrize("field,value", [("score_claim", True), ("promotion_eligible", True),
    ("timing_clearance", True), ("decode_wall_clock", {}), ("measured_t4_decode_seconds", 1),
    ("projected_t4_decode_seconds", 1), ("candidate_score", 0.1), ("timing_passed", True)])
def test_false_authority_refused(fixture, field, value):
    fixture["intent"][field] = value
    write(fixture["path"], sign(fixture["intent"]))
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_INTENT_FALSE_AUTHORITY_REFUSED"):
        validate(fixture)


def test_canonical_digest_is_not_file_digest(fixture):
    assert cs.prefire_file_reference(fixture["path"])["sha256"] != fixture["intent"]["intent_sha256"]
    fixture["intent"]["intent_sha256"] = "f" * 64
    write(fixture["path"], fixture["intent"])
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_INTENT_SCHEMA_REFUSED"):
        validate(fixture)


def test_contract_live_source_drift(fixture):
    (fixture["repo"] / cs.PREFIRE_IMPLEMENTATION_PATHS[0]).write_text("changed\n")
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_CONTRACT_DRIFT_REFUSED"):
        validate(fixture)


def test_identity_drift(fixture):
    (fixture["root"] / "cpr1/semantic_receiver.py").write_text("changed\n")
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_IDENTITY_DRIFT_REFUSED"):
        validate(fixture)


def test_pointer_drift(fixture):
    pointer = json.loads(fixture["pointer"].read_text())
    pointer["our_local_frontier_contest_cuda"]["score"] = 0.19
    write(fixture["pointer"], pointer)
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_POINTER_DRIFT_REFUSED"):
        validate(fixture)


@pytest.mark.parametrize("name", ["manifest_validation", "twin_encode", "archive_parseback", "raw_identity_n600", "literal_census", "retention_manifest"])
def test_evidence_byte_drift(fixture, name):
    Path(fixture["intent"]["evidence"][name]["path"]).write_text("{}")
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_NON_TIMING_GATE_REFUSED"):
        validate(fixture)


def test_risk_refuses_forged_arithmetic_but_keeps_legacy_leg(fixture):
    ref = fixture["intent"]["evidence"]["timing_risk"]
    risk = json.loads(Path(ref["path"]).read_text())
    leg = json.loads(Path(risk["source_t4_leg"]["path"]).read_text())
    assert set(leg["candidate_t4_receipt"]) == {"path", "sha256"}
    risk["calculation"]["t4_risk_ceiling_seconds"] = 1
    risk["risk_sha256"] = cs.prefire_digest(risk, "risk_sha256")
    fixture["intent"]["evidence"]["timing_risk"] = write(Path(ref["path"]), risk)
    write(fixture["path"], sign(fixture["intent"]))
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_RISK_EVIDENCE_REFUSED"):
        validate(fixture)


def auth_fixture(f, monkeypatch):
    monkeypatch.setattr(cs, "_pf_output", lambda path: Path(path))
    out = f["repo"] / "authorized_output"
    source_path = f["store"] / "price.txt"
    source_path.write_text(json.dumps({"chargeable_resources": ["gpu", "cpu", "memory"],
        "rates": dict.fromkeys(("gpu", "cpu", "memory"), 1e-05)}))
    resources = [{"resource": name, "quantity": qty, "usd_per_unit_second": 0.00001,
                  "price_field": ["rates", name], "upper_bound_usd": qty * 0.00001 * 4800}
                 for name, qty in (("gpu", 1), ("cpu", 4), ("memory", 16))]
    cost = write(f["store"] / "cost.json", {"gpu": "T4", "currency": "USD", "paid_dispatches": 1,
        "remote_seconds": 4800, "provider_price_fetched_at_utc": datetime.now(UTC).isoformat(),
        "provider_price_source": {**cs.prefire_file_reference(source_path), "url": "https://modal.com/pricing"},
        "resources": resources, "upper_bound_usd": sum(r["upper_bound_usd"] for r in resources)})
    actual_validate = cs.validate_prefire_intent
    monkeypatch.setattr(cs, "validate_prefire_intent", lambda path, **kw: actual_validate(path, repo=f["repo"], pointer_path=f["pointer"]))
    auth = cs.build_first_measurement_authorization(intent_path=f["path"], lane_id="fixture_lane",
        instance_job_id="fixture_job", output_dir=out, cost_path=Path(cost["path"]), repo=f["repo"])
    path = f["repo"] / "authorization.json"
    write(path, auth)
    ledger = f["repo"] / ".omx/state/modal_call_id_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"call_id": "fc-fixture", "event_type": "dispatched", "lane_id": auth["lane_id"],
        "expected_axis": "contest_cuda", "gpu": "T4", "archive_count": 1,
        "first_measurement_authorization_sha256": auth["authorization_sha256"], "prefire_intent_sha256": auth["intent"]["digest"],
        "intent_file_sha256": auth["intent"]["file_sha256"], "intent_file_bytes": auth["intent"]["file_bytes"],
        "instance_job_id": auth["instance_job_id"]}) + "\n")
    return auth, path


@pytest.mark.parametrize("field,value", [("file_sha256", "f" * 64), ("file_bytes", 1), ("digest", "f" * 64)])
def test_authorization_binds_all_three_intent_values(fixture, monkeypatch, field, value):
    auth, path = auth_fixture(fixture, monkeypatch)
    auth["intent"][field] = value
    auth["authorization_sha256"] = cs.prefire_digest(auth, "authorization_sha256")
    write(path, auth)
    with pytest.raises(cs.PrefireRefusal, match="FIRST_MEASUREMENT_AUTHORIZATION_REFUSED"):
        cs.validate_first_measurement_authorization(path, fixture["path"], fixture["intent"], repo=fixture["repo"])


def test_nonce_single_use_and_forward_transitions(fixture, monkeypatch):
    auth, path = auth_fixture(fixture, monkeypatch)
    assert cs.validate_first_measurement_authorization(path, fixture["path"], fixture["intent"], repo=fixture["repo"]) == auth
    before = cs.prefire_file_reference(fixture["path"])
    assert cs.reserve_first_measurement(auth, repo=fixture["repo"])["state"] == "RESERVED"
    with pytest.raises(cs.PrefireRefusal, match="FIRST_MEASUREMENT_REPLAY_REFUSED"):
        cs.reserve_first_measurement(auth, repo=fixture["repo"])
    assert cs.transition_first_measurement(auth, "SPAWNED", call_id="fc-fixture", repo=fixture["repo"])["state"] == "SPAWNED"
    with pytest.raises(cs.PrefireRefusal, match="FIRST_MEASUREMENT_REPLAY_REFUSED"):
        cs.transition_first_measurement(auth, "RESERVED", call_id="fc-fixture", repo=fixture["repo"])
    assert cs.prefire_file_reference(fixture["path"]) == before


@pytest.mark.parametrize("extra", [["--axis", "cuda"], ["--seal", "x"], ["--output-dir", "x"], ["--repin-receiver"], ["--poller-deadline-s", "5400"]])
def test_first_fire_cli_override_refuses_before_subprocess(tmp_path, monkeypatch, extra):
    tool = load_tool("fire_modal_auth_eval")
    monkeypatch.setattr(tool.subprocess, "run", lambda *a, **kw: pytest.fail("subprocess reached"))
    assert tool.main(["--first-measurement", str(tmp_path / "intent.json"),
        "--first-measurement-authorization", str(tmp_path / "auth.json"), *extra]) == 9
    refusal = json.loads(next(tmp_path.glob("PREFIRE_REFUSAL_*.json")).read_text())
    assert refusal["code"] == "FIRST_MEASUREMENT_ARGUMENT_REFUSED"


def test_missing_authorization_refuses_before_subprocess(tmp_path, monkeypatch):
    tool = load_tool("fire_modal_auth_eval")
    monkeypatch.setattr(tool.subprocess, "run", lambda *a, **kw: pytest.fail("subprocess reached"))
    assert tool.main(["--first-measurement", str(tmp_path / "intent.json")]) == 9
    assert json.loads(next(tmp_path.glob("PREFIRE_REFUSAL_*.json")).read_text())["code"] == "FIRST_MEASUREMENT_AUTHORIZATION_REFUSED"


def completion_fixture(f, monkeypatch):
    auth, auth_path = auth_fixture(f, monkeypatch)
    monkeypatch.setattr(cs, "__file__", str(f["repo"] / "src/tac/candidate_seal.py"))
    cs.reserve_first_measurement(auth, repo=f["repo"])
    cs.transition_first_measurement(auth, "SPAWNED", call_id="fc-fixture", repo=f["repo"])
    context = {"prefire_intent_sha256": f["intent"]["intent_sha256"],
        "first_measurement_authorization_sha256": auth["authorization_sha256"],
        "lane_id": auth["lane_id"], "instance_job_id": auth["instance_job_id"], "receipt_path": auth["receipt_path"],
        "inflate_timeout_seconds": 1800, "evaluate_timeout_seconds": 1800,
        "modal_function_timeout_seconds": 4800, "poller_deadline_seconds": 5400,
        "source_snapshot": {"schema": "modal_source_snapshot.v1", "complete": True,
            "verify_failures": [], "missing_in_source": [], "files_digest": "a" * 64},
        "exact_argv": ["modal", "--gpu", "T4", "--scorer-device", "cuda", "--inflate-device", "auto",
            "--inflate-timeout", "1800", "--evaluate-timeout", "1800", "--claim-policy", "require_active",
            "--lane-id", auth["lane_id"], "--instance-job-id", auth["instance_job_id"], "--output-dir", auth["output_dir"],
            "--expected-archive-sha256", f["intent"]["candidate"]["archive"]["sha256"]]}
    request = write(Path(auth["output_dir"]) / "modal_cuda_auth_eval_local_request.json", context)
    path = _t4_receipt(f["root"], Path(auth["receipt_path"]), seconds=900,
        **context, call_id="fc-fixture", worker_request=request, avg_segnet_dist=0.001, avg_posenet_dist=0.0,
        archive_size_bytes=f["intent"]["candidate"]["archive"]["bytes"], score_claim=False,
        promotion_eligible=False, adjudication_required=True)
    result = json.loads(path.read_text())
    artifact = json.loads(result["artifacts"]["contest_auth_eval.json"])
    artifact.update(avg_segnet_dist=0.001, avg_posenet_dist=0.0)
    result["artifacts"]["contest_auth_eval.json"] = json.dumps(artifact)
    write(path, result)
    return auth, auth_path, path


def test_completed_fixture_seal_uses_unchanged_direct_builder(fixture, monkeypatch):
    auth, auth_path, receipt = completion_fixture(fixture, monkeypatch)
    before = cs.prefire_file_reference(fixture["path"])
    cs.transition_first_measurement(auth, "HARVESTED", call_id="fc-fixture", receipt_path=receipt, repo=fixture["repo"])
    output = fixture["repo"] / "completed_seal.json"
    document = cs.complete_first_fire_intent(intent_path=fixture["path"], authorization_path=auth_path,
        receipt_path=receipt, out_path=output, repo=fixture["repo"], pointer_path=fixture["pointer"])
    assert document["schema"] == "candidate_seal.v3"
    assert document["decode_wall_clock"] == build_t4_direct_leg(t4_receipt_path=receipt,
        runtime_dir=fixture["root"], archive_path=fixture["root"] / "archive.zip")
    assert cs.validate_seal(output, pointer_path=fixture["pointer"], require_decode_wall_clock=True).ok
    assert cs.prefire_file_reference(fixture["path"]) == before
    assert set(document["first_measurement_receipt"]) == {"path", "bytes", "sha256"}
    assert set(document["decode_wall_clock_reference"]) == {"path", "bytes", "sha256"}
    assert json.loads(Path(document["decode_wall_clock_reference"]["path"]).read_text()) == document["decode_wall_clock"]
    assert set(document["decode_wall_clock"]["candidate_t4_receipt"]) == {"path", "sha256"}
    # A legacy helper failure remains a typed normal-consumer refusal, never a traceback.
    def missing_component(*args):
        raise cs.SealContractError("missing exact evaluator component")
    monkeypatch.setattr(cs, "_pf_completed_score", missing_component)
    verdict = cs.validate_seal(output, pointer_path=fixture["pointer"], require_decode_wall_clock=True)
    assert verdict.verdict == "FIRST_MEASUREMENT_RESULT_REFUSED"
    assert not verdict.ok


@pytest.mark.parametrize("mutation,code", [
    (lambda d: d.update(inflate_timeout_seconds=1799), "FIRST_MEASUREMENT_TIMEOUT_REFUSED"),
    (lambda d: d.update(returncode=124), "FIRST_MEASUREMENT_TIMEOUT_REFUSED"),
    (lambda d: d["artifacts"].update({"contest_auth_eval.stdout.log": "warm"}), "FIRST_MEASUREMENT_WARM_REFUSED"),
    (lambda d: d["artifacts"].update({"contest_auth_eval.json": json.dumps({"n_samples": 600, "inflate_elapsed_seconds": 1260.000000001})}), "FIRST_MEASUREMENT_T4_POLICY_REFUSED"),
    (lambda d: d.update(avg_segnet_dist=1), "FIRST_MEASUREMENT_RESULT_REFUSED"),
    (lambda d: d.update(call_id="fc-wrong"), "FIRST_MEASUREMENT_RESULT_REFUSED"),
    (lambda d: d.update(expected_archive_sha256="f" * 64), "FIRST_MEASUREMENT_RESULT_REFUSED"),
    (lambda d: d.update(prefire_intent_sha256="f" * 64), "FIRST_MEASUREMENT_RESULT_REFUSED"),
])
def test_harvest_refusals(fixture, monkeypatch, mutation, code):
    auth, auth_path, receipt = completion_fixture(fixture, monkeypatch)
    result = json.loads(receipt.read_text())
    mutation(result)
    write(receipt, result)
    cs.transition_first_measurement(auth, "HARVESTED", call_id="fc-fixture", receipt_path=receipt, repo=fixture["repo"])
    with pytest.raises(cs.PrefireRefusal, match=code):
        cs.complete_first_fire_intent(intent_path=fixture["path"], authorization_path=auth_path,
            receipt_path=receipt, out_path=fixture["repo"] / "completed.json", repo=fixture["repo"], pointer_path=fixture["pointer"])
    assert not (fixture["repo"] / "completed.json").exists()


def prepare_fire(fixture, monkeypatch):
    import tac.deploy.modal.auth_eval as worker
    import tac.deploy.modal.single_flight as flight
    auth, path = auth_fixture(fixture, monkeypatch)
    tool = load_tool("fire_modal_auth_eval")
    monkeypatch.setattr(tool, "REPO", fixture["repo"])
    monkeypatch.setattr(tool, "verify_dispatch_paths", lambda *a: [])
    monkeypatch.setattr(worker, "require_active_modal_auth_eval_claim", lambda **kw: {"agent": "MAIN"})
    monkeypatch.setattr(flight, "single_flight_findings", lambda **kw: [])
    monkeypatch.setattr(tool, "first_measurement_cloud_apps", lambda: [])
    monkeypatch.setattr(tool.subprocess, "run", lambda *a, **kw: pytest.fail("subprocess reached before dry-run/refusal"))
    return auth, path, tool, flight


def test_dry_run_consumes_nothing_and_explicit_argv(fixture, monkeypatch, capsys):
    auth, path, tool, _ = prepare_fire(fixture, monkeypatch)
    before = set(fixture["repo"].rglob("*"))
    argv = ["--first-measurement", str(fixture["path"]), "--first-measurement-authorization", str(path), "--dry-run"]
    assert tool.main(argv) == 0
    assert tool.main(argv) == 0
    text = capsys.readouterr().out
    assert '"--inflate-timeout"' in text and '"1800"' in text and '"--claim-policy"' in text
    assert not cs.first_measurement_consumption_path(auth, fixture["repo"]).exists()
    assert before == set(fixture["repo"].rglob("*"))


@pytest.mark.parametrize("cloud", [None, ["active-scored-job"]])
def test_unknown_or_busy_cloud_refuses_before_dispatch(fixture, monkeypatch, cloud):
    _, path, tool, flight = prepare_fire(fixture, monkeypatch)
    monkeypatch.setattr(tool, "first_measurement_cloud_apps", lambda: cloud)
    assert tool.main(["--first-measurement", str(fixture["path"]), "--first-measurement-authorization", str(path)]) == 9
    refused = json.loads(next(fixture["repo"].glob("PREFIRE_REFUSAL_*.json")).read_text())
    assert refused["code"] == "FIRST_MEASUREMENT_LANE_REFUSED"


def test_recovery_and_mirror_cannot_unquarantine(fixture):
    from tac.deploy.modal.auth_eval import _recovered_claim_flags
    poller = load_tool("modal_harvest_poller")
    result = cs.quarantine_first_measurement_result({"passed": True, "score_claim": True, "promotion_eligible": True},
        {"prefire_intent_sha256": fixture["intent"]["intent_sha256"]})
    assert result["score_claim"] is result["promotion_eligible"] is False
    assert poller.build_anchor_mirror(result, source_receipt=fixture["path"], lane_id="fixture")[0] is None
    assert poller._stage_pointer_move_packet(result=result, result_path=fixture["path"], out_dir=fixture["repo"], lane_id="fixture") is None
    flags = _recovered_claim_flags(out_dir=fixture["repo"], result_without_artifacts=result, score_axis="contest_cuda")
    assert flags["score_claim"] is flags["promotion_eligible"] is False


@pytest.mark.parametrize("change", ["stale", "future", "five_dollars", "resource_quantity", "source_price", "second_dispatch"])
def test_cost_fails_closed(fixture, monkeypatch, change):
    auth, path = auth_fixture(fixture, monkeypatch)
    cost_path = Path(auth["cost_preflight"]["path"])
    cost = json.loads(cost_path.read_text())
    if change == "stale":
        cost["provider_price_fetched_at_utc"] = "2000-01-01T00:00:00Z"
    elif change == "future":
        cost["provider_price_fetched_at_utc"] = "2200-01-01T00:00:00Z"
    elif change == "five_dollars":
        cost["upper_bound_usd"] = 5.0
    elif change == "resource_quantity":
        cost["resources"][0]["quantity"] = 2
    elif change == "source_price":
        cost["resources"][0]["usd_per_unit_second"] = 0.00002
    else:
        cost["paid_dispatches"] = 2
    auth["cost_preflight"] = write(cost_path, cost)
    auth["authorization_sha256"] = cs.prefire_digest(auth, "authorization_sha256")
    write(path, auth)
    with pytest.raises(cs.PrefireRefusal, match="FIRST_MEASUREMENT_AUTHORIZATION_REFUSED"):
        cs.validate_first_measurement_authorization(path, fixture["path"], fixture["intent"], repo=fixture["repo"])


def test_duplicate_json_and_nested_false_authority(fixture):
    fixture["path"].write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_INTENT_SCHEMA_REFUSED"):
        validate(fixture)
    fixture["intent"]["candidate"]["runtime"]["candidate_score"] = 0.1
    write(fixture["path"], sign(fixture["intent"]))
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_INTENT_FALSE_AUTHORITY_REFUSED"):
        validate(fixture)


def test_remote_retention_keeps_every_materialized_file(tmp_path):
    root = tmp_path / "persistent_volume"
    root.mkdir()
    raw = root / "candidate.raw"
    raw.write_bytes(b"actual small fixture raw bytes")
    context = {"exact_argv": ["fixture"], "first_measurement_authorization_sha256": "a" * 64,
               "prefire_intent_sha256": "b" * 64}
    ref = cs.retain_first_measurement_worker_tree(root, context)
    doc = json.loads(Path(ref["path"]).read_text())
    assert doc["payloads"] == [cs.prefire_file_reference(raw)]
    assert doc["cleanup_disposition"] == "BLOCKED_KEEP_BYTES"
    assert raw.read_bytes() == b"actual small fixture raw bytes"


def test_unregistered_call_cannot_transition(fixture, monkeypatch):
    auth, _ = auth_fixture(fixture, monkeypatch)
    cs.reserve_first_measurement(auth, repo=fixture["repo"])
    with pytest.raises(cs.PrefireRefusal, match="FIRST_MEASUREMENT_RESULT_REFUSED"):
        cs.transition_first_measurement(auth, "SPAWNED", call_id="fc-unregistered", repo=fixture["repo"])
    record = json.loads(cs.first_measurement_consumption_path(auth, fixture["repo"]).read_text())
    assert record["state"] == "RESERVED"


def test_intent_producer_emits_and_self_validates(fixture):
    contract = fixture["intent"]["contract"]
    write(fixture["repo"] / cs.PREFIRE_FREEZE, {"implementation_commit": COMMIT,
        "implementation_manifest": contract["implementation_manifest"]})
    output = fixture["repo"] / "new_intent.json"
    doc = cs.build_prefire_intent(candidate_id="fixture_produced", runtime_dir=fixture["root"],
        evidence_paths={key: Path(ref["path"]) for key, ref in fixture["intent"]["evidence"].items()},
        public_entrypoint_smoke=fixture["intent"]["public_entrypoint_smoke"], net_ds_threshold=-1e-8,
        retained_paths=fixture["intent"]["retained_payload_paths"], falsifiers=fixture["intent"]["falsifiers"],
        out_path=output, repo=fixture["repo"], pointer_path=fixture["pointer"])
    assert cs.prefire_digest(doc, "intent_sha256") == doc["intent_sha256"]
    assert cs.validate_seal(output).verdict == "PREFIRE_INTENT_SCHEMA_REFUSED"


def test_worker_wrapper_retains_inputs_raw_and_quarantines(tmp_path, monkeypatch):
    import ast
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[3] / "experiments/modal_auth_eval.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "run_auth_eval")
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    commits = []
    def inner(**kwargs):
        root = Path(kwargs["retained_work_root"])
        assert (root / "INPUT_archive.zip").read_bytes() == b"archive fixture"
        (root / "raw").write_bytes(b"raw fixture")
        return {"passed": True, "score_claim": True, "promotion_eligible": True, "artifacts": {}}
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=32 * 1024**3))
    namespace = {"Path": Path, "AUTH_CACHE_VOLUME_ROOT": tmp_path, "AUTH_CACHE_VOLUME_NAME": "fixture-volume",
        "auth_cache_vol": SimpleNamespace(commit=lambda: commits.append(True)), "_run_auth_eval_fail_closed": inner}
    exec(compile(module, str(source), "exec"), namespace)
    context = {"first_measurement_authorization_sha256": "a" * 64, "prefire_intent_sha256": "b" * 64,
               "exact_argv": ["fixture-worker"]}
    result = namespace["run_auth_eval"](b"archive fixture", "c" * 64, 15, first_measurement_context=context)
    assert result["score_claim"] is result["promotion_eligible"] is False
    assert len(commits) == 2
    assert (tmp_path / "first_measurements" / ("a" * 64) / "raw").read_bytes() == b"raw fixture"
    manifest = json.loads(result["artifacts"]["FIRST_MEASUREMENT_RETENTION.json"])
    assert len(manifest["payloads"]) == 2
    with pytest.raises(FileExistsError):
        namespace["run_auth_eval"](b"archive fixture", "c" * 64, 15, first_measurement_context=context)


def test_cloud_preflight_uses_sdk_without_subprocess(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from modal.client import _Client
    tool = load_tool("fire_modal_auth_eval")
    client = SimpleNamespace(stub=SimpleNamespace(TaskList=AsyncMock(return_value=SimpleNamespace(tasks=[]))))
    monkeypatch.setattr(_Client, "from_env", AsyncMock(return_value=client))
    monkeypatch.setattr(tool.subprocess, "Popen", lambda *a, **kw: pytest.fail("provider subprocess reached"))
    assert tool.first_measurement_cloud_apps() == []
    client.stub.TaskList.return_value = SimpleNamespace(tasks=[SimpleNamespace(app_id="ap-active")])
    assert tool.first_measurement_cloud_apps() == ["ap-active"]
    client.stub.TaskList.side_effect = ValueError("unknown provider state")
    assert tool.first_measurement_cloud_apps() is None


def test_worker_local_custody_survives_snapshot_import(fixture, monkeypatch):
    import ast
    import os

    source = Path(__file__).resolve().parents[3] / "experiments/modal_auth_eval.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node.decorator_list = []
    # Execute the actual local validation prefix, stopping before upload/provider work.
    stop = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "prepared" for t in n.targets))
    node.body = node.body[:stop]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    auth, auth_path = auth_fixture(fixture, monkeypatch)
    cs.reserve_first_measurement(auth, repo=fixture["repo"])
    monkeypatch.chdir(fixture["repo"])
    monkeypatch.setattr(cs, "__file__", str(fixture["repo"] / "snapshot/src/tac/candidate_seal.py"))
    monkeypatch.setenv("PACT_MODAL_SOURCE_ROOT", str(fixture["repo"] / "snapshot"))
    seen = []
    original_validate = cs.validate_prefire_intent
    def validate(path, **kwargs):
        seen.append(kwargs)
        assert kwargs["repo"] == fixture["repo"]
        assert kwargs["pointer_path"] == fixture["repo"] / ".omx/state/canonical_frontier_pointer.json"
        return original_validate(path, repo=kwargs["repo"], pointer_path=fixture["pointer"])
    monkeypatch.setattr(cs, "validate_prefire_intent", validate)
    context_path = fixture["repo"] / "worker_context.json"
    write(context_path, {"intent_path": str(fixture["path"]), "authorization_path": str(auth_path),
        "source_snapshot": {"complete": True}, "prefire_intent_sha256": fixture["intent"]["intent_sha256"],
        "first_measurement_authorization_sha256": auth["authorization_sha256"]})
    namespace = {"Path": Path, "os": os}
    exec(compile(module, str(source), "exec"), namespace)
    namespace["main"](archive=str(fixture["root"] / "archive.zip"), submission_dir=str(fixture["root"]),
        output_dir=auth["output_dir"], lane_id=auth["lane_id"], instance_job_id=auth["instance_job_id"],
        expected_archive_sha256=fixture["intent"]["candidate"]["archive"]["sha256"],
        first_measurement_context=str(context_path), gpu="T4", scorer_device="cuda", inflate_device="auto",
        inflate_timeout=1800, evaluate_timeout=1800, claim_policy="require_active", detach=True, provider_detach_ack=True)
    assert len(seen) == 1
