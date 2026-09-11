"""Scorer-free contract fixtures; these do NOT prove RLC2's real producer/harvest door.

The broad fixtures substitute Git history and the 3.6GB raw-file transport; the
amendment custody tests read real isolated Git objects. Archive,
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
AMENDMENT_COMMIT = "b2" * 20
REAL_PF_GIT = cs._pf_git


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
    for tree in (root, base):
        (tree / "MANIFEST.sha256").write_text("".join(
            f"{cs.sha256_file(path)}  {path.relative_to(tree).as_posix()}\n"
            for path in sorted(tree.rglob("*")) if path.is_file() and path.name != "archive.zip"))
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
    candidate_diagnostic = {**diagnostic("candidate_local.json", base, 110), "cold": True, "n_samples": 600}
    smap = {r[0]: list(r[1:]) for r in cs.prefire_risk_receiver_rows(base)}
    cmap = {r[0]: list(r[1:]) for r in cs.prefire_risk_receiver_rows(root)}
    delta = write(store / "delta.json", {"source_receiver_sha256": cs.measure_prefire_risk_receiver_digest(base),
        "candidate_receiver_sha256": cs.measure_prefire_risk_receiver_digest(root), "files": [{"relative_path": p, "source": smap.get(p),
        "candidate": cmap.get(p)} for p in sorted(smap.keys() | cmap.keys())]})
    fraction = 110 / 100 - 1
    risk = {"schema": cs.PREFIRE_RISK_SCHEMA, "mode": "completed_t4_receiver_delta", "authority": False,
        "timing_clearance": False, "source_t4_leg": leg_ref, "source_receiver": {
            "digest_definition": cs.PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "sha256": cs.measure_prefire_risk_receiver_digest(base),
            "t4_direct_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
            "t4_direct_sha256": leg["receiver_sha256"]},
        "candidate_receiver": {"digest_definition": cs.PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "sha256": cs.measure_prefire_risk_receiver_digest(root)},
        "diagnostic_reference_receiver": {"path": str(base),
            "digest_definition": cs.PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "sha256": cs.measure_prefire_risk_receiver_digest(root),
            "receipt_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
            "receipt_sha256": measure_receiver_digest(base)},
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
    base_blobs = {name: (tmp_path / name).read_bytes() for name in cs.PREFIRE_IMPLEMENTATION_PATHS}
    (tmp_path / cs.PREFIRE_IMPLEMENTATION_PATHS[0]).write_text("# amended committed source fixture\n")
    amended_rows = [{"path": name, "sha256": cs.sha256_file(tmp_path / name)}
                    for name in sorted(cs.PREFIRE_IMPLEMENTATION_PATHS)]
    amended_manifest = write(store / "amended_implementation.json", amended_rows)
    amended_memo = tmp_path / ".omx/research/amendment.md"
    amended_memo.write_text("Fixture prospective amendment.\n")
    amendment = {"schema": cs.PREFIRE_CONTRACT_AMENDMENT_SCHEMA,
        "amendment_id": cs.PREFIRE_CONTRACT_AMENDMENT_ID,
        "adjudication_memo": cs.prefire_file_reference(amended_memo),
        "definition_change": {
            "scope": "candidate_prefire_timing_risk.v1 only; legacy decode_wall_clock unchanged",
            "digest_definition": cs.PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "excluded_relative_paths": ["MANIFEST.sha256"], "raw_manifest_still_required": True,
            "executable_difference_policy": "REFUSE"},
        "reference_receiver": {"path": str(base)}, "candidate_receiver": {"path": str(root)},
        "implementation_commit": AMENDMENT_COMMIT, "implementation_manifest": amended_manifest,
        "score_claim": False}
    write(tmp_path / cs.PREFIRE_FREEZE, {"implementation_commit": COMMIT,
        "implementation_manifest": implementation, "amendments": [amendment]})
    def git(repo, *args):
        if args[:2] == ("show", "-s"):
            return (b"2026-09-10T00:30:00+00:00" if args[3] == AMENDMENT_COMMIT
                    else b"2026-09-09T00:00:00+00:00")
        if args[0] == "show":
            ref, rel = args[1].split(":", 1)
            if ref == COMMIT and rel in base_blobs:
                return base_blobs[rel]
            return (repo / rel).read_bytes()
        if args[:2] == ("rev-parse", "HEAD"):
            return COMMIT.encode()
        return b""
    monkeypatch.setattr(cs, "_pf_git", git)
    intent = {"schema": cs.PREFIRE_INTENT_SCHEMA, "state": "PREFIRE_FIRST_MEASUREMENT_ONLY", "candidate_id": "fixture_candidate",
        "created_at_utc": "2026-09-10T01:00:00Z", "created_by": "fixture-producer", "producer_source_commit": COMMIT,
        "score_claim": False, "promotion_eligible": False, "timing_clearance": False,
        "contract": {"adjudication_memo": cs.prefire_file_reference(memo_path), "implementation_commit": COMMIT,
            "implementation_manifest": implementation, "implementation_manifest_sha256": implementation["sha256"],
            "amendment": amendment},
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
    from experiments.contest_auth_eval import _runtime_dependency_manifest
    from tac.deploy.modal.auth_eval import modal_uploaded_submission_dir_runtime_manifest

    auth, auth_path = auth_fixture(f, monkeypatch)
    upstream = f["repo"] / "upstream"
    upstream.mkdir(exist_ok=True)
    (upstream / "evaluate.py").write_bytes((Path(cs.__file__).resolve().parents[2] / "upstream/evaluate.py").read_bytes())
    local = _runtime_dependency_manifest(f["root"] / "inflate.sh", upstream)
    normal = modal_uploaded_submission_dir_runtime_manifest(local)
    worker = modal_uploaded_submission_dir_runtime_manifest(local,
        remote_submission_dir="/retained/fixture/out/submission_dir")
    worker["runtime_files_sha256"] = local["runtime_files_sha256"]
    monkeypatch.setattr(cs, "__file__", str(f["repo"] / "src/tac/candidate_seal.py"))
    cs.reserve_first_measurement(auth, repo=f["repo"])
    cs.transition_first_measurement(auth, "SPAWNED", call_id="fc-fixture", repo=f["repo"])
    context = {"prefire_intent_sha256": f["intent"]["intent_sha256"],
        "first_measurement_authorization_sha256": auth["authorization_sha256"],
        "lane_id": auth["lane_id"], "instance_job_id": auth["instance_job_id"], "receipt_path": auth["receipt_path"],
        "inflate_timeout_seconds": 1800, "evaluate_timeout_seconds": 1800,
        "modal_function_timeout_seconds": 4800, "poller_deadline_seconds": 5400,
        "expected_runtime_content_tree_sha256": normal["runtime_content_tree_sha256"],
        "source_snapshot": {"schema": "modal_source_snapshot.v1", "complete": True,
            "verify_failures": [], "missing_in_source": [], "files_digest": "a" * 64},
        "exact_argv": ["modal", "--gpu", "T4", "--scorer-device", "cuda", "--inflate-device", "auto",
            "--inflate-timeout", "1800", "--evaluate-timeout", "1800", "--claim-policy", "require_active",
            "--lane-id", auth["lane_id"], "--instance-job-id", auth["instance_job_id"], "--output-dir", auth["output_dir"],
            "--expected-archive-sha256", f["intent"]["candidate"]["archive"]["sha256"],
            "--expected-runtime-content-tree-sha256", normal["runtime_content_tree_sha256"]]}
    write(Path(auth["output_dir"]) / "FIRST_MEASUREMENT_CONTEXT.json", context)
    request = write(Path(auth["output_dir"]) / "modal_cuda_auth_eval_local_request.json",
        {**context, "expected_runtime_tree_sha256": normal["runtime_tree_sha256"]})
    path = _t4_receipt(f["root"], Path(auth["receipt_path"]), seconds=900,
        **context, call_id="fc-fixture", worker_request=request, avg_segnet_dist=0.001, avg_posenet_dist=0.0,
        archive_size_bytes=f["intent"]["candidate"]["archive"]["bytes"], score_claim=False,
        promotion_eligible=False, adjudication_required=True)
    result = json.loads(path.read_text())
    artifact = json.loads(result["artifacts"]["contest_auth_eval.json"])
    artifact.update(avg_segnet_dist=0.001, avg_posenet_dist=0.0)
    result["artifacts"]["contest_auth_eval.json"] = json.dumps(artifact)
    result["artifacts"]["provenance.json"] = json.dumps({
        "inflate_script": worker["runtime_root"] + "/inflate.sh", "inflate_runtime_manifest": worker})
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
    custody = document["first_measurement_runtime_custody"]
    assert document["decode_wall_clock"]["first_measurement_runtime_custody"] == custody
    assert {k: v for k, v in document["decode_wall_clock"].items()
            if k != "first_measurement_runtime_custody"} == build_t4_direct_leg(t4_receipt_path=receipt,
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


def test_first_measurement_argv_and_manifest_name_both_real_digests(fixture, monkeypatch, capsys):
    from tac.decode_wall_clock import measure_t4_runtime_digest

    auth, path, tool, _ = prepare_fire(fixture, monkeypatch)
    (fixture["repo"] / "upstream").symlink_to(Path(__file__).resolve().parents[3] / "upstream", target_is_directory=True)
    foreign = fixture["repo"] / "foreign_cwd"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    assert tool.main(["--first-measurement", str(fixture["path"]),
        "--first-measurement-authorization", str(path), "--dry-run"]) == 0
    context = json.loads(capsys.readouterr().out)
    cmd = context["argv"]
    expected = measure_t4_runtime_digest(fixture["root"])
    content = tool.measure_fire_runtime_digests(fixture["root"])["modal_uploaded_runtime"]["runtime_content_tree_sha256"]
    intent_digest = cs.measure_runtime_digest(fixture["root"]).sha256
    assert expected != intent_digest
    assert cmd[cmd.index("--expected-runtime-tree-sha256") + 1] == expected
    assert cmd[cmd.index("--expected-runtime-content-tree-sha256") + 1] == content
    assert context["expected_runtime_content_tree_sha256"] == content
    assert context["runtime_digests"] == {
        "intent_runtime": {"digest_definition": "tac.candidate_seal.measure_runtime_digest", "sha256": intent_digest},
        "expected_runtime_tree": {"digest_definition": "tac.decode_wall_clock.measure_t4_runtime_digest", "sha256": expected},
        "expected_runtime_content_tree": {
            "digest_definition": "tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest.runtime_content_tree_sha256",
            "sha256": content},
    }
    manifest = tool.first_measurement_manifest(tool.axis_spec("cuda"), context)
    manifest_path = tool.write_fire_manifest(foreign, manifest)
    assert json.loads(manifest_path.read_text())["runtime_digests"] == context["runtime_digests"]
    assert not cs.first_measurement_consumption_path(auth, fixture["repo"]).exists()


def test_first_measurement_still_refuses_changed_intent_runtime_digest(fixture, monkeypatch):
    _, path, tool, _ = prepare_fire(fixture, monkeypatch)
    fixture["intent"]["candidate"]["runtime"]["sha256"] = "f" * 64
    write(fixture["path"], sign(fixture["intent"]))
    assert tool.main(["--first-measurement", str(fixture["path"]),
        "--first-measurement-authorization", str(path), "--dry-run"]) == 9
    refusal = json.loads(next(fixture["repo"].glob("PREFIRE_REFUSAL_*.json")).read_text())
    assert refusal["code"] == "PREFIRE_IDENTITY_DRIFT_REFUSED"
    assert "runtime sha256 differs" in refusal["detail"]


def test_content_pin_accepts_relocation_but_refuses_one_changed_byte(fixture):
    from experiments.contest_auth_eval import _runtime_dependency_manifest, _validate_expected_runtime_tree
    from tac.deploy.modal.auth_eval import modal_uploaded_submission_dir_runtime_manifest

    root = fixture["root"]
    upstream = Path(__file__).resolve().parents[3] / "upstream"
    local = _runtime_dependency_manifest(root / "inflate.sh", upstream)
    normal = modal_uploaded_submission_dir_runtime_manifest(local)
    retained = modal_uploaded_submission_dir_runtime_manifest(local, remote_submission_dir="/retained/run/out/submission_dir")
    expected = normal["runtime_content_tree_sha256"]
    assert retained["runtime_tree_sha256"] != normal["runtime_tree_sha256"]
    assert retained["runtime_content_tree_sha256"] == expected
    prov = {"inflate_runtime_manifest": retained}
    _validate_expected_runtime_tree(prov, None, expected)
    assert prov["inflate_runtime_manifest"]["runtime_tree_sha256"] == retained["runtime_tree_sha256"]
    with pytest.raises(RuntimeError, match="runtime tree hash mismatch"):
        _validate_expected_runtime_tree(prov, normal["runtime_tree_sha256"])
    with pytest.raises(RuntimeError, match="runtime tree hash mismatch"):
        _validate_expected_runtime_tree(prov, normal["runtime_tree_sha256"], expected)
    source = root / "inflate.py"
    source.write_bytes(source.read_bytes() + b"\n")
    changed = modal_uploaded_submission_dir_runtime_manifest(
        _runtime_dependency_manifest(root / "inflate.sh", upstream), remote_submission_dir=retained["runtime_root"])
    with pytest.raises(RuntimeError, match="runtime content tree hash mismatch"):
        _validate_expected_runtime_tree({"inflate_runtime_manifest": changed}, None, expected)
    with pytest.raises(RuntimeError, match="runtime content tree hash mismatch"):
        _validate_expected_runtime_tree({}, None, expected)


@pytest.mark.parametrize("changed_pin", [None, "argv", "context"])
def test_local_content_pin_validation_and_worker_argv(fixture, changed_pin):
    """Execute actual local guard and worker argv branch; never invoke a provider."""
    import ast

    source = Path(__file__).resolve().parents[3] / "experiments/modal_auth_eval.py"
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    guard = next(n for n in main.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
        and n.test.id == "first_context" and "runtime content digest differs" in ast.unparse(n))
    content = load_tool("fire_modal_auth_eval").measure_fire_runtime_digests(fixture["root"])["modal_uploaded_runtime"]["runtime_content_tree_sha256"]
    ns = {"first_context": {"expected_runtime_content_tree_sha256": "f" * 64 if changed_pin == "context" else content},
        "expected_runtime_content_tree_sha256": content,
        "requested_runtime_content_tree_sha256": "f" * 64 if changed_pin == "argv" else content, "_pf_require": cs._pf_require}
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec")
    if changed_pin:
        with pytest.raises(cs.PrefireRefusal, match="PREFIRE_IDENTITY_DRIFT_REFUSED"):
            exec(code, ns)
        return
    exec(code, ns)
    inner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_auth_eval_inner")
    branch = next(n for n in inner.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
        and n.test.id == "expected_runtime_content_tree_sha256")
    ns = {"cmd": [], "expected_runtime_content_tree_sha256": content,
        "retained_work_root": "/retained/run", "expected_runtime_tree_sha256": "a" * 64}
    code = compile(ast.Module(body=[branch], type_ignores=[]), str(source), "exec")
    exec(code, ns)
    assert ns["cmd"] == ["--expected-runtime-content-tree-sha256", content]
    ns.update(cmd=[], expected_runtime_content_tree_sha256="", retained_work_root="")
    exec(code, ns)
    assert ns["cmd"] == ["--expected-runtime-tree-sha256", "a" * 64]
    ns.update(expected_runtime_content_tree_sha256=content)
    with pytest.raises(ValueError, match="requires first-measurement retention"):
        exec(code, ns)


def test_content_pin_survives_fail_closed_wrapper():
    import ast

    source = Path(__file__).resolve().parents[3] / "experiments/modal_auth_eval.py"
    node = next(n for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_auth_eval_fail_closed")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"_run_auth_eval_inner": lambda **kwargs: kwargs}
    exec(compile(module, str(source), "exec"), ns)
    result = ns["_run_auth_eval_fail_closed"](archive_bytes=b"fixture", archive_sha256="a" * 64,
        archive_size_bytes=7, inflate_sh_rel="inflate.sh", submission_dir_zip_bytes=None,
        submission_dir_zip_sha256=None, source_repo_commit=COMMIT, inflate_timeout=1800, evaluate_timeout=1800,
        retained_work_root="/retained/run", expected_runtime_content_tree_sha256="b" * 64)
    assert result["expected_runtime_content_tree_sha256"] == "b" * 64


def test_authorize_tool_imports_experiments_from_a_foreign_cwd(tmp_path):
    """The standalone consumer must close its transitive import graph without PYTHONPATH."""
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[3]
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('authorize_tool', "
        f"{str(repo / 'tools' / 'authorize_candidate_first_measurement.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "from experiments.contest_auth_eval import _runtime_dependency_manifest\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


@pytest.mark.parametrize("first_measurement", [True, False])
def test_local_registration_uses_repo_ledger_when_imported_from_snapshot(fixture, monkeypatch, first_measurement):
    """Execute the real call expression and ledger append, without a provider spawn."""
    import ast

    import tac.deploy.modal.call_id_ledger as ledger

    auth, _ = auth_fixture(fixture, monkeypatch)
    source = Path(__file__).resolve().parents[3] / "experiments/modal_auth_eval.py"
    main = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    call = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "register_dispatched_call_id_fail_closed")
    snapshot = fixture["repo"] / "snapshot"
    snapshot.mkdir()
    snapshot_ledger = snapshot / ".omx/state/modal_call_id_ledger.jsonl"
    monkeypatch.setattr(ledger, "MODAL_CALL_ID_LEDGER_PATH", snapshot_ledger)
    monkeypatch.setattr(ledger, "MODAL_CALL_ID_LEDGER_LOCK", snapshot_ledger.with_suffix(".jsonl.lock"))
    monkeypatch.setattr(ledger, "MODAL_CALL_ID_LEDGER_INDEX_PATH", snapshot / "index.json")
    monkeypatch.chdir(fixture["repo"])
    monkeypatch.setenv("PACT_MODAL_SOURCE_ROOT", str(snapshot))
    repo_ledger = fixture["repo"] / ".omx/state/modal_call_id_ledger.jsonl"
    before = repo_ledger.read_bytes()
    cs.reserve_first_measurement(auth, repo=fixture["repo"])
    context = {"prefire_intent_sha256": fixture["intent"]["intent_sha256"],
        "first_measurement_authorization_sha256": auth["authorization_sha256"],
        "intent_file_sha256": auth["intent"]["file_sha256"], "intent_file_bytes": auth["intent"]["file_bytes"],
        "instance_job_id": auth["instance_job_id"]}
    namespace = {"Path": Path, "register_dispatched_call_id_fail_closed": ledger.register_dispatched_call_id_fail_closed,
        "call_id": "fc-ffi5-fixture", "lane_id": auth["lane_id"], "gpu_key": "T4", "axis_label": "contest_cuda",
        "first_context": context if first_measurement else None, "inflate_timeout": 1800, "evaluate_timeout": 1800,
        "source_repo_commit": COMMIT, "claim_agent": "MAIN", "archive_sha256": fixture["intent"]["candidate"]["archive"]["sha256"],
        "pairing": {}}
    eval(compile(ast.Expression(call), str(source), "eval"), namespace)
    assert repo_ledger.read_bytes().startswith(before)
    row = json.loads(repo_ledger.read_text().splitlines()[-1])
    assert row["call_id"] == "fc-ffi5-fixture" and row["event_type"] == "dispatched"
    assert not snapshot_ledger.exists()
    assert not snapshot_ledger.with_suffix(".jsonl.lock").exists()
    assert repo_ledger.with_suffix(".jsonl.lock").exists()
    if first_measurement:
        assert cs.transition_first_measurement(auth, "SPAWNED", call_id=row["call_id"], repo=fixture["repo"])["state"] == "SPAWNED"


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
        "implementation_manifest": contract["implementation_manifest"], "amendments": [contract["amendment"]]})
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
        assert kwargs["expected_runtime_content_tree_sha256"] == "d" * 64
        assert (root / "INPUT_archive.zip").read_bytes() == b"archive fixture"
        (root / "raw").write_bytes(b"raw fixture")
        return {"passed": True, "score_claim": True, "promotion_eligible": True, "artifacts": {}}
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=32 * 1024**3))
    namespace = {"Path": Path, "AUTH_CACHE_VOLUME_ROOT": tmp_path, "AUTH_CACHE_VOLUME_NAME": "fixture-volume",
        "auth_cache_vol": SimpleNamespace(commit=lambda: commits.append(True)), "_run_auth_eval_fail_closed": inner}
    exec(compile(module, str(source), "exec"), namespace)
    context = {"first_measurement_authorization_sha256": "a" * 64, "prefire_intent_sha256": "b" * 64,
               "expected_runtime_content_tree_sha256": "d" * 64, "exact_argv": ["fixture-worker"]}
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


def test_prefire_risk_accepts_only_regenerated_manifest_indirection(fixture):
    root = fixture["root"]
    risk = json.loads(Path(fixture["intent"]["evidence"]["timing_risk"]["path"]).read_text())
    reference = Path(risk["diagnostic_reference_receiver"]["path"])
    for tree in (root, reference):
        for line in (tree / "MANIFEST.sha256").read_text().splitlines():
            digest, rel = line.split("  ", 1)
            assert digest == cs.sha256_file(tree / rel)
    assert (root / "MANIFEST.sha256").read_bytes() != (reference / "MANIFEST.sha256").read_bytes()
    assert cs.prefire_risk_receiver_rows(root) == cs.prefire_risk_receiver_rows(reference)
    assert cs.measure_prefire_risk_receiver_digest(root) == cs.measure_prefire_risk_receiver_digest(reference)
    assert measure_receiver_digest(root) != measure_receiver_digest(reference)
    assert "MANIFEST.sha256" in {row[0] for row in cs.prefire_receiver_rows(root)}
    assert "MANIFEST.sha256" in cs.measure_runtime_digest(root).file_map()
    assert validate(fixture)["schema"] == cs.PREFIRE_INTENT_SCHEMA
    for field in ("source_receiver", "candidate_receiver", "diagnostic_reference_receiver"):
        original = risk[field]["digest_definition"]
        risk[field]["digest_definition"] = "unversioned"
        risk["risk_sha256"] = cs.prefire_digest(risk, "risk_sha256")
        ref = write(fixture["store"] / "wrong_definition.json", risk)
        with pytest.raises(cs.PrefireRefusal, match="PREFIRE_RISK_EVIDENCE_REFUSED"):
            cs.validate_prefire_risk(ref, fixture["intent"], repo=fixture["repo"])
        risk[field]["digest_definition"] = original
    for field, key in (("source_receiver", "t4_direct_sha256"),
                       ("diagnostic_reference_receiver", "receipt_sha256")):
        original = risk[field][key]
        risk[field][key] = "e" * 64
        risk["risk_sha256"] = cs.prefire_digest(risk, "risk_sha256")
        ref = write(fixture["store"] / "wrong_legacy_join.json", risk)
        with pytest.raises(cs.PrefireRefusal, match="PREFIRE_RISK_EVIDENCE_REFUSED"):
            cs.validate_prefire_risk(ref, fixture["intent"], repo=fixture["repo"])
        risk[field][key] = original
    delta_path = Path(risk["receiver_delta_manifest"]["path"])
    delta = json.loads(delta_path.read_text())
    for field in ("files", "source_receiver_sha256", "candidate_receiver_sha256"):
        changed = dict(delta)
        changed[field] = delta["files"][:-1] if field == "files" else "e" * 64
        risk["receiver_delta_manifest"] = write(delta_path, changed)
        risk["risk_sha256"] = cs.prefire_digest(risk, "risk_sha256")
        ref = write(fixture["store"] / "wrong_delta.json", risk)
        with pytest.raises(cs.PrefireRefusal, match="complete normalized receiver delta differs"):
            cs.validate_prefire_risk(ref, fixture["intent"], repo=fixture["repo"])


def test_prefire_risk_refuses_nonpin_inflate_byte_change(fixture):
    ref = fixture["intent"]["evidence"]["timing_risk"]
    assert cs.validate_prefire_risk(ref, fixture["intent"], repo=fixture["repo"])
    path = fixture["root"] / "inflate.py"
    before = path.read_bytes()
    # One byte outside either AST pin span, leaving valid Python.
    after = before.replace(b"stand-in", b"stand-In", 1)
    assert sum(a != b for a, b in zip(before, after, strict=True)) == 1
    path.write_bytes(after)
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_RISK_EVIDENCE_REFUSED: receiver risk endpoints differ"):
        cs.validate_prefire_risk(ref, fixture["intent"], repo=fixture["repo"])
    for after in (before + b"\nARCHIVE_BYTES = 1\n",
                  before.replace(b"ARCHIVE_BYTES = ", b"ARCHIVE_BYTES = 1 + ", 1),
                  before.replace(b"ARCHIVE_BYTES", b"REMOVED_BYTES", 1)):
        path.write_bytes(after)
        with pytest.raises(cs.PrefireRefusal, match="PREFIRE_RISK_EVIDENCE_REFUSED"):
            cs.validate_prefire_risk(ref, fixture["intent"], repo=fixture["repo"])


def test_prefire_risk_manifest_exclusion_does_not_bypass_dependency_manifest_validation(fixture):
    intent, root = fixture["intent"], fixture["root"]
    assert validate(fixture)
    digest = cs.measure_prefire_risk_receiver_digest(root)
    (root / "MANIFEST.sha256").write_text("stale raw dependency manifest\n")
    assert cs.measure_prefire_risk_receiver_digest(root) == digest
    # Rebind outer identities so refusal must come from the independent dependency
    # listing, which still pins the formerly valid manifest bytes.
    runtime = cs.measure_runtime_digest(root)
    receiver = measure_receiver_digest(root)
    intent["candidate"]["runtime"] = {"path": str(root), **runtime.to_dict()}
    intent["candidate"]["normalized_receiver"]["sha256"] = receiver
    smoke = _public_smoke(root, root / "archive.zip")
    for group in ("public_path_probes", "inflate_sh_smokes"):
        intent["public_entrypoint_smoke"][group]["candidate"] = smoke[group]["candidate"]
    for name in ("candidate_manifest", "manifest_validation", "twin_encode", "archive_parseback",
                 "raw_identity_n600", "literal_census"):
        path = Path(intent["evidence"][name]["path"])
        doc = json.loads(path.read_text())
        doc.update(runtime_sha256=runtime.sha256, receiver_sha256=receiver)
        if name == "manifest_validation":
            doc["manifest"] = intent["evidence"]["candidate_manifest"]
        intent["evidence"][name] = write(path, doc)
    write(fixture["path"], sign(intent))
    # ddm_pr18 moves this refusal EARLIER and makes it specific: the intent's own tree is now
    # checked against its listing at the identity stage, so a stale listing never reaches the
    # dependency gate. The dependency gate itself is unchanged and is exercised below.
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_IDENTITY_DRIFT_REFUSED: derived listing invalid"):
        validate(fixture)


def test_dependency_manifest_gate_still_refuses_an_unverified_listing(fixture):
    """The pr18 identity check does not replace the independent verification receipt gate."""
    intent = fixture["intent"]
    assert validate(fixture)
    path = Path(intent["evidence"]["manifest_validation"]["path"])
    doc = json.loads(path.read_text())
    doc["all_hashes_passed"] = False
    intent["evidence"]["manifest_validation"] = write(path, doc)
    write(fixture["path"], sign(intent))
    with pytest.raises(cs.PrefireRefusal,
                       match="PREFIRE_NON_TIMING_GATE_REFUSED: dependency manifest/verification not complete"):
        validate(fixture)


def test_prefire_amendment_keeps_legacy_t4_direct_manifest_inclusive(fixture):
    from tac.decode_wall_clock import validate_decode_wall_clock

    risk = json.loads(Path(fixture["intent"]["evidence"]["timing_risk"]["path"]).read_text())
    leg = json.loads(Path(risk["source_t4_leg"]["path"]).read_text())
    root, archive = Path(leg["runtime_dir"]), Path(leg["archive_path"])
    assert not validate_decode_wall_clock(leg, runtime_dir=root, archive_path=archive)[0]
    risk_before = cs.measure_prefire_risk_receiver_digest(root)
    runtime_before = cs.measure_runtime_digest(root).sha256
    manifest = root / "MANIFEST.sha256"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    assert cs.measure_prefire_risk_receiver_digest(root) == risk_before
    assert cs.measure_runtime_digest(root).sha256 != runtime_before
    assert measure_receiver_digest(root) != leg["receiver_sha256"]
    assert validate_decode_wall_clock(leg, runtime_dir=root, archive_path=archive)[0]


def _contract_git_history(f, monkeypatch):
    """Real loose Git objects in the isolated fixture; no shared index or processes."""
    import hashlib
    import zlib

    repo, intent = f["repo"], f["intent"]
    def obj(kind, data):
        raw = f"{kind} {len(data)}\0".encode() + data
        digest = hashlib.sha1(raw).hexdigest()
        path = repo / ".git/objects" / digest[:2] / digest[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zlib.compress(raw))
        return digest
    def tree(files):
        entries = {}
        for path, data in files.items():
            name, sep, rest = path.partition("/")
            if sep:
                entries.setdefault(name, {})[rest] = data
            else:
                entries[name] = data
        body = b""
        for name, data in sorted(entries.items(), key=lambda item: item[0] + ("/" if isinstance(item[1], dict) else "")):
            directory = isinstance(data, dict)
            digest = tree(data) if directory else obj("blob", data)
            body += ("40000" if directory else "100644").encode() + b" " + name.encode() + b"\0" + bytes.fromhex(digest)
        return obj("tree", body)
    def commit(files, parent=None, when="2026-09-10T02:00:00+00:00"):
        timestamp = int(datetime.fromisoformat(when).timestamp())
        body = f"tree {tree(files)}\n" + (f"parent {parent}\n" if parent else "")
        body += f"author Fixture <fixture@example.invalid> {timestamp} +0000\n"
        body += f"committer Fixture <fixture@example.invalid> {timestamp} +0000\n\ncontract fixture\n"
        digest = obj("commit", body.encode())
        (repo / ".git/HEAD").write_text(digest + "\n")
        return digest
    base_files = dict.fromkeys(cs.PREFIRE_IMPLEMENTATION_PATHS, b"# synthetic committed source fixture\n")
    base_files[cs.PREFIRE_MEMO] = (repo / cs.PREFIRE_MEMO).read_bytes()
    base = commit(base_files, when="2026-09-09T00:00:00+00:00")
    production = commit(base_files, base, "2026-09-10T00:00:00+00:00")
    amended_files = {name: (repo / name).read_bytes() for name in base_files}
    memo = Path(intent["contract"]["amendment"]["adjudication_memo"]["path"])
    amended_files[memo.relative_to(repo).as_posix()] = memo.read_bytes()
    amended = commit(amended_files, production, "2026-09-10T00:30:00+00:00")
    intent["contract"]["implementation_commit"] = base
    intent["contract"]["amendment"]["implementation_commit"] = amended
    intent["producer_source_commit"] = production
    path = Path(intent["evidence"]["candidate_manifest"]["path"])
    manifest = json.loads(path.read_text())
    manifest["producer_source_commit"] = production
    intent["evidence"]["candidate_manifest"] = write(path, manifest)
    path = Path(intent["evidence"]["manifest_validation"]["path"])
    verification = json.loads(path.read_text())
    verification["manifest"] = intent["evidence"]["candidate_manifest"]
    intent["evidence"]["manifest_validation"] = write(path, verification)
    path = Path(intent["evidence"]["twin_encode"]["path"])
    twin = json.loads(path.read_text())
    for index, ref in enumerate(twin["executions"]):
        execution_path = Path(ref["path"])
        execution = json.loads(execution_path.read_text())
        execution["producer_source_commit"] = production
        twin["executions"][index] = write(execution_path, execution)
    intent["evidence"]["twin_encode"] = write(path, twin)
    write(repo / cs.PREFIRE_FREEZE, {"implementation_commit": base,
        "implementation_manifest": intent["contract"]["implementation_manifest"],
        "amendments": [intent["contract"]["amendment"]]})
    write(f["path"], sign(intent))
    def land():
        files = {name: (repo / name).read_bytes() for name in amended_files}
        files[cs.PREFIRE_FREEZE] = (repo / cs.PREFIRE_FREEZE).read_bytes()
        files[f["path"].relative_to(repo).as_posix()] = f["path"].read_bytes()
        return commit(files, amended)
    head = land()
    monkeypatch.setattr(cs, "_pf_git", REAL_PF_GIT)
    orphan = commit(base_files, when="2026-09-08T00:00:00+00:00")
    (repo / ".git/HEAD").write_text(head + "\n")
    return {"base": base, "production": production, "amended": amended, "head": head,
            "orphan": orphan, "land": land}


def test_prefire_amendment_allows_pinned_old_evidence_but_refuses_pre_amendment_intent(fixture, monkeypatch):
    history = _contract_git_history(fixture, monkeypatch)
    intent, repo = fixture["intent"], fixture["repo"]
    manifest_path = Path(intent["evidence"]["candidate_manifest"]["path"])
    original = manifest_path.read_bytes()
    assert validate(fixture)
    assert manifest_path.read_bytes() == original
    assert history["base"] != history["amended"]
    intent["created_at_utc"] = "2026-09-10T00:15:00Z"
    with pytest.raises(cs.PrefireRefusal, match="amendment implementation must precede new intent"):
        cs._pf_contract(intent, repo, None)
    intent["created_at_utc"] = "2026-09-10T01:00:00Z"
    for timestamp in ("2026-09-08T00:00:00Z", "2026-09-10T02:00:00Z"):
        manifest = json.loads(original)
        manifest["production_started_at_utc"] = timestamp
        intent["evidence"]["candidate_manifest"] = write(manifest_path, manifest)
        with pytest.raises(cs.PrefireRefusal, match="implementation must precede every producer timestamp"):
            cs._pf_contract(intent, repo, None)
    manifest_path.write_bytes(original)
    intent["evidence"]["candidate_manifest"] = cs.prefire_file_reference(manifest_path)
    intent["contract"].pop("amendment")
    with pytest.raises(cs.PrefireRefusal, match="contract fields differ"):
        cs._pf_contract(intent, repo, None)


def test_prefire_amendment_requires_latest_frozen_row_and_live_committed_sources(fixture, monkeypatch):
    history = _contract_git_history(fixture, monkeypatch)
    intent, repo = fixture["intent"], fixture["repo"]
    assert validate(fixture)
    freeze_path = repo / cs.PREFIRE_FREEZE
    freeze_bytes = freeze_path.read_bytes()
    source_path = repo / cs.PREFIRE_IMPLEMENTATION_PATHS[0]
    source_bytes = source_path.read_bytes()
    source_path.write_bytes(source_bytes + b"# drift\n")
    with pytest.raises(cs.PrefireRefusal, match="live file differs from committed blob"):
        cs._pf_contract(intent, repo, None)
    source_path.write_bytes(source_bytes)
    fixture["path"].write_bytes(fixture["path"].read_bytes() + b"\n")
    with pytest.raises(cs.PrefireRefusal, match="live file differs from committed blob"):
        cs._pf_contract(intent, repo, fixture["path"])
    write(fixture["path"], sign(intent))
    frozen = json.loads(freeze_bytes)
    frozen["amendments"].append({**frozen["amendments"][-1], "score_claim": False, "new_row": True})
    write(freeze_path, frozen)
    with pytest.raises(cs.PrefireRefusal, match="live file differs from committed blob"):
        cs._pf_contract(intent, repo, None)
    history["land"]()
    with pytest.raises(cs.PrefireRefusal, match="latest exact frozen amendment required"):
        cs._pf_contract(intent, repo, None)
    freeze_path.write_bytes(freeze_bytes)
    history["land"]()
    for contract in (intent["contract"], intent["contract"]["amendment"]):
        ref = contract["implementation_manifest"]
        path = Path(ref["path"])
        original = path.read_bytes()
        rows = json.loads(original)
        rows[0]["sha256"] = "e" * 64
        contract["implementation_manifest"] = write(path, rows)
        if contract is intent["contract"]:
            contract["implementation_manifest_sha256"] = contract["implementation_manifest"]["sha256"]
        frozen = json.loads(freeze_bytes)
        frozen["implementation_manifest"] = intent["contract"]["implementation_manifest"]
        frozen["amendments"] = [intent["contract"]["amendment"]]
        write(freeze_path, frozen)
        history["land"]()
        with pytest.raises(cs.PrefireRefusal, match="committed implementation drift"):
            cs._pf_contract(intent, repo, None)
        path.write_bytes(original)
        contract["implementation_manifest"] = ref
        intent["contract"]["implementation_manifest_sha256"] = intent["contract"]["implementation_manifest"]["sha256"]
    freeze_path.write_bytes(freeze_bytes)
    history["land"]()
    intent["contract"]["amendment"]["definition_change"]["raw_manifest_still_required"] = 1
    with pytest.raises(cs.PrefireRefusal, match="latest exact frozen amendment required"):
        cs._pf_contract(intent, repo, None)
    intent["contract"]["amendment"]["definition_change"]["raw_manifest_still_required"] = True
    intent["producer_source_commit"] = history["orphan"]
    with pytest.raises(cs.PrefireRefusal, match="commit ancestry differs"):
        cs._pf_contract(intent, repo, None)
    intent["producer_source_commit"] = history["production"]
    intent["contract"]["amendment"]["implementation_commit"] = history["orphan"]
    frozen = json.loads(freeze_bytes)
    frozen["amendments"] = [intent["contract"]["amendment"]]
    write(freeze_path, frozen)
    history["land"]()
    with pytest.raises(cs.PrefireRefusal, match="commit ancestry differs"):
        cs._pf_contract(intent, repo, None)
    intent["contract"]["amendment"]["implementation_commit"] = history["amended"]
    freeze_path.write_bytes(freeze_bytes)
    history["land"]()
    (repo / ".git/HEAD").write_text(history["production"] + "\n")
    # Keep the freeze query available so this direction specifically attacks ancestry.
    real_git = cs._pf_git
    def old_head(repo, *args):
        if args == ("show", "HEAD:" + cs.PREFIRE_FREEZE):
            return freeze_bytes
        return real_git(repo, *args)
    monkeypatch.setattr(cs, "_pf_git", old_head)
    with pytest.raises(cs.PrefireRefusal, match="commit ancestry differs"):
        cs._pf_contract(intent, repo, None)


def test_first_measurement_manifest_is_axis_tagged_and_writable(tmp_path):
    """rlc5 real control (2026-09-10): the first-measurement manifest must carry the axis tags
    ``write_fire_manifest`` refuses without, or the reserved nonce is spent on a refusal."""
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location("fire_tool", _P("tools/fire_modal_auth_eval.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    manifest = tool.first_measurement_manifest(tool.axis_spec("cuda"), {"lane_id": "l", "instance_job_id": "j"})
    assert manifest["axis"] == "cuda" and manifest["evidence_axis_tag"] == "[contest-CUDA]"
    assert manifest["score_axis"] == "contest_cuda" and manifest["stage5_entrypoint"]
    path = tool.write_fire_manifest(tmp_path, manifest)
    assert path.exists() and path.name == "FIRE_MANIFEST.json"


@pytest.mark.parametrize("surface", ["context", "request", "result", "worker", "missing_provenance",
    "root", "tree", "files_digest", "file_count", "worker_file", "runtime_byte",
    "missing_argv", "duplicate_argv", "equals_argv", "wrong_argv"])
def test_completion_runtime_custody_refuses_each_join_drift(fixture, monkeypatch, surface):
    auth, _, receipt = completion_fixture(fixture, monkeypatch)
    output = Path(auth["output_dir"])
    paths = {"context": output / "FIRST_MEASUREMENT_CONTEXT.json",
             "request": output / "modal_cuda_auth_eval_local_request.json", "result": receipt}
    documents = {key: json.loads(path.read_text()) for key, path in paths.items()}
    if surface in paths:
        documents[surface]["expected_runtime_content_tree_sha256"] = "0" * 64
    elif surface.endswith("argv"):
        flag = "--expected-runtime-content-tree-sha256"
        argv = documents["request"]["exact_argv"]
        index = argv.index(flag)
        if surface == "missing_argv":
            del argv[index:index + 2]
        elif surface == "duplicate_argv":
            argv += argv[index:index + 2]
        elif surface == "equals_argv":
            argv += [flag + "=" + argv[index + 1]]
        else:
            argv[index + 1] = "0" * 64
        for document in documents.values():
            document["exact_argv"] = argv
    elif surface == "missing_provenance":
        documents["result"]["artifacts"].pop("provenance.json")
    elif surface == "runtime_byte":
        with (fixture["root"] / "inflate.sh").open("ab") as stream:
            stream.write(b"\n# real changed runtime byte\n")
    else:
        provenance = json.loads(documents["result"]["artifacts"]["provenance.json"])
        worker = provenance["inflate_runtime_manifest"]
        if surface == "worker":
            worker["runtime_content_tree_sha256"] = "0" * 64
        elif surface == "root":
            worker["runtime_root"] = "/different/submission_dir"
            provenance["inflate_script"] = worker["runtime_root"] + "/inflate.sh"
        elif surface == "tree":
            worker["runtime_tree_sha256"] = "0" * 64
        elif surface == "files_digest":
            worker["runtime_files_sha256"] = "0" * 64
        elif surface == "file_count":
            worker["runtime_file_count"] += 1
        else:
            worker["files"][0]["sha256"] = "0" * 64
        documents["result"]["artifacts"]["provenance.json"] = json.dumps(provenance)
    for name, path in paths.items():
        write(path, documents[name])
    with pytest.raises(cs.PrefireRefusal, match="FIRST_MEASUREMENT_RESULT_REFUSED"):
        cs.validate_first_measurement_runtime_custody(runtime_dir=fixture["root"], output_dir=output,
            receipt_path=receipt)


def test_completion_seal_rechecks_custody_and_refuses_unequal_objects(fixture, monkeypatch):
    auth, auth_path, receipt = completion_fixture(fixture, monkeypatch)
    cs.transition_first_measurement(auth, "HARVESTED", call_id="fc-fixture", receipt_path=receipt, repo=fixture["repo"])
    output = fixture["repo"] / "completed.json"
    document = cs.complete_first_fire_intent(intent_path=fixture["path"], authorization_path=auth_path,
        receipt_path=receipt, out_path=output, repo=fixture["repo"], pointer_path=fixture["pointer"])
    custody = document["first_measurement_runtime_custody"]
    assert custody["normal_projected_runtime_tree_sha256"] != custody["retained_runtime_tree_sha256"]
    # JSON re-read removes Python aliasing: alter the seal alone, keeping the leg unchanged.
    altered = json.loads(output.read_text())
    altered["first_measurement_runtime_custody"]["runtime_file_count"] += 1
    altered["seal_sha256"] = cs.compute_seal_sha256(altered)
    write(output, altered)
    verdict = cs.validate_seal(output, pointer_path=fixture["pointer"], require_decode_wall_clock=True)
    assert not verdict.ok and "custody differ" in verdict.summary()
    write(output, document)
    context = Path(auth["output_dir"]) / "FIRST_MEASUREMENT_CONTEXT.json"
    changed = json.loads(context.read_text())
    changed["expected_runtime_content_tree_sha256"] = "0" * 64
    write(context, changed)
    verdict = cs.validate_seal(output, pointer_path=fixture["pointer"], require_decode_wall_clock=True)
    assert not verdict.ok and "context runtime content differs" in verdict.summary()


def _consumer_fix_row(f, history, previous, *, batch="fixture-consumers"):
    definition = f["intent"]["contract"]["amendment"]
    return {"schema": cs.PREFIRE_CONTRACT_CONSUMER_FIX_SCHEMA, "fix_batch_id": batch,
        "previous_row_sha256": cs.prefire_digest(previous),
        "definition_parent_sha256": cs.prefire_digest(definition), "definition_change": False,
        "consumer_fixes": [{"fix_id": batch + suffix, "scope": "candidate_seal completion custody",
            "reason": "retained worker relocated outside the normal upload root",
            "causal_commit": history["amended"], "tests": ["test_completion_runtime_custody_refuses_each_join_drift"]}
            for suffix in ("-content", "-projection")],
        "implementation_commit": history["amended"],
        "implementation_manifest": definition["implementation_manifest"],
        "adjudication_memo": definition["adjudication_memo"], "score_claim": False}


def _freeze_consumer_rows(f, history, rows, *, pin=True):
    path = f["repo"] / cs.PREFIRE_FREEZE
    frozen = json.loads(path.read_text())
    frozen["amendments"] = rows
    write(path, frozen)
    if pin:
        f["intent"]["contract"].update(amendment=rows[-1],
            definition_parent_sha256=cs.prefire_digest(rows[0]), latest_row_sha256=cs.prefire_digest(rows[-1]))
        write(f["path"], sign(f["intent"]))
    history["land"]()


def test_consumer_fix_batch_keeps_definition_and_refuses_stale_latest(fixture, monkeypatch):
    history = _contract_git_history(fixture, monkeypatch)
    definition = fixture["intent"]["contract"]["amendment"]
    row = _consumer_fix_row(fixture, history, definition)
    second = _consumer_fix_row(fixture, history, row, batch="second-consumers")
    # Legacy implementation snapshots do not create new definitions.
    legacy_snapshot = {**definition, "note": "same definition; later implementation snapshot"}
    row["previous_row_sha256"] = cs.prefire_digest(legacy_snapshot)
    second["previous_row_sha256"] = cs.prefire_digest(row)
    _freeze_consumer_rows(fixture, history, [definition, legacy_snapshot, row])
    assert validate(fixture)
    _freeze_consumer_rows(fixture, history, [definition, legacy_snapshot, row, second], pin=False)
    with pytest.raises(cs.PrefireRefusal, match="latest exact frozen amendment required"):
        validate(fixture)
    _freeze_consumer_rows(fixture, history, [definition, legacy_snapshot, row, second])
    assert validate(fixture)
    assert fixture["intent"]["contract"]["definition_parent_sha256"] == cs.prefire_digest(definition)


def _superseding_amendment(f, history, previous):
    """ddm_pr18's row: a NEW definition that may follow consumer fixes by pinning the chain."""
    definition = f["intent"]["contract"]["amendment"]
    return {"schema": cs.PREFIRE_CONTRACT_AMENDMENT_SCHEMA,
        "amendment_id": cs.PREFIRE_CONTRACT_AMENDMENT_ID_PR18,
        "adjudication_memo": definition["adjudication_memo"],
        "definition_change": dict(cs.PREFIRE_AMENDMENT_DEFINITIONS[cs.PREFIRE_CONTRACT_AMENDMENT_ID_PR18]),
        "definition_parent_sha256": cs.prefire_digest(definition),
        "previous_row_sha256": cs.prefire_digest(previous),
        "implementation_commit": history["amended"],
        "implementation_manifest": definition["implementation_manifest"], "score_claim": False}


def test_a_superseding_amendment_may_follow_consumer_fixes_and_becomes_the_parent(fixture, monkeypatch):
    history = _contract_git_history(fixture, monkeypatch)
    definition = fixture["intent"]["contract"]["amendment"]
    fix = _consumer_fix_row(fixture, history, definition)
    pr18 = _superseding_amendment(fixture, history, fix)
    _freeze_consumer_rows(fixture, history, [definition, fix, pr18])
    fixture["intent"]["contract"]["definition_parent_sha256"] = cs.prefire_digest(pr18)
    write(fixture["path"], sign(fixture["intent"]))
    history["land"]()
    assert validate(fixture)
    # A later consumer fix must now parent to the NEW definition, not the superseded one.
    after = _consumer_fix_row(fixture, history, pr18, batch="post-pr18-consumers")
    after["definition_parent_sha256"] = cs.prefire_digest(pr18)
    _freeze_consumer_rows(fixture, history, [definition, fix, pr18, after])
    fixture["intent"]["contract"]["definition_parent_sha256"] = cs.prefire_digest(pr18)
    write(fixture["path"], sign(fixture["intent"]))
    history["land"]()
    assert validate(fixture)
    stale = _consumer_fix_row(fixture, history, pr18, batch="stale-parent-consumers")
    _freeze_consumer_rows(fixture, history, [definition, fix, pr18, stale])
    with pytest.raises(cs.PrefireRefusal, match="consumer-fix definition parent differs"):
        validate(fixture)


@pytest.mark.parametrize("mutation,message", [
    ("definition", "amendment definition differs"),
    ("unknown_id", "unknown amendment id"),
    ("parent", "superseding amendment names a different definition parent"),
    ("previous", "superseding amendment breaks the append-only row chain"),
    ("legacy_after_fix", "definition snapshot after consumer fixes is unsupported"),
])
def test_a_superseding_amendment_cannot_reset_the_chain(fixture, monkeypatch, mutation, message):
    history = _contract_git_history(fixture, monkeypatch)
    definition = fixture["intent"]["contract"]["amendment"]
    fix = _consumer_fix_row(fixture, history, definition)
    pr18 = _superseding_amendment(fixture, history, fix)
    if mutation == "definition":
        pr18["definition_change"]["excluded_relative_paths"] = []
    elif mutation == "unknown_id":
        pr18["amendment_id"] = "ddm_unreviewed_definition"
    elif mutation == "parent":
        pr18["definition_parent_sha256"] = "0" * 64
    elif mutation == "previous":
        pr18["previous_row_sha256"] = "0" * 64
    else:
        pr18 = {**definition, "note": "a legacy snapshot replayed after a consumer fix"}
    _freeze_consumer_rows(fixture, history, [definition, fix, pr18])
    with pytest.raises(cs.PrefireRefusal, match=message):
        validate(fixture)


@pytest.mark.parametrize("mutation", ["previous", "parent", "definition", "receiver", "threshold", "evidence",
    "missing_id", "duplicate_id", "unsorted", "empty_tests", "empty_fixes", "causal_orphan",
    "implementation_orphan", "implementation_unreachable", "manifest_drift", "live_drift", "latest_pin", "parent_pin"])
def test_consumer_fix_typed_refusals_with_real_git(fixture, monkeypatch, mutation):
    history = _contract_git_history(fixture, monkeypatch)
    definition = fixture["intent"]["contract"]["amendment"]
    row = _consumer_fix_row(fixture, history, definition)
    if mutation in ("previous", "parent"):
        row["previous_row_sha256" if mutation == "previous" else "definition_parent_sha256"] = "0" * 64
    elif mutation == "definition":
        row["definition_change"] = {"new_rule": True}
    elif mutation in ("receiver", "threshold", "evidence"):
        row[mutation] = {"override": True}
    elif mutation == "missing_id":
        row["consumer_fixes"][0].pop("fix_id")
    elif mutation == "duplicate_id":
        row["consumer_fixes"][1]["fix_id"] = row["consumer_fixes"][0]["fix_id"]
    elif mutation == "unsorted":
        row["consumer_fixes"].reverse()
    elif mutation == "empty_tests":
        row["consumer_fixes"][0]["tests"] = []
    elif mutation == "empty_fixes":
        row["consumer_fixes"] = []
    elif mutation == "causal_orphan":
        row["consumer_fixes"][0]["causal_commit"] = history["orphan"]
    elif mutation.startswith("implementation_"):
        row["implementation_commit"] = history["orphan"] if mutation.endswith("orphan") else "0" * 40
    elif mutation == "manifest_drift":
        path = fixture["store"] / "drifted_manifest.json"
        manifest = json.loads(Path(row["implementation_manifest"]["path"]).read_text())
        manifest[0]["sha256"] = "0" * 64
        row["implementation_manifest"] = write(path, manifest)
    _freeze_consumer_rows(fixture, history, [definition, row])
    if mutation == "live_drift":
        with (fixture["repo"] / cs.PREFIRE_IMPLEMENTATION_PATHS[0]).open("ab") as stream:
            stream.write(b"# uncommitted drift\n")
    elif mutation in ("latest_pin", "parent_pin"):
        fixture["intent"]["contract"]["latest_row_sha256" if mutation == "latest_pin" else "definition_parent_sha256"] = "0" * 64
        write(fixture["path"], sign(fixture["intent"]))
        history["land"]()
    with pytest.raises(cs.PrefireRefusal, match="PREFIRE_CONTRACT_DRIFT_REFUSED"):
        validate(fixture)


def test_run3_retained_provenance_projection_and_terminal_refusal(tmp_path):
    """Real retained bytes, read-only; synthesized join envelope is explicitly not a harvest."""
    from experiments.contest_auth_eval import _runtime_dependency_manifest, _validate_expected_runtime_tree
    from tac.deploy.modal.auth_eval import modal_uploaded_submission_dir_runtime_manifest

    retained = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run3")
    provenance_path = retained / "provenance.json"
    if not provenance_path.is_file():
        pytest.skip("run3 retained provenance is not mounted on this host")
    before = cs.prefire_file_reference(provenance_path)
    assert before["sha256"] == "4cd0463214a90b8919489eabe77062df8720c5d8405faf2e3d5ef10ddbd74405"
    provenance = json.loads(provenance_path.read_text())
    context = json.loads((retained / "FIRST_MEASUREMENT_CONTEXT.json").read_text())
    intent = json.loads(Path(context["intent_path"]).read_text())
    root = Path(intent["candidate"]["runtime"]["path"])
    repo = Path(cs.__file__).resolve().parents[2]
    local = _runtime_dependency_manifest(root / "inflate.sh", repo / "upstream")
    normal = modal_uploaded_submission_dir_runtime_manifest(local)
    worker = provenance["inflate_runtime_manifest"]
    assert normal["runtime_content_tree_sha256"] == worker["runtime_content_tree_sha256"]
    assert local["runtime_files_sha256"] == worker["runtime_files_sha256"]
    assert normal["runtime_tree_sha256"] != worker["runtime_tree_sha256"]
    _validate_expected_runtime_tree(provenance, None, normal["runtime_content_tree_sha256"])
    with pytest.raises(RuntimeError, match="inflate runtime tree hash mismatch"):
        _validate_expected_runtime_tree(provenance, normal["runtime_tree_sha256"])
    # Run3 predates the complete content join. A synthetic envelope lets the REAL
    # unmodified provenance exercise the new projection, without upgrading run3.
    envelope = {"expected_runtime_content_tree_sha256": normal["runtime_content_tree_sha256"],
        "expected_runtime_tree_sha256": normal["runtime_tree_sha256"],
        "exact_argv": ["--expected-runtime-content-tree-sha256", normal["runtime_content_tree_sha256"]]}
    write(tmp_path / "FIRST_MEASUREMENT_CONTEXT.json", envelope)
    write(tmp_path / "modal_cuda_auth_eval_local_request.json", envelope)
    receipt = tmp_path / "synthetic_join_not_a_harvest.json"
    document = {**envelope, "artifacts": {"provenance.json": provenance_path.read_text()}}
    write(receipt, document)
    custody = cs.validate_first_measurement_runtime_custody(runtime_dir=root, output_dir=tmp_path, receipt_path=receipt)
    assert custody["retained_runtime_tree_sha256"] == worker["runtime_tree_sha256"]
    assert custody["runtime_file_count"] == 48
    # The normal-root tree is the wrong tree for the retained root even when content passes.
    worker["runtime_tree_sha256"] = normal["runtime_tree_sha256"]
    document["artifacts"]["provenance.json"] = json.dumps(provenance)
    write(receipt, document)
    with pytest.raises(cs.PrefireRefusal, match="retained-root projection runtime_tree_sha256 differs"):
        cs.validate_first_measurement_runtime_custody(runtime_dir=root, output_dir=tmp_path, receipt_path=receipt)
    auth = json.loads(Path(context["authorization_path"]).read_text())
    with pytest.raises(cs.PrefireRefusal, match="FIRST_MEASUREMENT_RESULT_REFUSED"):
        cs._pf_completion_facts(intent, auth, retained / "MODAL_REMOTE_RESULT.json", repo=repo)
    assert cs.prefire_file_reference(provenance_path) == before


def test_completed_seal_anchor_mirror_lifts_quarantine_only_with_a_valid_seal(tmp_path, monkeypatch):
    """rlc5 → move 44 (2026-09-10): a completed candidate_seal.v3 must reach experiments/results as an anchor
    mirror, and a receipt still carrying its quarantine keys must be refused by the poller's builder."""
    import importlib.util
    import json
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location("wcsam", _P("tools/write_completed_seal_anchor_mirror.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    poller = tool._load_poller()
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    result = {"prefire_intent_sha256": "x" * 64, "score_recomputed_from_components": 0.137, "expected_archive_sha256": "a" * 64,
              "expected_archive_size_bytes": 1, "gpu_model": "Tesla T4", "score_axis": "contest_cuda", "n_samples": 600}
    receipt.write_text(json.dumps(result))
    payload, blocker = poller.build_anchor_mirror(json.loads(receipt.read_text()), lane_id="l", source_receipt=receipt)
    assert payload is None and "quarantined" in blocker
    lifted = json.loads(receipt.read_text())
    for key in tool.QUARANTINE_KEYS:
        lifted.pop(key, None)
    payload, blocker = poller.build_anchor_mirror(lifted, lane_id="l", source_receipt=receipt)
    assert blocker is None and payload["score"] == 0.137 and payload["archive_sha256"] == "a" * 64
    bad_seal = tmp_path / "seal.json"
    bad_seal.write_text(json.dumps({"schema": "candidate_seal.v2"}))
    monkeypatch.setattr("tac.candidate_seal.validate_seal", lambda *a, **k: {"ok": False, "problems": ["nope"]})
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        tool.completed_seal_result(bad_seal)


def test_call_id_ledger_accepts_reconciliation_event_types():
    """pr16 §5 / pr17 §4: the reconciliation events must be representable in the canonical ledger taxonomy."""
    from tac.deploy.modal import call_id_ledger as L
    assert "reconciled_terminal_failure" in L.VALID_EVENT_TYPES and "reconciled_terminal_success" in L.VALID_EVENT_TYPES
    assert L.EVENT_RECONCILED_TERMINAL_FAILURE == "reconciled_terminal_failure"
