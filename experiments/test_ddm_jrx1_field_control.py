"""Regression for the observed parallel rename/census failure, on real payload bytes."""
import concurrent.futures
import importlib.util
import json
import multiprocessing
import uuid
from pathlib import Path


def _worker(module_path, directory, payload_path, worker):
    spec = importlib.util.spec_from_file_location("jrx1_test_driver", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = Path(payload_path).read_bytes()
    directory = Path(directory)
    # Repeated shared destinations exercise the same .pending name contention.
    facts = []
    for index in range(8):
        facts.append(module.locked_retain(directory / f"shared_{index}.zip", payload))
        facts.append(module.locked_retain(directory / f"worker_{worker}_{index}.zip", payload))
        module.stable_storage()
    return facts


def test_concurrent_retention_preserves_real_archive():
    module_path = Path(__file__).with_name("ddm_jrx1_field_control.py")
    payload = Path("/Volumes/APDataStore/pact/ddm_pd3/candidate/candidate_runtime/archive.zip")
    directory = Path("/Volumes/APDataStore/pact/ddm_jrx1/regression") / uuid.uuid4().hex
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(_worker, str(module_path.resolve()), str(directory), str(payload), i)
                   for i in range(16)]
        facts = [row for f in futures for row in f.result(timeout=120)]
    expected = payload.read_bytes()
    payloads = [directory / f"shared_{i}.zip" for i in range(8)]
    payloads += [directory / f"worker_{w}_{i}.zip" for w in range(16) for i in range(8)]
    actual = {p for p in directory.glob("*.zip") if not p.name.startswith("._")}
    assert actual == set(payloads)
    assert not list(directory.glob("*.pending"))
    for path in payloads:
        assert path.read_bytes() == expected
    (directory / "RESULT.json").write_text(json.dumps({
        "status": "PASS", "workers": 8, "tasks": 16, "unique_payloads": 136, "facts": facts,
        "appledouble_metadata_files": len(list(directory.glob("._*"))),
        "mechanism": "simultaneous same-destination and disjoint-destination writes plus full census",
    }, indent=2) + "\n")
