from __future__ import annotations

import json
from pathlib import Path

from tac.optimizer.generators.apogee_intn import ApogeeIntNGenerator
from tac.optimizer.sweep_plugin import (
    CandidateGenerator,
    DispatchSpec,
    list_generators,
    load_generator,
    register_generator,
    reset_registry,
    unregister_generator,
)


class _DummyGenerator(CandidateGenerator):
    name = "dummy"

    def __call__(self) -> list[dict]:
        return [{"candidate_id": "dummy", "archive_bytes": 1}]

    def build_dispatch(self, candidate: dict, *, label: str) -> DispatchSpec:
        return DispatchSpec(label=label, cmd=["echo", str(candidate["candidate_id"])])


def test_registry_loads_registered_generator() -> None:
    reset_registry()
    register_generator("dummy", _DummyGenerator)

    assert list_generators() == ["dummy"]
    assert isinstance(load_generator("dummy"), _DummyGenerator)

    unregister_generator("dummy")
    assert list_generators() == []


def test_apogee_intn_generator_reads_repack_metadata(tmp_path: Path) -> None:
    repack_dir = tmp_path / "experiments/results/apogee_int6_repack_20260504_claude"
    repack_dir.mkdir(parents=True)
    (repack_dir / "apogee_int6_archive.zip").write_bytes(b"zip")
    (repack_dir / "repack_metadata.json").write_text(
        json.dumps(
            {
                "archive_size_bytes": 170450,
                "rel_err_pct_per_weight": 3.25,
                "n_intn_layers": 11,
                "candidate_archive_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    generator = ApogeeIntNGenerator(repo=tmp_path)
    candidates = generator()

    assert [candidate["candidate_id"] for candidate in candidates] == ["apogee_int6"]
    candidate = candidates[0]
    assert candidate["ready_for_exact_eval_dispatch"] is False
    assert candidate["evidence_semantics"] == "byte_only_forensic"
    assert "missing_contest_faithful_distortion_model" in candidate["dispatch_blockers"]

    dispatch = generator.build_dispatch(candidate, label="dryrun-apogee-int6")
    assert dispatch.cwd == tmp_path.resolve()
    assert "--expected-archive-sha256" in dispatch.cmd
    assert "--expected-archive-size-bytes" in dispatch.cmd
