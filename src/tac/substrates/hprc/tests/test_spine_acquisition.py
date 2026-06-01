# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tac.archive_byte_profile import contest_rate_term
from tac.substrates.hprc.representation_spine import (
    HprcRepresentationFamily,
    build_generic_neural_spine_packet,
    write_representation_spine_projection,
)
from tac.substrates.hprc.spine_acquisition import build_spine_acquisition_report

REPO = Path(__file__).resolve().parents[5]


def test_acquisition_ranks_shared_spine_under_hard_byte_ceiling(tmp_path: Path) -> None:
    rnerv = _projection(
        tmp_path / "rnerv",
        family=HprcRepresentationFamily.RNERV,
        decoder=b"d" * 40,
        latents=b"l" * 20,
    )
    pact_vq = _projection(
        tmp_path / "pvq",
        family=HprcRepresentationFamily.PACT_NERV_VQ,
        decoder=b"d" * 20,
        codebooks=b"c" * 10,
        selectors=b"s" * 2,
        manifest_extra={"num_pairs": 32},
    )

    report = build_spine_acquisition_report(
        projection_manifest_paths=[rnerv, pact_vq],
        hard_byte_ceilings=[10_000],
    )

    assert report["schema"] == "hprc_spine_acquisition_report.v1"
    assert report["score_claim"] is False
    rows = {row["family"]: row for row in report["rows"]}
    assert rows["rnerv"]["stack_role"]["position"] == "primary_learned_receiver_carrier"
    assert rows["pact_nerv_vq"]["stack_role"]["position"] == (
        "latent_codebook_base_or_residual_codec"
    )
    assert rows["rnerv"]["recommended_next_action"] == (
        "run_full_replay_then_exact_gate_before_residual_bytes"
    )
    assert rows["pact_nerv_vq"]["coverage"]["valid_for_base_comparison"] is False
    assert rows["pact_nerv_vq"]["recommended_next_action"] == (
        "scale_or_train_to_full_600_pair_coverage_before_base_byte_comparison"
    )
    best = report["best_under_each_ceiling"]["10000"]
    assert best is not None
    assert best["family"] == "rnerv"


def test_residual_admission_requires_measured_nonrate_gain(tmp_path: Path) -> None:
    manifest = _projection(
        tmp_path / "residual",
        family=HprcRepresentationFamily.PACT_NERV,
        decoder=b"d" * 30,
        residual=b"r" * 17,
    )

    report = build_spine_acquisition_report(
        projection_manifest_paths=[manifest],
        hard_byte_ceilings=[10_000],
    )

    row = report["rows"][0]
    residual = row["residual_section_admission"][0]
    assert residual["section"] == "residual_rc"
    assert residual["status"] == "measurement_required_before_admission"
    assert residual["required_measured_nonrate_improvement"] == -contest_rate_term(17)


def test_acquisition_cli_writes_report(tmp_path: Path) -> None:
    tool = _load_acquisition_tool()
    manifest = _projection(
        tmp_path / "finer",
        family=HprcRepresentationFamily.FINER_IMPLICIT,
        decoder=b"atom",
    )
    out = tmp_path / "queue.json"

    rc = tool.main(
        [
            "--projection-manifest",
            manifest.as_posix(),
            "--hard-byte-ceiling",
            "10000",
            "--output",
            out.as_posix(),
            "--repo-root",
            REPO.as_posix(),
        ]
    )

    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["rows"][0]["family"] == "finer_implicit"
    assert report["rows"][0]["stack_role"]["position"] == "implicit_residual_or_procedural_atom"


def _projection(
    out: Path,
    *,
    family: HprcRepresentationFamily,
    decoder: bytes,
    latents: bytes = b"",
    codebooks: bytes = b"",
    selectors: bytes = b"",
    residual: bytes = b"",
    manifest_extra: dict[str, object] | None = None,
) -> Path:
    spine = build_generic_neural_spine_packet(
        family=family,
        decoder_blob=decoder,
        latents_blob=latents,
        codebooks_blob=codebooks,
        selectors_blob=selectors,
        residual_blob=residual,
        manifest_extra=manifest_extra,
    )
    written = write_representation_spine_projection(output_dir=out, spine=spine)
    return Path(written["manifest_path"])


def _load_acquisition_tool():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "build_hprc_spine_acquisition_queue_test",
        REPO / "tools/build_hprc_spine_acquisition_queue.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
