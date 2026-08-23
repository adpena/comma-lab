from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments/ddm_jf1_joint_field_model_refit.py"


def _load():
    spec = importlib.util.spec_from_file_location("ddm_jf1_joint_field_model_refit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replace_hpac_preserves_every_non_hpac_section() -> None:
    jf1 = _load()
    header = jf1.jg2.RX1_HEADER.pack(b"RX1M", 1, 2, 3, 4, 3, 4, 5)
    member = header + b"old" + b"sema" + b"carry" + b"tail"
    replaced = jf1._replace_hpac(member, b"new-model")
    sections = jf1.jg2.split_member(replaced)
    assert sections["hpac"] == b"new-model"
    assert sections["semantic"] == b"sema"
    assert sections["carrier"] == b"carry"
    assert sections["tail"] == b"tail"
    assert jf1.jg2.RX1_HEADER.unpack(sections["header"])[4] == 4


def test_fixed_model_decomposition_pins_all_six_ld1_rungs() -> None:
    jf1 = _load()
    assert set(jf1.LD1_FIXED_STREAM_BYTES) == {
        "k002500",
        "k005000",
        "k010000",
        "k020000",
        "k040000",
        "k060000",
    }
    assert jf1.SHIPPED_STREAM_BYTES + jf1.SHIPPED_MODEL_BYTES == 127_292
    assert jf1.FIELD_SHA256["null"] == jf1.BASE_FIELD_SHA256


def test_candidate_runtime_pin_changes_only_the_exact_archive_literal(tmp_path: Path) -> None:
    jf1 = _load()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    source = f'ARCHIVE_SHA256 = "{jf1.DX2_ARCHIVE_SHA256}"\n'
    (runtime / "inflate.py").write_text(source, encoding="utf-8")
    digest = "a" * 64
    record = jf1._patch_runtime_archive_pin(runtime, digest)
    assert digest in (runtime / "inflate.py").read_text(encoding="utf-8")
    assert jf1.DX2_ARCHIVE_SHA256 not in (runtime / "inflate.py").read_text(encoding="utf-8")
    assert record["sha256"] == jf1.sha256_file(runtime / "inflate.py")
