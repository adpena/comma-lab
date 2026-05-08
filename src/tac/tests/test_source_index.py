from __future__ import annotations

import textwrap

from tac import preflight
from tac.source_index import SourceIndex, get_current_source_index, source_index_context


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def test_source_index_discovers_sorted_source_files_and_skips_mirrors(tmp_path):
    _write(tmp_path / "src/tac/b.py", "B = 1\n")
    _write(tmp_path / "src/tac/a.py", "A = 1\n")
    _write(tmp_path / "experiments/comma_lab_public_export/mirror.py", "MIRROR = 1\n")
    _write(tmp_path / "experiments/__pycache__/cached.py", "CACHED = 1\n")

    index = SourceIndex(tmp_path)

    rels = [index.repo_relative(path) for path in index.files(["experiments", "src/tac"], pattern="*.py")]

    assert rels == ["src/tac/a.py", "src/tac/b.py"]
    assert index.stats()["file_list_misses"] == 1
    assert index.files(["experiments", "src/tac"], pattern="*.py") == index.files(
        ["experiments", "src/tac"], pattern="*.py"
    )
    assert index.stats()["file_list_hits"] == 2


def test_source_index_caches_text_and_ast_within_one_scan(tmp_path):
    module = tmp_path / "src/tac/module.py"
    _write(module, "VALUE = 'first.bin'\n")
    index = SourceIndex(tmp_path)

    first_text = index.read_text(module)
    first_ast = index.python_ast(module)
    module.write_text("VALUE = 'second.bin'\n", encoding="utf-8")

    assert index.read_text(module) == first_text
    assert index.python_ast(module) is first_ast
    assert SourceIndex(tmp_path).read_text(module) == "VALUE = 'second.bin'\n"
    assert index.stats()["text_hits"] >= 1
    assert index.stats()["ast_hits"] == 1


def test_source_index_context_is_root_scoped(tmp_path):
    other = tmp_path / "other"
    other.mkdir()

    with source_index_context(tmp_path) as index:
        assert get_current_source_index(tmp_path) is index
        assert get_current_source_index(other) is None

    assert get_current_source_index(tmp_path) is None


def test_preflight_scanners_share_source_index_ast_cache(tmp_path):
    _write(
        tmp_path / "scripts/consumer.py",
        """
        from pathlib import Path

        TARGET = Path("renderer_fp4.bin")
        TARGET.exists()
        """,
    )
    _write(
        tmp_path / "experiments/producer.py",
        """
        import torch

        OUTPUT = "renderer_fp4.bin"

        def load_renderer(path):
            magic = Path(path).read_bytes()[:4]
            if magic == b"FP4A":
                return None
            return torch.load(path, weights_only=True)
        """,
    )
    _write(
        tmp_path / "experiments/comma_lab_public_export/mirror.py",
        """
        PHANTOM = "phantom_only.bin"
        """,
    )

    with source_index_context(tmp_path) as index:
        assert preflight.preflight_loader_format_safety(
            repo_root=tmp_path,
            scan_dirs=["experiments"],
            strict=True,
            verbose=False,
        ) == []
        assert preflight.preflight_filename_contract(
            repo_root=tmp_path,
            consumer_files=["scripts/consumer.py"],
            producer_dirs=["experiments"],
            strict=True,
            verbose=False,
        ) == []
        stats = index.stats()

    assert stats["file_list_hits"] >= 1
    assert stats["ast_hits"] >= 1
    assert stats["ast_cache_entries"] >= 2
