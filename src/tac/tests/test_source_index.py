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


def test_source_index_builds_shared_text_facts_once(tmp_path):
    module = tmp_path / "src/tac/module.py"
    _write(
        module,
        """
        import torch

        DEVICE = "cuda" if torch.cuda.is_available() else "mps"
        """,
    )
    index = SourceIndex(tmp_path)

    facts = index.text_facts(module)

    assert facts.rel_path == "src/tac/module.py"
    assert facts.line_count == 3
    assert facts.contains_all(("cuda.is_available", "mps"))
    assert facts.contains_any(("torch", "--no-eval-roundtrip"))
    assert index.text_facts(module) is facts
    stats = index.stats()
    assert stats["text_facts_misses"] == 1
    assert stats["text_facts_hits"] == 1
    assert stats["text_misses"] == 1


def test_source_index_parallel_facts_for_files_reuses_file_inventory(tmp_path):
    for idx in range(48):
        _write(tmp_path / f"src/tac/mod_{idx:02d}.py", f"VALUE_{idx} = 'kl_div batchmean'\n")
    index = SourceIndex(tmp_path)

    first = index.facts_for_files(["src/tac"], pattern="*.py")
    second = index.facts_for_files(["src/tac"], pattern="*.py")

    assert len(first) == 48
    assert all(facts.contains_all(("kl_div", "batchmean")) for facts in first)
    assert [facts.rel_path for facts in first] == [facts.rel_path for facts in second]
    stats = index.stats()
    assert stats["file_list_misses"] == 1
    assert stats["facts_group_misses"] == 1
    assert stats["facts_group_hits"] == 1
    assert stats["text_facts_cache_entries"] == 48
    assert stats["text_facts_hits"] == 0


def test_source_index_substring_index_filters_candidates(tmp_path):
    _write(tmp_path / "src/tac/a.py", "VALUE = 'kl_div batchmean'\n")
    _write(tmp_path / "src/tac/b.py", "VALUE = 'kl_div only'\n")
    _write(tmp_path / "scripts/c.py", "VALUE = 'batchmean only'\n")
    index = SourceIndex(tmp_path)

    matched = index.files_containing_substrings(
        ["scripts", "src/tac"],
        pattern="*.py",
        substrings=("kl_div", "batchmean"),
        require_all=True,
    )
    matched_again = index.files_containing_substrings(
        ["scripts", "src/tac"],
        pattern="*.py",
        substrings=("kl_div", "batchmean"),
        require_all=True,
    )

    assert [index.repo_relative(path) for path in matched] == ["src/tac/a.py"]
    assert matched_again == matched
    stats = index.stats()
    assert stats["substring_index_entries"] >= 2
    assert stats["substring_index_hits"] >= 2
    assert stats["substring_index_misses"] == 0


def test_source_index_persistent_text_facts_invalidate_on_mtime_size(tmp_path):
    module = tmp_path / "src/tac/module.py"
    _write(module, "VALUE = 'mps'\n")
    with source_index_context(tmp_path) as index:
        assert index.text_facts(module).contains("mps")

    warm_index = SourceIndex(tmp_path)
    assert warm_index.text_facts(module).contains("mps")
    warm_stats = warm_index.stats()
    assert warm_stats["text_facts_persistent_hits"] == 1
    assert warm_stats["text_misses"] == 0

    module.write_text("VALUE = 'kl_div batchmean'\n", encoding="utf-8")
    changed_index = SourceIndex(tmp_path)
    changed_facts = changed_index.text_facts(module)

    assert not changed_facts.contains("mps")
    assert changed_facts.contains_all(("kl_div", "batchmean"))
    changed_stats = changed_index.stats()
    assert changed_stats["text_facts_persistent_hits"] == 0
    assert changed_stats["text_misses"] == 1


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

    assert stats["text_hits"] >= 1
    assert stats["ast_cache_entries"] >= 2


def test_migrated_preflight_checks_use_shared_text_facts(tmp_path):
    _write(
        tmp_path / "src/tac/device.py",
        """
        import torch

        DEVICE = "cuda" if torch.cuda.is_available() else "mps"
        """,
    )
    _write(
        tmp_path / "experiments/train.py",
        """
        def train(*, eval_roundtrip=False):
            return eval_roundtrip
        """,
    )
    _write(
        tmp_path / "scripts/launch.py",
        """
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--no-eval-roundtrip", action="store_true")
        """,
    )

    with source_index_context(tmp_path) as index:
        assert preflight.check_no_mps_fallback_default(
            repo_root=tmp_path,
            strict=False,
            verbose=False,
        )
        assert preflight.check_no_eval_roundtrip_false(
            repo_root=tmp_path,
            strict=False,
            verbose=False,
        )
        assert preflight.check_no_disable_eval_roundtrip_flag(
            repo_root=tmp_path,
            strict=False,
            verbose=False,
        )
        stats = index.stats()

    assert stats["file_list_misses"] == 1
    assert stats["facts_group_misses"] == 1
    assert stats["facts_group_hits"] >= 5
    assert stats["text_facts_cache_entries"] == 3
    assert stats["text_facts_hits"] == 0
    assert stats["substring_index_entries"] >= 4
    assert stats["ast_cache_entries"] == 3
