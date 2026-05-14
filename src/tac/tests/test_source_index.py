# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import textwrap
from pathlib import Path

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


def test_source_index_files_by_pattern_matches_individual_file_lists(tmp_path):
    _write(tmp_path / "scripts/b.sh", "#!/usr/bin/env bash\n")
    _write(tmp_path / "scripts/a.py", "A = 1\n")
    _write(tmp_path / "src/tac/c.py", "C = 1\n")
    _write(tmp_path / "src/tac/d.sh", "#!/usr/bin/env bash\n")
    _write(tmp_path / "src/tac/ignored.md", "ignored\n")
    _write(tmp_path / "experiments/results/generated.py", "GENERATED = 1\n")
    _write(tmp_path / "src/tac/__pycache__/cached.py", "CACHED = 1\n")

    index = SourceIndex(tmp_path)

    grouped = index.files_by_pattern(
        ["scripts", "src/tac", "experiments"],
        patterns=("*.py", "*.sh"),
    )

    assert grouped["*.py"] == index.files(
        ["scripts", "src/tac", "experiments"],
        pattern="*.py",
    )
    assert grouped["*.sh"] == index.files(
        ["scripts", "src/tac", "experiments"],
        pattern="*.sh",
    )
    assert [index.repo_relative(path) for path in grouped["*.py"]] == [
        "scripts/a.py",
        "src/tac/c.py",
    ]
    assert [index.repo_relative(path) for path in grouped["*.sh"]] == [
        "scripts/b.sh",
        "src/tac/d.sh",
    ]
    stats = index.stats()
    assert stats["files_by_pattern_misses"] == 1
    assert stats["file_list_hits"] == 2
    assert stats["file_list_misses"] == 0


def test_source_index_files_by_pattern_is_deterministic_and_cached(tmp_path):
    _write(tmp_path / "scripts/z.sh", "#!/usr/bin/env bash\n")
    _write(tmp_path / "scripts/a.sh", "#!/usr/bin/env bash\n")
    _write(tmp_path / "scripts/m.py", "M = 1\n")
    _write(tmp_path / "scripts/a.py", "A = 1\n")
    index = SourceIndex(tmp_path)

    first = index.files_by_pattern(["scripts"], patterns=("*.sh", "*.py"))
    second = index.files_by_pattern(["scripts"], patterns=("*.sh", "*.py"))

    assert first == second
    assert [index.repo_relative(path) for path in first["*.sh"]] == [
        "scripts/a.sh",
        "scripts/z.sh",
    ]
    assert [index.repo_relative(path) for path in first["*.py"]] == [
        "scripts/a.py",
        "scripts/m.py",
    ]
    stats = index.stats()
    assert stats["files_by_pattern_misses"] == 1
    assert stats["files_by_pattern_hits"] == 1
    assert stats["files_by_pattern_cache_entries"] == 1
    assert stats["file_list_misses"] == 0


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


def test_source_index_skips_persistent_text_facts_on_github_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    module = tmp_path / "src/tac/module.py"
    _write(module, "VALUE = 'kl_div batchmean'\n")

    with source_index_context(tmp_path) as index:
        facts = index.text_facts(module)
        assert facts.contains_all(("kl_div", "batchmean"))
        assert index.stats()["persistent_text_facts_enabled"] == 0

    assert not (tmp_path / ".omx/cache/source_text_facts.json").exists()


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


def test_source_index_canonicalizes_absolute_dirs_for_file_cache(tmp_path):
    _write(tmp_path / "src/tac/a.py", "VALUE = 'kl_div batchmean'\n")
    index = SourceIndex(tmp_path)

    rel_files = index.files(["src/tac"], pattern="*.py")
    abs_files = index.files([tmp_path / "src/tac"], pattern="*.py")

    assert rel_files == abs_files
    stats = index.stats()
    assert stats["file_list_misses"] == 1
    assert stats["file_list_hits"] == 1


def test_source_index_canonicalizes_absolute_dirs_for_fact_and_substring_caches(tmp_path):
    _write(tmp_path / "src/tac/a.py", "VALUE = 'kl_div batchmean'\n")
    _write(tmp_path / "src/tac/b.py", "VALUE = 'kl_div only'\n")
    index = SourceIndex(tmp_path)

    rel_facts = index.facts_for_files(["src/tac"], pattern="*.py")
    abs_facts = index.facts_for_files([tmp_path / "src/tac"], pattern="*.py")
    rel_matches = index.files_containing_substrings(
        ["src/tac"],
        pattern="*.py",
        substrings=("kl_div", "batchmean"),
        require_all=True,
    )
    abs_matches = index.files_containing_substrings(
        [tmp_path / "src/tac"],
        pattern="*.py",
        substrings=("kl_div", "batchmean"),
        require_all=True,
    )

    assert [row.path for row in rel_facts] == [row.path for row in abs_facts]
    assert rel_matches == abs_matches
    stats = index.stats()
    assert stats["facts_group_misses"] == 1
    assert stats["facts_group_hits"] >= 1
    assert stats["substring_index_misses"] == 0
    assert stats["substring_index_hits"] >= 2


def test_source_index_substring_index_filters_candidates(tmp_path):
    _write(tmp_path / "src/tac/a.py", "VALUE = 'kl_div batchmean'\n")
    _write(tmp_path / "src/tac/b.py", "VALUE = 'kl_div only'\n")
    _write(tmp_path / "scripts/c.py", "VALUE = 'batchmean only'\n")
    _write(
        tmp_path / "experiments/roundtrip.py",
        """
        import torch.nn.functional as F

        def eval_roundtrip(x):
            return F.interpolate(x, size=(384, 512)).round()
        """,
    )
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
    roundtrip_matched = index.files_containing_substrings(
        ["experiments", "scripts", "src/tac"],
        pattern="*.py",
        substrings=(".round(", "F.interpolate"),
        require_all=True,
    )
    assert [index.repo_relative(path) for path in roundtrip_matched] == [
        "experiments/roundtrip.py"
    ]
    assert matched_again == matched
    stats = index.stats()
    assert stats["substring_index_entries"] >= 2
    assert stats["substring_index_hits"] >= 2
    assert stats["substring_index_misses"] == 0


def test_source_index_dispatch_attempted_is_indexed_for_audit_scans(tmp_path):
    _write(tmp_path / "tools/a.py", "score_claim = False\n")
    _write(tmp_path / "tools/b.py", "dispatch_attempted = False\n")
    _write(tmp_path / "tools/c.py", "other = True\n")
    index = SourceIndex(tmp_path)

    matched = index.files_containing_substrings(
        ["tools"],
        pattern="*.py",
        substrings=("score_claim", "dispatch_attempted"),
        require_all=False,
    )

    assert [index.repo_relative(path) for path in matched] == [
        "tools/a.py",
        "tools/b.py",
    ]
    stats = index.stats()
    assert stats["substring_index_misses"] == 0
    assert stats["text_hits"] == 0


def test_source_index_tag_grade_bypass_patterns_are_indexed(tmp_path):
    _write(tmp_path / "src/tac/a.py", "if evidence_grade in {'contest-cuda'}:\n    pass\n")
    _write(tmp_path / "src/tac/b.py", "if tag.startswith(\"[contest-CUDA\"):\n    pass\n")
    _write(tmp_path / "src/tac/c.py", "if other:\n    pass\n")
    index = SourceIndex(tmp_path)

    matched = index.files_containing_substrings(
        ["src/tac"],
        pattern="*.py",
        substrings=("evidence_grade in {", 'tag.startswith("[contest-CUDA")'),
        require_all=False,
    )

    assert [index.repo_relative(path) for path in matched] == [
        "src/tac/a.py",
        "src/tac/b.py",
    ]
    stats = index.stats()
    assert stats["substring_index_misses"] == 0
    assert stats["text_hits"] == 0


def test_source_index_score_aware_scorer_contract_patterns_are_indexed(tmp_path):
    _write(
        tmp_path / "src/tac/substrates/a/score_aware_loss.py",
        "self.seg_scorer(x)\nscore_pair_components(seg_scorer=s)\n",
    )
    _write(tmp_path / "src/tac/substrates/b/score_aware_loss.py", "VALUE = 1\n")
    index = SourceIndex(tmp_path)

    matched = index.files_containing_substrings(
        ["src/tac/substrates"],
        pattern="score_aware_loss.py",
        substrings=("self.seg_scorer(", "score_pair_components("),
        require_all=False,
    )

    assert [index.repo_relative(path) for path in matched] == [
        "src/tac/substrates/a/score_aware_loss.py",
    ]
    stats = index.stats()
    assert stats["substring_index_misses"] == 0
    assert stats["text_hits"] == 0


def test_source_index_comment_contract_prefilter_does_not_self_trigger():
    import tac.source_index as source_index

    needle = "overrides this stub"

    assert needle in source_index._DEFAULT_TEXT_FACT_NEEDLES
    assert needle not in Path(source_index.__file__).read_text(encoding="utf-8")


def test_source_index_casefold_prefilter_needles_stay_tiny():
    import tac.source_index as source_index

    casefold_needles = source_index._DEFAULT_CASEFOLD_TEXT_FACT_NEEDLES

    assert len(casefold_needles) <= 16
    assert "READY-TO-LAUNCH" in casefold_needles
    assert "mps" not in casefold_needles
    assert "score_claim" not in casefold_needles


def test_source_index_casefold_query_uses_narrow_prefilter(tmp_path):
    _write(tmp_path / "docs/a.md", "ready-to-launch\n")
    _write(tmp_path / "docs/b.md", "ordinary launch note\n")
    index = SourceIndex(tmp_path)

    matched = index.files_containing_casefold_substrings(
        ["docs"],
        pattern="*.md",
        substrings=("READY-TO-LAUNCH",),
        require_all=True,
    )

    assert [index.repo_relative(path) for path in matched] == ["docs/a.md"]
    stats = index.stats()
    assert stats["text_hits"] == 0


def test_source_index_unknown_substring_queries_still_scan_text(tmp_path):
    _write(tmp_path / "src/tac/a.py", "VALUE = 'custom_runtime_contract'\n")
    _write(tmp_path / "src/tac/b.py", "VALUE = 'other'\n")
    index = SourceIndex(tmp_path)

    matched = index.files_containing_substrings(
        ["src/tac"],
        pattern="*.py",
        substrings=("custom_runtime_contract",),
        require_all=True,
    )

    assert [index.repo_relative(path) for path in matched] == ["src/tac/a.py"]
    stats = index.stats()
    assert stats["text_hits"] >= 1
    assert stats["substring_index_misses"] >= 1


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


def test_source_index_persistent_text_facts_invalidate_on_ctime_even_same_mtime_size(
    tmp_path,
):
    module = tmp_path / "src/tac/module.py"
    _write(module, "VALUE = 'mps'\n")
    original_stat = module.stat()
    with source_index_context(tmp_path) as index:
        assert index.text_facts(module).contains("mps")

    module.write_text("VALUE = 'cpu'\n", encoding="utf-8")
    # Simulate a same-size same-mtime rewrite; ctime/inode/device still prevent
    # stale persistent facts from being reused.
    os.utime(module, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    changed_index = SourceIndex(tmp_path)
    changed_facts = changed_index.text_facts(module)

    assert not changed_facts.contains("mps")
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


def test_source_index_fact_worker_budget_is_bounded(monkeypatch):
    import tac.source_index as source_index

    monkeypatch.delenv("PACT_SOURCE_INDEX_FACT_WORKERS", raising=False)
    assert source_index._source_index_fact_workers() == 8

    monkeypatch.setenv("PACT_SOURCE_INDEX_FACT_WORKERS", "2")
    assert source_index._source_index_fact_workers() == 2

    monkeypatch.setenv("PACT_SOURCE_INDEX_FACT_WORKERS", "999")
    assert source_index._source_index_fact_workers() == 32

    monkeypatch.setenv("PACT_SOURCE_INDEX_FACT_WORKERS", "not-an-int")
    assert source_index._source_index_fact_workers() == 8


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
    assert stats["facts_group_hits"] >= 3
    assert stats["text_facts_cache_entries"] == 3
    assert stats["text_facts_hits"] == 0
    assert stats["substring_index_entries"] >= 4
    assert stats["ast_cache_entries"] == 3
