from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessDataTests(unittest.TestCase):
    def test_resolve_local_commavq_cached_data_files_prefers_cached_snapshot(self) -> None:
        from tac.lossless.data import resolve_local_commavq_cached_data_files

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "hub" / "datasets--commaai--commavq" / "snapshots" / "snapshot-a"
            cache_root.mkdir(parents=True)
            shard0 = cache_root / "data-0000.tar.gz"
            shard1 = cache_root / "data-0001.tar.gz"
            shard0.write_bytes(b"a")
            shard1.write_bytes(b"b")

            with mock.patch.dict(os.environ, {"HF_HOME": tmpdir}, clear=False):
                resolved = resolve_local_commavq_cached_data_files(["data-0000.tar.gz", "data-0001.tar.gz"])

        self.assertEqual(resolved, [str(shard0), str(shard1)])

    def test_load_commavq_dataset_passes_cached_local_shards_to_loader(self) -> None:
        from tac.lossless.data import load_commavq_dataset

        calls: list[dict[str, object]] = []

        def fake_loader(dataset_name: str, **kwargs):
            calls.append({"dataset_name": dataset_name, **kwargs})
            return {"train": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "hub" / "datasets--commaai--commavq" / "snapshots" / "snapshot-a"
            cache_root.mkdir(parents=True)
            shard0 = cache_root / "data-0000.tar.gz"
            shard1 = cache_root / "data-0001.tar.gz"
            shard0.write_bytes(b"a")
            shard1.write_bytes(b"b")

            with mock.patch.dict(os.environ, {"HF_HOME": tmpdir}, clear=False):
                load_commavq_dataset(
                    split=[0, 1],
                    dataset_loader=fake_loader,
                    streaming=True,
                )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["dataset_name"], "commaai/commavq")
        self.assertEqual(calls[0]["streaming"], True)
        self.assertEqual(calls[0]["data_files"]["train"], [str(shard0), str(shard1)])


if __name__ == "__main__":
    unittest.main()
