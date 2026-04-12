from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


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

    def test_load_local_commavq_record_sample_reads_json_and_tokens_from_tarballs(self) -> None:
        from tac.lossless.data import load_local_commavq_record_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "data-0000.tar.gz"
            token_buf = io.BytesIO()
            np.save(token_buf, np.arange(2 * 8 * 16, dtype=np.int16).reshape(2, 8, 16), allow_pickle=False)
            json_bytes = json.dumps({"file_name": "clip_a.npy"}).encode("utf-8")

            with tarfile.open(shard_path, "w:gz") as tar:
                token_info = tarfile.TarInfo("clip_a.token.npy")
                token_info.size = len(token_buf.getvalue())
                tar.addfile(token_info, io.BytesIO(token_buf.getvalue()))

                json_info = tarfile.TarInfo("clip_a.json")
                json_info.size = len(json_bytes)
                tar.addfile(json_info, io.BytesIO(json_bytes))

            examples = load_local_commavq_record_sample([str(shard_path)], max_records=1)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["json"]["file_name"], "clip_a.npy")
        self.assertEqual(examples[0]["token.npy"].shape, (2, 8, 16))


if __name__ == "__main__":
    unittest.main()
