from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessCodecsTests(unittest.TestCase):
    def test_lzma_roundtrip_restores_numpy_tokens(self) -> None:
        from tac.lossless.codecs import lzma_roundtrip_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tokens.npy"
            compressed = root / "tokens.lzma"
            restored = root / "restored.npy"
            np.save(source, np.array([[1, 2], [3, 4]], dtype=np.int16))

            result = lzma_roundtrip_file(source_path=source, compressed_path=compressed, restored_path=restored)

            self.assertTrue(compressed.exists())
            self.assertTrue(restored.exists())
            self.assertEqual(result["archive_bytes"], compressed.stat().st_size)
            self.assertTrue(np.array_equal(np.load(restored), np.load(source)))


if __name__ == "__main__":
    unittest.main()
