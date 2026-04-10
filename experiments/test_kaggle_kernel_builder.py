from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import kaggle_kernel_builder as mod


class KaggleKernelBuilderTests(unittest.TestCase):
    def test_build_metadata_defaults_to_private_gpu_script(self) -> None:
        metadata = mod.build_kernel_metadata(
            username="alice",
            slug="comma-lab-test",
            title="comma-lab test",
            code_file="run_kernel.py",
        )

        self.assertEqual(metadata["id"], "alice/comma-lab-test")
        self.assertEqual(metadata["code_file"], "run_kernel.py")
        self.assertEqual(metadata["kernel_type"], "script")
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertTrue(metadata["enable_internet"])
        self.assertEqual(metadata["dataset_sources"], [])

    def test_write_bundle_copies_files_and_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            (source / "pkg").mkdir(parents=True)
            (source / "pkg" / "helper.py").write_text("VALUE = 1\n")

            bundle = root / "bundle"
            spec = mod.KaggleKernelSpec(
                slug="comma-lab-test",
                title="comma-lab test",
                module_name="pkg.helper",
                args=["--epochs", "10"],
                include_paths=(source / "pkg" / "helper.py",),
            )

            mod.write_bundle(bundle_dir=bundle, username="alice", spec=spec, repo_root=source)

            metadata = json.loads((bundle / "kernel-metadata.json").read_text())
            launcher = (bundle / "run_kernel.py").read_text()
            copied = (bundle / "pkg" / "helper.py").read_text()

        self.assertEqual(metadata["id"], "alice/comma-lab-test")
        self.assertIn("import pkg.helper as target_module", launcher)
        self.assertIn("--epochs", launcher)
        self.assertIn("ensure_runtime_dependencies()", launcher)
        self.assertIn("ensure_writable_root()", launcher)
        self.assertIn("/kaggle/working", launcher)
        self.assertIn("git-lfs", launcher)
        self.assertEqual(copied, "VALUE = 1\n")

    def test_write_bundle_supports_direct_code_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "script.py"
            source.write_text("print('hi')\n")
            asset = root / "reports" / "raw" / "artifact.zip"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"zip")

            bundle = root / "bundle"
            spec = mod.KaggleKernelSpec(
                slug="comma-lab-direct",
                title="comma-lab direct",
                code_source=source,
                code_file="script.py",
                include_paths=(source, asset),
            )

            mod.write_bundle(bundle_dir=bundle, username="alice", spec=spec, repo_root=root)

            metadata = json.loads((bundle / "kernel-metadata.json").read_text())
            copied_script = (bundle / "script.py").read_text()
            copied_asset = (bundle / "artifact.zip").read_bytes()

        self.assertEqual(metadata["code_file"], "script.py")
        self.assertEqual(copied_script, "print('hi')\n")
        self.assertEqual(copied_asset, b"zip")


if __name__ == "__main__":
    unittest.main()
