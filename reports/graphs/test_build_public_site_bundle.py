from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reports.graphs.build_public_site_bundle import (
    build_public_site_bundle,
    sanitize_text,
)


class PublicSiteBundleTests(unittest.TestCase):
    def test_sanitize_text_redacts_private_surfaces(self) -> None:
        local_path = "/" + "Users/adpena/Projects/pact/reports/raw/x"
        tmp_path = "/" + "private/tmp/pact-mine/x"
        vast_endpoint = "ssh4." + "vast.ai:25850"
        lightning_url = "https://lightning.ai/" + "adpena/comma-lab/studios/foo/app?x=1"
        modal_app = "ap-" + "AbCdEf1234567890"
        modal_call = "fc-" + "ABCDEF1234567890ABCDEF"
        secret_assignment = "CLOUDFLARE_API_TOKEN" + "=secret-token"
        text = (
            f"path={local_path} "
            f"tmp={tmp_path} "
            f"ssh={vast_endpoint} "
            f"url={lightning_url} "
            f"modal={modal_app} "
            f"call={modal_call} "
            f"{secret_assignment}"
        )

        sanitized, records = sanitize_text(text)

        self.assertNotIn(local_path, sanitized)
        self.assertNotIn(tmp_path, sanitized)
        self.assertNotIn(vast_endpoint, sanitized)
        self.assertNotIn(lightning_url, sanitized)
        self.assertNotIn(modal_app, sanitized)
        self.assertNotIn(modal_call, sanitized)
        self.assertIn("${LOCAL_PATH_REDACTED}", sanitized)
        self.assertIn("${VAST_SSH_REDACTED}", sanitized)
        self.assertIn("${LIGHTNING_PRIVATE_URL_REDACTED}", sanitized)
        self.assertIn("${MODAL_ID_REDACTED}", sanitized)
        self.assertIn("CLOUDFLARE_API_TOKEN" + "=${SECRET_REDACTED}", sanitized)
        self.assertGreaterEqual(sum(count for _, count in records), 6)

    def test_sanitize_text_rewrites_private_comma_lab_repo_url(self) -> None:
        private_url = "https://github.com/adpena/" + "comma-lab/tree/main/docs"

        sanitized, records = sanitize_text(f"repo={private_url}\n")

        self.assertNotIn(private_url, sanitized)
        self.assertIn("https://github.com/adpena/tac", sanitized)
        self.assertEqual(records[0][0], "private_comma_lab_github_url")

    def test_build_public_site_bundle_redacts_and_preserves_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "site"
            output = root / "public_site"
            source.mkdir()
            (source / "index.html").write_text("<h1>Apogee</h1>\n", encoding="utf-8")
            private_path = "/" + "Users/adpena/Projects/pact/private"
            (source / "data.json").write_text(
                json.dumps({"path": private_path}) + "\n",
                encoding="utf-8",
            )
            (source / "clip.bin").write_bytes(b"\x00\x01\x02")

            manifest = build_public_site_bundle(
                source,
                output,
                max_asset_bytes=1024,
                strict_hygiene=False,
            )

            self.assertTrue((output / "index.html").is_file())
            self.assertEqual((output / "clip.bin").read_bytes(), b"\x00\x01\x02")
            data = json.loads((output / "data.json").read_text(encoding="utf-8"))
            self.assertEqual(data["path"], "${LOCAL_PATH_REDACTED}")
            self.assertEqual(manifest["redaction_count"], 1)
            self.assertEqual(manifest["public_link_violation_count"], 0)
            self.assertTrue((output / "public_site_manifest.json").is_file())

    def test_build_public_site_bundle_scans_final_manifest_without_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "site"
            output = root / "public_site"
            source.mkdir()
            (source / "index.html").write_text("<h1>Apogee</h1>\n", encoding="utf-8")

            manifest = build_public_site_bundle(
                source,
                output,
                max_asset_bytes=1024,
                strict_hygiene=True,
            )
            manifest_text = (output / "public_site_manifest.json").read_text(encoding="utf-8")

            self.assertNotIn(str(root), manifest_text)
            self.assertEqual(manifest["source"], "${EXTERNAL_PATH}/site")
            self.assertEqual(manifest["output"], "${EXTERNAL_PATH}/public_site")
            self.assertEqual(manifest["public_link_count"], 0)

    def test_build_public_site_bundle_rewrites_private_comma_lab_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "site"
            output = root / "public_site"
            source.mkdir()
            private_url = "https://github.com/adpena/" + "comma-lab"
            (source / "index.html").write_text(
                f"<a href='{private_url}'>private</a>\n",
                encoding="utf-8",
            )

            manifest = build_public_site_bundle(
                source,
                output,
                max_asset_bytes=1024,
                strict_hygiene=True,
            )

            self.assertEqual(manifest["public_link_violation_count"], 0)
            self.assertIn(
                "https://github.com/adpena/tac",
                (output / "index.html").read_text(encoding="utf-8"),
            )

    def test_build_public_site_bundle_rejects_large_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "site"
            output = root / "public_site"
            source.mkdir()
            (source / "large.mp4").write_bytes(b"x" * 8)

            with self.assertRaises(RuntimeError):
                build_public_site_bundle(
                    source,
                    output,
                    max_asset_bytes=4,
                    oversized_policy="fail",
                    strict_hygiene=False,
                )

    def test_build_public_site_bundle_omits_large_assets_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "site"
            output = root / "public_site"
            source.mkdir()
            (source / "large.gif").write_bytes(b"x" * 8)

            manifest = build_public_site_bundle(
                source,
                output,
                max_asset_bytes=4,
                strict_hygiene=False,
            )

            self.assertFalse((output / "large.gif").exists())
            self.assertEqual(
                manifest["omitted_oversized_assets"],
                [{"path": "large.gif", "bytes": 8}],
            )


if __name__ == "__main__":
    unittest.main()
