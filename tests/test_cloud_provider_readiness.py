from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "cloud_provider_readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cloud_provider_readiness", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CloudProviderReadinessTests(unittest.TestCase):
    def test_find_cli_command_falls_back_to_uv_for_kaggle(self) -> None:
        mod = load_module()

        def fake_which(name: str) -> str | None:
            return "/usr/bin/uv" if name == "uv" else None

        self.assertEqual(
            mod.find_cli_command("kaggle", uv_package="kaggle", which=fake_which),
            ["uv", "run", "--with", "kaggle", "kaggle"],
        )

    def test_redact_removes_home_email_and_aws_account(self) -> None:
        mod = load_module()
        text = (
            f"{Path.home()}/secret owner adpena@example.com "
            '{"Account": "123456789012"}'
        )

        redacted = mod.redact(text)

        self.assertNotIn(str(Path.home()), redacted)
        self.assertIn("~/secret", redacted)
        self.assertIn("a***@example.com", redacted)
        self.assertIn('"Account": "1234********12"', redacted)

    def test_probe_kaggle_is_proxy_only_even_when_ready(self) -> None:
        mod = load_module()

        def fake_runner(command: list[str], _timeout: int):
            return mod.CommandResult(command=command, returncode=0, stdout='slug has status "COMPLETE"\n')

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_home = Path(tmpdir)
            (fake_home / ".kaggle").mkdir()
            (fake_home / ".kaggle" / "kaggle.json").write_text('{"username":"alice","key":"x"}')
            with mock.patch.object(mod.Path, "home", return_value=fake_home):
                with mock.patch.object(mod.shutil, "which", side_effect=lambda name: "/usr/bin/uv" if name == "uv" else None):
                    payload = mod.probe_kaggle(runner=fake_runner)

        self.assertEqual(payload.status, "ready_proxy")
        self.assertTrue(payload.proxy_only)
        self.assertFalse(payload.exact_cuda_evidence_allowed)
        self.assertIn("score_claim=false", " ".join(payload.next_actions))

    def test_probe_modal_cli_ready_still_requires_billing_and_cuda_probe(self) -> None:
        mod = load_module()

        def fake_runner(command: list[str], _timeout: int):
            return mod.CommandResult(command=command, returncode=0, stdout="modal client version: 1.4.2\n")

        with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/modal"):
            payload = mod.probe_modal(runner=fake_runner)

        self.assertEqual(payload.status, "ready_cli_check_runtime_probe_next")
        self.assertFalse(payload.exact_cuda_evidence_allowed)
        self.assertIn("modal_billing_not_checked", payload.blockers)
        self.assertIn("cuda_runtime_import_probe_not_run", payload.blockers)

    def test_probe_lightning_sdk_ready_still_requires_credit_and_claim(self) -> None:
        mod = load_module()

        def fake_runner(command: list[str], _timeout: int):
            return mod.CommandResult(command=command, returncode=0, stdout="2026.4.10\n")

        payload = mod.probe_lightning(runner=fake_runner)

        self.assertEqual(payload.status, "ready_sdk_check_credit_quota_next")
        self.assertFalse(payload.exact_cuda_evidence_allowed)
        self.assertIn("credits_or_quota_not_checked", payload.blockers)
        self.assertIn("no_dispatch_claim", payload.blockers)

    def test_probe_vastai_offer_ready_still_requires_claim_probe_and_heartbeat(self) -> None:
        mod = load_module()

        def fake_runner(command: list[str], _timeout: int):
            return mod.CommandResult(command=command, returncode=0, stdout='[{"id": 123}]\n')

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_home = Path(tmpdir)
            (fake_home / ".vast_api_key").write_text("secret\n")
            with mock.patch.object(mod.Path, "home", return_value=fake_home):
                with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/vastai"):
                    payload = mod.probe_vastai(runner=fake_runner)

        self.assertEqual(payload.status, "ready_offer_query_claim_heartbeat_next")
        self.assertFalse(payload.exact_cuda_evidence_allowed)
        self.assertIn("no_dispatch_claim", payload.blockers)
        self.assertIn("heartbeat_not_checked", payload.blockers)

    def test_collect_readiness_uses_provider_contract_order(self) -> None:
        mod = load_module()

        def fake_provider(name: str):
            return mod.ProviderReadiness(
                provider=name,
                status="stub",
                score_lowering_role="stub",
                exact_cuda_evidence_allowed=False,
                proxy_only=False,
            )

        with mock.patch.object(mod, "probe_modal", return_value=fake_provider("modal")), \
             mock.patch.object(mod, "probe_kaggle", return_value=fake_provider("kaggle")), \
             mock.patch.object(mod, "probe_lightning", return_value=fake_provider("lightning")), \
             mock.patch.object(mod, "probe_vastai", return_value=fake_provider("vastai")), \
             mock.patch.object(mod, "probe_aws", return_value=fake_provider("aws")), \
             mock.patch.object(mod, "probe_azure", return_value=fake_provider("azure")), \
             mock.patch.object(mod, "probe_gcp", return_value=fake_provider("gcp")):
            payload = mod.collect_readiness(kaggle_kernel="unit/kernel", timeout_s=1)

        providers = [row["provider"] for row in payload["providers"]]
        self.assertEqual(providers, list(mod.provider_contracts()))

    def test_probe_aws_classifies_expired_session(self) -> None:
        mod = load_module()

        def fake_runner(command: list[str], _timeout: int):
            return mod.CommandResult(
                command=command,
                returncode=255,
                stderr="Your session has expired. Please reauthenticate using 'aws login'.",
            )

        with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/aws"):
            payload = mod.probe_aws(runner=fake_runner)

        self.assertEqual(payload.status, "blocked_auth")
        self.assertEqual(payload.blockers, ["aws_session_expired"])
        self.assertFalse(payload.exact_cuda_evidence_allowed)

    def test_probe_gcp_blocks_when_billing_is_disabled(self) -> None:
        mod = load_module()
        seen: list[list[str]] = []

        def fake_runner(command: list[str], _timeout: int):
            seen.append(command)
            if command[:3] == ["gcloud", "auth", "list"]:
                return mod.CommandResult(command=command, returncode=0, stdout="a@example.com\n")
            if command[:3] == ["gcloud", "config", "get-value"]:
                return mod.CommandResult(command=command, returncode=0, stdout="demo-project\n")
            if command[:4] == ["gcloud", "billing", "projects", "describe"]:
                return mod.CommandResult(command=command, returncode=0, stdout='{"billingEnabled": false}\n')
            raise AssertionError(command)

        with mock.patch.object(mod.shutil, "which", return_value="/usr/bin/gcloud"):
            payload = mod.probe_gcp(runner=fake_runner)

        self.assertEqual(payload.status, "blocked_billing")
        self.assertIn("gcp_billing_not_enabled_or_not_readable", payload.blockers)
        self.assertFalse(payload.exact_cuda_evidence_allowed)
        self.assertGreaterEqual(len(seen), 3)


if __name__ == "__main__":
    unittest.main()
