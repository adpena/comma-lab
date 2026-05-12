"""Tests for the Kaggle T1 Ballé parallel-sweep harness.

Covers per-harness invariants:

1. Kernel script's P100 FATAL exit on sm_60 (mocked torch).
2. Kernel script's no-CUDA FATAL exit (mocked torch).
3. AST extraction of the Tier-1 manifest from the trainer source.
4. ``build_trainer_argv`` threads every manifest flag + no hardcoded list.
5. Operator wrapper rejects slugs >= 25 chars (Kaggle "Notebook not found" trap).
6. Operator wrapper rejects non-alphanumeric variant slugs.
7. Operator wrapper enforces 2-session cap on Kaggle free tier (smoke).
8. Harvester ``classify_terminal_status`` maps rc=2 to ``failed_kaggle_p100_assignment``.
9. Harvester ``classify_terminal_status`` maps rc=0 + complete to ``completed_kaggle``.
10. Harvester ``append_cost_band_anchor_from_summary`` invokes the anchor tool.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_sibling_module(rel_path: str, mod_name: str):
    """Load a top-level script as an importable module for testing."""
    full = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full)
    assert spec is not None and spec.loader is not None, f"cannot load {full}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestP100FatalExit(unittest.TestCase):
    """The kernel script must FATAL-exit on sm_60 BEFORE any heavy import."""

    def setUp(self):
        self.kernel = _load_sibling_module(
            "experiments/kaggle_t1_balle_sweep.py", "_test_kaggle_kernel"
        )

    def test_p100_sm60_exits_with_rc_2(self):
        fake_props = mock.Mock(
            name="Tesla P100-PCIE-16GB",
            major=6, minor=0, total_memory=16 * 1024**3,
        )
        # Mock.name='X' makes .name return X
        fake_props.name = "Tesla P100-PCIE-16GB"
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_properties.return_value = fake_props
        with self.assertRaises(SystemExit) as ctx:
            self.kernel.assert_cuda_t4_or_better(_torch_module=fake_torch)
        self.assertEqual(ctx.exception.code, self.kernel.P100_FATAL_RC)
        self.assertEqual(self.kernel.P100_FATAL_RC, 2)

    def test_no_cuda_exits_with_rc_99(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False
        with self.assertRaises(SystemExit) as ctx:
            self.kernel.assert_cuda_t4_or_better(_torch_module=fake_torch)
        self.assertEqual(ctx.exception.code, self.kernel.NO_CUDA_RC)

    def test_t4_sm75_passes(self):
        fake_props = mock.Mock()
        fake_props.name = "Tesla T4"
        fake_props.major = 7
        fake_props.minor = 5
        fake_props.total_memory = 15 * 1024**3
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_properties.return_value = fake_props
        info = self.kernel.assert_cuda_t4_or_better(_torch_module=fake_torch)
        self.assertEqual(info["gpu_name"], "Tesla T4")
        self.assertEqual(info["major"], 7)


class TestFalseAuthorityContract(unittest.TestCase):
    """Kaggle is a sweep/proxy substrate unless exact adjudication is wired."""

    def test_kernel_docstring_carries_proxy_only_contract(self):
        text = (REPO_ROOT / "experiments/kaggle_t1_balle_sweep.py").read_text()

        self.assertIn("score_claim=false", text)
        self.assertIn("promotion_eligible=false", text)
        self.assertIn("ready_for_exact_eval_dispatch=false", text)
        self.assertNotIn("this anchor is\n   ``[contest-CUDA]``", text)

    def test_recipe_notes_carry_proxy_only_contract(self):
        text = (
            REPO_ROOT
            / ".omx/operator_authorize_recipes/kaggle_t1_balle_sweep.yaml"
        ).read_text()

        self.assertIn("score_claim=false", text)
        self.assertIn("promotion_eligible=false", text)
        self.assertIn("ready_for_exact_eval_dispatch=false", text)
        self.assertNotIn("Anchors produced here are tagged [contest-CUDA]", text)


class TestTier1ManifestExtraction(unittest.TestCase):
    """AST-extract the Tier-1 flag manifest from the trainer source.

    Per CLAUDE.md "Deployment version checklist" we do NOT hardcode the
    Tier-1 list; this test pins that the harness reads the trainer manifest.
    """

    def setUp(self):
        self.kernel = _load_sibling_module(
            "experiments/kaggle_t1_balle_sweep.py", "_test_kaggle_kernel_tier1"
        )

    def test_extract_real_trainer_manifest(self):
        trainer = REPO_ROOT / "experiments" / "train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
        self.assertTrue(trainer.is_file(), f"trainer source not found at {trainer}")
        flags = self.kernel.extract_tier1_flags(trainer)
        # The trainer's manifest must include at least these per the 2026-05-12
        # engineering audit; raise if a future trainer renames them silently.
        for expected in (
            "--enable-autocast-fp16",
            "--enable-mp4-codec-sim",
            "--enable-t20-kl-pose-distill",
            "--segmentation-surrogate",
        ):
            self.assertIn(expected, flags, f"manifest missing {expected}")

    def test_raises_on_missing_constant(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "no_tier1.py"
            tmp.write_text("X = 1\n")
            with self.assertRaises(KeyError):
                self.kernel.extract_tier1_flags(tmp)


class TestBuildTrainerArgv(unittest.TestCase):
    """``build_trainer_argv`` must thread every manifest flag — no hardcoding."""

    def setUp(self):
        self.kernel = _load_sibling_module(
            "experiments/kaggle_t1_balle_sweep.py", "_test_kaggle_kernel_argv"
        )

    def test_threads_every_manifest_flag(self):
        manifest = {
            "--enable-autocast-fp16": {"default": None},
            "--enable-mp4-codec-sim": {"default": None},
            "--mp4-codec-sim-noise-std": {"default": "0.0"},
            "--segmentation-surrogate": {"default": "soft_cosine"},
        }
        argv = self.kernel.build_trainer_argv(
            tier1_flags=manifest,
            epochs=1500, batch_size=32,
            output_dir=Path("/kaggle/working/t1"),
            video_path=Path("/kaggle/input/datasets/x/v.mkv"),
            auth_eval=True,
            smoke=False,
        )
        # Boolean flags (default=None) emit just the flag name.
        self.assertIn("--enable-autocast-fp16", argv)
        self.assertIn("--enable-mp4-codec-sim", argv)
        # Value flags emit "--flag VALUE" pairs.
        i = argv.index("--mp4-codec-sim-noise-std")
        self.assertEqual(argv[i + 1], "0.0")
        i = argv.index("--segmentation-surrogate")
        self.assertEqual(argv[i + 1], "soft_cosine")
        # Core flags threaded.
        self.assertIn("--output-dir", argv)
        self.assertIn("--epochs", argv)
        self.assertIn("--auth-eval", argv)
        self.assertNotIn("--smoke", argv)
        self.assertIn("--enable-scorer-domain-loss", argv)
        # --device cuda — CLAUDE.md forbids cpu/mps.
        i = argv.index("--device")
        self.assertEqual(argv[i + 1], "cuda")

    def test_smoke_threads_smoke_flag(self):
        manifest: dict[str, dict] = {}
        argv = self.kernel.build_trainer_argv(
            tier1_flags=manifest,
            epochs=1, batch_size=1,
            output_dir=Path("/tmp/out"),
            video_path=Path("/tmp/v.mkv"),
            auth_eval=False, smoke=True,
        )
        self.assertIn("--smoke", argv)
        self.assertNotIn("--auth-eval", argv)


class TestOperatorWrapperSlugValidation(unittest.TestCase):
    """The wrapper must refuse long or non-alphanumeric slugs at parse time."""

    SCRIPT = REPO_ROOT / "scripts" / "operator_authorize_kaggle_t1_balle_sweep.sh"

    def test_long_variant_rejected(self):
        if not self.SCRIPT.is_file():
            self.skipTest(f"wrapper script not at {self.SCRIPT}")
        # variant 'this-is-definitely-too-long' → slug length > 25 → exit 2.
        proc = subprocess.run(
            ["bash", str(self.SCRIPT),
             "--variant", "this-is-definitely-too-long",
             "--kaggle-dataset-slug", "x/y", "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_special_chars_rejected(self):
        if not self.SCRIPT.is_file():
            self.skipTest("wrapper script not present")
        proc = subprocess.run(
            ["bash", str(self.SCRIPT),
             "--variant", "BAD_CHARS!", "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)

    def test_valid_variant_dry_run_succeeds(self):
        if not self.SCRIPT.is_file():
            self.skipTest("wrapper script not present")
        # Stub the kaggle CLI to a no-op that pretends 0 active kernels.
        with tempfile.TemporaryDirectory() as td:
            stub = Path(td) / "kaggle"
            stub.write_text(
                "#!/bin/bash\n"
                'if [ "$1" = "kernels" ] && [ "$2" = "list" ]; then\n'
                "  echo 'ref,title,status'\n"
                "  exit 0\n"
                "fi\n"
                "exit 0\n"
            )
            stub.chmod(0o755)
            env = os.environ.copy()
            env["KAGGLE_CMD"] = str(stub)
            proc = subprocess.run(
                ["bash", str(self.SCRIPT),
                 "--variant", "a", "--dry-run"],
                capture_output=True, text=True, timeout=30, env=env,
            )
            self.assertEqual(
                proc.returncode, 0,
                msg=f"stdout={proc.stdout}\nstderr={proc.stderr}",
            )


class TestHarvesterTerminalStatusClassification(unittest.TestCase):
    """``classify_terminal_status`` taxonomy."""

    def setUp(self):
        self.harv = _load_sibling_module(
            "tools/harvest_kaggle_kernels.py", "_test_kaggle_harvest"
        )

    def test_complete_rc0_completed_kaggle(self):
        status = self.harv.classify_terminal_status(
            kaggle_status="complete",
            summary={"trainer_returncode": 0},
        )
        self.assertEqual(status, "completed_kaggle")

    def test_complete_rc2_p100_trap(self):
        status = self.harv.classify_terminal_status(
            kaggle_status="complete",
            summary={"trainer_returncode": 2},
        )
        self.assertEqual(status, "failed_kaggle_p100_assignment")

    def test_complete_rc99_no_cuda(self):
        status = self.harv.classify_terminal_status(
            kaggle_status="complete",
            summary={"trainer_returncode": 99},
        )
        self.assertEqual(status, "failed_kaggle_no_cuda")

    def test_running_is_in_flight(self):
        status = self.harv.classify_terminal_status(
            kaggle_status="running", summary=None,
        )
        self.assertTrue(status.startswith("in_flight_kaggle_"))

    def test_error_with_rc2_still_p100(self):
        status = self.harv.classify_terminal_status(
            kaggle_status="error",
            summary={"trainer_returncode": 2},
        )
        self.assertEqual(status, "failed_kaggle_p100_assignment")

    def test_unknown_status_becomes_stale(self):
        status = self.harv.classify_terminal_status(
            kaggle_status=None, summary=None,
        )
        self.assertEqual(status, "stale_superseded_kaggle")


class TestHarvesterCostBandAnchor(unittest.TestCase):
    """``append_cost_band_anchor_from_summary`` must invoke the anchor tool."""

    def setUp(self):
        self.harv = _load_sibling_module(
            "tools/harvest_kaggle_kernels.py", "_test_kaggle_harvest_anchor"
        )

    def test_invokes_real_anchor_tool(self):
        anchor_tool = REPO_ROOT / "tools" / "append_cost_band_anchor.py"
        if not anchor_tool.is_file():
            self.skipTest("append_cost_band_anchor.py missing")
        summary = {
            "trainer": "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
            "gpu_info": {"gpu_name": "Tesla T4"},
            "epochs": 1500, "batch_size": 32,
            "wall_clock_sec": 7200.0,
            "trainer_returncode": 0,
        }
        # We use mock.patch on subprocess.run so the real CLI is NOT invoked
        # against the real posterior file under .omx/state — we just verify
        # the harvester emits the correct command shape.
        with mock.patch.object(
            self.harv.subprocess, "run",
            return_value=mock.Mock(returncode=0, stdout="ok", stderr=""),
        ) as mock_run:
            result = self.harv.append_cost_band_anchor_from_summary(
                summary=summary,
                dispatch_label="kaggle_t1_balle_a_20260512T000000Z",
                anchor_tool=anchor_tool,
            )
            self.assertTrue(result["appended"])
            emitted_argv = mock_run.call_args.args[0]
            self.assertIn("--platform", emitted_argv)
            self.assertEqual(
                emitted_argv[emitted_argv.index("--platform") + 1],
                "kaggle",
            )
            self.assertEqual(
                emitted_argv[emitted_argv.index("--gpu") + 1],
                "Tesla_T4",
            )
            # Free tier: cost must be 0.00.
            self.assertEqual(
                emitted_argv[emitted_argv.index("--actual-cost-usd") + 1],
                "0.00",
            )

    def test_missing_anchor_tool_returns_skip(self):
        result = self.harv.append_cost_band_anchor_from_summary(
            summary={},
            dispatch_label="x",
            anchor_tool=Path("/nonexistent/tool.py"),
        )
        self.assertFalse(result["appended"])
        self.assertEqual(result["reason"], "anchor_tool_missing")


if __name__ == "__main__":
    unittest.main()
